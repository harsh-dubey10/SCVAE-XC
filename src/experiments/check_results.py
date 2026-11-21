"""Sanity-check every strict cross-culture result against the processed data.

For each ``subject_XX.json`` under the given roots it checks that:
  * the confusion matrix sums to target_samples, its rows equal true_counts and
    its columns equal prediction_counts;
  * accuracy and macro-F1 recompute exactly from the confusion matrix;
  * true_counts equal the labels of that test subject in the processed data;
  * target/source sample counts match the split (pooled: all subjects minus
    the target; cross: every subject of the source dataset);
  * the subject_norm field matches the run name;
  * every run holds each test subject x seed exactly once, and deterministic
    SVM records are identical across seeds.
Folds below chance (< 0.30) or collapsed (one class >= 90% of predictions)
are listed as warnings, not errors.

    python -m src.experiments.check_results results/crossculture results/scvae_fix
"""

import argparse
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


DATA = Path("data/processed")


@lru_cache(maxsize=None)
def subject_labels(dataset, subject):
    """Label vector of one subject over all of its sessions (session order)."""
    labels = []
    for session_dir in sorted((DATA / dataset).iterdir(), key=lambda p: p.name):
        path = session_dir / f"subject{subject}.pth"
        if session_dir.is_dir() and path.exists():
            labels.extend(torch.load(path, weights_only=True)["labels"])
    return np.array(labels)


@lru_cache(maxsize=None)
def dataset_subjects(dataset):
    return sorted({int(p.stem.removeprefix("subject")) for p in (DATA / dataset).glob("*/subject*.pth")})


def macro_f1(cm):
    f1 = []
    for c in range(len(cm)):
        tp = cm[c, c]
        precision = tp / cm[:, c].sum() if cm[:, c].sum() else 0.0
        recall = tp / cm[c].sum() if cm[c].sum() else 0.0
        f1.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1))


def check_record(record, run_name, errors, warnings, where):
    cm = np.array(record["confusion_matrix"])
    n = cm.sum()
    if n != record["target_samples"]:
        errors.append(f"{where}: confusion matrix sums to {n}, target_samples {record['target_samples']}")
    if cm.sum(axis=1).tolist() != record["true_counts"]:
        errors.append(f"{where}: CM rows {cm.sum(axis=1).tolist()} != true_counts {record['true_counts']}")
    if cm.sum(axis=0).tolist() != record["prediction_counts"]:
        errors.append(f"{where}: CM columns != prediction_counts")
    if abs(np.trace(cm) / n - record["accuracy"]) > 1e-9:
        errors.append(f"{where}: accuracy {record['accuracy']} != CM {np.trace(cm) / n}")
    if abs(macro_f1(cm) - record["macro_f1"]) > 1e-6:
        errors.append(f"{where}: macro_f1 {record['macro_f1']} != CM {macro_f1(cm)}")
    if any(isinstance(v, float) and np.isnan(v) for v in (record["accuracy"], record["macro_f1"])):
        errors.append(f"{where}: NaN metric")

    target = Path(record["data_dir"]).name
    source = Path(record.get("source_data_dir") or record["data_dir"]).name
    subject = record["test_subject"]
    labels = subject_labels(target, subject)
    if np.bincount(labels, minlength=3).tolist() != record["true_counts"]:
        errors.append(f"{where}: true_counts {record['true_counts']} != data {np.bincount(labels, minlength=3).tolist()}")
    if len(labels) != record["target_samples"]:
        errors.append(f"{where}: target_samples {record['target_samples']} != data {len(labels)}")

    if record.get("split") == "pooled":
        expected = sum(len(subject_labels(target, s)) for s in dataset_subjects(target) if s != subject)
    elif record.get("split") == "cross":
        expected = sum(len(subject_labels(source, s)) for s in dataset_subjects(source))
    else:
        expected = None
    if expected is not None and record["source_samples"] != expected:
        errors.append(f"{where}: source_samples {record['source_samples']} != expected {expected}")

    if ("subjnorm" in run_name) != (record.get("subject_norm") == "recording"):
        errors.append(f"{where}: subject_norm {record.get('subject_norm')} does not match run name")

    counts = record["prediction_counts"]
    if max(counts) >= 0.9 * sum(counts):
        warnings.append(f"{where}: collapsed predictions {counts} (acc {record['accuracy']:.3f})")
    elif record["accuracy"] < 0.30:
        warnings.append(f"{where}: below chance, acc {record['accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    errors, warnings, n_records, n_runs = [], [], 0, 0
    for root in args.roots:
        run_dirs = sorted({p.parents[2] for p in root.rglob("session_all/seed_*/subject_*.json")})
        for run_dir in run_dirs:
            n_runs += 1
            by_seed = defaultdict(dict)
            for path in sorted(run_dir.glob("session_all/seed_*/subject_*.json")):
                record = json.loads(path.read_text())
                seed = int(re.search(r"seed_(\d+)", str(path)).group(1))
                by_seed[seed][record["test_subject"]] = record
                n_records += 1
                check_record(record, run_dir.name, errors, warnings, str(path.relative_to(root)))
            target = Path(next(iter(next(iter(by_seed.values())).values()))["data_dir"]).name
            wanted = set(dataset_subjects(target))
            for seed, records in by_seed.items():
                if set(records) != wanted:
                    errors.append(f"{run_dir}: seed {seed} has subjects {sorted(records)}, expected {sorted(wanted)}")
            first = next(iter(by_seed.values()))
            if any(r.get("deterministic_across_seeds") for r in first.values()):
                for seed, records in by_seed.items():
                    for s, r in records.items():
                        if r["confusion_matrix"] != first[s]["confusion_matrix"]:
                            errors.append(f"{run_dir}: deterministic record differs for seed {seed} subject {s}")

    print(f"Checked {n_records} records in {n_runs} runs.")
    print(f"Errors: {len(errors)}")
    for e in errors[:50]:
        print("  ERROR", e)
    print(f"Warnings: {len(warnings)}")
    for w in warnings:
        print("  warn ", w)


if __name__ == "__main__":
    main()
