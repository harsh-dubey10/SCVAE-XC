"""Summarise strict cross-culture runs next to Liu et al. (J. Neural Eng. 2022).

Reads ``results/crossculture/<target>/<run>/session_all/seed_*/subject_*.json``
where ``<run>`` ends in ``__pooled`` (within-culture leave-one-subject-out) or
``__from_<source>`` (cross-culture). Accuracy is averaged over seeds per test
subject, then reported as mean and std over test subjects, as in the paper.

The paper's numbers come from its tables 6 and 8 (EEG, DNN and SVM), which
used 1 s windows, so the ``window == "1s"`` rows are the matched comparison. Its
released code picks the epoch, learning rate and SVM C by test accuracy, so
they are optimistic, not strict held-out results.

    python -m src.experiments.summarize_crossculture
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CULTURE = {"SEED_CHN": "Chinese (SEED, 15)", "SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}
# The paper used the 12-subject Chinese multimodal subset; SEED_CHN rows are a 15-subject check.
PAPER_NAME = {"SEED_CHN_MM": "SEED_CHN"}
# (source, target) -> accuracy %, EEG only. Source == target means within-culture LOSO.
PAPER = {
    "dnn": {
        ("SEED_CHN", "SEED_CHN"): 82.81, ("SEED_GER", "SEED_GER"): 65.87, ("SEED_FRA", "SEED_FRA"): 64.18,
        ("SEED_CHN", "SEED_FRA"): 57.28, ("SEED_CHN", "SEED_GER"): 55.32,
        ("SEED_GER", "SEED_CHN"): 64.34, ("SEED_GER", "SEED_FRA"): 58.93,
        ("SEED_FRA", "SEED_CHN"): 66.19, ("SEED_FRA", "SEED_GER"): 60.10,
    },
    "svm": {
        ("SEED_CHN", "SEED_CHN"): 72.63, ("SEED_GER", "SEED_GER"): 55.64, ("SEED_FRA", "SEED_FRA"): 50.10,
        ("SEED_CHN", "SEED_FRA"): 46.81, ("SEED_CHN", "SEED_GER"): 47.88,
        ("SEED_GER", "SEED_CHN"): 47.39, ("SEED_GER", "SEED_FRA"): 47.69,
        ("SEED_FRA", "SEED_CHN"): 52.30, ("SEED_FRA", "SEED_GER"): 48.88,
    },
}


SUFFIXES = {"_EEG_EYE": "eeg+eye", "_EYE": "eye"}


def split_name(name):
    """SEED_GER_EYE_1s -> ("SEED_GER", "eye", "1s")."""
    window = "1s" if name.endswith("_1s") else "4s"
    name = name.removesuffix("_1s")
    for suffix, modality in SUFFIXES.items():
        if name.endswith(suffix):
            return name.removesuffix(suffix), modality, window
    return name, "eeg", window


def load_run(run_dir):
    per_subject = defaultdict(list)
    for path in run_dir.glob("session_all/seed_*/subject_*.json"):
        record = json.loads(path.read_text())
        if record.get("deterministic_across_seeds") and per_subject[record["test_subject"]]:
            continue  # the SVM record is copied under every seed; count it once
        per_subject[record["test_subject"]].append((record["accuracy"], record["macro_f1"], record["prediction_counts"]))
    return per_subject


def summarise(per_subject):
    accuracy = np.array([np.mean([a for a, _, _ in folds]) for folds in per_subject.values()])
    macro_f1 = np.array([np.mean([f for _, f, _ in folds]) for folds in per_subject.values()])
    # A fold is "collapsed" when one class takes >= 90% of predictions.
    collapsed = sum(max(c) >= 0.9 * sum(c) for folds in per_subject.values() for _, _, c in folds)
    folds = sum(len(f) for f in per_subject.values())
    return accuracy, macro_f1, collapsed, folds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/crossculture"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/crossculture_summary.csv"))
    args = parser.parse_args()

    rows = []
    for target_dir in sorted(p for p in args.root.iterdir() if p.is_dir() and split_name(p.name)[0] in CULTURE):
        target, modality, window = split_name(target_dir.name)
        for run_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            match = re.fullmatch(r"(.+?)__(pooled|from_(\w+))", run_dir.name)
            if not match:
                continue
            method = match.group(1)
            source = target if match.group(2) == "pooled" else split_name(match.group(3))[0]
            per_subject = load_run(run_dir)
            if not per_subject:
                continue
            accuracy, macro_f1, collapsed, folds = summarise(per_subject)
            paper_key = "svm" if "svm" in method else "dnn"
            rows.append({
                "setting": "within" if source == target else "cross",
                "window": window,
                "modality": modality,
                "train": CULTURE[source],
                "test": CULTURE[target],
                "method": method,
                "subjects": len(per_subject),
                "folds": folds,
                "accuracy_mean": round(100 * accuracy.mean(), 2),
                "accuracy_std": round(100 * accuracy.std(), 2),
                "macro_f1_mean": round(100 * macro_f1.mean(), 2),
                "collapsed_folds": collapsed,
                f"paper_eeg_{paper_key}": PAPER[paper_key].get(
                    (PAPER_NAME.get(source, source), PAPER_NAME.get(target, target))),
            })

    rows.sort(key=lambda r: (r["setting"] != "within", r["train"], r["test"], r["modality"], r["window"], r["method"]))
    fields = ["setting", "modality", "window", "train", "test", "method", "subjects", "folds", "accuracy_mean", "accuracy_std",
              "macro_f1_mean", "collapsed_folds", "paper_eeg_svm", "paper_eeg_dnn"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)

    print("| Setting | Modality | Win | Train → Test | Method | Acc. % (± std) | Macro-F1 % | Collapsed | Paper SVM | Paper DNN |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['setting']} | {r['modality']} | {r['window']} | {r['train']} → {r['test']} | {r['method']} | "
              f"{r['accuracy_mean']:.1f} ± {r['accuracy_std']:.1f} | {r['macro_f1_mean']:.1f} | "
              f"{r['collapsed_folds']}/{r['folds']} | {r.get('paper_eeg_svm') or '—'} | {r.get('paper_eeg_dnn') or '—'} |")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
