"""Paired comparison of strict SCVAE-Net runs.

Each run is a directory laid out as ``session_XX/seed_YY/subject_ZZ.json``.
Folds are paired by (session, seed, test subject); only folds present in every
run are compared. The first run is the baseline for the Wilcoxon tests.

    python -m src.experiments.compare_runs \
        results/reference_scvae/strict__bn_source \
        results/reference_scvae/strict__entropy_weighted_mmd__bn_source \
        --output results/tables/bn_uncertainty_comparison.csv
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.diagnose_collapse import prediction_profile


def load_run(run_dir):
    folds = {}
    for path in sorted(Path(run_dir).glob("session_*/seed_*/subject_*.json")):
        record = json.loads(path.read_text())
        key = (record["session"], record["seed"], record["test_subject"])
        folds[key] = {
            "accuracy": record["accuracy"],
            "macro_f1": record["macro_f1"],
            "status": prediction_profile(record["prediction_counts"])["collapse_status"],
        }
    return folds


def describe(name, folds, keys, baseline=None):
    accuracy = np.array([folds[k]["accuracy"] for k in keys])
    macro_f1 = np.array([folds[k]["macro_f1"] for k in keys])
    statuses = [folds[k]["status"] for k in keys]
    # Across-seed spread of the per-seed means, the usual reporting unit.
    seed_means = [
        np.mean([folds[k]["accuracy"] for k in keys if k[1] == seed])
        for seed in sorted({k[1] for k in keys})
    ]
    row = {
        "run": name,
        "folds": len(keys),
        "accuracy_mean": accuracy.mean(),
        "accuracy_fold_std": accuracy.std(),
        "accuracy_seed_std": float(np.std(seed_means)),
        "macro_f1_mean": macro_f1.mean(),
        "collapsed": statuses.count("collapsed"),
        "partial": statuses.count("partial"),
        "failed_f1_below_0.35": int((macro_f1 < 0.35).sum()),
    }
    for session in sorted({k[0] for k in keys}):
        row[f"accuracy_session_{session}"] = np.mean([folds[k]["accuracy"] for k in keys if k[0] == session])
    if baseline is not None:
        base = np.array([baseline[k]["accuracy"] for k in keys])
        difference = accuracy - base
        row["delta_vs_baseline"] = difference.mean()
        row["wins_ties_losses"] = f"{(difference > 0).sum()}/{(difference == 0).sum()}/{(difference < 0).sum()}"
        row["wilcoxon_p"] = float(wilcoxon(accuracy, base).pvalue) if np.any(difference) else 1.0
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories; the first is the baseline.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    runs = {str(run): load_run(run) for run in args.runs}
    for name, folds in runs.items():
        if not folds:
            parser.error(f"no fold JSONs under {name}")
    keys = sorted(set.intersection(*(set(folds) for folds in runs.values())))
    names = list(runs)
    rows = [describe(names[0], runs[names[0]], keys)]
    rows += [describe(name, runs[name], keys, runs[names[0]]) for name in names[1:]]

    for row in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    if args.output is not None:
        fieldnames = []
        for row in rows:
            fieldnames.extend(key for key in row if key not in fieldnames)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
