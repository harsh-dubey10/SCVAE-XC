"""How much does test-set hyperparameter selection add to Liu et al. (2022)?

Reproduces the paper's linear-SVM pipeline (raw DE-LDS features flattened to
310 values, ``LinearSVC``, each set standardised with its own statistics as in
the released code) in its two subject-independent settings. ``--window 1s
--grid full`` matches the paper most closely: its unimodal EEG models used
1 s windows, and its released code searches both C grids.

  pooled  leave one subject out within a culture, all sessions (ICSI)
  cross   train on one culture, test each subject of another (CCSI)

For every fold, the accuracy at each C of the paper's grid (2^-10 ... 2^9) is
recorded, so the same fits give:

  oracle      best C on the test subject, as the released code does (leaky)
  fixed_c1    C = 1, fixed in advance (strict)
  source_cv   C picked by 3-fold subject-grouped CV on the source only (strict)

Standardising the target with its own statistics uses unlabelled target data
only, which the strict protocol allows, so oracle minus strict isolates the
effect of test-label selection.

    python -m src.baselines.liu_leakage_audit
"""

import argparse
import os
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.io import loadmat
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import scale
from sklearn.svm import LinearSVC


# The paper's released svm_classification searches both grids and keeps the
# best test accuracy: 2^-10 ... 2^9, then 0.1 ... 19.6 in steps of 0.5.
COARSE_GRID = [2.0 ** c for c in range(-10, 10)]
FINE_GRID = [float(c) for c in np.arange(0.1, 20, 0.5)]
# Under SLURM use only the allocated cores (os.cpu_count() sees the whole node).
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
C_GRID = COARSE_GRID  # set from --grid in main()
SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
SOURCES = {  # 4 s paths; --window 1s swaps 4s for 1s in each path
    "SEED_CHN": Path("data/raw/SEED/ExtractedFeatures_4s"),
    "SEED_GER": Path("data/raw/SEED_GER/02-EEG-DE-features/eeg_used_4s"),
    "SEED_FRA": Path("data/raw/SEED_FRA/02-EEG-DE-features/eeg_used_4s"),
    # Chinese multimodal subset (12 subjects), same extraction as GER/FRA.
    "SEED_CHN_MM": Path("data/raw/SEED_CHN_MM/02-EEG-DE-feature/eeg_used_4s"),
}


def load_chinese(directory):
    """Official SEED 4 s de_LDS, channel-major like the released code."""
    subjects = {}
    for subject in range(1, 16):
        xs, ys = [], []
        for path in sorted(directory.glob(f"{subject}_*.mat")):
            mat = loadmat(path)
            for trial, label in enumerate(SEED_LABELS, start=1):
                features = np.swapaxes(mat[f"de_LDS{trial}"], 0, 1)  # T x 62 x 5
                xs.append(features.reshape(len(features), -1))
                ys.append(np.full(len(features), label + 1))
        subjects[subject] = (np.concatenate(xs), np.concatenate(ys))
    return subjects


def load_bcmi(directory):
    subjects = {}
    for path in sorted(directory.glob("*_*.npz")):
        subject = int(path.stem.split("_")[0])
        npz = np.load(path)
        parts = [pickle.loads(npz["train_data"]), pickle.loads(npz["test_data"])]
        x = np.concatenate([np.stack([part[band] for band in BANDS], axis=-1) for part in parts])
        y = np.concatenate([npz["train_label"], npz["test_label"]]).astype(int)
        old = subjects.get(subject, (np.empty((0, 310)), np.empty(0, dtype=int)))
        subjects[subject] = (np.concatenate([old[0], x.reshape(len(x), -1)]), np.concatenate([old[1], y]))
    return subjects


def load(culture, source_dir=None, window="4s"):
    directory = source_dir or Path(str(SOURCES[culture]).replace("4s", window))
    return load_chinese(directory) if culture == "SEED_CHN" else load_bcmi(directory)


def fit_one(c, train_x, train_y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return LinearSVC(C=c, dual=False, max_iter=2000).fit(train_x, train_y)


def fit_grid(train_x, train_y):
    """One LinearSVC per C, in parallel; the primal solver suits samples >> features."""
    return Parallel(n_jobs=N_JOBS)(delayed(fit_one)(c, train_x, train_y) for c in C_GRID)


def source_cv_choice(subjects):
    """Pick C by subject-grouped 3-fold CV on source data only."""
    x = np.concatenate([subjects[s][0] for s in subjects])
    y = np.concatenate([subjects[s][1] for s in subjects])
    groups = np.concatenate([np.full(len(subjects[s][1]), s) for s in subjects])
    scores = np.zeros(len(C_GRID))
    for train, valid in GroupKFold(n_splits=3).split(x, y, groups):
        models = fit_grid(scale(x[train]), y[train])
        valid_x = scale(x[valid])
        scores += [model.score(valid_x, y[valid]) for model in models]
    return int(np.argmax(scores))


def fold_record(models, target_x, target_y, cv_index):
    accuracies = [float(model.score(scale(target_x), target_y)) for model in models]
    return {
        "accuracy_per_c": accuracies,
        "oracle": max(accuracies),
        "oracle_c": C_GRID[int(np.argmax(accuracies))],
        "fixed_c1": accuracies[C_GRID.index(1.0)],
        "source_cv": accuracies[cv_index],
        "source_cv_c": C_GRID[cv_index],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cultures", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    parser.add_argument("--seed-dir", type=Path, help="Override the official SEED ExtractedFeatures_4s directory.")
    parser.add_argument("--window", choices=["4s", "1s"], default="4s",
                        help="1s matches the paper's unimodal EEG experiments.")
    parser.add_argument("--grid", choices=["coarse", "full"], default="coarse",
                        help="full = the paper's two C grids (60 values); coarse = 2^-10..2^9 only.")
    parser.add_argument("--output", type=Path, help="Default: results/crossculture/liu_leakage_audit[_<window>_<grid>].json")
    args = parser.parse_args()
    global C_GRID
    C_GRID = COARSE_GRID + FINE_GRID if args.grid == "full" else COARSE_GRID
    if args.output is None:
        name = "liu_leakage_audit" if (args.window, args.grid) == ("4s", "coarse") else f"liu_leakage_audit_{args.window}_{args.grid}"
        args.output = Path(f"results/crossculture/{name}.json")
    print(f"window {args.window}, {len(C_GRID)} C values -> {args.output}")

    data = {c: load(c, args.seed_dir if c == "SEED_CHN" else None, args.window) for c in args.cultures}
    results = json.loads(args.output.read_text()) if args.output.exists() else {}

    for source in args.cultures:
        for target in args.cultures:
            key = f"{source}->{target}"
            if key in results:
                print(f"skip {key}")
                continue
            folds = {}
            if source == target:  # pooled leave-one-subject-out
                for subject, (target_x, target_y) in data[target].items():
                    train = {s: v for s, v in data[source].items() if s != subject}
                    models = fit_grid(scale(np.concatenate([v[0] for v in train.values()])),
                                      np.concatenate([v[1] for v in train.values()]))
                    folds[subject] = fold_record(models, target_x, target_y, source_cv_choice(train))
                    print(f"{key} subject {subject}: oracle {folds[subject]['oracle']:.3f} "
                          f"cv {folds[subject]['source_cv']:.3f} c1 {folds[subject]['fixed_c1']:.3f}", flush=True)
            else:  # cross-culture: one model per C, shared by every target subject
                train = data[source]
                models = fit_grid(scale(np.concatenate([v[0] for v in train.values()])),
                                  np.concatenate([v[1] for v in train.values()]))
                cv_index = source_cv_choice(train)
                for subject, (target_x, target_y) in data[target].items():
                    folds[subject] = fold_record(models, target_x, target_y, cv_index)
                print(f"{key}: oracle {np.mean([f['oracle'] for f in folds.values()]):.3f} "
                      f"cv {np.mean([f['source_cv'] for f in folds.values()]):.3f}", flush=True)
            results[key] = folds
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=1))

    print("\n| Train -> Test | Oracle (paper-style, leaky) | Source-CV C (strict) | C = 1 (strict) |")
    print("|---|---|---|---|")
    for key, folds in results.items():
        values = {m: 100 * np.mean([f[m] for f in folds.values()]) for m in ["oracle", "source_cv", "fixed_c1"]}
        print(f"| {key} | {values['oracle']:.1f} | {values['source_cv']:.1f} | {values['fixed_c1']:.1f} |")


if __name__ == "__main__":
    main()
