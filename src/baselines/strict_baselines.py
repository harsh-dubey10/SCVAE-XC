"""SVM and MLP baselines on the strict protocol.

Same folds as ``train_scvae_reference``. With ``--split session`` (default),
for each session one held-out subject is the target and the other subjects of
that session are the source. ``--split pooled`` leaves one subject out over all
sessions, and ``--split cross`` trains on every subject of ``--source-data-dir``
and tests each subject of ``--data-dir`` (all sessions). Features are the
flattened 62 x 5 DE-LDS windows. The scaler and models see source data only;
target labels are read once, for the final evaluation. Output follows the
``<run>/session_XX/seed_XX/subject_XX.json`` layout, so ``compare_runs``
can pair these folds with SCVAE runs.

The SVMs are deterministic, so each is fitted once per fold and its record is
written under every seed with ``"deterministic_across_seeds": true``.
``linsvm`` is the linear SVM of Liu et al. (2022); its C is chosen by 3-fold
subject-grouped cross-validation on the source only, never on the target.

    python -m src.baselines.strict_baselines --model svm
    python -m src.baselines.strict_baselines --model mlp
    python -m src.baselines.strict_baselines --model svm --split pooled \
        --data-dir data/processed/SEED_GER --output-root results/crossculture/SEED_GER
    python -m src.baselines.strict_baselines --model svm --split cross \
        --source-data-dir data/processed/SEED_4s_trials \
        --data-dir data/processed/SEED_GER --output-root results/crossculture/SEED_GER
"""

import argparse
import os
import json
from pathlib import Path
from types import SimpleNamespace

import warnings

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.mlp import MLP
from src.experiments.train_scvae_reference import (
    DEFAULT_DATA_DIR,
    NUM_CLASSES,
    all_subjects,
    available_subjects,
    load_dataset,
    load_split,
    load_subject_all_sessions,
    load_subject_session,
    set_seed,
)

# Under SLURM use only the allocated cores (os.cpu_count() sees the whole node).
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
SEEDS = [12, 34, 56]
LINSVM_C_GRID = [2.0 ** c for c in range(-14, 3, 2)]  # the 4 s runs chose 2^-8, the old lower edge


def metrics(actual, predictions):
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=range(NUM_CLASSES)).tolist(),
        "prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES).tolist(),
        "true_counts": np.bincount(actual, minlength=NUM_CLASSES).tolist(),
    }


def make_svm():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))


def make_linsvm(c):
    return make_pipeline(StandardScaler(), LinearSVC(C=c, dual=False, max_iter=2000))


def fit_linsvm_cell(c, x, y, train, valid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return make_linsvm(c).fit(x[train], y[train]).score(x[valid], y[valid])


def fit_linsvm(source_x, source_y, groups):
    """Pick C by subject-grouped 3-fold CV on the source, then refit on all of it."""
    splits = list(GroupKFold(n_splits=3).split(source_x, source_y, groups))
    scores = Parallel(n_jobs=N_JOBS)(
        delayed(fit_linsvm_cell)(c, source_x, source_y, train, valid)
        for c in LINSVM_C_GRID for train, valid in splits
    )
    mean_scores = np.array(scores).reshape(len(LINSVM_C_GRID), len(splits)).mean(axis=1)
    best_c = LINSVM_C_GRID[int(np.argmax(mean_scores))]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        return make_linsvm(best_c).fit(source_x, source_y), best_c


def source_groups(split_args, test_subject):
    """Subject ID of every source window, in the order load_split concatenates them."""
    if split_args.split == "session":
        subjects = [s for s in available_subjects(split_args.session, split_args.data_dir) if s != test_subject]
        sizes = [len(load_subject_session(s, split_args.session, split_args.data_dir)[1]) for s in subjects]
    else:
        root = split_args.source_data_dir if split_args.split == "cross" else split_args.data_dir
        subjects = [s for s in all_subjects(root) if split_args.split == "cross" or s != test_subject]
        sizes = [len(load_subject_all_sessions(s, root)[1]) for s in subjects]
    return np.repeat(subjects, sizes)


def fit_mlp(source_x, source_y, epochs, device, seed):
    """Train on source data only; returns a function that predicts target windows."""
    set_seed(seed)
    scaler = StandardScaler().fit(source_x)
    xs = torch.tensor(scaler.transform(source_x), dtype=torch.float32)
    loader = DataLoader(TensorDataset(xs, torch.tensor(source_y)), batch_size=256, shuffle=True)
    model = MLP(input_dim=xs.shape[1], num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x.to(device)), batch_y.to(device)).backward()
            optimizer.step()
    model.eval()

    def predict_proba(target_x):
        xt = torch.tensor(scaler.transform(target_x), dtype=torch.float32)
        with torch.no_grad():
            return torch.softmax(model(xt.to(device)), dim=1).cpu().numpy()

    def predict(target_x):
        return predict_proba(target_x).argmax(axis=1)

    predict.proba = predict_proba
    return predict


def fit_mlp_selftrain(source_x, source_y, target_x, epochs, device, seed, rounds=3, threshold=0.9):
    """Normalise-then-adapt: class-balanced self-training on the unlabelled target.

    Each round keeps, per emotion, the same number k of the most confident
    target windows (softmax >= threshold; k = the smallest class count), adds
    them to the source with their pseudo-labels and retrains from scratch.
    Target labels are never read. Returns the final predictions and, per
    round, how many target windows were added.
    """
    predict = fit_mlp(source_x, source_y, epochs, device, seed)
    added = []
    for _ in range(rounds):
        proba = predict.proba(target_x)
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= threshold))[0] for c in range(NUM_CLASSES)]
        k = min(len(pool) for pool in pools)
        if k < 10:  # too few confident windows in some class: stop adapting
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        predict = fit_mlp(np.concatenate([source_x, target_x[chosen]]),
                          np.concatenate([source_y, pseudo[chosen]]), epochs, device, seed)
    return predict(target_x), added


def write_record(record, output_dir, fold_dir, seed, subject):
    run_dir = output_dir / fold_dir / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"subject_{subject:02d}.json").write_text(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=["svm", "linsvm", "mlp", "mlp_st"], required=True)
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--epochs", type=int, default=30, help="MLP only; matches the legacy baseline.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=["session", "pooled", "cross"], default="session")
    parser.add_argument("--source-data-dir", type=Path, help="Source culture for --split cross.")
    parser.add_argument("--subject-norm", choices=["none", "recording"], default="none",
                        help="Standardise each recording with its own statistics (label-free, transductive).")
    parser.add_argument("--output-root", type=Path, default=Path("results/strict_baselines"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if (args.split == "cross") != (args.source_data_dir is not None):
        parser.error("--source-data-dir is required for, and only valid with, --split cross")

    suffix = f"__from_{args.source_data_dir.name}" if args.split == "cross" else {"session": "", "pooled": "__pooled"}[args.split]
    if args.subject_norm != "none":
        suffix = f"__subjnorm{suffix}"
    output_dir = args.output_root / f"{args.model}{suffix}"

    if args.split == "session":
        folds = [(session, subject) for session in args.sessions for subject in available_subjects(session, args.data_dir)]
    else:
        folds = [("all", subject) for subject in all_subjects(args.data_dir)]

    cross_source = None
    if args.split == "cross":
        # Every target subject shares one source set, so each model is fitted once.
        source_x, source_y = load_dataset(args.source_data_dir, subject_norm=args.subject_norm)
        cross_source = (source_x.flatten(start_dim=1).numpy(), source_y.numpy())
        cross_predictors = {}

    for session, subject in folds:
        split_args = SimpleNamespace(
            split=args.split, session=None if session == "all" else session,
            data_dir=args.data_dir, source_data_dir=args.source_data_dir, subject_norm=args.subject_norm,
        )
        if cross_source is None:
            source_x, source_y, target_x, target_y = load_split(subject, split_args)
            source_x, source_y = source_x.flatten(start_dim=1).numpy(), source_y.numpy()
        else:
            _, _, target_x, target_y = load_split(subject, split_args)
            source_x, source_y = cross_source
        target_x, target_y = target_x.flatten(start_dim=1).numpy(), target_y.numpy()
        fold_dir = "session_all" if session == "all" else f"session_{session:02d}"
        base = {
            "protocol": "strict",
            "split": args.split,
            "subject_norm": args.subject_norm,
            "model": args.model,
            "session": session,
            "test_subject": subject,
            "data_dir": str(args.data_dir),
            "source_data_dir": str(args.source_data_dir or args.data_dir),
            "source_samples": len(source_x),
            "target_samples": len(target_x),
        }
        if args.model in ("svm", "linsvm"):
            if cross_source is None or args.model not in cross_predictors:
                if args.model == "svm":
                    fitted, extra = make_svm().fit(source_x, source_y), {}
                else:
                    groups = source_groups(split_args, subject)
                    assert len(groups) == len(source_x), "source groups out of line with source windows"
                    fitted, best_c = fit_linsvm(source_x, source_y, groups)
                    extra = {"c": best_c, "c_selection": "3-fold subject-grouped CV on source"}
                if cross_source is not None:
                    cross_predictors[args.model] = (fitted, extra)
            else:
                fitted, extra = cross_predictors[args.model]
            result = metrics(target_y, fitted.predict(target_x))
            for seed in args.seeds:
                write_record({**base, "seed": seed, "deterministic_across_seeds": True, **extra, **result},
                             output_dir, fold_dir, seed, subject)
            print(f"{args.model} {fold_dir} subject {subject:2d}: {result['accuracy']:.3f} {extra}", flush=True)
        elif args.model == "mlp_st":
            # Self-training adapts to each target subject, so nothing is shared across folds.
            for seed in args.seeds:
                predictions, added = fit_mlp_selftrain(source_x, source_y, target_x, args.epochs, args.device, seed)
                result = metrics(target_y, predictions)
                write_record({**base, "seed": seed, "epochs": args.epochs, "self_training_rounds": 3,
                              "pseudo_threshold": 0.9, "pseudo_labels_added": added, **result},
                             output_dir, fold_dir, seed, subject)
                print(f"mlp_st {fold_dir} seed {seed} subject {subject:2d}: {result['accuracy']:.3f} added {added}", flush=True)
        else:
            for seed in args.seeds:
                if cross_source is not None:
                    if seed not in cross_predictors:
                        cross_predictors[seed] = fit_mlp(source_x, source_y, args.epochs, args.device, seed)
                    predict = cross_predictors[seed]
                else:
                    predict = fit_mlp(source_x, source_y, args.epochs, args.device, seed)
                result = metrics(target_y, predict(target_x))
                write_record({**base, "seed": seed, "epochs": args.epochs, **result},
                             output_dir, fold_dir, seed, subject)
                print(f"mlp {fold_dir} seed {seed} subject {subject:2d}: {result['accuracy']:.3f}", flush=True)

if __name__ == "__main__":
    main()
