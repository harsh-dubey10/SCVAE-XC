"""How much of cross-subject emotion accuracy is film recognition?

Within each culture, leave one subject out and predict emotion, twice, on the
same test windows and with the same amount of training data:

  seen     the training people's windows include the test films
  unseen   the test films are left out of training (other films only)

The films (clips, the same in every session) are dealt into 3 folds, each
holding every emotion. For a test subject and a test fold, both variants train
on two folds of the other subjects: "seen" uses the test fold plus the next
fold, "unseen" uses the two other folds. Seen minus unseen is the accuracy
that depends on having trained on the very films being tested, i.e. on
recognising the film rather than the emotion.

Each recording is standardised with its own statistics (label-free), and the
model is fixed in advance (standardiser + logistic regression, C = 0.1,
balanced class weights). Paired over test subjects.

    python -m src.experiments.clip_heldout_emotion   # -> results/tables/clip_heldout_emotion.md
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score

from src.experiments.film_identification import MODALITIES, fit
from src.experiments.train_scvae_reference import available_subjects, normalize_recording, sessions_in

CULTURES = {"Chinese": "SEED_CHN_MM_EEG_EYE", "German": "SEED_GER_EEG_EYE", "French": "SEED_FRA_EEG_EYE"}
N_FOLDS = 3


def load(root):
    xs, ys, clips, subjects = [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            xs.append(normalize_recording(torch.stack(data["data"]).float().flatten(start_dim=1)).numpy())
            ys.append(np.asarray(data["labels"]))
            clips.append(np.asarray(data["trials"]))
            subjects.append(np.full(len(xs[-1]), subject))
    return [np.concatenate(v) for v in (xs, ys, clips, subjects)]


def clip_folds(y, clips):
    """Deal each emotion's clips round-robin into N_FOLDS folds: clip -> fold."""
    fold = {}
    for emotion in np.unique(y):
        for i, clip in enumerate(sorted(np.unique(clips[y == emotion]))):
            fold[clip] = i % N_FOLDS
    return np.array([fold[c] for c in clips])


def run_fold(x, y, folds, subjects, subject, test_fold, columns):
    test = (subjects == subject) & (folds == test_fold)
    others = subjects != subject
    seen = others & np.isin(folds, [test_fold, (test_fold + 1) % N_FOLDS])
    unseen = others & (folds != test_fold)
    scores = {}
    for name, train in (("seen", seen), ("unseen", unseen)):
        predicted = fit(x[train][:, columns], y[train]).predict(x[test][:, columns])
        scores[name] = balanced_accuracy_score(y[test], predicted)
    return subject, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/clip_heldout_emotion.md"))
    args = parser.parse_args()

    results = []
    report = ["# Emotion accuracy when the test films were seen vs unseen in training", "",
              "Balanced accuracy (%), leave one subject out, mean over test subjects (each averaged over the 3 "
              "film folds). Same test windows and training size in both columns. Chance = 33%.", "",
              "| Culture | Modality | Seen films | Unseen films | Drop | Subjects dropping | Wilcoxon p |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for culture, root in CULTURES.items():
        x, y, clips, subjects = load(args.data_root / root)
        folds = clip_folds(y, clips)
        for modality, columns in MODALITIES.items():
            runs = Parallel(n_jobs=-1)(
                delayed(run_fold)(x, y, folds, subjects, s, f, columns)
                for s in np.unique(subjects) for f in range(N_FOLDS))
            per_subject = {s: {k: np.mean([r[k] for t, r in runs if t == s]) for k in ("seen", "unseen")}
                           for s in np.unique(subjects)}
            seen = np.array([v["seen"] for v in per_subject.values()])
            unseen = np.array([v["unseen"] for v in per_subject.values()])
            drop = seen - unseen
            p = float(wilcoxon(drop).pvalue)
            results.append({"culture": culture, "modality": modality, "seen": seen.tolist(),
                            "unseen": unseen.tolist(), "p": p})
            line = (f"| {culture} | {modality} | {100 * seen.mean():.1f} | {100 * unseen.mean():.1f} | "
                    f"{100 * drop.mean():+.1f} | {(drop > 0).sum()}/{len(drop)} | {p:.3f} |")
            report.append(line)
            print(line)
    args.output.write_text("\n".join(report) + "\n")
    args.output.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
