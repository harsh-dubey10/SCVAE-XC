"""Stimulus check: can a feature set tell apart clips that share one emotion?

Within each culture and emotion, a logistic regression is trained to predict
which clip a window came from, leaving one subject out (all sessions,
per-recording normalisation, as in the normalised baselines). Emotion is held
fixed, so accuracy above the majority rate means the features carry
clip-specific information (e.g. screen brightness driving pupil size) that is
shared across people. Clips are labelled per session, so a clip repeated in
several sessions counts as several classes (conservative).

    python -m src.experiments.clip_identity_check   # -> results/tables/clip_identity_check.md
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.train_scvae_reference import available_subjects, normalize_recording, sessions_in
from src.experiments.verify_claims import CULTURES, NAMES

EMOTIONS = ["negative", "neutral", "positive"]
MODALITIES = {"EEG": "", "eye": "_EYE"}


def load(root):
    xs, ys, clips, groups = [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = normalize_recording(torch.stack(data["data"]).float()).flatten(start_dim=1).numpy()
            xs.append(x)
            ys.append(np.array(data["labels"]))
            clips.append(100 * session + np.array(data["trials"]))
            groups.append(np.full(len(x), subject))
    return [np.concatenate(v) for v in (xs, ys, clips, groups)]


def fold(x, clip, group, subject):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))
        return model.fit(x[group != subject], clip[group != subject]).score(x[group == subject], clip[group == subject])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/clip_identity_check.md"))
    args = parser.parse_args()
    report = ["# Clip-identity check", "",
              "Accuracy (%) of predicting the clip among clips of the same emotion, leave-one-subject-out, "
              "mean over held-out subjects. Majority = always guessing the most frequent clip.", "",
              "| Culture | Emotion | Clips | Majority | EEG | eye |", "|---|---|---:|---:|---:|---:|"]
    for culture in CULTURES:
        loaded = {name: load(args.data_root / f"{culture}{suffix}") for name, suffix in MODALITIES.items()}
        for emotion, emotion_name in enumerate(EMOTIONS):
            cells = []
            for x, y, clip, group in loaded.values():
                keep = y == emotion
                scores = Parallel(n_jobs=-1)(delayed(fold)(x[keep], clip[keep], group[keep], s)
                                             for s in np.unique(group))
                cells.append(f"{100 * np.mean(scores):.1f}")
            counts = np.unique(clip[keep], return_counts=True)[1]
            report.append(f"| {NAMES[culture]} | {emotion_name} | {len(counts)} | "
                          f"{100 * counts.max() / counts.sum():.1f} | " + " | ".join(cells) + " |")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
