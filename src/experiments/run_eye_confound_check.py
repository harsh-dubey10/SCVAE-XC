"""Eye-feature confound check: is the eye gain emotion, or film recognition through the pupil?

1. 3-class and 2. positive-vs-negative accuracy, two training cultures -> the third (every test film unseen), for
eeg / eeg+pupil / eeg+movement / eeg+eye; 3. film identification within each culture (person-held-out) from pupil,
movement or EEG alone. StandardScaler + logistic regression, C = 0.01.

    python -m src.experiments.run_eye_confound_check

Design: docs/superpowers/specs/2026-10-29-eye-confound-check-design.md.
"""

import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.norm_fusion import load_culture
from src.experiments.run_film_curve import CULTURES, source_pool

C = 0.01
EEG = np.arange(310)
PUPIL = 310 + np.arange(12)       # inferred: columns 0-11 of the 33 eye features
MOVEMENT = 310 + np.arange(12, 33)
SETS = {"eeg": EEG, "eeg+pupil": np.r_[EEG, PUPIL], "eeg+movement": np.r_[EEG, MOVEMENT],
        "eeg+eye": np.r_[EEG, PUPIL, MOVEMENT]}
FILM_SETS = {"pupil": PUPIL, "movement": MOVEMENT, "eeg": EEG}
NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))


def emotion_score(target, features, binary):
    """Mean over test people of accuracy; binary: positive vs negative windows only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y, _ = source_pool(target)
        tgt = load_culture(target)
        keep, tkeep = (y != 1, tgt.y != 1) if binary else (np.ones(len(y), bool), np.ones(len(tgt.y), bool))
        model = lr().fit(x[keep][:, features], y[keep])
        correct = model.predict(tgt.x[tkeep][:, features]) == tgt.y[tkeep]
        people = tgt.subjects[tkeep]
    return float(np.mean([100 * correct[people == s].mean() for s in np.unique(people)]))


def film_id_score(culture, features):
    """Film identification within one culture: 3 person-held-out folds, mean accuracy over folds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = load_culture(culture)
        group_of = {s: i % 3 for i, s in enumerate(np.unique(data.subjects))}
        groups = np.array([group_of[s] for s in data.subjects])
        scores = []
        for g in range(3):
            model = lr().fit(data.x[groups != g][:, features], data.clips[groups != g])
            scores.append(100 * (model.predict(data.x[groups == g][:, features]) == data.clips[groups == g]).mean())
    return float(np.mean(scores)), 100 / len(np.unique(data.clips))


def main():
    emotion_jobs = [(t, s, b) for b in (False, True) for t in CULTURES for s in SETS]
    film_jobs = [(c, f) for c in CULTURES for f in FILM_SETS]
    emo = Parallel(n_jobs=-1)(delayed(emotion_score)(t, SETS[s], b) for t, s, b in emotion_jobs)
    film = Parallel(n_jobs=-1)(delayed(film_id_score)(c, FILM_SETS[f]) for c, f in film_jobs)
    emo = dict(zip(emotion_jobs, emo)); film = dict(zip(film_jobs, film))

    lines = ["# Eye-feature confound check", "",
             "Two training cultures → the third, every test film unseen; logistic regression (C = 0.01); accuracy %, "
             "mean over test people. Pupil = eye columns 0–11 (inferred from the data, not documented). "
             "Design: docs/superpowers/specs/2026-10-29-eye-confound-check-design.md.", ""]
    mean = {}
    for binary, title in ((False, "1. Three classes"), (True, "2. Positive vs negative")):
        lines += [f"## {title}", "", "| Target | " + " | ".join(SETS) + " |", "|---|" + "---:|" * len(SETS)]
        for t in CULTURES:
            lines.append(f"| {NAMES[t]} | " + " | ".join(f"{emo[(t, s, binary)]:.1f}" for s in SETS) + " |")
        mean[binary] = {s: np.mean([emo[(t, s, binary)] for t in CULTURES]) for s in SETS}
        lines += ["| **Mean** | " + " | ".join(f"**{mean[binary][s]:.1f}**" for s in SETS) + " |", ""]
    lines += ["## 3. Film identification within a culture (person-held-out)", "",
              "| Culture | chance | " + " | ".join(FILM_SETS) + " |", "|---|---:|" + "---:|" * len(FILM_SETS)]
    for c in CULTURES:
        lines.append(f"| {NAMES[c]} | {film[(c, 'eeg')][1]:.1f} | " + " | ".join(f"{film[(c, f)][0]:.1f}" for f in FILM_SETS) + " |")
    film_mean = {f: np.mean([film[(c, f)][0] for c in CULTURES]) for f in FILM_SETS}
    lines += ["| **Mean** | | " + " | ".join(f"**{film_mean[f]:.1f}**" for f in FILM_SETS) + " |", ""]

    m3, m2 = mean[False], mean[True]
    eye_gain, pupil_gain = m3["eeg+eye"] - m3["eeg"], m3["eeg+pupil"] - m3["eeg"]
    pupil_binary = m2["eeg+pupil"] - m2["eeg"]
    confound = (eye_gain > 0 and pupil_gain >= 2 / 3 * eye_gain and pupil_binary < 1
                and film_mean["pupil"] > film_mean["movement"])
    movement_enough = m3["eeg+movement"] >= m3["eeg+eye"] - 1.5
    emotion = pupil_binary >= 2
    verdict = ("PUPIL CONFOUND: drop pupil" if confound else "DROP PUPIL (adds little beyond movement)"
               if movement_enough else "PUPIL CARRIES EMOTION: keep, with caveat" if emotion else "INCONCLUSIVE")
    lines += ["## Pre-registered decision", "",
              f"- 3-class: eye gain {eye_gain:+.1f}; pupil gain {pupil_gain:+.1f} "
              f"({100 * pupil_gain / eye_gain:.0f}% of it); movement gain {m3['eeg+movement'] - m3['eeg']:+.1f}.",
              f"- Positive vs negative: pupil gain {pupil_binary:+.1f}; movement gain {m2['eeg+movement'] - m2['eeg']:+.1f}.",
              f"- Film identification: pupil {film_mean['pupil']:.1f} vs movement {film_mean['movement']:.1f} "
              f"vs EEG {film_mean['eeg']:.1f}.", f"- **{verdict}.**"]
    table = "\n".join(lines)
    Path("results/tables/eye_confound_summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
