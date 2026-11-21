"""Per-film accuracy from saved probabilities (no training): positive vs negative per window, per film and day,
and per film pooled over a person's 3 recording days; also 3-class per film. v3 fair (5-seed mean) and SCVAE-XC
(3-seed mean); two cultures -> the third, and one culture.

    python -m src.experiments.per_film_accuracy

Design: docs/superpowers/specs/2026-11-05-per-film-accuracy-design.md.
"""

from pathlib import Path

import numpy as np

from src.baselines.norm_fusion import load_culture
from src.experiments.run_norm_fusion_v3 import SEEDS
from src.experiments.run_xc_v3_combination import CACHE, ORDER, xc_seed_average
from src.experiments.summarize_dcca_am import NAMES

OUT = Path("results/tables/per_film_accuracy_summary.md")
LEVELS = ["window pos/neg", "film, one day, pos/neg", "film, 3 days, pos/neg", "film, one day, 3-class",
          "film, 3 days, 3-class"]


def v3_proba(sources, target_name, setting):
    folder = CACHE / target_name / f"{setting}__from_{'+'.join(sources)}"
    return np.mean([np.load(folder / f"seed_{s}.npz")["proba"] for s in SEEDS], axis=0)


def person_scores(proba, y, clips, recordings):
    """Accuracy % at each level for one person; also the item count per level."""
    pn = y != 1
    window = np.mean(np.where(proba[pn, 2] > proba[pn, 0], 2, 0) == y[pn])
    out, counts = [window], [int(pn.sum())]
    for three_class in (False, True):
        for pooled in (False, True):
            keys = sorted({(0 if pooled else r, c) for r, c in zip(recordings, clips)})
            hits = []
            for r, c in keys:
                rows = (clips == c) & (True if pooled else recordings == r)
                label = y[rows][0]
                if not three_class and label == 1:
                    continue
                mean = proba[rows].mean(0)
                guess = mean.argmax() if three_class else (2 if mean[2] > mean[0] else 0)
                hits.append(guess == label)
            out.append(np.mean(hits)); counts.append(len(hits))
    order = [0, 1, 2, 3, 4]  # window, pn one day, pn pooled, 3c one day, 3c pooled
    return 100 * np.array(out)[order], np.array(counts)[order]


def setting_scores(sources, target_name, setting, model):
    target = load_culture(target_name)
    arm = "eye_2src" if setting == "emotion_two_cultures" else "eye_1src"
    xc = xc_seed_average(arm, sources, target, target_name) if model == "SCVAE-XC" else None
    v3 = v3_proba(sources, target_name, setting) if model == "v3" else None
    scores, counts = [], 0
    for person in np.unique(target.subjects):
        mask = target.subjects == person
        proba = xc[person] if xc is not None else v3[mask]
        s, n = person_scores(proba, target.y[mask], target.clips[mask], target.recordings[mask])
        scores.append(s); counts = counts + n
    return np.mean(scores, 0), counts


def main():
    lines = ["# Per-film accuracy (exploratory; saved probabilities, no training)", "",
             "Accuracy %, mean over test people. pos/neg: positive and negative films only, decided by mean "
             "p(positive) vs p(negative). 3 days: windows of the person's 3 recording days of the same film pooled. "
             "Design: docs/superpowers/specs/2026-11-05-per-film-accuracy-design.md.", ""]
    for setting, title in (("emotion_two_cultures", "Two training cultures → the third"),
                           ("emotion", "One training culture")):
        for model in ("v3", "SCVAE-XC"):
            lines += [f"## {title}: {model}", "", "| Train → Test | " + " | ".join(LEVELS) + " |",
                      "|---|" + "---:|" * len(LEVELS)]
            rows, total = [], 0
            for sources, target_name, st in ORDER:
                if st != setting:
                    continue
                m, n = setting_scores(sources, target_name, setting, model)
                rows.append(m); total = total + n
                lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[target_name]} | "
                             + " | ".join(f"{v:.1f}" for v in m) + " |")
            a = np.mean(rows, 0)
            lines += ["| **Average** | " + " | ".join(f"**{v:.1f}**" for v in a) + " |",
                      "| items scored | " + " | ".join(str(int(v)) for v in total) + " |", ""]
    lines.append("Prediction (v3, two cultures): window pos/neg 78, film one day 85, film 3 days 90; 3-class film 3 days 78.")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
