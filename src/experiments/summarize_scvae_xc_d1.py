"""SCVAE-XC D1: eeg+pupil input (pupil_*) vs all 33 eye features (eye_*, step 4); 3-seed ensembles, strict.

Secondary: positive vs negative accuracy on positive and negative windows only, deciding between the two classes by
the ensemble's probabilities.

    python -m src.experiments.summarize_scvae_xc_d1

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md (D1).
"""

from pathlib import Path

import numpy as np

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_improve import seed_files
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, ROOT, arm, compare, fmt

PREDICTED = {"pupil_2src": 67.5, "pupil_1src": 61.0}
OUT = Path("results/tables/scvae_xc_d1_summary.md")


def positive_negative(name, target, sources):
    """Mean over test people of accuracy on positive and negative windows (classes 2 and 0)."""
    scores = []
    for saved in seed_files(ROOT / target / f"{name}__from_{'+'.join(sources)}").values():
        labels, proba = saved[0]["labels"], np.mean([s["proba"] for s in saved], axis=0)
        keep = labels != 1
        predicted = np.where(proba[keep, 2] > proba[keep, 0], 2, 0)
        scores.append(100 * np.mean(predicted == labels[keep]))
    return np.mean(scores) if scores else None


def main():
    lines = ["# SCVAE-XC D1: eeg+pupil input", "",
             "Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final "
             "evaluation. p: Wilcoxon over people, pupil vs eye. Pos/neg: positive vs negative windows only.", ""]
    gains = {}
    for title, rows, n in (("1. One training culture", [([s], t) for s, t in DIRECTIONS], 1),
                           ("2. Two training cultures → the third",
                            [([c for c in CULTURES if c != t], t) for t in CULTURES], 2)):
        lines += [f"## {title}", "", "| Train → Test | eye (step 4) | pupil | p | pos/neg eye | pos/neg pupil | runs (pupil) |",
                  "|---|---:|---:|---:|---:|---:|---|"]
        table = []
        for sources, t in rows:
            eye, _ = arm(t, f"eye_{n}src", sources)
            pupil, cov = arm(t, f"pupil_{n}src", sources)
            m_pupil, m_eye, p = compare(pupil, eye)
            pn_eye, pn_pupil = positive_negative(f"eye_{n}src", t, sources), positive_negative(f"pupil_{n}src", t, sources)
            if m_pupil is not None:
                table.append((m_eye, m_pupil, pn_eye, pn_pupil))
            lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[t]} | {fmt(m_eye)} | {fmt(m_pupil)} | {p} "
                         f"| {fmt(pn_eye)} | {fmt(pn_pupil)} | {cov} |")
        if table:
            e, u, pe, pu = np.mean(table, 0)
            lines.append(f"| **Average** | **{e:.1f}** | **{u:.1f}** | | **{pe:.1f}** | **{pu:.1f}** | |")
            gains[n] = (u - e, len(table) == len(rows))
        lines.append("")
    lines += [f"Predicted: {', '.join(f'{k} {v}' for k, v in PREDICTED.items())}.", "", "## Decision (pre-registered)", ""]
    complete = all(gains.get(n, (0, False))[1] for n in (1, 2))
    if complete:
        ok_2, ok_1 = gains[2][0] >= -0.5, gains[1][0] >= -1.0
        lines += [f"- pupil_2src − eye_2src: {gains[2][0]:+.1f} (needs ≥ −0.5): {'met' if ok_2 else 'not met'}.",
                  f"- pupil_1src − eye_1src: {gains[1][0]:+.1f} (needs ≥ −1.0): {'met' if ok_1 else 'not met'}.",
                  f"- **D1: {'ADOPT' if ok_1 and ok_2 else 'KEEP all 33 eye features'}.**"]
    else:
        lines.append("- **D1: incomplete, no verdict.**")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
