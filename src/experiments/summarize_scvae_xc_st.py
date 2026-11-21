"""SCVAE-XC self-training (eye_*_st) vs step 4 (eye_*), 3-seed ensembles, strict.

    python -m src.experiments.summarize_scvae_xc_st

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md (self-training).
"""

from pathlib import Path

import numpy as np

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, V3_TWO, arm, compare, fmt

PREDICTED = {"eye_2src_st": 68.8, "eye_1src_st": 61.5}
OUT = Path("results/tables/scvae_xc_st_summary.md")


def main():
    lines = ["# SCVAE-XC + v3 self-training", "",
             "Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final "
             "evaluation. p: Wilcoxon over people against step 4.", ""]
    gains = {}
    for title, rows, n in (("1. One training culture", [([s], t) for s, t in DIRECTIONS], 1),
                           ("2. Two training cultures → the third",
                            [([c for c in CULTURES if c != t], t) for t in CULTURES], 2)):
        extra = " | NormFusion v3 fair" if n == 2 else ""
        lines += [f"## {title}", "", f"| Train → Test | eye_{n}src (step 4) | eye_{n}src_st | p{extra} | runs |",
                  "|---|---:|---:|---:|" + ("---:|" if n == 2 else "") + "---|"]
        table, wins = [], 0
        for sources, t in rows:
            base, _ = arm(t, f"eye_{n}src", sources)
            new, cov = arm(t, f"eye_{n}src_st", sources)
            m_new, m_base, p = compare(new, base)
            if m_new is not None:
                table.append((m_base, m_new)); wins += m_new > m_base
            v3 = f" | {V3_TWO[t]:.1f}" if n == 2 else ""
            lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[t]} | {fmt(m_base)} | {fmt(m_new)} | {p}{v3} | {cov} |")
        if table:
            b, u = np.mean(table, 0)
            v3 = f" | {np.mean(list(V3_TWO.values())):.1f}" if n == 2 else ""
            lines.append(f"| **Average** | **{b:.1f}** | **{u:.1f}** | {v3} | |")
            gains[n] = (u - b, wins, len(table) == len(rows))
        lines.append("")
    lines += [f"Predicted: {', '.join(f'{k} {v}' for k, v in PREDICTED.items())}.", "", "## Decision (pre-registered)", ""]
    if 2 in gains and gains[2][2]:
        g, w, _ = gains[2]
        go = g >= 1.0 and w >= 2
        lines += [f"- eye_2src_st − eye_2src: {g:+.1f}, higher in {w} of 3 (needs ≥ +1.0 and ≥ 2 of 3).",
                  f"- **Self-training: {'GO' if go else 'NO-GO'}.**"]
        if 1 in gains:
            lines.append(f"- One culture (no decision): {gains[1][0]:+.1f}, higher in {gains[1][1]} of 6.")
    else:
        lines.append("- **Self-training: incomplete, no verdict.**")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
