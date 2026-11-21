"""SCVAE-XC Stage 2, step 3: emotion-aware alignment on top of step 4 (3-seed ensembles, strict).

1. One training culture: eye_1src (step 4) vs eye_1src_ca.
2. Two training cultures -> the third: eye_2src (step 4) vs eye_2src_cmmd and eye_2src_ca, with NormFusion v3 fair.

    python -m src.experiments.summarize_scvae_xc_step3

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md.
"""

from pathlib import Path

import numpy as np

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, V3_TWO, arm, compare, fmt

PREDICTED = {"eye_2src_cmmd": 68, "eye_2src_ca": 69, "eye_1src_ca": 62}
OUT = Path("results/tables/scvae_xc_step3_summary.md")


def main():
    lines = ["# SCVAE-XC Stage 2, step 3: emotion-aware alignment", "",
             "Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final "
             "evaluation. p: Wilcoxon over people against step 4.", "",
             "## 1. One training culture", "",
             "| Train → Test | eye_1src (step 4) | eye_1src_ca | p | runs (ca) |", "|---|---:|---:|---:|---|"]
    gains_1, wins_1, rows_1 = [], 0, []
    for s, t in DIRECTIONS:
        base, _ = arm(t, "eye_1src", [s])
        new, cov = arm(t, "eye_1src_ca", [s])
        m_new, m_base, p = compare(new, base)
        if m_new is not None:
            gains_1.append(m_new - m_base); wins_1 += m_new > m_base; rows_1.append((m_base, m_new))
        lines.append(f"| {NAMES[s]} → {NAMES[t]} | {fmt(m_base)} | {fmt(m_new)} | {p} | {cov} |")
    if rows_1:
        b, n = np.mean(rows_1, 0)
        lines.append(f"| **Average** | **{b:.1f}** | **{n:.1f}** | | |")
    lines += ["", "## 2. Two training cultures → the third", "",
              "| Train → Test | eye_2src (step 4) | eye_2src_cmmd | eye_2src_ca | p (ca) | NormFusion v3 fair | runs (ca) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    gains_2, wins_2, rows_2 = [], 0, []
    for t in CULTURES:
        sources = [c for c in CULTURES if c != t]
        base, _ = arm(t, "eye_2src", sources)
        cmmd, _ = arm(t, "eye_2src_cmmd", sources)
        new, cov = arm(t, "eye_2src_ca", sources)
        m_new, m_base, p = compare(new, base)
        m_cmmd = np.mean(list(cmmd.values())) if cmmd else None
        if m_new is not None:
            gains_2.append(m_new - m_base); wins_2 += m_new > m_base
            rows_2.append((m_base, m_cmmd if m_cmmd is not None else np.nan, m_new, V3_TWO[t]))
        lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[t]} | {fmt(m_base)} | {fmt(m_cmmd)} | "
                     f"{fmt(m_new)} | {p} | {V3_TWO[t]:.1f} | {cov} |")
    if rows_2:
        b, c, n, v = np.nanmean(rows_2, 0)
        lines.append(f"| **Average** | **{b:.1f}** | **{c:.1f}** | **{n:.1f}** | | {v:.1f} | |")
    lines += ["", f"Predicted: {', '.join(f'{k} {v}' for k, v in PREDICTED.items())}.", "",
              "## Go check (pre-registered)", ""]
    go_2 = bool(gains_2) and np.mean(gains_2) >= 1.5 and wins_2 >= 2
    go_1 = bool(gains_1) and np.mean(gains_1) >= 1.5 and wins_1 >= 4
    if gains_2:
        lines.append(f"- eye_2src_ca − eye_2src: {np.mean(gains_2):+.1f} points, higher in {wins_2} of {len(gains_2)} "
                     f"(needs ≥ +1.5 and ≥ 2 of 3): {'met' if go_2 else 'not met'}.")
    if gains_1:
        lines.append(f"- eye_1src_ca − eye_1src: {np.mean(gains_1):+.1f} points, higher in {wins_1} of {len(gains_1)} "
                     f"(needs ≥ +1.5 and ≥ 4 of 6): {'met' if go_1 else 'not met'}.")
    complete = len(gains_1) == len(DIRECTIONS) and len(gains_2) == len(CULTURES)
    verdict = ("GO" if go_1 or go_2 else "NO-GO") if complete else "incomplete, no verdict"
    lines.append(f"- **Step 3: {verdict}.**")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
