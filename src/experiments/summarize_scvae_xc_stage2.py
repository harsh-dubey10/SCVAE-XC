"""SCVAE-XC Stage 2, step 4: eye branch and two training cultures (3-seed ensembles, strict).

1. One training culture: SCVAE-fixed (2026-10-09 runs) vs the re-implementation (eeg_1src) vs + eye (eye_1src).
2. Two training cultures -> the third: eeg_2src and eye_2src vs the reference (per person, the mean of
   SCVAE-fixed's two one-culture directions into that target), with NormFusion v3 fair for context.

    python -m src.experiments.summarize_scvae_xc_stage2

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md.
"""

from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_improve import coverage, coverage_text, ensemble
from src.experiments.summarize_scvae_xc_stage1 import PEOPLE, run_dirs

ROOT = Path("results/scvae_xc/stage2")
CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
# results/tables/norm_fusion_v3_summary.md, "Two cultures -> the third", v3 fair.
V3_TWO = {"SEED_CHN_MM": 72.8, "SEED_FRA": 67.9, "SEED_GER": 68.6}
PREDICTED = {"eeg_1src": 53, "eye_1src": 57, "eeg_2src": 57, "eye_2src": 61}


def arm(target, name, sources):
    run = ROOT / target / f"{name}__from_{'+'.join(sources)}"
    return {p: v[0] for p, v in ensemble(run).items()}, coverage_text(coverage(run, PEOPLE[target], kind="proba"))


def compare(new, ref):
    people = sorted(set(new) & set(ref))
    if not people:
        return None, None, "–"
    a, b = np.array([new[p] for p in people]), np.array([ref[p] for p in people])
    return a.mean(), b.mean(), (f"{wilcoxon(a - b).pvalue:.3f}" if np.any(a != b) else "–")


def fmt(v):
    return "–" if v is None else f"{v:.1f}"


def main():
    lines = ["# SCVAE-XC Stage 2, step 4: eye branch and two training cultures", "",
             "Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final "
             "evaluation. p: Wilcoxon over people against the reference column.", "",
             "## 1. One training culture", "",
             "| Train → Test | SCVAE-fixed (2026-10-09) | eeg_1src (re-implementation) | eye_1src (+ eye) | p (eye vs fixed) | runs (eye) |",
             "|---|---:|---:|---:|---:|---|"]
    one = {"fixed": [], "eeg_1src": [], "eye_1src": []}
    wins_1 = 0
    for s, t in DIRECTIONS:
        fixed = {p: v[0] for p, v in ensemble(run_dirs(s, t)[0]).items()}
        eeg, _ = arm(t, "eeg_1src", [s])
        eye, cov = arm(t, "eye_1src", [s])
        m_eye, m_fixed, p = compare(eye, fixed)
        m_eeg = np.mean(list(eeg.values())) if eeg else None
        if m_eye is not None:
            one["fixed"].append(m_fixed); one["eye_1src"].append(m_eye); wins_1 += m_eye > m_fixed
        if m_eeg is not None:
            one["eeg_1src"].append(m_eeg)
        lines.append(f"| {NAMES[s]} → {NAMES[t]} | {fmt(np.mean(list(fixed.values())))} | {fmt(m_eeg)} | {fmt(m_eye)} "
                     f"| {p} | {cov} |")
    avg = {k: (np.mean(v) if v else None) for k, v in one.items()}
    lines.append(f"| **Average** | **{fmt(avg['fixed'])}** | **{fmt(avg['eeg_1src'])}** | **{fmt(avg['eye_1src'])}** | | |")
    lines += ["", f"Predicted: eeg_1src {PREDICTED['eeg_1src']}, eye_1src {PREDICTED['eye_1src']}. "
              f"eye_1src higher than SCVAE-fixed in {wins_1} of {len(one['eye_1src'])} directions.", "",
              "## 2. Two training cultures → the third", "",
              "| Train → Test | Reference (SCVAE-fixed, mean of the two one-culture directions) | eeg_2src | eye_2src "
              "| p (eye_2src vs ref) | NormFusion v3 fair (context) | runs (eye) |", "|---|---:|---:|---:|---:|---:|---|"]
    two = {"ref": [], "eeg_2src": [], "eye_2src": []}
    wins_2 = 0
    for t in CULTURES:
        sources = [c for c in CULTURES if c != t]
        per_dir = [{p: v[0] for p, v in ensemble(run_dirs(s, t)[0]).items()} for s in sources]
        ref = {p: np.mean([d[p] for d in per_dir]) for p in set.intersection(*(set(d) for d in per_dir))}
        eeg, _ = arm(t, "eeg_2src", sources)
        eye, cov = arm(t, "eye_2src", sources)
        m_eye, m_ref, p = compare(eye, ref)
        m_eeg = np.mean(list(eeg.values())) if eeg else None
        if m_eye is not None:
            two["ref"].append(m_ref); two["eye_2src"].append(m_eye); wins_2 += m_eye > m_ref
        if m_eeg is not None:
            two["eeg_2src"].append(m_eeg)
        lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[t]} | {fmt(np.mean(list(ref.values())))} | "
                     f"{fmt(m_eeg)} | {fmt(m_eye)} | {p} | {V3_TWO[t]:.1f} | {cov} |")
    avg2 = {k: (np.mean(v) if v else None) for k, v in two.items()}
    lines.append(f"| **Average** | **{fmt(avg2['ref'])}** | **{fmt(avg2['eeg_2src'])}** | **{fmt(avg2['eye_2src'])}** "
                 f"| | {np.mean(list(V3_TWO.values())):.1f} | |")
    go_two = avg2["eye_2src"] is not None and avg2["eye_2src"] - avg2["ref"] >= 5 and wins_2 == 3
    go_one = avg["eye_1src"] is not None and avg["eye_1src"] - avg["fixed"] >= 2 and wins_1 >= 4
    lines += ["", f"Predicted: eeg_2src {PREDICTED['eeg_2src']}, eye_2src {PREDICTED['eye_2src']}.", "",
              "## Go check (pre-registered)", "",
              f"- eye_2src − reference: {fmt(None if avg2['eye_2src'] is None else avg2['eye_2src'] - avg2['ref'])} "
              f"points, higher in {wins_2} of 3 (needs ≥ +5 and 3 of 3): {'met' if go_two else 'not met'}.",
              f"- eye_1src − SCVAE-fixed: {fmt(None if avg['eye_1src'] is None else avg['eye_1src'] - avg['fixed'])} "
              f"points, higher in {wins_1} of 6 (needs ≥ +2 and ≥ 4): {'met' if go_one else 'not met'}.",
              f"- **Step 4: {'GO' if go_two or go_one else 'NO-GO'}.**", ""]
    text = "\n".join(lines)
    Path("results/tables/scvae_xc_stage2_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
