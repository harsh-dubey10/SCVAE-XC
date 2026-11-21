"""SCVAE-XC Stage 2, step 5 (test-time adaptation) and the test-label-selected column (3-seed ensembles).

1. Step 5, strict: eye_1src / eye_2src (step 4) vs eye_*_tta_im and eye_2src_tta_ent, with the pre-registered check.
2. Test-label-selected (paper protocol, never accuracy): the step 4 arms re-run with per-epoch test predictions
   (results/scvae_xc/peek); strict (re-run check), one epoch for all test people of a direction, best epoch per person.

    python -m src.experiments.summarize_scvae_xc_step5

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md.
"""

import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_improve import ensemble
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, V3_TWO, arm, compare, fmt

PEEK = Path("results/scvae_xc/peek")
STAGE2 = Path("results/scvae_xc/stage2")
OUT = Path("results/tables/scvae_xc_step5_summary.md")
PREDICTED = {"eye_2src_tta_im": 68, "eye_2src_tta_ent": 67, "eye_1src_tta_im": 61}


def settings():
    """(label, test culture, sources) for the 6 one-culture directions, then the 3 two-culture settings."""
    return ([(f"{NAMES[s]} → {NAMES[t]}", t, [s]) for s, t in DIRECTIONS]
            + [(f"{' + '.join(NAMES[c] for c in CULTURES if c != t)} → {NAMES[t]}", t, [c for c in CULTURES if c != t])
               for t in CULTURES])


def epoch_accuracy(run_dir):
    """Test person -> per-epoch accuracy % of the seed-mean per-epoch predictions (seeds with all epochs saved)."""
    files = defaultdict(list)
    for path in sorted(Path(run_dir).glob("session_all/seed_*/subject_*_epochs.npz")):
        files[int(re.search(r"subject_(\d+)_epochs", path.name).group(1))].append(np.load(path))
    out = {}
    for person, saved in files.items():
        labels = saved[0]["labels"]
        mean = np.mean([s["proba"].astype(np.float32) for s in saved], axis=0)  # epochs x windows x classes
        out[person] = 100 * (mean.argmax(2) == labels).mean(1)
    return out


def step5(lines):
    lines += ["## 1. Step 5: per-person test-time adaptation (strict)", "",
              "| Train → Test | step 4 | tta_im | tta_ent | p (im) | NormFusion v3 fair | runs (im) |",
              "|---|---:|---:|---:|---:|---:|---|"]
    gains, wins, rows = {1: [], 2: []}, {1: 0, 2: 0}, {1: [], 2: []}
    for i, (label, t, sources) in enumerate(settings()):
        k = len(sources)
        base, _ = arm(t, f"eye_{k}src", sources)
        new, cov = arm(t, f"eye_{k}src_tta_im", sources)
        ent, _ = arm(t, f"eye_{k}src_tta_ent", sources) if k == 2 else ({}, "")
        m_new, m_base, p = compare(new, base)
        m_ent = np.mean(list(ent.values())) if ent else None
        if m_new is not None:
            gains[k].append(m_new - m_base); wins[k] += m_new > m_base; rows[k].append((m_base, m_new))
        v3 = f"{V3_TWO[t]:.1f}" if k == 2 else ""
        lines.append(f"| {label} | {fmt(m_base)} | {fmt(m_new)} | {fmt(m_ent)} | {p} | {v3} | {cov} |")
        if i == len(DIRECTIONS) - 1 and rows[1]:
            b, n = np.mean(rows[1], 0)
            lines.append(f"| **Average, one culture** | **{b:.1f}** | **{n:.1f}** | | | | |")
    if rows[2]:
        b, n = np.mean(rows[2], 0)
        lines.append(f"| **Average, two cultures** | **{b:.1f}** | **{n:.1f}** | | | {np.mean(list(V3_TWO.values())):.1f} | |")
    go_2 = bool(gains[2]) and np.mean(gains[2]) >= 1.5 and wins[2] >= 2
    go_1 = bool(gains[1]) and np.mean(gains[1]) >= 1.5 and wins[1] >= 4
    lines += ["", f"Predicted: {', '.join(f'{k} {v}' for k, v in PREDICTED.items())}.", "", "Go check (pre-registered):", ""]
    if gains[2]:
        lines.append(f"- eye_2src_tta_im − eye_2src: {np.mean(gains[2]):+.1f}, higher in {wins[2]} of {len(gains[2])} "
                     f"(needs ≥ +1.5 and ≥ 2 of 3): {'met' if go_2 else 'not met'}.")
    if gains[1]:
        lines.append(f"- eye_1src_tta_im − eye_1src: {np.mean(gains[1]):+.1f}, higher in {wins[1]} of {len(gains[1])} "
                     f"(needs ≥ +1.5 and ≥ 4 of 6): {'met' if go_1 else 'not met'}.")
    complete = len(gains[1]) == len(DIRECTIONS) and len(gains[2]) == len(CULTURES)
    lines.append(f"- **Step 5: {('GO' if go_1 or go_2 else 'NO-GO') if complete else 'incomplete, no verdict'}.**")


def peek(lines):
    lines += ["", "## 2. Test-label-selected column (paper protocol — NOT accuracy)", "",
              "Strict = epoch chosen on held-out source people (the reportable number). The other two columns choose "
              "the epoch by reading the test people's labels, as the published papers did; shown only to measure the "
              "inflation.", "",
              "| Train → Test | step 4 strict (original runs) | re-run strict | one epoch for all | best epoch per person | inflation (per person) |",
              "|---|---:|---:|---:|---:|---:|"]
    avg = defaultdict(list)
    for i, (label, t, sources) in enumerate(settings()):
        name = f"eye_{len(sources)}src__from_{'+'.join(sources)}"
        orig = {p: v[0] for p, v in ensemble(STAGE2 / t / name).items()}
        rerun = {p: v[0] for p, v in ensemble(PEEK / t / name).items()}
        curves = epoch_accuracy(PEEK / t / name)
        if not curves:
            lines.append(f"| {label} | {fmt(np.mean(list(orig.values())) if orig else None)} | – | – | – | – |")
            continue
        people = sorted(curves)
        table = np.array([curves[p] for p in people])  # people x epochs
        one, each = table.mean(0).max(), table.max(1).mean()
        o, r = np.mean([orig[p] for p in people]), np.mean([rerun[p] for p in people])
        k = len(sources)
        for key, v in (("orig", o), ("rerun", r), ("one", one), ("each", each)):
            avg[(k, key)].append(v)
        lines.append(f"| {label} | {o:.1f} | {r:.1f} | {one:.1f} | {each:.1f} | {each - r:+.1f} |")
        if i in (len(DIRECTIONS) - 1, len(settings()) - 1):
            a = {key: np.mean(avg[(k, key)]) for key in ("orig", "rerun", "one", "each")}
            lines.append(f"| **Average, {'one culture' if k == 1 else 'two cultures'}** | **{a['orig']:.1f}** | "
                         f"**{a['rerun']:.1f}** | **{a['one']:.1f}** | **{a['each']:.1f}** | **{a['each'] - a['rerun']:+.1f}** |")
    lines += ["", "Predicted: eye_2src strict 67, one epoch 70, per person 76; eye_1src strict 60, one epoch 63, "
              "per person 70. Re-run check: re-run strict within ±1 point of the original runs."]


def main():
    lines = ["# SCVAE-XC Stage 2, step 5 and the test-label-selected column", "",
             "Accuracy %, 3-seed ensemble, mean over test people. p: Wilcoxon over people against step 4.", ""]
    step5(lines)
    peek(lines)
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
