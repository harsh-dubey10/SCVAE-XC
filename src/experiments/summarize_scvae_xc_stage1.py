"""SCVAE-XC Stage 1: SCVAE-fixed vs SCVAE-fixed with the SLC-pretrained encoder (3-seed ensembles, strict).

Both arms use the configuration chosen per test culture on 2026-10-09 and seeds 12, 34, 56; they differ only in
the encoder's initialisation. Also reports the linear-probe diagnostic from results/slc_pretrain.

    python -m src.experiments.summarize_scvae_xc_stage1

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md.
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES
from src.experiments.summarize_scvae_improve import coverage, coverage_text, ensemble

CHOSEN = {"SEED_CHN_MM": "strict__bn_source__lr_cosine__tshuf__wd0.0001__noise0.1__sval{k}__subjnorm",
          "SEED_GER": "strict__bn_source__lr_cosine__tshuf__wd0.0001__sval{k}__subjnorm",
          "SEED_FRA": "strict__bn_source__lr_cosine__tshuf__lrate0.0005__wd0.0001__noise0.1__sval{k}__subjnorm"}
PEOPLE = {"SEED_CHN_MM": 12, "SEED_GER": 8, "SEED_FRA": 8}
PREDICTED, GO_MARGIN, GO_WINS = 56.0, 2.0, 4


def run_dirs(source, target):
    name = CHOSEN[target].format(k=3 if source == "SEED_CHN_MM" else 2)
    return (Path("results/scvae_improve/grid") / target / f"{name}__from_{source}",
            Path("results/scvae_xc/stage1") / target / f"{name}__slc__from_{source}")


def probes():
    values = {}
    for path in Path("results/slc_pretrain").glob("*/*/seed_*.json"):
        for name, acc in json.loads(path.read_text())["probe_accuracy"].items():
            values.setdefault(name, []).append(100 * acc)
    return {k: float(np.mean(v)) for k, v in values.items()}


def main():
    lines = ["# SCVAE-XC Stage 1: stimulus-locked contrastive pretraining (SLC)", "",
             "Strict accuracy %, 3-seed ensemble (mean of saved probabilities), mean over test people. Both arms: "
             "the configuration chosen per test culture (2026-10-09), seeds 12/34/56; only the encoder's "
             "initialisation differs. No target label is read before the final evaluation.", "",
             "| Train → Test | SCVAE-fixed | SCVAE-fixed + SLC | Difference | p (Wilcoxon over people) | runs |",
             "|---|---:|---:|---:|---:|---|"]
    base_avg, slc_avg, wins = [], [], 0
    for source, target in DIRECTIONS:
        base_dir, slc_dir = run_dirs(source, target)
        base, slc = ensemble(base_dir), ensemble(slc_dir)
        people = sorted(set(base) & set(slc))
        if not people:
            lines.append(f"| {NAMES[source]} → {NAMES[target]} | – | – | – | – | "
                         f"{coverage_text(coverage(slc_dir, PEOPLE[target], kind='proba'))} |")
            continue
        b = np.array([base[p][0] for p in people]); s = np.array([slc[p][0] for p in people])
        base_avg.append(b.mean()); slc_avg.append(s.mean()); wins += s.mean() > b.mean()
        p = f"{wilcoxon(s - b).pvalue:.3f}" if np.any(s != b) else "–"
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {b.mean():.1f} | {s.mean():.1f} | {s.mean() - b.mean():+.1f} "
                     f"| {p} | {coverage_text(coverage(slc_dir, PEOPLE[target], kind='proba'))} |")
    if base_avg:
        diff = np.mean(slc_avg) - np.mean(base_avg)
        go = diff >= GO_MARGIN and wins >= GO_WINS and len(base_avg) == 6
        lines += [f"| **Average ({len(base_avg)} directions)** | **{np.mean(base_avg):.1f}** | **{np.mean(slc_avg):.1f}** "
                  f"| {diff:+.1f} | | |", "",
                  f"Pre-registered: SLC ensemble {PREDICTED:.0f}%; go if >= +{GO_MARGIN:.0f} points and higher in "
                  f">= {GO_WINS} of 6. Result: {diff:+.1f} points, higher in {wins} of {len(base_avg)}. "
                  f"**{'GO' if go else 'NO-GO'}.**"]
    pr = probes()
    lines += ["", "## Linear-probe diagnostic (mean of 18 direction-seeds)", "",
              "Logistic regression (C = 0.1) on source embeddings, scored on the target (mean over people). "
              "Predicted: SLC > raw features by 2 points.", "",
              "| Features | Probe accuracy |", "|---|---:|"]
    lines += [f"| {name} | {acc:.1f} |" for name, acc in pr.items()]
    text = "\n".join(lines) + "\n"
    Path("results/tables/scvae_xc_stage1_summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
