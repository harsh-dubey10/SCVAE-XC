"""SCVAE-XC vs NormFusion v3 beyond ensemble accuracy (saved files only, no training).

1. Single model vs single model: SCVAE-XC one seed (one network) vs v3 one seed, and v3 with less ensembling
   (no_bagging: one bag per expert; no_selftrain; v1). Per person, the mean over seeds of per-seed accuracy.
2. Calibration of the window probabilities: ECE (15 equal-width bins on the top probability), Brier score,
   negative log-likelihood, mean confidence, and accuracy on each person's most confident 50% of windows
   (does confidence rank right windows first, regardless of scale?). Seed-averaged (SCVAE-XC 3, v3 5) and
   single-seed.

Unit = test person (28); paired two-sided Wilcoxon, 95% bootstrap CI over people.

    python -m src.experiments.xc_vs_v3_single_and_calibration   # -> results/tables/xc_vs_v3_extra_summary.md
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.baselines.norm_fusion import load_culture
from src.experiments.run_norm_fusion_v3 import SEEDS as V3_SEEDS
from src.experiments.run_xc_v3_combination import CACHE
from src.experiments.summarize_scvae_improve import seed_files
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, ROOT as XC_ROOT

OUT = Path("results/tables/xc_vs_v3_extra_summary.md")
V3_ROOT = Path("results/norm_fusion_v3")
RNG = np.random.default_rng(0)
SETTINGS = {"two": ("emotion_two_cultures", "eye_2src"), "one": ("emotion", "eye_1src")}


def sources_for(target, setting):
    rest = [c for c in CULTURES if c != target]
    return [rest] if setting == "two" else [[s] for s in rest]


def v3_variant(target, v3_setting, sources, variant):
    per = {}
    for path in (V3_ROOT / target / f"{v3_setting}__from_{'+'.join(sources)}").glob(f"seed_*/{variant}.json"):
        for p, v in json.loads(path.read_text())["per_subject"].items():
            per.setdefault(int(p), []).append(100 * v["accuracy"])
    return {p: float(np.mean(v)) for p, v in per.items()}


def calib(proba, y, bins=15):
    proba = np.clip(proba, 1e-7, 1)
    conf, pred = proba.max(1), proba.argmax(1)
    hit = (pred == y).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    ece = sum(abs(hit[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any())
    onehot = np.eye(proba.shape[1])[y]
    top = np.argsort(-conf)[: len(conf) // 2]
    return {"accuracy": 100 * hit.mean(), "confidence": 100 * conf.mean(), "ECE": 100 * ece,
            "Brier": float(((proba - onehot) ** 2).sum(1).mean()), "NLL": float(-np.log(proba[np.arange(len(y)), y]).mean()),
            "acc top-50% conf": 100 * hit[top].mean()}


def collect(setting):
    """(culture, person) -> dict of per-person metrics for each model; one-culture = mean over the 2 directions."""
    v3_setting, arm = SETTINGS[setting]
    rows = {}
    for t in CULTURES:
        target = load_culture(t)
        per_dir = []
        for src in sources_for(t, setting):
            name = "+".join(src)
            xc = seed_files(XC_ROOT / t / f"{arm}__from_{name}")
            v3_seeds = [np.load(CACHE / t / f"{v3_setting}__from_{name}" / f"seed_{s}.npz")["proba"] for s in V3_SEEDS]
            variants = {v: v3_variant(t, v3_setting, src, v) for v in ("fair", "no_bagging", "no_selftrain", "v1")}
            d = {}
            for person, saved in xc.items():
                m = target.subjects == person
                y = target.y[m]
                assert np.array_equal(saved[0]["labels"], y)
                xc_single = [calib(s["proba"], y) for s in saved]
                v3_single = [calib(p[m], y) for p in v3_seeds]
                d[person] = {
                    "SCVAE-XC ens": calib(np.mean([s["proba"] for s in saved], 0), y),
                    "v3 ens": calib(np.mean([p[m] for p in v3_seeds], 0), y),
                    "SCVAE-XC single": {k: np.mean([c[k] for c in xc_single]) for k in xc_single[0]},
                    "v3 single": {k: np.mean([c[k] for c in v3_single]) for k in v3_single[0]},
                    **{f"v3 {v} (single seed)": {"accuracy": variants[v][person]} for v in variants},
                }
            per_dir.append(d)
        for person in set.intersection(*(set(d) for d in per_dir)):
            rows[(t, person)] = {model: {k: float(np.mean([d[person][model][k] for d in per_dir]))
                                         for k in per_dir[0][person][model]}
                                 for model in per_dir[0][person]}
    return rows


def paired(rows, a, b, metric):
    x = np.array([r[a][metric] for r in rows.values()])
    y = np.array([r[b][metric] for r in rows.values()])
    d = x - y
    boot = RNG.choice(d, (10000, len(d))).mean(1)
    p = wilcoxon(d).pvalue if np.any(d) else 1.0
    return x.mean(), y.mean(), d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), p


def main():
    lines = ["# SCVAE-XC vs NormFusion v3: single models and calibration", "",
             "Saved probabilities and per-seed records only. Unit = test person (28); one-culture scores averaged "
             "over the two directions into each person's culture. Paired Wilcoxon, 95% bootstrap CI. "
             "Generated by `src/experiments/xc_vs_v3_single_and_calibration.py`.", ""]
    for setting, title in (("two", "Two cultures → the third"), ("one", "One culture → another")):
        rows = collect(setting)
        lines += [f"## {title}", "", "### 1. Accuracy: single model vs single model", "",
                  "| A | B | A | B | Δ | 95% CI | p |", "|---|---|---:|---:|---:|---|---:|"]
        for a, b in [("SCVAE-XC single", "v3 single"),
                     ("SCVAE-XC single", "v3 no_bagging (single seed)"),
                     ("SCVAE-XC single", "v3 no_selftrain (single seed)"),
                     ("SCVAE-XC single", "v3 v1 (single seed)"),
                     ("SCVAE-XC ens", "v3 ens")]:
            xa, xb, d, lo, hi, p = paired(rows, a, b, "accuracy")
            lines.append(f"| {a} | {b} | {xa:.1f} | {xb:.1f} | {d:+.1f} | {lo:+.1f} to {hi:+.1f} | {p:.2g} |")
        lines += ["", "### 2. Calibration (lower is better for ECE, Brier, NLL)", "",
                  "| Metric | SCVAE-XC ens | v3 ens | p | SCVAE-XC single | v3 single | p |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for metric in ("accuracy", "confidence", "ECE", "Brier", "NLL", "acc top-50% conf"):
            e = paired(rows, "SCVAE-XC ens", "v3 ens", metric)
            s = paired(rows, "SCVAE-XC single", "v3 single", metric)
            f = ".3f" if metric in ("Brier", "NLL") else ".1f"
            lines.append(f"| {metric} | {e[0]:{f}} | {e[1]:{f}} | {e[5]:.2g} | {s[0]:{f}} | {s[1]:{f}} | {s[5]:.2g} |")
        lines.append("")
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
