"""SCVAE-XC + NormFusion v3: equal-weight mean of the two models' seed-averaged probabilities (strict, untuned).

v3's fair probabilities are re-generated with fair_probas (5 seeds) and cached under results/xc_v3_combination/;
each seed must reproduce the stored fair.json accuracy within 0.5 points. SCVAE-XC: the step 4 probability files.

    python -m src.experiments.run_xc_v3_combination --task i --jobs 16   # cache v3 for ORDER[i] (cluster)
    python -m src.experiments.run_xc_v3_combination                       # combine and summarise

Design: docs/superpowers/specs/2026-11-04-xc-v3-combination-design.md.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.baselines.norm_fusion import load_culture
from src.experiments.run_norm_fusion_v3 import SEEDS, SETTINGS, fair_probas
from src.experiments.summarize_dcca_am import NAMES
from src.experiments.summarize_scvae_improve import seed_files
from src.experiments.summarize_scvae_xc_stage2 import ROOT as XC_ROOT

V3_ROOT, CACHE = Path("results/norm_fusion_v3"), Path("results/xc_v3_combination")
OUT = Path("results/tables/xc_v3_combination_summary.md")
PREDICTED = {"combined": 71.0, "v3 seed-average": 70.0}


def v3_seed_average(sources, target, target_name, setting, n_jobs):
    """Mean over v3's 5 seeds of its fair per-window probabilities on the whole target culture."""
    name = "+".join(sources)
    probas = []
    for seed in SEEDS:
        cache = CACHE / target_name / f"{setting}__from_{name}" / f"seed_{seed}.npz"
        if cache.exists():
            proba = np.load(cache)["proba"]
        else:
            proba, _ = fair_probas([load_culture(c) for c in sources], target, seed, n_jobs)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, proba=proba.astype(np.float32))
        stored = json.loads((V3_ROOT / target_name / f"{setting}__from_{name}" / f"seed_{seed}" / "fair.json").read_text())
        mine = 100 * np.mean([(proba.argmax(1) == target.y)[target.subjects == int(s)].mean() for s in stored["per_subject"]])
        theirs = 100 * np.mean([v["accuracy"] for v in stored["per_subject"].values()])
        assert abs(mine - theirs) <= 0.5, f"{name} -> {target_name} seed {seed}: {mine:.2f} vs stored {theirs:.2f}"
        probas.append(proba)
    return np.mean(probas, axis=0)


def xc_seed_average(arm, sources, target, target_name):
    """Person -> SCVAE-XC 3-seed mean probabilities (rows in load_culture order for that person)."""
    out = {}
    for person, saved in seed_files(XC_ROOT / target_name / f"{arm}__from_{'+'.join(sources)}").items():
        assert np.array_equal(saved[0]["labels"], target.y[target.subjects == person])
        out[person] = np.mean([s["proba"] for s in saved], axis=0)
    return out


def score(sources, target_name, setting, arm, n_jobs):
    target = load_culture(target_name)
    v3 = v3_seed_average(sources, target, target_name, setting, n_jobs)
    xc = xc_seed_average(arm, sources, target, target_name)
    rows = {}
    for person, p_xc in xc.items():
        mask = target.subjects == person
        y, p_v3 = target.y[mask], v3[mask]
        rows[person] = [100 * np.mean(p.argmax(1) == y) for p in (p_xc, p_v3, (p_xc + p_v3) / 2)]
    return rows


# (sources, target, setting); two-culture settings first: they decide.
ORDER = sorted([(tuple(s), t, st) for s, t, st in SETTINGS if st in ("emotion", "emotion_two_cultures")],
               key=lambda k: k[2] != "emotion_two_cultures")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--task", type=int, help="Only cache v3's probabilities for ORDER[task] (0..8).")
    args = parser.parse_args()
    if args.task is not None:
        sources, target_name, setting = ORDER[args.task]
        v3_seed_average(sources, load_culture(target_name), target_name, setting, args.jobs)
        print(f"cached {setting} {'+'.join(sources)} -> {target_name}", flush=True)
        return
    lines = ["# SCVAE-XC + NormFusion v3 combination", "",
             "Strict accuracy %, mean over test people. SCVAE-XC: 3-seed probability mean; v3: 5-seed probability mean "
             "(fair model); combined: equal-weight mean of the two. p: Wilcoxon over people, combined vs the better "
             "of the two. Design: docs/superpowers/specs/2026-11-04-xc-v3-combination-design.md.", ""]
    verdict_rows = []
    for title, setting, arm in (("Two training cultures → the third", "emotion_two_cultures", "eye_2src"),
                                ("One training culture", "emotion", "eye_1src")):
        lines += [f"## {title}", "", "| Train → Test | SCVAE-XC | v3 seed-average | combined | p |", "|---|---:|---:|---:|---:|"]
        means = []
        for sources, target, st in ORDER:
            if st != setting:
                continue
            rows = score(sources, target, setting, arm, args.jobs)
            table = np.array(list(rows.values()))
            m = table.mean(0)
            best = table[:, 0] if m[0] >= m[1] else table[:, 1]
            p = wilcoxon(table[:, 2] - best).pvalue if np.any(table[:, 2] != best) else float("nan")
            means.append(m)
            lines.append(f"| {' + '.join(NAMES[c] for c in sources)} → {NAMES[target]} | {m[0]:.1f} | {m[1]:.1f} | "
                         f"{m[2]:.1f} | {p:.3f} |")
            if setting == "emotion_two_cultures":
                verdict_rows.append(m)
        a = np.mean(means, 0)
        lines += [f"| **Average** | **{a[0]:.1f}** | **{a[1]:.1f}** | **{a[2]:.1f}** | |", ""]
    v = np.array(verdict_rows)
    gain = v[:, 2].mean() - max(v[:, 0].mean(), v[:, 1].mean())
    wins = int(np.sum((v[:, 2] > v[:, 0]) & (v[:, 2] > v[:, 1])))
    go = gain >= 1.0 and wins >= 2
    lines += [f"Predicted (two cultures): {', '.join(f'{k} {x}' for k, x in PREDICTED.items())}.", "",
              "## Decision (pre-registered, two cultures)", "",
              f"- combined − best single: {gain:+.1f} (needs ≥ +1.0); combined beats both in {wins} of 3 (needs ≥ 2).",
              f"- **{'GO: combined is the headline fair number' if go else 'NO-GO: v3 remains the best fair model'}.**"]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
