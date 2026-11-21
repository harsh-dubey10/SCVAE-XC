"""Significance tests for the headline comparisons in the deck (existing results only, no training).

Unit of analysis: the test person (12 Chinese + 8 German + 8 French = 28). For one-culture settings each person's
score is the mean of the two directions into their culture, so every person counts once. For each comparison:
mean difference, 95% bootstrap CI over people, people better / worse, two-sided Wilcoxon signed-rank p, and
Holm-adjusted p across the whole family. Per-film results: one-sample Wilcoxon against 50% (chance).

Sources (all strict / fair numbers):
  SCVAE-XC        results/scvae_xc/stage2 (eye_1src, eye_2src, eeg_2src), 3-seed probability mean
  SCVAE-Net       results/scvae_improve/grid, SCVAE-fixed, 3-seed probability mean
  NormFusion v3   results/norm_fusion_v3/*/seed_*/fair.json, mean over 5 seeds of per-person accuracy
  DCCA-AM         results/dcca_am, architecture 0, last epoch, mean over 3 seeds

    python -m src.experiments.significance_tests   # -> results/tables/significance_summary.md
"""

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, fair_score, load_runs
from src.experiments.summarize_scvae_improve import ensemble
from src.experiments.summarize_scvae_xc_stage1 import run_dirs
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, ROOT as XC_ROOT

OUT = Path("results/tables/significance_summary.md")
V3_ROOT = Path("results/norm_fusion_v3")
RNG = np.random.default_rng(0)


def others(target):
    return [c for c in CULTURES if c != target]


def xc(target, arm, sources):
    return {p: v[0] for p, v in ensemble(XC_ROOT / target / f"{arm}__from_{'+'.join(sources)}").items()}


def scvae_net(source, target):
    return {p: v[0] for p, v in ensemble(run_dirs(source, target)[0]).items()}


def v3(target, setting, sources):
    per = {}
    for path in sorted((V3_ROOT / target / f"{setting}__from_{'+'.join(sources)}").glob("seed_*/fair.json")):
        for p, v in json.loads(path.read_text())["per_subject"].items():
            per.setdefault(int(p), []).append(100 * v["accuracy"])
    return {p: float(np.mean(v)) for p, v in per.items()}


DCCA_RUNS = None


def dcca(source, target):
    global DCCA_RUNS
    DCCA_RUNS = DCCA_RUNS or load_runs(Path("results/dcca_am"))
    seed_runs = DCCA_RUNS[(target, "emotion", source)]
    subjects = next(iter(seed_runs.values()))[0]["confusion"][0]
    return {int(s): 100 * float(np.mean([fair_score(a, s, False) for a in seed_runs.values()])) for s in subjects}


def one_culture(score):
    """(culture, person) -> mean over the two one-culture directions into that culture."""
    out = {}
    for t in CULTURES:
        per_dir = [score(s, t) for s in others(t)]
        for p in set.intersection(*(set(d) for d in per_dir)):
            out[(t, p)] = float(np.mean([d[p] for d in per_dir]))
    return out


def two_culture(score):
    return {(t, p): v for t in CULTURES for p, v in score(t).items()}


def boot_ci(x, n=10000):
    means = RNG.choice(x, size=(n, len(x))).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def compare(a, b):
    keys = sorted(set(a) & set(b))
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    d = x - y
    lo, hi = boot_ci(d)
    p = wilcoxon(d).pvalue if np.any(d) else 1.0
    per_culture = {t: float(np.mean([a[k] - b[k] for k in keys if k[0] == t])) for t in CULTURES}
    return dict(n=len(keys), a=x.mean(), b=y.mean(), d=d.mean(), lo=lo, hi=hi, better=int((d > 0).sum()),
                worse=int((d < 0).sum()), p=p, per_culture=per_culture)


def holm(ps):
    order = np.argsort(ps)
    adj, running = np.empty(len(ps)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def per_film():
    """Person-level 3-day pos/neg film accuracy for SCVAE-XC and v3 (two cultures), via per_film_accuracy."""
    from src.baselines.norm_fusion import load_culture
    from src.experiments.per_film_accuracy import person_scores, v3_proba
    from src.experiments.run_xc_v3_combination import xc_seed_average
    out = {"SCVAE-XC": {}, "NormFusion v3": {}}
    for t in CULTURES:
        target = load_culture(t)
        probs = {"SCVAE-XC": xc_seed_average("eye_2src", others(t), target, t),
                 "NormFusion v3": v3_proba(others(t), t, "emotion_two_cultures")}
        for person in np.unique(target.subjects):
            m = target.subjects == person
            for name, pr in probs.items():
                proba = pr[person] if isinstance(pr, dict) else pr[m]
                s, _ = person_scores(proba, target.y[m], target.clips[m], target.recordings[m])
                out[name][(t, int(person))] = float(s[2])
    return out


def main():
    xc2 = two_culture(lambda t: xc(t, "eye_2src", others(t)))
    eeg2 = two_culture(lambda t: xc(t, "eeg_2src", others(t)))
    v3_2 = two_culture(lambda t: v3(t, "emotion_two_cultures", others(t)))
    xc1 = one_culture(lambda s, t: xc(t, "eye_1src", [s]))
    net1 = one_culture(scvae_net)
    v3_1 = one_culture(lambda s, t: v3(t, "emotion", [s]))
    dcca1 = one_culture(dcca)

    rows = [
        ("Two cultures", "SCVAE-XC vs NormFusion v3", xc2, v3_2, "slides 6-8: v3 is ahead by 2.5"),
        ("Two cultures", "SCVAE-XC vs SCVAE-Net (one culture)", xc2, net1, "slide 6: +14.0"),
        ("Two cultures", "SCVAE-XC: EEG + eye vs EEG only", xc2, eeg2, "eye branch, two cultures"),
        ("Two vs one", "SCVAE-XC: two cultures vs one", xc2, xc1, "slide 4: +6.8"),
        ("One culture", "SCVAE-XC vs SCVAE-Net", xc1, net1, "slide 4: eye branch +7.2"),
        ("One culture", "SCVAE-XC vs DCCA-AM (fair)", xc1, dcca1, "slide 8: +16.7"),
        ("One culture", "NormFusion v3 vs DCCA-AM (fair)", v3_1, dcca1, "slide 8"),
        ("One culture", "NormFusion v3 vs SCVAE-XC", v3_1, xc1, "slide 8: 65.1 vs 60.5"),
    ]
    results = [compare(a, b) for _, _, a, b, _ in rows]
    adj = holm(np.array([r["p"] for r in results]))

    lines = ["# Significance of the headline comparisons", "",
             "Existing strict results, no new training. Unit = test person (28: 12 Chinese, 8 German, 8 French); "
             "one-culture scores are each person's mean over the two directions into their culture. "
             "Δ = A − B in accuracy points, 95% bootstrap CI over people, two-sided Wilcoxon signed-rank, "
             "Holm correction across the 8 comparisons. Generated by `src/experiments/significance_tests.py`.", "",
             "| Setting | Comparison (A vs B) | A | B | Δ | 95% CI | A better / worse | p | p (Holm) | "
             "Δ Chinese / German / French | Deck claim |",
             "|---|---|---:|---:|---:|---|---|---:|---:|---|---|"]
    for (setting, label, _, _, claim), r, pa in zip(rows, results, adj):
        pc = " / ".join(f"{r['per_culture'][t]:+.1f}" for t in CULTURES)
        lines.append(f"| {setting} | {label} | {r['a']:.1f} | {r['b']:.1f} | {r['d']:+.1f} | "
                     f"{r['lo']:+.1f} to {r['hi']:+.1f} | {r['better']} / {r['worse']} of {r['n']} | "
                     f"{r['p']:.2g} | {pa:.2g} | {pc} | {claim} |")

    lines += ["", "## Per-film positive vs negative (two cultures, 3 days pooled)", "",
              "| Model | mean % | 95% CI | p vs 50% (Wilcoxon) | people ≤ 50% |", "|---|---:|---|---:|---:|"]
    films = per_film()
    for name, scores in films.items():
        x = np.array(list(scores.values()))
        lo, hi = boot_ci(x)
        lines.append(f"| {name} | {x.mean():.1f} | {lo:.1f} to {hi:.1f} | {wilcoxon(x - 50).pvalue:.2g} | "
                     f"{int((x <= 50).sum())} of {len(x)} |")
    r = compare(films["NormFusion v3"], films["SCVAE-XC"])
    lines += ["", f"v3 − SCVAE-XC per film: {r['d']:+.1f} (95% CI {r['lo']:+.1f} to {r['hi']:+.1f}), "
              f"Wilcoxon p = {r['p']:.2g}. Each person contributes 10-14 film decisions, so per-person scores are "
              "coarse (steps of 7-10 points).", ""]
    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
