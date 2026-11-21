"""Culture-MMD ablation: SCVAE-XC (EEG + eye, two training cultures) with and without the MMD between the two
training cultures, and both against one training culture. Strict 3-seed probability-mean ensembles throughout.

Runs: cluster/scvae_xc_ablation_cmmd.txt (identical to stage2 eye_2src except --culture-mmd-weight 0).
Unit: the test person (28). One-culture score: mean over the two directions into the person's culture.

    python -m src.experiments.ablation_culture_mmd   # -> results/tables/ablation_culture_mmd.md
"""

from pathlib import Path

import numpy as np

from src.experiments.significance_tests import compare, holm, one_culture, others, two_culture, xc
from src.experiments.summarize_scvae_improve import ensemble
from src.experiments.summarize_scvae_xc_stage2 import CULTURES

ABLATION_ROOT = Path("results/scvae_xc/ablation_no_culture_mmd")
OUT = Path("results/tables/ablation_culture_mmd.md")
NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}


def no_cmmd(target):
    run = ABLATION_ROOT / target / f"eye_2src__from_{'+'.join(others(target))}"
    return {p: v[0] for p, v in ensemble(run).items()}


def main():
    full = two_culture(lambda t: xc(t, "eye_2src", others(t)))
    ablated = two_culture(no_cmmd)
    single = one_culture(lambda s, t: xc(t, "eye_1src", [s]))
    rows = [("Two cultures: full vs no culture MMD", full, ablated),
            ("Two cultures without culture MMD vs one culture", ablated, single),
            ("Two cultures full vs one culture", full, single)]
    results = [compare(a, b) for _, a, b in rows]
    adjusted = holm([r["p"] for r in results])

    def culture_mean(score):  # table convention: mean over cultures of per-person means
        return np.mean([np.mean([v for (t, _), v in score.items() if t == c]) for c in CULTURES])

    lines = ["# Culture-MMD ablation (SCVAE-XC, EEG + eye)", "",
             "Strict accuracy %, 3-seed probability mean. Per-culture columns: mean over that culture's people.", "",
             "| Variant | " + " | ".join(NAMES[c] for c in CULTURES) + " | Mean of cultures | Mean of 28 people |",
             "|---|" + "---:|" * (len(CULTURES) + 2)]
    for name, score in [("Two cultures, full (culture MMD 0.1)", full), ("Two cultures, no culture MMD", ablated),
                        ("One culture (mean of two directions)", single)]:
        per = [np.mean([v for (t, _), v in score.items() if t == c]) for c in CULTURES]
        lines.append(f"| {name} | " + " | ".join(f"{x:.1f}" for x in per)
                     + f" | {culture_mean(score):.1f} | {np.mean(list(score.values())):.1f} |")
    lines += ["", "| Comparison (A vs B) | n | Δ (28 people) | 95% CI | A better / worse | p | p (Holm, 3) |",
              "|---|---:|---:|---|---|---:|---:|"]
    for (name, _, _), r, p_adj in zip(rows, results, adjusted):
        lines.append(f"| {name} | {r['n']} | {r['d']:+.1f} | {r['lo']:+.1f} to {r['hi']:+.1f} | "
                     f"{r['better']} / {r['worse']} | {r['p']:.2g} | {p_adj:.2g} |")
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
