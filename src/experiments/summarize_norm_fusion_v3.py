"""Summarise NormFusion v3 against v1 (same machine), v2, DCCA-AM and the pre-registered prediction.

Tables: the 6 one-to-one directions with the prediction and the success criterion; the two-culture settings;
new films; ablations; the choices per seed; clip-level accuracy (separate); and the check of the same-machine
v1 against the saved laptop v1 (seeds 12, 34, 56; differences above 2 points are flagged).

    python -m src.experiments.summarize_norm_fusion_v3   # -> results/tables/norm_fusion_v3_summary.md
"""

import argparse
from pathlib import Path

import numpy as np

from src.experiments.run_norm_fusion_v3 import SEEDS, VARIANTS
from src.experiments.summarize_dcca_am import DIRECTIONS, load_runs
from src.experiments.summarize_norm_fusion import dcca_per_subject, fmt, load_nf, nf_per_subject
from src.experiments.summarize_norm_fusion_v2 import label, mean_of, paired_p, v2_paper_rules

# Pre-registered in docs/superpowers/specs/2026-10-02-norm-fusion-v3-design.md, before any v3 run.
PREDICTED = {("SEED_CHN_MM", "SEED_GER"): 54, ("SEED_CHN_MM", "SEED_FRA"): 60, ("SEED_GER", "SEED_CHN_MM"): 64,
             ("SEED_GER", "SEED_FRA"): 65, ("SEED_FRA", "SEED_CHN_MM"): 76, ("SEED_FRA", "SEED_GER"): 70}
FLAG_POINTS = 2.0


def success(ours, theirs):
    """Spec criterion: higher on the mean over directions and in at least 4 of the 6. Returns (met, wins)."""
    wins = sum(ours[d] > theirs[d] for d in ours)
    return bool(np.mean(list(ours.values())) > np.mean(list(theirs.values())) and wins >= 4), wins


def complete(seed_records):
    """Every pre-registered seed has both the fair and the v1 record."""
    return all(s in seed_records and {"fair", "v1"} <= set(seed_records[s]) for s in SEEDS)


def v1_check(laptop, cluster):
    """|same-machine v1 - laptop v1| in a direction's mean over people, over the seeds both have; flagged
    above FLAG_POINTS, or when no seed is in common. Returns (difference as a fraction or None, flagged)."""
    seeds = sorted({s for s, r in laptop.items() if "fair" in r} & {s for s, r in cluster.items() if "v1" in r})
    if not seeds:
        return None, True
    mean = lambda records, variant: np.mean([np.mean([p["accuracy"] for p in records[s][variant]["per_subject"].values()])
                                             for s in seeds])
    diff = float(abs(mean(cluster, "v1") - mean(laptop, "fair")))
    return diff, diff * 100 > FLAG_POINTS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/norm_fusion_v3"))
    parser.add_argument("--v1-root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--v2-root", type=Path, default=Path("results/norm_fusion_v2"))
    parser.add_argument("--dcca-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/norm_fusion_v3_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    v3, laptop_v1, v2, dcca = load_nf(args.root), load_nf(args.v1_root), load_nf(args.v2_root), load_runs(args.dcca_root)
    v2_fair = lambda key: fmt(nf_per_subject(v2[key], "fair"), rng) if key in v2 else "–"

    lines = ["# NormFusion v3 across cultures (EEG + eye)", "",
             "v3 is a post-hoc model with a pre-registered design and prediction "
             "(docs/superpowers/specs/2026-10-02-norm-fusion-v3-design.md); v1 is the primary, pre-specified "
             "model. v1 here is re-run on the same machine as v3.", "",
             "Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
             "p: paired Wilcoxon over test people, v3 fair vs v1.", "",
             "| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v3 predicted | v2 fair | DCCA-AM fair "
             "| v3 paper's rules |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ours, theirs = {}, {}
    for source, target in DIRECTIONS:
        key = (target, "emotion", source)
        if key not in v3:
            continue
        mine, base = nf_per_subject(v3[key], "fair"), nf_per_subject(v3[key], "v1")
        ours[(source, target)], theirs[(source, target)] = mean_of(mine), mean_of(base)
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(mine, rng)} | "
                     f"{fmt(nf_per_subject(v3[key], 'fair', 'macro_f1'), rng)} | {paired_p(mine, base)} | "
                     f"{PREDICTED[(source, target)]} | {v2_fair(key)} | "
                     f"{fmt(dcca_per_subject(dcca, key, 'fair'), rng)} | {fmt(v2_paper_rules(v3[key]), rng)} |")
    if ours:
        lines += ["", f"Average over {len(ours)} directions: v3 fair {100 * np.mean(list(ours.values())):.1f}% "
                      f"(predicted {np.mean([PREDICTED[d] for d in ours]):.1f}%), v1 "
                      f"{100 * np.mean(list(theirs.values())):.1f}%."]
        if len(ours) == 6 and all(complete(v3[(t, "emotion", s)]) for s, t in ours):
            met, wins = success(ours, theirs)
            lines.append(f"Success criterion (higher than v1 on average and in at least 4 of 6): "
                         f"{'met' if met else 'not met'} (v3 higher in {wins} of 6).")
        else:
            lines.append("Success criterion: pending (not all 6 directions have all 5 seeds).")

    lines += ["", "## Two cultures → the third (predicted 70–75%)", "",
              "| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v2 fair | chosen memberships |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for key in sorted(k for k in v3 if k[1] == "emotion_two_cultures"):
        mine, base = nf_per_subject(v3[key], "fair"), nf_per_subject(v3[key], "v1")
        chosen = ", ".join(sorted({"+".join(r["fair"]["config"]["membership"]) for r in v3[key].values()}))
        lines.append(f"| {label(key[2])} → {label(key[0])} | {fmt(base, rng)} | {fmt(mine, rng)} | "
                     f"{fmt(nf_per_subject(v3[key], 'fair', 'macro_f1'), rng)} | {paired_p(mine, base)} | "
                     f"{v2_fair(key)} | {chosen} |")

    lines += ["", "## New films (German ↔ French, shared films removed from training)", "",
              "| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v2 fair | DCCA-AM |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for source, target in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")):
        key = (target, "emotion_newfilms", source)
        if key not in v3:
            continue
        mine, base = nf_per_subject(v3[key], "fair"), nf_per_subject(v3[key], "v1")
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(mine, rng)} | "
                     f"{fmt(nf_per_subject(v3[key], 'fair', 'macro_f1'), rng)} | {paired_p(mine, base)} | "
                     f"{v2_fair(key)} | {fmt(dcca_per_subject(dcca, key, 'fair'), rng)} |")

    ablation = ["", "## Ablations (fair accuracy %, mean over test people)", "",
                "| Setting | Train → Test | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---:|" * len(VARIANTS)]
    choices = ["", "## Choices per seed (membership; C for EEG / eye / early)", "",
               "| Setting | Train → Test | Seed | Membership | C |", "|---|---|---:|---|---|"]
    clips = ["", "## Clip-level accuracy % (majority vote over each clip's windows; uses the known clip boundaries "
                 "at test time — not comparable with any window-level number)", "",
             "| Setting | Train → Test | v1 | v3 fair |", "|---|---|---:|---:|"]
    check = ["", "## Same-machine v1 vs saved laptop v1 (seeds 12, 34, 56)", "",
             "| Setting | Train → Test | difference (points) | flag |", "|---|---|---:|---|"]
    for key in sorted(v3):
        target, setting, source = key
        name = f"| {setting} | {label(source)} → {label(target)} |"
        ablation.append(f"{name} " + " | ".join(f"{100 * mean_of(nf_per_subject(v3[key], v)):.1f}" for v in VARIANTS)
                        + " |")
        for seed, records in sorted(v3[key].items()):
            cfg = records["fair"]["config"]
            choices.append(f"{name} {seed} | {'+'.join(cfg['membership'])} | "
                           f"{cfg['cs']['eeg']:g} / {cfg['cs']['eye']:g} / {cfg['cs']['early']:g} |")
        clips.append(f"{name} {100 * mean_of(nf_per_subject(v3[key], 'v1', 'clip_accuracy')):.1f} | "
                     f"{100 * mean_of(nf_per_subject(v3[key], 'fair', 'clip_accuracy')):.1f} |")
        if key in laptop_v1:
            diff, flagged = v1_check(laptop_v1[key], v3[key])
            shown = "no common seeds" if diff is None else f"{100 * diff:.2f}"
            check.append(f"{name} {shown} | {'CHECK' if flagged else 'ok'} |")
    text = "\n".join(lines + ablation + choices + clips + check) + "\n"
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
