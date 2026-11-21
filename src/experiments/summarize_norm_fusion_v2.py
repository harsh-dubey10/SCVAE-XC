"""Summarise NormFusion v2 against v1 and DCCA-AM.

Tables: the 6 one-to-one directions (v1 fair, v2 fair, macro-F1, paired Wilcoxon v2 vs v1, v2 and DCCA-AM
under the paper's rules, DCCA-AM fair); the two-culture settings (v2 vs v1 only); new films; ablations;
the fair choices; clip-level accuracy (separate, never compared with window-level numbers); and the check
that v2's "v1" variant reproduces the saved v1 results.

    python -m src.experiments.summarize_norm_fusion_v2   # -> results/tables/norm_fusion_v2_summary.md
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.run_norm_fusion_v2 import VARIANTS
from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, load_runs
from src.experiments.summarize_norm_fusion import dcca_per_subject, fmt, load_nf, nf_per_subject


def label(culture):
    return " + ".join(NAMES[c] for c in culture.split("+"))


def v2_paper_rules(seed_records):
    """Best fair-grid candidate per test person (or the fair choice with self-training), seeds averaged."""
    per_seed = [{s: max(v) for s, v in r["fair"]["candidates"]["accuracy"].items()} for r in seed_records.values()]
    return {s: float(np.mean([p[s] for p in per_seed])) for s in sorted(per_seed[0], key=int)}


def paired_p(a, b):
    diff = [a[s] - b[s] for s in a]
    return f"{wilcoxon(diff).pvalue:.3f}" if any(diff) else "1"


def v1_reproduction(v1_runs, v2_runs):
    """v2's "v1" variant against the saved v1 fair runs, on every (setting, seed) both have."""
    compared, matched, max_diff = 0, 0, 0.0
    for key, seeds in v2_runs.items():
        for seed, records in seeds.items():
            saved = v1_runs.get(key, {}).get(seed, {}).get("fair")
            if saved is None or "v1" not in records:
                continue
            mine = records["v1"]
            compared += 1
            matched += [mine["config"][1], mine["config"][2], mine["config"][3][0]] == saved["config"]
            max_diff = max([max_diff] + [abs(mine["per_subject"][s]["accuracy"] - v["accuracy"])
                                         for s, v in saved["per_subject"].items()])
    return {"compared": compared, "configs_matched": matched, "max_diff": max_diff}


def mean_of(values):
    return float(np.mean(list(values.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/norm_fusion_v2"))
    parser.add_argument("--v1-root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--dcca-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/norm_fusion_v2_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    v2, v1, dcca = load_nf(args.root), load_nf(args.v1_root), load_runs(args.dcca_root)

    lines = ["# NormFusion v2 vs v1 vs DCCA-AM across cultures (EEG + eye)", "",
             "v2 is a post-hoc model: its ideas came from v1's results and from an analysis of emotion signatures "
             "that used the labels of all three cultures. v1 is the primary, pre-specified result.", "",
             "Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
             "p: paired Wilcoxon over test people, v2 fair vs v1 fair.", "",
             "| Train → Test | v1 fair | v2 fair | v2 fair macro-F1 | p (v2 vs v1) | v2 paper's rules "
             "| DCCA-AM fair | DCCA-AM paper's rules |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    direction_means = []
    for source, target in DIRECTIONS:
        key = (target, "emotion", source)
        if key not in v2:
            continue
        ours = nf_per_subject(v2[key], "fair")
        base = nf_per_subject(v1[key], "fair") if key in v1 else None
        theirs = dcca_per_subject(dcca, key, "fair")
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{fmt(nf_per_subject(v2[key], 'fair', 'macro_f1'), rng)} | "
                     f"{paired_p(ours, base) if base else '–'} | {fmt(v2_paper_rules(v2[key]), rng)} | "
                     f"{fmt(theirs, rng)} | {fmt(dcca_per_subject(dcca, key, 'paper'), rng)} |")
        direction_means.append((mean_of(ours), mean_of(base) if base else None, mean_of(theirs) if theirs else None))
    if direction_means:
        n = len(direction_means)
        line = f"Average over {n} directions: v2 fair {100 * np.mean([m[0] for m in direction_means]):.1f}%"
        if all(m[1] is not None for m in direction_means):
            line += (f", v1 fair {100 * np.mean([m[1] for m in direction_means]):.1f}%; v2 higher than v1 in "
                     f"{sum(m[0] > m[1] for m in direction_means)} of {n}")
        if all(m[2] is not None for m in direction_means):
            line += f"; DCCA-AM fair {100 * np.mean([m[2] for m in direction_means]):.1f}%"
        lines += ["", line + "."]

    lines += ["", "## Two cultures → the third (compared only with v1 in the same setting)", "",
              "| Train → Test | v1 | v2 fair | v2 fair macro-F1 | p (v2 vs v1) |", "|---|---:|---:|---:|---:|"]
    for key in sorted(k for k in v2 if k[1] == "emotion_two_cultures"):
        ours, base = nf_per_subject(v2[key], "fair"), nf_per_subject(v2[key], "v1")
        lines.append(f"| {label(key[2])} → {label(key[0])} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{fmt(nf_per_subject(v2[key], 'fair', 'macro_f1'), rng)} | {paired_p(ours, base)} |")

    lines += ["", "## New films (German ↔ French, shared films removed from training), fair accuracy %", "",
              "| Train → Test | v1 | v2 | v2 macro-F1 | p (v2 vs v1) | DCCA-AM |", "|---|---:|---:|---:|---:|---:|"]
    for source, target in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")):
        key = (target, "emotion_newfilms", source)
        if key not in v2:
            continue
        ours, base = nf_per_subject(v2[key], "fair"), nf_per_subject(v2[key], "v1")
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{fmt(nf_per_subject(v2[key], 'fair', 'macro_f1'), rng)} | {paired_p(ours, base)} | "
                     f"{fmt(dcca_per_subject(dcca, key, 'fair'), rng)} |")

    lines += ["", "## Ablations (fair accuracy %, mean over test people)", "",
              "| Setting | Train → Test | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---:|" * len(VARIANTS)]
    choices = ["", "## Fair choices (fusion, C_EEG, C_eye, weights neg/neu/pos, k) per seed", "",
               "| Setting | Train → Test | Seed | Choice | CV score |", "|---|---|---:|---|---:|"]
    clips = ["", "## Clip-level accuracy % (majority vote over each clip's windows; uses the known clip boundaries "
                 "at test time — not comparable with any window-level number)", "",
             "| Setting | Train → Test | v1 | v2 fair |", "|---|---|---:|---:|"]
    for key in sorted(v2):
        target, setting, source = key
        present = [v for v in VARIANTS if all(v in r for r in v2[key].values())]
        lines.append(f"| {setting} | {label(source)} → {label(target)} | " + " | ".join(
            f"{100 * mean_of(nf_per_subject(v2[key], v)):.1f}" if v in present else "–" for v in VARIANTS) + " |")
        for seed, records in sorted(v2[key].items()):
            fusion, c_eeg, c_eye, weights, k, _ = records["fair"]["config"]
            shown = f"{fusion}, {c_eeg}, {c_eye}, {'/'.join(f'{w:g}' for w in weights) if weights else '–'}, {k}"
            choices.append(f"| {setting} | {label(source)} → {label(target)} | {seed} | {shown} | "
                           f"{100 * records['fair']['cv_score']:.1f} |")
        clips.append(f"| {setting} | {label(source)} → {label(target)} | "
                     f"{100 * mean_of(nf_per_subject(v2[key], 'v1', 'clip_accuracy')):.1f} | "
                     f"{100 * mean_of(nf_per_subject(v2[key], 'fair', 'clip_accuracy')):.1f} |")

    check = v1_reproduction(v1, v2)
    lines += choices + clips + ["", "## v1 reproduction check", "",
                                f"v2's v1 variant vs the saved v1 results: {check['compared']} runs compared, "
                                f"{check['configs_matched']} with the same choice, largest per-person accuracy "
                                f"difference {check['max_diff']:.2e} "
                                + ("(reproduced)." if check["compared"] and check["max_diff"] < 1e-9
                                   and check["configs_matched"] == check["compared"] else "(NOT reproduced).")]
    text = "\n".join(lines) + "\n"
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
