"""Summarise NormFusion and compare it with DCCA-AM under the same evaluation.

Main table per direction: the paper's reported DCCA-AM, DCCA-AM and NormFusion
under the paper's rules, DCCA-AM and NormFusion fair, a paired Wilcoxon test of
the two fair columns over test people, and the best simple normalised model.
Then the ablations (fair accuracy of each variant) and the new-films setting.
DCCA-AM columns show "–" until a direction's DCCA-AM runs are complete
(3 seeds x 21 architectures x 100 epochs).

    python -m src.experiments.summarize_norm_fusion   # -> results/tables/norm_fusion_summary.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.run_norm_fusion import VARIANTS
from src.experiments.summarize_dcca_am import (DIRECTIONS, NAMES, PAPER_DCCA, accuracy, best_simple, completeness,
                                               fair_score, load_runs, mean_ci, paper_score)


def load_nf(root):
    runs = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/*__from_*/seed_*/*.json"):
        record = json.loads(path.read_text())
        runs[(record["target"], record["setting"], record["source"])][record["seed"]][record["variant"]] = record
    return runs


def nf_per_subject(seed_records, variant, metric="accuracy"):
    subjects = sorted(next(iter(seed_records.values()))[variant]["per_subject"], key=int)
    return {s: float(np.mean([r[variant]["per_subject"][s][metric] for r in seed_records.values()]))
            for s in subjects}


def nf_paper_rules(seed_records):
    per_seed = [{s: max(accuracy(table[s], False) for table in r["fair"]["candidates"].values())
                 for s in r["fair"]["per_subject"]} for r in seed_records.values()]
    return {s: float(np.mean([p[s] for p in per_seed])) for s in sorted(per_seed[0], key=int)}


def dcca_per_subject(dcca_runs, key, mode, n_archs=21, epochs=100, seeds=3):
    """Seed-averaged DCCA-AM score per subject, or None unless the runs are complete."""
    seed_runs = dcca_runs.get(key)
    if not seed_runs or len(seed_runs) < seeds or completeness(seed_runs, n_archs, epochs):
        return None
    first = next(iter(next(iter(seed_runs.values())).values()))
    subjects = sorted(first["confusion"][0], key=int)
    k = max(max(archs) for archs in seed_runs.values())
    score = (lambda a, s: fair_score(a, s, False)) if mode == "fair" else (lambda a, s: paper_score(a, s, k, False))
    return {s: float(np.mean([score(archs, s) for archs in seed_runs.values()])) for s in subjects}


def matched_comparison(ours, theirs):
    """Averages over the directions both models have (per-direction mean over people), and directions won."""
    shared = [d for d in ours if theirs.get(d)]
    if not shared:
        return {"directions": 0, "ours": None, "theirs": None, "wins": 0}
    mine = {d: np.mean(list(ours[d].values())) for d in shared}
    other = {d: np.mean(list(theirs[d].values())) for d in shared}
    return {"directions": len(shared), "ours": float(np.mean(list(mine.values()))),
            "theirs": float(np.mean(list(other.values()))), "wins": sum(mine[d] > other[d] for d in shared)}


def fmt(values, rng):
    return mean_ci(np.array(list(values.values())), rng) if values else "–"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--dcca-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/norm_fusion_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    nf, dcca, simple = load_nf(args.root), load_runs(args.dcca_root), best_simple()

    main_table = ["# NormFusion vs DCCA-AM across cultures (EEG + eye)", "",
                  "Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
                  "p: paired Wilcoxon, NormFusion fair vs DCCA-AM fair, over test people.", "",
                  "| Train → Test | DCCA-AM reported | DCCA-AM paper's rules | NormFusion paper's rules "
                  "| DCCA-AM fair | NormFusion fair | NormFusion fair macro-F1 | p | Best simple |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ablation = ["", "## Ablations (fair accuracy %, mean over test people)", "",
                "| Train → Test | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    ours_by_direction, theirs_by_direction = {}, {}
    for source, target in DIRECTIONS:
        seed_records = nf.get((target, "emotion", source))
        if not seed_records:
            continue
        ours = nf_per_subject(seed_records, "fair")
        theirs = dcca_per_subject(dcca, (target, "emotion", source), "fair")
        p = "–"
        if theirs:
            diff = [ours[s] - theirs[s] for s in ours]
            p = f"{wilcoxon(diff).pvalue:.3f}" if any(diff) else "1"
        ref = simple.get((NAMES[source], NAMES[target]))
        main_table.append(
            f"| {NAMES[source]} → {NAMES[target]} | {PAPER_DCCA[(source, target)]:.1f} | "
            f"{fmt(dcca_per_subject(dcca, (target, 'emotion', source), 'paper'), rng)} | "
            f"{fmt(nf_paper_rules(seed_records), rng)} | {fmt(theirs, rng)} | {fmt(ours, rng)} | "
            f"{fmt(nf_per_subject(seed_records, 'fair', 'macro_f1'), rng)} | {p} | "
            f"{f'{ref:.1f}' if ref else '–'} |")
        ours_by_direction[(source, target)] = ours
        theirs_by_direction[(source, target)] = theirs
        row = [100 * np.mean(list(nf_per_subject(seed_records, v).values())) for v in VARIANTS]
        ablation.append(f"| {NAMES[source]} → {NAMES[target]} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    matched = matched_comparison(ours_by_direction, theirs_by_direction)
    all_ours = np.mean([np.mean(list(v.values())) for v in ours_by_direction.values()])
    main_table += ["", f"NormFusion fair, all {len(ours_by_direction)} directions: {100 * all_ours:.1f}%."]
    if matched["directions"]:
        main_table.append(
            f"Over the {matched['directions']} directions where DCCA-AM is complete: NormFusion fair "
            f"{100 * matched['ours']:.1f}% vs DCCA-AM fair {100 * matched['theirs']:.1f}%; NormFusion higher in "
            f"{matched['wins']} of {matched['directions']}. Success criterion (spec): higher on average over the "
            f"6 directions and in at least 4 — "
            + ("met." if matched["directions"] == 6 and matched["ours"] > matched["theirs"] and matched["wins"] >= 4
               else "not met." if matched["directions"] == 6 else "pending (not all directions complete)."))
    else:
        main_table.append("DCCA-AM: no complete direction yet.")

    newfilms = ["", "## New films (German ↔ French, shared films removed from training), fair accuracy %", "",
                "| Train → Test | NormFusion, all films | NormFusion, new films | DCCA-AM, new films |", "|---|---:|---:|---:|"]
    for source, target in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")):
        seed_records = nf.get((target, "emotion_newfilms", source))
        if not seed_records:
            continue
        newfilms.append(f"| {NAMES[source]} → {NAMES[target]} | "
                        f"{fmt(nf_per_subject(nf[(target, 'emotion', source)], 'fair'), rng)} | "
                        f"{fmt(nf_per_subject(seed_records, 'fair'), rng)} | "
                        f"{fmt(dcca_per_subject(dcca, (target, 'emotion_newfilms', source), 'fair'), rng)} |")

    text = "\n".join(main_table + ablation + newfilms) + "\n"
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
