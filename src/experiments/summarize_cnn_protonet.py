"""Summarise the CNN and prototypical-network few-shot study.

1. Cross-culture average of every variant at N = 0, 1, 2 clips per emotion, with the pre-registered prediction.
2. Per direction at N = 1 and N = 2, and wins against v3+person (the calibration study's combined model).
3. The pre-registered success check.

    python -m src.experiments.summarize_cnn_protonet

Design: docs/superpowers/specs/2026-10-24-cnn-protonet-design.md.
"""

import argparse
from pathlib import Path

import numpy as np

from src.experiments.summarize_calibration import load, p_value, person_means
from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, mean_ci

ROWS = [("v3", "v3+person"), ("mlp_clf", "mlp_clf+person"), ("cnn_clf", "cnn_clf+person"), (None, "mlp_proto"),
        (None, "cnn_proto"), (None, "mlp_proto+v3"), (None, "cnn_proto+v3"), (None, "cnn_proto+cnn_clf")]
LABELS = {"v3+person": "NormFusion v3 (+ person model) — calibration study",
          "mlp_clf+person": "MLP classifier (+ person model)", "cnn_clf+person": "CNN classifier (+ person model)",
          "mlp_proto": "MLP prototypical network", "cnn_proto": "CNN prototypical network",
          "mlp_proto+v3": "MLP prototypical + v3", "cnn_proto+v3": "CNN prototypical + v3",
          "cnn_proto+cnn_clf": "CNN prototypical + CNN classifier"}
# Pre-registered in the design (accuracy % at N = 0, 1, 2).
PREDICTED = {"cnn_clf+person": (62, 61, 65), "mlp_proto": (None, 58, 61), "cnn_proto": (None, 58, 61),
             "cnn_proto+v3": (None, 65, 68)}
SUCCESS_MARGIN, SUCCESS_WINS = 3.0, 4


def direction_means(runs, n, variant):
    """{direction: {person: accuracy}} at one N."""
    return {(s, t): person_means(runs[(t, "cross", s)], n, variant) for s, t in DIRECTIONS if (t, "cross", s) in runs}


def avg(by_direction):
    return 100 * float(np.mean([np.mean(list(p.values())) for p in by_direction.values()])) if by_direction else None


def fmt(value):
    return "–" if value is None else f"{value:.1f}"


def summary(runs):
    rng = np.random.default_rng(0)
    seeds = {f"{NAMES[s]}→{NAMES[t]}": len(runs.get((t, "cross", s), {})) for s, t in DIRECTIONS}
    lines = ["# CNN and prototypical-network few-shot study", "",
             "Accuracy %, cross-culture, same test windows and calibration pools as the calibration study "
             "(results/tables/calibration_summary.md); mean over test people (seeds and 5 pools averaged per person), "
             "then over the 6 directions. All weights and hyperparameters fixed in advance.", "",
             "Runs per direction (of 3 seeds): " + ", ".join(f"{k} {v}" for k, v in seeds.items()), "",
             "## 1. Average over the 6 directions", "",
             "| Model | No calibration | 1 clip / emotion | 2 clips / emotion | Predicted (0 / 1 / 2) |",
             "|---|---:|---:|---:|---|"]
    for zero, calibrated in ROWS:
        values = [avg(direction_means(runs, 0, zero)) if zero else None] + \
                 [avg(direction_means(runs, n, calibrated)) for n in (1, 2)]
        predicted = " / ".join(fmt(v) for v in PREDICTED.get(calibrated, (None,) * 3))
        lines.append(f"| {LABELS[calibrated]} | " + " | ".join(fmt(v) for v in values) + f" | {predicted} |")
    for n in (1, 2):
        lines += ["", f"## {n + 1}. Per direction, {n} clip{'s' if n > 1 else ''} per emotion "
                  "(mean [95% bootstrap CI over people]; p: Wilcoxon over people against v3+person)", "",
                  "| Train → Test | " + " | ".join(LABELS[c].split(" (")[0].split(" —")[0] for _, c in ROWS) + " |",
                  "|---|" + "---:|" * len(ROWS)]
        reference = direction_means(runs, n, "v3+person")
        per_variant = {c: direction_means(runs, n, c) for _, c in ROWS}
        for s, t in DIRECTIONS:
            if (s, t) not in reference:
                continue
            cells = []
            for _, c in ROWS:
                people = per_variant[c][(s, t)]
                cell = mean_ci(np.array(list(people.values())), rng)
                cells.append(cell if c == "v3+person" else f"{cell} p={p_value(people, reference[(s, t)])}")
            lines.append(f"| {NAMES[s]} → {NAMES[t]} | " + " | ".join(cells) + " |")
        wins = {c: sum(np.mean(list(per_variant[c][d].values())) > np.mean(list(reference[d].values()))
                       for d in reference) for _, c in ROWS if c != "v3+person"}
        lines.append("| **Wins vs v3+person** | – | " + " | ".join(f"{wins[c]}/{len(reference)}"
                                                            for _, c in ROWS if c != "v3+person") + " |")
    reference = avg(direction_means(runs, 2, "v3+person"))
    best = max((c for _, c in ROWS if c != "v3+person"), key=lambda c: avg(direction_means(runs, 2, c)) or 0)
    best_value = avg(direction_means(runs, 2, best))
    ref_n2, best_n2 = direction_means(runs, 2, "v3+person"), direction_means(runs, 2, best)
    wins = sum(np.mean(list(best_n2[d].values())) > np.mean(list(ref_n2[d].values())) for d in ref_n2)
    met = best_value - reference >= SUCCESS_MARGIN and wins >= SUCCESS_WINS
    lines += ["", "## 4. Pre-registered success check (N = 2)", "",
              f"Best variant: {LABELS[best]} {best_value:.1f}% vs v3+person {reference:.1f}% "
              f"({best_value - reference:+.1f} points; needs >= +{SUCCESS_MARGIN:.0f}), wins {wins}/{len(ref_n2)} "
              f"(needs >= {SUCCESS_WINS}). **Success {'MET' if met else 'NOT MET'}.**", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-root", type=Path, default=Path("results/cnn_protonet/scores"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/cnn_protonet_summary.md"))
    args = parser.parse_args()
    text = summary(load(args.scores_root))
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
