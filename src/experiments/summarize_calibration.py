"""Summarise the calibration study in presentation order.

1. Cross-culture: the honest model without calibration next to 1 and 2 calibration clips per emotion, all on the
   same test clips (the pool films are removed from testing for every N).
2. Same-culture reference with and without calibration.
3. Culture gap (same-culture minus cross-culture, same test people) at each N.
4. Film leak: same-film minus film-disjoint calibration.
5. Modality: person model on EEG, eye, or both; person model alone.
6. Per-session accuracy (calibration comes from session 1).
Plus the pre-registered prediction and a figure of accuracy against N.

    python -m src.experiments.summarize_calibration
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.run_calibration import NS, VARIANTS
from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, mean_ci

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
# Pre-registered in docs/superpowers/specs/2026-10-04-calibration-design.md (accuracy % at N = 0, 1, 2).
PREDICTED = {"cross": (65, 74, 78), "same": (77, 81, 83), "gap": (12, 7, 5), "samefilm": (None, 4, 5)}
SERIES = {"cross": ("Cross-culture (6 directions)", "#2a78d6", "o"),
          "same": ("Same culture (3 cultures)", "#eb6834", "s")}


def load(root):
    """{(target, mode, source): {seed: scores}}; mode is cross or same (same-culture groups merged)."""
    runs = defaultdict(dict)
    for path in sorted(root.glob("*/*/seed_*.json")):
        folder, target = path.parent.name, path.parent.parent.name
        mode, source = folder.split("__from_")
        seed = int(path.stem.split("_")[1])
        key = (target, "same" if mode.startswith("same") else "cross", source)
        runs[key].setdefault(seed, {}).update(json.loads(path.read_text()))
    return runs


def person_means(seed_runs, n, variant, metric="accuracy"):
    """Per person: mean over seeds and draws."""
    values = defaultdict(list)
    for people in seed_runs.values():
        for person, draws in people.items():
            for by_n in draws.values():
                entry = by_n.get(str(n), {}).get(variant)
                if entry is not None:
                    values[person].append(entry[metric])
    return {p: float(np.mean(v)) for p, v in values.items()}


def culture_gap(same, crosses):
    """Same-culture minus the mean of the cross-culture directions, over the people present in both."""
    people = [p for p in same if all(p in c for c in crosses)]
    return float(np.mean([same[p] - np.mean([c[p] for c in crosses]) for p in people]))


def fmt(values, rng):
    return mean_ci(np.array(list(values.values())), rng) if values else "–"


def p_value(a, b):
    diff = [a[p] - b[p] for p in a if p in b]
    return f"{wilcoxon(diff).pvalue:.3f}" if len(diff) > 1 and any(diff) else "–"


def figure(path, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for key, values in curves.items():
        label, color, marker = SERIES[key]
        ax.plot(NS, values, color=color, linewidth=2, marker=marker, markersize=8, label=label,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.annotate(f"{values[-1]:.1f}%", (NS[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", color="#0b0b0b", fontsize=9)
    ax.set_xticks(NS, ["No calibration", "1 clip / emotion", "2 clips / emotion"])
    ax.set_ylabel("Accuracy (%)", color="#0b0b0b")
    ax.set_xlim(-0.3, NS[-1] + 0.5)
    ax.grid(axis="y", color="#e4e3dd", linewidth=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ad")
    ax.tick_params(colors="#3d3d3a")
    ax.set_title("Accuracy with N calibration clips per person (film-disjoint)", fontsize=10, color="#0b0b0b",
                 loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/calibration/scores"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/calibration_summary.md"))
    parser.add_argument("--figure", type=Path, default=Path("results/figures/calibration_curve.png"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = load(args.root)
    label = lambda c: NAMES[c]
    at = lambda key, n, variant="combined": person_means(runs[key], n, variant)

    lines = ["# Cross-cultural calibration (with N calibration clips per person)", "",
             "Accuracy %, mean over test people [95% bootstrap CI]; seeds (12, 34, 56) and 5 calibration pools "
             "averaged per person. Every N is scored on the same test clips: the 2 pool films per emotion are "
             "removed from testing for every N, including N = 0, so the no-calibration column differs slightly from "
             "the full-test v3 numbers. Calibration films are never test films. Calibrated numbers are not "
             "comparable with published results that use no calibration.", "",
             "## 1. Cross-culture: honest model, then calibration", "",
             "| Train → Test | No calibration | 1 clip / emotion | 2 clips / emotion | p (2 vs 0) |",
             "|---|---:|---:|---:|---:|"]
    cross_means = []
    for source, target in DIRECTIONS:
        key = (target, "cross", source)
        if key not in runs:
            continue
        values = [at(key, n) for n in NS]
        cross_means.append([np.mean(list(v.values())) for v in values])
        lines.append(f"| {label(source)} → {label(target)} | " + " | ".join(fmt(v, rng) for v in values)
                     + f" | {p_value(values[2], values[0])} |")
    if cross_means:
        avg = 100 * np.mean(cross_means, axis=0)
        lines += [f"| **Average ({len(cross_means)} directions)** | " + " | ".join(f"**{a:.1f}**" for a in avg) + " | |",
                  "| Predicted (pre-registered) | " + " | ".join(str(p) for p in PREDICTED["cross"]) + " | |"]

    lines += ["", "## 2. Same-culture reference (train on other people of the same culture)", "",
              "| Test culture | No calibration | 1 clip / emotion | 2 clips / emotion | p (2 vs 0) |",
              "|---|---:|---:|---:|---:|"]
    same_means, gaps = [], {n: [] for n in NS}
    gap_rows = []
    for culture in CULTURES:
        key = (culture, "same", culture)
        if key not in runs:
            continue
        values = [at(key, n) for n in NS]
        same_means.append([np.mean(list(v.values())) for v in values])
        lines.append(f"| {label(culture)} | " + " | ".join(fmt(v, rng) for v in values)
                     + f" | {p_value(values[2], values[0])} |")
        crosses = [(t, "cross", s) for s, t in DIRECTIONS if t == culture and (t, "cross", s) in runs]
        if crosses:
            row = [culture_gap(at(key, n), [at(c, n) for c in crosses]) for n in NS]
            for n, g in zip(NS, row):
                gaps[n].append(g)
            gap_rows.append(f"| {label(culture)} | " + " | ".join(f"{100 * g:.1f}" for g in row) + " |")
    if same_means:
        avg = 100 * np.mean(same_means, axis=0)
        lines += [f"| **Average ({len(same_means)} cultures)** | " + " | ".join(f"**{a:.1f}**" for a in avg) + " | |",
                  "| Predicted (pre-registered) | " + " | ".join(str(p) for p in PREDICTED["same"]) + " | |"]

    lines += ["", "## 3. Culture gap (same-culture minus cross-culture, same test people; points)", "",
              "Includes film familiarity: same-culture training people watched the test person's films. Upper bound.",
              "", "| Test culture | No calibration | 1 clip / emotion | 2 clips / emotion |", "|---|---:|---:|---:|"]
    lines += gap_rows
    if gap_rows:
        lines += ["| **Average** | " + " | ".join(f"**{100 * np.mean(gaps[n]):.1f}**" for n in NS) + " |",
                  "| Predicted (pre-registered) | " + " | ".join(str(p) for p in PREDICTED["gap"]) + " |"]

    lines += ["", "## 4. Film leak: calibrating on films that are also tested (leaky) vs film-disjoint", "",
              "| Train → Test | 1 clip: disjoint | 1 clip: same film | 2 clips: disjoint | 2 clips: same film |",
              "|---|---:|---:|---:|---:|"]
    leak = {1: [], 2: []}
    for source, target in DIRECTIONS:
        key = (target, "cross", source)
        if key not in runs:
            continue
        cells = []
        for n in (1, 2):
            disjoint, samefilm = mean_of(at(key, n)), mean_of(at(key, n, "combined_samefilm"))
            leak[n].append(samefilm - disjoint)
            cells += [f"{100 * disjoint:.1f}", f"{100 * samefilm:.1f}"]
        lines.append(f"| {label(source)} → {label(target)} | " + " | ".join(cells) + " |")
    if leak[1]:
        lines.append(f"| **Inflation (same − disjoint)** | | **{100 * np.mean(leak[1]):+.1f}** | | "
                     f"**{100 * np.mean(leak[2]):+.1f}** |")

    lines += ["", "## 5. Modality of the person model (cross-culture, accuracy %)", "",
              "| Train → Test | N | combined (EEG + eye) | EEG only | eye only | person model alone |",
              "|---|---:|---:|---:|---:|---:|"]
    for source, target in DIRECTIONS:
        key = (target, "cross", source)
        if key not in runs:
            continue
        for n in (1, 2):
            lines.append(f"| {label(source)} → {label(target)} | {n} | " + " | ".join(
                f"{100 * mean_of(at(key, n, v)):.1f}" for v in ("combined", "combined_eeg", "combined_eye",
                                                                 "person_only")) + " |")

    lines += ["", "## 6. Accuracy by session (cross-culture average; calibration comes from session 1)", "",
              "| N | session 1 | session 2 | session 3 |", "|---:|---:|---:|---:|"]
    for n in NS:
        by_session = defaultdict(list)
        for source, target in DIRECTIONS:
            for people in runs.get((target, "cross", source), {}).values():
                for draws in people.values():
                    for by_n in draws.values():
                        for s, v in by_n[str(n)]["combined"]["sessions"].items():
                            by_session[s].append(v)
        lines.append(f"| {n} | " + " | ".join(f"{100 * np.mean(by_session[s]):.1f}" if by_session[s] else "–"
                                             for s in ("1", "2", "3")) + " |")

    lines += ["", "## 7. Follow-ups (exploratory, not pre-registered)", "",
              "persession: a calibration film per emotion at the start of every session, one person model per "
              "session. learned: the weight on the person model chosen by leave-one-film-out on the calibration "
              "windows (2 films per emotion needed). Same test clips as above.", "",
              "| Train → Test | N | combined (pre-registered) | persession | learned | persession + learned |",
              "|---|---:|---:|---:|---:|---:|"]
    followup_means = defaultdict(list)
    for mode, keys in (("cross", [((t, "cross", s), f"{label(s)} → {label(t)}") for s, t in DIRECTIONS]),
                       ("same", [((c, "same", c), f"{label(c)} (same culture)") for c in CULTURES])):
        for key, name in keys:
            if key not in runs:
                continue
            for n in (1, 2):
                cells = []
                for variant in ("combined", "persession", "learned", "persession_learned"):
                    value = mean_of(at(key, n, variant))
                    if not np.isnan(value):
                        followup_means[(mode, n, variant)].append(value)
                    cells.append("–" if np.isnan(value) else f"{100 * value:.1f}")
                lines.append(f"| {name} | {n} | " + " | ".join(cells) + " |")
    for mode, name in (("cross", "Cross-culture average"), ("same", "Same-culture average")):
        for n in (1, 2):
            cells = [f"**{100 * np.mean(followup_means[(mode, n, v)]):.1f}**" if followup_means[(mode, n, v)] else "–"
                     for v in ("combined", "persession", "learned", "persession_learned")]
            if any(c != "–" for c in cells):
                lines.append(f"| **{name}** | {n} | " + " | ".join(cells) + " |")

    curves = {}
    if cross_means:
        curves["cross"] = list(100 * np.mean(cross_means, axis=0))
    if same_means:
        curves["same"] = list(100 * np.mean(same_means, axis=0))
    if curves:
        figure(args.figure, curves)
        lines += ["", f"Figure: `{args.figure}`."]
    text = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


def mean_of(values):
    return float(np.mean(list(values.values()))) if values else float("nan")


if __name__ == "__main__":
    main()
