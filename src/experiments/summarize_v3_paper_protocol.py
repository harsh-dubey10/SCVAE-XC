"""Summarise NormFusion v3 fair against v3 under the papers' test-label-selected rules.

  fair                 v3 as designed: no test labels used (the reportable number)
  test-selected (set)  one candidate per run, the one with the best mean accuracy over the test people
                       (SCVAE's released code: checkpoint chosen on the test set)
  test-selected (person) per test person, the candidate with that person's best accuracy (Liu et al.: best C
                       per test subject)

Test-label-selected columns are not accuracy; they show how much the papers' rules inflate a fair model.

    python -m src.experiments.summarize_v3_paper_protocol
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, PAPER_DCCA, mean_ci

RULES = ("fair", "set", "person")


def load(root):
    runs = defaultdict(list)
    for path in sorted(root.glob("*/cross__from_*/seed_*.json")):
        record = json.loads(path.read_text())
        runs[(record["source"], record["target"])].append(record)
    return runs


def person_accuracy(records):
    """{rule: {person: accuracy, averaged over seeds}}."""
    values = {rule: defaultdict(list) for rule in RULES}
    for record in records:
        table = np.array([record["accuracy"][p] for p in record["accuracy"]])
        best_set = int(table.mean(axis=0).argmax())
        fair = record["keys"].index(record["fair"])
        for row, person in zip(table, record["accuracy"]):
            values["fair"][person].append(row[fair])
            values["set"][person].append(row[best_set])
            values["person"][person].append(row.max())
    return {rule: {p: float(np.mean(v)) for p, v in by.items()} for rule, by in values.items()}


def summary(runs):
    rng = np.random.default_rng(0)
    n_keys = sorted({len(r["keys"]) for records in runs.values() for r in records})
    lines = ["# NormFusion v3: fair vs the papers' test-label-selected rules", "",
             "Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
             f"Candidates per run: {', '.join(map(str, n_keys))} (expert sets x C per expert without "
             "self-training, plus self-training rounds 1-3 at the fair Cs). **Only the fair column is accuracy.** "
             "The test-label-selected columns choose settings by accuracy on the test people, as the papers did, "
             "and are shown only to measure that inflation.", "",
             "| Train → Test | Seeds | v3 fair | v3 test-selected (one setting per run) | "
             "v3 test-selected (per person) | DCCA-AM as published (test-selected) |",
             "|---|---:|---:|---:|---:|---:|"]
    means = {rule: [] for rule in RULES}
    paper = []
    for source, target in DIRECTIONS:
        if (source, target) not in runs:
            continue
        acc = person_accuracy(runs[(source, target)])
        cells = []
        for rule in RULES:
            values = np.array(list(acc[rule].values()))
            means[rule].append(100 * values.mean())
            cells.append(mean_ci(values, rng))
        paper.append(PAPER_DCCA[(source, target)])
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {len(runs[(source, target)])} | " + " | ".join(cells)
                     + f" | {PAPER_DCCA[(source, target)]:.1f} |")
    avg = {rule: float(np.mean(v)) for rule, v in means.items()}
    lines.append(f"| **Average** | | **{avg['fair']:.1f}** | {avg['set']:.1f} | {avg['person']:.1f} | "
                 f"{np.mean(paper):.1f} |")
    lines += ["", f"Inflation from test-label selection: +{avg['set'] - avg['fair']:.1f} points (one setting per "
              f"run), +{avg['person'] - avg['fair']:.1f} points (per person).", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/v3_paper_protocol"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/v3_paper_protocol_summary.md"))
    args = parser.parse_args()
    text = summary(load(args.root))
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
