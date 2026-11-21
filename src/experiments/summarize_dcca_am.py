"""Summarise the DCCA-AM runs under three evaluation modes, next to Liu et al.

  paper's rules    best test accuracy over architectures 1..k and all epochs, per test person (as released)
  fair             architecture 0 (150/35, fixed in advance), last epoch
  fair, new films  the same, trained without the 5 shared German/French films

Per test person, scores are averaged over seeds; tables give the mean over test
people with a 95% bootstrap CI. Also: the leakage curve (paper's rules for
k = 1, 2, 5, 10, 20) and film identification (balanced accuracy, block
permutation p), with the linear E2 model alongside.

    python -m src.experiments.summarize_dcca_am   # -> results/tables/dcca_am_summary.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.film_identification import block_permutations, per_subject_accuracy

NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}
DIRECTIONS = [("SEED_CHN_MM", "SEED_GER"), ("SEED_CHN_MM", "SEED_FRA"), ("SEED_GER", "SEED_CHN_MM"),
              ("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_CHN_MM"), ("SEED_FRA", "SEED_GER")]
# Liu et al. 2022, table 9 (CCSI, DCCA-AM).
PAPER_DCCA = {("SEED_CHN_MM", "SEED_FRA"): 73.84, ("SEED_CHN_MM", "SEED_GER"): 73.63,
              ("SEED_GER", "SEED_CHN_MM"): 84.28, ("SEED_GER", "SEED_FRA"): 77.06,
              ("SEED_FRA", "SEED_CHN_MM"): 82.09, ("SEED_FRA", "SEED_GER"): 77.92}
K_CURVE = [1, 2, 5, 10, 20]


def load_runs(root):
    runs = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/*__from_*/seed_*/arch_*.json"):
        record = json.loads(path.read_text())
        runs[(record["target"], record["task"], record["source"])][record["seed"]][record["arch"]] = record
    return runs


def accuracy(cm, balanced):
    cm = np.asarray(cm, dtype=float)
    if not balanced:
        return float(np.trace(cm) / cm.sum())
    totals = cm.sum(axis=1)
    return float(np.mean(np.diag(cm)[totals > 0] / totals[totals > 0]))


def fair_score(arch_records, subject, balanced):
    return accuracy(arch_records[0]["confusion"][-1][subject], balanced)


def paper_score(arch_records, subject, k, balanced):
    return max(accuracy(epoch[subject], balanced)
               for index in range(1, k + 1) if index in arch_records
               for epoch in arch_records[index]["confusion"])


def completeness(seed_runs, n_archs, epochs):
    """Problems that would make a mode's number unrepresentative: missing architectures or short runs."""
    problems = []
    for seed, archs in sorted(seed_runs.items()):
        for index in range(n_archs):
            if index not in archs:
                problems.append(f"seed {seed}: arch {index} missing")
            elif archs[index]["epochs"] != epochs or len(archs[index]["confusion"]) != epochs:
                problems.append(f"seed {seed}: arch {index} has {len(archs[index]['confusion'])} epochs, not {epochs}")
    return problems


def health(seed_runs):
    """(skipped batches, non-finite epochs), summed over seeds and architectures."""
    records = [r for archs in seed_runs.values() for r in archs.values()]
    return sum(r["skipped_batches"] for r in records), sum(r["nonfinite_epochs"] for r in records)


def film_p_value(seed_runs, categories, rng, n_permutations):
    """Block-permutation p for the seed-averaged fair accuracy (architecture 0, final epoch).

    Each permutation relabels the test films once and scores every seed's
    predictions against the same relabelling, so the null matches the
    seed-averaged statistic; seeds are taken in sorted order.
    """
    finals = [seed_runs[seed][0]["final"] for seed in sorted(seed_runs)]
    subjects = sorted(finals[0], key=int)
    true = np.concatenate([finals[0][s]["true"] for s in subjects])
    recordings = np.concatenate([finals[0][s]["recording"] for s in subjects])
    subject_ids = np.concatenate([[int(s)] * len(finals[0][s]["true"]) for s in subjects])
    preds = [np.concatenate([final[s]["pred"] for s in subjects]) for final in finals]

    def statistic(labels):
        return np.mean([per_subject_accuracy(labels, pred, subject_ids).mean() for pred in preds])

    observed = statistic(true)
    null = [statistic(shuffled) for shuffled in block_permutations(true, recordings, list(range(categories)), rng,
                                                                    n_permutations)]
    return float((1 + sum(n >= observed for n in null)) / (1 + n_permutations))


def per_subject(seed_runs, score):
    subjects = sorted(next(iter(next(iter(seed_runs.values())).values()))["confusion"][0], key=int)
    return np.array([np.mean([score(archs, s) for archs in seed_runs.values()]) for s in subjects])


def mean_ci(values, rng):
    boot = rng.choice(values, (10000, len(values))).mean(axis=1)
    return f"{100 * values.mean():.1f} [{100 * np.percentile(boot, 2.5):.1f}, {100 * np.percentile(boot, 97.5):.1f}]"


def best_simple():
    """Best normalised simple EEG + eye model per direction (4 s), from crossculture_summary.csv."""
    best = {}
    for row in csv.DictReader(open("results/tables/crossculture_summary.csv")):
        if row["setting"] == "cross" and row["modality"] == "eeg+eye" and row["window"] == "4s" \
                and row["method"].endswith("__subjnorm"):
            key = (row["train"], row["test"])
            best[key] = max(best.get(key, 0.0), float(row["accuracy_mean"]))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/tables/dcca_am_summary.md"))
    parser.add_argument("--archs", type=int, default=21, help="Architectures expected per seed (0..archs-1).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Summarise even if runs are missing or short (the report is then marked INCOMPLETE).")
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = load_runs(args.root)
    simple = best_simple()
    k_max = max(K_CURVE)
    problems = [f"{target} {task} from {source}: {p}" for (target, task, source), seed_runs in sorted(runs.items())
                for p in completeness(seed_runs, args.archs, args.epochs)]
    if problems and not args.allow_incomplete:
        raise SystemExit("Incomplete runs (use --allow-incomplete to summarise anyway):\n" + "\n".join(problems))

    lines = ["# DCCA-AM across cultures: paper vs paper's rules vs fair"
             + (f" (INCOMPLETE: {len(problems)} problems)" if problems else ""), "",
             "Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person.", "",
             "| Train → Test | Paper (reported) | Paper's rules (ours) | Fair | Fair, new films | Best simple, normalised "
             "| Skipped batches / non-finite epochs |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    curve = ["", "## Leakage curve: paper's rules when searching k architectures", "",
             "| Train → Test | " + " | ".join(f"k = {k}" for k in K_CURVE) + " | Fair |",
             "|---|" + "---:|" * (len(K_CURVE) + 1)]
    for source, target in DIRECTIONS:
        seed_runs = runs.get((target, "emotion", source))
        if not seed_runs:
            continue
        paper = per_subject(seed_runs, lambda a, s: paper_score(a, s, k_max, False))
        fair = per_subject(seed_runs, lambda a, s: fair_score(a, s, False))
        newfilms = runs.get((target, "emotion_newfilms", source))
        new = mean_ci(per_subject(newfilms, lambda a, s: fair_score(a, s, False)), rng) if newfilms else "same as fair"
        ref = simple.get((NAMES[source], NAMES[target]))
        skipped, nonfinite = health(seed_runs)
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {PAPER_DCCA[(source, target)]:.1f} | "
                     f"{mean_ci(paper, rng)} | {mean_ci(fair, rng)} | {new} | {f'{ref:.1f}' if ref else '–'} | "
                     f"{skipped} / {nonfinite} |")
        ks = [100 * per_subject(seed_runs, lambda a, s, k=k: paper_score(a, s, k, False)).mean() for k in K_CURVE]
        curve.append(f"| {NAMES[source]} → {NAMES[target]} | " + " | ".join(f"{v:.1f}" for v in ks)
                     + f" | {100 * fair.mean():.1f} |")

    e2 = {(r["train"], r["test"], r["task"]): r["accuracy"]
          for r in json.loads(Path("results/tables/film_identification.json").read_text())["tasks"]
          if r["modality"] == "EEG + eye"}
    film = ["", "## Film identification (German ↔ French shared films), balanced accuracy %", "",
            "| Train → Test | Task | Chance | DCCA-AM fair | p | DCCA-AM paper's rules | Linear (E2) |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for (target, task, source), seed_runs in sorted(runs.items()):
        if not task.startswith("film_"):
            continue
        first_record = next(iter(next(iter(seed_runs.values())).values()))
        categories = len(next(iter(first_record["confusion"][0].values())))
        fair = per_subject(seed_runs, lambda a, s: fair_score(a, s, True))
        paper = per_subject(seed_runs, lambda a, s: paper_score(a, s, k_max, True))
        p = film_p_value(seed_runs, categories, rng, args.permutations)
        linear = e2.get((NAMES[source], NAMES[target], task.removeprefix("film_")))
        linear_text = f"{100 * linear:.1f}" if linear is not None else "–"
        film.append(f"| {NAMES[source]} → {NAMES[target]} | {task.removeprefix('film_')} | {100 / categories:.0f} | "
                    f"{mean_ci(fair, rng)} | {p:.4f} | {100 * paper.mean():.1f} | {linear_text} |")

    args.output.write_text("\n".join(lines + curve + film) + "\n")
    print("\n".join(lines + curve + film))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
