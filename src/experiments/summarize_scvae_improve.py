"""Tables for the SCVAE cross-culture and unseen-film study (spec 2026-10-09).

Strict columns are accuracy on unseen people. Columns marked "test-label-selected"
choose the checkpoint by test accuracy, as the released code did; they are a
labelled comparison, not accuracy.

    python -m src.experiments.summarize_scvae_improve
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from src.experiments.film_identification import FILMS
from src.experiments.make_scvae_improve_manifest import GRID, config_name
from src.experiments.select_crossculture_config import CULTURES, NAMES, choose, load
from src.experiments.train_scvae_reference import all_subjects

GRID_ROOT = Path("results/scvae_improve/grid")
PAPER_ROOT = Path("results/scvae_improve/paper_protocol")
FILM_ROOT = Path("results/scvae_improve/films")
RELEASED_ROOT = Path("results/paper_scvae")
BASELINE_CSV = Path("results/tables/crossculture_summary.csv")
SIMPLE_FILM_MD = Path("results/tables/clip_heldout_emotion.md")
OUTPUT = Path("results/tables/scvae_improve_summary.md")
SIMPLE = ("mlp__subjnorm", "svm__subjnorm", "linsvm__subjnorm")
CULTURE_OF = {"Chinese": "SEED_CHN_MM", "German": "SEED_GER", "French": "SEED_FRA"}
PAIRS = [(s, t) for s in CULTURES for t in CULTURES if s != t]
SHARED_COLUMN = {"SEED_GER": 0, "SEED_FRA": 1}  # position of each culture's trial in FILMS values after emotion
SUCCESS_WINS = 4


def per_subject(run_dir):
    """Test subject -> (accuracy %, macro-F1 %), averaged over seeds; {} if nothing was run."""
    accuracy, f1 = defaultdict(list), defaultdict(list)
    for path in Path(run_dir).glob("session_all/seed_*/subject_*.json"):
        record = json.loads(path.read_text())
        accuracy[record["test_subject"]].append(record["accuracy"])
        f1[record["test_subject"]].append(record["macro_f1"])
    return {s: (100 * float(np.mean(accuracy[s])), 100 * float(np.mean(f1[s]))) for s in accuracy}


def seed_files(run_dir):
    """Test subject -> list of loaded per-seed .npz files."""
    files = defaultdict(list)
    for path in sorted(Path(run_dir).glob("session_all/seed_*/subject_*_proba.npz")):
        files[int(re.search(r"subject_(\d+)_proba", path.name).group(1))].append(np.load(path))
    return files


def ensemble_predictions(saved):
    labels = saved[0]["labels"]
    assert all(np.array_equal(s["labels"], labels) for s in saved), "seeds disagree on the target windows"
    return labels, np.mean([s["proba"] for s in saved], axis=0).argmax(axis=1)


def ensemble(run_dir):
    """Test subject -> (accuracy %, macro-F1 %) of the mean of the seeds' saved probabilities."""
    out = {}
    for subject, saved in seed_files(run_dir).items():
        labels, predictions = ensemble_predictions(saved)
        out[subject] = (100 * float(np.mean(predictions == labels)),
                        100 * float(f1_score(labels, predictions, average="macro")))
    return out


def shared_split(run_dir, culture):
    """Test subject -> (shared films, other films) balanced accuracy % over negative and positive windows."""
    shared = {trials[SHARED_COLUMN[culture]]: emotion for emotion, *trials in FILMS.values()}
    out = {}
    for subject, saved in seed_files(run_dir).items():
        labels, predictions = ensemble_predictions(saved)
        trials = saved[0]["trials"]
        for trial, emotion in shared.items():
            assert set(labels[trials == trial].tolist()) <= {emotion}, f"trial {trial} is not emotion {emotion}"
        in_shared = np.isin(trials, list(shared))
        scores = []
        for part in (in_shared, ~in_shared):
            recalls = [np.mean(predictions[part & (labels == c)] == c) for c in (0, 2) if (part & (labels == c)).any()]
            scores.append(100 * float(np.mean(recalls)))
        out[subject] = tuple(scores)
    return out


def coverage(run_dir, n_people, kind="json"):
    """(seeds present, people with a result in every seed, people expected)."""
    pattern = "subject_*_proba.npz" if kind == "proba" else "subject_*.json"
    seeds = sorted(Path(run_dir).glob("session_all/seed_*")) if run_dir else []
    if not seeds:
        return 0, 0, n_people
    people = set.intersection(*({int(re.search(r"subject_(\d+)", p.name).group(1)) for p in seed.glob(pattern)}
                                for seed in seeds))
    return len(seeds), len(people), n_people


def coverage_text(counts, seeds=3):
    found, people, expected = counts
    return f"{found} seeds, {people}/{expected} people" + ("" if found == seeds and people == expected else " ⚠")


def n_people(culture):
    return len(all_subjects(Path("data/processed") / culture))


def find_run(root, target, name):
    """The run directory whose name (sval count merged) equals ``name``, or None."""
    directory = Path(root) / target
    if not directory.is_dir():
        return None
    for run in directory.iterdir():
        if re.sub(r"__sval\d+", "__sval", run.name) == name:
            return run
    return None


def baselines(csv_path=BASELINE_CSV):
    """(source, target) -> best normalised simple model accuracy (EEG, 4 s), chosen after the fact."""
    best = {}
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            if (row["setting"], row["modality"], row["window"]) == ("cross", "eeg", "4s") and row["method"] in SIMPLE:
                key = (CULTURE_OF[row["train"]], CULTURE_OF[row["test"]])
                best[key] = max(best.get(key, 0.0), float(row["accuracy_mean"]))
    return best


def film_people(root, culture, base_name, condition):
    """Person -> accuracy % averaged over the 3 film folds (seeds averaged); people missing a fold are dropped."""
    folds = []
    for fold in range(3):
        run = find_run(root, culture, f"{base_name}__films_{condition}{fold}__pooled")
        folds.append(per_subject(run) if run else {})
    people = set.intersection(*(set(f) for f in folds))
    return {p: float(np.mean([f[p][0] for f in folds])) for p in sorted(people)}


def simple_film_reference(path=SIMPLE_FILM_MD):
    """Culture name -> (seen, unseen) balanced accuracy of the simple EEG model."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and cells[1] == "EEG":
                out[cells[0]] = (float(cells[2]), float(cells[3]))
    return out


def mean_acc(results):
    return float(np.mean([a for a, _ in results.values()])) if results else None


def bootstrap_ci(values, n=10000, seed=0):
    values = np.asarray(values)
    means = np.random.default_rng(seed).choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_line(label, a, b):
    """'a − b' over paired people, with CI and Wilcoxon, or a note when nothing is paired."""
    if len(a) < 2:
        return f"- {label}: not enough paired people."
    difference = np.array(a) - np.array(b)
    low, high = bootstrap_ci(difference)
    p_value = wilcoxon(a, b).pvalue if np.any(difference) else 1.0
    return (f"- {label}: {difference.mean():+.1f} points over {len(a)} pairs "
            f"(95% CI {low:+.1f} to {high:+.1f}), Wilcoxon p = {p_value:.3g}.")


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def part1(scores, chosen, lines):
    base = baselines()
    rows, fixed_pairs, released_pairs, wins = [], [], [], 0
    for source, target in PAIRS:
        config = chosen[target]
        fixed_dir = find_run(GRID_ROOT, target, f"{config}__from_{source}") if config else None
        released = per_subject(RELEASED_ROOT / target / f"strict__bn_source__from_{source}")
        fixed_ens = ensemble(fixed_dir) if fixed_dir else {}
        p1_dir = PAPER_ROOT / target / f"reported_diagnostic__from_{source}"
        row = dict(direction=f"{NAMES[source]} → {NAMES[target]}", baseline=base.get((source, target)),
                   released=mean_acc(released), single=mean_acc(per_subject(fixed_dir) if fixed_dir else {}),
                   ens=mean_acc(fixed_ens),
                   ens_f1=float(np.mean([f for _, f in fixed_ens.values()])) if fixed_ens else None,
                   p1=mean_acc(per_subject(p1_dir)),
                   seeds=(f"{coverage_text(coverage(fixed_dir, n_people(target), kind='proba'))} / "
                          f"{coverage_text(coverage(p1_dir, n_people(target)))}"))
        rows.append(row)
        wins += row["ens"] is not None and row["baseline"] is not None and row["ens"] > row["baseline"]
        for subject in sorted(set(fixed_ens) & set(released)):
            fixed_pairs.append(fixed_ens[subject][0])
            released_pairs.append(released[subject][0])

    def average(key):
        values = [r[key] for r in rows]
        return None if any(v is None for v in values) else float(np.mean(values))

    lines += ["", "## Part 1 — Across cultures: accuracy (%) per direction", "",
              "| Train → Test | Best simple model | SCVAE released (strict) | SCVAE-fixed, 1 seed (strict) | "
              "SCVAE-fixed, 3-seed ensemble (strict) | Ensemble macro-F1 || Released, **test-label-selected** "
              "| runs complete (fixed ensemble / test-label-selected) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['direction']} | {fmt(r['baseline'])} | {fmt(r['released'])} | {fmt(r['single'])} | "
                     f"**{fmt(r['ens'])}** | {fmt(r['ens_f1'])} || {fmt(r['p1'])} | {r['seeds']} |")
    lines.append(f"| **Average** | {fmt(average('baseline'))} | {fmt(average('released'))} | "
                 f"{fmt(average('single'))} | **{fmt(average('ens'))}** | {fmt(average('ens_f1'))} || "
                 f"{fmt(average('p1'))} | |")
    lines += ["", paired_line("SCVAE-fixed ensemble − SCVAE released (strict), (direction, person) pairs",
                              fixed_pairs, released_pairs)]
    if average("p1") is not None and average("released") is not None:
        lines.append(f"- Peeking gain, released recipe (test-label-selected − strict): "
                     f"{average('p1') - average('released'):+.1f} points.")
    ens_avg, base_avg = average("ens"), average("baseline")
    if ens_avg is not None:
        met = ens_avg > base_avg and wins >= SUCCESS_WINS
        lines += ["", "**Pre-registered success criterion:** "
                  f"SCVAE-fixed ensemble {ens_avg:.1f}% vs best simple models {base_avg:.1f}%; wins {wins} of 6 "
                  f"(need > {base_avg:.1f}% and ≥ {SUCCESS_WINS}). **{'MET' if met else 'NOT MET'}.** "
                  "Predicted: 58% strict, 72% test-label-selected."]

    lines += ["", "### Grid factors (exploratory; mean over all 6 directions and the other factors)", "",
              "| Factor | Level | Mean accuracy |", "|---|---|---:|"]
    for factor, levels in [("learning rate", {"0.0025": lambda c: "__lrate" not in c,
                                              "0.0005": lambda c: "__lrate0.0005" in c}),
                           ("BN at test", {"source": lambda c: "__bn_source" in c,
                                           "target": lambda c: "__bn_target" in c}),
                           ("input noise", {"0": lambda c: "__noise" not in c,
                                            "0.1": lambda c: "__noise0.1" in c})]:
        for level, matches in levels.items():
            values = [v for (c, _, _), v in scores.items() if matches(c)]
            lines.append(f"| {factor} | {level} | {fmt(float(np.mean(values)) if values else None)} |")


def part2(chosen, lines):
    models = [("SCVAE released + BN fix (strict)", lambda c: "strict__bn_source"),
              ("SCVAE-fixed (strict)", lambda c: chosen[c]),
              ("SCVAE released, **test-label-selected**", lambda c: "reported_diagnostic")]
    reference = simple_film_reference()
    lines += ["", "## Part 2 — Within each culture: seen vs unseen films (accuracy %)", "",
              "Same test windows in both columns; same amount of training data summed over the 3 film folds. "
              "Each person is averaged over the folds and the seeds.", "",
              "| Culture | Model | Seen films | Unseen films | Drop | People dropping | Wilcoxon p | Runs complete |",
              "|---|---|---:|---:|---:|---|---:|---|"]
    drops = defaultdict(list)
    for culture in CULTURES:
        for label, base_name in models:
            name = base_name(culture)
            seen = film_people(FILM_ROOT, culture, name, "seen") if name else {}
            unseen = film_people(FILM_ROOT, culture, name, "unseen") if name else {}
            people = sorted(set(seen) & set(unseen))
            runs = [find_run(FILM_ROOT, culture, f"{name}__films_{c}{f}__pooled") if name else None
                    for c in ("seen", "unseen") for f in range(3)]
            complete = sum(coverage(r, n_people(culture))[:2] == (3, n_people(culture)) for r in runs)
            flag = f"{complete}/6" + ("" if complete == 6 else " ⚠")
            if len(people) < 2:
                lines.append(f"| {NAMES[culture]} | {label} | — | — | — | — | — | {flag} |")
                continue
            a, b = np.array([seen[p] for p in people]), np.array([unseen[p] for p in people])
            p_value = wilcoxon(a, b).pvalue if np.any(a - b) else 1.0
            drops[label].append(float((a - b).mean()))
            lines.append(f"| {NAMES[culture]} | {label} | {a.mean():.1f} | {b.mean():.1f} | {(a - b).mean():+.1f} | "
                         f"{int(np.sum(a > b))}/{len(people)} | {p_value:.3g} | {flag} |")
        if NAMES[culture] in reference:
            seen, unseen = reference[NAMES[culture]]
            lines.append(f"| {NAMES[culture]} | simple model, EEG (balanced acc., `clip_heldout_emotion.md`) | "
                         f"{seen:.1f} | {unseen:.1f} | {seen - unseen:+.1f} | | | |")
    lines += ["", "Average drop over the cultures with results: " + ("; ".join(
        f"{label} {np.mean(values):+.1f}" for label, values in drops.items()) or "none yet") +
        ". Predicted: ≈ +8 (strict models), ≈ +15 (test-label-selected)."]


def part3(chosen, lines):
    lines += ["", "## Part 3 — German ↔ French: shared vs other films (SCVAE-fixed ensemble, strict)", "",
              "Balanced accuracy (%) over negative and positive windows (neutral has no shared film).", "",
              "| Train → Test | Shared films | Other films | Shared − other | People higher on shared | Wilcoxon p | "
              "Runs complete |", "|---|---:|---:|---:|---|---:|---|"]
    for source, target in [("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")]:
        config = chosen[target]
        run = find_run(GRID_ROOT, target, f"{config}__from_{source}") if config else None
        split = shared_split(run, target) if run else {}
        flag = coverage_text(coverage(run, n_people(target), kind="proba"))
        if len(split) < 2:
            lines.append(f"| {NAMES[source]} → {NAMES[target]} | — | — | — | — | — | {flag} |")
            continue
        shared, other = np.array([v[0] for v in split.values()]), np.array([v[1] for v in split.values()])
        p_value = wilcoxon(shared, other).pvalue if np.any(shared - other) else 1.0
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {shared.mean():.1f} | {other.mean():.1f} | "
                     f"{(shared - other).mean():+.1f} | {int(np.sum(shared > other))}/{len(split)} | {p_value:.3g} | "
                     f"{flag} |")
    lines.append("\nPredicted: shared ≈ 5 points above other films.")


def main():
    scores = load(GRID_ROOT, pattern="*/*")
    configs = sorted(config_name(*g) for g in GRID)
    lines = ["# SCVAE-Net across cultures and on unseen films", "",
             "Spec: `docs/superpowers/specs/2026-10-09-scvae-crossculture-improvement-design.md`. "
             "Strict columns: no target label read before the final evaluation. "
             "**Test-label-selected** columns pick the checkpoint by test accuracy (the released code's policy); "
             "they are a labelled comparison, not accuracy.", "",
             "## Configuration chosen per test culture (from the other two cultures' transfers only)", "",
             "| Test culture | Chosen configuration | Validation acc. |", "|---|---|---:|"]
    chosen = {}
    for target in CULTURES:
        result = choose(scores, configs, target)
        chosen[target] = result[0] if result else None
        lines.append(f"| {NAMES[target]} | {result[0] if result else 'incomplete'} | "
                     f"{fmt(result[1] if result else None)} |")
    part1(scores, chosen, lines)
    part2(chosen, lines)
    part3(chosen, lines)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
