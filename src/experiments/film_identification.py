"""Cross-culture film identification: does a model trained on Germans know which film a French person watches?

SEED-GER and SEED-FRA share five FilmStim excerpts (same film, overlapping
scenes; excerpt lengths differ). Within one emotion every candidate film has the
same label, so identifying the film above chance means the signals carry
film-specific information that transfers across people and cultures: a
stimulus confound for emotion recognition.

Tasks (balanced accuracy, chance = 1 / number of films):
  negative     Dead Man Walking / A Perfect World / Schindler's List
  negative_2   Dead Man Walking / A Perfect World (Schindler's excerpts differ most: 7 vs 1.3 min)
  positive     The Dinner Game / The Visitors
  all_5        all five films

Each recording is standardised with its own statistics over all of its windows
(label-free, as in the normalised baselines) before the shared-film windows are
selected. The model (standardiser + logistic regression, C = 0.1, balanced
class weights) is fixed in advance; the test culture is only predicted. The
p-value permutes which film each (recording, film) block of the test culture
is labelled as, within each recording, so the predictions stay fixed and the
block structure is preserved.

Control for session timing: Dead Man Walking is shown before A Perfect World in
both cultures, so a negative_2 model could learn "early vs late". It is applied
to the other negative films of the test culture (shown at various times); if it
tracked timing, late films would be called A Perfect World.

    python -m src.experiments.film_identification   # -> results/tables/film_identification.md
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.experiments.train_scvae_reference import available_subjects, normalize_recording, sessions_in

NEGATIVE, POSITIVE = 0, 2
# Film -> (emotion label, processed trial index in SEED-GER, in SEED-FRA). German
# stimuli 2 and 19 were not used, so German stimulus n > 2 is trial n - 1 (n > 19: n - 2).
FILMS = {
    "Dead Man Walking": (NEGATIVE, 3, 7),
    "A Perfect World": (NEGATIVE, 16, 10),
    "Schindler's List": (NEGATIVE, 2, 16),
    "The Dinner Game": (POSITIVE, 9, 5),
    "The Visitors": (POSITIVE, 18, 9),
}
CULTURES = {"German": ("SEED_GER_EEG_EYE", 1), "French": ("SEED_FRA_EEG_EYE", 2)}
TASKS = {
    "negative": ["Dead Man Walking", "A Perfect World", "Schindler's List"],
    "negative_2": ["Dead Man Walking", "A Perfect World"],
    "positive": ["The Dinner Game", "The Visitors"],
    "all_5": list(FILMS),
}
# Negative films that only one culture saw: processed trial index -> (name, start minute).
OTHER_NEGATIVE = {
    "German": {6: ("The Piano", 24), 11: ("Saving Private Ryan", 42), 15: ("Misery", 55)},
    "French": {2: ("City of Angels", 4), 4: ("E.T.", 11), 15: ("Dangerous Mind", 46), 19: ("Life is Beautiful", 56)},
}
MODALITIES = {"EEG + eye": slice(0, 343), "EEG": slice(0, 310), "eye": slice(310, 343)}


def load_culture(root, column):
    """Shared-film windows of one culture: features, film index, subject, recording id."""
    xs, films, subjects, recordings = [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = normalize_recording(torch.stack(data["data"]).float().flatten(start_dim=1)).numpy()
            trials, labels = np.asarray(data["trials"]), np.asarray(data["labels"])
            for film, (emotion, *trial_of) in enumerate(FILMS.values()):
                rows = trials == trial_of[column - 1]
                if not rows.any() or set(labels[rows]) != {emotion}:
                    raise ValueError(f"{root} s{session} subject{subject}: {list(FILMS)[film]} "
                                     f"has labels {set(labels[rows])}, expected {emotion}")
                xs.append(x[rows])
                films.append(np.full(rows.sum(), film))
                subjects.append(np.full(rows.sum(), subject))
                recordings.append(np.full(rows.sum(), 100 * session + subject))
    return [np.concatenate(v) for v in (xs, films, subjects, recordings)]


def load_trials(root, trials):
    """Windows of the given processed trials, each recording standardised over all its windows."""
    windows = {t: [] for t in trials}
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = normalize_recording(torch.stack(data["data"]).float().flatten(start_dim=1)).numpy()
            for t in trials:
                windows[t].append(x[np.asarray(data["trials"]) == t])
    return {t: np.concatenate(v) for t, v in windows.items()}


def timing_control(train, test_root, test, column, columns):
    """Fraction of each test-culture negative film that a DMW-vs-APW model calls 'A Perfect World'."""
    dmw, apw = list(FILMS).index("Dead Man Walking"), list(FILMS).index("A Perfect World")
    x, films = train[0], train[1]
    keep = np.isin(films, [dmw, apw])
    model = fit(x[keep][:, columns], films[keep] == apw)
    shared = {FILMS[name][column]: (name, None) for name in ("Dead Man Walking", "A Perfect World")}
    trials = {**shared, **OTHER_NEGATIVE[test]}
    windows = load_trials(test_root, trials)
    return {name: float(model.predict(windows[t][:, columns]).mean()) for t, (name, _) in trials.items()}


def fit(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))
        return model.fit(x, y)


def per_subject_accuracy(true, predicted, subjects):
    return np.array([balanced_accuracy_score(true[subjects == s], predicted[subjects == s])
                     for s in np.unique(subjects)])


def block_permutations(films, recordings, task_films, rng, n):
    """Relabel each (recording, film) block with a film permuted within its recording."""
    blocks = {(r, f): np.where((recordings == r) & (films == f))[0]
              for r in np.unique(recordings) for f in task_films}
    for _ in range(n):
        shuffled = films.copy()
        for r in np.unique(recordings):
            for f, g in zip(task_films, rng.permutation(task_films)):
                shuffled[blocks[(r, f)]] = g
        yield shuffled


def evaluate(train, test, task, columns, rng, n_permutations):
    task_films = [list(FILMS).index(name) for name in TASKS[task]]
    (xa, fa, _, _), (xb, fb, sb, rb) = train, test
    keep_a, keep_b = np.isin(fa, task_films), np.isin(fb, task_films)
    model = fit(xa[keep_a][:, columns], fa[keep_a])
    predicted = model.predict(xb[keep_b][:, columns])
    true, subjects, recordings = fb[keep_b], sb[keep_b], rb[keep_b]
    accuracy = per_subject_accuracy(true, predicted, subjects)
    null = np.array([per_subject_accuracy(shuffled, predicted, subjects).mean()
                     for shuffled in block_permutations(true, recordings, task_films, rng, n_permutations)])
    return {
        "accuracy": float(accuracy.mean()),
        "per_subject": accuracy.round(4).tolist(),
        "subjects_above_chance": int((accuracy > 1 / len(task_films)).sum()),
        "chance": 1 / len(task_films),
        "null_mean": float(null.mean()),
        "p": float((1 + (null >= accuracy.mean()).sum()) / (1 + n_permutations)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/tables/film_identification.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    data = {name: load_culture(args.data_root / root, column) for name, (root, column) in CULTURES.items()}

    results = []
    for (source, target), task, modality in itertools.product(
            itertools.permutations(CULTURES, 2), TASKS, MODALITIES):
        result = evaluate(data[source], data[target], task, MODALITIES[modality], rng, args.permutations)
        results.append({"train": source, "test": target, "task": task, "modality": modality, **result})
        print(f"{source} -> {target} {task:10s} {modality:9s} {100 * result['accuracy']:5.1f}% "
              f"(chance {100 * result['chance']:.0f}, p = {result['p']:.4f})")

    report = ["# Cross-culture film identification (SEED-GER <-> SEED-FRA shared films)", "",
              "Balanced accuracy (%), mean over test people; above = test people above chance; "
              f"p from {args.permutations} block permutations of the test films (one-sided).", ""]
    for task in TASKS:
        report += [f"## {task}: {', '.join(TASKS[task])} (chance {100 / len(TASKS[task]):.0f}%)", "",
                   "| Train → Test | Modality | Accuracy | Above chance | p |", "|---|---|---:|---:|---:|"]
        for r in (r for r in results if r["task"] == task):
            report.append(f"| {r['train']} → {r['test']} | {r['modality']} | {100 * r['accuracy']:.1f} | "
                          f"{r['subjects_above_chance']}/{len(r['per_subject'])} | {r['p']:.4f} |")
        report.append("")

    controls = []
    report += ["## Timing control: fraction of each test-culture negative film called 'A Perfect World'", "",
               "By a Dead Man Walking vs A Perfect World model trained on the other culture. "
               "Start minute in the test session in brackets.", ""]
    for source, target in itertools.permutations(CULTURES, 2):
        root, column = CULTURES[target]
        starts = {**{name: None for name in ("Dead Man Walking", "A Perfect World")},
                  **dict(OTHER_NEGATIVE[target].values())}
        report += [f"| {source} → {target} | " + " | ".join(
            f"{name}" + (f" ({start})" if start is not None else "") for name, start in starts.items()) + " |",
                   "|---|" + "---:|" * len(starts)]
        for modality, columns in MODALITIES.items():
            fractions = timing_control(data[source], args.data_root / root, target, column, columns)
            controls.append({"train": source, "test": target, "modality": modality, "called_apw": fractions})
            report.append(f"| {modality} | " + " | ".join(f"{fractions[name]:.2f}" for name in starts) + " |")
        report.append("")
    args.output.write_text("\n".join(report))
    args.output.with_suffix(".json").write_text(json.dumps({"tasks": results, "timing_control": controls}, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
