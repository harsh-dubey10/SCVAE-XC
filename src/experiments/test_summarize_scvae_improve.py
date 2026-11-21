"""Checks for the SCVAE study summary.

    .venv/bin/python -m src.experiments.test_summarize_scvae_improve
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.experiments.summarize_scvae_improve import ensemble, film_people, find_run, per_subject, shared_split

root = Path(tempfile.mkdtemp())
run = root / "SEED_GER" / "strict__bn_source__sval3__subjnorm__from_SEED_CHN_MM"
labels = np.array([0, 2, 0, 2])
trials = np.array([3, 9, 6, 13])  # German trials 3 and 9 are shared films (negative, positive); 6 and 13 are not
# Mean of the two seeds predicts [0, 2, 2, 0]: right on windows 0-1 (shared), wrong on 2-3 (other films).
seed_probas = {12: [[.9, .05, .05], [.1, .1, .8], [.2, .1, .7], [.3, .3, .4]],
               34: [[.5, .1, .4], [.3, .1, .6], [.1, .2, .7], [.6, .3, .1]]}
for seed, proba in seed_probas.items():
    directory = run / "session_all" / f"seed_{seed}"
    directory.mkdir(parents=True)
    np.savez(directory / "subject_01_proba.npz", proba=np.array(proba, dtype=np.float32), labels=labels, trials=trials)
    (directory / "subject_01.json").write_text(json.dumps({"test_subject": 1, "accuracy": 0.5, "macro_f1": 0.4}))

assert ensemble(run)[1][0] == 50.0
assert shared_split(run, "SEED_GER") == {1: (100.0, 0.0)}
assert per_subject(run) == {1: (50.0, 40.0)}
assert find_run(root, "SEED_GER", "strict__bn_source__sval__subjnorm__from_SEED_CHN_MM") == run
assert find_run(root, "SEED_GER", "strict__bn_source__sval__subjnorm__from_SEED_FRA") is None
assert per_subject(root / "missing") == {} and ensemble(root / "missing") == {}

# film_people: average over the three folds; a person missing a fold is dropped.
for fold, accuracies in enumerate([(0.6, 0.3), (0.9, 0.3), (0.3, None)]):
    directory = root / "films" / "SEED_FRA" / f"strict__bn_source__films_seen{fold}__pooled" / "session_all" / "seed_12"
    directory.mkdir(parents=True)
    for subject, accuracy in zip((1, 2), accuracies):
        if accuracy is not None:
            (directory / f"subject_{subject:02d}.json").write_text(
                json.dumps({"test_subject": subject, "accuracy": accuracy, "macro_f1": accuracy}))
people = film_people(root / "films", "SEED_FRA", "strict__bn_source", "seen")
assert list(people) == [1] and abs(people[1] - 60.0) < 1e-9
print("Summary tests passed.")

# Review I2: coverage = (seeds, people present in every seed, expected people).
from src.experiments.summarize_scvae_improve import coverage, coverage_text

partial = root / "partial"
for seed, subjects in [(12, (1, 2)), (34, (1,))]:
    directory = partial / "session_all" / f"seed_{seed}"
    directory.mkdir(parents=True)
    for subject in subjects:
        (directory / f"subject_{subject:02d}.json").write_text(json.dumps({"test_subject": subject}))
assert coverage(partial, 2) == (2, 1, 2)
assert coverage_text(coverage(partial, 2)) == "2 seeds, 1/2 people ⚠"
assert coverage_text(coverage(run, 1, kind="proba")) == "2 seeds, 1/1 people ⚠"  # 3 seeds expected
assert coverage(root / "missing", 4) == (0, 0, 4)
print("Coverage tests passed.")
