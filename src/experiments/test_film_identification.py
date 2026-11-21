"""Checks for film_identification: the shared-film mapping and the permutation null.

    python -m src.experiments.test_film_identification
"""

from pathlib import Path

import numpy as np

from src.experiments.film_identification import CULTURES, FILMS, TASKS, block_permutations, load_culture

# Windows per recording (4 s) for each shared film, from the clip durations in the
# German and French information sheets; a wrong trial index shows up as a mismatch.
WINDOWS = {"German": [55, 34, 105, 36, 43], "French": [75, 66, 19, 25, 32]}

for name, (root, column) in CULTURES.items():
    x, films, subjects, recordings = load_culture(Path("data/processed") / root, column)
    assert x.shape[1] == 343
    for recording in np.unique(recordings):
        counts = np.bincount(films[recordings == recording], minlength=len(FILMS))
        assert counts.tolist() == WINDOWS[name], (name, recording, counts)

# The permutation keeps each block intact and only swaps films within a recording.
rng = np.random.default_rng(0)
films = np.array([0, 0, 1, 1, 1, 0, 1])
recordings = np.array([1, 1, 1, 1, 1, 2, 2])
for shuffled in block_permutations(films, recordings, [0, 1], rng, 20):
    for r in (1, 2):
        for f in (0, 1):
            assert len(set(shuffled[(recordings == r) & (films == f)])) == 1
        assert sorted(set(shuffled[recordings == r])) == [0, 1]

assert set(TASKS["all_5"]) == set(FILMS)
print("film_identification checks passed")
