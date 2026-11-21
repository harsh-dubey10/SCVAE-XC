"""Checks for the DCCA-AM inputs.

    python -m src.experiments.test_dcca_data
"""

import numpy as np

from src.experiments.dcca_data import load_recordings, select, shared_trials
from src.experiments.film_identification import FILMS, TASKS

expected_recordings = {"SEED_CHN_MM": 36, "SEED_GER": 20, "SEED_FRA": 24}
data = {}
for culture, count in expected_recordings.items():
    recordings = load_recordings(culture)
    assert len(recordings) == count, (culture, len(recordings))
    for r in recordings:
        assert r.eeg.shape == (len(r.labels), 310) and r.eye.shape == (len(r.labels), 33)
        assert np.isfinite(r.eeg).all() and np.isfinite(r.eye).all()
        assert set(np.unique(r.labels)) == {0, 1, 2}
        # raw DE, not the per-window z-scored processed EEG
        assert abs(r.eeg.mean(axis=1)).max() > 1.0
    data[culture] = recordings

# New films: exactly the 5 shared films disappear from German and French recordings.
for culture in ("SEED_GER", "SEED_FRA"):
    kept = select(data[culture], drop_shared_of=culture)
    for before, after in zip(data[culture], kept):
        assert set(np.unique(before.trials)) - set(np.unique(after.trials)) == shared_trials(culture)
        assert len(after.labels) == (~np.isin(before.trials, list(shared_trials(culture)))).sum()

# Film labels: index of the film within the task, same windows as the shared trials.
for culture, column in (("SEED_GER", 1), ("SEED_FRA", 2)):
    films = select(data[culture], film_task="negative", culture=culture)
    for before, after in zip(data[culture], films):
        trial_of = [FILMS[name][column] for name in TASKS["negative"]]
        assert sorted(np.unique(after.labels)) == [0, 1, 2]
        assert len(after.labels) == np.isin(before.trials, trial_of).sum()
print("dcca_data checks passed")
