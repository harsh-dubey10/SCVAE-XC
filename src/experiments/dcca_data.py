"""DCCA-AM inputs as the paper used them: raw DE-LDS EEG (310) + raw eye features (33).

The EEG comes from the BCMI ``eeg_used_4s`` files (the processed EEG is
z-scored per window, which the paper did not do). The eye features, labels
and clip ids come from the processed ``<culture>_EEG_EYE`` files, whose eye
columns are in raw units. Each recording's two sources must agree in length
and labels.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from src.experiments.film_identification import FILMS, TASKS
from src.preprocessing.convert_seed_crossculture import load_session

RAW_EEG = {
    "SEED_CHN_MM": Path("data/raw/SEED_CHN_MM/02-EEG-DE-feature/eeg_used_4s"),
    "SEED_GER": Path("data/raw/SEED_GER/02-EEG-DE-features/eeg_used_4s"),
    "SEED_FRA": Path("data/raw/SEED_FRA/02-EEG-DE-features/eeg_used_4s"),
}
PROCESSED = Path("data/processed")
FILM_COLUMN = {"SEED_GER": 1, "SEED_FRA": 2}  # position of the trial index in FILMS values


@dataclass
class Recording:
    subject: int
    session: int
    eeg: np.ndarray
    eye: np.ndarray
    labels: np.ndarray
    trials: np.ndarray


def load_recordings(culture):
    recordings = []
    for path in sorted(RAW_EEG[culture].glob("*_*.npz"), key=lambda p: tuple(map(int, p.stem.split("_")))):
        subject, session = map(int, path.stem.split("_"))
        features, labels, _ = load_session(path)
        processed = torch.load(PROCESSED / f"{culture}_EEG_EYE" / str(session) / f"subject{subject}.pth",
                               weights_only=True)
        x = torch.stack(processed["data"]).numpy()
        if len(x) != len(features) or list(labels) != list(processed["labels"]):
            raise ValueError(f"{culture} {path.stem}: raw EEG and processed eye disagree")
        recordings.append(Recording(subject, session, features.reshape(len(features), -1).astype(np.float32),
                                    x[:, 310:].astype(np.float32), np.asarray(labels), np.asarray(processed["trials"])))
    return recordings


def shared_trials(culture):
    return {values[FILM_COLUMN[culture]] for values in FILMS.values()}


def _keep(recording, rows, labels=None):
    return replace(recording, eeg=recording.eeg[rows], eye=recording.eye[rows],
                   labels=recording.labels[rows] if labels is None else labels, trials=recording.trials[rows])


def select(recordings, drop_shared_of=None, film_task=None, culture=None):
    """Drop the shared films, or keep only one film task's windows labelled by film index."""
    out = []
    for r in recordings:
        if drop_shared_of is not None:
            out.append(_keep(r, ~np.isin(r.trials, list(shared_trials(drop_shared_of)))))
        elif film_task is not None:
            trial_to_film = {FILMS[name][FILM_COLUMN[culture]]: i for i, name in enumerate(TASKS[film_task])}
            rows = np.isin(r.trials, list(trial_to_film))
            out.append(_keep(r, rows, np.array([trial_to_film[t] for t in r.trials[rows]])))
        else:
            out.append(r)
    return out
