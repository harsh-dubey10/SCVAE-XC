"""Checks for NormFusion's pieces on small synthetic data and on the real loaders.

    python -m src.baselines.test_norm_fusion
"""

import inspect

import numpy as np

from src.baselines.norm_fusion import (C_GRID, EEG, W_GRID, Config, Culture, candidate_probas, config_proba,
                                       cv_cells, expert_probas, load_culture, select_config, self_train)
from src.experiments.clip_heldout_emotion import clip_folds
from src.experiments.dcca_data import shared_trials

rng = np.random.default_rng(0)


def synthetic(n_subjects=6, clips_per_emotion=3, windows=20, signal=1.0):
    xs, ys, clips, subjects = [], [], [], []
    for s in range(1, n_subjects + 1):
        for emotion in range(3):
            for k in range(clips_per_emotion):
                x = rng.normal(size=(windows, 343))
                x[:, :5] += signal * emotion  # EEG carries the emotion
                xs.append(x); ys.append(np.full(windows, emotion))
                clips.append(np.full(windows, 1 + emotion * clips_per_emotion + k)); subjects.append(np.full(windows, s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects)])


data = synthetic()

# Film-robust cells never share a person or a clip between training and validation; 9 cells.
cells = cv_cells(data.subjects, data.y, data.clips, "person_film")
assert len(cells) == 9 and len(cv_cells(data.subjects, data.y, data.clips, "person")) == 3
for train, val in cells:
    assert val.any() and train.any()
    assert not set(data.subjects[train]) & set(data.subjects[val])
    assert not set(data.clips[train]) & set(data.clips[val])
    assert set(data.y[val]) == {0, 1, 2}

# Candidate fusion: probabilities sum to 1; w = 1 equals the EEG expert alone.
probas = candidate_probas(data.x[:200], data.y[:200], data.x[200:260], seed=12, use_mlp=False, early=False)
assert len(probas) == len(C_GRID) ** 2 * len(W_GRID)
assert all(np.allclose(p.sum(axis=1), 1) for p in probas.values())
eeg_only = expert_probas(data.x[:200, EEG], data.y[:200], data.x[200:260, EEG], seed=12, use_mlp=False)
assert np.allclose(probas[Config(0.1, 1.0, 1.0)], eeg_only[0.1])
assert np.allclose(config_proba(data.x[:200], data.y[:200], data.x[200:260], Config(0.1, 1.0, 1.0), 12, False, False),
                   eeg_only[0.1])

# Selection is deterministic, ties go to the first candidate, and `keep` restricts the search.
best, scores = select_config(data, seed=12, use_mlp=False, early=False)
assert (best, scores) == select_config(data, seed=12, use_mlp=False, early=False)
top = max(scores.values())
assert best == next(cfg for cfg in scores if scores[cfg] == top)
eye_best, _ = select_config(data, seed=12, use_mlp=False, early=False, keep=lambda cfg: cfg.w == 0.0)
assert eye_best.w == 0.0
early_best, early_scores = select_config(data, seed=12, use_mlp=False, early=True)
assert early_best.c_eye is None and len(early_scores) == len(C_GRID)

# Self-training never receives test labels, and with no signal it stops after 0 rounds.
assert list(inspect.signature(self_train).parameters) == ["src_x", "src_y", "tgt_x", "cfg", "seed", "use_mlp", "early"]
noise = synthetic(signal=0.0)
proba, added = self_train(noise.x[:540], noise.y[:540], noise.x[540:], Config(0.001, 0.001, 0.5), 12, False, False)
assert proba.shape == (len(noise.x) - 540, 3) and added == []
proba, added = self_train(data.x[:540], data.y[:540], data.x[540:], Config(1.0, 1.0, 1.0), 12, True, False)
assert proba.shape == (len(data.x) - 540, 3) and len(added) >= 1

# Real loaders: normalisation changes the features; dropping shared films keeps every emotion in every fold.
german = load_culture("SEED_GER")
assert german.x.shape[1] == 343 and abs(german.x.mean()) < 0.05
raw = load_culture("SEED_GER", normalise=False)
assert not np.allclose(raw.x, german.x)
for culture in ("SEED_GER", "SEED_FRA"):
    reduced = load_culture(culture, drop_shared=True)
    assert not set(np.unique(reduced.clips)) & shared_trials(culture)
    folds = clip_folds(reduced.y, reduced.clips)
    assert all(set(reduced.y[folds == f]) == {0, 1, 2} for f in range(3))
# Each window carries its recording; every recording is one person's contiguous block in presentation order.
assert len(german.recordings) == len(german.x) and len(np.unique(german.recordings)) == 20
for r in np.unique(german.recordings):
    idx = np.flatnonzero(german.recordings == r)
    assert (np.diff(idx) == 1).all() and (np.diff(german.clips[idx]) >= 0).all()
    assert len(set(german.subjects[idx])) == 1
reduced = load_culture("SEED_GER", drop_shared=True)
assert len(reduced.recordings) == len(reduced.x)
print("norm_fusion checks passed")
