"""Checks for run_cnn_protonet on small synthetic data.

    python -m src.experiments.test_run_cnn_protonet
"""

import numpy as np
import torch

from src.experiments.run_calibration import film_pool, split
from src.experiments.run_cnn_protonet import (CHANNELS, GRID_C, GRID_R, Encoder, ProtoNet, episode, proto_proba,
                                              to_grid)

rng = np.random.default_rng(3)

# Grid: 62 distinct cells, and every DE value lands in its own channel's cell for its own band.
assert len(CHANNELS) == 62 and len(set(zip(GRID_R.tolist(), GRID_C.tolist()))) == 62
eeg = torch.arange(310, dtype=torch.float32).unsqueeze(0)
grid = to_grid(eeg)
for ch in (0, 27, 61):
    for band in range(5):
        assert grid[0, band, GRID_R[ch], GRID_C[ch]] == ch * 5 + band
assert grid.count_nonzero() == 309  # value 0 (channel 0, band 0) is the only zero placed
for kind in ("mlp", "cnn"):
    assert Encoder(kind)(torch.randn(4, 343)).shape == (4, 64)

# Synthetic people: two people, two sessions, 4 films per emotion, 10 windows per film per session.
xs, ys, clips, subjects, recordings = [], [], [], [], []
for person in (1, 2):
    for session in (1, 2):
        for film in range(1, 13):
            xs.append(rng.normal(size=(10, 343)).astype(np.float32)); ys.append(np.full(10, (film - 1) // 4))
            clips.append(np.full(10, film)); subjects.append(np.full(10, person))
            recordings.append(np.full(10, 1000 * session + person))
x, y, clips, subjects, recordings = (np.concatenate(v) for v in (xs, ys, clips, subjects, recordings))

# Episodes: support from one person's first session, query from the same person's other films only.
for _ in range(20):
    support, query = episode(rng, y, clips, subjects, recordings, np.array([1, 2]))
    people = set(subjects[np.concatenate(support + [query])].tolist())
    assert len(people) == 1
    assert all(set((recordings[s] // 1000).tolist()) == {1} and set(y[s].tolist()) == {e} for e, s in enumerate(support))
    assert not set(clips[np.concatenate(support)].tolist()) & set(clips[query].tolist())

# Prototype predictions read labels on calibration windows only: shuffling the test labels changes nothing.
model = ProtoNet("cnn").eval()
with torch.no_grad():
    emb = model.encoder(torch.tensor(x)).numpy()
pool = film_pool(y, clips, subjects, person=1, draw=0)
calib, test = split(y, clips, subjects, recordings, person=1, pool=pool, n=1)
shuffled = y.copy(); shuffled[test] = rng.permutation(shuffled[test])
a, b = proto_proba(model, emb, y, calib, test), proto_proba(model, emb, shuffled, calib, test)
assert np.array_equal(a, b) and a.shape == (test.sum(), 3) and np.allclose(a.sum(1), 1, atol=1e-5)
print("ok")
