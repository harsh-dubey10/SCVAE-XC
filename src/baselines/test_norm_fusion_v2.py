"""Checks for NormFusion v2's pieces on small synthetic data and on the real loaders.

    python -m src.baselines.test_norm_fusion_v2
"""

import inspect

import numpy as np

from src.baselines.norm_fusion import C_GRID, Config, Culture, cv_cells, load_culture, select_config
from src.baselines.norm_fusion import self_train as v1_self_train
from src.baselines.norm_fusion_v2 import (EMOTION_WEIGHTS, LATE_WEIGHTS, OFFSET, SMOOTH_K, SMOOTHINGS, V1_WEIGHTS,
                                          VARIANT_KEEP, Candidate, choose, combine, cv_scores, experts,
                                          fit_candidate, fuse, grid_probas, self_train, smooth)

rng = np.random.default_rng(0)


def synthetic(n_subjects=6, clips_per_emotion=3, windows=20, signal=1.0, offset=0.0):
    """One recording per person, clips in a fixed mixed order, EEG (and weakly eye) carrying the emotion."""
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    order = [(e, k) for k in range(clips_per_emotion) for e in (2, 0, 1)]
    for s in range(1, n_subjects + 1):
        for emotion, k in order:
            x = rng.normal(size=(windows, 343)) + offset
            x[:, :5] += signal * emotion
            x[:, 310:315] += 0.5 * signal * emotion
            xs.append(x); ys.append(np.full(windows, emotion))
            clips.append(np.full(windows, 1 + emotion * clips_per_emotion + k))
            subjects.append(np.full(windows, s)); recordings.append(np.full(windows, 1000 + s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])


# Two cultures as one training set: ids stay apart, the first culture's ids are unchanged.
a, b = synthetic(4), synthetic(3)
both = combine([a, b])
assert len(both.x) == len(a.x) + len(b.x)
assert (both.subjects[:len(a.x)] == a.subjects).all() and (both.clips[:len(a.x)] == a.clips).all()
assert (both.subjects[len(a.x):] == b.subjects + OFFSET).all()
assert (both.recordings[len(a.x):] == b.recordings + OFFSET).all()
assert len(np.unique(both.subjects)) == 7 and len(np.unique(both.clips)) == 18
# CV cells over the combined set never put a person or a (culture, clip) on both sides.
for train, val in cv_cells(both.subjects, both.y, both.clips, "person_film"):
    assert not set(both.subjects[train]) & set(both.subjects[val])
    assert not set(both.clips[train]) & set(both.clips[val])


def brute_smooth(proba, recordings, k, causal):
    out = np.empty_like(proba)
    for r in np.unique(recordings):
        idx = np.flatnonzero(recordings == r)
        for j, i in enumerate(idx):
            lo, hi = (max(j - k + 1, 0), j + 1) if causal else (max(j - k // 2, 0), min(j + k // 2 + 1, len(idx)))
            out[i] = proba[idx[lo:hi]].mean(axis=0)
    return out


p = rng.dirichlet(np.ones(3), size=40)
recs = np.repeat([7, 8], 20)
# k = 1 is the identity (the very same array); every k matches a direct per-recording mean.
assert smooth(p, recs, 1) is p
for k in SMOOTH_K:
    for causal in (False, True):
        assert np.allclose(smooth(p, recs, k, causal), brute_smooth(p, recs, k, causal)), (k, causal)
        assert np.allclose(smooth(p, recs, k, causal).sum(axis=1), 1)
# Only order within one recording matters: changing recording 8 leaves recording 7 unchanged, and
# interleaving the two recordings in the array gives the same per-window result.
q = p.copy(); q[20:] = rng.dirichlet(np.ones(3), size=20)
assert np.allclose(smooth(q, recs, 9)[:20], smooth(p, recs, 9)[:20])
order = np.argsort(np.tile(np.arange(20), 2), kind="stable")  # 7, 8, 7, 8, ... keeping each recording's order
assert np.allclose(smooth(p[order], recs[order], 9), smooth(p, recs, 9)[order])
# k longer than the recording: the mean of the windows that exist, no NaN.
short = smooth(p[:2], np.array([1, 1]), 15)
assert np.allclose(short, p[:2].mean(axis=0)) and not np.isnan(short).any()
# Causal smoothing never looks ahead: the first window is unchanged, changing the last window changes nothing before it.
assert np.allclose(smooth(p, recs, 5, True)[0], p[0])
r = p.copy(); r[19] = [1, 0, 0]
assert np.allclose(smooth(r, recs, 5, True)[:19], smooth(p, recs, 5, True)[:19])

# Fusion grids: 27 per-emotion triples, v1's 5 equal triples, 29 in the sorted union with equal triples ascending.
assert len(EMOTION_WEIGHTS) == 27 and len(V1_WEIGHTS) == 5 and len(LATE_WEIGHTS) == 29
equal = [w for w in LATE_WEIGHTS if len(set(w)) == 1]
assert [w[0] for w in equal] == [0.0, 0.25, 0.5, 0.75, 1.0]
eeg, eye = rng.dirichlet(np.ones(3), size=50), rng.dirichlet(np.ones(3), size=50)
for weights in LATE_WEIGHTS:
    assert np.allclose(fuse(eeg, eye, weights).sum(axis=1), 1), weights
# Equal weights give v1's fusion bit for bit; all-EEG / all-eye need only that expert.
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    assert np.array_equal(fuse(eeg, eye, (w, w, w)), w * eeg + (1 - w) * eye)
# Real expert probabilities are float32: equal weights must stay float32 and match v1 bit for bit.
eeg32, eye32 = eeg.astype(np.float32), eye.astype(np.float32)
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    fused = fuse(eeg32, eye32, (w, w, w))
    assert fused.dtype == np.float32 and np.array_equal(fused, w * eeg32 + (1 - w) * eye32), w
assert np.array_equal(fuse(eeg, None, (1.0, 1.0, 1.0)), eeg) and np.array_equal(fuse(None, eye, (0.0, 0.0, 0.0)), eye)
# Per-emotion weights act per class: (1, 0, 0) takes negative from EEG, neutral and positive from eye, renormalised.
mixed = np.column_stack([eeg[:, 0], eye[:, 1], eye[:, 2]])
assert np.allclose(fuse(eeg, eye, (1.0, 0.0, 0.0)), mixed / mixed.sum(axis=1, keepdims=True))

# The grid: late fusion (4 x 4 C x 29 weight triples x 9 smoothings), then early fusion (4 C x 9 smoothings).
assert SMOOTHINGS == [(1, False), (3, False), (5, False), (9, False), (15, False),
                      (3, True), (5, True), (9, True), (15, True)]
data = synthetic()
ex = experts(data.x[:540], data.y[:540], data.x[540:], seed=12, use_mlp=False)
grid = list(grid_probas(ex, data.recordings[540:]))
assert len(grid) == len(C_GRID) ** 2 * len(LATE_WEIGHTS) * 9 + len(C_GRID) * 9 == 4212
assert len({cand for cand, _ in grid}) == len(grid)
first_early = next(i for i, (cand, _) in enumerate(grid) if cand.fusion == "early")
assert all(cand.fusion == "late" for cand, _ in grid[:first_early])
lookup = dict(grid)
cand = Candidate("late", 0.1, 1.0, (1.0, 0.0, 0.5), 5, False)
assert np.allclose(lookup[cand], smooth(fuse(ex["eeg"][0.1], ex["eye"][1.0], (1.0, 0.0, 0.5)), data.recordings[540:], 5))
assert np.array_equal(lookup[Candidate("early", 0.01, None, None, 1, False)], ex["early"][0.01])

# Variant subsets.
fair_grid = [c for c, _ in grid if VARIANT_KEEP["fair"](c)]
assert len(fair_grid) == len(C_GRID) ** 2 * 27 * 5 + len(C_GRID) * 5 == 2180
assert not any(c.causal for c in fair_grid)
assert all(c.k == 1 for c, _ in grid if VARIANT_KEEP["no_smoothing"](c))
assert all(c.causal or c.k == 1 for c, _ in grid if VARIANT_KEEP["causal"](c))
assert all(c.fusion == "late" and c.weights in V1_WEIGHTS for c, _ in grid if VARIANT_KEEP["v1_fusion"](c))
v1_grid = [c for c, _ in grid if VARIANT_KEEP["v1"](c)]
assert len(v1_grid) == 80 and all(c.k == 1 and not c.causal for c in v1_grid)
# Within the v1 subset, grid order is v1's order: (C_EEG, C_eye, w ascending).
assert [(c.c_eeg, c.c_eye, c.weights[0]) for c in v1_grid] == \
       [(ce, cy, w) for ce in C_GRID for cy in C_GRID for w in (0.0, 0.25, 0.5, 0.75, 1.0)]

# Cross-validation: deterministic, every candidate scored; the v1 subset gives v1's scores and v1's choice.
scores = cv_scores(data, seed=12, use_mlp=False)
assert list(scores) == [c for c, _ in grid] and scores == cv_scores(data, seed=12, use_mlp=False)
v1_best, v1_scores = select_config(data, seed=12, use_mlp=False, early=False)
for c in v1_grid:
    assert abs(scores[c] - v1_scores[(c.c_eeg, c.c_eye, c.weights[0])]) < 1e-12, c
best_v1 = choose(scores, VARIANT_KEEP["v1"])
assert (best_v1.c_eeg, best_v1.c_eye, best_v1.weights[0]) == tuple(v1_best)
# choose: best inside the subset, first in grid order on ties.
tie = {Candidate("late", 1, 1, (0.0, 0.0, 0.0), 1, False): 0.5, Candidate("late", 1, 1, (0.5, 0.5, 0.5), 1, False): 0.5,
       Candidate("early", 1, None, None, 3, False): 0.9}
assert choose(tie, VARIANT_KEEP["v1"]).weights == (0.0, 0.0, 0.0)
assert choose(tie, VARIANT_KEEP["fair"]).fusion == "early"

# fit_candidate equals the corresponding grid entry before smoothing.
cand = Candidate("late", 0.1, 1.0, (1.0, 0.0, 0.5), 5, False)
assert np.allclose(fit_candidate(data.x[:540], data.y[:540], data.x[540:], cand, 12, False),
                   fuse(ex["eeg"][0.1], ex["eye"][1.0], (1.0, 0.0, 0.5)))
early = Candidate("early", 0.01, None, None, 3, False)
assert np.allclose(fit_candidate(data.x[:540], data.y[:540], data.x[540:], early, 12, False), ex["early"][0.01])

# Self-training never receives test labels; with k = 1 and equal weights it is v1's self-training exactly.
assert list(inspect.signature(self_train).parameters) == ["src_x", "src_y", "tgt_x", "tgt_recordings", "cand",
                                                          "seed", "use_mlp"]
src, tgt = synthetic(6), synthetic(4, offset=0.3)
src32, tgt32 = src.x.astype(np.float32), tgt.x.astype(np.float32)  # real features are float32
for w in (0.5, 1.0):
    mine, mine_added = self_train(src32, src.y, tgt32, tgt.recordings, Candidate("late", 1.0, 1.0, (w, w, w), 1, False),
                                  12, True)
    theirs, theirs_added = v1_self_train(src32, src.y, tgt32, Config(1.0, 1.0, w), 12, True, False)
    assert mine_added == theirs_added and np.array_equal(mine, theirs), w
# With k > 1 the returned probabilities are smoothed: valid, and different from the k = 1 run.
smoothed, added = self_train(src.x, src.y, tgt.x, tgt.recordings, Candidate("late", 1.0, 1.0, (1.0, 0.5, 0.5), 9, False),
                             12, False)
plain, _ = self_train(src.x, src.y, tgt.x, tgt.recordings, Candidate("late", 1.0, 1.0, (1.0, 0.5, 0.5), 1, False),
                      12, False)
assert smoothed.shape == (len(tgt.x), 3) and np.allclose(smoothed.sum(axis=1), 1) and not np.allclose(smoothed, plain)
# No signal: self-training stops after 0 rounds.
noise_src, noise_tgt = synthetic(6, signal=0.0), synthetic(4, signal=0.0)
assert self_train(noise_src.x, noise_src.y, noise_tgt.x, noise_tgt.recordings,
                  Candidate("late", 0.001, 0.001, (0.5, 0.5, 0.5), 1, False), 12, False)[1] == []

# Real data: in two-culture training each film fold holds films of both cultures.
from src.experiments.clip_heldout_emotion import clip_folds
real = combine([load_culture("SEED_CHN_MM"), load_culture("SEED_GER")])
folds = clip_folds(real.y, real.clips)
for f in range(3):
    held = np.unique(real.clips[folds == f])
    assert (held < OFFSET).any() and (held >= OFFSET).any()
print("norm_fusion_v2 checks passed")
