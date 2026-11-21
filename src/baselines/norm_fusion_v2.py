"""NormFusion v2: v1 plus temporal smoothing, per-emotion fusion weights, early fusion as a candidate,
and training on two cultures.

Reuses v1's experts, cross-validation cells and self-training rule (src/baselines/norm_fusion.py).
Design: docs/superpowers/specs/2026-10-01-norm-fusion-v2-design.md.
"""

import itertools
from collections import defaultdict, namedtuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.baselines.norm_fusion import (C_GRID, EEG, EYE, ST_MIN, ST_ROUNDS, ST_THRESHOLD, W_GRID, Culture, cv_cells,
                                       expert_probas)

OFFSET = 1_000_000  # added per culture index to clip, person and recording ids in two-culture training
SMOOTH_K = (1, 3, 5, 9, 15)
EMOTION_WEIGHTS = tuple(itertools.product((0.0, 0.5, 1.0), repeat=3))  # (w_negative, w_neutral, w_positive)
V1_WEIGHTS = tuple((w, w, w) for w in W_GRID)
LATE_WEIGHTS = tuple(sorted(set(EMOTION_WEIGHTS) | set(V1_WEIGHTS)))
SMOOTHINGS = [(k, False) for k in SMOOTH_K] + [(k, True) for k in SMOOTH_K if k > 1]
Candidate = namedtuple("Candidate", "fusion c_eeg c_eye weights k causal")

_v2_fusion = lambda c: c.fusion == "early" or c.weights in EMOTION_WEIGHTS
_v1_fusion = lambda c: c.fusion == "late" and c.weights in V1_WEIGHTS
VARIANT_KEEP = {
    "fair": lambda c: _v2_fusion(c) and not c.causal,
    "no_smoothing": lambda c: _v2_fusion(c) and c.k == 1,
    "causal": lambda c: _v2_fusion(c) and (c.causal or c.k == 1),
    "v1_fusion": lambda c: _v1_fusion(c) and not c.causal,
    "v1": lambda c: _v1_fusion(c) and c.k == 1,
}


def combine(cultures):
    """Several cultures as one training set; films are (culture, clip), people and recordings likewise."""
    shift = lambda field: np.concatenate([getattr(c, field) + OFFSET * i for i, c in enumerate(cultures)])
    return Culture(np.concatenate([c.x for c in cultures]), np.concatenate([c.y for c in cultures]),
                   shift("clips"), shift("subjects"), shift("recordings"))


def smooth(proba, recordings, k, causal=False):
    """Each window's probabilities averaged over k windows of its own recording, in array order.

    Centred: the window and k // 2 on each side; causal: the window and the k - 1 before it. Edges use the
    windows that exist. Only the order of windows is used (no clip boundaries, no labels). k = 1: unchanged.
    """
    if k == 1:
        return proba
    out = np.empty_like(proba)
    for r in np.unique(recordings):
        idx = np.flatnonzero(recordings == r)
        cumulative = np.vstack([np.zeros((1, proba.shape[1])), np.cumsum(proba[idx], axis=0)])
        i = np.arange(len(idx))
        lo = np.maximum(i - (k - 1 if causal else k // 2), 0)
        hi = np.minimum(i + (1 if causal else k // 2 + 1), len(idx))
        out[idx] = (cumulative[hi] - cumulative[lo]) / (hi - lo)[:, None]
    return out


def fuse(eeg, eye, weights):
    """p_c ∝ w_c p_EEG,c + (1 - w_c) p_eye,c, renormalised; equal w_c already sum to 1 and are left as in v1.

    Equal weights use a Python scalar, as v1 does, so float32 probabilities stay float32 and match v1 exactly.
    """
    equal = len(set(weights)) == 1
    w = weights[0] if equal else np.asarray(weights)
    proba = 0
    if np.any(w > 0):
        proba = proba + w * eeg
    if np.any(w < 1):
        proba = proba + (1 - w) * eye
    return proba if equal else proba / proba.sum(axis=1, keepdims=True)


def experts(train_x, train_y, eval_x, seed, use_mlp=True):
    """Per-C probabilities on eval_x of the EEG expert, the eye expert and the early-fusion expert (343 features)."""
    return {"eeg": expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp),
            "eye": expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp),
            "early": expert_probas(train_x, train_y, eval_x, seed, use_mlp)}


def grid_probas(ex, recordings):
    """(candidate, smoothed probabilities) for the whole grid, in grid order: late fusion over
    C_EEG x C_eye x weights x smoothing, then early fusion over C x smoothing."""
    for c_eeg in C_GRID:
        for c_eye in C_GRID:
            for weights in LATE_WEIGHTS:
                fused = fuse(ex["eeg"][c_eeg], ex["eye"][c_eye], weights)
                for k, causal in SMOOTHINGS:
                    yield Candidate("late", c_eeg, c_eye, weights, k, causal), smooth(fused, recordings, k, causal)
    for c in C_GRID:
        for k, causal in SMOOTHINGS:
            yield Candidate("early", c, None, None, k, causal), smooth(ex["early"][c], recordings, k, causal)


def cv_scores(data, seed, use_mlp=True):
    """Mean validation balanced accuracy over v1's 9 people-and-films cells for every candidate, in grid order.

    Each cell's model predicts every window of the held-out people's recordings, the predictions are smoothed
    within each recording, and only the held-out fold's windows are scored.
    """
    scores = defaultdict(list)
    for train, val in cv_cells(data.subjects, data.y, data.clips, "person_film"):
        held = np.isin(data.subjects, np.unique(data.subjects[val]))
        scored, truth = val[held], data.y[held][val[held]]
        ex = experts(data.x[train], data.y[train], data.x[held], seed, use_mlp)
        for cand, proba in grid_probas(ex, data.recordings[held]):
            scores[cand].append(balanced_accuracy_score(truth, proba[scored].argmax(axis=1)))
    return {cand: float(np.mean(v)) for cand, v in scores.items()}


def choose(scores, keep):
    """The best-scoring candidate that `keep` allows; the first in grid order on ties."""
    allowed = {cand: s for cand, s in scores.items() if keep(cand)}
    return max(allowed, key=allowed.get)


def fit_candidate(train_x, train_y, eval_x, cand, seed, use_mlp=True):
    """Unsmoothed probabilities on eval_x for one candidate, fitting only the experts it uses."""
    if cand.fusion == "early":
        return expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=(cand.c_eeg,))[cand.c_eeg]
    w = np.asarray(cand.weights)
    eeg = expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp, cs=(cand.c_eeg,))[cand.c_eeg] \
        if (w > 0).any() else None
    eye = expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp, cs=(cand.c_eye,))[cand.c_eye] \
        if (w < 1).any() else None
    return fuse(eeg, eye, cand.weights)


def self_train(src_x, src_y, tgt_x, tgt_recordings, cand, seed, use_mlp=True):
    """v1's class-balanced self-training, with pseudo-labels taken from the candidate's smoothed probabilities.

    Returns the final smoothed probabilities on the target and the windows added per round.
    """
    predict = lambda x, y: smooth(fit_candidate(x, y, tgt_x, cand, seed, use_mlp), tgt_recordings, cand.k, cand.causal)
    proba = predict(src_x, src_y)
    added = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        proba = predict(np.concatenate([src_x, tgt_x[chosen]]), np.concatenate([src_y, pseudo[chosen]]))
    return proba, added
