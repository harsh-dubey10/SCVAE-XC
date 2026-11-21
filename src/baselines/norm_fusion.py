"""NormFusion: a simple cross-culture EEG + eye emotion model.

Per-recording normalisation; an EEG expert and an eye expert (logistic
regression + 3 small MLPs each); late fusion p = w p_EEG + (1 - w) p_eye;
C and w chosen by cross-validation inside the training culture that holds
out people and films together; then class-balanced self-training on the
unlabelled test culture. Design: docs/superpowers/specs/2026-10-01-norm-fusion-model-design.md.
"""

import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.strict_baselines import fit_mlp
from src.experiments.clip_heldout_emotion import clip_folds
from src.experiments.dcca_data import shared_trials
from src.experiments.train_scvae_reference import available_subjects, normalize_recording, sessions_in

C_GRID = (0.001, 0.01, 0.1, 1.0)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EEG, EYE = slice(0, 310), slice(310, 343)
MLP_EPOCHS, MLP_SEEDS = 30, 3
ST_ROUNDS, ST_THRESHOLD, ST_MIN = 3, 0.9, 10
ROOTS = {"SEED_CHN_MM": "SEED_CHN_MM_EEG_EYE", "SEED_GER": "SEED_GER_EEG_EYE", "SEED_FRA": "SEED_FRA_EEG_EYE"}

Config = namedtuple("Config", "c_eeg c_eye w")


@dataclass
class Culture:
    x: np.ndarray
    y: np.ndarray
    clips: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray = None


def load_culture(culture, normalise=True, drop_shared=False):
    root = Path("data/processed") / ROOTS[culture]
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = torch.stack(data["data"]).float().flatten(start_dim=1)
            xs.append((normalize_recording(x) if normalise else x).numpy())
            ys.append(np.asarray(data["labels"]))
            clips.append(np.asarray(data["trials"]))
            subjects.append(np.full(len(x), subject))
            recordings.append(np.full(len(x), 1000 * int(session) + int(subject)))
    data = Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])
    if drop_shared:
        keep = ~np.isin(data.clips, list(shared_trials(culture)))
        data = Culture(data.x[keep], data.y[keep], data.clips[keep], data.subjects[keep], data.recordings[keep])
    return data


def fit_lr(x, y, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)).fit(x, y)


def expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=C_GRID):
    """Per C: class probabilities on eval_x of (LR(C) + 3 MLPs) / 4, or of LR(C) alone."""
    lr = {c: fit_lr(train_x, train_y, c).predict_proba(eval_x) for c in cs}
    if not use_mlp:
        return lr
    mlp = sum(fit_mlp(train_x, train_y, MLP_EPOCHS, "cpu", seed + i).proba(eval_x) for i in range(MLP_SEEDS))
    return {c: (p + mlp) / (1 + MLP_SEEDS) for c, p in lr.items()}


def candidate_probas(train_x, train_y, eval_x, seed, use_mlp, early):
    """Fused probabilities on eval_x for every candidate, in grid order."""
    if early:
        return {Config(c, None, None): p for c, p in expert_probas(train_x, train_y, eval_x, seed, use_mlp).items()}
    eeg = expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp)
    eye = expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp)
    return {Config(ce, cy, w): w * eeg[ce] + (1 - w) * eye[cy] for ce in C_GRID for cy in C_GRID for w in W_GRID}


def cv_cells(subjects, y, clips, mode):
    """(train, validation) masks: 3 person groups x 3 film folds ("person_film"), or 3 person groups ("person")."""
    group_of = {s: i % 3 for i, s in enumerate(np.unique(subjects))}
    groups = np.array([group_of[s] for s in subjects])
    if mode == "person":
        return [(groups != g, groups == g) for g in range(3)]
    folds = clip_folds(y, clips)
    return [((groups != g) & (folds != f), (groups == g) & (folds == f)) for g in range(3) for f in range(3)]


def select_config(data, seed, use_mlp, early, mode="person_film", keep=None):
    """Mean validation balanced accuracy of each candidate; the best (first in grid order on ties)."""
    scores = defaultdict(list)
    for train, val in cv_cells(data.subjects, data.y, data.clips, mode):
        for cfg, proba in candidate_probas(data.x[train], data.y[train], data.x[val], seed, use_mlp, early).items():
            if keep is None or keep(cfg):
                scores[cfg].append(balanced_accuracy_score(data.y[val], proba.argmax(axis=1)))
    mean = {cfg: float(np.mean(v)) for cfg, v in scores.items()}
    return max(mean, key=mean.get), mean


def config_proba(train_x, train_y, eval_x, cfg, seed, use_mlp, early):
    """Probabilities on eval_x for one candidate, fitting only the experts it uses."""
    if early:
        return expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=(cfg.c_eeg,))[cfg.c_eeg]
    proba = 0
    if cfg.w > 0:
        proba = proba + cfg.w * expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp,
                                              cs=(cfg.c_eeg,))[cfg.c_eeg]
    if cfg.w < 1:
        proba = proba + (1 - cfg.w) * expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp,
                                                    cs=(cfg.c_eye,))[cfg.c_eye]
    return proba


def self_train(src_x, src_y, tgt_x, cfg, seed, use_mlp, early):
    """Class-balanced self-training on unlabelled target windows; returns (probabilities, windows added per round)."""
    proba = config_proba(src_x, src_y, tgt_x, cfg, seed, use_mlp, early)
    added = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        proba = config_proba(np.concatenate([src_x, tgt_x[chosen]]), np.concatenate([src_y, pseudo[chosen]]),
                             tgt_x, cfg, seed, use_mlp, early)
    return proba, added
