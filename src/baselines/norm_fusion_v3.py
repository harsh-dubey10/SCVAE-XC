"""NormFusion v3: bagged EEG, eye and early experts averaged into an ensemble, C per expert by v1's
film-robust cross-validation, membership by cross-culture validation when two training cultures exist,
and v1's self-training on the ensemble.

Design: docs/superpowers/specs/2026-10-02-norm-fusion-v3-design.md.
"""

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

from src.baselines.norm_fusion import (C_GRID, EEG, EYE, MLP_EPOCHS, MLP_SEEDS, ST_MIN, ST_ROUNDS, ST_THRESHOLD,
                                       cv_cells, expert_probas, fit_lr)
from src.baselines.norm_fusion_v2 import OFFSET
from src.baselines.strict_baselines import fit_mlp

N_BAGS, BAG_FRACTION = 10, 0.8
FEATURES = {"eeg": EEG, "eye": EYE, "early": slice(0, 343)}
EXPERTS = tuple(FEATURES)
MEMBERSHIPS = (("eeg",), ("eeg", "eye"), ("eeg", "early"), ("eeg", "eye", "early"))


def bag_people(subjects, seed, n_bags=N_BAGS):
    """Per bag, the training people it uses: round(0.8 n) of each culture's people, without replacement."""
    people = np.unique(subjects)
    bags = []
    for b in range(n_bags):
        rng = np.random.default_rng(seed * 1000 + b)
        chosen = []
        for culture in np.unique(people // OFFSET):
            mine = people[people // OFFSET == culture]
            chosen += sorted(rng.choice(mine, size=round(BAG_FRACTION * len(mine)), replace=False).tolist())
        bags.append(np.array(chosen))
    return bags


def bag_specs(subjects, seed, bagged, n_bags=N_BAGS):
    """(people, MLP seeds) per bag; unbagged: one bag of all people with v1's MLP seeds."""
    if not bagged:
        return [(np.unique(subjects), tuple(seed + i for i in range(MLP_SEEDS)))]
    return [(people, tuple(seed + 100 * (b + 1) + i for i in range(MLP_SEEDS)))
            for b, people in enumerate(bag_people(subjects, seed, n_bags))]


def fit_bag(x, y, eval_x, c, seeds):
    """Probabilities on eval_x of LR(C) and one MLP per seed, averaged exactly as v1's expert_probas."""
    torch.set_num_threads(1)
    with threadpool_limits(1):
        lr = fit_lr(x, y, c).predict_proba(eval_x)
        mlp = sum(fit_mlp(x, y, MLP_EPOCHS, "cpu", s).proba(eval_x) for s in seeds)
    return (lr + mlp) / (1 + len(seeds))


def member_probas(train, eval_x, members, cs, bags, n_jobs, extra_x=None, extra_y=None):
    """Per expert, probabilities on eval_x averaged over its bags. Each bag is fitted on its own people's
    windows of `train`, plus (extra_x, extra_y) when given (self-training's pseudo-labelled windows)."""
    jobs = []
    for member in members:
        features = FEATURES[member]
        for people, seeds in bags:
            mine = np.isin(train.subjects, people)
            x, y = train.x[mine][:, features], train.y[mine]
            if extra_x is not None:
                x, y = np.concatenate([x, extra_x[:, features]]), np.concatenate([y, extra_y])
            jobs.append((member, x, y, eval_x[:, features], cs[member], seeds))
    fitted = Parallel(n_jobs=n_jobs)(delayed(fit_bag)(x, y, ev, c, s) for _, x, y, ev, c, s in jobs)
    per_member = {m: [] for m in members}
    for (member, *_), proba in zip(jobs, fitted):
        per_member[member].append(proba)
    return {m: sum(p) / len(p) for m, p in per_member.items()}


def ensemble(probas, membership):
    return sum(probas[m] for m in membership) / len(membership)


def _cell_scores(train_x, train_y, val_x, val_y, seed):
    torch.set_num_threads(1)
    with threadpool_limits(1):
        probas = expert_probas(train_x, train_y, val_x, seed, True)
    return {c: balanced_accuracy_score(val_y, p.argmax(axis=1)) for c, p in probas.items()}


def choose_cs(data, seed, n_jobs):
    """Per expert, the C with the best mean balanced accuracy over v1's 9 people-and-films cells, scoring the
    expert alone (unbagged, v1's seeds); ties to the smaller C. Returns ({expert: C}, {expert: {C: score}})."""
    cells = cv_cells(data.subjects, data.y, data.clips, "person_film")
    jobs = [(expert, train, val) for expert in EXPERTS for train, val in cells]
    results = Parallel(n_jobs=n_jobs)(
        delayed(_cell_scores)(data.x[train][:, FEATURES[e]], data.y[train], data.x[val][:, FEATURES[e]],
                              data.y[val], seed) for e, train, val in jobs)
    scores = {e: {c: float(np.mean([r[c] for (expert, _, _), r in zip(jobs, results) if expert == e]))
                  for c in C_GRID} for e in EXPERTS}
    chosen = {e: max(C_GRID, key=lambda c: scores[e][c]) for e in EXPERTS}
    return chosen, scores


def choose_membership(culture_a, culture_b, seed, n_jobs, n_bags=N_BAGS):
    """Train on one culture, validate on the other, both ways; the membership with the best mean balanced
    accuracy (first in MEMBERSHIPS order on ties). The test culture is never an input."""
    scores = {m: [] for m in MEMBERSHIPS}
    for train, val in ((culture_a, culture_b), (culture_b, culture_a)):
        cs, _ = choose_cs(train, seed, n_jobs)
        probas = member_probas(train, val.x, EXPERTS, cs, bag_specs(train.subjects, seed, True, n_bags), n_jobs)
        for m in MEMBERSHIPS:
            scores[m].append(balanced_accuracy_score(val.y, ensemble(probas, m).argmax(axis=1)))
    mean = {m: float(np.mean(v)) for m, v in scores.items()}
    best = max(MEMBERSHIPS, key=mean.get)
    return best, {"+".join(m): s for m, s in mean.items()}


def self_train(train, tgt_x, membership, cs, bags, n_jobs, first=None):
    """v1's class-balanced self-training on the ensemble's probabilities; every bag of every member is refitted
    on its own people plus all chosen target windows. `first`: precomputed member probabilities on tgt_x.
    Returns the final probabilities and the windows added per round."""
    probas = first if first is not None else member_probas(train, tgt_x, membership, cs, bags, n_jobs)
    proba = ensemble(probas, membership)
    added = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        probas = member_probas(train, tgt_x, membership, cs, bags, n_jobs, tgt_x[chosen], pseudo[chosen])
        proba = ensemble(probas, membership)
    return proba, added
