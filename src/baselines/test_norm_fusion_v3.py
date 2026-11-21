"""Checks for NormFusion v3's pieces on small synthetic float32 data.

    python -m src.baselines.test_norm_fusion_v3
"""

import inspect

import numpy as np

from src.baselines.norm_fusion import EEG, EYE, Config, Culture, expert_probas
from src.baselines.norm_fusion import self_train as v1_self_train
from src.baselines.norm_fusion_v2 import OFFSET, combine
from src.baselines.norm_fusion_v3 import (EXPERTS, MEMBERSHIPS, bag_people, bag_specs, choose_cs, choose_membership,
                                          ensemble, fit_bag, member_probas, self_train)

rng = np.random.default_rng(0)


def synthetic(n_subjects=6, signal=1.0, offset=0.0):
    """Float32 features like the real data; one recording per person; EEG and (weakly) eye carry the emotion."""
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    for s in range(1, n_subjects + 1):
        for emotion, k in [(e, k) for k in range(3) for e in (2, 0, 1)]:
            x = rng.normal(size=(20, 343)) + offset
            x[:, :5] += signal * emotion
            x[:, 310:315] += 0.5 * signal * emotion
            xs.append(x.astype(np.float32)); ys.append(np.full(20, emotion))
            clips.append(np.full(20, 1 + 3 * emotion + k))
            subjects.append(np.full(20, s)); recordings.append(np.full(20, 1000 + s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])


# Bags: round(0.8 n) people of each culture, deterministic per seed, different between bags.
a, b = synthetic(8), synthetic(4)
both = combine([a, b])
bags = bag_people(both.subjects, seed=12)
assert len(bags) == 10 and all((x == y).all() for x, y in zip(bags, bag_people(both.subjects, seed=12)))
for people in bags:
    assert (people < OFFSET).sum() == 6 and (people >= OFFSET).sum() == 3  # round(6.4), round(3.2)
    assert len(set(people.tolist())) == len(people) and set(people.tolist()) <= set(np.unique(both.subjects).tolist())
assert len({tuple(p.tolist()) for p in bags}) > 1
assert not all((x == y).all() for x, y in zip(bags, bag_people(both.subjects, seed=34)))
specs = bag_specs(a.subjects, seed=12, bagged=True)
assert [s for _, s in specs[:2]] == [(112, 113, 114), (212, 213, 214)]
(everyone, seeds), = bag_specs(a.subjects, seed=12, bagged=False)
assert seeds == (12, 13, 14) and everyone.tolist() == list(range(1, 9))
assert len(bag_specs(a.subjects, seed=12, bagged=True, n_bags=3)) == 3

# One unbagged bag is v1's expert, bit for bit.
data = synthetic(6)
train, held = data.subjects <= 4, data.subjects > 4
mine = fit_bag(data.x[train][:, EEG], data.y[train], data.x[held][:, EEG], 0.1, (12, 13, 14))
theirs = expert_probas(data.x[train][:, EEG], data.y[train], data.x[held][:, EEG], 12, True, cs=(0.1,))[0.1]
assert mine.dtype == np.float32 and np.array_equal(mine, theirs)

# member_probas: each bag is fitted only on its own people; results do not depend on the number of workers.
part = Culture(*[v[train] for v in (data.x, data.y, data.clips, data.subjects, data.recordings)])
spec = [(np.array([1, 2]), (5, 6, 7)), (np.array([2, 3, 4]), (8, 9, 10))]
probas = member_probas(part, data.x[held], ("eeg",), {"eeg": 0.1}, spec, n_jobs=1)
by_hand = [fit_bag(part.x[np.isin(part.subjects, p)][:, EEG], part.y[np.isin(part.subjects, p)], data.x[held][:, EEG],
                   0.1, s) for p, s in spec]
assert np.array_equal(probas["eeg"], (by_hand[0] + by_hand[1]) / 2)
cs = {"eeg": 0.1, "eye": 1.0, "early": 0.01}
assert all(np.array_equal(member_probas(part, data.x[held], EXPERTS, cs, spec, n_jobs=1)[m],
                          member_probas(part, data.x[held], EXPERTS, cs, spec, n_jobs=3)[m]) for m in EXPERTS)

# The {EEG, eye} ensemble of unbagged experts is v1's late fusion with w = 0.5, bit for bit.
unbagged = bag_specs(part.subjects, seed=12, bagged=False)
experts = member_probas(part, data.x[held], ("eeg", "eye"), cs, unbagged, n_jobs=2)
v1_eye = expert_probas(part.x[:, EYE], part.y, data.x[held][:, EYE], 12, True, cs=(1.0,))[1.0]
v1_eeg = expert_probas(part.x[:, EEG], part.y, data.x[held][:, EEG], 12, True, cs=(0.1,))[0.1]
assert np.array_equal(ensemble(experts, ("eeg", "eye")), 0 + 0.5 * v1_eeg + 0.5 * v1_eye)
assert MEMBERSHIPS == (("eeg",), ("eeg", "eye"), ("eeg", "early"), ("eeg", "eye", "early"))

# C per expert: one of the grid, deterministic, the best score with ties to the smaller C.
chosen, scores = choose_cs(data, seed=12, n_jobs=4)
assert set(chosen) == set(EXPERTS) and (chosen, scores) == choose_cs(data, seed=12, n_jobs=2)
for expert in EXPERTS:
    best = max(scores[expert].values())
    assert chosen[expert] == min(c for c, s in scores[expert].items() if s == best)

# Cross-culture membership: never given a test culture; deterministic; one of the four options.
assert list(inspect.signature(choose_membership).parameters) == ["culture_a", "culture_b", "seed", "n_jobs", "n_bags"]
c1, c2 = synthetic(6), synthetic(6, offset=0.3)
membership, m_scores = choose_membership(c1, c2, seed=12, n_jobs=4, n_bags=2)
assert membership in MEMBERSHIPS and set(m_scores) == {"+".join(m) for m in MEMBERSHIPS}
assert (membership, m_scores) == choose_membership(c1, c2, seed=12, n_jobs=1, n_bags=2)

# Self-training: never given test labels; {EEG, eye} unbagged is v1's self-training with w = 0.5, bit for bit.
assert "tgt_y" not in inspect.signature(self_train).parameters
src, tgt = synthetic(6, signal=3.0), synthetic(4, signal=3.0, offset=0.3)  # strong enough for confident windows
for c_eeg, c_eye in ((1.0, 1.0), (0.1, 0.01)):
    mine, mine_added = self_train(src, tgt.x, ("eeg", "eye"), {"eeg": c_eeg, "eye": c_eye},
                                  bag_specs(src.subjects, 12, bagged=False), n_jobs=3)
    theirs, theirs_added = v1_self_train(src.x, src.y, tgt.x, Config(c_eeg, c_eye, 0.5), 12, True, False)
    assert mine_added == theirs_added and len(mine_added) >= 1 and np.array_equal(mine, theirs), (c_eeg, c_eye)
    if (c_eeg, c_eye) == (1.0, 1.0):
        reference, reference_added = mine, mine_added
# Precomputed first-round probabilities give the same result; no signal stops after 0 rounds.
first = member_probas(src, tgt.x, ("eeg", "eye"), {"eeg": 1.0, "eye": 1.0}, bag_specs(src.subjects, 12, False), 2)
again, again_added = self_train(src, tgt.x, ("eeg", "eye"), {"eeg": 1.0, "eye": 1.0},
                                bag_specs(src.subjects, 12, bagged=False), n_jobs=2, first=first)
assert again_added == reference_added and np.array_equal(again, reference)
noise_src, noise_tgt = synthetic(6, signal=0.0), synthetic(4, signal=0.0)
assert self_train(noise_src, noise_tgt.x, ("eeg",), {"eeg": 0.001}, bag_specs(noise_src.subjects, 12, True, 2), 2)[1] == []
print("norm_fusion_v3 checks passed")
