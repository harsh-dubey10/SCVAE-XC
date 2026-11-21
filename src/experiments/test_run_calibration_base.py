"""Checks for run_calibration_base on small synthetic float32 cultures.

    python -m src.experiments.test_run_calibration_base
"""

import numpy as np

from src.baselines.norm_fusion import Culture
from src.experiments.run_calibration_base import TASKS, same_culture_groups, subset
from src.experiments.run_norm_fusion_v3 import fair_probas, run_direction

rng = np.random.default_rng(3)


def synthetic(n_subjects, offset, signal=2.0):
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


# 15 base tasks: 6 cross-culture directions, then 3 cultures x 3 same-culture groups.
assert len(TASKS) == 15 and [t[2] for t in TASKS].count("cross") == 6 and [t[2] for t in TASKS].count("same") == 9

# T1.1 fair_probas is v3's fair model: the same predictions as run_direction's fair variant.
source, target = synthetic(6, 0.0), synthetic(4, 0.3)
proba, config = fair_probas([source], target, seed=12, n_jobs=4, n_bags=2)
records = run_direction([source], target, seed=12, n_jobs=4, n_bags=2, keep_predictions=True)
assert proba.shape == (len(target.x), 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-5)
assert proba.argmax(axis=1).tolist() == records["fair"]["predictions"]
assert config == records["fair"]["config"]

# T1.2 same-culture groups: 3 groups dealt round-robin over sorted people; every person tested exactly once,
# never in training and test at once.
subjects = np.repeat([3, 1, 8, 5, 2, 7, 4, 6], 5)
groups = same_culture_groups(subjects)
assert len(groups) == 3
tested = [p for _, test in groups for p in test.tolist()]
assert sorted(tested) == list(range(1, 9))
for train, test in groups:
    assert not set(train.tolist()) & set(test.tolist())
    assert set(train.tolist()) | set(test.tolist()) == set(range(1, 9))
assert groups[0][1].tolist() == [1, 4, 7]
part = subset(source, np.array([2, 5]))
assert set(np.unique(part.subjects).tolist()) == {2, 5} and len(part.x) == len(part.recordings) == 2 * 180
print("run_calibration_base checks passed")
