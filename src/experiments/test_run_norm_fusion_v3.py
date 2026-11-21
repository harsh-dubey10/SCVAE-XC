"""Checks for run_norm_fusion_v3 on small synthetic float32 cultures.

    python -m src.experiments.test_run_norm_fusion_v3
"""

import numpy as np

from src.baselines.norm_fusion import Culture
from src.baselines.norm_fusion_v3 import choose_membership
from src.experiments.run_norm_fusion import run_direction as v1_run_direction
from src.experiments.run_norm_fusion_v3 import SETTINGS, VARIANTS, run_direction, setting_name

rng = np.random.default_rng(1)


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


# The 11 settings: 6 one-to-one, 3 two-culture, 2 new films.
assert len(SETTINGS) == 11 and [s[2] for s in SETTINGS].count("emotion_two_cultures") == 3
assert setting_name(("SEED_GER", "SEED_FRA")) == "SEED_GER+SEED_FRA"

source, target = synthetic(6, 0.0), synthetic(4, 0.3)
records = run_direction([source], target, seed=12, n_jobs=4, n_bags=2, keep_predictions=True)
assert set(records) == set(VARIANTS)
for name, record in records.items():
    assert set(record["per_subject"]) == {"1", "2", "3", "4"}, name
    assert len(record["predictions"]) == len(target.x), name
assert records["fair"]["config"]["membership"] == ["eeg", "eye", "early"]
assert records["members_all"]["predictions"] == records["fair"]["predictions"]
assert records["no_selftrain"]["added"] == []
table = records["fair"]["candidates"]
assert table["keys"] == ["eeg", "eeg+eye", "eeg+early", "eeg+eye+early", "fair_selftrained"]
assert all(len(v) == 5 for v in table["accuracy"].values())

# The v1 variant is v1's own run.
v1 = v1_run_direction(source, target, seed=12, variants=["fair"])["fair"]
assert records["v1"]["config"] == v1["config"] and records["v1"]["added"] == v1["added"]
assert all(records["v1"]["per_subject"][s]["accuracy"] == v1["per_subject"][s]["accuracy"] for s in v1["per_subject"])

# Test labels cannot influence anything: with shuffled target labels every choice, every added window count and
# every per-window prediction is unchanged.
shuffled = Culture(target.x, rng.permutation(target.y), target.clips, target.subjects, target.recordings)
again = run_direction([source], shuffled, seed=12, n_jobs=2, n_bags=2, keep_predictions=True)
for name in VARIANTS:
    assert again[name]["config"] == records[name]["config"], name
    assert again[name]["added"] == records[name]["added"], name
    assert again[name]["predictions"] == records[name]["predictions"], name  # also: 2 workers vs 4 identical

# Two training cultures: the membership is the cross-culture choice, and it does not change when the test
# culture is replaced by different data.
c1, c2 = synthetic(4, 0.0), synthetic(4, 0.2)
two = run_direction([c1, c2], target, seed=12, n_jobs=4, n_bags=2)
expected, _ = choose_membership(c1, c2, seed=12, n_jobs=4, n_bags=2)
assert two["fair"]["config"]["membership"] == list(expected)
other = run_direction([c1, c2], synthetic(4, -0.5), seed=12, n_jobs=4, n_bags=2)
assert other["fair"]["config"] == two["fair"]["config"]
print("run_norm_fusion_v3 checks passed")
