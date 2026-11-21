"""Checks for run_norm_fusion on small synthetic cultures.

    python -m src.experiments.test_run_norm_fusion
"""

import numpy as np

from src.baselines.norm_fusion import Culture
from src.experiments.run_norm_fusion import VARIANTS, run_direction, score

rng = np.random.default_rng(1)


def synthetic(n_subjects, offset):
    xs, ys, clips, subjects = [], [], [], []
    for s in range(1, n_subjects + 1):
        for emotion in range(3):
            for k in range(3):
                x = rng.normal(size=(20, 343)) + offset
                x[:, :5] += emotion
                x[:, 310:315] += 0.5 * emotion
                xs.append(x); ys.append(np.full(20, emotion))
                clips.append(np.full(20, 1 + 3 * emotion + k)); subjects.append(np.full(20, s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects)])


source, target = synthetic(6, 0.0), synthetic(4, 0.3)
records = run_direction(source, target, seed=12, variants=VARIANTS, raw_source=source, raw_target=target)
assert set(records) == set(VARIANTS)
for name, record in records.items():
    assert set(record["per_subject"]) == {"1", "2", "3", "4"}, name
    cm = np.array(record["per_subject"]["1"]["confusion"])
    assert cm.shape == (3, 3) and cm.sum() == 180
assert records["eeg_only"]["config"][2] == 1.0 and records["eye_only"]["config"][2] == 0.0
assert records["no_selftrain"]["added"] == [] and records["no_selftrain"]["config"] == records["fair"]["config"]
assert len(records["fair"]["candidates"]) == 81 and "fair_selftrained" in records["fair"]["candidates"]
assert records["early_fusion"]["config"][1] is None

# Test labels cannot influence any choice: with the target labels shuffled, every variant picks the same
# configuration, adds the same pseudo-labelled windows, and makes the same predictions (per-person prediction
# counts = confusion column sums); only the scores may differ.
shuffled = Culture(target.x, rng.permutation(target.y), target.clips, target.subjects)
again = run_direction(source, shuffled, seed=12, variants=VARIANTS, raw_source=source, raw_target=shuffled)
for name in VARIANTS:
    assert again[name]["config"] == records[name]["config"], name
    assert again[name]["added"] == records[name]["added"], name
    for s in records[name]["per_subject"]:
        assert (np.array(again[name]["per_subject"][s]["confusion"]).sum(axis=0).tolist()
                == np.array(records[name]["per_subject"][s]["confusion"]).sum(axis=0).tolist()), (name, s)

# A run with only the fair variant does not compute the others.
assert set(run_direction(source, target, seed=12, variants=["fair"])) == {"fair"}

s = score(np.array([0, 1, 2, 2]), np.eye(3)[[0, 1, 1, 2]], np.array([7, 7, 7, 7]))
assert s["7"]["accuracy"] == 0.75 and np.array(s["7"]["confusion"]).trace() == 3
print("run_norm_fusion checks passed")
