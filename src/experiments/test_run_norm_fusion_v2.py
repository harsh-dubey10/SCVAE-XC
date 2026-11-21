"""Checks for run_norm_fusion_v2 on small synthetic cultures.

    python -m src.experiments.test_run_norm_fusion_v2
"""

import numpy as np

from src.baselines.norm_fusion import Culture
from src.experiments.run_norm_fusion import run_direction as v1_run_direction
from src.experiments.run_norm_fusion_v2 import NEWFILMS_VARIANTS, VARIANTS, load_source, run_direction, score_v2

rng = np.random.default_rng(1)


def synthetic(n_subjects, offset):
    """One recording per person, clips in a fixed mixed order; EEG and (weakly) eye carry the emotion."""
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    for s in range(1, n_subjects + 1):
        for emotion, k in [(e, k) for k in range(3) for e in (2, 0, 1)]:
            x = rng.normal(size=(20, 343)) + offset
            x[:, :5] += emotion
            x[:, 310:315] += 0.5 * emotion
            xs.append(x); ys.append(np.full(20, emotion))
            clips.append(np.full(20, 1 + 3 * emotion + k))
            subjects.append(np.full(20, s)); recordings.append(np.full(20, 1000 + s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])


source, target = synthetic(6, 0.0), synthetic(4, 0.3)
records = run_direction(source, target, seed=12, variants=VARIANTS)
assert set(records) == set(VARIANTS)
for name, record in records.items():
    assert set(record["per_subject"]) == {"1", "2", "3", "4"}, name
    person = record["per_subject"]["1"]
    assert np.array(person["confusion"]).sum() == 180 and 0 <= person["clip_accuracy"] <= 1
assert records["no_selftrain"]["added"] == [] and records["no_selftrain"]["config"] == records["fair"]["config"]
assert records["no_smoothing"]["config"][4] == 1 and records["v1"]["config"][4] == 1
assert records["v1"]["config"][0] == "late" and len(set(records["v1"]["config"][3])) == 1
assert records["fair"]["config"][5] is False
table = records["fair"]["candidates"]
assert len(table["keys"]) == 2181 and table["keys"][-1] == "fair_selftrained"
assert all(len(v) == 2181 for v in table["accuracy"].values())
assert table["accuracy"]["1"][-1] == round(records["fair"]["per_subject"]["1"]["accuracy"], 4)

# "v2 minus all changes" is v1: same choice, same pseudo-labels, same per-person results.
v1 = v1_run_direction(source, target, seed=12, variants=["fair"])["fair"]
mine = records["v1"]
assert [mine["config"][1], mine["config"][2], mine["config"][3][0]] == v1["config"]
assert mine["added"] == v1["added"] and abs(mine["cv_score"] - v1["cv_score"]) < 1e-12
for s in v1["per_subject"]:
    assert mine["per_subject"][s]["accuracy"] == v1["per_subject"][s]["accuracy"], s
    assert mine["per_subject"][s]["confusion"] == v1["per_subject"][s]["confusion"], s

# Test labels cannot influence any choice: with the target labels shuffled, every variant picks the same
# candidate, adds the same windows and makes the same predictions (confusion column sums); only scores differ.
shuffled = Culture(target.x, rng.permutation(target.y), target.clips, target.subjects, target.recordings)
again = run_direction(source, shuffled, seed=12, variants=VARIANTS)
for name in VARIANTS:
    assert again[name]["config"] == records[name]["config"], name
    assert again[name]["added"] == records[name]["added"], name
    for s in records[name]["per_subject"]:
        assert (np.array(again[name]["per_subject"][s]["confusion"]).sum(axis=0).tolist()
                == np.array(records[name]["per_subject"][s]["confusion"]).sum(axis=0).tolist()), (name, s)

# Only the requested variants are computed.
assert set(run_direction(source, target, seed=12, variants=NEWFILMS_VARIANTS)) == set(NEWFILMS_VARIANTS)

# Clip-level accuracy: majority vote of each (recording, clip)'s windows.
tiny = Culture(np.zeros((6, 343)), np.array([0, 0, 0, 1, 1, 1]), np.array([1, 1, 1, 2, 2, 2]),
               np.array([7] * 6), np.array([1] * 6))
proba = np.eye(3)[[0, 0, 1, 2, 2, 1]]
s = score_v2(tiny, proba)["7"]
assert s["accuracy"] == 3 / 6 and s["clip_accuracy"] == 0.5  # clip 1 voted 0 (right), clip 2 voted 2 (wrong)

# Two-culture sources load as one combined culture.
both = load_source("SEED_GER+SEED_FRA", "emotion_two_cultures")
assert both.subjects.max() > 1_000_000 and len(np.unique(both.recordings)) == 20 + 24
print("run_norm_fusion_v2 checks passed")
