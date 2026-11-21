"""Checks for summarize_norm_fusion_v3.

    python -m src.experiments.test_summarize_norm_fusion_v3
"""

import json
import sys
import tempfile
from pathlib import Path

from src.experiments import summarize_norm_fusion_v3
from src.experiments.summarize_norm_fusion_v3 import PREDICTED, complete, success, v1_check

# Pre-registered prediction, as written in the spec.
assert PREDICTED[("SEED_GER", "SEED_CHN_MM")] == 64 and abs(sum(PREDICTED.values()) / 6 - 64.83) < 0.01

# Success: higher on the average AND in at least 4 of the 6 directions.
assert success({d: 0.70 for d in "abcdef"}, {d: 0.60 for d in "abcdef"}) == (True, 6)
three = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.5, "e": 0.5, "f": 0.5}
assert success(three, {d: 0.6 for d in "abcdef"}) == (False, 3)  # higher on average, but only 3 wins
assert success({d: 0.60 for d in "abcdef"}, {d: 0.60 for d in "abcdef"}) == (False, 0)

# v1 check: per direction, mean over people of the seeds both have; flagged above 2 points.
laptop = {12: {"fair": {"per_subject": {"1": {"accuracy": 0.60}, "2": {"accuracy": 0.70}}}}}
cluster = {12: {"v1": {"per_subject": {"1": {"accuracy": 0.61}, "2": {"accuracy": 0.70}}}},
           78: {"v1": {"per_subject": {"1": {"accuracy": 0.10}, "2": {"accuracy": 0.10}}}}}  # seed 78: not compared
diff, flagged = v1_check(laptop, cluster)
assert abs(diff - 0.005) < 1e-9 and not flagged
cluster[12]["v1"]["per_subject"]["2"]["accuracy"] = 0.75
assert v1_check(laptop, cluster)[1]  # 3 points apart
# No seeds in common: nothing compared, which is flagged (never a silent "ok").
diff, flagged = v1_check(laptop, {78: cluster[78]})
assert diff is None and flagged
# The verdict waits until every direction has all 5 seeds of both fair and v1.
assert complete({12: {"fair": 1, "v1": 1}, 34: {"fair": 1, "v1": 1}, 56: {"fair": 1, "v1": 1},
                 78: {"fair": 1, "v1": 1}, 90: {"fair": 1, "v1": 1}})
assert not complete({12: {"fair": 1, "v1": 1}, 34: {"fair": 1, "v1": 1}, 56: {"fair": 1}})

# The summary builds from records on disk and shows the prediction and the success line.
people = {str(i): {"accuracy": 0.6, "macro_f1": 0.5, "clip_accuracy": 0.7} for i in range(1, 5)}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for seed in (12, 34):
        for variant in summarize_norm_fusion_v3.VARIANTS:
            record = {"config": {"membership": ["eeg", "eye", "early"], "cs": {"eeg": 0.1, "eye": 1.0, "early": 0.01}},
                      "added": [], "per_subject": people, "variant": variant, "source": "SEED_GER",
                      "target": "SEED_CHN_MM", "seed": seed, "setting": "emotion"}
            if variant == "fair":
                record["candidates"] = {"keys": ["eeg", "fair_selftrained"],
                                        "accuracy": {s: [0.5, 0.6] for s in people}}
            if variant == "v1":
                record["config"] = [0.1, 1.0, 0.5]
            path = root / "v3" / "SEED_CHN_MM" / "emotion__from_SEED_GER" / f"seed_{seed}" / f"{variant}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
    sys.argv = ["x", "--root", str(root / "v3"), "--v1-root", str(root / "none"), "--v2-root", str(root / "none"),
                "--dcca-root", str(root / "none"), "--output", str(root / "out.md")]
    summarize_norm_fusion_v3.main()
    text = (root / "out.md").read_text()
assert "German → Chinese" in text and "| 64 |" in text and "pre-registered" in text
assert "Success criterion: pending" in text  # 2 seeds on disk, not 5
print("summarize_norm_fusion_v3 checks passed")
