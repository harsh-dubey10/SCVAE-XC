"""Checks for summarize_norm_fusion_v2.

    python -m src.experiments.test_summarize_norm_fusion_v2
"""

from src.experiments.summarize_norm_fusion_v2 import label, paired_p, v1_reproduction, v2_paper_rules

assert label("SEED_GER") == "German" and label("SEED_CHN_MM+SEED_GER") == "Chinese + German"

# Paper's rules: per seed the best candidate per person, then the mean over seeds.
seed_records = {12: {"fair": {"per_subject": {"1": {}, "2": {}},
                              "candidates": {"keys": ["a", "b"], "accuracy": {"1": [0.5, 0.7], "2": [0.6, 0.4]}}}},
                34: {"fair": {"per_subject": {"1": {}, "2": {}},
                              "candidates": {"keys": ["a", "b"], "accuracy": {"1": [0.9, 0.1], "2": [0.2, 0.4]}}}}}
result = v2_paper_rules(seed_records)
assert abs(result["1"] - 0.8) < 1e-9 and abs(result["2"] - 0.5) < 1e-9

assert paired_p({"1": 0.5, "2": 0.6}, {"1": 0.5, "2": 0.6}) == "1"
assert float(paired_p({str(i): 0.6 + 0.01 * i for i in range(8)}, {str(i): 0.5 for i in range(8)})) < 0.05

# v1 reproduction: compares v2's "v1" variant with the saved v1 "fair" on the keys and seeds both have.
v1_runs = {("T", "emotion", "S"): {12: {"fair": {"config": [0.1, 1.0, 0.5],
                                                 "per_subject": {"1": {"accuracy": 0.6}, "2": {"accuracy": 0.7}}}}}}
v2_runs = {("T", "emotion", "S"): {12: {"v1": {"config": ["late", 0.1, 1.0, [0.5, 0.5, 0.5], 1, False],
                                               "per_subject": {"1": {"accuracy": 0.6}, "2": {"accuracy": 0.7}}}}},
           ("T", "emotion_two_cultures", "S+U"): {12: {"v1": {"config": [], "per_subject": {}}}}}
assert v1_reproduction(v1_runs, v2_runs) == {"compared": 1, "configs_matched": 1, "max_diff": 0.0}
v2_runs[("T", "emotion", "S")][12]["v1"]["per_subject"]["2"]["accuracy"] = 0.65
v2_runs[("T", "emotion", "S")][12]["v1"]["config"][1] = 0.01
check = v1_reproduction(v1_runs, v2_runs)
assert check["configs_matched"] == 0 and abs(check["max_diff"] - 0.05) < 1e-9
# The two-culture and new-films tables report macro-F1 next to accuracy (spec: accuracy and macro-F1 per setting).
import json
import sys
import tempfile
from pathlib import Path

from src.experiments import summarize_norm_fusion_v2

people = {str(i): {"accuracy": 0.5 + 0.01 * i, "macro_f1": 0.4 + 0.01 * i, "clip_accuracy": 0.6} for i in range(1, 6)}
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for target, setting, source in (("SEED_FRA", "emotion_two_cultures", "SEED_CHN_MM+SEED_GER"),
                                    ("SEED_FRA", "emotion_newfilms", "SEED_GER")):
        for variant in ("fair", "v1"):
            record = {"config": ["late", 0.1, 1.0, [0.5, 0.5, 0.5], 1, False], "cv_score": 0.5, "added": [],
                      "per_subject": people, "variant": variant, "source": source, "target": target, "seed": 12,
                      "setting": setting}
            path = root / "v2" / target / f"{setting}__from_{source}" / "seed_12" / f"{variant}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record))
    sys.argv = ["x", "--root", str(root / "v2"), "--v1-root", str(root / "none"), "--dcca-root", str(root / "none"),
                "--output", str(root / "out.md")]
    summarize_norm_fusion_v2.main()
    text = (root / "out.md").read_text()
two = text.split("## Two cultures")[1].split("##")[0]
new = text.split("## New films")[1].split("##")[0]
assert "v2 fair macro-F1" in two and "v2 macro-F1" in new
assert "43.0 [" in two and "43.0 [" in new  # mean macro-F1 of the five people (0.41..0.45)
print("summarize_norm_fusion_v2 checks passed")
