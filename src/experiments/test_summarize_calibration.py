"""Checks for summarize_calibration.

    python -m src.experiments.test_summarize_calibration
"""

import json
import sys
import tempfile
from pathlib import Path

from src.experiments import summarize_calibration
from src.experiments.summarize_calibration import PREDICTED, culture_gap, person_means


def scores(base, step):
    """person -> draw -> N -> variant -> scores, accuracy rising by `step` per N."""
    out = {}
    for person in ("1", "2"):
        out[person] = {}
        for draw in ("0", "1"):
            out[person][draw] = {}
            for n in ("0", "1", "2"):
                acc = base + step * int(n) + (0.02 if person == "2" else 0.0)
                variants = (["base_only", "combined"] if n == "0" else summarize_calibration.VARIANTS + ["persession"]
                            + (["learned", "persession_learned"] if n == "2" else []))
                out[person][draw][n] = {v: {"accuracy": acc, "macro_f1": acc - 0.01, "sessions": {"1": acc}}
                                        for v in variants}
    return out


# Per person: mean over seeds and draws.
runs = {12: scores(0.60, 0.05), 34: scores(0.70, 0.05)}
assert abs(person_means(runs, 0, "combined")["1"] - 0.65) < 1e-9
assert abs(person_means(runs, 2, "combined")["2"] - 0.77) < 1e-9

# Culture gap: same-culture minus the mean of the cross-culture directions into that culture, same people only.
same = {"1": 0.80, "2": 0.90}
crosses = [{"1": 0.60, "2": 0.70}, {"1": 0.70, "2": 0.80, "9": 0.10}]  # person 9 is not in `same`: ignored
assert abs(culture_gap(same, crosses) - 0.15) < 1e-9

# Pre-registered values as in the spec.
assert PREDICTED["cross"] == (65, 74, 78) and PREDICTED["same"] == (77, 81, 83)

# The summary builds from files, labels calibrated numbers, and writes the figure.
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for folder, target, base in (("cross__from_SEED_GER", "SEED_FRA", 0.6), ("same_group0__from_SEED_FRA", "SEED_FRA", 0.8)):
        path = root / "scores" / target / folder / "seed_12.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(scores(base, 0.05)))
    sys.argv = ["x", "--root", str(root / "scores"), "--output", str(root / "out.md"),
                "--figure", str(root / "curve.png")]
    summarize_calibration.main()
    text = (root / "out.md").read_text()
    assert (root / "curve.png").stat().st_size > 1000
assert "with N calibration clips per person" in text and "German → French" in text
assert "| French | 81.0" in text  # same-culture N = 0: people at 80 and 82
# Follow-ups are reported in their own section, labelled exploratory (not pre-registered).
followup = text.split("## 7.")[1]
assert "exploratory" in followup and "German → French" in followup and "persession" not in text.split("## 7.")[0]
print("summarize_calibration checks passed")
