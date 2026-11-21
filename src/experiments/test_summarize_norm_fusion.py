"""Checks for summarize_norm_fusion.

    python -m src.experiments.test_summarize_norm_fusion
"""

from src.experiments.summarize_norm_fusion import dcca_per_subject, matched_comparison, nf_paper_rules, nf_per_subject

good, half = [[10, 0], [0, 10]], [[5, 5], [5, 5]]
seed_records = {
    12: {"fair": {"per_subject": {"1": {"accuracy": 0.6}, "2": {"accuracy": 0.8}},
                  "candidates": {"a": {"1": half, "2": half}, "b": {"1": good, "2": half}}}},
    34: {"fair": {"per_subject": {"1": {"accuracy": 0.4}, "2": {"accuracy": 0.6}},
                  "candidates": {"a": {"1": half, "2": good}, "b": {"1": half, "2": half}}}},
}
assert nf_per_subject(seed_records, "fair") == {"1": 0.5, "2": 0.7}
seed_records[12]["fair"]["per_subject"]["1"]["macro_f1"] = 0.2
seed_records[12]["fair"]["per_subject"]["2"]["macro_f1"] = 0.4
seed_records[34]["fair"]["per_subject"]["1"]["macro_f1"] = 0.4
seed_records[34]["fair"]["per_subject"]["2"]["macro_f1"] = 0.6
assert nf_per_subject(seed_records, "fair", "macro_f1") == {"1": 0.30000000000000004, "2": 0.5}
assert nf_paper_rules(seed_records) == {"1": 0.75, "2": 0.75}

# Missing DCCA-AM results give None, not an error.
assert dcca_per_subject({}, ("SEED_FRA", "emotion", "SEED_GER"), "fair") is None
dcca = {("SEED_FRA", "emotion", "SEED_GER"): {12: {0: {"epochs": 2, "confusion": [{"1": half}, {"1": good}]},
                                                    1: {"epochs": 2, "confusion": [{"1": good}, {"1": half}]}}}}
key = ("SEED_FRA", "emotion", "SEED_GER")
assert dcca_per_subject(dcca, key, "fair", n_archs=2, epochs=2, seeds=1) == {"1": 1.0}
assert dcca_per_subject(dcca, key, "paper", n_archs=2, epochs=2, seeds=1) == {"1": 1.0}
# Incomplete DCCA-AM runs (missing architectures, short runs, or missing seeds) are not reported.
assert dcca_per_subject(dcca, key, "fair", n_archs=3, epochs=2, seeds=1) is None
assert dcca_per_subject(dcca, key, "fair", n_archs=2, epochs=100, seeds=1) is None
assert dcca_per_subject(dcca, key, "fair", n_archs=2, epochs=2, seeds=3) is None
# The success criterion compares averages over the SAME directions and counts directions won.
ours = {"a": {"1": 0.6, "2": 0.8}, "b": {"1": 0.5}, "c": {"1": 0.9}}
theirs = {"a": {"1": 0.5, "2": 0.5}, "b": {"1": 0.6}}  # direction c not finished for DCCA-AM
result = matched_comparison(ours, theirs)
assert (result["directions"], result["wins"]) == (2, 1)
assert abs(result["ours"] - 0.6) < 1e-9 and abs(result["theirs"] - 0.55) < 1e-9
assert matched_comparison(ours, {}) == {"directions": 0, "ours": None, "theirs": None, "wins": 0}
print("summarize_norm_fusion checks passed")
