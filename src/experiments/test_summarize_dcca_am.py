"""Checks for the DCCA-AM summary selection rules.

    python -m src.experiments.test_summarize_dcca_am
"""

import numpy as np

from src.experiments.summarize_dcca_am import accuracy, completeness, fair_score, film_p_value, health, paper_score

good, bad, half = [[10, 0], [0, 10]], [[0, 10], [10, 0]], [[5, 5], [5, 5]]
# arch 0 is mediocre at the last epoch but perfect at epoch 0; arch 2 is perfect at its last epoch.
records = {
    0: {"confusion": [{"1": good}, {"1": half}]},
    1: {"confusion": [{"1": bad}, {"1": bad}]},
    2: {"confusion": [{"1": half}, {"1": good}]},
}
assert accuracy(good, balanced=False) == 1.0 and accuracy([[3, 1], [0, 0]], balanced=True) == 0.75
# Fair = architecture 0, last epoch, whatever the test set says elsewhere.
assert fair_score(records, "1", balanced=False) == 0.5
# Paper's rules search architectures 1..k and every epoch.
assert paper_score(records, "1", k=1, balanced=False) == 0.0
assert paper_score(records, "1", k=2, balanced=False) == 1.0
# More architectures searched never lowers the paper-rules score.
assert paper_score(records, "1", k=2, balanced=False) >= paper_score(records, "1", k=1, balanced=False)
# Completeness: every seed needs architectures 0..20 trained for the full number of epochs.
full = {i: {"epochs": 100, "confusion": [{"1": good}] * 100, "skipped_batches": i, "nonfinite_epochs": 0}
        for i in range(21)}
assert completeness({12: full, 34: full}, n_archs=21, epochs=100) == []
short = {**full, 3: {**full[3], "epochs": 5}}
missing = {k: v for k, v in full.items() if k != 7}
problems = completeness({12: short, 34: missing}, n_archs=21, epochs=100)
assert any("seed 12" in p and "arch 3" in p for p in problems) and any("seed 34" in p and "arch 7" in p for p in problems)

# Health: total skipped batches and non-finite epochs over all seeds and architectures.
assert health({12: full, 34: full}) == (2 * sum(range(21)), 0)


# The film p-value uses every seed, and does not depend on the order seeds were loaded in.
def film_seed(pred):
    return {0: {"final": {"1": {"pred": pred, "true": [0, 0, 1, 1], "recording": [101] * 4},
                          "2": {"pred": pred, "true": [0, 0, 1, 1], "recording": [102] * 4}}}}


perfect, wrong = film_seed([0, 0, 1, 1]), film_seed([1, 1, 0, 0])
a = film_p_value({12: perfect, 34: wrong}, 2, np.random.default_rng(0), 200)
b = film_p_value({34: wrong, 12: perfect}, 2, np.random.default_rng(0), 200)
assert a == b
assert film_p_value({12: perfect, 34: perfect}, 2, np.random.default_rng(0), 200) < 0.5 < a
print("summarize_dcca_am checks passed")
