"""Checks for train_dcca_am on a tiny random problem.

    python -m src.experiments.test_train_dcca_am
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.experiments.dcca_data import Recording
import torch

from src.experiments.train_dcca_am import architecture, existing_ok, run_one, run_tasks, write_json
from src.models.dcca_am import DCCA_AM, OUTPUT_DIM, make_optimizers, train_epoch

rng = np.random.default_rng(0)


def fake(subject, n=61):
    return Recording(subject, 1, rng.normal(size=(n, 310)).astype(np.float32),
                     rng.normal(size=(n, 33)).astype(np.float32), rng.integers(0, 3, n), np.ones(n, dtype=int))


assert architecture(12, 0) == (150, 35)
h1, h2 = architecture(12, 7)
assert 100 <= h1 < 200 and 20 <= h2 < 50 and architecture(12, 7) == (h1, h2) != architecture(34, 7)

# 61 + 60 = 121 source windows -> 4 batches of 30 and a final batch of 1, which the CCA step cannot handle.
record = run_one([fake(1, 61), fake(2, 60)], [fake(5), fake(6)], (150, 35), 3, epochs=3, device="cpu", seed=12, index=0)
assert len(record["confusion"]) == 3 and set(record["confusion"][0]) == {"5", "6"}
assert np.array(record["confusion"][-1]["5"]).sum() == 61
assert record["skipped_batches"] == 3, record["skipped_batches"]  # one 1-sample batch per epoch
assert record["final"]["6"]["recording"] == [106] * 61 and len(record["final"]["6"]["pred"]) == 61
assert isinstance(record["nonfinite_epochs"], int)

# Atomic write: no temporary file left, content readable.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "a" / "arch_00.json"
    write_json(path, record)
    assert json.loads(path.read_text())["layer_sizes"] == [150, 35]
    assert [p.name for p in path.parent.iterdir()] == ["arch_00.json"]

assert record["shuffle"] is True
unshuffled = run_one([fake(1, 61), fake(2, 60)], [fake(5)], (150, 35), 3, epochs=1, device="cpu", seed=12,
                     index=0, shuffle=False)
assert unshuffled["shuffle"] is False and unshuffled["skipped_batches"] == 1

# --skip-existing trusts a file only if it was made with the same epochs and shuffle setting.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "arch_00.json"
    write_json(path, {**record, "epochs": 5})
    assert not existing_ok(path, epochs=100, shuffle=True)
    write_json(path, {**record, "epochs": 100})
    assert existing_ok(path, epochs=100, shuffle=True) and not existing_ok(path, epochs=100, shuffle=False)
    assert not existing_ok(Path(tmp) / "missing.json", epochs=100, shuffle=True)

# A systematic error (every batch fails) aborts instead of writing an untrained network.
broken = DCCA_AM(310, 33, [150, 35, OUTPUT_DIM], 3)
try:
    train_epoch(broken, make_optimizers(broken), torch.randn(90, 310), torch.randn(90, 7), torch.randint(0, 3, (90,)))
    raise AssertionError("train_epoch should have raised")
except RuntimeError as error:
    assert "batches failed" in str(error)

# Divergence is recorded and the run completes.
import src.experiments.train_dcca_am as trainer


def diverging_epoch(model, *args, **kwargs):
    skipped = train_epoch(model, *args, **kwargs)
    with torch.no_grad():
        model.classification.weight.fill_(float("nan"))  # simulate a blow-up during the epoch
    return skipped


trainer.train_epoch = diverging_epoch
poisoned = run_one([fake(1, 61), fake(2, 60)], [fake(5)], (150, 35), 3, epochs=3, device="cpu", seed=12, index=0)
trainer.train_epoch = train_epoch
assert poisoned["nonfinite_epochs"] == 3 and len(poisoned["confusion"]) == 3

assert run_tasks("film") == ["film_negative", "film_negative_2", "film_positive", "film_all_5"]
print("train_dcca_am checks passed")
