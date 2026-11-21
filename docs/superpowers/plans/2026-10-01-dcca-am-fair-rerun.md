# DCCA-AM Fair Re-run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the paper's DCCA-AM in the 6 cross-culture directions and report the same trained networks under the paper's selection rules, fairly, and fairly on new films, plus a film-identification variant, on the IITGN cluster.

**Architecture:** A faithful PyTorch port of the released DCCA-AM (`src/models/dcca_am.py`), a loader that pairs the raw DE-LDS EEG with the raw eye features per recording (`src/experiments/dcca_data.py`), a training script that saves every epoch's per-person confusion matrices for every architecture (`src/experiments/train_dcca_am.py`), and a summary that derives all evaluation modes from those saved results without retraining (`src/experiments/summarize_dcca_am.py`). Cluster array job via `cluster/dcca_array.sbatch` + `cluster/dcca_sweep.txt`.

**Tech Stack:** Python 3 in `.venv`, PyTorch 2.x, NumPy, scikit-learn, SciPy; SLURM on IITGN Singularity.

**Spec:** `docs/superpowers/specs/2026-10-01-dcca-am-fair-rerun-design.md`

## Global Constraints

- Interpreter: `.venv/bin/python`. There is no pytest: tests are plain-assert modules run as `.venv/bin/python -m <module>` and print one "... checks passed" line (repo convention, e.g. `src/experiments/test_film_identification.py`).
- Network as released: EEG 310 → h1 → h2 → 12 and eye 33 → h1 → h2 → 12, sigmoid hidden layers, linear output; attention fusion; linear classifier 12 → categories.
- Training as released: RMSprop, lr 1e-4 on all parameters, plus 5e-5 on `parameters()[0:8]` and on `parameters()[8:16]`; batch 30, not shuffled, with the released final (possibly empty or tiny) batch; 100 epochs; CCA gradient weight −0.1; a batch that raises is skipped (counted).
- Scaling as released: `MinMaxScaler` fitted on the whole training set (EEG and eye separately), and separately on each test person's windows (all their sessions).
- Architectures: index 0 = (150, 35), fixed in advance; indices 1–20 drawn with `h1 ~ U[100, 200)`, `h2 ~ U[20, 50)` from `np.random.default_rng([seed, index])`.
- Seeds: 12, 34, 56.
- Fair mode = architecture 0, epoch 100 (last). Paper's rules = best test accuracy over architectures 1..k and all epochs, per test person.
- Output: `results/dcca_am/<target>/<task>__from_<source>/seed_<s>/arch_<kk>.json`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Never write "never been done before" or similar novelty claims in code, logs or docs.

## Review Focus

- A final training batch of 1 sample makes the CCA step divide by zero, and an empty one would give a NaN loss; the run must skip either, count it, and continue (1-sample case pinned in Task 1, Step 1 and Task 3, Step 1).
- SEED-GER subjects 5–8 have no session-3 file; the loader must return exactly 20 German recordings, 24 French and 36 Chinese (pinned in Task 2, Step 1).
- A run interrupted mid-write must not leave a truncated JSON that `--skip-existing` then trusts; writes go to a temporary file and are renamed (pinned in Task 3, Step 1).
- A network that diverges (NaN outputs) must be recorded, not crash the run: the epoch is flagged in `nonfinite_epochs` and its predictions still counted (pinned in Task 3, Step 1).
- The fair summary must pick architecture 0, last epoch, even when another architecture or epoch scores higher on the test set (pinned in Task 4, Step 1).

---

### Task 1: Faithful DCCA-AM model

**Files:**
- Create: `src/models/dcca_am.py`
- Test: `src/models/test_dcca_am.py`

**Interfaces:**
- Produces:
  - `cca_metric_derivative(H1: np.ndarray, H2: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]`
  - `class DCCA_AM(nn.Module)`: `__init__(eeg_dim: int, eye_dim: int, layer_sizes: list[int], categories: int)`; `forward(x1, x2) -> (logits, cca_loss, output1, output2, partial_h1, partial_h2)`; `predict(x1, x2) -> torch.LongTensor`
  - `make_optimizers(model) -> tuple[Optimizer, Optimizer, Optimizer]` (classifier, transform 1, transform 2)
  - `train_epoch(model, optimizers, x1, x2, y, batch_size=30) -> int` (number of skipped batches)
  - `OUTPUT_DIM = 12`

- [ ] **Step 1: Write the failing test**

Create `src/models/test_dcca_am.py`:

```python
"""Checks for the DCCA-AM port.

    python -m src.models.test_dcca_am
"""

import numpy as np
import torch

from src.models.dcca_am import OUTPUT_DIM, DCCA_AM, cca_metric_derivative, make_optimizers, train_epoch


def attention_loop(x1, x2, w):
    """The released per-sample loop (dcca_utils.AttentionFusion.forward)."""
    fused = torch.empty_like(x1)
    for i in range(len(x1)):
        a = torch.exp(torch.dot(x1[i], w))
        b = torch.exp(torch.dot(x2[i], w))
        alpha_1 = a / (a + b)
        fused[i] = alpha_1 * x1[i] + (1 - alpha_1) * x2[i]
    return fused


def sum_singular_values(H1, H2, r=1e-3):
    """sum of singular values of S11^-1/2 S12 S22^-1/2, the quantity the derivative is for."""
    m = H1.shape[0]
    a, b = H1 - H1.mean(0), H2 - H2.mean(0)
    s12 = a.T @ b / (m - 1)
    s11 = a.T @ a / (m - 1) + r * np.eye(a.shape[1])
    s22 = b.T @ b / (m - 1) + r * np.eye(b.shape[1])

    def root_inv(s):
        d, v = np.linalg.eigh(s)
        return v @ np.diag(d ** -0.5) @ v.T

    return np.linalg.svd(root_inv(s11) @ s12 @ root_inv(s22), compute_uv=False).sum()


torch.manual_seed(0)
rng = np.random.default_rng(0)

# Batched attention equals the released loop.
model = DCCA_AM(310, 33, [150, 35, OUTPUT_DIM], 3)
x1, x2 = torch.randn(40, OUTPUT_DIM), torch.randn(40, OUTPUT_DIM)
fused, _ = model.attention_fusion(x1, x2)
assert torch.allclose(fused, attention_loop(x1, x2, model.attention_fusion.attention_weights), atol=1e-6)

# The CCA derivative is the gradient of the sum of canonical correlations.
H1, H2 = rng.normal(size=(50, 3)), rng.normal(size=(50, 3)) + 0.5 * rng.normal(size=(50, 3))
_, d1, _ = cca_metric_derivative(H1, H2)
E = rng.normal(size=H1.shape)
eps = 1e-6
numeric = (sum_singular_values(H1 + eps * E, H2) - sum_singular_values(H1 - eps * E, H2)) / (2 * eps)
assert abs(numeric - (d1 * E).sum()) < 1e-4 * max(1.0, abs(numeric)), (numeric, (d1 * E).sum())

# One epoch on random data runs, changes the weights, and skips the 1-sample last batch.
x1, x2 = torch.randn(61, 310), torch.randn(61, 33)  # 61 = 2 * 30 + 1: final batch of one
y = torch.randint(0, 3, (61,))
before = [p.detach().clone() for p in model.parameters()]
skipped = train_epoch(model, make_optimizers(model), x1, x2, y)
assert skipped == 1, skipped
assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
assert model.predict(x1, x2).shape == (61,)

# The optimiser parameter slices match the released code (model1 + first layer of model2, then the rest).
params = list(model.parameters())
assert len(params) == 15 and params[0:8][-1] is model.model2.layers[0][0].bias
print("dcca_am checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.models.test_dcca_am`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.dcca_am'`

- [ ] **Step 3: Write the implementation**

Create `src/models/dcca_am.py`:

```python
"""DCCA-AM (Liu et al., J. Neural Eng. 2022), ported from the released
CrossCultureCode ``DCCA-AM/dcca_utils.py`` and ``chinese_dcca_attention.py``.

Kept as released: the network, the NumPy CCA gradient (weight -0.1), RMSprop
with the overlapping parameter slices, unshuffled batches of 30 including the
final partial batch, and skipping a batch that raises. Two changes: an empty
final batch (when the training size is a multiple of 30) is skipped rather
than fed to the network, where it would give a NaN loss; and the
attention fusion is batched instead of a per-sample loop (same maths; tested
against the loop). ``predict`` skips the CCA step, which does not affect the
logits.
"""

import numpy as np
import torch
import torch.nn as nn

OUTPUT_DIM = 12
LEARNING_RATE = 1e-4
BATCH_SIZE = 30
CCA_GRAD_WEIGHT = -0.1


def cca_metric_derivative(H1, H2):
    """Released ``cca_metric_derivative``: (-corr, d corr / d H1, d corr / d H2), rows = samples."""
    r1 = 1e-3
    r2 = 1e-3
    H1 = H1.T
    H2 = H2.T
    o1 = o2 = H1.shape[0]
    m = H1.shape[1]

    H1bar = H1 - H1.mean(axis=1).reshape([-1, 1])
    H2bar = H2 - H2.mean(axis=1).reshape([-1, 1])

    SigmaHat12 = (1.0 / (m - 1)) * np.matmul(H1bar, H2bar.T)
    SigmaHat11 = (1.0 / (m - 1)) * np.matmul(H1bar, H1bar.T) + r1 * np.eye(o1)
    SigmaHat22 = (1.0 / (m - 1)) * np.matmul(H2bar, H2bar.T) + r2 * np.eye(o2)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)

    posInd1 = np.where(D1 > 0)[0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = np.where(D2 > 0)[0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]

    SigmaHat11RootInv = np.matmul(np.matmul(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.matmul(np.matmul(V2, np.diag(D2 ** -0.5)), V2.T)
    Tval = np.matmul(np.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)
    tmp = np.matmul(Tval.T, Tval)
    corr = np.sqrt(np.trace(tmp))
    cca_loss = -1 * corr

    U_t, D_t, V_prime_t = np.linalg.svd(Tval)
    Delta12 = SigmaHat11RootInv @ U_t @ V_prime_t @ SigmaHat22RootInv
    Delta11 = SigmaHat11RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat11RootInv
    Delta22 = SigmaHat22RootInv @ U_t @ np.diag(D_t) @ U_t.T @ SigmaHat22RootInv
    Delta11 = -0.5 * Delta11
    Delta22 = -0.5 * Delta22

    DerivativeH1 = (1.0 / (m - 1)) * (2 * (Delta11 @ H1bar) + Delta12 @ H2bar)
    DerivativeH2 = (1.0 / (m - 1)) * (2 * (Delta22 @ H2bar) + Delta12 @ H1bar)
    return cca_loss, DerivativeH1.T, DerivativeH2.T


class TransformLayers(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super().__init__()
        sizes = [input_size] + list(layer_sizes)
        layers = []
        for i in range(len(sizes) - 1):
            if i == len(sizes) - 2:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1])))
            else:
                layers.append(nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Sigmoid()))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class AttentionFusion(nn.Module):
    """alpha_1 = exp(x1.w) / (exp(x1.w) + exp(x2.w)); fused = alpha_1 x1 + (1 - alpha_1) x2."""

    def __init__(self, output_dim):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.randn(output_dim, requires_grad=True))

    def forward(self, x1, x2):
        alpha = torch.softmax(torch.stack([x1 @ self.attention_weights, x2 @ self.attention_weights], dim=1), dim=1)
        return alpha[:, :1] * x1 + alpha[:, 1:] * x2, alpha


class DCCA_AM(nn.Module):
    def __init__(self, eeg_dim, eye_dim, layer_sizes, categories):
        super().__init__()
        self.model1 = TransformLayers(eeg_dim, layer_sizes)
        self.model2 = TransformLayers(eye_dim, layer_sizes)
        self.classification = nn.Linear(OUTPUT_DIM, categories)
        self.attention_fusion = AttentionFusion(OUTPUT_DIM)

    def forward(self, x1, x2):
        output1 = self.model1(x1)
        output2 = self.model2(x2)
        cca_loss, partial_h1, partial_h2 = cca_metric_derivative(
            output1.detach().cpu().numpy(), output2.detach().cpu().numpy())
        fused, _ = self.attention_fusion(output1, output2)
        return self.classification(fused), cca_loss, output1, output2, partial_h1, partial_h2

    @torch.no_grad()
    def predict(self, x1, x2):
        fused, _ = self.attention_fusion(self.model1(x1), self.model2(x2))
        return self.classification(fused).argmax(dim=1)


def make_optimizers(model):
    """Released: one RMSprop on all parameters, two on overlapping slices at half the rate."""
    params = list(model.parameters())
    return (torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE),
            torch.optim.RMSprop(params[0:8], lr=LEARNING_RATE / 2),
            torch.optim.RMSprop(params[8:16], lr=LEARNING_RATE / 2))


def train_epoch(model, optimizers, x1, x2, y, batch_size=BATCH_SIZE):
    """One released epoch: unshuffled batches, then the remainder (possibly empty). Returns skipped batches."""
    classifier_opt, opt1, opt2 = optimizers
    loss_fn = nn.CrossEntropyLoss()
    batches = len(y) // batch_size
    skipped = 0
    model.train()
    for b in range(batches + 1):
        rows = slice(batch_size * b, None if b == batches else batch_size * (b + 1))
        if len(y[rows]) == 0:  # released code would feed an empty batch (NaN loss); skip it instead
            skipped += 1
            continue
        try:
            logits, _, output1, output2, partial_h1, partial_h2 = model(x1[rows], x2[rows])
            loss = loss_fn(logits, y[rows])
            opt1.zero_grad()
            opt2.zero_grad()
            classifier_opt.zero_grad()
            output1.backward(CCA_GRAD_WEIGHT * torch.as_tensor(partial_h1, dtype=torch.float, device=output1.device),
                             retain_graph=True)
            output2.backward(CCA_GRAD_WEIGHT * torch.as_tensor(partial_h2, dtype=torch.float, device=output2.device),
                             retain_graph=True)
            loss.backward()
            opt1.step()
            opt2.step()
            classifier_opt.step()
        except Exception:  # released code: `except: continue`
            skipped += 1
    return skipped
```

Note on the `len(params) == 15` assertion: model1 has 3 Linear layers (6 tensors), model2 6, classification 2, attention 1 → 15; `params[0:8]` therefore ends with model2's first bias, exactly as in the released code.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.models.test_dcca_am`
Expected: `dcca_am checks passed`

If the CCA finite-difference assertion fails, do NOT change `cca_metric_derivative` (it must stay as released). Instead report the two printed numbers: a mismatch is a finding about the released code and goes in the write-up.

- [ ] **Step 5: Commit**

```bash
git add src/models/dcca_am.py src/models/test_dcca_am.py
git commit -m "Port the released DCCA-AM with a batched attention fusion

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Paired raw EEG + eye loader

**Files:**
- Create: `src/experiments/dcca_data.py`
- Test: `src/experiments/test_dcca_data.py`

**Interfaces:**
- Consumes: `src.preprocessing.convert_seed_crossculture.load_session(path) -> (features[n,62,5], labels[n], train_count)`; `src.experiments.film_identification.FILMS` (name → (emotion, German trial, French trial)) and `TASKS` (task → list of film names).
- Produces:
  - `RAW_EEG: dict[str, Path]` (culture → raw `eeg_used_4s` dir)
  - `@dataclass Recording(subject: int, session: int, eeg: np.ndarray[n,310], eye: np.ndarray[n,33], labels: np.ndarray[n], trials: np.ndarray[n])`
  - `load_recordings(culture: str) -> list[Recording]`
  - `shared_trials(culture: str) -> set[int]`
  - `select(recordings, drop_shared_of: str | None = None, film_task: str | None = None, culture: str | None = None) -> list[Recording]` — returns recordings restricted to kept windows; with `film_task`, `labels` become the film index within `TASKS[film_task]`.

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_dcca_data.py`:

```python
"""Checks for the DCCA-AM inputs.

    python -m src.experiments.test_dcca_data
"""

import numpy as np

from src.experiments.dcca_data import load_recordings, select, shared_trials
from src.experiments.film_identification import FILMS, TASKS

expected_recordings = {"SEED_CHN_MM": 36, "SEED_GER": 20, "SEED_FRA": 24}
data = {}
for culture, count in expected_recordings.items():
    recordings = load_recordings(culture)
    assert len(recordings) == count, (culture, len(recordings))
    for r in recordings:
        assert r.eeg.shape == (len(r.labels), 310) and r.eye.shape == (len(r.labels), 33)
        assert np.isfinite(r.eeg).all() and np.isfinite(r.eye).all()
        assert set(np.unique(r.labels)) == {0, 1, 2}
        # raw DE, not the per-window z-scored processed EEG
        assert abs(r.eeg.mean(axis=1)).max() > 1.0
    data[culture] = recordings

# New films: exactly the 5 shared films disappear from German and French recordings.
for culture in ("SEED_GER", "SEED_FRA"):
    kept = select(data[culture], drop_shared_of=culture)
    for before, after in zip(data[culture], kept):
        assert set(np.unique(before.trials)) - set(np.unique(after.trials)) == shared_trials(culture)
        assert len(after.labels) == (~np.isin(before.trials, list(shared_trials(culture)))).sum()

# Film labels: index of the film within the task, same windows as the shared trials.
for culture, column in (("SEED_GER", 1), ("SEED_FRA", 2)):
    films = select(data[culture], film_task="negative", culture=culture)
    for before, after in zip(data[culture], films):
        trial_of = [FILMS[name][column] for name in TASKS["negative"]]
        assert sorted(np.unique(after.labels)) == [0, 1, 2]
        assert len(after.labels) == np.isin(before.trials, trial_of).sum()
print("dcca_data checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_dcca_data`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.dcca_data'`

- [ ] **Step 3: Write the implementation**

Create `src/experiments/dcca_data.py`:

```python
"""DCCA-AM inputs as the paper used them: raw DE-LDS EEG (310) + raw eye features (33).

The EEG comes from the BCMI ``eeg_used_4s`` files (the processed EEG is
z-scored per window, which the paper did not do). The eye features, labels
and clip ids come from the processed ``<culture>_EEG_EYE`` files, whose eye
columns are in raw units. Each recording's two sources must agree in length
and labels.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from src.experiments.film_identification import FILMS, TASKS
from src.preprocessing.convert_seed_crossculture import load_session

RAW_EEG = {
    "SEED_CHN_MM": Path("data/raw/SEED_CHN_MM/02-EEG-DE-feature/eeg_used_4s"),
    "SEED_GER": Path("data/raw/SEED_GER/02-EEG-DE-features/eeg_used_4s"),
    "SEED_FRA": Path("data/raw/SEED_FRA/02-EEG-DE-features/eeg_used_4s"),
}
PROCESSED = Path("data/processed")
FILM_COLUMN = {"SEED_GER": 1, "SEED_FRA": 2}  # position of the trial index in FILMS values


@dataclass
class Recording:
    subject: int
    session: int
    eeg: np.ndarray
    eye: np.ndarray
    labels: np.ndarray
    trials: np.ndarray


def load_recordings(culture):
    recordings = []
    for path in sorted(RAW_EEG[culture].glob("*_*.npz"), key=lambda p: tuple(map(int, p.stem.split("_")))):
        subject, session = map(int, path.stem.split("_"))
        features, labels, _ = load_session(path)
        processed = torch.load(PROCESSED / f"{culture}_EEG_EYE" / str(session) / f"subject{subject}.pth",
                               weights_only=True)
        x = torch.stack(processed["data"]).numpy()
        if len(x) != len(features) or list(labels) != list(processed["labels"]):
            raise ValueError(f"{culture} {path.stem}: raw EEG and processed eye disagree")
        recordings.append(Recording(subject, session, features.reshape(len(features), -1).astype(np.float32),
                                    x[:, 310:].astype(np.float32), np.asarray(labels), np.asarray(processed["trials"])))
    return recordings


def shared_trials(culture):
    return {values[FILM_COLUMN[culture]] for values in FILMS.values()}


def _keep(recording, rows, labels=None):
    return replace(recording, eeg=recording.eeg[rows], eye=recording.eye[rows],
                   labels=recording.labels[rows] if labels is None else labels, trials=recording.trials[rows])


def select(recordings, drop_shared_of=None, film_task=None, culture=None):
    """Drop the shared films, or keep only one film task's windows labelled by film index."""
    out = []
    for r in recordings:
        if drop_shared_of is not None:
            out.append(_keep(r, ~np.isin(r.trials, list(shared_trials(drop_shared_of)))))
        elif film_task is not None:
            trial_to_film = {FILMS[name][FILM_COLUMN[culture]]: i for i, name in enumerate(TASKS[film_task])}
            rows = np.isin(r.trials, list(trial_to_film))
            out.append(_keep(r, rows, np.array([trial_to_film[t] for t in r.trials[rows]])))
        else:
            out.append(r)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_dcca_data`
Expected: `dcca_data checks passed`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/dcca_data.py src/experiments/test_dcca_data.py
git commit -m "Load raw DE-LDS EEG paired with raw eye features for DCCA-AM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Training script that saves every epoch

**Files:**
- Create: `src/experiments/train_dcca_am.py`
- Test: `src/experiments/test_train_dcca_am.py`

**Interfaces:**
- Consumes: Task 1 (`DCCA_AM`, `make_optimizers`, `train_epoch`, `OUTPUT_DIM`); Task 2 (`load_recordings`, `select`); `src.experiments.train_scvae_reference.set_seed(seed)`; `src.experiments.film_identification.TASKS`.
- Produces:
  - `architecture(seed: int, index: int) -> tuple[int, int]`
  - `run_one(source: list[Recording], target: list[Recording], layer_sizes: tuple[int, int], categories: int, epochs: int, device: str, seed: int, index: int) -> dict` with keys `layer_sizes`, `skipped_batches`, `nonfinite_epochs`, `confusion` (list over epochs of `{str(subject): [[int]]}`), `final` (`{str(subject): {"pred": [...], "true": [...], "recording": [...]}}`)
  - `write_json(path: Path, record: dict) -> None` (atomic)
  - `run_tasks(task: str) -> list[str]`: `"emotion" → ["emotion"]`, `"emotion_newfilms" → ["emotion_newfilms"]`, `"film" → ["film_negative", "film_negative_2", "film_positive", "film_all_5"]`
  - CLI: `--source --target --task {emotion,emotion_newfilms,film} --seed --archs 0-20 --epochs 100 --device --output-root results/dcca_am --skip-existing`

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_train_dcca_am.py`:

```python
"""Checks for train_dcca_am on a tiny random problem.

    python -m src.experiments.test_train_dcca_am
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.experiments.dcca_data import Recording
from src.experiments.train_dcca_am import architecture, run_one, run_tasks, write_json

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

assert run_tasks("film") == ["film_negative", "film_negative_2", "film_positive", "film_all_5"]
print("train_dcca_am checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_train_dcca_am`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.train_dcca_am'`

- [ ] **Step 3: Write the implementation**

Create `src/experiments/train_dcca_am.py`:

```python
"""Train DCCA-AM across cultures and save every epoch's per-person results.

One run = (source, target, task, seed, architecture). A single network is
trained on the whole source culture; after every epoch it predicts every
target person, and the per-person confusion matrices are saved, so the
paper's selection (best test accuracy over architectures and epochs) and the
fair choice (architecture 0, last epoch) are both computed later from the
same runs, without retraining. Target labels are only used to fill the
confusion matrices.

Tasks: emotion (the paper's setting); emotion_newfilms (German <-> French,
source without the 5 shared films); film (German <-> French, identify the
shared film within each of film_identification.TASKS).

    python -m src.experiments.train_dcca_am --source SEED_GER --target SEED_FRA --task emotion --seed 12
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from src.experiments.dcca_data import load_recordings, select
from src.experiments.film_identification import TASKS as FILM_TASKS
from src.experiments.train_scvae_reference import set_seed
from src.models.dcca_am import DCCA_AM, OUTPUT_DIM, make_optimizers, train_epoch

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
FIXED_ARCHITECTURE = (150, 35)


def architecture(seed, index):
    if index == 0:
        return FIXED_ARCHITECTURE
    rng = np.random.default_rng([seed, index])
    return int(rng.integers(100, 200)), int(rng.integers(20, 50))


def run_tasks(task):
    return [f"film_{name}" for name in FILM_TASKS] if task == "film" else [task]


def scaled(recordings):
    """MinMax-scale EEG and eye, fitted on these recordings together (as released)."""
    eeg = np.concatenate([r.eeg for r in recordings])
    eye = np.concatenate([r.eye for r in recordings])
    return MinMaxScaler().fit_transform(eeg), MinMaxScaler().fit_transform(eye)


def run_one(source, target, layer_sizes, categories, epochs, device, seed, index):
    set_seed(1000 * seed + index)
    to = lambda a, dtype=torch.float: torch.as_tensor(a, dtype=dtype, device=device)
    src_eeg, src_eye = scaled(source)
    x1, x2 = to(src_eeg), to(src_eye)
    y = to(np.concatenate([r.labels for r in source]), torch.long)
    people = {}
    for subject in sorted({r.subject for r in target}):
        recs = [r for r in target if r.subject == subject]
        eeg, eye = scaled(recs)
        people[subject] = (to(eeg), to(eye), np.concatenate([r.labels for r in recs]),
                           np.concatenate([np.full(len(r.labels), 100 * r.session + r.subject) for r in recs]))

    model = DCCA_AM(x1.shape[1], x2.shape[1], [*layer_sizes, OUTPUT_DIM], categories).to(device)
    optimizers = make_optimizers(model)
    record = {"layer_sizes": list(layer_sizes), "skipped_batches": 0, "nonfinite_epochs": 0, "confusion": []}
    for _ in range(epochs):
        record["skipped_batches"] += train_epoch(model, optimizers, x1, x2, y)
        model.eval()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            record["nonfinite_epochs"] += 1
        epoch = {}
        for subject, (eeg, eye, labels, _) in people.items():
            pred = model.predict(eeg, eye).cpu().numpy()
            epoch[str(subject)] = confusion_matrix(labels, pred, labels=range(categories)).tolist()
        record["confusion"].append(epoch)
    record["final"] = {str(s): {"pred": model.predict(eeg, eye).cpu().numpy().tolist(), "true": labels.tolist(),
                                "recording": recording.tolist()}
                       for s, (eeg, eye, labels, recording) in people.items()}
    return record


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)


def parse_archs(text):
    first, last = map(int, text.split("-"))
    return range(first, last + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=CULTURES, required=True)
    parser.add_argument("--target", choices=CULTURES, required=True)
    parser.add_argument("--task", choices=["emotion", "emotion_newfilms", "film"], default="emotion")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--archs", default="0-20", help="Architecture indices, inclusive range (0 = fixed 150/35).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("source and target must differ (cross-culture only)")
    if args.task != "emotion" and {args.source, args.target} != {"SEED_GER", "SEED_FRA"}:
        parser.error(f"--task {args.task} needs SEED_GER <-> SEED_FRA (the shared films)")

    source_all, target_all = load_recordings(args.source), load_recordings(args.target)
    for task in run_tasks(args.task):
        if task == "emotion":
            source, target, categories = source_all, target_all, 3
        elif task == "emotion_newfilms":
            source, target, categories = select(source_all, drop_shared_of=args.source), target_all, 3
        else:
            film_task = task.removeprefix("film_")
            source = select(source_all, film_task=film_task, culture=args.source)
            target = select(target_all, film_task=film_task, culture=args.target)
            categories = len(FILM_TASKS[film_task])
        out_dir = args.output_root / args.target / f"{task}__from_{args.source}" / f"seed_{args.seed}"
        for index in parse_archs(args.archs):
            path = out_dir / f"arch_{index:02d}.json"
            if args.skip_existing and path.exists():
                continue
            start = time.time()
            record = run_one(source, target, architecture(args.seed, index), categories,
                             args.epochs, args.device, args.seed, index)
            record.update(source=args.source, target=args.target, task=task, seed=args.seed, arch=index,
                          epochs=args.epochs, seconds=round(time.time() - start, 1))
            write_json(path, record)
            last = record["confusion"][-1]
            acc = np.mean([np.trace(np.array(cm)) / np.sum(cm) for cm in last.values()])
            print(f"{task} {args.source}->{args.target} seed {args.seed} arch {index} {record['layer_sizes']}: "
                  f"last-epoch acc {100 * acc:.1f}%, skipped {record['skipped_batches']}, "
                  f"{record['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_train_dcca_am`
Expected: `train_dcca_am checks passed`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/train_dcca_am.py src/experiments/test_train_dcca_am.py
git commit -m "Train DCCA-AM across cultures, saving every epoch per test person

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Summary with the three evaluation modes

**Files:**
- Create: `src/experiments/summarize_dcca_am.py`
- Test: `src/experiments/test_summarize_dcca_am.py`

**Interfaces:**
- Consumes: JSON records written by Task 3; `src.experiments.film_identification.block_permutations(films, recordings, task_films, rng, n)` and `per_subject_accuracy(true, predicted, subjects)`; `results/tables/crossculture_summary.csv`; `results/tables/film_identification.json` (`{"tasks": [...], ...}`).
- Produces:
  - `load_runs(root: Path) -> dict[tuple[str, str, str], dict[int, dict[int, dict]]]` keyed `(target, task, source)` → seed → arch → record
  - `accuracy(cm: list[list[int]], balanced: bool) -> float`
  - `fair_score(arch_records: dict[int, dict], subject: str, balanced: bool) -> float` (arch 0, last epoch)
  - `paper_score(arch_records: dict[int, dict], subject: str, k: int, balanced: bool) -> float` (max over archs 1..k, all epochs)
  - `PAPER_DCCA: dict[tuple[str, str], float]` (source, target) → %
  - CLI writes `results/tables/dcca_am_summary.md`

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_summarize_dcca_am.py`:

```python
"""Checks for the DCCA-AM summary selection rules.

    python -m src.experiments.test_summarize_dcca_am
"""

from src.experiments.summarize_dcca_am import accuracy, fair_score, paper_score

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
print("summarize_dcca_am checks passed")
```

Note: `accuracy([[3, 1], [0, 0]], balanced=True)` has a class with no true windows; balanced accuracy averages recall over classes that occur (`3/4`), which is how sklearn's `balanced_accuracy_score` behaves.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_summarize_dcca_am`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.summarize_dcca_am'`

- [ ] **Step 3: Write the implementation**

Create `src/experiments/summarize_dcca_am.py`:

```python
"""Summarise the DCCA-AM runs under three evaluation modes, next to Liu et al.

  paper's rules    best test accuracy over architectures 1..k and all epochs, per test person (as released)
  fair             architecture 0 (150/35, fixed in advance), last epoch
  fair, new films  the same, trained without the 5 shared German/French films

Per test person, scores are averaged over seeds; tables give the mean over test
people with a 95% bootstrap CI. Also: the leakage curve (paper's rules for
k = 1, 2, 5, 10, 20) and film identification (balanced accuracy, block
permutation p), with the linear E2 model alongside.

    python -m src.experiments.summarize_dcca_am   # -> results/tables/dcca_am_summary.md
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.experiments.film_identification import block_permutations, per_subject_accuracy

NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}
DIRECTIONS = [("SEED_CHN_MM", "SEED_GER"), ("SEED_CHN_MM", "SEED_FRA"), ("SEED_GER", "SEED_CHN_MM"),
              ("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_CHN_MM"), ("SEED_FRA", "SEED_GER")]
# Liu et al. 2022, table 9 (CCSI, DCCA-AM).
PAPER_DCCA = {("SEED_CHN_MM", "SEED_FRA"): 73.84, ("SEED_CHN_MM", "SEED_GER"): 73.63,
              ("SEED_GER", "SEED_CHN_MM"): 84.28, ("SEED_GER", "SEED_FRA"): 77.06,
              ("SEED_FRA", "SEED_CHN_MM"): 82.09, ("SEED_FRA", "SEED_GER"): 77.92}
K_CURVE = [1, 2, 5, 10, 20]


def load_runs(root):
    runs = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/*__from_*/seed_*/arch_*.json"):
        record = json.loads(path.read_text())
        runs[(record["target"], record["task"], record["source"])][record["seed"]][record["arch"]] = record
    return runs


def accuracy(cm, balanced):
    cm = np.asarray(cm, dtype=float)
    if not balanced:
        return float(np.trace(cm) / cm.sum())
    totals = cm.sum(axis=1)
    return float(np.mean(np.diag(cm)[totals > 0] / totals[totals > 0]))


def fair_score(arch_records, subject, balanced):
    return accuracy(arch_records[0]["confusion"][-1][subject], balanced)


def paper_score(arch_records, subject, k, balanced):
    return max(accuracy(epoch[subject], balanced)
               for index in range(1, k + 1) if index in arch_records
               for epoch in arch_records[index]["confusion"])


def per_subject(seed_runs, score):
    subjects = sorted(next(iter(next(iter(seed_runs.values())).values()))["confusion"][0], key=int)
    return np.array([np.mean([score(archs, s) for archs in seed_runs.values()]) for s in subjects])


def mean_ci(values, rng):
    boot = rng.choice(values, (10000, len(values))).mean(axis=1)
    return f"{100 * values.mean():.1f} [{100 * np.percentile(boot, 2.5):.1f}, {100 * np.percentile(boot, 97.5):.1f}]"


def best_simple():
    """Best normalised simple EEG + eye model per direction (4 s), from crossculture_summary.csv."""
    best = {}
    for row in csv.DictReader(open("results/tables/crossculture_summary.csv")):
        if row["setting"] == "cross" and row["modality"] == "eeg+eye" and row["window"] == "4s" \
                and row["method"].endswith("__subjnorm"):
            key = (row["train"], row["test"])
            best[key] = max(best.get(key, 0.0), float(row["accuracy_mean"]))
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("results/tables/dcca_am_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    runs = load_runs(args.root)
    simple = best_simple()
    k_max = max(K_CURVE)

    lines = ["# DCCA-AM across cultures: paper vs paper's rules vs fair", "",
             "Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person.", "",
             "| Train → Test | Paper (reported) | Paper's rules (ours) | Fair | Fair, new films | Best simple, normalised |",
             "|---|---:|---:|---:|---:|---:|"]
    curve = ["", "## Leakage curve: paper's rules when searching k architectures", "",
             "| Train → Test | " + " | ".join(f"k = {k}" for k in K_CURVE) + " | Fair |",
             "|---|" + "---:|" * (len(K_CURVE) + 1)]
    for source, target in DIRECTIONS:
        seed_runs = runs.get((target, "emotion", source))
        if not seed_runs:
            continue
        paper = per_subject(seed_runs, lambda a, s: paper_score(a, s, k_max, False))
        fair = per_subject(seed_runs, lambda a, s: fair_score(a, s, False))
        newfilms = runs.get((target, "emotion_newfilms", source))
        new = mean_ci(per_subject(newfilms, lambda a, s: fair_score(a, s, False)), rng) if newfilms else "same as fair"
        ref = simple.get((NAMES[source], NAMES[target]))
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {PAPER_DCCA[(source, target)]:.1f} | "
                     f"{mean_ci(paper, rng)} | {mean_ci(fair, rng)} | {new} | {ref if ref else '–'} |")
        ks = [100 * per_subject(seed_runs, lambda a, s, k=k: paper_score(a, s, k, False)).mean() for k in K_CURVE]
        curve.append(f"| {NAMES[source]} → {NAMES[target]} | " + " | ".join(f"{v:.1f}" for v in ks)
                     + f" | {100 * fair.mean():.1f} |")

    e2 = {(r["train"], r["test"], r["task"]): r["accuracy"]
          for r in json.loads(Path("results/tables/film_identification.json").read_text())["tasks"]
          if r["modality"] == "EEG + eye"}
    film = ["", "## Film identification (German ↔ French shared films), balanced accuracy %", "",
            "| Train → Test | Task | Chance | DCCA-AM fair | p | DCCA-AM paper's rules | Linear (E2) |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for (target, task, source), seed_runs in sorted(runs.items()):
        if not task.startswith("film_"):
            continue
        first_record = next(iter(next(iter(seed_runs.values())).values()))
        categories = len(next(iter(first_record["confusion"][0].values())))
        fair = per_subject(seed_runs, lambda a, s: fair_score(a, s, True))
        paper = per_subject(seed_runs, lambda a, s: paper_score(a, s, k_max, True))
        final = next(iter(seed_runs.values()))[0]["final"]  # first seed, architecture 0
        pred = np.concatenate([final[s]["pred"] for s in sorted(final, key=int)])
        true = np.concatenate([final[s]["true"] for s in sorted(final, key=int)])
        rec = np.concatenate([final[s]["recording"] for s in sorted(final, key=int)])
        subj = np.concatenate([[int(s)] * len(final[s]["pred"]) for s in sorted(final, key=int)])
        observed = per_subject_accuracy(true, pred, subj).mean()
        null = [per_subject_accuracy(shuffled, pred, subj).mean()
                for shuffled in block_permutations(true, rec, list(range(categories)), rng, args.permutations)]
        p = (1 + sum(n >= observed for n in null)) / (1 + args.permutations)
        linear = e2.get((NAMES[source], NAMES[target], task.removeprefix("film_")))
        linear_text = f"{100 * linear:.1f}" if linear is not None else "–"
        film.append(f"| {NAMES[source]} → {NAMES[target]} | {task.removeprefix('film_')} | {100 / categories:.0f} | "
                    f"{mean_ci(fair, rng)} | {p:.4f} | {100 * paper.mean():.1f} | {linear_text} |")

    args.output.write_text("\n".join(lines + curve + film) + "\n")
    print("\n".join(lines + curve + film))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_summarize_dcca_am`
Expected: `summarize_dcca_am checks passed`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/summarize_dcca_am.py src/experiments/test_summarize_dcca_am.py
git commit -m "Summarise DCCA-AM under the paper's rules, fair, and fair on new films

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Local smoke run, cluster job files, submission

**Files:**
- Create: `cluster/dcca_array.sbatch`
- Create: `cluster/dcca_sweep.txt`

**Interfaces:**
- Consumes: the Task 3 CLI; `cluster/env.sh` (sets `PYTHON`); the 2-GPU-per-user QOS.
- Produces: 30 array tasks writing `results/dcca_am/...`.

- [ ] **Step 1: Smoke run end to end on the laptop (tiny), and time it**

Run:
```bash
time .venv/bin/python -m src.experiments.train_dcca_am --source SEED_GER --target SEED_FRA --task emotion \
  --seed 12 --archs 0-1 --epochs 5 --device cpu --output-root /private/tmp/claude-501/-Users-harshvardhandubey-Downloads-EEG-Emotion/71cfe907-9903-4070-bd35-771fc40ef16d/scratchpad/dcca_smoke
time .venv/bin/python -m src.experiments.train_dcca_am --source SEED_GER --target SEED_FRA --task film \
  --seed 12 --archs 0-1 --epochs 5 --device cpu --output-root /private/tmp/claude-501/-Users-harshvardhandubey-Downloads-EEG-Emotion/71cfe907-9903-4070-bd35-771fc40ef16d/scratchpad/dcca_smoke
.venv/bin/python -m src.experiments.summarize_dcca_am --root /private/tmp/claude-501/-Users-harshvardhandubey-Downloads-EEG-Emotion/71cfe907-9903-4070-bd35-771fc40ef16d/scratchpad/dcca_smoke \
  --permutations 50 --output /private/tmp/claude-501/-Users-harshvardhandubey-Downloads-EEG-Emotion/71cfe907-9903-4070-bd35-771fc40ef16d/scratchpad/dcca_smoke/summary.md
```
Expected: two lines per task/architecture with a last-epoch accuracy and seconds; a summary table with one German → French row and four film rows. Compute the full-run estimate as `seconds_per_epoch × 100 epochs × 21 architectures × 30 tasks` (for the Chinese-source directions scale by source size: CHN_MM ≈ 30k windows vs GER ≈ 17k). If the estimate exceeds 40 GPU-hours, change `--archs 0-20` to `--archs 0-10` in Step 2 and record the reason in the commit message.

- [ ] **Step 2: Write the cluster files**

Create `cluster/dcca_array.sbatch`:

```bash
#!/usr/bin/env bash
# DCCA-AM cross-culture re-run: task i runs line i of cluster/dcca_sweep.txt
# (all architectures 0-20, 100 epochs; finished architectures are skipped).
#
#   sbatch --array=1-$(grep -cv '^\s*\(#\|$\)' cluster/dcca_sweep.txt) cluster/dcca_array.sbatch
#
#SBATCH --job-name=dcca
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=results/logs/slurm/%x_%A_%a.out
#SBATCH --error=results/logs/slurm/%x_%A_%a.err
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
MANIFEST="${MANIFEST:-cluster/dcca_sweep.txt}"
GPU=1
source cluster/env.sh

ARGS="$(grep -v '^\s*\(#\|$\)' "$MANIFEST" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [ -z "$ARGS" ]; then
  echo "No manifest line ${SLURM_ARRAY_TASK_ID} in ${MANIFEST}" >&2
  exit 1
fi
echo "Task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${ARGS}"

# shellcheck disable=SC2086
"${PYTHON[@]}" -m src.experiments.train_dcca_am \
    --archs 0-20 --epochs 100 --device "${DEVICE:-cuda}" --skip-existing ${ARGS}
```

Create `cluster/dcca_sweep.txt`:

```text
# DCCA-AM fair re-run (docs/superpowers/specs/2026-10-01-dcca-am-fair-rerun-design.md).
# dcca_array.sbatch adds: --archs 0-20 --epochs 100 --device cuda --skip-existing
# emotion: 6 directions x 3 seeds
--source SEED_CHN_MM --target SEED_GER --task emotion --seed 12
--source SEED_CHN_MM --target SEED_GER --task emotion --seed 34
--source SEED_CHN_MM --target SEED_GER --task emotion --seed 56
--source SEED_CHN_MM --target SEED_FRA --task emotion --seed 12
--source SEED_CHN_MM --target SEED_FRA --task emotion --seed 34
--source SEED_CHN_MM --target SEED_FRA --task emotion --seed 56
--source SEED_GER --target SEED_CHN_MM --task emotion --seed 12
--source SEED_GER --target SEED_CHN_MM --task emotion --seed 34
--source SEED_GER --target SEED_CHN_MM --task emotion --seed 56
--source SEED_GER --target SEED_FRA --task emotion --seed 12
--source SEED_GER --target SEED_FRA --task emotion --seed 34
--source SEED_GER --target SEED_FRA --task emotion --seed 56
--source SEED_FRA --target SEED_CHN_MM --task emotion --seed 12
--source SEED_FRA --target SEED_CHN_MM --task emotion --seed 34
--source SEED_FRA --target SEED_CHN_MM --task emotion --seed 56
--source SEED_FRA --target SEED_GER --task emotion --seed 12
--source SEED_FRA --target SEED_GER --task emotion --seed 34
--source SEED_FRA --target SEED_GER --task emotion --seed 56
# emotion, new films: German <-> French without the 5 shared films in training
--source SEED_GER --target SEED_FRA --task emotion_newfilms --seed 12
--source SEED_GER --target SEED_FRA --task emotion_newfilms --seed 34
--source SEED_GER --target SEED_FRA --task emotion_newfilms --seed 56
--source SEED_FRA --target SEED_GER --task emotion_newfilms --seed 12
--source SEED_FRA --target SEED_GER --task emotion_newfilms --seed 34
--source SEED_FRA --target SEED_GER --task emotion_newfilms --seed 56
# film identification: German <-> French, the four film tasks of film_identification
--source SEED_GER --target SEED_FRA --task film --seed 12
--source SEED_GER --target SEED_FRA --task film --seed 34
--source SEED_GER --target SEED_FRA --task film --seed 56
--source SEED_FRA --target SEED_GER --task film --seed 12
--source SEED_FRA --target SEED_GER --task film --seed 34
--source SEED_FRA --target SEED_GER --task film --seed 56
```

Check the count: `grep -cv '^\s*\(#\|$\)' cluster/dcca_sweep.txt` → `30`.

- [ ] **Step 3: Commit**

```bash
git add cluster/dcca_array.sbatch cluster/dcca_sweep.txt
git commit -m "Add the DCCA-AM cluster array (30 tasks)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Push and submit once the leakage audits have finished**

Run:
```bash
ssh singularity 'squeue -h -u harsh.dubey -n audit1s,audit4s'   # must print nothing
cluster/sync.sh push
ssh singularity 'cd EEG-Emotion && sbatch --array=1-30 cluster/dcca_array.sbatch && squeue -u harsh.dubey'
```
Expected: `Submitted batch job <id>`; two tasks running, the rest pending on `QOSMaxGRESPerUser`. Two minutes later check the first `.err` files are empty and the `.out` files show `arch 0` lines:
```bash
ssh singularity 'cd EEG-Emotion && tail -3 results/logs/slurm/dcca_*_1.out results/logs/slurm/dcca_*_1.err'
```

---

### Task 6: Pull, verify, write up

**Files:**
- Modify: `EXPLANATION.md` (section 7, new subsection "DCCA-AM, re-run fairly"; section 8 status line)
- Create: `results/tables/dcca_am_summary.md` (generated)

- [ ] **Step 1: Pull and check completeness**

Run:
```bash
cluster/sync.sh pull
.venv/bin/python - <<'EOF'
from pathlib import Path
import json
files = list(Path("results/dcca_am").glob("*/*/seed_*/arch_*.json"))
print(len(files), "files; expected", (18 + 6) * 21 + 6 * 4 * 21)
bad = [f for f in files if json.loads(f.read_text())["nonfinite_epochs"] or not json.loads(f.read_text())["confusion"]]
print("runs with non-finite epochs:", len(bad))
print("skipped batches (max per run):", max(json.loads(f.read_text())["skipped_batches"] for f in files))
EOF
```
Expected: `1008 files; expected 1008` (18 emotion + 6 new-films tasks × 21 architectures, plus 6 film tasks × 4 film sub-tasks × 21). If fewer, resubmit the incomplete array indices (finished architectures are skipped). Report the non-finite and skipped counts in the write-up whatever they are.

- [ ] **Step 2: Generate the summary**

Run: `.venv/bin/python -m src.experiments.summarize_dcca_am`
Expected: `Saved: results/tables/dcca_am_summary.md` with 6 emotion rows, a 6-row leakage curve and 8 film rows.

- [ ] **Step 3: Write it up**

In `EXPLANATION.md`, add a subsection "DCCA-AM, re-run fairly" at the end of section 7 containing: the main table from `dcca_am_summary.md` (paper / paper's rules / fair / fair new films / best simple), the leakage curve table, the film-identification table, one paragraph per table stating what the numbers show (including when they contradict our expectations), and the counts of skipped batches and non-finite runs. In section 8, change the DCCA-AM item to done and cite `python -m src.experiments.summarize_dcca_am`.

- [ ] **Step 4: Commit**

```bash
git add results/dcca_am results/tables/dcca_am_summary.md EXPLANATION.md
git commit -m "Add the DCCA-AM cross-culture re-run: paper's rules vs fair vs new films

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
