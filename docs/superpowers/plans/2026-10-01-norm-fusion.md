# NormFusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run NormFusion — per-recording normalisation, separate EEG and eye experts, a learned late-fusion weight chosen by people-and-films-held-out validation on the training culture, and class-balanced self-training on the unlabelled test culture — in the 6 cross-culture directions, with ablations and the matched comparison against DCCA-AM.

**Architecture:** `src/baselines/norm_fusion.py` holds the model pieces (loading, experts, candidate fusion, film-robust selection, self-training). `src/experiments/run_norm_fusion.py` runs one (source, target, seed, setting) per call for all variants, in parallel over calls, and writes one JSON per variant. `src/experiments/summarize_norm_fusion.py` builds the tables and the comparison with `results/dcca_am`. Everything runs on the laptop CPU.

**Tech Stack:** Python 3 in `.venv`, NumPy, scikit-learn, PyTorch (existing `src/baselines/mlp.py` via `strict_baselines.fit_mlp`), SciPy, joblib.

**Spec:** `docs/superpowers/specs/2026-10-01-norm-fusion-model-design.md`

## Global Constraints

- Interpreter `.venv/bin/python`; tests are plain-assert modules run as `.venv/bin/python -m <module>` that print one "... checks passed" line (no pytest).
- Inputs: processed `<culture>_EEG_EYE` (EEG = columns 0:310, eye = 310:343), standardised per recording with `normalize_recording` (except the `no_norm` ablation).
- Logistic regression: standardiser + `LogisticRegression(C, class_weight="balanced", max_iter=2000)`, C ∈ {0.001, 0.01, 0.1, 1}.
- MLP: `strict_baselines.fit_mlp(x, y, 30, "cpu", seed)`, 3 seeds per expert: run seed s → MLP seeds s, s+1, s+2.
- Expert probability = mean of LR(C) and the 3 MLPs (LR alone in `lr_only`).
- Fusion `p = w · p_EEG + (1 − w) · p_eye`, w ∈ {0, 0.25, 0.5, 0.75, 1}.
- Selection: 3 person groups (sorted subjects dealt round-robin) × 3 film folds (`clip_heldout_emotion.clip_folds`); train on other groups' windows from other folds, validate on the group's windows from the fold; score = mean balanced accuracy over the 9 cells; ties go to the first candidate in grid order.
- Self-training: 3 rounds, threshold 0.9, per-class k = smallest pool, stop if k < 10, refit on source + chosen target windows with predicted labels (as `strict_baselines.fit_mlp_selftrain`), on the whole test culture's windows; test labels never passed in.
- Seeds 12, 34, 56. Directions: the 6 cross-culture pairs; "new films": German ↔ French with the 5 shared films removed from the source (`dcca_data.shared_trials`).
- Output `results/norm_fusion/<target>/<setting>__from_<source>/seed_<s>/<variant>.json`, setting ∈ {emotion, emotion_newfilms}.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No novelty claims in code or docs.

**Plan-level ruling on the spec (recorded here, carried to the ledger):** the spec's paper's-rules search includes "with and without self-training" for every candidate. Self-training each of the 80 fusion candidates would cost about 80× the fair run. The paper's-rules candidate set is therefore the 80 no-self-training candidates plus the fair-selected candidate with self-training (81). Cost if wrong: the paper's-rules number for NormFusion is slightly lower than a full search would give, which only makes the matched comparison harder for us, not easier.

## Review Focus

- A test culture where self-training finds fewer than 10 confident windows for some emotion must return the un-adapted prediction and record 0 rounds, not fail (pinned in Task 1, Step 1).
- Equal cross-validation scores (common with w = 0 or 1, where one C is irrelevant) must resolve the same way every run — the first candidate in grid order (pinned in Task 1, Step 1).
- After removing the 5 shared films, the German and French sources must still contain all three emotions in every film fold (pinned in Task 1, Step 1).
- Parallel runs must not oversubscribe the CPU: each worker sets `torch.set_num_threads(1)` (pinned in Task 2, Step 3, checked by the timing note in Task 4).
- The summary must work while `results/dcca_am` is missing or incomplete, printing "–" for DCCA-AM columns rather than failing (pinned in Task 3, Step 1).

---

### Task 1: Model pieces

**Files:**
- Create: `src/baselines/norm_fusion.py`
- Test: `src/baselines/test_norm_fusion.py`

**Interfaces:**
- Consumes: `src.baselines.strict_baselines.fit_mlp(x, y, epochs, device, seed) -> predict` (with `predict.proba(x) -> np.ndarray[n, 3]`); `src.experiments.clip_heldout_emotion.clip_folds(y, clips) -> np.ndarray[int]`; `src.experiments.dcca_data.shared_trials(culture) -> set[int]`; `src.experiments.train_scvae_reference.normalize_recording, sessions_in, available_subjects`.
- Produces:
  - `C_GRID`, `W_GRID`, `EEG`, `EYE` constants
  - `@dataclass Culture(x: np.ndarray[n,343], y, clips, subjects)`
  - `Config = namedtuple("Config", "c_eeg c_eye w")` (early fusion: `Config(c, None, None)`)
  - `load_culture(culture: str, normalise=True, drop_shared=False) -> Culture`
  - `expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=C_GRID) -> dict[float, np.ndarray]`
  - `candidate_probas(train_x, train_y, eval_x, seed, use_mlp, early) -> dict[Config, np.ndarray]`
  - `cv_cells(subjects, y, clips, mode) -> list[tuple[mask, mask]]`, mode ∈ {"person_film", "person"}
  - `select_config(data: Culture, seed, use_mlp, early, mode="person_film", keep=None) -> tuple[Config, dict[Config, float]]`
  - `config_proba(train_x, train_y, eval_x, cfg, seed, use_mlp, early) -> np.ndarray`
  - `self_train(src_x, src_y, tgt_x, cfg, seed, use_mlp, early) -> tuple[np.ndarray, list[int]]`

- [ ] **Step 1: Write the failing test**

Create `src/baselines/test_norm_fusion.py`:

```python
"""Checks for NormFusion's pieces on small synthetic data and on the real loaders.

    python -m src.baselines.test_norm_fusion
"""

import inspect

import numpy as np

from src.baselines.norm_fusion import (C_GRID, EEG, W_GRID, Config, Culture, candidate_probas, config_proba,
                                       cv_cells, expert_probas, load_culture, select_config, self_train)
from src.experiments.clip_heldout_emotion import clip_folds
from src.experiments.dcca_data import shared_trials

rng = np.random.default_rng(0)


def synthetic(n_subjects=6, clips_per_emotion=3, windows=20, signal=1.0):
    xs, ys, clips, subjects = [], [], [], []
    for s in range(1, n_subjects + 1):
        for emotion in range(3):
            for k in range(clips_per_emotion):
                x = rng.normal(size=(windows, 343))
                x[:, :5] += signal * emotion  # EEG carries the emotion
                xs.append(x); ys.append(np.full(windows, emotion))
                clips.append(np.full(windows, 1 + emotion * clips_per_emotion + k)); subjects.append(np.full(windows, s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects)])


data = synthetic()

# Film-robust cells never share a person or a clip between training and validation; 9 cells.
cells = cv_cells(data.subjects, data.y, data.clips, "person_film")
assert len(cells) == 9 and len(cv_cells(data.subjects, data.y, data.clips, "person")) == 3
for train, val in cells:
    assert val.any() and train.any()
    assert not set(data.subjects[train]) & set(data.subjects[val])
    assert not set(data.clips[train]) & set(data.clips[val])
    assert set(data.y[val]) == {0, 1, 2}

# Candidate fusion: probabilities sum to 1; w = 1 equals the EEG expert alone.
probas = candidate_probas(data.x[:200], data.y[:200], data.x[200:260], seed=12, use_mlp=False, early=False)
assert len(probas) == len(C_GRID) ** 2 * len(W_GRID)
assert all(np.allclose(p.sum(axis=1), 1) for p in probas.values())
eeg_only = expert_probas(data.x[:200, EEG], data.y[:200], data.x[200:260, EEG], seed=12, use_mlp=False)
assert np.allclose(probas[Config(0.1, 1.0, 1.0)], eeg_only[0.1])
assert np.allclose(config_proba(data.x[:200], data.y[:200], data.x[200:260], Config(0.1, 1.0, 1.0), 12, False, False),
                   eeg_only[0.1])

# Selection is deterministic, ties go to the first candidate, and `keep` restricts the search.
best, scores = select_config(data, seed=12, use_mlp=False, early=False)
assert (best, scores) == select_config(data, seed=12, use_mlp=False, early=False)
top = max(scores.values())
assert best == next(cfg for cfg in scores if scores[cfg] == top)
eye_best, _ = select_config(data, seed=12, use_mlp=False, early=False, keep=lambda cfg: cfg.w == 0.0)
assert eye_best.w == 0.0
early_best, early_scores = select_config(data, seed=12, use_mlp=False, early=True)
assert early_best.c_eye is None and len(early_scores) == len(C_GRID)

# Self-training never receives test labels, and with no signal it stops after 0 rounds.
assert list(inspect.signature(self_train).parameters) == ["src_x", "src_y", "tgt_x", "cfg", "seed", "use_mlp", "early"]
noise = synthetic(signal=0.0)
proba, added = self_train(noise.x[:540], noise.y[:540], noise.x[540:], Config(0.001, 0.001, 0.5), 12, False, False)
assert proba.shape == (len(noise.x) - 540, 3) and added == []
proba, added = self_train(data.x[:540], data.y[:540], data.x[540:], Config(1.0, 1.0, 1.0), 12, True, False)
assert proba.shape == (len(data.x) - 540, 3) and len(added) >= 1

# Real loaders: normalisation changes the features; dropping shared films keeps every emotion in every fold.
german = load_culture("SEED_GER")
assert german.x.shape[1] == 343 and abs(german.x.mean()) < 0.05
raw = load_culture("SEED_GER", normalise=False)
assert not np.allclose(raw.x, german.x)
for culture in ("SEED_GER", "SEED_FRA"):
    reduced = load_culture(culture, drop_shared=True)
    assert not set(np.unique(reduced.clips)) & shared_trials(culture)
    folds = clip_folds(reduced.y, reduced.clips)
    assert all(set(reduced.y[folds == f]) == {0, 1, 2} for f in range(3))
print("norm_fusion checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baselines.norm_fusion'`

- [ ] **Step 3: Write the implementation**

Create `src/baselines/norm_fusion.py`:

```python
"""NormFusion: a simple cross-culture EEG + eye emotion model.

Per-recording normalisation; an EEG expert and an eye expert (logistic
regression + 3 small MLPs each); late fusion p = w p_EEG + (1 - w) p_eye;
C and w chosen by cross-validation inside the training culture that holds
out people and films together; then class-balanced self-training on the
unlabelled test culture. Design: docs/superpowers/specs/2026-10-01-norm-fusion-model-design.md.
"""

import warnings
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.strict_baselines import fit_mlp
from src.experiments.clip_heldout_emotion import clip_folds
from src.experiments.dcca_data import shared_trials
from src.experiments.train_scvae_reference import available_subjects, normalize_recording, sessions_in

C_GRID = (0.001, 0.01, 0.1, 1.0)
W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EEG, EYE = slice(0, 310), slice(310, 343)
MLP_EPOCHS, MLP_SEEDS = 30, 3
ST_ROUNDS, ST_THRESHOLD, ST_MIN = 3, 0.9, 10
ROOTS = {"SEED_CHN_MM": "SEED_CHN_MM_EEG_EYE", "SEED_GER": "SEED_GER_EEG_EYE", "SEED_FRA": "SEED_FRA_EEG_EYE"}

Config = namedtuple("Config", "c_eeg c_eye w")


@dataclass
class Culture:
    x: np.ndarray
    y: np.ndarray
    clips: np.ndarray
    subjects: np.ndarray


def load_culture(culture, normalise=True, drop_shared=False):
    root = Path("data/processed") / ROOTS[culture]
    xs, ys, clips, subjects = [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = torch.stack(data["data"]).float().flatten(start_dim=1)
            xs.append((normalize_recording(x) if normalise else x).numpy())
            ys.append(np.asarray(data["labels"]))
            clips.append(np.asarray(data["trials"]))
            subjects.append(np.full(len(x), subject))
    data = Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects)])
    if drop_shared:
        keep = ~np.isin(data.clips, list(shared_trials(culture)))
        data = Culture(data.x[keep], data.y[keep], data.clips[keep], data.subjects[keep])
    return data


def fit_lr(x, y, c):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight="balanced", max_iter=2000)).fit(x, y)


def expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=C_GRID):
    """Per C: class probabilities on eval_x of (LR(C) + 3 MLPs) / 4, or of LR(C) alone."""
    lr = {c: fit_lr(train_x, train_y, c).predict_proba(eval_x) for c in cs}
    if not use_mlp:
        return lr
    mlp = sum(fit_mlp(train_x, train_y, MLP_EPOCHS, "cpu", seed + i).proba(eval_x) for i in range(MLP_SEEDS))
    return {c: (p + mlp) / (1 + MLP_SEEDS) for c, p in lr.items()}


def candidate_probas(train_x, train_y, eval_x, seed, use_mlp, early):
    """Fused probabilities on eval_x for every candidate, in grid order."""
    if early:
        return {Config(c, None, None): p for c, p in expert_probas(train_x, train_y, eval_x, seed, use_mlp).items()}
    eeg = expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp)
    eye = expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp)
    return {Config(ce, cy, w): w * eeg[ce] + (1 - w) * eye[cy] for ce in C_GRID for cy in C_GRID for w in W_GRID}


def cv_cells(subjects, y, clips, mode):
    """(train, validation) masks: 3 person groups x 3 film folds ("person_film"), or 3 person groups ("person")."""
    group_of = {s: i % 3 for i, s in enumerate(np.unique(subjects))}
    groups = np.array([group_of[s] for s in subjects])
    if mode == "person":
        return [(groups != g, groups == g) for g in range(3)]
    folds = clip_folds(y, clips)
    return [((groups != g) & (folds != f), (groups == g) & (folds == f)) for g in range(3) for f in range(3)]


def select_config(data, seed, use_mlp, early, mode="person_film", keep=None):
    """Mean validation balanced accuracy of each candidate; the best (first in grid order on ties)."""
    scores = defaultdict(list)
    for train, val in cv_cells(data.subjects, data.y, data.clips, mode):
        for cfg, proba in candidate_probas(data.x[train], data.y[train], data.x[val], seed, use_mlp, early).items():
            if keep is None or keep(cfg):
                scores[cfg].append(balanced_accuracy_score(data.y[val], proba.argmax(axis=1)))
    mean = {cfg: float(np.mean(v)) for cfg, v in scores.items()}
    return max(mean, key=mean.get), mean


def config_proba(train_x, train_y, eval_x, cfg, seed, use_mlp, early):
    """Probabilities on eval_x for one candidate, fitting only the experts it uses."""
    if early:
        return expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=(cfg.c_eeg,))[cfg.c_eeg]
    proba = 0
    if cfg.w > 0:
        proba = proba + cfg.w * expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp,
                                              cs=(cfg.c_eeg,))[cfg.c_eeg]
    if cfg.w < 1:
        proba = proba + (1 - cfg.w) * expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp,
                                                    cs=(cfg.c_eye,))[cfg.c_eye]
    return proba


def self_train(src_x, src_y, tgt_x, cfg, seed, use_mlp, early):
    """Class-balanced self-training on unlabelled target windows; returns (probabilities, windows added per round)."""
    proba = config_proba(src_x, src_y, tgt_x, cfg, seed, use_mlp, early)
    added = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        proba = config_proba(np.concatenate([src_x, tgt_x[chosen]]), np.concatenate([src_y, pseudo[chosen]]),
                             tgt_x, cfg, seed, use_mlp, early)
    return proba, added
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion`
Expected: `norm_fusion checks passed`

If the noise-case assertion `added == []` fails, the synthetic data (not the code) produced 10+ windows at ≥ 0.9 confidence in every class. Then replace that assertion with `assert all(a % 3 == 0 for a in added)` (additions are class-balanced) and record the change as a ledger ruling.

- [ ] **Step 5: Commit**

```bash
git add src/baselines/norm_fusion.py src/baselines/test_norm_fusion.py
git commit -m "Add NormFusion: experts, late fusion, film-robust selection, self-training

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Runner with variants

**Files:**
- Create: `src/experiments/run_norm_fusion.py`
- Test: `src/experiments/test_run_norm_fusion.py`

**Interfaces:**
- Consumes: Task 1 (all of `norm_fusion`); `src.experiments.train_dcca_am.write_json(path, record)`.
- Produces:
  - `VARIANTS = ["fair", "no_selftrain", "eeg_only", "eye_only", "lr_only", "early_fusion", "person_cv", "no_norm"]`
  - `score(true, proba, subjects) -> dict[str, dict]` per subject: `accuracy`, `macro_f1`, `confusion`
  - `run_direction(source: Culture, target: Culture, seed: int, variants: list[str], raw_source: Culture | None = None, raw_target: Culture | None = None) -> dict[str, dict]` (variant → record with `config`, `cv_score`, `added`, `per_subject`; the `fair` record also has `candidates`: `{str(Config): {subject: confusion}}`)
  - CLI: `--seeds 12 34 56 --jobs 4 --output-root results/norm_fusion --skip-existing`

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_run_norm_fusion.py`:

```python
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

# A run with only the fair variant does not compute the others.
assert set(run_direction(source, target, seed=12, variants=["fair"])) == {"fair"}

s = score(np.array([0, 1, 2, 2]), np.eye(3)[[0, 1, 1, 2]], np.array([7, 7, 7, 7]))
assert s["7"]["accuracy"] == 0.75 and np.array(s["7"]["confusion"]).trace() == 3
print("run_norm_fusion checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_run_norm_fusion`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.run_norm_fusion'`

- [ ] **Step 3: Write the implementation**

Create `src/experiments/run_norm_fusion.py`:

```python
"""Run NormFusion across cultures: fair result, ablations, and per-candidate test scores.

For each (source, target, seed) and setting (emotion; emotion_newfilms =
German <-> French without the 5 shared films in training), every variant is
written to results/norm_fusion/<target>/<setting>__from_<source>/seed_<s>/<variant>.json.

  fair          film-robust selection of (C_eeg, C_eye, w), then self-training
  no_selftrain  fair without self-training
  eeg_only      w = 1;   eye_only  w = 0
  lr_only       experts without the MLPs
  early_fusion  one expert on the concatenated features
  person_cv     selection by person-only cross-validation
  no_norm       without per-recording normalisation

The fair record also keeps the confusion matrix of every candidate on every
test person (the 80 fusion candidates without self-training plus the fair
choice with it), for the paper's-rules comparison. Test labels are used only
to score.

    python -m src.experiments.run_norm_fusion --jobs 4
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score

from src.baselines.norm_fusion import candidate_probas, load_culture, select_config, self_train
from src.experiments.train_dcca_am import write_json

VARIANTS = ["fair", "no_selftrain", "eeg_only", "eye_only", "lr_only", "early_fusion", "person_cv", "no_norm"]
CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
SEEDS = [12, 34, 56]


def score(true, proba, subjects):
    pred = proba.argmax(axis=1)
    return {str(s): {"accuracy": float((pred[subjects == s] == true[subjects == s]).mean()),
                     "macro_f1": float(f1_score(true[subjects == s], pred[subjects == s], average="macro")),
                     "confusion": confusion_matrix(true[subjects == s], pred[subjects == s], labels=range(3)).tolist()}
            for s in np.unique(subjects)}


def _record(cfg, cv_score, added, proba, target):
    return {"config": list(cfg), "cv_score": cv_score, "added": added,
            "per_subject": score(target.y, proba, target.subjects)}


def run_direction(source, target, seed, variants, raw_source=None, raw_target=None):
    records = {}
    adapt = lambda cfg, use_mlp=True, early=False, src=source, tgt=target: self_train(
        src.x, src.y, tgt.x, cfg, seed, use_mlp, early)
    if {"fair", "no_selftrain", "eeg_only", "eye_only"} & set(variants):
        best, scores = select_config(source, seed, use_mlp=True, early=False)
        if "fair" in variants or "no_selftrain" in variants:
            candidates = candidate_probas(source.x, source.y, target.x, seed, use_mlp=True, early=False)
            if "no_selftrain" in variants:
                records["no_selftrain"] = _record(best, scores[best], [], candidates[best], target)
            if "fair" in variants:
                proba, added = adapt(best)
                records["fair"] = _record(best, scores[best], added, proba, target)
                table = {str(tuple(cfg)): p for cfg, p in candidates.items()}
                table["fair_selftrained"] = proba
                records["fair"]["candidates"] = {
                    key: {s: v["confusion"] for s, v in score(target.y, p, target.subjects).items()}
                    for key, p in table.items()}
        for name, w in (("eeg_only", 1.0), ("eye_only", 0.0)):
            if name in variants:
                cfg = max((c for c in scores if c.w == w), key=scores.get)
                proba, added = adapt(cfg)
                records[name] = _record(cfg, scores[cfg], added, proba, target)
    for name, kwargs in (("lr_only", dict(use_mlp=False, early=False)), ("early_fusion", dict(use_mlp=True, early=True)),
                         ("person_cv", dict(use_mlp=True, early=False, mode="person"))):
        if name in variants:
            cfg, scores = select_config(source, seed, **kwargs)
            proba, added = adapt(cfg, kwargs["use_mlp"], kwargs["early"])
            records[name] = _record(cfg, scores[cfg], added, proba, target)
    if "no_norm" in variants:
        cfg, scores = select_config(raw_source, seed, use_mlp=True, early=False)
        proba, added = adapt(cfg, src=raw_source, tgt=raw_target)
        records["no_norm"] = _record(cfg, scores[cfg], added, proba, raw_target)
    return records


def job(source, target, seed, setting, output_root, skip_existing):
    torch.set_num_threads(1)
    variants = VARIANTS if setting == "emotion" else ["fair", "no_selftrain"]
    out = output_root / target / f"{setting}__from_{source}" / f"seed_{seed}"
    if skip_existing and all((out / f"{v}.json").exists() for v in variants):
        return f"skip {setting} {source}->{target} seed {seed}"
    src = load_culture(source, drop_shared=setting == "emotion_newfilms")
    tgt = load_culture(target)
    raw = dict(raw_source=load_culture(source, normalise=False), raw_target=load_culture(target, normalise=False)) \
        if "no_norm" in variants else {}
    records = run_direction(src, tgt, seed, variants, **raw)
    for name, record in records.items():
        write_json(out / f"{name}.json", {**record, "variant": name, "source": source, "target": target,
                                          "seed": seed, "setting": setting})
    fair = np.mean([v["accuracy"] for v in records["fair"]["per_subject"].values()])
    return f"{setting} {source}->{target} seed {seed}: fair {100 * fair:.1f}% config {records['fair']['config']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    calls = [(s, t, seed, "emotion") for s, t in itertools.permutations(CULTURES, 2) for seed in args.seeds]
    calls += [(s, t, seed, "emotion_newfilms") for s, t in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER"))
              for seed in args.seeds]
    for line in Parallel(n_jobs=args.jobs)(delayed(job)(*call, args.output_root, args.skip_existing) for call in calls):
        print(line, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_run_norm_fusion`
Expected: `run_norm_fusion checks passed`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/run_norm_fusion.py src/experiments/test_run_norm_fusion.py
git commit -m "Run NormFusion across cultures with ablations and per-candidate scores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Summary and the comparison with DCCA-AM

**Files:**
- Create: `src/experiments/summarize_norm_fusion.py`
- Test: `src/experiments/test_summarize_norm_fusion.py`

**Interfaces:**
- Consumes: Task 2 records; `src.experiments.summarize_dcca_am.load_runs(root)`, `fair_score(arch_records, subject, balanced)`, `paper_score(arch_records, subject, k, balanced)`, `accuracy(cm, balanced)`, `PAPER_DCCA`, `NAMES`, `DIRECTIONS`, `best_simple()`, `mean_ci(values, rng)`.
- Produces:
  - `load_nf(root: Path) -> dict[tuple[str, str, str], dict[int, dict[str, dict]]]` keyed (target, setting, source) → seed → variant → record
  - `nf_per_subject(seed_records: dict[int, dict[str, dict]], variant: str, metric: str = "accuracy") -> dict[str, float]` (seed-averaged metric per subject; metric ∈ {"accuracy", "macro_f1"})
  - `nf_paper_rules(seed_records) -> dict[str, float]` (per seed, best candidate per subject; then averaged over seeds)
  - `dcca_per_subject(dcca_runs, key, mode) -> dict[str, float] | None`, mode ∈ {"fair", "paper"}
  - CLI writes `results/tables/norm_fusion_summary.md`

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_summarize_norm_fusion.py`:

```python
"""Checks for summarize_norm_fusion.

    python -m src.experiments.test_summarize_norm_fusion
"""

from src.experiments.summarize_norm_fusion import dcca_per_subject, nf_paper_rules, nf_per_subject

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
dcca = {("SEED_FRA", "emotion", "SEED_GER"): {12: {0: {"confusion": [{"1": half}, {"1": good}]},
                                                    1: {"confusion": [{"1": good}, {"1": half}]}}}}
assert dcca_per_subject(dcca, ("SEED_FRA", "emotion", "SEED_GER"), "fair") == {"1": 1.0}
assert dcca_per_subject(dcca, ("SEED_FRA", "emotion", "SEED_GER"), "paper") == {"1": 1.0}
print("summarize_norm_fusion checks passed")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_summarize_norm_fusion`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.summarize_norm_fusion'`

- [ ] **Step 3: Write the implementation**

Create `src/experiments/summarize_norm_fusion.py`:

```python
"""Summarise NormFusion and compare it with DCCA-AM under the same evaluation.

Main table per direction: the paper's reported DCCA-AM, DCCA-AM and NormFusion
under the paper's rules, DCCA-AM and NormFusion fair, a paired Wilcoxon test of
the two fair columns over test people, and the best simple normalised model.
Then the ablations (fair accuracy of each variant) and the new-films setting.
DCCA-AM columns show "–" until results/dcca_am exists.

    python -m src.experiments.summarize_norm_fusion   # -> results/tables/norm_fusion_summary.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.run_norm_fusion import VARIANTS
from src.experiments.summarize_dcca_am import (DIRECTIONS, NAMES, PAPER_DCCA, accuracy, best_simple, fair_score,
                                               load_runs, mean_ci, paper_score)


def load_nf(root):
    runs = defaultdict(lambda: defaultdict(dict))
    for path in root.glob("*/*__from_*/seed_*/*.json"):
        record = json.loads(path.read_text())
        runs[(record["target"], record["setting"], record["source"])][record["seed"]][record["variant"]] = record
    return runs


def nf_per_subject(seed_records, variant, metric="accuracy"):
    subjects = sorted(next(iter(seed_records.values()))[variant]["per_subject"], key=int)
    return {s: float(np.mean([r[variant]["per_subject"][s][metric] for r in seed_records.values()]))
            for s in subjects}


def nf_paper_rules(seed_records):
    per_seed = [{s: max(accuracy(table[s], False) for table in r["fair"]["candidates"].values())
                 for s in r["fair"]["per_subject"]} for r in seed_records.values()]
    return {s: float(np.mean([p[s] for p in per_seed])) for s in sorted(per_seed[0], key=int)}


def dcca_per_subject(dcca_runs, key, mode):
    seed_runs = dcca_runs.get(key)
    if not seed_runs:
        return None
    first = next(iter(next(iter(seed_runs.values())).values()))
    subjects = sorted(first["confusion"][0], key=int)
    k = max(max(archs) for archs in seed_runs.values())
    score = (lambda a, s: fair_score(a, s, False)) if mode == "fair" else (lambda a, s: paper_score(a, s, k, False))
    return {s: float(np.mean([score(archs, s) for archs in seed_runs.values()])) for s in subjects}


def fmt(values, rng):
    return mean_ci(np.array(list(values.values())), rng) if values else "–"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--dcca-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/norm_fusion_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    nf, dcca, simple = load_nf(args.root), load_runs(args.dcca_root), best_simple()

    main_table = ["# NormFusion vs DCCA-AM across cultures (EEG + eye)", "",
                  "Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
                  "p: paired Wilcoxon, NormFusion fair vs DCCA-AM fair, over test people.", "",
                  "| Train → Test | DCCA-AM reported | DCCA-AM paper's rules | NormFusion paper's rules "
                  "| DCCA-AM fair | NormFusion fair | NormFusion fair macro-F1 | p | Best simple |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    ablation = ["", "## Ablations (fair accuracy %, mean over test people)", "",
                "| Train → Test | " + " | ".join(VARIANTS) + " |", "|---|" + "---:|" * len(VARIANTS)]
    averages = defaultdict(list)
    for source, target in DIRECTIONS:
        seed_records = nf.get((target, "emotion", source))
        if not seed_records:
            continue
        ours = nf_per_subject(seed_records, "fair")
        theirs = dcca_per_subject(dcca, (target, "emotion", source), "fair")
        p = "–"
        if theirs:
            diff = [ours[s] - theirs[s] for s in ours]
            p = f"{wilcoxon(diff).pvalue:.3f}" if any(diff) else "1"
        ref = simple.get((NAMES[source], NAMES[target]))
        main_table.append(
            f"| {NAMES[source]} → {NAMES[target]} | {PAPER_DCCA[(source, target)]:.1f} | "
            f"{fmt(dcca_per_subject(dcca, (target, 'emotion', source), 'paper'), rng)} | "
            f"{fmt(nf_paper_rules(seed_records), rng)} | {fmt(theirs, rng)} | {fmt(ours, rng)} | "
            f"{fmt(nf_per_subject(seed_records, 'fair', 'macro_f1'), rng)} | {p} | "
            f"{f'{ref:.1f}' if ref else '–'} |")
        averages["NormFusion fair"].append(np.mean(list(ours.values())))
        if theirs:
            averages["DCCA-AM fair"].append(np.mean(list(theirs.values())))
        row = [100 * np.mean(list(nf_per_subject(seed_records, v).values())) for v in VARIANTS]
        ablation.append(f"| {NAMES[source]} → {NAMES[target]} | " + " | ".join(f"{v:.1f}" for v in row) + " |")
    main_table += ["", "Average over directions: " + ", ".join(
        f"{name} {100 * np.mean(v):.1f}% ({len(v)} directions)" for name, v in averages.items())]

    newfilms = ["", "## New films (German ↔ French, shared films removed from training), fair accuracy %", "",
                "| Train → Test | NormFusion, all films | NormFusion, new films | DCCA-AM, new films |", "|---|---:|---:|---:|"]
    for source, target in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")):
        seed_records = nf.get((target, "emotion_newfilms", source))
        if not seed_records:
            continue
        newfilms.append(f"| {NAMES[source]} → {NAMES[target]} | "
                        f"{fmt(nf_per_subject(nf[(target, 'emotion', source)], 'fair'), rng)} | "
                        f"{fmt(nf_per_subject(seed_records, 'fair'), rng)} | "
                        f"{fmt(dcca_per_subject(dcca, (target, 'emotion_newfilms', source), 'fair'), rng)} |")

    text = "\n".join(main_table + ablation + newfilms) + "\n"
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_summarize_norm_fusion`
Expected: `summarize_norm_fusion checks passed`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/summarize_norm_fusion.py src/experiments/test_summarize_norm_fusion.py
git commit -m "Summarise NormFusion against DCCA-AM under matched evaluation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Run, verify, write up

**Files:**
- Create (generated): `results/norm_fusion/**`, `results/tables/norm_fusion_summary.md`
- Modify: `EXPLANATION.md` (new section "NormFusion" after section 7; section 8 status)

- [ ] **Step 1: Time one call, then run everything**

Run:
```bash
time .venv/bin/python -c "
from pathlib import Path
from src.experiments.run_norm_fusion import job
print(job('SEED_GER', 'SEED_FRA', 12, 'emotion', Path('/private/tmp/claude-501/-Users-harshvardhandubey-Downloads-EEG-Emotion/71cfe907-9903-4070-bd35-771fc40ef16d/scratchpad/nf_smoke'), False))"
```
Expected: one line `emotion SEED_GER->SEED_FRA seed 12: fair <acc>% config [...]`, in roughly 10–20 minutes. Total = 24 calls (18 emotion + 6 new-films, which are about a quarter of the work) ≈ 20 × that time ÷ `--jobs`. With `--jobs 4` (each worker single-threaded) expect about 1–2 hours; if it is much longer, use `--jobs 6` on an 8-core laptop.

Then run, in the background:
```bash
.venv/bin/python -m src.experiments.run_norm_fusion --jobs 4 --skip-existing > results/logs/norm_fusion.log 2>&1
```
Expected: 24 result lines in `results/logs/norm_fusion.log`, no tracebacks.

- [ ] **Step 2: Check completeness**

Run:
```bash
.venv/bin/python - <<'EOF'
from pathlib import Path
files = list(Path("results/norm_fusion").glob("*/*/seed_*/*.json"))
print(len(files), "files; expected", 18 * 8 + 6 * 2)
EOF
```
Expected: `156 files; expected 156`.

- [ ] **Step 3: Summarise**

Run: `.venv/bin/python -m src.experiments.summarize_norm_fusion`
Expected: `Saved: results/tables/norm_fusion_summary.md` with 6 main rows, 6 ablation rows and 2 new-films rows (DCCA-AM columns "–" if its cluster runs are not yet pulled; rerun this step after they are).

- [ ] **Step 4: Write it up**

In `EXPLANATION.md`, add a section "NormFusion: our model" after section 7 containing: the 5-part description from the spec in plain words; the main table (with the honesty constraint stated: the design was informed by earlier analyses of the same data, and all choices inside a run use the training culture only); the ablation table with one sentence per variant on what removing it does; the new-films table; and whether the success criterion (better than DCCA-AM fair on average and in at least 4 of 6 directions) is met — stated plainly either way. In section 8, add the NormFusion item as done.

- [ ] **Step 5: Commit**

```bash
git add results/norm_fusion results/tables/norm_fusion_summary.md EXPLANATION.md
git commit -m "Add NormFusion results, ablations and the comparison with DCCA-AM

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
