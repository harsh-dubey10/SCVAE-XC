# NormFusion v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run NormFusion v2 — v1 plus temporal smoothing within each recording, per-emotion EEG/eye fusion weights with early fusion as an alternative, and two-culture training — chosen by v1's film-robust cross-validation, with ablations, a v1 re-run that must reproduce the saved v1 results, and a summary against v1 and DCCA-AM.

**Architecture:** `src/baselines/norm_fusion_v2.py` adds the v2 pieces (smoothing, per-emotion fusion, one candidate grid that contains every variant's candidates, CV scoring of the grid, candidate fitting, self-training on smoothed probabilities, two-culture data) and reuses v1's experts, CV cells and self-training rule from `src/baselines/norm_fusion.py`. One cross-validation pass per run scores the whole grid; each variant is a filter over that grid. `src/experiments/run_norm_fusion_v2.py` runs one (source, target, seed, setting) per call and writes one JSON per variant; `src/experiments/summarize_norm_fusion_v2.py` builds the tables. Laptop CPU.

**Tech Stack:** Python 3 in `.venv`, NumPy, scikit-learn, PyTorch (via `strict_baselines.fit_mlp`), SciPy, joblib.

**Spec:** `docs/superpowers/specs/2026-10-01-norm-fusion-v2-design.md` (builds on `docs/superpowers/specs/2026-10-01-norm-fusion-model-design.md` and the v1 plan `docs/superpowers/plans/2026-10-01-norm-fusion.md`).

## Global Constraints

- Interpreter `.venv/bin/python`; tests are plain-assert modules run as `.venv/bin/python -m <module>` that print one "... checks passed" line (no pytest).
- Everything in v1 is kept unless changed: per-recording normalisation; EEG expert (columns 0:310) and eye expert (310:343), each LR(C) + 3 MLPs averaged, C ∈ {0.001, 0.01, 0.1, 1}; self-training 3 rounds, threshold 0.9, per-class k = smallest pool, stop if k < 10, on the whole test culture.
- Smoothing: within each recording, in presentation order, mean of the class probabilities over a centred window of length k ∈ {1, 3, 5, 9, 15}; edges use the windows that exist; k = 1 is no smoothing. Only window order is used, never clip boundaries or labels. Causal (trailing window) only as an ablation.
- Self-training picks its pseudo-labels from the smoothed probabilities.
- Per-emotion late fusion: p_c ∝ w_c · p_EEG,c + (1 − w_c) · p_eye,c, w_c ∈ {0, 0.5, 1} independently for negative (label 0), neutral (1), positive (2) — 27 combinations — renormalised to sum to 1. Early fusion: one expert on all 343 features.
- Selection: v1's 3 person groups × 3 film folds; train on other groups' windows from other folds; the model predicts every window of the held-out people's recordings, predictions are smoothed within each recording, and only the held-out fold's windows are scored; mean balanced accuracy over the 9 cells; ties to the first candidate in grid order. Jointly chosen: (C_EEG, C_eye, fusion, k).
- Two-culture training: train on two cultures, test on the third; each recording normalised on its own as before; person groups dealt across both cultures; films identified by (culture, clip).
- Settings: the 6 one-to-one directions; the 3 two-culture settings; German ↔ French without the 5 shared films in training (new films). Seeds 12, 34, 56.
- Variants (fair): `fair`; `no_smoothing` (k = 1); `causal` (trailing smoothing); `v1_fusion` (single weight w ∈ {0, 0.25, 0.5, 0.75, 1}, no per-emotion weights, no early fusion, smoothing chosen); `no_selftrain`; `v1` (k = 1, single weight, no early fusion) in every setting. In the one-to-one directions `v1` must reproduce the saved v1 results (`results/norm_fusion/<target>/emotion__from_<source>/seed_<s>/fair.json`).
- Paper's rules: the best candidate per test person over all fair-grid candidates (C_EEG, C_eye, fusion, k) without self-training, plus the fair choice with it.
- Window-level accuracy is the headline. Clip-level accuracy (majority vote of each clip's windows, using known clip boundaries at test time) is reported separately, clearly labelled, never compared with a window-level number.
- Reported: accuracy and macro-F1 per setting, mean over test people with 95% bootstrap CIs; paired Wilcoxon v2 vs v1 per one-to-one direction; DCCA-AM fair and paper's rules where complete. The two-culture setting is compared only with v1 in the same setting.
- v2 is reported as a post-hoc model, v1 stays the primary pre-specified result (spec's honesty constraint); the summary carries that sentence.
- Output `results/norm_fusion_v2/<target>/<setting>__from_<source>/seed_<s>/<variant>.json`, setting ∈ {emotion, emotion_two_cultures, emotion_newfilms}; a two-culture source is written `SEED_A+SEED_B`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No novelty claims in code or docs.

**Data facts this plan relies on (checked 2026-10-02):** each processed file `data/processed/<culture>_EEG_EYE/<session>/subject<n>.pth` is one recording, with windows stored in presentation order (trial index ascending 1..N, label sequence matching SEED's clip order). Sessions 1–3; German session 3 has only people 1–4, so Germany has 20 recordings, China 36, France 24.

**Plan-level rulings on the spec (record them in the ledger):**
1. *One grid, filtered per variant.* The cross-validation scores one grid that is the union of every variant's candidates: late fusion over C_EEG × C_eye × (the 27 per-emotion weight triples ∪ v1's 5 equal triples, sorted) × smoothing, then early fusion over C × smoothing; smoothing = every k centred, then every k > 1 causal. Each variant chooses the best candidate inside its own subset, first in grid order on ties. Within the `v1` subset the order is v1's (C_EEG, C_eye, w ascending), so ties resolve as in v1. Cost if wrong: none for correctness — each variant's choice depends only on its own subset's scores.
2. *Paper's-rules storage.* The fair record stores each test person's plain accuracy for every fair-grid candidate without self-training (2,180 candidates, rounded to 4 decimals) instead of confusion matrices, to keep results near 15 MB. Cost if wrong: no balanced-accuracy variant of the paper's-rules number for v2 (the v1 summary uses plain accuracy too).
3. *New films* runs `fair`, `no_selftrain` and `v1` (v1 ran `fair` and `no_selftrain` there), so v2 vs v1 is matched on the same data.

## Review Focus

1. **Recordings that are not contiguous in the array** (the CV's held-out mask and `drop_shared` both cut recordings; the two-culture data interleaves nothing but the code must not assume it) — smoothing must follow each recording's own windows in array order and never mix two recordings. Pinned in Task 2 (interleaved recordings test).
2. **k longer than a recording** (a recording of 2 windows with k = 15) — the mean must use the windows that exist, with no NaN or index error. Pinned in Task 2.
3. **Person, recording or film ids colliding across cultures** in two-culture training (person 1 of China and person 1 of Germany would share a CV group; clip 3 of each would be treated as one film) — ids must be offset by culture. Pinned in Task 1 (synthetic) and the real-data check in Task 1.
4. **Uneven per-emotion weights** (e.g. (1, 0, 0.5)) give unnormalised probabilities that would distort the 0.9 self-training threshold — fused probabilities must sum to 1 for all 27 triples, and equal triples must give v1's numbers bit for bit. Pinned in Task 2.
5. **Test information reaching a choice** through the new paths (smoothing uses target recordings; clip-level scoring uses target clips) — with target labels shuffled every variant must pick the same candidate, add the same pseudo-labelled windows and make the same predictions. Pinned in Task 5.

---

### Task 1: Recording ids and two-culture data

**Files:**
- Modify: `src/baselines/norm_fusion.py` (the `Culture` dataclass and `load_culture`)
- Modify: `src/baselines/test_norm_fusion.py` (real-loader section)
- Create: `src/baselines/norm_fusion_v2.py`
- Create: `src/baselines/test_norm_fusion_v2.py`

**Interfaces:**
- Consumes: v1 `Culture(x, y, clips, subjects)`, `load_culture(culture, normalise=True, drop_shared=False)`.
- Produces: `Culture` gains a fifth field `recordings: np.ndarray = None` (one id per window, `1000 * session + subject` from `load_culture`); `norm_fusion_v2.combine(cultures: list[Culture]) -> Culture` with clips, subjects and recordings offset by `OFFSET * index` (`OFFSET = 1_000_000`), first culture unchanged.

- [ ] **Step 1: Write the failing tests**

Append to the real-loader section of `src/baselines/test_norm_fusion.py`, just before the final `print`:

```python
# Each window carries its recording; every recording is one person's contiguous block in presentation order.
assert len(german.recordings) == len(german.x) and len(np.unique(german.recordings)) == 20
for r in np.unique(german.recordings):
    idx = np.flatnonzero(german.recordings == r)
    assert (np.diff(idx) == 1).all() and (np.diff(german.clips[idx]) >= 0).all()
    assert len(set(german.subjects[idx])) == 1
reduced = load_culture("SEED_GER", drop_shared=True)
assert len(reduced.recordings) == len(reduced.x)
```

Create `src/baselines/test_norm_fusion_v2.py`:

```python
"""Checks for NormFusion v2's pieces on small synthetic data and on the real loaders.

    python -m src.baselines.test_norm_fusion_v2
"""

import numpy as np

from src.baselines.norm_fusion import Culture, cv_cells, load_culture
from src.baselines.norm_fusion_v2 import OFFSET, combine

rng = np.random.default_rng(0)


def synthetic(n_subjects=6, clips_per_emotion=3, windows=20, signal=1.0, offset=0.0):
    """One recording per person, clips in a fixed mixed order, EEG (and weakly eye) carrying the emotion."""
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    order = [(e, k) for k in range(clips_per_emotion) for e in (2, 0, 1)]
    for s in range(1, n_subjects + 1):
        for emotion, k in order:
            x = rng.normal(size=(windows, 343)) + offset
            x[:, :5] += signal * emotion
            x[:, 310:315] += 0.5 * signal * emotion
            xs.append(x); ys.append(np.full(windows, emotion))
            clips.append(np.full(windows, 1 + emotion * clips_per_emotion + k))
            subjects.append(np.full(windows, s)); recordings.append(np.full(windows, 1000 + s))
    return Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])


# Two cultures as one training set: ids stay apart, the first culture's ids are unchanged.
a, b = synthetic(4), synthetic(3)
both = combine([a, b])
assert len(both.x) == len(a.x) + len(b.x)
assert (both.subjects[:len(a.x)] == a.subjects).all() and (both.clips[:len(a.x)] == a.clips).all()
assert (both.subjects[len(a.x):] == b.subjects + OFFSET).all()
assert (both.recordings[len(a.x):] == b.recordings + OFFSET).all()
assert len(np.unique(both.subjects)) == 7 and len(np.unique(both.clips)) == 18
# CV cells over the combined set never put a person or a (culture, clip) on both sides.
for train, val in cv_cells(both.subjects, both.y, both.clips, "person_film"):
    assert not set(both.subjects[train]) & set(both.subjects[val])
    assert not set(both.clips[train]) & set(both.clips[val])

# Real data: in two-culture training each film fold holds films of both cultures.
from src.experiments.clip_heldout_emotion import clip_folds
real = combine([load_culture("SEED_CHN_MM"), load_culture("SEED_GER")])
folds = clip_folds(real.y, real.clips)
for f in range(3):
    held = np.unique(real.clips[folds == f])
    assert (held < OFFSET).any() and (held >= OFFSET).any()
print("norm_fusion_v2 checks passed")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion`
Expected: FAIL with `AttributeError: 'Culture' object has no attribute 'recordings'`.

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.baselines.norm_fusion_v2'`.

- [ ] **Step 3: Implement**

In `src/baselines/norm_fusion.py`, give `Culture` a recordings field and fill it in `load_culture`:

```python
@dataclass
class Culture:
    x: np.ndarray
    y: np.ndarray
    clips: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray = None
```

```python
def load_culture(culture, normalise=True, drop_shared=False):
    root = Path("data/processed") / ROOTS[culture]
    xs, ys, clips, subjects, recordings = [], [], [], [], []
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            x = torch.stack(data["data"]).float().flatten(start_dim=1)
            xs.append((normalize_recording(x) if normalise else x).numpy())
            ys.append(np.asarray(data["labels"]))
            clips.append(np.asarray(data["trials"]))
            subjects.append(np.full(len(x), subject))
            recordings.append(np.full(len(x), 1000 * int(session) + int(subject)))
    data = Culture(*[np.concatenate(v) for v in (xs, ys, clips, subjects, recordings)])
    if drop_shared:
        keep = ~np.isin(data.clips, list(shared_trials(culture)))
        data = Culture(data.x[keep], data.y[keep], data.clips[keep], data.subjects[keep], data.recordings[keep])
    return data
```

Create `src/baselines/norm_fusion_v2.py`:

```python
"""NormFusion v2: v1 plus temporal smoothing, per-emotion fusion weights, early fusion as a candidate,
and training on two cultures.

Reuses v1's experts, cross-validation cells and self-training rule (src/baselines/norm_fusion.py).
Design: docs/superpowers/specs/2026-10-01-norm-fusion-v2-design.md.
"""

import numpy as np

from src.baselines.norm_fusion import Culture

OFFSET = 1_000_000  # added per culture index to clip, person and recording ids in two-culture training


def combine(cultures):
    """Several cultures as one training set; films are (culture, clip), people and recordings likewise."""
    shift = lambda field: np.concatenate([getattr(c, field) + OFFSET * i for i, c in enumerate(cultures)])
    return Culture(np.concatenate([c.x for c in cultures]), np.concatenate([c.y for c in cultures]),
                   shift("clips"), shift("subjects"), shift("recordings"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion && .venv/bin/python -m src.baselines.test_norm_fusion_v2 && .venv/bin/python -m src.experiments.test_run_norm_fusion`
Expected: `norm_fusion checks passed`, `norm_fusion_v2 checks passed`, `run_norm_fusion checks passed` (v1 is unaffected by the new field).

- [ ] **Step 5: Commit**

```bash
git add src/baselines/norm_fusion.py src/baselines/test_norm_fusion.py src/baselines/norm_fusion_v2.py src/baselines/test_norm_fusion_v2.py
git commit -m "NormFusion v2: recording ids per window and two-culture training data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Smoothing and per-emotion fusion

**Files:**
- Modify: `src/baselines/norm_fusion_v2.py`
- Modify: `src/baselines/test_norm_fusion_v2.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `smooth(proba: (n, 3) array, recordings: (n,) array, k: int, causal: bool = False) -> (n, 3) array` (returns `proba` itself when k = 1); `fuse(eeg, eye, weights: tuple[float, float, float]) -> (n, 3) array` (`eeg` may be None when every w_c is 0, `eye` None when every w_c is 1); constants `SMOOTH_K = (1, 3, 5, 9, 15)`, `EMOTION_WEIGHTS` (27 triples, `itertools.product` order), `V1_WEIGHTS` (5 equal triples from v1's `W_GRID`), `LATE_WEIGHTS = tuple(sorted(set(EMOTION_WEIGHTS) | set(V1_WEIGHTS)))` (29 triples).

- [ ] **Step 1: Write the failing tests**

In `src/baselines/test_norm_fusion_v2.py`, extend the import line to

```python
from src.baselines.norm_fusion_v2 import (EMOTION_WEIGHTS, LATE_WEIGHTS, OFFSET, SMOOTH_K, V1_WEIGHTS, combine,
                                          fuse, smooth)
```

and add before the real-data section:

```python
def brute_smooth(proba, recordings, k, causal):
    out = np.empty_like(proba)
    for r in np.unique(recordings):
        idx = np.flatnonzero(recordings == r)
        for j, i in enumerate(idx):
            lo, hi = (max(j - k + 1, 0), j + 1) if causal else (max(j - k // 2, 0), min(j + k // 2 + 1, len(idx)))
            out[i] = proba[idx[lo:hi]].mean(axis=0)
    return out


p = rng.dirichlet(np.ones(3), size=40)
recs = np.repeat([7, 8], 20)
# k = 1 is the identity (the very same array); every k matches a direct per-recording mean.
assert smooth(p, recs, 1) is p
for k in SMOOTH_K:
    for causal in (False, True):
        assert np.allclose(smooth(p, recs, k, causal), brute_smooth(p, recs, k, causal)), (k, causal)
        assert np.allclose(smooth(p, recs, k, causal).sum(axis=1), 1)
# Only order within one recording matters: changing recording 8 leaves recording 7 unchanged, and
# interleaving the two recordings in the array gives the same per-window result.
q = p.copy(); q[20:] = rng.dirichlet(np.ones(3), size=20)
assert np.allclose(smooth(q, recs, 9)[:20], smooth(p, recs, 9)[:20])
order = np.argsort(np.tile(np.arange(20), 2), kind="stable")  # 7, 8, 7, 8, ... keeping each recording's order
assert np.allclose(smooth(p[order], recs[order], 9), smooth(p, recs, 9)[order])
# k longer than the recording: the mean of the windows that exist, no NaN.
short = smooth(p[:2], np.array([1, 1]), 15)
assert np.allclose(short, p[:2].mean(axis=0)) and not np.isnan(short).any()
# Causal smoothing never looks ahead: the first window is unchanged, changing the last window changes nothing before it.
assert np.allclose(smooth(p, recs, 5, True)[0], p[0])
r = p.copy(); r[19] = [1, 0, 0]
assert np.allclose(smooth(r, recs, 5, True)[:19], smooth(p, recs, 5, True)[:19])

# Fusion grids: 27 per-emotion triples, v1's 5 equal triples, 29 in the sorted union with equal triples ascending.
assert len(EMOTION_WEIGHTS) == 27 and len(V1_WEIGHTS) == 5 and len(LATE_WEIGHTS) == 29
equal = [w for w in LATE_WEIGHTS if len(set(w)) == 1]
assert [w[0] for w in equal] == [0.0, 0.25, 0.5, 0.75, 1.0]
eeg, eye = rng.dirichlet(np.ones(3), size=50), rng.dirichlet(np.ones(3), size=50)
for weights in LATE_WEIGHTS:
    assert np.allclose(fuse(eeg, eye, weights).sum(axis=1), 1), weights
# Equal weights give v1's fusion bit for bit; all-EEG / all-eye need only that expert.
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    assert np.array_equal(fuse(eeg, eye, (w, w, w)), w * eeg + (1 - w) * eye)
assert np.array_equal(fuse(eeg, None, (1.0, 1.0, 1.0)), eeg) and np.array_equal(fuse(None, eye, (0.0, 0.0, 0.0)), eye)
# Per-emotion weights act per class: (1, 0, 0) takes negative from EEG, neutral and positive from eye, renormalised.
mixed = np.column_stack([eeg[:, 0], eye[:, 1], eye[:, 2]])
assert np.allclose(fuse(eeg, eye, (1.0, 0.0, 0.0)), mixed / mixed.sum(axis=1, keepdims=True))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: FAIL with `ImportError: cannot import name 'EMOTION_WEIGHTS'`.

- [ ] **Step 3: Implement**

In `src/baselines/norm_fusion_v2.py`, change the imports and add the constants and the two functions after `combine`:

```python
import itertools

import numpy as np

from src.baselines.norm_fusion import W_GRID, Culture

OFFSET = 1_000_000  # added per culture index to clip, person and recording ids in two-culture training
SMOOTH_K = (1, 3, 5, 9, 15)
EMOTION_WEIGHTS = tuple(itertools.product((0.0, 0.5, 1.0), repeat=3))  # (w_negative, w_neutral, w_positive)
V1_WEIGHTS = tuple((w, w, w) for w in W_GRID)
LATE_WEIGHTS = tuple(sorted(set(EMOTION_WEIGHTS) | set(V1_WEIGHTS)))
```

```python
def smooth(proba, recordings, k, causal=False):
    """Each window's probabilities averaged over k windows of its own recording, in array order.

    Centred: the window and k // 2 on each side; causal: the window and the k - 1 before it. Edges use the
    windows that exist. Only the order of windows is used (no clip boundaries, no labels). k = 1: unchanged.
    """
    if k == 1:
        return proba
    out = np.empty_like(proba)
    for r in np.unique(recordings):
        idx = np.flatnonzero(recordings == r)
        cumulative = np.vstack([np.zeros((1, proba.shape[1])), np.cumsum(proba[idx], axis=0)])
        i = np.arange(len(idx))
        lo = np.maximum(i - (k - 1 if causal else k // 2), 0)
        hi = np.minimum(i + (1 if causal else k // 2 + 1), len(idx))
        out[idx] = (cumulative[hi] - cumulative[lo]) / (hi - lo)[:, None]
    return out


def fuse(eeg, eye, weights):
    """p_c ∝ w_c p_EEG,c + (1 - w_c) p_eye,c, renormalised; equal w_c already sum to 1 and are left as in v1."""
    w = np.asarray(weights)
    proba = 0
    if (w > 0).any():
        proba = proba + w * eeg
    if (w < 1).any():
        proba = proba + (1 - w) * eye
    if len(set(weights)) > 1:
        proba = proba / proba.sum(axis=1, keepdims=True)
    return proba
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: `norm_fusion_v2 checks passed`.

- [ ] **Step 5: Commit**

```bash
git add src/baselines/norm_fusion_v2.py src/baselines/test_norm_fusion_v2.py
git commit -m "NormFusion v2: smoothing within each recording and per-emotion fusion weights

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Candidate grid, cross-validation scores and per-variant choice

**Files:**
- Modify: `src/baselines/norm_fusion_v2.py`
- Modify: `src/baselines/test_norm_fusion_v2.py`

**Interfaces:**
- Consumes: v1 `C_GRID`, `EEG`, `EYE`, `cv_cells(subjects, y, clips, mode)`, `expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=C_GRID) -> {C: (n, 3)}`, `select_config(data, seed, use_mlp, early)`; Task 2's `smooth`, `fuse`, `LATE_WEIGHTS`, `EMOTION_WEIGHTS`, `V1_WEIGHTS`, `SMOOTH_K`.
- Produces:
  - `Candidate = namedtuple("Candidate", "fusion c_eeg c_eye weights k causal")` — late: `("late", C_EEG, C_eye, (w_neg, w_neu, w_pos), k, causal)`; early: `("early", C, None, None, k, causal)`.
  - `SMOOTHINGS`: list of `(k, causal)`, every k centred then every k > 1 causal (9 pairs).
  - `experts(train_x, train_y, eval_x, seed, use_mlp=True) -> {"eeg": {C: p}, "eye": {C: p}, "early": {C: p}}`.
  - `grid_probas(ex, recordings)`: generator of `(Candidate, smoothed (n, 3) proba)` in grid order (4,212 candidates).
  - `cv_scores(data, seed, use_mlp=True) -> {Candidate: float}` in grid order.
  - `VARIANT_KEEP: dict[str, Callable[[Candidate], bool]]` with keys `fair`, `no_smoothing`, `causal`, `v1_fusion`, `v1`.
  - `choose(scores, keep) -> Candidate` (best inside the subset, first in grid order on ties).

- [ ] **Step 1: Write the failing tests**

Extend the import block of `src/baselines/test_norm_fusion_v2.py` to

```python
from src.baselines.norm_fusion import C_GRID, Culture, cv_cells, load_culture, select_config
from src.baselines.norm_fusion_v2 import (EMOTION_WEIGHTS, LATE_WEIGHTS, OFFSET, SMOOTH_K, SMOOTHINGS, V1_WEIGHTS,
                                          VARIANT_KEEP, Candidate, choose, combine, cv_scores, experts, fuse,
                                          grid_probas, smooth)
```

and add before the real-data section:

```python
# The grid: late fusion (4 x 4 C x 29 weight triples x 9 smoothings), then early fusion (4 C x 9 smoothings).
assert SMOOTHINGS == [(1, False), (3, False), (5, False), (9, False), (15, False),
                      (3, True), (5, True), (9, True), (15, True)]
data = synthetic()
ex = experts(data.x[:540], data.y[:540], data.x[540:], seed=12, use_mlp=False)
grid = list(grid_probas(ex, data.recordings[540:]))
assert len(grid) == len(C_GRID) ** 2 * len(LATE_WEIGHTS) * 9 + len(C_GRID) * 9 == 4212
assert len({cand for cand, _ in grid}) == len(grid)
first_early = next(i for i, (cand, _) in enumerate(grid) if cand.fusion == "early")
assert all(cand.fusion == "late" for cand, _ in grid[:first_early])
lookup = dict(grid)
cand = Candidate("late", 0.1, 1.0, (1.0, 0.0, 0.5), 5, False)
assert np.allclose(lookup[cand], smooth(fuse(ex["eeg"][0.1], ex["eye"][1.0], (1.0, 0.0, 0.5)), data.recordings[540:], 5))
assert np.array_equal(lookup[Candidate("early", 0.01, None, None, 1, False)], ex["early"][0.01])

# Variant subsets.
fair_grid = [c for c, _ in grid if VARIANT_KEEP["fair"](c)]
assert len(fair_grid) == len(C_GRID) ** 2 * 27 * 5 + len(C_GRID) * 5 == 2180
assert not any(c.causal for c in fair_grid)
assert all(c.k == 1 for c, _ in grid if VARIANT_KEEP["no_smoothing"](c))
assert all(c.causal or c.k == 1 for c, _ in grid if VARIANT_KEEP["causal"](c))
assert all(c.fusion == "late" and c.weights in V1_WEIGHTS for c, _ in grid if VARIANT_KEEP["v1_fusion"](c))
v1_grid = [c for c, _ in grid if VARIANT_KEEP["v1"](c)]
assert len(v1_grid) == 80 and all(c.k == 1 and not c.causal for c in v1_grid)
# Within the v1 subset, grid order is v1's order: (C_EEG, C_eye, w ascending).
assert [(c.c_eeg, c.c_eye, c.weights[0]) for c in v1_grid] == \
       [(ce, cy, w) for ce in C_GRID for cy in C_GRID for w in (0.0, 0.25, 0.5, 0.75, 1.0)]

# Cross-validation: deterministic, every candidate scored; the v1 subset gives v1's scores and v1's choice.
scores = cv_scores(data, seed=12, use_mlp=False)
assert list(scores) == [c for c, _ in grid] and scores == cv_scores(data, seed=12, use_mlp=False)
v1_best, v1_scores = select_config(data, seed=12, use_mlp=False, early=False)
for c in v1_grid:
    assert abs(scores[c] - v1_scores[(c.c_eeg, c.c_eye, c.weights[0])]) < 1e-12, c
best_v1 = choose(scores, VARIANT_KEEP["v1"])
assert (best_v1.c_eeg, best_v1.c_eye, best_v1.weights[0]) == tuple(v1_best)
# choose: best inside the subset, first in grid order on ties.
tie = {Candidate("late", 1, 1, (0.0, 0.0, 0.0), 1, False): 0.5, Candidate("late", 1, 1, (0.5, 0.5, 0.5), 1, False): 0.5,
       Candidate("early", 1, None, None, 3, False): 0.9}
assert choose(tie, VARIANT_KEEP["v1"]).weights == (0.0, 0.0, 0.0)
assert choose(tie, VARIANT_KEEP["fair"]).fusion == "early"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: FAIL with `ImportError: cannot import name 'SMOOTHINGS'`.

- [ ] **Step 3: Implement**

In `src/baselines/norm_fusion_v2.py`, extend the imports:

```python
import itertools
from collections import defaultdict, namedtuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from src.baselines.norm_fusion import C_GRID, EEG, EYE, W_GRID, Culture, cv_cells, expert_probas
```

Add after the constants:

```python
SMOOTHINGS = [(k, False) for k in SMOOTH_K] + [(k, True) for k in SMOOTH_K if k > 1]
Candidate = namedtuple("Candidate", "fusion c_eeg c_eye weights k causal")

_v2_fusion = lambda c: c.fusion == "early" or c.weights in EMOTION_WEIGHTS
_v1_fusion = lambda c: c.fusion == "late" and c.weights in V1_WEIGHTS
VARIANT_KEEP = {
    "fair": lambda c: _v2_fusion(c) and not c.causal,
    "no_smoothing": lambda c: _v2_fusion(c) and c.k == 1,
    "causal": lambda c: _v2_fusion(c) and (c.causal or c.k == 1),
    "v1_fusion": lambda c: _v1_fusion(c) and not c.causal,
    "v1": lambda c: _v1_fusion(c) and c.k == 1,
}
```

Add after `fuse`:

```python
def experts(train_x, train_y, eval_x, seed, use_mlp=True):
    """Per-C probabilities on eval_x of the EEG expert, the eye expert and the early-fusion expert (343 features)."""
    return {"eeg": expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp),
            "eye": expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp),
            "early": expert_probas(train_x, train_y, eval_x, seed, use_mlp)}


def grid_probas(ex, recordings):
    """(candidate, smoothed probabilities) for the whole grid, in grid order: late fusion over
    C_EEG x C_eye x weights x smoothing, then early fusion over C x smoothing."""
    for c_eeg in C_GRID:
        for c_eye in C_GRID:
            for weights in LATE_WEIGHTS:
                fused = fuse(ex["eeg"][c_eeg], ex["eye"][c_eye], weights)
                for k, causal in SMOOTHINGS:
                    yield Candidate("late", c_eeg, c_eye, weights, k, causal), smooth(fused, recordings, k, causal)
    for c in C_GRID:
        for k, causal in SMOOTHINGS:
            yield Candidate("early", c, None, None, k, causal), smooth(ex["early"][c], recordings, k, causal)


def cv_scores(data, seed, use_mlp=True):
    """Mean validation balanced accuracy over v1's 9 people-and-films cells for every candidate, in grid order.

    Each cell's model predicts every window of the held-out people's recordings, the predictions are smoothed
    within each recording, and only the held-out fold's windows are scored.
    """
    scores = defaultdict(list)
    for train, val in cv_cells(data.subjects, data.y, data.clips, "person_film"):
        held = np.isin(data.subjects, np.unique(data.subjects[val]))
        scored, truth = val[held], data.y[held][val[held]]
        ex = experts(data.x[train], data.y[train], data.x[held], seed, use_mlp)
        for cand, proba in grid_probas(ex, data.recordings[held]):
            scores[cand].append(balanced_accuracy_score(truth, proba[scored].argmax(axis=1)))
    return {cand: float(np.mean(v)) for cand, v in scores.items()}


def choose(scores, keep):
    """The best-scoring candidate that `keep` allows; the first in grid order on ties."""
    allowed = {cand: s for cand, s in scores.items() if keep(cand)}
    return max(allowed, key=allowed.get)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: `norm_fusion_v2 checks passed`. If the v1-score check fails by more than 1e-12, stop and report — do not loosen it: it means the v2 CV is not v1's CV on the v1 subset.

- [ ] **Step 5: Commit**

```bash
git add src/baselines/norm_fusion_v2.py src/baselines/test_norm_fusion_v2.py
git commit -m "NormFusion v2: one candidate grid scored by v1's film-robust CV, filtered per variant

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Fitting one candidate and self-training on smoothed probabilities

**Files:**
- Modify: `src/baselines/norm_fusion_v2.py`
- Modify: `src/baselines/test_norm_fusion_v2.py`

**Interfaces:**
- Consumes: v1 `expert_probas`, `ST_ROUNDS`, `ST_THRESHOLD`, `ST_MIN`, v1 `self_train(src_x, src_y, tgt_x, cfg, seed, use_mlp, early)` and `Config` (for the equivalence test); Tasks 2–3.
- Produces: `fit_candidate(train_x, train_y, eval_x, cand, seed, use_mlp=True) -> (n, 3)` unsmoothed probabilities, fitting only the experts the candidate uses; `self_train(src_x, src_y, tgt_x, tgt_recordings, cand, seed, use_mlp=True) -> (smoothed (n, 3) proba, list[int] added per round)`.

- [ ] **Step 1: Write the failing tests**

Extend the imports of `src/baselines/test_norm_fusion_v2.py`:

```python
import inspect

from src.baselines.norm_fusion import C_GRID, Config, Culture, cv_cells, load_culture, select_config
from src.baselines.norm_fusion import self_train as v1_self_train
```

and add `fit_candidate, self_train` to the `norm_fusion_v2` import list. Add before the real-data section:

```python
# fit_candidate equals the corresponding grid entry before smoothing.
cand = Candidate("late", 0.1, 1.0, (1.0, 0.0, 0.5), 5, False)
assert np.allclose(fit_candidate(data.x[:540], data.y[:540], data.x[540:], cand, 12, False),
                   fuse(ex["eeg"][0.1], ex["eye"][1.0], (1.0, 0.0, 0.5)))
early = Candidate("early", 0.01, None, None, 3, False)
assert np.allclose(fit_candidate(data.x[:540], data.y[:540], data.x[540:], early, 12, False), ex["early"][0.01])

# Self-training never receives test labels; with k = 1 and equal weights it is v1's self-training exactly.
assert list(inspect.signature(self_train).parameters) == ["src_x", "src_y", "tgt_x", "tgt_recordings", "cand",
                                                          "seed", "use_mlp"]
src, tgt = synthetic(6), synthetic(4, offset=0.3)
for w in (0.5, 1.0):
    mine, mine_added = self_train(src.x, src.y, tgt.x, tgt.recordings, Candidate("late", 1.0, 1.0, (w, w, w), 1, False),
                                  12, True)
    theirs, theirs_added = v1_self_train(src.x, src.y, tgt.x, Config(1.0, 1.0, w), 12, True, False)
    assert mine_added == theirs_added and np.array_equal(mine, theirs), w
# With k > 1 the returned probabilities are smoothed: valid, and different from the k = 1 run.
smoothed, added = self_train(src.x, src.y, tgt.x, tgt.recordings, Candidate("late", 1.0, 1.0, (1.0, 0.5, 0.5), 9, False),
                             12, False)
plain, _ = self_train(src.x, src.y, tgt.x, tgt.recordings, Candidate("late", 1.0, 1.0, (1.0, 0.5, 0.5), 1, False),
                      12, False)
assert smoothed.shape == (len(tgt.x), 3) and np.allclose(smoothed.sum(axis=1), 1) and not np.allclose(smoothed, plain)
# No signal: self-training stops after 0 rounds.
noise_src, noise_tgt = synthetic(6, signal=0.0), synthetic(4, signal=0.0)
assert self_train(noise_src.x, noise_src.y, noise_tgt.x, noise_tgt.recordings,
                  Candidate("late", 0.001, 0.001, (0.5, 0.5, 0.5), 1, False), 12, False)[1] == []
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: FAIL with `ImportError: cannot import name 'fit_candidate'`.

- [ ] **Step 3: Implement**

Extend the v1 import in `src/baselines/norm_fusion_v2.py`:

```python
from src.baselines.norm_fusion import (C_GRID, EEG, EYE, ST_MIN, ST_ROUNDS, ST_THRESHOLD, W_GRID, Culture, cv_cells,
                                       expert_probas)
```

Add after `choose`:

```python
def fit_candidate(train_x, train_y, eval_x, cand, seed, use_mlp=True):
    """Unsmoothed probabilities on eval_x for one candidate, fitting only the experts it uses."""
    if cand.fusion == "early":
        return expert_probas(train_x, train_y, eval_x, seed, use_mlp, cs=(cand.c_eeg,))[cand.c_eeg]
    w = np.asarray(cand.weights)
    eeg = expert_probas(train_x[:, EEG], train_y, eval_x[:, EEG], seed, use_mlp, cs=(cand.c_eeg,))[cand.c_eeg] \
        if (w > 0).any() else None
    eye = expert_probas(train_x[:, EYE], train_y, eval_x[:, EYE], seed, use_mlp, cs=(cand.c_eye,))[cand.c_eye] \
        if (w < 1).any() else None
    return fuse(eeg, eye, cand.weights)


def self_train(src_x, src_y, tgt_x, tgt_recordings, cand, seed, use_mlp=True):
    """v1's class-balanced self-training, with pseudo-labels taken from the candidate's smoothed probabilities.

    Returns the final smoothed probabilities on the target and the windows added per round.
    """
    predict = lambda x, y: smooth(fit_candidate(x, y, tgt_x, cand, seed, use_mlp), tgt_recordings, cand.k, cand.causal)
    proba = predict(src_x, src_y)
    added = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        proba = predict(np.concatenate([src_x, tgt_x[chosen]]), np.concatenate([src_y, pseudo[chosen]]))
    return proba, added
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m src.baselines.test_norm_fusion_v2`
Expected: `norm_fusion_v2 checks passed`. The v1 equality is exact (`np.array_equal`); if it fails, stop and report rather than switching to `allclose`.

- [ ] **Step 5: Commit**

```bash
git add src/baselines/norm_fusion_v2.py src/baselines/test_norm_fusion_v2.py
git commit -m "NormFusion v2: fit one candidate and self-train on smoothed probabilities

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Runner — all settings, variants, paper's-rules table, clip-level scores

**Files:**
- Create: `src/experiments/run_norm_fusion_v2.py`
- Create: `src/experiments/test_run_norm_fusion_v2.py`

**Interfaces:**
- Consumes: Tasks 1–4; v1 `load_culture`, v1 runner `score(true, proba, subjects)` (window accuracy, macro-F1, confusion per person) and `run_direction` (for the equivalence test); `train_dcca_am.write_json(path, record)`.
- Produces:
  - `VARIANTS = ["fair", "no_smoothing", "causal", "v1_fusion", "no_selftrain", "v1"]`, `NEWFILMS_VARIANTS = ["fair", "no_selftrain", "v1"]`.
  - `score_v2(target: Culture, proba) -> {subject_str: {"accuracy", "macro_f1", "confusion", "clip_accuracy"}}`.
  - `run_direction(source, target, seed, variants, use_mlp=True) -> {variant: record}`; record = `{"config": list(candidate), "cv_score": float, "added": [int], "per_subject": score_v2(...)}`; the fair record also has `"candidates": {"keys": [str], "accuracy": {subject_str: [float]}}` — the 2,180 fair-grid candidates without self-training, then `"fair_selftrained"`.
  - `load_source(source, setting) -> Culture` (`"A+B"` → `combine`); `job(...)`; CLI `python -m src.experiments.run_norm_fusion_v2 --jobs 4`.
  - JSON record fields: `config`, `cv_score`, `added`, `per_subject`, (fair) `candidates`, `variant`, `source`, `target`, `seed`, `setting`.

- [ ] **Step 1: Write the failing tests**

Create `src/experiments/test_run_norm_fusion_v2.py`:

```python
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_run_norm_fusion_v2`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.run_norm_fusion_v2'`.

- [ ] **Step 3: Implement**

Create `src/experiments/run_norm_fusion_v2.py`:

```python
"""Run NormFusion v2 across cultures: fair result, ablations, v1 re-run and per-candidate test accuracy.

Settings: emotion (the 6 one-to-one directions), emotion_two_cultures (train on two cultures, test on the
third; source written "A+B"), emotion_newfilms (German <-> French without the 5 shared films in training).
Every variant is written to results/norm_fusion_v2/<target>/<setting>__from_<source>/seed_<s>/<variant>.json.

  fair          film-robust choice of (C_EEG, C_eye, fusion, k) over per-emotion late fusion and early fusion,
                with centred smoothing, then self-training on the smoothed probabilities
  no_smoothing  fair with k = 1
  causal        fair with trailing instead of centred smoothing
  v1_fusion     single weight w (v1's grid), no per-emotion weights, no early fusion; smoothing chosen
  no_selftrain  the fair choice without self-training
  v1            v1 itself (k = 1, single weight, no early fusion); reproduces results/norm_fusion

The fair record also keeps every test person's accuracy for each fair-grid candidate without self-training,
plus the fair choice with it, for the paper's-rules comparison. Test labels are used only to score.

    python -m src.experiments.run_norm_fusion_v2 --jobs 4
"""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.baselines.norm_fusion import load_culture
from src.baselines.norm_fusion_v2 import (VARIANT_KEEP, choose, combine, cv_scores, experts, fit_candidate,
                                          grid_probas, self_train, smooth)
from src.experiments.run_norm_fusion import score
from src.experiments.train_dcca_am import write_json

VARIANTS = ["fair", "no_smoothing", "causal", "v1_fusion", "no_selftrain", "v1"]
NEWFILMS_VARIANTS = ["fair", "no_selftrain", "v1"]
CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
SEEDS = [12, 34, 56]


def score_v2(target, proba):
    """v1's per-person window scores plus clip-level accuracy (majority vote of each recording's clip)."""
    scores = score(target.y, proba, target.subjects)
    pred = proba.argmax(axis=1)
    for s in np.unique(target.subjects):
        mine = target.subjects == s
        correct = []
        for r, c in sorted(set(zip(target.recordings[mine].tolist(), target.clips[mine].tolist()))):
            windows = mine & (target.recordings == r) & (target.clips == c)
            correct.append(np.bincount(pred[windows], minlength=3).argmax() == target.y[windows][0])
        scores[str(s)]["clip_accuracy"] = float(np.mean(correct))
    return scores


def candidate_table(source, target, seed, use_mlp, fair_proba):
    """Each test person's accuracy for every fair-grid candidate without self-training, then the fair choice with it."""
    keys, accuracy = [], defaultdict(list)
    people = {str(s): target.subjects == s for s in np.unique(target.subjects)}
    for cand, proba in grid_probas(experts(source.x, source.y, target.x, seed, use_mlp), target.recordings):
        if VARIANT_KEEP["fair"](cand):
            keys.append(str(tuple(cand)))
            correct = proba.argmax(axis=1) == target.y
            for s, mask in people.items():
                accuracy[s].append(round(float(correct[mask].mean()), 4))
    keys.append("fair_selftrained")
    correct = fair_proba.argmax(axis=1) == target.y
    for s, mask in people.items():
        accuracy[s].append(round(float(correct[mask].mean()), 4))
    return {"keys": keys, "accuracy": dict(accuracy)}


def run_direction(source, target, seed, variants, use_mlp=True):
    scores = cv_scores(source, seed, use_mlp)
    adapted = {}  # candidate -> (probabilities, added): variants that choose the same candidate share the run

    def adapt(cand):
        if cand not in adapted:
            adapted[cand] = self_train(source.x, source.y, target.x, target.recordings, cand, seed, use_mlp)
        return adapted[cand]

    records = {}
    for name in variants:
        cand = choose(scores, VARIANT_KEEP["fair" if name == "no_selftrain" else name])
        if name == "no_selftrain":
            proba = smooth(fit_candidate(source.x, source.y, target.x, cand, seed, use_mlp), target.recordings,
                           cand.k, cand.causal)
            added = []
        else:
            proba, added = adapt(cand)
        records[name] = {"config": list(cand), "cv_score": scores[cand], "added": added,
                         "per_subject": score_v2(target, proba)}
        if name == "fair":
            records[name]["candidates"] = candidate_table(source, target, seed, use_mlp, proba)
    return records


def load_source(source, setting):
    if "+" in source:
        return combine([load_culture(c) for c in source.split("+")])
    return load_culture(source, drop_shared=setting == "emotion_newfilms")


def job(source, target, seed, setting, output_root, skip_existing):
    torch.set_num_threads(1)
    variants = NEWFILMS_VARIANTS if setting == "emotion_newfilms" else VARIANTS
    out = output_root / target / f"{setting}__from_{source}" / f"seed_{seed}"
    if skip_existing and all((out / f"{v}.json").exists() for v in variants):
        return f"skip {setting} {source}->{target} seed {seed}"
    records = run_direction(load_source(source, setting), load_culture(target), seed, variants)
    for name, record in records.items():
        write_json(out / f"{name}.json", {**record, "variant": name, "source": source, "target": target,
                                          "seed": seed, "setting": setting})
    fair = np.mean([v["accuracy"] for v in records["fair"]["per_subject"].values()])
    v1 = np.mean([v["accuracy"] for v in records["v1"]["per_subject"].values()])
    return (f"{setting} {source}->{target} seed {seed}: fair {100 * fair:.1f}% (v1 {100 * v1:.1f}%) "
            f"config {records['fair']['config']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/norm_fusion_v2"))
    parser.add_argument("--settings", nargs="+", default=["emotion", "emotion_two_cultures", "emotion_newfilms"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    calls = []
    if "emotion" in args.settings:
        calls += [(s, t, seed, "emotion") for s, t in itertools.permutations(CULTURES, 2) for seed in args.seeds]
    if "emotion_two_cultures" in args.settings:
        calls += [("+".join(c for c in CULTURES if c != t), t, seed, "emotion_two_cultures")
                  for t in CULTURES for seed in args.seeds]
    if "emotion_newfilms" in args.settings:
        calls += [(s, t, seed, "emotion_newfilms") for s, t in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER"))
                  for seed in args.seeds]
    for line in Parallel(n_jobs=args.jobs)(delayed(job)(*call, args.output_root, args.skip_existing) for call in calls):
        print(line, flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_run_norm_fusion_v2`
Expected: `run_norm_fusion_v2 checks passed`. The v1-equivalence check is exact; if it fails, stop and report.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/run_norm_fusion_v2.py src/experiments/test_run_norm_fusion_v2.py
git commit -m "Run NormFusion v2 in all settings with ablations, a v1 re-run and per-candidate scores

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Summary against v1 and DCCA-AM

**Files:**
- Create: `src/experiments/summarize_norm_fusion_v2.py`
- Create: `src/experiments/test_summarize_norm_fusion_v2.py`

**Interfaces:**
- Consumes: v1 summary `load_nf(root)`, `nf_per_subject(seed_records, variant, metric="accuracy")`, `dcca_per_subject(dcca_runs, key, mode)`, `fmt(values, rng)`; `summarize_dcca_am` `DIRECTIONS`, `NAMES`, `load_runs`; Task 5's `VARIANTS`, record format.
- Produces: `label(culture_or_pair) -> str`; `v2_paper_rules(seed_records) -> {subject: float}`; `paired_p(a, b) -> str`; `v1_reproduction(v1_runs, v2_runs) -> {"compared": int, "configs_matched": int, "max_diff": float}`; CLI writing `results/tables/norm_fusion_v2_summary.md`.

- [ ] **Step 1: Write the failing test**

Create `src/experiments/test_summarize_norm_fusion_v2.py`:

```python
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
print("summarize_norm_fusion_v2 checks passed")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_summarize_norm_fusion_v2`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.experiments.summarize_norm_fusion_v2'`.

- [ ] **Step 3: Implement**

Create `src/experiments/summarize_norm_fusion_v2.py`:

```python
"""Summarise NormFusion v2 against v1 and DCCA-AM.

Tables: the 6 one-to-one directions (v1 fair, v2 fair, macro-F1, paired Wilcoxon v2 vs v1, v2 and DCCA-AM
under the paper's rules, DCCA-AM fair); the two-culture settings (v2 vs v1 only); new films; ablations;
the fair choices; clip-level accuracy (separate, never compared with window-level numbers); and the check
that v2's "v1" variant reproduces the saved v1 results.

    python -m src.experiments.summarize_norm_fusion_v2   # -> results/tables/norm_fusion_v2_summary.md
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from src.experiments.run_norm_fusion_v2 import VARIANTS
from src.experiments.summarize_dcca_am import DIRECTIONS, NAMES, load_runs
from src.experiments.summarize_norm_fusion import dcca_per_subject, fmt, load_nf, nf_per_subject


def label(culture):
    return " + ".join(NAMES[c] for c in culture.split("+"))


def v2_paper_rules(seed_records):
    """Best fair-grid candidate per test person (or the fair choice with self-training), seeds averaged."""
    per_seed = [{s: max(v) for s, v in r["fair"]["candidates"]["accuracy"].items()} for r in seed_records.values()]
    return {s: float(np.mean([p[s] for p in per_seed])) for s in sorted(per_seed[0], key=int)}


def paired_p(a, b):
    diff = [a[s] - b[s] for s in a]
    return f"{wilcoxon(diff).pvalue:.3f}" if any(diff) else "1"


def v1_reproduction(v1_runs, v2_runs):
    """v2's "v1" variant against the saved v1 fair runs, on every (setting, seed) both have."""
    compared, matched, max_diff = 0, 0, 0.0
    for key, seeds in v2_runs.items():
        for seed, records in seeds.items():
            saved = v1_runs.get(key, {}).get(seed, {}).get("fair")
            if saved is None or "v1" not in records:
                continue
            mine = records["v1"]
            compared += 1
            matched += [mine["config"][1], mine["config"][2], mine["config"][3][0]] == saved["config"]
            max_diff = max([max_diff] + [abs(mine["per_subject"][s]["accuracy"] - v["accuracy"])
                                         for s, v in saved["per_subject"].items()])
    return {"compared": compared, "configs_matched": matched, "max_diff": max_diff}


def mean_of(values):
    return float(np.mean(list(values.values())))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/norm_fusion_v2"))
    parser.add_argument("--v1-root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--dcca-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/norm_fusion_v2_summary.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    v2, v1, dcca = load_nf(args.root), load_nf(args.v1_root), load_runs(args.dcca_root)

    lines = ["# NormFusion v2 vs v1 vs DCCA-AM across cultures (EEG + eye)", "",
             "v2 is a post-hoc model: its ideas came from v1's results and from an analysis of emotion signatures "
             "that used the labels of all three cultures. v1 is the primary, pre-specified result.", "",
             "Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. "
             "p: paired Wilcoxon over test people, v2 fair vs v1 fair.", "",
             "| Train → Test | v1 fair | v2 fair | v2 fair macro-F1 | p (v2 vs v1) | v2 paper's rules "
             "| DCCA-AM fair | DCCA-AM paper's rules |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    direction_means = []
    for source, target in DIRECTIONS:
        key = (target, "emotion", source)
        if key not in v2:
            continue
        ours = nf_per_subject(v2[key], "fair")
        base = nf_per_subject(v1[key], "fair") if key in v1 else None
        theirs = dcca_per_subject(dcca, key, "fair")
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{fmt(nf_per_subject(v2[key], 'fair', 'macro_f1'), rng)} | "
                     f"{paired_p(ours, base) if base else '–'} | {fmt(v2_paper_rules(v2[key]), rng)} | "
                     f"{fmt(theirs, rng)} | {fmt(dcca_per_subject(dcca, key, 'paper'), rng)} |")
        direction_means.append((mean_of(ours), mean_of(base) if base else None, mean_of(theirs) if theirs else None))
    if direction_means:
        n = len(direction_means)
        line = f"Average over {n} directions: v2 fair {100 * np.mean([m[0] for m in direction_means]):.1f}%"
        if all(m[1] is not None for m in direction_means):
            line += (f", v1 fair {100 * np.mean([m[1] for m in direction_means]):.1f}%; v2 higher than v1 in "
                     f"{sum(m[0] > m[1] for m in direction_means)} of {n}")
        if all(m[2] is not None for m in direction_means):
            line += f"; DCCA-AM fair {100 * np.mean([m[2] for m in direction_means]):.1f}%"
        lines += ["", line + "."]

    lines += ["", "## Two cultures → the third (compared only with v1 in the same setting)", "",
              "| Train → Test | v1 | v2 fair | p (v2 vs v1) |", "|---|---:|---:|---:|"]
    for key in sorted(k for k in v2 if k[1] == "emotion_two_cultures"):
        ours, base = nf_per_subject(v2[key], "fair"), nf_per_subject(v2[key], "v1")
        lines.append(f"| {label(key[2])} → {label(key[0])} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{paired_p(ours, base)} |")

    lines += ["", "## New films (German ↔ French, shared films removed from training), fair accuracy %", "",
              "| Train → Test | v1 | v2 | p (v2 vs v1) | DCCA-AM |", "|---|---:|---:|---:|---:|"]
    for source, target in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")):
        key = (target, "emotion_newfilms", source)
        if key not in v2:
            continue
        ours, base = nf_per_subject(v2[key], "fair"), nf_per_subject(v2[key], "v1")
        lines.append(f"| {label(source)} → {label(target)} | {fmt(base, rng)} | {fmt(ours, rng)} | "
                     f"{paired_p(ours, base)} | {fmt(dcca_per_subject(dcca, key, 'fair'), rng)} |")

    lines += ["", "## Ablations (fair accuracy %, mean over test people)", "",
              "| Setting | Train → Test | " + " | ".join(VARIANTS) + " |", "|---|---|" + "---:|" * len(VARIANTS)]
    choices = ["", "## Fair choices (fusion, C_EEG, C_eye, weights neg/neu/pos, k) per seed", "",
               "| Setting | Train → Test | Seed | Choice | CV score |", "|---|---|---:|---|---:|"]
    clips = ["", "## Clip-level accuracy % (majority vote over each clip's windows; uses the known clip boundaries "
                 "at test time — not comparable with any window-level number)", "",
             "| Setting | Train → Test | v1 | v2 fair |", "|---|---|---:|---:|"]
    for key in sorted(v2):
        target, setting, source = key
        present = [v for v in VARIANTS if all(v in r for r in v2[key].values())]
        lines.append(f"| {setting} | {label(source)} → {label(target)} | " + " | ".join(
            f"{100 * mean_of(nf_per_subject(v2[key], v)):.1f}" if v in present else "–" for v in VARIANTS) + " |")
        for seed, records in sorted(v2[key].items()):
            fusion, c_eeg, c_eye, weights, k, _ = records["fair"]["config"]
            shown = f"{fusion}, {c_eeg}, {c_eye}, {'/'.join(f'{w:g}' for w in weights) if weights else '–'}, {k}"
            choices.append(f"| {setting} | {label(source)} → {label(target)} | {seed} | {shown} | "
                           f"{100 * records['fair']['cv_score']:.1f} |")
        clips.append(f"| {setting} | {label(source)} → {label(target)} | "
                     f"{100 * mean_of(nf_per_subject(v2[key], 'v1', 'clip_accuracy')):.1f} | "
                     f"{100 * mean_of(nf_per_subject(v2[key], 'fair', 'clip_accuracy')):.1f} |")

    check = v1_reproduction(v1, v2)
    lines += choices + clips + ["", "## v1 reproduction check", "",
                                f"v2's v1 variant vs the saved v1 results: {check['compared']} runs compared, "
                                f"{check['configs_matched']} with the same choice, largest per-person accuracy "
                                f"difference {check['max_diff']:.2e} "
                                + ("(reproduced)." if check["compared"] and check["max_diff"] < 1e-9
                                   and check["configs_matched"] == check["compared"] else "(NOT reproduced).")]
    text = "\n".join(lines) + "\n"
    args.output.write_text(text)
    print(text)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m src.experiments.test_summarize_norm_fusion_v2`
Expected: `summarize_norm_fusion_v2 checks passed`.

- [ ] **Step 5: Commit**

```bash
git add src/experiments/summarize_norm_fusion_v2.py src/experiments/test_summarize_norm_fusion_v2.py
git commit -m "Summarise NormFusion v2 against v1 and DCCA-AM, with the v1 reproduction check

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Run, verify and record

**Files:**
- Create (outputs): `results/norm_fusion_v2/**`, `results/tables/norm_fusion_v2_summary.md`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: the committed v2 results and summary.

- [ ] **Step 1: Run every test module once more**

Run:
```bash
for m in src.baselines.test_norm_fusion src.baselines.test_norm_fusion_v2 src.experiments.test_run_norm_fusion \
         src.experiments.test_run_norm_fusion_v2 src.experiments.test_summarize_norm_fusion \
         src.experiments.test_summarize_norm_fusion_v2; do .venv/bin/python -m $m || break; done
```
Expected: six "... checks passed" lines.

- [ ] **Step 2: Time one real run and check v1 reproduction on it**

Run: `time .venv/bin/python -c "from pathlib import Path; from src.experiments.run_norm_fusion_v2 import job; print(job('SEED_GER', 'SEED_FRA', 12, 'emotion', Path('results/_v2_smoke'), False))"`
Expected: one line with `fair ..% (v1 64.x%)` and a wall time. The printed v1 accuracy must equal the mean of `results/norm_fusion/SEED_FRA/emotion__from_SEED_GER/seed_12/fair.json`'s per-person accuracies; check with
`.venv/bin/python -c "import json,numpy as np; r=json.load(open('results/norm_fusion/SEED_FRA/emotion__from_SEED_GER/seed_12/fair.json')); s=json.load(open('results/_v2_smoke/SEED_FRA/emotion__from_SEED_GER/seed_12/v1.json')); print(r['config'], s['config']); print(max(abs(r['per_subject'][k]['accuracy']-s['per_subject'][k]['accuracy']) for k in r['per_subject']))"`
Expected: matching choices and `0.0`. If not, stop and report before the full run. Estimate the full run as (wall time × 33 runs × about 1.3 for the larger two-culture runs) / 4 workers; if it exceeds 6 hours, report the estimate and ask whether to run it on the cluster instead. Delete `results/_v2_smoke` afterwards (it is git-ignored by `results/_*/`).

- [ ] **Step 3: Full run**

Run: `.venv/bin/python -m src.experiments.run_norm_fusion_v2 --jobs 4 --skip-existing 2>&1 | tee results/logs/norm_fusion_v2.log`
Expected: 33 lines, one per (setting, source, target, seed); 18 emotion, 9 emotion_two_cultures, 6 emotion_newfilms.

- [ ] **Step 4: Summarise and verify**

Run: `.venv/bin/python -m src.experiments.summarize_norm_fusion_v2`
Expected: the tables filled in, and the last line `24 runs compared, 24 with the same choice, largest per-person accuracy difference 0.00e+00 (reproduced).` (18 emotion + 6 new-films runs; the new-films v1 comparison is against v1's saved new-films fair runs). If it says NOT reproduced, stop and report.

- [ ] **Step 5: Commit**

```bash
git add results/norm_fusion_v2 results/tables/norm_fusion_v2_summary.md
git commit -m "Add NormFusion v2 results and summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
