# SCVAE Cross-Culture and Unseen-Film Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the strict SCVAE-fixed options and film-held-out splits to the SCVAE runner, run the cross-culture grid, the labelled test-label-selected runs and the unseen-film runs on the cluster, and summarise Parts 1–3 of the spec.

**Architecture:** All training changes are new CLI flags on `src/experiments/train_scvae_reference.py`; their defaults reproduce today's behaviour and run names. Configuration choice reuses `select_crossculture_config.py` (leave-target-culture-out), refactored into importable functions. A manifest generator writes two cluster stages (grid + P1, then films); a summariser builds every table, the 3-seed ensemble and the success check.

**Tech Stack:** Python 3, PyTorch, NumPy, scikit-learn, SciPy; SLURM array via `cluster/train_array.sbatch`. Tests are plain-assert scripts run as `.venv/bin/python -m src.experiments.<test_module>` (no pytest in this repo).

**Spec:** `docs/superpowers/specs/2026-10-09-scvae-crossculture-improvement-design.md`

## Global Constraints

- Strict protocol: no target label is read before the single final evaluation; epoch chosen on held-out *source* subjects only; configuration for target culture T chosen only from transfers between the other two cultures.
- Test-label-selected numbers (`--protocol reported_diagnostic`) are labelled "test-label-selected" in every table and never placed in an accuracy column.
- New flags default to current behaviour; existing run names (`results/paper_scvae/...`, `results/scvae_fix/...`) must not change.
- Grid fixed flags: `--subject-norm recording --target-shuffle on --lr-schedule cosine --weight-decay 1e-4 --source-val-subjects K --epochs 100 --save-proba`, standard MMD weight 0.1. Varied: learning rate ∈ {0.0025, 0.0005}; `--bn-recalibration` ∈ {source, target}; `--input-noise` ∈ {0, 0.1}.
- K = 3 for a Chinese (`SEED_CHN_MM`) source, 2 for `SEED_GER` / `SEED_FRA`.
- Film folds: each emotion's clips, sorted, dealt round-robin into 3 folds; seen = folds {F, F+1}, unseen = the two folds ≠ F; test = the test person's fold-F windows; recordings standardised over all their windows before selection.
- Seeds 12, 34, 56. Cultures `SEED_CHN_MM`, `SEED_GER`, `SEED_FRA`; 6 cross directions.
- Success (Part 1, strict): SCVAE-fixed 3-seed ensemble average > 57.6% over 6 directions **and** beats the best simple model in ≥ 4 of 6.
- Cluster: every job requests `--gres=gpu:1`; array jobs must not start before `setup_env.sbatch` finishes; commits stay local (no remote).

## Review Focus

- Default flags: a line of `cluster/paper_sweep.txt` must produce the same run name as before (otherwise `--skip-existing` and old tables silently split).
- Source-validation subjects must be disjoint from training, never the target person, and from the source only — for cross and pooled splits and every seed.
- Film folds: unseen training must never contain a test-fold clip, and the test windows must be identical for seen and unseen.
- Changing the target labels on disk must not change saved probabilities or the selected epoch under `--protocol strict`.
- A missing or partial run directory (failed cluster task) must show as "—" in the tables, not crash, and seed counts must be visible.

---

### Task 1: Run-name builder and new CLI flags

**Files:**
- Modify: `src/experiments/train_scvae_reference.py` (`parse_args`, new `build_run_name`, new constant)
- Test: `src/experiments/test_scvae_improve.py` (create)

**Interfaces:**
- Produces: `DEFAULT_LEARNING_RATE = 0.0025`; `build_run_name(args) -> str` (reads `film_fold` with `getattr`, so namespaces without it work); `parse_args(argv=None)` with new attributes `weight_decay: float`, `input_noise: float`, `source_val_subjects: int`, `save_proba: bool`, `film_fold: int | None`, `film_condition: str | None`.
- Run-name suffix order: protocol[__alignment] → `__bn_<x>` → `__lr_<schedule>` → `__tgtbn_<x>` → `__tshuf` → `__mmd<w>` → `__lrate<lr>` → `__wd<w>` → `__noise<s>` → `__sval<k>` → `__subjnorm` → `__films_<condition><fold>` → `__pooled` | `__from_<source>`.

- [ ] **Step 1: Write the failing test** — create `src/experiments/test_scvae_improve.py`:

```python
"""Checks for the SCVAE cross-culture and unseen-film options (spec 2026-10-09).

    .venv/bin/python -m src.experiments.test_scvae_improve
"""

import tempfile
from pathlib import Path

import torch

from src.experiments.train_scvae_reference import build_run_name, parse_args


def write_culture(root, n_subjects, offset, seed):
    """A tiny one-session culture: 9 clips (emotion = (clip - 1) % 3) of 4 windows per subject.

    Feature [0, 0] holds the clip index, so tests can see which clips were loaded (without --subject-norm).
    """
    generator = torch.Generator().manual_seed(seed)
    (root / "1").mkdir(parents=True, exist_ok=True)
    for subject in range(1, n_subjects + 1):
        trials = [clip for clip in range(1, 10) for _ in range(4)]
        labels = [(clip - 1) % 3 for clip in trials]
        data = []
        for clip, label in zip(trials, labels):
            window = torch.randn(62, 5, generator=generator) + offset + 1.5 * label
            window[0, 0] = clip
            data.append(window)
        torch.save({"data": data, "labels": labels, "trials": trials}, root / "1" / f"subject{subject}.pth")


TMP = Path(tempfile.mkdtemp())
SOURCE, TARGET = TMP / "SEED_GER", TMP / "SEED_FRA"
write_culture(SOURCE, 6, 0.0, seed=1)
write_culture(TARGET, 3, 0.4, seed=2)
CROSS = ["--split", "cross", "--source-data-dir", str(SOURCE), "--data-dir", str(TARGET)]
POOLED = ["--split", "pooled", "--data-dir", str(SOURCE)]

# T1.1 defaults keep the existing run names (paper_sweep.txt cross_released and pooled lines).
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "source"])
assert args.run_name == "strict__bn_source__from_SEED_GER", args.run_name
assert (args.weight_decay, args.input_noise, args.source_val_subjects, args.save_proba) == (0.0, 0.0, 0, False)
assert (args.film_fold, args.film_condition) == (None, None)
assert parse_args(["--all-subjects", *POOLED, "--bn-recalibration", "source", "--subject-norm", "recording"]
                  ).run_name == "strict__bn_source__subjnorm__pooled"

# T1.2 scvae_fix-style line keeps its name.
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "source", "--alignment", "class_conditional_mmd",
                   "--target-shuffle", "on", "--subject-norm", "recording"])
assert args.run_name == "strict__class_conditional_mmd__bn_source__tshuf__subjnorm__from_SEED_GER", args.run_name

# T1.3 grid line: every new suffix, in order.
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "target", "--lr-schedule", "cosine",
                   "--target-shuffle", "on", "--learning-rate", "0.0005", "--weight-decay", "1e-4",
                   "--input-noise", "0.1", "--source-val-subjects", "2", "--subject-norm", "recording", "--save-proba"])
assert args.run_name == ("strict__bn_target__lr_cosine__tshuf__lrate0.0005__wd0.0001__noise0.1__sval2"
                         "__subjnorm__from_SEED_GER"), args.run_name
assert build_run_name(args) == args.run_name

# T1.4 film runs: released, fixed (with pooled source validation) and test-label-selected names.
assert parse_args(["--all-subjects", *POOLED, "--bn-recalibration", "source", "--film-fold", "1",
                   "--film-condition", "unseen"]).run_name == "strict__bn_source__films_unseen1__pooled"
assert parse_args(["--all-subjects", *POOLED, "--source-val-subjects", "2", "--subject-norm", "recording",
                   "--film-fold", "0", "--film-condition", "seen"]
                  ).run_name == "strict__sval2__subjnorm__films_seen0__pooled"
assert parse_args(["--all-subjects", *POOLED, "--protocol", "reported_diagnostic", "--film-fold", "2",
                   "--film-condition", "seen"]).run_name == "reported_diagnostic__films_seen2__pooled"
assert parse_args(["--all-subjects", *CROSS, "--protocol", "reported_diagnostic"]).run_name == \
    "reported_diagnostic__from_SEED_GER"


def rejected(argv):
    try:
        parse_args(argv)
    except SystemExit:
        return True
    return False


# T1.5 invalid combinations are rejected.
assert rejected(["--all-subjects", *CROSS, "--weight-decay", "-1"])
assert rejected(["--all-subjects", *CROSS, "--input-noise", "-0.1"])
assert rejected(["--all-subjects", *CROSS, "--source-val-subjects", "5"])  # 6 source subjects: must keep 2
assert rejected(["--all-subjects", *POOLED, "--source-val-subjects", "4"])  # 5 other subjects: must keep 2
assert rejected(["--all-subjects", "--split", "session", "--session", "1", "--data-dir", str(SOURCE),
                 "--source-val-subjects", "1"])
assert rejected(["--all-subjects", *POOLED, "--film-fold", "0"])  # condition missing
assert rejected(["--all-subjects", *CROSS, "--film-fold", "0", "--film-condition", "seen"])  # pooled only

print("SCVAE improvement tests passed.")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `ImportError: cannot import name 'build_run_name'`

- [ ] **Step 3: Implement.** In `train_scvae_reference.py`, add below `LOSS_WEIGHTS`:

```python
DEFAULT_LEARNING_RATE = 0.0025
```

Add before `parse_args`:

```python
def build_run_name(args):
    """Result directory name: the protocol plus a suffix for every non-default option."""
    name = args.protocol if args.alignment == "standard_mmd" else f"{args.protocol}__{args.alignment}"
    if args.bn_recalibration != "none":
        name = f"{name}__bn_{args.bn_recalibration}"
    if args.lr_schedule != "constant":
        name = f"{name}__lr_{args.lr_schedule}"
    if args.target_bn_update != "on":
        name = f"{name}__tgtbn_{args.target_bn_update}"
    if args.target_shuffle == "on":
        name = f"{name}__tshuf"
    if args.mmd_weight != LOSS_WEIGHTS["mmd"]:
        name = f"{name}__mmd{args.mmd_weight:g}"
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        name = f"{name}__lrate{args.learning_rate:g}"
    if args.weight_decay:
        name = f"{name}__wd{args.weight_decay:g}"
    if args.input_noise:
        name = f"{name}__noise{args.input_noise:g}"
    if args.source_val_subjects:
        name = f"{name}__sval{args.source_val_subjects}"
    if args.subject_norm != "none":
        name = f"{name}__subjnorm"
    if getattr(args, "film_fold", None) is not None:
        name = f"{name}__films_{args.film_condition}{args.film_fold}"
    if args.split == "pooled":
        name = f"{name}__pooled"
    elif args.split == "cross":
        name = f"{name}__from_{args.source_data_dir.name}"
    return name
```

In `parse_args`: change the signature to `def parse_args(argv=None):` and `args = parser.parse_args(argv)`; change the `--learning-rate` default to `DEFAULT_LEARNING_RATE`; add after `--mmd-weight`:

```python
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight decay (0 = released behaviour).")
    parser.add_argument("--input-noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to source inputs in training only (0 = off).")
    parser.add_argument("--source-val-subjects", type=int, default=0,
                        help=("--split cross or pooled: hold out this many whole source subjects and keep the "
                              "epoch with the best accuracy on them (source labels only). 0 = fixed final epoch."))
    parser.add_argument("--save-proba", action="store_true",
                        help="Also save the target's per-window probabilities, labels and clips (.npz).")
    parser.add_argument("--film-fold", type=int, choices=[0, 1, 2], default=None,
                        help="--split pooled: test only on this fold of films (each fold holds every emotion).")
    parser.add_argument("--film-condition", choices=["seen", "unseen"], default=None,
                        help=("With --film-fold F. seen: train on folds F and F+1 (includes the test films); "
                              "unseen: train on the two other folds. Same amount of training data."))
```

Add at the end of the validation block (after `args.subjects` is set and the `--split cross` checks), before the run name:

```python
    if args.weight_decay < 0 or args.input_noise < 0 or args.source_val_subjects < 0:
        parser.error("--weight-decay, --input-noise and --source-val-subjects must be non-negative")
    if args.source_val_subjects:
        if args.split == "cross":
            source_count = len(all_subjects(args.source_data_dir))
        elif args.split == "pooled":
            source_count = len(args.subjects) - 1
        else:
            parser.error("--source-val-subjects applies only to --split cross or pooled")
        if source_count - args.source_val_subjects < 2:
            parser.error("--source-val-subjects must leave at least two source subjects for training")
    if (args.film_fold is None) != (args.film_condition is None):
        parser.error("--film-fold and --film-condition go together")
    if args.film_fold is not None and args.split != "pooled":
        parser.error("--film-fold applies only to --split pooled")
```

Replace the whole run-name block (from `args.run_name = args.protocol if ...` to the `__from_` line) with:

```python
    args.run_name = build_run_name(args)
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve && .venv/bin/python -m src.experiments.test_scvae_reference`
Expected: `SCVAE improvement tests passed.` and `Reference SCVAE component tests passed.`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/train_scvae_reference.py src/experiments/test_scvae_improve.py
git commit -m "Add weight decay, noise, source-validation, save-proba and film-fold flags to the SCVAE runner"
```

---

### Task 2: Weight decay and input noise in training

**Files:**
- Modify: `src/experiments/train_scvae_reference.py` (`train_fold`, `write_fold_artifacts`; new `make_optimizer`, `add_input_noise`)
- Test: `src/experiments/test_scvae_improve.py`

**Interfaces:**
- Consumes: `args.weight_decay`, `args.input_noise` (Task 1).
- Produces: `make_optimizer(model, args) -> torch.optim.Adam`; `add_input_noise(x, std) -> Tensor` (returns `x` itself when `std == 0`, drawing no random numbers, so default runs are unchanged).

- [ ] **Step 1: Write the failing test** — add `add_input_noise, make_optimizer` to the `train_scvae_reference` import, add `from src.models.scvae_reference import ReferenceSCVAE`, and append before the final `print`:

```python
# T2.1 weight decay reaches the optimiser; default is 0.
args = parse_args(["--all-subjects", *CROSS, "--weight-decay", "1e-4"])
assert make_optimizer(ReferenceSCVAE(), args).param_groups[0]["weight_decay"] == 1e-4
assert make_optimizer(ReferenceSCVAE(), parse_args(["--all-subjects", *CROSS])).param_groups[0]["weight_decay"] == 0.0

# T2.2 noise 0 returns the same tensor and draws no random numbers; noise > 0 changes the batch.
x = torch.randn(4, 62, 5)
state = torch.get_rng_state()
assert add_input_noise(x, 0.0) is x and torch.equal(torch.get_rng_state(), state)
noisy = add_input_noise(x, 0.1)
assert noisy.shape == x.shape and not torch.equal(noisy, x) and (noisy - x).std() < 0.2
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `ImportError: cannot import name 'add_input_noise'`

- [ ] **Step 3: Implement.** Add after `copy_state_dict_to_cpu`:

```python
def make_optimizer(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def add_input_noise(x, std):
    """Gaussian noise for training inputs; std 0 returns x unchanged and draws nothing."""
    return x if std == 0 else x + std * torch.randn_like(x)
```

In `train_fold`, replace `optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)` with `optimizer = make_optimizer(model, args)`, and replace `source_recon, source_mu, source_logvar, source_logits = model(source_batch_x)` with:

```python
            # Denoising: the model sees noisy source inputs but reconstructs the clean ones.
            source_recon, source_mu, source_logvar, source_logits = model(
                add_input_noise(source_batch_x, args.input_noise)
            )
```

(`reconstruction_loss(source_recon, source_batch_x)` is unchanged, so the clean input is the target.) In `write_fold_artifacts`, add to `record` after `"learning_rate"`: `"weight_decay": args.weight_decay, "input_noise": args.input_noise,`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `SCVAE improvement tests passed.`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/train_scvae_reference.py src/experiments/test_scvae_improve.py
git commit -m "Apply weight decay and denoising input noise in SCVAE training"
```

---

### Task 3: Source-validation subjects and epoch selection

**Files:**
- Modify: `src/experiments/train_scvae_reference.py` (`load_dataset`, `load_split`, `train_fold`, `write_fold_artifacts`; new `load_subjects`, `choose_source_val_subjects`, `source_dir`, `source_candidates`, `source_val_subjects`, `load_source`, `source_validation_accuracy`)
- Test: `src/experiments/test_scvae_improve.py`

**Interfaces:**
- Consumes: `args.source_val_subjects`, `args.seed` (Task 1).
- Produces:
  - `load_subjects(data_dir, subjects, subject_norm="none") -> (x, y)`.
  - `choose_source_val_subjects(subjects, k, seed) -> list[int]` (sorted; deterministic).
  - `source_dir(args) -> Path` (`source_data_dir` for cross, else `data_dir`).
  - `source_candidates(args, test_subject) -> list[int]` (cross: every source subject; pooled: every other subject).
  - `source_val_subjects(args, test_subject) -> list[int]` (`[]` when off; `getattr` so `strict_baselines.py`'s args still work).
  - `load_source(args, subjects, test_subject) -> (x, y)` (Task 4 adds the film mask inside it).
  - `load_split` signature unchanged; source excludes validation subjects.
  - Under strict + validation, `selection` has `policy="best_source_validation_epoch_no_target_label_access"`, `selected_epoch`, `source_validation_subjects`, `source_validation_curve`.

- [ ] **Step 1: Write the failing test** — add `choose_source_val_subjects, load_split, source_val_subjects, train_fold` to the import; append before the final `print`:

```python
# T3.1 validation subjects: deterministic per seed, k of them, from the candidates only.
for seed in (12, 34, 56):
    chosen = choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, seed)
    assert chosen == choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, seed)
    assert len(chosen) == 2 and set(chosen) <= {1, 2, 3, 4, 5, 6} and chosen == sorted(chosen)
    pooled = parse_args(["--test-subject", "3", *POOLED, "--source-val-subjects", "2", "--seed", str(seed)])
    assert 3 not in source_val_subjects(pooled, 3) and len(source_val_subjects(pooled, 3)) == 2

GRID_ARGS = ["--test-subject", "1", *CROSS, "--epochs", "3", "--batch-size", "16", "--source-val-subjects", "2",
             "--subject-norm", "recording", "--target-shuffle", "on", "--lr-schedule", "cosine",
             "--weight-decay", "1e-4", "--input-noise", "0.1", "--bn-recalibration", "source",
             "--device", "cpu", "--save-proba", "--output-dir", str(TMP / "out"),
             "--checkpoint-dir", str(TMP / "ckpt")]
args = parse_args(GRID_ARGS)
source_x, source_y, target_x, target_y = load_split(1, args)
assert len(source_x) == 4 * 36 and len(target_x) == 36  # 6 source subjects, 2 held out

# T3.2 a strict cross fold selects its epoch on the source-validation subjects.
model, metrics, training, n_source, n_target, steps, selection = train_fold(1, args, torch.device("cpu"))
assert n_source == 144 and n_target == 36
assert selection["policy"] == "best_source_validation_epoch_no_target_label_access"
assert selection["source_validation_subjects"] == choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, args.seed)
assert len(selection["source_validation_curve"]) == 3 and 1 <= selection["selected_epoch"] <= 3
assert selection["target_evaluations_during_training"] == 0
assert selection["source_validation_curve"][selection["selected_epoch"] - 1] == max(selection["source_validation_curve"])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `ImportError: cannot import name 'choose_source_val_subjects'`

- [ ] **Step 3: Implement.** Add `import copy` to the imports. Replace `load_dataset` with:

```python
def load_subjects(data_dir, subjects, subject_norm="none"):
    parts = [load_subject_all_sessions(s, data_dir, subject_norm) for s in subjects]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


def load_dataset(data_dir, exclude_subject=None, subject_norm="none"):
    return load_subjects(data_dir, [s for s in all_subjects(data_dir) if s != exclude_subject], subject_norm)


def choose_source_val_subjects(subjects, k, seed):
    """k whole source subjects held out for epoch selection, fixed by the seed."""
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(sorted(subjects), size=k, replace=False))


def source_dir(args):
    return args.source_data_dir if args.split == "cross" else args.data_dir


def source_candidates(args, test_subject):
    """Every subject that may be used as source: the other culture (cross) or the other people (pooled)."""
    if args.split == "cross":
        return all_subjects(args.source_data_dir)
    return [s for s in all_subjects(args.data_dir) if s != test_subject]


def source_val_subjects(args, test_subject):
    k = getattr(args, "source_val_subjects", 0)
    if not k:
        return []
    return choose_source_val_subjects(source_candidates(args, test_subject), k, args.seed)


def load_source(args, subjects, test_subject):
    """Source windows of these subjects."""
    return load_subjects(source_dir(args), subjects, args.subject_norm)
```

Replace `load_split`'s body after the `session` branch with:

```python
    target_x, target_y = load_subject_all_sessions(test_subject, args.data_dir, norm)
    held_out = source_val_subjects(args, test_subject)
    training = [s for s in source_candidates(args, test_subject) if s not in held_out]
    source_x, source_y = load_source(args, training, test_subject)
    return source_x, source_y, target_x, target_y
```

(For `pooled` without validation this loads the same subjects in the same order as `load_dataset(..., exclude_subject=test_subject)` did.) Add after `source_batchnorm_staleness`:

```python
def source_validation_accuracy(model, train_x, val_x, val_y, device, batch_size):
    """Accuracy on held-out source subjects (source labels only).

    Scored on a copy whose BatchNorm statistics are re-estimated on the
    training source, so stale target-dominated running statistics do not
    decide the epoch. The trained model itself is untouched.
    """
    scored = recalibrate_batchnorm(copy.deepcopy(model), train_x, device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(val_x), batch_size):
            predictions.append(scored(val_x[start:start + batch_size].to(device))[3].argmax(dim=1).cpu())
    return float((torch.cat(predictions) == val_y).float().mean())
```

In `train_fold`, after `source_x, source_y, target_x, target_y = load_split(test_subject, args)`:

```python
    held_out = source_val_subjects(args, test_subject) if args.split != "session" else []
    validate = bool(held_out) and args.protocol == "strict"
    if validate:
        val_x, val_y = load_source(args, held_out, test_subject)
        print(f"Source-validation subjects: {held_out} ({len(val_x)} windows)")
    validation_curve, best_val, best_val_state, best_val_epoch = [], None, None, None
```

Inside the epoch loop, right after `if scheduler is not None: scheduler.step()`:

```python
        if validate:
            val_accuracy = source_validation_accuracy(model, source_x, val_x, val_y, device, args.batch_size)
            validation_curve.append(val_accuracy)
            if best_val is None or val_accuracy > best_val:
                best_val, best_val_state, best_val_epoch = val_accuracy, copy_state_dict_to_cpu(model), epoch
```

and append `+ (f" | src-val {validation_curve[-1]:.4f}" if validate else "")` to the epoch-log f-string argument. In the `if args.protocol == "strict":` branch, as the first statements:

```python
        if validate:
            # Restore the epoch the held-out source subjects chose; target labels are still unread.
            model.load_state_dict(best_val_state)
```

and replace the strict `selection` dict with:

```python
        selection = {
            "policy": "best_source_validation_epoch_no_target_label_access" if validate
            else "fixed_final_epoch_no_target_label_access_during_training",
            "target_evaluations_during_training": 0,
            "selected_epoch": best_val_epoch if validate else args.epochs,
            "selected_update": update_count,
        }
        if validate:
            selection["source_validation_subjects"] = held_out
            selection["source_validation_curve"] = validation_curve
```

In `write_fold_artifacts`, add to `record`: `"source_val_subjects": args.source_val_subjects,`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `SCVAE improvement tests passed.`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/train_scvae_reference.py src/experiments/test_scvae_improve.py
git commit -m "Choose the SCVAE epoch on held-out source subjects"
```

---

### Task 4: Film folds (seen vs unseen films)

**Files:**
- Modify: `src/experiments/train_scvae_reference.py` (`load_source`, `load_split`, `write_fold_artifacts`; new `N_FILM_FOLDS`, `load_subject_trials`, `load_subjects_trials`, `film_folds`, `culture_film_folds`, `film_training_folds`, `film_mask`, `target_film_mask`)
- Test: `src/experiments/test_scvae_improve.py`

**Interfaces:**
- Consumes: `args.film_fold`, `args.film_condition` (Task 1); `source_candidates`, `load_source` (Task 3).
- Produces:
  - `load_subject_trials(subject, data_dir) -> LongTensor` (clip index per window, `load_subject_all_sessions` order).
  - `load_subjects_trials(data_dir, subjects) -> LongTensor`.
  - `film_folds(labels: np.ndarray, trials: np.ndarray) -> dict[int, int]` (clip → fold).
  - `culture_film_folds(data_dir, subjects) -> dict[int, int]` (from those subjects' labels and clips).
  - `film_training_folds(test_fold, condition) -> set[int]`.
  - `film_mask(trials, folds, allowed) -> BoolTensor`.
  - `target_film_mask(args, test_subject) -> BoolTensor`.
  - `load_source` applies the training-film mask when `args.film_fold` is set; `load_split` keeps only fold-F target windows.

- [ ] **Step 1: Write the failing test** — add `film_folds, film_training_folds` to the import and `import numpy as np` at the top; append before the final `print`:

```python
# T4.1 film folds: every fold holds every emotion; seen/unseen training folds.
folds = film_folds(np.array([(c - 1) % 3 for c in range(1, 10)]), np.arange(1, 10))
assert sorted(folds) == list(range(1, 10))
for fold in range(3):
    assert sorted((c - 1) % 3 for c, f in folds.items() if f == fold) == [0, 1, 2]
assert film_training_folds(0, "seen") == {0, 1} and film_training_folds(2, "seen") == {2, 0}
assert film_training_folds(0, "unseen") == {1, 2}

# T4.2 loaded windows: unseen training never holds a test clip; same amount of training data;
# identical test windows in both conditions. Feature [0, 0] is the clip index (no --subject-norm).
test_clips = {c for c, f in folds.items() if f == 0}
loaded = {}
for condition in ("seen", "unseen"):
    film_args = parse_args(["--test-subject", "1", *POOLED, "--film-fold", "0", "--film-condition", condition])
    sx, sy, tx, ty = load_split(1, film_args)
    train_clips = set(sx[:, 0, 0].round().int().tolist())
    assert len(train_clips) == 6 and len(sx) == 5 * 6 * 4
    assert set(tx[:, 0, 0].round().int().tolist()) == test_clips and len(tx) == 3 * 4
    assert (train_clips & test_clips) == (test_clips if condition == "seen" else set())
    loaded[condition] = (tx, ty)
assert torch.equal(loaded["seen"][0], loaded["unseen"][0]) and torch.equal(loaded["seen"][1], loaded["unseen"][1])
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `ImportError: cannot import name 'film_folds'`

- [ ] **Step 3: Implement.** Add after `load_subject_all_sessions`:

```python
N_FILM_FOLDS = 3


def load_subject_trials(subject, data_dir):
    """Clip index of every window, in load_subject_all_sessions order."""
    return torch.cat([
        torch.tensor(torch.load(data_dir / str(session) / f"subject{subject}.pth", weights_only=True)["trials"])
        for session in sessions_in(data_dir)
        if subject in available_subjects(session, data_dir)
    ])


def load_subjects_trials(data_dir, subjects):
    return torch.cat([load_subject_trials(s, data_dir) for s in subjects])


def film_folds(labels, trials):
    """Clip -> fold: each emotion's clips, sorted, dealt round-robin (as clip_heldout_emotion.py)."""
    fold = {}
    for emotion in np.unique(labels):
        for i, clip in enumerate(sorted(np.unique(trials[labels == emotion]))):
            fold[int(clip)] = i % N_FILM_FOLDS
    return fold


def culture_film_folds(data_dir, subjects):
    """Film folds from the clip emotions of these (source) subjects."""
    labels = torch.cat([load_subject_all_sessions(s, data_dir)[1] for s in subjects]).numpy()
    return film_folds(labels, load_subjects_trials(data_dir, subjects).numpy())


def film_training_folds(test_fold, condition):
    """seen: the test fold and the next; unseen: the two other folds. Same amount of training data."""
    if condition == "seen":
        return {test_fold, (test_fold + 1) % N_FILM_FOLDS}
    return {f for f in range(N_FILM_FOLDS) if f != test_fold}


def film_mask(trials, folds, allowed):
    return torch.tensor([folds[int(t)] in allowed for t in trials], dtype=torch.bool)


def target_film_mask(args, test_subject):
    """The test person's windows of the test fold of films."""
    folds = culture_film_folds(args.data_dir, source_candidates(args, test_subject))
    return film_mask(load_subject_trials(test_subject, args.data_dir), folds, {args.film_fold})
```

Replace `load_source` with:

```python
def load_source(args, subjects, test_subject):
    """Source windows of these subjects; with --film-fold only the training films.

    Recordings are standardised over all their windows first (label-free), then filtered.
    """
    x, y = load_subjects(source_dir(args), subjects, args.subject_norm)
    if getattr(args, "film_fold", None) is None:
        return x, y
    folds = culture_film_folds(args.data_dir, source_candidates(args, test_subject))
    keep = film_mask(load_subjects_trials(args.data_dir, subjects), folds,
                     film_training_folds(args.film_fold, args.film_condition))
    return x[keep], y[keep]
```

In `load_split`, before `return source_x, source_y, target_x, target_y`:

```python
    if getattr(args, "film_fold", None) is not None:
        keep = target_film_mask(args, test_subject)
        target_x, target_y = target_x[keep], target_y[keep]
```

In `write_fold_artifacts`, add to `record`: `"film_fold": args.film_fold, "film_condition": args.film_condition,`.

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `SCVAE improvement tests passed.`

- [ ] **Step 5: Commit**

```bash
git add src/experiments/train_scvae_reference.py src/experiments/test_scvae_improve.py
git commit -m "Add seen/unseen film folds to the SCVAE pooled split"
```

---

### Task 5: Saved probabilities and the target-label invariance check

**Files:**
- Modify: `src/experiments/train_scvae_reference.py` (`evaluate`, `write_fold_artifacts`; new `target_trials`)
- Test: `src/experiments/test_scvae_improve.py`

**Interfaces:**
- Consumes: `load_subject_trials`, `target_film_mask` (Task 4).
- Produces: `evaluate(...)` metrics gain `"probabilities": np.ndarray [n, 3]` and `"labels": np.ndarray [n]`; `target_trials(args, test_subject) -> np.ndarray`; `write_fold_artifacts` strips both arrays from the JSON and, with `--save-proba`, writes `<run_dir>/subject_XX_proba.npz` with `proba` (float32 [n, 3]), `labels` (int64 [n]), `trials` (int64 [n]).

- [ ] **Step 1: Write the failing test** — add `write_fold_artifacts` to the import and `import json` at the top; append before the final `print`:

```python
# T5.1 probabilities: rows sum to 1, argmax is the reported prediction, JSON stays small, npz written.
probabilities = metrics["probabilities"]
assert probabilities.shape == (36, 3) and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
assert np.bincount(probabilities.argmax(axis=1), minlength=3).tolist() == metrics["prediction_counts"]
record, run_dir = write_fold_artifacts(model, metrics, training, 1, args, torch.device("cpu"),
                                       n_source, n_target, steps, selection)
saved = np.load(run_dir / "subject_01_proba.npz")
assert np.allclose(saved["proba"], probabilities) and saved["labels"].tolist() == target_y.tolist()
assert saved["trials"].tolist() == [c for c in range(1, 10) for _ in range(4)]
stored = json.loads((run_dir / "subject_01.json").read_text())
assert "probabilities" not in stored and "labels" not in stored and stored["source_val_subjects"] == 2

# T5.2 strict: target labels on disk do not influence training (same probabilities, same epoch).
path = TARGET / "1" / "subject1.pth"
original = torch.load(path, weights_only=True)
torch.save({**original, "labels": original["labels"][::-1]}, path)
_, relabelled, _, _, _, _, relabelled_selection = train_fold(1, parse_args(GRID_ARGS), torch.device("cpu"))
torch.save(original, path)
assert np.allclose(relabelled["probabilities"], probabilities)
assert relabelled_selection["selected_epoch"] == selection["selected_epoch"]

# T5.3 a strict film fold with pooled source validation runs end to end; saved clips are the test films.
film_args = parse_args(["--test-subject", "1", *POOLED, "--film-fold", "0", "--film-condition", "unseen",
                        "--epochs", "2", "--batch-size", "8", "--source-val-subjects", "2", "--device", "cpu",
                        "--save-proba", "--output-dir", str(TMP / "out"), "--checkpoint-dir", str(TMP / "ckpt")])
result = train_fold(1, film_args, torch.device("cpu"))
assert result[6]["policy"] == "best_source_validation_epoch_no_target_label_access" and result[4] == 12
assert 1 not in result[6]["source_validation_subjects"]
_, film_dir = write_fold_artifacts(*result[:3], 1, film_args, torch.device("cpu"), *result[3:])
assert set(np.load(film_dir / "subject_01_proba.npz")["trials"].tolist()) == test_clips
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve`
Expected: `KeyError: 'probabilities'`

- [ ] **Step 3: Implement.** Replace `evaluate` with:

```python
def evaluate(model, target_x, target_y, device, batch_size):
    """Evaluate a trained model against the target labels."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(target_x), batch_size):
            logits = model(target_x[start:start + batch_size].to(device))[3]
            probabilities.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(probabilities).numpy()
    predictions = probabilities.argmax(axis=1).tolist()

    actual = target_y.tolist()
    accuracy = accuracy_score(actual, predictions)
    macro_f1 = f1_score(actual, predictions, average="macro")
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=list(range(NUM_CLASSES))).tolist(),
        "prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES).tolist(),
        "true_counts": np.bincount(actual, minlength=NUM_CLASSES).tolist(),
        "probabilities": probabilities,
        "labels": np.asarray(actual, dtype=np.int64),
    }
```

Add after `target_film_mask`:

```python
def target_trials(args, test_subject):
    """Clip index of each evaluated target window, in evaluation order."""
    if args.split == "session":
        data = torch.load(args.data_dir / str(args.session) / f"subject{test_subject}.pth", weights_only=True)
        return np.asarray(data["trials"], dtype=np.int64)
    trials = load_subject_trials(test_subject, args.data_dir)
    if getattr(args, "film_fold", None) is not None:
        trials = trials[target_film_mask(args, test_subject)]
    return trials.numpy().astype(np.int64)
```

In `write_fold_artifacts`, as the first statements:

```python
    arrays = {key: metrics[key] for key in ("probabilities", "labels")}
    metrics = {key: value for key, value in metrics.items() if key not in arrays}
```

and after the CSV is written:

```python
    if args.save_proba:
        np.savez(run_dir / f"{stem}_proba.npz", proba=arrays["probabilities"].astype(np.float32),
                 labels=arrays["labels"], trials=target_trials(args, test_subject))
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_scvae_improve && .venv/bin/python -m src.experiments.test_scvae_reference`
Expected: both pass.

- [ ] **Step 5: Real-data smoke runs** (CPU, seconds each):

Run: `.venv/bin/python -m src.experiments.train_scvae_reference --test-subject 1 --split cross --source-data-dir data/processed/SEED_GER --data-dir data/processed/SEED_FRA --epochs 2 --max-train-batches 2 --source-val-subjects 2 --subject-norm recording --target-shuffle on --lr-schedule cosine --weight-decay 1e-4 --input-noise 0.1 --bn-recalibration target --save-proba --device cpu --output-dir /tmp/scvae_smoke --checkpoint-dir /tmp/scvae_smoke_ckpt`
Expected: `Source-validation subjects: [...]`, two epoch lines with `src-val`, `Accuracy: ...`, and a `subject_01_proba.npz` under `/tmp/scvae_smoke/strict__bn_target__lr_cosine__tshuf__wd0.0001__noise0.1__sval2__subjnorm__from_SEED_GER/session_all/seed_12/`.

Run: `.venv/bin/python -m src.experiments.train_scvae_reference --test-subject 1 --split pooled --data-dir data/processed/SEED_CHN_MM --film-fold 0 --film-condition unseen --epochs 2 --max-train-batches 2 --bn-recalibration source --save-proba --device cpu --output-dir /tmp/scvae_smoke --checkpoint-dir /tmp/scvae_smoke_ckpt`
Expected: target samples about one third of the subject's windows; results under `strict__bn_source__films_unseen0__pooled`.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/train_scvae_reference.py src/experiments/test_scvae_improve.py
git commit -m "Save SCVAE target probabilities and clips; check target labels cannot influence strict runs"
```

---

### Task 6: Importable leave-target-culture-out selection

**Files:**
- Modify: `src/experiments/select_crossculture_config.py`
- Test: `src/experiments/test_select_crossculture_config.py` (create)

**Interfaces:**
- Produces:
  - `load(root, pattern="ep*/*/*") -> dict[(config, source, target), float]`; config = the directory parts above `<target>` joined with spaces, plus the run name before `__from_`, with `__sval<k>` normalised to `__sval`.
  - `choose(scores, configs, target) -> tuple[str, float] | None`.

- [ ] **Step 1: Capture current output for the regression check**

Run: `.venv/bin/python -m src.experiments.select_crossculture_config > /tmp/select_before.txt`

- [ ] **Step 2: Write the failing test** — create `src/experiments/test_select_crossculture_config.py`:

```python
"""Checks for leave-target-culture-out configuration choice.

    .venv/bin/python -m src.experiments.test_select_crossculture_config
"""

import json
import tempfile
from pathlib import Path

from src.experiments.select_crossculture_config import CULTURES, choose, load

C, G, F = CULTURES
# "a" is best on the German<->French validation transfers; "b" is far better on the transfers
# into Chinese, which must not influence the choice for Chinese.
scores = {}
for config, validation, test in [("a", 60, 40), ("b", 50, 99)]:
    scores.update({(config, G, F): validation, (config, F, G): validation,
                   (config, G, C): test, (config, F, C): test})
assert choose(scores, ["a", "b"], C) == ("a", 60)
assert choose(scores, ["a", "b"], G) is None  # transfers into German were never run
del scores[("a", F, C)]
assert choose(scores, ["a", "b"], C) == ("b", 50)  # "a" incomplete on a test transfer

# load: grid layout; sval2 / sval3 merge into one configuration.
root = Path(tempfile.mkdtemp())
for target, source, k, accuracy in [(G, C, 3, 0.5), (G, F, 2, 0.7)]:
    run = root / target / f"strict__bn_source__sval{k}__subjnorm__from_{source}" / "session_all" / "seed_12"
    run.mkdir(parents=True)
    for subject in (1, 2):
        (run / f"subject_{subject:02d}.json").write_text(json.dumps({"test_subject": subject, "accuracy": accuracy}))
loaded = load(root, pattern="*/*")
assert loaded == {("strict__bn_source__sval__subjnorm", C, G): 50.0,
                  ("strict__bn_source__sval__subjnorm", F, G): 70.0}, loaded
print("Selection tests passed.")
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_select_crossculture_config`
Expected: `ImportError: cannot import name 'choose'`

- [ ] **Step 4: Implement.** Replace `load` with:

```python
def load(root, pattern="ep*/*/*"):
    """(config, source, target) -> mean accuracy over test subjects (seeds averaged)."""
    scores = {}
    for run_dir in root.glob(pattern):
        parts = run_dir.relative_to(root).parts
        target = parts[-2]
        match = re.fullmatch(r"(.+?)__from_(\w+)", run_dir.name)
        if not match:
            continue
        # The number of held-out source subjects depends on the source culture, not the configuration.
        config = " ".join(parts[:-2] + (re.sub(r"__sval\d+", "__sval", match.group(1)),))
        per_subject = defaultdict(list)
        for path in run_dir.glob("session_all/seed_*/subject_*.json"):
            record = json.loads(path.read_text())
            per_subject[record["test_subject"]].append(record["accuracy"])
        if per_subject:
            scores[(config, match.group(2), target)] = 100 * np.mean([np.mean(v) for v in per_subject.values()])
    return scores


def choose(scores, configs, target):
    """Best configuration for ``target``, scored only on the transfers between the other two cultures."""
    others = [c for c in CULTURES if c != target]
    validation = [(others[0], others[1]), (others[1], others[0])]
    tests = [(o, target) for o in others]
    eligible = [c for c in configs if all((c, s, t) in scores for s, t in validation + tests)]
    if not eligible:
        return None
    best = max(eligible, key=lambda c: np.mean([scores[(c, s, t)] for s, t in validation]))
    return best, float(np.mean([scores[(best, s, t)] for s, t in validation]))
```

In `main`: add `parser.add_argument("--pattern", default="ep*/*/*", help="Run directories below --root; '*/*' for <target>/<run>.")`, call `load(args.root, args.pattern)`, and replace the body of `for target in CULTURES:` with:

```python
        others = [c for c in CULTURES if c != target]
        chosen = choose(scores, configs, target)
        if chosen is None:
            print(f"| {NAMES[target]} | — | incomplete | — | — |")
            continue
        best, val = chosen
        test = ", ".join(f"{NAMES[o]}→{NAMES[target]} {scores[(best, o, target)]:.1f}" for o in others)
        chosen_from = f"{NAMES[others[0]]}→{NAMES[others[1]]} & {NAMES[others[1]]}→{NAMES[others[0]]}"
        print(f"| {NAMES[target]} | {chosen_from} | {best} | {val:.1f} | {test} |")
```

- [ ] **Step 5: Run tests and the regression check**

Run: `.venv/bin/python -m src.experiments.test_select_crossculture_config && .venv/bin/python -m src.experiments.select_crossculture_config > /tmp/select_after.txt && diff /tmp/select_before.txt /tmp/select_after.txt`
Expected: `Selection tests passed.` and no diff output.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/select_crossculture_config.py src/experiments/test_select_crossculture_config.py
git commit -m "Make leave-target-culture-out configuration choice importable"
```

---

### Task 7: Cluster manifests (grid + P1, then films) and `.npz` pull

**Files:**
- Create: `src/experiments/make_scvae_improve_manifest.py`
- Test: `src/experiments/test_make_scvae_improve_manifest.py` (create)
- Modify: `cluster/sync.sh` (pull branch fetches `*.npz`)
- Create (generated): `cluster/scvae_improve_grid.txt`; later `cluster/scvae_improve_films.txt`

**Interfaces:**
- Consumes: `build_run_name` (Task 1); `load`, `choose`, `CULTURES` (Task 6).
- Produces: `GRID: list[(lr, bn, noise)]`; `config_name(lr, bn, noise) -> str` (the selector's key); `grid_lines() -> list[str]` (144 grid + 18 P1 cross = 162); `film_lines(chosen: dict[culture, (lr, bn, noise)]) -> list[str]` (162); `chosen_from_grid(root) -> dict`. Result roots `results/scvae_improve/{grid,paper_protocol,films}/<target>`.

- [ ] **Step 1: Write the failing test** — create `src/experiments/test_make_scvae_improve_manifest.py`:

```python
"""Checks for the SCVAE improvement cluster manifests.

    .venv/bin/python -m src.experiments.test_make_scvae_improve_manifest
"""

import shlex
from collections import Counter

from src.experiments.make_scvae_improve_manifest import GRID, config_name, film_lines, grid_lines
from src.experiments.train_scvae_reference import parse_args

# sbatch prepends these; manifest options must override them.
SBATCH = ["--all-subjects", "--protocol", "strict", "--epochs", "300"]

lines = grid_lines()
assert len(lines) == 162 and len(set(lines)) == 162
kinds, names = Counter(), set()
for line in lines:
    args = parse_args([*SBATCH, *shlex.split(line)])
    assert args.split == "cross" and args.source_data_dir != args.data_dir and args.save_proba
    if args.protocol == "strict":
        kinds["grid"] += 1
        assert args.epochs == 100 and "/grid/" in args.output_dir
        assert args.source_val_subjects == (3 if args.source_data_dir.name == "SEED_CHN_MM" else 2)
        names.add(args.run_name.split("__from_")[0].replace(f"__sval{args.source_val_subjects}", "__sval"))
    else:
        kinds["p1"] += 1
        assert args.run_name.startswith("reported_diagnostic__from_") and args.epochs == 300
        assert "/paper_protocol/" in args.output_dir
assert kinds == {"grid": 144, "p1": 18}
assert names == {config_name(*g) for g in GRID} and len(names) == 8

chosen = {culture: GRID[-1] for culture in ("SEED_CHN_MM", "SEED_GER", "SEED_FRA")}
films = film_lines(chosen)
assert len(films) == 162 and len(set(films)) == 162
kinds = Counter()
for line in films:
    args = parse_args([*SBATCH, *shlex.split(line)])
    assert args.split == "pooled" and args.film_fold in (0, 1, 2) and "/films/" in args.output_dir
    if args.protocol == "reported_diagnostic":
        kinds["p1"] += 1
    elif args.weight_decay:
        kinds["fixed"] += 1
        assert args.source_val_subjects == (3 if args.data_dir.name == "SEED_CHN_MM" else 2)
        assert args.run_name.startswith(config_name(*GRID[-1]).replace("__sval", f"__sval{args.source_val_subjects}"))
    else:
        kinds["released"] += 1
        assert args.bn_recalibration == "source" and args.epochs == 300
assert kinds == {"p1": 54, "fixed": 54, "released": 54}
print("Manifest tests passed.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_make_scvae_improve_manifest`
Expected: `ModuleNotFoundError: No module named 'src.experiments.make_scvae_improve_manifest'`

- [ ] **Step 3: Implement** — create `src/experiments/make_scvae_improve_manifest.py`:

```python
"""Cluster manifests for the SCVAE cross-culture and unseen-film study (spec 2026-10-09).

    python -m src.experiments.make_scvae_improve_manifest grid    # -> cluster/scvae_improve_grid.txt (162)
    python -m src.experiments.make_scvae_improve_manifest films   # after the grid -> cluster/scvae_improve_films.txt

Lines are appended by cluster/train_array.sbatch after its own
--protocol strict --epochs 300, so the options here override those.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

from src.experiments.select_crossculture_config import CULTURES, choose, load
from src.experiments.train_scvae_reference import LOSS_WEIGHTS, build_run_name

SEEDS = (12, 34, 56)
GRID = [(lr, bn, noise) for lr in (0.0025, 0.0005) for bn in ("source", "target") for noise in (0.0, 0.1)]
FIXED = ("--subject-norm recording --target-shuffle on --lr-schedule cosine --weight-decay 0.0001 "
         "--epochs 100 --log-every 25")
RELEASED = "--epochs 300 --log-every 100"
GRID_ROOT = Path("results/scvae_improve/grid")


def val_k(source):
    """Held-out source subjects: 3 of the 11-12 Chinese, 2 of the 7-8 German / French."""
    return 3 if source == "SEED_CHN_MM" else 2


def directions():
    return [(s, t) for t in CULTURES for s in CULTURES if s != t]


def recipe(lr, noise, source):
    return f"{FIXED} --source-val-subjects {val_k(source)} --learning-rate {lr:g} --input-noise {noise:g}"


def cross_paths(kind, source, target, seed):
    return (f"--split cross --source-data-dir data/processed/{source} --data-dir data/processed/{target} "
            f"--seed {seed} --save-proba --output-dir results/scvae_improve/{kind}/{target} "
            f"--checkpoint-dir checkpoints/scvae_improve/{kind}/{target}")


def film_paths(culture, fold, condition, seed):
    return (f"--split pooled --data-dir data/processed/{culture} --film-fold {fold} --film-condition {condition} "
            f"--seed {seed} --save-proba --output-dir results/scvae_improve/films/{culture} "
            f"--checkpoint-dir checkpoints/scvae_improve/films/{culture}")


def config_name(lr, bn, noise):
    """The selector's key for a grid configuration (run name without source, sval count merged)."""
    args = SimpleNamespace(
        protocol="strict", alignment="standard_mmd", bn_recalibration=bn, lr_schedule="cosine",
        target_bn_update="on", target_shuffle="on", mmd_weight=LOSS_WEIGHTS["mmd"], learning_rate=lr,
        weight_decay=0.0001, input_noise=noise, source_val_subjects=1, subject_norm="recording", split="session",
    )
    return build_run_name(args).replace("__sval1", "__sval")


def grid_lines():
    """Part 1: the 8-configuration strict grid, then P1 (released, test-label-selected) across cultures."""
    grid = [f"{cross_paths('grid', s, t, seed)} {recipe(lr, noise, s)} --bn-recalibration {bn}"
            for lr, bn, noise in GRID for s, t in directions() for seed in SEEDS]
    p1 = [f"--protocol reported_diagnostic {cross_paths('paper_protocol', s, t, seed)} {RELEASED}"
          for s, t in directions() for seed in SEEDS]
    return grid + p1


def film_lines(chosen):
    """Part 2: released + BN fix, SCVAE-fixed (chosen per culture), P1; seen and unseen films."""
    lines = []
    for culture in CULTURES:
        lr, bn, noise = chosen[culture]
        for fold in range(3):
            for condition in ("seen", "unseen"):
                for seed in SEEDS:
                    paths = film_paths(culture, fold, condition, seed)
                    lines += [f"{paths} --bn-recalibration source {RELEASED}",
                              f"{paths} {recipe(lr, noise, culture)} --bn-recalibration {bn}",
                              f"--protocol reported_diagnostic {paths} {RELEASED}"]
    return lines


def chosen_from_grid(root=GRID_ROOT):
    scores = load(root, pattern="*/*")
    by_name = {config_name(*g): g for g in GRID}
    chosen = {}
    for target in CULTURES:
        result = choose(scores, sorted(by_name), target)
        if result is None:
            raise SystemExit(f"grid incomplete for target {target}; pull all results first")
        chosen[target] = by_name[result[0]]
        print(f"{target}: {result[0]} (validation {result[1]:.1f})")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["grid", "films"])
    args = parser.parse_args()
    if args.stage == "grid":
        lines, path = grid_lines(), Path("cluster/scvae_improve_grid.txt")
    else:
        lines, path = film_lines(chosen_from_grid()), Path("cluster/scvae_improve_films.txt")
    header = (f"# SCVAE cross-culture / unseen-film study, {args.stage} stage ({len(lines)} tasks); spec 2026-10-09.\n"
              f"# Submit: sbatch --export=ALL,MANIFEST={path} --array=1-{len(lines)} cluster/train_array.sbatch\n")
    path.write_text(header + "\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests and generate the grid manifest**

Run: `.venv/bin/python -m src.experiments.test_make_scvae_improve_manifest && .venv/bin/python -m src.experiments.make_scvae_improve_manifest grid && grep -cv '^\s*\(#\|$\)' cluster/scvae_improve_grid.txt`
Expected: `Manifest tests passed.`, `Wrote 162 lines ...`, `162`.

- [ ] **Step 5: Let `sync.sh pull` fetch saved probabilities.** In `cluster/sync.sh`, pull branch, change `--include='*.csv'` to `--include='*.csv' --include='*.npz'`.

Run: `bash -n cluster/sync.sh`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/make_scvae_improve_manifest.py src/experiments/test_make_scvae_improve_manifest.py cluster/scvae_improve_grid.txt cluster/sync.sh
git commit -m "Add the SCVAE study cluster manifests and pull saved probabilities"
```

---

### Task 8: Summary tables (Parts 1–3), ensemble and success check

**Files:**
- Create: `src/experiments/summarize_scvae_improve.py`
- Test: `src/experiments/test_summarize_scvae_improve.py` (create)
- Output: `results/tables/scvae_improve_summary.md`

**Interfaces:**
- Consumes: `load`, `choose`, `CULTURES`, `NAMES` (Task 6); `config_name`, `GRID` (Task 7); `FILMS` from `src/experiments/film_identification.py` (film → (emotion, German trial, French trial)); npz from Task 5.
- Produces: `per_subject(run_dir) -> dict[int, (acc %, f1 %)]` (`{}` if missing); `ensemble(run_dir) -> dict[int, (acc %, f1 %)]`; `find_run(root, target, name) -> Path | None` (`__sval<k>` normalised); `baselines(csv_path) -> dict[(source, target), float]`; `film_people(root, culture, base_name, condition) -> dict[int, float]`; `shared_split(run_dir, culture) -> dict[int, (shared %, other %)]`; `main()`.

- [ ] **Step 1: Write the failing test** — create `src/experiments/test_summarize_scvae_improve.py`:

```python
"""Checks for the SCVAE study summary.

    .venv/bin/python -m src.experiments.test_summarize_scvae_improve
"""

import json
import tempfile
from pathlib import Path

import numpy as np

from src.experiments.summarize_scvae_improve import ensemble, film_people, find_run, per_subject, shared_split

root = Path(tempfile.mkdtemp())
run = root / "SEED_GER" / "strict__bn_source__sval3__subjnorm__from_SEED_CHN_MM"
labels = np.array([0, 2, 0, 2])
trials = np.array([3, 9, 6, 13])  # German trials 3 and 9 are shared films (negative, positive); 6 and 13 are not
# Mean of the two seeds predicts [0, 2, 2, 0]: right on windows 0-1 (shared), wrong on 2-3 (other films).
seed_probas = {12: [[.9, .05, .05], [.1, .1, .8], [.2, .1, .7], [.3, .3, .4]],
               34: [[.5, .1, .4], [.3, .1, .6], [.1, .2, .7], [.6, .3, .1]]}
for seed, proba in seed_probas.items():
    directory = run / "session_all" / f"seed_{seed}"
    directory.mkdir(parents=True)
    np.savez(directory / "subject_01_proba.npz", proba=np.array(proba, dtype=np.float32), labels=labels, trials=trials)
    (directory / "subject_01.json").write_text(json.dumps({"test_subject": 1, "accuracy": 0.5, "macro_f1": 0.4}))

# Ensemble predictions: [0, 2, 2, 0] -> right on the two shared windows, wrong on the two others.
assert ensemble(run)[1][0] == 50.0
assert shared_split(run, "SEED_GER") == {1: (100.0, 0.0)}
assert per_subject(run) == {1: (50.0, 40.0)}
assert find_run(root, "SEED_GER", "strict__bn_source__sval__subjnorm__from_SEED_CHN_MM") == run
assert find_run(root, "SEED_GER", "strict__bn_source__sval__subjnorm__from_SEED_FRA") is None
assert per_subject(root / "missing") == {} and ensemble(root / "missing") == {}

# film_people: average over the three folds; a person missing a fold is dropped.
for fold, accuracies in enumerate([(0.6, 0.3), (0.9, 0.3), (0.3, None)]):
    directory = root / "films" / "SEED_FRA" / f"strict__bn_source__films_seen{fold}__pooled" / "session_all" / "seed_12"
    directory.mkdir(parents=True)
    for subject, accuracy in zip((1, 2), accuracies):
        if accuracy is not None:
            (directory / f"subject_{subject:02d}.json").write_text(
                json.dumps({"test_subject": subject, "accuracy": accuracy, "macro_f1": accuracy}))
people = film_people(root / "films", "SEED_FRA", "strict__bn_source", "seen")
assert list(people) == [1] and abs(people[1] - 60.0) < 1e-9
print("Summary tests passed.")
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m src.experiments.test_summarize_scvae_improve`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `src/experiments/summarize_scvae_improve.py`:

```python
"""Tables for the SCVAE cross-culture and unseen-film study (spec 2026-10-09).

Strict columns are accuracy on unseen people. Columns marked "test-label-selected"
choose the checkpoint by test accuracy, as the released code did; they are a
labelled comparison, not accuracy.

    python -m src.experiments.summarize_scvae_improve
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score

from src.experiments.film_identification import FILMS
from src.experiments.make_scvae_improve_manifest import GRID, config_name
from src.experiments.select_crossculture_config import CULTURES, NAMES, choose, load

GRID_ROOT = Path("results/scvae_improve/grid")
PAPER_ROOT = Path("results/scvae_improve/paper_protocol")
FILM_ROOT = Path("results/scvae_improve/films")
RELEASED_ROOT = Path("results/paper_scvae")
BASELINE_CSV = Path("results/tables/crossculture_summary.csv")
SIMPLE_FILM_MD = Path("results/tables/clip_heldout_emotion.md")
OUTPUT = Path("results/tables/scvae_improve_summary.md")
SIMPLE = ("mlp__subjnorm", "svm__subjnorm", "linsvm__subjnorm")
CULTURE_OF = {"Chinese": "SEED_CHN_MM", "German": "SEED_GER", "French": "SEED_FRA"}
PAIRS = [(s, t) for s in CULTURES for t in CULTURES if s != t]
SHARED_COLUMN = {"SEED_GER": 0, "SEED_FRA": 1}  # position of each culture's trial in FILMS values after emotion
SUCCESS_WINS = 4


def per_subject(run_dir):
    """Test subject -> (accuracy %, macro-F1 %), averaged over seeds; {} if nothing was run."""
    accuracy, f1 = defaultdict(list), defaultdict(list)
    for path in Path(run_dir).glob("session_all/seed_*/subject_*.json"):
        record = json.loads(path.read_text())
        accuracy[record["test_subject"]].append(record["accuracy"])
        f1[record["test_subject"]].append(record["macro_f1"])
    return {s: (100 * float(np.mean(accuracy[s])), 100 * float(np.mean(f1[s]))) for s in accuracy}


def seed_files(run_dir):
    """Test subject -> list of loaded per-seed .npz files."""
    files = defaultdict(list)
    for path in sorted(Path(run_dir).glob("session_all/seed_*/subject_*_proba.npz")):
        files[int(re.search(r"subject_(\d+)_proba", path.name).group(1))].append(np.load(path))
    return files


def ensemble_predictions(saved):
    labels = saved[0]["labels"]
    assert all(np.array_equal(s["labels"], labels) for s in saved), "seeds disagree on the target windows"
    return labels, np.mean([s["proba"] for s in saved], axis=0).argmax(axis=1)


def ensemble(run_dir):
    """Test subject -> (accuracy %, macro-F1 %) of the mean of the seeds' saved probabilities."""
    out = {}
    for subject, saved in seed_files(run_dir).items():
        labels, predictions = ensemble_predictions(saved)
        out[subject] = (100 * float(np.mean(predictions == labels)),
                        100 * float(f1_score(labels, predictions, average="macro")))
    return out


def shared_split(run_dir, culture):
    """Test subject -> (shared films, other films) balanced accuracy % over negative and positive windows."""
    shared = {trials[SHARED_COLUMN[culture]]: emotion for emotion, *trials in FILMS.values()}
    out = {}
    for subject, saved in seed_files(run_dir).items():
        labels, predictions = ensemble_predictions(saved)
        trials = saved[0]["trials"]
        for trial, emotion in shared.items():
            assert set(labels[trials == trial].tolist()) <= {emotion}, f"trial {trial} is not emotion {emotion}"
        in_shared = np.isin(trials, list(shared))
        scores = []
        for part in (in_shared, ~in_shared):
            recalls = [np.mean(predictions[part & (labels == c)] == c) for c in (0, 2) if (part & (labels == c)).any()]
            scores.append(100 * float(np.mean(recalls)))
        out[subject] = tuple(scores)
    return out


def seed_count(run_dir):
    return len(list(Path(run_dir).glob("session_all/seed_*"))) if run_dir else 0


def find_run(root, target, name):
    """The run directory whose name (sval count merged) equals ``name``, or None."""
    directory = Path(root) / target
    if not directory.is_dir():
        return None
    for run in directory.iterdir():
        if re.sub(r"__sval\d+", "__sval", run.name) == name:
            return run
    return None


def baselines(csv_path=BASELINE_CSV):
    """(source, target) -> best normalised simple model accuracy (EEG, 4 s), chosen after the fact."""
    best = {}
    with open(csv_path) as handle:
        for row in csv.DictReader(handle):
            if (row["setting"], row["modality"], row["window"]) == ("cross", "eeg", "4s") and row["method"] in SIMPLE:
                key = (CULTURE_OF[row["train"]], CULTURE_OF[row["test"]])
                best[key] = max(best.get(key, 0.0), float(row["accuracy_mean"]))
    return best


def film_people(root, culture, base_name, condition):
    """Person -> accuracy % averaged over the 3 film folds (seeds averaged); people missing a fold are dropped."""
    folds = []
    for fold in range(3):
        run = find_run(root, culture, f"{base_name}__films_{condition}{fold}__pooled")
        folds.append(per_subject(run) if run else {})
    people = set.intersection(*(set(f) for f in folds))
    return {p: float(np.mean([f[p][0] for f in folds])) for p in sorted(people)}


def simple_film_reference(path=SIMPLE_FILM_MD):
    """Culture name -> (seen, unseen) balanced accuracy of the simple EEG model."""
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 4 and cells[1] == "EEG":
                out[cells[0]] = (float(cells[2]), float(cells[3]))
    return out


def mean_acc(results):
    return float(np.mean([a for a, _ in results.values()])) if results else None


def bootstrap_ci(values, n=10000, seed=0):
    values = np.asarray(values)
    means = np.random.default_rng(seed).choice(values, size=(n, len(values))).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_line(label, a, b):
    """'a − b' over paired people, with CI and Wilcoxon, or a note when nothing is paired."""
    if len(a) < 2:
        return f"- {label}: not enough paired people."
    difference = np.array(a) - np.array(b)
    low, high = bootstrap_ci(difference)
    p_value = wilcoxon(a, b).pvalue if np.any(difference) else 1.0
    return (f"- {label}: {difference.mean():+.1f} points over {len(a)} pairs "
            f"(95% CI {low:+.1f} to {high:+.1f}), Wilcoxon p = {p_value:.3g}.")


def fmt(value):
    return "—" if value is None else f"{value:.1f}"


def part1(scores, chosen, lines):
    base = baselines()
    rows, fixed_pairs, released_pairs, wins = [], [], [], 0
    for source, target in PAIRS:
        config = chosen[target]
        fixed_dir = find_run(GRID_ROOT, target, f"{config}__from_{source}") if config else None
        released = per_subject(RELEASED_ROOT / target / f"strict__bn_source__from_{source}")
        fixed_ens = ensemble(fixed_dir) if fixed_dir else {}
        p1_dir = PAPER_ROOT / target / f"reported_diagnostic__from_{source}"
        row = dict(direction=f"{NAMES[source]} → {NAMES[target]}", baseline=base.get((source, target)),
                   released=mean_acc(released), single=mean_acc(per_subject(fixed_dir) if fixed_dir else {}),
                   ens=mean_acc(fixed_ens),
                   ens_f1=float(np.mean([f for _, f in fixed_ens.values()])) if fixed_ens else None,
                   p1=mean_acc(per_subject(p1_dir)), seeds=f"{seed_count(fixed_dir)}/{seed_count(p1_dir)}")
        rows.append(row)
        wins += row["ens"] is not None and row["baseline"] is not None and row["ens"] > row["baseline"]
        for subject in sorted(set(fixed_ens) & set(released)):
            fixed_pairs.append(fixed_ens[subject][0])
            released_pairs.append(released[subject][0])

    def average(key):
        values = [r[key] for r in rows]
        return None if any(v is None for v in values) else float(np.mean(values))

    lines += ["", "## Part 1 — Across cultures: accuracy (%) per direction", "",
              "| Train → Test | Best simple model | SCVAE released (strict) | SCVAE-fixed, 1 seed (strict) | "
              "SCVAE-fixed, 3-seed ensemble (strict) | Ensemble macro-F1 || Released, **test-label-selected** "
              "| seeds (fixed/P1) |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lines.append(f"| {r['direction']} | {fmt(r['baseline'])} | {fmt(r['released'])} | {fmt(r['single'])} | "
                     f"**{fmt(r['ens'])}** | {fmt(r['ens_f1'])} || {fmt(r['p1'])} | {r['seeds']} |")
    lines.append(f"| **Average** | {fmt(average('baseline'))} | {fmt(average('released'))} | "
                 f"{fmt(average('single'))} | **{fmt(average('ens'))}** | {fmt(average('ens_f1'))} || "
                 f"{fmt(average('p1'))} | |")
    lines += ["", paired_line("SCVAE-fixed ensemble − SCVAE released (strict), (direction, person) pairs",
                              fixed_pairs, released_pairs)]
    if average("p1") is not None and average("released") is not None:
        lines.append(f"- Peeking gain, released recipe (test-label-selected − strict): "
                     f"{average('p1') - average('released'):+.1f} points.")
    ens_avg, base_avg = average("ens"), average("baseline")
    if ens_avg is not None:
        met = ens_avg > base_avg and wins >= SUCCESS_WINS
        lines += ["", "**Pre-registered success criterion:** "
                  f"SCVAE-fixed ensemble {ens_avg:.1f}% vs best simple models {base_avg:.1f}%; wins {wins} of 6 "
                  f"(need > {base_avg:.1f}% and ≥ {SUCCESS_WINS}). **{'MET' if met else 'NOT MET'}.** "
                  "Predicted: 58% strict, 72% test-label-selected."]

    lines += ["", "### Grid factors (exploratory; mean over all 6 directions and the other factors)", "",
              "| Factor | Level | Mean accuracy |", "|---|---|---:|"]
    for factor, levels in [("learning rate", {"0.0025": lambda c: "__lrate" not in c,
                                              "0.0005": lambda c: "__lrate0.0005" in c}),
                           ("BN at test", {"source": lambda c: "__bn_source" in c,
                                           "target": lambda c: "__bn_target" in c}),
                           ("input noise", {"0": lambda c: "__noise" not in c,
                                            "0.1": lambda c: "__noise0.1" in c})]:
        for level, matches in levels.items():
            values = [v for (c, _, _), v in scores.items() if matches(c)]
            lines.append(f"| {factor} | {level} | {fmt(float(np.mean(values)) if values else None)} |")


def part2(chosen, lines):
    models = [("SCVAE released + BN fix (strict)", lambda c: "strict__bn_source"),
              ("SCVAE-fixed (strict)", lambda c: chosen[c]),
              ("SCVAE released, **test-label-selected**", lambda c: "reported_diagnostic")]
    reference = simple_film_reference()
    lines += ["", "## Part 2 — Within each culture: seen vs unseen films (accuracy %)", "",
              "Same test windows in both columns; same amount of training data. Each person is averaged over the "
              "3 film folds and the seeds.", "",
              "| Culture | Model | Seen films | Unseen films | Drop | People dropping | Wilcoxon p |",
              "|---|---|---:|---:|---:|---|---:|"]
    drops = defaultdict(list)
    for culture in CULTURES:
        for label, base_name in models:
            name = base_name(culture)
            seen = film_people(FILM_ROOT, culture, name, "seen") if name else {}
            unseen = film_people(FILM_ROOT, culture, name, "unseen") if name else {}
            people = sorted(set(seen) & set(unseen))
            if len(people) < 2:
                lines.append(f"| {NAMES[culture]} | {label} | — | — | — | — | — |")
                continue
            a, b = np.array([seen[p] for p in people]), np.array([unseen[p] for p in people])
            p_value = wilcoxon(a, b).pvalue if np.any(a - b) else 1.0
            drops[label].append(float((a - b).mean()))
            lines.append(f"| {NAMES[culture]} | {label} | {a.mean():.1f} | {b.mean():.1f} | {(a - b).mean():+.1f} | "
                         f"{int(np.sum(a > b))}/{len(people)} | {p_value:.3g} |")
        if NAMES[culture] in reference:
            seen, unseen = reference[NAMES[culture]]
            lines.append(f"| {NAMES[culture]} | simple model, EEG (balanced acc., `clip_heldout_emotion.md`) | "
                         f"{seen:.1f} | {unseen:.1f} | {seen - unseen:+.1f} | | |")
    lines += ["", "Average drop over the cultures with results: " + "; ".join(
        f"{label} {np.mean(values):+.1f}" for label, values in drops.items()) +
        ". Predicted: ≈ +8 (strict models), ≈ +15 (test-label-selected)."]


def part3(chosen, lines):
    lines += ["", "## Part 3 — German ↔ French: shared vs other films (SCVAE-fixed ensemble, strict)", "",
              "Balanced accuracy (%) over negative and positive windows (neutral has no shared film).", "",
              "| Train → Test | Shared films | Other films | Shared − other | People higher on shared | Wilcoxon p |",
              "|---|---:|---:|---:|---|---:|"]
    for source, target in [("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER")]:
        config = chosen[target]
        run = find_run(GRID_ROOT, target, f"{config}__from_{source}") if config else None
        split = shared_split(run, target) if run else {}
        if len(split) < 2:
            lines.append(f"| {NAMES[source]} → {NAMES[target]} | — | — | — | — | — |")
            continue
        shared, other = np.array([v[0] for v in split.values()]), np.array([v[1] for v in split.values()])
        p_value = wilcoxon(shared, other).pvalue if np.any(shared - other) else 1.0
        lines.append(f"| {NAMES[source]} → {NAMES[target]} | {shared.mean():.1f} | {other.mean():.1f} | "
                     f"{(shared - other).mean():+.1f} | {int(np.sum(shared > other))}/{len(split)} | {p_value:.3g} |")
    lines.append("\nPredicted: shared ≈ 5 points above other films.")


def main():
    scores = load(GRID_ROOT, pattern="*/*")
    configs = sorted(config_name(*g) for g in GRID)
    lines = ["# SCVAE-Net across cultures and on unseen films", "",
             "Spec: `docs/superpowers/specs/2026-10-09-scvae-crossculture-improvement-design.md`. "
             "Strict columns: no target label read before the final evaluation. "
             "**Test-label-selected** columns pick the checkpoint by test accuracy (the released code's policy); "
             "they are a labelled comparison, not accuracy.", "",
             "## Configuration chosen per test culture (from the other two cultures' transfers only)", "",
             "| Test culture | Chosen configuration | Validation acc. |", "|---|---|---:|"]
    chosen = {}
    for target in CULTURES:
        result = choose(scores, configs, target)
        chosen[target] = result[0] if result else None
        lines.append(f"| {NAMES[target]} | {result[0] if result else 'incomplete'} | "
                     f"{fmt(result[1] if result else None)} |")
    part1(scores, chosen, lines)
    part2(chosen, lines)
    part3(chosen, lines)
    OUTPUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m src.experiments.test_summarize_scvae_improve`
Expected: `Summary tests passed.`

- [ ] **Step 5: Smoke-run with no new results** (must not crash)

Run: `.venv/bin/python -m src.experiments.summarize_scvae_improve | head -40`
Expected: choice table `incomplete` everywhere, `—` cells, the simple-model reference rows in Part 2, no traceback. Then `rm results/tables/scvae_improve_summary.md`.

- [ ] **Step 6: Commit**

```bash
git add src/experiments/summarize_scvae_improve.py src/experiments/test_summarize_scvae_improve.py
git commit -m "Add the SCVAE study summary: Parts 1-3, ensemble and success check"
```

---

### Task 9: Run on the cluster and report

**Files:**
- Create (generated): `cluster/scvae_improve_films.txt`, `results/scvae_improve/**`, `results/tables/scvae_improve_summary.md`
- Modify: `EXPLANATION.md` (section 7 and a new subsection on unseen films)

- [ ] **Step 1: Full local test pass**

Run: `for t in test_scvae_reference test_scvae_improve test_select_crossculture_config test_make_scvae_improve_manifest test_summarize_scvae_improve; do .venv/bin/python -m src.experiments.$t || break; done`
Expected: five "passed" lines.

- [ ] **Step 2: Push code and check the cluster environment**

Run: `cluster/sync.sh push && ssh singularity 'cd ~/EEG-Emotion && source cluster/env.sh'`
Expected: `Python: ...`. If "No Python environment found": `ssh singularity 'cd ~/EEG-Emotion && sbatch cluster/setup_env.sbatch'` and submit Step 3 with `--dependency=afterok:<that job id>`.

- [ ] **Step 3: Submit stage 1 (grid + P1 across cultures, 162 tasks)**

Run: `ssh singularity 'cd ~/EEG-Emotion && sbatch --export=ALL,MANIFEST=cluster/scvae_improve_grid.txt --array=1-162 cluster/train_array.sbatch'`
Expected: `Submitted batch job <id1>`.

- [ ] **Step 4: Wait, then check for failures**

Run (when `ssh singularity squeue -u harsh.dubey` lists none of `<id1>`): `ssh singularity 'sacct -j <id1> --format=JobID,State -n | grep -v COMPLETED | grep -v "\.ba\|\.ex" || echo all-completed'`
Expected: `all-completed`. Failed tasks: read `results/logs/slurm/scvae_<id1>_<task>.err` on the cluster, fix, resubmit those indices with `--array=<list>` (`--skip-existing` keeps finished folds).

- [ ] **Step 5: Pull, choose configurations, generate and submit stage 2 (films, 162 tasks)**

Run: `cluster/sync.sh pull && ls results/scvae_improve/grid/SEED_GER/*/session_all/seed_12/*_proba.npz | head -2 && .venv/bin/python -m src.experiments.make_scvae_improve_manifest films && cluster/sync.sh push && ssh singularity 'cd ~/EEG-Emotion && sbatch --export=ALL,MANIFEST=cluster/scvae_improve_films.txt --array=1-162 cluster/train_array.sbatch'`
Expected: two `.npz` paths, three `<target>: <config> (validation ...)` lines, `Wrote 162 lines`, `Submitted batch job <id2>`.

- [ ] **Step 6: Wait, check failures (as in Step 4 with `<id2>`), pull and summarise**

Run: `cluster/sync.sh pull && .venv/bin/python -m src.experiments.summarize_scvae_improve`
Expected: Part 1 seeds `3/3` in every row, every cell filled, a `MET` or `NOT MET` line; Part 2 all 9 culture × model rows filled; Part 3 both rows filled.

- [ ] **Step 7: Update `EXPLANATION.md`.** Section 7: the Part 1 direction table (strict and test-label-selected columns separated and labelled) and the success verdict exactly as printed. New subsection after it, "SCVAE on unseen films": the Part 2 table and the Part 3 table, each with its prediction next to the result. Report outcomes as found, including NOT MET.

- [ ] **Step 8: Commit**

```bash
git add cluster/scvae_improve_films.txt results/scvae_improve results/tables/scvae_improve_summary.md EXPLANATION.md
git commit -m "Add SCVAE cross-culture and unseen-film results"
```
