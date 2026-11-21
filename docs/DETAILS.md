# Details: strict reproduction of SCVAE-Net, cluster setup


Cross-subject emotion recognition (negative / neutral / positive) on the SEED
EEG dataset. It uses a supervised-contrastive VAE (SCVAE-Net) with MMD
alignment between source subjects and an unlabelled held-out subject.

The repository has three goals:

1. **Reproduce SCVAE-Net honestly.** The released training code picks
   checkpoints by target-subject accuracy. The `strict` protocol here never
   reads target labels until the final evaluation.
2. **Explain its failures.** Some strict folds collapse to a single predicted
   class. We traced this to stale BatchNorm running statistics.
3. **Test extensions**, such as uncertainty-weighted MMD, against the fixed
   baseline.

## Headline results

Strict protocol, 4-second `de_LDS` features, 3 sessions × 3 seeds × 15
held-out subjects = 135 folds. Tables are in `results/tables/`.

| Run | Mean acc. | Macro-F1 | Collapsed / partial folds | vs. previous row |
|---|---|---|---|---|
| Strict baseline (`strict`) | 0.650 | 0.604 | 10 / 37 | — |
| + BN recalibration on source (`strict__bn_source`) | **0.759** | **0.747** | 0 / 0 | +0.109, 86/9/40 W/T/L, Wilcoxon p = 2e-9 |
| + entropy-weighted MMD (`strict__entropy_weighted_mmd__bn_source`) | 0.746 | 0.735 | 0 / 0 | −0.013, 61/10/64, p = 0.22 |

What the results show so far:

- **Collapse comes from BatchNorm, not from domain shift.** Collapsed folds fit
  the source data in train mode (CE ≈ 0.01) but reach only ≈ 45% eval-mode
  *source* accuracy. The classifier's BN running statistics lag weights that
  are still moving under a constant learning rate. They are not contaminated
  by target data. See [Why BatchNorm statistics go stale](#why-batchnorm-statistics-go-stale-mechanism-sweep).
- **Uncertainty-weighted MMD does not beat the fixed baseline** (entropy
  variant above; the posterior variant is also lower on the folds finished so
  far). Always compare new alignment variants against `bn_source`, not against
  the 0.650 baseline.
- **A cosine learning-rate schedule prevents collapse without
  recalibration.** On the 19-fold mechanism subset it scores 0.722 vs. 0.615
  for the released constant-LR training (13/5 wins, p = 0.048), with
  eval-mode source accuracy 1.00 on every fold.
- Strict SVM and MLP baselines on the same 135 folds score 0.502 and 0.532
  (`results/tables/strict_baselines_vs_scvae.csv`).

Side experiments that are not part of the current story are archived outside
the repository in `~/Documents/EEG-Emotion-SCVAE_archive/` (same relative
paths). They are the 1-second-window ablation (collapses because of BN and
reaches only ≈ 0.42 after the fix), the few-shot proof of concept, the
single-fold `reported_diagnostic` run, the earlier pooled-LOSO tables and
checkpoints, and cross-culture runs on the 15-subject official SEED features.

## Repository layout

```
src/
  preprocessing/   SEED → per-subject tensors; dataset loaders; verification
  models/          scvae_reference.py (released SCVAE-Net)
  experiments/     training entry points, diagnostics, sweeps, run comparison
  baselines/       strict SVM/MLP baselines; Liu et al. leakage audit
  legacy/          earlier in-house SCVAE, plain VAE and their LOSO pipeline
cluster/           SLURM array job, Singularity definition, manifest, rsync helper
data/processed/    converted tensors (SEED, SEED_4s_trials, SEED_CHN_MM, SEED_GER, SEED_FRA)
results/           per-fold CSV/JSON, diagnostics, summary tables, SLURM logs
checkpoints/       model weights (large; not synced back from the cluster)
```

## Setup

Run every command from the repository root. The modules import from `src`, so
use `python -m src....`, not `python src/....py`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The commands below assume the virtualenv is active.

### Data

Each converted file is `data/processed/<dataset>/<session>/subject<N>.pth`. It
holds a dict with `data` (a list of `62 × 5` tensors: channels ×
δ/θ/α/β/γ bands of DE-LDS features, z-scored per sample) and `labels`
(0 = negative, 1 = neutral, 2 = positive).

| Directory | Contents | Used by |
|---|---|---|
| `data/processed/SEED` | Released SCVAE-Net 4 s `de_LDS` features (default) | all strict runs |
| `data/processed/SEED_4s_trials` | Official 4 s features with trial IDs kept (alias `SEED_CHN`) | few-shot protocol; 15-subject cross-culture check |
| `data/processed/SEED_GER`, `SEED_FRA` | BCMI SEED-GER / SEED-FRA 4 s DE (LDS-smoothed) features with clip IDs (build `_1s` variants with `--window-seconds 1` if needed) | cross-culture experiments |
| `data/processed/SEED_CHN_MM` | Chinese SEED multimodal subset: the 12 subjects with eye tracking (1–5, 8–14), EEG re-extracted by the same pipeline as GER/FRA | Chinese side of cross-culture (what Liu et al. used) |

`SEED_CHN_MM` is built with `--dataset CHN` from `data/raw/SEED_CHN_MM`. Its
EEG is the same recordings as SEED (r ≈ 0.98) but not the same numbers
(offset ≈ +2.4, different smoothing), so use it rather than `SEED_CHN` whenever
Chinese is compared with German or French.

SEED-GER has 8 subjects but only 20 sessions: subjects 5–8 completed two, so
session 3 has 4 subjects. SEED-FRA has 8 subjects × 3 sessions. Both are built
from the licensed BCMI download in `data/raw/SEED_{GER,FRA}`:

```bash
python -m src.preprocessing.convert_seed_crossculture --dataset GER --window-seconds 4 \
  --source-dir data/raw/SEED_GER/02-EEG-DE-features/eeg_used_4s --output-dir data/processed/SEED_GER
```

The training and baseline scripts read the subject list from the files, so
`--data-dir data/processed/SEED_GER` (or `SEED_FRA`) is all they need.

The official directories are built with `convert_seed_extracted_features`
(see below) and record their source in `provenance.txt`. Check any dataset
with:

```bash
python -m src.preprocessing.verify_processed                      # default: data/processed/SEED
python -m src.preprocessing.verify_processed --data-dir data/processed/SEED_GER
```

SEED is distributed by BCMI (SJTU) under a licence agreement. The raw and
feature files are not part of this repository.

## Evaluation protocols

Results from different protocols must never be pooled.

| Protocol | Target labels used for | Reportable? | Output root |
|---|---|---|---|
| `strict` (default) | final evaluation only | **yes** | `…/strict*/` |
| `reported_diagnostic` | checkpoint selection (mirrors released code) | no, diagnostic only | `…/reported_diagnostic/` |
| few-shot calibration | 1 labelled trial per class fine-tunes the classifier | separate protocol | `results/fewshot_scvae/` |

## Strict SCVAE-Net reference reproduction

`train_scvae_reference` implements the released SCVAE-Net architecture and its
session-wise cross-subject protocol. Unlike the released training script, it
does not inspect target labels during training or checkpoint selection.

```bash
# Fast wiring test; never report these one-batch metrics.
python -m src.experiments.test_scvae_reference
python -m src.experiments.train_scvae_reference \
  --session 1 --test-subject 2 --epochs 1 --max-train-batches 1 \
  --output-dir results/reference_scvae/smoke \
  --checkpoint-dir checkpoints/reference_scvae/smoke

# One strict fold. This is the first meaningful reproduction run.
python -m src.experiments.train_scvae_reference \
  --session 1 --test-subject 2 --seed 12 --epochs 300 \
  --output-dir results/reference_scvae \
  --checkpoint-dir checkpoints/reference_scvae

# Full session-wise reproduction, run sequentially. Prefer the cluster (below).
python -m src.experiments.train_scvae_reference \
  --session 1 --all-subjects --seed 12 --epochs 300
```

Each fold writes `subject_XX.csv`, `subject_XX.json` and `subject_XX.pt` under
`<run>/session_XX/seed_XX`. An all-subject run also writes `summary.csv`.
Common flags:

| Flag | Values (default first) | Purpose |
|---|---|---|
| `--protocol` | `strict`, `reported_diagnostic` | see table above |
| `--alignment` | `standard_mmd`, `entropy_weighted_mmd`, `posterior_weighted_mmd`, `combined_uncertainty_mmd` | domain-alignment loss |
| `--bn-recalibration` | `none`, `source`, `target` | re-estimate BN stats after the last epoch |
| `--lr-schedule` | `constant`, `cosine` | constant is the released behaviour |
| `--target-bn-update` | `on`, `off` | whether target passes update BN running stats |
| `--data-dir` | `data/processed/SEED` | switch feature set |
| `--skip-existing` | — | skip folds whose result JSON exists (resumable) |
| `--device` | `auto`, `cpu`, `mps`, `cuda` | |

Run `python -m src.experiments.train_scvae_reference --help` for the rest
(learning rate, latent size, temperatures, logging).

### Prediction collapse and BatchNorm recalibration

`diagnose_collapse` audits saved strict folds for collapse, meaning folds that
predict almost every target window as one class. It reports label-free signals
(predicted-class shares, confidence, latent statistics, source accuracy). It
uses target labels only to validate those signals.

```bash
python -m src.experiments.diagnose_collapse \
  --results-root results/reference_scvae/strict \
  --output-dir results/diagnostics/collapse/reference_4s
```

On the frozen 4-second baseline, 10 of 135 folds collapse and 37 are partial.
`--bn-recalibration source` re-estimates the classifier's BN statistics from
source features after the final epoch, with no target data. `target` uses
unlabelled target features instead, as MMD already does. Neither reads target
labels.

```bash
python -m src.experiments.train_scvae_reference \
  --protocol strict --bn-recalibration source \
  --session 1 --test-subject 2 --seed 12 --epochs 300
```

Runs are stored under `strict__bn_<mode>` (or
`strict__<alignment>__bn_<mode>`). `src/experiments/sweep_bn_uncertainty.sh`
runs the full strict sweep over standard and uncertainty-weighted MMD. It is
resumable, and `JOBS=3` runs three folds in parallel. Compare runs fold by
fold with:

```bash
python -m src.experiments.compare_runs \
  results/reference_scvae/strict__bn_source \
  results/reference_scvae/strict__entropy_weighted_mmd__bn_source \
  results/reference_scvae/strict__posterior_weighted_mmd__bn_source \
  results/reference_scvae/strict__combined_uncertainty_mmd__bn_source
```

A retrained fold (session 1, seed 12, subject 2) did not reproduce the frozen
checkpoint's training loss with the same seed. The cause is either MPS
nondeterminism or code changes made since. Always compare runs trained in the
same sweep.

### Why BatchNorm statistics go stale (mechanism sweep)

The running statistics are stale, not contaminated by target data. The
weights still move at the end of training (constant learning rate), so the
exponential average lags behind them. `sweep_bn_mechanism.sh` retrains the 10
collapsed and 9 healthy folds listed in `bn_mechanism_folds.txt`, with no
post-hoc recalibration, in three variants: the released training,
`--target-bn-update off` and `--lr-schedule cosine`. Every run records
`bn_running_mean_rel_distance` and `eval_source_accuracy` in
`final_training_metrics`.

```bash
JOBS=3 src/experiments/sweep_bn_mechanism.sh    # → results/bn_mechanism/
python -m src.experiments.summarize_bn_mechanism

# Inference-only evidence on the saved collapsed checkpoints (no training):
python -m src.experiments.check_bn_running_stats   # running mean vs any source/target mixture
python -m src.experiments.replay_bn_updates        # replay released BN updates on frozen weights
```

The replay recovers all 10 collapsed folds (target accuracy 0.345 → 0.778,
eval-mode source accuracy 0.45 → 0.98). The saved running mean is 4–33% away
from every source/target mixture, while the source and target means differ by
only 0–8%.

| Variant (19 folds) | Mean acc. |
|---|---|
| released (constant LR) | 0.615 |
| `--target-bn-update off` | 0.584 (p = 0.14; no help) |
| `--lr-schedule cosine` | **0.722** (p = 0.048; 0 folds with any class ≥ 70% share) |

Which fold collapses varies at random between reruns. The next step is cosine
LR on all 135 folds, with and without `bn_source`.

### Uncertainty-weighted MMD

`--alignment standard_mmd` is the frozen reference. The three strict
ablations are `entropy_weighted_mmd`, `posterior_weighted_mmd` and
`combined_uncertainty_mmd`. Target weights are detached from backpropagation
and include a uniform component. This stops the model from lowering the
alignment loss by changing its own uncertainty.

```bash
python -m src.experiments.train_scvae_reference \
  --protocol strict --alignment combined_uncertainty_mmd \
  --session 1 --test-subject 1 --seed 12 --epochs 3 --max-train-batches 1 \
  --output-dir results/uncertainty_smoke \
  --checkpoint-dir checkpoints/uncertainty_smoke
```

### Official 1-second DE/LDS feature ablation

The strict baseline uses the released SCVAE-Net's **4-second** `de_LDS`
features. Do not overwrite it. The official SEED 1-second features are a
separate temporal-resolution experiment, not a replacement baseline.

```bash
python -m src.preprocessing.convert_seed_extracted_features \
  --source-dir <SEED_EEG>/ExtractedFeatures_1s \
  --output-dir data/processed/SEED_1s --window-seconds 1
```

To evaluate it, pass `--data-dir data/processed/SEED_1s`. Store the results
under a separate root and label them as a 1-second-window ablation.

### Few-shot target calibration (separate protocol)

This is **not** strict cross-subject evaluation. The held-out subject
contributes one complete labelled film trial per emotion. Only the classifier
is fine-tuned on these trials, and testing uses the remaining twelve complete
trials. Windows from a calibration trial never appear in the test split. The
base SCVAE stage is transductive: MMD sees all target features, without their
labels.

```bash
# Trial-aware copy of the official 4-second data.
python -m src.preprocessing.convert_seed_extracted_features \
  --source-dir <SEED_EEG>/ExtractedFeatures_4s \
  --output-dir data/processed/SEED_4s_trials --window-seconds 4

# One deliberately hard proof-of-concept fold.
python -m src.experiments.train_scvae_fewshot \
  --session 1 --test-subject 2 --seed 12 \
  --support-trials-per-class 1 --epochs 300 --calibration-epochs 20 \
  --output-dir results/fewshot_scvae \
  --checkpoint-dir checkpoints/fewshot_scvae
```

Its output is labelled `fewshot_transductive_trial_disjoint`. Never combine it
with strict, target-label-free results.

### Released-protocol diagnostic

This mode copies the released code, which checks target accuracy after every
update and keeps the best target checkpoint. It is a diagnostic comparison,
not a valid held-out result. Its artifacts go under `reported_diagnostic/`.

```bash
python -m src.experiments.train_scvae_reference \
  --protocol reported_diagnostic --session 1 --test-subject 2 \
  --seed 12 --epochs 300 --log-every 25 \
  --output-dir results/reference_scvae \
  --checkpoint-dir checkpoints/reference_scvae
```

## Cross-culture extension (SEED, SEED-GER, SEED-FRA)

Plain-language overview: [EXPLANATION.md](EXPLANATION.md). The baseline paper
is Liu et al., *J. Neural Eng.* 19 (2022) 026012. We follow its two
subject-independent settings with `--split`, and the strict protocol still
applies:

| `--split` | Source | Target | Liu et al. name |
|---|---|---|---|
| `session` (default) | other subjects, same session | one subject, one session | (SEED protocol) |
| `pooled` | other subjects, all sessions | one subject, all sessions | ICSI |
| `cross` | every subject of `--source-data-dir` | one subject of `--data-dir`, all sessions | CCSI |

Use `data/processed/SEED_CHN_MM` (12 subjects, same extraction as GER/FRA, the
subset the paper used) for the Chinese side. `data/processed/SEED_CHN` is an alias of
`SEED_4s_trials` (official SEED 4 s `de_LDS`, all 15 subjects); its extraction
differs slightly, so treat it as a 15-subject check. `SEED` (the SCVAE-Net release) is kept for the
session-wise results above. Pooled and cross folds are written under
`session_all/`, and the run name gets a `__pooled` or `__from_<source>`
suffix.

```bash
# SCVAE-Net with the BN fix, Chinese -> German, one target subject
python -m src.experiments.train_scvae_reference --split cross --bn-recalibration source \
  --source-data-dir data/processed/SEED_CHN --data-dir data/processed/SEED_GER \
  --test-subject 1 --seed 12 --epochs 300 --output-dir results/crossculture/SEED_GER

# All strict SVM/MLP baselines (within + 6 cross directions), then the table
src/experiments/sweep_crossculture_baselines.sh
python -m src.experiments.summarize_crossculture   # -> results/tables/crossculture_summary.csv
```

**Their released code selects results by test accuracy.** In
`data/raw/SEED_GER/code/CrossCultureCode-main`, the SVM (C), the KNN (k), the
DNN (the best of 15,000 epochs × 20 learning rates) and DCCA-AM (epoch and
layer sizes) all keep the setting with the best test accuracy. The DNN also
scales the test set with its own statistics. Treat the paper's numbers as
optimistic references, not strict targets. (The released scripts are the
Chinese subject-dependent versions.)

## Earlier LOSO pipeline

The code lives in `src/legacy/`. `train_scvae_loso` is the original in-house
SCVAE with pooled leave-one-subject-out evaluation. The reference pipeline
above has replaced it as the main line of work.

```bash
# Wiring check: one subject, one epoch, one batch. Do not report.
python -m src.legacy.train_scvae_loso \
  --test-subject 2 --seed 42 --epochs 1 --max-train-batches 1 \
  --output results/tables/smoke_subject_02.csv \
  --checkpoint-dir checkpoints/scvae/smoke

# Full LOSO run, sequential.
python -m src.legacy.train_scvae_loso --seed 42

# One fold per job. Give every concurrent job a unique output path.
python -m src.legacy.train_scvae_loso \
  --test-subject 2 --seed 42 \
  --output results/tables/scvae/seed_42/subject_02.csv \
  --checkpoint-dir checkpoints/scvae/seed_42

# Sanity checks.
python -m src.legacy.test_supcon
python -m src.legacy.diagnose_scvae
```

Baselines: `python -m src.baselines.svm`, `python -m src.baselines.mlp` and
`python -m src.legacy.train_vae_loso`. The `src/legacy/save_{svm,mlp}_results`
scripts only write the already-recorded per-subject numbers to
`results/tables/`.

## Running on the SLURM + Singularity cluster

Run full sweeps on the cluster. Keep local runs to smoke tests. Everything
needed is in `cluster/`. Each array task runs one line of
`cluster/manifest.txt` (one configuration × 15 held-out subjects) on one GPU.
Tasks skip folds whose result JSON already exists, so resubmitting is safe.

The scripts are set up for the IITGN **Singularity** cluster
(8 L40S GPUs; partition `gpu` with QOS
`32c2g5dw`: at most 2 GPUs and 32 cores per user, 5-day wall time; no account
needed; see the Singularity user guide). Each job writes a `.out` (or `.log`) and a
`.err` file to `results/logs/slurm/`. Python comes from the cluster's
`anaconda3-2024` module in a conda env named `scvae`, which the job scripts
activate through `cluster/env.sh` (container and virtualenv are fallbacks). Three rules shape the workflow:
no work on the login node (even the environment is built by a job), at most 2
GPU tasks run at once (the rest queue), and home files untouched for 21 days
are deleted with no backup, so pull results often.

```bash
# Local, once: an SSH alias in ~/.ssh/config (campus network or VPN required)
#   Host singularity
#       HostName <cluster-host>
#       Port <port>
#       User <username>
# (or skip the alias: ssh <username>@<cluster-host>, CLUSTER=<username>@<cluster-host>)

# Local: copy code, processed data and raw DE features to the cluster.
REMOTE_DIR=<path-on-cluster> cluster/sync.sh push

# Cluster, once: create the conda env as a job, then read
# results/logs/slurm/setup_<jobid>.out for "cuda available True".
sbatch cluster/setup_env.sbatch

# Cluster: queue everything (paper sweep, main SEED sweep, both leakage audits).
cluster/submit_all.sh
squeue -u $USER

# Local: fetch result JSON/CSV and SLURM logs (checkpoints stay remote).
REMOTE_DIR=<path-on-cluster> cluster/sync.sh pull
```

The main SEED sweep is `cluster/main_sweep.txt`: the released training,
`bn_source`, `bn_target`, cosine LR, cosine + `bn_source` and the
`reported_diagnostic` leakage audit. That is 6 configurations × 3 sessions ×
3 seeds, i.e. 54 array tasks written to `results/main_sweep/`. Keeping every
headline row in one sweep lets them be compared fold by fold.

```bash
sbatch --export=ALL,MANIFEST=cluster/main_sweep.txt --array=1-54 cluster/train_array.sbatch
```

The cross-culture paper sweep is `cluster/paper_sweep.txt`: 54 tasks, 3 seeds
each, written to `results/paper_scvae/<target>/`. It covers SCVAE-Net within
each culture (BN fix, with and without per-recording normalisation) and across
cultures (the released recipe with the BN fix, and class-conditional MMD with a
shuffled target and 30 epochs). All seeds run on the cluster, so every number
comes from the same hardware. It replaces the earlier 99-task
`crossculture_sweep.txt`, whose released-recipe cross-culture runs swap or
collapse emotions.

```bash
sbatch --export=ALL,MANIFEST=cluster/paper_sweep.txt --array=1-54 cluster/train_array.sbatch
python -m src.experiments.summarize_crossculture --root results/paper_scvae
python -m src.experiments.check_results results/paper_scvae
```

To queue everything at once (paper sweep, main SEED sweep and both leakage
audits), run `cluster/submit_all.sh` on the cluster; `DRY_RUN=1` prints the
commands without submitting.

CPU-only work (strict baselines, the leakage audit) runs through
`cluster/cpu_task.sbatch`, which takes a Python module and its arguments:

```bash
sbatch cluster/cpu_task.sbatch src.baselines.liu_leakage_audit --cultures SEED_CHN_MM SEED_GER SEED_FRA
```

SVM and MLP baselines on the same strict session-wise folds run locally in
minutes. They write to `results/strict_baselines/{svm,mlp}/`:

```bash
python -m src.baselines.strict_baselines --model svm
python -m src.baselines.strict_baselines --model mlp
```

Set `--partition` and `--account` in `cluster/train_array.sbatch` and
`cluster/cpu_task.sbatch` if your cluster requires them. For a new experiment, write a new manifest (one line of
`train_scvae_reference` flags per task) and pass it with
`sbatch --export=ALL,MANIFEST=cluster/<name>.txt ...`. Logs go to
`results/logs/slurm/`.

## Reporting rules

- Report only `strict` runs as cross-subject results. Label few-shot,
  `reported_diagnostic` and 1 s runs as such, and keep them separate.
- Never report smoke-test (`--max-train-batches`) metrics.
- Compare variants fold by fold, within the same sweep, using `compare_runs`
  (it reports W/T/L and the Wilcoxon p-value).
