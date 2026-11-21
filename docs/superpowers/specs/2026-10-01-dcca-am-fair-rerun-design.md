# DCCA-AM, re-run fairly: design

*1 October 2026. Status: draft for review.*

## Goal

Liu et al. (J. Neural Eng. 2022) report that DCCA-AM, a deep EEG + eye fusion
network, reaches 73.6–84.3% across cultures (their table 9), far above any
simple model. Its released code picks the architecture and the epoch by test
accuracy. We re-run DCCA-AM in the 6 cross-culture directions and report the
same trained networks three ways, to measure how much of the reported number
is test-label selection and how much is film recognition. We also train it to
identify films (as E2 did with a linear model), to test directly whether the
paper's fusion representation carries film identity across cultures.

**Success:** for each direction, a table with the paper's number, our
paper-rules number, our fair number and our fair new-films number (mean over
test people, 95% CI, 3 seeds), plus a curve of the paper-rules number against
the number of architectures searched, plus the DCCA-AM film-identification
accuracy next to E2's linear one. Whatever the numbers turn out to be, the
result is reported.

**Out of scope (later, if wanted):** within-culture (table 7; one model per
test person, about 20× the compute); within-person (tables 5, the released
script's own setting); BDAE and the other fusion methods.

## What the released code does (the reference)

`data/raw/SEED_GER/code/CrossCultureCode-main/DCCA-AM/` (Chinese,
within-person script only; the cross-culture code was not released):

- **Network:** two transform networks, EEG (310 → h1 → h2 → 12) and eye
  (33 → h1 → h2 → 12), sigmoid hidden layers, linear output; attention fusion
  `α_i = softmax(x1_i·w, x2_i·w)`, fused = `α1 x1 + α2 x2`; linear classifier
  12 → 3.
- **Training:** RMSprop, lr 1e-4 for the classifier optimiser (all
  parameters) and 5e-5 for two transform optimisers (parameter slices
  `[0:8]` and `[8:16]`, which overlap the classifier optimiser and each
  other); batch 30, not shuffled; 100 epochs. The CCA gradient is computed in
  NumPy per batch and back-propagated with weight −0.1, plus cross-entropy.
  Errors inside a batch are skipped silently (`except: continue`).
- **Scaling:** `MinMaxScaler` fitted separately on the training set and on the
  test set (each set uses its own min and max; label-free).
- **Selection (the leak):** 100 random architectures, `h1 ~ U[100, 200)`,
  `h2 ~ U[20, 50)`; after every epoch the test accuracy is computed and the
  best is kept. The reported accuracy is the best over architectures × epochs.

## Design

### 1. Model: `src/models/dcca_am.py`

A faithful port of `DCCA_AM`, `TransformLayers`, `AttentionFusion` and
`cca_metric_derivative`, with the same optimisers, parameter slices, gradient
weight, batch order and silent-skip behaviour (counted and logged rather than
hidden). One change for speed: the per-sample attention loop becomes one
batched operation; a test checks it gives the same output as the loop.

### 2. Data: `src/experiments/dcca_data.py`

- **EEG:** the raw DE-LDS features (310 = 62 channels × 5 bands), loaded from
  `data/raw/<culture>/02-EEG-DE-feature*/eeg_used_4s/*.npz` as in
  `liu_leakage_audit.load_bcmi` (the processed EEG files are z-scored per
  window, which the paper did not do).
- **Eye:** the 33 raw eye features, from columns 310:343 of the processed
  `data/processed/<culture>_EEG_EYE` files (raw units).
- **Clip id and label per window:** from the processed files.
- Per recording, the two sources must have the same number of windows and the
  same labels; any mismatch stops the run.
- Cultures: SEED_CHN_MM (12 people), SEED_GER (8), SEED_FRA (8); 4 s windows.
- Scaling as released: `MinMaxScaler` fitted on the whole training set, and
  separately on each test person's windows (all their sessions).

### 3. Training runs: `src/experiments/train_dcca_am.py`

One run = (direction, seed, architecture index, task). For the 6 cross-culture
directions, a single network is trained on the whole source culture and
evaluated on every person of the target culture after every epoch, so one
run serves all test people. Per run it saves, for every epoch, each test
person's accuracy, macro-F1 and prediction counts, plus the number of skipped
batches. Architectures: index 0 is the fixed one (h1 = 150, h2 = 35, the
middle of their range); indices 1–20 are drawn from their distribution with a
seeded generator. Seeds 12, 34, 56. Output layout like the other runs:
`results/dcca_am/<target>/<task>__from_<source>/seed_<s>/arch_<k>.json`.

Tasks:

- **emotion, all films:** the paper's setting.
- **emotion, new films:** German ↔ French only; the 5 shared films are removed
  from the source training data (Chinese directions are already new-film).
- **film identification:** German ↔ French only; the label is the film, within
  one emotion, using E2's tasks and film mapping
  (`film_identification.FILMS`, `TASKS`); evaluated with balanced accuracy and
  E2's block-permutation test. Reported in fair mode (architecture 0, epoch
  100), with the paper-rules number alongside.

### 4. Evaluation modes: `src/experiments/summarize_dcca_am.py`

All computed from the saved per-epoch results, without retraining:

| Mode | Architecture | Epoch | Uses test labels to choose? |
|---|---|---|---|
| Paper's rules | best of indices 1–20, per test person | best, per test person | yes (as released) |
| Fair | index 0 (fixed in advance) | 100 (last) | no |
| Fair, new films | index 0, trained without the shared films | 100 | no |

Also: the paper-rules accuracy when searching only the first k architectures,
k = 1, 2, 5, 10, 20 (the leakage curve), and our best simple normalised model
from `crossculture_summary.csv` for reference. Means over test people with
95% bootstrap CIs; paper's table 9 numbers alongside.

### 5. Cluster

- Manifest `cluster/dcca_sweep.txt`, one line per (direction, seed, task),
  each training all 21 architectures in sequence: emotion 6 × 3 = 18 lines,
  new-films 2 × 3 = 6, film identification 2 × 3 = 6; 30 tasks.
- `train_array.sbatch`-style GPU array job, after the leakage audits finish.
- Estimate (to confirm with the smoke run): 5–10 GPU-hours in total. The NumPy
  CCA step runs on the CPU every batch, so the job asks for 4 cores.
- `--skip-existing`, so a resubmission resumes.

### 6. Checks before submitting

- `src/models/test_dcca_am.py`: batched attention equals the loop; the CCA
  derivative matches a finite-difference check on a small random case; one
  training step runs on random data.
- `src/experiments/test_dcca_data.py`: EEG/eye alignment and labels for every
  recording; the new-films task removes exactly the shared films.
- The fair-mode summary takes only (architecture 0, epoch 100) and is tested
  to never read target labels before scoring.
- Local smoke run: one direction, one seed, 2 architectures, 5 epochs; timed
  to refine the estimate.

## Risks

- **Reproduction gap:** without the released cross-culture code, our
  paper-rules number may not reach theirs. The leakage curve still shows how
  much selection adds; the gap is reported, not hidden.
- **Instability:** the CCA step can fail (singular covariances); skipped
  batches are counted per run and reported.
- **Compute:** if the smoke run shows it is much slower than estimated, reduce
  to 10 architectures and say so.
