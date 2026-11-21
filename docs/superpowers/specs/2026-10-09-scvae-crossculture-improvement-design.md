# SCVAE-Net across cultures and unseen films — design

*9 October 2026. Status: revised after the professor's input (Direction 1, films left out of training, merged
in); draft for review.*

## Goal

1. Raise SCVAE-Net's cross-culture accuracy (train on one of Chinese / German / French, test on another, EEG
   only, 4 s DE features) under a strict protocol, against the simple normalised baselines.
2. Measure how much of SCVAE's accuracy is **film recognition**: score it on films that were left out of
   training, with and without the papers' test-label checkpoint selection.

Every number is in one of two kinds of column, always labelled:

- **Strict (reportable as accuracy):** no target label is read before the single final evaluation; settings are
  chosen on held-out *source* subjects and on the *other two* cultures' transfers only.
- **Test-label-selected (never reported as accuracy):** the released code's policy, evaluate on the test person
  after every update and keep the best update. Used only to show what that policy adds.

## Where we start

| Setting | Number | Source |
|---|---:|---|
| Cross-culture, SCVAE released + BN fix, average of 6 directions | 52.1 | `results/paper_scvae/` |
| Cross-culture, best normalised simple model, average | 57.6 | `results/tables/crossculture_summary.csv` |
| Within Chinese SEED, SCVAE + BN fix, strict | 74.3 | `EXPLANATION.md` §4 |
| Within Chinese SEED, released protocol (test-label-selected) | 93.1 | `EXPLANATION.md` §4 |
| Within culture, simple model, EEG, seen → unseen films | 75.9 → 69.5 (Chinese), 58.5 → 44.7 (German), 59.7 → 55.8 (French) | `results/tables/clip_heldout_emotion.md` |

The published SCVAE figures above 90% are within-Chinese; the published cross-culture DNN figures (Liu et al.)
are 55–66%.

**Films across cultures.** Each processed window carries its clip index (`trials`). The Chinese films differ
from the German and French ones, so cross-culture test films are already unseen, except for 5 films shown in
both Germany and France (scene-matched, not cut-identical; mapping in `film_identification.py`, `FILMS`). Within
a culture, every person watched the same films, so leaving films out matters most there.

## Part 1 — SCVAE-fixed across cultures (strict)

Each change answers something seen in the current runs:

| Observation | Change |
|---|---|
| Source accuracy 99–100%, classification loss ~0.01; 100 epochs worse than 30 | weight decay, input noise, cosine learning rate, epoch chosen on held-out source subjects |
| Released cross runs use no per-person normalisation, the main driver of the simple models | `--subject-norm recording` |
| Target batches in recording order are mostly one film, so class-blind MMD pulls all target emotions to the source mixture | `--target-shuffle on` |
| Target batches use their own BatchNorm statistics in training, source statistics at test | BN at test from source vs target subject, chosen without target labels |
| Nothing is chosen on held-out data | source-validation epoch selection + leave-target-culture-out configuration choice |

**New options** (`train_scvae_reference.py`; all default to the current behaviour and run names):

- `--weight-decay W` (Adam; suffix `__wd<W>`), `--input-noise S` (Gaussian noise on source inputs in training
  only, clean reconstruction target; suffix `__noise<S>`), `--learning-rate` non-default adds `__lrate<lr>`.
- `--source-val-subjects K` (`--split cross` or `pooled`): K whole source subjects, chosen from the seed, are
  removed from training; after every epoch their eval-mode accuracy (BatchNorm re-estimated on the training
  source) is computed with source labels only; the best epoch is restored before the final evaluation. Suffix
  `__sval<K>`. K = 3 when the source has 11–12 subjects (Chinese), 2 when it has 7–8 (German / French).
- `--save-proba`: save the target's per-window probabilities, labels and clip indices (`.npz`).
- `--protocol reported_diagnostic` stays restricted to standard MMD without `--bn-recalibration`.

**Grid.** Fixed: `--subject-norm recording --target-shuffle on --lr-schedule cosine --weight-decay 1e-4
--source-val-subjects K --epochs 100 --save-proba`, standard MMD weight 0.1. Varied (8 configurations):
learning rate ∈ {0.0025, 0.0005} × BN at test ∈ {source, target} × input noise ∈ {0, 0.1}.
8 × 6 directions × 3 seeds = **144 tasks**.

**Configuration choice (no target labels).** For each target culture T, a configuration is scored on the
transfers between the other two cultures only and the winner is reported on the transfers into T
(`select_crossculture_config.py`). SCVAE-fixed's prediction per person = the mean of the three seeds'
probabilities (label-free ensemble); single-seed numbers are reported too.

**P1 across cultures (test-label-selected).** The released recipe (300 epochs, no normalisation, recording-order
target, MMD 0.1) with `--protocol reported_diagnostic`. 6 directions × 3 seeds = **18 tasks**.
(P2, the fixed recipe with test-label selection, is dropped: test-label selection is already measured in
Part 2.)

## Part 2 — Unseen films within each culture

Same design as `clip_heldout_emotion.py`, now with SCVAE. Each culture's clips are dealt into 3 film folds
(each emotion's clips, sorted, round-robin), so every fold holds every emotion. For a test person and a test
fold F, the model trains on the **other people** only, twice, with the same amount of training data:

- **seen:** the other people's windows of folds F and F+1 (training includes the test films);
- **unseen:** the other people's windows of the two folds other than F (test films never seen).

Both are scored on the same test windows: the test person's windows of fold F. Recordings are standardised over
all their windows (label-free) before the fold is selected, as in `clip_heldout_emotion.py`. The unlabelled
target batches for MMD are the test person's fold-F windows.

**New options:** `--film-fold F` (0–2) and `--film-condition seen|unseen`, `--split pooled` only; suffix
`__films_<condition><F>`. Source-validation subjects (when used) keep the same film mask as the training source.

**Models** (each: 3 cultures × 3 folds × 2 conditions × 3 seeds = 54 tasks):

- **Released + BN fix, strict:** 300 epochs, `--bn-recalibration source`.
- **SCVAE-fixed, strict:** the configuration chosen in Part 1 with that culture as the target, with
  `--source-val-subjects K`.
- **Released, test-label-selected (P1):** 300 epochs, `--protocol reported_diagnostic`.

**162 tasks.** Submitted after the Part 1 grid (SCVAE-fixed needs the chosen configurations).

## Part 3 — Shared films, German ↔ French

No new training. From the saved probabilities of SCVAE-fixed (Part 1, chosen configuration, ensemble), accuracy
on German → French and French → German is split into the 5 shared films vs the other films. The shared films
are 3 negative and 2 positive, so the split is reported with balanced accuracy and per-emotion recall, and the
comparison is restricted to the negative and positive classes (neutral has no shared film).

## Reported

- **Part 1:** per direction: best simple model | SCVAE released strict | SCVAE-fixed strict (1 seed, 3-seed
  ensemble) || P1 test-label-selected. Accuracy and macro-F1, averaged per person; 95% bootstrap CIs; paired
  Wilcoxon SCVAE-fixed ensemble vs released over (direction, person) pairs. Peeking gain P1 − released strict.
  Grid factor means (exploratory).
- **Part 2:** per culture and model: seen, unseen, drop (seen − unseen), people dropping, Wilcoxon over people
  (each person averaged over folds and seeds). Plus the simple-model drop from `clip_heldout_emotion.md` for
  reference. The test-label-selected rows are labelled.
- **Part 3:** per direction: shared-film vs other-film balanced accuracy (negative and positive only), per-person
  paired difference, Wilcoxon.

## Success criterion and predictions

Pre-registered, written before any run:

- **Success (Part 1, strict):** SCVAE-fixed (3-seed ensemble, chosen as above) averages **above 57.6%** over the
  6 directions **and** beats the best simple model in at least 4 of 6.
- Predictions: SCVAE-fixed strict 58%; P1 across cultures 72%.
- Part 2 (average over the 3 cultures): seen → unseen drop ≈ 8 points for SCVAE strict (both models) and
  ≈ 15 points for the test-label-selected P1.
- Part 3: shared films ≈ 5 points above the other films.
- The user's target of 80–90% applies to the test-label-selected columns only; results are reported as they
  come out, and test-label-selected runs are not tuned further to reach a target.

## Compute

324 tasks in total: 144 (grid) + 18 (P1 across cultures) + 162 (films). Grid tasks: ≤ 100 epochs with one
source-validation pass per epoch. P1 tasks evaluate the test person after every update (300 epochs × ≤ 20
updates). Film tasks train on two thirds of the films. Estimate under 50 GPU-hours, as array jobs via
`cluster/train_array.sbatch` with manifests from `make_scvae_improve_manifest.py` (manifest lines override the
sbatch's `--protocol strict` and `--epochs`).

Order: grid + P1 across cultures (162 tasks) → choose configurations → film stage (162 tasks).

## Code

- `src/experiments/train_scvae_reference.py`: the options above; film folds; trials saved with probabilities.
- `src/experiments/select_crossculture_config.py`: importable `load` / `choose`.
- `src/experiments/make_scvae_improve_manifest.py`: `grid` and `films` stages.
- `src/experiments/summarize_scvae_improve.py`: Parts 1–3 → `results/tables/scvae_improve_summary.md`.
- Results under `results/scvae_improve/{grid,paper_protocol,films}/<target>/...`.

## Tests

- Defaults reproduce the current run names (lines of `cluster/paper_sweep.txt`).
- Source-validation subjects are never trained on, never the target, and come only from the source.
- Film folds: every fold holds every emotion; unseen training never contains a test-fold clip; seen and unseen
  training sets have the same number of clips; test windows are identical in both conditions.
- Strict: changing the target labels on disk changes no saved probability and no selected epoch.
- `--input-noise 0` draws no random numbers; weight decay reaches the optimiser.
- The ensemble is the mean of the seeds' saved probabilities.
- Configuration choice for target T never reads a result whose target is T.
- Part 3: the shared-film trial mapping matches each window's emotion label (as `film_identification.py`
  checks).

## Risks

- Holding out 2 of 7–8 German / French subjects removes about 25% of training data; if source-validation curves
  are flat, fall back to a fixed 30 epochs (decided from those curves, not target accuracy).
- Per-person normalisation also drives the simple models, so SCVAE-fixed may only draw level with them.
- In Part 2, each test fold has only 1–3 clips per emotion, so per-person numbers are noisy; the conclusions rest
  on the paired, per-culture averages.
- The test-label-selected numbers may stay well below 90% across cultures and on unseen films; they are reported
  as found.
