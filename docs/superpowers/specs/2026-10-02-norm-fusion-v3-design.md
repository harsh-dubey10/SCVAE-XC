# NormFusion v3: design

*2 October 2026. Status: draft for review.*

## Goal

Raise fair cross-culture accuracy over NormFusion v1 (63.1% average over the 6 one-to-one directions) by
removing the step where v1 and v2 lose accuracy: **selection**. Inside one training culture, the
cross-validation picks a fusion candidate that does not transfer to a new culture (v2: per-emotion weights
fit the training culture and fall 4–8 points with German as the training culture; the candidate grid
contains options worth about 77% for German → Chinese that fair selection cannot find).

v3 replaces selection by **averaging** (a fixed ensemble of bagged experts) and, where two training cultures
exist, makes its one remaining choice by **cross-culture validation**. Training runs on the cluster.

In plain words: v3 averages several simple models, each trained many times on different subsets of people,
so no single bad choice can sink it; when it trains on two cultures, it picks its settings by checking how a
model trained on one culture does on the other.

## Honesty: pre-registration

All three cultures' test results have been looked at twice (v1, v2), and no untouched EEG + eye data exists
(eye features exist only for 4 s windows). v3 is therefore a **post-hoc model**. Its credibility comes from
fixing everything below — the model, every grid, the selection rules, the seeds, the variants, and a numeric
prediction — in this document before any v3 run, and reporting every result whatever it is. v1 stays the
primary, pre-specified model. The write-up states this.

## Model

Everything not changed here is as in v1 (`src/baselines/norm_fusion.py`): 4 s windows, 343 features
standardised per recording (EEG = columns 0:310, eye = 310:343), labels negative / neutral / positive,
logistic regression = standardiser + `LogisticRegression(C, class_weight="balanced", max_iter=2000)`,
MLP = `strict_baselines.fit_mlp(x, y, 30, "cpu", seed)`. No temporal smoothing (v2: never chosen; the
features are already smooth in time).

### Experts and bags

Three experts: **EEG** (310 features), **eye** (33), **early** (all 343).

Each expert is **10 bags**. Bag b is trained on a random 80% of the training people
(`round(0.8 × n)` people, without replacement; in two-culture training, 80% of each culture's people,
rounded, so both cultures stay in every bag), drawn with `numpy.random.default_rng(seed * 1000 + b)`.
A bag is LR(C) plus 3 MLPs with seeds `seed + 100 * (b + 1) + i`, i = 0, 1, 2; the bag's probability is the
mean of the four. The expert's probability is the mean over its 10 bags.

**C per expert** ∈ {0.001, 0.01, 0.1, 1}, chosen separately for each expert by v1's people-and-films
cross-validation (3 person groups × 3 film folds) inside the training data, scoring that expert alone
(unbagged: LR(C) + 3 MLPs with seeds seed, seed + 1, seed + 2, as in v1) by mean balanced accuracy over the
9 cells; ties to the smaller C.

### Ensemble

The ensemble's probability is the plain mean of its members' probabilities (no learned weights). Membership
options, in this fixed order: {EEG}, {EEG, eye}, {EEG, early}, {EEG, eye, early}. {EEG, eye} is late fusion
with w = 0.5.

### Self-training

v1's rule on the ensemble's probabilities: 3 rounds; per class the windows with confidence ≥ 0.9; k = the
smallest class count; stop if k < 10; add the k most confident windows of each class with their predicted
labels and refit. A refit refits every bag of every member on its own training people plus **all** chosen
target windows, with the same C and the same people sample. Test labels are never passed in.

## What is chosen, and how

- **One-to-one directions and new films (one training culture):** membership fixed at {EEG, eye, early};
  only each expert's C is chosen (above).
- **Two-culture settings (train on cultures A + B, test on T):** C per expert is chosen on the pooled A + B
  data (person groups dealt across both cultures, films identified by (culture, clip), as in v2). The
  **membership** is chosen by cross-culture validation, never touching T: for the fold A → B, choose C per
  expert inside A, fit the bagged experts on A, predict B, and score each membership by balanced accuracy on
  B; likewise B → A; the membership with the best mean over the two folds wins, ties to the first in the
  fixed order. Validation runs without self-training.

## Evaluation

- **Settings:** the 6 one-to-one directions; the 3 two-culture settings (Chinese + German → French,
  Chinese + French → German, German + French → Chinese); new films (German ↔ French, the 5 shared films
  removed from training). 11 settings × 5 seeds (12, 34, 56, 78, 90) = 55 runs.
- **Variants (fair):**
  - `fair` — v3 as above.
  - `no_bagging` — each expert one bag on all training people (v1's expert), same C and membership as fair.
  - `no_selftrain` — fair's ensemble without self-training.
  - `members_eeg`, `members_eeg_eye`, `members_eeg_early`, `members_all` — each fixed membership,
    self-trained, same C (all settings; in one-to-one directions `members_all` equals `fair`).
  - `v1` — v1 itself (`run_norm_fusion.run_direction`, variant `fair`), on the same machine as v3, for a
    matched comparison in every setting.
- **Paper's rules:** per test person, the best of the 4 memberships without self-training, or the fair
  model; seeds averaged.
- **Reported:** window-level accuracy and macro-F1 per setting, mean over test people with 95% bootstrap
  CIs, seeds averaged per person; paired Wilcoxon v3 fair vs v1 (same machine) per setting; v3 next to v1,
  v2 and DCCA-AM fair (and paper's rules) where those exist; the chosen membership and Cs per run; the
  pre-registered prediction next to the outcome.
- **v1 check:** the v1 variant run on the cluster is compared with the saved laptop v1 results
  (`results/norm_fusion`, seeds 12, 34, 56). The cluster's CPUs (x86) round differently from the laptop's
  (ARM), so small differences are expected and reported, not required to be zero; a large difference
  (more than 2 points in a direction's mean) stops the write-up until explained. Exact v1 equivalence of the
  code is checked by the unit tests on one machine.

## Pre-registered prediction

Written before any v3 run:

| Train → Test | v1 | v3 predicted |
|---|---:|---:|
| Chinese → German | 51.7 | 54 |
| Chinese → French | 61.5 | 60 |
| German → Chinese | 52.9 | 64 |
| German → French | 64.6 | 65 |
| French → Chinese | 75.9 | 76 |
| French → German | 72.1 | 70 |
| **Average** | **63.1** | **65 (range 63–68)** |
| Two cultures → the third | 68–74 | 70–75 |

**Success criterion:** v3 fair is higher than v1 on the 6-direction average **and** in at least 4 of the 6
directions (v1 from the same machine). Missing it is reported as the result.

## Compute

Cluster (IITGN Singularity): partition `gpu`; per user at most 32 CPUs and 2 GPUs, so at most 2 jobs at a
time, each must request `--gres=gpu:1`. A Slurm array of 11 tasks, one per setting; each task requests 16
CPUs and runs its 5 seeds one after another. Within a seed, the independent fits of each stage (bags ×
experts, CV cells) run in parallel over the task's CPUs, each fit single-threaded and seeded on its own, so
results do not depend on the number of workers. All computation is on CPU (the GPU is requested only
because the queue requires it), so v3 and its v1 reference share one arithmetic. Estimate: about 85
CPU-hours, 3–5 hours wall time; one run is timed on the laptop first and the estimate updated before
submitting.

## Code

- `src/baselines/norm_fusion_v3.py`: people samples for bags, bagged experts, C choice per expert,
  ensemble, cross-culture membership choice, self-training on the ensemble. Reuses v1's LR, MLP, CV cells
  and loading, and v2's `combine` for two-culture data.
- `src/experiments/run_norm_fusion_v3.py`: runs one setting's seeds (or one seed) and writes
  `results/norm_fusion_v3/<target>/<setting>__from_<source>/seed_<s>/<variant>.json`; `--task N` selects
  the setting for the array.
- `src/experiments/summarize_norm_fusion_v3.py`: the tables above, the v1 check, the prediction next to
  the outcome → `results/tables/norm_fusion_v3_summary.md`.
- `cluster/norm_fusion_v3.sbatch`: the 11-task array.

## Tests (plain-assert modules, float32 features like the real data)

- Bag people samples: deterministic per seed, `round(0.8 n)` people (per culture in two-culture training),
  different between bags, only training people.
- A bag is fitted only on its own people; with 1 bag on all people an expert equals v1's expert exactly.
- {EEG, eye} with one bag on all people equals v1's late fusion with w = 0.5 exactly.
- Cross-culture validation never receives the test culture (its function has no target argument), and the
  chosen membership is unchanged when the test culture's data is replaced.
- With the test labels shuffled, every choice, every added pseudo-labelled window and every per-window
  prediction is unchanged (predictions compared window by window).
- Results do not depend on the number of parallel workers (1 vs 4 give identical outputs).
- The `v1` variant equals v1's `run_direction` output.

## Risks

- **Post-hoc optimism** (see pre-registration).
- **Averaging can be pulled down by a weak member** (e.g. eye-only 40.8% for German → Chinese); the
  `members_*` variants show how much.
- **Cross-culture validation has only two folds** and 4 options; it can still pick wrongly.
- **Cross-hardware differences** make the cluster v1 differ slightly from the laptop v1; v3 is compared only
  with the v1 from its own machine.
