# NormFusion v2: design

*1 October 2026. Status: draft for review.*

## Goal

Raise cross-culture accuracy over NormFusion v1 (fair average 63.1%) while
keeping the same fair rules: every choice inside a run uses the training
culture(s) only, and test labels are read once, to score. v2 adds three
things, each tested by removing it:

1. **Temporal smoothing** of the predictions within a recording.
2. **Per-emotion modality weights** (EEG and eye can count differently for
   each emotion), with early fusion as an alternative the selection may pick.
3. **Training on two cultures**, testing on the third, as an additional
   setting.

**What success means:** v2 is reported against v1 and against DCCA-AM (fair
and paper's rules) in every setting, whatever the outcome. No accuracy
target is set in advance. Expected range, stated before running: about
68–72% average over the 6 one-to-one directions, higher in the best
direction and in the two-culture setting.

## Honesty constraint (stronger than for v1)

v2's ideas come from looking at v1's results and at an analysis of emotion
signatures that used the labels of **all three cultures**, including the
ones v2 is tested on (EEG separates positive from negative similarly across
cultures; eye mostly separates neutral; eye's positive/negative pattern
differs between cultures). So v2 is a **post-hoc** model and its numbers are
more optimistic than v1's. To limit this:

- v2 does not hard-code any of those findings. Per-emotion weights, the
  fusion type and the smoothing length are chosen by the same film-robust
  cross-validation inside the training culture(s) as in v1, from grids fixed
  here.
- v1 stays the primary, pre-specified result. v2 is reported as a second,
  post-hoc model, with this paragraph in the write-up.

## Model

Everything in v1 is kept unless changed here: per-recording normalisation;
EEG expert and eye expert (logistic regression, C ∈ {0.001, 0.01, 0.1, 1},
+ 3 MLPs); class-balanced self-training (3 rounds, threshold 0.9, stop if
k < 10) on the unlabelled test windows.

### 1. Temporal smoothing

Within each recording, in presentation order, each window's class
probabilities are replaced by the mean over a centred window of length
k ∈ {1, 3, 5, 9, 15} (k = 1 is no smoothing; edges use the windows that
exist). Only the time order of windows is used, never the clip boundaries or
labels, so smoothing may run across the boundary between two clips, as it
would in a real recording. The analysis is offline, so centred (not causal)
smoothing is allowed; a causal variant (trailing window) is reported as an
ablation.

Self-training picks its pseudo-labels from the smoothed probabilities.

### 2. Fusion choice

Candidates, all scored by the same cross-validation:

- **Per-emotion late fusion:** p_c ∝ w_c · p_EEG,c + (1 − w_c) · p_eye,c for
  each emotion c, with w_c ∈ {0, 0.5, 1} independently for negative, neutral
  and positive (27 combinations), renormalised to sum to 1.
- **Early fusion:** one expert on the concatenated 343 features.

So the selection may end up at v1's single weight (equal w_c), at
per-emotion weights, or at early fusion.

### 3. Selection

As in v1: 3 person groups × 3 film folds inside the training data; train on
the other groups' windows from the other folds; validate on the held-out
group's windows from the held-out fold; mean balanced accuracy over the 9
cells; ties to the first candidate in grid order. For smoothing, the model
predicts every window of the held-out people's recordings, the predictions
are smoothed within each recording, and only the held-out fold's windows are
scored. Jointly chosen: (C_EEG, C_eye, fusion, k).

### 4. Two-culture training

Train on two cultures, test on the third (3 settings: Chinese + German →
French, Chinese + French → German, German + French → Chinese). Each culture's
recordings are normalised on their own as before. For the cross-validation,
person groups are dealt across both cultures, and films are identified by
(culture, clip), so film folds hold out films from both training cultures.

## Evaluation

- **Settings:** the 6 one-to-one directions, the 3 two-culture settings, and
  German ↔ French without the 5 shared films in training (new films).
- **Seeds:** 12, 34, 56.
- **Fair:** as above.
- **Paper's rules:** the best candidate per test person, over all
  (C_EEG, C_eye, fusion, k) without self-training, plus the fair choice with
  it.
- **Ablations (fair):** no smoothing (k = 1); causal smoothing; v1 fusion
  (single weight w, no per-emotion weights, no early-fusion option); no
  self-training; and **v1** itself (k = 1, single weight, no early-fusion
  option), run in every setting including the two-culture ones, so v2 is always
  compared with v1 on the same data. In the one-to-one directions this v1 run
  must reproduce the saved v1 results, as a check.
- **Window-level accuracy is the headline,** as in the paper. A **clip-level**
  accuracy (majority vote of the windows of each clip, which uses the known
  clip boundaries at test time) is reported separately and clearly labelled;
  it is not compared with any window-level number.
- **Reported:** accuracy and macro-F1 per setting, mean over test people with
  95% bootstrap CIs; paired Wilcoxon v2 vs v1 per one-to-one direction; next to
  DCCA-AM fair and paper's rules where those runs are complete.

## Code (to be planned in detail)

- `src/baselines/norm_fusion_v2.py`: smoothing, per-emotion fusion, the
  extended candidate set and selection, two-culture data; reuses v1's experts,
  CV cells and self-training.
- `src/experiments/run_norm_fusion_v2.py`, `summarize_norm_fusion_v2.py`:
  as for v1, writing to `results/norm_fusion_v2/`.
- Tests (plain-assert modules): smoothing uses only order within one recording
  and k = 1 is the identity; per-emotion weights with all w_c equal reproduce
  v1's fusion; two-culture film folds never put a (culture, clip) in both
  training and validation; label-shuffle invariance of every choice (as for
  v1); "v2 minus all changes" equals v1 on a small case.

## Compute

Laptop CPU. The extra candidates are cheap (smoothing and per-emotion weights
reuse the cached expert probabilities); the two-culture settings train on
roughly twice the data. Estimate: about 2–3 hours with 4 parallel workers.

## Risks

- **Post-hoc optimism** (see the honesty constraint).
- **Smoothing across clip boundaries** blurs the start and end of each clip;
  the selection may pick a short k for that reason.
- **The two-culture setting is not the paper's setting** (they train on one
  culture), so it is compared only with v1 in the same setting, not with
  DCCA-AM's numbers.
