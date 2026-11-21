# NormFusion: our cross-culture EEG + eye model, design

*1 October 2026. Status: draft for review.*

## Goal

A simple model for cross-culture emotion recognition from EEG + eye
movements that beats the paper's DCCA-AM **under the same evaluation**:
our fair result against DCCA-AM's fair result, and our paper's-rules result
against DCCA-AM's paper's-rules result (both from
`docs/superpowers/specs/2026-10-01-dcca-am-fair-rerun-design.md`). Each part
of the model targets a problem found earlier in this project, and each is
tested by removing it.

**Success:** our fair accuracy is higher than DCCA-AM's fair accuracy on
average over the 6 cross-culture directions and in at least 4 of them, with
per-person 95% CIs and a paired Wilcoxon test per direction. The result is
reported whatever it is.

**Not in scope:** within-culture or within-person settings; deep sequence
models; new preprocessing.

## Honesty constraint (stated up front)

The design is informed by earlier analyses of these same datasets (for
example, that per-recording normalisation helps and that eye features drop
more across cultures). There is no untouched fourth dataset. To limit this:
the whole design, its candidate grid and its selection rule are fixed in this
document before any NormFusion run; all choices inside a run use the
training culture only; every component is reported with an ablation; and the
write-up states that the design was informed by prior analyses of the same
data.

## The model

### 1. Inputs and normalisation

The existing processed `<culture>_EEG_EYE` features (310 EEG values per
window, z-scored per window, plus 33 raw eye values), standardised per
recording with `normalize_recording` for both training and test people
(label-free; the same step as the normalised baselines). EEG = columns 0:310,
eye = 310:343.

### 2. Two modality experts

One expert for EEG and one for eye. Each expert is a small ensemble:

- logistic regression (standardiser + `LogisticRegression`, class-balanced),
  C chosen from {0.001, 0.01, 0.1, 1};
- the existing MLP (`src/baselines/mlp.py`, 30 epochs, Adam 1e-3), 3 seeds.

The expert's output is the average of these models' class probabilities
(logistic regression at its chosen C, and the 3 MLP seeds).

### 3. Late fusion with a learned modality weight

`p = w · p_EEG + (1 − w) · p_eye`, with w chosen from
{0, 0.25, 0.5, 0.75, 1} on the training culture. Eye and EEG transfer
differently across cultures, so the weight is chosen rather than fixed.

### 4. Film-robust selection (the key design choice)

C and w are chosen by cross-validation inside the training culture that holds
out **people and films together**, so the selection rewards features that
generalise to new films rather than recognise them (the confound measured in
E1):

- the training culture's clips are dealt into 3 folds, each holding every
  emotion (as in `clip_heldout_emotion.clip_folds`); its people into 3
  groups;
- for each (person group, film fold): train on the other people's windows
  from the other two film folds; validate on the held-out people's windows
  from the held-out film fold;
- score = mean balanced accuracy over the 9 validation cells; pick the best
  (C, w).

### 5. Unlabelled adaptation to the test culture (transductive, stated)

After fitting, class-balanced self-training on the test culture's unlabelled
windows, as in `strict_baselines.fit_mlp_selftrain`: 3 rounds; each round
adds, per emotion, the same number k of the most confident test windows
(fused probability ≥ 0.9; k = the smallest class count; stop if k < 10) with
their predicted labels, and refits both experts. Test labels are never read.
Like the normalisation, this uses unlabelled test data, and it is reported as
such. Settings are fixed in advance (the values used earlier), not tuned.

### 6. Output

Predicted emotion per window = argmax of the fused probabilities.

## Evaluation

- **Directions:** the 6 cross-culture directions (train on all people of one
  culture, test each person of another), and German ↔ French without the 5
  shared films in training ("new films"), as for DCCA-AM.
- **Seeds:** 12, 34, 56 (the MLP seeds and the self-training order).
- **Fair:** the procedure above, choices from the training culture only.
- **Paper's rules (for the matched comparison):** every candidate (C, w,
  with and without self-training) is scored on the test person and the best
  is kept per person, as the paper's code does. The per-candidate test
  predictions are saved so this is computed without retraining.
- **Ablations (fair, each removes one part):** no normalisation; early
  fusion (concatenated features, one ensemble) instead of late fusion;
  person-only cross-validation instead of person + film; no self-training;
  logistic regression only (no MLP ensemble); EEG only (w = 1); eye only
  (w = 0).
- **Reported per direction:** accuracy and macro-F1, mean over test people
  with 95% bootstrap CIs; paired Wilcoxon against DCCA-AM fair; next to the
  paper's reported DCCA-AM number and the best simple normalised baseline.

## Code (to be planned in detail)

- `src/baselines/norm_fusion.py`: experts, fusion, film-robust selection,
  self-training.
- `src/experiments/run_norm_fusion.py`: runs directions × seeds × variants
  (fair, ablations), saves per-person results and per-candidate test
  predictions to `results/norm_fusion/<target>/<variant>__from_<source>/seed_<s>/`.
- `src/experiments/summarize_norm_fusion.py`: tables, ablations, and the
  comparison with `results/dcca_am` (reusing `summarize_dcca_am` loaders).
- Tests (plain-assert modules): fusion weight bounds and probability sums;
  the film-robust folds never put a validation film or person in training;
  self-training never reads test labels (labels replaced by a sentinel);
  fair selection uses only training-culture scores.

## Compute

Laptop CPU. Estimate: about 1–2 hours for 6 directions × 3 seeds × 8
variants (logistic regression and a 30-epoch MLP on ≤ 30k windows). No
cluster needed.

## Risks

- **It may not beat DCCA-AM.** Then that is the result: reported with the
  ablations, which still show what matters.
- **Self-training can drift** when the first predictions are poor (it added
  about 0 across cultures before); the ablation shows whether it helps.
- **Design informed by the same data** (see the honesty constraint); stated
  in the write-up.
