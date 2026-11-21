# Cross-cultural calibration: design

*4 October 2026. Status: draft for review.*

## Goal

Measure how much a few minutes of labelled data from a new person closes the cross-cultural gap, with an
evaluation that cannot be inflated by film recognition. The results are presented after the honest model:

1. **Honest model, no calibration** — NormFusion v3 fair, as already run (65.1% average one-to-one).
2. **The same model plus calibration** — N labelled clips per emotion from the new person.
3. **Same-culture reference** — the same pipeline trained on other people of the test person's own culture,
   with and without calibration, so the "culture gap" can be read off at each N.

All three are scored on **the same test clips** of each person, so the comparison is direct.

## Questions (the novelty)

- **Q1 — How much does culture cost, in calibration?** Accuracy against N for cross-culture and same-culture
  training. If the gap shrinks to a few points after one clip per emotion, the cross-cultural gap is mostly a
  per-person offset rather than a deep cultural difference.
- **Q2 — Does naive calibration leak through film recognition?** Calibrating on the same films that are later
  tested lets a model recognise the film instead of the emotion. We compare film-disjoint calibration (ours)
  with same-film calibration and report the difference.
- **Q3 — Which modality needs calibration?** The person model on EEG only, eye only, and both.

Calibration on SEED itself is studied (few-shot fine-tuning, meta-learning); no calibration study exists on the
cross-cultural Chinese / German / French EEG + eye data. The novelty is the cross-cultural measurement and the
film-disjoint protocol, not a new network.

## Calibration protocol

- **Clips per emotion N ∈ {0, 1, 2}.** For each test person and draw, a **calibration pool** of 2 film ids per
  emotion (6 films) is drawn at random from the films that person watched
  (`numpy.random.default_rng(1000 * draw + person)`); calibration with N uses the first N of each emotion's pool.
- **Calibration windows** come from the person's **first session** only (as at the start of a real session),
  all windows of the chosen films.
- **Test windows:** every window of the person, in all sessions, whose film id is **not** in the pool (the pool
  is excluded for every N, including N = 0, so all N are scored on identical clips). Per emotion this leaves
  3 films for China (5 per emotion), 4 for Germany (6), 5 for France (7).
- **Draws:** 5 random pools per person (draw = 0..4); results are averaged over draws, then over people.
- **Same-film variant (Q2 only):** calibration windows taken from the person's first session of films that
  **stay** in the test set (test = all sessions except session 1's calibration windows). This is the leaky
  protocol, run only to measure its inflation.

## Model

- **Base model:** NormFusion v3 fair (bagged EEG / eye / early experts, C per expert by film-robust CV,
  self-training on the unlabelled test culture), re-run with its per-window probabilities saved. Seeds 12, 34,
  56. The base model never sees calibration labels; its unlabelled self-training data are the whole target
  culture, as before.
- **Person model:** logistic regression (standardiser + `LogisticRegression(C=0.1, class_weight="balanced",
  max_iter=2000)`) trained on the person's calibration windows only, on all 343 features (Q3 variants: EEG
  310, eye 33). C = 0.1 is fixed in advance; no choice uses test labels.
- **Calibrated prediction:** the mean of the base model's and the person model's probabilities
  (weight 0.5, fixed in advance). For N = 0 the calibrated prediction is the base model.
- **Variants (reported):** `combined` (the above, primary), `person_only`, `base_only` (= N = 0),
  `combined_eeg`, `combined_eye`, `combined_samefilm` (Q2).

## Settings

- **Cross-culture:** the 6 one-to-one directions (train on one culture, test on another).
- **Same-culture reference:** for each culture, people are dealt into 3 groups (sorted ids, round-robin);
  for each group the base model is trained on the other two groups of the same culture and tested on that
  group (same v3 pipeline, film-robust CV inside the training people). Every person is a test person once.
- Two-culture settings and new films are not part of this study.

## Reported

- Per direction and for the same-culture reference: accuracy and macro-F1 at N = 0, 1, 2 (mean over people,
  95% bootstrap CIs, seeds and draws averaged per person); paired Wilcoxon N = 1 vs 0 and 2 vs 0.
- **Culture gap** at each N: same-culture minus cross-culture accuracy for the same test people (test
  culture fixed, averaged over the two source cultures).
- Q2: combined same-film minus combined film-disjoint, at N = 1 and 2.
- Q3: combined_eeg and combined_eye next to combined.
- The presentation order: honest table (N = 0) → calibrated table (N = 1, 2) → culture-gap figure (accuracy
  against N, cross vs same culture) → film-leak bar.
- Calibrated numbers are labelled "with N clips of calibration per person" everywhere and are never placed
  in the same column as published results without calibration.

## Pre-registered prediction

Written before any run (cross-culture average over the 6 directions; same-culture average over 3 cultures):

| | N = 0 | N = 1 | N = 2 |
|---|---:|---:|---:|
| Cross-culture | 65 | 74 | 78 |
| Same-culture | 77 | 81 | 83 |
| Culture gap | 12 | 7 | 5 |
| Same-film inflation (Q2) | — | +4 | +5 |

## Compute

- Cross-culture base: v3 fair only (no ablations), 6 directions × 3 seeds = 18 runs with saved probabilities.
- Same-culture base: 3 cultures × 3 groups × 3 seeds = 27 runs (smaller training sets).
- Person models are tiny (seconds each). Estimate: about 25 CPU-hours, on the cluster (11-task style array,
  `--gres=gpu:1`, 16 CPUs) or overnight on the laptop.

## Code

- `src/experiments/run_norm_fusion_v3.py`: add `--save-proba` (writes the fair per-window probabilities, with
  window ids, to an `.npz` next to the JSON) and a same-culture mode.
- `src/experiments/run_calibration.py`: builds pools, trains person models, scores every N, draw and variant
  from saved base probabilities; writes `results/calibration/...`.
- `src/experiments/summarize_calibration.py`: the tables and figure above → `results/tables/calibration_summary.md`
  and `results/figures/calibration_curve.png`.

## Tests

- Calibration and test windows never share a film id (film-disjoint variants); the same-film variant does,
  by construction, and only there.
- Every N of a person is scored on exactly the same test windows.
- Pools are deterministic per (draw, person) and contain 2 films per emotion.
- The person model is trained only on that person's calibration windows; no test-window label is read except
  to score (labels shuffled on test windows leave all predictions unchanged).
- N = 0 `combined` equals `base_only`.
- Same-culture groups never put a person in both training and test.

## Risks

- **Fewer test clips** per person (pool removed): wider intervals; draws reduce the variance of the pool choice.
- **Session effects:** calibration from session 1 and testing on sessions 1–3 mixes within- and cross-session
  transfer; the per-session breakdown is reported.
- **Fixed C and weight** may be sub-optimal; they are not tuned, by design.
- **Film familiarity in the same-culture reference:** the test person's films were also watched by the
  training people of the same culture (German and French also share 5 films across cultures; Chinese films are
  not shared). The culture gap therefore includes film familiarity and is reported as an upper bound.
