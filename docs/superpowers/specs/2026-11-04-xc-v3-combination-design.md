# SCVAE-XC + NormFusion v3 combination — design (4 November 2026)

*Written before any run.* SCVAE-XC (a VAE on EEG+eye) and NormFusion v3 (bagged logistic-regression/MLP experts with
self-training) are different model families; averaging their probabilities may help if their errors differ.

## Setting

Strict throughout; no target label is read before the final scoring, and nothing is tuned.

- **SCVAE-XC:** `eye_2src` (two cultures → the third) and `eye_1src` (one → one), the existing 3-seed probability
  files (step ④), averaged over seeds.
- **v3:** its fair model, re-generated with `fair_probas` (the function the published v3 runs use), seeds
  12/34/56/78/90, averaged over seeds. Check: per-seed accuracy must match the stored `fair.json` within 0.5 points;
  otherwise stop and report.
- **Combined:** the equal-weight mean of the two seed-averaged probabilities (weight fixed at 0.5, not tuned).
- Also reported: v3's 5-seed probability average alone. Seed averaging alone may explain a gain.

Metric: window accuracy % per test person, mean over people; Wilcoxon over people.

## Prediction

Two cultures: combined 71.0; v3 seed-average 70.0; SCVAE-XC 67.3. One culture: combined 66.0.

## Decision rule (two cultures decides)

**GO — combined becomes the headline fair number** if, averaged over the 3 targets, combined ≥ max(v3 seed-average,
SCVAE-XC) + 1.0 **and** combined beats both in ≥ 2 of 3 targets. Otherwise **NO-GO**: report v3 (69.8, its
published fair number) as the best fair model. The one-culture rows are reported without a decision.

Script: `src/experiments/run_xc_v3_combination.py` → `results/tables/xc_v3_combination_summary.md`.

## Result (4 November 2026, cluster job 7291)

**NO-GO** (pre-registered). Every v3 seed reproduced its stored fair accuracy within 0.5 points. Two cultures:
SCVAE-XC 67.3, v3 seed-average 70.0, combined 69.6 (−0.5 vs v3; beats both in 2 of 3 targets but the mean gain
rule fails). One culture: 60.5 / 65.4 / 63.3. Averaging in SCVAE-XC pulls v3 down where v3 is much stronger
(German/French → Chinese, −4 to −6), and helps only where the two are close (Chinese + French → German +1.6).
v3 seed-average (70.0, 5-seed probability mean of the published fair model) stays the best fair number; its
published per-seed mean is 69.8. Prediction (71.0) wrong. Summary: `results/tables/xc_v3_combination_summary.md`.
