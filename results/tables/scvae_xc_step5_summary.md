# SCVAE-XC Stage 2, step 5 and the test-label-selected column

Accuracy %, 3-seed ensemble, mean over test people. p: Wilcoxon over people against step 4.

## 1. Step 5: per-person test-time adaptation (strict)

| Train → Test | step 4 | tta_im | tta_ent | p (im) | NormFusion v3 fair | runs (im) |
|---|---:|---:|---:|---:|---:|---|
| Chinese → German | 57.5 | 57.8 | – | 0.672 |  | 3 seeds, 8/8 people |
| Chinese → French | 59.1 | 60.6 | – | 0.078 |  | 3 seeds, 8/8 people |
| German → Chinese | 57.0 | 56.2 | – | 0.311 |  | 3 seeds, 12/12 people |
| German → French | 60.7 | 60.4 | – | 0.844 |  | 3 seeds, 8/8 people |
| French → Chinese | 66.4 | 66.9 | – | 0.496 |  | 3 seeds, 12/12 people |
| French → German | 62.2 | 62.3 | – | 0.875 |  | 3 seeds, 8/8 people |
| **Average, one culture** | **60.5** | **60.7** | | | | |
| German + French → Chinese | 68.1 | 69.6 | 68.3 | 0.019 | 72.8 | 3 seeds, 12/12 people |
| Chinese + French → German | 67.3 | 66.9 | 67.2 | 0.547 | 68.6 | 3 seeds, 8/8 people |
| Chinese + German → French | 66.4 | 66.6 | 66.1 | 0.938 | 67.9 | 3 seeds, 8/8 people |
| **Average, two cultures** | **67.3** | **67.7** | | | 69.8 | |

Predicted: eye_2src_tta_im 68, eye_2src_tta_ent 67, eye_1src_tta_im 61.

Go check (pre-registered):

- eye_2src_tta_im − eye_2src: +0.5, higher in 2 of 3 (needs ≥ +1.5 and ≥ 2 of 3): not met.
- eye_1src_tta_im − eye_1src: +0.2, higher in 4 of 6 (needs ≥ +1.5 and ≥ 4 of 6): not met.
- **Step 5: NO-GO.**

## 2. Test-label-selected column (paper protocol — NOT accuracy)

Strict = epoch chosen on held-out source people (the reportable number). The other two columns choose the epoch by reading the test people's labels, as the published papers did; shown only to measure the inflation.

| Train → Test | step 4 strict (original runs) | re-run strict | one epoch for all | best epoch per person | inflation (per person) |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 57.5 | 57.5 | 61.6 | 63.3 | +5.8 |
| Chinese → French | 59.1 | 59.1 | 59.5 | 61.7 | +2.5 |
| German → Chinese | 57.0 | 57.0 | 58.5 | 62.6 | +5.6 |
| German → French | 60.7 | 60.7 | 63.6 | 65.8 | +5.1 |
| French → Chinese | 66.4 | 66.4 | 68.5 | 72.5 | +6.1 |
| French → German | 62.2 | 62.2 | 66.6 | 69.5 | +7.3 |
| **Average, one culture** | **60.5** | **60.5** | **63.0** | **65.9** | **+5.4** |
| German + French → Chinese | 68.1 | 68.1 | 69.7 | 74.9 | +6.8 |
| Chinese + French → German | 67.3 | 67.3 | 67.9 | 71.4 | +4.1 |
| Chinese + German → French | 66.4 | 66.4 | 67.6 | 70.5 | +4.1 |
| **Average, two cultures** | **67.3** | **67.3** | **68.4** | **72.3** | **+5.0** |

Predicted: eye_2src strict 67, one epoch 70, per person 76; eye_1src strict 60, one epoch 63, per person 70. Re-run check: re-run strict within ±1 point of the original runs.
