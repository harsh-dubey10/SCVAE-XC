# SCVAE-XC Stage 2, step 3: emotion-aware alignment

Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final evaluation. p: Wilcoxon over people against step 4.

## 1. One training culture

| Train → Test | eye_1src (step 4) | eye_1src_ca | p | runs (ca) |
|---|---:|---:|---:|---|
| Chinese → German | 57.5 | 55.6 | 0.195 | 3 seeds, 8/8 people |
| Chinese → French | 59.1 | 57.8 | 0.383 | 3 seeds, 8/8 people |
| German → Chinese | 57.0 | 56.7 | 0.910 | 3 seeds, 12/12 people |
| German → French | 60.7 | 59.6 | 0.406 | 3 seeds, 8/8 people |
| French → Chinese | 66.4 | 67.0 | 0.424 | 3 seeds, 12/12 people |
| French → German | 62.2 | 61.1 | 0.641 | 3 seeds, 8/8 people |
| **Average** | **60.5** | **59.6** | | |

## 2. Two training cultures → the third

| Train → Test | eye_2src (step 4) | eye_2src_cmmd | eye_2src_ca | p (ca) | NormFusion v3 fair | runs (ca) |
|---|---:|---:|---:|---:|---:|---|
| German + French → Chinese | 68.1 | 67.9 | 69.7 | 0.266 | 72.8 | 3 seeds, 12/12 people |
| Chinese + French → German | 67.3 | 67.9 | 66.8 | 0.742 | 68.6 | 3 seeds, 8/8 people |
| Chinese + German → French | 66.4 | 66.4 | 66.6 | 0.688 | 67.9 | 3 seeds, 8/8 people |
| **Average** | **67.3** | **67.4** | **67.7** | | 69.8 | |

Predicted: eye_2src_cmmd 68, eye_2src_ca 69, eye_1src_ca 62.

## Go check (pre-registered)

- eye_2src_ca − eye_2src: +0.5 points, higher in 2 of 3 (needs ≥ +1.5 and ≥ 2 of 3): not met.
- eye_1src_ca − eye_1src: -0.9 points, higher in 1 of 6 (needs ≥ +1.5 and ≥ 4 of 6): not met.
- **Step 3: NO-GO.**
