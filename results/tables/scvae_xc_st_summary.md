# SCVAE-XC + v3 self-training

Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final evaluation. p: Wilcoxon over people against step 4.

## 1. One training culture

| Train → Test | eye_1src (step 4) | eye_1src_st | p | runs |
|---|---:|---:|---:|---|
| Chinese → German | 57.5 | 57.8 | 0.945 | 3 seeds, 8/8 people |
| Chinese → French | 59.1 | 59.8 | 0.742 | 3 seeds, 8/8 people |
| German → Chinese | 57.0 | 55.8 | 0.391 | 3 seeds, 12/12 people |
| German → French | 60.7 | 61.2 | 0.742 | 3 seeds, 8/8 people |
| French → Chinese | 66.4 | 65.7 | 0.266 | 3 seeds, 12/12 people |
| French → German | 62.2 | 63.0 | 0.578 | 3 seeds, 8/8 people |
| **Average** | **60.5** | **60.6** |  | |

## 2. Two training cultures → the third

| Train → Test | eye_2src (step 4) | eye_2src_st | p | NormFusion v3 fair | runs |
|---|---:|---:|---:|---:|---|
| German + French → Chinese | 68.1 | 68.7 | 0.850 | 72.8 | 3 seeds, 12/12 people |
| Chinese + French → German | 67.3 | 65.5 | 0.023 | 68.6 | 3 seeds, 8/8 people |
| Chinese + German → French | 66.4 | 65.8 | 0.727 | 67.9 | 3 seeds, 8/8 people |
| **Average** | **67.3** | **66.7** |  | 69.8 | |

Predicted: eye_2src_st 68.8, eye_1src_st 61.5.

## Decision (pre-registered)

- eye_2src_st − eye_2src: -0.6, higher in 1 of 3 (needs ≥ +1.0 and ≥ 2 of 3).
- **Self-training: NO-GO.**
- One culture (no decision): +0.1, higher in 4 of 6.
