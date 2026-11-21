# SCVAE-XC Stage 2, step 4: eye branch and two training cultures

Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final evaluation. p: Wilcoxon over people against the reference column.

## 1. One training culture

| Train → Test | SCVAE-fixed (2026-10-09) | eeg_1src (re-implementation) | eye_1src (+ eye) | p (eye vs fixed) | runs (eye) |
|---|---:|---:|---:|---:|---|
| Chinese → German | 47.5 | 45.9 | 57.5 | 0.016 | 3 seeds, 8/8 people |
| Chinese → French | 48.6 | 50.1 | 59.1 | 0.008 | 3 seeds, 8/8 people |
| German → Chinese | 58.5 | 57.3 | 57.0 | 0.791 | 3 seeds, 12/12 people |
| German → French | 54.5 | 57.5 | 60.7 | 0.109 | 3 seeds, 8/8 people |
| French → Chinese | 59.9 | 60.7 | 66.4 | 0.064 | 3 seeds, 12/12 people |
| French → German | 50.6 | 52.1 | 62.2 | 0.008 | 3 seeds, 8/8 people |
| **Average** | **53.3** | **53.9** | **60.5** | | |

Predicted: eeg_1src 53, eye_1src 57. eye_1src higher than SCVAE-fixed in 5 of 6 directions.

## 2. Two training cultures → the third

| Train → Test | Reference (SCVAE-fixed, mean of the two one-culture directions) | eeg_2src | eye_2src | p (eye_2src vs ref) | NormFusion v3 fair (context) | runs (eye) |
|---|---:|---:|---:|---:|---:|---|
| German + French → Chinese | 59.2 | 58.9 | 68.1 | 0.009 | 72.8 | 3 seeds, 12/12 people |
| Chinese + French → German | 49.1 | 51.8 | 67.3 | 0.008 | 68.6 | 3 seeds, 8/8 people |
| Chinese + German → French | 51.5 | 54.1 | 66.4 | 0.008 | 67.9 | 3 seeds, 8/8 people |
| **Average** | **53.3** | **54.9** | **67.3** | | 69.8 | |

Predicted: eeg_2src 57, eye_2src 61.

## Go check (pre-registered)

- eye_2src − reference: 14.0 points, higher in 3 of 3 (needs ≥ +5 and 3 of 3): met.
- eye_1src − SCVAE-fixed: 7.2 points, higher in 5 of 6 (needs ≥ +2 and ≥ 4): met.
- **Step 4: GO.**
