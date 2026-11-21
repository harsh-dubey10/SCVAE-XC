# SCVAE-XC D1: eeg+pupil input

Strict accuracy %, 3-seed ensemble, mean over test people. No target label read before the final evaluation. p: Wilcoxon over people, pupil vs eye. Pos/neg: positive vs negative windows only.

## 1. One training culture

| Train → Test | eye (step 4) | pupil | p | pos/neg eye | pos/neg pupil | runs (pupil) |
|---|---:|---:|---:|---:|---:|---|
| Chinese → German | 57.5 | 54.0 | 0.461 | 66.4 | 67.4 | 3 seeds, 8/8 people |
| Chinese → French | 59.1 | 54.5 | 0.078 | 67.3 | 68.8 | 3 seeds, 8/8 people |
| German → Chinese | 57.0 | 61.4 | 0.021 | 61.0 | 76.6 | 3 seeds, 12/12 people |
| German → French | 60.7 | 61.4 | 0.844 | 70.9 | 73.9 | 3 seeds, 8/8 people |
| French → Chinese | 66.4 | 66.4 | 0.569 | 80.3 | 85.3 | 3 seeds, 12/12 people |
| French → German | 62.2 | 63.2 | 1.000 | 73.4 | 74.8 | 3 seeds, 8/8 people |
| **Average** | **60.5** | **60.1** | | **69.9** | **74.5** | |

## 2. Two training cultures → the third

| Train → Test | eye (step 4) | pupil | p | pos/neg eye | pos/neg pupil | runs (pupil) |
|---|---:|---:|---:|---:|---:|---|
| German + French → Chinese | 68.1 | 65.9 | 0.531 | 75.9 | 84.0 | 3 seeds, 12/12 people |
| Chinese + French → German | 67.3 | 64.3 | 0.148 | 75.8 | 72.9 | 3 seeds, 8/8 people |
| Chinese + German → French | 66.4 | 63.0 | 0.195 | 77.2 | 75.0 | 3 seeds, 8/8 people |
| **Average** | **67.3** | **64.4** | | **76.3** | **77.3** | |

Predicted: pupil_2src 67.5, pupil_1src 61.0.

## Decision (pre-registered)

- pupil_2src − eye_2src: -2.9 (needs ≥ −0.5): not met.
- pupil_1src − eye_1src: -0.3 (needs ≥ −1.0): met.
- **D1: KEEP all 33 eye features.**
