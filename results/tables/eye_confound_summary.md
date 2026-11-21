# Eye-feature confound check

Two training cultures → the third, every test film unseen; logistic regression (C = 0.01); accuracy %, mean over test people. Pupil = eye columns 0–11 (inferred from the data, not documented). Design: docs/superpowers/specs/2026-10-29-eye-confound-check-design.md.

## 1. Three classes

| Target | eeg | eeg+pupil | eeg+movement | eeg+eye |
|---|---:|---:|---:|---:|
| Chinese | 62.5 | 66.1 | 62.8 | 67.7 |
| German | 49.1 | 60.1 | 51.0 | 57.5 |
| French | 54.3 | 60.9 | 53.4 | 61.5 |
| **Mean** | **55.3** | **62.4** | **55.7** | **62.2** |

## 2. Positive vs negative

| Target | eeg | eeg+pupil | eeg+movement | eeg+eye |
|---|---:|---:|---:|---:|
| Chinese | 80.5 | 83.8 | 72.0 | 76.8 |
| German | 66.7 | 68.1 | 66.5 | 67.8 |
| French | 72.2 | 71.5 | 69.2 | 72.9 |
| **Mean** | **73.1** | **74.5** | **69.2** | **72.5** |

## 3. Film identification within a culture (person-held-out)

| Culture | chance | pupil | movement | eeg |
|---|---:|---:|---:|---:|
| Chinese | 6.7 | 36.6 | 33.4 | 48.1 |
| German | 5.6 | 43.2 | 55.8 | 30.8 |
| French | 4.8 | 31.1 | 35.4 | 23.3 |
| **Mean** | | **36.9** | **41.5** | **34.1** |

## Pre-registered decision

- 3-class: eye gain +6.9; pupil gain +7.1 (102% of it); movement gain +0.4.
- Positive vs negative: pupil gain +1.4; movement gain -3.9.
- Film identification: pupil 36.9 vs movement 41.5 vs EEG 34.1.
- **INCONCLUSIVE.**
