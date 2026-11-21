# Cross-cultural calibration (with N calibration clips per person)

Accuracy %, mean over test people [95% bootstrap CI]; seeds (12, 34, 56) and 5 calibration pools averaged per person. Every N is scored on the same test clips: the 2 pool films per emotion are removed from testing for every N, including N = 0, so the no-calibration column differs slightly from the full-test v3 numbers. Calibration films are never test films. Calibrated numbers are not comparable with published results that use no calibration.

## 1. Cross-culture: honest model, then calibration

| Train → Test | No calibration | 1 clip / emotion | 2 clips / emotion | p (2 vs 0) |
|---|---:|---:|---:|---:|
| Chinese → German | 50.1 [44.6, 55.5] | 51.5 [45.0, 58.7] | 56.4 [50.4, 62.9] | 0.055 |
| Chinese → French | 59.9 [55.5, 64.1] | 58.3 [53.2, 63.1] | 61.6 [57.5, 66.7] | 0.945 |
| German → Chinese | 64.1 [59.1, 69.5] | 67.6 [60.7, 74.3] | 72.7 [68.0, 77.5] | 0.009 |
| German → French | 68.3 [63.1, 73.5] | 61.5 [58.4, 64.9] | 65.5 [62.9, 68.6] | 0.461 |
| French → Chinese | 76.8 [72.1, 81.2] | 71.7 [65.4, 77.9] | 76.0 [71.8, 80.6] | 0.850 |
| French → German | 68.2 [63.4, 72.3] | 59.9 [52.2, 68.3] | 64.8 [58.4, 71.4] | 0.148 |
| **Average (6 directions)** | **64.6** | **61.7** | **66.2** | |
| Predicted (pre-registered) | 65 | 74 | 78 | |

## 2. Same-culture reference (train on other people of the same culture)

| Test culture | No calibration | 1 clip / emotion | 2 clips / emotion | p (2 vs 0) |
|---|---:|---:|---:|---:|
| Chinese | 88.7 [85.9, 91.1] | 79.9 [73.0, 85.5] | 82.2 [77.5, 86.6] | 0.009 |
| German | 75.3 [68.8, 82.1] | 67.7 [61.4, 74.1] | 73.6 [69.0, 78.9] | 0.641 |
| French | 75.9 [70.5, 80.5] | 69.6 [64.8, 74.6] | 71.4 [67.7, 75.2] | 0.039 |
| **Average (3 cultures)** | **80.0** | **72.4** | **75.7** | |
| Predicted (pre-registered) | 77 | 81 | 83 | |

## 3. Culture gap (same-culture minus cross-culture, same test people; points)

Includes film familiarity: same-culture training people watched the test person's films. Upper bound.

| Test culture | No calibration | 1 clip / emotion | 2 clips / emotion |
|---|---:|---:|---:|
| Chinese | 18.2 | 10.2 | 7.8 |
| German | 16.2 | 12.0 | 13.0 |
| French | 11.8 | 9.7 | 7.9 |
| **Average** | **15.4** | **10.6** | **9.5** |
| Predicted (pre-registered) | 12 | 7 | 5 |

## 4. Film leak: calibrating on films that are also tested (leaky) vs film-disjoint

| Train → Test | 1 clip: disjoint | 1 clip: same film | 2 clips: disjoint | 2 clips: same film |
|---|---:|---:|---:|---:|
| Chinese → German | 51.5 | 51.1 | 56.4 | 57.1 |
| Chinese → French | 58.3 | 57.3 | 61.6 | 66.7 |
| German → Chinese | 67.6 | 67.8 | 72.7 | 75.4 |
| German → French | 61.5 | 59.5 | 65.5 | 68.2 |
| French → Chinese | 71.7 | 70.3 | 76.0 | 77.7 |
| French → German | 59.9 | 61.3 | 64.8 | 66.1 |
| **Inflation (same − disjoint)** | | **-0.5** | | **+2.3** |

## 5. Modality of the person model (cross-culture, accuracy %)

| Train → Test | N | combined (EEG + eye) | EEG only | eye only | person model alone |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 1 | 51.5 | 50.2 | 47.8 | 44.7 |
| Chinese → German | 2 | 56.4 | 53.2 | 56.1 | 51.9 |
| Chinese → French | 1 | 58.3 | 57.0 | 53.9 | 47.0 |
| Chinese → French | 2 | 61.6 | 59.9 | 58.8 | 52.9 |
| German → Chinese | 1 | 67.6 | 69.1 | 55.3 | 61.0 |
| German → Chinese | 2 | 72.7 | 72.5 | 62.9 | 66.9 |
| German → French | 1 | 61.5 | 60.9 | 55.7 | 47.0 |
| German → French | 2 | 65.5 | 63.5 | 61.7 | 52.9 |
| French → Chinese | 1 | 71.7 | 72.8 | 60.8 | 61.0 |
| French → Chinese | 2 | 76.0 | 75.6 | 66.8 | 66.9 |
| French → German | 1 | 59.9 | 58.3 | 57.4 | 44.7 |
| French → German | 2 | 64.8 | 61.4 | 64.3 | 51.9 |

## 6. Accuracy by session (cross-culture average; calibration comes from session 1)

| N | session 1 | session 2 | session 3 |
|---:|---:|---:|---:|
| 0 | 66.0 | 66.1 | 64.9 |
| 1 | 66.5 | 60.6 | 61.8 |
| 2 | 70.2 | 66.3 | 65.9 |

## 7. Follow-ups (exploratory, not pre-registered)

persession: a calibration film per emotion at the start of every session, one person model per session. learned: the weight on the person model chosen by leave-one-film-out on the calibration windows (2 films per emotion needed). Same test clips as above.

| Train → Test | N | combined (pre-registered) | persession | learned | persession + learned |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 1 | 51.5 | 52.1 | – | – |
| Chinese → German | 2 | 56.4 | 57.0 | 53.2 | 54.2 |
| Chinese → French | 1 | 58.3 | 57.6 | – | – |
| Chinese → French | 2 | 61.6 | 61.0 | 60.8 | 63.0 |
| German → Chinese | 1 | 67.6 | 69.3 | – | – |
| German → Chinese | 2 | 72.7 | 74.7 | 70.3 | 71.9 |
| German → French | 1 | 61.5 | 59.3 | – | – |
| German → French | 2 | 65.5 | 63.0 | 66.5 | 69.0 |
| French → Chinese | 1 | 71.7 | 74.3 | – | – |
| French → Chinese | 2 | 76.0 | 77.8 | 77.1 | 79.4 |
| French → German | 1 | 59.9 | 57.6 | – | – |
| French → German | 2 | 64.8 | 64.4 | 67.3 | 67.6 |
| Chinese (same culture) | 1 | 79.9 | 82.4 | – | – |
| Chinese (same culture) | 2 | 82.2 | 84.0 | 86.7 | 87.9 |
| German (same culture) | 1 | 67.7 | 64.1 | – | – |
| German (same culture) | 2 | 73.6 | 69.5 | 73.4 | 73.5 |
| French (same culture) | 1 | 69.6 | 66.1 | – | – |
| French (same culture) | 2 | 71.4 | 69.1 | 75.3 | 74.5 |
| **Cross-culture average** | 1 | **61.7** | **61.7** | – | – |
| **Cross-culture average** | 2 | **66.2** | **66.3** | **65.8** | **67.5** |
| **Same-culture average** | 1 | **72.4** | **70.9** | – | – |
| **Same-culture average** | 2 | **75.7** | **74.2** | **78.5** | **78.7** |

Figure: `results/figures/calibration_curve.png`.
