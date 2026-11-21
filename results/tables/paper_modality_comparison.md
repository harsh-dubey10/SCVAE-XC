# Eye and EEG + eye: ours (strict) vs Liu et al. 2022

Accuracy %, mean over test people [95% bootstrap CI]. ↑ / ↓: the paper's number lies below /
above our CI; ≈: inside it. The paper's settings were partly chosen with test labels.

## Eye, linear SVM

| Train → Test | Paper | Ours, not normalised | Ours, normalised |
|---|---:|---:|---:|
| Chinese → Chinese | 69.9 | 56.9 [50.8, 62.6] ↓ | 71.8 [67.3, 76.3] ≈ |
| German → German | 77.9 | 69.2 [60.7, 75.4] ↓ | 81.5 [75.6, 86.8] ≈ |
| French → French | 71.8 | 71.0 [66.8, 75.4] ≈ | 77.0 [73.1, 80.9] ↑ |
| Chinese → German | 51.8 | 35.4 [30.9, 39.4] ↓ | 47.0 [42.6, 51.2] ↓ |
| Chinese → French | 57.6 | 37.9 [31.1, 43.0] ↓ | 54.4 [47.5, 60.1] ≈ |
| German → Chinese | 51.4 | 38.8 [34.1, 43.8] ↓ | 46.8 [41.3, 52.0] ≈ |
| German → French | 65.4 | 48.5 [43.9, 52.9] ↓ | 62.1 [57.9, 66.3] ≈ |
| French → Chinese | 51.3 | 43.6 [36.1, 51.3] ↓ | 54.9 [47.6, 61.6] ≈ |
| French → German | 59.7 | 47.4 [44.2, 50.7] ↓ | 68.8 [62.8, 73.9] ↑ |

## Eye, MLP vs DNN

| Train → Test | Paper | Ours, not normalised | Ours, normalised |
|---|---:|---:|---:|
| Chinese → Chinese | 80.3 | 62.4 [57.4, 67.8] ↓ | 74.5 [68.5, 80.6] ≈ |
| German → German | 84.3 | 69.2 [63.8, 74.4] ↓ | 83.7 [78.5, 88.4] ≈ |
| French → French | 79.8 | 69.7 [64.1, 75.4] ↓ | 80.7 [75.2, 85.6] ≈ |
| Chinese → German | 62.4 | 33.5 [29.3, 37.9] ↓ | 50.6 [44.7, 56.8] ↓ |
| Chinese → French | 62.4 | 42.0 [36.2, 47.5] ↓ | 51.5 [43.8, 58.3] ↓ |
| German → Chinese | 56.7 | 43.4 [38.6, 48.0] ↓ | 47.2 [43.8, 50.8] ↓ |
| German → French | 67.1 | 48.7 [46.1, 51.4] ↓ | 56.7 [52.2, 61.6] ↓ |
| French → Chinese | 63.0 | 40.2 [35.1, 45.3] ↓ | 52.9 [45.1, 60.0] ↓ |
| French → German | 65.4 | 48.9 [45.2, 52.4] ↓ | 64.8 [58.9, 69.8] ≈ |

## EEG + eye, linear SVM vs Concat

| Train → Test | Paper | Ours, not normalised | Ours, normalised |
|---|---:|---:|---:|
| Chinese → Chinese | 83.0 | 61.0 [55.6, 66.2] ↓ | 84.4 [79.2, 88.6] ≈ |
| German → German | 77.3 | 55.0 [47.3, 63.1] ↓ | 79.6 [74.0, 86.1] ≈ |
| French → French | 71.9 | 51.6 [44.5, 58.5] ↓ | 71.9 [65.1, 78.3] ≈ |
| Chinese → German | 58.2 | 41.6 [35.7, 47.1] ↓ | 54.1 [49.2, 59.6] ≈ |
| Chinese → French | 63.9 | 42.9 [38.4, 47.7] ↓ | 58.3 [54.4, 63.0] ↓ |
| German → Chinese | 52.6 | 47.2 [41.8, 53.4] ≈ | 61.2 [54.6, 67.5] ↑ |
| German → French | 63.6 | 43.3 [36.3, 51.3] ↓ | 65.4 [59.8, 71.2] ≈ |
| French → Chinese | 59.8 | 50.2 [45.7, 54.6] ↓ | 70.2 [66.2, 74.1] ↑ |
| French → German | 62.5 | 52.4 [46.3, 58.2] ↓ | 65.0 [59.6, 70.0] ≈ |

## Eye minus EEG, per direction (paper DNN vs our normalised MLP)

| Train → Test | Paper eye − EEG | Ours eye − EEG |
|---|---:|---:|
| Chinese → Chinese | -2.5 | -4.4 |
| German → German | +18.4 | +26.5 |
| French → French | +15.7 | +23.9 |
| Chinese → German | +7.1 | +6.2 |
| Chinese → French | +5.1 | -0.9 |
| German → Chinese | -7.7 | -16.4 |
| German → French | +8.2 | +1.2 |
| French → Chinese | -3.2 | -9.5 |
| French → German | +5.3 | +12.1 |
