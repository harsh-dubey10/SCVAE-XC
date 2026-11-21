# DCCA-AM across cultures: paper vs paper's rules vs fair

Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person.

| Train → Test | Paper (reported) | Paper's rules (ours) | Fair | Fair, new films | Best simple, normalised | Skipped batches / non-finite epochs |
|---|---:|---:|---:|---:|---:|---:|
| Chinese → German | 73.6 | 58.9 [53.9, 64.7] | 35.3 [33.5, 37.3] | same as fair | 54.1 | 0 / 0 |
| Chinese → French | 73.8 | 64.3 [60.5, 67.9] | 46.1 [41.8, 50.0] | same as fair | 58.3 | 0 / 0 |
| German → Chinese | 84.3 | 69.2 [65.9, 72.7] | 47.2 [42.5, 52.2] | same as fair | 66.5 | 0 / 0 |
| German → French | 77.1 | 65.5 [62.1, 68.7] | 49.8 [44.4, 55.2] | 37.4 [33.6, 41.2] | 65.4 | 0 / 0 |
| French → Chinese | 82.1 | 72.3 [68.3, 76.5] | 44.1 [40.2, 48.3] | same as fair | 72.1 | 0 / 0 |
| French → German | 77.9 | 68.5 [62.8, 74.5] | 40.3 [36.0, 44.3] | 38.0 [32.7, 43.4] | 65.0 | 0 / 0 |

## Leakage curve: paper's rules when searching k architectures

| Train → Test | k = 1 | k = 2 | k = 5 | k = 10 | k = 20 | Fair |
|---|---:|---:|---:|---:|---:|---:|
| Chinese → German | 49.2 | 52.4 | 55.2 | 58.1 | 58.9 | 35.3 |
| Chinese → French | 56.7 | 59.3 | 61.0 | 63.3 | 64.3 | 46.1 |
| German → Chinese | 58.8 | 60.6 | 66.0 | 68.1 | 69.2 | 47.2 |
| German → French | 53.8 | 58.9 | 62.6 | 63.7 | 65.5 | 49.8 |
| French → Chinese | 63.0 | 64.7 | 68.5 | 70.3 | 72.3 | 44.1 |
| French → German | 57.7 | 59.2 | 63.6 | 65.2 | 68.5 | 40.3 |

## Film identification (German ↔ French shared films), balanced accuracy %

| Train → Test | Task | Chance | DCCA-AM fair | p | DCCA-AM paper's rules | Linear (E2) |
|---|---|---:|---:|---:|---:|---:|
| German → French | all_5 | 20 | 17.9 [13.2, 22.4] | 0.8581 | 41.0 | 13.8 |
| German → French | negative | 33 | 23.5 [15.3, 32.6] | 0.9985 | 58.3 | 29.6 |
| German → French | negative_2 | 50 | 61.2 [51.1, 71.8] | 0.0090 | 90.4 | 65.0 |
| German → French | positive | 50 | 36.2 [31.0, 40.8] | 0.9985 | 84.5 | 20.8 |
| French → German | all_5 | 20 | 18.1 [16.1, 20.1] | 0.7746 | 52.4 | 19.5 |
| French → German | negative | 33 | 27.2 [25.2, 29.1] | 0.9620 | 67.4 | 40.1 |
| French → German | negative_2 | 50 | 51.1 [39.4, 63.5] | 0.4203 | 100.0 | 90.2 |
| French → German | positive | 50 | 28.4 [16.5, 40.1] | 0.9995 | 78.3 | 29.8 |
