# NormFusion v3: fair vs the papers' test-label-selected rules

Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. Candidates per run: 145 (expert sets x C per expert without self-training, plus self-training rounds 1-3 at the fair Cs). **Only the fair column is accuracy.** The test-label-selected columns choose settings by accuracy on the test people, as the papers did, and are shown only to measure that inflation.

| Train → Test | Seeds | v3 fair | v3 test-selected (one setting per run) | v3 test-selected (per person) | DCCA-AM as published (test-selected) |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 3 | 50.8 [45.6, 55.8] | 57.7 [52.3, 63.6] | 59.7 [54.4, 65.5] | 73.6 |
| Chinese → French | 3 | 60.4 [55.3, 65.2] | 60.6 [55.1, 65.7] | 64.5 [61.1, 68.3] | 73.8 |
| German → Chinese | 3 | 65.1 [59.9, 70.7] | 74.8 [69.2, 79.6] | 75.8 [71.3, 80.0] | 84.3 |
| German → French | 3 | 68.5 [64.1, 72.9] | 69.8 [66.0, 73.3] | 73.0 [69.8, 76.3] | 77.1 |
| French → Chinese | 3 | 77.1 [72.7, 81.0] | 78.0 [74.2, 81.4] | 81.5 [77.2, 85.8] | 82.1 |
| French → German | 3 | 67.9 [63.0, 72.1] | 72.3 [68.5, 75.8] | 75.1 [70.8, 78.8] | 77.9 |
| **Average** | | **65.0** | 68.9 | 71.6 | 78.1 |

Inflation from test-label selection: +3.9 points (one setting per run), +6.6 points (per person).
