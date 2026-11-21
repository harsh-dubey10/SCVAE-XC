# NormFusion vs DCCA-AM across cultures (EEG + eye)

Accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. p: paired Wilcoxon, NormFusion fair vs DCCA-AM fair, over test people.

| Train → Test | DCCA-AM reported | DCCA-AM paper's rules | NormFusion paper's rules | DCCA-AM fair | NormFusion fair | NormFusion fair macro-F1 | p | Best simple |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 73.6 | 58.9 [53.9, 64.7] | 56.8 [50.7, 63.5] | 35.3 [33.5, 37.3] | 51.7 [46.0, 58.2] | 51.4 [45.3, 57.8] | 0.008 | 54.1 |
| Chinese → French | 73.8 | 64.3 [60.4, 68.0] | 62.5 [57.7, 68.0] | 46.1 [41.9, 50.0] | 61.5 [56.3, 67.3] | 61.9 [56.6, 67.6] | 0.016 | 58.3 |
| German → Chinese | 84.3 | 69.2 [66.0, 72.8] | 67.1 [63.1, 71.2] | 47.2 [42.5, 52.2] | 52.9 [48.5, 57.2] | 51.8 [47.3, 56.3] | 0.092 | 66.5 |
| German → French | 77.1 | 65.5 [62.1, 68.7] | 68.8 [63.8, 73.9] | 49.8 [44.4, 55.2] | 64.6 [59.3, 69.9] | 63.2 [57.5, 68.7] | 0.008 | 65.4 |
| French → Chinese | 82.1 | 72.3 [68.2, 76.6] | 77.3 [71.9, 82.5] | 44.1 [40.1, 48.2] | 75.9 [69.9, 82.0] | 75.7 [69.7, 81.7] | 0.000 | 72.1 |
| French → German | 77.9 | 68.5 [62.7, 74.6] | 73.5 [68.5, 77.4] | 40.3 [35.9, 44.3] | 72.1 [67.3, 76.2] | 71.0 [66.0, 75.4] | 0.008 | 65.0 |

NormFusion fair, all 6 directions: 63.1%.
Over the 6 directions where DCCA-AM is complete: NormFusion fair 63.1% vs DCCA-AM fair 43.8%; NormFusion higher in 6 of 6. Success criterion (spec): higher on average over the 6 directions and in at least 4 — met.

## Ablations (fair accuracy %, mean over test people)

| Train → Test | fair | no_selftrain | eeg_only | eye_only | lr_only | early_fusion | person_cv | no_norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 51.7 | 53.0 | 42.3 | 50.6 | 51.0 | 50.8 | 51.9 | 38.7 |
| Chinese → French | 61.5 | 58.1 | 52.0 | 54.2 | 60.2 | 58.7 | 60.6 | 43.9 |
| German → Chinese | 52.9 | 54.7 | 66.5 | 40.8 | 46.3 | 68.7 | 45.6 | 48.5 |
| German → French | 64.6 | 63.2 | 60.1 | 60.1 | 62.1 | 66.5 | 60.6 | 44.1 |
| French → Chinese | 75.9 | 68.0 | 64.4 | 58.1 | 70.4 | 78.0 | 61.3 | 49.1 |
| French → German | 72.1 | 69.2 | 56.1 | 69.1 | 69.1 | 65.4 | 69.5 | 51.9 |

## New films (German ↔ French, shared films removed from training), fair accuracy %

| Train → Test | NormFusion, all films | NormFusion, new films | DCCA-AM, new films |
|---|---:|---:|---:|
| German → French | 64.6 [59.2, 70.1] | 60.6 [55.8, 65.3] | 37.4 [33.7, 41.3] |
| French → German | 72.1 [67.4, 76.1] | 68.1 [62.1, 73.6] | 38.0 [32.7, 43.2] |
