# NormFusion v2 vs v1 vs DCCA-AM across cultures (EEG + eye)

v2 is a post-hoc model: its ideas came from v1's results and from an analysis of emotion signatures that used the labels of all three cultures. v1 is the primary, pre-specified result.

Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. p: paired Wilcoxon over test people, v2 fair vs v1 fair.

| Train → Test | v1 fair | v2 fair | v2 fair macro-F1 | p (v2 vs v1) | v2 paper's rules | DCCA-AM fair | DCCA-AM paper's rules |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 51.7 [46.0, 58.3] | 60.3 [52.9, 68.1] | 60.0 [52.5, 67.7] | 0.008 | 66.4 [58.6, 74.6] | 35.3 [33.5, 37.3] | 58.9 [53.9, 64.5] |
| Chinese → French | 61.5 [56.2, 67.2] | 60.8 [57.4, 65.5] | 61.2 [58.0, 66.0] | 0.547 | 65.3 [61.7, 70.0] | 46.1 [41.8, 50.0] | 64.3 [60.4, 67.9] |
| German → Chinese | 52.9 [48.6, 57.2] | 44.6 [39.9, 49.1] | 43.4 [38.6, 47.8] | 0.001 | 76.9 [72.4, 81.1] | 47.2 [42.4, 52.0] | 69.2 [66.0, 72.8] |
| German → French | 64.6 [59.2, 69.8] | 60.8 [55.4, 65.9] | 58.6 [53.1, 63.9] | 0.023 | 72.7 [69.3, 76.1] | 49.8 [44.4, 55.2] | 65.5 [62.1, 68.7] |
| French → Chinese | 75.9 [69.9, 82.0] | 74.1 [68.3, 79.9] | 73.7 [67.9, 79.6] | 0.266 | 78.4 [73.8, 82.9] | 44.1 [40.3, 48.2] | 72.3 [68.2, 76.6] |
| French → German | 72.1 [67.4, 76.0] | 72.8 [66.0, 78.3] | 71.8 [64.7, 77.6] | 0.742 | 76.9 [71.0, 82.1] | 40.3 [36.2, 44.4] | 68.5 [62.7, 74.6] |

Average over 6 directions: v2 fair 62.3%, v1 fair 63.1%; v2 higher than v1 in 2 of 6; DCCA-AM fair 43.8%.

## Two cultures → the third (compared only with v1 in the same setting)

| Train → Test | v1 | v2 fair | v2 fair macro-F1 | p (v2 vs v1) |
|---|---:|---:|---:|---:|
| German + French → Chinese | 73.9 [68.3, 79.5] | 72.5 [67.7, 77.5] | 72.0 [67.1, 77.2] | 0.339 |
| Chinese + German → French | 68.2 [62.8, 72.1] | 70.5 [66.8, 74.6] | 70.7 [67.0, 74.8] | 0.312 |
| Chinese + French → German | 70.9 [65.5, 76.0] | 71.4 [66.4, 76.1] | 70.1 [64.5, 75.4] | 0.383 |

## New films (German ↔ French, shared films removed from training), fair accuracy %

| Train → Test | v1 | v2 | v2 macro-F1 | p (v2 vs v1) | DCCA-AM |
|---|---:|---:|---:|---:|---:|
| German → French | 60.6 [55.6, 65.2] | 56.6 [51.3, 61.5] | 56.4 [50.9, 61.2] | 0.078 | 37.4 [33.6, 41.3] |
| French → German | 68.1 [62.1, 73.7] | 66.2 [60.9, 71.3] | 65.4 [59.8, 70.7] | 0.148 | 38.0 [32.8, 43.4] |

## Ablations (fair accuracy %, mean over test people)

| Setting | Train → Test | fair | no_smoothing | causal | v1_fusion | no_selftrain | v1 |
|---|---|---:|---:|---:|---:|---:|---:|
| emotion | French → Chinese | 74.1 | 74.1 | 74.1 | 75.9 | 69.9 | 75.9 |
| emotion | German → Chinese | 44.6 | 44.6 | 44.6 | 52.9 | 46.5 | 52.9 |
| emotion_two_cultures | German + French → Chinese | 72.5 | 72.5 | 72.5 | 73.9 | 69.7 | 73.9 |
| emotion | Chinese → French | 60.8 | 60.8 | 60.8 | 61.5 | 61.8 | 61.5 |
| emotion | German → French | 60.8 | 60.8 | 60.8 | 64.6 | 58.7 | 64.6 |
| emotion_newfilms | German → French | 56.6 | – | – | – | 53.3 | 60.6 |
| emotion_two_cultures | Chinese + German → French | 70.5 | 70.5 | 70.5 | 68.2 | 69.9 | 68.2 |
| emotion | Chinese → German | 60.3 | 60.3 | 60.3 | 51.7 | 60.3 | 51.7 |
| emotion | French → German | 72.8 | 72.8 | 72.8 | 72.1 | 70.6 | 72.1 |
| emotion_newfilms | French → German | 66.2 | – | – | – | 65.3 | 68.1 |
| emotion_two_cultures | Chinese + French → German | 71.4 | 71.4 | 71.4 | 70.9 | 69.9 | 70.9 |

## Fair choices (fusion, C_EEG, C_eye, weights neg/neu/pos, k) per seed

| Setting | Train → Test | Seed | Choice | CV score |
|---|---|---:|---|---:|
| emotion | French → Chinese | 12 | late, 0.001, 0.001, 0/0/1, 1 | 68.5 |
| emotion | French → Chinese | 34 | late, 0.001, 0.001, 0/0/1, 1 | 68.5 |
| emotion | French → Chinese | 56 | late, 0.001, 0.001, 0/0/1, 1 | 69.7 |
| emotion | German → Chinese | 12 | late, 0.001, 0.1, 0/1/0, 1 | 69.2 |
| emotion | German → Chinese | 34 | late, 0.001, 0.1, 0/1/0, 1 | 67.6 |
| emotion | German → Chinese | 56 | late, 0.001, 1.0, 0/1/0, 1 | 68.3 |
| emotion_two_cultures | German + French → Chinese | 12 | late, 0.001, 0.001, 0/0/1, 1 | 69.2 |
| emotion_two_cultures | German + French → Chinese | 34 | late, 0.001, 0.001, 0/0/1, 1 | 69.4 |
| emotion_two_cultures | German + French → Chinese | 56 | late, 0.001, 0.001, 0/0/1, 1 | 69.6 |
| emotion | Chinese → French | 12 | late, 0.01, 0.1, 0.5/0/1, 1 | 76.3 |
| emotion | Chinese → French | 34 | late, 0.001, 0.01, 1/0/1, 1 | 75.5 |
| emotion | Chinese → French | 56 | late, 0.01, 0.1, 0.5/0/1, 1 | 76.2 |
| emotion | German → French | 12 | late, 0.001, 0.1, 0/1/0, 1 | 69.2 |
| emotion | German → French | 34 | late, 0.001, 0.1, 0/1/0, 1 | 67.6 |
| emotion | German → French | 56 | late, 0.001, 1.0, 0/1/0, 1 | 68.3 |
| emotion_newfilms | German → French | 12 | early, 0.001, None, –, 1 | 57.7 |
| emotion_newfilms | German → French | 34 | early, 0.001, None, –, 1 | 58.1 |
| emotion_newfilms | German → French | 56 | early, 0.001, None, –, 1 | 57.5 |
| emotion_two_cultures | Chinese + German → French | 12 | late, 0.001, 0.01, 0.5/0/1, 1 | 72.7 |
| emotion_two_cultures | Chinese + German → French | 34 | late, 0.001, 0.01, 0.5/0/1, 1 | 72.8 |
| emotion_two_cultures | Chinese + German → French | 56 | late, 0.01, 0.01, 0/0/1, 1 | 73.1 |
| emotion | Chinese → German | 12 | late, 0.01, 0.1, 0.5/0/1, 1 | 76.3 |
| emotion | Chinese → German | 34 | late, 0.001, 0.01, 1/0/1, 1 | 75.5 |
| emotion | Chinese → German | 56 | late, 0.01, 0.1, 0.5/0/1, 1 | 76.2 |
| emotion | French → German | 12 | late, 0.001, 0.001, 0/0/1, 1 | 68.5 |
| emotion | French → German | 34 | late, 0.001, 0.001, 0/0/1, 1 | 68.5 |
| emotion | French → German | 56 | late, 0.001, 0.001, 0/0/1, 1 | 69.7 |
| emotion_newfilms | French → German | 12 | late, 0.001, 0.1, 0.5/0/1, 1 | 65.6 |
| emotion_newfilms | French → German | 34 | late, 0.01, 0.1, 0/0/1, 1 | 66.4 |
| emotion_newfilms | French → German | 56 | late, 0.001, 0.01, 0.5/0/1, 1 | 67.0 |
| emotion_two_cultures | Chinese + French → German | 12 | late, 1.0, 1.0, 0/0.5/1, 1 | 72.1 |
| emotion_two_cultures | Chinese + French → German | 34 | late, 0.01, 0.001, 0/0.5/1, 1 | 73.1 |
| emotion_two_cultures | Chinese + French → German | 56 | late, 0.001, 0.001, 0.5/0/1, 1 | 73.1 |

## Clip-level accuracy % (majority vote over each clip's windows; uses the known clip boundaries at test time — not comparable with any window-level number)

| Setting | Train → Test | v1 | v2 fair |
|---|---|---:|---:|
| emotion | French → Chinese | 76.0 | 74.8 |
| emotion | German → Chinese | 53.4 | 45.3 |
| emotion_two_cultures | German + French → Chinese | 73.9 | 73.0 |
| emotion | Chinese → French | 64.7 | 65.6 |
| emotion | German → French | 62.8 | 58.1 |
| emotion_newfilms | German → French | 60.9 | 57.5 |
| emotion_two_cultures | Chinese + German → French | 69.4 | 73.1 |
| emotion | Chinese → German | 52.9 | 61.8 |
| emotion | French → German | 67.8 | 69.1 |
| emotion_newfilms | French → German | 61.7 | 61.5 |
| emotion_two_cultures | Chinese + French → German | 67.7 | 67.9 |

## v1 reproduction check

v2's v1 variant vs the saved v1 results: 24 runs compared, 24 with the same choice, largest per-person accuracy difference 0.00e+00 (reproduced).
