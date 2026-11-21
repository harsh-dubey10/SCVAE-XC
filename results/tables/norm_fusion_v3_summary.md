# NormFusion v3 across cultures (EEG + eye)

v3 is a post-hoc model with a pre-registered design and prediction (docs/superpowers/specs/2026-10-02-norm-fusion-v3-design.md); v1 is the primary, pre-specified model. v1 here is re-run on the same machine as v3.

Window-level accuracy %, mean over test people [95% bootstrap CI], seeds averaged per person. p: paired Wilcoxon over test people, v3 fair vs v1.

| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v3 predicted | v2 fair | DCCA-AM fair | v3 paper's rules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 51.2 [45.3, 58.1] | 51.2 [46.0, 56.3] | 51.0 [45.7, 56.3] | 0.844 | 54 | 60.3 [52.9, 67.9] | 35.3 [33.5, 37.3] | 54.2 [49.4, 59.6] |
| Chinese → French | 60.9 [56.0, 66.3] | 60.4 [55.5, 65.4] | 60.8 [56.2, 65.7] | 0.461 | 60 | 60.8 [57.4, 65.7] | 46.1 [41.8, 50.0] | 61.5 [57.5, 66.1] |
| German → Chinese | 54.3 [49.8, 58.7] | 65.5 [60.4, 70.9] | 65.4 [60.3, 70.6] | 0.000 | 64 | 44.6 [39.9, 49.1] | 47.2 [42.4, 52.0] | 67.9 [62.7, 73.4] |
| German → French | 64.7 [59.6, 69.6] | 68.5 [64.0, 72.9] | 68.2 [63.9, 72.6] | 0.016 | 65 | 60.8 [55.7, 66.0] | 49.8 [44.4, 55.2] | 70.4 [66.7, 74.1] |
| French → Chinese | 75.4 [68.9, 81.8] | 77.1 [72.5, 81.1] | 76.4 [71.7, 80.6] | 0.569 | 76 | 74.1 [68.3, 80.0] | 44.1 [40.3, 48.2] | 77.5 [72.9, 81.6] |
| French → German | 72.4 [67.5, 76.2] | 67.8 [63.0, 72.3] | 66.5 [61.4, 71.5] | 0.008 | 70 | 72.8 [65.9, 78.4] | 40.3 [36.2, 44.4] | 70.6 [66.1, 75.1] |

Average over 6 directions: v3 fair 65.1% (predicted 64.8%), v1 63.1%.
Success criterion (higher than v1 on average and in at least 4 of 6): not met (v3 higher in 3 of 6).

## Two cultures → the third (predicted 70–75%)

| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v2 fair | chosen memberships |
|---|---:|---:|---:|---:|---:|---|
| German + French → Chinese | 73.4 [67.9, 78.8] | 72.8 [67.0, 78.6] | 72.6 [66.8, 78.6] | 0.622 | 72.5 [67.6, 77.6] | eeg+eye |
| Chinese + German → French | 68.0 [62.8, 71.7] | 67.9 [62.2, 72.3] | 68.2 [62.4, 72.6] | 0.945 | 70.5 [66.8, 74.6] | eeg+eye+early |
| Chinese + French → German | 70.3 [65.0, 75.7] | 68.6 [62.3, 74.5] | 67.7 [61.2, 73.7] | 0.383 | 71.4 [66.4, 76.0] | eeg+eye+early |

## New films (German ↔ French, shared films removed from training)

| Train → Test | v1 | v3 fair | v3 macro-F1 | p (v3 vs v1) | v2 fair | DCCA-AM |
|---|---:|---:|---:|---:|---:|---:|
| German → French | 61.0 [56.8, 65.2] | 59.7 [56.4, 62.5] | 59.5 [56.1, 62.2] | 0.461 | 56.6 [51.4, 61.5] | 37.4 [33.6, 41.3] |
| French → German | 66.0 [61.9, 69.9] | 61.3 [57.2, 65.7] | 60.5 [56.1, 65.2] | 0.008 | 66.2 [60.9, 71.4] | 38.0 [32.7, 43.4] |

## Ablations (fair accuracy %, mean over test people)

| Setting | Train → Test | fair | no_bagging | no_selftrain | members_eeg | members_eeg_eye | members_eeg_early | members_all | v1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| emotion | French → Chinese | 77.1 | 78.0 | 72.3 | 65.5 | 75.0 | 71.5 | 77.1 | 75.4 |
| emotion | German → Chinese | 65.5 | 63.5 | 63.9 | 66.2 | 58.9 | 71.5 | 65.5 | 54.3 |
| emotion_two_cultures | German + French → Chinese | 72.8 | 72.5 | 69.1 | 66.9 | 72.8 | 75.1 | 75.9 | 73.4 |
| emotion | Chinese → French | 60.4 | 61.1 | 58.1 | 51.5 | 60.8 | 55.7 | 60.4 | 60.9 |
| emotion | German → French | 68.5 | 67.0 | 66.9 | 60.4 | 67.2 | 66.7 | 68.5 | 64.7 |
| emotion_newfilms | German → French | 59.7 | 57.3 | 56.0 | 54.4 | 62.1 | 57.8 | 59.7 | 61.0 |
| emotion_two_cultures | Chinese + German → French | 67.9 | 67.3 | 67.4 | 57.8 | 68.1 | 63.6 | 67.9 | 68.0 |
| emotion | Chinese → German | 51.2 | 51.1 | 52.0 | 43.7 | 51.8 | 46.5 | 51.2 | 51.2 |
| emotion | French → German | 67.8 | 68.7 | 66.0 | 55.1 | 71.6 | 61.7 | 67.8 | 72.4 |
| emotion_newfilms | French → German | 61.3 | 60.4 | 58.4 | 48.6 | 63.8 | 55.4 | 61.3 | 66.0 |
| emotion_two_cultures | Chinese + French → German | 68.6 | 68.4 | 66.8 | 54.3 | 71.3 | 60.6 | 68.6 | 70.3 |

## Choices per seed (membership; C for EEG / eye / early)

| Setting | Train → Test | Seed | Membership | C |
|---|---|---:|---|---|
| emotion | French → Chinese | 12 | eeg+eye+early | 1 / 0.01 / 0.001 |
| emotion | French → Chinese | 34 | eeg+eye+early | 0.001 / 0.01 / 1 |
| emotion | French → Chinese | 56 | eeg+eye+early | 0.001 / 0.001 / 0.001 |
| emotion | French → Chinese | 78 | eeg+eye+early | 1 / 0.001 / 0.01 |
| emotion | French → Chinese | 90 | eeg+eye+early | 1 / 0.01 / 0.001 |
| emotion | German → Chinese | 12 | eeg+eye+early | 0.001 / 1 / 0.01 |
| emotion | German → Chinese | 34 | eeg+eye+early | 0.001 / 0.1 / 0.001 |
| emotion | German → Chinese | 56 | eeg+eye+early | 0.001 / 0.1 / 0.01 |
| emotion | German → Chinese | 78 | eeg+eye+early | 0.001 / 0.01 / 0.01 |
| emotion | German → Chinese | 90 | eeg+eye+early | 0.001 / 0.1 / 0.1 |
| emotion_two_cultures | German + French → Chinese | 12 | eeg+eye | 0.1 / 0.1 / 0.1 |
| emotion_two_cultures | German + French → Chinese | 34 | eeg+eye | 0.001 / 0.01 / 0.01 |
| emotion_two_cultures | German + French → Chinese | 56 | eeg+eye | 0.001 / 0.01 / 0.001 |
| emotion_two_cultures | German + French → Chinese | 78 | eeg+eye | 0.01 / 1 / 0.1 |
| emotion_two_cultures | German + French → Chinese | 90 | eeg+eye | 0.01 / 0.01 / 0.001 |
| emotion | Chinese → French | 12 | eeg+eye+early | 0.001 / 0.001 / 0.001 |
| emotion | Chinese → French | 34 | eeg+eye+early | 0.001 / 0.001 / 1 |
| emotion | Chinese → French | 56 | eeg+eye+early | 0.01 / 0.001 / 0.01 |
| emotion | Chinese → French | 78 | eeg+eye+early | 0.001 / 0.01 / 0.1 |
| emotion | Chinese → French | 90 | eeg+eye+early | 0.001 / 0.001 / 0.01 |
| emotion | German → French | 12 | eeg+eye+early | 0.001 / 1 / 0.01 |
| emotion | German → French | 34 | eeg+eye+early | 0.001 / 0.1 / 0.001 |
| emotion | German → French | 56 | eeg+eye+early | 0.001 / 0.1 / 0.01 |
| emotion | German → French | 78 | eeg+eye+early | 0.001 / 0.01 / 0.01 |
| emotion | German → French | 90 | eeg+eye+early | 0.001 / 0.1 / 0.1 |
| emotion_newfilms | German → French | 12 | eeg+eye+early | 0.001 / 1 / 0.1 |
| emotion_newfilms | German → French | 34 | eeg+eye+early | 0.1 / 0.01 / 0.001 |
| emotion_newfilms | German → French | 56 | eeg+eye+early | 0.01 / 0.01 / 0.001 |
| emotion_newfilms | German → French | 78 | eeg+eye+early | 0.01 / 1 / 0.001 |
| emotion_newfilms | German → French | 90 | eeg+eye+early | 0.001 / 0.1 / 0.001 |
| emotion_two_cultures | Chinese + German → French | 12 | eeg+eye+early | 0.001 / 0.01 / 0.001 |
| emotion_two_cultures | Chinese + German → French | 34 | eeg+eye+early | 0.001 / 1 / 0.001 |
| emotion_two_cultures | Chinese + German → French | 56 | eeg+eye+early | 0.001 / 1 / 0.001 |
| emotion_two_cultures | Chinese + German → French | 78 | eeg+eye+early | 0.01 / 1 / 0.001 |
| emotion_two_cultures | Chinese + German → French | 90 | eeg+eye+early | 0.001 / 0.1 / 0.001 |
| emotion | Chinese → German | 12 | eeg+eye+early | 0.001 / 0.001 / 0.001 |
| emotion | Chinese → German | 34 | eeg+eye+early | 0.001 / 0.001 / 1 |
| emotion | Chinese → German | 56 | eeg+eye+early | 0.01 / 0.001 / 0.01 |
| emotion | Chinese → German | 78 | eeg+eye+early | 0.001 / 0.01 / 0.1 |
| emotion | Chinese → German | 90 | eeg+eye+early | 0.001 / 0.001 / 0.01 |
| emotion | French → German | 12 | eeg+eye+early | 1 / 0.01 / 0.001 |
| emotion | French → German | 34 | eeg+eye+early | 0.001 / 0.01 / 1 |
| emotion | French → German | 56 | eeg+eye+early | 0.001 / 0.001 / 0.001 |
| emotion | French → German | 78 | eeg+eye+early | 1 / 0.001 / 0.01 |
| emotion | French → German | 90 | eeg+eye+early | 1 / 0.01 / 0.001 |
| emotion_newfilms | French → German | 12 | eeg+eye+early | 1 / 0.001 / 1 |
| emotion_newfilms | French → German | 34 | eeg+eye+early | 1 / 0.001 / 1 |
| emotion_newfilms | French → German | 56 | eeg+eye+early | 1 / 0.1 / 1 |
| emotion_newfilms | French → German | 78 | eeg+eye+early | 1 / 1 / 1 |
| emotion_newfilms | French → German | 90 | eeg+eye+early | 1 / 1 / 1 |
| emotion_two_cultures | Chinese + French → German | 12 | eeg+eye+early | 1 / 0.001 / 0.001 |
| emotion_two_cultures | Chinese + French → German | 34 | eeg+eye+early | 1 / 1 / 1 |
| emotion_two_cultures | Chinese + French → German | 56 | eeg+eye+early | 1 / 0.001 / 0.001 |
| emotion_two_cultures | Chinese + French → German | 78 | eeg+eye+early | 0.1 / 0.01 / 0.001 |
| emotion_two_cultures | Chinese + French → German | 90 | eeg+eye+early | 1 / 0.01 / 0.01 |

## Clip-level accuracy % (majority vote over each clip's windows; uses the known clip boundaries at test time — not comparable with any window-level number)

| Setting | Train → Test | v1 | v3 fair |
|---|---|---:|---:|
| emotion | French → Chinese | 75.6 | 76.9 |
| emotion | German → Chinese | 54.8 | 65.6 |
| emotion_two_cultures | German + French → Chinese | 73.5 | 73.3 |
| emotion | Chinese → French | 64.2 | 63.7 |
| emotion | German → French | 63.1 | 68.3 |
| emotion_newfilms | German → French | 61.5 | 61.3 |
| emotion_two_cultures | Chinese + German → French | 69.9 | 69.4 |
| emotion | Chinese → German | 52.6 | 51.8 |
| emotion | French → German | 68.4 | 64.6 |
| emotion_newfilms | French → German | 60.1 | 57.2 |
| emotion_two_cultures | Chinese + French → German | 67.3 | 65.9 |

## Same-machine v1 vs saved laptop v1 (seeds 12, 34, 56)

| Setting | Train → Test | difference (points) | flag |
|---|---|---:|---|
| emotion | French → Chinese | 0.69 | ok |
| emotion | German → Chinese | 2.20 | CHECK |
| emotion | Chinese → French | 0.02 | ok |
| emotion | German → French | 0.65 | ok |
| emotion_newfilms | German → French | 0.16 | ok |
| emotion | Chinese → German | 0.32 | ok |
| emotion | French → German | 0.43 | ok |
| emotion_newfilms | French → German | 3.77 | CHECK |
