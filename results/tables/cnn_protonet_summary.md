# CNN and prototypical-network few-shot study

Accuracy %, cross-culture, same test windows and calibration pools as the calibration study (results/tables/calibration_summary.md); mean over test people (seeds and 5 pools averaged per person), then over the 6 directions. All weights and hyperparameters fixed in advance.

Runs per direction (of 3 seeds): Chinese→German 3, Chinese→French 3, German→Chinese 3, German→French 3, French→Chinese 3, French→German 3

## 1. Average over the 6 directions

| Model | No calibration | 1 clip / emotion | 2 clips / emotion | Predicted (0 / 1 / 2) |
|---|---:|---:|---:|---|
| NormFusion v3 (+ person model) — calibration study | 64.6 | 61.7 | 66.2 | – / – / – |
| MLP classifier (+ person model) | 60.4 | 61.4 | 63.3 | – / – / – |
| CNN classifier (+ person model) | 60.5 | 62.2 | 64.5 | 62.0 / 61.0 / 65.0 |
| MLP prototypical network | – | 50.4 | 55.2 | – / 58.0 / 61.0 |
| CNN prototypical network | – | 50.3 | 55.2 | – / 58.0 / 61.0 |
| MLP prototypical + v3 | – | 57.0 | 60.9 | – / – / – |
| CNN prototypical + v3 | – | 57.8 | 61.5 | – / 65.0 / 68.0 |
| CNN prototypical + CNN classifier | – | 60.6 | 61.9 | – / – / – |

## 2. Per direction, 1 clip per emotion (mean [95% bootstrap CI over people]; p: Wilcoxon over people against v3+person)

| Train → Test | NormFusion v3 | MLP classifier | CNN classifier | MLP prototypical network | CNN prototypical network | MLP prototypical + v3 | CNN prototypical + v3 | CNN prototypical + CNN classifier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 51.5 [45.0, 58.6] | 51.3 [45.5, 57.5] p=0.844 | 53.1 [44.9, 60.0] p=0.742 | 47.3 [39.1, 56.8] p=0.312 | 45.1 [37.9, 53.3] p=0.109 | 50.0 [42.0, 58.7] p=0.641 | 48.2 [41.3, 56.3] p=0.148 | 50.6 [43.6, 57.6] p=0.844 |
| Chinese → French | 58.3 [53.3, 63.2] | 57.4 [53.5, 61.4] p=0.742 | 57.9 [55.9, 60.1] p=0.844 | 49.8 [44.8, 55.6] p=0.039 | 48.5 [42.3, 55.5] p=0.023 | 53.9 [48.7, 59.5] p=0.195 | 53.0 [47.2, 59.6] p=0.148 | 56.0 [53.5, 59.1] p=0.547 |
| German → Chinese | 67.6 [60.5, 74.3] | 65.3 [59.5, 70.5] p=0.151 | 66.0 [60.0, 71.2] p=0.339 | 47.0 [41.9, 52.0] p=0.000 | 48.6 [43.2, 53.7] p=0.001 | 54.3 [48.7, 59.8] p=0.003 | 56.5 [51.1, 61.4] p=0.007 | 61.8 [56.4, 66.8] p=0.092 |
| German → French | 61.5 [58.3, 64.8] | 62.7 [59.3, 66.6] p=0.641 | 63.7 [60.0, 67.5] p=0.312 | 48.0 [42.6, 53.6] p=0.016 | 47.8 [41.9, 54.2] p=0.016 | 56.3 [51.4, 61.3] p=0.250 | 59.0 [54.5, 63.7] p=0.383 | 62.4 [57.3, 68.2] p=0.547 |
| French → Chinese | 71.7 [65.3, 77.9] | 69.6 [64.7, 73.9] p=0.424 | 69.0 [63.0, 74.3] p=0.339 | 57.6 [52.9, 62.2] p=0.000 | 57.3 [52.6, 61.8] p=0.000 | 66.8 [62.8, 71.0] p=0.110 | 67.3 [64.2, 70.6] p=0.176 | 68.3 [64.4, 71.8] p=0.266 |
| French → German | 59.9 [52.2, 68.4] | 61.8 [55.7, 67.9] p=0.461 | 63.3 [58.2, 68.3] p=0.250 | 52.8 [47.5, 58.2] p=0.195 | 54.3 [49.3, 58.9] p=0.055 | 60.8 [56.0, 65.9] p=0.312 | 62.7 [57.7, 67.3] p=0.195 | 64.6 [59.4, 69.2] p=0.312 |
| **Wins vs v3+person** | – | 2/6 | 3/6 | 0/6 | 0/6 | 1/6 | 1/6 | 2/6 |

## 3. Per direction, 2 clips per emotion (mean [95% bootstrap CI over people]; p: Wilcoxon over people against v3+person)

| Train → Test | NormFusion v3 | MLP classifier | CNN classifier | MLP prototypical network | CNN prototypical network | MLP prototypical + v3 | CNN prototypical + v3 | CNN prototypical + CNN classifier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chinese → German | 56.4 [50.6, 63.0] | 53.5 [47.7, 59.6] p=0.195 | 54.8 [46.5, 61.6] p=0.945 | 47.9 [39.6, 56.5] p=0.109 | 46.5 [39.3, 54.9] p=0.039 | 51.0 [42.4, 59.5] p=0.195 | 49.7 [42.4, 57.7] p=0.078 | 51.5 [43.6, 58.8] p=0.312 |
| Chinese → French | 61.6 [57.3, 66.5] | 59.0 [55.7, 62.6] p=0.148 | 59.3 [57.5, 61.2] p=0.383 | 53.6 [47.8, 59.6] p=0.055 | 52.0 [45.4, 59.0] p=0.055 | 57.2 [51.8, 62.9] p=0.312 | 55.8 [49.5, 62.3] p=0.109 | 57.1 [54.0, 60.3] p=0.148 |
| German → Chinese | 72.7 [67.9, 77.5] | 68.3 [63.6, 72.6] p=0.034 | 69.1 [64.3, 73.3] p=0.077 | 51.5 [46.9, 55.8] p=0.000 | 51.9 [45.8, 57.2] p=0.000 | 58.7 [53.8, 63.2] p=0.001 | 60.5 [55.2, 65.3] p=0.002 | 62.9 [57.9, 67.7] p=0.007 |
| German → French | 65.5 [62.9, 68.7] | 63.8 [60.3, 67.7] p=0.547 | 65.7 [61.7, 69.6] p=0.945 | 55.9 [51.1, 60.6] p=0.023 | 56.2 [50.4, 62.1] p=0.023 | 62.0 [57.5, 66.2] p=0.461 | 65.0 [60.4, 69.4] p=1.000 | 63.8 [58.9, 69.0] p=0.742 |
| French → Chinese | 76.0 [71.8, 80.4] | 72.1 [68.6, 75.0] p=0.064 | 72.1 [67.5, 76.4] p=0.110 | 63.2 [57.8, 68.6] p=0.000 | 62.6 [57.5, 67.7] p=0.000 | 71.9 [68.0, 76.1] p=0.064 | 71.6 [67.8, 75.4] p=0.129 | 69.9 [65.6, 73.6] p=0.042 |
| French → German | 64.8 [58.5, 71.5] | 62.9 [57.4, 68.2] p=0.383 | 66.1 [60.9, 70.9] p=0.844 | 59.2 [52.8, 65.6] p=0.312 | 61.8 [55.9, 67.8] p=0.312 | 64.7 [59.6, 69.9] p=0.742 | 66.6 [61.0, 72.1] p=0.547 | 66.4 [61.0, 71.4] p=0.742 |
| **Wins vs v3+person** | – | 0/6 | 2/6 | 0/6 | 0/6 | 0/6 | 1/6 | 1/6 |

## 4. Pre-registered success check (N = 2)

Best variant: CNN classifier (+ person model) 64.5% vs v3+person 66.2% (-1.7 points; needs >= +3), wins 2/6 (needs >= 4). **Success NOT MET.**
