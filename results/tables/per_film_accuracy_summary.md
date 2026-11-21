# Per-film accuracy (exploratory; saved probabilities, no training)

Accuracy %, mean over test people. pos/neg: positive and negative films only, decided by mean p(positive) vs p(negative). 3 days: windows of the person's 3 recording days of the same film pooled. Design: docs/superpowers/specs/2026-11-05-per-film-accuracy-design.md.

## Two training cultures → the third: v3

| Train → Test | window pos/neg | film, one day, pos/neg | film, 3 days, pos/neg | film, one day, 3-class | film, 3 days, 3-class |
|---|---:|---:|---:|---:|---:|
| German + French → Chinese | 83.2 | 83.1 | 92.5 | 73.7 | 81.1 |
| Chinese + French → German | 77.0 | 74.1 | 80.2 | 66.7 | 77.1 |
| Chinese + German → French | 77.6 | 77.7 | 80.4 | 70.0 | 77.4 |
| **Average** | **79.3** | **78.3** | **84.4** | **70.1** | **78.5** |
| items scored | 46212 | 936 | 328 | 1404 | 492 |

## Two training cultures → the third: SCVAE-XC

| Train → Test | window pos/neg | film, one day, pos/neg | film, 3 days, pos/neg | film, one day, 3-class | film, 3 days, 3-class |
|---|---:|---:|---:|---:|---:|
| German + French → Chinese | 75.9 | 76.9 | 85.8 | 68.9 | 76.7 |
| Chinese + French → German | 75.8 | 75.7 | 80.2 | 67.0 | 75.7 |
| Chinese + German → French | 77.2 | 78.9 | 83.0 | 69.2 | 76.2 |
| **Average** | **76.3** | **77.2** | **83.0** | **68.4** | **76.2** |
| items scored | 46212 | 936 | 328 | 1404 | 492 |

## One training culture: v3

| Train → Test | window pos/neg | film, one day, pos/neg | film, 3 days, pos/neg | film, one day, 3-class | film, 3 days, 3-class |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 63.7 | 62.5 | 69.8 | 52.1 | 61.1 |
| Chinese → French | 69.0 | 72.6 | 78.6 | 64.5 | 70.8 |
| German → Chinese | 76.0 | 76.7 | 82.5 | 65.9 | 72.8 |
| German → French | 76.9 | 71.4 | 74.1 | 68.3 | 76.2 |
| French → Chinese | 91.7 | 92.5 | 95.8 | 77.2 | 84.4 |
| French → German | 79.2 | 76.2 | 84.4 | 65.0 | 72.2 |
| **Average** | **76.1** | **75.3** | **80.9** | **65.5** | **72.9** |
| items scored | 92424 | 1872 | 656 | 2808 | 984 |

## One training culture: SCVAE-XC

| Train → Test | window pos/neg | film, one day, pos/neg | film, 3 days, pos/neg | film, one day, 3-class | film, 3 days, 3-class |
|---|---:|---:|---:|---:|---:|
| Chinese → German | 66.4 | 67.7 | 67.7 | 59.4 | 62.5 |
| Chinese → French | 67.3 | 72.6 | 76.8 | 64.5 | 70.2 |
| German → Chinese | 61.0 | 62.5 | 61.7 | 57.6 | 61.1 |
| German → French | 70.9 | 66.4 | 74.1 | 60.1 | 70.2 |
| French → Chinese | 80.3 | 80.6 | 88.3 | 66.5 | 73.9 |
| French → German | 73.4 | 71.4 | 76.0 | 59.6 | 63.9 |
| **Average** | **69.9** | **70.2** | **74.1** | **61.3** | **67.0** |
| items scored | 92424 | 1872 | 656 | 2808 | 984 |

Prediction (v3, two cultures): window pos/neg 78, film one day 85, film 3 days 90; 3-class film 3 days 78.
