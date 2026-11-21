# SCVAE-XC Stage 1: stimulus-locked contrastive pretraining (SLC)

Strict accuracy %, 3-seed ensemble (mean of saved probabilities), mean over test people. Both arms: the configuration chosen per test culture (2026-10-09), seeds 12/34/56; only the encoder's initialisation differs. No target label is read before the final evaluation.

| Train → Test | SCVAE-fixed | SCVAE-fixed + SLC | Difference | p (Wilcoxon over people) | runs |
|---|---:|---:|---:|---:|---|
| Chinese → German | 47.5 | 46.0 | -1.5 | 0.547 | 3 seeds, 8/8 people |
| Chinese → French | 48.6 | 50.8 | +2.3 | 0.461 | 3 seeds, 8/8 people |
| German → Chinese | 58.5 | 56.1 | -2.4 | 0.204 | 3 seeds, 12/12 people |
| German → French | 54.5 | 56.7 | +2.1 | 0.250 | 3 seeds, 8/8 people |
| French → Chinese | 59.9 | 58.5 | -1.3 | 0.339 | 3 seeds, 12/12 people |
| French → German | 50.6 | 49.2 | -1.5 | 0.742 | 3 seeds, 8/8 people |
| **Average (6 directions)** | **53.3** | **52.9** | -0.4 | | |

Pre-registered: SLC ensemble 56%; go if >= +2 points and higher in >= 4 of 6. Result: -0.4 points, higher in 2 of 6. **NO-GO.**

## Linear-probe diagnostic (mean of 18 direction-seeds)

Logistic regression (C = 0.1) on source embeddings, scored on the target (mean over people). Predicted: SLC > raw features by 2 points.

| Features | Probe accuracy |
|---|---:|
| raw | 53.1 |
| random_encoder | 47.2 |
| slc_encoder | 45.9 |
