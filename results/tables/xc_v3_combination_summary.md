# SCVAE-XC + NormFusion v3 combination

Strict accuracy %, mean over test people. SCVAE-XC: 3-seed probability mean; v3: 5-seed probability mean (fair model); combined: equal-weight mean of the two. p: Wilcoxon over people, combined vs the better of the two. Design: docs/superpowers/specs/2026-11-04-xc-v3-combination-design.md.

## Two training cultures → the third

| Train → Test | SCVAE-XC | v3 seed-average | combined | p |
|---|---:|---:|---:|---:|
| German + French → Chinese | 68.1 | 73.3 | 69.5 | 0.204 |
| Chinese + French → German | 67.3 | 68.8 | 70.4 | 0.945 |
| Chinese + German → French | 66.4 | 68.1 | 68.8 | 0.461 |
| **Average** | **67.3** | **70.0** | **69.6** | |

## One training culture

| Train → Test | SCVAE-XC | v3 seed-average | combined | p |
|---|---:|---:|---:|---:|
| Chinese → German | 57.5 | 50.9 | 56.4 | 0.461 |
| Chinese → French | 59.1 | 60.6 | 61.2 | 0.383 |
| German → Chinese | 57.0 | 65.8 | 58.0 | 0.012 |
| German → French | 60.7 | 69.1 | 66.0 | 0.008 |
| French → Chinese | 66.4 | 77.5 | 71.6 | 0.002 |
| French → German | 62.2 | 68.5 | 66.3 | 0.195 |
| **Average** | **60.5** | **65.4** | **63.3** | |

Predicted (two cultures): combined 71.0, v3 seed-average 70.0.

## Decision (pre-registered, two cultures)

- combined − best single: -0.5 (needs ≥ +1.0); combined beats both in 2 of 3 (needs ≥ 2).
- **NO-GO: v3 remains the best fair model.**
