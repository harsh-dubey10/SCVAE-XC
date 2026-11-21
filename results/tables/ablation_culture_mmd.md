# Loss ablations (SCVAE-XC, EEG + eye, two training cultures)

Strict accuracy %, 3-seed probability mean. Per-culture columns: mean over that culture's people.

| Variant | Chinese | German | French | Mean of cultures | Mean of 28 people |
|---|---:|---:|---:|---:|---:|
| Two cultures, full | 68.1 | 67.3 | 66.4 | 67.3 | 67.4 |
| Two cultures, no culture MMD | 68.3 | 66.9 | 66.4 | 67.2 | 67.3 |
| Two cultures, no supervised contrastive loss | 68.6 | 67.5 | 66.1 | 67.4 | 67.6 |
| Two cultures, no source-target MMD | 64.7 | 67.4 | 66.6 | 66.2 | 66.0 |
| One culture (mean of two directions) | 61.7 | 59.9 | 59.9 | 60.5 | 60.7 |

| Comparison (A vs B) | n | Δ (28 people) | 95% CI | A better / worse | p | p (Holm, 5) |
|---|---:|---:|---|---|---:|---:|
| Two cultures: full vs no culture MMD | 28 | +0.0 | -1.4 to +1.5 | 14 / 13 | 1 | 1 |
| Two cultures: full vs no supervised contrastive loss | 28 | -0.2 | -1.8 to +1.5 | 13 / 15 | 0.8 | 1 |
| Two cultures: full vs no source-target MMD | 28 | +1.3 | -0.8 to +3.7 | 14 / 13 | 0.47 | 1 |
| Two cultures without culture MMD vs one culture | 28 | +6.7 | +4.7 to +8.6 | 23 / 5 | 2.8e-06 | 1.1e-05 |
| Two cultures full vs one culture | 28 | +6.7 | +4.8 to +8.6 | 25 / 3 | 1.3e-06 | 6.3e-06 |
