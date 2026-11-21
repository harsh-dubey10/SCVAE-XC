# SCVAE-XC in SCVAE-Net's within-SEED settings

SEED_CHN_MM (12 Chinese participants, same 15 films), EEG + eye, 3-seed probability ensemble, 100 epochs. Accuracy %, mean over people per session, then over the 3 sessions. **Test-person epoch** = the epoch with the best accuracy on each test person, as in SCVAE-Net's code (reads test labels). **Held-out epoch** = chosen without test data. Design: docs/superpowers/specs/2026-11-06-scvae-xc-within-seed-design.md.

| Setting | Session 1 | Session 2 | Session 3 | Average (held-out epoch) | Average (test-person epoch) | SCVAE-Net published (test-person epoch) | runs |
|---|---:|---:|---:|---:|---:|---:|---|
| cross-subject | 82.4 / 88.5 | 81.9 / 88.0 | 89.6 / 94.3 | 84.6 | **90.3** | 95.01 | 36/36 |
| cross-session | 90.4 / 93.2 | 91.3 / 94.3 | 88.1 / 93.1 | 89.9 | **93.5** | 96.84 | 36/36 |

Session cells: held-out epoch / test-person epoch. SCVAE-Net's numbers are on the 15-person EEG-only SEED release (300 epochs); ours on the 12-person multimodal release with the eye branch (100 epochs).
