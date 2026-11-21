# SCVAE-XC in SCVAE-Net's within-SEED settings — design (6 November 2026)

*Written before running.* To place SCVAE-XC next to SCVAE-Net's published 95.01% (cross-subject) and 96.84%
(cross-session), run SCVAE-XC in the same two settings on SEED_CHN_MM (the 12 Chinese SEED participants with eye
data; same 15 films), with the eye branch (`eeg_eye`), step 4 recipe (100 epochs, learning rate 0.0025, no input
noise, seeds 12/34/56, 3-seed probability ensemble).

- **Cross-subject** (per session 1, 2, 3): train on the other 11 people of that session, test on the held-out
  person's same session (leave-one-subject-out). Same films in training and test, as in SCVAE-Net.
- **Cross-session** (per person and target session): train on the person's other two sessions, test on the third.

Two epoch rules, reported side by side:
- **Test-person epoch (SCVAE-Net's code):** for each test person, the epoch with the best accuracy on that person
  (from the per-epoch predictions, `--save-epoch-proba`). This reads test labels; it matches the published protocol
  and is labelled as such.
- **Held-out epoch:** cross-subject, epoch chosen on 2 held-out source people (as in step 4); cross-session, on a
  random 20% of the source sessions' windows (no test data).

**Prediction:** cross-subject test-person epoch 88, held-out 76; cross-session test-person epoch 92, held-out 86.
Descriptive comparison; no go rule. Laptop run. Script `src/experiments/run_scvae_xc_within.py`,
table `results/tables/scvae_xc_within_seed_summary.md`.

## Result (6 November 2026, cluster jobs 7427 / 7430; all 216 runs)

| Setting | Held-out epoch | Test-person epoch (SCVAE-Net's rule) | SCVAE-Net published |
|---|---:|---:|---:|
| Cross-subject (sessions 1/2/3: 82.4/81.9/89.6 · 88.5/88.0/94.3) | 84.6 | 90.3 | 95.01 |
| Cross-session (90.4/91.3/88.1 · 93.2/94.3/93.1) | 89.9 | 93.5 | 96.84 |

Prediction 88 / 76 (cross-subject) and 92 / 86 (cross-session): all four came in higher. Hardware check: 47 runs
done on both the laptop and the cluster agree (mean 83.8 vs 84.0); the low early session-1 numbers were one hard
person (subject 1, 33–54%). Differences from SCVAE-Net: 12-person multimodal release with the eye branch vs their
15-person EEG-only release, 100 vs 300 epochs. Table: `results/tables/scvae_xc_within_seed_summary.md`.
