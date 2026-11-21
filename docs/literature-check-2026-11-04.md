# Literature check: published "SOTA" numbers vs our setting (4 November 2026)

Checked from HTML pages, Crossref/EuropePMC records and code repositories (PDF full texts not read).

| Paper | Reported (cross-subject) | Protocol | Model selection |
|---|---|---|---|
| SCVAE-Net, *Sensors* 26(10):3217, 2026 | SEED 95.01, SEED-IV 74.94; cross-session SEED 96.84 | LOSO within SEED, code uses session 1 only, same 15 films in train and test | **Best test accuracy**: `if acc_test > best_acc` in Main_train.py (github.com/braverSheep/SCVAE-Net); no validation set |
| CSCL, *Sci Rep* 15:28295, 2025 | SEED 97.70 (no SEED-IV) | text says LOSO, Table 2 says subject-wise 10-fold | no code; "re-tuned if performance drops" |
| PEPBNet, CogSci proceedings | SEED 96.30, SEED-IV 87.04 | not verified | not verified |
| FSA-Net, *Tsinghua Sci. Tech.* 2026 | SEED 90.80, SEED-IV 82.74 | not stated in abstract | no code |
| EmotionCLIP, arXiv 2511.05293 | SEED 88.69, SEED-IV 73.50 | 32 labelled target samples | max accuracy over epochs on target data |

All are within-dataset LOSO with the same films in training and test; none is comparable to cross-culture testing.

**Fair cross-corpus references (DE features):** McdPL (arXiv 2508.11663) SEED↔SEED-IV average 58.9; HEDN
(arXiv 2511.06782) SEED→SEED-IV 69.7, SEED-IV→SEED 73.6 (not checked in full). Realistic fair range ≈ 55–70%.

**Official cross-culture baseline** (Liu et al., *J. Neural Eng.* 19:026012, 2022, DCCA-AM): 73.6–84.3 (1 → 1).
Our re-run (`results/tables/dcca_am_summary.md`): paper's rules 58.9–72.3, fair 35.3–49.8; test-based search over
k architectures reproduces the inflation.

**Caveat found:** the SEED-FRA and SEED-GER pages state the neutral stimuli are the same as SEED's. Neutral test
films may therefore have been seen in training when Chinese is a source; "every test film unseen" holds for
positive and negative films only. To verify against the stimulus lists.

## Follow-up: SCVAE-Net-like methods for cross-culture transfer (5 November 2026)

Abstract/HTML level only, not full texts.

- **mdJPT** (arXiv 2510.22197, code ncclab-sustech/mdJPT_nips2025): raw-EEG multi-dataset pretraining with
  stimulus-locked contrastive (ISA) + per-subject covariance-centroid alignment (CDA). Zero-shot held-out SEED
  72.47 (DE baseline 50.54, LaBraM 48.54); no validation set, CDA weight chosen by a target ablation.
- **EmoVAE** (CogSci): VAE + class-conditional MMD; SEED→SEED-VII 55.0 (from a search snippet).
- **VAE-CapsNet** (*Knowl.-Based Syst.* 311:113018, 2025): VAE for session variance; cross-dataset numbers not found.
- **MMDA-VAE** (IEEE/CAA JAS 9(9), 2022): EEG + eye product-of-experts VAE; cross-session only, few labels.
- **DMMR** (AAAI 2024): domain-generalisation multi-decoder autoencoder; SEED 88.27 LOSO; no code found.
- **MSBLS** (IEEE TCSS 11(2), 2024): cross-culture SEED-GER/FRA experiments; numbers not verified.
- **Checkpoint selection study** (arXiv 2607.27655): best of 80 vs 5 checkpoints +6.2 on the selection pool,
  −1.2 elsewhere, independent evidence for strict selection.
- "One Model for All" (arXiv 2511.08444): withdrawn.

Ranked extensions of SCVAE-XC (estimates, untested): (1) more training films from SEED-IV/V (happy→positive,
sad/fear→negative) + CDA, +2 to +5; (2) session-invariant latent (session adversary/CDA, cross-session
reconstruction), +1 to +3; (3) product-of-experts EEG+eye fusion, 0 to +2. No method reports a strict,
unseen-film cross-corpus result above ~70–72%.
