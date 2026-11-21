# SCVAE-Net across cultures and on unseen films

Spec: `docs/superpowers/specs/2026-10-09-scvae-crossculture-improvement-design.md`. Strict columns: no target label read before the final evaluation. **Test-label-selected** columns pick the checkpoint by test accuracy (the released code's policy); they are a labelled comparison, not accuracy.

## Configuration chosen per test culture (from the other two cultures' transfers only)

| Test culture | Chosen configuration | Validation acc. |
|---|---|---:|
| Chinese | strict__bn_source__lr_cosine__tshuf__wd0.0001__noise0.1__sval__subjnorm | 51.9 |
| German | strict__bn_source__lr_cosine__tshuf__wd0.0001__sval__subjnorm | 53.9 |
| French | strict__bn_source__lr_cosine__tshuf__lrate0.0005__wd0.0001__noise0.1__sval__subjnorm | 51.3 |

## Part 1 — Across cultures: accuracy (%) per direction

| Train → Test | Best simple model | SCVAE released (strict) | SCVAE-fixed, 1 seed (strict) | SCVAE-fixed, 3-seed ensemble (strict) | Ensemble macro-F1 || Released, **test-label-selected** | runs complete (fixed ensemble / test-label-selected) |
|---|---:|---:|---:|---:|---:|---:|---|
| Chinese → German | 46.2 | 48.3 | 46.5 | **47.5** | 47.6 || 69.2 | 3 seeds, 8/8 people / 3 seeds, 8/8 people |
| Chinese → French | 52.5 | 55.0 | 48.8 | **48.6** | 48.6 || 68.7 | 3 seeds, 8/8 people / 3 seeds, 8/8 people |
| German → Chinese | 63.6 | 47.8 | 55.6 | **58.5** | 58.8 || 73.8 | 3 seeds, 12/12 people / 3 seeds, 12/12 people |
| German → French | 60.5 | 54.9 | 52.9 | **54.5** | 54.0 || 70.8 | 3 seeds, 8/8 people / 3 seeds, 8/8 people |
| French → Chinese | 64.3 | 57.4 | 56.9 | **59.9** | 59.5 || 77.5 | 3 seeds, 12/12 people / 3 seeds, 12/12 people |
| French → German | 58.3 | 49.0 | 49.4 | **50.6** | 49.3 || 70.5 | 3 seeds, 8/8 people / 3 seeds, 8/8 people |
| **Average** | 57.6 | 52.1 | 51.7 | **53.3** | 53.0 || 71.7 | |

- SCVAE-fixed ensemble − SCVAE released (strict), (direction, person) pairs: +2.0 points over 56 pairs (95% CI -1.1 to +5.1), Wilcoxon p = 0.221.
- Peeking gain, released recipe (test-label-selected − strict): +19.7 points.

**Pre-registered success criterion:** SCVAE-fixed ensemble 53.3% vs best simple models 57.6%; wins 1 of 6 (need > 57.6% and ≥ 4). **NOT MET.** Predicted: 58% strict, 72% test-label-selected.

### Grid factors (exploratory; mean over all 6 directions and the other factors)

| Factor | Level | Mean accuracy |
|---|---|---:|
| learning rate | 0.0025 | 51.9 |
| learning rate | 0.0005 | 51.5 |
| BN at test | source | 51.8 |
| BN at test | target | 51.5 |
| input noise | 0 | 51.7 |
| input noise | 0.1 | 51.6 |

## Part 2 — Within each culture: seen vs unseen films (accuracy %)

Same test windows in both columns; same amount of training data summed over the 3 film folds. Each person is averaged over the folds and the seeds.

| Culture | Model | Seen films | Unseen films | Drop | People dropping | Wilcoxon p | Runs complete |
|---|---|---:|---:|---:|---|---:|---|
| Chinese | SCVAE released + BN fix (strict) | 80.5 | 73.3 | +7.2 | 11/12 | 0.00146 | 6/6 |
| Chinese | SCVAE-fixed (strict) | 77.4 | 69.5 | +7.8 | 12/12 | 0.000488 | 6/6 |
| Chinese | SCVAE released, **test-label-selected** | 94.9 | 90.9 | +4.0 | 10/12 | 0.00244 | 6/6 |
| Chinese | simple model, EEG (balanced acc., `clip_heldout_emotion.md`) | 75.9 | 69.5 | +6.4 | | | |
| German | SCVAE released + BN fix (strict) | 59.1 | 47.7 | +11.4 | 8/8 | 0.00781 | 6/6 |
| German | SCVAE-fixed (strict) | 57.2 | 42.4 | +14.8 | 8/8 | 0.00781 | 6/6 |
| German | SCVAE released, **test-label-selected** | 82.8 | 75.1 | +7.6 | 8/8 | 0.00781 | 6/6 |
| German | simple model, EEG (balanced acc., `clip_heldout_emotion.md`) | 58.5 | 44.7 | +13.8 | | | |
| French | SCVAE released + BN fix (strict) | 68.2 | 59.3 | +8.9 | 5/8 | 0.25 | 6/6 |
| French | SCVAE-fixed (strict) | 61.3 | 54.3 | +7.0 | 8/8 | 0.00781 | 6/6 |
| French | SCVAE released, **test-label-selected** | 84.7 | 80.2 | +4.5 | 5/8 | 0.109 | 6/6 |
| French | simple model, EEG (balanced acc., `clip_heldout_emotion.md`) | 59.7 | 55.8 | +3.9 | | | |

Average drop over the cultures with results: SCVAE released + BN fix (strict) +9.2; SCVAE-fixed (strict) +9.9; SCVAE released, **test-label-selected** +5.4. Predicted: ≈ +8 (strict models), ≈ +15 (test-label-selected).

## Part 3 — German ↔ French: shared vs other films (SCVAE-fixed ensemble, strict)

Balanced accuracy (%) over negative and positive windows (neutral has no shared film).

| Train → Test | Shared films | Other films | Shared − other | People higher on shared | Wilcoxon p | Runs complete |
|---|---:|---:|---:|---|---:|---|
| German → French | 57.2 | 53.9 | +3.3 | 4/8 | 0.945 | 3 seeds, 8/8 people |
| French → German | 61.3 | 51.0 | +10.3 | 5/8 | 0.547 | 3 seeds, 8/8 people |

Predicted: shared ≈ 5 points above other films.
