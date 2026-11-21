# Eye-feature confound check — design (29 October 2026)

*Written before any run.* Eye features add +7 to +12 points to 3-class accuracy but ~0 to positive-vs-negative
(28 October). Question: is the eye gain emotion, or film recognition through the pupil's response to screen
brightness?

## Feature groups

The 33 BCMI eye features carry no names in the released files. Columns 0–11 form one block with r = 0.9–1.0 among
themselves and values unlike the rest; this matches the standard 12 SEED-series pupil features (pupil diameter X/Y:
DE in 4 bands, mean, std). **Inferred from the data, not documented**: `pupil` = columns 0–11, `movement` = 12–32
(dispersion, fixation, saccade, blink and event statistics).

## Analyses (all fair; two training cultures → the third, every test film unseen as in the film learning curve)

StandardScaler + logistic regression, class-balanced, C = 0.01, all source films (shared German/French films kept
once, never a test film). Feature sets: `eeg`, `eeg+pupil`, `eeg+movement`, `eeg+eye` (all 33).

1. **3-class accuracy** per feature set (mean over test people, then over the 3 targets).
2. **Positive vs negative**: the same, trained and tested on positive and negative windows only.
3. **Film identification** (luminance proxy): within each culture, predict which film a window comes from
   (15/18/21 classes), 3 person-held-out folds, features `pupil`, `movement`, `eeg` alone. Chance ≈ 5–7%.

## Prediction

3-class: eeg 55, eeg+pupil 60, eeg+movement 57, eeg+eye 62. Positive vs negative: pupil adds < 1 point.
Film identification: pupil > movement.

## Decision rule

- **Pupil confound → drop pupil:** (eeg+pupil − eeg) ≥ 2/3 of (eeg+eye − eeg) in 3-class, **and** pupil adds < +1 in
  positive vs negative, **and** pupil film-ID > movement film-ID. Then the reportable models use `eeg+movement`.
- **Drop pupil (cheap) also if** eeg+movement ≥ eeg+eye − 1.5 in 3-class (pupil adds little beyond movement).
- **Pupil carries emotion:** pupil adds ≥ +2 in positive vs negative. Keep it, report with the caveat.
- Otherwise inconclusive; report all numbers.

Script: `src/experiments/run_eye_confound_check.py`; table `results/tables/eye_confound_summary.md`.

## Result (29 October 2026, laptop, 19 s)

**Pre-registered decision: INCONCLUSIVE.** 3-class: eeg 55.3, eeg+pupil 62.4, eeg+movement 55.7, eeg+eye 62.2 — the
pupil supplies the whole eye gain (+7.1 of +6.9); movement adds nothing. Positive vs negative: eeg 73.1, eeg+pupil
74.5 (+1.4: between the confound bar < 1 and the emotion bar ≥ 2), eeg+movement 69.2 (−3.9), eeg+eye 72.5.
Film identification (chance 5–7%): pupil 36.9, movement 41.5, EEG 34.1 — movement identifies films best, so the
"pupil is the film-recognition channel" condition fails.

Interpretation (after the fact): the pupil mainly separates emotional from neutral films (3-class +7.1 vs
positive/negative +1.4), consistent with pupil dilation tracking arousal, a well-documented effect; a brightness
difference between neutral and emotional films cannot be excluded without film luminance. Movement features carry
film content, not emotion: they add nothing to 3-class and hurt positive vs negative. Exploratory recommendation,
to be confirmed in a pre-registered run: use eeg+pupil and drop the movement features (equal 3-class accuracy,
+2.0 on positive vs negative, fewer film-specific features).
