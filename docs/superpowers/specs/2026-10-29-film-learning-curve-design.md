# Films-per-emotion learning curve — design (29 October 2026)

*Written before any run. Diagnostic study; decides whether requesting SEED-IV / SEED-V (many more films) is worth it.*

## Question

Every model plateaus at about 60–75% fair accuracy, and a person-specific model trained on the person's own clips
of the same session does no better on new clips (57–69%, exploratory, 28 October). Hypothesis: the bottleneck is the
number of **distinct films per emotion** seen in training (5–7 per culture), not people, cultures or the model.

## Setting

Two training cultures → the third (as SCVAE-XC step ④ / v3 two-culture). All source people are used; only the films
vary. Test: every person and every film of the target culture (per-recording normalisation as everywhere).

Each film counts once: the 5 German/French shared films (3 negative, 2 positive) are dropped from the French source
when French is a source (German + French → Chinese; Chinese + French → German) and from the German source for
Chinese + German → French. Films per emotion available (min over emotions): → Chinese 10, → German 9, → French 8.

## Arms

- **Curve:** F ∈ {1, 2, 3, 4, 6, 8, max} films per emotion, drawn at random from the pooled source films (stratified
  by emotion), 10 random draws per F (draw seeds 0–9). All windows of the drawn films.
- **Diversity vs quantity (fixed window budget):** for each draw, the window count of the F = 2 set; "concentrated"
  = that F = 2 set; "spread" = the same number of windows drawn uniformly at random from F = 8 films. Isolates film
  diversity from the amount of data.
- **Model:** StandardScaler + logistic regression, class-balanced, C = 0.01, fixed in advance; EEG+eye (343) and
  EEG only (310). Simple on purpose: the shape of the curve matters, not the level.

Metric: accuracy % per test person, mean over people, then mean ± sd over the 10 draws; averaged over the 3 targets.

## Prediction

EEG+eye: F = 1 ≈ 50, F = 4 ≈ 60, F = 8 ≈ 64, rising roughly linearly in log F with no plateau by F = 8.
Spread beats concentrated at the same window budget by ≥ 3 points.

## Decision rule

- **More films help** (request SEED-IV / SEED-V): accuracy at F = 8 − F = 4 ≥ +2 points (EEG+eye, 3-target mean)
  **and** spread − concentrated ≥ +2.
- **Plateau** (more films alone will not fix it): F = 8 − F = 4 < +1.
- Otherwise inconclusive; report the curve and the fitted slope per doubling of F.

Script: `src/experiments/run_film_curve.py`; output `results/film_curve/`, table `results/tables/film_curve_summary.md`.

## Result (29 October 2026, laptop, 39 s)

**Pre-registered decision (EEG+eye): MORE FILMS HELP.** Mean over the 3 targets: F = 1 50.1, 2 55.4, 3 57.0, 4 59.1,
6 60.9, 8 61.5 (F = 8 − F = 4 = +2.4; slope +3.8 points per doubling, but flattening: +1.8 from 4 → 6, +0.6 from
6 → 8; Chinese target peaks at F = 6). At a fixed window budget, spreading the windows over 8 films beats 2 films
by **+7.2** (62.6 vs 55.4) — film diversity, not data amount, drives accuracy. EEG only: same shape, 4–7 points
lower (F = 8 54.6; spread − concentrated +5.6). Summary: `results/tables/film_curve_summary.md`.
