# Per-film positive-vs-negative accuracy — design (5 November 2026)

*Written before computing. Exploratory and descriptive: the question was formed after seeing the 3-day pooling
result (28 October), so this is not a pre-registered test, only a pre-written prediction. No training; saved
probabilities only. No target label is used for any choice; labels are read only to score.*

**Question:** "Did this person feel positive or negative about this film?", a different task from window-level
3-class recognition, reported as such.

**Models (fixed in advance):** the two-culture models, test culture unseen.
- **Primary:** NormFusion v3 fair, 5-seed probability mean (cache `results/xc_v3_combination/`).
- **Secondary:** SCVAE-XC `eye_2src`, 3-seed probability mean.
- **Context:** the same for one culture (6 directions).

**Decision per item:** mean of the model's 3-class probabilities over the item's windows; positive if
mean p(positive) > mean p(negative). Positive and negative films only; neutral films are excluded.

**Levels:**
1. Window: each positive/negative window separately.
2. Film, one day: one answer per (person, recording day, film).
3. Film, 3 days pooled: one answer per (person, film), windows of all 3 days together.

Also reported: 3-class versions of levels 2 and 3 (argmax of the mean probability, all films), so nothing is hidden.

**Metric:** accuracy % per test person, mean over people, averaged over the 3 targets. Also the total number of
items, since there are few per person (10/12/14 films per person for Chinese/German/French).

**Prediction (v3, two cultures):** level 1 78, level 2 85, level 3 90. 3-class film, 3 days: 78.

Script: `src/experiments/per_film_accuracy.py` → `results/tables/per_film_accuracy_summary.md`.

## Result (5 November 2026)

v3, two cultures (mean over the 3 targets): window pos/neg **79.3**, film one day **78.3**, film 3 days **84.4**;
3-class film 3 days 78.5. Prediction 78 / 85 / 90 / 78: window and 3-class right, per-film levels too high.
German + French → Chinese reaches **92.5** (film, 3 days, pos/neg; 120 items); German 80.2, French 80.4.
SCVAE-XC: 76.3 / 77.2 / 83.0. One culture, v3: 76.1 / 75.3 / 80.9 (French → Chinese 95.8).

Pooling one day's windows per film adds nothing over windows (78.3 vs 79.3); pooling the 3 days adds +5 to +6:
the errors are day-specific, consistent with 28 October.

**Caveat — shared films.** The two-culture models were trained with all films, including the 5 German/French
shared films (3 negative, 2 positive). For German/French targets, film 3 days pos/neg on shared vs unseen films:
v3 German 82.5 (40) vs 78.6 (56), French 80.0 (40) vs 80.6 (72); SCVAE-XC German 85.0 vs 76.8, French 85.0 vs
81.9. Small advantage for seen films. The Chinese target has no shared positive/negative film, so 92.5 is on
unseen films (neutral films excluded here). Summary: `results/tables/per_film_accuracy_summary.md`.
