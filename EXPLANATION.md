# Lost in Normalization: What Really Drives Cross-Cultural EEG Emotion Recognition

*A plain-language guide to this project: what we are doing, why, what we have found so far, and what comes next.*

> **Status note (1 October 2026):** sections 4 and 7 now use 3-seed runs from
> the IITGN cluster (section 7 is partly complete). Other numbers are still
> **preliminary** (mostly one seed, laptop runs). The verification plan is in
> section 8.

---

## Proposal

**Research question:** what actually makes EEG emotion recognition transfer to
new people and new cultures, once it is evaluated correctly (without using the
test labels)?

**Aim:** We provide a leakage-free re-evaluation of cross-cultural EEG emotion
recognition and analyse how **person-specific normalisation**, **model
complexity**, **latent alignment** and **multimodal information** affect
generalisation across Chinese, German and French populations.

**Hypothesis:** simple person-specific normalisation contributes more
consistently to cross-cultural generalisation than complex alignment or
multimodal models.

**Conclusion we would state only if verified experiments support it:** "Our
results show that simple person-specific normalisation contributes more
consistently to cross-cultural generalisation than complex alignment or
multimodal models."

**What is new here:** not normalisation itself (per-subject normalisation is
used in the EEG literature), and not a new network. The contribution is the
**controlled, leakage-free comparison across three cultures** that separates
the effects of evaluation protocol, normalisation, model complexity, alignment
and eye-tracking information, plus an explanation of *why* a
published model (SCVAE-Net) fails across cultures.

**The story, in four steps:**

1. **Leakage audit:** how much do published numbers depend on choosing
   settings with the test labels? (Section 3.)
2. **Normalisation analysis:** how much does standardising each person's data
   with their own statistics help simple models? (Section 6.)
3. **SCVAE failure analysis:** why does a complex published model fail across
   cultures, and can it be repaired? (Section 7.)
4. **Multimodal analysis:** do brain signals or eye movements
   transfer better across cultures, and does normalisation help both?
   (Section 6, preliminary.)

---

## 1. The problem in simple words

- People watch short film clips chosen to make them feel sad, neutral or happy,
  while a 62-electrode cap records their brain activity.
- The recording is cut into **4-second windows**. Each window becomes a small
  table of numbers: 62 electrodes × 5 frequency bands (delta, theta, alpha,
  beta, gamma). These numbers are called **DE features** (differential
  entropy); think of them as "how much energy each brain region has in each
  rhythm".
- A model learns to map that table to one of the three emotions.

**The hard part:** every brain is different. A model trained on people A–N
often does badly on a new person O. It does even worse on a person from another
country, who watched different films, possibly in a foreign language.

---

## 2. The datasets

| Dataset                       | Who     | Subjects     | Sessions                              | Notes                                                                      |
|-------------------------------|---------|--------------|---------------------------------------|----------------------------------------------------------------------------|
| **SEED**                      | Chinese | 15           | 3 each                                | Chinese films. Used for our SCVAE-Net findings (section 4).                |
| **SEED multimodal** (Chinese) | Chinese | 12 of the 15 | 3 each                                | The Chinese side of the cross-culture work: the same subset the paper used |
| **SEED-GER**                  | German  | 8            | 20 in total (subjects 5–8 did only 2) | English-language films (no German versions existed)                        |
| **SEED-FRA**                  | French  | 8            | 3 each                                | French films                                                               |

All were recorded by the same lab (BCMI, Shanghai Jiao Tong University) with
the same 62-channel cap. The Chinese multimodal, German and French features
come from the **same processing pipeline**, so differences between them are
not caused by processing. All three also include eye-tracking data (33
eye-movement features per window), analysed in section 6.

---

## 3. What "fair testing" means here

Imagine a student who sees the exam answers while studying. Their score says
nothing about what they learned. The same thing happens in machine learning
when the **test person's labels** are used to make choices during training,
such as "stop training at the epoch where the test accuracy was highest".

- **Leaky protocol:** try many settings, keep whichever scores best *on the test
  person*. The number looks great but is optimistic.
- **Strict protocol (ours):** every choice is fixed in advance. The test
  person's labels are used **once**, at the very end, to compute the score.

We found this kind of leakage in **both** papers we build on:

1. **SCVAE-Net** (the model we reproduce): its released code keeps the
   checkpoint with the best *test* accuracy.
2. **Liu et al. 2022** (the cross-culture paper, our baseline): its released
   code picks the SVM setting, the number of neighbours for KNN, the DNN's
   training epoch (out of 15,000) and its learning rate (out of 20) by *test*
   accuracy. **Caveat:** the released scripts are the Chinese
   *subject-dependent* versions. The paper does not say how settings were
   chosen for the cross-subject and cross-culture experiments, so we cannot
   yet claim those numbers were selected the same way.

**How much could it matter? (preliminary)** We re-ran an approximation of the
paper's linear-SVM setup (4-second windows and only its coarse C grid, one
run) twice:
once picking its setting by test accuracy (their way) and once picking it from
the training people only (fair). The difference is the pure leakage effect:

| Setting                                        | Leakage (points) |
|------------------------------------------------|-----------------:|
| Within one culture (Chinese / German / French) |  1.2 / 2.7 / 3.7 |
| Across cultures, most directions               |        1.2 – 3.2 |
| German → Chinese                               |              7.7 |
| German → French                                |             14.0 |

So leakage is **small within a culture** but can be **large across cultures**,
where a setting tuned on one culture transfers poorly and peeking at the test
culture helps most. On average it is worth about 4 points. That is real and
worth reporting, but it is **not** the main reason the paper's numbers were
higher than our first baselines. The main reasons are normalisation (section
6) and the window length: the paper's EEG-only models used 1-second windows,
while we used 4-second ones. We checked this with a matched 1-second
comparison: 4 seconds is actually better (section 6).

---

## 4. What we have already found (on Chinese SEED)

We ran SCVAE-Net strictly on 135 test runs (3 sessions × 3 random seeds × 15
held-out people), all on the IITGN cluster (`results/main_sweep/`). Accuracy
is averaged per person first; the gains are paired over the 15 people.

| What we ran                                              |      Accuracy | What it shows                                                                  |
|----------------------------------------------------------|--------------:|--------------------------------------------------------------------------------|
| SCVAE-Net, strict                                        |     **66.5%** | 7 of 135 runs "collapsed": the model predicted almost only one emotion         |
| + **BatchNorm fix** (recomputed on training data)        |     **74.3%** | no collapses; +7.8 points (95% CI +5.9 to +10.0), all 15 people improve        |
| BatchNorm recomputed on the test person's data instead   |         75.4% | +8.9; only +1.0 over the fix above (p = 0.06)                                  |
| Decaying (cosine) learning rate instead of the fix       |         73.0% | +6.5, no collapses (14 of 15 people improve)                                   |
| Cosine learning rate + BatchNorm fix                     |         72.5% | no better than either alone                                                    |
| Released code's protocol (best epoch by test accuracy)   |         93.1% | +18.8 over the fixed model: what peeking at the test labels is worth here      |
| + uncertainty-weighted alignment (our original idea)*    |         74.6% | no better than the fix alone                                                   |
| SVM / MLP baselines*                                     | 50.2% / 53.2% | far below                                                                      |

\* Earlier single-machine runs (Mac GPU), not yet repeated on the cluster.
Those earlier runs gave 65.0% → 75.9% (+10.9) for the first two rows: the
direction is the same, but single runs are noisy (fold-level agreement
between the two machines is only r = 0.3–0.6), so the cluster numbers above
replace them.

**What is BatchNorm, and what went wrong?** Inside the network, a *BatchNorm*
layer rescales the signals so they have a stable range. During training it
keeps a running average of what "normal" looks like and uses that average at
test time. Because the learning rate never decreases, the network keeps
changing right to the end, and the running average falls behind: it describes
an older version of the network. At test time the network is fed a stale idea
of "normal", and some runs collapse. **Re-computing that average once after
training** (a few seconds, no labels needed) fixes it. A decaying (cosine)
learning rate also prevents it.

The surprise: the network was not failing because people's brains differ (the
usual explanation). It was failing because of a bookkeeping error in
normalisation.

---

## 5. The new part: cross-culture

We now ask the same question across cultures.

**Two test settings (both copied from Liu et al. so we can compare):**

- **Within culture:** leave one person out, train on the rest of the same
  culture, test on that person (all of their sessions).
- **Across cultures:** train on everyone from one culture (for example Chinese)
  and test on each person from another (for example German). There are 6
  directions.

**The factors we compare:** every person, and every culture, has a different
"baseline level" in their brain signals. We test how far **normalising each
person's recording with its own statistics** (no labels needed) closes that
gap, and compare it with heavier approaches: a complex model (SCVAE-Net),
alignment losses that pull cultures together, class-aware alignment, and
self-training on the new person's unlabelled data.

---

## 6. Cross-culture results so far (simple baselines, strict; preliminary)

Simple models under the strict protocol, run on a laptop (29 September 2026),
4-second windows, EEG only. "Normalised" means every recording was
standardised with its own statistics (no labels used). The paper's numbers are
for reference only: they were picked using the test labels.

Accuracy in %, averaged over test people:

| Setting | Train → Test      | Best model, not normalised | RBF SVM, norm. | Linear SVM, norm. | MLP, norm. | Paper SVM* | Paper DNN* |
|---------|-------------------|---------------------------:|---------------:|------------------:|-----------:|-----------:|-----------:|
| within  | Chinese → Chinese |                       55.3 |       **79.7** |              78.7 |       78.9 |       72.6 |       82.8 |
| within  | German → German   |                       48.3 |       **62.4** |              59.4 |       57.2 |       55.6 |       65.9 |
| within  | French → French   |                       41.0 |       **66.5** |              57.1 |       56.8 |       50.1 |       64.2 |
| cross   | Chinese → German  |                       42.6 |           44.1 |          **46.2** |       44.5 |       47.9 |       55.3 |
| cross   | Chinese → French  |                       42.8 |           52.0 |              52.2 |   **52.5** |       46.8 |       57.3 |
| cross   | German → Chinese  |                       47.2 |           61.0 |              62.4 |   **63.6** |       47.4 |       64.3 |
| cross   | German → French   |                       42.4 |           54.2 |          **60.5** |       55.5 |       47.7 |       58.9 |
| cross   | French → Chinese  |                       47.1 |       **64.3** |              62.3 |       62.4 |       52.3 |       66.2 |
| cross   | French → German   |                       43.6 |           51.4 |          **58.3** |       52.7 |       48.9 |       60.1 |

\* Selected using the test labels (optimistic). The linear SVM is the paper's
own classifier, but here its setting is chosen from the training people only.

The full table (± spread, macro-F1, collapsed runs) is in
`results/tables/crossculture_summary.csv`.

**What this suggests (preliminary, pending significance tests):**

1. **Normalisation gives the largest gains we have seen.** Across 27 model ×
   setting combinations, the mean gain is positive in all 27 and ranges from
   **+1.9 to +27.5 points**; its 95% CI excludes zero in 26 of 27, and in most
   settings every test person improves. Collapsed runs drop to zero.
2. **With normalisation, fair simple models reach or exceed the paper's
   reported SVM numbers**: our mean is higher in 8 of 9 settings, and our 95%
   CI lies clearly above their number in 7 of 9. Our CI is **never** above the
   paper's DNN. This is *not* a like-for-like comparison: the paper used
   1-second windows, a different pipeline, and settings partly chosen with
   test labels.
3. **Some directions are much harder.** Models trained on Chinese data transfer
   worst; German/French → Chinese transfers well (61–64%). The paper reports
   the same pattern.
4. **Window length does not explain the differences.** With normalisation,
   1-second windows are never significantly better than 4-second ones (4 s has
   the higher mean in 8 of 9 settings, significantly in 1).

**A statistical limit to keep in mind:** with only 8 German or 8 French people,
a per-setting test cannot reach p < 0.008 even if every person improves, so
after correcting for many tests those settings cannot be "significant" whatever
the effect size. We therefore report effect sizes, 95% CIs and wins per person
as the main evidence. Full statistics: `results/tables/verification_report.md`.

**How to read this table:**
- *Accuracy* is averaged over test people. Chance level is about 33%; always
  guessing "negative" gives about 41% on German and French and about 34% on
  Chinese.
- *Macro-F1* (in the full table) is like accuracy, but it punishes a model that
  ignores one of the emotions.
- *Collapsed* (in the full table) counts runs where one emotion took ≥90% of the
  predictions.

### Eye movements and EEG + eye (strict; preliminary)

The same simple models, run on the 33 eye-movement features of the
multimodal datasets (pupil size, blinks, fixations, saccades, per 4-second
window), and on EEG and eye features joined together ("EEG + eye"). Laptop
runs (30 September 2026); the SVMs are deterministic and the MLP has 3 seeds.

Accuracy in %, best of the three normalised models per cell, averaged over
test people (95% CI in brackets):

| Setting | Train → Test      | EEG               | Eye               | EEG + eye             |
|---------|-------------------|-------------------|-------------------|-----------------------|
| within  | Chinese → Chinese | 79.7 [75.6, 83.4] | 74.5 [68.6, 80.6] | **87.0** [83.6, 90.3] |
| within  | German → German   | 62.4 [53.7, 71.4] | **83.7** [78.4, 88.5] | 79.6 [73.9, 86.2] |
| within  | French → French   | 66.5 [62.2, 71.6] | **80.7** [75.2, 85.7] | 77.8 [74.3, 81.4] |
| cross   | Chinese → German  | 46.2 [40.6, 52.5] | 50.6 [44.7, 56.8] | **54.1** [49.2, 59.6] |
| cross   | Chinese → French  | 52.5 [48.0, 57.9] | 54.4 [47.3, 60.1] | **58.3** [54.3, 62.9] |
| cross   | German → Chinese  | 63.6 [59.3, 67.9] | 47.2 [43.8, 50.8] | **66.4** [61.6, 71.2] |
| cross   | German → French   | 60.5 [54.5, 66.5] | 62.1 [57.9, 66.4] | **65.4** [59.9, 71.2] |
| cross   | French → Chinese  | 64.4 [60.4, 68.4] | 54.9 [47.9, 61.6] | **72.1** [66.8, 76.8] |
| cross   | French → German   | 58.3 [50.1, 67.3] | **68.8** [62.8, 73.9] | 65.0 [59.6, 70.1] |

Picking the best of three models per cell is a small selection effect. Every
model, the per-person tests and the unnormalised runs are in
`results/tables/modality_report.md`
(`python -m src.experiments.compare_modalities`).

**What this suggests (preliminary, same caveats as above):**

1. **Normalisation helps eye features just as much.** Across 54 model ×
   setting combinations for eye and EEG + eye, the mean gain is positive in
   all 54 (**+3.0 to +30.6 points**), with a 95% CI above zero in 50 of 54.
2. **Within a culture, eye movements alone are strong.** For German and French
   they beat EEG by 10 to 27 points, with the CI above zero for every model.
   For Chinese they are 4 to 7 points worse.
3. **Across cultures, eye movements do not transfer better than EEG.** Eye
   accuracy falls further than EEG accuracy when moving to another culture in
   **all 18** model × direction pairs (on average 13.8 points more). Eye alone
   is clearly worse than EEG when testing on Chinese people (7 to 21 points
   below), about equal when testing on French people, and clearly better
   only for French → German (CI above zero for 2 of 3 models).
4. **Combining helps most of the time.** EEG + eye beats normalised EEG in
   25 of 27 model × setting pairs (within: 9 of 9, +12.4 points on average;
   across: 16 of 18, +5.8 points, CI above zero in 14). The exception is
   German → Chinese, where the linear SVM and MLP lose 1 to 3 points.
5. **Normalisation still matters more than adding eye data.** Without
   normalisation, EEG + eye is poor (41 to 61%); normalisation adds 6 to 31
   points to it, while adding eye data to normalised EEG adds −2.5 to +20.
   This is consistent with the hypothesis, though the two effects are not
   directly comparable.

**Compared with Liu et al. 2022 (eye and EEG + eye).** The paper reports eye
and fusion results for all three cultures (its tables 6–9). For eye data the
windows (4 s) and the 33 features are the same as ours, so this comparison is
closer to like-for-like than the EEG one. Full table:
`results/tables/paper_modality_comparison.md`
(`python -m src.experiments.compare_paper_modalities`).

| Comparison (paper vs ours, 9 settings)        | Ours not normalised | Ours normalised                    |
|-----------------------------------------------|---------------------|------------------------------------|
| Eye, linear SVM                               | below in 8 of 9     | level in 6, above in 2, below in 1 |
| EEG + eye, linear SVM vs the paper's Concat   | below in 8 of 9     | level in 6, above in 2, below in 1 |
| Eye, MLP vs the paper's DNN                   | below in 9 of 9     | level in 4, below in 5 (all cross) |

("Level" means the paper's number lies inside our 95% CI.)

1. **Normalisation is what reaches the paper's numbers.** Without it our fair
   models are far below the paper. With it they match the paper's linear SVM
   and its concatenation fusion in almost every setting.
2. **The paper's eye DNN stays ahead across cultures** (5 of 6 directions,
   by 9.5 to 12 points). Its DNN epoch was chosen by test accuracy, and the
   leakage audit shows this matters most across cultures, so the gap cannot
   be attributed to the DNN itself without a strict re-run.
3. **The paper's eye vs EEG pattern replicates.** Eye is better than EEG for
   German and French and worse for Chinese, with the same sign in 8 of 9
   settings (the exception, Chinese → French, differs by under 1 point on our
   side).
4. **Its best fusion (DCCA-AM, 74–84% across cultures) has no counterpart
   here.** It is up to 20 points above any simple model, including German →
   Chinese at 84.3%, above the paper's own within-Chinese EEG result.
   Reproducing it strictly would be the real test of "complex fusion beats
   normalisation".

**A stimulus caveat.** People in the same culture watched the same clips, and
pupil size follows screen brightness. To test this, we asked a model to tell
apart *clips of the same emotion* in people it had not seen. Both feature sets
do better than guessing (EEG 12 to 27%, eye 17 to 29%, against a majority rate
of 6 to 12%). Eye features do it 2 to 8 points better in every case
(`results/tables/clip_identity_check.md`). Part of the strong within-culture
eye result may therefore come from recognising the clip rather than the
emotion. Clip identity cannot help across cultures, which watched different
films, and that fits the larger drop seen for eye features there.


### The film confound, measured (strict; 1 October 2026)

**1. Emotion accuracy falls when the test films are new to the model.** Within
each culture (leave one person out, EEG + eye data, normalised, logistic
regression with fixed settings), we predicted emotion twice on the same test
windows with the same amount of training data: once when the training people
had watched the same films, and once when those films were left out of
training. The usual cross-subject protocol is the "seen" column.

| Culture | Modality  | Seen films | Unseen films |     Drop | People dropping |
|---------|-----------|-----------:|-------------:|---------:|----------------:|
| Chinese | EEG + eye |       84.9 |         69.2 | **15.7** |           12/12 |
| Chinese | EEG       |       75.9 |         69.5 |      6.4 |           10/12 |
| Chinese | eye       |       70.7 |         51.0 | **19.7** |           12/12 |
| German  | EEG + eye |       72.2 |         54.3 | **17.8** |             8/8 |
| German  | EEG       |       58.5 |         44.7 |     13.7 |             8/8 |
| German  | eye       |       80.5 |         65.7 | **14.8** |             8/8 |
| French  | EEG + eye |       75.4 |         64.7 | **10.7** |             8/8 |
| French  | EEG       |       59.7 |         55.8 |      3.9 |  7/8 (p = 0.20) |
| French  | eye       |       79.6 |         62.9 | **16.7** |             8/8 |

Every drop is positive, and all but French EEG are significant (Wilcoxon over
people, p ≤ 0.008). Eye features lose 15 to 20 points, EEG 4 to 14. So a
large part of the usual cross-subject "emotion" accuracy, especially from eye
movements, comes from having trained on the very films being tested.
`results/tables/clip_heldout_emotion.md`
(`python -m src.experiments.clip_heldout_emotion`). The drop is if anything
an underestimate: different excerpts of the same film (for example the
Chinese "World Heritage in China" clips) can fall in different folds.

**2. The film is recognisable across cultures, from eye movements.** German
and French participants watched five of the same FilmStim films. We trained a
model on one culture to tell which film a person of the other culture was
watching, within one emotion (so the emotion label cannot help). For *Dead
Man Walking* vs *A Perfect World* (both negative, chance 50%):

| Train → Test    | EEG + eye         | EEG              | Eye               |
|-----------------|-------------------|------------------|-------------------|
| German → French | 65.0 (p = 0.01)   | 55.3 (n.s.)      | **84.8** (p < 0.001) |
| French → German | 90.2 (p < 0.001)  | 42.7 (n.s.)      | **95.8** (p < 0.001) |

A control rules out session timing (*Dead Man Walking* is shown first in both
cultures): applied to the other negative films of the test culture, the eye
model calls late films such as *Dangerous Mind* "A Perfect World" only 8% of
the time, but it does call the German *The Piano* "A Perfect World" 90% of the
time. What transfers is therefore the visual character of the excerpt (for
example brightness and scene type), not the film as a story. The other film
sets (positive pair, all three negative films, all five) are at or below
chance, plausibly because the two cultures saw different excerpts of those
films (*Schindler's List*: 7 min vs 1.3 min).
`results/tables/film_identification.md`
(`python -m src.experiments.film_identification`).

**What this means:** eye-movement "emotion" accuracy is substantially
stimulus recognition, and the effect is visual enough to cross cultures for a
well-matched excerpt. EEG is affected less but not zero. Any emotion
recognition result evaluated on films the model has trained on should be read
with this in mind; this applies to the paper's results and to ours.

---

## 7. SCVAE-Net and self-training across cultures (preliminary, 1 seed)

> **Update (1 October 2026):** the SCVAE rows below come from the complete
> 3-seed cluster sweep (`results/paper_scvae/`, 504 records, all checked).
> They **overturn** the earlier single-seed picture: with the BatchNorm fix the
> released recipe does not collapse and is level with class-aware alignment.

We also tested whether more complex approaches add anything on top of
normalisation. Accuracy (%) per cross-culture direction:

| Train → Test      | SCVAE released recipe + BN fix | SCVAE class-aware alignment | Best normalised simple model |
|-------------------|-------------------------------:|----------------------------:|-----------------------------:|
| Chinese → German  |                       **48.3** |                        46.4 |                         46.2 |
| Chinese → French  |                       **55.0** |                        49.1 |                         52.5 |
| German → Chinese  |                           47.8 |                        55.3 |                     **63.6** |
| German → French   |                           54.9 |                        55.9 |                     **60.5** |
| French → Chinese  |                           57.4 |                        60.4 |                     **64.3** |
| French → German   |                           49.0 |                        49.8 |                     **58.3** |
| **Average**       |                           52.1 |                        52.8 |                     **57.6** |

The "best normalised simple model" is chosen per
direction after the fact, which flatters the baselines slightly.

**What we learned:**

- **No collapses with the BatchNorm fix.** None of the 168 released-recipe
  cross-culture runs collapsed, and its accuracy (52.1% on average) is level
  with class-aware alignment (52.8%). The earlier "~42%, emotions swapped or
  collapsed" figure could not be traced to a run in the repository (it
  probably lacked the BN fix) and is withdrawn.
- **Class-aware alignment is not a clear improvement.** It helps when
  Chinese is the target (+3 to +6 points) but hurts when Chinese is the
  source (−2 and −6 points).
- **SCVAE still trails the normalised simple models by about 5 points on
  average.** The gap is large when Chinese people are the test group (4 to 16
  points), but when trained on Chinese data the released recipe matches or
  beats them (Chinese → German, Chinese → French).
- **Self-training** (normalise, then learn from the model's own confident
  predictions on the new person) adds +1.9 points within Chinese (every one of
  12 people improved) but about 0 across cultures, so it is reported as an
  ablation, not as a method.
- **Within one culture**, SCVAE-Net (BN fix, 3 seeds on the cluster) never
  beats the normalised RBF SVM:

  | Culture | SCVAE + BN fix | Normalised RBF SVM | Normalised MLP |
  |---------|---------------:|-------------------:|---------------:|
  | Chinese |           70.2 |               79.7 |           78.9 |
  | German  |           57.6 |               62.4 |           57.2 |
  | French  |           59.2 |               66.5 |           56.8 |

  On Chinese it is about 9.5 points below the SVM, losing for 10 of 12 people
  (p = 0.002). On German it is level with the MLP (4 of 8 people) and about 5
  points below the SVM. On French it is about 7 points below the SVM (losing
  for all 8 people) and level with the MLP.
- **Per-recording normalisation does not help SCVAE, unlike every simple
  model.** It lowers SCVAE by 7.4 points on Chinese (3 of 12 people improve)
  and 11.1 points on German (0 of 8, p = 0.008), raises French by 3.9 (not
  significant), and brings back some collapsed runs (8 of 84). A plausible
  reason is that SCVAE already normalises internally (BatchNorm, MMD
  alignment), so the extra step removes information it relied on. This is a
  limit on the "normalisation helps" claim: it holds for the simple models,
  not for every architecture.

### SCVAE-fixed and SCVAE on unseen films (strict; 9 October 2026, 3 seeds)

Spec: `docs/superpowers/specs/2026-10-09-scvae-crossculture-improvement-design.md`.
Full tables: `results/tables/scvae_improve_summary.md` (cluster jobs 6260 and
6424; every run complete, 3 seeds, all people). Columns marked
*test-label-selected* choose the checkpoint by test accuracy, as the released
code does; they are a labelled comparison, not accuracy.

**Part 1: a fixed SCVAE across cultures does not beat the simple models.**
"SCVAE-fixed" adds per-person normalisation, shuffled target batches, weight
decay, input noise, a decaying learning rate and an epoch chosen on held-out
*source* people; one of 8 settings is chosen per test culture from the other
two cultures' transfers only, and 3 seeds are averaged.

| Train → Test      | Best simple model | SCVAE released | SCVAE-fixed | Released, *test-label-selected* |
|-------------------|------------------:|---------------:|------------:|--------------------------------:|
| Chinese → German  |              46.2 |           48.3 |    **47.5** |                            69.2 |
| Chinese → French  |              52.5 |           55.0 |    **48.6** |                            68.7 |
| German → Chinese  |              63.6 |           47.8 |    **58.5** |                            73.8 |
| German → French   |              60.5 |           54.9 |    **54.5** |                            70.8 |
| French → Chinese  |              64.3 |           57.4 |    **59.9** |                            77.5 |
| French → German   |              58.3 |           49.0 |    **50.6** |                            70.5 |
| **Average**       |          **57.6** |           52.1 |    **53.3** |                            71.7 |

- Pre-registered success (above 57.6% and at least 4 of 6 wins): **NOT MET**
  (53.3%, 1 win). Predicted 58%.
- SCVAE-fixed vs released: +2.0 points (95% CI −1.1 to +5.1, p = 0.22) over
  56 (direction, person) pairs; large when Chinese is the test set, negative
  for Chinese → French.
- Test-label selection adds +19.7 points (predicted total 72%, observed
  71.7%), almost exactly the +18.8 it adds within Chinese SEED (section 4).

**Part 2: within each culture, about 9–10 points of SCVAE's accuracy need the
test films in training.** Films are dealt into 3 groups that each hold every
emotion; each person is tested on one group, after training on the other
people either with the test films (seen) or without them (unseen), on the same
test windows.

| Culture | Model                            | Seen films | Unseen films |  Drop | People dropping |      p |
|---------|----------------------------------|-----------:|-------------:|------:|----------------:|-------:|
| Chinese | SCVAE released + BN fix          |       80.5 |         73.3 |  +7.2 |           11/12 |  0.001 |
| Chinese | SCVAE-fixed                      |       77.4 |         69.5 |  +7.8 |           12/12 | 0.0005 |
| Chinese | Released, *test-label-selected*  |       94.9 |         90.9 |  +4.0 |           10/12 |  0.002 |
| German  | SCVAE released + BN fix          |       59.1 |         47.7 | +11.4 |             8/8 |  0.008 |
| German  | SCVAE-fixed                      |       57.2 |         42.4 | +14.8 |             8/8 |  0.008 |
| German  | Released, *test-label-selected*  |       82.8 |         75.1 |  +7.6 |             8/8 |  0.008 |
| French  | SCVAE released + BN fix          |       68.2 |         59.3 |  +8.9 |             5/8 |   0.25 |
| French  | SCVAE-fixed                      |       61.3 |         54.3 |  +7.0 |             8/8 |  0.008 |
| French  | Released, *test-label-selected*  |       84.7 |         80.2 |  +4.5 |             5/8 |   0.11 |

- Strict models drop +9.2 (released) and +9.9 (fixed) on average; predicted +8.
- Test-label selection drops only +5.4 (predicted +15): picking the best
  checkpoint on the test person inflates seen and unseen films alike, by 17–27
  points on unseen films. The paper-style number is mostly peeking, and it
  survives on films the model never saw.
- On unseen films the strict SCVAE models are level with the simple model of
  `clip_heldout_emotion.md` (Chinese 69.5–73.3 vs 69.5, German 42.4–47.7 vs
  44.7, French 54.3–59.3 vs 55.8; the simple model's numbers are balanced
  accuracy, so the comparison is approximate).
- Seen and unseen training sizes are equal summed over the 3 groups; within a
  group they differ by one clip per emotion for Chinese (5 clips per emotion)
  and French (7), as in `clip_heldout_emotion.md`.

**Part 3: German ↔ French shared films (SCVAE-fixed).** The 5 films both
cultures watched score above the other films (+3.3 German → French, +10.3
French → German, balanced accuracy over negative and positive windows), but
neither difference is significant (p = 0.95 and 0.55, 8 people each).

---

## 8. What happens next

**Verification first (before any number is presented as a result):**

1. ✅ Done: every number recomputed from the raw per-person results, with
   per-person tests, CIs and Holm correction
   (`results/tables/verification_report.md`).
2. Re-run the leakage check with the paper's exact setup: its full C grid,
   its scaling and 1-second windows, over several runs.
3. Document exactly what is known about how Liu et al. chose their settings:
   the released code (subject-dependent scripts) picks them by test accuracy,
   but the procedure for the cross-subject and cross-culture settings is not
   stated. We should ask the authors.

**Then complete the analysis:**

4. ✅ Done: SCVAE-Net within each culture on all people (section 7).
   3-seed cluster runs of the SEED BatchNorm sweep (section 4) and the
   cross-culture SCVAE sweep (section 7) are complete.
5. ✅ Done (preliminary): eye-tracking analysis with simple models
   (section 6). Still open: removing brightness-driven pupil features to
   measure the clip effect.
6. ✅ Done: the SCVAE paper sweep, 3 seeds (section 7). 🔄 Running on the
   cluster: the leakage audit with 1 s and 4 s windows and the paper's full C
   grid.

---

## 9. Caveats to mention when presenting

- **Normalisation uses the test person's unlabelled recordings.** It never sees
  their answers, but it needs their data in advance ("transductive"). We apply
  it to every method equally and say so clearly. A stricter variant that only
  uses the first few minutes of a recording can be added.
- **"Culture" is mixed with other factors.** Each culture watched different
  films, the Germans watched English films, and all participants were students
  living in Shanghai. The careful wording is "cross-population transfer", with
  culture as one factor.
- **Small groups:** 8 German, 8 French and 12 Chinese people, so we report
  per-person results and effect sizes, not only p-values.

---

## 10. Where things are in the repository

| What                                      | Where                                                                                                                              |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------|
| Raw downloads (licensed, never committed) | `data/raw/SEED_CHN_MM`, `SEED_GER`, `SEED_FRA`                                                                                     |
| Converted data                            | `data/processed/SEED_CHN_MM`, `SEED_GER`, `SEED_FRA` (cross-culture); `SEED` (SCVAE-Net findings)                                  |
| Converter                                 | `src/preprocessing/convert_seed_crossculture.py`                                                                                   |
| Training (SCVAE-Net)                      | `src/experiments/train_scvae_reference.py` (`--split session / pooled / cross`, `--subject-norm recording`)                        |
| Simple baselines                          | `src/baselines/strict_baselines.py`; all runs: `src/experiments/sweep_crossculture_baselines.sh` (`NORM=recording` for normalised) |
| Leakage check                             | `src/baselines/liu_leakage_audit.py`                                                                                               |
| Results table                             | `results/tables/crossculture_summary.csv` (`python -m src.experiments.summarize_crossculture`)                                     |
| Side experiments (archived)               | `~/Documents/EEG-Emotion-SCVAE_archive/`                                                                                           |
| Technical details                         | `README.md`                                                                                                                        |

### Glossary

- **EEG:** brain activity recorded from electrodes on the scalp.
- **DE feature:** a number describing the energy of one electrode in one frequency band during a 4-second window.
- **Leave-one-subject-out:** train on everyone except one person, test on that person, and repeat for every person.
- **Source / target:** the people the model learns from, and the person or culture it is tested on.
- **BatchNorm:** a layer that rescales signals inside the network using running averages.
- **AdaBN:** re-estimating those averages on the new domain's data (no labels needed).
- **MMD alignment:** a training loss that pushes the source and target feature distributions to look alike.
- **Collapse:** the model predicts nearly the same emotion for everything.
- **Leakage:** using the test answers, directly or indirectly, to make training choices.
