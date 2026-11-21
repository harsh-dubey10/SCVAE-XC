# Cross-cultural calibration — implementation plan (short form)

> Spec approved 2026-10-04 (`docs/superpowers/specs/2026-10-04-calibration-design.md`); the user asked to go
> ahead and to have the tests explained step by step. Executed inline, test-first, on `dcca-am-rerun`.

**Data fact (checked 2026-10-04):** a clip id is the same film in every session (same emotion per id in all
sessions); per emotion 5 films (China), 6 (Germany), 7 (France). Session = recording id // 1000.

## Tasks and their tests

1. **Base probabilities** — `fair_probas(sources, target, seed, n_jobs, n_bags)` in
   `src/experiments/run_norm_fusion_v3.py` (v3 fair only, probabilities returned) and
   `src/experiments/run_calibration_base.py` (15 tasks: 6 cross-culture directions + 3 cultures × 3 same-culture
   groups; seeds 12, 34, 56; writes `results/calibration/base/.../seed_<s>.npz`).
   - T1.1 `fair_probas` predicts exactly what v3 `run_direction`'s fair variant predicts.
   - T1.2 same-culture groups: every person is a test person exactly once, never in training and test at once.
2. **Calibration** — `src/experiments/run_calibration.py`: pools, splits, person models, scoring.
   - T2.1 pools: deterministic per (draw, person), 2 films per emotion, only films the person watched,
     different draws differ.
   - T2.2 film-disjoint split: calibration and test never share a film id; calibration only from the person's
     first session and only that person's windows.
   - T2.3 identical test windows for N = 0, 1, 2 and for every film-disjoint variant.
   - T2.4 same-film variant shares films by construction, and its calibration windows are not test windows.
   - T2.5 N = 0 `combined` equals `base_only` exactly; `combined` = 0.5 base + 0.5 person exactly.
   - T2.6 modality variants use EEG columns 0:310 / eye 310:343 only.
   - T2.7 test labels are never used except to score: shuffling the labels of test windows leaves every
     prediction unchanged.
3. **Summary** — `src/experiments/summarize_calibration.py`: tables in the presentation order, culture gap,
   film-leak, modality, prediction; figure accuracy vs N.
   - T3.1 culture gap uses the same test people on both sides; prediction table matches the spec; calibrated
     numbers are labelled "with N calibration clips per person".
4. **Run** — time one base task on the laptop, run the base tasks (cluster or laptop), run calibration,
   summarise, commit.
