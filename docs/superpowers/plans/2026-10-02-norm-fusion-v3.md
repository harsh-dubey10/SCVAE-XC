# NormFusion v3 Implementation Plan (short form)

> The user approved the spec and asked for direct implementation (2026-10-02), so this plan is a task list
> with interfaces, not full code. Executed inline, test-first, on branch `dcca-am-rerun`.

**Goal:** Build, test, time and run NormFusion v3 on the cluster: bagged EEG / eye / early experts, a plain
average ensemble, C per expert by v1's film-robust CV, membership by cross-culture validation in two-culture
settings, v1's self-training on the ensemble; 11 settings × 5 seeds; summary against v1 (same machine), v2,
DCCA-AM and the pre-registered prediction.

**Spec:** `docs/superpowers/specs/2026-10-02-norm-fusion-v3-design.md`

## Global Constraints

- `.venv/bin/python`; plain-assert test modules printing "... checks passed"; synthetic test features are
  float32 like the real data.
- Bags: 10; people `round(0.8 n)` per culture (culture = person id // `norm_fusion_v2.OFFSET`), without
  replacement, `default_rng(seed * 1000 + b)`; MLP seeds `seed + 100 * (b + 1) + i`, i = 0..2. Unbagged
  (no_bagging, C choice): all people, MLP seeds seed, seed + 1, seed + 2 (v1's).
- A bag's probability is computed exactly as v1's `expert_probas` (LR + sum of 3 MLPs, divided by 4), so one
  unbagged bag equals v1's expert bit for bit.
- Memberships in order: (eeg,), (eeg, eye), (eeg, early), (eeg, eye, early). Ensemble = plain mean.
- Seeds 12, 34, 56, 78, 90. Output `results/norm_fusion_v3/<target>/<setting>__from_<source>/seed_<s>/<variant>.json`.
- All computation on CPU; each parallel fit single-threaded; results independent of worker count.
- Cluster: `--gres=gpu:1`, 16 CPUs per task, array of 11 with at most 2 running.

## Tasks

1. **Model** — `src/baselines/norm_fusion_v3.py`, test `src/baselines/test_norm_fusion_v3.py`.
   Produces `bag_people(subjects, seed) -> list[np.ndarray]`, `bag_specs(subjects, seed, bagged) ->
   list[(people, seeds)]`, `fit_bag(x, y, eval_x, c, seeds) -> proba`, `member_probas(train, eval_x, members,
   cs, bags, n_jobs, extra_x=None, extra_y=None) -> {member: proba}`, `ensemble(probas, membership)`,
   `choose_cs(data, seed, n_jobs) -> ({expert: C}, scores)`, `choose_membership(culture_a, culture_b, seed,
   n_jobs) -> (membership, scores)`, `self_train(train, tgt_x, membership, cs, bags, n_jobs, first=None) ->
   (proba, added)`.
2. **Runner** — `src/experiments/run_norm_fusion_v3.py`, test `src/experiments/test_run_norm_fusion_v3.py`.
   `SETTINGS` (11), `VARIANTS`, `run_direction(sources: list[Culture], target, seed, n_jobs,
   keep_predictions=False)`, `job(task, seeds, n_jobs, output_root, skip_existing)`, CLI `--task/--all`.
   Tests: variants present, label-shuffle invariance per window, 1 vs 4 workers identical, `v1` variant
   equals v1's `run_direction`, cross-culture choice unchanged when the target is replaced.
3. **Summary** — `src/experiments/summarize_norm_fusion_v3.py` + test: main table with prediction, two-culture,
   new films, ablations, choices, clip-level, v1 cluster-vs-laptop check (flag > 2 points), success criterion.
4. **Cluster** — `cluster/norm_fusion_v3.sbatch`; time one run on the laptop, update the estimate, push,
   submit, pull, summarise, commit.
