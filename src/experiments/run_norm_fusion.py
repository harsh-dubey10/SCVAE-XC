"""Run NormFusion across cultures: fair result, ablations, and per-candidate test scores.

For each (source, target, seed) and setting (emotion; emotion_newfilms =
German <-> French without the 5 shared films in training), every variant is
written to results/norm_fusion/<target>/<setting>__from_<source>/seed_<s>/<variant>.json.

  fair          film-robust selection of (C_eeg, C_eye, w), then self-training
  no_selftrain  fair without self-training
  eeg_only      w = 1;   eye_only  w = 0
  lr_only       experts without the MLPs
  early_fusion  one expert on the concatenated features
  person_cv     selection by person-only cross-validation
  no_norm       without per-recording normalisation

The fair record also keeps the confusion matrix of every candidate on every
test person (the 80 fusion candidates without self-training plus the fair
choice with it), for the paper's-rules comparison. Test labels are used only
to score.

    python -m src.experiments.run_norm_fusion --jobs 4
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, f1_score

from src.baselines.norm_fusion import candidate_probas, load_culture, select_config, self_train
from src.experiments.train_dcca_am import write_json

VARIANTS = ["fair", "no_selftrain", "eeg_only", "eye_only", "lr_only", "early_fusion", "person_cv", "no_norm"]
CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
SEEDS = [12, 34, 56]


def score(true, proba, subjects):
    pred = proba.argmax(axis=1)
    return {str(s): {"accuracy": float((pred[subjects == s] == true[subjects == s]).mean()),
                     "macro_f1": float(f1_score(true[subjects == s], pred[subjects == s], average="macro")),
                     "confusion": confusion_matrix(true[subjects == s], pred[subjects == s], labels=range(3)).tolist()}
            for s in np.unique(subjects)}


def _record(cfg, cv_score, added, proba, target):
    return {"config": list(cfg), "cv_score": cv_score, "added": added,
            "per_subject": score(target.y, proba, target.subjects)}


def run_direction(source, target, seed, variants, raw_source=None, raw_target=None):
    records = {}
    adapt = lambda cfg, use_mlp=True, early=False, src=source, tgt=target: self_train(
        src.x, src.y, tgt.x, cfg, seed, use_mlp, early)
    if {"fair", "no_selftrain", "eeg_only", "eye_only"} & set(variants):
        best, scores = select_config(source, seed, use_mlp=True, early=False)
        if "fair" in variants or "no_selftrain" in variants:
            candidates = candidate_probas(source.x, source.y, target.x, seed, use_mlp=True, early=False)
            if "no_selftrain" in variants:
                records["no_selftrain"] = _record(best, scores[best], [], candidates[best], target)
            if "fair" in variants:
                proba, added = adapt(best)
                records["fair"] = _record(best, scores[best], added, proba, target)
                table = {str(tuple(cfg)): p for cfg, p in candidates.items()}
                table["fair_selftrained"] = proba
                records["fair"]["candidates"] = {
                    key: {s: v["confusion"] for s, v in score(target.y, p, target.subjects).items()}
                    for key, p in table.items()}
        for name, w in (("eeg_only", 1.0), ("eye_only", 0.0)):
            if name in variants:
                cfg = max((c for c in scores if c.w == w), key=scores.get)
                proba, added = adapt(cfg)
                records[name] = _record(cfg, scores[cfg], added, proba, target)
    for name, kwargs in (("lr_only", dict(use_mlp=False, early=False)), ("early_fusion", dict(use_mlp=True, early=True)),
                         ("person_cv", dict(use_mlp=True, early=False, mode="person"))):
        if name in variants:
            cfg, scores = select_config(source, seed, **kwargs)
            proba, added = adapt(cfg, kwargs["use_mlp"], kwargs["early"])
            records[name] = _record(cfg, scores[cfg], added, proba, target)
    if "no_norm" in variants:
        cfg, scores = select_config(raw_source, seed, use_mlp=True, early=False)
        proba, added = adapt(cfg, src=raw_source, tgt=raw_target)
        records["no_norm"] = _record(cfg, scores[cfg], added, proba, raw_target)
    return records


def job(source, target, seed, setting, output_root, skip_existing):
    torch.set_num_threads(1)
    variants = VARIANTS if setting == "emotion" else ["fair", "no_selftrain"]
    out = output_root / target / f"{setting}__from_{source}" / f"seed_{seed}"
    if skip_existing and all((out / f"{v}.json").exists() for v in variants):
        return f"skip {setting} {source}->{target} seed {seed}"
    src = load_culture(source, drop_shared=setting == "emotion_newfilms")
    tgt = load_culture(target)
    raw = dict(raw_source=load_culture(source, normalise=False), raw_target=load_culture(target, normalise=False)) \
        if "no_norm" in variants else {}
    records = run_direction(src, tgt, seed, variants, **raw)
    for name, record in records.items():
        write_json(out / f"{name}.json", {**record, "variant": name, "source": source, "target": target,
                                          "seed": seed, "setting": setting})
    fair = np.mean([v["accuracy"] for v in records["fair"]["per_subject"].values()])
    return f"{setting} {source}->{target} seed {seed}: fair {100 * fair:.1f}% config {records['fair']['config']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/norm_fusion"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    calls = [(s, t, seed, "emotion") for s, t in itertools.permutations(CULTURES, 2) for seed in args.seeds]
    calls += [(s, t, seed, "emotion_newfilms") for s, t in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER"))
              for seed in args.seeds]
    for line in Parallel(n_jobs=args.jobs)(delayed(job)(*call, args.output_root, args.skip_existing) for call in calls):
        print(line, flush=True)


if __name__ == "__main__":
    main()
