"""Run NormFusion v2 across cultures: fair result, ablations, v1 re-run and per-candidate test accuracy.

Settings: emotion (the 6 one-to-one directions), emotion_two_cultures (train on two cultures, test on the
third; source written "A+B"), emotion_newfilms (German <-> French without the 5 shared films in training).
Every variant is written to results/norm_fusion_v2/<target>/<setting>__from_<source>/seed_<s>/<variant>.json.

  fair          film-robust choice of (C_EEG, C_eye, fusion, k) over per-emotion late fusion and early fusion,
                with centred smoothing, then self-training on the smoothed probabilities
  no_smoothing  fair with k = 1
  causal        fair with trailing instead of centred smoothing
  v1_fusion     single weight w (v1's grid), no per-emotion weights, no early fusion; smoothing chosen
  no_selftrain  the fair choice without self-training
  v1            v1 itself (k = 1, single weight, no early fusion); reproduces results/norm_fusion

The fair record also keeps every test person's accuracy for each fair-grid candidate without self-training,
plus the fair choice with it, for the paper's-rules comparison. Test labels are used only to score.

    python -m src.experiments.run_norm_fusion_v2 --jobs 4
"""

import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.baselines.norm_fusion import load_culture
from src.baselines.norm_fusion_v2 import (VARIANT_KEEP, choose, combine, cv_scores, experts, fit_candidate,
                                          grid_probas, self_train, smooth)
from src.experiments.run_norm_fusion import score
from src.experiments.train_dcca_am import write_json

VARIANTS = ["fair", "no_smoothing", "causal", "v1_fusion", "no_selftrain", "v1"]
NEWFILMS_VARIANTS = ["fair", "no_selftrain", "v1"]
CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
SEEDS = [12, 34, 56]


def score_v2(target, proba):
    """v1's per-person window scores plus clip-level accuracy (majority vote of each recording's clip)."""
    scores = score(target.y, proba, target.subjects)
    pred = proba.argmax(axis=1)
    for s in np.unique(target.subjects):
        mine = target.subjects == s
        correct = []
        for r, c in sorted(set(zip(target.recordings[mine].tolist(), target.clips[mine].tolist()))):
            windows = mine & (target.recordings == r) & (target.clips == c)
            correct.append(np.bincount(pred[windows], minlength=3).argmax() == target.y[windows][0])
        scores[str(s)]["clip_accuracy"] = float(np.mean(correct))
    return scores


def candidate_table(source, target, seed, use_mlp, fair_proba):
    """Each test person's accuracy for every fair-grid candidate without self-training, then the fair choice with it."""
    keys, accuracy = [], defaultdict(list)
    people = {str(s): target.subjects == s for s in np.unique(target.subjects)}
    for cand, proba in grid_probas(experts(source.x, source.y, target.x, seed, use_mlp), target.recordings):
        if VARIANT_KEEP["fair"](cand):
            keys.append(str(tuple(cand)))
            correct = proba.argmax(axis=1) == target.y
            for s, mask in people.items():
                accuracy[s].append(round(float(correct[mask].mean()), 4))
    keys.append("fair_selftrained")
    correct = fair_proba.argmax(axis=1) == target.y
    for s, mask in people.items():
        accuracy[s].append(round(float(correct[mask].mean()), 4))
    return {"keys": keys, "accuracy": dict(accuracy)}


def run_direction(source, target, seed, variants, use_mlp=True):
    scores = cv_scores(source, seed, use_mlp)
    adapted = {}  # candidate -> (probabilities, added): variants that choose the same candidate share the run

    def adapt(cand):
        if cand not in adapted:
            adapted[cand] = self_train(source.x, source.y, target.x, target.recordings, cand, seed, use_mlp)
        return adapted[cand]

    records = {}
    for name in variants:
        cand = choose(scores, VARIANT_KEEP["fair" if name == "no_selftrain" else name])
        if name == "no_selftrain":
            proba = smooth(fit_candidate(source.x, source.y, target.x, cand, seed, use_mlp), target.recordings,
                           cand.k, cand.causal)
            added = []
        else:
            proba, added = adapt(cand)
        records[name] = {"config": list(cand), "cv_score": scores[cand], "added": added,
                         "per_subject": score_v2(target, proba)}
        if name == "fair":
            records[name]["candidates"] = candidate_table(source, target, seed, use_mlp, proba)
    return records


def load_source(source, setting):
    if "+" in source:
        return combine([load_culture(c) for c in source.split("+")])
    return load_culture(source, drop_shared=setting == "emotion_newfilms")


def job(source, target, seed, setting, output_root, skip_existing):
    torch.set_num_threads(1)
    variants = NEWFILMS_VARIANTS if setting == "emotion_newfilms" else VARIANTS
    out = output_root / target / f"{setting}__from_{source}" / f"seed_{seed}"
    if skip_existing and all((out / f"{v}.json").exists() for v in variants):
        return f"skip {setting} {source}->{target} seed {seed}"
    records = run_direction(load_source(source, setting), load_culture(target), seed, variants)
    for name, record in records.items():
        write_json(out / f"{name}.json", {**record, "variant": name, "source": source, "target": target,
                                          "seed": seed, "setting": setting})
    fair = np.mean([v["accuracy"] for v in records["fair"]["per_subject"].values()])
    v1 = np.mean([v["accuracy"] for v in records["v1"]["per_subject"].values()])
    return (f"{setting} {source}->{target} seed {seed}: fair {100 * fair:.1f}% (v1 {100 * v1:.1f}%) "
            f"config {records['fair']['config']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--output-root", type=Path, default=Path("results/norm_fusion_v2"))
    parser.add_argument("--settings", nargs="+", default=["emotion", "emotion_two_cultures", "emotion_newfilms"])
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    calls = []
    if "emotion" in args.settings:
        calls += [(s, t, seed, "emotion") for s, t in itertools.permutations(CULTURES, 2) for seed in args.seeds]
    if "emotion_two_cultures" in args.settings:
        calls += [("+".join(c for c in CULTURES if c != t), t, seed, "emotion_two_cultures")
                  for t in CULTURES for seed in args.seeds]
    if "emotion_newfilms" in args.settings:
        calls += [(s, t, seed, "emotion_newfilms") for s, t in (("SEED_GER", "SEED_FRA"), ("SEED_FRA", "SEED_GER"))
                  for seed in args.seeds]
    for line in Parallel(n_jobs=args.jobs)(delayed(job)(*call, args.output_root, args.skip_existing) for call in calls):
        print(line, flush=True)


if __name__ == "__main__":
    main()
