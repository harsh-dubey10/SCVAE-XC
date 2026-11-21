"""Run NormFusion v3 across cultures: fair result, ablations and a same-machine v1 reference.

Settings (11): the 6 one-to-one directions (emotion), the 3 two-culture settings (emotion_two_cultures,
source written "A+B"), and German <-> French without the 5 shared films in training (emotion_newfilms).
Every variant is written to results/norm_fusion_v3/<target>/<setting>__from_<source>/seed_<s>/<variant>.json.

  fair               bagged experts, C per expert by film-robust CV; membership fixed at all three experts
                     (one training culture) or chosen by cross-culture validation (two); self-training
  no_bagging         fair with one bag of all training people per expert (v1's expert)
  no_selftrain       fair's ensemble without self-training
  members_<m>        each fixed membership (eeg, eeg_eye, eeg_early, all), self-trained
  v1                 v1 itself (film-robust choice of C_EEG, C_eye, w; self-training), on this machine

The fair record also keeps each test person's accuracy for every membership without self-training, plus the
fair model, for the paper's-rules comparison. Test labels are used only to score.

    python -m src.experiments.run_norm_fusion_v3 --task 0 --jobs 16     # one setting, all seeds
    python -m src.experiments.run_norm_fusion_v3 --all --jobs 8
"""

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import torch

from src.baselines.norm_fusion import load_culture, select_config
from src.baselines.norm_fusion import self_train as v1_self_train
from src.baselines.norm_fusion_v2 import combine
from src.baselines.norm_fusion_v3 import (EXPERTS, MEMBERSHIPS, N_BAGS, bag_specs, choose_cs, choose_membership,
                                          ensemble, member_probas, self_train)
from src.experiments.run_norm_fusion_v2 import score_v2
from src.experiments.train_dcca_am import write_json

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
SEEDS = [12, 34, 56, 78, 90]
MEMBER_NAMES = {("eeg",): "eeg", ("eeg", "eye"): "eeg_eye", ("eeg", "early"): "eeg_early",
                ("eeg", "eye", "early"): "all"}
VARIANTS = ["fair", "no_bagging", "no_selftrain"] + [f"members_{n}" for n in MEMBER_NAMES.values()] + ["v1"]
SETTINGS = ([((s,), t, "emotion") for s, t in itertools.permutations(CULTURES, 2)]
            + [(tuple(c for c in CULTURES if c != t), t, "emotion_two_cultures") for t in CULTURES]
            + [(("SEED_GER",), "SEED_FRA", "emotion_newfilms"), (("SEED_FRA",), "SEED_GER", "emotion_newfilms")])


def setting_name(sources):
    return "+".join(sources)


def run_direction(sources, target, seed, n_jobs, n_bags=N_BAGS, keep_predictions=False):
    """All variants for one (training cultures, test culture, seed). `sources`: one or two Culture objects."""
    train = sources[0] if len(sources) == 1 else combine(sources)
    cs, c_scores = choose_cs(train, seed, n_jobs)
    membership, membership_scores = (choose_membership(*sources, seed, n_jobs, n_bags) if len(sources) == 2
                                     else (MEMBERSHIPS[-1], None))
    config = {"membership": list(membership), "cs": cs, "c_scores": c_scores, "membership_scores": membership_scores}
    bags = bag_specs(train.subjects, seed, True, n_bags)
    first = member_probas(train, target.x, EXPERTS, cs, bags, n_jobs)

    def record(proba, added, cfg):
        out = {"config": cfg, "added": added, "per_subject": score_v2(target, proba)}
        if keep_predictions:
            out["predictions"] = proba.argmax(axis=1).tolist()
        return out

    adapted = {m: self_train(train, target.x, m, cs, bags, n_jobs, first={e: first[e] for e in m})
               for m in MEMBERSHIPS}
    records = {"fair": record(*adapted[membership], config),
               "no_selftrain": record(ensemble(first, membership), [], config),
               "no_bagging": record(*self_train(train, target.x, membership, cs,
                                                bag_specs(train.subjects, seed, False), n_jobs), config)}
    for m, name in MEMBER_NAMES.items():
        records[f"members_{name}"] = record(*adapted[m], {**config, "membership": list(m)})
    people = {str(s): target.subjects == s for s in np.unique(target.subjects)}
    table = [ensemble(first, m) for m in MEMBERSHIPS] + [adapted[membership][0]]
    records["fair"]["candidates"] = {
        "keys": ["+".join(m) for m in MEMBERSHIPS] + ["fair_selftrained"],
        "accuracy": {s: [round(float((p.argmax(axis=1) == target.y)[mask].mean()), 4) for p in table]
                     for s, mask in people.items()}}
    torch.set_num_threads(1)
    v1_config, v1_scores = select_config(train, seed, use_mlp=True, early=False)
    proba, added = v1_self_train(train.x, train.y, target.x, v1_config, seed, True, False)
    records["v1"] = {**record(proba, added, list(v1_config)), "cv_score": v1_scores[v1_config]}
    return records


def fair_probas(sources, target, seed, n_jobs, n_bags=N_BAGS):
    """v3's fair model alone: its per-window probabilities on the target and its configuration."""
    train = sources[0] if len(sources) == 1 else combine(sources)
    cs, c_scores = choose_cs(train, seed, n_jobs)
    membership, membership_scores = (choose_membership(*sources, seed, n_jobs, n_bags) if len(sources) == 2
                                     else (MEMBERSHIPS[-1], None))
    config = {"membership": list(membership), "cs": cs, "c_scores": c_scores, "membership_scores": membership_scores}
    proba, _ = self_train(train, target.x, membership, cs, bag_specs(train.subjects, seed, True, n_bags), n_jobs)
    return proba, config


def job(task, seeds, n_jobs, output_root, skip_existing):
    sources, target_name, setting = SETTINGS[task]
    source = setting_name(sources)
    cultures = [load_culture(c, drop_shared=setting == "emotion_newfilms") for c in sources]
    target = load_culture(target_name)
    for seed in seeds:
        out = output_root / target_name / f"{setting}__from_{source}" / f"seed_{seed}"
        if skip_existing and all((out / f"{v}.json").exists() for v in VARIANTS):
            print(f"skip {setting} {source}->{target_name} seed {seed}", flush=True)
            continue
        start = time.time()
        records = run_direction(cultures, target, seed, n_jobs)
        for name, record in records.items():
            write_json(out / f"{name}.json", {**record, "variant": name, "source": source, "target": target_name,
                                              "seed": seed, "setting": setting})
        mean = lambda v: 100 * np.mean([p["accuracy"] for p in records[v]["per_subject"].values()])
        print(f"{setting} {source}->{target_name} seed {seed}: fair {mean('fair'):.1f}% (v1 {mean('v1'):.1f}%) "
              f"membership {records['fair']['config']['membership']} cs {records['fair']['config']['cs']} "
              f"[{(time.time() - start) / 60:.1f} min]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="Setting index 0..10 (see SETTINGS).")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("results/norm_fusion_v3"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for task in range(len(SETTINGS)) if args.all else [args.task]:
        job(task, args.seeds, args.jobs, args.output_root, args.skip_existing)


if __name__ == "__main__":
    main()
