"""Base model probabilities for the calibration study: NormFusion v3 fair, per window, saved for later scoring.

Tasks (15): the 6 cross-culture directions (train on one culture, test on another), then for each culture the
3 same-culture groups (people dealt round-robin into 3 groups; train on the other two groups of the same
culture, test on the group). Each task runs seeds 12, 34, 56 and writes
results/calibration/base/<target>/<mode>__from_<source>/seed_<s>.npz with the probabilities and the target
windows' labels, people, recordings and clips (labels are used only to score).

    python -m src.experiments.run_calibration_base --task 0 --jobs 16
    python -m src.experiments.run_calibration_base --all --jobs 8

Design: docs/superpowers/specs/2026-10-04-calibration-design.md.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from src.baselines.norm_fusion import Culture, load_culture
from src.experiments.run_norm_fusion_v3 import CULTURES, fair_probas

SEEDS = [12, 34, 56]
TASKS = ([((s,), t, "cross") for s, t in itertools.permutations(CULTURES, 2)]
         + [(c, g, "same") for c in CULTURES for g in range(3)])


def same_culture_groups(subjects):
    """(training people, test people) for each of 3 groups dealt round-robin over the sorted people."""
    people = np.unique(subjects)
    return [(people[np.arange(len(people)) % 3 != g], people[np.arange(len(people)) % 3 == g]) for g in range(3)]


def subset(culture, people):
    keep = np.isin(culture.subjects, people)
    return Culture(culture.x[keep], culture.y[keep], culture.clips[keep], culture.subjects[keep],
                   culture.recordings[keep])


def task_data(task):
    """(training cultures, target, target name, output folder name) for one task."""
    first, second, mode = TASKS[task]
    if mode == "cross":
        return [load_culture(first[0])], load_culture(second), second, f"cross__from_{first[0]}"
    culture = load_culture(first)
    train_people, test_people = same_culture_groups(culture.subjects)[second]
    return [subset(culture, train_people)], subset(culture, test_people), first, f"same_group{second}__from_{first}"


def job(task, seeds, n_jobs, output_root, skip_existing):
    sources, target, target_name, folder = task_data(task)
    for seed in seeds:
        out = output_root / target_name / folder / f"seed_{seed}.npz"
        if skip_existing and out.exists():
            print(f"skip {folder} -> {target_name} seed {seed}", flush=True)
            continue
        start = time.time()
        proba, config = fair_probas(sources, target, seed, n_jobs)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, proba=proba, y=target.y, subjects=target.subjects, recordings=target.recordings,
                            clips=target.clips, config=json.dumps(config, default=str))
        tmp.replace(out)
        accuracy = 100 * (proba.argmax(axis=1) == target.y).mean()
        print(f"{folder} -> {target_name} seed {seed}: window accuracy {accuracy:.1f}% "
              f"[{(time.time() - start) / 60:.1f} min]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="Task index 0..14 (see TASKS).")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("results/calibration/base"))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()
    for task in range(len(TASKS)) if args.all else [args.task]:
        job(task, args.seeds, args.jobs, args.output_root, args.skip_existing)


if __name__ == "__main__":
    main()
