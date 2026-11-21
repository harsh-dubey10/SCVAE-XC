"""NormFusion v3 under the papers' test-label-selected protocol, next to its fair result.

The papers chose settings by accuracy on the test people (Liu et al.: the best of 60 SVM C values per test
person; SCVAE's released code: the best checkpoint on the test set). Here v3 gets a comparable search space:

  no self-training   every non-empty set of experts (eeg, eye, early) x a C per expert in the set (124)
  self-training      every expert set at the fair Cs, after round 1, 2 and 3 (21 checkpoints)

Each candidate is scored on every test person. The fair model (all three experts, Cs chosen by v3's film-robust
CV on the training culture, three self-training rounds) is one of the candidates and is also stored on its own.
Test labels are used only to score; the selection happens in summarize_v3_paper_protocol.py.

Writes results/v3_paper_protocol/<target>/cross__from_<source>/seed_<s>.json.

    python -m src.experiments.run_v3_paper_protocol --task 0 --seeds 12 --jobs 8
    python -m src.experiments.run_v3_paper_protocol --all --jobs 8
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from src.baselines.norm_fusion import C_GRID, ST_MIN, ST_ROUNDS, ST_THRESHOLD, load_culture
from src.baselines.norm_fusion_v3 import EXPERTS, MEMBERSHIPS, bag_specs, choose_cs, ensemble, member_probas
from src.experiments.run_norm_fusion_v3 import CULTURES

SEEDS = [12, 34, 56]
TASKS = list(itertools.permutations(CULTURES, 2))
SUBSETS = [m for r in (1, 2, 3) for m in itertools.combinations(EXPERTS, r)]
FAIR = MEMBERSHIPS[-1]


def self_train_rounds(train, tgt_x, membership, cs, bags, n_jobs, first):
    """v3's self-training, returning the ensemble's probabilities after every round that adds windows."""
    proba = ensemble(first, membership)
    rounds = []
    for _ in range(ST_ROUNDS):
        confidence, pseudo = proba.max(axis=1), proba.argmax(axis=1)
        pools = [np.where((pseudo == c) & (confidence >= ST_THRESHOLD))[0] for c in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = np.concatenate([pool[np.argsort(-confidence[pool])[:k]] for pool in pools])
        proba = ensemble(member_probas(train, tgt_x, membership, cs, bags, n_jobs, tgt_x[chosen], pseudo[chosen]),
                         membership)
        rounds.append(proba)
    return rounds


def candidates(train, target, seed, n_jobs):
    """{candidate name: probabilities on the target}, and the fair model's name."""
    bags = bag_specs(train.subjects, seed, True)
    by_c = {c: member_probas(train, target.x, EXPERTS, {e: c for e in EXPERTS}, bags, n_jobs) for c in C_GRID}
    out = {}
    for members in SUBSETS:
        for cs in itertools.product(C_GRID, repeat=len(members)):
            name = "+".join(f"{m}@{c}" for m, c in zip(members, cs))
            out[name] = sum(by_c[c][m] for m, c in zip(members, cs)) / len(members)
    fair_cs, _ = choose_cs(train, seed, n_jobs)
    for members in SUBSETS:
        first = {m: by_c[fair_cs[m]][m] for m in members}
        rounds = self_train_rounds(train, target.x, members, fair_cs, bags, n_jobs, first)
        for r, proba in enumerate(rounds, 1):
            out["+".join(f"{m}@{fair_cs[m]}" for m in members) + f"|st{r}"] = proba
    fair_rounds = [k for k in out if k.startswith("+".join(f"{m}@{fair_cs[m]}" for m in FAIR) + "|st")]
    fair = max(fair_rounds, key=lambda k: int(k.split("|st")[1])) if fair_rounds else \
        "+".join(f"{m}@{fair_cs[m]}" for m in FAIR)
    return out, fair


def run(task, seed, n_jobs, output_root):
    source_name, target_name = TASKS[task]
    out = output_root / target_name / f"cross__from_{source_name}" / f"seed_{seed}.json"
    if out.exists():
        print(f"skip {out}", flush=True)
        return
    start = time.time()
    train, target = load_culture(source_name), load_culture(target_name)
    probas, fair = candidates(train, target, seed, n_jobs)
    people = np.unique(target.subjects)
    keys = sorted(probas)
    correct = {k: probas[k].argmax(axis=1) == target.y for k in keys}
    record = {"source": source_name, "target": target_name, "seed": seed, "fair": fair, "keys": keys,
              "windows": {str(p): int((target.subjects == p).sum()) for p in people},
              "accuracy": {str(p): [round(float(correct[k][target.subjects == p].mean()), 5) for k in keys]
                           for p in people}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record))
    fair_acc = 100 * np.mean([record["accuracy"][str(p)][keys.index(fair)] for p in people])
    print(f"{source_name} -> {target_name} seed {seed}: fair {fair_acc:.1f}% ({fair}), {len(keys)} candidates "
          f"[{(time.time() - start) / 60:.1f} min]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="Direction index 0..5 (see TASKS).")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("results/v3_paper_protocol"))
    args = parser.parse_args()
    for task in range(len(TASKS)) if args.all else [args.task]:
        for seed in args.seeds:
            run(task, seed, args.jobs, args.output_root)


if __name__ == "__main__":
    main()
