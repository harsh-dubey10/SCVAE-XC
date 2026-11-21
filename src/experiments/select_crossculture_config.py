"""Pick SCVAE-Net cross-culture settings without touching the test culture.

For each target culture T, every configuration is scored only on the
transfers between the other two cultures (A -> B and B -> A). The
best-scoring configuration is then reported on the transfers into T
(A -> T, B -> T). T's labels never influence the choice, unlike the
test-selected numbers in the baseline paper.

Reads results/scvae_fix/ep<E>/<target>/<run>/session_all/seed_*/subject_*.json.

    python -m src.experiments.select_crossculture_config
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}


def load(root, pattern="ep*/*/*"):
    """(config, source, target) -> mean accuracy over test subjects (seeds averaged)."""
    scores = {}
    for run_dir in root.glob(pattern):
        parts = run_dir.relative_to(root).parts
        target = parts[-2]
        match = re.fullmatch(r"(.+?)__from_(\w+)", run_dir.name)
        if not match:
            continue
        # The number of held-out source subjects depends on the source culture, not the configuration.
        config = " ".join(parts[:-2] + (re.sub(r"__sval\d+", "__sval", match.group(1)),))
        per_subject = defaultdict(list)
        for path in run_dir.glob("session_all/seed_*/subject_*.json"):
            record = json.loads(path.read_text())
            per_subject[record["test_subject"]].append(record["accuracy"])
        if per_subject:
            scores[(config, match.group(2), target)] = 100 * np.mean([np.mean(v) for v in per_subject.values()])
    return scores


def choose(scores, configs, target):
    """Best configuration for ``target``, scored only on the transfers between the other two cultures."""
    others = [c for c in CULTURES if c != target]
    validation = [(others[0], others[1]), (others[1], others[0])]
    tests = [(o, target) for o in others]
    eligible = [c for c in configs if all((c, s, t) in scores for s, t in validation + tests)]
    if not eligible:
        return None
    best = max(eligible, key=lambda c: np.mean([scores[(c, s, t)] for s, t in validation]))
    return best, float(np.mean([scores[(best, s, t)] for s, t in validation]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/scvae_fix"))
    parser.add_argument("--pattern", default="ep*/*/*", help="Run directories below --root; '*/*' for <target>/<run>.")
    args = parser.parse_args()
    scores = load(args.root, args.pattern)
    configs = sorted({c for c, _, _ in scores})
    pairs = [(s, t) for t in CULTURES for s in CULTURES if s != t]

    print("All configurations x directions (diagnostic only; not used for selection):\n")
    print("| config | " + " | ".join(f"{NAMES[s][:3]}→{NAMES[t][:3]}" for s, t in pairs) + " |")
    print("|---|" + "---|" * len(pairs))
    for c in configs:
        print(f"| {c} | " + " | ".join(f"{scores[(c, s, t)]:.1f}" if (c, s, t) in scores else "—" for s, t in pairs) + " |")

    print("\nSelection without the test culture:\n")
    print("| Target | Chosen from | Chosen config | Validation acc. | Test directions (acc.) |")
    print("|---|---|---|---|---|")
    for target in CULTURES:
        others = [c for c in CULTURES if c != target]
        chosen = choose(scores, configs, target)
        if chosen is None:
            print(f"| {NAMES[target]} | — | incomplete | — | — |")
            continue
        best, val = chosen
        test = ", ".join(f"{NAMES[o]}→{NAMES[target]} {scores[(best, o, target)]:.1f}" for o in others)
        chosen_from = f"{NAMES[others[0]]}→{NAMES[others[1]]} & {NAMES[others[1]]}→{NAMES[others[0]]}"
        print(f"| {NAMES[target]} | {chosen_from} | {best} | {val:.1f} | {test} |")

if __name__ == "__main__":
    main()
