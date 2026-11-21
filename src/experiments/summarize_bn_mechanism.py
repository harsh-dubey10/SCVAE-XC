"""Summarise the BN collapse mechanism sweep (results/bn_mechanism)."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results/bn_mechanism")
FOLDS = Path("src/experiments/bn_mechanism_folds.txt")
VARIANTS = {"released": "strict", "target_bn_off": "strict__tgtbn_off", "lr_cosine": "strict__lr_cosine"}


def main():
    folds = [line.split() for line in FOLDS.read_text().splitlines() if line and not line.startswith("#")]
    table = defaultdict(dict)
    for variant, run in VARIANTS.items():
        for session, seed, subject, status in folds:
            path = ROOT / run / f"session_{int(session):02d}" / f"seed_{seed}" / f"subject_{int(subject):02d}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            counts = np.array(record["prediction_counts"])
            table[(session, seed, subject, status)][variant] = {
                "acc": record["accuracy"],
                "max_share": counts.max() / counts.sum(),
                "src_acc": record["final_training_metrics"]["eval_source_accuracy"],
                "stale": record["final_training_metrics"]["bn_running_mean_rel_distance"],
            }
    print(f"{'fold':<22}" + "".join(f"{v:>34}" for v in VARIANTS))
    print(f"{'':<22}" + "   acc  share  srcacc stale" * 0 + "".join(f"{'acc share src stale':>34}" for _ in VARIANTS))
    for key, row in table.items():
        cells = "".join(
            f"{row[v]['acc']:>14.3f}{row[v]['max_share']:>6.2f}{row[v]['src_acc']:>7.3f}{row[v]['stale']:>7.3f}"
            if v in row else f"{'-':>34}" for v in VARIANTS
        )
        print(f"s{key[0]} seed{key[1]:<3} subj{key[2]:<3} {key[3][:4]:<5}" + cells)
    print()
    for group in ["collapsed", "healthy"]:
        for variant in VARIANTS:
            vals = [row[variant] for key, row in table.items() if key[3] == group and variant in row]
            if not vals:
                continue
            accs = np.array([v["acc"] for v in vals])
            collapsed = sum(v["max_share"] >= 0.9 for v in vals)
            print(
                f"{group:<10} {variant:<14} n={len(vals):2d}  acc {accs.mean():.3f}  "
                f"collapsed(share>=0.9) {collapsed}  src_acc {np.mean([v['src_acc'] for v in vals]):.3f}  "
                f"staleness {np.mean([v['stale'] for v in vals]):.3f}"
            )


if __name__ == "__main__":
    main()
