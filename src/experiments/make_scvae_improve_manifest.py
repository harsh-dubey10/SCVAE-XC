"""Cluster manifests for the SCVAE cross-culture and unseen-film study (spec 2026-10-09).

    python -m src.experiments.make_scvae_improve_manifest grid    # -> cluster/scvae_improve_grid.txt (162)
    python -m src.experiments.make_scvae_improve_manifest films   # after the grid -> cluster/scvae_improve_films.txt

Lines are appended by cluster/train_array.sbatch after its own
--protocol strict --epochs 300, so the options here override those.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

from src.experiments.select_crossculture_config import CULTURES, choose, load
from src.experiments.train_scvae_reference import LOSS_WEIGHTS, all_subjects, build_run_name

SEEDS = (12, 34, 56)
GRID = [(lr, bn, noise) for lr in (0.0025, 0.0005) for bn in ("source", "target") for noise in (0.0, 0.1)]
FIXED = ("--subject-norm recording --target-shuffle on --lr-schedule cosine --weight-decay 0.0001 "
         "--epochs 100 --log-every 25")
RELEASED = "--epochs 300 --log-every 100"
GRID_ROOT = Path("results/scvae_improve/grid")


def val_k(source):
    """Held-out source subjects: 3 of the 11-12 Chinese, 2 of the 7-8 German / French."""
    return 3 if source == "SEED_CHN_MM" else 2


def directions():
    return [(s, t) for t in CULTURES for s in CULTURES if s != t]


def recipe(lr, noise, source):
    return f"{FIXED} --source-val-subjects {val_k(source)} --learning-rate {lr:g} --input-noise {noise:g}"


def cross_paths(kind, source, target, seed):
    return (f"--split cross --source-data-dir data/processed/{source} --data-dir data/processed/{target} "
            f"--seed {seed} --save-proba --output-dir results/scvae_improve/{kind}/{target} "
            f"--checkpoint-dir checkpoints/scvae_improve/{kind}/{target}")


def film_paths(culture, fold, condition, seed):
    return (f"--split pooled --data-dir data/processed/{culture} --film-fold {fold} --film-condition {condition} "
            f"--seed {seed} --save-proba --output-dir results/scvae_improve/films/{culture} "
            f"--checkpoint-dir checkpoints/scvae_improve/films/{culture}")


def config_name(lr, bn, noise):
    """The selector's key for a grid configuration (run name without source, sval count merged)."""
    args = SimpleNamespace(
        protocol="strict", alignment="standard_mmd", bn_recalibration=bn, lr_schedule="cosine",
        target_bn_update="on", target_shuffle="on", mmd_weight=LOSS_WEIGHTS["mmd"], learning_rate=lr,
        weight_decay=0.0001, input_noise=noise, source_val_subjects=1, subject_norm="recording", split="session",
    )
    return build_run_name(args).replace("__sval1", "__sval")


def grid_lines():
    """Part 1: the 8-configuration strict grid, then P1 (released, test-label-selected) across cultures."""
    grid = [f"{cross_paths('grid', s, t, seed)} {recipe(lr, noise, s)} --bn-recalibration {bn}"
            for lr, bn, noise in GRID for s, t in directions() for seed in SEEDS]
    p1 = [f"--protocol reported_diagnostic {cross_paths('paper_protocol', s, t, seed)} {RELEASED}"
          for s, t in directions() for seed in SEEDS]
    return grid + p1


def film_lines(chosen):
    """Part 2: released + BN fix, SCVAE-fixed (chosen per culture), P1; seen and unseen films."""
    lines = []
    for culture in CULTURES:
        lr, bn, noise = chosen[culture]
        for fold in range(3):
            for condition in ("seen", "unseen"):
                for seed in SEEDS:
                    paths = film_paths(culture, fold, condition, seed)
                    lines += [f"{paths} --bn-recalibration source {RELEASED}",
                              f"{paths} {recipe(lr, noise, culture)} --bn-recalibration {bn}",
                              f"--protocol reported_diagnostic {paths} {RELEASED}"]
    return lines


def expected_subjects(data_root=Path("data/processed")):
    """Culture -> every test person its runs must contain."""
    return {culture: all_subjects(data_root / culture) for culture in CULTURES}


def grid_missing(root, expected):
    """Every (configuration, direction, seed) run that lacks a test person's result."""
    missing = []
    for lr, bn, noise in GRID:
        for source, target in directions():
            name = config_name(lr, bn, noise).replace("__sval", f"__sval{val_k(source)}") + f"__from_{source}"
            for seed in SEEDS:
                directory = Path(root) / target / name / "session_all" / f"seed_{seed}"
                absent = [s for s in expected[target] if not (directory / f"subject_{s:02d}.json").exists()]
                if absent:
                    missing.append(f"{directory}: people {absent}")
    return missing


def chosen_from_grid(root=GRID_ROOT, expected=None):
    # A partial grid could let a configuration win on fewer seeds or people; refuse to choose from one.
    missing = grid_missing(root, expected or expected_subjects())
    if missing:
        raise SystemExit("grid incomplete; resubmit these before the films stage:\n" + "\n".join(missing))
    scores = load(root, pattern="*/*")
    by_name = {config_name(*g): g for g in GRID}
    chosen = {}
    for target in CULTURES:
        result = choose(scores, sorted(by_name), target)
        if result is None:
            raise SystemExit(f"grid incomplete for target {target}; pull all results first")
        chosen[target] = by_name[result[0]]
        print(f"{target}: {result[0]} (validation {result[1]:.1f})")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["grid", "films"])
    args = parser.parse_args()
    if args.stage == "grid":
        lines, path = grid_lines(), Path("cluster/scvae_improve_grid.txt")
    else:
        lines, path = film_lines(chosen_from_grid()), Path("cluster/scvae_improve_films.txt")
    header = (f"# SCVAE cross-culture / unseen-film study, {args.stage} stage ({len(lines)} tasks); spec 2026-10-09.\n"
              f"# Submit: sbatch --export=ALL,MANIFEST={path} --array=1-{len(lines)} cluster/train_array.sbatch\n")
    path.write_text(header + "\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} lines to {path}")


if __name__ == "__main__":
    main()
