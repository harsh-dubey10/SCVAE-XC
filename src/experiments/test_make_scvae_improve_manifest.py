"""Checks for the SCVAE improvement cluster manifests.

    .venv/bin/python -m src.experiments.test_make_scvae_improve_manifest
"""

import shlex
from collections import Counter

from src.experiments.make_scvae_improve_manifest import GRID, config_name, film_lines, grid_lines
from src.experiments.train_scvae_reference import parse_args

# sbatch prepends these; manifest options must override them.
SBATCH = ["--all-subjects", "--protocol", "strict", "--epochs", "300"]

lines = grid_lines()
assert len(lines) == 162 and len(set(lines)) == 162
kinds, names = Counter(), set()
for line in lines:
    args = parse_args([*SBATCH, *shlex.split(line)])
    assert args.split == "cross" and args.source_data_dir != args.data_dir and args.save_proba
    if args.protocol == "strict":
        kinds["grid"] += 1
        assert args.epochs == 100 and "/grid/" in args.output_dir
        assert args.source_val_subjects == (3 if args.source_data_dir.name == "SEED_CHN_MM" else 2)
        names.add(args.run_name.split("__from_")[0].replace(f"__sval{args.source_val_subjects}", "__sval"))
    else:
        kinds["p1"] += 1
        assert args.run_name.startswith("reported_diagnostic__from_") and args.epochs == 300
        assert "/paper_protocol/" in args.output_dir
assert kinds == {"grid": 144, "p1": 18}
assert names == {config_name(*g) for g in GRID} and len(names) == 8

chosen = {culture: GRID[-1] for culture in ("SEED_CHN_MM", "SEED_GER", "SEED_FRA")}
films = film_lines(chosen)
assert len(films) == 162 and len(set(films)) == 162
kinds = Counter()
for line in films:
    args = parse_args([*SBATCH, *shlex.split(line)])
    assert args.split == "pooled" and args.film_fold in (0, 1, 2) and "/films/" in args.output_dir
    if args.protocol == "reported_diagnostic":
        kinds["p1"] += 1
    elif args.weight_decay:
        kinds["fixed"] += 1
        assert args.source_val_subjects == (3 if args.data_dir.name == "SEED_CHN_MM" else 2)
        assert args.run_name.startswith(config_name(*GRID[-1]).replace("__sval", f"__sval{args.source_val_subjects}"))
    else:
        kinds["released"] += 1
        assert args.bn_recalibration == "source" and args.epochs == 300
assert kinds == {"p1": 54, "fixed": 54, "released": 54}
print("Manifest tests passed.")

# Review I1: the films stage must not be chosen from a partial grid.
import tempfile
from pathlib import Path

from src.experiments.make_scvae_improve_manifest import SEEDS, chosen_from_grid, directions, grid_missing, val_k

root = Path(tempfile.mkdtemp())
expected = {"SEED_CHN_MM": [1, 2], "SEED_GER": [1, 2, 3], "SEED_FRA": [4, 5]}
for lr, bn, noise in GRID:
    for source, target in directions():
        name = config_name(lr, bn, noise).replace("__sval", f"__sval{val_k(source)}") + f"__from_{source}"
        for seed in SEEDS:
            directory = root / target / name / "session_all" / f"seed_{seed}"
            directory.mkdir(parents=True)
            for subject in expected[target]:
                (directory / f"subject_{subject:02d}.json").write_text(
                    '{"test_subject": %d, "accuracy": %.2f}' % (subject, 0.5 + 0.01 * GRID.index((lr, bn, noise))))
assert grid_missing(root, expected) == []
assert set(chosen_from_grid(root, expected)) == {"SEED_CHN_MM", "SEED_GER", "SEED_FRA"}
lost = next(root.glob("SEED_GER/*__from_SEED_FRA/session_all/seed_34/subject_02.json"))
lost.unlink()
missing = grid_missing(root, expected)
assert len(missing) == 1 and "seed_34" in missing[0] and "SEED_FRA" in missing[0], missing
try:
    chosen_from_grid(root, expected)
    raise AssertionError("a partial grid must abort")
except SystemExit as stop:
    assert "seed_34" in str(stop)
print("Partial-grid tests passed.")
