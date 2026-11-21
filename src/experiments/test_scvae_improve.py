"""Checks for the SCVAE cross-culture and unseen-film options (spec 2026-10-09).

    .venv/bin/python -m src.experiments.test_scvae_improve
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from src.experiments.train_scvae_reference import (
    add_input_noise,
    build_run_name,
    choose_source_val_subjects,
    film_folds,
    film_training_folds,
    load_split,
    make_optimizer,
    parse_args,
    source_val_subjects,
    train_fold,
    write_fold_artifacts,
)
from src.models.scvae_reference import ReferenceSCVAE


def write_culture(root, n_subjects, offset, seed):
    """A tiny one-session culture: 9 clips (emotion = (clip - 1) % 3) of 4 windows per subject.

    Feature [0, 0] holds the clip index, so tests can see which clips were loaded (without --subject-norm).
    """
    generator = torch.Generator().manual_seed(seed)
    (root / "1").mkdir(parents=True, exist_ok=True)
    for subject in range(1, n_subjects + 1):
        trials = [clip for clip in range(1, 10) for _ in range(4)]
        labels = [(clip - 1) % 3 for clip in trials]
        data = []
        for clip, label in zip(trials, labels):
            window = torch.randn(62, 5, generator=generator) + offset + 1.5 * label
            window[0, 0] = clip
            data.append(window)
        torch.save({"data": data, "labels": labels, "trials": trials}, root / "1" / f"subject{subject}.pth")


TMP = Path(tempfile.mkdtemp())
SOURCE, TARGET = TMP / "SEED_GER", TMP / "SEED_FRA"
write_culture(SOURCE, 6, 0.0, seed=1)
write_culture(TARGET, 3, 0.4, seed=2)
CROSS = ["--split", "cross", "--source-data-dir", str(SOURCE), "--data-dir", str(TARGET)]
POOLED = ["--split", "pooled", "--data-dir", str(SOURCE)]

# T1.1 defaults keep the existing run names (paper_sweep.txt cross_released and pooled lines).
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "source"])
assert args.run_name == "strict__bn_source__from_SEED_GER", args.run_name
assert (args.weight_decay, args.input_noise, args.source_val_subjects, args.save_proba) == (0.0, 0.0, 0, False)
assert (args.film_fold, args.film_condition) == (None, None)
assert parse_args(["--all-subjects", *POOLED, "--bn-recalibration", "source", "--subject-norm", "recording"]
                  ).run_name == "strict__bn_source__subjnorm__pooled"

# T1.2 scvae_fix-style line keeps its name.
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "source", "--alignment", "class_conditional_mmd",
                   "--target-shuffle", "on", "--subject-norm", "recording"])
assert args.run_name == "strict__class_conditional_mmd__bn_source__tshuf__subjnorm__from_SEED_GER", args.run_name

# T1.3 grid line: every new suffix, in order.
args = parse_args(["--all-subjects", *CROSS, "--bn-recalibration", "target", "--lr-schedule", "cosine",
                   "--target-shuffle", "on", "--learning-rate", "0.0005", "--weight-decay", "1e-4",
                   "--input-noise", "0.1", "--source-val-subjects", "2", "--subject-norm", "recording", "--save-proba"])
assert args.run_name == ("strict__bn_target__lr_cosine__tshuf__lrate0.0005__wd0.0001__noise0.1__sval2"
                         "__subjnorm__from_SEED_GER"), args.run_name
assert build_run_name(args) == args.run_name

# T1.4 film runs: released, fixed (with pooled source validation) and test-label-selected names.
assert parse_args(["--all-subjects", *POOLED, "--bn-recalibration", "source", "--film-fold", "1",
                   "--film-condition", "unseen"]).run_name == "strict__bn_source__films_unseen1__pooled"
assert parse_args(["--all-subjects", *POOLED, "--source-val-subjects", "2", "--subject-norm", "recording",
                   "--film-fold", "0", "--film-condition", "seen"]
                  ).run_name == "strict__sval2__subjnorm__films_seen0__pooled"
assert parse_args(["--all-subjects", *POOLED, "--protocol", "reported_diagnostic", "--film-fold", "2",
                   "--film-condition", "seen"]).run_name == "reported_diagnostic__films_seen2__pooled"
assert parse_args(["--all-subjects", *CROSS, "--protocol", "reported_diagnostic"]).run_name == \
    "reported_diagnostic__from_SEED_GER"


def rejected(argv):
    try:
        parse_args(argv)
    except SystemExit:
        return True
    return False


# T1.5 invalid combinations are rejected.
assert rejected(["--all-subjects", *CROSS, "--weight-decay", "-1"])
assert rejected(["--all-subjects", *CROSS, "--input-noise", "-0.1"])
assert rejected(["--all-subjects", *CROSS, "--source-val-subjects", "5"])  # 6 source subjects: must keep 2
assert rejected(["--all-subjects", *POOLED, "--source-val-subjects", "4"])  # 5 other subjects: must keep 2
assert rejected(["--all-subjects", "--split", "session", "--session", "1", "--data-dir", str(SOURCE),
                 "--source-val-subjects", "1"])
assert rejected(["--all-subjects", *POOLED, "--film-fold", "0"])  # condition missing
assert rejected(["--all-subjects", *CROSS, "--film-fold", "0", "--film-condition", "seen"])  # pooled only

# T2.1 weight decay reaches the optimiser; default is 0.
args = parse_args(["--all-subjects", *CROSS, "--weight-decay", "1e-4"])
assert make_optimizer(ReferenceSCVAE(), args).param_groups[0]["weight_decay"] == 1e-4
assert make_optimizer(ReferenceSCVAE(), parse_args(["--all-subjects", *CROSS])).param_groups[0]["weight_decay"] == 0.0

# T2.2 noise 0 returns the same tensor and draws no random numbers; noise > 0 changes the batch.
x = torch.randn(4, 62, 5)
state = torch.get_rng_state()
assert add_input_noise(x, 0.0) is x and torch.equal(torch.get_rng_state(), state)
noisy = add_input_noise(x, 0.1)
assert noisy.shape == x.shape and not torch.equal(noisy, x) and (noisy - x).std() < 0.2

# T3.1 validation subjects: deterministic per seed, k of them, from the candidates only.
for seed in (12, 34, 56):
    chosen = choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, seed)
    assert chosen == choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, seed)
    assert len(chosen) == 2 and set(chosen) <= {1, 2, 3, 4, 5, 6} and chosen == sorted(chosen)
    pooled = parse_args(["--test-subject", "3", *POOLED, "--source-val-subjects", "2", "--seed", str(seed)])
    assert 3 not in source_val_subjects(pooled, 3) and len(source_val_subjects(pooled, 3)) == 2

GRID_ARGS = ["--test-subject", "1", *CROSS, "--epochs", "3", "--batch-size", "16", "--source-val-subjects", "2",
             "--subject-norm", "recording", "--target-shuffle", "on", "--lr-schedule", "cosine",
             "--weight-decay", "1e-4", "--input-noise", "0.1", "--bn-recalibration", "source",
             "--device", "cpu", "--save-proba", "--output-dir", str(TMP / "out"),
             "--checkpoint-dir", str(TMP / "ckpt")]
args = parse_args(GRID_ARGS)
source_x, source_y, target_x, target_y = load_split(1, args)
assert len(source_x) == 4 * 36 and len(target_x) == 36  # 6 source subjects, 2 held out

# T3.2 a strict cross fold selects its epoch on the source-validation subjects.
model, metrics, training, n_source, n_target, steps, selection = train_fold(1, args, torch.device("cpu"))
assert n_source == 144 and n_target == 36
assert selection["policy"] == "best_source_validation_epoch_no_target_label_access"
assert selection["source_validation_subjects"] == choose_source_val_subjects([1, 2, 3, 4, 5, 6], 2, args.seed)
assert len(selection["source_validation_curve"]) == 3 and 1 <= selection["selected_epoch"] <= 3
assert selection["target_evaluations_during_training"] == 0
assert selection["source_validation_curve"][selection["selected_epoch"] - 1] == max(selection["source_validation_curve"])

# T4.1 film folds: every fold holds every emotion; seen/unseen training folds.
folds = film_folds(np.array([(c - 1) % 3 for c in range(1, 10)]), np.arange(1, 10))
assert sorted(folds) == list(range(1, 10))
for fold in range(3):
    assert sorted((c - 1) % 3 for c, f in folds.items() if f == fold) == [0, 1, 2]
assert film_training_folds(0, "seen") == {0, 1} and film_training_folds(2, "seen") == {2, 0}
assert film_training_folds(0, "unseen") == {1, 2}

# T4.2 loaded windows: unseen training never holds a test clip; same amount of training data;
# identical test windows in both conditions. Feature [0, 0] is the clip index (no --subject-norm).
test_clips = {c for c, f in folds.items() if f == 0}
loaded = {}
for condition in ("seen", "unseen"):
    film_args = parse_args(["--test-subject", "1", *POOLED, "--film-fold", "0", "--film-condition", condition])
    sx, sy, tx, ty = load_split(1, film_args)
    train_clips = set(sx[:, 0, 0].round().int().tolist())
    assert len(train_clips) == 6 and len(sx) == 5 * 6 * 4
    assert set(tx[:, 0, 0].round().int().tolist()) == test_clips and len(tx) == 3 * 4
    assert (train_clips & test_clips) == (test_clips if condition == "seen" else set())
    loaded[condition] = (tx, ty)
assert torch.equal(loaded["seen"][0], loaded["unseen"][0]) and torch.equal(loaded["seen"][1], loaded["unseen"][1])

# T5.1 probabilities: rows sum to 1, argmax is the reported prediction, JSON stays small, npz written.
probabilities = metrics["probabilities"]
assert probabilities.shape == (36, 3) and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
assert np.bincount(probabilities.argmax(axis=1), minlength=3).tolist() == metrics["prediction_counts"]
record, run_dir = write_fold_artifacts(model, metrics, training, 1, args, torch.device("cpu"),
                                       n_source, n_target, steps, selection)
saved = np.load(run_dir / "subject_01_proba.npz")
assert np.allclose(saved["proba"], probabilities) and saved["labels"].tolist() == target_y.tolist()
assert saved["trials"].tolist() == [c for c in range(1, 10) for _ in range(4)]
stored = json.loads((run_dir / "subject_01.json").read_text())
assert "probabilities" not in stored and "labels" not in stored and stored["source_val_subjects"] == 2

# T5.2 strict: target labels on disk do not influence training (same probabilities, same epoch).
path = TARGET / "1" / "subject1.pth"
original = torch.load(path, weights_only=True)
torch.save({**original, "labels": original["labels"][::-1]}, path)
_, relabelled, _, _, _, _, relabelled_selection = train_fold(1, parse_args(GRID_ARGS), torch.device("cpu"))
torch.save(original, path)
assert np.allclose(relabelled["probabilities"], probabilities)
assert relabelled_selection["selected_epoch"] == selection["selected_epoch"]

# T5.3 a strict film fold with pooled source validation runs end to end; saved clips are the test films.
film_args = parse_args(["--test-subject", "1", *POOLED, "--film-fold", "0", "--film-condition", "unseen",
                        "--epochs", "2", "--batch-size", "8", "--source-val-subjects", "2", "--device", "cpu",
                        "--save-proba", "--output-dir", str(TMP / "out"), "--checkpoint-dir", str(TMP / "ckpt")])
result = train_fold(1, film_args, torch.device("cpu"))
assert result[6]["policy"] == "best_source_validation_epoch_no_target_label_access" and result[4] == 12
assert 1 not in result[6]["source_validation_subjects"]
_, film_dir = write_fold_artifacts(*result[:3], 1, film_args, torch.device("cpu"), *result[3:])
assert set(np.load(film_dir / "subject_01_proba.npz")["trials"].tolist()) == test_clips

print("SCVAE improvement tests passed.")
