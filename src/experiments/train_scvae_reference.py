"""Session-wise SCVAE-Net reproduction with strict and diagnostic protocols.

``strict`` is the reportable protocol: target labels are accessed once, after
the fixed final epoch. ``reported_diagnostic`` mirrors the released code's
target-accuracy checkpoint selection and is intentionally isolated from strict
results. It is diagnostic only and must never be reported as held-out accuracy.
"""

import argparse
import copy
import csv
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.scvae_reference import (
    MultiKernelMMD,
    ReferenceSCVAE,
    ReferenceSupConLoss,
    TargetUncertaintyWeights,
    UncertaintyWeightedMMD,
    class_conditional_mmd,
    kl_loss,
    recalibrate_batchnorm,
    reconstruction_loss,
)


DEFAULT_DATA_DIR = Path("data/processed/SEED")
NUM_CLASSES = 3
LOSS_WEIGHTS = {
    "reconstruction": 0.001,
    "kl": 0.0012,
    "classification": 1.0,
    "supcon": 0.052,
    "mmd": 0.1,
}
DEFAULT_LEARNING_RATE = 0.0025


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_recording(x):
    """Standardise each channel x band feature over one recording's windows.

    Label-free: uses only that recording's own features, so applying it to the
    target is transductive adaptation (like MMD), not target-label access.
    """
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)


def load_subject_session(subject, session, data_dir, subject_norm="none"):
    path = data_dir / str(session) / f"subject{subject}.pth"
    data = torch.load(path, weights_only=True)
    x = torch.stack(data["data"]).float()
    if subject_norm == "recording":
        x = normalize_recording(x)
    return x, torch.tensor(data["labels"], dtype=torch.long)


def available_subjects(session, data_dir):
    """Subjects with a file for this session (15 for SEED; SEED-GER session 3 has 4)."""
    return sorted(int(path.stem.removeprefix("subject")) for path in (data_dir / str(session)).glob("subject*.pth"))


def load_fold(test_subject, session, data_dir, subject_norm="none"):
    source_x, source_y = [], []
    for subject in available_subjects(session, data_dir):
        x, y = load_subject_session(subject, session, data_dir, subject_norm)
        if subject == test_subject:
            target_x, target_y = x, y
        else:
            source_x.append(x)
            source_y.append(y)
    return torch.cat(source_x), torch.cat(source_y), target_x, target_y


def sessions_in(data_dir):
    return sorted(int(path.name) for path in data_dir.iterdir() if path.is_dir() and path.name.isdigit())


def all_subjects(data_dir):
    return sorted({subject for session in sessions_in(data_dir) for subject in available_subjects(session, data_dir)})


def load_subject_all_sessions(subject, data_dir, subject_norm="none"):
    """Every session a subject completed, concatenated (SEED-GER subjects 5-8 have two)."""
    parts = [
        load_subject_session(subject, session, data_dir, subject_norm)
        for session in sessions_in(data_dir)
        if subject in available_subjects(session, data_dir)
    ]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


N_FILM_FOLDS = 3


def load_subject_trials(subject, data_dir):
    """Clip index of every window, in load_subject_all_sessions order."""
    return torch.cat([
        torch.tensor(torch.load(data_dir / str(session) / f"subject{subject}.pth", weights_only=True)["trials"])
        for session in sessions_in(data_dir)
        if subject in available_subjects(session, data_dir)
    ])


def load_subjects_trials(data_dir, subjects):
    return torch.cat([load_subject_trials(s, data_dir) for s in subjects])


def film_folds(labels, trials):
    """Clip -> fold: each emotion's clips, sorted, dealt round-robin (as clip_heldout_emotion.py)."""
    fold = {}
    for emotion in np.unique(labels):
        for i, clip in enumerate(sorted(np.unique(trials[labels == emotion]))):
            fold[int(clip)] = i % N_FILM_FOLDS
    return fold


def culture_film_folds(data_dir, subjects):
    """Film folds from the clip emotions of these (source) subjects."""
    labels = torch.cat([load_subject_all_sessions(s, data_dir)[1] for s in subjects]).numpy()
    return film_folds(labels, load_subjects_trials(data_dir, subjects).numpy())


def film_training_folds(test_fold, condition):
    """seen: the test fold and the next; unseen: the two other folds. Same amount of training data."""
    if condition == "seen":
        return {test_fold, (test_fold + 1) % N_FILM_FOLDS}
    return {f for f in range(N_FILM_FOLDS) if f != test_fold}


def film_mask(trials, folds, allowed):
    return torch.tensor([folds[int(t)] in allowed for t in trials], dtype=torch.bool)


def load_subjects(data_dir, subjects, subject_norm="none"):
    parts = [load_subject_all_sessions(s, data_dir, subject_norm) for s in subjects]
    return torch.cat([x for x, _ in parts]), torch.cat([y for _, y in parts])


def load_dataset(data_dir, exclude_subject=None, subject_norm="none"):
    return load_subjects(data_dir, [s for s in all_subjects(data_dir) if s != exclude_subject], subject_norm)


def choose_source_val_subjects(subjects, k, seed):
    """k whole source subjects held out for epoch selection, fixed by the seed."""
    rng = np.random.default_rng(seed)
    return sorted(int(s) for s in rng.choice(sorted(subjects), size=k, replace=False))


def source_dir(args):
    return args.source_data_dir if args.split == "cross" else args.data_dir


def source_candidates(args, test_subject):
    """Every subject that may be used as source: the other culture (cross) or the other people (pooled)."""
    if args.split == "cross":
        return all_subjects(args.source_data_dir)
    return [s for s in all_subjects(args.data_dir) if s != test_subject]


def source_val_subjects(args, test_subject):
    k = getattr(args, "source_val_subjects", 0)
    if not k:
        return []
    return choose_source_val_subjects(source_candidates(args, test_subject), k, args.seed)


def target_film_mask(args, test_subject):
    """The test person's windows of the test fold of films."""
    folds = culture_film_folds(args.data_dir, source_candidates(args, test_subject))
    return film_mask(load_subject_trials(test_subject, args.data_dir), folds, {args.film_fold})


def target_trials(args, test_subject):
    """Clip index of each evaluated target window, in evaluation order."""
    if args.split == "session":
        data = torch.load(args.data_dir / str(args.session) / f"subject{test_subject}.pth", weights_only=True)
        return np.asarray(data["trials"], dtype=np.int64)
    trials = load_subject_trials(test_subject, args.data_dir)
    if getattr(args, "film_fold", None) is not None:
        trials = trials[target_film_mask(args, test_subject)]
    return trials.numpy().astype(np.int64)


def load_source(args, subjects, test_subject):
    """Source windows of these subjects; with --film-fold only the training films.

    Recordings are standardised over all their windows first (label-free), then filtered.
    """
    x, y = load_subjects(source_dir(args), subjects, args.subject_norm)
    if getattr(args, "film_fold", None) is None:
        return x, y
    folds = culture_film_folds(args.data_dir, source_candidates(args, test_subject))
    keep = film_mask(load_subjects_trials(args.data_dir, subjects), folds,
                     film_training_folds(args.film_fold, args.film_condition))
    return x[keep], y[keep]


def load_split(test_subject, args):
    """Source/target tensors for one fold of the chosen split.

    session: one session; the other subjects of that session are the source.
    pooled:  leave-one-subject-out over all sessions (Liu et al. 2022 ICSI).
    cross:   every subject and session of --source-data-dir is the source; the
             target is one subject of --data-dir, all sessions (Liu et al. CCSI).
    """
    norm = args.subject_norm
    if args.split == "session":
        return load_fold(test_subject, args.session, args.data_dir, norm)
    target_x, target_y = load_subject_all_sessions(test_subject, args.data_dir, norm)
    held_out = source_val_subjects(args, test_subject)
    training = [s for s in source_candidates(args, test_subject) if s not in held_out]
    source_x, source_y = load_source(args, training, test_subject)
    if getattr(args, "film_fold", None) is not None:
        keep = target_film_mask(args, test_subject)
        target_x, target_y = target_x[keep], target_y[keep]
    return source_x, source_y, target_x, target_y


def fold_dir_name(args):
    return f"session_{args.session:02d}" if args.split == "session" else "session_all"


def evaluate(model, target_x, target_y, device, batch_size):
    """Evaluate a trained model against the target labels."""
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(target_x), batch_size):
            logits = model(target_x[start:start + batch_size].to(device))[3]
            probabilities.append(torch.softmax(logits, dim=1).cpu())
    probabilities = torch.cat(probabilities).numpy()
    predictions = probabilities.argmax(axis=1).tolist()

    actual = target_y.tolist()
    accuracy = accuracy_score(actual, predictions)
    macro_f1 = f1_score(actual, predictions, average="macro")
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=list(range(NUM_CLASSES))).tolist(),
        "prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES).tolist(),
        "true_counts": np.bincount(actual, minlength=NUM_CLASSES).tolist(),
        "probabilities": probabilities,
        "labels": np.asarray(actual, dtype=np.int64),
    }


def source_batchnorm_staleness(model, source_x, source_y, device, batch_size):
    """Label-free-for-target check of how stale the BatchNorm running stats are.

    Compares the classifier BatchNorm running mean with the mean of its input
    on source data under the final weights, and reports eval-mode source
    accuracy. Uses source labels only.
    """
    model.eval()
    batchnorm = model.classifier[1]
    features, predictions = [], []
    with torch.no_grad():
        for start in range(0, len(source_x), batch_size):
            mu = model.encode(source_x[start:start + batch_size].to(device))[0]
            features.append(model.classifier[0](mu).cpu())
            predictions.append(model.classifier(mu).argmax(dim=1).cpu())
    fresh_mean = torch.cat(features).mean(dim=0)
    running_mean = batchnorm.running_mean.detach().cpu()
    return {
        "bn_running_mean_rel_distance": float((running_mean - fresh_mean).norm() / fresh_mean.norm()),
        "eval_source_accuracy": float((torch.cat(predictions) == source_y).float().mean()),
    }


def source_validation_accuracy(model, train_x, val_x, val_y, device, batch_size):
    """Accuracy on held-out source subjects (source labels only).

    Scored on a copy whose BatchNorm statistics are re-estimated on the
    training source, so stale target-dominated running statistics do not
    decide the epoch. The trained model itself is untouched.
    """
    scored = recalibrate_batchnorm(copy.deepcopy(model), train_x, device)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(val_x), batch_size):
            predictions.append(scored(val_x[start:start + batch_size].to(device))[3].argmax(dim=1).cpu())
    return float((torch.cat(predictions) == val_y).float().mean())


def copy_state_dict_to_cpu(model):
    """Keep an immutable diagnostic checkpoint without retaining GPU/MPS memory."""
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def make_optimizer(model, args):
    return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)


def add_input_noise(x, std):
    """Gaussian noise for training inputs; std 0 returns x unchanged and draws nothing."""
    return x if std == 0 else x + std * torch.randn_like(x)


def train_fold(test_subject, args, device):
    set_seed(args.seed)
    source_x, source_y, target_x, target_y = load_split(test_subject, args)
    held_out = source_val_subjects(args, test_subject) if args.split != "session" else []
    validate = bool(held_out) and args.protocol == "strict"
    if validate:
        val_x, val_y = load_source(args, held_out, test_subject)
        print(f"Source-validation subjects: {held_out} ({len(val_x)} windows)")
    validation_curve, best_val, best_val_state, best_val_epoch = [], None, None, None

    source_loader = DataLoader(
        TensorDataset(source_x, source_y),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    # Target labels are deliberately excluded from this training loader. The
    # released code keeps it in recording order, so each target batch holds one
    # or two film clips (mostly a single emotion); --target-shuffle on mixes them.
    target_loader = DataLoader(
        TensorDataset(target_x),
        batch_size=args.batch_size,
        shuffle=args.target_shuffle == "on",
        drop_last=False,
    )

    model = ReferenceSCVAE(latent_dim=args.latent_dim).to(device)
    if getattr(args, "init_encoder", None):
        # Encoder pretrained without emotion labels (stimulus-locked contrastive, pretrain_slc.py).
        model.encoder.load_state_dict(torch.load(args.init_encoder, map_location=device, weights_only=True))
    optimizer = make_optimizer(model, args)
    if args.alignment in ("standard_mmd", "class_conditional_mmd"):
        mmd = MultiKernelMMD()
        uncertainty_weights = None
    else:
        mmd = UncertaintyWeightedMMD()
        uncertainty_weights = TargetUncertaintyWeights(
            args.alignment,
            temperature=args.uncertainty_temperature,
            uniform_mix=args.uniform_weight_mix,
            entropy_mix=args.entropy_mix,
        )
    supcon = ReferenceSupConLoss(temperature=args.temperature)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine" else None
    )
    batchnorm_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    steps_per_epoch = min(len(source_loader), len(target_loader))
    if args.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_train_batches)

    print(f"\n{fold_dir_name(args)} | target subject {test_subject}")
    print(f"Source samples: {len(source_x)} | target samples: {len(target_x)}")
    print(f"Training steps per epoch: {steps_per_epoch}")

    best_metrics = None
    best_state_dict = None
    best_epoch = None
    best_update = None
    update_count = 0

    final_train_metrics = {}
    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {name: 0.0 for name in ["total", "recon", "kl", "classification", "supcon", "mmd"]}
        if uncertainty_weights is not None:
            totals.update({
                "target_entropy": 0.0,
                "target_posterior_uncertainty": 0.0,
                "target_combined_uncertainty": 0.0,
                "target_weight_effective_samples": 0.0,
            })
        if args.alignment == "class_conditional_mmd":
            totals["target_confident_fraction"] = 0.0
        target_iterator = cycle(target_loader)
        source_iterator = iter(source_loader)

        for _ in range(steps_per_epoch):
            source_batch_x, source_batch_y = next(source_iterator)
            target_batch_x = next(target_iterator)[0]
            source_batch_x = source_batch_x.to(device)
            source_batch_y = source_batch_y.to(device)
            target_batch_x = target_batch_x.to(device)

            # Denoising: the model sees noisy source inputs but reconstructs the clean ones.
            source_recon, source_mu, source_logvar, source_logits = model(
                add_input_noise(source_batch_x, args.input_noise)
            )
            if args.target_bn_update == "off":
                # Normalise with target batch statistics as usual, but keep
                # the running statistics source-only (momentum 0 = no update).
                momenta = [layer.momentum for layer in batchnorm_layers]
                for layer in batchnorm_layers:
                    layer.momentum = 0.0
            target_recon, target_mu, target_logvar, target_logits = model(target_batch_x)
            if args.target_bn_update == "off":
                for layer, momentum in zip(batchnorm_layers, momenta):
                    layer.momentum = momentum
            if args.alignment == "class_conditional_mmd":
                # No alignment until the classifier's target pseudo-labels are usable.
                if epoch <= args.align_warmup_epochs:
                    alignment_loss = source_mu.sum() * 0.0
                    confident_fraction = torch.tensor(0.0)
                else:
                    alignment_loss, confident_fraction = class_conditional_mmd(
                        mmd, source_mu, source_batch_y, target_mu, target_logits,
                        threshold=args.pseudo_threshold, num_classes=NUM_CLASSES,
                    )
                weight_diagnostics = {"target_confident_fraction": confident_fraction}
            elif uncertainty_weights is None:
                alignment_loss = mmd(source_mu, target_mu)
                weight_diagnostics = None
            else:
                target_weights, weight_diagnostics = uncertainty_weights(target_logits, target_logvar)
                alignment_loss = mmd(source_mu, target_mu, target_weights)

            losses = {
                "recon": reconstruction_loss(source_recon, source_batch_x)
                + reconstruction_loss(target_recon, target_batch_x),
                "kl": kl_loss(source_mu, source_logvar) + kl_loss(target_mu, target_logvar),
                "classification": torch.nn.functional.cross_entropy(source_logits, source_batch_y),
                "supcon": supcon(model.contrastive_views(source_mu), source_batch_y),
                "mmd": alignment_loss,
            }
            total = (
                LOSS_WEIGHTS["reconstruction"] * losses["recon"]
                + LOSS_WEIGHTS["kl"] * losses["kl"]
                + LOSS_WEIGHTS["classification"] * losses["classification"]
                + LOSS_WEIGHTS["supcon"] * losses["supcon"]
                + args.mmd_weight * losses["mmd"]
            )

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            update_count += 1

            # This is deliberately a diagnostic-only recreation of the
            # released code's target-label checkpoint selection. It is never
            # reached under the strict, reportable protocol.
            if args.protocol == "reported_diagnostic":
                diagnostic_metrics = evaluate(
                    model, target_x, target_y, device, args.batch_size
                )
                if (
                    best_metrics is None
                    or diagnostic_metrics["accuracy"] > best_metrics["accuracy"]
                ):
                    best_metrics = diagnostic_metrics
                    best_state_dict = copy_state_dict_to_cpu(model)
                    best_epoch = epoch
                    best_update = update_count
                # `evaluate` switches to inference mode; restore training
                # mode before the next source-target update.
                model.train()

            totals["total"] += total.item()
            for name, loss in losses.items():
                totals[name] += loss.item()
            if weight_diagnostics is not None:
                for name in totals:
                    if name in weight_diagnostics:
                        totals[name] += weight_diagnostics[name].item()

        final_train_metrics = {name: value / steps_per_epoch for name, value in totals.items()}
        if scheduler is not None:
            scheduler.step()
        if validate:
            val_accuracy = source_validation_accuracy(model, source_x, val_x, val_y, device, args.batch_size)
            validation_curve.append(val_accuracy)
            if best_val is None or val_accuracy > best_val:
                best_val, best_val_state, best_val_epoch = val_accuracy, copy_state_dict_to_cpu(model), epoch
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"total {final_train_metrics['total']:.4f} | "
                f"cls {final_train_metrics['classification']:.4f} | "
                f"supcon {final_train_metrics['supcon']:.4f} | "
                f"mmd {final_train_metrics['mmd']:.4f}"
                + (f" | src-val {validation_curve[-1]:.4f}" if validate else "")
            )

    if args.protocol == "strict":
        if validate:
            # Restore the epoch the held-out source subjects chose; target labels are still unread.
            model.load_state_dict(best_val_state)
        # Measured before any recalibration, on the weights training produced.
        final_train_metrics.update(
            source_batchnorm_staleness(model, source_x, source_y, device, args.batch_size)
        )
        if args.bn_recalibration == "source":
            recalibrate_batchnorm(model, source_x, device)
        elif args.bn_recalibration == "target":
            # Unlabelled target features only, as already used by MMD.
            recalibrate_batchnorm(model, target_x, device)
        elif args.bn_recalibration == "target_culture":
            # Culture-aware BN: unlabelled windows of every target-culture subject.
            culture_x, _ = load_dataset(args.data_dir, subject_norm=args.subject_norm)
            recalibrate_batchnorm(model, culture_x, device)
        # Strict protocol: target labels first appear here, after final epoch.
        metrics = evaluate(model, target_x, target_y, device, args.batch_size)
        selection = {
            "policy": "best_source_validation_epoch_no_target_label_access" if validate
            else "fixed_final_epoch_no_target_label_access_during_training",
            "target_evaluations_during_training": 0,
            "selected_epoch": best_val_epoch if validate else args.epochs,
            "selected_update": update_count,
        }
        if validate:
            selection["source_validation_subjects"] = held_out
            selection["source_validation_curve"] = validation_curve
    else:
        model.load_state_dict(best_state_dict)
        metrics = best_metrics
        selection = {
            "policy": "diagnostic_best_target_accuracy_after_each_training_update",
            "target_evaluations_during_training": update_count,
            "selected_epoch": best_epoch,
            "selected_update": best_update,
        }

    return (
        model,
        metrics,
        final_train_metrics,
        len(source_x),
        len(target_x),
        steps_per_epoch,
        selection,
    )


def write_fold_artifacts(
    model,
    metrics,
    training,
    test_subject,
    args,
    device,
    source_count,
    target_count,
    steps,
    selection,
):
    arrays = {key: metrics[key] for key in ("probabilities", "labels")}
    metrics = {key: value for key, value in metrics.items() if key not in arrays}
    run_dir = Path(args.output_dir) / args.run_name / fold_dir_name(args) / f"seed_{args.seed}"
    checkpoint_dir = Path(args.checkpoint_dir) / args.run_name / fold_dir_name(args) / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    stem = f"subject_{test_subject:02d}"
    record = {
        "protocol": args.protocol,
        "alignment": args.alignment,
        "run_name": args.run_name,
        "session": args.session if args.split == "session" else "all",
        "split": args.split,
        "source_data_dir": str(args.source_data_dir) if args.split == "cross" else str(args.data_dir),
        "test_subject": test_subject,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "input_noise": args.input_noise,
        "latent_dim": args.latent_dim,
        "temperature": args.temperature,
        "data_dir": str(args.data_dir),
        "bn_recalibration": args.bn_recalibration,
        "lr_schedule": args.lr_schedule,
        "target_bn_update": args.target_bn_update,
        "subject_norm": args.subject_norm,
        "target_shuffle": args.target_shuffle,
        "source_val_subjects": args.source_val_subjects,
        "init_encoder": str(args.init_encoder) if getattr(args, "init_encoder", None) else None,
        "film_fold": args.film_fold,
        "film_condition": args.film_condition,
        "uncertainty": {
            "temperature": args.uncertainty_temperature,
            "uniform_weight_mix": args.uniform_weight_mix,
            "entropy_mix": args.entropy_mix,
            "weights_detached_from_backpropagation": args.alignment != "standard_mmd",
        },
        "source_samples": source_count,
        "target_samples": target_count,
        "steps_per_epoch": steps,
        "loss_weights": {**LOSS_WEIGHTS, "mmd": args.mmd_weight},
        "target_labels_used_only_after_training": args.protocol == "strict",
        "target_label_policy": selection,
        "final_training_metrics": training,
        **metrics,
    }
    json_path = run_dir / f"{stem}.json"
    csv_path = run_dir / f"{stem}.csv"
    checkpoint_path = checkpoint_dir / f"{stem}.pt"

    with json_path.open("w") as output:
        json.dump(record, output, indent=2)
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=[
            "protocol", "alignment", "bn_recalibration", "session", "test_subject", "seed", "accuracy", "macro_f1",
            "epochs", "batch_size", "learning_rate", "latent_dim", "temperature",
        ])
        writer.writeheader()
        writer.writerow({key: record[key] for key in writer.fieldnames})
    if args.save_proba:
        np.savez(run_dir / f"{stem}_proba.npz", proba=arrays["probabilities"].astype(np.float32),
                 labels=arrays["labels"], trials=target_trials(args, test_subject))
    torch.save({"model_state_dict": model.state_dict(), "record": record}, checkpoint_path)

    print(f"Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Checkpoint policy: {selection['policy']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved metadata: {json_path}")
    print(f"Saved checkpoint: {checkpoint_path}")
    return record, run_dir


def write_summary(records, run_dir):
    if len(records) < 2:
        return
    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["test_subject", "accuracy", "macro_f1", "seed", "session"])
        writer.writeheader()
        writer.writerows({key: record[key] for key in writer.fieldnames} for record in records)
        writer.writerow({})
        writer.writerow({
            "test_subject": "mean",
            "accuracy": float(np.mean([record["accuracy"] for record in records])),
            "macro_f1": float(np.mean([record["macro_f1"] for record in records])),
        })
        writer.writerow({
            "test_subject": "std",
            "accuracy": float(np.std([record["accuracy"] for record in records])),
            "macro_f1": float(np.std([record["macro_f1"] for record in records])),
        })
    print(f"Saved summary: {summary_path}")


def build_run_name(args):
    """Result directory name: the protocol plus a suffix for every non-default option."""
    name = args.protocol if args.alignment == "standard_mmd" else f"{args.protocol}__{args.alignment}"
    if args.bn_recalibration != "none":
        name = f"{name}__bn_{args.bn_recalibration}"
    if args.lr_schedule != "constant":
        name = f"{name}__lr_{args.lr_schedule}"
    if args.target_bn_update != "on":
        name = f"{name}__tgtbn_{args.target_bn_update}"
    if args.target_shuffle == "on":
        name = f"{name}__tshuf"
    if args.mmd_weight != LOSS_WEIGHTS["mmd"]:
        name = f"{name}__mmd{args.mmd_weight:g}"
    if args.learning_rate != DEFAULT_LEARNING_RATE:
        name = f"{name}__lrate{args.learning_rate:g}"
    if args.weight_decay:
        name = f"{name}__wd{args.weight_decay:g}"
    if args.input_noise:
        name = f"{name}__noise{args.input_noise:g}"
    if args.source_val_subjects:
        name = f"{name}__sval{args.source_val_subjects}"
    if args.subject_norm != "none":
        name = f"{name}__subjnorm"
    if getattr(args, "init_encoder", None):
        name = f"{name}__slc"
    if getattr(args, "film_fold", None) is not None:
        name = f"{name}__films_{args.film_condition}{args.film_fold}"
    if args.split == "pooled":
        name = f"{name}__pooled"
    elif args.split == "cross":
        name = f"{name}__from_{args.source_data_dir.name}"
    return name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    fold_group = parser.add_mutually_exclusive_group(required=True)
    fold_group.add_argument("--test-subject", type=int)
    fold_group.add_argument("--all-subjects", action="store_true", help="Run all folds sequentially; do not use in concurrent jobs.")
    parser.add_argument("--session", type=int, choices=[1, 2, 3], help="Required for --split session.")
    parser.add_argument(
        "--split",
        choices=["session", "pooled", "cross"],
        default="session",
        help=(
            "session: one session, leave one subject out (default, SEED protocol). "
            "pooled: leave one subject out over all sessions. "
            "cross: train on all of --source-data-dir, test each --data-dir subject."
        ),
    )
    parser.add_argument("--source-data-dir", type=Path, help="Source culture for --split cross.")
    parser.add_argument(
        "--subject-norm",
        choices=["none", "recording"],
        default="none",
        help=(
            "recording: standardise every feature over each recording (subject x session), for "
            "source and target alike. Label-free but transductive: it needs the target's unlabelled windows."
        ),
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.07, help="Released-code default; its CLI variable 0.1 was unused.")
    parser.add_argument(
        "--alignment",
        choices=["standard_mmd", "entropy_weighted_mmd", "posterior_weighted_mmd", "combined_uncertainty_mmd",
                 "class_conditional_mmd"],
        default="standard_mmd",
        help="Use standard MMD or an explicitly labelled uncertainty-weighted MMD ablation.",
    )
    parser.add_argument("--target-shuffle", choices=["off", "on"], default="off",
                        help="off keeps target batches in recording order (released behaviour).")
    parser.add_argument("--mmd-weight", type=float, default=LOSS_WEIGHTS["mmd"],
                        help="Weight of the source-target alignment loss (released value 0.1; 0 disables alignment).")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Adam weight decay (0 = released behaviour).")
    parser.add_argument("--input-noise", type=float, default=0.0,
                        help="Std of Gaussian noise added to source inputs in training only (0 = off).")
    parser.add_argument("--source-val-subjects", type=int, default=0,
                        help=("--split cross or pooled: hold out this many whole source subjects and keep the "
                              "epoch with the best accuracy on them (source labels only). 0 = fixed final epoch."))
    parser.add_argument("--init-encoder", type=Path, default=None,
                        help="Initialise the encoder from a label-free pretrained state dict (suffix __slc).")
    parser.add_argument("--save-proba", action="store_true",
                        help="Also save the target's per-window probabilities, labels and clips (.npz).")
    parser.add_argument("--film-fold", type=int, choices=[0, 1, 2], default=None,
                        help="--split pooled: test only on this fold of films (each fold holds every emotion).")
    parser.add_argument("--film-condition", choices=["seen", "unseen"], default=None,
                        help=("With --film-fold F. seen: train on folds F and F+1 (includes the test films); "
                              "unseen: train on the two other folds. Same amount of training data."))
    parser.add_argument("--pseudo-threshold", type=float, default=0.9,
                        help="class_conditional_mmd: minimum softmax confidence for a target pseudo-label.")
    parser.add_argument("--align-warmup-epochs", type=int, default=10,
                        help="class_conditional_mmd: epochs of source-only training before aligning.")
    parser.add_argument("--uncertainty-temperature", type=float, default=0.25)
    parser.add_argument("--uniform-weight-mix", type=float, default=0.2)
    parser.add_argument("--entropy-mix", type=float, default=0.5)
    parser.add_argument(
        "--protocol",
        choices=["strict", "reported_diagnostic"],
        default="strict",
        help=(
            "strict is reportable and evaluates target labels only once. "
            "reported_diagnostic selects the best target-accuracy update, mirroring "
            "the released code; it is diagnostic only."
        ),
    )
    parser.add_argument(
        "--bn-recalibration",
        choices=["none", "source", "target", "target_culture"],
        default="none",
        help=(
            "Re-estimate BatchNorm statistics after the final epoch from source features, "
            "the unlabelled target subject (target) or every unlabelled subject of the "
            "target culture (target_culture, --split cross only). 'none' keeps the released behaviour."
        ),
    )
    parser.add_argument(
        "--lr-schedule",
        choices=["constant", "cosine"],
        default="constant",
        help="constant is the released behaviour; cosine anneals the learning rate to ~0 by the final epoch.",
    )
    parser.add_argument(
        "--target-bn-update",
        choices=["on", "off"],
        default="on",
        help="off stops target forward passes from updating BatchNorm running statistics.",
    )
    parser.add_argument("--skip-existing", action="store_true", help="Skip folds whose result JSON already exists.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max-train-batches", type=int, default=None, help="Smoke-test only; do not use for results.")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default="results/reference_scvae")
    parser.add_argument("--checkpoint-dir", default="checkpoints/reference_scvae")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 2 or args.log_every < 1:
        parser.error("epochs, batch size, and log interval must be positive; batch size must be at least 2")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be at least 1")
    if args.uncertainty_temperature <= 0:
        parser.error("--uncertainty-temperature must be positive")
    if not 0 <= args.uniform_weight_mix <= 1 or not 0 <= args.entropy_mix <= 1:
        parser.error("uncertainty mixing values must be in [0, 1]")
    if args.protocol == "reported_diagnostic" and args.alignment != "standard_mmd":
        parser.error("reported_diagnostic is reserved for the released standard-MMD audit")
    if not args.data_dir.is_dir():
        parser.error(f"--data-dir does not exist: {args.data_dir}")
    if args.split == "session":
        if args.session is None:
            parser.error("--split session requires --session")
        args.subjects = available_subjects(args.session, args.data_dir)
    else:
        if args.session is not None:
            parser.error(f"--split {args.split} uses every session; drop --session")
        args.subjects = all_subjects(args.data_dir)
    if args.split == "cross":
        if args.source_data_dir is None or not args.source_data_dir.is_dir():
            parser.error("--split cross requires an existing --source-data-dir")
        if args.source_data_dir.resolve() == args.data_dir.resolve():
            parser.error("--source-data-dir must differ from --data-dir; use --split pooled within one dataset")
    elif args.source_data_dir is not None:
        parser.error("--source-data-dir applies only to --split cross")
    if args.bn_recalibration == "target_culture" and args.split != "cross":
        parser.error("--bn-recalibration target_culture needs --split cross (a different target culture)")
    if args.split != "cross" and len(args.subjects) < 2:
        parser.error(f"{args.data_dir} needs at least two subjects for this split, found {args.subjects}")
    if args.test_subject is not None and args.test_subject not in args.subjects:
        parser.error(f"--test-subject {args.test_subject} is not available for this split; available: {args.subjects}")
    if args.protocol == "reported_diagnostic" and args.bn_recalibration != "none":
        parser.error("--bn-recalibration applies only to the strict protocol")
    if args.weight_decay < 0 or args.input_noise < 0 or args.source_val_subjects < 0:
        parser.error("--weight-decay, --input-noise and --source-val-subjects must be non-negative")
    if args.source_val_subjects:
        if args.split == "cross":
            source_count = len(all_subjects(args.source_data_dir))
        elif args.split == "pooled":
            source_count = len(args.subjects) - 1
        else:
            parser.error("--source-val-subjects applies only to --split cross or pooled")
        if source_count - args.source_val_subjects < 2:
            parser.error("--source-val-subjects must leave at least two source subjects for training")
    if (args.film_fold is None) != (args.film_condition is None):
        parser.error("--film-fold and --film-condition go together")
    if args.film_fold is not None and args.split != "pooled":
        parser.error("--film-fold applies only to --split pooled")
    args.run_name = build_run_name(args)
    return args


def main():
    args = parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    subjects = args.subjects if args.all_subjects else [args.test_subject]
    print(f"SCVAE-Net reference reproduction ({args.protocol}; {args.alignment})")
    print(f"Device: {device} | split: {args.split} | {fold_dir_name(args)} | subjects: {list(subjects)}")
    print(f"Data: {args.data_dir}" + (f" | source: {args.source_data_dir}" if args.split == "cross" else ""))
    if args.protocol == "strict":
        print("Target labels are not read during training or checkpoint selection.")
    else:
        print("DIAGNOSTIC ONLY: target labels select the best update during training.")

    records = []
    latest_run_dir = None
    for subject in subjects:
        existing = (
            Path(args.output_dir) / args.run_name / fold_dir_name(args)
            / f"seed_{args.seed}" / f"subject_{subject:02d}.json"
        )
        if args.skip_existing and existing.exists():
            print(f"Skipping subject {subject}: {existing} exists")
            records.append(json.loads(existing.read_text()))
            latest_run_dir = existing.parent
            continue
        model, metrics, training, source_count, target_count, steps, selection = train_fold(
            subject, args, device
        )
        record, latest_run_dir = write_fold_artifacts(
            model,
            metrics,
            training,
            subject,
            args,
            device,
            source_count,
            target_count,
            steps,
            selection,
        )
        records.append(record)
    write_summary(records, latest_run_dir)


if __name__ == "__main__":
    main()
