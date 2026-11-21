"""Few-shot, transductive cross-subject SCVAE adaptation on complete SEED trials.

This is intentionally a different protocol from strict target-label-free LOSO.
For each held-out subject, complete labelled target trials are calibration data;
all remaining complete trials are the test set. Target features (but never test
labels) are available to MMD during base training, matching the transductive
SCVAE setting. Calibration and test windows never share a film trial.
"""

import argparse
import csv
import json
import random
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.train_scvae_reference import LOSS_WEIGHTS, get_device, set_seed
from src.models.scvae_reference import (
    MultiKernelMMD,
    ReferenceSCVAE,
    ReferenceSupConLoss,
    kl_loss,
    reconstruction_loss,
)


NUM_SUBJECTS = 15
NUM_CLASSES = 3


def load_subject_session(data_dir, subject, session):
    path = data_dir / str(session) / f"subject{subject}.pth"
    data = torch.load(path, weights_only=True)
    required = {"data", "labels", "trials"}
    if not required <= data.keys():
        raise ValueError(
            f"{path} requires trial metadata. Build it with "
            "src.preprocessing.convert_seed_extracted_features."
        )
    return (
        torch.stack(data["data"]).float(),
        torch.tensor(data["labels"], dtype=torch.long),
        torch.tensor(data["trials"], dtype=torch.long),
    )


def load_fold(data_dir, session, test_subject):
    source_x, source_y = [], []
    for subject in range(1, NUM_SUBJECTS + 1):
        x, y, trials = load_subject_session(data_dir, subject, session)
        if subject == test_subject:
            target_x, target_y, target_trials = x, y, trials
        else:
            source_x.append(x)
            source_y.append(y)
    return torch.cat(source_x), torch.cat(source_y), target_x, target_y, target_trials


def select_support_trials(labels, trials, trials_per_class, episode_seed):
    rng = np.random.default_rng(episode_seed)
    support_trials = []
    for label in range(NUM_CLASSES):
        candidates = torch.unique(trials[labels == label]).cpu().numpy()
        if len(candidates) < trials_per_class:
            raise ValueError(f"Class {label} has only {len(candidates)} trials")
        support_trials.extend(rng.choice(candidates, size=trials_per_class, replace=False).tolist())
    support_trials = sorted(int(trial) for trial in support_trials)
    support_mask = torch.isin(trials, torch.tensor(support_trials, dtype=trials.dtype))
    return support_mask, support_trials


def evaluate(model, x, y, device, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            logits = model(x[start:start + batch_size].to(device))[3]
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    actual = y.tolist()
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(actual, predictions).tolist(),
        "prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES).tolist(),
        "true_counts": np.bincount(actual, minlength=NUM_CLASSES).tolist(),
    }


def train_base(model, source_x, source_y, target_x, args, device):
    source_loader = DataLoader(TensorDataset(source_x, source_y), batch_size=args.batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(target_x), batch_size=args.batch_size, shuffle=False)
    steps = min(len(source_loader), len(target_loader))
    mmd = MultiKernelMMD()
    supcon = ReferenceSupConLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final_metrics = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {key: 0.0 for key in ("total", "recon", "kl", "classification", "supcon", "mmd")}
        source_iter = iter(source_loader)
        target_iter = cycle(target_loader)
        for _ in range(steps):
            source_batch_x, source_batch_y = next(source_iter)
            target_batch_x = next(target_iter)[0]
            source_batch_x, source_batch_y = source_batch_x.to(device), source_batch_y.to(device)
            target_batch_x = target_batch_x.to(device)
            source_recon, source_mu, source_logvar, source_logits = model(source_batch_x)
            target_recon, target_mu, target_logvar, _ = model(target_batch_x)
            losses = {
                "recon": reconstruction_loss(source_recon, source_batch_x) + reconstruction_loss(target_recon, target_batch_x),
                "kl": kl_loss(source_mu, source_logvar) + kl_loss(target_mu, target_logvar),
                "classification": torch.nn.functional.cross_entropy(source_logits, source_batch_y),
                "supcon": supcon(model.contrastive_views(source_mu), source_batch_y),
                "mmd": mmd(source_mu, target_mu),
            }
            total = (
                LOSS_WEIGHTS["reconstruction"] * losses["recon"]
                + LOSS_WEIGHTS["kl"] * losses["kl"]
                + LOSS_WEIGHTS["classification"] * losses["classification"]
                + LOSS_WEIGHTS["supcon"] * losses["supcon"]
                + LOSS_WEIGHTS["mmd"] * losses["mmd"]
            )
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            totals["total"] += total.item()
            for key, loss in losses.items():
                totals[key] += loss.item()
        final_metrics = {key: value / steps for key, value in totals.items()}
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"Base epoch {epoch:03d}/{args.epochs} | total {final_metrics['total']:.4f} | cls {final_metrics['classification']:.4f}")
    return final_metrics, steps


def calibrate_classifier(model, support_x, support_y, args, device):
    """Tune only the classifier; the representation learned from source data stays fixed."""
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.decoder.parameters():
        parameter.requires_grad = False
    for parameter in model.view_one.parameters():
        parameter.requires_grad = False
    for parameter in model.view_two.parameters():
        parameter.requires_grad = False

    loader = DataLoader(TensorDataset(support_x, support_y), batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.calibration_learning_rate)
    last_loss = None
    for epoch in range(1, args.calibration_epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in loader:
            _, _, _, logits = model(batch_x.to(device))
            loss = torch.nn.functional.cross_entropy(logits, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        last_loss = float(np.mean(epoch_losses))
        if epoch == 1 or epoch % args.calibration_log_every == 0 or epoch == args.calibration_epochs:
            print(f"Calibration epoch {epoch:03d}/{args.calibration_epochs} | cls {last_loss:.4f}")
    return last_loss


def write_artifacts(record, model, args):
    run_dir = Path(args.output_dir) / "fewshot_transductive" / f"k_{args.support_trials_per_class:02d}" / f"session_{args.session:02d}" / f"seed_{args.seed}"
    checkpoint_dir = Path(args.checkpoint_dir) / "fewshot_transductive" / f"k_{args.support_trials_per_class:02d}" / f"session_{args.session:02d}" / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"subject_{args.test_subject:02d}"
    with (run_dir / f"{stem}.json").open("w") as file:
        json.dump(record, file, indent=2)
    fields = ["protocol", "session", "test_subject", "seed", "support_trials_per_class", "pre_calibration_accuracy", "pre_calibration_macro_f1", "accuracy", "macro_f1"]
    with (run_dir / f"{stem}.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: record[key] for key in fields})
    torch.save({"model_state_dict": model.state_dict(), "record": record}, checkpoint_dir / f"{stem}.pt")
    print(f"Saved results: {run_dir / f'{stem}.json'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/SEED_4s_trials"))
    parser.add_argument("--session", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--test-subject", type=int, choices=range(1, NUM_SUBJECTS + 1), required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--support-trials-per-class", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0025)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--calibration-epochs", type=int, default=20)
    parser.add_argument("--calibration-learning-rate", type=float, default=0.0005)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--calibration-log-every", type=int, default=5)
    parser.add_argument("--output-dir", default="results/fewshot_scvae")
    parser.add_argument("--checkpoint-dir", default="checkpoints/fewshot_scvae")
    args = parser.parse_args()
    if not args.data_dir.is_dir():
        parser.error(f"Data directory not found: {args.data_dir}")
    if min(args.epochs, args.batch_size, args.support_trials_per_class, args.calibration_epochs) < 1:
        parser.error("epochs, batch size, support trials, and calibration epochs must be positive")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    source_x, source_y, target_x, target_y, target_trials = load_fold(args.data_dir, args.session, args.test_subject)
    episode_seed = args.seed + 1000 * args.test_subject + 100 * args.session
    support_mask, support_trials = select_support_trials(target_y, target_trials, args.support_trials_per_class, episode_seed)
    test_mask = ~support_mask
    support_x, support_y = target_x[support_mask], target_y[support_mask]
    test_x, test_y = target_x[test_mask], target_y[test_mask]

    print("Few-shot transductive SCVAE")
    print(f"Device: {device}; session {args.session}; target subject {args.test_subject}")
    print(f"Support trials: {support_trials}; support windows: {len(support_x)}; test windows: {len(test_x)}")
    model = ReferenceSCVAE().to(device)
    base_metrics, steps = train_base(model, source_x, source_y, target_x, args, device)
    pre = evaluate(model, test_x, test_y, device, args.batch_size)
    calibration_loss = calibrate_classifier(model, support_x, support_y, args, device)
    post = evaluate(model, test_x, test_y, device, args.batch_size)
    record = {
        "protocol": "fewshot_transductive_trial_disjoint",
        "session": args.session,
        "test_subject": args.test_subject,
        "seed": args.seed,
        "episode_seed": episode_seed,
        "data_dir": str(args.data_dir),
        "feature_window_seconds": 4,
        "support_trials_per_class": args.support_trials_per_class,
        "support_trials": support_trials,
        "support_windows": int(len(support_x)),
        "test_windows": int(len(test_x)),
        "target_test_labels_used_only_for_final_metrics": True,
        "test_trial_features_seen_unlabelled_during_base_mmd": True,
        "base_epochs": args.epochs,
        "base_steps_per_epoch": steps,
        "base_training_metrics": base_metrics,
        "calibration_epochs": args.calibration_epochs,
        "calibration_classifier_loss": calibration_loss,
        "pre_calibration_accuracy": pre["accuracy"],
        "pre_calibration_macro_f1": pre["macro_f1"],
        "accuracy": post["accuracy"],
        "macro_f1": post["macro_f1"],
        "confusion_matrix": post["confusion_matrix"],
        "prediction_counts": post["prediction_counts"],
        "true_counts": post["true_counts"],
    }
    print(f"Pre-calibration: accuracy {pre['accuracy']:.4f}, macro-F1 {pre['macro_f1']:.4f}")
    print(f"Post-calibration: accuracy {post['accuracy']:.4f}, macro-F1 {post['macro_f1']:.4f}")
    write_artifacts(record, model, args)


if __name__ == "__main__":
    main()
