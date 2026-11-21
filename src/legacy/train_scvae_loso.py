import argparse
import csv
import os
import random
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.legacy.scvae import (
    SCVAE,
    reconstruction_loss,
    kl_loss,
    supervised_contrastive_loss,
    mmd_loss,
    classification_loss,
)


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "data/processed/SEED"

RESULT_DIR = "results/tables"
CHECKPOINT_DIR = "checkpoints/scvae"

NUM_SUBJECTS = 15
NUM_CLASSES = 3

EPOCHS = 20

BATCH_SIZE = 256
LEARNING_RATE = 1e-3

LATENT_DIM = 32

# ------------------------------------------------------------
# SCVAE paper loss weights
# ------------------------------------------------------------

LAMBDA_RECON = 0.001
LAMBDA_KL = 0.0012
LAMBDA_CLS = 1.0
LAMBDA_SUPCON = 0.052
LAMBDA_MMD = 0.1

SEED = 42


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    """Set all available random-number generators for a repeatable fold."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# LOAD SUBJECT
# ============================================================

def load_subject(subject_id):

    X_list = []
    y_list = []

    for session in [1, 2, 3]:

        path = os.path.join(
            DATA_DIR,
            str(session),
            f"subject{subject_id}.pth"
        )

        data = torch.load(
            path,
            weights_only=True
        )

        session_X = torch.stack(
            data["data"]
        ).float()

        session_y = torch.tensor(
            data["labels"],
            dtype=torch.long
        )

        X_list.append(session_X)
        y_list.append(session_y)

    X = torch.cat(
        X_list,
        dim=0
    )

    y = torch.cat(
        y_list,
        dim=0
    )

    return X, y


# ============================================================
# SOURCE-ONLY STANDARDIZATION
# ============================================================

def standardize(source_X, target_X):

    mean = source_X.mean(
        dim=0,
        keepdim=True
    )

    std = source_X.std(
        dim=0,
        keepdim=True
    )

    std = torch.clamp(
        std,
        min=1e-6
    )

    source_X = (
        source_X - mean
    ) / std

    target_X = (
        target_X - mean
    ) / std

    return source_X, target_X


# ============================================================
# TRAIN ONE LOSO FOLD
# ============================================================

def train_fold(test_subject, *, epochs, seed, max_train_batches=None):

    set_seed(seed)

    print("\n")
    print("=" * 70)
    print(
        f"SCVAE LOSO — TARGET SUBJECT {test_subject}"
    )
    print("=" * 70)

    source_subjects = [
        s
        for s in range(
            1,
            NUM_SUBJECTS + 1
        )
        if s != test_subject
    ]

    print(
        f"Source subjects: {source_subjects}"
    )

    print(
        f"Target subject:  {test_subject}"
    )

    print(f"Seed:            {seed}")

    print(
        f"Device:          {DEVICE}"
    )

    print(
        "Loss weights: "
        f"recon={LAMBDA_RECON}, "
        f"KL={LAMBDA_KL}, "
        f"cls={LAMBDA_CLS}, "
        f"SupCon={LAMBDA_SUPCON}, "
        f"MMD={LAMBDA_MMD}"
    )

    # ========================================================
    # LOAD SOURCE SUBJECTS
    # ========================================================

    source_X_list = []
    source_y_list = []

    for subject in source_subjects:

        X, y = load_subject(
            subject
        )

        source_X_list.append(X)
        source_y_list.append(y)

        print(
            f"Loaded subject {subject:2d}: "
            f"{X.shape[0]} samples"
        )

    source_X = torch.cat(
        source_X_list,
        dim=0
    )

    source_y = torch.cat(
        source_y_list,
        dim=0
    )

    # ========================================================
    # LOAD TARGET SUBJECT
    #
    # target_y is retained ONLY for final evaluation.
    # It is never passed into the training loop.
    # ========================================================

    target_X, target_y = load_subject(
        test_subject
    )

    print()
    print(
        f"Total source samples: "
        f"{source_X.shape[0]}"
    )

    print(
        f"Total target samples: "
        f"{target_X.shape[0]}"
    )

    # ========================================================
    # SOURCE-ONLY STANDARDIZATION
    # ========================================================

    source_X, target_X = standardize(
        source_X,
        target_X
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    source_dataset = TensorDataset(
        source_X,
        source_y
    )

    target_dataset = TensorDataset(
        target_X
    )

    source_loader = DataLoader(
        source_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    target_loader = DataLoader(
        target_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = SCVAE(
        channels=62,
        feature_dim=5,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(1, epochs + 1):

        model.train()

        total_loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        cls_sum = 0.0
        supcon_sum = 0.0
        mmd_sum = 0.0

        source_correct = 0
        source_total = 0

        num_batches = 0

        target_iterator = iter(
            target_loader
        )

        for (
            source_batch_X,
            source_batch_y
        ) in source_loader:

            # ------------------------------------------------
            # Get unlabeled target batch
            # ------------------------------------------------

            try:

                target_batch_X = next(
                    target_iterator
                )[0]

            except StopIteration:

                target_iterator = iter(
                    target_loader
                )

                target_batch_X = next(
                    target_iterator
                )[0]

            # ------------------------------------------------
            # Move tensors to device
            # ------------------------------------------------

            source_batch_X = source_batch_X.to(
                DEVICE
            )

            source_batch_y = source_batch_y.to(
                DEVICE
            )

            target_batch_X = target_batch_X.to(
                DEVICE
            )

            optimizer.zero_grad()

            # =================================================
            # SOURCE FORWARD PASS
            # =================================================

            (
                source_reconstruction,
                source_mu,
                source_logvar,
                source_z,
                source_logits
            ) = model(
                source_batch_X
            )

            # =================================================
            # TARGET FORWARD PASS
            # =================================================

            (
                target_reconstruction,
                target_mu,
                target_logvar,
                target_z,
                target_logits
            ) = model(
                target_batch_X
            )

            # =================================================
            # RECONSTRUCTION
            # =================================================

            recon_source = reconstruction_loss(
                source_reconstruction,
                source_batch_X
            )

            recon_target = reconstruction_loss(
                target_reconstruction,
                target_batch_X
            )

            recon = (
                recon_source
                + recon_target
            ) / 2.0

            # =================================================
            # KL
            # =================================================

            kl_source = kl_loss(
                source_mu,
                source_logvar
            )

            kl_target = kl_loss(
                target_mu,
                target_logvar
            )

            kl = (
                kl_source
                + kl_target
            ) / 2.0

            # =================================================
            # SOURCE CLASSIFICATION
            #
            # ONLY SOURCE LABELS are used.
            # =================================================

            cls = classification_loss(
                source_logits,
                source_batch_y
            )

            # =================================================
            # MULTI-VIEW SUPERVISED CONTRASTIVE LEARNING
            #
            # μ
            # ├── projection head 1
            # └── projection head 2
            #          ↓
            #       [B, 2, 32]
            # =================================================

            source_views = (
                model.get_contrastive_views(
                    source_mu
                )
            )

            supcon = supervised_contrastive_loss(
                source_views,
                source_batch_y
            )

            # =================================================
            # MMD
            #
            # Align source and target latent distributions.
            #
            # Target labels are NOT used.
            # =================================================

            mmd = mmd_loss(
                source_mu,
                target_mu
            )

            # =================================================
            # WEIGHTED TOTAL LOSS
            # =================================================

            total_loss = (
                LAMBDA_RECON * recon
                + LAMBDA_KL * kl
                + LAMBDA_CLS * cls
                + LAMBDA_SUPCON * supcon
                + LAMBDA_MMD * mmd
            )

            # =================================================
            # BACKPROPAGATION
            # =================================================

            total_loss.backward()

            optimizer.step()

            # =================================================
            # TRAINING STATISTICS
            # =================================================

            total_loss_sum += (
                total_loss.item()
            )

            recon_sum += (
                recon.item()
            )

            kl_sum += (
                kl.item()
            )

            cls_sum += (
                cls.item()
            )

            supcon_sum += (
                supcon.item()
            )

            mmd_sum += (
                mmd.item()
            )

            # Source training accuracy
            predicted_source = (
                source_logits
                .argmax(dim=1)
            )

            source_correct += (
                predicted_source
                == source_batch_y
            ).sum().item()

            source_total += (
                source_batch_y
                .shape[0]
            )

            num_batches += 1

            if (
                max_train_batches is not None
                and num_batches >= max_train_batches
            ):
                break

        # ====================================================
        # EPOCH STATISTICS
        # ====================================================

        source_accuracy = (
            source_correct
            / source_total
        )

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Total {total_loss_sum / num_batches:.4f} | "
            f"Recon {recon_sum / num_batches:.4f} | "
            f"KL {kl_sum / num_batches:.4f} | "
            f"Cls {cls_sum / num_batches:.4f} | "
            f"SupCon {supcon_sum / num_batches:.4f} | "
            f"MMD {mmd_sum / num_batches:.4f} | "
            f"Source Acc {source_accuracy:.4f}"
        )

    # ========================================================
    # FINAL EVALUATION
    #
    # This is the FIRST point where target_y is used.
    # ========================================================

    model.eval()

    predictions = []
    actual = []

    target_eval_dataset = TensorDataset(
        target_X,
        target_y
    )

    target_eval_loader = DataLoader(
        target_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    with torch.no_grad():

        for (
            X_batch,
            y_batch
        ) in target_eval_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            (
                reconstruction,
                mu,
                logvar,
                z,
                logits
            ) = model(
                X_batch
            )

            predicted = (
                logits
                .argmax(dim=1)
                .cpu()
                .numpy()
            )

            predictions.extend(
                predicted
            )

            actual.extend(
                y_batch.numpy()
            )

    # ========================================================
    # METRICS
    # ========================================================

    accuracy = accuracy_score(
        actual,
        predictions
    )

    macro_f1 = f1_score(
        actual,
        predictions,
        average="macro"
    )

    cm = confusion_matrix(
        actual,
        predictions
    )

    prediction_counts = np.bincount(
        predictions,
        minlength=NUM_CLASSES
    )

    true_counts = np.bincount(
        actual,
        minlength=NUM_CLASSES
    )

    print()
    print("-" * 70)
    print(
        f"TARGET SUBJECT {test_subject} RESULTS"
    )
    print("-" * 70)

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Macro-F1 : {macro_f1:.4f}"
    )

    print(
        f"Predictions: {prediction_counts.tolist()}"
    )

    print(
        f"True labels: {true_counts.tolist()}"
    )

    print()
    print("Confusion matrix:")
    print(cm)

    return {
        "subject": test_subject,
        "seed": seed,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "model_state_dict": model.state_dict(),
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, output_file):

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject",
                "seed",
                "accuracy",
                "macro_f1"
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# MAIN
# ============================================================

def save_checkpoint(result, checkpoint_dir):
    """Save each fold independently so cluster jobs never overwrite a model."""

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"scvae_loso_subject_{result['subject']:02d}_seed_{result['seed']}.pt"
    )

    torch.save(
        {
            "model_state_dict": result["model_state_dict"],
            "test_subject": result["subject"],
            "seed": result["seed"],
            "latent_dim": LATENT_DIM,
            "channels": 62,
            "feature_dim": 5,
            "num_classes": NUM_CLASSES,
            "loss_weights": {
                "reconstruction": LAMBDA_RECON,
                "kl": LAMBDA_KL,
                "classification": LAMBDA_CLS,
                "supcon": LAMBDA_SUPCON,
                "mmd": LAMBDA_MMD,
            },
        },
        checkpoint_path,
    )

    return checkpoint_path


def main():

    parser = argparse.ArgumentParser(
        description="Train SCVAE with leave-one-subject-out evaluation on SEED."
    )
    parser.add_argument(
        "--test-subject",
        type=int,
        choices=range(1, NUM_SUBJECTS + 1),
        help="Held-out subject. Omit only to run all 15 folds sequentially.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help="Limit batches per epoch for a smoke test; never use for final results.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="CSV output path. Required for concurrent cluster jobs.",
    )
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.max_train_batches is not None and args.max_train_batches < 1:
        parser.error("--max-train-batches must be at least 1")

    test_subjects = (
        [args.test_subject]
        if args.test_subject is not None
        else list(range(1, NUM_SUBJECTS + 1))
    )

    if args.output is not None:
        output_file = args.output
    elif args.test_subject is None:
        output_file = os.path.join(RESULT_DIR, "scvae_loso_results.csv")
    else:
        output_file = os.path.join(
            RESULT_DIR,
            f"scvae_loso_subject_{args.test_subject:02d}_seed_{args.seed}.csv",
        )

    print()
    print("=" * 70)
    print(
        "SCVAE — LOSO"
    )
    print("=" * 70)

    print(
        f"Device        : {DEVICE}"
    )

    print(
        f"Epochs        : {args.epochs}"
    )

    print(
        f"Batch size    : {BATCH_SIZE}"
    )

    print(
        f"Learning rate : {LEARNING_RATE}"
    )

    print(
        f"Latent dim    : {LATENT_DIM}"
    )

    print(
        "Loss weights:"
    )

    print(
        f"  Reconstruction = {LAMBDA_RECON}"
    )

    print(
        f"  KL             = {LAMBDA_KL}"
    )

    print(
        f"  Classification = {LAMBDA_CLS}"
    )

    print(
        f"  SupCon         = {LAMBDA_SUPCON}"
    )

    print(
        f"  MMD            = {LAMBDA_MMD}"
    )

    print("=" * 70)

    print(f"Seed          : {args.seed}")
    print(f"Folds         : {test_subjects}")
    print(f"Results file  : {output_file}")
    if args.max_train_batches is not None:
        print(f"Smoke-test batches per epoch: {args.max_train_batches}")

    results = []

    for test_subject in test_subjects:
        result = train_fold(
            test_subject,
            epochs=args.epochs,
            seed=args.seed,
            max_train_batches=args.max_train_batches,
        )

        checkpoint_path = save_checkpoint(result, args.checkpoint_dir)
        result_for_csv = {
            key: value
            for key, value in result.items()
            if key != "model_state_dict"
        }
        results.append(result_for_csv)
        save_results(results, output_file)

        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved results after subject {test_subject}: {output_file}")

    print("\n" + "=" * 70)
    print("SCVAE LOSO COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
