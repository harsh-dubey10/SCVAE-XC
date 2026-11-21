import os
import random

import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix
)

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

NUM_SUBJECTS = 15
NUM_CLASSES = 3

# Diagnostic run: exactly one LOSO fold. Run this script separately for any
# additional subjects; a list is not a valid subject identifier.
TEST_SUBJECT = 2

EPOCHS = 20
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
LATENT_DIM = 32

LAMBDA_RECON = 1.0
LAMBDA_KL = 1.0
LAMBDA_CLS = 1.0
LAMBDA_SUPCON = 1.0
LAMBDA_MMD = 1.0

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

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
# STANDARDIZATION
# ============================================================

def standardize(source_X, target_X):

    # IMPORTANT:
    # Statistics are calculated ONLY
    # from the source subjects.

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
# MAIN DIAGNOSTIC FOLD
# ============================================================

def train_fold(test_subject):

    print()
    print("=" * 70)
    print(
        f"SCVAE — SINGLE-FOLD DIAGNOSTIC "
        f"(TARGET SUBJECT {test_subject})"
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

    print(
        f"Device:          {DEVICE}"
    )

    print(
        f"Epochs:          {EPOCHS}"
    )

    print(
        f"Batch size:      {BATCH_SIZE}"
    )

    print(
        f"Learning rate:   {LEARNING_RATE}"
    )

    print(
        f"Latent dimension:{LATENT_DIM}"
    )

    print(
        "Loss weights:    all 1.0"
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
    # LOAD TARGET
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
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        cls_sum = 0.0
        supcon_sum = 0.0
        mmd_sum = 0.0

        num_batches = 0

        target_iterator = iter(
            target_loader
        )

        for source_batch_X, source_batch_y in source_loader:

            # ------------------------------------------------
            # Get target batch
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
            # Move to device
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
            # FORWARD — SOURCE
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
            # FORWARD — TARGET
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
                recon_source +
                recon_target
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
                kl_source +
                kl_target
            ) / 2.0

            # =================================================
            # CLASSIFICATION
            # =================================================

            cls = classification_loss(
                source_logits,
                source_batch_y
            )

            # =================================================
            # SUPERVISED CONTRASTIVE
            # =================================================

            supcon = supervised_contrastive_loss(
                source_mu,
                source_batch_y
            )

            # =================================================
            # MMD
            # =================================================

            mmd = mmd_loss(
                source_mu,
                target_mu
            )

            # =================================================
            # TOTAL LOSS
            # =================================================

            total_loss = (
                LAMBDA_RECON * recon
                + LAMBDA_KL * kl
                + LAMBDA_CLS * cls
                + LAMBDA_SUPCON * supcon
                + LAMBDA_MMD * mmd
            )

            # =================================================
            # BACKPROP
            # =================================================

            total_loss.backward()

            optimizer.step()

            # =================================================
            # RECORD
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

            num_batches += 1

        # ====================================================
        # PRINT EPOCH
        # ====================================================

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Total {total_loss_sum / num_batches:.4f} | "
            f"Recon {recon_sum / num_batches:.4f} | "
            f"KL {kl_sum / num_batches:.4f} | "
            f"Cls {cls_sum / num_batches:.4f} | "
            f"SupCon {supcon_sum / num_batches:.4f} | "
            f"MMD {mmd_sum / num_batches:.4f}"
        )

    # ========================================================
    # SOURCE TRAINING ACCURACY
    # ========================================================

    model.eval()

    source_predictions = []
    source_actual = []

    source_eval_dataset = TensorDataset(
        source_X,
        source_y
    )

    source_eval_loader = DataLoader(
        source_eval_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    with torch.no_grad():

        for X_batch, y_batch in source_eval_loader:

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

            source_predictions.extend(
                predicted
            )

            source_actual.extend(
                y_batch.numpy()
            )

    source_accuracy = accuracy_score(
        source_actual,
        source_predictions
    )

    source_macro_f1 = f1_score(
        source_actual,
        source_predictions,
        average="macro"
    )

    # ========================================================
    # TARGET EVALUATION
    # ========================================================

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

    all_logits = []

    with torch.no_grad():

        for X_batch, y_batch in target_eval_loader:

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

            all_logits.append(
                logits.cpu()
            )

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    predictions_np = np.array(
        predictions
    )

    actual_np = np.array(
        actual
    )

    logits_np = all_logits.numpy()

    # ========================================================
    # TARGET METRICS
    # ========================================================

    accuracy = accuracy_score(
        actual_np,
        predictions_np
    )

    macro_f1 = f1_score(
        actual_np,
        predictions_np,
        average="macro"
    )

    cm = confusion_matrix(
        actual_np,
        predictions_np
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    print()
    print("=" * 70)
    print("SOURCE TRAINING PERFORMANCE")
    print("=" * 70)

    print(
        f"Source Accuracy : "
        f"{source_accuracy:.4f}"
    )

    print(
        f"Source Macro-F1 : "
        f"{source_macro_f1:.4f}"
    )

    print()
    print("=" * 70)
    print("TARGET PREDICTION DIAGNOSTICS")
    print("=" * 70)

    print()
    print("Target prediction counts:")

    print(
        np.bincount(
            predictions_np,
            minlength=NUM_CLASSES
        )
    )

    print()
    print("Target true-label counts:")

    print(
        np.bincount(
            actual_np,
            minlength=NUM_CLASSES
        )
    )

    print()
    print("First 20 predictions:")

    print(
        predictions_np[:20]
    )

    print()
    print("First 20 true labels:")

    print(
        actual_np[:20]
    )

    print()
    print("First 10 logits:")

    print(
        logits_np[:10]
    )

    print()
    print("Logit mean:")

    print(
        logits_np.mean(
            axis=0
        )
    )

    print()
    print("Logit standard deviation:")

    print(
        logits_np.std(
            axis=0
        )
    )

    # ========================================================
    # TARGET RESULTS
    # ========================================================

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

    print()
    print("Confusion matrix:")

    print(cm)

    print()
    print("=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"Source accuracy : {source_accuracy:.4f}"
    )

    print(
        f"Target accuracy : {accuracy:.4f}"
    )

    print(
        f"Generalization gap: "
        f"{source_accuracy - accuracy:.4f}"
    )

    print(
        f"SupCon final value: "
        f"{supcon_sum / num_batches:.4f}"
    )

    print(
        f"MMD final value: "
        f"{mmd_sum / num_batches:.4f}"
    )

    return {
        "subject": test_subject,
        "source_accuracy": source_accuracy,
        "source_macro_f1": source_macro_f1,
        "target_accuracy": accuracy,
        "target_macro_f1": macro_f1
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SCVAE — SINGLE-FOLD DIAGNOSTIC"
    )
    print("=" * 70)

    result = train_fold(
        TEST_SUBJECT
    )

    print()
    print("=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print(
        f"Target subject: "
        f"{result['subject']}"
    )

    print(
        f"Source accuracy: "
        f"{result['source_accuracy']:.4f}"
    )

    print(
        f"Target accuracy: "
        f"{result['target_accuracy']:.4f}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
