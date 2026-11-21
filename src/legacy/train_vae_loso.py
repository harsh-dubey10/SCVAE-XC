import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.legacy.vae import VAE, vae_loss


DATA_DIR = "data/processed/SEED"
RESULTS_DIR = "results/tables"
CHECKPOINT_DIR = "checkpoints/baseline"

SUBJECTS = range(1, 16)

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
LATENT_DIM = 32
BETA = 1.0
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject(subject):
    """Load all three sessions for one subject."""

    X = []
    y = []

    for session in range(1, 4):

        path = os.path.join(
            DATA_DIR,
            str(session),
            f"subject{subject}.pth"
        )

        data = torch.load(
            path,
            weights_only=False
        )

        samples = torch.stack(
            data["data"]
        ).float()

        labels = torch.tensor(
            data["labels"],
            dtype=torch.long
        )

        X.append(samples)
        y.append(labels)

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # [N, 62, 5] -> [N, 310]
    X = X.reshape(X.shape[0], -1)

    return X, y


def train_vae(model, X_train, device):

    dataset = TensorDataset(X_train)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0

        for (X_batch,) in loader:

            X_batch = X_batch.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar, z = model(
                X_batch
            )

            loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction,
                X_batch,
                mu,
                logvar,
                beta=BETA
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction_loss.item()
            total_kl += kl_loss.item()

        if (epoch + 1) % 10 == 0:

            n_batches = len(loader)

            print(
                f"    Epoch {epoch + 1:2d}/{EPOCHS} "
                f"Loss: {total_loss / n_batches:.4f} "
                f"Recon: {total_reconstruction / n_batches:.4f} "
                f"KL: {total_kl / n_batches:.4f}"
            )


def encode_dataset(model, X, device):

    model.eval()

    latent = []

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            BATCH_SIZE
        ):

            X_batch = X[
                start:start + BATCH_SIZE
            ].to(device)

            mu, logvar = model.encode(
                X_batch
            )

            # Use μ as the deterministic latent representation.
            latent.append(
                mu.cpu()
            )

    return torch.cat(latent, dim=0)


def main():

    set_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("SEED — VAE + LOSO")
    print("=" * 60)

    print("Device:", device)

    # --------------------------------------------------
    # Load all subjects once
    # --------------------------------------------------

    subjects = {}

    print("\nLoading data...")

    for subject in SUBJECTS:

        X, y = load_subject(subject)

        subjects[subject] = (X, y)

        print(
            f"Subject {subject:2d}: "
            f"{len(X)} samples"
        )

    all_results = []

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------
    # LOSO
    # --------------------------------------------------

    for test_subject in SUBJECTS:

        print("\n" + "=" * 60)
        print(f"Test subject: {test_subject}")
        print("=" * 60)

        X_train_list = []
        y_train_list = []

        X_test, y_test = subjects[test_subject]

        for subject in SUBJECTS:

            if subject == test_subject:
                continue

            X_subject, y_subject = subjects[subject]

            X_train_list.append(X_subject)
            y_train_list.append(y_subject)

        X_train = torch.cat(
            X_train_list,
            dim=0
        )

        y_train = torch.cat(
            y_train_list,
            dim=0
        )

        print("Train samples:", len(X_train))
        print("Test samples :", len(X_test))

        # --------------------------------------------------
        # Fold-specific standardization
        # --------------------------------------------------

        mean = X_train.mean(dim=0)
        std = X_train.std(dim=0)

        std[std < 1e-6] = 1.0

        X_train = (
            X_train - mean
        ) / std

        X_test = (
            X_test - mean
        ) / std

        # --------------------------------------------------
        # Train VAE
        # --------------------------------------------------

        model = VAE(
            input_dim=310,
            latent_dim=LATENT_DIM
        ).to(device)

        train_vae(
            model,
            X_train,
            device
        )

        # --------------------------------------------------
        # Extract latent representations
        # --------------------------------------------------

        Z_train = encode_dataset(
            model,
            X_train,
            device
        )

        Z_test = encode_dataset(
            model,
            X_test,
            device
        )

        print(
            "Latent train shape:",
            Z_train.shape
        )

        print(
            "Latent test shape :",
            Z_test.shape
        )

        # --------------------------------------------------
        # Classifier on latent representation
        #
        # IMPORTANT:
        # Classifier sees only training-subject labels.
        # --------------------------------------------------

        classifier = LogisticRegression(
            max_iter=2000,
            random_state=42
        )

        classifier.fit(
            Z_train.numpy(),
            y_train.numpy()
        )

        y_pred = classifier.predict(
            Z_test.numpy()
        )

        accuracy = accuracy_score(
            y_test.numpy(),
            y_pred
        )

        macro_f1 = f1_score(
            y_test.numpy(),
            y_pred,
            average="macro"
        )

        cm = confusion_matrix(
            y_test.numpy(),
            y_pred
        )

        print(
            f"\nAccuracy : {accuracy:.4f}"
        )

        print(
            f"Macro-F1 : {macro_f1:.4f}"
        )

        print("Confusion matrix:")
        print(cm)

        # --------------------------------------------------
        # Save checkpoint
        # --------------------------------------------------

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"vae_loso_test_subject_{test_subject}.pt"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "latent_dim": LATENT_DIM,
                "input_dim": 310,
                "test_subject": test_subject
            },
            checkpoint_path
        )

        all_results.append(
            {
                "subject": test_subject,
                "accuracy": accuracy,
                "macro_f1": macro_f1
            }
        )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    accuracies = np.array(
        [r["accuracy"] for r in all_results]
    )

    macro_f1s = np.array(
        [r["macro_f1"] for r in all_results]
    )

    print("\n" + "=" * 60)
    print("FINAL VAE LOSO RESULTS")
    print("=" * 60)

    for result in all_results:

        print(
            f"Subject {result['subject']:2d} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro-F1: {result['macro_f1']:.4f}"
        )

    mean_accuracy = accuracies.mean()
    std_accuracy = accuracies.std()

    mean_f1 = macro_f1s.mean()
    std_f1 = macro_f1s.std()

    print(
        "\nMean Accuracy : "
        f"{mean_accuracy:.4f}"
    )

    print(
        "Std Accuracy  : "
        f"{std_accuracy:.4f}"
    )

    print(
        "\nMean Macro-F1 : "
        f"{mean_f1:.4f}"
    )

    print(
        "Std Macro-F1  : "
        f"{std_f1:.4f}"
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    output_file = os.path.join(
        RESULTS_DIR,
        "vae_loso_results.csv"
    )

    with open(
        output_file,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "subject",
            "accuracy",
            "macro_f1"
        ])

        for result in all_results:

            writer.writerow([
                result["subject"],
                result["accuracy"],
                result["macro_f1"]
            ])

        writer.writerow([])

        writer.writerow([
            "mean",
            mean_accuracy,
            mean_f1
        ])

        writer.writerow([
            "std",
            std_accuracy,
            std_f1
        ])

    print(
        f"\nSaved results to: {output_file}"
    )


if __name__ == "__main__":
    main()
