import os
import csv
import random
import numpy as np
import torch

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_DIR = "data/processed/SEED"
SUBJECTS = range(1, 16)

BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 1e-3
SEED = 42
RESULT_FILE = "results/tables/mlp_loso_results.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, input_dim=310, num_classes=3):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def load_subject(subject):
    """Load all 3 sessions for one subject."""

    X = []
    y = []

    for session in range(1, 4):

        path = os.path.join(
            DATA_DIR,
            str(session),
            f"subject{subject}.pth"
        )

        data = torch.load(path, weights_only=False)

        samples = torch.stack(data["data"]).float()
        labels = torch.tensor(data["labels"], dtype=torch.long)

        X.append(samples)
        y.append(labels)

    X = torch.cat(X, dim=0)
    y = torch.cat(y, dim=0)

    # 62 × 5 → 310 features
    X = X.reshape(X.shape[0], -1)

    return X, y


def train_model(model, X_train, y_train, device):

    dataset = TensorDataset(X_train, y_train)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    for epoch in range(EPOCHS):

        total_loss = 0.0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            output = model(X_batch)

            loss = criterion(output, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)

            print(
                f"    Epoch {epoch + 1:2d}/{EPOCHS} "
                f"Loss: {avg_loss:.4f}"
            )


def evaluate_model(model, X_test, y_test, device):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X_test), BATCH_SIZE):

            X_batch = X_test[
                start:start + BATCH_SIZE
            ].to(device)

            output = model(X_batch)

            pred = torch.argmax(
                output,
                dim=1
            )

            predictions.append(
                pred.cpu()
            )

    y_pred = torch.cat(predictions).numpy()
    y_true = y_test.numpy()

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    return accuracy, macro_f1, cm


def main():

    set_seed(SEED)

    # Use Apple Silicon GPU when available.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 60)
    print("SEED — Leave-One-Subject-Out MLP Baseline")
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
        # IMPORTANT:
        # Standardization is done using training subjects only.
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
        # Create model
        # --------------------------------------------------

        model = MLP().to(device)

        train_model(
            model,
            X_train,
            y_train,
            device
        )

        accuracy, macro_f1, cm = evaluate_model(
            model,
            X_test,
            y_test,
            device
        )

        print(
            f"\nAccuracy : {accuracy:.4f}"
        )

        print(
            f"Macro-F1 : {macro_f1:.4f}"
        )

        print("Confusion matrix:")

        print(cm)

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
    print("FINAL LOSO RESULTS")
    print("=" * 60)

    for result in all_results:

        print(
            f"Subject {result['subject']:2d} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Macro-F1: {result['macro_f1']:.4f}"
        )

    print("\nMean Accuracy :", f"{accuracies.mean():.4f}")
    print("Std Accuracy  :", f"{accuracies.std():.4f}")

    print("\nMean Macro-F1 :", f"{macro_f1s.mean():.4f}")
    print("Std Macro-F1  :", f"{macro_f1s.std():.4f}")

    os.makedirs(os.path.dirname(RESULT_FILE), exist_ok=True)
    with open(RESULT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "accuracy", "macro_f1"],
        )
        writer.writeheader()
        writer.writerows(all_results)
        writer.writerow({})
        writer.writerow({
            "subject": "mean",
            "accuracy": accuracies.mean(),
            "macro_f1": macro_f1s.mean(),
        })
        writer.writerow({
            "subject": "std",
            "accuracy": accuracies.std(),
            "macro_f1": macro_f1s.std(),
        })

    print(f"Saved results to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
