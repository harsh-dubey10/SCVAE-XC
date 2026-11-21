import glob
import csv
import os

import numpy as np
import torch

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


DATA_DIR = "data/processed/SEED"
SUBJECTS = range(1, 16)
RESULT_FILE = "results/tables/svm_loso_results.csv"


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

        # [samples, 62, 5]
        samples = torch.stack(data["data"]).numpy()

        # [samples]
        labels = np.asarray(data["labels"])

        X.append(samples)
        y.append(labels)

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Flatten 62 × 5 → 310 features
    X = X.reshape(X.shape[0], -1)

    return X, y


def main():

    all_results = []

    print("=" * 60)
    print("SEED — Leave-One-Subject-Out SVM Baseline")
    print("=" * 60)

    for test_subject in SUBJECTS:

        print(f"\nTest subject: {test_subject}")

        X_train = []
        y_train = []

        X_test, y_test = load_subject(test_subject)

        for subject in SUBJECTS:

            if subject == test_subject:
                continue

            X_subject, y_subject = load_subject(subject)

            X_train.append(X_subject)
            y_train.append(y_subject)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        print("Train samples:", len(X_train))
        print("Test samples :", len(X_test))

        # StandardScaler is fitted ONLY on training subjects.
        # This prevents test-subject information from leaking into training.
        clf = make_pipeline(
            StandardScaler(),
            SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale"
            )
        )

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Macro-F1 : {macro_f1:.4f}")
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
    # Overall results
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
