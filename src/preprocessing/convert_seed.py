import os
import glob
import torch
import numpy as np
from scipy.io import loadmat


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

INPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "extracted", "SEED"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "SEED"
)


# --------------------------------------------------
# SEED emotion labels
# -1 = negative
#  0 = neutral
# +1 = positive
#
# SCVAE converts these to:
#  0 = negative
#  1 = neutral
#  2 = positive
# --------------------------------------------------

SEED_LABELS = [
    1, 0, -1, -1, 0,
    1, -1, 0, 1, 1,
    0, -1, 0, 1, -1
]


# --------------------------------------------------
# Standardization
# Exactly as used in the original SCVAE code:
#
# (x - x.mean()) / x.std()
# --------------------------------------------------

def standardize(x):
    return (x - x.mean()) / x.std()


# --------------------------------------------------
# Find the three session files for a subject
# --------------------------------------------------

def get_subject_files(subject):
    files = glob.glob(
        os.path.join(INPUT_DIR, f"{subject}_*.mat")
    )

    files.sort()

    if len(files) != 3:
        raise RuntimeError(
            f"Subject {subject}: expected 3 session files, "
            f"found {len(files)}"
        )

    return files


# --------------------------------------------------
# Process one subject in one session
# --------------------------------------------------

def process_subject_session(subject, session_file):

    mat = loadmat(session_file)

    samples = []
    labels = []

    # 15 emotion trials
    for trial in range(15):

        key = f"de_LDS{trial + 1}"

        if key not in mat:
            raise KeyError(
                f"{key} not found in {session_file}"
            )

        # Shape:
        # (62 channels, time windows, 5 frequency bands)
        de = mat[key]

        for t in range(de.shape[1]):

            # One sample:
            # (62 channels, 5 frequency bands)
            x = torch.tensor(
                de[:, t, :],
                dtype=torch.float32
            )

            # Same normalization as SCVAE
            x = standardize(x)

            samples.append(x)

            # Convert:
            # -1 -> 0
            #  0 -> 1
            # +1 -> 2
            labels.append(SEED_LABELS[trial] + 1)

    return samples, labels


# --------------------------------------------------
# Main conversion
# --------------------------------------------------

def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    subjects = range(1, 16)

    for subject in subjects:

        print(f"\nProcessing subject {subject}...")

        subject_files = get_subject_files(subject)

        for session_idx, session_file in enumerate(
            subject_files, start=1
        ):

            print(
                f"  Session {session_idx}: "
                f"{os.path.basename(session_file)}"
            )

            samples, labels = process_subject_session(
                subject,
                session_file
            )

            output_session_dir = os.path.join(
                OUTPUT_DIR,
                str(session_idx)
            )

            os.makedirs(
                output_session_dir,
                exist_ok=True
            )

            output_file = os.path.join(
                output_session_dir,
                f"subject{subject}.pth"
            )

            torch.save(
                {
                    "data": samples,
                    "labels": labels
                },
                output_file
            )

            print(
                f"    Samples: {len(samples)}"
            )

            print(
                f"    Saved: {output_file}"
            )

    print("\nConversion complete.")


if __name__ == "__main__":
    main()
