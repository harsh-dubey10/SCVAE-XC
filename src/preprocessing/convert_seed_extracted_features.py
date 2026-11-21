"""Convert official SEED DE/LDS MATLAB features into isolated PyTorch tensors.

The SCVAE-Net release uses 4-second ``de_LDS`` features.  This tool also
supports the official 1-second feature directory for an explicitly labelled
window-length ablation.  It never overwrites the existing baseline dataset.
"""

import argparse
from pathlib import Path

import torch
from scipy.io import loadmat


SEED_LABELS = [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
NUM_SUBJECTS = 15


def standardize(sample):
    return (sample - sample.mean()) / sample.std()


def subject_session_paths(source_dir, subject):
    paths = sorted(source_dir.glob(f"{subject}_*.mat"))
    if len(paths) != 3:
        raise RuntimeError(f"Subject {subject}: expected three session files, found {len(paths)}")
    return paths


def convert_file(path):
    mat = loadmat(path)
    samples, labels, trials = [], [], []
    for trial, raw_label in enumerate(SEED_LABELS, start=1):
        key = f"de_LDS{trial}"
        if key not in mat:
            raise KeyError(f"{path}: missing {key}")
        features = mat[key]
        if features.ndim != 3 or features.shape[0] != 62 or features.shape[2] != 5:
            raise ValueError(f"{path}: {key} has unexpected shape {features.shape}")
        for time_index in range(features.shape[1]):
            sample = torch.tensor(features[:, time_index, :], dtype=torch.float32)
            samples.append(standardize(sample))
            labels.append(raw_label + 1)
            trials.append(trial)
    return samples, labels, trials


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", choices=["1", "4"], required=True)
    parser.add_argument("--overwrite", action="store_true", help="Replace only files in the selected output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {args.source_dir}")
    if args.output_dir.exists() and any(args.output_dir.rglob("*.pth")) and not args.overwrite:
        raise FileExistsError(f"Output already contains tensors: {args.output_dir}. Use a new directory or --overwrite.")

    provenance = {
        "source_dir": str(args.source_dir.resolve()),
        "feature_key": "de_LDS",
        "window_seconds": int(args.window_seconds),
        "normalization": "per_sample_zscore_x_minus_mean_over_std",
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
    }
    for subject in range(1, NUM_SUBJECTS + 1):
        for session, source_path in enumerate(subject_session_paths(args.source_dir, subject), start=1):
            samples, labels, trials = convert_file(source_path)
            destination = args.output_dir / str(session) / f"subject{subject}.pth"
            destination.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"data": samples, "labels": labels, "trials": trials, "provenance": provenance},
                destination,
            )
            print(f"subject {subject:02d}, session {session}: {len(samples)} samples -> {destination}")

    (args.output_dir / "provenance.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in provenance.items()) + "\n"
    )


if __name__ == "__main__":
    main()
