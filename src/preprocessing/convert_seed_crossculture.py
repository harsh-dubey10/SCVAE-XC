"""Convert BCMI multimodal DE features (SEED-GER, SEED-FRA and the Chinese
SEED multimodal subset) into the per-subject tensor format.

BCMI ships each session as ``<subject>_<session>.npz`` with a within-subject
train/test split (first 12 clips / the rest; first 9 / last 6 for Chinese). Both parts are merged in
presentation order so the strict pipeline can hold out whole subjects.

The files carry labels but no clip IDs. Clip IDs are rebuilt from the clip
durations in ``<Culture>_information.xlsx`` and checked against the label runs
of every file. The features are LDS-smoothed DE (lag-1 autocorrelation ~0.96,
like SEED ``de_LDS``) and are z-scored per sample exactly as
``convert_seed_extracted_features`` does for SEED.

SEED-GER has 20 sessions: subjects 5-8 completed only two (Liu et al., J.
Neural Eng. 2022), so their session-3 files do not exist. The Chinese
multimodal subset has the 12 SEED subjects with eye tracking (1-5, 8-14, same
numbering as SEED). Its EEG was re-extracted by the same pipeline as SEED-GER
and SEED-FRA, so it is the Chinese side for cross-culture comparisons.
"""

import argparse
import itertools
import pickle
import re
from pathlib import Path

import numpy as np
import torch


BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# Clip labels (-1 negative, 0 neutral, 1 positive) and durations in seconds,
# in presentation order, from <Culture>_information.xlsx.
DATASETS = {
    "CHN": {
        # SEED clip order; durations are the official 1 s window counts.
        "clip_labels": [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1],
        "clip_seconds": [235, 233, 206, 238, 185, 195, 237, 216, 265, 237, 235, 233, 235, 238, 206],
        "stimulus_numbers": list(range(1, 16)),
        "train_clips": 9,
    },
    "GER": {
        # Stimuli 2 and 19 were not used; the 18 remaining clips are listed.
        "clip_labels": [1, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, 0, -1, -1, 0, 1],
        "clip_seconds": [131, 420, 223, 143, 136, 135, 228, 227, 145, 162, 175, 289, 103, 97, 336, 136, 256, 172],
        "stimulus_numbers": [1, *range(3, 19), 20],
        "train_clips": 12,
    },
    "FRA": {
        "clip_labels": [1, -1, 0, -1, 1, 0, -1, 0, 1, -1, 0, 1, 1, 0, -1, -1, 0, 1, -1, 0, 1],
        "clip_seconds": [172, 255, 178, 275, 100, 129, 300, 146, 129, 267, 133, 144, 107, 144, 127, 77, 166, 133, 126, 87, 175],
        "stimulus_numbers": list(range(1, 22)),
        "train_clips": 12,
    },
}


def standardize(sample):
    return (sample - sample.mean()) / sample.std()


def label_runs(labels):
    return [(int(label), len(list(group))) for label, group in itertools.groupby(labels)]


def clip_ids(labels, dataset, window_seconds):
    """Assign a 1-based clip index to every window, validated against the labels.

    Consecutive clips with the same label form one label run in the file. A run
    holding one clip takes all of its windows; in a multi-clip run every clip
    but the last takes ``seconds // window`` windows and the last takes the rest.
    Returns the per-window clip IDs and the per-clip deviations from
    ``seconds // window`` (SEED-GER clip 1 is one 4 s window short).
    """
    spec = DATASETS[dataset]
    expected = [(label + 1, seconds // window_seconds) for label, seconds in zip(spec["clip_labels"], spec["clip_seconds"])]
    expected_runs = [list(group) for _, group in itertools.groupby(enumerate(expected), key=lambda item: item[1][0])]
    actual_runs = label_runs(labels)
    if [run[0][1][0] for run in expected_runs] != [label for label, _ in actual_runs]:
        raise ValueError(f"{dataset}: label runs {actual_runs} do not match the clip table")

    ids, deviations = [], {}
    for run, (_, run_length) in zip(expected_runs, actual_runs):
        remaining = run_length
        for position, (clip_index, (_, expected_windows)) in enumerate(run):
            windows = remaining if position == len(run) - 1 else expected_windows
            if windows != expected_windows:
                deviations[clip_index + 1] = windows - expected_windows
            if windows <= 0:
                raise ValueError(f"{dataset}: clip {clip_index + 1} would get {windows} windows")
            ids.extend([clip_index + 1] * windows)
            remaining -= windows
    return ids, deviations


def load_session(path):
    npz = np.load(path)
    parts = [pickle.loads(npz["train_data"]), pickle.loads(npz["test_data"])]
    features = np.concatenate([np.stack([part[band] for band in BANDS], axis=-1) for part in parts])
    labels = np.concatenate([npz["train_label"], npz["test_label"]]).astype(int)
    if features.shape[1:] != (62, 5) or len(features) != len(labels):
        raise ValueError(f"{path}: unexpected shapes {features.shape}, {labels.shape}")
    return features, labels, len(npz["train_label"])


def load_eye(eye_dir, stem, n_windows, train_count):
    """The 33 BCMI eye-movement features of one session, aligned with its EEG windows."""
    eye = pickle.load(open(eye_dir / stem, "rb"))
    x = np.concatenate([eye["train_data_eye"], eye["test_data_eye"]]).astype(np.float32)
    if len(x) != n_windows or len(eye["train_data_eye"]) != train_count:
        raise ValueError(f"{eye_dir / stem}: {len(x)} eye windows, EEG has {n_windows}")
    if not np.isfinite(x).all():
        raise ValueError(f"{eye_dir / stem}: non-finite eye features")
    return x


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--source-dir", type=Path, required=True, help="Directory holding <subject>_<session>.npz files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window-seconds", choices=["1", "4"], required=True)
    parser.add_argument(
        "--modality", choices=["eeg", "eye", "eeg_eye"], default="eeg",
        help="eeg: 62x5 DE windows (z-scored per window). eye: the 33 eye-movement features "
             "(raw units). eeg_eye: the flattened z-scored EEG followed by the raw eye features "
             "(343 values). Eye features exist for 4 s windows only.",
    )
    parser.add_argument("--eye-dir", type=Path, help="Directory of <subject>_<session> eye pickles (eye modalities).")
    parser.add_argument("--overwrite", action="store_true", help="Replace only files in the selected output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.modality != "eeg" and (args.eye_dir is None or args.window_seconds != "4"):
        raise SystemExit("--modality eye/eeg_eye needs --eye-dir and --window-seconds 4")
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Feature directory not found: {args.source_dir}")
    if args.output_dir.exists() and any(args.output_dir.rglob("*.pth")) and not args.overwrite:
        raise FileExistsError(f"Output already contains tensors: {args.output_dir}. Use a new directory or --overwrite.")

    window_seconds = int(args.window_seconds)
    provenance = {
        "dataset": f"SEED-{args.dataset}",
        "source_dir": str(args.source_dir.resolve()),
        "feature_key": {"eeg": "DE (LDS-smoothed, BCMI eeg_used)",
                        "eye": "33 BCMI eye-movement features (raw units)",
                        "eeg_eye": "flattened per-window z-scored DE (310) + raw eye features (33)"}[args.modality],
        "window_seconds": window_seconds,
        "normalization": "per_sample_zscore_x_minus_mean_over_std",
        "label_mapping": {"negative": 0, "neutral": 1, "positive": 2},
        "merged_split": f"BCMI train_data (first {DATASETS[args.dataset]['train_clips']} clips) + test_data (remaining clips), presentation order",
        "trials": "1-based clip index in presentation order; stimulus numbers "
                  + str(DATASETS[args.dataset]["stimulus_numbers"]),
    }

    paths = sorted(args.source_dir.glob("*_*.npz"), key=lambda p: tuple(map(int, re.findall(r"\d+", p.stem))))
    if not paths:
        raise FileNotFoundError(f"No <subject>_<session>.npz files in {args.source_dir}")
    all_deviations = set()
    for path in paths:
        subject, session = map(int, path.stem.split("_"))
        features, labels, train_count = load_session(path)
        trials, deviations = clip_ids(labels, args.dataset, window_seconds)
        all_deviations.add(tuple(sorted(deviations.items())))
        split = DATASETS[args.dataset]["train_clips"]
        if trials[train_count - 1] != split or trials[train_count] != split + 1:
            raise ValueError(f"{path}: BCMI train/test split is not at clip {split}/{split + 1}")

        eeg = [standardize(torch.tensor(window, dtype=torch.float32)) for window in features]
        if args.modality == "eeg":
            samples = eeg
        else:
            eye = torch.from_numpy(load_eye(args.eye_dir, path.stem, len(features), train_count))
            # Eye features keep their units; per-recording normalisation or the
            # models' scalers standardise them per feature later.
            samples = list(eye) if args.modality == "eye" else [
                torch.cat([e.flatten(), y]) for e, y in zip(eeg, eye)]
        destination = args.output_dir / str(session) / f"subject{subject}.pth"
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"data": samples, "labels": labels.tolist(), "trials": trials, "provenance": provenance},
            destination,
        )
        print(f"subject {subject:02d}, session {session}: {len(samples)} samples, {max(trials)} clips -> {destination}")

    provenance["clip_window_deviations"] = sorted(all_deviations)
    (args.output_dir / "provenance.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in provenance.items()) + "\n"
    )


if __name__ == "__main__":
    main()
