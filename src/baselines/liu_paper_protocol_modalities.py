"""Liu et al. (2022) linear-SVM protocol for EEG, eye and EEG + eye across cultures.

Extends ``liu_leakage_audit`` (EEG only) to the eye and fusion features, so
EEG alone and EEG + eye are scored under the same rules as the paper:

- features: raw DE (310, channel-major) from the BCMI files; eye = the 33
  raw eye-movement features stored in ``data/processed/<culture>_EEG_EYE``
  (same windows and order as the DE files, checked on load); EEG + eye =
  both concatenated (343), the paper's "Concat" fusion;
- ``LinearSVC``; the training set and each test subject standardised with
  their own statistics, as in the released code;
- the paper's full C grid (60 values).

Per test subject the accuracy at every C is kept, so the same fits give

  oracle      best C on the test subject: the paper's rule (test-label selected)
  source_cv   C by 3-fold subject-grouped CV on the source culture (fair)
  fixed_c1    C = 1 fixed in advance (fair)

    python -m src.baselines.liu_paper_protocol_modalities
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import scale

from src.baselines import liu_leakage_audit as audit

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
MODALITIES = ["eeg", "eye", "eeg_eye"]


def load_eye(culture, eeg):
    """Eye features per subject, all sessions in order, aligned with ``eeg``."""
    root = Path(f"data/processed/{culture}_EEG_EYE")
    sessions = sorted(p.name for p in root.iterdir() if p.is_dir())
    eye = {}
    for subject, (x, y) in eeg.items():
        parts, labels = [], []
        for session in sessions:
            path = root / session / f"subject{subject}.pth"
            if path.exists():
                record = torch.load(path, weights_only=False)
                parts.append(torch.stack(record["data"]).numpy())
                labels.append(np.asarray(record["labels"]))
        stacked, labels = np.concatenate(parts), np.concatenate(labels)
        assert len(stacked) == len(x) and (labels == y).all(), f"{culture} subject {subject}: eye rows do not match DE rows"
        # The stored EEG part is the per-window z-score of the same DE rows: a further alignment check.
        z = (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
        assert np.abs(z - stacked[:, :310]).max() < 0.05, f"{culture} subject {subject}: EEG rows out of line"
        eye[subject] = stacked[:, 310:]
    return eye


def features(eeg, eye, modality):
    if modality == "eeg":
        return eeg
    if modality == "eye":
        return {s: (eye[s], eeg[s][1]) for s in eeg}
    return {s: (np.concatenate([eeg[s][0], eye[s]], axis=1), eeg[s][1]) for s in eeg}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modalities", nargs="+", default=MODALITIES, choices=MODALITIES)
    parser.add_argument("--output", type=Path, default=Path("results/crossculture/liu_paper_protocol_modalities_4s_full.json"))
    args = parser.parse_args()
    audit.C_GRID = audit.COARSE_GRID + audit.FINE_GRID

    eeg = {c: audit.load(c) for c in CULTURES}
    eye = {c: load_eye(c, eeg[c]) for c in CULTURES}
    results = json.loads(args.output.read_text()) if args.output.exists() else {}

    for modality in args.modalities:
        data = {c: features(eeg[c], eye[c], modality) for c in CULTURES}
        for source in CULTURES:
            train = data[source]
            models = None
            for target in CULTURES:
                key = f"{modality}:{source}->{target}"
                if source == target or key in results:
                    continue
                if models is None:
                    models = audit.fit_grid(scale(np.concatenate([v[0] for v in train.values()])),
                                            np.concatenate([v[1] for v in train.values()]))
                    cv_index = audit.source_cv_choice(train)
                folds = {str(s): audit.fold_record(models, x, y, cv_index) for s, (x, y) in data[target].items()}
                results[key] = folds
                print(f"{key}: paper's rule {100 * np.mean([f['oracle'] for f in folds.values()]):.1f} "
                      f"source-CV {100 * np.mean([f['source_cv'] for f in folds.values()]):.1f} "
                      f"C=1 {100 * np.mean([f['fixed_c1'] for f in folds.values()]):.1f}", flush=True)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
