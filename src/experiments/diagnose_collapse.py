"""Post-hoc prediction-collapse diagnostics for strict SCVAE-Net folds.

A fold "collapses" when the trained model assigns (almost) every target window
to one emotion class. This script audits saved fold artifacts and separates
two kinds of evidence:

* label-free signals, computed only from target features and the model. These
  are usable for model selection or early warning under the strict protocol.
* label-based outcomes (accuracy, per-class recall), used here only to check
  whether the label-free signals actually identify failed folds.

Usage:

    python -m src.experiments.diagnose_collapse \
        --results-root results/reference_scvae/strict \
        --output-dir results/diagnostics/collapse/reference_4s

Add ``--no-checkpoints`` to use only the prediction counts stored in each
fold JSON (fast; no model or data loading).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from src.experiments.train_scvae_reference import (
    DEFAULT_DATA_DIR,
    NUM_CLASSES,
    get_device,
    load_fold,
)
from src.models.scvae_reference import ReferenceSCVAE


# Label-free thresholds on the target predicted-class distribution.
COLLAPSE_MAX_SHARE = 0.90
PARTIAL_MAX_SHARE = 0.60
PARTIAL_MIN_SHARE = 0.05
# Label-based definition of a failed fold, used only to validate the flags.
FAILED_MACRO_F1 = 0.35


def normalized_entropy(probabilities, axis=-1):
    probabilities = np.clip(probabilities, 1e-12, 1.0)
    return -(probabilities * np.log(probabilities)).sum(axis=axis) / np.log(NUM_CLASSES)


def prediction_profile(prediction_counts):
    """Label-free summary of the target predicted-class histogram."""
    counts = np.asarray(prediction_counts, dtype=float)
    shares = counts / counts.sum()
    max_share = float(shares.max())
    min_share = float(shares.min())
    if max_share >= COLLAPSE_MAX_SHARE:
        status = "collapsed"
    elif max_share >= PARTIAL_MAX_SHARE or min_share < PARTIAL_MIN_SHARE:
        status = "partial"
    else:
        status = "healthy"
    return {
        "pred_share_max": max_share,
        "pred_share_min": min_share,
        "pred_dominant_class": int(shares.argmax()),
        "pred_classes_used": int((shares >= PARTIAL_MIN_SHARE).sum()),
        "pred_entropy_norm": float(normalized_entropy(shares)),
        "collapse_status": status,
    }


@torch.no_grad()
def encode_all(model, x, device, batch_size=512):
    mus, probabilities = [], []
    for start in range(0, len(x), batch_size):
        _, mu, _, logits = model(x[start:start + batch_size].to(device))
        mus.append(mu.cpu())
        probabilities.append(torch.softmax(logits, dim=1).cpu())
    return torch.cat(mus).numpy(), torch.cat(probabilities).numpy()


def checkpoint_signals(checkpoint_path, record, device, fold_cache):
    """Recompute label-free target statistics from a saved fold checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ReferenceSCVAE(latent_dim=record.get("latent_dim", 256)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_dir = Path(record.get("data_dir", DEFAULT_DATA_DIR))
    key = (str(data_dir), record["session"], record["test_subject"])
    if key not in fold_cache:
        fold_cache.clear()
        fold_cache[key] = load_fold(record["test_subject"], record["session"], data_dir)
    source_x, source_y, target_x, _ = fold_cache[key]

    source_mu, source_prob = encode_all(model, source_x, device)
    target_mu, target_prob = encode_all(model, target_x, device)

    source_spread = source_mu.std(axis=0).mean()
    target_spread = target_mu.std(axis=0).mean()
    # Distance between domain means, relative to the source latent spread.
    domain_gap = np.linalg.norm(source_mu.mean(axis=0) - target_mu.mean(axis=0))
    predictions = target_prob.argmax(axis=1)
    return {
        "target_mean_confidence": float(target_prob.max(axis=1).mean()),
        "target_mean_sample_entropy": float(normalized_entropy(target_prob).mean()),
        "target_mean_prob_entropy": float(normalized_entropy(target_prob.mean(axis=0))),
        "target_source_spread_ratio": float(target_spread / max(source_spread, 1e-12)),
        "domain_mean_gap_rel": float(domain_gap / max(source_spread, 1e-12)),
        # Source labels are allowed: they are training labels.
        "source_accuracy": float((source_prob.argmax(axis=1) == source_y.numpy()).mean()),
        "recomputed_prediction_counts": np.bincount(predictions, minlength=NUM_CLASSES).tolist(),
    }


def label_outcomes(record):
    matrix = np.asarray(record["confusion_matrix"], dtype=float)
    recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1)
    return {
        "accuracy": record["accuracy"],
        "macro_f1": record["macro_f1"],
        "recall_class_0": float(recall[0]),
        "recall_class_1": float(recall[1]),
        "recall_class_2": float(recall[2]),
        "failed": bool(record["macro_f1"] < FAILED_MACRO_F1),
    }


def checkpoint_path_for(json_path):
    parts = list(json_path.with_suffix(".pt").parts)
    if "results" not in parts:
        return None
    parts[parts.index("results")] = "checkpoints"
    path = Path(*parts)
    return path if path.exists() else None


def is_strict(record):
    return str(record.get("protocol", "")).startswith("strict")


def collect(results_root, use_checkpoints, device):
    rows, fold_cache = [], {}
    json_paths = sorted(results_root.rglob("subject_*.json"))
    for json_path in json_paths:
        record = json.loads(json_path.read_text())
        if not is_strict(record) or "prediction_counts" not in record:
            continue
        row = {
            "path": str(json_path),
            "alignment": record.get("alignment", "standard_mmd"),
            "session": record["session"],
            "seed": record["seed"],
            "test_subject": record["test_subject"],
            "final_train_mmd": record["final_training_metrics"]["mmd"],
            "final_train_classification": record["final_training_metrics"]["classification"],
            **prediction_profile(record["prediction_counts"]),
        }
        checkpoint_path = checkpoint_path_for(json_path) if use_checkpoints else None
        if checkpoint_path is not None:
            signals = checkpoint_signals(checkpoint_path, record, device, fold_cache)
            if signals.pop("recomputed_prediction_counts") != record["prediction_counts"]:
                print(f"warning: recomputed predictions differ from saved ones: {json_path}")
            row.update(signals)
        row.update(label_outcomes(record))
        rows.append(row)
        print(
            f"S{row['session']} seed {row['seed']:>3} subj {row['test_subject']:>2} | "
            f"{row['collapse_status']:>9} | max share {row['pred_share_max']:.2f} | "
            f"acc {row['accuracy']:.3f} F1 {row['macro_f1']:.3f}"
        )
    return rows


def write_rows(rows, path):
    fieldnames = []
    for row in rows:
        fieldnames.extend(key for key in row if key not in fieldnames)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    statuses = [row["collapse_status"] for row in rows]
    failed = np.array([row["failed"] for row in rows])
    accuracy = np.array([row["accuracy"] for row in rows])
    flagged = np.array([status != "healthy" for status in statuses])

    summary = {
        "folds": len(rows),
        "status_counts": {s: statuses.count(s) for s in ["healthy", "partial", "collapsed"]},
        "failed_folds_macro_f1_below": FAILED_MACRO_F1,
        "failed_count": int(failed.sum()),
        "flag_vs_failed": {
            "flagged_and_failed": int((flagged & failed).sum()),
            "flagged_not_failed": int((flagged & ~failed).sum()),
            "missed_failed": int((~flagged & failed).sum()),
        },
        "mean_accuracy_all": float(accuracy.mean()),
        "mean_accuracy_healthy_only": float(accuracy[~flagged].mean()) if (~flagged).any() else None,
        "flagged_by_subject": {},
        "flagged_by_session_seed": {},
        "signals": {},
    }
    for row in rows:
        if row["collapse_status"] == "healthy":
            continue
        subject = str(row["test_subject"])
        run = f"session_{row['session']}_seed_{row['seed']}"
        summary["flagged_by_subject"][subject] = summary["flagged_by_subject"].get(subject, 0) + 1
        summary["flagged_by_session_seed"][run] = summary["flagged_by_session_seed"].get(run, 0) + 1

    # How well does each label-free signal track the label-based outcome?
    # Signals where a higher value means *more* collapse are negated for AUROC.
    signal_direction = {
        "pred_entropy_norm": 1,
        "pred_share_max": -1,
        "target_mean_prob_entropy": 1,
        "target_mean_sample_entropy": 1,
        "target_mean_confidence": -1,
        "target_source_spread_ratio": 1,
        "domain_mean_gap_rel": -1,
        "final_train_mmd": -1,
        "source_accuracy": 1,
    }
    for name, direction in signal_direction.items():
        if name not in rows[0]:
            continue
        values = np.array([row[name] for row in rows], dtype=float)
        entry = {"spearman_vs_accuracy": float(spearmanr(values, accuracy).statistic)}
        if 0 < failed.sum() < len(failed):
            # AUROC for "healthy fold" given the signal; 0.5 is uninformative.
            entry["auroc_healthy"] = float(roc_auc_score(~failed, direction * values))
        summary["signals"][name] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=Path("results/reference_scvae/strict"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/diagnostics/collapse/reference_4s"))
    parser.add_argument("--no-checkpoints", action="store_true", help="Use saved prediction counts only.")
    args = parser.parse_args()

    device = get_device()
    rows = collect(args.results_root, not args.no_checkpoints, device)
    if not rows:
        parser.error(f"no strict fold JSONs found under {args.results_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(rows, args.output_dir / "folds.csv")
    summary = summarize(rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved: {args.output_dir / 'folds.csv'}")
    print(f"Saved: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
