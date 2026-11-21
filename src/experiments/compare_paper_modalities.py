"""Compare our strict eye and EEG + eye results with Liu et al. (J. Neural Eng. 2022).

The paper's eye-movement and fusion numbers come from its tables 6 and 7
(within culture, leave-one-subject-out) and 8 and 9 (across cultures). Its eye
features are the same 33 BCMI features on the same 4 s windows as ours, so the
eye comparison is closer to like-for-like than the EEG one (the paper's EEG
used 1 s windows). Its settings were partly chosen with the test labels, so
its numbers are optimistic.

Matched pairs:
  eye      paper linear SVM  vs our linsvm   (fair C from the training people)
  eye      paper DNN         vs our mlp
  eeg+eye  paper Concat      vs our linsvm   (concatenated features + linear SVM)
  eeg+eye  paper DCCA-AM     (their best fusion; no counterpart of ours)

For each of our runs, normalised and not, we report the mean over test people
and a 95% bootstrap CI, and whether the paper's number lies inside it.

    python -m src.experiments.compare_paper_modalities   # -> results/tables/paper_modality_comparison.md
"""
import argparse
from pathlib import Path

import numpy as np

from src.experiments.summarize_crossculture import CULTURE, load_run, summarise

DIRECTIONS = [
    ("SEED_CHN_MM", "SEED_CHN_MM"), ("SEED_GER", "SEED_GER"), ("SEED_FRA", "SEED_FRA"),
    ("SEED_CHN_MM", "SEED_GER"), ("SEED_CHN_MM", "SEED_FRA"),
    ("SEED_GER", "SEED_CHN_MM"), ("SEED_GER", "SEED_FRA"),
    ("SEED_FRA", "SEED_CHN_MM"), ("SEED_FRA", "SEED_GER"),
]
# Liu et al. 2022, accuracy % in DIRECTIONS order (tables 6-9).
PAPER = {
    "eye_svm": [69.94, 77.91, 71.78, 51.76, 57.60, 51.38, 65.41, 51.31, 59.74],
    "eye_dnn": [80.26, 84.28, 79.85, 62.44, 62.38, 56.69, 67.12, 62.99, 65.39],
    "eeg_svm": [72.63, 55.64, 50.10, 47.88, 46.81, 47.39, 47.69, 52.30, 48.88],
    "eeg_dnn": [82.81, 65.87, 64.18, 55.32, 57.28, 64.34, 58.93, 66.19, 60.10],
    "concat": [83.00, 77.30, 71.89, 58.22, 63.91, 52.57, 63.62, 59.75, 62.51],
    "dcca_am": [84.04, 82.07, 79.57, 73.63, 73.84, 84.28, 77.06, 82.09, 77.92],
}
# (label, our modality suffix, our method, paper key)
PAIRS = [
    ("Eye, linear SVM", "_EYE", "linsvm", "eye_svm"),
    ("Eye, MLP vs DNN", "_EYE", "mlp", "eye_dnn"),
    ("EEG + eye, linear SVM vs Concat", "_EEG_EYE", "linsvm", "concat"),
]


def name(culture):
    return CULTURE[culture].split(" ")[0]


def bootstrap_ci(values, rng, n=10000):
    means = rng.choice(values, (n, len(values))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def ours(root, source, target, suffix, method, rng):
    run = f"{method}__pooled" if source == target else f"{method}__from_{source}{suffix}"
    per_subject = load_run(root / f"{target}{suffix}" / run)
    if not per_subject:
        return None
    accuracy = 100 * summarise(per_subject)[0]
    return accuracy.mean(), *bootstrap_ci(accuracy, rng)


def cell(result, paper):
    if result is None:
        return "–"
    mean, low, high = result
    mark = "↑" if low > paper else "↓" if high < paper else "≈"
    return f"{mean:.1f} [{low:.1f}, {high:.1f}] {mark}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("results/crossculture"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/paper_modality_comparison.md"))
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    lines = ["# Eye and EEG + eye: ours (strict) vs Liu et al. 2022", "",
             "Accuracy %, mean over test people [95% bootstrap CI]. ↑ / ↓: the paper's number lies below /",
             "above our CI; ≈: inside it. The paper's settings were partly chosen with test labels.", ""]
    for label, suffix, method, key in PAIRS:
        lines += [f"## {label}", "",
                  "| Train → Test | Paper | Ours, not normalised | Ours, normalised |",
                  "|---|---:|---:|---:|"]
        for (source, target), paper in zip(DIRECTIONS, PAPER[key]):
            raw = ours(args.root, source, target, suffix, method, rng)
            norm = ours(args.root, source, target, suffix, f"{method}__subjnorm", rng)
            lines.append(f"| {name(source)} → {name(target)} | {paper:.1f} | {cell(raw, paper)} | {cell(norm, paper)} |")
        lines.append("")

    # Does each source show eye transferring better than EEG? Paper DNN vs our normalised MLP.
    lines += ["## Eye minus EEG, per direction (paper DNN vs our normalised MLP)", "",
              "| Train → Test | Paper eye − EEG | Ours eye − EEG |", "|---|---:|---:|"]
    for i, (source, target) in enumerate(DIRECTIONS):
        eye = ours(args.root, source, target, "_EYE", "mlp__subjnorm", rng)
        eeg = ours(args.root, source, target, "", "mlp__subjnorm", rng)
        paper = PAPER["eye_dnn"][i] - PAPER["eeg_dnn"][i]
        own = f"{eye[0] - eeg[0]:+.1f}" if eye and eeg else "–"
        lines.append(f"| {name(source)} → {name(target)} | {paper:+.1f} | {own} |")
    lines.append("")

    args.output.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
