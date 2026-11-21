"""SCVAE-XC in SCVAE-Net's within-SEED settings, on SEED_CHN_MM with the eye branch (step 4 recipe).

cross_subject: per session, train on the other people of that session, test on the held-out person's same session.
cross_session: per person, train on their other two sessions, test on the third.
Each run saves the held-out-epoch prediction and the per-epoch predictions (for SCVAE-Net's test-person epoch rule).

    python -m src.experiments.run_scvae_xc_within [--jobs 4]      # train (skips finished runs), then summarise

Design: docs/superpowers/specs/2026-11-06-scvae-xc-within-seed-design.md.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from src.baselines.norm_fusion import load_culture
from src.experiments.train_scvae_reference import choose_source_val_subjects
from src.experiments.train_scvae_xc import EEG_DIM, EYE_DIMS, train_person

CULTURE, MODALITY, SEEDS, SESSIONS = "SEED_CHN_MM", "eeg_eye", (12, 34, 56), (1, 2, 3)
ROOT = Path("results/scvae_xc/within")
OUT = Path("results/tables/scvae_xc_within_seed_summary.md")
PUBLISHED = {"cross_subject": 95.01, "cross_session": 96.84}


def trainer_args(seed):
    return argparse.Namespace(sources=[CULTURE], modality=MODALITY, seed=seed, epochs=100, batch_size=128,
                              learning_rate=0.0025, input_noise=0.0, weight_decay=1e-4, culture_mmd_weight=0.1,
                              align="none", tta="none", self_train=False, save_epoch_proba=True)


def as_t(a, dtype=torch.float32):
    return torch.tensor(a, dtype=dtype)


def split(data, setting, session, person, seed):
    """(source tuple for train_person, target x, target y): rows chosen by session and person; no test row in source."""
    x = data.x[:, :EEG_DIM + EYE_DIMS[MODALITY]]
    in_session = data.recordings // 1000 == session
    if setting == "cross_subject":
        test = in_session & (data.subjects == person)
        pool = in_session & (data.subjects != person)
        val_people = choose_source_val_subjects(np.unique(data.subjects[pool]).tolist(), 2, seed)
        val = pool & np.isin(data.subjects, val_people)
    else:
        test = in_session & (data.subjects == person)
        pool = ~in_session & (data.subjects == person)
        rng = np.random.default_rng(seed)
        val = pool & (rng.random(len(pool)) < 0.2)
    train = pool & ~val
    src = (as_t(x[train]), as_t(data.y[train], torch.long), as_t(np.zeros(int(train.sum())), torch.long),
           as_t(x[val]), as_t(data.y[val], torch.long), {})
    return src, as_t(x[test]), data.y[test]


def run_one(setting, session, person, seed, device="cpu"):
    out = ROOT / setting / f"session_{session}" / f"subject_{person:02d}" / f"seed_{seed}.npz"
    if out.exists():
        return str(out)
    torch.set_num_threads(2)
    data = load_culture(CULTURE)
    src, tx, ty = split(data, setting, session, person, seed)
    proba, record, epoch_proba = train_person(src, tx, trainer_args(seed), torch.device(device))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, proba=proba.astype(np.float32), epochs=epoch_proba, labels=ty,
                        selected_epoch=record["selected_epoch"])
    return str(out)


def summarise():
    people = np.unique(load_culture(CULTURE).subjects)
    lines = ["# SCVAE-XC in SCVAE-Net's within-SEED settings", "",
             "SEED_CHN_MM (12 Chinese participants, same 15 films), EEG + eye, 3-seed probability ensemble, 100 epochs. "
             "Accuracy %, mean over people per session, then over the 3 sessions. **Test-person epoch** = the epoch with "
             "the best accuracy on each test person, as in SCVAE-Net's code (reads test labels). **Held-out epoch** = "
             "chosen without test data. Design: docs/superpowers/specs/2026-11-06-scvae-xc-within-seed-design.md.", "",
             "| Setting | Session 1 | Session 2 | Session 3 | Average (held-out epoch) | Average (test-person epoch) | SCVAE-Net published (test-person epoch) | runs |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for setting in ("cross_subject", "cross_session"):
        held, peek, cells, n = [], [], [], 0
        for session in SESSIONS:
            h, p = [], []
            for person in people:
                files = [ROOT / setting / f"session_{session}" / f"subject_{person:02d}" / f"seed_{s}.npz" for s in SEEDS]
                if not all(f.exists() for f in files):
                    continue
                saved = [np.load(f) for f in files]
                y = saved[0]["labels"]
                h.append(100 * np.mean(np.mean([s["proba"] for s in saved], 0).argmax(1) == y))
                per_epoch = np.mean([s["epochs"].astype(np.float32) for s in saved], 0)
                p.append(100 * (per_epoch.argmax(2) == y).mean(1).max())
                n += 1
            if h:
                held.append(np.mean(h)); peek.append(np.mean(p))
                cells.append(f"{np.mean(h):.1f} / {np.mean(p):.1f}")
            else:
                cells.append("–")
        avg = lambda v: f"{np.mean(v):.1f}" if len(v) == 3 else "–"
        lines.append(f"| {setting.replace('_', '-')} | " + " | ".join(cells) + f" | {avg(held)} | **{avg(peek)}** | "
                     f"{PUBLISHED[setting]:.2f} | {n}/{len(people) * 3} |")
    lines += ["", "Session cells: held-out epoch / test-person epoch. SCVAE-Net's numbers are on the 15-person EEG-only "
              "SEED release (300 epochs); ours on the 12-person multimodal release with the eye branch (100 epochs)."]
    OUT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--settings", nargs="+", default=["cross_subject", "cross_session"])
    parser.add_argument("--sessions", nargs="+", type=int, default=list(SESSIONS))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not args.summary_only:
        people = np.unique(load_culture(CULTURE).subjects).tolist()
        jobs = [(st, se, p, s) for st in args.settings for se in args.sessions for p in people for s in SEEDS]
        Parallel(n_jobs=args.jobs, verbose=5)(delayed(run_one)(*j, device=args.device) for j in jobs)
    summarise()


if __name__ == "__main__":
    main()
