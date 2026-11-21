"""Stimulus-locked contrastive (SLC) pretraining of the SCVAE encoder, without emotion labels.

Within a culture every recording's windows are time-aligned: the same film has the same number of windows for
every person and session. Windows of two different recordings at the same moment of the same film are a positive
pair; other (film, moment) keys in the batch are negatives (NT-Xent). Each batch comes from one culture,
alternating source and target, so the loss never separates the cultures. Both cultures' windows are used
unlabelled (target: transductive, like MMD); emotion labels are never read during pretraining.

A linear-probe diagnostic follows: logistic regression on the source embeddings (source labels), scored on the
target per person (target labels used only to score), for the SLC encoder, a random-initialised encoder and the
raw features.

Writes checkpoints/slc/<target>/from_<source>/seed_<s>.pt (encoder state dict) and
results/slc_pretrain/<target>/from_<source>/seed_<s>.json.

    python -m src.experiments.pretrain_slc --source SEED_GER --target SEED_FRA --seed 12

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md (Stage 1).
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.baselines.norm_fusion import fit_lr
from src.experiments.train_scvae_reference import (available_subjects, get_device, normalize_recording, sessions_in,
                                                   set_seed)
from src.models.scvae_reference import ReferenceEncoder

STEPS, KEYS_PER_BATCH, JITTER, NOISE = 3000, 128, 1, 0.1
TEMPERATURE, LR, WEIGHT_DECAY, LATENT, PROJECTION = 0.1, 1e-3, 1e-4, 256, 128
PROBE_C = 0.1


def load_windows(culture):
    """(x (n, 62, 5) normalised per recording, labels, clips, moments, recordings, subjects) of one culture.

    The moment of a window is its position within its film in that recording; films must have the same number
    of windows in every recording (checked)."""
    root = Path("data/processed") / culture
    parts = defaultdict(list)
    lengths = {}
    for session in sessions_in(root):
        for subject in available_subjects(session, root):
            data = torch.load(root / str(session) / f"subject{subject}.pth", weights_only=True)
            clips = np.asarray(data["trials"])
            moments = np.zeros(len(clips), dtype=np.int64)
            for clip in np.unique(clips):
                where = np.where(clips == clip)[0]
                assert np.all(np.diff(where) == 1), f"{culture} {session}/{subject}: film {clip} not contiguous"
                assert lengths.setdefault(int(clip), len(where)) == len(where), f"{culture}: film {clip} not aligned"
                moments[where] = np.arange(len(where))
            parts["x"].append(normalize_recording(torch.stack(data["data"]).float()))
            parts["y"].append(np.asarray(data["labels"]))
            parts["clips"].append(clips)
            parts["moments"].append(moments)
            parts["recordings"].append(np.full(len(clips), 1000 * session + subject))
            parts["subjects"].append(np.full(len(clips), subject))
    return (torch.cat(parts["x"]), *[np.concatenate(parts[k]) for k in ("y", "clips", "moments", "recordings",
                                                                          "subjects")])


class KeySampler:
    """Batches of (view one, view two) window indices: KEYS_PER_BATCH distinct (film, moment) keys, two different
    recordings per key, the second view's moment jittered by up to JITTER windows within the film."""

    def __init__(self, clips, moments, rng):
        self.rng = rng
        self.index = defaultdict(list)
        for i, key in enumerate(zip(clips.tolist(), moments.tolist())):
            self.index[key].append(i)
        self.index = {k: np.array(v) for k, v in self.index.items()}
        self.keys = [k for k, v in self.index.items() if len(v) >= 2]

    def batch(self):
        chosen = self.rng.choice(len(self.keys), KEYS_PER_BATCH, replace=False)
        one, two = [], []
        for c in chosen:
            clip, moment = self.keys[c]
            first = self.rng.choice(self.index[(clip, moment)])
            jittered = (clip, moment + int(self.rng.integers(-JITTER, JITTER + 1)))
            pool = self.index.get(jittered, self.index[(clip, moment)])
            pool = pool[pool != first]
            if len(pool) == 0:
                pool = self.index[(clip, moment)][self.index[(clip, moment)] != first]
            one.append(first)
            two.append(self.rng.choice(pool))
        return np.array(one), np.array(two)


def nt_xent(a, b, temperature=TEMPERATURE):
    z = F.normalize(torch.cat([a, b]), dim=1)
    sim = z @ z.T / temperature
    sim.fill_diagonal_(float("-inf"))
    n = len(a)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(a.device)
    return F.cross_entropy(sim, target)


def pretrain(cultures, seed, device):
    """Train encoder + projection head with NT-Xent, alternating the cultures batch by batch."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = ReferenceEncoder(latent_dim=LATENT).to(device)
    head = nn.Sequential(nn.Linear(LATENT, LATENT), nn.ReLU(), nn.Linear(LATENT, PROJECTION)).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=LR,
                                 weight_decay=WEIGHT_DECAY)
    samplers = [KeySampler(c[3], c[4], rng) for c in cultures]
    xs = [c[0].to(device) for c in cultures]
    curve = []
    for step in range(STEPS):
        which = step % len(cultures)
        one, two = samplers[which].batch()
        views = [xs[which][idx] for idx in (one, two)]
        views = [v + NOISE * torch.randn_like(v) for v in views]
        loss = nt_xent(*[head(encoder(v)[0]) for v in views])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == STEPS - 1:
            curve.append([step, loss.item()])
    return encoder.eval(), curve


@torch.no_grad()
def embed(encoder, x, device, batch=2048):
    return torch.cat([encoder(x[i:i + batch].to(device))[0].cpu() for i in range(0, len(x), batch)]).numpy()


def probe(train_x, train_y, test_x, test_y, test_subjects):
    """Mean over target people of the window accuracy of a logistic regression fitted on the source."""
    pred = fit_lr(train_x, train_y, PROBE_C).predict(test_x)
    return float(np.mean([(pred[test_subjects == s] == test_y[test_subjects == s]).mean()
                          for s in np.unique(test_subjects)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoints/slc"))
    parser.add_argument("--output-root", type=Path, default=Path("results/slc_pretrain"))
    args = parser.parse_args()
    device = get_device() if args.device == "auto" else torch.device(args.device)
    start = time.time()
    source, target = load_windows(args.source), load_windows(args.target)
    encoder, curve = pretrain([source, target], args.seed, device)
    ckpt = args.checkpoint_root / args.target / f"from_{args.source}" / f"seed_{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, ckpt)

    set_seed(args.seed)
    random_encoder = ReferenceEncoder(latent_dim=LATENT).to(device).eval()
    probes = {"raw": probe(source[0].flatten(1).numpy(), source[1], target[0].flatten(1).numpy(), target[1], target[5])}
    for name, enc in (("random_encoder", random_encoder), ("slc_encoder", encoder)):
        probes[name] = probe(embed(enc, source[0], device), source[1], embed(enc, target[0], device), target[1],
                             target[5])
    out = args.output_root / args.target / f"from_{args.source}" / f"seed_{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": args.source, "target": args.target, "seed": args.seed,
                               "checkpoint": str(ckpt), "loss_curve": curve, "probe_accuracy": probes,
                               "settings": {"steps": STEPS, "keys_per_batch": KEYS_PER_BATCH, "jitter": JITTER,
                                            "noise": NOISE, "temperature": TEMPERATURE, "lr": LR,
                                            "weight_decay": WEIGHT_DECAY, "probe_c": PROBE_C}}, indent=1))
    print(f"{args.source} -> {args.target} seed {args.seed}: loss {curve[0][1]:.2f} -> {curve[-1][1]:.2f}; probe "
          + ", ".join(f"{k} {100 * v:.1f}%" for k, v in probes.items())
          + f" [{(time.time() - start) / 60:.1f} min]", flush=True)


if __name__ == "__main__":
    main()
