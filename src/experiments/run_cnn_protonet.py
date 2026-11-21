"""CNN and prototypical-network few-shot study, scored exactly like the calibration study.

For every cross-culture direction and seed, four networks are trained on the source culture only:
{MLP, CNN} encoder x {classifier, prototypical network}. The CNN reads the EEG as 5 bands on a 9 x 9 electrode
map. The prototypical network is trained in episodes that mimic calibration: one source person, K films per
emotion from their first session as support, their windows of other films as query. Each test person is then
scored with the calibration study's pools and splits (same test windows for every N); target labels are read
only on calibration windows. Variants (0.5 / 0.5 averages):

  N = 0    v3, mlp_clf, cnn_clf
  N >= 1   v3+person, mlp_clf+person, cnn_clf+person, mlp_proto, cnn_proto, mlp_proto+v3, cnn_proto+v3,
           cnn_proto+cnn_clf

Writes results/cnn_protonet/scores/<target>/cross__from_<source>/seed_<s>.json in the calibration score format.

    python -m src.experiments.run_cnn_protonet --task 0 --seed 12
    python -m src.experiments.run_cnn_protonet --all

Design: docs/superpowers/specs/2026-10-24-cnn-protonet-design.md.
"""

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.baselines.mlp import set_seed
from src.baselines.norm_fusion import load_culture
from src.experiments.run_calibration import DRAWS, NS, _score, calibrated_proba, film_pool, split
from src.experiments.run_norm_fusion_v3 import CULTURES

SEEDS = [12, 34, 56]
TASKS = list(itertools.permutations(CULTURES, 2))
EMBED, DROPOUT, LR, WEIGHT_DECAY = 64, 0.3, 1e-3, 1e-4
CLF_EPOCHS, BATCH = 30, 256
EPISODES, QUERY = 3000, 192
# SEED 62-channel order on a 9 x 9 scalp grid (row, column).
GRID_ROWS = [[None, None, None, "FP1", "FPZ", "FP2", None, None, None],
             [None, None, None, "AF3", None, "AF4", None, None, None],
             ["F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8"],
             ["FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8"],
             ["T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8"],
             ["TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8"],
             ["P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8"],
             [None, "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", None],
             [None, "CB1", None, "O1", "OZ", "O2", None, "CB2", None]]
CHANNELS = ["FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FC5",
            "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2",
            "P4", "P6", "P8", "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ", "O2", "CB2"]
_CELL = {name: (r, c) for r, row in enumerate(GRID_ROWS) for c, name in enumerate(row) if name}
GRID_R = torch.tensor([_CELL[ch][0] for ch in CHANNELS])
GRID_C = torch.tensor([_CELL[ch][1] for ch in CHANNELS])


def to_grid(eeg):
    """(n, 310) channel-major DE features (62 channels x 5 bands) -> (n, 5, 9, 9); empty cells are 0."""
    grid = eeg.new_zeros(len(eeg), 5, 9, 9)
    grid[:, :, GRID_R, GRID_C] = eeg.view(-1, 62, 5).transpose(1, 2)
    return grid


class Encoder(nn.Module):
    """EEG branch (MLP or CNN) and eye branch, joined into an EMBED-dimensional embedding."""

    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.eeg = nn.Sequential(nn.Conv2d(5, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1),
                                     nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(DROPOUT),
                                     nn.Linear(64 * 4 * 4, 128), nn.ReLU())
        else:
            self.eeg = nn.Sequential(nn.Linear(310, 256), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(256, 128),
                                     nn.ReLU())
        self.eye = nn.Sequential(nn.Linear(33, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(DROPOUT), nn.Linear(128 + 64, EMBED))

    def forward(self, x):
        eeg = to_grid(x[:, :310]) if self.kind == "cnn" else x[:, :310]
        return self.head(torch.cat([self.eeg(eeg), self.eye(x[:, 310:])], dim=1))


class Classifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder, self.out = Encoder(kind), nn.Linear(EMBED, 3)

    def forward(self, x):
        return self.out(torch.relu(self.encoder(x)))


class ProtoNet(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.encoder, self.log_scale = Encoder(kind), nn.Parameter(torch.zeros(()))

    def logits(self, query, prototypes):
        return -self.log_scale.exp() * torch.cdist(query, prototypes) ** 2


def fit_classifier(x, y, kind, seed):
    set_seed(seed)
    model = Classifier(kind)
    xs, ys = torch.tensor(x), torch.tensor(y)
    weights = torch.tensor(len(y) / (3 * np.bincount(y, minlength=3)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(CLF_EPOCHS):
        model.train()
        for idx in torch.randperm(len(y), generator=generator).split(BATCH):
            optimizer.zero_grad()
            loss_fn(model(xs[idx]), ys[idx]).backward()
            optimizer.step()
    return model.eval()


def episode(rng, y, clips, subjects, recordings, people):
    """(support indices per emotion, query indices) for one random source person and K in {1, 2}."""
    person = rng.choice(people)
    mine = subjects == person
    first = mine & (recordings // 1000 == (recordings[mine] // 1000).min())
    k = int(rng.integers(1, 3))
    films = {e: rng.choice(np.unique(clips[first & (y == e)]), k, replace=False) for e in range(3)}
    support = [np.where(first & np.isin(clips, films[e]))[0] for e in range(3)]
    query = np.where(mine & ~np.isin(clips, np.concatenate(list(films.values()))))[0]
    return support, rng.choice(query, min(QUERY, len(query)), replace=False)


def fit_protonet(x, y, clips, subjects, recordings, kind, seed):
    set_seed(seed)
    model = ProtoNet(kind)
    xs, ys = torch.tensor(x), torch.tensor(y)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    rng, people = np.random.default_rng(seed), np.unique(subjects)
    model.train()
    for _ in range(EPISODES):
        support, query = episode(rng, y, clips, subjects, recordings, people)
        emb = model.encoder(xs[np.concatenate(support + [query])])
        sizes = [len(s) for s in support]
        prototypes = torch.stack([part.mean(0) for part in emb[:sum(sizes)].split(sizes)])
        optimizer.zero_grad()
        nn.functional.cross_entropy(model.logits(emb[sum(sizes):], prototypes), ys[query]).backward()
        optimizer.step()
    return model.eval()


def proto_proba(model, emb, y, calib, evaluate):
    """Probabilities on the `evaluate` windows from prototypes of the calibration windows (labels read there only)."""
    with torch.no_grad():
        emb = torch.tensor(emb)
        prototypes = torch.stack([emb[calib & (y == e)].mean(0) for e in range(3)])
        return torch.softmax(model.logits(emb[evaluate], prototypes), dim=1).numpy()


def person_scores(probas, embeddings, models, x, y, clips, subjects, recordings, person, draw):
    """{N: {variant: scores}} for one person and one calibration pool (pools and splits as in run_calibration)."""
    pool = film_pool(y, clips, subjects, person, draw)
    out = {}
    for n in NS:
        calib, test = split(y, clips, subjects, recordings, person, pool, n)
        score = lambda proba: _score(y, recordings, test, proba)
        if n == 0:
            out[n] = {name: score(probas[name][test]) for name in ("v3", "mlp_clf", "cnn_clf")}
            continue
        out[n] = {f"{name}+person": score(calibrated_proba("combined", probas[name], x, y, calib, test, n))
                  for name in ("v3", "mlp_clf", "cnn_clf")}
        proto = {kind: proto_proba(models[kind], embeddings[kind], y, calib, test) for kind in ("mlp", "cnn")}
        for kind in ("mlp", "cnn"):
            out[n][f"{kind}_proto"] = score(proto[kind])
            out[n][f"{kind}_proto+v3"] = score(0.5 * proto[kind] + 0.5 * probas["v3"][test])
        out[n]["cnn_proto+cnn_clf"] = score(0.5 * proto["cnn"] + 0.5 * probas["cnn_clf"][test])
    return out


def run(task, seed, base_root, output_root):
    source_name, target_name = TASKS[task]
    out = output_root / target_name / f"cross__from_{source_name}" / f"seed_{seed}.json"
    if out.exists():
        print(f"skip {out}", flush=True)
        return
    start = time.time()
    source, target = load_culture(source_name), load_culture(target_name)
    base = np.load(base_root / target_name / f"cross__from_{source_name}" / f"seed_{seed}.npz")
    for field in ("y", "clips", "subjects", "recordings"):
        assert np.array_equal(getattr(target, field), base[field]), f"v3 base windows do not line up ({field})"
    scaler = StandardScaler().fit(source.x)
    src_x = scaler.transform(source.x).astype(np.float32)
    tgt_x = torch.tensor(scaler.transform(target.x).astype(np.float32))
    probas, embeddings, models = {"v3": base["proba"]}, {}, {}
    for kind in ("mlp", "cnn"):
        classifier = fit_classifier(src_x, source.y, kind, seed)
        with torch.no_grad():
            probas[f"{kind}_clf"] = torch.softmax(classifier(tgt_x), 1).numpy()
        models[kind] = fit_protonet(src_x, source.y, source.clips, source.subjects, source.recordings, kind, seed)
        with torch.no_grad():
            embeddings[kind] = models[kind].encoder(tgt_x).numpy()
    scores = {}
    for person in np.unique(target.subjects):
        for draw in range(DRAWS):
            result = person_scores(probas, embeddings, models, target.x, target.y, target.clips, target.subjects,
                                   target.recordings, person, draw)
            scores.setdefault(str(person), {})[str(draw)] = {str(n): v for n, v in result.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores))
    print(f"{source_name} -> {target_name} seed {seed}: done [{(time.time() - start) / 60:.1f} min]", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--task", type=int, help="Direction index 0..5 (see TASKS).")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--base-root", type=Path, default=Path("results/calibration/base"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cnn_protonet/scores"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    for task in range(len(TASKS)) if args.all else [args.task]:
        for seed in args.seed:
            run(task, seed, args.base_root, args.output_root)


if __name__ == "__main__":
    main()
