"""SCVAE-XC Stage 2 trainer: SCVAE-fixed across cultures, optionally with an eye branch and two training cultures.

Training follows SCVAE-fixed's strict recipe (train_scvae_reference.py): the released losses and weights, Adam with
weight decay, cosine schedule, shuffled target batches, input noise on the source, the epoch chosen on held-out
source people (BatchNorm re-estimated on the training source), BatchNorm re-estimated on the source before the
single final evaluation, per-recording normalisation, one model per test person. Target labels are read only at
that final evaluation.

  --modality eeg       the SCVAE encoder on the 62 x 5 DE features (re-implementation of SCVAE-fixed)
  --modality eeg_eye   plus a 33-feature eye branch joined before the latent; the decoder reconstructs both
  --modality eeg_pupil the same with only the 12 pupil features (eye columns 0-11, inferred)
  --sources A B        two training cultures, k held-out validation people from each, and an MMD between the two
                       source cultures' latents in every batch (--culture-mmd-weight)
  --align cmmd         step 3: after --align-warmup epochs, an MMD per emotion between the source latents and the
                       confidently pseudo-labelled target latents (--align-confidence)
  --align ca           cmmd plus a per-person equal-emotion term (log 3 - entropy of the batch-mean target prediction)
  --self-train         after selection, v3's class-balanced self-training: up to 3 rounds of adding confident
                       (>= 0.9) target windows, equal per emotion, with predicted labels, then 10 epochs of fine-tuning
  --tta im|ent         step 5: after training, adapt the encoder to the test person without labels (classifier
                       frozen, eval mode): information maximisation (im) or prediction entropy only (ent)
  --save-epoch-proba   also save the test person's predictions after every epoch (subject_<id>_epochs.npz), for
                       the separately labelled test-label-selected column; training is unchanged

Writes results/scvae_xc/stage2/<target>/<arm>__from_<sources>/session_all/seed_<s>/subject_<id>{.json,_proba.npz}
(the proba files match summarize_scvae_improve.seed_files).

    python -m src.experiments.train_scvae_xc --sources SEED_GER --target SEED_FRA --modality eeg_eye --seed 12

Plan: docs/superpowers/plans/2026-10-24-scvae-xc.md (Stage 2, steps 4, 3 and 5).
"""

import argparse
import copy
import json
import time
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.norm_fusion import load_culture
from src.experiments.train_scvae_reference import (LOSS_WEIGHTS, add_input_noise, choose_source_val_subjects,
                                                   get_device, set_seed)
from src.models.scvae_reference import (MultiKernelMMD, ReferenceDecoder, ReferenceEncoder, ReferenceSupConLoss,
                                        kl_loss, recalibrate_batchnorm, reconstruction_loss)

EEG_DIM, EYE_HIDDEN = 310, 64
EYE_DIMS = {"eeg": 0, "eeg_eye": 33, "eeg_pupil": 12}
VAL_PEOPLE = {"SEED_CHN_MM": 3, "SEED_GER": 2, "SEED_FRA": 2}
# Self-training: v3's rounds, threshold and minimum (src/baselines/norm_fusion.py); fine-tuning per round.
ST_ROUNDS, ST_THRESHOLD, ST_MIN, ST_EPOCHS, ST_LR_SCALE = 3, 0.9, 10, 10, 0.1


class XCSCVAE(nn.Module):
    """ReferenceSCVAE with an optional eye branch of eye_dim features; forward returns (reconstruction, mu, logvar, logits)."""

    def __init__(self, eye_dim, latent_dim=256, num_classes=3):
        super().__init__()
        self.eye = eye_dim > 0
        self.eeg_encoder = ReferenceEncoder(latent_dim=latent_dim)
        joined = 128 * 5 + (EYE_HIDDEN if self.eye else 0)
        if self.eye:
            self.eye_encoder = nn.Sequential(nn.Linear(eye_dim, EYE_HIDDEN), nn.ReLU(inplace=True))
            self.eye_decoder = nn.Linear(latent_dim, eye_dim)
        self.fc_mu, self.fc_logvar = nn.Linear(joined, latent_dim), nn.Linear(joined, latent_dim)
        self.decoder = ReferenceDecoder(latent_dim=latent_dim)
        self.view_one = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True))
        self.view_two = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(inplace=True))
        self.classifier = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.BatchNorm1d(latent_dim),
                                        nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(latent_dim, num_classes))

    def encode(self, x):
        enc = self.eeg_encoder
        h = enc.relu(enc.conv1(x[:, :EEG_DIM].reshape(-1, 62, 5)))
        h = h + enc.attention(h)
        h = enc.relu(enc.conv2(h)).flatten(start_dim=1)
        if self.eye:
            h = torch.cat([h, self.eye_encoder(x[:, EEG_DIM:])], dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decoder(z).flatten(start_dim=1)
        if self.eye:
            recon = torch.cat([recon, self.eye_decoder(z)], dim=1)
        return recon, mu, logvar, self.classifier(mu)

    def contrastive_views(self, mu):
        return torch.stack([self.view_one(mu), self.view_two(mu)], dim=1)


def features(culture, modality):
    return culture.x[:, :EEG_DIM + EYE_DIMS[modality]]


def source_split(sources, modality, seed):
    """(train x, y, culture index, val x, val y, held-out people per culture)."""
    parts = {"tx": [], "ty": [], "tc": [], "vx": [], "vy": []}
    held = {}
    for i, (name, culture) in enumerate(sources):
        val = choose_source_val_subjects(np.unique(culture.subjects).tolist(), VAL_PEOPLE[name], seed)
        held[name] = val
        is_val = np.isin(culture.subjects, val)
        x = features(culture, modality)
        parts["tx"].append(x[~is_val]); parts["ty"].append(culture.y[~is_val])
        parts["tc"].append(np.full(int((~is_val).sum()), i))
        parts["vx"].append(x[is_val]); parts["vy"].append(culture.y[is_val])
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    as_t = lambda a, dtype=torch.float32: torch.tensor(a, dtype=dtype)
    return (as_t(cat["tx"]), as_t(cat["ty"], torch.long), as_t(cat["tc"], torch.long), as_t(cat["vx"]),
            as_t(cat["vy"], torch.long), held)


@torch.no_grad()
def predict(model, x, device, batch=512):
    model.eval()
    return torch.cat([torch.softmax(model(x[i:i + batch].to(device))[3], 1).cpu() for i in range(0, len(x), batch)])


@torch.no_grad()
def predict_mean(model, x, device, batch=512):
    """Like predict, from the latent mean only: no random draw, so calling it does not change training."""
    model.eval()
    return torch.cat([torch.softmax(model.classifier(model.encode(x[i:i + batch].to(device))[0]), 1).cpu()
                      for i in range(0, len(x), batch)])


def val_accuracy(model, train_x, val_x, val_y, device, target_x=None):
    """Source-validation accuracy, and (if target_x is given) the target predictions of the same scored copy."""
    scored = recalibrate_batchnorm(copy.deepcopy(model), train_x, device)
    acc = float((predict(scored, val_x, device).argmax(1) == val_y).float().mean())
    return acc, (None if target_x is None else predict_mean(scored, target_x, device).numpy())


def entropy(p):
    return -(p * torch.log(p + 1e-8)).sum(1)


def adapt_to_person(model, target_x, args, device):
    """Step 5: a few label-free passes over the person's windows updating the encoder only (SHOT / TENT style)."""
    head = set(model.classifier.parameters())
    params = [p for name, p in model.named_parameters()
              if p not in head and not name.startswith(("decoder", "eye_decoder", "view_", "fc_logvar"))]
    optimizer = torch.optim.Adam(params, lr=args.tta_learning_rate)
    loader = DataLoader(TensorDataset(target_x), batch_size=args.batch_size, shuffle=True)
    model.eval()
    for _ in range(args.tta_passes):
        for (bx,) in loader:
            p = torch.softmax(model.classifier(model.encode(bx.to(device))[0]), 1)
            loss = entropy(p).mean()
            if args.tta == "im":
                loss = loss - entropy(p.mean(0, keepdim=True)).squeeze() + np.log(p.shape[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def class_mmd(mmd, s_mu, by, t_mu, t_logits, confidence, minimum=4):
    """Mean over emotions of the MMD between source latents of an emotion and confident target latents given it."""
    prob, label = torch.softmax(t_logits.detach(), 1).max(1)
    keep = prob >= confidence
    terms = [mmd(s_mu[by == k], t_mu[keep & (label == k)]) for k in range(t_logits.shape[1])
             if (by == k).sum() >= minimum and (keep & (label == k)).sum() >= minimum]
    return torch.stack(terms).mean() if terms else None


def balance_loss(t_logits):
    """log K - entropy of the batch-mean prediction: zero when the batch's emotions are predicted equally often."""
    mean = torch.softmax(t_logits, 1).mean(0)
    return np.log(t_logits.shape[1]) + (mean * torch.log(mean + 1e-8)).sum()


def train_epoch(model, optimizer, source_loader, target_loader, losses, epoch, args, device):
    """One pass of the step 4 losses (plus step 3 alignment if on) over min(source, target) batches."""
    mmd, supcon = losses
    two = len(args.sources) == 2
    model.train()
    target_iter, source_iter = cycle(target_loader), iter(source_loader)
    for _ in range(min(len(source_loader), len(target_loader))):
        bx, by, bc = (t.to(device) for t in next(source_iter))
        tx = next(target_iter)[0].to(device)
        s_recon, s_mu, s_logvar, s_logits = model(add_input_noise(bx, args.input_noise))
        t_recon, t_mu, t_logvar, t_logits = model(tx)
        total = (LOSS_WEIGHTS["reconstruction"] * (reconstruction_loss(s_recon, bx) + reconstruction_loss(t_recon, tx))
                 + LOSS_WEIGHTS["kl"] * (kl_loss(s_mu, s_logvar) + kl_loss(t_mu, t_logvar))
                 + LOSS_WEIGHTS["classification"] * nn.functional.cross_entropy(s_logits, by)
                 + getattr(args, "supcon_weight", LOSS_WEIGHTS["supcon"]) * supcon(model.contrastive_views(s_mu), by)
                 + getattr(args, "mmd_weight", LOSS_WEIGHTS["mmd"]) * mmd(s_mu, t_mu))
        if two and (bc == 0).sum() > 1 and (bc == 1).sum() > 1:
            total = total + args.culture_mmd_weight * mmd(s_mu[bc == 0], s_mu[bc == 1])
        if args.align != "none" and epoch > args.align_warmup:
            conditional = class_mmd(mmd, s_mu, by, t_mu, t_logits, args.align_confidence)
            if conditional is not None:
                total = total + LOSS_WEIGHTS["mmd"] * conditional
            if args.align == "ca":
                total = total + args.balance_weight * balance_loss(t_logits)
        optimizer.zero_grad()
        total.backward()
        optimizer.step()


def self_train(model, src, target_x, losses, args, device):
    """v3's class-balanced self-training on the selected model: up to ST_ROUNDS rounds of adding the top-k confident
    target windows per predicted emotion (k = smallest pool, stop if k < ST_MIN) to the source set with their
    predicted labels (culture index 2: outside the between-culture MMD), then ST_EPOCHS fixed epochs of fine-tuning
    at ST_LR_SCALE x the base rate. Returns the windows added per round."""
    sx, sy, sc = src[:3]
    target_loader = DataLoader(TensorDataset(target_x), batch_size=args.batch_size, shuffle=True)
    added = []
    for _ in range(ST_ROUNDS):
        recalibrate_batchnorm(model, sx, device)
        confidence, pseudo = predict(model, target_x, device).max(1)
        pools = [torch.where((pseudo == k) & (confidence >= ST_THRESHOLD))[0] for k in range(3)]
        k = min(len(pool) for pool in pools)
        if k < ST_MIN:
            break
        chosen = torch.cat([pool[torch.argsort(-confidence[pool])[:k]] for pool in pools])
        added.append(int(len(chosen)))
        loader = DataLoader(TensorDataset(torch.cat([sx, target_x[chosen]]), torch.cat([sy, pseudo[chosen]]),
                                          torch.cat([sc, torch.full((len(chosen),), 2, dtype=sc.dtype)])),
                            batch_size=args.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=ST_LR_SCALE * args.learning_rate,
                                     weight_decay=args.weight_decay)
        for _ in range(ST_EPOCHS):
            train_epoch(model, optimizer, loader, target_loader, losses, args.epochs, args, device)
    recalibrate_batchnorm(model, sx, device)
    return added


def train_person(src, target_x, args, device, model_out=None):
    """Probabilities on target_x of one strict SCVAE-XC model, and its training record. If model_out is a list, the
    trained model is appended to it (used to export the demo model)."""
    set_seed(args.seed)
    sx, sy, sc, vx, vy, held = src
    source_loader = DataLoader(TensorDataset(sx, sy, sc), batch_size=args.batch_size, shuffle=True)
    target_loader = DataLoader(TensorDataset(target_x), batch_size=args.batch_size, shuffle=True)
    model = XCSCVAE(eye_dim=EYE_DIMS[args.modality]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    losses = (MultiKernelMMD(), ReferenceSupConLoss(temperature=0.07))
    best, best_state, best_epoch, curve, epoch_proba = -1.0, None, None, [], []
    for epoch in range(1, args.epochs + 1):
        train_epoch(model, optimizer, source_loader, target_loader, losses, epoch, args, device)
        scheduler.step()
        acc, target_proba = val_accuracy(model, sx, vx, vy, device, target_x if args.save_epoch_proba else None)
        if target_proba is not None:
            epoch_proba.append(target_proba.astype(np.float16))
        curve.append(round(acc, 4))
        if acc > best:
            best, best_epoch = acc, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    recalibrate_batchnorm(model, sx, device)
    record = {"selected_epoch": best_epoch, "source_val_curve": curve, "source_val_people": held}
    if args.self_train:
        record["self_train_added"] = self_train(model, src, target_x, losses, args, device)
    if args.tta != "none":
        adapt_to_person(model, target_x, args, device)
        record["tta"] = {"objective": args.tta, "passes": args.tta_passes, "learning_rate": args.tta_learning_rate}
    if model_out is not None:
        model_out.append(model)
    return predict(model, target_x, device).numpy(), record, (np.stack(epoch_proba) if epoch_proba else None)


def arm_name(args):
    suffix = (("" if args.align == "none" else f"_{args.align}") + ("_st" if args.self_train else "")
              + ("" if args.tta == "none" else f"_tta_{args.tta}"))
    return f"{ {'eeg': 'eeg', 'eeg_eye': 'eye', 'eeg_pupil': 'pupil'}[args.modality]}_{len(args.sources)}src{suffix}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--modality", choices=list(EYE_DIMS), default="eeg")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--test-subject", type=int, help="One test person; default: all, sequentially.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--input-noise", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--culture-mmd-weight", type=float, default=0.1)
    parser.add_argument("--supcon-weight", type=float, default=LOSS_WEIGHTS["supcon"],
                        help="Supervised contrastive loss weight (ablation: 0).")
    parser.add_argument("--mmd-weight", type=float, default=LOSS_WEIGHTS["mmd"],
                        help="Source-target MMD weight (ablation: 0).")
    parser.add_argument("--align", choices=["none", "cmmd", "ca"], default="none")
    parser.add_argument("--align-warmup", type=int, default=10)
    parser.add_argument("--align-confidence", type=float, default=0.9)
    parser.add_argument("--balance-weight", type=float, default=1.0)
    parser.add_argument("--tta", choices=["none", "im", "ent"], default="none")
    parser.add_argument("--tta-passes", type=int, default=2)
    parser.add_argument("--tta-learning-rate", type=float, default=1e-4)
    parser.add_argument("--self-train", action="store_true", help="v3's class-balanced self-training after selection.")
    parser.add_argument("--save-epoch-proba", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("results/scvae_xc/stage2"))
    args = parser.parse_args()
    assert args.target not in args.sources and 1 <= len(args.sources) <= 2
    device = get_device() if args.device == "auto" else torch.device(args.device)
    src = source_split([(s, load_culture(s)) for s in args.sources], args.modality, args.seed)
    target = load_culture(args.target)
    run_dir = (args.output_root / args.target / f"{arm_name(args)}__from_{'+'.join(args.sources)}" / "session_all"
               / f"seed_{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    people = [args.test_subject] if args.test_subject else np.unique(target.subjects).tolist()
    for person in people:
        out = run_dir / f"subject_{person:02d}.json"
        if out.exists():
            print(f"skip {out}", flush=True)
            continue
        start = time.time()
        mine = target.subjects == person
        x = torch.tensor(features(target, args.modality)[mine], dtype=torch.float32)
        y, trials = target.y[mine], target.clips[mine]
        proba, record, epoch_proba = train_person(src, x, args, device)
        if epoch_proba is not None:
            np.savez_compressed(run_dir / f"subject_{person:02d}_epochs.npz", proba=epoch_proba, labels=y, trials=trials)
        accuracy = float((proba.argmax(1) == y).mean())
        np.savez(run_dir / f"subject_{person:02d}_proba.npz", proba=proba.astype(np.float32), labels=y, trials=trials)
        out.write_text(json.dumps({**record, "accuracy": accuracy, "arm": arm_name(args), "sources": args.sources,
                                   "target": args.target, "seed": args.seed, "test_subject": person,
                                   "settings": {k: v for k, v in vars(args).items() if k not in ("output_root",)},
                                   "target_labels_used_only_after_training": True}, default=str, indent=1))
        print(f"{'+'.join(args.sources)} -> {args.target} {arm_name(args)} seed {args.seed} person {person}: "
              f"{100 * accuracy:.1f}% (epoch {record['selected_epoch']}) [{time.time() - start:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
