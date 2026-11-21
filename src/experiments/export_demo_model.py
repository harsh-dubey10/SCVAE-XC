"""Train one SCVAE-XC model for the class demo and export it, plus the test culture's recordings.

The model is trained on the two other cultures (labelled) and shown on --target, an unseen culture whose windows are
used only unlabelled, for the MMD alignment, as in training. Step 4 recipe, seed 12, epoch chosen on held-out
training people. Writes, to <out>/models/<chinese|german|french>/:
  weights.pt   plain weights for scvae_xc_model.py (which must already be in <out>)
  data.npz     the target culture's windows (per-recording normalised, float16) with person, session, clip, label
  card.json    training setup and the model's accuracy on the target culture

    python -m src.experiments.export_demo_model --out ~/Downloads/SCVAE-XC_Demo --target SEED_GER
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from src.baselines.norm_fusion import load_culture
from src.experiments.train_scvae_reference import get_device
from src.experiments.train_scvae_xc import EEG_DIM, EYE_DIMS, source_split, train_person

CULTURES = {"SEED_CHN_MM": "chinese", "SEED_GER": "german", "SEED_FRA": "french"}
MODALITY = "eeg_eye"
KEEP = ("eeg_encoder.conv1", "eeg_encoder.attention", "eeg_encoder.conv2", "eye_encoder", "fc_mu", "classifier")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=list(CULTURES), required=True)
    parser.add_argument("--epochs", type=int, default=100)
    cli = parser.parse_args()
    sources = [c for c in CULTURES if c != cli.target]
    out = cli.out.expanduser() / "models" / CULTURES[cli.target]
    out.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(sources=sources, modality=MODALITY, seed=12, epochs=cli.epochs, batch_size=128,
                              learning_rate=0.0025, input_noise=0.0, weight_decay=1e-4, culture_mmd_weight=0.1,
                              align="none", tta="none", self_train=False, save_epoch_proba=False)
    src = source_split([(s, load_culture(s)) for s in sources], MODALITY, args.seed)
    target = load_culture(cli.target)
    x = torch.tensor(target.x[:, :EEG_DIM + EYE_DIMS[MODALITY]], dtype=torch.float32)
    models = []
    proba, record, _ = train_person(src, x, args, get_device(), model_out=models)
    model = models[0].to("cpu").eval()
    torch.save({k: v for k, v in model.state_dict().items() if k.startswith(KEEP)}, out / "weights.pt")

    spec = importlib.util.spec_from_file_location("scvae_xc_model", cli.out.expanduser() / "scvae_xc_model.py")
    standalone = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(standalone)
    with torch.no_grad():
        check = standalone.load(out / "weights.pt")(x[:512]).numpy()
    assert np.allclose(check, proba[:512], atol=1e-4), "standalone model disagrees with the trained model"

    np.savez_compressed(out / "data.npz", x=target.x[:, :343].astype(np.float16), person=target.subjects,
                        session=target.recordings // 1000, clip=target.clips, label=target.y)
    names = [CULTURES[s].capitalize() for s in sources]
    card = {"model": "SCVAE-XC (eye branch)", "test_culture": CULTURES[cli.target].capitalize(),
            "trained_on": f"{names[0]} + {names[1]} (labelled)",
            "tested_on": f"{CULTURES[cli.target].capitalize()}: labels never used in training; windows used unlabelled "
                         "for alignment",
            "recipe": vars(args), "selected_epoch": record["selected_epoch"],
            "classes": ["negative", "neutral", "positive"],
            "input": "343 = 310 DE (62 ch x 5 bands) + 33 eye, per-recording normalised"}
    (out / "card.json").write_text(json.dumps(card, indent=1, default=str))
    acc = 100 * float(np.mean([(proba[target.subjects == s].argmax(1) == target.y[target.subjects == s]).mean()
                               for s in np.unique(target.subjects)]))
    print(json.dumps({"target": cli.target, "selected_epoch": record["selected_epoch"], "window_accuracy": round(acc, 1)}))


if __name__ == "__main__":
    main()
