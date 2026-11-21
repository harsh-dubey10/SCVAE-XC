"""Train DCCA-AM across cultures and save every epoch's per-person results.

One run = (source, target, task, seed, architecture). A single network is
trained on the whole source culture; after every epoch it predicts every
target person, and the per-person confusion matrices are saved, so the
paper's selection (best test accuracy over architectures and epochs) and the
fair choice (architecture 0, last epoch) are both computed later from the
same runs, without retraining. Target labels are only used to fill the
confusion matrices.

Windows are shuffled each epoch (seeded); --no-shuffle keeps the released
clip order, which underfits (see src/models/dcca_am.py).

Tasks: emotion (the paper's setting); emotion_newfilms (German <-> French,
source without the 5 shared films); film (German <-> French, identify the
shared film within each of film_identification.TASKS).

    python -m src.experiments.train_dcca_am --source SEED_GER --target SEED_FRA --task emotion --seed 12
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from src.experiments.dcca_data import load_recordings, select
from src.experiments.film_identification import TASKS as FILM_TASKS
from src.experiments.train_scvae_reference import set_seed
from src.models.dcca_am import DCCA_AM, OUTPUT_DIM, make_optimizers, train_epoch

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
FIXED_ARCHITECTURE = (150, 35)


def architecture(seed, index):
    if index == 0:
        return FIXED_ARCHITECTURE
    rng = np.random.default_rng([seed, index])
    return int(rng.integers(100, 200)), int(rng.integers(20, 50))


def run_tasks(task):
    return [f"film_{name}" for name in FILM_TASKS] if task == "film" else [task]


def scaled(recordings):
    """MinMax-scale EEG and eye, fitted on these recordings together (as released)."""
    eeg = np.concatenate([r.eeg for r in recordings])
    eye = np.concatenate([r.eye for r in recordings])
    return MinMaxScaler().fit_transform(eeg), MinMaxScaler().fit_transform(eye)


def run_one(source, target, layer_sizes, categories, epochs, device, seed, index, shuffle=True):
    set_seed(1000 * seed + index)
    to = lambda a, dtype=torch.float: torch.as_tensor(a, dtype=dtype, device=device)
    src_eeg, src_eye = scaled(source)
    x1, x2 = to(src_eeg), to(src_eye)
    y = to(np.concatenate([r.labels for r in source]), torch.long)
    people = {}
    for subject in sorted({r.subject for r in target}):
        recs = [r for r in target if r.subject == subject]
        eeg, eye = scaled(recs)
        people[subject] = (to(eeg), to(eye), np.concatenate([r.labels for r in recs]),
                           np.concatenate([np.full(len(r.labels), 100 * r.session + r.subject) for r in recs]))

    model = DCCA_AM(x1.shape[1], x2.shape[1], [*layer_sizes, OUTPUT_DIM], categories).to(device)
    optimizers = make_optimizers(model)
    generator = torch.Generator().manual_seed(1000 * seed + index) if shuffle else None
    record = {"layer_sizes": list(layer_sizes), "shuffle": shuffle, "skipped_batches": 0, "nonfinite_epochs": 0,
              "confusion": []}
    finite = lambda: all(torch.isfinite(p).all() for p in model.parameters())
    for _ in range(epochs):
        if finite():  # a diverged network is not trained further, but every epoch is still evaluated
            record["skipped_batches"] += train_epoch(model, optimizers, x1, x2, y, generator=generator)
        model.eval()
        if not finite():
            record["nonfinite_epochs"] += 1
        epoch = {}
        for subject, (eeg, eye, labels, _) in people.items():
            pred = model.predict(eeg, eye).cpu().numpy()
            epoch[str(subject)] = confusion_matrix(labels, pred, labels=range(categories)).tolist()
        record["confusion"].append(epoch)
    record["final"] = {str(s): {"pred": model.predict(eeg, eye).cpu().numpy().tolist(), "true": labels.tolist(),
                                "recording": recording.tolist()}
                       for s, (eeg, eye, labels, recording) in people.items()}
    return record


def existing_ok(path, epochs, shuffle):
    """True if ``path`` holds a finished run made with the same epochs and shuffle setting."""
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    return record.get("epochs") == epochs and record.get("shuffle") == shuffle


def write_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".json.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)


def parse_archs(text):
    first, last = map(int, text.split("-"))
    return range(first, last + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=CULTURES, required=True)
    parser.add_argument("--target", choices=CULTURES, required=True)
    parser.add_argument("--task", choices=["emotion", "emotion_newfilms", "film"], default="emotion")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--archs", default="0-20", help="Architecture indices, inclusive range (0 = fixed 150/35).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path("results/dcca_am"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="Feed windows in the released clip order instead of shuffling each epoch.")
    args = parser.parse_args()
    if args.source == args.target:
        parser.error("source and target must differ (cross-culture only)")
    if args.task != "emotion" and {args.source, args.target} != {"SEED_GER", "SEED_FRA"}:
        parser.error(f"--task {args.task} needs SEED_GER <-> SEED_FRA (the shared films)")

    source_all, target_all = load_recordings(args.source), load_recordings(args.target)
    for task in run_tasks(args.task):
        if task == "emotion":
            source, target, categories = source_all, target_all, 3
        elif task == "emotion_newfilms":
            source, target, categories = select(source_all, drop_shared_of=args.source), target_all, 3
        else:
            film_task = task.removeprefix("film_")
            source = select(source_all, film_task=film_task, culture=args.source)
            target = select(target_all, film_task=film_task, culture=args.target)
            categories = len(FILM_TASKS[film_task])
        out_dir = args.output_root / args.target / f"{task}__from_{args.source}" / f"seed_{args.seed}"
        for index in parse_archs(args.archs):
            path = out_dir / f"arch_{index:02d}.json"
            if args.skip_existing and existing_ok(path, args.epochs, not args.no_shuffle):
                continue
            start = time.time()
            record = run_one(source, target, architecture(args.seed, index), categories,
                             args.epochs, args.device, args.seed, index, shuffle=not args.no_shuffle)
            record.update(source=args.source, target=args.target, task=task, seed=args.seed, arch=index,
                          epochs=args.epochs, seconds=round(time.time() - start, 1))
            write_json(path, record)
            last = record["confusion"][-1]
            acc = np.mean([np.trace(np.array(cm)) / np.sum(cm) for cm in last.values()])
            print(f"{task} {args.source}->{args.target} seed {args.seed} arch {index} {record['layer_sizes']}: "
                  f"last-epoch acc {100 * acc:.1f}%, skipped {record['skipped_batches']}, "
                  f"{record['seconds']} s", flush=True)


if __name__ == "__main__":
    main()
