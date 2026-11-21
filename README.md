# SCVAE-XC

**Can a model trained on one culture's brain signals read emotions in another culture?**

SCVAE-XC recognises whether a person is watching a positive, neutral or negative film from their **EEG and eye
movements**. It is trained on people from one or two cultures and tested on people from a culture it has never seen
(Chinese, German, French; SEED, SEED-GER, SEED-FRA datasets). It builds on SCVAE-Net, adding an eye-movement branch
and training on more than one culture.

## Results

| Task (trained on 2 cultures, tested on the 3rd) | Accuracy |
|---|---|
| Positive vs negative, one answer per film | **83%** (chance 50%) |
| Positive / neutral / negative, every 4-second window | **67.3%** (chance 33%) |
| Published methods re-run under the same rules: DCCA-AM / SCVAE-Net | 43.8% / 53.3% |

No test labels are used to choose a model or a training epoch.

## Brain map

![Scalp maps](figures/scalp_effect.png)

Each head is seen from above, nose at the top. **Red** means more brain activity during positive films, **blue**
more during negative films. Columns are frequency bands, rows are cultures. In all three cultures, positive films
show more **beta and gamma activity above the ears** (temporal electrodes T7/T8). White circles mark electrodes
that are significant across all 28 people.

## What's where

| Folder | Contents |
|---|---|
| `src/experiments/train_scvae_xc.py` | training SCVAE-XC |
| `src/baselines/` | the methods we compare against |
| `src/experiments/` | analysis, statistics, figures (`scalp_maps.py` makes the brain map) |
| `results/tables/` | result tables |
| `demo/` | small web demo (FastAPI + Gradio) |
| `docs/` | experiment plans and [more details](docs/DETAILS.md) |

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m src.experiments.train_scvae_xc --sources SEED_GER SEED_FRA --target SEED_CHN_MM --modality eeg_eye --seed 12
```

**Data:** the SEED datasets are licensed and not included. Request them from
[BCMI, SJTU](https://bcmi.sjtu.edu.cn/home/seed/) and put the features under `data/raw/`.
