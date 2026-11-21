"""FastAPI backend for the SCVAE-XC emotion demo.

    uvicorn api:app --host 0.0.0.0 --port 8000      # docs at http://localhost:8000/docs

One SCVAE-XC model per test culture (models/<chinese|german|french>/), each trained on the other two cultures.

Endpoints
  GET  /health                     models, what each was trained on, and its accuracy on its test culture
  GET  /cultures                   the available test cultures
  GET  /recordings?culture=german  people, sessions (days) and film clips of that culture
  POST /predict_demo               {"culture": "german", "person": 3, "clip": 5, "session": 1}  (session null = 3 days pooled)
  POST /predict                    {"culture": "german", "windows": [[343 numbers], ...]}       (your own feature windows)
"""

import json
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import scvae_xc_model

HERE = Path(__file__).resolve().parent
CLASSES = scvae_xc_model.CLASSES


def scores(proba, d):
    """Accuracy on the whole test culture: per 4-s window, and per film with the 3 days pooled."""
    window = 100 * float(np.mean(proba.argmax(1) == d["label"]))
    three, pn = [], []
    for p in np.unique(d["person"]):
        for c in np.unique(d["clip"]):
            rows = (d["person"] == p) & (d["clip"] == c)
            mean, truth = proba[rows].mean(0), int(d["label"][rows][0])
            three.append(mean.argmax() == truth)
            if truth != 1:
                pn.append((2 if mean[2] > mean[0] else 0) == truth)
    return {"window_3class": round(window, 1), "film_3class_3days": round(100 * float(np.mean(three)), 1),
            "film_pos_vs_neg_3days": round(100 * float(np.mean(pn)), 1)}


MODELS = {}
for folder in sorted((HERE / "models").iterdir()):
    if not (folder / "weights.pt").exists():
        continue
    model = scvae_xc_model.load(folder / "weights.pt")
    d = dict(np.load(folder / "data.npz"))
    d["x"] = d["x"].astype(np.float32)
    with torch.no_grad():  # each demo set is small: predict it once at startup
        proba = model(torch.tensor(d["x"])).numpy()
    MODELS[folder.name] = {"model": model, "data": d, "proba": proba,
                           "card": json.loads((folder / "card.json").read_text()), "accuracy": scores(proba, d)}

app = FastAPI(title="SCVAE-XC emotion recognition demo",
              description="EEG + eye features -> negative / neutral / positive. Each model is trained on two "
                          "cultures and shown on the third, which it never saw with labels.")


class Windows(BaseModel):
    culture: str = "german"
    windows: list[list[float]] = Field(..., description="n windows x 343 features (310 EEG DE + 33 eye), normalised per recording")


class DemoRequest(BaseModel):
    culture: str = "german"
    person: int
    clip: int
    session: int | None = Field(None, description="1, 2 or 3; null pools all three recording days")


def get(culture):
    if culture not in MODELS:
        raise HTTPException(404, f"unknown culture {culture!r}; available: {list(MODELS)}")
    return MODELS[culture]


def summarise(proba):
    mean = proba.mean(0)
    return {"per_window": proba.round(4).tolist(),
            "per_window_label": [CLASSES[i] for i in proba.argmax(1)],
            "mean_probability": {c: round(float(p), 4) for c, p in zip(CLASSES, mean)},
            "answer_3class": CLASSES[int(mean.argmax())],
            "answer_pos_vs_neg": "positive" if mean[2] > mean[0] else "negative"}


@app.get("/health")
def health():
    return {"status": "ok", "models": {name: {"trained_on": m["card"]["trained_on"], "tested_on": m["card"]["tested_on"],
                                              "accuracy": m["accuracy"]} for name, m in MODELS.items()}}


@app.get("/cultures")
def cultures():
    return list(MODELS)


@app.get("/recordings")
def recordings(culture: str = "german"):
    d = get(culture)["data"]
    clips = {int(c): CLASSES[int(d["label"][d["clip"] == c][0])] for c in np.unique(d["clip"])}
    return {"people": [int(p) for p in np.unique(d["person"])], "sessions": [1, 2, 3], "clips": clips}


@app.post("/predict")
def predict(req: Windows):
    x = np.asarray(req.windows, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 343:
        raise HTTPException(422, f"expected n x 343 features, got {list(x.shape)}")
    with torch.no_grad():
        return summarise(get(req.culture)["model"](torch.tensor(x)).numpy())


@app.post("/predict_demo")
def predict_demo(req: DemoRequest):
    m = get(req.culture)
    d = m["data"]
    rows = (d["person"] == req.person) & (d["clip"] == req.clip)
    if req.session is not None:
        rows &= d["session"] == req.session
    if not rows.any():
        raise HTTPException(404, "no windows for that person / clip / session")
    out = summarise(m["proba"][rows])
    truth = CLASSES[int(d["label"][rows][0])]
    out.update({"culture": req.culture, "trained_on": m["card"]["trained_on"], "true_label": truth,
                "n_windows": int(rows.sum()), "window_day": [int(s) for s in d["session"][rows]],
                "days": [int(s) for s in np.unique(d["session"][rows])],
                "correct_3class": out["answer_3class"] == truth,
                "correct_pos_vs_neg": None if truth == "neutral" else out["answer_pos_vs_neg"] == truth})
    return out
