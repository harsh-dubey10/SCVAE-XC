# SCVAE-XC live demo: Colab + FastAPI + Gradio

Trained SCVAE-XC models served by a **FastAPI** backend, with a **Gradio** frontend that calls it over HTTP.
There is one model per **test culture**, chosen with the "Test culture" dropdown. Each is trained on the other two
cultures and shown on a culture it never saw with labels.

```
Gradio frontend  --HTTP POST-->  FastAPI backend (/predict_demo, /predict)  -->  SCVAE-XC model (PyTorch)
      (public gradio.live link)        (optional public ngrok URL)
```

## Files

| File | What it is |
|---|---|
| `SCVAE_XC_Demo_Colab.ipynb` | The Colab notebook: install → upload → start FastAPI → (optional ngrok) → launch Gradio |
| `SCVAE-XC_Demo.zip` (in Downloads) | This whole folder zipped, to upload into Colab |
| `api.py` | FastAPI backend. Docs at `/docs` |
| `app.py` | Gradio frontend (calls the API via `API_URL`) |
| `run_local.py` | **Backup:** runs backend + frontend on the laptop with one command |
| `scvae_xc_model.py` | The SCVAE-XC inference model in plain PyTorch |
| `models/chinese/` | Model trained on German + French, with the Chinese recordings (12 people × 3 days × 15 clips) |
| `models/german/` | Model trained on Chinese + French, with the German recordings (8 people × 3 days × 18 clips) |
| `models/french/` | Model trained on Chinese + German, with the French recordings (8 people × 3 days × 21 clips) |

Each `models/<culture>/` folder has `weights.pt`, `data.npz` (4-s windows, 343 features) and `card.json`.

To retrain one (from the EEG-Emotion repository):
`python -m src.experiments.export_demo_model --out ~/Downloads/SCVAE-XC_Demo --target SEED_GER` (or `SEED_CHN_MM`, `SEED_FRA`).

## Option A: Google Colab (as planned)

1. Open https://colab.research.google.com → **File → Upload notebook** → `SCVAE_XC_Demo_Colab.ipynb`.
2. **Runtime → Run all.** When the upload button appears, choose `SCVAE-XC_Demo.zip` from Downloads.
3. Step 4 (optional): paste a free ngrok token (https://dashboard.ngrok.com) to get a public FastAPI URL and show the
   interactive API docs (`.../docs`). Leave it empty to skip; the demo works without it.
4. Step 6 prints a public **`https://….gradio.live`** link: open it on the projector.

A GPU is not needed: prediction takes milliseconds on CPU.

## Option B: laptop backup (no Colab needed)

```bash
cd ~/Downloads/SCVAE-XC_Demo
pip install -r requirements.txt      # torch, fastapi, uvicorn, gradio, requests, matplotlib
python run_local.py                  # opens the demo in the browser and prints the FastAPI /docs URL
python run_local.py --share          # also prints a public gradio.live link
```

The backend picks a free port from 8765 upward (port 8000 is already used by another server on this laptop).

## What to show in class (about 2 minutes)

1. **Pick the test culture** in the dropdown: "this model never saw these people's labels; it learned from the other two cultures."
2. **Correct examples to start with** (all 3 days pooled, unless noted):

| Test culture | Positive | Negative | Neutral |
|---|---|---|---|
| german | participant 1, Clip 1 | participant 1, Clip 2 | participant 1, Clip 10 |
| chinese | participant 2, Clip 6 | participant 1, Clip 3 | participant 2, Clip 2 |
| french | participant 1, Clip 5 (also Day 2 alone) | participant 1, Clip 7 | participant 1, Clip 6 |

3. **Explain the graph:** each point is one 4-second window; the three lines are the model's probability of negative (red),
   neutral (grey) and positive (green). The answer is the average over all windows. With "All 3 days pooled", dashed
   lines mark where each recording day starts.
4. **Show day-to-day noise:** pick a film, switch between Day 1, Day 2, Day 3, then "All 3 days pooled". Single days can
   disagree; pooling the days is more reliable (the "one answer per film" slide).
5. Optional: open the FastAPI `/docs` page and run `/predict_demo` there, to show that the frontend just calls an API.
6. **Be honest if it is wrong:** some films are misclassified, as expected at about 70% accuracy.

## Accuracy of the demo models (each on its whole test culture)

| Test culture (trained on the other two) | 3 classes, per 4-s window | 3 classes, per film, 3 days | Pos vs neg, per film, 3 days |
|---|---:|---:|---:|
| Chinese | 73.2% | 79.4% | 84.2% |
| German | 69.5% | 73.6% | 78.1% |
| French | 67.2% | 78.0% | 80.4% |

These are single models (one seed, trained once with all test-culture windows used unlabelled for alignment). The
slides use the full evaluation (one model per test person, 3 seeds averaged: 67.3% per window, 83.0% per film for
positive vs negative), so the numbers differ slightly.

## API reference

- `GET /health`: each model, what it was trained on, and the accuracy above
- `GET /cultures`: `["chinese", "french", "german"]`
- `GET /recordings?culture=german`: people, days (1–3) and clips with their true emotion
- `POST /predict_demo`: `{"culture": "german", "person": 1, "clip": 2, "session": null}` (`session` 1–3, or `null` for all 3 days)
- `POST /predict`: `{"culture": "german", "windows": [[343 numbers], ...]}` for your own normalised feature windows

Every response includes per-window probabilities, the average probability, the 3-class answer and the positive-vs-negative answer.
