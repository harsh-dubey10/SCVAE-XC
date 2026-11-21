"""Gradio frontend for the SCVAE-XC demo. Talks to the FastAPI backend over HTTP.

    API_URL=http://127.0.0.1:8000 python app.py            # or set API_URL to the ngrok URL
"""

import os

import gradio as gr
import matplotlib
import numpy as np
import requests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
COLORS = {"negative": "#E03131", "neutral": "#A79B95", "positive": "#2F9E44"}
DAY_CHOICES = ["Day 1", "Day 2", "Day 3", "All 3 days pooled"]


def api(method, path, **kw):
    r = requests.request(method, API_URL + path, timeout=60, headers={"ngrok-skip-browser-warning": "1"}, **kw)
    r.raise_for_status()
    return r.json()


def timeline(per_window, days, title):
    """Probability of each emotion for every 4-s window; dashed lines separate the recording days."""
    p, days = np.asarray(per_window), np.asarray(days)
    t = np.arange(len(p)) * 4 / 60
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for i, c in enumerate(["negative", "neutral", "positive"]):
        ax.plot(t, p[:, i], color=COLORS[c], label=c, linewidth=2)
    starts = [0] + [i for i in range(1, len(days)) if days[i] != days[i - 1]]
    for k, s in enumerate(starts):
        if k:
            ax.axvline(t[s], color="#444444", linestyle="--", linewidth=1)
        ax.text(t[s] + 0.02, 1.02, f"Day {days[s]}", fontsize=9, color="#444444", transform=ax.get_xaxis_transform())
    ax.set_xlabel("time (minutes): each point is one 4-second window")
    ax.set_ylabel("probability")
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=16)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def result_text(out):
    mark = lambda ok: "✅ correct" if ok else "❌ wrong"
    lines = [f"### Model says: **{out['answer_3class'].upper()}**",
             f"True emotion of this film: **{out['true_label']}** → {mark(out['correct_3class'])}",
             f"Windows used: {out['n_windows']} (day {', '.join(map(str, out['days']))})"]
    if out["correct_pos_vs_neg"] is not None:
        lines.append(f"Positive vs negative answer: **{out['answer_pos_vs_neg']}** → {mark(out['correct_pos_vs_neg'])}")
    else:
        lines.append("Neutral film: the positive-vs-negative metric does not apply.")
    lines.append(f"_Model trained on {out['trained_on']}; never saw this culture's labels._")
    return "\n\n".join(lines)


def culture_info(culture, health):
    m = health["models"][culture]
    a = m["accuracy"]
    return (f"**Test culture: {culture.capitalize()}**. Model trained on {m['trained_on']}. "
            f"On all {culture.capitalize()} participants: **{a['window_3class']}%** per 4-s window (3 classes), "
            f"**{a['film_3class_3days']}%** per film with the 3 days pooled, "
            f"**{a['film_pos_vs_neg_3days']}%** positive vs negative per film.")


def build():
    health = api("GET", "/health")
    cultures = api("GET", "/cultures")
    default = "german" if "german" in cultures else cultures[0]

    def choices(culture):
        rec = api("GET", "/recordings", params={"culture": culture})
        return [str(p) for p in rec["people"]], [f"Clip {c}" for c in sorted(map(int, rec["clips"]))]

    def on_culture(culture):
        people, clips = choices(culture)
        return (gr.update(choices=people, value=people[0]), gr.update(choices=clips, value=clips[0]),
                culture_info(culture, health))

    def predict_demo(culture, person, clip, day):
        session = None if day == DAY_CHOICES[-1] else DAY_CHOICES.index(day) + 1
        out = api("POST", "/predict_demo", json={"culture": culture, "person": int(person),
                                                  "clip": int(clip.split()[1]), "session": session})
        title = f"{culture.capitalize()} participant {person}, {clip}, {day.lower()}"
        return out["mean_probability"], result_text(out), timeline(out["per_window"], out["window_day"], title)

    def predict_file(culture, file):
        if file is None:
            raise gr.Error("Upload a .npy or .csv file with n rows x 343 features.")
        path = file if isinstance(file, str) else file.name
        x = np.atleast_2d(np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=","))
        out = api("POST", "/predict", json={"culture": culture, "windows": x.tolist()})
        text = (f"### Model says: **{out['answer_3class'].upper()}**\n\n{len(x)} windows; "
                f"positive vs negative: **{out['answer_pos_vs_neg']}**")
        return out["mean_probability"], text, timeline(out["per_window"], [1] * len(x), "Uploaded windows")

    people, clips = choices(default)
    with gr.Blocks(title="SCVAE-XC emotion demo") as demo:
        gr.Markdown("# Decoding the Feels, Across Cultures\n"
                    "**SCVAE-XC** reads 4-second windows of EEG + eye-tracking features and predicts "
                    "**negative / neutral / positive**. Each model is trained on two cultures and tested on the "
                    "third, which it never saw with labels.")
        with gr.Tab("Demo recordings"):
            with gr.Row():
                culture = gr.Dropdown(cultures, value=default, label="Test culture (trained on the other two)")
                person = gr.Dropdown(people, value=people[0], label="Participant")
                clip = gr.Dropdown(clips, value=clips[0], label="Film clip")
                day = gr.Radio(DAY_CHOICES, value=DAY_CHOICES[-1], label="Recording day")
            info = gr.Markdown(culture_info(default, health))
            go = gr.Button("Predict emotion", variant="primary")
            with gr.Row():
                probs = gr.Label(label="Average probability", num_top_classes=3)
                verdict = gr.Markdown()
            plot = gr.Plot(label="Prediction over time")
            gr.Markdown("**How to read the graph:** each point is one 4-second window. The three lines are the model's "
                        "probability for negative (red), neutral (grey) and positive (green). The answer above is the "
                        "average over all windows. With all 3 days pooled, dashed lines mark where each recording day starts.")
            culture.change(on_culture, [culture], [person, clip, info])
            go.click(predict_demo, [culture, person, clip, day], [probs, verdict, plot])
        with gr.Tab("Upload your own features"):
            gr.Markdown("A `.npy` or `.csv` file with one row per 4-s window and 343 columns "
                        "(310 EEG DE + 33 eye), normalised per recording.")
            culture2 = gr.Dropdown(cultures, value=default, label="Use the model tested on")
            f = gr.File(label="Feature file")
            go2 = gr.Button("Predict", variant="primary")
            probs2, verdict2, plot2 = gr.Label(num_top_classes=3), gr.Markdown(), gr.Plot()
            go2.click(predict_file, [culture2, f], [probs2, verdict2, plot2])
        gr.Markdown(f"Backend: `{API_URL}` (FastAPI)")
    return demo


if __name__ == "__main__":
    build().launch(share=os.environ.get("SHARE", "0") == "1")
