"""Films-per-emotion learning curve: two training cultures -> the third, varying only how many distinct source films
per emotion the model sees. Also a fixed-window-budget comparison (F = 2 films vs the same number of windows spread
over F = 8 films). StandardScaler + logistic regression, C = 0.01, EEG+eye and EEG only.

    python -m src.experiments.run_film_curve

Design: docs/superpowers/specs/2026-10-29-film-learning-curve-design.md.
"""

import json
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.baselines.norm_fusion import load_culture

CULTURES = ["SEED_CHN_MM", "SEED_GER", "SEED_FRA"]
F_GRID = (1, 2, 3, 4, 6, 8)
DRAWS, C = 10, 0.01
FEATURES = {"eeg_eye": slice(0, 343), "eeg": slice(0, 310)}
OUT = Path("results/film_curve")


def source_pool(target):
    """Pooled source windows with a film id per window; German/French shared films kept only once and never a test film."""
    drop = "SEED_GER" if target == "SEED_FRA" else "SEED_FRA"
    xs, ys, films = [], [], []
    for culture in (c for c in CULTURES if c != target):
        data = load_culture(culture, drop_shared=culture == drop)
        xs.append(data.x); ys.append(data.y)
        films.append(np.array([f"{culture}:{c}" for c in data.clips]))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(films)


def films_by_emotion(y, films):
    return {e: sorted(set(films[y == e])) for e in range(3)}


def draw_films(by_emotion, f, seed):
    rng = np.random.default_rng(seed)
    return sorted(sum((rng.choice(by_emotion[e], size=f, replace=False).tolist() for e in range(3)), []))


def fit_score(x, y, target, rows, features):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=3000))
        model.fit(x[rows][:, features], y[rows])
    correct = model.predict(target.x[:, features]) == target.y
    return float(np.mean([100 * correct[target.subjects == s].mean() for s in np.unique(target.subjects)]))


def run_target(target, n_jobs):
    x, y, films = source_pool(target)
    tgt = load_culture(target)
    by_emotion = films_by_emotion(y, films)
    f_max = min(len(v) for v in by_emotion.values())
    grid = sorted({f for f in F_GRID if f <= f_max} | {f_max})
    jobs, keys = [], []
    for f in grid:
        for d in range(DRAWS):
            rows = np.isin(films, draw_films(by_emotion, f, 1000 * f + d))
            for name, feats in FEATURES.items():
                keys.append(("curve", f, d, name)); jobs.append((rows, feats))
    for d in range(DRAWS):  # fixed budget: the F = 2 draw's window count, spread over F = 8 films
        budget = int(np.isin(films, draw_films(by_emotion, 2, 2000 + d)).sum())
        spread = np.flatnonzero(np.isin(films, draw_films(by_emotion, min(8, f_max), 8000 + d)))
        chosen = np.zeros(len(y), bool)
        chosen[np.random.default_rng(9000 + d).choice(spread, size=min(budget, len(spread)), replace=False)] = True
        for name, feats in FEATURES.items():
            keys.append(("spread", 8, d, name)); jobs.append((chosen, feats))
    scores = Parallel(n_jobs=n_jobs)(delayed(fit_score)(x, y, tgt, rows, feats) for rows, feats in jobs)
    out = {"target": target, "films_per_emotion_available": {e: len(v) for e, v in by_emotion.items()},
           "grid": grid, "results": [dict(zip(("arm", "films", "draw", "features"), k), accuracy=s)
                                      for k, s in zip(keys, scores)]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{target}.json").write_text(json.dumps(out, indent=1))
    return out


def summarize(results):
    lines = ["# Films-per-emotion learning curve", "",
             "Two training cultures → the third; all source people, only the number of distinct source films per "
             "emotion varies. Logistic regression (C = 0.01). Accuracy %, mean over test people, mean ± sd over 10 "
             "random film draws. Design: docs/superpowers/specs/2026-10-29-film-learning-curve-design.md.", ""]
    names = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}
    for feats in FEATURES:
        grid = sorted(set().union(*[r["grid"] for r in results]))
        lines += [f"## {feats}", "", "| Target | " + " | ".join(f"F = {f}" for f in grid) + " | spread (budget of F = 2) |",
                  "|---|" + "---:|" * (len(grid) + 1)]
        mean_at = {f: [] for f in grid}; conc, spr = [], []
        for r in results:
            cells = []
            for f in grid:
                v = [e["accuracy"] for e in r["results"] if e["arm"] == "curve" and e["films"] == f and e["features"] == feats]
                cells.append(f"{np.mean(v):.1f} ± {np.std(v):.1f}" if v else "–")
                if v: mean_at[f].append(np.mean(v))
            s = [e["accuracy"] for e in r["results"] if e["arm"] == "spread" and e["features"] == feats]
            c2 = [e["accuracy"] for e in r["results"] if e["arm"] == "curve" and e["films"] == 2 and e["features"] == feats]
            conc.append(np.mean(c2)); spr.append(np.mean(s))
            lines.append(f"| {names[r['target']]} | " + " | ".join(cells) + f" | {np.mean(s):.1f} ± {np.std(s):.1f} |")
        lines.append("| **Mean** | " + " | ".join(f"**{np.mean(v):.1f}**" if len(v) == len(results) else "–"
                                                for v in mean_at.values()) + f" | **{np.mean(spr):.1f}** |")
        common = [f for f in grid if len(mean_at[f]) == len(results)]
        slope = np.polyfit(np.log2(common), [np.mean(mean_at[f]) for f in common], 1)[0]
        gain = np.mean(mean_at[8]) - np.mean(mean_at[4])
        lines += ["", f"- F = 8 − F = 4: {gain:+.1f} points; spread − concentrated (F = 2 budget): "
                      f"{np.mean(spr) - np.mean(conc):+.1f}; fitted slope {slope:+.1f} points per doubling of F.", ""]
        if feats == "eeg_eye":
            verdict = ("MORE FILMS HELP" if gain >= 2 and np.mean(spr) - np.mean(conc) >= 2
                       else "PLATEAU" if gain < 1 else "INCONCLUSIVE")
            lines += [f"**Pre-registered decision (EEG+eye): {verdict}.**", ""]
    return "\n".join(lines)


def main():
    results = [run_target(t, n_jobs=-1) for t in CULTURES]
    table = summarize(results)
    Path("results/tables/film_curve_summary.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
