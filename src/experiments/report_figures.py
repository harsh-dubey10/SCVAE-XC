"""Figures and benchmark numbers for the IEEE course report (report/), from saved results only.

    python -m src.experiments.report_figures   # -> report/figures/*.pdf, *.png and report/figures/numbers.json

Palette: dataviz reference categorical slots (blue = SCVAE-XC, orange = NormFusion, aqua = third series),
neutral grey for published / baseline methods.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from src.baselines.norm_fusion import load_culture
from src.experiments.run_norm_fusion_v3 import SEEDS as V3_SEEDS
from src.experiments.run_xc_v3_combination import CACHE
from src.experiments.significance_tests import dcca, one_culture, others, scvae_net, two_culture, v3, xc
from src.experiments.summarize_scvae_improve import seed_files
from src.experiments.summarize_scvae_xc_stage2 import CULTURES, ROOT as XC_ROOT

OUT = Path("report/figures")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, LIGHT_GREY, INK, INK2 = "#8a8985", "#c9c8c3", "#0b0b0b", "#52514e"
RNG = np.random.default_rng(0)
NAMES = {"SEED_CHN_MM": "Chinese", "SEED_GER": "German", "SEED_FRA": "French"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})
COL = 3.5  # IEEE column width, inches


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def liu_svm(source, target):
    data = json.loads(Path("results/crossculture/liu_paper_protocol_modalities_4s_full.json").read_text())
    return {int(p): 100 * r["source_cv"] for p, r in data[f"eeg_eye:{source}->{target}"].items()
            if isinstance(r, dict) and "source_cv" in r}


def culture_mean(scores):
    """Mean over cultures of the mean over that culture's people (the convention of the tables)."""
    return float(np.mean([np.mean([v for (c, _), v in scores.items() if c == t]) for t in CULTURES]))


def ci(values, n=10000):
    m = RNG.choice(np.asarray(values), (n, len(values))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b):
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    lo, hi = ci(d)
    return {"delta": float(d.mean()), "lo": lo, "hi": hi, "p": float(wilcoxon(d).pvalue), "n": len(keys)}


def benchmark(numbers):
    one = {"DCCA-AM (re-impl.)": one_culture(dcca), "Linear SVM (re-impl.)": one_culture(liu_svm),
           "SCVAE-Net (re-impl.)": one_culture(scvae_net), "SCVAE-XC (ours)": one_culture(lambda s, t: xc(t, "eye_1src", [s])),
           "NormFusion (ours)": one_culture(lambda s, t: v3(t, "emotion", [s]))}
    two = {"SCVAE-XC (ours)": two_culture(lambda t: xc(t, "eye_2src", others(t))),
           "NormFusion (ours)": two_culture(lambda t: v3(t, "emotion_two_cultures", others(t)))}
    numbers["benchmark"] = {
        "one": {k: {"mean": culture_mean(v), "ci": ci(list(v.values()))} for k, v in one.items()},
        "two": {k: {"mean": culture_mean(v), "ci": ci(list(v.values()))} for k, v in two.items()},
        "tests": {"xc1_vs_svm": paired(one["SCVAE-XC (ours)"], one["Linear SVM (re-impl.)"]),
                  "v3_1_vs_svm": paired(one["NormFusion (ours)"], one["Linear SVM (re-impl.)"])}}
    fig, axes = plt.subplots(2, 1, figsize=(COL, 2.9), gridspec_kw={"height_ratios": [5, 2.2]})
    colors = {"SCVAE-XC (ours)": BLUE, "NormFusion (ours)": ORANGE}
    for ax, (title, group) in zip(axes, (("One training culture → another", one), ("Two training cultures → the third", two))):
        labels = list(group)
        for i, k in enumerate(labels):
            vals = list(group[k].values())
            m, (lo, hi) = culture_mean(group[k]), ci(vals)
            c = colors.get(k, GREY)
            ax.barh(i, m, height=0.6, color=c, edgecolor="white", linewidth=1)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            ax.text(hi + 1.2, i, f"{m:.1f}", va="center", color=INK, fontsize=7.5)
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(100 / 3, color=INK2, lw=0.6, ls=(0, (2, 2)))
        ax.set_xlim(0, 90)
        ax.set_title(title, fontsize=7.5, color=INK, loc="left")
        ax.grid(axis="x", color="#e6e5e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].text(100 / 3 + 0.8, -0.55, "chance", fontsize=6.5, color=INK2)
    axes[1].set_xlabel("3-class accuracy (%)")
    fig.tight_layout(h_pad=0.8)
    save(fig, "benchmark")


def leakage(numbers):
    k = [1, 2, 5, 10, 20]
    table = {"Chinese → German": [49.2, 52.4, 55.2, 58.1, 58.9, 35.3], "Chinese → French": [56.7, 59.3, 61.0, 63.3, 64.3, 46.1],
             "German → Chinese": [58.8, 60.6, 66.0, 68.1, 69.2, 47.2], "German → French": [53.8, 58.9, 62.6, 63.7, 65.5, 49.8],
             "French → Chinese": [63.0, 64.7, 68.5, 70.3, 72.3, 44.1], "French → German": [57.7, 59.2, 63.6, 65.2, 68.5, 40.3]}
    arr = np.array(list(table.values()))  # from results/tables/dcca_am_summary.md
    mean = arr.mean(0)
    numbers["leakage_curve_mean"] = mean.tolist()
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    for row in arr:
        ax.plot(k, row[:5], color=LIGHT_GREY, lw=0.8)
    ax.plot(k, mean[:5], color=INK, lw=2, marker="o", ms=4, label="test-label selected (mean of 6 directions)")
    ax.axhline(mean[5], color=BLUE, lw=2, label=f"fair: fixed architecture, last epoch ({mean[5]:.1f})")
    ax.axhline(78.1, color=GREY, lw=1, ls=(0, (4, 2)), label="published DCCA-AM (78.1)")
    ax.set_xscale("log")
    ax.set_xticks(k, [str(x) for x in k])
    ax.set_xlabel("architectures searched on the test person's labels, k")
    ax.set_ylabel("3-class accuracy (%)")
    ax.set_ylim(30, 85)
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=1)
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    save(fig, "leakage")


def film_curve():
    f = [1, 2, 3, 4, 6, 8]  # results/tables/film_curve_summary.md, EEG + eye, logistic regression
    per = {"Chinese": [54.2, 57.6, 61.5, 65.2, 67.7, 66.8], "German": [48.8, 53.6, 52.9, 56.4, 55.3, 57.5],
           "French": [47.2, 55.0, 56.5, 55.7, 59.7, 60.2]}
    mean = [50.1, 55.4, 57.0, 59.1, 60.9, 61.5]
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for name, row in per.items():
        ax.plot(f, row, color=LIGHT_GREY, lw=0.8)
        ax.text(8.3, row[-1], name, fontsize=6.5, color=INK2, va="center")
    ax.plot(f, mean, color=BLUE, lw=2, marker="o", ms=4)
    for x, y in zip(f, mean):
        if x in (1, 2, 4, 8):
            ax.text(x, y + 1.0, f"{y:.1f}", ha="center", va="bottom", fontsize=6.5, color=INK,
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=5)
    ax.set_xscale("log")
    ax.set_xticks(f, [str(x) for x in f])
    ax.set_xlim(0.9, 11)
    ax.set_xlabel("distinct training films per emotion")
    ax.set_ylabel("3-class accuracy (%)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.5)
    save(fig, "film_curve")


def pooling():
    # results/tables/per_film_accuracy_summary.md (two cultures) and scvae_xc_step5 test-label column.
    levels = ["4-s window", "film, one day", "film, 3 days"]
    data = {"SCVAE-XC": {"3-class": [67.3, 68.4, 76.2], "pos/neg": [76.3, 77.2, 83.0]},
            "NormFusion": {"3-class": [70.0, 70.1, 78.5], "pos/neg": [79.3, 78.3, 84.4]}}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.8), sharey=True)
    for ax, task in zip(axes, ("3-class", "pos/neg")):
        x = np.arange(3)
        for j, (model, c) in enumerate((("SCVAE-XC", BLUE), ("NormFusion", ORANGE))):
            vals = data[model][task]
            ax.bar(x + (j - 0.5) * 0.36, vals, width=0.34, color=c, edgecolor="white", linewidth=1, label=model)
            for xi, v in zip(x, vals):
                ax.text(xi + (j - 0.5) * 0.36, v + 0.8, f"{v:.0f}", ha="center", fontsize=6, color=INK)
        ax.set_xticks(x, levels, fontsize=6.5, rotation=15)
        ax.set_title("3 classes (chance 33)" if task == "3-class" else "positive vs negative (chance 50)", fontsize=7)
        ax.set_ylim(30, 92)
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy (%)")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "pooling")


def reliability(numbers):
    conf_hit = {"SCVAE-XC, one network": [], "SCVAE-XC, 3 seeds": [], "NormFusion, 5 seeds": []}
    for t in CULTURES:
        target = load_culture(t)
        name = "+".join(others(t))
        files = seed_files(XC_ROOT / t / f"eye_2src__from_{name}")
        v3p = np.mean([np.load(CACHE / t / f"emotion_two_cultures__from_{name}" / f"seed_{s}.npz")["proba"]
                       for s in V3_SEEDS], 0)
        for person, saved in files.items():
            m = target.subjects == person
            y = target.y[m]
            for key, p in (("SCVAE-XC, 3 seeds", np.mean([s["proba"] for s in saved], 0)),
                           ("NormFusion, 5 seeds", v3p[m])):
                conf_hit[key].append((p.max(1), p.argmax(1) == y))
            for s in saved:
                conf_hit["SCVAE-XC, one network"].append((s["proba"].max(1), s["proba"].argmax(1) == y))
    edges = np.linspace(1 / 3, 1, 9)
    fig, ax = plt.subplots(figsize=(COL * 0.62, 2.1))
    ax.plot([0.33, 1], [0.33, 1], color=INK2, lw=0.6, ls=(0, (2, 2)))
    numbers["reliability"] = {}
    for (key, pairs), c in zip(conf_hit.items(), (AQUA, BLUE, ORANGE)):
        conf = np.concatenate([p[0] for p in pairs])
        hit = np.concatenate([p[1] for p in pairs]).astype(float)
        idx = np.clip(np.digitize(conf, edges) - 1, 0, len(edges) - 2)
        xs, ys = [], []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() > 200:
                xs.append(conf[sel].mean()); ys.append(hit[sel].mean())
        ax.plot(xs, ys, color=c, lw=2, marker="o", ms=3, label=key)
        numbers["reliability"][key] = {"mean_conf": float(conf.mean()), "acc": float(hit.mean())}
    ax.set_xlabel("confidence (top probability)")
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.33, 1); ax.set_ylim(0.33, 1)
    ax.legend(fontsize=6, loc="upper left")
    ax.grid(color="#e6e5e0", lw=0.5)
    save(fig, "reliability")


def architecture():
    fig, ax = plt.subplots(figsize=(COL * 2, 1.55))
    ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis("off")

    def box(x, y, w, h, text, fc="#f3f2ee", ec=INK2, bold=False):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
                                                       fc=fc, ec=ec, lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, color=INK,
                fontweight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2))

    ax.set_ylim(0, 36)
    box(0.5, 19, 13, 8, "EEG: DE features\n62 ch × 5 bands")
    box(0.5, 4, 13, 8, "Eye: 33 features\npupil, fixation,\nsaccade, blink", fc="#e3effb", ec=BLUE)
    box(18, 19, 15, 8, "Conv1d + SE channel\nattention + Conv1d")
    box(18, 4, 15, 8, "Eye branch\nLinear 33→64, ReLU", fc="#e3effb", ec=BLUE)
    box(38, 11, 14, 9, "Latent z\n$\\mathcal{N}(\\mu,\\sigma^2)$, 256-d", bold=True)
    box(33, 26, 24, 8, "MMD on $\\mu$ ($\\mathcal{L}_{MMD}$): sources ↔ test\nperson (unlabelled); culture A ↔ B",
        fc="#e3effb", ec=BLUE)
    box(60, 22, 19, 6.5, "Decoder: rebuild EEG + eye\n($\\mathcal{L}_{rec}$, $\\mathcal{L}_{KL}$)")
    box(60, 12, 19, 6.5, "Classifier: pos / neu / neg\n($\\mathcal{L}_{CE}$)")
    box(60, 2, 19, 6.5, "Projection heads\n($\\mathcal{L}_{SupCon}$)")
    box(83, 8, 16.5, 15, "Training: 2 source\ncultures + test\nperson's windows;\nepoch chosen on\nheld-out source people")
    arrow(13.8, 23, 17.7, 23); arrow(13.8, 8, 17.7, 8)
    arrow(33.3, 23, 37.7, 17.5); arrow(33.3, 8, 37.7, 13.5)
    arrow(45, 20.3, 45, 25.7)
    for y in (25.2, 15.2, 5.2):
        arrow(52.3, 15.5, 59.7, y)
    ax.text(0.5, 32, "blue = added by SCVAE-XC", fontsize=6, color=BLUE)
    save(fig, "architecture")


def xc_vs_v3(numbers):
    """(a) two-culture accuracy per test culture; (b) gain from adding a second training culture."""
    xc2 = two_culture(lambda t: xc(t, "eye_2src", others(t)))
    xc1 = one_culture(lambda s, t: xc(t, "eye_1src", [s]))
    v2 = two_culture(lambda t: v3(t, "emotion_two_cultures", others(t)))
    v1 = one_culture(lambda s, t: v3(t, "emotion", [s]))
    panels = {"a": {"SCVAE-XC": xc2, "NormFusion": v2},
              "b": {"SCVAE-XC": {k: xc2[k] - xc1[k] for k in xc2}, "NormFusion": {k: v2[k] - v1[k] for k in v2}}}
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.9), gridspec_kw={"width_ratios": [1, 1]})
    numbers["xc_vs_v3"] = {}
    for ax, (key, title) in zip(axes, (("a", "(a) two cultures → third"), ("b", "(b) gain from a 2nd culture"))):
        x = np.arange(3)
        for j, (model, c) in enumerate((("SCVAE-XC", BLUE), ("NormFusion", ORANGE))):
            vals, los, his = [], [], []
            for t in CULTURES:
                v = [s for (cu, _), s in panels[key][model].items() if cu == t]
                m, (lo, hi) = float(np.mean(v)), ci(v)
                vals.append(m); los.append(m - lo); his.append(hi - m)
            numbers["xc_vs_v3"][f"{key}:{model}"] = vals
            pos = x + (j - 0.5) * 0.36
            ax.bar(pos, vals, width=0.34, color=c, edgecolor="white", linewidth=1, label=model)
            ax.errorbar(pos, vals, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5)
        ax.set_xticks(x, [NAMES[t] for t in CULTURES], fontsize=6.5)
        ax.set_title(title, fontsize=7, loc="left")
        ax.grid(axis="y", color="#e6e5e0", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylim(40, 85); axes[0].set_ylabel("accuracy (%)")
    axes[1].set_ylabel("points"); axes[1].axhline(0, color=INK2, lw=0.6)
    axes[0].legend(fontsize=6, loc="upper right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "xc_vs_v3")


def pupil():
    """(a) 3-class accuracy by eye input, two cultures -> third: linear model and SCVAE-XC.
    (b) film identification within a culture by feature group.
    Sources: results/tables/eye_confound_summary.md, scvae_xc_stage2_summary.md, scvae_xc_d1_summary.md."""
    groups = ["EEG only", "EEG +\npupil", "EEG +\nmovement", "EEG +\nall eye"]
    linear = [55.3, 62.4, 55.7, 62.2]
    scvae = [54.9, 64.4, None, 67.3]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.0), gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    x = np.arange(4)
    for j, (name, vals, c) in enumerate((("linear model", linear, GREY), ("SCVAE-XC", scvae, BLUE))):
        pos = x + (j - 0.5) * 0.38
        for xi, v in zip(pos, vals):
            if v is None:
                continue
            ax.bar(xi, v, width=0.36, color=c, edgecolor="white", linewidth=1)
            ax.text(xi, v + 0.6, f"{v:.0f}", ha="center", fontsize=6, color=INK)
        ax.bar(0, 0, color=c, label=name)
    ax.set_xticks(x, groups, fontsize=6)
    ax.set_ylim(45, 72)
    ax.set_ylabel("3-class accuracy (%)")
    ax.set_title("(a) recognising emotion", fontsize=7, loc="left")
    ax.legend(fontsize=6, loc="upper left")
    ax = axes[1]
    feats, ident = ["EEG", "pupil", "eye\nmovement"], [34.1, 36.9, 41.5]
    ax.bar(np.arange(3), ident, width=0.6, color=AQUA, edgecolor="white", linewidth=1)
    for xi, v in enumerate(ident):
        ax.text(xi, v + 0.8, f"{v:.1f}", ha="center", fontsize=6, color=INK)
    ax.axhline(5.7, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ax.text(2.35, 7, "chance", fontsize=6, color=INK2, ha="right")
    ax.set_xticks(np.arange(3), feats, fontsize=6)
    ax.set_ylim(0, 48)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("(b) naming the film", fontsize=7, loc="left")
    for a in axes:
        a.grid(axis="y", color="#e6e5e0", lw=0.5)
        a.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8)
    save(fig, "pupil")


def main():
    numbers = {}
    pupil()
    xc_vs_v3(numbers)
    benchmark(numbers)
    leakage(numbers)
    film_curve()
    pooling()
    reliability(numbers)
    architecture()
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
