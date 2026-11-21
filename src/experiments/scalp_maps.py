"""Scalp maps for the IEEE course report: where on the head positive and negative films differ, whether that
pattern is shared across cultures, and which bands / electrodes SCVAE-XC relies on. Uses saved data and the three
exported SCVAE-XC demo models only (no training).

    .venv/bin/python -m src.experiments.scalp_maps
    -> report/figures/scalp_effect.{pdf,png}, scalp_occlusion.{pdf,png}, scalp_numbers.json

Features: 310 DE features = 62 channels x 5 bands (channel-major), z-scored across the 310 features within each
4-s window by the dataset conversion, then standardised per recording (load_culture). Effect sizes are therefore
*relative* power (a channel-band compared with the window's own average), within-recording.
"""

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RBFInterpolator
from scipy.stats import false_discovery_control, pearsonr, ttest_1samp, wilcoxon

from src.baselines.norm_fusion import load_culture
from src.experiments.report_figures import COL, INK, INK2, NAMES, save

LOCS = Path(__file__).with_name("channel_62_pos.locs")
DEMO = Path(os.environ.get("SCVAE_XC_DEMO", "demo"))
DEMO_DIRS = {"SEED_CHN_MM": "chinese", "SEED_GER": "german", "SEED_FRA": "french"}
CULTURES = ("SEED_CHN_MM", "SEED_GER", "SEED_FRA")
BANDS = ("delta", "theta", "alpha", "beta", "gamma")
NEG, POS = 0, 2
LEFT_TEMPORAL, RIGHT_TEMPORAL = ("FT7", "T7", "TP7"), ("FT8", "T8", "TP8")
RNG = np.random.default_rng(0)
torch.set_num_threads(1)


def electrodes():
    rows = [line.split() for line in LOCS.read_text().splitlines() if line.strip()]
    theta = np.deg2rad([float(r[1]) for r in rows])
    radius = np.array([float(r[2]) for r in rows])
    names = [r[3] for r in rows]
    return names, np.c_[radius * np.sin(theta), radius * np.cos(theta)]


NAMES_CH, POS_XY = electrodes()
HEAD_R = 0.56


def topomap(ax, values, vmax, mark=None, cmap="RdBu_r"):
    grid = np.linspace(-HEAD_R, HEAD_R, 120)
    gx, gy = np.meshgrid(grid, grid)
    pts = np.c_[gx.ravel(), gy.ravel()]
    z = RBFInterpolator(POS_XY, values, kernel="thin_plate_spline", smoothing=1e-3)(pts).reshape(gx.shape)
    z[np.hypot(gx, gy) > HEAD_R] = np.nan
    norm = TwoSlopeNorm(0, -vmax, vmax) if cmap == "RdBu_r" else plt.Normalize(0, vmax)
    im = ax.contourf(gx, gy, z, levels=21, cmap=cmap, norm=norm, extend="both")
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(HEAD_R * np.cos(t), HEAD_R * np.sin(t), color=INK2, lw=0.6)
    ax.plot([-0.06, 0, 0.06], [HEAD_R - 0.005, HEAD_R + 0.07, HEAD_R - 0.005], color=INK2, lw=0.6)
    ax.scatter(*POS_XY.T, s=0.6, color=INK2, lw=0)
    if mark is not None and mark.any():
        ax.scatter(*POS_XY[mark].T, s=5, facecolor="white", edgecolor=INK, lw=0.4, zorder=3)
    ax.set_xlim(-0.65, 0.65), ax.set_ylim(-0.62, 0.68), ax.set_aspect("equal"), ax.axis("off")
    return im


def person_effects(culture):
    """Cohen's d (positive - negative) for each of the 310 features, per recording, averaged per person."""
    d = load_culture(culture)
    x = d.x[:, :310]
    per_person = {}
    for rec in np.unique(d.recordings):
        m = d.recordings == rec
        pos, neg = x[m & (d.y == POS)], x[m & (d.y == NEG)]
        pooled = np.sqrt((pos.var(0, ddof=1) * (len(pos) - 1) + neg.var(0, ddof=1) * (len(neg) - 1))
                         / (len(pos) + len(neg) - 2))
        per_person.setdefault(d.subjects[m][0], []).append((pos.mean(0) - neg.mean(0)) / (pooled + 1e-6))
    return np.array([np.mean(v, 0) for v in per_person.values()])  # people x 310


def to_grid(v):
    return v.reshape(62, 5)  # channel x band


def effect_figure(effects, numbers):
    allp = np.concatenate([effects[c] for c in CULTURES])
    t = ttest_1samp(allp, 0)
    q = false_discovery_control(t.pvalue)
    sig = to_grid(q < 0.05)
    rows = [(NAMES[c], to_grid(effects[c].mean(0)), None) for c in CULTURES]
    rows.append((f"All ({len(allp)})", to_grid(allp.mean(0)), sig))
    vmax = float(np.percentile(np.abs(np.concatenate([r[1].ravel() for r in rows])), 98))
    fig, axes = plt.subplots(4, 5, figsize=(2 * COL * 0.72, 4 * 0.98))
    for i, (label, g, mark) in enumerate(rows):
        for b, band in enumerate(BANDS):
            im = topomap(axes[i, b], g[:, b], vmax, None if mark is None else mark[:, b])
            if i == 0:
                axes[i, b].set_title(band, fontsize=8, color=INK)
        axes[i, 0].text(-0.78, 0.03, label, rotation=90, ha="center", va="center", fontsize=8, color=INK)
    fig.subplots_adjust(wspace=0.02, hspace=0.04, right=0.88)
    cb = fig.colorbar(im, cax=fig.add_axes([0.9, 0.3, 0.012, 0.4]), ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels([f"{-vmax:.2f}", "0", f"{vmax:.2f}"])
    cb.set_label("Cohen's d, positive − negative", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    save(fig, "scalp_effect")
    numbers["n_people"] = {NAMES[c]: int(len(effects[c])) for c in CULTURES}
    numbers["fdr_significant_features"] = {band: int(sig[:, b].sum()) for b, band in enumerate(BANDS)}
    numbers["vmax_d"] = vmax
    order = np.argsort(-np.abs(allp.mean(0)))[:10]
    numbers["top_features"] = [{"channel": NAMES_CH[i // 5], "band": BANDS[i % 5], "d": float(allp.mean(0)[i]),
                                "q": float(q[i])} for i in order]
    numbers["band_mean_d"] = {NAMES[c]: {band: float(to_grid(effects[c].mean(0))[:, b].mean())
                                         for b, band in enumerate(BANDS)} for c in CULTURES}


def lateral_temporal(effects, numbers):
    """Mean d over lateral temporal electrodes, beta + gamma (the SEED 'stable pattern' region)."""
    idx = [NAMES_CH.index(n) for n in LEFT_TEMPORAL + RIGHT_TEMPORAL]
    out = {}
    for c in CULTURES + ("all",):
        e = np.concatenate([effects[k] for k in CULTURES]) if c == "all" else effects[c]
        v = np.stack([to_grid(p)[idx][:, 3:5].mean() for p in e])
        out["All" if c == "all" else NAMES[c]] = {"mean_d": float(v.mean()), "positive_people": int((v > 0).sum()),
                                                  "n": len(v), "p_wilcoxon": float(wilcoxon(v).pvalue)}
    numbers["lateral_temporal_beta_gamma"] = out


def consistency(effects, numbers, splits=500):
    """Similarity of the 310-feature effect pattern between cultures, and within a culture (random halves)."""
    means = {c: effects[c].mean(0) for c in CULTURES}
    between = {f"{NAMES[a]}-{NAMES[b]}": float(pearsonr(means[a], means[b])[0])
               for i, a in enumerate(CULTURES) for b in CULTURES[i + 1:]}
    within = {}
    for c in CULTURES:
        e, rs = effects[c], []
        for _ in range(splits):
            p = RNG.permutation(len(e))
            h = len(e) // 2
            rs.append(pearsonr(e[p[:h]].mean(0), e[p[h:]].mean(0))[0])
        within[NAMES[c]] = float(np.mean(rs))
    # between cultures with half the people per culture, so it is comparable with the split-half numbers
    between_half = {}
    for i, a in enumerate(CULTURES):
        for b in CULTURES[i + 1:]:
            rs = [pearsonr(effects[a][RNG.permutation(len(effects[a]))[:len(effects[a]) // 2]].mean(0),
                           effects[b][RNG.permutation(len(effects[b]))[:len(effects[b]) // 2]].mean(0))[0]
                  for _ in range(splits)]
            between_half[f"{NAMES[a]}-{NAMES[b]}"] = float(np.mean(rs))
    numbers["pattern_correlation"] = {"between_full": between, "between_half": between_half,
                                      "within_split_half": within}


def load_model(folder):
    import sys
    sys.path.insert(0, str(DEMO))
    from scvae_xc_model import load
    return load(DEMO / "models" / folder / "weights.pt")


def accuracy(model, x, y):
    with torch.no_grad():
        return float((model(torch.from_numpy(x)).argmax(1).numpy() == y).mean() * 100)


def occlusion(numbers, random_sets=200):
    """Accuracy drop of each exported SCVAE-XC model on its own (unseen) test culture when features are set to the
    recording mean (0 after per-recording standardisation): one band, the eyes, single electrodes, and the 6 lateral
    temporal electrodes compared with 6 random electrodes."""
    temporal = [NAMES_CH.index(n) for n in LEFT_TEMPORAL + RIGHT_TEMPORAL]
    bands, groups, single, region = {}, {}, {}, {}
    for c in CULTURES:
        z = np.load(DEMO / "models" / DEMO_DIRS[c] / "data.npz")
        x, y = z["x"].astype(np.float32), z["label"]
        model = load_model(DEMO_DIRS[c])
        base = accuracy(model, x, y)

        def drop(cols):
            xo = x.copy()
            xo[:, cols] = 0
            return base - accuracy(model, xo, y)

        def chans(ch, bs=range(5)):
            return [i * 5 + b for i in ch for b in bs]

        name = NAMES[c]
        bands[name] = {band: drop(chans(range(62), [b])) for b, band in enumerate(BANDS)}
        groups[name] = {"base_acc": base, "all_eeg": drop(list(range(310))), "eye": drop(list(range(310, 343)))}
        single[name] = float(np.abs([drop(chans([ch])) for ch in range(62)]).max())
        rand = np.array([drop(chans(RNG.choice(62, 6, replace=False))) for _ in range(random_sets)])
        t = drop(chans(temporal))
        region[name] = {"temporal": t, "random_mean": float(rand.mean()),
                        "random_lo": float(np.percentile(rand, 2.5)), "random_hi": float(np.percentile(rand, 97.5)),
                        "rank_p": float((1 + (rand >= t).sum()) / (1 + len(rand)))}
    numbers["occlusion"] = {"bands": bands, "groups": groups, "max_single_electrode_abs": single, "temporal": region}

    colors = {"Chinese": "#2a78d6", "German": "#eb6834", "French": "#1baf7a"}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(2 * COL * 0.72, 1.7), gridspec_kw={"width_ratios": [1.6, 1]})
    w = 0.26
    for k, name in enumerate(colors):
        vals = [bands[name][b] for b in BANDS] + [groups[name]["eye"]]
        ax.bar(np.arange(6) + (k - 1) * w, vals, w, color=colors[name], label=name)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xticks(range(6), list(BANDS) + ["eye"])
    ax.set_ylabel("accuracy drop (points)")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) remove one band, or the eyes", fontsize=8, loc="left")
    for k, name in enumerate(colors):
        r = region[name]
        bx.bar(k - 0.18, r["temporal"], 0.34, color=colors[name])
        bx.bar(k + 0.18, r["random_mean"], 0.34, color="#c9c8c3",
               yerr=[[r["random_mean"] - r["random_lo"]], [r["random_hi"] - r["random_mean"]]],
               error_kw={"lw": 0.6, "capsize": 1.5, "ecolor": INK2})
    bx.axhline(0, color=INK2, lw=0.6)
    bx.set_xticks(range(3), list(colors))
    bx.set_title("(b) remove 6 electrodes", fontsize=8, loc="left")
    bx.text(0.98, 0.97, "colour: lateral temporal\ngrey: 6 random (95% range)", transform=bx.transAxes,
            ha="right", va="top", fontsize=6.5, color=INK2)
    fig.tight_layout()
    save(fig, "scalp_occlusion")


def main():
    numbers = {}
    effects = {c: person_effects(c) for c in CULTURES}
    effect_figure(effects, numbers)
    lateral_temporal(effects, numbers)
    consistency(effects, numbers)
    occlusion(numbers)
    Path("report/figures/scalp_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
