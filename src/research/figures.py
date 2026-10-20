"""F1-F10. One palette, one marker and one dash per variant across every figure.

Each figure is drawn once per theme: F*.png + F*.svg (light, for REPORT.md and the paper) and
F*.dark.svg (dashboard). Variant colours meet 3:1 contrast on white and on #121214.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.research.analysis import FRICTIONS, POSE_SIGMAS, TAU, values  # noqa: E402
from src.research.stats import bootstrap_ci, iqm  # noqa: E402

PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#9467BD", "#B8860B"]  # Okabe-Ito based
MARKERS = ["o", "s", "^", "D", "v", "P"]
DASHES = ["solid", "dashed", "dashdot", "dotted", "solid", "dashed"]
THEMES = {
    "light": {"bg": "#FFFFFF", "fg": "#1A1A1A", "muted": "#5F6368", "grid": "#E3E3E3"},
    "dark": {"bg": "#121214", "fg": "#EDEDEF", "muted": "#82828B", "grid": "#2A2A2E"},
}


def style(variants):
    """Fixed encoding per variant, in experiment order; exported to the dashboard too."""
    return {v: {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)],
                "dash": DASHES[i % len(DASHES)]}
            for i, v in enumerate(variants)}


def _themed(theme):
    t = THEMES[theme]
    return {"figure.facecolor": t["bg"], "axes.facecolor": t["bg"], "savefig.facecolor": t["bg"],
            "text.color": t["fg"], "axes.labelcolor": t["fg"], "axes.edgecolor": t["muted"],
            "xtick.color": t["fg"], "ytick.color": t["fg"], "grid.color": t["grid"],
            "axes.grid": True, "legend.frameon": False, "font.size": 10, "svg.fonttype": "none"}


def save_all(draw, name, out_dir):
    """Call draw(theme) -> fig for both themes and write PNG/SVG (light) and dark SVG."""
    os.makedirs(out_dir, exist_ok=True)
    for theme in THEMES:
        with plt.rc_context(_themed(theme)):
            fig = draw(theme)
            if theme == "light":
                fig.savefig(os.path.join(out_dir, f"{name}.png"), dpi=150, bbox_inches="tight")
                fig.savefig(os.path.join(out_dir, f"{name}.svg"), bbox_inches="tight")
            else:
                fig.savefig(os.path.join(out_dir, f"{name}.dark.svg"), bbox_inches="tight")
            plt.close(fig)


def f1_curves(curves, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.array(curves["episodes"])
        for v, s in sty.items():
            if v not in curves:
                continue
            c = curves[v]
            ax.plot(x, c["iqm"], color=s["color"], linestyle=s["dash"], marker=s["marker"],
                    markevery=max(1, len(x) // 8), label=v)
            ax.fill_between(x, c["lo"], c["hi"], color=s["color"], alpha=0.18, linewidth=0)
        ax.set(title="F1 Learning curves (test terrains)", xlabel="training episodes",
               ylabel="champion test distance (px)")
        ax.legend()
        return fig
    return draw


def f2_final(runs, sty, rng):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, (v, s) in enumerate(sty.items()):
            if v not in runs:
                continue
            for j, (metric, label) in enumerate((("train_distance", "train"), ("test_distance", "test"))):
                x = values(runs[v], metric)
                xpos = i + (j - 0.5) * 0.35
                jitter = np.linspace(-0.08, 0.08, len(x))
                ax.scatter(xpos + jitter, x, s=14, color=s["color"], marker=s["marker"],
                           alpha=0.35 if label == "train" else 0.8, linewidths=0)
                lo, hi = bootstrap_ci(x, rng, n=2000)
                ax.errorbar(xpos, iqm(x), yerr=[[max(0, iqm(x) - lo)], [max(0, hi - iqm(x))]], fmt="_", markersize=18,
                            color=THEMES[theme]["fg"], capsize=4)
        ax.set_xticks(range(len(sty)), [f"{v}\n(train | test)" for v in sty])
        ax.set(title="F2 Final distance per seed (bar = IQM, 95% CI)", ylabel="distance (px)")
        return fig
    return draw


def f3_gap(runs, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        allv = []
        for v, s in sty.items():
            if v not in runs:
                continue
            tr, te, co = (values(runs[v], m) for m in ("train_distance", "test_distance", "control_distance"))
            ax.scatter(tr, te, color=s["color"], marker=s["marker"], s=22, label=f"{v} test")
            ax.scatter(tr, co, facecolors="none", edgecolors=s["color"], marker=s["marker"], s=22,
                       label=f"{v} control")
            allv += [*tr, *te, *co]
        lim = (min(allv + [0]), max(allv + [1]))
        ax.plot(lim, lim, color=THEMES[theme]["muted"], linewidth=1, linestyle="--")
        ax.set(title="F3 Generalization gap (filled = 5-8% test, hollow = ≤4% control)",
               xlabel="train-distribution distance (px)", ylabel="held-out distance (px)")
        ax.legend(fontsize=8, ncol=2)
        return fig
    return draw


def f4_profile(runs, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(7, 4))
        allv = np.concatenate([values(r, "test_distance") for r in runs.values()])
        taus = np.linspace(allv.min(), allv.max(), 200)
        for v, s in sty.items():
            if v in runs:
                x = values(runs[v], "test_distance")
                ax.step(taus, [(x > t).mean() for t in taus], where="post", color=s["color"],
                        linestyle=s["dash"], label=v)
        ax.set(title="F4 Performance profile", xlabel="threshold τ on test distance (px)",
               ylabel="fraction of runs with M1 > τ", ylim=(0, 1.02))
        ax.legend()
        return fig
    return draw


def f5_matrix(rows, sty):
    names = list(sty)
    delta = np.full((len(names), len(names)), np.nan)
    text = [["" for _ in names] for _ in names]
    for r in rows:
        i, j = names.index(r["a"]), names.index(r["b"])
        star = "*" if r["p_holm"] < 0.05 else ""
        delta[i, j], delta[j, i] = r["cliffs_delta"], -r["cliffs_delta"]
        text[i][j] = f"{r['cliffs_delta']:+.2f}{star}\nP={r['p_improvement']:.2f}"
        text[j][i] = f"{-r['cliffs_delta']:+.2f}{star}\nP={1 - r['p_improvement']:.2f}"

    def draw(theme):
        fig, ax = plt.subplots(figsize=(1.6 * len(names) + 2, 1.3 * len(names) + 1.5))
        im = ax.imshow(delta, cmap="PuOr_r", vmin=-1, vmax=1)
        ax.grid(False)
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(j, i, text[i][j] or "—", ha="center", va="center", fontsize=8,
                        color="#111111" if abs(np.nan_to_num(delta[i, j])) < 0.6 else "#FFFFFF")
        ax.set_xticks(range(len(names)), names)
        ax.set_yticks(range(len(names)), names)
        ax.set(title="F5 Row vs column on test distance\n(Cliff's δ, P(row > col); * Holm p < 0.05)")
        fig.colorbar(im, ax=ax, label="Cliff's δ (row beats column > 0)")
        return fig
    return draw


def f6_efficiency(runs, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(7, 4))
        for v, s in sty.items():
            if v not in runs:
                continue
            rs = runs[v]
            hit = np.sort([r["episodes_to_tau"] for r in rs if r["episodes_to_tau"] is not None])
            if len(hit):
                ax.step(np.r_[0, hit], np.arange(len(hit) + 1) / len(rs), where="post",
                        color=s["color"], linestyle=s["dash"],
                        label=f"{v} ({len(rs) - len(hit)} of {len(rs)} never reached)")
            else:
                ax.plot([], [], color=s["color"], label=f"{v} (none reached)")
        ax.set(title=f"F6 Episodes until champion test distance ≥ {TAU:.0f} px (ECDF)",
               xlabel="training episodes", ylabel="fraction of runs", ylim=(0, 1.02))
        ax.legend(fontsize=8)
        return fig
    return draw


def f7_robustness(robust, sty, rng):
    panels = [("friction", FRICTIONS, "friction scale"), ("pose", POSE_SIGMAS, "initial leg-angle noise σ (rad)"),
              ("dt", [1, 0.5], "physics timestep (× dt)")]

    def draw(theme):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
        for ax, (key, xs, label) in zip(axes, panels):
            for v, s in sty.items():
                if v not in robust:
                    continue
                m = robust[v][key]
                ys = iqm(m, axis=0)
                cis = np.array([bootstrap_ci(m[:, i], rng, n=1000) for i in range(m.shape[1])])
                ax.plot(xs, ys, color=s["color"], marker=s["marker"], linestyle=s["dash"], label=v)
                ax.fill_between(xs, cis[:, 0], cis[:, 1], color=s["color"], alpha=0.15, linewidth=0)
            ax.set(xlabel=label)
        axes[0].set(ylabel="test distance (px)")
        axes[0].legend(fontsize=8)
        fig.suptitle("F7 Robustness (IQM, 95% CI); a drop at dt/2 flags a physics exploit")
        return fig
    return draw


def f8_tradeoff(runs, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(6, 4))
        for v, s in sty.items():
            if v in runs:
                pts = [(r["test_distance"], r["cot"]) for r in runs[v] if not np.isnan(r["cot"])]
                if pts:
                    x, y = zip(*pts)
                    ax.scatter(x, y, color=s["color"], marker=s["marker"], s=22, label=v)
        ax.set(title="F8 Efficiency trade-off (lower right = fast and cheap)",
               xlabel="test distance (px)", ylabel="cost of transport (Σ|action| / px)")
        ax.legend(fontsize=8)
        return fig
    return draw


def f9_gait(gaits, sty):
    names = [v for v in sty if v in gaits]

    def draw(theme):
        fig = plt.figure(figsize=(11, 1.4 * len(names) + 1.5))
        grid = fig.add_gridspec(len(names), 2, width_ratios=[2.4, 1])
        for i, v in enumerate(names):
            ax = fig.add_subplot(grid[i, 0])
            tr = gaits[v]["trace"]
            ax.imshow(tr.T, aspect="auto", cmap="coolwarm", vmin=-np.pi / 3, vmax=np.pi / 3,
                      interpolation="nearest")
            ax.grid(False)
            ax.set_yticks(range(4), [f"leg {k}" for k in range(4)], fontsize=7)
            ax.set_ylabel(f"{v}\nseed {gaits[v]['median_seed']}", fontsize=8, color=sty[v]["color"])
            for x in range(0, len(tr), 30):
                ax.axvline(x, color=THEMES[theme]["muted"], linewidth=0.5)
            if i == len(names) - 1:
                ax.set_xlabel("step (vertical lines = one CPG period)")
        ax = fig.add_subplot(grid[:, 1])
        bins = np.linspace(0, 4, 33)
        for v in names:
            if gaits[v]["ratios"]:
                ax.hist(gaits[v]["ratios"], bins=bins, histtype="step", color=sty[v]["color"],
                        linestyle=sty[v]["dash"], linewidth=1.5, label=v)
        ax.axvline(1.0, color=THEMES[theme]["muted"], linestyle="--", linewidth=1)
        ax.set(xlabel="leg-0 dominant frequency ÷ CPG frequency", ylabel="runs")
        ax.legend(fontsize=8)
        fig.suptitle("F9 Gait: median-seed champion leg angles, and frequency lock to the CPG clock")
        return fig
    return draw


def f10_falls(summary, sty):
    def draw(theme):
        fig, ax = plt.subplots(figsize=(6, 3.8))
        names = [v for v in sty if v in summary]
        for i, v in enumerate(names):
            f = summary[v]["fall_pooled"]
            lo, hi = f["wilson"]
            ax.bar(i, f["rate"], color=sty[v]["color"], hatch=["", "//", "..", "xx", "\\\\", "--"][i % 6],
                   edgecolor=THEMES[theme]["bg"])
            ax.errorbar(i, f["rate"], yerr=[[max(0, f["rate"] - lo)], [max(0, hi - f["rate"])]], color=THEMES[theme]["fg"],
                        capsize=5)
        ax.set_xticks(range(len(names)), names)
        ax.set(title="F10 Fall rate on test terrains (95% Wilson CI)", ylabel="fraction of episodes",
               ylim=(0, 1))
        return fig
    return draw
