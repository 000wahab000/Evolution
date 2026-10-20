"""REPORT.md: summary table, hypothesis verdicts, pairwise stats, F1-F10 and reproducibility."""

import numpy as np

from src.research.analysis import EXPLOIT_DROP, TAU

FIGURES = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10"]


def _fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _iqm_ci(s, nd=1):
    if not s:
        return "—"
    return f"{_fmt(s['iqm'], nd)} [{_fmt(s['ci'][0], nd)}, {_fmt(s['ci'][1], nd)}]"


def captions(summary, confirm, gaits, replayed):
    best = max(summary, key=lambda v: summary[v]["test_distance"]["iqm"])
    gaps = ", ".join(f"{v} {_fmt(s['gap_test']['iqm'])} px" for v, s in summary.items() if s["gap_test"])
    flagged = [v for v, s in summary.items() if s["exploit_flag"]]
    locked = ", ".join(f"{v} {g['locked']:.0%}" for v, g in (gaits or {}).items() if g["locked"] is not None)
    return {
        "F1": f"Higher and earlier is better; overlapping bands mean no clear difference. Best final IQM: {best}.",
        "F2": f"One dot per seed (faint = train-distribution, solid = test); {best} has the highest test IQM "
              f"({_fmt(summary[best]['test_distance']['iqm'])} px).",
        "F3": f"Points far below the diagonal overfit. Train − test gap (IQM): {gaps}.",
        "F4": "A curve that sits above another dominates at every threshold.",
        "F5": "Positive δ means the row beats the column; * marks pairs with Holm-adjusted p < 0.05.",
        "F6": f"Left is cheaper; runs that never reached {TAU:.0f} px are counted in the legend.",
        "F7": ("Flatter curves are more robust; a drop at dt/2 flags a physics exploit. "
               + (f"Flagged (> {EXPLOIT_DROP:.0%} drop): {', '.join(flagged)}." if flagged else "No variant flagged."))
        if replayed else "Skipped (analyze.py --no-replay).",
        "F8": "Lower right is fast and cheap.",
        "F9": (f"A leg frequency near 1× the CPG frequency means the gait locks to the clock. Locked share: "
               f"{locked or 'n/a'}.") if replayed else "Skipped (analyze.py --no-replay).",
        "F10": "Lower is better; non-overlapping Wilson bars mean a clear difference.",
    }


def build(exp, summary, rows, confirm, caps, repro):
    lines = [f"# {exp.name}: results", ""]
    if exp.hypotheses_tag:
        lines += [f"Pre-registration tag: `{exp.hypotheses_tag}` (experiments/HYPOTHESES.md)", ""]
    lines += ["## Summary (test terrains unless noted; IQM [95% bootstrap CI] across seeds)", "",
              "| Variant | Seeds | M1 test distance (px) | M2 fall rate [Wilson] | M3 cost of transport | "
              "M4 gap train−test / train−control (px) | M5 median episodes to τ (reached) | dt/2 drop |",
              "|---|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        f, e = s["fall_pooled"], s["episodes_to_tau"]
        drop = "—" if s["dt2_drop"] is None else f"{s['dt2_drop']:+.0%}" + (" ⚠" if s["exploit_flag"] else "")
        lines.append(
            f"| {v} | {s['n']} | {_iqm_ci(s['test_distance'])} | {f['rate']:.2f} [{f['wilson'][0]:.2f}, "
            f"{f['wilson'][1]:.2f}] | {_iqm_ci(s['cot'], 2)} | {_fmt(s['gap_test'] and s['gap_test']['iqm'])} / "
            f"{_fmt(s['gap_control'] and s['gap_control']['iqm'])} | {_fmt(e['median'], 0)} ({e['reached']}/{e['of']}) "
            f"| {drop} |")
    flagged = [v for v, s in summary.items() if s["exploit_flag"]]
    if flagged:
        lines += ["", f"**⚠ Possible physics exploit:** {', '.join(flagged)} (test-distance IQM drops by "
                  f"more than {EXPLOIT_DROP:.0%} at dt/2)."]

    lines += ["", "## Hypotheses (confirmatory, Holm within this experiment)", "",
              "| Id | Contrast | Metric | IQM diff [95% CI] | p (Holm) | Cliff's δ | Verdict |",
              "|---|---|---|---|---|---|---|"]
    for h, r in zip([h for h in exp.hypotheses if h["a"] in summary and h["b"] in summary], confirm):
        sign = ">" if r["direction"] == "greater" else "<"
        lines.append(f"| {r['id']} | {r['a']} {sign} {r['b']} | {r['metric']} | {r['iqm_diff']:+.1f} "
                     f"[{r['diff_ci_lo']:+.1f}, {r['diff_ci_hi']:+.1f}] | {r['p_holm']:.3g} | "
                     f"{r['cliffs_delta']:+.2f} | **{r['verdict']}** |")
    lines += ["", "Supported = Holm-adjusted p < 0.05 and the IQM-difference CI excludes 0 in the predicted "
              "direction.", "", "## All pairs on test distance (exploratory, Holm over all pairs)", "",
              "| A | B | IQM diff [95% CI] | p | p (Holm) | Cliff's δ | P(A > B) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['a']} | {r['b']} | {r['iqm_diff']:+.1f} [{r['diff_ci_lo']:+.1f}, {r['diff_ci_hi']:+.1f}] "
                     f"| {r['p']:.3g} | {r['p_holm']:.3g} | {r['cliffs_delta']:+.2f} | {r['p_improvement']:.2f} |")

    lines += ["", "## Figures", ""]
    for f in FIGURES:
        lines += [f"### {f}", "", f"![{f}](figures/{f}.png)", "", caps[f], ""]

    lines += ["## Reproducibility", "", "```"]
    lines += [f"{k}: {v}" for k, v in repro.items()]
    lines += ["```", ""]
    return "\n".join(lines)
