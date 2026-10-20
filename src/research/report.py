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


def _signed(x, nd=1):
    return "—" if x is None or np.isnan(x) else f"{x:+.{nd}f}"


def captions(summary, confirm, gaits, replayed):
    best = max(summary, key=lambda v: summary[v]["test_distance"]["iqm"])
    gaps = ", ".join(f"{v} {_fmt(s['gap_test']['iqm'])} px" for v, s in summary.items() if s["gap_test"])
    flagged = [f"{v} (seeds {', '.join(map(str, s['dt2_flagged']['seeds']))})"
               for v, s in summary.items() if s["exploit_flag"]]
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
               + (f"Seeds losing > {EXPLOIT_DROP:.0%} at dt/2: {'; '.join(flagged)}." if flagged
                  else "No seed flagged."))
        if replayed else "Skipped (analyze.py --no-replay).",
        "F8": "Lower right is fast and cheap.",
        "F9": (f"A leg frequency near 1× the CPG frequency means the gait locks to the clock. Locked share: "
               f"{locked or 'n/a'}.") if replayed else "Skipped (analyze.py --no-replay).",
        "F10": "Lower is better. Bars: mean per-seed fall rate; CI resamples seeds (episodes within a seed "
               "are not independent).",
    }


def build(exp, summary, rows, confirm, caps, repro, problems=()):
    lines = [f"# {exp.name}: results", ""]
    if exp.hypotheses_tag:
        lines += [f"Pre-registration tag: `{exp.hypotheses_tag}` (experiments/HYPOTHESES.md)", ""]
    if problems:
        lines += [f"> **⚠ Incomplete data ({len(problems)} problem(s)); results below cover finished runs "
                  "only.**", ">"] + [f"> - {p}" for p in problems[:20]] + [""]
    lines += ["## Summary (test terrains unless noted; IQM [95% bootstrap CI] across seeds)", "",
              "| Variant | Seeds | M1 test distance (px) | M2 fall rate [95% CI, by seed] | M3 cost of transport | "
              "M4 gap train−test / train−control (px) | M5 median episodes to τ (reached) | "
              "dt/2 drop (IQM) | seeds flagged at dt/2 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for v, s in summary.items():
        f, e, fl = s["fall"], s["episodes_to_tau"], s["dt2_flagged"]
        drop = "—" if s["dt2_drop"] is None else f"{s['dt2_drop']:+.0%}"
        lines.append(
            f"| {v} | {s['n']} | {_iqm_ci(s['test_distance'])} | {_fmt(f['mean'], 2)} [{_fmt(f['ci'][0], 2)}, "
            f"{_fmt(f['ci'][1], 2)}] | {_iqm_ci(s['cot'], 2)} | {_fmt(s['gap_test'] and s['gap_test']['iqm'])} / "
            f"{_fmt(s['gap_control'] and s['gap_control']['iqm'])} | {_fmt(e['median'], 0)} ({e['reached']}/{e['of']}) "
            f"| {drop} | {len(fl['seeds'])}/{fl['of']}{' ⚠' if fl['seeds'] else ''} |")
    flagged = [f"{v} seeds {', '.join(map(str, s['dt2_flagged']['seeds']))}"
               for v, s in summary.items() if s["exploit_flag"]]
    if flagged:
        lines += ["", f"**⚠ Possible physics exploit:** {'; '.join(flagged)} lose more than "
                  f"{EXPLOIT_DROP:.0%} of test distance at dt/2."]

    lines += ["", "## Hypotheses (confirmatory, Holm within this experiment)", "",
              "| Id | Contrast | Metric | Seeds (A/B) | IQM diff [95% CI] | p (Holm) | Cliff's δ | Verdict |",
              "|---|---|---|---|---|---|---|---|"]
    for r in confirm:
        sign = ">" if r["direction"] == "greater" else "<"
        lines.append(f"| {r['id']} | {r['a']} {sign} {r['b']} | {r['metric']} | {r['n_a']}/{r['n_b']} | "
                     f"{_signed(r['iqm_diff'])} [{_signed(r['diff_ci_lo'])}, {_signed(r['diff_ci_hi'])}] | "
                     f"{r['p_holm']:.3g} | {_signed(r['cliffs_delta'], 2)} | **{r['verdict']}** |")
    lines += ["", "Supported = Holm-adjusted p < 0.05 and the IQM-difference CI excludes 0 in the predicted "
              "direction. A hypothesis without data stays in the Holm family with p = 1.", "",
              "## All pairs on test distance (exploratory, Holm over all pairs)", "",
              "| A | B | IQM diff [95% CI] | p | p (Holm) | Cliff's δ | P(A > B) |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['a']} | {r['b']} | {_signed(r['iqm_diff'])} [{_signed(r['diff_ci_lo'])}, "
                     f"{_signed(r['diff_ci_hi'])}] | {r['p']:.3g} | {r['p_holm']:.3g} | "
                     f"{_signed(r['cliffs_delta'], 2)} | {_fmt(r['p_improvement'], 2)} |")

    lines += ["", "## Figures", ""]
    for f in FIGURES:
        lines += [f"### {f}", "", f"![{f}](figures/{f}.png)", "", caps[f], ""]

    lines += ["## Reproducibility", "", "```"]
    lines += [f"{k}: {v}" for k, v in repro.items()]
    lines += ["```", ""]
    return "\n".join(lines)
