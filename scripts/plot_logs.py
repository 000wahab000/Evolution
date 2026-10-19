"""
Plot and visualize evolutionary training progress from data/logs/evolution.log and data/logs/history.csv.

Usage:
    python scripts/plot_logs.py                # Plot latest training run and save PNG + HTML
    python scripts/plot_logs.py --all-runs     # Compare all historical training sessions
    python scripts/plot_logs.py --live         # Live monitor active training loop (refreshes every 2s)
    python scripts/plot_logs.py --show         # Open interactive matplotlib window
"""

import argparse
import os
import re
import time
import matplotlib
import numpy as np

# Use non-interactive Agg backend by default unless --show is passed
if "SHOW_GUI" not in os.environ:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log_file(log_path="data/logs/evolution.log"):
    """
    Parses evolution.log into distinct training sessions.
    Returns a list of sessions, each containing lists of:
    generations, best_fitness, avg_fitness, worst_fitness, diversity, timestamps.
    """
    if not os.path.exists(log_path):
        return []

    sessions = []
    current_session = {
        "gens": [],
        "best": [],
        "avg": [],
        "worst": [],
        "div": [],
        "timestamps": [],
    }

    # Matches lines like:
    # 2026-08-31 02:44:49,125 [INFO] Gen  40 | Best:  596.50 | Avg:  182.27 | Worst: -257.45 | Div: 0.2124
    pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+).*?Gen\s+(\d+)\s+\|\s+Best:\s+([-\d\.]+)\s+\|\s+Avg:\s+([-\d\.]+)\s+\|\s+Worst:\s+([-\d\.]+)\s+\|\s+Div:\s+([-\d\.]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ts, gen, best, avg, worst, div = match.groups()
                gen = int(gen)
                best = float(best)
                avg = float(avg)
                worst = float(worst)
                div = float(div)

                # If generation resets to 0 (or lower than previous), it's a new training run
                if current_session["gens"] and gen <= current_session["gens"][-1]:
                    sessions.append(current_session)
                    current_session = {
                        "gens": [],
                        "best": [],
                        "avg": [],
                        "worst": [],
                        "div": [],
                        "timestamps": [],
                    }

                current_session["gens"].append(gen)
                current_session["best"].append(best)
                current_session["avg"].append(avg)
                current_session["worst"].append(worst)
                current_session["div"].append(div)
                current_session["timestamps"].append(ts)

    if current_session["gens"]:
        sessions.append(current_session)

    return sessions


def parse_history_csv(csv_path="data/logs/history.csv"):
    """Parses current session from history.csv."""
    if not os.path.exists(csv_path):
        return None

    import csv

    gens, best, avg, worst, div = [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gens.append(int(row["Generation"]))
                best.append(float(row["Best_Fitness"]))
                avg.append(float(row["Average_Fitness"]))
                worst.append(float(row["Worst_Fitness"]))
                div.append(float(row["Diversity_Score"]))
            except (KeyError, ValueError):
                continue

    if not gens:
        return None

    return {
        "gens": gens,
        "best": best,
        "avg": avg,
        "worst": worst,
        "div": div,
        "timestamps": [],
    }


def plot_single_run(session, output_png="data/logs/evolution_graph.png", show=False):
    """Generates a high-quality dual-pane plot for a training run."""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    gens = np.array(session["gens"])
    best = np.array(session["best"])
    avg = np.array(session["avg"])
    worst = np.array(session["worst"])
    div = np.array(session["div"])

    if len(gens) == 0:
        print("No generation data to plot.")
        return

    # Style configuration
    plt.style.use("fast")
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8", labelsize=10)
        ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
        for spine in ax.spines.values():
            spine.set_color("#334155")

    # Pane 1: Fitness Over Time
    ax1.plot(gens, best, label="Best Fitness", color="#38bdf8", linewidth=2.5, zorder=4)
    ax1.plot(gens, avg, label="Average Fitness", color="#4ade80", linewidth=2.0, zorder=3)
    ax1.plot(gens, worst, label="Worst Fitness", color="#f87171", linewidth=1.2, alpha=0.7, zorder=2)

    # Shaded confidence bands
    ax1.fill_between(gens, avg, best, color="#38bdf8", alpha=0.15, label="Top Population Spread")
    ax1.fill_between(gens, worst, avg, color="#f87171", alpha=0.08)

    # Highlight Peak Fitness
    peak_idx = int(np.argmax(best))
    peak_gen = gens[peak_idx]
    peak_val = best[peak_idx]
    ax1.scatter([peak_gen], [peak_val], color="#facc15", s=100, zorder=5, edgecolors="#ffffff", linewidths=1.5)
    ax1.annotate(
        f"All-Time Peak: {peak_val:.2f}\n(Gen {peak_gen})",
        xy=(peak_gen, peak_val),
        xytext=(peak_gen + (max(gens) * 0.04), peak_val * 0.92),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#1e293b", edgecolor="#facc15", alpha=0.95),
        color="#facc15",
        fontweight="bold",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="#facc15", lw=1.5),
    )

    ax1.set_ylabel("Fitness Score", color="#f8fafc", fontsize=12, fontweight="bold", labelpad=10)
    ax1.set_title(
        f"Creature Locomotion Evolution — Latest Run ({len(gens)} Generations)",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    legend1 = ax1.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)
    legend1.get_frame().set_alpha(0.85)

    # Pane 2: Genetic Diversity Score
    ax2.plot(gens, div, label="Population Diversity (Std Dev)", color="#c084fc", linewidth=2.0, zorder=3)
    ax2.fill_between(gens, 0, div, color="#c084fc", alpha=0.15)
    ax2.set_ylabel("Diversity", color="#f8fafc", fontsize=11, fontweight="bold", labelpad=10)
    ax2.set_xlabel("Generation", color="#f8fafc", fontsize=12, fontweight="bold", labelpad=8)
    legend2 = ax2.legend(loc="upper right", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=9)
    legend2.get_frame().set_alpha(0.85)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f" Saved evolution plot to: {output_png}")

    if show:
        plt.show()
    plt.close(fig)


def plot_all_runs(sessions, output_png="data/logs/evolution_all_runs.png"):
    """Plots fitness progression curves for all historical training runs."""
    if not sessions:
        print("No sessions found to plot.")
        return

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.style.use("fast")
    fig, ax = plt.subplots(figsize=(13, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.25, color="#64748b")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    colors = plt.cm.viridis(np.linspace(0.3, 0.95, len(sessions)))
    for idx, (session, color) in enumerate(zip(sessions, colors)):
        gens = session["gens"]
        best = session["best"]
        label = f"Run #{idx + 1} ({session['timestamps'][0][:16]} | {len(gens)} gens)"
        ax.plot(gens, best, label=label, color=color, linewidth=2.0, alpha=0.85)

    ax.set_xlabel("Generation", color="#f8fafc", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Best Fitness Score", color="#f8fafc", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Historical Comparison — {len(sessions)} Evolutionary Training Runs",
        color="#f8fafc",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    legend = ax.legend(loc="upper left", facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", fontsize=8)
    legend.get_frame().set_alpha(0.85)

    plt.tight_layout()
    plt.savefig(output_png, dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    print(f" Saved multi-run comparison to: {output_png}")
    plt.close(fig)


def generate_interactive_html(session, output_html="data/logs/dashboard.html"):
    """Creates a standalone interactive Chart.js HTML report."""
    os.makedirs(os.path.dirname(output_html), exist_ok=True)

    gens_json = str(session["gens"])
    best_json = str([round(x, 2) for x in session["best"]])
    avg_json = str([round(x, 2) for x in session["avg"]])
    worst_json = str([round(x, 2) for x in session["worst"]])
    div_json = str([round(x, 4) for x in session["div"]])

    max_fitness = max(session["best"]) if session["best"] else 0
    final_avg = session["avg"][-1] if session["avg"] else 0
    final_div = session["div"][-1] if session["div"] else 0
    total_gens = len(session["gens"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evolutionary Locomotion — Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ background-color: #0b1120; color: #e2e8f0; padding: 24px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; }}
        h1 {{ font-size: 24px; font-weight: 700; color: #38bdf8; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; text-align: center; }}
        .card-label {{ font-size: 12px; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 6px; }}
        .card-val {{ font-size: 28px; font-weight: 700; color: #f8fafc; }}
        .card-val.peak {{ color: #facc15; }}
        .card-val.avg {{ color: #4ade80; }}
        .card-val.div {{ color: #c084fc; }}
        .chart-container {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 20px; margin-bottom: 24px; }}
        .footer {{ text-align: center; color: #64748b; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <h1>Evolution Training Dashboard</h1>
            <p style="color: #94a3b8; font-size: 14px; margin-top: 4px;">Dynamic fitness and diversity telemetry</p>
        </div>
        <div style="color: #64748b; font-size: 12px;">Auto-generated by plot_logs.py</div>
    </div>

    <div class="stats-grid">
        <div class="card">
            <div class="card-label">Generations Evaluated</div>
            <div class="card-val">{total_gens}</div>
        </div>
        <div class="card">
            <div class="card-label">All-Time Peak Fitness</div>
            <div class="card-val peak">{max_fitness:.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Latest Average Fitness</div>
            <div class="card-val avg">{final_avg:.2f}</div>
        </div>
        <div class="card">
            <div class="card-label">Current Population Diversity</div>
            <div class="card-val div">{final_div:.4f}</div>
        </div>
    </div>

    <div class="chart-container">
        <h2 style="font-size: 16px; margin-bottom: 16px; color: #f8fafc;">Fitness Over Generations</h2>
        <canvas id="fitnessChart" height="90"></canvas>
    </div>

    <div class="chart-container">
        <h2 style="font-size: 16px; margin-bottom: 16px; color: #f8fafc;">Genetic Diversity Score</h2>
        <canvas id="diversityChart" height="40"></canvas>
    </div>

    <div class="footer">Evolved Quadruped Locomotion Engine</div>

    <script>
        const gens = {gens_json};
        const best = {best_json};
        const avg = {avg_json};
        const worst = {worst_json};
        const div = {div_json};

        new Chart(document.getElementById('fitnessChart'), {{
            type: 'line',
            data: {{
                labels: gens,
                datasets: [
                    {{
                        label: 'Best Fitness',
                        data: best,
                        borderColor: '#38bdf8',
                        backgroundColor: 'rgba(56, 189, 248, 0.1)',
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.2
                    }},
                    {{
                        label: 'Average Fitness',
                        data: avg,
                        borderColor: '#4ade80',
                        backgroundColor: 'transparent',
                        borderWidth: 2,
                        tension: 0.2
                    }},
                    {{
                        label: 'Worst Fitness',
                        data: worst,
                        borderColor: '#f87171',
                        backgroundColor: 'transparent',
                        borderWidth: 1,
                        borderDash: [4, 4],
                        tension: 0.2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#e2e8f0' }} }}
                }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('diversityChart'), {{
            type: 'line',
            data: {{
                labels: gens,
                datasets: [
                    {{
                        label: 'Diversity Score',
                        data: div,
                        borderColor: '#c084fc',
                        backgroundColor: 'rgba(192, 132, 252, 0.15)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2
                    }}
                ]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{ labels: {{ color: '#e2e8f0' }} }}
                }},
                scales: {{
                    x: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }},
                    y: {{ grid: {{ color: '#334155' }}, ticks: {{ color: '#94a3b8' }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f" Generated interactive HTML dashboard: {output_html}")


def run_live_monitoring(interval=2):
    """Continuously monitors evolution.log / history.csv and updates plots live."""
    print(f" Starting live telemetry monitor (updating every {interval}s). Press Ctrl+C to stop.")
    last_len = 0
    try:
        while True:
            sessions = parse_log_file("data/logs/evolution.log")
            if not sessions:
                session = parse_history_csv("data/logs/history.csv")
            else:
                session = sessions[-1]

            if session and len(session["gens"]) != last_len:
                last_len = len(session["gens"])
                plot_single_run(session, "data/logs/evolution_graph.png")
                generate_interactive_html(session, "data/logs/dashboard.html")
                print(f"-> Updated telemetry: Gen {session['gens'][-1]} | Best: {session['best'][-1]:.2f}")

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nLive monitor stopped.")


def main():
    parser = argparse.ArgumentParser(description="Evolutionary Training Progress Visualizer")
    parser.add_argument("--log", default="data/logs/evolution.log", help="Path to evolution.log")
    parser.add_argument("--csv", default="data/logs/history.csv", help="Path to history.csv")
    parser.add_argument("--all-runs", action="store_true", help="Plot comparison of all historical runs")
    parser.add_argument("--live", action="store_true", help="Continuously refresh graphs while training")
    parser.add_argument("--show", action="store_true", help="Display interactive window")
    parser.add_argument("--out", default="data/logs/evolution_graph.png", help="Output PNG path")
    args = parser.parse_args()

    if args.live:
        run_live_monitoring()
        return

    # Parse data
    sessions = parse_log_file(args.log)
    latest_session = sessions[-1] if sessions else parse_history_csv(args.csv)

    if not latest_session:
        print("No log or history data found in data/logs/.")
        return

    if args.all_runs and len(sessions) > 1:
        plot_all_runs(sessions, "data/logs/evolution_all_runs.png")

    plot_single_run(latest_session, output_png=args.out, show=args.show)
    generate_interactive_html(latest_session, "data/logs/dashboard.html")


if __name__ == "__main__":
    main()
