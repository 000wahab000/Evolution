import csv
import json
import os
from flask import Flask, jsonify, render_template

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/history")
def get_history():
    history_file = os.path.join("logs", "history.csv")
    if not os.path.exists(history_file):
        return jsonify([])

    history = []
    try:
        with open(history_file, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                history.append(
                    {
                        "Generation": int(row["Generation"]) if "Generation" in row and row["Generation"] != "" else 0,
                        "Best_Fitness": float(row["Best_Fitness"]) if "Best_Fitness" in row and row["Best_Fitness"] != "" else 0.0,
                        "Average_Fitness": float(row["Average_Fitness"]) if "Average_Fitness" in row and row["Average_Fitness"] != "" else 0.0,
                        "Worst_Fitness": float(row["Worst_Fitness"]) if "Worst_Fitness" in row and row["Worst_Fitness"] != "" else 0.0,
                        "Diversity_Score": float(row["Diversity_Score"]) if "Diversity_Score" in row and row["Diversity_Score"] != "" else 0.0,
                    }
                )
        return jsonify(history)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/best")
def get_best():
    meta_file = os.path.join("models", "universal_best_meta.json")
    if not os.path.exists(meta_file):
        return jsonify({})

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/random")
def get_random():
    """Run one episode with a fresh, untrained CreatureBrain and return its fitness."""
    try:
        from src.environment import CreatureEnv
        from src.model import CreatureBrain

        model = CreatureBrain()
        env = CreatureEnv(fitness_mode="baseline")
        state = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = model.get_action(state)
            state, reward, done, fallen = env.step(action)
            total_reward += reward

        if fallen:
            total_reward -= 100.0

        return jsonify({"fitness": round(total_reward, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
