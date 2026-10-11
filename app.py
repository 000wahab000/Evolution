import csv
import json
import multiprocessing
import os
import threading
import time
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import numpy as np

from src.environment import CreatureEnv
from src.evolution import Population
from src.model import CreatureBrain
from src.utils import atomic_save, load_state_dict_file, load_universal_best_meta, save_universal_best

app = Flask(__name__)
# static/ holds the recorded replays (scripts/record_replay.py); short cache so a redeploy shows up quickly
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 300

# Global Training State
training_lock = threading.Lock()
training_state = {
    "is_training": False,
    "current_generation": 0,
    "total_generations": 100,
    "best_fitness": None,
    "avg_fitness": None,
    "diversity_score": None,
    "population_size": 50,
    "mutation_rate": 0.15,
    "fitness_mode": "baseline",
    "workers": int(os.environ.get("TRAINING_WORKERS", 4)),
    "training_enabled": os.environ.get("TRAINING_ENABLED", "1") != "0",
    "paused": False,
    "error": None,
}
training_thread = None


def run_training_loop():
    global training_state
    try:
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        history_path = os.path.join("logs", "history.csv")
        if not os.path.exists(history_path):
            with open(history_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["Generation", "Best_Fitness", "Average_Fitness", "Worst_Fitness", "Diversity_Score"]
                )

        pop = Population(
            CreatureBrain,
            pop_size=training_state["population_size"],
            mutation_rate=training_state["mutation_rate"],
        )

        workers = int(training_state.get("workers", 4))
        pool = multiprocessing.Pool(processes=workers) if workers > 1 else None

        try:
            for gen in range(training_state["total_generations"]):
                while training_state["paused"] and training_state["is_training"]:
                    time.sleep(0.5)

                if not training_state["is_training"]:
                    break

                training_state["current_generation"] = gen + 1
                pop.mutation_rate = training_state["mutation_rate"]

                # Evaluate population across pool
                pop.evaluate(fitness_mode=training_state["fitness_mode"], pool=pool)

                gen_best_fitness = max(pop.fitnesses)
                gen_best_idx = pop.fitnesses.index(gen_best_fitness)
                avg_fit = float(np.mean(pop.fitnesses))
                worst_fit = float(np.min(pop.fitnesses))

                param_matrix = np.stack([np.concatenate([p.ravel() for p in m.parameters()]) for m in pop.models])
                diversity = float(param_matrix.std(axis=0, ddof=1).mean())  # sample std keeps Diversity_Score comparable with older logs

                training_state["best_fitness"] = float(gen_best_fitness)
                training_state["avg_fitness"] = avg_fit
                training_state["diversity_score"] = diversity

                # Save session best
                atomic_save(pop.models[gen_best_idx].state_dict(), os.path.join("models", "best_model.npz"))

                # Save top 5
                sorted_indices = np.argsort(pop.fitnesses)[::-1]
                for idx_rank in range(min(5, len(pop.models))):
                    atomic_save(
                        pop.models[sorted_indices[idx_rank]].state_dict(),
                        os.path.join("models", f"top_{idx_rank}.npz"),
                    )

                # Universal record update
                current_univ = float(load_universal_best_meta().get("best_fitness", -float("inf")))
                if gen_best_fitness > current_univ:
                    save_universal_best(
                        model=pop.models[gen_best_idx],
                        fitness=gen_best_fitness,
                        generation=gen + 1,
                        fitness_mode=training_state["fitness_mode"],
                        pop_size=training_state["population_size"],
                        mutation_rate=training_state["mutation_rate"],
                    )

                with open(history_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([gen + 1, round(gen_best_fitness, 2), round(avg_fit, 2), round(worst_fit, 2), round(diversity, 4)])

                pop.evolve()
                time.sleep(0.01)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    except Exception as e:
        training_state["error"] = str(e)
    finally:
        training_state["is_training"] = False
        training_state["paused"] = False


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
    try:
        return jsonify(load_universal_best_meta())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/random")
def get_random():
    if not training_state["training_enabled"]:
        return jsonify({"message": "Live simulation is disabled on this server"}), 403
    try:
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


@app.route("/api/training/status")
def get_training_status():
    return jsonify(training_state)


@app.route("/api/training/start", methods=["POST"])
def start_training():
    global training_thread, training_state
    if not training_state["training_enabled"]:
        return jsonify({"message": "Training is disabled on this server"}), 403
    if training_state["is_training"]:
        return jsonify({"message": "Training already in progress"}), 400

    data = request.get_json() or {}
    training_state["total_generations"] = int(data.get("generations", 50))
    training_state["population_size"] = int(data.get("pop_size", 50))
    training_state["mutation_rate"] = float(data.get("mutation_rate", 0.15))
    training_state["fitness_mode"] = str(data.get("fitness_mode", "baseline"))
    training_state["is_training"] = True
    training_state["paused"] = False
    training_state["error"] = None

    training_thread = threading.Thread(target=run_training_loop, daemon=True)
    training_thread.start()
    return jsonify({"message": "Training started", "state": training_state})


@app.route("/api/training/pause", methods=["POST"])
def pause_training():
    if not training_state["is_training"]:
        return jsonify({"message": "No active training to pause"}), 400
    training_state["paused"] = not training_state["paused"]
    return jsonify({"paused": training_state["paused"]})


@app.route("/api/training/stop", methods=["POST"])
def stop_training():
    training_state["is_training"] = False
    training_state["paused"] = False
    return jsonify({"message": "Training stopped"})


@app.route("/api/stream/sim")
def stream_simulation():
    # Each stream pins a gunicorn thread forever; the public demo plays recorded replays instead
    if not training_state["training_enabled"]:
        return jsonify({"message": "Live simulation is disabled on this server"}), 403
    model_type = request.args.get("model", "best")  # 'best', 'universal', 'top_0', 'random'

    def generate():
        brain = CreatureBrain()
        if model_type == "universal" and os.path.exists(os.path.join("models", "universal_best.npz")):
            brain.load_state_dict(load_state_dict_file(os.path.join("models", "universal_best.npz")))
        elif model_type == "best" and os.path.exists(os.path.join("models", "best_model.npz")):
            brain.load_state_dict(load_state_dict_file(os.path.join("models", "best_model.npz")))
        elif model_type.startswith("top_"):
            top_path = os.path.join("models", f"{model_type}.npz")
            if os.path.exists(top_path):
                brain.load_state_dict(load_state_dict_file(top_path))

        env = CreatureEnv(fitness_mode="baseline")
        state = env.reset()
        done = False
        step_idx = 0

        while True:
            action = brain.get_action(state)
            state, reward, done, fallen = env.step(action)
            step_idx += 1

            frame = env.get_render_state()
            frame["done"] = bool(done)
            frame["fallen"] = bool(fallen)
            frame["reward"] = float(reward)
            frame["actions"] = [round(float(a), 3) for a in action]
            frame["distance_meters"] = round(float(env.torso.position.x - 400.0) * 0.02, 2)

            yield f"data: {json.dumps(frame)}\n\n"
            time.sleep(1.0 / 45.0)

            if done or step_idx >= 600:
                state = env.reset()
                step_idx = 0
                time.sleep(0.4)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/models/download/<filename>")
def download_model(filename):
    """Secure endpoint to download model checkpoints directly from browser."""
    safe_names = ["universal_best.npz", "best_model.npz", "top_0.npz", "top_1.npz", "top_2.npz", "top_3.npz", "top_4.npz"]
    if filename not in safe_names:
        return jsonify({"error": "Invalid model filename"}), 400
    models_dir = os.path.abspath("models")
    return send_from_directory(models_dir, filename, as_attachment=True, max_age=0)  # change during training


@app.route("/api/logs/tail")
def get_log_tail():
    """Returns the last N lines from evolution.log."""
    log_file = os.path.join("logs", "evolution.log")
    if not os.path.exists(log_file):
        return jsonify([])
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return jsonify(lines[-50:])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, threaded=True)

