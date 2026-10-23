import csv
import json
import multiprocessing
import os
import threading
import time
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import numpy as np
import torch

from src.environment import CreatureEnv
from src.evolution import Population
from src.model import CreatureBrain
from src.utils import atomic_torch_save, load_universal_best_meta, save_universal_best
from src.s3_sync import pull_models, push_model

# Sync existing checkpoints from S3 on startup if configured
pull_models()

app = Flask(__name__)

# Global Training State
training_lock = threading.Lock()
training_state = {
    "is_training": False,
    "current_generation": 0,
    "total_generations": 100,
    "best_fitness": None,
    "avg_fitness": None,
    "worst_fitness": None,
    "diversity_score": None,
    "population_size": 50,
    "mutation_rate": 0.15,
    "fitness_mode": "baseline",
    "workers": 4,
    "paused": False,
    "error": None,
    "gens_per_min": None,
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
            gen_times = []
            for gen in range(training_state["total_generations"]):
                while training_state["paused"] and training_state["is_training"]:
                    time.sleep(0.5)

                if not training_state["is_training"]:
                    break

                gen_start = time.time()
                training_state["current_generation"] = gen + 1
                pop.mutation_rate = training_state["mutation_rate"]

                # Evaluate population across pool
                adaptive_steps = min(600, 200 + gen * 6)
                pop.evaluate(fitness_mode=training_state["fitness_mode"], max_steps=adaptive_steps, pool=pool)

                gen_best_fitness = max(pop.fitnesses)
                gen_best_idx = pop.fitnesses.index(gen_best_fitness)
                avg_fit = float(np.mean(pop.fitnesses))
                worst_fit = float(np.min(pop.fitnesses))

                with torch.no_grad():
                    param_matrix = torch.stack([
                        torch.cat([p.view(-1) for p in m.parameters()]) for m in pop.models
                    ])
                    diversity = float(torch.std(param_matrix, dim=0).mean().item())

                training_state["best_fitness"] = float(gen_best_fitness)
                training_state["avg_fitness"] = avg_fit
                training_state["worst_fitness"] = worst_fit
                training_state["diversity_score"] = diversity

                gen_elapsed = time.time() - gen_start
                gen_times.append(gen_elapsed)
                if len(gen_times) >= 3:
                    avg_gen_time = sum(gen_times[-5:]) / len(gen_times[-5:])
                    training_state["gens_per_min"] = round(60.0 / avg_gen_time, 1)

                # Save session best
                best_model_path = os.path.join("models", "best_model.pth")
                atomic_torch_save(pop.models[gen_best_idx].state_dict(), best_model_path)
                push_model(best_model_path, "models/best_model.pth")

                # Save top 5
                sorted_indices = np.argsort(pop.fitnesses)[::-1]
                for idx_rank in range(min(5, len(pop.models))):
                    top_model_path = os.path.join("models", f"top_{idx_rank}.pth")
                    atomic_torch_save(
                        pop.models[sorted_indices[idx_rank]].state_dict(),
                        top_model_path,
                    )
                    push_model(top_model_path, f"models/top_{idx_rank}.pth")

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
                    push_model(os.path.join("models", "universal_best.pth"), "models/universal_best.pth")
                    push_model(os.path.join("models", "universal_best.json"), "models/universal_best.json")

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
    model_type = request.args.get("model", "best")  # 'best', 'universal', 'top_0', 'random'

    def generate():
        brain = CreatureBrain()
        if model_type == "universal" and os.path.exists(os.path.join("models", "universal_best.pth")):
            brain.load_state_dict(torch.load(os.path.join("models", "universal_best.pth"), weights_only=True))
        elif model_type == "best" and os.path.exists(os.path.join("models", "best_model.pth")):
            brain.load_state_dict(torch.load(os.path.join("models", "best_model.pth"), weights_only=True))
        elif model_type.startswith("top_"):
            top_path = os.path.join("models", f"{model_type}.pth")
            if os.path.exists(top_path):
                brain.load_state_dict(torch.load(top_path, weights_only=True))
        brain.eval()

        env = CreatureEnv(fitness_mode="baseline")
        state = env.reset()
        done = False
        step_idx = 0

        prev_frame = {}

        while True:
            frame_start = time.perf_counter()
            action = brain.get_action(state)
            state, reward, done, fallen = env.step(action)
            step_idx += 1

            frame = env.get_render_state()
            frame["done"] = bool(done)
            frame["fallen"] = bool(fallen)
            frame["reward"] = float(reward)
            frame["actions"] = [round(float(a), 3) for a in action]
            frame["distance_meters"] = round(float(env.torso.position.x - 400.0) * 0.02, 2)

            if not prev_frame:
                payload = {"type": "full"}
                payload.update(frame)
            else:
                delta = {k: v for k, v in frame.items() if v != prev_frame.get(k)}
                delta["type"] = "delta"
                delta["steps"] = frame["steps"]
                payload = delta
            prev_frame = frame.copy()
            if done or step_idx >= 600:
                prev_frame = {}
            yield "data: " + json.dumps(payload) + "\n\n"
            frame_end = time.perf_counter()
            time.sleep(max(0.0, (1.0 / 45.0) - (frame_end - frame_start)))

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
    safe_names = ["universal_best.pth", "best_model.pth", "top_0.pth", "top_1.pth", "top_2.pth", "top_3.pth", "top_4.pth"]
    if filename not in safe_names:
        return jsonify({"error": "Invalid model filename"}), 400
    models_dir = os.path.abspath("models")
    return send_from_directory(models_dir, filename, as_attachment=True)


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

