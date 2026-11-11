"""Watch the best genome from ``ga_train.py`` walk around in MuJoCo.

    python mujoco/evolution/scripts/play_best.py
    python mujoco/evolution/scripts/play_best.py --episodes 3 --speed 1.0
    python mujoco/evolution/scripts/play_best.py --headless

``--headless`` runs the same rollout without opening a window, which is handy
for checking that the saved brain still scores what training reported.

The camera follows the torso and the run loops: after every episode (fall or
step limit) the dog is put back at the start pose, so you can leave the window
open and just watch.  Press ``Esc`` or close the window to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np

# Same import dance as ga_train.py: append the repo root so the repo's
# ``mujoco/`` folder does not shadow the installed ``mujoco`` package.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mujoco_env import MujocoDogEnv  # noqa: E402

from src.model import CreatureBrain  # noqa: E402

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "best_mujoco.npz"
DEFAULT_XML = Path(__file__).resolve().parent.parent / "models" / "smoke.xml"


def load_brain(path: Path, obs_size: int, act_size: int) -> CreatureBrain:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Train first: python mujoco/evolution/scripts/ga_train.py"
        )
    with np.load(path) as weights:
        state_dict = {key: weights[key] for key in weights.files}

    brain = CreatureBrain(input_size=obs_size, output_size=act_size)
    brain.load_state_dict(state_dict)
    return brain


def build_env(max_steps: int, model_path: Path, xml_path: str | None) -> tuple[MujocoDogEnv, CreatureBrain]:
    env = MujocoDogEnv(max_steps=max_steps, xml_path=xml_path)
    return env, load_brain(model_path, env.obs_size, env.act_size)


def rollout(env: MujocoDogEnv, brain: CreatureBrain) -> tuple[float, float, bool, int]:
    """Run one episode and return (total_reward, distance, fallen, steps)."""
    state = env.reset()
    total_reward = 0.0
    steps = 0
    fallen = False

    for _ in range(env.max_steps):
        state, reward, done, fallen = env.step(brain.get_action(state))
        total_reward += reward
        steps += 1
        if done:
            break

    return total_reward, float(env.data.qpos[0] - env.start_x), fallen, steps


def run_headless(env: MujocoDogEnv, brain: CreatureBrain, episodes: int, quiet: bool) -> None:
    for episode in range(episodes):
        reward, distance, fallen, steps = rollout(env, brain)
        if quiet:
            continue
        status = "fell" if fallen else "stayed up"
        print(
            f"Episode {episode + 1}: reward {reward:8.2f} | distance {distance:6.2f} m "
            f"| {steps:3d} steps | {status}"
        )


def run_viewer(
    env: MujocoDogEnv,
    brain: CreatureBrain,
    episodes: int,
    speed: float,
    verbose: bool,
    show_ui: bool,
) -> None:
    """Real-time viewer loop.  ``episodes=0`` means loop forever."""
    state = env.reset()
    episode = 0
    steps_done = 0
    distance = 0.0
    distances: list[float] = []
    started = time.perf_counter()
    last_report = started

    with mujoco.viewer.launch_passive(
        env.model, env.data, show_left_ui=show_ui, show_right_ui=show_ui
    ) as viewer:
        with viewer.lock():
            viewer.cam.distance = 2.2
            viewer.cam.elevation = -18.0
            viewer.cam.azimuth = 90.0

        while viewer.is_running():
            sim_before = float(env.data.time)
            wall_before = time.perf_counter()

            state, _reward, done, _fallen = env.step(brain.get_action(state))
            steps_done += 1

            # Keep the camera pointed at the torso so the dog never walks away.
            viewer.cam.lookat[:] = env.data.qpos[:3]
            viewer.sync()

            if done:
                episode += 1
                distance = float(env.data.qpos[0] - env.start_x)
                distances.append(distance)
                if verbose:
                    print(
                        f"Episode {episode}: distance {distance:6.2f} m "
                        f"| {steps_done:3d} steps"
                        + (" | fell" if env.data.qpos[2] < 0.1 else "")
                    )
                steps_done = 0
                if episodes and episode >= episodes:
                    break
                state = env.reset()
                continue

            # Throttle so the simulation runs at roughly real time (speed 1.0).
            sim_elapsed = (float(env.data.time) - sim_before) / max(speed, 1e-6)
            remaining = sim_elapsed - (time.perf_counter() - wall_before)
            if remaining > 0:
                time.sleep(min(remaining, 0.02))

            # Heartbeat so a silent console still proves it is running.
            now = time.perf_counter()
            if now - last_report >= 10.0:
                recent = distances[-20:] or [0.0]
                print(
                    f"[{now - started:5.0f}s] episodes {episode:4d} | "
                    f"last distance {distance:5.2f} m | mean(last 20) "
                    f"{np.mean(recent):5.2f} m",
                    flush=True,
                )
                last_report = now

    if distances:
        print(f"\nFinished {episode} episodes | mean distance {np.mean(distances):.2f} m")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="weights file (.npz)")
    parser.add_argument("--xml", default=str(DEFAULT_XML), help="MuJoCo model file")
    parser.add_argument("--episodes", type=int, default=0, help="0 = loop forever")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="steps before an episode resets (2000 = 5 s of sim time)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument(
        "--no-ui", action="store_true", help="hide the Simulate side panels"
    )
    parser.add_argument("--verbose", action="store_true", help="print every episode")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without a window and print the score instead",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    env, brain = build_env(args.max_steps, Path(args.model), args.xml)

    if args.headless:
        run_headless(env, brain, args.episodes or 1, quiet=False)
    else:
        run_viewer(
            env,
            brain,
            args.episodes,
            args.speed,
            args.verbose,
            not args.no_ui,
        )


if __name__ == "__main__":
    main()
