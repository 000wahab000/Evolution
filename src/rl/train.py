"""PPO baseline that writes the same run folder as the GA, so `scripts/analyze.py` can compare
the two directly (needs `pip install -e .[rl]`).

Same body, same 15->16->4 tanh brain, same reward, same train distribution, same step budget:
`total_timesteps = episodes_budget * max_steps`. The only thing that changes is the learning
rule, which is the point of the experiment.

The champion is picked on validation terrains only (never test), exactly like the GA runner,
and it is stored as CreatureBrain weights so the existing replay/robustness/gait tooling reads
it unchanged.
"""

import os
import shutil
import time
from functools import partial

import numpy as np

from src.research.config import SCHEMA_VERSION, make_rng
from src.research.evaluate import (
    evaluate_genome,
    fixed_scenarios,
    from_flat,
    summarize,
    to_flat,
)
from src.research.runlog import (
    is_done,
    replace_with_retry,
    utc_now,
    write_done,
    write_json,
    write_meta,
    write_text,
)
from src.research.runner import (
    EVAL_COLS,
    FILES,
    HISTORY_COLS,
    StaleRunError,
    _csv,
    run_dir,
    stale_reason,
)
from src.terrain import TERRAIN_VERSION

# Keys accepted in RunConfig.optimizer_params for a PPO variant. Anything else (e.g. a shared
# base config's pop_size) is ignored with a note, never passed to PPO as a surprise kwarg.
PPO_KEYS = ("learning_rate", "n_steps", "batch_size", "n_epochs", "gamma", "gae_lambda",
            "clip_range", "ent_coef", "vf_coef", "max_grad_norm", "policy_kwargs")


def _make_env(cfg, seed, rank):
    """Module-level so SubprocVecEnv can pickle it under spawn (Windows)."""
    import torch

    from src.rl.envs import ScenarioGymEnv

    torch.set_num_threads(1)  # the physics is CPU bound; extra BLAS threads only fight it
    return ScenarioGymEnv(cfg, make_rng(seed, "ppo_env", rank))


def _evaluate(model, cfg, val, test, control, state, evals, log):
    """Score the current mean policy on validation (champion pick) and held-out sets."""
    from src.rl.export import policy_to_state_dict

    cand = to_flat(policy_to_state_dict(model.policy))
    v = float(evaluate_genome(cand, cfg, val)[0])
    state["eval_eps"] += len(val)
    if v > state["champ_val"]:
        state["champ"], state["champ_val"] = cand.copy(), v
    t = summarize(evaluate_genome(state["champ"], cfg, test)[1])
    c = summarize(evaluate_genome(state["champ"], cfg, control)[1])
    state["eval_eps"] += len(test) + len(control)
    episodes = int(model.num_timesteps // cfg.max_steps)
    evals.append([episodes, t["distance"], t["fall_rate"], t["cot"],
                  c["distance"], c["fall_rate"], c["cot"]])
    log(f"{cfg.run_id}: {episodes} episodes | test {t['distance']:.1f}px "
        f"fall {t['fall_rate']:.2f} | champion {state['champ_val']:.1f}")


def run_one_ppo(cfg, out_root="results", log=print):
    """Train one PPO run to its step budget and write its folder. Returns 'done' or 'skipped'."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        SubprocVecEnv,
        VecMonitor,
        VecNormalize,
    )

    from src.rl.policy import TanhActorCriticPolicy

    d = run_dir(out_root, cfg)
    if is_done(d):
        reason = stale_reason(d, cfg)
        if reason:
            raise StaleRunError(f"{d} holds a finished run from another config ({reason}); "
                                "delete that folder to rerun it")
        return "skipped"
    if os.path.exists(d):
        shutil.rmtree(d)  # leftovers of a crashed run
    os.makedirs(d)
    write_json(os.path.join(d, "config.json"), cfg.to_dict())
    started, t0 = utc_now(), time.time()

    params = dict(cfg.optimizer_params or {})
    n_envs = int(params.pop("n_envs", 1))
    ignored = sorted(set(params) - set(PPO_KEYS))
    hp = {k: params[k] for k in PPO_KEYS if k in params}
    if ignored:
        log(f"{cfg.run_id}: ignoring optimizer_params {ignored} (not PPO hyperparameters)")

    gamma = float(hp.get("gamma", 0.99))
    n_steps = int(hp.get("n_steps", 2048))
    batch_size = int(hp.get("batch_size", 64))  # paper Table 3
    total_steps = int(cfg.episodes_budget) * int(cfg.max_steps)

    envs = [partial(_make_env, cfg, cfg.seed, i) for i in range(n_envs)]
    venv = DummyVecEnv(envs) if n_envs == 1 else SubprocVecEnv(envs, start_method="spawn")
    venv = VecMonitor(venv)
    # norm_obs=False: observations are hand-scaled and the CPG clock is meaningful.
    # norm_reward=True: the raw return is ~3600/episode, which is a bad PPO signal scale.
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)

    val, test, control, train_eval = (fixed_scenarios(cfg, s) for s in
                                      ("validation", "test", "control", "train_eval"))
    history, evals = [], []
    state = {"champ": None, "champ_val": -np.inf, "eval_eps": 0, "diversity": float("nan"),
             "next_eval": int(cfg.eval_every_episodes) * int(cfg.max_steps)}

    class PPOEval(BaseCallback):
        def _on_rollout_end(self):
            buf = self.model.ep_info_buffer
            if not buf:
                return
            returns = [ep["r"] for ep in buf]
            with torch.no_grad():
                state["diversity"] = float(self.model.policy.log_std.exp().mean().item())
            history.append([int(self.num_timesteps // cfg.max_steps), int(self.model._n_updates),
                            float(np.max(returns)), float(np.mean(returns)), state["diversity"]])

        def _on_step(self):
            if self.num_timesteps >= state["next_eval"]:
                _evaluate(self.model, cfg, val, test, control, state, evals, log)
                step = int(cfg.eval_every_episodes) * int(cfg.max_steps)
                state["next_eval"] = (self.num_timesteps // step + 1) * step
            return True

    model = PPO(TanhActorCriticPolicy, venv, seed=cfg.seed, device="cpu", verbose=0,
                n_steps=n_steps, batch_size=batch_size,
                **{k: v for k, v in hp.items() if k not in ("n_steps", "batch_size")})
    try:
        model.learn(total_timesteps=total_steps, callback=PPOEval(), progress_bar=False)
    finally:
        venv.close()

    if state["champ"] is None:  # budget ended before the first eval grid point
        _evaluate(model, cfg, val, test, control, state, evals, log)

    final = {
        "schema_version": SCHEMA_VERSION,
        "test": summarize(evaluate_genome(state["champ"], cfg, test)[1]),
        "control": summarize(evaluate_genome(state["champ"], cfg, control)[1]),
        "train": summarize(evaluate_genome(state["champ"], cfg, train_eval)[1]),
        "test_dt2": summarize(evaluate_genome(state["champ"], cfg, test, substeps=2)[1]),
        # GA "diversity" is population spread; a single PPO policy has no equivalent, so the
        # summary reports it as not applicable. The history.csv column keeps the policy std.
        "diversity": float("nan"),
        "generations": int(getattr(model, "_n_updates", 0)),
        "episodes_used": int(model.num_timesteps // cfg.max_steps),
        "eval_episodes": state["eval_eps"],
        "champion_validation_return": float(state["champ_val"]),
    }
    write_text(os.path.join(d, "history.csv"), _csv(HISTORY_COLS, history))
    write_text(os.path.join(d, "evals.csv"), _csv(EVAL_COLS, evals))
    tmp = os.path.join(d, "champion.tmp.npz")
    np.savez(tmp, **from_flat(state["champ"]))
    replace_with_retry(tmp, os.path.join(d, "champion.npz"))
    write_json(os.path.join(d, "final_test.json"), final)
    write_meta(d, cfg, started, time.time() - t0, final["episodes_used"],
               final["eval_episodes"], TERRAIN_VERSION)
    write_done(d, FILES)
    return "done"
