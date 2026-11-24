"""End-to-end PPO run folder: the six artifacts the analyzer reads (needs the `rl` extra)."""

import pytest

pytest.importorskip("stable_baselines3")


def test_run_one_ppo_writes_the_ga_run_folder(tmp_path):
    from src.research.config import RunConfig
    from src.rl.train import run_one_ppo

    cfg = RunConfig(
        experiment="t", variant="ppo", seed=0, optimizer="ppo",
        episodes_budget=2, rollouts_k=1, max_steps=8, eval_every_episodes=1,
        optimizer_params={"n_envs": 1, "n_steps": 8, "batch_size": 4, "n_epochs": 1},
        test_terrain_seeds=[1000], control_terrain_seeds=[2000],
        validation_terrain_seeds=[3000], train_eval_terrain_seeds=[4000],
    )
    assert run_one_ppo(cfg, str(tmp_path), log=lambda *_: None) == "done"

    d = tmp_path / "t" / "ppo" / "seed_0"
    for name in ("config.json", "meta.json", "history.csv", "evals.csv",
                 "champion.npz", "final_test.json", "DONE"):
        assert (d / name).exists(), name
    assert run_one_ppo(cfg, str(tmp_path), log=lambda *_: None) == "skipped"
