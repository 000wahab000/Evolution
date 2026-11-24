# PPO baseline (roadmap item 15)

PPO on the same body, the same brain, the same reward and the same step budget as the genetic
algorithm. It exists to test the README's claim that evolution is enough here, not to replace it.
Needs `pip install -e .[rl]` (Stable-Baselines3 + torch). Nothing in the default path imports it.

## Run it

```bash
pip install -e .[rl]
python scripts/run_experiment.py experiments/smoke_ppo.json --workers 1   # ~2 min sanity check
python scripts/run_experiment.py experiments/exp6_ppo.json --workers 5    # 5 seeds, ~4-8 h
python scripts/analyze.py results/exp6_ppo                                # GA vs PPO, same stats
```

`exp6_ppo` has two variants, `ga` and `ppo`, so `REPORT.md` compares them with the usual
IQM/CI/Holm machinery (H12/H13 in `experiments/HYPOTHESES.md`). Each run is 25,000 episodes
= 15M environment steps for both variants. SB3 collects whole 2,048-step rollouts, so a PPO run
ends at 15,001,600 steps (25,002 episodes); the 0.01% overshoot is the only budget difference.

## What had to change

The Gymnasium wrapper already existed, but four things would have quietly broken PPO:

1. **The `-100` fall penalty was outside the env.** `run_episode` subtracts it from the episode
   total; `env.step()` never did. An RL agent would have learned to dive. Fixed with
   `CreatureEnv(fall_penalty=100)`, which returns the penalty on the terminal step. The GA keeps
   its version (`fall_penalty=0`), so GA numbers are unchanged.
2. **`terminated` and `truncated` were the same flag.** The 600-step timeout was reported as a
   real terminal, so PPO zeroed the value bootstrap every episode. Now `env.truncated` marks the
   time limit and the wrapper reports it as truncation.
3. **The wrapper trained on the wrong distribution.** It used the legacy course with no
   randomization. `src/rl/envs.py` now samples episodes with the same `sample_scenario()` the
   research runner uses (random ≤4% courses, friction ×0.9–1.1, pose noise).
4. **Reward scale.** The raw return is ~3600/episode. `VecNormalize(norm_reward=True,
   norm_obs=False)` fixes the training signal without touching the exported policy. Observations
   are *not* normalized: they are hand-scaled and the CPG clock is meaningful.

## Why the brain is held fixed

SB3's `net_arch=[16]` is not one hidden layer: `create_mlp` appends a final `Linear(16, 16)`, and
the action head is linear. So `src/rl/policy.py` builds the extractor itself:

```
policy head: Linear(15, 16) -> Tanh -> Linear(16, 4) -> Tanh
value head:  Linear(15, 16) -> Tanh -> Linear(16, 1)
```

That is exactly `CreatureBrain` (`src/model.py`) plus a value head. Consequences:

- the deterministic action exports into `champion.npz` to float32 precision
  (`src/rl/export.py`, checked by `tests/test_rl_policy.py`);
- F7/F9 robustness and gait replay the PPO champion through the same `rollout()` code;
- the comparison isolates the learning rule, which is the actual question.

`hidden_dim=64` also exports, but its weights are not a `CreatureBrain`, so the research figures
would need a guard. The 16-unit policy is the one `exp6_ppo` uses.

## Things to know

- **`n_envs=1` by default.** `--workers` already runs seeds in parallel. Set `n_envs>1` only when
  running a single seed; it uses `SubprocVecEnv`, which needs the spawn-safe module-level
  `_make_env`.
- **`log_std_init=-1.0`** (initial std ≈ 0.37). Actions are bounded at ±1, so SB3's default std
  of 1.0 would clip roughly half of the early samples.
- **`batch_size=64`, matching Table 3.** `n_steps=2048` with a 64-sample minibatch is 32
  minibatches per rollout, so 10 epochs is 320 gradient steps per rollout (a 256 minibatch gave
  80). 256 is 4x faster but takes 4x fewer gradient steps, so `exp6_ppo` uses the paper value.
  Every other hyperparameter here is the paper's or `ppo2`'s.
- **Constant learning rate.** Table 3 lists 3e-4; the official `ppo2` implementation also anneals
  it linearly to 0. SB3 keeps it constant unless given a schedule.
- **The champion is chosen on validation terrains only**, every 1,250 episodes, exactly like the
  GA runner. Test and control never select.
- **`history.csv` columns are GA-shaped.** `generation` is the PPO update count,
  `best_train`/`mean_train` are episode returns from the last 100 episodes, `diversity` is the
  mean policy std (exploration spread, not population spread). `final_test.json` reports
  `diversity` as NaN, because a single policy has no population to measure.
- **Chaotic replay still applies.** The same trained policy can walk differently on another CPU,
  for the reason described in the main README.
