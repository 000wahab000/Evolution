# Codebase Questions — Evolution-1

Questions about every file and design decision in this repo.
If you're an agent picking this up, answer all of them before touching anything.

---

## `src/environment.py` — Physics & Reward

1. **Fitness mode mismatch**: `CreatureEnv.__init__` defaults to `fitness_mode="baseline"` but `main.py --train` also defaults to `"baseline"`. If someone calls `CreatureEnv()` directly (e.g. in `--render`), they get `"baseline"` even though training used `"all_penalties"`. Should both always default to the same mode, and should that mode be a single constant defined in one place?

2. **Gravity value**: Gravity is `(0, 900)`. PyGame/Pymunk Y-axis points down, so this is correct — but is 900 px/s² calibrated to anything? At 1/60 timestep, per-step gravity impulse is 15 px/s. Is this too strong/too weak for a creature with leg length 40px?

3. **Ground position**: Ground segment at `Y=500`, torso at `Y=450`, leg center at `Y=450+10+20=480`, so foot bottom is at `Y=500`. Does this mean the creature spawns with feet exactly touching ground, or is there a gap due to shape radius?

4. **Leg anchor positions**: Anchors are `(-22, 10), (-18, 10), (18, 10), (22, 10)`. The rear-outer and rear-inner are only 4px apart, and front-inner/outer are also 4px apart. Why not space them more evenly? Does this cause the two "pairs" to collide even with `ShapeFilter(group=1)`?

5. **`torso_shape` / `leg_shapes` attributes removed**: The user edited `environment.py` and removed `self.torso_shape` and `self.leg_shapes` attributes. The render loop in `main.py` previously used `envs[i].torso_shape.color` and `envs[i].leg_shapes` to color creatures. After the user's edit these crash with `AttributeError`. Is there a plan to add per-creature coloring back, or is it intentionally removed?

6. **Tilt penalty scaling**: `reward -= 20 * abs(self.torso.angle) / 60.0`. At max angle before fall (1.2 rad), this is `20 * 1.2 / 60 = 0.4` penalty per step, or `240` total over 600 steps. Forward displacement at a modest 2 px/step is `1200`. Is the tilt penalty strong enough to actually discourage tilting, or is it drowned out by the displacement reward?

7. **Energy penalty scaling**: `reward -= energy_cost * 0.05 / 60.0`. With 4 motors at full action (`|a|=1` each), that's `4 * 0.05 / 60 = 0.003` per step or `1.8` total. This is negligibly small. Does it actually prevent energy abuse?

8. **Standstill detection threshold**: `abs(self.torso.velocity.x) < 2.0` — is 2 px/s the right threshold? The torso can be oscillating in place (moving ±1.9 px/s while legs cycle) and never trigger the standstill counter. Should it track displacement over a window instead of instantaneous velocity?

9. **`still_steps` not initialized in `__init__`**: `self.still_steps` is only initialized in `reset()`, not in `__init__`. If `step()` is called before `reset()`, this crashes with `AttributeError`. Is this intentional (caller must call reset first)?

10. **`get_state()` state vector size**: State has 5 torso values + 4 legs × 2 values + 2 CPG values = 15 total. `CreatureBrain` expects `input_size=15`. If you ever add/remove a leg or state feature, both must change together — is this coupling documented anywhere?

11. **CPG phase period**: `phase = 2.0 * np.pi * (self.steps % 30) / 30.0` — period is 30 steps = 0.5s at 60Hz. Is this matched to any natural gait frequency? What happens if the creature naturally finds a stride at 40 or 20 steps — does the CPG help or hurt it?

12. **Fall condition Y threshold**: `self.torso.position.y > 520` — the ground is at Y=500 and torso starts at Y=450. At Y=520 the torso center is 20px below ground, meaning it has fully sunk. Should this be `> 505` (just below ground surface) to catch falls earlier?

---

## `src/model.py` — Neural Network

13. **Network size**: 15 → 16 → 4 with Tanh activations, 324 total parameters. Is this too small to learn complex gaits? A creature with 4 legs and phase-coupled motion might need at minimum a recurrent layer or a larger hidden layer to memorize stride patterns. Has 16 hidden units been validated as sufficient?

14. **Weight initialization**: `nn.init.normal_(param, 0, 0.25)`. Standard deviation of 0.25 means most initial actions are small. For a GA, this is fine since mutation will explore. But is there any analysis showing 0.25 is better than 0.1 or 1.0 for initial exploration?

15. **No hidden layer between input and output**: The architecture is `Linear(15,16) → Tanh → Linear(16,4) → Tanh`. There's only one hidden layer. More complex locomotion policies typically need 2+ layers. Is this intentional to keep the parameter count low for the GA?

16. **Output clamping**: Output is already `[-1, 1]` due to final `Tanh`. In `environment.py`, motor rate is `float(actions[i]) * 10.0`, so the effective motor velocity range is `[-10, 10]`. Is this range calibrated to what `motor.max_force = 150000` can physically achieve?

17. **`get_action` always in `torch.no_grad()`**: Correct for inference, but if someone ever wanted to add gradient-based fine-tuning alongside the GA, this would need to be reconsidered. Is gradient-based training a future plan?

---

## `src/evolution.py` — Genetic Algorithm

18. **Elite count hardcoded to 5**: `elites = [self.models[i] for i in sorted_indices[:5]]`. This is hardcoded regardless of `pop_size`. With `pop_size=10`, keeping 5 elites (50%) is very different from keeping 5 out of 100. Should elite count be proportional to population size (e.g. top 10%)?

19. **Mutation rate vs scale**: Default `mutation_rate=0.15` (15% of params mutated) and `mutation_scale=0.35` (noise std). These are applied together — is there any empirical justification for these values? Have different rates been tested?

20. **70/30 mutate/crossover split**: 70% pure mutation from elites, 30% crossover+mutation. Why 70/30 and not 50/50 or 80/20? Has this been tuned?

21. **`random.choices(elites, k=2)` for crossover**: `random.choices` samples with replacement, meaning `parent1` and `parent2` can be the same individual. This produces a child identical to a clone of the parent (crossover of identical parents = the parent). Should it be `random.sample(elites, 2)` to force distinct parents?

22. **Uniform crossover mask**: `mask = (torch.rand_like(p1) < 0.5).float()` — this is parameter-wise 50/50 mixing. For a 2-layer network, does mixing individual weights make more sense than mixing entire layers or neurons?

23. **No fitness normalization before selection**: All models pull from elites only — there's no fitness-proportionate or rank-proportionate selection (roulette wheel / tournament). Only the top 5 ever reproduce. Does this cause premature convergence after the population collapses to 5 similar individuals?

24. **Fitnesses reset to `[0.0] * pop_size` after evolve**: If `evolve()` is called twice without evaluating, fitnesses are already 0 and the second call would sort incorrectly. Is there a guard against this?

---

## `main.py` — Training & Render Loop

25. **`train()` signature vs CLI default mismatch (fixed but worth noting)**: After the debug fix, CLI defaults to `"all_penalties"` and `train()` signature defaults to `"baseline"`. If `train()` is called directly (e.g. from `test_training.py`), it uses `"baseline"`. Should `train()` signature default also be `"all_penalties"`?

26. **Top-5 saved per generation, best_model saved overall**: `top_0..4.pth` are overwritten every generation with the current gen's top 5. `best_model.pth` is the all-time best. When rendering, `top_0..4.pth` are the **last generation's** best, not the all-time best. Is this intentional? Should `best_model.pth` be used as `top_0` fallback across all 5 slots if only one model exists?

27. **Diversity score computation**: `np.mean(np.std(all_params, axis=0))` — this is the mean standard deviation across all parameter positions across the population. Does this accurately capture behavioral diversity, or just weight diversity? Two models can have very different weights but produce identical outputs (due to symmetry in Tanh networks).

28. **`plot_training_history` removed from `main.py`**: The user's edit removed the `from src.plot import plot_training_history` import and the `plot_training_history(show=False)` call at end of training. Is there still a `src/plot.py` file? If so, how do you generate the plot now?

29. **`--plot` flag removed**: The user removed the `--plot` argparse argument. Is there now no way to plot from the CLI? Should it be re-added?

30. **Render loop: `dones` reset when ALL done**: When `all(dones)` is true, all envs reset and the race restarts. But individual creatures that fall early just freeze — their `(DONE)` state is shown but they don't reset independently. Is there a desired behavior where a fallen creature respawns and keeps racing?

31. **Camera follows leader (`tx = 400 - best_x`)**: When the leader is at `X=400` (start), camera is stationary. When leader moves right, camera pans left. But if leader falls back (negative displacement), camera pans right past the start. Is there a minimum camera position (e.g. never go left of start)?

32. **Float `tx` used as integer pixel offset**: `tx = 400 - best_x` is a float. `screen_x = x + tx` is float + int. `pygame.draw.line` and `screen.blit` accept floats in newer pygame versions but this should be verified — is this causing any sub-pixel rendering artifacts?

33. **Early-fall penalty formula**: `500.0 * (1.0 - env.steps / 600.0)`. At step 300 (half-episode), penalty is 250. But the creature has only run half the episode — if it managed 150px displacement before falling at step 300, its total is `150 - 250 = -100`. Does this make early-falling creatures score worse than creatures that immediately fall (who have near-zero displacement + ~500 penalty)? Does the math actually discourage diving?

---

## `tests/` — Test Suite

34. **`test_environment.py` creates `CreatureEnv()` with no args in `test_environment_multiple_resets`**: This uses the default `fitness_mode`. After the user's edit, the default is `"baseline"`. The test doesn't assert which mode is active. Should it explicitly pass `fitness_mode="baseline"` to be unambiguous?

35. **`test_training.py` hardcodes `fitness_mode="baseline"`**: After the debug fix, `"all_penalties"` is the intended production mode. The integration test trains with `"baseline"` and checks model/log creation. Should it also test `"all_penalties"` to ensure the tilt/energy penalty paths don't crash?

36. **No test for fall condition**: There's no test that verifies a creature actually falls when `torso.angle > 1.2` and that `done=True, fallen=True` is returned. This is the core termination logic and it's untested.

37. **No test for the standstill (`stuck`) condition**: `still_steps > 60` causing `done=True` is also untested. A creature that freezes should terminate — but there's no test confirming this.

38. **No test for the early-fall penalty in `main.py`**: The penalty is applied in `train()`, not in `environment.py`, so it can't be unit-tested via `CreatureEnv.step()`. Should it be moved into the environment so it's testable?

39. **`test_mutate_modifies_parameters` uses `mutation_rate=1.0`**: Forces 100% of params to mutate. A real test should also confirm that at `mutation_rate=0.0`, no params change (i.e., the `mutate` function is correct when rate is 0).

---

## `templates/index.html` — Web Dashboard

40. **Dashboard still has `linear-gradient` background**: `background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)`. The user explicitly asked for this to be removed multiple times. Is it intentionally left here, or is this a missed edit?

41. **`{{ url_for('video_feed') }}`**: This is a Jinja2 template tag — it expects a Flask server with a `video_feed` route. Is there still an `app.py` Flask server in the repo? Where does it live and is it functional?

42. **The `index.html` serves a video feed but the video feed route presumably renders simulation frames**: How does the Flask server access `pymunk`/`pygame` render frames? Does it run a background thread with `pygame` and stream JPEG frames via MJPEG? Is this architecture still working after the render loop was rewritten in `main.py`?

---

## `pyproject.toml` — Dependencies

43. **`matplotlib` removed from dependencies**: The user removed `matplotlib>=3.7.0` from `pyproject.toml`. But if `src/plot.py` still exists and imports `matplotlib`, it will crash at runtime. Is `src/plot.py` also deleted?

44. **`requirements.txt` vs `pyproject.toml`**: Both files define dependencies. Are they in sync? Which one is the source of truth?

45. **No `flask` in dependencies**: The `templates/index.html` implies a Flask server, but Flask is not listed in `pyproject.toml` or `requirements.txt`. Is Flask still a dependency of this project?

---

## Overall Architecture

46. **Single `CreatureEnv` instance reused across all training episodes**: In `train()`, one `env = CreatureEnv(...)` is created and `.reset()` is called between each individual's evaluation. Does pymunk correctly reset all joint states, velocities, and motor rates on `.reset()`? Or do any physics states leak between evaluations?

47. **No reproducibility guarantee despite `--seed` flag**: `torch.manual_seed`, `np.random.seed`, and `random.seed` are set, but pymunk's internal C physics engine is not seeded. Is the simulation deterministic given the same inputs, or does pymunk have internal randomness?

48. **No checkpointing of optimizer/GA state mid-run**: If training is interrupted (Ctrl+C), the run starts from scratch. The top models up to that point are saved, but the population state is lost. Should the population be serialized periodically (e.g. every 10 gens) so training can resume?

49. **`models/` can contain stale `.pth` files from previous training runs**: When a new training starts, `history.csv` is wiped but old `.pth` files are not deleted. A new 50-gen run will overwrite `top_0..4.pth` at each generation but `best_model.pth` from a previous run might be better than anything in the current run. Is this the desired behavior?

50. **What is the goal?**: The README says the goal is to explore artificial life and what evolution means. But what does "success" look like? A creature that walks 500px in 600 steps? One that walks indefinitely? One that develops a stable gait? Without a clear success criterion, it's hard to know when training is "done" or when the reward function is working.
