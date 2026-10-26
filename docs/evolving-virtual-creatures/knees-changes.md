# Knees: change record

A record of everything that changed for the "knees" work, written so you can explain each change truthfully. Everything below was read from git (`2ccf649..76a2da5` and `76a2da5..e2fb262`) or from files on disk. Where a statement rests on something I could not see in git, it is marked **[unverified]** or **[from the author]**. Line numbers refer to the working tree at `e2fb262` (current HEAD, `feat/knees-v2`), unless said otherwise.

## 1. Summary

The knees project replaces the walker's four stiff 40 px legs with an optional thigh + shin leg that has a powered knee. It is done as two research steps from Sims (1994), "Evolving Virtual Creatures".

`git log --oneline` (newest first):

| Hash | Message |
|---|---|
| `e2fb262` | feat(knees-v2): penalty fall rule, enforced knee limits, Spot-style knees, GH Actions runner |
| `76a2da5` | feat(knees): optional thigh+shin legs with powered knees, exp3 comparison |
| `2ccf649` | Merge pull request #9 (worktree cleanup consolidation); the base both steps sit on |

**Step 1 (knees v1, `76a2da5`).** Adds `CreatureEnv(knees=True)`: each leg is a 20 px thigh and a 20 px shin joined by a one-way knee (0 to 90 degrees) with its own motor at half the hip force. The brain grows from 15 inputs / 4 outputs (324 weights) to 23 inputs / 8 outputs (520 weights). The research runner learns a `knees` flag that is left out of the config hash when false, so old run hashes do not change. A 2-variant experiment (`exp3_knees`, straight vs knees, 10 seeds) and two pre-registered hypotheses (H6, H7) were added. Result: knees did worse. Every kneed champion "fell" on every test course (fall rate 1.00, distance 160 px vs 394 px for straight).

**Step 2 (knees v2, `e2fb262`).** Diagnosis of step 1 said the kneed creatures were not tipping over; the torso touched the ground while still level, and the knees had bent far past their 90 degree limit. Step 2 therefore adds (a) a `fall_rule="penalty"` option where torso-ground contact costs 2 reward per step instead of ending the run, (b) `clamp_knees()`, which forces knees back inside their limits, (c) a `knees="spot"` variant (14 to 150 degree knees, bent 30 degrees at rest, hips plus/minus 90 degrees), (d) `exp4_knees_v2` and `pilot4_knees_v2` configs with hypotheses H8, H9 (and H10, descriptive), and (e) a GitHub Actions workflow that trains one job per (variant, seed). Exp4 has **not** been run in full yet (see section 6).

## 2. Step 1: knees v1 (commit `76a2da5`)

Diffstat: 22 files, +710 / -12.

### 2.1 `src/environment.py` (+70 / -5 lines)

**Constructor flag** (`__init__`, line 33 and 59). New keyword `knees=False` and `self.knees = bool(knees)`. It is opt-in, so every existing caller gets the old creature. Why: the plan's rule R2 was that turning knees off must change nothing the straight creature produced before.

**Leg construction.** In `create_creature`, right after `self.leg_length = leg_length`, the kneed path is taken and returns early, so the straight-leg loop below it is untouched:

```python
if self.knees:
    self.create_kneed_legs(leg_mass=1, leg_width=10, seg=20)
    return
```

`create_kneed_legs` (now line 119) builds, for each of the four anchors, two boxes of `10 x 20` px with mass 1 each. The straight leg is one `10 x 40` box with mass 2, so total length and mass are unchanged and standing height is the same. Why: so any difference in results comes from the joint, not from a taller or heavier body (plan KTD1). Both shapes get `friction = 1.2` and `ShapeFilter(group=1)` (the same group as the straight legs, so legs do not collide with each other or the torso).

`self.legs` still holds the thighs, so code that reads hip angles (gait analysis, replays) keeps working. Shins are in the new lists `self.shins` and `self.shin_shapes`.

**Joints and motors** (per leg, step-1 version):

```python
pivot = pymunk.PivotJoint(self.torso, thigh, anchor, (0, -seg / 2))
limit = pymunk.RotaryLimitJoint(self.torso, thigh, -np.pi / 3, np.pi / 3)   # hip +-60 deg
motor = pymunk.SimpleMotor(self.torso, thigh, 0);  motor.max_force = 150000
knee = pymunk.PivotJoint(thigh, shin, (0, seg / 2), (0, -seg / 2))
knee_limit = pymunk.RotaryLimitJoint(thigh, shin, -np.pi / 2, 0)           # one-way, 0..90 deg
knee_motor = pymunk.SimpleMotor(thigh, shin, 0);   knee_motor.max_force = 75000
```

- Hip: same pivot/limit/force pattern as the straight leg (150000, +-60 degrees).
- Knee force 75000 = half the hip. Reason (plan KTD3): Sims section 3.3 scales joint strength with the cross-section of the parts joined; the hip joins a 20 px torso to a 10 px thigh, the knee joins two 10 px parts.
- Knee limit `-pi/2 .. 0`: the relative angle `shin.angle - thigh.angle` may only be negative. The sign is a pymunk y-down convention (see section 7 on the docs mistake about this).
- Motor order: the four hip motors are appended first in the loop, the four knee motors are collected and added afterwards with `self.motors.extend(knee_motors)`. So actions 0-3 are hips, 4-7 are knees.

**Reset** (the `if self.knees:` block in `reset`). Straight reset only repositions `self.legs`. For kneed bodies the shin must also be placed, so reset puts the thigh center at `anchor + h*(−sin th, cos th)` and the shin center at `3h` along the same line (`h = seg/2 = 10`), and sets `leg.angle = shin.angle = th` where `th` is the pose-noise angle. Both bodies get zero velocity. Why: pose noise must rotate the whole leg about its hip, otherwise the shin would be left behind.

**Step** (line 222). Old: `motor.rate = float(action) * 10.0`. New:

```python
motor.rate = float(action) * (-10.0 if self.knees and i >= 4 else 10.0)
```

Knee motors (index 4-7) have their sign flipped. This is so that, for the brain, a knee action of -1 bends the knee and +1 straightens it (confirm in section 7). For straight bodies `self.knees` is false, so the expression is `action * 10.0`, identical to before.

**State vector** (`get_state`, line ~297). The old `np.array([...])` literal became a list `values`, with the original 15 entries in the same order; if knees, 8 more are appended:

```python
if self.knees:
    for leg, shin in zip(legs, self.shins):
        values += [shin.angle - leg.angle, shin.angular_velocity * 0.1]
```

So indices 0-14 keep their meaning and 15-22 are (relative knee angle, knee angular velocity x 0.1) for each leg. The 0.1 scaling matches the hips. Total state length 23.

The docstring in step 1 also said "a positive knee action swings the foot backward". That sentence was wrong and was replaced in step 2.

### 2.2 `src/research/config.py` (+5 / -1)

- `RunConfig` gets `knees: bool = False` (line 25).
- `sha256()` now drops the key when false:

```python
d = asdict(self)
if not d["knees"]:
    del d["knees"]  # keeps every pre-knees run hash unchanged
return config_sha256(d)
```

Why: the run hash is how the runner decides whether an existing results folder matches the config. If a new field with a default value were hashed, every existing results folder (including Colab runs on Drive) would look stale. Leaving the default out keeps old hashes identical.

### 2.3 `src/research/evaluate.py` (+9 / -1)

- New constants: `KNEE_SHAPES = {"w1": (16, 23), "b1": (16,), "w2": (8, 16), "b2": (8,)}` and `KNEE_DIM` (line 23). Counting: 16x23 + 16 + 8x16 + 8 = 368 + 16 + 128 + 8 = **520**. Straight: 16x15 + 16 + 4x16 + 4 = 240 + 16 + 64 + 4 = **324**.
- `dim(cfg)` (line 26) returns `KNEE_DIM if cfg.knees else DIM`. `DIM` stays for code that has no config (e.g. the playground).
- `from_flat` (line 36) picks the shape set by vector length (`KNEE_SHAPES if len(vec) == KNEE_DIM else SHAPES`). This means the flat vector alone says which body it belongs to; no extra flag is stored.
- `make_env` passes `knees=cfg.knees` to `CreatureEnv`.

### 2.4 `src/research/runner.py` (+2 / -2)

`run_one` used `DIM` for the optimizer size; now `dim(cfg)`. The import of `DIM` was replaced with `dim`. That is the only runner change.

### 2.5 `experiments/exp3_knees.json` (new, 36 lines)

- optimizer `ga`, `pop_size` 50, `episodes_budget` 25000, `rollouts_k` 5, `eval_every_episodes` 1250.
- seeds `[0..9]` (10 seeds).
- variants: `straight: {}` and `knees: {"knees": true}`.
- hypotheses H6 and H7 (below).

The seed count is relevant to section 6: the file has 10 seeds; the blog says 30 were pre-registered **[from the author; not visible in git, I did not find a 30-seed entry in the committed files]**.

### 2.6 `experiments/HYPOTHESES.md` (+11 / -1)

Pre-registers, before training:
- **H6:** kneed walkers reach a higher held-out test distance IQM than straight ones at equal budget (test distance, A>B).
- **H7:** kneed walkers fall less often on held-out test courses (fall rate, A<B).
- A rule that if the kneed validation curve is still rising over the last 20% of budget, H6/H7 are reported as inconclusive, not refuted, because the kneed genome is bigger (520 vs 324). Knee strength may only be tuned before the full run, using train/validation splits.
- Two rows added to the contrasts table and "exp3: H6, H7" added to the Holm correction sets (Holm is applied only within each experiment's confirmatory set).

### 2.7 Tests (added in step 1)

`tests/test_knees_env.py`:
- `test_default_env_unchanged`: default env has a 15-value state, 4 legs, 4 motors.
- `test_kneed_reset_shape`: kneed state length 23, 4 thighs, 4 shins, 8 motors, and the knee-angle entries (indices 15, 17, 19, 21) are exactly 0 at reset.
- `test_positive_knee_action_swings_foot_backward`: held knee action +1 for 30 steps; knee angle stays within (0, pi/2+0.05] in absolute value and the shin end ends up behind the thigh end. **Replaced in step 2** (see 3.8) because the sign convention in its name was wrong.
- `test_kneed_zero_actions_stand`: 30 zero-action steps, never `fallen`.
- `test_kneed_pose_noise`: after `reset(pose_noise=[0.1,-0.1,0.2,0.0])`, each thigh angle equals the noise and the knee angular velocities are ~0.
- `test_kneed_episode_runs`: a `CreatureBrain(input_size=23, output_size=8)` runs 100 steps via `run_episode` without error.

`tests/test_knees_research.py`:
- `test_straight_hash_unchanged`: hash of a default config equals the hash of the dict with `knees` removed (proves old hashes are preserved).
- `test_knees_hash_differs`: knees config hash is different from straight.
- `test_flat_round_trip_knees`: a random 520-vector goes through `from_flat`/`to_flat` unchanged, with the expected 4 shapes.
- `test_run_one_knees_deterministic`: two tiny runs write identical `history.csv`, and the champion `w1` has shape (16, 23).
- `test_unknown_variant_key_rejected`: a variant with an unknown key (`bogus`) raises `ValueError`.

### 2.8 Docs added in step 1

- `docs/evolving-virtual-creatures/README.md`: names the paper, describes Evolution-1, lists the roadmap and the backlog (items 7-21).
- `01-knees/plan.md` (266 lines), `abstract.md`, `results.md` (117 lines), and `screenshots/F1..F10` PNG charts from the analysis.
- The step-1 plan also mentioned `tests/test_knees_cli.py` and `main.py --knees`; neither exists in the commit (see section 6).
- `results.md` reports: distance 394.2 px [190.4, 718.3] straight vs 160.0 px [148.0, 187.0] knees; fall rate 0.17 vs 1.00; H6 inconclusive (p = 0.054), H7 not supported (p = 0.0001). Cause given: the walkers learned to "dive".

## 3. Step 2: knees v2 (commit `e2fb262`)

Diffstat: 16 files, +478 / -29.

### 3.1 `fall_rule` in `src/environment.py`

Constructor (line 34) gains `fall_rule="touch"` plus validation:

```python
if knees not in (False, True, "spot"):
    raise ValueError("knees must be False, True or 'spot'")
if fall_rule not in ("touch", "penalty"):
    raise ValueError("fall_rule must be 'touch' or 'penalty'")
```

`self.fall_rule = fall_rule` (line 61). In `step` (line 256):

```python
if self.fall_rule == "penalty":
    if body_touched_ground:
        reward -= 2.0
    fallen = abs(torso_angle) > 1.2 or self.torso.position.y > deep
elif self.fitness_mode in ["ground_termination", "touch_ground"]:
    ...old branch...
```

Meaning: with `penalty`, torso-on-ground costs 2 reward every step it happens and does **not** end the run. Tipping past 1.2 rad or sinking below the course still ends it as a fall. With `touch` (the default) the old branches run unchanged.

Why: the step-1 diagnosis (done on seeds 0, 1, 4 **[scratchpad, section 5]**) showed episodes ended while the torso was level (tilt 0.07-0.78 rad). The creatures were crouching and the rules counted a crouch as death. The author picked "option 3" (a reward cost) over a grace period or tilt-only rule so that crouching costs reward but lying on the ground never pays. Why -2: the plan's reasoning is that about 2-3 px/step x reward factor 2 for a decent walker is roughly cancelled by -2/step.

Note: the `fitness_mode` options (`ground_termination`, energy/tilt penalties) are untouched; the `penalty` branch is checked first and bypasses them.

### 3.2 `clamp_knees()` and the soft-limit leak

New method (line 158):

```python
def clamp_knees(self):
    lo, hi = self.knee_range
    for thigh, shin in zip(self.legs, self.shins):
        rel = shin.angle - thigh.angle
        if lo <= rel <= hi:
            continue
        d = (lo if rel < lo else hi) - rel
        knee = thigh.local_to_world((0, self.seg_length / 2))
        c, s_ = math.cos(d), math.sin(d)
        off = shin.position - knee
        shin.position = (knee.x + c * off.x - s_ * off.y, knee.y + s_ * off.x + c * off.y)
        shin.angle += d
        if (shin.angular_velocity - thigh.angular_velocity) * d < 0:
            shin.angular_velocity = thigh.angular_velocity
```

It is called inside the substep loop (line 227), kneed bodies only:

```python
for _ in range(self.substeps):
    self.space.step(3 / 60.0 / self.substeps)
    if self.knees:
        self.clamp_knees()
```

**Why it was needed.** A pymunk `RotaryLimitJoint` is a soft constraint: it only pushes back after the angle is already past the limit, and a strong enough motor can overpower it. The docstring records the observation: "116-128 deg seen on a 90 deg limit". `results.md` says it leaked up to ~30 degrees past either stop within one physics step. **[Those 116-128 and ~30 numbers come from the author's diagnostic, which is not in the repo.]**

**What the clamp does.** After every physics step, if a knee angle is outside `knee_range`, it rotates the shin back to the limit about the knee pivot (world point at the thigh's lower end) and, if the shin is still spinning outward relative to the thigh, sets its angular velocity to the thigh's. The limit joint is still added; the clamp is an extra hard stop.

**Why only kneed bodies.** So the straight creature runs the exact same code as before (`self.knees` false skips it). This is the point of the fingerprint check (section 5).

Test: `test_knee_stays_within_limit` (see 3.8).

### 3.3 The `spot` variant

`knees` is now `False | True | "spot"`; `self.spot = knees == "spot"` (line 60), and `self.knees` is still `bool(knees)`, so `"spot"` also enables kneed legs.

- **Knee range** (line 121): `self.knee_range = (-radians(150), -radians(14)) if self.spot else (-pi/2, 0.0)`. This is used for both the limit joint (`*self.knee_range`) and `clamp_knees`.
- **Hip range** (line 144): `hip_range = np.pi / 2 if self.spot else np.pi / 3`, i.e. plus/minus 90 degrees instead of 60.
- **Reset pose** (line 195-):
  - knee = -30 degrees; `thigh_tilt = -knee / 2` (so the foot ends up roughly under the hip).
  - the torso is lowered so the feet touch the ground: `torso_y = 500 - 10 - 2*h*(cos(tilt) + cos(tilt+knee))/2` (ground top at y=500 minus half the 20 px torso height... **[the geometric reading of the constants 500 and 10 is my interpretation of the code comment "feet touch the ground"; I did not verify against the ground definition]**).
  - each leg: thigh angle `th = thigh_tilt + pose_noise[i]`, shin angle `th + knee`; positions computed from the hip point and the knee point.
  - for non-spot kneed bodies `knee = 0` and `thigh_tilt = 0`, `torso_y = 450`, so the body is the same as step 1 (the code was rewritten in a more general form, and the 450 position is the same as before).

Why Spot: the abstract cites the Boston Dynamics Spot SDK ("About Spot" page): knees about 14 to 160 degrees and body 191 mm off the floor. The code uses 150 for the upper bound, not 160 **[the abstract states Spot's real range as 14-160; the choice of 150 for the sim is the author's; I found no written reason for 150 vs 160]**. The abstract's reason for never fully straightening: a straight knee has almost no leverage to start bending, because the motor pushes along the leg, not across it.

### 3.4 Research plumbing

`src/research/config.py`:
- `knees: bool | str = False` (comment: False, True or "spot") and `fall_rule: str = "touch"`.
- `sha256()` now also drops `fall_rule` when it is `"touch"`:

```python
if d["fall_rule"] == "touch":
    del d["fall_rule"]
```

Same reason as in step 1: default-valued fields must not change old hashes. A `knees="spot"` config has a truthy `knees` value, so it stays in the hash and differs from `True`.

`src/research/evaluate.py`: `make_env` also passes `fall_rule=cfg.fall_rule`. `dim(cfg)` already returns 520 for any truthy `knees`, so `"spot"` uses the 520-weight brain with no change.

### 3.5 Experiment configs

`experiments/exp4_knees_v2.json` (40 lines): same settings as exp3 (ga, pop 50, 25000 episodes, seeds 0-9, `eval_every_episodes` 1250) plus top-level `"fall_rule": "penalty"` and three variants: `straight: {}`, `knees: {"knees": true}`, `knees_spot: {"knees": "spot"}`. Hypotheses H8 and H9 (below). That would be 3 variants x 10 seeds = 30 jobs.

`experiments/pilot4_knees_v2.json` (19 lines): seeds `[100, 101]`, pop 100, 30000 episodes, `eval_every_episodes` 3000, same 3 variants and `penalty`, `hypotheses: []`, and its own `test_terrain_seeds` 5000-5009. Its `_note` says why: tune knee settings only here, and use different test courses so exp4's test set (terrain seeds 1000-1009) stays unseen.

### 3.6 `experiments/HYPOTHESES.md`

- **H8:** under the penalty rule, `knees` beats `straight` on test distance IQM.
- **H9:** `knees_spot` beats `knees` on test distance IQM.
- **H10:** descriptive only, not tested: kneed walkers fall less than exp3's 100%.
- Holm set for exp4 is H8, H9. Absolute distances are not compared with exp3 because the hardware differs (laptop vs GitHub runner).
- Contrast table gets H8 and H9 rows.

### 3.7 `.github/workflows/experiment.yml` (new, 89 lines)

Triggers (lines 3-11): `push` to branches `experiment/**`, and `workflow_dispatch` with an `experiment` input (default `experiments/exp4_knees_v2.json`).

**Why a push trigger.** The comment in the file says: "dispatch manually once this file is on the default branch." GitHub only offers manual dispatch for workflow files on the default branch, and this file lives on a feature branch. Pushing a branch named `experiment/<config-name>` runs `experiments/<config-name>.json` without needing a merge first. (This is the stated reasoning in the file; I did not test it.)

Jobs:
1. **plan** (ubuntu-latest): checks out, sets up Python 3.12, resolves the config path (`INPUT_EXP` or `experiments/${GITHUB_REF_NAME#experiment/}.json`), fails if the file is missing, and writes a matrix of `{variant, seed}` pairs to `$GITHUB_OUTPUT` (one Python one-liner reading the config's `variants` and `seeds`).
2. **run** (matrix, `fail-fast: false`, `timeout-minutes: 300`): one job per (variant, seed). Installs `requirements.txt scipy cma matplotlib`, runs `python scripts/run_experiment.py "$EXP" --only "$VARIANT" --seeds "$SEED" --workers 1`, and uploads `results/` as artifact `run-<variant>-<seed>`. `fail-fast: false` so one failed seed does not cancel the rest.
3. **collect** (`needs: [plan, run]`, `if: always() && needs.plan.result == 'success'`): downloads all `run-*` artifacts merged into `results/`, rebuilds the manifest, runs `verify_results.py` (with `continue-on-error: true`), runs `analyze.py ... --no-replay --allow-incomplete`, and uploads `results/<exp name>` as artifact `results-<name>`. `always()` means it still produces a report if some runs failed.

**Why values go through env vars (injection safety).** Values like `inputs.experiment`, `matrix.variant` and `matrix.seed` are user-controlled text. If they were pasted straight into the `run:` script with `${{ ... }}`, GitHub would substitute them into the shell text before the shell parses it, so a crafted value could inject commands. Passing them as `env:` and using `"$EXP"`, `"$VARIANT"` and `"$SEED"` keeps them as plain data (shell variables, quoted). Examples in the file: `INPUT_EXP`, `EXP`/`VARIANT`/`SEED`, and `EXP: ${{ needs.plan.outputs.exp }}` at the job level for collect. The Python one-liner also reads the path from `os.environ['EXP']` rather than interpolating it. **[This is the standard GitHub security guidance; the file follows it, but the author's stated reason is what I was given, not text in the repo.]**

**Why collect rebuilds the manifest instead of running training.** The step is named "Rebuild manifest (never trains)". Each run job produced only its own `results/<exp>/<variant>/seed_N/` folder. `collect` merges those folders and needs the experiment-level `experiment.json` and `manifest.json` so `verify_results.py` and `analyze.py` accept the folder. It calls `_write_manifest(e, 'results', {})` directly (a private function in `runner.py`, line 149) rather than `run_experiment.py`, because running the main script on a merged folder could retrain anything it judged missing or stale. Calling `_write_manifest` only writes the manifest from what is on disk. **[The "could retrain" reasoning is inferred from the step name and the code; it is not stated in a comment.]**

### 3.8 Tests changed or added in step 2

Changed:
- `tests/test_knees_env.py`: the old foot-backward test became `test_negative_knee_action_bends_knee`. It runs action -1 and +1 for 30 steps. For -1, each knee angle is within `[-pi/2 - 0.09, -0.3]` (bent). For +1, `abs(rel) <= 0.2` (straight, with a comment "one leg jitters to ~0.155 mid-stride").
- `tests/test_knees_research.py::test_straight_hash_unchanged`: now deletes both `knees` and `fall_rule` before comparing.

New, `tests/test_knees_v2_env.py`:
- `test_default_env_unchanged`: default env still 15 inputs, 4 legs, 4 motors.
- `test_invalid_args_raise`: `knees="foo"` and `fall_rule="foo"` raise `ValueError`.
- `test_penalty_rule_costs_two_and_does_not_fall`: torso position forced to y=490 (below ground top); `touch` falls, `penalty` doesn't, and the penalty reward is about 2 lower (tolerance 0.5).
- `test_penalty_rule_tilt_still_falls`: torso angle 1.3 rad -> `fallen` True under `penalty`.
- `test_knee_stays_within_limit` (parametrized for `True` and `"spot"`): over 60 steps with three action plans (all +1, all -1, random in [-1, 1] with `Random(0)`), every knee angle stays within the range with 5 degrees tolerance.
- `test_spot_reset_pose_and_stands`: spot reset gives 23 inputs, 8 motors, every knee at -30.0 degrees, and no fall in 30 zero-action steps.

New, `tests/test_knees_v2_research.py`:
- `test_default_hash_ignores_new_fields`: default `RunConfig` hash equals hash of the dict minus `knees` and `fall_rule`.
- `test_new_fields_change_hash`: `fall_rule="penalty"` and `knees="spot"` each change the hash.
- `test_tiny_spot_penalty_run_deterministic`: two tiny `run_one` runs with spot + penalty return `"done"`, give identical `history.csv`, and champion `w1` shape (16, 23).

I did not run the tests for this record (instructions: read-only, nothing heavy). See section 9.

### 3.9 Docs in step 2

- `02-knees-v2/plan.md` (new, 102 lines) and `02-knees-v2/abstract.md` (new, 33 lines).
- `README.md`: roadmap renumbered. Step 2 is now "Knees v2"; contact sensors, multi-course, settle-before-scoring, size-scaled mutation and dashboard moved to steps 3-7.
- `01-knees/abstract.md`: one sentence corrected about the knee direction ("foot swinging backward" -> "the foot folds forward under the body").
- `01-knees/results.md`: one caveat line added about the knee limit leak (see section 7).
- `experiments/HYPOTHESES.md`: see 3.6.
- There is **no** `02-knees-v2/results.md` yet because exp4 has not run.

## 4. Uncommitted work

`git status --porcelain` at the time of writing:

```
?? docs/blog/
```

Only `docs/blog/` is untracked. Contents:
- `docs/blog/01-knees-part-1.md` (146 lines): first blog draft, "I gave my walking robot knees. It learned to fall forward instead."
- `docs/blog/01-knees-part-1.v2.md` (147 lines): second draft. It still says `images/...` for its own images and reuses charts from `../evolving-virtual-creatures/01-knees/screenshots/` (F1, F2).
- `docs/blog/images/`: `dive-replay.gif`, `fitness-reward.png`, `fitness-fall-rule.png`, `knee-code.png`, `replay-course-a.png`, `brain-explorer-knees.png`.

Which images the v2 draft references (grep of `.png`/`.gif`): `dive-replay.gif`, `fitness-reward.png`, `fitness-fall-rule.png`, `knee-code.png`, `replay-course-a.png`, plus F2 and F1. `brain-explorer-knees.png` is in the folder but my grep did not find it referenced in v2 **[it may be used in v1 or not at all; not checked]**.

How each image was made **[from the author's brief and the scratchpad scripts; I did not rerun them]**:
- `dive-replay.gif`, `replay-course-a.png`, `fitness-*.png`, `brain-explorer-knees.png`: rendered by the blog image renderer `scratchpad/blog_img/make.py` (section 5). I cannot tell from git which image comes from which function inside it.
- `knee-code.png` and the fitness code images: code screenshots produced by `scratchpad/code_shots/shot.py`.

**Warning on the "byte-for-byte" claim.** The v1 draft still contains it (lines 38 and 141). The v2 draft does not contain the word "byte" at all (grep). So the claim was removed only in v2; if v1 is ever published it needs the same edit. The v2 draft also contains the deviation note about 30 vs 10 seeds (line 88) and "Log deviations when you make them" (line 128).

**Results folders.** `results/exp3_knees/` (10 seeds x 2 variants + `summary/` with `REPORT.md`, `summary.csv`, `pairwise.csv`, `figures/`) and `results/pilot4_knees_v2/` (3 variants + `summary/`) are on disk. `git check-ignore -v results` returns `.gitignore:32:results/`, so **`results/` is ignored, not tracked**, and these runs are not in git. They exist only on this machine (and for the pilot, as GitHub Actions artifacts, which expire **[expiry not checked]**).

## 5. Tools that live outside the repo (scratchpad, NOT committed)

These exist **only on the author's machine** in the session scratchpad
`C:\Users\Wahab\AppData\Local\Temp\claude\d--wahab-stuff-wahab-code-Evolution-1\2bb5af0a-272d-49de-be02-588ee9f68214\scratchpad\`. Nobody else can run them, and the temp directory can be cleaned. I listed the directory and confirmed these names exist; I did not read or run them, so what each does below is from the author's description.

| Tool | Purpose |
|---|---|
| `fingerprint.py`, `fingerprint_before.txt` | Ran the straight creature step by step and compared its output before vs after the change to show it was unchanged. Both exist in the scratchpad (also `fingerprint_u1.txt`, `fingerprint_u2.txt`, `fp_after.txt`, `fp_final.txt`). The commit messages say "verified byte-identical" but the proof is this untracked script. The blog no longer says "byte-for-byte" (v2) because the script isn't in the repo. |
| exp3 kneed-fall diagnostic (seeds 0, 1, 4) | Source of "tilt 0.07-0.78 rad" and "knee 116-128 degrees" quoted in `02-knees-v2/abstract.md` and plan. I could not identify which file it is (`knee_probe.py`, `watch_knees.py`, `check.py` are candidates, **unverified**). |
| `pilot_diag/diag.py` | Diagnostic for the pilot run results. |
| `blog_img/make.py` | Renders the blog images. |
| `code_shots/shot.py` | Makes the code screenshots. |
| `blog_preview_build/build.py`, `build_v2.py` | Build an HTML preview of the blog drafts (v1 and v2). |

Two private claude.ai pages exist (a results page and a "Brain Explorer") **[from the author; I have no URLs and cannot see them]**. They are not in the repo.

These should be committed (or the claims dependent on them removed) before the blog is published.

## 6. What we did NOT do, and why

| Item | Status | Reason |
|---|---|---|
| `main.py --knees` / dashboard showing kneed creatures | Not done (`grep -c knees main.py` = 0; no `test_knees_cli.py`) | Deferred to the dashboard step (roadmap step 7); out of scope for a research PR. The step-1 plan listed it, but it was not built. |
| Size-scaled mutation (Sims sec. 6.1) | Not done | Roadmap step 6, a separate one-idea-per-PR change. README notes the 520-weight brain gets ~78 changes/child vs ~49 for 324 weights, so the comparison is slightly confounded. |
| Contact sensors, multi-course scoring, settle-before-scoring | Not done | Roadmap steps 3, 4, 5. |
| 30 seeds pre-registered for exp3 | Only 10 run | To save laptop time (CPU at 87 C). Not logged as a deviation at the time; the v2 blog now discloses it. **[from the author; committed exp3 config has 10 seeds]** |
| Fingerprint script and diagnostics | Not committed | They are scratchpad files (section 5). Should be committed before the blog is published. |
| Raw results | Not in git | `results/` is in `.gitignore` (line 32). |
| Ablations for exp4: knees with fixed limit under old fall rule; straight with the 520-weight brain | Not added | Proposed, waiting on the author's decision. |
| Full exp4 run (30 jobs) | Not run | Waiting on pilot-based decisions. |
| `fitness_mode` options (tilt/energy penalties) | Untouched, unused | They exist in the code but experiments use baseline mode. |
| Re-run of knees v1 after the limit fix | Not done | Step 1 numbers include the leak; `results.md` has a caveat saying so. |
| Replay differences Linux vs Windows | Not handled | Pilot found champions trained on Linux (GitHub) replay differently on Windows (4 of 6 differ **[from the author]**). So exp4 analysis with replays should run on CI; not implemented. The workflow currently uses `--no-replay`. |
| `knees_spot` pilot champions | Not fixed | They tip over on every course and were flagged by the dt/2 exploit check **[from the author; pilot results are on disk but I did not analyze them]**. Waiting on the author's decision. |
| Backlog #7-21 (README) | Not started unless noted | See list below. |
| PR for `feat/knees-v2` | Not opened | PR #10 `feat(knees): optional powered knees + exp3 comparison` (branch `feat/knees`) is OPEN (confirmed with `gh pr list`). No PR for `feat/knees-v2`. |

Backlog items from `docs/evolving-virtual-creatures/README.md`:

| # | Idea | Status (from README) |
|---|---|---|
| 7 | Energy penalty in fitness | Partly exists: `--fitness-mode energy_penalized` subtracts sum of abs(action); not used in the research runner |
| 8 | Learning-curve plot in README | Plots exist (`scripts/plot_logs.py`, F1); not embedded in the top-level README |
| 9 | Ablation table | CPG ablation exists (`exp1_cpg.json`); knees is step 1; sensors and multi-course rows come later |
| 10 | GIF of best walker | Exists (`docs/media/walking.gif`); refresh after a better walker |
| 11 | Single config file | Experiments already use JSON; body params still hard-coded in `src/environment.py`; YAML not planned |
| 12 | Symmetry constraint | Not started |
| 13 | Push recovery | Not started |
| 14 | Curriculum | Not started |
| 15 | PPO baseline | Not started (Gymnasium wrapper `src/gym_env.py` exists) |
| 16 | Domain randomization | Partly exists (friction x0.9-1.1, start pose); mass/damping not randomized |
| 17 | Recurrent brain | Not started |
| 18 | Body co-design | Listed as a later idea (Sims sec. 2) |
| 19 | MuJoCo 3D port | Not started |
| 20 | One-page tech report | Per-step abstract/results are raw material |
| 21 | Sim-to-real section | Not started |

## 7. Known issues and corrections

1. **Knee direction docs were wrong in step 1.** Step-1 `abstract.md` said the foot swings backward; the step-1 docstring said "a positive knee action swings the foot backward"; the step-1 test was named `test_positive_knee_action_swings_foot_backward`. Step 2 changed all three. Current truth in the code: knee range is negative (`-pi/2 .. 0`), knee motor rate is `action * -10`, and `test_negative_knee_action_bends_knee` asserts that action -1 gives a bent angle in `[-pi/2-0.09, -0.3]` and +1 gives about 0. The current docstring says "-1 bends the knee (foot moves forward of the knee), +1 straightens it". I confirmed the sign flip and the range in the code (`environment.py` lines 121, 222) and the test's assertions; I did not run the simulation.
2. **Soft-limit leak.** Limit joints let the motor push the knee past its limit (116-128 degrees seen on a 90 degree limit **[from the author's diagnostic]**). Fixed in step 2 by `clamp_knees()`. Step-1 results include the bug; `results.md` now says so. Step 1 results were not re-run.
3. **`results.md` fall-rate row label.** The headline table's lead-in says "IQM with 95% CI in brackets", but `summary/REPORT.md` labels M2 as "fall rate [95% CI, by seed]" and its figure caption says "Bars: mean per-seed fall rate". So the 0.17 in `results.md` is the **mean**, not an IQM. The author states the straight IQM is 0.15 **[not visible in the files I read; `REPORT.md` shows 0.17 only]**. Fix: relabel the row as a mean.
4. **"difference +0.8" in `results.md` H7 row.** `results.md` (and `REPORT.md` line 17) show `+0.8 [+0.7, +1.0]`. The author says the right value is +0.85 (1.00 - 0.15, using the IQM). **[Not verifiable from the files I read; the values in REPORT.md are rounded to +0.8.]** Fix by recomputing from `pairwise.csv`.
5. **`git_commit` in REPORT.md.** `results/exp3_knees/summary/REPORT.md` line 92: `git_commit: 2ccf649605ef038f93a1fc2f2727155b5215bce1 (dirty: True)`. So exp3 was trained from base commit `2ccf649` with uncommitted changes, not from `76a2da5`. That is expected (training happened before the knees commit) but a reader may be confused, and the knees code is only in the dirty working tree at the time.
6. **Blog v1 vs v2.** v1 still contains "byte-for-byte" claims (lines 38, 141).
7. **Plan mentions things that do not exist**: `tests/test_knees_cli.py`, `main.py --knees`, `data/models/knees/` (plan KTD9, R6). They are not in `76a2da5`.
8. **Leftover step-1 docstring detail**: `create_kneed_legs` earlier code assumed a straight pose with `450` as the torso y; step 2 generalized it. No bug found, just noting that reset code was rewritten.

## 8. Who did what

- **The author (you)** chose the papers and ideas, decided one idea per PR, picked fall rule option 3, the Spot variant, GitHub Actions for training, and the pilot, and reviews the results. **[from the author; the plan files record these as "session-settled: user-directed".]**
- **Claude Opus** planned and verified. **Claude Sonnet subagents** wrote the code, tests and docs. **Claude Haiku subagents** did lookups. **[from the author]** The committed plans differ slightly: the step-1 plan says "U1-U5 are coded by Haiku 5.5 subagents", and the step-2 plan says "Sonnet 5.5 subagents (low effort) write the code". If you want to state this exactly, use the plan files as the source; I cannot see from git which model wrote which line.
- **Commit trailers:** `git log -3 --format=%B` shows `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` on both `76a2da5` and `e2fb262`. The base `2ccf649` has none.

## 9. How to verify all of this yourself

Git (read-only):

```
git log --oneline -8
git diff 2ccf649 76a2da5 --stat
git diff 2ccf649 76a2da5 -- src tests experiments
git diff 76a2da5 e2fb262 --stat
git diff 76a2da5 e2fb262 -- src tests experiments .github
git status --porcelain
git check-ignore -v results
git log -3 --format=%B
gh pr list
```

Tests (these run physics and tiny training runs, so they take a short while):

```
python -m pytest tests/test_knees_env.py tests/test_knees_research.py tests/test_knees_v2_env.py tests/test_knees_v2_research.py -q
python -m pytest -q
ruff check src tests
```

Straight-creature unchanged check (author's scratchpad scripts, only on this machine): run `fingerprint.py` and compare with `fingerprint_before.txt` in the scratchpad directory from section 5. If those files are gone, the claim "straight creature unchanged" is supported only by `test_default_env_unchanged` (shape checks), `test_straight_hash_unchanged`/`test_default_hash_ignores_new_fields` (hash checks), not by a step-by-step trajectory comparison.

Results and reports (local only):

```
python scripts/verify_results.py results/exp3_knees
python scripts/analyze.py results/exp3_knees
```

Look at `results/exp3_knees/summary/REPORT.md` for the M2 label and the H7 row discussed in section 7.

## 10. Things I could not verify

The first draft of this file left the items below unverified. The ones marked **Resolved** were checked afterwards, and the evidence is given for each.

- **Resolved: 30-seed pre-registration.** `git show 76a2da5:experiments/HYPOTHESES.md`, line 71: "**Seeds:** 30 per variant (0-29)". `exp3_knees.json` ran 10. This is an undisclosed deviation, which the v2 blog now discloses.
- **Resolved: straight fall-rate IQM 0.15, difference +0.85.** These are in `results/exp3_knees/summary/summary.csv` (columns `fall_rate_iqm` and `iqm_diff`). That folder isn't tracked in git, so this can be checked only on the author's machine. `REPORT.md` and `results.md` round the difference to +0.8 and label the mean (0.17) as IQM.
- **Resolved: Linux vs Windows replay mismatch, and the `knees_spot` pilot tipping.** Recorded in `scratchpad/pilot_diag/summary.txt`, which is not in the repo. Replayed on Windows, 4 of 6 pilot champions don't reproduce their CI `final_test.json` mean distance; knees_spot seed 100 is the only exact match. Both knees_spot champions end all 10 test courses by tilt, at about 541 steps.
- **Resolved: the GitHub Actions workflow runs.** Pilot run 40206005186 finished: plan, all 6 run jobs, and collect all passed (`gh run view 40206005186`).
- **Resolved: the two private claude.ai pages.** Results page https://claude.ai/artifact/9M8NqgjEyswmAeb1eAaQ71; Brain Explorer https://claude.ai/artifact/G8WaBZvtQF7VSYZJdeh4j1. Both are private to the author's account.
- **Still unverifiable: diagnostic numbers** (tilt 0.07–0.78 rad, knees at 116–128°, ~28–30° leak per step). They came from a diagnostic run on kneed seeds 0, 1 and 4 during the step-1 work session. That script can't be clearly identified in the scratchpad now. They should be regenerated with a committed script before the blog is published; until then, treat them as unverified.
- **Still unverified: the contents of the scratchpad scripts.** The file names exist, but they weren't re-read for this document.
- **Not run for this document: the test suite.** It was last run during step 2 (117 passed); rerun it with `python -m pytest -q`.

## Step 2b: physics guards (uncommitted)

All of this sits behind one opt-in flag, `guards` (default `False`). With it off, the env produces byte-identical states (tested) and run hashes do not change.

### Why

A diagnostic on the `pilot4_knees_v2` champions found three problems:

- **Spot glitch.** The `knees_spot` champion (seed 100) stands almost still for 530 steps with all four feet down and its knees pressed against the 14 degree limit. At steps 531-533 the physics blows up: torso angular velocity reaches -9.7, hips read +341/-404 degrees, and the body is launched about 380 px.
- **Hip limits leak.** Plain knees hips reached -114..+194 degrees against a +-60 degree limit. Hips use a soft `RotaryLimitJoint`, and only knees had `clamp_knees()`.
- **Stuck rule never fires.** The Spot champion moved 179 px in 530 steps, but the stuck rule needs |vx| < 2 px/s, so slow shuffling never counts.

### What changed, file by file

`src/environment.py`
- `CreatureEnv(..., guards=False)`.
- The clamp was generalised into one helper. `clamp_knees()` now calls it, and `clamp_hips()` is new (limit +-90 degrees for spot, +-60 otherwise, pivot on the torso anchor). Hips are clamped after every substep when `guards` is on.

```python
def _clamp_joint(parent, child, pivot, lo, hi): ...   # was the body of clamp_knees
def clamp_hips(self):
    hip = math.pi / 2 if self.spot else math.pi / 3
    for leg, anchor in zip(self.legs, self.anchors):
        self._clamp_joint(self.torso, leg, self.torso.local_to_world(anchor), -hip, hip)
```

- Sanity cutoff: after the substeps, if any body (torso, thighs, shins) has speed > `SANE_MAX_SPEED` or |angular velocity| > `SANE_MAX_OMEGA` (or is non-finite), the step returns `(state, 0.0, True, False)` and sets `env.invalid = True`. No -100 (that is only applied for `fallen`), and the triggering step earns no reward.
- Distance freeze: `env.score_x` returns `last_sane_x` (updated after every sane step) when guards are on, otherwise the torso x.
- Stuck rule with guards: `x_hist[-1] - x_hist[-1 - STUCK_WINDOW] < STUCK_MIN_PROGRESS`. The stillness penalty is off with guards. Without guards both old rules are untouched.

`src/research/config.py`: `RunConfig.guards: bool = False`; deleted from the dict in `sha256()` when false. It is in `RUN_FIELDS`, so it can be set at top level of an experiment JSON.

`src/research/evaluate.py`: `make_env` passes `guards=cfg.guards`. `rollout` measures distance from `env.score_x` and, with guards, adds `out["invalid"]`. `summarize` adds an `invalids` count only when any rollout has the key. `analysis.py` is not yet taught to report it.

`tests/test_guards.py` (new, 13 tests). `tests/test_knees_research.py` and `tests/test_knees_v2_research.py`: the hash-mirror tests also `del d["guards"]` (the hash itself is unchanged).

`experiments/pilot4b_knees_v2_guards.json` (new): `pilot4_knees_v2` plus top-level `"guards": true`, so every variant gets it. `experiments/exp4_knees_v2.json`: `"guards": true` added at top level.

### Thresholds and how they were chosen

Replayed all six pilot champions (seeds 100/101) on test courses 5000-5009, measured over all bodies.

| Constant | Value | Normal-walking max | Basis |
|---|---|---|---|
| `SANE_MAX_SPEED` | 1250 px/s | 407 (straight); knees p99 381 | ~3x |
| `SANE_MAX_OMEGA` | 60 rad/s | 13.0 (straight, Spot before 530); knees p99 19.2 | ~3x the knees p99 |
| `STUCK_WINDOW` / `STUCK_MIN_PROGRESS` | 150 steps / 60 px | Spot shuffle ~50 px per 150 steps (min 34 px per 120) | Spot is cut at step 150 |

The raw plain-knees maxima (1840 px/s, 168 rad/s) are themselves blow-ups caused by the leaking hips, so p99 was used for knees. The stuck values follow from the Spot rate of ~0.34 px/step; a walker only needs 0.4 px/step to pass, and the straight and knees champions cover 3-5 px/step.

### Root cause of the Spot blow-up (standalone replay, course 5000, stop at step 535)

| Variant | Result |
|---|---|
| baseline | blows up (max torso \|w\| 15.3, hips to 924 degrees) |
| (a) `clamp_knees` disabled | **no blow-up** (max \|w\| 0.23, hips 21 degrees); creature just stands (x=52) |
| (b) hips also clamped | **still blows up** (max \|w\| 24.8, hips held to 90 degrees, torso angle 0.98) |
| (c) substeps doubled | worse: blows up and falls by step 418 |

So the hip clamp is not the fix. The knee clamp is what triggers the blow-up here: with the knees pressed on the 14 degree limit, the per-substep position teleport appears to inject energy against the ground contacts. The sanity cutoff is therefore the actual protection for this failure, not just a backstop. I did not test why the knee clamp misbehaves (for example velocity-only correction), so that stays open.

### Checks run

- Full suite: 130 passed, 0 failed (pinned to 4 cores, Idle).
- Old champions replayed with guards on (test courses 5000-5009): Spot ends as stuck at step 150 on all 10 courses (before: 541-542 steps); no invalids for Spot. Straight/knees champions get much shorter episodes with guards (for example knees seed 100 falls at step 75), because they were trained with leaking hips and now cannot use it. This is expected, and is what the retraining comparison is for.

### Still unverified

- No retraining has been done with guards; `pilot4b` has not been run. Whether guards cost or help trained performance is unknown.
- The thresholds come from only 6 champions that were trained without guards. Walkers trained under guards may move differently, so SANE and STUCK values could need revisiting.
- The stuck rule's effect was not isolated from the hip clamp on the old straight/knees champions.
- The root cause of why the knee clamp blows up Spot against the ground is not established.
- `analysis.py` does not report `invalids` yet.

## Step 2c: dashboard showcase

- `scripts/record_replay.py`: when `env.knees` is set, each frame gets 12 extra columns after `vx` (shin x, y, angle per leg), and the replay gets `"knees": true`, `"seg": [10, 20]` and `"shin_fields"`. Only the 4 hip actions are stored (kneed envs have 8). Straight replays are byte-compatible with the old format. `--champion` already builds the env through `make_env(cfg, ...)`, which passes `knees`, `fall_rule` and `guards` from `config.json`.
- `web/templates/index.html`: the 2D view draws thighs and shins at the recorded size (shins in a darker blue and lighter green); the 3D view creates shin meshes on the first kneed frame. Two new buttons, "Knees (v2 pilot)" and "Straight legs (v2 pilot)", plus the note "v2 pilot champions, 2 seeds; see the knees research write-up". Live mode falls back to Session Best for these two.
- New replays from `results/pilot4b_knees_v2_guards/{knees,straight}/seed_101/champion.npz`: `knees.json` (128 KB, 2603 px, 586 steps) and `straight_v2.json` (79 KB, 1345 px, 600 steps).
