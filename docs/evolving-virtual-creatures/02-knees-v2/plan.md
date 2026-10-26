---
title: Knees v2 - Plan
type: feat
date: 2026-10-25
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# Knees v2 - Plan

## Goal Capsule

- **Objective:** Find out whether knees help once the creature is no longer killed for crouching, with numbers good enough for the blog.
- **Means:** a penalty-based fall rule, a fix for knees bending past their limit, a Spot-inspired knee variant, and a 3-variant experiment trained on GitHub Actions (KTD1-KTD6).
- **Stop conditions:** stop if the default straight creature's numbers change (R1). Don't start training until the user approves the walkthrough.
- **Execution profile:** Sonnet 5.5 subagents (low effort) write the code, unit by unit, on branch `feat/knees-v2` (based on `feat/knees`) in the main checkout. Never use worktrees. Opus plans and verifies.

---

## Product Contract

### Problem Frame

In step 1, every kneed champion "fell" in 31-44 steps. A diagnosis of seeds 0, 1 and 4 showed the cause: the episode ended when the **torso touched the ground** while the body was still upright (tilt 0.07-0.78 rad). The knees had bent to **116-128 degrees**, past their 90 degree limit, because the knee motor overpowers the limit joint. Boston Dynamics' Spot sits with its body 191 mm off the floor and uses knees between 14 and 160 degrees. A low body and a bent knee are normal for a real quadruped, but they count as death in our rules.

### Requirements

- R1. All existing defaults are unchanged: the straight creature, the old fall rule and existing run hashes (verified by the fingerprint check from step 1).
- R2. A new opt-in fall rule `penalty`: touching the ground with the torso costs reward on every step that it touches, and no longer ends the episode. Tipping over (tilt > 1.2 rad) and sinking below the course still end it as a fall.
- R3. Knees never bend past their configured limit (tolerance 5 degrees).
- R4. A `spot` knee variant: knee range 14-150 degrees, a standing pose with the knees bent 30 degrees, and a hip range of ±90 degrees.
- R5. Experiment 4 compares `straight`, `knees` and `knees_spot`, all three under the `penalty` rule, with 10 seeds each. Hypotheses are written down before training.
- R6. Training runs on GitHub Actions, with one job per (variant, seed), and the results are downloaded and verified locally.
- R7. `02-knees-v2/` gets `abstract.md`, `results.md` and `screenshots/`, like step 1.

### Key Decisions

- **Fall rule = penalty (option 3), not a grace period or tilt-only.** (session-settled: user-directed, chosen over the grace period and tilt-only: crouching should cost reward, not end the run.) Governs R2.
- **Add a Spot-inspired variant.** (session-settled: user-directed, chosen over Spot ranges replacing the plain knees: three variants keep each change measurable.) Governs R4, R5.
- **Train on GitHub Actions.** (session-settled: user-directed, chosen over the laptop, which overheats.) Governs R6.

### Scope Boundaries

- Settle-before-scoring, contact sensors and size-scaled mutation are not part of this step.
- The dashboard is unchanged.
- A known risk, not a fix: the dive may still pay off under the penalty rule, because the episode ends via "stuck" about 60 steps after the creature stops. If the dive persists, settle-before-scoring is the next lever, and `results.md` will say so.

---

## Planning Contract

### Key Technical Decisions

- KTD1. **`CreatureEnv(fall_rule="touch")` is the default.** `"penalty"` applies −2 reward per step while the torso touches the ground, and `body_touched_ground` no longer sets `fallen`. −2 per step roughly cancels the reward of a decent walker (about 2-3 px/step × 2), so lying on the ground never earns reward. The value is fixed before training.
- KTD2. **Knee limit fix applies to the kneed path only.** Any mechanism the implementer proves with a test is acceptable, for example raising the limit joint's `max_bias`/`error_bias` stiffness, or more solver iterations set only when `knees` is on. Straight runs stay byte-identical (R1).
- KTD3. **`knees` takes `False`, `True` or `"spot"`.** For `"spot"`, the knee limit is 14°-150° in the backward-bend convention, the reset pose bends each knee 30° and lowers the torso so the feet still touch the ground, and the hip limit is ±90°. No new class.
- KTD4. **`RunConfig` gains `fall_rule: str = "touch"`.** Like `knees`, it is left out of `sha256()` when it has its default value, so old hashes are kept. `knees` accepts `"spot"`. The genome size for `"spot"` is 520, the same as for knees.
- KTD5. **The GitHub Actions workflow is `.github/workflows/experiment.yml`,** with manual `workflow_dispatch` and an `experiment` input naming the config file.
  - A first job reads the config and emits a matrix of (variant, seed).
  - Each run job installs the dependencies, runs `scripts/run_experiment.py <cfg> --only <v> --seeds <s> --workers 1`, and uploads `results/<exp>/<v>/seed_<s>/` as an artifact.
  - A final job downloads all the artifacts, rebuilds `results/<exp>/` together with `experiment.json` and the manifest, runs `verify_results.py` and `analyze.py --no-replay`, and uploads the whole folder as a single artifact.
- KTD6. **All three variants run on the same kind of GitHub runner.** Absolute distances are not compared with exp3, which ran on the laptop. Only within-experiment contrasts count.

### Hypotheses (pre-registered in `experiments/HYPOTHESES.md`)

- H8: under the penalty rule, `knees` reaches a greater held-out test distance than `straight`.
- H9: `knees_spot` reaches a greater held-out test distance than `knees`.
- H10: `knees` falls less often than in exp3 (descriptive: exp3's fall rate was 100%).

---

## Implementation Units

### U1. Fall rule `penalty` + knee limit fix + `spot` knees (env)
- **Files:** `src/environment.py`, `tests/test_knees_v2_env.py`
- **Tests:** a default env is unchanged (fingerprint). With `penalty`, a torso pressed to the ground costs −2 per step and doesn't end the episode. Tilt > 1.2 still ends it as a fall. For both kneed variants, 60 steps of full knee action never take the knee more than 5° past its limit. `spot` resets with the knees bent 30° and the feet on the ground, and doesn't fall within 30 zero-action steps.

### U2. Research plumbing
- **Files:** `src/research/config.py`, `src/research/evaluate.py`, `tests/test_knees_v2_research.py`
- **Tests:** the default hash is unchanged. A `fall_rule`/`spot` config gets a distinct hash. A tiny `run_one` with `knees="spot"` and `fall_rule="penalty"` runs and is deterministic.

### U3. Experiment 4 + GitHub Actions workflow
- **Files:** `experiments/exp4_knees_v2.json`, `experiments/HYPOTHESES.md`, `.github/workflows/experiment.yml`
- **Verification:** the config loads. The workflow passes `actionlint` if it's available, otherwise a careful read. The matrix is 3 variants × 10 seeds = 30 jobs.

### U4. Docs
- **Files:** `docs/evolving-virtual-creatures/02-knees-v2/abstract.md`, and the roadmap in `docs/evolving-virtual-creatures/README.md` gets the knees v2 step.

### U5. Train and report (after the user says go)
- Push the branch, dispatch the workflow, download the results artifact into `results/exp4_knees_v2/`, verify, run the analysis with replays locally pinned to 4 cores, write `results.md` and copy the charts to `screenshots/`.

---

## Verification Contract

| Check | Proves |
|---|---|
| `python -m pytest -q` | nothing broke |
| fingerprint script vs `fingerprint_before.txt` | R1 |
| `ruff check src tests` | style |
| `verify_results.py results/exp4_knees_v2` | complete, untampered runs |
