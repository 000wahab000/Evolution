---
title: Research-Grade Experiments and Metric Visualizations - Plan
type: feat
date: 2026-10-20
topic: research-grade-experiments
artifact_contract: ce-unified-plan/v1
artifact_readiness: draft-for-review
product_contract_source: user-request
execution: code
---

# Research-Grade Experiments and Metric Visualizations - Plan

## Goal Capsule

- **Objective:** Turn Evolution-1 from a demo into a reproducible experimental testbed that can answer one narrow question with error bars and show the answer visually.
- **Means:**
  - Seeded configs and per-run output folders.
  - Multi-rollout fitness on randomized terrains, with a held-out test set.
  - Swappable brain-input variants and optimizers, compared on an equal episode budget.
  - A statistics module, a figure suite, and a one-command report.
- **Product Authority:** User request 2026-10-20 ("make this walkable bot a research project level"; "use graphical difference of metrics ... our project NEEDS to benefit SOMEONE"). Research basis: `docs/research/2026-10-20-research-direction.md`.
- **Executor:** Phases P1-P9 are handed to a coding agent (Sonnet) one phase per PR. Running the experiments and writing the paper are not part of this plan.
- **Open Blockers:** The "Decisions for review" section must be answered before P5 and P6.

---

## Who this benefits (the acceptance lens)

Every phase is judged by whether it serves one of these users:

| User | What they need | What delivers it |
|---|---|---|
| **A student / self-taught researcher** (including the author) | A laptop-sized testbed whose results they can trust and rerun | One-command experiments, seeds, saved configs, `REPORT.md` |
| **A researcher reading the paper** | A measured answer to "does a CPG clock help evolved controllers generalize?", with error bars and fair baselines | Held-out terrain eval, equal budgets, stats with CIs, figures |
| **A dashboard visitor** | An honest picture of what changes when you change one thing | Figures showing distributions across seeds, not one cherry-picked walker |

**Definition of done for the whole plan:** someone with no prior context clones the repo and runs:

```
python scripts/run_experiment.py experiments/exp1_cpg.json
python scripts/analyze.py results/exp1_cpg
```

on 8 cores. Within about 4 hours they get `results/exp1_cpg/REPORT.md` containing every figure below, a stats table, and the exact commands and git hash that produced it.

---

## Product Contract

### Problem Frame
Today:
- `main.py --train` produces one run on one fixed terrain, with one deterministic rollout per genome.
- `--seed` seeds only the parent process. That happens to be enough now, because rollouts are deterministic, but it breaks once rollouts become stochastic.
- The "downhill" segment in `CreatureEnv.terrain_changing()` reuses the uphill formula, and there is a 40 px cliff at x=1600.
- There are no baselines, ablations, held-out evaluation or statistics.
- `scripts/plot_logs.py` plots single-run best/avg/worst curves, which cannot show differences between methods.

### Research questions (pre-registered: write them down before running)

**Experiment 1, CPG ablation:**
- H1: CPG-on reaches a higher held-out test distance IQM than CPG-zeros at an equal episode budget.
- H2: the CPG advantage persists or grows under terrain shift (test terrains harder than train terrains).
- H3: the noise control does not match CPG-on. This rules out "any extra input helps."

**Experiment 2, optimizer comparison** (CPG-on brain):
- H4: CMA-ES ≥ GA in held-out distance at an equal episode budget.
- H5: all optimizers beat random search (sanity floor).

These hypotheses go into `experiments/HYPOTHESES.md` in P6. Results are reported whether they confirm or refute them.

### Metrics (what is measured and why)

All metrics are computed on **held-out test terrains** unless marked "train".

| ID | Metric | Definition | Why it matters |
|---|---|---|---|
| M1 | **Test distance** (primary) | mean torso x-displacement over the test terrain set | the actual task; not inflated by reward shaping |
| M2 | Fall rate | fraction of test episodes ending in `fallen` | robustness |
| M3 | Cost of transport | Σ\|action\| ÷ distance (sim units), episodes with distance ≤ 0 excluded | efficiency |
| M4 | Generalization gap | train distance − test distance | overfitting to the training terrain |
| M5 | Sample efficiency | episodes until best-so-far test distance ≥ τ (τ = 300 px, configurable) | cost of learning |
| M6 | Robustness curve | M1 under friction ×{0.6…1.4}, initial-pose noise, and dt/2 | credibility; catches physics exploits |
| M7 | Gait signature | per-leg dominant frequency (FFT of leg angle), phase lag between legs, lag to the CPG clock | *why* the CPG helps (entrainment) |
| M8 | Population diversity | mean per-parameter std across the population (existing `Diversity_Score`) | stagnation diagnosis |

**Aggregation:**
- Per run: final metric on the test set.
- Across seeds: IQM (interquartile mean) with 95% stratified-bootstrap CI (10,000 resamples), plus the median.
- Pairwise tests: two-sided Mann-Whitney U, Holm-corrected within each experiment.
- Effect size: Cliff's delta, and probability of improvement P(X > Y).

### Figures (the graphical difference of metrics)

All figures use one colorblind-safe palette with a fixed color per variant across every figure. Each is saved as PNG and SVG and gets a one-line caption stating what to look for. Variants are always on the same axes so differences are visible at a glance.

| Fig | Shows | Metric | Visual form | How to read it |
|---|---|---|---|---|
| F1 | **Learning curves** | best-so-far M1 vs **episodes** | IQM line + 95% CI band per variant | higher and earlier is better; overlapping bands mean no clear difference |
| F2 | **Final performance distribution** | M1 per seed | one dot per seed + IQM marker + CI bar, train and test side by side | shows spread and outliers, not just the mean |
| F3 | **Generalization gap** | train M1 vs test M1 per seed | scatter with a y=x diagonal | points far below the diagonal are overfitting |
| F4 | **Performance profile** | fraction of runs with M1 > τ, swept over τ | step curves per variant | a curve that sits above another dominates at every threshold |
| F5 | **Pairwise effect matrix** | Cliff's delta and P(improvement) | annotated heatmap; stars mark Holm-significant pairs | answers "is A really better than B?" in one picture |
| F6 | **Sample efficiency** | M5 | ECDF of episodes-to-threshold, unreached runs shown as censored | left is cheaper |
| F7 | **Robustness** | M6 | M1 vs perturbation strength, IQM + CI per variant | flatter curves are more robust; a drop at dt/2 flags physics exploits |
| F8 | **Efficiency trade-off** | M3 vs M1 per seed | scatter per variant | lower-right is fast and cheap |
| F9 | **Gait diagram** | M7 for the median-seed champion of each variant | footfall / leg-phase raster over time + leg-frequency vs CPG-frequency histogram | does the gait lock to the CPG clock? |
| F10 | **Fall rate** | M2 | bar per variant with Wilson CI | |

`REPORT.md` embeds F1-F10, a summary table (variant × M1-M5, IQM [CI]), the pairwise stats table, H1-H5 with a verdict of supported / not supported / inconclusive, and a reproducibility block (git hash, config, seeds, platform, package versions, wall time).

### Requirements
- **R1 Reproducibility.** Every run is fully determined by (config, seed). Re-running a seed gives bit-identical history on the same machine. Each run folder stores the resolved config, the git commit hash plus a dirty flag, Python/numpy/pymunk versions, platform and wall time.
- **R2 Terrain.**
  - The downhill bug is fixed.
  - Terrain is a seeded generator of piecewise-linear courses.
  - Train and test terrain sets are disjoint seed ranges. A "shifted" test set uses steeper slopes than training allows.
  - A `legacy` terrain option reproduces the current course exactly, so existing replays and `universal_best` still work.
- **R3 Evaluation.**
  - Fitness is the mean over k rollouts (default k=5), each on a sampled train terrain with seeded perturbations (friction ×U[0.9,1.1], initial joint-angle noise σ=0.05 rad).
  - Every rollout returns M1-M3 raw values.
  - An episode counter is the single source of truth for budget.
- **R4 Brain-input variants** (input size stays 15 so parameter counts match):
  - `cpg` (current)
  - `zeros` (CPG inputs fixed at 0)
  - `noise` (CPG inputs replaced by seeded uniform noise in [-1,1])
  - `shuffled` (a random fixed phase permutation of the 30-step table)
- **R5 Optimizers.** One ask/tell interface (`ask() -> list[genome]`, `tell(fitnesses)`), with all budgets counted in episodes:
  - `ga`: the current algorithm, behavior unchanged.
  - `cmaes`: via the `cma` package.
  - `openai_es`: antithetic sampling, rank-shaped fitness, Adam. Small NumPy implementation; each antithetic pair counts as 2 genomes.
  - `random_search`: Mania et al. 2018 basic random search (V1).
- **R6 Runner.**
  - Runs variants × seeds in parallel *across runs* (one process per run).
  - Resumable: skips finished seeds and atomically writes per-run outputs.
  - Logs per-generation history and a test-set evaluation every N episodes (for F1/F6).
- **R7 Stats.** `src/research/stats.py`: IQM, stratified bootstrap CI, Mann-Whitney + Holm, Cliff's delta, P(improvement), Wilson CI. Each function has known-answer tests.
- **R8 Figures and report.** F1-F10 and `REPORT.md` generated by one command from the results folder alone, with no re-simulation except F7 and F9, which replay the saved champions.
- **R9 Credibility checks.** dt/2 replay of champions (part of F7), a determinism test, and a tiny smoke config (`experiments/smoke.json`, under 2 minutes) used by the tests.
- **R10 No regressions.** `main.py --train`, `--render`, `app.py`, `scripts/record_replay.py` and all existing tests keep working, with `legacy` terrain and k=1 as their defaults.

### Scope Boundaries
- No change to the default physics timestep, the brain architecture, or the GA's hyperparameters.
- No JAX/Brax/MuJoCo port. A benchmark replication is future work.
- No morphology evolution (v4).
- No paper writing.
- Dependencies: `scipy` and `cma` go in a new optional `research` extra; `matplotlib` is already in `viz`. The server image stays unchanged.

---

## Implementation Units

Each phase is one PR. A phase is not done until its tests pass and `pytest` is green overall.

### P1. Config, seeding, run folders
- **Files:** new `src/research/__init__.py`, `src/research/config.py`, `src/research/runlog.py`, `tests/test_research_config.py`.
- **Do:**
  - `ExperimentConfig` dataclass, loaded from JSON (no YAML dependency), with fields:
    - `name`, `seeds`, `variants` (each a dict of overrides)
    - `optimizer`, `optimizer_params`, `brain_input`
    - `episodes_budget`, `rollouts_k`, `terrain`, `perturbations`
    - `eval_every_episodes`, `test_terrain_seeds`, `shift_test_terrain_seeds`
  - `resolve(variant, seed)` returns a frozen RunConfig.
  - `make_rng(run_seed, *keys)` returns a `numpy.random.Generator` built from `SeedSequence([run_seed, *keys])`. **No global `np.random` / `random` calls in any new code.**
  - `runlog.write_meta(run_dir, cfg)` writes the git hash (`git rev-parse HEAD`, dirty flag), versions, platform and start/end time.
- **Tests:**
  - JSON round-trip.
  - The same (config, seed) gives the same RNG stream.
  - Different keys give different streams.

### P2. Terrain module and bug fix
- **Files:** new `src/terrain.py`; edit `src/environment.py` (`CreatureEnv.__init__(…, terrain=None)`, replacing `terrain_changing()` with `add_terrain(space, spec)`); `tests/test_terrain.py`.
- **Do:**
  - `TerrainSpec` holds a list of (x, y) points plus friction.
  - `legacy_course()` reproduces today's segments **exactly**, bug included, for backward compatibility.
  - `fixed_course()` is the corrected course: flat to 600, uphill to 1600, **actual downhill** back to y=500 at 2600, flat to 3000, no cliff.
  - `random_course(rng, max_slope, length, n_segments)` builds a continuous piecewise-linear course with a flat 200 px spawn zone.
  - `CreatureEnv` defaults to `legacy_course()` (R10).
  - Fix the fallen check: it uses absolute y > 520 and a torso-bottom ≥ 495 test, which assume flat ground at y=500. Compute ground height under the torso from the spec instead. **Behavior change only for non-legacy terrains;** legacy keeps the old check.
- **Tests:**
  - The legacy course's segment endpoints and friction equal today's `terrain_changing()` geometry exactly.
  - A rollout of `data/models/universal_best.npz` gives the same distance through the old and new env code paths in the same process. Don't compare against the committed replay JSON: the README notes 1e-6 forward-pass differences across machines change the walk.
  - `fixed_course` is continuous with no vertical gap larger than 1 px.
  - `random_course` is deterministic for a seed and respects `max_slope`.

### P3. Multi-rollout evaluation and metrics
- **Files:** new `src/research/evaluate.py`; `tests/test_research_evaluate.py`.
- **Do:**
  - `rollout(genome_params, brain_input, terrain_spec, perturb, rng, max_steps=600) -> dict` returning `distance, fell, steps, energy, cot, return`, plus an optional trajectory (leg angles, CPG phase) for F9.
  - `evaluate_genome(params, run_cfg, rng, split="train"|"test"|"shift")` samples k terrains and perturbations and returns the mean fitness plus per-rollout metrics.
  - The fitness used for selection is the existing reward (`displacement*2`, stillness penalty, -100 on fall) so the GA stays comparable with today's. Selection uses the train split only.
  - `EpisodeCounter`: every rollout increments it, and the optimizer loop stops at `episodes_budget`.
- **Tests:**
  - k=1 on the legacy course with no perturbation equals the current `evolution.evaluate_genome` reward.
  - Same rng gives the same result.
  - The counter equals pop × k × generations.

### P4. Brain-input variants
- **Files:** edit `src/environment.py` (`get_state` takes a `cpg_mode` set at construction: `cpg|zeros|noise|shuffled`, with `noise` drawing from an env-owned rng); `tests/test_environment.py`.
- **Do:** state shape stays (15,). `cpg` stays the default and bit-identical to today.
- **Tests:**
  - Each mode gives the expected values in indices 13-14.
  - `cpg` mode is unchanged from the current output.

### P5. Optimizers behind ask/tell
- **Files:** new `src/research/optimizers.py`; `tests/test_optimizers.py`. Flat 324-vector to/from `CreatureBrain` helpers live here.
- **Do:**
  - `GAOptimizer` wraps the logic of `Population.evolve` using an injected rng. With equal seeding it gives the same generation-to-generation behavior as `Population`.
  - `CMAESOptimizer` uses `cma.CMAEvolutionStrategy(x0=0, sigma0, {'seed': …, 'popsize': …})`. Fitness is negated, because `cma` minimizes.
  - `OpenAIESOptimizer` and `RandomSearchOptimizer` as in R5.
  - Shared `make_optimizer(name, dim, params, rng)`.
- **Tests:**
  - Each optimizer improves on a 10-dim sphere function within 2,000 evaluations.
  - Each counts evaluations correctly (antithetic pairs = 2).
  - Each is deterministic per seed.

### P6. Experiment runner and configs
- **Files:**
  - `scripts/run_experiment.py`
  - `experiments/exp1_cpg.json` (variants: cpg, zeros, noise, shuffled; optimizer `ga`)
  - `experiments/exp2_optimizers.json` (ga, cmaes, openai_es, random_search; `cpg`)
  - `experiments/smoke.json` (2 variants × 2 seeds × 2,000 episodes)
  - `experiments/HYPOTHESES.md` (H1-H5, verbatim from this plan)
  - `tests/test_runner_smoke.py`
- **Do:**
  - `run_experiment.py CONFIG [--workers N] [--only variant] [--seeds 0-4]`.
  - Output: `results/<name>/<variant>/seed_<n>/{config.json, meta.json, history.csv, evals.csv, champion.npz, final_test.json, DONE}`.
    - `history.csv` columns: episodes, generation, best_train, mean_train, diversity.
    - `evals.csv` columns: episodes, then M1-M3 on the test and shift sets for the current champion.
  - Skip seeds that already have `DONE`. Atomic writes via `src/utils.atomic_save` patterns.
  - Add `results/` to `.gitignore`, except committed `REPORT.md` + figures per finished experiment.
- **Defaults** (subject to the review decisions): 20 seeds, 25,000 episodes per run (pop 50 × k 5 × 100 gens for the GA), eval every 1,250 episodes on 10 test terrains + 10 shift terrains. Estimate: ~20 core-min per run; 80 runs per experiment ≈ 27 core-h ≈ 3.5 h on 8 cores.
- **Tests:** the smoke config runs end to end and is resumable (a second invocation is a no-op).

### P7. Statistics module
- **Files:** new `src/research/stats.py`; `tests/test_stats.py`.
- **Do:** `iqm`, `bootstrap_ci(values, stat=iqm, n=10000, rng)`, `mann_whitney(a, b)` (scipy), `holm(pvalues)`, `cliffs_delta(a, b)`, `prob_improvement(a, b)`, `wilson_ci(k, n)`.
- **Tests:** known answers. Cliff's delta is ±1 for fully separated samples. Holm on a textbook example. IQM of 1..8 is 4.5. The bootstrap CI of a constant is that constant. Wilson matches a reference value.

### P8. Analysis, figures, report
- **Files:** `scripts/analyze.py`, `src/research/figures.py`, `src/research/report.py`; `tests/test_analyze_smoke.py`.
- **Do:**
  - `analyze.py results/<name>` writes `summary.json`, `summary.csv`, `pairwise.csv`, then F1-F10 as `figures/F*.png|svg`, then `REPORT.md`.
  - F7 and F9 replay saved champions (F7 adds friction sweep, pose noise and dt/2).
  - The palette is defined once in `figures.py`. Every figure has a title, axis labels with units, a legend and a caption string.
  - Captions and H-verdicts come from the stats (e.g. H1 is "supported" only if the Holm-corrected p < 0.05 **and** the CI of the IQM difference excludes 0).
- **Tests:** running on the smoke results produces all 10 PNGs and a REPORT.md containing every figure link and the reproducibility block.

### P9. Credibility checks
- **Files:** `tests/test_determinism.py`; F7 already covers dt/2.
- **Do:**
  - The same seed run twice gives identical `history.csv`.
  - A champion's legacy-course distance at dt and dt/2 is recorded in `final_test.json`.
  - REPORT.md flags any variant whose IQM drops more than 30% at dt/2 ("possible physics exploit").

### P10 (optional, after review). Dashboard "Experiments" view
- **Files:** `app.py` (static route for `results/*/REPORT.md` assets), `web/templates/index.html`.
- **Do:**
  - A tab listing finished experiments that shows F1, F2 and F5 plus the summary table.
  - Replay buttons for each variant's median-seed champion, so visitors can *see* the walking difference behind the numbers.
  - Static files only, with no simulation on the server, consistent with the deploy constraints.

---

## Execution Order and Handoff Notes (for the coding agent)

1. P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9, one PR each. P7 depends only on P1, so it can run in parallel with P2-P6.
2. **Do not change** the default physics, reward, GA hyperparameters, or `CreatureEnv` defaults (R10). New behavior is opt-in through config.
3. New code uses injected `numpy.random.Generator`s only. No global RNG.
4. Match the repo's style: small modules, docstrings only where non-obvious, NumPy over loops in hot paths, no heavy frameworks (`docs/STRATEGY.md` boundaries).
5. After P6, run `experiments/smoke.json` and attach the generated REPORT.md to the PR for a visual check.
6. Running the full experiments is a separate, human-approved step after P9.

## Decisions for review (please correct)

1. **Primary metric:** held-out test distance (M1) rather than training fitness. *Recommended.*
2. **Seeds per variant:** 20 (≈3.5 h per experiment on 8 cores) or 30 (≈5 h).
3. **Episode budget per run:** 25,000 (≈100 GA generations at k=5). More budget makes CMA-ES's advantage show more clearly; less is faster.
4. **Hyperparameter fairness:** (a) every optimizer at library or paper defaults, with the GA at its current values (*recommended*, simpler and honest), or (b) an equal small tuning budget per optimizer on train terrains only.
5. **Terrain shift:** train max slope 4% (similar to today's hill), test 4%, shift test 8%.
6. **Optional `research` extra:** add `scipy` and `cma` (both pure-Python/NumPy-friendly, about 40 MB, not in the server image). OK?
7. **P10 dashboard view:** in scope now, or after the first experiment finishes?
8. **Evolved CPG period** (a gene for the clock period, from the original idea A): left out to keep parameter counts equal across variants. Add it later as an exp1b?
