---
title: Research-Grade Experiments and Metric Visualizations - Plan
type: feat
date: 2026-10-20
topic: research-grade-experiments
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
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
- **Executor:** Phases P1-P10 are handed to a coding agent (Sonnet) one phase per PR. Running the experiments and writing the paper are not part of this plan.
- **Open Blockers:** None. All decisions answered 2026-10-20 (see Decisions).

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

on a 4-core/8-thread laptop. Within about 4-6 hours (overnight) they get `results/exp1_cpg/REPORT.md` containing every figure below, a stats table, and the exact commands and git hash that produced it.

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
- H2: the CPG advantage persists or grows under terrain shift (test slopes randomly 5-8%, steeper than the 4% used in training). Measured per seed as shift drop = control M1 − test M1; supported if cpg's shift drop is lower than zeros'.
- H3: CPG-on outperforms the noise control, so the CPG benefit is not explained by uninformative extra input. (Noise vs zeros is reported as a descriptive diagnostic, not a hypothesis.)

**Experiment 2, optimizer comparison** (CPG-on brain):
- H4: CMA-ES > GA in held-out distance at an equal episode budget. A non-significant result is reported as inconclusive.
- H5: all optimizers beat random search (sanity floor).

These hypotheses go into `experiments/HYPOTHESES.md` in P6. Results are reported whether they confirm or refute them.

### Metrics (what is measured and why)

All metrics are computed on **held-out test terrains** unless marked "train".

| ID | Metric | Definition | Why it matters |
|---|---|---|---|
| M1 | **Test distance** (primary) | mean torso x-displacement over the test terrain set | the actual task; not inflated by reward shaping |
| M2 | Fall rate | fraction of test episodes ending in `fallen` | robustness |
| M3 | Cost of transport | Σ\|action\| ÷ distance (sim units), episodes with distance ≤ 0 excluded | efficiency |
| M4 | Generalization gap | two values: gap_control = train − control distance (memorization); gap_test = train − test distance (memorization + shift) | overfitting to the training terrain |
| M5 | Sample efficiency | episodes at the first eval where the train-selected champion's test distance ≥ τ (τ = 300 px, configurable) | cost of learning |
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

- **Tokens:** the palette is named hex tokens (one per variant, plus neutral text/grid/CI-band tokens) in a light set (REPORT.md, paper) and a dark set matching the dashboard (`--bg #09090b`, `--card-bg #121214`). Each variant colour meets 3:1 contrast against both white and `#121214`. `figures.py` writes `F*.svg` (light) and `F*.dark.svg`.
- **Not colour alone:** each variant also gets a fixed marker shape and line dash, used in every figure and exported in the dashboard bundle so Chart.js uses the same encoding.

| Fig | Shows | Metric | Visual form | How to read it |
|---|---|---|---|---|
| F1 | **Learning curves** | M1 of the train-selected champion vs **episodes** | IQM line + 95% CI band per variant | higher and earlier is better; overlapping bands mean no clear difference |
| F2 | **Final performance distribution** | M1 per seed | one dot per seed + IQM marker + CI bar, train and test side by side | shows spread and outliers, not just the mean |
| F3 | **Generalization gap** | train M1 vs test M1 per seed (control M1 as a second marker) | scatter with a y=x diagonal | points far below the diagonal are overfitting |
| F4 | **Performance profile** | fraction of runs with M1 > τ, swept over τ | step curves per variant | a curve that sits above another dominates at every threshold |
| F5 | **Pairwise effect matrix** | Cliff's delta and P(improvement) | annotated heatmap; stars mark Holm-significant pairs | answers "is A really better than B?" in one picture |
| F6 | **Sample efficiency** | M5 | ECDF of episodes-to-threshold, unreached runs shown as censored | left is cheaper |
| F7 | **Robustness** | M6 | M1 vs perturbation strength, IQM + CI per variant | flatter curves are more robust; a drop at dt/2 flags physics exploits |
| F8 | **Efficiency trade-off** | M3 vs M1 per seed | scatter per variant | lower-right is fast and cheap |
| F9 | **Gait diagram** | M7 for the median-seed champion of each variant | footfall / leg-phase raster over time + leg-frequency vs CPG-frequency histogram | does the gait lock to the CPG clock? |
| F10 | **Fall rate** | M2 | bar per variant with Wilson CI | lower is better; non-overlapping Wilson bars mean a clear difference |

`REPORT.md` embeds F1-F10, a summary table (variant × M1-M5, IQM [CI]), the pairwise stats table, H1-H5 with a verdict of supported / not supported / inconclusive, and a reproducibility block (git hash, config, seeds, platform, package versions, wall time).

### Requirements
- **R1 Reproducibility.** Every run is fully determined by (config, seed). Re-running a seed gives bit-identical history on the same machine. Each run folder stores the resolved config, the git commit hash plus a dirty flag, Python/numpy/pymunk versions, platform and wall time.
- **R2 Terrain.**
  - The downhill bug is fixed.
  - Terrain is a seeded generator of piecewise-linear courses.
  - Train and test terrain sets are disjoint seed ranges.
  - Train terrains: max slope 4%.
  - **Test terrains (main, M1): each segment's slope drawn at random from 5-8%,** harder than anything seen in training.
  - Control set: unseen terrains at the training slope (≤4%). Used for M4 and H2, to separate memorizing one course from failing on harder ground.
  - A `legacy` terrain option reproduces the current course exactly, so existing replays and `universal_best` still work.
- **R3 Evaluation.**
  - Fitness is the mean over k rollouts (default k=5), each on a sampled train terrain with seeded perturbations (friction ×U[0.9,1.1], initial joint-angle noise σ=0.05 rad).
  - Every rollout returns M1-M3 raw values.
  - An episode counter is the single source of truth for budget.
  - **Champion rule (test set stays held out):** the champion at any point is the genome with the best mean train-split fitness, re-evaluated on a fixed set of train-distribution validation terrains disjoint from test and control. `evals.csv` logs that champion's test/control M1-M3 with no max over evaluations, and `final_test.json` evaluates the final champion. Test and control terrains are never used for selection.
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

### Results storage (researched 2026-10-20)
- **Decision:** plain files, one folder per run. No SQLite, Parquet or tracker (MLflow, W&B and Aim all need servers or heavy dependencies and can't feed a static dashboard).
- **Why it fits:**
  - Each run writes only its own folder, so parallel processes never fight over a shared file.
  - Folders diff cleanly in git, need nothing but the standard library and numpy, and are easy to read in analysis.
  - Estimated total: about 30-40 KB per run, about 10 MB for all 240 runs.
- **Layout:**
  ```
  results/<exp>/
    manifest.json      schema_version, experiment name, config sha256, git commit+dirty,
                       variant x seed status table (written by the parent runner only)
    experiment.json    resolved ExperimentConfig + hypotheses id
    <variant>/seed_<n>/
      config.json      resolved RunConfig + schema_version
      meta.json        run card (see below)
      history.csv      episodes, generation, best_train, mean_train, diversity
      evals.csv        episodes, M1-M3 on test (5-8%) and control (<=4%) sets
      champion.npz     w1, b1, w2, b2 as float32 (same as CreatureBrain), load with allow_pickle=False
      final_test.json  runner-time values only: final M1-M3, M8, and test-set distance at dt and dt/2
                       (M6 sweeps and M7 gait are computed by analyze.py from champion.npz)
      DONE             JSON {status, sha256 of every file above}, written LAST
    summary/           summary.json, summary.csv, pairwise.csv, figures/F*.png|svg, REPORT.md
  ```
- **Run card (`meta.json`):** schema_version, run_id (`exp/variant/seed`), seed, git_commit, git_dirty, python/numpy/pymunk/cma versions, platform, cpu, started_utc, ended_utc, wall_s, episodes_used, config_sha256, terrain_version.
- **Versioning:** `schema_version: 1` everywhere. Readers fail loudly on an unknown version, and old runs are never rewritten.
- **Crash safety on Windows:**
  - Write to a temp file, then `os.replace`.
  - Retry `os.replace` up to 5× with backoff on `PermissionError` (antivirus file locks).
  - `DONE` is written last; a folder without `DONE` counts as unfinished and is rerun.
- **What goes in git:**
  - `results/` is gitignored by default.
  - Each finished experiment is force-added once (about 10 MB, including its summary and figures), after `scripts/verify_results.py` passes.
  - Never re-commit raw results on every rerun. Tag the commit instead.
- **Archiving (at paper time, not part of this plan):** a Zenodo upload for a DOI (data CC-BY-4.0, code MIT), plus `CITATION.cff`.
- **Reproducibility statement:** bit-identical only on the same machine. Physics is chaotic across CPUs and BLAS builds, as the README already notes.

---

## Implementation Units

Each phase is one PR. A phase is not done until its tests pass and `pytest` is green overall.

### P1. Config, seeding, run folders
- **Files:** new `src/research/__init__.py`, `src/research/config.py`, `src/research/runlog.py`, `tests/test_research_config.py`; edit `pyproject.toml`.
- **Do:**
  - Add the optional dependency group `research = ["scipy>=1.11", "cma>=3.3", "matplotlib>=3.7.0"]` to `pyproject.toml`. Do **not** add these to `requirements.txt` or the Dockerfile; the server image stays unchanged. Document `pip install -e .[research]` in the README.
  - `ExperimentConfig` dataclass, loaded from JSON (no YAML dependency), with fields:
    - `name`, `seeds`, `variants` (each a dict of overrides)
    - `optimizer`, `optimizer_params`, `brain_input`
    - `episodes_budget`, `rollouts_k`, `terrain`, `perturbations`
    - `eval_every_episodes`, `test_terrain_seeds` (5-8% slopes), `control_terrain_seeds` (≤4% slopes)
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
  - `TerrainSpec` holds an ordered list of segments, each `(a, b, radius, friction)`. Order matters: shapes are added to the space in list order.
  - `legacy_course()` reproduces today's ground **exactly**, bug included, for backward compatibility: the flat base `(-10000,500)-(3000,500)` (radius 5, friction 1.2) from `create_ground()`, then the two sloped segments (radius 4, friction 1.8) from `terrain_changing()`, in today's add order.
  - `CreatureEnv` builds **all** ground from the spec. For non-legacy specs there is no implicit flat y=500 plane (it would clip downhill segments).
  - `fixed_course()` is the corrected course: flat to 600, uphill to 1600, **actual downhill** back to y=500 at 2600, flat to 3000, no cliff.
  - `random_course(rng, max_slope, length, n_segments)` builds a continuous piecewise-linear course with a flat 200 px spawn zone.
  - `CreatureEnv` defaults to `legacy_course()` (R10).
  - Fix the fallen check: it uses absolute y > 520 and a torso-bottom ≥ 495 test, which assume flat ground at y=500. Compute ground height under the torso from the spec instead. **Behavior change only for non-legacy terrains;** legacy keeps the old check.
- **Tests:**
  - The legacy course's segment endpoints and friction equal today's `terrain_changing()` geometry exactly.
  - A rollout of `data/models/universal_best.npz` gives the same distance on `CreatureEnv()` (default legacy spec) and on an env built by a reference builder copied verbatim into the test from today's `create_ground()` + `terrain_changing()`, in the same process. Don't compare against the committed replay JSON: the README notes 1e-6 forward-pass differences across machines change the walk.
  - `fixed_course` is continuous with no vertical gap larger than 1 px.
  - `random_course` is deterministic for a seed and respects `max_slope`.

### P3. Multi-rollout evaluation and metrics
- **Files:** new `src/research/evaluate.py`; `tests/test_research_evaluate.py`.
- **Do:**
  - `rollout(genome_params, brain_input, terrain_spec, perturb, rng, max_steps=600) -> dict` returning `distance, fell, steps, energy, cot, return`, plus an optional trajectory (leg angles, CPG phase) for F9.
  - `evaluate_genome(params, run_cfg, rng, split="train"|"test"|"control")` samples k terrains and perturbations and returns the mean fitness plus per-rollout metrics.
  - The fitness used for selection is the existing reward (`displacement*2`, stillness penalty, -100 on fall) so the GA stays comparable with today's. Selection uses the train split only.
  - `EpisodeCounter`: every **train-split** rollout increments it, and the optimizer loop stops at `episodes_budget`. Test/control evaluation rollouts go to a separate eval counter (recorded in `meta.json`) and never count toward the budget or the `episodes` column.
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
  - `GAOptimizer` reimplements `Population.evolve`'s operators with identical hyperparameters (5 elites, 15% mutate-only branch, 0.5 uniform crossover mask, mutation_rate 0.15, scale 0.35), drawing only from the injected rng, with genomes initialized from the rng like `CreatureBrain`. Bit-equality with `Population` is not required (it uses the global `random`/`np.random`). `Population` stays untouched for `main.py`/`app.py`.
  - `CMAESOptimizer` uses `cma.CMAEvolutionStrategy(x0, sigma0, {'seed': …, 'popsize': …})` with x0 = a seeded `CreatureBrain`-style init (same as the GA) and sigma0 = 0.35 (the GA's mutation scale; `cma` has no default). Fitness is negated, because `cma` minimizes.
  - All optimizers use `rollouts_k = 5`. Exact OpenAI-ES values (population, sigma, Adam lr) and ARS V1 values (directions N, step size, noise) are pinned in `experiments/exp2_optimizers.json` with the cited paper table, and copied into `HYPOTHESES.md` before any full run.
  - `OpenAIESOptimizer` and `RandomSearchOptimizer` as in R5.
  - Shared `make_optimizer(name, dim, params, rng)`.
- **Tests:**
  - Each optimizer improves on a 10-dim sphere function within 2,000 evaluations.
  - Each counts evaluations correctly (antithetic pairs = 2).
  - Each is deterministic per seed.
  - `GAOptimizer` produces the expected elite and offspring counts per generation.

### P6. Experiment runner and configs
- **Files:**
  - `scripts/run_experiment.py`
  - `experiments/exp1_cpg.json` (variants: cpg, zeros, noise, shuffled; optimizer `ga`)
  - `experiments/exp2_optimizers.json` (ga, cmaes, openai_es, random_search; `cpg`)
  - `experiments/smoke.json` (2 variants × 2 seeds × 2,000 episodes)
  - `experiments/HYPOTHESES.md`: H1-H5 verbatim, plus the confirmatory contrasts (H1 cpg vs zeros; H2 shift drop cpg vs zeros; H3 cpg vs noise; H4 cmaes vs ga; H5 each optimizer vs random_search), Holm applied only within each experiment's confirmatory set at alpha 0.05, the verdict rule, the champion rule, pinned optimizer settings, and seed exclusions (none). All other pairs are exploratory. Commit and tag it before the first full run; REPORT.md records the tag.
  - `tests/test_runner_smoke.py`
- **Do:**
  - `run_experiment.py CONFIG [--workers N] [--only variant] [--seeds 0-4]`.
  - Output: `results/<name>/<variant>/seed_<n>/{config.json, meta.json, history.csv, evals.csv, champion.npz, final_test.json, DONE}`.
    - `history.csv` columns: episodes, generation, best_train, mean_train, diversity.
    - `evals.csv` columns: episodes, then M1-M3 on the test (5-8%) and control (≤4%) sets for the current champion.
  - Skip seeds that already have `DONE`. Atomic writes via `src/utils.atomic_save` patterns.
  - Write the layout, run card, `manifest.json` and checksummed `DONE` exactly as in **Results storage** above. Only the parent process writes `manifest.json`.
  - Add `results/` to `.gitignore`, so half-finished runs can never be committed by accident.
  - Add `scripts/verify_results.py <exp>`. It checks that every planned run has `DONE` with matching checksums and that `manifest.json` is complete. Only then does it print the `git add -f results/<exp>` command to publish that experiment. It never runs git itself.
- **Defaults (decided 2026-10-20):** 30 seeds, 25,000 episodes per run (pop 50 × k 5 × 100 gens for the GA), eval every 1,250 episodes on 10 test terrains + 5 control terrains.
- **Estimate** (measured 0.026 s per 600-step episode on the i7-7700HQ): ~12 core-min per run. 120 runs per experiment ≈ 24 core-h. On 4 physical cores (hyperthreads add ~30%) that's ≈ 5 h. Run overnight with `--workers 6` so the laptop stays usable.
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
  - **dt/2 definition:** each brain step runs two physics substeps of `1.5/60` with the action held. The CPG index, stillness/stuck thresholds and the 600-step cap keep counting brain steps, so sim duration and clock period match the dt run and only the integrator changes. Default dt and legacy behavior stay bit-identical.
  - The same seed run twice gives identical `history.csv`.
  - Check that the runner (P6) recorded each champion's test-set distance at dt and dt/2 in `final_test.json`.
  - REPORT.md flags any variant whose IQM drops more than 30% at dt/2 ("possible physics exploit").

### P10. Dashboard "Experiments" view (in scope)
- **Files:** add `scripts/export_dashboard.py`; edit `scripts/record_replay.py` (`--champion PATH --out PATH`), `web/templates/index.html`. `app.py` needs no change if the bundle lives under `web/static/`.
- **Data flow:**
  1. `analyze.py` writes `summary.json` and figures. These are the source of truth for both the paper and the site.
  2. `export_dashboard.py results/<exp>` writes `web/static/experiments/<exp>/`:
     - `index.json`: summary table, verdicts, colors, captions.
     - `curves.json`: F1 IQM and CI on the eval grid, downsampled.
     - `runs.json`: per-seed final M1 for F2.
     - Copies of the F3, F4, F6-F10 dark SVGs.
     - `web/static/experiments/index.json` lists published experiments for the picker.
     - Budget: under 300 KB per experiment.
  3. `record_replay.py` writes `web/static/replays/<exp>/<variant>_median.json` for each variant's median-seed champion. With `--champion PATH` it reads the sibling `config.json` and builds the env with that run's brain input and the first test terrain, recording that terrain.
  4. The new Experiments tab fetches `index.json`, then:
     - Draws F1 (CI bands) and F2 (per-seed dots) with the **already-loaded Chart.js 4.4.0**. No new library; Plotly (~1.1 MB) is ruled out.
     - Renders F5 as a colored HTML table with aria-labels.
     - Shows the other figures as lazy-loaded SVGs.
     - Keeps every caption and the summary table as a text alternative.
  5. A "Watch the difference" button per variant plays its median champion's replay in the existing 3D viewer. `loadReplay` accepts a path under `/static/replays/`; a click scrolls to and focuses the viewport, switches off live mode, and sets the source label to `<exp> / <variant> / median seed N`. A button is disabled (with a title) when its replay is absent from `index.json`.
- **States:** loading text, an empty state ("No experiments published yet; run scripts/export_dashboard.py"), and a fetch-failure message with retry, announced in an aria-live region in the replay loader's wording style. If a figure fails, show its caption and table alone.
- **Layout order:** experiment picker → H1-H5 verdicts → F1, F2 and the Watch buttons → summary table → F5 table → other figures in collapsible sections that lazy-load on open. Below 600 px, tables scroll inside their own container, not the page; SVGs scale to container width.
- **Constraints:** static files only, no server simulation. Watch Render's free-tier bandwidth (5 GB/month per our deploy notes; confirm on render.com/pricing). Keep a dashboard visit under 1 MB.
- **Tests:** `export_dashboard.py` on the smoke results produces a valid bundle under 300 KB. Add a Flask test that the static bundle is served.

---

## Execution Order and Handoff Notes (for the coding agent)

1. P1 → P2 → P3 → P4 → P5 → P6 → P7 → P8 → P9 → P10, one PR each. P7 depends only on P1, so it can run in parallel with P2-P6. P10 depends on P8.
2. **Do not change** the default physics, reward, GA hyperparameters, or `CreatureEnv` defaults (R10). New behavior is opt-in through config.
3. New code uses injected `numpy.random.Generator`s only. No global RNG.
4. Match the repo's style: small modules, docstrings only where non-obvious, NumPy over loops in hot paths, no heavy frameworks (`docs/STRATEGY.md` boundaries).
5. After P6, run `experiments/smoke.json` and attach the run-folder listing and a `history.csv` to the PR. After P8, run `scripts/analyze.py` on the smoke results and attach the generated REPORT.md for a visual check.
6. Running the full experiments is a separate, human-approved step after P10.

## Decisions (answered 2026-10-20)

1. **Primary metric:** distance on held-out test terrains (M1). Confirmed.
2. **Seeds per variant:** 30.
3. **Episode budget per run:** 25,000. Measured feasible on the user's laptop (≈5 h per experiment overnight).
4. **Hyperparameter fairness:** option (a), confirmed. Every optimizer runs at its published or library default (`cma` package defaults for CMA-ES; Salimans et al. 2017 for OpenAI-ES; Mania et al. 2018 V1 for random search); the GA keeps its current values. No per-method tuning. The paper states this.
5. **Terrain:** train ≤4%; test slopes random 5-8% (user decision); small ≤4% control set kept for the generalization-gap metric.
6. **`research` optional dependency group:** confirmed. scipy (statistics), cma (CMA-ES baseline) and matplotlib; laptop-only, added in P1.
7. **P10 dashboard view:** in scope. Storage decided: plain files plus a static dashboard bundle (see Results storage).
8. **Evolved CPG period:** dropped. Basic variants first.
