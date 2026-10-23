# Handoff: Evolution-1 (2026-10-22)

Paste this into a new Claude Code session opened on `D:\wahab stuff\wahab code\Evolution-1`.

## Who I am / how to work with me
- Student, learning as I go. **Go slow: one step at a time, plain English, explain new terms.** The last session went too fast for me.
- Follow my global CLAUDE.md (plan before multi-file work, wait for my OK, end with What changed / Concepts / Judge questions).

## The project
Evolution-1: a 2D Pymunk quadruped (torso + 4 legs) whose 324-parameter NumPy brain (15→16→4) learns to walk by a genetic algorithm. Inputs include a CPG "clock" (sin/cos rhythm). Live demo on Render auto-deploys from `main` (so merging to main = deploying).

## Current state
- **PR #5** (`claude/walkable-bot-research-f94c22`): docs only. Research direction (`docs/research/2026-10-20-research-direction.md`) and the experiments plan. Ready to merge.
- **PR #6** (`claude/research-experiments-metrics-05a6c8`): the research code (P1-P10). Seeded runs, held-out test terrains, 4 brain-input variants, 4 optimizers, stats, F1-F10 figures, REPORT.md, dashboard Experiments tab. 69 tests pass. Smoke run works end to end.
  - The review said **not ready**. Six fixes are needed before real experiments:
    1. Resume trusts `DONE` only, so an edited config keeps stale runs (`src/research/runner.py:55`).
    2. "Train" distance is scored on the same validation terrains used for selection (`runner.py:104`).
    3. The `shuffled` CPG permutation is redrawn every episode instead of once per run (`src/environment.py:29`).
    4. Missing runs silently shrink n and the Holm family (`analysis.py:108`).
    5. One failing run aborts all the others (`runner.py:149`).
    6. A Watch click crashes the replay loop: null `querySelector` (`web/templates/index.html:963`).
  - **Also found in the smoke run:**
    - Runs are about 3× slower than estimated (a full 25k-episode run may take 30+ min).
    - The dt/2 column shows "+60% ⚠" and is ambiguous: is it a drop or a gain?
    - `verify_results.py` needs the results folder, not the config.
- **Merging is blocked by a permission rule.** I have to run `gh pr merge 5 --merge` and `gh pr merge 6 --merge` myself.

## Watch out
- **My main folder** (`D:\wahab stuff\wahab code\Evolution-1`) is 15 commits behind GitHub and has old uncommitted work: an AWS/S3 experiment plus my removal of the race-window metrics in `main.py`. **Don't switch branches or reset there without backing it up to a branch first**, and ask me.
- The worktree `walkable-bot-research-f94c22` is checked out on PR #6's head (detached HEAD). It holds untracked smoke results (`results/smoke`, `web/static/experiments/smoke`), `.claude/launch.json` and this file. Nothing important is unsaved.
- The empty folder `.claude/worktrees/research-experiments-metrics-05a6c8` couldn't be deleted (something had it open). Delete it once that session is closed.

## Decisions I still owe
1. **Repo cleanup** (proposed, not done):
   - back up the main folder to `backup/aws-and-hud`, then reset it to match GitHub
   - delete the 5 merged remote branches and the leftover local branches
   - untrack `data/logs/evolution.log`
   - gitignore `results/` and the dashboard bundle
2. **New goal:** reach the finish line (x=3000) within 60 s (1,200 steps). Recommended scoring: distance + a finish bonus + a speed bonus. "Reached the end yes/no" alone would score everyone 0, so evolution couldn't learn.
3. **Race window:**
   - remove the live metrics (port my uncommitted deletion from the main folder)
   - show one bar-chart results screen at the end instead

## NEXT TASK (start here)
**Plan first, then wait for my OK:**
1. **A notebook "playground"** (`notebooks/playground.ipynb`) that runs on Colab and locally in VS Code:
   - Cell 1: install and load.
   - Cell 2: SETTINGS, the only cell I edit (cpg on/off, mutation rate, seeds).
   - Cell 3: run a small experiment.
   - Cell 4: a bar chart comparing this run with my previous runs.
   - Cell 5: watch the best walker (no pygame on Colab).
2. **A Gymnasium wrapper** for `CreatureEnv`, registered as `Evolution1-Quadruped-v0`:
   - observation Box(15), action Box(4)
   - `gymnasium` as an optional dependency
   - installable via `pip install git+https://github.com/000wahab000/Evolution`

Facts for the plan:
- Colab's free tier has 2 CPU cores; my laptop (i7-7700HQ) has 4 cores / 8 threads and 16 GB RAM.
- Episodes take about 0.026 s of physics, but full runs are slower because of the evaluations.
- Build on PR #6's code, or wait until it's merged. Ask me which.
