---
title: Render Deploy and Repo Cleanup - Plan
type: feat
date: 2026-10-11
topic: render-deploy-and-cleanup
artifact_contract: ce-unified-plan/v1
artifact_readiness: implementation-ready
product_contract_source: ce-plan-bootstrap
execution: code
---

# Render Deploy and Repo Cleanup - Plan

## Goal Capsule

- **Objective:** Put the Evolution-1 web dashboard on a public URL and strip leftover experiment files from the repo.
- **Means:** Existing Dockerfile deployed as a Render free web service, with a demo-mode switch that turns off web-triggered training (KTD1-KTD3).
- **Product Authority:** User request 2026-10-11 ("deploy on render or any service you think is good, remove any trash"), scope confirmed in session.
- **Open Blockers:** None.

---

## Product Contract

### Summary
Deploy the dashboard (live creature stream, training history, best-model download) to Render's free Docker tier as a watch-only demo, and delete tracked files that are generated, duplicated, stale, or unreferenced.

### Problem Frame
Evolution-1 only runs on the author's machine. The Docker image binds a hardcoded port 8000, so it does not respect the `$PORT` that hosting platforms inject. The training endpoint is unauthenticated and forks a 4-process pool, so on a public 0.1-CPU / 512 MB instance any visitor could stall the server and overwrite the shipped best checkpoint. The repo also carries generated plots (one a byte-identical duplicate), a stray research-tool JSON file, a back-compat shim the AGENTS.md "no root pollution" rule argues against, a 5.5 MB unreferenced gif, and config pointing at directories and files that no longer exist.

### Key Decisions
- **Render over Vercel** (session-settled: agent-recommended, user-confirmed — chosen over Vercel: serverless functions cannot hold long SSE streams or a background training thread, and PyTorch exceeds the function size limit). Governs R1.
- **Public deploy is watch-only by default** (session-settled: user-confirmed — chosen over leaving training open: free tier has 0.1 CPU and anyone could overwrite the saved walker). Governs R3.
- **Deploy from `main`; orchestrator merges the PR after tests and a local container run pass** (session-settled: user-confirmed — chosen over deploying the feature branch). Governs R1.
- **Delete `assets/FINNALY WALKING.gif` and `.agents/evolved-locomotion-timeline.html`** (session-settled: user-confirmed — recoverable from git history). Governs R5.

### Requirements

#### Deployment
- R1. The app runs as a public Render web service built from the repo `Dockerfile`, auto-deploying from `main`.
- R2. The container binds to `$PORT` when set and falls back to 8000, so `docker compose up` keeps working unchanged.
- R3. A `TRAINING_ENABLED` env var (default on) gates `POST /api/training/start`; when off it returns 403, `/api/training/status` reports `training_enabled: false`, and the dashboard disables the Run button and says training is off.
- R4. The Docker build context excludes files the server never reads (docs, assets, agent files, tests, worktrees) but keeps `models/` and `logs/` so the demo has checkpoints and history.

#### Cleanup
- R5. Remove: root `plot_logs.py`, `logs/plan.json`, `logs/dashboard.html`, `logs/evolution_graph.png`, `assets/evolution_graph.png`, `assets/evolution_all_runs.png`, `assets/FINNALY WALKING.gif`, `.agents/evolved-locomotion-timeline.html`.
- R6. Generated plot outputs are gitignored so they don't come back.
- R7. Stale config references go: `plot_logs` in pyproject `py-modules`, `norecursedirs = ["spinningup"]`, and the `CONCEPTS.md` line in `.agents/AGENTS.md`.

### Acceptance Examples
- AE1. **Covers:** R2, R4. **Given** the image built locally, **when** run with `PORT=10000 TRAINING_ENABLED=0`, **then** `GET /` returns 200 on port 10000 and `/api/stream/sim` emits `data:` frames.
- AE2. **Covers:** R3. **Given** `TRAINING_ENABLED=0`, **when** a client POSTs `/api/training/start`, **then** it gets 403 and `is_training` stays false.
- AE3. **Covers:** R1. **Given** the PR merged, **when** Render finishes the deploy, **then** the `onrender.com` URL serves the dashboard and the creature stream.

### Scope Boundaries
- No changes to training logic, physics, terrain, or dashboard styling beyond the demo-mode message.
- No persistent disk, auth, or custom domain (free tier has no disk; trained checkpoints on the server reset each deploy).
- Keep: `.agents/AGENTS.md`, `.agents/QUESTIONS.md`, `.agents/plugins/`, `docs/`, `STRATEGY.md`, `compose.yaml`, `logs/history.csv`, `logs/evolution.log`, `models/*`, `requirements.txt` as-is.

#### Deferred to Follow-Up Work
- Render health check path (`/api/training/status`): the Render MCP create call has no health-check field; set it in the dashboard if wanted.
- Splitting `pygame`/`matplotlib` out of the server image to shrink it.

### Success Criteria
- `pytest` passes with no regressions plus new demo-mode tests.
- Live Render URL shows the walking creature.

---

## Planning Contract

### Key Technical Decisions
- KTD1. **Shell-form CMD with `${PORT:-8000}`** in the Dockerfile, `exec` so gunicorn stays PID 1 and gets signals. Render injects `PORT=10000` for Docker services. (Governs R2.)
- KTD2. **`training_enabled` lives in the existing `training_state` dict**, read once from env at import. It ships to the UI through the existing status endpoint with no new route, and tests flip it by patching the dict. Also read the hardcoded `workers: 4` from `TRAINING_WORKERS` (default 4) so a free-tier user who turns training on can drop it to 1. (Governs R3.)
- KTD3. **Render service via Render MCP `create_web_service`**: runtime `docker`, plan `free`, region `singapore` (same as the user's other services), branch `main`, env `TRAINING_ENABLED=0`, `TRAINING_WORKERS=1`. Workspace `tea-d567t1juibrs739f9lm0`, repo `https://github.com/000wahab000/Evolution` (public). (Governs R1.)
- KTD4. **Subagents don't commit.** Both units edit the same worktree on disjoint files; the orchestrator reviews, then commits once. (Process.)

---

## Implementation Units

### U1. Repo cleanup
- **Goal:** Remove trash and stale references.
- **Requirements:** R5, R6, R7.
- **Files:** deletions listed in R5, `.gitignore`, `pyproject.toml`, `.agents/AGENTS.md`.
- **Approach:** `git rm` each R5 path. Append the four generated plot paths to `.gitignore`. Drop `"plot_logs"` from `py-modules` and the `norecursedirs` line. Remove the `CONCEPTS.md` bullet from AGENTS.md. Do not touch README (its plot commands already point at `scripts/plot_logs.py`).
- **Test Scenarios:** grep shows no remaining reference to deleted paths outside `scripts/plot_logs.py` output defaults; `pytest` still collects and passes.
- **Verification:** `git status` shows only intended deletions/edits; `python -m pytest -q`.

### U2. Deploy hardening
- **Goal:** Make the image run correctly and safely on Render.
- **Requirements:** R2, R3, R4.
- **Files:** `Dockerfile`, `.dockerignore`, `app.py`, `templates/index.html`, `tests/test_api.py`, `README.md`.
- **Approach:**
  - Dockerfile CMD → `["sh", "-c", "exec gunicorn app:app --bind 0.0.0.0:${PORT:-8000} --workers 1 --threads 8 --timeout 120 --no-control-socket"]`.
  - `.dockerignore`: add `assets/`, `docs/`, `.agents/`, `.claude/`, `tests/`, `.pytest_cache/`, `STRATEGY.md`.
  - `app.py`: add `"training_enabled": os.environ.get("TRAINING_ENABLED", "1") != "0"` and `"workers": int(os.environ.get("TRAINING_WORKERS", 4))` in `training_state`; `start_training` returns `403 {"message": "Training is disabled on this server"}` when off.
  - `index.html` `pollTrainingStatus` idle branch: when `state.training_enabled === false`, keep `btnStart` disabled and set status text to "Demo mode: training disabled".
  - `tests/test_api.py`: 403 when disabled (patch `training_state`), status exposes `training_enabled`.
  - README: short "Deploy (Render)" section: Docker runtime, `TRAINING_ENABLED`/`TRAINING_WORKERS`, free-tier sleep and ephemeral disk.
- **Test Scenarios:** AE1, AE2.
- **Verification:** `python -m pytest -q`; `docker build` + `docker run -e PORT=10000 -e TRAINING_ENABLED=0 -p 10000:10000` then curl `/`, `/api/training/status`, POST `/api/training/start` (expect 403), and read a few bytes of `/api/stream/sim`. If Docker isn't running, report that instead of skipping silently.

### U3. Ship and deploy (orchestrator)
- **Goal:** Get reviewed changes onto `main` and live on Render.
- **Requirements:** R1.
- **Approach:** Review diff (`ce-code-review`), commit, push, open PR (`ce-commit-push-pr`), merge after checks, create the Render service (KTD3), watch the deploy logs, hit the live URL.
- **Verification:** AE3.

---

## Verification Contract

### Automated Tests
- `python -m pytest -q`

### Manual Verification
1. Local container smoke test (U2).
2. Render deploy reaches `live`; the public URL renders the canvas and streams frames; `POST /api/training/start` returns 403.

---

## Definition of Done
- All R5 files gone, references cleaned, tests green.
- PR merged to `main`; Render service live; URL reported to the user.
