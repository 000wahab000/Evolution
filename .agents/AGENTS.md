# AGENTS.md — Evolved Locomotion Build Instructions

> **READ THIS ENTIRE FILE BEFORE TOUCHING ANY CODE.**
> These are not suggestions. Every rule below is enforced. Violation means the
> build is wrong. Start from the top, work to the bottom, stop when done.

---

## 0. PROJECT SNAPSHOT

**What this is:** A genetic algorithm that evolves neural network weights to
control the joints of a segmented creature (quadruped) so it learns to walk.

**Stack — locked, do not change:**

| Layer | Choice |
|---|---|
| Runtime | Node.js + browser Canvas |
| Physics | `planck` (Box2D port, already in `package.json`) |
| Neural net | Hand-written from scratch — **no ML libraries** |
| GA | Hand-written from scratch — mutation, crossover, selection |
| Body | Torso + 4 legs, 1 segment + 1 revolute joint per leg (5 bodies, 4 joints) |
| Bundler | Vite (already in `package.json`) |

**Key numbers:**
- NN inputs: ~9 (4x joint-angle, 4x joint angular-velocity, 1x torso tilt)
- NN outputs: 4 (one torque value per leg joint)
- Genome: flat float32 array of all NN weights
- Population: 30-50 genomes
- Selection: truncation, keep top ~20%
- Fitness: horizontal distance travelled in a 5-10 s headless trial
- Checkpoints: Day 12 (GA converging) and Day 18 (code freeze)

---

## 1. ABSOLUTE RULES — THESE OVERRIDE EVERYTHING

1. **Do not install any machine-learning library.** The neural net must be
   written by hand with plain JavaScript arrays and loops. No TensorFlow,
   ONNX, Brain.js, Synaptic, etc.

2. **Do not install any alternative physics library.** `planck` is the one.
   Do not add Matter.js, Cannon.js, Rapier, or anything else.

3. **Do not add any feature that is on the cut list** (see Section 4).
   Quietly sneaking cut items back in is the #1 failure mode.

4. **Checkpoints are hard gates.** If the GA is not visibly converging
   (fitness trending up) by the end of Day 12, do not proceed to Phase 3.
   Reduce to 2-leg creature and re-run. Document what you changed.

5. **No live-training-as-demo-default.** Pre-trained weights + generation
   replay is the safe demo path. Live training is a bonus only if the loop
   runs fast enough.

6. **Do not rename, move, or delete existing files without a comment
   explaining why.** The repo is small; keep it traceable.

7. **Every commit message must state the day number** (e.g. `day 3: torso + 4 legs`).

8. **Run the headless GA loop in Node.js, not the browser.** Browser-tab
   physics is too slow for training. Rendering happens separately.

9. **Keep the file structure flat and obvious.** No deep nesting, no barrel
   files, no abstraction layers before the thing works.

10. **Physics from scratch is out of scope.** Do not attempt it before
    Phase 2 checkpoint. After that, it is still the lowest-priority stretch
    item and only if you are more than a full day ahead.

---

## 2. TASK LIST — EXECUTE IN ORDER, ONE PHASE AT A TIME

Mark each task `[x]` when complete. Do not start Phase N+1 tasks until all
Phase N tasks are marked done and the checkpoint (if any) is verified.

---

### PHASE 0 — Skeleton (Days 1-2)

Goal: Physics-step-to-canvas loop works. No creature yet.

- [ ] **D1-1** `npm run dev` works; blank canvas renders at 60 fps
- [ ] **D1-2** `planck` world created; single static rectangle body added
- [ ] **D1-3** World step called every frame; body position logged to console to confirm it runs
- [ ] **D1-4** Body drawn on canvas using its planck position (manual draw, no planck renderer)
- [ ] **D2-1** Create 2-3 rigid-body segments connected by 1-2 `RevoluteJoint`s
- [ ] **D2-2** Apply a fixed manual torque to each joint every frame
- [ ] **D2-3** Observe: joints do NOT explode or invert. If they do, fix joint limits before continuing.

**Phase 0 exit criterion:** a jointed thing moves on screen without blowing up.

---

### PHASE 0.5 — Quadruped Body (Day 3)

Goal: Full creature body with correct joint count.

- [ ] **D3-1** Create torso body (rectangle, central)
- [ ] **D3-2** Create 4 leg bodies (one per corner), each attached to torso via `RevoluteJoint`
- [ ] **D3-3** Set joint limits (e.g. +/-45 deg) so legs cannot fold through the torso
- [ ] **D3-4** Apply random torque to all 4 joints; confirm all 4 move independently
- [ ] **D3-5** Visually verify: creature looks like a blocky quadruped, not a pile of rectangles

**Phase 0.5 exit criterion:** 5 bodies, 4 joints, all moving, stable under random torque.

---

### PHASE 1 — The Brain (Days 4-6)

Goal: A hand-written NN drives the creature.

- [ ] **D4-1** Create `src/nn.js` — class `NeuralNet(inputSize, hiddenSize, outputSize)`
- [ ] **D4-2** Implement `forward(inputs)` — full matrix multiply + tanh activation, NO libraries
- [ ] **D4-3** Weights stored as a single flat array (`getWeights()` / `setWeights(arr)`)
- [ ] **D4-4** Unit-test `forward`: fixed input -> fixed output (deterministic, log it)
- [ ] **D5-1** Wire NN inputs from planck: read 4x joint angle, 4x joint angular velocity, 1x torso tilt
- [ ] **D5-2** Wire NN outputs to planck: apply 4 torques to the 4 joints each physics step
- [ ] **D6-1** Remove all hand-coded random torque from Phase 0/0.5 code
- [ ] **D6-2** Spawn one creature with a randomly initialised weight array; observe it attempt to move
- [ ] **D6-3** Confirm: changing the weight array changes the movement (the wiring is real)

**Phase 1 checkpoint:** NN controls a physical quadruped end-to-end, even if behaviour is garbage.

---

### PHASE 2 — Evolution (Days 7-12) WARNING: HIGHEST RISK

Goal: Fitness curve trends upward across generations.

- [ ] **D7-1** Create `src/genome.js` — functions `randomGenome(size)`, `mutate(genome, rate, sigma)`, `crossover(a, b)`
  - `mutate`: add Gaussian noise to each weight with probability `rate`
  - `crossover`: uniform or single-point blend of two parents
- [ ] **D7-2** Verify mutate: print genome before/after; differences exist and are small
- [ ] **D8-1** Create `src/fitness.js` — `runTrial(genome, durationMs)` runs planck headless (no canvas)
- [ ] **D8-2** Fitness = torso X position at end of trial minus X position at start
- [ ] **D8-3** Log fitness for 5 random genomes; values should differ and not all be zero
- [ ] **D9-1** Create `src/ga.js` — `runGeneration(population)` returns new population
  - Score all genomes with `runTrial`
  - Sort by fitness descending
  - Keep top 20% (elites)
  - Fill rest by crossover of two random elites + mutate
- [ ] **D9-2** Run 3 generations manually; log best fitness per generation
- [ ] **D10-1** Create a headless Node.js runner script `train.js`; runs 50+ generations, logs best fitness each gen to a CSV or JSON file
- [ ] **D10-2** Run `train.js`; confirm loop completes without crashing; save weights of best genome
- [ ] **D11-1** Visualise best genome from saved weights; watch behaviour
- [ ] **D11-2** If creature just falls forward and slides — apply fix: penalise large torso rotation OR require torso height above threshold during trial
- [ ] **D11-3** Re-run training with fix; observe if behaviour changes toward actual gait
- [ ] **D12-1** (Buffer) If GA not converging: reduce to 2-leg creature, update body creation, re-run
- [ ] **D12-2** (Buffer) Document the generation at which gait visibly emerged; save that checkpoint's weights

**Phase 2 checkpoint (GATE):**
- Best fitness increases consistently across 20+ generations (plot or log proves it)
- There is a specific generation you can point to where movement shifted from flailing to purposeful
- **If neither is true by end of Day 12 -> do not continue to Phase 3. Fix the GA first.**

---

### PHASE 3 — Demo Legibility (Days 13-16)

Goal: A judge can understand what happened in 15 seconds, without narration.

- [ ] **D13-1** Decide demo mode: pre-trained replay (default) or live-training bonus
- [ ] **D13-2** Load saved best-genome weights file into browser; render that creature walking
- [ ] **D14-1** Render multiple creatures simultaneously (show the population, not just one)
- [ ] **D14-2** Overlay HUD: current generation number + best fitness score, live
- [ ] **D15-1** Implement generation replay: cycle through saved snapshots — Gen 1 (chaos) -> Gen mid (improving) -> Gen late (gait)
- [ ] **D15-2** Replay must be self-contained and require zero user interaction to play
- [ ] **D16-1** (Buffer) Fix any rendering sync bugs, visual glitches, or HUD misalignment

**Phase 3 exit criterion:** You can hand a stranger the URL and they understand the evolution story without being told anything.

---

### PHASE 4 — Stretch (Day 17, CONDITIONAL ONLY)

**Only start if Phase 2 and Phase 3 are fully complete and you are 1+ day ahead of schedule. Pick exactly ONE of the following, in priority order:**

- [ ] **D17-A** Variable leg count toggle (2 / 4 / 6) — highest value
- [ ] **D17-B** Second body type side-by-side comparison
- [ ] **D17-C** Alternate fitness (speed, or jump height)

Do not attempt more than one.

---

### PHASE 5 — Lockdown (Days 18-19)

- [ ] **D18-1** Code freeze — no new features from this point
- [ ] **D18-2** Bug sweep: run the full demo from cold start; fix any crashes or hangs
- [ ] **D18-3** Rehearse demo explanation out loud — be ready to state: fitness function, mutation rate, selection method, genome structure
- [ ] **D19-1** Do not touch code. Rest.

---

## 3. FILE STRUCTURE TO CREATE

```
Evolution-1/
├── src/
│   ├── nn.js          # NeuralNet class, hand-written, no libraries
│   ├── genome.js      # randomGenome, mutate, crossover
│   ├── fitness.js     # runTrial — headless planck simulation
│   ├── ga.js          # runGeneration — full GA loop
│   ├── creature.js    # buildCreature(world) — creates the planck bodies + joints
│   ├── render.js      # drawCreature(ctx, creature) — canvas drawing only
│   └── main.js        # browser entry: loads weights, runs demo renderer
├── train.js           # Node.js headless training runner (NOT browser)
├── data/
│   └── best.json      # saved genome weights + fitness log (gitignored if large)
├── index.html         # Vite entry, loads main.js
├── package.json       # already exists — do not change dependencies
└── evolved-locomotion-timeline.html  # reference only, do not modify
```

---

## 4. CUT LIST — DO NOT ADD THESE BACK

These are permanently out of scope. If you find yourself writing code for
any of the following, stop and delete it.

| Banned Item | Reason |
|---|---|
| Arms / articulated head | Search space too large for 19-day solo build |
| Any ML library | The from-scratch NN is the entire technical depth of the project |
| Physics from scratch | Only considered after Phase 2 gate, and only if far ahead |
| Live training as demo default | Too risky in front of a judge |
| Geopolitical / strategy simulation | Different project entirely, abandoned |
| Historical mode / counterfactual mode | Artifact of old project scope |
| Multiple simultaneous fitness goals | Over-engineering, keep fitness simple |

---

## 5. TECHNICAL SPECIFICATIONS (REFERENCE)

| Parameter | Value | Notes |
|---|---|---|
| Body | Torso + 4 legs, 1 segment + 1 joint each | 5 rigid bodies, 4 revolute joints |
| NN inputs | 9 floats | 4x joint angle, 4x joint angular velocity, 1x torso tilt |
| NN outputs | 4 floats | one torque per leg joint, tanh-scaled |
| Genome | flat Float32Array of all NN weights | size = (inputs x hidden) + hidden + (hidden x outputs) + outputs |
| Mutation | Gaussian noise, rate ~0.05-0.1, sigma ~0.1-0.3 | tune if not converging |
| Crossover | Uniform or single-point | don't over-engineer |
| Population size | 30-50 | 50 is fine, go lower if headless loop is slow |
| Selection | Truncation, top ~20% survive | simple, effective |
| Fitness | delta-x (horizontal displacement) over 5-10 s | add upright penalty if degenerate solutions appear |
| Trial | Headless Node.js, no canvas | planck runs without a DOM |
| Training run | 50-200 generations | stop when fitness plateaus for 20+ gens |

---

## 6. COMMON FAILURE MODES + FIXES

| Symptom | Likely Cause | Fix |
|---|---|---|
| Joints explode on creation | Joint anchor point outside body bounds | Move anchor to body edge, not centre |
| All fitness scores are 0 | Trial duration too short, creature not moving | Increase trial duration; check torque scale |
| Fitness never increases | Mutation rate too high (random walk) | Lower mutation rate to 0.05; check crossover |
| Creature falls forward and "walks" | Degenerate gait — rewarded for falling | Add penalty: torso angle > 45 deg -> subtract fitness |
| Headless loop crashes in Node | planck requires CommonJS require, not ES import | Use `const planck = require('planck');` in train.js |
| Browser renders nothing | main.js not loading saved weights correctly | Console-log the loaded genome; check JSON parse |
| Replay is choppy | Saving every generation is too much data | Save only every 5th generation |

---

## 7. DEFINITION OF DONE

The project is complete when:

1. `node train.js` runs 50+ generations headless and produces a `data/best.json` with a fitness log
2. `npm run dev` opens a browser tab showing the best evolved creature walking
3. The generation replay cycles through early / mid / late with visible improvement
4. You can verbally explain: fitness function, genome structure, mutation, crossover, selection — without looking at notes
5. Code is frozen after Day 18

---

*Generated from `evolved-locomotion-timeline.html` — 19-day build plan, Aug 12-31.*
*Do not modify this file unless the build plan itself changes.*
