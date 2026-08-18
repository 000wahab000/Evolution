# HOW_TO_RUN_SMALL_MODEL.md
# A session-by-session execution guide for running Gemini Flash through this build

> This file is for YOU (the human operator).
> It tells you exactly what to paste into Gemini Flash at the start of each session,
> what to check before moving on, and how to recover if things go wrong.
> The small model does NOT need to read this file.

---

## SETUP — Do This Once Before Any Session

### 1. Confirm Node is installed
Open a terminal in the project folder and run:
```
node --version
npm --version
```
Both must print a version number. If not, install Node.js from https://nodejs.org (LTS version).

### 2. Confirm packages are installed
```
npm install
```
Run this once. It installs `planck` and `vite` from the existing `package.json`.
You should see a `node_modules/` folder appear.

### 3. Confirm the project folder looks like this before starting:
```
Evolution-1/
├── .agents/
│   ├── AGENTS.md
│   └── rules/
│       ├── TASKS_SMALL_MODEL.md
│       └── HOW_TO_RUN_SMALL_MODEL.md  ← this file
├── node_modules/        ← after npm install
├── evolved-locomotion-timeline.html
├── package.json
├── package-lock.json
└── .gitignore
```
Nothing else. Do not create any files yourself. The model creates them.

---

## HOW TO START EACH SESSION — THE STANDARD OPENING PROMPT

At the start of EVERY new Gemini Flash session, paste this exact block.
Replace `[STEP NAME]` with what you want it to do that session.

```
You are a coding assistant working on a JavaScript project called Evolution-1.
It is a genetic algorithm that evolves neural network weights to make a creature walk.

RULES (read before doing anything):
1. Do NOT install any npm package. The only packages are planck and vite (already installed).
2. Do NOT use any machine-learning library. The neural net is hand-written.
3. Do NOT create any file that is not listed in your task.
4. Do NOT modify package.json, package-lock.json, or evolved-locomotion-timeline.html.
5. Work on ONE step at a time. After each step, stop and tell me what you did and what the VERIFY result was.

Your task this session: [STEP NAME]

Full instructions for this task are below:
[PASTE THE EXACT STEP FROM TASKS_SMALL_MODEL.md HERE]
```

---

## SESSION PLAN — One Session Per Step

Run each session separately. A "session" = one conversation window with the model.
Do not combine multiple steps into one session unless marked "(can combine)".

---

### SESSION 0 — Project Setup
**What to give the model:**

Opening prompt (use the standard block above), with task:
```
STEP 0-1 — Add scripts to package.json
```
Paste STEP 0-1 from TASKS_SMALL_MODEL.md.

**What you do after:**
- Run `npm run dev` yourself.
- Confirm a localhost URL appears.
- Stop server with Ctrl+C.
- ✅ Mark STEP 0-1 done.

---

### SESSION 1 — Create index.html + blank canvas
**What to give the model:**
```
Task: Create index.html (STEP 0-2 from my build guide)
```
Paste STEP 0-2 from TASKS_SMALL_MODEL.md.

**What you do after:**
- Run `npm run dev`, open browser.
- Confirm: dark page, black rectangle canvas, no console errors.
- ✅ Mark STEP 0-2 done.

---

### SESSION 2 — Physics world + falling box (can combine with SESSION 3)
**What to give the model:**

Paste STEP 0-3. You may also paste STEP 0-4 in the same session if STEP 0-3 passes.

Tell the model:
```
First do STEP 0-3. Tell me the VERIFY result. If it passes, then do STEP 0-4.
```

**What you do after:**
- STEP 0-3: Green box falls and lands. Console logs position every second.
- STEP 0-4: Two coloured rectangles, joint stays intact.
- ✅ Mark STEP 0-3 and STEP 0-4 done.

---

### SESSION 3 — Build creature.js
**What to give the model:**

Opening prompt + STEP 1-1 from TASKS_SMALL_MODEL.md.

Tell the model:
```
Create the file exactly as shown. Do not change variable names or structure.
After creating it, confirm the file is saved and show me the first 20 lines.
```

**What you do after:**
- Read the file yourself. Check: `buildCreature` is exported, `joints` array has 4 items.
- ✅ Mark STEP 1-1 done.

---

### SESSION 4 — Create render.js
**What to give the model:**

Opening prompt + STEP 1-2.

**What you do after:**
- Open `src/render.js`. Check: `drawCreature` and `drawGround` are exported.
- ✅ Mark STEP 1-2 done.

---

### SESSION 5 — Wire creature to canvas (main.js update)
**What to give the model:**

Opening prompt + STEP 1-3.

Tell the model:
```
Replace the ENTIRE content of src/main.js with the code below. Do not keep any old code.
```

**What you do after:**
- Run `npm run dev`.
- Confirm: green torso + 4 orange legs visible. All 4 legs twitch. Nothing explodes.
- If joints explode → paste EMERGENCY FIX "Joints explode on spawn" to the model.
- ✅ Mark STEP 1-3 done. This completes Phase 0 and 0.5.

---

### SESSION 6 — Neural Net (nn.js)
**What to give the model:**

Opening prompt + STEP 2-1.

Tell the model:
```
Create src/nn.js exactly as shown. After creating it, paste this test in the browser console and show me the output:

import('/src/nn.js').then(({ NeuralNet }) => {
  const nn = new NeuralNet(9, 16, 4);
  const w = new Array(nn.totalWeights).fill(0.1);
  nn.setWeights(w);
  console.log('total weights:', nn.totalWeights);
  console.log('output:', nn.forward(new Array(9).fill(0.5)));
});
```

**What you check:**
- `totalWeights` must be `212` (9×16 + 16 + 16×4 + 4 = 144+16+64+4 = 228... recalc: 9*16=144, +16=160, 16*4=64, +4=68, total=228). It should print 228.
- Output must be an array of 4 numbers, all between -1 and 1.
- Running it twice gives IDENTICAL output (deterministic).
- ✅ Mark STEP 2-1 done.

> NOTE: The totalWeights = (9×16) + 16 + (16×4) + 4 = 144 + 16 + 64 + 4 = **228**

---

### SESSION 7 — Genome functions (genome.js)
**What to give the model:**

Opening prompt + STEP 2-2.

Tell the model:
```
Create src/genome.js exactly as shown.
After creating it, confirm:
1. randomGenome, mutate, crossover are all exported.
2. The mutate function uses .map() and returns a NEW array.
3. The crossover function uses .map() and returns a NEW array.
Show me lines 1-10 of the file.
```

**What you check:**
- All 3 functions exported.
- `mutate` does NOT modify the input (uses `.map()`).
- ✅ Mark STEP 2-2 done.

---

### SESSION 8 — Fitness function (fitness.js)
**What to give the model:**

Opening prompt + STEP 2-3.

Tell the model:
```
Create src/fitness.js exactly as shown.
After creating it, confirm:
1. runTrial is exported.
2. The inputs array inside the step loop has exactly 9 values.
3. world.step(DT) is called inside the for loop.
Show me the section of code where inputs are gathered.
```

**What you check:**
- `runTrial` exported.
- The loop: 4 joints × 2 values + 1 tilt = 9 inputs total.
- `world.step` is INSIDE the loop, not outside it.
- ✅ Mark STEP 2-3 done.

---

### SESSION 9 — GA loop (ga.js)
**What to give the model:**

Opening prompt + STEP 2-4.

Tell the model:
```
Create src/ga.js exactly as shown.
After creating it, confirm:
1. runGeneration and createPopulation are both exported.
2. The elite count uses Math.max(2, ...) so it is never zero.
3. The newPop starts with the elites (they are preserved unchanged).
Show me lines 1-30 of ga.js.
```

**What you check:**
- Both functions exported.
- `Math.max(2, ...)` is present.
- `const newPop = [...elites]` is the line that preserves elites.
- ✅ Mark STEP 2-4 done.

---

### SESSION 10 — train.js (headless Node runner)
**What to give the model:**

Opening prompt + STEP 2-5.

Tell the model:
```
Create train.js at the PROJECT ROOT (not inside src/).
After creating it, run: node train.js
Watch the first 5 lines of output and paste them here.
```

**What you check:**
- First line: `Genome size: 228 weights`
- Lines like: `gen   1 | best fitness: X.XXX`
- Numbers are not all zero.
- No crash.
- If it crashes with `ERR_REQUIRE_ESM` → paste EMERGENCY FIX from TASKS_SMALL_MODEL.md to the model.
- ✅ Mark STEP 2-5 done.

---

### SESSION 11 — Full training run
**This session has no model involvement.**

YOU run this yourself in the terminal:
```
node train.js
```

Let it run completely (100 generations, ~5-15 minutes depending on your machine).

**What you check after:**
- `data/best.json` exists.
- Open the file. Check `log` has 100 entries.
- Compare `log[0].bestFitness` vs `log[99].bestFitness`. Second must be higher.
- If fitness went DOWN or stayed flat → open a model session and show it the log. Ask it to diagnose using AGENTS.md Section 6.
- ✅ Mark STEP 2-6 and PHASE 1 CHECKPOINT done.

---

### SESSION 12 — Demo renderer (main.js final update)
**What to give the model:**

Opening prompt + STEP 3-1.

Tell the model:
```
Replace the ENTIRE content of src/main.js with the code below.
After saving it, run npm run dev, open the browser, and confirm:
1. No console errors.
2. A creature is visible and moving forward.
3. The HUD (top-left) shows generation and fitness numbers.
Tell me what you see.
```

**What you check:**
- Creature walks or at least moves forward.
- HUD visible.
- Camera scrolls to follow the creature.
- If `Cannot GET /data/best.json` → paste EMERGENCY FIX from TASKS_SMALL_MODEL.md.
- ✅ Mark STEP 3-1 done.

---

### SESSION 13 — Generation replay
**What to give the model:**

Opening prompt + STEP 3-2.

Tell the model:
```
Modify src/main.js to add the snapshot replay cycle as shown below.
After saving, run npm run dev. Wait 8 seconds and watch the HUD label.
Tell me what gen numbers appear in the HUD as it cycles.
```

**What you check:**
- HUD label changes every 8 seconds.
- At least 2 different gen numbers appear.
- Creature behaviour looks different between early and late snapshots.
- ✅ Mark STEP 3-2 done.

---

### SESSION 14 — Final lockdown check
**This session has no model involvement.**

Run through this checklist yourself:

```
[ ] node train.js  →  completes without crash, data/best.json written
[ ] npm run dev    →  creature visible, HUD visible, replay cycles
[ ] package.json   →  only planck and vite in dependencies
[ ] No extra files created (no tensorflow, brain.js, matter.js etc)
[ ] data/best.json →  log[99].bestFitness > log[0].bestFitness
```

If all pass: **YOU ARE DONE. Do not touch the code.**

---

## RECOVERY PLAYBOOK — What to Do When Things Break

### Model wrote wrong code / hallucinated an API
1. Do NOT try to fix it in the same session.
2. Start a NEW session.
3. Use the standard opening prompt.
4. Tell the model: `The previous attempt wrote incorrect code. Please replace [FILENAME] entirely with the exact code from my instructions below:` Then paste the exact step again.

### Model added extra imports or packages
1. Open the file yourself. Delete the bad import lines manually.
2. In the next session, tell the model: `Do not add any imports that are not in my instructions. Only use planck and the files listed.`

### Fitness is stuck at 0 for 20+ generations
Session prompt:
```
My training log shows fitness is 0 for every generation.
Here is my src/fitness.js: [paste the file]
Diagnose and fix. The issue is likely one of:
1. Trial duration too short (increase durationSec from 6 to 10)
2. TORQUE_SCALE is 0 (must be 8)
3. world.step() is outside the loop (must be inside)
Check all three and fix whichever is wrong.
```

### Joints explode on creature spawn
Session prompt:
```
When I run npm run dev, the creature's legs fly off screen immediately.
Here is my src/creature.js: [paste the file]
The anchor points for the revolute joints must be INSIDE the torso body.
The torso is Box(0.75, 0.25) centred at (startX, startY).
Anchors at x=±0.6, y=±0.25 are valid. Fix any anchor that is outside those bounds.
```

### train.js crashes with module error
Session prompt:
```
node train.js crashes with this error: [paste error]
Check package.json — the "type" field must be "module" for ES imports to work.
If it says "commonjs", change it to "module" and re-run node train.js.
```

### Browser shows nothing after loading best.json
Session prompt:
```
The browser console shows: Cannot GET /data/best.json
The file exists at data/best.json but Vite cannot serve it.
Move the file from data/best.json to public/best.json.
Then in src/main.js, change the fetch path from '/data/best.json' to '/best.json'.
```

---

## TIPS FOR MANAGING QUOTA

- Each step in TASKS_SMALL_MODEL.md is designed to fit in 1 short session.
- Steps 0-1 through 0-4: ~200 tokens each. Do all 4 in one session.
- Steps 2-1 through 2-4 (nn.js, genome.js, fitness.js, ga.js): ~400 tokens each. One per session.
- The full training run (SESSION 11) uses zero quota — it is a terminal command you run yourself.
- If you are low on quota, skip SESSION 13 (replay) — the demo still works without it.

## MINIMUM VIABLE SESSIONS (if very low quota)

If you need to cut sessions down, these are the ONLY sessions that absolutely must happen:

| Priority | Session | Why |
|---|---|---|
| MUST | SESSION 2 (main.js + planck) | Nothing works without physics loop |
| MUST | SESSION 3 (creature.js) | No creature without this |
| MUST | SESSION 6 (nn.js) | No brain without this |
| MUST | SESSION 8 (fitness.js) | No training without this |
| MUST | SESSION 9 (ga.js) | No evolution without this |
| MUST | SESSION 10 (train.js) | No weights saved without this |
| MUST | SESSION 11 (run training) | Zero quota — just run terminal command |
| MUST | SESSION 12 (main.js demo) | Nothing to show without this |
| SKIP if broke | SESSION 4 (render.js) | Merge with SESSION 5 — create both in one shot |
| SKIP if broke | SESSION 7 (genome.js) | Merge with SESSION 9 — create both in one shot |
| SKIP if broke | SESSION 13 (replay) | Nice to have, not required |

---

*This file is for the human operator. Small model never reads this.*
*For architecture rules: AGENTS.md. For exact code steps: TASKS_SMALL_MODEL.md.*
