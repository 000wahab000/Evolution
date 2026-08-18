# TASKS_SMALL_MODEL.md
# Step-by-step build guide — optimised for small/quota-limited models

> **HOW TO USE THIS FILE**
> - Read ONE step at a time. Do exactly what it says. Nothing more.
> - Every step has: ACTION, CODE (copy-paste exact), VERIFY (run this to confirm it worked).
> - Do not proceed to the next step until VERIFY passes.
> - If something breaks, check AGENTS.md Section 6 (Failure Modes) first.
> - Do NOT install any package not listed below. Do NOT create any file not listed below.

---

## BEFORE YOU START — READ ONCE

**Locked packages (already in package.json — do not add anything else):**
- `planck` — physics engine
- `vite` — bundler / dev server

**You will create exactly these files, in this order:**
1. `index.html`
2. `src/main.js`
3. `src/creature.js`
4. `src/render.js`
5. `src/nn.js`
6. `src/genome.js`
7. `src/fitness.js`
8. `src/ga.js`
9. `train.js`

**Never touch:** `package.json`, `package-lock.json`, `evolved-locomotion-timeline.html`

---

## PHASE 0 — SKELETON
### Goal: Physics loop runs, one body appears on screen.

---

### STEP 0-1 — Add scripts to package.json

**ACTION:** Open `package.json`. Replace the `"scripts"` block with the one below.

```json
"scripts": {
  "dev": "vite",
  "build": "vite build"
},
```

**VERIFY:** Run `npm run dev` in terminal. A localhost URL should appear. Opening it in browser should show a blank page (no errors in console yet). Stop the server with Ctrl+C.

---

### STEP 0-2 — Create index.html

**ACTION:** Create file `index.html` at the project root with this exact content:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Evolved Locomotion</title>
  <style>
    body { margin: 0; background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; }
    canvas { border: 1px solid #333; }
  </style>
</head>
<body>
  <canvas id="c" width="900" height="400"></canvas>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
```

**VERIFY:** Run `npm run dev`, open browser. You should see a dark page with a black rectangle (the canvas). No console errors.

---

### STEP 0-3 — Create src/main.js with planck world

**ACTION:** Create folder `src/`. Create file `src/main.js` with this exact content:

```js
import planck from 'planck';

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

// Create planck world with gravity
const world = planck.World({ gravity: planck.Vec2(0, -10) });

// Create a static ground body
const ground = world.createBody({ type: 'static', position: planck.Vec2(0, -5) });
ground.createFixture(planck.Box(20, 0.5), { friction: 0.8 });

// Create one dynamic box to confirm physics works
const box = world.createBody({ type: 'dynamic', position: planck.Vec2(0, 5) });
box.createFixture(planck.Box(0.5, 0.5), { density: 1.0, friction: 0.3 });

const SCALE = 40; // pixels per meter
const OX = canvas.width / 2;
const OY = canvas.height / 2;

function toScreen(v) {
  return { x: OX + v.x * SCALE, y: OY - v.y * SCALE };
}

function step() {
  world.step(1 / 60);

  // Log position to console every 60 frames
  step._count = (step._count || 0) + 1;
  if (step._count % 60 === 0) {
    const pos = box.getPosition();
    console.log('box pos:', pos.x.toFixed(2), pos.y.toFixed(2));
  }

  // Draw
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw ground
  const gp = toScreen(ground.getPosition());
  ctx.fillStyle = '#555';
  ctx.fillRect(gp.x - 800, gp.y - 10, 1600, 20);

  // Draw box
  const bp = toScreen(box.getPosition());
  ctx.fillStyle = '#9fef6b';
  ctx.fillRect(bp.x - 20, bp.y - 20, 40, 40);

  requestAnimationFrame(step);
}

requestAnimationFrame(step);
```

**VERIFY:** Run `npm run dev`. Open browser. You should see:
- A green box falling and landing on a grey ground line.
- In the browser console: `box pos: 0.00 X.XX` printing every second, Y value decreasing until it rests near 0.

---

### STEP 0-4 — Add RevoluteJoint test (2 bodies, 1 joint)

**ACTION:** Replace the "Create one dynamic box" block in `src/main.js` with this:

```js
// Segment A (parent)
const segA = world.createBody({ type: 'dynamic', position: planck.Vec2(0, 3) });
segA.createFixture(planck.Box(0.6, 0.15), { density: 1.0, friction: 0.5 });

// Segment B (child)
const segB = world.createBody({ type: 'dynamic', position: planck.Vec2(0.6, 3) });
segB.createFixture(planck.Box(0.5, 0.1), { density: 1.0, friction: 0.5 });

// Revolute joint connecting A to B at the right edge of A
const joint = world.createJoint(planck.RevoluteJoint({
  lowerAngle: -Math.PI / 4,
  upperAngle:  Math.PI / 4,
  enableLimit: true,
  enableMotor: false,
  maxMotorTorque: 5,
}, segA, segB, planck.Vec2(0.6, 3)));
```

Then replace the draw block to draw both segments:

```js
// Draw segA
const ap = toScreen(segA.getPosition());
ctx.fillStyle = '#9fef6b';
ctx.fillRect(ap.x - 24, ap.y - 6, 48, 12);

// Draw segB
const sp = toScreen(segB.getPosition());
ctx.fillStyle = '#ef8b6b';
ctx.fillRect(sp.x - 20, sp.y - 4, 40, 8);
```

**VERIFY:** Two coloured rectangles appear connected. When they fall they do NOT spin endlessly or fly off screen. They rest on the ground. Joint stays intact.

---

## PHASE 0.5 — QUADRUPED BODY
### Goal: 5 bodies, 4 joints. Stable under random torque.

---

### STEP 1-1 — Create src/creature.js

**ACTION:** Create `src/creature.js` with this exact content. Do not change variable names.

```js
import planck from 'planck';

const DEG45 = Math.PI / 4;

/**
 * buildCreature(world)
 * Returns: { torso, legs: [fl, fr, bl, br], joints: [jfl, jfr, jbl, jbr] }
 * fl = front-left, fr = front-right, bl = back-left, br = back-right
 */
export function buildCreature(world, startX = 0, startY = 3) {
  // --- TORSO ---
  const torso = world.createBody({
    type: 'dynamic',
    position: planck.Vec2(startX, startY),
    linearDamping: 0.1,
    angularDamping: 0.5,
  });
  torso.createFixture(planck.Box(0.75, 0.25), { density: 2.0, friction: 0.3 });

  // --- LEG OFFSETS (from torso centre) ---
  const legOffsets = [
    { x: -0.6, y: -0.25, label: 'fl' }, // front-left
    { x:  0.6, y: -0.25, label: 'fr' }, // front-right
    { x: -0.6, y:  0.25, label: 'bl' }, // back-left  (actually: negative x = "back" visually when creature faces right)
    { x:  0.6, y:  0.25, label: 'br' }, // back-right
  ];

  const legs = [];
  const joints = [];

  for (const off of legOffsets) {
    const legX = startX + off.x;
    const legY = startY + off.y - 0.4; // leg hangs below anchor point

    const leg = world.createBody({
      type: 'dynamic',
      position: planck.Vec2(legX, legY),
      linearDamping: 0.1,
    });
    leg.createFixture(planck.Box(0.08, 0.4), { density: 1.0, friction: 0.8 });

    // Joint anchor is at the torso surface where the leg attaches
    const anchorWorld = planck.Vec2(startX + off.x, startY + off.y);

    const joint = world.createJoint(planck.RevoluteJoint({
      lowerAngle: -DEG45,
      upperAngle:  DEG45,
      enableLimit: true,
      enableMotor: true,
      maxMotorTorque: 8,
      motorSpeed: 0,
    }, torso, leg, anchorWorld));

    legs.push(leg);
    joints.push(joint);
  }

  return { torso, legs, joints };
}
```

**VERIFY:** File saved. No syntax errors (open a terminal and run `node -e "require('./src/creature.js')"` — it will fail because of ES modules, that is fine. Just make sure the file has no obvious typos by eye-checking it).

---

### STEP 1-2 — Create src/render.js

**ACTION:** Create `src/render.js`:

```js
const SCALE = 40;

export function worldToScreen(canvas, v) {
  return {
    x: canvas.width / 2 + v.x * SCALE,
    y: canvas.height / 2 - v.y * SCALE,
  };
}

export function drawCreature(ctx, canvas, creature, color = '#9fef6b') {
  const { torso, legs } = creature;

  // Draw torso
  drawBody(ctx, canvas, torso, color, 60, 20);

  // Draw legs
  for (const leg of legs) {
    drawBody(ctx, canvas, leg, '#ef8b6b', 6, 32);
  }
}

function drawBody(ctx, canvas, body, color, halfW, halfH) {
  const pos = body.getPosition();
  const angle = body.getAngle();
  const s = worldToScreen(canvas, pos);

  ctx.save();
  ctx.translate(s.x, s.y);
  ctx.rotate(-angle);
  ctx.fillStyle = color;
  ctx.fillRect(-halfW, -halfH, halfW * 2, halfH * 2);
  ctx.restore();
}

export function drawGround(ctx, canvas) {
  ctx.fillStyle = '#333';
  ctx.fillRect(0, canvas.height / 2 + 5 * SCALE - 10, canvas.width, 20);
}
```

**VERIFY:** File saved, no typos.

---

### STEP 1-3 — Update src/main.js to use creature + random torque

**ACTION:** Replace the entire content of `src/main.js` with:

```js
import planck from 'planck';
import { buildCreature } from './creature.js';
import { drawCreature, drawGround } from './render.js';

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');

const world = planck.World({ gravity: planck.Vec2(0, -10) });

// Ground
const ground = world.createBody({ type: 'static', position: planck.Vec2(0, -5) });
ground.createFixture(planck.Box(50, 0.5), { friction: 0.8 });

// Build creature
const creature = buildCreature(world, 0, 2);

let frame = 0;

function loop() {
  frame++;

  // Apply random torque to each joint every 30 frames
  if (frame % 30 === 0) {
    for (const joint of creature.joints) {
      joint.setMotorSpeed((Math.random() - 0.5) * 6);
    }
  }

  world.step(1 / 60);

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  drawGround(ctx, canvas);
  drawCreature(ctx, canvas, creature);

  requestAnimationFrame(loop);
}

requestAnimationFrame(loop);
```

**VERIFY:** Run `npm run dev`. You should see:
- A green torso rectangle with 4 orange leg rectangles attached.
- All 4 legs twitch and move independently.
- The creature stays roughly on screen — it may fall or slide but does NOT explode.
- If legs fly off or joints break → go to AGENTS.md Section 6, "Joints explode on creation".

---

## PHASE 1 — THE BRAIN
### Goal: Hand-written neural net controls the creature.

---

### STEP 2-1 — Create src/nn.js

**ACTION:** Create `src/nn.js` with this exact content:

```js
/**
 * NeuralNet — hand-written, no libraries.
 * Architecture: inputSize -> hiddenSize (tanh) -> outputSize (tanh)
 */
export class NeuralNet {
  constructor(inputSize, hiddenSize, outputSize) {
    this.inputSize  = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Weight counts
    this.w1Size = inputSize * hiddenSize;
    this.b1Size = hiddenSize;
    this.w2Size = hiddenSize * outputSize;
    this.b2Size = outputSize;
    this.totalWeights = this.w1Size + this.b1Size + this.w2Size + this.b2Size;
  }

  // Set weights from a flat array
  setWeights(arr) {
    let i = 0;
    this.w1 = arr.slice(i, i += this.w1Size); // inputSize x hiddenSize
    this.b1 = arr.slice(i, i += this.b1Size);
    this.w2 = arr.slice(i, i += this.w2Size); // hiddenSize x outputSize
    this.b2 = arr.slice(i, i += this.b2Size);
  }

  // Get weights as a flat array
  getWeights() {
    return [...this.w1, ...this.b1, ...this.w2, ...this.b2];
  }

  // Forward pass — returns array of outputSize numbers in range [-1, 1]
  forward(inputs) {
    const { hiddenSize, outputSize, inputSize } = this;

    // Hidden layer
    const hidden = new Array(hiddenSize);
    for (let j = 0; j < hiddenSize; j++) {
      let sum = this.b1[j];
      for (let k = 0; k < inputSize; k++) {
        sum += inputs[k] * this.w1[k * hiddenSize + j];
      }
      hidden[j] = Math.tanh(sum);
    }

    // Output layer
    const output = new Array(outputSize);
    for (let j = 0; j < outputSize; j++) {
      let sum = this.b2[j];
      for (let k = 0; k < hiddenSize; k++) {
        sum += hidden[k] * this.w2[k * outputSize + j];
      }
      output[j] = Math.tanh(sum);
    }

    return output;
  }
}
```

**VERIFY:** Open browser console (dev server running). Paste this and press Enter:
```js
// Quick unit test — paste in browser console or Node REPL
import('/src/nn.js').then(({ NeuralNet }) => {
  const nn = new NeuralNet(9, 16, 4);
  const w = new Array(nn.totalWeights).fill(0.1);
  nn.setWeights(w);
  const out = nn.forward(new Array(9).fill(0.5));
  console.log('NN output:', out); // Must be 4 numbers, all in range [-1, 1]
});
```
You must see 4 numbers printed. They must be consistent — running the same call twice gives the same result.

---

### STEP 2-2 — Create src/genome.js

**ACTION:** Create `src/genome.js`:

```js
/**
 * Genome helpers — all genomes are plain JS arrays of floats.
 */

// Box-Muller: returns a gaussian random number, mean=0, std=1
function gaussRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Create a random genome of `size` weights, uniform in [-1, 1] */
export function randomGenome(size) {
  return Array.from({ length: size }, () => (Math.random() * 2 - 1));
}

/**
 * Mutate a genome by adding Gaussian noise to each weight with probability `rate`.
 * sigma: standard deviation of the noise (0.1–0.3 recommended)
 * Returns a NEW array — does not modify the input.
 */
export function mutate(genome, rate = 0.08, sigma = 0.2) {
  return genome.map(w => Math.random() < rate ? w + gaussRandom() * sigma : w);
}

/**
 * Uniform crossover: for each index, pick from parent a or b with 50/50 chance.
 * Returns a NEW array.
 */
export function crossover(a, b) {
  return a.map((w, i) => Math.random() < 0.5 ? w : b[i]);
}
```

**VERIFY:** Run this in Node (not browser — this is a plain CJS-compatible test):
```
node -e "
const g = require('./src/genome.js'); // will fail on ES module
"
```
It will fail with ES module error — that is expected. Instead, verify by eye: confirm `randomGenome`, `mutate`, `crossover` are all exported. Confirm `mutate` returns a new array (check: it uses `.map()`). Confirm `crossover` uses `.map()`.

---

### STEP 2-3 — Create src/fitness.js

**ACTION:** Create `src/fitness.js`. This file runs headless (no canvas, no DOM).

```js
import planck from 'planck';
import { NeuralNet } from './nn.js';

const INPUT_SIZE  = 9;
const HIDDEN_SIZE = 16;
const OUTPUT_SIZE = 4;
const TORQUE_SCALE = 8; // max torque applied per joint

/**
 * Build a minimal planck world with ground + quadruped creature.
 * Returns { world, torso, legs, joints }
 */
function buildHeadlessWorld() {
  const world = planck.World({ gravity: planck.Vec2(0, -10) });

  // Ground
  const ground = world.createBody({ type: 'static', position: planck.Vec2(0, -5) });
  ground.createFixture(planck.Box(200, 0.5), { friction: 0.8 });

  const DEG45 = Math.PI / 4;
  const startX = 0;
  const startY = 2;

  // Torso
  const torso = world.createBody({
    type: 'dynamic',
    position: planck.Vec2(startX, startY),
    linearDamping: 0.1,
    angularDamping: 0.5,
  });
  torso.createFixture(planck.Box(0.75, 0.25), { density: 2.0, friction: 0.3 });

  // Legs + joints
  const legOffsets = [
    { x: -0.6, y: -0.25 },
    { x:  0.6, y: -0.25 },
    { x: -0.6, y:  0.25 },
    { x:  0.6, y:  0.25 },
  ];

  const legs = [];
  const joints = [];

  for (const off of legOffsets) {
    const leg = world.createBody({
      type: 'dynamic',
      position: planck.Vec2(startX + off.x, startY + off.y - 0.4),
      linearDamping: 0.1,
    });
    leg.createFixture(planck.Box(0.08, 0.4), { density: 1.0, friction: 0.8 });

    const anchor = planck.Vec2(startX + off.x, startY + off.y);
    const joint = world.createJoint(planck.RevoluteJoint({
      lowerAngle: -DEG45,
      upperAngle:  DEG45,
      enableLimit: true,
      enableMotor: true,
      maxMotorTorque: TORQUE_SCALE,
      motorSpeed: 0,
    }, torso, leg, anchor));

    legs.push(leg);
    joints.push(joint);
  }

  return { world, torso, legs, joints };
}

/**
 * runTrial(genome, durationSec)
 * Runs the creature headless for durationSec seconds.
 * Returns fitness = horizontal displacement of torso (meters).
 * Negative fitness if creature falls over badly.
 */
export function runTrial(genome, durationSec = 6) {
  const { world, torso, legs, joints } = buildHeadlessWorld();

  const nn = new NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
  nn.setWeights(genome);

  const startX = torso.getPosition().x;
  const DT = 1 / 60;
  const steps = Math.floor(durationSec / DT);

  for (let i = 0; i < steps; i++) {
    // Gather inputs
    const inputs = [];
    for (const joint of joints) {
      inputs.push(joint.getJointAngle());           // joint angle
      inputs.push(joint.getJointSpeed());           // joint angular velocity
    }
    inputs.push(torso.getAngle());                  // torso tilt (1 value)
    // inputs now has 4*2 + 1 = 9 values

    // Forward pass
    const outputs = nn.forward(inputs);

    // Apply torques
    for (let j = 0; j < joints.length; j++) {
      joints[j].setMotorSpeed(outputs[j] * TORQUE_SCALE);
    }

    world.step(DT);
  }

  const endX   = torso.getPosition().x;
  const tilt   = Math.abs(torso.getAngle());
  const deltaX = endX - startX;

  // Penalty: if creature tilts more than 80 degrees, subtract fitness
  const tiltPenalty = tilt > 1.4 ? deltaX * 0.5 : 0;

  return deltaX - tiltPenalty;
}
```

**VERIFY:** This file cannot be run directly in the browser (it is used by train.js). Check that:
- `runTrial` is exported.
- The `inputs` array inside the loop has exactly 9 values (4 joints × 2 readings + 1 torso tilt).
- `world.step(DT)` is called inside the loop.

---

### STEP 2-4 — Create src/ga.js

**ACTION:** Create `src/ga.js`:

```js
import { randomGenome, mutate, crossover } from './genome.js';
import { runTrial } from './fitness.js';

const POPULATION_SIZE = 40;
const ELITE_FRACTION  = 0.2;  // keep top 20%
const MUTATION_RATE   = 0.08;
const MUTATION_SIGMA  = 0.2;

/**
 * runGeneration(population)
 * population: array of genome arrays
 * Returns: { newPopulation, scoredPop }
 * scoredPop: [{ genome, fitness }] sorted by fitness descending
 */
export function runGeneration(population) {
  // Score every genome
  const scored = population.map(genome => ({
    genome,
    fitness: runTrial(genome),
  }));

  // Sort best first
  scored.sort((a, b) => b.fitness - a.fitness);

  // Keep elites
  const eliteCount = Math.max(2, Math.floor(POPULATION_SIZE * ELITE_FRACTION));
  const elites = scored.slice(0, eliteCount).map(s => s.genome);

  // Breed new population
  const newPop = [...elites]; // elites survive unchanged
  while (newPop.length < POPULATION_SIZE) {
    const parentA = elites[Math.floor(Math.random() * elites.length)];
    const parentB = elites[Math.floor(Math.random() * elites.length)];
    const child = mutate(crossover(parentA, parentB), MUTATION_RATE, MUTATION_SIGMA);
    newPop.push(child);
  }

  return { newPopulation: newPop, scoredPop: scored };
}

/** Create an initial random population */
export function createPopulation(genomeSize) {
  return Array.from({ length: POPULATION_SIZE }, () => randomGenome(genomeSize));
}
```

**VERIFY:** File saved. Confirm `runGeneration` and `createPopulation` are exported. Confirm `eliteCount` is at least 2 (never zero).

---

### STEP 2-5 — Create train.js (headless Node runner)

**ACTION:** Create `train.js` at the project root (NOT inside `src/`):

```js
// train.js — run with: node train.js
// Uses CommonJS-compatible planck import
// All src/ files use ES modules via dynamic import

import { NeuralNet } from './src/nn.js';
import { createPopulation, runGeneration } from './src/ga.js';
import { runTrial } from './src/fitness.js';
import { randomGenome } from './src/genome.js';
import fs from 'fs';

const GENERATIONS    = 100;
const GENOME_SIZE    = new NeuralNet(9, 16, 4).totalWeights;
const DATA_DIR       = './data';
const OUT_FILE       = `${DATA_DIR}/best.json`;
const SNAPSHOT_EVERY = 5; // save a snapshot every N generations

if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

console.log(`Genome size: ${GENOME_SIZE} weights`);
console.log(`Starting evolution — ${GENERATIONS} generations\n`);

let population = createPopulation(GENOME_SIZE);
const log      = [];       // { gen, bestFitness }
const snapshots = [];      // { gen, genome, fitness } every SNAPSHOT_EVERY gens

for (let gen = 1; gen <= GENERATIONS; gen++) {
  const { newPopulation, scoredPop } = runGeneration(population);
  population = newPopulation;

  const best = scoredPop[0];
  log.push({ gen, bestFitness: best.fitness });

  console.log(`gen ${String(gen).padStart(3)} | best fitness: ${best.fitness.toFixed(3)}`);

  // Save snapshot every N generations
  if (gen % SNAPSHOT_EVERY === 0 || gen === 1) {
    snapshots.push({ gen, genome: best.genome, fitness: best.fitness });
  }
}

// Save result
const result = {
  bestGenome:  population[0],  // population is already sorted after last gen
  log,
  snapshots,
};
fs.writeFileSync(OUT_FILE, JSON.stringify(result, null, 2));
console.log(`\nDone. Best genome saved to ${OUT_FILE}`);
```

**VERIFY:** Run `node train.js`. You should see:
```
Genome size: 212 weights
Starting evolution — 100 generations

gen   1 | best fitness: X.XXX
gen   2 | best fitness: X.XXX
...
```
It must not crash. Fitness values must not all be zero. If all are zero → see AGENTS.md Section 6. Let it run for at least 10 generations to confirm fitness is changing. You can Ctrl+C early to stop.

---

### STEP 2-6 — Run full training

**ACTION:** Run the full training loop:
```
node train.js
```
Let it complete all 100 generations. This will take several minutes.

**VERIFY:** After it finishes:
- File `data/best.json` exists.
- It contains `bestGenome` (a long array of numbers), `log` (100 entries), and `snapshots`.
- Open the file and check that `log[99].bestFitness` is higher than `log[0].bestFitness`. If not, the GA is not converging — see AGENTS.md Section 6.

---

## PHASE 1 CHECKPOINT — GATE

**Before continuing, confirm ALL of the following:**
- [ ] `data/best.json` exists and has 100 log entries
- [ ] `log[0].bestFitness` < `log[99].bestFitness` (fitness increased over training)
- [ ] No crashes during training run

**If any check fails → do NOT continue to Phase 3. Debug the GA first.**

---

## PHASE 3 — DEMO RENDERER
### Goal: Load saved weights, show the creature walking in the browser.

---

### STEP 3-1 — Update src/main.js to load best.json and render

**ACTION:** Replace `src/main.js` entirely with:

```js
import planck from 'planck';
import { buildCreature } from './creature.js';
import { drawCreature, drawGround } from './render.js';
import { NeuralNet } from './nn.js';

const canvas = document.getElementById('c');
const ctx    = canvas.getContext('2d');

const INPUT_SIZE  = 9;
const HIDDEN_SIZE = 16;
const OUTPUT_SIZE = 4;
const TORQUE_SCALE = 8;

// Load best.json and start rendering
fetch('/data/best.json')
  .then(r => r.json())
  .then(data => startDemo(data))
  .catch(err => {
    console.error('Could not load data/best.json — run node train.js first.', err);
  });

function startDemo(data) {
  const { bestGenome, snapshots } = data;

  const nn = new NeuralNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
  nn.setWeights(bestGenome);

  const world = planck.World({ gravity: planck.Vec2(0, -10) });

  // Ground
  const ground = world.createBody({ type: 'static', position: planck.Vec2(0, -5) });
  ground.createFixture(planck.Box(200, 0.5), { friction: 0.8 });

  const creature = buildCreature(world, 0, 2);

  // HUD state
  let genLabel = `Best genome (gen ${data.log[data.log.length - 1].gen})`;
  let fitnessLabel = `Fitness: ${data.log[data.log.length - 1].bestFitness.toFixed(2)}`;

  // Camera follows torso
  let camX = 0;

  function loop() {
    // Gather inputs
    const inputs = [];
    for (const joint of creature.joints) {
      inputs.push(joint.getJointAngle());
      inputs.push(joint.getJointSpeed());
    }
    inputs.push(creature.torso.getAngle());

    const outputs = nn.forward(inputs);
    for (let j = 0; j < creature.joints.length; j++) {
      creature.joints[j].setMotorSpeed(outputs[j] * TORQUE_SCALE);
    }

    world.step(1 / 60);

    // Camera tracks torso
    camX = creature.torso.getPosition().x;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(-camX * 40 + canvas.width / 2, 0); // scroll camera
    drawGround(ctx, canvas);
    drawCreature(ctx, canvas, creature);
    ctx.restore();

    // HUD
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    ctx.fillRect(10, 10, 280, 50);
    ctx.fillStyle = '#9fef6b';
    ctx.font = '13px monospace';
    ctx.fillText(genLabel, 18, 30);
    ctx.fillText(fitnessLabel, 18, 48);

    requestAnimationFrame(loop);
  }

  loop();
}
```

**VERIFY:** Run `npm run dev`. Open browser. You should see:
- The evolved creature walking (or at least moving forward).
- A HUD in the top-left showing generation and fitness.
- Camera scrolling to follow the creature.
- No console errors about JSON parse.

---

### STEP 3-2 — Add generation replay (snapshots viewer)

**ACTION:** Add a second canvas or a "replay" button is NOT needed. Instead, add this replay cycle to `src/main.js` inside `startDemo`:

After the `nn.setWeights(bestGenome)` line, add:

```js
  // --- SNAPSHOT REPLAY ---
  // Cycle through: snapshot[0] (early chaos) -> mid -> best
  // Each snapshot runs for 8 seconds of real time, then switches
  const replaySnapshots = snapshots.length > 0
    ? [snapshots[0], snapshots[Math.floor(snapshots.length / 2)], snapshots[snapshots.length - 1]]
    : [{ genome: bestGenome, fitness: data.log[data.log.length - 1].bestFitness, gen: data.log[data.log.length - 1].gen }];

  let replayIndex = 0;
  let replayFrames = 0;
  const FRAMES_PER_SNAPSHOT = 60 * 8; // 8 seconds at 60fps
```

Then inside the `loop()` function, replace the `nn.setWeights(bestGenome)` call at the top with this logic:

```js
    // Replay cycle
    replayFrames++;
    if (replayFrames >= FRAMES_PER_SNAPSHOT) {
      replayFrames = 0;
      replayIndex = (replayIndex + 1) % replaySnapshots.length;
      nn.setWeights(replaySnapshots[replayIndex].genome);
      // Reset creature position (rebuild world)
      // Simple approach: just update the NN weights — creature keeps walking
    }

    const snap = replaySnapshots[replayIndex];
    genLabel     = `Gen ${snap.gen} replay`;
    fitnessLabel = `Fitness: ${snap.fitness.toFixed(2)}`;
```

**VERIFY:** Open browser. Every 8 seconds the HUD label should change between gen numbers (e.g. Gen 1, Gen 50, Gen 100). The creature may behave differently for each snapshot (early ones flail, late ones walk).

---

## PHASE 5 — LOCKDOWN

### STEP 4-1 — Code freeze checklist

**ACTION:** Run each command below and confirm it works with no errors:

```
npm run dev
```
Open browser → creature walks → HUD shows → replay cycles.

```
node train.js
```
Confirm it completes 100 generations and writes `data/best.json`.

**VERIFY — Final definition of done:**
- [ ] `node train.js` runs to completion without crashing
- [ ] `data/best.json` exists and has `log` with increasing fitness
- [ ] `npm run dev` opens a browser tab with a walking creature
- [ ] HUD shows generation number and fitness
- [ ] Replay cycles through at least 2 snapshots
- [ ] No ML libraries installed (check `package.json` — only `planck` and `vite`)
- [ ] No files renamed or deleted

**If all boxes are checked: YOU ARE DONE. Do not touch the code.**

---

## QUICK REFERENCE — NUMBERS TO NEVER CHANGE

| Thing | Value | Where used |
|---|---|---|
| NN input size | 9 | nn.js, fitness.js, main.js, train.js |
| NN hidden size | 16 | nn.js, fitness.js, main.js, train.js |
| NN output size | 4 | nn.js, fitness.js, main.js, train.js |
| Torque scale | 8 | fitness.js, main.js |
| Population size | 40 | ga.js |
| Elite fraction | 0.2 (top 20%) | ga.js |
| Mutation rate | 0.08 | ga.js |
| Mutation sigma | 0.2 | ga.js |
| Trial duration | 6 sec | fitness.js |
| Timestep | 1/60 | fitness.js, main.js |

---

## EMERGENCY FIXES — copy-paste solutions

### "Fitness is always 0"
In `src/fitness.js`, change `durationSec = 6` to `durationSec = 10`. Also check that `TORQUE_SCALE` is 8, not 0.

### "Joints explode on spawn"
In `src/creature.js`, verify `anchorWorld` coordinates are within the torso bounding box:
- Torso half-width = 0.75, half-height = 0.25
- Anchors at `x = ±0.6, y = ±0.25` are inside those bounds. Do not change to ±1.0 or larger.

### "Fitness never increases after 50 gens"
In `src/ga.js`, try:
```js
const MUTATION_RATE = 0.05; // lower (was 0.08)
const MUTATION_SIGMA = 0.15; // lower (was 0.2)
```
Re-run `node train.js`.

### "node train.js crashes with ERR_REQUIRE_ESM"
Add `"type": "module"` to `package.json` if not already there. Already set: `"type": "commonjs"` — change it to `"module"` and re-run.

### "Browser shows nothing"
Open DevTools console. If error is `Cannot GET /data/best.json`:
- Run `node train.js` first to generate the file.
- If Vite doesn't serve the `data/` folder, move `best.json` to `public/best.json` and change the fetch path to `/best.json`.

---

*This file is for small-model execution. For architecture overview and absolute rules, read AGENTS.md first.*
