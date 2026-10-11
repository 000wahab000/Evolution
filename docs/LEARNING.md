# Learning Evolution-1

A study guide built around this repo. Each module covers one layer of the project. It names the files to read, the concepts behind them, where to learn them, and questions you should be able to answer **out loud, without notes**. If you can answer a module's questions, you can explain that part of the project to anyone.

Work through the modules in order. Modules 1-5 are the core of the project: they're what makes the creature walk. Modules 6-10 cover how it runs and ships.

---

## The 60-second explanation (goal of this guide)

Practise until you can say this in your own words and answer follow-ups on any sentence:

> A four-legged creature made of rigid bodies lives in a 2D physics simulation (Pymunk). Its brain is a tiny neural network: 15 numbers describing its body go in, 4 motor speeds come out. That's 324 weights in total. We don't train it with gradients. We run a genetic algorithm: make 50 random brains, let each one try to walk for 600 steps, keep the 5 that went furthest right without falling, and breed the next 50 from them with crossover and mutation. Repeat for ~100 generations and a walking gait emerges. A sine/cosine "clock" input (a central pattern generator) gives the brain a rhythm to lock its stride to. Training runs in parallel across CPU cores. The website plays back recorded episodes in a 3D view, because the physics is chaotic: the same brain walks differently on a different CPU, so we show what we verified instead of re-simulating it live.

---

## Map of the system

```
            TRAINING (your laptop)                         DEMO (Render)
 main.py / app.py training thread                  browser loads index.html
   Population (src/evolution.py)                     fetches static/replays/best.json
     ├─ 50 × CreatureBrain (src/model.py)            plays frames with requestAnimationFrame
     └─ multiprocessing.Pool                         draws them with Three.js (3D) or canvas (2D)
          evaluate_genome() × 50 in parallel
            CreatureEnv (src/environment.py)       Flask (app.py) only serves files + small JSON
              Pymunk physics, reward, falls        TRAINING_ENABLED=0 → no training, no live sim
   saves models/*.npz, logs/history.csv
 scripts/record_replay.py → static/replays/*.json  ← the bridge between the two
```

---

## Module 1: Python and NumPy arrays

**Read:** `src/model.py` (25 lines)

**Concepts:**
- arrays and shapes
- matrix-vector multiply (`@`)
- broadcasting (`+ b1`)
- `float32` vs `float64`
- `np.random.normal`, `rand`, `randn`
- boolean masks
- `np.where`
- in-place ops (`+=`, `p[:] = ...`) vs making copies

**Study:**
- NumPy, "the absolute basics for beginners": https://numpy.org/doc/stable/user/absolute_beginners.html
- NumPy broadcasting: https://numpy.org/doc/stable/user/basics.broadcasting.html

**You can explain it when you can answer:**
1. `w1` has shape `(16, 15)`. What shape is `w1 @ x` when `x` has shape `(15,)`? Why is `w1` stored as (out, in) and not (in, out)?
2. In `Population.mutate`, why does `param += ...` change the brain, but `param = param + ...` wouldn't?
3. Why does `clone_model` need to copy the arrays? What bug would you get if two brains shared the same `w1` array?

---

## Module 2: The brain (a tiny neural network)

**Read:** `src/model.py`, and `get_state()` in `src/environment.py`

**Concepts:**
- neurons, weights and biases
- a layer is `activation(W @ x + b)`
- `tanh` squashes values to (-1, 1)
- a forward pass
- the parameter count: 15×16 + 16 + 16×4 + 4 = **324**
- why input scaling matters (`* 0.1`, `* 0.01` in `get_state`)

There is **no backpropagation** here. The weights are only ever changed by evolution.

**Study:**
- 3Blue1Brown, *Neural networks* chapter 1, "But what is a neural network?" (YouTube). Watch chapter 1 only; the later chapters cover backprop, which you don't use.
- Michael Nielsen, *Neural Networks and Deep Learning*, chapter 1: http://neuralnetworksanddeeplearning.com/chap1.html

**Questions:**
1. Walk through `get_action(state)` line by line. What does each of the 4 outputs control?
2. Why `tanh` on the output layer specifically? (Hint: `step()` multiplies each action by 10 to get a motor speed.)
3. The torso velocity is multiplied by `0.01` before going into the network. What happens to a `tanh` neuron if an input is 300 instead of 3?
4. Why did we replace PyTorch with about 10 lines of NumPy? What was PyTorch giving us that we didn't need?

---

## Module 3: Physics simulation

**Read:** `create_creature()`, `create_ground()`, `terrain_changing()` and `step()` in `src/environment.py`

**Concepts:**
- rigid bodies (mass, moment of inertia)
- shapes vs bodies
- friction
- gravity: `(0, 900)`, where y points *down* because it uses pygame coordinates
- constraints:
  - `PivotJoint`: the hip hinge
  - `RotaryLimitJoint`: legs are limited to ±60°
  - `SimpleMotor`: drives a joint at a target angular speed
- collision filtering: `ShapeFilter(group=1)`, so the creature's own parts don't collide with each other
- the fixed timestep `space.step(3/60)`: each step is 0.05 s of simulated time, so 600 steps is 30 s

**Study:**
- Pymunk docs, overview and tutorials: https://www.pymunk.org/en/latest/overview.html
- Pymunk constraints (PivotJoint, SimpleMotor, RotaryLimitJoint): https://www.pymunk.org/en/latest/pymunk.constraints.html
- Glenn Fiedler, "Fix Your Timestep!": https://gafferongames.com/post/fix_your_timestep/

**Questions:**
1. What does `motor.rate = action * 10` physically do to a leg? What does `max_force = 150000` limit?
2. Why does the timestep size matter? What might happen if you changed `3/60` to `1/60`, or to `10/60`?
3. Look at `terrain_changing()`. The second segment is called "the downhill", but it goes *up*. Why? (In pygame coordinates, smaller y is higher.) **This is your first exercise below.**

---

## Module 4: The agent loop (state, action, reward, termination)

**Read:** `step()` and `get_state()` in `src/environment.py`, and `evaluate_genome()` in `src/evolution.py`

**Concepts:**
- **The loop** (the same idea that reinforcement learning uses): observe → act → the world steps → reward.
- **Observation design:**
  - 5 torso values
  - 4 legs × (angle relative to the torso, angular velocity)
  - a 2-value clock
- **Reward shaping:** `displacement * 2` per step, so total fitness is about 2 × distance. The best model's 997 fitness means about 500 px walked.
- **Penalties:** for standing still, for tilt and energy (depending on the fitness mode), and −100 for falling.
- **Termination:** after 600 steps, a fall, or being stuck (60 still steps).
- **Central Pattern Generator (CPG):** `CPG_CLOCK` is a sine/cosine table with a 30-step period (1.5 s of sim time). It lets a memoryless network produce a rhythm.
- **Specification gaming:** evolution optimises exactly what you wrote, not what you meant.

**Study:**
- Lehman et al., *The Surprising Creativity of Digital Evolution* (2018), a collection of stories about evolution exploiting simulators: https://arxiv.org/abs/1803.03453
- DeepMind, "Specification gaming: the flip side of AI ingenuity": https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Ijspeert, *Central pattern generators for locomotion control in animals and robots: a review* (Neural Networks, 2008). Read the introduction and the robotics section.
- OpenAI Spinning Up, "Key Concepts in RL" (for vocabulary; you're not using RL): https://spinningup.openai.com/en/latest/spinningup/rl_intro.html

**Questions:**
1. The brain has no memory: the same input always gives the same output. So how can it walk in a rhythm? What would happen without the `cpg_sin`/`cpg_cos` inputs?
2. Why is `fallen` punished with −100 rather than just ending the episode?
3. Name one way a creature could score well without "walking" that this reward allows. How would you stop it?
4. `ground_termination` mode and `baseline` mode gave `best_model` the same score. Why? (Hint: what does `best_model` never do?)

---

## Module 5: Evolution (the genetic algorithm)

**Read:** `Population` in `src/evolution.py`, and the training loop in `train()` in `main.py`

**Concepts:**
- **Population and fitness**
- **Truncation selection with elitism:** the top 5 survive unchanged.
- **Uniform crossover:** each weight comes from parent 1 or parent 2 with a 50/50 chance.
- **Gaussian mutation:** each weight has a 15% chance of getting noise ×0.35.
- **Exploration vs exploitation:** mutation strength against selection pressure.
- **Diversity:** the average per-weight standard deviation across the population. Watch it fall as the population converges.
- **Why a GA is used here instead of gradients:** the fitness is a black box. You can't take the derivative of a physics engine (easily).

**Study:**
- Karl Sims, *Evolving Virtual Creatures* (SIGGRAPH 1994). This is the original version of your project. Read the paper and watch the video.
- David Ha, "A Visual Guide to Evolution Strategies": https://blog.otoro.net/2017/10/29/visual-evolution-strategies/
- Such et al. (Uber AI), *Deep Neuroevolution* (2017): a simple GA like yours, competitive with RL: https://arxiv.org/abs/1712.06567
- Salimans et al. (OpenAI), *Evolution Strategies as a Scalable Alternative to RL* (2017): https://arxiv.org/abs/1703.03864

**Questions:**
1. Trace `evolve()`: how many of the next 50 brains are unchanged elites, and how is each of the other 45 made?
2. What would happen with `mutation_scale = 3.0`? With `0.001`?
3. What does a falling `Diversity_Score` in `logs/history.csv` tell you? Why is diversity reaching zero bad?
4. Why does keeping the elites unchanged guarantee the best fitness never goes *down* from one generation to the next, in a deterministic simulation?

---

## Module 6: Parallelism

**Read:** the `pool` usage in `Population.evaluate()` and `main.py`, and `evaluate_genome()`

**Concepts:**
- **Processes vs threads.** Python's GIL stops threads from running Python code in parallel, so CPU-heavy work needs processes.
- **Pickling:** a worker receives a `state_dict` of arrays, not a brain object. That's why `evaluate_genome` is a top-level function.
- **Embarrassingly parallel work:** each genome's rollout is independent. `evolve()` is cheap and runs serially.

**Study:**
- Python `multiprocessing` docs (Pool, and the "Programming guidelines" section): https://docs.python.org/3/library/multiprocessing.html
- Real Python, "Speed Up Your Python Program With Concurrency": https://realpython.com/python-concurrency/

**Questions:**
1. Why can't you just use `threading` to evaluate 50 brains at once?
2. Why does `evaluate_genome` build a fresh `CreatureEnv` inside the worker instead of receiving one?
3. `tests/test_parallel.py` checks that parallel results match sequential ones. Why would they ever differ?

---

## Module 7: Floating point, chaos and determinism

This is the most surprising thing in the project.

**What happened:** after replacing PyTorch with NumPy, the *same weights* walked differently. `best_model` went from 1211 to 997 fitness. The weights were copied bit-for-bit. The only difference was the order of additions inside the matrix multiply, which changed values by about 0.000001. Even PyTorch alone gave different walks with a different CPU kernel.

**Concepts:**
- floating-point rounding: `(a + b) + c != a + (b + c)`
- **chaotic systems:** tiny differences in starting conditions grow exponentially over time (contacts and falls amplify them)
- BLAS libraries (MKL, OpenBLAS) choose different code paths on different CPUs
- determinism (same machine, same result) vs reproducibility (any machine)

**Study:**
- Glenn Fiedler, "Floating Point Determinism": https://gafferongames.com/post/floating_point_determinism/
- "What Every Computer Scientist Should Know About Floating-Point Arithmetic": read the first part only (rounding error)
- Veritasium, "Chaos: The Science of the Butterfly Effect" (YouTube), for intuition

**Questions:**
1. Why does the website play a *recorded* replay instead of simulating the brain on the server?
2. If an evolved gait breaks when a weight changes by 0.000001, what does that say about the gait? (It's fragile: it exploits one exact trajectory.)
3. The old "universal best" scored 1155 on 31 Aug, but only about 7 under PyTorch on today's terrain, before NumPy was involved at all. What changed? (Check `git log -p src/environment.py`.) A fitness score is only valid for the exact environment that produced it.
4. **Exercise idea:** evaluate each genome 3 times with tiny random noise on its starting pose, and use the *average* as fitness. Why would that evolve sturdier walkers?

---

## Module 8: The web server

**Read:** `app.py` (routes, `training_state`, `stream_simulation`, the 403 guards) and the `CMD` line in `Dockerfile`

**Concepts:**
- **HTTP basics:** routes, GET vs POST, status codes (200, 400, 403), JSON APIs.
- **Flask** turns a function into a route.
- **A background thread for training,** with the status shared through a dict.
- **Server-Sent Events (SSE):** a response that never ends and pushes `data:` lines.
- **gunicorn workers and threads:** `--workers 1 --threads 8` means at most 8 requests at the same time. Each open SSE stream holds one thread *forever*. That's why the public demo returns 403 for `/api/stream/sim`.
- **Environment variables as configuration:** `PORT`, `TRAINING_ENABLED`, `TRAINING_WORKERS`.
- **Caching headers:** `max-age=300` on replays.

**Study:**
- Flask quickstart: https://flask.palletsprojects.com/en/stable/quickstart/
- MDN, "Using server-sent events": https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events
- MDN, HTTP response status codes: https://developer.mozilla.org/en-US/docs/Web/HTTP/Status
- gunicorn design (workers vs threads): https://docs.gunicorn.org/en/stable/design.html

**Questions:**
1. Trace what happens from clicking "Run Evolution" to a new row appearing in `logs/history.csv`.
2. Why could 8 visitors take the whole site down if `/api/stream/sim` were public?
3. Why is `training_enabled` read from an environment variable instead of being hard-coded?

---

## Module 9: The browser side (rendering, 3D, animation)

**Read:** in `templates/index.html`, `loadReplay`, `replayPose`, `frameLoop`, `init3D` / `draw3D`, `draw2D`, and the polling code at the bottom. Also read `scripts/record_replay.py`.

**Concepts:**
- **The DOM and `fetch()`.**
- **`requestAnimationFrame`:** the browser calls you once per screen refresh and passes in a timestamp.
- **Interpolation** between recorded frames, so 45 fps data looks smooth at 60 to 144 Hz.
- **The 2D canvas API.**
- **Three.js:** scene, camera, renderer, mesh = geometry + material, lights, shadows. Build meshes once; move them every frame.
- **Mapping 2D physics to 3D:** `x → x`, `y → -y`, `angle → -rotation.z`.
- **A real bug from this project:** the rAF timestamp can be a few milliseconds *older* than the moment the replay finished loading. Elapsed time went negative, the code read `frames[-1]`, threw an error, and the animation loop died at step 0, but only on the live site, where network delay changed the timing. The fix was `Math.max(0, now - playT0)`. (See PR #2 on GitHub.)

**Study:**
- MDN, `requestAnimationFrame` (read the part about the timestamp): https://developer.mozilla.org/en-US/docs/Web/API/Window/requestAnimationFrame
- MDN Canvas tutorial: https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API/Tutorial
- Three.js manual, Fundamentals → Scenegraph → Lights → Shadows: https://threejs.org/manual/#en/fundamentals

**Questions:**
1. Each replay frame is 20 numbers. What are they? (See `FIELDS` in `record_replay.py`.)
2. Why interpolate between frames instead of just showing frame `floor(t)`?
3. Why did we only catch this bug on the deployed site, even though the race can happen anywhere?

---

## Module 10: Shipping it (Docker, Render, accessibility)

**Read:** `Dockerfile`, `.dockerignore`, `requirements.txt`, the "Deploy (Render)" section of `README.md`, and both plans in `docs/plans/`

**Concepts:**
- **Docker:** an image is your app plus its dependencies, frozen. Each line of the Dockerfile adds a layer, and layers are cached.
- **Image size:** PyTorch alone was 478 MB, which is why removing it mattered.
- **What `.dockerignore` is for.**
- **Render's free tier:**
  - 512 MB RAM and 0.1 CPU
  - sleeps after 15 minutes idle
  - no persistent disk, so trained models are wiped on each deploy
  - 5 GB/month bandwidth
- **Deploy-on-merge:** merging to `main` *is* deploying to production.
- **Accessibility basics:** labels, keyboard focus, `aria-live` regions, colour contrast (WCAG AA is 4.5:1).

**Study:**
- Docker, "Get started" (parts 1-3): https://docs.docker.com/get-started/
- Render docs: free instances (https://render.com/docs/free) and Docker deploys (https://render.com/docs/docker)
- WebAIM contrast checker: https://webaim.org/resources/contrastchecker/
- MDN, ARIA live regions: https://developer.mozilla.org/en-US/docs/Web/Accessibility/ARIA/ARIA_Live_Regions

**Questions:**
1. Why does the Docker `CMD` use `sh -c` and `${PORT:-8000}` instead of a fixed port?
2. Why does the public site run with `TRAINING_ENABLED=0`? Give two separate reasons.
3. You retrained and got a better walker. What do you need to run and commit before the website shows it?

---

## Exercises (do these yourself; that's how it sticks)

Do them roughly in this order. Each one is small, and each forces you to understand one module.

1. **Fix the downhill** (Modules 3 and 4). The second slope in `terrain_changing()` should descend, so swap its y formula. Re-run `python scripts/record_replay.py` and look at the terrain in the 3D view.
2. **Write the forward pass in JavaScript** (Modules 1, 2 and 9). Load `best_model.npz` (or export its weights to JSON with a 5-line Python script) and reproduce `get_action` in the browser console. Check your 4 outputs against Python's for the same input.
3. **Plot diversity against generation** from `logs/history.csv` (Module 5). Explain the shape of the curve.
4. **Remove the CPG inputs** (set them to 0) and train for 30 generations (Module 4). Compare the best fitness with and without them, and explain the difference.
5. **Make the fitness robust to noise** (Module 7): evaluate each genome over 3 slightly perturbed starts and average the results. Do the evolved gaits survive the NumPy/PyTorch kind of difference better?
6. **Add a real goal** (Modules 3 and 4): end the episode with a bonus when `torso.position.x >= 2900`, and make `record_replay.py` mark it.

---

## Then: real 3D (v3/v4)

Once you can answer the questions above, real 3D physics is the same project with a bigger world:

- **Engine:** MuJoCo (`pip install mujoco`). Start from Gymnasium's `Ant` quadruped. You describe the robot in XML (MJCF) instead of Python.
- **Brain:** about 29 inputs (3D orientation, 8 joints) and 8 outputs (a hip and a knee per leg). Same genetic algorithm, about 1,000 weights.
- **Fitness:** also penalise sideways drift and rolling, because in 3D the creature can fall over in ways a 2D one can't.
- **Same-engine rule:** replay in the same engine you trained in. MuJoCo also has WebAssembly builds for the browser.

**Study before starting:**
- MuJoCo docs, Overview and the MJCF modeling guide: https://mujoco.readthedocs.io/
- Gymnasium Ant environment: https://gymnasium.farama.org/environments/mujoco/ant/
- The MuJoCo Python tutorial notebook, linked from the MuJoCo docs
