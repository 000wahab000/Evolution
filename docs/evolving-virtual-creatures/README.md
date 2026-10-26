# Evolving Virtual Creatures — applying Sims (1994) to Evolution-1

Paper: Karl Sims, "Evolving Virtual Creatures", SIGGRAPH '94, pp. 15–22 (Thinking Machines Corporation).

## The paper

Sims uses a genetic algorithm to evolve both the body and the brain of 3D creatures. A genetic algorithm keeps a population of candidates, scores them, and breeds new ones from the better scorers with small random changes. In the paper, bodies are built from jointed parts, and each joint has its own muscle (an "effector"). The creatures evolve to swim, walk, jump and follow a light.

## Evolution-1

Evolution-1 is a 2D quadruped: a torso with four rigid legs. Its brain is a tiny neural network with 324 adjustable weights. We evolve those weights with a genetic algorithm and no backpropagation. The result learns to walk right without falling over.

## Roadmap

Each step borrows one idea from the paper.

1. Knees: split each leg into thigh + shin with a powered knee (sec. 2–3.3). See [01-knees](./01-knees/).
2. Knees v2: penalty fall rule, knee-limit fix, Spot-inspired knees (Spot SDK; Sims sec. 5.2). See [02-knees-v2](./02-knees-v2/).
3. Foot contact sensors: the brain feels when a foot touches the ground (sec. 3.1).
4. Score on several courses: average fitness over several random courses so the walker can't memorise one (sec. 5.4).
5. Settle before scoring: let the body come to rest before measuring distance (sec. 5.2).
6. Size-scaled mutation: scale each weight's mutation chance by 1/genome size so every child gets about the same number of changes, whatever the brain size (sec. 6.1). Today every weight has a fixed 15% chance, so the 520-weight kneed brain gets about 78 changes per child against about 49 for the 324-weight straight brain.
7. Dashboard: show the new bodies in the browser.

Later ideas: fitness-proportional breeding (sec. 6), and evolving the body shape itself (sec. 2).

## Backlog beyond the paper

Ideas collected from outside suggestions, not from Sims; each has a status showing what Evolution-1 already has.

### Quick wins (1–3 hours each)

| # | Idea | Why | Status in Evolution-1 |
|---|------|-----|-----------------------|
| 7 | Energy penalty in fitness (subtract joint effort from score) | Stops walkers that thrash for distance. | Partly exists: `--fitness-mode energy_penalized` subtracts sum \|action\|, not torque × velocity. Not used in the research runner yet. |
| 8 | Learning-curve plot in the README | Shows that training actually works. | Plots exist (`scripts/plot_logs.py`, research figure F1); not yet embedded in the top-level README. |
| 9 | Ablation table (with/without CPG, sensors, multi-course) | Shows why each piece matters. | CPG ablation exists as `experiments/exp1_cpg.json`; knees is step 1 here. Sensors and multi-course rows come with roadmap steps 2 and 3. |
| 10 | Short GIF of the best walker at the top of the README | People see the result before reading. | Exists (`docs/media/walking.gif`); refresh after a better walker. |
| 11 | Single config file for body, GA and terrain params | Makes runs reproducible. | Experiments already use JSON configs (`experiments/*.json`); body params are still hard-coded in `src/environment.py`. YAML not planned (the project chose JSON). |

### Medium (1–3 days each)

| # | Idea | Why | Status in Evolution-1 |
|---|------|-----|-----------------------|
| 12 | Symmetry constraint (mirror left/right leg weights) | Gives a smaller search space and is biologically motivated. | Not started. Test it as an ablation. |
| 13 | Push recovery (random impulse mid-run) | Separates walking from robust walking. | Not started. |
| 14 | Curriculum: flat → hills → complex | Introduces difficulty step by step. | Not started (training courses are random from the start). |
| 15 | PPO baseline on the same body | Compares evolution against standard RL. | Not started. A Gymnasium wrapper already exists (`src/gym_env.py`), so stable-baselines3 PPO can use the same Pymunk body without MuJoCo. |
| 16 | Domain randomization (friction, mass, joint damping) | Makes the walker robust to physics differences (sim-to-real). | Partly exists: training randomizes friction ×0.9–1.1 and starting pose. Mass and damping not randomized; ranges are narrow. |
| 17 | Recurrent brain (LSTM/GRU) | Memory lets the gait adapt instead of following a fixed clock. | Not started. |

### Big (1–2 weeks)

| # | Idea | Why | Status in Evolution-1 |
|---|------|-----|-----------------------|
| 18 | Body co-design (evolve leg lengths and mass with the brain) | It is Sims' core idea and an active research direction. | Listed as a later idea in the roadmap above (Sims sec. 2). |
| 19 | MuJoCo 3D port | Gives a real 3D quadruped. | Not started; long-term. |
| 20 | One-page tech report (PDF) | A short write-up of method, ablations, results and limits. | The per-step `abstract.md` / `results.md` files are the raw material. |
| 21 | Sim-to-real section | Explains how the controller could run on a real robot. | Not started; one README paragraph is enough. |

## How each step is documented

Each step gets a folder with:

- `abstract.md`: what we change and why.
- `results.md`: the numbers, old versus new.
- `screenshots/`: charts.
