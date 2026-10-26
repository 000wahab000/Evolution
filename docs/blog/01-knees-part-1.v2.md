# Knees made my robot fall every time. The bug was in my rules.

<!--
Alt titles:
- What a failed knee experiment taught me about reward design
- Separating what I measured from what I guessed: a failed evolutionary robotics experiment
- Evolution found a loophole in my fall rule
- Knees, a dive, and a bug
-->

I gave my simulated four-legged robot knees, expecting better handling of slopes. Instead, every kneed champion fell on every test course.

![Straight and kneed champions on a test course](images/dive-replay.gif)
> **Why it's here:** The result before the argument: the whole run on a held-out test course. **How to read it:** Straight is blue (top) and knees orange (bottom); counters and ruler are in px, so watch the lower lane end. **Why it matters:** It shows what 100 of 100 falls looks like in motion.

What I found points at my own rules and a bug, not knees as such. Two of those are in the code. The third, that the reward makes diving pay, is still a guess.

## The experiment

Evolution-1 is a 2D Pymunk quadruped. Its brain is an MLP with 16 hidden units, evolved by a genetic algorithm (population 50, 100 generations, 25,000 training episodes per run). Training courses are random slopes up to 4%, fresh each generation. Every 1,250 episodes, the GA's best is scored on 5 validation courses and kept if it beats the champion.

The final score comes from 10 held-out test courses with 5-8% slopes, which neither body saw in training or selection.

I compared two bodies. The kneed leg has the same total length and mass.

| | Straight | Knees |
|---|---|---|
| Leg | one 40 px stick, mass 2 | 20 px thigh + 20 px shin, mass 1 each |
| Motors | hip only (max force 150,000) | hip + knee (knee 75,000) |
| Inputs / outputs | 15 / 4 | 23 / 8 |
| Weights | 324 | 520 |

The knee motor gets half the hip force, following Sims 1994 §3.3, where strength scales with cross-section.

Reward per step is +2 times forward torso movement, and -0.1 for standing still after step 45. A fall costs -100, once. An episode ends on a fall, when the walker is stuck (still for 60+ steps, no -100), or at 600 steps.

![The reward code used in the experiment](images/fitness-reward.png)
> **Why it's here:** The numbers behind the fitness claims. **How to read it:** Highlighted: the fall penalty (line 28), forward reward (194) and still penalty (215). **Why it matters:** A fall costs 100 and a pixel pays 2, so 50 px of progress buys back a fall, and the dive hypothesis rests on that trade.

The fall rule matters most. A "fall" is any of these: tilt above 1.2 rad, the torso sinking 25 px below ground, or the lowest corner of the torso box reaching ground height. That last check is purely geometric, so a level but low torso counts as a fall. Optional tilt and energy terms were off (default `baseline` mode).

![The fall rule used in the experiment](images/fitness-fall-rule.png)
> **Why it's here:** The code behind the rule above, and my main suspect. **How to read it:** Line 205 flags the lowest torso corner reaching ground height; lines 219 and 221 join it to tilt and sinking with `or`, and line 223 ends the episode. **Why it matters:** Any one condition kills the walker, so tilt is not required.

## How I built the knees

Each leg is a thigh and a shin, two boxes of 20 × 10 px with mass 1, friction 1.2 and collision group 1.

The hip pivots at the torso anchor, with a ±60° limit and motor force 150,000. The knee pivots between the thigh's bottom and the shin's top. Its limit is −90° to 0, so it bends one way, and its `SimpleMotor` gets half the hip force, 75,000.

![The knee-building code](images/knee-code.png)
> **Why it's here:** The knee code as it was during the experiment, so you can check my description. **How to read it:** Lines 134-137 are the hip; the highlighted 138-141 are the knee (pivot, −π/2 to 0 limit, motor 75,000). **Why it matters:** Line 139 shows the limit is a soft `RotaryLimitJoint`, the setup for the bug below.

The brain has 8 outputs, hips first and then knees. The step function scales each output by 10 for a hip and by −10 for a knee. −1 bends the knee, putting the foot forward of it, and +1 straightens it.

Each knee adds two inputs: the relative angle (shin minus thigh) and the shin's angular velocity times 0.1. Four knees add 8, so 15 inputs become 23. Weights:

```
straight: 16×15 + 16 + 4×16 + 4  = 324
knees:    16×23 + 16 + 8×16 + 8  = 520
```

The limit is a soft `RotaryLimitJoint`. It corrects only after it has been crossed, which is why the motor could push the knee past 90°. That is the [bug described below](#what-i-measured-what-i-found-what-im-guessing).

Knees is an opt-in flag, off by default, so the old creature is unchanged.

## Results

I ran 10 seeds per body. A seed is one random starting point for a training run. I report the IQM, the mean of the middle 50% of seeds, so one odd run cannot swing it. Brackets are 95% bootstrap confidence intervals (10,000 resamples). Cliff's δ is how often one group beats the other: -1 means knees are always lower, 0 no difference.

| Test courses | Straight | Knees |
|---|---|---|
| Distance, IQM [95% CI] | 394 px [190, 718] | 160 px [148, 187] |
| Fall rate | 17/100 episodes | 100/100 episodes |
| Seeds averaging ≥300 px | 6/10 | 0/10 |
| Cliff's δ, distance (knees vs straight) | -0.52 | |
| p, distance (Mann-Whitney, Holm) | 0.054 | |
| Cliff's δ, falls | +1.00 | |
| p, falls (Holm) | ≈ 0.0001 | |

The 100 episodes per body are 10 seeds times 10 courses. A random kneed seed beats a random straight seed 24% of the time.

![Final test distance per seed](../evolving-virtual-creatures/01-knees/screenshots/F2-final-test-distance.png)
> **Why it's here:** The spread behind the IQM. **How to read it:** One dot per seed, distance in px; knees are blue and straight orange, each with train (light, left) and test (dark, right), and the bar is the IQM with its 95% bootstrap interval. **Why it matters:** Knee dots sit within about 140-250 px, while straight dots run from about 140 to over 1,100 px, hence the wide interval and p of 0.054.

The kneed champions also fell on every easy course: the train-like evaluation courses in each seed's `final_test.json` show a fall rate of 1.0 for all 10 kneed seeds. The gap is large but uncertain: the straight interval runs from 190 to 718 px, and the two-sided p of 0.054 does not clear 0.05.

Deviation: My pre-registration said 30 seeds per variant. I ran 10 to save laptop time (34 core-hours, and the CPU hit 87 °C), and I did not record that as a deviation at the time. With 10 seeds, a p of 0.054 is underpowered, not evidence of "no difference".

![Learning curves on test courses](../evolving-virtual-creatures/01-knees/screenshots/F1-learning-curves.png)
> **Why it's here:** When knees stop improving. **How to read it:** X is training episodes and y is champion test distance in px; each line is the IQM over 10 seeds with its 95% bootstrap band, knees a solid blue line and straight a dashed orange one. **Why it matters:** The blue line climbs to about 130 px by 6,000 episodes (a quarter of the budget) and then creeps to about 160, while orange keeps climbing and its band widens.

From each seed's `history.csv`, the median best training score at 20/40/60/80/100% of the budget was 246/427/516/594/782 for straight and 112/153/182/172/174 for knees. The kneed median test distance was 154 px at 80% and 162 px at 100%.

## What I measured, what I found, what I'm guessing

**Found in the code.** I inspected the code and ran a diagnostic replay of kneed seeds 0, 1 and 4. Two things came out of it.

First, the fall events had small tilt, between 0.07 and 0.78 rad. These were low torsos, not tip-overs: the geometric check from the experiment section fired.

![Replay at the end of a kneed episode](images/replay-course-a.png)
> **Why it's here:** The frozen evidence frame from the replay GIF. **How to read it:** This is the last frame of the two-lane view, straight in blue (214 px) on top and knees in orange (263 px) below with the red FELL badge. The kneed torso is tilted about 40-45°, roughly 0.78 rad. **Why it matters:** That is below the 1.2 rad (69°) threshold, so the geometric check ended the episode, not a tip-over.

Second, the knees bent to 116-128°, past the 90° limit. The limit is a soft joint that corrects only after it is violated, and the 10 rad/s motor outran it. This was my bug.

**Hypothesis.** Here is why I think the GA settles on a dive. No experiment has isolated this, so the arithmetic is illustration, not proof.

With +2 per pixel and one -100 fall, a 200 px lunge earns about +400 minus 100, so about +300. The kneed champions' per-seed test means were 138-245 px. That beats an early clumsy walker, so the GA can settle there.

I underplayed a second warning. The analysis reruns each champion at half the physics timestep, and flags it as possibly exploiting the simulator if distance drops by over 30%. 8 of 10 straight seeds and 10 of 10 kneed seeds were flagged, with an IQM drop of about 72-73% (73.2% knees, 72.1% straight). Both gaits depend heavily on step size.

Straight legs also fell, 17/100 under the same rule, so the rule alone does not explain 100/100. A bent knee lowers the torso toward the geometric trigger; a stiff stick cannot.

## What this does not show

Knees are not shown to be harmful in general: this was one design under one fall rule, with a limit that did not hold.

The dive is only a hypothesis, not shown to cause the gap, and I have not separated the fall rule from the limit bug. Absolute distances are partly simulator-dependent.

The bundled change (body, 4 motors and a bigger brain) limits both predictions. H6, distance: I cannot say which part caused the gap. H7, falls: the 100/100 result is about this whole package under this fall rule, not about knees alone.

## Lessons from this case

- **Test what ends an episode on its own.** A level torso can be "dead" under a geometric rule, and I only saw that after replaying the falls.
- **Check that limits hold under full motor power.** My 90° limit let the knee reach 128°.
- **A fitness number says "worse", never "why".**
- **One change can be three.** "Add knees" was a body, four motors and a bigger brain, so I cannot say which mattered.
- **Log deviations when you make them.** I cut 30 seeds to 10 and did not write it down.

## What's next (Part 2)

This is planned, not run. Part 2 trains straight, knees, and a Spot-inspired knee (14-150°, standing bent 30°), all with a new fall rule: -2 reward per step while the torso touches the ground, instead of ending the episode. Knee limits are enforced in code after every physics step. Training moves to GitHub Actions.

Part 2 fixes the rule and the limit together, so it will not say which mattered. Two runs would separate them: (a) the kneed body with the enforced limit under the old fall rule, which I would want to run too, and (b) the straight body under the new penalty rule, which is already in Part 2's plan.

## How I built this

I designed the experiments, chose what to test from the papers I read, and made the calls on fairness and results. AI coding agents (Claude) wrote most of the code from my plans; I reviewed the changes, and automated tests checked every run. Unit tests check that the straight creature's sensors and config hash are unchanged. The mistakes here, including the knee bug, are mine.

## Code and data

Repo: [github.com/000wahab000/Evolution](https://github.com/000wahab000/Evolution). Experiment config: [`experiments/exp3_knees.json`](https://github.com/000wahab000/Evolution/blob/main/experiments/exp3_knees.json). Hypotheses: [`experiments/HYPOTHESES.md`](https://github.com/000wahab000/Evolution/blob/main/experiments/HYPOTHESES.md). Fall and reward code: [`src/environment.py`](https://github.com/000wahab000/Evolution/blob/main/src/environment.py).
Full numbers: [`results.md`](https://github.com/000wahab000/Evolution/blob/main/docs/evolving-virtual-creatures/01-knees/results.md).

The raw per-seed results and the diagnostic script will be published with the repo before this post goes live.

<!-- TODO: check links resolve after merge -->
