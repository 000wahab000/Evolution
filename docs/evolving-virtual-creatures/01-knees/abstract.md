# Step 1 — Knees

## What we're changing

Today each leg is one stiff 40-pixel stick with one motor at the hip. With knees, each leg becomes a 20 px thigh and a 20 px shin joined by a knee. The knee bends only one way (0 to 90°, with the foot swinging backward) and has its own motor.

Total leg length and mass stay the same, so standing height doesn't change. The knee motor is half as strong as the hip motor. This follows Sims' rule that a joint's strength scales with the thickness of the parts it joins: the hip joins a 20 px thick torso, while the knee joins two 10 px parts.

## Why it might help

A stiff stick can't lift its foot over a rise, so on slopes it scrapes. A knee lets the foot fold up and clear the ground.

An older saved model fell on 5 of 6 new courses. It was trained on one fixed course, though. Some of that failure is probably memorising the course, not only stiff legs. So that result is a hint, not proof.

## What the brain gets

The brain now also senses each knee's angle and speed (15 to 23 inputs) and drives 4 more motors (4 to 8 outputs). It grows from 324 to 520 adjustable numbers.

## How we keep the test fair

- Both creatures train from scratch.
- They use the same random seeds, the same number of training episodes, and the same random gentle training courses (slopes up to 4%).
- Each run's best walker is then scored on steeper courses (5–8%) that neither creature saw.
- We compare distance travelled and how often each falls, across many seeds, with confidence intervals (a range that shows how much the number could move by chance).

## Known limits

- The kneed brain is bigger. With the same training budget it may still be improving when training stops. We will show the learning curves. If the kneed walker is still climbing, we will call the result inconclusive, not "knees don't help".
- All four knees bend the same way. Real quadrupeds bend front and hind legs differently.
- Knee strength is fixed in advance. It may be tuned only before the real run, never after seeing test results.

## What would count as "knees help"

Kneed walkers go further and fall less on the unseen steep courses, and the difference holds up across seeds. That means the confidence intervals don't overlap, or the pre-registered statistical test (written down before training) says the hypothesis is supported.
