# Step 2 — Knees v2: stop punishing the crouch

## What step 1 showed

Every kneed walker "fell" in 31–44 steps. We looked at why. The run ended when the torso touched the ground, while the body was still level (tilt 0.07–0.78 rad). The knees had bent to 116–128°, past their 90° limit, because the knee motor was strong enough to overpower the limit joint.

So the creatures were not tipping over. They were crouching, and our rules counted a crouch as death.

## What we change

1. **The fall rule.** Touching the ground with the torso now costs 2 points per step instead of ending the run. Tipping over still ends it. A creature can crouch, but lying on the ground never pays.
2. **Knees that stay inside their limit.** We fix the knee so it no longer bends past its limit.
3. **A third body, inspired by Boston Dynamics' Spot.** Its knees bend 14–150°, it stands slightly bent at 30°, and its hips swing ±90°. Spot's real knee range is 14–160°, and it sits with its body 191 mm off the floor (Spot SDK, "About Spot" page).

## Why Spot never straightens its knee fully

A fully straight knee has almost no leverage to start bending. The motor pushes along the leg, not across it. A small built-in bend gives the motor something to pull against, so we give our Spot-style knees the same.

## The test

We compare three bodies: `straight`, `knees` and `knees_spot`. All three train under the new fall rule, with 10 seeds each. Training runs on GitHub Actions, one cloud machine per run.

Absolute distances are not comparable with step 1, because it ran on a different machine. Only comparisons within this experiment count.

## Predictions, written down before training

- **H8:** `knees` beats `straight` on held-out test distance.
- **H9:** `knees_spot` beats plain `knees`.
- **H10:** `knees` falls less often than in step 1, where 100% fell.

## Known risk

The dive might still pay. A creature that dives forward and then stops can still collect distance before the run ends as "stuck". If that happens, "settle before scoring" is the next step.
