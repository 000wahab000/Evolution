---
title: Research Direction - Making Evolution-1 Research-Grade
date: 2026-10-20
status: findings (reviewed)
---

# Research Direction: Making Evolution-1 Research-Grade

Findings from a research pass on how evolutionary-robotics / neuroevolution research is done and published, plus an independent reviewer critique of those findings. This is the input to the implementation plan in `docs/plans/`.

## Where the project is today

- 2D Pymunk quadruped, 15→16→4 tanh MLP (324 params), GA with top-5 elitism, uniform crossover, Gaussian mutation (rate 0.15, scale 0.35), CPG sin/cos inputs.
- **Not research-grade yet:**
  - No seeding: every result is a single run.
  - One rollout per genome on one fixed terrain, so selection rewards overfitting that terrain.
  - No baselines or ablations; the "CPG is needed" claim is untested.
  - The "downhill" terrain reuses the uphill formula (known bug).
  - Reward weights are hand-tuned and unjustified.
  - Physics step `space.step(3/60)` (~50 ms) is coarse, so gaits may exploit integrator error.

## What the field rewards

- **Rigor over novelty.** Many seeds (20-30), confidence intervals, equal evaluation budgets, ablation tables, and a held-out test set are what make a small study publishable.
- **Fair comparisons:**
  - Budget is counted in **episodes**, not generations.
  - Every method gets **equal tuning effort** (or all run at library defaults).
  - Multiple comparisons are corrected (Holm).
  - Report IQM with bootstrap CIs (Agarwal et al. 2021, `rliable`).
- **Use existing baselines:**
  - `pycma` (or evosax) for CMA-ES and OpenAI-ES.
  - Random search (Mania et al. 2018) is a strong, cheap baseline.
- **Novelty warning.** "CPG vs no CPG" and "GA vs CMA-ES" alone are known results. The question has to be sharper and stated before running, e.g. *does the CPG's benefit survive terrain distribution shift, and does the network actually use it (entrainment)?*
- **Simulator credibility.** A 2D sim is acceptable at ALIFE, weak at GECCO. Check gaits at dt/2, and ideally confirm the headline result on one standard benchmark later.

## Who this benefits

The project only matters if someone can use what it produces:

1. **Students and self-taught researchers.** A small, fast, readable, fully reproducible evolutionary-robotics testbed, small enough to run on a laptop in hours, with a worked example of a fair method comparison.
2. **Researchers.** A measured answer, with error bars, to a narrow question (does a CPG clock help evolved controllers generalize?), plus every config and seed needed to rerun it.
3. **Dashboard visitors.** The live demo shows *measured* differences between variants, not just one cherry-picked walker.

Every metric therefore needs a figure that makes the difference between variants visible and honest (distributions and CIs, not single bests).

## Compute budget (measured)

- One 600-step episode with the MLP takes about 0.05 s.
- A run of pop 50 × 100 generations × 5 rollouts is about 20 CPU-minutes.
- 160 runs (2 experiments × 4 variants × 20 seeds) is about 50 core-hours, roughly 7 hours on 8 cores.
- Compute is not the bottleneck, so 30 seeds and longer runs are affordable.

## Venues (verified 2026-10-20)

| Venue | Format | Deadline | Notes |
|---|---|---|---|
| EvoStar 2027 (Mainz, 31 Mar-2 Apr) | full paper | 1 Nov 2026 | too close |
| EvoStar 2027 | Late-Breaking Abstract | 21 Mar 2027 | **realistic first target** |
| ALIFE 2027 (Prague, 19-23 Jul) | full paper | CfP not out (2026's was 30 Mar) | best fit for the study |
| GECCO 2027 (Krakow, 12-16 Jul) | 4-page poster paper | likely late Jan 2027 | same date as full papers |
| GECCO 2027 | 2-page Late-Breaking Abstract | likely ~Apr 2027 | |
| GECCO Student Workshop | workshop paper | per yearly call | feedback from mentors |

arXiv cs.NE needs an endorsement without an institutional email. The usual route is to ask an author of a paper you cite.

## Study path

1. **Ludobots** (Josh Bongard's evolutionary-robotics course run on Reddit, 10 assignments). Hands-on first.
2. **Neuroevolution** by Risi, Tang, Ha and Miikkulainen. Free online at neuroevolutionbook.com, with exercises. Plus the UT Austin CS378 Neuroevolution materials.
3. **Prerequisites:** linear algebra, probability, nonparametric statistics (Mann-Whitney, bootstrap). Read papers with Keshav's three-pass method.
4. **Core papers:**
   - Sims 1994, *Evolving Virtual Creatures*
   - Ijspeert 2008 on CPGs
   - Salimans 2017 on ES
   - Mania 2018 on random search
   - Mouret & Clune 2015 on MAP-Elites
   - Cully 2015, *Robots that can adapt like animals*
   - Agarwal 2021 on RL statistics
   - Mertan & Cheney 2025, arXiv 2508.17464. Brain-body co-optimization discards promising bodies but also reaches solutions a fixed body can't. Reading for a future morphology version, not reproducible on a fixed body.
5. **Community:** GECCO Student Workshop, ISAL/ALIFE student community, SIGEVO Summer School. Ask an author for feedback *before* submitting.

## Reviewer critique applied

An independent review of the first draft corrected the following:

- GECCO's 2-page track is Late-Breaking Abstracts; poster papers are 4 pages.
- The arXiv 2508.17464 "reproduce it" hook is not testable without evolvable morphology.
- The EvoStar LBA (21 Mar 2027) had been missed.
- Fair-comparison rules (episode budgets, equal tuning, Holm, IQM) were missing.
- Novelty needs a sharper, pre-stated hypothesis.
- Social-media quotes were not evidence of academic norms.
- Compute is cheap: quantified above.
