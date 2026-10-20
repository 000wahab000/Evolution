# Pre-registered hypotheses and analysis plan

Written before any full run. Commit and tag this file (for example `git tag prereg-v1`) before
the first full run, and put the tag in each experiment config's `hypotheses_tag`. REPORT.md
prints that tag. Results are reported whether they confirm or refute these hypotheses.

## Hypotheses

**Experiment 1, CPG ablation** (`experiments/exp1_cpg.json`, optimizer `ga`):
- **H1:** CPG-on reaches a higher held-out test distance IQM than CPG-zeros at an equal episode budget.
- **H2:** the CPG advantage persists or grows under terrain shift (test slopes 5-8%, steeper than the
  ≤4% used in training). Measured per seed as shift drop = control M1 − test M1. Supported if cpg's
  shift drop is lower than zeros'.
- **H3:** CPG-on outperforms the noise control, so the CPG benefit is not explained by uninformative
  extra input. Noise vs zeros is reported as a descriptive diagnostic, not a hypothesis.

**Experiment 2, optimizer comparison** (`experiments/exp2_optimizers.json`, CPG-on brain):
- **H4:** CMA-ES > GA in held-out distance at an equal episode budget. A non-significant result is
  reported as inconclusive.
- **H5:** every optimizer (ga, cmaes, openai_es) beats random search (sanity floor). Tested as H5a-c.

## Confirmatory contrasts

| Id | A | B | Metric (per seed) | Predicted |
|---|---|---|---|---|
| H1 | cpg | zeros | test distance (M1) | A > B |
| H2 | cpg | zeros | shift drop = control M1 − test M1 | A < B |
| H3 | cpg | noise | test distance (M1) | A > B |
| H4 | cmaes | ga | test distance (M1) | A > B |
| H5a-c | ga, cmaes, openai_es | random_search | test distance (M1) | A > B |

All other pairs are **exploratory** and are labelled as such in REPORT.md.

## Analysis rules

- **Tests:** two-sided Mann-Whitney U per contrast. Holm correction is applied only within each
  experiment's confirmatory set (exp1: H1-H3; exp2: H4, H5a-c), alpha = 0.05.
- **Verdict:** *supported* if the Holm-adjusted p < 0.05 **and** the 95% bootstrap CI (10,000
  resamples) of the IQM difference A − B excludes 0 in the predicted direction. *Not supported* if
  both hold in the opposite direction. Otherwise *inconclusive*.
- **Effect sizes:** Cliff's delta and P(A > B), reported for every pair.
- **Champion rule:** each generation's train-best genome is re-scored on validation terrains (train
  distribution, disjoint from test and control); the champion is the best of those so far. Test and
  control terrains are never used for selection. M1 is the final champion's mean distance on the
  10 test terrains.
- **Budget:** 25,000 training episodes per run. Only train rollouts count; validation, test and
  control evaluations are tracked separately.
- **Seeds:** 30 per variant (0-29). **Exclusions: none.** A crashed run is rerun, never dropped.
- **Optimizer settings:** pinned in `experiments/exp2_optimizers.json` (see its `_note`). No per-method
  tuning. The ARS V1 values must be checked against Mania et al. 2018 before the full run.
- **Power note:** with 30 seeds, these tests reliably detect only large effects (Cliff's delta ≳ 0.4).
  A null H2 is reported as inconclusive, not as evidence against.
