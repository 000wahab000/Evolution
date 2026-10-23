# Papers related to Evolution-1

Reading list for the project: a GA + tiny-MLP walker in Pymunk, with CPG clock inputs, fair-comparison experiments, and a planned walk toward terrain generalization, morphology, and multi-agent battles.

**Status:** Found via web search. Titles, authors, venues, and IDs were checked against search results. I did **not** read the full text of most papers. Items marked *unverified* are listed in the last section.

Read in the order below. Groups A–C are what the current code already uses. D–F are what the experiments and planned versions need.

---

## A. Evolving walking creatures (the original idea)

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Karl Sims, *Evolving Virtual Creatures*** | SIGGRAPH 1994, pp. 15–22 | [PDF](https://www.karlsims.com/papers/siggraph94.pdf) · [DOI](https://dl.acm.org/doi/10.1145/192161.192167) | The ancestor of this project: a GA evolves both body and neural controller in 3D physics. Read it first. Note: one secondary summary says walking was *not* achieved in this paper, so check the actual results. |
| **Cheney, MacCurdy, Clune, Lipson, *Unshackling evolution***  | GECCO 2013 (ACM DOI 10.1145/2463372.2463404) | Not on arXiv (search found only ACM and author copies) | Evolves soft robots with a CPPN generative encoding. Good for the morphology idea in v4. |
| **Mertan & Cheney, *Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential*** | ALife 2025 | [arXiv 2508.17464](https://arxiv.org/abs/2508.17464) | Maps the whole morphology–fitness landscape (about 1.3M bodies) and shows co-optimization searches stall. Relevant once the body becomes evolvable. Already cited in `docs/research/2026-10-20-research-direction.md`. |

## B. Central pattern generators (the CPG clock input)

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Ijspeert, *Central pattern generators for locomotion control in animals and robots: a review*** | *Neural Networks* 21(4), 642–653, 2008 | [DOI 10.1016/j.neunet.2008.03.014](https://doi.org/10.1016/j.neunet.2008.03.014) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0893608008000804) | The CPG background behind `CPG_CLOCK`. Full text not available in search; read the intro and the robotics section (also in `docs/LEARNING.md`, Module 4). |
| **Ijspeert, Crespi, Ryczko, Cabelguen, *From Swimming to Walking with a Salamander Robot Driven by a Spinal Cord Model*** | *Science* 315, 1416–1420, 2007 | [DOI 10.1126/science.1138353](https://www.science.org/doi/abs/10.1126/science.1138353) · [PDF copy (UW course)](https://faculty.washington.edu/minster/bio_inspired_robotics_2022/papers/Ijspeert_crespi_Ryczko_cabelguen_swimming_walking_salamander_science07.pdf) | A CPG model that switches gait with a single input. Shows the rhythm idea in a real robot. Useful for experiment 1 (CPG on vs. zeros). |

## C. Gradient-free optimizers (the GA vs. ES vs. random search comparison)

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Salimans et al., *Evolution Strategies as a Scalable Alternative to Reinforcement Learning*** | arXiv 1703.03864 (2017) | [arXiv](https://arxiv.org/abs/1703.03864) | OpenAI-ES. The `openai_es` optimizer in `experiments/exp2_optimizers.json` comes from here. The abstract reports ES solving 3D humanoid walking. |
| **Such et al., *Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative…***  | arXiv 1712.06567 (2017) | [arXiv](https://arxiv.org/abs/1712.06567) | The argument that a simple GA (like `src/evolution.py`) is competitive for deep RL. Direct support for the project's choice of GA. |
| **Mania, Guy, Recht, *Simple random search provides a competitive approach to reinforcement learning*** | arXiv 1803.07055 (2018) | [arXiv](https://arxiv.org/abs/1803.07055) | The ARS baseline in exp2. Also says results vary a lot across seeds and hyperparameters, which is why the project uses 30 seeds. Check its ARS values against the paper before the full run (per `HYPOTHESES.md`). |
| **Hansen, *The CMA Evolution Strategy: A Tutorial*** | arXiv 1604.00772 (2016) | [arXiv](https://arxiv.org/abs/1604.00772) | The CMA-ES tutorial for exp2's `cmaes` optimizer. Read sections 1–3 for the idea; the appendix has the parameter settings. |

## D. Diversity, novelty, and quality–diversity (the "not just one best walker" idea)

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Mouret & Clune, *Illuminating search spaces by mapping elites*** (MAP-Elites) | arXiv 1504.04909 (2015) | [arXiv](https://arxiv.org/abs/1504.04909) | Keeps the best solution per behavior cell instead of one global best. Basis for any future "gait archive". Tutorial notebook: [jbmouret/map_elites_tutorial](https://github.com/jbmouret/map_elites_tutorial). |
| **Cully, Clune, Tarapore, Mouret, *Robots that can adapt like animals*** | *Nature* 521, 503–507, 2015 | [Nature](https://www.nature.com/articles/nature14422) · [code](https://github.com/resibots/cully_2015_nature) | Real hexapod that recovers from damage using a MAP-Elites archive of walking gaits. The closest real-robot version of "keep many gaits, pick one when something breaks". |
| **Lehman & Stanley, *Abandoning Objectives: Evolution Through the Search for Novelty Alone*** | *Evolutionary Computation* 19(2), 189–223, 2011 | [Semantic Scholar](https://www.semanticscholar.org/paper/Abandoning-Objectives:-Evolution-Through-the-Search-Lehman-Stanley/0de77eceda6308618132204b28755ac1e63648c5) | Argues that fitness can trap search (deception). Their biped-walking experiment is directly relevant to the "go right, don't fall" reward. Use it when you ask whether the reward causes the local optimum at x≈900. |
| **Stanley, Clune, Lehman, Miikkulainen, *Designing neural networks through neuroevolution*** | *Nature Machine Intelligence* 1, 24–35, 2019 | [DOI 10.1038/s42256-018-0006-z](https://www.nature.com/articles/s42256-018-0006-z) | Survey of neuroevolution: novelty, diversity, indirect encoding. Good single overview of the whole field. |

## E. Generalization, open-ended environments, and the reality gap

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Wang, Lehman, Clune, Stanley, *Paired Open-Ended Trailblazer (POET)*** | arXiv 1901.01753 (2019); also GECCO 2019 | [arXiv](https://arxiv.org/abs/1901.01753) | Evolves the terrains and the walkers together. The model for the v3 "procedural terrain" roadmap item. Follow-up: *Enhanced POET* (ICML 2020), [arXiv 2003.08536](https://arxiv.org/abs/2003.08536). |
| **Cobbe et al., *Quantifying Generalization in Reinforcement Learning*** | arXiv 1812.02341 (2018, rev. 2019) | [arXiv](https://arxiv.org/abs/1812.02341) | Shows agents overfit to their training levels and introduces train/test splits for procedural environments. Backs the project's train-slope vs. test-slope (≤4% vs. 5–8%) design. |
| **Koos, Mouret, Doncieux, *The Transferability Approach: Crossing the Reality Gap in Evolutionary Robotics*** | *IEEE TEVC* 17(1), 122–145 (year listed as 2012 or 2013 in different sources) | [arXiv companion 1307.1870](https://arxiv.org/abs/1307.1870) | Uses a simulator-vs-reality transferability score so the EA can avoid gaits that only work in the simulator. Relevant to the `dt/2` physics-exploit check in `HYPOTHESES.md`. |

## F. Statistics and reproducibility (how to report results)

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Henderson et al., *Deep Reinforcement Learning that Matters*** | arXiv 1709.06560; AAAI 2018 | [arXiv](https://arxiv.org/abs/1709.06560) | Shows that seed and implementation choices can flip RL results. Reason to report many seeds and to write down all settings. |
| **Agarwal et al., *Deep Reinforcement Learning at the Edge of the Statistical Precipice*** | arXiv 2108.13264; NeurIPS 2021 | [arXiv](https://arxiv.org/abs/2108.13264) · [rliable](https://agarwl.github.io/rliable) | The IQM + bootstrap CI method in `HYPOTHESES.md`. Read the sections on IQM and performance profiles. |

## G. Surveys and "what evolution actually does"

| Paper | Venue | Link | Why it matters here |
|---|---|---|---|
| **Doncieux, Bredeche, Mouret, Eiben, *Evolutionary Robotics: What, Why, and Where to*** | *Frontiers in Robotics and AI* 2, art. 4, 2015 | [DOI 10.3389/frobt.2015.00004](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2015.00004/full) | Field overview: what evolutionary robotics is and how a project like this fits. Good first read for the big picture. |
| **Lehman et al., *The Surprising Creativity of Digital Evolution***  | arXiv 1803.03453 (2018) | [arXiv](https://arxiv.org/abs/1803.03453) | Stories of evolution exploiting simulators. Already in `docs/LEARNING.md` (Module 4). Connects to the "specification gaming" questions in the learning guide. |

---

## Suggested reading order (about 2 weeks)

1. Sims 1994 (A), then Doncieux 2015 (G).
2. Ijspeert 2008 intro + Ijspeert 2007 (B). Tie back to `CPG_CLOCK` and exp1.
3. Salimans 2017, Such 2017, Mania 2018 (C). Tie back to `evolve()` and exp2.
4. Lehman & Stanley 2011 and Mouret & Clune 2015 (D).
5. Cobbe 2019 and POET (E). Tie back to `random_course()` and the v3 roadmap.
6. Agarwal 2021 and Henderson 2018 (F), before the full 30-seed run.
7. Mertan & Cheney 2025 (A) and Stanley 2019 (D) when you start on morphology (v4+).

## Unverified or to check

- **Lehman et al. 2018 (arXiv 1803.03453):** taken from `docs/LEARNING.md`; I did not re-verify it in this session.
- **Cheney et al. 2013:** no arXiv ID found; cited via ACM DOI from a Dryad metadata record.
- **Koos et al.:** publication year differs by source (2012 vs. 2013). Check the IEEE page.
- **Agarwal et al. NeurIPS outstanding-paper award:** only from secondary listings.
- **Sims 1994 walking result:** one secondary summary says walking was not achieved; verify against the paper before quoting it.
- **Ijspeert 2008:** full text was not available to me; read the intro from the publisher page.
- **Full text not read:** all entries were checked from search snippets and abstracts, not by reading the papers.

## Check yourself

Why does the Such 2017 result support using a simple GA here, and what does it *not* show about your 324-weight brain? (Hint: their networks were millions of parameters.)
