"""GA training for the playground notebook. The generation loop runs in the caller's process;
only genome evaluation goes to worker processes (so the worker function lives in this module)."""

import os
import pickle
import time

import numpy as np

from src.research.config import RunConfig, make_rng
from src.research.evaluate import (
    DIM,
    evaluate_genome,
    fixed_scenarios,
    from_flat,
    make_env,
    train_scenarios,
)
from src.research.optimizers import make_optimizer

PRETRAINED = os.path.join("data", "models", "universal_best.npz")  # same 15->16->4 brain


def make_cfg(cpg, seed):
    return RunConfig(experiment="playground", variant="cpg" if cpg else "no_cpg", seed=seed,
                     brain_input="cpg" if cpg else "zeros", rollouts_k=2)


def eval_one(args):
    """Worker: (genome, cfg, scenarios) -> (training reward, mean distance)."""
    genome, cfg, scenarios = args
    ret, rolls = evaluate_genome(genome, cfg, scenarios)
    return ret, float(np.mean([r["distance"] for r in rolls]))


def ckpt_path(run_name, seed):
    return os.path.join("results", "playground", f"{run_name}_seed{seed}.ckpt")


def load_pretrained():
    d = np.load(PRETRAINED)
    return np.concatenate([d[k].ravel() for k in ("w1", "b1", "w2", "b2")])


def save_ckpt(path, opt, train_rng, gen, hist, champion, cpg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = dict(pop=opt.pop, opt_rng=opt.rng.bit_generator.state, train_rng=train_rng.bit_generator.state,
                 gen=gen, hist=hist, champion=champion, cpg=cpg)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(state, f)
    os.replace(path + ".tmp", path)  # never leaves a half-written checkpoint


def show_walker(genome, cfg, max_frames=None):
    """Replay a genome on the first test terrain; returns an HTML animation (side view)."""
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from matplotlib import animation

    from src.model import CreatureBrain

    scn = fixed_scenarios(cfg, "test")[0]
    env = make_env(cfg, scn)
    brain = CreatureBrain.from_state_dict(from_flat(np.array(genome)))
    state, frames, done = env.reset(), [], False
    while not done and len(frames) < (max_frames or cfg.max_steps):
        state, _, done, _ = env.step(brain.get_action(state))
        s = env.get_render_state()
        frames.append((s["torso"], s["legs"]))

    xs = [f[0]["x"] for f in frames]
    fig, ax = plt.subplots(figsize=(8, 3))
    terrain_x = np.linspace(min(xs) - 200, max(xs) + 300, 200)
    ground = [scn.terrain.ground_top(x) for x in terrain_x]
    ax.plot(terrain_x, [g if g is not None else np.nan for g in ground], color="saddlebrown")
    ax.set_xlim(min(xs) - 150, max(xs) + 250)
    ax.set_ylim(620, 330)  # y points down in the simulation
    ax.set_aspect("equal")
    (body,) = ax.plot([], [], "k", lw=6)
    legs_ln = [ax.plot([], [], "tab:red", lw=3)[0] for _ in frames[0][1]]

    def draw(i):
        t, legs = frames[i]
        c, s = np.cos(t["angle"]), np.sin(t["angle"])
        body.set_data([t["x"] - 30 * c, t["x"] + 30 * c], [t["y"] - 30 * s, t["y"] + 30 * s])
        for ln, lg in zip(legs_ln, legs):
            h = lg["height"] / 2
            a = lg["angle"]
            ln.set_data([lg["x"] + h * np.sin(a), lg["x"] - h * np.sin(a)],
                        [lg["y"] - h * np.cos(a), lg["y"] + h * np.cos(a)])
        ax.set_xlim(t["x"] - 200, t["x"] + 300)
        return [body, *legs_ln]

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=50, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def _draw_chart(hist, seed):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.plot(hist["best"], label="best")
    ax.plot(hist["mean"], label="mean")
    ax.set(xlabel="generation", ylabel="distance (px)", title=f"seed {seed}")
    ax.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig


def train(seed, cpg, mutation_rate, generations, population, run_name, pool=None,
          keep_training=False, checkpoint_every=10, start_from="random", chart_every=5,
          preview_every=0, log=print):
    """Evolve one seed. Returns (test distance, champion genome list, history).
    pool = a ProcessPoolExecutor (None = evaluate in this process)."""
    cfg = make_cfg(cpg, seed)
    opt = make_optimizer("ga", DIM, {"pop_size": population, "mutation_rate": mutation_rate},
                         make_rng(seed, "optimizer"))
    rng = make_rng(seed, "train")
    gen, hist, champion = 0, {"best": [], "mean": []}, None
    path = ckpt_path(run_name, seed)
    if keep_training and os.path.exists(path):
        with open(path, "rb") as f:
            st = pickle.load(f)
        if len(st["pop"]) != population or st.get("cpg", cpg) != cpg:
            raise ValueError(f"{path} was saved with other POPULATION/CPG settings; "
                             "use the same settings or a new RUN_NAME")
        opt.pop, gen, hist, champion = st["pop"], st["gen"], st["hist"], st["champion"]
        opt.rng.bit_generator.state, rng.bit_generator.state = st["opt_rng"], st["train_rng"]
        log(f"seed {seed} | resuming from generation {gen}")
    elif start_from == "pretrained":  # the first walker is the saved one, the rest are noisy copies
        base = load_pretrained()
        opt.pop = [base] + [base + opt.rng.normal(0.0, 0.1, DIM) for _ in range(population - 1)]
    target = gen + generations
    mapper = pool.map if pool else map
    chart = preview = None
    t0, start_gen = time.time(), gen
    while gen < target:
        scenarios = train_scenarios(cfg, rng)  # same terrains for the whole generation
        genomes = opt.ask()
        res = list(mapper(eval_one, [(g, cfg, scenarios) for g in genomes]))
        opt.tell([r[0] for r in res])
        gen += 1
        dists = [r[1] for r in res]
        hist["best"].append(max(dists))
        hist["mean"].append(float(np.mean(dists)))
        champion = opt.recommend().tolist()
        per_gen = (time.time() - t0) / (gen - start_gen)
        eta = f"{per_gen * (target - gen) / 60:.0f} min"
        log(f"seed {seed} | gen {gen:3d}/{target} | best {max(dists):.0f} px | "
            f"mean {np.mean(dists):.0f} px | {per_gen:.1f} s/gen | ETA {eta}")
        if gen % checkpoint_every == 0 or gen == target:
            save_ckpt(path, opt, rng, gen, hist, champion, cpg)
        if chart_every and (gen % chart_every == 0 or gen == target):
            from IPython.display import display

            fig = _draw_chart(hist, seed)
            if chart is None:
                chart = display(fig, display_id=True)
            else:
                chart.update(fig)
        if preview_every and gen % preview_every == 0:
            from IPython.display import display

            html = show_walker(champion, cfg, max_frames=150)
            if preview is None:
                preview = display(html, display_id=True)
            else:
                preview.update(html)
    test = evaluate_genome(np.array(champion), cfg, fixed_scenarios(cfg, "test"))[1]  # unseen terrains
    return float(np.mean([r["distance"] for r in test])), champion, hist
