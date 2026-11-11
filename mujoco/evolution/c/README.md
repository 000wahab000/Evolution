# MuJoCo dog in C

C port of `mujoco/evolution/scripts/{mujoco_env,ga_train,play_best}.py` and the
`CreatureBrain` / GA bits from `src/`. Same algorithm, no Python at run time:
MuJoCo's own C API does the physics, and the `.npz` weights are read and
written by a small stored-zip reader.

```
c/
  play_best.c        replay a saved brain, print reward / distance
  ga_train.c         evolve brains, write history.csv + best .npz
  src/
    brain.[ch]       26 -> 16 -> 8 tanh MLP, mutate, crossover, npz I/O
    dogenv.[ch]      mj_loadXML / reset / step / rollout
    npz.[ch]         ZIP_STORED + .npy, float32 only
    rng.[ch]         xoshiro256** + Box-Muller normal
  build.ps1
```

## Build

Needs gcc and the `mujoco.dll` that the Python venv already downloaded.

```powershell
winget install --id BrechtSanders.WinLibs.POSIX.UCRT -e   # once, then open a new shell
.\mujoco\evolution\c\build.ps1
```

The script finds `mujoco.dll` + `include/` in `.venv-mujoco`, links against
them, and copies the dll next to the exe. Use `-MujocoDir PATH` for another
MuJoCo install.

## Run

From the repo root, because the default paths are relative:

```powershell
# replay the brain the Python trainer saved
.\mujoco\evolution\c\build\play_best.exe
.\mujoco\evolution\c\build\play_best.exe --episodes 5 --max-steps 2000

# train a fresh one
.\mujoco\evolution\c\build\ga_train.exe --generations 30 --pop 20
```

`ga_train.exe` defaults to `best_mujoco_c.npz` so it never overwrites the
Python-trained `best_mujoco.npz`. The file it writes is a normal `.npz`, so
`play_best.py` can replay it and `brain_load_npz` can read the Python one —
the two trainers are interchangeable.

## What is fixed, and why

`brain_t` is sized for this dog: 8 hinge joints + one free joint gives
26 observations, and `smoke.xml` has 8 motors. A different XML fails loudly
with the actual counts instead of silently reading the wrong offsets.

The float32 matmuls are not bit-identical to numpy's BLAS, so a long rollout
can drift away from the Python one even though both start from the same
weights. Short rollouts and every observation on step 1 match.

## Not ported yet

The `mujoco.viewer` window (GLFW + OpenGL). Both exes are headless; `--headless`
is accepted and ignored so the Python command lines still work.
