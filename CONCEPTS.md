# Concepts

> Shared domain vocabulary for this project — entities, named processes, and status concepts with project-specific meaning. Seeded with core domain vocabulary, then accretes as ce-compound and ce-compound-refresh process learnings; direct edits are fine. Glossary only, not a spec or catch-all.

### `CreatureEnv`
The 2D Pymunk rigid-body physics environment consisting of a quad-legged creature body with torso, 4 rotational limit joints, 4 simple motors, and a horizontal friction ground segment.

### `CreatureBrain`
A PyTorch neural network policy module with linear layers and Tanh activations mapping a 15-dimensional state vector (torso kinematics, joint states, CPG phase clock) to 4 continuous motor velocity commands in `[-1.0, 1.0]`.

### `Fitness Mode`
The active reward evaluation configuration governing evolutionary selection pressure:
- `baseline`: Forward displacement with leg activity bonus.
- `tilt_penalized`: Baseline with continuous torso inclination penalties.
- `energy_penalized`: Baseline with motor action magnitude penalties.
- `all_penalties`: Combines tilt and energy penalties.
- `ground_termination`: Eliminates continuous tilt penalties; immediately terminates the episode and penalizes the score if the torso contacts the ground.

### `Universal Best Model`
The highest-fitness neural network checkpoint (`models/universal_best.pth`) and companion metadata (`models/universal_best_meta.json`) preserved across all independent training runs.

### `Central Pattern Generator (CPG)`
Internal sinusoidal rhythmic phase clock (`sin(phase), cos(phase)`) embedded in the environment observation vector to provide cyclic temporal stride cues for locomotion coordination.
