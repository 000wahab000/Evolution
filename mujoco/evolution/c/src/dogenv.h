/* The MujocoDogEnv from mujoco_env.py: reset/step on a MuJoCo model, no Python. */
#ifndef DOGENV_H
#define DOGENV_H

#include <stddef.h>

#include <mujoco/mujoco.h>

#include "brain.h"

#define DOG_FALLEN_HEIGHT 0.25
#define DOG_MAX_HINGES 16

typedef struct {
    mjModel *model;
    mjData *data;
    int max_steps;
    int steps;
    double start_x;

    int nangle;
    int nvel;
    int angle_idx[DOG_MAX_HINGES];
    int vel_idx[DOG_MAX_HINGES];

    int free_qpos;
    int free_dof;

    int obs_size;
    int act_size;
    float obs[BRAIN_IN];
} dog_env_t;

int dog_env_init(dog_env_t *env, const char *xml_path, int max_steps, char *err,
                 size_t errsz);
void dog_env_free(dog_env_t *env);

/* Resets to the pose baked into the XML and returns the first observation. */
const float *dog_env_reset(dog_env_t *env);

/* One physics step. Returns the reward and fills done/fallen. */
double dog_env_step(dog_env_t *env, const float *action, int *done, int *fallen);

typedef struct {
    double reward;
    double distance;
    int fallen;
    int steps;
} episode_result_t;

/* Full rollout, the same loop play_best.py and ga_train.py use. */
episode_result_t dog_env_rollout(dog_env_t *env, const brain_t *brain);

#endif
