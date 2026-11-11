#include "dogenv.h"

#include <stdio.h>
#include <string.h>

static void observe(dog_env_t *env)
{
    const mjData *d = env->data;
    float *o = env->obs;

    for (int i = 0; i < env->nangle; i++) {
        *o++ = (float)d->qpos[env->angle_idx[i]];
    }
    for (int i = 0; i < env->nvel; i++) {
        *o++ = (float)d->qvel[env->vel_idx[i]];
    }

    /* Free-joint layout: xyz + quaternion in qpos, linear + angular in qvel. */
    *o++ = (float)d->qpos[env->free_qpos + 2];
    for (int i = 3; i < 7; i++) {
        *o++ = (float)d->qpos[env->free_qpos + i];
    }
    for (int i = 3; i < 6; i++) {
        *o++ = (float)d->qvel[env->free_dof + i];
    }
    for (int i = 0; i < 2; i++) {
        *o++ = (float)d->qvel[env->free_dof + i];
    }
}

int dog_env_init(dog_env_t *env, const char *xml_path, int max_steps, char *err,
                 size_t errsz)
{
    memset(env, 0, sizeof *env);
    env->free_qpos = -1;
    env->free_dof = -1;
    env->max_steps = max_steps;

    char mjerr[1000];
    mjerr[0] = '\0';
    env->model = mj_loadXML(xml_path, NULL, mjerr, (int)sizeof mjerr);
    if (env->model == NULL) {
        snprintf(err, errsz, "mj_loadXML(\"%s\"): %s", xml_path, mjerr);
        return 0;
    }

    env->data = mj_makeData(env->model);
    if (env->data == NULL) {
        snprintf(err, errsz, "mj_makeData failed for \"%s\"", xml_path);
        mj_deleteModel(env->model);
        env->model = NULL;
        return 0;
    }

    for (int i = 0; i < env->model->njnt; i++) {
        int type = env->model->jnt_type[i];
        if (type == mjJNT_FREE && env->free_qpos < 0) {
            env->free_qpos = (int)env->model->jnt_qposadr[i];
            env->free_dof = (int)env->model->jnt_dofadr[i];
        } else if (type == mjJNT_HINGE) {
            if (env->nangle >= DOG_MAX_HINGES) {
                snprintf(err, errsz, "%s has more than %d hinge joints", xml_path,
                         DOG_MAX_HINGES);
                dog_env_free(env);
                return 0;
            }
            env->angle_idx[env->nangle++] = (int)env->model->jnt_qposadr[i];
            env->vel_idx[env->nvel++] = (int)env->model->jnt_dofadr[i];
        }
    }

    if (env->free_qpos < 0) {
        snprintf(err, errsz, "%s needs a free joint on the torso", xml_path);
        dog_env_free(env);
        return 0;
    }
    if (env->free_qpos + 7 > env->model->nq || env->free_dof + 6 > env->model->nv) {
        snprintf(err, errsz, "%s has a free joint smaller than 7 qpos / 6 qvel", xml_path);
        dog_env_free(env);
        return 0;
    }

    env->obs_size = 2 * env->nangle + 1 + 4 + 3 + 2;
    env->act_size = (int)env->model->nu;

    if (env->obs_size != BRAIN_IN) {
        snprintf(err, errsz,
                 "%s produces %d observations but brain_t expects %d "
                 "(8 hinges, one free joint)",
                 xml_path, env->obs_size, BRAIN_IN);
        dog_env_free(env);
        return 0;
    }
    if (env->act_size != BRAIN_OUT) {
        snprintf(err, errsz, "%s has %d actuators but brain_t expects %d", xml_path,
                 env->act_size, BRAIN_OUT);
        dog_env_free(env);
        return 0;
    }
    return 1;
}

void dog_env_free(dog_env_t *env)
{
    if (env->data != NULL) {
        mj_deleteData(env->data);
    }
    if (env->model != NULL) {
        mj_deleteModel(env->model);
    }
    env->data = NULL;
    env->model = NULL;
}

const float *dog_env_reset(dog_env_t *env)
{
    mj_resetData(env->model, env->data);
    mj_forward(env->model, env->data);
    env->steps = 0;
    env->start_x = env->data->qpos[env->free_qpos];
    observe(env);
    return env->obs;
}

double dog_env_step(dog_env_t *env, const float *action, int *done, int *fallen)
{
    for (int i = 0; i < env->act_size; i++) {
        float a = action[i];
        if (a < -1.0f) {
            a = -1.0f;
        } else if (a > 1.0f) {
            a = 1.0f;
        }
        env->data->ctrl[i] = a;
    }

    mj_step(env->model, env->data);
    env->steps++;

    double reward = (env->data->qpos[env->free_qpos] - env->start_x) * 2.0;
    *fallen = env->data->qpos[env->free_qpos + 2] < DOG_FALLEN_HEIGHT;
    *done = *fallen || env->steps >= env->max_steps;

    observe(env);
    return reward;
}

episode_result_t dog_env_rollout(dog_env_t *env, const brain_t *brain)
{
    episode_result_t result = {0.0, 0.0, 0, 0};
    const float *state = dog_env_reset(env);
    float action[BRAIN_OUT];

    for (int i = 0; i < env->max_steps; i++) {
        int done = 0;
        int fallen = 0;

        brain_forward(brain, state, action);
        result.reward += dog_env_step(env, action, &done, &fallen);
        result.steps++;
        result.fallen = fallen;

        if (done) {
            break;
        }
    }

    result.distance = env->data->qpos[env->free_qpos] - env->start_x;
    return result;
}
