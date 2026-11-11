#include "brain.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "npz.h"

_Static_assert(sizeof(brain_t) ==
                   (BRAIN_HIDDEN * BRAIN_IN + BRAIN_HIDDEN + BRAIN_OUT * BRAIN_HIDDEN +
                    BRAIN_OUT) *
                       sizeof(float),
               "brain_t must stay a flat float array so the GA can walk it");

float *brain_params(brain_t *b)
{
    return &b->w1[0][0];
}

const float *brain_params_const(const brain_t *b)
{
    return &b->w1[0][0];
}

size_t brain_param_count(void)
{
    return sizeof(brain_t) / sizeof(float);
}

void brain_random(brain_t *b, rng_t *rng, float sigma)
{
    float *p = brain_params(b);
    for (size_t i = 0; i < brain_param_count(); i++) {
        p[i] = sigma * rng_normal_f(rng);
    }
}

void brain_forward(const brain_t *b, const float *x, float *out)
{
    float hidden[BRAIN_HIDDEN];

    for (int i = 0; i < BRAIN_HIDDEN; i++) {
        float sum = b->b1[i];
        for (int j = 0; j < BRAIN_IN; j++) {
            sum += b->w1[i][j] * x[j];
        }
        hidden[i] = tanhf(sum);
    }

    for (int i = 0; i < BRAIN_OUT; i++) {
        float sum = b->b2[i];
        for (int j = 0; j < BRAIN_HIDDEN; j++) {
            sum += b->w2[i][j] * hidden[j];
        }
        out[i] = tanhf(sum);
    }
}

void brain_mutate(brain_t *b, rng_t *rng, float rate, float scale)
{
    float *p = brain_params(b);
    for (size_t i = 0; i < brain_param_count(); i++) {
        if (rng_uniform(rng) < (double)rate) {
            p[i] += (float)(rng_normal(rng) * (double)scale);
        }
    }
}

void brain_crossover(const brain_t *a, const brain_t *b, brain_t *child, rng_t *rng)
{
    *child = *a;
    float *dst = brain_params(child);
    const float *src = brain_params_const(b);
    for (size_t i = 0; i < brain_param_count(); i++) {
        if (rng_uniform(rng) < 0.5) {
            dst[i] = src[i];
        }
    }
}

static int load_array(npz_t *z, const char *name, float *dst, size_t dst_count,
                      char *err, size_t errsz)
{
    int ndim = 0;
    size_t dims[2] = {0, 0};
    float *values = NULL;
    if (!npz_get_f32(z, name, &values, &ndim, dims, err, errsz)) {
        return 0;
    }

    size_t count = 1;
    for (int i = 0; i < ndim; i++) {
        count *= dims[i];
    }
    if (count != dst_count) {
        snprintf(err, errsz, "%s holds %zu values, brain expects %zu", name, count,
                 dst_count);
        free(values);
        return 0;
    }

    memcpy(dst, values, count * sizeof(float));
    free(values);
    return 1;
}

int brain_load_npz(brain_t *b, const char *path, char *err, size_t errsz)
{
    npz_t z;
    if (!npz_open(&z, path, err, errsz)) {
        return 0;
    }

    int ok = load_array(&z, "w1.npy", &b->w1[0][0], BRAIN_HIDDEN * BRAIN_IN, err,
                        errsz) &&
             load_array(&z, "b1.npy", b->b1, BRAIN_HIDDEN, err, errsz) &&
             load_array(&z, "w2.npy", &b->w2[0][0], BRAIN_OUT * BRAIN_HIDDEN, err,
                        errsz) &&
             load_array(&z, "b2.npy", b->b2, BRAIN_OUT, err, errsz);

    npz_close(&z);
    return ok;
}

int brain_save_npz(const brain_t *b, const char *path, char *err, size_t errsz)
{
    npz_entry_t entries[4] = {
        {"w1.npy", &b->w1[0][0], 2, {BRAIN_HIDDEN, BRAIN_IN}},
        {"b1.npy", b->b1, 1, {BRAIN_HIDDEN, 0}},
        {"w2.npy", &b->w2[0][0], 2, {BRAIN_OUT, BRAIN_HIDDEN}},
        {"b2.npy", b->b2, 1, {BRAIN_OUT, 0}},
    };
    return npz_write_f32(path, entries, 4, err, errsz);
}
