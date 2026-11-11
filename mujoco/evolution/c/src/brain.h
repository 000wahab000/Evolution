/* The CreatureBrain MLP from src/model.py: 26 -> 16 -> 8, tanh, float32. */
#ifndef BRAIN_H
#define BRAIN_H

#include <stddef.h>

#include "rng.h"

#define BRAIN_IN 26
#define BRAIN_HIDDEN 16
#define BRAIN_OUT 8

typedef struct {
    float w1[BRAIN_HIDDEN][BRAIN_IN];
    float b1[BRAIN_HIDDEN];
    float w2[BRAIN_OUT][BRAIN_HIDDEN];
    float b2[BRAIN_OUT];
} brain_t;

/* All four arrays are contiguous in this order, which is what the GA walks. */
float *brain_params(brain_t *b);
const float *brain_params_const(const brain_t *b);
size_t brain_param_count(void);

void brain_random(brain_t *b, rng_t *rng, float sigma);
void brain_forward(const brain_t *b, const float *x, float *out);
void brain_mutate(brain_t *b, rng_t *rng, float rate, float scale);
void brain_crossover(const brain_t *a, const brain_t *b, brain_t *child, rng_t *rng);

int brain_load_npz(brain_t *b, const char *path, char *err, size_t errsz);
int brain_save_npz(const brain_t *b, const char *path, char *err, size_t errsz);

#endif
