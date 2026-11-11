/* xoshiro256** plus a Box-Muller normal, so the GA does not need numpy. */
#ifndef RNG_H
#define RNG_H

#include <stdint.h>

typedef struct {
    uint64_t s[4];
    double spare;
    int has_spare;
} rng_t;

void rng_seed(rng_t *rng, uint64_t seed);
uint64_t rng_next_u64(rng_t *rng);

/* Uniform in [0, 1). */
double rng_uniform(rng_t *rng);

/* Standard normal, mean 0, sd 1. */
double rng_normal(rng_t *rng);
float rng_normal_f(rng_t *rng);

/* Uniform integer in [0, n). */
int rng_int(rng_t *rng, int n);

#endif
