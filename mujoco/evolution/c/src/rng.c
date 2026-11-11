#include "rng.h"

#include <math.h>

#define TWO_PI 6.283185307179586476925286766559

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void rng_seed(rng_t *rng, uint64_t seed)
{
    uint64_t x = seed;
    for (int i = 0; i < 4; i++) {
        rng->s[i] = splitmix64(&x);
    }
    rng->spare = 0.0;
    rng->has_spare = 0;
}

static uint64_t rotl(uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

uint64_t rng_next_u64(rng_t *rng)
{
    uint64_t *s = rng->s;
    uint64_t result = rotl(s[1] * 5, 7) * 9;
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

double rng_uniform(rng_t *rng)
{
    /* 53 random bits, the same amount of precision a double can hold. */
    return (double)(rng_next_u64(rng) >> 11) * (1.0 / 9007199254740992.0);
}

double rng_normal(rng_t *rng)
{
    if (rng->has_spare) {
        rng->has_spare = 0;
        return rng->spare;
    }

    double u1;
    do {
        u1 = rng_uniform(rng);
    } while (u1 <= 1e-12);

    double u2 = rng_uniform(rng);
    double radius = sqrt(-2.0 * log(u1));

    rng->spare = radius * sin(TWO_PI * u2);
    rng->has_spare = 1;
    return radius * cos(TWO_PI * u2);
}

float rng_normal_f(rng_t *rng)
{
    return (float)rng_normal(rng);
}

int rng_int(rng_t *rng, int n)
{
    if (n <= 1) {
        return 0;
    }
    return (int)(rng_next_u64(rng) % (uint64_t)n);
}
