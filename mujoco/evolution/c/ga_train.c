/* C port of ga_train.py: genetic algorithm over the 26 -> 16 -> 8 tanh brain.
 *
 *     ga_train.exe --generations 30 --pop 20
 *     ga_train.exe --generations 100 --pop 50 --max-steps 1000 --seed 7
 *
 * Writes the same history.csv columns and a .npz that np.load() can read, so
 * play_best.py can still replay whatever this produced.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#define make_dir(path) _mkdir(path)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define make_dir(path) mkdir(path, 0777)
#endif

#include "src/brain.h"
#include "src/dogenv.h"

#define DEFAULT_XML "mujoco/evolution/models/smoke.xml"
#define DEFAULT_OUT "mujoco/evolution/results"
#define DEFAULT_MODEL "mujoco/evolution/models/best_mujoco_c.npz"

#define FALL_PENALTY 100.0
#define ELITE_MUTATION_CHANCE 0.15
#define CROSSOVER_PICK 0.5

static void usage(FILE *out)
{
    fprintf(out,
            "usage: ga_train [options]\n"
            "\n"
            "  --generations N     generations to run              [30]\n"
            "  --pop N             brains per generation           [20]\n"
            "  --max-steps N       steps per episode               [600]\n"
            "  --mutation-rate F   per-weight mutation chance      [0.15]\n"
            "  --mutation-scale F  mutation noise sd               [0.35]\n"
            "  --seed N            RNG seed                        [0]\n"
            "  --xml PATH          MuJoCo model file               [%s]\n"
            "  --out DIR           where history.csv goes          [%s]\n"
            "  --model PATH        best brain to save              [%s]\n"
            "  --workers N         accepted; this build is single-process\n"
            "  -h, --help          show this text\n",
            DEFAULT_XML, DEFAULT_OUT, DEFAULT_MODEL);
}

static int parse_int(const char *text, int *out)
{
    char *end = NULL;
    long v = strtol(text, &end, 10);
    if (end == text || *end != '\0' || v < 0 || v > 1000000) {
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_float(const char *text, float *out)
{
    char *end = NULL;
    float v = strtof(text, &end);
    if (end == text || *end != '\0') {
        return 0;
    }
    *out = v;
    return 1;
}

static int ensure_dir(const char *path)
{
    char tmp[1024];
    if (path[0] == '\0') {
        return 1;
    }
    if (strlen(path) >= sizeof tmp) {
        return 0;
    }
    snprintf(tmp, sizeof tmp, "%s", path);

    for (char *p = tmp + 1; *p != '\0'; p++) {
        if (*p == '/' || *p == '\\') {
            char saved = *p;
            *p = '\0';
            make_dir(tmp);
            *p = saved;
        }
    }
    make_dir(tmp);
    return 1;
}

/* --------------------------------------------------------------- evolution */

static const double *sort_fitness;

static int compare_desc(const void *a, const void *b)
{
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    double fa = sort_fitness[ia];
    double fb = sort_fitness[ib];
    if (fa < fb) {
        return 1;
    }
    if (fa > fb) {
        return -1;
    }
    return ia - ib;
}

/* Same scheme as MujocoPopulation.evolve: keep the top few, then breed. */
static brain_t *evolve(const brain_t *pop, const double *fitness, int pop_size,
                       rng_t *rng, float mutation_rate, float mutation_scale)
{
    int *order = malloc((size_t)pop_size * sizeof *order);
    brain_t *next = malloc((size_t)pop_size * sizeof *next);
    if (order == NULL || next == NULL) {
        free(order);
        free(next);
        return NULL;
    }

    for (int i = 0; i < pop_size; i++) {
        order[i] = i;
    }
    sort_fitness = fitness;
    qsort(order, (size_t)pop_size, sizeof *order, compare_desc);

    int elite_count = pop_size / 5;
    if (elite_count < 1) {
        elite_count = 1;
    }
    if (elite_count > 5) {
        elite_count = 5;
    }

    brain_t *elites = malloc((size_t)elite_count * sizeof *elites);
    if (elites == NULL) {
        free(order);
        free(next);
        return NULL;
    }
    for (int i = 0; i < elite_count; i++) {
        elites[i] = pop[order[i]];
        next[i] = elites[i];
    }

    for (int filled = elite_count; filled < pop_size; filled++) {
        if (rng_uniform(rng) < ELITE_MUTATION_CHANCE) {
            next[filled] = elites[rng_int(rng, elite_count)];
            brain_mutate(&next[filled], rng, mutation_rate, mutation_scale);
        } else {
            int a = rng_int(rng, elite_count);
            int b = a;
            if (elite_count >= 2) {
                b = rng_int(rng, elite_count - 1);
                if (b >= a) {
                    b++;
                }
            }
            brain_crossover(&elites[a], &elites[b], &next[filled], rng);
            brain_mutate(&next[filled], rng, mutation_rate, mutation_scale);
        }
    }

    free(elites);
    free(order);
    return next;
}

/* -------------------------------------------------------------------- main */

int main(int argc, char **argv)
{
    int generations = 30;
    int pop_size = 20;
    int max_steps = 600;
    int seed = 0;
    int workers = 1;
    float mutation_rate = 0.15f;
    float mutation_scale = 0.35f;
    const char *xml_path = DEFAULT_XML;
    const char *out_dir = DEFAULT_OUT;
    const char *model_path = DEFAULT_MODEL;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        int has_value = i + 1 < argc;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(arg, "--generations") == 0 && has_value) {
            if (!parse_int(argv[++i], &generations)) {
                fprintf(stderr, "error: --generations wants a whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--pop") == 0 && has_value) {
            if (!parse_int(argv[++i], &pop_size) || pop_size < 1) {
                fprintf(stderr, "error: --pop wants a positive whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--max-steps") == 0 && has_value) {
            if (!parse_int(argv[++i], &max_steps) || max_steps < 1) {
                fprintf(stderr, "error: --max-steps wants a positive whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--mutation-rate") == 0 && has_value) {
            if (!parse_float(argv[++i], &mutation_rate)) {
                fprintf(stderr, "error: --mutation-rate wants a number\n");
                return 2;
            }
        } else if (strcmp(arg, "--mutation-scale") == 0 && has_value) {
            if (!parse_float(argv[++i], &mutation_scale)) {
                fprintf(stderr, "error: --mutation-scale wants a number\n");
                return 2;
            }
        } else if (strcmp(arg, "--seed") == 0 && has_value) {
            if (!parse_int(argv[++i], &seed)) {
                fprintf(stderr, "error: --seed wants a whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--workers") == 0 && has_value) {
            if (!parse_int(argv[++i], &workers)) {
                fprintf(stderr, "error: --workers wants a whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--xml") == 0 && has_value) {
            xml_path = argv[++i];
        } else if (strcmp(arg, "--out") == 0 && has_value) {
            out_dir = argv[++i];
        } else if (strcmp(arg, "--model") == 0 && has_value) {
            model_path = argv[++i];
        } else {
            fprintf(stderr, "error: unknown option '%s'\n\n", arg);
            usage(stderr);
            return 2;
        }
    }

    if (workers > 1) {
        printf("note: --workers %d ignored, the C build evaluates genomes in one process\n",
               workers);
    }

    dog_env_t env;
    char err[512];
    if (!dog_env_init(&env, xml_path, max_steps, err, sizeof err)) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }

    printf("Model: %s\n", xml_path);
    printf("Observation size %d -> %d -> action size %d\n", env.obs_size, BRAIN_HIDDEN,
           env.act_size);

    rng_t rng;
    rng_seed(&rng, (uint64_t)seed);

    brain_t *pop = malloc((size_t)pop_size * sizeof *pop);
    double *fitness = calloc((size_t)pop_size, sizeof *fitness);
    episode_result_t *runs = calloc((size_t)pop_size, sizeof *runs);
    if (pop == NULL || fitness == NULL || runs == NULL) {
        fprintf(stderr, "error: out of memory for a population of %d\n", pop_size);
        dog_env_free(&env);
        free(pop);
        free(fitness);
        free(runs);
        return 1;
    }
    for (int i = 0; i < pop_size; i++) {
        brain_random(&pop[i], &rng, 0.25f);
    }

    if (!ensure_dir(out_dir)) {
        fprintf(stderr, "error: cannot create output directory \"%s\"\n", out_dir);
        dog_env_free(&env);
        free(pop);
        free(fitness);
        free(runs);
        return 1;
    }

    /* The .npz may live somewhere else than history.csv. */
    char model_dir[1024];
    snprintf(model_dir, sizeof model_dir, "%s", model_path);
    char *last_sep = strrchr(model_dir, '/');
    char *last_back = strrchr(model_dir, '\\');
    if (last_back != NULL && (last_sep == NULL || last_back > last_sep)) {
        last_sep = last_back;
    }
    if (last_sep != NULL) {
        *last_sep = '\0';
        if (!ensure_dir(model_dir)) {
            fprintf(stderr, "error: cannot create model directory \"%s\"\n", model_dir);
            dog_env_free(&env);
            free(pop);
            free(fitness);
            free(runs);
            return 1;
        }
    }

    char history_path[1200];
    snprintf(history_path, sizeof history_path, "%s/history.csv", out_dir);
    FILE *history = fopen(history_path, "w");
    if (history == NULL) {
        fprintf(stderr, "error: cannot write \"%s\": %s\n", history_path,
                strerror(errno));
        dog_env_free(&env);
        free(pop);
        free(fitness);
        free(runs);
        return 1;
    }
    fprintf(history,
            "generation,best_fitness,mean_fitness,best_distance,best_steps\n");

    double best_fitness = -1e300;
    int have_best = 0;
    int status = 0;

    for (int gen = 0; gen < generations; gen++) {
        double total = 0.0;
        int best_idx = 0;
        for (int i = 0; i < pop_size; i++) {
            episode_result_t r = dog_env_rollout(&env, &pop[i]);
            double score = r.reward;
            if (r.fallen) {
                score -= FALL_PENALTY;
            }
            runs[i] = r;
            fitness[i] = score;
            total += score;
            if (i == 0 || score > fitness[best_idx]) {
                best_idx = i;
            }
        }

        double gen_best = fitness[best_idx];
        double gen_mean = total / (double)pop_size;

        if (!have_best || gen_best > best_fitness) {
            best_fitness = gen_best;
            have_best = 1;
            if (!brain_save_npz(&pop[best_idx], model_path, err, sizeof err)) {
                fprintf(stderr, "error: %s\n", err);
                status = 1;
                break;
            }
        }

        printf("Gen %3d | best %8.2f | mean %8.2f | distance %7.2f m | steps %3d\n",
               gen, gen_best, gen_mean, runs[best_idx].distance, runs[best_idx].steps);
        fprintf(history, "%d,%.6f,%.6f,%.6f,%d\n", gen, gen_best, gen_mean,
                runs[best_idx].distance, runs[best_idx].steps);
        fflush(history);

        if (gen + 1 < generations) {
            brain_t *next =
                evolve(pop, fitness, pop_size, &rng, mutation_rate, mutation_scale);
            if (next == NULL) {
                fprintf(stderr, "error: out of memory during evolution\n");
                status = 1;
                break;
            }
            free(pop);
            pop = next;
            memset(fitness, 0, (size_t)pop_size * sizeof *fitness);
            memset(runs, 0, (size_t)pop_size * sizeof *runs);
        }
    }

    fclose(history);

    if (status == 0 && !have_best) {
        if (!brain_save_npz(&pop[0], model_path, err, sizeof err)) {
            fprintf(stderr, "error: %s\n", err);
            status = 1;
        }
    }

    if (status == 0) {
        printf("\nSaved best model to %s (fitness %.2f)\n", model_path, best_fitness);
    }

    dog_env_free(&env);
    free(pop);
    free(fitness);
    free(runs);
    return status;
}
