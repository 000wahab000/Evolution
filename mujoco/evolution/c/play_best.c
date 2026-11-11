/* C port of play_best.py: run the trained brain in MuJoCo, print the scores.
 *
 *     play_best.exe
 *     play_best.exe --episodes 3 --max-steps 2000
 *     play_best.exe --model "mujoco/evolution/models/best_mujoco.npz"
 *
 * There is no window yet (the GLFW viewer is the Python-only part), so this
 * always runs headless. --headless is accepted for symmetry with the script.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "src/brain.h"
#include "src/dogenv.h"

#define DEFAULT_MODEL "mujoco/evolution/models/best_mujoco.npz"
#define DEFAULT_XML "mujoco/evolution/models/smoke.xml"

static void usage(FILE *out)
{
    fprintf(out,
            "usage: play_best [options]\n"
            "\n"
            "  --model PATH      weights file (.npz) [%s]\n"
            "  --xml PATH        MuJoCo model file    [%s]\n"
            "  --episodes N      how many episodes, 0 = 1 for now [1]\n"
            "  --max-steps N     steps before an episode resets [2000]\n"
            "  --verbose         accepted, same output either way\n"
            "  --headless        accepted; this build is always headless\n"
            "  --quiet           print nothing unless something fails\n"
            "  -h, --help        show this text\n",
            DEFAULT_MODEL, DEFAULT_XML);
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

int main(int argc, char **argv)
{
    const char *model_path = DEFAULT_MODEL;
    const char *xml_path = DEFAULT_XML;
    int episodes = 1;
    int max_steps = 2000;
    int quiet = 0;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        int has_value = i + 1 < argc;

        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(stdout);
            return 0;
        } else if (strcmp(arg, "--model") == 0 && has_value) {
            model_path = argv[++i];
        } else if (strcmp(arg, "--xml") == 0 && has_value) {
            xml_path = argv[++i];
        } else if (strcmp(arg, "--episodes") == 0 && has_value) {
            if (!parse_int(argv[++i], &episodes)) {
                fprintf(stderr, "error: --episodes wants a whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--max-steps") == 0 && has_value) {
            if (!parse_int(argv[++i], &max_steps) || max_steps == 0) {
                fprintf(stderr, "error: --max-steps wants a positive whole number\n");
                return 2;
            }
        } else if (strcmp(arg, "--quiet") == 0) {
            quiet = 1;
        } else if (strcmp(arg, "--verbose") == 0 || strcmp(arg, "--headless") == 0 ||
                   strcmp(arg, "--no-ui") == 0) {
            /* Kept so the Python command lines still work. */
        } else if (strcmp(arg, "--speed") == 0 && has_value) {
            i++; /* no window, nothing to throttle */
        } else {
            fprintf(stderr, "error: unknown option '%s'\n\n", arg);
            usage(stderr);
            return 2;
        }
    }

    dog_env_t env;
    char err[512];
    if (!dog_env_init(&env, xml_path, max_steps, err, sizeof err)) {
        fprintf(stderr, "error: %s\n", err);
        return 1;
    }

    brain_t brain;
    if (!brain_load_npz(&brain, model_path, err, sizeof err)) {
        fprintf(stderr, "error: %s\n", err);
        if (strstr(err, "cannot open") != NULL) {
            fprintf(stderr, "       train first: ga_train.exe\n");
        }
        dog_env_free(&env);
        return 1;
    }

    if (!quiet) {
        printf("Model: %s\n", xml_path);
        printf("Observation size %d -> %d -> action size %d\n", env.obs_size,
               BRAIN_HIDDEN, env.act_size);
    }

    int total = episodes > 0 ? episodes : 1;
    for (int episode = 0; episode < total; episode++) {
        episode_result_t r = dog_env_rollout(&env, &brain);
        if (quiet) {
            continue;
        }
        printf("Episode %d: reward %8.2f | distance %6.2f m | %3d steps | %s\n",
               episode + 1, r.reward, r.distance, r.steps,
               r.fallen ? "fell" : "stayed up");
    }

    dog_env_free(&env);
    return 0;
}
