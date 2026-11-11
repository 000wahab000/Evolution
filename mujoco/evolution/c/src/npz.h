/* Minimal .npz reader/writer: stored (uncompressed) entries only, float32 only.
 *
 * numpy's np.savez writes ZIP_STORED entries, so no zlib is needed. Anything
 * deflated is rejected with a clear message instead of silently misreading.
 */
#ifndef NPZ_H
#define NPZ_H

#include <stddef.h>

typedef struct {
    unsigned char *buf;
    size_t size;
    size_t cd_offset;
    unsigned int cd_count;
} npz_t;

/* Reads the whole archive into memory. Returns 1 on success. */
int npz_open(npz_t *z, const char *path, char *err, size_t errsz);
void npz_close(npz_t *z);

/* Copies one float32 array out of the archive. The caller owns *values. */
int npz_get_f32(const npz_t *z, const char *name, float **values,
                int *ndim, size_t dims[2], char *err, size_t errsz);

typedef struct {
    const char *name;   /* entry name inside the archive, usually "x.npy" */
    const float *values;
    int ndim;           /* 1 or 2 */
    size_t dims[2];
} npz_entry_t;

/* Writes a ZIP_STORED .npz that np.load() can read back. */
int npz_write_f32(const char *path, const npz_entry_t *entries, int count,
                  char *err, size_t errsz);

#endif
