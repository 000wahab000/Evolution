#include "npz.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ZIP_EOCD_SIG 0x06054b50u
#define ZIP_CD_SIG 0x02014b50u
#define ZIP_LOCAL_SIG 0x04034b50u

static uint16_t rd16(const unsigned char *p)
{
    return (uint16_t)(p[0] | (p[1] << 8));
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static void wr16(unsigned char *p, uint16_t v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static void wr32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

static void set_err(char *err, size_t errsz, const char *fmt, const char *arg)
{
    if (err != NULL && errsz > 0) {
        snprintf(err, errsz, fmt, arg != NULL ? arg : "");
    }
}

/* ---------------------------------------------------------------- reading */

int npz_open(npz_t *z, const char *path, char *err, size_t errsz)
{
    memset(z, 0, sizeof *z);

    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        set_err(err, errsz, "cannot open %s", path);
        return 0;
    }

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        set_err(err, errsz, "cannot seek %s", path);
        return 0;
    }
    long size = ftell(f);
    if (size <= 0) {
        fclose(f);
        set_err(err, errsz, "%s is empty or unseekable", path);
        return 0;
    }
    rewind(f);

    z->buf = malloc((size_t)size);
    if (z->buf == NULL) {
        fclose(f);
        set_err(err, errsz, "out of memory reading %s", path);
        return 0;
    }
    z->size = (size_t)size;

    if (fread(z->buf, 1, z->size, f) != z->size) {
        fclose(f);
        npz_close(z);
        set_err(err, errsz, "short read on %s", path);
        return 0;
    }
    fclose(f);

    /* The end-of-central-directory record sits in the last 22 + 65535 bytes. */
    size_t lowest = z->size > 22 + 65535 ? z->size - (22 + 65535) : 0;
    long eocd = -1;
    for (size_t i = z->size - 22 + 1; i-- > lowest;) {
        if (rd32(z->buf + i) == ZIP_EOCD_SIG) {
            eocd = (long)i;
            break;
        }
    }
    if (eocd < 0) {
        npz_close(z);
        set_err(err, errsz, "%s is not a zip archive", path);
        return 0;
    }

    z->cd_count = rd16(z->buf + eocd + 10);
    z->cd_offset = rd32(z->buf + eocd + 16);
    if (z->cd_offset >= z->size) {
        npz_close(z);
        set_err(err, errsz, "%s has a corrupt central directory", path);
        return 0;
    }
    return 1;
}

void npz_close(npz_t *z)
{
    free(z->buf);
    memset(z, 0, sizeof *z);
}

static const unsigned char *find_entry(const npz_t *z, const char *name)
{
    size_t pos = z->cd_offset;
    size_t namelen = strlen(name);

    for (unsigned int i = 0; i < z->cd_count; i++) {
        if (pos + 46 > z->size || rd32(z->buf + pos) != ZIP_CD_SIG) {
            return NULL;
        }
        uint16_t name_len = rd16(z->buf + pos + 28);
        uint16_t extra_len = rd16(z->buf + pos + 30);
        uint16_t comment_len = rd16(z->buf + pos + 32);

        if ((size_t)name_len == namelen && pos + 46 + namelen <= z->size &&
            memcmp(z->buf + pos + 46, name, namelen) == 0) {
            return z->buf + pos;
        }
        pos += 46u + name_len + extra_len + comment_len;
    }
    return NULL;
}

static int parse_npy_header(const unsigned char *blob, size_t avail, int *ndim,
                            size_t dims[2], size_t *data_offset, char *err,
                            size_t errsz)
{
    if (avail < 10 || blob[0] != 0x93 || memcmp(blob + 1, "NUMPY", 5) != 0) {
        set_err(err, errsz, "not a .npy entry: %s", "");
        return 0;
    }

    unsigned int major = blob[6];
    size_t header_len;
    size_t prefix;
    if (major == 1) {
        header_len = rd16(blob + 8);
        prefix = 10;
    } else if (major == 2 || major == 3) {
        header_len = rd32(blob + 8);
        prefix = 12;
    } else {
        set_err(err, errsz, "unsupported .npy version %s", "");
        return 0;
    }

    if (prefix + header_len > avail) {
        set_err(err, errsz, "truncated .npy header: %s", "");
        return 0;
    }

    char header[512];
    if (header_len >= sizeof header) {
        set_err(err, errsz, ".npy header too long: %s", "");
        return 0;
    }
    memcpy(header, blob + prefix, header_len);
    header[header_len] = '\0';

    if (strstr(header, "<f4") == NULL) {
        set_err(err, errsz, "only little-endian float32 is supported: %s", "");
        return 0;
    }
    if (strstr(header, "True") != NULL) {
        set_err(err, errsz, "Fortran-ordered arrays are not supported: %s", "");
        return 0;
    }

    const char *shape = strstr(header, "'shape'");
    if (shape == NULL) {
        shape = strstr(header, "\"shape\"");
    }
    if (shape == NULL || (shape = strchr(shape, '(')) == NULL) {
        set_err(err, errsz, ".npy header has no shape: %s", "");
        return 0;
    }
    shape++;

    int n = 0;
    while (*shape != '\0' && n <= 2) {
        while (*shape == ' ' || *shape == ',') {
            shape++;
        }
        if (*shape == ')') {
            break;
        }
        char *end = NULL;
        unsigned long v = strtoul(shape, &end, 10);
        if (end == shape) {
            break;
        }
        if (n < 2) {
            dims[n] = (size_t)v;
        }
        n++;
        shape = end;
    }
    if (n <= 0 || n > 2) {
        set_err(err, errsz, "unsupported array rank in .npy entry: %s", "");
        return 0;
    }

    *ndim = n;
    *data_offset = prefix + header_len;
    return 1;
}

int npz_get_f32(const npz_t *z, const char *name, float **values, int *ndim,
                size_t dims[2], char *err, size_t errsz)
{
    const unsigned char *entry = find_entry(z, name);
    if (entry == NULL) {
        set_err(err, errsz, "archive has no entry %s", name);
        return 0;
    }

    uint16_t method = rd16(entry + 10);
    uint32_t local_off = rd32(entry + 42);
    if (method != 0) {
        set_err(err, errsz, "%s is compressed; re-save with np.savez (not _compressed)", name);
        return 0;
    }
    if ((size_t)local_off + 30 > z->size || rd32(z->buf + local_off) != ZIP_LOCAL_SIG) {
        set_err(err, errsz, "corrupt local header for %s", name);
        return 0;
    }

    uint16_t name_len = rd16(z->buf + local_off + 26);
    uint16_t extra_len = rd16(z->buf + local_off + 28);
    size_t data_start = (size_t)local_off + 30u + name_len + extra_len;
    if (data_start > z->size) {
        set_err(err, errsz, "corrupt local header for %s", name);
        return 0;
    }

    size_t data_offset = 0;
    if (!parse_npy_header(z->buf + data_start, z->size - data_start, ndim, dims,
                          &data_offset, err, errsz)) {
        return 0;
    }

    size_t count = 1;
    for (int i = 0; i < *ndim; i++) {
        count *= dims[i];
    }
    if (data_start + data_offset + count * 4 > z->size) {
        set_err(err, errsz, "truncated array data for %s", name);
        return 0;
    }

    float *copy = malloc(count * sizeof(float));
    if (copy == NULL) {
        set_err(err, errsz, "out of memory reading %s", name);
        return 0;
    }
    memcpy(copy, z->buf + data_start + data_offset, count * sizeof(float));
    *values = copy;
    return 1;
}

/* ---------------------------------------------------------------- writing */

static uint32_t crc_table[256];
static int crc_ready = 0;

static uint32_t crc32_bytes(const unsigned char *data, size_t n)
{
    if (!crc_ready) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) {
                if (c & 1u) {
                    c = 0xEDB88320u ^ (c >> 1);
                } else {
                    c >>= 1;
                }
            }
            crc_table[i] = c;
        }
        crc_ready = 1;
    }

    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) {
        c = crc_table[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    }
    return c ^ 0xFFFFFFFFu;
}

/* Builds "<magic><ver><len><dict><pad>\n<data>" for one float32 array. */
static unsigned char *make_npy(const npz_entry_t *e, size_t *out_size)
{
    char shape[64];
    if (e->ndim == 2) {
        snprintf(shape, sizeof shape, "(%zu, %zu)", e->dims[0], e->dims[1]);
    } else {
        snprintf(shape, sizeof shape, "(%zu,)", e->dims[0]);
    }

    char dict[256];
    snprintf(dict, sizeof dict,
             "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);

    size_t dict_len = strlen(dict);
    /* Total header (magic + version + length field + text) is padded to a
     * multiple of 64, with the dict terminated by a newline. */
    size_t fixed = 10;
    size_t pad = 64 - ((fixed + dict_len + 1) % 64);
    if (pad == 64) {
        pad = 0;
    }
    size_t header_len = dict_len + pad + 1;

    size_t count = 1;
    for (int i = 0; i < e->ndim; i++) {
        count *= e->dims[i];
    }
    size_t data_bytes = count * sizeof(float);
    size_t total = fixed + header_len + data_bytes;

    unsigned char *blob = malloc(total);
    if (blob == NULL) {
        return NULL;
    }

    blob[0] = 0x93;
    memcpy(blob + 1, "NUMPY", 5);
    blob[6] = 1;
    blob[7] = 0;
    wr16(blob + 8, (uint16_t)header_len);
    memcpy(blob + 10, dict, dict_len);
    memset(blob + 10 + dict_len, ' ', pad);
    blob[10 + header_len - 1] = '\n';
    memcpy(blob + 10 + header_len, e->values, data_bytes);

    *out_size = total;
    return blob;
}

int npz_write_f32(const char *path, const npz_entry_t *entries, int count,
                  char *err, size_t errsz)
{
    unsigned char **blobs = calloc((size_t)count, sizeof *blobs);
    size_t *blob_sizes = calloc((size_t)count, sizeof *blob_sizes);
    if (blobs == NULL || blob_sizes == NULL) {
        free(blobs);
        free(blob_sizes);
        set_err(err, errsz, "out of memory writing %s", path);
        return 0;
    }

    size_t total = 22;
    for (int i = 0; i < count; i++) {
        blobs[i] = make_npy(&entries[i], &blob_sizes[i]);
        if (blobs[i] == NULL) {
            for (int k = 0; k < i; k++) {
                free(blobs[k]);
            }
            free(blobs);
            free(blob_sizes);
            set_err(err, errsz, "out of memory writing %s", path);
            return 0;
        }
        total += 30 + strlen(entries[i].name) + blob_sizes[i];
        total += 46 + strlen(entries[i].name);
    }

    unsigned char *zip = malloc(total);
    if (zip == NULL) {
        for (int i = 0; i < count; i++) {
            free(blobs[i]);
        }
        free(blobs);
        free(blob_sizes);
        set_err(err, errsz, "out of memory writing %s", path);
        return 0;
    }

    size_t pos = 0;
    size_t *offsets = calloc((size_t)count, sizeof *offsets);
    size_t *crcs = calloc((size_t)count, sizeof *crcs);
    if (offsets == NULL || crcs == NULL) {
        free(offsets);
        free(crcs);
        free(zip);
        for (int i = 0; i < count; i++) {
            free(blobs[i]);
        }
        free(blobs);
        free(blob_sizes);
        set_err(err, errsz, "out of memory writing %s", path);
        return 0;
    }

    /* Local file headers plus data. */
    for (int i = 0; i < count; i++) {
        size_t name_len = strlen(entries[i].name);
        offsets[i] = pos;
        crcs[i] = crc32_bytes(blobs[i], blob_sizes[i]);

        unsigned char *p = zip + pos;
        wr32(p + 0, ZIP_LOCAL_SIG);
        wr16(p + 4, 20);
        wr16(p + 6, 0);
        wr16(p + 8, 0);
        wr16(p + 10, 0);          /* time */
        wr16(p + 12, 0x0021);     /* date: 1980-01-01 */
        wr32(p + 14, crcs[i]);
        wr32(p + 18, (uint32_t)blob_sizes[i]);
        wr32(p + 22, (uint32_t)blob_sizes[i]);
        wr16(p + 26, (uint16_t)name_len);
        wr16(p + 28, 0);
        memcpy(p + 30, entries[i].name, name_len);
        memcpy(p + 30 + name_len, blobs[i], blob_sizes[i]);
        pos += 30 + name_len + blob_sizes[i];
    }

    size_t cd_start = pos;

    /* Central directory. */
    for (int i = 0; i < count; i++) {
        size_t name_len = strlen(entries[i].name);
        unsigned char *p = zip + pos;
        wr32(p + 0, ZIP_CD_SIG);
        wr16(p + 4, 20);
        wr16(p + 6, 20);
        wr16(p + 8, 0);
        wr16(p + 10, 0);
        wr16(p + 12, 0);
        wr16(p + 14, 0x0021);
        wr32(p + 16, crcs[i]);
        wr32(p + 20, (uint32_t)blob_sizes[i]);
        wr32(p + 24, (uint32_t)blob_sizes[i]);
        wr16(p + 28, (uint16_t)name_len);
        wr16(p + 30, 0);
        wr16(p + 32, 0);
        wr16(p + 34, 0);
        wr16(p + 36, 0);
        wr32(p + 38, 0);
        wr32(p + 42, (uint32_t)offsets[i]);
        memcpy(p + 46, entries[i].name, name_len);
        pos += 46 + name_len;
    }

    size_t cd_size = pos - cd_start;

    /* End of central directory. */
    unsigned char *p = zip + pos;
    wr32(p + 0, ZIP_EOCD_SIG);
    wr16(p + 4, 0);
    wr16(p + 6, 0);
    wr16(p + 8, (uint16_t)count);
    wr16(p + 10, (uint16_t)count);
    wr32(p + 12, (uint32_t)cd_size);
    wr32(p + 16, (uint32_t)cd_start);
    wr16(p + 20, 0);
    pos += 22;

    FILE *f = fopen(path, "wb");
    int ok = 0;
    if (f == NULL) {
        set_err(err, errsz, "cannot open %s for writing", path);
    } else if (fwrite(zip, 1, pos, f) != pos) {
        set_err(err, errsz, "short write on %s", path);
        fclose(f);
    } else {
        ok = fclose(f) == 0;
        if (!ok) {
            set_err(err, errsz, "cannot flush %s", path);
        }
    }

    free(offsets);
    free(crcs);
    free(zip);
    for (int i = 0; i < count; i++) {
        free(blobs[i]);
    }
    free(blobs);
    free(blob_sizes);
    return ok;
}
