/**
 * @file coupling_stencil.c
 * @ingroup mftl_coupling
 * @brief Direct Cartesian stencils over sparse bricks of original fluid samples.
 *
 * The bricks contain band column indices, never interpolated fluid values.
 * Their ownership is immaterial: the window's normal gather fills those columns.
 * A complete uniform stencil evaluates twelve one-dimensional kernel weights;
 * boundaries, missing samples and nonuniform metrics use the common row builder.
 */
#include "coupling_internal.h"
#include "mftl_memory.h"
#include <stdlib.h>

#define BRICK_SIDE 4
#define BRICK_SIZE 64

typedef struct
{
  uint64_t key;
  int64_t col;
  int slot;
} brick_entry;

static int cmp_brick(const void *a, const void *b)
{
  const brick_entry *x = a, *y = b;
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  return (x->slot > y->slot) - (x->slot < y->slot);
}

static uint64_t brick_key(const cpl_location *l, const int64_t j[3])
{
  const uint64_t nx = ((uint64_t) l->dims[0] + 3) / BRICK_SIDE;
  const uint64_t ny = ((uint64_t) l->dims[1] + 3) / BRICK_SIDE;
  return (uint64_t) (j[0] / BRICK_SIDE) + nx * ((uint64_t) (j[1] / BRICK_SIDE) + ny * (uint64_t) (j[2] / BRICK_SIDE));
}

static uint64_t brick_hash(uint64_t key)
{
  key ^= key >> 30;
  key *= UINT64_C(0xbf58476d1ce4e5b9);
  key ^= key >> 27;
  key *= UINT64_C(0x94d049bb133111eb);
  return key ^ (key >> 31);
}

void MFTLCplBricksBuild(cpl_band *b, const cpl_location *l)
{
  brick_entry *e = MFTLCplAlloc((size_t) MFTL_MAX(b->n_dof, 1) * sizeof *e, "Cartesian bricks");
  for (int64_t i = 0; i < b->n_dof; ++i)
  {
    int64_t j[3];
    cpl_lat_unkey(l, b->key[i], j);
    e[i].key = brick_key(l, j);
    e[i].slot = (int) ((j[0] % 4) + 4 * ((j[1] % 4) + 4 * (j[2] % 4)));
    e[i].col = i;
  }
  if (b->n_dof > 1) qsort(e, (size_t) b->n_dof, sizeof *e, cmp_brick);
  for (int64_t i = 0; i < b->n_dof; ++i)
    if (i == 0 || e[i].key != e[i - 1].key) ++b->n_bricks;
  b->brick_key = MFTLCplAlloc((size_t) MFTL_MAX(b->n_bricks, 1) * sizeof(uint64_t), "Cartesian bricks");
  b->brick_col = MFTLCplAlloc((size_t) MFTL_MAX(b->n_bricks, 1) * BRICK_SIZE * sizeof(int64_t), "Cartesian bricks");
  for (int64_t i = 0; i < b->n_bricks * BRICK_SIZE; ++i) b->brick_col[i] = -1;
  b->brick_hash_size = 1;
  while (b->brick_hash_size < 2 * b->n_bricks) b->brick_hash_size *= 2;
  b->brick_hash = MFTLCplCalloc((size_t) b->brick_hash_size, sizeof(int64_t), "Cartesian bricks");
  int64_t n = -1;
  for (int64_t i = 0; i < b->n_dof; ++i)
  {
    if (i == 0 || e[i].key != e[i - 1].key)
    {
      b->brick_key[++n] = e[i].key;
      uint64_t h = brick_hash(e[i].key) & (uint64_t) (b->brick_hash_size - 1);
      while (b->brick_hash[h]) h = (h + 1) & (uint64_t) (b->brick_hash_size - 1);
      b->brick_hash[h] = n + 1;
    }
    b->brick_col[n * BRICK_SIZE + e[i].slot] = e[i].col;
  }
  MFTLCplFree(e);
}

void MFTLCplBricksFree(cpl_band *b)
{
  MFTLCplFree(b->brick_key);
  MFTLCplFree(b->brick_col);
  MFTLCplFree(b->brick_hash);
}

static int64_t find_brick(const cpl_band *b, uint64_t key)
{
  uint64_t h = brick_hash(key) & (uint64_t) (b->brick_hash_size - 1);
  while (b->brick_hash[h])
  {
    const int64_t j = b->brick_hash[h] - 1;
    if (b->brick_key[j] == key) return j;
    h = (h + 1) & (uint64_t) (b->brick_hash_size - 1);
  }
  return -1;
}

int MFTLCplCartesianRow(const cpl_band *b, const cpl_location *l, const MFTL_Domain *d, const cpl_kernel *kernel,
                        const double x[3], double tolerance, double max_amplification, cpl_tensor_row *row, double *residual, double *amplification)
{
  if (!b->brick_hash || b->n_dof > INT32_MAX) return -1;
  row->count = 0;
  int64_t index[3][4];
  double r[12];
  double *phi = row->phi;
  for (int a = 0; a < 3; ++a)
  {
    const double t = (x[a] - l->origin[a]) / l->spacing[a];
    if (!isfinite(t) || t < -(double) l->dims[a] || t > 2.0 * (double) l->dims[a]) return -1;
    const int64_t base = (int64_t) floor(t) - 1;
    for (int i = 0; i < 4; ++i)
    {
      const int64_t raw = base + i;
      const int64_t j = d->is_periodic[a] ? cpl_mod(raw, l->dims[a]) : raw;
      index[a][i] = j;
      /* Use the same coordinates as the general periodic-image path. */
      double p = l->origin[a] + (double) j * l->spacing[a];
      if (d->is_periodic[a]) p += (double) ((raw - j) / l->dims[a]) * d->domain_size[a];
      r[4 * a + i] = (p - x[a]) / l->spacing[a];
    }
  }
  kernel->eval(r, phi, 12, kernel->ctx);
  for (int i = 0; i < 12; ++i)
    if (!isfinite(phi[i]) || phi[i] < 0.0) return -1;
  /* Up to 27 brick combinations at a periodic seam of a non-multiple-of-four
   * lattice. Cache each lookup here; the 64 stencil entries use arithmetic. */
  uint64_t keys[64];
  int64_t bricks[64];
  int nb = 0, n = 0;
  double moments[4] = {0, 0, 0, 0};
  for (int k = 0; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
      {
        const double v = phi[i] * phi[4 + j] * phi[8 + k];
        const int entry = i + 4 * (j + 4 * k);
        row->col[entry] = -1;
        if (v == 0.0) continue;
        const int64_t ix[3] = {index[0][i], index[1][j], index[2][k]};
        for (int a = 0; a < 3; ++a)
          if (ix[a] < 0 || ix[a] >= l->dims[a]) return -1;
        const uint64_t key = brick_key(l, ix);
        int at = 0;
        while (at < nb && keys[at] != key) ++at;
        if (at == nb)
        {
          keys[nb] = key;
          bricks[nb++] = find_brick(b, key);
        }
        if (bricks[at] < 0) return -1;
        const int slot = (int) ((ix[0] % 4) + 4 * ((ix[1] % 4) + 4 * (ix[2] % 4)));
        const int64_t c = b->brick_col[BRICK_SIZE * bricks[at] + slot];
        if (c < 0) return -1; /* a wall/mask or a coverage failure must take the checked path */
        row->col[entry] = (int32_t) c;
        ++n;
        moments[0] += v;
        moments[1] += v * r[i];
        moments[2] += v * r[4 + j];
        moments[3] += v * r[8 + k];
      }
  *residual = fabs(moments[0] - 1.0);
  for (int a = 1; a < 4; ++a) *residual = fmax(*residual, fabs(moments[a]));
  if (*residual > tolerance || moments[0] > max_amplification) return -1;
  row->count = n;
  *amplification = moments[0];
  return n;
}

/** @brief Exactly the multiplication order used to check the row's moments. */
static inline double tensor_weight(const cpl_tensor_row *r, int e)
{
  return r->phi[e & 3] * r->phi[4 + ((e >> 2) & 3)] * r->phi[8 + (e >> 4)];
}

double MFTLCplTensorValue(const cpl_tensor_row *r, const double *v)
{
  double sum = 0.0;
  for (int e = 0; e < 64; ++e)
    if (r->col[e] >= 0) sum += tensor_weight(r, e) * v[r->col[e]];
  return sum;
}

void MFTLCplTensorSpread(const cpl_tensor_row *r, double load, double *v)
{
  for (int e = 0; e < 64; ++e)
    if (r->col[e] >= 0) v[r->col[e]] += tensor_weight(r, e) * load;
}

void MFTLCplTensorEntries(const cpl_tensor_row *r, int64_t col[64], double weights[64])
{
  int n = 0;
  for (int e = 0; e < 64; ++e)
    if (r->col[e] >= 0)
    {
      col[n] = r->col[e];
      weights[n++] = tensor_weight(r, e);
    }
}
