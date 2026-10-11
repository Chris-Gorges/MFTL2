/**
 * @file coupling_directory.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief Geometry the coupler routes and certifies with: lattice index ranges
 *        and their periodic images, box subtraction, the ranks a box touches,
 *        and a rank's own directory entries.
 *
 * ONE POSITION FORMULA. An image of lattice sample `i` (unwrapped) sits at the
 * canonical coordinate of `i mod N` plus `floor(i/N)` periods -- never at
 * `origin + i*spacing`, which differs in the last bit across the seam. Index
 * ranges are found by estimate-and-step against that same formula, so a sample
 * exactly on a request box's face is in or out identically on every rank that
 * evaluates it.
 */

#include "coupling_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Lattice arithmetic
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTLCplLatCoord(const cpl_location *l, int k, int64_t j)
{
  if (l->kind == MFTL_LATTICE_UNIFORM) return l->origin[k] + (double) j * l->spacing[k];
  return l->coords[k][j];
}

double MFTLCplLatWidth(const cpl_location *l, int k, int64_t j)
{
  if (l->kind == MFTL_LATTICE_UNIFORM) return l->spacing[k];
  return l->faces[k][j + 1] - l->faces[k][j];
}

double MFTLCplLatPos(const cpl_location *l, const MFTL_Domain *d, int k, int64_t i)
{
  if (!d->is_periodic[k]) return MFTLCplLatCoord(l, k, i);
  const int64_t n = l->dims[k];
  const int64_t s = cpl_floordiv(i, n);
  return MFTLCplLatCoord(l, k, i - s * n) + (double) s * d->domain_size[k];
}

/** @brief First index of an ascending table not below @p x. */
static int64_t lower_bound(const double *t, int64_t n, double x)
{
  int64_t a = 0, b = n;
  while (a < b)
  {
    const int64_t m = a + (b - a) / 2;
    if (t[m] < x)
      a = m + 1;
    else
      b = m;
  }
  return a;
}

/** @brief An index near the first sample at or above @p x (unwrapped on a periodic axis). */
static int64_t estimate(const cpl_location *l, const MFTL_Domain *d, int k, double x)
{
  const int64_t n = l->dims[k];
  if (l->kind == MFTL_LATTICE_UNIFORM)
  {
    double e = floor((x - l->origin[k]) / l->spacing[k]);
    const double cap = 8.0 * (double) n + 8.0;
    if (e < -cap) e = -cap;
    if (e > cap) e = cap;
    return (int64_t) e;
  }
  if (!d->is_periodic[k]) return lower_bound(l->coords[k], n, x);
  double s = floor((x - d->global_min[k]) * d->inv_domain_size[k]);
  if (s < -8.0) s = -8.0;
  if (s > 8.0) s = 8.0;
  const double xc = x - s * d->domain_size[k];
  return lower_bound(l->coords[k], n, xc) + (int64_t) s * n;
}

/** @brief The first index whose position is >= @p x (> when @p strict). */
static int64_t first_at_or_above(const cpl_location *l, const MFTL_Domain *d, int k, double x, bool strict)
{
  const bool per = d->is_periodic[k];
  const int64_t n = l->dims[k];
  int64_t i = estimate(l, d, k, x);
  if (!per)
  {
    if (i < 0) i = 0;
    if (i > n) i = n;
  }
#define CPL_ABOVE(idx) (strict ? (MFTLCplLatPos(l, d, k, (idx)) > x) : (MFTLCplLatPos(l, d, k, (idx)) >= x))
  /* step down while the previous one still qualifies */
  while ((per || i > 0) && CPL_ABOVE(i - 1)) --i;
  /* step up while this one does not */
  while ((per || i < n) && !CPL_ABOVE(i)) ++i;
#undef CPL_ABOVE
  return i;
}

bool MFTLCplLatRange(const cpl_location *l, const MFTL_Domain *d, int k, double lo, double hi, int64_t *a, int64_t *b)
{
  if (!(lo <= hi)) return false;
  *a = first_at_or_above(l, d, k, lo, false);
  *b = first_at_or_above(l, d, k, hi, true);
  return *a < *b;
}

int64_t MFTLCplLatDistinct(const cpl_location *l, const MFTL_Domain *d, int k, int64_t a, int64_t b)
{
  if (b <= a) return 0;
  if (d->is_periodic[k]) return MFTL_MIN(b - a, l->dims[k]);
  return b - a;
}

int MFTLCplLatIntersect(const cpl_location *l, const MFTL_Domain *d, int k, int64_t a, int64_t b, int64_t olo, int64_t ohi, int64_t *lo_out, int64_t *hi_out,
                      int max)
{
  int m = 0;
  if (b <= a || ohi <= olo || max < 2) return 0;
  if (!d->is_periodic[k])
  {
    const int64_t x = MFTL_MAX(a, olo), y = MFTL_MIN(b, ohi);
    if (x < y)
    {
      lo_out[m] = x;
      hi_out[m] = y;
      ++m;
    }
    return m;
  }
  const int64_t n = l->dims[k];
  int64_t ia[2], ib[2];
  int ni = 0;
  if (b - a >= n)
  {
    ia[0] = 0;
    ib[0] = n;
    ni = 1;
  }
  else
  {
    const int64_t ja = cpl_mod(a, n), len = b - a;
    if (ja + len <= n)
    {
      ia[0] = ja;
      ib[0] = ja + len;
      ni = 1;
    }
    else
    {
      ia[0] = ja;
      ib[0] = n;
      ia[1] = 0;
      ib[1] = ja + len - n;
      ni = 2;
    }
  }
  for (int t = 0; t < ni; ++t)
  {
    const int64_t x = MFTL_MAX(ia[t], olo), y = MFTL_MIN(ib[t], ohi);
    if (x < y)
    {
      lo_out[m] = x;
      hi_out[m] = y;
      ++m;
    }
  }
  return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Box subtraction
 * ═══════════════════════════════════════════════════════════════════════════ */

bool MFTLCplBoxSubtract(double *rem, int *n_rem, int cap, const double *sub, const double sliver[3])
{
  int n = *n_rem;
  bool ok = true;
  for (int i = 0; i < n;)
  {
    double *r = &rem[6 * i];
    /* Require positive overlap in each nondegenerate dimension. Merely
     * touching faces must not regenerate the same piece forever. A remainder
     * that is itself a plane, line or point instead needs inclusive coverage
     * in its degenerate dimensions (e.g. a provider on an empty rank). */
    bool cut = true;
    bool degenerate[3];
    for (int k = 0; k < 3; ++k)
    {
      degenerate[k] = (r[k] == r[3 + k]);
      if (degenerate[k])
      {
        if (r[k] < sub[k] || r[k] > sub[3 + k]) cut = false;
      }
      else if (!(fmax(r[k], sub[k]) < fmin(r[3 + k], sub[3 + k])))
        cut = false;
    }
    if (!cut)
    {
      ++i;
      continue;
    }
    /* Remove r (swap with the last), then append its pieces outside sub. */
    double box[6];
    memcpy(box, r, sizeof box);
    memcpy(r, &rem[6 * (n - 1)], sizeof box);
    --n;
    for (int k = 0; k < 3; ++k)
    {
      if (box[k] < sub[k])
      {
        double piece[6];
        memcpy(piece, box, sizeof piece);
        piece[3 + k] = sub[k];
        bool thin = false;
        for (int q = 0; q < 3; ++q)
          if (!degenerate[q] && piece[3 + q] - piece[q] < sliver[q]) thin = true;
        if (!thin)
        {
          if (n < cap)
            memcpy(&rem[6 * n++], piece, sizeof piece);
          else
            ok = false;
        }
        box[k] = sub[k];
      }
      if (box[3 + k] > sub[3 + k])
      {
        double piece[6];
        memcpy(piece, box, sizeof piece);
        piece[k] = sub[3 + k];
        bool thin = false;
        for (int q = 0; q < 3; ++q)
          if (!degenerate[q] && piece[3 + q] - piece[q] < sliver[q]) thin = true;
        if (!thin)
        {
          if (n < cap)
            memcpy(&rem[6 * n++], piece, sizeof piece);
          else
            ok = false;
        }
        box[3 + k] = sub[3 + k];
      }
    }
    /* the middle part is covered; do not advance i: slot i now holds another box */
  }
  *n_rem = n;
  return ok;
}

bool MFTLCplIboxSubtract(int64_t *rem, int *n_rem, int cap, const int64_t *sub)
{
  int n = *n_rem;
  bool ok = true;
  for (int i = 0; i < n;)
  {
    int64_t *r = &rem[6 * i];
    bool overlap = true;
    for (int k = 0; k < 3; ++k)
      if (r[3 + k] <= sub[k] || sub[3 + k] <= r[k]) overlap = false;
    if (!overlap)
    {
      ++i;
      continue;
    }
    int64_t box[6];
    memcpy(box, r, sizeof box);
    memcpy(r, &rem[6 * (n - 1)], sizeof box);
    --n;
    for (int k = 0; k < 3; ++k)
    {
      if (box[k] < sub[k])
      {
        int64_t piece[6];
        memcpy(piece, box, sizeof piece);
        piece[3 + k] = sub[k];
        if (n < cap)
          memcpy(&rem[6 * n++], piece, sizeof piece);
        else
          ok = false;
        box[k] = sub[k];
      }
      if (box[3 + k] > sub[3 + k])
      {
        int64_t piece[6];
        memcpy(piece, box, sizeof piece);
        piece[k] = sub[3 + k];
        if (n < cap)
          memcpy(&rem[6 * n++], piece, sizeof piece);
        else
          ok = false;
        box[3 + k] = sub[3 + k];
      }
    }
  }
  *n_rem = n;
  return ok;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The process grid
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLCplRanksTouching(const MFTL_Domain *d, const double lo[3], const double hi[3], int *out, int cap)
{
  int64_t c0[3], c1[3];
  for (int k = 0; k < 3; ++k)
  {
    const int64_t cuts = d->cuts[k];
    const double f = d->grid_factors[k];
    double a = floor((lo[k] - d->global_min[k]) * f), b = floor((hi[k] - d->global_min[k]) * f);
    if (d->is_periodic[k])
    {
      if (b - a + 1.0 >= (double) cuts)
      {
        c0[k] = 0;
        c1[k] = cuts - 1;
      }
      else
      {
        c0[k] = (int64_t) a;
        c1[k] = (int64_t) b;
      }
    }
    else
    {
      if (a < 0.0) a = 0.0;
      if (b < 0.0) b = 0.0;
      if (a > (double) (cuts - 1)) a = (double) (cuts - 1);
      if (b > (double) (cuts - 1)) b = (double) (cuts - 1);
      c0[k] = (int64_t) a;
      c1[k] = (int64_t) b;
    }
  }
  /* The cell ranges never wrap onto themselves (a periodic axis spanning every
   * cell is enumerated once), and rank_map is a bijection, so every rank
   * appears at most once and the count is the product of the ranges. */
  const int64_t total = (c1[0] - c0[0] + 1) * (c1[1] - c0[1] + 1) * (c1[2] - c0[2] + 1);
  if (total > (int64_t) cap) return (int) total;
  int n = 0;
  for (int64_t z = c0[2]; z <= c1[2]; ++z)
    for (int64_t y = c0[1]; y <= c1[1]; ++y)
      for (int64_t x = c0[0]; x <= c1[0]; ++x)
      {
        const int64_t cx = cpl_mod(x, d->cuts[0]), cy = cpl_mod(y, d->cuts[1]), cz = cpl_mod(z, d->cuts[2]);
        out[n++] = d->rank_map[cx + cy * d->cuts[0] + cz * d->cuts[0] * d->cuts[1]];
      }
  MFTLSortIds(out, n);
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Compact bounds trees over directory certificates
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  uint64_t key;
  int entry;
} dir_order;

static int cmp_dir_order(const void *a, const void *b)
{
  const dir_order *x = a, *y = b;
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  return (x->entry > y->entry) - (x->entry < y->entry);
}

void MFTLCplDirIndexBuild(cpl_dir_index *ix, const cpl_dir_entry *e, int n, int location)
{
  memset(ix, 0, sizeof *ix);
  double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX}, hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int i = 0; i < n; ++i)
    if (e[i].location == location)
    {
      ix->n++;
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(lo[k], e[i].mlo[k]);
        hi[k] = fmax(hi[k], e[i].mhi[k]);
      }
    }
  dir_order *order = MFTLCplAlloc((size_t) MFTL_MAX(ix->n, 1) * sizeof *order, "directory index");
  int at = 0;
  for (int i = 0; i < n; ++i)
    if (e[i].location == location)
    {
      uint64_t u[3];
      for (int k = 0; k < 3; ++k)
      {
        const double x = 0.5 * (e[i].mlo[k] + e[i].mhi[k]);
        const double f = hi[k] > lo[k] ? (x - lo[k]) / (hi[k] - lo[k]) : 0.0;
        u[k] = (uint64_t) (fmax(0.0, fmin(1.0, f)) * MFTL_MORTON_MAX);
      }
      order[at++] = (dir_order) {MFTLMorton3D(u[0], u[1], u[2]), i};
    }
  if (ix->n > 1) qsort(order, (size_t) ix->n, sizeof *order, cmp_dir_order);
  ix->ent = MFTLCplAlloc((size_t) MFTL_MAX(ix->n, 1) * sizeof(int), "directory index");
  double *boxes = MFTLCplAlloc(6 * (size_t) MFTL_MAX(ix->n, 1) * sizeof(double), "directory index");
  for (int i = 0; i < ix->n; ++i)
  {
    ix->ent[i] = order[i].entry;
    for (int k = 0; k < 3; ++k)
    {
      boxes[6 * i + k] = e[ix->ent[i]].mlo[k];
      boxes[6 * i + 3 + k] = e[ix->ent[i]].mhi[k];
    }
  }
  MFTLCplBoxTreeBuild(&ix->tree, boxes, ix->n, 4, false);
  MFTLCplFree(boxes);
  MFTLCplFree(order);
}

void MFTLCplDirIndexFree(cpl_dir_index *ix)
{
  MFTLCplFree(ix->tree.box);
  MFTLCplFree(ix->ent);
  memset(ix, 0, sizeof *ix);
}

static int directory_query(cpl_dir_index *ix, const cpl_dir_entry *e, const double lo[3], const double hi[3], bool metric, int **out, int *cap)
{
  int n = 0;
  int64_t cursor = 1, leaf;
  while ((leaf = MFTLCplBoxTreeNext(&ix->tree, lo, hi, &cursor)) >= 0)
    for (int64_t j = leaf * ix->tree.group; j < MFTL_MIN((leaf + 1) * ix->tree.group, ix->n); ++j)
    {
      const int i = ix->ent[j];
      if (!(e[i].roles & (metric ? 2 : 1))) continue;
      if (!cpl_box_overlap(metric ? e[i].mlo : e[i].lo, metric ? e[i].mhi : e[i].hi, lo, hi)) continue;
      if (n == *cap)
      {
        *cap = *cap ? 2 * *cap : 32;
        *out = MFTLCplRealloc(*out, (size_t) *cap * sizeof(int), "directory query");
      }
      (*out)[n++] = i;
    }
  return n;
}

int MFTLCplDirIndexQuery(cpl_dir_index *ix, const cpl_dir_entry *e, const double lo[3], const double hi[3], int **out, int *cap)
{
  return directory_query(ix, e, lo, hi, false, out, cap);
}

int MFTLCplDirMetricQuery(cpl_dir_index *ix, const cpl_dir_entry *e, const double lo[3], const double hi[3], int **out, int *cap)
{
  return directory_query(ix, e, lo, hi, true, out, cap);
}

int MFTLCplDirectoryOwnEntries(const MFTL_SolverGrid *g, const MFTL_Domain *d, double pilot, cpl_dir_entry **out)
{
  int cap = 16, n = 0;
  cpl_dir_entry *e = (cpl_dir_entry *) MFTLCplAlloc((size_t) cap * sizeof(cpl_dir_entry), "directory entries");
#define CPL_PUSH()                                                                                              \
  do                                                                                                            \
  {                                                                                                             \
    if (n == cap)                                                                                               \
    {                                                                                                           \
      cap *= 2;                                                                                                 \
      e = (cpl_dir_entry *) MFTLCplRealloc(e, (size_t) cap * sizeof(cpl_dir_entry), "directory entries");      \
    }                                                                                                           \
    memset(&e[n], 0, sizeof e[n]);                                                                              \
  } while (0)

  for (int i = 0; i < g->n_blocks; ++i)
  {
    const cpl_block *b = &g->blocks[i];
    const cpl_location *l = &g->loc[b->location];
    if (b->hi[0] <= b->lo[0] || b->hi[1] <= b->lo[1] || b->hi[2] <= b->lo[2]) continue;
    CPL_PUSH();
    cpl_dir_entry *x = &e[n++];
    x->location = b->location;
    x->roles = l->kind == MFTL_LATTICE_UNIFORM ? 3 : 1;
    for (int k = 0; k < 3; ++k)
    {
      x->ilo[k] = b->lo[k];
      x->ihi[k] = b->hi[k];
      x->lo[k] = MFTLCplLatCoord(l, k, b->lo[k]) - MFTLCplLatWidth(l, k, b->lo[k]);
      x->hi[k] = MFTLCplLatCoord(l, k, b->hi[k] - 1) + MFTLCplLatWidth(l, k, b->hi[k] - 1);
      x->hmin[k] = MFTL_DBL_MAX;
      x->hmax[k] = 0.0;
      for (int64_t j = b->lo[k]; j < b->hi[k]; ++j)
      {
        const double w = MFTLCplLatWidth(l, k, j);
        x->hmin[k] = fmin(x->hmin[k], w);
        x->hmax[k] = fmax(x->hmax[k], w);
        if (l->kind == MFTL_LATTICE_UNIFORM) break;
      }
    }
  }

  for (int li = 0; li < g->n_loc; ++li)
  {
    const cpl_location *l = &g->loc[li];
    if (cpl_is_lattice(l)) continue;
    double hmin[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX}, hmax[3] = {0.0, 0.0, 0.0};
    for (int pc = 0; pc < l->n_pieces; ++pc)
    {
      const cpl_points *p = &g->points[l->piece_ref[pc]];
      if (p->n == 0) continue;
      for (int k = 0; k < 3; ++k)
      {
        hmin[k] = fmin(hmin[k], p->tile_hmin[k]);
        hmax[k] = fmax(hmax[k], p->tile_hmax[k]);
      }
    }
    for (int k = 0; k < 3; ++k)
    {
      if (l->prov_hmin[k] > 0.0) hmin[k] = fmin(hmin[k], l->prov_hmin[k]);
      if (l->prov_hmax[k] > 0.0) hmax[k] = fmax(hmax[k], l->prov_hmax[k]);
    }
    if (l->n_prov > 0)
    {
      for (int r = 0; r < l->n_prov; ++r)
      {
        CPL_PUSH();
        cpl_dir_entry *x = &e[n++];
        x->location = li;
        x->roles = l->exposure == MFTL_GRID_FULL ? 1 : 3;
        for (int k = 0; k < 3; ++k)
        {
          x->lo[k] = l->prov_boxes[6 * r + k];
          x->hi[k] = l->prov_boxes[6 * r + 3 + k];
          x->hmin[k] = hmin[k];
          x->hmax[k] = hmax[k];
        }
      }
    }
    else
    {
      for (int pc = 0; pc < l->n_pieces; ++pc)
      {
        const cpl_points *p = &g->points[l->piece_ref[pc]];
        for (int t = 0; t < p->n_tiles; ++t)
        {
          CPL_PUSH();
          cpl_dir_entry *x = &e[n++];
          x->location = li;
          x->roles = 3;
          for (int k = 0; k < 3; ++k)
          {
            x->lo[k] = p->tile_ebox[6 * t + k];
            x->hi[k] = p->tile_ebox[6 * t + 3 + k];
            x->hmin[k] = p->tile_h[6 * t + k];
            x->hmax[k] = p->tile_h[6 * t + 3 + k];
          }
        }
      }
    }
    if (l->n_prov > 0 && l->exposure == MFTL_GRID_FULL)
      for (int pc = 0; pc < l->n_pieces; ++pc)
      {
        const cpl_points *p = &g->points[l->piece_ref[pc]];
        for (int t = 0; t < p->n_tiles; ++t)
        {
          CPL_PUSH();
          cpl_dir_entry *x = &e[n++];
          x->location = li;
          x->roles = 2;
          for (int k = 0; k < 3; ++k)
          {
            x->lo[k] = p->tile_pbox[6 * t + k];
            x->hi[k] = p->tile_pbox[6 * t + 3 + k];
            x->hmin[k] = p->tile_h[6 * t + k];
            x->hmax[k] = p->tile_h[6 * t + 3 + k];
          }
        }
      }

  }
  /* A rectilinear lattice declares every position and width independently of
   * ownership. Publish its metric tiles even across ownership holes: otherwise
   * an unowned coarse cell could disappear from the pilot and shrink a request
   * until that very hole escaped the coverage check. Share tile construction
   * across ranks; these role-2 records certify resolution, never ownership. */
  for (int li = 0; li < g->n_loc; ++li)
  {
    const cpl_location *l = &g->loc[li];
    if (l->kind != MFTL_LATTICE_RECTILINEAR) continue;
    const int64_t dims[3] = {(l->dims[0] + 7) / 8, (l->dims[1] + 7) / 8, (l->dims[2] + 7) / 8};
    const int64_t total = dims[0] * dims[1] * dims[2];
    for (int64_t t = d->rank; t < total; t += d->nproc)
    {
      CPL_PUSH();
      cpl_dir_entry *x = &e[n++];
      x->location = li;
      x->roles = 2;
      const int64_t start[3] = {8 * (t % dims[0]), 8 * ((t / dims[0]) % dims[1]), 8 * (t / (dims[0] * dims[1]))};
      for (int k = 0; k < 3; ++k)
      {
        const int64_t end = MFTL_MIN(start[k] + 8, l->dims[k]);
        x->lo[k] = MFTLCplLatCoord(l, k, start[k]);
        x->hi[k] = MFTLCplLatCoord(l, k, end - 1);
        x->hmin[k] = MFTL_DBL_MAX;
        for (int64_t j = start[k]; j < end; ++j)
        {
          const double h = MFTLCplLatWidth(l, k, j);
          x->hmin[k] = fmin(x->hmin[k], h);
          x->hmax[k] = fmax(x->hmax[k], h);
        }
      }
    }
  }
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k)
    {
      const double r = e[i].roles & 2 ? pilot * e[i].hmax[k] : 0.0;
      e[i].mlo[k] = e[i].lo[k] - r;
      e[i].mhi[k] = e[i].hi[k] + r;
    }

#undef CPL_PUSH
  *out = e;
  return n;
}
