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
  int n = 0;
  for (int64_t z = c0[2]; z <= c1[2]; ++z)
    for (int64_t y = c0[1]; y <= c1[1]; ++y)
      for (int64_t x = c0[0]; x <= c1[0]; ++x)
      {
        const int64_t cx = cpl_mod(x, d->cuts[0]), cy = cpl_mod(y, d->cuts[1]), cz = cpl_mod(z, d->cuts[2]);
        /* The cell ranges never wrap onto themselves (a periodic axis spanning
         * every cell is enumerated once), and rank_map is a bijection, so
         * every rank appears at most once. */
        if (n < cap) out[n++] = d->rank_map[cx + cy * d->cuts[0] + cz * d->cuts[0] * d->cuts[1]];
      }
  MFTLSortIds(out, n);
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A grid over the directory
 * ═══════════════════════════════════════════════════════════════════════════ */

static int cmp_i64pair(const void *a, const void *b)
{
  const int64_t *x = (const int64_t *) a, *y = (const int64_t *) b;
  if (x[0] != y[0]) return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}

void MFTLCplDirIndexBuild(cpl_dir_index *ix, const cpl_dir_entry *e, int n, int location)
{
  memset(ix, 0, sizeof *ix);
  double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX}, hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX}, wsum[3] = {0, 0, 0};
  int m = 0;
  for (int i = 0; i < n; ++i)
  {
    if (e[i].location != location) continue;
    ++m;
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = fmin(lo[k], e[i].lo[k]);
      hi[k] = fmax(hi[k], e[i].hi[k]);
      wsum[k] += e[i].hi[k] - e[i].lo[k];
    }
  }
  ix->stamp = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(n, 1), sizeof(int), "directory index");
  if (m == 0)
  {
    ix->n_cells = 0;
    ix->off = (int64_t *) MFTLCallocOrDie(1, sizeof(int64_t), "directory index");
    ix->ent = (int *) MFTLMallocOrDie(sizeof(int), "directory index");
    return;
  }
  /* cells about the mean entry size, at most 64 per axis */
  for (int k = 0; k < 3; ++k)
  {
    const double span = fmax(hi[k] - lo[k], 1e-300);
    double c = fmax(wsum[k] / m, span / 64.0);
    if (!(c > 0.0)) c = span;
    ix->lo[k] = lo[k];
    ix->cell[k] = c;
    ix->dims[k] = (int64_t) floor(span / c) + 1;
  }
  ix->n_cells = ix->dims[0] * ix->dims[1] * ix->dims[2];
  int64_t np = 0, cap = 0;
  int64_t *pairs = MFTL_NULL;
  for (int i = 0; i < n; ++i)
  {
    if (e[i].location != location) continue;
    int64_t c0[3], c1[3];
    for (int k = 0; k < 3; ++k)
    {
      c0[k] = (int64_t) floor((e[i].lo[k] - ix->lo[k]) / ix->cell[k]);
      c1[k] = (int64_t) floor((e[i].hi[k] - ix->lo[k]) / ix->cell[k]);
      if (c0[k] < 0) c0[k] = 0;
      if (c1[k] >= ix->dims[k]) c1[k] = ix->dims[k] - 1;
    }
    for (int64_t z = c0[2]; z <= c1[2]; ++z)
      for (int64_t y = c0[1]; y <= c1[1]; ++y)
        for (int64_t x = c0[0]; x <= c1[0]; ++x)
        {
          if (np == cap)
          {
            cap = cap ? 2 * cap : 256;
            pairs = (int64_t *) MFTLReallocOrDie(pairs, 2 * (size_t) cap * sizeof(int64_t), "directory index");
          }
          pairs[2 * np] = x + ix->dims[0] * (y + ix->dims[1] * z);
          pairs[2 * np + 1] = i;
          ++np;
        }
  }
  if (np > 1) qsort(pairs, (size_t) np, 2 * sizeof(int64_t), cmp_i64pair);
  ix->off = (int64_t *) MFTLCallocOrDie((size_t) ix->n_cells + 1, sizeof(int64_t), "directory index");
  ix->ent = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(np, 1) * sizeof(int), "directory index");
  for (int64_t p = 0; p < np; ++p)
  {
    ix->off[pairs[2 * p] + 1]++;
    ix->ent[p] = (int) pairs[2 * p + 1];
  }
  for (int64_t c = 0; c < ix->n_cells; ++c) ix->off[c + 1] += ix->off[c];
  free(pairs);
}

void MFTLCplDirIndexFree(cpl_dir_index *ix)
{
  free(ix->off);
  free(ix->ent);
  free(ix->stamp);
  memset(ix, 0, sizeof *ix);
}

int MFTLCplDirIndexQuery(cpl_dir_index *ix, const cpl_dir_entry *e, const double lo[3], const double hi[3], int **out, int *cap)
{
  int n = 0;
  if (ix->n_cells == 0) return 0;
  int64_t c0[3], c1[3];
  for (int k = 0; k < 3; ++k)
  {
    const double a = floor((lo[k] - ix->lo[k]) / ix->cell[k]), b = floor((hi[k] - ix->lo[k]) / ix->cell[k]);
    if (b < 0.0 || a > (double) (ix->dims[k] - 1)) return 0;
    c0[k] = (a < 0.0) ? 0 : (int64_t) a;
    c1[k] = (b > (double) (ix->dims[k] - 1)) ? ix->dims[k] - 1 : (int64_t) b;
  }
  if (++ix->stamp_now == 0x7fffffff)
  {
    ix->stamp_now = 1;
    /* the stamps are only ever compared for equality with stamp_now */
  }
  for (int64_t z = c0[2]; z <= c1[2]; ++z)
    for (int64_t y = c0[1]; y <= c1[1]; ++y)
      for (int64_t x = c0[0]; x <= c1[0]; ++x)
      {
        const int64_t c = x + ix->dims[0] * (y + ix->dims[1] * z);
        for (int64_t p = ix->off[c]; p < ix->off[c + 1]; ++p)
        {
          const int i = ix->ent[p];
          if (ix->stamp[i] == ix->stamp_now) continue;
          ix->stamp[i] = ix->stamp_now;
          if (!cpl_box_overlap(e[i].lo, e[i].hi, lo, hi)) continue;
          if (n == *cap)
          {
            *cap = (*cap == 0) ? 64 : 2 * *cap;
            *out = (int *) MFTLReallocOrDie(*out, (size_t) *cap * sizeof(int), "directory query");
          }
          (*out)[n++] = i;
        }
      }
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The cover: which parts of a region some ownership box claims
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLCplCoverBuild(cpl_cover *cv, cpl_dir_index *ix, const cpl_dir_entry *e, const MFTL_Domain *d, const double lo[3], const double hi[3],
                       const double cell[3], const double sliver[3])
{
  memset(cv, 0, sizeof *cv);
  int64_t n_cells = 1;
  for (int k = 0; k < 3; ++k)
  {
    const double span = fmax(hi[k] - lo[k], 0.0);
    double c = fmax(cell[k], span / 128.0);
    if (!(c > 0.0)) c = fmax(span, 1e-300);
    cv->lo[k] = lo[k];
    cv->cell[k] = c;
    cv->dims[k] = (int64_t) ceil(span / c);
    if (cv->dims[k] < 1) cv->dims[k] = 1;
    n_cells *= cv->dims[k];
  }
  cv->frag_off = (int64_t *) MFTLCallocOrDie((size_t) n_cells + 1, sizeof(int64_t), "cover");
  int64_t n_frag = 0, cap_frag = 0;
  int cap_rem = 64, n_rem = 0, *got = MFTL_NULL, gcap = 0;
  double *rem = (double *) MFTLMallocOrDie(6 * (size_t) cap_rem * sizeof(double), "cover");
  for (int64_t c = 0; c < n_cells; ++c)
  {
    const int64_t ci[3] = {c % cv->dims[0], (c / cv->dims[0]) % cv->dims[1], c / (cv->dims[0] * cv->dims[1])};
    n_rem = 1;
    for (int k = 0; k < 3; ++k)
    {
      rem[k] = cv->lo[k] + (double) ci[k] * cv->cell[k];
      rem[3 + k] = (ci[k] == cv->dims[k] - 1) ? hi[k] : cv->lo[k] + (double) (ci[k] + 1) * cv->cell[k];
    }
    for (int tz = -1; tz <= 1 && n_rem > 0; ++tz)
      for (int ty = -1; ty <= 1 && n_rem > 0; ++ty)
        for (int tx = -1; tx <= 1 && n_rem > 0; ++tx)
        {
          const int t[3] = {tx, ty, tz};
          bool okimg = true;
          double qlo[3], qhi[3];
          for (int k = 0; k < 3; ++k)
          {
            if (t[k] != 0 && !d->is_periodic[k]) okimg = false;
            qlo[k] = rem[k] - (double) t[k] * d->domain_size[k];
            qhi[k] = rem[3 + k] - (double) t[k] * d->domain_size[k];
          }
          if (!okimg) continue;
          /* the cell, not the remainder: the remainder only shrinks */
          for (int k = 0; k < 3; ++k)
          {
            qlo[k] = cv->lo[k] + (double) ci[k] * cv->cell[k] - (double) t[k] * d->domain_size[k];
            qhi[k] = ((ci[k] == cv->dims[k] - 1) ? hi[k] : cv->lo[k] + (double) (ci[k] + 1) * cv->cell[k]) - (double) t[k] * d->domain_size[k];
          }
          const int ng = MFTLCplDirIndexQuery(ix, e, qlo, qhi, &got, &gcap);
          for (int gi = 0; gi < ng && n_rem > 0; ++gi)
          {
            double eb[6];
            for (int k = 0; k < 3; ++k)
            {
              eb[k] = e[got[gi]].lo[k] + (double) t[k] * d->domain_size[k];
              eb[3 + k] = e[got[gi]].hi[k] + (double) t[k] * d->domain_size[k];
            }
            if (cap_rem < 6 * n_rem + 6)
            {
              cap_rem = 6 * n_rem + 6;
              rem = (double *) MFTLReallocOrDie(rem, 6 * (size_t) cap_rem * sizeof(double), "cover");
            }
            MFTLCplBoxSubtract(rem, &n_rem, cap_rem, eb, sliver);
          }
        }
    if (n_frag + n_rem > cap_frag)
    {
      cap_frag = MFTL_MAX(2 * cap_frag, n_frag + n_rem + 16);
      cv->frag = (double *) MFTLReallocOrDie(cv->frag, 6 * (size_t) cap_frag * sizeof(double), "cover");
    }
    memcpy(&cv->frag[6 * n_frag], rem, 6 * (size_t) n_rem * sizeof(double));
    n_frag += n_rem;
    cv->frag_off[c + 1] = n_frag;
  }
  free(rem);
  free(got);
  cv->built = true;
}

void MFTLCplCoverFree(cpl_cover *cv)
{
  free(cv->frag_off);
  free(cv->frag);
  memset(cv, 0, sizeof *cv);
}

/** @brief Appends one box to a growable list. */
static void emit_box(double **out, int *n_out, int *cap, const double b[6])
{
  if (*n_out == *cap)
  {
    *cap = (*cap == 0) ? 16 : 2 * *cap;
    *out = (double *) MFTLReallocOrDie(*out, 6 * (size_t) *cap * sizeof(double), "cover");
  }
  memcpy(&(*out)[6 * (*n_out)++], b, 6 * sizeof(double));
}

int MFTLCplCoverQuery(const cpl_cover *cv, const double lo[3], const double hi[3], const double sliver[3], double **out, int *n_out, int *cap)
{
  int added = 0;

  /* outside the cover's region: report it, it was never checked */
  for (int k = 0; k < 3; ++k)
  {
    const double rlo = cv->lo[k], rhi = cv->lo[k] + (double) cv->dims[k] * cv->cell[k];
    if (lo[k] < rlo - sliver[k] || hi[k] > rhi + sliver[k])
    {
      const double b[6] = {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
      emit_box(out, n_out, cap, b);
      return 1;
    }
  }
  int64_t c0[3], c1[3];
  for (int k = 0; k < 3; ++k)
  {
    c0[k] = (int64_t) floor((lo[k] - cv->lo[k]) / cv->cell[k]);
    c1[k] = (int64_t) floor((hi[k] - cv->lo[k]) / cv->cell[k]);
    if (c0[k] < 0) c0[k] = 0;
    if (c1[k] >= cv->dims[k]) c1[k] = cv->dims[k] - 1;
  }
  for (int64_t z = c0[2]; z <= c1[2]; ++z)
    for (int64_t y = c0[1]; y <= c1[1]; ++y)
      for (int64_t x = c0[0]; x <= c1[0]; ++x)
      {
        const int64_t c = x + cv->dims[0] * (y + cv->dims[1] * z);
        for (int64_t f = cv->frag_off[c]; f < cv->frag_off[c + 1]; ++f)
        {
          const double *fb = &cv->frag[6 * f];
          double b[6];
          bool thin = false;
          for (int k = 0; k < 3; ++k)
          {
            b[k] = fmax(fb[k], lo[k]);
            b[3 + k] = fmin(fb[3 + k], hi[k]);
            if (b[3 + k] - b[k] < sliver[k]) thin = true;
          }
          if (!thin)
          {
            emit_box(out, n_out, cap, b);
            ++added;
          }
        }
      }
  return added;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * This rank's directory entries
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLCplDirectoryOwnEntries(const MFTL_SolverGrid *g, const MFTL_Domain *d, cpl_dir_entry **out)
{
  (void) d;
  int cap = 16, n = 0;
  cpl_dir_entry *e = (cpl_dir_entry *) MFTLMallocOrDie((size_t) cap * sizeof(cpl_dir_entry), "directory entries");
#define CPL_PUSH()                                                                                              \
  do                                                                                                            \
  {                                                                                                             \
    if (n == cap)                                                                                               \
    {                                                                                                           \
      cap *= 2;                                                                                                 \
      e = (cpl_dir_entry *) MFTLReallocOrDie(e, (size_t) cap * sizeof(cpl_dir_entry), "directory entries");      \
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
  }
#undef CPL_PUSH
  *out = e;
  return n;
}
