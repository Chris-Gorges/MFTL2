/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief Fixtures and independent references shared by the coupling tests.
 *
 * Nothing here calls the coupler's internals: the references (kernel formulas,
 * the moment solve, brute-force rows over the whole lattice) are written out
 * again so that a bug in the library cannot appear on both sides of a
 * comparison.
 */

#ifndef MFTL_TEST_COUPLING_FIXTURES_H
#define MFTL_TEST_COUPLING_FIXTURES_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_coupling.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/* ═════════════════════════════════════════════════════════════════════════════
 * Independent references
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Peskin's 4-point function, written out again. */
static double ref_peskin(double r)
{
  r = fabs(r);
  if (r >= 2.0) return 0.0;
  if (r <= 1.0) return (3.0 - 2.0 * r + sqrt(1.0 + 4.0 * r - 4.0 * r * r)) / 8.0;
  return (5.0 - 2.0 * r - sqrt(-7.0 + 12.0 * r - 4.0 * r * r)) / 8.0;
}

static double ref_cosine(double r)
{
  r = fabs(r);
  return (r >= 2.0) ? 0.0 : 0.25 * (1.0 + cos(0.5 * MFTL_PI * r));
}

/** @brief Gaussian elimination with partial pivoting; 0 on success. */
static int ref_solve4(double A[4][4], double b[4], double x[4])
{
  int piv[4] = {0, 1, 2, 3};
  double M[4][5];
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j) M[i][j] = A[i][j];
    M[i][4] = b[i];
  }
  for (int c = 0; c < 4; ++c)
  {
    int best = c;
    for (int r = c + 1; r < 4; ++r)
      if (fabs(M[r][c]) > fabs(M[best][c])) best = r;
    if (M[best][c] == 0.0) return 1;
    if (best != c)
      for (int j = 0; j < 5; ++j)
      {
        const double t = M[c][j];
        M[c][j] = M[best][j];
        M[best][j] = t;
      }
    (void) piv;
    for (int r = c + 1; r < 4; ++r)
    {
      const double f = M[r][c] / M[c][c];
      for (int j = c; j < 5; ++j) M[r][j] -= f * M[c][j];
    }
  }
  for (int i = 3; i >= 0; --i)
  {
    double s = M[i][4];
    for (int j = i + 1; j < 4; ++j) s -= M[i][j] * x[j];
    x[i] = s / M[i][i];
  }
  return 0;
}

/** @brief The global lattice as the test sees it. */
typedef struct
{
  double origin[3], h[3], L[3], gmin[3];
  int64_t N[3];
  bool per[3];
  int kernel;  ///< 0 Peskin, 1 cosine
  double scale;  ///< support_scale; 0 means 1
} ref_lattice;

static int64_t ref_mod(int64_t a, int64_t b) { return ((a % b) + b) % b; }
static int64_t ref_fdiv(int64_t a, int64_t b) { return (a - ref_mod(a, b)) / b; }

/** @brief Position of unwrapped index i on axis k, the canonical coordinate plus a period. */
static double ref_pos(const ref_lattice *R, int k, int64_t i)
{
  if (!R->per[k]) return R->origin[k] + (double) i * R->h[k];
  const int64_t j = ref_mod(i, R->N[k]), s = ref_fdiv(i, R->N[k]);
  return (R->origin[k] + (double) j * R->h[k]) + (double) s * R->L[k];
}

/**
 * @brief Brute-force LINEAR row at x over the whole lattice: keys and weights.
 * @return Number of entries, or -1 if the moment solve failed.
 */
static int ref_row(const ref_lattice *R, const double x[3], uint64_t *keys, double *w, double *dd, int cap)
{
  const double hw = 2.0;
  const double sc = (R->scale > 0.0) ? R->scale : 1.0;
  int n = 0;
  int64_t a[3], b[3];
  for (int k = 0; k < 3; ++k)
  {
    a[k] = (int64_t) floor((x[k] - hw * sc * R->h[k] - R->origin[k]) / R->h[k]) - 2;
    b[k] = (int64_t) ceil((x[k] + hw * sc * R->h[k] - R->origin[k]) / R->h[k]) + 2;
  }
  for (int64_t i = a[0]; i <= b[0]; ++i)
    for (int64_t j = a[1]; j <= b[1]; ++j)
      for (int64_t l = a[2]; l <= b[2]; ++l)
      {
        const int64_t idx[3] = {i, j, l};
        bool ok = true;
        double d[3], psi = 1.0;
        int64_t wrapped[3];
        for (int k = 0; k < 3 && ok; ++k)
        {
          if (!R->per[k] && (idx[k] < 0 || idx[k] >= R->N[k])) ok = false;
          wrapped[k] = R->per[k] ? ref_mod(idx[k], R->N[k]) : idx[k];
          d[k] = (ref_pos(R, k, idx[k]) - x[k]) / (sc * R->h[k]);
          if (fabs(d[k]) >= hw) ok = false;
        }
        if (!ok) continue;
        for (int k = 0; k < 3; ++k) psi *= (R->kernel == 0) ? ref_peskin(d[k]) : ref_cosine(d[k]);
        psi /= sc * sc * sc; /* q / det(H) with q = h^3, H = sc h */
        if (psi == 0.0) continue;
        if (n >= cap) return -1;
        keys[n] = (uint64_t) (wrapped[0] + R->N[0] * (wrapped[1] + R->N[1] * wrapped[2]));
        w[n] = psi;
        memcpy(&dd[3 * n], d, sizeof d);
        ++n;
      }
  double m0 = 0.0, m1[3] = {0, 0, 0};
  for (int e = 0; e < n; ++e)
  {
    m0 += w[e];
    for (int k = 0; k < 3; ++k) m1[k] += w[e] * dd[3 * e + k];
  }
  if (fabs(m0 - 1.0) <= 1e-10 && fabs(m1[0]) <= 1e-10 && fabs(m1[1]) <= 1e-10 && fabs(m1[2]) <= 1e-10) return n;
  double M[4][4] = {{0}}, rhs[4] = {1, 0, 0, 0}, c[4];
  for (int e = 0; e < n; ++e)
  {
    const double p[4] = {1.0, dd[3 * e], dd[3 * e + 1], dd[3 * e + 2]};
    for (int r = 0; r < 4; ++r)
      for (int s = 0; s < 4; ++s) M[r][s] += w[e] * p[r] * p[s];
  }
  if (ref_solve4(M, rhs, c)) return -1;
  for (int e = 0; e < n; ++e) w[e] *= c[0] + c[1] * dd[3 * e] + c[2] * dd[3 * e + 1] + c[3] * dd[3 * e + 2];
  return n;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Fixtures
 * ═════════════════════════════════════════════════════════════════════════════ */

static MFTL_Ensemble *make_ens(const double lo[3], const double hi[3], const bool per[3])
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double l[3] = {lo[0], lo[1], lo[2]}, h[3] = {hi[0], hi[1], hi[2]};
  bool p[3] = {per[0], per[1], per[2]};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, l, h, p, 4);
}

/** @brief A sphere generated on rank 0 and scattered by a redistribution. */
static void add_sphere(MFTL_Ensemble *ens, const double c[3], double r, int refine)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(obj, cc, r, refine);
}

static void scatter(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
}

typedef enum
{
  DEC_ALIGNED,
  DEC_SLAB,
  DEC_REVERSED,
  DEC_ROOT
} decomposition;

/** @brief This rank's owned index box of an N^3 lattice under a decomposition. */
static void owned_box(const int64_t N[3], decomposition dec, int64_t lo[3], int64_t hi[3])
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  if (dec == DEC_ROOT)
  {
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = 0;
      hi[k] = (g_rank == 0) ? N[k] : 0;
    }
    return;
  }
  if (dec == DEC_SLAB)
  {
    for (int k = 0; k < 2; ++k)
    {
      lo[k] = 0;
      hi[k] = N[k];
    }
    lo[2] = N[2] * g_rank / g_nproc;
    hi[2] = N[2] * (g_rank + 1) / g_nproc;
    return;
  }
  const int r = (dec == DEC_REVERSED) ? g_nproc - 1 - g_rank : g_rank;
  /* the MFTL grid's cell of rank r: comm_cart has no reordering, C order */
  int co[3];
  co[2] = r % cuts[2];
  co[1] = (r / cuts[2]) % cuts[1];
  co[0] = r / (cuts[1] * cuts[2]);
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = N[k] * co[k] / cuts[k];
    hi[k] = N[k] * (co[k] + 1) / cuts[k];
  }
}

/** @brief One lattice location registered as one block, with its field array. */
typedef struct
{
  int loc;
  int piece;
  ref_lattice ref;
  int64_t lo[3], hi[3];
  int ghost;
  int64_t alo[3], adims[3];
  int64_t n_elem;
  double *a;
} lat_block;

static int add_lattice(MFTL_SolverGrid *g, const char *name, const ref_lattice *R, decomposition dec, int ghost, lat_block *out)
{
  memset(out, 0, sizeof *out);
  out->ref = *R;
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = name;
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = R->N[k];
    ld.lattice.origin[k] = R->origin[k];
    ld.lattice.spacing[k] = R->h[k];
  }
  out->loc = MFTLSolverGridAddLocation(g, &ld);
  owned_box(R->N, dec, out->lo, out->hi);
  out->ghost = ghost;
  out->n_elem = 1;
  for (int k = 0; k < 3; ++k)
  {
    out->alo[k] = out->lo[k] - ghost;
    out->adims[k] = (out->hi[k] - out->lo[k]) + 2 * ghost;
    out->n_elem *= out->adims[k];
  }
  out->a = (double *) calloc((size_t) (out->n_elem > 0 ? out->n_elem : 1), sizeof(double));
  bool empty = false;
  for (int k = 0; k < 3; ++k)
    if (out->hi[k] <= out->lo[k]) empty = true;
  out->piece = -1;
  if (!empty)
  {
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = out->loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = out->lo[k];
      bd.owned_hi[k] = out->hi[k];
      bd.array_lo[k] = out->alo[k];
      bd.array_dims[k] = out->adims[k];
      bd.ghost_layers[k] = ghost;
    }
    out->piece = MFTLSolverGridAddBlock(g, &bd);
  }
  return out->loc;
}

static void free_block(lat_block *b) { free(b->a); }

typedef double (*field_fn)(const double x[3], void *ctx);

/**
 * @brief Fills a block's array: owned cells with f, ghost cells with f at the
 *        wrapped position when @p ghosts_valid, NaN otherwise (and always NaN
 *        beyond a wall), so a read the coupler should not make is visible.
 */
static void fill_block(lat_block *b, field_fn f, void *ctx, bool ghosts_valid)
{
  for (int64_t e = 0; e < b->n_elem; ++e)
  {
    int64_t i[3];
    i[0] = b->alo[0] + e % b->adims[0];
    i[1] = b->alo[1] + (e / b->adims[0]) % b->adims[1];
    i[2] = b->alo[2] + e / (b->adims[0] * b->adims[1]);
    bool owned = true, real = true;
    double x[3];
    for (int k = 0; k < 3; ++k)
    {
      if (i[k] < b->lo[k] || i[k] >= b->hi[k]) owned = false;
      if (!b->ref.per[k] && (i[k] < 0 || i[k] >= b->ref.N[k])) real = false;
      const int64_t j = b->ref.per[k] ? ref_mod(i[k], b->ref.N[k]) : i[k];
      x[k] = b->ref.origin[k] + (double) j * b->ref.h[k];
    }
    b->a[e] = (owned || (ghosts_valid && real)) ? f(x, ctx) : NAN;
  }
}

static MFTL_SolverField block_field(lat_block *b, uint64_t version, bool halo_valid)
{
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = b->loc;
  F.data = (b->piece >= 0) ? &b->a : MFTL_NULL;
  F.stride = 1;
  F.version = version;
  F.halo_valid = halo_valid;
  return F;
}

/** @brief The element of owned lattice index idx in a block's array. */
static int64_t block_elem(const lat_block *b, const int64_t idx[3])
{
  return (idx[0] - b->alo[0]) + b->adims[0] * ((idx[1] - b->alo[1]) + b->adims[1] * (idx[2] - b->alo[2]));
}

/* the fields */
typedef struct
{
  double a, g[3];
} affine;

static double f_affine(const double x[3], void *ctx)
{
  const affine *A = (const affine *) ctx;
  return A->a + A->g[0] * x[0] + A->g[1] * x[1] + A->g[2] * x[2];
}

static double f_periodic(const double x[3], void *ctx)
{
  (void) ctx;
  return sin(2.0 * MFTL_PI * x[0]) * cos(2.0 * MFTL_PI * x[1]) + 0.5 * sin(4.0 * MFTL_PI * x[2]) + 0.25;
}

/** @brief Deterministic per-key pseudo-random value in [-1, 1). */
static double hash01(uint64_t k)
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return (double) (k >> 11) / (double) (1ull << 52) - 1.0;
}

typedef struct
{
  int64_t N[3];
  uint64_t salt;
} rnd_ctx;

static double f_random(const double x[3], void *ctx)
{
  const rnd_ctx *r = (const rnd_ctx *) ctx;
  /* recover the lattice index from the position via the key the position implies */
  (void) r;
  const uint64_t k = (uint64_t) llround(x[0] * 1e6) * 1000003ull ^ (uint64_t) llround(x[1] * 1e6) * 999983ull ^ (uint64_t) llround(x[2] * 1e6) * 1000033ull;
  return hash01(k ^ r->salt);
}

/** @brief A coupler over one ensemble and grid, with a window and a binding of the vertex markers. */
typedef struct
{
  MFTL_SolverGrid *grid;
  MFTL_Coupler *c;
  MFTL_CouplingWindow *win;
  MFTL_CouplingBinding *b;
} coupled;

static int couple(MFTL_Ensemble *ens, MFTL_SolverGrid *grid, const MFTL_CouplerParams *p, double skin, coupled *out)
{
  memset(out, 0, sizeof *out);
  out->grid = grid;
  int st = MFTLCouplerCreate(ens, grid, p, &out->c);
  if (st != MFTL_COUPLING_OK) return st;
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  MFTL_WindowSpec ws = {{skin, skin, skin}};
  MFTL_SupportRequest *req = MFTL_NULL;
  st = MFTLCouplerRequestSupport(out->c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(out->c, req, &out->win);
  MFTLCouplerReleaseRequest(&req);
  if (st != MFTL_COUPLING_OK) return st;
  return MFTLCouplerBind(out->c, &ms, out->win, &out->b);
}

static void uncouple(coupled *x)
{
  MFTLCouplerReleaseBinding(&x->b);
  MFTLCouplerReleaseWindow(&x->win);
  MFTLCouplerDestroy(&x->c);
}

/** @brief A reference entry's key and its position in the row, ordered by both. */
typedef struct
{
  uint64_t key;
  int f;
} ref_slot;

static int cmp_ref_slot(const void *a, const void *b)
{
  const ref_slot *x = (const ref_slot *) a, *y = (const ref_slot *) b;
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  return (x->f > y->f) - (x->f < y->f);
}

/** @brief The LAST reference entry with this key, as a scan of the row finds it; -1 if none. */
static int ref_slot_find(const ref_slot *sorted, int nr, uint64_t key)
{
  int lo = 0, hi = nr; /* first slot with a larger key */
  while (lo < hi)
  {
    const int mid = lo + (hi - lo) / 2;
    if (sorted[mid].key <= key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (lo > 0 && sorted[lo - 1].key == key) ? sorted[lo - 1].f : -1;
}

/**
 * @brief Does every row of the binding equal the brute-force row? Returns the worst weight error.
 *
 * Each binding entry is looked up in the reference row by key. The row is sorted
 * once for that, which matters at a wide support: thousands of entries per row
 * made a scan per entry the most expensive part of the coupling suite in Debug.
 */
static double rows_vs_reference(const coupled *x, int loc_pos, const ref_lattice *R, int *n_bad)
{
  const int64_t nm = MFTLCouplerBindingMarkerCount(x->b);
  const double *pos = MFTLCouplerBindingPositions(x->b);
  enum
  {
    CAP = 8192
  };
  uint64_t *rk = (uint64_t *) malloc(CAP * sizeof(uint64_t));
  double *rw = (double *) malloc(CAP * sizeof(double)), *dd = (double *) malloc(3 * CAP * sizeof(double));
  ref_slot *sorted = (ref_slot *) malloc(CAP * sizeof(ref_slot));
  double worst = 0.0;
  *n_bad = 0;
  for (int64_t i = 0; i < nm; ++i)
  {
    int n = 0;
    const uint64_t *keys;
    const double *w;
    if (MFTLCouplerBindingRow(x->b, loc_pos, i, &n, &keys, &w) != MFTL_SUCCESS)
    {
      ++*n_bad;
      continue;
    }
    const int nr = ref_row(R, &pos[3 * i], rk, rw, dd, CAP);
    if (nr != n)
    {
      ++*n_bad;
      continue;
    }
    for (int f = 0; f < nr; ++f) sorted[f] = (ref_slot){rk[f], f};
    qsort(sorted, (size_t) nr, sizeof(ref_slot), cmp_ref_slot);
    for (int e = 0; e < n; ++e)
    {
      const int found = ref_slot_find(sorted, nr, keys[e]);
      if (found < 0)
      {
        ++*n_bad;
        break;
      }
      worst = fmax(worst, fabs(rw[found] - w[e]));
    }
  }
  free(sorted);
  free(rk);
  free(rw);
  free(dd);
  return worst;
}


/* ═════════════════════════════════════════════════════════════════════════════
 * Point clouds: one jittered sample per cell of a uniform grid
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef enum
{
  OWN_SCATTERED,  ///< owner = hash(key) mod P: every rank owns samples everywhere
  OWN_SLAB,  ///< z-slabs
  OWN_ALIGNED  ///< the MFTL box the sample lies in
} point_ownership;

typedef struct
{
  int loc, piece;
  int64_t nc;
  int64_t n;
  uint64_t *key;
  double *pos, *meas, *h;
  double *a;
} pts_block;

/** @brief Every sample of the global cloud, generated identically on every rank. */
static void point_of_cell(int64_t nc, double jitter, int64_t c, double x[3], uint64_t *key)
{
  const int64_t idx[3] = {c % nc, (c / nc) % nc, c / (nc * nc)};
  const double H = 1.0 / (double) nc;
  for (int k = 0; k < 3; ++k) x[k] = (idx[k] + 0.5) * H + jitter * H * 0.5 * hash01((uint64_t) c * 3u + (uint64_t) k + 17u);
  *key = (uint64_t) c * 7u + 1000u; /* not the cell index: keys are arbitrary ids */
}

static double var_h_of(double H, const double x[3]) { return H * (1.0 + 0.3 * sin(2.0 * MFTL_PI * x[0]) * cos(2.0 * MFTL_PI * x[1])); }

static int add_points(MFTL_SolverGrid *g, const char *name, int64_t nc, double jitter, point_ownership own, bool var_h, pts_block *out)
{
  memset(out, 0, sizeof *out);
  out->nc = nc;
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = name;
  ld.lattice.kind = MFTL_LATTICE_NONE;
  out->loc = MFTLSolverGridAddLocation(g, &ld);
  const int64_t total = nc * nc * nc;
  const double H = 1.0 / (double) nc;
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  out->key = (uint64_t *) malloc((size_t) total * sizeof(uint64_t));
  out->pos = (double *) malloc(3 * (size_t) total * sizeof(double));
  out->meas = (double *) malloc((size_t) total * sizeof(double));
  out->h = (double *) malloc(3 * (size_t) total * sizeof(double));
  out->a = (double *) calloc((size_t) total, sizeof(double));
  for (int64_t c = 0; c < total; ++c)
  {
    double x[3];
    uint64_t key;
    point_of_cell(nc, jitter, c, x, &key);
    int owner = 0;
    if (own == OWN_SCATTERED)
      owner = (int) ((uint64_t) (hash01(key) * 1e9 + 2e9) % (uint64_t) g_nproc);
    else if (own == OWN_SLAB)
      owner = (int) ((c / (nc * nc)) * g_nproc / nc);
    else
    {
      int co[3];
      for (int k = 0; k < 3; ++k)
      {
        co[k] = (int) (x[k] * cuts[k]);
        if (co[k] >= cuts[k]) co[k] = cuts[k] - 1;
      }
      owner = (co[0] * cuts[1] + co[1]) * cuts[2] + co[2];
    }
    if (owner != g_rank) continue;
    const int64_t i = out->n++;
    out->key[i] = key;
    memcpy(&out->pos[3 * i], x, sizeof x);
    out->meas[i] = H * H * H;
    for (int k = 0; k < 3; ++k) out->h[3 * i + k] = var_h ? var_h_of(H, x) : H;
  }
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = out->loc;
  pd.n = out->n;
  pd.keys = out->key;
  pd.pos = out->pos;
  pd.measure = out->meas;
  pd.h = var_h ? out->h : MFTL_NULL;
  for (int k = 0; k < 3; ++k) pd.h_uniform[k] = H;
  out->piece = MFTLSolverGridAddPoints(g, &pd);
  return out->loc;
}

static void free_points(pts_block *b)
{
  free(b->key);
  free(b->pos);
  free(b->meas);
  free(b->h);
  free(b->a);
}

static MFTL_SolverField points_field(pts_block *b, uint64_t version)
{
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = b->loc;
  F.data = &b->a;
  F.stride = 1;
  F.version = version;
  return F;
}

/** @brief Every rank's local array gathered to every rank, @p stride doubles per entry. */
static double *allgather_doubles(const double *mine, int n_mine, int stride, int *n_total)
{
  int *cnt = (int *) malloc((size_t) g_nproc * sizeof(int)), *dsp = (int *) malloc((size_t) g_nproc * sizeof(int));
  const int m = n_mine * stride;
  MPI_Allgather(&m, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int r = 0; r < g_nproc; ++r)
  {
    dsp[r] = tot;
    tot += cnt[r];
  }
  double *all = (double *) malloc((size_t) (tot > 0 ? tot : 1) * sizeof(double));
  MPI_Allgatherv(mine, m, MPI_DOUBLE, all, cnt, dsp, MPI_DOUBLE, MPI_COMM_WORLD);
  free(cnt);
  free(dsp);
  *n_total = tot / stride;
  return all;
}

/** @brief Releases everything a test built, in dependency order. */
static void teardown(coupled *x, MFTL_SolverGrid **g, MFTL_Ensemble **ens)
{
  uncouple(x);
  MFTLSolverGridDestroy(g);
  MFTLEnsembleDestroy(ens);
}

#endif  // MFTL_TEST_COUPLING_FIXTURES_H
