/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Irregular solver decompositions for the coupling tests, and the
 *        brute-force reference over a global lattice.
 *
 * A random k-d partition of the lattice into several blocks per rank (some
 * ranks own none), each with its own ghost width PER AXIS and padding beyond
 * the ghosts that holds NaN, assigned to ranks by a hash, so a block's
 * neighbours are arbitrary ranks; lattices with a different count, spacing and
 * periodicity per axis on a domain that need not start at the origin; markers
 * where arithmetic is tight. The reference shares no discovery, coverage,
 * routing, band, schedule or candidate code with the coupler: every index whose
 * image lies in a marker's support, enumerated by the test, through the row
 * core MFTLCouplingAssembleRow() (tested on its own in test_coupling_kernels).
 */

#ifndef MFTL_TEST_COUPLING_IRREGULAR_H
#define MFTL_TEST_COUPLING_IRREGULAR_H

#include "test_coupling_fixtures.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * A lattice, a domain and a deterministic random number stream
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief splitmix64: a deterministic stream identical on every rank. */
static uint64_t adv_next(uint64_t *s)
{
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/** @brief Uniform in [0, 1). */
static double adv_unif(uint64_t *s) { return (double) (adv_next(s) >> 11) / (double) (1ull << 53); }

/** @brief The global lattice of one location as the test sees it. */
typedef struct
{
  int64_t N[3];
  double origin[3], h[3];
  double gmin[3], L[3];
  bool per[3];
  const double *coords[3];  ///< rectilinear: sample coordinates per axis (NULL: uniform)
  const double *faces[3];  ///< rectilinear: cell bounds per axis
} adv_lat;

/** @brief Canonical coordinate of index j, 0 <= j < N. */
static double adv_coord(const adv_lat *A, int k, int64_t j) { return A->coords[k] ? A->coords[k][j] : A->origin[k] + (double) j * A->h[k]; }

/** @brief Position of unwrapped index i on axis k: the canonical coordinate of i mod N plus whole periods. */
static double adv_pos(const adv_lat *A, int k, int64_t i)
{
  if (!A->per[k]) return adv_coord(A, k, i);
  const int64_t j = ref_mod(i, A->N[k]), s = ref_fdiv(i, A->N[k]);
  return adv_coord(A, k, j) + (double) s * A->L[k];
}

/** @brief The location descriptor of a lattice. */
static int adv_add_location(MFTL_SolverGrid *g, const char *name, const adv_lat *A, MFTL_BoundaryPolicy bnd)
{
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = name;
  ld.boundary = bnd;
  ld.lattice.kind = A->coords[0] ? MFTL_LATTICE_RECTILINEAR : MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = A->N[k];
    ld.lattice.origin[k] = A->origin[k];
    ld.lattice.spacing[k] = A->h[k];
    ld.lattice.coords[k] = A->coords[k];
    ld.lattice.faces[k] = A->faces[k];
  }
  return MFTLSolverGridAddLocation(g, &ld);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Irregular decompositions: random k-d boxes, several per rank
 * ═════════════════════════════════════════════════════════════════════════════ */

enum
{
  ADV_MAX_BOXES = 256
};

/** @brief This rank's blocks of one location, with their field arrays. */
typedef struct
{
  int n;
  int64_t lo[ADV_MAX_BOXES][3], hi[ADV_MAX_BOXES][3];
  int64_t alo[ADV_MAX_BOXES][3], adims[ADV_MAX_BOXES][3];
  int ghost[ADV_MAX_BOXES][3];
  int64_t n_elem[ADV_MAX_BOXES];
  int piece[ADV_MAX_BOXES];
  int n_all;  ///< every box of the GLOBAL partition, on every rank (to find a sample's owner)
  int64_t all_box[ADV_MAX_BOXES][6];
  int all_owner[ADV_MAX_BOXES];
  int ncomp;  ///< interleaved components per element (the stride)
  double *a[ADV_MAX_BOXES];  ///< [n_elem * ncomp]
  double *ptr[ADV_MAX_BOXES];  ///< base pointer per piece, for one component
  long n_owned_total;  ///< owned samples of the location over all of this rank's blocks
} adv_blocks;

/**
 * @brief Splits [0, N) into @p n_boxes boxes by recursive bisection at random
 *        positions (the largest box first, along its longest axis), assigns
 *        each to a rank by a hash, and registers this rank's. Some ranks get
 *        several, some none; every rank with @p empty_block also registers a
 *        block of zero volume. Ghost widths are random per block and axis in
 *        [0, max_ghost], and every array is padded by up to two more cells.
 */
/** @brief An optional solid: true where a lattice index has NO sample. */
typedef bool (*adv_solid_fn)(const int64_t idx[3]);

static void adv_decompose_on(int rank, int nproc, MFTL_SolverGrid *g, int loc, const adv_lat *A, int n_boxes, uint64_t seed, int max_ghost, int ncomp,
                             bool empty_block, adv_solid_fn solid, adv_blocks *B)
{
  memset(B, 0, sizeof *B);
  B->ncomp = ncomp;
  int64_t box[ADV_MAX_BOXES][6];
  int nb = 1;
  for (int k = 0; k < 3; ++k)
  {
    box[0][k] = 0;
    box[0][3 + k] = A->N[k];
  }
  uint64_t s = seed;
  while (nb < n_boxes && nb < ADV_MAX_BOXES)
  {
    int best = -1;
    int64_t bv = 0;
    for (int b = 0; b < nb; ++b)
    {
      const int64_t v = (box[b][3] - box[b][0]) * (box[b][4] - box[b][1]) * (box[b][5] - box[b][2]);
      if (v > bv)
      {
        bv = v;
        best = b;
      }
    }
    int ax = 0;
    for (int k = 1; k < 3; ++k)
      if (box[best][3 + k] - box[best][k] > box[best][3 + ax] - box[best][ax]) ax = k;
    const int64_t len = box[best][3 + ax] - box[best][ax];
    if (len < 2) break;
    const int64_t cut = box[best][ax] + 1 + (int64_t) (adv_next(&s) % (uint64_t) (len - 1));
    memcpy(box[nb], box[best], sizeof box[0]);
    box[best][3 + ax] = cut;
    box[nb][ax] = cut;
    ++nb;
  }
  B->n_all = nb;
  for (int b = 0; b < nb; ++b)
  {
    const int owner = (int) (adv_next(&s) % (uint64_t) nproc);
    memcpy(B->all_box[b], box[b], sizeof box[b]);
    B->all_owner[b] = owner;
    int gh[3], pad[3];
    for (int k = 0; k < 3; ++k)
    {
      gh[k] = (max_ghost > 0) ? (int) (adv_next(&s) % (uint64_t) (max_ghost + 1)) : 0;
      pad[k] = (int) (adv_next(&s) % 3u);
    }
    if (owner != rank) continue;
    const int i = B->n++;
    B->n_elem[i] = 1;
    for (int k = 0; k < 3; ++k)
    {
      B->lo[i][k] = box[b][k];
      B->hi[i][k] = box[b][3 + k];
      B->ghost[i][k] = gh[k];
      B->alo[i][k] = B->lo[i][k] - gh[k] - pad[k];
      B->adims[i][k] = (B->hi[i][k] - B->lo[i][k]) + 2 * (gh[k] + pad[k]);
      B->n_elem[i] *= B->adims[i][k];
    }
    B->n_owned_total += (long) ((B->hi[i][0] - B->lo[i][0]) * (B->hi[i][1] - B->lo[i][1]) * (B->hi[i][2] - B->lo[i][2]));
  }
  if (empty_block && B->n < ADV_MAX_BOXES)
  {
    /* a legal block owning nothing: lo == hi on one axis */
    const int i = B->n++;
    B->n_elem[i] = 1;
    for (int k = 0; k < 3; ++k)
    {
      B->lo[i][k] = (k == 0) ? A->N[0] / 2 : 0;
      B->hi[i][k] = (k == 0) ? A->N[0] / 2 : A->N[k];
      B->ghost[i][k] = 0;
      B->alo[i][k] = B->lo[i][k];
      B->adims[i][k] = (k == 0) ? 1 : A->N[k];
      B->n_elem[i] *= B->adims[i][k];
    }
  }
  for (int i = 0; i < B->n; ++i)
  {
    B->a[i] = (double *) malloc((size_t) B->n_elem[i] * (size_t) ncomp * sizeof(double));
    for (int64_t e = 0; e < B->n_elem[i] * ncomp; ++e) B->a[i][e] = NAN;
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = B->lo[i][k];
      bd.owned_hi[k] = B->hi[i][k];
      bd.array_lo[k] = B->alo[i][k];
      bd.array_dims[k] = B->adims[i][k];
      bd.ghost_layers[k] = B->ghost[i][k];
    }
    bd.stride[0] = 1;
    bd.stride[1] = B->adims[i][0];
    bd.stride[2] = B->adims[i][0] * B->adims[i][1];
    unsigned char *mask = MFTL_NULL;
    const int64_t vol = (B->hi[i][0] - B->lo[i][0]) * (B->hi[i][1] - B->lo[i][1]) * (B->hi[i][2] - B->lo[i][2]);
    if (solid && vol > 0)
    {
      mask = (unsigned char *) malloc((size_t) vol);
      int64_t m = 0;
      for (int64_t k3 = B->lo[i][2]; k3 < B->hi[i][2]; ++k3)
        for (int64_t j3 = B->lo[i][1]; j3 < B->hi[i][1]; ++j3)
          for (int64_t i3 = B->lo[i][0]; i3 < B->hi[i][0]; ++i3)
          {
            const int64_t idx[3] = {i3, j3, k3};
            mask[m++] = solid(idx) ? 0 : 1;
          }
      bd.active = mask;
    }
    B->piece[i] = MFTLSolverGridAddBlock(g, &bd);
    free(mask);
  }
}

static void adv_decompose(MFTL_SolverGrid *g, int loc, const adv_lat *A, int n_boxes, uint64_t seed, int max_ghost, int ncomp, bool empty_block, adv_blocks *B)
{
  adv_decompose_on(g_rank, g_nproc, g, loc, A, n_boxes, seed, max_ghost, ncomp, empty_block, MFTL_NULL, B);
}

static void adv_free_blocks(adv_blocks *B)
{
  for (int i = 0; i < B->n; ++i) free(B->a[i]);
  memset(B, 0, sizeof *B);
}

/** @brief Lattice index of element e of block i. */
static void adv_elem_index(const adv_blocks *B, int i, int64_t e, int64_t idx[3])
{
  idx[0] = B->alo[i][0] + e % B->adims[i][0];
  idx[1] = B->alo[i][1] + (e / B->adims[i][0]) % B->adims[i][1];
  idx[2] = B->alo[i][2] + e / (B->adims[i][0] * B->adims[i][1]);
}

/**
 * @brief Fills component @p comp of every block: owned samples with f; ghost
 *        samples with f at the wrapped position when @p ghosts_valid and the
 *        index exists; NaN everywhere else (padding, beyond walls, invalid ghosts).
 */
static void adv_fill(adv_blocks *B, const adv_lat *A, int comp, field_fn f, void *ctx, bool ghosts_valid)
{
  for (int i = 0; i < B->n; ++i)
    for (int64_t e = 0; e < B->n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(B, i, e, idx);
      bool owned = true, ghost = true, real = true;
      double x[3];
      for (int k = 0; k < 3; ++k)
      {
        if (idx[k] < B->lo[i][k] || idx[k] >= B->hi[i][k]) owned = false;
        if (idx[k] < B->lo[i][k] - B->ghost[i][k] || idx[k] >= B->hi[i][k] + B->ghost[i][k]) ghost = false;
        if (!A->per[k] && (idx[k] < 0 || idx[k] >= A->N[k])) real = false;
        const int64_t j = A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k];
        x[k] = real ? adv_coord(A, k, j) : 0.0;
      }
      B->a[i][e * B->ncomp + comp] = (owned || (ghosts_valid && ghost && real)) ? f(x, ctx) : NAN;
    }
}

/** @brief A field over component @p comp of the blocks' arrays. */
static MFTL_SolverField adv_field(adv_blocks *B, int loc, int comp, uint64_t version, bool halo_valid)
{
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  for (int i = 0; i < B->n; ++i) B->ptr[i] = B->a[i] + comp;
  F.location = loc;
  F.data = (B->n > 0) ? B->ptr : MFTL_NULL;
  F.stride = B->ncomp;
  F.version = version;
  F.halo_valid = halo_valid;
  return F;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The brute-force reference
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The reference row of a marker at @p x over the GLOBAL lattice: every
 *        unwrapped index whose image lies in the closed support, through the
 *        row core. Returns the entry count, or -1 if the row core refused.
 */
static int adv_ref_row(const adv_lat *A, const MFTL_CouplerParams *P, const double x[3], uint64_t *keys, double *w, int cap)
{
  MFTL_KernelInfo ki;
  if (P->kernel == MFTL_KERNEL_USER)
    ki.half_width = P->user_kernel.half_width;
  else
    MFTLCouplingKernelInfo(P->kernel, &ki);
  const double gmax = (P->failure & MFTL_STENCIL_GROW) ? P->max_growth : 1.0;
  int64_t a[3], b[3];
  double R[3];
  for (int k = 0; k < 3; ++k)
  {
    R[k] = ki.half_width * P->support_scale * gmax * A->h[k];
    a[k] = (int64_t) floor((x[k] - R[k] - A->origin[k]) / A->h[k]) - 1;
    b[k] = (int64_t) ceil((x[k] + R[k] - A->origin[k]) / A->h[k]) + 1;
    if (!A->per[k])
    {
      a[k] = (a[k] < 0) ? 0 : a[k];
      b[k] = (b[k] > A->N[k] - 1) ? A->N[k] - 1 : b[k];
    }
  }
  int n = 0;
  double *X = (double *) malloc(3 * (size_t) cap * sizeof(double)), *q = (double *) malloc((size_t) cap * sizeof(double));
  for (int64_t i = a[0]; i <= b[0]; ++i)
    for (int64_t j = a[1]; j <= b[1]; ++j)
      for (int64_t l = a[2]; l <= b[2]; ++l)
      {
        const int64_t idx[3] = {i, j, l};
        double p[3];
        bool in = true;
        for (int k = 0; k < 3; ++k)
        {
          p[k] = adv_pos(A, k, idx[k]);
          if (fabs(p[k] - x[k]) > R[k]) in = false;
        }
        if (!in) continue;
        if (n >= cap)
        {
          free(X);
          free(q);
          return -1;
        }
        int64_t wr[3];
        for (int k = 0; k < 3; ++k) wr[k] = A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k];
        keys[n] = (uint64_t) (wr[0] + A->N[0] * (wr[1] + A->N[1] * wr[2]));
        memcpy(&X[3 * n], p, sizeof p);
        q[n] = A->h[0] * A->h[1] * A->h[2];
        ++n;
      }
  const int st = MFTLCouplingAssembleRow(P, x, A->h, n, X, q, w, MFTL_NULL);
  free(X);
  free(q);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

/**
 * @brief Compares one bound row with the reference as SPARSE VECTORS: every key
 *        of either must carry the same weight in both (absent = 0). A weight
 *        that is zero to round-off may be present on one side only -- a sample
 *        exactly on the support boundary -- but a sample of real weight missing
 *        from the bound row is a mismatch. Returns the worst difference.
 */
static double adv_row_diff(int n1, const uint64_t *k1, const double *w1, int n2, const uint64_t *k2, const double *w2)
{
  double worst = 0.0;
  for (int e = 0; e < n1; ++e)
  {
    double o = 0.0;
    for (int f = 0; f < n2; ++f)
      if (k2[f] == k1[e]) o += w2[f];
    worst = fmax(worst, fabs(w1[e] - o));
  }
  for (int f = 0; f < n2; ++f)
  {
    bool found = false;
    for (int e = 0; e < n1; ++e)
      if (k1[e] == k2[f]) found = true;
    if (!found) worst = fmax(worst, fabs(w2[f]));
  }
  /* a key twice in one row would be a double-counted sample */
  for (int e = 0; e < n1; ++e)
    for (int f = e + 1; f < n1; ++f)
      if (k1[e] == k1[f]) worst = fmax(worst, 1.0);
  return worst;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Markers
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The global marker list, identical on every rank; this rank keeps i % P == rank. */
typedef struct
{
  int64_t n_all, n;
  double *all;  ///< [3 n_all]
  double *pos;  ///< [3 n] this rank's
  uint64_t *keys;  ///< [n]
} adv_markers;

static void adv_markers_take_on(int rank, int nproc, adv_markers *M)
{
  M->pos = (double *) malloc(3 * (size_t) (M->n_all > 0 ? M->n_all : 1) * sizeof(double));
  M->keys = (uint64_t *) malloc((size_t) (M->n_all > 0 ? M->n_all : 1) * sizeof(uint64_t));
  M->n = 0;
  for (int64_t i = 0; i < M->n_all; ++i)
  {
    if (i % nproc != rank) continue;
    memcpy(&M->pos[3 * M->n], &M->all[3 * i], 3 * sizeof(double));
    M->keys[M->n] = (uint64_t) i;
    ++M->n;
  }
}

static void adv_markers_take(adv_markers *M) { adv_markers_take_on(g_rank, g_nproc, M); }

static void adv_markers_free(adv_markers *M)
{
  free(M->all);
  free(M->pos);
  free(M->keys);
  memset(M, 0, sizeof *M);
}

/**
 * @brief Markers that make the arithmetic tight: random ones, plus on samples,
 *        on bin boundaries (bins are half the reach), on rank boundaries, on the
 *        seam of a periodic axis (min, one ulp below min -- which wraps onto
 *        max -- max itself, and several periods away), and on a wall. Positions
 *        on a walled axis stay at least @p wall_gap inside.
 */
static void adv_markers_make_cuts(const int cuts[3], const adv_lat *A, double reach, int n_random, uint64_t seed, double wall_gap, adv_markers *M)
{
  int cap = n_random + 64;
  M->all = (double *) malloc(3 * (size_t) cap * sizeof(double));
  M->n_all = 0;
  uint64_t s = seed;
  /* the axis value a special marker uses on axes it is not special on */
  double mid[3];
  for (int k = 0; k < 3; ++k) mid[k] = A->gmin[k] + (0.37 + 0.1 * k) * A->L[k];
#define ADV_PUSH(x0, x1, x2)                       \
  do                                               \
  {                                                \
    if (M->n_all < cap)                            \
    {                                              \
      M->all[3 * M->n_all + 0] = (x0);             \
      M->all[3 * M->n_all + 1] = (x1);             \
      M->all[3 * M->n_all + 2] = (x2);             \
      ++M->n_all;                                  \
    }                                              \
  } while (0)
  for (int i = 0; i < n_random; ++i)
  {
    double x[3];
    for (int k = 0; k < 3; ++k)
    {
      const double lo = A->per[k] ? A->gmin[k] : A->gmin[k] + wall_gap;
      const double hi = A->per[k] ? A->gmin[k] + A->L[k] : A->gmin[k] + A->L[k] - wall_gap;
      x[k] = lo + adv_unif(&s) * (hi - lo);
    }
    ADV_PUSH(x[0], x[1], x[2]);
  }
  for (int k = 0; k < 3; ++k)
  {
    double x[3] = {mid[0], mid[1], mid[2]};
    /* on a sample */
    x[k] = adv_pos(A, k, A->N[k] / 2);
    ADV_PUSH(x[0], x[1], x[2]);
    /* on a bin boundary, and an ulp either side */
    const double bin = 0.5 * reach;
    const double bb = A->gmin[k] + floor(0.5 * A->L[k] / bin) * bin;
    x[k] = bb;
    ADV_PUSH(x[0], x[1], x[2]);
    x[k] = nextafter(bb, -1e300);
    ADV_PUSH(x[0], x[1], x[2]);
    x[k] = nextafter(bb, 1e300);
    ADV_PUSH(x[0], x[1], x[2]);
    /* on every interior rank boundary of this axis */
    for (int c = 1; c < cuts[k]; ++c)
    {
      x[k] = A->gmin[k] + (double) c * A->L[k] / (double) cuts[k];
      ADV_PUSH(x[0], x[1], x[2]);
    }
    if (A->per[k])
    {
      x[k] = A->gmin[k];
      ADV_PUSH(x[0], x[1], x[2]);
      x[k] = nextafter(A->gmin[k], -1e300);
      ADV_PUSH(x[0], x[1], x[2]);
      x[k] = A->gmin[k] + A->L[k];
      ADV_PUSH(x[0], x[1], x[2]);
      x[k] = A->gmin[k] + 0.31 * A->L[k] - 3.0 * A->L[k];
      ADV_PUSH(x[0], x[1], x[2]);
      x[k] = A->gmin[k] + 0.62 * A->L[k] + 2.0 * A->L[k];
      ADV_PUSH(x[0], x[1], x[2]);
    }
    else if (wall_gap == 0.0)
    {
      x[k] = A->gmin[k];
      ADV_PUSH(x[0], x[1], x[2]);
      x[k] = A->gmin[k] + A->L[k];
      ADV_PUSH(x[0], x[1], x[2]);
    }
  }
#undef ADV_PUSH
  adv_markers_take(M);
}

/** @brief As adv_markers_make_cuts(), on the domain's own rank boundaries. */
static void adv_markers_make(const MFTL_Domain *d, const adv_lat *A, double reach, int n_random, uint64_t seed, double wall_gap, adv_markers *M)
{
  adv_markers_make_cuts(d->cuts, A, reach, n_random, seed, wall_gap, M);
}

static double adv_affine_fn(const double x[3], void *ctx) { return f_affine(x, ctx); }


/** @brief The global domain of a scenario: the first lattice's box. */
static MFTL_Ensemble *adv_ensemble(const adv_lat *A)
{
  double lo[3], hi[3];
  bool per[3];
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = A->gmin[k];
    hi[k] = A->gmin[k] + A->L[k];
    per[k] = A->per[k];
  }
  return make_ens(lo, hi, per);
}

/** @brief The default anisotropic, offset, partly periodic lattice. */
static adv_lat adv_default_lattice(bool cell_x, const bool per[3])
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  const int64_t n[3] = {13, 9, 11};
  const double L[3] = {1.3, 0.75, 1.1}, gmin[3] = {-0.65, 2.0, -7.25};
  for (int k = 0; k < 3; ++k)
  {
    A.per[k] = per[k];
    A.gmin[k] = gmin[k];
    A.L[k] = L[k];
    const bool cell = (k == 0) ? cell_x : true;
    if (cell || per[k])
    {
      /* cell-centred, or node-centred on a periodic axis: n samples per period */
      A.N[k] = n[k];
      A.h[k] = L[k] / (double) n[k];
      A.origin[k] = gmin[k] + (cell ? 0.5 * A.h[k] : 0.0);
    }
    else
    {
      /* node-centred on a walled axis: samples on both walls */
      A.N[k] = n[k] + 1;
      A.h[k] = L[k] / (double) n[k];
      A.origin[k] = gmin[k];
    }
  }
  return A;
}

/** @brief A walled unit box on @p comm. */
static MFTL_Ensemble *adv_unit_box(MPI_Comm comm, bool periodic)
{
  int np;
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {periodic, periodic, periodic};
  return MFTLEnsembleCreate(comm, cuts, lo, hi, per, 4);
}

/** @brief Requests, windows and binds a GLOBAL list of USER markers (marker i on rank i % P) at locations @p locs. */
static int adv_bind_list(MFTL_Coupler *c, int n_all, const double *all, int n_locs, const int *locs, MFTL_CouplingWindow **w, MFTL_CouplingBinding **b,
                         int64_t *n_mine)
{
  double *pos = (double *) malloc(3 * (size_t) (n_all > 0 ? n_all : 1) * sizeof(double));
  uint64_t *keys = (uint64_t *) malloc((size_t) (n_all > 0 ? n_all : 1) * sizeof(uint64_t));
  int n = 0;
  for (int i = 0; i < n_all; ++i)
    if (i % g_nproc == g_rank)
    {
      memcpy(&pos[3 * n], &all[3 * i], 3 * sizeof(double));
      keys[n++] = (uint64_t) i;
    }
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = n;
  ms.user_pos = pos;
  ms.user_keys = keys;
  ms.n_locations = n_locs;
  ms.locations = locs;
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  *w = MFTL_NULL;
  *b = MFTL_NULL;
  int st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, w);
  MFTLCouplerReleaseRequest(&req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(c, &ms, *w, b);
  if (n_mine) *n_mine = n;
  free(pos);
  free(keys);
  return st;
}

#endif  // MFTL_TEST_COUPLING_IRREGULAR_H
