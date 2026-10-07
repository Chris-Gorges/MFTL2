/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Rows the solver builds itself (plan 06, C3): MFTLCouplerRequestSamples()
 *        and MFTLCouplerImportRows().
 *
 * The strongest check needs no tolerance argument: trilinear (Q1) rows a
 * solver evaluates on its own elements ARE the HAT2 kernel's rows on a node
 * lattice, so the imported binding -- over a window of named samples, routed
 * by key with owners the solver names -- must interpolate and spread what the
 * coupler's own kernel binding does over a spatial window, to round-off. The
 * two share the transfer code and nothing upstream of it: no discovery, no
 * coverage proof, no candidate gathering, no row assembly.
 *
 * Beyond that: triquadratic Lagrange rows reproduce quadratics (no kernel row
 * does); lowest-order Raviart-Thomas-like rows on a directional face set
 * reproduce constant vectors and refuse to be declared LINEAR; a column far
 * from every marker is fetched and its load returned exactly once; a modal row
 * passes as NONE and fails as CONSTANT; and every protocol failure -- a wrong
 * owner, a masked sample, an unexported one (by key, then exported and
 * found), a column outside the window, a NaN, a broken row pointer, a kernel
 * bind on a keyed window, a stage of imported rows -- is agreed on every rank.
 */

#include "test_coupling_irregular.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * The solver side: elements over a node lattice
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The rank owning lattice index idx in the decomposition every rank knows. */
static int lat_owner(const adv_blocks *B, const int64_t idx[3])
{
  for (int b = 0; b < B->n_all; ++b)
  {
    const int64_t *x = B->all_box[b];
    if (idx[0] >= x[0] && idx[0] < x[3] && idx[1] >= x[1] && idx[1] < x[4] && idx[2] >= x[2] && idx[2] < x[5]) return B->all_owner[b];
  }
  return -1;
}

static uint64_t lat_key(const adv_lat *A, const int64_t idx[3]) { return (uint64_t) (idx[0] + A->N[0] * (idx[1] + A->N[1] * idx[2])); }

/** @brief Rows built by the test, in the solver's own terms. */
typedef struct
{
  int64_t n;  ///< markers
  int64_t *row_ptr;
  uint64_t *keys;
  int *owner;
  double *w;
  int64_t nnz, cap;
} native;

static void native_init(native *R, int64_t n)
{
  memset(R, 0, sizeof *R);
  R->n = n;
  R->row_ptr = (int64_t *) calloc((size_t) (n + 1), sizeof(int64_t));
}

static void native_push(native *R, uint64_t key, int owner, double w)
{
  if (R->nnz == R->cap)
  {
    R->cap = R->cap ? 2 * R->cap : 256;
    R->keys = (uint64_t *) realloc(R->keys, (size_t) R->cap * sizeof(uint64_t));
    R->owner = (int *) realloc(R->owner, (size_t) R->cap * sizeof(int));
    R->w = (double *) realloc(R->w, (size_t) R->cap * sizeof(double));
  }
  R->keys[R->nnz] = key;
  R->owner[R->nnz] = owner;
  R->w[R->nnz] = w;
  R->nnz++;
}

static void native_free(native *R)
{
  free(R->row_ptr);
  free(R->keys);
  free(R->owner);
  free(R->w);
}

/**
 * @brief Per-axis Lagrange nodes and weights of order p (1 trilinear, 2
 *        triquadratic) at x: the element's first index i0 (unwrapped) and the
 *        p+1 weights.
 */
static void lagrange_axis(const adv_lat *A, int k, int p, double x, int64_t *i0, double *w)
{
  const double xi = (x - A->origin[k]) / A->h[k];
  if (p == 1)
  {
    int64_t i = (int64_t) floor(xi);
    if (!A->per[k]) i = (i < 0) ? 0 : ((i > A->N[k] - 2) ? A->N[k] - 2 : i);
    const double t = xi - (double) i;
    *i0 = i;
    w[0] = 1.0 - t;
    w[1] = t;
    return;
  }
  int64_t c = (int64_t) floor(xi + 0.5);
  if (!A->per[k]) c = (c < 1) ? 1 : ((c > A->N[k] - 2) ? A->N[k] - 2 : c);
  const double s = xi - (double) c;
  *i0 = c - 1;
  w[0] = 0.5 * s * (s - 1.0);
  w[1] = 1.0 - s * s;
  w[2] = 0.5 * s * (s + 1.0);
}

/** @brief The tensor Lagrange row of order p at x, scaled by @p scale, appended to R. */
static void lagrange_row(const adv_lat *A, const adv_blocks *B, int p, const double x[3], double scale, native *R)
{
  int64_t i0[3];
  double w[3][3];
  for (int k = 0; k < 3; ++k) lagrange_axis(A, k, p, x[k], &i0[k], w[k]);
  for (int c = 0; c <= p; ++c)
    for (int b = 0; b <= p; ++b)
      for (int a = 0; a <= p; ++a)
      {
        int64_t idx[3] = {i0[0] + a, i0[1] + b, i0[2] + c};
        for (int k = 0; k < 3; ++k)
          if (A->per[k]) idx[k] = ref_mod(idx[k], A->N[k]);
        native_push(R, lat_key(A, idx), lat_owner(B, idx), scale * w[0][a] * w[1][b] * w[2][c]);
      }
}

/** @brief The rows of this rank's markers, closed. */
static void native_close(native *R, int64_t i) { R->row_ptr[i + 1] = R->nnz; }

/** @brief RequestSamples over the union of R's columns at one location, then the window. */
static int keyed_window(MFTL_Coupler *c, int loc, const native *R, MFTL_CouplingWindow **w, MFTL_SupportRequest **req_out)
{
  MFTL_SampleList L = {loc, R->nnz, R->keys, R->owner};
  MFTL_SupportRequest *req = MFTL_NULL;
  *w = MFTL_NULL;
  int st = MFTLCouplerRequestSamples(c, 1, &L, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, w);
  if (req_out)
    *req_out = req;
  else
    MFTLCouplerReleaseRequest(&req);
  return st;
}

static int import_one(MFTL_Coupler *c, const MFTL_MarkerSpec *ms, MFTL_CouplingWindow *w, int loc, const native *R, MFTL_Reproduction decl,
                      MFTL_CouplingBinding **b)
{
  MFTL_NativeRows nr = {loc, R->row_ptr, R->keys, R->w, decl};
  return MFTLCouplerImportRows(c, ms, w, 1, &nr, b);
}

/** @brief The scenario: a node lattice periodic in x, cell-centred and walled in y and z, decomposed into random blocks. */
typedef struct
{
  adv_lat A;
  MFTL_Ensemble *ens;
  MFTL_SolverGrid *g;
  int loc;
  adv_blocks B;
  int n_mine;
  double *pos;
  uint64_t *keys;
} scene;

static void scene_make(scene *S, int n_markers, uint64_t seed, double margin)
{
  memset(S, 0, sizeof *S);
  const bool per[3] = {true, false, false};
  S->A = adv_default_lattice(false, per);
  S->ens = adv_ensemble(&S->A);
  S->g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  S->loc = adv_add_location(S->g, "nodes", &S->A, MFTL_BOUNDARY_TRUNCATE);
  adv_decompose(S->g, S->loc, &S->A, 3 * g_nproc + 1, seed, 1, 1, false, &S->B);
  S->pos = (double *) malloc(3 * (size_t) (n_markers > 0 ? n_markers : 1) * sizeof(double));
  S->keys = (uint64_t *) malloc((size_t) (n_markers > 0 ? n_markers : 1) * sizeof(uint64_t));
  uint64_t s = seed * 7u + 1u;
  for (int i = 0; i < n_markers; ++i)
  {
    double x[3];
    for (int k = 0; k < 3; ++k)
    {
      const double lo = S->A.per[k] ? S->A.gmin[k] : S->A.origin[k] + margin * S->A.h[k];
      const double hi = S->A.per[k] ? S->A.gmin[k] + S->A.L[k] : S->A.origin[k] + (double) (S->A.N[k] - 1) * S->A.h[k] - margin * S->A.h[k];
      x[k] = lo + (hi - lo) * adv_unif(&s);
    }
    if (i % g_nproc != g_rank) continue;
    memcpy(&S->pos[3 * S->n_mine], x, sizeof x);
    S->keys[S->n_mine++] = (uint64_t) i;
  }
}

static void scene_free(scene *S)
{
  MFTLSolverGridDestroy(&S->g);
  MFTLEnsembleDestroy(&S->ens);
  adv_free_blocks(&S->B);
  free(S->pos);
  free(S->keys);
}

static MFTL_MarkerSpec user_markers(const scene *S)
{
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = S->n_mine;
  ms.user_pos = S->pos;
  ms.user_keys = S->keys;
  return ms;
}

static double f_quadratic(const double x[3], void *ctx)
{
  (void) ctx;
  return 0.3 + x[1] * x[1] - 0.7 * x[1] * x[2] + 1.3 * x[2] * x[2] + 0.4 * x[2];
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * Trilinear rows imported over a keyed window against HAT2 kernel rows bound
 * over a spatial window: the same values and the same owner loads, to
 * round-off -- two routes to one operator that share nothing upstream of the
 * transfer. Both reproduce affine fields; the keyed band holds remote samples.
 */
static int test_imported_trilinear_rows_equal_the_hat_kernel(void)
{
  scene S;
  scene_make(&S, 60, 41, 0.0);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.kernel = MFTL_KERNEL_HAT2;
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(S.ens, S.g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  native R;
  native_init(&R, S.n_mine);
  for (int i = 0; i < S.n_mine; ++i)
  {
    lagrange_row(&S.A, &S.B, 1, &S.pos[3 * i], 1.0, &R);
    native_close(&R, i);
  }
  MFTL_CouplingWindow *wk = MFTL_NULL;
  TEST_ASSERT(agreed(keyed_window(c, S.loc, &R, &wk, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplerStats st;
  MFTLCouplerReduceStats(c, &st);
  const long remote = st.band_remote;
  MFTL_MarkerSpec ms = user_markers(&S);
  MFTL_CouplingBinding *bi = MFTL_NULL;
  TEST_ASSERT(agreed(import_one(c, &ms, wk, S.loc, &R, MFTL_REPRODUCE_LINEAR, &bi) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_BindingQuality qi;
  MFTLCouplerBindingQuality(bi, &qi);
  TEST_ASSERT(agreed(qi.n_rows == S.n_mine && qi.n_failed == 0, MPI_COMM_WORLD));

  /* the kernel route */
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *ws = MFTL_NULL;
  MFTL_CouplingBinding *bk = MFTL_NULL;
  MFTL_WindowSpec wsp = {{0, 0, 0}, {0, 0, 0}};
  int s1 = MFTLCouplerRequestSupport(c, &ms, &wsp, &req);
  if (s1 == MFTL_COUPLING_OK) s1 = MFTLCouplerPrepareWindow(c, req, &ws);
  MFTLCouplerReleaseRequest(&req);
  if (s1 == MFTL_COUPLING_OK) s1 = MFTLCouplerBind(c, &ms, ws, &bk);
  TEST_ASSERT(agreed(s1 == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  const int64_t nm = S.n_mine > 0 ? S.n_mine : 1;
  double *Ui = (double *) malloc((size_t) nm * sizeof(double)), *Uk = (double *) malloc((size_t) nm * sizeof(double));
  rnd_ctx rc = {{0, 0, 0}, 99};
  adv_fill(&S.B, &S.A, 0, f_random, &rc, false);
  MFTL_SolverField F = adv_field(&S.B, S.loc, 0, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bi, 1, &F, Ui) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bk, 1, &F, Uk) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_v = 0.0;
  for (int i = 0; i < S.n_mine; ++i) worst_v = fmax(worst_v, fabs(Ui[i] - Uk[i]));
  affine aff = {0.4, {0.0, 1.3, -0.6}};
  adv_fill(&S.B, &S.A, 0, adv_affine_fn, &aff, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, bi, 1, &F, Ui) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_a = 0.0;
  for (int i = 0; i < S.n_mine; ++i) worst_a = fmax(worst_a, fabs(Ui[i] - f_affine(&S.pos[3 * i], &aff)));

  /* the same loads spread both ways land identically */
  double *L = (double *) malloc((size_t) nm * sizeof(double));
  for (int i = 0; i < S.n_mine; ++i) L[i] = hash01(S.keys[i] + 5u);
  adv_blocks T1 = S.B, T2 = S.B;
  for (int i = 0; i < S.B.n; ++i)
  {
    T1.a[i] = (double *) calloc((size_t) S.B.n_elem[i], sizeof(double));
    T2.a[i] = (double *) calloc((size_t) S.B.n_elem[i], sizeof(double));
  }
  MFTL_SolverField F1 = adv_field(&T1, S.loc, 0, 1, false), F2 = adv_field(&T2, S.loc, 0, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, bi, 1, &F1, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, bk, 1, &F2, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst_b = 0.0, sum_b = 0.0;
  for (int i = 0; i < S.B.n; ++i)
    for (int64_t e = 0; e < S.B.n_elem[i]; ++e)
    {
      worst_b = fmax(worst_b, fabs(T1.a[i][e] - T2.a[i][e]));
      sum_b += T1.a[i][e];
    }
  double sum_l = 0.0;
  for (int i = 0; i < S.n_mine; ++i) sum_l += L[i];
  double sums[2] = {sum_b, sum_l};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0)
    printf("    imported Q1 vs HAT2 kernel: values %.1e, loads %.1e, affine %.1e, force %.1e, keyed remote band %ld\n", worst_v, worst_b, worst_a,
           fabs(sums[0] - sums[1]), remote);
  TEST_ASSERT(agreed(worst_v <= 1e-14 && worst_b <= 1e-14 && worst_a <= 1e-12, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(sums[0] - sums[1]) <= 1e-12 * (fabs(sums[1]) + 1.0), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || remote > 0, MPI_COMM_WORLD));

  for (int i = 0; i < S.B.n; ++i)
  {
    free(T1.a[i]);
    free(T2.a[i]);
  }
  free(L);
  free(Ui);
  free(Uk);
  MFTLCouplerReleaseBinding(&bk);
  MFTLCouplerReleaseWindow(&ws);
  MFTLCouplerReleaseBinding(&bi);
  MFTLCouplerReleaseWindow(&wk);
  MFTLCouplerDestroy(&c);
  native_free(&R);
  scene_free(&S);
  return 0;
}

/**
 * Triquadratic Lagrange rows -- negative weights and all -- reproduce a
 * quadratic field exactly, which no kernel row does, and pass their declared
 * LINEAR check. A modal row (half a trilinear row: coefficients, not values)
 * passes as NONE and fails as CONSTANT on every rank, naming its marker.
 */
static int test_quadratic_and_modal_rows(void)
{
  scene S;
  scene_make(&S, 40, 43, 0.0);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(S.ens, S.g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  native R2, Rm;
  native_init(&R2, S.n_mine);
  native_init(&Rm, S.n_mine);
  for (int i = 0; i < S.n_mine; ++i)
  {
    lagrange_row(&S.A, &S.B, 2, &S.pos[3 * i], 1.0, &R2);
    native_close(&R2, i);
    lagrange_row(&S.A, &S.B, 1, &S.pos[3 * i], 0.5, &Rm);
    native_close(&Rm, i);
  }
  /* one window for both: the union of their columns */
  native U;
  native_init(&U, 0);
  for (int64_t e = 0; e < R2.nnz; ++e) native_push(&U, R2.keys[e], R2.owner[e], 0.0);
  for (int64_t e = 0; e < Rm.nnz; ++e) native_push(&U, Rm.keys[e], Rm.owner[e], 0.0);
  MFTL_CouplingWindow *w = MFTL_NULL;
  TEST_ASSERT(agreed(keyed_window(c, S.loc, &U, &w, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec ms = user_markers(&S);
  MFTL_CouplingBinding *b2 = MFTL_NULL, *bm = MFTL_NULL;
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R2, MFTL_REPRODUCE_LINEAR, &b2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_BindingQuality q2;
  MFTLCouplerBindingQuality(b2, &q2);
  adv_fill(&S.B, &S.A, 0, f_quadratic, MFTL_NULL, false);
  MFTL_SolverField F = adv_field(&S.B, S.loc, 0, 1, false);
  double *U2 = (double *) malloc((size_t) (S.n_mine > 0 ? S.n_mine : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b2, 1, &F, U2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst = 0.0;
  for (int i = 0; i < S.n_mine; ++i) worst = fmax(worst, fabs(U2[i] - f_quadratic(&S.pos[3 * i], MFTL_NULL)));
  double minw = q2.min_weight;
  MPI_Allreduce(MPI_IN_PLACE, &minw, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  if (g_rank == 0) printf("    triquadratic rows: quadratic field %.1e, most negative weight %.3f\n", worst, minw);
  TEST_ASSERT(agreed(worst <= 1e-12, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(minw < -0.01, MPI_COMM_WORLD));

  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &Rm, MFTL_REPRODUCE_NONE, &bm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bm);
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &Rm, MFTL_REPRODUCE_CONSTANT, &bm) == MFTL_COUPLING_BAD_STENCIL && bm == MFTL_NULL, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R2, MFTL_REPRODUCE_LINEAR, &bm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bm);

  free(U2);
  MFTLCouplerReleaseBinding(&b2);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  native_free(&R2);
  native_free(&Rm);
  native_free(&U);
  scene_free(&S);
  return 0;
}

/**
 * A row may reach anywhere: every marker's row also takes a little of the
 * node at the far corner of the domain (the rest of the weight trilinear, so
 * the row still sums to one but no longer reproduces linears -- declared
 * CONSTANT). That node lies outside every kernel's reach, is fetched from its
 * owner by every rank that needs it, and receives every rank's share of the
 * load exactly once: the owners' loads equal a transpose every rank recomputes
 * from the rows it can generate itself for the GLOBAL marker list.
 */
static int test_a_column_far_from_every_marker(void)
{
  enum
  {
    NA = 36
  };
  scene S;
  scene_make(&S, NA, 47, 0.0);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(S.ens, S.g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double eps = 0.125;
  int64_t far[3] = {S.A.N[0] - 1, S.A.N[1] - 1, S.A.N[2] - 1};
  const uint64_t far_key = lat_key(&S.A, far);
  const int far_owner = lat_owner(&S.B, far);
  native R;
  native_init(&R, S.n_mine);
  for (int i = 0; i < S.n_mine; ++i)
  {
    lagrange_row(&S.A, &S.B, 1, &S.pos[3 * i], 1.0 - eps, &R);
    native_push(&R, far_key, far_owner, eps);
    native_close(&R, i);
  }
  MFTL_CouplingWindow *w = MFTL_NULL;
  TEST_ASSERT(agreed(keyed_window(c, S.loc, &R, &w, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec ms = user_markers(&S);
  MFTL_CouplingBinding *b = MFTL_NULL;
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R, MFTL_REPRODUCE_LINEAR, &b) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R, MFTL_REPRODUCE_CONSTANT, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  double *L = (double *) malloc((size_t) (S.n_mine > 0 ? S.n_mine : 1) * sizeof(double));
  for (int i = 0; i < S.n_mine; ++i) L[i] = 1.0 + hash01(S.keys[i] * 3u + 1u);
  adv_blocks T = S.B;
  for (int i = 0; i < S.B.n; ++i) T.a[i] = (double *) calloc((size_t) S.B.n_elem[i], sizeof(double));
  MFTL_SolverField F = adv_field(&T, S.loc, 0, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* every rank: the global markers' rows and loads, the transpose at its owned nodes */
  double worst = 0.0, far_load = 0.0, far_ref = 0.0;
  {
    uint64_t s = 47u * 7u + 1u; /* scene_make's marker stream */
    for (int i = 0; i < NA; ++i)
    {
      double x[3];
      for (int k = 0; k < 3; ++k)
      {
        const double lo = S.A.per[k] ? S.A.gmin[k] : S.A.origin[k];
        const double hi = S.A.per[k] ? S.A.gmin[k] + S.A.L[k] : S.A.origin[k] + (double) (S.A.N[k] - 1) * S.A.h[k];
        x[k] = lo + (hi - lo) * adv_unif(&s);
      }
      native Ri;
      native_init(&Ri, 1);
      lagrange_row(&S.A, &S.B, 1, x, 1.0 - eps, &Ri);
      native_push(&Ri, far_key, far_owner, eps);
      const double Li = 1.0 + hash01((uint64_t) i * 3u + 1u);
      for (int64_t e = 0; e < Ri.nnz; ++e)
      {
        if (Ri.owner[e] != g_rank) continue;
        int64_t idx[3];
        idx[0] = (int64_t) (Ri.keys[e] % (uint64_t) S.A.N[0]);
        idx[1] = (int64_t) ((Ri.keys[e] / (uint64_t) S.A.N[0]) % (uint64_t) S.A.N[1]);
        idx[2] = (int64_t) (Ri.keys[e] / (uint64_t) (S.A.N[0] * S.A.N[1]));
        if (Ri.keys[e] == far_key) far_ref += Ri.w[e] * Li;
        for (int bb = 0; bb < S.B.n; ++bb)
        {
          bool in = true;
          for (int k = 0; k < 3; ++k)
            if (idx[k] < S.B.lo[bb][k] || idx[k] >= S.B.hi[bb][k]) in = false;
          if (!in) continue;
          const int64_t el = (idx[0] - S.B.alo[bb][0]) + S.B.adims[bb][0] * ((idx[1] - S.B.alo[bb][1]) + S.B.adims[bb][1] * (idx[2] - S.B.alo[bb][2]));
          T.a[bb][el] -= Ri.w[e] * Li; /* what is left must be zero */
        }
      }
      native_free(&Ri);
    }
    for (int bb = 0; bb < S.B.n; ++bb)
      for (int64_t el = 0; el < S.B.n_elem[bb]; ++el) worst = fmax(worst, fabs(T.a[bb][el]));
    far_load = far_ref;
  }
  MPI_Allreduce(MPI_IN_PLACE, &far_load, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("    far column: owner loads vs the transpose %.1e; the far node's load %.4f from %d markers\n", worst, far_load, NA);
  TEST_ASSERT(agreed(worst <= 1e-13, MPI_COMM_WORLD));
  for (int i = 0; i < S.B.n; ++i) free(T.a[i]);
  free(L);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  native_free(&R);
  scene_free(&S);
  return 0;
}

/**
 * Lowest-order Raviart-Thomas-like rows on the faces of a hexahedral grid,
 * registered as a DIRECTIONAL point set with oriented area vectors: each
 * component is interpolated linearly between its two faces, u_x = (1 - xi)
 * q_lo / s_lo + xi q_hi / s_hi. They reproduce every constant vector and every
 * field whose component k varies along axis k only, are accepted as CONSTANT
 * and refused as LINEAR -- the declaration is checked, not trusted -- and the
 * owners are named in a REVERSED solver communicator, translated by the coupler.
 */
static int test_raviart_thomas_rows_on_a_face_set(void)
{
  const int n = 6;
  const double h = 1.0 / n;
  MPI_Comm rev;
  MPI_Comm_split(MPI_COMM_WORLD, 0, g_nproc - 1 - g_rank, &rev);
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(rev);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "rt-faces";
  ld.directional = true;
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  /* faces: family f, normal index a in [0, n], tangential b, c in [0, n) */
  const int64_t nf = 3 * (int64_t) n * n * (n + 1);
  double *X = (double *) malloc(3 * (size_t) nf * sizeof(double)), *Sv = (double *) malloc(3 * (size_t) nf * sizeof(double));
  double *q = (double *) malloc((size_t) nf * sizeof(double));
  uint64_t *key = (uint64_t *) malloc((size_t) nf * sizeof(uint64_t));
  int *owner_world = (int *) malloc((size_t) nf * sizeof(int));
  int64_t n_mine = 0;
  double *mX = (double *) malloc(3 * (size_t) nf * sizeof(double)), *mS = (double *) malloc(3 * (size_t) nf * sizeof(double));
  double *mq = (double *) malloc((size_t) nf * sizeof(double));
  uint64_t *mk = (uint64_t *) malloc((size_t) nf * sizeof(uint64_t));
  for (int f = 0; f < 3; ++f)
    for (int a = 0; a <= n; ++a)
      for (int b = 0; b < n; ++b)
        for (int cc = 0; cc < n; ++cc)
        {
          const int64_t i = (((int64_t) f * (n + 1) + a) * n + b) * n + cc;
          const int t1 = (f + 1) % 3, t2 = (f + 2) % 3;
          X[3 * i + f] = a * h;
          X[3 * i + t1] = (b + 0.5) * h;
          X[3 * i + t2] = (cc + 0.5) * h;
          const double sg = (hash01((uint64_t) i * 7u + 3u) < 0.0) ? -1.0 : 1.0;
          for (int k = 0; k < 3; ++k) Sv[3 * i + k] = (k == f) ? sg * h * h : 0.0;
          q[i] = h * h * h;
          key[i] = (uint64_t) i * 2u + 1u;
          owner_world[i] = (int) ((uint64_t) (hash01((uint64_t) i * 5u + 1u) * 1e9 + 2e9) % (uint64_t) g_nproc);
          if (owner_world[i] != g_rank) continue;
          memcpy(&mX[3 * n_mine], &X[3 * i], 3 * sizeof(double));
          memcpy(&mS[3 * n_mine], &Sv[3 * i], 3 * sizeof(double));
          mq[n_mine] = q[i];
          mk[n_mine++] = key[i];
        }
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.n = n_mine;
  pd.keys = mk;
  pd.pos = mX;
  pd.measure = mq;
  pd.direction = mS;
  for (int k = 0; k < 3; ++k) pd.h_uniform[k] = h;
  TEST_ASSERT(agreed(MFTLSolverGridAddPoints(g, &pd) >= 0, MPI_COMM_WORLD));
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* markers anywhere in the box; rows in the cell containing each */
  enum
  {
    NA = 40
  };
  double pos[3 * NA];
  uint64_t mkeys[NA];
  int nm = 0;
  uint64_t s = 71;
  for (int i = 0; i < NA; ++i)
  {
    double x[3];
    for (int k = 0; k < 3; ++k) x[k] = 0.999 * adv_unif(&s);
    if (i % g_nproc != g_rank) continue;
    memcpy(&pos[3 * nm], x, sizeof x);
    mkeys[nm++] = (uint64_t) i;
  }
  int64_t *rp = (int64_t *) calloc((size_t) nm + 1, sizeof(int64_t));
  uint64_t *ck = (uint64_t *) malloc(6 * (size_t) (nm > 0 ? nm : 1) * sizeof(uint64_t));
  int *co = (int *) malloc(6 * (size_t) (nm > 0 ? nm : 1) * sizeof(int));
  double *cw = (double *) malloc(18 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  int64_t nnz = 0;
  for (int i = 0; i < nm; ++i)
  {
    int cell[3];
    double xi[3];
    for (int k = 0; k < 3; ++k)
    {
      cell[k] = (int) floor(pos[3 * i + k] / h);
      xi[k] = pos[3 * i + k] / h - cell[k];
    }
    for (int f = 0; f < 3; ++f)
      for (int side = 0; side < 2; ++side)
      {
        const int t1 = (f + 1) % 3, t2 = (f + 2) % 3;
        const int64_t fi = (((int64_t) f * (n + 1) + cell[f] + side) * n + cell[t1]) * n + cell[t2];
        const double lam = side ? xi[f] : 1.0 - xi[f];
        ck[nnz] = key[fi];
        /* the owner in the reversed communicator */
        co[nnz] = g_nproc - 1 - owner_world[fi];
        for (int k = 0; k < 3; ++k) cw[3 * nnz + k] = (k == f) ? lam / Sv[3 * fi + f] : 0.0;
        ++nnz;
      }
    rp[i + 1] = nnz;
  }
  MFTL_SampleList L = {loc, nnz, ck, co};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestSamples(c, 1, &L, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = nm;
  ms.user_pos = pos;
  ms.user_keys = mkeys;
  MFTL_NativeRows nr = {loc, rp, ck, cw, MFTL_REPRODUCE_LINEAR};
  MFTL_CouplingBinding *b = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerImportRows(c, &ms, w, 1, &nr, &b) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
  nr.reproduces = MFTL_REPRODUCE_CONSTANT;
  TEST_ASSERT(agreed(MFTLCouplerImportRows(c, &ms, w, 1, &nr, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindingComponents(b, loc) == 3, MPI_COMM_WORLD));

  /* u = (a0 + a1 x, b0 + b1 y, c0 + c1 z): each component along its own axis */
  const double cf[6] = {0.7, -1.2, 0.3, 0.9, -0.4, 1.6};
  double *val = (double *) malloc((size_t) (n_mine > 0 ? n_mine : 1) * sizeof(double));
  for (int64_t i = 0; i < n_mine; ++i)
  {
    const double u[3] = {cf[0] + cf[1] * mX[3 * i], cf[2] + cf[3] * mX[3 * i + 1], cf[4] + cf[5] * mX[3 * i + 2]};
    val[i] = mS[3 * i] * u[0] + mS[3 * i + 1] * u[1] + mS[3 * i + 2] * u[2];
  }
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = loc;
  F.data = &val;
  double U[3 * NA];
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst = 0.0;
  for (int i = 0; i < nm; ++i)
  {
    const double u[3] = {cf[0] + cf[1] * pos[3 * i], cf[2] + cf[3] * pos[3 * i + 1], cf[4] + cf[5] * pos[3 * i + 2]};
    for (int k = 0; k < 3; ++k) worst = fmax(worst, fabs(U[3 * i + k] - u[k]));
  }
  if (g_rank == 0) printf("    Raviart-Thomas-like rows (owners named in a reversed communicator): %.1e\n", worst);
  TEST_ASSERT(agreed(worst <= 1e-13, MPI_COMM_WORLD));

  free(val);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  MPI_Comm_free(&rev);
  free(rp);
  free(ck);
  free(co);
  free(cw);
  free(X);
  free(Sv);
  free(q);
  free(key);
  free(owner_world);
  free(mX);
  free(mS);
  free(mq);
  free(mk);
  return 0;
}

/** @brief A solid: one node of the lattice has no sample. */
static bool solid_one(const int64_t idx[3]) { return idx[0] == 4 && idx[1] == 3 && idx[2] == 5; }

/**
 * Every way the protocol can be misused, each agreed on every rank before
 * anything is built or moved: a wrong owner, a masked sample, an unexported
 * sample -- named back to its owner by key, exported, and then found --
 * locations that differ between ranks, a column outside the window, a NaN
 * weight, a broken row pointer, a QUADRATURE or gradient import, a kernel bind
 * on a keyed window, a stage of imported rows. A key repeated within a row is
 * summed: splitting a weight changes no bit.
 */
static int test_the_keyed_protocol_refuses_misuse(void)
{
  scene S;
  scene_make(&S, 24, 53, 0.0);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(S.ens, S.g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  native R;
  native_init(&R, S.n_mine);
  for (int i = 0; i < S.n_mine; ++i)
  {
    lagrange_row(&S.A, &S.B, 1, &S.pos[3 * i], 1.0, &R);
    native_close(&R, i);
  }
  /* a wrong owner, on one rank only */
  if (R.nnz > 0 && g_rank == 0) R.owner[0] = (R.owner[0] + 1) % g_nproc;
  MFTL_CouplingWindow *w = MFTL_NULL;
  const int wrong = keyed_window(c, S.loc, &R, &w, MFTL_NULL);
  TEST_ASSERT(agreed(g_nproc == 1 ? wrong == MFTL_COUPLING_OK : wrong == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerReleaseWindow(&w);
  if (R.nnz > 0 && g_rank == 0)
  {
    const int64_t i0[3] = {(int64_t) (R.keys[0] % (uint64_t) S.A.N[0]), (int64_t) ((R.keys[0] / (uint64_t) S.A.N[0]) % (uint64_t) S.A.N[1]),
                           (int64_t) (R.keys[0] / (uint64_t) (S.A.N[0] * S.A.N[1]))};
    R.owner[0] = lat_owner(&S.B, i0);
  }
  /* an owner outside the communicator (rank 0 always holds marker 0) */
  const int keep_owner = (R.nnz > 0) ? R.owner[0] : 0;
  if (g_rank == 0) R.owner[0] = g_nproc + 3;
  TEST_ASSERT(agreed(keyed_window(c, S.loc, &R, &w, MFTL_NULL) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  if (g_rank == 0) R.owner[0] = keep_owner;
  MFTLCouplerReleaseWindow(&w);
  /* one rank naming a location the grid does not have */
  {
    MFTL_SampleList L = {(g_rank == 0) ? S.loc + 7 : S.loc, 0, MFTL_NULL, MFTL_NULL};
    MFTL_SupportRequest *rq = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSamples(c, 1, &L, &rq) == MFTL_COUPLING_INVALID && rq == MFTL_NULL, MPI_COMM_WORLD));
  }
  /* the good window */
  TEST_ASSERT(agreed(keyed_window(c, S.loc, &R, &w, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec ms = user_markers(&S);
  MFTL_CouplingBinding *b = MFTL_NULL;
  /* a kernel bind on a keyed window */
  TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &b) == MFTL_COUPLING_INVALID && b == MFTL_NULL, MPI_COMM_WORLD));
  /* a NaN weight, a broken row pointer, a column not in the window -- each on
   * the last rank holding markers (24 markers: at np = 27 the last ranks hold none) */
  const int bad_rank = ((g_nproc < 24) ? g_nproc : 24) - 1;
  for (int t = 0; t < 3; ++t)
  {
    double keep_w = 0.0;
    int64_t keep_rp = 0;
    uint64_t keep_k = 0;
    if (g_rank == bad_rank)
    {
      keep_w = R.w[0];
      keep_rp = R.row_ptr[1];
      keep_k = R.keys[0];
      if (t == 0) R.w[0] = NAN;
      if (t == 1) R.row_ptr[1] = R.row_ptr[0] - 1;
      if (t == 2) R.keys[0] = (uint64_t) (S.A.N[0] * S.A.N[1] * S.A.N[2] + 5); /* no such sample anywhere */
    }
    TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R, MFTL_REPRODUCE_LINEAR, &b) == MFTL_COUPLING_INVALID && b == MFTL_NULL, MPI_COMM_WORLD));
    if (g_rank == bad_rank)
    {
      R.w[0] = keep_w;
      R.row_ptr[1] = keep_rp;
      R.keys[0] = keep_k;
    }
  }
  /* QUADRATURE and gradient imports */
  MFTL_MarkerSpec mq = ms;
  mq.kind = MFTL_MARKERS_QUADRATURE;
  TEST_ASSERT(agreed(import_one(c, &mq, w, S.loc, &R, MFTL_REPRODUCE_LINEAR, &b) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  mq = ms;
  mq.operation = MFTL_OPERATOR_GRADIENT_FIT;
  TEST_ASSERT(agreed(import_one(c, &mq, w, S.loc, &R, MFTL_REPRODUCE_LINEAR, &b) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  /* the good import; no stage of it; a split weight changes no bit */
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &R, MFTL_REPRODUCE_LINEAR, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_StageGeometry sg = {S.pos, MFTL_NULL};
  MFTL_CouplingBinding *bs = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(c, b, &sg, &bs) == MFTL_COUPLING_INVALID && bs == MFTL_NULL, MPI_COMM_WORLD));
  native Rs;
  native_init(&Rs, S.n_mine);
  for (int i = 0; i < S.n_mine; ++i)
  {
    for (int64_t e = R.row_ptr[i]; e < R.row_ptr[i + 1]; ++e)
    {
      /* split into a power-of-two part and the rest, the rest first */
      const double part = ldexp(1.0, -3);
      if (R.w[e] > part)
      {
        native_push(&Rs, R.keys[e], R.owner[e], R.w[e] - part);
        native_push(&Rs, R.keys[e], R.owner[e], part);
      }
      else
        native_push(&Rs, R.keys[e], R.owner[e], R.w[e]);
    }
    native_close(&Rs, i);
  }
  MFTL_CouplingBinding *b2 = MFTL_NULL;
  TEST_ASSERT(agreed(import_one(c, &ms, w, S.loc, &Rs, MFTL_REPRODUCE_LINEAR, &b2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  rnd_ctx rc = {{0, 0, 0}, 5};
  adv_fill(&S.B, &S.A, 0, f_random, &rc, false);
  MFTL_SolverField F = adv_field(&S.B, S.loc, 0, 1, false);
  double U1[24], U2[24];
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b2, 1, &F, U2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double worst = 0.0;
  for (int i = 0; i < S.n_mine; ++i) worst = fmax(worst, fabs(U1[i] - U2[i]));
  TEST_ASSERT(agreed(worst <= 4e-16, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b2);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  native_free(&Rs);
  native_free(&R);
  scene_free(&S);
  return 0;
}

/**
 * Coverage by key. A masked (solid) node named as a column is INVALID: there
 * is no such sample. On a SELECTED lattice that has exported nothing, every
 * named column is NEED_COVERAGE, and each owner gets exactly its named keys as
 * one-sample regions; exporting just those and retrying succeeds -- the same
 * request/fulfil loop as a support request, at sample granularity. A SELECTED
 * point set names an unregistered key back by key alone.
 */
static int test_coverage_by_key(void)
{
  const bool per[3] = {true, false, false};
  adv_lat A = adv_default_lattice(false, per);
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "nodes", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose_on(g_rank, g_nproc, g, loc, &A, 2 * g_nproc + 1, 59, 0, 1, false, solid_one, &B);
  /* SELECTED with nothing exported */
  MFTL_CoverageDesc cv = {loc, MFTL_GRID_SELECTED, 0, MFTL_NULL, MFTL_NULL};
  MFTLSolverGridSetCoverage(g, &cv);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* every rank names a few nodes (one of them on rank 0: the solid) */
  enum
  {
    NK = 5
  };
  uint64_t keys[NK];
  int owners[NK];
  int nk = 0;
  for (int t = 0; t < NK; ++t)
  {
    int64_t idx[3] = {(g_rank * 3 + t * 2) % A.N[0], (g_rank + t) % A.N[1], (2 * g_rank + 3 * t) % A.N[2]};
    if (solid_one(idx)) continue;
    keys[nk] = lat_key(&A, idx);
    owners[nk++] = lat_owner(&B, idx);
  }
  MFTL_SampleList L = {loc, nk, keys, owners};
  MFTL_SupportRequest *req = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerRequestSamples(c, 1, &L, &req) == MFTL_COUPLING_NEED_COVERAGE, MPI_COMM_WORLD));
  MFTL_RequestView v;
  MFTLCouplerGetRequests(req, &v);
  /* export exactly what was named here */
  int64_t *boxes = (int64_t *) malloc(6 * (size_t) (v.n_missing > 0 ? v.n_missing : 1) * sizeof(int64_t));
  int nb = 0, bad = 0;
  for (int r = 0; r < v.n_missing; ++r)
  {
    if (!v.missing[r].by_key) ++bad;
    for (int k = 0; k < 3; ++k)
    {
      boxes[6 * nb + k] = v.missing[r].index_lo[k];
      boxes[6 * nb + 3 + k] = v.missing[r].index_hi[k];
      if (v.missing[r].index_hi[k] - v.missing[r].index_lo[k] != 1) ++bad;
    }
    const uint64_t kk = (uint64_t) (v.missing[r].index_lo[0] + A.N[0] * (v.missing[r].index_lo[1] + A.N[1] * v.missing[r].index_lo[2]));
    if (kk != v.missing[r].key) ++bad;
    ++nb;
  }
  TEST_ASSERT(agreed(bad == 0, MPI_COMM_WORLD));
  MFTLCouplerReleaseRequest(&req);
  MFTL_CoverageDesc cv2 = {loc, MFTL_GRID_SELECTED, nb, boxes, MFTL_NULL};
  MFTLSolverGridSetCoverage(g, &cv2);
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingWindow *w = MFTL_NULL;
  int st = MFTLCouplerRequestSamples(c, 1, &L, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseWindow(&w);
  /* the solid, named by rank 0 */
  const int64_t sidx[3] = {4, 3, 5};
  uint64_t sk = lat_key(&A, sidx);
  int so = lat_owner(&B, sidx);
  MFTL_SampleList Ls = {loc, (g_rank == 0) ? 1 : 0, &sk, &so};
  TEST_ASSERT(agreed(MFTLCouplerRequestSamples(c, 1, &Ls, &req) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  free(boxes);
  adv_free_blocks(&B);

  /* a SELECTED point set: an unregistered key comes back by key */
  g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "pts";
  const int pl = MFTLSolverGridAddLocation(g, &ld);
  const double box[6] = {A.gmin[0], A.gmin[1], A.gmin[2], A.gmin[0] + A.L[0], A.gmin[1] + A.L[1], A.gmin[2] + A.L[2]};
  MFTL_ProviderRegions pr = {pl, (g_rank == 0) ? 1 : 0, box, {0.1, 0.1, 0.1}, {0.1, 0.1, 0.1}};
  MFTLSolverGridSetProviderRegions(g, &pr);
  MFTL_CoverageDesc cp = {pl, MFTL_GRID_SELECTED, 0, MFTL_NULL, MFTL_NULL};
  MFTLSolverGridSetCoverage(g, &cp);
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const uint64_t pk = 4242;
  const int po = 0;
  MFTL_SampleList Lp = {pl, 1, &pk, &po};
  TEST_ASSERT(agreed(MFTLCouplerRequestSamples(c, 1, &Lp, &req) == MFTL_COUPLING_NEED_COVERAGE, MPI_COMM_WORLD));
  MFTLCouplerGetRequests(req, &v);
  int named = 0;
  for (int r = 0; r < v.n_missing; ++r)
    if (v.missing[r].by_key && v.missing[r].key == pk) ++named;
  int total_named = named;
  MPI_Allreduce(MPI_IN_PLACE, &total_named, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(total_named == g_nproc && (g_rank != 0 || named == g_nproc), MPI_COMM_WORLD));
  MFTLCouplerReleaseRequest(&req);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling: imported native rows (np = %d) ===\n", g_nproc);

  RUN_TEST(test_imported_trilinear_rows_equal_the_hat_kernel);
  RUN_TEST(test_quadratic_and_modal_rows);
  RUN_TEST(test_a_column_far_from_every_marker);
  RUN_TEST(test_raviart_thomas_rows_on_a_face_set);
  RUN_TEST(test_the_keyed_protocol_refuses_misuse);
  RUN_TEST(test_coverage_by_key);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
