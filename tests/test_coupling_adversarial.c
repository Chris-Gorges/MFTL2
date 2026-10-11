/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Attacks on the coupler: configurations chosen to make it return a
 *        plausible wrong answer.
 *
 * test_coupling.c and test_coupling_protocol.c check the coupler on the
 * configurations it was designed around -- one block per rank, cubic lattices,
 * the unit box. Every test here breaks one of those habits, because a bug that
 * survives them lives exactly in the habit:
 *
 *   - IRREGULAR DECOMPOSITIONS: a random k-d partition of the lattice into
 *     several blocks per rank (some ranks own none, some own an empty block),
 *     each with its own ghost width PER AXIS and padding beyond the ghosts that
 *     holds NaN, assigned to ranks by a hash, so a block's neighbours are
 *     arbitrary ranks;
 *   - ANISOTROPIC, OFFSET LATTICES: a different count and spacing per axis, a
 *     domain that does not start at the origin, periodic on some axes and walled
 *     on others, cell- and node-centred locations bound together;
 *   - MARKERS WHERE ARITHMETIC IS TIGHT: exactly on samples, on bin and rank
 *     boundaries, on the periodic seam (including the one-ulp-below-min
 *     position that wraps onto max), several periods away, on a wall;
 *   - SOLVER LAYOUTS: interleaved (AoS) components through the stride, more
 *     ghosted blocks than the halo route tracks, two couplers interleaving
 *     their transfers.
 *
 * The reference is brute force over the GLOBAL lattice: every index whose image
 * lies in a marker's support, enumerated by the test, through the row core
 * MFTLCouplingAssembleRow() (tested on its own in test_coupling_kernels). It
 * shares no discovery, coverage, routing, band, schedule or candidate code
 * with the coupler. Spreading is checked against the same rows applied to
 * every marker of every rank (gathered), into the samples each rank owns.
 */

#include "test_coupling_irregular.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * One scenario, end to end
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  adv_lat lat[2];  ///< up to two locations (e.g. cell and node centred)
  int n_loc;
  MFTL_CouplerParams P;
  int n_boxes;
  uint64_t seed;
  int max_ghost;
  bool halo_valid;
  int ncomp;
  int n_random;
  double wall_gap;
  double skin;
} adv_scenario;

typedef struct
{
  long rows, bad_rows;
  double worst_row, worst_affine, worst_spread, power_rel;
  long halo, remote, self;
  int status;
} adv_result;

/** @brief Is lattice index @p idx in block @p i's owned box? */
static bool adv_owned(const adv_blocks *B, int i, const int64_t idx[3])
{
  for (int k = 0; k < 3; ++k)
    if (idx[k] < B->lo[i][k] || idx[k] >= B->hi[i][k]) return false;
  return true;
}

/** @brief Element of OWNED wrapped index @p idx in block @p i. */
static int64_t adv_owned_elem(const adv_blocks *B, int i, const int64_t idx[3])
{
  return (idx[0] - B->alo[i][0]) + B->adims[i][0] * ((idx[1] - B->alo[i][1]) + B->adims[i][1] * (idx[2] - B->alo[i][2]));
}

enum
{
  ADV_ROWCAP = 4096
};

/** @brief Every bound row of location @p l against the brute-force row. */
static void adv_check_rows(const adv_scenario *S, int l, const MFTL_CouplingBinding *b, adv_result *res)
{
  const int64_t nm = MFTLCouplerBindingMarkerCount(b);
  const double *bpos = MFTLCouplerBindingPositions(b);
  uint64_t *rk = (uint64_t *) malloc(ADV_ROWCAP * sizeof(uint64_t)), *kc = (uint64_t *) malloc(ADV_ROWCAP * sizeof(uint64_t));
  double *rw = (double *) malloc(ADV_ROWCAP * sizeof(double)), *wc = (double *) malloc(ADV_ROWCAP * sizeof(double));
  for (int64_t i = 0; i < nm; ++i)
  {
    int n = 0;
    const uint64_t *keys;
    const double *w;
    ++res->rows;
    if (MFTLCouplerBindingRow(b, l, i, &n, &keys, &w) != MFTL_SUCCESS || n > ADV_ROWCAP)
    {
      ++res->bad_rows;
      continue;
    }
    memcpy(kc, keys, (size_t) n * sizeof(uint64_t));
    memcpy(wc, w, (size_t) n * sizeof(double));
    const int nr = adv_ref_row(&S->lat[l], &S->P, &bpos[3 * i], rk, rw, ADV_ROWCAP);
    if (nr < 0)
    {
      ++res->bad_rows;
      continue;
    }
    const double diff = adv_row_diff(n, kc, wc, nr, rk, rw);
    if (diff > 1e-11) ++res->bad_rows;
    res->worst_row = fmax(res->worst_row, diff);
  }
  free(rk);
  free(kc);
  free(rw);
  free(wc);
}

/**
 * @brief The spread of field @p f (location @p l, component @p cpt) against
 *        the gathered brute-force transpose, owned sample by owned sample.
 *        @p all holds every marker's frame position and loads, @p stride doubles each.
 */
static void adv_check_spread_field(const adv_scenario *S, int l, int cpt, int f, const adv_blocks *B, const double *all, int tot, int stride, adv_result *res)
{
  const adv_lat *A = &S->lat[l];
  uint64_t *rk = (uint64_t *) malloc(ADV_ROWCAP * sizeof(uint64_t));
  double *rw = (double *) malloc(ADV_ROWCAP * sizeof(double));
  double **exp = (double **) calloc((size_t) (B->n > 0 ? B->n : 1), sizeof(double *));
  double **sc = (double **) calloc((size_t) (B->n > 0 ? B->n : 1), sizeof(double *));
  for (int i = 0; i < B->n; ++i)
  {
    exp[i] = (double *) calloc((size_t) B->n_elem[i], sizeof(double));
    sc[i] = (double *) calloc((size_t) B->n_elem[i], sizeof(double));
  }
  for (int m = 0; m < tot; ++m)
  {
    const int n = adv_ref_row(A, &S->P, &all[stride * m], rk, rw, ADV_ROWCAP);
    /* A weight's round-off is relative to its ROW, not to itself: a sample
     * reached only by weights that are zero in exact arithmetic (a marker on a
     * node next to a wall) holds noise of the row's size. */
    double rowabs = 0.0;
    for (int e = 0; e < n; ++e) rowabs += fabs(rw[e]);
    const double load = all[stride * m + 3 + f];
    for (int e = 0; e < n; ++e)
    {
      const int64_t idx[3] = {(int64_t) (rk[e] % (uint64_t) A->N[0]), (int64_t) ((rk[e] / (uint64_t) A->N[0]) % (uint64_t) A->N[1]),
                              (int64_t) (rk[e] / (uint64_t) (A->N[0] * A->N[1]))};
      for (int i = 0; i < B->n; ++i)
      {
        if (!adv_owned(B, i, idx)) continue;
        const int64_t el = adv_owned_elem(B, i, idx);
        exp[i][el] += rw[e] * load;
        sc[i][el] += rowabs * fabs(load);
      }
    }
  }
  for (int i = 0; i < B->n; ++i)
    for (int64_t e = 0; e < B->n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(B, i, e, idx);
      if (!adv_owned(B, i, idx)) continue;
      const double got = B->a[i][e * S->ncomp + cpt];
      res->worst_spread = fmax(res->worst_spread, fabs(got - exp[i][e]) / fmax(sc[i][e], 1e-12));
    }
  for (int i = 0; i < B->n; ++i)
  {
    free(exp[i]);
    free(sc[i]);
  }
  free(exp);
  free(sc);
  free(rk);
  free(rw);
}

/** @brief Every element of every block's arrays: owned ones to @p owned_value, the rest to @p other. */
static void adv_set(adv_blocks *B, int ncomp, double owned_value, double other)
{
  for (int i = 0; i < B->n; ++i)
    for (int64_t e = 0; e < B->n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(B, i, e, idx);
      const double v = adv_owned(B, i, idx) ? owned_value : other;
      for (int cpt = 0; cpt < ncomp; ++cpt) B->a[i][e * ncomp + cpt] = v;
    }
}

/** @brief Whether every non-owned element still holds @p other. */
static bool adv_untouched(const adv_blocks *B, int ncomp, double other)
{
  for (int i = 0; i < B->n; ++i)
    for (int64_t e = 0; e < B->n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(B, i, e, idx);
      if (adv_owned(B, i, idx)) continue;
      for (int cpt = 0; cpt < ncomp; ++cpt)
        if (B->a[i][e * ncomp + cpt] != other) return false;
    }
  return true;
}

/** @brief Sum over owned elements of u * b, and of |u * b|. */
static void adv_owned_dot(const adv_blocks *Bu, const adv_blocks *Bb, int ncomp, double *dot, double *abs_dot)
{
  for (int i = 0; i < Bb->n; ++i)
    for (int64_t e = 0; e < Bb->n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(Bb, i, e, idx);
      if (!adv_owned(Bb, i, idx)) continue;
      for (int cpt = 0; cpt < ncomp; ++cpt)
      {
        const double p = Bu->a[i][e * ncomp + cpt] * Bb->a[i][e * ncomp + cpt];
        *dot += p;
        *abs_dot += fabs(p);
      }
    }
}

/**
 * @brief Runs a scenario: registers the irregular decomposition, binds USER
 *        markers, and checks every row, an affine interpolation, the spread
 *        against the gathered brute-force transpose, and the power identity.
 */
static void adv_run(const adv_scenario *S, adv_result *res)
{
  memset(res, 0, sizeof *res);
  MFTL_Ensemble *ens = adv_ensemble(&S->lat[0]);
  const MFTL_Domain *d = &ens->domain;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  int loc[2];
  adv_blocks B[2];
  for (int l = 0; l < S->n_loc; ++l)
  {
    char name[16];
    snprintf(name, sizeof name, "loc%d", l);
    loc[l] = adv_add_location(g, name, &S->lat[l], MFTL_BOUNDARY_TRUNCATE);
    adv_decompose(g, loc[l], &S->lat[l], S->n_boxes, S->seed + 977u * (uint64_t) l, S->max_ghost, S->ncomp, l == 0 && (g_rank % 3 == 1), &B[l]);
  }
  MFTL_KernelInfo ki;
  MFTLCouplingKernelInfo(S->P.kernel, &ki);
  double reach = 0.0;
  for (int l = 0; l < S->n_loc; ++l)
    for (int k = 0; k < 3; ++k)
      reach = fmax(reach, ki.half_width * S->P.support_scale * ((S->P.failure & MFTL_STENCIL_GROW) ? S->P.max_growth : 1.0) * S->lat[l].h[k]);
  adv_markers M;
  adv_markers_make(d, &S->lat[0], reach, S->n_random, S->seed ^ 0x5151u, S->wall_gap, &M);

  MFTL_Coupler *c = MFTL_NULL;
  int st = MFTLCouplerCreate(ens, g, &S->P, &c);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = M.n;
  ms.user_pos = M.pos;
  ms.user_keys = M.keys;
  MFTL_WindowSpec ws = {{S->skin, S->skin, S->skin}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *win = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &win);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(c, &ms, win, &b);
  MFTLCouplerReleaseRequest(&req);
  res->status = st;
  if (st != MFTL_COUPLING_OK)
  {
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&win);
    MFTLCouplerDestroy(&c);
    for (int l = 0; l < S->n_loc; ++l) adv_free_blocks(&B[l]);
    adv_markers_free(&M);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
    return;
  }
  MFTL_CouplerStats cs;
  MFTLCouplerGetStats(c, &cs);
  res->halo = cs.band_halo;
  res->remote = cs.band_remote;
  res->self = cs.band_self;
  const int64_t nm = MFTLCouplerBindingMarkerCount(b);
  const double *bpos = MFTLCouplerBindingPositions(b);

  /* ── rows ── */
  for (int l = 0; l < S->n_loc; ++l) adv_check_rows(S, l, b, res);

  /* ── interpolation of an affine field per location and component ── */
  affine Af[2][3];
  MFTL_SolverField F[6];
  double *ptrs[6][ADV_MAX_BOXES];
  int nf = 0;
  for (int l = 0; l < S->n_loc; ++l)
    for (int cpt = 0; cpt < S->ncomp; ++cpt, ++nf)
    {
      affine *A = &Af[l][cpt];
      A->a = 0.25 + 0.5 * cpt - 0.125 * l;
      /* on a periodic axis a linear function is not periodic: constant there */
      for (int k = 0; k < 3; ++k) A->g[k] = S->lat[l].per[k] ? 0.0 : ((k == cpt % 3) ? 1.0 + l : -0.5 * (k + 1));
      adv_fill(&B[l], &S->lat[l], cpt, adv_affine_fn, A, S->halo_valid);
      memset(&F[nf], 0, sizeof F[nf]);
      for (int i = 0; i < B[l].n; ++i) ptrs[nf][i] = B[l].a[i] + cpt;
      F[nf].location = loc[l];
      F[nf].data = (B[l].n > 0) ? ptrs[nf] : MFTL_NULL;
      F[nf].stride = S->ncomp;
      F[nf].version = 7;
      F[nf].halo_valid = S->halo_valid;
    }
  double *U = (double *) malloc((size_t) (nm > 0 ? nm * nf : 1) * sizeof(double));
  st = MFTLCouplerInterpolate(c, b, nf, F, U);
  if (st != MFTL_COUPLING_OK) res->status = st;
  for (int64_t i = 0; i < nm && st == MFTL_COUPLING_OK; ++i)
    for (int f = 0; f < nf; ++f)
    {
      const int l = f / S->ncomp, cpt = f % S->ncomp;
      const affine *A = &Af[l][cpt];
      /* the frame position is the one the row was built at (the function is constant on periodic axes) */
      double scale = fabs(A->a);
      for (int k = 0; k < 3; ++k) scale += fabs(A->g[k]) * (fabs(S->lat[l].gmin[k]) + S->lat[l].L[k]);
      res->worst_affine = fmax(res->worst_affine, fabs(U[i * nf + f] - f_affine(&bpos[3 * i], (void *) A)) / scale);
    }

  /* ── spread: every field gets a random load per marker; ghosts and padding must stay untouched ── */
  double *Ld = (double *) malloc((size_t) (nm > 0 ? nm * nf : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i)
    for (int f = 0; f < nf; ++f) Ld[i * nf + f] = hash01(M.keys[i] * 31u + (uint64_t) f);
  for (int l = 0; l < S->n_loc; ++l) adv_set(&B[l], S->ncomp, 0.0, -777.0);
  st = MFTLCouplerSpread(c, b, nf, F, Ld, MFTL_SPREAD_INTEGRATED);
  if (st != MFTL_COUPLING_OK) res->status = st;
  for (int l = 0; l < S->n_loc; ++l)
    if (!adv_untouched(&B[l], S->ncomp, -777.0)) res->worst_spread = fmax(res->worst_spread, 1.0);
  const int stride = 3 + nf;
  double *mine = (double *) malloc((size_t) (nm > 0 ? nm * stride : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i)
  {
    memcpy(&mine[stride * i], &bpos[3 * i], 3 * sizeof(double));
    for (int f = 0; f < nf; ++f) mine[stride * i + 3 + f] = Ld[i * nf + f];
  }
  int tot = 0;
  double *all = allgather_doubles(mine, (int) nm, stride, &tot);
  for (int f = 0; f < nf; ++f) adv_check_spread_field(S, f / S->ncomp, f % S->ncomp, f, &B[f / S->ncomp], all, tot, stride, res);

  /* ── power identity with random fields: U.F == u.b ── */
  {
    double s[4] = {0, 0, 0, 0};
    adv_blocks Bu[2];
    MFTL_SolverField Fu[6];
    double *pu[6][ADV_MAX_BOXES];
    for (int f = 0; f < nf; ++f)
    {
      const int l = f / S->ncomp, cpt = f % S->ncomp;
      if (cpt == 0)
      {
        Bu[l] = B[l];
        for (int i = 0; i < B[l].n; ++i) Bu[l].a[i] = (double *) malloc((size_t) B[l].n_elem[i] * (size_t) S->ncomp * sizeof(double));
      }
      rnd_ctx rc = {{S->lat[l].N[0], S->lat[l].N[1], S->lat[l].N[2]}, 1234u + (uint64_t) (7 * l + cpt)};
      adv_fill(&Bu[l], &S->lat[l], cpt, f_random, &rc, S->halo_valid);
      Fu[f] = F[f];
      for (int i = 0; i < B[l].n; ++i) pu[f][i] = Bu[l].a[i] + cpt;
      Fu[f].data = (B[l].n > 0) ? pu[f] : MFTL_NULL;
      Fu[f].version = 8;
    }
    if (MFTLCouplerInterpolate(c, b, nf, Fu, U) != MFTL_COUPLING_OK) res->status = MFTL_COUPLING_INVALID;
    for (int64_t i = 0; i < nm * nf; ++i)
    {
      s[0] += U[i] * Ld[i];
      s[1] += fabs(U[i] * Ld[i]);
    }
    for (int l = 0; l < S->n_loc; ++l) adv_owned_dot(&Bu[l], &B[l], S->ncomp, &s[2], &s[3]);
    MPI_Allreduce(MPI_IN_PLACE, s, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    res->power_rel = fabs(s[0] - s[2]) / fmax(fmax(s[1], s[3]), 1e-300);
    for (int l = 0; l < S->n_loc; ++l)
      for (int i = 0; i < B[l].n; ++i) free(Bu[l].a[i]);
  }

  free(all);
  free(mine);
  free(Ld);
  free(U);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&win);
  MFTLCouplerDestroy(&c);
  for (int l = 0; l < S->n_loc; ++l) adv_free_blocks(&B[l]);
  adv_markers_free(&M);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
}

/** @brief Reduces a result and judges it; prints the offenders on rank 0. */
static int adv_judge(const char *what, adv_result *r)
{
  long cnt[3] = {r->rows, r->bad_rows, (long) r->status};
  MPI_Allreduce(MPI_IN_PLACE, cnt, 3, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  double w[4] = {r->worst_row, r->worst_affine, r->worst_spread, r->power_rel};
  MPI_Allreduce(MPI_IN_PLACE, w, 4, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  long bad = 0;
  MPI_Allreduce(&r->bad_rows, &bad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  const int ok = cnt[2] == MFTL_COUPLING_OK && bad == 0 && w[0] <= 1e-11 && w[1] <= 1e-12 && w[2] <= 1e-12 && w[3] <= 1e-13;
  if (!ok && g_rank == 0)
    printf("\n  %s: status %s, %ld bad rows, row %.3e, affine %.3e, spread %.3e, power %.3e", what, MFTLCouplingStatusName((int) cnt[2]), bad, w[0], w[1],
           w[2], w[3]);
  return ok;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Irregular decompositions of anisotropic, offset lattices, with a cell-
 *        and a node-centred location bound together, give the brute-force rows,
 *        exact affine interpolation, the brute-force transpose and the power
 *        identity -- for every periodicity pattern, several seeds, and with
 *        and without valid ghosts. Markers include the seam, bin and rank
 *        boundaries and positions several periods away.
 */
static int test_irregular_decompositions_match_brute_force(void)
{
  int ok = 1;
  long halo_seen = 0, remote_seen = 0;
  const bool pats[4][3] = {{true, true, true}, {false, false, false}, {true, false, true}, {false, true, false}};
  for (int pat = 0; pat < 4; ++pat)
    for (int variant = 0; variant < 2; ++variant)
    {
      adv_scenario S;
      memset(&S, 0, sizeof S);
      S.n_loc = 2;
      S.lat[0] = adv_default_lattice(true, pats[pat]);
      S.lat[1] = adv_default_lattice(false, pats[pat]);
      MFTLCouplerDefaultParams(&S.P);
      S.n_boxes = 3 * g_nproc + 2 + variant;
      S.seed = 1000u + 17u * (uint64_t) pat + (uint64_t) variant;
      S.max_ghost = 3;
      S.halo_valid = (variant == 0);
      S.ncomp = 1;
      S.n_random = 40;
      S.wall_gap = 0.0;
      S.skin = (variant == 0) ? 0.0 : 0.05;
      adv_result r;
      adv_run(&S, &r);
      char what[64];
      snprintf(what, sizeof what, "pattern %d variant %d", pat, variant);
      if (!adv_judge(what, &r)) ok = 0;
      halo_seen += r.halo;
      remote_seen += r.remote;
    }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* non-vacuity: with several ranks, both the halo and the remote routes ran */
  if (g_nproc > 1)
  {
    TEST_ASSERT(anywhere(halo_seen > 0, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(remote_seen > 0, MPI_COMM_WORLD));
  }
  return 0;
}

/**
 * @brief Every built-in kernel, and support growth, through the same pipeline:
 *        a corrected cosine row, the radial Wendland kernel, a C0 hat, Roma's
 *        1.5-wide kernel and a scaled B-spline, all against brute force.
 */
static int test_every_kernel_through_the_pipeline(void)
{
  int ok = 1;
  const MFTL_CouplingKernel ks[6] = {MFTL_KERNEL_COSINE4, MFTL_KERNEL_WENDLAND_C2, MFTL_KERNEL_HAT2, MFTL_KERNEL_ROMA3, MFTL_KERNEL_BSPLINE3,
                                     MFTL_KERNEL_PESKIN4};
  const bool per[3] = {true, false, true};
  for (int i = 0; i < 6; ++i)
  {
    adv_scenario S;
    memset(&S, 0, sizeof S);
    S.n_loc = 1;
    S.lat[0] = adv_default_lattice(true, per);
    MFTLCouplerDefaultParams(&S.P);
    S.P.kernel = ks[i];
    if (ks[i] == MFTL_KERNEL_BSPLINE3) S.P.support_scale = 1.3;
    if (ks[i] == MFTL_KERNEL_PESKIN4)
    {
      /* growth is armed: every row is built within the grown window */
      S.P.failure = MFTL_STENCIL_GROW;
      S.P.max_growth = 1.4;
    }
    if (ks[i] == MFTL_KERNEL_HAT2) S.P.max_amplification = 8.0;
    S.n_boxes = 2 * g_nproc + 1;
    S.seed = 77u + (uint64_t) i;
    S.max_ghost = 2;
    S.halo_valid = (i % 2 == 0);
    S.ncomp = 1;
    S.n_random = 30;
    /* cell-centred samples on the walled axis: a marker on the wall would extrapolate */
    S.wall_gap = 0.5 * S.lat[0].h[1];
    S.skin = 0.01;
    adv_result r;
    adv_run(&S, &r);
    char what[64];
    snprintf(what, sizeof what, "kernel %d", (int) ks[i]);
    if (!adv_judge(what, &r)) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief Interleaved components (an AoS velocity array, stride 3) give the same
 *        bits as three separate arrays, in both directions.
 */
static int test_interleaved_components_through_the_stride(void)
{
  adv_scenario S;
  memset(&S, 0, sizeof S);
  const bool per[3] = {true, true, false};
  S.n_loc = 1;
  S.lat[0] = adv_default_lattice(true, per);
  MFTLCouplerDefaultParams(&S.P);
  S.n_boxes = 2 * g_nproc + 3;
  S.seed = 4242u;
  S.max_ghost = 2;
  S.halo_valid = true;
  S.ncomp = 3;
  S.n_random = 30;
  S.wall_gap = 0.5 * S.lat[0].h[2];
  S.skin = 0.0;
  adv_result r;
  adv_run(&S, &r);
  TEST_ASSERT(agreed(adv_judge("stride 3", &r), MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief A rank holding more ghosted blocks than the halo route tracks (32) is
 *        still exact: the blocks past the limit are fetched instead of read in
 *        place, never read from a ghost layer the coupler did not account for.
 */
static int test_more_ghosted_blocks_than_the_halo_route_tracks(void)
{
  adv_scenario S;
  memset(&S, 0, sizeof S);
  const bool per[3] = {true, true, true};
  S.n_loc = 1;
  S.lat[0] = adv_default_lattice(true, per);
  MFTLCouplerDefaultParams(&S.P);
  S.n_boxes = 40 * g_nproc;
  S.seed = 31337u;
  S.max_ghost = 3;
  S.halo_valid = true;
  S.ncomp = 1;
  S.n_random = 25;
  S.skin = 0.02;
  adv_result r;
  adv_run(&S, &r);
  TEST_ASSERT(agreed(adv_judge("many blocks", &r), MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief Two couplers on one ensemble interleave their split-phase transfers:
 *        A begins, B runs a whole fetch and spread, A ends. Each communicates on
 *        its own context, so neither sees the other's messages.
 */
static int test_two_couplers_interleave(void)
{
  const bool per[3] = {true, true, true};
  adv_lat A = adv_default_lattice(true, per);
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g1 = MFTLSolverGridCreate(MPI_COMM_WORLD), *g2 = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int l1 = adv_add_location(g1, "a", &A, MFTL_BOUNDARY_TRUNCATE);
  const int l2 = adv_add_location(g2, "b", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B1, B2;
  adv_decompose(g1, l1, &A, 2 * g_nproc, 5u, 0, 1, false, &B1);
  adv_decompose(g2, l2, &A, 3 * g_nproc, 6u, 1, 1, false, &B2);
  adv_markers M;
  adv_markers_make(&ens->domain, &A, 2.0 * A.h[0], 20, 9u, 0.0, &M);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = M.n;
  ms.user_pos = M.pos;
  ms.user_keys = M.keys;
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_Coupler *c[2] = {MFTL_NULL, MFTL_NULL};
  MFTL_CouplingWindow *w[2] = {MFTL_NULL, MFTL_NULL};
  MFTL_CouplingBinding *b[2] = {MFTL_NULL, MFTL_NULL};
  MFTL_SolverGrid *gs[2] = {g1, g2};
  for (int i = 0; i < 2; ++i)
  {
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, gs[i], MFTL_NULL, &c[i]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SupportRequest *req = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(c[i], &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c[i], req, &w[i]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    TEST_ASSERT(agreed(MFTLCouplerBind(c[i], &ms, w[i], &b[i]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  }
  rnd_ctx rc = {{A.N[0], A.N[1], A.N[2]}, 5u};
  adv_fill(&B1, &A, 0, f_random, &rc, false);
  adv_fill(&B2, &A, 0, f_random, &rc, true);
  MFTL_SolverField F1 = adv_field(&B1, l1, 0, 1, false);
  double *p2[ADV_MAX_BOXES];
  for (int i = 0; i < B2.n; ++i) p2[i] = B2.a[i];
  MFTL_SolverField F2 = F1;
  F2.location = l2;
  F2.data = (B2.n > 0) ? p2 : MFTL_NULL;
  F2.halo_valid = true;
  const int64_t nm = M.n;
  double *U1 = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double)), *U2 = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  double *V1 = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  /* the reference: A alone */
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c[0], b[0], 1, &F1, V1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* A begins ... */
  MFTL_TransferSpec ts = {MFTL_TRANSFER_FETCH, w[0], MFTL_NULL, 1, &F1, MFTL_SPREAD_INTEGRATED};
  MFTL_TransferReady *ra = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(c[0], &ts, &ra) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetchBegin(c[0], ra) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* ... B runs a whole fetch and a spread ... */
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c[1], b[1], 1, &F2, U2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *Ld = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i) Ld[i] = 1.0 + hash01(M.keys[i]);
  TEST_ASSERT(agreed(MFTLCouplerSpread(c[1], b[1], 1, &F2, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* ... A ends */
  MFTL_CouplingFetch *fa = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetchEnd(c[0], &fa) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(b[0], fa, U1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int same = 1;
  for (int64_t i = 0; i < nm; ++i)
    if (U1[i] != V1[i] || U2[i] != V1[i]) same = 0; /* both read the same random field through different decompositions */
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&fa);
  MFTLCouplerReleaseReady(&ra);
  for (int i = 0; i < 2; ++i)
  {
    MFTLCouplerReleaseBinding(&b[i]);
    MFTLCouplerReleaseWindow(&w[i]);
    MFTLCouplerDestroy(&c[i]);
  }
  free(U1);
  free(U2);
  free(V1);
  free(Ld);
  adv_markers_free(&M);
  adv_free_blocks(&B1);
  adv_free_blocks(&B2);
  MFTLSolverGridDestroy(&g1);
  MFTLSolverGridDestroy(&g2);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Truncation: walls, the lattice's own edge, and solids
 * ═════════════════════════════════════════════════════════════════════════════ */

static bool solid_cube(const int64_t idx[3]) { return idx[0] >= 6 && idx[0] < 10 && idx[1] >= 6 && idx[1] < 10 && idx[2] >= 6 && idx[2] < 10; }

/** @brief Binds one marker list against one location; returns the agreed status and the job-wide truncation count and affine error. */
static int adv_truncation_case(MFTL_Ensemble *ens, const adv_lat *A, MFTL_BoundaryPolicy bnd, adv_solid_fn solid, int n_all, const double *all, long *n_trunc,
                               double *affine_err)
{
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", A, bnd);
  adv_blocks B;
  adv_decompose_on(g_rank, g_nproc, g, loc, A, 2 * g_nproc + 1, 99u, 1, 1, false, solid, &B);
  affine Af = {0.5, {1.0, -2.0, 0.75}};
  adv_fill(&B, A, 0, adv_affine_fn, &Af, true);
  MFTL_Coupler *c = MFTL_NULL;
  int st = MFTLCouplerCreate(ens, g, MFTL_NULL, &c);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  if (st == MFTL_COUPLING_OK) st = adv_bind_list(c, n_all, all, 1, &loc, &w, &b, &n);
  *n_trunc = 0;
  *affine_err = 0.0;
  if (st == MFTL_COUPLING_OK)
  {
    MFTL_BindingQuality q;
    MFTLCouplerBindingQuality(b, &q);
    *n_trunc = (long) q.n_truncated;
    MFTL_SolverField F = adv_field(&B, loc, 0, 1, true);
    double *U = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
    st = MFTLCouplerInterpolate(c, b, 1, &F, U);
    const double *bp = MFTLCouplerBindingPositions(b);
    for (int64_t i = 0; i < n; ++i) *affine_err = fmax(*affine_err, fabs(U[i] - f_affine(&bp[3 * i], &Af)));
    free(U);
  }
  MPI_Allreduce(MPI_IN_PLACE, n_trunc, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, affine_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  return st;
}

/**
 * @brief A row is TRUNCATED exactly when a lattice position with weight is
 *        missing from it -- past a wall, past the edge of a lattice smaller than
 *        the domain, or on a masked (solid) cell -- and the boundary policy
 *        judges all three alike. A support that merely crosses a wall with no
 *        position beyond it is complete, and is not refused.
 */
static int test_truncation_is_counted_exactly(void)
{
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, false);
  const double h = 1.0 / 16.0;
  adv_lat full;
  memset(&full, 0, sizeof full);
  for (int k = 0; k < 3; ++k)
  {
    full.N[k] = 16;
    full.h[k] = h;
    full.origin[k] = 0.5 * h;
    full.L[k] = 1.0;
  }
  long nt;
  double ae;

  /* 1. a lattice covering only x < 1/2: a support across its edge is truncated */
  adv_lat half = full;
  half.N[0] = 8;
  {
    const double edge[3] = {0.5 - 0.5 * h, 0.5, 0.5}, inner[3] = {0.25, 0.5, 0.5};
    TEST_ASSERT(agreed(adv_truncation_case(ens, &half, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, edge, &nt, &ae) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &half, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, inner, &nt, &ae) == MFTL_COUPLING_OK && nt == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &half, MFTL_BOUNDARY_TRUNCATE, MFTL_NULL, 1, edge, &nt, &ae) == MFTL_COUPLING_OK && nt == 1 && ae <= 1e-12,
                       MPI_COMM_WORLD));
  }

  /* 2. a solid cube of masked cells: a row reaching into it is truncated */
  {
    const double next[3] = {10.5 * h, 8.0 * h, 8.0 * h}, far[3] = {0.2, 0.2, 0.8};
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_ERROR, solid_cube, 1, next, &nt, &ae) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_ERROR, solid_cube, 1, far, &nt, &ae) == MFTL_COUPLING_OK && nt == 0, MPI_COMM_WORLD));
    const double both[6] = {10.5 * h, 8.0 * h, 8.0 * h, 0.2, 0.2, 0.8};
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_TRUNCATE, solid_cube, 2, both, &nt, &ae) == MFTL_COUPLING_OK && nt == 1 && ae <= 1e-12,
                       MPI_COMM_WORLD));
  }

  /* 3. the wall, to the half cell: cell centres at h/2, 3h/2, ...; Peskin reaches 2h.
   *    At 1.6h the nearest position beyond the wall (-h/2) is 2.1h away: complete.
   *    At 1.5h it is exactly 2h away, where the kernel is zero: complete.
   *    At 1.4h it is 1.9h away: truncated. */
  {
    const double a[3] = {1.6 * h, 0.5, 0.5}, bq[3] = {1.5 * h, 0.5, 0.5}, cq[3] = {1.4 * h, 0.5, 0.5};
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, a, &nt, &ae) == MFTL_COUPLING_OK && nt == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, bq, &nt, &ae) == MFTL_COUPLING_OK && nt == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &full, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, cq, &nt, &ae) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
  }

  /* 4. a node lattice with samples ON the walls: a marker on the wall misses the node beyond it */
  {
    adv_lat nodes = full;
    for (int k = 0; k < 3; ++k)
    {
      nodes.N[k] = 17;
      nodes.origin[k] = 0.0;
    }
    const double onwall[3] = {0.0, 0.5, 0.5}, inside[3] = {2.0 * h, 0.5, 0.5};
    TEST_ASSERT(agreed(adv_truncation_case(ens, &nodes, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, onwall, &nt, &ae) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &nodes, MFTL_BOUNDARY_ERROR, MFTL_NULL, 1, inside, &nt, &ae) == MFTL_COUPLING_OK && nt == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(adv_truncation_case(ens, &nodes, MFTL_BOUNDARY_TRUNCATE, MFTL_NULL, 1, onwall, &nt, &ae) == MFTL_COUPLING_OK && nt == 1 && ae <= 1e-12,
                       MPI_COMM_WORLD));
  }
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Loads
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A non-finite load on ONE rank is refused on every rank before any
 *        target is written: a NaN spread into the solver would poison the
 *        owner's sample and, through its next halo exchange, its neighbours.
 */
static int test_a_non_finite_load_is_refused_everywhere(void)
{
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, true);
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 12;
    A.h[k] = 1.0 / 12.0;
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = true;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc, 3u, 1, 2, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double all[3 * 24];
  uint64_t s = 11u;
  for (int i = 0; i < 24; ++i)
    for (int k = 0; k < 3; ++k) all[3 * i + k] = adv_unif(&s);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, 24, all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  for (int i = 0; i < B.n; ++i)
    for (int64_t e = 0; e < B.n_elem[i] * 2; ++e) B.a[i][e] = 3.0;
  MFTL_SolverField F[2] = {adv_field(&B, loc, 0, 1, false), adv_field(&B, loc, 0, 1, false)};
  double *ptr1[ADV_MAX_BOXES];
  for (int i = 0; i < B.n; ++i) ptr1[i] = B.a[i] + 1;
  F[1].data = (B.n > 0) ? ptr1 : MFTL_NULL;
  double *L = (double *) malloc((size_t) (2 * n > 0 ? 2 * n : 1) * sizeof(double));
  const double bad[3] = {NAN, INFINITY, -INFINITY};
  for (int t = 0; t < 3; ++t)
  {
    for (int64_t i = 0; i < 2 * n; ++i) L[i] = 1.0;
    /* the last rank holding a marker poisons its last load of the second field */
    int holder = (n > 0) ? g_rank : -1;
    MPI_Allreduce(MPI_IN_PLACE, &holder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == holder) L[2 * n - 1] = bad[t];
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    int untouched = 1;
    for (int i = 0; i < B.n; ++i)
      for (int64_t e = 0; e < B.n_elem[i] * 2; ++e)
        if (B.a[i][e] != 3.0) untouched = 0;
    TEST_ASSERT(agreed(untouched, MPI_COMM_WORLD));
  }
  /* and the same loads, finite, go through */
  for (int64_t i = 0; i < 2 * n; ++i) L[i] = 1.0;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  free(L);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Markers at the SAME position -- on one rank and on two -- get the same
 *        row to the bit, and a load split between two of them spreads like the
 *        whole load on one.
 */
static int test_coincident_markers(void)
{
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, true);
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 10 + k;
    A.h[k] = 1.0 / (double) A.N[k];
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = true;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc + 1, 8u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* markers 0..3 at one point (round-robin over ranks), 4 elsewhere */
  const double p[3] = {0.313, 0.777, 0.0421};
  double all[3 * 5];
  for (int i = 0; i < 4; ++i) memcpy(&all[3 * i], p, sizeof p);
  all[12] = 0.6;
  all[13] = 0.1;
  all[14] = 0.9;
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, 5, all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* every rank's copy of the coincident marker's row, gathered and compared */
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  enum
  {
    ROWCAP = 4096
  };
  double *mine = (double *) malloc(2 * ROWCAP * sizeof(double));
  int nm = 0;
  for (int64_t i = 0; i < n; ++i)
    if (ref[i].key < 4)
    {
      int ne = 0;
      const uint64_t *keys;
      const double *wts;
      MFTLCouplerBindingRow(b, 0, i, &ne, &keys, &wts);
      for (int e = 0; e < ne && nm < ROWCAP; ++e)
      {
        memcpy(&mine[2 * nm], &keys[e], sizeof(double));
        mine[2 * nm + 1] = wts[e];
        ++nm;
      }
    }
  int tot = 0;
  double *rows = allgather_doubles(mine, nm, 2, &tot);
  free(mine);
  /* four copies of one row: entries come in four identical runs */
  int ok = (tot % 4 == 0);
  const int len = tot / 4;
  for (int r = 1; r < 4 && ok; ++r)
    for (int e = 0; e < len && ok; ++e)
      if (memcmp(&rows[2 * e], &rows[2 * (r * len + e)], 2 * sizeof(double)) != 0) ok = 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  free(rows);
  /* a load split over the four copies spreads like the whole load on one */
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
  double L[64], tgt_split = 0.0, tgt_whole = 0.0, worst = 0.0;
  double *first[ADV_MAX_BOXES];
  for (int pass = 0; pass < 2; ++pass)
  {
    for (int i = 0; i < B.n; ++i)
      for (int64_t e = 0; e < B.n_elem[i]; ++e) B.a[i][e] = 0.0;
    for (int64_t i = 0; i < n; ++i)
      L[i] = (ref[i].key == 4) ? 0.5 : (pass == 0 ? 0.25 : (ref[i].key == 0 ? 1.0 : 0.0));
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double sum = 0.0;
    for (int i = 0; i < B.n; ++i)
    {
      if (pass == 0) first[i] = (double *) malloc((size_t) B.n_elem[i] * sizeof(double));
      for (int64_t e = 0; e < B.n_elem[i]; ++e)
      {
        sum += B.a[i][e];
        if (pass == 0)
          first[i][e] = B.a[i][e];
        else
          worst = fmax(worst, fabs(first[i][e] - B.a[i][e]));
      }
    }
    if (pass == 0)
      tgt_split = sum;
    else
      tgt_whole = sum;
  }
  for (int i = 0; i < B.n; ++i) free(first[i]);
  /* sample by sample: the split load lands where the whole one does */
  TEST_ASSERT(agreed(worst <= 1e-15, MPI_COMM_WORLD));
  double s2[2] = {tgt_split, tgt_whole};
  MPI_Allreduce(MPI_IN_PLACE, s2, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  /* force conservation, both ways: 1 + 0.5 */
  TEST_ASSERT(agreed(fabs(s2[0] - 1.5) <= 1e-13 && fabs(s2[1] - 1.5) <= 1e-13, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * A stretched periodic lattice whose cells straddle the seam
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Width of sample j along axis k of a rectilinear test lattice. */
static double adv_width(const adv_lat *A, int k, int64_t j) { return A->faces[k][j + 1] - A->faces[k][j]; }

/**
 * @brief The reference row on a rectilinear lattice: the Shepard metric of the
 *        sample widths over the pilot neighbourhood (the test's own sum over
 *        every lattice image), then every image in the support, through the
 *        row core.
 */
static int adv_ref_row_rect(const adv_lat *A, const MFTL_CouplerParams *P, const double hmax[3], const double x[3], uint64_t *keys, double *w, int cap,
                            double H[3])
{
  (void) hmax;
  /* index ranges wide enough for the pilot and for the support (both < half a period) */
  int64_t a[3], b[3];
  for (int k = 0; k < 3; ++k)
  {
    a[k] = A->per[k] ? -A->N[k] : 0;
    b[k] = A->per[k] ? 2 * A->N[k] : A->N[k];
  }
  double sw = 0.0, sh[3] = {0, 0, 0};
  for (int64_t i = a[0]; i < b[0]; ++i)
    for (int64_t j = a[1]; j < b[1]; ++j)
      for (int64_t l = a[2]; l < b[2]; ++l)
      {
        const int64_t idx[3] = {i, j, l};
        double rho2 = 0.0;
        bool in = true;
        double hj[3];
        for (int k = 0; k < 3; ++k)
        {
          hj[k] = adv_width(A, k, A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k]);
          const double t = (adv_pos(A, k, idx[k]) - x[k]) / (P->pilot_factor * hj[k]);
          if (fabs(t) > 1.0) in = false;
          rho2 += t * t;
          hj[k] = adv_width(A, k, A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k]);
        }
        if (!in || rho2 >= 1.0) continue;
        const double rho = sqrt(rho2), t = 1.0 - rho, wt = t * t * t * t * (4.0 * rho + 1.0);
        sw += wt;
        for (int k = 0; k < 3; ++k) sh[k] += wt * hj[k];
      }
  if (!(sw > 0.0)) return -1;
  for (int k = 0; k < 3; ++k) H[k] = sh[k] / sw;
  MFTL_KernelInfo ki;
  MFTLCouplingKernelInfo(P->kernel, &ki);
  int n = 0;
  double *X = (double *) malloc(3 * (size_t) cap * sizeof(double)), *q = (double *) malloc((size_t) cap * sizeof(double));
  for (int64_t i = a[0]; i < b[0]; ++i)
    for (int64_t j = a[1]; j < b[1]; ++j)
      for (int64_t l = a[2]; l < b[2]; ++l)
      {
        const int64_t idx[3] = {i, j, l};
        double p[3], m = 1.0;
        int64_t wr[3];
        bool in = true;
        for (int k = 0; k < 3; ++k)
        {
          p[k] = adv_pos(A, k, idx[k]);
          if (fabs(p[k] - x[k]) > ki.half_width * P->support_scale * H[k]) in = false;
          wr[k] = A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k];
          m *= adv_width(A, k, wr[k]);
        }
        if (!in || n >= cap) continue;
        keys[n] = (uint64_t) (wr[0] + A->N[0] * (wr[1] + A->N[1] * wr[2]));
        memcpy(&X[3 * n], p, sizeof p);
        q[n] = m;
        ++n;
      }
  const int st = MFTLCouplingAssembleRow(P, x, H, n, X, q, w, MFTL_NULL);
  free(X);
  free(q);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

/**
 * @brief A STRETCHED, PERIODIC rectilinear lattice whose first face is not on
 *        the domain's lower bound -- so one cell straddles the seam and its
 *        sample sits just below max -- plus a walled axis whose samples are
 *        off-centre in their cells and a periodic one of equal widths. Rows equal
 *        the brute-force rows with the test's own Shepard metric, the spread
 *        is their transpose, and an affine field along the walled axis is
 *        reproduced, on an irregular decomposition with ghost reads.
 */
static int test_a_stretched_periodic_lattice_straddling_the_seam(void)
{
  MFTL_Ensemble *ens = adv_unit_box(MPI_COMM_WORLD, false);
  /* replace the walled box by one periodic in x and z */
  MFTLEnsembleDestroy(&ens);
  {
    int cuts[3];
    test_grid_for(g_nproc, cuts);
    double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    bool per[3] = {true, false, true};
    ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
  }
  const int64_t N[3] = {10, 8, 9};
  double *fc[3], *cc[3];
  for (int k = 0; k < 3; ++k)
  {
    fc[k] = (double *) malloc((size_t) (N[k] + 1) * sizeof(double));
    cc[k] = (double *) malloc((size_t) N[k] * sizeof(double));
  }
  for (int64_t i = 0; i <= N[0]; ++i)
  {
    const double t = (double) i / (double) N[0];
    fc[0][i] = 0.03 + t - 0.1 * sin(2.0 * MFTL_PI * t) / (2.0 * MFTL_PI);
  }
  fc[0][0] = 0.03;
  fc[0][N[0]] = 1.03;
  for (int64_t i = 0; i < N[0]; ++i) cc[0][i] = 0.5 * (fc[0][i] + fc[0][i + 1]);
  for (int64_t i = 0; i <= N[1]; ++i)
  {
    const double t = (double) i / (double) N[1];
    fc[1][i] = t - 0.12 * sin(2.0 * MFTL_PI * t) / (2.0 * MFTL_PI);
  }
  fc[1][0] = 0.0;
  fc[1][N[1]] = 1.0;
  for (int64_t i = 0; i < N[1]; ++i) cc[1][i] = fc[1][i] + 0.3 * (fc[1][i + 1] - fc[1][i]);
  for (int64_t i = 0; i <= N[2]; ++i) fc[2][i] = (double) i / (double) N[2];
  for (int64_t i = 0; i < N[2]; ++i) cc[2][i] = 0.5 * (fc[2][i] + fc[2][i + 1]);
  adv_lat A;
  memset(&A, 0, sizeof A);
  double hmax[3] = {0, 0, 0};
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = N[k];
    A.L[k] = 1.0;
    A.per[k] = (k != 1);
    A.coords[k] = cc[k];
    A.faces[k] = fc[k];
    A.h[k] = 1.0 / (double) N[k];
    for (int64_t i = 0; i < N[k]; ++i) hmax[k] = fmax(hmax[k], fc[k][i + 1] - fc[k][i]);
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "stretched", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc + 2, 2027u, 2, 1, false, &B);
  affine Af = {0.7, {0.0, 1.25, 0.0}};
  adv_fill(&B, &A, 0, adv_affine_fn, &Af, true);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* markers: random, on the seam, on the straddling sample, on a sample, off the walls by a cell */
  double all[3 * 48];
  int na = 0;
  uint64_t s = 606u;
  for (int i = 0; i < 36; ++i, ++na)
    for (int k = 0; k < 3; ++k) all[3 * na + k] = (k == 1) ? 0.2 + 0.6 * adv_unif(&s) : adv_unif(&s);
  const double sp[12][3] = {{0.0, 0.5, 0.5},           {nextafter(0.0, -1.0), 0.5, 0.5}, {0.03, 0.4, 0.2},  {cc[0][9], 0.5, 0.0},
                            {cc[0][9] - 1.0, 0.6, 0.9}, {nextafter(1.0, 0.0), 0.45, 0.4}, {cc[0][3], cc[1][4], cc[2][5]}, {0.5, 0.5, 1.0},
                            {0.98, 0.3, 0.999},         {0.031, 0.7, 0.001},              {0.5, cc[1][2], 0.5},             {2.03, 0.5, -3.5}};
  for (int i = 0; i < 12; ++i, ++na) memcpy(&all[3 * na], sp[i], sizeof sp[i]);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, na, all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *bp = MFTLCouplerBindingPositions(b);
  enum
  {
    CAP = 4096
  };
  uint64_t *rk = (uint64_t *) malloc(CAP * sizeof(uint64_t));
  double *rw = (double *) malloc(CAP * sizeof(double));
  double worst = 0.0, H[3];
  int bad = 0;
  for (int64_t i = 0; i < n; ++i)
  {
    int ne = 0;
    const uint64_t *keys;
    const double *wts;
    MFTLCouplerBindingRow(b, 0, i, &ne, &keys, &wts);
    uint64_t kc[CAP];
    double wc[CAP];
    memcpy(kc, keys, (size_t) ne * sizeof(uint64_t));
    memcpy(wc, wts, (size_t) ne * sizeof(double));
    const int nr = adv_ref_row_rect(&A, &P, hmax, &bp[3 * i], rk, rw, CAP, H);
    if (nr < 0)
    {
      ++bad;
      continue;
    }
    worst = fmax(worst, adv_row_diff(ne, kc, wc, nr, rk, rw));
  }
  if (bad || worst > 1e-11) printf("\n  rank %d: %d bad rows, worst %.3e", g_rank, bad, worst);
  TEST_ASSERT(agreed(bad == 0 && worst <= 1e-11, MPI_COMM_WORLD));
  /* the affine field along the walled axis */
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, true);
  double *U = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double aerr = 0.0;
  for (int64_t i = 0; i < n; ++i) aerr = fmax(aerr, fabs(U[i] - f_affine(&bp[3 * i], &Af)));
  TEST_ASSERT(agreed(aerr <= 1e-12, MPI_COMM_WORLD));
  /* the spread, against the gathered brute-force transpose */
  double *Ld = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < n; ++i) Ld[i] = hash01(ref[i].key + 5u);
  for (int i = 0; i < B.n; ++i)
    for (int64_t e = 0; e < B.n_elem[i]; ++e) B.a[i][e] = 0.0;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *mine = (double *) malloc((size_t) (n > 0 ? 4 * n : 1) * sizeof(double));
  for (int64_t i = 0; i < n; ++i)
  {
    memcpy(&mine[4 * i], &bp[3 * i], 3 * sizeof(double));
    mine[4 * i + 3] = Ld[i];
  }
  int tot = 0;
  double *gat = allgather_doubles(mine, (int) n, 4, &tot);
  double sworst = 0.0;
  for (int i = 0; i < B.n; ++i)
  {
    double *ex = (double *) calloc((size_t) B.n_elem[i], sizeof(double)), *sc = (double *) calloc((size_t) B.n_elem[i], sizeof(double));
    for (int m = 0; m < tot; ++m)
    {
      const int nr = adv_ref_row_rect(&A, &P, hmax, &gat[4 * m], rk, rw, CAP, H);
      double ra = 0.0;
      for (int e = 0; e < nr; ++e) ra += fabs(rw[e]);
      for (int e = 0; e < nr; ++e)
      {
        const int64_t idx[3] = {(int64_t) (rk[e] % (uint64_t) N[0]), (int64_t) ((rk[e] / (uint64_t) N[0]) % (uint64_t) N[1]),
                                (int64_t) (rk[e] / (uint64_t) (N[0] * N[1]))};
        bool own = true;
        for (int k = 0; k < 3; ++k)
          if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) own = false;
        if (!own) continue;
        const int64_t el = (idx[0] - B.alo[i][0]) + B.adims[i][0] * ((idx[1] - B.alo[i][1]) + B.adims[i][1] * (idx[2] - B.alo[i][2]));
        ex[el] += rw[e] * gat[4 * m + 3];
        sc[el] += ra * fabs(gat[4 * m + 3]);
      }
    }
    for (int64_t e = 0; e < B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&B, i, e, idx);
      bool own = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) own = false;
      if (own) sworst = fmax(sworst, fabs(B.a[i][e] - ex[e]) / fmax(sc[e], 1e-12));
    }
    free(ex);
    free(sc);
  }
  TEST_ASSERT(agreed(sworst <= 1e-12, MPI_COMM_WORLD));
  MFTL_BindingQuality q;
  MFTLCouplerBindingQuality(b, &q);
  TEST_ASSERT(anywhere(q.n_corrected > 0, MPI_COMM_WORLD));
  free(gat);
  free(mine);
  free(Ld);
  free(U);
  free(rk);
  free(rw);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  for (int k = 0; k < 3; ++k)
  {
    free(fc[k]);
    free(cc[k]);
  }
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * The partition reaches no bit
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The irregular scenario on @p comm: two locations, ghost reads, the
 *        special markers; interpolates a random field. Returns (key, value)
 *        pairs, value index = 2 * marker + location.
 */
static int adv_interp_on(MPI_Comm comm, int64_t *n_out, uint64_t **keys, double **vals)
{
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  const bool per[3] = {true, false, true};
  adv_lat A[2] = {adv_default_lattice(true, per), adv_default_lattice(false, per)};
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3], hi[3];
  bool pp[3];
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = A[0].gmin[k];
    hi[k] = A[0].gmin[k] + A[0].L[k];
    pp[k] = per[k];
  }
  MFTL_Ensemble *ens = MFTLEnsembleCreate(comm, cuts, lo, hi, pp, 4);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(comm);
  int loc[2];
  adv_blocks B[2];
  for (int l = 0; l < 2; ++l)
  {
    loc[l] = adv_add_location(g, l ? "nodes" : "cells", &A[l], MFTL_BOUNDARY_TRUNCATE);
    adv_decompose_on(rank, np, g, loc[l], &A[l], 3 * np + 1, 7u + (uint64_t) l, 3, 1, false, MFTL_NULL, &B[l]);
    rnd_ctx rc = {{A[l].N[0], A[l].N[1], A[l].N[2]}, 31u + (uint64_t) l};
    adv_fill(&B[l], &A[l], 0, f_random, &rc, true);
  }
  adv_markers M;
  /* the rank boundaries of the LARGEST tested grid: the same markers at every count */
  const int cuts27[3] = {3, 3, 3};
  adv_markers_make_cuts(cuts27, &A[0], 2.0 * A[0].h[0], 50, 77u, 0.0, &M);
  free(M.pos);
  free(M.keys);
  adv_markers_take_on(rank, np, &M);
  MFTL_Coupler *c = MFTL_NULL;
  int st = MFTLCouplerCreate(ens, g, MFTL_NULL, &c);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = M.n;
  ms.user_pos = M.pos;
  ms.user_keys = M.keys;
  /* the binding lists the locations in reverse: rows and fields are matched by location, not position */
  const int rev[2] = {loc[1], loc[0]};
  ms.n_locations = 2;
  ms.locations = rev;
  MFTL_WindowSpec ws = {{0.03, 0.03, 0.03}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &w);
  MFTLCouplerReleaseRequest(&req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(c, &ms, w, &b);
  *n_out = M.n;
  *keys = (uint64_t *) malloc((size_t) (M.n > 0 ? M.n : 1) * sizeof(uint64_t));
  *vals = (double *) malloc((size_t) (M.n > 0 ? 2 * M.n : 1) * sizeof(double));
  if (st == MFTL_COUPLING_OK)
  {
    MFTL_SolverField F[2];
    double *p0[ADV_MAX_BOXES], *p1[ADV_MAX_BOXES];
    for (int i = 0; i < B[0].n; ++i) p0[i] = B[0].a[i];
    for (int i = 0; i < B[1].n; ++i) p1[i] = B[1].a[i];
    memset(F, 0, sizeof F);
    F[0].location = loc[0];
    F[0].data = B[0].n ? p0 : MFTL_NULL;
    F[0].halo_valid = true;
    F[1].location = loc[1];
    F[1].data = B[1].n ? p1 : MFTL_NULL;
    F[1].halo_valid = true;
    st = MFTLCouplerInterpolate(c, b, 2, F, *vals);
    const MFTL_MarkerRef *r = MFTLCouplerBindingMarkers(b);
    for (int64_t i = 0; i < M.n; ++i) (*keys)[i] = r[i].key;
  }
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  for (int l = 0; l < 2; ++l) adv_free_blocks(&B[l]);
  adv_markers_free(&M);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return st;
}

/**
 * @brief The irregular, two-location, ghost-reading scenario gives the SAME
 *        BITS at every process count as on one rank alone -- a stronger form of
 *        the claim test_coupling makes on aligned blocks.
 */
static int test_irregular_scenario_is_bitwise_partition_independent(void)
{
  int64_t n = 0;
  uint64_t *keys = MFTL_NULL;
  double *vals = MFTL_NULL;
  TEST_ASSERT(agreed(adv_interp_on(MPI_COMM_WORLD, &n, &keys, &vals) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *mine = (double *) malloc((size_t) (n > 0 ? 3 * n : 1) * sizeof(double));
  for (int64_t i = 0; i < n; ++i)
  {
    memcpy(&mine[3 * i], &keys[i], sizeof(double));
    mine[3 * i + 1] = vals[2 * i];
    mine[3 * i + 2] = vals[2 * i + 1];
  }
  int tot = 0;
  double *all = allgather_doubles(mine, (int) n, 3, &tot);
  int ok = 1;
  if (g_rank == 0)
  {
    int64_t ns = 0;
    uint64_t *ks = MFTL_NULL;
    double *vs = MFTL_NULL;
    const int st1 = adv_interp_on(MPI_COMM_SELF, &ns, &ks, &vs);
    ok = st1 == MFTL_COUPLING_OK && ns == tot;
    if (!ok) printf("\n  one rank: status %s, %lld markers against %d", MFTLCouplingStatusName(st1), (long long) ns, tot);
    for (int i = 0; i < tot && ok; ++i)
    {
      uint64_t k;
      memcpy(&k, &all[3 * i], sizeof k);
      int64_t j = -1;
      for (int64_t q = 0; q < ns; ++q)
        if (ks[q] == k) j = q;
      if (j < 0 || all[3 * i + 1] != vs[2 * j] || all[3 * i + 2] != vs[2 * j + 1])
      {
        printf("\n  marker %llu: %.17g %.17g vs one rank %.17g %.17g", (unsigned long long) k, all[3 * i + 1], all[3 * i + 2], j >= 0 ? vs[2 * j] : NAN,
               j >= 0 ? vs[2 * j + 1] : NAN);
        ok = 0;
      }
    }
    free(ks);
    free(vs);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  free(all);
  free(mine);
  free(keys);
  free(vals);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Point clouds registered awkwardly
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief One rank's share of the global jittered cloud, as two pieces with reversed, gapped array indices. */
typedef struct
{
  int loc;
  int64_t n[2];
  uint64_t *key[2];
  double *pos[2];
  int64_t *aidx[2];
  double *arr[2];  ///< [2 n + 1]: the samples at odd elements, gaps between
  int64_t len[2];
} adv_cloud;

static void adv_cloud_make(MFTL_SolverGrid *g, int64_t nc, double jitter, adv_cloud *C)
{
  memset(C, 0, sizeof *C);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "cloud";
  ld.lattice.kind = MFTL_LATTICE_NONE;
  C->loc = MFTLSolverGridAddLocation(g, &ld);
  const int64_t total = nc * nc * nc;
  const double H = 1.0 / (double) nc;
  for (int p = 0; p < 2; ++p)
  {
    C->key[p] = (uint64_t *) malloc((size_t) total * sizeof(uint64_t));
    C->pos[p] = (double *) malloc(3 * (size_t) total * sizeof(double));
  }
  for (int64_t c = 0; c < total; ++c)
  {
    double x[3];
    uint64_t key;
    point_of_cell(nc, jitter, c, x, &key);
    const int owner = (int) ((key / 3u) % (uint64_t) g_nproc);
    if (owner != g_rank) continue;
    const int p = (int) ((key / 7u) & 1u);
    C->key[p][C->n[p]] = key;
    memcpy(&C->pos[p][3 * C->n[p]], x, sizeof x);
    C->n[p]++;
  }
  for (int p = 0; p < 2; ++p)
  {
    const int64_t n = C->n[p];
    C->aidx[p] = (int64_t *) malloc((size_t) (n > 0 ? n : 1) * sizeof(int64_t));
    C->len[p] = 2 * n + 1;
    C->arr[p] = (double *) malloc((size_t) C->len[p] * sizeof(double));
    for (int64_t i = 0; i < n; ++i) C->aidx[p][i] = 2 * (n - 1 - i) + 1;
    double *meas = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double)), *ext = (double *) malloc(3 * (size_t) (n > 0 ? n : 1) * sizeof(double));
    for (int64_t i = 0; i < n; ++i)
    {
      meas[i] = H * H * H;
      for (int k = 0; k < 3; ++k) ext[3 * i + k] = 0.5 * H + 0.5 * jitter * H + 1e-9;
    }
    MFTL_PointSetDesc pd;
    memset(&pd, 0, sizeof pd);
    pd.location = C->loc;
    pd.n = n;
    pd.keys = C->key[p];
    pd.pos = C->pos[p];
    pd.measure = meas;
    for (int k = 0; k < 3; ++k) pd.h_uniform[k] = H;
    pd.extent = ext;
    pd.array_index = C->aidx[p];
    MFTLSolverGridAddPoints(g, &pd);
    free(meas);
    free(ext);
  }
}

static void adv_cloud_free(adv_cloud *C)
{
  for (int p = 0; p < 2; ++p)
  {
    free(C->key[p]);
    free(C->pos[p]);
    free(C->aidx[p]);
    free(C->arr[p]);
  }
}

/** @brief The reference row over the GLOBAL cloud (minimum image), through the row core. */
static int adv_ref_cloud_row(const MFTL_Domain *d, int64_t nc, double jitter, const MFTL_CouplerParams *P, const double x[3], uint64_t *keys, double *w,
                             int cap)
{
  const int64_t total = nc * nc * nc;
  const double H = 1.0 / (double) nc, Hv[3] = {H, H, H};
  double *X = (double *) malloc(3 * (size_t) cap * sizeof(double)), *q = (double *) malloc((size_t) cap * sizeof(double));
  int n = 0;
  for (int64_t c = 0; c < total; ++c)
  {
    double p[3], dv[3];
    uint64_t k;
    point_of_cell(nc, jitter, c, p, &k);
    MFTLGetMinimumImageVector(d, dv, x, p);
    if (fabs(dv[0]) > 2.0 * H || fabs(dv[1]) > 2.0 * H || fabs(dv[2]) > 2.0 * H || n >= cap) continue;
    for (int a = 0; a < 3; ++a) X[3 * n + a] = x[a] + dv[a];
    q[n] = H * H * H;
    keys[n++] = k;
  }
  const int st = MFTLCouplingAssembleRow(P, x, Hv, n, X, q, w, MFTL_NULL);
  free(X);
  free(q);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

/**
 * @brief A point cloud registered as two pieces per rank, each with REVERSED,
 *        GAPPED array indices and explicit cell extents, on a box periodic in x
 *        and z: rows equal brute force over the global cloud (markers on
 *        samples, on the seam, periods away), the spread lands at the right
 *        array elements and nowhere else, and the power identity holds.
 */
static int test_point_clouds_registered_awkwardly(void)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {true, false, true};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
  const int64_t nc = 9;
  const double jitter = 0.6, H = 1.0 / (double) nc;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  adv_cloud C;
  adv_cloud_make(g, nc, jitter, &C);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.max_amplification = 6.0;
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double all[3 * 48];
  int na = 0;
  uint64_t s = 4711u;
  for (int i = 0; i < 30; ++i, ++na)
    for (int k = 0; k < 3; ++k) all[3 * na + k] = (k == 1) ? 0.25 + 0.5 * adv_unif(&s) : adv_unif(&s);
  /* on samples of the global cloud */
  for (int64_t cc = 100; cc < 700 && na < 38; cc += 97, ++na)
  {
    uint64_t kk;
    point_of_cell(nc, jitter, cc, &all[3 * na], &kk);
    if (all[3 * na + 1] < 0.25 || all[3 * na + 1] > 0.75) all[3 * na + 1] = 0.5;
  }
  const double sp[6][3] = {{0.0, 0.5, 0.0}, {nextafter(0.0, -1.0), 0.4, nextafter(1.0, 0.0)}, {1.0, 0.6, 1.0}, {-2.25, 0.5, 3.5}, {0.999, 0.31, 0.001},
                           {0.5, 0.5, 0.5}};
  for (int i = 0; i < 6; ++i, ++na) memcpy(&all[3 * na], sp[i], sizeof sp[i]);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, na, all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *bp = MFTLCouplerBindingPositions(b);
  enum
  {
    CAP = 2048
  };
  uint64_t *rk = (uint64_t *) malloc(CAP * sizeof(uint64_t));
  double *rw = (double *) malloc(CAP * sizeof(double));
  double worst = 0.0;
  int bad = 0;
  for (int64_t i = 0; i < n; ++i)
  {
    int ne = 0;
    const uint64_t *keys;
    const double *wts;
    MFTLCouplerBindingRow(b, 0, i, &ne, &keys, &wts);
    uint64_t kc[CAP];
    double wc[CAP];
    memcpy(kc, keys, (size_t) ne * sizeof(uint64_t));
    memcpy(wc, wts, (size_t) ne * sizeof(double));
    const int nr = adv_ref_cloud_row(&ens->domain, nc, jitter, &P, &bp[3 * i], rk, rw, CAP);
    if (nr < 0)
    {
      ++bad;
      continue;
    }
    worst = fmax(worst, adv_row_diff(ne, kc, wc, nr, rk, rw));
  }
  if (bad || worst > 1e-11) printf("\n  rank %d: %d bad rows, worst %.3e", g_rank, bad, worst);
  TEST_ASSERT(agreed(bad == 0 && worst <= 1e-11, MPI_COMM_WORLD));

  /* fields at the gapped elements: affine in y (walled), constant across the periodic axes */
  affine Af = {-0.4, {0.0, 2.0, 0.0}};
  for (int p = 0; p < 2; ++p)
  {
    for (int64_t e = 0; e < C.len[p]; ++e) C.arr[p][e] = NAN;
    for (int64_t i = 0; i < C.n[p]; ++i) C.arr[p][C.aidx[p][i]] = f_affine(&C.pos[p][3 * i], &Af);
  }
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = C.loc;
  F.data = C.arr;
  double *U = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double aerr = 0.0;
  for (int64_t i = 0; i < n; ++i) aerr = fmax(aerr, fabs(U[i] - f_affine(&bp[3 * i], &Af)));
  TEST_ASSERT(agreed(aerr <= 1e-12, MPI_COMM_WORLD));

  /* the spread: right elements, nothing in the gaps */
  for (int p = 0; p < 2; ++p)
    for (int64_t e = 0; e < C.len[p]; ++e) C.arr[p][e] = (e % 2 == 1) ? 0.0 : -5.0;
  double *Ld = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < n; ++i) Ld[i] = hash01(ref[i].key * 13u + 1u);
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int gaps = 1;
  for (int p = 0; p < 2; ++p)
    for (int64_t e = 0; e < C.len[p]; e += 2)
      if (C.arr[p][e] != -5.0) gaps = 0;
  TEST_ASSERT(agreed(gaps, MPI_COMM_WORLD));
  double *mine = (double *) malloc((size_t) (n > 0 ? 4 * n : 1) * sizeof(double));
  for (int64_t i = 0; i < n; ++i)
  {
    memcpy(&mine[4 * i], &bp[3 * i], 3 * sizeof(double));
    mine[4 * i + 3] = Ld[i];
  }
  int tot = 0;
  double *gat = allgather_doubles(mine, (int) n, 4, &tot);
  double sworst = 0.0;
  for (int p = 0; p < 2; ++p)
  {
    double *ex = (double *) calloc((size_t) (C.n[p] > 0 ? C.n[p] : 1), sizeof(double)), *sc = (double *) calloc((size_t) (C.n[p] > 0 ? C.n[p] : 1), sizeof(double));
    for (int m = 0; m < tot; ++m)
    {
      const int nr = adv_ref_cloud_row(&ens->domain, nc, jitter, &P, &gat[4 * m], rk, rw, CAP);
      double ra = 0.0;
      for (int e = 0; e < nr; ++e) ra += fabs(rw[e]);
      for (int e = 0; e < nr; ++e)
        for (int64_t i = 0; i < C.n[p]; ++i)
          if (C.key[p][i] == rk[e])
          {
            ex[i] += rw[e] * gat[4 * m + 3];
            sc[i] += ra * fabs(gat[4 * m + 3]);
          }
    }
    for (int64_t i = 0; i < C.n[p]; ++i) sworst = fmax(sworst, fabs(C.arr[p][C.aidx[p][i]] - ex[i]) / fmax(sc[i], 1e-12));
    free(ex);
    free(sc);
  }
  TEST_ASSERT(agreed(sworst <= 1e-12, MPI_COMM_WORLD));
  (void) H;
  free(gat);
  free(mine);
  free(Ld);
  free(U);
  free(rk);
  free(rw);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_cloud_free(&C);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A marker an ulp below a bin face, where (x - min) / b rounds UP onto
 *        the integer so that floor() names the next bin -- whose interval
 *        [min + i b, min + (i+1) b] starts above x -- binds in the window
 *        requested for it with no skin. It used to be STALE there: the request
 *        binned it by floor(), containment judged it by the interval, the two
 *        disagreed, and a fresh request reproduced the same bins, so a solver
 *        loop could never bind it.
 */
static int test_a_marker_an_ulp_below_a_bin_face_binds(void)
{
  /* find such a position: on a periodic unit box with N cells the window bins
   * are half of Peskin's reach, i.e. h = 1/N; search N and the faces */
  const bool per[3] = {true, true, true};
  adv_lat A;
  double xbad = NAN;
  for (int64_t N = 5; N < 64 && isnan(xbad); ++N)
  {
    const double b = 1.0 / (double) N;
    for (int64_t i = 1; i <= N && isnan(xbad); ++i)
    {
      double x = nextafter((double) i * b, -1e300);
      for (int j = 0; j < 4; ++j, x = nextafter(x, -1e300))
      {
        /* the quotient can round UP onto the next integer: floor() then names
         * a bin whose interval starts above x */
        const int64_t ib = (int64_t) floor(x / b);
        if (x < 1.0 && (x < (double) ib * b || x > (double) ib * b + b))
        {
          xbad = x;
          break;
        }
      }
    }
    memset(&A, 0, sizeof A);
    for (int k = 0; k < 3; ++k)
    {
      A.N[k] = N;
      A.h[k] = b;
      A.origin[k] = 0.5 * b;
      A.L[k] = 1.0;
      A.per[k] = true;
    }
  }
  (void) per;
  TEST_ASSERT(!isnan(xbad)); /* non-vacuity: the inconsistent position exists on this lattice */
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, g_nproc + 1, 12u, 0, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double all[3] = {xbad, A.gmin[1] + 0.5 * A.L[1], A.gmin[2] + 0.5 * A.L[2]};
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *bd = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, 1, all, 0, MFTL_NULL, &w, &bd, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* and the row is the brute-force row */
  int ok = 1;
  if (n == 1)
  {
    MFTL_CouplerParams P;
    MFTLCouplerDefaultParams(&P);
    uint64_t rk[512];
    double rw[512];
    int ne = 0;
    const uint64_t *keys;
    const double *wts;
    MFTLCouplerBindingRow(bd, 0, 0, &ne, &keys, &wts);
    const int nr = adv_ref_row(&A, &P, MFTLCouplerBindingPositions(bd), rk, rw, 512);
    ok = nr > 0 && adv_row_diff(ne, keys, wts, nr, rk, rw) <= 1e-12;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bd);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
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
  if (g_rank == 0) printf("\n=== Coupling under attack (np = %d) ===\n", g_nproc);

  RUN_TEST(test_irregular_decompositions_match_brute_force);
  RUN_TEST(test_every_kernel_through_the_pipeline);
  RUN_TEST(test_interleaved_components_through_the_stride);
  RUN_TEST(test_more_ghosted_blocks_than_the_halo_route_tracks);
  RUN_TEST(test_two_couplers_interleave);
  RUN_TEST(test_truncation_is_counted_exactly);
  RUN_TEST(test_a_non_finite_load_is_refused_everywhere);
  RUN_TEST(test_coincident_markers);
  RUN_TEST(test_a_stretched_periodic_lattice_straddling_the_seam);
  RUN_TEST(test_irregular_scenario_is_bitwise_partition_independent);
  RUN_TEST(test_point_clouds_registered_awkwardly);
  RUN_TEST(test_a_marker_an_ulp_below_a_bin_face_binds);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
