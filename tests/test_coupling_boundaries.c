/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Walls: the mirror boundary policy, against a reference built from
 *        lattice INDICES.
 *
 * The coupler finds images by reflecting positions about the wall planes. The
 * reference never reflects anything: it enumerates the virtual lattice indices
 * beyond a wall, maps each to the index whose sample it mirrors by integer
 * arithmetic, and composes parities and wall shares axis by axis -- so a wrong
 * wall plane, a missing corner image, a doubled sample on the wall or a parity
 * applied in the wrong order is a mismatch, not a shared mistake.
 *
 * Then the physics: a profile linear through the wall value is reproduced
 * exactly at any distance from an ODD wall (the image rule is exact for it);
 * the loads the samples receive plus the walls' reactions equal the markers'
 * total; and the power identity holds with the wall work included.
 */

#include "test_coupling_irregular.h"

/** @brief A walled (or partly periodic) box and a wall-aligned lattice of it. */
typedef struct
{
  adv_lat A;
  bool node[3];  ///< samples on the walls (node) or at cell centres
  MFTL_WallParity wall[6];
} mirror_case;

static mirror_case mirror_case_make(const bool per[3], const bool node[3], const MFTL_WallParity wall[6])
{
  mirror_case M;
  memset(&M, 0, sizeof M);
  const int64_t n[3] = {11, 9, 10};
  const double L[3] = {1.1, 0.9, 1.25}, gmin[3] = {-0.3, 0.2, 1.0};
  for (int k = 0; k < 3; ++k)
  {
    M.A.per[k] = per[k];
    M.A.gmin[k] = gmin[k];
    M.A.L[k] = L[k];
    M.A.h[k] = L[k] / (double) n[k];
    M.node[k] = node[k] && !per[k];
    M.A.N[k] = M.node[k] ? n[k] + 1 : n[k];
    M.A.origin[k] = gmin[k] + (M.node[k] ? 0.0 : 0.5 * M.A.h[k]);
  }
  for (int f = 0; f < 6; ++f) M.wall[f] = wall[f];
  return M;
}

static int mirror_location(MFTL_SolverGrid *g, const char *name, const mirror_case *M)
{
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = name;
  ld.boundary = MFTL_BOUNDARY_MIRROR;
  for (int f = 0; f < 6; ++f) ld.wall[f] = M->wall[f];
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = M->A.N[k];
    ld.lattice.origin[k] = M->A.origin[k];
    ld.lattice.spacing[k] = M->A.h[k];
  }
  return MFTLSolverGridAddLocation(g, &ld);
}

/**
 * @brief Index arithmetic of one axis: the real index a virtual index i
 *        mirrors (itself inside the lattice), and which face it crossed (-1
 *        none, 0 lo, 1 hi). A cell lattice mirrors i < 0 to -1 - i; a node
 *        lattice, whose end samples sit ON the walls, to -i.
 */
static int64_t mirror_index(const mirror_case *M, int k, int64_t i, int *face)
{
  const int64_t N = M->A.N[k];
  *face = -1;
  if (M->A.per[k] || (i >= 0 && i < N)) return i;
  if (i < 0)
  {
    *face = 0;
    return M->node[k] ? -i : -1 - i;
  }
  *face = 1;
  return M->node[k] ? 2 * (N - 1) - i : 2 * N - 1 - i;
}

/**
 * @brief The reference composed row at x: every virtual index in the support,
 *        weights over the COMPLETE virtual lattice through the row core, then
 *        each weight moved to the index it mirrors, times the parities, its
 *        wall shares composed in x, then y, then z.
 * @return Entry count, or -1.
 */
static int mirror_ref_row(const mirror_case *M, const MFTL_CouplerParams *P, const double x[3], uint64_t *keys, double *w, double coef[6], int cap)
{
  const adv_lat *A = &M->A;
  MFTL_KernelInfo ki;
  MFTLCouplingKernelInfo(P->kernel, &ki);
  int64_t a[3], b[3];
  for (int k = 0; k < 3; ++k)
  {
    const double R = ki.half_width * P->support_scale * A->h[k];
    a[k] = (int64_t) floor((x[k] - R - A->origin[k]) / A->h[k]) - 1;
    b[k] = (int64_t) ceil((x[k] + R - A->origin[k]) / A->h[k]) + 1;
  }
  enum
  {
    VCAP = 4096
  };
  double *X = (double *) malloc(3 * VCAP * sizeof(double)), *q = (double *) malloc(VCAP * sizeof(double)), *wv = (double *) malloc(VCAP * sizeof(double));
  int64_t (*vi)[3] = malloc(VCAP * sizeof *vi);
  int nv = 0;
  for (int64_t i = a[0]; i <= b[0]; ++i)
    for (int64_t j = a[1]; j <= b[1]; ++j)
      for (int64_t l = a[2]; l <= b[2]; ++l)
      {
        const int64_t idx[3] = {i, j, l};
        double p[3];
        bool in = true;
        for (int k = 0; k < 3; ++k)
        {
          p[k] = A->per[k] ? adv_pos(A, k, idx[k]) : A->origin[k] + (double) idx[k] * A->h[k];
          if (fabs(p[k] - x[k]) >= ki.half_width * P->support_scale * A->h[k]) in = false;
        }
        if (!in || nv >= VCAP) continue;
        memcpy(&X[3 * nv], p, sizeof p);
        q[nv] = A->h[0] * A->h[1] * A->h[2];
        memcpy(vi[nv], idx, sizeof idx);
        ++nv;
      }
  const int st = MFTLCouplingAssembleRow(P, x, A->h, nv, X, q, wv, MFTL_NULL);
  int n = 0;
  for (int g = 0; g < 6; ++g) coef[g] = 0.0;
  if (st == MFTL_COUPLING_OK)
    for (int e = 0; e < nv; ++e)
    {
      int64_t src[3];
      double s = 1.0, cf[6] = {0, 0, 0, 0, 0, 0};
      for (int k = 0; k < 3; ++k)
      {
        int face;
        src[k] = mirror_index(M, k, vi[e][k], &face);
        if (A->per[k]) src[k] = ref_mod(src[k], A->N[k]);
        if (face < 0) continue;
        const int f = k + 3 * face;
        const double sk = (M->wall[f] == MFTL_WALL_ODD) ? -1.0 : 1.0;
        for (int g = 0; g < 6; ++g) cf[g] *= sk;
        cf[f] += 1.0 - sk;
        s *= sk;
      }
      for (int g = 0; g < 6; ++g) coef[g] += wv[e] * cf[g];
      const uint64_t key = (uint64_t) (src[0] + A->N[0] * (src[1] + A->N[1] * src[2]));
      int at = -1;
      for (int r = 0; r < n; ++r)
        if (keys[r] == key) at = r;
      if (at < 0)
      {
        if (n >= cap) continue;
        at = n++;
        keys[at] = key;
        w[at] = 0.0;
      }
      w[at] += wv[e] * s;
    }
  free(X);
  free(q);
  free(wv);
  free(vi);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

/** @brief One mirror setup: a coupler over one location and USER markers. */
typedef struct
{
  MFTL_Ensemble *ens;
  MFTL_SolverGrid *g;
  int loc;
  adv_blocks B;
  MFTL_Coupler *c;
  MFTL_CouplingWindow *w;
  MFTL_CouplingBinding *b;
  int64_t n;
} mirror_setup;

static int mirror_bind(const mirror_case *M, const MFTL_CouplerParams *P, int n_all, const double *all, mirror_setup *x)
{
  memset(x, 0, sizeof *x);
  x->ens = adv_ensemble(&M->A);
  x->g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  x->loc = mirror_location(x->g, "mirrored", M);
  adv_decompose(x->g, x->loc, &M->A, 2 * g_nproc + 1, 9090u, 2, 1, false, &x->B);
  int st = MFTLCouplerCreate(x->ens, x->g, P, &x->c);
  if (st == MFTL_COUPLING_OK) st = adv_bind_list(x->c, n_all, all, 0, MFTL_NULL, &x->w, &x->b, &x->n);
  return st;
}

static void mirror_teardown(mirror_setup *x)
{
  MFTLCouplerReleaseBinding(&x->b);
  MFTLCouplerReleaseWindow(&x->w);
  MFTLCouplerDestroy(&x->c);
  adv_free_blocks(&x->B);
  MFTLSolverGridDestroy(&x->g);
  MFTLEnsembleDestroy(&x->ens);
}

/** @brief Markers all over a box, many within a support of a wall, some on walls, corners and edges. */
static int mirror_markers(const mirror_case *M, double *all, int cap)
{
  const adv_lat *A = &M->A;
  int n = 0;
  uint64_t s = 3141u;
  for (int i = 0; i < 40 && n < cap; ++i, ++n)
    for (int k = 0; k < 3; ++k)
    {
      /* half of them hug a wall: within two cells of it */
      const double u = adv_unif(&s);
      double t = u;
      if (!A->per[k] && i % 2 == 0) t = (u < 0.5) ? 2.0 * u * 2.0 * A->h[k] / A->L[k] : 1.0 - 2.0 * (u - 0.5) * 2.0 * A->h[k] / A->L[k];
      all[3 * n + k] = A->gmin[k] + t * A->L[k];
    }
  /* on a wall, on an edge, in a corner (walled axes only) */
  for (int pat = 1; pat < 8 && n < cap; ++pat)
  {
    bool ok = true;
    for (int k = 0; k < 3; ++k)
      if ((pat & (1 << k)) && A->per[k]) ok = false;
    if (!ok) continue;
    for (int k = 0; k < 3; ++k) all[3 * n + k] = (pat & (1 << k)) ? ((pat & 2) ? A->gmin[k] + A->L[k] : A->gmin[k]) : A->gmin[k] + 0.43 * A->L[k];
    ++n;
  }
  return n;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Rows
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Mirror rows equal the reference composed from lattice indices -- for
 *        cell and node lattices, EVEN, ODD and mixed faces, on walls, edges and
 *        corners -- including every wall coefficient. On a wall-aligned lattice
 *        the virtual lattice is complete, so Peskin's rows need no correction
 *        at the wall at all, and EVEN rows stay nonnegative where a truncated
 *        row would not.
 */
static int test_mirror_rows_match_the_index_reference(void)
{
  const bool walls[3] = {false, false, false}, mixed[3] = {false, true, false};
  const bool cells[3] = {false, false, false}, nodes[3] = {true, true, true}, some[3] = {true, false, true};
  const MFTL_WallParity even[6] = {0, 0, 0, 0, 0, 0}, odd[6] = {1, 1, 1, 1, 1, 1}, mix[6] = {1, 0, 1, 0, 1, 1};
  const struct
  {
    const bool *per, *node;
    const MFTL_WallParity *wall;
  } cases[6] = {{walls, cells, even}, {walls, cells, odd}, {walls, nodes, odd}, {walls, some, mix}, {mixed, cells, mix}, {mixed, nodes, even}};
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  int ok = 1;
  long mirrored = 0, corrected = 0;
  double min_even = 0.0;
  for (int ci = 0; ci < 6; ++ci)
  {
    mirror_case M = mirror_case_make(cases[ci].per, cases[ci].node, cases[ci].wall);
    double all[3 * 64];
    const int na = mirror_markers(&M, all, 64);
    mirror_setup x;
    if (!agreed(mirror_bind(&M, &P, na, all, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD))
    {
      if (g_rank == 0) printf("\n  case %d: bind refused", ci);
      ok = 0;
      mirror_teardown(&x);
      continue;
    }
    const double *bp = MFTLCouplerBindingPositions(x.b);
    uint64_t rk[2048];
    double rw[2048], rc[6];
    double worst = 0.0;
    for (int64_t i = 0; i < x.n; ++i)
    {
      int ne = 0;
      const uint64_t *keys;
      const double *wts;
      MFTLCouplerBindingRow(x.b, 0, i, &ne, &keys, &wts);
      uint64_t kc[2048];
      double wc[2048], cc[6];
      memcpy(kc, keys, (size_t) ne * sizeof(uint64_t));
      memcpy(wc, wts, (size_t) ne * sizeof(double));
      MFTLCouplerBindingWallCoefficients(x.b, 0, i, cc);
      const int nr = mirror_ref_row(&M, &P, &bp[3 * i], rk, rw, rc, 2048);
      if (nr < 0)
      {
        ok = 0;
        continue;
      }
      worst = fmax(worst, adv_row_diff(ne, kc, wc, nr, rk, rw));
      for (int g = 0; g < 6; ++g) worst = fmax(worst, fabs(cc[g] - rc[g]));
      if (ci == 0)
        for (int e = 0; e < ne; ++e) min_even = fmin(min_even, wc[e]);
    }
    if (worst > 1e-12)
    {
      printf("\n  rank %d case %d: worst %.3e", g_rank, ci, worst);
      ok = 0;
    }
    MFTL_BindingQuality q;
    MFTLCouplerBindingQuality(x.b, &q);
    mirrored += (long) q.n_mirrored;
    corrected += (long) q.n_corrected;
    mirror_teardown(&x);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* non-vacuity: images were used; and completeness: no correction was needed */
  TEST_ASSERT(anywhere(mirrored > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(corrected == 0, MPI_COMM_WORLD));
  /* positivity survives the wall under EVEN parity */
  TEST_ASSERT(agreed(min_even >= 0.0, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Physics
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double uw;  ///< the wall value of the x-lo face
  double g;  ///< the gradient along x
  double x0;  ///< the x-lo wall
  double t;  ///< a tangential gradient along y
} wall_profile;

static double f_wall_profile(const double x[3], void *ctx)
{
  const wall_profile *W = (const wall_profile *) ctx;
  return W->uw + W->g * (x[0] - W->x0) + W->t * x[1];
}

/**
 * @brief The image rule is exact for a profile linear through the wall value:
 *        with an ODD x-lo face of value u_w and EVEN y faces, u = u_w + g (x -
 *        x_w) + t y is interpolated EXACTLY at every marker, against the wall
 *        and in its corners, with no moment correction. On a node lattice the
 *        sample on the wall is the wall value itself.
 */
static int test_a_profile_through_the_wall_value_is_exact(void)
{
  const bool per[3] = {false, false, true};
  for (int nd = 0; nd < 2; ++nd)
  {
    const bool node[3] = {nd == 1, nd == 1, false};
    const MFTL_WallParity wall[6] = {MFTL_WALL_ODD, MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_ODD, MFTL_WALL_EVEN, MFTL_WALL_EVEN};
    mirror_case M = mirror_case_make(per, node, wall);
    double all[3 * 64];
    const int na = mirror_markers(&M, all, 64);
    mirror_setup x;
    TEST_ASSERT(agreed(mirror_bind(&M, MFTL_NULL, na, all, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    /* both x faces ODD: the profile must be linear THROUGH both wall values */
    /* (a tangential gradient would make the wall value vary along the wall,
     * which one number per face cannot carry: exactness needs t = 0 here) */
    wall_profile W = {0.75, -1.5, M.A.gmin[0], 0.0};
    adv_fill(&x.B, &M.A, 0, f_wall_profile, &W, true);
    double wv[6] = {0, 0, 0, 0, 0, 0};
    wv[0] = W.uw;                  /* x-lo */
    wv[3] = W.uw + W.g * M.A.L[0]; /* x-hi */
    MFTL_SolverField F = adv_field(&x.B, x.loc, 0, 1, true);
    F.wall_value = wv;
    double *U = (double *) malloc((size_t) (x.n > 0 ? x.n : 1) * sizeof(double));
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const double *bp = MFTLCouplerBindingPositions(x.b);
    double worst = 0.0;
    for (int64_t i = 0; i < x.n; ++i) worst = fmax(worst, fabs(U[i] - f_wall_profile(&bp[3 * i], &W)));
    if (worst > 1e-13) printf("\n  rank %d node %d: profile off by %.3e", g_rank, nd, worst);
    TEST_ASSERT(agreed(worst <= 1e-13, MPI_COMM_WORLD));
    /* the same field with the wall values withheld is NOT reproduced: the coefficients matter */
    F.wall_value = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double off = 0.0;
    for (int64_t i = 0; i < x.n; ++i) off = fmax(off, fabs(U[i] - f_wall_profile(&bp[3 * i], &W)));
    TEST_ASSERT(anywhere(off > 1e-3, MPI_COMM_WORLD));
    free(U);
    mirror_teardown(&x);
  }
  return 0;
}

/**
 * @brief Spreading under mirror walls: the loads the samples receive plus the
 *        ODD faces' reactions equal the markers' total (each component); the
 *        power identity holds with the wall work, \f$U \cdot F = u \cdot b +
 *        \sum_f u_{w,f} R_f\f$; and the spread is the transpose of the
 *        reference rows, sample by sample. EVEN faces absorb nothing.
 */
static int test_mirror_spreading_balances_with_the_walls(void)
{
  const bool per[3] = {false, true, false};
  const bool node[3] = {false, false, true};
  const MFTL_WallParity wall[6] = {MFTL_WALL_ODD, MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_ODD, MFTL_WALL_EVEN, MFTL_WALL_ODD};
  mirror_case M = mirror_case_make(per, node, wall);
  double all[3 * 64];
  const int na = mirror_markers(&M, all, 64);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  mirror_setup x;
  TEST_ASSERT(agreed(mirror_bind(&M, &P, na, all, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  const double *bp = MFTLCouplerBindingPositions(x.b);
  double *L = (double *) malloc((size_t) (x.n > 0 ? x.n : 1) * sizeof(double)), *U = (double *) malloc((size_t) (x.n > 0 ? x.n : 1) * sizeof(double));
  for (int64_t i = 0; i < x.n; ++i) L[i] = 1.0 + hash01(ref[i].key * 3u + 1u);
  /* target zero, then the spread with reactions requested */
  for (int i = 0; i < x.B.n; ++i)
    for (int64_t e = 0; e < x.B.n_elem[i]; ++e) x.B.a[i][e] = 0.0;
  MFTL_SolverField T = adv_field(&x.B, x.loc, 0, 1, false);
  double R[6] = {0, 0, 0, 0, 0, 0};
  T.wall_reaction = R;
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &T, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* the reaction is identical on every rank */
  double Rmax[6], Rmin[6];
  for (int g = 0; g < 6; ++g) Rmax[g] = Rmin[g] = R[g];
  MPI_Allreduce(MPI_IN_PLACE, Rmax, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, Rmin, 6, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  int same = 1;
  for (int g = 0; g < 6; ++g)
    if (Rmax[g] != Rmin[g]) same = 0;
  TEST_ASSERT(same);
  /* EVEN faces and the periodic axis absorb nothing; the ODD ones something */
  TEST_ASSERT(R[1] == 0.0 && R[4] == 0.0 && R[2] == 0.0);
  TEST_ASSERT(fabs(R[0]) + fabs(R[3]) + fabs(R[5]) > 0.0);
  /* balance: received + reactions == total */
  double s[2] = {0.0, 0.0};
  for (int64_t i = 0; i < x.n; ++i) s[0] += L[i];
  for (int i = 0; i < x.B.n; ++i)
    for (int64_t e = 0; e < x.B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&x.B, i, e, idx);
      bool own = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < x.B.lo[i][k] || idx[k] >= x.B.hi[i][k]) own = false;
      if (own) s[1] += x.B.a[i][e];
    }
  MPI_Allreduce(MPI_IN_PLACE, s, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double rsum = R[0] + R[1] + R[2] + R[3] + R[4] + R[5];
  if (g_rank == 0 && !(fabs(s[1] + rsum - s[0]) <= 1e-12 * s[0])) printf("\n  balance: %.17g + %.17g vs %.17g", s[1], rsum, s[0]);
  TEST_ASSERT(fabs(s[1] + rsum - s[0]) <= 1e-12 * s[0]);

  /* power: a random field with random wall values */
  adv_blocks Bu = x.B;
  double *pu[ADV_MAX_BOXES];
  for (int i = 0; i < x.B.n; ++i)
  {
    Bu.a[i] = (double *) malloc((size_t) x.B.n_elem[i] * sizeof(double));
    pu[i] = Bu.a[i];
  }
  rnd_ctx rcx = {{M.A.N[0], M.A.N[1], M.A.N[2]}, 77u};
  adv_fill(&Bu, &M.A, 0, f_random, &rcx, false);
  MFTL_SolverField Fu = T;
  Fu.data = x.B.n ? pu : MFTL_NULL;
  Fu.wall_reaction = MFTL_NULL;
  const double wv[6] = {0.3, -9.0, 7.0, -0.8, 11.0, 1.7}; /* EVEN and periodic faces' values must not matter */
  Fu.wall_value = wv;
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &Fu, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double p[3] = {0.0, 0.0, 0.0};
  for (int64_t i = 0; i < x.n; ++i)
  {
    p[0] += U[i] * L[i];
    p[2] += fabs(U[i] * L[i]);
  }
  for (int i = 0; i < x.B.n; ++i)
    for (int64_t e = 0; e < x.B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&x.B, i, e, idx);
      bool own = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < x.B.lo[i][k] || idx[k] >= x.B.hi[i][k]) own = false;
      if (own) p[1] += Bu.a[i][e] * x.B.a[i][e];
    }
  MPI_Allreduce(MPI_IN_PLACE, p, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  double wall_work = 0.0;
  for (int g = 0; g < 6; ++g) wall_work += wv[g] * R[g];
  if (g_rank == 0 && !(fabs(p[0] - p[1] - wall_work) <= 1e-12 * p[2])) printf("\n  power: %.17g vs %.17g + %.17g", p[0], p[1], wall_work);
  TEST_ASSERT(fabs(p[0] - p[1] - wall_work) <= 1e-12 * p[2]);

  /* the transpose, sample by sample, against the reference rows */
  double *mine = (double *) malloc((size_t) (x.n > 0 ? 4 * x.n : 1) * sizeof(double));
  for (int64_t i = 0; i < x.n; ++i)
  {
    memcpy(&mine[4 * i], &bp[3 * i], 3 * sizeof(double));
    mine[4 * i + 3] = L[i];
  }
  int tot = 0;
  double *gat = allgather_doubles(mine, (int) x.n, 4, &tot);
  double worst = 0.0;
  uint64_t rk[2048];
  double rw[2048], rc[6];
  for (int i = 0; i < x.B.n; ++i)
  {
    double *ex = (double *) calloc((size_t) x.B.n_elem[i], sizeof(double));
    for (int m = 0; m < tot; ++m)
    {
      const int nr = mirror_ref_row(&M, &P, &gat[4 * m], rk, rw, rc, 2048);
      for (int e = 0; e < nr; ++e)
      {
        const int64_t idx[3] = {(int64_t) (rk[e] % (uint64_t) M.A.N[0]), (int64_t) ((rk[e] / (uint64_t) M.A.N[0]) % (uint64_t) M.A.N[1]),
                                (int64_t) (rk[e] / (uint64_t) (M.A.N[0] * M.A.N[1]))};
        bool own = true;
        for (int k = 0; k < 3; ++k)
          if (idx[k] < x.B.lo[i][k] || idx[k] >= x.B.hi[i][k]) own = false;
        if (!own) continue;
        ex[(idx[0] - x.B.alo[i][0]) + x.B.adims[i][0] * ((idx[1] - x.B.alo[i][1]) + x.B.adims[i][1] * (idx[2] - x.B.alo[i][2]))] += rw[e] * gat[4 * m + 3];
      }
    }
    for (int64_t e = 0; e < x.B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&x.B, i, e, idx);
      bool own = true;
      for (int k = 0; k < 3; ++k)
        if (idx[k] < x.B.lo[i][k] || idx[k] >= x.B.hi[i][k]) own = false;
      if (own) worst = fmax(worst, fabs(x.B.a[i][e] - ex[e]));
    }
    free(ex);
  }
  TEST_ASSERT(agreed(worst <= 1e-12, MPI_COMM_WORLD));
  for (int i = 0; i < x.B.n; ++i) free(Bu.a[i]);
  free(gat);
  free(mine);
  free(L);
  free(U);
  mirror_teardown(&x);
  return 0;
}

/**
 * @brief What cannot be mirrored is refused, on every rank: a marker outside
 *        the domain on a walled axis, and a support that reaches across BOTH
 *        walls of an axis (a domain thinner than the kernel). A marker exactly
 *        on a wall is fine, and a sample ON the wall is not counted twice.
 */
static int test_what_cannot_be_mirrored_is_refused(void)
{
  const bool per[3] = {false, false, false}, node[3] = {true, true, true};
  const MFTL_WallParity wall[6] = {MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_EVEN, MFTL_WALL_EVEN};
  mirror_case M = mirror_case_make(per, node, wall);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  /* outside, by a quarter cell */
  {
    const double out[3] = {M.A.gmin[0] - 0.25 * M.A.h[0], M.A.gmin[1] + 0.5, M.A.gmin[2] + 0.5};
    mirror_setup x;
    TEST_ASSERT(agreed(mirror_bind(&M, &P, 1, out, &x) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
    mirror_teardown(&x);
  }
  /* on the wall: EVEN images complete it; the sample on the wall has the plain kernel's weight, once */
  {
    const double on[3] = {M.A.gmin[0], M.A.gmin[1] + 4.0 * M.A.h[1], M.A.gmin[2] + 5.0 * M.A.h[2]};
    mirror_setup x;
    TEST_ASSERT(agreed(mirror_bind(&M, &P, 1, on, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    int ok = 1;
    if (x.n == 1)
    {
      int ne = 0;
      const uint64_t *keys;
      const double *wts;
      MFTLCouplerBindingRow(x.b, 0, 0, &ne, &keys, &wts);
      /* the marker sits on node (0, 4, 5): its weight is phi(0)^3 = 1/8, and node (1, 4, 5)
       * gets phi(1) phi(0)^2 TWICE (itself and the image of the node at -1) */
      double w0 = -1.0, w1 = -1.0;
      for (int e = 0; e < ne; ++e)
      {
        if (keys[e] == (uint64_t) (0 + M.A.N[0] * (4 + M.A.N[1] * 5))) w0 = wts[e];
        if (keys[e] == (uint64_t) (1 + M.A.N[0] * (4 + M.A.N[1] * 5))) w1 = wts[e];
      }
      ok = fabs(w0 - 0.125) <= 1e-15 && fabs(w1 - 2.0 * 0.25 * 0.25) <= 1e-15;
    }
    /* agreed, never asserted locally: a rank returning early strands the others in the teardown */
    const int all_ok = agreed(ok, MPI_COMM_WORLD);
    mirror_teardown(&x);
    TEST_ASSERT(all_ok);
  }
  /* a box thinner than the support on one axis */
  {
    mirror_case T = M;
    T.A.L[2] = 3.0 * T.A.h[2];
    T.A.N[2] = 4;
    const double in[3] = {T.A.gmin[0] + 0.5, T.A.gmin[1] + 0.5, T.A.gmin[2] + 1.5 * T.A.h[2]};
    mirror_setup x;
    TEST_ASSERT(agreed(mirror_bind(&T, &P, 1, in, &x) == MFTL_COUPLING_BAD_STENCIL, MPI_COMM_WORLD));
    mirror_teardown(&x);
  }
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Constrained samples: u_c = sum_j P_cj u_j + u_bc, transposed by the coupler
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Is lattice index (wrapped key) constrained in the test's pattern? */
static bool con_pattern(const adv_lat *A, uint64_t key)
{
  const int64_t i = (int64_t) (key % (uint64_t) A->N[0]), j = (int64_t) ((key / (uint64_t) A->N[0]) % (uint64_t) A->N[1]),
                k = (int64_t) (key / (uint64_t) (A->N[0] * A->N[1]));
  return (i + 2 * j + 3 * k) % 5 == 0;
}

/** @brief The global owner of a wrapped lattice key, from the partition every rank knows. */
static int con_owner(const adv_blocks *B, const adv_lat *A, uint64_t key)
{
  const int64_t j[3] = {(int64_t) (key % (uint64_t) A->N[0]), (int64_t) ((key / (uint64_t) A->N[0]) % (uint64_t) A->N[1]),
                        (int64_t) (key / (uint64_t) (A->N[0] * A->N[1]))};
  for (int b = 0; b < B->n_all; ++b)
  {
    const int64_t *x = B->all_box[b];
    if (j[0] >= x[0] && j[0] < x[3] && j[1] >= x[1] && j[1] < x[4] && j[2] >= x[2] && j[2] < x[5]) return B->all_owner[b];
  }
  return -1;
}

/**
 * @brief The constraint row of a constrained key: up to three FREE neighbours
 *        (periodic box), with coefficients that sum to @p sum.
 */
static int con_row(const adv_lat *A, uint64_t key, double sum, uint64_t *fk, double *cf)
{
  const int64_t i = (int64_t) (key % (uint64_t) A->N[0]), j = (int64_t) ((key / (uint64_t) A->N[0]) % (uint64_t) A->N[1]),
                k = (int64_t) (key / (uint64_t) (A->N[0] * A->N[1]));
  const int64_t off[6][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
  int n = 0;
  for (int o = 0; o < 6 && n < 3; ++o)
  {
    const int64_t ii = ref_mod(i + off[o][0], A->N[0]), jj = ref_mod(j + off[o][1], A->N[1]), kk = ref_mod(k + off[o][2], A->N[2]);
    const uint64_t nk = (uint64_t) (ii + A->N[0] * (jj + A->N[1] * kk));
    if (con_pattern(A, nk)) continue;
    fk[n++] = nk;
  }
  const double base[3] = {0.5, 0.3, 0.2};
  double tot = 0.0;
  for (int e = 0; e < n; ++e) tot += base[e];
  for (int e = 0; e < n; ++e) cf[e] = sum * base[e] / tot;
  return n;
}

static double u_bc_of(uint64_t key) { return 0.1 * hash01(key * 41u + 9u); }

/**
 * @brief Constrained samples on an irregular decomposition: the samples' loads
 *        go to their free samples (remote ones included) as \f$P^T b_c\f$, the
 *        constrained elements receive nothing, `constraint_load` records
 *        \f$b_c\f$, the result equals brute-force \f$(W P)^T F\f$, and the power
 *        identity holds with the affine part's work. With rows of P summing to
 *        one (hanging nodes) the force reaching the free samples is the
 *        markers' total.
 */
static int test_constraints_are_transposed_exactly(void)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  const int64_t n[3] = {10, 9, 11};
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = n[k];
    A.L[k] = 1.0;
    A.h[k] = 1.0 / (double) n[k];
    A.origin[k] = 0.5 * A.h[k];
    A.per[k] = true;
  }
  for (int variant = 0; variant < 2; ++variant)
  {
    const double psum = (variant == 0) ? 1.0 : 0.7;
    MFTL_Ensemble *ens = adv_ensemble(&A);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
    adv_blocks B;
    adv_decompose(g, loc, &A, 3 * g_nproc + 1, 4321u + (uint64_t) variant, 1, 1, false, &B);
    /* this rank's constrained samples and their rows */
    int64_t ncon = 0, nent = 0, remote = 0;
    const int64_t total = A.N[0] * A.N[1] * A.N[2];
    uint64_t *ck = (uint64_t *) malloc((size_t) total * sizeof(uint64_t)), *fk = (uint64_t *) malloc(3 * (size_t) total * sizeof(uint64_t));
    int64_t *rp = (int64_t *) malloc((size_t) (total + 1) * sizeof(int64_t));
    double *cf = (double *) malloc(3 * (size_t) total * sizeof(double));
    int *fo = (int *) malloc(3 * (size_t) total * sizeof(int));
    rp[0] = 0;
    for (uint64_t key = 0; key < (uint64_t) total; ++key)
    {
      if (!con_pattern(&A, key) || con_owner(&B, &A, key) != g_rank) continue;
      ck[ncon] = key;
      const int m = con_row(&A, key, psum, &fk[nent], &cf[nent]);
      for (int e = 0; e < m; ++e)
      {
        fo[nent + e] = con_owner(&B, &A, fk[nent + e]);
        if (fo[nent + e] != g_rank) ++remote;
      }
      nent += m;
      rp[++ncon] = nent;
    }
    MFTL_ConstraintDesc cd = {loc, ncon, ck, rp, fk, cf, fo};
    TEST_ASSERT(agreed(MFTLSolverGridSetConstraints(g, &cd) == MFTL_SUCCESS, MPI_COMM_WORLD));
    if (g_nproc > 1) TEST_ASSERT(anywhere(remote > 0, MPI_COMM_WORLD)); /* non-vacuity: some free samples live elsewhere */
    MFTL_Coupler *c = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    adv_markers M;
    adv_markers_make(&ens->domain, &A, 2.0 * A.h[0], 50, 919u, 0.0, &M);
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    int64_t nm = 0;
    TEST_ASSERT(agreed(adv_bind_list(c, (int) M.n_all, M.all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

    /* u: random on free samples, constrained ones reconstructed (ghosts alike) */
    rnd_ctx rc = {{A.N[0], A.N[1], A.N[2]}, 55u};
    adv_fill(&B, &A, 0, f_random, &rc, true);
    for (int i = 0; i < B.n; ++i)
      for (int64_t e = 0; e < B.n_elem[i]; ++e)
      {
        int64_t idx[3];
        adv_elem_index(&B, i, e, idx);
        int64_t j[3];
        for (int k = 0; k < 3; ++k) j[k] = ref_mod(idx[k], A.N[k]);
        const uint64_t key = (uint64_t) (j[0] + A.N[0] * (j[1] + A.N[1] * j[2]));
        if (!con_pattern(&A, key) || isnan(B.a[i][e])) continue;
        uint64_t fkk[3];
        double cff[3];
        const int m = con_row(&A, key, psum, fkk, cff);
        double v = u_bc_of(key);
        for (int q = 0; q < m; ++q)
        {
          const int64_t fi[3] = {(int64_t) (fkk[q] % (uint64_t) A.N[0]), (int64_t) ((fkk[q] / (uint64_t) A.N[0]) % (uint64_t) A.N[1]),
                                 (int64_t) (fkk[q] / (uint64_t) (A.N[0] * A.N[1]))};
          double x[3];
          for (int k = 0; k < 3; ++k) x[k] = A.origin[k] + (double) fi[k] * A.h[k];
          v += cff[q] * f_random(x, &rc);
        }
        B.a[i][e] = v;
      }
    double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double)), *Ld = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
    MFTL_SolverField F = adv_field(&B, loc, 0, 1, true);
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
    for (int64_t i = 0; i < nm; ++i) Ld[i] = 1.0 + hash01(ref[i].key * 11u);
    /* targets: free samples 0, constrained ones a sentinel the spread must not touch */
    adv_blocks T = B;
    double *pt[ADV_MAX_BOXES];
    for (int i = 0; i < B.n; ++i)
    {
      T.a[i] = (double *) malloc((size_t) B.n_elem[i] * sizeof(double));
      pt[i] = T.a[i];
      for (int64_t e = 0; e < B.n_elem[i]; ++e)
      {
        int64_t idx[3];
        adv_elem_index(&B, i, e, idx);
        int64_t j[3];
        for (int k = 0; k < 3; ++k) j[k] = ref_mod(idx[k], A.N[k]);
        T.a[i][e] = con_pattern(&A, (uint64_t) (j[0] + A.N[0] * (j[1] + A.N[1] * j[2]))) ? 123.0 : 0.0;
      }
    }
    MFTL_SolverField Ft = F;
    Ft.data = B.n ? pt : MFTL_NULL;
    Ft.halo_valid = false;
    double *bc = (double *) calloc((size_t) (ncon > 0 ? ncon : 1), sizeof(double));
    Ft.constraint_load = bc;
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &Ft, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

    /* constrained elements untouched; power with the affine work; force when P sums to one */
    int untouched = 1;
    double pw[4] = {0, 0, 0, 0}, frc[2] = {0, 0};
    for (int64_t i = 0; i < nm; ++i)
    {
      pw[0] += U[i] * Ld[i];
      pw[3] += fabs(U[i] * Ld[i]);
      frc[0] += Ld[i];
    }
    for (int i = 0; i < B.n; ++i)
      for (int64_t e = 0; e < B.n_elem[i]; ++e)
      {
        int64_t idx[3];
        adv_elem_index(&B, i, e, idx);
        bool own = true;
        for (int k = 0; k < 3; ++k)
          if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) own = false;
        if (!own) continue;
        const uint64_t key = (uint64_t) (idx[0] + A.N[0] * (idx[1] + A.N[1] * idx[2]));
        if (con_pattern(&A, key))
        {
          if (T.a[i][e] != 123.0) untouched = 0;
          continue;
        }
        pw[1] += B.a[i][e] * T.a[i][e];
        frc[1] += T.a[i][e];
      }
    for (int64_t q = 0; q < ncon; ++q) pw[2] += u_bc_of(ck[q]) * bc[q];
    MPI_Allreduce(MPI_IN_PLACE, pw, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, frc, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(untouched, MPI_COMM_WORLD));
    if (g_rank == 0 && !(fabs(pw[0] - pw[1] - pw[2]) <= 1e-12 * pw[3])) printf("\n  power: U.F %.17g vs u.b %.17g + work %.17g", pw[0], pw[1], pw[2]);
    TEST_ASSERT(fabs(pw[0] - pw[1] - pw[2]) <= 1e-12 * pw[3]);
    if (variant == 0) TEST_ASSERT(fabs(frc[0] - frc[1]) <= 1e-12 * frc[0]);
    if (variant == 1) TEST_ASSERT(fabs(frc[0] - frc[1]) > 1e-3 * frc[0]); /* a sum below one keeps a share back: the check bites */

    /* against brute force (W P)^T F, owned free samples, and W^T F on the constrained */
    {
      MFTL_CouplerParams P;
      MFTLCouplerDefaultParams(&P);
      const double *bp = MFTLCouplerBindingPositions(b);
      double *mine = (double *) malloc((size_t) (nm > 0 ? 4 * nm : 1) * sizeof(double));
      for (int64_t i = 0; i < nm; ++i)
      {
        memcpy(&mine[4 * i], &bp[3 * i], 3 * sizeof(double));
        mine[4 * i + 3] = Ld[i];
      }
      int tot = 0;
      double *gat = allgather_doubles(mine, (int) nm, 4, &tot);
      double *bref = (double *) calloc((size_t) total, sizeof(double)), *cref = (double *) calloc((size_t) total, sizeof(double));
      uint64_t rk[1024];
      double rw[1024];
      for (int m = 0; m < tot; ++m)
      {
        const int nr = adv_ref_row(&A, &P, &gat[4 * m], rk, rw, 1024);
        for (int e = 0; e < nr; ++e)
        {
          const double v = rw[e] * gat[4 * m + 3];
          if (!con_pattern(&A, rk[e]))
          {
            bref[rk[e]] += v;
            continue;
          }
          cref[rk[e]] += v;
          uint64_t fkk[3];
          double cff[3];
          const int mm = con_row(&A, rk[e], psum, fkk, cff);
          for (int q = 0; q < mm; ++q) bref[fkk[q]] += cff[q] * v;
        }
      }
      double worst = 0.0;
      for (int i = 0; i < B.n; ++i)
        for (int64_t e = 0; e < B.n_elem[i]; ++e)
        {
          int64_t idx[3];
          adv_elem_index(&B, i, e, idx);
          bool own = true;
          for (int k = 0; k < 3; ++k)
            if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) own = false;
          const uint64_t key = (uint64_t) (idx[0] + A.N[0] * (idx[1] + A.N[1] * idx[2]));
          if (own && !con_pattern(&A, key)) worst = fmax(worst, fabs(T.a[i][e] - bref[key]));
        }
      for (int64_t q = 0; q < ncon; ++q) worst = fmax(worst, fabs(bc[q] - cref[ck[q]]));
      TEST_ASSERT(agreed(worst <= 1e-12, MPI_COMM_WORLD));
      free(bref);
      free(cref);
      free(gat);
      free(mine);
    }
    for (int i = 0; i < B.n; ++i) free(T.a[i]);
    free(bc);
    free(U);
    free(Ld);
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
    MFTLCouplerDestroy(&c);
    adv_markers_free(&M);
    adv_free_blocks(&B);
    free(ck);
    free(fk);
    free(rp);
    free(cf);
    free(fo);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/**
 * @brief Constraints that cannot be right are refused on every rank before
 *        anything uses them: a constrained sample this rank does not own, a
 *        free sample claimed of a rank that does not own it, a free sample
 *        that is itself constrained, and a halo-transposed spread of a
 *        constrained location (the solver's accumulation would bypass P^T).
 *        Clearing the constraints makes the grid valid again.
 */
static int test_constraints_that_cannot_be_right_are_refused(void)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 8;
    A.L[k] = 1.0;
    A.h[k] = 0.125;
    A.origin[k] = 0.0625;
    A.per[k] = true;
  }
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, g_nproc + 1, 77u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* the first sample this rank owns, and one it does not (if any) */
  int64_t mine = -1, theirs = -1;
  for (uint64_t key = 0; key < 512; ++key)
  {
    const int o = con_owner(&B, &A, key);
    if (o == g_rank && mine < 0) mine = (int64_t) key;
    if (o != g_rank && theirs < 0) theirs = (int64_t) key;
  }
  /* the highest rank that owns a sample makes the mistake */
  int last = (mine >= 0) ? g_rank : -1;
  MPI_Allreduce(MPI_IN_PLACE, &last, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  for (int attempt = 0; attempt < 3; ++attempt)
  {
    /* only the last rank makes the mistake */
    uint64_t ck = 0, fk = 0;
    int64_t rp[2] = {0, 1};
    double cf = 1.0;
    int fo = 0;
    int64_t n = 0;
    if (g_rank == last && mine >= 0)
    {
      n = 1;
      if (attempt == 0 && theirs >= 0)
      {
        ck = (uint64_t) theirs; /* not owned here */
        fk = (uint64_t) mine;
        fo = g_rank;
      }
      else if (attempt == 0)
      {
        ck = 999999u; /* np = 1: a key outside the lattice */
        fk = (uint64_t) mine;
        fo = g_rank;
      }
      else if (attempt == 1)
      {
        ck = (uint64_t) mine;
        fk = (uint64_t) ((mine + 1) % 512);
        fo = (con_owner(&B, &A, fk) + 1) % g_nproc; /* the wrong owner (np = 1: itself, but the key...) */
        if (g_nproc == 1) fk = (uint64_t) mine;    /* ... is then the constrained key itself */
      }
      else
      {
        ck = (uint64_t) mine;
        fk = (uint64_t) mine; /* itself constrained */
        fo = g_rank;
      }
    }
    MFTL_ConstraintDesc cd = {loc, n, &ck, rp, &fk, &cf, &fo};
    TEST_ASSERT(agreed(MFTLSolverGridSetConstraints(g, &cd) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  }
  /* cleared: valid again */
  {
    MFTL_ConstraintDesc none = {loc, 0, MFTL_NULL, MFTL_NULL, MFTL_NULL, MFTL_NULL, MFTL_NULL};
    TEST_ASSERT(agreed(MFTLSolverGridSetConstraints(g, &none) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  }
  /* a good constraint, then a halo-transposed spread of it: refused */
  {
    uint64_t ck = 0, fk = 0;
    int64_t rp[2] = {0, 1};
    double cf = 0.5;
    int fo = g_rank;
    int64_t n = 0;
    if (mine >= 0)
      for (uint64_t key = 0; key < 512; ++key)
        if (key != (uint64_t) mine && con_owner(&B, &A, key) == g_rank)
        {
          ck = (uint64_t) mine;
          fk = key;
          n = 1;
          break;
        }
    MFTL_ConstraintDesc cd = {loc, n, &ck, rp, &fk, &cf, &fo};
    TEST_ASSERT(agreed(MFTLSolverGridSetConstraints(g, &cd) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double all[3] = {0.5, 0.5, 0.5};
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    int64_t nm = 0;
    TEST_ASSERT(agreed(adv_bind_list(c, 1, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
    F.halo_transpose = true;
    double L = 1.0;
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, &L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
  }
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Constraint indices cached by a window must not outlive a map change. */
static int test_constraint_updates_invalidate_windows(void)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 8;
    A.L[k] = 1.0;
    A.h[k] = 0.125;
    A.origin[k] = 0.0625;
    A.per[k] = true;
  }
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, g_nproc + 1, 77u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const uint64_t ck = 4 + 8 * (4 + 8 * 4), fk = ck + 1;
  const int co = con_owner(&B, &A, ck), fo = con_owner(&B, &A, fk);
  const int64_t rp[2] = {0, 1};
  const double cf = 0.5, all[3] = {0.52, 0.53, 0.54};
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
  int ok = 1;
  for (int clear = 0; clear < 2; ++clear)
  {
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    TEST_ASSERT(agreed(adv_bind_list(c, 1, all, 0, MFTL_NULL, &w, &b, MFTL_NULL) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_TransferSpec ts = {MFTL_TRANSFER_SPREAD, w, b, 1, &F, MFTL_SPREAD_INTEGRATED};
    MFTL_TransferReady *ready = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(c, &ts, &ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_CouplerStats s0, s1;
    MFTLCouplerReduceStats(c, &s0);
    /* Only the constraint owner changes its map. Adding and removing it must
     * invalidate every consumer, without rebuilding identical directory entries. */
    if (g_rank == co)
    {
      MFTL_ConstraintDesc cd = {loc, clear ? 0 : 1, &ck, rp, &fk, &cf, &fo};
      ok &= MFTLSolverGridSetConstraints(g, &cd) == MFTL_SUCCESS;
    }
    TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReduceStats(c, &s1);
    ok &= s1.n_directory_builds == s0.n_directory_builds;
    if (!clear)
    {
      const double load = 1.0;
      const int st = MFTLCouplerSpreadBegin(c, ready, &load);
      ok &= st == MFTL_COUPLING_STALE;
      if (st == MFTL_COUPLING_OK) MFTLCouplerSpreadEnd(c);
    }
    MFTLCouplerReleaseReady(&ready);
    const int st = MFTLCouplerPrepareTransfer(c, &ts, &ready);
    if (st != MFTL_COUPLING_STALE && g_rank == 0) printf("\n  constraint %s: old window returned %d", clear ? "removal" : "addition", st);
    ok &= st == MFTL_COUPLING_STALE;
    MFTLCouplerReleaseReady(&ready);
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
  }
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/** @brief Constraint owner ranks belong to the registered solver communicator. */
static int test_constraints_with_reordered_solver_ranks(void)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 8;
    A.L[k] = 1.0;
    A.h[k] = 0.125;
    A.origin[k] = 0.0625;
    A.per[k] = true;
  }
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MPI_Comm reversed;
  MPI_Comm_split(MPI_COMM_WORLD, 0, g_nproc - 1 - g_rank, &reversed);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(reversed);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, g_nproc + 1, 77u, 1, 1, false, &B);
  const uint64_t ck = 4 + 8 * (4 + 8 * 4), fk = 0;
  const int co = con_owner(&B, &A, ck), fo = g_nproc - 1 - con_owner(&B, &A, fk);
  const int64_t rp[2] = {0, 1};
  const double cf = 1.0;
  MFTL_ConstraintDesc cd = {loc, (g_rank == co) ? 1 : 0, &ck, rp, &fk, &cf, &fo};
  TEST_ASSERT(agreed(MFTLSolverGridSetConstraints(g, &cd) == MFTL_SUCCESS, MPI_COMM_WORLD));
  MFTL_Coupler *c = MFTL_NULL;
  int ok = MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK;
  if (agreed(ok, MPI_COMM_WORLD))
  {
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    const double all[3] = {0.5625, 0.5625, 0.5625};
    int64_t nm = 0;
    TEST_ASSERT(agreed(adv_bind_list(c, 1, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double moved = 0.0;
    for (int64_t i = 0; i < nm; ++i)
    {
      int ne = 0;
      const uint64_t *keys;
      const double *weights;
      MFTLCouplerBindingRow(b, loc, i, &ne, &keys, &weights);
      for (int e = 0; e < ne; ++e)
        if (keys[e] == ck) moved += weights[e];
    }
    MPI_Allreduce(MPI_IN_PLACE, &moved, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ok &= moved > 0.0;
    MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
    for (int pc = 0; pc < B.n; ++pc) memset(B.a[pc], 0, (size_t) B.n_elem[pc] * sizeof(double));
    const double load = 1.0;
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, &load, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    /* The far-away free key receives exactly the constrained row entry;
     * it lies outside the marker's direct support. */
    for (int pc = 0; pc < B.n; ++pc)
      for (int64_t k = B.lo[pc][2]; k < B.hi[pc][2]; ++k)
        for (int64_t j = B.lo[pc][1]; j < B.hi[pc][1]; ++j)
          for (int64_t i = B.lo[pc][0]; i < B.hi[pc][0]; ++i)
          {
            const uint64_t key = (uint64_t) (i + 8 * (j + 8 * k));
            const int64_t e = (i - B.alo[pc][0]) + B.adims[pc][0] * ((j - B.alo[pc][1]) + B.adims[pc][1] * (k - B.alo[pc][2]));
            if (key == ck || key == fk)
              ok &= fabs(B.a[pc][e] - (key == fk ? moved : 0.0)) < 1e-14;
          }
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
    MFTLCouplerDestroy(&c);
  }
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MPI_Comm_free(&reversed);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/** @brief Earlier ordinary rows can need more scratch space than mirror rows. */
static int test_mirror_after_a_dense_ordinary_location(void)
{
  mirror_case M;
  memset(&M, 0, sizeof M);
  for (int k = 0; k < 3; ++k)
  {
    M.A.N[k] = 16;
    M.A.L[k] = 1.0;
    M.A.h[k] = 1.0 / 16.0;
    M.A.origin[k] = 1.0 / 32.0;
  }
  MFTL_Ensemble *ens = adv_ensemble(&M.A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  pts_block pb;
  const int dense = add_points(g, "dense", 24, 0.0, OWN_SLAB, false, &pb);
  MFTLSolverGridReset(g);
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = dense;
  pd.n = pb.n;
  pd.keys = pb.key;
  pd.pos = pb.pos;
  pd.measure = pb.meas;
  /* An oversampled cloud with a wide kernel makes the first row much larger. */
  for (int k = 0; k < 3; ++k) pd.h_uniform[k] = 0.125;
  TEST_ASSERT(agreed(MFTLSolverGridAddPoints(g, &pd) >= 0, MPI_COMM_WORLD));
  const int loc = mirror_location(g, "mirror", &M);
  adv_blocks B;
  adv_decompose(g, loc, &M.A, g_nproc + 1, 77u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  const double all[3] = {0.02, 0.5, 0.5};
  int64_t nm = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, 1, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int ok = 1;
  for (int64_t i = 0; i < nm; ++i)
  {
    uint64_t rk[2048];
    double rw[2048], rc[6], cc[6];
    const int nr = mirror_ref_row(&M, &P, all, rk, rw, rc, 2048);
    int ne = 0;
    const uint64_t *keys;
    const double *weights;
    MFTLCouplerBindingRow(b, loc, i, &ne, &keys, &weights);
    ok &= nr >= 0 && adv_row_diff(ne, keys, weights, nr, rk, rw) < 1e-12;
    MFTLCouplerBindingWallCoefficients(b, loc, i, cc);
    for (int f = 0; f < 6; ++f) ok &= fabs(cc[f] - rc[f]) < 1e-12;
  }
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  free_points(&pb);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling boundaries (np = %d) ===\n", g_nproc);

  RUN_TEST(test_mirror_rows_match_the_index_reference);
  RUN_TEST(test_a_profile_through_the_wall_value_is_exact);
  RUN_TEST(test_mirror_spreading_balances_with_the_walls);
  RUN_TEST(test_what_cannot_be_mirrored_is_refused);
  RUN_TEST(test_constraints_are_transposed_exactly);
  RUN_TEST(test_constraints_that_cannot_be_right_are_refused);
  RUN_TEST(test_constraint_updates_invalidate_windows);
  RUN_TEST(test_constraints_with_reordered_solver_ranks);
  RUN_TEST(test_mirror_after_a_dense_ordinary_location);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
