/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief Tests of the distributed coupler: interpolation from solver samples
 *        to surface markers and its transpose, spreading.
 *
 * EVERY REFERENCE HERE SHARES NO CODE WITH THE COUPLER'S PIPELINE. Rows are
 * recomputed by brute force over the GLOBAL lattice (every index, enumerated
 * by the test, with its own Peskin formula and its own Gaussian-elimination
 * moment solve), and spreading is recomputed by gathering every marker to every
 * rank and summing the brute-force rows into the DOFs each rank owns. Nothing
 * of discovery, coverage, routing, the band or the schedules is shared, so a
 * DOF fetched twice, a load returned to the wrong owner, a missed periodic
 * image or a row silently renormalised over an incomplete support all show up
 * as a numerical mismatch rather than as a self-consistent wrong answer.
 *
 * Ghost cells the coupler is NOT allowed to read hold NaN, so a stray read of
 * the solver's array poisons the result instead of passing unnoticed.
 *
 * The solver's decomposition is chosen independently of MFTL's process grid:
 * ALIGNED (the same boxes), SLAB (z-slabs in rank order -- every marker rank
 * talks to ranks that are not its MFTL neighbours), REVERSED (the aligned
 * boxes, assigned to rank P-1-r) and ROOT (rank 0 owns every sample). Each
 * multi-rank test asserts it actually crossed ranks (non-vacuity).
 */

#include "test_coupling_fixtures.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: rows
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Rows equal brute force, on every solver decomposition, walled and
 *        periodic: the key set exactly, the weights to round-off.
 */
static int test_rows_match_brute_force(void)
{
  int ok = 1;
  long remote = 0, direct = 0;
  for (int per = 0; per < 2; ++per)
    for (int dec = 0; dec < 4; ++dec)
    {
      const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
      const bool pp[3] = {per != 0, per != 0, per != 0};
      MFTL_Ensemble *ens = make_ens(lo, hi, pp);
      const double c[3] = {0.5, 0.5, 0.5};
      add_sphere(ens, c, 0.3, 3);
      scatter(ens);

      ref_lattice R;
      memset(&R, 0, sizeof R);
      for (int k = 0; k < 3; ++k)
      {
        R.N[k] = 20;
        R.h[k] = 1.0 / 20.0;
        R.origin[k] = 0.5 * R.h[k];
        R.L[k] = 1.0;
        R.per[k] = pp[k];
      }
      MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
      lat_block blk;
      add_lattice(g, "cells", &R, (decomposition) dec, 0, &blk);
      coupled x;
      const int st = couple(ens, g, MFTL_NULL, 0.01, &x);
      if (!agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD))
      {
        if (g_rank == 0) printf("\n  couple failed: %s (per %d dec %d)\n", MFTLCouplingStatusName(st), per, dec);
        ok = 0;
      }
      else
      {
        int bad = 0;
        const double err = rows_vs_reference(&x, 0, &R, &bad);
        if (bad || err > 1e-13) ok = 0;
        if (bad || err > 1e-13) printf("\n  rank %d per %d dec %d: %d bad rows, worst %.3e", g_rank, per, dec, bad, err);
        MFTL_CouplerStats s;
        MFTLCouplerGetStats(x.c, &s);
        remote += s.band_remote;
        MFTL_BindingQuality q;
        MFTLCouplerBindingQuality(x.b, &q);
        direct += q.n_cartesian;
      }
      uncouple(&x);
      free_block(&blk);
      MFTLSolverGridDestroy(&g);
      MFTLEnsembleDestroy(&ens);
    }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(g_nproc == 1 || anywhere(remote > 0, MPI_COMM_WORLD));
  TEST_ASSERT(anywhere(direct > 0, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: interpolation and its transpose
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Linear fields are interpolated exactly, near walls too (corrected rows). */
static int test_linear_fields_are_reproduced(void)
{
  int ok = 1;
  long corrected = 0, truncated = 0;
  for (int dec = 0; dec < 4; ++dec)
  {
    const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    const bool pp[3] = {false, false, false};
    MFTL_Ensemble *ens = make_ens(lo, hi, pp);
    const double c[3] = {0.5, 0.5, 0.5};
    add_sphere(ens, c, 0.47, 3); /* nearly touches every wall: truncated rows */
    scatter(ens);
    ref_lattice R;
    memset(&R, 0, sizeof R);
    for (int k = 0; k < 3; ++k)
    {
      R.N[k] = 16;
      R.h[k] = 1.0 / 16.0;
      R.origin[k] = 0.5 * R.h[k];
      R.L[k] = 1.0;
    }
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    add_lattice(g, "cells", &R, (decomposition) dec, 0, &blk);
    affine A = {0.7, {1.3, -2.1, 0.4}};
    fill_block(&blk, f_affine, &A, false);
    coupled x;
    int st = couple(ens, g, MFTL_NULL, 0.0, &x);
    if (!agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD))
    {
      ok = 0;
      uncouple(&x);
      free_block(&blk);
      MFTLSolverGridDestroy(&g);
      MFTLEnsembleDestroy(&ens);
      continue;
    }
    const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
    double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
    MFTL_SolverField F = block_field(&blk, 1, false);
    st = MFTLCouplerInterpolate(x.c, x.b, 1, &F, U);
    if (st != MFTL_COUPLING_OK) ok = 0;
    const double *pos = MFTLCouplerBindingPositions(x.b);
    for (int64_t i = 0; i < nm && st == MFTL_COUPLING_OK; ++i)
    {
      const double exact = f_affine(&pos[3 * i], &A);
      if (!(fabs(U[i] - exact) <= 1e-12 * (1.0 + fabs(exact)))) ok = 0;
    }
    MFTL_BindingQuality q;
    MFTLCouplerBindingQuality(x.b, &q);
    corrected += q.n_corrected;
    truncated += q.n_truncated;
    free(U);
    uncouple(&x);
    free_block(&blk);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* the sphere reaches within the kernel of every wall: corrected one-sided rows happened */
  TEST_ASSERT(anywhere(corrected > 0 && truncated > 0, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief Spreading is the exact transpose: every owned DOF receives exactly the
 *        brute-force sum over ALL markers on ALL ranks, and the discrete power
 *        identity u.b = U.F holds to round-off.
 */
static int test_spread_is_the_brute_force_transpose(void)
{
  int ok = 1;
  for (int per = 0; per < 2; ++per)
    for (int dec = 0; dec < 4; ++dec)
    {
      const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
      const bool pp[3] = {per != 0, per != 0, per != 0};
      MFTL_Ensemble *ens = make_ens(lo, hi, pp);
      /* on the periodic box, put the sphere across the seam */
      const double c[3] = {per ? 0.95 : 0.5, 0.5, per ? 0.02 : 0.5};
      add_sphere(ens, c, 0.25, 3);
      scatter(ens);
      ref_lattice R;
      memset(&R, 0, sizeof R);
      for (int k = 0; k < 3; ++k)
      {
        R.N[k] = 18;
        R.h[k] = 1.0 / 18.0;
        R.origin[k] = 0.5 * R.h[k];
        R.L[k] = 1.0;
        R.per[k] = pp[k];
      }
      MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
      lat_block blk;
      add_lattice(g, "cells", &R, (decomposition) dec, 0, &blk);
      coupled x;
      const int st = couple(ens, g, MFTL_NULL, 0.02, &x);
      if (!agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD))
      {
        ok = 0;
        uncouple(&x);
        free_block(&blk);
        MFTLSolverGridDestroy(&g);
        MFTLEnsembleDestroy(&ens);
        continue;
      }
      /* loads: deterministic per marker key */
      const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
      const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
      const double *pos = MFTLCouplerBindingPositions(x.b);
      double *F = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
      for (int64_t i = 0; i < nm; ++i) F[i] = hash01(ref[i].key * 7 + 3);

      for (int64_t e = 0; e < blk.n_elem; ++e) blk.a[e] = 0.0;
      MFTL_SolverField T = block_field(&blk, 1, false);
      if (MFTLCouplerSpread(x.c, x.b, 1, &T, F, MFTL_SPREAD_INTEGRATED) != MFTL_COUPLING_OK) ok = 0;

      /* gather every marker to every rank */
      int nloc = (int) nm, *counts = (int *) malloc((size_t) g_nproc * sizeof(int)), *displ = (int *) malloc((size_t) g_nproc * sizeof(int));
      MPI_Allgather(&nloc, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
      int total = 0;
      for (int r = 0; r < g_nproc; ++r)
      {
        displ[r] = total;
        total += counts[r];
      }
      double *mine = (double *) malloc((size_t) (nloc > 0 ? 4 * nloc : 1) * sizeof(double));
      for (int64_t i = 0; i < nm; ++i)
      {
        memcpy(&mine[4 * i], &pos[3 * i], 3 * sizeof(double));
        mine[4 * i + 3] = F[i];
      }
      for (int r = 0; r < g_nproc; ++r)
      {
        counts[r] *= 4;
        displ[r] *= 4;
      }
      double *all = (double *) malloc((size_t) (total > 0 ? 4 * total : 1) * sizeof(double));
      MPI_Allgatherv(mine, 4 * nloc, MPI_DOUBLE, all, counts, displ, MPI_DOUBLE, MPI_COMM_WORLD);

      /* brute-force transpose into the DOFs this rank owns */
      double *expect = (double *) calloc((size_t) (blk.n_elem > 0 ? blk.n_elem : 1), sizeof(double));
      double *scale = (double *) calloc((size_t) (blk.n_elem > 0 ? blk.n_elem : 1), sizeof(double));
      uint64_t rk[512];
      double rw[512], dd[3 * 512];
      for (int i = 0; i < total; ++i)
      {
        const int n = ref_row(&R, &all[4 * i], rk, rw, dd, 512);
        for (int e = 0; e < n; ++e)
        {
          const int64_t idx[3] = {(int64_t) (rk[e] % (uint64_t) R.N[0]), (int64_t) ((rk[e] / (uint64_t) R.N[0]) % (uint64_t) R.N[1]),
                                  (int64_t) (rk[e] / (uint64_t) (R.N[0] * R.N[1]))};
          bool own = blk.piece >= 0;
          for (int k = 0; k < 3; ++k)
            if (idx[k] < blk.lo[k] || idx[k] >= blk.hi[k]) own = false;
          if (!own) continue;
          expect[block_elem(&blk, idx)] += rw[e] * all[4 * i + 3];
          scale[block_elem(&blk, idx)] += fabs(rw[e] * all[4 * i + 3]);
        }
      }
      double worst = 0.0;
      for (int64_t e = 0; e < blk.n_elem; ++e) worst = fmax(worst, fabs(expect[e] - blk.a[e]) / (1e-300 + fmax(scale[e], 1e-12)));
      if (worst > 1e-12)
      {
        ok = 0;
        printf("\n  rank %d per %d dec %d: spread differs by %.3e (relative)", g_rank, per, dec, worst);
      }

      /* power identity with a random field */
      rnd_ctx rc = {{R.N[0], R.N[1], R.N[2]}, 99};
      lat_block u = blk;
      u.a = (double *) malloc((size_t) (blk.n_elem > 0 ? blk.n_elem : 1) * sizeof(double));
      fill_block(&u, f_random, &rc, false);
      double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
      MFTL_SolverField Fu = block_field(&u, 2, false);
      if (MFTLCouplerInterpolate(x.c, x.b, 1, &Fu, U) != MFTL_COUPLING_OK) ok = 0;
      double s[4] = {0, 0, 0, 0};
      for (int64_t i = 0; i < nm; ++i)
      {
        s[0] += U[i] * F[i];
        s[1] += fabs(U[i] * F[i]);
      }
      for (int64_t k3 = blk.lo[2]; k3 < blk.hi[2]; ++k3)
        for (int64_t j3 = blk.lo[1]; j3 < blk.hi[1]; ++j3)
          for (int64_t i3 = blk.lo[0]; i3 < blk.hi[0]; ++i3)
          {
            const int64_t idx[3] = {i3, j3, k3};
            const int64_t e = block_elem(&blk, idx);
            s[2] += u.a[e] * blk.a[e];
            s[3] += fabs(u.a[e] * blk.a[e]);
          }
      MPI_Allreduce(MPI_IN_PLACE, s, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      if (!(fabs(s[0] - s[2]) <= 1e-13 * fmax(s[1], s[3])))
      {
        ok = 0;
        if (g_rank == 0) printf("\n  power identity: U.F %.17g vs u.b %.17g", s[0], s[2]);
      }
      free(U);
      free(u.a);
      free(expect);
      free(scale);
      free(all);
      free(mine);
      free(counts);
      free(displ);
      free(F);
      uncouple(&x);
      free_block(&blk);
      MFTLSolverGridDestroy(&g);
      MFTLEnsembleDestroy(&ens);
    }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}


/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: unstructured point samples
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The reference row of a point cloud: every sample of the GLOBAL cloud
 *        (generated by the test on every rank), at its minimum image about the
 *        marker, the metric replicated by the test's own Shepard average, and the
 *        row core applied to that candidate set. Shares the row arithmetic
 *        (tested on its own in test_coupling_kernels) and nothing else.
 */
static int ref_point_row(const MFTL_Domain *d, int64_t nc, double jitter, bool var_h, const double x[3], uint64_t *keys, double *w, int cap)
{
  const int64_t total = nc * nc * nc;
  const double Hc = 1.0 / (double) nc;
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  double H[3] = {Hc, Hc, Hc};
  if (var_h)
  {
    double sw = 0.0, sh = 0.0;
    for (int64_t c = 0; c < total; ++c)
    {
      double p[3], dv[3];
      uint64_t k;
      point_of_cell(nc, jitter, c, p, &k);
      MFTLGetMinimumImageVector(d, dv, x, p);
      const double hj = var_h_of(Hc, p);
      const double rho = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]) / (P.pilot_factor * hj);
      if (rho >= 1.0) continue;
      const double t = 1.0 - rho, wt = pow(Hc / hj, 3) * t * t * t * t * (4.0 * rho + 1.0);
      sw += wt;
      sh += wt * var_h_of(Hc, p);
    }
    H[0] = H[1] = H[2] = sh / sw;
  }
  int n = 0;
  double *X = (double *) malloc(3 * (size_t) cap * sizeof(double)), *q = (double *) malloc((size_t) cap * sizeof(double));
  for (int64_t c = 0; c < total; ++c)
  {
    double p[3], dv[3];
    uint64_t k;
    point_of_cell(nc, jitter, c, p, &k);
    MFTLGetMinimumImageVector(d, dv, x, p);
    if (fabs(dv[0]) >= 2.0 * H[0] || fabs(dv[1]) >= 2.0 * H[1] || fabs(dv[2]) >= 2.0 * H[2]) continue;
    if (n >= cap) break;
    for (int a = 0; a < 3; ++a) X[3 * n + a] = x[a] + dv[a];
    q[n] = Hc * Hc * Hc;
    keys[n] = k;
    ++n;
  }
  const int st = MFTLCouplingAssembleRow(&P, x, H, n, X, q, w, MFTL_NULL);
  free(X);
  free(q);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

static int test_point_clouds_match_their_reference(void)
{
  int ok = 1;
  long remote = 0;
  for (int per = 0; per < 2; ++per)
    for (int own = 0; own < 3; ++own)
      for (int var = 0; var < 2; ++var)
      {
        const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
        const bool pp[3] = {per != 0, per != 0, per != 0};
        MFTL_Ensemble *ens = make_ens(lo, hi, pp);
        const double c[3] = {per ? 0.93 : 0.5, 0.5, 0.5};
        add_sphere(ens, c, 0.27, 2);
        scatter(ens);
        MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
        pts_block pb;
        add_points(g, "centroids", 9, 0.6, (point_ownership) own, var != 0, &pb);
        coupled x;
        const int st = couple(ens, g, MFTL_NULL, 0.01, &x);
        if (!agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD))
        {
          if (g_rank == 0) printf("\n  couple: %s (per %d own %d var %d)", MFTLCouplingStatusName(st), per, own, var);
          ok = 0;
          teardown(&x, &g, &ens);
          free_points(&pb);
          continue;
        }
        const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
        const double *pos = MFTLCouplerBindingPositions(x.b);
        uint64_t rk[1024];
        double rw[1024], worst = 0.0;
        int bad = 0;
        for (int64_t i = 0; i < nm; ++i)
        {
          int n = 0;
          const uint64_t *keys;
          const double *w;
          MFTLCouplerBindingRow(x.b, 0, i, &n, &keys, &w);
          const int nr = ref_point_row(&ens->domain, 9, 0.6, var != 0, &pos[3 * i], rk, rw, 1024);
          int nz = 0;
          for (int e = 0; e < nr; ++e)
            if (rw[e] != 0.0) ++nz;
          if (nr < 0 || nz != n)
          {
            ++bad;
            continue;
          }
          for (int e = 0; e < n; ++e)
          {
            int f = -1;
            for (int r = 0; r < nr; ++r)
              if (rk[r] == keys[e]) f = r;
            if (f < 0)
            {
              ++bad;
              break;
            }
            worst = fmax(worst, fabs(rw[f] - w[e]));
          }
        }
        if (bad || worst > 1e-11)
        {
          ok = 0;
          printf("\n  rank %d per %d own %d var %d: %d bad rows, worst %.3e", g_rank, per, own, var, bad, worst);
        }
        MFTL_CouplerStats stt;
        MFTLCouplerGetStats(x.c, &stt);
        remote += stt.band_remote;
        teardown(&x, &g, &ens);
        free_points(&pb);
      }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(g_nproc == 1 || anywhere(remote > 0, MPI_COMM_WORLD));
  return 0;
}

/** @brief An affine field on a scattered, irregular cloud is interpolated exactly. */
static int test_point_clouds_reproduce_affine_fields(void)
{
  int ok = 1;
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.5, 0.5, 0.5};
  add_sphere(ens, c, 0.3, 3);
  scatter(ens);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  pts_block pb;
  add_points(g, "centroids", 10, 0.8, OWN_SCATTERED, true, &pb);
  affine A = {-0.3, {0.9, 1.7, -2.2}};
  for (int64_t i = 0; i < pb.n; ++i) pb.a[i] = f_affine(&pb.pos[3 * i], &A);
  coupled x;
  const int st = couple(ens, g, MFTL_NULL, 0.0, &x);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  MFTL_SolverField F = points_field(&pb, 1);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *pos = MFTLCouplerBindingPositions(x.b);
  for (int64_t i = 0; i < nm; ++i)
    if (!(fabs(U[i] - f_affine(&pos[3 * i], &A)) <= 1e-12)) ok = 0;
  free(U);
  teardown(&x, &g, &ens);
  free_points(&pb);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: staggered vectors, conservation
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A MAC grid: each velocity component on its own face lattice. An affine
 *        vector field is exact; spreading conserves force per component and
 *        torque across components (each component's first moment is its own
 *        location's); the power identity holds over the three locations.
 */
static int test_mac_vectors_force_and_torque(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.45, 0.55, 0.5};
  add_sphere(ens, c, 0.35, 3);
  scatter(ens);
  const int64_t N = 14;
  const double h = 1.0 / (double) N;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block f[3];
  const char *names[3] = {"u_faces", "v_faces", "w_faces"};
  for (int comp = 0; comp < 3; ++comp)
  {
    ref_lattice R;
    memset(&R, 0, sizeof R);
    for (int k = 0; k < 3; ++k)
    {
      R.N[k] = (k == comp) ? N + 1 : N;
      R.h[k] = h;
      R.origin[k] = (k == comp) ? 0.0 : 0.5 * h;
      R.L[k] = 1.0;
    }
    add_lattice(g, names[comp], &R, DEC_SLAB, 0, &f[comp]);
  }
  const double M[3][3] = {{0.3, -1.1, 0.7}, {2.0, 0.1, -0.4}, {-0.6, 0.9, -0.4}}, b0[3] = {0.2, -0.5, 1.1};
  affine comps[3];
  for (int k = 0; k < 3; ++k)
  {
    comps[k].a = b0[k];
    for (int j = 0; j < 3; ++j) comps[k].g[j] = M[k][j];
    fill_block(&f[k], f_affine, &comps[k], false);
  }
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.0, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  const double *pos = MFTLCouplerBindingPositions(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  double *U = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  MFTL_SolverField F[3] = {block_field(&f[0], 1, false), block_field(&f[1], 1, false), block_field(&f[2], 1, false)};
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 3, F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int ok = 1;
  for (int64_t i = 0; i < nm; ++i)
    for (int k = 0; k < 3; ++k)
      if (!(fabs(U[3 * i + k] - f_affine(&pos[3 * i], &comps[k])) <= 1e-12)) ok = 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* spread random vector loads */
  double *Ld = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  double s[14] = {0};
  for (int64_t i = 0; i < nm; ++i)
  {
    for (int k = 0; k < 3; ++k) Ld[3 * i + k] = hash01(ref[i].key * 5 + (uint64_t) k);
    for (int k = 0; k < 3; ++k) s[k] += Ld[3 * i + k];
    const double *xi = &pos[3 * i];
    s[3] += xi[1] * Ld[3 * i + 2] - xi[2] * Ld[3 * i + 1];
    s[4] += xi[2] * Ld[3 * i + 0] - xi[0] * Ld[3 * i + 2];
    s[5] += xi[0] * Ld[3 * i + 1] - xi[1] * Ld[3 * i + 0];
    for (int k = 0; k < 3; ++k) s[12] += U[3 * i + k] * Ld[3 * i + k];
  }
  lat_block T[3];
  MFTL_SolverField TF[3];
  for (int k = 0; k < 3; ++k)
  {
    T[k] = f[k];
    T[k].a = (double *) calloc((size_t) (f[k].n_elem > 0 ? f[k].n_elem : 1), sizeof(double));
    TF[k] = block_field(&T[k], 2, false);
  }
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 3, TF, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  for (int k = 0; k < 3; ++k)
    for (int64_t kk = T[k].lo[2]; kk < T[k].hi[2]; ++kk)
      for (int64_t jj = T[k].lo[1]; jj < T[k].hi[1]; ++jj)
        for (int64_t ii = T[k].lo[0]; ii < T[k].hi[0]; ++ii)
        {
          const int64_t idx[3] = {ii, jj, kk};
          const double bj = T[k].a[block_elem(&T[k], idx)];
          double X[3];
          for (int a = 0; a < 3; ++a) X[a] = T[k].ref.origin[a] + (double) idx[a] * h;
          s[6 + k] += bj;
          /* torque contributions: component k at its own location */
          if (k == 0)
          {
            s[10] += X[2] * bj;
            s[11] -= X[1] * bj;
          }
          if (k == 1)
          {
            s[11] += X[0] * bj;
            s[9] -= X[2] * bj;
          }
          if (k == 2)
          {
            s[9] += X[1] * bj;
            s[10] -= X[0] * bj;
          }
          s[13] += f[k].a[block_elem(&f[k], idx)] * bj;
        }
  MPI_Allreduce(MPI_IN_PLACE, s, 14, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (int k = 0; k < 3; ++k) TEST_ASSERT(fabs(s[k] - s[6 + k]) <= 1e-12 * (1.0 + fabs(s[k])));
  for (int k = 0; k < 3; ++k) TEST_ASSERT(fabs(s[3 + k] - s[9 + k]) <= 1e-12 * (1.0 + fabs(s[3 + k])));
  TEST_ASSERT(fabs(s[12] - s[13]) <= 1e-12 * (1.0 + fabs(s[12])));
  free(U);
  free(Ld);
  for (int k = 0; k < 3; ++k)
  {
    free(T[k].a);
    free_block(&f[k]);
  }
  teardown(&x, &g, &ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: routes
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Reading valid ghost layers in place is the same arithmetic as
 *        fetching from the owners -- bitwise -- with fewer bytes. And the route
 *        is real: with deliberately wrong ghosts the answer changes, which is
 *        what proves the halo was read at all.
 */
static int test_the_halo_route_changes_no_bit(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.5, 0.5, 0.5};
  add_sphere(ens, c, 0.3, 3);
  scatter(ens);
  ref_lattice R;
  memset(&R, 0, sizeof R);
  for (int k = 0; k < 3; ++k)
  {
    R.N[k] = 24;
    R.h[k] = 1.0 / 24.0;
    R.origin[k] = 0.5 * R.h[k];
    R.L[k] = 1.0;
    R.per[k] = true;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_ALIGNED, 2, &blk);
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.02, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  double *U[3];
  long bytes[3];
  for (int run = 0; run < 3; ++run)
  {
    U[run] = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
    fill_block(&blk, f_periodic, MFTL_NULL, run != 1);
    if (run == 2) /* lie: corrupt the ghosts and still claim they are valid */
      for (int64_t e = 0; e < blk.n_elem; ++e)
      {
        int64_t i[3] = {blk.alo[0] + e % blk.adims[0], blk.alo[1] + (e / blk.adims[0]) % blk.adims[1], blk.alo[2] + e / (blk.adims[0] * blk.adims[1])};
        bool owned = true;
        for (int k = 0; k < 3; ++k)
          if (i[k] < blk.lo[k] || i[k] >= blk.hi[k]) owned = false;
        if (!owned) blk.a[e] = 1e3;
      }
    MFTL_CouplerStats s0, s1;
    MFTLCouplerGetStats(x.c, &s0);
    MFTL_SolverField F = block_field(&blk, (uint64_t) run, run != 1);
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U[run]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerGetStats(x.c, &s1);
    bytes[run] = s1.bytes_fetch - s0.bytes_fetch;
  }
  int same = 1, differs = 0, nan = 0;
  for (int64_t i = 0; i < nm; ++i)
  {
    if (U[0][i] != U[1][i]) same = 0;
    if (U[2][i] != U[0][i]) differs = 1;
    if (!isfinite(U[1][i])) nan = 1;
  }
  MFTL_CouplerStats st;
  MFTLCouplerGetStats(x.c, &st);
  TEST_ASSERT(agreed(same && !nan, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(bytes[0] <= bytes[1], MPI_COMM_WORLD));
  if (g_nproc > 1)
  {
    TEST_ASSERT(anywhere(st.band_halo > 0, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(differs, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(bytes[0] < bytes[1], MPI_COMM_WORLD));
  }
  for (int run = 0; run < 3; ++run) free(U[run]);
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/**
 * @brief A kernel wider than a rank box (support_scale 4.5: half-width 9 h,
 *        more than a box of a 3x3x3 grid) reaches owners several boxes away;
 *        the rows still equal brute force.
 */
static int test_a_support_wider_than_a_rank_box(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.5, 0.5, 0.5};
  add_sphere(ens, c, 0.2, 2);
  scatter(ens);
  ref_lattice R;
  memset(&R, 0, sizeof R);
  for (int k = 0; k < 3; ++k)
  {
    R.N[k] = 24;
    R.h[k] = 1.0 / 24.0;
    R.origin[k] = 0.5 * R.h[k];
    R.L[k] = 1.0;
    R.per[k] = true;
  }
  R.scale = 4.5;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_ALIGNED, 0, &blk);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.support_scale = 4.5;
  coupled x;
  const int st = couple(ens, g, &P, 0.0, &x);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int bad = 0;
  const double err = rows_vs_reference(&x, 0, &R, &bad);
  if (bad || err > 1e-13) printf("\n  rank %d: %d bad rows, worst %.3e", g_rank, bad, err);
  TEST_ASSERT(agreed(bad == 0 && err <= 1e-13, MPI_COMM_WORLD));
  MFTL_CouplerStats s;
  MFTLCouplerGetStats(x.c, &s);
  if (g_nproc == 27) TEST_ASSERT(anywhere(s.fetch_peers >= 9, MPI_COMM_WORLD));
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests: reproducibility
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Interpolates f_periodic at the vertex markers of a fresh setup; returns U, keys. */
static int run_interpolation(MPI_Comm comm, double scale, decomposition dec, int64_t *n_out, double **U_out, uint64_t **keys_out)
{
  int rank, np;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {scale, scale, scale};
  bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(comm, cuts, lo, hi, pp, 4);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {0.5 * scale, 0.25 * scale, 0.5 * scale};
  MFTLGenerateSphere(obj, cc, 0.3 * scale, 3);
  scatter(ens);

  /* the lattice and its decomposition, computed for THIS communicator */
  const int64_t N = 20;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(comm);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "cells";
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = N;
    ld.lattice.spacing[k] = scale / (double) N;
    ld.lattice.origin[k] = 0.5 * scale / (double) N;
  }
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  int64_t olo[3], ohi[3];
  if (dec == DEC_SLAB)
  {
    olo[0] = olo[1] = 0;
    ohi[0] = ohi[1] = N;
    olo[2] = N * rank / np;
    ohi[2] = N * (rank + 1) / np;
  }
  else
  {
    int co[3];
    co[2] = rank % cuts[2];
    co[1] = (rank / cuts[2]) % cuts[1];
    co[0] = rank / (cuts[1] * cuts[2]);
    for (int k = 0; k < 3; ++k)
    {
      olo[k] = N * co[k] / cuts[k];
      ohi[k] = N * (co[k] + 1) / cuts[k];
    }
  }
  int64_t n_el = (ohi[0] - olo[0]) * (ohi[1] - olo[1]) * (ohi[2] - olo[2]);
  double *a = (double *) malloc((size_t) (n_el > 0 ? n_el : 1) * sizeof(double));
  if (n_el > 0)
  {
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = bd.array_lo[k] = olo[k];
      bd.owned_hi[k] = ohi[k];
      bd.array_dims[k] = ohi[k] - olo[k];
    }
    MFTLSolverGridAddBlock(g, &bd);
    int64_t e = 0;
    for (int64_t k3 = olo[2]; k3 < ohi[2]; ++k3)
      for (int64_t j3 = olo[1]; j3 < ohi[1]; ++j3)
        for (int64_t i3 = olo[0]; i3 < ohi[0]; ++i3)
        {
          /* the field is defined on the UNSCALED lattice, so it does not depend on the unit */
          const double xu[3] = {(i3 + 0.5) / (double) N, (j3 + 0.5) / (double) N, (k3 + 0.5) / (double) N};
          a[e++] = f_periodic(xu, MFTL_NULL);
        }
  }
  MFTL_Coupler *c = MFTL_NULL;
  int st = MFTLCouplerCreate(ens, g, MFTL_NULL, &c);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  MFTL_WindowSpec ws = {{0.01 * scale, 0.01 * scale, 0.01 * scale}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *win = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerRequestSupport(c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(c, req, &win);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(c, &ms, win, &b);
  MFTLCouplerReleaseRequest(&req);
  *n_out = MFTLCouplerBindingMarkerCount(b);
  *U_out = (double *) malloc((size_t) (*n_out > 0 ? *n_out : 1) * sizeof(double));
  *keys_out = (uint64_t *) malloc((size_t) (*n_out > 0 ? *n_out : 1) * sizeof(uint64_t));
  if (st == MFTL_COUPLING_OK)
  {
    MFTL_SolverField F;
    memset(&F, 0, sizeof F);
    F.location = loc;
    F.data = (n_el > 0) ? &a : MFTL_NULL;
    F.stride = 1;
    st = MFTLCouplerInterpolate(c, b, 1, &F, *U_out);
    const MFTL_MarkerRef *r = MFTLCouplerBindingMarkers(b);
    for (int64_t i = 0; i < *n_out; ++i) (*keys_out)[i] = r[i].key;
  }
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&win);
  MFTLCouplerDestroy(&c);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  free(a);
  return st;
}

/** @brief Sort (key, value) pairs by key, in place. */
static void sort_kv(uint64_t *k, double *v, int64_t n)
{
  for (int64_t i = 1; i < n; ++i)
  {
    const uint64_t kk = k[i];
    const double vv = v[i];
    int64_t j = i - 1;
    while (j >= 0 && k[j] > kk)
    {
      k[j + 1] = k[j];
      v[j + 1] = v[j];
      --j;
    }
    k[j + 1] = kk;
    v[j + 1] = vv;
  }
}

/**
 * @brief The interpolated values do not depend on the process count, the
 *        solver's decomposition or the length unit -- TO THE BIT. Rows are
 *        assembled over candidates in a global bin order and fetched values are
 *        copies, so nothing the partition decides reaches the arithmetic. The
 *        reference is the same setup on rank 0 alone, on MPI_COMM_SELF.
 */
static int test_results_are_bitwise_partition_and_unit_independent(void)
{
  int64_t n = 0;
  double *U = MFTL_NULL;
  uint64_t *keys = MFTL_NULL;
  int ok = 1;
  for (int variant = 0; variant < 3; ++variant)
  {
    const double scale = (variant == 2) ? ldexp(1.0, -20) : 1.0;
    const decomposition dec = (variant == 1) ? DEC_ALIGNED : DEC_SLAB;
    const int st = run_interpolation(MPI_COMM_WORLD, scale, dec, &n, &U, &keys);
    TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    /* gather (key, U) on rank 0 */
    double *mine = (double *) malloc((size_t) (2 * n > 0 ? 2 * n : 1) * sizeof(double));
    for (int64_t i = 0; i < n; ++i)
    {
      memcpy(&mine[2 * i], &keys[i], sizeof(double));
      mine[2 * i + 1] = U[i];
    }
    int tot = 0;
    double *all = allgather_doubles(mine, (int) n, 2, &tot);
    if (g_rank == 0)
    {
      int64_t ns = 0;
      double *Us = MFTL_NULL;
      uint64_t *ks = MFTL_NULL;
      const int st1 = run_interpolation(MPI_COMM_SELF, 1.0, DEC_SLAB, &ns, &Us, &ks);
      uint64_t *kd = (uint64_t *) malloc((size_t) (tot > 0 ? tot : 1) * sizeof(uint64_t));
      double *vd = (double *) malloc((size_t) (tot > 0 ? tot : 1) * sizeof(double));
      for (int i = 0; i < tot; ++i)
      {
        memcpy(&kd[i], &all[2 * i], sizeof(double));
        vd[i] = all[2 * i + 1];
      }
      sort_kv(kd, vd, tot);
      sort_kv(ks, Us, ns);
      if (st1 != MFTL_COUPLING_OK || ns != tot) ok = 0;
      for (int64_t i = 0; i < ns && ok; ++i)
        if (kd[i] != ks[i] || vd[i] != Us[i])
        {
          printf("\n  variant %d: key %llu: %.17g vs serial %.17g", variant, (unsigned long long) kd[i], vd[i], Us[i]);
          ok = 0;
        }
      free(kd);
      free(vd);
      free(Us);
      free(ks);
    }
    free(all);
    free(mine);
    free(U);
    free(keys);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief Chunked messages (40 bytes each) carry the same bits as whole ones,
 *        and two identical runs give identical bits.
 */
static int test_chunking_and_repetition_change_no_bit(void)
{
  int64_t n[3];
  double *U[3];
  uint64_t *k[3];
  for (int run = 0; run < 3; ++run)
  {
    MFTLCouplerSetChunkLimit(run == 2 ? 40 : 0);
    TEST_ASSERT(agreed(run_interpolation(MPI_COMM_WORLD, 1.0, DEC_SLAB, &n[run], &U[run], &k[run]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  }
  MFTLCouplerSetChunkLimit(0);
  int ok = (n[0] == n[1] && n[0] == n[2]);
  for (int64_t i = 0; ok && i < n[0]; ++i)
    if (U[0][i] != U[1][i] || U[0][i] != U[2][i] || k[0][i] != k[2][i]) ok = 0;
  for (int run = 0; run < 3; ++run)
  {
    free(U[run]);
    free(k[run]);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief The density form is the integrated form divided by each sample's
 *        measure, applied after accumulation; spreading ADDS to the target.
 */
static int test_density_form_and_additive_targets(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.5, 0.5, 0.5};
  add_sphere(ens, c, 0.3, 2);
  scatter(ens);
  ref_lattice R;
  memset(&R, 0, sizeof R);
  for (int k = 0; k < 3; ++k)
  {
    R.N[k] = 12;
    R.h[k] = 1.0 / 12.0;
    R.origin[k] = 0.5 * R.h[k];
    R.L[k] = 1.0;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_REVERSED, 0, &blk);
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.0, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  double *L = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i) L[i] = ref[i].weight * (1.0 + hash01(ref[i].key));
  double *I = (double *) calloc((size_t) (blk.n_elem > 0 ? blk.n_elem : 1), sizeof(double));
  double *D = (double *) malloc((size_t) (blk.n_elem > 0 ? blk.n_elem : 1) * sizeof(double));
  for (int64_t e = 0; e < blk.n_elem; ++e) D[e] = 5.0; /* additive: the target keeps what it had */
  lat_block bi = blk, bd = blk;
  bi.a = I;
  bd.a = D;
  MFTL_SolverField FI = block_field(&bi, 1, false), FD = block_field(&bd, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &FI, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &FD, L, MFTL_SPREAD_DENSITY) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double q = R.h[0] * R.h[1] * R.h[2];
  int ok = 1;
  for (int64_t e = 0; e < blk.n_elem; ++e)
    if (!(fabs((D[e] - 5.0) * q - I[e]) <= 1e-14 * (1.0 + fabs(I[e])))) ok = 0;
  free(L);
  free(I);
  free(D);
  free_block(&blk);
  teardown(&x, &g, &ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/** @brief Vertex dual areas and centroid areas both sum to the surface area. */
static int test_marker_weights_sum_to_the_area(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.97, 0.5, 0.03};
  add_sphere(ens, c, 0.3, 3);
  scatter(ens);
  ref_lattice R;
  memset(&R, 0, sizeof R);
  for (int k = 0; k < 3; ++k)
  {
    R.N[k] = 16;
    R.h[k] = 1.0 / 16.0;
    R.origin[k] = 0.5 * R.h[k];
    R.L[k] = 1.0;
    R.per[k] = true;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
  MFTL_Coupler *cpl = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cpl) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double s[3] = {0, 0, 0};
  for (int kind = 0; kind < 2; ++kind)
  {
    MFTL_MarkerSpec ms;
    memset(&ms, 0, sizeof ms);
    ms.kind = (kind == 0) ? MFTL_MARKERS_VERTICES : MFTL_MARKERS_CENTROIDS;
    MFTL_WindowSpec ws = {{0, 0, 0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cpl, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cpl, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerBind(cpl, &ms, w, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const MFTL_MarkerRef *r = MFTLCouplerBindingMarkers(b);
    for (int64_t i = 0; i < MFTLCouplerBindingMarkerCount(b); ++i) s[kind] += r[i].weight;
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
    MFTLCouplerReleaseRequest(&req);
  }
  for (int o = 0; o < ens->n_local_objects; ++o)
    for (int t = 0; t < ens->local_objects[o]->tris.n_local; ++t) s[2] += MFTLGetTriangleArea(ens->local_objects[o], t);
  MPI_Allreduce(MPI_IN_PLACE, s, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(fabs(s[0] - s[2]) <= 1e-12 * s[2] && fabs(s[1] - s[2]) <= 1e-12 * s[2]);
  MFTLCouplerDestroy(&cpl);
  free_block(&blk);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A stretched (rectilinear) lattice: per-axis coordinate tables, a
 *        variable metric, LINEAR rows exact for affine fields, and the power
 *        identity.
 */
static int test_a_stretched_lattice(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double c[3] = {0.5, 0.5, 0.5};
  add_sphere(ens, c, 0.3, 3);
  scatter(ens);
  const int64_t N = 18;
  double *faces = (double *) malloc((size_t) (N + 1) * sizeof(double)), *coords = (double *) malloc((size_t) N * sizeof(double));
  for (int64_t i = 0; i <= N; ++i)
  {
    const double s = (double) i / (double) N;
    faces[i] = s - 0.12 * sin(2.0 * MFTL_PI * s) / (2.0 * MFTL_PI); /* finer at the walls, coarser inside */
  }
  faces[0] = 0.0;
  faces[N] = 1.0;
  for (int64_t i = 0; i < N; ++i) coords[i] = 0.5 * (faces[i] + faces[i + 1]);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "stretched";
  ld.lattice.kind = MFTL_LATTICE_RECTILINEAR;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = N;
    ld.lattice.coords[k] = coords;
    ld.lattice.faces[k] = faces;
  }
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  int64_t olo[3], ohi[3];
  const int64_t NN[3] = {N, N, N};
  owned_box(NN, DEC_SLAB, olo, ohi);
  const int64_t n_el = (ohi[0] - olo[0]) * (ohi[1] - olo[1]) * (ohi[2] - olo[2]);
  double *a = (double *) malloc((size_t) (n_el > 0 ? n_el : 1) * sizeof(double));
  affine A = {0.25, {1.5, -0.75, 2.5}};
  if (n_el > 0)
  {
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = bd.array_lo[k] = olo[k];
      bd.owned_hi[k] = ohi[k];
      bd.array_dims[k] = ohi[k] - olo[k];
    }
    TEST_ASSERT(MFTLSolverGridAddBlock(g, &bd) == 0);
    int64_t e = 0;
    for (int64_t k3 = olo[2]; k3 < ohi[2]; ++k3)
      for (int64_t j3 = olo[1]; j3 < ohi[1]; ++j3)
        for (int64_t i3 = olo[0]; i3 < ohi[0]; ++i3)
        {
          const double x[3] = {coords[i3], coords[j3], coords[k3]};
          a[e++] = f_affine(x, &A);
        }
  }
  coupled x;
  const int st = couple(ens, g, MFTL_NULL, 0.0, &x);
  TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = loc;
  F.data = (n_el > 0) ? &a : MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *pos = MFTLCouplerBindingPositions(x.b);
  int ok = 1;
  for (int64_t i = 0; i < nm; ++i)
    if (!(fabs(U[i] - f_affine(&pos[3 * i], &A)) <= 1e-12)) ok = 0;
  MFTL_BindingQuality q;
  MFTLCouplerBindingQuality(x.b, &q);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* the metric varies, so rows away from a perfect lattice needed corrections */
  TEST_ASSERT(anywhere(q.n_corrected > 0, MPI_COMM_WORLD));
  free(U);
  free(a);
  free(faces);
  free(coords);
  teardown(&x, &g, &ens);
  return 0;
}

/** @brief A remote coarse sample cannot change a fine region's metric or halo.
 * Full providers deliberately claim the whole domain: their ownership bounds
 * must not be reused as uniform resolution bounds. */
static int test_distant_coarsening_does_not_expand_a_fine_band(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1}, region[6] = {0, 0, 0, 1, 1, 1};
  const bool periodic[3] = {false, false, false};
  enum { N = 24, TOTAL = N * N * N, NM = 12 };
  double previous[NM];
  long previous_band = -1;
  int ok = 1;
  for (int pass = 0; pass < 2; ++pass)
  {
    MFTL_Ensemble *ens = make_ens(lo, hi, periodic);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    MFTL_LocationDesc ld = {0};
    ld.name = "graded";
    const int loc = MFTLSolverGridAddLocation(g, &ld);
    const int cap = TOTAL / g_nproc + 1;
    double *pos = malloc(3 * (size_t) cap * sizeof(double)), *h = malloc(3 * (size_t) cap * sizeof(double));
    double *q = malloc((size_t) cap * sizeof(double)), *v = malloc((size_t) cap * sizeof(double));
    uint64_t *keys = malloc((size_t) cap * sizeof(uint64_t));
    int n = 0;
    for (int i = g_rank; i < TOTAL; i += g_nproc)
    {
      const int idx[3] = {i % N, (i / N) % N, i / (N * N)};
      keys[n] = (uint64_t) i;
      q[n] = 1.0 / TOTAL;
      for (int k = 0; k < 3; ++k)
      {
        pos[3 * n + k] = (idx[k] + 0.5) / N;
        h[3 * n + k] = i == TOTAL - 1 ? (pass ? 0.42 : 0.18) : 1.0 / N;
      }
      v[n] = f_periodic(&pos[3 * n], MFTL_NULL);
      ++n;
    }
    MFTL_PointSetDesc pd = {0};
    pd.location = loc; pd.n = n; pd.keys = keys; pd.pos = pos; pd.h = h; pd.measure = q;
    TEST_ASSERT(agreed(MFTLSolverGridAddPoints(g, &pd) >= 0, MPI_COMM_WORLD));
    MFTL_ProviderRegions pr = {0};
    pr.location = loc; pr.n_boxes = 1; pr.boxes = region;
    TEST_ASSERT(agreed(MFTLSolverGridSetProviderRegions(g, &pr) == MFTL_SUCCESS, MPI_COMM_WORLD));
    MFTL_Coupler *c = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    double points[3 * NM], values[NM];
    for (int i = 0; i < NM; ++i)
      for (int k = 0; k < 3; ++k) points[3 * i + k] = 0.19 + 0.035 * (1.0 + hash01((uint64_t) (i * 3 + k)));
    MFTL_MarkerSpec ms = {0};
    ms.kind = MFTL_MARKERS_USER; ms.n_user = g_rank == g_nproc - 1 ? NM : 0; ms.user_pos = points;
    MFTL_WindowSpec ws = {{0.01, 0.01, 0.01}};
    MFTL_SupportRequest *req = MFTL_NULL;
    MFTL_CouplingWindow *w = MFTL_NULL;
    MFTL_CouplingBinding *b = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(c, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SolverField field = {0};
    field.location = loc; field.data = &v;
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &field, values) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_CouplerStats stats;
    MFTLCouplerGetStats(c, &stats);
    double moved[3 * NM], stage_values[NM];
    for (int i = 0; i < 3 * NM; ++i) moved[i] = points[i] + 0.008;
    MFTL_StageGeometry stage = {0};
    stage.pos = moved;
    MFTL_CouplingBinding *stage_binding = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerBindStage(c, b, &stage, &stage_binding) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, stage_binding, 1, &field, stage_values) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_CouplerStats reused;
    MFTLCouplerGetStats(c, &reused);
    if (reused.n_candidate_builds != stats.n_candidate_builds || reused.n_candidate_reuses < stats.n_candidate_reuses + ms.n_user) ok = 0;
    ref_lattice lattice = {0};
    for (int k = 0; k < 3; ++k)
    {
      lattice.N[k] = N; lattice.h[k] = 1.0 / N; lattice.origin[k] = 0.5 / N; lattice.L[k] = 1.0;
    }
    for (int i = 0; i < ms.n_user; ++i)
    {
      uint64_t rk[128];
      double rw[128], offsets[3 * 128], expect = 0.0;
      const int nr = ref_row(&lattice, &moved[3 * i], rk, rw, offsets, 128);
      if (nr <= 0) ok = 0;
      for (int j = 0; j < nr; ++j)
      {
        const double xp[3] = {(rk[j] % N + 0.5) / N, ((rk[j] / N) % N + 0.5) / N, (rk[j] / (N * N) + 0.5) / N};
        expect += rw[j] * f_periodic(xp, MFTL_NULL);
      }
      if (fabs(expect - stage_values[i]) > 2e-13) ok = 0;
    }
    MFTLCouplerReleaseBinding(&stage_binding);
    const long band = stats.band_self + stats.band_remote;
    if (band > TOTAL / 10) ok = 0;
    if (pass)
    {
      if (band != previous_band) ok = 0;
      if (ms.n_user && memcmp(previous, values, sizeof values)) ok = 0;
    }
    else
    {
      previous_band = band;
      if (ms.n_user) memcpy(previous, values, sizeof values);
    }
    MFTLCouplerReleaseBinding(&b); MFTLCouplerReleaseWindow(&w); MFTLCouplerReleaseRequest(&req);
    MFTLCouplerDestroy(&c); MFTLSolverGridDestroy(&g); MFTLEnsembleDestroy(&ens);
    free(pos); free(h); free(q); free(v); free(keys);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling (np = %d) ===\n", g_nproc);

  RUN_TEST(test_distant_coarsening_does_not_expand_a_fine_band);
  RUN_TEST(test_rows_match_brute_force);
  RUN_TEST(test_linear_fields_are_reproduced);
  RUN_TEST(test_spread_is_the_brute_force_transpose);
  RUN_TEST(test_point_clouds_match_their_reference);
  RUN_TEST(test_point_clouds_reproduce_affine_fields);
  RUN_TEST(test_mac_vectors_force_and_torque);
  RUN_TEST(test_the_halo_route_changes_no_bit);
  RUN_TEST(test_a_support_wider_than_a_rank_box);
  RUN_TEST(test_results_are_bitwise_partition_and_unit_independent);
  RUN_TEST(test_chunking_and_repetition_change_no_bit);
  RUN_TEST(test_density_form_and_additive_targets);
  RUN_TEST(test_marker_weights_sum_to_the_area);
  RUN_TEST(test_a_stretched_lattice);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
