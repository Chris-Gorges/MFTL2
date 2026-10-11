/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief The coupler's contracts, as opposed to its numerics: coverage,
 *        selection, staleness, failure and agreement.
 *
 * Every test here tries to make the coupler return a plausible answer it is
 * not entitled to -- a row over a support with a hole in it, an unprovided
 * region read as a wall, a binding applied after the surface moved, a status
 * that differs between ranks -- and asserts that it refuses, on EVERY rank,
 * before anything is written. A coupler that renormalises over what happened
 * to arrive produces numbers that look right; these tests are what stop it.
 */

#include "test_coupling_fixtures.h"

/** @brief A walled unit box with a sphere scattered over the ranks. */
static MFTL_Ensemble *sphere_ens(const double c[3], double r, int refine, bool periodic)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {periodic, periodic, periodic};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  add_sphere(ens, c, r, refine);
  scatter(ens);
  return ens;
}

static ref_lattice cells(int64_t N, bool periodic)
{
  ref_lattice R;
  memset(&R, 0, sizeof R);
  for (int k = 0; k < 3; ++k)
  {
    R.N[k] = N;
    R.h[k] = 1.0 / (double) N;
    R.origin[k] = 0.5 * R.h[k];
    R.L[k] = 1.0;
    R.per[k] = periodic;
  }
  return R;
}

static MFTL_MarkerSpec vertices(void)
{
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  return ms;
}

/** @brief Interpolates and returns U sorted by marker key (for comparison across runs). */
static int interpolate_sorted(coupled *x, int nf, MFTL_SolverField *F, int64_t *n, uint64_t **keys, double **U)
{
  *n = MFTLCouplerBindingMarkerCount(x->b);
  *U = (double *) malloc((size_t) (*n > 0 ? *n * nf : 1) * sizeof(double));
  *keys = (uint64_t *) malloc((size_t) (*n > 0 ? *n : 1) * sizeof(uint64_t));
  const int st = MFTLCouplerInterpolate(x->c, x->b, nf, F, *U);
  const MFTL_MarkerRef *r = MFTLCouplerBindingMarkers(x->b);
  for (int64_t i = 0; i < *n; ++i) (*keys)[i] = r[i].key;
  return st;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Selection and coverage
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A SELECTED band built by the request protocol alone -- start with
 *        nothing exported, export exactly what each NEED_COVERAGE names, retry
 *        -- converges, and then gives the FULL answer to the bit while exporting
 *        a fraction of the grid. Selection-only updates keep the directory.
 */
static int test_a_selected_band_built_from_requests_equals_full(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  int64_t nf = 0, ns = 0;
  uint64_t *kf = MFTL_NULL, *ks = MFTL_NULL;
  double *Uf = MFTL_NULL, *Us = MFTL_NULL;
  affine A = {0.1, {0.2, 0.3, -0.4}};
  long exported_total = 0, owned_total = 0, rounds = 0;
  MFTL_CouplerStats stats;
  memset(&stats, 0, sizeof stats);

  for (int sel = 0; sel < 2; ++sel)
  {
    MFTL_Ensemble *ens = sphere_ens(c, 0.25, 3, false);
    ref_lattice R = cells(24, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
    fill_block(&blk, f_affine, &A, false);
    int n_boxes = 0, cap = 64;
    int64_t *boxes = (int64_t *) malloc(6 * (size_t) cap * sizeof(int64_t));
    if (sel)
    {
      MFTL_CoverageDesc cv = {blk.loc, MFTL_GRID_SELECTED, 0, MFTL_NULL, MFTL_NULL};
      MFTLSolverGridSetCoverage(g, &cv);
    }
    coupled x;
    memset(&x, 0, sizeof x);
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &x.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms = vertices();
    MFTL_WindowSpec ws = {{0.01, 0.01, 0.01}};
    int st = MFTL_COUPLING_NEED_COVERAGE;
    MFTL_SupportRequest *req = MFTL_NULL;
    for (int round = 0; round < 6 && st == MFTL_COUPLING_NEED_COVERAGE; ++round)
    {
      MFTLCouplerReleaseRequest(&req);
      st = MFTLCouplerRequestSupport(x.c, &ms, &ws, &req);
      if (st != MFTL_COUPLING_NEED_COVERAGE) break;
      ++rounds;
      MFTL_RequestView v;
      MFTLCouplerGetRequests(req, &v);
      TEST_ASSERT(agreed(v.n_unprovided == 0, MPI_COMM_WORLD)); /* everything is owned; only unexported */
      for (int i = 0; i < v.n_missing; ++i)
      {
        if (n_boxes == cap)
        {
          cap *= 2;
          boxes = (int64_t *) realloc(boxes, 6 * (size_t) cap * sizeof(int64_t));
        }
        for (int k = 0; k < 3; ++k)
        {
          boxes[6 * n_boxes + k] = v.missing[i].index_lo[k];
          boxes[6 * n_boxes + 3 + k] = v.missing[i].index_hi[k];
        }
        ++n_boxes;
      }
      MFTL_CoverageDesc cv = {blk.loc, MFTL_GRID_SELECTED, n_boxes, boxes, MFTL_NULL};
      MFTLSolverGridSetCoverage(g, &cv);
      TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(x.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    }
    TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(x.c, req, &x.win) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    TEST_ASSERT(agreed(MFTLCouplerBind(x.c, &ms, x.win, &x.b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SolverField F = block_field(&blk, 1, false);
    TEST_ASSERT(agreed(interpolate_sorted(&x, 1, &F, sel ? &ns : &nf, sel ? &ks : &kf, sel ? &Us : &Uf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    if (sel)
    {
      MFTLCouplerGetStats(x.c, &stats);
      /* USED within EXPORTED, sample by sample */
      if (blk.piece >= 0)
      {
        const int64_t n = (blk.hi[0] - blk.lo[0]) * (blk.hi[1] - blk.lo[1]) * (blk.hi[2] - blk.lo[2]);
        unsigned char *used = (unsigned char *) malloc((size_t) n), *exp = (unsigned char *) malloc((size_t) n);
        MFTLCouplerBandMask(x.win, blk.loc, blk.piece, MFTL_BAND_USED, used);
        MFTLCouplerBandMask(x.win, blk.loc, blk.piece, MFTL_BAND_EXPORTED, exp);
        int sub = 1;
        for (int64_t i = 0; i < n; ++i)
        {
          if (used[i] && !exp[i]) sub = 0;
          exported_total += exp[i];
        }
        owned_total += (long) n;
        TEST_ASSERT(sub);
        free(used);
        free(exp);
      }
    }
    free(boxes);
    free_block(&blk);
    teardown(&x, &g, &ens);
  }
  /* bit-identical, marker by marker (same partition, same order) */
  int same = (nf == ns);
  for (int64_t i = 0; same && i < nf; ++i)
    if (kf[i] != ks[i] || Uf[i] != Us[i]) same = 0;
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  long tot[3] = {exported_total, owned_total, rounds};
  MPI_Allreduce(MPI_IN_PLACE, tot, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("\n  selected band: %ld of %ld samples exported (%.1f %%), %ld request rounds", tot[0], tot[1], 100.0 * tot[0] / tot[1], tot[2]);
  TEST_ASSERT(tot[2] > 0);                    /* the protocol actually ran */
  TEST_ASSERT(tot[0] > 0 && tot[0] < tot[1] / 2); /* and exported a band, not the grid */
  TEST_ASSERT(agreed(stats.n_directory_builds == 1 && stats.n_directory_kept > 0, MPI_COMM_WORLD));
  free(kf);
  free(ks);
  free(Uf);
  free(Us);
  return 0;
}

/**
 * @brief A single unexported cell inside a SELECTED export is found exactly
 *        -- the request comes back NEED_COVERAGE and the owner is told that one
 *        index -- even though the remaining samples would give a perfectly
 *        conditioned linear fit. The same cell MASKED (no such sample: a solid)
 *        is not missing, and its rows are corrected around it.
 */
static int test_a_one_cell_hole_is_found_and_a_masked_cell_is_not(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  const double r = 0.25;
  const int64_t N = 20;
  /* the cell containing the vertex at (0.5 + r, 0.5, 0.5): certainly in a support */
  const int64_t hole[3] = {(int64_t) ((0.5 + r) * N), (int64_t) (0.5 * N), (int64_t) (0.5 * N)};
  for (int variant = 0; variant < 2; ++variant)
  {
    MFTL_Ensemble *ens = sphere_ens(c, r, 3, false);
    ref_lattice R = cells(N, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    memset(&blk, 0, sizeof blk);
    /* register by hand: the hole is a coverage gap (variant 0) or a mask (variant 1) */
    MFTL_LocationDesc ld;
    memset(&ld, 0, sizeof ld);
    ld.name = "cells";
    ld.lattice.kind = MFTL_LATTICE_UNIFORM;
    for (int k = 0; k < 3; ++k)
    {
      ld.lattice.dims[k] = N;
      ld.lattice.spacing[k] = R.h[k];
      ld.lattice.origin[k] = R.origin[k];
    }
    const int loc = MFTLSolverGridAddLocation(g, &ld);
    int64_t lo[3], hi[3];
    owned_box(R.N, DEC_SLAB, lo, hi);
    bool mine = true, empty = false;
    for (int k = 0; k < 3; ++k)
    {
      if (hole[k] < lo[k] || hole[k] >= hi[k]) mine = false;
      if (hi[k] <= lo[k]) empty = true;
    }
    const int64_t nx = hi[0] - lo[0], ny = hi[1] - lo[1], nz = hi[2] - lo[2];
    unsigned char *mask = MFTL_NULL;
    if (!empty)
    {
      MFTL_BlockDesc bd;
      memset(&bd, 0, sizeof bd);
      bd.location = loc;
      for (int k = 0; k < 3; ++k)
      {
        bd.owned_lo[k] = bd.array_lo[k] = lo[k];
        bd.owned_hi[k] = hi[k];
        bd.array_dims[k] = hi[k] - lo[k];
      }
      if (variant == 1 && mine)
      {
        mask = (unsigned char *) malloc((size_t) (nx * ny * nz));
        memset(mask, 1, (size_t) (nx * ny * nz));
        mask[(hole[0] - lo[0]) + nx * ((hole[1] - lo[1]) + ny * (hole[2] - lo[2]))] = 0;
        bd.active = mask;
      }
      MFTLSolverGridAddBlock(g, &bd);
      if (variant == 0)
      {
        /* the owned box minus the hole: six boxes */
        int64_t b[6 * 6];
        int nb = 0;
        if (mine)
        {
          for (int k = 0; k < 3; ++k)
          {
            int64_t lo1[3], hi1[3], lo2[3], hi2[3];
            for (int q = 0; q < 3; ++q)
            {
              lo1[q] = lo2[q] = (q < k) ? hole[q] : lo[q];
              hi1[q] = hi2[q] = (q < k) ? hole[q] + 1 : hi[q];
            }
            hi1[k] = hole[k];
            lo2[k] = hole[k] + 1;
            for (int q = 0; q < 3; ++q)
            {
              b[6 * nb + q] = lo1[q];
              b[6 * nb + 3 + q] = hi1[q];
            }
            ++nb;
            for (int q = 0; q < 3; ++q)
            {
              b[6 * nb + q] = lo2[q];
              b[6 * nb + 3 + q] = hi2[q];
            }
            ++nb;
          }
        }
        else
        {
          for (int q = 0; q < 3; ++q)
          {
            b[q] = lo[q];
            b[3 + q] = hi[q];
          }
          nb = 1;
        }
        MFTL_CoverageDesc cv = {loc, MFTL_GRID_SELECTED, nb, b, MFTL_NULL};
        MFTLSolverGridSetCoverage(g, &cv);
      }
    }
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms = vertices();
    MFTL_WindowSpec ws = {{0, 0, 0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    const int st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
    if (variant == 0)
    {
      TEST_ASSERT(agreed(st == MFTL_COUPLING_NEED_COVERAGE, MPI_COMM_WORLD));
      MFTL_RequestView v;
      MFTLCouplerGetRequests(req, &v);
      int exact = 1;
      for (int i = 0; i < v.n_missing; ++i)
        for (int k = 0; k < 3; ++k)
          if (v.missing[i].index_lo[k] != hole[k] || v.missing[i].index_hi[k] != hole[k] + 1) exact = 0;
      TEST_ASSERT(agreed(exact && (!mine || v.n_missing > 0), MPI_COMM_WORLD));
      /* and no window can be made of it */
      MFTL_CouplingWindow *w = MFTL_NULL;
      TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_NEED_COVERAGE && w == MFTL_NULL, MPI_COMM_WORLD));
    }
    else
    {
      TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      MFTL_CouplingWindow *w = MFTL_NULL;
      MFTL_CouplingBinding *b = MFTL_NULL;
      TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      /* no row names the masked key, and the rows around it were corrected */
      const uint64_t hk = (uint64_t) (hole[0] + N * (hole[1] + N * hole[2]));
      int named = 0;
      for (int64_t i = 0; i < MFTLCouplerBindingMarkerCount(b); ++i)
      {
        int n;
        const uint64_t *keys;
        const double *wt;
        MFTLCouplerBindingRow(b, 0, i, &n, &keys, &wt);
        for (int e = 0; e < n; ++e)
          if (keys[e] == hk) named = 1;
      }
      MFTL_BindingQuality q;
      MFTLCouplerBindingQuality(b, &q);
      TEST_ASSERT(agreed(!named, MPI_COMM_WORLD));
      TEST_ASSERT(anywhere(q.n_corrected > 0, MPI_COMM_WORLD));
      MFTLCouplerReleaseBinding(&b);
      MFTLCouplerReleaseWindow(&w);
    }
    MFTLCouplerReleaseRequest(&req);
    MFTLCouplerDestroy(&cp);
    free(mask);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/**
 * @brief A part of the lattice nobody registered is UNPROVIDED: never a wall,
 *        never a truncated row. The same for a point location whose clouds
 *        leave a region of the domain unclaimed.
 */
static int test_an_ownership_gap_is_never_a_wall(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  for (int kind = 0; kind < 2; ++kind)
  {
    MFTL_Ensemble *ens = sphere_ens(c, 0.3, 2, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    pts_block pb;
    memset(&pb, 0, sizeof pb);
    if (kind == 0)
    {
      /* the lattice declares 20^3 samples, but the slab 8 <= k < 12 is registered by nobody */
      ref_lattice R = cells(20, false);
      MFTL_LocationDesc ld;
      memset(&ld, 0, sizeof ld);
      ld.name = "cells";
      ld.lattice.kind = MFTL_LATTICE_UNIFORM;
      for (int k = 0; k < 3; ++k)
      {
        ld.lattice.dims[k] = 20;
        ld.lattice.spacing[k] = R.h[k];
        ld.lattice.origin[k] = R.origin[k];
      }
      const int loc = MFTLSolverGridAddLocation(g, &ld);
      if (g_rank == 0)
        for (int part = 0; part < 2; ++part)
        {
          MFTL_BlockDesc bd;
          memset(&bd, 0, sizeof bd);
          bd.location = loc;
          for (int k = 0; k < 3; ++k)
          {
            bd.owned_lo[k] = bd.array_lo[k] = 0;
            bd.owned_hi[k] = 20;
            bd.array_dims[k] = 20;
          }
          bd.owned_lo[2] = (part == 0) ? 0 : 12;
          bd.owned_hi[2] = (part == 0) ? 8 : 20;
          bd.array_lo[2] = 0;
          bd.array_dims[2] = 20;
          TEST_ASSERT(MFTLSolverGridAddBlock(g, &bd) >= 0);
        }
      (void) blk;
    }
    else
    {
      /* Two point sets, below z = 0.4 and above z = 0.6, and nobody claims the
       * slab between: a forgotten registration. (Inside ONE set, a hole thinner
       * than a 64-sample tile is claimed by the tile's box -- certified empty,
       * not unprovided; the header says so, and ProviderRegions make it exact.) */
      MFTL_LocationDesc ld;
      memset(&ld, 0, sizeof ld);
      ld.name = "points";
      const int loc = MFTLSolverGridAddLocation(g, &ld);
      const int64_t nc = 10;
      const double H = 0.1;
      uint64_t *key = (uint64_t *) malloc(1000 * sizeof(uint64_t));
      double *pos = (double *) malloc(3000 * sizeof(double)), *meas = (double *) malloc(1000 * sizeof(double));
      for (int part = 0; part < 2; ++part)
      {
        int64_t n = 0;
        if (g_rank == 0)
          for (int64_t cc = 0; cc < nc * nc * nc; ++cc)
          {
            double x[3];
            uint64_t k;
            point_of_cell(nc, 0.0, cc, x, &k);
            if ((part == 0) ? !(x[2] < 0.4) : !(x[2] > 0.6)) continue;
            key[n] = k;
            memcpy(&pos[3 * n], x, sizeof x);
            meas[n] = H * H * H;
            ++n;
          }
        MFTL_PointSetDesc pd;
        memset(&pd, 0, sizeof pd);
        pd.location = loc;
        pd.n = n;
        pd.keys = key;
        pd.pos = pos;
        pd.measure = meas;
        pd.h_uniform[0] = pd.h_uniform[1] = pd.h_uniform[2] = H;
        double ext[3000];
        for (int64_t i = 0; i < n; ++i) ext[3 * i] = ext[3 * i + 1] = ext[3 * i + 2] = 0.5 * H;
        pd.extent = ext;
        TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) >= 0);
      }
      free(key);
      free(pos);
      free(meas);
    }
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms = vertices();
    MFTL_WindowSpec ws = {{0, 0, 0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    const int st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
    if (st != MFTL_COUPLING_NEED_COVERAGE && g_rank == 0) printf("\n  kind %d: %s", kind, MFTLCouplingStatusName(st));
    TEST_ASSERT(agreed(st == MFTL_COUPLING_NEED_COVERAGE, MPI_COMM_WORLD));
    MFTL_RequestView v;
    MFTLCouplerGetRequests(req, &v);
    /* the requester reports regions nobody owns, all inside the gap's slab */
    int inside = 1;
    for (int i = 0; i < v.n_unprovided; ++i)
      if (v.unprovided[i].box[5] < 0.35 || v.unprovided[i].box[2] > 0.65) inside = 0;
    TEST_ASSERT(agreed(inside, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(v.n_unprovided > 0, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    MFTLCouplerDestroy(&cp);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/** @brief Two owners of one sample: INVALID on every rank, for lattices and for points. */
static int test_duplicate_ownership_is_invalid(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  /* on one rank: overlapping blocks and duplicate keys are refused at registration */
  {
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    ref_lattice R = cells(8, false);
    lat_block a;
    add_lattice(g, "cells", &R, DEC_ROOT, 0, &a);
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = a.loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = bd.array_lo[k] = 2;
      bd.owned_hi[k] = 5;
      bd.array_dims[k] = 3;
    }
    if (g_rank == 0) TEST_ASSERT(MFTLSolverGridAddBlock(g, &bd) == MFTL_ERROR);
    free_block(&a);
    MFTLSolverGridDestroy(&g);
  }
  if (g_nproc < 2) return 0;
  for (int kind = 0; kind < 2; ++kind)
  {
    MFTL_Ensemble *ens = sphere_ens(c, 0.3, 2, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    pts_block pb;
    if (kind == 0)
    {
      /* every rank claims the whole lattice */
      ref_lattice R = cells(10, false);
      add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
      MFTL_BlockDesc bd;
      memset(&bd, 0, sizeof bd);
      bd.location = blk.loc;
      for (int k = 0; k < 3; ++k)
      {
        bd.owned_lo[k] = bd.array_lo[k] = 0;
        bd.owned_hi[k] = 10;
        bd.array_dims[k] = 10;
      }
      if (blk.piece < 0) MFTLSolverGridAddBlock(g, &bd); /* ranks owning no slab claim everything */
      MFTLSolverGridReset(g);
      MFTLSolverGridAddBlock(g, &bd);
    }
    else
    {
      add_points(g, "points", 8, 0.3, OWN_SLAB, false, &pb);
      /* rank 1 also registers rank 0's first sample under the same key */
      MFTLSolverGridReset(g);
      double x[3];
      uint64_t k0;
      point_of_cell(8, 0.3, 0, x, &k0);
      const int64_t n = (g_rank == 1) ? pb.n + 1 : pb.n;
      uint64_t *k = (uint64_t *) malloc((size_t) (n + 1) * sizeof(uint64_t));
      double *p = (double *) malloc(3 * (size_t) (n + 1) * sizeof(double)), *m = (double *) malloc((size_t) (n + 1) * sizeof(double));
      memcpy(k, pb.key, (size_t) pb.n * sizeof(uint64_t));
      memcpy(p, pb.pos, 3 * (size_t) pb.n * sizeof(double));
      memcpy(m, pb.meas, (size_t) pb.n * sizeof(double));
      if (g_rank == 1)
      {
        k[pb.n] = k0;
        memcpy(&p[3 * pb.n], x, sizeof x);
        m[pb.n] = 1.0 / 512.0;
      }
      MFTL_PointSetDesc pd;
      memset(&pd, 0, sizeof pd);
      pd.location = pb.loc;
      pd.n = n;
      pd.keys = k;
      pd.pos = p;
      pd.measure = m;
      pd.h_uniform[0] = pd.h_uniform[1] = pd.h_uniform[2] = 1.0 / 8.0;
      MFTLSolverGridAddPoints(g, &pd);
      free(k);
      free(p);
      free(m);
    }
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms = vertices();
    MFTL_WindowSpec ws = {{0, 0, 0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    /* the key duplicated for points is a corner sample; give every rank a USER marker there */
    if (kind == 1)
    {
      double x[3];
      uint64_t k0;
      point_of_cell(8, 0.3, 0, x, &k0);
      ms.kind = MFTL_MARKERS_USER;
      ms.n_user = 1;
      ms.user_pos = x;
      const int st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
      TEST_ASSERT(agreed(st == MFTL_COUPLING_INVALID && req == MFTL_NULL, MPI_COMM_WORLD));
    }
    else
    {
      const int st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
      TEST_ASSERT(agreed(st == MFTL_COUPLING_INVALID && req == MFTL_NULL, MPI_COMM_WORLD));
    }
    MFTLCouplerDestroy(&cp);
    if (kind == 0) free_block(&blk);
    if (kind == 1) free_points(&pb);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/**
 * @brief The SELECTED protocol for a point location: provider regions route
 *        the requests, the coverage certificate says what is complete. And its
 *        limit, stated rather than hidden: a FALSE certificate (one sample
 *        withheld inside a certified region) cannot be detected from a point
 *        cloud -- the request succeeds and the answer differs from the full
 *        export, which only an independent full input reveals.
 */
static int test_the_selected_point_protocol_and_its_limit(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  int64_t nU[3];
  uint64_t *kU[3];
  double *U[3];
  long rounds = 0;
  affine A = {0.0, {0.0, 0.0, 0.0}};
  for (int variant = 0; variant < 3; ++variant) /* 0 full, 1 selected by requests, 2 false certificate */
  {
    MFTL_Ensemble *ens = sphere_ens(c, 0.25, 2, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    pts_block all;
    MFTL_SolverGrid *scratch = MFTLSolverGridCreate(MPI_COMM_WORLD);
    add_points(scratch, "p", 24, 0.5, OWN_SLAB, false, &all); /* this rank's owned samples */
    MFTLSolverGridDestroy(&scratch);
    for (int64_t i = 0; i < all.n; ++i) all.a[i] = sin(3.0 * all.pos[3 * i]) * cos(2.0 * all.pos[3 * i + 1]) + all.pos[3 * i + 2];
    (void) A;
    MFTL_LocationDesc ld;
    memset(&ld, 0, sizeof ld);
    ld.name = "p";
    const int loc = MFTLSolverGridAddLocation(g, &ld);
    /* Invert owner = floor(z_index * P / 24). The endpoints are CEILs;
     * floor endpoints claim another rank's samples when P does not divide 24. */
    const double zlo = (double) ((24 * g_rank + g_nproc - 1) / g_nproc) / 24.0;
    const double zhi = (double) ((24 * (g_rank + 1) + g_nproc - 1) / g_nproc) / 24.0;
    const double slab[6] = {0.0, 0.0, zlo, 1.0, 1.0, zhi};
    /* the exported subset */
    unsigned char *exp = (unsigned char *) calloc((size_t) (all.n > 0 ? all.n : 1), 1);
    int64_t withheld = -1;
    if (variant == 0) memset(exp, 1, (size_t) all.n);
    if (variant == 2)
    {
      memset(exp, 1, (size_t) all.n);
      /* withhold the sample nearest (0.5 + 0.28, 0.5, 0.5): inside a support */
      double best = 1e30;
      for (int64_t i = 0; i < all.n; ++i)
      {
        const double dx = all.pos[3 * i] - 0.75, dy = all.pos[3 * i + 1] - 0.5, dz = all.pos[3 * i + 2] - 0.5;
        if (dx * dx + dy * dy + dz * dz < best)
        {
          best = dx * dx + dy * dy + dz * dz;
          withheld = i;
        }
      }
      if (best > 0.03 * 0.03) withheld = -1; /* not this rank's */
      if (withheld >= 0) exp[withheld] = 0;
    }
    double *cov = (double *) malloc(6 * 65536 * sizeof(double));
    int n_cov = 0;
    if (variant != 0)
    {
      MFTL_ProviderRegions pr = {loc, 1, slab, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}};
      MFTLSolverGridSetProviderRegions(g, &pr);
      if (variant == 2)
      {
        memcpy(cov, slab, sizeof slab); /* the lie: complete everywhere */
        n_cov = 1;
      }
    }
    /* register the exported samples (a compacted copy) and the field array */
    int64_t ne = 0;
    pts_block pb = all;
    pb.key = (uint64_t *) malloc((size_t) (all.n > 0 ? all.n : 1) * sizeof(uint64_t));
    pb.pos = (double *) malloc(3 * (size_t) (all.n > 0 ? all.n : 1) * sizeof(double));
    pb.meas = (double *) malloc((size_t) (all.n > 0 ? all.n : 1) * sizeof(double));
    pb.a = (double *) malloc((size_t) (all.n > 0 ? all.n : 1) * sizeof(double));
    MFTL_Coupler *cp = MFTL_NULL;
    MFTL_SupportRequest *req = MFTL_NULL;
    int st = MFTL_COUPLING_NEED_COVERAGE;
    MFTL_MarkerSpec ms = vertices();
    MFTL_WindowSpec ws = {{0.01, 0.01, 0.01}};
    for (int round = 0; round < 6; ++round)
    {
      ne = 0;
      for (int64_t i = 0; i < all.n; ++i)
        if (exp[i])
        {
          pb.key[ne] = all.key[i];
          memcpy(&pb.pos[3 * ne], &all.pos[3 * i], 3 * sizeof(double));
          pb.meas[ne] = all.meas[i];
          pb.a[ne] = all.a[i];
          ++ne;
        }
      MFTLSolverGridReset(g);
      if (variant != 0)
      {
        MFTL_ProviderRegions pr = {loc, 1, slab, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}, {1.0 / 24.0, 1.0 / 24.0, 1.0 / 24.0}};
        MFTLSolverGridSetProviderRegions(g, &pr);
      }
      MFTL_PointSetDesc pd;
      memset(&pd, 0, sizeof pd);
      pd.location = loc;
      pd.n = ne;
      pd.keys = pb.key;
      pd.pos = pb.pos;
      pd.measure = pb.meas;
      pd.h_uniform[0] = pd.h_uniform[1] = pd.h_uniform[2] = 1.0 / 24.0;
      MFTLSolverGridAddPoints(g, &pd);
      if (variant != 0)
      {
        MFTL_CoverageDesc cv = {loc, MFTL_GRID_SELECTED, n_cov, MFTL_NULL, cov};
        MFTLSolverGridSetCoverage(g, &cv);
      }
      if (cp == MFTL_NULL)
        TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      else
        TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      MFTLCouplerReleaseRequest(&req);
      st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
      if (st != MFTL_COUPLING_NEED_COVERAGE) break;
      ++rounds;
      MFTL_RequestView v;
      MFTLCouplerGetRequests(req, &v);
      for (int i = 0; i < v.n_missing && n_cov < 65536; ++i)
      {
        const double *b = v.missing[i].box;
        memcpy(&cov[6 * n_cov++], b, 6 * sizeof(double));
        for (int64_t j = 0; j < all.n; ++j)
        {
          const double *x = &all.pos[3 * j];
          if (x[0] >= b[0] && x[0] <= b[3] && x[1] >= b[1] && x[1] <= b[4] && x[2] >= b[2] && x[2] <= b[5]) exp[j] = 1;
        }
      }
    }
    TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    coupled x;
    memset(&x, 0, sizeof x);
    x.c = cp;
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &x.win) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    st = MFTLCouplerBind(cp, &ms, x.win, &x.b);
    if (st != MFTL_COUPLING_OK && g_rank == 0) printf("\n  point variant %d bind: %s\n", variant, MFTLCouplingStatusName(st));
    TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SolverField F = points_field(&pb, 1);
    TEST_ASSERT(agreed(interpolate_sorted(&x, 1, &F, &nU[variant], &kU[variant], &U[variant]) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    long exported = (long) ne, owned = (long) all.n;
    long t2[2] = {exported, owned};
    MPI_Allreduce(MPI_IN_PLACE, t2, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (variant == 1 && g_rank == 0) printf("\n  exported %ld of %ld, %d coverage boxes", t2[0], t2[1], n_cov);
    if (variant == 1) TEST_ASSERT(t2[0] < t2[1]);
    teardown(&x, &g, &ens);
    free(cov);
    free(exp);
    free(pb.key);
    free(pb.pos);
    free(pb.meas);
    free(pb.a);
    free_points(&all);
  }
  int same01 = (nU[0] == nU[1]), differs02 = 0;
  for (int64_t i = 0; same01 && i < nU[0]; ++i)
    if (U[0][i] != U[1][i]) same01 = 0;
  for (int64_t i = 0; i < nU[0] && i < nU[2]; ++i)
    if (U[0][i] != U[2][i]) differs02 = 1;
  TEST_ASSERT(agreed(same01, MPI_COMM_WORLD));
  long r = rounds;
  MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(r > 0);
  /* the false certificate went through -- and changed the answer */
  TEST_ASSERT(anywhere(differs02, MPI_COMM_WORLD));
  for (int v = 0; v < 3; ++v)
  {
    free(kU[v]);
    free(U[v]);
  }
  return 0;
}

/** @brief Closed coverage certificates also cover provider planes, lines and points. */
static int test_degenerate_provider_regions_can_be_certified(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1}, pos[3] = {0.5, 0.5, 0.5};
  const bool periodic[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, periodic);
  MFTL_SolverGrid *grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "empty solid";
  const int loc = MFTLSolverGridAddLocation(grid, &ld);
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.h_uniform[0] = pd.h_uniform[1] = pd.h_uniform[2] = 0.125;
  MFTLSolverGridAddPoints(grid, &pd);
  const double regions[24] = {0, 0, 0, 1, 1, 1, 0.5, 0, 0, 0.5, 1, 1,
                             0.5, 0.5, 0, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
  MFTL_ProviderRegions pr = {loc, g_rank == 0 ? 4 : 0, regions, {0.125, 0.125, 0.125}, {0.125, 0.125, 0.125}};
  MFTLSolverGridSetProviderRegions(grid, &pr);
  double boxes[48];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k)
    {
      boxes[6 * i + k] = ((i >> (2 - k)) & 1) * 0.5;
      boxes[6 * i + 3 + k] = boxes[6 * i + k] + 0.5;
    }
  MFTL_Coupler *cp = MFTL_NULL;
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = 1;
  ms.user_pos = pos;
  MFTL_WindowSpec ws = {{0, 0, 0}};
  for (int half = 1; half <= 2; ++half)
  {
    MFTL_CoverageDesc cv = {loc, g_rank == 0 ? MFTL_GRID_SELECTED : MFTL_GRID_FULL, g_rank == 0 ? 4 * half : 0, MFTL_NULL, boxes};
    MFTLSolverGridSetCoverage(grid, &cv);
    const int init = cp ? MFTLCouplerUpdateGrid(cp) : MFTLCouplerCreate(ens, grid, MFTL_NULL, &cp);
    TEST_ASSERT(agreed(init == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SupportRequest *req = MFTL_NULL;
    const int st = MFTLCouplerRequestSupport(cp, &ms, &ws, &req);
    TEST_ASSERT(agreed(st == (half == 1 ? MFTL_COUPLING_NEED_COVERAGE : MFTL_COUPLING_OK), MPI_COMM_WORLD));
    MFTL_RequestView view;
    MFTLCouplerGetRequests(req, &view);
    int ok = 1;
    for (int i = 0; i < view.n_missing; ++i)
      for (int k = 0; k < 3; ++k)
        if (view.missing[i].box[k] == view.missing[i].box[3 + k]) ok = 0;
    TEST_ASSERT(agreed(ok, MPI_COMM_WORLD)); /* all lower-dimensional regions are covered even after the first half */
    MFTLCouplerReleaseRequest(&req);
  }
  MFTLCouplerDestroy(&cp);
  MFTLSolverGridDestroy(&grid);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Stages, windows, staleness
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A stage bind within the skin is exactly a fresh bind at those
 *        positions (bitwise); a window survives motion within its skin; one
 *        marker on ONE rank beyond it makes the status STALE on EVERY rank.
 */
static int test_stage_binds_windows_and_the_skin(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  MFTL_Ensemble *ens = sphere_ens(c, 0.3, 3, false);
  ref_lattice R = cells(20, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_REVERSED, 0, &blk);
  const double skin = 0.02;
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, skin, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  const double *p0 = MFTLCouplerBindingPositions(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  double *ps = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i)
    for (int k = 0; k < 3; ++k) ps[3 * i + k] = p0[3 * i + k] + 0.45 * skin * hash01(ref[i].key * 3 + (uint64_t) k);
  MFTL_StageGeometry sg = {ps, MFTL_NULL};
  MFTL_CouplingBinding *bs = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(x.c, x.b, &sg, &bs) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* move the surface itself there and bind afresh in the SAME window */
  for (int64_t i = 0; i < nm; ++i)
  {
    MFTL_Object *obj = ens->local_objects[ref[i].slot];
    memcpy(&obj->verts.pos[3 * ref[i].entity], &ps[3 * i], 3 * sizeof(double));
  }
  MFTLUpdateGhostPositions(ens);
  MFTLEnsembleMarkChanged(ens);
  MFTL_CouplingBinding *bf = MFTL_NULL;
  MFTL_MarkerSpec ms = vertices();
  TEST_ASSERT(agreed(MFTLCouplerBind(x.c, &ms, x.win, &bf) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  int same = (MFTLCouplerBindingMarkerCount(bf) == nm);
  for (int64_t i = 0; same && i < nm; ++i)
  {
    int n1, n2;
    const uint64_t *k1, *k2;
    const double *w1, *w2;
    MFTLCouplerBindingRow(bs, 0, i, &n1, &k1, &w1);
    uint64_t kk[1024];
    for (int e = 0; e < n1; ++e) kk[e] = k1[e];
    MFTLCouplerBindingRow(bf, 0, i, &n2, &k2, &w2);
    if (n1 != n2) same = 0;
    for (int e = 0; same && e < n1; ++e)
      if (kk[e] != k2[e] || w1[e] != w2[e]) same = 0;
  }
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  /* the base binding is now stale: the surface moved */
  MFTL_SolverField F = block_field(&blk, 1, false);
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  MFTL_CouplingFetch *f = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(x.c, x.win, 1, &F, &f) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(nm == 0 || MFTLCouplerInterpolateFetched(x.b, f, U) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(bf, f, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&f);

  /* one marker, on one rank, outside the window: STALE everywhere, nothing returned */
  int lowest = g_nproc;
  {
    int me = (nm > 0) ? g_rank : g_nproc;
    MPI_Allreduce(&me, &lowest, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
  const double *pf = MFTLCouplerBindingPositions(bf);
  memcpy(ps, pf, 3 * (size_t) nm * sizeof(double));
  /* not merely farther than the skin -- motion inside a requested bin is free --
   * but to a corner of the box no marker's bin is near */
  if (g_rank == lowest) ps[0] = ps[1] = ps[2] = 0.02;
  MFTL_CouplingBinding *bad = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(x.c, bf, &sg, &bad) == MFTL_COUPLING_STALE && bad == MFTL_NULL, MPI_COMM_WORLD));
  /* the stage binding of the old base survives its base being stale only as a handle: release order is enforced */
  TEST_ASSERT(agreed(MFTLCouplerReleaseBinding(&x.b) == MFTL_ERROR && x.b != MFTL_NULL, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bs);
  TEST_ASSERT(agreed(MFTLCouplerReleaseWindow(&x.win) == MFTL_ERROR && x.win != MFTL_NULL, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bf);
  free(U);
  free(ps);
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/**
 * @brief A marker just OUTSIDE the window -- in the bin next to a requested
 *        one, farther from it than the skin -- is STALE, never bound: its
 *        support reaches samples nobody certified, and a row over them would be
 *        a silently incomplete one. Inside the bin it is bound, and exact.
 */
static int test_a_marker_one_bin_outside_the_window_is_stale(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const int64_t N = 32;
  ref_lattice R = cells(N, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
  affine A = {0.3, {1.0, -1.0, 0.5}};
  fill_block(&blk, f_affine, &A, false);
  MFTL_Coupler *cp = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* the window's bins are half the reach (mftl_coupling.h, MFTL_WindowSpec):
   * Peskin's half-width 2 times h, halved */
  const double b = 1.0 / (double) N;
  const double p0[3] = {(floor(0.5 / b) + 0.5) * b, (floor(0.5 / b) + 0.5) * b, (floor(0.5 / b) + 0.5) * b};
  double up[3];
  memcpy(up, p0, sizeof up);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = (g_rank == 0) ? 1 : 0;
  ms.user_pos = up;
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingBinding *b1 = MFTL_NULL;
  /* within its own bin: fine, and exact */
  up[0] = p0[0] + 0.45 * b;
  TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double U = 0.0;
  MFTL_SolverField F = block_field(&blk, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(cp, b1, 1, &F, &U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_rank != 0 || fabs(U - f_affine(up, &A)) <= 1e-12, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b1);
  /* the middle of the next bin: half a bin beyond the window, skin zero */
  up[0] = p0[0] + b;
  TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b1) == MFTL_COUPLING_STALE && b1 == MFTL_NULL, MPI_COMM_WORLD));
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerReleaseRequest(&req);
  MFTLCouplerDestroy(&cp);
  free_block(&blk);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Handles that no longer describe the grid or the surface are refused
 *        collectively: a grid update stales every window; a spread with a
 *        binding of a moved surface is STALE; a fetch of one window cannot feed
 *        a binding of another; and (Debug) positions written without
 *        MFTLObjectMarkChanged() are caught by the binding's checksum.
 */
static int test_stale_handles_are_refused(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  MFTL_Ensemble *ens = sphere_ens(c, 0.3, 2, false);
  ref_lattice R = cells(16, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.01, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  double *L = (double *) calloc((size_t) (nm > 0 ? nm : 1), sizeof(double));
  MFTL_SolverField F = block_field(&blk, 1, false);

  /* a second window of the same coupler: its fetch cannot feed the first binding */
  MFTL_MarkerSpec ms = vertices();
  MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(x.c, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(x.c, req, &w2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingFetch *f2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(x.c, w2, 1, &F, &f2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(x.b, f2, U) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&f2);

#ifndef NDEBUG
  /* move a vertex without announcing it: the Debug checksum refuses the spread */
  const bool has = ens->n_local_objects > 0 && ens->local_objects[0]->verts.n_local > 0;
  const double keep = has ? ens->local_objects[0]->verts.pos[0] : 0.0;
  if (has) ens->local_objects[0]->verts.pos[0] = nextafter(keep, 2.0); /* one ulp is enough */
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  if (has) ens->local_objects[0]->verts.pos[0] = keep;
#endif
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* announced motion: the binding is stale for spreading, on every rank */
  MFTL_TransferSpec ts = {MFTL_TRANSFER_SPREAD, MFTL_NULL, x.b, 1, &F, MFTL_SPREAD_INTEGRATED};
  MFTL_TransferReady *ready = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(x.c, &ts, &ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLEnsembleMarkChanged(ens);
  TEST_ASSERT(agreed(MFTLCouplerSpreadBegin(x.c, ready, L) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  MFTLCouplerReleaseReady(&ready);
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));

  /* an OWNERSHIP update stales every window, even the fresh one (an export-only
   * one keeps the windows whose samples stay exported -- test_coupling_lifecycle) */
  MFTL_CoverageDesc cv = {blk.loc, MFTL_GRID_FULL, 0, MFTL_NULL, MFTL_NULL};
  {
    MFTL_BlockDesc nb;
    memset(&nb, 0, sizeof nb);
    nb.location = blk.loc;
    for (int k = 0; k < 3; ++k) nb.array_dims[k] = 1; /* owns nothing: a change of what is OWNED all the same */
    MFTLSolverGridAddBlock(g, &nb);
  }
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(x.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingBinding *b2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBind(x.c, &ms, w2, &b2) == MFTL_COUPLING_STALE && b2 == MFTL_NULL, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetch(x.c, w2, 1, &F, &f2) == MFTL_COUPLING_STALE && f2 == MFTL_NULL, MPI_COMM_WORLD));
  /* and the request handle of the old epoch cannot make a window */
  MFTL_CouplingWindow *w3 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(x.c, req, &w3) == MFTL_COUPLING_STALE && w3 == MFTL_NULL, MPI_COMM_WORLD));
  /* a change of the grid not yet taken up is stale too */
  MFTLSolverGridSetCoverage(g, &cv);
  MFTL_SupportRequest *r2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(x.c, &ms, &ws, &r2) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  MFTLCouplerUpdateGrid(x.c);
  MFTLCouplerReleaseRequest(&req);
  MFTLCouplerReleaseWindow(&w2);
  MFTL_CouplerStats st;
  MFTLCouplerGetStats(x.c, &st);
  TEST_ASSERT(agreed(st.n_stale >= 3, MPI_COMM_WORLD));
  free(U);
  free(L);
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/**
 * @brief A redistribution that moves a body's entities between ranks renews
 *        the generations it rebuilds, so a binding of the old pools is STALE
 *        even when the caller forgot MFTLObjectMarkChanged() after moving the
 *        vertices -- the forgotten call is exactly the case this guards.
 */
static int test_a_redistribution_stales_a_binding(void)
{
  if (g_nproc < 2) return 0; /* one rank: nothing moves between ranks */
  const double c[3] = {0.5, 0.5, 0.5};
  MFTL_Ensemble *ens = sphere_ens(c, 0.2, 2, true);
  ref_lattice R = cells(12, true);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.0, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* translate by a quarter period on every axis, WITHOUT announcing it */
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] += 0.25;
      MFTLApplyPeriodicity(&ens->domain, &obj->verts.pos[3 * v]);
    }
  }
  scatter(ens);
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  double *L = (double *) calloc((size_t) (nm > 0 ? nm : 1), sizeof(double));
  MFTL_SolverField F = block_field(&blk, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  free(L);
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/**
 * @brief A failed bind returns nothing and changes nothing: the binding made
 *        before it still works. Here the failure is a wall under an ERROR
 *        boundary policy, on the ranks whose markers reach a wall only.
 */
static int test_a_failed_bind_changes_nothing(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  MFTL_Ensemble *ens = sphere_ens(c, 0.46, 3, false);
  ref_lattice R = cells(16, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block a;
  add_lattice(g, "trunc", &R, DEC_SLAB, 0, &a);
  /* the same lattice again, under the ERROR policy */
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "strict";
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  ld.boundary = MFTL_BOUNDARY_ERROR;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = R.N[k];
    ld.lattice.spacing[k] = R.h[k];
    ld.lattice.origin[k] = R.origin[k];
  }
  const int strict = MFTLSolverGridAddLocation(g, &ld);
  if (a.piece >= 0)
  {
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = strict;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = a.lo[k];
      bd.owned_hi[k] = a.hi[k];
      bd.array_lo[k] = a.alo[k];
      bd.array_dims[k] = a.adims[k];
    }
    MFTLSolverGridAddBlock(g, &bd);
  }
  affine A = {1.0, {0.5, 0.25, -0.125}};
  fill_block(&a, f_affine, &A, false);
  MFTL_Coupler *cp = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_MarkerSpec ms = vertices();
  MFTL_WindowSpec ws = {{0, 0, 0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseRequest(&req);
  const int only_trunc = a.loc, only_strict = strict;
  MFTL_MarkerSpec mt = ms, mst = ms;
  mt.n_locations = 1;
  mt.locations = &only_trunc;
  mst.n_locations = 1;
  mst.locations = &only_strict;
  MFTL_CouplingBinding *bt = MFTL_NULL, *bs = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBind(cp, &mt, w, &bt) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_BindingQuality q;
  MFTLCouplerBindingQuality(bt, &q);
  TEST_ASSERT(anywhere(q.n_truncated > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBind(cp, &mst, w, &bs) == MFTL_COUPLING_BAD_STENCIL && bs == MFTL_NULL, MPI_COMM_WORLD));
  /* the earlier binding is untouched and correct */
  const int64_t nm = MFTLCouplerBindingMarkerCount(bt);
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  MFTL_SolverField F = block_field(&a, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(cp, bt, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *pos = MFTLCouplerBindingPositions(bt);
  int ok = 1;
  for (int64_t i = 0; i < nm; ++i)
    if (!(fabs(U[i] - f_affine(&pos[3 * i], &A)) <= 1e-12)) ok = 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  free(U);
  MFTLCouplerReleaseBinding(&bt);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&cp);
  free_block(&a);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Sampling, markers, empty ranks
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A surface much coarser than the fluid is measured and, under ERROR,
 *        refused for spreading; interpolation is still allowed.
 */
static int test_the_sampling_policy(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  for (int fine = 0; fine < 2; ++fine)
  {
    MFTL_Ensemble *ens = sphere_ens(c, 0.3, fine ? 4 : 1, false);
    ref_lattice R = cells(fine ? 8 : 32, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    add_lattice(g, "cells", &R, DEC_SLAB, 0, &blk);
    for (int pol = 0; pol < 2; ++pol)
    {
      MFTL_CouplerParams P;
      MFTLCouplerDefaultParams(&P);
      P.sampling = pol ? MFTL_SAMPLING_ERROR : MFTL_SAMPLING_REPORT;
      MFTL_Coupler *cp = MFTL_NULL;
      TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      MFTL_SpacingResult sr;
      const int st = MFTLCouplerSurfaceSpacing(cp, &sr);
      if (fine)
      {
        TEST_ASSERT(agreed(st == MFTL_COUPLING_OK && sr.n_offending == 0 && sr.max_ratio < 1.0, MPI_COMM_WORLD));
      }
      else
      {
        TEST_ASSERT(agreed(sr.n_offending > 0 && sr.max_ratio > 3.0 && sr.worst_body == 0 && sr.worst_triangle_gid >= 0, MPI_COMM_WORLD));
        TEST_ASSERT(agreed(st == (pol ? MFTL_COUPLING_BAD_SAMPLING : MFTL_COUPLING_OK), MPI_COMM_WORLD));
        MFTL_MarkerSpec ms = vertices();
        MFTL_WindowSpec ws = {{0, 0, 0}};
        MFTL_SupportRequest *req = MFTL_NULL;
        MFTL_CouplingWindow *w = MFTL_NULL;
        MFTL_CouplingBinding *b = MFTL_NULL;
        TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
        TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
        TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b) == (pol ? MFTL_COUPLING_BAD_SAMPLING : MFTL_COUPLING_OK), MPI_COMM_WORLD));
        MFTLCouplerReleaseBinding(&b);
        ms.interpolation_only = true;
        TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
        MFTLCouplerReleaseBinding(&b);
        MFTLCouplerReleaseWindow(&w);
        MFTLCouplerReleaseRequest(&req);
      }
      MFTLCouplerDestroy(&cp);
    }
    free_block(&blk);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/**
 * @brief USER markers anywhere -- every one of them on rank 0, spread over the
 *        whole domain, far outside rank 0's box. Spatial directory queries
 *        discover the providers without rebuilding the directory, and the answer
 *        stays exact. Spreading conserves the total load.
 */
static int test_user_markers_far_from_their_rank(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  /* walled: a periodic box with two cuts per axis puts every point within the
   * initial reach of every rank box through an image, and nothing has to grow */
  const bool pp[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  ref_lattice R = cells(16, false);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_ALIGNED, 1, &blk);
  fill_block(&blk, f_periodic, MFTL_NULL, false);
  MFTL_Coupler *cp = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int nu = (g_rank == 0) ? 200 : 0;
  double *up = (double *) malloc(3 * 200 * sizeof(double)), *uw = (double *) malloc(200 * sizeof(double));
  for (int i = 0; i < nu; ++i)
  {
    for (int k = 0; k < 3; ++k) up[3 * i + k] = 0.5 + 0.49 * hash01((uint64_t) (7 * i + k));
    uw[i] = 1.0 + 0.5 * hash01((uint64_t) (1000 + i));
  }
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = nu;
  ms.user_pos = up;
  ms.user_weight = uw;
  MFTL_WindowSpec ws = {{0, 0, 0}};
  MFTL_SupportRequest *req = MFTL_NULL;
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  MFTL_CouplerStats s0, s1;
  MFTLCouplerGetStats(cp, &s0);
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerGetStats(cp, &s1);
  if (g_nproc > 1) TEST_ASSERT(agreed(s1.n_directory_builds == s0.n_directory_builds, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBind(cp, &ms, w, &b) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double U[200];
  MFTL_SolverField F = block_field(&blk, 1, false);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(cp, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* exact against the brute-force rows */
  int ok = 1;
  uint64_t rk[512];
  double rw[512], dd[3 * 512];
  const double *pos = MFTLCouplerBindingPositions(b);
  for (int i = 0; i < nu; ++i)
  {
    const int n = ref_row(&R, &pos[3 * i], rk, rw, dd, 512);
    double ref = 0.0;
    for (int e = 0; e < n; ++e)
    {
      const double xk[3] = {R.origin[0] + (double) (rk[e] % 16) * R.h[0], R.origin[1] + (double) ((rk[e] / 16) % 16) * R.h[1],
                            R.origin[2] + (double) (rk[e] / 256) * R.h[2]};
      ref += rw[e] * f_periodic(xk, MFTL_NULL);
    }
    if (!(fabs(ref - U[i]) <= 1e-13)) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  /* spread the weights; their total arrives */
  for (int64_t e = 0; e < blk.n_elem; ++e) blk.a[e] = 0.0;
  TEST_ASSERT(agreed(MFTLCouplerSpread(cp, b, 1, &F, uw, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double t[2] = {0, 0};
  for (int i = 0; i < nu; ++i) t[0] += uw[i];
  for (int64_t k3 = blk.lo[2]; k3 < blk.hi[2]; ++k3)
    for (int64_t j3 = blk.lo[1]; j3 < blk.hi[1]; ++j3)
      for (int64_t i3 = blk.lo[0]; i3 < blk.hi[0]; ++i3)
      {
        const int64_t idx[3] = {i3, j3, k3};
        t[1] += blk.a[block_elem(&blk, idx)];
      }
  MPI_Allreduce(MPI_IN_PLACE, t, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(fabs(t[0] - t[1]) <= 1e-12 * t[0]);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerReleaseRequest(&req);
  MFTLCouplerDestroy(&cp);
  free(up);
  free(uw);
  free_block(&blk);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Ranks with no markers, ranks with no samples, and an ensemble with no
 *        bodies at all: every collective is still entered, transfers are no-ops,
 *        nothing hangs.
 */
static int test_empty_ranks_and_an_empty_ensemble(void)
{
  for (int variant = 0; variant < 2; ++variant)
  {
    const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
    const bool pp[3] = {false, false, false};
    MFTL_Ensemble *ens = make_ens(lo, hi, pp);
    if (variant == 0)
    {
      /* a small sphere inside one rank's box; all samples on rank 0 */
      const double c[3] = {0.2, 0.2, 0.2};
      add_sphere(ens, c, 0.05, 2);
      scatter(ens);
    }
    ref_lattice R = cells(12, false);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    lat_block blk;
    add_lattice(g, "cells", &R, DEC_ROOT, 0, &blk);
    affine A = {0.5, {1.0, 2.0, 3.0}};
    fill_block(&blk, f_affine, &A, false);
    coupled x;
    TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.0, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
    double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
    MFTL_SolverField F = block_field(&blk, 1, false);
    TEST_ASSERT(agreed(MFTLCouplerInterpolate(x.c, x.b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const double *pos = MFTLCouplerBindingPositions(x.b);
    int ok = 1;
    for (int64_t i = 0; i < nm; ++i)
      if (!(fabs(U[i] - f_affine(&pos[3 * i], &A)) <= 1e-12)) ok = 0;
    TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerSpread(x.c, x.b, 1, &F, U, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    long n = (long) nm;
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT((variant == 1) == (n == 0));
    free(U);
    free_block(&blk);
    teardown(&x, &g, &ens);
  }
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Invalid input
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Bad registrations are refused where they are made; bad state is INVALID on every rank. */
static int test_invalid_input_is_refused(void)
{
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "";
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR);
  ld.name = "a_name_that_is_far_too_long_for_a_location";
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR);
  ld.name = "rect";
  ld.lattice.kind = MFTL_LATTICE_RECTILINEAR;
  const double coords[3] = {0.1, 0.5, 0.3}, faces[4] = {0.0, 0.2, 0.6, 1.0};
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = 3;
    ld.lattice.coords[k] = coords;
    ld.lattice.faces[k] = faces;
  }
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR); /* coord outside its cell */
  memset(&ld, 0, sizeof ld);
  ld.name = "pts";
  const int lp = MFTLSolverGridAddLocation(g, &ld);
  const uint64_t keys[2] = {1, 2};
  double pos[6] = {0.1, 0.1, 0.1, NAN, 0.2, 0.2};
  const double meas[2] = {1.0, 1.0}, zero[2] = {1.0, 0.0};
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = lp;
  pd.n = 2;
  pd.keys = keys;
  pd.pos = pos;
  pd.measure = meas;
  pd.h_uniform[0] = pd.h_uniform[1] = pd.h_uniform[2] = 0.1;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  pos[3] = 0.2;
  pd.measure = zero;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  pd.measure = meas;
  double extent[6] = {INFINITY, 0.1, 0.1, 0.1, 0.1, 0.1};
  pd.extent = extent;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  pd.extent = MFTL_NULL;
  MFTL_ProviderRegions invalid_bounds = {lp, 0, MFTL_NULL, {NAN, 0, 0}, {0, 0, 0}};
  TEST_ASSERT(MFTLSolverGridSetProviderRegions(g, &invalid_bounds) == MFTL_ERROR);
  pd.h_uniform[1] = 0.0;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  memset(&ld, 0, sizeof ld);
  ld.name = "lat";
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = 4;
    ld.lattice.spacing[k] = 0.25;
    ld.lattice.origin[k] = 0.125;
  }
  const int ll = MFTLSolverGridAddLocation(g, &ld);
  MFTL_BlockDesc bd;
  memset(&bd, 0, sizeof bd);
  bd.location = ll;
  for (int k = 0; k < 3; ++k)
  {
    bd.owned_lo[k] = 0;
    bd.owned_hi[k] = 5; /* past the lattice */
    bd.array_dims[k] = 5;
  }
  TEST_ASSERT(MFTLSolverGridAddBlock(g, &bd) == MFTL_ERROR);
  for (int k = 0; k < 3; ++k)
  {
    bd.owned_hi[k] = 4;
    bd.ghost_layers[k] = 1; /* the array has no room for them */
    bd.array_dims[k] = 4;
  }
  TEST_ASSERT(MFTLSolverGridAddBlock(g, &bd) == MFTL_ERROR);
  MFTLSolverGridDestroy(&g);

  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);

  /* a periodic axis the lattice does not span */
  {
    MFTL_SolverGrid *g2 = MFTLSolverGridCreate(MPI_COMM_WORLD);
    MFTL_LocationDesc d2;
    memset(&d2, 0, sizeof d2);
    d2.name = "short";
    d2.lattice.kind = MFTL_LATTICE_UNIFORM;
    for (int k = 0; k < 3; ++k)
    {
      d2.lattice.dims[k] = 10;
      d2.lattice.spacing[k] = 0.09;
      d2.lattice.origin[k] = 0.05;
    }
    MFTLSolverGridAddLocation(g2, &d2);
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, MFTL_NULL, &cp) == MFTL_COUPLING_INVALID && cp == MFTL_NULL, MPI_COMM_WORLD));
    MFTLSolverGridDestroy(&g2);
  }
  /* a point outside the box, on one rank only */
  {
    MFTL_SolverGrid *g2 = MFTLSolverGridCreate(MPI_COMM_WORLD);
    MFTL_LocationDesc d2;
    memset(&d2, 0, sizeof d2);
    d2.name = "p";
    const int l2 = MFTLSolverGridAddLocation(g2, &d2);
    double p2[3] = {0.5, 0.5, (g_rank == g_nproc - 1) ? 1.5 : 0.5};
    const uint64_t k2 = (uint64_t) g_rank;
    const double m2 = 1.0;
    MFTL_PointSetDesc q;
    memset(&q, 0, sizeof q);
    q.location = l2;
    q.n = 1;
    q.keys = &k2;
    q.pos = p2;
    q.measure = &m2;
    q.h_uniform[0] = q.h_uniform[1] = q.h_uniform[2] = 0.1;
    MFTLSolverGridAddPoints(g2, &q);
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, MFTL_NULL, &cp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLSolverGridDestroy(&g2);
  }
  /* invalid parameters; and parameters that differ between ranks */
  {
    MFTL_SolverGrid *g2 = MFTLSolverGridCreate(MPI_COMM_WORLD);
    ref_lattice R = cells(8, true);
    lat_block blk;
    add_lattice(g2, "cells", &R, DEC_SLAB, 0, &blk);
    MFTL_CouplerParams P;
    MFTLCouplerDefaultParams(&P);
    P.growth_step = 1.0;
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, &P, &cp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLCouplerDefaultParams(&P);
    if (g_rank == g_nproc - 1) P.max_amplification = 5.0;
    if (g_nproc > 1) TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, &P, &cp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    /* a valid coupler: then a skin past half the period, and skins that differ */
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms;
    memset(&ms, 0, sizeof ms);
    ms.kind = MFTL_MARKERS_USER;
    MFTL_WindowSpec big = {{0.45, 0.45, 0.45}};
    MFTL_SupportRequest *req = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &big, &req) == MFTL_COUPLING_INVALID && req == MFTL_NULL, MPI_COMM_WORLD));
    MFTL_WindowSpec mixed = {{(g_rank == 0) ? 0.01 : 0.02, 0.01, 0.01}};
    if (g_nproc > 1) TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &mixed, &req) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    /* transfers whose field lists differ between ranks */
    MFTL_WindowSpec ws = {{0.0, 0.0, 0.0}};
    MFTL_CouplingWindow *w = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(cp, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_SolverField F[2] = {block_field(&blk, 1, false), block_field(&blk, 1, false)};
    MFTL_CouplingFetch *f = MFTL_NULL;
    if (g_nproc > 1)
    {
      TEST_ASSERT(agreed(MFTLCouplerFetch(cp, w, (g_rank == 0) ? 1 : 2, F, &f) == MFTL_COUPLING_INVALID && f == MFTL_NULL, MPI_COMM_WORLD));
      F[0].halo_valid = (g_rank == 0);
      TEST_ASSERT(agreed(MFTLCouplerFetch(cp, w, 1, F, &f) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    }
    /* a field naming a location the window was not made for */
    MFTL_SolverField bad = F[1];
    bad.location = 7;
    TEST_ASSERT(agreed(MFTLCouplerFetch(cp, w, 1, &bad, &f) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLCouplerReleaseWindow(&w);
    MFTLCouplerReleaseRequest(&req);
    MFTLCouplerDestroy(&cp);
    free_block(&blk);
    MFTLSolverGridDestroy(&g2);
  }
  /* the solver's communicator must hold the ensemble's processes */
  if (g_nproc > 1)
  {
    MFTL_SolverGrid *g2 = MFTLSolverGridCreate(MPI_COMM_SELF);
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, MFTL_NULL, &cp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLSolverGridDestroy(&g2);
  }
  /* names that differ between ranks */
  if (g_nproc > 1)
  {
    MFTL_SolverGrid *g2 = MFTLSolverGridCreate(MPI_COMM_WORLD);
    MFTL_LocationDesc d2;
    memset(&d2, 0, sizeof d2);
    d2.name = (g_rank == 0) ? "pressure" : "velocity";
    MFTLSolverGridAddLocation(g2, &d2);
    MFTL_Coupler *cp = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g2, MFTL_NULL, &cp) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    MFTLSolverGridDestroy(&g2);
  }
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* A small two-location fixture with a marker on EVERY rank, including ranks
 * that own no solver samples. This makes rank-local failures non-vacuous. */
typedef struct
{
  MFTL_Ensemble *ens;
  MFTL_SolverGrid *grid;
  MFTL_Coupler *c;
  MFTL_SupportRequest *req;
  MFTL_CouplingWindow *win;
  MFTL_CouplingBinding *binding;
  lat_block block[2];
  MFTL_SolverField field[2];
  MFTL_MarkerSpec markers;
  double pos[3];
} contract_fixture;

static int contract_setup(contract_fixture *f, bool subset)
{
  memset(f, 0, sizeof *f);
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool per[3] = {false, false, false};
  f->ens = make_ens(lo, hi, per);
  f->grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  ref_lattice lattice = cells(16, false);
  affine a = {0.2, {1.0, 2.0, -0.5}};
  for (int i = 0; i < 2; ++i)
  {
    add_lattice(f->grid, i ? "second" : "first", &lattice, DEC_SLAB, 0, &f->block[i]);
    fill_block(&f->block[i], f_affine, &a, false);
    f->field[i] = block_field(&f->block[i], 1, false);
  }
  f->pos[0] = 0.28;
  f->pos[1] = f->pos[2] = 0.5;
  f->markers.kind = MFTL_MARKERS_USER;
  f->markers.n_user = 1;
  f->markers.user_pos = f->pos;
  MFTL_WindowSpec ws = {{0.3, 0.05, 0.05}};
  int st = MFTLCouplerCreate(f->ens, f->grid, MFTL_NULL, &f->c);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerRequestSupport(f->c, &f->markers, &ws, &f->req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(f->c, f->req, &f->win);
  if (subset)
  {
    f->markers.n_locations = 1;
    f->markers.locations = &f->block[0].loc;
  }
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerBind(f->c, &f->markers, f->win, &f->binding);
  return st;
}

static void contract_teardown(contract_fixture *f)
{
  MFTLCouplerReleaseBinding(&f->binding);
  MFTLCouplerReleaseWindow(&f->win);
  MFTLCouplerReleaseRequest(&f->req);
  MFTLCouplerDestroy(&f->c);
  for (int i = 0; i < 2; ++i) free_block(&f->block[i]);
  MFTLSolverGridDestroy(&f->grid);
  MFTLEnsembleDestroy(&f->ens);
}

static int test_skin_can_span_several_bins(void)
{
  contract_fixture f;
  TEST_ASSERT(agreed(contract_setup(&f, false) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  f.pos[0] += 0.25; /* four bins, still strictly inside the requested skin */
  MFTL_CouplingBinding *moved = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBind(f.c, &f.markers, f.win, &moved) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double u = -123;
  affine a = {0.2, {1.0, 2.0, -0.5}};
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(f.c, moved, 1, f.field, &u) == MFTL_COUPLING_OK &&
                         fabs(u - f_affine(f.pos, &a)) < 1e-12, MPI_COMM_WORLD));
  MFTL_StageGeometry stage = {f.pos, MFTL_NULL};
  MFTL_CouplingBinding *next = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(f.c, moved, &stage, &next) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&next);
  MFTLCouplerReleaseBinding(&moved);
  contract_teardown(&f);
  return 0;
}

static int test_grid_edits_invalidate_every_consumer(void)
{
  contract_fixture f;
  TEST_ASSERT(agreed(contract_setup(&f, false) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingFetch *fetch = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(f.c, f.win, 1, f.field, &fetch) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* Only one rank changes registration -- what it OWNS -- before UpdateGrid has been called. */
  if (g_rank == 0)
  {
    MFTL_BlockDesc nb;
    memset(&nb, 0, sizeof nb);
    for (int k = 0; k < 3; ++k) nb.array_dims[k] = 1;
    MFTLSolverGridAddBlock(f.grid, &nb);
  }
  MFTL_CouplingWindow *win = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(f.c, f.req, &win) == MFTL_COUPLING_STALE && !win, MPI_COMM_WORLD));
  MFTL_CouplingBinding *binding = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBind(f.c, &f.markers, f.win, &binding) == MFTL_COUPLING_STALE && !binding, MPI_COMM_WORLD));
  MFTL_StageGeometry stage = {f.pos, MFTL_NULL};
  TEST_ASSERT(agreed(MFTLCouplerBindStage(f.c, f.binding, &stage, &binding) == MFTL_COUPLING_STALE && !binding, MPI_COMM_WORLD));
  MFTL_CouplingFetch *extra = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(f.c, f.win, 1, f.field, &extra) == MFTL_COUPLING_STALE && !extra, MPI_COMM_WORLD));
  double u = -123;
  int unchanged = 1;
  if (g_rank == 0)
    unchanged = MFTLCouplerInterpolateFetched(f.binding, fetch, &u) == MFTL_COUPLING_STALE && u == -123;
  TEST_ASSERT(agreed(unchanged, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(f.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(f.binding, fetch, &u) == MFTL_COUPLING_STALE && u == -123, MPI_COMM_WORLD));
  unsigned char mask[4096];
  memset(mask, 123, sizeof mask);
  int mask_ok = 1;
  if (f.block[0].piece >= 0)
    mask_ok = MFTLCouplerBandMask(f.win, 0, f.block[0].piece, MFTL_BAND_USED, mask) == MFTL_ERROR && mask[0] == 123;
  TEST_ASSERT(agreed(mask_ok, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&fetch);
  contract_teardown(&f);
  return 0;
}

static int test_failed_interpolation_preserves_all_outputs(void)
{
  contract_fixture f;
  TEST_ASSERT(agreed(contract_setup(&f, true) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplingFetch *fetch = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(f.c, f.win, 2, f.field, &fetch) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplerStats stats;
  MFTL_SpacingResult spacing;
  MFTL_Coupler *invalid = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(f.ens, f.grid, MFTL_NULL, g_rank == 0 ? MFTL_NULL : &invalid) == MFTL_COUPLING_INVALID && !invalid,
                     MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerReduceStats(f.c, g_rank == 0 ? MFTL_NULL : &stats) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSurfaceSpacing(f.c, g_rank == 0 ? MFTL_NULL : &spacing) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTL_CouplingFetch *extra = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetch(f.c, f.win, 1, f.field, g_rank == 0 ? MFTL_NULL : &extra) == MFTL_COUPLING_INVALID && !extra,
                     MPI_COMM_WORLD));
  double u[2] = {-123, -456};
  /* Field 0 is bound, field 1 is not: validation must precede EVERY write. */
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(f.binding, fetch, u) == MFTL_COUPLING_INVALID &&
                         u[0] == -123 && u[1] == -456, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(f.c, f.binding, 2, f.field, u) == MFTL_COUPLING_INVALID &&
                         u[0] == -123 && u[1] == -456, MPI_COMM_WORLD));
  /* An invalid output on ONE rank must leave all other ranks untouched too. */
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(f.c, f.binding, 1, f.field, g_rank == 0 ? MFTL_NULL : u) == MFTL_COUPLING_INVALID &&
                         u[0] == -123, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&fetch);
  contract_teardown(&f);
  return 0;
}

static int test_transfer_tokens_retain_and_revalidate_handles(void)
{
  contract_fixture f;
  TEST_ASSERT(agreed(contract_setup(&f, false) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_TransferSpec spec = {MFTL_TRANSFER_SPREAD, MFTL_NULL, f.binding, 1, f.field, MFTL_SPREAD_INTEGRATED};
  MFTL_TransferReady *spread = MFTL_NULL, *fetch = MFTL_NULL, *fetch2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(f.c, &spec, &spread) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* The token borrows the rows until released. */
  TEST_ASSERT(agreed(MFTLCouplerReleaseBinding(&f.binding) == MFTL_ERROR && f.binding, MPI_COMM_WORLD));
  spec.direction = MFTL_TRANSFER_FETCH;
  spec.window = f.win;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(f.c, &spec, &fetch) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  spec.n_fields = 2;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(f.c, &spec, &fetch2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  if (g_nproc > 1)
    TEST_ASSERT(agreed(MFTLCouplerFetchBegin(f.c, g_rank == 0 ? fetch : fetch2) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  MFTL_CouplingFetch *values = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetchBegin(f.c, fetch) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetchEnd(f.c, g_rank == 0 ? MFTL_NULL : &values) == MFTL_COUPLING_INVALID && !values, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetchEnd(f.c, &values) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&values);

  double load = 2.0;
  double *saved = (double *) malloc((size_t) f.block[0].n_elem * sizeof(double));
  memcpy(saved, f.block[0].a, (size_t) f.block[0].n_elem * sizeof(double));
  /* This used to return locally on rank 0 while its peers waited in MPI. */
  TEST_ASSERT(agreed(MFTLCouplerSpread(f.c, f.binding, 1, f.field, g_rank == 0 ? MFTL_NULL : &load,
                                     MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(memcmp(saved, f.block[0].a, (size_t) f.block[0].n_elem * sizeof(double)) == 0, MPI_COMM_WORLD));
  free(saved);
  TEST_ASSERT(agreed(MFTLCouplerSpreadBegin(f.c, spread, &load) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseReady(&spread); /* refused until End */
  TEST_ASSERT(agreed(spread != MFTL_NULL, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSpreadEnd(f.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* Tokens prepared before a change of what is OWNED cannot start an old schedule
   * (an export-only change that keeps the window keeps them: test_coupling_lifecycle). */
  if (g_rank == 0)
  {
    MFTL_BlockDesc nb;
    memset(&nb, 0, sizeof nb);
    for (int k = 0; k < 3; ++k) nb.array_dims[k] = 1;
    MFTLSolverGridAddBlock(f.grid, &nb);
  }
  TEST_ASSERT(agreed(MFTLCouplerFetchBegin(f.c, fetch) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSpreadBegin(f.c, spread, &load) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(f.c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetchBegin(f.c, fetch) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  MFTLCouplerReleaseReady(&spread);
  MFTLCouplerReleaseBinding(&f.binding);
  TEST_ASSERT(agreed(MFTLCouplerReleaseWindow(&f.win) == MFTL_ERROR && f.win, MPI_COMM_WORLD));
  MFTLCouplerDestroy(&f.c); /* refused collectively while tokens/windows live */
  TEST_ASSERT(agreed(f.c != MFTL_NULL, MPI_COMM_WORLD));
  MFTLCouplerReleaseReady(&fetch);
  MFTLCouplerReleaseReady(&fetch2);
  contract_teardown(&f);
  return 0;
}

static int test_masked_duplicate_cannot_cancel_an_ownership_hole(void)
{
  if (g_nproc < 2) return 0;
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool per[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  MFTL_SolverGrid *grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  ref_lattice lattice = cells(8, false);
  lat_block block;
  add_lattice(grid, "cells", &lattice, DEC_SLAB, 0, &block);
  MFTLSolverGridReset(grid);
  if (g_rank < 2)
  {
    MFTL_BlockDesc bd = {0};
    bd.location = 0;
    unsigned char inactive[256] = {0};
    for (int k = 0; k < 3; ++k) bd.owned_hi[k] = bd.array_dims[k] = k == 2 ? 4 : 8;
    bd.active = g_rank == 1 ? inactive : MFTL_NULL;
    TEST_ASSERT(MFTLSolverGridAddBlock(grid, &bd) >= 0);
  }
  MFTL_Coupler *cp = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, grid, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double pos[3] = {0.5, 0.5, 0.5};
  MFTL_MarkerSpec ms = {0};
  ms.kind = MFTL_MARKERS_USER;
  ms.n_user = 1;
  ms.user_pos = pos;
  MFTL_WindowSpec ws = {{0.5, 0.5, 0.5}};
  MFTL_SupportRequest *req = MFTL_NULL;
  /* The active and masked counts add to 8^3, and returned active keys are
   * unique, yet half the grid is unowned. Counts alone cannot prove coverage. */
  TEST_ASSERT(agreed(MFTLCouplerRequestSupport(cp, &ms, &ws, &req) == MFTL_COUPLING_INVALID && !req, MPI_COMM_WORLD));
  MFTLCouplerDestroy(&cp);
  free_block(&block);
  MFTLSolverGridDestroy(&grid);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_each_empty_selected_provider_needs_resolution(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool per[3] = {false, false, false};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  MFTL_SolverGrid *grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc location = {0};
  location.name = "points";
  const int loc = MFTLSolverGridAddLocation(grid, &location);
  if (g_rank == 0)
  {
    const double box[6] = {0, 0, 0, 1, 1, 1};
    MFTL_ProviderRegions regions = {loc, 1, box, {0, 0, 0}, {0, 0, 0}};
    MFTLSolverGridSetProviderRegions(grid, &regions);
    MFTL_CoverageDesc coverage = {loc, MFTL_GRID_SELECTED, 0, MFTL_NULL, MFTL_NULL};
    MFTLSolverGridSetCoverage(grid, &coverage);
  }
  else
  {
    const double pos[3] = {0.5, 0.5, 0.5}, measure = 0.001;
    const uint64_t key = (uint64_t) g_rank;
    MFTL_PointSetDesc points = {0};
    points.location = loc;
    points.n = 1;
    points.keys = &key;
    points.pos = pos;
    points.measure = &measure;
    for (int k = 0; k < 3; ++k) points.h_uniform[k] = 0.1;
    MFTLSolverGridAddPoints(grid, &points);
  }
  MFTL_Coupler *cp = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, grid, MFTL_NULL, &cp) == MFTL_COUPLING_INVALID && !cp, MPI_COMM_WORLD));
  if (g_rank == 0)
  {
    const double box[6] = {0, 0, 0, 1, 1, 1};
    MFTL_ProviderRegions regions = {loc, 1, box, {0.2, 0.2, 0.2}, {0.2, 0.2, 0.2}};
    MFTLSolverGridSetProviderRegions(grid, &regions);
  }
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, grid, MFTL_NULL, &cp) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerDestroy(&cp);
  MFTLSolverGridDestroy(&grid);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Every key some row names is USED by its owner, and USED implies
 *        REQUESTED: the masks describe the band that was really built.
 */
static int test_band_masks_describe_the_band(void)
{
  const double c[3] = {0.5, 0.5, 0.5};
  MFTL_Ensemble *ens = sphere_ens(c, 0.3, 3, true);
  ref_lattice R = cells(16, true);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  lat_block blk;
  add_lattice(g, "cells", &R, DEC_REVERSED, 0, &blk);
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, 0.0, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* all row keys, everywhere */
  int cap = 1 << 16, n = 0;
  double *keys = (double *) malloc((size_t) cap * sizeof(double));
  for (int64_t i = 0; i < MFTLCouplerBindingMarkerCount(x.b); ++i)
  {
    int m;
    const uint64_t *k;
    const double *w;
    MFTLCouplerBindingRow(x.b, 0, i, &m, &k, &w);
    for (int e = 0; e < m && n < cap; ++e) keys[n++] = (double) k[e];
  }
  int tot = 0;
  double *all = allgather_doubles(keys, n, 1, &tot);
  int ok = 1;
  long used_count = 0;
  if (blk.piece >= 0)
  {
    const int64_t nx = blk.hi[0] - blk.lo[0], ny = blk.hi[1] - blk.lo[1], nz = blk.hi[2] - blk.lo[2];
    unsigned char *used = (unsigned char *) malloc((size_t) (nx * ny * nz)), *req = (unsigned char *) malloc((size_t) (nx * ny * nz));
    MFTLCouplerBandMask(x.win, blk.loc, blk.piece, MFTL_BAND_USED, used);
    MFTLCouplerBandMask(x.win, blk.loc, blk.piece, MFTL_BAND_REQUESTED, req);
    for (int64_t s = 0; s < nx * ny * nz; ++s)
    {
      if (used[s] && !req[s]) ok = 0;
      used_count += used[s];
    }
    for (int i = 0; i < tot; ++i)
    {
      const uint64_t k = (uint64_t) all[i];
      const int64_t idx[3] = {(int64_t) (k % 16), (int64_t) ((k / 16) % 16), (int64_t) (k / 256)};
      bool own = true;
      for (int q = 0; q < 3; ++q)
        if (idx[q] < blk.lo[q] || idx[q] >= blk.hi[q]) own = false;
      if (own && !used[(idx[0] - blk.lo[0]) + nx * ((idx[1] - blk.lo[1]) + ny * (idx[2] - blk.lo[2]))]) ok = 0;
    }
    free(used);
    free(req);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(anywhere(used_count > 0, MPI_COMM_WORLD));
  free(keys);
  free(all);
  free_block(&blk);
  teardown(&x, &g, &ens);
  return 0;
}

/** @brief A missing coarse lattice cell must still influence the metric bound.
 * The global lattice declares its width even when no provider owns it. Its
 * pilot reaches a marker well beyond the fine cells' own support radius. */
static int test_an_unowned_coarse_cell_cannot_hide_from_the_pilot(void)
{
  enum { N = 24 };
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool periodic[3] = {false, false, false};
  double faces[3][N + 1], coords[3][N];
  for (int k = 0; k < 3; ++k)
  {
    for (int j = 0; j <= N; ++j) faces[k][j] = k == 0 && j < N ? 0.02 * j : (double) j / N;
    for (int j = 0; j < N; ++j) coords[k][j] = 0.5 * (faces[k][j] + faces[k][j + 1]);
  }
  int ok = 1;
  for (int missing = 0; missing < 2; ++missing)
  {
    MFTL_Ensemble *ens = make_ens(lo, hi, periodic);
    MFTL_SolverGrid *grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
    MFTL_LocationDesc ld = {0};
    ld.name = "unowned coarse pilot";
    ld.lattice.kind = MFTL_LATTICE_RECTILINEAR;
    for (int k = 0; k < 3; ++k)
    {
      ld.lattice.dims[k] = N;
      ld.lattice.coords[k] = coords[k];
      ld.lattice.faces[k] = faces[k];
    }
    const int loc = MFTLSolverGridAddLocation(grid, &ld);
    int added = loc >= 0;
    if (g_rank == 0)
    {
      MFTL_BlockDesc block = {0};
      block.location = loc;
      for (int k = 0; k < 3; ++k) block.owned_hi[k] = block.array_dims[k] = k == 0 ? N - missing : N;
      added = added && MFTLSolverGridAddBlock(grid, &block) >= 0;
    }
    TEST_ASSERT(agreed(added, MPI_COMM_WORLD));
    MFTL_Coupler *coupler = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, grid, MFTL_NULL, &coupler) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    const double point[3] = {0.2, 0.5, 0.5};
    MFTL_MarkerSpec markers = {0};
    markers.kind = MFTL_MARKERS_USER;
    markers.n_user = g_rank == g_nproc - 1 ? 1 : 0;
    markers.user_pos = point;
    MFTL_WindowSpec window = {{0, 0, 0}};
    MFTL_SupportRequest *request = MFTL_NULL;
    const int status = MFTLCouplerRequestSupport(coupler, &markers, &window, &request);
    if (status != (missing ? MFTL_COUPLING_NEED_COVERAGE : MFTL_COUPLING_OK)) ok = 0;
    MFTL_RequestView view = {0};
    if (request) MFTLCouplerGetRequests(request, &view);
    if (missing && !anywhere(view.n_unprovided > 0, MPI_COMM_WORLD)) ok = 0;
    MFTLCouplerReleaseRequest(&request);
    MFTLCouplerDestroy(&coupler);
    MFTLSolverGridDestroy(&grid);
    MFTLEnsembleDestroy(&ens);
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
  if (g_rank == 0) printf("\n=== Coupling protocol (np = %d) ===\n", g_nproc);

  RUN_TEST(test_a_selected_band_built_from_requests_equals_full);
  RUN_TEST(test_a_one_cell_hole_is_found_and_a_masked_cell_is_not);
  RUN_TEST(test_an_ownership_gap_is_never_a_wall);
  RUN_TEST(test_an_unowned_coarse_cell_cannot_hide_from_the_pilot);
  RUN_TEST(test_duplicate_ownership_is_invalid);
  RUN_TEST(test_the_selected_point_protocol_and_its_limit);
  RUN_TEST(test_degenerate_provider_regions_can_be_certified);
  RUN_TEST(test_stage_binds_windows_and_the_skin);
  RUN_TEST(test_a_marker_one_bin_outside_the_window_is_stale);
  RUN_TEST(test_stale_handles_are_refused);
  RUN_TEST(test_a_redistribution_stales_a_binding);
  RUN_TEST(test_a_failed_bind_changes_nothing);
  RUN_TEST(test_the_sampling_policy);
  RUN_TEST(test_user_markers_far_from_their_rank);
  RUN_TEST(test_empty_ranks_and_an_empty_ensemble);
  RUN_TEST(test_invalid_input_is_refused);
  RUN_TEST(test_band_masks_describe_the_band);
  RUN_TEST(test_skin_can_span_several_bins);
  RUN_TEST(test_grid_edits_invalidate_every_consumer);
  RUN_TEST(test_failed_interpolation_preserves_all_outputs);
  RUN_TEST(test_transfer_tokens_retain_and_revalidate_handles);
  RUN_TEST(test_masked_duplicate_cannot_cancel_an_ownership_hole);
  RUN_TEST(test_each_empty_selected_provider_needs_resolution);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
