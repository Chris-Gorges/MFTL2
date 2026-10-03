/*
 * @Author: Christian Gorges
 * @Date: 2026-10-03
 *
 * @brief Tests for the extent of a body on a periodic axis and THE PERIODIC
 *        EXTENT RULE: on every periodic axis a body either wraps the axis or
 *        is narrower than half the period.
 *
 * Two halves, and they test different things:
 *
 *   THE MEASUREMENT (MFTLPeriodicExtent*, local). A synthetic chain of
 *   triangles of known extent is placed at a hundred offsets on each axis of a
 *   box with a different, non-round period per axis, and the measured extent
 *   must be the true one whenever the chain is narrower than 3L/4 -- at every
 *   offset, including the ones that straddle both the seam and the middle of
 *   the axis, where the two-frame measurement it replaces reported "covers the
 *   axis". That those offsets exist is ASSERTED (non-vacuity), and so is the
 *   claim that every extent the two-frame code got right comes out
 *   bit-identical, which is what keeps the topology change's results unchanged.
 *
 *   THE CHECK (MFTLEnsembleCheckPeriodicExtents, collective). Real meshes --
 *   stretched icospheres and level-set films -- redistributed over the grid,
 *   with the expected verdict computed from the generated vertices before
 *   partitioning, on rank 0, by arithmetic that shares nothing with the helper.
 *   Every claim is made on an agreed() verdict, and the share of bodies held by
 *   two or more ranks is asserted, so the reduction is genuinely exercised.
 *
 * Registered at np = 1, 2, 4, 8 and 27 for the usual reasons: a face, an edge,
 * a corner, and an interior rank whose 26 neighbours are distinct.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/* A box with a different, non-round period per axis and an origin off zero, so
 * neither a cubic assumption nor an origin of zero can hide in the code. */
static const double g_lo[3] = {-0.7, 0.2, -1.9};
static const double g_len[3] = {2.0, 1.3, 3.1};

static MFTL_Ensemble *make_ens(const bool per[3])
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double lo[3], hi[3];
  bool p[3];
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = g_lo[k];
    hi[k] = g_lo[k] + g_len[k];
    p[k] = per[k];
  }
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, p, 8);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. The measurement, on synthetic chains
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The two-frame resolve the four-frame one replaced, verbatim: the regression reference. */
static void two_frame_resolve(const MFTL_Domain *dom, const double *e, int k, double *centre, double *half, bool *full)
{
  const double lo0 = e[k], hi0 = -e[3 + k];
  double lo = lo0, hi = hi0;
  *full = MFTL_FALSE;
  if (dom->is_periodic[k])
  {
    const double L = dom->domain_size[k];
    const double lo1 = e[6 + k], hi1 = -e[9 + k];
    if (hi1 - lo1 < hi0 - lo0)
    {
      lo = lo1 - 0.5 * L;
      hi = hi1 - 0.5 * L;
    }
    if (hi - lo >= L) *full = MFTL_TRUE;
  }
  *centre = 0.5 * (lo + hi);
  *half = 0.5 * (hi - lo);
}

/**
 * @brief A strip of `n` triangles along axis @p k from @p a to `a + E`, stored
 *        as a mesh would store it: corner 0 wrapped into the box, the other
 *        corners its minimum-image neighbours. The other two coordinates are
 *        fixed and inside the box.
 */
static void add_chain(const MFTL_Domain *dom, int k, double a, double E, int n, double acc[MFTL_PERIODIC_EXTENT_SLOTS])
{
  const double h = E / n;
  for (int i = 0; i < n; ++i)
  {
    double T[3][3];
    for (int j = 0; j < 3; ++j)
      for (int m = 0; m < 3; ++m) T[j][m] = dom->global_min[m] + 0.3 * dom->domain_size[m] + 0.01 * (double) j;
    double x0 = a + i * h;
    const double L = dom->domain_size[k];
    x0 -= L * floor((x0 - dom->global_min[k]) / L);
    T[0][k] = x0;
    T[1][k] = x0 + h;
    T[2][k] = x0 + 0.5 * h;
    MFTLPeriodicExtentAddTriangle(dom, (const double(*)[3]) T, acc);
  }
}

static int test_a_chain_is_measured_exactly_below_three_quarters_at_every_offset(void)
{
  const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  MFTL_Ensemble *ens = make_ens(per);
  TEST_ASSERT(ens != MFTL_NULL);
  const MFTL_Domain *dom = &ens->domain;

  int ok = 1, n_exact = 0, n_cover = 0, n_two_frame_misses = 0, n_identical = 0, n_compared = 0;
  for (int k = 0; k < 3; ++k)
  {
    const double L = dom->domain_size[k];
    for (int s = 0; s < 20; ++s)
    {
      const double E = (0.02 + 0.05 * s) * L;  // 0.02 .. 0.97: never on 1/2 or 3/4 exactly
      for (int i = 0; i < 101; ++i)
      {
        const double a = dom->global_min[k] + L * (double) i / 101.0;
        double acc[MFTL_PERIODIC_EXTENT_SLOTS];
        MFTLPeriodicExtentInit(acc);
        add_chain(dom, k, a, E, 40, acc);

        MFTL_PeriodicExtent x;
        MFTLPeriodicExtentResolve(dom, acc, &x);
        if (x.empty) ok = 0;

        if (E < MFTL_PERIODIC_EXTENT_COVER_FRACTION * L)
        {
          /* Exact: the width, and the centre up to a period. */
          double dc = x.centre[k] - (a + 0.5 * E);
          dc -= L * nearbyint(dc / L);
          if (x.covers[k] || fabs(2.0 * x.half[k] - E) > 1e-12 * L || fabs(dc) > 1e-12 * L) ok = 0;
          n_exact++;

          double c2, h2;
          bool full2;
          two_frame_resolve(dom, acc, k, &c2, &h2, &full2);
          if (full2) n_two_frame_misses++;
          if (!full2 && 2.0 * h2 < MFTL_PERIODIC_EXTENT_COVER_FRACTION * L)
          {
            /* Wherever the old measurement was exact, the new one is the SAME BITS. */
            n_compared++;
            if (memcmp(&c2, &x.centre[k], sizeof(double)) == 0 && memcmp(&h2, &x.half[k], sizeof(double)) == 0) n_identical++;
          }
        }
        else
        {
          if (!x.covers[k]) ok = 0;
          n_cover++;
        }

        /* The two other axes are a point: never covering, zero width. */
        for (int m = 0; m < 3; ++m)
          if (m != k && (x.covers[m] || x.half[m] > 0.02)) ok = 0;
      }
    }
  }
  if (g_rank == 0 && !ok) printf("\n  a chain was mismeasured\n  ");
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n_exact > 0 && n_cover > 0, MPI_COMM_WORLD));
  /* NON-VACUITY of the regression: placements the two-frame code called
   * "covers the axis" although the chain is compact -- and the four frames
   * measured exactly above. */
  TEST_ASSERT(agreed(n_two_frame_misses > 50, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n_compared > 1000 && n_identical == n_compared, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_closed_ring_covers_the_axis_at_every_offset(void)
{
  const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  MFTL_Ensemble *ens = make_ens(per);
  const MFTL_Domain *dom = &ens->domain;
  int ok = 1;
  for (int k = 0; k < 3; ++k)
  {
    const double L = dom->domain_size[k];
    for (int i = 0; i < 101; ++i)
    {
      double acc[MFTL_PERIODIC_EXTENT_SLOTS];
      MFTLPeriodicExtentInit(acc);
      add_chain(dom, k, dom->global_min[k] + L * (double) i / 101.0, L, 37, acc);  // closes on itself
      MFTL_PeriodicExtent x;
      MFTLPeriodicExtentResolve(dom, acc, &x);
      if (!x.covers[k]) ok = 0;
    }
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_an_empty_or_non_periodic_extent_is_plain(void)
{
  const bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *ens = make_ens(per);
  const MFTL_Domain *dom = &ens->domain;

  double acc[MFTL_PERIODIC_EXTENT_SLOTS];
  MFTLPeriodicExtentInit(acc);
  MFTL_PeriodicExtent x;
  MFTLPeriodicExtentResolve(dom, acc, &x);
  int ok = x.empty;
  for (int k = 0; k < 3; ++k) ok = ok && !x.covers[k] && x.centre[k] == 0.0 && x.half[k] == 0.0;

  /* Non-periodic: plain min/max, and the shifted frames are never written. */
  add_chain(dom, 0, -0.5, 1.7, 17, acc);
  MFTLPeriodicExtentResolve(dom, acc, &x);
  ok = ok && !x.empty && !x.covers[0] && fabs(x.centre[0] - (-0.5 + 0.85)) < 1e-14 && fabs(x.half[0] - 0.85) < 1e-14;
  for (int i = 6; i < MFTL_PERIODIC_EXTENT_SLOTS; ++i) ok = ok && acc[i] == MFTL_DBL_MAX;

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. The check, on real bodies
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief An icosphere stretched to vertex extent @p ext about @p c, and its true
 *        extent, measured from the vertices by the minimum image about @p c.
 *
 * Generated on rank 0 (MFTLGenerateSphere() is collective); the true extent is
 * broadcast so every rank knows the expected verdict.
 */
static void add_stretched(MFTL_Ensemble *ens, const double c[3], const double ext[3], double truth[3])
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(obj, cc, 0.25, 3);

  double t[3] = {0.0, 0.0, 0.0};
  if (g_rank == 0)
  {
    const MFTL_Domain *dom = &ens->domain;
    /* The sphere's own vertex extent, then the scale that makes it ext[k]. */
    double lo[3] = {1e300, 1e300, 1e300}, hi[3] = {-1e300, -1e300, -1e300};
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double d[3];
      MFTLGetMinimumImageVector(dom, d, c, &obj->verts.pos[3 * v]);
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(lo[k], d[k]);
        hi[k] = fmax(hi[k], d[k]);
      }
    }
    double s[3];
    for (int k = 0; k < 3; ++k)
    {
      s[k] = ext[k] / (hi[k] - lo[k]);
      lo[k] = 1e300;
      hi[k] = -1e300;
    }
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double d[3];
      double *p = &obj->verts.pos[3 * v];
      MFTLGetMinimumImageVector(dom, d, c, p);
      for (int k = 0; k < 3; ++k)
      {
        const double u = c[k] + s[k] * d[k];
        lo[k] = fmin(lo[k], u);
        hi[k] = fmax(hi[k], u);
        p[k] = u;
      }
      MFTLApplyPeriodicity(dom, p);
    }
    for (int k = 0; k < 3; ++k) t[k] = hi[k] - lo[k];
  }
  MPI_Bcast(t, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int k = 0; k < 3; ++k) truth[k] = t[k];
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
}

/** @brief Bodies held by two or more ranks: the non-vacuity measure of a holder reduction. */
static int shared_bodies(MFTL_Ensemble *ens)
{
  const int b = ens->n_local_objects;
  double *r = (double *) malloc((size_t) (2 * b + 1) * sizeof(double));
  const MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MAX};
  for (int o = 0; o < b; ++o) r[2 * o] = r[2 * o + 1] = (double) g_rank;
  MFTLEnsembleReduceBodies(ens, r, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
  int n = 0;
  for (int o = 0; o < b; ++o)
    if ((int) r[2 * o] == g_rank && r[2 * o + 1] > r[2 * o]) n++;
  free(r);
  int total = 0;
  MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return total;
}

/** @brief The report must be the same bits on every rank. */
static int report_is_agreed(const MFTL_PeriodicExtentReport *r)
{
  double mine[5] = {r->n_violations, r->n_covering, r->first_body_gid, r->first_axis, r->first_extent};
  double lo[5], hi[5];
  MPI_Allreduce(mine, lo, 5, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(mine, hi, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return memcmp(lo, hi, sizeof lo) == 0;
}

/**
 * @brief Eight bodies elongated along one periodic axis, placed so that every
 *        one wider than L/2 straddles both the seam and the middle of the axis
 *        -- the placement the two-frame measurement could not see -- plus one
 *        compact body wider than 3L/4, which is reported as covering.
 */
static int test_the_check_flags_exactly_the_bodies_reaching_half_the_period(void)
{
  /* Extent and centre along the long axis, both as fractions of the period. */
  static const double frac[8] = {0.30, 0.45, 0.49, 0.52, 0.60, 0.70, 0.80, 0.40};
  static const double cfrac[8] = {0.00, 0.25, 0.50, 0.25, 0.75, 0.10, 0.40, 0.90};

  int shared_total = 0;
  for (int axis = 0; axis < 3; ++axis)
  {
    const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
    MFTL_Ensemble *ens = make_ens(per);
    TEST_ASSERT(ens != MFTL_NULL);
    const MFTL_Domain *dom = &ens->domain;
    const double L = dom->domain_size[axis];

    double truth[8][3];
    int expect_viol = 0, expect_cover = 0, expect_first = -1;
    for (int b = 0; b < 8; ++b)
    {
      double c[3], ext[3];
      for (int k = 0; k < 3; ++k)
      {
        c[k] = dom->global_min[k] + dom->domain_size[k] * fmod(0.13 + 0.37 * b, 1.0);
        ext[k] = 0.3 * dom->domain_size[k];
      }
      c[axis] = dom->global_min[axis] + cfrac[b] * L;
      ext[axis] = frac[b] * L;
      add_stretched(ens, c, ext, truth[b]);

      const double E = truth[b][axis];
      if (E >= MFTL_PERIODIC_EXTENT_COVER_FRACTION * L)
        expect_cover++;
      else if (E >= MFTL_PERIODIC_EXTENT_RULE_FRACTION * L)
      {
        expect_viol++;
        if (expect_first < 0) expect_first = b;
      }
    }
    MFTLRedistribute(ens);
    shared_total += shared_bodies(ens);

    MFTL_PeriodicExtentReport rep;
    const int rc = MFTLEnsembleCheckPeriodicExtents(ens, &rep);

    TEST_ASSERT(agreed(expect_viol == 3 && expect_cover == 1 && expect_first == 3, MPI_COMM_WORLD));  // the fixture is what it claims
    TEST_ASSERT(agreed(rc == MFTL_ERROR, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(report_is_agreed(&rep), MPI_COMM_WORLD));
    if (g_rank == 0 && !(rep.n_violations == 3 && rep.n_covering == 1 && rep.first_body_gid == 3 && rep.first_axis == axis))
      printf("\n  axis %d: %d violations, %d covering, first %d along %d\n  ", axis, rep.n_violations, rep.n_covering, rep.first_body_gid, rep.first_axis);
    TEST_ASSERT(agreed(rep.n_violations == expect_viol && rep.n_covering == expect_cover, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(rep.first_body_gid == expect_first && rep.first_axis == axis, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(rep.first_extent - truth[expect_first][axis]) <= 1e-12 * L, MPI_COMM_WORLD));
    MFTLEnsembleDestroy(&ens);
  }
  /* Non-vacuity: the verdicts above were reduced over several holders. */
  TEST_ASSERT(agreed(g_nproc == 1 || shared_total > 0, MPI_COMM_WORLD));
  return 0;
}

static int test_bodies_under_half_the_period_pass_at_every_offset(void)
{
  const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  MFTL_Ensemble *ens = make_ens(per);
  const MFTL_Domain *dom = &ens->domain;

  /* 64 offsets per axis, every axis at 0.49 of its period. */
  double worst = 0.0;
  for (int b = 0; b < 64; ++b)
  {
    double c[3], ext[3], truth[3];
    const int mult[3] = {1, 17, 41};
    for (int k = 0; k < 3; ++k)
    {
      c[k] = dom->global_min[k] + dom->domain_size[k] * (double) ((b * mult[k]) % 64) / 64.0;
      ext[k] = 0.49 * dom->domain_size[k];
    }
    add_stretched(ens, c, ext, truth);
    for (int k = 0; k < 3; ++k) worst = fmax(worst, truth[k] / dom->domain_size[k]);
  }
  MFTLRedistribute(ens);
  const int shared = shared_bodies(ens);

  MFTL_PeriodicExtentReport rep;
  const int rc = MFTLEnsembleCheckPeriodicExtents(ens, &rep);
  TEST_ASSERT(agreed(worst < 0.5, MPI_COMM_WORLD));  // the fixture obeys the rule
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS && rep.n_violations == 0 && rep.n_covering == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(rep.first_body_gid == -1 && rep.first_axis == -1 && rep.first_extent == 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || shared > 0, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A level-set film z = z0 + A sin(2 pi x / Lx) cos(2 pi y / Ly) over the
 *        whole box, built on rank 0: closed on the torus when y is periodic, a
 *        ribbon with its rims on the y walls when it is not.
 */
static void add_film(MFTL_Ensemble *ens)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (g_rank == 0)
  {
    const MFTL_Domain *dom = &ens->domain;
    int n[3] = {40, 26, 31};
    double sp[3];
    bool per[3];
    for (int k = 0; k < 3; ++k)
    {
      per[k] = dom->is_periodic[k];
      if (!per[k]) n[k] += 1;  // a node on each wall
      sp[k] = dom->domain_size[k] / (per[k] ? n[k] : n[k] - 1);
    }
    const int lo[3] = {0, 0, 0};
    MFTL_SdfGrid g;
    MFTLSdfGridInitWindow(&g, dom->global_min, sp, n, per, lo, n);
    for (int k = 0; k < n[2]; ++k)
      for (int j = 0; j < n[1]; ++j)
        for (int i = 0; i < n[0]; ++i)
        {
          const double x = i * sp[0], y = j * sp[1], z = k * sp[2];
          g.phi[((size_t) k * n[1] + j) * n[0] + i] =
              z - (0.43 * dom->domain_size[2] + 0.2 * sin(2.0 * MFTL_PI * x / dom->domain_size[0]) * cos(2.0 * MFTL_PI * y / dom->domain_size[1]));
        }
    MFTLBuildMeshFromLevelSet(obj, &g, 1e-2);
    MFTLSdfGridFree(&g);
  }
  MFTLPublishGeneratedBodies(ens);
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const p = ens->local_objects[o];
    if (!MFTLObjectHasTopology(p)) MFTLObjectBuildDerived(p);
    MFTLComputeTriangleGeometry(p, p->tris.n);
  }
}

static int test_a_wrapping_film_covers_its_axes_and_is_no_violation(void)
{
  for (int ribbon = 0; ribbon < 2; ++ribbon)
  {
    const bool per[3] = {MFTL_TRUE, ribbon ? MFTL_FALSE : MFTL_TRUE, MFTL_FALSE};
    MFTL_Ensemble *ens = make_ens(per);
    add_film(ens);
    /* And a small compact droplet beside it, which neither covers nor violates. */
    const double c[3] = {0.1, 0.9, 0.9}, ext[3] = {0.3, 0.3, 0.3};
    double truth[3];
    add_stretched(ens, c, ext, truth);
    MFTLRedistribute(ens);
    const int shared = shared_bodies(ens);

    /* What the film is, on every rank that holds a piece of it. */
    int film_ok = 1;
    MFTL_Object *film = test_body(ens, 0);
    double acc[MFTL_PERIODIC_EXTENT_SLOTS + 1];
    MFTLPeriodicExtentInit(acc);
    MFTLPeriodicExtentAddObject(film, acc);
    MFTL_BodyReduceOp ops[MFTL_PERIODIC_EXTENT_SLOTS];
    for (int j = 0; j < MFTL_PERIODIC_EXTENT_SLOTS; ++j) ops[j] = MFTL_BODY_REDUCE_MIN;
    {
      /* Reduce the film's row over its holders, by local slot. */
      const int b = ens->n_local_objects;
      double *rows = (double *) malloc((size_t) MFTL_MAX(b, 1) * MFTL_PERIODIC_EXTENT_SLOTS * sizeof(double));
      for (int o = 0; o < b; ++o)
      {
        MFTLPeriodicExtentInit(&rows[o * MFTL_PERIODIC_EXTENT_SLOTS]);
        MFTLPeriodicExtentAddObject(ens->local_objects[o], &rows[o * MFTL_PERIODIC_EXTENT_SLOTS]);
      }
      MFTLEnsembleReduceBodies(ens, rows, MFTL_PERIODIC_EXTENT_SLOTS, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
      const int slot = MFTLEnsembleFindLocalSlot(ens, 0);
      if (slot >= 0)
      {
        MFTL_PeriodicExtent x;
        MFTLPeriodicExtentResolve(&ens->domain, &rows[slot * MFTL_PERIODIC_EXTENT_SLOTS], &x);
        film_ok = x.covers[0] && (ribbon ? !x.covers[1] : x.covers[1]) && !x.covers[2];
      }
      free(rows);
    }

    MFTL_PeriodicExtentReport rep;
    const int rc = MFTLEnsembleCheckPeriodicExtents(ens, &rep);
    TEST_ASSERT(agreed(film_ok, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(rc == MFTL_SUCCESS && rep.n_violations == 0 && rep.n_covering == 1, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(report_is_agreed(&rep), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(g_nproc == 1 || shared > 0, MPI_COMM_WORLD));
    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

static int test_a_domain_without_periodic_axes_makes_no_collective(void)
{
  const bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *ens = make_ens(per);
  const double c[3] = {0.3, 0.85, -0.35}, ext[3] = {1.9, 1.2, 3.0};  // wider than half on every axis
  double truth[3];
  add_stretched(ens, c, ext, truth);
  MFTLRedistribute(ens);

  MFTL_BodyReduceStats before, after;
  MFTLEnsembleGetBodyReduceStats(&before);
  MFTL_PeriodicExtentReport rep;
  const int rc = MFTLEnsembleCheckPeriodicExtents(ens, &rep);
  MFTLEnsembleGetBodyReduceStats(&after);

  TEST_ASSERT(agreed(rc == MFTL_SUCCESS && rep.n_violations == 0 && rep.n_covering == 0 && rep.first_body_gid == -1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.n_holders == before.n_holders && after.n_global == before.n_global, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLEnsembleCheckPeriodicExtents(MFTL_NULL, &rep) == MFTL_ERROR, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Periodic extents (np = %d) ===\n", g_nproc);

  RUN_TEST(test_a_chain_is_measured_exactly_below_three_quarters_at_every_offset);
  RUN_TEST(test_a_closed_ring_covers_the_axis_at_every_offset);
  RUN_TEST(test_an_empty_or_non_periodic_extent_is_plain);
  RUN_TEST(test_the_check_flags_exactly_the_bodies_reaching_half_the_period);
  RUN_TEST(test_bodies_under_half_the_period_pass_at_every_offset);
  RUN_TEST(test_a_wrapping_film_covers_its_axes_and_is_no_violation);
  RUN_TEST(test_a_domain_without_periodic_axes_makes_no_collective);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
