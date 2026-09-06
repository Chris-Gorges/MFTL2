/*
 * @Author: Christian Gorges
 * @Date: 2026-09-04
 *
 * @brief Unit tests for MFTLEnsembleReduceBodies -- the per-body reduction.
 *
 * The function has two routes to the same answer: an MPI_Neighbor_allgatherv
 * over each body's HOLDERS, and the comm_world MPI_Allreduce over a buffer
 * indexed by body id that it exists to stop taking. Everything here is built
 * around that, because it is what makes the module testable at all:
 *
 *   THE TWO ROUTES ARE EACH OTHER'S CROSS-CHECK, AND BOTH ARE CHECKED AGAINST A
 *   THIRD IMPLEMENTATION WRITTEN HERE. reference_reduce() below shares no code
 *   with either -- it stages by id, fills identities and calls MPI_Allreduce
 *   directly -- so a bug in the machinery both routes share cannot hide by
 *   appearing on both sides. The comparison is BITWISE, not approximate: the
 *   holder route sums in a canonical rank order precisely so that it can be, and
 *   a tolerance would let exactly the ordering bug it guards against through.
 *
 * THE PAYLOAD IS CHOSEN TO EXPOSE A MISSING CONTRIBUTOR. Slot 4 is a SUM of 1.0
 * per holder, i.e. the holder count, and slots 0/1 are the MIN and MAX of the
 * contributing rank id. A dropped contribution changes all three, whereas a
 * payload built only from mesh quantities can hide one behind another rank
 * holding a similar share.
 *
 * NON-VACUITY. A fixture in which every body sits on exactly one rank tests the
 * exchange and nothing else -- both routes would agree with the halo deleted. So
 * every test here reports how many bodies genuinely have more than one holder,
 * and asserts it, in the spirit of tests/test_bvh_ghosting.c.
 *
 * Registered at five process counts, and each reaches something the others do
 * not:
 *
 *   1    no decomposition. The route is the identity and nothing is sent, which
 *        is the one case where a missed collective would NOT hang -- so it is
 *        where an accidental early return hides
 *   2    2x1x1, and 4 -> 2x2x1: grids with an axis of a single cut, where a
 *        periodic wrap maps a non-zero stencil offset back onto this rank and
 *        comm_graph therefore carries a SELF-LOOP. Folding that block in twice
 *        is invisible to MIN and MAX and doubles every SUM
 *   8    2x2x2: every one of the 26 offsets is a distinct rank pair, and a body
 *        on a corner has seven other holders
 *   27   3x3x3: the first grid with an interior rank whose 26 neighbours are 26
 *        DISTINCT ranks, and the first where three cells on an axis make the
 *        offset span test reject a body that is in fact well placed
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/** @brief Values in one test record, and the operators over them. */
#define N_SLOTS 5

static const MFTL_BodyReduceOp g_ops[N_SLOTS] = {MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MIN,
                                                 MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

static void grid_for(int nproc, int cuts[3])
{
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
  if (nproc == 4) { cuts[0] = 2; cuts[1] = 2; cuts[2] = 1; }
  else if (nproc == 8) { cuts[0] = 2; cuts[1] = 2; cuts[2] = 2; }
  else if (nproc == 27) { cuts[0] = 3; cuts[1] = 3; cuts[2] = 3; }
}

static MFTL_Ensemble *make_ensemble(void)
{
  int cuts[3];
  grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};

  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
}

/** @brief Bodies in the fixture. Twelfths of the domain hit both a 2-cut and a 3-cut plane. */
#define N_BODIES 12

/**
 * @brief Twelve droplets on a lattice of twelfths, then one redistribution.
 *
 * The stride is 1/6 of the domain, so some centres land exactly on a cut plane
 * at cuts = 2 (x = 0) and others on one at cuts = 3 (x = +-1/3), and one sits on
 * the periodic seam. That is what produces bodies with several holders at every
 * rank count without the placement having to know the grid.
 *
 * The radius is 0.2 against a rank box of at least 2/3, so every body fits
 * inside a 2x2x2 block of cells and the holder route is available -- which is
 * the regime this function exists for.
 */
static MFTL_Ensemble *make_droplets(void)
{
  MFTL_Ensemble *ens = make_ensemble();

  for (int i = 0; i < N_BODIES; ++i)
  {
    MFTL_Object *obj = MFTLObjectCreate(i);
    MFTLEnsembleAddObject(ens, obj);

    double c[3] = {-1.0 + 2.0 * (double) i / (double) N_BODIES, -1.0 + 2.0 * (double) ((i * 5) % N_BODIES) / (double) N_BODIES,
                         -1.0 + 2.0 * (double) ((i * 7) % N_BODIES) / (double) N_BODIES};
    MFTLGenerateSphere(obj, c, 0.2, 1);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  MFTLRedistribute(ens);
  return ens;
}

/** @brief Fills this rank's contribution for every local body. */
static void fill_contribution(const MFTL_Ensemble *ens, double *rows)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    double *const r = &rows[(size_t) o * N_SLOTS];

    r[0] = (double) g_rank;  // MIN: the lowest-numbered holder
    r[1] = (double) g_rank;  // MAX: the highest
    r[2] = (double) obj->tris.n_local;  // MIN: the smallest share
    r[3] = 1.0;  // SUM: how many holders
    r[4] = (double) obj->tris.n_local;  // SUM: the whole body
  }
}

/**
 * @brief The answer, computed by a route that shares nothing with either of them.
 *
 * Staged by body id, identities everywhere this rank holds nothing, one
 * MPI_Allreduce per operator over comm_world. It is `O(bodies)` on every rank
 * and would not scale, which is the entire point of the code it is checking --
 * a reference is allowed to be the implementation that was replaced.
 *
 * @param ens  The ensemble.
 * @param rows [n_local_objects * N_SLOTS] contribution in, reduction out.
 */
static void reference_reduce(const MFTL_Ensemble *ens, double *rows)
{
  const int nb = ens->n_bodies_global;
  double *loc = (double *) malloc((size_t) nb * N_SLOTS * sizeof(double));
  double *glo = (double *) malloc((size_t) nb * N_SLOTS * sizeof(double));

  for (int g = 0; g < nb; ++g)
  {
    for (int j = 0; j < N_SLOTS; ++j)
    {
      loc[(size_t) g * N_SLOTS + j] = (g_ops[j] == MFTL_BODY_REDUCE_MIN) ? MFTL_DBL_MAX : ((g_ops[j] == MFTL_BODY_REDUCE_MAX) ? -MFTL_DBL_MAX : 0.0);
    }
  }

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    for (int j = 0; j < N_SLOTS; ++j) loc[(size_t) g * N_SLOTS + j] = rows[(size_t) o * N_SLOTS + j];
  }

  for (int j = 0; j < N_SLOTS; ++j)
  {
    const MPI_Op op = (g_ops[j] == MFTL_BODY_REDUCE_MIN) ? MPI_MIN : ((g_ops[j] == MFTL_BODY_REDUCE_MAX) ? MPI_MAX : MPI_SUM);
    for (int g = 0; g < nb; ++g)
    {
      double in = loc[(size_t) g * N_SLOTS + j], out = 0.0;
      MPI_Allreduce(&in, &out, 1, MPI_DOUBLE, op, MPI_COMM_WORLD);
      glo[(size_t) g * N_SLOTS + j] = out;
    }
  }

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    for (int j = 0; j < N_SLOTS; ++j) rows[(size_t) o * N_SLOTS + j] = glo[(size_t) g * N_SLOTS + j];
  }

  free(loc);
  free(glo);
}

/** @brief Reductions that took each route since the last reset. */
static void route_split(long out[2])
{
  MFTL_BodyReduceStats st;
  MFTLEnsembleGetBodyReduceStats(&st);
  out[0] = st.n_holders;
  out[1] = st.n_global;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Tests
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Both routes reproduce the reference, to the bit.
 *
 * The comparison is memcmp on the doubles. A tolerance would pass while the
 * holder route folded contributions in receive order, which differs between
 * ranks and is the one ordering bug the canonical rank order exists to prevent.
 */
static int test_both_routes_match_the_reference(void)
{
  MFTL_Ensemble *ens = make_droplets();
  const int n = ens->n_local_objects;
  const size_t bytes = (size_t) MFTL_MAX(n * N_SLOTS, 1) * sizeof(double);

  double *ref = (double *) malloc(bytes);
  double *via_holders = (double *) malloc(bytes);
  double *via_global = (double *) malloc(bytes);
  double *via_auto = (double *) malloc(bytes);

  fill_contribution(ens, ref);
  memcpy(via_holders, ref, bytes);
  memcpy(via_global, ref, bytes);
  memcpy(via_auto, ref, bytes);

  reference_reduce(ens, ref);
  MFTLEnsembleReduceBodies(ens, via_holders, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_HOLDERS);
  MFTLEnsembleReduceBodies(ens, via_global, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_GLOBAL);
  MFTLEnsembleReduceBodies(ens, via_auto, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  int ok = (memcmp(ref, via_holders, (size_t) n * N_SLOTS * sizeof(double)) == 0);
  ok = ok && (memcmp(ref, via_global, (size_t) n * N_SLOTS * sizeof(double)) == 0);
  ok = ok && (memcmp(ref, via_auto, (size_t) n * N_SLOTS * sizeof(double)) == 0);

  /* NON-VACUITY: the holder count in slot 3. Bodies with only one holder would
   * make every route agree with the halo removed. */
  int shared = 0, max_holders = 0;
  for (int o = 0; o < n; ++o)
  {
    const int h = (int) ref[(size_t) o * N_SLOTS + 3];
    if (h > 1) ++shared;
    if (h > max_holders) max_holders = h;
  }

  int shared_total = 0, holders_max = 0;
  MPI_Allreduce(&shared, &shared_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&max_holders, &holders_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (g_rank == 0)
  {
    printf("\n    np=%d: %d body-on-rank entries have several holders (%d bodies), widest body has %d\n", g_nproc, shared_total, N_BODIES, holders_max);
  }

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(g_nproc == 1 ? (holders_max == 1) : (holders_max > 1));

  free(via_auto);
  free(via_global);
  free(via_holders);
  free(ref);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Every holder of a body ends up with the SAME BITS, not merely the same value.
 *
 * A per-body statistic is what a solver branches on to decide whether to remesh
 * that body, and that decision is collective. Two holders disagreeing in the
 * last ulp is therefore not a rounding curiosity: it is two ranks entering
 * different collectives. MPI_Allreduce gives this for free; the holder route has
 * to arrange it, by folding contributions in ascending source-rank order.
 */
static int test_every_holder_gets_the_same_bits(void)
{
  MFTL_Ensemble *ens = make_droplets();
  const int n = ens->n_local_objects;
  const int nb = ens->n_bodies_global;

  double *rows = (double *) malloc((size_t) MFTL_MAX(n * N_SLOTS, 1) * sizeof(double));
  fill_contribution(ens, rows);
  MFTLEnsembleReduceBodies(ens, rows, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* One 64-bit digest per body, reduced twice. A holder contributes its digest,
   * a non-holder the identity, so MIN == MAX exactly when every holder agreed. */
  uint64_t *lo = (uint64_t *) malloc((size_t) nb * sizeof(uint64_t));
  uint64_t *hi = (uint64_t *) malloc((size_t) nb * sizeof(uint64_t));
  for (int g = 0; g < nb; ++g)
  {
    lo[g] = UINT64_MAX;
    hi[g] = 0;
  }

  for (int o = 0; o < n; ++o)
  {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *p = (const unsigned char *) &rows[(size_t) o * N_SLOTS];
    for (size_t b = 0; b < N_SLOTS * sizeof(double); ++b)
    {
      h ^= p[b];
      h *= 1099511628211ull;
    }
    const int g = ens->local_objects[o]->global_id;
    lo[g] = h;
    hi[g] = h;
  }

  MPI_Allreduce(MPI_IN_PLACE, lo, nb, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, hi, nb, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);

  int ok = 1;
  for (int g = 0; g < nb; ++g)
  {
    if (hi[g] == 0) continue;  // held by nobody, which cannot happen here but says so
    if (lo[g] != hi[g]) ok = 0;
  }

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  free(hi);
  free(lo);
  free(rows);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A generated ensemble, before its first redistribution, takes the global route.
 *
 * Rank 0 holds every entity and every other rank holds an EMPTY object. Such a
 * rank owns nothing, so it appears in no other holder's halo and cannot be
 * discovered -- the one hole in the local test that has to be closed from the
 * inside, by the empty holder vetoing.
 *
 * The failure this pins is not hypothetical: it is what made
 * test_ensemble_update_tolerates_empty_ranks fail at np = 4, where rank 2 is not
 * a neighbour of rank 0 on a 4x1x1 grid and would have kept its identities.
 */
static int test_an_empty_holder_forces_the_global_route(void)
{
  MFTL_Ensemble *ens = make_ensemble();

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.1, 0.1, 0.1};
  MFTLGenerateSphere(obj, c, 0.3, 1);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* Deliberately NOT redistributed: this is the state a solver's first timestep
   * starts in. */
  const bool fits = MFTLEnsembleBodyHoldersFitTheGraph(ens);

  MFTLEnsembleResetBodyReduceStats();
  double rows[N_SLOTS];
  fill_contribution(ens, rows);
  MFTLEnsembleReduceBodies(ens, rows, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  long split[2];
  route_split(split);

  /* Every rank holds the body, so every rank must see the same holder count. */
  const int holders = (int) rows[3];

  if (g_rank == 0) printf("\n    np=%d: unredistributed -> probe %s, route %s, %d holders\n", g_nproc, fits ? "passes" : "vetoes",
                          split[1] > 0 ? "global" : "holders", holders);

  TEST_ASSERT(agreed(holders == g_nproc, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 ? (split[0] + split[1] == 0) : (split[1] == 1), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || !fits || g_rank == 0, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A body wider than a rank box takes the global route, and is still exact.
 *
 * The regime the holder route cannot serve: a sphere spanning the whole domain
 * has holders two cells apart, so some rank's offset span reaches 2 and the
 * fallback is what keeps the answer right. It is registered here rather than
 * left to the remeshing suite because the FALLBACK IS THE PART THAT IS EASY TO
 * LEAVE UNEXERCISED -- a droplet fixture never reaches it.
 */
static int test_a_spanning_body_falls_back_and_stays_exact(void)
{
  MFTL_Ensemble *ens = make_ensemble();

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 0.9, 2);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLRedistribute(ens);

  const int n = ens->n_local_objects;
  const size_t bytes = (size_t) MFTL_MAX(n * N_SLOTS, 1) * sizeof(double);
  double *ref = (double *) malloc(bytes);
  double *got = (double *) malloc(bytes);
  fill_contribution(ens, ref);
  memcpy(got, ref, bytes);

  reference_reduce(ens, ref);

  MFTLEnsembleResetBodyReduceStats();
  MFTLEnsembleReduceBodies(ens, got, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
  long split[2];
  route_split(split);

  const int ok = (memcmp(ref, got, (size_t) n * N_SLOTS * sizeof(double)) == 0);
  const int holders = (n > 0) ? (int) ref[3] : 0;
  int holders_max = 0;
  MPI_Allreduce(&holders, &holders_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  if (g_rank == 0) printf("\n    np=%d: r=0.9 sphere -> %d holders, route %s\n", g_nproc, holders_max, split[1] > 0 ? "global" : "holders");

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY, and the measure is "more holders than a 26-neighbourhood can
   * contain" rather than "every rank". A 2x2x2 block of cells is 8 ranks, so
   * above that the holder set provably does not fit and the fallback is the only
   * thing that can serve the body. It is NOT all 27 at np = 27: the shell of
   * radius 0.9 misses the central cell entirely, which is the right answer and
   * was the first form of this assertion. */
  TEST_ASSERT(agreed(g_nproc <= 8 || holders_max > 8, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc <= 8 || split[1] == 1, MPI_COMM_WORLD));

  free(got);
  free(ref);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A rank holding no bodies at all still enters the collective.
 *
 * One droplet in one cell, so at any rank count above one most ranks hold
 * nothing. The guard inside has to read the GLOBAL body count: on the local one
 * the empty ranks would return early and the holders would block in a
 * neighbourhood collective nobody else entered. A hang is the failure, so the
 * test is that this returns at all.
 */
static int test_a_rank_with_no_bodies_still_participates(void)
{
  MFTL_Ensemble *ens = make_ensemble();

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {-0.85, -0.85, -0.85};
  MFTLGenerateSphere(obj, c, 0.08, 1);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLRedistribute(ens);

  const int n = ens->n_local_objects;
  double *rows = (double *) malloc((size_t) MFTL_MAX(n * N_SLOTS, 1) * sizeof(double));
  fill_contribution(ens, rows);
  MFTLEnsembleReduceBodies(ens, rows, N_SLOTS, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* A no-op width must not become an early return either: it is agreed on every
   * rank, so it may skip the collective, but only if EVERY rank skips it. */
  MFTLEnsembleReduceBodies(ens, rows, 0, g_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  const int empty_here = (n == 0) ? 1 : 0;
  int empty_ranks = 0;
  MPI_Allreduce(&empty_here, &empty_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  int ok = 1;
  for (int o = 0; o < n; ++o) ok = ok && ((int) rows[(size_t) o * N_SLOTS + 3] >= 1);

  if (g_rank == 0) printf("\n    np=%d: %d of %d ranks hold nothing\n", g_nproc, empty_ranks, g_nproc);

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(g_nproc == 1 ? (empty_ranks == 0) : (empty_ranks > 0));

  free(rows);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The per-body statistics and counts still describe the whole body.
 *
 * The end-to-end statement, through the two library functions that now go
 * through the reduction: after one timestep's worth of calls, every holder of a
 * body must report the SAME global counts and the same quality extremes, and
 * they must match a direct reduction over the ensemble.
 */
static int test_the_library_consumers_still_see_the_whole_body(void)
{
  MFTL_Ensemble *ens = make_droplets();

  MFTLEnsembleUpdate(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  const int nb = ens->n_bodies_global;
  int *tri_lo = (int *) malloc((size_t) nb * sizeof(int));
  int *tri_hi = (int *) malloc((size_t) nb * sizeof(int));
  int *owned = (int *) calloc((size_t) nb, sizeof(int));
  for (int g = 0; g < nb; ++g)
  {
    tri_lo[g] = 1 << 30;
    tri_hi[g] = 0;
  }

  int ok = 1;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const int g = obj->global_id;
    tri_lo[g] = obj->global_n_triangles;
    tri_hi[g] = obj->global_n_triangles;
    owned[g] = obj->tris.n_local;
    /* Refine 1 is 80 triangles; every body is one whole sphere. */
    ok = ok && (obj->global_n_triangles == 80);
    ok = ok && (obj->quality.min_triangle_area > 0.0);
    ok = ok && (obj->quality.max_aspect_ratio <= 1.0 + 1e-12);
    ok = ok && (obj->quality.min_aspect_ratio > 0.5);
  }

  MPI_Allreduce(MPI_IN_PLACE, tri_lo, nb, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, tri_hi, nb, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, owned, nb, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  for (int g = 0; g < nb; ++g)
  {
    ok = ok && (tri_hi[g] == 80) && (tri_lo[g] == 80);  // every holder agrees
    ok = ok && (owned[g] == 80);  // and it really is the sum of the shares
  }

  ok = ok && (ens->total_global_triangles == 80 * N_BODIES);

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  free(owned);
  free(tri_hi);
  free(tri_lo);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief In the droplet regime a whole timestep takes no body-proportional comm_world collective.
 *
 * THE GATE OF THIS WHOLE CHANGE, stated as a test rather than as a measurement.
 * Every per-body reduction a timestep makes -- the redistribution skip vote,
 * MFTLEnsembleUpdate's eleven statistics, MFTLEnsembleUpdateGlobalCounts' two
 * counts -- must take the holder route on a fixture where every body fits in a
 * rank box. One that does not is not slower by a constant: its payload grows
 * with the body count on every rank, so it is what makes a weak-scaling sweep
 * bend.
 */
static int test_a_droplet_timestep_takes_no_global_reduction(void)
{
  MFTL_Ensemble *ens = make_droplets();

  /* Two settling steps first: a generated ensemble starts with every body on
   * rank 0 and empty shells everywhere else, which correctly vetoes. */
  for (int s = 0; s < 2; ++s)
  {
    MFTLUpdateGhostPositions(ens);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    MFTLRedistribute(ens);
  }

  MFTLEnsembleResetBodyReduceStats();

  MFTLUpdateGhostPositions(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);
  MFTLEnsembleUpdate(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  long split[2];
  route_split(split);

  if (g_rank == 0) printf("\n    np=%d: one timestep -> %ld holder reductions, %ld global\n", g_nproc, split[0], split[1]);

  /* At one rank the reduction is the identity and neither counter moves, which
   * is the right answer and not the one being asserted. */
  TEST_ASSERT(agreed(g_nproc == 1 ? (split[0] == 0) : (split[0] == 3), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(split[1] == 0, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A distributed remeshing pass takes no body-proportional comm_world collective either.
 *
 * THE SIBLING OF THE TEST ABOVE, AND IT EXISTS BECAUSE NOTHING ELSE CAN SEE THE
 * FAILURE IT GUARDS. The spanning driver hands out per-phase global id blocks
 * with MFTLEnsembleExscanBodies, and that call site spent a while pinned to
 * #MFTL_BODY_REDUCE_SCOPE_GLOBAL. Every mesh digest in
 * docs/parallel-remeshing-analysis.md reproduced bit for bit while it was, and
 * every remeshing test passed -- BECAUSE THE TWO ROUTES COMPUTE THE SAME
 * ANSWER. That is the whole point of them, so no test of the ANSWER can ever
 * distinguish them; the only observable is which communicator was used, and
 * this is where it is observed.
 *
 * The regression it prevents is a scaling one and therefore invisible on this
 * machine: `exscan_over_comm_world` stages `3 x n_bodies_global x n` doubles per
 * rank and sends two collectives of a third of that, ONCE PER PHASE, four to
 * eight times per remesh. At a million bodies that is 48 MB of staging and 32 MB
 * on the wire per rank per phase, to hand out ids for the hundred bodies the
 * rank holds.
 *
 * SPANNING DRIVER ON A DISPERSED FIXTURE, deliberately. The strategy and the
 * route are independent questions -- the strategy asks how big a body is
 * relative to a rank box, the route asks whether a body's holders fit in one
 * 26-neighbourhood -- and this isolates the second. It is not a contrived
 * combination: MFTLRemeshEnsemble dispatches per ENSEMBLE, so one body wider
 * than a rank box sends a million droplets down the spanning driver, and
 * MFTLRemeshEnsembleParallel is a public entry point in its own right.
 *
 * A body genuinely wider than a rank box has holders that do not fit and takes
 * the global route; that is the correct answer and
 * test_a_spanning_body_falls_back_and_stays_exact covers it.
 */
static int test_a_droplet_remesh_takes_no_global_reduction(void)
{
  MFTL_Ensemble *ens = make_droplets();

  /* Settle first, exactly as the timestep gate does: a generated ensemble has
   * every body on rank 0 and an empty shell everywhere else, which correctly
   * vetoes the holder route until the first redistribution has run. */
  for (int s = 0; s < 2; ++s)
  {
    MFTLUpdateGhostPositions(ens);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    MFTLRedistribute(ens);
  }

  /* Target 0.05 against a refine-1 droplet of radius 0.2, whose edges are about
   * 0.11 -- so the pass has genuine refinement to do and takes more than one
   * phase, which is what makes the per-phase scan the thing being counted.
   * 4 * delta = 0.2 is below the narrowest rank box at every registered process
   * count, so the coarsening fallback stays at {1,1,1} and the partition the
   * route test sees is the fine one. */
  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 0.05, 4, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = 2;
  params.parallel_phase_budget = 4;

  MFTLEnsembleResetBodyReduceStats();

  MFTL_RemeshStats st;
  const int rc = MFTLRemeshEnsembleParallel(ens, &params, &st);

  long split[2];
  route_split(split);

  if (g_rank == 0)
  {
    printf("\n    np=%d: one spanning remesh (%d phases, coarse group %d) -> %ld holder reductions, %ld global\n", g_nproc, st.n_phases,
           st.n_coarse_groups, split[0], split[1]);
  }

  TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* Non-vacuity in two directions. The pass must have run more than one phase,
   * or the per-phase scan this is about was taken once; and the fallback must
   * have stayed off, or the partition is coarse and the global route is the
   * right answer rather than the regression. */
  TEST_ASSERT(agreed(st.n_phases > 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_coarse_groups == 1, MPI_COMM_WORLD));

  /* At one rank every route is the identity and neither counter moves. */
  TEST_ASSERT(agreed(g_nproc == 1 ? (split[0] == 0) : (split[0] > 0), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(split[1] == 0, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The exclusive prefix sum
 *
 * A DIFFERENT PROPERTY FROM A REDUCTION, AND IT NEEDS A DIFFERENT ORACLE. The
 * reduction tests above ask "is the combination right"; a scan additionally has
 * to hand every holder a DISJOINT range, and the union of the ranges has to be
 * exactly [0, total). That is what the remesher uses it for -- two ranks that
 * both split an edge of one body must not produce the same global id -- and it
 * is a statement about the ranges taken TOGETHER, so it can only be checked by
 * gathering them.
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief One rank's contribution to a body's scan: a small, rank-dependent count.
 *
 * Deliberately not the triangle count. A payload that happens to be zero on some
 * holder cannot distinguish "my range starts after yours" from "we were handed
 * the same start", which is exactly the failure a scan has and a reduction does
 * not.
 *
 * @param ens  The ensemble.
 * @param rows [n_local_objects] receives the contributions.
 */
static void fill_scan_contribution(const MFTL_Ensemble *ens, double *rows)
{
  for (int o = 0; o < ens->n_local_objects; ++o) rows[o] = (double) (1 + (g_rank % 3) + (ens->local_objects[o]->global_id % 2));
}

/**
 * @brief Every holder's range is disjoint, and together they tile [0, total).
 *
 * Checked per body by gathering `(prefix, count, total)` from every rank onto
 * rank 0 and walking each body's holders in ascending rank order: the prefix of
 * the first holder must be 0, each next prefix must be the previous prefix plus
 * the previous count, and the last must close on `total`. Any dropped
 * contributor, double-counted self block or receive-order fold breaks one of
 * those three.
 *
 * The `total` is also checked against the plain reduction of the same values, so
 * the two entry points cross-check each other.
 */
static int test_the_scan_tiles_every_body_exactly_once(void)
{
  MFTL_Ensemble *ens = make_droplets();
  const int n = ens->n_local_objects;
  const int nb = ens->n_bodies_global;

  double *const mine = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const pre = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const tot = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));

  fill_scan_contribution(ens, mine);
  memcpy(pre, mine, (size_t) n * sizeof(double));
  TEST_ASSERT(agreed(MFTLEnsembleExscanBodies(ens, pre, tot, 1, MFTL_BODY_REDUCE_SCOPE_AUTO) == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* The same values through the reduction. The scan's `total` has no business
   * disagreeing with it, and if it does, one of the two folded a block twice. */
  {
    double *const red = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
    MFTL_BodyReduceOp sum = MFTL_BODY_REDUCE_SUM;
    memcpy(red, mine, (size_t) n * sizeof(double));
    MFTLEnsembleReduceBodies(ens, red, 1, &sum, MFTL_BODY_REDUCE_SCOPE_AUTO);

    int same = 1;
    for (int o = 0; o < n; ++o)
    {
      if (red[o] != tot[o]) same = 0;
    }
    TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
    free(red);
  }

  /* ── The tiling, gathered ─────────────────────────────────────────────── */
  double *const flat = (double *) calloc((size_t) nb * 3, sizeof(double));
  double *const all = (double *) calloc((size_t) nb * 3 * (size_t) g_nproc, sizeof(double));

  for (int g = 0; g < nb; ++g) flat[3 * g] = -1.0;  // "this rank is not a holder"

  for (int o = 0; o < n; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    flat[3 * g + 0] = pre[o];
    flat[3 * g + 1] = mine[o];
    flat[3 * g + 2] = tot[o];
  }

  MPI_Gather(flat, nb * 3, MPI_DOUBLE, all, nb * 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int ok = 1;
  int shared = 0;  // bodies with more than one holder -- the non-vacuity measure
  if (g_rank == 0)
  {
    for (int g = 0; g < nb; ++g)
    {
      double cursor = 0.0;
      double body_total = -1.0;
      int holders = 0;

      for (int r = 0; r < g_nproc; ++r)
      {
        const double *const rec = &all[(size_t) r * nb * 3 + 3 * g];
        if (rec[0] < 0.0) continue;  // not a holder
        holders++;

        if (rec[0] != cursor) ok = 0;  // ranges must abut, in ascending rank order
        cursor += rec[1];

        if (body_total < 0.0)
          body_total = rec[2];
        else if (rec[2] != body_total)
          ok = 0;  // every holder must agree about the total
      }

      if (holders == 0) continue;
      if (cursor != body_total) ok = 0;  // and together they must close on it
      if (holders > 1) shared++;
    }
    printf("\n    np=%d: %d of %d bodies have more than one holder\n", g_nproc, shared, nb);
  }

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY. With every body on exactly one rank the scan is the identity
   * and the tiling above is satisfied by a function that returns zero. */
  MPI_Bcast(&shared, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(g_nproc == 1 || shared > 0, MPI_COMM_WORLD));

  free(all);
  free(flat);
  free(tot);
  free(pre);
  free(mine);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The two routes give the scan the same answer, to the bit.
 *
 * Same argument as for the reduction: the holder route folds in ascending source
 * rank and the global route is an `MPI_Exscan`, so the two are independent
 * spellings of one total order and a discrepancy is an ordering bug.
 */
static int test_both_scan_routes_agree(void)
{
  MFTL_Ensemble *ens = make_droplets();
  const int n = ens->n_local_objects;

  double *const mine = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const a_pre = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const a_tot = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const b_pre = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  double *const b_tot = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));

  fill_scan_contribution(ens, mine);

  memcpy(a_pre, mine, (size_t) n * sizeof(double));
  MFTLEnsembleExscanBodies(ens, a_pre, a_tot, 1, MFTL_BODY_REDUCE_SCOPE_HOLDERS);

  memcpy(b_pre, mine, (size_t) n * sizeof(double));
  MFTLEnsembleExscanBodies(ens, b_pre, b_tot, 1, MFTL_BODY_REDUCE_SCOPE_GLOBAL);

  const int bytes = n * (int) sizeof(double);
  TEST_ASSERT(agreed(n == 0 || memcmp(a_pre, b_pre, (size_t) bytes) == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n == 0 || memcmp(a_tot, b_tot, (size_t) bytes) == 0, MPI_COMM_WORLD));

  free(b_tot);
  free(b_pre);
  free(a_tot);
  free(a_pre);
  free(mine);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Per-body reductions (np = %d) ===\n", g_nproc);

  RUN_TEST(test_both_routes_match_the_reference);
  RUN_TEST(test_every_holder_gets_the_same_bits);
  RUN_TEST(test_an_empty_holder_forces_the_global_route);
  RUN_TEST(test_a_spanning_body_falls_back_and_stays_exact);
  RUN_TEST(test_a_rank_with_no_bodies_still_participates);
  RUN_TEST(test_the_library_consumers_still_see_the_whole_body);
  RUN_TEST(test_a_droplet_timestep_takes_no_global_reduction);
  RUN_TEST(test_a_droplet_remesh_takes_no_global_reduction);
  RUN_TEST(test_the_scan_tiles_every_body_exactly_once);
  RUN_TEST(test_both_scan_routes_agree);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
