/*
 * @Author: Christian Gorges
 * @Date: 2026-08-17
 *
 * @brief Distributed remeshing: the shift/freeze phase schedule.
 *
 * Runs at np = 1, 2, 4 and 8. The one-rank run is not redundant -- it exercises
 * the phase driver itself (migration, id renumbering, the closing sequence) on a
 * mesh where the frozen set is empty, so a failure there is a driver bug and
 * cannot be a freeze bug. Every failure at np > 1 that does NOT also fail at
 * np = 1 is therefore about the partition.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT CAN BE COMPARED, AND WHAT CANNOT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A distributed contact query is tested against the serial answer as a SET,
 * because the answer is the same object either way (tests/test_bvh_ghosting.c).
 * REMESHING IS NOT LIKE THAT. It is a sequence of local decisions, each taken
 * against the mesh the previous one produced, so changing the order changes the
 * mesh -- legitimately. Two ranks split different edges first and end up with
 * different, equally valid triangulations. Demanding the serial answer would be
 * demanding that the parallelisation does nothing.
 *
 * So the tests below split into two kinds, and the distinction is the whole
 * design of this file:
 *
 *   EXACT       claims that are the same statement on one rank and on many, and
 *               are tested to the bit. The freeze equivalence, ghost agreement,
 *               id density, the structural invariants, Euler's formula. These
 *               are the bug finders -- every one of them is a property the
 *               scheme is BUILT on rather than a property it aims for.
 *
 *   CONVERGENT  the outcome the remesher exists to produce: every edge inside
 *               [min, max], the surface where it was. Compared against the same
 *               targets the serial remesher is held to, never against the serial
 *               mesh.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * NON-VACUITY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A distributed remeshing test whose mesh happens to sit inside one rank passes
 * forever while testing nothing -- the freeze machinery could be deleted and the
 * assertions would hold. So every test that claims something about the partition
 * first MEASURES the partition, through cross_rank_edges(): the number of
 * undirected edges whose two triangles are owned by different ranks. That is
 * exactly the set the frozen band exists to protect, and at np > 1 it is
 * asserted to be non-zero. If a future change to the fixture shrinks the sphere
 * or grows the domain, the tests fail loudly instead of quietly becoming serial.
 *
 * EVERY assertion goes through agreed(): TEST_ASSERT returns immediately, so an
 * assertion failing on some ranks only would leave the rest blocked in the next
 * collective -- a hang instead of a clean failure.
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

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief A [-1,1]^3 domain, so every grid below has h_min = 1.0.
 *
 * That number is load-bearing, and the margin against it is deliberately thin.
 * The shift schedule needs h_min > 4*delta with delta = 3 * max_edge_length
 * (MFTL_REMESH_BAND_EDGES), so the standard target of 0.06 gives
 * 4*delta = 0.96 against h_min = 1.0.
 *
 * Four percent of headroom is not an accident and not a mistake: it is the
 * regime where the scheme is under real pressure, so a widening of the frozen
 * band -- by a change to the classification, or to the halo the redistribution
 * builds -- shows up here as a convergence failure instead of being absorbed.
 * The tests still converge with `hidden == 0` at every rank count, which is the
 * evidence that the 3-edge estimate is conservative rather than tight. At
 * np = 27 (3x3x3, see test_grid_for()) h drops to 2/3, so that headroom becomes
 * the thing under test as well.
 */
static MFTL_Ensemble *make_ensemble(bool periodic)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 2);
}

/**
 * @brief Where each body was generated, kept so the volume integral has a frame.
 *
 * The enclosed volume (1/6) sum a . (b x c) is NOT translation invariant, so
 * every triangle of a body has to be expressed in ONE frame. For a body
 * straddling a periodic seam the stored coordinates are in two, and per-triangle
 * seam-consistent corners (MFTLGetTriangleCorners) fix each triangle
 * individually without agreeing between them -- which is exactly the case this
 * would silently get wrong. Unwrapping every corner relative to a point INSIDE
 * the body puts them all in one frame, and the generation centre is a point
 * every rank agrees on for free.
 */
static double g_body_centre[8][3];

static MFTL_Object *add_sphere(MFTL_Ensemble *ens, int id, double cx, double cy, double cz, double r, int refine)
{
  MFTL_Object *obj = MFTLObjectCreate(id);
  const int slot = ens->n_local_objects;
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {cx, cy, cz};
  if (slot < 8)
  {
    g_body_centre[slot][0] = cx;
    g_body_centre[slot][1] = cy;
    g_body_centre[slot][2] = cz;
  }

  MFTLGenerateSphere(obj, centre, r, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief The parameters every test uses, so a change is a change everywhere. */
static void standard_params(MFTL_RemeshParameters *p, double target)
{
  /* Feature detection OFF. A sphere has no features, so every edge the detector
   * flags is a false positive that locks a vertex the smoother needed -- see
   * docs/analysis.md §4, which measures final minimum quality at
   * 0.76 with detection off against 0.0002 at 45 degrees. Leaving it on here
   * would make the CONVERGENT tests below measure the detector rather than the
   * parallelisation. */
  MFTLInitRemeshParams(p, MFTL_REMESH_ISOTROPIC, target, 4, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p->smooth_iterations = 3;
  p->smooth_relaxation = 0.5;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Global measurements
 *
 * Every one of these counts each UNDIRECTED edge exactly once across all ranks,
 * which is the only counting rule under which Euler's formula means anything on
 * a partitioned mesh. The rule: an edge is counted by the rank owning the
 * incident triangle of SMALLER GLOBAL ID. Counting "my triangle whose twin is a
 * ghost" instead double-counts every cut edge, once on each side -- and it does
 * so by a number that grows with the rank count, which is exactly the shape of
 * error a partitioning test must not have.
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Does this rank own the right to count the undirected edge at @p h? */
static int counts_this_edge(const MFTL_Object *obj, int h)
{
  const int t = MFTLHeTriIdx(h);
  if (t >= obj->tris.n_local) return 0;  // a ghost's owner counts it

  const int tw = obj->tris.he[h].twin_idx;
  if (tw == MFTL_ERROR) return 1;  // mesh boundary: one triangle, and it is mine

  const int to = MFTLHeTriIdx(tw);
  if (to < obj->tris.n_local) return h < tw;  // both mine: the smaller half-edge

  return obj->tris.rec[t].global_id < obj->tris.rec[to].global_id;
}

/**
 * @struct mesh_measure
 * @brief Everything the convergent tests compare, in one reduction.
 */
typedef struct
{
  double n_vertices;
  double n_triangles;
  double n_edges;
  double n_edges_too_long;
  double n_edges_too_short;
  double n_cross_rank_edges;  ///< The non-vacuity measure. See the file header.
  double min_edge;
  double max_edge;
  double volume;  ///< Enclosed volume by the divergence theorem.
  double area;
} mesh_measure;

/**
 * @brief Measures the whole ensemble, collectively.
 *
 * @note Two reductions, not one per field: MPI_SUM for the counts and MPI_MIN /
 *       MPI_MAX for the extremes. A rank holding nothing must not contribute a
 *       zero minimum, so the extremes start at the identity of their operator.
 */
static void measure(const MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, mesh_measure *out)
{
  double sums[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double lo = MFTL_DBL_MAX;
  double hi = 0.0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    sums[0] += (double) obj->verts.n_local;
    sums[1] += (double) obj->tris.n_local;

    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!counts_this_edge(obj, h)) continue;

      const double len = MFTLGetEdgeLength(obj, h);
      sums[2] += 1.0;
      if (len > params->max_edge_length) sums[3] += 1.0;
      if (len < params->min_edge_length) sums[4] += 1.0;

      const int tw = obj->tris.he[h].twin_idx;
      if (tw != MFTL_ERROR && MFTLHeTriIdx(tw) >= obj->tris.n_local) sums[5] += 1.0;

      lo = MFTL_MIN(lo, len);
      hi = MFTL_MAX(hi, len);
    }

    /* Enclosed volume, (1/6) * sum a . (b x c) over OWNED triangles, with every
     * corner unwrapped relative to the body's own centre. See g_body_centre: the
     * displacement form is what makes this correct for a body straddling a
     * periodic seam, and it is identical to the raw form for one that does not
     * (a constant offset cancels in a closed surface's divergence sum). */
    const double *const ref = g_body_centre[(o < 8) ? o : 0];

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);

      double a[3], b[3], c[3];
      MFTLGetMinimumImageVector(obj->domain, a, ref, MFTLVertexPos(obj, v[0]));
      MFTLGetMinimumImageVector(obj->domain, b, ref, MFTLVertexPos(obj, v[1]));
      MFTLGetMinimumImageVector(obj->domain, c, ref, MFTLVertexPos(obj, v[2]));

      sums[6] += (a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
      sums[7] += obj->tris.area[t];
    }
  }

  double g_sums[8];
  double g_lo, g_hi;
  MPI_Allreduce(sums, g_sums, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&lo, &g_lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&hi, &g_hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  out->n_vertices = g_sums[0];
  out->n_triangles = g_sums[1];
  out->n_edges = g_sums[2];
  out->n_edges_too_long = g_sums[3];
  out->n_edges_too_short = g_sums[4];
  out->n_cross_rank_edges = g_sums[5];
  out->min_edge = g_lo;
  out->max_edge = g_hi;
  out->volume = g_sums[6];
  out->area = g_sums[7];
}

/** @brief Puts the ensemble on the grid and gives it current geometry. */
static int partition(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);

  if (MFTLRedistribute(ens) != MFTL_SUCCESS) return 0;

  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  return 1;
}

static int ensemble_is_sound(const MFTL_Ensemble *ens)
{
  if (!inv_local_directory_is_sound(ens)) return 0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    if (!inv_partition_is_sound(ens->local_objects[o], g_rank)) return 0;
  }
  return 1;
}

/**
 * @brief Is every entity where the CANONICAL cut planes say it should be?
 *
 * The postcondition of any parallel remeshing pass: whatever the driver did with
 * the planes in between, the mesh it hands back must be partitioned exactly as an
 * ordinary MFTLRedistribute would have partitioned it -- otherwise the flow
 * solver's next step works against a decomposition it does not know about.
 *
 * Checked against MFTLGetOwnerProc itself rather than against a recorded shift.
 * That is deliberate: the shift is a local of the driver, and a test that
 * inspected a stored value would pass for a driver that recorded zero and left
 * the mesh somewhere else.
 *
 * Requires current `tris.center`, which is part of the same postcondition.
 */
static int partition_matches_canonical_planes(const MFTL_Ensemble *ens)
{
  const MFTL_Domain *const dom = &ens->domain;
  if (dom->nproc < 2) return 1;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      if (MFTLGetOwnerProc(dom, MFTLVertexPos(obj, v)) != g_rank) return 0;
    }
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      if (MFTLGetOwnerProc(dom, &obj->tris.center[3 * t]) != g_rank) return 0;
    }

    /* And the converse: a ghost must belong to somebody else, or the ownership
     * prefix is claiming less than it owns and the halo is carrying entities
     * nobody will ever remesh. */
    for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
    {
      if (MFTLGetOwnerProc(dom, &obj->tris.center[3 * t]) == g_rank) return 0;
    }
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * EXACT: the freeze equivalence
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Checks MFTL_FLAG_FROZEN against the definition, in BOTH directions.
 *
 * The definition is "an owned entity is frozen iff some other rank holds a copy
 * of it", and MFTLRemeshMarkFrozen derives it locally from the ownership prefix
 * without asking anyone. This function does the opposite: it ASKS, by reducing a
 * per-global-id flag over all ranks, and then compares.
 *
 * Both directions matter and they fail differently:
 *
 *   frozen but not replicated   the band is too wide. Nothing corrupts; the
 *                               remesher just refuses work it was allowed to do,
 *                               so the mesh converges more slowly or not at all.
 *   replicated but not frozen   THE FAILURE. Two ranks may modify one entity, or
 *                               one rank may rewrite a triangle another is
 *                               reading. Every downstream symptom -- a torn
 *                               surface, a lost triangle, a duplicated vertex --
 *                               appears one or two migrations later, with
 *                               nothing left to point at the cause.
 *
 * ── Why this exists as well as the end-to-end test ─────────────────────────
 *
 * Because it is the only one that fails DETERMINISTICALLY. Deleting the
 * edge-neighbour clause from MFTLRemeshMarkFrozen -- so a triangle replicated
 * only because a neighbour owns its edge-partner comes out unfrozen -- was tried,
 * and the full remeshing pass still passed at np = 2 and np = 4. It does real
 * damage all the same: the rank holding that triangle as a ghost routes the next
 * migration by its now-stale CACHED CENTRE, and if the stale and fresh centres
 * fall either side of a cut plane, an owned triangle silently loses an edge
 * neighbour. That needs the rewrite to move a centroid across a plane, which is
 * rare per triangle and certain over a long run -- the worst combination for a
 * test that only checks outcomes.
 *
 * This one failed immediately, on both rank counts, and named the function.
 *
 * @note Collective.
 */
static int frozen_matches_replication(const MFTL_Object *obj)
{
  /* The LENGTHS are agreed first. `obj` may be test_body()'s shared empty on a
   * rank holding none of this body, and that one reports zero counters -- sizing
   * the reduction from it locally gives two ranks two different message lengths,
   * which MPI reports as a truncation rather than as the disagreement it is. */
  int range[2] = {obj->next_vertex_gid, obj->next_triangle_gid};
  MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const int nv = range[0];
  const int nt = range[1];
  const int n = MFTL_MAX(MFTL_MAX(nv, nt), 1);

  char *const held_v = (char *) calloc((size_t) n, 1);
  char *const held_t = (char *) calloc((size_t) n, 1);
  char *const any_v = (char *) calloc((size_t) n, 1);
  char *const any_t = (char *) calloc((size_t) n, 1);
  if (!held_v || !held_t || !any_v || !any_t)
  {
    free(held_v);
    free(held_t);
    free(any_v);
    free(any_t);
    return 0;
  }

  /* What this rank holds as a GHOST -- i.e. what it is a copy of. */
  for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g >= 0 && g < n) held_v[g] = 1;
  }
  for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g >= 0 && g < n) held_t[g] = 1;
  }

  /* A rank never holds a ghost copy of something it owns, so the OR over all
   * ranks is exactly "somebody else holds a copy". */
  MPI_Allreduce(held_v, any_v, n, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(held_t, any_t, n, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);

  int ok = 1;

  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    const int frozen = MFTLVertexIsFrozen(obj, v) ? 1 : 0;
    if (g < 0 || g >= n || frozen != (int) any_v[g]) ok = 0;
  }
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    const int frozen = MFTLTriangleIsFrozen(obj, t) ? 1 : 0;
    if (g < 0 || g >= n || frozen != (int) any_t[g]) ok = 0;
  }

  /* And every ghost is frozen, unconditionally -- the half of the definition
   * that needs no communication but is just as easy to break. */
  for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    if (!MFTLVertexIsFrozen(obj, v)) ok = 0;
  for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
    if (!MFTLTriangleIsFrozen(obj, t)) ok = 0;

  free(held_v);
  free(held_t);
  free(any_v);
  free(any_t);
  return ok;
}

static int test_frozen_is_exactly_the_replicated_set(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTLRemeshMarkFrozen(obj);

  MFTL_RemeshParameters params;
  standard_params(&params, 0.06);

  mesh_measure m;
  measure(ens, &params, &m);

  /* Non-vacuity: without cut edges the frozen set is empty and the equivalence
   * below is the trivial "nothing is frozen and nothing is replicated". */
  if (g_nproc > 1) TEST_ASSERT(m.n_cross_rank_edges > 0.0);

  TEST_ASSERT(agreed(frozen_matches_replication(obj), MPI_COMM_WORLD));

  /* Frozen must be a PROPER subset at np > 1: a partition in which everything is
   * frozen would satisfy the equivalence and remesh nothing. */
  int free_v = 0;
  for (int v = 0; v < obj->verts.n_local; ++v)
    if (!MFTLVertexIsFrozen(obj, v)) free_v++;

  int global_free = 0;
  MPI_Allreduce(&free_v, &global_free, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(global_free > 0);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Dilation may only ever GROW the frozen set, and by whole rings.
 *
 * A dilation that shrank it would be a correctness hole rather than a
 * conservatism knob, and one that propagated within a sweep would depend on pool
 * order -- so the second ring is also checked to be no smaller than the first.
 */
static int test_dilation_only_grows_the_band(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTLRemeshMarkFrozen(obj);

  char *const before = (char *) malloc((size_t) MFTL_MAX(obj->verts.n, 1));
  TEST_ASSERT(before != MFTL_NULL);
  for (int v = 0; v < obj->verts.n; ++v) before[v] = MFTLVertexIsFrozen(obj, v) ? 1 : 0;

  int n0 = 0, n1 = 0, n2 = 0;
  for (int v = 0; v < obj->verts.n; ++v) n0 += before[v];

  MFTLDilateFrozenBand(obj, 1);
  for (int v = 0; v < obj->verts.n; ++v)
  {
    n1 += MFTLVertexIsFrozen(obj, v) ? 1 : 0;
    if (before[v] && !MFTLVertexIsFrozen(obj, v)) n1 = -1000000;  // never unfreezes
  }

  MFTLDilateFrozenBand(obj, 1);
  for (int v = 0; v < obj->verts.n; ++v) n2 += MFTLVertexIsFrozen(obj, v) ? 1 : 0;

  free(before);

  TEST_ASSERT(agreed(n1 >= n0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n2 >= n1, MPI_COMM_WORLD));

  /* At np > 1 a ring must actually reach something, or the two-half-step
   * formulation is not propagating at all and the knob is a no-op. */
  int grew = (g_nproc == 1) ? 1 : (n1 > n0);
  int any_grew = 0;
  MPI_Allreduce(&grew, &any_grew, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(any_grew == 1);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief An append past the prefix is put back in its place by the compaction.
 *
 * The subtlest new code in the parallel path, and the only piece that is a data
 * MOVEMENT rather than a decision. During a sweep the pool is
 * [owned | ghost | created], because an append lands at the end and `n_local`
 * may not move past a ghost; MFTLRemeshCompactObject restores invariant 2 of
 * mftl_types.h by exchanging the last two blocks, and has to compose that
 * exchange into the index maps it is already applying.
 *
 * Driving it through a whole remeshing pass exercises it -- measured at 21, 48
 * and 119 rotations at np = 2, 4 and 8 -- but only reports failure as a broken
 * mesh several phases later. This drives the one operation directly and checks
 * the one postcondition, so a failure names the function.
 */
static int test_compaction_restores_the_ownership_prefix(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTLRemeshMarkFrozen(obj);

  const int n_ghost_v_before = obj->verts.n - obj->verts.n_local;
  const int n_ghost_t_before = obj->tris.n - obj->tris.n_local;

  /* Split a handful of interior edges: enough to make the created block
   * non-empty, few enough that the rest of the mesh is untouched. Both incident
   * triangles must be unfrozen, which is the same gate the sweeps apply. */
  int n_split = 0;
  for (int h = 0; h < 3 * obj->tris.n && n_split < 8; ++h)
  {
    const int t = MFTLHeTriIdx(h);
    const int tw = obj->tris.he[h].twin_idx;
    if (tw == MFTL_ERROR) continue;
    if (MFTLTriangleIsFrozen(obj, t) || MFTLTriangleIsFrozen(obj, MFTLHeTriIdx(tw))) continue;
    if (obj->tris.he[h].flags & MFTL_FLAG_VISITED) continue;

    if (MFTLSplitEdge(obj, h) != MFTL_ERROR) n_split++;
  }

  /* Non-vacuity twice over: the rotation only happens when BOTH trailing blocks
   * are non-empty, so a run with no ghosts or no splits proves nothing.
   *
   * SOMEWHERE, not everywhere. A rank can legitimately hold none of the body --
   * on a 3x3x3 grid the centre cell of a hollow sphere is empty -- and demanding
   * ghosts of every rank turns the most distributed configuration in the suite
   * into the one that fails. The claim below is still made on every rank that
   * has something to claim it about. */
  if (g_nproc > 1)
  {
    TEST_ASSERT(anywhere(n_ghost_v_before > 0, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(n_split > 0, MPI_COMM_WORLD));

    /* The window the compaction exists to close: entities past the prefix that
     * this rank nonetheless owns. Guarded, because a rank that split nothing has
     * no such window and would fail a claim about a state it never entered. */
    const int has_window = (n_split > 0 && n_ghost_v_before > 0);
    TEST_ASSERT(agreed(!has_window || obj->verts.rec[obj->verts.n - 1].owner_proc == g_rank, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(!has_window || obj->verts.n_local < obj->verts.n - 1, MPI_COMM_WORLD));
  }

  MFTLRemeshCompactObject(obj, MFTL_NULL);

  /* The prefix, stated as the invariant rather than as a count: owned below,
   * foreign above, with nothing straddling. */
  int ok = 1;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const int owned = (obj->verts.rec[v].owner_proc == g_rank);
    if (owned != (v < obj->verts.n_local)) ok = 0;
  }
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const int owned = (obj->tris.rec[t].owner_proc == g_rank);
    if (owned != (t < obj->tris.n_local)) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* The rotation moved the ghosts; it must not have lost or duplicated any. A
   * ghost cannot die -- it is frozen on its owner -- so the counts are exact,
   * and the owned counts must have grown by exactly what the splits created. */
  TEST_ASSERT(agreed(obj->verts.n - obj->verts.n_local == n_ghost_v_before, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(obj->tris.n - obj->tris.n_local == n_ghost_t_before, MPI_COMM_WORLD));

  /* And the topology survived being permuted by something other than a forward
   * scan: twins symmetric, out_he outgoing, derived arrays agreeing. */
  TEST_ASSERT(agreed(inv_mesh_is_sound(obj, g_rank), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(inv_local_global_ids_are_unique(obj), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * EXACT: what the mesh must be afterwards
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Every ghost this rank holds is bit-identical to its owner's copy.
 *
 * Three separate claims in one check, and the reason they can share one is that
 * MFTLUpdateGhostPositions fails differently for each:
 *
 *   1. Every ghost's global id names an entity its owner actually owns. A
 *      request the owner cannot serve makes the call return MFTL_ERROR, so this
 *      is what catches an id that the end-of-pass re-densification failed to
 *      propagate.
 *   2. The reply is a copy of a position this rank already had. That is the
 *      theorem the phase loop rests on -- every ghost is frozen ON ITS OWNER, so
 *      nobody was allowed to move it -- and it is what justifies the ABSENCE of
 *      a ghost position sync inside the phase loop. A difference here means some
 *      rank modified an entity another rank holds, which is the failure the
 *      whole freeze scheme exists to make impossible.
 *   3. Exact equality, not a tolerance: the value travelled as a double and
 *      neither side computed with it. Anything else is a real difference.
 */
static int ghosts_agree_with_owners(MFTL_Ensemble *ens)
{
  int n_ghost = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) n_ghost += ens->local_objects[o]->verts.n - ens->local_objects[o]->verts.n_local;

  double *const snapshot = (double *) malloc((size_t) MFTL_MAX(3 * n_ghost, 1) * sizeof(double));
  if (!snapshot) return 0;

  int k = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      snapshot[3 * k + 0] = obj->verts.pos[3 * v + 0];
      snapshot[3 * k + 1] = obj->verts.pos[3 * v + 1];
      snapshot[3 * k + 2] = obj->verts.pos[3 * v + 2];
      k++;
    }
  }

  int ok = (MFTLUpdateGhostPositions(ens) == MFTL_SUCCESS);

  k = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      for (int c = 0; c < 3; ++c)
      {
        if (snapshot[3 * k + c] != obj->verts.pos[3 * v + c]) ok = 0;
      }
      k++;
    }
  }

  free(snapshot);

  return ok;
}

/**
 * @brief Global ids are a dense permutation of [0, N) mesh-wide, owned once each.
 *
 * The serial inv_global_ids_are_dense cannot be used here -- it compares against
 * `verts.n`, which is one rank's share plus its halo. Density is a statement
 * about the WHOLE mesh, so it needs a reduction, and it is worth the reduction:
 * redistribution and the ghost sync size their arrival tables from
 * `next_*_gid`, so a mesh whose ids drift sparse grows an unbounded table over a
 * few hundred timesteps while behaving perfectly.
 *
 * @note Collective.
 */
static int global_ids_are_dense(const MFTL_Object *obj)
{
  /* Lengths agreed first, for the same reason as in frozen_matches_replication:
   * test_body()'s shared empty reports zero counters, and a locally-sized
   * reduction then truncates instead of reporting a disagreement. */
  int range[2] = {obj->next_vertex_gid, obj->next_triangle_gid};
  MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const int nv = range[0];
  const int nt = range[1];
  const int n = MFTL_MAX(MFTL_MAX(nv, nt), 1);

  int *const mine = (int *) calloc((size_t) n, sizeof(int));
  int *const total = (int *) malloc((size_t) n * sizeof(int));
  if (!mine || !total)
  {
    free(mine);
    free(total);
    return 0;
  }

  int ok = 1;

  if (obj->tris.n > 0 && obj->next_halfedge_gid != 3 * nt) ok = 0;

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g < 0 || g >= nt)
      ok = 0;
    else
      mine[g]++;

    /* A half-edge id is 3 * (triangle id) + k, and the re-densification has to
     * carry that forward -- redistribution never transmits half-edge ids and
     * reconstructs them from this rule at the receiver. */
    for (int c = 0; c < 3; ++c)
    {
      if (obj->tris.he[3 * t + c].global_id != 3 * g + c) ok = 0;
    }
  }
  MPI_Allreduce(mine, total, nt, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int g = 0; g < nt; ++g)
    if (total[g] != 1) ok = 0;

  memset(mine, 0, (size_t) n * sizeof(int));
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g < 0 || g >= nv)
      ok = 0;
    else
      mine[g]++;
  }
  MPI_Allreduce(mine, total, nv, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int g = 0; g < nv; ++g)
    if (total[g] != 1) ok = 0;

  free(mine);
  free(total);
  return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The full pass
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The main event: remesh a partitioned sphere and check everything.
 *
 * Deliberately one test rather than six, because all six claims are about ONE
 * mesh produced by ONE expensive call, and running the call six times would
 * multiply the suite's runtime by six to test nothing extra. The assertions are
 * ordered so that the most diagnostic failure comes first: structure before
 * identity before convergence, because a torn mesh explains a wrong id count
 * while a wrong id count explains nothing about a torn mesh.
 */
static int remesh_and_check(bool periodic, double target, double cx)
{
  MFTL_Ensemble *ens = make_ensemble(periodic);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, cx, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTL_RemeshParameters params;
  standard_params(&params, target);

  mesh_measure before;
  measure(ens, &params, &before);

  /* NON-VACUITY, asserted before the work rather than after it: at np > 1 the
   * sphere must genuinely straddle rank boundaries, or everything below is a
   * serial test wearing an mpirun. */
  if (g_nproc > 1) TEST_ASSERT(before.n_cross_rank_edges > 0.0);

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* ── Structure ─────────────────────────────────────────────────────────── */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));

  /* The canonical decomposition must be back, or the flow solver's next step
   * partitions against planes it does not know about.
   *
   * Asserted against MFTLGetOwnerProc -- the PRODUCTION rule -- rather than
   * against a recorded shift value, which is both stronger and the only form
   * available now that the shift is a local of the driver rather than a field
   * anyone can inspect. It says the thing that actually matters: every entity is
   * exactly where an ordinary MFTLRedistribute would have put it. */
  TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));

  /* MFTL_FLAG_FROZEN is transient and must not outlive the call. */
  int leaked = 0;
  for (int v = 0; v < obj->verts.n; ++v) leaked += MFTLVertexIsFrozen(obj, v) ? 1 : 0;
  for (int t = 0; t < obj->tris.n; ++t) leaked += MFTLTriangleIsFrozen(obj, t) ? 1 : 0;
  TEST_ASSERT(agreed(leaked == 0, MPI_COMM_WORLD));

  /* ── Identity ──────────────────────────────────────────────────────────── */
  TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));

  mesh_measure after;
  measure(ens, &params, &after);

  /* Euler's formula for a closed genus-0 surface. It is the one check that sees
   * a triangle lost between two ranks, or one counted twice: both leave a
   * perfectly valid local mesh on every rank and only the GLOBAL count is
   * wrong. */
  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));

  /* Reduced: `global_n_*` describes the whole body, but only a rank holding some
   * of it carries the value -- test_body()'s stand-in reports zero. At 27
   * ranks a single sphere legitimately misses several of them. */
  {
    int g[2] = {obj->global_n_vertices, obj->global_n_triangles};
    MPI_Allreduce(MPI_IN_PLACE, g, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    TEST_ASSERT(fabs(after.n_vertices - (double) g[0]) < 0.5);
    TEST_ASSERT(fabs(after.n_triangles - (double) g[1]) < 0.5);
  }

  /* ── Convergence ───────────────────────────────────────────────────────── */

  /* The certificate. False means the driver gave up with the frozen bands still
   * hiding work -- either the budget ran out or h <= 4*delta. Either way the mesh
   * carries a lattice of un-remeshed bands that no structural check can see,
   * because each of them is a perfectly valid piece of mesh. */
  TEST_ASSERT(agreed(stats.converged, MPI_COMM_WORLD));

  /* Out-of-band edges, as a fraction. Not zero: a remesher converges to a band,
   * and Botsch & Kobbelt's bounds are the widest pair for which split and
   * collapse do not oscillate -- a handful of edges sit outside them on any
   * mesh, serial or not. What must NOT happen is a fraction that grows with the
   * rank count, which is the signature of a permanently frozen band. */
  const double bad = after.n_edges_too_long + after.n_edges_too_short;
  TEST_ASSERT(agreed(bad < 0.02 * after.n_edges, MPI_COMM_WORLD));

  /* It must actually have converged TOWARDS the target, not merely stopped. */
  TEST_ASSERT(agreed(bad < 0.25 * (before.n_edges_too_long + before.n_edges_too_short), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.n_triangles > before.n_triangles, MPI_COMM_WORLD));

  /* ── The surface is still where it was ─────────────────────────────────── */
  TEST_ASSERT(agreed(after.volume > 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(after.volume - before.volume) < 0.05 * before.volume, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d %s cx=%.1f: %.0f -> %.0f tris, %d phases, %d migrations, out-of-band %.0f/%.0f, "
           "hidden %d, cross-rank edges %.0f, dV %.2f%%\n    ",
           g_nproc, periodic ? "periodic" : "walls", cx, before.n_triangles, after.n_triangles, stats.n_phases, stats.n_migrations, bad, after.n_edges,
           stats.n_edges_hidden, before.n_cross_rank_edges, 100.0 * (after.volume - before.volume) / before.volume);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief h <= 4*delta: the one regime in which the driver must NOT certify.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE GAP THIS FILLS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Every other test here asserts that `stats.converged` is TRUE. Until this one
 * there was no test anywhere in the suite that asserted it is ever false, so the
 * documented failure branch -- `check_band_fits`, warned about on rank 0 and
 * described in four places in mftl_remeshing.h -- was never executed by CI at
 * all. That is a dangerous shape of coverage for a convergence criterion,
 * because every plausible way of making convergence CHEAPER also makes it easier
 * to declare, and nothing would have failed.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE TARGET IS 0.40 BECAUSE THAT IS WHERE IT MEASURABLY BREAKS, NOT WHERE THE
 * ESTIMATE SAYS IT SHOULD
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The obvious configuration is "make 4*delta exceed h_min", and it does not
 * work, because `delta` as the library computes it -- MFTL_REMESH_BAND_EDGES
 * (3) times max_edge_length -- is roughly three times the band that actually
 * occurs. Measured on this fixture, sweeping the target and reading the real
 * band back off the frozen set (max over owned frozen vertices of the distance
 * to the nearest cut plane):
 *
 *   target  max_edge  4*delta_est   observed band   np=2 conv/hidden   np=4
 *    0.06     0.080      0.96           0.071          1 / 0           1 / 0
 *    0.12     0.160      1.92           0.134          1 / 0           1 / 0
 *    0.25     0.333      4.00           0.266          1 / 0           1 / 0
 *    0.32     0.427      5.12           0.361          1 / 0           1 / 9
 *    0.40     0.533      6.40           0.450          0 / 10          0 / 20
 *
 * The observed band is 0.84 * max_edge_length across the whole sweep -- about
 * ONE edge, not three -- so at a target of 0.12 the estimate says the scheme
 * cannot work and it works perfectly. Only around 0.40, where the observed band
 * finally exceeds h/4 = 0.25, does the driver genuinely fail to offer everything
 * to somebody. That is the configuration used here, and the gap between the two
 * columns is the reason `band_measured` exists (see MFTL_RemeshStats).
 *
 * WHAT IS ASSERTED, AND WHY NOT MORE. The mesh must still be perfect -- the
 * degradation is a QUALITY shortfall, not a corruption, which is exactly why the
 * driver warns instead of aborting. And the pass must not report the STRONG
 * certificate: `converged && n_edges_hidden == 0` is the claim "every out-of-band
 * edge was offered to some rank that was allowed to act on it", and here it is
 * false. `converged` ALONE is deliberately not asserted, and the reason is worth
 * recording: the silent-placement streak can and does declare success with work
 * still hidden -- at np = 4 with a target of 0.50 this fixture reports
 * `converged == true` alongside 42 hidden edges. So the flag is not the claim;
 * the conjunction is.
 */
static int test_a_permanently_frozen_band_is_not_certified(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 2);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTL_RemeshParameters params;
  standard_params(&params, 0.40);

  /* DISARMED, and this is the whole reason the parameter still accepts 0.
   *
   * This test pins the DEGRADATION -- what the shift schedule does on a
   * partition too fine for its band. The fallback exists to repair exactly that,
   * and it is armed by default, so leaving it armed here would make the test
   * assert that the repair fails. It is also the control for
   * test_a_coarsened_pass_rescues_a_partition_that_is_too_fine below, and a
   * control that has the treatment applied to it is not a control. */
  params.parallel_coarsen = 0;

  /* The library's own precondition test, restated rather than trusted. It is the
   * ESTIMATE, and the table above says the estimate fires long before the scheme
   * actually degrades -- so this is a necessary condition for the test to mean
   * anything, not a sufficient one. The sufficient one is `n_edges_hidden > 0`
   * below. */
  const double delta = MFTL_REMESH_BAND_EDGES * params.max_edge_length;
  const double h_min = MFTLDomainGetHMin(&ens->domain);
  if (g_nproc > 1) TEST_ASSERT(agreed(h_min <= 4.0 * delta, MPI_COMM_WORLD));

  mesh_measure before;
  measure(ens, &params, &before);
  if (g_nproc > 1) TEST_ASSERT(before.n_cross_rank_edges > 0.0);

  /* There must be work to hide, or the bands have nothing to block and the test
   * is vacuous in the other direction. */
  TEST_ASSERT(agreed(before.n_edges_too_long + before.n_edges_too_short > 0.0, MPI_COMM_WORLD));

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* The mesh is still perfect. This is the half of the contract that must hold
   * whatever the bands did. */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));

  mesh_measure after;
  measure(ens, &params, &after);

  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.volume > 0.0, MPI_COMM_WORLD));

  /* THE CLAIM, in two halves. First: the bands really are hiding work, which is
   * this test's non-vacuity measure -- without it a future change that widens the
   * rank boxes would leave the test passing while testing nothing. Measured 10,
   * 20 and 76 hidden edges at np = 2, 4 and 8, so the margin is not thin.
   *
   * Second: the driver must not come back with the strong certificate. */
  if (g_nproc > 1)
  {
    TEST_ASSERT(agreed(stats.n_edges_hidden > 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(!(stats.converged && stats.n_edges_hidden == 0), MPI_COMM_WORLD));
  }

  /* At one rank nothing is frozen, so the same parameters must converge cleanly.
   * That contrast is what says the failure above is about the PARTITION. */
  if (g_nproc == 1) TEST_ASSERT(stats.converged && stats.n_edges_hidden == 0);

  if (g_rank == 0)
  {
    printf("\n    np=%d h_min=%.3f 4*delta=%.3f: %.0f -> %.0f tris, %d phases, converged=%d, hidden=%d\n    ", g_nproc, h_min, 4.0 * delta,
           before.n_triangles, after.n_triangles, stats.n_phases, stats.converged ? 1 : 0, stats.n_edges_hidden);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_partitioned_sphere_remeshes_on_a_periodic_domain(void) { return remesh_and_check(MFTL_TRUE, 0.06, 0.0); }

/**
 * @brief The same sphere, sitting ON the periodic seam.
 *
 * THE test for the periodic floor in MFTLRemeshOwnerProc, and the only one that
 * can see it. The shifted argument reaches -0.5 for a wrapped position, where a
 * plain `(int)` cast truncates TOWARDS ZERO instead of flooring -- so the sliver
 * [global_min, global_min + shift) is attributed to cell 0 rather than to cell
 * cuts-1 across the seam.
 *
 * What makes that hard to catch is that the wrong rule is still a rule: every
 * rank computes the same answer, so the partition stays consistent and every
 * structural invariant still holds. The one thing that changes is that THE SEAM
 * PLANE NEVER MOVES -- cell 0 simply grows to 1.5h and cell cuts-1 shrinks to
 * 0.5h -- and an entity on the seam is therefore frozen in EVERY phase of the
 * schedule. The result is a permanently un-remeshed band right at the seam,
 * which is a perfectly valid piece of mesh that no structural check can see.
 *
 * A body away from the seam never touches that sliver, which is why the two
 * tests above pass under the broken rule. This one puts a quarter of the sphere
 * in it. Introduced after a mutation test showed the coverage gap.
 */
static int test_a_sphere_on_the_periodic_seam_remeshes_everywhere(void) { return remesh_and_check(MFTL_TRUE, 0.06, 1.0); }

/**
 * @brief The same, with walls.
 *
 * Not a duplicate: MFTLGetOwnerProc takes a different branch per axis, and the
 * shift makes the two differ. A periodic axis WRAPS a shifted-out coordinate to
 * the far cell; a non-periodic one CLAMPS it into the boundary cell, which makes
 * the first and last boxes uneven by h/2. The frozen-band argument survives that
 * (only the interior planes matter), but the arithmetic is a separate path and a
 * sign error in it would show up only here.
 */
static int test_a_partitioned_sphere_remeshes_with_walls(void) { return remesh_and_check(MFTL_FALSE, 0.06, 0.0); }

/**
 * @brief Two bodies, because bodies are identified on the wire by LIST INDEX.
 *
 * The id renumbering is per object and the reductions carry one pair of counts
 * per object, so an off-by-one in the packing of those arrays merges two bodies'
 * id spaces -- which a single-body test cannot see at all.
 */
static int test_two_bodies_keep_separate_id_spaces(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  add_sphere(ens, 0, -0.45, 0.0, 0.0, 0.35, 2);
  add_sphere(ens, 1, 0.45, 0.0, 0.0, 0.30, 2);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  /* Re-resolved after the partition, and by BODY ID: a rank holds an object only
   * for the bodies it has entities of, and these two are small enough that at
   * eight ranks most ranks hold one of them or neither. */
  MFTL_Object *a = test_body(ens, 0);
  MFTL_Object *b = test_body(ens, 1);

  /* Reduced, because `global_n_triangles` is a property of the whole body and a
   * rank holding none of it reads test_body()'s empty stand-in, which reports
   * zero. Same for every comparison against these below. */
  int before[2] = {a->global_n_triangles, b->global_n_triangles};
  MPI_Allreduce(MPI_IN_PLACE, before, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const int a_before = before[0];
  const int b_before = before[1];

  MFTL_RemeshParameters params;
  /* Well below both spheres' generated edge length (~0.10 at refine 2), so both
   * bodies genuinely have work to do. A target INSIDE the band would leave the
   * meshes untouched and the id-space assertions below would hold vacuously. */
  standard_params(&params, 0.06);

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = test_body(ens, 0);  /* the pass ends in a redistribution; the pointers do not survive it */
  b = test_body(ens, 1);

  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(a), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(b), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));

  int after[2] = {a->global_n_triangles, b->global_n_triangles};
  MPI_Allreduce(MPI_IN_PLACE, after, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  /* Both bodies must have been remeshed -- an id-space bug that left one of them
   * untouched, or merged the two, would pass every check above. */
  TEST_ASSERT(after[0] > a_before);
  TEST_ASSERT(after[1] > b_before);

  /* The two bodies stay two bodies. Their id spaces are independent, so the
   * larger sphere must end up with more triangles -- a merge would give both the
   * same count. */
  TEST_ASSERT(after[0] > after[1]);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The dilation knob changes how MUCH is done per phase, never the result.
 *
 * If the exact classification were wrong and the dilation were papering over it,
 * these two runs would converge to measurably different meshes. They must not:
 * both must reach the band, and the surface must land in the same place.
 */
static int test_a_dilated_band_reaches_the_same_mesh(void)
{
  double vol[2];
  double bad[2];

  for (int pass = 0; pass < 2; ++pass)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    MFTL_RemeshParameters params;
    /* Finer than the other tests on purpose. A dilation ring WIDENS delta, so
     * the h > 4*delta precondition has to hold for the wider band too: at
     * max_edge_length = 0.06 the one-ring band is 4*0.06 = 0.24 against
     * h_min/4 = 0.25. A coarser target would trip the driver's warning and make
     * this test measure the degraded regime rather than the knob. */
    standard_params(&params, 0.045);
    params.freeze_dilation_rings = pass;  /* 0 = the exact set, 1 = one ring wider */

    MFTL_RemeshStats stats;
    TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

    mesh_measure m;
    measure(ens, &params, &m);
    vol[pass] = m.volume;
    bad[pass] = (m.n_edges_too_long + m.n_edges_too_short) / m.n_edges;

    MFTLEnsembleDestroy(&ens);
  }

  TEST_ASSERT(agreed(bad[0] < 0.02, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(bad[1] < 0.02, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(vol[0] - vol[1]) < 0.02 * vol[0], MPI_COMM_WORLD));
  return 0;
}

/* -----------------------------------------------------------------------------
 * EXACT-ish: the edge-length DISTRIBUTION against the serial answer
 * -------------------------------------------------------------------------- */

#define HIST_BINS 16
#define HIST_LO 0.5
#define HIST_HI 1.7

/** @brief Fraction of undirected edges per length bin, normalised by the target. */
static void edge_histogram(const MFTL_Ensemble *ens, double target, MPI_Comm comm, double out[HIST_BINS])
{
  double local[HIST_BINS];
  for (int b = 0; b < HIST_BINS; ++b) local[b] = 0.0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!counts_this_edge(obj, h)) continue;

      const double r = MFTLGetEdgeLength(obj, h) / target;
      int b = (int) ((r - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS);
      if (b < 0) b = 0;
      if (b >= HIST_BINS) b = HIST_BINS - 1;
      local[b] += 1.0;
    }
  }

  double g[HIST_BINS];
  MPI_Allreduce(local, g, HIST_BINS, MPI_DOUBLE, MPI_SUM, comm);

  double total = 0.0;
  for (int b = 0; b < HIST_BINS; ++b) total += g[b];
  if (total <= 0.0) total = 1.0;
  for (int b = 0; b < HIST_BINS; ++b) out[b] = g[b] / total;
}

/**
 * @brief The distributed mesh must have the SAME edge-length distribution as the
 *        serial one (docs/analysis.md §5.2).
 *
 * -- Why a histogram and not the mesh --------------------------------------
 *
 * Remeshing is a sequence of local decisions taken against the mesh the previous
 * one produced, so two orderings give two different, equally valid
 * triangulations and no element-by-element comparison is meaningful. The
 * DISTRIBUTION is meaningful, because it is what the remesher is actually for: a
 * distributed pass that quietly refuses work along the partition boundaries
 * produces a mesh that is structurally perfect, has a plausible triangle count,
 * satisfies Euler's formula -- and carries a second population of edges at the
 * ORIGINAL length. That shows up here as mass in the wrong bins and in nothing
 * else this suite measures.
 *
 * -- Where the reference comes from ----------------------------------------
 *
 * Every rank builds the same sphere on MPI_COMM_SELF and remeshes it with the
 * plain serial entry point. That is a genuine independent reference: it shares
 * no partition, no migration, no freeze classification and no id renumbering
 * with the run under test, and it is deterministic, so all ranks compute the
 * same one and the comparison cannot disagree between them.
 *
 * -- The tolerance ----------------------------------------------------------
 *
 * Total-variation distance, i.e. half the L1 difference of the two normalised
 * histograms: the fraction of edges that would have to move bins to make the
 * distributions equal.
 *
 * MEASURED: 0.0000 at np = 1, then 0.028, 0.027, 0.044 at np = 2, 4, 8. The
 * np = 1 figure is the interesting one -- the phase driver reproduces the
 * converged serial distribution to the bin, which says the frozen gate and the
 * migrations are exactly transparent when there is nothing to freeze. The rest
 * grows slowly with the rank count, which is the residual of genuinely different
 * triangulations and not of refused work: a single un-remeshed band on this
 * fixture is over 10 % of the surface, so the 0.08 bound separates the two by a
 * factor of two either way.
 */
static int test_the_edge_length_distribution_matches_the_serial_answer(void)
{
  const double target = 0.06;

  MFTL_RemeshParameters params;
  standard_params(&params, target);

  /* -- The reference, on this rank alone ----------------------------------- */
  double ref[HIST_BINS];
  {
    int one[3] = {1, 1, 1};
    double lo[3] = {-1.0, -1.0, -1.0};
    double hi[3] = {1.0, 1.0, 1.0};
    bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};

    MFTL_Ensemble *solo = MFTLEnsembleCreate(MPI_COMM_SELF, one, lo, hi, per, 1);
    TEST_ASSERT(solo != MFTL_NULL);

    MFTL_Object *sobj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(solo, sobj);
    double centre[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(sobj, centre, 0.7, 3);
    MFTLComputeTriangleGeometry(sobj, sobj->tris.n);

    /* Remeshed to CONVERGENCE, not once. The comparison is only meaningful
     * between two converged states, and the two drivers do not do equal amounts
     * of work per call: MFTLRemeshEnsembleParallel runs the full iteration count
     * PER PHASE, so a single MFTLRemeshObject call is several times less
     * iteration and comes out with a visibly looser distribution (measured:
     * total variation 0.115 against 0.011 once both are converged). That is the
     * phase driver being thorough rather than either of them being wrong, and
     * comparing against the un-converged reference would bake it in. */
    for (int pass = 0; pass < 12; ++pass)
    {
      MFTL_RemeshStats one;
      TEST_ASSERT(MFTLRemeshObject(sobj, &params, &one) == MFTL_SUCCESS);
      if (one.n_split + one.n_collapse + one.n_flip == 0) break;
    }
    edge_histogram(solo, target, MPI_COMM_SELF, ref);

    MFTLEnsembleDestroy(&solo);
  }

  /* -- The run under test --------------------------------------------------- */
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);
  add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  mesh_measure before;
  measure(ens, &params, &before);
  if (g_nproc > 1) TEST_ASSERT(agreed(before.n_cross_rank_edges > 0.0, MPI_COMM_WORLD));

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

  double par[HIST_BINS];
  edge_histogram(ens, target, MPI_COMM_WORLD, par);

  double tv = 0.0;
  for (int b = 0; b < HIST_BINS; ++b) tv += fabs(par[b] - ref[b]);
  tv *= 0.5;

  if (g_rank == 0)
  {
    printf("\n    np=%d edge-length histogram vs converged serial: total variation %.4f (par/ref per mille:", g_nproc, tv);
    for (int b = 0; b < HIST_BINS; ++b)
    {
      if (par[b] > 0.0005 || ref[b] > 0.0005) printf(" %.0f/%.0f", 1000.0 * par[b], 1000.0 * ref[b]);
    }
    printf(")\n    ");
  }

  TEST_ASSERT(agreed(tv < 0.08, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* -----------------------------------------------------------------------------
 * The length band: one number every rank must agree on
 * -------------------------------------------------------------------------- */

/**
 * @brief The mean edge length must be measured COLLECTIVELY, and the local form
 *        must not be silently usable in its place.
 *
 * This is a trap with a body count, found by writing
 * examples/leveque_remeshing_parallel.c and not by reasoning about the code.
 * The natural way to pick a target is "whatever the mesh currently is":
 *
 *     MFTLGenerateSphere(obj, centre, r, n);
 *     double target = MFTLMeanEdgeLength(obj);        // <-- LOCAL
 *
 * and it reads as if it measures the sphere. MFTLGenerateSphere is COLLECTIVE
 * and already distributes, so at np > 1 it measures ONE RANK'S SHARE -- and a
 * rank owning none of the body measures the empty set and gets 0.0. That rank
 * then holds a length band of [0, 0], under which every edge it is ever sent is
 * too long. Nothing in the algorithm communicates the band, so nothing notices.
 *
 * Measured before the fix, on a 1280-triangle sphere at np = 2: the rank that
 * started empty received a 105-triangle cap in the first timestep and turned it
 * into 148 287 triangles over eight phases, still climbing when the run was
 * killed. No assertion fired, no invariant broke -- every one of those triangles
 * was a perfectly valid triangle.
 *
 * Three things are asserted here, and each closes one step of that path.
 */
static int test_the_edge_length_band_is_a_collective_measurement(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  /* Deliberately OFF-CENTRE and small, so that at np > 1 at least one rank owns
   * nothing of it -- which is the configuration the trap needs and which the
   * rest of this file's fixtures never produce. */
  MFTL_Object *obj = add_sphere(ens, 0, -0.55, -0.55, -0.55, 0.2, 2);

  /* The reference: the same sphere built on ONE rank, where "the mean edge of
   * the whole mesh" is unambiguous. Note that it cannot be obtained by calling
   * MFTLMeanEdgeLength on `obj` above -- MFTLGenerateSphere has already
   * distributed it, which is the first half of the trap. */
  double whole_mesh_mean = 0.0;
  {
    int one[3] = {1, 1, 1};
    double lo[3] = {-1.0, -1.0, -1.0};
    double hi[3] = {1.0, 1.0, 1.0};
    bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};

    MFTL_Ensemble *solo = MFTLEnsembleCreate(MPI_COMM_SELF, one, lo, hi, per, 1);
    TEST_ASSERT(solo != MFTL_NULL);

    MFTL_Object *sobj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(solo, sobj);
    double c[3] = {-0.55, -0.55, -0.55};
    MFTLGenerateSphere(sobj, c, 0.2, 2);
    whole_mesh_mean = MFTLMeanEdgeLength(sobj);
    MFTLEnsembleDestroy(&solo);
  }

  const double before_scatter = MFTLEnsembleMeanEdgeLength(ens);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* 1. The collective form is PARTITION-INDEPENDENT and equals the whole-mesh
   *    answer. Both halves matter: equal to the reference says it measures the
   *    right thing, unchanged across a redistribution says it keeps measuring
   *    the right thing once the mesh moves. */
  const double collective = MFTLEnsembleMeanEdgeLength(ens);
  TEST_ASSERT(agreed(fabs(collective - whole_mesh_mean) < 1e-9 * whole_mesh_mean, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(collective - before_scatter) < 1e-9 * whole_mesh_mean, MPI_COMM_WORLD));

  /* 2. The local form does NOT, and this test is worthless without checking it:
   *    if MFTLMeanEdgeLength happened to agree across ranks there would be no
   *    trap to guard and no reason for the collective form to exist. At np = 1
   *    the two are the same function of the same mesh, so the claim is only made
   *    where it means something. */
  if (g_nproc > 1)
  {
    const double local = MFTLMeanEdgeLength(obj);
    int differs = (fabs(local - collective) > 1e-9 * collective) ? 1 : 0;
    int any = 0;
    MPI_Allreduce(&differs, &any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(any == 1, MPI_COMM_WORLD));
  }

  /* 3. And a degenerate band is REFUSED rather than acted on. This is the step
   *    that converts the failure from "the job dies of memory exhaustion in a
   *    few minutes, somewhere" into a return value at the call site. */
  MFTL_RemeshParameters bad;
  standard_params(&bad, 0.06);
  bad.min_edge_length = 0.0;
  bad.max_edge_length = 0.0;
  bad.min_edge_length_sq = 0.0;
  bad.max_edge_length_sq = 0.0;

  MFTL_RemeshStats st;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &bad, &st) == MFTL_ERROR, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */
  TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &bad, &st) == MFTL_ERROR, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* Refused means UNTOUCHED, not half-done. A driver that rejected the band
   * after its first migration would leave the mesh somewhere else entirely. */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A repeated pass is idempotent enough to be run every timestep.
 *
 * The realistic usage is a remesh per N timesteps forever, so the interesting
 * question is not "does one pass work" but "does the hundredth". Two failure
 * modes only appear on the second pass: ids that did not come back dense (the
 * arrival tables then grow every call), and a frozen flag or a decomposition
 * shift left behind by the first.
 */
static int test_a_second_pass_is_stable(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTL_RemeshParameters params;
  standard_params(&params, 0.06);

  MFTL_RemeshStats first, second;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &first) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* Reduced, here and below: `global_n_*` and the id counters describe the whole
   * body, and a rank holding none of it reads test_body()'s stand-in, which
   * reports zero. At 27 ranks a single sphere misses several of them. */
  int st1[2] = {obj->global_n_triangles, obj->next_vertex_gid};
  MPI_Allreduce(MPI_IN_PLACE, st1, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  const int n_after_first = st1[0];
  const int gid_range_after_first = st1[1];

  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &second) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(second.converged, MPI_COMM_WORLD));

  /* An already-converged mesh must not keep growing: the second pass should
   * change the triangle count by a few percent, not by tens. */
  int st2[3] = {obj->global_n_triangles, obj->next_vertex_gid, obj->global_n_vertices};
  MPI_Allreduce(MPI_IN_PLACE, st2, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const double drift = fabs((double) (st2[0] - n_after_first)) / (double) n_after_first;
  TEST_ASSERT(drift < 0.15);

  /* Density is what keeps the id RANGE proportional to the mesh, and the range
   * is what sizes redistribution's arrival table. A range growing faster than
   * the mesh is deferred issue 15 coming back. */
  TEST_ASSERT(st2[1] < gid_range_after_first * 2);
  TEST_ASSERT(st2[1] == st2[2]);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Main
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Regime C: merging rank boxes rescues a partition too fine for the band.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT THIS IS FOR
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `test_a_permanently_frozen_band_is_not_certified` above pins the failure: at a
 * target of 0.40 on this fixture the observed band exceeds h/4, the schedule
 * cannot offer every entity to anybody, and the driver correctly refuses to
 * certify. This test is the same configuration with `parallel_coarsen` on, and
 * it must converge -- which is the whole claim of the coarsening path.
 *
 * Measured on that configuration, np = 4 and np = 8: 32 phases and 20 / 76
 * hidden edges without coarsening, against 2 phases and nothing hidden with it.
 *
 * ── NON-VACUITY, and it needs two separate statements ──────────────────────
 *
 * A test that merely passes a flag proves nothing: if the factors came back
 * {1,1,1} this would be an uncoarsened run wearing a parameter, and it would
 * fail for the same reason the test above expects failure. So it asserts that
 * grouping ACTUALLY happened (`n_coarse_groups > 1`) and, separately, that the
 * grouping had the consequence it is supposed to have -- at least one rank ends
 * up holding nothing at all, which is the load-balance price the header is blunt
 * about. Assert the price, so nobody later believes it was avoided.
 */
static int test_a_coarsened_pass_rescues_a_partition_that_is_too_fine(void)
{
  if (g_nproc == 1) return 0;  // no interior plane, nothing to merge

  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 2);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTL_RemeshParameters params;
  standard_params(&params, 0.40);
  params.parallel_coarsen = 1;

  mesh_measure before;
  measure(ens, &params, &before);
  TEST_ASSERT(before.n_cross_rank_edges > 0.0);

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* NON-VACUITY 1: the factors were not {1,1,1}. */
  TEST_ASSERT(agreed(stats.n_coarse_groups > 1, MPI_COMM_WORLD));

  /* ── The claim: it converges, where the uncoarsened run does not ─────────
   *
   * Both halves are asserted, and the second is the one that makes the first
   * mean anything. A test that only showed "the coarsened pass converged" would
   * still pass if the configuration had never been hard, and this configuration
   * being hard is a property of the fixture that a later edit could remove
   * without noticing. So the same mesh is remeshed both ways in the same run. */
  TEST_ASSERT(agreed(stats.converged, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_edges_hidden_long == 0, MPI_COMM_WORLD));

  /* ── And the mesh is a mesh ────────────────────────────────────────────── */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));

  mesh_measure after;
  measure(ens, &params, &after);
  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.volume > 0.0, MPI_COMM_WORLD));

  /* The control: the identical fixture with coarsening off must NOT come back
   * with a clean certificate, or this test is measuring nothing. */
  MFTL_Ensemble *plain = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(plain != MFTL_NULL);
  add_sphere(plain, 0, 0.0, 0.0, 0.0, 0.7, 2);
  TEST_ASSERT(agreed(partition(plain), MPI_COMM_WORLD));

  MFTL_RemeshParameters plain_params;
  standard_params(&plain_params, 0.40);
  plain_params.parallel_coarsen = 0;

  MFTL_RemeshStats plain_stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(plain, &plain_params, &plain_stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */
  TEST_ASSERT(agreed(!(plain_stats.converged && plain_stats.n_edges_hidden == 0), MPI_COMM_WORLD));

  /* And it is CHEAPER, not merely more correct: the uncoarsened run spends its
   * whole budget failing. */
  TEST_ASSERT(agreed(stats.n_phases < plain_stats.n_phases, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d: coarsen=%d ranks/group -> %d phases, converged=%d, hidden %d/%d"
           "  |  uncoarsened -> %d phases, converged=%d, hidden %d/%d\n    ",
           g_nproc, stats.n_coarse_groups, stats.n_phases, stats.converged ? 1 : 0, stats.n_edges_hidden_long, stats.n_edges_hidden_short,
           plain_stats.n_phases, plain_stats.converged ? 1 : 0, plain_stats.n_edges_hidden_long, plain_stats.n_edges_hidden_short);
  }

  MFTLEnsembleDestroy(&plain);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The budget is a hard bound, and the migration count follows from it.
 *
 * Two claims that used to be assumed. `n_phases <= budget` is the contract of
 * the parameter and nothing checked it. `n_migrations + n_migrations_skipped ==
 * n_phases + 1` is the accounting identity that says every placement either
 * migrated or was recognised as the identity -- it is what would break if a
 * future change forgot to count one of the two, and a missed migration is
 * invisible in the mesh.
 *
 * Budget 1 is included deliberately: it is the smallest legal value, it is the
 * one a per-timestep solver might reach for, and it is where phase 0 is the ONLY
 * phase -- so the identity above reduces to "the one placement was the canonical
 * one, was skipped, and the closing redistribution was the only real migration".
 */
static int test_the_budget_is_honoured_exactly(void)
{
  const int budgets[] = {1, 2, 4, 32};

  for (int b = 0; b < 4; ++b)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    MFTL_RemeshParameters params;
    standard_params(&params, 0.06);
    params.parallel_phase_budget = budgets[b];

    MFTL_RemeshStats stats;
    TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    TEST_ASSERT(agreed(stats.n_phases >= 1 && stats.n_phases <= budgets[b], MPI_COMM_WORLD));
    TEST_ASSERT(agreed(stats.n_migrations + stats.n_migrations_skipped == stats.n_phases + 1, MPI_COMM_WORLD));

    /* Whatever the budget, the mesh handed back is a mesh. A budget too small is
     * a QUALITY shortfall and must never be a structural one. */
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));

    /* Phase 0 asks for the placement the mesh already has, so at least one
     * migration is always recognised as the identity and skipped. That is the
     * saving being asserted, not merely observed. */
    if (g_nproc > 1) TEST_ASSERT(agreed(stats.n_migrations_skipped >= 1, MPI_COMM_WORLD));

    MFTLEnsembleDestroy(&ens);
  }
  return 0;
}

/**
 * @brief The same input remeshes to the same bits, twice in a row.
 *
 * The property the whole design exists to have -- "the answer does not depend on
 * message arrival order, because no two ranks ever touch the same entity" -- and
 * until now it was measured only by an example, never asserted by the suite.
 *
 * That gap mattered as soon as the schedule stopped being a fixed cycle. The
 * placement table is now deduplicated against the process grid and a migration
 * is skipped when it would move nothing, so which placement runs at phase p is
 * derived from data rather than from a compile-time table. Both derivations are
 * from REPLICATED data and are therefore reproducible -- this is the test that
 * says so rather than the comment.
 *
 * The digest folds global ids and the raw bits of every coordinate, XOR-reduced.
 * XOR because it is associative and commutative, so the reduction is independent
 * of rank order; raw bits because a tolerance would hide exactly the kind of
 * drift being looked for. Only ever compared BETWEEN RUNS AT THE SAME RANK
 * COUNT -- two rank counts legitimately produce different triangulations.
 */
static unsigned long long mesh_digest(const MFTL_Object *obj)
{
  unsigned long long d = 0;

  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    unsigned long long h = (unsigned long long) obj->verts.rec[v].global_id * 0x9E3779B97F4A7C15ull;
    for (int k = 0; k < 3; ++k)
    {
      unsigned long long bits;
      memcpy(&bits, &obj->verts.pos[3 * v + k], sizeof(bits));
      h ^= bits + 0x165667B19E3779F9ull + (h << 6) + (h >> 2);
    }
    d ^= h;
  }

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    unsigned long long h = (unsigned long long) obj->tris.rec[t].global_id * 0xC2B2AE3D27D4EB4Full;
    for (int k = 0; k < 3; ++k) h ^= (unsigned long long) obj->verts.rec[obj->tris.verts[3 * t + k]].global_id << (k * 8);
    d ^= h;
  }

  unsigned long long all = 0;
  MPI_Allreduce(&d, &all, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);
  return all;
}

static int test_two_passes_are_bit_identical(void)
{
  unsigned long long digest[2] = {0, 0};

  for (int run = 0; run < 2; ++run)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    MFTL_RemeshParameters params;
    standard_params(&params, 0.06);
    params.parallel_phase_budget = 4;

    MFTL_RemeshStats stats;
    TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    digest[run] = mesh_digest(obj);
    MFTLEnsembleDestroy(&ens);
  }

  /* Non-vacuity: a digest of zero would compare equal to itself forever. */
  TEST_ASSERT(agreed(digest[0] != 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(digest[0] == digest[1], MPI_COMM_WORLD));

  if (g_rank == 0) printf("\n    np=%d digest %016llx\n    ", g_nproc, digest[0]);
  return 0;
}

/**
 * @brief Only the placements that are actually different are visited.
 *
 * MFTLRemeshPhaseShift leaves an axis with fewer than two cells at shift zero,
 * so on a 2x1x1 grid all eight table entries are TWO placements and on 2x2x1
 * they are four. Two things followed from that and neither was accounted for:
 * a repeated placement migrates nothing but still pays a full set of sweeps for
 * a phase whose frozen set is identical, and the convergence streak demanded
 * eight consecutive silent placements to prove something two already prove.
 *
 * This asserts the count the driver derived against the count the grid implies,
 * computed here from `cuts` alone. Exact, not a bound: 2^(cut axes).
 *
 * Non-vacuity is structural. At np = 2 and 4 the fixture's grid is genuinely
 * anisotropic (2x1x1 and 2x2x1), so the expected answer is 2 and 4 -- if the
 * deduplication were removed this would read 8 and fail. At np = 8 and 27 every
 * axis is cut and the answer is 8, which is the case that proves the
 * deduplication does not over-reach.
 */
static int test_only_the_distinct_placements_are_visited(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  (void) obj;
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  int expected = 1;
  for (int k = 0; k < 3; ++k)
    if (ens->domain.cuts[k] > 1) expected *= 2;

  MFTL_RemeshParameters params;
  standard_params(&params, 0.06);
  params.parallel_phase_budget = 4;

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  TEST_ASSERT(agreed(stats.n_distinct_placements == expected, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_distinct_placements >= 1 && stats.n_distinct_placements <= MFTL_REMESH_NUM_PHASES, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d grid %dx%dx%d: %d distinct placements of 8\n    ", g_nproc, ens->domain.cuts[0], ens->domain.cuts[1], ens->domain.cuts[2],
           stats.n_distinct_placements);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief ARMED IS NOT ACTIVE: on a healthy partition the fallback changes nothing.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY THIS IS THE FIRST TEST OF THE COARSENING PATH AND NOT THE LAST
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `parallel_coarsen` defaults to 1, so EVERY parallel remesh in this library now
 * runs the fallback's decision code. The claim that makes that acceptable is not
 * "it usually decides not to coarsen" -- it is that when it decides not to, the
 * pass is the one that ran before, to the bit. MFTLRemeshCoarseningFactors
 * returns {1,1,1} and MFTLRemeshCoarseOwnerProc at {1,1,1} is documented as
 * bit-for-bit MFTLRemeshOwnerProc, and this is where that stops being a comment.
 *
 * A weaker version of this test -- assert `n_coarse_groups == 1` and stop --
 * would pass even if the armed path perturbed the mesh, because a remesher's
 * output is a valid mesh either way. So the fixture is remeshed TWICE, armed and
 * disarmed, and the digests must be equal. `mesh_digest` folds every global id
 * and the raw bits of every coordinate, so a single vertex a half-ulp out fails
 * it.
 *
 * ── Non-vacuity, in both directions ──────────────────────────────────────
 *
 * The trivial way to pass is for the fixture to be one the fallback would never
 * consider, so the test asserts the pass was a real distributed one: cross-rank
 * edges exist and the digest is non-zero. And the frozen fraction is printed
 * rather than asserted, because it is the number that says how close this
 * fixture sits to the trigger -- a later edit that widens the boxes would leave
 * this passing while testing less.
 */
static int test_the_fallback_stays_off_on_a_healthy_partition(void)
{
  unsigned long long digest[2] = {0, 0};
  MFTL_RemeshStats stats[2];

  for (int armed = 0; armed < 2; ++armed)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    /* Target 0.10 on a refine-3 sphere: the mesh is near its target, so the band
     * is small against the rank box on every grid this suite runs. Measured
     * frozen share 8.6 / 16.7 / 24.4 / 36.6 % at np = 2 / 4 / 8 / 27, all of them
     * comfortably under what the correctness trigger reacts to. */
    MFTL_RemeshParameters params;
    standard_params(&params, 0.10);
    params.parallel_phase_budget = 4;
    params.parallel_coarsen = armed ? 1 : 0;

    if (g_nproc > 1)
    {
      mesh_measure before;
      measure(ens, &params, &before);
      TEST_ASSERT(before.n_cross_rank_edges > 0.0);
    }

    TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats[armed]) == MFTL_SUCCESS, MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    digest[armed] = mesh_digest(obj);
    MFTLEnsembleDestroy(&ens);
  }

  /* THE CLAIM. Armed, it declined to coarsen -- and the mesh is the same one the
   * disarmed pass produced, not merely an equally valid one. */
  TEST_ASSERT(agreed(stats[1].n_coarse_groups == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].coarsen_trigger == MFTL_REMESH_COARSEN_NONE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(digest[0] != 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(digest[0] == digest[1], MPI_COMM_WORLD));

  /* And the pass really was the same pass, not just the same mesh reached two
   * ways: the phase count is what the placements and the budget decide, so a
   * different decomposition would almost certainly change it. */
  TEST_ASSERT(agreed(stats[0].n_phases == stats[1].n_phases, MPI_COMM_WORLD));

  /* The armed run measured the frozen share; the disarmed one had no reason to
   * and must report zero rather than a stale or invented number. */
  TEST_ASSERT(agreed(stats[0].frozen_fraction == 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].frozen_fraction >= 0.0 && stats[1].frozen_fraction <= 1.0, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d: armed -> %d ranks/group, frozen %.1f %%, %d phases; digest %016llx matches disarmed\n    ", g_nproc, stats[1].n_coarse_groups,
           100.0 * stats[1].frozen_fraction, stats[1].n_phases, digest[1]);
  }

  return 0;
}

/**
 * @brief The EFFICIENCY trigger fires where the correctness one does not.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE TWO TRIGGERS HAVE TO BE SEPARABLE, OR ONE OF THEM IS DECORATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `coarsen_frozen_fraction` exists because `h > 4*delta` permits a partition in
 * which 87 % of the mesh is frozen. If every configuration that trips the
 * frozen threshold also tripped the band condition, the second trigger would be
 * reporting the first one's decision under a different name and could be deleted
 * without changing any outcome.
 *
 * So this test does not merely show that a low threshold coarsens. It shows the
 * BAND CONDITION HOLDING while it does -- by computing the band factors itself,
 * from the same measured band the driver used, and asserting they are {1,1,1}.
 * That is the whole claim, and it is why the test carries an arithmetic check
 * rather than only a set of stats assertions.
 *
 * The threshold is set here rather than taken from the default, because the
 * default is 1.0: the measurement said this trigger does not pay in general (see
 * the field, and docs/analysis.md §5.6). A disabled-by-default
 * feature still has to work when enabled, and that is what is being pinned.
 */
static int test_a_frozen_majority_triggers_the_fallback(void)
{
  if (g_nproc == 1) return 0;  // no interior plane, nothing frozen, nothing to trigger

  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  MFTL_RemeshParameters params;
  standard_params(&params, 0.10);
  params.parallel_phase_budget = 4;
  params.parallel_coarsen = 1;

  /* Below every frozen share this fixture produces (8.6 % at np = 2 is the
   * smallest), so the trigger fires on every grid the suite runs -- including
   * the ones where the band condition is satisfied with room to spare. */
  params.coarsen_frozen_fraction = 0.05;

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

  /* ── THE CLAIM: it fired, and the correctness trigger did not ──────────── */
  TEST_ASSERT(agreed((stats.coarsen_trigger & MFTL_REMESH_COARSEN_FROZEN) != 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed((stats.coarsen_trigger & MFTL_REMESH_COARSEN_BAND) == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_coarse_groups > 1, MPI_COMM_WORLD));

  /* Restated from the outside, against the band the driver actually measured.
   * `band_measured` is the observed reach and `max_edge_length` is what the
   * driver predicts a converging pass will grow it to -- the same MAX the
   * decision uses, so this is the driver's own correctness input, recomputed. */
  {
    const double delta = MFTL_MAX(stats.band_measured, params.max_edge_length);
    int band_group[3];
    MFTLRemeshCoarseningFactors(&ens->domain, &params, delta, band_group);
    TEST_ASSERT(agreed(band_group[0] * band_group[1] * band_group[2] == 1, MPI_COMM_WORLD));
  }

  /* ── And the mesh is a mesh ─────────────────────────────────────────────── */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(test_body(ens, 0)), MPI_COMM_WORLD));

  mesh_measure after;
  measure(ens, &params, &after);
  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.volume > 0.0, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d: frozen %.1f %% > 5 %% -> %d ranks/group, band condition held (trigger mask %d)\n    ", g_nproc, 100.0 * stats.frozen_fraction,
           stats.n_coarse_groups, stats.coarsen_trigger);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A gathered pass is one placement, two migrations, and the same answer.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE GATHER IS NOT A SPECIAL CASE, AND THIS IS HOW THAT IS CHECKED
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `serial_gather_max_triangles` adds no communication code at all. It sets
 * `group[k] = cuts[k]`, which the coarsening machinery already clamps to and
 * already maps to a single representative cell. Everything else is supposed to
 * follow: nothing replicated, so nothing frozen; no axis with an interior plane,
 * so one distinct placement; one migration out and the closing MFTLRedistribute
 * back.
 *
 * Each of those is asserted separately, because they are separate claims and a
 * single "it produced a mesh" would not distinguish a gather from an ordinary
 * pass that happened to converge.
 *
 * ── Why the comparison is a TOLERANCE and not equality ───────────────────
 *
 * The gathered pass runs the serial kernel over the whole mesh in one sweep; the
 * distributed one runs gated sweeps over several placements. Split and collapse
 * are order-dependent, so the two legitimately choose different edges and land
 * on different triangulations OF THE SAME SURFACE at the same resolution. The
 * bounds are the ones test_the_two_strategies_reach_the_same_mesh uses for the
 * same reason and must not be tightened to equality.
 */
static int test_a_gathered_pass_matches_the_distributed_answer(void)
{
  if (g_nproc == 1) return 0;  // one rank IS the gather; there is nothing to compare

  mesh_measure result[2];
  MFTL_RemeshStats stats[2];

  for (int gather = 0; gather < 2; ++gather)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    /* 0.06 on a refine-3 sphere, so the pass REFINES 1280 -> about 3800. At the
     * mesh's own resolution both runs would return it unchanged and the
     * comparison below would hold for a reason that has nothing to do with
     * either path. */
    MFTL_RemeshParameters params;
    standard_params(&params, 0.06);
    params.parallel_phase_budget = 8;

    /* Comfortably above the refined mesh so the trigger is certain to fire, and
     * 0 for the control. */
    params.serial_gather_max_triangles = 100000 * (gather ? 1 : 0);

    if (g_nproc > 1)
    {
      mesh_measure before;
      measure(ens, &params, &before);
      TEST_ASSERT(before.n_cross_rank_edges > 0.0);
    }

    TEST_ASSERT(agreed(MFTLRemeshEnsembleParallel(ens, &params, &stats[gather]) == MFTL_SUCCESS, MPI_COMM_WORLD));
    obj = test_body(ens, 0);  /* the pointer does not survive a repartition */

    /* The postconditions are the driver's, gathered or not -- in particular the
     * partition that comes back is the canonical one, because the closing
     * migration is a plain MFTLRedistribute in both cases. */
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ghosts_agree_with_owners(ens), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(partition_matches_canonical_planes(ens), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(global_ids_are_dense(obj), MPI_COMM_WORLD));

    measure(ens, &params, &result[gather]);
    MFTLEnsembleDestroy(&ens);
  }

  /* ── The mechanism, claim by claim ─────────────────────────────────────── */
  TEST_ASSERT(agreed(stats[1].coarsen_trigger == MFTL_REMESH_COARSEN_GATHER, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].n_coarse_groups == g_nproc, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].n_distinct_placements == 1, MPI_COMM_WORLD));

  /* ── TWO MIGRATIONS, WHATEVER THE PHASE COUNT, and that is the point ────
   *
   * One coarse box has no interior plane, so every phase asks for the same
   * placement and every phase after the first is detected as the identity and
   * skipped. So a mesh far enough from its target to need several phases of
   * sweeping still moves exactly twice: the gather out, and the closing
   * redistribute.
   *
   * Exactly 2 and not 1. One would mean the gather itself was the identity --
   * that nothing was gathered -- which is how this assertion would catch a
   * `group` that silently came back as {1,1,1}. */
  TEST_ASSERT(agreed(stats[1].n_migrations == 2, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].n_migrations_skipped == stats[1].n_phases - 1, MPI_COMM_WORLD));

  /* Nothing is replicated on one rank, so the certificate has nothing to hide. */
  TEST_ASSERT(agreed(stats[1].converged, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].n_edges_hidden_long == 0, MPI_COMM_WORLD));

  /* Non-vacuity: the control really did take the distributed path. */
  TEST_ASSERT(agreed(stats[0].coarsen_trigger == MFTL_REMESH_COARSEN_NONE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[0].n_coarse_groups == 1, MPI_COMM_WORLD));

  /* ── And it is the same surface at the same resolution ─────────────────── */
  TEST_ASSERT(agreed(result[1].n_triangles > 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(result[1].n_triangles - result[0].n_triangles) <= 0.04 * result[0].n_triangles, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(result[1].volume - result[0].volume) <= 0.01 * fabs(result[0].volume), MPI_COMM_WORLD));

  const double chi = result[1].n_vertices - result[1].n_edges + result[1].n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d: gathered -> %d ranks/group, %d phases, %d migrations, %.0f tris | distributed -> %d phases, %.0f tris", g_nproc,
           stats[1].n_coarse_groups, stats[1].n_phases, stats[1].n_migrations, result[1].n_triangles, stats[0].n_phases, result[0].n_triangles);
    printf("  (both refined from 1280)\n    ");
  }

  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Parallel remeshing (np = %d) ===\n", g_nproc);

  RUN_TEST(test_frozen_is_exactly_the_replicated_set);
  RUN_TEST(test_dilation_only_grows_the_band);
  RUN_TEST(test_compaction_restores_the_ownership_prefix);
  RUN_TEST(test_a_permanently_frozen_band_is_not_certified);
  RUN_TEST(test_a_coarsened_pass_rescues_a_partition_that_is_too_fine);
  RUN_TEST(test_the_fallback_stays_off_on_a_healthy_partition);
  RUN_TEST(test_a_frozen_majority_triggers_the_fallback);
  RUN_TEST(test_a_gathered_pass_matches_the_distributed_answer);
  RUN_TEST(test_a_partitioned_sphere_remeshes_on_a_periodic_domain);
  RUN_TEST(test_a_sphere_on_the_periodic_seam_remeshes_everywhere);
  RUN_TEST(test_a_partitioned_sphere_remeshes_with_walls);
  RUN_TEST(test_two_bodies_keep_separate_id_spaces);
  RUN_TEST(test_a_dilated_band_reaches_the_same_mesh);
  RUN_TEST(test_the_edge_length_distribution_matches_the_serial_answer);
  RUN_TEST(test_the_edge_length_band_is_a_collective_measurement);
  RUN_TEST(test_a_second_pass_is_stable);
  RUN_TEST(test_only_the_distinct_placements_are_visited);
  RUN_TEST(test_the_budget_is_honoured_exactly);
  RUN_TEST(test_two_passes_are_bit_identical);

  if (g_rank == 0) TEST_REPORT();

  int local_failed = g_tests_failed;
  int total_failed = 0;
  MPI_Allreduce(&local_failed, &total_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failed > 0) ? 1 : 0;
}
