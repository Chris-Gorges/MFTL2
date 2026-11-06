/*
 * @Author: Christian Gorges
 * @Date: 2026-08-17
 *
 * @brief Distributed remeshing by MOVING THE BODIES: the dispersed strategy.
 *
 * Runs at np = 1, 2, 4 and 8. The companion file test_remeshing_parallel.c tests
 * the other strategy -- moving the cut planes -- and the two share a target and a
 * fixture on purpose, because the strongest test here is that they AGREE.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT MAKES THIS TESTABLE AT ALL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Remeshing is a sequence of local decisions taken against the mesh the previous
 * one produced, so two orderings give two different, equally valid
 * triangulations. Nothing here may compare meshes element by element. What CAN
 * be compared is the four things a remesher is actually for -- how many
 * triangles, how many edges outside the length band, the enclosed volume, and
 * the structural invariants -- and those are the same statement on one rank and
 * on many.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * NON-VACUITY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The failure mode of a test like this is that the bodies happen to sit inside
 * one rank each already, so the centering does nothing and the whole strategy
 * could be deleted with every assertion still passing. The fixture therefore
 * places bodies ON the cut planes -- one on all three of them -- and
 * `count_split_bodies()` counts how many are genuinely spread over more than one
 * rank BEFORE the pass. At np > 1 that count is asserted to be non-zero, so a
 * future change that shrinks the bodies or grows the domain fails loudly instead
 * of quietly becoming a serial test.
 *
 * EVERY assertion goes through agreed(): TEST_ASSERT returns immediately, so an
 * assertion failing on some ranks only would leave the rest blocked in the next
 * collective -- a hang instead of a clean failure.
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

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief [-1,1]^3, so every grid above has h_min = 1.0 and the arithmetic is legible. */
static MFTL_Ensemble *make_ensemble(bool periodic)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 2);
}

static MFTL_Object *add_sphere(MFTL_Ensemble *ens, int id, double cx, double cy, double cz, double r, int refine)
{
  MFTL_Object *obj = MFTLObjectCreate(id);
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {cx, cy, cz};
  MFTLGenerateSphere(obj, centre, r, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/**
 * @brief Four droplets, each far smaller than a rank box, each ON a cut plane.
 *
 * The placement is what makes the test non-vacuous at every rank count the suite
 * runs, and the three grids cut different axes:
 *
 *   np = 2   x = 0            bodies A and B straddle
 *   np = 4   x = 0, y = 0     A, B, C
 *   np = 8   x = 0, y = 0, z = 0   all four
 *
 * Radius 0.15 against h_min = 1.0 is 0.15 * h, comfortably inside the 0.4 * h
 * that AUTO calls dispersed and comfortably inside the 0.5 * h that keeps the
 * centering migration a 26-neighbour exchange.
 */
static void add_four_droplets(MFTL_Ensemble *ens)
{
  add_sphere(ens, 0, 0.00, 0.00, 0.00, 0.15, 2);
  add_sphere(ens, 1, 0.00, 0.50, 0.50, 0.13, 2);
  add_sphere(ens, 2, -0.50, 0.00, 0.50, 0.12, 2);
  add_sphere(ens, 3, 0.50, 0.50, 0.00, 0.10, 2);
}

/** @brief The parameters every test uses, so a change is a change everywhere. */
static void standard_params(MFTL_RemeshParameters *p, double target)
{
  /* Feature detection OFF: a sphere has no features, so every flagged edge is a
   * false positive that locks a vertex the smoother needed. See
   * docs/analysis.md §4. */
  MFTLInitRemeshParams(p, MFTL_REMESH_ISOTROPIC, target, 4, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p->smooth_iterations = 3;
  p->smooth_relaxation = 0.5;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Global measurements
 *
 * Each UNDIRECTED edge is counted exactly once across all ranks -- attributed to
 * the owner of the incident triangle of SMALLER GLOBAL ID. Counting "my triangle
 * whose twin is a ghost" instead double-counts every cut edge, by an amount that
 * GROWS WITH THE RANK COUNT, which is the one shape of error a partitioning test
 * must not have.
 * ─────────────────────────────────────────────────────────────────────────── */

static int counts_this_edge(const MFTL_Object *obj, int h)
{
  const int t = MFTLHeTriIdx(h);
  if (t >= obj->tris.n_local) return 0;

  const int tw = obj->tris.he[h].twin_idx;
  if (tw == MFTL_ERROR) return 1;

  const int to = MFTLHeTriIdx(tw);
  if (to < obj->tris.n_local) return h < tw;

  return obj->tris.rec[t].global_id < obj->tris.rec[to].global_id;
}

typedef struct
{
  double n_vertices;
  double n_triangles;
  double n_edges;
  double n_out_of_band;
  double volume;  ///< Enclosed volume by the divergence theorem, summed over bodies.
} mesh_measure;

static void measure(const MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, mesh_measure *out)
{
  double s[5] = {0, 0, 0, 0, 0};

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    s[0] += (double) obj->verts.n_local;
    s[1] += (double) obj->tris.n_local;

    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!counts_this_edge(obj, h)) continue;

      const double len = MFTLGetEdgeLength(obj, h);
      s[2] += 1.0;
      if (len > params->max_edge_length || len < params->min_edge_length) s[3] += 1.0;
    }

    /* (1/6) sum p0 . (p1 x p2) over OWNED triangles, built from the SEAM-
     * CONSISTENT corners rather than from raw positions. The droplets in this
     * file sit on periodic seams on purpose, and raw coordinates would make the
     * divergence-theorem sum meaningless for exactly those bodies. */
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double c[3][3];
      MFTLGetTriangleCorners(obj, t, c);

      s[4] += (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) + c[0][1] * (c[1][2] * c[2][0] - c[1][0] * c[2][2]) +
               c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])) /
              6.0;
    }
  }

  double g[5];
  MPI_Allreduce(s, g, 5, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  out->n_vertices = g[0];
  out->n_triangles = g[1];
  out->n_edges = g[2];
  out->n_out_of_band = g[3];
  out->volume = g[4];
}

/**
 * @brief How many bodies are genuinely spread over more than one rank?
 *
 * The non-vacuity measure. Counts, globally, the (body, rank) pairs holding at
 * least one OWNED triangle and subtracts one per body: zero means every body
 * already lives on a single rank and the centering has nothing to do.
 */
static int count_split_bodies(const MFTL_Ensemble *ens)
{
  int shares = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) shares += (ens->local_objects[o]->tris.n_local > 0) ? 1 : 0;

  int total = 0;
  MPI_Allreduce(&shares, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  /* Minus the number of bodies in the ENSEMBLE, not the number this rank holds:
   * `total` is a global sum of (body, rank) pairs, so subtracting a local count
   * would give a different answer on every rank. */
  return total - ens->n_bodies_global;
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
 * @brief Are this body's global ids a dense permutation of [0, N)?
 *
 * Not the serial invariant: that one compares against the local pool size, which
 * on a partitioned mesh is a fraction of the truth. Here each rank marks the ids
 * it OWNS in an array sized by `next_*_gid`, the marks are OR-reduced, and the
 * result must be all ones with no gaps.
 *
 * Density is not cosmetic. redistribute.c and the ghost exchanges size their
 * arrival tables from `next_*_gid`, so a pass that leaves holes grows those
 * tables without bound over a run of constant mesh size.
 */
static int global_ids_are_dense(const MFTL_Object *obj)
{
  /* Two things this has to get right now that a rank may hold none of the body.
   *
   * The LENGTH is agreed first: `obj` may be test_body()'s shared empty, whose
   * counters are zero, and a locally-sized reduction would then truncate rather
   * than report the disagreement.
   *
   * And nothing returns early. Every exit used to be a bare `return 0`, which on
   * one rank leaves the others waiting in the Allreduce -- a hang instead of a
   * failure, which is exactly what the suite's `agreed()` discipline exists to
   * prevent. The verdict is accumulated and the collective always runs. */
  int range[2] = {obj->next_vertex_gid, obj->next_triangle_gid};
  MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const int nv = range[0];
  const int nt = range[1];
  if (nv < 0 || nt < 0) return 0;  // agreed on every rank, so returning here is safe

  char *const mark = (char *) calloc((size_t) (nv + nt) + 2, 1);
  char *const got = (char *) calloc((size_t) (nv + nt) + 2, 1);

  int ok = 1;

  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g < 0 || g >= nv)
      ok = 0;
    else if (mark[g])  // owned twice: two ranks claim the same id
      ok = 0;
    else
      mark[g] = 1;
  }
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g < 0 || g >= nt)
      ok = 0;
    else if (mark[nv + g])
      ok = 0;
    else
      mark[nv + g] = 1;
  }

  MPI_Allreduce(mark, got, nv + nt, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);

  for (int i = 0; i < nv + nt; ++i)
  {
    if (!got[i]) ok = 0;
  }

  free(mark);
  free(got);
  return ok;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. Strategy selection
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The classification, from both sides of its one threshold.
 *
 * One test rather than two, because the claim is a COMPARISON and testing only
 * the side you expect passes for a function that always returns that answer.
 */
static int test_the_strategy_follows_the_body_size(void)
{
  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);

  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);
    add_four_droplets(ens);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    const MFTL_RemeshParallelStrategy s = MFTLRemeshChooseStrategy(ens, &params);
    /* At one rank the question is meaningless and SPANNING is the documented
     * answer -- MFTLRemeshEnsemble never reaches either driver there. */
    TEST_ASSERT(agreed(s == (g_nproc > 1 ? MFTL_REMESH_PARALLEL_DISPERSED : MFTL_REMESH_PARALLEL_SPANNING), MPI_COMM_WORLD));
    MFTLEnsembleDestroy(&ens);
  }

  {
    /* r = 0.7 against h_min = 1.0: 0.7 > 0.4, so no rank box can hold it. */
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);
    add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 2);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    TEST_ASSERT(agreed(MFTLRemeshChooseStrategy(ens, &params) == MFTL_REMESH_PARALLEL_SPANNING, MPI_COMM_WORLD));
    MFTLEnsembleDestroy(&ens);
  }

  return 0;
}

/**
 * @brief A droplet ON the periodic seam must be measured at its true size.
 *
 * THE test for the circular mean in MFTLRemeshMeasureBodies, and it is the only
 * one that can distinguish it from the obvious implementation. A body straddling
 * x = +/-1 has half its vertices near +1 and half near -1, so:
 *
 *   ARITHMETIC MEAN   centroid lands near x = 0 -- the one place the body
 *                     certainly is not -- and the radius, being the distance to
 *                     the furthest corner, comes out near 1.0. The body is then
 *                     classified SPANNING, the ensemble takes the phase schedule
 *                     it does not need, and NOTHING FAILS. It is simply slower,
 *                     forever, for every emulsion with a droplet on a boundary.
 *
 *   CIRCULAR MEAN     centroid on the seam, radius 0.12, classified DISPERSED.
 *
 * The assertion is on the classification rather than on the centroid, because
 * that is the consequence a user can observe.
 */
static int test_a_seam_straddling_droplet_is_not_measured_as_domain_sized(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  /* x = 1.0 IS the seam of [-1, 1]. MFTLGenerateSphere wraps its vertices, so
   * the stored coordinates really are split between the two faces. */
  add_sphere(ens, 0, 1.0, 0.0, 0.0, 0.12, 2);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);

  TEST_ASSERT(agreed(MFTLRemeshChooseStrategy(ens, &params) == (g_nproc > 1 ? MFTL_REMESH_PARALLEL_DISPERSED : MFTL_REMESH_PARALLEL_SPANNING),
                     MPI_COMM_WORLD));

  /* And it must survive the pass it was just classified into. A body whose
   * centroid is computed on the seam is also a body whose centering migration
   * crosses the seam, which is a different piece of arithmetic. */
  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_bodies_not_centred == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(test_body(ens, 0)), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. The pass itself
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The main event: four split droplets, centred, remeshed, handed back.
 *
 * One test rather than eight, because all eight claims are about ONE mesh
 * produced by ONE expensive call. Ordered so that the most diagnostic failure
 * comes first: structure before identity before convergence, because a torn mesh
 * explains a wrong id count while a wrong id count explains nothing about a torn
 * mesh.
 */
static int test_four_split_droplets_are_centred_and_remeshed(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  add_four_droplets(ens);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);

  mesh_measure before;
  measure(ens, &params, &before);

  /* NON-VACUITY, asserted before the work rather than after it. */
  const int split = count_split_bodies(ens);
  if (g_nproc > 1) TEST_ASSERT(agreed(split > 0, MPI_COMM_WORLD));

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* ── Structure ─────────────────────────────────────────────────────────── */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  /* ONE agreed() for the whole ensemble, not one per local object. The loop bound
   * is now this rank's body count, so an agreed() inside it would make a
   * rank-dependent number of Allreduces -- a hang, and exactly the failure the
   * `agreed()` discipline exists to avoid. */
  {
    int sound = 1;
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      if (MFTLCheckMeshIntegrity(ens->local_objects[o]) != MFTL_SUCCESS) sound = 0;
    }
    TEST_ASSERT(agreed(sound, MPI_COMM_WORLD));
  }

  /* MFTL_FLAG_FROZEN is the spanning driver's transient state and this driver
   * never sets it. Asserting it anyway is what would catch a future refactor
   * that routed the dispersed path through the gated sweeps and left it behind. */
  int leaked = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v) leaked += MFTLVertexIsFrozen(obj, v) ? 1 : 0;
    for (int t = 0; t < obj->tris.n; ++t) leaked += MFTLTriangleIsFrozen(obj, t) ? 1 : 0;
  }
  TEST_ASSERT(agreed(leaked == 0, MPI_COMM_WORLD));

  /* ── The strategy did what it says ─────────────────────────────────────── */
  TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_bodies_not_centred == 0, MPI_COMM_WORLD));

  /* EXACTLY TWO migrations, whatever the mesh looked like -- out and back. This
   * is the entire performance claim of the dispersed strategy against the
   * spanning one (which measured 5 to 18 on a comparable problem), so it is
   * asserted rather than printed. */
  TEST_ASSERT(agreed(stats.n_migrations == 2, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_phases == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.converged, MPI_COMM_WORLD));

  /* ── Identity ──────────────────────────────────────────────────────────────
   * Over the BODY-ID space, not this rank's list: the check is collective, so
   * every rank has to run it for the same bodies in the same order. */
  for (int b = 0; b < ens->n_bodies_global; ++b)
  {
    TEST_ASSERT(agreed(global_ids_are_dense(test_body(ens, b)), MPI_COMM_WORLD));
  }

  mesh_measure after;
  measure(ens, &params, &after);

  /* Euler's formula, summed over four closed genus-0 bodies. The one check that
   * sees a triangle lost between two ranks or counted twice: both leave a
   * perfectly valid local mesh everywhere and only the GLOBAL count is wrong. */
  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0 * ens->n_bodies_global) < 0.5, MPI_COMM_WORLD));

  /* Over every body, and REDUCED. `global_n_triangles` is a property of the whole
   * body, but only a rank holding some of it carries the value -- the dense
   * view's stand-in reports zero -- so the per-body maximum over ranks is what
   * "the ensemble's triangle count" actually is. */
  double global_tris = 0.0;
  {
    const int nb = ens->n_bodies_global;
    int *const per_body = (int *) calloc((size_t) MFTL_MAX(nb, 1), sizeof(int));
    for (int b = 0; b < nb; ++b)
    {
      const MFTL_Object *const o = MFTLEnsembleFindLocalObject(ens, b);
      per_body[b] = o ? o->global_n_triangles : 0;
    }
    MPI_Allreduce(MPI_IN_PLACE, per_body, MFTL_MAX(nb, 1), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    for (int b = 0; b < nb; ++b) global_tris += (double) per_body[b];
    free(per_body);
  }
  TEST_ASSERT(agreed(fabs(after.n_triangles - global_tris) < 0.5, MPI_COMM_WORLD));

  /* ── Convergence ───────────────────────────────────────────────────────── */
  TEST_ASSERT(agreed(after.n_triangles > before.n_triangles, MPI_COMM_WORLD));
  /* THREE PER CENT, AND THE NUMBER IS MEASURED RATHER THAN CHOSEN. Which rank
   * remeshes a body decides the ORDER its pool arrives in, and split and
   * collapse are order-dependent -- the same reason
   * test_the_two_strategies_reach_the_same_mesh carries a 4 % tolerance on the
   * triangle count and must not be tightened to equality. Measured over the
   * configurations that are all equally correct here -- np = 1, 2, 4 and 8, with
   * the balancer on and off -- this quantity spans 1.31 % to 2.17 %, so a
   * threshold at 2 % is inside the noise and fails on the assignment rather than
   * on the remeshing. What it is actually for is catching a pass that did not
   * converge, which reports tens of per cent. */
  TEST_ASSERT(agreed(after.n_out_of_band < 0.03 * after.n_edges, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.volume > 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(after.volume - before.volume) < 0.05 * before.volume, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d dispersed: %.0f -> %.0f tris, %d split bodies before, %d migrations, "
           "out-of-band %.0f/%.0f, dV %.2f%%\n    ",
           g_nproc, before.n_triangles, after.n_triangles, split, stats.n_migrations, after.n_out_of_band, after.n_edges,
           100.0 * (after.volume - before.volume) / before.volume);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The two strategies must reach the same mesh, to within their freedom.
 *
 * The strongest test in this file, and the only one that could catch a
 * systematic error shared by nothing else: the two drivers have no code in
 * common past MFTLRemeshIterate -- different migrations, different id schemes,
 * different convergence criteria -- so an agreement between them is evidence
 * about the answer rather than about one implementation.
 *
 * The tolerance is 4 % on the triangle count and 1 % on the volume, and it must
 * NOT be tightened to equality: split and collapse are order-dependent, so the
 * two drivers legitimately choose different edges and land on different
 * triangulations of the same surface at the same resolution. Demanding equality
 * would be demanding that the parallelisation does nothing.
 */
static int test_the_two_strategies_reach_the_same_mesh(void)
{
  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);

  mesh_measure result[2];

  for (int which = 0; which < 2; ++which)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);
    add_four_droplets(ens);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    MFTL_RemeshStats stats;
    const int rc = (which == 0) ? MFTLRemeshEnsembleDispersed(ens, &params, &stats) : MFTLRemeshEnsembleParallel(ens, &params, &stats);

    TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(stats.converged, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

    measure(ens, &params, &result[which]);
    MFTLEnsembleDestroy(&ens);
  }

  const double dn = fabs(result[0].n_triangles - result[1].n_triangles);
  const double dv = fabs(result[0].volume - result[1].volume);

  TEST_ASSERT(agreed(dn < 0.04 * result[1].n_triangles, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(dv < 0.01 * fabs(result[1].volume), MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d dispersed %.0f tris / spanning %.0f tris (%.1f%%), volume %.6f vs %.6f\n    ", g_nproc, result[0].n_triangles,
           result[1].n_triangles, 100.0 * dn / result[1].n_triangles, result[0].volume, result[1].volume);
  }

  return 0;
}

/**
 * @brief Forcing DISPERSED on a body that cannot be centred must not corrupt.
 *
 * The strategy is a caller-settable parameter, and a body grows between the
 * measurement and the pass, so "the caller was wrong" is a state this code will
 * reach in production. What it must do then is detect it BEFORE remeshing
 * anything and hand the whole ensemble to the other driver -- never leave a
 * half-remeshed ensemble, whose per-rank id blocks no Exscan ever agreed on,
 * for the spanning driver to inherit.
 *
 * At np = 1 there is nothing to be wrong about: one rank holds everything, so
 * the body IS centred and the fallback correctly does not fire. The assertion is
 * therefore conditional, and that condition is itself the claim.
 */
static int test_a_spanning_body_forced_dispersed_falls_back_cleanly(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.06);
  params.parallel_strategy = MFTL_REMESH_PARALLEL_DISPERSED;

  mesh_measure before;
  measure(ens, &params, &before);

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* A 0.7 sphere against h = 1.0 cannot fit in a rank box at any np > 1, so the
   * fallback must have fired -- and the stats must say so, or a production run
   * would pay the wasted migration silently, every remesh, forever. */
  if (g_nproc > 1)
  {
    TEST_ASSERT(agreed(stats.n_bodies_not_centred > 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_SPANNING, MPI_COMM_WORLD));
  }

  /* And the answer is right anyway. That is the whole point of the fallback. */
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(test_body(ens, 0)) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(test_body(ens, 0)), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.converged, MPI_COMM_WORLD));

  mesh_measure after;
  measure(ens, &params, &after);

  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.n_triangles > before.n_triangles, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief MFTLRemeshEnsemble's AUTO dispatch must reach the dispersed driver.
 *
 * Everything above calls the driver directly. This one goes through the public
 * entry point, which is where the strategy is resolved -- and resolving it is a
 * COLLECTIVE decision. Two ranks entering different drivers is an immediate
 * deadlock with no diagnostic, so a test that only ever calls the drivers
 * directly would never see it. At np > 1 this test hangs rather than failing if
 * that ever breaks, which is the honest failure mode for a consensus bug.
 */
static int test_the_public_entry_point_dispatches_to_dispersed(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);

  add_four_droplets(ens);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);
  TEST_ASSERT(params.parallel_strategy == MFTL_REMESH_PARALLEL_AUTO);

  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

  if (g_nproc > 1) TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief AUTO must measure the bodies ONCE, not twice.
 *
 * MFTLRemeshMeasureBodies is the only O(num_objects) collective in the module:
 * one Allreduce of 7 doubles per body and then one of 1 per body. The obvious
 * spelling of the dispatcher -- choose a strategy, then call the public driver --
 * takes it twice and throws the first set of centroids away, which at ten
 * thousand droplets is half a megabyte per rank per pass computed for nothing.
 *
 * Nothing in a profile would show that, and nothing in the result would change,
 * so the arrangement that avoids it (MFTLRemeshEnsemble threads one measurement
 * from MFTLRemeshChooseStrategyMeasured into MFTLRemeshEnsembleDispersedMeasured)
 * is exactly the kind that gets undone by a later tidy-up. Hence a count.
 *
 * The public MFTLRemeshEnsembleDispersed still measures for itself -- it has no
 * caller to inherit from -- so that path is asserted separately at one.
 */
static int test_the_strategy_costs_one_measurement_not_two(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);
  add_four_droplets(ens);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.025);
  params.parallel_strategy = MFTL_REMESH_PARALLEL_AUTO;

  MFTL_RemeshStats stats;

  MFTLRemeshResetMeasurementCount();
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  const int through_auto = MFTLRemeshMeasurementCount();

  /* Non-vacuity: at one rank the dispatcher takes the serial path and never
   * measures at all, so the claim below would be trivially true. */
  if (g_nproc > 1)
  {
    TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(through_auto == 1, MPI_COMM_WORLD));
  }
  else
  {
    TEST_ASSERT(agreed(through_auto == 0, MPI_COMM_WORLD));
  }

  MFTLRemeshResetMeasurementCount();
  params.parallel_strategy = MFTL_REMESH_PARALLEL_DISPERSED;
  TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLRemeshMeasurementCount() == 1, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A pile-up of droplets is spread, and the answer does not change.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE FIXTURE IS THE TEST, AND IT HAS TO BE BUILT AGAINST THE REAL GRID
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Bodies go where their centroids are, so an imbalance is a statement about
 * WHERE the bodies are and nothing else. Sixteen droplets spread over the domain
 * would be roughly balanced already and the balancer would correctly do nothing
 * -- which is a passing test that measures nothing.
 *
 * So the droplets are packed into ONE rank's box, resolved against the actual
 * process grid at run time rather than at a hardcoded coordinate: the box of the
 * rank owning the domain's lower corner is found from `cuts`, and every droplet
 * is placed inside it. That is the same discipline test_bvh_ghosting.c uses for
 * its thirty placements, and for the same reason -- the grid is 2x1x1, 2x2x1,
 * 2x2x2 or 3x3x3 depending on np, and a coordinate that piles up on one of them
 * is spread across another.
 *
 * ── The four claims, and why each is separately necessary ────────────────
 *
 * BALANCED     `load_imbalance` falls to within the slack. The headline.
 * NON-VACUOUS  `n_bodies_reassigned > 0`. Without it, an imbalance that fixed
 *              itself -- or a fixture that was never piled up -- reads as
 *              success. This is the assertion a future edit to the fixture
 *              would trip.
 * CONTROLLED   the unbalanced run really is imbalanced. A control that is not
 *              measured is an assumption.
 * UNCHANGED    the mesh matches the unbalanced run's within the suite's
 *              tolerance. Which rank remeshes a body must not change what the
 *              body becomes -- and it very nearly does not: the serial kernel is
 *              deterministic and the body arrives complete, so the only freedom
 *              is pool order.
 */
static int test_the_balancer_evens_out_a_pile_up(void)
{
  if (g_nproc == 1) return 0;  // one rank holds everything; there is no imbalance to fix

  mesh_measure result[2];
  MFTL_RemeshStats stats[2];

  for (int balanced = 0; balanced < 2; ++balanced)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    /* One rank's box, from the grid this run actually has. The lower-corner cell
     * spans [-1, -1 + 2/cuts[k]) on each axis; the droplets sit inside it with
     * room for their radius. */
    double box[3];
    for (int k = 0; k < 3; ++k) box[k] = 2.0 / (double) ens->domain.cuts[k];

    /* Sixteen bodies so that N_bodies >= nproc at every rank count this suite
     * runs (2, 4, 8), which is the regime where an imbalance is fixable at all
     * -- below it some rank must idle whatever the assignment. */
    const int n_bodies = 16;
    for (int b = 0; b < n_bodies; ++b)
    {
      /* A deterministic scatter INSIDE the single box: a 4x2x2 lattice of
       * positions, inset so no droplet reaches the box's faces. */
      const double u[3] = {(double) ((b % 4) + 1) / 5.0, (double) (((b / 4) % 2) + 1) / 3.0, (double) ((b / 8) + 1) / 3.0};

      const double c[3] = {-1.0 + u[0] * box[0], -1.0 + u[1] * box[1], -1.0 + u[2] * box[2]};

      /* Radii deliberately UNEQUAL, so that equalising body COUNT and equalising
       * TRIANGLE COUNT are different answers and the weighted one is the one
       * being tested. Refine 1 and 2 differ by 4x in triangles. */
      add_sphere(ens, b, c[0], c[1], c[2], 0.02 * box[0], (b % 3 == 0) ? 2 : 1);
    }

    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    MFTL_RemeshParameters params;
    standard_params(&params, 0.02 * box[0]);
    params.dispersed_balance_slack = balanced ? 1.25 : 0.0;

    TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats[balanced]) == MFTL_SUCCESS, MPI_COMM_WORLD));

    TEST_ASSERT(agreed(stats[balanced].strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(stats[balanced].n_bodies_not_centred == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
    TEST_ASSERT(agreed(count_split_bodies(ens) == 0, MPI_COMM_WORLD));

    measure(ens, &params, &result[balanced]);
    MFTLEnsembleDestroy(&ens);
  }

  /* CONTROLLED: the unbalanced assignment really did pile up. Every droplet is
   * inside one rank's box, so its imbalance is nproc -- one rank holds all the
   * weight -- and anything at all below that would mean the fixture leaked. */
  TEST_ASSERT(agreed(stats[0].n_bodies_reassigned == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[0].load_imbalance > 1.5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[0].n_ranks_idle == g_nproc - 1, MPI_COMM_WORLD));

  /* NON-VACUOUS and BALANCED. */
  TEST_ASSERT(agreed(stats[1].n_bodies_reassigned > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].load_imbalance < stats[0].load_imbalance, MPI_COMM_WORLD));
  /* Sixteen bodies over at most eight ranks, so every rank can be given one and
   * "no rank idle" is a reachable claim rather than an aspiration. Below
   * N_bodies == nproc it would not be: a body is the smallest unit of work this
   * strategy has, so some rank must idle and no assignment can fix it. */
  TEST_ASSERT(agreed(16 >= g_nproc, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats[1].n_ranks_idle == 0, MPI_COMM_WORLD));

  /* The cap is `MAX(mean * slack, heaviest body)`, so the reachable imbalance is
   * bounded by the heaviest body rather than by the slack alone -- asserting
   * `<= 1.25` would be asserting that the bodies happen to be divisible, which
   * they are not. Both bounds are checked, which is the honest statement. */
  {
    const double heaviest_share = (double) g_nproc / 16.0;  // 16 bodies, weights within 4x
    const double reachable = MFTL_MAX(1.25, 4.0 * heaviest_share);
    TEST_ASSERT(agreed(stats[1].load_imbalance <= reachable, MPI_COMM_WORLD));
  }

  /* UNCHANGED: same surface, same resolution. Same tolerance and same reason as
   * test_the_two_strategies_reach_the_same_mesh. */
  TEST_ASSERT(agreed(result[1].n_triangles > 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(result[1].n_triangles - result[0].n_triangles) <= 0.04 * result[0].n_triangles, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(result[1].volume - result[0].volume) <= 0.01 * fabs(result[0].volume), MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d: unbalanced -> imbalance %.2f, %d idle | balanced -> imbalance %.2f, %d idle, %d/16 bodies moved\n    ", g_nproc,
           stats[0].load_imbalance, stats[0].n_ranks_idle, stats[1].load_imbalance, stats[1].n_ranks_idle, stats[1].n_bodies_reassigned);
  }

  return 0;
}

/**
 * @brief A rank that holds no body puts NOTHING on the wire for the balancer,
 *        however many bodies the ensemble has.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE ONE OBSERVABLE A SCALING CHANGE HAS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The balancer used to broadcast `4 * n_bodies_global` doubles so that every
 * rank could run the same greedy over the same table; it now decides each body
 * on one rank and delivers the verdict one hop along `comm_graph`. Both are
 * correct, both are reproducible, and both balance the same fixture to within a
 * few per cent -- so `test_the_balancer_evens_out_a_pile_up` above passes either
 * way and so does every digest, every soundness check and every volume in this
 * file. The difference is invisible to all of them, exactly as a per-body
 * reduction pinned to `comm_world` is invisible to everything except
 * MFTLEnsembleGetBodyReduceStats().
 *
 * So the assertion is on the PAYLOAD, and the sharp form of it is a DIFFERENCE:
 * double the body count and a rank that holds none of them must stage the same
 * number of doubles it staged before. Under replication that number doubles.
 *
 * ── Why the fixture is the pile-up and not the four droplets ──────────────
 *
 * Because it needs ranks that hold nothing while the ensemble is large, and
 * "every body inside one rank's box" is the only fixture here that produces one
 * at every rank count. It is also the case that makes the test non-vacuous in
 * the other direction: the balancer really runs, really moves bodies, and the
 * rank that DOES hold them really does stage more as the ensemble grows -- which
 * is asserted too, so that a counter wired to nothing cannot pass.
 */
static int test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold(void)
{
  if (g_nproc == 1) return 0;  // one rank holds everything; there is no non-holder

  const int counts[2] = {16, 32};
  long payload[2] = {0, 0};
  int held[2] = {0, 0};
  int reassigned[2] = {0, 0};

  for (int c = 0; c < 2; ++c)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    double box[3];
    for (int k = 0; k < 3; ++k) box[k] = 2.0 / (double) ens->domain.cuts[k];

    /* The same lattice as the pile-up fixture, stretched to `counts[c]` bodies
     * and kept inside the lower-corner cell so that exactly one rank holds
     * anything at all. */
    for (int b = 0; b < counts[c]; ++b)
    {
      const double u[3] = {(double) ((b % 4) + 1) / 5.0, (double) (((b / 4) % 2) + 1) / 3.0, (double) ((b / 8) % 4 + 1) / 5.0};
      const double x[3] = {-1.0 + u[0] * box[0], -1.0 + u[1] * box[1], -1.0 + u[2] * box[2]};
      add_sphere(ens, b, x[0], x[1], x[2], 0.02 * box[0], 1);
    }

    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    held[c] = ens->n_local_objects;

    MFTL_RemeshParameters params;
    standard_params(&params, 0.02 * box[0]);
    params.dispersed_balance_slack = 1.25;

    MFTL_RemeshStats stats;
    MFTLRemeshResetBalancerPayload();
    TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
    payload[c] = MFTLRemeshBalancerPayload();

    reassigned[c] = stats.n_bodies_reassigned;

    TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(stats.n_bodies_unreachable == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

    MFTLEnsembleDestroy(&ens);
  }

  /* NON-VACUOUS, both halves. Some rank must have held nothing -- otherwise
   * there is no non-holder to make a claim about -- and the balancer must have
   * actually moved bodies, or the payload being flat says nothing. */
  int any_empty = (held[0] == 0 && held[1] == 0) ? 1 : 0;
  MPI_Allreduce(MPI_IN_PLACE, &any_empty, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(any_empty == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(reassigned[0] > 0 && reassigned[1] > 0, MPI_COMM_WORLD));

  /* THE CLAIM. On a rank that held nothing the staged payload is the fixed
   * neighbourhood term and nothing else, so doubling the ensemble changes it by
   * zero. `4 * (32 - 16) = 64` more doubles is what the replicated form added. */
  const bool empty_here = (held[0] == 0 && held[1] == 0);
  TEST_ASSERT(agreed(!empty_here || payload[0] == payload[1], MPI_COMM_WORLD));

  /* And the counter is wired to something: the rank that DID hold the bodies
   * staged more of them the second time. Three doubles per body it speaks for,
   * so the growth is at least the sixteen bodies that were added. */
  const bool holder_here = (held[1] > held[0]);
  TEST_ASSERT(agreed(!holder_here || payload[1] >= payload[0] + 3 * 16, MPI_COMM_WORLD));

  {
    long shown[4] = {empty_here ? payload[0] : -1, empty_here ? payload[1] : -1, holder_here ? payload[0] : -1, holder_here ? payload[1] : -1};
    MPI_Allreduce(MPI_IN_PLACE, shown, 4, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0)
    {
      printf("\n    np=%d, 16 -> 32 bodies: a rank holding NONE staged %ld -> %ld doubles, one holding them all %ld -> %ld"
             " (replication staged %d -> %d on EVERY rank)\n    ",
             g_nproc, shown[0], shown[1], shown[2], shown[3], 4 * counts[0], 4 * counts[1]);
    }
  }

  return 0;
}

/**
 * @brief The balancer is a pure function of replicated data, so it repeats.
 *
 * The assignment involves a qsort and a greedy search, and both are places where
 * a comparator that is not a TOTAL order, or a candidate scan that reads
 * `dom->neighbors`, would produce a per-rank answer. Neither failure crashes:
 * MFTL_OwnerMap with two ranks naming different destinations for one entity is
 * undefined, and what it does in practice is drop or duplicate entities quietly.
 *
 * Two runs of the same fixture must give the same mesh to the bit. That is a
 * necessary condition for the assignment agreeing across ranks, and it is the
 * one a test can check without reaching inside the driver -- an assignment that
 * differed BETWEEN ranks would have failed the soundness checks above long
 * before it reached here, and an assignment that differed BETWEEN RUNS is what
 * this catches.
 */
static int test_the_balancer_is_deterministic(void)
{
  double volume[2] = {0.0, 0.0};
  double triangles[2] = {0.0, 0.0};

  for (int run = 0; run < 2; ++run)
  {
    MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
    TEST_ASSERT(ens != MFTL_NULL);

    add_four_droplets(ens);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    MFTL_RemeshParameters params;
    standard_params(&params, 0.05);

    MFTL_RemeshStats stats;
    TEST_ASSERT(agreed(MFTLRemeshEnsembleDispersed(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));

    mesh_measure m;
    measure(ens, &params, &m);
    volume[run] = m.volume;
    triangles[run] = m.n_triangles;

    MFTLEnsembleDestroy(&ens);
  }

  /* Non-vacuity: an empty ensemble would compare equal to itself forever. */
  TEST_ASSERT(agreed(triangles[0] > 0.0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(triangles[0] == triangles[1], MPI_COMM_WORLD));
  TEST_ASSERT(agreed(volume[0] == volume[1], MPI_COMM_WORLD));

  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Exactness against the serial remesher
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief splitmix64's finaliser: a cheap, well-mixed 64-bit hash step. */
static uint64_t mix64(uint64_t z)
{
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

/**
 * @brief One order-independent digest per body over OWNED entities, XOR-reduced over @p comm.
 *
 * A vertex contributes its global id and the raw bits of its position, a
 * triangle its global id and its corners' global ids in rotation order, so
 * equal digests mean the same surface, the same triangulation AND the same
 * numbering. XOR is exact and associative, so the partition cannot reach it.
 */
static void body_digests(const MFTL_Ensemble *ens, int n_bodies, uint64_t *out, MPI_Comm comm)
{
  uint64_t *local = (uint64_t *) calloc((size_t) n_bodies, sizeof(uint64_t));
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    uint64_t d = 0;
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      uint64_t h = mix64((uint64_t) (uint32_t) obj->verts.rec[v].global_id + 0x9E3779B97F4A7C15ull);
      for (int k = 0; k < 3; ++k)
      {
        uint64_t bits;
        memcpy(&bits, &obj->verts.pos[3 * v + k], sizeof bits);
        h = mix64(h ^ bits);
      }
      d ^= h;
    }
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      int c[3];
      MFTLGetTriangleVertices(obj, t, c);
      uint64_t h = mix64((uint64_t) (uint32_t) obj->tris.rec[t].global_id + 0xD1B54A32D192ED03ull);
      for (int k = 0; k < 3; ++k) h = mix64(h ^ (uint64_t) (uint32_t) obj->verts.rec[c[k]].global_id);
      d ^= h;
    }
    local[obj->global_id] ^= d;
  }
  MPI_Allreduce(local, out, n_bodies, MPI_UINT64_T, MPI_BXOR, comm);
  free(local);
}

/** @brief A smooth, position-only displacement of the owned vertices: the same bits on every rank. */
static void deform(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      double *p = MFTLVertexPosMut(obj, v);
      const double d[3] = {0.02 * sin(9.0 * p[1] + 1.0), 0.02 * sin(9.0 * p[2] + 2.0), 0.02 * sin(9.0 * p[0])};
      for (int k = 0; k < 3; ++k) p[k] += d[k];
      MFTLApplyPeriodicity(obj->domain, p);
    }
    MFTLObjectMarkChanged(obj);
  }
}

/**
 * @brief A dispersed remesh is the serial remesh, bit for bit, pass after pass.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY THIS CAN BE AN EQUALITY WHEN THE SPANNING TEST CANNOT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The dispersed driver moves each body whole to one rank and runs the serial
 * kernel on it, so the only thing the partition can change is the POOL ORDER
 * the kernel is handed -- redistribution assembles a body as kept entities
 * followed by arrivals in source-rank order. MFTLRemeshObject() puts the body in
 * global-id order before it sweeps, so that order cannot reach the result. The
 * spanning driver decides band edges against a different state and can only be
 * compared statistically (test_the_two_strategies_reach_the_same_mesh).
 *
 * The reference is the same four droplets on rank 0 alone, in an
 * MPI_COMM_SELF ensemble, through the serial path. Three passes, the last two
 * after a deformation, so the second and third pass start from a body the
 * closing redistribution has split again and the next centering reassembles.
 *
 * Non-vacuous by `count_split_bodies() > 0` at np > 1: bodies straddling a cut
 * plane arrive from two or more ranks. Mutation-checked: without the canonical
 * order the digests differ at np = 2, 4 and 8, and so do the counts after three
 * passes (678 / 674 / 664 triangles against 670).
 */
static int test_a_dispersed_remesh_is_the_serial_remesh_to_the_bit(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  TEST_ASSERT(ens != MFTL_NULL);
  add_four_droplets(ens);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  const int n_bodies = ens->n_bodies_global;
  if (g_nproc > 1) TEST_ASSERT(agreed(count_split_bodies(ens) > 0, MPI_COMM_WORLD));

  /* The serial reference, on rank 0 only. */
  MFTL_Ensemble *ser = MFTL_NULL;
  if (g_rank == 0)
  {
    int cuts[3] = {1, 1, 1};
    double lo[3] = {-1.0, -1.0, -1.0};
    double hi[3] = {1.0, 1.0, 1.0};
    bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
    ser = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, per, 4);
    add_four_droplets(ser);
  }

  MFTL_RemeshParameters params;
  standard_params(&params, 0.05);
  params.parallel_strategy = MFTL_REMESH_PARALLEL_DISPERSED;

  uint64_t *dist = (uint64_t *) calloc((size_t) n_bodies, sizeof(uint64_t));
  uint64_t *ref = (uint64_t *) calloc((size_t) n_bodies, sizeof(uint64_t));
  int all_equal = 1, all_dispersed = 1;
  double tri[2] = {0.0, 0.0};

  for (int pass = 0; pass < 3; ++pass)
  {
    if (pass > 0)
    {
      deform(ens);
      MFTLUpdateGhostPositions(ens);
      if (ser) deform(ser);
    }
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);

    MFTL_RemeshStats st;
    const int ok = MFTLRemeshEnsemble(ens, &params, &st) == MFTL_SUCCESS;
    all_dispersed &= (g_nproc == 1 || st.strategy == MFTL_REMESH_PARALLEL_DISPERSED);  // one rank takes the serial path
    if (ser)
    {
      for (int o = 0; o < ser->n_local_objects; ++o) MFTLComputeTriangleGeometry(ser->local_objects[o], ser->local_objects[o]->tris.n);
      if (MFTLRemeshEnsemble(ser, &params, MFTL_NULL) != MFTL_SUCCESS) all_equal = 0;
      body_digests(ser, n_bodies, ref, MPI_COMM_SELF);
    }
    TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

    body_digests(ens, n_bodies, dist, MPI_COMM_WORLD);
    MPI_Bcast(ref, n_bodies, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    for (int b = 0; b < n_bodies; ++b) all_equal &= (dist[b] == ref[b]);
  }

  mesh_measure m;
  measure(ens, &params, &m);
  tri[0] = m.n_triangles;
  if (ser)
    for (int o = 0; o < ser->n_local_objects; ++o) tri[1] += ser->local_objects[o]->tris.n;
  MPI_Bcast(&tri[1], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (g_rank == 0)
    printf("\n    np=%d: %d bodies, 3 passes, %.0f triangles distributed vs %.0f serial, digests %s\n    ", g_nproc, n_bodies, tri[0], tri[1],
           all_equal ? "equal" : "DIFFER");

  TEST_ASSERT(agreed(all_dispersed, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(tri[0] == tri[1], MPI_COMM_WORLD));
  TEST_ASSERT(agreed(all_equal, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  free(dist);
  free(ref);
  if (ser) MFTLEnsembleDestroy(&ser);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Small bodies on a fine grid stay DISPERSED instead of falling to spanning.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE FAILURE THIS PREVENTS IS A STRATEGY CHOICE MADE BY THE RANK COUNT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `r_max > 0.4 * h_min` sends the whole ensemble to the spanning driver. Raise
 * the rank count on a fixed domain and `h_min` shrinks, so droplets that have
 * not changed at all become "too large" -- and a spray takes the phase schedule,
 * which for it is two migrations against thirty-four.
 *
 * With the fallback armed, the yardstick becomes the coarsened box, and the
 * ensemble stays where it belongs.
 *
 * ── BOTH DIRECTIONS ARE ASSERTED, and that is the point ──────────────────
 *
 * A test that only showed DISPERSED with the fallback armed would pass on a
 * fixture where the fine test already said DISPERSED, i.e. while proving
 * nothing. So the same ensemble is measured with the fallback disarmed and must
 * come back SPANNING. The pair is the claim; neither half is.
 *
 * The relaxation is deliberately bounded -- it needs `N_bodies >= nproc`, or it
 * would take a jet that spanning would have shared and give it to one rank -- so
 * the fixture carries enough bodies to satisfy that at every rank count here.
 */
static int test_small_bodies_at_a_fine_grid_stay_dispersed(void)
{
  if (g_nproc < 4) return 0;  // needs a grid with 3+ cells on an axis; see below

  /* ── A SLAB GRID, BUILT HERE RATHER THAN TAKEN FROM THE SUITE ────────────
   *
   * The relaxation refuses to merge the WHOLE grid into one box -- that is the
   * gather, it hands every body to one rank, and it belongs to the spanning path
   * where a triangle count bounds it. The suite's cubic grids have exactly two
   * cells per axis up to np = 8, so doubling any axis consumes the entire grid
   * and the refusal is correct: there is no partial merge to test.
   *
   * `{nproc, 1, 1}` has nproc cells on x, so merging two of them leaves nproc/2
   * groups and the relaxation is a genuine partial one. It is also the honest
   * shape for this feature: a slab decomposition is what a domain gets when the
   * rank count outruns one axis, which is exactly the regime where h_min stops
   * describing the bodies and starts describing the rank count.
   */
  int cuts[3] = {g_nproc, 1, 1};
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 2);
  TEST_ASSERT(ens != MFTL_NULL);

  const double h_min = MFTLDomainGetHMin(&ens->domain);
  TEST_ASSERT(h_min > 0.0);

  /* Just above 0.4 * h_min, so the fine test says SPANNING and the SMALLEST
   * possible merge -- two boxes on one axis -- is enough to make it fit. Sizing
   * it to need more would test the same branch while giving up more ranks. */
  const double r = 0.45 * h_min;

  /* The relaxation's own precondition is N_bodies >= nproc: below it, dispersed
   * already leaves ranks idle and relaxing buys no parallelism, it only takes a
   * body the spanning schedule would have shared and gives it to one rank. */
  const int n_bodies = g_nproc;
  for (int b = 0; b < n_bodies; ++b)
  {
    /* One droplet per rank box, at its centre, so the fixture is balanced and a
     * failure here is unambiguously about the STRATEGY rather than the balancer. */
    const double cx = -1.0 + (2.0 * ((double) b + 0.5)) / (double) g_nproc;
    add_sphere(ens, b, cx, 0.0, 0.0, r, 1);
  }

  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  MFTL_RemeshParameters params;
  standard_params(&params, 0.5 * r);

  /* ── BOTH DIRECTIONS, and neither half is the claim on its own ───────────
   *
   * Armed-says-DISPERSED alone would pass on a fixture the fine test already
   * called dispersed, i.e. while proving nothing. The pair is what says the
   * yardstick moved. */
  params.parallel_coarsen = 0;
  TEST_ASSERT(agreed(MFTLRemeshChooseStrategy(ens, &params) == MFTL_REMESH_PARALLEL_SPANNING, MPI_COMM_WORLD));

  params.parallel_coarsen = 1;
  TEST_ASSERT(agreed(MFTLRemeshChooseStrategy(ens, &params) == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));

  /* And the driver AGREES with the chooser. If it did not, the admission test
   * would reject bodies the chooser accepted and the pass would fall back to the
   * spanning driver after a wasted measurement -- visible only as this field. */
  MFTL_RemeshStats stats;
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &params, &stats) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.strategy == MFTL_REMESH_PARALLEL_DISPERSED, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(stats.n_bodies_not_centred == 0, MPI_COMM_WORLD));

  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(count_split_bodies(ens) == 0, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("\n    np=%d grid %dx1x1: %d bodies of r=%.4f against h_min=%.4f (0.4h=%.4f) -> disarmed SPANNING, armed DISPERSED\n    ", g_nproc, g_nproc,
           n_bodies, r, h_min, 0.4 * h_min);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Dispersed distributed remeshing (np = %d) ===\n", g_nproc);

  RUN_TEST(test_the_strategy_follows_the_body_size);
  RUN_TEST(test_a_seam_straddling_droplet_is_not_measured_as_domain_sized);
  RUN_TEST(test_four_split_droplets_are_centred_and_remeshed);
  RUN_TEST(test_the_two_strategies_reach_the_same_mesh);
  RUN_TEST(test_a_spanning_body_forced_dispersed_falls_back_cleanly);
  RUN_TEST(test_the_public_entry_point_dispatches_to_dispersed);
  RUN_TEST(test_the_strategy_costs_one_measurement_not_two);
  RUN_TEST(test_the_balancer_evens_out_a_pile_up);
  RUN_TEST(test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold);
  RUN_TEST(test_the_balancer_is_deterministic);
  RUN_TEST(test_small_bodies_at_a_fine_grid_stay_dispersed);
  RUN_TEST(test_a_dispersed_remesh_is_the_serial_remesh_to_the_bit);

  if (g_rank == 0) TEST_REPORT();

  int local_failed = g_tests_failed;
  int total_failed = 0;
  MPI_Allreduce(&local_failed, &total_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failed > 0) ? 1 : 0;
}
