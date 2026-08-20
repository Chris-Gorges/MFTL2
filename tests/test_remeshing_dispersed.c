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

static void grid_for(int nproc, int cuts[3])
{
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
  if (nproc == 4)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 1;
  }
  else if (nproc == 8)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 2;
  }
}

/** @brief [-1,1]^3, so every grid above has h_min = 1.0 and the arithmetic is legible. */
static MFTL_Ensemble *make_ensemble(bool periodic)
{
  int cuts[3];
  grid_for(g_nproc, cuts);

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
   * docs/remeshing-quality-analysis.md. */
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

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];

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
  for (int o = 0; o < ens->num_objects; ++o) shares += (ens->objects[o]->tris.n_local > 0) ? 1 : 0;

  int total = 0;
  MPI_Allreduce(&shares, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return total - ens->num_objects;
}

/** @brief Puts the ensemble on the grid and gives it current geometry. */
static int partition(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) return 0;
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
  return 1;
}

static int ensemble_is_sound(const MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->num_objects; ++o)
  {
    if (!inv_partition_is_sound(ens->objects[o], g_rank)) return 0;
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
  const int nv = obj->next_vertex_gid;
  const int nt = obj->next_triangle_gid;
  if (nv < 0 || nt < 0) return 0;

  char *const mark = (char *) calloc((size_t) (nv + nt) + 2, 1);
  char *const got = (char *) calloc((size_t) (nv + nt) + 2, 1);

  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g < 0 || g >= nv)
    {
      free(mark);
      free(got);
      return 0;
    }
    if (mark[g])  // owned twice: two ranks claim the same id
    {
      free(mark);
      free(got);
      return 0;
    }
    mark[g] = 1;
  }
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g < 0 || g >= nt)
    {
      free(mark);
      free(got);
      return 0;
    }
    if (mark[nv + g])
    {
      free(mark);
      free(got);
      return 0;
    }
    mark[nv + g] = 1;
  }

  MPI_Allreduce(mark, got, nv + nt, MPI_CHAR, MPI_MAX, MPI_COMM_WORLD);

  int ok = 1;
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
  TEST_ASSERT(agreed(global_ids_are_dense(ens->objects[0]), MPI_COMM_WORLD));

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
  for (int o = 0; o < ens->num_objects; ++o)
  {
    TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(ens->objects[o]) == MFTL_SUCCESS, MPI_COMM_WORLD));
  }

  /* MFTL_FLAG_FROZEN is the spanning driver's transient state and this driver
   * never sets it. Asserting it anyway is what would catch a future refactor
   * that routed the dispersed path through the gated sweeps and left it behind. */
  int leaked = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
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

  /* ── Identity ──────────────────────────────────────────────────────────── */
  for (int o = 0; o < ens->num_objects; ++o)
  {
    TEST_ASSERT(agreed(global_ids_are_dense(ens->objects[o]), MPI_COMM_WORLD));
  }

  mesh_measure after;
  measure(ens, &params, &after);

  /* Euler's formula, summed over four closed genus-0 bodies. The one check that
   * sees a triangle lost between two ranks or counted twice: both leave a
   * perfectly valid local mesh everywhere and only the GLOBAL count is wrong. */
  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  TEST_ASSERT(agreed(fabs(chi - 2.0 * ens->num_objects) < 0.5, MPI_COMM_WORLD));

  double global_tris = 0.0;
  for (int o = 0; o < ens->num_objects; ++o) global_tris += (double) ens->objects[o]->global_n_triangles;
  TEST_ASSERT(agreed(fabs(after.n_triangles - global_tris) < 0.5, MPI_COMM_WORLD));

  /* ── Convergence ───────────────────────────────────────────────────────── */
  TEST_ASSERT(agreed(after.n_triangles > before.n_triangles, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.n_out_of_band < 0.02 * after.n_edges, MPI_COMM_WORLD));
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
  TEST_ASSERT(agreed(MFTLCheckMeshIntegrity(ens->objects[0]) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(global_ids_are_dense(ens->objects[0]), MPI_COMM_WORLD));
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

  if (g_rank == 0) TEST_REPORT();

  int local_failed = g_tests_failed;
  int total_failed = 0;
  MPI_Allreduce(&local_failed, &total_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failed > 0) ? 1 : 0;
}
