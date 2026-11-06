/**
 * @file test_remeshing_sizing_parallel.c
 * @brief Curvature sizing on a PARTITIONED mesh: the field on ghost copies,
 *        the spanning driver, and the two distributed strategies as each
 *        other's cross-check.
 *
 * Every result is compared against something that shares none of the
 * distribution: the sizing field against the same field computed on one rank
 * with MPI_COMM_SELF before partitioning (matched by vertex global id), the
 * distributed pass against MFTLRemeshObject() of the same body on one rank, and
 * the dispersed driver against the spanning one. Every fixture straddles the
 * cut planes of every grid the suite runs, and the straddling is MEASURED
 * (anywhere()): a distributed test whose curved region sits inside one rank
 * would pass with the ghost exchange deleted.
 *
 * Every assertion is made on an agreed() verdict (test_mesh_invariants.h).
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Fixtures
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief [-1,1]^3 on the suite's grid for this rank count. */
static MFTL_Ensemble *make_ensemble(MPI_Comm comm, bool periodic)
{
  int cuts[3] = {1, 1, 1};
  int np;
  MPI_Comm_size(comm, &np);
  test_grid_for(np, cuts);
  double lo[3] = {-1.0, -1.0, -1.0}, hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};
  return MFTLEnsembleCreate(comm, cuts, lo, hi, per, 2);
}

/** @brief An icosphere scaled to semi-axes (a, b, c) about (cx, cy, cz); optionally only its upper half (an open cap). */
static void add_ellipsoid(MFTL_Ensemble *ens, double cx, double cy, double cz, double a, double b, double c, int refine, bool cap)
{
  /* Built on rank 0 of @p ens only, like every generator (partition() then
   * publishes and spreads it), and by the same arithmetic on every rank's
   * MPI_COMM_SELF reference -- so the two have the same vertex global ids. */
  if (ens->domain.rank != 0)
  {
    MFTL_Object *empty = MFTLObjectCreate(ens->n_bodies_global);
    MFTLEnsembleAddObject(ens, empty);
    MFTLObjectBuildDerived(empty);
    return;
  }
  MFTL_Ensemble *tmp_ens = make_ensemble(MPI_COMM_SELF, MFTL_FALSE);
  MFTL_Object *tmp = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(tmp_ens, tmp);
  double origin[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(tmp, origin, 1.0, refine);

  const int nv = tmp->verts.n, nt = tmp->tris.n;
  double *pos = malloc((size_t) nv * 3 * sizeof(double));
  int *tri = malloc((size_t) nt * 3 * sizeof(int));
  int *map = malloc((size_t) nv * sizeof(int));
  for (int v = 0; v < nv; ++v) map[v] = -1;

  int kt = 0, kv = 0;
  for (int t = 0; t < nt; ++t)
  {
    int w[3];
    MFTLGetTriangleVertices(tmp, t, w);
    if (cap)
    {
      double zc = 0.0;
      for (int k = 0; k < 3; ++k) zc += MFTLVertexPos(tmp, w[k])[2];
      if (zc < 0.0) continue;
    }
    for (int k = 0; k < 3; ++k)
    {
      if (map[w[k]] < 0)
      {
        const double *p = MFTLVertexPos(tmp, w[k]);
        pos[3 * kv + 0] = cx + a * p[0];
        pos[3 * kv + 1] = cy + b * p[1];
        pos[3 * kv + 2] = cz + c * p[2];
        map[w[k]] = kv++;
      }
      tri[3 * kt + k] = map[w[k]];
    }
    kt++;
  }

  MFTL_Object *obj = MFTLObjectCreate(ens->n_bodies_global);
  MFTLEnsembleAddObject(ens, obj);
  MFTLBuildIndexedMesh(obj, pos, kv, tri, kt);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  free(pos);
  free(tri);
  free(map);
  MFTLEnsembleDestroy(&tmp_ens);
}

static int partition(MFTL_Ensemble *ens)
{
  MFTLPublishGeneratedBodies(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) return 0;
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  return 1;
}

static int ensemble_is_sound(const MFTL_Ensemble *ens)
{
  if (!inv_local_directory_is_sound(ens)) return 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (!inv_partition_is_sound(ens->local_objects[o], g_rank)) return 0;
  return 1;
}

static void curvature_params(MFTL_RemeshParameters *p, double lmax)
{
  MFTLInitRemeshParams(p, MFTL_REMESH_ISOTROPIC, lmax, 4, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p->smooth_iterations = 3;
  p->smooth_relaxation = 0.5;
  MFTLRemeshUseCurvatureSizing(p, 0.01 * lmax, 0.3 * lmax, lmax);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Measurements
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The undirected edge at @p h is counted by exactly one rank. */
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
  double n_vertices, n_triangles, n_edges, n_boundary;
  double volume;  ///< Of a closed body about its own centre (the fixtures are not periodic).
  double tip_sum, tip_n, waist_sum, waist_n;  ///< Edge lengths with |x - cx| above / below fractions of a.
} measure_t;

/** @brief Global counts, the enclosed volume and the edge lengths at the tips and the waist of an x-elongated body at @p cx. */
static void measure(const MFTL_Ensemble *ens, double cx, double a, measure_t *m)
{
  double s[10] = {0};
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    s[0] += obj->verts.n_local;
    s[1] += obj->tris.n_local;
    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!counts_this_edge(obj, h)) continue;
      s[2] += 1.0;
      if (obj->tris.he[h].twin_idx == MFTL_ERROR) s[3] += 1.0;
      const double *p = MFTLVertexPos(obj, MFTLHeOrigin(obj, h)), *q = MFTLVertexPos(obj, MFTLHeTarget(obj, h));
      const double x = fabs(0.5 * (p[0] + q[0]) - cx) / a;
      const double len = MFTLGetEdgeLength(obj, h);
      if (x > 0.85) s[6] += len, s[7] += 1.0;
      if (x < 0.2) s[8] += len, s[9] += 1.0;
    }
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      const double *A = MFTLVertexPos(obj, v[0]), *B = MFTLVertexPos(obj, v[1]), *C = MFTLVertexPos(obj, v[2]);
      s[4] += (A[0] * (B[1] * C[2] - B[2] * C[1]) - A[1] * (B[0] * C[2] - B[2] * C[0]) + A[2] * (B[0] * C[1] - B[1] * C[0])) / 6.0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, s, 10, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  m->n_vertices = s[0];
  m->n_triangles = s[1];
  m->n_edges = s[2];
  m->n_boundary = s[3];
  m->volume = s[4];
  m->tip_sum = s[6];
  m->tip_n = s[7];
  m->waist_sum = s[8];
  m->waist_n = s[9];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The sizing field on ghost copies
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Every copy of every vertex holds the target the unpartitioned mesh
 *        gives it -- after MFTLRemeshEnsemblePrepareSizing(), and NOT after the
 *        local half alone (the non-vacuity measure: the exchange is doing
 *        something).
 *
 * The reference is the same body built and sized on each rank alone, matched by
 * vertex global id (both are built by the same generator, and a redistribution
 * keeps ids). The tolerance is relative 1e-12, not equality: a one-ring sum is
 * the same set of terms on both sides but accumulated in a different triangle
 * order. The open cap exercises the second exchange: its rim vertices average
 * their interior neighbours, which near a cut plane are ghosts.
 */
static int ghost_case(bool cap)
{
  MFTL_RemeshParameters p;
  curvature_params(&p, 0.1);

  /* Reference: the whole body on this rank. */
  MFTL_Ensemble *ref_ens = make_ensemble(MPI_COMM_SELF, MFTL_FALSE);
  add_ellipsoid(ref_ens, 0.0, 0.0, 0.0, 0.7, 0.35, 0.35, 3, cap);
  MFTL_Object *ref = ref_ens->local_objects[0];
  TEST_ASSERT(agreed(MFTLRemeshPrepareSizing(ref, &p) == MFTL_SUCCESS, MPI_COMM_WORLD));
  const int n_ref = ref->next_vertex_gid;
  double *L_ref = malloc((size_t) n_ref * sizeof(double));
  for (int v = 0; v < ref->verts.n; ++v) L_ref[ref->verts.rec[v].global_id] = ref->verts.sizing[v];

  MFTL_Ensemble *ens = make_ensemble(MPI_COMM_WORLD, MFTL_FALSE);
  add_ellipsoid(ens, 0.0, 0.0, 0.0, 0.7, 0.35, 0.35, 3, cap);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

  /* The local half: owned vertices right, ghosts at the halo rim wrong. */
  int wrong_local = 0, sized_cross = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    MFTLRemeshPrepareSizing(obj, &p);
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      const double r = L_ref[obj->verts.rec[v].global_id];
      if (fabs(obj->verts.sizing[v] - r) > 1e-9 * r) wrong_local++;
    }
  }

  TEST_ASSERT(agreed(MFTLRemeshEnsemblePrepareSizing(ens, &p) == MFTL_SUCCESS, MPI_COMM_WORLD));

  int bad = 0, n_checked = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const double r = L_ref[obj->verts.rec[v].global_id];
      if (!(fabs(obj->verts.sizing[v] - r) <= 1e-12 * r)) bad++;
      n_checked++;
    }
    /* A cut edge whose end is sized below the coarsest target: the curved part
     * of the body really is shared between ranks. */
    for (int t = 0; t < obj->tris.n_local; ++t)
      for (int k = 0; k < 3; ++k)
      {
        const int tw = obj->tris.he[3 * t + k].twin_idx;
        if (tw == MFTL_ERROR || MFTLHeTriIdx(tw) < obj->tris.n_local) continue;
        if (obj->verts.sizing[MFTLHeTarget(obj, 3 * t + k)] < 0.95 * p.target_length) sized_cross++;
      }
  }

  TEST_ASSERT(agreed(bad == 0, MPI_COMM_WORLD));
  if (g_nproc > 1)
  {
    TEST_ASSERT(anywhere(wrong_local > 0, MPI_COMM_WORLD));
    TEST_ASSERT(anywhere(sized_cross > 0, MPI_COMM_WORLD));
  }
  int g[3] = {wrong_local, sized_cross, n_checked};
  MPI_Allreduce(MPI_IN_PLACE, g, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0)
    printf("\n    np=%d %s: %d vertex copies checked; %d ghosts wrong before the exchange; %d sized cut edges", g_nproc, cap ? "cap" : "closed", g[2],
           g[0], g[1]);

  free(L_ref);
  MFTLEnsembleDestroy(&ens);
  MFTLEnsembleDestroy(&ref_ens);
  return 0;
}

static int test_every_ghost_holds_its_owners_target(void) { return ghost_case(MFTL_FALSE); }
static int test_rim_targets_average_their_neighbours_across_ranks(void) { return ghost_case(MFTL_TRUE); }

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The spanning driver, against one rank
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief An elongated ellipsoid across every cut plane, remeshed by the spanning
 *        driver, and the same body remeshed by MFTLRemeshObject() on one rank.
 *
 * Asserted: a sound partition, Euler 2, the convergence certificate, volume
 * held, and the RESULT of the sizing -- the triangle count and the
 * tip-to-waist edge-length ratio -- within a few per cent of the serial answer. Split and collapse are order-dependent, so the two runs land on
 * different triangulations of the same surface and equality would be wrong
 * (cf. test_the_two_strategies_reach_the_same_mesh).
 */
static int test_the_spanning_driver_under_curvature_sizing(void)
{
  const double a = 0.7, b = 0.3;
  MFTL_RemeshParameters p;
  curvature_params(&p, 0.08);
  p.parallel_strategy = MFTL_REMESH_PARALLEL_SPANNING;

  /* One rank, the reference -- remeshed to its FIXED POINT, not once. The
   * curvature field is recomputed by every call (and by every phase of the
   * spanning driver) from the mesh as it then is, and the first call reads it
   * off the coarse input, which underestimates the curvature at the tips; a
   * phased pass sees finer meshes and grades more strongly. Both settle on the
   * same self-consistent field, which is where a timestep loop lives. */
  MFTL_Ensemble *ref_ens = make_ensemble(MPI_COMM_SELF, MFTL_FALSE);
  add_ellipsoid(ref_ens, 0.0, 0.0, 0.0, a, b, b, 3, MFTL_FALSE);
  MFTL_RemeshStats rs;
  int calls = 0;
  do
  {
    TEST_ASSERT(agreed(MFTLRemeshObject(ref_ens->local_objects[0], &p, &rs) == MFTL_SUCCESS, MPI_COMM_WORLD));
  } while (++calls < 12 && rs.n_split + rs.n_collapse + rs.n_flip > 0);
  const MFTL_Object *ref = ref_ens->local_objects[0];
  double ref_tip = 0, ref_tn = 0, ref_waist = 0, ref_wn = 0;
  for (int h = 0; h < 3 * ref->tris.n; ++h)
  {
    if (ref->tris.he[h].twin_idx < h) continue;
    const double *P = MFTLVertexPos(ref, MFTLHeOrigin(ref, h)), *Q = MFTLVertexPos(ref, MFTLHeTarget(ref, h));
    const double x = fabs(0.5 * (P[0] + Q[0])) / a;
    if (x > 0.85) ref_tip += MFTLGetEdgeLength(ref, h), ref_tn += 1.0;
    if (x < 0.2) ref_waist += MFTLGetEdgeLength(ref, h), ref_wn += 1.0;
  }
  const double ref_ratio = (ref_tip / ref_tn) / (ref_waist / ref_wn);
  const double ref_tris = ref->tris.n;

  /* Distributed. */
  MFTL_Ensemble *ens = make_ensemble(MPI_COMM_WORLD, MFTL_FALSE);
  add_ellipsoid(ens, 0.0, 0.0, 0.0, a, b, b, 3, MFTL_FALSE);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  measure_t before, after;
  measure(ens, 0.0, a, &before);

  int cut = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) cut += ens->local_objects[o]->tris.n - ens->local_objects[o]->tris.n_local;
  if (g_nproc > 1) TEST_ASSERT(anywhere(cut > 0, MPI_COMM_WORLD));

  /* To its fixed point as well: the operation counts are rank-local, so the
   * stopping test is their reduced sum, which every rank sees alike. */
  MFTL_RemeshStats st;
  int d_calls = 0;
  for (;;)
  {
    TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &p, &st) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(st.converged, MPI_COMM_WORLD));
    int ops = st.n_split + st.n_collapse + st.n_flip;
    MPI_Allreduce(MPI_IN_PLACE, &ops, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (++d_calls >= 12 || ops == 0) break;
  }
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  measure(ens, 0.0, a, &after);

  const double chi = after.n_vertices - after.n_edges + after.n_triangles;
  const double ratio = (after.tip_sum / after.tip_n) / (after.waist_sum / after.waist_n);
  const double dn = fabs(after.n_triangles - ref_tris) / ref_tris;
  if (g_rank == 0)
    printf("\n    np=%d: %.0f -> %.0f tris after %d calls (one rank %.0f after %d, %.1f%%), tip/waist %.3f (one rank %.3f), last pass %d phases, dV %.2f%%",
           g_nproc, before.n_triangles, after.n_triangles, d_calls, ref_tris, calls, 100.0 * dn, ratio, ref_ratio, st.n_phases,
           100.0 * (after.volume / before.volume - 1.0));

  TEST_ASSERT(agreed(fabs(chi - 2.0) < 0.5, MPI_COMM_WORLD));
  /* Curvature: +1.3 / 3.9 / 6.0 / 4.0 % over the one-rank fixed point at
   * np = 2 / 4 / 8 / 27 -- bounded, not growing. A redistribution does not
   * carry the field, so the spanning driver recomputes it after every migration
   * (up to 18 times in the first pass at np = 8) on a mesh that is part
   * remeshed, and each recomputation is a chance to split at a dip that the
   * hysteresis of the band then keeps. */
  TEST_ASSERT(agreed(dn < 0.08, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(after.volume / before.volume - 1.0) < 0.02, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(ratio - ref_ratio) < 0.08 * ref_ratio, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  MFTLEnsembleDestroy(&ref_ens);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The two strategies, each other's cross-check
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Four elongated droplets, each on a cut plane, remeshed by the
 *        dispersed and by the spanning driver. The two share no code past the
 *        kernel; agreement within 4 % in triangles and 1 % in volume is the
 *        tolerance of the uniform cross-check, for the same reason.
 */
static int test_the_two_strategies_agree_under_curvature_sizing(void)
{
  MFTL_RemeshParameters p;
  curvature_params(&p, 0.04);

  measure_t result[2];
  for (int which = 0; which < 2; ++which)
  {
    MFTL_Ensemble *ens = make_ensemble(MPI_COMM_WORLD, MFTL_TRUE);
    add_ellipsoid(ens, 0.00, 0.00, 0.00, 0.16, 0.08, 0.08, 2, MFTL_FALSE);
    add_ellipsoid(ens, 0.00, 0.50, 0.50, 0.08, 0.14, 0.08, 2, MFTL_FALSE);
    add_ellipsoid(ens, -0.50, 0.00, 0.50, 0.08, 0.08, 0.13, 2, MFTL_FALSE);
    add_ellipsoid(ens, 0.50, 0.50, 0.00, 0.12, 0.07, 0.07, 2, MFTL_FALSE);
    TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));

    MFTL_RemeshStats st;
    const int rc = (which == 0) ? MFTLRemeshEnsembleDispersed(ens, &p, &st) : MFTLRemeshEnsembleParallel(ens, &p, &st);
    TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(st.converged, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
    /* Not periodic-safe in general, but every droplet sits inside the box. */
    measure(ens, 0.0, 1.0, &result[which]);
    MFTLEnsembleDestroy(&ens);
  }
  const double dn = fabs(result[0].n_triangles - result[1].n_triangles) / result[1].n_triangles;
  const double dv = fabs(result[0].volume - result[1].volume) / fabs(result[1].volume);
  if (g_rank == 0)
    printf("\n    np=%d: dispersed %.0f / spanning %.0f tris (%.1f%%), volume %.1e", g_nproc, result[0].n_triangles, result[1].n_triangles, 100.0 * dn, dv);
  TEST_ASSERT(agreed(dn < 0.04, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(dv < 0.01, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief A block valid on some ranks and not on others is refused on every
 *        rank, before anything is touched.
 */
static int test_an_invalid_sizing_is_refused_on_every_rank(void)
{
  MFTL_Ensemble *ens = make_ensemble(MPI_COMM_WORLD, MFTL_FALSE);
  add_ellipsoid(ens, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2, MFTL_FALSE);
  TEST_ASSERT(agreed(partition(ens), MPI_COMM_WORLD));
  MFTL_RemeshParameters p;
  curvature_params(&p, 0.1);
  if (g_rank == g_nproc - 1) p.curvature_tolerance = 0.0;
  MFTL_RemeshStats st;
  TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &p, &st) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Sizing on a partitioned mesh (np = %d) ===\n", g_nproc);
  RUN_TEST(test_every_ghost_holds_its_owners_target);
  RUN_TEST(test_rim_targets_average_their_neighbours_across_ranks);
  RUN_TEST(test_the_spanning_driver_under_curvature_sizing);
  RUN_TEST(test_the_two_strategies_agree_under_curvature_sizing);
  RUN_TEST(test_an_invalid_sizing_is_refused_on_every_rank);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
