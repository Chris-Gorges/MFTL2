/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for the derived per-triangle arrays and the Morton reorder.
 *
 * These two are what replaced the previous version's "flatten" module, and the
 * properties worth pinning are the ones that module got wrong or made hard:
 *
 *   - the derived arrays say EXACTLY what a half-edge walk says (they are an
 *     optimisation, and the moment they disagree they are a second, wrong mesh);
 *   - the frames are seam-consistent for an object straddling a periodic
 *     boundary, with no mode flag and no fallback path for the consumer;
 *   - staleness is tracked, so a loop that moves vertices cannot silently leave
 *     the geometry one timestep behind;
 *   - the Morton reorder permutes AoS and SoA together and is a pure
 *     relabelling: the mesh afterwards is the same mesh.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;  ///< non-periodic
static MFTL_Ensemble *g_per = MFTL_NULL;  ///< periodic in x, domain [0,1]^3

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

static MFTL_Object *make_sphere(MFTL_Ensemble *ens, int refinements)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 1.0, refinements);
  return obj;
}

/** @brief One triangle deliberately straddling the periodic x boundary. */
static MFTL_Object *make_seam_triangle(void)
{
  static const double pos[9] = {0.95, 0.50, 0.50, 0.05, 0.50, 0.50, 0.00, 0.62, 0.50};
  static const int tris[3] = {0, 1, 2};

  MFTL_Object *obj = MFTLObjectCreate(7);
  MFTLEnsembleAddObject(g_per, obj);
  MFTLBuildIndexedMesh(obj, pos, 3, tris, 1);
  return obj;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Correctness of the derived arrays
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_derived_arrays_agree_with_the_half_edges(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 3);

  TEST_ASSERT(MFTLObjectHasTopology(obj));
  TEST_ASSERT(MFTLObjectHasFrames(obj));
  TEST_ASSERT(inv_derived_matches_topology(obj));
  TEST_ASSERT(inv_frames_match_positions(obj, 1e-12));

  /* tris.neighbors is documented as twin/3; check that literally, since the
   * whole justification for storing it is that reading it beats recomputing it. */
  for (int t = 0; t < obj->tris.n; ++t)
  {
    for (int k = 0; k < 3; ++k)
    {
      const int tw = obj->tris.he[3 * t + k].twin_idx;
      const int expect = (tw == MFTL_ERROR) ? MFTL_ERROR : tw / 3;
      TEST_ASSERT_EQUAL_INT(expect, obj->tris.neighbors[3 * t + k]);
    }
  }

  // Neighbourhood is symmetric: if t sees u, u sees t back.
  for (int t = 0; t < obj->tris.n; ++t)
  {
    for (int k = 0; k < 3; ++k)
    {
      const int u = obj->tris.neighbors[3 * t + k];
      if (u == MFTL_ERROR) continue;
      int found = 0;
      for (int j = 0; j < 3; ++j)
      {
        if (obj->tris.neighbors[3 * u + j] == t) found = 1;
      }
      TEST_ASSERT(found);
    }
  }
  return 0;
}

static int test_frames_reproduce_area_and_normal(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 2);

  /* The streaming path and the scalar path must agree to round-off. If they ever
   * do not, one of the two is what every consumer is silently using. */
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double area_ref;
    double normal_ref[3];
    MFTLGetTriangleProperties(obj, t, &area_ref, normal_ref, MFTL_NULL);

    const double *e = &obj->tris.edge[6 * t];
    const double nx = e[1] * e[5] - e[2] * e[4];
    const double ny = e[2] * e[3] - e[0] * e[5];
    const double nz = e[0] * e[4] - e[1] * e[3];
    const double mag = sqrt(nx * nx + ny * ny + nz * nz);

    TEST_ASSERT_EQUAL_DOUBLE(area_ref, 0.5 * mag, 1e-14);
    TEST_ASSERT_EQUAL_DOUBLE(normal_ref[0], nx / mag, 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE(normal_ref[1], ny / mag, 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE(normal_ref[2], nz / mag, 1e-12);
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Periodic seam
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_frames_are_seam_consistent(void)
{
  MFTL_Object *obj = make_seam_triangle();

  /* The naive difference across the seam is -0.90; the true edge is +0.10. The
   * previous version solved this by unwrapping all POSITIONS, which only worked
   * while the object fitted in half a domain and needed a flag telling consumers
   * whether it had. The edge vectors carry the correction instead, which is
   * exact for every object of any size, because a single triangle always is. */
  const double *e = &obj->tris.edge[0];
  TEST_ASSERT_EQUAL_DOUBLE(0.10, e[0], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.00, e[1], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.00, e[2], 1e-12);

  TEST_ASSERT_EQUAL_DOUBLE(0.05, e[3], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.12, e[4], 1e-12);

  // The anchor stays the STORED (wrapped) position -- it is a position, and
  // positions are what a rank lookup expects.
  TEST_ASSERT_EQUAL_DOUBLE(0.95, obj->tris.anchor[0], 1e-15);

  /* The area a consumer computes from the frame is the small true triangle, not
   * the domain-spanning sliver a naive difference would produce. */
  const double area = MFTLGetTriangleArea(obj, 0);
  TEST_ASSERT(area < 0.01);
  TEST_ASSERT(area > 0.0);

  TEST_ASSERT(inv_frames_match_positions(obj, 1e-14));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Staleness tracking
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_moving_vertices_invalidates_the_frames(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 1);
  TEST_ASSERT(MFTLObjectHasFrames(obj));

  /* The motion iterator clears the frame bit on the way out, so a subsequent
   * frame loop asserts in Debug. The previous version left this to the caller's
   * memory, and a forgotten MFTLSyncFlatPositions produced geometry one timestep
   * stale -- silently, and only where the mesh had actually moved. */
  MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += 0.25;);

  TEST_ASSERT(!MFTLObjectHasFrames(obj));
  TEST_ASSERT(MFTLObjectHasTopology(obj));  // topology did NOT change

  MFTLObjectRefreshFrames(obj);
  TEST_ASSERT(MFTLObjectHasFrames(obj));
  TEST_ASSERT(inv_frames_match_positions(obj, 1e-12));

  // The refresh must have picked the motion up, not merely re-set the flag.
  TEST_ASSERT_EQUAL_DOUBLE(obj->verts.pos[3 * obj->tris.verts[0] + 0], obj->tris.anchor[0], 1e-15);
  return 0;
}

static int test_read_only_vertex_loop_keeps_the_frames(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 1);

  double sum = 0.0;
  MFTL_FOREACH_VERTEX_POS(obj, obj->verts.n, sum += px * px + py * py + pz * pz;);

  // A unit sphere: every vertex is at radius 1.
  TEST_ASSERT_EQUAL_DOUBLE((double) obj->verts.n, sum, 1e-9);

  // Reading positions changes nothing, so the frames stay valid.
  TEST_ASSERT(MFTLObjectHasFrames(obj));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Morton reorder
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief A relabelling-invariant signature of the mesh.
 *
 * Each triangle becomes its three vertex GLOBAL IDs, sorted; the list of those
 * triples is then sorted. Two meshes with the same signature are the same mesh
 * up to the numbering of the pools, which is exactly what a reorder is allowed
 * to change.
 */
static int signature_cmp(const void *a, const void *b)
{
  const int *x = (const int *) a;
  const int *y = (const int *) b;
  for (int k = 0; k < 3; ++k)
  {
    if (x[k] != y[k]) return (x[k] < y[k]) ? -1 : 1;
  }
  return 0;
}

/** @brief Plain int comparator. NOT signature_cmp, which reads three per element. */
static int int_cmp(const void *a, const void *b)
{
  const int x = *(const int *) a;
  const int y = *(const int *) b;
  return (x > y) - (x < y);
}

static int *mesh_signature(const MFTL_Object *obj)
{
  const int nt = obj->tris.n;
  int *sig = (int *) malloc(3 * (size_t) nt * sizeof(int));
  if (!sig) return MFTL_NULL;

  for (int t = 0; t < nt; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    for (int k = 0; k < 3; ++k) sig[3 * t + k] = obj->verts.rec[v[k]].global_id;
    // sort the triple
    for (int i = 0; i < 3; ++i)
    {
      for (int j = i + 1; j < 3; ++j)
      {
        if (sig[3 * t + j] < sig[3 * t + i])
        {
          const int tmp = sig[3 * t + i];
          sig[3 * t + i] = sig[3 * t + j];
          sig[3 * t + j] = tmp;
        }
      }
    }
  }
  qsort(sig, (size_t) nt, 3 * sizeof(int), signature_cmp);
  return sig;
}

static int test_reorder_is_a_pure_relabelling(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 3);
  const int nv = obj->verts.n;
  const int nt = obj->tris.n;

  // Snapshot: position per vertex GLOBAL id, plus the connectivity signature.
  double *pos_by_gid = (double *) malloc(3 * (size_t) nv * sizeof(double));
  TEST_ASSERT(pos_by_gid != MFTL_NULL);
  for (int v = 0; v < nv; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    for (int k = 0; k < 3; ++k) pos_by_gid[3 * g + k] = obj->verts.pos[3 * v + k];
  }
  int *sig_before = mesh_signature(obj);
  TEST_ASSERT(sig_before != MFTL_NULL);

  MFTLObjectReorderMorton(obj);

  // Same entity counts, still a sound mesh.
  TEST_ASSERT_EQUAL_INT(nv, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(nt, obj->tris.n);
  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  // Every vertex still carries its own position: the AoS record and the SoA
  // position moved together. This is the assertion that would fail if the
  // permutation were applied to one and not the other.
  for (int v = 0; v < nv; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    for (int k = 0; k < 3; ++k) TEST_ASSERT_EQUAL_DOUBLE(pos_by_gid[3 * g + k], obj->verts.pos[3 * v + k], 0.0);
  }

  // ...and the connectivity is the same mesh.
  int *sig_after = mesh_signature(obj);
  TEST_ASSERT(sig_after != MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(0, memcmp(sig_before, sig_after, 3 * (size_t) nt * sizeof(int)));

  free(sig_before);
  free(sig_after);
  free(pos_by_gid);
  return 0;
}

static int test_reorder_produces_monotone_morton_keys(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 3);
  MFTLObjectReorderMorton(obj);

  const int nv = obj->verts.n;

  // Recompute the keys the reorder used: object bounding box, one shared scale.
  double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int v = 0; v < nv; ++v)
  {
    for (int k = 0; k < 3; ++k)
    {
      const double c = obj->verts.pos[3 * v + k];
      if (c < lo[k]) lo[k] = c;
      if (c > hi[k]) hi[k] = c;
    }
  }
  double range = hi[0] - lo[0];
  if (hi[1] - lo[1] > range) range = hi[1] - lo[1];
  if (hi[2] - lo[2] > range) range = hi[2] - lo[2];
  const double inv_range = MFTL_MORTON_MAX / range;

  uint64_t prev = 0;
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const uint64_t ix = (uint64_t) ((obj->verts.pos[3 * v + 0] - lo[0]) * inv_range);
    const uint64_t iy = (uint64_t) ((obj->verts.pos[3 * v + 1] - lo[1]) * inv_range);
    const uint64_t iz = (uint64_t) ((obj->verts.pos[3 * v + 2] - lo[2]) * inv_range);
    const uint64_t key = MFTLMorton3D(ix, iy, iz);
    TEST_ASSERT(key >= prev);  // the pool IS the space-filling curve
    prev = key;
  }
  return 0;
}

static int test_reorder_improves_edge_locality(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 4);

  /* Mean |index difference| across every edge. This is the quantity the reorder
   * exists to reduce: two vertices joined by an edge are close in space, so
   * after a Morton renumbering they should be close in memory too. */
  double before = 0.0;
  double after = 0.0;
  const int n_he = MFTLNumHalfEdges(obj);

  for (int h = 0; h < n_he; ++h) before += fabs((double) (MFTLHeTarget(obj, h) - MFTLHeOrigin(obj, h)));

  MFTLObjectReorderMorton(obj);

  for (int h = 0; h < n_he; ++h) after += fabs((double) (MFTLHeTarget(obj, h) - MFTLHeOrigin(obj, h)));

  printf("\n  mean edge index distance: %.1f -> %.1f  ", before / n_he, after / n_he);
  TEST_ASSERT(after < before);
  return 0;
}

static int test_reorder_preserves_the_ownership_prefix(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 2);
  const int nv = obj->verts.n;
  const int nt = obj->tris.n;

  /* Declare the second half of each pool to be ghosts. The reorder must sort the
   * owned and ghost segments SEPARATELY -- sorting the whole pool at once would
   * interleave them and destroy the prefix invariant that every ownership test
   * in the library relies on. */
  const int nv_local = nv / 2;
  const int nt_local = nt / 2;

  int *owned_gids = (int *) malloc((size_t) nv_local * sizeof(int));
  TEST_ASSERT(owned_gids != MFTL_NULL);
  for (int v = 0; v < nv_local; ++v) owned_gids[v] = obj->verts.rec[v].global_id;
  qsort(owned_gids, (size_t) nv_local, sizeof(int), int_cmp);

  obj->verts.n_local = nv_local;
  obj->tris.n_local = nt_local;
  for (int v = nv_local; v < nv; ++v) obj->verts.rec[v].owner_proc = 12345;
  for (int t = nt_local; t < nt; ++t) obj->tris.rec[t].owner_proc = 12345;

  MFTLObjectReorderMorton(obj);

  TEST_ASSERT_EQUAL_INT(nv_local, obj->verts.n_local);
  TEST_ASSERT_EQUAL_INT(nt_local, obj->tris.n_local);

  // The SAME set of entities is still in the owned prefix, just reordered.
  int *after_gids = (int *) malloc((size_t) nv_local * sizeof(int));
  TEST_ASSERT(after_gids != MFTL_NULL);
  for (int v = 0; v < nv_local; ++v)
  {
    after_gids[v] = obj->verts.rec[v].global_id;
    TEST_ASSERT(obj->verts.rec[v].owner_proc != 12345);
  }
  qsort(after_gids, (size_t) nv_local, sizeof(int), int_cmp);
  TEST_ASSERT_EQUAL_INT(0, memcmp(owned_gids, after_gids, (size_t) nv_local * sizeof(int)));

  for (int v = nv_local; v < nv; ++v) TEST_ASSERT_EQUAL_INT(12345, obj->verts.rec[v].owner_proc);
  for (int t = nt_local; t < nt; ++t) TEST_ASSERT_EQUAL_INT(12345, obj->tris.rec[t].owner_proc);

  // Topology survived intact despite the two segments being permuted separately.
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));
  TEST_ASSERT(inv_derived_matches_topology(obj));

  free(owned_gids);
  free(after_gids);
  return 0;
}

static int test_reorder_carries_cached_geometry_along(void)
{
  MFTL_Object *obj = make_sphere(g_ens, 2);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  const int nt = obj->tris.n;
  double *area_by_gid = (double *) malloc((size_t) nt * sizeof(double));
  TEST_ASSERT(area_by_gid != MFTL_NULL);
  for (int t = 0; t < nt; ++t) area_by_gid[obj->tris.rec[t].global_id] = obj->tris.area[t];

  MFTLObjectReorderMorton(obj);

  /* A reorder must not silently discard geometry the caller computed one line
   * earlier -- which is why the cached arrays are permuted rather than dropped. */
  for (int t = 0; t < nt; ++t) TEST_ASSERT_EQUAL_DOUBLE(area_by_gid[obj->tris.rec[t].global_id], obj->tris.area[t], 0.0);

  free(area_by_gid);
  return 0;
}

static int test_reorder_of_a_trivial_mesh_is_safe(void)
{
  // Empty object: nothing to order, and nothing may crash.
  MFTL_Object *empty = MFTLObjectCreate(99);
  MFTLEnsembleAddObject(g_ens, empty);
  MFTLObjectReorderMorton(empty);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(empty));

  // Single triangle: below the sort threshold in both pools.
  MFTL_Object *tri = MFTLObjectCreate(98);
  MFTLEnsembleAddObject(g_ens, tri);
  const double pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const int tv[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(tri, pos, 3, tv, 1);
  MFTLObjectReorderMorton(tri);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(tri));
  TEST_ASSERT(inv_derived_matches_topology(tri));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 64);

  double plo[3] = {0.0, 0.0, 0.0};
  double phi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {true, false, false};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 4);

  printf("--- Running MFTL Derived-Array Tests ---\n");

  RUN_TEST(test_derived_arrays_agree_with_the_half_edges);
  RUN_TEST(test_frames_reproduce_area_and_normal);
  RUN_TEST(test_frames_are_seam_consistent);
  RUN_TEST(test_moving_vertices_invalidates_the_frames);
  RUN_TEST(test_read_only_vertex_loop_keeps_the_frames);
  RUN_TEST(test_reorder_is_a_pure_relabelling);
  RUN_TEST(test_reorder_produces_monotone_morton_keys);
  RUN_TEST(test_reorder_improves_edge_locality);
  RUN_TEST(test_reorder_preserves_the_ownership_prefix);
  RUN_TEST(test_reorder_carries_cached_geometry_along);
  RUN_TEST(test_reorder_of_a_trivial_mesh_is_safe);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
