/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for mesh generation.
 *
 * The icosphere is the fixture the whole library is exercised on, so its
 * invariants are worth pinning precisely -- especially the winding, which is
 * silent when wrong: a mesh with inverted normals is perfectly valid
 * topologically and produces a negative enclosed volume, a flipped surface
 * tension and an inverted indicator function.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

/* ─────────────────────────────────────────────────────────────────────────────
 * Helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static MFTL_Object *make_sphere(double cx, double cy, double cz, double r, int refinements)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, obj);
  double c[3] = {cx, cy, cz};
  MFTLGenerateSphere(obj, c, r, refinements);
  return obj;
}

/** @brief Total surface area, from the derived frames. */
static double total_area(MFTL_Object *obj)
{
  double sum = 0.0;
  MFTL_FOREACH_TRIANGLE_EDGE(obj, obj->tris.n,
                             const double nx = e1y * e2z - e1z * e2y;
                             const double ny = e1z * e2x - e1x * e2z;
                             const double nz = e1x * e2y - e1y * e2x;
                             sum += 0.5 * sqrt(nx * nx + ny * ny + nz * nz););
  return sum;
}

/**
 * @brief Enclosed volume by the divergence theorem: V = (1/3) sum A_i (c_i . n_i).
 *
 * Positive exactly when the normals point outward, which is why this doubles as
 * the orientation test a solver actually cares about.
 */
static double enclosed_volume(MFTL_Object *obj, const double origin[3])
{
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  double v = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double *c = &obj->tris.center[3 * t];
    const double *n = &obj->tris.normal[3 * t];
    const double d = (c[0] - origin[0]) * n[0] + (c[1] - origin[1]) * n[1] + (c[2] - origin[2]) * n[2];
    v += obj->tris.area[t] * d;
  }
  return v / 3.0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * MFTLBuildIndexedMesh
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_indexed_mesh_rejects_bad_input(void)
{
  MFTL_Object *obj = MFTLObjectCreate(10);
  MFTLEnsembleAddObject(g_ens, obj);

  const double pos[9] = {0, 0, 0, 1, 0, 0, 0, 1, 0};
  const int tv[3] = {0, 1, 2};

  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildIndexedMesh(obj, MFTL_NULL, 3, tv, 1));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildIndexedMesh(obj, pos, 3, MFTL_NULL, 1));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildIndexedMesh(obj, pos, 2, tv, 1));  // fewer than 3 vertices
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildIndexedMesh(obj, pos, 3, tv, 0));  // no triangles

  // The object must be untouched by a rejected call.
  TEST_ASSERT_EQUAL_INT(0, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(0, obj->tris.n);
  return 0;
}

static int test_indexed_mesh_establishes_the_conventions(void)
{
  MFTL_Object *obj = MFTLObjectCreate(11);
  MFTLEnsembleAddObject(g_ens, obj);

  const double pos[12] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  const int tv[6] = {0, 1, 2, 0, 2, 3};
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(obj, pos, 4, tv, 2));

  /* Slot k is the ORIGIN of half-edge 3t+k -- the convention every other module
   * assumes. Checked against the caller's input list directly, because this is
   * the one place it is established. */
  for (int t = 0; t < 2; ++t)
  {
    for (int k = 0; k < 3; ++k)
    {
      TEST_ASSERT_EQUAL_INT(tv[3 * t + k], MFTLHeOrigin(obj, 3 * t + k));
      TEST_ASSERT_EQUAL_INT(tv[3 * t + (k + 1) % 3], MFTLHeTarget(obj, 3 * t + k));
      TEST_ASSERT_EQUAL_INT(tv[3 * t + k], obj->tris.verts[3 * t + k]);
    }
  }

  TEST_ASSERT(inv_global_ids_are_dense(obj));
  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));

  // Generation leaves the derived arrays ready to use.
  TEST_ASSERT(MFTLObjectHasTopology(obj));
  TEST_ASSERT(MFTLObjectHasFrames(obj));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Icosphere counts and structure
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_sphere_entity_counts(void)
{
  for (int r = 0; r <= 4; ++r)
  {
    MFTL_Object *obj = make_sphere(0, 0, 0, 1.0, r);

    int pow4 = 1;
    for (int i = 0; i < r; ++i) pow4 *= 4;

    TEST_ASSERT_EQUAL_INT(10 * pow4 + 2, obj->verts.n);
    TEST_ASSERT_EQUAL_INT(20 * pow4, obj->tris.n);
    TEST_ASSERT_EQUAL_INT(60 * pow4, MFTLNumHalfEdges(obj));

    // Everything generated is locally owned.
    TEST_ASSERT_EQUAL_INT(obj->verts.n, obj->verts.n_local);
    TEST_ASSERT_EQUAL_INT(obj->tris.n, obj->tris.n_local);

    // Euler characteristic of a sphere.
    const int V = obj->verts.n, F = obj->tris.n, E = 3 * F / 2;
    TEST_ASSERT_EQUAL_INT(2, V - E + F);
  }
  return 0;
}

static int test_sphere_is_a_sound_closed_mesh(void)
{
  MFTL_Object *obj = make_sphere(0, 0, 0, 1.0, 3);

  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  // A closed surface carries no boundary flags at all.
  TEST_ASSERT_EQUAL_INT(0, MFTLMarkMeshBoundary(obj));
  for (int v = 0; v < obj->verts.n; ++v) TEST_ASSERT(!MFTLVertexIsMeshBoundary(obj, v));

  // Every vertex of an icosphere has valence 5 or 6 -- exactly twelve have 5.
  int n_valence5 = 0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    int ring[MFTL_MAX_VALENCE];
    const int k = MFTLGetVertexOneRing(obj, v, ring, MFTL_MAX_VALENCE, MFTL_TRUE);
    TEST_ASSERT(k == 5 || k == 6);
    if (k == 5) n_valence5++;
  }
  TEST_ASSERT_EQUAL_INT(12, n_valence5);
  return 0;
}

static int test_vertices_lie_on_the_sphere(void)
{
  const double cx = 1.5, cy = -0.5, cz = 0.25, r = 2.0;
  MFTL_Object *obj = make_sphere(cx, cy, cz, r, 3);

  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *p = MFTLVertexPos(obj, v);
    const double d = sqrt((p[0] - cx) * (p[0] - cx) + (p[1] - cy) * (p[1] - cy) + (p[2] - cz) * (p[2] - cz));
    TEST_ASSERT_EQUAL_DOUBLE(r, d, 1e-12);
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Orientation -- the invariant that is silent when broken
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_triangle_winding_is_outward(void)
{
  const double c[3] = {0.5, -1.0, 2.0};
  MFTL_Object *obj = make_sphere(c[0], c[1], c[2], 1.0, 3);

  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* On a convex body every face normal must have a positive component along the
   * ray from the centre to the face. One inverted triangle is enough to flip the
   * sign of the surface-tension force on that element. */
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double *ctr = &obj->tris.center[3 * t];
    const double *n = &obj->tris.normal[3 * t];
    const double d = (ctr[0] - c[0]) * n[0] + (ctr[1] - c[1]) * n[1] + (ctr[2] - c[2]) * n[2];
    TEST_ASSERT(d > 0.0);
  }
  return 0;
}

static int test_enclosed_volume_is_positive_and_converges(void)
{
  const double origin[3] = {0.0, 0.0, 0.0};
  const double r = 1.0;
  const double exact = 4.0 / 3.0 * MFTL_PI * r * r * r;

  double prev_err = 1e30;
  for (int ref = 1; ref <= 4; ++ref)
  {
    MFTL_Object *obj = make_sphere(0, 0, 0, r, ref);
    const double v = enclosed_volume(obj, origin);

    TEST_ASSERT(v > 0.0);  // outward orientation, as a solver would see it
    TEST_ASSERT(v < exact);  // inscribed polyhedron: always an underestimate

    const double err = exact - v;
    TEST_ASSERT(err < prev_err);  // monotone convergence
    prev_err = err;
  }

  // At refinement 4 the discretisation error is well under a percent.
  TEST_ASSERT(prev_err / exact < 0.01);
  return 0;
}

static int test_surface_area_converges_from_below(void)
{
  const double r = 1.0;
  const double exact = 4.0 * MFTL_PI * r * r;

  double prev = 0.0;
  for (int ref = 1; ref <= 4; ++ref)
  {
    MFTL_Object *obj = make_sphere(0, 0, 0, r, ref);
    const double a = total_area(obj);

    TEST_ASSERT(a < exact);  // inscribed
    TEST_ASSERT(a > prev);  // monotone
    prev = a;
  }

  TEST_ASSERT((exact - prev) / exact < 0.01);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Degenerate and repeated use
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_zero_radius_sphere_is_valid(void)
{
  /* Degenerate but legal: every vertex collapses onto the centre. It must still
   * be a structurally sound mesh, and nothing may produce a NaN. */
  MFTL_Object *obj = make_sphere(1.0, 1.0, 1.0, 0.0, 1);

  TEST_ASSERT_EQUAL_INT(42, obj->verts.n);
  TEST_ASSERT(inv_implicit_indexing(obj));
  TEST_ASSERT(inv_twins_are_symmetric(obj));

  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  for (int t = 0; t < obj->tris.n; ++t)
  {
    TEST_ASSERT(!isnan(obj->tris.area[t]));
    TEST_ASSERT(obj->tris.area[t] < 1e-5);
  }
  return 0;
}

static int test_sphere_survives_a_morton_reorder(void)
{
  MFTL_Object *obj = make_sphere(0, 0, 0, 1.0, 3);

  const double area_before = total_area(obj);
  const double origin[3] = {0, 0, 0};
  const double vol_before = enclosed_volume(obj, origin);

  MFTLObjectReorderMorton(obj);

  /* Renumbering must be geometrically invisible. Comparing to round-off rather
   * than loosely: the same triangles are summed, only in a different order, so
   * the difference is floating-point reassociation and nothing else. */
  TEST_ASSERT_EQUAL_DOUBLE(area_before, total_area(obj), 1e-10);
  TEST_ASSERT_EQUAL_DOUBLE(vol_before, enclosed_volume(obj, origin), 1e-10);

  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-8.0, -8.0, -8.0};
  double hi[3] = {8.0, 8.0, 8.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 64);

  printf("--- Running MFTL Mesh Generation Tests ---\n");

  RUN_TEST(test_indexed_mesh_rejects_bad_input);
  RUN_TEST(test_indexed_mesh_establishes_the_conventions);
  RUN_TEST(test_sphere_entity_counts);
  RUN_TEST(test_sphere_is_a_sound_closed_mesh);
  RUN_TEST(test_vertices_lie_on_the_sphere);
  RUN_TEST(test_triangle_winding_is_outward);
  RUN_TEST(test_enclosed_volume_is_positive_and_converges);
  RUN_TEST(test_surface_area_converges_from_below);
  RUN_TEST(test_zero_radius_sphere_is_valid);
  RUN_TEST(test_sphere_survives_a_morton_reorder);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
