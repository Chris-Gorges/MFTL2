/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for mftl_geometry.h.
 *
 * The periodic helpers get most of the attention: they are the only place in the
 * library where a plausible-looking answer can be off by a whole domain length,
 * and every geometric quantity downstream inherits the mistake.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_open = MFTL_NULL;  ///< non-periodic, [-4, 4]^3
static MFTL_Ensemble *g_per = MFTL_NULL;  ///< periodic in x and y, [0, 10]^3

/* ─────────────────────────────────────────────────────────────────────────────
 * Positions
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_vertex_position_is_the_soa_array(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_open, obj);
  MFTLObjectSetCounts(obj, 4, 4, 0, 0);

  for (int v = 0; v < 4; ++v)
  {
    double *p = MFTLVertexPosMut(obj, v);
    p[0] = v;
    p[1] = 2 * v;
    p[2] = 3 * v;
  }

  /* The accessor must be a view ON the pool array, not a copy of it: there is
   * exactly one storage location for a coordinate in this design, and this is
   * the assertion that says so. */
  for (int v = 0; v < 4; ++v)
  {
    TEST_ASSERT(MFTLVertexPos(obj, v) == &obj->verts.pos[3 * v]);
    TEST_ASSERT_EQUAL_DOUBLE((double) v, obj->verts.pos[3 * v + 0], 0.0);
    TEST_ASSERT_EQUAL_DOUBLE((double) (3 * v), MFTLVertexPos(obj, v)[2], 0.0);
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Minimum image
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_minimum_image_on_a_non_periodic_domain(void)
{
  const MFTL_Domain *dom = &g_open->domain;
  const double a[3] = {-3.0, 0.0, 1.0};
  const double b[3] = {3.5, -2.0, 1.0};

  double d[3];
  MFTLGetMinimumImageVector(dom, d, a, b);

  // No wrapping anywhere: it is exactly the difference.
  TEST_ASSERT_EQUAL_DOUBLE(6.5, d[0], 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(-2.0, d[1], 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, d[2], 1e-15);
  return 0;
}

static int test_minimum_image_takes_the_short_way_round(void)
{
  const MFTL_Domain *dom = &g_per->domain;  // period 10 in x and y, open in z
  const double a[3] = {0.5, 0.5, 0.5};
  const double b[3] = {9.5, 9.5, 9.5};

  double d[3];
  MFTLGetMinimumImageVector(dom, d, a, b);

  // x and y wrap: the short way is -1, not +9.
  TEST_ASSERT_EQUAL_DOUBLE(-1.0, d[0], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(-1.0, d[1], 1e-12);
  // z does not wrap, so it stays +9.
  TEST_ASSERT_EQUAL_DOUBLE(9.0, d[2], 1e-12);
  return 0;
}

static int test_minimum_image_is_antisymmetric_and_bounded(void)
{
  const MFTL_Domain *dom = &g_per->domain;

  for (int i = 0; i < 200; ++i)
  {
    const double a[3] = {0.05 * i, 10.0 - 0.03 * i, 0.5};
    const double b[3] = {10.0 - 0.02 * i, 0.04 * i, 1.5};

    double ab[3];
    double ba[3];
    MFTLGetMinimumImageVector(dom, ab, a, b);
    MFTLGetMinimumImageVector(dom, ba, b, a);

    for (int k = 0; k < 2; ++k)
    {
      TEST_ASSERT_EQUAL_DOUBLE(-ab[k], ba[k], 1e-12);
      // Never longer than half a period on a periodic axis. That is the whole
      // definition, and it is what stops a seam-crossing edge from inflating an
      // area by a factor of the domain size.
      TEST_ASSERT(fabs(ab[k]) <= dom->domain_half_size[k] + 1e-12);
    }
  }
  return 0;
}

static int test_minimum_image_survives_far_displacements(void)
{
  const MFTL_Domain *dom = &g_per->domain;
  const double a[3] = {1.0, 1.0, 0.0};

  /* The round-based form is correct for ANY displacement. The branchless
   * comparison form it replaced silently required |d| < 1.5 L, which is exactly
   * the kind of precondition nobody checks. */
  for (int n = -5; n <= 5; ++n)
  {
    const double b[3] = {1.5 + 10.0 * n, 1.0, 0.0};
    double d[3];
    MFTLGetMinimumImageVector(dom, d, a, b);
    TEST_ASSERT_EQUAL_DOUBLE(0.5, d[0], 1e-9);
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Wrapping and ownership
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_apply_periodicity_wraps_into_the_box(void)
{
  const MFTL_Domain *dom = &g_per->domain;

  double p[3] = {10.5, -0.5, 12.0};
  MFTLApplyPeriodicity(dom, p);

  TEST_ASSERT_EQUAL_DOUBLE(0.5, p[0], 1e-12);  // wrapped down
  TEST_ASSERT_EQUAL_DOUBLE(9.5, p[1], 1e-12);  // wrapped up
  TEST_ASSERT_EQUAL_DOUBLE(12.0, p[2], 1e-12);  // z is not periodic: untouched

  // A position already inside is left alone.
  double q[3] = {5.0, 5.0, 5.0};
  MFTLApplyPeriodicity(dom, q);
  TEST_ASSERT_EQUAL_DOUBLE(5.0, q[0], 0.0);
  return 0;
}

static int test_owner_proc_lookup_covers_the_domain(void)
{
  const MFTL_Domain *dom = &g_open->domain;

  /* On one rank every position belongs to rank 0. The interesting part is that
   * the lookup is total: no coordinate anywhere in (or slightly outside) the box
   * may produce an out-of-range table index, because a bad index here is a read
   * of arbitrary memory rather than a wrong rank. */
  for (int i = -2; i <= 12; ++i)
  {
    for (int j = -2; j <= 12; ++j)
    {
      double p[3] = {-4.0 + 0.8 * i, -4.0 + 0.8 * j, 0.0};
      const int owner = MFTLGetOwnerProc(dom, p);
      TEST_ASSERT(owner >= 0 && owner < dom->nproc);
    }
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Triangle properties
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_triangle_properties_of_a_known_triangle(void)
{
  MFTL_Object *obj = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(g_open, obj);

  // Right triangle with legs 3 and 4 in the z = 1 plane: area 6, normal +z.
  const double pos[9] = {0.0, 0.0, 1.0, 3.0, 0.0, 1.0, 0.0, 4.0, 1.0};
  const int tv[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(obj, pos, 3, tv, 1);

  double area;
  double normal[3];
  double center[3];
  MFTLGetTriangleProperties(obj, 0, &area, normal, center);

  TEST_ASSERT_EQUAL_DOUBLE(6.0, area, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, normal[0], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, normal[1], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, normal[2], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, center[0], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(4.0 / 3.0, center[1], 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, center[2], 1e-12);

  // The convenience wrappers must agree with the fused call.
  TEST_ASSERT_EQUAL_DOUBLE(6.0, MFTLGetTriangleArea(obj, 0), 1e-12);

  double n2[3];
  MFTLGetTriangleNormal(obj, 0, n2);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, n2[2], 1e-12);
  return 0;
}

static int test_degenerate_triangle_does_not_produce_nan(void)
{
  MFTL_Object *obj = MFTLObjectCreate(3);
  MFTLEnsembleAddObject(g_open, obj);

  // Three collinear points: zero area, and no normal worth having.
  const double pos[9] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0};
  const int tv[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(obj, pos, 3, tv, 1);

  double area;
  double normal[3];
  MFTLGetTriangleProperties(obj, 0, &area, normal, MFTL_NULL);

  /* A NaN here would poison every reduction it reaches -- the min/max quality
   * statistics in particular, which are global. The guard on the reciprocal's
   * divisor keeps the failure local and visible.
   *
   * EXACTLY zero, not merely small: the guard used to be applied to the squared
   * magnitude BEFORE the square root, which floored every reported area at
   * 0.5*sqrt(MFTL_VERYSMALL) = 5e-7 and made `quality.min_triangle_area` unable
   * to report a degenerate triangle at all. This assertion is what pins the
   * guard to the divisor. */
  TEST_ASSERT_EQUAL_DOUBLE(0.0, area, 0.0);

  /* Zero, which is the same "not computable" sentinel normalise_vertex_normals
   * uses -- a unit normal never has zero length. */
  for (int k = 0; k < 3; ++k)
  {
    TEST_ASSERT(!isnan(normal[k]));
    TEST_ASSERT_EQUAL_DOUBLE(0.0, normal[k], 0.0);
  }
  return 0;
}

/**
 * @brief A triangle far smaller than the old area floor is reported exactly.
 *
 * Two separate consequences of clamping the magnitude rather than the divisor,
 * and both are silent:
 *
 *   the AREA was floored at 5e-7, so a legitimately small triangle reported an
 *   area orders of magnitude too large -- and the mean-ratio quality
 *   q = 4*sqrt(3)*A / (a^2+b^2+c^2) built from it exceeded 1, which is
 *   impossible for a measure maximised by the equilateral triangle;
 *
 *   the NORMAL was divided by that floor instead of by its own length, so it
 *   came out proportionally SHORT rather than unit -- here by a factor of 1e-6.
 *
 * Nothing about either is visible on a mesh of ordinary scale, which is why this
 * needs a test rather than an assertion.
 */
static int test_a_very_small_triangle_is_measured_exactly(void)
{
  MFTL_Object *obj = MFTLObjectCreate(31);
  MFTLEnsembleAddObject(g_open, obj);

  /* Legs of 1e-6 in the z = 0 plane: area 5e-13, a million times below the old
   * floor of 5e-7. */
  const double s = 1e-6;
  const double pos[9] = {0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, s, 0.0};
  const int tv[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(obj, pos, 3, tv, 1);

  double area;
  double normal[3];
  MFTLGetTriangleProperties(obj, 0, &area, normal, MFTL_NULL);

  TEST_ASSERT_EQUAL_DOUBLE(0.5 * s * s, area, 1e-25);

  const double len = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, len, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, normal[2], 1e-12);

  /* And the quality built from that area stays inside its own range. The legs are
   * s and s, the hypotenuse s*sqrt(2), so a^2+b^2+c^2 = 4s^2 and the mean ratio
   * of a right isoceles triangle is sqrt(3)/2. With the area floored it came out
   * at 4*sqrt(3)*5e-7/(4e-12), i.e. about 8.7e5. */
  const double q = MFTL_FOUR_SQRT3 * area / (4.0 * s * s);
  TEST_ASSERT(q > 0.0 && q <= 1.0);
  TEST_ASSERT_EQUAL_DOUBLE(sqrt(3.0) / 2.0, q, 1e-12);

  /* The bulk path must agree -- it is the one the per-timestep loop uses, and it
   * carries its own copy of the expression. */
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, 1);
  TEST_ASSERT_EQUAL_DOUBLE(0.5 * s * s, obj->tris.area[0], 1e-25);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, obj->tris.normal[2], 1e-12);
  return 0;
}

/**
 * @brief A triangle normal is a UNIT vector at every triangle scale, on BOTH paths.
 *
 * The sweep exists because a short normal is the worst available failure in a
 * physics kernel: it points the right way, so orientation tests, winding checks
 * and pictures all still pass, while it silently scales every quantity built from
 * it -- surface tension, the indicator-function gradient, any flux through the
 * face. Nothing downstream is positioned to notice a normal of length 0.999 or
 * 1e-6.
 *
 * Both connectivity paths are checked at every scale: the construction-time
 * half-edge path and the steady-state compact-index path.
 */
static int test_triangle_normals_are_unit_at_every_scale(void)
{
  for (int e = -8; e <= 8; ++e)
  {
    const double s = pow(10.0, (double) e);

    MFTL_Object *obj = MFTLObjectCreate(40 + e);
    MFTLEnsembleAddObject(g_open, obj);

    /* Deliberately scalene and off-axis, so no component can be zero by symmetry
     * and a per-component error cannot cancel. */
    const double pos[9] = {0.0, 0.0, 0.0, s, 0.5 * s, 0.25 * s, 0.25 * s, s, 0.75 * s};
    const int tv[3] = {0, 1, 2};
    MFTLBuildIndexedMesh(obj, pos, 3, tv, 1);

    double n[3];
    MFTLGetTriangleProperties(obj, 0, MFTL_NULL, n, MFTL_NULL);
    const double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);

    if (fabs(len - 1.0) > 1e-12)
    {
      printf("\n  scalar path, triangle scale 1e%d -> normal length %.17g ", e, len);
      TEST_FAIL();
    }

    /* Half-edge path, then compact-index gathered path. */
    MFTLObjectInvalidateDerived(obj);
    MFTLComputeTriangleGeometry(obj, 1);
    if (!inv_triangle_normals_are_unit(obj, 1e-12))
    {
      printf("\n  walk kernel, triangle scale 1e%d ", e);
      TEST_FAIL();
    }

    MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, 1);
    if (!inv_triangle_normals_are_unit(obj, 1e-12))
    {
      printf("\n  gathered kernel, triangle scale 1e%d ", e);
      TEST_FAIL();
    }
  }
  return 0;
}

/** @brief Every normal of a real mesh is unit, on both kernels and across a seam. */
static int test_normals_of_a_whole_mesh_are_unit(void)
{
  MFTL_Object *obj = MFTLObjectCreate(60);
  MFTLEnsembleAddObject(g_per, obj);

  /* On the periodic ensemble and straddling x = 0, so the seam-corrected edge
   * vectors are exercised too -- a normal built from an uncorrected difference
   * would be a domain-length vector and nowhere near unit. */
  double c[3] = {0.0, 5.0, 5.0};
  MFTLGenerateSphere(obj, c, 2.0, 3);

  MFTLObjectInvalidateDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  TEST_ASSERT(inv_triangle_normals_are_unit(obj, 1e-12));

  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  TEST_ASSERT(inv_triangle_normals_are_unit(obj, 1e-12));

  /* And the per-vertex pass, whose output feeds the remesher's tangent-plane
   * projection -- a short vertex normal there leaves a residual normal component
   * in the displacement, which is exactly the surface shrinkage the projection
   * exists to remove. */
  MFTLComputeVertexNormals(obj);
  TEST_ASSERT(inv_vertex_normals_are_unit(obj, 1e-12));

  return 0;
}

static int test_edge_lengths_are_periodic_aware(void)
{
  MFTL_Object *obj = MFTLObjectCreate(4);
  MFTLEnsembleAddObject(g_per, obj);

  // Two corners just either side of the x seam of a period-10 domain.
  const double pos[9] = {9.5, 5.0, 5.0, 0.5, 5.0, 5.0, 9.5, 6.0, 5.0};
  const int tv[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(obj, pos, 3, tv, 1);

  // Half-edge 0 runs v0 -> v1, i.e. straight across the seam: the true length is
  // 1.0, not 9.0.
  TEST_ASSERT_EQUAL_DOUBLE(1.0, MFTLGetEdgeLength(obj, 0), 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, MFTLGetEdgeLengthSquared(obj, 0), 1e-12);

  double e[3];
  MFTLGetEdgeVector(obj, 0, e);
  TEST_ASSERT_EQUAL_DOUBLE(1.0, e[0], 1e-12);

  // Whole-triangle area follows: 0.5 * 1 * 1, not 0.5 * 9 * 1.
  TEST_ASSERT_EQUAL_DOUBLE(0.5, MFTLGetTriangleArea(obj, 0), 1e-12);
  return 0;
}

static int test_bulk_geometry_matches_the_scalar_path(void)
{
  MFTL_Object *obj = MFTLObjectCreate(5);
  MFTLEnsembleAddObject(g_open, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 1.5, 2);

  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  for (int t = 0; t < obj->tris.n; ++t)
  {
    double area;
    double normal[3];
    double center[3];
    MFTLGetTriangleProperties(obj, t, &area, normal, center);

    TEST_ASSERT_EQUAL_DOUBLE(area, obj->tris.area[t], 1e-15);
    for (int k = 0; k < 3; ++k)
    {
      TEST_ASSERT_EQUAL_DOUBLE(normal[k], obj->tris.normal[3 * t + k], 1e-15);
      TEST_ASSERT_EQUAL_DOUBLE(center[k], obj->tris.center[3 * t + k], 1e-15);
    }
  }
  return 0;
}

/**
 * @brief The two kernels behind MFTLComputeTriangleGeometry agree BIT FOR BIT.
 *
 * The construction-time path reads the half-edges; the steady-state path streams
 * compact vertex indices. They must give the same answer. Run on the periodic
 * ensemble because that also compares their two minimum-image loop shapes.
 */
static int test_triangle_geometry_paths_are_bit_identical(void)
{
  MFTL_Object *obj = MFTLObjectCreate(11);
  MFTLEnsembleAddObject(g_per, obj);

  /* Straddling the x = 0 face, so a real share of the triangles need the seam
   * correction -- otherwise this would only test the non-periodic branch. */
  double c[3] = {0.0, 5.0, 5.0};
  MFTLGenerateSphere(obj, c, 2.0, 3);

  const int nt = obj->tris.n;
  const size_t bytes = (size_t) nt * sizeof(double);

  double *walk_area = (double *) malloc(bytes);
  double *walk_normal = (double *) malloc(3 * bytes);
  double *walk_center = (double *) malloc(3 * bytes);

  /* The generator leaves the derived arrays BUILT, so the walk has to be asked
   * for explicitly here. That is worth knowing in itself: after generation, and
   * after a redistribution, the streaming path is what a caller gets for free. */
  MFTLObjectInvalidateDerived(obj);
  TEST_ASSERT(!MFTLObjectHasTopology(obj));
  MFTLComputeTriangleGeometry(obj, nt);
  memcpy(walk_area, obj->tris.area, bytes);
  memcpy(walk_normal, obj->tris.normal, 3 * bytes);
  memcpy(walk_center, obj->tris.center, 3 * bytes);

  /* Same object and positions, compact connectivity now current. */
  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(MFTLObjectHasTopology(obj));
  memset(obj->tris.area, 0, bytes);
  memset(obj->tris.normal, 0, 3 * bytes);
  memset(obj->tris.center, 0, 3 * bytes);
  MFTLComputeTriangleGeometry(obj, nt);

  const int same = (memcmp(walk_area, obj->tris.area, bytes) == 0) && (memcmp(walk_normal, obj->tris.normal, 3 * bytes) == 0) &&
                   (memcmp(walk_center, obj->tris.center, 3 * bytes) == 0);

  free(walk_area);
  free(walk_normal);
  free(walk_center);

  TEST_ASSERT(same);
  return 0;
}

/** @brief The dispatch honours `count`, and does not touch triangle count..n. */
static int test_triangle_geometry_respects_the_count(void)
{
  MFTL_Object *obj = MFTLObjectCreate(12);
  MFTLEnsembleAddObject(g_per, obj);
  double c[3] = {5.0, 5.0, 5.0};
  MFTLGenerateSphere(obj, c, 1.0, 2);
  MFTLObjectBuildDerived(obj);

  const int nt = obj->tris.n;
  const int half = nt / 2;

  for (int t = 0; t < nt; ++t) obj->tris.area[t] = -1.0;
  MFTLComputeTriangleGeometry(obj, half);

  for (int t = 0; t < half; ++t) TEST_ASSERT(obj->tris.area[t] > 0.0);
  for (int t = half; t < nt; ++t) TEST_ASSERT(obj->tris.area[t] == -1.0);

  /* And the same on the walk path, since the bound lives in each kernel. */
  MFTLObjectInvalidateDerived(obj);
  for (int t = 0; t < nt; ++t) obj->tris.area[t] = -1.0;
  MFTLComputeTriangleGeometry(obj, half);

  for (int t = 0; t < half; ++t) TEST_ASSERT(obj->tris.area[t] > 0.0);
  for (int t = half; t < nt; ++t) TEST_ASSERT(obj->tris.area[t] == -1.0);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  g_open = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 16);

  double plo[3] = {0.0, 0.0, 0.0};
  double phi[3] = {10.0, 10.0, 10.0};
  bool per[3] = {true, true, false};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 16);

  printf("--- Running MFTL Geometry Tests ---\n");

  RUN_TEST(test_vertex_position_is_the_soa_array);
  RUN_TEST(test_minimum_image_on_a_non_periodic_domain);
  RUN_TEST(test_minimum_image_takes_the_short_way_round);
  RUN_TEST(test_minimum_image_is_antisymmetric_and_bounded);
  RUN_TEST(test_minimum_image_survives_far_displacements);
  RUN_TEST(test_apply_periodicity_wraps_into_the_box);
  RUN_TEST(test_owner_proc_lookup_covers_the_domain);
  RUN_TEST(test_triangle_properties_of_a_known_triangle);
  RUN_TEST(test_degenerate_triangle_does_not_produce_nan);
  RUN_TEST(test_a_very_small_triangle_is_measured_exactly);
  RUN_TEST(test_triangle_normals_are_unit_at_every_scale);
  RUN_TEST(test_normals_of_a_whole_mesh_are_unit);
  RUN_TEST(test_edge_lengths_are_periodic_aware);
  RUN_TEST(test_bulk_geometry_matches_the_scalar_path);
  RUN_TEST(test_triangle_geometry_paths_are_bit_identical);
  RUN_TEST(test_triangle_geometry_respects_the_count);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_open);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
