/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for MFTLComputeVertexNormals (src/geometry/vertex_geometry.c).
 *
 * The pass is a scatter, which is the shape most likely to be subtly wrong: an
 * index computed one way in the accumulation and another way in a consumer, a
 * missed zeroing, a vertex whose contributions are counted twice. So the tests
 * do not check the implementation against itself -- most of them compare it with
 * a REFERENCE built from a completely different code path (a half-edge one-ring
 * walk plus MFTLGetTriangleProperties), and the rest check physics that no implementation detail can
 * fake.
 *
 * Multi-rank note: MFTLGenerateSphere is collective and gives rank 0 the whole
 * mesh, so at nproc > 1 every other rank holds an EMPTY object. That is exactly
 * the degenerate path worth covering, and it is why every verdict here goes
 * through agreed() -- TEST_ASSERT returns immediately, so an assertion that
 * fails on one rank only would leave the others blocked in the next collective.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_open = MFTL_NULL;  ///< non-periodic, [-4, 4]^3
static MFTL_Ensemble *g_per = MFTL_NULL;   ///< periodic in all three axes, [0, 8]^3

/* ─────────────────────────────────────────────────────────────────────────────
 * An independent reference implementation
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Area-weighted vertex normals computed the slow, obvious way.
 *
 * Deliberately shares nothing with the implementation under test: it walks the
 * half-edge fan instead of streaming triangles, and it gets each face's area and
 * unit normal from MFTLGetTriangleProperties. Agreement therefore checks two
 * independent traversal strategies over the same authoritative positions.
 */
static void reference_vertex_normals(const MFTL_Object *obj, double *out)
{
  const int nv = obj->verts.n;

  for (int v = 0; v < nv; ++v)
  {
    double sx = 0.0, sy = 0.0, sz = 0.0;

    int ring[MFTL_MAX_VALENCE];
    const int count = MFTLGetVertexOneRing(obj, v, ring, MFTL_MAX_VALENCE, MFTL_FALSE);

    for (int i = 0; i < count; ++i)
    {
      double area = 0.0;
      double n[3];
      MFTLGetTriangleProperties(obj, ring[i], &area, n, MFTL_NULL);

      /* The implementation accumulates e1 x e2, whose magnitude is 2*area. The
       * factor 2 is common to every term and vanishes in the normalisation, so
       * area-weighting the unit normals is the same statement. */
      sx += 2.0 * area * n[0];
      sy += 2.0 * area * n[1];
      sz += 2.0 * area * n[2];
    }

    const double len = sqrt(sx * sx + sy * sy + sz * sz);
    const double inv = (len > 1e-30) ? 1.0 / len : 0.0;
    out[3 * v + 0] = sx * inv;
    out[3 * v + 1] = sy * inv;
    out[3 * v + 2] = sz * inv;
  }
}

/**
 * @brief Largest component-wise difference between two normal fields.
 *
 * ── Why this and not the angle between them ────────────────────────────────
 *
 * The obvious metric, acos(a . b), is the WRONG tool for comparing two vectors
 * that are supposed to be identical, and it produced a false failure here before
 * this note existed. Near dot == 1 the arc cosine has infinite derivative: a dot
 * product one ulp below 1 comes out as acos(1 - 1e-16) = 1.5e-8 radians, so the
 * metric has a NOISE FLOOR of about 1e-6 degrees no matter how exact the inputs
 * are. Two normal fields agreeing to 1.5 ulp measured 1.5e-6 degrees apart and
 * looked like a bug.
 *
 * The chord |a - b| has derivative 1 at the point of interest and simply reports
 * what the difference is: the same two fields measure 3.3e-16 here. Use acos
 * only where the angle is genuinely large -- the radial and tilt checks below --
 * and never to ask "are these the same vector".
 */
static double max_deviation(const double *a, const double *b, int nv)
{
  double worst = 0.0;
  for (int i = 0; i < 3 * nv; ++i)
  {
    const double d = fabs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

/** @brief Angle in degrees between two unit vectors. For LARGE angles only. */
static double angle_deg(const double *a, const double *b)
{
  double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  if (dot > 1.0) dot = 1.0;
  if (dot < -1.0) dot = -1.0;
  return acos(dot) * 180.0 / MFTL_PI;
}

/** @brief Builds a sphere, Morton-orders it and computes its vertex normals. */
static MFTL_Object *make_sphere(MFTL_Ensemble *ens, const double centre[3], double radius, int refinement)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {centre[0], centre[1], centre[2]};
  MFTLGenerateSphere(obj, c, radius, refinement);
  MFTLObjectReorderMorton(obj);
  MFTLComputeVertexNormals(obj);
  return obj;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 1. Basic contract
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_every_normal_is_unit_length(void)
{
  const double c[3] = {0.0, 0.0, 0.0};
  MFTL_Object *obj = make_sphere(g_open, c, 1.0, 4);

  int ok = 1;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *n = &obj->verts.normal[3 * v];
    const double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (fabs(len - 1.0) > 1e-12) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

static int test_normals_agree_with_the_reference(void)
{
  const double c[3] = {0.0, 0.0, 0.0};
  MFTL_Object *obj = make_sphere(g_open, c, 1.0, 4);

  int ok = 1;
  if (obj->verts.n > 0)
  {
    double *ref = (double *) malloc(3 * (size_t) obj->verts.n * sizeof(double));
    reference_vertex_normals(obj, ref);

    /* Different summation order and a different route to each face normal, so
     * the two agree to rounding rather than exactly. Measured at 3.3e-16 -- 1.5
     * ulp -- so 1e-13 is a real bound with headroom, not a tolerance chosen to
     * make the test pass. */
    if (max_deviation(obj->verts.normal, ref, obj->verts.n) > 1e-13) ok = 0;

    free(ref);
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

static int test_recomputing_gives_the_same_answer(void)
{
  const double c[3] = {0.0, 0.0, 0.0};
  MFTL_Object *obj = make_sphere(g_open, c, 1.0, 3);

  int ok = 1;
  if (obj->verts.n > 0)
  {
    const size_t bytes = 3 * (size_t) obj->verts.n * sizeof(double);
    double *first = (double *) malloc(bytes);
    memcpy(first, obj->verts.normal, bytes);

    /* The accumulator is the output array, so a missing memset would double
     * every contribution -- which normalisation would then HIDE, since 2n and n
     * have the same direction. Running it three times and comparing bit for bit
     * is what actually catches it: with a missing clear the third run differs
     * from the first in the rounding of the un-normalised sum. */
    MFTLComputeVertexNormals(obj);
    MFTLComputeVertexNormals(obj);

    if (memcmp(first, obj->verts.normal, bytes) != 0) ok = 0;
    free(first);
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 2. Physics the implementation cannot fake
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_sphere_normals_point_radially_outward(void)
{
  const double c[3] = {0.5, -1.0, 0.25};
  MFTL_Object *obj = make_sphere(g_open, c, 1.5, 4);

  int ok = 1;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *p = MFTLVertexPos(obj, v);
    const double *n = &obj->verts.normal[3 * v];
    const double r[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};

    /* Outward, not merely parallel: this is the same orientation statement the
     * winding invariant makes about face normals, carried to the vertices. A
     * sign error anywhere in the cross product flips every one of these. */
    if (r[0] * n[0] + r[1] * n[1] + r[2] * n[2] <= 0.0) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

static int test_radial_error_falls_under_refinement(void)
{
  const double c[3] = {0.0, 0.0, 0.0};

  double err[3] = {0.0, 0.0, 0.0};
  for (int i = 0; i < 3; ++i)
  {
    MFTL_Object *obj = make_sphere(g_open, c, 1.0, 2 + i);

    double worst = 0.0;
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      const double rl = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      const double radial[3] = {p[0] / rl, p[1] / rl, p[2] / rl};

      /* These angles are 0.5 degrees down to 0.03 degrees -- five orders above
       * the arc cosine's noise floor, so acos is legitimate here. */
      const double d = angle_deg(radial, &obj->verts.normal[3 * v]);
      if (d > worst) worst = d;
    }
    err[i] = worst;
  }

  /* A rank holding no mesh measures a zero error at every level, which is
   * vacuously monotone -- so this only asserts where there is a mesh, and the
   * verdict is agreed across ranks either way. */
  int ok = 1;
  if (g_open->local_objects[g_open->n_local_objects - 1]->verts.n > 0)
  {
    if (!(err[1] < err[0] && err[2] < err[1])) ok = 0;
    if (!(err[2] < 0.3)) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 3. Periodicity: the seam must not be visible in the result
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_normals_are_unaffected_by_a_periodic_seam(void)
{
  /* The same sphere twice in the same fully periodic domain: once well inside
   * it, once centred exactly on the corner so that every one of its triangles
   * near the seam has vertices wrapped to the opposite face.
   *
   * The vertex normal uses minimum-image edge vectors, so the two must produce
   * the SAME normal at corresponding
   * vertices. If the seam handling were wrong, the wrapped sphere's rim vertices
   * would get normals pointing across the domain. */
  const double inside[3] = {4.0, 4.0, 4.0};
  const double corner[3] = {0.0, 0.0, 0.0};

  MFTL_Object *a = make_sphere(g_per, inside, 1.0, 4);
  MFTL_Object *b = make_sphere(g_per, corner, 1.0, 4);

  int ok = 1;
  if (a->verts.n > 0)
  {
    if (a->verts.n != b->verts.n) ok = 0;

    /* Both are generated by the same deterministic subdivision from the same
     * refinement, but each is Morton-ordered against its OWN bounding box -- and
     * the wrapped sphere's box is the whole domain. So the comparison is made by
     * global id, which is the identity that survives a reordering. */
    if (ok)
    {
      double *by_gid = (double *) malloc(3 * (size_t) b->verts.n * sizeof(double));
      for (int v = 0; v < b->verts.n; ++v)
      {
        const int g = b->verts.rec[v].global_id;
        by_gid[3 * g + 0] = b->verts.normal[3 * v + 0];
        by_gid[3 * g + 1] = b->verts.normal[3 * v + 1];
        by_gid[3 * g + 2] = b->verts.normal[3 * v + 2];
      }

      /* Measured at 7.4e-15. It is larger than the 3.3e-16 of the reference
       * comparison for a real reason rather than noise: the wrapped sphere's
       * edge vectors are recovered as (d - L) + L, whose cancellation costs
       * about ulp(L)/|d| in relative terms. That is the intrinsic price of the
       * minimum-image convention, it is bounded, and 1e-12 bounds it. */
      for (int v = 0; v < a->verts.n && ok; ++v)
      {
        const int g = a->verts.rec[v].global_id;
        if (max_deviation(&a->verts.normal[3 * v], &by_gid[3 * g], 1) > 1e-12) ok = 0;
      }
      free(by_gid);
    }
  }

  TEST_ASSERT(agreed(ok, g_per->domain.comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 4. Degenerate inputs
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_isolated_vertex_gets_a_zero_normal(void)
{
  /* One triangle plus one vertex that belongs to nothing. The isolated vertex
   * must come out exactly zero -- a detectable "not computed", rather than a
   * normalised arbitrary direction that a consumer would take for a real
   * normal. */
  MFTL_Object *obj = MFTLObjectCreate(7);
  MFTLEnsembleAddObject(g_open, obj);
  MFTLObjectSetCounts(obj, 4, 4, 1, 1);

  const double p[4][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {2.0, 2.0, 2.0}};
  for (int v = 0; v < 4; ++v)
  {
    double *q = MFTLVertexPosMut(obj, v);
    q[0] = p[v][0];
    q[1] = p[v][1];
    q[2] = p[v][2];
    obj->verts.rec[v].out_he_idx = MFTL_ERROR;
    obj->verts.rec[v].global_id = v;
  }

  /* Half-edge 3t+k ends at vertex (k+1)%3, so that slot k is the origin --
   * the convention tris.verts follows. */
  obj->tris.he[0].vertex_idx = 1;
  obj->tris.he[1].vertex_idx = 2;
  obj->tris.he[2].vertex_idx = 0;
  for (int h = 0; h < 3; ++h) obj->tris.he[h].twin_idx = MFTL_ERROR;
  obj->verts.rec[0].out_he_idx = 0;
  obj->verts.rec[1].out_he_idx = 1;
  obj->verts.rec[2].out_he_idx = 2;

  MFTLObjectBuildDerived(obj);
  MFTLComputeVertexNormals(obj);

  int ok = 1;
  for (int k = 0; k < 3; ++k)
  {
    if (obj->verts.normal[3 * 3 + k] != 0.0) ok = 0;
  }

  /* The three real vertices all belong to one triangle in the z = 0 plane, so
   * each must carry that triangle's normal exactly. */
  for (int v = 0; v < 3; ++v)
  {
    if (fabs(obj->verts.normal[3 * v + 0]) > 1e-15) ok = 0;
    if (fabs(obj->verts.normal[3 * v + 1]) > 1e-15) ok = 0;
    if (fabs(obj->verts.normal[3 * v + 2] - 1.0) > 1e-15) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

static int test_empty_object_is_handled(void)
{
  MFTL_Object *obj = MFTLObjectCreate(9);
  MFTLEnsembleAddObject(g_open, obj);
  MFTLObjectBuildDerived(obj);  // vacuously current on an empty mesh

  MFTLComputeVertexNormals(obj);  // must not touch a NULL pool pointer

  TEST_ASSERT(agreed(obj->verts.n == 0, g_open->domain.comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 5. Interaction with the rest of the per-timestep pipeline
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_normals_follow_vertex_motion(void)
{
  const double c[3] = {0.0, 0.0, 0.0};
  MFTL_Object *obj = make_sphere(g_open, c, 1.0, 4);

  /* Squash the sphere into an ellipsoid and recompute from current positions. */
  MFTL_FOREACH_OWNED_VERTEX(obj, pos[2] *= 0.5;);
  MFTLComputeVertexNormals(obj);

  int ok = 1;
  if (obj->verts.n > 0)
  {
    double *ref = (double *) malloc(3 * (size_t) obj->verts.n * sizeof(double));
    reference_vertex_normals(obj, ref);
    if (max_deviation(obj->verts.normal, ref, obj->verts.n) > 1e-13) ok = 0;
    free(ref);

    /* On an oblate ellipsoid the normal is no longer radial anywhere except at
     * the poles and the equator. This angle is genuinely large, so acos is the
     * right tool here -- and a large tilt is what says the normals actually
     * followed the motion instead of being left one step stale. */
    double max_tilt = 0.0;
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      const double rl = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      if (rl < MFTL_VERYSMALL) continue;

      const double radial[3] = {p[0] / rl, p[1] / rl, p[2] / rl};
      const double d = angle_deg(radial, &obj->verts.normal[3 * v]);
      if (d > max_tilt) max_tilt = d;
    }
    if (!(max_tilt > 5.0)) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_open->domain.comm_world));
  return 0;
}

static int test_mesh_stays_sound_around_the_pass(void)
{
  const double c[3] = {0.0, 0.0, 0.0};
  MFTL_Object *obj = make_sphere(g_open, c, 1.0, 3);

  /* The pass writes one SoA array and nothing else. Everything structural about
   * the mesh must be exactly as it was. */
  TEST_ASSERT(agreed(inv_mesh_is_sound(obj, g_open->domain.rank) ? 1 : 0, g_open->domain.comm_world));
  TEST_ASSERT(agreed(MFTLObjectPoolsAreConsistent(obj) ? 1 : 0, g_open->domain.comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  cuts[0] = nproc;  // a thin decomposition, so the test runs at any rank count

  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  g_open = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 32);

  double plo[3] = {0.0, 0.0, 0.0};
  double phi[3] = {8.0, 8.0, 8.0};
  bool per[3] = {true, true, true};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 32);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) printf("--- Running MFTL Vertex Normal Tests (nproc = %d) ---\n", nproc);

  RUN_TEST(test_every_normal_is_unit_length);
  RUN_TEST(test_normals_agree_with_the_reference);
  RUN_TEST(test_recomputing_gives_the_same_answer);
  RUN_TEST(test_sphere_normals_point_radially_outward);
  RUN_TEST(test_radial_error_falls_under_refinement);
  RUN_TEST(test_normals_are_unaffected_by_a_periodic_seam);
  RUN_TEST(test_isolated_vertex_gets_a_zero_normal);
  RUN_TEST(test_empty_object_is_handled);
  RUN_TEST(test_normals_follow_vertex_motion);
  RUN_TEST(test_mesh_stays_sound_around_the_pass);

  if (rank == 0) TEST_REPORT();

  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_open);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
