/**
 * @file test_remeshing_sizing.c
 * @brief The curvature-adaptive sizing mode of the remesher (Dunyach et al.
 *        2013) and the curvature estimator it rests on.
 *
 * The references share nothing with the code under test: curvatures of the
 * sphere and of an ellipsoid are the closed-form ones, the sizing formula is
 * re-evaluated in the test, and every remeshing result is judged by geometry
 * the test measures itself (edge lengths by region, triangle orientation,
 * enclosed volume). The uniform mode is the control in every comparison: a
 * claim that the adaptive mode concentrates triangles is only a claim if the
 * uniform mode, on the same input, does not.
 *
 * Serial by construction (one rank holds every fixture); the distributed
 * behaviour is in test_remeshing_sizing_parallel.c.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_remeshing.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

/* ═══════════════════════════════════════════════════════════════════════════
 * Fixtures
 * ═══════════════════════════════════════════════════════════════════════════ */

static MFTL_Object *fixture_new(void)
{
  MFTL_Object *obj = MFTLObjectCreate(g_ens->n_local_objects);
  MFTLEnsembleAddObject(g_ens, obj);
  return obj;
}

/** @brief The unit icosphere scaled by (sx, sy, sz): an ellipsoid with semi-axes sx, sy, sz. */
static MFTL_Object *make_ellipsoid(int refinements, double sx, double sy, double sz)
{
  MFTL_Object *obj = fixture_new();
  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, refinements);
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *p = MFTLVertexPosMut(obj, v);
    p[0] *= sx;
    p[1] *= sy;
    p[2] *= sz;
  }
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/**
 * @brief A flat nx-by-ny grid on [0, wx] x [0, wy], z = 0, each quad cut along
 *        the same diagonal: every interior vertex has valence six.
 */
static MFTL_Object *make_patch(int nx, int ny, double wx, double wy)
{
  const int nvx = nx + 1, nvy = ny + 1, nv = nvx * nvy, nt = 2 * nx * ny;
  double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
  int *tri = (int *) malloc((size_t) nt * 3 * sizeof(int));
  for (int j = 0; j < nvy; ++j)
    for (int i = 0; i < nvx; ++i)
    {
      const int v = j * nvx + i;
      pos[3 * v + 0] = wx * (double) i / (double) nx;
      pos[3 * v + 1] = wy * (double) j / (double) ny;
      pos[3 * v + 2] = 0.0;
    }
  int t = 0;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      const int v00 = j * nvx + i, v10 = v00 + 1, v01 = v00 + nvx, v11 = v01 + 1;
      tri[3 * t + 0] = v00, tri[3 * t + 1] = v10, tri[3 * t + 2] = v11, ++t;
      tri[3 * t + 0] = v00, tri[3 * t + 1] = v11, tri[3 * t + 2] = v01, ++t;
    }
  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, nv, tri, nt);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  free(pos);
  free(tri);
  return obj;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Measurements
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Closed-form mean curvature and maximum absolute curvature of x^2/a^2 + y^2/b^2 + z^2/c^2 = 1 at @p p. */
static void ellipsoid_curvature(const double p[3], double a, double b, double c, double *H, double *K)
{
  const double g = p[0] * p[0] / (a * a * a * a) + p[1] * p[1] / (b * b * b * b) + p[2] * p[2] / (c * c * c * c);
  const double abc2 = a * a * b * b * c * c;
  *K = 1.0 / (abc2 * g * g);
  *H = fabs(p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - a * a - b * b - c * c) / (2.0 * abc2 * pow(g, 1.5));
}

/** @brief Mean length of the undirected edges whose midpoint has |x| in [lo, hi). */
static double mean_edge_in_band(const MFTL_Object *obj, double lo, double hi, int *count)
{
  double sum = 0.0;
  int n = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int tw = obj->tris.he[h].twin_idx;
    if (tw != MFTL_ERROR && h > tw) continue;
    const double *a = MFTLVertexPos(obj, MFTLHeOrigin(obj, h));
    const double *b = MFTLVertexPos(obj, MFTLHeTarget(obj, h));
    const double x = fabs(0.5 * (a[0] + b[0]));
    if (x < lo || x >= hi) continue;
    sum += MFTLGetEdgeLength(obj, h);
    n++;
  }
  if (count) *count = n;
  return (n > 0) ? sum / (double) n : 0.0;
}

/** @brief Triangles whose normal points towards the origin (inverted on a convex body). */
static int count_inverted(const MFTL_Object *obj)
{
  int bad = 0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double n[3], c[3];
    MFTLGetTriangleProperties(obj, t, MFTL_NULL, n, c);
    if (n[0] * c[0] + n[1] * c[1] + n[2] * c[2] <= 0.0) bad++;
  }
  return bad;
}

/** @brief Enclosed volume by the divergence theorem. */
static double enclosed_volume(const MFTL_Object *obj)
{
  double vol = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    const double *a = MFTLVertexPos(obj, v[0]), *b = MFTLVertexPos(obj, v[1]), *c = MFTLVertexPos(obj, v[2]);
    vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return vol / 6.0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. ESTIMATORS
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_curvature_of_a_sphere(void)
{
  double prev = 1.0;
  for (int r = 2; r <= 4; ++r)
  {
    MFTL_Object *obj = make_ellipsoid(r, 2.0, 2.0, 2.0);
    const int n = obj->verts.n;
    double *H = malloc((size_t) n * sizeof(double)), *K = malloc((size_t) n * sizeof(double));
    MFTLComputeVertexCurvatures(obj, H, K);
    double eh = 0.0, ek = 0.0;
    for (int v = 0; v < n; ++v)
    {
      eh = fmax(eh, fabs(H[v] - 0.5) / 0.5);
      ek = fmax(ek, fabs(K[v] - 0.25) / 0.25);
    }
    printf("\n    refine %d: max rel error H %.2e, K %.2e", r, eh, ek);
    TEST_ASSERT(eh < prev);  // converges
    prev = eh;
    if (r == 4) TEST_ASSERT(eh < 0.02 && ek < 0.05);
    free(H);
    free(K);
  }
  return 0;
}

static int test_curvature_of_an_ellipsoid(void)
{
  const double a = 1.0, b = 0.5, c = 0.5;
  double prev = 1.0;
  for (int r = 3; r <= 5; ++r)
  {
    MFTL_Object *obj = make_ellipsoid(r, a, b, c);
    const int n = obj->verts.n;
    double *H = malloc((size_t) n * sizeof(double)), *K = malloc((size_t) n * sizeof(double));
    MFTLComputeVertexCurvatures(obj, H, K);
    double err = 0.0;
    for (int v = 0; v < n; ++v)
    {
      double He, Ke;
      ellipsoid_curvature(MFTLVertexPos(obj, v), a, b, c, &He, &Ke);
      const double km = H[v] + sqrt(fmax(H[v] * H[v] - K[v], 0.0));
      const double ke = He + sqrt(fmax(He * He - Ke, 0.0));
      err += (km - ke) * (km - ke) / (ke * ke);
    }
    err = sqrt(err / n);
    printf("\n    refine %d: rms rel error of kappa_max %.2e", r, err);
    TEST_ASSERT(err < prev);
    prev = err;
    free(H);
    free(K);
  }
  TEST_ASSERT(prev < 0.05);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. CURVATURE SIZING
 * ═══════════════════════════════════════════════════════════════════════════ */

static MFTL_RemeshParameters curvature_params(double eps, double lmin, double lmax, int iterations)
{
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, lmax, iterations, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  MFTLRemeshUseCurvatureSizing(&p, eps, lmin, lmax);
  return p;
}

static int test_the_sizing_formula(void)
{
  const MFTL_RemeshParameters p = curvature_params(0.01, 0.02, 0.4, 1);
  TEST_ASSERT(MFTLRemeshCurvatureSizing(0.0, &p) == 0.4);
  TEST_ASSERT(MFTLRemeshCurvatureSizing(-3.0, &p) == 0.4);
  TEST_ASSERT(MFTLRemeshCurvatureSizing(NAN, &p) == 0.4);
  TEST_ASSERT(MFTLRemeshCurvatureSizing(2.0 / 0.01, &p) == 0.02);  // the formula's L^2 is zero here
  TEST_ASSERT(MFTLRemeshCurvatureSizing(1e30, &p) == 0.02);
  TEST_ASSERT(MFTLRemeshCurvatureSizing(INFINITY, &p) == 0.02);

  double prev = 1e9;
  for (double k = 0.01; k < 1e4; k *= 1.3)
  {
    const double L = MFTLRemeshCurvatureSizing(k, &p);
    TEST_ASSERT(L == L && L >= 0.02 && L <= 0.4 && L <= prev);
    prev = L;
    const double ref = sqrt(6.0 * 0.01 / k - 3.0 * 0.01 * 0.01);
    if (ref > 0.02 * 1.000001 && ref < 0.4 * 0.999999) TEST_ASSERT(fabs(L - ref) <= 1e-14 * ref);
  }
  return 0;
}

static int test_a_sphere_is_sized_uniformly(void)
{
  MFTL_Object *obj = make_ellipsoid(4, 1.0, 1.0, 1.0);
  const MFTL_RemeshParameters p = curvature_params(0.002, 0.01, 0.5, 1);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshPrepareSizing(obj, &p));
  TEST_ASSERT(obj->verts.has_sizing && MFTLObjectPoolsAreConsistent(obj));
  const double L = MFTLRemeshCurvatureSizing(1.0, &p);
  double worst = 0.0;
  for (int v = 0; v < obj->verts.n; ++v) worst = fmax(worst, fabs(obj->verts.sizing[v] - L) / L);
  printf("\n    L(1/R) = %.5f, worst deviation %.2e", L, worst);
  TEST_ASSERT(worst < 0.02);
  return 0;
}

static int test_an_open_flat_patch_is_sized_at_the_coarsest(void)
{
  MFTL_Object *obj = make_patch(6, 6, 1.0, 1.0);
  const MFTL_RemeshParameters p = curvature_params(0.002, 0.01, 0.3, 1);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshPrepareSizing(obj, &p));
  /* Interior: no curvature. Rim: the mean of its interior neighbours, not the
   * half-fan's meaningless "curvature". */
  for (int v = 0; v < obj->verts.n; ++v) TEST_ASSERT(obj->verts.sizing[v] == 0.3);
  return 0;
}

static int test_an_invalid_block_is_refused(void)
{
  MFTL_Object *obj = make_ellipsoid(1, 1.0, 1.0, 1.0);
  MFTL_RemeshParameters p = curvature_params(0.0, 0.01, 0.5, 1);
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshPrepareSizing(obj, &p));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshObject(obj, &p, MFTL_NULL));
  p = curvature_params(0.01, 0.6, 0.5, 1);  // L_min > L_max
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshPrepareSizing(obj, &p));
  p = curvature_params(0.01, 0.1, 0.5, 1);
  p.smooth_mode = MFTL_SMOOTH_VERTEX_BASED;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshPrepareSizing(obj, &p));
  return 0;
}

/**
 * @brief The sign of the smoother's density: on a flat grid with targets
 *        growing in x, the interior vertices move towards SMALL x, where the
 *        mesh is to be finer. The printed Eq. 6 of Dunyach et al. (weight
 *        |t| L) would move them the other way; uniform weights do not move a
 *        regular grid at all.
 */
static int test_the_smoother_moves_towards_the_fine_side(void)
{
  MFTL_Object *obj = make_patch(8, 8, 1.0, 1.0);
  MFTLObjectBuildDerived(obj);
  MFTLDetectFeatures(obj, 180.0, 45.0, MFTL_NULL);
  MFTLObjectEnableSizing(obj);
  for (int v = 0; v < obj->verts.n; ++v) obj->verts.sizing[v] = 0.05 + 0.5 * MFTLVertexPos(obj, v)[0];

  double *before = malloc(3 * (size_t) obj->verts.n * sizeof(double));
  memcpy(before, obj->verts.pos, 3 * (size_t) obj->verts.n * sizeof(double));

  MFTL_RemeshParameters u;
  MFTLInitRemeshParams(&u, MFTL_REMESH_ISOTROPIC, 0.125, 1, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  MFTLSmoothTangential(obj, &u, 1);
  double moved = 0.0;
  for (int i = 0; i < 3 * obj->verts.n; ++i) moved = fmax(moved, fabs(obj->verts.pos[i] - before[i]));
  TEST_ASSERT(moved < 1e-12);

  const MFTL_RemeshParameters p = curvature_params(0.01, 0.01, 1.0, 1);
  MFTLSmoothTangential(obj, &p, 1);
  int n_interior = 0, n_left = 0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    if (obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER)) continue;
    n_interior++;
    if (obj->verts.pos[3 * v] < before[3 * v] - 1e-6) n_left++;
    TEST_ASSERT(fabs(obj->verts.pos[3 * v + 2]) < 1e-15);  // stays in the plane
  }
  TEST_ASSERT(n_interior == 49 && n_left == n_interior);
  free(before);
  return 0;
}

static int test_curvature_mode_on_a_sphere_is_uniform_mode(void)
{
  /* L(1/R) is the target both runs aim at. */
  MFTL_RemeshParameters c = curvature_params(0.004, 0.02, 1.0, 10);
  const double L = MFTLRemeshCurvatureSizing(1.0, &c);
  MFTL_RemeshParameters u;
  MFTLInitRemeshParams(&u, MFTL_REMESH_ISOTROPIC, L, 10, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);

  MFTL_Object *a = make_ellipsoid(3, 1.0, 1.0, 1.0);
  MFTL_Object *b = make_ellipsoid(3, 1.0, 1.0, 1.0);
  MFTL_RemeshStats sa, sb;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(a, &c, &sa));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(b, &u, &sb));
  printf("\n    L = %.4f: curvature %d tris, uniform %d tris", L, a->tris.n, b->tris.n);
  TEST_ASSERT(fabs((double) a->tris.n - b->tris.n) < 0.1 * b->tris.n);
  TEST_ASSERT(inv_mesh_is_sound(a, 0) && inv_mesh_is_closed(a));
  TEST_ASSERT_EQUAL_INT(0, count_inverted(a));
  return 0;
}

static int test_curvature_mode_refines_where_curved(void)
{
  const double a = 1.0, b = 0.4;
  /* Tip kappa_max = a / b^2 = 6.25, waist kappa_max = 1 / b = 2.5. */
  MFTL_RemeshParameters c = curvature_params(0.002, 0.01, 1.0, 12);
  const double L_tip = MFTLRemeshCurvatureSizing(a / (b * b), &c);
  const double L_waist = MFTLRemeshCurvatureSizing(1.0 / b, &c);
  MFTL_RemeshParameters u;
  MFTLInitRemeshParams(&u, MFTL_REMESH_ISOTROPIC, sqrt(L_tip * L_waist), 12, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);

  MFTL_Object *adapt = make_ellipsoid(3, a, b, b);
  MFTL_Object *unif = make_ellipsoid(3, a, b, b);
  const double v0 = enclosed_volume(adapt);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(adapt, &c, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(unif, &u, MFTL_NULL));

  int nt, nw;
  const double ta = mean_edge_in_band(adapt, 0.85, 2.0, &nt), wa = mean_edge_in_band(adapt, 0.0, 0.2, &nw);
  const double tu = mean_edge_in_band(unif, 0.85, 2.0, MFTL_NULL), wu = mean_edge_in_band(unif, 0.0, 0.2, MFTL_NULL);
  printf("\n    predicted tip/waist %.3f; curvature %.3f (%.4f/%.4f, %d/%d edges); uniform %.3f; %d vs %d tris; dV %.2e", L_tip / L_waist, ta / wa, ta, wa,
         nt, nw, tu / wu, adapt->tris.n, unif->tris.n, enclosed_volume(adapt) / v0 - 1.0);
  TEST_ASSERT(ta / wa < 0.8 * (tu / wu));
  TEST_ASSERT(fabs(ta / wa - L_tip / L_waist) < 0.2 * L_tip / L_waist);
  TEST_ASSERT(inv_mesh_is_sound(adapt, 0) && inv_mesh_is_closed(adapt));
  TEST_ASSERT_EQUAL_INT(0, count_inverted(adapt));
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. LENGTH UNIT
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Remeshes a scrambled ellipsoid at unit scale and at 2^-20 under curvature sizing; bit-for-bit equal up to the scale. */
static int test_curvature_mode_is_independent_of_the_length_unit(void)
{
  const double s = ldexp(1.0, -20);
  MFTL_Object *unit = make_ellipsoid(3, 1.0, 0.6, 0.6);
  unsigned seed = 24681357u;
  for (int v = 0; v < unit->verts.n; ++v)
  {
    double *q = MFTLVertexPosMut(unit, v);
    for (int k = 0; k < 3; ++k)
    {
      seed = seed * 1103515245u + 12345u;
      q[k] *= 1.0 + 0.08 * ((double) ((seed >> 16) & 0x7fff) / 32767.0 - 0.5);
    }
  }
  MFTLComputeTriangleGeometry(unit, unit->tris.n);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0 * s, -4.0 * s, -4.0 * s}, hi[3] = {4.0 * s, 4.0 * s, 4.0 * s};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *small_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, periodic, 1);
  MFTL_Object *small = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(small_ens, small);
  double centre[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(small, centre, s, 3);
  for (int i = 0; i < 3 * unit->verts.n; ++i) small->verts.pos[i] = unit->verts.pos[i] * s;
  MFTLComputeTriangleGeometry(small, small->tris.n);

  const double target = MFTLMeanEdgeLength(unit);
  MFTL_RemeshParameters pu, ps;
  MFTLInitRemeshParams(&pu, MFTL_REMESH_ISOTROPIC, target, 8, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  MFTLInitRemeshParams(&ps, MFTL_REMESH_ISOTROPIC, target * s, 8, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  MFTLRemeshUseCurvatureSizing(&pu, 0.02 * target, 0.25 * target, 1.5 * target);
  MFTLRemeshUseCurvatureSizing(&ps, 0.02 * target * s, 0.25 * target * s, 1.5 * target * s);
  pu.smooth_iterations = ps.smooth_iterations = 2;

  MFTL_RemeshStats su, ss;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(unit, &pu, &su));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(small, &ps, &ss));
  printf("\n    unit: %d/%d/%d -> %d tris; x2^-20: %d/%d/%d -> %d tris", su.n_split, su.n_collapse, su.n_flip, unit->tris.n, ss.n_split, ss.n_collapse,
         ss.n_flip, small->tris.n);
  TEST_ASSERT(su.n_split + su.n_collapse > 0 && su.n_flip > 0);
  TEST_ASSERT_EQUAL_INT(su.n_split, ss.n_split);
  TEST_ASSERT_EQUAL_INT(su.n_collapse, ss.n_collapse);
  TEST_ASSERT_EQUAL_INT(su.n_flip, ss.n_flip);
  TEST_ASSERT_EQUAL_INT(unit->tris.n, small->tris.n);
  int same = 1;
  for (int i = 0; i < 3 * unit->verts.n; ++i) same &= (small->verts.pos[i] == unit->verts.pos[i] * s);
  TEST_ASSERT(same);
  MFTLEnsembleDestroy(&small_ens);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  g_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, periodic, 64);
  if (!g_ens) MPI_Abort(MPI_COMM_WORLD, 1);

  printf("\n=== 1. Estimators ===\n");
  RUN_TEST(test_curvature_of_a_sphere);
  RUN_TEST(test_curvature_of_an_ellipsoid);

  printf("\n=== 2. Curvature sizing ===\n");
  RUN_TEST(test_the_sizing_formula);
  RUN_TEST(test_a_sphere_is_sized_uniformly);
  RUN_TEST(test_an_open_flat_patch_is_sized_at_the_coarsest);
  RUN_TEST(test_an_invalid_block_is_refused);
  RUN_TEST(test_the_smoother_moves_towards_the_fine_side);
  RUN_TEST(test_curvature_mode_on_a_sphere_is_uniform_mode);
  RUN_TEST(test_curvature_mode_refines_where_curved);

  printf("\n=== 3. Length unit ===\n");
  RUN_TEST(test_curvature_mode_is_independent_of_the_length_unit);

  TEST_REPORT();
  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
