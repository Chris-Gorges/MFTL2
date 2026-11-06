/**
 * @file test_remeshing_placement.c
 * @brief The least-squares solver, the paraboloid placement of split and
 *        collapse vertices, the volume-preserving collapse, and the Phong
 *        back-projection of the smoother.
 *
 * The references share nothing with the code under test: exact solutions of
 * consistent systems, the unit sphere's own radius, an enclosed volume summed
 * by the test over the whole mesh, and positions the test scales itself.
 * Serial; the distributed path is covered by the LeVeque conservation CTests
 * with the options on.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

static MFTL_Object *make_sphere(int refinements)
{
  MFTL_Object *obj = MFTLObjectCreate(g_ens->n_local_objects);
  MFTLEnsembleAddObject(g_ens, obj);
  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, refinements);
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief A reproducible number in [-1, 1]. */
static double rnd(unsigned *seed)
{
  *seed = *seed * 1103515245u + 12345u;
  return 2.0 * (double) ((*seed >> 8) & 0xffffff) / (double) 0xffffff - 1.0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. THE SOLVER
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A consistent full-rank system is solved to round-off, and an inconsistent one satisfies the normal equations. */
static int test_qrcp_solves_full_rank_systems(void)
{
  unsigned seed = 4711u;
  const int m = 40, n = 6;
  double A[6 * 40], A0[6 * 40], b[40], b0[40], x[6], xt[6];
  for (int j = 0; j < n; ++j) xt[j] = rnd(&seed);
  for (int i = 0; i < m * n; ++i) A0[i] = rnd(&seed);
  for (int i = 0; i < m; ++i)
  {
    b0[i] = 0.0;
    for (int j = 0; j < n; ++j) b0[i] += A0[j * m + i] * xt[j];
  }
  memcpy(A, A0, sizeof A);
  memcpy(b, b0, sizeof b);
  TEST_ASSERT_EQUAL_INT(6, MFTLLeastSquaresQRCP(m, n, 3, A, b, 1e-3, x, MFTL_NULL));
  double err = 0.0;
  for (int j = 0; j < n; ++j) err = fmax(err, fabs(x[j] - xt[j]));
  TEST_ASSERT(err < 1e-13);

  /* Inconsistent: A^T (A x - b) = 0. */
  for (int i = 0; i < m; ++i) b0[i] += 0.1 * rnd(&seed);
  memcpy(A, A0, sizeof A);
  memcpy(b, b0, sizeof b);
  MFTLLeastSquaresQRCP(m, n, 3, A, b, 1e-3, x, MFTL_NULL);
  double worst = 0.0, scale = 0.0;
  for (int j = 0; j < n; ++j)
  {
    double g = 0.0;
    for (int i = 0; i < m; ++i)
    {
      double r = -b0[i];
      for (int k = 0; k < n; ++k) r += A0[k * m + i] * x[k];
      g += A0[j * m + i] * r;
      scale += fabs(A0[j * m + i] * b0[i]);
    }
    worst = fmax(worst, fabs(g));
  }
  TEST_ASSERT(worst < 1e-13 * scale);
  return 0;
}

/**
 * @brief Samples on a circle around the origin cannot separate the height from
 *        the curvature (x^2 + y^2 = 1). The grouped pivoting keeps the height and
 *        drops a curvature term: the plane through the data, not the
 *        minimum-norm mixture (which would put c0 at 2/3).
 */
static int test_qrcp_drops_curvature_it_cannot_determine(void)
{
  double A[6 * 8], b[8], x[6];
  unsigned char kept[6];
  for (int i = 0; i < 8; ++i)
  {
    const double t = 2.0 * MFTL_PI * i / 8.0, cx = cos(t), sy = sin(t);
    const double row[6] = {1.0, cx, sy, cx * cx, cx * sy, sy * sy};
    for (int c = 0; c < 6; ++c) A[c * 8 + i] = row[c];
    b[i] = 1.0;
  }
  TEST_ASSERT_EQUAL_INT(5, MFTLLeastSquaresQRCP(8, 6, 3, A, b, 1e-3, x, kept));
  TEST_ASSERT(kept[0] && kept[1] && kept[2]);
  TEST_ASSERT(fabs(x[0] - 1.0) < 1e-13);
  TEST_ASSERT(fabs(x[3]) < 1e-13 && fabs(x[5]) < 1e-13);
  return 0;
}

/** @brief Scaling the whole system by a power of two changes no bit of the solution. */
static int test_qrcp_is_independent_of_the_scale(void)
{
  unsigned seed = 99u;
  const int m = 25;
  double A[6 * 25], As[6 * 25], b[25], bs[25], x[6], xs[6];
  for (int i = 0; i < 6 * m; ++i) A[i] = rnd(&seed);
  for (int i = 0; i < m; ++i) b[i] = rnd(&seed);
  const double s = ldexp(1.0, -40);
  for (int i = 0; i < 6 * m; ++i) As[i] = A[i] * s;
  for (int i = 0; i < m; ++i) bs[i] = b[i] * s;
  MFTLLeastSquaresQRCP(m, 6, 3, A, b, 1e-3, x, MFTL_NULL);
  MFTLLeastSquaresQRCP(m, 6, 3, As, bs, 1e-3, xs, MFTL_NULL);
  for (int j = 0; j < 6; ++j) TEST_ASSERT(x[j] == xs[j]);
  return 0;
}

/** @brief Heights, and heights with slopes, sampled from a paraboloid give back its coefficients. */
static int test_the_fit_recovers_a_paraboloid(void)
{
  const double c_true[6] = {0.3, -0.2, 0.1, 0.5, -0.25, 0.15};
  double pts[3 * 20], sl[2 * 20], w[20];
  unsigned seed = 7u;
  for (int j = 0; j < 20; ++j)
  {
    const double x = 1.5 * rnd(&seed), y = 1.5 * rnd(&seed);
    pts[3 * j] = x;
    pts[3 * j + 1] = y;
    pts[3 * j + 2] = c_true[0] + c_true[1] * x + c_true[2] * y + c_true[3] * x * x + c_true[4] * x * y + c_true[5] * y * y;
    sl[2 * j] = c_true[1] + 2.0 * c_true[3] * x + c_true[4] * y;
    sl[2 * j + 1] = c_true[2] + c_true[4] * x + 2.0 * c_true[5] * y;
    w[j] = 0.2 + 0.8 * fabs(rnd(&seed));
  }
  double c[6];
  TEST_ASSERT_EQUAL_INT(6, MFTLFitHeightParaboloid(20, pts, MFTL_NULL, MFTL_NULL, w, 1e-3, c, MFTL_NULL));
  for (int k = 0; k < 6; ++k) TEST_ASSERT(fabs(c[k] - c_true[k]) < 1e-12);
  TEST_ASSERT_EQUAL_INT(6, MFTLFitHeightParaboloid(20, pts, sl, MFTL_NULL, w, 1e-3, c, MFTL_NULL));
  for (int k = 0; k < 6; ++k) TEST_ASSERT(fabs(c[k] - c_true[k]) < 1e-12);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. PLACEMENT
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Splits every edge of a unit icosphere once and returns the mean
 *        |1 - |x|| of the vertices the splits created.
 */
static double split_all_error(int refinements, MFTL_RemeshPlacement split, MFTL_RemeshPlacement collapse)
{
  MFTL_Object *obj = make_sphere(refinements);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 1e-3, 1, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);  // every edge is too long
  p.split_placement = split;
  p.collapse_placement = collapse;
  MFTLDetectFeatures(obj, p.feature_angle_threshold, p.feature_corner_angle, MFTL_NULL);

  const int n_v0 = obj->verts.n;
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h)
    if (MFTLSplitApproved(obj, h, &p)) MFTLSplitEdge(obj, h, &p);

  double sum = 0.0;
  for (int v = n_v0; v < obj->verts.n; ++v) sum += fabs(1.0 - MFTLVectorNorm(MFTLVertexPos(obj, v)));
  return sum / (double) MFTL_MAX(obj->verts.n - n_v0, 1);
}

/**
 * @brief Split points on a sphere: the chord midpoint is off by the sagitta,
 *        O(h^2); the fitted points converge at least two orders faster.
 */
static int test_split_points_land_on_the_sphere(void)
{
  MFTLRemeshResetFitCounters();
  double e[3][3];
  for (int r = 2; r <= 4; ++r)
  {
    e[r - 2][0] = split_all_error(r, MFTL_REMESH_PLACE_MIDPOINT, MFTL_REMESH_PLACE_MIDPOINT);
    e[r - 2][1] = split_all_error(r, MFTL_REMESH_PLACE_PARABOLOID, MFTL_REMESH_PLACE_MIDPOINT);
    e[r - 2][2] = split_all_error(r, MFTL_REMESH_PLACE_PARABOLOID_NORMALS, MFTL_REMESH_PLACE_MIDPOINT);
    printf("\n    refine %d: mean |1-r| midpoint %.3e, paraboloid %.3e, with normals %.3e", r, e[r - 2][0], e[r - 2][1], e[r - 2][2]);
  }
  MFTL_RemeshFitCounters fc;
  MFTLRemeshGetFitCounters(&fc);
  printf("\n    %ld fits, %ld fallbacks", fc.n_fits, fc.n_fallbacks);
  for (int k = 1; k <= 2; ++k)
  {
    TEST_ASSERT(e[2][k] < 0.05 * e[2][0]);
    TEST_ASSERT(e[1][k] / e[2][k] > 8.0);  // fourth order would be 16
  }
  TEST_ASSERT(fc.n_fallbacks == 0);
  return 0;
}

/** @brief Enclosed volume of a closed serial mesh, summed from the origin over its live triangles. */
static double enclosed_volume(const MFTL_Object *obj)
{
  double six_v = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    if (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) continue;
    int c[3];
    MFTLGetTriangleVertices(obj, t, c);
    const double *x0 = MFTLVertexPos(obj, c[0]), *x1 = MFTLVertexPos(obj, c[1]), *x2 = MFTLVertexPos(obj, c[2]);
    six_v += x0[0] * (x1[1] * x2[2] - x1[2] * x2[1]) + x0[1] * (x1[2] * x2[0] - x1[0] * x2[2]) + x0[2] * (x1[0] * x2[1] - x1[1] * x2[0]);
  }
  return six_v / 6.0;
}

/** @brief A unit icosphere squeezed into a 1 : 0.6 : 0.6 ellipsoid and jittered by 4 %. */
static MFTL_Object *make_rough_ellipsoid(int refinements)
{
  MFTL_Object *obj = make_sphere(refinements);
  unsigned seed = 13579u;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *q = MFTLVertexPosMut(obj, v);
    q[1] *= 0.6;
    q[2] *= 0.6;
    for (int k = 0; k < 3; ++k) q[k] *= 1.0 + 0.04 * rnd(&seed);
  }
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief What a run of single collapses did. */
typedef struct
{
  int n;  ///< Collapses performed.
  double max_dv;  ///< Largest |change of the enclosed volume| of one collapse.
  double max_off_mid;  ///< Largest distance of a survivor from its edge's midpoint.
  double mean_off_sphere;  ///< Mean |1 - |survivor||.
} CollapseRun;

/**
 * @brief Collapses up to @p max_ops approved edges of @p obj one at a time
 *        (every edge is short under a target three mean edges long), measuring
 *        each against the whole mesh's volume before and after.
 */
static CollapseRun collapse_run(MFTL_Object *obj, const MFTL_RemeshParameters *p, int max_ops)
{
  CollapseRun r = {0, 0.0, 0.0, 0.0};
  MFTLDetectFeatures(obj, p->feature_angle_threshold, p->feature_corner_angle, MFTL_NULL);
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he && r.n < max_ops; ++h)
  {
    if (obj->tris.rec[h / 3].flags & MFTL_FLAG_DEAD) continue;
    if (!MFTLCollapseApproved(obj, h, p, MFTL_NULL)) continue;
    const int b = MFTLHeTarget(obj, h);
    const double *pa = MFTLVertexPos(obj, MFTLHeOrigin(obj, h)), *pb = MFTLVertexPos(obj, b);
    const double mid[3] = {0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1]), 0.5 * (pa[2] + pb[2])};
    const double v0 = enclosed_volume(obj);
    if (MFTLCollapseEdge(obj, h, p) != MFTL_SUCCESS) continue;
    const double *q = MFTLVertexPos(obj, b);
    const double d[3] = {q[0] - mid[0], q[1] - mid[1], q[2] - mid[2]};
    r.max_dv = fmax(r.max_dv, fabs(enclosed_volume(obj) - v0));
    r.max_off_mid = fmax(r.max_off_mid, MFTLVectorNorm(d));
    r.mean_off_sphere += fabs(1.0 - MFTLVectorNorm(q));
    r.n++;
  }
  r.mean_off_sphere /= (double) MFTL_MAX(r.n, 1);
  MFTLRemeshCompactObject(obj, MFTL_NULL);
  return r;
}

/** @brief A coarsening parameter block: every edge is shorter than `min_edge_length`. */
static MFTL_RemeshParameters coarsening(const MFTL_Object *obj, MFTL_RemeshPlacement split, MFTL_RemeshPlacement collapse, bool keep_volume)
{
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 3.0 * MFTLMeanEdgeLength(obj), 1, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.split_placement = split;
  p.collapse_placement = collapse;
  p.collapse_preserves_volume = keep_volume;
  return p;
}

/**
 * @brief The two placements are separate switches: a paraboloid collapse
 *        placement leaves every split at the midpoint, to the bit, and a
 *        paraboloid split placement leaves every collapse at the midpoint.
 */
static int test_split_and_collapse_placements_are_separate(void)
{
  const double e_mid = split_all_error(3, MFTL_REMESH_PLACE_MIDPOINT, MFTL_REMESH_PLACE_MIDPOINT);
  const double e_cp = split_all_error(3, MFTL_REMESH_PLACE_MIDPOINT, MFTL_REMESH_PLACE_PARABOLOID);
  const double e_sp = split_all_error(3, MFTL_REMESH_PLACE_PARABOLOID, MFTL_REMESH_PLACE_MIDPOINT);
  printf("\n    splits, mean |1-r|: both midpoint %.3e, paraboloid collapses %.3e, paraboloid splits %.3e", e_mid, e_cp, e_sp);
  TEST_ASSERT(e_cp == e_mid);
  TEST_ASSERT(e_sp < 0.05 * e_mid);

  MFTL_Object *obj = make_sphere(3);
  MFTL_RemeshParameters p = coarsening(obj, MFTL_REMESH_PLACE_PARABOLOID, MFTL_REMESH_PLACE_MIDPOINT, MFTL_FALSE);
  const CollapseRun mid = collapse_run(obj, &p, 60);
  obj = make_sphere(3);
  p = coarsening(obj, MFTL_REMESH_PLACE_MIDPOINT, MFTL_REMESH_PLACE_PARABOLOID, MFTL_FALSE);
  const CollapseRun fit = collapse_run(obj, &p, 60);
  printf("\n    collapses, mean |1-r|: paraboloid splits %.3e (%d, %.1e off the midpoint), paraboloid collapses %.3e (%d)", mid.mean_off_sphere, mid.n,
         mid.max_off_mid, fit.mean_off_sphere, fit.n);
  TEST_ASSERT(mid.n >= 30 && fit.n >= 30);
  TEST_ASSERT(mid.max_off_mid < 1e-15);
  TEST_ASSERT(fit.mean_off_sphere < 0.1 * mid.mean_off_sphere);  // 14x measured: the merged vertex's edges are long
  return 0;
}

/**
 * @brief Under `collapse_preserves_volume` every single collapse of a rough
 *        ellipsoid keeps the enclosed volume to round-off, from the midpoint
 *        and from either fitted point; without it each loses volume.
 */
static int test_a_volume_preserving_collapse_keeps_the_volume(void)
{
  static const struct
  {
    MFTL_RemeshPlacement place;
    bool keep;
    const char *name;
  } C[] = {{MFTL_REMESH_PLACE_MIDPOINT, MFTL_FALSE, "midpoint"},
           {MFTL_REMESH_PLACE_PARABOLOID, MFTL_FALSE, "paraboloid"},
           {MFTL_REMESH_PLACE_MIDPOINT, MFTL_TRUE, "midpoint, volume kept"},
           {MFTL_REMESH_PLACE_PARABOLOID, MFTL_TRUE, "paraboloid, volume kept"},
           {MFTL_REMESH_PLACE_PARABOLOID_NORMALS, MFTL_TRUE, "paraboloid with normals, volume kept"}};
  double v_ref = 0.0, lost[2] = {0.0, 0.0};
  for (size_t c = 0; c < sizeof C / sizeof C[0]; ++c)
  {
    MFTL_Object *obj = make_rough_ellipsoid(3);
    v_ref = enclosed_volume(obj);
    const MFTL_RemeshParameters p = coarsening(obj, MFTL_REMESH_PLACE_MIDPOINT, C[c].place, C[c].keep);
    const CollapseRun r = collapse_run(obj, &p, 80);
    printf("\n    %-38s %d collapses, largest |dV| %.2e of V", C[c].name, r.n, r.max_dv / v_ref);
    TEST_ASSERT(r.n >= 40);
    TEST_ASSERT(inv_mesh_is_sound(obj, 0) && inv_mesh_is_closed(obj));
    if (C[c].keep)
      TEST_ASSERT(r.max_dv < 1e-13 * v_ref);
    else
      lost[c] = r.max_dv;
  }
  TEST_ASSERT(lost[0] > 1e-5 * v_ref && lost[1] > 1e-5 * v_ref);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. PROJECTION
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Moves every vertex of a unit sphere tangentially by up to @p amp of an edge, back onto the sphere. */
static void jitter_on_sphere(MFTL_Object *obj, double amp)
{
  unsigned seed = 1234u;
  const double h = MFTLMeanEdgeLength(obj);
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *p = MFTLVertexPosMut(obj, v);
    for (int k = 0; k < 3; ++k) p[k] += amp * h * rnd(&seed);
    MFTLNormalizeVector(p);
  }
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
}

/**
 * @brief Smoothing a jittered sphere, twenty calls of three sweeps each, every
 *        call projecting onto the mesh it received.
 *
 * Without projection a tangential move leaves the sphere by the tangent plane's
 * offset, outwards, and the next call starts from there. Projected onto the
 * Phong surface of each call's input the vertices stay near the sphere: closer
 * after one call (angle-weighted normals are first-order on an irregular mesh,
 * so the lift is second-order with a smaller constant), and after twenty,
 * because each call corrects from where the previous one left the mesh. (The
 * enclosed volume is no measure here: relaxing a jittered mesh changes the
 * inscribed polyhedron's volume with every vertex still on the sphere.)
 */
static int test_projected_smoothing_stays_on_the_sphere(void)
{
  double dev[3][2], drift[3][2];
  for (int r = 2; r <= 4; ++r)
    for (int proj = 0; proj < 2; ++proj)
    {
      MFTL_Object *obj = make_sphere(r);
      jitter_on_sphere(obj, 0.2);
      MFTL_RemeshParameters p;
      MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, MFTLMeanEdgeLength(obj), 1, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
      p.projection = proj ? MFTL_REMESH_PROJECT_PHONG : MFTL_REMESH_PROJECT_NONE;
      for (int call = 0; call < 20; ++call)
      {
        MFTLDetectFeatures(obj, p.feature_angle_threshold, p.feature_corner_angle, MFTL_NULL);
        MFTLSmoothTangential(obj, &p, 3);
        if (call == 0)
        {
          double sum = 0.0;
          for (int v = 0; v < obj->verts.n; ++v) sum += fabs(1.0 - MFTLVectorNorm(MFTLVertexPos(obj, v)));
          dev[r - 2][proj] = sum / obj->verts.n;
        }
      }
      double sum = 0.0;
      for (int v = 0; v < obj->verts.n; ++v) sum += fabs(1.0 - MFTLVectorNorm(MFTLVertexPos(obj, v)));
      drift[r - 2][proj] = sum / obj->verts.n;
      TEST_ASSERT(inv_mesh_is_sound(obj, 0) && inv_mesh_is_closed(obj));
    }
  for (int r = 2; r <= 4; ++r)
    printf("\n    refine %d: mean |1-r| after one call: tangential %.3e, projected %.3e; after 20 calls: %.3e, %.3e", r, dev[r - 2][0], dev[r - 2][1],
           drift[r - 2][0], drift[r - 2][1]);
  for (int r = 0; r < 3; ++r)
  {
    TEST_ASSERT(dev[r][1] < 0.5 * dev[r][0]);
    TEST_ASSERT(drift[r][1] < 0.5 * drift[r][0]);
  }
  return 0;
}

/**
 * @brief The whole kernel with both options, at unit scale and at 2^-20:
 *        bit-for-bit the same mesh up to the scale. Any absolute threshold in
 *        the fit, the solver or the projection breaks it.
 */
static int test_placement_and_projection_are_independent_of_the_length_unit(void)
{
  const double s = ldexp(1.0, -20);
  MFTL_Object *unit = make_sphere(3);
  unsigned seed = 24681357u;
  for (int v = 0; v < unit->verts.n; ++v)
  {
    double *q = MFTLVertexPosMut(unit, v);
    q[0] *= 1.0;
    q[1] *= 0.6;
    q[2] *= 0.6;
    for (int k = 0; k < 3; ++k) q[k] *= 1.0 + 0.04 * rnd(&seed);
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

  const double target = 0.8 * MFTLMeanEdgeLength(unit);
  static const struct
  {
    MFTL_RemeshPlacement place;
    bool keep;
    const char *name;
  } V[] = {{MFTL_REMESH_PLACE_PARABOLOID, MFTL_FALSE, "positions"},
           {MFTL_REMESH_PLACE_PARABOLOID_NORMALS, MFTL_FALSE, "normals"},
           {MFTL_REMESH_PLACE_MIDPOINT, MFTL_TRUE, "midpoint, volume kept"},
           {MFTL_REMESH_PLACE_PARABOLOID, MFTL_TRUE, "positions, volume kept"}};
  for (size_t variant = 0; variant < sizeof V / sizeof V[0]; ++variant)
  {
    /* Alternate refining and coarsening, so every variant collapses too. */
    const double tv = (variant % 2) ? 1.5 * target : target;
    MFTL_RemeshParameters pu, ps;
    MFTLInitRemeshParams(&pu, MFTL_REMESH_ISOTROPIC, tv, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
    MFTLInitRemeshParams(&ps, MFTL_REMESH_ISOTROPIC, tv * s, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
    pu.split_placement = ps.split_placement = V[variant].place;
    pu.collapse_placement = ps.collapse_placement = V[variant].place;
    pu.collapse_preserves_volume = ps.collapse_preserves_volume = V[variant].keep;
    pu.projection = ps.projection = MFTL_REMESH_PROJECT_PHONG;
    pu.smooth_iterations = ps.smooth_iterations = 2;

    MFTL_RemeshStats su, ss;
    TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(unit, &pu, &su));
    TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(small, &ps, &ss));
    printf("\n    %s: %d/%d/%d ops, %d projected, %d missed", V[variant].name, su.n_split, su.n_collapse, su.n_flip, su.n_projected,
           su.n_projection_missed);
    TEST_ASSERT(su.n_split + su.n_collapse > 0 && su.n_projected > 0);
    TEST_ASSERT(!V[variant].keep || su.n_collapse > 0);
    TEST_ASSERT_EQUAL_INT(su.n_split, ss.n_split);
    TEST_ASSERT_EQUAL_INT(su.n_collapse, ss.n_collapse);
    TEST_ASSERT_EQUAL_INT(su.n_projected, ss.n_projected);
    TEST_ASSERT_EQUAL_INT(unit->verts.n, small->verts.n);
    int same = 1;
    for (int i = 0; i < 3 * unit->verts.n; ++i) same &= (small->verts.pos[i] == unit->verts.pos[i] * s);
    TEST_ASSERT(same);
    TEST_ASSERT(inv_mesh_is_sound(unit, 0) && inv_mesh_is_closed(unit));
  }
  MFTLEnsembleDestroy(&small_ens);
  return 0;
}

/** @brief An invalid block is refused before anything is touched. */
static int test_an_invalid_placement_is_refused(void)
{
  MFTL_Object *obj = make_sphere(2);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, MFTLMeanEdgeLength(obj), 1, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.collapse_placement = MFTL_REMESH_PLACE_PARABOLOID;
  p.fit_radius = 1.0;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshObject(obj, &p, MFTL_NULL));
  p.collapse_placement = MFTL_REMESH_PLACE_MIDPOINT;
  p.split_placement = MFTL_REMESH_PLACE_PARABOLOID;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshObject(obj, &p, MFTL_NULL));
  p.fit_radius = 3.0;
  p.split_placement = (MFTL_RemeshPlacement) 5;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshObject(obj, &p, MFTL_NULL));
  p.split_placement = MFTL_REMESH_PLACE_MIDPOINT;
  p.projection = (MFTL_RemeshProjection) 7;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRemeshObject(obj, &p, MFTL_NULL));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  g_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, periodic, 64);
  if (!g_ens) MPI_Abort(MPI_COMM_WORLD, 1);

  printf("\n=== 1. Least squares ===\n");
  RUN_TEST(test_qrcp_solves_full_rank_systems);
  RUN_TEST(test_qrcp_drops_curvature_it_cannot_determine);
  RUN_TEST(test_qrcp_is_independent_of_the_scale);
  RUN_TEST(test_the_fit_recovers_a_paraboloid);

  printf("\n=== 2. Placement ===\n");
  RUN_TEST(test_split_points_land_on_the_sphere);
  RUN_TEST(test_split_and_collapse_placements_are_separate);
  RUN_TEST(test_a_volume_preserving_collapse_keeps_the_volume);

  printf("\n=== 3. Projection ===\n");
  RUN_TEST(test_projected_smoothing_stays_on_the_sphere);
  RUN_TEST(test_placement_and_projection_are_independent_of_the_length_unit);
  RUN_TEST(test_an_invalid_placement_is_refused);

  TEST_REPORT();
  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
