/**
 * @file test_curvature.c
 * @brief Signed curvature (MFTLComputeVertexCurvature(), both methods), the
 *        fitted vertex normal (#MFTL_NORMAL_WEIGHT_FIT) and the identities that
 *        tie them to the surface-tension forces.
 *
 * Every reference is closed-form and shares nothing with the code under test:
 * the curvature of a sphere, the shape operator of an ellipsoid and of a torus
 * evaluated by the test, Gauss-Bonnet, a plane. The mixed area the identity
 * test needs is Meyer et al.'s Fig. 4 written out again here. What each test
 * is for:
 *
 *   values        kappa = 2/R, K = 1/R^2 and k1 = k2 = 1/R on a sphere, with the
 *                 SIGN of the orientation: reversed triangles give -2/R;
 *   convergence   the fit with fitted normals converges at second order on an
 *                 IRREGULAR sphere, where the cotangent formula does not
 *                 converge pointwise (the reason the fit exists), and the fitted
 *                 normals themselves converge at second order;
 *   shapes        an ellipsoid (curvatures and principal directions) and a
 *                 torus (both signs of kappa and K, directions along the
 *                 meridian);
 *   integrals     Gauss-Bonnet to round-off on a sphere and a torus;
 *   flat          zero on a plane, exactly;
 *   identity      Frenet vertex with triangle normals = -sigma kappa_COTAN
 *                 A_mixed n: Tryggvason's force IS Meyer's operator;
 *   Laplace       -sum F.n / sum A = 2 sigma / R exactly for that force (Euler's
 *                 theorem for the homogeneous area), at second order for the
 *                 forms with fitted vertex normals;
 *   units         a mesh scaled by 2^-20 gives every curvature times 2^20 and
 *                 every force times 2^-20, bit for bit, on a periodic seam;
 *   fallback      a vertex the fit cannot serve gets the cotangent values and
 *                 is counted; bad arguments are refused.
 *
 * Serial by construction; the distributed behaviour is in
 * test_surface_tension.c.
 */

#include <math.h>
#include <stdint.h>
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

static double noise(uint64_t id, uint64_t salt)
{
  uint64_t z = id * 0x9E3779B97F4A7C15ull + salt * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * ((double) (z >> 11) / 9007199254740992.0) - 1.0;
}

static double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

static void unit3(double v[3])
{
  const double m = sqrt(dot3(v, v));
  for (int k = 0; k < 3; ++k) v[k] /= m;
}

static MFTL_Object *fixture_new(void)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, obj);
  return obj;
}

static void fixture_drop(MFTL_Object *obj)
{
  MFTLEnsembleEraseLocalSlot(g_ens, MFTLEnsembleFindLocalSlot(g_ens, obj->global_id));
  MFTLObjectDestroy(&obj);
}

static void settle(MFTL_Object *obj)
{
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLObjectMarkChanged(obj);
}

/** @brief An icosphere of radius R at the origin; with @p jitter > 0 every vertex moved tangentially by up to jitter edges and put back on the sphere. */
static MFTL_Object *make_sphere(int level, double R, double jitter)
{
  MFTL_Object *obj = fixture_new();
  const double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, R, level);
  if (jitter > 0.0)
  {
    const double h = MFTLMeanEdgeLength(obj);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double *p = &obj->verts.pos[3 * v], n[3] = {p[0], p[1], p[2]}, t1[3], t2[3];
      unit3(n);
      const double ax[3] = {fabs(n[0]) < 0.6 ? 1.0 : 0.0, fabs(n[0]) < 0.6 ? 0.0 : 1.0, 0.0};
      t1[0] = n[1] * ax[2] - n[2] * ax[1], t1[1] = n[2] * ax[0] - n[0] * ax[2], t1[2] = n[0] * ax[1] - n[1] * ax[0];
      unit3(t1);
      t2[0] = n[1] * t1[2] - n[2] * t1[1], t2[1] = n[2] * t1[0] - n[0] * t1[2], t2[2] = n[0] * t1[1] - n[1] * t1[0];
      const double a = jitter * h * noise((uint64_t) v, 3), b = jitter * h * noise((uint64_t) v, 4);
      for (int k = 0; k < 3; ++k) p[k] += a * t1[k] + b * t2[k];
      unit3(p);
      for (int k = 0; k < 3; ++k) p[k] *= R;
    }
  }
  settle(obj);
  return obj;
}

/** @brief The torus (R0, r0) on an nu x nv grid, optionally jittered in (u, v) by up to jitter cells. */
static MFTL_Object *make_torus(int nv, double R0, double r0, double jitter)
{
  const int nu = (int) lround(nv * R0 / r0), n = nu * nv;
  double *pos = (double *) malloc((size_t) 3 * n * sizeof(double));
  int *tri = (int *) malloc((size_t) 6 * n * sizeof(int));
  for (int i = 0; i < nu; ++i)
    for (int j = 0; j < nv; ++j)
    {
      const int id = i * nv + j;
      const double u = 2.0 * MFTL_PI * (i + jitter * noise((uint64_t) id, 1)) / nu, w = 2.0 * MFTL_PI * (j + jitter * noise((uint64_t) id, 2)) / nv;
      pos[3 * id + 0] = (R0 + r0 * cos(w)) * cos(u);
      pos[3 * id + 1] = (R0 + r0 * cos(w)) * sin(u);
      pos[3 * id + 2] = r0 * sin(w);
    }
  int t = 0;
  for (int i = 0; i < nu; ++i)
    for (int j = 0; j < nv; ++j)
    {
      const int a = i * nv + j, b = ((i + 1) % nu) * nv + j, c = ((i + 1) % nu) * nv + (j + 1) % nv, d = i * nv + (j + 1) % nv;
      tri[3 * t + 0] = a, tri[3 * t + 1] = b, tri[3 * t + 2] = c, ++t;
      tri[3 * t + 0] = a, tri[3 * t + 1] = c, tri[3 * t + 2] = d, ++t;
    }
  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, n, tri, t);
  free(pos);
  free(tri);
  settle(obj);
  return obj;
}

/** @brief The torus reference at a point on it: outward normal, kappa, K, k1 (meridian, always the larger) and its direction. */
static void torus_reference(const double p[3], double R0, double r0, double n[3], double *kappa, double *gauss, double *k1, double e1[3])
{
  const double u = atan2(p[1], p[0]), w = atan2(p[2], sqrt(p[0] * p[0] + p[1] * p[1]) - R0);
  n[0] = cos(w) * cos(u), n[1] = cos(w) * sin(u), n[2] = sin(w);
  const double km = 1.0 / r0, kp = cos(w) / (R0 + r0 * cos(w));
  *kappa = km + kp;
  *gauss = km * kp;
  *k1 = km;
  e1[0] = -sin(w) * cos(u), e1[1] = -sin(w) * sin(u), e1[2] = cos(w);
}

/**
 * @brief The ellipsoid reference (semi-axes @p ax) at a point on it, from the
 *        shape operator P Hess(F) P / |grad F| of F = sum x_k^2 / a_k^2 - 1.
 */
static void ellipsoid_reference(const double p[3], const double ax[3], double *kappa, double *gauss, double *k1, double *k2, double e1[3])
{
  double g[3], n[3];
  for (int k = 0; k < 3; ++k) g[k] = 2.0 * p[k] / (ax[k] * ax[k]);
  const double gm = sqrt(dot3(g, g));
  for (int k = 0; k < 3; ++k) n[k] = g[k] / gm;
  int a = 0;
  for (int k = 1; k < 3; ++k)
    if (fabs(n[k]) < fabs(n[a])) a = k;
  double t1[3] = {-n[a] * n[0], -n[a] * n[1], -n[a] * n[2]}, t2[3];
  t1[a] += 1.0;
  unit3(t1);
  t2[0] = n[1] * t1[2] - n[2] * t1[1], t2[1] = n[2] * t1[0] - n[0] * t1[2], t2[2] = n[0] * t1[1] - n[1] * t1[0];
  const double *t[2] = {t1, t2};
  double B[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
    {
      double s = 0.0;
      for (int k = 0; k < 3; ++k) s += t[i][k] * (2.0 / (ax[k] * ax[k])) * t[j][k];
      B[i][j] = s / gm;
    }
  const double tr = B[0][0] + B[1][1], det = B[0][0] * B[1][1] - B[0][1] * B[1][0], d = sqrt(fmax(0.25 * tr * tr - det, 0.0));
  *kappa = tr;
  *gauss = det;
  *k1 = 0.5 * tr + d;
  *k2 = 0.5 * tr - d;
  const double phi = 0.5 * atan2(2.0 * B[0][1], B[0][0] - B[1][1]);
  for (int k = 0; k < 3; ++k) e1[k] = cos(phi) * t1[k] + sin(phi) * t2[k];
}

/** @brief Meyer et al.'s mixed area of every vertex (their Fig. 4), written out independently of the library. */
static double *mixed_areas(const MFTL_Object *obj)
{
  double *A = (double *) calloc((size_t) obj->verts.n, sizeof(double));
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    const double a = MFTLGetTriangleArea(obj, t);
    double d[3], l2[3];
    for (int k = 0; k < 3; ++k)
    {
      const double *p0 = T[k], *p1 = T[(k + 1) % 3], *p2 = T[(k + 2) % 3];
      const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      const double o[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
      d[k] = dot3(u, w);
      l2[k] = dot3(o, o);
    }
    for (int k = 0; k < 3; ++k)
    {
      if (d[0] >= 0.0 && d[1] >= 0.0 && d[2] >= 0.0)
        A[v[k]] += (l2[(k + 1) % 3] * d[(k + 1) % 3] + l2[(k + 2) % 3] * d[(k + 2) % 3]) / (16.0 * a);
      else
        A[v[k]] += (d[k] < 0.0) ? 0.5 * a : 0.25 * a;
    }
  }
  return A;
}

/** @brief Curvature records of every vertex; for FIT with the given normals first. */
static double *curvature(MFTL_Object *obj, MFTL_CurvatureMethod m, MFTL_NormalWeighting w, int ncomp, int *fallback)
{
  if (m == MFTL_CURVATURE_FIT) MFTLComputeVertexNormals(obj, w);
  double *rec = (double *) malloc((size_t) ncomp * obj->verts.n * sizeof(double));
  const int rc = MFTLComputeVertexCurvature(obj, m, ncomp, rec, fallback);
  MFTL_ASSERT(rc == MFTL_SUCCESS);
  (void) rc;
  return rec;
}

/** @brief max over vertices of |kappa - 2/R| R / 2. */
static double sphere_kappa_error(const MFTL_Object *obj, const double *rec, int ncomp, double R)
{
  double e = 0.0;
  for (int v = 0; v < obj->verts.n; ++v) e = fmax(e, fabs(rec[ncomp * v] - 2.0 / R) * R / 2.0);
  return e;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Values and signs
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_a_sphere_has_kappa_two_over_r_with_the_sign_of_its_orientation(void)
{
  const double R = 2.0;
  MFTL_Object *obj = make_sphere(4, R, 0.0);
  for (int m = 0; m < 2; ++m)
  {
    int fb = -1;
    double *rec = curvature(obj, (MFTL_CurvatureMethod) m, MFTL_NORMAL_WEIGHT_FIT, MFTL_CURVATURE_COMPONENTS, &fb);
    TEST_ASSERT_EQUAL_INT(0, fb);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const double *q = &rec[MFTL_CURVATURE_COMPONENTS * v];
      const double tol = (m == MFTL_CURVATURE_COTAN) ? 1e-5 : 1e-2;
      TEST_ASSERT(fabs(q[MFTL_CURVATURE_KAPPA] - 2.0 / R) <= tol * 2.0 / R);
      TEST_ASSERT(fabs(q[MFTL_CURVATURE_GAUSS] - 1.0 / (R * R)) <= 1e-2 / (R * R));
      TEST_ASSERT(fabs(q[MFTL_CURVATURE_K1] - 1.0 / R) <= 1e-2 / R && fabs(q[MFTL_CURVATURE_K2] - 1.0 / R) <= 1e-2 / R);
      TEST_ASSERT(q[MFTL_CURVATURE_K1] >= q[MFTL_CURVATURE_K2]);
      const double *d = &q[MFTL_CURVATURE_DIR1];
      TEST_ASSERT(fabs(dot3(d, d) - 1.0) <= 1e-12);  // a unit tangent even where it is arbitrary
      double n[3] = {obj->verts.pos[3 * v], obj->verts.pos[3 * v + 1], obj->verts.pos[3 * v + 2]};
      unit3(n);
      TEST_ASSERT(fabs(dot3(d, n)) <= 1e-2);
    }
    free(rec);
  }

  /* The same sphere wound the other way: its normals point inward, kappa < 0. */
  const int nv = obj->verts.n, nt = obj->tris.n;
  double *pos = (double *) malloc((size_t) 3 * nv * sizeof(double));
  int *tri = (int *) malloc((size_t) 3 * nt * sizeof(int));
  memcpy(pos, obj->verts.pos, (size_t) 3 * nv * sizeof(double));
  for (int t = 0; t < nt; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    tri[3 * t + 0] = v[0], tri[3 * t + 1] = v[2], tri[3 * t + 2] = v[1];
  }
  fixture_drop(obj);
  MFTL_Object *inv = fixture_new();
  MFTLBuildIndexedMesh(inv, pos, nv, tri, nt);
  settle(inv);
  for (int m = 0; m < 2; ++m)
  {
    double *rec = curvature(inv, (MFTL_CurvatureMethod) m, MFTL_NORMAL_WEIGHT_FIT, 4, MFTL_NULL);
    for (int v = 0; v < nv; ++v)
    {
      TEST_ASSERT(fabs(rec[4 * v + MFTL_CURVATURE_KAPPA] + 2.0 / R) <= 1e-2 * 2.0 / R);
      TEST_ASSERT(rec[4 * v + MFTL_CURVATURE_GAUSS] > 0.0);  // K does not depend on the orientation
      TEST_ASSERT(rec[4 * v + MFTL_CURVATURE_K1] < 0.0);
    }
    free(rec);
  }
  free(pos);
  free(tri);
  fixture_drop(inv);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Convergence on an irregular mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_the_fit_converges_where_the_cotangent_formula_does_not(void)
{
  double e_fit[6], e_cot[6], e_nfit[6], e_nangle[6], h[6];
  for (int l = 3; l <= 5; ++l)
  {
    MFTL_Object *obj = make_sphere(l, 1.0, 0.3);
    h[l] = MFTLMeanEdgeLength(obj);

    /* The normals: fitted against angle-weighted, worst vertex. */
    for (int k = 0; k < 2; ++k)
    {
      MFTLComputeVertexNormals(obj, k == 0 ? MFTL_NORMAL_WEIGHT_FIT : MFTL_NORMAL_WEIGHT_ANGLE);
      double e = 0.0;
      for (int v = 0; v < obj->verts.n; ++v)
      {
        double n[3] = {obj->verts.pos[3 * v], obj->verts.pos[3 * v + 1], obj->verts.pos[3 * v + 2]};
        unit3(n);
        const double *q = &obj->verts.normal[3 * v];
        const double d[3] = {q[0] - n[0], q[1] - n[1], q[2] - n[2]};
        e = fmax(e, sqrt(dot3(d, d)));  // the chord, never acos of a dot product
      }
      (k == 0 ? e_nfit : e_nangle)[l] = e;
    }

    double *rf = curvature(obj, MFTL_CURVATURE_FIT, MFTL_NORMAL_WEIGHT_FIT, 1, MFTL_NULL);
    double *rc = curvature(obj, MFTL_CURVATURE_COTAN, MFTL_NORMAL_WEIGHT_ANGLE, 1, MFTL_NULL);
    e_fit[l] = sphere_kappa_error(obj, rf, 1, 1.0);
    e_cot[l] = sphere_kappa_error(obj, rc, 1, 1.0);
    printf("\n    level %d: kappa max rel. error FIT %.2e COTAN %.2e; normal chord FIT %.2e ANGLE %.2e", l, e_fit[l], e_cot[l], e_nfit[l], e_nangle[l]);
    free(rf);
    free(rc);
    fixture_drop(obj);
  }
  for (int l = 4; l <= 5; ++l)
  {
    const double r = log(h[l - 1] / h[l]);
    TEST_ASSERT(log(e_fit[l - 1] / e_fit[l]) / r > 1.7);  // second order
    TEST_ASSERT(log(e_nfit[l - 1] / e_nfit[l]) / r > 1.7);
    TEST_ASSERT(log(e_nangle[l - 1] / e_nangle[l]) / r < 1.3);  // first order: the reason for the fit
  }
  TEST_ASSERT(e_fit[5] < 2e-3);
  TEST_ASSERT(e_cot[5] > 10.0 * e_fit[5]);  // the documented limit of the cotangent formula
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Shapes with two different curvatures
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_an_ellipsoid_has_its_principal_curvatures_and_directions(void)
{
  const double ax[3] = {1.0, 0.7, 0.5};
  double prev = 1.0;
  for (int l = 3; l <= 5; ++l)
  {
    MFTL_Object *obj = make_sphere(l, 1.0, 0.0);
    for (int v = 0; v < obj->verts.n; ++v)
      for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] *= ax[k];
    settle(obj);
    double *rf = curvature(obj, MFTL_CURVATURE_FIT, MFTL_NORMAL_WEIGHT_FIT, MFTL_CURVATURE_COMPONENTS, MFTL_NULL);
    double *rc = curvature(obj, MFTL_CURVATURE_COTAN, MFTL_NORMAL_WEIGHT_ANGLE, MFTL_CURVATURE_COMPONENTS, MFTL_NULL);
    double ek = 0.0, eg = 0.0, ep = 0.0, dir_fit = 0.0, dir_cot = 0.0;
    int nd = 0;
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double kappa, gauss, k1, k2, e1[3];
      ellipsoid_reference(&obj->verts.pos[3 * v], ax, &kappa, &gauss, &k1, &k2, e1);
      const double *q = &rf[MFTL_CURVATURE_COMPONENTS * v];
      ek = fmax(ek, fabs(q[MFTL_CURVATURE_KAPPA] - kappa) / 8.0);  // 8 = max kappa on this ellipsoid
      eg = fmax(eg, fabs(q[MFTL_CURVATURE_GAUSS] - gauss) / 16.0);
      ep = fmax(ep, fmax(fabs(q[MFTL_CURVATURE_K1] - k1), fabs(q[MFTL_CURVATURE_K2] - k2)) / 4.0);
      if (k1 - k2 > 0.5)
      {
        /* The angle between lines, by the chord of the better-signed pair. */
        const double *df = &q[MFTL_CURVATURE_DIR1], *dc = &rc[MFTL_CURVATURE_COMPONENTS * v + MFTL_CURVATURE_DIR1];
        dir_fit += 1.0 - fabs(dot3(df, e1));
        dir_cot += 1.0 - fabs(dot3(dc, e1));
        nd++;
      }
    }
    dir_fit = sqrt(2.0 * dir_fit / nd);  // RMS chord, ~ the angle in radians
    dir_cot = sqrt(2.0 * dir_cot / nd);
    printf("\n    level %d: FIT max error kappa %.2e K %.2e k1,k2 %.2e; direction RMS FIT %.2e COTAN %.2e rad", l, ek, eg, ep, dir_fit, dir_cot);
    TEST_ASSERT(ek < prev);
    prev = ek;
    if (l == 5) TEST_ASSERT(ek < 2e-3 && eg < 2e-3 && ep < 3e-3 && dir_fit < 2e-3 && dir_cot < 2e-2);
    free(rf);
    free(rc);
    fixture_drop(obj);
  }
  return 0;
}

static int test_a_torus_has_both_signs_and_its_meridian_direction(void)
{
  const double R0 = 1.0, r0 = 0.6;  // r0 > R0 / 2: kappa < 0 on the inner equator
  MFTL_Object *obj = make_torus(96, R0, r0, 0.2);
  int negative_kappa = 0, negative_gauss = 0;
  for (int m = 0; m < 2; ++m)
  {
    double *rec = curvature(obj, (MFTL_CurvatureMethod) m, MFTL_NORMAL_WEIGHT_FIT, MFTL_CURVATURE_COMPONENTS, MFTL_NULL);
    double ek = 0.0, ed = 0.0;
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double n[3], kappa, gauss, k1, e1[3];
      torus_reference(&obj->verts.pos[3 * v], R0, r0, n, &kappa, &gauss, &k1, e1);
      const double *q = &rec[MFTL_CURVATURE_COMPONENTS * v];
      ek = fmax(ek, fabs(q[MFTL_CURVATURE_KAPPA] - kappa));
      ed = fmax(ed, 1.0 - fabs(dot3(&q[MFTL_CURVATURE_DIR1], e1)));
      if (m == MFTL_CURVATURE_FIT && fabs(kappa) > 0.3) TEST_ASSERT((q[MFTL_CURVATURE_KAPPA] > 0.0) == (kappa > 0.0));
      if (m == MFTL_CURVATURE_FIT && fabs(gauss) > 0.3) TEST_ASSERT((q[MFTL_CURVATURE_GAUSS] > 0.0) == (gauss > 0.0));
      if (m == MFTL_CURVATURE_FIT) negative_kappa += kappa < -0.3, negative_gauss += gauss < -0.3;
    }
    printf("\n    %s: max |kappa error| %.2e (max kappa %.2f), worst direction 1 - |cos| %.2e", m == 0 ? "COTAN" : "FIT", ek, 1.0 / r0 + 1.0 / (R0 + r0), ed);
    if (m == MFTL_CURVATURE_FIT) TEST_ASSERT(ek < 0.1 && ed < 1e-3);  // 3.5 % of the largest kappa, measured
    free(rec);
  }
  TEST_ASSERT(negative_kappa > 0 && negative_gauss > 0);  // non-vacuity: both signs were tested
  fixture_drop(obj);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Integrals, planes
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_gauss_bonnet_holds_to_round_off(void)
{
  for (int s = 0; s < 2; ++s)
  {
    MFTL_Object *obj = (s == 0) ? make_sphere(4, 0.37, 0.3) : make_torus(48, 1.0, 0.4, 0.2);
    double *rec = curvature(obj, MFTL_CURVATURE_COTAN, MFTL_NORMAL_WEIGHT_ANGLE, 2, MFTL_NULL);
    double *A = mixed_areas(obj);
    double total = 0.0;
    for (int v = 0; v < obj->verts.n; ++v) total += rec[2 * v + MFTL_CURVATURE_GAUSS] * A[v];
    const double chi = (s == 0) ? 2.0 : 0.0;
    printf("\n    %s: sum K A = %.17g, 2 pi chi = %.17g", s == 0 ? "sphere" : "torus", total, 2.0 * MFTL_PI * chi);
    TEST_ASSERT(fabs(total - 2.0 * MFTL_PI * chi) <= 1e-11);
    free(rec);
    free(A);
    fixture_drop(obj);
  }
  return 0;
}

static int test_a_plane_has_no_curvature(void)
{
  enum
  {
    nx = 12
  };
  double pos[3 * (nx + 1) * (nx + 1)];
  int tri[6 * nx * nx], t = 0;
  for (int j = 0; j <= nx; ++j)
    for (int i = 0; i <= nx; ++i)
    {
      const int v = j * (nx + 1) + i;
      /* An irregular planar grid in a tilted plane. */
      const double x = (i + 0.3 * noise((uint64_t) v, 7) * (i > 0 && i < nx)) / nx, y = (j + 0.3 * noise((uint64_t) v, 8) * (j > 0 && j < nx)) / nx;
      pos[3 * v + 0] = x;
      pos[3 * v + 1] = y;
      pos[3 * v + 2] = 0.25 * x - 0.5 * y;
    }
  for (int j = 0; j < nx; ++j)
    for (int i = 0; i < nx; ++i)
    {
      const int a = j * (nx + 1) + i, b = a + 1, c = a + nx + 2, d = a + nx + 1;
      tri[t++] = a, tri[t++] = b, tri[t++] = c;
      tri[t++] = a, tri[t++] = c, tri[t++] = d;
    }
  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, (nx + 1) * (nx + 1), tri, 2 * nx * nx);
  settle(obj);
  const double h = 1.0 / nx;
  for (int m = 0; m < 2; ++m)
  {
    double *rec = curvature(obj, (MFTL_CurvatureMethod) m, MFTL_NORMAL_WEIGHT_FIT, 4, MFTL_NULL);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      if (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY) continue;  // one-sided there (COTAN)
      /* kappa and K to round-off. k1,2 = kappa/2 +- sqrt(kappa^2/4 - K) turn a
       * rounding-sized K into its square root: that much, and no more. */
      TEST_ASSERT(fabs(rec[4 * v + MFTL_CURVATURE_KAPPA]) * h <= 1e-12);
      TEST_ASSERT(fabs(rec[4 * v + MFTL_CURVATURE_GAUSS]) * h * h <= 1e-12);
      TEST_ASSERT(fabs(rec[4 * v + MFTL_CURVATURE_K1]) * h <= 1e-6 && fabs(rec[4 * v + MFTL_CURVATURE_K2]) * h <= 1e-6);
    }
    free(rec);
  }
  fixture_drop(obj);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Tryggvason's vertex force is Meyer's operator
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_the_frenet_vertex_force_is_the_cotangent_operator(void)
{
  MFTL_Object *obj = make_sphere(3, 1.3, 0.3);
  double *rec = curvature(obj, MFTL_CURVATURE_COTAN, MFTL_NORMAL_WEIGHT_ANGLE, 1, MFTL_NULL);
  double *A = mixed_areas(obj);
  MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
  p.contour_normal = MFTL_CONTOUR_NORMAL_TRIANGLES;
  p.sigma = 0.07;
  double *F = (double *) malloc((size_t) 3 * obj->verts.n * sizeof(double));
  TEST_ASSERT(MFTLComputeSurfaceTension(obj, &p, F) == MFTL_SUCCESS);
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);  // COTAN projects on the area-weighted normal (its own sum)
  double worst = 0.0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double fn = dot3(&F[3 * v], &obj->verts.normal[3 * v]);
    const double expect = -p.sigma * rec[v] * A[v];
    worst = fmax(worst, fabs(fn - expect) / fabs(expect));
  }
  printf("\n    worst relative difference %.2e", worst);
  TEST_ASSERT(worst <= 1e-12);
  free(F);
  free(rec);
  free(A);
  fixture_drop(obj);
  return 0;
}

/**
 * @brief The Laplace pressure -sum F.n / sum A against 2 sigma / R on a
 *        jittered sphere: EXACT for the area gradient at every resolution
 *        (A is homogeneous of degree 2 in the positions, so sum x.grad A = 2A),
 *        converging for the forms with fitted vertex normals.
 */
static int test_the_laplace_pressure_of_a_sphere(void)
{
  const double R = 0.8, sigma = 0.3;
  double err[6][3], h[6];
  for (int l = 2; l <= 4; ++l)
  {
    MFTL_Object *obj = make_sphere(l, R, 0.3);
    h[l] = MFTLMeanEdgeLength(obj);
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);
    double *F = (double *) malloc((size_t) 3 * (obj->tris.n > obj->verts.n ? obj->tris.n : obj->verts.n) * sizeof(double));
    for (int k = 0; k < 3; ++k)
    {
      MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
      p.sigma = sigma;
      p.form = (k == 2) ? MFTL_TENSION_FRENET_ELEMENT : MFTL_TENSION_FRENET_VERTEX;
      p.contour_normal = (k == 0) ? MFTL_CONTOUR_NORMAL_TRIANGLES : MFTL_CONTOUR_NORMAL_VERTICES;
      TEST_ASSERT(MFTLComputeSurfaceTension(obj, &p, F) == MFTL_SUCCESS);
      double fn = 0.0, area = 0.0;
      const int rows = (k == 2) ? obj->tris.n : obj->verts.n;
      for (int i = 0; i < rows; ++i)
      {
        double x[3];
        if (k == 2)
          MFTLGetTriangleCenter(obj, i, x);
        else
          memcpy(x, &obj->verts.pos[3 * i], sizeof(x));
        unit3(x);
        fn += dot3(&F[3 * i], x);
      }
      for (int t = 0; t < obj->tris.n; ++t) area += MFTLGetTriangleArea(obj, t);
      err[l][k] = (-fn / area) / (2.0 * sigma / R) - 1.0;
    }
    printf("\n    level %d: Laplace error area gradient %+.1e, vertex/fitted %+.1e, element/fitted %+.1e", l, err[l][0], err[l][1], err[l][2]);
    free(F);
    fixture_drop(obj);
  }
  for (int l = 2; l <= 4; ++l) TEST_ASSERT(fabs(err[l][0]) <= 1e-13);
  for (int k = 1; k < 3; ++k) TEST_ASSERT(log(fabs(err[3][k] / err[4][k])) / log(h[3] / h[4]) > 1.5);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Length unit
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Everything this module computes, on one ensemble: the fitted normals,
 *        both curvature records and the four force variants.
 */
static void compute_all(MFTL_Object *obj, double *nrm, double *cot, double *fit, double *forces[4])
{
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);
  memcpy(nrm, obj->verts.normal, (size_t) 3 * obj->verts.n * sizeof(double));
  MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_COTAN, MFTL_CURVATURE_COMPONENTS, cot, MFTL_NULL);
  MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_FIT, MFTL_CURVATURE_COMPONENTS, fit, MFTL_NULL);
  for (int k = 0; k < 4; ++k)
  {
    MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
    p.form = (k < 2) ? MFTL_TENSION_FRENET_VERTEX : MFTL_TENSION_FRENET_ELEMENT;
    p.contour_normal = (k % 2) ? MFTL_CONTOUR_NORMAL_VERTICES : MFTL_CONTOUR_NORMAL_TRIANGLES;
    MFTLComputeSurfaceTension(obj, &p, forces[k]);
  }
}

static int test_curvature_and_force_are_independent_of_the_length_unit(void)
{
  const double s = ldexp(1.0, -20);
  double *nrm[2], *cot[2], *fit[2], *forces[2][4];
  int nv = 0, nt = 0;
  for (int pass = 0; pass < 2; ++pass)
  {
    /* A periodic box with the body across its seam: the minimum image is exercised. */
    const double f = pass ? s : 1.0;
    const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {4.0 * f, 4.0 * f, 4.0 * f};
    const bool per[3] = {true, true, true};
    const int cuts[3] = {1, 1, 1};
    MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, per, 2);
    MFTL_Object *obj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(ens, obj);
    const double c[3] = {0.0, 2.0 * f, 3.9 * f};
    MFTLGenerateSphere(obj, c, 1.0 * f, 3);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double *p = &obj->verts.pos[3 * v], d[3];
      MFTLGetMinimumImageVector(&ens->domain, d, c, p);
      const double r = 1.0 + 0.1 * noise((uint64_t) obj->verts.rec[v].global_id, 9);  // exact in binary? no matter: same factor in both passes
      for (int k = 0; k < 3; ++k) p[k] = c[k] + r * d[k];
      MFTLApplyPeriodicity(&ens->domain, p);
    }
    settle(obj);
    nv = obj->verts.n, nt = obj->tris.n;
    nrm[pass] = malloc((size_t) 3 * nv * sizeof(double));
    cot[pass] = malloc((size_t) MFTL_CURVATURE_COMPONENTS * nv * sizeof(double));
    fit[pass] = malloc((size_t) MFTL_CURVATURE_COMPONENTS * nv * sizeof(double));
    for (int k = 0; k < 4; ++k) forces[pass][k] = malloc((size_t) 3 * (k < 2 ? nv : nt) * sizeof(double));
    compute_all(obj, nrm[pass], cot[pass], fit[pass], forces[pass]);
    MFTLEnsembleDestroy(&ens);
  }

  int same = 1;
  for (int i = 0; i < 3 * nv; ++i) same &= nrm[1][i] == nrm[0][i];
  for (int v = 0; v < nv; ++v)
    for (int c = 0; c < MFTL_CURVATURE_COMPONENTS; ++c)
    {
      const double scale = (c == MFTL_CURVATURE_GAUSS) ? s * s : (c >= MFTL_CURVATURE_DIR1) ? 1.0 : s;
      same &= cot[1][MFTL_CURVATURE_COMPONENTS * v + c] * scale == cot[0][MFTL_CURVATURE_COMPONENTS * v + c];
      same &= fit[1][MFTL_CURVATURE_COMPONENTS * v + c] * scale == fit[0][MFTL_CURVATURE_COMPONENTS * v + c];
    }
  for (int k = 0; k < 4; ++k)
    for (int i = 0; i < 3 * (k < 2 ? nv : nt); ++i) same &= forces[1][k][i] == forces[0][k][i] * s;
  TEST_ASSERT(same);

  for (int pass = 0; pass < 2; ++pass)
  {
    free(nrm[pass]);
    free(cot[pass]);
    free(fit[pass]);
    for (int k = 0; k < 4; ++k) free(forces[pass][k]);
  }
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Fallback and refusals
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_a_vertex_the_fit_cannot_serve_gets_the_cotangent_values(void)
{
  /* Two triangles: every vertex has two or three neighbours, one side only. */
  const double pos[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.1, 0.0, 1.0, 0.2, 1.0, 1.0, 0.0};
  const int tri[6] = {0, 1, 2, 1, 3, 2};
  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, 4, tri, 2);
  settle(obj);
  int fb = 0;
  double *fit = curvature(obj, MFTL_CURVATURE_FIT, MFTL_NORMAL_WEIGHT_ANGLE, 4, &fb);
  double *cot = curvature(obj, MFTL_CURVATURE_COTAN, MFTL_NORMAL_WEIGHT_ANGLE, 4, MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(2, fb);  // the corners 0 and 3 have two neighbours
  TEST_ASSERT(memcmp(&fit[0], &cot[0], 4 * sizeof(double)) == 0);
  TEST_ASSERT(memcmp(&fit[12], &cot[12], 4 * sizeof(double)) == 0);
  free(fit);
  free(cot);

  double out[16];
  TEST_ASSERT(MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_COTAN, 3, out, MFTL_NULL) == MFTL_ERROR);
  TEST_ASSERT(MFTLComputeVertexCurvature(obj, (MFTL_CurvatureMethod) 7, 1, out, MFTL_NULL) == MFTL_ERROR);
  TEST_ASSERT(MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_COTAN, 1, MFTL_NULL, MFTL_NULL) == MFTL_ERROR);
  TEST_ASSERT(MFTLComputeVertexCurvature(MFTL_NULL, MFTL_CURVATURE_COTAN, 1, out, MFTL_NULL) == MFTL_ERROR);
  MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
  p.sigma_field = 0;  // no field is registered
  TEST_ASSERT(MFTLComputeSurfaceTension(obj, &p, out) == MFTL_ERROR);
  p = MFTLSurfaceTensionDefaultParameters();
  p.rim = (MFTL_RimForce) 9;
  TEST_ASSERT(MFTLComputeSurfaceTension(obj, &p, out) == MFTL_ERROR);
  TEST_ASSERT(MFTLComputeSurfaceTension(obj, MFTL_NULL, out) == MFTL_ERROR);
  fixture_drop(obj);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  const int cuts[3] = {1, 1, 1};
  const double lo[3] = {-5.0, -5.0, -5.0}, hi[3] = {5.0, 5.0, 5.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, MFTL_NULL, 4);

  RUN_TEST(test_a_sphere_has_kappa_two_over_r_with_the_sign_of_its_orientation);
  RUN_TEST(test_the_fit_converges_where_the_cotangent_formula_does_not);
  RUN_TEST(test_an_ellipsoid_has_its_principal_curvatures_and_directions);
  RUN_TEST(test_a_torus_has_both_signs_and_its_meridian_direction);
  RUN_TEST(test_gauss_bonnet_holds_to_round_off);
  RUN_TEST(test_a_plane_has_no_curvature);
  RUN_TEST(test_the_frenet_vertex_force_is_the_cotangent_operator);
  RUN_TEST(test_the_laplace_pressure_of_a_sphere);
  RUN_TEST(test_curvature_and_force_are_independent_of_the_length_unit);
  RUN_TEST(test_a_vertex_the_fit_cannot_serve_gets_the_cotangent_values);

  MFTLEnsembleDestroy(&g_ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  TEST_REPORT();
  return g_tests_failed > 0;
}
