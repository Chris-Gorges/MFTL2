/**
 * @file curvature_comparison.c
 * @author Christian Gorges
 * @date 2026-11-15
 *
 * @brief Every curvature estimator and both surface-tension forms of the
 *        library, on shapes whose curvature is known in closed form, at a
 *        sequence of resolutions: the measurements behind
 *        src/geometry/README.md "Signed curvature" and "Surface tension".
 *
 * Shapes (every vertex lies EXACTLY on the analytic surface, so every error is
 * the estimator's):
 *
 *   sphere        icosphere, R = 1 (regular, valence 5 and 6)
 *   sphere-jit    the icosphere with every vertex moved tangentially by up to
 *                 0.3 of an edge and put back on the sphere (irregular)
 *   sphere-rem    an icosphere one level finer, isotropically remeshed to the
 *                 coarser level's edge length, put back on the sphere: what a
 *                 front-tracking run's remesher leaves (irregular, valence 4-8)
 *   sphere-mc     marching cubes of the sphere's distance on a lattice, put
 *                 back on the sphere: what a topology change leaves (slivers)
 *   ellipsoid     the icosphere scaled to semi-axes (1, 0.7, 0.5)
 *   torus         R = 1, r = 0.6 on a structured (u, v) grid: both signs of
 *                 the Gaussian curvature, and kappa < 0 on the inner equator
 *   torus-jit     the torus grid jittered in (u, v) by up to 0.2 of a cell
 *
 * Curvature methods: COTAN (Meyer et al. 2003: cotangent mean-curvature normal,
 * angle deficit, principal curvatures from both, Meyer's tensor fit for the
 * direction) and FIT (one-ring height paraboloid through positions and
 * normals) with area-weighted, angle-weighted and fitted
 * (MFTL_NORMAL_WEIGHT_FIT) normals, and -- as a diagnostic of what the fit
 * could do with perfect data -- EXACT normals. Before them, the accuracy of
 * the vertex normals themselves, Max's (1999) weights included for reference.
 *
 * Forces (sigma = 1): the Frenet vertex form with triangle normals (the
 * book's eq. 7.59, the area gradient, the library default) and with fitted
 * vertex normals along the contour, and the Frenet element form with triangle,
 * area-weighted vertex and fitted vertex normals on its edges. Each force is
 * read back as a curvature, kappa_F = -F . n / A, with A the area the force is
 * the integral over: the triangle's for the element form, the barycentric
 * cell's for the vertex form with vertex normals, and for the area gradient
 * Meyer's MIXED area, because it is the integral over any cell through the
 * edge midpoints and the mixed cell is the one it matches pointwise. Every
 * force read back with the BARYCENTRIC area (what a solver dividing by the
 * coupler's marker weight gets) is reported too. Then the spurious tangential
 * part, the conservation of force and torque, and on the spheres the Laplace
 * pressure -sum F.n / sum A against 2 sigma / R.
 *
 * Errors are relative to the largest |reference| on the shape (pointwise
 * relative errors are meaningless where kappa or K vanish): L2 is the RMS over
 * vertices (or triangles), Linf the maximum. The principal direction is judged
 * where |k1 - k2| exceeds a fifth of the largest |kappa|, as the angle
 * between lines.
 *
 * Usage: curvature_comparison [--levels LO HI] [--shape NAME] [--timing]
 *        (serial; run with one rank)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_remeshing.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * Analytic references
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef enum
{
  SH_SPHERE,
  SH_SPHERE_JIT,
  SH_SPHERE_REM,
  SH_SPHERE_MC,
  SH_ELLIPSOID,
  SH_TORUS,
  SH_TORUS_JIT,
  SH_COUNT
} Shape;

static const char *const g_shape_name[SH_COUNT] = {"sphere", "sphere-jit", "sphere-rem", "sphere-mc", "ellipsoid", "torus", "torus-jit"};

static const double g_ax[3] = {1.0, 0.7, 0.5};  // ellipsoid semi-axes
static const double g_tR = 1.0, g_tr = 0.6;  // torus radii

/** Closed-form geometry at a surface point: outward normal, kappa = k1 + k2, K, k1 >= k2, e1. */
typedef struct
{
  double n[3], kappa, gauss, k1, k2, e1[3];
} Ref;

static void normalise(double v[3])
{
  const double m = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (m > 0.0)
    for (int k = 0; k < 3; ++k) v[k] /= m;
}

static void cross(const double a[3], const double b[3], double c[3])
{
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

static double dot(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/** @brief The closest point on the ellipsoid to @p p (Newton on the Lagrange multiplier). */
static void ellipsoid_project(const double p[3], double q[3])
{
  double t = 0.0;
  for (int it = 0; it < 60; ++it)
  {
    double f = -1.0, df = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double a2 = g_ax[k] * g_ax[k], d = a2 + t, qk = p[k] * a2 / d;
      f += qk * qk / a2;
      df += -2.0 * p[k] * p[k] * a2 / (d * d * d);
    }
    const double dt = f / df;
    t -= dt;
    if (fabs(dt) < 1e-16 * (1.0 + fabs(t))) break;
  }
  for (int k = 0; k < 3; ++k) q[k] = p[k] * g_ax[k] * g_ax[k] / (g_ax[k] * g_ax[k] + t);
}

static void torus_project(const double p[3], double q[3])
{
  const double u = atan2(p[1], p[0]), rho = sqrt(p[0] * p[0] + p[1] * p[1]) - g_tR, v = atan2(p[2], rho);
  q[0] = (g_tR + g_tr * cos(v)) * cos(u);
  q[1] = (g_tR + g_tr * cos(v)) * sin(u);
  q[2] = g_tr * sin(v);
}

/** @brief Moves @p p onto the analytic surface of @p sh (closest point). */
static void project(Shape sh, const double p[3], double q[3])
{
  if (sh == SH_ELLIPSOID)
    ellipsoid_project(p, q);
  else if (sh == SH_TORUS || sh == SH_TORUS_JIT)
    torus_project(p, q);
  else
  {
    q[0] = p[0], q[1] = p[1], q[2] = p[2];
    normalise(q);
  }
}

/** @brief The reference at a point ON the surface. */
static void reference(Shape sh, const double p[3], Ref *r)
{
  if (sh == SH_TORUS || sh == SH_TORUS_JIT)
  {
    const double u = atan2(p[1], p[0]), rho = sqrt(p[0] * p[0] + p[1] * p[1]) - g_tR, v = atan2(p[2], rho);
    const double cu = cos(u), su = sin(u), cv = cos(v), sv = sin(v);
    r->n[0] = cv * cu, r->n[1] = cv * su, r->n[2] = sv;
    const double km = 1.0 / g_tr, kp = cv / (g_tR + g_tr * cv);  // meridian (around the tube), parallel
    r->kappa = km + kp;
    r->gauss = km * kp;
    r->k1 = km;  // always the larger for R > 0
    r->k2 = kp;
    r->e1[0] = -sv * cu, r->e1[1] = -sv * su, r->e1[2] = cv;
    return;
  }
  if (sh == SH_ELLIPSOID)
  {
    /* Shape operator of F = sum x_k^2/a_k^2 - 1: P Hess(F) P / |grad F|, in a tangent frame. */
    double g[3];
    for (int k = 0; k < 3; ++k) g[k] = 2.0 * p[k] / (g_ax[k] * g_ax[k]);
    const double gm = sqrt(dot(g, g));
    double n[3] = {g[0] / gm, g[1] / gm, g[2] / gm};
    memcpy(r->n, n, sizeof(n));
    double t1[3] = {0.0, 0.0, 0.0};
    int a = 0;
    for (int k = 1; k < 3; ++k)
      if (fabs(n[k]) < fabs(n[a])) a = k;
    t1[a] = 1.0;
    for (int k = 0; k < 3; ++k) t1[k] -= n[a] * n[k];
    normalise(t1);
    double t2[3];
    cross(n, t1, t2);
    double B[2][2];
    const double *t[2] = {t1, t2};
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
      {
        double s = 0.0;
        for (int k = 0; k < 3; ++k) s += t[i][k] * (2.0 / (g_ax[k] * g_ax[k])) * t[j][k];
        B[i][j] = s / gm;
      }
    const double tr = B[0][0] + B[1][1], det = B[0][0] * B[1][1] - B[0][1] * B[1][0];
    const double disc = sqrt(fmax(0.25 * tr * tr - det, 0.0));
    r->kappa = tr;
    r->gauss = det;
    r->k1 = 0.5 * tr + disc;
    r->k2 = 0.5 * tr - disc;
    const double phi = 0.5 * atan2(2.0 * B[0][1], B[0][0] - B[1][1]);
    for (int k = 0; k < 3; ++k) r->e1[k] = cos(phi) * t1[k] + sin(phi) * t2[k];
    return;
  }
  r->n[0] = p[0], r->n[1] = p[1], r->n[2] = p[2];
  normalise(r->n);
  r->kappa = 2.0;
  r->gauss = 1.0;
  r->k1 = r->k2 = 1.0;
  r->e1[0] = r->e1[1] = r->e1[2] = 0.0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Meshes
 * ═════════════════════════════════════════════════════════════════════════════ */

static MFTL_Ensemble *g_ens = MFTL_NULL;

static double noise(unsigned long long id, unsigned long long salt)
{
  unsigned long long z = id * 0x9E3779B97F4A7C15ull + salt * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * ((double) (z >> 11) / 9007199254740992.0) - 1.0;
}

static MFTL_Object *new_object(void)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, obj);
  return obj;
}

static void finish(MFTL_Object *obj)
{
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLObjectMarkChanged(obj);
}

static MFTL_Object *make_icosphere(int level)
{
  MFTL_Object *obj = new_object();
  const double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 1.0, level);
  return obj;
}

static MFTL_Object *make_torus(int level, double jitter)
{
  const int nv = 6 << level, nu = (int) lround(nv * g_tR / g_tr);
  const int n = nu * nv;
  double *pos = (double *) malloc((size_t) 3 * n * sizeof(double));
  int *tri = (int *) malloc((size_t) 6 * n * sizeof(int));
  for (int i = 0; i < nu; ++i)
    for (int j = 0; j < nv; ++j)
    {
      const int id = i * nv + j;
      const double u = 2.0 * MFTL_PI * (i + jitter * noise((unsigned long long) id, 1)) / nu;
      const double v = 2.0 * MFTL_PI * (j + jitter * noise((unsigned long long) id, 2)) / nv;
      pos[3 * id + 0] = (g_tR + g_tr * cos(v)) * cos(u);
      pos[3 * id + 1] = (g_tR + g_tr * cos(v)) * sin(u);
      pos[3 * id + 2] = g_tr * sin(v);
    }
  int t = 0;
  for (int i = 0; i < nu; ++i)
    for (int j = 0; j < nv; ++j)
    {
      const int a = i * nv + j, b = ((i + 1) % nu) * nv + j, c = ((i + 1) % nu) * nv + (j + 1) % nv, d = i * nv + (j + 1) % nv;
      tri[3 * t + 0] = a, tri[3 * t + 1] = b, tri[3 * t + 2] = c, ++t;
      tri[3 * t + 0] = a, tri[3 * t + 1] = c, tri[3 * t + 2] = d, ++t;
    }
  MFTL_Object *obj = new_object();
  MFTLBuildIndexedMesh(obj, pos, n, tri, t);
  free(pos);
  free(tri);
  return obj;
}

static MFTL_Object *make_shape(Shape sh, int level)
{
  MFTL_Object *obj = MFTL_NULL;
  switch (sh)
  {
    case SH_SPHERE: obj = make_icosphere(level); break;
    case SH_SPHERE_JIT:
    {
      obj = make_icosphere(level);
      const double h = MFTLMeanEdgeLength(obj);
      for (int v = 0; v < obj->verts.n; ++v)
      {
        double *p = &obj->verts.pos[3 * v], t1[3], t2[3], n[3] = {p[0], p[1], p[2]};
        normalise(n);
        const double ax[3] = {fabs(n[0]) < 0.6 ? 1.0 : 0.0, fabs(n[0]) < 0.6 ? 0.0 : 1.0, 0.0};
        cross(n, ax, t1);
        normalise(t1);
        cross(n, t1, t2);
        const double a = 0.3 * h * noise((unsigned long long) v, 3), b = 0.3 * h * noise((unsigned long long) v, 4);
        for (int k = 0; k < 3; ++k) p[k] += a * t1[k] + b * t2[k];
        normalise(p);
      }
      break;
    }
    case SH_SPHERE_REM:
    {
      MFTL_Object *coarse = make_icosphere(level);
      const double target = MFTLMeanEdgeLength(coarse);
      MFTLEnsembleEraseLocalSlot(g_ens, g_ens->n_local_objects - 1);
      MFTLObjectDestroy(&coarse);
      obj = make_icosphere(level + 1);
      MFTL_RemeshParameters rp;
      memset(&rp, 0, sizeof(rp));
      MFTLInitRemeshParams(&rp, MFTL_REMESH_ISOTROPIC, target, 5, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
      MFTLRemeshObject(obj, &rp, MFTL_NULL);
      for (int v = 0; v < obj->verts.n; ++v) normalise(&obj->verts.pos[3 * v]);
      break;
    }
    case SH_SPHERE_MC:
    {
      const int N = 5 << level;  // lattice cells across [-1.25, 1.25]
      const double dx = 2.5 / N, o[3] = {-1.25, -1.25, -1.25}, sp[3] = {dx, dx, dx};
      const int dims[3] = {N + 1, N + 1, N + 1};
      MFTL_SdfGrid grid;
      MFTLSdfGridInit(&grid, o, sp, dims);
      for (int k = 0; k <= N; ++k)
        for (int j = 0; j <= N; ++j)
          for (int i = 0; i <= N; ++i)
          {
            /* An irrational offset keeps the lattice off the sphere's symmetry. */
            const double x = o[0] + i * dx + 0.0123, y = o[1] + j * dx - 0.0071, z = o[2] + k * dx + 0.0049;
            grid.phi[i + (size_t) dims[0] * (j + (size_t) dims[1] * k)] = sqrt(x * x + y * y + z * z) - 1.0;
          }
      obj = new_object();
      MFTLBuildMeshFromLevelSet(obj, &grid, 1e-2);
      MFTLSdfGridFree(&grid);
      for (int v = 0; v < obj->verts.n; ++v)
      {
        double *p = &obj->verts.pos[3 * v];
        p[0] += 0.0123, p[1] -= 0.0071, p[2] += 0.0049;
        normalise(p);
      }
      break;
    }
    case SH_ELLIPSOID:
      obj = make_icosphere(level);
      for (int v = 0; v < obj->verts.n; ++v)
        for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] *= g_ax[k];
      break;
    case SH_TORUS: obj = make_torus(level, 0.0); break;
    case SH_TORUS_JIT: obj = make_torus(level, 0.2); break;
    default: break;
  }
  finish(obj);
  return obj;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Measurements
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double l2, linf;
  long n;
} Err;

static void err_add(Err *e, double x)
{
  e->l2 += x * x;
  e->linf = fmax(e->linf, fabs(x));
  e->n++;
}

static double err_l2(const Err *e) { return e->n ? sqrt(e->l2 / (double) e->n) : 0.0; }

/** @brief Angle in degrees between the LINES through a and b. */
static double line_angle(const double a[3], const double b[3])
{
  const double c = fabs(dot(a, b)) / sqrt(dot(a, a) * dot(b, b));
  return acos(fmin(c, 1.0)) * 180.0 / MFTL_PI;
}

typedef struct
{
  double kappa_scale, gauss_scale;
} Scales;

static Scales shape_scales(Shape sh, const MFTL_Object *obj)
{
  Scales s = {0.0, 0.0};
  for (int v = 0; v < obj->verts.n; ++v)
  {
    Ref r;
    reference(sh, &obj->verts.pos[3 * v], &r);
    s.kappa_scale = fmax(s.kappa_scale, fabs(r.kappa));
    s.gauss_scale = fmax(s.gauss_scale, fabs(r.gauss));
  }
  return s;
}

typedef enum
{
  M_COTAN,
  M_FIT_AREA,
  M_FIT_ANGLE,
  M_FIT_EXACT,
  M_FIT_FITN,
  M_COUNT
} Method;

static const char *const g_method_name[M_COUNT] = {"COTAN", "FIT (area normals)", "FIT (angle normals)", "FIT (exact normals)", "FIT (fit normals)"};

/** For the normals table only: Max (1999) weights, (e_a x e_b) / (|e_a|^2 |e_b|^2) per corner -- exact when the one-ring lies on a sphere. */
static void max_normals(MFTL_Object *obj)
{
  memset(obj->verts.normal, 0, 3 * (size_t) obj->verts.n * sizeof(double));
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    const double *p0 = &obj->verts.pos[3 * v[0]], *p1 = &obj->verts.pos[3 * v[1]], *p2 = &obj->verts.pos[3 * v[2]];
    double e1[3], e2[3], c[3], n[3];
    for (int k = 0; k < 3; ++k) e1[k] = p1[k] - p0[k], e2[k] = p2[k] - p0[k], c[k] = p2[k] - p1[k];
    cross(e1, e2, n);
    const double l1 = dot(e1, e1), l2 = dot(e2, e2), lc = dot(c, c);
    const double w[3] = {1.0 / (l1 * l2), 1.0 / (l1 * lc), 1.0 / (l2 * lc)};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) obj->verts.normal[3 * v[i] + k] += w[i] * n[k];
  }
  for (int v = 0; v < obj->verts.n; ++v) normalise(&obj->verts.normal[3 * v]);
}

/** @brief Area weights, the library's #MFTL_NORMAL_WEIGHT_AREA. */
static void area_normals(MFTL_Object *obj) { MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA); }

typedef struct
{
  Err kappa, gauss, k1, k2, dir;
  int fallback;
} CurvatureResult;

static CurvatureResult measure_curvature(Shape sh, MFTL_Object *obj, Method m, const Scales *sc)
{
  CurvatureResult res;
  memset(&res, 0, sizeof(res));
  const int n = obj->verts.n;
  double *rec = (double *) malloc((size_t) MFTL_CURVATURE_COMPONENTS * n * sizeof(double));

  if (m == M_FIT_AREA) area_normals(obj);
  if (m == M_FIT_ANGLE) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE);
  if (m == M_FIT_FITN) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);
  if (m == M_FIT_EXACT)
    for (int v = 0; v < n; ++v)
    {
      Ref r;
      reference(sh, &obj->verts.pos[3 * v], &r);
      memcpy(&obj->verts.normal[3 * v], r.n, sizeof(r.n));
    }
  MFTLComputeVertexCurvature(obj, m == M_COTAN ? MFTL_CURVATURE_COTAN : MFTL_CURVATURE_FIT, MFTL_CURVATURE_COMPONENTS, rec, &res.fallback);

  for (int v = 0; v < n; ++v)
  {
    Ref r;
    reference(sh, &obj->verts.pos[3 * v], &r);
    const double *q = &rec[MFTL_CURVATURE_COMPONENTS * v];
    err_add(&res.kappa, (q[MFTL_CURVATURE_KAPPA] - r.kappa) / sc->kappa_scale);
    err_add(&res.gauss, (q[MFTL_CURVATURE_GAUSS] - r.gauss) / sc->gauss_scale);
    err_add(&res.k1, (q[MFTL_CURVATURE_K1] - r.k1) / sc->kappa_scale);
    err_add(&res.k2, (q[MFTL_CURVATURE_K2] - r.k2) / sc->kappa_scale);
    if (r.k1 - r.k2 > 0.2 * sc->kappa_scale) err_add(&res.dir, line_angle(&q[MFTL_CURVATURE_DIR1], r.e1));
  }
  free(rec);
  return res;
}

typedef enum
{
  F_VERTEX_TRI,
  F_VERTEX_FIT,
  F_ELEMENT_TRI,
  F_ELEMENT_AREA,
  F_ELEMENT_FIT,
  F_COUNT
} Form;

static const char *const g_form_name[F_COUNT] = {"Frenet vertex, triangle normals (the book's eq. 7.59: the area gradient; the library default)",
                                                 "Frenet vertex, fitted vertex normals", "Frenet element, triangle normals",
                                                 "Frenet element, area-weighted vertex normals", "Frenet element, fitted vertex normals"};

static bool form_is_vertex(Form f) { return f == F_VERTEX_TRI || f == F_VERTEX_FIT; }

typedef struct
{
  Err kappa;  // curvature read back from the force
  Err kappa_bary;  // vertex form only: read back with the barycentric area
  Err tangential;  // |F_t| / (sigma kappa_scale A)
  double net_force, net_torque;  // |sum F| / sum |F|, |sum x x F| / (sum |x||F|)
  double laplace;  // sphere: relative error of -sum F.n / sum A against 2 sigma / R
} ForceResult;

static ForceResult measure_force(Shape sh, MFTL_Object *obj, Form f, const Scales *sc)
{
  ForceResult res;
  memset(&res, 0, sizeof(res));
  MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
  p.form = form_is_vertex(f) ? MFTL_TENSION_FRENET_VERTEX : MFTL_TENSION_FRENET_ELEMENT;
  p.contour_normal = (f == F_VERTEX_TRI || f == F_ELEMENT_TRI) ? MFTL_CONTOUR_NORMAL_TRIANGLES : MFTL_CONTOUR_NORMAL_VERTICES;
  if (f == F_ELEMENT_AREA) area_normals(obj);
  if (f == F_ELEMENT_FIT || f == F_VERTEX_FIT) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);

  const int rows = form_is_vertex(f) ? obj->verts.n : obj->tris.n;
  double *F = (double *) malloc((size_t) 3 * rows * sizeof(double));
  double *area = (double *) calloc((size_t) rows, sizeof(double));
  double *bary = (double *) calloc((size_t) rows, sizeof(double));
  MFTLComputeSurfaceTension(obj, &p, F);

  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double a = MFTLGetTriangleArea(obj, t);
    if (form_is_vertex(f))
    {
      /* Meyer et al. Fig. 4, written out here independently of the library. */
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      const double *x[3] = {&obj->verts.pos[3 * v[0]], &obj->verts.pos[3 * v[1]], &obj->verts.pos[3 * v[2]]};
      double d[3], l2[3];  // corner dot products, squared length of the edge opposite each corner
      for (int k = 0; k < 3; ++k)
      {
        const double *p0 = x[k], *p1 = x[(k + 1) % 3], *p2 = x[(k + 2) % 3];
        const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
        const double o[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
        d[k] = dot(u, w);
        l2[k] = dot(o, o);
      }
      for (int k = 0; k < 3; ++k)
      {
        bary[v[k]] += a / 3.0;
        if (d[0] >= 0.0 && d[1] >= 0.0 && d[2] >= 0.0)
        {
          /* Voronoi share: (|e_kj|^2 cot(at third) + |e_kl|^2 cot(at second)) / 8, cot = d / (2A). */
          const int j = (k + 1) % 3, m = (k + 2) % 3;
          area[v[k]] += (l2[j] * d[j] + l2[m] * d[m]) / (16.0 * a);
        }
        else
          area[v[k]] += (d[k] < 0.0) ? 0.5 * a : 0.25 * a;
      }
    }
    else
      area[t] = bary[t] = a;
  }

  double sf[3] = {0, 0, 0}, st[3] = {0, 0, 0}, sabs = 0.0, sabs_t = 0.0, fn = 0.0, atot = 0.0;
  for (int i = 0; i < rows; ++i)
  {
    double x[3];
    if (form_is_vertex(f))
      memcpy(x, &obj->verts.pos[3 * i], sizeof(x));
    else
      MFTLGetTriangleCenter(obj, i, x);
    double q[3];
    project(sh, x, q);
    Ref r;
    reference(sh, q, &r);
    const double *Fi = &F[3 * i];
    const double Fn = dot(Fi, r.n);
    const double cell = (f == F_VERTEX_TRI) ? area[i] : bary[i];  // the cell the force is the integral over
    err_add(&res.kappa, (-Fn / cell - r.kappa) / sc->kappa_scale);
    err_add(&res.kappa_bary, (-Fn / bary[i] - r.kappa) / sc->kappa_scale);
    const double ft[3] = {Fi[0] - Fn * r.n[0], Fi[1] - Fn * r.n[1], Fi[2] - Fn * r.n[2]};
    err_add(&res.tangential, sqrt(dot(ft, ft)) / (sc->kappa_scale * bary[i]));
    double tq[3];
    cross(x, Fi, tq);
    const double fa = sqrt(dot(Fi, Fi));
    for (int k = 0; k < 3; ++k) sf[k] += Fi[k], st[k] += tq[k];
    sabs += fa;
    sabs_t += fa * sqrt(dot(x, x));
    fn += Fn;
    atot += bary[i];
  }
  res.net_force = sqrt(dot(sf, sf)) / sabs;
  res.net_torque = sqrt(dot(st, st)) / sabs_t;
  res.laplace = (-fn / atot - 2.0) / 2.0;
  free(F);
  free(area);
  free(bary);
  return res;
}

/** Observed order between two levels: log(e_coarse / e_fine) / log(h_coarse / h_fine). */
static double order(double e0, double e1, double h0, double h1) { return (e0 > 0.0 && e1 > 0.0) ? log(e0 / e1) / log(h0 / h1) : 0.0; }

/* ═════════════════════════════════════════════════════════════════════════════
 * Driver
 * ═════════════════════════════════════════════════════════════════════════════ */

static void drop_object(MFTL_Object *obj)
{
  const int slot = MFTLEnsembleFindLocalSlot(g_ens, obj->global_id);
  MFTLEnsembleEraseLocalSlot(g_ens, slot);
  MFTLObjectDestroy(&obj);
}

static void run_shape(Shape sh, int lo, int hi)
{
  enum
  {
    MAXL = 10
  };
  double h[MAXL];
  int nv[MAXL];
  CurvatureResult cr[MAXL][M_COUNT];
  ForceResult fr[MAXL][F_COUNT];

  for (int l = lo; l <= hi; ++l)
  {
    MFTL_Object *obj = make_shape(sh, l);
    h[l] = MFTLMeanEdgeLength(obj);
    nv[l] = obj->verts.n;
    const Scales sc = shape_scales(sh, obj);
    for (int m = 0; m < M_COUNT; ++m) cr[l][m] = measure_curvature(sh, obj, (Method) m, &sc);
    for (int f = 0; f < F_COUNT; ++f) fr[l][f] = measure_force(sh, obj, (Form) f, &sc);
    drop_object(obj);
  }

  printf("\n## %s\n", g_shape_name[sh]);
  printf("\nvertex normal error, RMS / max degrees (area | angle | Max | fit):\n");
  for (int l = lo; l <= hi; ++l)
  {
    MFTL_Object *obj = make_shape(sh, l);
    printf("  level %d:", l);
    for (int k = 0; k < 4; ++k)
    {
      if (k == 0) area_normals(obj);
      if (k == 1) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE);
      if (k == 2) max_normals(obj);
      if (k == 3) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);
      Err e;
      memset(&e, 0, sizeof(e));
      for (int v = 0; v < obj->verts.n; ++v)
      {
        Ref r;
        reference(sh, &obj->verts.pos[3 * v], &r);
        const double *q = &obj->verts.normal[3 * v];
        const double d[3] = {q[0] - r.n[0], q[1] - r.n[1], q[2] - r.n[2]};
        err_add(&e, 2.0 * asin(fmin(0.5 * sqrt(dot(d, d)), 1.0)) * 180.0 / MFTL_PI);
      }
      printf("  %.2e / %.2e", err_l2(&e), e.linf);
    }
    int inverted = 0;
    for (int t = 0; t < obj->tris.n; ++t)
    {
      double c[3], q[3], nt[3];
      MFTLGetTriangleCenter(obj, t, c);
      MFTLGetTriangleNormal(obj, t, nt);
      project(sh, c, q);
      Ref r;
      reference(sh, q, &r);
      if (dot(nt, r.n) <= 0.0) inverted++;
    }
    printf("%s\n", inverted ? "  (INVERTED TRIANGLES)" : "");
    drop_object(obj);
  }
  for (int m = 0; m < M_COUNT; ++m)
  {
    printf("\n### curvature: %s\n\n", g_method_name[m]);
    printf("| level | vertices | h | kappa L2 | kappa Linf | order Linf | K L2 | K Linf | k1 Linf | k2 Linf | dir RMS deg | dir max deg | fallbacks |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (int l = lo; l <= hi; ++l)
    {
      const CurvatureResult *c = &cr[l][m];
      char ord[16] = "--";
      if (l > lo) snprintf(ord, sizeof(ord), "%.2f", order(cr[l - 1][m].kappa.linf, c->kappa.linf, h[l - 1], h[l]));
      printf("| %d | %d | %.4f | %.2e | %.2e | %s | %.2e | %.2e | %.2e | %.2e | %.2f | %.2f | %d |\n", l, nv[l], h[l], err_l2(&c->kappa), c->kappa.linf, ord,
             err_l2(&c->gauss), c->gauss.linf, c->k1.linf, c->k2.linf, err_l2(&c->dir), c->dir.linf, c->fallback);
    }
    printf("\norders (kappa L2 / Linf, K L2):");
    for (int l = lo + 1; l <= hi; ++l)
      printf("  %.2f / %.2f, %.2f", order(err_l2(&cr[l - 1][m].kappa), err_l2(&cr[l][m].kappa), h[l - 1], h[l]),
             order(cr[l - 1][m].kappa.linf, cr[l][m].kappa.linf, h[l - 1], h[l]), order(err_l2(&cr[l - 1][m].gauss), err_l2(&cr[l][m].gauss), h[l - 1], h[l]));
    printf("\n");
  }
  for (int f = 0; f < F_COUNT; ++f)
  {
    printf("\n### force: %s\n\n", g_form_name[f]);
    printf(
        "| level | h | kappa_F L2 | kappa_F Linf | kappa_F Linf, barycentric area | tangential L2 | tangential Linf | net force | net torque | Laplace error |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|\n");
    for (int l = lo; l <= hi; ++l)
    {
      const ForceResult *r = &fr[l][f];
      char lap[24] = "--";
      if (sh <= SH_SPHERE_MC) snprintf(lap, sizeof(lap), "%+.2e", r->laplace);
      printf("| %d | %.4f | %.2e | %.2e | %.2e | %.2e | %.2e | %.1e | %.1e | %s |\n", l, h[l], err_l2(&r->kappa), r->kappa.linf, r->kappa_bary.linf,
             err_l2(&r->tangential), r->tangential.linf, r->net_force, r->net_torque, lap);
    }
    printf("\norders (kappa_F L2 / Linf):");
    for (int l = lo + 1; l <= hi; ++l)
      printf("  %.2f / %.2f", order(err_l2(&fr[l - 1][f].kappa), err_l2(&fr[l][f].kappa), h[l - 1], h[l]),
             order(fr[l - 1][f].kappa.linf, fr[l][f].kappa.linf, h[l - 1], h[l]));
    printf("\n");
  }
}

/**
 * @brief Nanoseconds per vertex of each kernel at one level, best of seven.
 *
 * The forces are timed WITHOUT the vertex normals they read (computed once
 * before); the normals have rows of their own, so a force with fitted normals
 * costs its row plus the fitted-normal row.
 */
static void run_timing(int level)
{
  MFTL_Object *obj = make_shape(SH_SPHERE_REM, level);
  const int n = obj->verts.n;
  double *rec = (double *) malloc((size_t) MFTL_CURVATURE_COMPONENTS * n * sizeof(double));
  double *F = (double *) malloc((size_t) 3 * (obj->tris.n > n ? obj->tris.n : n) * sizeof(double));

  enum
  {
    K_COUNT = 12
  };
  const char *names[K_COUNT] = {"vertex normals, area (reference)",
                                "vertex normals, angle",
                                "vertex normals, fit",
                                "COTAN, kappa only",
                                "COTAN, kappa and K",
                                "COTAN, full record (with Meyer's direction fit)",
                                "FIT, kappa only",
                                "FIT, full record",
                                "Frenet vertex, triangle normals",
                                "Frenet vertex, vertex normals",
                                "Frenet element, triangle normals",
                                "Frenet element, vertex normals"};
  printf("\n## timing: sphere-rem level %d, %d vertices, %d triangles (ns per vertex)\n\n", level, n, obj->tris.n);
  printf("| kernel | ns/vertex |\n|---|---|\n");
  for (int k = 0; k < K_COUNT; ++k)
  {
    double best = 1e30;
    for (int rep = 0; rep < 7; ++rep)
    {
      if (k >= 6) MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);  // what FIT and the vertex-normal forces read
      MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
      p.form = (k >= 10) ? MFTL_TENSION_FRENET_ELEMENT : MFTL_TENSION_FRENET_VERTEX;
      p.contour_normal = (k == 8 || k == 10) ? MFTL_CONTOUR_NORMAL_TRIANGLES : MFTL_CONTOUR_NORMAL_VERTICES;
      const double t0 = MPI_Wtime();
      if (k <= 2)
        MFTLComputeVertexNormals(obj, (MFTL_NormalWeighting) k);
      else if (k <= 5)
        MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_COTAN, (k == 3) ? 1 : (k == 4) ? 2 : MFTL_CURVATURE_COMPONENTS, rec, MFTL_NULL);
      else if (k <= 7)
        MFTLComputeVertexCurvature(obj, MFTL_CURVATURE_FIT, (k == 6) ? 1 : MFTL_CURVATURE_COMPONENTS, rec, MFTL_NULL);
      else
        MFTLComputeSurfaceTension(obj, &p, F);
      best = fmin(best, MPI_Wtime() - t0);
    }
    printf("| %s | %.1f |\n", names[k], 1e9 * best / n);
  }
  free(rec);
  free(F);
  drop_object(obj);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int lo = 1, hi = 5, timing = 0, only = -1;
  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--levels") && i + 2 < argc)
    {
      lo = atoi(argv[++i]);
      hi = atoi(argv[++i]);
    }
    else if (!strcmp(argv[i], "--timing"))
      timing = 1;
    else if (!strcmp(argv[i], "--shape") && i + 1 < argc)
    {
      ++i;
      for (int s = 0; s < SH_COUNT; ++s)
        if (!strcmp(argv[i], g_shape_name[s])) only = s;
    }
  }
  if (lo < 0) lo = 0;
  if (hi > 8) hi = 8;

  const int cuts[3] = {1, 1, 1};
  const double gmin[3] = {-3.0, -3.0, -3.0}, gmax[3] = {3.0, 3.0, 3.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, gmin, gmax, MFTL_NULL, 4);

  printf("# curvature and surface tension: every method on closed-form shapes\n");
  for (int s = 0; s < SH_COUNT; ++s)
    if (only < 0 || only == s) run_shape((Shape) s, lo, hi);
  if (timing) run_timing(hi);

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return 0;
}
