/*
 * @Author: Christian Gorges
 * @Date: 2026-09-25
 *
 * @brief Unit tests for MFTLRasterizeSignedDistance (src/geometry/signed_distance.c).
 *
 * The magnitude of a signed distance is easy to get right and easy to test. The
 * SIGN is neither, and it is the part the topology change stands on: marching
 * cubes turns a wrong sign at one node into a spurious bubble or a hole. So the
 * sign is checked against a completely independent inside/outside test -- the
 * generalised winding number, a sum of solid angles over every triangle that
 * shares nothing with the closest-point machinery -- on the shapes where
 * pseudo-normals are hardest: sharp non-convex vertices, a neck thinner than
 * the lattice spacing, and a gap between two bodies narrower than one cell.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;  ///< non-periodic [-4, 4]^3
static MFTL_Ensemble *g_per = MFTL_NULL;  ///< periodic in all axes, [0, 4]^3

/* ─────────────────────────────────────────────────────────────────────────────
 * The independent reference: winding number by solid angles
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Signed solid angle of triangle (A,B,C) seen from x (Van Oosterom & Strackee). */
static double solid_angle(const double x[3], const double A[3], const double B[3], const double C[3])
{
  double a[3], b[3], c[3];
  for (int k = 0; k < 3; ++k)
  {
    a[k] = A[k] - x[k];
    b[k] = B[k] - x[k];
    c[k] = C[k] - x[k];
  }
  const double la = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const double lb = sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
  const double lc = sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
  const double det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  const double den = la * lb * lc + (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) * lc + (a[0] * c[0] + a[1] * c[1] + a[2] * c[2]) * lb +
                     (b[0] * c[0] + b[1] * c[1] + b[2] * c[2]) * la;
  return 2.0 * atan2(det, den);
}

/** @brief 1 inside a closed outward-wound mesh, 0 outside. Unwrapped corners. */
static double winding_number(const MFTL_Object *obj, const double x[3])
{
  double w = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double c[3][3];
    MFTLGetTriangleCorners(obj, t, c);
    w += solid_angle(x, c[0], c[1], c[2]);
  }
  return w / (4.0 * MFTL_PI);
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Shapes, built by the (separately tested) marching cubes from analytic fields
 * ─────────────────────────────────────────────────────────────────────────── */

typedef double (*field_fn)(const double p[3], const void *arg);

static MFTL_Object *mesh_from_field(MFTL_Ensemble *ens, const double lo[3], const double hi[3], double dx, field_fn f, const void *arg)
{
  int dims[3];
  const double sp[3] = {dx, dx, dx};
  for (int k = 0; k < 3; ++k) dims[k] = (int) ceil((hi[k] - lo[k]) / dx) + 1;
  MFTL_SdfGrid g;
  MFTLSdfGridInit(&g, lo, sp, dims);
  for (int k = 0; k < dims[2]; ++k)
    for (int j = 0; j < dims[1]; ++j)
      for (int i = 0; i < dims[0]; ++i)
      {
        const double p[3] = {lo[0] + i * dx, lo[1] + j * dx, lo[2] + k * dx};
        g.phi[((size_t) k * dims[1] + j) * dims[0] + i] = f(p, arg);
      }
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  MFTLBuildMeshFromLevelSet(obj, &g, 1e-2);
  MFTLSdfGridFree(&g);
  return obj;
}

static double sphere_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;
  return sqrt((p[0] - s[0]) * (p[0] - s[0]) + (p[1] - s[1]) * (p[1] - s[1]) + (p[2] - s[2]) * (p[2] - s[2])) - s[3];
}

/** @brief Two spheres joined by a thin cylinder: a dumbbell. args: R, neck radius, half-separation. */
static double dumbbell_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;
  const double a[4] = {-s[2], 0, 0, s[0]}, b[4] = {s[2], 0, 0, s[0]};
  const double cyl = fmax(sqrt(p[1] * p[1] + p[2] * p[2]) - s[1], fabs(p[0]) - s[2]);
  return fmin(fmin(sphere_field(p, a), sphere_field(p, b)), cyl);
}

/** @brief A star: a sphere with sharp spikes and dents, strongly non-convex. */
static double star_field(const double p[3], const void *arg)
{
  (void) arg;
  const double r = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  if (r < 1e-12) return -1.0;
  const double u = p[0] / r, v = p[1] / r, w = p[2] / r;
  const double bump = 0.35 * sin(5.0 * u) * cos(4.0 * v) + 0.25 * cos(6.0 * w + 1.0);
  return r - (1.0 + bump);
}

/** @brief Computes angle-weighted vertex normals, the rasteriser's precondition. */
static void prepare(MFTL_Object *obj) { MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE); }

/** @brief A standalone grid over [lo, lo + (n-1) dx]. */
static void blank_grid(MFTL_SdfGrid *g, const double lo[3], double dx, int n)
{
  const double sp[3] = {dx, dx, dx};
  const int dims[3] = {n, n, n};
  MFTLSdfGridInit(g, lo, sp, dims);
}

static void node_coord(const MFTL_SdfGrid *g, size_t idx, double x[3])
{
  const int i = (int) (idx % (size_t) g->dims[0]);
  const int j = (int) ((idx / (size_t) g->dims[0]) % (size_t) g->dims[1]);
  const int k = (int) (idx / ((size_t) g->dims[0] * g->dims[1]));
  x[0] = MFTLSdfGridCoord(g, 0, g->lo[0] + i);
  x[1] = MFTLSdfGridCoord(g, 1, g->lo[1] + j);
  x[2] = MFTLSdfGridCoord(g, 2, g->lo[2] + k);
}

/**
 * @brief Counts nodes whose SDF sign disagrees with the winding number.
 * @param n_checked Out: nodes with a known value, for the non-vacuity check.
 */
static int sign_mismatches(const MFTL_Object *obj, const MFTL_SdfGrid *g, int *n_checked)
{
  int bad = 0;
  *n_checked = 0;
  const size_t n = (size_t) g->dims[0] * g->dims[1] * g->dims[2];

  /* The reference is O(triangles) per node, so check a deterministic sample of
   * about 2500 known nodes rather than all of them. Every node with a CLOSE
   * value -- the ones whose sign came from an edge or a vertex feature, where
   * pseudo-normals matter -- is kept regardless of the sample. */
  size_t known = 0;
  for (size_t i = 0; i < n; ++i)
    if (g->phi[i] != MFTL_SDF_UNKNOWN) known++;
  const size_t stride = (known > 2500) ? known / 2500 : 1;
  const double near = 0.2 * fmin(g->spacing[0], fmin(g->spacing[1], g->spacing[2]));

  size_t seen = 0;
  for (size_t i = 0; i < n; ++i)
  {
    if (g->phi[i] == MFTL_SDF_UNKNOWN) continue;
    if ((seen++ % stride) != 0 && fabs(g->phi[i]) > near) continue;
    double x[3];
    node_coord(g, i, x);
    const double w = winding_number(obj, x);
    /* A node ON the surface has w = 1/2 and either sign is right. */
    if (fabs(w - 0.5) < 0.05) continue;
    (*n_checked)++;
    if ((g->phi[i] < 0.0) != (w > 0.5)) bad++;
  }
  return bad;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. The merge rule
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_merge_is_commutative_associative_and_has_an_identity(void)
{
  const double vals[] = {0.3, -0.3, 0.1, -0.2, 0.0, -0.0, 5.0, MFTL_SDF_UNKNOWN};
  const int n = (int) (sizeof vals / sizeof vals[0]);
  for (int i = 0; i < n; ++i)
  {
    TEST_ASSERT(MFTLSdfMerge(MFTL_SDF_UNKNOWN, vals[i]) == vals[i] || vals[i] == MFTL_SDF_UNKNOWN);
    for (int j = 0; j < n; ++j)
    {
      const double ab = MFTLSdfMerge(vals[i], vals[j]), ba = MFTLSdfMerge(vals[j], vals[i]);
      TEST_ASSERT(ab == ba);
      for (int k = 0; k < n; ++k)
      {
        const double l = MFTLSdfMerge(MFTLSdfMerge(vals[i], vals[j]), vals[k]);
        const double r = MFTLSdfMerge(vals[i], MFTLSdfMerge(vals[j], vals[k]));
        TEST_ASSERT(l == r);
      }
    }
  }
  /* Ties go inside. */
  TEST_ASSERT(MFTLSdfMerge(0.3, -0.3) == -0.3);
  TEST_ASSERT(MFTLSdfMerge(-0.3, 0.3) == -0.3);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. Magnitude and sign on analytic shapes
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_sphere_is_rasterised_to_its_distance_function(void)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, obj);
  double c[3] = {0.1, -0.05, 0.02};
  MFTLGenerateSphere(obj, c, 1.0, 4);
  prepare(obj);

  MFTL_SdfGrid g;
  const double lo[3] = {-1.6, -1.6, -1.6};
  blank_grid(&g, lo, 0.1, 33);
  const double cutoff = 0.35;
  MFTLRasterizeSignedDistance(&g, obj, obj->tris.n, cutoff);

  const double s[4] = {c[0], c[1], c[2], 1.0};
  int known = 0;
  const size_t n = (size_t) 33 * 33 * 33;
  for (size_t i = 0; i < n; ++i)
  {
    double x[3];
    node_coord(&g, i, x);
    const double exact = sphere_field(x, s);
    if (g.phi[i] == MFTL_SDF_UNKNOWN)
    {
      /* Nothing within the cutoff: the node really is farther than that. */
      TEST_ASSERT(fabs(exact) > cutoff - 0.01);
      continue;
    }
    known++;
    TEST_ASSERT(fabs(g.phi[i]) <= cutoff);
    /* The icosphere is within ~5e-3 of the true sphere at this refinement. */
    TEST_ASSERT(fabs(g.phi[i] - exact) < 6e-3);
  }
  TEST_ASSERT(known > 3000);
  MFTLSdfGridFree(&g);
  return 0;
}

static int test_the_sign_agrees_with_the_winding_number_on_hard_shapes(void)
{
  struct
  {
    const char *name;
    field_fn f;
    double arg[4];
    double lo, hi, mesh_dx;
  } shapes[] = {
      {"star", star_field, {0, 0, 0, 0}, -1.8, 1.8, 0.09},
      /* Neck radius 0.07, mesh spacing 0.05, grid spacing 0.06: the neck is
       * barely resolved by the mesh and NOT by the rasterisation lattice. */
      {"dumbbell", dumbbell_field, {0.5, 0.07, 0.55, 0}, -1.2, 1.2, 0.05},
  };

  for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; ++s)
  {
    const double lo[3] = {shapes[s].lo, shapes[s].lo, shapes[s].lo}, hi[3] = {shapes[s].hi, shapes[s].hi, shapes[s].hi};
    MFTL_Object *obj = mesh_from_field(g_ens, lo, hi, shapes[s].mesh_dx, shapes[s].f, shapes[s].arg);
    TEST_ASSERT(inv_mesh_is_closed(obj));
    prepare(obj);

    MFTL_SdfGrid g;
    const double dx = 0.06;
    const int n = (int) ((shapes[s].hi - shapes[s].lo) / dx) + 1;
    /* Offset by a third of a cell so nodes do not line up with the mesh's own
     * lattice -- that would put every node on a vertex or an edge. */
    const double glo[3] = {shapes[s].lo + dx / 3, shapes[s].lo + dx / 7, shapes[s].lo + dx / 5};
    blank_grid(&g, glo, dx, n);
    MFTLRasterizeSignedDistance(&g, obj, obj->tris.n, 2.0 * dx);

    int checked = 0;
    const int bad = sign_mismatches(obj, &g, &checked);
    if (bad) printf("\n  %s: %d of %d signs wrong\n  ", shapes[s].name, bad, checked);
    TEST_ASSERT_EQUAL_INT(0, bad);
    TEST_ASSERT(checked > 2000);
    MFTLSdfGridFree(&g);
  }
  return 0;
}

static int test_a_gap_narrower_than_a_cell_stays_outside(void)
{
  /* Two spheres 0.03 apart, lattice spacing 0.05: the nodes in the gap are
   * nearest to BOTH surfaces at once and must still come out positive. This is
   * the configuration a coalescence event is detected on, and a sign error here
   * would merge the two before anybody asked. */
  MFTL_Object *a = MFTLObjectCreate(0), *b = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, a);
  MFTLEnsembleAddObject(g_ens, b);
  double ca[3] = {-0.515, 0.0, 0.0}, cb[3] = {0.515, 0.0, 0.0};
  MFTLGenerateSphere(a, ca, 0.5, 4);
  MFTLGenerateSphere(b, cb, 0.5, 4);
  prepare(a);
  prepare(b);

  MFTL_SdfGrid g;
  const double lo[3] = {-0.2, -0.2, -0.2};
  blank_grid(&g, lo, 0.025, 17);
  MFTLRasterizeSignedDistance(&g, a, a->tris.n, 0.15);
  MFTLRasterizeSignedDistance(&g, b, b->tris.n, 0.15);

  int in_gap = 0;
  for (size_t i = 0; i < (size_t) 17 * 17 * 17; ++i)
  {
    double x[3];
    node_coord(&g, i, x);
    const double r = sqrt(x[1] * x[1] + x[2] * x[2]);
    if (fabs(x[0]) < 0.012 && r < 0.05)
    {
      in_gap++;
      TEST_ASSERT(g.phi[i] > 0.0);
    }
  }
  TEST_ASSERT(in_gap > 0);
  MFTLSdfGridFree(&g);
  return 0;
}

/**
 * @brief The reason the rasteriser demands ANGLE-weighted vertex normals.
 *
 * Jagged, strongly non-convex star-shaped meshes (an icosphere with every
 * vertex pushed to a random radius in [0.35, 1.65]), and query points whose
 * nearest feature over the WHOLE mesh is a vertex -- the only case where the
 * vertex pseudo-normal decides the sign. Each point is rasterised on its own
 * one-node grid, so what is tested is the rasteriser's own path.
 *
 * Measured while writing this: angle weighting 0 wrong of ~15 000, area
 * weighting 122 wrong. The area run is kept as a CONTROL and must fail: without
 * it this test could pass with the weighting argument silently ignored.
 */
static int test_vertex_signs_on_a_spiky_mesh_need_angle_weighting(void)
{
  unsigned st = 99u;
  int wrong[2] = {0, 0}, tested = 0;

  for (int trial = 0; trial < 30; ++trial)
  {
    MFTL_Object *obj = MFTLObjectCreate(0);
    MFTLEnsembleAddObject(g_ens, obj);
    double c[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(obj, c, 1.0, 1 + trial % 2);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      st = st * 1664525u + 1013904223u;
      const double r = 0.35 + 1.3 * ((double) (st >> 8) / (double) (1u << 24));
      for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] *= r;
    }

    for (int s = 0; s < 1200; ++s)
    {
      double d[3];
      st = st * 1664525u + 1013904223u;
      const int v = (int) ((st >> 8) % (unsigned) obj->verts.n);
      for (int k = 0; k < 3; ++k)
      {
        st = st * 1664525u + 1013904223u;
        d[k] = (double) (st >> 8) / (double) (1u << 24) - 0.5;
      }
      st = st * 1664525u + 1013904223u;
      const double len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      const double scale = 0.02 + 0.3 * ((double) (st >> 8) / (double) (1u << 24));
      double x[3];
      for (int k = 0; k < 3; ++k) x[k] = obj->verts.pos[3 * v + k] + scale * d[k] / len;

      /* Keep only points whose closest feature in the whole mesh is a vertex. */
      double best = MFTL_DBL_MAX;
      MFTL_TriangleFeature bf = MFTL_TRI_FEATURE_FACE;
      for (int t = 0; t < obj->tris.n; ++t)
      {
        double T[3][3];
        MFTLGetTriangleCorners(obj, t, T);
        MFTL_TriangleFeature f;
        const double d2 = MFTLClosestPointOnTriangle(x, T, MFTL_NULL, &f);
        if (d2 < best)
        {
          best = d2;
          bf = f;
        }
      }
      if (bf < MFTL_TRI_FEATURE_VERTEX0 || bf > MFTL_TRI_FEATURE_VERTEX2) continue;

      const bool inside = winding_number(obj, x) > 0.5;
      tested++;
      for (int w = 0; w < 2; ++w)
      {
        MFTLComputeVertexNormals(obj, w == 0 ? MFTL_NORMAL_WEIGHT_ANGLE : MFTL_NORMAL_WEIGHT_AREA);
        MFTL_SdfGrid g;
        const double sp[3] = {1.0, 1.0, 1.0};
        const int dims[3] = {1, 1, 1};
        MFTLSdfGridInit(&g, x, sp, dims);
        MFTLRasterizeSignedDistance(&g, obj, obj->tris.n, 1.0);
        if ((g.phi[0] < 0.0) != inside) wrong[w]++;
        MFTLSdfGridFree(&g);
      }
    }
  }

  printf("\n  vertex-nearest points: %d; wrong signs: angle %d, area %d (control)\n  ", tested, wrong[0], wrong[1]);
  TEST_ASSERT(tested > 1500);
  TEST_ASSERT_EQUAL_INT(0, wrong[0]);
  TEST_ASSERT(wrong[1] > 0);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. Order independence: why a distributed rasterisation equals a serial one
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_rasterising_in_pieces_is_bit_identical_to_all_at_once(void)
{
  const double lo[3] = {-1.8, -1.8, -1.8}, hi[3] = {1.8, 1.8, 1.8};
  MFTL_Object *obj = mesh_from_field(g_ens, lo, hi, 0.09, star_field, MFTL_NULL);
  prepare(obj);

  MFTL_SdfGrid first, again;
  const double glo[3] = {-1.7, -1.7, -1.7};
  blank_grid(&first, glo, 0.07, 49);
  blank_grid(&again, glo, 0.07, 49);
  const size_t n = (size_t) 49 * 49 * 49;

  /* 1. Pieces, in reverse order, overlapping: [0, 2n/3) then [0, n/3) then
   *    everything. Idempotence plus commutativity of the merge. */
  const int nt = obj->tris.n;
  MFTLRasterizeSignedDistance(&first, obj, nt, 0.2);
  MFTLRasterizeSignedDistance(&again, obj, (2 * nt) / 3, 0.2);
  MFTLRasterizeSignedDistance(&again, obj, nt / 3, 0.2);
  MFTLRasterizeSignedDistance(&again, obj, nt, 0.2);
  TEST_ASSERT(memcmp(first.phi, again.phi, n * sizeof(double)) == 0);

  /* Rasterising the same triangles once more changes nothing at all. */
  TEST_ASSERT(MFTLRasterizeSignedDistance(&again, obj, nt, 0.2) == 0);

  /* 2. A PERMUTED pool -- the triangle order a different partition would
   *    produce. The normals are permuted with it rather than recomputed, so the
   *    only thing that changed is the order of contribution. */
  MFTLObjectReorderMorton(obj);
  MFTL_SdfGrid permuted;
  blank_grid(&permuted, glo, 0.07, 49);
  MFTLRasterizeSignedDistance(&permuted, obj, nt, 0.2);
  TEST_ASSERT(memcmp(first.phi, permuted.phi, n * sizeof(double)) == 0);

  MFTLSdfGridFree(&first);
  MFTLSdfGridFree(&again);
  MFTLSdfGridFree(&permuted);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Periodicity and windows
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_body_across_the_seam_is_measured_in_its_own_image(void)
{
  /* A sphere centred ON the corner of a periodic box, rasterised into a window
   * that starts on the far side of the seam. Every node must see the distance to
   * the nearest image, not a domain length. */
  const MFTL_Domain *dom = &g_per->domain;
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_per, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 0.6, 3);
  prepare(obj);

  const int n = 40;
  const double dx = dom->domain_size[0] / n;
  const double origin[3] = {dom->global_min[0], dom->global_min[1], dom->global_min[2]};
  const double sp[3] = {dx, dx, dx};
  const int n_global[3] = {n, n, n};
  const bool per[3] = {true, true, true};
  const int wlo[3] = {-6, 35, -3};  // straddles the seam on every axis
  const int dims[3] = {12, 12, 12};
  MFTL_SdfGrid g;
  MFTLSdfGridInitWindow(&g, origin, sp, n_global, per, wlo, dims);
  MFTLRasterizeSignedDistance(&g, obj, obj->tris.n, 0.3);

  const double s[4] = {0.0, 0.0, 0.0, 0.6};
  int known = 0;
  for (int k = 0; k < 12; ++k)
    for (int j = 0; j < 12; ++j)
      for (int i = 0; i < 12; ++i)
      {
        const double v = g.phi[(k * 12 + j) * 12 + i];
        if (v == MFTL_SDF_UNKNOWN) continue;
        known++;
        /* Node coordinates, then the minimum image of the centre. */
        double x[3] = {MFTLSdfGridCoord(&g, 0, wlo[0] + i), MFTLSdfGridCoord(&g, 1, wlo[1] + j), MFTLSdfGridCoord(&g, 2, wlo[2] + k)};
        double d[3];
        MFTLGetMinimumImageVector(dom, d, s, x);
        const double exact = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - 0.6;
        TEST_ASSERT(fabs(v - exact) < 0.02);
      }
  TEST_ASSERT(known > 200);
  MFTLSdfGridFree(&g);
  return 0;
}

static int test_mesh_to_field_to_mesh_preserves_shape_and_topology(void)
{
  /* The round trip the topology change performs on every body it touches:
   * rasterise, then marching cubes. Closed in, closed out; genus preserved;
   * volume within the lattice's second-order error. A cutoff just above the
   * cell diagonal must be enough -- that is the claim that lets the pipeline
   * skip any sign propagation. */
  const double lo[3] = {-1.8, -1.8, -1.8}, hi[3] = {1.8, 1.8, 1.8};
  MFTL_Object *src = mesh_from_field(g_ens, lo, hi, 0.06, star_field, MFTL_NULL);
  prepare(src);

  const double dx = 0.08;
  MFTL_SdfGrid g;
  const double glo[3] = {-1.9, -1.9, -1.9};
  blank_grid(&g, glo, dx, 49);
  MFTLRasterizeSignedDistance(&g, src, src->tris.n, 1.05 * sqrt(3.0) * dx);

  MFTL_Object *dst = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, dst);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(dst, &g, 1e-2));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(inv_mesh_is_sound(dst, 0));
  TEST_ASSERT(inv_mesh_is_closed(dst));
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(dst, MFTL_NULL));

  double vs = 0.0, vd = 0.0;
  for (int pass = 0; pass < 2; ++pass)
  {
    const MFTL_Object *o = pass ? dst : src;
    double v = 0.0;
    for (int t = 0; t < o->tris.n; ++t)
    {
      double c[3][3];
      MFTLGetTriangleCorners(o, t, c);
      v += (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
            c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])) /
           6.0;
    }
    if (pass)
      vd = v;
    else
      vs = v;
  }
  TEST_ASSERT(fabs(vd - vs) / vs < 0.01);
  return 0;
}

static int test_an_open_patch_and_a_degenerate_face_are_harmless(void)
{
  /* A square patch with a rim, plus one zero-area triangle. Nodes near the rim
   * are signed by the face normal of the rim triangle; nothing is NaN. */
  static const double pos[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 2, 0, 0, 3, 0, 0};
  static const int tri[] = {0, 1, 2, 0, 2, 3, 1, 4, 5};
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(obj, pos, 6, tri, 3));
  prepare(obj);

  MFTL_SdfGrid g;
  const double lo[3] = {-0.5, -0.5, -0.5};
  blank_grid(&g, lo, 0.1, 31);
  MFTLRasterizeSignedDistance(&g, obj, obj->tris.n, 0.3);
  for (size_t i = 0; i < (size_t) 31 * 31 * 31; ++i)
  {
    TEST_ASSERT(!isnan(g.phi[i]));
    if (g.phi[i] == MFTL_SDF_UNKNOWN) continue;
    double x[3];
    node_coord(&g, i, x);
    /* Above the square (normal +z): positive; below: negative. */
    if (x[0] > 0.05 && x[0] < 0.95 && x[1] > 0.05 && x[1] < 0.95 && fabs(x[2]) > 0.05) TEST_ASSERT((g.phi[i] > 0.0) == (x[2] > 0.0));
  }
  MFTLSdfGridFree(&g);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0}, hi[3] = {4.0, 4.0, 4.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 32);
  double plo[3] = {0.0, 0.0, 0.0}, phi[3] = {4.0, 4.0, 4.0};
  bool per[3] = {true, true, true};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 8);

  printf("--- Running MFTL Signed Distance Tests ---\n");

  RUN_TEST(test_merge_is_commutative_associative_and_has_an_identity);
  RUN_TEST(test_a_sphere_is_rasterised_to_its_distance_function);
  RUN_TEST(test_the_sign_agrees_with_the_winding_number_on_hard_shapes);
  RUN_TEST(test_a_gap_narrower_than_a_cell_stays_outside);
  RUN_TEST(test_vertex_signs_on_a_spiky_mesh_need_angle_weighting);
  RUN_TEST(test_rasterising_in_pieces_is_bit_identical_to_all_at_once);
  RUN_TEST(test_a_body_across_the_seam_is_measured_in_its_own_image);
  RUN_TEST(test_mesh_to_field_to_mesh_preserves_shape_and_topology);
  RUN_TEST(test_an_open_patch_and_a_degenerate_face_are_harmless);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
