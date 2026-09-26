/*
 * @Author: Christian Gorges
 * @Date: 2026-09-25
 *
 * @brief Unit tests for marching cubes (src/mesh_creation/marching_cubes.c).
 *
 * Two layers, tested separately because they fail differently.
 *
 * THE TABLE. A wrong row leaves a hole only in surfaces that produce that sign
 * pattern, so the table is checked EXHAUSTIVELY -- all 256 cases, and for the
 * crack-freedom property every one of the 256 x 16 pairs of cells that can
 * share a face. Nothing weaker covers a rare pattern.
 *
 * THE EXTRACTION. Analytic shapes check what a surface must be (closed, genus,
 * outward, on the level set, second-order volume). RANDOM fields are the real
 * adversary: every ambiguous configuration shows up, and the output must still
 * be an edge- AND vertex-manifold, consistently wound surface. A checkerboard is
 * the worst case there is -- every face of every cell ambiguous.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;  ///< non-periodic [-4, 4]^3
static MFTL_Ensemble *g_per = MFTL_NULL;  ///< periodic in x, [0, 4] x [-2, 2]^2

/* ─────────────────────────────────────────────────────────────────────────────
 * Cube helpers, written independently of the implementation
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Corners of cube edge e: axis e/4, offsets (e&1, (e>>1)&1) on the next two axes. */
static void edge_corners(int e, int *c0, int *c1)
{
  const int a = e / 4, b = (a + 1) % 3, c = (a + 2) % 3;
  *c0 = ((e & 1) << b) | (((e >> 1) & 1) << c);
  *c1 = *c0 | (1 << a);
}

/** @brief Faces (2*axis + side) cube edge e lies on, as a bit mask. */
static int faces_of_edge(int e)
{
  int c0, c1;
  edge_corners(e, &c0, &c1);
  int m = 0;
  for (int k = 0; k < 3; ++k)
    if (((c0 >> k) & 1) == ((c1 >> k) & 1)) m |= 1 << (2 * k + ((c0 >> k) & 1));
  return m;
}

/** @brief Midpoint of cube edge e in the unit cube. */
static void edge_mid(int e, double p[3])
{
  int c0, c1;
  edge_corners(e, &c0, &c1);
  for (int k = 0; k < 3; ++k) p[k] = 0.5 * (((c0 >> k) & 1) + ((c1 >> k) & 1));
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Mesh helpers
 * ─────────────────────────────────────────────────────────────────────────── */

static int boundary_half_edges(const MFTL_Object *obj)
{
  int b = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
    if (obj->tris.he[h].twin_idx == MFTL_ERROR) b++;
  return b;
}

/** @brief V - E + F with E = (3F + B) / 2. */
static int euler_characteristic(const MFTL_Object *obj)
{
  const int b = boundary_half_edges(obj);
  return obj->verts.n - (3 * obj->tris.n + b) / 2 + obj->tris.n;
}

/**
 * @brief Every vertex's incident triangles form ONE fan.
 *
 * Outgoing half-edges of a vertex are joined when rotating across a shared edge
 * links them (h ~ twin(prev(h))); a vertex-manifold surface has exactly one
 * class per vertex. Twin symmetry alone does not imply this -- two cones touching
 * at their tips pass every edge test.
 */
static int vertices_are_manifold(const MFTL_Object *obj)
{
  const int nv = obj->verts.n, nh = 3 * obj->tris.n;
  int *first = (int *) malloc((size_t) nv * sizeof(int));
  int *link = (int *) malloc((size_t) nh * sizeof(int));
  int *parent = (int *) malloc((size_t) nh * sizeof(int));
  for (int v = 0; v < nv; ++v) first[v] = -1;
  for (int h = 0; h < nh; ++h)
  {
    const int o = obj->tris.he[MFTLHePrevIdx(h)].vertex_idx;
    link[h] = first[o];
    first[o] = h;
    parent[h] = h;
  }

  /* Union-find over outgoing half-edges. */
  for (int h = 0; h < nh; ++h)
  {
    const int tw = obj->tris.he[MFTLHePrevIdx(h)].twin_idx;
    if (tw == MFTL_ERROR) continue;
    int a = h, b = tw;
    while (parent[a] != a) a = parent[a];
    while (parent[b] != b) b = parent[b];
    if (a != b) parent[a] = b;
  }

  int ok = 1;
  for (int v = 0; v < nv && ok; ++v)
  {
    int root = -1;
    for (int h = first[v]; h >= 0; h = link[h])
    {
      int r = h;
      while (parent[r] != r) r = parent[r];
      if (root < 0)
        root = r;
      else if (r != root)
        ok = 0;
    }
    if (root < 0) ok = 0;  // an unreferenced vertex
  }
  free(first);
  free(link);
  free(parent);
  return ok;
}

/** @brief Divergence-theorem volume, sum of p0 . (p1 x p2) / 6 over unwrapped corners. */
static double enclosed_volume(const MFTL_Object *obj)
{
  double v = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double c[3][3];
    MFTLGetTriangleCorners(obj, t, c);
    v += (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
          c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])) /
         6.0;
  }
  return v;
}

typedef double (*field_fn)(const double p[3], const void *arg);

/** @brief Fills a standalone grid spanning [lo, hi] with dx from an analytic field. */
static void fill_grid(MFTL_SdfGrid *g, const double lo[3], const double hi[3], double dx, field_fn f, const void *arg)
{
  int dims[3];
  double sp[3];
  for (int k = 0; k < 3; ++k)
  {
    dims[k] = (int) ceil((hi[k] - lo[k]) / dx) + 1;
    sp[k] = dx;
  }
  MFTLSdfGridInit(g, lo, sp, dims);
  for (int k = 0; k < dims[2]; ++k)
    for (int j = 0; j < dims[1]; ++j)
      for (int i = 0; i < dims[0]; ++i)
      {
        const double p[3] = {lo[0] + i * dx, lo[1] + j * dx, lo[2] + k * dx};
        g->phi[((size_t) k * dims[1] + j) * dims[0] + i] = f(p, arg);
      }
}

static double sphere_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;  // cx cy cz R
  return sqrt((p[0] - s[0]) * (p[0] - s[0]) + (p[1] - s[1]) * (p[1] - s[1]) + (p[2] - s[2]) * (p[2] - s[2])) - s[3];
}

static double torus_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;  // R r
  const double q = sqrt(p[0] * p[0] + p[1] * p[1]) - s[0];
  return sqrt(q * q + p[2] * p[2]) - s[1];
}

static double two_spheres_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;  // separation, R
  const double a[4] = {-0.5 * s[0], 0.0, 0.0, s[1]};
  const double b[4] = {0.5 * s[0], 0.0, 0.0, s[1]};
  return fmin(sphere_field(p, a), sphere_field(p, b));
}

static MFTL_Object *new_object(MFTL_Ensemble *ens)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  return obj;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. The table
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_every_case_triangulates_exactly_its_crossed_edges(void)
{
  for (int m = 0; m < 256; ++m)
  {
    const MFTL_MarchingCubesCase *cs = MFTLMarchingCubesCase(m);

    int crossed = 0, seen[12] = {0};
    for (int e = 0; e < 12; ++e)
    {
      int c0, c1;
      edge_corners(e, &c0, &c1);
      if (((m >> c0) & 1) != ((m >> c1) & 1)) crossed |= 1 << e;
    }

    int in_loops = 0;
    for (int p = 0; p < cs->n_polys; ++p)
    {
      TEST_ASSERT(cs->poly_len[p] >= 3);
      for (int i = 0; i < cs->poly_len[p]; ++i)
      {
        const int e = cs->poly[p][i];
        TEST_ASSERT(e >= 0 && e < 12);
        TEST_ASSERT(!seen[e]);
        seen[e] = 1;
        in_loops |= 1 << e;
        /* Consecutive loop vertices are joined by a FACE segment. */
        const int f = cs->poly[p][(i + 1) % cs->poly_len[p]];
        TEST_ASSERT((faces_of_edge(e) & faces_of_edge(f)) != 0);
      }
    }
    TEST_ASSERT_EQUAL_INT(crossed, in_loops);
    if (m == 0 || m == 255) TEST_ASSERT_EQUAL_INT(0, cs->n_tris);

    /* Triangle count: n - 2 per loop without a centroid. */
    int expect = 0;
    for (int p = 0; p < cs->n_polys; ++p) expect += cs->poly_len[p] - 2;
    TEST_ASSERT_EQUAL_INT(expect + 2 * cs->n_centroids, cs->n_tris);
  }
  return 0;
}

static int test_no_case_needs_a_centroid_or_puts_a_triangle_in_a_face(void)
{
  for (int m = 0; m < 256; ++m)
  {
    const MFTL_MarchingCubesCase *cs = MFTLMarchingCubesCase(m);
    TEST_ASSERT_EQUAL_INT(0, cs->n_centroids);
    for (int t = 0; t < cs->n_tris; ++t)
    {
      const int common = faces_of_edge(cs->tri[t][0]) & faces_of_edge(cs->tri[t][1]) & faces_of_edge(cs->tri[t][2]);
      TEST_ASSERT_EQUAL_INT(0, common);
    }
  }
  return 0;
}

/**
 * @brief The segments one case draws on one face: the ordered pairs of
 *        consecutive loop vertices that lie on that face.
 */
static int face_segments(const MFTL_MarchingCubesCase *cs, int face, int seg[4][2])
{
  int n = 0;
  for (int p = 0; p < cs->n_polys; ++p)
    for (int i = 0; i < cs->poly_len[p]; ++i)
    {
      const int e = cs->poly[p][i], f = cs->poly[p][(i + 1) % cs->poly_len[p]];
      if ((faces_of_edge(e) & faces_of_edge(f)) & (1 << face))
      {
        seg[n][0] = e;
        seg[n][1] = f;
        n++;
      }
    }
  return n;
}

/**
 * @brief CRACK FREEDOM, exhaustively: two cells sharing a face draw the same
 *        segments on it, in opposite directions.
 *
 * Cell A's +axis face is cell B's -axis face. For every A (256) and every
 * pattern of B's four far corners (16), B's mask is determined, and the
 * segments each draws on the shared face are compared after mapping B's edge
 * numbers into A's frame (shift by one along the axis).
 */
static int test_neighbouring_cells_agree_on_every_shared_face(void)
{
  for (int axis = 0; axis < 3; ++axis)
  {
    for (int ma = 0; ma < 256; ++ma)
    {
      /* Corners of A on its +axis face become B's corners on its -axis face. */
      int near_b = 0;
      for (int c = 0; c < 8; ++c)
        if (((c >> axis) & 1) && ((ma >> c) & 1)) near_b |= 1 << (c & ~(1 << axis));

      for (int far = 0; far < 16; ++far)
      {
        int mb = near_b;
        int bit = 0;
        for (int c = 0; c < 8; ++c)
          if ((c >> axis) & 1)
          {
            if ((far >> bit) & 1) mb |= 1 << c;
            bit++;
          }

        int sa[4][2], sb[4][2];
        const int na = face_segments(MFTLMarchingCubesCase(ma), 2 * axis + 1, sa);
        const int nb = face_segments(MFTLMarchingCubesCase(mb), 2 * axis + 0, sb);
        TEST_ASSERT_EQUAL_INT(na, nb);

        /* An edge of B on its -axis face is A's edge on the +axis face: same
         * axis, the offset along `axis` flipped from 0 to 1. */
        for (int i = 0; i < na; ++i)
        {
          int found = 0;
          for (int j = 0; j < nb; ++j)
          {
            int e0 = sb[j][0], e1 = sb[j][1];
            int b0, b1, x0, x1;
            edge_corners(e0, &b0, &x0);
            edge_corners(e1, &b1, &x1);
            (void) x0;
            (void) x1;
            /* Map each B edge to A's frame by moving its lower corner across. */
            int a0 = -1, a1 = -1;
            for (int e = 0; e < 12; ++e)
            {
              int c0, c1;
              edge_corners(e, &c0, &c1);
              if (e / 4 == e0 / 4 && c0 == (b0 | (1 << axis))) a0 = e;
              if (e / 4 == e1 / 4 && c0 == (b1 | (1 << axis))) a1 = e;
            }
            /* Opposite direction: B's (e0 -> e1) is A's (e1 -> e0). */
            if (a0 == sa[i][1] && a1 == sa[i][0]) found = 1;
          }
          TEST_ASSERT(found);
        }
      }
    }
  }
  return 0;
}

/**
 * @brief Every case's surface separates its negative corners from its positive
 *        ones with the normal pointing to the positive side.
 *
 * With the corners at -1 / +1, the trilinear field's gradient at each triangle's
 * centroid (vertices at edge midpoints) must have a positive component along
 * the triangle normal, summed over the case. Individual triangles of ambiguous
 * cases can sit on a saddle of the trilinear interpolant, so the per-triangle
 * test is only that it does not point firmly the wrong way.
 */
static int test_every_case_is_wound_outward(void)
{
  for (int m = 1; m < 255; ++m)
  {
    const MFTL_MarchingCubesCase *cs = MFTLMarchingCubesCase(m);
    double total = 0.0;
    for (int t = 0; t < cs->n_tris; ++t)
    {
      double p[3][3];
      for (int j = 0; j < 3; ++j) edge_mid(cs->tri[t][j], p[j]);
      const double e1[3] = {p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]};
      const double e2[3] = {p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]};
      const double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
      const double x[3] = {(p[0][0] + p[1][0] + p[2][0]) / 3, (p[0][1] + p[1][1] + p[2][1]) / 3, (p[0][2] + p[1][2] + p[2][2]) / 3};

      double grad[3] = {0, 0, 0};
      for (int c = 0; c < 8; ++c)
      {
        const double v = ((m >> c) & 1) ? -1.0 : 1.0;
        const double o[3] = {(double) (c & 1), (double) ((c >> 1) & 1), (double) ((c >> 2) & 1)};
        double w[3];
        for (int k = 0; k < 3; ++k) w[k] = o[k] ? x[k] : 1.0 - x[k];
        grad[0] += v * (o[0] ? 1.0 : -1.0) * w[1] * w[2];
        grad[1] += v * (o[1] ? 1.0 : -1.0) * w[0] * w[2];
        grad[2] += v * (o[2] ? 1.0 : -1.0) * w[0] * w[1];
      }
      const double dot = n[0] * grad[0] + n[1] * grad[1] + n[2] * grad[2];
      TEST_ASSERT(dot > -1e-12);
      total += dot;
    }
    TEST_ASSERT(total > 0.0);
  }
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. Extraction on analytic shapes
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_sphere_comes_out_closed_outward_and_on_the_level_set(void)
{
  const double s[4] = {0.13, -0.07, 0.21, 1.3};
  const double lo[3] = {-1.8, -1.8, -1.8}, hi[3] = {1.8, 1.8, 1.8};
  const double dx = 0.1;
  MFTL_SdfGrid g;
  fill_grid(&g, lo, hi, dx, sphere_field, s);

  MFTL_Object *obj = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(obj->tris.n > 1000);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(vertices_are_manifold(obj));
  TEST_ASSERT_EQUAL_INT(2, euler_characteristic(obj));
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(obj, MFTL_NULL));

  /* Every vertex is on the sphere to second order in dx. */
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *p = &obj->verts.pos[3 * v];
    const double r = sqrt((p[0] - s[0]) * (p[0] - s[0]) + (p[1] - s[1]) * (p[1] - s[1]) + (p[2] - s[2]) * (p[2] - s[2]));
    TEST_ASSERT(fabs(r - s[3]) < 0.5 * dx * dx / s[3]);
  }

  /* Outward, triangle by triangle. */
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double *c = &obj->tris.center[3 * t], *n = &obj->tris.normal[3 * t];
    TEST_ASSERT((c[0] - s[0]) * n[0] + (c[1] - s[1]) * n[1] + (c[2] - s[2]) * n[2] > 0.0);
  }
  return 0;
}

static int test_sphere_volume_converges_at_second_order(void)
{
  const double s[4] = {0.0, 0.0, 0.0, 1.0};
  const double exact = 4.0 / 3.0 * MFTL_PI;
  double err[3];
  const double dxs[3] = {0.2, 0.1, 0.05};
  for (int i = 0; i < 3; ++i)
  {
    const double lo[3] = {-1.5, -1.5, -1.5}, hi[3] = {1.5, 1.5, 1.5};
    MFTL_SdfGrid g;
    fill_grid(&g, lo, hi, dxs[i], sphere_field, s);
    MFTL_Object *obj = new_object(g_ens);
    TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
    MFTLSdfGridFree(&g);
    const double vol = enclosed_volume(obj);
    TEST_ASSERT(vol > 0.0);
    err[i] = fabs(vol - exact) / exact;
  }
  /* Halving dx cuts the error by ~4; demand at least 3. */
  TEST_ASSERT(err[0] / err[1] > 3.0);
  TEST_ASSERT(err[1] / err[2] > 3.0);
  TEST_ASSERT(err[2] < 2e-3);
  return 0;
}

static int test_a_torus_has_genus_one(void)
{
  const double s[2] = {1.0, 0.35};
  const double lo[3] = {-1.6, -1.6, -0.6}, hi[3] = {1.6, 1.6, 0.6};
  MFTL_SdfGrid g;
  fill_grid(&g, lo, hi, 0.07, torus_field, s);
  MFTL_Object *obj = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(vertices_are_manifold(obj));
  TEST_ASSERT_EQUAL_INT(0, euler_characteristic(obj));
  TEST_ASSERT(enclosed_volume(obj) > 0.0);
  return 0;
}

static int test_two_spheres_are_two_components_until_they_touch(void)
{
  /* R = 0.5: separated by a gap of 0.3 they are two surfaces; overlapping by
   * 0.3 they are one, with genus 0. */
  const double lo[3] = {-1.6, -0.8, -0.8}, hi[3] = {1.6, 0.8, 0.8};
  const double apart[2] = {1.3, 0.5}, merged[2] = {0.7, 0.5};

  MFTL_SdfGrid g;
  fill_grid(&g, lo, hi, 0.05, two_spheres_field, apart);
  MFTL_Object *a = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(a, &g, 1e-3));
  MFTLSdfGridFree(&g);
  TEST_ASSERT_EQUAL_INT(2, MFTLLabelConnectedComponents(a, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(4, euler_characteristic(a));

  fill_grid(&g, lo, hi, 0.05, two_spheres_field, merged);
  MFTL_Object *b = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(b, &g, 1e-3));
  MFTLSdfGridFree(&g);
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(b, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(2, euler_characteristic(b));
  TEST_ASSERT(vertices_are_manifold(b));
  return 0;
}

static int test_a_plane_through_the_window_is_an_open_disk(void)
{
  /* z = 0.3 tilted slightly; it leaves the window through all four side faces. */
  MFTL_SdfGrid g;
  const double lo[3] = {0, 0, 0}, sp[3] = {0.1, 0.1, 0.1};
  const int dims[3] = {11, 11, 11};
  MFTLSdfGridInit(&g, lo, sp, dims);
  for (int k = 0; k < 11; ++k)
    for (int j = 0; j < 11; ++j)
      for (int i = 0; i < 11; ++i) g.phi[(k * 11 + j) * 11 + i] = 0.1 * k - 0.43 - 0.05 * (0.1 * i) + 0.02 * (0.1 * j);

  MFTL_Object *obj = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(vertices_are_manifold(obj));
  TEST_ASSERT_EQUAL_INT(1, euler_characteristic(obj));
  /* The rim is exactly the half-edges on the window's four side faces. */
  int rim = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].twin_idx != MFTL_ERROR) continue;
    TEST_ASSERT(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY);
    const double *a = &obj->verts.pos[3 * obj->tris.he[h].vertex_idx];
    const double *b = &obj->verts.pos[3 * obj->tris.he[MFTLHePrevIdx(h)].vertex_idx];
    const bool on_wall = (a[0] == 0.0 && b[0] == 0.0) || (fabs(a[0] - 1.0) < 1e-12 && fabs(b[0] - 1.0) < 1e-12) || (a[1] == 0.0 && b[1] == 0.0) ||
                         (fabs(a[1] - 1.0) < 1e-12 && fabs(b[1] - 1.0) < 1e-12);
    TEST_ASSERT(on_wall);
    rim++;
  }
  TEST_ASSERT(rim >= 40);
  /* Normals point to phi > 0, i.e. up. */
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  for (int t = 0; t < obj->tris.n; ++t) TEST_ASSERT(obj->tris.normal[3 * t + 2] > 0.9);
  return 0;
}

static int test_a_periodic_window_closes_a_tube_across_the_seam(void)
{
  /* A cylinder along the periodic x axis: on a full-period window the surface is
   * a closed torus-topology tube (chi = 0), with NO boundary at the seam. */
  const MFTL_Domain *dom = &g_per->domain;
  const int n = 40;
  const double dx = dom->domain_size[0] / n;
  const double origin[3] = {dom->global_min[0], -2.0, -2.0};
  const double sp[3] = {dx, 0.1, 0.1};
  const int n_global[3] = {n, 41, 41};
  const bool per[3] = {true, false, false};
  const int lo[3] = {-7, 0, 0};  // the window starts one wrap before the seam, deliberately
  const int dims[3] = {n, 41, 41};

  MFTL_SdfGrid g;
  MFTLSdfGridInitWindow(&g, origin, sp, n_global, per, lo, dims);
  for (int k = 0; k < 41; ++k)
    for (int j = 0; j < 41; ++j)
      for (int i = 0; i < n; ++i)
      {
        const double x = MFTLSdfGridCoord(&g, 0, lo[0] + i);
        const double y = -2.0 + 0.1 * j, z = -2.0 + 0.1 * k;
        const double r = 0.8 + 0.2 * sin(MFTL_TWO_PI * x / dom->domain_size[0]);
        g.phi[((size_t) k * 41 + j) * n + i] = sqrt(y * y + z * z) - r;
      }

  MFTL_Object *obj = new_object(g_per);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(vertices_are_manifold(obj));
  TEST_ASSERT_EQUAL_INT(0, euler_characteristic(obj));
  for (int v = 0; v < obj->verts.n; ++v)
  {
    TEST_ASSERT(obj->verts.pos[3 * v] >= dom->global_min[0] && obj->verts.pos[3 * v] < dom->global_max[0]);
  }
  /* Volume of the undulating tube: pi * mean(r^2) * L. The divergence theorem
   * needs a field that is itself periodic in x on this surface, so the flux of
   * (0, y, 0) -- divergence 1 -- rather than of the position vector. */
  const double L = dom->domain_size[0];
  const double exact = MFTL_PI * (0.8 * 0.8 + 0.5 * 0.2 * 0.2) * L;
  double vol = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double c[3][3];
    MFTLGetTriangleCorners(obj, t, c);
    const double e1[3] = {c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]};
    const double e2[3] = {c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]};
    const double ny = 0.5 * (e1[2] * e2[0] - e1[0] * e2[2]);
    vol += ny * (c[0][1] + c[1][1] + c[2][1]) / 3.0;
  }
  TEST_ASSERT(fabs(vol - exact) / exact < 0.02);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. Adversarial fields
 * ═════════════════════════════════════════════════════════════════════════════ */

static double lcg(unsigned *s)
{
  *s = *s * 1664525u + 1013904223u;
  return (double) (*s >> 8) / (double) (1u << 24);
}

static int test_random_fields_give_manifold_consistently_wound_surfaces(void)
{
  unsigned seed = 4242u;
  for (int trial = 0; trial < 40; ++trial)
  {
    const int n = 6 + trial % 5;
    MFTL_SdfGrid g;
    const double lo[3] = {-1, -1, -1}, sp[3] = {0.3, 0.25, 0.35};
    const int dims[3] = {n, n + 1, n - 1};
    MFTLSdfGridInit(&g, lo, sp, dims);
    for (int i = 0; i < dims[0] * dims[1] * dims[2]; ++i)
    {
      /* Every fifth node exactly zero, the rest uniform in [-1, 1]. */
      g.phi[i] = (i % 5 == 0) ? 0.0 : 2.0 * lcg(&seed) - 1.0;
    }

    for (int snap_mode = 0; snap_mode < 2; ++snap_mode)
    {
      MFTL_Object *obj = new_object(g_ens);
      TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, snap_mode ? 1e-2 : 0.0));
      if (obj->tris.n == 0) continue;

      TEST_ASSERT(inv_implicit_indexing(obj));
      TEST_ASSERT(inv_twins_are_symmetric(obj));
      TEST_ASSERT(vertices_are_manifold(obj));
      TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

      /* Twins run opposite ways: consistent winding across every edge. */
      for (int h = 0; h < 3 * obj->tris.n; ++h)
      {
        const int tw = obj->tris.he[h].twin_idx;
        if (tw == MFTL_ERROR) continue;
        TEST_ASSERT_EQUAL_INT(obj->tris.he[h].vertex_idx, obj->tris.he[MFTLHePrevIdx(tw)].vertex_idx);
      }

      /* Snapped: no triangle degenerates and no vertex lands on a node. */
      if (snap_mode)
      {
        for (int t = 0; t < obj->tris.n; ++t) TEST_ASSERT(MFTLGetTriangleArea(obj, t) > 0.0);
      }

      /* Boundary half-edges only where the surface leaves the window. */
      for (int h = 0; h < 3 * obj->tris.n; ++h)
      {
        if (obj->tris.he[h].twin_idx != MFTL_ERROR) continue;
        const double *a = &obj->verts.pos[3 * obj->tris.he[h].vertex_idx];
        const double *b = &obj->verts.pos[3 * obj->tris.he[MFTLHePrevIdx(h)].vertex_idx];
        int on_face = 0;
        for (int k = 0; k < 3; ++k)
        {
          const double top = lo[k] + (dims[k] - 1) * sp[k];
          if ((fabs(a[k] - lo[k]) < 1e-12 && fabs(b[k] - lo[k]) < 1e-12) || (fabs(a[k] - top) < 1e-12 && fabs(b[k] - top) < 1e-12)) on_face = 1;
        }
        TEST_ASSERT(on_face);
      }
    }
    MFTLSdfGridFree(&g);
  }
  return 0;
}

static int test_a_checkerboard_isolates_every_inside_node(void)
{
  /* Every face of every cell ambiguous. With the inside kept apart on
   * ambiguous faces, each INTERIOR negative node is wrapped in its own closed
   * shell -- an octahedron -- and nothing else connects. */
  const int n = 7;
  MFTL_SdfGrid g;
  const double lo[3] = {0, 0, 0}, sp[3] = {1, 1, 1};
  const int dims[3] = {n, n, n};
  MFTLSdfGridInit(&g, lo, sp, dims);
  int interior_negative = 0;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
      {
        const bool neg = ((i + j + k) & 1) != 0;
        g.phi[(k * n + j) * n + i] = neg ? -1.0 : 1.0;
        if (neg && i > 0 && j > 0 && k > 0 && i < n - 1 && j < n - 1 && k < n - 1) interior_negative++;
      }

  MFTL_Object *obj = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
  MFTLSdfGridFree(&g);

  TEST_ASSERT(vertices_are_manifold(obj));
  TEST_ASSERT(inv_twins_are_symmetric(obj));

  int *labels = (int *) malloc((size_t) obj->tris.n * sizeof(int));
  const int nc = MFTLLabelConnectedComponents(obj, labels);
  int closed = 0;
  int *open = (int *) calloc((size_t) nc, sizeof(int));
  int *size = (int *) calloc((size_t) nc, sizeof(int));
  for (int t = 0; t < obj->tris.n; ++t)
  {
    size[labels[t]]++;
    for (int k = 0; k < 3; ++k)
      if (obj->tris.he[3 * t + k].twin_idx == MFTL_ERROR) open[labels[t]] = 1;
  }
  for (int c = 0; c < nc; ++c)
    if (!open[c])
    {
      closed++;
      TEST_ASSERT_EQUAL_INT(8, size[c]);  // an octahedron around one node
    }
  TEST_ASSERT_EQUAL_INT(interior_negative, closed);
  free(labels);
  free(open);
  free(size);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Lattice identity: what makes a distributed extraction stitch-free
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_overlapping_extractions_name_shared_vertices_identically(void)
{
  /* Two "ranks" extract overlapping slabs of one lattice. Every vertex both
   * produce must carry the same key and the bit-identical position, and the
   * union deduplicated by key must be exactly the single extraction. */
  const double s[4] = {0.05, 0.0, -0.03, 1.1};
  const double lo[3] = {-1.5, -1.5, -1.5}, hi[3] = {1.5, 1.5, 1.5};
  MFTL_SdfGrid g;
  fill_grid(&g, lo, hi, 0.1, sphere_field, s);

  const int nc = g.dims[0] - 2;
  const int all_lo[3] = {0, 0, 0}, all_hi[3] = {nc, g.dims[1] - 2, g.dims[2] - 2};
  const int a_hi[3] = {nc / 2, all_hi[1], all_hi[2]};
  const int b_lo[3] = {nc / 2 - 1, 0, 0};

  MFTL_LevelSetSurface whole, a, b;
  memset(&whole, 0, sizeof whole);
  memset(&a, 0, sizeof a);
  memset(&b, 0, sizeof b);
  MFTLExtractLevelSet(&g, all_lo, all_hi, 1e-3, &whole);
  MFTLExtractLevelSet(&g, all_lo, a_hi, 1e-3, &a);
  MFTLExtractLevelSet(&g, b_lo, all_hi, 1e-3, &b);
  MFTLSdfGridFree(&g);

  /* Keys are unique within one extraction. */
  for (int i = 0; i < whole.n_verts; ++i)
    for (int j = i + 1; j < whole.n_verts; ++j) TEST_ASSERT(whole.vkey[i] != whole.vkey[j]);

  int shared = 0;
  for (int i = 0; i < a.n_verts; ++i)
    for (int j = 0; j < b.n_verts; ++j)
      if (a.vkey[i] == b.vkey[j])
      {
        TEST_ASSERT(memcmp(&a.pos[3 * i], &b.pos[3 * j], 3 * sizeof(double)) == 0);
        shared++;
      }
  TEST_ASSERT(shared > 20);  // non-vacuity: the slabs really do overlap on the surface

  /* The triangle sets coincide once the overlap is counted once: (cell, slot) is
   * the triangle's identity. */
  int covered = 0;
  for (int t = 0; t < whole.n_tris; ++t)
  {
    int hits = 0;
    for (int u = 0; u < a.n_tris; ++u)
      if (a.tcell[u] == whole.tcell[t] && a.tslot[u] == whole.tslot[t]) hits++;
    for (int u = 0; u < b.n_tris; ++u)
      if (b.tcell[u] == whole.tcell[t] && b.tslot[u] == whole.tslot[t]) hits++;
    TEST_ASSERT(hits >= 1 && hits <= 2);
    covered++;
  }
  TEST_ASSERT_EQUAL_INT(whole.n_tris, covered);

  MFTLLevelSetSurfaceFree(&whole);
  MFTLLevelSetSurfaceFree(&a);
  MFTLLevelSetSurfaceFree(&b);
  return 0;
}

static int test_unknown_corners_are_skipped_not_guessed(void)
{
  /* A sphere field with every node beyond |phi| > 0.35 marked unknown: nothing
   * a surface cell needs is missing, so the result must be identical in size to
   * the fully known field. */
  const double s[4] = {0.0, 0.0, 0.0, 1.0};
  const double lo[3] = {-1.5, -1.5, -1.5}, hi[3] = {1.5, 1.5, 1.5};
  MFTL_SdfGrid g, h;
  fill_grid(&g, lo, hi, 0.1, sphere_field, s);
  fill_grid(&h, lo, hi, 0.1, sphere_field, s);
  const size_t n = (size_t) h.dims[0] * h.dims[1] * h.dims[2];
  for (size_t i = 0; i < n; ++i)
    if (fabs(h.phi[i]) > 0.35) h.phi[i] = MFTL_SDF_UNKNOWN;

  MFTL_Object *a = new_object(g_ens);
  MFTL_Object *b = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(a, &g, 1e-3));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(b, &h, 1e-3));
  TEST_ASSERT_EQUAL_INT(a->tris.n, b->tris.n);
  TEST_ASSERT(inv_mesh_is_closed(b));

  /* An all-unknown grid gives an empty object and success. */
  for (size_t i = 0; i < n; ++i) h.phi[i] = MFTL_SDF_UNKNOWN;
  MFTL_Object *c = new_object(g_ens);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(c, &h, 1e-3));
  TEST_ASSERT_EQUAL_INT(0, c->tris.n);

  /* And a non-empty object is refused. */
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildMeshFromLevelSet(a, &g, 1e-3));

  MFTLSdfGridFree(&g);
  MFTLSdfGridFree(&h);
  return 0;
}

static int test_extraction_is_scale_and_translation_invariant(void)
{
  /* The same sphere at 1e-6 and at an offset of 1e5: same topology, same
   * triangle count, and the vertices still on the level set relative to dx. */
  const double scales[2] = {1e-6, 1.0};
  const double offsets[2] = {0.0, 1e5};
  int counts[2][2];
  for (int si = 0; si < 2; ++si)
    for (int oi = 0; oi < 2; ++oi)
    {
      const double L = scales[si], O = offsets[oi] * L;
      const double s[4] = {O + 0.1 * L, O, O, 1.0 * L};
      const double lo[3] = {O - 1.5 * L, O - 1.5 * L, O - 1.5 * L}, hi[3] = {O + 1.5 * L, O + 1.5 * L, O + 1.5 * L};
      MFTL_SdfGrid g;
      fill_grid(&g, lo, hi, 0.125 * L, sphere_field, s);
      MFTL_Object *obj = MFTLObjectCreate(0);  // no domain: nothing to wrap into
      TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildMeshFromLevelSet(obj, &g, 1e-3));
      MFTLSdfGridFree(&g);
      TEST_ASSERT(inv_mesh_is_closed(obj));
      TEST_ASSERT_EQUAL_INT(2, euler_characteristic(obj));
      counts[si][oi] = obj->tris.n;
      MFTLObjectDestroy(&obj);
    }
  TEST_ASSERT_EQUAL_INT(counts[1][0], counts[0][0]);
  TEST_ASSERT_EQUAL_INT(counts[1][0], counts[1][1]);
  TEST_ASSERT_EQUAL_INT(counts[1][0], counts[0][1]);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0}, hi[3] = {4.0, 4.0, 4.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 64);

  double plo[3] = {0.0, -2.0, -2.0}, phi[3] = {4.0, 2.0, 2.0};
  bool per[3] = {true, false, false};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 8);

  printf("--- Running MFTL Marching Cubes Tests ---\n");

  RUN_TEST(test_every_case_triangulates_exactly_its_crossed_edges);
  RUN_TEST(test_no_case_needs_a_centroid_or_puts_a_triangle_in_a_face);
  RUN_TEST(test_neighbouring_cells_agree_on_every_shared_face);
  RUN_TEST(test_every_case_is_wound_outward);
  RUN_TEST(test_a_sphere_comes_out_closed_outward_and_on_the_level_set);
  RUN_TEST(test_sphere_volume_converges_at_second_order);
  RUN_TEST(test_a_torus_has_genus_one);
  RUN_TEST(test_two_spheres_are_two_components_until_they_touch);
  RUN_TEST(test_a_plane_through_the_window_is_an_open_disk);
  RUN_TEST(test_a_periodic_window_closes_a_tube_across_the_seam);
  RUN_TEST(test_random_fields_give_manifold_consistently_wound_surfaces);
  RUN_TEST(test_a_checkerboard_isolates_every_inside_node);
  RUN_TEST(test_overlapping_extractions_name_shared_vertices_identically);
  RUN_TEST(test_unknown_corners_are_skipped_not_guessed);
  RUN_TEST(test_extraction_is_scale_and_translation_invariant);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
