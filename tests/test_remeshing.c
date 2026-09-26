/*
 * @Author: Christian Gorges
 * @Date: 2026-08-15
 *
 * @brief Unit tests for src/remeshing.
 *
 * Organised by what can go wrong, not by function:
 *
 *   1. OPERATORS      each of split/collapse/flip leaves a SOUND mesh. Every one
 *                     of them asserts inv_mesh_is_sound(), because the failure
 *                     mode of a half-edge operator is never a crash -- it is a
 *                     structure that is still traversable and no longer
 *                     describes a surface.
 *   2. PREDICATES     the approvals refuse what they must refuse. These are the
 *                     tests that matter most: an operator applied where it was
 *                     legal is easy, and every clause of MFTLCollapseApproved
 *                     exists because of a mesh that looked fine afterwards.
 *   3. FEATURES       exact counts on shapes whose feature set is known by
 *                     construction (a cube has 12 sharp edges and 8 corners; a
 *                     sphere has none).
 *   4. SMOOTHING      the three projection subspaces, tested by their invariants:
 *                     a corner does not move, a regular ridge is a FIXED POINT,
 *                     and a sphere does not shrink.
 *   5. FULL PASS      convergence to the target length, and what the object looks
 *                     like on return.
 *
 * The shapes are built here rather than generated because their answers have to
 * be known independently of the code under test. A cube's twelve 90-degree edges
 * are a fact about a cube.
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

static void fixture_free(MFTL_Object *obj)
{
  /* The ensemble owns its objects, so removing one by hand would leave a
   * dangling slot. Tests are short-lived; the ensemble is destroyed in main. */
  (void) obj;
}

static MFTL_Object *make_sphere(int refinements)
{
  MFTL_Object *obj = fixture_new();
  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, refinements);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief The unit cube as 12 triangles, every face wound outward. */
static MFTL_Object *make_cube(void)
{
  static const double pos[24] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1};
  static const int tri[36] = {
      0, 3, 2, 0, 2, 1,  // z = 0
      4, 5, 6, 4, 6, 7,  // z = 1
      0, 1, 5, 0, 5, 4,  // y = 0
      1, 2, 6, 1, 6, 5,  // x = 1
      2, 3, 7, 2, 7, 6,  // y = 1
      3, 0, 4, 3, 4, 7  // x = 0
  };

  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, 8, tri, 12);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/**
 * @brief An n-gon bipyramid: a closed surface with exactly ONE feature curve.
 *
 * The equator is a ring of n sharp edges (the two cones meet there at about 90
 * degrees for h == R); the meridian edges are shallow for large n. So every
 * equator vertex has exactly two feature edges and is a RIDGE vertex, and the two
 * apexes have none. That is a feature configuration a sphere cannot produce and a
 * cube cannot either -- a cube is all corners.
 *
 * @param angle_offsets Optional [n] perturbation of the equator angles, for
 *        testing redistribution. MFTL_NULL gives a REGULAR polygon.
 */
static MFTL_Object *make_bipyramid(int n, const double *angle_offsets)
{
  const double R = 1.0;
  const double h = 1.0;

  double *pos = (double *) malloc((size_t) (n + 2) * 3 * sizeof(double));
  int *tri = (int *) malloc((size_t) (2 * n) * 3 * sizeof(int));

  for (int i = 0; i < n; ++i)
  {
    const double th = 2.0 * MFTL_PI * (double) i / (double) n + (angle_offsets ? angle_offsets[i] : 0.0);
    pos[3 * i + 0] = R * cos(th);
    pos[3 * i + 1] = R * sin(th);
    pos[3 * i + 2] = 0.0;
  }

  const int top = n;
  const int bot = n + 1;
  pos[3 * top + 0] = 0.0;
  pos[3 * top + 1] = 0.0;
  pos[3 * top + 2] = h;
  pos[3 * bot + 0] = 0.0;
  pos[3 * bot + 1] = 0.0;
  pos[3 * bot + 2] = -h;

  for (int i = 0; i < n; ++i)
  {
    const int j = (i + 1) % n;
    tri[6 * i + 0] = i;
    tri[6 * i + 1] = j;
    tri[6 * i + 2] = top;
    tri[6 * i + 3] = j;
    tri[6 * i + 4] = i;
    tri[6 * i + 5] = bot;
  }

  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, n + 2, tri, 2 * n);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  free(pos);
  free(tri);
  return obj;
}

/**
 * @brief A flat (nx+1) x (ny+1) grid of vertices as an OPEN triangulated patch.
 *
 * The only open surface available: no open-surface generator has been ported, and
 * every boundary code path -- the split of a rim edge, the collapse that must not
 * pull the rim inwards, the closing edge of an open fan that carries a feature
 * flag with no outgoing half-edge -- is unreachable without one.
 */
static MFTL_Object *make_patch(int nx, int ny)
{
  const int nvx = nx + 1;
  const int nvy = ny + 1;
  const int nv = nvx * nvy;
  const int nt = 2 * nx * ny;

  double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
  int *tri = (int *) malloc((size_t) nt * 3 * sizeof(int));

  for (int j = 0; j < nvy; ++j)
  {
    for (int i = 0; i < nvx; ++i)
    {
      const int v = j * nvx + i;
      pos[3 * v + 0] = (double) i / (double) nx;
      pos[3 * v + 1] = (double) j / (double) ny;
      pos[3 * v + 2] = 0.0;
    }
  }

  int t = 0;
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      const int v00 = j * nvx + i;
      const int v10 = v00 + 1;
      const int v01 = v00 + nvx;
      const int v11 = v01 + 1;

      tri[3 * t + 0] = v00;
      tri[3 * t + 1] = v10;
      tri[3 * t + 2] = v11;
      t++;
      tri[3 * t + 0] = v00;
      tri[3 * t + 1] = v11;
      tri[3 * t + 2] = v01;
      t++;
    }
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

/** @brief Enclosed volume by the divergence theorem. Positive iff wound outward. */
static double enclosed_volume(const MFTL_Object *obj)
{
  double vol = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    const double *a = MFTLVertexPos(obj, v[0]);
    const double *b = MFTLVertexPos(obj, v[1]);
    const double *c = MFTLVertexPos(obj, v[2]);

    vol += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return vol / 6.0;
}

/** @brief Smallest mean-ratio quality over all triangles: 1 equilateral, 0 degenerate. */
static double min_quality(const MFTL_Object *obj)
{
  double worst = 1.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);

    double e[3][3];
    MFTLGetMinimumImageVector(obj->domain, e[0], MFTLVertexPos(obj, v[0]), MFTLVertexPos(obj, v[1]));
    MFTLGetMinimumImageVector(obj->domain, e[1], MFTLVertexPos(obj, v[1]), MFTLVertexPos(obj, v[2]));
    MFTLGetMinimumImageVector(obj->domain, e[2], MFTLVertexPos(obj, v[2]), MFTLVertexPos(obj, v[0]));

    double sum_sq = 0.0;
    for (int k = 0; k < 3; ++k) sum_sq += e[k][0] * e[k][0] + e[k][1] * e[k][1] + e[k][2] * e[k][2];

    const double area = MFTLGetTriangleArea(obj, t);
    const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
    if (q < worst) worst = q;
  }
  return worst;
}

/** @brief Standard deviation of the undirected edge lengths. */
static double edge_length_stddev(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;
  double sum = 0.0;
  double sum_sq = 0.0;
  int n = 0;

  for (int h = 0; h < n_he; ++h)
  {
    const int tw = obj->tris.he[h].twin_idx;
    if (tw != MFTL_ERROR && h > tw) continue;

    const double l = MFTLGetEdgeLength(obj, h);
    sum += l;
    sum_sq += l * l;
    n++;
  }
  if (n == 0) return 0.0;

  const double mean = sum / (double) n;
  const double var = sum_sq / (double) n - mean * mean;
  return (var > 0.0) ? sqrt(var) : 0.0;
}

/** @brief Number of triangles whose normal points into the sphere. */
static int count_inverted_vs_center(const MFTL_Object *obj, const double center[3])
{
  int bad = 0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double n[3];
    double c[3];
    MFTLGetTriangleProperties(obj, t, MFTL_NULL, n, c);

    const double r[3] = {c[0] - center[0], c[1] - center[1], c[2] - center[2]};
    if (n[0] * r[0] + n[1] * r[1] + n[2] * r[2] <= 0.0) bad++;
  }
  return bad;
}

/** @brief Standard set of parameters with feature detection off. */
static MFTL_RemeshParameters params_for(const MFTL_Object *obj, double factor, double feature_angle)
{
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, factor * MFTLMeanEdgeLength(obj), 10, MFTL_SMOOTH_TRIANGLE_BASED, feature_angle);
  return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. OPERATORS
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_split_appends_one_vertex_and_two_triangles(void)
{
  MFTL_Object *obj = make_sphere(1);

  const int v0 = obj->verts.n;
  const int t0 = obj->tris.n;

  const int m = MFTLSplitEdge(obj, 0);
  TEST_ASSERT(m != MFTL_ERROR);

  TEST_ASSERT_EQUAL_INT(v0 + 1, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(t0 + 2, obj->tris.n);

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_halfedge_gids_are_implicit(obj));
  TEST_ASSERT(inv_local_global_ids_are_unique(obj));

  fixture_free(obj);
  return 0;
}

static int test_split_places_the_vertex_at_the_midpoint(void)
{
  MFTL_Object *obj = make_sphere(1);

  const int a = MFTLHeOrigin(obj, 0);
  const int b = MFTLHeTarget(obj, 0);

  double expect[3];
  for (int k = 0; k < 3; ++k) expect[k] = 0.5 * (MFTLVertexPos(obj, a)[k] + MFTLVertexPos(obj, b)[k]);

  const int m = MFTLSplitEdge(obj, 0);
  TEST_ASSERT(m != MFTL_ERROR);

  for (int k = 0; k < 3; ++k) TEST_ASSERT_EQUAL_DOUBLE(expect[k], MFTLVertexPos(obj, m)[k], 1e-15);

  /* The new vertex is joined to both endpoints and to both opposite corners:
   * valence 4 for an interior split, always. */
  int ring[MFTL_MAX_VALENCE];
  TEST_ASSERT_EQUAL_INT(4, MFTLGetVertexOneRingVertices(obj, m, ring, MFTL_MAX_VALENCE, MFTL_FALSE));

  fixture_free(obj);
  return 0;
}

static int test_split_of_a_boundary_edge_makes_two_boundary_edges(void)
{
  MFTL_Object *obj = make_patch(3, 3);

  /* Find a rim half-edge: no twin, and flagged as a real mesh boundary. */
  int h_bnd = MFTL_ERROR;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].twin_idx == MFTL_ERROR)
    {
      h_bnd = h;
      break;
    }
  }
  TEST_ASSERT(h_bnd != MFTL_ERROR);
  TEST_ASSERT(MFTLHeIsMeshBoundary(obj, h_bnd));

  const int t0 = obj->tris.n;
  const int m = MFTLSplitEdge(obj, h_bnd);
  TEST_ASSERT(m != MFTL_ERROR);

  /* One triangle only: a boundary edge has one incident face, not two. */
  TEST_ASSERT_EQUAL_INT(t0 + 1, obj->tris.n);

  /* The midpoint inherits the rim, and so do both halves of the split edge --
   * this is the flag that has to travel with the EDGE while the half-edge
   * records are rewritten into different slots. */
  TEST_ASSERT(MFTLVertexIsMeshBoundary(obj, m));

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));

  int n_rim = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].twin_idx == MFTL_ERROR)
    {
      TEST_ASSERT(MFTLHeIsMeshBoundary(obj, h));
      n_rim++;
    }
  }
  /* A 3x3 patch has 12 rim edges; splitting one makes 13. */
  TEST_ASSERT_EQUAL_INT(13, n_rim);

  fixture_free(obj);
  return 0;
}

static int test_split_preserves_the_feature_flag_on_both_halves(void)
{
  MFTL_Object *obj = make_bipyramid(24, MFTL_NULL);
  MFTLDetectFeatures(obj, 45.0, 45.0, MFTL_NULL);

  int h_feat = MFTL_ERROR;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].flags & MFTL_FLAG_FEATURE)
    {
      h_feat = h;
      break;
    }
  }
  TEST_ASSERT(h_feat != MFTL_ERROR);

  const int a = MFTLHeOrigin(obj, h_feat);
  const int b = MFTLHeTarget(obj, h_feat);
  const int ga = obj->verts.rec[a].global_id;
  const int gb = obj->verts.rec[b].global_id;

  const int m = MFTLSplitEdge(obj, h_feat);
  TEST_ASSERT(m != MFTL_ERROR);
  MFTLObjectBuildDerived(obj);

  /* Both halves of a feature edge must still be feature edges: a ridge that
   * develops a gap at a split point is invisible until the smoother releases the
   * vertex in the gap and the curve kinks. */
  int found = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int o = MFTLHeOrigin(obj, h);
    const int tgt = MFTLHeTarget(obj, h);
    if (o != m && tgt != m) continue;

    const int other = (o == m) ? tgt : o;
    const int g = obj->verts.rec[other].global_id;
    if (g != ga && g != gb) continue;

    TEST_ASSERT(obj->tris.he[h].flags & MFTL_FLAG_FEATURE);
    found++;
  }
  TEST_ASSERT(found == 4);  // two sub-edges, two directions each

  fixture_free(obj);
  return 0;
}

static int test_collapse_removes_one_vertex_and_two_triangles(void)
{
  MFTL_Object *obj = make_sphere(2);
  MFTLDetectFeatures(obj, 180.0, 45.0, MFTL_NULL);

  /* Both bounds relaxed: min so that every edge is short enough to be a
   * candidate, max so that the stretch clause does not veto the merged vertex's
   * new edges -- a collapse on a uniform mesh makes edges about 1.6x the mean,
   * which is outside the 4/3 band by design. */
  MFTL_RemeshParameters p = params_for(obj, 1.0, 180.0);
  p.min_edge_length = 1e9;
  p.max_edge_length = 1e9;
  MFTLRemeshParamsFinalise(&p);

  int h = MFTL_ERROR;
  for (int i = 0; i < 3 * obj->tris.n; ++i)
  {
    if (MFTLCollapseApproved(obj, i, &p, MFTL_NULL))
    {
      h = i;
      break;
    }
  }
  TEST_ASSERT(h != MFTL_ERROR);

  const int v0 = obj->verts.n;
  const int t0 = obj->tris.n;

  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCollapseEdge(obj, h));
  TEST_ASSERT_EQUAL_INT(2, MFTLRemeshCompactObject(obj, MFTL_NULL) - 1);  // 1 vertex + 2 triangles

  TEST_ASSERT_EQUAL_INT(v0 - 1, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(t0 - 2, obj->tris.n);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_local_global_ids_are_unique(obj));

  fixture_free(obj);
  return 0;
}

static int test_flip_preserves_counts_and_swaps_the_diagonal(void)
{
  MFTL_Object *obj = make_sphere(2);

  const int h = 0;
  const int a = MFTLHeOrigin(obj, h);
  const int b = MFTLHeTarget(obj, h);
  const int c = MFTLHeTarget(obj, MFTLHeNextIdx(h));
  const int d = MFTLHeTarget(obj, MFTLHeNextIdx(MFTLHeTwin(obj, h)));

  const int v0 = obj->verts.n;
  const int t0 = obj->tris.n;
  const int gid0 = obj->tris.rec[MFTLHeTriIdx(h)].global_id;

  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLFlipEdge(obj, h));

  /* A flip allocates nothing and retires nothing. */
  TEST_ASSERT_EQUAL_INT(v0, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(t0, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(gid0, obj->tris.rec[MFTLHeTriIdx(h)].global_id);

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_halfedge_gids_are_implicit(obj));

  /* C and D are now joined, A and B are not. */
  int ring[MFTL_MAX_VALENCE];
  int n = MFTLGetVertexOneRingVertices(obj, c, ring, MFTL_MAX_VALENCE, MFTL_FALSE);
  int has_d = 0;
  for (int i = 0; i < n; ++i)
    if (ring[i] == d) has_d = 1;
  TEST_ASSERT(has_d);

  n = MFTLGetVertexOneRingVertices(obj, a, ring, MFTL_MAX_VALENCE, MFTL_FALSE);
  int has_b = 0;
  for (int i = 0; i < n; ++i)
    if (ring[i] == b) has_b = 1;
  TEST_ASSERT(!has_b);

  fixture_free(obj);
  return 0;
}

static int test_flip_is_an_involution(void)
{
  MFTL_Object *obj = make_sphere(2);

  const int t1 = MFTLHeTriIdx(0);
  const int t2 = MFTLHeTriIdx(MFTLHeTwin(obj, 0));

  int before1[3];
  int before2[3];
  MFTLGetTriangleVertices(obj, t1, before1);
  MFTLGetTriangleVertices(obj, t2, before2);

  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLFlipEdge(obj, 0));

  /* After the first flip the shared edge is slot 1 of t1 (the new diagonal). */
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLFlipEdge(obj, 3 * t1 + 1));

  int after1[3];
  int after2[3];
  MFTLGetTriangleVertices(obj, t1, after1);
  MFTLGetTriangleVertices(obj, t2, after2);

  /* The same two triangles, up to rotation and to which slot each landed in. */
  int matched = 0;
  for (int rot = 0; rot < 3; ++rot)
  {
    if (after1[0] == before1[rot] && after1[1] == before1[(rot + 1) % 3] && after1[2] == before1[(rot + 2) % 3]) matched++;
    if (after1[0] == before2[rot] && after1[1] == before2[(rot + 1) % 3] && after1[2] == before2[(rot + 2) % 3]) matched++;
    if (after2[0] == before1[rot] && after2[1] == before1[(rot + 1) % 3] && after2[2] == before1[(rot + 2) % 3]) matched++;
    if (after2[0] == before2[rot] && after2[1] == before2[(rot + 1) % 3] && after2[2] == before2[(rot + 2) % 3]) matched++;
  }
  TEST_ASSERT_EQUAL_INT(2, matched);

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));

  fixture_free(obj);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. PREDICATES
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_collapse_refuses_to_dissolve_a_tetrahedron(void)
{
  /* Every vertex of a tetrahedron has valence 3, so any collapse leaves a wing
   * vertex with two triangles sharing two edges -- a pocket, not a surface. The
   * link condition alone does NOT catch this (each pair of vertices does share
   * exactly the other two), which is why the valence clauses exist separately. */
  static const double pos[12] = {1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1};
  static const int tri[12] = {0, 1, 2, 0, 2, 3, 0, 3, 1, 1, 3, 2};

  MFTL_Object *obj = fixture_new();
  MFTLBuildIndexedMesh(obj, pos, 4, tri, 4);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  MFTL_RemeshParameters p = params_for(obj, 1.0, 180.0);
  p.min_edge_length = 1e9;  // no edge is too long to be a candidate
  MFTLRemeshParamsFinalise(&p);

  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    int reason = MFTL_REMESH_REJECT_NONE;
    TEST_ASSERT(!MFTLCollapseApproved(obj, h, &p, &reason));
    TEST_ASSERT(reason == MFTL_REMESH_REJECT_LINK);
  }

  fixture_free(obj);
  return 0;
}

static int test_collapse_refuses_to_pull_a_boundary_vertex_inward(void)
{
  MFTL_Object *obj = make_patch(4, 4);
  MFTLDetectFeatures(obj, 180.0, 45.0, MFTL_NULL);

  MFTL_RemeshParameters p = params_for(obj, 1.0, 180.0);
  p.min_edge_length = 1e9;
  MFTLRemeshParamsFinalise(&p);

  int checked = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int a = MFTLHeOrigin(obj, h);
    const int b = MFTLHeTarget(obj, h);

    /* An edge running from the rim into the interior: collapsing it retires the
     * rim vertex and the patch's edge moves. */
    if (!MFTLVertexIsMeshBoundary(obj, a) || MFTLVertexIsMeshBoundary(obj, b)) continue;

    int reason = MFTL_REMESH_REJECT_NONE;
    TEST_ASSERT(!MFTLCollapseApproved(obj, h, &p, &reason));
    TEST_ASSERT(reason == MFTL_REMESH_REJECT_FEATURE);
    checked++;
  }
  TEST_ASSERT(checked > 0);

  fixture_free(obj);
  return 0;
}

static int test_collapse_refuses_a_corner(void)
{
  MFTL_Object *obj = make_cube();
  MFTLDetectFeatures(obj, 45.0, 45.0, MFTL_NULL);

  MFTL_RemeshParameters p = params_for(obj, 1.0, 45.0);
  p.min_edge_length = 1e9;
  MFTLRemeshParamsFinalise(&p);

  /* Every vertex of a cube is a corner, so no collapse may retire any of them. */
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    int reason = MFTL_REMESH_REJECT_NONE;
    TEST_ASSERT(!MFTLCollapseApproved(obj, h, &p, &reason));
  }

  fixture_free(obj);
  return 0;
}

static int test_collapse_refuses_a_fold(void)
{
  /*
   * A fold in the PLANE, where "inverted" is unambiguous and needs no tolerance.
   *
   * Vertex B of a flat patch is dragged far outside it, PERPENDICULAR to the
   * edge A-B. That detail is the test: the survivor always lands on the segment
   * A-B, so dragging B along the original A-B direction moves it out through the
   * two triangles the collapse removes anyway and inverts nothing. Turning the
   * direction makes the segment leave A's one-ring hexagon through a SURVIVING
   * triangle, whose normal then goes from +z to exactly -z.
   *
   * Every other clause approves it: the link condition holds (A and B share
   * exactly two neighbours), the valences are ample, and both length bounds are
   * relaxed here so the stretch clause cannot claim the credit. So the fold guard
   * is the only thing that can refuse it, and switching the guard off must let it
   * through -- otherwise this test would pass for the wrong reason.
   */
  MFTL_Object *obj = make_patch(4, 4);

  const int a = 2 * 5 + 2;  // centre of the grid, valence 6
  const int b = a + 1;  // its neighbour in +x

  double *pb = MFTLVertexPosMut(obj, b);
  pb[1] = 3.0;  // far outside the square, across the A-B direction

  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLDetectFeatures(obj, 180.0, 45.0, MFTL_NULL);

  MFTL_RemeshParameters guarded = params_for(obj, 1.0, 180.0);
  guarded.min_edge_length = 1e9;
  guarded.max_edge_length = 1e9;
  MFTLRemeshParamsFinalise(&guarded);

  MFTL_RemeshParameters unguarded = guarded;
  unguarded.max_normal_deviation_cos = -2.0;  // below every possible cosine: no guard at all

  int h_ab = MFTL_ERROR;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (MFTLHeOrigin(obj, h) == a && MFTLHeTarget(obj, h) == b)
    {
      h_ab = h;
      break;
    }
  }
  TEST_ASSERT(h_ab != MFTL_ERROR);

  int reason = MFTL_REMESH_REJECT_NONE;
  TEST_ASSERT(MFTLCollapseApproved(obj, h_ab, &unguarded, MFTL_NULL));  // otherwise legal...
  TEST_ASSERT(!MFTLCollapseApproved(obj, h_ab, &guarded, &reason));  // ...and the guard stops it
  TEST_ASSERT_EQUAL_INT(MFTL_REMESH_REJECT_FOLD, reason);

  fixture_free(obj);
  return 0;
}

static int test_split_only_offers_each_edge_once(void)
{
  MFTL_Object *obj = make_sphere(2);

  MFTL_RemeshParameters p = params_for(obj, 0.5, 180.0);

  /* Both directions of an edge describe the same edge. If the sweep approved
   * both, every long edge would be split twice. */
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int tw = obj->tris.he[h].twin_idx;
    if (tw == MFTL_ERROR) continue;
    TEST_ASSERT(!(MFTLSplitApproved(obj, h, &p) && MFTLSplitApproved(obj, tw, &p)));
  }

  fixture_free(obj);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. FEATURES
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_a_smooth_sphere_has_no_features(void)
{
  MFTL_Object *obj = make_sphere(3);

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  TEST_ASSERT_EQUAL_INT(0, MFTLDetectFeatures(obj, 45.0, 45.0, &st));
  TEST_ASSERT_EQUAL_INT(0, st.n_corner_vertices);
  TEST_ASSERT_EQUAL_INT(0, st.n_ridge_vertices);
  TEST_ASSERT_EQUAL_INT(obj->verts.n, st.n_free_vertices);

  fixture_free(obj);
  return 0;
}

static int test_a_cube_has_twelve_feature_edges_and_eight_corners(void)
{
  MFTL_Object *obj = make_cube();

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  /* Facts about a cube, not about the code: twelve 90-degree edges, six
   * 0-degree face diagonals, and every vertex a junction of three sharp edges. */
  TEST_ASSERT_EQUAL_INT(12, MFTLDetectFeatures(obj, 45.0, 45.0, &st));
  TEST_ASSERT_EQUAL_INT(12, st.n_feature_edges);
  TEST_ASSERT_EQUAL_INT(8, st.n_corner_vertices);
  TEST_ASSERT_EQUAL_INT(0, st.n_ridge_vertices);

  for (int v = 0; v < obj->verts.n; ++v) TEST_ASSERT(obj->verts.rec[v].flags & MFTL_FLAG_CORNER);

  fixture_free(obj);
  return 0;
}

static int test_a_threshold_of_180_disables_detection(void)
{
  MFTL_Object *obj = make_cube();

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  TEST_ASSERT_EQUAL_INT(0, MFTLDetectFeatures(obj, 180.0, 45.0, &st));
  TEST_ASSERT_EQUAL_INT(0, st.n_corner_vertices);
  TEST_ASSERT_EQUAL_INT(0, st.n_ridge_vertices);
  TEST_ASSERT_EQUAL_INT(obj->verts.n, st.n_free_vertices);

  fixture_free(obj);
  return 0;
}

static int test_a_bipyramid_equator_is_a_ridge(void)
{
  const int n = 24;
  MFTL_Object *obj = make_bipyramid(n, MFTL_NULL);

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  /* Exactly the n equator edges are sharp; the meridians of a 24-gon cone are
   * about 15 degrees apart and are not. Every equator vertex therefore has
   * exactly two feature edges and may SLIDE along the ring -- the one case a cube
   * (all corners) and a sphere (no features) between them never produce. */
  TEST_ASSERT_EQUAL_INT(n, MFTLDetectFeatures(obj, 45.0, 45.0, &st));
  TEST_ASSERT_EQUAL_INT(n, st.n_ridge_vertices);
  TEST_ASSERT_EQUAL_INT(0, st.n_corner_vertices);
  TEST_ASSERT_EQUAL_INT(2, st.n_free_vertices);  // the two apexes

  fixture_free(obj);
  return 0;
}

static int test_a_patch_rim_is_a_feature_at_any_threshold(void)
{
  MFTL_Object *obj = make_patch(4, 4);

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  /* A flat patch has no dihedral angle anywhere, so the only features are its
   * rim -- and a rim is a feature of the SURFACE, not of its curvature, so
   * disabling angle detection must not disable it. 4x4 patch: 16 rim edges. */
  TEST_ASSERT_EQUAL_INT(16, MFTLDetectFeatures(obj, 180.0, 45.0, &st));

  /* Each rim vertex has exactly two rim edges except the four grid corners --
   * which are geometric corners of the patch, and are pinned. Sixteen rim
   * vertices in all. */
  TEST_ASSERT_EQUAL_INT(16, st.n_ridge_vertices + st.n_corner_vertices);

  for (int v = 0; v < obj->verts.n; ++v)
  {
    const bool on_rim = MFTLVertexIsMeshBoundary(obj, v);
    const bool classified = (obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER)) != 0;
    TEST_ASSERT(on_rim == classified);
  }

  fixture_free(obj);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. SMOOTHING
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_corners_do_not_move(void)
{
  MFTL_Object *obj = make_cube();
  MFTLDetectFeatures(obj, 45.0, 45.0, MFTL_NULL);

  double before[24];
  memcpy(before, obj->verts.pos, sizeof(before));

  MFTL_RemeshParameters p = params_for(obj, 1.0, 45.0);
  MFTLSmoothTangential(obj, &p, 20);

  /* Every vertex of a cube is a corner, so twenty sweeps must be the identity --
   * exactly, not approximately. */
  for (int i = 0; i < 24; ++i) TEST_ASSERT_EQUAL_DOUBLE(before[i], obj->verts.pos[i], 0.0);

  fixture_free(obj);
  return 0;
}

static int test_a_regular_ridge_is_a_fixed_point(void)
{
  /*
   * THE decisive test for the 1D projection.
   *
   * A regular polygon ridge is already perfectly redistributed, so the smoother
   * must not move it at all. Projecting the Laplacian displacement onto the ridge
   * tangent gives exactly zero by symmetry: the displacement towards the chord
   * midpoint is purely radial, the tangent is perpendicular to it.
   *
   * Moving the vertex TO that chord midpoint instead -- the un-projected rule --
   * shrinks the ring by cos(pi/n) per sweep. For n = 24 and 20 sweeps that
   * is a 16 % collapse of a feature curve the caller asked to have PRESERVED, and
   * nothing reports it.
   */
  const int n = 24;
  MFTL_Object *obj = make_bipyramid(n, MFTL_NULL);
  MFTLDetectFeatures(obj, 45.0, 45.0, MFTL_NULL);

  double *before = (double *) malloc((size_t) obj->verts.n * 3 * sizeof(double));
  memcpy(before, obj->verts.pos, (size_t) obj->verts.n * 3 * sizeof(double));

  MFTL_RemeshParameters p = params_for(obj, 1.0, 45.0);
  MFTLSmoothTangential(obj, &p, 20);

  double max_drift = 0.0;
  double min_radius = 1e30;

  for (int v = 0; v < n; ++v)
  {
    for (int k = 0; k < 3; ++k) max_drift = MFTL_MAX(max_drift, fabs(before[3 * v + k] - obj->verts.pos[3 * v + k]));

    const double r = sqrt(obj->verts.pos[3 * v + 0] * obj->verts.pos[3 * v + 0] + obj->verts.pos[3 * v + 1] * obj->verts.pos[3 * v + 1]);
    min_radius = MFTL_MIN(min_radius, r);
  }

  printf("\n    regular ridge: max drift %.3e, min radius %.12f (chord rule would give %.4f) ", max_drift, min_radius, pow(cos(MFTL_PI / n), 20.0));

  TEST_ASSERT(max_drift < 1e-12);
  TEST_ASSERT(min_radius > 1.0 - 1e-12);

  free(before);
  fixture_free(obj);
  return 0;
}

static int test_ridge_smoothing_redistributes_along_the_curve(void)
{
  /* The complement of the previous test: a ridge that is NOT evenly spaced must
   * even out, and must do so without leaving the plane the ridge lies in. */
  const int n = 24;
  double offsets[24];
  for (int i = 0; i < n; ++i) offsets[i] = (i % 2) ? 0.06 : -0.06;

  MFTL_Object *obj = make_bipyramid(n, offsets);
  MFTLDetectFeatures(obj, 45.0, 45.0, MFTL_NULL);

  double spread_before = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const int j = (i + 1) % n;
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = obj->verts.pos[3 * j + k] - obj->verts.pos[3 * i + k];
    const double l = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    spread_before = MFTL_MAX(spread_before, l);
  }

  MFTL_RemeshParameters p = params_for(obj, 1.0, 45.0);
  MFTLSmoothTangential(obj, &p, 40);

  double spread_after = 0.0;
  double max_out_of_plane = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const int j = (i + 1) % n;
    double d[3];
    for (int k = 0; k < 3; ++k) d[k] = obj->verts.pos[3 * j + k] - obj->verts.pos[3 * i + k];
    spread_after = MFTL_MAX(spread_after, sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]));

    max_out_of_plane = MFTL_MAX(max_out_of_plane, fabs(obj->verts.pos[3 * i + 2]));
  }

  printf("\n    ridge spacing: longest gap %.5f -> %.5f, out of plane %.2e ", spread_before, spread_after, max_out_of_plane);

  TEST_ASSERT(spread_after < spread_before);  // it redistributed
  TEST_ASSERT(max_out_of_plane < 1e-12);  // and stayed on the curve's plane

  fixture_free(obj);
  return 0;
}

static int test_tangential_smoothing_does_not_shrink_a_sphere(void)
{
  /*
   * The 2D projection, tested by what it exists to prevent. An unprojected
   * Laplacian pulls every vertex of a convex surface inwards by about h^2 * kappa
   * / 8 per sweep, so it visibly deflates a sphere; removing the normal component
   * of the displacement is exactly the term that stops it.
   */
  MFTL_Object *obj = make_sphere(3);

  /* Perturb tangentially-ish so the smoother has real work to do, but keep the
   * surface where it is: scatter each vertex and renormalise it back to r = 1. */
  unsigned seed = 12345u;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *p = MFTLVertexPosMut(obj, v);
    for (int k = 0; k < 3; ++k)
    {
      seed = seed * 1103515245u + 12345u;
      p[k] += 0.10 * ((double) ((seed >> 16) & 0x7fff) / 32767.0 - 0.5);
    }
    const double r = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    for (int k = 0; k < 3; ++k) p[k] /= r;
  }

  MFTLDetectFeatures(obj, 180.0, 45.0, MFTL_NULL);

  const double stddev_before = edge_length_stddev(obj);

  MFTL_RemeshParameters p = params_for(obj, 1.0, 180.0);
  MFTLSmoothTangential(obj, &p, 50);

  double mean_r = 0.0;
  double min_r = 1e30;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *q = MFTLVertexPos(obj, v);
    const double r = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    mean_r += r;
    min_r = MFTL_MIN(min_r, r);
  }
  mean_r /= (double) obj->verts.n;

  const double stddev_after = edge_length_stddev(obj);

  printf("\n    sphere after 50 sweeps: mean r %.6f, min r %.6f, edge stddev %.5f -> %.5f ", mean_r, min_r, stddev_before, stddev_after);

  TEST_ASSERT(fabs(mean_r - 1.0) < 0.01);  // no deflation
  TEST_ASSERT(min_r > 0.97);
  TEST_ASSERT(stddev_after < stddev_before);  // and it did smooth

  fixture_free(obj);
  return 0;
}

static int test_area_weighting_is_actually_applied(void)
{
  /*
   * MFTL_SMOOTH_TRIANGLE_BASED must be the AREA-WEIGHTED centroid mean. Dropping
   * the weight leaves a smoother that is very nearly MFTL_SMOOTH_VERTEX_BASED,
   * so a caller choosing
   * it for isotropy silently got the umbrella operator. The two modes must
   * therefore differ on a fan with unequal triangle areas.
   */
  MFTL_Object *obj_a = make_sphere(2);
  MFTL_Object *obj_b = make_sphere(2);

  /* Make one vertex's fan strongly anisotropic in AREA while leaving the
   * neighbour positions -- which is all the umbrella operator sees -- untouched
   * for both copies. */
  for (int v = 0; v < 3; ++v)
  {
    double *pa = MFTLVertexPosMut(obj_a, v);
    double *pb = MFTLVertexPosMut(obj_b, v);
    for (int k = 0; k < 3; ++k)
    {
      pa[k] *= 1.6;
      pb[k] *= 1.6;
    }
  }

  MFTLDetectFeatures(obj_a, 180.0, 45.0, MFTL_NULL);
  MFTLDetectFeatures(obj_b, 180.0, 45.0, MFTL_NULL);

  MFTL_RemeshParameters pa = params_for(obj_a, 1.0, 180.0);
  pa.smooth_mode = MFTL_SMOOTH_TRIANGLE_BASED;
  MFTL_RemeshParameters pb = pa;
  pb.smooth_mode = MFTL_SMOOTH_VERTEX_BASED;

  MFTLSmoothTangential(obj_a, &pa, 1);
  MFTLSmoothTangential(obj_b, &pb, 1);

  double max_diff = 0.0;
  for (int i = 0; i < 3 * obj_a->verts.n; ++i) max_diff = MFTL_MAX(max_diff, fabs(obj_a->verts.pos[i] - obj_b->verts.pos[i]));

  printf("\n    area-weighted vs umbrella: max position difference %.3e ", max_diff);
  TEST_ASSERT(max_diff > 1e-4);

  fixture_free(obj_a);
  fixture_free(obj_b);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. FULL PASS
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_remesh_refines_towards_the_target(void)
{
  MFTL_Object *obj = make_sphere(2);

  const double target = 0.5 * MFTLMeanEdgeLength(obj);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, target, 12, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.smooth_iterations = 3;

  MFTL_RemeshStats st;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, &st));

  const double mean = MFTLMeanEdgeLength(obj);
  printf("\n    refine: %d tris, mean edge %.5f (target %.5f, band %.5f..%.5f), splits %d ", obj->tris.n, mean, target, p.min_edge_length,
         p.max_edge_length, st.n_split);

  TEST_ASSERT(st.n_split > 0);
  TEST_ASSERT(mean > p.min_edge_length && mean < p.max_edge_length);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));
  TEST_ASSERT(inv_halfedge_gids_are_implicit(obj));

  fixture_free(obj);
  return 0;
}

static int test_remesh_coarsens_towards_the_target(void)
{
  MFTL_Object *obj = make_sphere(4);
  const int t0 = obj->tris.n;

  const double target = 2.0 * MFTLMeanEdgeLength(obj);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, target, 12, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.smooth_iterations = 3;

  MFTL_RemeshStats st;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, &st));

  printf("\n    coarsen: %d -> %d tris, mean edge %.5f (target %.5f), collapses %d ", t0, obj->tris.n, MFTLMeanEdgeLength(obj), target, st.n_collapse);

  TEST_ASSERT(st.n_collapse > 0);
  TEST_ASSERT(obj->tris.n < t0 / 2);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));

  fixture_free(obj);
  return 0;
}

static int test_remesh_improves_quality_and_inverts_nothing(void)
{
  MFTL_Object *obj = make_sphere(3);

  /* Scramble the vertex distribution without moving the surface, so the mesh is
   * badly graded but geometrically still a sphere. */
  unsigned seed = 987654321u;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *q = MFTLVertexPosMut(obj, v);
    for (int k = 0; k < 3; ++k)
    {
      seed = seed * 1103515245u + 12345u;
      q[k] += 0.22 * ((double) ((seed >> 16) & 0x7fff) / 32767.0 - 0.5);
    }
    const double r = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    for (int k = 0; k < 3; ++k) q[k] /= r;
  }
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  const double q_before = min_quality(obj);
  const double vol_before = enclosed_volume(obj);

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, MFTLMeanEdgeLength(obj), 15, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.smooth_iterations = 3;

  MFTL_RemeshStats st;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, &st));

  const double q_after = min_quality(obj);
  const double vol_after = enclosed_volume(obj);
  const double center[3] = {0.0, 0.0, 0.0};

  printf("\n    quality: min q %.4f -> %.4f, volume %.5f -> %.5f, inverted %d ", q_before, q_after, vol_before, vol_after, count_inverted_vs_center(obj, center));

  TEST_ASSERT(q_after > q_before);
  TEST_ASSERT(q_after > 0.3);
  TEST_ASSERT_EQUAL_INT(0, count_inverted_vs_center(obj, center));
  TEST_ASSERT(fabs(vol_after - vol_before) / vol_before < 0.05);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));

  fixture_free(obj);
  return 0;
}

static int test_remesh_leaves_the_object_ready_to_stream(void)
{
  MFTL_Object *obj = make_sphere(2);

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.7 * MFTLMeanEdgeLength(obj), 6, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);

  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, MFTL_NULL));

  /* The documented postcondition: compact connectivity describes the remeshed
   * half-edge structure. */
  TEST_ASSERT(MFTLObjectHasTopology(obj));
  TEST_ASSERT(inv_derived_matches_topology(obj));

  /* tris.normal must describe THIS mesh, not the one before the pass. */
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double n[3];
    MFTLGetTriangleNormal(obj, t, n);
    for (int k = 0; k < 3; ++k) TEST_ASSERT_EQUAL_DOUBLE(n[k], obj->tris.normal[3 * t + k], 1e-12);
  }

  /* ...and be unit, including on whatever slivers the pass left behind. A
   * remeshed mesh is exactly where short normals would first appear, because it
   * is the only place in the library that deliberately creates triangles far
   * smaller than the ones it started with. */
  TEST_ASSERT(inv_triangle_normals_are_unit(obj, 1e-12));
  TEST_ASSERT(inv_vertex_normals_are_unit(obj, 1e-12));

  fixture_free(obj);
  return 0;
}

static int test_remesh_converges_and_then_does_nothing(void)
{
  MFTL_Object *obj = make_sphere(3);

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, MFTLMeanEdgeLength(obj), 10, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);

  MFTL_RemeshStats first;
  MFTLRemeshObject(obj, &p, &first);

  MFTL_RemeshStats second;
  MFTLRemeshObject(obj, &p, &second);

  const int ops_first = first.n_split + first.n_collapse + first.n_flip;
  const int ops_second = second.n_split + second.n_collapse + second.n_flip;

  printf("\n    idempotence: %d ops then %d ops ", ops_first, ops_second);

  /* A remesher that keeps finding work on a mesh it has already remeshed is
   * oscillating -- most often split against collapse, which is what the 4/5 and
   * 4/3 band exists to prevent, or flip against flip, which the STRICT valence
   * inequality prevents. */
  TEST_ASSERT(ops_second <= ops_first);
  TEST_ASSERT(ops_second < obj->tris.n / 10);

  fixture_free(obj);
  return 0;
}

static int test_remesh_preserves_a_cube(void)
{
  /* The end-to-end feature test: a cube remeshed at a fine target length must
   * come out as a cube. Its eight corners may not move, its twelve edges must
   * stay straight, and every vertex must stay on the surface -- which for a cube
   * means each coordinate is 0, 1, or between them on exactly one face. */
  MFTL_Object *obj = make_cube();

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.30, 12, MFTL_SMOOTH_TRIANGLE_BASED, 45.0);
  p.smooth_iterations = 2;

  MFTL_RemeshStats st;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, &st));

  int off_surface = 0;
  double worst = 0.0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *q = MFTLVertexPos(obj, v);

    /* Distance to the cube's surface: zero if the vertex lies on any face. */
    double best = 1e30;
    for (int k = 0; k < 3; ++k)
    {
      best = MFTL_MIN(best, fabs(q[k]));
      best = MFTL_MIN(best, fabs(q[k] - 1.0));
    }
    worst = MFTL_MAX(worst, best);
    if (best > 1e-9) off_surface++;
  }

  printf("\n    cube: %d tris, %d vertices off the surface (worst %.2e), features %d, corners %d ", obj->tris.n, off_surface, worst, st.n_feature_edges,
         st.n_corner_vertices);

  TEST_ASSERT_EQUAL_INT(0, off_surface);
  TEST_ASSERT(st.n_corner_vertices >= 8);
  TEST_ASSERT(fabs(enclosed_volume(obj) - 1.0) < 1e-9);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  TEST_ASSERT(inv_mesh_is_closed(obj));

  fixture_free(obj);
  return 0;
}

static int test_remesh_of_an_open_patch_keeps_its_rim(void)
{
  MFTL_Object *obj = make_patch(4, 4);

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.15, 10, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.smooth_iterations = 2;

  MFTL_RemeshStats st;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(obj, &p, &st));

  /* The rim must still be a closed loop of boundary edges, every rim vertex must
   * still sit on the square's border, and nothing may have drifted off the plane.
   */
  int n_rim = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].twin_idx != MFTL_ERROR) continue;
    TEST_ASSERT(MFTLHeIsMeshBoundary(obj, h));
    n_rim++;
  }
  TEST_ASSERT(n_rim >= 16);

  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *q = MFTLVertexPos(obj, v);
    TEST_ASSERT(fabs(q[2]) < 1e-12);
    TEST_ASSERT(q[0] > -1e-9 && q[0] < 1.0 + 1e-9);
    TEST_ASSERT(q[1] > -1e-9 && q[1] < 1.0 + 1e-9);

    if (!MFTLVertexIsMeshBoundary(obj, v)) continue;

    const double d = MFTL_MIN(MFTL_MIN(fabs(q[0]), fabs(q[0] - 1.0)), MFTL_MIN(fabs(q[1]), fabs(q[1] - 1.0)));
    TEST_ASSERT(d < 1e-9);
  }

  printf("\n    patch: %d tris, %d rim edges ", obj->tris.n, n_rim);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));

  fixture_free(obj);
  return 0;
}

static int test_compaction_removes_orphans(void)
{
  MFTL_Object *obj = make_sphere(2);

  /* A vertex nothing references. Compaction must drop it and say so, rather than
   * carry a record whose out_he_idx points at a half-edge that no longer exists.
   */
  const int v_extra = obj->verts.n;
  MFTLObjectSetCounts(obj, obj->verts.n + 1, obj->verts.n + 1, obj->tris.n, obj->tris.n);
  obj->verts.rec[v_extra].out_he_idx = MFTL_ERROR;
  obj->verts.rec[v_extra].global_id = obj->next_vertex_gid++;
  obj->verts.rec[v_extra].owner_proc = 0;
  obj->verts.rec[v_extra].flags = 0;
  MFTLObjectBuildDerived(obj);

  MFTL_RemeshStats st;
  memset(&st, 0, sizeof(st));

  TEST_ASSERT_EQUAL_INT(1, MFTLRemeshCompactObject(obj, &st));
  TEST_ASSERT_EQUAL_INT(1, st.n_orphans_removed);
  TEST_ASSERT_EQUAL_INT(0, st.n_vertices_removed);
  TEST_ASSERT_EQUAL_INT(v_extra, obj->verts.n);

  TEST_ASSERT(inv_mesh_is_sound(obj, 0));

  fixture_free(obj);
  return 0;
}

/**
 * @brief Remeshing commutes with a change of units, to the bit.
 *
 * A front-tracking run in SI units meshes a millimetre droplet with edges near
 * 1e-4, so every test the remesher makes has to be DIMENSIONLESS -- a ratio, or a
 * comparison against zero. The fold test used to compare a squared area vector
 * (dimension L^4) against 1e-15; below an edge length of about 1.8e-4 every
 * triangle then counted as degenerate, and every collapse and every flip was
 * refused while nothing reported it.
 *
 * Scaling by a power of two is exact in floating point -- every sum, product,
 * quotient and square root of scaled operands is the scaled result with the same
 * rounding -- so a remesher with no length scale of its own must produce the
 * SAME mesh times 2^-20, bit for bit, in the same pool order. Any absolute
 * threshold anywhere in the pass breaks that, which is why the comparison is
 * exact and not approximate. Non-vacuity: the pass must split, collapse AND flip.
 */
static int test_remeshing_is_independent_of_the_length_unit(void)
{
  const double s = ldexp(1.0, -20);

  /* The unit mesh: a scrambled sphere, so that all three operators have work. */
  MFTL_Object *unit = make_sphere(3);
  unsigned seed = 24681357u;
  for (int v = 0; v < unit->verts.n; ++v)
  {
    double *q = MFTLVertexPosMut(unit, v);
    for (int k = 0; k < 3; ++k)
    {
      seed = seed * 1103515245u + 12345u;
      q[k] += 0.22 * ((double) ((seed >> 16) & 0x7fff) / 32767.0 - 0.5);
    }
    const double r = sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    for (int k = 0; k < 3; ++k) q[k] /= r;
  }
  MFTLComputeTriangleGeometry(unit, unit->tris.n);

  /* The same mesh in a box 2^20 times smaller: same generator, so the same
   * topology and pool order, then every position scaled exactly. */
  int cuts[3] = {1, 1, 1};
  double lo[3] = {-4.0 * s, -4.0 * s, -4.0 * s};
  double hi[3] = {4.0 * s, 4.0 * s, 4.0 * s};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *small_ens = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, periodic, 1);
  TEST_ASSERT(small_ens != MFTL_NULL);

  MFTL_Object *small = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(small_ens, small);
  double centre[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(small, centre, s, 3);
  TEST_ASSERT_EQUAL_INT(unit->verts.n, small->verts.n);
  for (int i = 0; i < 3 * unit->verts.n; ++i) small->verts.pos[i] = unit->verts.pos[i] * s;
  MFTLComputeTriangleGeometry(small, small->tris.n);

  const double target = MFTLMeanEdgeLength(unit);
  MFTL_RemeshParameters pu;
  MFTL_RemeshParameters ps;
  MFTLInitRemeshParams(&pu, MFTL_REMESH_ISOTROPIC, target, 15, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  MFTLInitRemeshParams(&ps, MFTL_REMESH_ISOTROPIC, target * s, 15, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  pu.smooth_iterations = ps.smooth_iterations = 3;

  MFTL_RemeshStats su;
  MFTL_RemeshStats ss;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(unit, &pu, &su));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRemeshObject(small, &ps, &ss));

  printf("\n    unit: %d split / %d collapse / %d flip -> %d tris;  x2^-20: %d / %d / %d -> %d tris ", su.n_split, su.n_collapse, su.n_flip, unit->tris.n,
         ss.n_split, ss.n_collapse, ss.n_flip, small->tris.n);

  TEST_ASSERT(su.n_split > 0 && su.n_collapse > 0 && su.n_flip > 0);
  TEST_ASSERT_EQUAL_INT(su.n_split, ss.n_split);
  TEST_ASSERT_EQUAL_INT(su.n_collapse, ss.n_collapse);
  TEST_ASSERT_EQUAL_INT(su.n_flip, ss.n_flip);
  TEST_ASSERT_EQUAL_INT(unit->verts.n, small->verts.n);
  TEST_ASSERT_EQUAL_INT(unit->tris.n, small->tris.n);

  int same = 1;
  for (int i = 0; i < 3 * unit->verts.n; ++i) same &= (small->verts.pos[i] == unit->verts.pos[i] * s);
  for (int t = 0; t < unit->tris.n; ++t)
    for (int k = 0; k < 3; ++k) same &= (small->tris.verts[3 * t + k] == unit->tris.verts[3 * t + k]);
  TEST_ASSERT(same);

  TEST_ASSERT(inv_mesh_is_sound(small, 0));
  TEST_ASSERT(inv_mesh_is_closed(small));

  MFTLEnsembleDestroy(&small_ens);
  fixture_free(unit);
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

  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 64);
  if (!g_ens) MPI_Abort(MPI_COMM_WORLD, 1);

  printf("\n=== 1. Operators ===\n");
  RUN_TEST(test_split_appends_one_vertex_and_two_triangles);
  RUN_TEST(test_split_places_the_vertex_at_the_midpoint);
  RUN_TEST(test_split_of_a_boundary_edge_makes_two_boundary_edges);
  RUN_TEST(test_split_preserves_the_feature_flag_on_both_halves);
  RUN_TEST(test_collapse_removes_one_vertex_and_two_triangles);
  RUN_TEST(test_flip_preserves_counts_and_swaps_the_diagonal);
  RUN_TEST(test_flip_is_an_involution);

  printf("\n=== 2. Predicates ===\n");
  RUN_TEST(test_collapse_refuses_to_dissolve_a_tetrahedron);
  RUN_TEST(test_collapse_refuses_to_pull_a_boundary_vertex_inward);
  RUN_TEST(test_collapse_refuses_a_corner);
  RUN_TEST(test_collapse_refuses_a_fold);
  RUN_TEST(test_split_only_offers_each_edge_once);

  printf("\n=== 3. Features ===\n");
  RUN_TEST(test_a_smooth_sphere_has_no_features);
  RUN_TEST(test_a_cube_has_twelve_feature_edges_and_eight_corners);
  RUN_TEST(test_a_threshold_of_180_disables_detection);
  RUN_TEST(test_a_bipyramid_equator_is_a_ridge);
  RUN_TEST(test_a_patch_rim_is_a_feature_at_any_threshold);

  printf("\n=== 4. Smoothing ===\n");
  RUN_TEST(test_corners_do_not_move);
  RUN_TEST(test_a_regular_ridge_is_a_fixed_point);
  RUN_TEST(test_ridge_smoothing_redistributes_along_the_curve);
  RUN_TEST(test_tangential_smoothing_does_not_shrink_a_sphere);
  RUN_TEST(test_area_weighting_is_actually_applied);

  printf("\n=== 5. Full pass ===\n");
  RUN_TEST(test_remesh_refines_towards_the_target);
  RUN_TEST(test_remesh_coarsens_towards_the_target);
  RUN_TEST(test_remesh_improves_quality_and_inverts_nothing);
  RUN_TEST(test_remesh_leaves_the_object_ready_to_stream);
  RUN_TEST(test_remesh_converges_and_then_does_nothing);
  RUN_TEST(test_remesh_preserves_a_cube);
  RUN_TEST(test_remesh_of_an_open_patch_keeps_its_rim);
  RUN_TEST(test_remeshing_is_independent_of_the_length_unit);
  RUN_TEST(test_compaction_removes_orphans);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();

  return (g_tests_failed > 0) ? 1 : 0;
}
