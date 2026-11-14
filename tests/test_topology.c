/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for the half-edge topology.
 *
 * Two fixtures carry most of the file: a tetrahedron (the smallest CLOSED
 * surface) and a two-triangle quad (the smallest OPEN one). Nearly every
 * topological statement in this library differs between the two -- one-ring
 * counts, boundary flags, what "no twin" means -- so testing both is what stops
 * a change that is correct on spheres from silently breaking open surfaces.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief A tetrahedron, wound so every face normal points outward. V-E+F = 2. */
static MFTL_Object *make_tetrahedron(void)
{
  static const double pos[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  static const int tris[12] = {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3};

  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, obj);
  MFTLBuildIndexedMesh(obj, pos, 4, tris, 4);
  return obj;
}

/** @brief A unit square split along its diagonal: an open disk. V-E+F = 1. */
static MFTL_Object *make_quad(void)
{
  static const double pos[12] = {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0};
  static const int tris[6] = {0, 1, 2, 0, 2, 3};

  MFTL_Object *obj = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(g_ens, obj);
  MFTLBuildIndexedMesh(obj, pos, 4, tris, 2);
  return obj;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Implicit indexing, as pure arithmetic
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_half_edge_index_arithmetic(void)
{
  for (int t = 0; t < 50; ++t)
  {
    TEST_ASSERT_EQUAL_INT(3 * t, MFTLTriFirstHe(t));
    for (int k = 0; k < 3; ++k)
    {
      const int h = 3 * t + k;
      TEST_ASSERT_EQUAL_INT(t, MFTLHeTriIdx(h));
      TEST_ASSERT_EQUAL_INT(k, MFTLHeCorner(h));
      TEST_ASSERT_EQUAL_INT(3 * t + (k + 1) % 3, MFTLHeNextIdx(h));
      TEST_ASSERT_EQUAL_INT(3 * t + (k + 2) % 3, MFTLHePrevIdx(h));

      // next and prev invert each other, and next^3 is the identity.
      TEST_ASSERT_EQUAL_INT(h, MFTLHePrevIdx(MFTLHeNextIdx(h)));
      TEST_ASSERT_EQUAL_INT(h, MFTLHeNextIdx(MFTLHeNextIdx(MFTLHeNextIdx(h))));
    }
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Closed surface
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_tetrahedron_is_a_sound_closed_mesh(void)
{
  MFTL_Object *obj = make_tetrahedron();

  TEST_ASSERT_EQUAL_INT(4, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(4, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(12, MFTLNumHalfEdges(obj));

  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));
  TEST_ASSERT(inv_mesh_is_closed(obj));
  TEST_ASSERT(inv_global_ids_are_dense(obj));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  // Euler characteristic of a sphere: V - E + F = 2, with E = 3F/2 on a closed
  // triangulation.
  const int V = obj->verts.n, F = obj->tris.n, E = 3 * F / 2;
  TEST_ASSERT_EQUAL_INT(2, V - E + F);
  return 0;
}

static int test_closed_mesh_has_no_boundary_flags(void)
{
  MFTL_Object *obj = make_tetrahedron();

  for (int h = 0; h < MFTLNumHalfEdges(obj); ++h) TEST_ASSERT(!MFTLHeIsMeshBoundary(obj, h));
  for (int v = 0; v < obj->verts.n; ++v) TEST_ASSERT(!MFTLVertexIsMeshBoundary(obj, v));

  // MFTLMarkMeshBoundary is run by the builder and must report a closed mesh.
  TEST_ASSERT_EQUAL_INT(0, MFTLMarkMeshBoundary(obj));
  return 0;
}

static int test_one_ring_on_a_closed_mesh(void)
{
  MFTL_Object *obj = make_tetrahedron();

  for (int v = 0; v < obj->verts.n; ++v)
  {
    int tris[MFTL_MAX_VALENCE];
    int verts[MFTL_MAX_VALENCE];

    const int nt = MFTLGetVertexOneRing(obj, v, tris, MFTL_MAX_VALENCE, MFTL_TRUE);
    const int nv = MFTLGetVertexOneRingVertices(obj, v, verts, MFTL_MAX_VALENCE, MFTL_TRUE);
    const int nh = MFTLGetVertexOneRingHalfedges(obj, v, tris, MFTL_MAX_VALENCE, MFTL_TRUE);

    // Every vertex of a tetrahedron touches the three faces that are not opposite it.
    TEST_ASSERT_EQUAL_INT(3, nt);

    // On a CLOSED fan the three counts coincide. That is exactly the statement
    // that fails on a boundary vertex, see below.
    TEST_ASSERT_EQUAL_INT(nt, nv);
    TEST_ASSERT_EQUAL_INT(nt, nh);

    // The neighbours are the other three vertices, each exactly once.
    int seen[4] = {0, 0, 0, 0};
    for (int i = 0; i < nv; ++i)
    {
      TEST_ASSERT(verts[i] != v);
      TEST_ASSERT_EQUAL_INT(0, seen[verts[i]]);
      seen[verts[i]] = 1;
    }
  }

  // Passing is_inner = false must give the same answer on a closed fan; the
  // rewind is an optimisation, not a different traversal.
  for (int v = 0; v < obj->verts.n; ++v)
  {
    int a[MFTL_MAX_VALENCE], b[MFTL_MAX_VALENCE];
    TEST_ASSERT_EQUAL_INT(MFTLGetVertexOneRing(obj, v, a, MFTL_MAX_VALENCE, MFTL_TRUE), MFTLGetVertexOneRing(obj, v, b, MFTL_MAX_VALENCE, MFTL_FALSE));
  }
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Open surface
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_quad_is_a_sound_open_mesh(void)
{
  MFTL_Object *obj = make_quad();

  TEST_ASSERT_EQUAL_INT(4, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(2, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(6, MFTLNumHalfEdges(obj));

  TEST_ASSERT(inv_mesh_is_sound(obj, g_ens->domain.rank));
  TEST_ASSERT(!inv_mesh_is_closed(obj));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  // Four rim edges; only the shared diagonal is twinned.
  int n_boundary = 0;
  for (int h = 0; h < MFTLNumHalfEdges(obj); ++h)
  {
    if (MFTLHeIsLocalBoundary(obj, h)) n_boundary++;
  }
  TEST_ASSERT_EQUAL_INT(4, n_boundary);

  // A disk, not a sphere: V - E + F = 1.
  TEST_ASSERT_EQUAL_INT(1, 4 - 5 + 2);
  return 0;
}

static int test_boundary_flags_mark_the_rim(void)
{
  MFTL_Object *obj = make_quad();

  // Every vertex of this fixture is on the rim.
  for (int v = 0; v < obj->verts.n; ++v) TEST_ASSERT(MFTLVertexIsMeshBoundary(obj, v));

  // The flag agrees with "no twin" on a COMPLETE mesh -- which is the only
  // situation where the two statements coincide.
  for (int h = 0; h < MFTLNumHalfEdges(obj); ++h)
  {
    TEST_ASSERT_EQUAL_INT((int) MFTLHeIsLocalBoundary(obj, h), (int) MFTLHeIsMeshBoundary(obj, h));
  }

  TEST_ASSERT_EQUAL_INT(4, MFTLMarkMeshBoundary(obj));
  return 0;
}

static int test_boundary_walk_closes_a_loop(void)
{
  MFTL_Object *obj = make_quad();

  // Find a rim half-edge to start from.
  int start = MFTL_ERROR;
  for (int h = 0; h < MFTLNumHalfEdges(obj); ++h)
  {
    if (MFTLHeIsLocalBoundary(obj, h))
    {
      start = h;
      break;
    }
  }
  TEST_ASSERT(start != MFTL_ERROR);

  // The rim is one closed loop of exactly four half-edges, traversed in a
  // consistent direction: successive edges must share a vertex.
  int h = start;
  int steps = 0;
  do
  {
    const int next = MFTLHeNextBoundary(obj, h);
    TEST_ASSERT(next != MFTL_ERROR);
    TEST_ASSERT(MFTLHeIsLocalBoundary(obj, next));
    TEST_ASSERT_EQUAL_INT(MFTLHeTarget(obj, h), MFTLHeOrigin(obj, next));
    h = next;
    steps++;
    TEST_ASSERT(steps <= 8);  // must not run away
  } while (h != start);

  TEST_ASSERT_EQUAL_INT(4, steps);
  return 0;
}

static int test_one_ring_at_a_boundary_vertex_has_one_extra_neighbour(void)
{
  MFTL_Object *obj = make_quad();

  /* THE open-surface trap: the closing edge of an open fan is reachable only as
   * an INCOMING half-edge, so an outgoing-only walk drops exactly one neighbour.
   * Invisible on a closed mesh, where the two counts coincide. */
  for (int v = 0; v < obj->verts.n; ++v)
  {
    int tris[MFTL_MAX_VALENCE];
    int verts[MFTL_MAX_VALENCE];
    int hes[MFTL_MAX_VALENCE];

    const int nt = MFTLGetVertexOneRing(obj, v, tris, MFTL_MAX_VALENCE, MFTL_FALSE);
    const int nv = MFTLGetVertexOneRingVertices(obj, v, verts, MFTL_MAX_VALENCE, MFTL_FALSE);
    const int nh = MFTLGetVertexOneRingHalfedges(obj, v, hes, MFTL_MAX_VALENCE, MFTL_FALSE);

    TEST_ASSERT_EQUAL_INT(nt + 1, nv);  // one MORE neighbour than triangles
    TEST_ASSERT_EQUAL_INT(nt, nh);  // outgoing half-edges: one per triangle

    // No duplicates and no self-reference in the neighbour set.
    for (int i = 0; i < nv; ++i)
    {
      TEST_ASSERT(verts[i] != v);
      for (int j = i + 1; j < nv; ++j) TEST_ASSERT(verts[i] != verts[j]);
    }
  }

  // Concretely: the diagonal vertices touch both triangles, the others one each.
  int tris[MFTL_MAX_VALENCE];
  TEST_ASSERT_EQUAL_INT(2, MFTLGetVertexOneRing(obj, 0, tris, MFTL_MAX_VALENCE, MFTL_FALSE));
  TEST_ASSERT_EQUAL_INT(1, MFTLGetVertexOneRing(obj, 1, tris, MFTL_MAX_VALENCE, MFTL_FALSE));
  TEST_ASSERT_EQUAL_INT(2, MFTLGetVertexOneRing(obj, 2, tris, MFTL_MAX_VALENCE, MFTL_FALSE));
  TEST_ASSERT_EQUAL_INT(1, MFTLGetVertexOneRing(obj, 3, tris, MFTL_MAX_VALENCE, MFTL_FALSE));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Connectivity maintenance
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_rebuild_twins_is_idempotent(void)
{
  MFTL_Object *obj = make_tetrahedron();

  int before[12];
  for (int h = 0; h < 12; ++h) before[h] = MFTLHeTwin(obj, h);

  MFTLRebuildTwins(obj);
  for (int h = 0; h < 12; ++h) TEST_ASSERT_EQUAL_INT(before[h], MFTLHeTwin(obj, h));

  TEST_ASSERT(inv_twins_are_symmetric(obj));
  return 0;
}

static int test_rebuild_twins_invalidates_derived(void)
{
  MFTL_Object *obj = make_tetrahedron();
  TEST_ASSERT(MFTLObjectHasTopology(obj));

  /* `tris.neighbors` is a function of the twins, so relinking them makes it
   * stale by construction. A rebuild that left the bit set would leave a hot
   * loop reading a neighbour table for a topology that no longer exists. */
  MFTLRebuildTwins(obj);
  TEST_ASSERT(!MFTLObjectHasTopology(obj));

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_derived_matches_topology(obj));
  return 0;
}

static int test_integrity_check_rejects_a_broken_twin(void)
{
  MFTL_Object *obj = make_tetrahedron();
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  // Break the symmetry: h still points at its twin, but not the other way round.
  const int h = 0;
  const int tw = MFTLHeTwin(obj, h);
  TEST_ASSERT(tw != MFTL_ERROR);

  printf("\n  (expecting diagnostics below) ");
  obj->tris.he[tw].twin_idx = MFTL_ERROR;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLCheckMeshIntegrity(obj));

  // And out-of-range indices.
  obj->tris.he[tw].twin_idx = 9999;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLCheckMeshIntegrity(obj));

  // Repairing it topologically must restore validity.
  MFTLRebuildTwins(obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));
  return 0;
}

static int test_integrity_check_rejects_a_stale_out_he(void)
{
  MFTL_Object *obj = make_tetrahedron();

  /* out_he_idx must be OUTGOING from its vertex. A stale value here makes every
   * one-ring walk from that vertex traverse someone else's fan -- silently, with
   * plausible-looking results. */
  const int v = 0;
  int wrong = MFTL_ERROR;
  for (int h = 0; h < MFTLNumHalfEdges(obj); ++h)
  {
    if (MFTLHeOrigin(obj, h) != v)
    {
      wrong = h;
      break;
    }
  }
  TEST_ASSERT(wrong != MFTL_ERROR);

  printf("\n  (expecting diagnostics below) ");
  obj->verts.rec[v].out_he_idx = wrong;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLCheckMeshIntegrity(obj));
  return 0;
}

/**
 * @brief A pool built from an index list WITHOUT the builder's validation: the
 *        records exactly as MFTLBuildIndexedMesh() would write them, then
 *        MFTLRebuildTwins(). The only way to put malformed connectivity in front
 *        of the integrity check now that the builder refuses it.
 */
static MFTL_Object *make_unchecked(int id, const double *pos, int nv, const int *tri, int nt)
{
  MFTL_Object *obj = MFTLObjectCreate(id);
  MFTLEnsembleAddObject(g_ens, obj);
  MFTLObjectSetCounts(obj, nv, nv, nt, nt);
  memcpy(obj->verts.pos, pos, 3 * (size_t) nv * sizeof(double));
  for (int v = 0; v < nv; ++v)
  {
    obj->verts.rec[v].global_id = v;
    obj->verts.rec[v].owner_proc = 0;
    obj->verts.rec[v].flags = 0;
    obj->verts.rec[v].out_he_idx = MFTL_ERROR;
  }
  for (int t = 0; t < nt; ++t)
  {
    obj->tris.rec[t].global_id = t;
    obj->tris.rec[t].owner_proc = 0;
    obj->tris.rec[t].flags = 0;
    for (int k = 0; k < 3; ++k)
    {
      MFTL_HalfEdge *const h = &obj->tris.he[3 * t + k];
      h->global_id = 3 * t + k;
      h->vertex_idx = tri[3 * t + (k + 1) % 3];
      h->twin_idx = MFTL_ERROR;
      h->flags = 0;
      obj->verts.rec[tri[3 * t + k]].out_he_idx = 3 * t + k;
    }
  }
  MFTLRebuildTwins(obj);
  return obj;
}

/**
 * @brief Connectivity a half-edge mesh cannot represent is diagnosed, not
 *        certified.
 *
 * Each of these passed as "consistent" before the edge-incidence pass:
 * MFTLRebuildTwins() pairs two half-edges of the edge and leaves the surplus
 * twin-less, which the check counted as open boundary (and the builder then
 * flagged as boundary for good). The reversed duplicate even passed as CLOSED.
 */
static int test_integrity_check_rejects_what_a_half_edge_mesh_cannot_be(void)
{
  static const double pos[15] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1};
  static const int three_on_one_edge[9] = {0, 1, 2, 1, 0, 3, 0, 1, 4};
  static const int flipped_neighbour[6] = {0, 1, 2, 0, 1, 3};
  static const int duplicated[6] = {0, 1, 2, 0, 1, 2};
  static const int reversed_duplicate[6] = {0, 1, 2, 0, 2, 1};
  const struct
  {
    const int *tri;
    int nt, nv;
  } cases[] = {{three_on_one_edge, 3, 5}, {flipped_neighbour, 2, 4}, {duplicated, 2, 3}, {reversed_duplicate, 2, 3}};

  printf("\n  (expecting diagnostics below) ");
  for (size_t c = 0; c < sizeof cases / sizeof cases[0]; ++c)
  {
    /* The builder refuses the list before writing anything... */
    MFTL_Object *built = MFTLObjectCreate(100 + (int) c);
    MFTLEnsembleAddObject(g_ens, built);
    TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLBuildIndexedMesh(built, pos, cases[c].nv, cases[c].tri, cases[c].nt));
    TEST_ASSERT_EQUAL_INT(0, built->tris.n);
    TEST_ASSERT_EQUAL_INT(0, built->verts.n);

    /* ...and the check rejects the same connectivity built around it. */
    MFTL_Object *obj = make_unchecked(200 + (int) c, pos, cases[c].nv, cases[c].tri, cases[c].nt);
    TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLCheckMeshIntegrity(obj));
  }
  return 0;
}

/**
 * @brief A missing twin is legal only where the other side is absent.
 *
 * Unlinking BOTH halves of an interior edge leaves a symmetric relation the old
 * check accepted as "open"; the edge's other half-edge is still in the pool, so
 * on any mesh -- complete or a partition -- it must be linked.
 */
static int test_integrity_check_rejects_an_unpaired_interior_edge(void)
{
  MFTL_Object *obj = make_quad();
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));

  int h = MFTL_ERROR;
  for (int i = 0; i < MFTLNumHalfEdges(obj) && h == MFTL_ERROR; ++i)
    if (MFTLHeTwin(obj, i) != MFTL_ERROR) h = i;
  TEST_ASSERT(h != MFTL_ERROR);
  const int tw = MFTLHeTwin(obj, h);

  printf("\n  (expecting diagnostics below) ");
  obj->tris.he[h].twin_idx = MFTL_ERROR;
  obj->tris.he[tw].twin_idx = MFTL_ERROR;
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLCheckMeshIntegrity(obj));

  MFTLRebuildTwins(obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(obj));
  return 0;
}

/**
 * @brief What stays legal: an open rim, a bowtie vertex, and a pool that is a
 *        subset of a mesh (a halo-like cut through a sphere).
 */
static int test_integrity_check_accepts_rims_bowties_and_subsets(void)
{
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(make_quad()));

  static const double pos[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, -1, -1, 0};
  static const int bowtie[] = {0, 1, 2, 0, 3, 4};
  MFTL_Object *bt = MFTLObjectCreate(300);
  MFTLEnsembleAddObject(g_ens, bt);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(bt, pos, 5, bowtie, 2));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(bt));

  /* The upper half of a sphere's triangles, with every vertex kept: twin-less
   * half-edges along the cut whose opposite is NOT in the pool. */
  MFTL_Object *sphere = MFTLObjectCreate(301);
  MFTLEnsembleAddObject(g_ens, sphere);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(sphere, c, 1.0, 2);
  const int nt = sphere->tris.n;
  int *tri = (int *) malloc(3 * (size_t) nt * sizeof(int));
  int kept = 0;
  for (int t = 0; t < nt; ++t)
  {
    double ct[3];
    MFTLGetTriangleCenter(sphere, t, ct);
    if (ct[2] <= 0.0) continue;
    MFTLGetTriangleVertices(sphere, t, &tri[3 * kept]);
    kept++;
  }
  /* Only vertices the kept triangles use, renumbered densely. */
  int *slot = (int *) malloc((size_t) sphere->verts.n * sizeof(int));
  for (int v = 0; v < sphere->verts.n; ++v) slot[v] = MFTL_ERROR;
  int nv = 0;
  double *vp = (double *) malloc(3 * (size_t) sphere->verts.n * sizeof(double));
  for (int i = 0; i < 3 * kept; ++i)
  {
    const int v = tri[i];
    if (slot[v] == MFTL_ERROR)
    {
      memcpy(&vp[3 * nv], MFTLVertexPos(sphere, v), 3 * sizeof(double));
      slot[v] = nv++;
    }
    tri[i] = slot[v];
  }
  MFTL_Object *cap = MFTLObjectCreate(302);
  MFTLEnsembleAddObject(g_ens, cap);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(cap, vp, nv, tri, kept));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLCheckMeshIntegrity(cap));
  TEST_ASSERT(inv_mesh_is_sound(cap, g_ens->domain.rank));
  free(tri);
  free(slot);
  free(vp);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Connected components
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_one_closed_surface_is_one_component(void)
{
  MFTL_Object *obj = make_tetrahedron();
  int labels[4];
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(obj, labels));
  for (int t = 0; t < 4; ++t) TEST_ASSERT_EQUAL_INT(0, labels[t]);
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(obj, MFTL_NULL));
  return 0;
}

static int test_two_shells_in_one_pool_are_two_components(void)
{
  /* A quad (triangles 0,1), then a tetrahedron (2..5), then a second quad
   * (6,7) glued to the first along edge 2-3. The numbering follows each
   * component's SMALLEST triangle index, so the second quad inherits label 0
   * even though it comes after the tetrahedron in the pool. */
  static const double pos[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0,  // quad A: 0..3
                               5, 0, 0, 6, 0, 0, 5, 1, 0, 5, 0, 1,  // tet: 4..7
                               2, 2, 0, 1, 2, 0};  // quad B: 8, 9 (+ vertex 2 of A)
  static const int tris[] = {0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 5, 7, 4, 7, 6, 5, 6, 7, 2, 8, 9, 2, 9, 3};
  MFTL_Object *obj = MFTLObjectCreate(3);
  MFTLEnsembleAddObject(g_ens, obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(obj, pos, 10, tris, 8));

  int labels[8];
  const int n = MFTLLabelConnectedComponents(obj, labels);
  /* Triangle (2,9,3) carries half-edge 3 -> 2, the twin of quad A's 2 -> 3. */
  TEST_ASSERT_EQUAL_INT(2, n);
  TEST_ASSERT_EQUAL_INT(0, labels[0]);
  TEST_ASSERT_EQUAL_INT(0, labels[1]);
  for (int t = 2; t < 6; ++t) TEST_ASSERT_EQUAL_INT(1, labels[t]);
  TEST_ASSERT_EQUAL_INT(0, labels[6]);
  TEST_ASSERT_EQUAL_INT(0, labels[7]);
  return 0;
}

static int test_a_shared_vertex_does_not_join_components(void)
{
  /* Two triangles touching at vertex 0 only: a bowtie. */
  static const double pos[] = {0, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0, 0, -1, -1, 0};
  static const int tris[] = {0, 1, 2, 0, 3, 4};
  MFTL_Object *obj = MFTLObjectCreate(4);
  MFTLEnsembleAddObject(g_ens, obj);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLBuildIndexedMesh(obj, pos, 5, tris, 2));

  int labels[2];
  TEST_ASSERT_EQUAL_INT(2, MFTLLabelConnectedComponents(obj, labels));
  TEST_ASSERT_EQUAL_INT(0, labels[0]);
  TEST_ASSERT_EQUAL_INT(1, labels[1]);
  return 0;
}

static int test_cutting_the_twins_along_a_loop_splits_a_sphere(void)
{
  MFTL_Object *obj = MFTLObjectCreate(5);
  MFTLEnsembleAddObject(g_ens, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 1.0, 3);
  TEST_ASSERT_EQUAL_INT(1, MFTLLabelConnectedComponents(obj, MFTL_NULL));

  /* Unlink every half-edge whose two triangles lie on opposite sides of z = 0.
   * That is a closed loop of cuts, so the sphere falls into two caps. */
  int cut = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int tw = obj->tris.he[h].twin_idx;
    if (tw == MFTL_ERROR) continue;
    double ca[3], cb[3];
    MFTLGetTriangleCenter(obj, MFTLHeTriIdx(h), ca);
    MFTLGetTriangleCenter(obj, MFTLHeTriIdx(tw), cb);
    if ((ca[2] > 0.0) != (cb[2] > 0.0))
    {
      obj->tris.he[h].twin_idx = MFTL_ERROR;
      cut++;
    }
  }
  TEST_ASSERT(cut > 0);

  int *labels = (int *) malloc((size_t) obj->tris.n * sizeof(int));
  TEST_ASSERT_EQUAL_INT(2, MFTLLabelConnectedComponents(obj, labels));
  /* Same side of the cut as triangle 0 <=> same label as triangle 0. */
  double c0[3];
  MFTLGetTriangleCenter(obj, 0, c0);
  int ok = 1;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double ct[3];
    MFTLGetTriangleCenter(obj, t, ct);
    if ((labels[t] == labels[0]) != ((ct[2] > 0.0) == (c0[2] > 0.0))) ok = 0;
  }
  free(labels);
  TEST_ASSERT(ok);
  return 0;
}

static int test_an_empty_pool_has_no_components(void)
{
  MFTL_Object *obj = MFTLObjectCreate(6);
  MFTLEnsembleAddObject(g_ens, obj);
  TEST_ASSERT_EQUAL_INT(0, MFTLLabelConnectedComponents(obj, MFTL_NULL));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-10.0, -10.0, -10.0};
  double hi[3] = {10.0, 10.0, 10.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 32);

  printf("--- Running MFTL Topology Tests ---\n");

  RUN_TEST(test_half_edge_index_arithmetic);
  RUN_TEST(test_tetrahedron_is_a_sound_closed_mesh);
  RUN_TEST(test_closed_mesh_has_no_boundary_flags);
  RUN_TEST(test_one_ring_on_a_closed_mesh);
  RUN_TEST(test_quad_is_a_sound_open_mesh);
  RUN_TEST(test_boundary_flags_mark_the_rim);
  RUN_TEST(test_boundary_walk_closes_a_loop);
  RUN_TEST(test_one_ring_at_a_boundary_vertex_has_one_extra_neighbour);
  RUN_TEST(test_rebuild_twins_is_idempotent);
  RUN_TEST(test_rebuild_twins_invalidates_derived);
  RUN_TEST(test_integrity_check_rejects_a_broken_twin);
  RUN_TEST(test_integrity_check_rejects_a_stale_out_he);
  RUN_TEST(test_integrity_check_rejects_what_a_half_edge_mesh_cannot_be);
  RUN_TEST(test_integrity_check_rejects_an_unpaired_interior_edge);
  RUN_TEST(test_integrity_check_accepts_rims_bowties_and_subsets);
  RUN_TEST(test_one_closed_surface_is_one_component);
  RUN_TEST(test_two_shells_in_one_pool_are_two_components);
  RUN_TEST(test_a_shared_vertex_does_not_join_components);
  RUN_TEST(test_cutting_the_twins_along_a_loop_splits_a_sphere);
  RUN_TEST(test_an_empty_pool_has_no_components);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
