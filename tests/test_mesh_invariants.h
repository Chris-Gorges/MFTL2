/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief The structural invariants an MFTL mesh must satisfy, as reusable checks.
 *
 * Every one of these is a property some module is allowed to ASSUME. Keeping
 * them here rather than re-deriving them per test file is what makes it cheap to
 * assert the whole set after any operation -- generation, reordering,
 * subdivision -- which is how a regression in one of them gets caught by the
 * test for something else.
 *
 * Each returns 1 on success and 0 on failure, so they compose with TEST_ASSERT.
 */

#ifndef MFTL_TEST_MESH_INVARIANTS_H
#define MFTL_TEST_MESH_INVARIANTS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"

/* ─────────────────────────────────────────────────────────────────────────────
 * 1. Implicit half-edge indexing
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Half-edge 3t+k belongs to triangle t and next(3t+k) == 3t+(k+1)%3.
 *
 * The invariant that lets `he_idx`, `next_idx` and `triangle_idx` not exist. It
 * is arithmetic, so this cannot fail unless the arithmetic itself is wrong --
 * which is exactly why it is worth pinning once: a "harmless" edit to
 * MFTLHeNextIdx would otherwise corrupt every traversal in the library.
 */
static int inv_implicit_indexing(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;

  for (int h = 0; h < n_he; ++h)
  {
    if (MFTLHeTriIdx(h) != h / 3) return 0;

    const int n1 = MFTLHeNextIdx(h);
    const int n2 = MFTLHeNextIdx(n1);
    const int n3 = MFTLHeNextIdx(n2);

    if (n3 != h) return 0;  // the cycle closes after exactly three steps
    if (n1 == h || n2 == h) return 0;  // ...and not before
    if (MFTLHeTriIdx(n1) != MFTLHeTriIdx(h)) return 0;  // all three in the same triangle
    if (MFTLHePrevIdx(n1) != h) return 0;  // prev inverts next

    /* The topological consequence: consecutive half-edges share a vertex. */
    if (MFTLHeOrigin(obj, n1) != MFTLHeTarget(obj, h)) return 0;
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 2. Twins
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief twin(twin(h)) == h, and the two agree about their endpoints. */
static int inv_twins_are_symmetric(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;

  for (int h = 0; h < n_he; ++h)
  {
    const int tw = MFTLHeTwin(obj, h);
    if (tw == MFTL_ERROR) continue;
    if (tw < 0 || tw >= n_he) return 0;
    if (MFTLHeTwin(obj, tw) != h) return 0;
    if (MFTLHeOrigin(obj, h) != MFTLHeTarget(obj, tw)) return 0;
    if (MFTLHeTarget(obj, h) != MFTLHeOrigin(obj, tw)) return 0;
  }
  return 1;
}

/** @brief Every half-edge has a twin -- true exactly for a closed mesh. */
static int inv_mesh_is_closed(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h)
  {
    if (MFTLHeTwin(obj, h) == MFTL_ERROR) return 0;
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 3. Vertices
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Every vertex's out_he_idx exists and actually STARTS at that vertex. */
static int inv_vertex_out_he_is_outgoing(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;

  for (int v = 0; v < obj->verts.n; ++v)
  {
    const int oh = obj->verts.rec[v].out_he_idx;
    if (oh == MFTL_ERROR || oh < 0 || oh >= n_he) return 0;
    if (MFTLHeOrigin(obj, oh) != v) return 0;
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 4. Identity
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Global IDs are a dense permutation of [0, n) for each entity class, and
 *        each next_*_gid sits exactly one past the last ID handed out.
 *
 * Density is not cosmetic: it is what lets a receiver size a lookup table by the
 * ID range instead of hashing, which is a large part of why this library needs
 * no hash maps.
 */
static int inv_global_ids_are_dense(const MFTL_Object *obj)
{
  const int nv = obj->verts.n;
  const int nt = obj->tris.n;
  const int nh = 3 * nt;

  if (obj->next_vertex_gid != nv) return 0;
  if (obj->next_triangle_gid != nt) return 0;
  if (obj->next_halfedge_gid != nh) return 0;

  int ok = 1;
  char *seen = (char *) calloc((size_t) (nv > nh ? nv : nh) + 1, 1);
  if (!seen) return 0;

  for (int v = 0; v < nv; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g < 0 || g >= nv || seen[g]) { ok = 0; break; }
    seen[g] = 1;
  }

  if (ok)
  {
    memset(seen, 0, (size_t) (nv > nh ? nv : nh) + 1);
    for (int t = 0; t < nt; ++t)
    {
      const int g = obj->tris.rec[t].global_id;
      if (g < 0 || g >= nt || seen[g]) { ok = 0; break; }
      seen[g] = 1;
    }
  }

  if (ok)
  {
    memset(seen, 0, (size_t) (nv > nh ? nv : nh) + 1);
    for (int h = 0; h < nh; ++h)
    {
      const int g = obj->tris.he[h].global_id;
      if (g < 0 || g >= nh || seen[g]) { ok = 0; break; }
      seen[g] = 1;
    }
  }

  free(seen);
  return ok;
}

/**
 * @brief Half-edge 3t+k carries the global id 3 * (triangle global id) + k.
 *
 * Established by MFTLBuildTopologyFromIndexList, preserved by
 * MFTLObjectReorderMorton (which moves half-edge triples as a unit) and by
 * redistribution. It is the wire-format half of implicit indexing: because it
 * holds mesh-wide, half-edge global ids are never transmitted -- the receiver
 * reconstructs them (see mftl_communication.h). Breaking it would silently give
 * redistributed half-edges wrong identities.
 */
static int inv_halfedge_gids_are_implicit(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    for (int k = 0; k < 3; ++k)
    {
      if (obj->tris.he[3 * t + k].global_id != 3 * g + k) return 0;
    }
  }
  return 1;
}

/**
 * @brief No global id appears twice in the LOCAL pool, per entity class.
 *
 * The distributed counterpart of inv_global_ids_are_dense, which only holds while
 * a mesh is complete on one rank. After a redistribution a rank holds an
 * arbitrary subset of each id range, but a DUPLICATE would mean the same entity
 * was written into two slots -- exactly what the arrival de-duplication exists to
 * prevent, and invisible to every other check.
 */
static int inv_local_global_ids_are_unique(const MFTL_Object *obj)
{
  const int nv = obj->verts.n;
  const int nt = obj->tris.n;
  const int n = (nv > nt) ? nv : nt;
  if (n == 0) return 1;

  int ok = 1;
  int *tmp = (int *) malloc((size_t) n * sizeof(int));
  if (!tmp) return 0;

  for (int v = 0; v < nv; ++v) tmp[v] = obj->verts.rec[v].global_id;
  MFTLSortIds(tmp, nv);
  for (int v = 1; v < nv; ++v)
    if (tmp[v] == tmp[v - 1]) ok = 0;

  if (ok)
  {
    for (int t = 0; t < nt; ++t) tmp[t] = obj->tris.rec[t].global_id;
    MFTLSortIds(tmp, nt);
    for (int t = 1; t < nt; ++t)
      if (tmp[t] == tmp[t - 1]) ok = 0;
  }

  free(tmp);
  return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 4b. Halo completeness -- what redistribution promises its consumers
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Every triangle incident to an OWNED vertex is present locally.
 *
 * The precondition MFTLComputeVertexNormals needs to be communication-free
 * (mftl_geometry.h), and the strongest of redistribution's halo guarantees.
 *
 * Stated as: a half-edge without a twin is either a genuine mesh boundary (the
 * flag says so, and it travels with the entity) or a partition rim -- and a
 * partition rim may not touch an owned vertex, because the triangle on the other
 * side of it is incident to that vertex and would therefore have been sent here.
 */
static int inv_halo_is_vertex_complete(const MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h)
  {
    if (MFTLHeTwin(obj, h) != MFTL_ERROR) continue;
    if (MFTLHeIsMeshBoundary(obj, h)) continue;

    if (!MFTLVertexIsGhost(obj, MFTLHeOrigin(obj, h))) return 0;
    if (!MFTLVertexIsGhost(obj, MFTLHeTarget(obj, h))) return 0;
  }
  return 1;
}

/**
 * @brief Every OWNED triangle has all three edge neighbours present locally.
 *
 * Not implied by vertex completeness: a triangle whose centre falls in this
 * rank's cell can have all three of its vertices owned elsewhere. The separate
 * "send to the owners of my edge neighbours" clause of the destination rule is
 * what covers that case, and this is the check that it is still there.
 */
static int inv_owned_triangles_have_all_neighbours(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    for (int k = 0; k < 3; ++k)
    {
      if (obj->tris.neighbors[3 * t + k] != MFTL_ERROR) continue;
      if (MFTLHeIsMeshBoundary(obj, 3 * t + k)) continue;
      return 0;
    }
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 5. Derived arrays agree with the authoritative topology
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief `tris.verts` and `tris.neighbors` say exactly what a half-edge walk says.
 *
 * This is the check that keeps the redundancy honest. Both arrays exist purely
 * to save memory traffic, so the moment they disagree with the records they are
 * not an optimisation, they are a second, wrong mesh.
 */
static int inv_derived_matches_topology(const MFTL_Object *obj)
{
  if (!MFTLObjectHasTopology(obj)) return 0;

  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    int nb[3];
    MFTLGetTriangleVertices(obj, t, v);
    MFTLGetTriangleNeighbors(obj, t, nb);

    for (int k = 0; k < 3; ++k)
    {
      if (obj->tris.verts[3 * t + k] != v[k]) return 0;
      if (obj->tris.neighbors[3 * t + k] != nb[k]) return 0;

      /* Slot k must be the ORIGIN of half-edge 3t+k. Getting this rotation
       * wrong is silent: the triangles are still valid, just rotated. */
      if (obj->tris.verts[3 * t + k] != MFTLHeOrigin(obj, 3 * t + k)) return 0;
    }
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 5b. Normals are UNIT vectors
 *
 * Deliberately NOT part of inv_mesh_is_sound(). A normal is cached geometry, and
 * redistribution is documented as leaving `tris.normal` holding unspecified
 * values (include/mftl_communication.h) -- so "the normals are unit" is a claim
 * about a mesh whose geometry pass has RUN, not about every valid mesh. Call
 * these wherever that is true.
 *
 * Why they are worth asserting at all: a normal that is short rather than unit is
 * the most dangerous shape of wrong a physics kernel can be handed. It points the
 * right way, so every visual check and every sign test passes, and it silently
 * scales whatever it multiplies -- surface tension, the indicator function
 * gradient, a contact angle projection. The failure is a wrong NUMBER, not a
 * wrong direction, and nothing downstream is in a position to notice.
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Every triangle normal has length 1 to within @p tol. */
static int inv_triangle_normals_are_unit(const MFTL_Object *obj, double tol)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double *n = &obj->tris.normal[3 * t];
    const double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (fabs(len - 1.0) > tol) return 0;
  }
  return 1;
}

/**
 * @brief Every vertex normal has length 1, or is exactly zero.
 *
 * Zero is the documented sentinel of MFTLComputeVertexNormals for a vertex with
 * no incident triangles -- an isolated vertex, or a ghost whose faces all live on
 * another rank. It is admitted here and nothing between 0 and 1 is, which is the
 * distinction that matters: "not computed" is recoverable, "computed and 0.7 long"
 * is not.
 */
static int inv_vertex_normals_are_unit(const MFTL_Object *obj, double tol)
{
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *n = &obj->verts.normal[3 * v];
    const double len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];

    if (len_sq == 0.0) continue;  // the "not computed" sentinel
    if (fabs(sqrt(len_sq) - 1.0) > tol) return 0;
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 6. Ownership
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Owned entities are an array prefix and `owner_proc` agrees with it.
 *
 * The single canonical notion of ownership in this version. The previous one
 * carried three (a prefix counter, an `is_ghost` flag and `owner_proc == rank`)
 * and recorded "pick one as canonical" as a deferred issue.
 */
static int inv_ownership_is_a_prefix(const MFTL_Object *obj, int rank)
{
  if (obj->verts.n_local > obj->verts.n) return 0;
  if (obj->tris.n_local > obj->tris.n) return 0;

  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    if (obj->verts.rec[v].owner_proc != rank) return 0;
    if (MFTLVertexIsGhost(obj, v)) return 0;
  }
  for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
  {
    if (!MFTLVertexIsGhost(obj, v)) return 0;
  }

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    if (obj->tris.rec[t].owner_proc != rank) return 0;
    if (MFTLTriangleIsGhost(obj, t)) return 0;
  }
  for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
  {
    if (!MFTLTriangleIsGhost(obj, t)) return 0;
  }

  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 7. Composite
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Every structural invariant a locally complete, derived mesh must have. */
static int inv_mesh_is_sound(const MFTL_Object *obj, int rank)
{
  if (!MFTLObjectPoolsAreConsistent(obj)) return 0;
  if (!inv_implicit_indexing(obj)) return 0;
  if (!inv_twins_are_symmetric(obj)) return 0;
  if (!inv_vertex_out_he_is_outgoing(obj)) return 0;
  if (!inv_derived_matches_topology(obj)) return 0;
  if (!inv_ownership_is_a_prefix(obj, rank)) return 0;
  return 1;
}

/**
 * @brief Everything a REDISTRIBUTED partition must satisfy.
 *
 * inv_mesh_is_sound plus the three statements that only become meaningful once a
 * mesh is spread over several ranks: local id uniqueness, and the two halo
 * completeness guarantees the destination rule exists to produce.
 */
static int inv_partition_is_sound(const MFTL_Object *obj, int rank)
{
  if (!inv_mesh_is_sound(obj, rank)) return 0;
  if (!inv_halfedge_gids_are_implicit(obj)) return 0;
  if (!inv_local_global_ids_are_unique(obj)) return 0;
  if (!inv_halo_is_vertex_complete(obj)) return 0;
  if (!inv_owned_triangles_have_all_neighbours(obj)) return 0;
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 8. Multi-rank helper
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Reduces a local verdict to a global one.
 *
 * MANDATORY in any multi-rank test. TEST_ASSERT returns immediately, so an
 * assertion that fails on only some ranks leaves the rest blocked in the next
 * collective -- a hang instead of a clean failure.
 */
static int agreed(int local_ok, MPI_Comm comm)
{
  int global_ok = 0;
  MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, comm);
  return global_ok;
}

/**
 * @brief Reduces a local observation to "this happened SOMEWHERE".
 *
 * The companion to agreed(), and the two are not interchangeable: agreed() is
 * for CLAIMS, which every rank must satisfy, and this is for NON-VACUITY
 * MEASURES, which only need to be true of the run.
 *
 * Using agreed() for a non-vacuity measure is a real trap and it bites exactly
 * where the coverage is most wanted. "Every rank holds at least one ghost" is
 * true on a 2x2x2 grid and false on 3x3x3, where a hollow body leaves the centre
 * cell empty -- so the test that was meant to prove the run was distributed
 * instead fails on the most distributed configuration there is. The measure
 * wanted is "some rank held a ghost", which is this.
 *
 * Still collective, still called by every rank, for the same reason as agreed().
 */
static int anywhere(int local_observation, MPI_Comm comm)
{
  int global = 0;
  MPI_Allreduce(&local_observation, &global, 1, MPI_INT, MPI_MAX, comm);
  return global;
}

#endif  // MFTL_TEST_MESH_INVARIANTS_H
