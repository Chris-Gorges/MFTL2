/*
 * MFTL2 core — half-edge topology: array accessors, navigation (next/prev/twin/origin/target),
 * the twin-state predicates, one-ring traversal, flattened connectivity, builders (triples,
 * distributed soups, staged copies), and the topology invariant API of plans/16 (declared class,
 * rings, validation, ghost completion, boundary loops).
 *
 * Twin states (plans/16 §2.1): every live half-edge of a quiescent mesh is LOCAL (twin stored
 * here), REMOTE (twin on another rank: an MPI rim) or BOUNDARY (a tagged physical boundary).
 * twin_gid is read only through the predicates below. A NULL from MFTLHeTwin means "no stored
 * twin" — either a physical boundary or an MPI rim — and must not be taken as a boundary.
 */

#ifndef MFTL_TOPOLOGY_H
#define MFTL_TOPOLOGY_H

#include "mftl_core.h"
#include "mftl_constants.h"
#include <assert.h>

// -----------------------------------------------------------------------------
// BASE ARRAY ACCESSORS
// -----------------------------------------------------------------------------

/**
 * @brief Get a pointer to a vertex by its local index.
 */
static inline MFTL_Vertex *MFTLGetVertex(const MFTL_Object *obj, int idx) { return &obj->vertices[idx]; }

/**
 * @brief Get a pointer to a half-edge by its local index.
 */
static inline MFTL_HalfEdge *MFTLGetHalfedge(const MFTL_Object *obj, int idx)
{
  if (idx == MFTL_ERROR) return MFTL_NULL;
  return &obj->half_edges[idx];
}

/**
 * @brief Get a pointer to a triangle by its local index.
 */
static inline MFTL_Triangle *MFTLGetTriangle(const MFTL_Object *obj, int idx)
{
  if (idx == MFTL_ERROR) return MFTL_NULL;
  return &obj->triangles[idx];
}

// -----------------------------------------------------------------------------
// VERTEX NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief The stored outgoing half-edge of a vertex (the outgoing BOUNDARY half-edge on a
 * boundary vertex, plans/16 §2.6).
 * @return Pointer to the half-edge, or NULL if the vertex has none (out_he_idx == -1).
 */
static inline MFTL_HalfEdge *MFTLVOutHe(const MFTL_Object *obj, const MFTL_Vertex *v) { return MFTLGetHalfedge(obj, v->out_he_idx); }

// -----------------------------------------------------------------------------
// TRIANGLE NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief The half-edge the triangle stores (he_idx; the "first" half-edge of the triangle).
 */
static inline MFTL_HalfEdge *MFTLTHe(const MFTL_Object *obj, const MFTL_Triangle *t) { return MFTLGetHalfedge(obj, t->he_idx); }

/**
 * @brief The three half-edges of a triangle in winding order, starting at he_idx.
 */
static inline void MFTLGetTriangleHalfedges(const MFTL_Object *obj, const MFTL_Triangle *tri, MFTL_HalfEdge *out_he[3])
{
  out_he[0] = (MFTL_HalfEdge *) &obj->half_edges[tri->he_idx];
  out_he[1] = (MFTL_HalfEdge *) &obj->half_edges[out_he[0]->next_idx];
  out_he[2] = (MFTL_HalfEdge *) &obj->half_edges[out_he[1]->next_idx];
}

/**
 * @brief The three vertices of a triangle in winding order: origin of he0, then the targets
 * of he0 and he1. (v1 - v0) x (v2 - v0) is the triangle normal.
 */
static inline void MFTLGetTriangleVertices(const MFTL_Object *obj, const MFTL_Triangle *tri, MFTL_Vertex *out_verts[3])
{
  MFTL_HalfEdge *out_he[3];
  MFTLGetTriangleHalfedges(obj, tri, out_he);

  // origin of he0 == target of he2
  out_verts[0] = (MFTL_Vertex *) &obj->vertices[out_he[2]->vertex_idx];
  out_verts[1] = (MFTL_Vertex *) &obj->vertices[out_he[0]->vertex_idx];
  out_verts[2] = (MFTL_Vertex *) &obj->vertices[out_he[1]->vertex_idx];
}

// -----------------------------------------------------------------------------
// HALF-EDGE NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief The vertex at the END (target) of a half-edge.
 */
static inline MFTL_Vertex *MFTLHeTarget(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetVertex(obj, he->vertex_idx); }

/**
 * @brief The vertex at the START (origin) of a half-edge: the target of the previous
 * half-edge of the same triangle (next->next), so it needs no twin.
 */
static inline MFTL_Vertex *MFTLHeOrigin(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  const MFTL_HalfEdge *next = &obj->half_edges[he->next_idx];
  const MFTL_HalfEdge *prev = &obj->half_edges[next->next_idx];
  return MFTLGetVertex(obj, prev->vertex_idx);
}

/**
 * @brief The stored twin (LOCAL state).
 * @return Pointer to the twin, or NULL when the twin is not stored on this rank: a physical
 * boundary (BOUNDARY) OR an MPI rim (REMOTE) — use the predicates to tell them apart.
 */
static inline MFTL_HalfEdge *MFTLHeTwin(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetHalfedge(obj, he->twin_idx); }

/* ---- Twin-state predicates (plans/16 §2.1). The ONLY sanctioned readers of twin_idx-as-state
   and twin_gid (enforced by tools/check_twin_gid_readers.sh); see MFTL_TWIN_* in
   mftl_constants.h. ---- */
static inline bool MFTLHeIsLocal(const MFTL_HalfEdge *he) { return he->twin_idx >= 0; }
static inline bool MFTLHeIsRemote(const MFTL_HalfEdge *he) { return he->twin_idx < 0 && he->twin_gid >= 0; }
static inline bool MFTLHeIsBoundary(const MFTL_HalfEdge *he) { return he->twin_idx < 0 && he->twin_gid <= MFTL_TWIN_NONE; }
/* Illegal at quiescent points: no stored twin and no REMOTE/BOUNDARY code. */
static inline bool MFTLHeIsUnresolved(const MFTL_HalfEdge *he) { return he->twin_idx < 0 && he->twin_gid == MFTL_TWIN_UNKNOWN; }
static inline int MFTLBoundaryCode(int tag) { return MFTL_TWIN_NONE - tag; }
/* Boundary tag of a BOUNDARY half-edge (asserts the state). */
static inline int MFTLHeBoundaryTag(const MFTL_HalfEdge *he)
{
  assert(MFTLHeIsBoundary(he));
  return MFTL_TWIN_NONE - he->twin_gid;
}
/* Mark a half-edge BOUNDARY with a tag / REMOTE with a partner gid (keeps the cache). */
static inline void MFTLHeSetBoundary(MFTL_HalfEdge *he, int tag)
{
  he->twin_idx = MFTL_ERROR;
  he->twin_gid = MFTLBoundaryCode(tag);
  he->boundary = MFTL_TRUE;
}
static inline void MFTLHeSetRemote(MFTL_HalfEdge *he, int partner_gid)
{
  he->twin_idx = MFTL_ERROR;
  he->twin_gid = partner_gid;
  he->boundary = MFTL_FALSE;
}

/* Global id of the opposite half-edge for LOCAL (the stored twin's id) and REMOTE (the
   retained rim id); MFTL_ERROR for BOUNDARY (and for the transient UNKNOWN state), so a
   boundary code is never mistaken for a gid. */
static inline int MFTLHeTwinGid(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  if (he->twin_idx >= 0) return obj->half_edges[he->twin_idx].global_id;
  return (he->twin_gid >= 0) ? he->twin_gid : MFTL_ERROR;
}

/* Raw state code for transport / copies: the twin gid (LOCAL, REMOTE), the boundary code
   (BOUNDARY) or MFTL_TWIN_UNKNOWN. Unpacking this value reproduces the state. */
static inline int MFTLHeTwinCode(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  return (he->twin_idx >= 0) ? obj->half_edges[he->twin_idx].global_id : he->twin_gid;
}

/* Rank-independent undirected edge id: min(gid, twin gid) for LOCAL/REMOTE, gid for BOUNDARY. */
static inline int MFTLHeEdgeId(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  const int tg = MFTLHeTwinGid(obj, he);
  return (tg != MFTL_ERROR && tg < he->global_id) ? tg : he->global_id;
}

/**
 * @brief The next half-edge of the same triangle (winding order).
 */
static inline MFTL_HalfEdge *MFTLHeNext(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetHalfedge(obj, he->next_idx); }

/**
 * @brief The previous half-edge of the same triangle (next->next).
 */
static inline MFTL_HalfEdge *MFTLHePrev(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  const MFTL_HalfEdge *next = &obj->half_edges[he->next_idx];
  return MFTLGetHalfedge(obj, next->next_idx);
}

/**
 * @brief The triangle this half-edge belongs to (every half-edge has one).
 */
static inline MFTL_Triangle *MFTLHeTriangle(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetTriangle(obj, he->triangle_idx); }

/**
 * @brief The triangle across the edge; requires a stored twin (LOCAL; asserted).
 */
static inline MFTL_Triangle *MFTLHeAdjacentTriangle(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  MFTL_HalfEdge *twin = MFTLHeTwin(obj, he);
  assert(twin != MFTL_NULL);
  return MFTLGetTriangle(obj, twin->triangle_idx);
}

/* VERTEX ONE-RING TRAVERSAL
   The fan of a vertex is walked through its outgoing half-edges (out -> twin -> next). Seen
   from the side into which the triangle normals point (outside an outward-wound body),
   next(twin(h)) turns CLOCKWISE around the vertex and twin(prev(h)) counter-clockwise. The
   walk stops at an outgoing half-edge without a stored twin (physical boundary OR MPI rim).
   Unless is_inner is set, the walk first rewinds counter-clockwise to such a stop, so an open
   or rim-cut fan is returned from one end to the other. MFTLVertexRing classifies a fan as
   closed / open-complete / incomplete. */

/**
 * @brief The triangles of the stored fan of v (see above); max_tris bounds out_tris.
 * is_inner = MFTL_TRUE skips the rewind (only valid for a closed fan).
 * @return The number of triangles.
 */
static inline int MFTLGetVertexOneRing(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Triangle **out_tris, int max_tris, bool is_inner)
{
  int count = 0;

  MFTL_HalfEdge *start_he = MFTL_NULL;
  start_he = MFTLVOutHe(obj, v);
  assert(start_he);

  MFTL_HalfEdge *curr = start_he;

  // rewind counter-clockwise to the first outgoing half-edge of an open / cut fan
  if (is_inner == MFTL_FALSE)
  {
    while (true)
    {
      MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
      MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);

      if (!cw_neighbor) break;  // no stored twin: boundary or rim

      if (cw_neighbor == start_he) break;  // closed fan

      curr = cw_neighbor;
    }
    start_he = curr;
  }

  // clockwise walk (next(twin)) collecting the triangles
  do
  {
    assert(count < max_tris && "Buffer overflow in MFTLGetVertexOneRing");

    out_tris[count++] = MFTLGetTriangle(obj, curr->triangle_idx);

    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, curr);

    // end of an open / cut fan (also checked with is_inner)
    if (twin == MFTL_NULL) break;

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief The neighbour vertices of v: the targets of the outgoing half-edges of its stored
 * fan (for an open fan the last neighbour across the final edge is not included).
 * is_inner as in MFTLGetVertexOneRing.
 * @return The number of neighbours.
 */
static inline int MFTLGetVertexOneRingVertices(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Vertex **out_verts, int max_verts, bool is_inner)
{
  int count = 0;

  MFTL_HalfEdge *start_he = MFTL_NULL;
  start_he = MFTLVOutHe(obj, v);
  assert(start_he);

  MFTL_HalfEdge *curr = start_he;

  if (is_inner == MFTL_FALSE)
  {
    while (true)
    {
      MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
      MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);

      if (!cw_neighbor || cw_neighbor == start_he) break;
      curr = cw_neighbor;
    }
    start_he = curr;
  }

  do
  {
    assert(count < max_verts && "Buffer overflow in MFTLGetVertexOneRingVertices");

    out_verts[count++] = MFTLHeTarget(obj, curr);

    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, curr);
    if (!twin) break;

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief The local indices of the outgoing half-edges of the stored fan of v (0 when v has no
 * outgoing half-edge). is_inner as in MFTLGetVertexOneRing.
 * @return The number of half-edges.
 */
static inline int MFTLGetVertexOneRingHalfedges(const MFTL_Object *obj, const MFTL_Vertex *v, int *out_he_indices, int max_he, bool is_inner)
{
  int count = 0;

  MFTL_HalfEdge *start_he = MFTLVOutHe(obj, v);
  if (!start_he) return 0;

  MFTL_HalfEdge *curr = start_he;

  if (is_inner == false)
  {
    while (true)
    {
      MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
      MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);

      if (!cw_neighbor || cw_neighbor == start_he) break;
      curr = cw_neighbor;
    }
    start_he = curr;
  }

  do
  {
    assert(count < max_he && "Buffer overflow in MFTLGetVertexOneRingHalfedges");

    out_he_indices[count++] = (int) (curr - obj->half_edges);

    MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
    if (!twin) break;

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief The triangles across the three edges of tri, in half-edge order; NULL where the edge
 * has no stored twin (physical boundary or MPI rim).
 */
static inline void MFTLGetTriangleNeighbors(MFTL_Object *obj, MFTL_Triangle *tri, MFTL_Triangle *out_neighbors[3])
{
  MFTL_HalfEdge *edges[3];
  MFTLGetTriangleHalfedges(obj, tri, edges);

  for (int i = 0; i < 3; i++)
  {
    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, edges[i]);

    if (twin)
    {
      out_neighbors[i] = MFTLHeTriangle(obj, twin);
    }
    else
    {
      out_neighbors[i] = MFTL_NULL;
    }
  }
}

/**
 * @brief Re-pairs the stored half-edges (rank-local): a half-edge u->v becomes LOCAL with the
 * stored v->u; an unpaired half-edge keeps its state code (REMOTE gid / BOUNDARY code / UNKNOWN).
 * Needs local vertex indices; refreshes the boundary cache and the boundary-vertex
 * out-half-edge convention (MFTLRefreshBoundaryCache).
 */
void MFTLRebuildTwins(MFTL_Object *obj);

/**
 * @brief After unpacking: converts vertex_idx and out_he_idx from global ids to local indices
 * through the id maps (a vertex whose packed out half-edge is not stored gets the next
 * half-edge of an incoming one). Twins are left to MFTLRebuildTwins. A half-edge whose target
 * vertex is missing is fatal.
 */
void MFTLConvertGlobalToLocalIndices(MFTL_Object *obj);

/**
 * @brief Builds the flattened connectivity arrays (rank-local):
 *  - flat_tri_verts[3t + j]: target vertex of the j-th half-edge of triangle t (a cyclic
 *    rotation of MFTLGetTriangleVertices' order, same orientation);
 *  - flat_tri_neighbors[3t + j]: triangle across that half-edge, -1 if its twin is not stored;
 *  - vert_offsets / vert_tris: CSR vertex -> incident stored triangles.
 * Rebuild after every topology change.
 */
void MFTLFlattenConnectivity(MFTL_Object *obj);

/**
 * @brief Rank-local structural check: index bounds, triangle half-edge cycles, twin symmetry
 * and vertex consistency, no unresolved twin state, BOUNDARY only on declared-open bodies.
 * Silent on success; prints every violation. Returns MFTL_SUCCESS / MFTL_ERROR.
 */
int MFTLCheckMeshIntegrity(MFTL_Object *object);

/**
 * @struct MFTL_Neighborhood
 * @brief Result of MFTLGetNeighborhood: gathered vertex and triangle local indices.
 */
typedef struct MFTL_Neighborhood
{
  int *vert_indices;  ///< Array of gathered vertex local indices
  int n_verts;  ///< Number of vertices found
  int cap_verts;  ///< Current capacity of the vertex array

  int *tri_indices;  ///< Array of gathered triangle local indices
  int n_tris;  ///< Number of triangles found
  int cap_tris;  ///< Current capacity of the triangle array
} MFTL_Neighborhood;

/**
 * @brief Frees the arrays of a neighbourhood result.
 */
static inline void MFTLFreeNeighborhood(MFTL_Neighborhood *nb)
{
  free(nb->vert_indices);
  free(nb->tri_indices);
  nb->n_verts = 0;
  nb->n_tris = 0;
}

/**
 * @brief Breadth-first gathering on the vertex/triangle incidence graph (rank-local, stored
 * entities only). Each step alternates the expansion: vertices -> incident triangles (CSR
 * arrays), triangles -> their corners (flat_tri_verts). One step from a vertex gives its
 * triangles; two steps also their vertices (the full one-ring).
 *
 * seed_indices are vertex (use_vertex_seeds) or triangle local indices; the seeds themselves
 * are part of the result. The result arrays are allocated here (free with
 * MFTLFreeNeighborhood). MFTL_FLAG_VISITED is used internally and cleared again.
 */
void MFTLGetNeighborhood(MFTL_Object *restrict obj, int *restrict seed_indices, int n_seeds, bool use_vertex_seeds, int steps,
                         MFTL_Neighborhood *restrict out_result);

/**
 * @brief Builds complete half-edge topology from a vertex list and a list of
 *        triangle index triples (generators and mesh I/O share this path).
 *
 * Wiring convention (matches the generators): for a triangle triple (a, b, c), the three
 * half-edges target the NEXT vertex (b, c, a) and cycle within the triangle. The caller is
 * responsible for consistent (e.g. outward) winding of the triples — see MFTLObjectVolume's
 * sign.
 *
 * Fills vertices/triangles/half_edges, assigns sequential global ids and sets
 * the next_*_gid counters, links one outgoing half-edge per vertex, then runs
 * MFTLRebuildTwins + MFTLFlattenConnectivity. Positions are stored verbatim with img = 0 (a
 * caller with continuous input wraps them first, e.g. MFTLWrapContinuous, and sets the
 * counters); the wrap flags are invalid until the next collective canonicalisation. The object
 * must be empty.
 *
 * Rank-local. A half-edge without a partner becomes BOUNDARY only when the object
 * is declared open (MFTLObjectDeclareOpen; tags = boundary-loop index ordered by the
 * smallest vertex gid of the loop); on a closed object it is an error. Duplicate directed
 * half-edges (non-manifold edge / inconsistent orientation / Mobius) and bow-tie vertices
 * are rejected. Returns MFTL_SUCCESS or MFTL_ERROR (the mesh is then not usable).
 */
int MFTLBuildTopologyFromTriples(MFTL_Object* obj, int nV, const double positions[][3], int nF, const int triples[][3]);

/**
 * @brief Builds an EMPTY object from a distributed triangle soup with global ids (closed body;
 * see MFTLBuildFromSoupEx for open bodies).
 *
 * Vertex i has global id vgid[i], position pos[i] and owner vowner[i] (NULL: this rank);
 * triangle t has global id tgid[t], vertex global ids tvg[t][0..2] (the order defines the
 * orientation) and owner towner[t] (NULL: this rank). The same vertex/triangle may be passed
 * on several ranks (identical ids); duplicates within one rank are ignored. Half-edge ids
 * are compact and identical on every rank holding the triangle; twins are paired locally and
 * the retained twin gid of every rim half-edge is resolved globally. Every entity is stored
 * as local with the given owner; call MFTLRedistribute afterwards. Collective; returns
 * MFTL_ERROR on every rank if the object was not empty or a triangle references an unknown
 * vertex anywhere.
 */
int MFTLBuildFromSoup(MFTL_Object *obj, int nV, const int *vgid, const double (*pos)[3], const int *vowner, int nT, const int *tgid, const int (*tvg)[3],
                      const int *towner);

/* Boundary declaration of a soup (plans/16 §2.2). Honoured only for a body declared open
   (MFTLObjectDeclareOpen, before the build). */
typedef struct MFTL_BoundaryEdgeDecl
{
  int vgid_a, vgid_b;  ///< undirected edge (vertex global ids)
  int tag;  ///< boundary tag >= 0
} MFTL_BoundaryEdgeDecl;

typedef struct MFTL_BoundaryDecl
{
  /* "BoundaryFromMesh": every edge unmatched on every rank is a boundary, auto-tagged by
     loop index ordered by the smallest vertex gid of the loop. */
  int from_mesh;
  int n_edges;  ///< explicit list (the union over all ranks is used)
  const MFTL_BoundaryEdgeDecl *edges;
} MFTL_BoundaryDecl;

/**
 * @brief MFTLBuildFromSoup with an explicit boundary declaration.
 *
 * Twins are paired locally, then every unpaired half-edge is resolved in GLOBAL discovery
 * mode (all ranks, before ownership): a partner anywhere -> REMOTE. An edge unmatched
 * everywhere becomes BOUNDARY only if the body is declared open AND decl declares it (explicit
 * edge list or from_mesh); any other unmatched edge, a declared edge that is not unmatched, a
 * duplicate directed half-edge (non-manifold / inconsistent orientation) or a rim edge that is
 * also interior elsewhere is a collective error. decl == NULL: closed body required.
 */
int MFTLBuildFromSoupEx(MFTL_Object *obj, int nV, const int *vgid, const double (*pos)[3], const int *vowner, int nT, const int *tgid,
                        const int (*tvg)[3], const int *towner, const MFTL_BoundaryDecl *decl);

/** @brief As MFTLBuildFromSoupEx, with the WRAPPED positions and explicit image
 *  counters of every vertex (native restart, §3.2); canonicalised afterwards like every build. */
int MFTLBuildFromSoupImg(MFTL_Object *obj, int nV, const int *vgid, const double (*pos)[3], const int (*vimg)[3], const int *vowner, int nT,
                         const int *tgid, const int (*tvg)[3], const int *towner, const MFTL_BoundaryDecl *decl);

/**
 * @brief Copies triangle tri_idx of src (with its half-edges and missing vertices,
 * payload and surface scalar) into dst unless dst already stores that triangle id. Twins
 * are linked through the retained twin gids. Returns the triangle's index in dst. Rank-local.
 *
 * The copy is STAGED — a half-edge whose twin is not in dst keeps the source's
 * state code (REMOTE gid / BOUNDARY code) and dst is not quiescent until the batch is closed
 * by MFTLObjectCommitCopy (collective).
 */
int MFTLObjectAppendTriangleCopy(MFTL_Object *dst, const MFTL_Object *src, int tri_idx);

/**
 * @brief Closes a batch of staged copies (collective, GLOBAL discovery mode).
 *
 * Every non-LOCAL, non-BOUNDARY half-edge of dst is resolved against the live authoritative
 * (non-ghost) half-edges of dst on all ranks. mode MFTL_COMMIT_ASSEMBLE: an unresolved one is a
 * collective error. mode MFTL_COMMIT_CUT: an unresolved copy of an existing source twin (the
 * explicit cut set) becomes BOUNDARY with ONE new tag (max existing tag + 1); dst must be
 * declared open. Duplicate authoritative copies of one half-edge must be identical (topology
 * and position). Returns MFTL_SUCCESS / MFTL_ERROR on every rank.
 */
int MFTLObjectCommitCopy(MFTL_Object *dst, int mode);

/* ---------------------------------------------------------------------------------------
 * Twin states, declared class, validation, ghost completion, rings and loops (plans/16 §2).
 * All "collective" functions use obj->domain->comm_cart.
 * --------------------------------------------------------------------------------------- */

/** @brief Declare the body open (1) or closed (0, default). Must be identical on all ranks. */
void MFTLObjectDeclareOpen(MFTL_Object *obj, int open);

/**
 * @brief Classifies the fan of vertex v_idx: MFTL_RING_CLOSED, MFTL_RING_OPEN_COMPLETE (the fan
 * runs from an incoming BOUNDARY to an outgoing BOUNDARY half-edge) or MFTL_RING_INCOMPLETE (the
 * walk reaches a REMOTE / unresolved half-edge, or the fan is malformed). When out_he != NULL the
 * outgoing half-edges are stored in walk order, h -> next(twin(h)) (clockwise seen from the
 * normal side; for an open fan: starting after the incoming boundary, ending with the outgoing
 * BOUNDARY half-edge), at most max_he; *n_out receives the count (may be NULL). Rank-local.
 */
int MFTLVertexRing(const MFTL_Object *obj, int v_idx, int *out_he, int max_he, int *n_out);

/**
 * @brief Rank-local twin-state checks (the cheap part of MFTLValidateTwinStates): every live
 * half-edge LOCAL/REMOTE/BOUNDARY, LOCAL symmetry and vertex consistency, boundary cache,
 * BOUNDARY only on declared-open bodies, no duplicate directed half-edge, boundary-vertex
 * out-half-edge convention and bow-ties on complete owned fans; with MFTL_VALIDATE_GHOSTS also
 * G1 (every edge of an owned triangle LOCAL or BOUNDARY) and G2 (complete fan of every owned
 * vertex). Returns the number of violations (printed with gids when verbose).
 */
int MFTLCheckTwinStatesLocal(const MFTL_Object *obj, int flags, int verbose);

/**
 * @brief Collective validation (plans/16 §2.3): the local checks plus remote liveness — every
 * REMOTE twin gid is held as a live authoritative (non-ghost) half-edge on some rank that names
 * the requester back, and every BOUNDARY half-edge carries the same tag on every rank storing
 * it. Returns MFTL_SUCCESS or MFTL_ERROR on every rank (offending gids printed).
 */
int MFTLValidateTwinStatesEx(MFTL_Object *obj, int flags);
static inline int MFTLValidateTwinStates(MFTL_Object *obj) { return MFTLValidateTwinStatesEx(obj, 0); }

/* Ghost-completion / discovery statistics of the last MFTLCompleteGhosts call (for tests). */
typedef struct MFTL_DiscoveryStats
{
  int rounds;  ///< completion rounds executed (the last one found nothing)
  long n_requested;  ///< global number of requested twin gids (all rounds)
  long n_round1;  ///< resolved by the owned-mode geometric round
  long n_fallback;  ///< resolved by the global fallback
  long n_unresolved;  ///< unresolved (collective error)
} MFTL_DiscoveryStats;

/**
 * @brief Guaranteed ghost completion (G1/G2, plans/16 §2.5; collective). Imports, as ghosts,
 * the twin triangle of every REMOTE edge of an owned triangle and of every REMOTE edge incident
 * to an owned vertex, until no rank has one. Each round: owned-mode discovery (candidate ranks
 * whose box meets the periodic ball B(edge midpoint, l_max/3 + 2 eps)) followed by the mandatory
 * global fallback. eps is the ghost-position staleness margin (0 right after an update).
 * Called by every MFTLRedistribute*; stats may be NULL.
 */
int MFTLCompleteGhosts(MFTL_Object *obj, double eps, MFTL_DiscoveryStats *stats);

/** @brief Test hook: 1 disables owned-mode round 1 (everything goes through the fallback). */
void MFTLDiscoverySetForceFallback(int force);

/** @brief Statistics of the last ghost completion run on this rank's process. */
void MFTLGetLastDiscoveryStats(MFTL_DiscoveryStats *stats);

/* Boundary loops (plans/16 §2.6), identical on every rank. */
typedef struct MFTL_BoundaryLoop
{
  int n;  ///< number of boundary edges (= vertices) in the loop
  int *edge_ids;  ///< edge ids in loop order (edge i runs vertex_gids[i] -> vertex_gids[i+1])
  int *vertex_gids;
  int tag;  ///< tag of the loop (the smallest tag when mixed)
  int mixed_tags;  ///< 1 when the loop carries more than one tag
  double length;  ///< sum of the (minimum-image) edge lengths
  int winding[3];  ///< sum of the edge lattice shifts around the loop (plans/16 §2.6/§3.3)
} MFTL_BoundaryLoop;

typedef struct MFTL_BoundaryLoopSet
{
  int n_loops;
  MFTL_BoundaryLoop *loops;  ///< ordered by the smallest vertex gid of each loop
} MFTL_BoundaryLoopSet;

/**
 * @brief Collective: builds all boundary loops from the owned boundary vertices (requires G2,
 * i.e. call after a redistribution, or on a single rank). Every owned boundary vertex emits
 * (in edge id, out edge id, vertex gid, tag, out-edge length); the records are gathered and
 * chained on every rank. Returns MFTL_ERROR (all ranks) on a broken chain.
 */
int MFTLBoundaryLoops(MFTL_Object *obj, MFTL_BoundaryLoopSet *out);
void MFTLFreeBoundaryLoops(MFTL_BoundaryLoopSet *set);

/* Internal (library): after local pairing, sets the boundary cache from the state and points
   every vertex with an outgoing BOUNDARY half-edge at it (plans/16 §2.6). */
void MFTLRefreshBoundaryCache(MFTL_Object *obj);

/** @brief Removes every entity (and payload) from an object; rank-local. */
void MFTLObjectClear(MFTL_Object *obj);

#endif  // MFTL_TOPOLOGY_H