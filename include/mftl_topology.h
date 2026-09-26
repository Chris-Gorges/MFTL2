/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:35:27
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
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
// 2. VERTEX NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief Get the outgoing half-edge starting from this vertex.
 * @return Pointer to the half-edge, or NULL if isolated.
 */
static inline MFTL_HalfEdge *MFTLVOutHe(const MFTL_Object *obj, const MFTL_Vertex *v) { return MFTLGetHalfedge(obj, v->out_he_idx); }

// -----------------------------------------------------------------------------
// 3. TRIANGLE NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief Get the first half-edge associated with this triangle.
 */
static inline MFTL_HalfEdge *MFTLTHe(const MFTL_Object *obj, const MFTL_Triangle *t) { return MFTLGetHalfedge(obj, t->he_idx); }

/**
 * @brief Get the three half-edges of a triangle in CCW order.
 * @param[out] out_he Array of 3 pointers to store the results.
 */
static inline void MFTLGetTriangleHalfedges(const MFTL_Object *obj, const MFTL_Triangle *tri, MFTL_HalfEdge *out_he[3])
{
  out_he[0] = (MFTL_HalfEdge *) &obj->half_edges[tri->he_idx];
  out_he[1] = (MFTL_HalfEdge *) &obj->half_edges[out_he[0]->next_idx];
  out_he[2] = (MFTL_HalfEdge *) &obj->half_edges[out_he[1]->next_idx];
}

/**
 * @brief Get the three vertices of a triangle in CCW order.
 * @param[out] out_verts Array of 3 pointers to store the results.
 */
static inline void MFTLGetTriangleVertices(const MFTL_Object *obj, const MFTL_Triangle *tri, MFTL_Vertex *out_verts[3])
{
  MFTL_HalfEdge *out_he[3];
  MFTLGetTriangleHalfedges(obj, tri, out_he);

  // Assign Vertices
  // Note order: The "first" vertex of the triangle is the Origin of HE0.
  // Origin of HE0 == Target of HE2.
  out_verts[0] = (MFTL_Vertex *) &obj->vertices[out_he[2]->vertex_idx];
  out_verts[1] = (MFTL_Vertex *) &obj->vertices[out_he[0]->vertex_idx];
  out_verts[2] = (MFTL_Vertex *) &obj->vertices[out_he[1]->vertex_idx];
}

// -----------------------------------------------------------------------------
// 4. HALFEDGE NAVIGATION (The Core Topology)
// -----------------------------------------------------------------------------

/**
 * @brief Get the vertex at the END (Target) of this half-edge.
 */
static inline MFTL_Vertex *MFTLHeTarget(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetVertex(obj, he->vertex_idx); }

/**
 * @brief Get the vertex at the START (Origin) of this half-edge.
 * * @note Uses the "Next->Next" traversal strategy. This is safer than 
 * using twin->vertex because it works on boundary edges where twin is NULL.
 */
static inline MFTL_Vertex *MFTLHeOrigin(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  // Jump forward twice in the triangle loop to find the "previous" edge
  const MFTL_HalfEdge *next = &obj->half_edges[he->next_idx];
  const MFTL_HalfEdge *prev = &obj->half_edges[next->next_idx];

  // The target of the "previous" edge is the origin of the "current" edge
  return MFTLGetVertex(obj, prev->vertex_idx);
}

/**
 * @brief Get the TWIN (Opposite) half-edge.
 * @return Pointer to twin, or NULL if this is a boundary edge.
 */
static inline MFTL_HalfEdge *MFTLHeTwin(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetHalfedge(obj, he->twin_idx); }

/* ---- Stage 16A twin-state predicates (plans/16 §2.1). The ONLY sanctioned readers of
   twin_idx-as-state and twin_gid; see MFTL_TWIN_* in mftl_constants.h. ---- */
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
   boundary code is never mistaken for a gid. Stage 15 (L3) / Stage 16A. */
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
 * @brief Get the NEXT half-edge in the triangle loop (CCW).
 */
static inline MFTL_HalfEdge *MFTLHeNext(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetHalfedge(obj, he->next_idx); }

/**
 * @brief Get the PREVIOUS half-edge in the triangle loop (CCW).
 * @note Implicitly calculates Next->Next.
 */
static inline MFTL_HalfEdge *MFTLHePrev(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  const MFTL_HalfEdge *next = &obj->half_edges[he->next_idx];
  return MFTLGetHalfedge(obj, next->next_idx);
}

/**
 * @brief Get the TRIANGLE this half-edge belongs to.
 * @return Pointer to triangle, or NULL if this is a boundary half-edge.
 */
static inline MFTL_Triangle *MFTLHeTriangle(const MFTL_Object *obj, const MFTL_HalfEdge *he) { return MFTLGetTriangle(obj, he->triangle_idx); }

/**
 * @brief Get the ADJACENT TRIANGLE this half-edge's twin belongs to.
 * @return Pointer to triangle, or NULL if this is a boundary half-edge.
 */
static inline MFTL_Triangle *MFTLHeAdjacentTriangle(const MFTL_Object *obj, const MFTL_HalfEdge *he)
{
  MFTL_HalfEdge *twin = MFTLHeTwin(obj, he);
  assert(twin != MFTL_NULL);
  return MFTLGetTriangle(obj, twin->triangle_idx);
}

/* VERTEX ONE-RING TRAVERSAL FUNCTIONS */

/**
 * @brief Retrieves the 1-ring of triangles surrounding a vertex.
 * * @param obj The mesh object.
 * @param v The vertex to query.
 * @param[out] out_tris User-allocated buffer for triangle pointers.
 * @param max_tris Safety limit for the buffer.
 * @param is_inner Set to MFTL_TRUE if you are CERTAIN the vertex is an inner vertex and has a full one ring neighborhood.
 * This skips the boundary search (Rewind step) for performance.
 * @return The number of triangles found.
 */
static inline int MFTLGetVertexOneRing(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Triangle **out_tris, int max_tris, bool is_inner)
{
  int count = 0;

  // 1. Start at the vertex's primary outgoing half-edge
  MFTL_HalfEdge *start_he = MFTL_NULL;
  start_he = MFTLVOutHe(obj, v);
  assert(start_he);

  MFTL_HalfEdge *curr = start_he;

  // 2. Rewind (Only if NOT inner)
  // If we know it's an inner vertex, we are already inside a closed loop,
  // so any starting edge is valid and we don't need to find a "start" point.
  if (is_inner == MFTL_FALSE)
  {
    while (true)
    {
      MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
      MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);

      // Stop if we hit a boundary
      if (!cw_neighbor) break;

      // Stop if we wrapped around (closed 1-ring)
      if (cw_neighbor == start_he) break;

      curr = cw_neighbor;
    }
    // Update start_he so the CCW loop knows when it has completed the circle
    start_he = curr;
  }

  // 3. Replay: Move Counter-Clockwise (CCW) to collect triangles.
  do
  {
    // Buffer safety check
    assert(count < max_tris && "Buffer overflow in MFTLGetVertexOneRing");

    // Add the triangle belonging to the current half-edge
    out_tris[count++] = MFTLGetTriangle(obj, curr->triangle_idx);

    // Move to the next outgoing half-edge in CCW order
    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, curr);

    // Boundary check (Required even for inner loops to detect the end of the loop logic in case of boundary edges)
    if (twin == MFTL_NULL) break;

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief Retrieves the 1-ring of neighbor vertices.
 *
 * @param obj The mesh object.
 * @param v The central vertex.
 * @param[out] out_verts User-allocated array for vertex pointers.
 * @param max_verts Safety limit for the buffer.
 * @param is_inner Optimization flag. Set to TRUE to skip boundary rewind.
 * @return The number of neighbors found.
 */
static inline int MFTLGetVertexOneRingVertices(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Vertex **out_verts, int max_verts, bool is_inner)
{
  int count = 0;

  MFTL_HalfEdge *start_he = MFTL_NULL;
  start_he = MFTLVOutHe(obj, v);
  assert(start_he);

  MFTL_HalfEdge *curr = start_he;

  // 2. Rewind (Optional)
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

  // 3. Replay (CCW)
  do
  {
    assert(count < max_verts && "Buffer overflow in MFTLGetVertexOneRingVertices");

    // Add Neighbor (Target of outgoing HE)
    out_verts[count++] = MFTLHeTarget(obj, curr);

    // Advance
    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, curr);
    if (!twin) break;

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief Retrieves the indices of the 1-ring of outgoing half-edges for a vertex.
 * This is the "Edge-based" variant of the one-ring search.
 *
 * @param obj The mesh object.
 * @param v The central vertex.
 * @param[out] out_he_indices User-allocated array for half-edge INDICES.
 * @param max_he Safety limit for the buffer (usually 64 is plenty).
 * @param is_inner Optimization flag. Set to true if you are certain the vertex is not on the boundary.
 * @return The number of edges found.
 */
static inline int MFTLGetVertexOneRingHalfedges(const MFTL_Object *obj, const MFTL_Vertex *v, int *out_he_indices, int max_he, bool is_inner)
{
  int count = 0;

  // 1. Get Start
  MFTL_HalfEdge *start_he = MFTLVOutHe(obj, v);
  if (!start_he) return 0;  // Isolated vertex

  MFTL_HalfEdge *curr = start_he;

  // 2. Rewind (Find the "first" edge in the fan if on boundary)
  if (is_inner == false)
  {
    while (true)
    {
      MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);  // Uses Next->Next
      MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);

      // Stop if boundary hit or loop closed
      if (!cw_neighbor || cw_neighbor == start_he) break;
      curr = cw_neighbor;
    }
    start_he = curr;
  }

  // 3. Replay (Collect Indices)
  do
  {
    assert(count < max_he && "Buffer overflow in MFTLGetVertexOneRingHalfedges");

    // CALCULATE INDEX: Pointer Arithmetic (Safe within this function scope)
    // This converts the pointer 'curr' back to its integer index in the pool.
    out_he_indices[count++] = (int) (curr - obj->half_edges);

    // Advance
    MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
    if (!twin) break;  // Boundary stop

    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  return count;
}

/**
 * @brief Retrieves the three neighboring triangles of a given triangle.
 * * The neighbors are ordered corresponding to the triangle's half-edges (CCW).
 * * @param obj The mesh object.
 * @param tri The triangle to query.
 * @param[out] out_neighbors An array of 3 MFTL_Triangle* pointers. 
 * Entries will be NULL if the edge is a boundary.
 */
static inline void MFTLGetTriangleNeighbors(MFTL_Object *obj, MFTL_Triangle *tri, MFTL_Triangle *out_neighbors[3])
{
  // 1. Get the three half-edges of the triangle in CCW order
  MFTL_HalfEdge *edges[3];
  MFTLGetTriangleHalfedges(obj, tri, edges);

  // 2. Iterate over the edges to find the adjacent triangles
  for (int i = 0; i < 3; i++)
  {
    // The neighbor is located across the Twin half-edge
    MFTL_HalfEdge *twin = MFTL_NULL;
    twin = MFTLHeTwin(obj, edges[i]);

    if (twin)
    {
      // Found a neighbor
      out_neighbors[i] = MFTLHeTriangle(obj, twin);
    }
    else
    {
      // This edge is a boundary (no neighbor)
      out_neighbors[i] = MFTL_NULL;
    }
  }
}

/**
 * @brief Rebuilds the twin indices for all half-edges.
 */
void MFTLRebuildTwins(MFTL_Object *obj);

/**
 * @brief Converts global ids to local indices based on the hash maps.
 */
void MFTLConvertGlobalToLocalIndices(MFTL_Object *obj);

/**
 * @brief Builds internal cache-friendly integer arrays for mesh connectivity.
 * Populates: flat_tri_verts, flat_tri_neighbors, vert_offsets, vert_tris.
 * 1. Triangle Maps (flat_tri_*): Direct lookup of vertices and neighbors for a triangle.
 * 2. Vertex Maps (vert_*): Compressed Sparse Row (CSR) structure for lookup of incident triangles for a vertex.
 */
void MFTLFlattenConnectivity(MFTL_Object *obj);

int MFTLCheckMeshIntegrity(MFTL_Object *object);

/**
 * @struct MFTL_Neighborhood
 * @brief Container for the results of a neighborhood search.
 * Includes dynamic arrays for vertices and triangles found during traversal.
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
 * @brief Helper to free the neighborhood result arrays.
 */
static inline void MFTLFreeNeighborhood(MFTL_Neighborhood *nb)
{
  free(nb->vert_indices);
  free(nb->tri_indices);
  nb->n_verts = 0;
  nb->n_tris = 0;
}

/**
 * @brief Gather the neighborhood of a set of seeds by traversing the mesh topology.
 *
 * @details This function performs a Breadth-First Search (BFS) starting from the provided
 * seed indices (vertices or triangles). It treats the mesh as a bipartite graph
 * where vertices connect to triangles, and triangles connect to vertices.
 * * It alternates expansion types at each step:
 * - If currently holding **Vertices**: It gathers all incident **Triangles**.
 * - If currently holding **Triangles**: It gathers all incident **Vertices**.
 * * This approach is more robust than purely edge-based traversal as it naturally 
 * captures the full "star" of a vertex or the full set of corners of a triangle.
 *
 * ### Optimization
 * The function uses the flattened connectivity arrays (`vert_offsets` / `vert_tris` for V->T 
 * and `flat_tri_verts` for T->V) to ensure maximum cache coherency and avoid pointer chasing.
 * It uses a double-buffered queue system to minimize memory allocation overhead during the loop.
 *
 * @note    The meaning of `steps` is topological hops, not geometric distance:
 * - **1 Step** from a Vertex gathers its incident Triangles (the "Umbrella").
 * - **2 Steps** from a Vertex gathers incident Triangles AND their other Vertices (the full "1-Ring").
 *
 * @param[in]  obj               The mesh object containing the topology.
 * @param[in]  seed_indices      Array of local indices serving as the starting front.
 * @param[in]  n_seeds           Number of indices in `seed_indices`.
 * @param[in]  use_vertex_seeds  Set to `true` if seeds are Vertex indices, `false` for Triangle indices.
 * @param[in]  steps             Number of bipartite hops to expand. 
 * (e.g., 2 steps = V->T->V).
 * @param[out] out_result        Pointer to a user-allocated `MFTL_Neighborhood` struct 
 * where the results will be stored. Arrays are allocated internally 
 * and must be freed by the user (or `MFTLFreeNeighborhood`).
 */
void MFTLGetNeighborhood(MFTL_Object *restrict obj, int *restrict seed_indices, int n_seeds, bool use_vertex_seeds, int steps,
                         MFTL_Neighborhood *restrict out_result);

/**
 * @brief Builds complete half-edge topology from a vertex list and a list of
 *        triangle index triples (generators and mesh I/O share this path).
 *
 * Wiring convention (matches generate_icosahedron/generators): for a triangle
 * triple (a, b, c), the three half-edges target the NEXT vertex (b, c, a) and
 * cycle within the triangle. The caller is responsible for consistent
 * (e.g. outward) winding of the triples — see MFTLObjectVolume's sign.
 *
 * Fills vertices/triangles/half_edges, assigns sequential global ids and sets
 * the next_*_gid counters, links one outgoing half-edge per vertex, then runs
 * MFTLRebuildTwins + MFTLFlattenConnectivity. Positions are stored verbatim
 * (NO periodic wrapping — callers apply MFTLApplyPeriodicity beforehand if
 * desired). The object must be empty.
 *
 * Stage 16A: rank-local. A half-edge without a partner becomes BOUNDARY only when the object
 * is declared open (MFTLObjectDeclareOpen; tags = boundary-loop index ordered by the
 * smallest vertex gid of the loop); on a closed object it is an error. Duplicate directed
 * half-edges (non-manifold edge / inconsistent orientation / Mobius) and bow-tie vertices
 * are rejected. Returns MFTL_SUCCESS or MFTL_ERROR (the mesh is then not usable).
 */
int MFTLBuildTopologyFromTriples(MFTL_Object* obj, int nV, const double positions[][3], int nF, const int triples[][3]);

/**
 * @brief Stage 15: builds an EMPTY object from a distributed triangle soup with global ids.
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

/* Stage 16A boundary declaration of a soup (plans/16 §2.2). Honoured only for a body declared
   open (MFTLObjectDeclareOpen, before the build). */
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
 * @brief Stage 16A: MFTLBuildFromSoup with an explicit boundary declaration.
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

/** @brief Stage 16B: as MFTLBuildFromSoupEx, with the WRAPPED positions and explicit image
 *  counters of every vertex (native restart, §3.2); canonicalised afterwards like every build. */
int MFTLBuildFromSoupImg(MFTL_Object *obj, int nV, const int *vgid, const double (*pos)[3], const int (*vimg)[3], const int *vowner, int nT,
                         const int *tgid, const int (*tvg)[3], const int *towner, const MFTL_BoundaryDecl *decl);

/**
 * @brief Stage 15: copies triangle tri_idx of src (with its half-edges and missing vertices,
 * payload and surface scalar) into dst unless dst already stores that triangle id. Twins
 * are linked through the retained twin gids. Returns the triangle's index in dst. Rank-local.
 *
 * Stage 16A: the copy is STAGED — a half-edge whose twin is not in dst keeps the source's
 * state code (REMOTE gid / BOUNDARY code) and dst is not quiescent until the batch is closed
 * by MFTLObjectCommitCopy (collective).
 */
int MFTLObjectAppendTriangleCopy(MFTL_Object *dst, const MFTL_Object *src, int tri_idx);

/**
 * @brief Stage 16A: closes a batch of staged copies (collective, GLOBAL discovery mode).
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
 * Stage 16A: twin states, declared class, validation, ghost completion, rings and loops
 * (plans/16 §2). All "collective" functions use obj->domain->comm_cart.
 * --------------------------------------------------------------------------------------- */

/** @brief Declare the body open (1) or closed (0, default). Must be identical on all ranks. */
void MFTLObjectDeclareOpen(MFTL_Object *obj, int open);

/**
 * @brief Classifies the fan of vertex v_idx: MFTL_RING_CLOSED, MFTL_RING_OPEN_COMPLETE (the fan
 * runs from an incoming BOUNDARY to an outgoing BOUNDARY half-edge) or MFTL_RING_INCOMPLETE (the
 * walk reaches a REMOTE / unresolved half-edge, or the fan is malformed). When out_he != NULL the
 * outgoing half-edges are stored in CCW order (for an open fan: starting after the incoming
 * boundary, ending with the outgoing BOUNDARY half-edge), at most max_he; *n_out receives the
 * count (may be NULL). Rank-local.
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
  int winding[3];  ///< Stage 16B: sum of the edge lattice shifts around the loop (§2.6/§3.3)
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

/** @brief Stage 15: removes every entity (and payload) from an object; rank-local. */
void MFTLObjectClear(MFTL_Object *obj);

#endif  // MFTL_TOPOLOGY_H