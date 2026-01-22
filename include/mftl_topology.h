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
static inline MFTL_Vertex *MFTLGetVertex(const MFTL_Object *obj, mftl_int_t idx) { return &obj->vertices[idx]; }

/**
 * @brief Get a pointer to a half-edge by its local index.
 */
static inline MFTL_HalfEdge *MFTLGetHalfedge(const MFTL_Object *obj, mftl_int_t idx) { return &obj->half_edges[idx]; }

/**
 * @brief Get a pointer to a triangle by its local index.
 */
static inline MFTL_Triangle *MFTLGetTriangle(const MFTL_Object *obj, mftl_int_t idx) { return &obj->triangles[idx]; }

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
static inline mftl_int_t MFTLGetVertexOneRing(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Triangle **out_tris, mftl_int_t max_tris, bool is_inner)
{
  mftl_int_t count = 0;

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
static inline mftl_int_t MFTLGetVertexOneRingVertices(const MFTL_Object *obj, const MFTL_Vertex *v, MFTL_Vertex **out_verts, mftl_int_t max_verts,
                                                      bool is_inner)
{
  mftl_int_t count = 0;

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
 * @brief Flattens mesh connectivity into integer arrays for cache-friendly processing.
 * * @param object The mesh object containing triangles and half-edges.
 * @param flat_tri_verts Output array for vertex indices (Size: 3 * n_triangles). Can be NULL.
 * @param flat_tri_neighbors Output array for neighbor triangle indices (Size: 3 * n_triangles). Can be NULL.
 */
void MFTLFlattenConnectivity(MFTL_Object *object, mftl_int_t *flat_tri_verts, mftl_int_t *flat_tri_neighbors);

mftl_int_t MFTLCheckMeshIntegrity(MFTL_Object *object);

#endif  // MFTL_TOPOLOGY_H