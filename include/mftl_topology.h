/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:35:27
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
 */

#ifndef MFTL_TOPOLOGY_H
#define MFTL_TOPOLOGY_H

#include "mftl_types.h"
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