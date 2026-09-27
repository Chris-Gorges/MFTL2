/*
 * MFTL2 intersection — linear bounding volume hierarchy (LBVH) and triangle-triangle
 * intersection / proximity between objects.
 *
 *  - LBVH: triangle centroids -> 30-bit Morton codes (Z-order) -> radix sort -> top-down split
 *    at the highest differing bit of the (code, index) keys; leaves hold <= 4 triangles; node
 *    boxes are float and inflated by a fixed 1e-4. Nodes are 32 bytes (two float triples and a
 *    packed 64-bit payload).
 *  - Pair queries: simultaneous traversal of two LBVHs; leaf pairs are tested with a
 *    Guigue-Devillers orientation-predicate triangle test, extended to a proximity test (9
 *    edge-edge and 6 vertex-face distances) with tolerance 1e-6.
 *  - Periodicity: node boxes are built from the stored (wrapped) positions, so a seam-crossing
 *    triangle has a stretched, conservative box; the box tests shift by one period on periodic
 *    axes, and each tested triangle pair is first placed in one common frame (minimum image).
 *  - Spatial ghosts (MFTL_CreateSpatialGhosts): owned triangles of one object whose box meets
 *    the other object's box on a neighbour rank are shipped there as ghosts.
 *
 * Known limitations: exactly coplanar triangle pairs are reported as intersecting (no 2-D
 * overlap test); the box inflation and proximity tolerance are absolute (not scaled with the
 * mesh).
 */

#ifndef MFTL_BVH_H
#define MFTL_BVH_H

#include "mftl_core.h"
#include <stdint.h>

/**
 * @struct MFTL_BVHNode
 * @brief LBVH node (32 bytes: two nodes per 64-byte cache line).
 *
 * For a Leaf Node:
 *   payload encodes: first_prim_index and prim_count
 * For an Internal Node:
 *   payload encodes: left_child index (right_child is always left_child + 1).
 */
typedef struct MFTL_BVHNode
{
  float aabb_min[3];  // 12 bytes
  float aabb_max[3];  // 12 bytes (total 24 bytes)

  // Payload: 8 bytes
  // Lowest bit of payload is a flag (1 = leaf, 0 = internal node).
  //
  // If leaf (bit 0 = 1):
  //   Bits 1-31: prim_count
  //   Bits 32-63: first_prim_index
  //
  // If internal (bit 0 = 0):
  //   Bits 1-63: left_child_index
  uint64_t payload;

} MFTL_BVHNode;

_Static_assert(sizeof(MFTL_BVHNode) == 32, "MFTL_BVHNode must be exactly 32 bytes for cache alignment!");

/**
 * @struct MFTL_BVH
 * @brief Top-level Bounding Volume Hierarchy structure.
 */
typedef struct MFTL_BVH
{
  MFTL_BVHNode* nodes;  ///< Array of all BVH nodes
  int* prim_indices;  ///< Triangle local indices in Morton order (leaves index into this)

  int nodes_used;  ///< Number of currently active nodes
  int nodes_capacity;  ///< Total allocated capacity for nodes array

  int root_index;  ///< Index of the root node (usually 0)
} MFTL_BVH;

/**
 * @brief Called for every intersecting / proximate triangle pair (local triangle indices of
 * objA and objB) found by a pair traversal.
 */
typedef void (*IntersectionCallback)(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, void* user_data);

/**
 * @brief Pairwise spatial ghosting (collective): every rank sends its owned-triangle bounding
 * boxes of objA and objB to its graph neighbours; owned triangles of one object whose box meets
 * the neighbour's box of the other object are shipped there and appended as ghosts (with their
 * vertices and payload; flat_tri_verts is extended, twins are not re-paired). Box tests are
 * periodic-aware. MFTL_ERROR (all ranks) on a payload-size mismatch.
 */
int MFTL_CreateSpatialGhosts(MFTL_Object* objA, MFTL_Object* objB);

/**
 * @brief Builds an LBVH over all stored triangles of obj (rank-local; needs flat_tri_verts).
 * @return A new BVH (free with MFTL_DestroyBVH), or NULL for an empty object.
 */
MFTL_BVH* MFTL_GenerateLBVH(MFTL_Object* obj);

/** @brief Frees a BVH. */
void MFTL_DestroyBVH(MFTL_BVH* bvh);

/**
 * @brief Simultaneous traversal of two LBVHs (rank-local): calls callback for every triangle
 * pair that intersects or lies within the 1e-6 proximity tolerance.
 */
void MFTL_IntersectObjects(MFTL_Object* objA, MFTL_BVH* bvhA, MFTL_Object* objB, MFTL_BVH* bvhB, IntersectionCallback callback, void* user_data);

/**
 * @brief Guigue-Devillers triangle-triangle intersection test (orientation predicates on the
 * flat_tri_verts corners; periodic domains: both triangles in one common frame).
 * @return 1 if they intersect, 0 if disjoint. Exactly coplanar pairs return 1 (conservative;
 * no 2-D overlap test).
 */
int MFTL_TriTriIntersect(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id);

/**
 * @brief Intersection of full ensembles (collective for the ghosting): pairwise spatial ghosts,
 * per-object LBVHs and pair traversal (self-intersection when ensB is NULL; object pairs i < j,
 * so no pair is tested twice). The callback runs rank-locally, on the stored triangles.
 */
void MFTLEnsembleIntersectObjects(MFTL_Ensemble* ensA, MFTL_Ensemble* ensB, IntersectionCallback callback, void* user_data);

#endif  // MFTL_BVH_H
