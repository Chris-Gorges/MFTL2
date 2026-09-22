/**
 * @file mftl_bvh.h
 * @brief Linear Bounding Volume Hierarchy (LBVH) implementation.
 *
 * This module provides a highly optimized, state-of-the-art implementation of a 
 * Bounding Volume Hierarchy specifically designed for parallel surface meshing.
 *
 * ### Key Algorithmic Features:
 * 1. **Data Oriented Design (DOD)**: The `MFTL_BVHNode` architecture is strictly compressed 
 *    to 32-bytes, matching exact cache-line boundaries for maximum pre-fetching efficiency.
 * 2. **Radix Tree Spatial Partitioning**: The topology is built utilizing a deterministically 
 *    stable binary split based on 30-bit Morton Codes (Z-Order curves). It leverages 
 *    `__builtin_clz` for hyper-fast highest differing bit analysis, slicing nodes exactly at their radix point.
 * 3. **MPI Spatial Ghosting**: Utilizes `MPI_Neighbor_alltoall[v]` for boundary-aware triangle packing.
 *    Nodes overlapping distributed bounds are sequentially transmitted, unpacked, and organically 
 *    fused into the mesh `flat_tri_verts` for cohesive cross-boundary resolution.
 * 4. **Guigue-Devillers Robust Intersection**: Abandons conventional SAT arrays for rigorous 
 *    3D Orientation determinants. Automatically resolves exact interval overlapping intervals on planes 
 *    while avoiding heavy two-dimensional geometric projections or topological linkage tracking.
 *
 * @author Christian Gorges
 * @date 2026-03-01
 */

#ifndef MFTL_BVH_H
#define MFTL_BVH_H

#include "mftl_core.h"
#include <stdint.h>

/**
 * @struct MFTL_BVHNode
 * @brief Node for the Linear Bounding Volume Hierarchy (LBVH).
 *
 * Designed to fit precisely into 32 bytes to ensure perfect cache-line alignment.
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

// Static assert to ensure 32-byte struct size
_Static_assert(sizeof(MFTL_BVHNode) == 32, "MFTL_BVHNode must be exactly 32 bytes for cache alignment!");

/**
 * @struct MFTL_BVH
 * @brief Top-level Bounding Volume Hierarchy structure.
 */
typedef struct MFTL_BVH
{
  MFTL_BVHNode* nodes;  ///< Array of all BVH nodes
  int* prim_indices;  ///< Reordered triangle indices from radix sort

  int nodes_used;  ///< Number of currently active nodes
  int nodes_capacity;  ///< Total allocated capacity for nodes array

  int root_index;  ///< Index of the root node (usually 0)
} MFTL_BVH;

/**
 * @brief Callback for processing overlapping primitive pairs.
 * @param objA Pointer to the first object.
 * @param triA_id Index of the intersected triangle in objA.
 * @param objB Pointer to the second object.
 * @param triB_id Index of the intersected triangle in objB.
 * @param user_data Pointer to an optional user-provided data structure.
 */
typedef void (*IntersectionCallback)(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, void* user_data);

/**
 * @brief Exchangers bounding box and intersecting triangles across MPI domains.
 * 
 * @param objA The first object.
 * @param objB The second object.
 * @return MFTL_SUCCESS on success.
 */
int MFTL_CreateSpatialGhosts(MFTL_Object* objA, MFTL_Object* objB);

/**
 * @brief Generates a Linear Bounding Volume Hierarchy (LBVH) for a given object.
 * 
 * Extracts normalized centroids, generates Morton Codes, and builds the tree
 * using a top-down split over the sorted codes. Finally refits the AABBs.
 * 
 * @param obj The mesh object (must have flat_tri_verts built).
 * @return A new BVH, or NULL on failure.
 */
MFTL_BVH* MFTL_GenerateLBVH(MFTL_Object* obj);

/**
 * @brief Destroys a BVH and frees its resources.
 * @param bvh The BVH to destroy.
 */
void MFTL_DestroyBVH(MFTL_BVH* bvh);

/**
 * @brief Simultaneously traverses two BVH trees to find intersecting Object primitives.
 * 
 * @param objA The first mesh object.
 * @param bvhA The first object's BVH.
 * @param objB The second mesh object.
 * @param bvhB The second object's BVH.
 * @param callback The function to call when a Triangle-Triangle intersection is found.
 * @param user_data User data to pass to the callback.
 */
void MFTL_IntersectObjects(MFTL_Object* objA, MFTL_BVH* bvhA, MFTL_Object* objB, MFTL_BVH* bvhB, IntersectionCallback callback, void* user_data);

/**
 * @brief Performs a robust exact Triangle-Triangle intersection test.
 *
 * Standard Guigue-Devillers triangle intersection. Uses the `flat_tri_verts`
 * for geometric lookups directly.
 *
 * @param objA The first mesh object.
 * @param triA_id Index of the first triangle.
 * @param objB The second mesh object.
 * @param triB_id Index of the second triangle.
 * @return 1 if they intersect, 0 if disjoint (or coplanar, if non-overlapping).
 */
int MFTL_TriTriIntersect(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id);

/**
 * @brief Scalable intersection pipeline for full ensembles: spatial ghosts,
 * per-object LBVHs and pair traversal (self-intersection when ensB is NULL;
 * object pairs i < j, so no pair is tested twice).
 */
void MFTLEnsembleIntersectObjects(MFTL_Ensemble* ensA, MFTL_Ensemble* ensB, IntersectionCallback callback, void* user_data);

#endif  // MFTL_BVH_H
