/*
 * @Author: Christian Gorges 
 * @Date: 2025-09-06 15:29:50 
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 20:18:43
 */
#ifndef MFTL_COMMUNICATIONS_H
#define MFTL_COMMUNICATIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>  // Required for offsetof
#include <stdint.h>  // For int64_t
#include <assert.h>
#include <stdbool.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "khash.h"

// --- Data structures for MPI communication ---
/**
 * @struct PackedVertex
 * @brief A pointer-free, serializable representation of an MFTL_Vertex.
 */
typedef struct
{
  // --- Identity & Ownership ---
  mftl_int_t global_id;
  mftl_int_t owner_proc;

  // --- Geometric Properties ---
  double pos[3];
  double normal[3];
  double meancurvature;
  double gaussiancurvature;
  double area;

  // --- State Information ---
  bool is_ghost;
  mftl_int_t flags;

  // --- Pointer Information ---
  mftl_int_t id_he_out;

  mftl_int_t body_id;

} PackedVertex;

/**
 * @struct PackedTriangle
 * @brief A pointer-free, serializable representation of an MFTL_Triangle.
 */
typedef struct
{
  // --- Identity & Ownership ---
  mftl_int_t global_id;
  mftl_int_t owner_proc;

  // --- Relational Data ---
  mftl_int_t he_id;  // id of one halfedge belonging to this triangle

  // --- Geometric Properties ---
  double normal[3];
  double area;
  double center[3];

  // --- State Information ---
  bool is_ghost;
  mftl_int_t flags;

  mftl_int_t body_id;
} PackedTriangle;

/**
 * @struct PackedHalfEdge
 * @brief A pointer-free, serializable representation of an MFTL_HalfEdge.
 */
typedef struct
{
  // --- Identity & Ownership ---
  mftl_int_t global_id;
  mftl_int_t owner_proc;

  // --- Relational Data (was pointers) ---
  mftl_int_t vertex_gid;  // ID of the vertex at the end of this edge
  mftl_int_t twin_gid;  // ID of the opposite half-edge
  mftl_int_t next_gid;  // ID of the next half-edge around the face
  mftl_int_t triangle_gid;  // ID of the triangle this edge belongs to

  // --- State Information ---
  bool boundary;
  mftl_int_t flags;
} PackedHalfEdge;

/**
 * @struct MFTL_PackedTriangleComplex
 * @brief A "super object" for sending a triangle and its full neighborhood.
 *
 * This struct groups a triangle, its three vertices, and its three interior
 * half-edges into a single object for efficient MPI transfer.
 */
typedef struct
{
  PackedTriangle triangle;
  PackedHalfEdge half_edges[3];
} PackedTriangleComplex;

// Define a struct to hold buffer info for the packing pass
typedef struct
{
  mftl_int_t count;  // Will be used as the current packing index
  mftl_int_t capacity;
  void *buffer;
} MFTL_SendBuffer;

// A lightweight struct to send just the necessary data.
typedef struct
{
  mftl_int_t global_id;
  double pos[3];
  double normal[3];
  double meancurvature;
  double gaussiancurvature;
  double area;
} GhostUpdatePayload;

// A map from a neighbor rank (mftl_int_t) to a list of GIDs (int64_t*).
// We'll also need to store the count/capacity of that list.
typedef struct
{
  mftl_int_t count;
  mftl_int_t capacity;
  mftl_int_t *gids;
} MFTL_GidList;

// --- Global MPI Datatype Handles ---
// Use 'extern' to make these available to any file that includes this header.
// They are defined and created in communication.c.
extern MPI_Datatype MPI_PackedVertex;
extern MPI_Datatype MPI_PackedTriangle;
extern MPI_Datatype MPI_PackedHalfEdge;
extern MPI_Datatype MPI_PackedTriangleComplex;
extern MPI_Datatype MPI_GhostUpdatePayload;

/**
 * @brief Creates and commits the custom MPI datatypes.
 *
 * This function should be called once by every process after MPI_Init().
 */
void MFTLRegisterMpiTypes();

/**
 * @brief Frees the custom MPI datatypes.
 *
 * This function should be called once by every process before MPI_Finalize().
 */
void MFTLFreeMpiTypes();

/**
 * @brief Determines which process owns a given 3D coordinate.
 * @details This function implements a deterministic ownership rule for points on
 * processor domain boundaries. A point on a shared boundary is assigned
 * to the processor with the lowest rank that contains it. This prevents
 * ambiguity and ensures each point has a unique owner.
 * The primary rule is [min, max) for all dimensions. If a point is not
 * found, it must lie on a max boundary, so a second pass checks for
 * containment in the closed interval [min, max].
 *
 * MPI works with 32 bit int, so its safer to stay with int here.
 */
static inline mftl_int_t MFTLGetOwnerProc(const MFTL_Domain *domain, const double pos[3])
{
  // Pre-check: Ensure the point is within the global domain.
  assert(pos[0] >= domain->global_min[0] && pos[0] <= domain->global_max[0] && pos[1] >= domain->global_min[1] && pos[1] <= domain->global_max[1] &&
         pos[2] >= domain->global_min[2] && pos[2] <= domain->global_max[2] && "Point is outside the global simulation box");

  // 1. Math Optimization:
  // Use pre-calculated factors (multiplication is faster than division).
  // Use (int) cast instead of floor() (faster, safe because assertions ensure pos >= min).

  int i = (int) ((pos[0] - domain->global_min[0]) * domain->grid_factors[0]);
  int j = (int) ((pos[1] - domain->global_min[1]) * domain->grid_factors[1]);
  int k = (int) ((pos[2] - domain->global_min[2]) * domain->grid_factors[2]);

  // 2. Clamp Indices
  // (Handling the edge case where pos is exactly on the max boundary)
  if (i >= domain->cuts[0]) i = domain->cuts[0] - 1;
  if (j >= domain->cuts[1]) j = domain->cuts[1] - 1;
  if (k >= domain->cuts[2]) k = domain->cuts[2] - 1;

  // 3. Lookup Optimization
  // Direct array access instead of MPI_Cart_rank
  int linear_idx = i + (j * domain->cuts[0]) + (k * domain->cuts[0] * domain->cuts[1]);

  // No assert needed for owner_rank if initialization logic is correct
  return domain->rank_map[linear_idx];
}

/**
 * @brief Intelligently redistributes the mesh using the fastest available communication pattern.
 *
 * @details This function implements a robust, multi-phase redistribution algorithm.
 *
 * 1.  SETUP & ANALYSIS: It begins by querying the MPI graph communicator to get a
 * list of direct neighbors. It then iterates through all local triangles,
 * determines their correct owner, and checks if the owner is a direct neighbor.
 *
 * 2.  DECISION: Based on the analysis, a global consensus is reached using
 * MPI_Allreduce. If all data movement is between direct neighbors, the function
 * selects a highly-optimized "fast path". Otherwise, it falls back to a
 * robust "global path".
 *
 * 3.  COMMUNICATION:
 * - Fast Path: Uses MPI_Neighbor_alltoallv for efficient, localized data exchange.
 * - Global Path: Uses MPI_Alltoallv to handle arbitrary data exchange between any
 * two processes in the simulation.
 *
 * 4.  REBUILDING: Each rank unpacks the received data, combines it with the
 * triangles it kept, de-duplicates vertices by their unique global IDs,
 * and reconstructs its new local portion of the mesh.
 *
 * @param ensemble The MFTL ensemble containing the objects and MPI info.
 * @param base The geometric basis for determining ownership (VERTEX_BASED or CENTER_BASED).
 * @return 0 on success, -1 on failure.
 */
mftl_int_t MFTLRedistributeObject(MFTL_Object *object);
mftl_int_t MFTLRedistributeListOfObjects(MFTL_Ensemble *ensemble);

/**
 * @brief Synchronizes ghost vertex positions for a list of objects.
 * @param ensemble The ensemble containing the objects.
 * @param object_lookup Optional hash map (BodyID -> Index). Pass MFTL_NULL to use linear search fallback.
 * @return MFTL_SUCCESS on success.
 */
mftl_int_t MFTLUpdateGhostPositions(MFTL_Ensemble *ens, khash_t(object_map) * object_lookup);

/**
 * @brief Extends the ghost layer by a specified number of topological steps.
 * @param ens The ensemble.
 * @param steps Number of steps to extend (e.g. 1 step = V->T->V).
 * @return MFTL_SUCCESS on success.
 */
mftl_int_t MFTLExtendGhostLayer(MFTL_Ensemble *ens, int steps);

#endif