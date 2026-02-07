/*
 * @Author: Christian Gorges
 * @Date: 2025-08-10 10:00:56
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-06 15:43:58
 */

#ifndef MFTL_REMESHING_H
#define MFTL_REMESHING_H

#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"

typedef enum
{
  MFTL_OP_NONE = 0,
  MFTL_OP_SPLIT,
  MFTL_OP_COLLAPSE,
  MFTL_OP_SWAP
} MFTL_RemeshOpType;

// The "Command" packet sent from Owner to Neighbor
typedef struct
{
  int8_t type;  // MFTL_RemeshOpType
  mftl_int_t edge_gid;  // The global ID of the edge being operated on

  // Payload for Split:
  double new_pos[3];  // Exact position of the new vertex
  mftl_int_t new_vert_gid;  // Pre-assigned GID for the new vertex (deterministic!)
  mftl_int_t new_edge_gid;  // Pre-assigned GID for the new halfedge

  // Payload for Collapse:
  mftl_int_t keep_vert_gid;  // The GID of the vertex to keep

} MFTL_RemeshAction;

// Dynamic array (vector) helper for buffering actions
typedef struct
{
  MFTL_RemeshAction *data;
  size_t count;
  size_t capacity;
} ActionBuffer;

// // Helper to determine edge ownership deterministically
// // Rule: The rank with the lower ID owns the shared edge.
// static inline bool MFTLIsEdgeOwner(int my_rank, int neighbor_rank) { return my_rank < neighbor_rank; }

// --- Flags for internal state ---
#define MFTL_FLAG_DEAD (1 << 0)  // Element is deleted
#define MFTL_FLAG_MODIFIED (1 << 1)  // Element was touched (for next pass)

// --- Remeshing Parameters ---
typedef struct MFTL_RemeshParameters
{
  double min_edge_length;
  double max_edge_length;
  double min_edge_length_sq;
  double max_edge_length_sq;
  double target_length;
  double aspect_ratio_tolerance;  // For collapse/swap
} MFTL_RemeshParameters;

// --- Core Topological Operators ---
// These return the index of the central vertex (split) or remaining vertex (collapse)
// or MFTL_ERROR on failure.

mftl_int_t MFTLSplitEdge(MFTL_Object *obj, mftl_int_t he_ab_idx);
mftl_int_t MFTLCollapseEdge(MFTL_Object *obj, mftl_int_t he_idx);
mftl_int_t MFTLSwapEdge(MFTL_Object *obj, mftl_int_t he_idx);

// --- High Level Routine ---
void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

#endif