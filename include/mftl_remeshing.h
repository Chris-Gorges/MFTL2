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

#endif