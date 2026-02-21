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

typedef enum
{
  MFTL_REMESH_ISOTROPIC,  // Botsch & Kobbelt (4/5 and 4/3 limits)
  MFTL_REMESH_ANISOTROPIC,  // Metric-driven remeshing
  MFTL_REMESH_CUSTOM  // Manual user-defined limits
} MFTL_RemeshType;

typedef enum
{
  MFTL_SMOOTH_VERTEX_BASED,
  MFTL_SMOOTH_TRIANGLE_BASED
} MFTL_SmoothMode;

typedef struct
{
  MFTL_RemeshType type;
  MFTL_SmoothMode smooth_mode;
  int iterations;

  // Length targets
  double target_length;
  double min_edge_length;
  double max_edge_length;
  double min_edge_length_sq;
  double max_edge_length_sq;

  double feature_angle_threshold;

  // ... any other parameters like smoothing weights, boundary locks, etc.
} MFTL_RemeshParameters;

/**
 * @brief Initializes remeshing parameters based on standard literature values.
 */
static inline void MFTLInitRemeshParams(MFTL_RemeshParameters *params, MFTL_RemeshType type, double target_length, int iterations, MFTL_SmoothMode smooth_mode,
                                        double feature_angle_threshold)
{
  params->type = type;
  params->target_length = target_length;
  params->iterations = iterations;
  params->smooth_mode = smooth_mode;
  params->feature_angle_threshold = feature_angle_threshold;

  switch (type)
  {
    case MFTL_REMESH_ISOTROPIC:
      // Standard Botsch & Kobbelt isotropic limits
      params->min_edge_length = (4.0 / 5.0) * target_length;
      params->max_edge_length = (4.0 / 3.0) * target_length;
      break;

    case MFTL_REMESH_ANISOTROPIC:
      // For metric-driven anisotropic remeshing, lengths are evaluated
      // in metric space where the target length is always exactly 1.0.
      // We set the physical target_length to 1.0 to reflect metric space.
      params->target_length = 1.0;
      params->min_edge_length = (4.0 / 5.0);
      params->max_edge_length = (4.0 / 3.0);
      // Note: Your distance functions will need to be updated to compute
      // metric distance instead of Euclidean distance if type == ANISOTROPIC.
      break;

    case MFTL_REMESH_CUSTOM:
      // Do nothing. The user must manually set min/max.
      // (You could set defaults here if you wanted).
      params->min_edge_length = target_length;
      params->max_edge_length = target_length;
      break;
  }

  // Pre-calculate squared values for fast distance checks
  params->min_edge_length_sq = params->min_edge_length * params->min_edge_length;
  params->max_edge_length_sq = params->max_edge_length * params->max_edge_length;
}

// --- Core Topological Operators ---
// These return the index of the central vertex (split) or remaining vertex (collapse)
// or MFTL_ERROR on failure.

mftl_int_t MFTLSplitEdge(MFTL_Object *obj, mftl_int_t he_ab_idx);
void MFTLCollapseEdge(MFTL_Object *obj, mftl_int_t he_ab_idx);
mftl_int_t MFTLSwapEdge(MFTL_Object *obj, mftl_int_t he_idx);

// --- High Level Routine ---
void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

#endif