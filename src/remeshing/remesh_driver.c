/*
 * MFTL2 remeshing — the serial remesh driver MFTLRemeshEnsemble (split / collapse / flip /
 * smoothing passes, compaction, optional volume-conserving smoothing) and the class-guarded
 * volume recovery of the drivers (plans/16 §4.4.4).
 */

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_remeshing.h"
#include "mftl_images.h"
#include "mftl_boundary_internal.h"
#include "mftl_phase_set.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

/* ------------------------------------------------------------------------------------------
 * Class-guarded volume recovery of the remesh drivers (plans/16 §4.4.4): the body's own
 * volume for a closed compact body, the singleton phase set's volume when that is DEFINED,
 * otherwise no recovery (counted in stat_recovery_skipped).
 * ------------------------------------------------------------------------------------------ */
void MFTLRemeshRecoveryTarget(MFTL_Object *obj, MFTL_RemeshParameters *params, MFTL_RecoveryTarget *t)
{
  t->kind = 0;
  t->target = 0.0;
  MFTLObjectEnsureWinding(obj);  // collective; unknown winding never counts as compact
  if (MFTLObjectHasVolume(obj))
  {
    // closed and compact: the body's own volume (bit-identical to the closed-surface path)
    t->kind = 1;
    t->target = MFTLObjectVolume(obj);
    return;
  }
  if (params->skip_volume_recovery) return;
  // open or wrapping: the singleton phase set decides (never MFTLObjectVolume)
  MFTL_PhaseSet *S = MFTLPhaseSetCreate(obj->domain);
  MFTLPhaseSetAddMember(S, obj, 1);
  double V = 0.0;
  if (MFTLPhaseSetVolume(S, &V) == MFTL_DEFINED)
  {
    t->kind = 2;
    t->target = V;
  }
  else
    params->stat_recovery_skipped++;
  MFTLPhaseSetDestroy(&S);
}

void MFTLRemeshRecover(MFTL_Ensemble *ens, MFTL_Object *obj, const MFTL_RecoveryTarget *t)
{
  if (t->kind == 1)
    MFTLVolumeRecovery(obj, t->target);
  else if (t->kind == 2)
  {
    MFTL_PhaseSet *S = MFTLPhaseSetCreate(obj->domain);
    MFTLPhaseSetAddMember(S, obj, 1);
    MFTLPhaseSetVolumeRecovery(S, ens, t->target, MFTL_NULL);
    MFTLPhaseSetDestroy(&S);
  }
}

/* Serial remesh driver (rank-local kernel; the collective parts are the recovery target, the
   convergence decision and the ghost refresh). Per iteration and object: triangle properties,
   feature detection, normals / curvatures when needed, then splits (descending half-edge
   index), collapses, flips and tangential smoothing — or MFTL1's sweeps for closed bodies with
   mftl1_criteria. Stops after `iterations` passes, or (iterations == MFTL_ERROR) when fewer
   than 10 operations happened, at most 16 passes. Then compaction, the optional
   volume-conserving smoothing stages and the class-guarded volume recovery. */
void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  int iter = 0;

  bool collapses_occurred[ens->num_objects];
  // entity counts before the pass (MFTLFinalizeRemeshing)
  int start_n_vertices[ens->num_objects];
  int start_n_triangles[ens->num_objects];
  int start_n_halfedges[ens->num_objects];
  MFTL_RecoveryTarget start_volume[ens->num_objects];

  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];

    collapses_occurred[i] = MFTL_FALSE;
    start_n_vertices[i] = obj->n_vertices;
    start_n_triangles[i] = obj->n_triangles;
    start_n_halfedges[i] = obj->n_half_edges;
    // recovery target (collective) with the class guard: the body's own volume only for closed
    // compact bodies, a set-level target for DEFINED singleton phase sets, else none
    MFTLRemeshRecoveryTarget(obj, params, &start_volume[i]);
  }

  // iterations == MFTL_ERROR: until convergence; otherwise a fixed maximum
  while (params->iterations == MFTL_ERROR || iter < params->iterations)
  {
    if (!params->quiet) printf("--- Iteration %d ---\n", iter + 1);

    int total_split = 0;
    int total_collapse = 0;
    int total_swap = 0;

    for (int i = 0; i < ens->num_objects; i++)
    {
      MFTL_Object *obj = ens->objects[i];

      int obj_split = 0;
      int obj_collapse = 0;
      int obj_swap = 0;

      // Refresh geometry for the criteria, fitting and smoothing, and re-detect features EVERY
      // iteration: the ops change the geometry and feature protection must track it without the
      // caller pre-setting flags. Detection consumes triangle normals, so it runs after the
      // property refresh.
      MFTLComputeTriangleProperties(obj);
      MFTLDetectFeatures(obj, params->feature_angle_threshold);
      if (params->vertex_fit != MFTL_FIT_NONE || params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE ||
          params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        MFTLComputeVertexNormals(obj);
        if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE || params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
          MFTLComputeVertexCurvatures(obj);
      }

      // the MFTL1-parity path only for closed bodies in the default mode (plans/16 §4.4.4)
      const int parity = params->mftl1_criteria && !MFTLObjectRemeshGeneralized(obj);
      if (parity)
      {
        // MFTL1's sweeps (MFTL_Refining / MFTL_Coarsening / MFTL_Flipping)
        MFTL1Sweeps(obj, params, &obj_split, &obj_collapse, &obj_swap);
        if (obj_collapse > 0) collapses_occurred[i] = MFTL_TRUE;
      }

      // splits
      for (int j = obj->n_half_edges - 1; j >= 0 && !params->skip_split && !parity; j--)
      {
        if (MFTLSplitApproved(obj, j, params))
        {
          MFTLSplitEdgeWithFit(obj, j, params);
          obj_split++;
        }
      }

      // collapses
      for (int j = 0; j < obj->n_half_edges && !params->skip_collapse && !parity; ++j)
      {
        if (CollapseApproved(obj, j, params))
        {
          MFTLCollapseEdgeWithFit(obj, j, params);
          collapses_occurred[i] = MFTL_TRUE;
          obj_collapse++;
        }
      }

      // flips (valence optimisation + quality/dihedral gates); interior, non-feature edges, each
      // edge once
      for (int j = 0; j < obj->n_half_edges && !params->skip_flip && !parity; ++j)
      {
        MFTL_HalfEdge *he = &obj->half_edges[j];

        if (he->flags & MFTL_FLAG_DEAD) continue;
        if (he->twin_idx == MFTL_ERROR) continue;
        if (j > he->twin_idx) continue;
        if (he->flags & MFTL_FLAG_FEATURE) continue;

        if (FlipApprovedWithParams(obj, j, params))
        {
          MFTLFlipEdge(obj, j);
          obj_swap++;
        }
      }

      // tangential smoothing (mode / iterations from params)
      if (!params->skip_tangential_smoothing) MFTLSmoothTangentialWithParams(obj, params);

      for (int j = 0; j < obj->n_half_edges; ++j)
      {
        obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
      }

      for (int j = 0; j < obj->n_vertices; ++j)
      {
        obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;
      }

      params->stat_splits += obj_split;
      params->stat_collapses += obj_collapse;
      params->stat_flips += obj_swap;
      total_split += obj_split;
      total_collapse += obj_collapse;
      total_swap += obj_swap;
    }

    if (!params->quiet) printf("Operations -> Split: %d | Collapsed: %d | Swapped: %d\n", total_split, total_collapse, total_swap);

    // Convergence. On a single rank (or a ghost-complete mesh) the local op count is the
    // whole story. When the serial kernel runs on a distributed ensemble
    // (MFTLRemeshEnsembleDispersedCentering gathers to rank 0; other ranks
    // hold empty objects), the decision must be collective — otherwise the
    // empty ranks would exit the loop early and desynchronize the
    // collectives in the smoothing / recovery stages below.
    int converged = !params->fixed_iterations && (total_split + total_collapse + total_swap) < 10;
    if (ens->domain.nproc > 1)
    {
      int converged_g = 0;
      MPI_Allreduce(&converged, &converged_g, 1, MPI_INT, MPI_MIN, ens->domain.comm_cart);
      converged = converged_g;
    }
    if (converged)
    {
      if (!params->quiet) printf("Remeshing converged successfully after %d iterations.\n", iter + 1);
      break;
    }

    // run-to-convergence mode stops after 16 passes
    if (iter >= 15 && params->iterations == MFTL_ERROR)
    {
      if (!params->quiet) printf("Warning: Remeshing reached failsafe limit (15 iterations). Exiting to prevent hang.\n");
      break;
    }

    iter++;
  }

  // compaction
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    MFTLFinalizeRemeshing(obj, start_n_vertices[i], start_n_triangles[i], start_n_halfedges[i], collapses_occurred[i]);
  }

  // Volume-conserving post-remesh stages on the finalised mesh. The serial edge-local smoothers
  // need every processed fan stored (a rank-local mesh); on a distributed mesh they fail loudly
  // (the parallel driver uses the *Parallel variants).
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    MFTLComputeTriangleProperties(obj);

    if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_KUPRAT)
      MFTLVolumeConservingSmoothing(obj, params);
    else if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_TSUR)
      MFTLTSUR3D(obj, params);

    if (params->roughness_smoothing) MFTLRoughnessSmoothing(obj, params);
  }

  // ALWAYS finish a remesh cycle with global volume recovery against the
  // pre-remesh volume: per-edge local conservation is first-order only and
  // drift accumulates across the split/collapse/flip/smooth pipeline.
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    if (!params->skip_volume_recovery) MFTLRemeshRecover(ens, obj, &start_volume[i]);
    MFTLComputeTriangleProperties(obj);
  }
  MFTLUpdateGhostPositions(ens, NULL);
}
