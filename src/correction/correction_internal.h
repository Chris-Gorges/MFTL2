/**
 * @file correction_internal.h
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief Internal interfaces of the front-correction module.
 *
 * ## THE FILES AND WHAT EACH MAY DO
 *
 *   undulation.c         TSUR3D: the edge operation, the local sweep and the
 *                        distributed schedule
 *   volume.c             reference volumes and the Newton correction, whose core
 *                        (MFTLCorrectionNewtonVolumes()) the topology change runs too
 *
 * The module makes NO MPI call of its own (`test_correction_mpi_is_confined`).
 * Per-vertex data travels through MFTLUpdateGhostPositions(), per-body values
 * through MFTLEnsembleReduceBodies(), and the statistics and the colour flags
 * of the undulation schedule through the library's small reductions
 * (MFTLAllreduceDoubles(), MFTLAllreduceLongLongs()).
 */

#ifndef MFTL_CORRECTION_INTERNAL_H
#define MFTL_CORRECTION_INTERNAL_H

#include "mftl_correction.h"

/** @cond INTERNAL */

/* ── volume.c ─────────────────────────────────────────────────────────────── */

/* The Newton volume correction over GROUPS of local bodies, the one both
 * MFTLEnsembleCorrectVolumes() (a group per body) and the topology change (a
 * group per cluster) run. The topology change includes this header for it. */
typedef struct
{
  int n_groups;  // groups; with per_body the local object count
  /* true: group g IS local slot g, reduced over that body's holders
   * (MFTLEnsembleReduceBodies()); false: an agreed list, the same on every
   * rank, reduced over comm_world. */
  bool per_body;
  const int *group;  // [n_local_objects] the group of local slot o, or -1: neither measured nor moved
  const int *axis;  // [n_groups] flux axis of MFTLBodyFluxVolume()
  const double *reference;  // [n_groups] its reference
  const double *centre;  // [3 n_groups] its centre (read on periodic axes)
  const double *target;  // [n_groups] the volume to restore
  int iterations;  // Newton steps, fixed: every rank enters the same collectives
  MFTL_RimProjector rim_project;  // applied to an owned rim vertex after every step; may be MFTL_NULL
  void *rim_project_user;
} MFTL_VolumeNewton;

/* Collective (every rank, even holding nothing). Ghost positions current on
 * entry and on return. volume[2g] / volume[2g+1]: the group's reduced volume
 * before the first step / after the last, the same on every rank holding a body
 * of it (every rank without per_body); offset[g] the summed offset. Moved bodies
 * keep each conservative density's amount per triangle and are marked changed. */
void MFTLCorrectionNewtonVolumes(MFTL_Ensemble *ens, const MFTL_VolumeNewton *nw, double *volume, double *offset);

/** @endcond */

#endif  // MFTL_CORRECTION_INTERNAL_H
