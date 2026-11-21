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
 *   correction_fields.c  the amount-per-triangle rule for conservative densities
 *   undulation.c         TSUR3D: the edge operation, the local sweep and the
 *                        distributed schedule
 *   volume.c             reference volumes and the Newton correction
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

/* ── correction_fields.c ───────────────────────────────────────────────────── */

/* The area of every triangle of obj, or NULL when obj carries no conservative
 * density (then MFTLCorrectionRescaleDensities() does nothing). Free with free(). */
double *MFTLCorrectionSnapshotAreas(const MFTL_Object *obj);
/* Density *= A_before / A_now on every triangle whose area changed, owned and
 * ghost alike: the amount per triangle is what a normal motion conserves, and
 * the rescale is path independent and identical on every holder. */
void MFTLCorrectionRescaleDensities(MFTL_Object *obj, const double *area_before);

/** @endcond */

#endif  // MFTL_CORRECTION_INTERNAL_H
