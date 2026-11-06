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
 *   correction_comm.c    EVERY MPI call of the module, and nothing else
 *                        (`test_correction_mpi_is_confined`).
 *   correction_fields.c  the amount-per-triangle rule for conservative densities
 *   undulation.c         TSUR3D: the edge operation, the local sweep and the
 *                        distributed schedule
 *   volume.c             reference volumes and the Newton correction
 *
 * The module calls MFTLUpdateGhostPositions() and MFTLEnsembleReduceBodies(),
 * which are other modules' collectives; it makes no MPI call of its own
 * outside correction_comm.c.
 */

#ifndef MFTL_CORRECTION_INTERNAL_H
#define MFTL_CORRECTION_INTERNAL_H

#include "mftl_correction.h"

/** @cond INTERNAL */

/* ── correction_comm.c ─────────────────────────────────────────────────────── */

/* Wall clock, seconds. */
double MFTLCorrectionTime(void);
/* In-place Allreduce over comm_world; op 0 MIN, 1 MAX, 2 SUM. */
void MFTLCorrectionAllreduceDouble(const MFTL_Domain *dom, double *v, int n, int op);
void MFTLCorrectionAllreduceLongLong(const MFTL_Domain *dom, long long *v, int n, int op);
/* This rank's cell in the process grid. */
void MFTLCorrectionCoords(const MFTL_Domain *dom, int coords[3]);

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
