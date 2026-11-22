/**
 * @file ensemble_integrals.c
 * @author Christian Gorges
 * @date 2026-10-03
 * @ingroup mftl_ensemble
 *
 * @brief Area, volume and centroid of every body: two per-body reductions over
 *        the holders around the local flux integrals of src/geometry/.
 *
 * Pass 1 decides HOW a body is measured -- its extent on every axis, which walls
 * its rims lie on, its total area vector -- and pass 2 measures it. Both are
 * MFTLEnsembleReduceBodies() calls, so every holder of a body takes the same
 * decision from the same reduced record and gets the same bits.
 */

#include "mftl_ensemble.h"

#include <math.h>
#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_types.h"

/**
 * @name Pass-1 record layout
 * @{
 */
#define MFTL_INT_EXT 0  ///< #MFTL_PERIODIC_EXTENT_SLOTS MIN: the extent.
#define MFTL_INT_WALLS (MFTL_INT_EXT + MFTL_PERIODIC_EXTENT_SLOTS)  ///< 6 MAX: a rim lies on wall 2k + side.
#define MFTL_INT_RIM (MFTL_INT_WALLS + 6)  ///< MAX: the body has a rim.
#define MFTL_INT_OFF (MFTL_INT_RIM + 1)  ///< MAX: a rim edge lies on no wall.
#define MFTL_INT_S (MFTL_INT_OFF + 1)  ///< 3 SUM: the total area vector.
#define MFTL_INT_AREA (MFTL_INT_S + 3)  ///< SUM: the area.
#define MFTL_INT_RANK (MFTL_INT_AREA + 1)  ///< MIN: the lowest holder.
#define MFTL_INT_P1 (MFTL_INT_RANK + 1)  ///< Slots per body.
/** @} */

/**
 * @brief A closed body ENCLOSES a region exactly when its area vector vanishes;
 *        a closed sheet across periodic axes has one of the size of the cross
 *        section. Relative to the area, so it means the same at any scale.
 */
#define MFTL_INT_ENCLOSED_TOL 1e-8

int MFTLEnsembleBodyIntegrals(MFTL_Ensemble *ens, MFTL_BodyIntegrals *out)
{
  if (ens == MFTL_NULL) return MFTL_ERROR;
  const int b = ens->n_local_objects;
  if (b > 0 && out == MFTL_NULL) return MFTL_ERROR;
  if (ens->n_bodies_global < 1) return MFTL_SUCCESS;  // agreed on every rank: enters no collective

  const MFTL_Domain *const dom = &ens->domain;

  /* ── Pass 1: how to measure ──────────────────────────────────────────────── */
  MFTL_BodyReduceOp ops1[MFTL_INT_P1];
  for (int j = 0; j < MFTL_INT_P1; ++j) ops1[j] = MFTL_BODY_REDUCE_MIN;
  for (int j = 0; j < 8; ++j) ops1[MFTL_INT_WALLS + j] = MFTL_BODY_REDUCE_MAX;
  for (int j = 0; j < 4; ++j) ops1[MFTL_INT_S + j] = MFTL_BODY_REDUCE_SUM;

  double *const r1 = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * MFTL_INT_P1, 1) * sizeof(double), "body integrals: pass 1");
  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    double *const r = &r1[(size_t) MFTL_INT_P1 * o];
    MFTLPeriodicExtentInit(&r[MFTL_INT_EXT]);
    bool off = MFTL_FALSE;
    int n_rim = 0;
    const int walls = MFTLBodyRimWalls(obj, &off, &n_rim);
    for (int w = 0; w < 6; ++w) r[MFTL_INT_WALLS + w] = (walls & (1 << w)) ? 1.0 : 0.0;
    r[MFTL_INT_RIM] = (n_rim > 0) ? 1.0 : 0.0;
    r[MFTL_INT_OFF] = off ? 1.0 : 0.0;
    /* The extent and the area vector from ONE gather of each triangle's corners. */
    for (int j = 0; j < 4; ++j) r[MFTL_INT_S + j] = 0.0;  // the three components and the area
    MFTLBodyShape(obj, &r[MFTL_INT_EXT], &r[MFTL_INT_S], &r[MFTL_INT_AREA]);
    r[MFTL_INT_RANK] = (double) dom->rank;
  }
  MFTLEnsembleReduceBodies(ens, r1, MFTL_INT_P1, ops1, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* ── The decision, identical on every holder ─────────────────────────────── */
  double *const centre = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(3 * b, 1) * sizeof(double), "body integrals: centres");
  for (int o = 0; o < b; ++o)
  {
    const double *const r = &r1[(size_t) MFTL_INT_P1 * o];
    MFTL_BodyIntegrals *const bi = &out[o];
    MFTL_PeriodicExtent x;
    MFTLPeriodicExtentResolve(dom, &r[MFTL_INT_EXT], &x);
    for (int k = 0; k < 3; ++k) centre[3 * o + k] = x.centre[k];

    bi->area = r[MFTL_INT_AREA];
    bi->volume = 0.0;
    bi->axis = -1;
    bi->reference = 0.0;
    bi->closed = !(r[MFTL_INT_RIM] > 0.0);
    bi->lowest_holder = (int) r[MFTL_INT_RANK];
    for (int k = 0; k < 3; ++k) bi->centroid[k] = 0.0;
    bi->volume_status = MFTL_INTEGRAL_UNSUPPORTED;

    const double *const S = &r[MFTL_INT_S];
    const double s_mag = MFTLVectorNorm(S);
    const bool encloses = bi->closed && s_mag <= MFTL_INT_ENCLOSED_TOL * bi->area;

    if (!bi->closed && r[MFTL_INT_OFF] > 0.0)
    {
      /* A rim off the walls: no wall closes the surface, there is no region. */
      bi->volume_status = MFTL_INTEGRAL_UNDEFINED;
    }
    else
    {
      int walls = 0;
      for (int w = 0; w < 6; ++w)
        if (r[MFTL_INT_WALLS + w] > 0.0) walls |= 1 << w;
      bi->axis = MFTLBodyFluxAxis(dom, walls, x.covers, x.centre, encloses ? MFTL_NULL : S, &bi->reference);
    }
    if (bi->axis >= 0) bi->volume_status = MFTL_INTEGRAL_DEFINED;
    for (int k = 0; k < 3; ++k)
    {
      bi->centroid_status[k] = bi->volume_status;
      if (bi->volume_status == MFTL_INTEGRAL_DEFINED && dom->is_periodic[k] && x.covers[k]) bi->centroid_status[k] = MFTL_INTEGRAL_UNDEFINED;
    }
  }
  free(r1);

  /* ── Pass 2: the flux integrals ──────────────────────────────────────────── */
  MFTL_BodyReduceOp ops2[MFTL_BODY_MOMENT_SLOTS];
  for (int j = 0; j < MFTL_BODY_MOMENT_SLOTS; ++j) ops2[j] = MFTL_BODY_REDUCE_SUM;
  double *const r2 = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b * MFTL_BODY_MOMENT_SLOTS, 1), sizeof(double), "body integrals: pass 2");
  for (int o = 0; o < b; ++o)
  {
    if (out[o].axis < 0) continue;  // an identity row: zeros
    MFTLBodyFluxMoments(ens->local_objects[o], out[o].axis, out[o].reference, &centre[3 * o], &r2[(size_t) MFTL_BODY_MOMENT_SLOTS * o]);
  }
  MFTLEnsembleReduceBodies(ens, r2, MFTL_BODY_MOMENT_SLOTS, ops2, MFTL_BODY_REDUCE_SCOPE_AUTO);

  for (int o = 0; o < b; ++o)
  {
    MFTL_BodyIntegrals *const bi = &out[o];
    if (bi->axis < 0) continue;
    const double *const q = &r2[(size_t) MFTL_BODY_MOMENT_SLOTS * o];
    const int k = bi->axis;
    bi->volume = q[0];
    if (!(bi->volume != 0.0))
    {
      for (int j = 0; j < 3; ++j)
        if (bi->centroid_status[j] == MFTL_INTEGRAL_DEFINED) bi->centroid_status[j] = MFTL_INTEGRAL_UNDEFINED;
      continue;
    }
    for (int j = 0; j < 3; ++j)
    {
      if (bi->centroid_status[j] != MFTL_INTEGRAL_DEFINED) continue;
      bi->centroid[j] = (j == k) ? bi->reference + q[1 + k] / (2.0 * bi->volume) : centre[3 * o + j] + q[1 + j] / bi->volume;
    }
    /* A centroid is a position: wrapped into the box on a periodic axis. */
    for (int j = 0; j < 3; ++j)
    {
      if (!dom->is_periodic[j] || bi->centroid_status[j] != MFTL_INTEGRAL_DEFINED) continue;
      bi->centroid[j] = MFTLPeriodicWrapCoordinate(bi->centroid[j], dom->global_min[j], dom->domain_size[j]);
    }
  }

  free(r2);
  free(centre);
  return MFTL_SUCCESS;
}
