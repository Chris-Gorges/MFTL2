/**
 * @file volume.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief Reference volumes and the Newton volume correction.
 *
 * The correction is the one the topology change applies to its clusters, per
 * body and callable on its own: a uniform offset d along the field u of
 * MFTLBodyVolumeDisplacement(), with d from Newton's method on the EXACT first
 * variation of the volume of that field, sum_t S_t . mean(u). The region a body's
 * volume measures, and the axis and reference it is measured along, are
 * MFTLEnsembleBodyIntegrals()'s, decided once per call from reduced data; each
 * step then needs one per-body reduction of two values (V and dV/dd) and one
 * ghost position exchange. The step count is fixed, so every rank enters the
 * same collectives.
 */

#include "correction_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/** @cond INTERNAL */

MFTL_VolumeCorrectionParameters MFTLVolumeCorrectionDefaultParameters(void)
{
  MFTL_VolumeCorrectionParameters p;
  p.iterations = 3;
  p.rim_project = MFTL_NULL;
  p.rim_project_user = MFTL_NULL;
  return p;
}

int MFTLEnsembleSetReferenceVolumes(MFTL_Ensemble *ens)
{
  if (ens == MFTL_NULL) return MFTL_ERROR;
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;
  const int b = ens->n_local_objects;
  MFTL_BodyIntegrals *const bi = (MFTL_BodyIntegrals *) MFTLMallocOrDie((size_t) MFTL_MAX(b, 1) * sizeof(MFTL_BodyIntegrals), "reference volumes");
  MFTLUpdateGhostPositions(ens);
  const int rc = MFTLEnsembleBodyIntegrals(ens, bi);
  for (int o = 0; o < b && rc == MFTL_SUCCESS; ++o)
    ens->local_objects[o]->reference_volume = (bi[o].volume_status == MFTL_INTEGRAL_DEFINED) ? bi[o].volume : NAN;
  free(bi);
  return rc;
}

int MFTLEnsembleCorrectVolumes(MFTL_Ensemble *ens, const MFTL_VolumeCorrectionParameters *p, MFTL_VolumeCorrectionStats *stats)
{
  const MFTL_VolumeCorrectionParameters def = MFTLVolumeCorrectionDefaultParameters();
  if (p == MFTL_NULL) p = &def;
  MFTL_VolumeCorrectionStats sink;
  MFTL_VolumeCorrectionStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  if (ens == MFTL_NULL || p->iterations < 0) return MFTL_ERROR;
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  MFTL_Domain *const dom = &ens->domain;
  const double t0 = MFTLCorrectionTime();
  const int b = ens->n_local_objects;
  const int n_iter = p->iterations;

  MFTLUpdateGhostPositions(ens);
  MFTL_BodyIntegrals *const bi = (MFTL_BodyIntegrals *) MFTLMallocOrDie((size_t) MFTL_MAX(b, 1) * sizeof(MFTL_BodyIntegrals), "volume correction: integrals");
  MFTLEnsembleBodyIntegrals(ens, bi);

  double *const rec = (double *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(b, 1) * sizeof(double), "volume correction: records");
  double *const offset = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(double), "volume correction: offsets");
  double **const area0 = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(double *), "volume correction: area snapshots");
  bool *const active = (bool *) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(bool), "volume correction: active bodies");
  double stat[4] = {0.0, 0.0, 0.0, 0.0};  // SUM: bodies, unmeasured; MAX: error before, after
  double max_off = 0.0;

  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    const bool has_ref = isfinite(obj->reference_volume);
    active[o] = has_ref && bi[o].volume_status == MFTL_INTEGRAL_DEFINED && obj->reference_volume != 0.0;
    if (bi[o].lowest_holder == dom->rank)
    {
      if (active[o]) stat[0] += 1.0;
      if (has_ref && !active[o]) stat[1] += 1.0;
    }
    if (active[o])
    {
      if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
      area0[o] = MFTLCorrectionSnapshotAreas(obj);
    }
  }

  const MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};
  for (int iter = 0; iter <= n_iter; ++iter)
  {
    for (int o = 0; o < b; ++o)
    {
      rec[2 * o] = rec[2 * o + 1] = 0.0;
      if (!active[o]) continue;
      MFTL_Object *const obj = ens->local_objects[o];
      /* Only centre[axis] is read, and only on a periodic axis -- where the
       * reference IS the body's centre (a closed body). */
      const double centre[3] = {bi[o].reference, bi[o].reference, bi[o].reference};
      double v = 0.0, a = 0.0;
      MFTLBodyFluxVolume(obj, bi[o].axis, bi[o].reference, centre, &v, &a);
      rec[2 * o] = v;
      if (iter < n_iter) rec[2 * o + 1] = MFTLBodyVolumeDisplacement(obj);
    }
    MFTLEnsembleReduceBodies(ens, rec, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

    for (int o = 0; o < b; ++o)
    {
      if (!active[o]) continue;
      const double ref = ens->local_objects[o]->reference_volume;
      const double err = fabs(rec[2 * o] / ref - 1.0);
      if (iter == 0 && err > stat[2]) stat[2] = err;
      if (iter == n_iter && err > stat[3]) stat[3] = err;
    }
    if (iter == n_iter) break;

    for (int o = 0; o < b; ++o)
    {
      if (!active[o]) continue;
      const double dvdd = rec[2 * o + 1];
      if (!(dvdd != 0.0) || !isfinite(dvdd)) continue;
      MFTL_Object *const obj = ens->local_objects[o];
      const double d = (obj->reference_volume - rec[2 * o]) / dvdd;
      if (!isfinite(d)) continue;
      offset[o] += d;
      for (int v = 0; v < obj->verts.n_local; ++v)
      {
        double *const q = &obj->verts.pos[3 * v];
        for (int k = 0; k < 3; ++k) q[k] += d * obj->verts.normal[3 * v + k];
        MFTLApplyPeriodicity(dom, q);
        /* A rim vertex slid within its wall; back onto the application's
         * constraint. The next step measures what the projection did. */
        if (p->rim_project && (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY))
        {
          double r[3] = {q[0], q[1], q[2]};
          if (p->rim_project(dom, r, p->rim_project_user) == MFTL_SUCCESS)
          {
            MFTLApplyPeriodicity(dom, r);
            for (int k = 0; k < 3; ++k) q[k] = r[k];
          }
        }
      }
    }
    MFTLUpdateGhostPositions(ens);
  }

  for (int o = 0; o < b; ++o)
  {
    if (!active[o]) continue;
    MFTL_Object *const obj = ens->local_objects[o];
    MFTLCorrectionRescaleDensities(obj, area0[o]);
    free(area0[o]);
    obj->derived &= ~MFTL_DERIVED_GEOMETRY;
    MFTLObjectMarkChanged(obj);
    if (fabs(offset[o]) > max_off) max_off = fabs(offset[o]);
  }

  MFTLCorrectionAllreduceDouble(dom, stat, 2, 2);
  double mx[4] = {stat[2], stat[3], max_off, MFTLCorrectionTime() - t0};
  MFTLCorrectionAllreduceDouble(dom, mx, 4, 1);
  st->n_bodies = (int) stat[0];
  st->n_bodies_unmeasured = (int) stat[1];
  st->max_error_before = mx[0];
  st->max_error_after = mx[1];
  st->max_offset = mx[2];
  st->t_total = mx[3];

  free(active);
  free(area0);
  free(offset);
  free(rec);
  free(bi);
  return MFTL_SUCCESS;
}

/** @endcond */
