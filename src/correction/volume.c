/**
 * @file volume.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief Reference volumes and the Newton volume correction.
 *
 * ONE correction, MFTLCorrectionNewtonVolumes(), over groups of bodies: a uniform
 * offset d per group along the field u of MFTLBodyVolumeDisplacement(), with d
 * from Newton's method on the EXACT first variation of the volume of that field,
 * sum_t S_t . mean(u). MFTLEnsembleCorrectVolumes() runs it with one group per
 * body, reduced over the body's holders, against its reference volume; the
 * topology change with one group per cluster, reduced over `comm_world`, against
 * the cluster's volume before the reconstruction. The region a group's volume
 * measures is the caller's, decided from reduced data. Each step needs one
 * reduction of two values per group (V and dV/dd) and one ghost position
 * exchange; the step count is fixed, so every rank enters the same collectives.
 */

#include "correction_internal.h"

#include "../object/fields_internal.h"

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

/* ═══════════════════════════════════════════════════════════════════════════
 * The Newton core: groups of bodies, one offset per group
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Moves every OWNED vertex of @p obj by @p d along the displacement field
 *        MFTLBodyVolumeDisplacement() left in `verts.normal`, wraps it, and puts
 *        a rim vertex back onto the application's constraint.
 *
 * A rim vertex slid within its wall; the next step measures what the projection
 * did. Ghosts are the caller's MFTLUpdateGhostPositions().
 */
static void displace_owned(MFTL_Object *obj, double d, const MFTL_VolumeNewton *nw)
{
  const MFTL_Domain *const dom = obj->domain;
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    double *const q = &obj->verts.pos[3 * v];
    for (int k = 0; k < 3; ++k) q[k] += d * obj->verts.normal[3 * v + k];
    MFTLApplyPeriodicity(dom, q);
    if (nw->rim_project != MFTL_NULL && (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY)) MFTLApplyRimProjector(nw->rim_project, nw->rim_project_user, dom, q);
  }
}

void MFTLCorrectionNewtonVolumes(MFTL_Ensemble *ens, const MFTL_VolumeNewton *nw, double *volume, double *offset)
{
  const int b = ens->n_local_objects;
  const int ng = nw->n_groups;
  const int n_iter = nw->iterations;
  MFTL_ASSERT(!nw->per_body || ng == b);

  double *const rec = (double *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(ng, 1) * sizeof(double), "volume correction: records");
  double *const step = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(ng, 1) * sizeof(double), "volume correction: steps");
  double **const area0 = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(double *), "volume correction: area snapshots");
  for (int g = 0; g < ng; ++g) offset[g] = 0.0;

  for (int o = 0; o < b; ++o)
  {
    if (nw->group[o] < 0) continue;
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    area0[o] = MFTLObjectSnapshotDensityAreas(obj);
  }

  const MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};
  for (int iter = 0; iter <= n_iter; ++iter)
  {
    for (int i = 0; i < 2 * ng; ++i) rec[i] = 0.0;
    for (int o = 0; o < b; ++o)
    {
      const int g = nw->group[o];
      if (g < 0) continue;
      MFTL_Object *const obj = ens->local_objects[o];
      double v = 0.0, a = 0.0;
      MFTLBodyFluxVolume(obj, nw->axis[g], nw->reference[g], &nw->centre[3 * g], &v, &a);
      rec[2 * g] += v;
      if (iter < n_iter) rec[2 * g + 1] += MFTLBodyVolumeDisplacement(obj);
    }
    /* The only branch around a collective is the mode, which is agreed. */
    if (nw->per_body)
      MFTLEnsembleReduceBodies(ens, rec, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
    else
      MFTLAllreduceDoubles(ens->domain.comm_world, rec, 2 * ng, MFTL_REDUCE_SUM);

    for (int g = 0; g < ng; ++g)
    {
      if (iter == 0) volume[2 * g] = rec[2 * g];
      if (iter == n_iter) volume[2 * g + 1] = rec[2 * g];
    }
    if (iter == n_iter) break;

    /* d = (V_target - V) / (dV/dd), once per group from the reduced values, so
     * every body of a group and every holder of a body take the same bits. */
    for (int g = 0; g < ng; ++g)
    {
      const double dvdd = rec[2 * g + 1];
      double d = (dvdd != 0.0 && isfinite(dvdd)) ? (nw->target[g] - rec[2 * g]) / dvdd : 0.0;
      if (!isfinite(d)) d = 0.0;
      step[g] = d;
      offset[g] += d;
    }
    for (int o = 0; o < b; ++o)
    {
      const int g = nw->group[o];
      if (g >= 0 && step[g] != 0.0) displace_owned(ens->local_objects[o], step[g], nw);
    }
    MFTLUpdateGhostPositions(ens);
  }

  for (int o = 0; o < b; ++o)
  {
    if (nw->group[o] < 0) continue;
    MFTL_Object *const obj = ens->local_objects[o];
    MFTLObjectRescaleDensities(obj, area0[o]);
    free(area0[o]);
    obj->derived &= ~MFTL_DERIVED_GEOMETRY;
    MFTLObjectMarkChanged(obj);
  }
  free(area0);
  free(step);
  free(rec);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Per body, against its reference volume
 * ═══════════════════════════════════════════════════════════════════════════ */

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
  const double t0 = MFTLWallTime();
  const int b = ens->n_local_objects;
  const size_t nb = (size_t) MFTL_MAX(b, 1);

  MFTLUpdateGhostPositions(ens);
  MFTL_BodyIntegrals *const bi = (MFTL_BodyIntegrals *) MFTLMallocOrDie(nb * sizeof(MFTL_BodyIntegrals), "volume correction: integrals");
  MFTLEnsembleBodyIntegrals(ens, bi);

  /* One group per held body: its own slot, its own region, its own reference. */
  int *const group = (int *) MFTLMallocOrDie(nb * sizeof(int), "volume correction: groups");
  int *const axis = (int *) MFTLMallocOrDie(nb * sizeof(int), "volume correction: axes");
  double *const real = (double *) MFTLMallocOrDie(nb * 8 * sizeof(double), "volume correction: per-body values");
  double *const reference = real, *const centre = real + nb, *const target = real + 4 * nb, *const volume = real + 5 * nb, *const offset = real + 7 * nb;
  double stat[4] = {0.0, 0.0, 0.0, 0.0};  // SUM: bodies, unmeasured; MAX: error before, after

  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const bool has_ref = isfinite(obj->reference_volume);
    const bool active = has_ref && bi[o].volume_status == MFTL_INTEGRAL_DEFINED && obj->reference_volume != 0.0;
    if (bi[o].lowest_holder == dom->rank)
    {
      if (active) stat[0] += 1.0;
      if (has_ref && !active) stat[1] += 1.0;
    }
    group[o] = active ? o : -1;
    axis[o] = bi[o].axis;
    reference[o] = bi[o].reference;
    /* Only centre[axis] is read, and only on a periodic axis -- where the
     * reference IS the body's centre (a closed body). */
    for (int k = 0; k < 3; ++k) centre[3 * o + k] = bi[o].reference;
    target[o] = obj->reference_volume;
  }

  MFTL_VolumeNewton nw = {.n_groups = b, .per_body = MFTL_TRUE, .group = group, .axis = axis, .reference = reference, .centre = centre, .target = target};
  nw.iterations = p->iterations;
  nw.rim_project = p->rim_project;
  nw.rim_project_user = p->rim_project_user;
  MFTLCorrectionNewtonVolumes(ens, &nw, volume, offset);

  double max_off = 0.0;
  for (int o = 0; o < b; ++o)
  {
    if (group[o] < 0) continue;
    stat[2] = fmax(stat[2], fabs(volume[2 * o] / target[o] - 1.0));
    stat[3] = fmax(stat[3], fabs(volume[2 * o + 1] / target[o] - 1.0));
    max_off = fmax(max_off, fabs(offset[o]));
  }

  MFTLAllreduceDoubles(dom->comm_world, stat, 2, MFTL_REDUCE_SUM);
  double mx[4] = {stat[2], stat[3], max_off, MFTLWallTime() - t0};
  MFTLAllreduceDoubles(dom->comm_world, mx, 4, MFTL_REDUCE_MAX);
  st->n_bodies = (int) stat[0];
  st->n_bodies_unmeasured = (int) stat[1];
  st->max_error_before = mx[0];
  st->max_error_after = mx[1];
  st->max_offset = mx[2];
  st->t_total = mx[3];

  free(real);
  free(axis);
  free(group);
  free(bi);
  return MFTL_SUCCESS;
}

/** @endcond */
