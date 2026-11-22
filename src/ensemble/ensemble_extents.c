/**
 * @file ensemble_extents.c
 * @author Christian Gorges
 * @date 2026-10-03
 * @ingroup mftl_ensemble
 *
 * @brief The periodic extent rule, checked over the ensemble.
 *
 * A loop over the local objects (MFTLPeriodicExtentAddObject()), ONE per-body
 * reduction over the holders, and a count agreed over `comm_world` -- the shape
 * every MFTLEnsemble* function has, with the arithmetic in src/geometry/.
 */

#include "mftl_ensemble.h"

#include <stdio.h>
#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_types.h"

/** @brief Slots per body: the extent, then the contributing rank (MIN: the lowest holder). */
#define MFTL_EXT_STRIDE (MFTL_PERIODIC_EXTENT_SLOTS + 1)

int MFTLEnsembleCheckPeriodicExtents(MFTL_Ensemble *ens, MFTL_PeriodicExtentReport *report)
{
  MFTL_PeriodicExtentReport sink;
  MFTL_PeriodicExtentReport *const rep = (report != MFTL_NULL) ? report : &sink;
  rep->n_violations = 0;
  rep->n_covering = 0;
  rep->first_body_gid = -1;
  rep->first_axis = -1;
  rep->first_extent = 0.0;

  if (ens == MFTL_NULL) return MFTL_ERROR;

  /* Both tests are identical on every rank: periodicity is a property of the
   * domain and the body count is agreed, so returning here enters no collective
   * that another rank enters. */
  const MFTL_Domain *const dom = &ens->domain;
  if (!(dom->is_periodic[0] || dom->is_periodic[1] || dom->is_periodic[2])) return MFTL_SUCCESS;
  if (ens->n_bodies_global < 1) return MFTL_SUCCESS;

  const int b = ens->n_local_objects;
  double *const acc = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * MFTL_EXT_STRIDE, 1) * sizeof(double), "periodic extents");
  MFTL_BodyReduceOp ops[MFTL_EXT_STRIDE];
  for (int j = 0; j < MFTL_EXT_STRIDE; ++j) ops[j] = MFTL_BODY_REDUCE_MIN;

  for (int o = 0; o < b; ++o)
  {
    double *const a = &acc[(size_t) MFTL_EXT_STRIDE * o];
    MFTLPeriodicExtentInit(a);
    MFTLPeriodicExtentAddObject(ens->local_objects[o], a);
    a[MFTL_PERIODIC_EXTENT_SLOTS] = (double) dom->rank;
  }
  MFTLEnsembleReduceBodies(ens, acc, MFTL_EXT_STRIDE, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* The verdict per body is identical on every holder; the lowest holder speaks
   * for it, so a shared body is counted once. */
  long long count[2] = {0, 0};  // violations, covering
  double key = MFTL_DBL_MAX;  // gid * 3 + axis of this rank's first violation: exact in a double
  double extent = 0.0;
  for (int o = 0; o < b; ++o)
  {
    const double *const a = &acc[(size_t) MFTL_EXT_STRIDE * o];
    MFTL_PeriodicExtent x;
    MFTLPeriodicExtentResolve(dom, a, &x);
    if (x.empty) continue;

    bool covering = MFTL_FALSE;
    int axis = -1;
    for (int k = 0; k < 3; ++k)
    {
      if (!dom->is_periodic[k]) continue;
      if (x.covers[k])
        covering = MFTL_TRUE;
      else if (axis < 0 && !(2.0 * x.half[k] < MFTL_PERIODIC_EXTENT_RULE_FRACTION * dom->domain_size[k]))
        axis = k;
    }
    if (axis >= 0)
    {
      const double kk = 3.0 * (double) ens->local_objects[o]->global_id + (double) axis;
      if (kk < key)
      {
        key = kk;
        extent = 2.0 * x.half[axis];
      }
    }
    if ((int) a[MFTL_PERIODIC_EXTENT_SLOTS] != dom->rank) continue;
    if (axis >= 0) count[0]++;
    if (covering) count[1]++;
  }
  free(acc);

  MFTLAllreduceLongLongs(dom->comm_world, count, 2, MFTL_REDUCE_SUM);
  rep->n_violations = (int) count[0];
  rep->n_covering = (int) count[1];
  if (count[0] == 0) return MFTL_SUCCESS;

  /* The first offender, then its extent from the ranks holding it: every holder
   * resolved the same reduced record, so the MIN is that value. */
  double first = key;
  MFTLAllreduceDoubles(dom->comm_world, &first, 1, MFTL_REDUCE_MIN);
  double ext = (key == first) ? extent : MFTL_DBL_MAX;
  MFTLAllreduceDoubles(dom->comm_world, &ext, 1, MFTL_REDUCE_MIN);

  rep->first_body_gid = (int) (first / 3.0);
  rep->first_axis = (int) (first - 3.0 * (double) rep->first_body_gid);
  rep->first_extent = ext;

  if (dom->rank == 0)
  {
    const char axis_name[3] = {'x', 'y', 'z'};
    fprintf(stderr,
            "[MFTL] MFTLEnsembleCheckPeriodicExtents: %d bod%s break%s the periodic extent rule. Body %d is %.6g wide along %c against a "
            "period of %.6g: a body that does not wrap a periodic axis must be narrower than half the period.\n",
            rep->n_violations, rep->n_violations == 1 ? "y" : "ies", rep->n_violations == 1 ? "s" : "", rep->first_body_gid, rep->first_extent,
            axis_name[rep->first_axis], dom->domain_size[rep->first_axis]);
  }
  return MFTL_ERROR;
}
