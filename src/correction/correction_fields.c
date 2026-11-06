/**
 * @file correction_fields.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief What a normal correction does to the optional fields.
 *
 * Both corrections move vertices along the surface normal and change no
 * connectivity. A normal motion carries no material across an edge, so a
 * CONSERVATIVE_DENSITY keeps its AMOUNT per triangle: rho_new A_new = rho A.
 * Because every operation of a call conserves the amount of every triangle it
 * touches, the composite is rho_end = rho_start A_start / A_end -- one rescale
 * at the end of the call, however many operations touched a triangle. Every
 * holder of a triangle gathers its corners in the same order from the same
 * (synchronised) positions, so the factor is the same bits on the owner and on
 * every ghost copy, and the copies stay consistent with no field exchange. A
 * triangle whose corners did not move has A_end == A_start bit for bit and is
 * left alone.
 *
 * Vertex fields keep their values: the vertex is the same material point.
 * (The remesher's smoother re-interpolates them because it moves vertices
 * TANGENTIALLY, over the material.)
 */

#include "correction_internal.h"

#include <math.h>
#include <stdlib.h>

#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"

static bool has_density(const MFTL_Object *obj)
{
  if (!obj->fields) return MFTL_FALSE;
  for (int f = 0; f < obj->fields->n; ++f)
    if (obj->fields->entries[f].entity == MFTL_FIELD_TRIANGLE && obj->fields->entries[f].mode == MFTL_FIELD_CONSERVATIVE_DENSITY) return MFTL_TRUE;
  return MFTL_FALSE;
}

static double triangle_area(const MFTL_Object *obj, int t)
{
  double T[3][3];
  MFTLGetTriangleCorners(obj, t, T);
  const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
  const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
  const double c[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  return 0.5 * sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
}

double *MFTLCorrectionSnapshotAreas(const MFTL_Object *obj)
{
  if (!has_density(obj)) return MFTL_NULL;
  const int nt = obj->tris.n;
  double *const a = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(double), "correction: triangle areas");
  for (int t = 0; t < nt; ++t) a[t] = triangle_area(obj, t);
  return a;
}

void MFTLCorrectionRescaleDensities(MFTL_Object *obj, const double *area_before)
{
  if (area_before == MFTL_NULL) return;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double a1 = triangle_area(obj, t);
    if (a1 == area_before[t] || !(a1 > 0.0)) continue;  // unmoved, or degenerate: nothing to divide by
    const double factor = area_before[t] / a1;
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *const d = &obj->fields->entries[f];
      if (d->entity != MFTL_FIELD_TRIANGLE || d->mode != MFTL_FIELD_CONSERVATIVE_DENSITY) continue;
      double *const row = MFTLObjectField(obj, f) + (size_t) t * d->ncomp;
      for (int c = 0; c < d->ncomp; ++c) row[c] *= factor;
    }
  }
}
