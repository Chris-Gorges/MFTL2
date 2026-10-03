/**
 * @file periodic_extent.c
 * @author Christian Gorges
 * @date 2026-10-03
 * @ingroup mftl_geometry
 *
 * @brief A body's extent on a periodic axis, measured in four frames.
 *
 * A body's stored coordinates on a periodic axis are wrapped into the domain
 * box, so a body straddling the seam has a raw min/max spanning the whole axis.
 * Measured in a frame whose cut lies in the body's GAP it is contiguous, and its
 * min/max is its true extent. Two frames (cuts at 0 and L/2) find such a cut for
 * every body narrower than L/2 and for no wider one that straddles both; four
 * frames (cuts at 0, L/4, L/2, 3L/4) find one for every body narrower than 3L/4.
 *
 * The frames at 0 and L/2 are computed and compared EXACTLY as the two-frame
 * code they replace (topology clusters), so every extent that code measured
 * below 3L/4 is reproduced to the bit; the quarter frames are only read when
 * neither measures less. Theory: src/geometry/README.md, "Extent on a periodic
 * axis".
 */

#include "mftl_geometry.h"

#include <math.h>

#include "mftl_constants.h"
#include "mftl_types.h"

/** @brief Offset of frame f as a fraction of the period, in slot order. */
static const double frame_shift[4] = {0.0, 0.5, 0.25, 0.75};

void MFTLPeriodicExtentInit(double acc[MFTL_PERIODIC_EXTENT_SLOTS])
{
  for (int i = 0; i < MFTL_PERIODIC_EXTENT_SLOTS; ++i) acc[i] = MFTL_DBL_MAX;
}

void MFTLPeriodicExtentAddTriangle(const MFTL_Domain *dom, const double T[3][3], double acc[MFTL_PERIODIC_EXTENT_SLOTS])
{
  for (int k = 0; k < 3; ++k)
  {
    const double lo = fmin(T[0][k], fmin(T[1][k], T[2][k]));
    const double hi = fmax(T[0][k], fmax(T[1][k], T[2][k]));
    acc[k] = fmin(acc[k], lo);
    acc[3 + k] = fmin(acc[3 + k], -hi);

    if (dom == MFTL_NULL || !dom->is_periodic[k]) continue;

    const double L = dom->domain_size[k];
    for (int f = 1; f < 4; ++f)
    {
      /* Shift corner 0 by the frame's offset and wrap it back into the box; the
       * other corners follow rigidly, so the triangle keeps its shape. */
      double c0 = T[0][k] + frame_shift[f] * L;
      c0 -= L * floor((c0 - dom->global_min[k]) / L);
      const double d = c0 - T[0][k];
      acc[6 * f + k] = fmin(acc[6 * f + k], lo + d);
      acc[6 * f + 3 + k] = fmin(acc[6 * f + 3 + k], -(hi + d));
    }
  }
}

void MFTLPeriodicExtentAddObject(const MFTL_Object *obj, double acc[MFTL_PERIODIC_EXTENT_SLOTS])
{
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    MFTLPeriodicExtentAddTriangle(obj->domain, T, acc);
  }
}

void MFTLPeriodicExtentResolve(const MFTL_Domain *dom, const double acc[MFTL_PERIODIC_EXTENT_SLOTS], MFTL_PeriodicExtent *out)
{
  out->empty = !(acc[0] < MFTL_DBL_MAX);
  for (int k = 0; k < 3; ++k)
  {
    out->covers[k] = MFTL_FALSE;
    if (out->empty)
    {
      out->centre[k] = 0.0;
      out->half[k] = 0.0;
      continue;
    }

    double lo = acc[k], hi = -acc[3 + k];
    if (dom != MFTL_NULL && dom->is_periodic[k])
    {
      const double L = dom->domain_size[k];
      const double cover = MFTL_PERIODIC_EXTENT_COVER_FRACTION * L;

      /* The two frames the extent was always measured in, compared as they
       * always were: the half frame only when STRICTLY tighter. */
      const double lo1 = acc[6 + k], hi1 = -acc[9 + k];
      if (hi1 - lo1 < hi - lo)
      {
        lo = lo1 - 0.5 * L;
        hi = hi1 - 0.5 * L;
      }

      /* A body straddling both of their cuts measures at least L in each; one of
       * the quarter frames then has its cut in the body's gap if the body is
       * narrower than 3L/4. */
      if (!(hi - lo < cover))
      {
        for (int f = 2; f < 4; ++f)
        {
          const double lf = acc[6 * f + k], hf = -acc[6 * f + 3 + k];
          if (hf - lf < cover && hf - lf < hi - lo)
          {
            lo = lf - frame_shift[f] * L;
            hi = hf - frame_shift[f] * L;
          }
        }
      }
      out->covers[k] = !(hi - lo < cover);
    }
    out->centre[k] = 0.5 * (lo + hi);
    out->half[k] = 0.5 * (hi - lo);
  }
}
