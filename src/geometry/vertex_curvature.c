/**
 * @file vertex_curvature.c
 * @author Christian Gorges
 * @date 2026-10-23
 * @ingroup mftl_geometry
 *
 * @brief Discrete curvature (Meyer et al. 2003), for the remesher's curvature
 *        sizing.
 *
 * Both curvatures are ONE-RING sums, which is what makes them exact on an owned
 * vertex of a distributed mesh with no communication: the redistribution halo
 * is vertex-complete, so every triangle incident to an owned vertex is present
 * locally. A ghost vertex receives a PARTIAL sum -- plausible and wrong,
 * exactly like a ghost's vertex normal.
 *
 * The pass is a scatter over triangles (MFTL_FOREACH_TRIANGLE_GEOMETRY), for
 * the same reason MFTLComputeVertexNormals() is one: a per-vertex fan walk
 * serialises dependent loads.
 *
 * Nothing here compares a dimensional quantity against a constant: degenerate
 * triangles are those whose cross product is exactly zero. Scaling a mesh by a
 * power of two therefore scales every output by the matching power, bit for
 * bit.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Mean and Gaussian curvature
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLComputeVertexCurvatures(const MFTL_Object *obj, double *mean_curvature, double *gauss_curvature)
{
  MFTL_ASSERT(obj != MFTL_NULL && mean_curvature != MFTL_NULL && gauss_curvature != MFTL_NULL);
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLComputeVertexCurvatures: tris.verts is stale -- MFTLObjectBuildDerived first");

  const int n_v = obj->verts.n;
  if (n_v == 0) return;

  /* Per vertex: the cotangent Laplacian (3), the angle sum (1) and the mixed
   * area (1), accumulated face by face and resolved in one pass at the end. */
  double *const acc = (double *) MFTLCallocOrDie((size_t) n_v * 5, sizeof(double), "vertex curvature accumulators");
  double *restrict lap = acc;
  double *restrict angle = acc + 3 * (size_t) n_v;
  double *restrict area = acc + 4 * (size_t) n_v;

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, obj->tris.n,
    /* Corner 0 at p0, edges e1 = p1 - p0 and e2 = p2 - p0; c = p2 - p1 is the
     * edge opposite corner 0. Every corner shares |e1 x e2| = 2A, so a
     * cotangent is a dot product over it. */
    const double cx = e2x - e1x, cy = e2y - e1y, cz = e2z - e1z;
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    if (!(mag > 0.0)) continue;  // a degenerate face has no angles to give

    const double l1 = e1x * e1x + e1y * e1y + e1z * e1z;  // |p1 - p0|^2
    const double l2 = e2x * e2x + e2y * e2y + e2z * e2z;  // |p2 - p0|^2
    const double lc = cx * cx + cy * cy + cz * cz;  // |p2 - p1|^2

    /* Dot products of the two edges AT each corner: 0 (e1, e2), 1 (-e1, c),
     * 2 (-e2, -c). Negative means obtuse there -- an exact sign test. */
    const double d0 = e1x * e2x + e1y * e2y + e1z * e2z;
    const double d1 = -(e1x * cx + e1y * cy + e1z * cz);
    const double d2 = e2x * cx + e2y * cy + e2z * cz;

    const double inv = 1.0 / mag;
    const double cot0 = d0 * inv, cot1 = d1 * inv, cot2 = d2 * inv;

    /* (cot alpha + cot beta)(x_i - x_j), one triangle's share per edge:
     * edge v1-v2 carries cot0, v0-v2 cot1, v0-v1 cot2. */
    lap[3 * v0 + 0] += -cot1 * e2x - cot2 * e1x;
    lap[3 * v0 + 1] += -cot1 * e2y - cot2 * e1y;
    lap[3 * v0 + 2] += -cot1 * e2z - cot2 * e1z;
    lap[3 * v1 + 0] += -cot0 * cx + cot2 * e1x;
    lap[3 * v1 + 1] += -cot0 * cy + cot2 * e1y;
    lap[3 * v1 + 2] += -cot0 * cz + cot2 * e1z;
    lap[3 * v2 + 0] += cot0 * cx + cot1 * e2x;
    lap[3 * v2 + 1] += cot0 * cy + cot1 * e2y;
    lap[3 * v2 + 2] += cot0 * cz + cot1 * e2z;

    angle[v0] += atan2(mag, d0);
    angle[v1] += atan2(mag, d1);
    angle[v2] += atan2(mag, d2);

    /* The MIXED area (Meyer et al., Fig. 4): the Voronoi share where the
     * triangle is not obtuse, else half the area to the obtuse corner and a
     * quarter to each of the others. The Voronoi shares of one triangle sum to
     * its area, so the mixed areas tile the surface either way. */
    const double A = 0.5 * mag;
    if (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
    {
      area[v0] += 0.125 * (l2 * cot1 + l1 * cot2);
      area[v1] += 0.125 * (l1 * cot2 + lc * cot0);
      area[v2] += 0.125 * (lc * cot0 + l2 * cot1);
    }
    else
    {
      area[v0] += (d0 < 0.0) ? 0.5 * A : 0.25 * A;
      area[v1] += (d1 < 0.0) ? 0.5 * A : 0.25 * A;
      area[v2] += (d2 < 0.0) ? 0.5 * A : 0.25 * A;
    });
  // clang-format on

  for (int v = 0; v < n_v; ++v)
  {
    const double a = area[v];
    if (!(a > 0.0))
    {
      mean_curvature[v] = 0.0;
      gauss_curvature[v] = 0.0;
      continue;
    }

    /* Delta x = (1 / 2A) SUM (cot alpha + cot beta)(x_i - x_j) = 2 H n. */
    const double *const L = &lap[3 * v];
    const double len = MFTL_SQRT(L[0] * L[0] + L[1] * L[1] + L[2] * L[2]);
    mean_curvature[v] = 0.25 * len / a;
    gauss_curvature[v] = (MFTL_TWO_PI - angle[v]) / a;
  }

  free(acc);
}
