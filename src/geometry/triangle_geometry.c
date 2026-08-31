/**
 * @file triangle_geometry.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_geometry
 *
 * @brief Per-triangle geometry: area, unit normal and centroid for a whole mesh.
 *
 * ## TWO CONNECTIVITY PATHS, ONE ENTRY POINT
 *
 * The same three outputs can be produced two ways, and which one is available
 * depends on the state of the object rather than on what the caller wants:
 *
 *   HALF EDGE  valid even before derived connectivity has been built. This is
 *              the construction-time fallback.
 *
 *   VERTICES   streams `tris.verts`, gathers the authoritative positions and
 *              creates seam-corrected edges in registers. Periodicity is
 *              unswitched outside the triangle loop.
 *
 * Both paths have identical geometry arithmetic. The normal steady-state path
 * is VERTICES; the half-edge fallback preserves the construction-time API.
 */

#include "mftl_geometry.h"

#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_types.h"

/**
 * @brief The general path: half-edge walk, gathered positions, minimum image.
 *
 * Written as a loop over MFTLGetTriangleProperties rather than open-coded,
 * because that function is the definition of what a triangle's geometry IS and
 * two spellings of it would be one too many. It is `static inline` in the header
 * and this is a tight loop around it, so nothing is called per triangle.
 *
 * The outputs go straight into the arrays: `&tris.normal[3*t]` is a three-double
 * slot of exactly the shape the output parameter wants, so this stays a direct
 * store with no stack round-trip.
 *
 * @param obj   The mesh. `tris.area`, `tris.normal` and `tris.center` are written.
 * @param count Triangles to process, from index 0.
 */
static void geometry_by_walk(MFTL_Object *obj, int count)
{
  for (int t = 0; t < count; ++t)
  {
    MFTLGetTriangleProperties(obj, t, &obj->tris.area[t], &obj->tris.normal[3 * t], &obj->tris.center[3 * t]);
  }
}

/**
 * @brief The hot path: compact indices plus gathered authoritative positions.
 *
 * The centroid still needs MFTLApplyPeriodicity, and that stays the branchy
 * library call deliberately: `p0 + (e1+e2)/3` is a true centroid but not
 * necessarily a WRAPPED position, and redistribution feeds it to
 * MFTLGetOwnerProc. The branchless alternative was measured 1.34-1.37x slower in
 * this exact shape of loop (see src/ensemble/ensemble_update.c), because it does
 * unconditional work on non-periodic axes too while these three branches are
 * perfectly predicted.
 *
 * @param obj   The mesh. Requires #MFTL_DERIVED_TOPOLOGY.
 * @param count Triangles to process, from index 0.
 */
static void geometry_by_vertices(MFTL_Object *obj, int count)
{
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, count,
    /* Unnormalised normal; its magnitude is twice the area, so one sqrt serves
     * both outputs and there is no second square root anywhere in the loop. */
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;

    /* The degeneracy guard is a test against zero on the DIVISOR, never a clamp
     * on the magnitude: clamping floors the reported AREA and leaves a small
     * triangle's normal short of unit. Argued in full, with the +1.1 % it costs,
     * in MFTLGetTriangleProperties (mftl_geometry.h). This must stay the exact
     * same expression -- test_triangle_geometry_paths_are_bit_identical compares
     * the two paths with memcmp, so a difference of one operation here is a
     * difference in the last bit there. */
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;

    out_area[t] = 0.5 * mag;
    out_normal[3 * t + 0] = nx * inv;
    out_normal[3 * t + 1] = ny * inv;
    out_normal[3 * t + 2] = nz * inv;

    double c[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
    MFTLApplyPeriodicity(dom, c);

    out_center[3 * t + 0] = c[0];
    out_center[3 * t + 1] = c[1];
    out_center[3 * t + 2] = c[2];);
}

void MFTLComputeTriangleGeometry(MFTL_Object *obj, int count)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(count <= obj->tris.n && "MFTLComputeTriangleGeometry: count exceeds the triangle pool");

  if (count <= 0) return;

  if ((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u)
    geometry_by_vertices(obj, count);
  else
    geometry_by_walk(obj, count);
}
