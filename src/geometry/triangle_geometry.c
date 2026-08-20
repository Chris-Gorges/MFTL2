/**
 * @file triangle_geometry.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_geometry
 *
 * @brief Per-triangle geometry: area, unit normal and centroid for a whole mesh.
 *
 * ## TWO KERNELS, ONE ENTRY POINT, AND THE TEST IS OUTSIDE THE LOOP
 *
 * The same three outputs can be produced two ways, and which one is available
 * depends on the state of the object rather than on what the caller wants:
 *
 *   WALK    three half-edge records per triangle, three gathered vertex
 *           positions, two minimum-image corrections. Valid on ANY object --
 *           a freshly generated mesh has no derived arrays and this is what a
 *           generator has to be able to call.
 *
 *   STREAM  `tris.anchor` and `tris.edge`, both contiguous, both already
 *           seam-consistent. No gather, no periodic arithmetic, no indirection.
 *           Valid only while MFTL_DERIVED_FRAMES is set.
 *
 * Measured by examples/geometry_update_benchmark.c, interleaved, best of many,
 * 5 120 to 327 680 triangles:
 *
 *     non-periodic    walk 2.1-2.6 ns/tri   stream 1.4-2.0 ns/tri   1.25-1.57x
 *     periodic        walk 3.5-3.8 ns/tri   stream 1.8-1.9 ns/tri   1.94-2.00x
 *
 * The periodic margin is the larger one because that is where the walk actually
 * pays for the minimum-image correction the frames already carry -- which is the
 * whole argument for `tris.edge` existing, arriving here as a number.
 *
 * What this does NOT buy is stated in docs/performance-analysis.md §9: in the
 * one production loop in this repository the phase that benefits is waiting on
 * first-touch of pools a redistribution has just rewritten, and the dispatch
 * moves it by about one percent. The kernel is faster; that pass is not
 * kernel-bound.
 *
 * ### Why this is a dispatch and not a second public function
 *
 * The alternative was `MFTLComputeTriangleGeometryFromFrames()` next to the
 * existing entry point, and it was rejected: it makes every call site restate a
 * fact the object already knows, and a call site that gets it wrong either
 * asserts (best case) or reads stale frames (worst). One test on `obj->derived`,
 * hoisted out of the loop, costs a predicted branch per CALL -- not per triangle
 * -- and cannot be got wrong.
 *
 * It also removes the temptation to "optimise" by refreshing the frames first.
 * MFTLObjectRefreshFrames gathers three positions per triangle and writes 72 B,
 * i.e. it costs about what the walk costs; paying it to enable a 1.6 ns kernel
 * is a net loss for a single pass. The fast path is worth taking when the frames
 * are current ANYWAY -- which after a redistribution they always are.
 *
 * ### Layering
 *
 * This implements a layer-2 header, so it tests `obj->derived` directly rather
 * than through MFTLObjectHasFrames() and writes the streaming loop out rather
 * than using MFTL_FOREACH_TRIANGLE_FRAME -- both of those live in mftl_object.h,
 * which is layer 3. src/geometry/vertex_geometry.c does the same thing for the
 * same reason. The loop body is identical to the macro's; if one changes, both
 * must.
 */

#include "mftl_geometry.h"

#include "mftl_constants.h"
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
 * @brief The hot path: two contiguous input streams, three contiguous outputs.
 *
 * Everything this loop needs is already in `tris.anchor` (corner 0) and
 * `tris.edge` ({v1-v0, v2-v0}, seam-consistent). So there is no gather, and no
 * minimum-image call -- the correction was applied once when the frames were
 * built and is baked into the edge vectors.
 *
 * The centroid still needs MFTLApplyPeriodicity, and that stays the branchy
 * library call deliberately: `p0 + (e1+e2)/3` is a true centroid but not
 * necessarily a WRAPPED position, and redistribution feeds it to
 * MFTLGetOwnerProc. The branchless alternative was measured 1.34-1.37x slower in
 * this exact shape of loop (see src/ensemble/ensemble_update.c), because it does
 * unconditional work on non-periodic axes too while these three branches are
 * perfectly predicted.
 *
 * Five `restrict` pointers, one invariant bound, no aliasing possible in fact --
 * the inputs live in the triangle pool block and so do the outputs, but at
 * disjoint cache-line-aligned sub-arrays, which is exactly what the single-block
 * layout guarantees.
 *
 * @param obj   The mesh. Requires #MFTL_DERIVED_FRAMES.
 * @param count Triangles to process, from index 0.
 */
static void geometry_by_frames(MFTL_Object *obj, int count)
{
  const double *const restrict anc = obj->tris.anchor;
  const double *const restrict ed = obj->tris.edge;

  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  const MFTL_Domain *const dom = obj->domain;

  for (int t = 0; t < count; ++t)
  {
    const double e1x = ed[6 * t + 0], e1y = ed[6 * t + 1], e1z = ed[6 * t + 2];
    const double e2x = ed[6 * t + 3], e2y = ed[6 * t + 4], e2z = ed[6 * t + 5];

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

    const double p0x = anc[3 * t + 0], p0y = anc[3 * t + 1], p0z = anc[3 * t + 2];

    double c[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
    MFTLApplyPeriodicity(dom, c);

    out_center[3 * t + 0] = c[0];
    out_center[3 * t + 1] = c[1];
    out_center[3 * t + 2] = c[2];
  }
}

void MFTLComputeTriangleGeometry(MFTL_Object *obj, int count)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(count <= obj->tris.n && "MFTLComputeTriangleGeometry: count exceeds the triangle pool");

  if (count <= 0) return;

  /* The frames are built for the WHOLE live prefix (build_frames runs to
   * tris.n), so any count the assertion above admits is covered by them. */
  if ((obj->derived & MFTL_DERIVED_FRAMES) != 0u)
    geometry_by_frames(obj, count);
  else
    geometry_by_walk(obj, count);
}
