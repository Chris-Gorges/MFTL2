/**
 * @file bvh_internal.h
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Private helpers shared by the BVH module's translation units.
 *
 * Not a public header: nothing outside src/bvh/ may include it, and nothing here
 * is exported.
 */

#ifndef MFTL_BVH_INTERNAL_H
#define MFTL_BVH_INTERNAL_H

#include <math.h>
#include <stdbool.h>

#include "mftl_bvh.h"
#include "mftl_constants.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Float bounds that are always a superset of the double ones
 *
 * The node AABB is float and the geometry is double, so a naive (float) cast
 * rounds to NEAREST and can shrink the box by one ulp on either side. One ulp is
 * enough: a triangle whose vertex sits exactly on the max face of its own leaf
 * gets excluded from it, and the traversal then reports a clean miss for a pair
 * that touches.
 *
 * Rounding the min DOWN and the max UP makes the float box a conservative
 * superset, which is the only direction a broad phase may ever be wrong in. The
 * cost is one nextafter-free trick: adding/subtracting one ulp via nextafterf is
 * a libm call, while multiplying by (1 +/- 2^-23) is two instructions and is
 * conservative for every magnitude including zero-crossings, provided the offset
 * is applied additively for values near zero. The form below does both: scale
 * for the magnitude, plus a denormal-proof absolute nudge.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Relative slack of one float ulp, doubled for safety against the cast. */
#define MFTL_BVH_FLOAT_SLACK 2.4e-7f
/** @brief Absolute slack, so the rounding is still outward through zero. */
#define MFTL_BVH_FLOAT_TINY 1.0e-30f

/**
 * @brief Largest float not exceeding @p v, with slack.
 * @param v The double to round down.
 * @return A float strictly below @p v. Rounding the box minimum DOWN is the only
 *         direction a broad phase may ever be wrong in.
 */
static inline float mftl_bvh_floor_f(double v)
{
  const float f = (float) v;
  return (f >= 0.0f) ? (f * (1.0f - MFTL_BVH_FLOAT_SLACK) - MFTL_BVH_FLOAT_TINY) : (f * (1.0f + MFTL_BVH_FLOAT_SLACK) - MFTL_BVH_FLOAT_TINY);
}

/**
 * @brief Smallest float not below @p v, with slack.
 * @param v The double to round up.
 * @return A float strictly above @p v.
 */
static inline float mftl_bvh_ceil_f(double v)
{
  const float f = (float) v;
  return (f >= 0.0f) ? (f * (1.0f + MFTL_BVH_FLOAT_SLACK) + MFTL_BVH_FLOAT_TINY) : (f * (1.0f - MFTL_BVH_FLOAT_SLACK) + MFTL_BVH_FLOAT_TINY);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. AABB overlap, with a shift on the B side
 *
 * The shift is the periodic image offset. It is added to B rather than
 * subtracted from A because A is the outer loop of the traversal and its node is
 * already in registers; a zero shift costs three adds the compiler hoists out of
 * the leaf case entirely.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Do two node boxes overlap, with @p b displaced by @p shift?
 *
 * @param a     Node from the A-side tree.
 * @param b     Node from the B-side tree.
 * @param shift Periodic image offset added to @p b. All zeros for the identity
 *              image.
 * @return `true` if the two boxes intersect on all three axes.
 */
static inline bool mftl_bvh_aabb_overlap(const MFTL_BVHNode *a, const MFTL_BVHNode *b, const float shift[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (a->aabb_max[k] < b->aabb_min[k] + shift[k]) return MFTL_FALSE;
    if (a->aabb_min[k] > b->aabb_max[k] + shift[k]) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/**
 * @brief Volume of a node's box; the descent heuristic picks the larger one.
 * @param n The node.
 * @return \f$\prod_k (\mathrm{max}_k - \mathrm{min}_k)\f$.
 */
static inline float mftl_bvh_node_volume(const MFTL_BVHNode *n)
{
  return (n->aabb_max[0] - n->aabb_min[0]) * (n->aabb_max[1] - n->aabb_min[1]) * (n->aabb_max[2] - n->aabb_min[2]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Shared build-side helpers
 *
 * These two are MODULE-PRIVATE -- declared here, defined in bvh_build.c, called
 * from bvh_ghosts.c, and absent from mftl_bvh.h. They still carry the MFTL
 * prefix because they need external linkage to cross a translation unit, and
 * the library's symbol rule ("nothing but MFTL* may be exported", CLAUDE.md) is
 * checked on the archive rather than on the headers. Do not call them from
 * outside src/bvh/.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Refits every node's AABB, and resolves the tolerance while doing it.
 *
 * Three linear passes over the node array, no recursion and no explicit stack:
 *
 *   1. REVERSE SCAN over [nodes_used-1 .. 0], fitting each leaf to its
 *      primitives and each internal node to its two children. The build
 *      allocates a child pair by post-incrementing `nodes_used`, so a child
 *      index is always greater than its parent's -- walking backwards therefore
 *      visits both children before the parent that merges them. That property is
 *      the whole reason the refit-only fast path is worth having.
 *   2. The root box now bounds every primitive, so `bounding_radius` and
 *      `tolerance` come straight off it.
 *   3. FORWARD SCAN inflating every node by the tolerance.
 *
 * ── Why the inflation is a separate pass and not part of pass 1 ────────────
 *
 * The tolerance depends on the bounding radius, which is only known once pass 1
 * has finished -- so inflating the leaves inside pass 1 would need the radius
 * from a pass that has not run yet. The alternative is a preliminary sweep over
 * the PRIMITIVES to find the box first, which costs nine doubles per primitive.
 *
 * Pass 3 avoids it because a uniform inflation COMMUTES with the min/max merge:
 * growing every leaf by `tol` and re-merging gives exactly the same tree as
 * merging first and growing every node by `tol`. So the correction is applied
 * once per NODE, over an array that is still in cache from pass 1, instead of
 * once per primitive over the mesh.
 *
 * Shared by MFTL_GenerateLBVH (as its last build step) and MFTL_RefitLBVH (as
 * the entire operation), so the two can never disagree about what a fitted tree
 * looks like -- which is what `test_refit_only_matches_full_rebuild` asserts.
 *
 * @param bvh The hierarchy. Every node's box, plus `bounding_radius` and
 *            `tolerance`, are rewritten.
 */
void MFTLBVHRefit(MFTL_BVH *bvh);

/**
 * @brief Morton codes, radix sort, tree build and fit over the CURRENT primitive
 *        set -- everything in a build that does not depend on where the
 *        primitives came from.
 *
 * Split out because MFTL_CreateSpatialGhosts has to redo exactly this after
 * appending remote primitives. The Morton order, the tree and the fit are all
 * functions of the FULL primitive set, so a tree built over the local half
 * cannot have leaves bolted onto it afterwards -- the sorted order the leaves
 * address would no longer be sorted.
 *
 * @param bvh The hierarchy. Expects `n_prims`, `n_local_prims`, the remote
 *            arrays, `nodes` and `prim_indices` to be sized and populated;
 *            rewrites the tree in place.
 */
void MFTLBVHBuildFromPrims(MFTL_BVH *bvh);

#endif  // MFTL_BVH_INTERNAL_H
