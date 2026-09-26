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
 * A plain (float) cast rounds to nearest and can shrink a box by one ulp, which
 * is enough to exclude a vertex lying on its own leaf's face and report a clean
 * miss. Rounding the minimum DOWN and the maximum UP keeps the float box a
 * conservative superset: scaling by (1 -/+ 2.4e-7), i.e. two float ulps
 * relative, leaves a margin of at least one ulp after both roundings, and an
 * absolute 1e-30 keeps the rounding outward through zero and the denormals.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Relative slack, about two float ulps (\f$2^{-22}\f$). */
#define MFTL_BVH_FLOAT_SLACK 2.4e-7f
/** @brief Absolute slack, so the rounding is still outward through zero. */
#define MFTL_BVH_FLOAT_TINY 1.0e-30f

/**
 * @brief A float guaranteed to be below @p v (outward rounding of a minimum).
 * @param v The double to round down.
 * @return A float strictly below @p v.
 */
static inline float mftl_bvh_floor_f(double v)
{
  const float f = (float) v;
  return (f >= 0.0f) ? (f * (1.0f - MFTL_BVH_FLOAT_SLACK) - MFTL_BVH_FLOAT_TINY) : (f * (1.0f + MFTL_BVH_FLOAT_SLACK) - MFTL_BVH_FLOAT_TINY);
}

/**
 * @brief A float guaranteed to be above @p v (outward rounding of a maximum).
 * @param v The double to round up.
 * @return A float strictly above @p v.
 */
static inline float mftl_bvh_ceil_f(double v)
{
  const float f = (float) v;
  return (f >= 0.0f) ? (f * (1.0f + MFTL_BVH_FLOAT_SLACK) + MFTL_BVH_FLOAT_TINY) : (f * (1.0f - MFTL_BVH_FLOAT_SLACK) + MFTL_BVH_FLOAT_TINY);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. AABB overlap, with the periodic image offset added to the B side.
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
 * 3. Module-private build helpers, defined in bvh_build.c. They carry the MFTL
 * prefix only because every external symbol must; do not call them from outside
 * src/bvh/.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Refits every node's box and resolves the tolerance, in three linear
 *        passes with no recursion.
 *
 * 1. Reverse scan over `[nodes_used-1 .. 0]`: leaves fit their primitives,
 *    internal nodes merge their children (every child index exceeds its
 *    parent's, so children are always visited first).
 * 2. `bounding_radius` from the root box; `tolerance` from it (or from
 *    `radius_hint`).
 * 3. Forward scan inflating every node by the tolerance -- valid because a
 *    uniform inflation commutes with the min/max merge.
 *
 * Shared by MFTL_GenerateLBVH() and MFTL_RefitLBVH()
 * (`test_refit_only_matches_full_rebuild`).
 *
 * @param bvh The hierarchy; boxes, `bounding_radius` and `tolerance` rewritten.
 */
void MFTLBVHRefit(MFTL_BVH *bvh);

/**
 * @brief Keys, sort, split and fit over the CURRENT primitive set -- the part of
 *        a build independent of where the primitives came from. Used by
 *        MFTL_GenerateLBVH() and, after attaching a patch, MFTL_BVHSetHalo()
 *        (the order is a function of the whole set, so leaves cannot be
 *        appended to an existing tree).
 *
 * @param bvh The hierarchy, with `n_prims`, `n_local_prims`, `halo`, `nodes`
 *            and `prim_indices` sized; the tree is rewritten in place.
 */
void MFTLBVHBuildFromPrims(MFTL_BVH *bvh);

#endif  // MFTL_BVH_INTERNAL_H
