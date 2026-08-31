/**
 * @file mftl_bvh.h
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Linear BVH over triangle primitives, and object-object contact detection.
 *
 * Broad phase (this file) answers "which triangle pairs are close enough to be
 * worth an exact test"; the narrow phase answers "are they actually touching".
 * Four decisions shape the module, and every one of them is forced by the object
 * model rather than chosen for taste.
 *
 * ## 1. A BVH IS BUILT OVER A PRIMITIVE SOUP, NOT OVER THE MESH
 *
 * A primitive is one triangle of some object, and the BVH stores no geometry of
 * its own at all -- only a permutation and a tree of boxes over it:
 *
 *   prims [0, n_local_prims)  triangle t of `obj`
 *   prims [n_local_prims, n)  triangle (p - n_local_prims) of `halo`
 *
 * `halo` is the spatial halo patch for this body: an ordinary MFTL_Object
 * holding the triangles that neighbouring ranks sent because they are near this
 * rank's share of some other body (MFTLExtendSpatialHalo, mftl_communication.h
 * §6). It is BORROWED, and both objects are `const` throughout this module.
 *
 * Because a patch is a real object, both kinds of primitive are reached through
 * the same MFTLGetTriangleCorners call and the distinction survives only as one
 * index comparison. The alternative -- a private array of nine doubles per
 * received triangle -- is a second geometry representation with its own growth
 * path and its own periodic-consistency argument, for data the object machinery
 * already knows how to hold.
 *
 * What has NOT changed is that received triangles stay out of the BODY's pools.
 * Appending them there would need the two-way ownership split of mftl_types.h
 * invariant 2 to become three-way, and would silently alter four passes that
 * loop to `tris.n` -- the Morton bounding box, MFTLMeanEdgeLength, the vertex
 * normal scatter and MFTLRebuildTwins. mftl_communication.h §6 sets out that
 * argument in full; the patch is a separate object precisely so none of it
 * applies.
 *
 * ## 2. PERIODICITY IS HANDLED AT TWO SCALES, AND THEY ARE DIFFERENT PROBLEMS
 *
 * A centroid normalisation on a periodic domain is the place to worry, and the
 * worry has two halves that need two different answers.
 *
 * WITHIN one object -- a triangle straddling the seam. Its three vertices are
 * stored wrapped, so at x = 0 and x = L-eps, and a centroid built from the raw
 * coordinates lands in the middle of the domain. That corrupts its Morton code
 * (the triangle sorts into a cell it is nowhere near) and, worse, its AABB, which
 * then spans the whole domain and is a false positive against everything.
 *
 *   Fixed by gathering every primitive's current positions and forming
 *   minimum-image edges from corner 0. The three corners are therefore mutually
 *   unwrapped, its AABB is tight, and its centroid is the true one. Guarded by
 *   `test_periodic_morton_of_a_straddling_object`.
 *
 * BETWEEN two objects -- two bodies on opposite faces of the domain that are
 * periodic neighbours. No per-primitive fix can see this: each body is internally
 * consistent and they are genuinely L apart in stored coordinates.
 *
 *   Fixed in MFTL_IntersectObjects, which traverses the pair ONCE PER PERIODIC
 *   IMAGE of B -- up to 27 offsets {-L,0,+L}^3, each rejected or accepted by a
 *   single root-AABB test before any descent. A non-periodic domain has exactly
 *   one image and pays one extra compare. Guarded by
 *   `test_contact_across_a_periodic_face`.
 *
 * ## 3. THE TOLERANCE SCALES; IT IS NOT 1e-4
 *
 * Inflating every leaf AABB by a hardcoded 1e-4 and passing a hardcoded 1e-6 into
 * the proximity test is the obvious arrangement, and both are absolute lengths,
 * so the module then works at exactly one length scale: in a run carrying a
 * millimetre droplet and a metre body -- the case this library exists for --
 * 1e-4 is a tenth of the droplet's radius (every leaf overlaps every other) and
 * 1e-7 of the body's (contact is missed until the surfaces interpenetrate).
 *
 * MFTL_BVHBuildParams replaces both with ONE dimensionless number times the
 * object's bounding radius, floored by an absolute value:
 *
 *     tol = MAX(proximity_floor, proximity_factor * bounding_radius)
 *
 * and that single tol both inflates the leaves and is the gap the narrow phase
 * tests against, so the broad phase can never cull a pair the narrow phase would
 * have accepted. Guarded by `test_detection_at_two_disparate_length_scales`,
 * which runs a mm droplet and a m body in ONE ensemble and requires both.
 *
 * ## 4. WHO REPORTS A CONTACT, ON A DISTRIBUTED MESH
 *
 * A contact between a triangle of A owned by rank p and a triangle of B owned by
 * rank q has to be reported exactly once. The rule is:
 *
 *     THE RANK THAT OWNS THE PRIMITIVE FROM OBJECT A REPORTS THE PAIR.
 *
 * so a leaf pair is tested only when its A-side primitive is local. Three
 * consequences, all of them load-bearing:
 *
 *   - the BVH of A is built over OWNED triangles only. Including the
 *     redistribution halo would make rank p and rank q both report the same
 *     contact, exactly as writing ghosts into a .vtu double-counts the surface;
 *   - only object B's triangles ever travel. MFTLExtendSpatialHalo is
 *     one-directional per ordered pair, which halves what a symmetric exchange
 *     would send;
 *   - `MFTLEnsembleIntersectObjects` iterates j > i, so "A" and "B" are
 *     well-defined and every unordered pair is visited once.
 *
 * A serial run satisfies all of this trivially -- everything is owned and there
 * are no remote prims.
 */

#ifndef MFTL_BVH_H
#define MFTL_BVH_H

#include <stdbool.h>
#include <stdint.h>

#include "mftl_constants.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The node
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_BVHNode
 * @brief One node of the hierarchy, 32 bytes exactly.
 *
 * The AABB is FLOAT and deliberately so. A traversal is a pointer-chasing loop
 * whose entire cost is the latency of getting the next node into a register, so
 * halving the node -- 32 B against the 56 B a double AABB would need -- is the
 * single biggest lever available on it: two nodes per cache line instead of one.
 *
 * Rounding is not a correctness problem here because the bounds are ROUNDED
 * OUTWARD when they are written (see bvh_build.c): a float AABB is a conservative
 * superset of the double one, so the broad phase can produce a false positive --
 * which the narrow phase then rejects in double -- but never a false negative.
 * Every coordinate the answer actually depends on stays double, all the way
 * through MFTL_TriTriIntersect and MFTL_TriTriProximity.
 *
 * `payload` packs the two node kinds into eight bytes:
 *
 *   bit 0 == 1   LEAF      bits  1-31: prim_count
 *                          bits 32-63: first_prim (index into prim_indices)
 *   bit 0 == 0   INTERNAL  bits  1-63: left_child. The right child is
 *                          left_child + 1, always -- children are allocated as a
 *                          contiguous pair, which is what makes the sibling free
 *                          to address and the parent free of a second index.
 */
typedef struct MFTL_BVHNode
{
  float aabb_min[3];  ///< 12 B, rounded DOWN from the double bound.
  float aabb_max[3];  ///< 12 B, rounded UP from the double bound.
  uint64_t payload;  ///< 8 B, see above.
} MFTL_BVHNode;

_Static_assert(sizeof(MFTL_BVHNode) == 32, "MFTL_BVHNode must be exactly 32 bytes for cache alignment!");

/**
 * @brief True if @p n is a leaf.
 * @param n The node.
 * @return `true` iff the payload's low bit is set. That bit is the ONLY thing
 *         distinguishing the two payload layouts, which is why it is bit 0.
 */
static inline bool MFTL_BVHNodeIsLeaf(const MFTL_BVHNode *n) { return (n->payload & 1u) != 0u; }
/**
 * @brief Number of primitives in leaf @p n.
 * @param n The node.
 * @return The leaf's primitive count. **Undefined on an internal node** -- test
 *         MFTL_BVHNodeIsLeaf() first.
 */
static inline int MFTL_BVHLeafCount(const MFTL_BVHNode *n) { return (int) ((n->payload >> 1) & 0x7FFFFFFFu); }
/**
 * @brief First slot of leaf @p n in `prim_indices`.
 * @param n The node.
 * @return Index into `MFTL_BVH::prim_indices`. Undefined on an internal node.
 */
static inline int MFTL_BVHLeafFirst(const MFTL_BVHNode *n) { return (int) (n->payload >> 32); }
/**
 * @brief Left child of internal node @p n.
 * @param n The node.
 * @return Node index of the left child; the RIGHT child is that plus one, which
 *         is what makes a node 32 bytes rather than 40. Undefined on a leaf.
 */
static inline int MFTL_BVHLeftChild(const MFTL_BVHNode *n) { return (int) (n->payload >> 1); }

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Build parameters
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Default dimensionless contact envelope, as a fraction of the bounding radius. */
#define MFTL_BVH_DEFAULT_PROXIMITY_FACTOR 1.0e-3
/** @brief Default maximum primitives in a leaf. */
#define MFTL_BVH_DEFAULT_LEAF_SIZE 4

/**
 * @struct MFTL_BVHBuildParams
 * @brief Everything the build needs that is not the geometry.
 *
 * Initialise with MFTL_BVHDefaultParams() and override what you mean; a
 * zero-initialised struct is NOT valid, because a zero factor and a zero floor
 * together give a zero tolerance and turn every grazing contact into a coin
 * flip on the last bit.
 */
typedef struct MFTL_BVHBuildParams
{
  /**
   * @brief Contact envelope as a fraction of the object's bounding radius.
   *
   * This is the number that makes the module scale-free. The envelope is
   * `factor * R`, so the same value describes "a thousandth of the body" for a
   * 20 um droplet and for a 2 m hull, and a run holding both gets a sensible
   * answer for each without the caller tracking units.
   *
   * A lubrication or thin-film model sets this to the gap at which its force law
   * switches on, divided by the body radius.
   */
  double proximity_factor;

  /**
   * @brief Absolute lower bound on the envelope, in domain length units.
   *
   * Two jobs. It keeps a degenerate object (radius 0, a single point, an empty
   * partition) from producing a zero-width envelope, and it lets a caller who
   * genuinely has an absolute physical gap say so: set `proximity_factor` to 0
   * and this to the gap.
   *
   * Default is 0 -- pure relative scaling.
   */
  double proximity_floor;

  /**
   * @brief Maximum primitives per leaf. 4 is the measured sweet spot; larger
   *        trades traversal steps for narrow-phase tests, which is the right
   *        direction once the leaf batch is vectorised and the wrong one until
   *        then.
   */
  int leaf_size;

  /**
   * @brief Include the object's redistribution ghosts as primitives.
   *
   * MFTL_FALSE by default, and that default is a correctness statement rather
   * than a performance one: a ghost is a copy of a triangle another rank owns,
   * and that rank tests it too, so including them reports every halo contact
   * once per rank holding the triangle. See section 4 of the file comment.
   *
   * Set it only to look AT the halo -- debugging, or a single-rank run where the
   * distinction is empty anyway.
   */
  bool include_ghosts;
} MFTL_BVHBuildParams;

/**
 * @brief The parameter set a caller should start from.
 * @return #MFTL_BVH_DEFAULT_PROXIMITY_FACTOR, no absolute floor,
 *         #MFTL_BVH_DEFAULT_LEAF_SIZE, and ghosts excluded.
 */
static inline MFTL_BVHBuildParams MFTL_BVHDefaultParams(void)
{
  MFTL_BVHBuildParams p;
  p.proximity_factor = MFTL_BVH_DEFAULT_PROXIMITY_FACTOR;
  p.proximity_floor = 0.0;
  p.leaf_size = MFTL_BVH_DEFAULT_LEAF_SIZE;
  p.include_ghosts = MFTL_FALSE;
  return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The hierarchy
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_BVH
 * @brief A hierarchy over one body's triangles, plus its spatial halo patch.
 *
 * Owns its nodes and its permutation, and NOTHING ELSE. Both objects it reads
 * are borrowed and are never modified; the caller must keep them -- and, for the
 * patch, the MFTL_SpatialHalo that owns it -- alive for the BVH's lifetime.
 * MFTL_RefitLBVH must be handed the same object it was built from.
 */
typedef struct MFTL_BVH
{
  MFTL_BVHNode *nodes;  ///< [nodes_capacity] the tree; the root is `root_index`.
  int *prim_indices;  ///< [n_prims] primitives in Morton order; leaves address this.

  int nodes_used;  ///< Nodes actually populated.
  int nodes_capacity;  ///< Nodes allocated. A binary tree over n leaves needs at most 2n-1.
  int root_index;  ///< Always 0 for a non-empty tree.

  /* --- The primitive soup ------------------------------------------------- */

  int n_prims;  ///< n_local_prims + n_halo_prims.
  int n_local_prims;  ///< Primitives taken from `obj`; prim id == triangle index.
  int n_halo_prims;  ///< Primitives taken from `halo`; prim id - n_local_prims == triangle index.

  /* --- Provenance and build state ----------------------------------------- */

  const MFTL_Object *obj;  ///< BORROWED. Never written through.

  /**
   * @brief The spatial halo patch for this body, or MFTL_NULL.
   *
   * An ordinary MFTL_Object holding the triangles neighbouring ranks sent
   * because they are near this rank's share of some other body -- see
   * MFTLExtendSpatialHalo (mftl_communication.h §6). BORROWED: it belongs to the
   * MFTL_SpatialHalo, and the caller must keep that alive for the BVH's lifetime.
   *
   * The BVH stores no geometry of its own for these, and does not need to: a
   * patch is a real object, so its current corners come from positions and
   * connectivity through exactly the accessor the local triangles use. The local/remote
   * distinction disappears from everything except one index comparison.
   */
  const MFTL_Object *halo;

  MFTL_BVHBuildParams params;  ///< The parameters this tree was built with, kept for the refit.

  double tolerance;  ///< The resolved envelope: MAX(floor, factor * bounding_radius).

  /**
   * @brief Half the LONGEST SIDE of the primitive AABB at the last fit.
   *
   * Not half the diagonal, which would also bound the object but equals
   * sqrt(3) * R for a sphere of radius R. `proximity_factor` is documented as a
   * fraction of the body, so this is the measure that makes 1e-3 of a 1 mm
   * droplet come out as 1 um rather than 1.73 um.
   */
  double bounding_radius;

  /**
   * @brief Bounding radius of the WHOLE body, when it is known. 0 = not known.
   *
   * ── Why a partition-local radius is not good enough ───────────────────────
   *
   * `bounding_radius` is measured from the primitives this rank holds, which is
   * a fraction of the body. Scaling the contact envelope by it would make the
   * envelope -- and therefore which contacts are detected -- a function of the
   * PROCESS COUNT: split a droplet over eight ranks and every rank derives a
   * smaller radius, so the same simulation detects contact later at np = 8 than
   * at np = 1. A physical predicate must not do that.
   *
   * MFTLSpatialHaloResolveRadii measures it in one MPI_Allreduce over the
   * body's global extent, covering every body in its request list in a single
   * collective, and MFTLBVHRefit prefers it whenever it is set. At nproc == 1 it
   * equals the local radius, so the serial and parallel answers agree by
   * construction rather than by luck.
   *
   * A caller using MFTL_GenerateLBVH on its own gets the partition-local radius
   * and should set this itself if that distinction matters to it.
   */
  double radius_hint;

  /**
   * @brief Primitive count the topology was built for.
   *
   * MFTL_RefitLBVH refuses -- rather than corrupting -- if the object's triangle
   * count has changed since the build, because the leaves address primitives by
   * index and a re-triangulation renumbers them. See its documentation.
   */
  int built_for_n_prims;
} MFTL_BVH;

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Primitive access
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Writes the three corners of primitive @p p into @p out, unwrapped.
 *
 * The one accessor both primitive kinds go through, so no consumer branches on
 * local-versus-remote. The corners are seam-consistent and gathered from the
 * source object's current positions with minimum-image edges.
 *
 * @param      bvh The hierarchy.
 * @param      p   Primitive index in `[0, n_prims)`.
 * @param[out] out Row \f$k\f$ is corner \f$k\f$.
 */
void MFTL_BVHPrimVerts(const MFTL_BVH *bvh, int p, double out[3][3]);

/**
 * @brief True if primitive @p p is a triangle of `bvh->obj` rather than a remote copy.
 * @param bvh The hierarchy.
 * @param p   Primitive index.
 * @return `true` iff \f$p < n_\mathrm{local\,prims}\f$. The whole local/remote
 *         distinction is this one comparison.
 */
static inline bool MFTL_BVHPrimIsLocal(const MFTL_BVH *bvh, int p) { return p < bvh->n_local_prims; }

/**
 * @brief Global triangle id of primitive @p p, whichever rank owns it.
 *
 * The identity a callback should record: a local pool index means nothing to the
 * rank that owns the other side of the contact, and is invalidated by the next
 * redistribution or remesh in any case.
 *
 * @param bvh The hierarchy.
 * @param p   Primitive index.
 * @return The triangle's global id.
 */
int MFTL_BVHPrimGlobalId(const MFTL_BVH *bvh, int p);

/**
 * @brief The rank owning primitive @p p.
 * @param bvh The hierarchy.
 * @param p   Primitive index.
 * @return The owning rank. For a local primitive this is the calling rank.
 */
int MFTL_BVHPrimOwner(const MFTL_BVH *bvh, int p);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Build, refit, destroy
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Builds a hierarchy over @p obj: Morton codes, radix split, AABB refit.
 *
 * 1. One primitive per triangle -- owned only unless `params.include_ghosts`.
 * 2. A 63-bit Morton key per primitive from its SEAM-CONSISTENT centroid,
 *    quantised against the primitive bounding box (MFTLMorton3D, mftl_object.h --
 *    the same key the pool renumbering uses, at 21 bits per axis rather than the
 *    customary 10, which is what stops a million-triangle mesh putting thousands
 *    of primitives on one code).
 * 3. A stable radix sort of (key, prim id), then a top-down binary split at the
 *    highest differing bit.
 * 4. Bottom-up AABB refit, inflated by the resolved tolerance.
 *
 * @param obj    The object. Read only; not retained beyond the returned BVH's
 *               borrowed pointer, which the caller must keep alive.
 * @param params MFTL_NULL for MFTL_BVHDefaultParams().
 * @return A new BVH, or MFTL_NULL only if @p obj is MFTL_NULL. Free with
 *         MFTL_DestroyBVH().
 *
 * @note An object with NO local triangles yields a valid EMPTY hierarchy rather
 *       than MFTL_NULL, and that is deliberate. A rank can legitimately hold
 *       part of body A and none of body B while still needing B's triangles from
 *       its neighbours -- there has to be somewhere to put them, and a NULL
 *       return would also make the request list rank-dependent and desynchronise
 *       the collective in MFTLExtendSpatialHalo.
 * @note The bounding radius, and therefore the tolerance, is measured over THIS
 *       RANK's primitives. MFTL_BVHSetGlobalRadius replaces it with the body's
 *       global radius; see MFTL_BVH::radius_hint for why that matters.
 * @note Local (serial) operation -- no MPI. Spatial ghosting is a separate,
 *       collective call that runs AFTER this one and rebuilds the tree.
 */
MFTL_BVH *MFTL_GenerateLBVH(const MFTL_Object *obj, const MFTL_BVHBuildParams *params);

/**
 * @brief Refit-only fast path: the geometry moved, the triangulation did not.
 *
 * Skips the centroid pass, the Morton codes, the radix sort and the topology
 * build, and re-runs only the bottom-up AABB pass over the existing tree. That
 * is the whole per-timestep cost of a BVH in an advection loop, where vertices
 * move every step and the connectivity changes only when the remesher runs.
 *
 * The refit is a REVERSE LINEAR SCAN, not a recursion: the build allocates a
 * child pair by post-incrementing `nodes_used`, so every child has a strictly
 * higher index than its parent, and walking the node array backwards therefore
 * visits both children before the parent they feed. One pass, contiguous, no
 * stack, no call overhead -- and it is why the refit is worth splitting out at
 * all.
 *
 * The tree keeps the topology it was built with, so its quality degrades as the
 * mesh deforms away from the configuration the Morton sort saw. Rebuild after a
 * redistribution, after a remesh, and whenever the deformation stops being small.
 *
 * @param bvh The hierarchy to refit, in place.
 * @param obj Must be the SAME object the BVH was generated from.
 * @return MFTL_SUCCESS, or MFTL_ERROR if the triangulation changed underneath it
 *         (the primitive count no longer matches `built_for_n_prims`) -- in which
 *         case the caller owes a full MFTL_GenerateLBVH. The BVH is left
 *         untouched rather than half-refitted, so the fallback costs nothing.
 *
 * @note MFTL_ERROR is a SUPPORTED outcome, not a programming error: "refit if
 *       you can, otherwise rebuild" is a legitimate way to drive a loop that
 *       remeshes occasionally, and it is deliberately not asserted on. A wrong
 *       OBJECT is a different matter and is asserted.
 *
 * @note Remote primitives keep the positions they arrived with. A refit does not
 *       re-run MFTLExtendSpatialHalo, so a loop that refits every step but
 *       ghosts every N steps has stale remote geometry in between -- which is
 *       correct only if the neighbours' surfaces move less than the tolerance in
 *       that window.
 * @note Local (serial) operation -- no MPI.
 */
int MFTL_RefitLBVH(MFTL_BVH *bvh, const MFTL_Object *obj);

/**
 * @brief Destroys a BVH and everything it owns. Safe on MFTL_NULL.
 *
 * @param bvh The hierarchy, or #MFTL_NULL. The borrowed object and halo patch
 *            are NOT freed -- they belong to the caller.
 */
void MFTL_DestroyBVH(MFTL_BVH *bvh);

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Attaching a spatial halo
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Adds the spatial halo patch of this body to the hierarchy and rebuilds it.
 *
 * The patch comes from MFTLExtendSpatialHalo (mftl_communication.h §6) -- the
 * BVH does not fetch it and does not own it. Splitting it that way is what keeps
 * this module free of MPI entirely: everything in src/bvh/ is a local operation,
 * and the one collective the pipeline needs belongs to the module that owns the
 * wire.
 *
 * The tree is rebuilt rather than extended, because the Morton order is a
 * function of the WHOLE primitive set -- bolting leaves onto a sorted array
 * stops it being sorted. The patch's triangles take part in the sort on equal
 * terms with the local ones, so the tree has no seam where the rank boundary was.
 *
 * @param bvh   The hierarchy, modified in place.
 * @param patch MFTLSpatialHaloPatch(halo, body_idx), or MFTL_NULL to drop the
 *              halo again. BORROWED -- it must outlive the BVH.
 * @return MFTL_SUCCESS, or MFTL_ERROR on a NULL hierarchy.
 *
 * @note Local (serial) operation -- no MPI.
 * @note Halo primitives are never eligible as the A side of a pair, so attaching
 *       a patch can only ever ADD contacts that this rank is responsible for
 *       reporting; it cannot duplicate one. See section 4 of the file comment.
 */
int MFTL_BVHSetHalo(MFTL_BVH *bvh, const MFTL_Object *patch);

/**
 * @brief Sets the body's GLOBAL bounding radius, so the envelope stops depending
 *        on the partition.
 *
 * Pass MFTLSpatialHaloResolveRadii's result for this body. See
 * MFTL_BVH::radius_hint for why a partition-local radius is not merely imprecise.
 * Refits the tree, because the leaf inflation changes with it.
 *
 * @param bvh    The hierarchy. Refitted in place.
 * @param radius Half the longest side of the body's global bounding box. A value
 *               \f$\le 0\f$ clears the hint and returns to the local measurement.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL hierarchy.
 */
int MFTL_BVHSetGlobalRadius(MFTL_BVH *bvh, double radius);

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Traversal
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Called once per primitive pair that passed the narrow phase.
 *
 * @param objA     Object A, as handed to MFTL_IntersectObjects.
 * @param primA    Primitive index in bvhA. Use MFTL_BVHPrimGlobalId() for an
 *                 identity that means something on another rank.
 * @param objB     Object B.
 * @param primB    Primitive index in bvhB.
 * @param user_data Whatever the caller passed through.
 */
typedef void (*IntersectionCallback)(const MFTL_Object *objA, int primA, const MFTL_Object *objB, int primB, void *user_data);

/**
 * @brief Reports every contacting primitive pair between two hierarchies.
 *
 * ── The traversal is an explicit stack, not a recursion ────────────────────
 *
 * A stack of (nodeA, nodeB) pairs, descending into whichever side has the larger
 * AABB so the two trees stay balanced against each other. Three things that buys
 * over a four-argument recursive descent:
 *
 *   - no call frame per node pair, and the hot state (both node pointers, the
 *     shift) stays in registers across the loop;
 *   - the stack is one contiguous array, so the "what to do next" data has the
 *     same locality as the nodes;
 *   - LEAF PAIRS ARE BATCHED. A leaf-leaf hit does not call the narrow phase
 *     immediately; it appends to a fixed-size batch which is flushed when full.
 *     That is what makes a future SIMD narrow phase a change to ONE function
 *     (`flush_batch` in bvh_traverse.c) rather than a rewrite of the traversal:
 *     the batch is already a contiguous array of independent tests.
 *
 * ── Periodicity ────────────────────────────────────────────────────────────
 *
 * The pair is traversed once per periodic image of B whose shifted root AABB
 * overlaps A's -- see section 2 of the file comment. Images are generated from
 * the domain's per-axis periodicity, so a non-periodic domain does exactly one
 * traversal.
 *
 * ── Which pairs are reported ───────────────────────────────────────────────
 *
 * Only pairs whose A-side primitive is LOCAL, so that a distributed run reports
 * each contact exactly once (section 4). The B side may be local or remote.
 *
 * @param objA      The object @p bvhA was built from.
 * @param bvhA       Hierarchy over object A. Refitted if its geometry moved.
 * @param objB      The object @p bvhB was built from. May be the same object as
 *                  @p objA only if the two BVHs are distinct; a BVH against
 *                  itself is not a self-intersection query and is not supported.
 * @param bvhB       Hierarchy over object B.
 * @param callback  Invoked per contacting pair. Must not modify either object.
 * @param user_data Passed through to @p callback untouched.
 *
 * @rank_local The communication happened in MFTLExtendSpatialHalo().
 */
void MFTL_IntersectObjects(const MFTL_Object *objA, MFTL_BVH *bvhA, const MFTL_Object *objB, MFTL_BVH *bvhB, IntersectionCallback callback, void *user_data);

/**
 * @brief The whole pipeline for an ensemble: ghost, build, traverse, destroy.
 *
 * @param ens      The ensemble. Collective; all ranks, same object list.
 * @param params   Build parameters, MFTL_NULL for the defaults.
 * @param callback Invoked per contacting pair.
 * @param user_data Passed through.
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments.
 *
 * @note Tests every unordered pair i < j of distinct objects. A single object is
 *       therefore a no-op: self-intersection of one surface is a different
 *       algorithm (adjacent triangles share vertices and always "touch"), and it
 *       is deliberately not what this does.
 * @note Collective.
 */
int MFTLEnsembleIntersectObjects(MFTL_Ensemble *ens, const MFTL_BVHBuildParams *params, IntersectionCallback callback, void *user_data);

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. Narrow phase
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Exact Guigue-Devillers triangle-triangle intersection test.
 *
 * Fully double precision, and it stays that way: the broad phase is float
 * because a conservative bound may be loose, and this is not because a
 * misclassified orientation is a wrong answer with no error term.
 *
 * @param A [3][3] corners of the first triangle, unwrapped.
 * @param B [3][3] corners of the second, unwrapped and in the SAME periodic
 *        image as @p A -- the caller applies the shift.
 * @return 1 if they intersect (coplanar overlap included), 0 otherwise.
 */
int MFTL_TriTriIntersect(const double A[3][3], const double B[3][3]);

/**
 * @brief Intersection, or a closest approach within @p tolerance.
 *
 * Runs MFTL_TriTriIntersect first and falls back to the exact minimum distance
 * between the two closed triangles -- nine edge-edge pairs and six point-face
 * projections, which is a complete cover of the possible witness pairs.
 *
 * @param A         Corners of the first triangle, unwrapped.
 * @param B         Corners of the second, unwrapped and in the SAME periodic
 *                  image as @p A -- the caller applies the shift.
 * @param tolerance Gap at which contact is declared. Squared internally, so it
 *                  may be zero (touching only).
 * @return 1 on contact, 0 otherwise.
 *
 * @measured{176 ns/pair, against 31 ns for MFTL_TriTriIntersect(). The fifteen
 *           distance sub-tests are where the time goes.}
 */
int MFTL_TriTriProximity(const double A[3][3], const double B[3][3], double tolerance);

/**
 * @brief Squared minimum distance between two closed triangles. 0 if they meet.
 *
 * Exposed because a contact model usually wants the gap, not just the predicate,
 * and recomputing it in the callback would repeat all fifteen sub-tests.
 *
 * @param A Corners of the first triangle, unwrapped.
 * @param B Corners of the second, in the same periodic image as @p A.
 * @return \f$d^2\f$, the squared minimum distance. Exactly 0 if they touch or
 *         intersect.
 */
double MFTL_TriTriDistanceSq(const double A[3][3], const double B[3][3]);

/**
 * @brief Signed volume of the tetrahedron (pa, pb, pc, pd), times six.
 *
 * Positive when pd is BELOW the plane of pa, pb, pc as seen along its normal.
 *
 * ── Why this one is adaptive and nothing else in the module is ─────────────
 *
 * This is the only predicate in the file whose SIGN, rather than magnitude,
 * decides the answer, and Guigue-Devillers asks for that sign up to EIGHT times
 * per triangle pair -- six to classify the corners against the two planes, and
 * two in check_min_max. The six-way permutation cascade evaluates none of its
 * own: it branches on signs that are already computed.
 *
 * Everything else here -- distances, areas, interval endpoints --
 * degrades continuously under rounding: a distance that is wrong in the last
 * ulp gives a contact decision that is wrong only for pairs sitting exactly on
 * the tolerance, which is a tie the caller has already accepted by choosing a
 * tolerance. A sign that is wrong makes the algorithm take a branch for a
 * configuration that does not exist, and it can then report a clean miss for
 * two triangles that overlap.
 *
 * The implementation is Shewchuk's: a floating-point estimate plus a
 * relative-error bound, and exact expansion arithmetic ONLY when the estimate
 * falls inside that bound. On non-degenerate input it is the plain determinant
 * plus one compare.
 *
 * Measured (examples/orient3d_predicate_profile.c): 2.00x the plain determinant
 * on general position, but AT MOST 6.2 % of a full MFTL_TriTriProximity call,
 * which is dominated by its fifteen distance sub-tests. The plain form did not
 * misclassify once in 20 000 samples at gaps down to 1e-12 of the body size --
 * three orders below where a lubrication model works -- so this buys robustness
 * at the rounding limit rather than fixing a failure that shows up in practice.
 * The full argument, both directions, is at the top of src/bvh/bvh_predicates.c.
 *
 * @param pa First point of the reference plane.
 * @param pb Second point.
 * @param pc Third point.
 * @param pd The point being classified.
 * @return \f$6V\f$, signed. Positive when @p pd is BELOW the plane of
 *         @p pa, @p pb, @p pc as seen along its normal. The SIGN is exact; the
 *         magnitude is not claimed to be.
 */
double MFTL_Orient3D(const double pa[3], const double pb[3], const double pc[3], const double pd[3]);

/**
 * @brief The non-adaptive determinant, for comparison and for benchmarks.
 *
 * Same value as MFTL_Orient3D() up to rounding, and the same SIGN except on
 * near-degenerate input -- which is exactly the input this library's contact
 * cases are made of. Not used by the traversal.
 *
 * @param pa First point of the reference plane.
 * @param pb Second point.
 * @param pc Third point.
 * @param pd The point being classified.
 * @return \f$6V\f$ from the plain \f$3\times3\f$ determinant. The sign is not
 *         guaranteed near degeneracy.
 */
double MFTL_Orient3DFast(const double pa[3], const double pb[3], const double pc[3], const double pd[3]);

#endif  // MFTL_BVH_H
