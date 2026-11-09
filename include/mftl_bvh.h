/**
 * @file mftl_bvh.h
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Linear BVH over triangle primitives, exact triangle-triangle
 *        predicates, and (distributed) object-object contact detection.
 *
 * The broad phase (a tree of float boxes) selects triangle pairs worth an exact
 * test; the narrow phase (exact, double precision) decides whether they touch.
 * Four decisions shape the module:
 *
 * 1. **A BVH is built over a primitive soup.** Primitives `[0, n_local_prims)`
 *    are triangles of `obj`, `[n_local_prims, n_prims)` triangles of `halo`, a
 *    spatial-halo patch (an ordinary #MFTL_Object, mftl_communication.h §6).
 *    Both are borrowed and `const`; the tree stores no geometry, and both kinds
 *    are read through MFTLGetTriangleCorners().
 * 2. **Periodicity at two scales.** Within an object, every primitive's corners
 *    are mutually unwrapped (minimum-image edges from corner 0), so a
 *    seam-straddling triangle has a tight box and its true centroid. Between
 *    objects, MFTL_IntersectObjects() traverses the pair once per periodic image
 *    of B (up to 27), each gated by one root-box test.
 * 3. **The envelope scales with the body:**
 *    `tol = MAX(proximity_floor, proximity_factor * R)`, with R the body's
 *    radius; the same tol inflates the leaves and is the gap the narrow phase
 *    tests, so the broad phase never culls a pair the narrow phase would accept.
 * 4. **A distributed contact is reported once**, by the rank owning the A-side
 *    primitive: A trees contain owned triangles only, only B triangles travel,
 *    and pairs are ordered `body_a < body_b` by body id.
 *
 * All of src/bvh/ is local; the one exchange (the spatial halo) belongs to the
 * communication module. Theory: src/bvh/README.md.
 */

#ifndef MFTL_BVH_H
#define MFTL_BVH_H

#include <stdbool.h>
#include <stdint.h>

#include "mftl_constants.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The node
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_BVHNode
 * @brief One node of the hierarchy, 32 bytes (two per cache line).
 *
 * The box is `float`, rounded OUTWARD from the double bounds, so it is a
 * conservative superset: the broad phase may produce false positives (rejected
 * by the double-precision narrow phase) but never false negatives.
 *
 * `payload`: bit 0 set -> LEAF, bits 1-31 primitive count, bits 32-63 first slot
 * in `prim_indices`; bit 0 clear -> INTERNAL, bits 1-63 left child, and the
 * right child is always left + 1 (children are allocated as a pair).
 */
typedef struct MFTL_BVHNode
{
  float aabb_min[3];  ///< 12 B, rounded DOWN from the double bound.
  float aabb_max[3];  ///< 12 B, rounded UP from the double bound.
  uint64_t payload;  ///< 8 B, see above.
} MFTL_BVHNode;

MFTL_STATIC_ASSERT(sizeof(MFTL_BVHNode) == 32, "MFTL_BVHNode must be exactly 32 bytes for cache alignment!");

/**
 * @brief True if @p n is a leaf.
 * @param n The node.
 * @return `true` iff bit 0 of the payload is set.
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
 * @brief Left child of internal node @p n; the right child is that plus one.
 * @param n The node.
 * @return Node index of the left child. Undefined on a leaf.
 */
static inline int MFTL_BVHLeftChild(const MFTL_BVHNode *n) { return (int) (n->payload >> 1); }

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Build parameters
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Default dimensionless contact envelope, as a fraction of the bounding radius. */
#define MFTL_BVH_DEFAULT_PROXIMITY_FACTOR 1.0e-3
/** @brief Default maximum primitives in a leaf. */
#define MFTL_BVH_DEFAULT_LEAF_SIZE 4

struct MFTL_BVH;  // defined in section 3; the pair filter below takes pointers to it

/**
 * @struct MFTL_BVHBuildParams
 * @brief Build and query parameters. Start from MFTL_BVHDefaultParams(); a
 *        zero-initialised struct gives a zero envelope.
 */
typedef struct MFTL_BVHBuildParams
{
  /**
   * @brief Contact envelope as a fraction of the body's bounding radius R
   *        (default 1e-3).
   *
   * Scale-free: the same value means a thousandth of the body for a 20 um
   * droplet and for a 2 m hull. A thin-film model sets its activation gap
   * divided by R.
   */
  double proximity_factor;

  /**
   * @brief Absolute lower bound on the envelope, in length units (default 0).
   *
   * Keeps a degenerate body from getting a zero envelope; with
   * `proximity_factor = 0` it states an absolute physical gap.
   */
  double proximity_floor;

  /** @brief Maximum primitives per leaf (default 4). */
  int leaf_size;

  /**
   * @brief Also use the object's redistribution ghosts as primitives (default
   *        false).
   *
   * A ghost is tested by its owner too, so enabling this reports halo contacts
   * once per holder; use it only to inspect the halo.
   */
  bool include_ghosts;

  /**
   * @brief Also test every body against itself (default false; read by
   *        MFTLEnsembleIntersectObjects() only).
   *
   * Pairs of triangles sharing a vertex are dropped (they touch by
   * construction), so what remains are non-adjacent parts of one surface that
   * are close -- two sheets of a fold, the sides of a thin neck: the query
   * topology changes are detected with. Each unordered pair is reported once, by
   * the owner of the triangle with the smaller global id. Must be the same on
   * every rank.
   */
  bool self_pairs;

  /**
   * @brief Optional cheap predicate run on every candidate pair before the exact
   *        narrow phase (default #MFTL_NULL).
   *
   * Return `false` to skip the pair. The topology-change detector, for
   * instance, keeps only triangles with opposing normals, rejecting most
   * candidates with a dot product. Read from the A-side tree and called with
   * the callback's `user_data`. It must be a pure function of the two
   * primitives, or the reported set would depend on the partition.
   */
  bool (*pair_filter)(const struct MFTL_BVH *bvhA, int primA, const struct MFTL_BVH *bvhB, int primB, void *user_data);
} MFTL_BVHBuildParams;

/**
 * @brief The default parameters.
 * @return Factor #MFTL_BVH_DEFAULT_PROXIMITY_FACTOR, no floor, leaf size
 *         #MFTL_BVH_DEFAULT_LEAF_SIZE, ghosts excluded, no self pairs, no filter.
 */
static inline MFTL_BVHBuildParams MFTL_BVHDefaultParams(void)
{
  MFTL_BVHBuildParams p;
  p.proximity_factor = MFTL_BVH_DEFAULT_PROXIMITY_FACTOR;
  p.proximity_floor = 0.0;
  p.leaf_size = MFTL_BVH_DEFAULT_LEAF_SIZE;
  p.include_ghosts = MFTL_FALSE;
  p.self_pairs = MFTL_FALSE;
  p.pair_filter = MFTL_NULL;
  return p;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The hierarchy
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_BVH
 * @brief A hierarchy over one body's triangles, optionally plus its spatial-halo
 *        patch.
 *
 * Owns only its nodes and its primitive permutation. The object and the patch
 * are borrowed and must outlive the BVH; MFTL_RefitLBVH() must be given the same
 * object.
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
   * @brief The spatial-halo patch of this body, or #MFTL_NULL. BORROWED from the
   *        #MFTL_SpatialHalo, which must outlive the BVH.
   */
  const MFTL_Object *halo;

  MFTL_BVHBuildParams params;  ///< The parameters this tree was built with, kept for the refit.

  double tolerance;  ///< The resolved envelope: MAX(floor, factor * bounding_radius).

  /**
   * @brief Half the LONGEST SIDE of the primitive bounding box at the last fit
   *        (equals R for a sphere; half the diagonal would be \f$\sqrt3 R\f$).
   */
  double bounding_radius;

  /**
   * @brief Radius of the WHOLE body when known (0 = unknown); preferred over
   *        `bounding_radius` by the fit.
   *
   * `bounding_radius` covers only this rank's primitives, so an envelope scaled
   * by it would depend on the process count. MFTLSpatialHaloResolveRadii()
   * provides the global value (a per-body reduction over the holders, published
   * one hop further); set it with MFTL_BVHSetGlobalRadius().
   */
  double radius_hint;

  /** @brief Primitive count the tree was built for; MFTL_RefitLBVH() refuses if it changed. */
  int built_for_n_prims;
} MFTL_BVH;

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Primitive access
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The three corners of primitive @p p, mutually unwrapped
 *        (MFTLGetTriangleCorners() of its source object).
 *
 * @param      bvh The hierarchy.
 * @param      p   Primitive index in `[0, n_prims)`.
 * @param[out] out Row k is corner k.
 */
void MFTL_BVHPrimVerts(const MFTL_BVH *bvh, int p, double out[3][3]);

/**
 * @brief The object and triangle a primitive is: `bvh->obj` for a local
 *        primitive, the halo patch for a remote one.
 *
 * A callback that needs more than corners (vertex ids, body id) reads them from
 * the returned object; a patch carries the body's real ids.
 *
 * @param      bvh The hierarchy.
 * @param      p   Primitive index in `[0, n_prims)`.
 * @param[out] tri Triangle index within the returned object.
 * @return The object the triangle lives in.
 */
const MFTL_Object *MFTL_BVHPrimTriangle(const MFTL_BVH *bvh, int p, int *tri);

/**
 * @brief True if primitive @p p is a triangle of `bvh->obj`, not a halo copy.
 * @param bvh The hierarchy.
 * @param p   Primitive index.
 * @return `true` iff \f$p < n_\mathrm{local\,prims}\f$.
 */
static inline bool MFTL_BVHPrimIsLocal(const MFTL_BVH *bvh, int p) { return p < bvh->n_local_prims; }

/**
 * @brief Global triangle id of primitive @p p -- the identity a callback should
 *        record (pool indices mean nothing on the other rank of a contact).
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
 * @brief Builds a linear BVH over @p obj.
 *
 * 1. One primitive per owned triangle (all triangles with `include_ghosts`).
 * 2. A 63-bit Morton key (MFTLMorton3D(), 21 bits per axis) of each primitive's
 *    seam-consistent, UNWRAPPED centroid, quantised with one common scale.
 * 3. A stable radix sort of (key, primitive) and a top-down split at the highest
 *    differing bit of the 95-bit key (key, index).
 * 4. Bottom-up box fit, then inflation by the resolved tolerance.
 *
 * An object without primitives yields a valid EMPTY tree (not #MFTL_NULL): a
 * rank may hold nothing of a body and still receive a patch of it.
 *
 * @param obj    The object; borrowed for the BVH's lifetime.
 * @param params #MFTL_NULL for MFTL_BVHDefaultParams().
 * @return A new BVH (free with MFTL_DestroyBVH()), or #MFTL_NULL if @p obj is
 *         #MFTL_NULL.
 *
 * @rank_local The tolerance uses this rank's radius until
 *             MFTL_BVHSetGlobalRadius() is called.
 */
MFTL_BVH *MFTL_GenerateLBVH(const MFTL_Object *obj, const MFTL_BVHBuildParams *params);

/**
 * @brief Refits the boxes after the vertices moved; tree and order are kept.
 *
 * Skips centroids, keys, sort and split; the refit is a reverse linear scan over
 * the nodes (every child index exceeds its parent's). The tree quality degrades
 * as the mesh deforms away from the configuration it was sorted for: rebuild
 * after redistribution, remeshing or large deformation. Halo primitives keep the
 * positions they arrived with.
 *
 * @param bvh The hierarchy, refitted in place.
 * @param obj The object it was built from (asserted).
 * @return #MFTL_SUCCESS, or #MFTL_ERROR -- a supported outcome, not asserted --
 *         when the primitive counts changed (the triangulation was modified);
 *         the BVH is then untouched and the caller rebuilds.
 *
 * @rank_local
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
 * @brief Attaches the spatial-halo patch of this body and REBUILDS the tree over
 *        local and patch primitives together.
 *
 * Rebuilt, not extended: the Morton order is a function of the whole primitive
 * set. Patch primitives are never the reporting A side, so attaching a patch
 * cannot duplicate a contact.
 *
 * @param bvh   The hierarchy.
 * @param patch MFTLSpatialHaloPatch() for this body, or #MFTL_NULL to detach.
 *              BORROWED; must outlive the BVH.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL hierarchy.
 *
 * @rank_local
 */
int MFTL_BVHSetHalo(MFTL_BVH *bvh, const MFTL_Object *patch);

/**
 * @brief Sets the body's GLOBAL radius (MFTLSpatialHaloRadius()) so the envelope
 *        no longer depends on the partition, and refits.
 *
 * @param bvh    The hierarchy.
 * @param radius Half the longest side of the body's global bounding box; a
 *               value \f$\le 0\f$ returns to the local measurement.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL hierarchy.
 */
int MFTL_BVHSetGlobalRadius(MFTL_BVH *bvh, double radius);

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Traversal
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Called once per primitive pair that passed the narrow phase.
 *
 * Receives the HIERARCHIES, because a primitive index is meaningful only with
 * its tree (past `n_local_prims` it names a patch triangle); resolve primitives
 * with MFTL_BVHPrimTriangle(), MFTL_BVHPrimVerts() or MFTL_BVHPrimGlobalId().
 *
 * @param bvhA      The A-side hierarchy; primA is always LOCAL to it.
 * @param primA     Primitive index in @p bvhA.
 * @param bvhB      The B-side hierarchy.
 * @param primB     Primitive index in @p bvhB; local or remote.
 * @param user_data Whatever the caller passed through.
 */
typedef void (*IntersectionCallback)(const MFTL_BVH *bvhA, int primA, const MFTL_BVH *bvhB, int primB, void *user_data);

/**
 * @brief Reports every contacting primitive pair between two hierarchies.
 *
 * For every periodic image of B whose shifted root box overlaps A's root, a
 * dual-tree descent over an explicit stack of node pairs splits the node with
 * the larger box volume; leaf-leaf candidates are collected into batches of 64
 * and passed to MFTL_TriTriProximity() at tolerance `MAX(tol_A, tol_B)`. Only
 * pairs with a LOCAL A-side primitive are tested, so a distributed run reports
 * each contact once.
 *
 * @param objA      The object @p bvhA was built from.
 * @param bvhA      Hierarchy over object A.
 * @param objB      The object @p bvhB was built from. May equal @p objA with a
 *                  distinct second tree: the self-proximity query, where pairs
 *                  sharing a vertex are dropped and each unordered pair is kept
 *                  once (smaller triangle global id first).
 * @param bvhB      Hierarchy over object B.
 * @param callback  Invoked per contacting pair; must not modify either object.
 * @param user_data Passed through to @p callback.
 *
 * @rank_local
 */
void MFTL_IntersectObjects(const MFTL_Object *objA, MFTL_BVH *bvhA, const MFTL_Object *objB, MFTL_BVH *bvhB, IntersectionCallback callback, void *user_data);

/**
 * @brief Contact detection for a whole ensemble: build, measure, exchange, attach,
 *        traverse, destroy.
 *
 * 1. MFTL_GenerateLBVH() for every body this rank holds;
 * 2. MFTLSpatialHaloResolveRadii() and MFTL_BVHSetGlobalRadius() -- the envelope
 *    is a fraction of the body, whose size is only known collectively;
 * 3. MFTLExtendSpatialHaloNearBodies() with those envelopes, which returns the
 *    patches AND the candidate body pairs (no all-pairs list is enumerated);
 * 4. MFTL_BVHSetHalo() per patched body; a body held nowhere here but patched
 *    gets a tree over a shared empty object;
 * 5. MFTL_IntersectObjects() per candidate pair (A = smaller body id). A self
 *    pair uses a second, tight (zero-envelope), patch-free A-side tree.
 *
 * Cost: trees `O(bodies held)`, traversals `O(candidate pairs)`; two collective
 * steps independent of the body count.
 *
 * @param ens       The ensemble.
 * @param params    Build parameters, or #MFTL_NULL for the defaults.
 * @param callback  Invoked per contacting pair.
 * @param user_data Passed through.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments.
 *
 * @collective Guarded by `n_bodies_global` (at least 2, or 1 with
 *             `self_pairs`). A contact between bodies whose holders are more
 *             than one cell apart is not found: the halo only ships along
 *             `comm_graph`.
 */
int MFTLEnsembleIntersectObjects(MFTL_Ensemble *ens, const MFTL_BVHBuildParams *params, IntersectionCallback callback, void *user_data);

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. Narrow phase
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Exact triangle-triangle intersection test (Guigue & Devillers 2003).
 *
 * Every branch is taken on the sign of MFTL_Orient3D(), whose sign is exact; no
 * intersection point is constructed. Coplanar pairs get a complete 2D test.
 *
 * @param A [3][3] corners of the first triangle.
 * @param B [3][3] corners of the second, in the SAME periodic image as @p A.
 * @return 1 if the closed triangles intersect (coplanar overlap included), else 0.
 */
int MFTL_TriTriIntersect(const double A[3][3], const double B[3][3]);

/**
 * @brief Contact predicate: intersection, or a minimum distance within
 *        @p tolerance.
 *
 * MFTL_TriTriIntersect() first (a piercing pair has positive distance to every
 * edge and vertex, so the distance test alone would miss it), then the exact
 * minimum distance of two disjoint closed triangles: nine edge-edge and six
 * point-face terms.
 *
 * @param A         Corners of the first triangle.
 * @param B         Corners of the second, in the same periodic image as @p A.
 * @param tolerance Gap at which contact is declared, >= 0.
 * @return 1 on contact, 0 otherwise.
 *
 * @measured{176 ns/pair, against 31 ns for MFTL_TriTriIntersect() alone.}
 */
int MFTL_TriTriProximity(const double A[3][3], const double B[3][3], double tolerance);

/**
 * @brief Squared minimum distance between two closed triangles; 0 if they meet.
 *
 * @param A Corners of the first triangle.
 * @param B Corners of the second, in the same periodic image as @p A.
 * @return \f$d^2\f$, exactly 0 if they touch or intersect.
 */
double MFTL_TriTriDistanceSq(const double A[3][3], const double B[3][3]);

/**
 * @brief Robust orientation predicate: six times the signed volume of the
 *        tetrahedron (pa, pb, pc, pd), with an EXACT sign (Shewchuk 1997).
 *
 * \f[
 *   \orient(\vect{a},\vect{b},\vect{c},\vect{d}) =
 *   (\vect{a}-\vect{d})\cdot\big[(\vect{b}-\vect{d})\times(\vect{c}-\vect{d})\big],
 * \f]
 * positive when @p pd lies below the plane through @p pa, @p pb, @p pc, i.e.
 * when those appear counter-clockwise seen from above. Adaptive: the plain
 * determinant plus a comparison against a rigorous error bound; exact expansion
 * arithmetic only when that bound cannot certify the sign.
 *
 * It is the one predicate whose SIGN decides an answer (Guigue-Devillers takes up
 * to eight per pair); a wrong sign can report a clean miss for crossing
 * triangles. Measured: 2.00x the plain determinant, at most 6.2 % of an
 * MFTL_TriTriProximity() call; the plain form never misclassified in 20 000
 * samples at gaps >= 1e-12 of the body size and fails only at 1e-14..1e-16
 * (examples/orient3d_predicate_profile.c). Requires `-ffp-contract=off`.
 *
 * @param pa First point of the reference plane.
 * @param pb Second point.
 * @param pc Third point.
 * @param pd The point being classified.
 * @return \f$6V\f$; the SIGN is exact, the magnitude approximate.
 */
double MFTL_Orient3D(const double pa[3], const double pb[3], const double pc[3], const double pd[3]);

/**
 * @brief The plain (non-adaptive) orientation determinant, for comparison and
 *        benchmarks; its sign is not guaranteed near degeneracy. Not used by the
 *        traversal.
 *
 * @param pa First point of the reference plane.
 * @param pb Second point.
 * @param pc Third point.
 * @param pd The point being classified.
 * @return \f$6V\f$ from the plain \f$3\times3\f$ determinant.
 */
double MFTL_Orient3DFast(const double pa[3], const double pb[3], const double pc[3], const double pd[3]);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_BVH_H
