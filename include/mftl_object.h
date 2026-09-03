/**
 * @file mftl_object.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief The mesh container: pool lifecycle, derived arrays, spatial ordering,
 *        and the hot-loop iterators.
 *
 * An MFTL_Object is one mesh. It owns two pools (mftl_types.h) and borrows the
 * MFTL_Domain of the ensemble it was added to.
 *
 * ## ONE ENTRY POINT FOR SIZE. THAT IS THE WHOLE POINT.
 *
 * Nothing outside src/object/object.c may write `verts.capacity`,
 * `tris.capacity`, any pool pointer, or `block`. Counts change through
 * MFTLObjectSetCounts(), which reserves before it assigns and therefore CANNOT
 * leave a count above a capacity. Capacity changes through MFTLObjectReserve()
 * or MFTLObjectReserveAdditional().
 *
 * Three capacities per object (`capacity_*`, `geom_cap_*`, `flat_cap_*`) plus a
 * rule in a comment saying that every site which grows a pool must remember to
 * call two other functions is the arrangement this replaces. Here there is one
 * capacity per pool, because a pool is one allocation. A growth that forgets an
 * array is not a bug you can make.
 *
 * ## THE DERIVED CONNECTIVITY AND WHEN IT GOES STALE
 *
 *   tris.verts, tris.neighbors     rebuilt by MFTLObjectBuildDerived
 *                                  invalid after any TOPOLOGY change
 *
 * Vertex motion does not invalidate connectivity. Geometric loops gather the
 * three current positions through `tris.verts` and form seam-corrected edge
 * vectors in registers, so there is no position-derived cache to refresh or
 * accidentally leave stale.
 */

#ifndef MFTL_OBJECT_H
#define MFTL_OBJECT_H

#include <math.h>
#include <stdint.h>

#include "mftl_constants.h"
#include "mftl_types.h"

// -----------------------------------------------------------------------------
// 1. LIFECYCLE
// -----------------------------------------------------------------------------

/**
 * @brief Allocates an empty object.
 *
 * Both pools start with zero capacity; a mesh generator or an I/O reader fills
 * them. The object has no domain until it is handed to MFTLEnsembleAddObject().
 *
 * @param global_id Identifies this object across all ranks.
 * @return The new object, or MFTL_NULL if the struct allocation failed.
 */
MFTL_Object *MFTLObjectCreate(int global_id);

/**
 * @brief Frees an object and NULLs the caller's pointer. Safe on NULL.
 *
 * Two free() calls do all the work -- one per pool block -- which is what a
 * single-allocation pool buys beyond the synchronisation guarantee.
 *
 * @param p_object Address of the object pointer. Set to #MFTL_NULL on return.
 */
void MFTLObjectDestroy(MFTL_Object **p_object);

// -----------------------------------------------------------------------------
// 2. SIZE (the only sanctioned way to change it)
// -----------------------------------------------------------------------------

/**
 * @brief Grows both pools to hold at least the requested entity counts, exactly.
 *
 * Exact growth, for bulk loads where the final size is known (generation, an I/O
 * read, the rebuild after a redistribution). A request at or below the current
 * capacity is a no-op: THIS function never shrinks, so a mesh of stable size
 * does not re-fault its pages every timestep, and callers may pre-grow and rely
 * on the next call early-returning. Giving capacity back is a separate, explicit
 * and gated operation -- MFTLObjectShrinkToFit().
 *
 * Growth PRESERVES contents -- every array, including the derived ones. The
 * alternative is to discard the geometry on every resize on the grounds that the
 * caller is about to recompute it. It usually is; when it is not, the result is a
 * silent zero. The copy happens O(log n) times over a mesh's life and is a memcpy
 * at memory bandwidth, which is not where any measurable time goes.
 *
 * @param obj        Object to grow.
 * @param n_vertices Vertices the pool must hold.
 * @param n_triangles Triangles the pool must hold. The half-edge array is sized
 *                   at 3x this automatically -- it is in the same block.
 */
void MFTLObjectReserve(MFTL_Object *obj, int n_vertices, int n_triangles);

/**
 * @brief Grows both pools by at least the given amounts, geometrically.
 *
 * For incremental appends, where doubling keeps the total number of
 * reallocations O(log n).
 *
 * @param obj             The mesh.
 * @param extra_vertices  Vertices that must fit beyond the current count.
 * @param extra_triangles Triangles that must fit beyond the current count.
 */
void MFTLObjectReserveAdditional(MFTL_Object *obj, int extra_vertices, int extra_triangles);

/**
 * @brief Sets the active and locally-owned counts of both pools.
 *
 * Reserves first, so the postcondition `n <= capacity` holds unconditionally --
 * this is the function that makes "counts and capacity disagree" unreachable.
 * Setting a count invalidates the derived arrays: they describe a different
 * number of entities than they did a moment ago.
 *
 * @param obj              The mesh.
 * @param n_vertices       Total vertices (owned + ghost).
 * @param n_local_vertices Owned vertices; they occupy [0, n_local_vertices).
 * @param n_triangles      Total triangles.
 * @param n_local_triangles Owned triangles.
 */
void MFTLObjectSetCounts(MFTL_Object *obj, int n_vertices, int n_local_vertices, int n_triangles, int n_local_triangles);

/**
 * @brief Empties the mesh, keeping the allocations.
 *
 * Counts and the global-ID counters go to zero; capacity is untouched, so
 * refilling the object does not re-allocate. Giving the storage back is the
 * other half and is never implicit: MFTLObjectShrinkToFit() for a mesh that is
 * merely smaller, MFTLObjectReleasePools() for one that is gone.
 *
 * @param obj The mesh. Counts and the global-ID counters go to zero.
 */
void MFTLObjectClear(MFTL_Object *obj);

/**
 * @brief Empties the mesh AND gives its two pool blocks back to the allocator.
 *
 * The counterpart to MFTLObjectClear(), which deliberately keeps the allocations
 * so that a mesh of stable size does not re-fault its pages every timestep. This
 * one is for the opposite case: a body that is no longer here at all.
 *
 * Growth is a RATCHET (see vertex_pool_reserve in src/object/object.c, and it is
 * the right default), so an object keeps whatever capacity it ever reached until
 * something explicitly asks for it back. This is the all-or-nothing form of that
 * ask and it requires an empty mesh; MFTLObjectShrinkToFit() is the graduated
 * form, for a body that is merely smaller than it was.
 *
 * Both are needed. On the generation path bodies are built whole on rank 0 --
 * `3680 * 4^refine + 176` bytes each -- and the opening redistribution then
 * leaves rank 0 with a SLIVER of the ones it still overlaps and NOTHING of the
 * rest. This function is what src/communication/redistribute.c calls for the
 * second group; the shrink is what handles the first. See
 * docs/scaling-analysis.md for what each is worth.
 *
 * @param obj The mesh. Must already hold no entities -- this frees storage, it
 *            does not decide what to discard. Afterwards the object is exactly as
 *            MFTLObjectCreate() left it, and the next MFTLObjectReserve()
 *            allocates from nothing.
 *
 * @warning Every pointer into either pool is dangling afterwards.
 *
 * @rank_local
 */
void MFTLObjectReleasePools(MFTL_Object *obj);

/**
 * @brief Gives back pool capacity a mesh has outgrown, if it is wasteful enough.
 *
 * The only way a pool ever gets smaller. MFTLObjectReserve() and
 * MFTLObjectSetCounts() raise `capacity` and never lower it, deliberately -- a
 * mesh of stable size must not re-fault its pages every timestep -- so a rank
 * whose share of a body SHRANK keeps the block it held at its high-water mark
 * for the rest of the run. That is not hypothetical: MFTLGenerateSphere() builds
 * every body whole on rank 0, and after the opening MFTLRedistribute() rank 0
 * holds a sliver of each in a block sized for all of it.
 *
 * Both pools are considered independently. One is shrunk when
 *
 *     capacity > slack * n
 *
 * and then to `MFTLGrowCapacity(0, n)`, the capacity an incrementally grown pool
 * would have reached -- which is `n` rounded up the doubling ladder, so the
 * result is in \f$[n, 2n)\f$ and the next append is still amortised. At `n == 0`
 * that target is 0 and the block is released outright, exactly as
 * MFTLObjectReleasePools() would.
 *
 * **No entity is ever discarded.** The target is at least `n` by construction;
 * this function moves STORAGE and does not decide what to drop. Lower the counts
 * first, through MFTLObjectSetCounts(), if that is what you meant.
 *
 * **Nothing observable changes.** `n`, `n_local`, every array's live prefix, both
 * MFTL_DERIVED_* bits and the partition-dirty stamp are exactly as they were --
 * this is a memory operation, not a mesh operation. In particular it must not be
 * spelled through MFTLObjectSetCounts(), which stamps the body dirty and would
 * silently defeat MFTLRedistributeSetSkipUnchangedBodies().
 *
 * @param obj   The mesh.
 * @param slack Waste factor at which a pool is worth reallocating. Use
 *              #MFTL_SHRINK_SLACK_DEFAULT unless you have measured otherwise;
 *              values below 4 can thrash against the doubling growth ladder, and
 *              that constant's documentation derives why. Must be at least 1.
 * @return `true` if either pool actually reallocated, so a caller can count the
 *         event without asking a second question.
 *
 * @warning Every pointer into either pool is dangling afterwards.
 *
 * @rank_local
 */
bool MFTLObjectShrinkToFit(MFTL_Object *obj, double slack);

/**
 * @brief Bytes the two pool blocks currently occupy.
 *
 * What this object costs in memory: `capacity` entities' worth of every array of
 * both pools, including the cache-line padding between sub-arrays, which is the
 * only honest form of the question since a pool is one allocation. Sizes are
 * re-derived from the block layout, so adding an array to a pool cannot leave
 * this stale.
 *
 * It reports CAPACITY, not the live count, which is the point: the gap between
 * the two is exactly what MFTLObjectShrinkToFit() recovers. It is also not a
 * resident-set measurement -- whether free() hands pages back to the OS is
 * allocator-dependent, so a drop here is an upper bound on the drop in RSS.
 *
 * @param obj The mesh.
 * @return Total bytes in `verts.block` plus `tris.block`. Zero for an object
 *         whose pools have never been reserved or have been released.
 *
 * @rank_local
 */
size_t MFTLObjectPoolBytes(const MFTL_Object *obj);

/**
 * @brief Debug-only structural check of the pools themselves.
 *
 * Verifies what the single-block layout is supposed to make true: counts within
 * capacity, owned counts within active counts, every pointer non-NULL exactly
 * when the capacity is non-zero, and each pointer inside its own block. Used by
 * the tests and by assertions; not a mesh validity check (that is
 * MFTLCheckMeshIntegrity).
 *
 * @return MFTL_TRUE if the pools are structurally sound.
 *
 * @param obj The mesh.
 * @return `true` if every count, capacity and pointer agrees. Intended for
 *         assertions and tests, not for a hot path.
 */
bool MFTLObjectPoolsAreConsistent(const MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 3. DERIVED ARRAYS
// -----------------------------------------------------------------------------

/**
 * @brief True if `tris.verts` and `tris.neighbors` describe the current topology.
 * @param obj The mesh.
 * @return `true` iff #MFTL_DERIVED_TOPOLOGY is set.
 */
static inline bool MFTLObjectHasTopology(const MFTL_Object *obj) { return (obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u; }

/**
 * @brief Declares every derived array stale. Call after editing topology by hand.
 * @param obj The mesh.
 */
static inline void MFTLObjectInvalidateDerived(MFTL_Object *obj) { obj->derived = 0u; }

/**
 * @brief Declares that the CURRENT partition no longer describes this mesh.
 *
 * MFTLRedistribute() may leave a body entirely alone when no entity of it changed
 * owner: a rank's local state after a redistribution is a pure function of
 * (global topology, owner map), so if neither input moved neither does the
 * output. This flag is the other half of that precondition -- it reports that
 * something which is NOT the owner map changed, so the body must be rebuilt even
 * though nothing moved.
 *
 * Three kinds of change need it, and the library already stamps all three:
 *
 *   1. ENTITY COUNTS. MFTLObjectSetCounts() is the enforced choke point for any
 *      creation or destruction, and it stamps.
 *   2. GLOBAL IDS. MFTLRemeshRenumberGlobalIds() and the spanning remesher's
 *      closing redensification rewrite ids of entities other ranks hold copies
 *      of. Ghost TRIANGLE ids in particular are repaired only by being dropped
 *      and re-received, so a skip would remove the repair and leave stale ids
 *      colliding with fresh ones -- silently, and several timesteps from the
 *      cause.
 *   3. ENTITY FLAGS. Redistribution replaces a ghost triangle's flags from its
 *      owner's copy but RETAINS a kept ghost vertex's local record, so a pass
 *      that writes flags without changing counts must stamp too.
 *      MFTLRemeshMarkFrozen() does.
 *
 * Anything else that edits a mesh in place without going through
 * MFTLObjectSetCounts() owes this call. It is CONSERVATIVE by design: a spurious
 * stamp costs one body's rebuild, a missing one is silent corruption.
 *
 * @param obj The mesh. Only `partition_dirty` is written; the derived bits are
 *            a separate question and are not touched.
 *
 * @rank_local It is a local statement about a local pool. Redistribution reduces
 *             the per-body votes of every rank before acting on any of them.
 */
static inline void MFTLObjectMarkPartitionDirty(MFTL_Object *obj) { obj->partition_dirty = MFTL_TRUE; }

/**
 * @brief Rebuilds every derived per-triangle array from the half-edge structure.
 *
 * One streaming pass emits `tris.verts` and `tris.neighbors`. Call it after any
 * topology change -- generation, remeshing, redistribution.
 *
 * @param obj The mesh. `tris.verts` and `tris.neighbors` are rewritten and
 *            #MFTL_DERIVED_TOPOLOGY is set.
 *
 * @rank_local
 */
void MFTLObjectBuildDerived(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 4. SPATIAL (MORTON) ORDERING
// -----------------------------------------------------------------------------

/** Bits per axis in the Morton key: 3 * 21 = 63, the most a uint64 key holds. */
#define MFTL_MORTON_BITS 21
/** Largest quantised coordinate, 2^21 - 1. */
#define MFTL_MORTON_MAX 2097151.0

/**
 * @brief Spreads the low 21 bits of @p x so that bit i ends up at position 3i.
 *
 * The classic magic-number bit interleave: five shift/mask steps double the gap
 * between bits each round, which is \f$O(\log b)\f$ instead of the 21 iterations
 * a bit-by-bit loop would take.
 *
 * @param x Value whose low 21 bits are spread. Higher bits are masked off.
 * @return \f$\sum_i x_i 2^{3i}\f$.
 */
static inline uint64_t MFTLMortonSpread3(uint64_t x)
{
  x &= UINT64_C(0x1fffff);
  x = (x | (x << 32)) & UINT64_C(0x1f00000000ffff);
  x = (x | (x << 16)) & UINT64_C(0x1f0000ff0000ff);
  x = (x | (x << 8)) & UINT64_C(0x100f00f00f00f00f);
  x = (x | (x << 4)) & UINT64_C(0x10c30c30c30c30c3);
  x = (x | (x << 2)) & UINT64_C(0x1249249249249249);
  return x;
}

/**
 * @brief 63-bit Morton (Z-order) key from three 21-bit integer coordinates.
 *
 * \f[
 *   Z(x,y,z) = \sum_{i=0}^{20} \Big( x_i 2^{3i} + y_i 2^{3i+1} + z_i 2^{3i+2} \Big)
 * \f]
 *
 * Sorting the keys ascending walks the Z-curve, and two points close in space
 * have keys agreeing in their high bits -- which is the locality property the
 * renumbering and the LBVH build both rely on.
 *
 * @param x Quantised x coordinate, 21 bits.
 * @param y Quantised y coordinate, 21 bits.
 * @param z Quantised z coordinate, 21 bits.
 * @return The 63-bit interleaved key.
 *
 * @warning Inputs above \f$2^{21}-1\f$ are MASKED, not clamped. Quantise
 *          against #MFTL_MORTON_MAX before calling.
 */
static inline uint64_t MFTLMorton3D(uint64_t x, uint64_t y, uint64_t z)
{
  return MFTLMortonSpread3(x) | (MFTLMortonSpread3(y) << 1) | (MFTLMortonSpread3(z) << 2);
}

/**
 * @brief Renumbers the mesh along a Morton curve so that neighbours in space are
 *        neighbours in memory.
 *
 * This is what replaces an `sfc_perm` / `sfc_inv` pair. Rather than storing a
 * permutation and a second, SFC-ordered copy of every position -- two index
 * spaces, which is the arrangement silent corruption lives in -- the permutation
 * is APPLIED to the pools. Both the AoS records and every SoA array move
 * together, and all indices are remapped, so afterwards there is still exactly
 * one index space and exactly one copy of every coordinate.
 *
 * What it does, in order:
 *   1. Morton key per vertex from `verts.pos`, quantised against the object's
 *      own bounding box so the resolution is spent where the mesh is.
 *   2. A stable sort WITHIN [0, n_local) and, separately, within [n_local, n),
 *      so the ownership prefix survives.
 *   3. Permutes every vertex array; remaps `he.vertex_idx`.
 *   4. Orders triangles by their smallest (already Morton-ordered) vertex index,
 *      which is a spatial key by construction and needs no second geometric
 *      pass; permutes the triangle arrays and their half-edge triples together.
 *   5. Remaps `he.twin_idx` and `rec.out_he_idx`, then rebuilds the derived
 *      arrays.
 *
 * Global IDs are NOT touched: they are the stable name of an entity across
 * ranks and across reorderings. Pool INDICES are, by definition, not stable
 * across this call -- anything holding one must refresh it.
 *
 * Cost is one radix sort plus a handful of linear permutation passes. Call it
 * after generation and after a redistribution; calling it every timestep is
 * waste, since incremental vertex motion does not degrade Morton locality
 * quickly.
 *
 * @param obj The mesh. Both pools are permuted in place.
 *
 * @rank_local Each rank orders its own partition.
 */
void MFTLObjectReorderMorton(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 5. GLOBAL STATISTICS
// -----------------------------------------------------------------------------

/**
 * @brief Sums the owned entity counts of ONE object across all ranks.
 *
 * Useful when only one object has been modified (e.g. during generation) and a
 * full ensemble reduction is not warranted. With ownership as a prefix this is
 * one Allreduce of two integers and no per-entity work at all -- the previous
 * version scanned every entity testing an `is_ghost` flag first.
 *
 * @note Collective on the object's domain communicator.
 *
 * @param object The mesh. `global_n_vertices` and `global_n_triangles` are
 *               overwritten on every rank.
 */
void MFTLObjectUpdateGlobalStats(MFTL_Object *object);

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. HOT-LOOP ITERATORS
 *
 * Macros, not functions, and deliberately so. C has no lambdas, so a function
 * taking the loop body would have to take a function POINTER -- which the
 * compiler cannot inline across a translation unit, cannot vectorise through,
 * and cannot hoist the loop-invariant setup out of. A macro pastes the body into
 * the loop textually, so the optimiser sees one flat loop over contiguous arrays
 * and nothing else.
 *
 * What every iterator guarantees, and what an open-coded loop often gets wrong:
 *
 *   - the arrays are hoisted into `restrict`-qualified locals, so the compiler
 *     may assume they do not alias each other or the output;
 *   - the loop bound is a local, not a struct member reloaded every iteration;
 *   - connectivity reads are sequential and there is no pointer chasing;
 *   - periodicity is unswitched once per call, never tested per triangle;
 *   - seam-corrected edges live only in registers.
 *
 * The body is variadic, so it may contain commas. `break` and `continue` behave
 * as in any for loop.
 *
 * ── An honest note on "SIMD" ──────────────────────────────────────────────
 *
 * These loops are shaped to expose the work to the compiler: restrict pointers,
 * an invariant bound and no branch per triangle. Position reads are indexed
 * gathers, so SIMD profitability remains architecture- and body-dependent.
 * They deliberately do NOT force it with a pragma: the caller's body may
 * legitimately carry a dependency the macro cannot see.
 *
 * The reliable properties are compact connectivity traffic and current
 * positions. There is no mesh-wide position-derived array to build, write,
 * invalidate or keep resident.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Iterates current triangle geometry from compact connectivity.
 *
 * Streams `tris.verts`, gathers the three authoritative vertex positions and
 * forms two minimum-image edge vectors in registers. Periodicity is tested once
 * outside the triangle loop. In a non-periodic domain the hot loop contains only
 * the gathers and subtractions; in a periodic domain its correction is
 * branchless, with zero reciprocal on non-periodic axes.
 *
 * In scope for the body:
 *   int    t                 triangle pool index
 *   int    v0, v1, v2        vertex pool indices
 *   double p0x, p0y, p0z     wrapped position of corner 0
 *   double e1x, e1y, e1z     v1 - v0, seam-consistent
 *   double e2x, e2y, e2z     v2 - v0, seam-consistent
 *
 * @param count Triangles to visit from index 0. Pass `obj->tris.n_local` for
 *        owned only (they are an array prefix) or `obj->tris.n` to include ghosts.
 */
#define MFTL_FOREACH_TRIANGLE_GEOMETRY(obj, count, ...)                                                \
  do                                                                                                   \
  {                                                                                                    \
    MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTL_FOREACH_TRIANGLE_GEOMETRY: stale tris.verts -- rebuild derived connectivity first"); \
    const int *restrict mftl__tv = (obj)->tris.verts;                                                  \
    const double *restrict mftl__pos = (obj)->verts.pos;                                               \
    const MFTL_Domain *const mftl__dom = (obj)->domain;                                                \
    const int mftl__n = (count);                                                                       \
    MFTL_ASSERT(mftl__n <= (obj)->tris.n && "MFTL_FOREACH_TRIANGLE_GEOMETRY: count exceeds triangle pool"); \
    const bool mftl__periodic = mftl__dom != MFTL_NULL &&                                              \
                                (mftl__dom->is_periodic[0] || mftl__dom->is_periodic[1] || mftl__dom->is_periodic[2]); \
    if (!mftl__periodic)                                                                                \
    {                                                                                                  \
      for (int t = 0; t < mftl__n; ++t)                                                               \
      {                                                                                                \
        const int v0 = mftl__tv[3 * t + 0], v1 = mftl__tv[3 * t + 1], v2 = mftl__tv[3 * t + 2];       \
        const double *const mftl__p0 = &mftl__pos[3 * v0];                                            \
        const double *const mftl__p1 = &mftl__pos[3 * v1];                                            \
        const double *const mftl__p2 = &mftl__pos[3 * v2];                                            \
        const double p0x = mftl__p0[0], p0y = mftl__p0[1], p0z = mftl__p0[2];                         \
        const double e1x = mftl__p1[0] - p0x, e1y = mftl__p1[1] - p0y, e1z = mftl__p1[2] - p0z;       \
        const double e2x = mftl__p2[0] - p0x, e2y = mftl__p2[1] - p0y, e2z = mftl__p2[2] - p0z;       \
        __VA_ARGS__;                                                                                   \
      }                                                                                                \
    }                                                                                                  \
    else                                                                                               \
    {                                                                                                  \
      const double mftl__l0 = mftl__dom->domain_size[0];                                              \
      const double mftl__l1 = mftl__dom->domain_size[1];                                              \
      const double mftl__l2 = mftl__dom->domain_size[2];                                              \
      const double mftl__i0 = mftl__dom->is_periodic[0] ? mftl__dom->inv_domain_size[0] : 0.0;         \
      const double mftl__i1 = mftl__dom->is_periodic[1] ? mftl__dom->inv_domain_size[1] : 0.0;         \
      const double mftl__i2 = mftl__dom->is_periodic[2] ? mftl__dom->inv_domain_size[2] : 0.0;         \
      for (int t = 0; t < mftl__n; ++t)                                                               \
      {                                                                                                \
        const int v0 = mftl__tv[3 * t + 0], v1 = mftl__tv[3 * t + 1], v2 = mftl__tv[3 * t + 2];       \
        const double *const mftl__p0 = &mftl__pos[3 * v0];                                            \
        const double *const mftl__p1 = &mftl__pos[3 * v1];                                            \
        const double *const mftl__p2 = &mftl__pos[3 * v2];                                            \
        const double p0x = mftl__p0[0], p0y = mftl__p0[1], p0z = mftl__p0[2];                         \
        const double mftl__d10 = mftl__p1[0] - p0x, mftl__d11 = mftl__p1[1] - p0y, mftl__d12 = mftl__p1[2] - p0z; \
        const double mftl__d20 = mftl__p2[0] - p0x, mftl__d21 = mftl__p2[1] - p0y, mftl__d22 = mftl__p2[2] - p0z; \
        const double e1x = mftl__d10 - mftl__l0 * nearbyint(mftl__d10 * mftl__i0);                    \
        const double e1y = mftl__d11 - mftl__l1 * nearbyint(mftl__d11 * mftl__i1);                    \
        const double e1z = mftl__d12 - mftl__l2 * nearbyint(mftl__d12 * mftl__i2);                    \
        const double e2x = mftl__d20 - mftl__l0 * nearbyint(mftl__d20 * mftl__i0);                    \
        const double e2y = mftl__d21 - mftl__l1 * nearbyint(mftl__d21 * mftl__i1);                    \
        const double e2z = mftl__d22 - mftl__l2 * nearbyint(mftl__d22 * mftl__i2);                    \
        __VA_ARGS__;                                                                                   \
      }                                                                                                \
    }                                                                                                  \
  } while (0)

/**
 * @brief Iterates vertex positions, read-only and streaming.
 *
 * For anything that reduces over positions -- bounding boxes, centres of mass,
 * extents. This is now a plain contiguous walk of `verts.pos`; in the previous
 * version the equivalent loop either strode ~100 bytes through the vertex
 * records or read a second, duplicated position array.
 *
 * In scope:
 *   int    v                 vertex pool index
 *   double px, py, pz        position, wrapped into the domain
 */
#define MFTL_FOREACH_VERTEX_POS(obj, count, ...)                                                    \
  do                                                                                                \
  {                                                                                                 \
    const double *restrict mftl__p = (obj)->verts.pos;                                              \
    const int mftl__n = (count);                                                                    \
    for (int v = 0; v < mftl__n; ++v)                                                               \
    {                                                                                               \
      const double px = mftl__p[3 * v + 0], py = mftl__p[3 * v + 1], pz = mftl__p[3 * v + 2];       \
      __VA_ARGS__;                                                                                  \
    }                                                                                               \
  } while (0)

/**
 * @brief Iterates the vertices this rank OWNS, writable.
 *
 * The motion iterator: advection, smoothing, anything that moves a vertex. It
 * exploits the ownership PREFIX, so there is no `owner_proc == rank` test in the
 * loop -- a hand-written advection loop almost always carries that branch, and
 * it is pure waste on a correctly partitioned mesh.
 *
 * In scope:
 *   int      v               vertex pool index
 *   double  *pos             three writable doubles
 *   MFTL_Vertex *rec         the record, for flags and ids
 */
#define MFTL_FOREACH_OWNED_VERTEX(obj, ...)                                            \
  do                                                                                   \
  {                                                                                    \
    double *restrict mftl__p = (obj)->verts.pos;                                       \
    MFTL_Vertex *restrict mftl__r = (obj)->verts.rec;                                  \
    const int mftl__n = (obj)->verts.n_local;                                          \
    for (int v = 0; v < mftl__n; ++v)                                                  \
    {                                                                                  \
      double *const pos = &mftl__p[3 * v];                                             \
      MFTL_Vertex *const rec = &mftl__r[v];                                            \
      (void) rec;                                                                      \
      __VA_ARGS__;                                                                     \
    }                                                                                  \
  } while (0)

#endif  // MFTL_OBJECT_H
