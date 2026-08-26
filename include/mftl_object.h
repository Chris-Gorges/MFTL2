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
 * ## THE DERIVED ARRAYS AND WHEN THEY GO STALE
 *
 *   tris.verts, tris.neighbors     rebuilt by MFTLObjectBuildDerived
 *                                  invalid after any TOPOLOGY change
 *   tris.anchor, tris.edge         rebuilt by MFTLObjectBuildDerived and
 *                                  refreshed by MFTLObjectRefreshFrames
 *                                  invalid after topology change OR vertex motion
 *
 * `obj->derived` records which of the two groups is current. Every iterator
 * below asserts on it in Debug and costs nothing in Release, and
 * MFTL_FOREACH_OWNED_VERTEX clears the frame bit on the way out because a loop
 * that moves vertices has invalidated the frames by construction. The previous
 * version required the caller to remember an explicit MFTLSyncFlatPositions
 * call, and getting it wrong produced geometry silently one timestep stale.
 */

#ifndef MFTL_OBJECT_H
#define MFTL_OBJECT_H

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
 * capacity is a no-op: pools never shrink, so a mesh of stable size does not
 * re-fault its pages every timestep.
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
 * refilling the object does not re-allocate.
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
 * Pools never shrink (see vertex_pool_reserve in src/object/object.c, and it is
 * the right default), so an object keeps whatever capacity it ever reached for
 * the rest of its life. On the generation path that is the whole mesh: bodies are
 * built on rank 0 and then redistributed, so rank 0 retains capacity for every
 * body in the ensemble -- `5120 * 4^refine + 176` bytes each, which
 * docs/scaling-analysis.md prices at 671 MB for 512 bodies at refine 4. Nothing
 * short of freeing the block recovers it.
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
 * @brief True if `tris.anchor` and `tris.edge` describe the current positions.
 * @param obj The mesh.
 * @return `true` iff #MFTL_DERIVED_FRAMES is set.
 */
static inline bool MFTLObjectHasFrames(const MFTL_Object *obj) { return (obj->derived & MFTL_DERIVED_FRAMES) != 0u; }

/**
 * @brief Declares every derived array stale. Call after editing topology by hand.
 * @param obj The mesh.
 */
static inline void MFTLObjectInvalidateDerived(MFTL_Object *obj) { obj->derived = 0u; }

/**
 * @brief Declares only the frames stale. Call after moving vertices by hand.
 *
 * #MFTL_FOREACH_OWNED_VERTEX does this on the way out, which is why a motion
 * pass written with the iterator cannot leave geometry a timestep stale.
 *
 * @param obj The mesh.
 */
static inline void MFTLObjectInvalidateFrames(MFTL_Object *obj) { obj->derived &= ~MFTL_DERIVED_FRAMES; }

/**
 * @brief Rebuilds every derived per-triangle array from the half-edge structure.
 *
 * One pass emits `tris.verts` and `tris.neighbors`; a second emits `tris.anchor`
 * and `tris.edge`. Call it after any topology change -- generation, remeshing,
 * redistribution.
 *
 * @param obj The mesh. `tris.verts`, `tris.neighbors`, `tris.anchor` and
 *            `tris.edge` are all rewritten, and both `derived` bits are set.
 *
 * @rank_local
 */
void MFTLObjectBuildDerived(MFTL_Object *obj);

/**
 * @brief Rebuilds only `tris.verts` and `tris.neighbors` from the half-edge records.
 *
 * The topology half of MFTLObjectBuildDerived(), separated because the two
 * halves answer to different invalidation rules and cost different amounts:
 * this one is 24 B per triangle written from a streaming read of the half-edge
 * array, while MFTLObjectRefreshFrames() is 72 B per triangle written from a
 * SCATTERED gather of three vertex positions.
 *
 * `MFTLObjectBuildDerived(obj)` is exactly this call followed by
 * MFTLObjectRefreshFrames(obj); a caller that wants both should still use it.
 * Reach for this one only when the frames are genuinely not wanted, or -- as the
 * redistribution does -- when the two costs must be attributed separately.
 *
 * @param obj The mesh. `tris.verts` and `tris.neighbors` are rewritten and
 *            #MFTL_DERIVED_TOPOLOGY is set. The frame bit is left alone, so a
 *            caller that does not follow up with MFTLObjectRefreshFrames() has
 *            an object whose frames are stale and correctly marked so.
 *
 * @rank_local
 */
void MFTLObjectBuildTopologyArrays(MFTL_Object *obj);

/**
 * @brief Rebuilds only `tris.anchor` and `tris.edge` from the current positions.
 *
 * The position-only update, for advection and smoothing. Requires
 * MFTLObjectHasTopology(), since it gathers through `tris.verts`.
 *
 * @param obj The mesh. `tris.anchor` and `tris.edge` are rewritten and
 *            #MFTL_DERIVED_FRAMES is set.
 *
 * @rank_local
 */
void MFTLObjectRefreshFrames(MFTL_Object *obj);

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
 * What every iterator guarantees, and what an open-coded loop usually gets
 * wrong:
 *
 *   - the arrays are hoisted into `restrict`-qualified locals, so the compiler
 *     may assume they do not alias each other or the output;
 *   - the loop bound is a local, not a struct member reloaded every iteration;
 *   - reads are SEQUENTIAL, with no pointer chasing and no gather;
 *   - the triangle iterators need no minimum-image correction, because
 *     `tris.edge` is already seam-consistent.
 *
 * The body is variadic, so it may contain commas. `break` and `continue` behave
 * as in any for loop.
 *
 * ── An honest note on "SIMD" ──────────────────────────────────────────────
 *
 * These loops are SHAPED so the compiler can vectorise them -- contiguous reads,
 * restrict pointers, an invariant bound, no branch and no gather per element.
 * They deliberately do NOT force it with a pragma: the caller's body may
 * legitimately carry a dependency the macro cannot see.
 *
 * Whether a given use vectorises is therefore decided by the BODY. Measured over
 * four kernels that differ only in how the body writes its result:
 *
 *   body writes contiguous arrays, no reduction   -> vectorises
 *   body reduces with fmin/fmax                   -> vectorises
 *   body reduces with `if (a < lo) lo = a;`       -> HARD REFUSAL, always
 *   body writes back into an AoS record           -> never vectorises
 *
 * The last line no longer applies to geometry, which is exactly why the geometry
 * left the records. The reliable win here is not the vector units, though: it is
 * the memory traffic. No pointer chasing, no gathers, no per-element branch.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Iterates triangle EDGE VECTORS. The cheapest loop in the library.
 *
 * Pure streaming over `tris.edge`: no gather, no indirection, no periodic
 * correction. Everything that depends only on a triangle's SHAPE -- area,
 * normal, edge lengths, aspect ratio, any quality measure -- needs exactly this
 * and nothing else, and touches 48 bytes per triangle to get it.
 *
 * In scope for the body:
 *   int    t                 triangle pool index
 *   double e1x, e1y, e1z     v1 - v0, seam-consistent
 *   double e2x, e2y, e2z     v2 - v0, seam-consistent
 *
 * @param count Triangles to visit from index 0. Pass `obj->tris.n_local` for
 *        owned only (they are an array prefix) or `obj->tris.n` to include ghosts.
 */
#define MFTL_FOREACH_TRIANGLE_EDGE(obj, count, ...)                                                    \
  do                                                                                                   \
  {                                                                                                    \
    MFTL_ASSERT(MFTLObjectHasFrames(obj) && "MFTL_FOREACH_TRIANGLE_EDGE: stale frames -- build/refresh first"); \
    const double *restrict mftl__e = (obj)->tris.edge;                                                 \
    const int mftl__n = (count);                                                                       \
    for (int t = 0; t < mftl__n; ++t)                                                                  \
    {                                                                                                  \
      const double e1x = mftl__e[6 * t + 0], e1y = mftl__e[6 * t + 1], e1z = mftl__e[6 * t + 2];       \
      const double e2x = mftl__e[6 * t + 3], e2y = mftl__e[6 * t + 4], e2z = mftl__e[6 * t + 5];       \
      __VA_ARGS__;                                                                                     \
    }                                                                                                  \
  } while (0)

/**
 * @brief The complete triangle FRAME: edge vectors plus the anchor at corner 0.
 *
 * The same loop with one more streaming read, `tris.anchor`. Use it when an
 * absolute position is genuinely needed -- a centroid, a bounding box; anything
 * shape-only should use MFTL_FOREACH_TRIANGLE_EDGE, which touches 24 B per
 * triangle less.
 *
 * The anchor is a per-triangle copy of corner 0, written at build time where it
 * is already in a register because both edge vectors are differences against it.
 * That deletes the last indirection from the loop: no gather at all, two
 * contiguous arrays in, whatever the body writes out. The expected shape of the
 * payoff is "free while the positions fit in cache, worth about 13 % once they do
 * not", which is what deleting a random read looks like. The second half has NOT
 * been reproduced on this machine -- see docs/performance-analysis.md, where the
 * gather this removes turns out not to cost anything measurable on a part with a
 * large L2. It is kept because 24 B per triangle is cheap insurance on a cluster
 * node where it will.
 *
 * Corner positions are p0, p0 + e1, p0 + e2. p0 is a WRAPPED position, so a
 * centroid derived from it must be wrapped back into the box before it is used
 * as a position.
 *
 * In scope: everything MFTL_FOREACH_TRIANGLE_EDGE provides, plus
 *   double p0x, p0y, p0z     position of corner 0
 */
#define MFTL_FOREACH_TRIANGLE_FRAME(obj, count, ...)                                                   \
  do                                                                                                   \
  {                                                                                                    \
    MFTL_ASSERT(MFTLObjectHasFrames(obj) && "MFTL_FOREACH_TRIANGLE_FRAME: stale frames -- build/refresh first"); \
    const double *restrict mftl__e = (obj)->tris.edge;                                                 \
    const double *restrict mftl__a = (obj)->tris.anchor;                                               \
    const int mftl__n = (count);                                                                       \
    for (int t = 0; t < mftl__n; ++t)                                                                  \
    {                                                                                                  \
      const double e1x = mftl__e[6 * t + 0], e1y = mftl__e[6 * t + 1], e1z = mftl__e[6 * t + 2];       \
      const double e2x = mftl__e[6 * t + 3], e2y = mftl__e[6 * t + 4], e2z = mftl__e[6 * t + 5];       \
      const double p0x = mftl__a[3 * t + 0], p0y = mftl__a[3 * t + 1], p0z = mftl__a[3 * t + 2];       \
      __VA_ARGS__;                                                                                     \
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
 * It CLEARS the frame bit on the way out, so a subsequent frame loop asserts in
 * Debug unless MFTLObjectRefreshFrames() has run. Leaving that to the caller's
 * memory is what makes a forgotten call produce geometry one timestep stale --
 * silently, and only where the mesh actually moved.
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
    MFTLObjectInvalidateFrames(obj);                                                   \
  } while (0)

#endif  // MFTL_OBJECT_H
