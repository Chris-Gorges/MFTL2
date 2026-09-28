/**
 * @file mftl_object.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief The mesh container: pool lifecycle, derived connectivity, Morton
 *        ordering, and the hot-loop iterators.
 *
 * An #MFTL_Object is one mesh (one body). It owns two pools (mftl_types.h) and
 * borrows the #MFTL_Domain of the ensemble it was added to.
 *
 * **Size has one entry point.** Only src/object/object.c writes a pool's
 * `capacity`, `block` or sub-array pointers. Counts change through
 * MFTLObjectSetCounts(), which reserves before it assigns, so `n <= capacity`
 * always holds; capacity grows through MFTLObjectReserve() /
 * MFTLObjectReserveAdditional() and is given back only explicitly
 * (MFTLObjectShrinkToFit(), MFTLObjectReleasePools()).
 *
 * **Derived connectivity** (`tris.verts`, `tris.neighbors`) is invalidated by
 * any topology change and rebuilt by MFTLObjectBuildDerived(). Vertex motion
 * invalidates nothing: geometric loops gather current positions and form
 * seam-corrected edges in registers, so no position-derived cache exists.
 *
 * Theory: src/object/README.md.
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
 * @brief Allocates an empty object with zero-capacity pools and no domain.
 *
 * @param global_id PROVISIONAL body id: MFTLEnsembleAddObject() overwrites it
 *                  with the next dense id. It is the real identity only of an
 *                  object that never joins an ensemble (a spatial-halo shadow
 *                  patch, a serial test fixture).
 * @return The new object, or #MFTL_NULL if the struct allocation failed.
 */
MFTL_Object *MFTLObjectCreate(int global_id);

/**
 * @brief Frees an object (two pool blocks and the struct) and NULLs the caller's
 *        pointer. Safe on NULL.
 *
 * @param p_object Address of the object pointer. Set to #MFTL_NULL on return.
 */
void MFTLObjectDestroy(MFTL_Object **p_object);

// -----------------------------------------------------------------------------
// 2. SIZE (the only sanctioned way to change it)
// -----------------------------------------------------------------------------

/**
 * @brief Grows both pools to hold at least the given counts, exactly.
 *
 * For bulk loads whose final size is known (generation, the rebuild after a
 * redistribution). A request at or below the current capacity is a no-op --
 * growth is a ratchet, so a mesh of stable size never re-faults its pages and a
 * caller may pre-grow and rely on the next reserve returning early. Contents of
 * every array, derived ones included, are preserved.
 *
 * @param obj         Object to grow.
 * @param n_vertices  Vertices the pool must hold.
 * @param n_triangles Triangles the pool must hold (half-edges: 3x, same block).
 */
void MFTLObjectReserve(MFTL_Object *obj, int n_vertices, int n_triangles);

/**
 * @brief Grows both pools by at least the given amounts, geometrically
 *        (MFTLGrowCapacity()), for incremental appends.
 *
 * @param obj             The mesh.
 * @param extra_vertices  Vertices that must fit beyond the current count.
 * @param extra_triangles Triangles that must fit beyond the current count.
 */
void MFTLObjectReserveAdditional(MFTL_Object *obj, int extra_vertices, int extra_triangles);

/**
 * @brief Sets the active and owned counts of both pools.
 *
 * Reserves first, so `n <= capacity` holds unconditionally. Clears the derived
 * bit and stamps the body partition-dirty (MFTLObjectMarkPartitionDirty()):
 * this is the choke point every entity creation and destruction passes through.
 *
 * @param obj               The mesh.
 * @param n_vertices        Total vertices (owned + ghost).
 * @param n_local_vertices  Owned vertices; they occupy [0, n_local_vertices).
 * @param n_triangles       Total triangles.
 * @param n_local_triangles Owned triangles; they occupy [0, n_local_triangles).
 */
void MFTLObjectSetCounts(MFTL_Object *obj, int n_vertices, int n_local_vertices, int n_triangles, int n_local_triangles);

/**
 * @brief Empties the mesh but keeps the allocations.
 *
 * Counts, global-id counters and global counts go to zero and the body is
 * stamped partition-dirty; capacity is untouched, so refilling does not
 * reallocate.
 *
 * @param obj The mesh.
 */
void MFTLObjectClear(MFTL_Object *obj);

/**
 * @brief Frees both pool blocks of an EMPTY object.
 *
 * The all-or-nothing counterpart of MFTLObjectShrinkToFit(), for a body that
 * has left this rank entirely (src/communication/redistribute.c uses it after
 * the opening scatter of a generated ensemble, where rank 0 held every body
 * whole -- \f$3440 \cdot 4^r + 128\f$ bytes per icosphere of refinement r).
 * Afterwards the object is exactly as MFTLObjectCreate() left it.
 *
 * @param obj The mesh; must hold no entities (asserted).
 *
 * @warning Every pointer into either pool is dangling afterwards.
 *
 * @rank_local
 */
void MFTLObjectReleasePools(MFTL_Object *obj);

/**
 * @brief Gives back pool capacity a mesh has outgrown, if the waste is large.
 *
 * The only way a pool gets smaller. Each pool is resized independently when
 *
 * \f[ \mathrm{capacity} > \mathrm{slack} \cdot n , \f]
 *
 * to `MFTLGrowCapacity(0, n)` -- the smallest \f$128\cdot2^k \ge n\f$, i.e.
 * back on the growth ladder -- and released entirely when `n == 0`. Counts,
 * contents, the derived bit and the partition stamp are unchanged: this moves
 * storage, it is not a mesh operation (and must not be spelled through
 * MFTLObjectSetCounts(), which would stamp the body dirty and defeat the
 * redistribution's skip of unchanged bodies).
 *
 * @param obj   The mesh.
 * @param slack Waste factor, at least 1; use #MFTL_SHRINK_SLACK_DEFAULT (4).
 *              Smaller values thrash against the doubling growth ladder.
 * @return `true` if either pool was reallocated.
 *
 * @warning Every pointer into either pool is dangling afterwards.
 *
 * @rank_local
 */
bool MFTLObjectShrinkToFit(MFTL_Object *obj, double slack);

/**
 * @brief Bytes the two pool blocks currently occupy, padding included.
 *
 * Derived from the block layout, so it cannot drift from it. It measures
 * CAPACITY, not the live count (the gap is what MFTLObjectShrinkToFit()
 * recovers), and is an upper bound on resident memory, since whether `free()`
 * returns pages to the OS is up to the allocator.
 *
 * @param obj The mesh.
 * @return Bytes in `verts.block` plus `tris.block`; 0 for unallocated pools.
 *
 * @rank_local
 */
size_t MFTLObjectPoolBytes(const MFTL_Object *obj);

/**
 * @brief Structural check of the pools themselves (not of the mesh).
 *
 * Counts within capacity, owned counts within active counts, and every
 * sub-array pointer exactly where the layout for this capacity puts it inside
 * `block` (NULL for an empty pool). For tests and assertions; mesh validity is
 * MFTLCheckMeshIntegrity().
 *
 * @param obj The mesh.
 * @return `true` if every count, capacity and pointer agrees.
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
 * @brief Declares that the current partition no longer describes this mesh.
 *
 * MFTLRedistribute() leaves a body untouched when no entity changed owner,
 * because a rank's post-redistribution state is a function of (global
 * topology, owner map) only. This stamp reports that something other than the
 * owner map changed, so the body must be rebuilt anyway. Three kinds of change
 * need it, and the library stamps all three:
 *
 * 1. entity counts -- MFTLObjectSetCounts() stamps;
 * 2. global ids -- the remeshers' renumbering stamps (ghost triangle ids are
 *    repaired only by being dropped and re-received in a rebuild);
 * 3. entity flags -- MFTLRemeshMarkFrozen() stamps (a kept ghost vertex keeps
 *    its local record through a redistribution).
 *
 * Any other in-place edit that bypasses MFTLObjectSetCounts() owes this call.
 * A spurious stamp costs one rebuild; a missing one silently corrupts ids.
 *
 * @param obj The mesh; only `partition_dirty` is written.
 *
 * @rank_local Redistribution reduces the per-body votes before acting.
 */
static inline void MFTLObjectMarkPartitionDirty(MFTL_Object *obj) { obj->partition_dirty = MFTL_TRUE; }

/**
 * @brief Rebuilds `tris.verts` and `tris.neighbors` from the half-edge records
 *        in one streaming pass and sets #MFTL_DERIVED_TOPOLOGY
 *        (#MFTL_DERIVED_GEOMETRY is left as it was).
 *
 * `verts[3t+k]` is the ORIGIN of half-edge 3t+k and `neighbors[3t+k]` the
 * triangle across it (#MFTL_ERROR without a twin). Call after any topology
 * change that did not rebuild them itself.
 *
 * @param obj The mesh.
 *
 * @rank_local
 */
void MFTLObjectBuildDerived(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 4. SPATIAL (MORTON) ORDERING
// -----------------------------------------------------------------------------

/** @brief Bits per axis of a Morton key: \f$3 \times 21 = 63\f$ fit a `uint64_t`. */
#define MFTL_MORTON_BITS 21
/** @brief Largest quantised Morton coordinate, \f$2^{21}-1\f$. */
#define MFTL_MORTON_MAX 2097151.0

/**
 * @brief Spreads the low 21 bits of @p x so that bit i moves to bit 3i.
 *
 * Five shift-and-mask steps (magic-number bit interleave), \f$O(\log b)\f$.
 *
 * @param x Value whose low 21 bits are spread; higher bits are masked off.
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
 * @brief 63-bit Morton (Z-order) key of three 21-bit coordinates.
 *
 * \f[
 *   Z(x,y,z) = \sum_{i=0}^{20} \Big( x_i 2^{3i} + y_i 2^{3i+1} + z_i 2^{3i+2} \Big)
 * \f]
 *
 * Ascending \f$Z\f$ traverses the Z-order curve; points in a common octree
 * cell share a key prefix, which the renumbering and the LBVH rely on.
 *
 * @param x Quantised x coordinate, 21 bits.
 * @param y Quantised y coordinate, 21 bits.
 * @param z Quantised z coordinate, 21 bits.
 * @return The interleaved key.
 *
 * @warning Inputs above #MFTL_MORTON_MAX are MASKED, not clamped.
 */
static inline uint64_t MFTLMorton3D(uint64_t x, uint64_t y, uint64_t z)
{
  return MFTLMortonSpread3(x) | (MFTLMortonSpread3(y) << 1) | (MFTLMortonSpread3(z) << 2);
}

/**
 * @brief Renumbers both pools along a Morton curve, so that neighbours in
 *        space are neighbours in memory.
 *
 * The permutation is APPLIED to the pools rather than stored, so there remains
 * one index space and one copy of every coordinate:
 *
 * 1. a Morton key per vertex from `verts.pos`, quantised to 21 bits per axis
 *    against the object's own bounding box with one common scale (cubic cells);
 * 2. a stable sort of `[0, n_local)` and, separately, of `[n_local, n)`, so the
 *    ownership prefix survives; all vertex arrays are permuted and
 *    `he.vertex_idx` remapped;
 * 3. triangles are ordered by their smallest (already Morton-ordered) vertex
 *    index, again owned and ghost separately; records, half-edge triples and
 *    geometry arrays move together, `twin_idx` and `out_he_idx` are remapped;
 * 4. the derived arrays are rebuilt.
 *
 * Global ids are untouched; pool indices are not stable across the call. A body
 * straddling a periodic seam gets a domain-wide box and coarser cells -- a
 * locality cost only. No time benefit has been measured on the development
 * machine (docs/analysis.md); its value is the single index space and LBVH
 * locality.
 *
 * @param obj The mesh; both pools are permuted in place.
 *
 * @rank_local Each rank orders its own partition; no MPI.
 */
void MFTLObjectReorderMorton(MFTL_Object *obj);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. HOT-LOOP ITERATORS
 *
 * Macros, because C has no closures: the body is pasted into one flat loop over
 * contiguous arrays that the optimiser sees whole. Every iterator hoists the
 * arrays into `restrict` locals and the bound into a local, reads connectivity
 * sequentially and keeps seam-corrected edges in registers. The body is
 * variadic (it may contain commas); `break` and `continue` work as in a `for`.
 * Position reads are indexed gathers, so vectorisation is left to the compiler
 * and never forced.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Iterates triangle geometry from compact connectivity.
 *
 * Streams `tris.verts`, gathers the three current positions and forms two
 * minimum-image edge vectors in registers. Periodicity is decided once, outside
 * the loop: the non-periodic branch is gathers and subtractions only, the
 * periodic one applies \f$d - L\,\mathrm{nearbyint}(d/L)\f$ branch-free, with
 * a zero reciprocal on non-periodic axes.
 *
 * In scope for the body:
 *
 *     int    t              triangle pool index
 *     int    v0, v1, v2     vertex pool indices (corner k = origin of half-edge 3t+k)
 *     double p0x, p0y, p0z  stored position of corner 0
 *     double e1x, e1y, e1z  p1 - p0, minimum image
 *     double e2x, e2y, e2z  p2 - p0, minimum image
 *
 * Requires #MFTL_DERIVED_TOPOLOGY (asserted in Debug).
 *
 * @param obj   The mesh.
 * @param count Triangles to visit from index 0: `obj->tris.n_local` for owned
 *              only, `obj->tris.n` to include ghosts.
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
 * @brief Iterates vertex positions, read-only, as one contiguous stream.
 *
 * For reductions over positions (bounding boxes, centres of mass). In scope:
 * `int v` (pool index) and `double px, py, pz` (the stored position).
 *
 * @param obj   The mesh.
 * @param count Vertices to visit from index 0 (`verts.n_local` or `verts.n`).
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
 * @brief Iterates the vertices this rank OWNS, writable -- the advection loop.
 *
 * The ownership prefix is the loop bound, so there is no per-vertex
 * `owner_proc == rank` test. In scope: `int v`, `double *pos` (three writable
 * doubles) and `MFTL_Vertex *rec`.
 *
 * @param obj The mesh.
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
