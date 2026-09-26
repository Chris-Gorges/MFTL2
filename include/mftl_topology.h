/**
 * @file mftl_topology.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_topology
 *
 * @brief Half-edge topology: navigation, and connectivity maintenance.
 *
 * ## EVERYTHING IS AN INDEX
 *
 * Accessors that return `MFTL_Vertex*` / `MFTL_Triangle*` only work while those
 * structs also carry the geometry. They do not: a vertex's position lives at
 * `obj->verts.pos[3*v]`, so a caller handed a pointer to the record has to
 * reverse-engineer the index before it can ask anything geometric. Every
 * accessor here therefore takes and returns
 * INDICES, and the records are reached with MFTLVertexRec / MFTLTriRec /
 * MFTLHeRec when the identity fields are actually wanted.
 *
 * MFTL_ERROR (-1) is the universal "no such entity" index.
 *
 * ## IMPLICIT HALF-EDGE INDEXING
 *
 * Half-edge 3t+k belongs to triangle t, and next(3t+k) == 3t+(k+1)%3. That is an
 * INVARIANT, not a convention that happens to hold after generation: anything
 * creating a triangle creates its three half-edges at 3t..3t+2, and the pools
 * are laid out so that no other arrangement is representable (mftl_types.h).
 *
 * What follows from it:
 *
 *   - `he_idx`, `next_idx` and `triangle_idx` are not stored. The four functions
 *     below compute them from the index alone -- no load, no cache miss, no way
 *     to be inconsistent. A whole family of "the next-cycle is broken" and "the
 *     half-edge does not point back at its triangle" corruptions cannot occur,
 *     which is why MFTLCheckMeshIntegrity is a third of the size it used to be.
 *   - the half-edge array needs no capacity of its own: it is 3x the triangle
 *     capacity, in the same allocation.
 *   - `origin(h) == target(prev(h))` costs one arithmetic op and one load, where
 *     a stored `prev` pointer would need two dependent loads.
 *
 * The price is that the mesh is triangles-only. It already was, everywhere.
 */

#ifndef MFTL_TOPOLOGY_H
#define MFTL_TOPOLOGY_H

#include "mftl_constants.h"
#include "mftl_types.h"

// -----------------------------------------------------------------------------
// 0. ENTITY FLAGS
// -----------------------------------------------------------------------------

/**
 * @brief `flags` bit marking an entity on the mesh's TOPOLOGICAL boundary.
 *
 * Set on every half-edge that has no neighbouring triangle anywhere in the mesh,
 * and on both endpoints of every such half-edge. Written by MFTLMarkMeshBoundary()
 * while the mesh is still COMPLETE (i.e. at generation time); from then on it
 * travels with the entity, so a redistributed open surface still knows which of
 * its edges are real boundary edges.
 *
 * This is deliberately NOT the same statement as `twin_idx == MFTL_ERROR`:
 *
 *   twin_idx == MFTL_ERROR   "no partner in THIS partition" -- true at a mesh
 *                            boundary AND at the outer rim of a ghost halo, so
 *                            it changes as the partition changes
 *   MFTL_FLAG_BOUNDARY       "no partner anywhere in the mesh" -- a property of
 *                            the surface itself, identical on every rank
 *
 * On a closed mesh the flag is clear everywhere, so code written against it
 * behaves identically on closed and open surfaces. Anything that must tell "the
 * surface ends here" from "the partition ends here" -- remeshing, boundary
 * conditions, halo completeness -- has to use the flag, never the index.
 */
#define MFTL_FLAG_BOUNDARY 0x1

/* ── The rest of the flag word ────────────────────────────────────────────────
 *
 * Every bit of `flags` is allocated HERE, including bits only one module reads,
 * for the same reason every struct is defined in mftl_types.h: a bit registry
 * split across headers is a bit registry with a collision in its future. Layer 2
 * defining a constant that only layer 3 uses introduces no dependency.
 *
 * Which module owns each bit, and what survives a full remeshing pass:
 *
 *   BOUNDARY   src/topology      persistent  -- a property of the surface
 *   DEAD       src/remeshing     transient   -- cleared by compaction, which is
 *                                the only thing that ever observes it
 *   VISITED    src/remeshing     transient   -- cleared between sweeps
 *   FEATURE    src/remeshing     persistent  -- re-derived per pass by
 *                                MFTLDetectFeatures, but travels with the
 *                                entity in between (a split must hand it to
 *                                both halves, or a ridge develops a gap)
 *   CORNER     src/remeshing     persistent  -- vertices only
 *   FROZEN     src/remeshing     transient   -- one parallel remeshing phase.
 *                                Recomputed from the partition after every
 *                                migration, never transmitted: it is a
 *                                statement about THIS rank's halo, so a
 *                                received copy of it would be another rank's
 *                                answer to a different question.
 */

/** @brief Entity is retired and will be removed by the next compaction. */
#define MFTL_FLAG_DEAD 0x2

/** @brief Entity was modified in the current remeshing sweep; do not touch again. */
#define MFTL_FLAG_VISITED 0x4

/**
 * @brief Half-edge: the dihedral angle across it exceeds the feature threshold.
 *        Vertex: it lies on a feature curve (exactly two incident feature edges).
 *
 * Set on BOTH half-edges of a sharp edge, so the test needs no twin hop.
 */
#define MFTL_FLAG_FEATURE 0x8

/**
 * @brief Vertex: a feature CORNER -- positionally locked by the smoother.
 *
 * Deliberately a separate bit from FEATURE rather than a count, because the
 * distinction is what the smoother branches on and re-deriving it means walking
 * the fan again for every vertex on every smoothing sweep. Set when a vertex has
 * one incident feature edge (the free end of a ridge, which would otherwise be
 * dragged off it) or three or more (a genuine corner).
 */
#define MFTL_FLAG_CORNER 0x10

/**
 * @brief The entity may not be MODIFIED by this rank in this remeshing phase.
 *
 * Set on every entity of which another rank holds a copy, and on every ghost.
 * That is the whole of the parallel remeshing correctness argument, so it is
 * worth stating as an equivalence rather than as a rule of thumb:
 *
 *     an owned entity is FROZEN  <=>  some other rank holds it as a ghost
 *
 * Two ranks therefore never modify the same entity, and no rank ever modifies
 * something another rank is reading -- not because they negotiate, but because
 * the sets are disjoint by construction. There is no message anywhere in the
 * scheme that says "may I".
 *
 * The equivalence is DERIVED, not communicated. MFTLRemeshMarkFrozen() reads it
 * straight off the ownership prefix: a triangle is replicated exactly when one
 * of its corners or one of its edge neighbours is a ghost, and both are the
 * index test `>= n_local`. See src/remeshing/remesh_parallel.c.
 *
 * TRANSIENT: recomputed from scratch after every migration and meaningless
 * outside a parallel remeshing pass. It is never packed onto the wire -- it
 * describes this rank's halo, so a neighbour's value answers a different
 * question.
 */
#define MFTL_FLAG_FROZEN 0x20

// -----------------------------------------------------------------------------
// 1. IMPLICIT HALF-EDGE ARITHMETIC
// -----------------------------------------------------------------------------

/**
 * @brief Index of triangle @p t's first half-edge. Its three are 3t, 3t+1, 3t+2.
 * @param t Triangle index.
 * @return \f$3t\f$.
 */
static inline int MFTLTriFirstHe(int t) { return 3 * t; }

/**
 * @brief The triangle half-edge @p h belongs to.
 * @param h Half-edge index.
 * @return \f$\lfloor h/3 \rfloor\f$. No load, no cache miss, no way to be inconsistent.
 */
static inline int MFTLHeTriIdx(int h) { return h / 3; }

/**
 * @brief The corner slot (0, 1 or 2) of half-edge @p h within its triangle.
 * @param h Half-edge index.
 * @return \f$h \bmod 3\f$.
 */
static inline int MFTLHeCorner(int h) { return h % 3; }

/**
 * @brief Next half-edge in the triangle's CCW cycle.
 * @param h Half-edge index.
 * @return \f$3\lfloor h/3 \rfloor + (h+1) \bmod 3\f$.
 */
static inline int MFTLHeNextIdx(int h)
{
  const int base = h - h % 3;
  return base + (h + 1 - base) % 3;
}

/**
 * @brief Previous half-edge in the triangle's CCW cycle (== next twice).
 * @param h Half-edge index.
 * @return \f$3\lfloor h/3 \rfloor + (h+2) \bmod 3\f$.
 */
static inline int MFTLHePrevIdx(int h)
{
  const int base = h - h % 3;
  return base + (h + 2 - base) % 3;
}

/**
 * @brief Number of active half-edges: three per triangle, always.
 * @param obj The mesh.
 * @return \f$3 n_\mathrm{tris}\f$, owned and ghost together.
 */
static inline int MFTLNumHalfEdges(const MFTL_Object *obj) { return 3 * obj->tris.n; }

/**
 * @brief Number of locally owned half-edges: three per owned triangle.
 * @param obj The mesh.
 * @return \f$3 n_\mathrm{local}\f$. They occupy the prefix \f$[0, 3\nlocal)\f$.
 */
static inline int MFTLNumLocalHalfEdges(const MFTL_Object *obj) { return 3 * obj->tris.n_local; }

// -----------------------------------------------------------------------------
// 2. RECORD ACCESSORS
// -----------------------------------------------------------------------------

/**
 * @brief The identity/topology record of vertex @p v.
 * @param obj The mesh.
 * @param v   Vertex index. Bounds-asserted in Debug.
 * @return Pointer into the pool's AoS array. The POSITION is not here -- it is
 *         `obj->verts.pos[3*v]`, see MFTLVertexPos().
 */
static inline MFTL_Vertex *MFTLVertexRec(const MFTL_Object *obj, int v)
{
  MFTL_ASSERT(v >= 0 && v < obj->verts.n);
  return &obj->verts.rec[v];
}

/**
 * @brief The identity record of triangle @p t.
 * @param obj The mesh.
 * @param t   Triangle index. Bounds-asserted in Debug.
 * @return Pointer into the pool's AoS array. 12 bytes of pure identity: the
 *         topology is implicit and the geometry is in the SoA arrays.
 */
static inline MFTL_Triangle *MFTLTriRec(const MFTL_Object *obj, int t)
{
  MFTL_ASSERT(t >= 0 && t < obj->tris.n);
  return &obj->tris.rec[t];
}

/**
 * @brief The topology record of half-edge @p h.
 * @param obj The mesh.
 * @param h   Half-edge index. Bounds-asserted in Debug.
 * @return Pointer into the pool's AoS array. Only `vertex_idx` and `twin_idx`
 *         are topology; `next` and `face` are arithmetic.
 */
static inline MFTL_HalfEdge *MFTLHeRec(const MFTL_Object *obj, int h)
{
  MFTL_ASSERT(h >= 0 && h < 3 * obj->tris.n);
  return &obj->tris.he[h];
}

// -----------------------------------------------------------------------------
// 3. OWNERSHIP
// -----------------------------------------------------------------------------

/**
 * @brief True if vertex @p v is a ghost, i.e. owned by another rank.
 *
 * Derived from the prefix invariant rather than stored. A loop over owned
 * entities should use the count as its bound instead of testing this per
 * element -- that is the whole reason ownership is a prefix.
 *
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` iff \f$v \ge \nlocal\f$.
 */
static inline bool MFTLVertexIsGhost(const MFTL_Object *obj, int v) { return v >= obj->verts.n_local; }

/**
 * @brief True if triangle @p t is a ghost. See MFTLVertexIsGhost().
 * @param obj The mesh.
 * @param t   Triangle index.
 * @return `true` iff @p t lies past the ownership prefix.
 */
static inline bool MFTLTriangleIsGhost(const MFTL_Object *obj, int t) { return t >= obj->tris.n_local; }

/**
 * @brief True if half-edge @p h belongs to a ghost triangle.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` iff \f$\lfloor h/3 \rfloor\f$ lies past the ownership prefix.
 */
static inline bool MFTLHeIsGhost(const MFTL_Object *obj, int h) { return MFTLHeTriIdx(h) >= obj->tris.n_local; }

/**
 * @brief True if this rank may not modify vertex @p v (see MFTL_FLAG_FROZEN).
 *
 * The flag rather than the index, and the difference matters in both
 * directions: a GHOST is always frozen, but so is an OWNED entity that a
 * neighbour holds a copy of -- and an entity created by this rank during the
 * phase is owned, unreplicated and free even though it sits past the ownership
 * prefix until the next compaction.
 *
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` iff #MFTL_FLAG_FROZEN is set on the record.
 */
static inline bool MFTLVertexIsFrozen(const MFTL_Object *obj, int v) { return (obj->verts.rec[v].flags & MFTL_FLAG_FROZEN) != 0; }

/**
 * @brief True if this rank may not modify triangle @p t. See MFTLVertexIsFrozen().
 * @param obj The mesh.
 * @param t   Triangle index.
 * @return `true` iff #MFTL_FLAG_FROZEN is set on the record.
 */
static inline bool MFTLTriangleIsFrozen(const MFTL_Object *obj, int t) { return (obj->tris.rec[t].flags & MFTL_FLAG_FROZEN) != 0; }

// -----------------------------------------------------------------------------
// 4. HALF-EDGE NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief Vertex at the END (target) of half-edge @p h.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return Vertex pool index. The only topological field a half-edge stores
 *         besides its twin.
 */
static inline int MFTLHeTarget(const MFTL_Object *obj, int h) { return obj->tris.he[h].vertex_idx; }

/**
 * @brief Vertex at the START (origin) of half-edge @p h.
 *
 * \f$\mathrm{origin}(h) = \mathrm{target}(\mathrm{prev}(h))\f$. Deliberately not
 * routed through the twin: that would not exist on a boundary half-edge, where
 * the origin very much does.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return Vertex pool index. One arithmetic op and one load.
 */
static inline int MFTLHeOrigin(const MFTL_Object *obj, int h) { return obj->tris.he[MFTLHePrevIdx(h)].vertex_idx; }

/**
 * @brief Twin of half-edge @p h, or #MFTL_ERROR if it has none in this partition.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return The oppositely oriented half-edge, or #MFTL_ERROR.
 * @warning #MFTL_ERROR here means "no partner in THIS partition", which is also
 *          true at a ghost halo's outer rim. For "no partner anywhere in the
 *          mesh" use MFTLHeIsMeshBoundary().
 */
static inline int MFTLHeTwin(const MFTL_Object *obj, int h) { return obj->tris.he[h].twin_idx; }

/**
 * @brief True if @p h has no twin in the LOCAL pools (mesh boundary OR halo rim).
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` iff `twin_idx == MFTL_ERROR`. **Partition-dependent** -- see
 *         MFTLHeIsMeshBoundary() for the partition-independent statement.
 */
static inline bool MFTLHeIsLocalBoundary(const MFTL_Object *obj, int h) { return obj->tris.he[h].twin_idx == MFTL_ERROR; }

/**
 * @brief True if @p h lies on the mesh's topological boundary.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` iff #MFTL_FLAG_BOUNDARY is set. Identical on every rank.
 * @note Meaningful only after MFTLMarkMeshBoundary() has run on the complete mesh.
 */
static inline bool MFTLHeIsMeshBoundary(const MFTL_Object *obj, int h) { return (obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY) != 0; }

/**
 * @brief True if vertex @p v lies on the mesh's topological boundary.
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` iff #MFTL_FLAG_BOUNDARY is set. Identical on every rank.
 * @note Meaningful only after MFTLMarkMeshBoundary() has run on the complete mesh.
 */
static inline bool MFTLVertexIsMeshBoundary(const MFTL_Object *obj, int v) { return (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY) != 0; }

/**
 * @brief The triangle across half-edge @p h, or MFTL_ERROR at a boundary.
 *
 * This is exactly what `tris.neighbors[h]` caches. Use the array in a streaming
 * loop (12 bytes per triangle instead of 48) and this function everywhere else.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return Triangle index across @p h, or #MFTL_ERROR if it has no twin here.
 */
static inline int MFTLHeAdjacentTriangle(const MFTL_Object *obj, int h)
{
  const int twin = obj->tris.he[h].twin_idx;
  return (twin == MFTL_ERROR) ? MFTL_ERROR : MFTLHeTriIdx(twin);
}

// -----------------------------------------------------------------------------
// 5. TRIANGLE NAVIGATION
// -----------------------------------------------------------------------------

/**
 * @brief The three vertices of triangle @p t in CCW order.
 *
 * Slot k is the ORIGIN of half-edge 3t+k, which is the convention
 * `tris.verts` follows -- the two must agree, or code that mixes them silently
 * rotates its triangles.
 *
 * Three loads with a 16-byte stride, all inside the one or two cache lines the
 * triangle's half-edges occupy. In a loop over many triangles prefer
 * `tris.verts`, which is contiguous.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] out Three vertex pool indices, in CCW order.
 */
static inline void MFTLGetTriangleVertices(const MFTL_Object *obj, int t, int out[3])
{
  const MFTL_HalfEdge *he = &obj->tris.he[3 * t];
  out[0] = he[2].vertex_idx;  // origin of he0 == target of he2
  out[1] = he[0].vertex_idx;
  out[2] = he[1].vertex_idx;
}

/**
 * @brief The three neighbouring triangles of @p t, indexed like its half-edges.
 *
 * Slot k is the triangle across half-edge \f$3t+k\f$, matching the convention
 * `tris.neighbors` follows.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] out Three triangle pool indices, #MFTL_ERROR where the edge has no
 *                 twin in this partition.
 */
static inline void MFTLGetTriangleNeighbors(const MFTL_Object *obj, int t, int out[3])
{
  const MFTL_HalfEdge *he = &obj->tris.he[3 * t];
  for (int k = 0; k < 3; ++k)
  {
    out[k] = (he[k].twin_idx == MFTL_ERROR) ? MFTL_ERROR : MFTLHeTriIdx(he[k].twin_idx);
  }
}

// -----------------------------------------------------------------------------
// 6. BOUNDARY WALKING
// -----------------------------------------------------------------------------

/**
 * @brief The next half-edge along an OPEN mesh's boundary.
 *
 * A boundary edge has no twin, so the boundary is not a half-edge cycle the way
 * a triangle is -- there is no "outer face" to walk. It is reconstructed instead
 * by rotating around the shared vertex: @p h ends at some vertex w, and the
 * boundary continues along the one outgoing half-edge of w that has no twin.
 * Starting from next(h) (outgoing from w by construction) and rotating through
 * the fan until the twin runs out finds exactly that edge.
 *
 * Successive calls therefore traverse a boundary loop in a consistent direction
 * and return to the starting half-edge after exactly as many steps as the loop
 * is long, which is what makes "the boundary is a union of closed loops"
 * checkable.
 *
 * @param obj The mesh object.
 * @param h   A half-edge with no twin (asserted in Debug).
 * @return The next boundary half-edge, or MFTL_ERROR if the fan is not manifold.
 *
 * @note LOCAL statement, like MFTLHeIsLocalBoundary: on a distributed mesh the
 *       walk follows partition rims as readily as real mesh boundaries.
 */
static inline int MFTLHeNextBoundary(const MFTL_Object *obj, int h)
{
  MFTL_ASSERT(MFTLHeIsLocalBoundary(obj, h) && "MFTLHeNextBoundary: half-edge is not on a boundary");

  int curr = MFTLHeNextIdx(h);

  /* Bounded rather than while(true): a corrupt (non-manifold) fan would
   * otherwise spin forever inside an inline function with no diagnostics. */
  for (int guard = 0; guard <= MFTL_MAX_VALENCE; ++guard)
  {
    const int twin = obj->tris.he[curr].twin_idx;
    if (twin == MFTL_ERROR) return curr;
    curr = MFTLHeNextIdx(twin);
  }

  MFTL_ASSERT(0 && "MFTLHeNextBoundary: fan around the target vertex does not close");
  return MFTL_ERROR;
}

// -----------------------------------------------------------------------------
// 7. VERTEX ONE-RING TRAVERSAL
// -----------------------------------------------------------------------------

/**
 * @brief Rewinds to the first outgoing half-edge of @p v's fan.
 *
 * On a closed fan any half-edge is a valid start, so this returns @p start
 * unchanged after one lap. On an open fan it walks clockwise until the rotation
 * runs out of twins, which is where the fan begins.
 *
 * The clockwise step is \f$h \mapsto \mathrm{twin}(\mathrm{prev}(h))\f$, the
 * inverse of the counter-clockwise step the one-ring walks use.
 *
 * @param      obj      The mesh.
 * @param      start    Any outgoing half-edge of the vertex.
 * @param[out] open_fan Set to `true` if the fan is open. May be #MFTL_NULL.
 * @return The first outgoing half-edge of the fan. Bounded by #MFTL_MAX_VALENCE
 *         steps, so a corrupt non-manifold fan asserts rather than spinning.
 */
static inline int MFTLVertexFanStart(const MFTL_Object *obj, int start, bool *open_fan)
{
  int curr = start;
  if (open_fan) *open_fan = MFTL_FALSE;

  for (int guard = 0; guard <= MFTL_MAX_VALENCE; ++guard)
  {
    const int cw = obj->tris.he[MFTLHePrevIdx(curr)].twin_idx;
    if (cw == MFTL_ERROR)
    {
      if (open_fan) *open_fan = MFTL_TRUE;
      return curr;
    }
    if (cw == start) return curr;
    curr = cw;
  }

  MFTL_ASSERT(0 && "MFTLVertexFanStart: fan does not close within MFTL_MAX_VALENCE");
  return curr;
}

/**
 * @brief The 1-ring of TRIANGLES around vertex @p v.
 *
 * @param obj       The mesh object.
 * @param v         Vertex pool index.
 * @param[out] out_tris Buffer for triangle pool indices.
 * @param max_tris  Capacity of @p out_tris (MFTL_MAX_VALENCE is normally plenty).
 * @param is_inner  Set true when the vertex is known to have a closed 1-ring;
 *                  skips the rewind. Must be false for any boundary vertex.
 * @return Number of triangles written.
 */
static inline int MFTLGetVertexOneRing(const MFTL_Object *obj, int v, int *out_tris, int max_tris, bool is_inner)
{
  int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return 0;  // isolated vertex

  if (!is_inner) start = MFTLVertexFanStart(obj, start, MFTL_NULL);

  int count = 0;
  int curr = start;
  do
  {
    MFTL_ASSERT(count < max_tris && "MFTLGetVertexOneRing: buffer overflow");
    if (count >= max_tris) break;

    out_tris[count++] = MFTLHeTriIdx(curr);

    const int twin = obj->tris.he[curr].twin_idx;
    if (twin == MFTL_ERROR) break;  // boundary stop
    curr = MFTLHeNextIdx(twin);
  } while (curr != start);

  return count;
}

/**
 * @brief The 1-ring of neighbour VERTICES around vertex @p v.
 *
 * @param      obj       The mesh.
 * @param      v         Vertex pool index.
 * @param[out] out_verts Buffer for the neighbour vertex indices.
 * @param      max_verts Capacity of @p out_verts.
 * @param      is_inner  Set `true` when the vertex is known to have a closed
 *                       1-ring; skips the rewind. Must be `false` for any
 *                       boundary vertex.
 * @return Number of neighbours. A vertex with a closed 1-ring has as many
 *         neighbours as incident triangles; a BOUNDARY vertex has one MORE,
 *         because its fan is bounded by two edges rather than closing on itself.
 *         That extra neighbour is emitted first, so the result stays in
 *         rotational order.
 *
 * @warning @p is_inner skips the rewind AND the extra-neighbour handling.
 *          Passing `true` for a boundary vertex silently drops a neighbour.
 */
static inline int MFTLGetVertexOneRingVertices(const MFTL_Object *obj, int v, int *out_verts, int max_verts, bool is_inner)
{
  int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return 0;

  bool open_fan = MFTL_FALSE;
  if (!is_inner) start = MFTLVertexFanStart(obj, start, &open_fan);

  int count = 0;

  /* The walk below steps through v's OUTGOING half-edges, and on an open fan the
   * far endpoint of the last boundary edge has none: that edge reaches v as an
   * INCOMING half-edge whose twin -- the outgoing direction -- is the one that
   * does not exist. next(start) is the far edge of the fan's first triangle, so
   * its target is exactly that missing neighbour. */
  if (open_fan)
  {
    MFTL_ASSERT(count < max_verts && "MFTLGetVertexOneRingVertices: buffer overflow");
    if (count < max_verts) out_verts[count++] = obj->tris.he[MFTLHeNextIdx(start)].vertex_idx;
  }

  int curr = start;
  do
  {
    MFTL_ASSERT(count < max_verts && "MFTLGetVertexOneRingVertices: buffer overflow");
    if (count >= max_verts) break;

    out_verts[count++] = obj->tris.he[curr].vertex_idx;

    const int twin = obj->tris.he[curr].twin_idx;
    if (twin == MFTL_ERROR) break;
    curr = MFTLHeNextIdx(twin);
  } while (curr != start);

  return count;
}

/**
 * @brief The 1-ring of OUTGOING half-edges of vertex @p v.
 *
 * @param      obj      The mesh.
 * @param      v        Vertex pool index.
 * @param[out] out_he   Buffer for the outgoing half-edge indices.
 * @param      max_he   Capacity of @p out_he.
 * @param      is_inner Set `true` when the 1-ring is known to be closed; skips
 *                      the rewind.
 * @return One index per incident triangle.
 *
 * @warning On a boundary vertex this is one FEWER than the number of incident
 * EDGES: the fan's closing edge exists only in the incoming direction. A
 * consumer iterating edges (edge lengths, umbrella smoothing, remeshing
 * candidates) would silently skip it -- use MFTLGetVertexOneRingVertices, which
 * does return the complete neighbour set, or reach the missing edge as
 * MFTLHePrevIdx(first) where `first` is the first index returned here.
 */
static inline int MFTLGetVertexOneRingHalfedges(const MFTL_Object *obj, int v, int *out_he, int max_he, bool is_inner)
{
  int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return 0;

  if (!is_inner) start = MFTLVertexFanStart(obj, start, MFTL_NULL);

  int count = 0;
  int curr = start;
  do
  {
    MFTL_ASSERT(count < max_he && "MFTLGetVertexOneRingHalfedges: buffer overflow");
    if (count >= max_he) break;

    out_he[count++] = curr;

    const int twin = obj->tris.he[curr].twin_idx;
    if (twin == MFTL_ERROR) break;
    curr = MFTLHeNextIdx(twin);
  } while (curr != start);

  return count;
}

// -----------------------------------------------------------------------------
// 8. CONNECTIVITY MAINTENANCE  (see src/topology/topology.c)
// -----------------------------------------------------------------------------

/**
 * @brief Rebuilds the twin index of every half-edge from the vertex indices.
 *
 * Pairs each half-edge (u -> v) with the half-edge (v -> u) by bucketing them on
 * their origin vertex -- O(V + H), no hashing, one scratch allocation. Half-edges
 * left without a partner keep `twin_idx == MFTL_ERROR`.
 *
 * Call this after any operation that creates half-edges without linking them.
 * It also invalidates the derived arrays of @p obj, since `tris.neighbors` is a
 * function of the twins.
 *
 * @note Local (serial) operation -- no MPI.
 *
 * @param obj The mesh. Every `he.twin_idx` is rewritten.
 */
void MFTLRebuildTwins(MFTL_Object *obj);

/**
 * @brief Records which half-edges and vertices lie on the mesh's boundary.
 *
 * Clears MFTL_FLAG_BOUNDARY on every half-edge and vertex, then sets it on every
 * half-edge without a twin and on both endpoints of each such half-edge. Run it
 * once per mesh, immediately after MFTLRebuildTwins, while the mesh is still
 * COMPLETE -- every generator does this, so a caller normally never has to.
 *
 * The complete-mesh precondition is the whole point: on a partitioned mesh a
 * missing twin also means "the neighbour is on another rank", and marking those
 * would turn the halo rim into a fake surface boundary.
 *
 * @note Local (serial) operation -- no MPI, and it asserts in Debug that the
 *       object holds no ghosts.
 *
 * @param obj The mesh. Must be COMPLETE -- see the note above.
 * @return The number of boundary half-edges found; 0 means the mesh is closed.
 */
int MFTLMarkMeshBoundary(MFTL_Object *obj);

/**
 * @brief Validates the internal consistency of the half-edge mesh.
 *
 * Checks index ranges, twin symmetry, twin vertex agreement, and
 * vertex -> outgoing-half-edge consistency. Diagnostics for every violation go
 * to stderr.
 *
 * Notably absent, because implicit indexing makes them unrepresentable: the
 * next-cycle check, the triangle <-> half-edge back-reference check, and the
 * "triangle has no half-edge" check. Those are three of the six failure modes an
 * explicitly-indexed half-edge mesh has to test for, and here they are not
 * expressible.
 *
 * A half-edge without a twin is accepted -- an open surface is a legitimate
 * mesh -- and the count is reported instead, which is the cheapest way to notice
 * that a mesh expected to be closed is not.
 *
 * @param obj The mesh.
 * @return #MFTL_SUCCESS if every structural check passes, else #MFTL_ERROR.
 */
int MFTLCheckMeshIntegrity(const MFTL_Object *obj);

/**
 * @brief Labels the edge-connected components of the triangles in the pool.
 *
 * Two triangles are in one component when a chain of twin links joins them. A
 * breadth-first sweep over `twin_idx`, O(T), one scratch queue, no recursion.
 *
 * ── What it is for ─────────────────────────────────────────────────────────
 *
 * The library assumes an #MFTL_Object is ONE connected surface (the per-body
 * reduction's holder route rests on it, see MFTLEnsembleReduceBodies). A
 * topology change is exactly the operation that can break that assumption --
 * a pinched-off drop is a second component of the body it left -- so the
 * topology-change module labels components to split a body into one object per
 * component. It is also the cheap way for a caller to COUNT drops.
 *
 * ── On a partitioned mesh ───────────────────────────────────────────────────
 *
 * It is purely local: every triangle in the pool, owned and ghost, and nothing
 * else. Two parts of a body joined only through triangles another rank holds
 * come out as two local components. That is the correct local answer and the
 * input the distributed labelling in src/topology_change/ starts from; it is not
 * the global component count.
 *
 * @param obj The mesh.
 * @param[out] labels `[obj->tris.n]` component of each triangle, numbered from 0
 *                    in order of each component's smallest triangle index. May
 *                    be #MFTL_NULL to only count.
 * @return The number of components; 0 for an empty pool.
 * @rank_local No MPI.
 */
int MFTLLabelConnectedComponents(const MFTL_Object *obj, int *labels);

#endif  // MFTL_TOPOLOGY_H
