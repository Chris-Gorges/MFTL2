/**
 * @file mftl_topology.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_topology
 *
 * @brief Half-edge topology: entity flags, implicit index arithmetic,
 *        navigation, and connectivity maintenance.
 *
 * **Everything is an index.** Accessors take and return pool indices
 * (#MFTL_ERROR = "no such entity"); records are reached with MFTLVertexRec(),
 * MFTLTriRec() and MFTLHeRec() when identity fields are wanted, and positions
 * with MFTLVertexPos().
 *
 * **Implicit half-edge indexing** (an invariant, mftl_types.h): half-edge
 * \f$3t+k\f$ belongs to triangle \f$t\f$, so
 *
 * \f[
 *   \mathrm{face}(h) = \lfloor h/3 \rfloor,\quad
 *   \mathrm{next}(h) = 3\lfloor h/3 \rfloor + (h+1) \bmod 3,\quad
 *   \mathrm{prev}(h) = 3\lfloor h/3 \rfloor + (h+2) \bmod 3,\quad
 *   \mathrm{origin}(h) = \mathrm{target}(\mathrm{prev}(h)).
 * \f]
 *
 * None of these is stored, so a broken next-cycle or a half-edge that does not
 * point back at its triangle cannot be represented. The price is a
 * triangles-only mesh. Theory: src/topology/README.md.
 */

#ifndef MFTL_TOPOLOGY_H
#define MFTL_TOPOLOGY_H

#include "mftl_constants.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

// -----------------------------------------------------------------------------
// 0. ENTITY FLAGS
// -----------------------------------------------------------------------------

/**
 * @brief Flag: the entity lies on the surface's TOPOLOGICAL boundary.
 *
 * Set on every half-edge without a partner anywhere in the mesh and on both of
 * its endpoints, by MFTLMarkMeshBoundary() while the mesh is still COMPLETE;
 * afterwards it travels with the entity through redistribution and remeshing.
 * It is a different statement from `twin_idx == MFTL_ERROR`, which means "no
 * partner in THIS partition" and is also true at the outer rim of a ghost halo.
 * Anything that must tell "the surface ends here" from "the partition ends
 * here" uses the flag. On a closed mesh it is clear everywhere.
 */
#define MFTL_FLAG_BOUNDARY 0x1

/* ── The flag word ─────────────────────────────────────────────────────────
 *
 * Every bit is allocated here, including bits only src/remeshing/ reads, so
 * the registry cannot collide:
 *
 *   BOUNDARY  0x01  topology    persistent (a property of the surface)
 *   DEAD      0x02  remeshing   transient, removed by compaction
 *   VISITED   0x04  remeshing   transient, cleared between sweeps
 *   FEATURE   0x08  remeshing   re-derived each pass, carried by splits between
 *   CORNER    0x10  remeshing   re-derived each pass, vertices only
 *   FROZEN    0x20  remeshing   transient, one parallel phase; never transmitted
 *   DONE      0x40  remeshing   transient, one spanning pass; travels (vertices, half-edges)
 *
 * Application code must not use bits 0x02 and 0x04 for its own markers.
 */

/** @brief Flag: entity is retired; removed by the next compaction. */
#define MFTL_FLAG_DEAD 0x2

/** @brief Flag: entity was modified in the current remeshing sweep; do not touch again. */
#define MFTL_FLAG_VISITED 0x4

/**
 * @brief Flag. Half-edge: the angle between the normals of its two triangles
 *        exceeds the feature threshold, or it is a mesh-boundary edge. Vertex: it lies on a
 *        feature curve (smoothed only along the curve).
 *
 * Set on BOTH half-edges of a sharp edge, so the test needs no twin hop.
 */
#define MFTL_FLAG_FEATURE 0x8

/**
 * @brief Flag, vertex only: a feature CORNER, which the smoother never moves.
 *
 * Set where a vertex has one incident feature edge (the end of a ridge), three
 * or more (a genuine corner), or two that meet at an angle sharper than
 * `feature_corner_angle`. A separate bit so the smoother does not re-walk the
 * fan to find out.
 */
#define MFTL_FLAG_CORNER 0x10

/**
 * @brief Flag: this rank may not modify the entity in the current parallel
 *        remeshing phase.
 *
 * Set on every ghost and on every OWNED entity of which another rank holds a
 * copy:
 *
 * \f[ \text{owned entity frozen} \iff \text{some other rank holds it as a ghost}. \f]
 *
 * Two ranks therefore never modify the same entity, and no rank modifies what
 * another is reading -- without any negotiation. MFTLRemeshMarkFrozen() derives
 * it locally from the ownership prefix: an owned triangle is replicated exactly
 * when one of its corners or edge neighbours is a ghost (index `>= n_local`).
 * Recomputed after every migration and never packed onto the wire: it describes
 * this rank's halo.
 */
#define MFTL_FLAG_FROZEN 0x20

/**
 * @brief Flag, vertices and half-edges: this spanning pass has already given the
 *        entity its share of the serial kernel.
 *
 * Set, after a placement's sweeps, on every owned vertex the placement left
 * unfrozen and on every half-edge both of whose ends it left unfrozen, in the
 * current pass (#MFTL_REMESH_SPANNING_ONCE) or the current iteration
 * (#MFTL_REMESH_SPANNING_NESTED). A flagged edge with two flagged ends was
 * offered to split, collapse and flip and nothing at it has moved since, so it
 * is not offered again (two flagged ends alone are not enough: they may have
 * been unfrozen in different placements); in the ONCE mode a flagged vertex is
 * not smoothed again unless an operation touched its fan, and reopening it
 * reopens every edge at it.
 * Such entities have no copy on any other rank, so the flag is set where the
 * record is unique and every later copy inherits it: it travels on the wire and
 * agrees on every holder. Cleared at the start and at the end of the pass.
 */
#define MFTL_FLAG_DONE 0x40

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
 * @brief True if vertex @p v is a ghost (owned by another rank): \f$v \ge \nlocal\f$.
 *
 * Loops over owned entities should use `n_local` as their bound instead.
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
 * @brief True if this rank may not modify vertex @p v (#MFTL_FLAG_FROZEN).
 *
 * The flag, not the index: an owned vertex another rank holds a copy of is
 * frozen too, while an entity created during the phase sits past the ownership
 * prefix until compaction and is nevertheless owned and free.
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
 * @brief Vertex at the START (origin) of half-edge @p h:
 *        \f$\mathrm{target}(\mathrm{prev}(h))\f$.
 *
 * Not routed through the twin, which does not exist on a boundary half-edge.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return Vertex pool index.
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
 * @brief The triangle across half-edge @p h, or #MFTL_ERROR without a local twin.
 *
 * What `tris.neighbors[h]` caches; streaming loops should read the array.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return Triangle index across @p h, or #MFTL_ERROR.
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
 * @brief The three vertices of triangle @p t in counter-clockwise order.
 *
 * Slot k is the ORIGIN of half-edge 3t+k -- the convention of `tris.verts`;
 * mixing conventions silently rotates triangles. Streaming loops should read
 * `tris.verts` instead.
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
 * @brief The next half-edge along a boundary loop (a twin-less half-edge).
 *
 * The boundary is not a stored cycle; it is recovered by rotating around the
 * target \f$w\f$ of @p h: starting at next(h), which leaves \f$w\f$, step
 * counter-clockwise (\f$h \mapsto \mathrm{next}(\mathrm{twin}(h))\f$) until
 * an outgoing half-edge without a twin is reached. Repeated calls traverse a
 * boundary loop consistently and return to the start after one lap.
 *
 * @param obj The mesh object.
 * @param h   A half-edge with no twin (asserted in Debug).
 * @return The next twin-less half-edge, or #MFTL_ERROR if the fan does not
 *         terminate within #MFTL_MAX_VALENCE steps (non-manifold input).
 *
 * @note Partition-dependent like MFTLHeIsLocalBoundary(): on a distributed mesh
 *       the walk follows halo rims as readily as real boundaries.
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
 * @brief Rewinds to the first outgoing half-edge of a vertex's fan.
 *
 * Steps clockwise, \f$h \mapsto \mathrm{twin}(\mathrm{prev}(h))\f$ (the inverse
 * of the counter-clockwise step \f$h \mapsto \mathrm{next}(\mathrm{twin}(h))\f$
 * the one-ring walks use). On an OPEN fan it stops where the rotation runs out
 * of twins, which is the fan's first half-edge. On a CLOSED fan it walks one
 * full lap and returns the counter-clockwise successor of @p start -- any
 * outgoing half-edge is a valid start there, but the lap costs one valence of
 * steps, which is why the walkers accept an `is_inner` shortcut.
 *
 * @param      obj      The mesh.
 * @param      start    Any outgoing half-edge of the vertex.
 * @param[out] open_fan Set to `true` if the fan is open. May be #MFTL_NULL.
 * @return A valid first outgoing half-edge. Bounded by #MFTL_MAX_VALENCE steps;
 *         a fan that does not close asserts in Debug.
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
 * @brief The 1-ring of TRIANGLES around vertex @p v, in counter-clockwise order.
 *
 * @param obj           The mesh object.
 * @param v             Vertex pool index.
 * @param[out] out_tris Buffer for triangle pool indices.
 * @param max_tris      Capacity of @p out_tris (#MFTL_MAX_VALENCE suffices on a
 *                      remeshed mesh). Overflow asserts in Debug, truncates in
 *                      Release.
 * @param is_inner      `true` if the 1-ring is known to be closed: skips the
 *                      rewind. Must be `false` for a boundary vertex.
 * @return Number of triangles written; 0 for an isolated vertex.
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
 * @brief The 1-ring of neighbour VERTICES around vertex @p v, in rotational order.
 *
 * @param      obj       The mesh.
 * @param      v         Vertex pool index.
 * @param[out] out_verts Buffer for the neighbour vertex indices.
 * @param      max_verts Capacity of @p out_verts.
 * @param      is_inner  `true` if the 1-ring is known to be closed: skips the
 *                       rewind. Must be `false` for any boundary vertex.
 * @return Number of neighbours. An interior vertex has as many neighbours as
 *         incident triangles; a vertex on an OPEN fan has one MORE, because
 *         the fan's closing edge is reachable only as an incoming half-edge.
 *         That extra neighbour is emitted first, keeping rotational order.
 *
 * @warning @p is_inner = `true` on a boundary vertex silently drops a neighbour.
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
 * @brief The OUTGOING half-edges of vertex @p v, one per incident triangle.
 *
 * @param      obj      The mesh.
 * @param      v        Vertex pool index.
 * @param[out] out_he   Buffer for the outgoing half-edge indices.
 * @param      max_he   Capacity of @p out_he.
 * @param      is_inner `true` if the 1-ring is known to be closed: skips the rewind.
 * @return One index per incident triangle.
 *
 * @warning On an open fan this is one FEWER than the incident EDGES: the closing
 *          edge exists only as the incoming half-edge `MFTLHePrevIdx(first)`.
 *          Edge iterations must use MFTLGetVertexOneRingVertices() or add it.
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
 * @brief Rebuilds the twin of every half-edge from the vertex indices.
 *
 * Half-edges are bucketed by ORIGIN vertex in an intrusive linked list; the twin
 * of \f$u \to v\f$ is the unlinked half-edge \f$v \to u\f$ in the bucket of
 * \f$v\f$. \f$O(V + H\cdot\bar{d})\f$ with \f$\bar{d}\f$ the mean valence, one
 * scratch allocation, no hashing. Pairing only unlinked candidates keeps the
 * relation symmetric on non-manifold input (surplus half-edges stay twin-less
 * for MFTLCheckMeshIntegrity() to report). Clears the derived bit.
 *
 * @param obj The mesh; every `he.twin_idx` is rewritten.
 *
 * @warning Pairs by vertex SLOT: two bodies sharing slots (e.g. foreign
 *          triangles appended to a pool) would be welded together.
 *
 * @rank_local
 */
void MFTLRebuildTwins(MFTL_Object *obj);

/**
 * @brief Sets #MFTL_FLAG_BOUNDARY on every twin-less half-edge and both its
 *        endpoints, after clearing it everywhere.
 *
 * Run once, after MFTLRebuildTwins(), while the mesh is COMPLETE -- every
 * generator does. On a partitioned mesh a missing twin may just mean "the
 * neighbour is on another rank", so marking then would turn the halo rim into
 * a fake surface boundary (asserted: the object must hold no ghosts).
 *
 * @param obj The mesh; must be complete.
 * @return The number of boundary half-edges; 0 for a closed mesh.
 *
 * @rank_local
 */
int MFTLMarkMeshBoundary(MFTL_Object *obj);

/**
 * @brief Validates the half-edge structure, printing every violation to stderr.
 *
 * Checks: every vertex has an in-range outgoing half-edge whose origin is that
 * vertex (an isolated vertex is reported as an error); every half-edge target is
 * in range; every twin is in range, symmetric (\f$\mathrm{twin}(\mathrm{twin}(h))
 * = h\f$) and has swapped endpoints. Next-cycle and face back-references need no
 * check under implicit indexing. Twin-less half-edges are legal (open surfaces,
 * halo rims) and only counted.
 *
 * It does NOT check that half-edges were created in aligned triples; that is
 * `inv_implicit_indexing()` in tests/test_mesh_invariants.h.
 *
 * @param obj The mesh.
 * @return #MFTL_SUCCESS if every check passes, else #MFTL_ERROR.
 */
int MFTLCheckMeshIntegrity(const MFTL_Object *obj);

/**
 * @brief Labels the edge-connected components of the triangles in the pool.
 *
 * Two triangles share a component when a chain of twin links joins them.
 * Breadth-first search over `twin_idx`, \f$O(T)\f$, one scratch queue. Seeds
 * are taken in ascending triangle index, so components are numbered in order of
 * their smallest triangle -- a function of the pool alone.
 *
 * An #MFTL_Object is assumed to be ONE connected surface (the per-body
 * reduction relies on it, MFTLEnsembleReduceBodies()); a topology change uses
 * this labelling to split a body into one object per component. On a
 * partitioned mesh the answer is LOCAL: parts joined only through another
 * rank's triangles count as separate components.
 *
 * @param obj The mesh.
 * @param[out] labels `[obj->tris.n]` component of each triangle, or #MFTL_NULL
 *                    to only count.
 * @return The number of components; 0 for an empty pool.
 * @rank_local No MPI.
 */
int MFTLLabelConnectedComponents(const MFTL_Object *obj, int *labels);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_TOPOLOGY_H
