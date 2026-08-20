/**
 * @file remesh_internal.h
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief Helpers shared by the remeshing translation units. Not a public header.
 *
 * Almost everything here is static inline and exists to make the operators in
 * remesh_ops.c read as topology rather than as index arithmetic. The two that
 * carry real weight are tri_write() -- which is the only place the
 * "he[3t+k].vertex_idx is the TARGET of corner k" convention is spelled out --
 * and remesh_new_triangle(), which is the only place a half-edge triple comes
 * into existence.
 *
 * The exceptions are at the bottom: the freeze gate the parallel driver imposes
 * on the sweeps, and MFTLRemeshIterate(), the one function with external linkage
 * that both drivers call. It is here rather than in include/ because it is not
 * part of the library's API -- but it still carries the MFTL prefix, because
 * nothing without one may be exported.
 */

#ifndef MFTL_REMESH_INTERNAL_H
#define MFTL_REMESH_INTERNAL_H

#include "mftl_remeshing.h"

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_types.h"

/**
 * @brief One-ring scratch buffer size.
 *
 * Larger than MFTL_MAX_VALENCE (32) on purpose. The library's walkers stop at
 * MFTL_MAX_VALENCE and a boundary vertex reports one MORE neighbour than it has
 * triangles, so a buffer of exactly MFTL_MAX_VALENCE cannot distinguish "valence
 * 32" from "the walk was truncated". With headroom, `count == MFTL_MAX_VALENCE`
 * is an honest answer and truncation is detectable -- and every operator below
 * REFUSES rather than proceeds on a truncated ring.
 */
#define MFTL_REMESH_RING_MAX 64

/* ═══════════════════════════════════════════════════════════════════════════
 * Liveness
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Has triangle @p t been retired by a collapse?
 * @param obj The mesh.
 * @param t   Triangle index.
 * @return `true` if #MFTL_FLAG_DEAD is set. Compaction is the only thing that
 *         ever observes it.
 */
static inline bool tri_is_dead(const MFTL_Object *obj, int t) { return (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) != 0; }

/**
 * @brief Has vertex @p v been retired by a collapse?
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` if #MFTL_FLAG_DEAD is set.
 */
static inline bool vert_is_dead(const MFTL_Object *obj, int v) { return (obj->verts.rec[v].flags & MFTL_FLAG_DEAD) != 0; }

/**
 * @brief Has half-edge @p h been retired with its triangle?
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` if #MFTL_FLAG_DEAD is set.
 */
static inline bool he_is_dead(const MFTL_Object *obj, int h) { return (obj->tris.he[h].flags & MFTL_FLAG_DEAD) != 0; }

/* ═══════════════════════════════════════════════════════════════════════════
 * Writing topology
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Writes triangle @p t as the CCW vertex triple (v0, v1, v2).
 *
 * Half-edge 3t+k runs from v[k] to v[(k+1)%3] and stores only its TARGET, so the
 * triple lands as (v1, v2, v0). This is the single point at which that rotation
 * is expressed; MFTLGetTriangleVertices() and `tris.verts` read it back out and
 * the three must agree, or triangles come out silently rotated.
 *
 * Deliberately does NOT touch twins, flags or global ids: a rewrite has to place
 * each logical edge's flags in its NEW slot, and only the caller knows which
 * slot that is.
 *
 * @param obj The mesh.
 * @param t   Triangle index.
 * @param v0  First corner.
 * @param v1  Second corner.
 * @param v2  Third corner. The three must be distinct (asserted in Debug).
 */
static inline void tri_write(MFTL_Object *obj, int t, int v0, int v1, int v2)
{
  MFTL_ASSERT(v0 != v1 && v1 != v2 && v2 != v0 && "tri_write: degenerate triangle");
  MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
  he[0].vertex_idx = v1;
  he[1].vertex_idx = v2;
  he[2].vertex_idx = v0;
}

/**
 * @brief Makes @p x and @p y twins. Both must be live half-edges.
 * @param obj The mesh.
 * @param x   First half-edge.
 * @param y   Second half-edge.
 */
static inline void he_pair(MFTL_Object *obj, int x, int y)
{
  MFTL_ASSERT(x != MFTL_ERROR && y != MFTL_ERROR);
  obj->tris.he[x].twin_idx = y;
  obj->tris.he[y].twin_idx = x;
}

/**
 * @brief Re-points a rim half-edge at @p outer, which may be MFTL_ERROR.
 *
 * The asymmetry is the point: @p inner has just been written into a new slot, so
 * its twin link must be set unconditionally, while @p outer is an untouched
 * half-edge of a neighbouring triangle that only needs its back-link updated.
 *
 * @param obj   The mesh.
 * @param inner The rewritten half-edge on the patch rim.
 * @param outer Its partner outside the patch, or #MFTL_ERROR.
 */
static inline void he_pair_outer(MFTL_Object *obj, int inner, int outer)
{
  obj->tris.he[inner].twin_idx = outer;
  if (outer != MFTL_ERROR) obj->tris.he[outer].twin_idx = inner;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Allocation
 *
 * BOTH OF THESE MAY REALLOCATE THE POOL BLOCKS. Any pointer into obj->verts.* or
 * obj->tris.* taken before the call is dangling afterwards. Every operator in
 * this module is therefore written to do all of its allocation FIRST, from
 * values it has already copied out, and only then take pointers.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Appends a vertex at @p pos with a fresh global id. @return Its index.
 *
 * ── Why the owned count is not always incremented ──────────────────────────
 *
 * Ownership is an array PREFIX (mftl_types.h, invariant 2), and an append lands
 * at the END of the pool. On a ghost-free object those are the same slot, so the
 * prefix grows with the pool and the entity is owned immediately.
 *
 * On a PARTITIONED object they are not, and incrementing `n_local` anyway --
 * which is what this function did while remeshing was serial-only -- would
 * declare the first ghost to be an owned vertex and shift every subsequent one
 * out from under the prefix. The new entity therefore sits PAST the prefix until
 * the end of the sweep, and is recognisable as this rank's by `owner_proc`;
 * MFTLRemeshCompactObject() re-establishes the prefix when it compacts.
 *
 * Nothing in between may use the index test to decide ownership, which is why
 * the parallel gate is MFTL_FLAG_FROZEN rather than `>= n_local`
 * (mftl_topology.h). A brand-new entity is owned, unreplicated and free, and its
 * cleared flag word says exactly that.
 *
 * @param obj The mesh. **May reallocate both pool blocks.**
 * @param pos Position of the new vertex.
 * @return Its pool index. `out_he_idx` is #MFTL_ERROR; the caller links it.
 */
static inline int remesh_new_vertex(MFTL_Object *obj, const double pos[3])
{
  const int v = obj->verts.n;
  const bool partitioned = (obj->verts.n_local < obj->verts.n);

  MFTLObjectReserveAdditional(obj, 1, 0);
  MFTLObjectSetCounts(obj, v + 1, partitioned ? obj->verts.n_local : v + 1, obj->tris.n, obj->tris.n_local);

  MFTL_Vertex *const rec = &obj->verts.rec[v];
  rec->out_he_idx = MFTL_ERROR;  // the caller links it
  rec->global_id = obj->next_vertex_gid++;
  rec->owner_proc = (obj->domain != MFTL_NULL) ? obj->domain->rank : 0;
  rec->flags = 0;

  obj->verts.pos[3 * v + 0] = pos[0];
  obj->verts.pos[3 * v + 1] = pos[1];
  obj->verts.pos[3 * v + 2] = pos[2];

  return v;
}

/**
 * @brief Appends a triangle AND its three half-edges. @return Its index.
 *
 * There is no separate half-edge allocator, and there cannot be one: the three
 * half-edges of triangle t are 3t..3t+2 in the same block, so they come into
 * existence exactly when it does. Their global ids are 3 * (triangle gid) + k,
 * which is the invariant redistribution relies on to avoid transmitting them.
 *
 * @param obj The mesh. **May reallocate both pool blocks.**
 * @return The new triangle's pool index. Its half-edges are at 3t, 3t+1, 3t+2
 *         with cleared twins and flags.
 */
static inline int remesh_new_triangle(MFTL_Object *obj)
{
  const int t = obj->tris.n;
  const bool partitioned = (obj->tris.n_local < obj->tris.n);  // see remesh_new_vertex

  MFTLObjectReserveAdditional(obj, 0, 1);
  MFTLObjectSetCounts(obj, obj->verts.n, obj->verts.n_local, t + 1, partitioned ? obj->tris.n_local : t + 1);

  MFTL_Triangle *const rec = &obj->tris.rec[t];
  rec->global_id = obj->next_triangle_gid++;
  rec->owner_proc = (obj->domain != MFTL_NULL) ? obj->domain->rank : 0;
  rec->flags = 0;

  MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
  for (int k = 0; k < 3; ++k)
  {
    he[k].vertex_idx = MFTL_ERROR;
    he[k].twin_idx = MFTL_ERROR;
    he[k].global_id = 3 * rec->global_id + k;
    he[k].flags = 0;
  }

  /* Kept in step so that inv_global_ids_are_dense's "next_halfedge_gid == 3 * n"
   * clause still means something after a pass that only creates triangles. */
  obj->next_halfedge_gid = 3 * obj->next_triangle_gid;

  return t;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Geometry helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Twice the area vector of triangle @p t, optionally with one or two of
 *        its corners moved to @p sub_pos.
 *
 * @param      obj     The mesh.
 * @param      t       Triangle index.
 * @param      sub_a   Vertex index to substitute, or #MFTL_ERROR.
 * @param      sub_b   Second vertex index to substitute, or #MFTL_ERROR.
 * @param      sub_pos Position both substituted corners take.
 * @param[out] out     \f$\vect{e}_1 \times \vect{e}_2\f$, un-normalised, so its
 *                     magnitude is twice the area.
 *
 * The substitution is what makes a fold-over test possible without editing the
 * mesh first: an operator asks "what would this triangle's normal be afterwards"
 * and compares. Passing MFTL_ERROR for both @p sub_a and @p sub_b gives the
 * current normal, from the same code path, so the comparison cannot be skewed by
 * two slightly different formulas.
 *
 * Un-normalised on purpose -- its magnitude is twice the area, which the callers
 * want anyway to reject degenerate results.
 */
static inline void tri_area_vector_substituted(const MFTL_Object *obj, int t, int sub_a, int sub_b, const double sub_pos[3], double out[3])
{
  int v[3];
  MFTLGetTriangleVertices(obj, t, v);

  const double *p[3];
  for (int k = 0; k < 3; ++k)
  {
    p[k] = (v[k] == sub_a || v[k] == sub_b) ? sub_pos : MFTLVertexPos(obj, v[k]);
  }

  double e1[3];
  double e2[3];
  MFTLGetMinimumImageVector(obj->domain, e1, p[0], p[1]);
  MFTLGetMinimumImageVector(obj->domain, e2, p[0], p[2]);

  out[0] = e1[1] * e2[2] - e1[2] * e2[1];
  out[1] = e1[2] * e2[0] - e1[0] * e2[2];
  out[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

/**
 * @brief cos of the angle between two area vectors, 0 if either is degenerate.
 *
 * Returning 0 rather than 1 for a degenerate input is what makes this usable as
 * a rejection test: an operation that produces a zero-area triangle should fail
 * the same threshold an inverting one fails, not pass it by accident.
 *
 * @param a First area vector.
 * @param b Second area vector.
 * @return \f$\cos\theta\f$ between them, or 0 if either is degenerate.
 */
static inline double area_vector_cos(const double a[3], const double b[3])
{
  const double la = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
  const double lb = b[0] * b[0] + b[1] * b[1] + b[2] * b[2];
  if (la < MFTL_VERYVERYSMALL || lb < MFTL_VERYVERYSMALL) return 0.0;
  return (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / MFTL_SQRT(la * lb);
}

/**
 * @brief Midpoint of @p pa and @p pb, minimum-image correct and wrapped.
 * @param      domain Supplies the periodicity.
 * @param      pa     First position.
 * @param      pb     Second position.
 * @param[out] out    The midpoint, wrapped back into the domain.
 */
static inline void midpoint_periodic(const MFTL_Domain *domain, const double pa[3], const double pb[3], double out[3])
{
  double d[3];
  MFTLGetMinimumImageVector(domain, d, pa, pb);
  out[0] = pa[0] + 0.5 * d[0];
  out[1] = pa[1] + 0.5 * d[1];
  out[2] = pa[2] + 0.5 * d[2];
  MFTLApplyPeriodicity(domain, out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The parallel gate
 *
 * Every one of these asks the SAME question -- "is every entity this operation
 * would make another rank's copy of stale inside my exclusive set?" -- of the
 * three different modified sets. What counts as modified is what TRAVELS on the
 * wire (mftl_communication.h): a vertex's position and identity, a triangle's
 * identity and its three vertex global ids. `twin_idx` and `out_he_idx` are
 * local index space, rebuilt at every receiver, so re-pointing the rim twins of
 * a frozen neighbour is not a modification and does not need gating.
 *
 * On a serial object no flag is ever set and all three collapse to `false`, so
 * the sweeps that call them are bit-identical to the serial remesher.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Would splitting or flipping @p h modify something this rank does not own?
 *
 * Both operators rewrite the two triangles incident to the edge and nothing else
 * -- a split additionally APPENDS, and an appended entity is by definition not
 * replicated anywhere. So the gate is the two incident triangles, and their
 * corners come along for free: a triangle is frozen whenever any of its corners
 * is a ghost, so an unfrozen triangle's corners are all owned.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` if the operation must be refused. Always `false` on a
 *         ghost-free object, which is what makes the gated sweeps bit-identical
 *         to the serial remesher.
 */
static inline bool freeze_blocks_pair(const MFTL_Object *obj, int h)
{
  if (MFTLTriangleIsFrozen(obj, MFTLHeTriIdx(h))) return MFTL_TRUE;

  const int tw = obj->tris.he[h].twin_idx;
  return (tw != MFTL_ERROR) && MFTLTriangleIsFrozen(obj, MFTLHeTriIdx(tw));
}

/**
 * @brief Would collapsing @p h modify something this rank does not own?
 *
 * The modified set is larger than it looks: the origin vertex dies, the target
 * MOVES, and every triangle in the origin's fan has that corner rewritten. Two
 * flag tests cover all of it, and the reason is worth stating because it is what
 * makes a fan walk unnecessary here:
 *
 *   a vertex is unfrozen  =>  every triangle incident to it is unfrozen
 *
 * which holds by construction -- MFTLRemeshMarkFrozen freezes the corners of
 * every triangle it freezes. So "both endpoints unfrozen" already asserts that
 * the whole fan is this rank's to rewrite, and it additionally guarantees that
 * the fan is locally COMPLETE, because an unfrozen vertex is owned and the
 * redistribution halo is vertex-complete. The link condition, the wing valences
 * and the fold-over test can therefore all be evaluated exactly.
 *
 * @param obj The mesh.
 * @param h   Half-edge index; its ORIGIN is the vertex that would be retired.
 * @return `true` if the collapse must be refused.
 */
static inline bool freeze_blocks_collapse(const MFTL_Object *obj, int h)
{
  return MFTLVertexIsFrozen(obj, MFTLHeOrigin(obj, h)) || MFTLVertexIsFrozen(obj, MFTLHeTarget(obj, h));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Shared between the serial and the parallel driver
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The split/collapse/compact/flip/smooth loop, gated by MFTL_FLAG_FROZEN.
 *
 * The whole of MFTLRemeshObject() except its no-ghost assertion, its stats
 * zeroing, its global-id renumbering and its closing geometry pass -- i.e. the
 * part both drivers need and neither may re-implement, since the sweep ORDER is
 * itself a correctness argument (see the header of remesh_driver.c).
 *
 * MFTLPascalCase despite being private to src/remeshing/, because it has
 * external linkage and nothing without the MFTL prefix may be exported from the
 * library (CLAUDE.md; the same reason MFTLBVHBuildFromPrims is spelled that way
 * in src/bvh/bvh_internal.h). It is absent from include/ and that is what makes
 * it private.
 *
 * @param      obj        The mesh.
 * @param      params     The parameter block.
 * @param[out] stats      May be #MFTL_NULL.
 * @param      iterations Outer iterations to run, overriding `params->iterations`.
 * @return Operations performed.
 */
int MFTLRemeshIterate(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, int iterations);

/* ═══════════════════════════════════════════════════════════════════════════
 * Measure once, decide once
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The public pair -- MFTLRemeshChooseStrategy then MFTLRemeshEnsembleDispersed
 * -- measures the bodies TWICE, and the first measurement throws away the
 * centroids the second one recomputes. That is four Allreduces where two do, and
 * two of them carry 7*n_bodies doubles: at ten thousand droplets, 560 kB per
 * rank that nobody reads.
 *
 * These two take the measurement as an argument instead. The public functions
 * become thin wrappers that measure and forward, so an external caller sees no
 * change; MFTLRemeshEnsemble uses these and pays for one measurement.
 *
 */

/**
 * @brief MFTLRemeshChooseStrategy() with the body measurement returned as well.
 * @param      ens      The ensemble.
 * @param      params   May be #MFTL_NULL for the default radius fraction.
 * @param[out] centroid `[3*num_objects]` area-weighted, seam-consistent, replicated.
 * @param[out] radius   `[num_objects]` max corner distance from it, replicated.
 * @return The resolved strategy, never #MFTL_REMESH_PARALLEL_AUTO.
 * @collective
 */
MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategyMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, double *centroid, double *radius);

/**
 * @brief MFTLRemeshEnsembleDispersed() with the body measurement supplied.
 * @param      ens      The ensemble, remeshed in place.
 * @param      params   The parameter block.
 * @param[out] stats    May be #MFTL_NULL.
 * @param      centroid `[3*num_objects]` from MFTLRemeshChooseStrategyMeasured().
 * @param      radius   `[num_objects]` from the same call.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if a migration failed.
 * @collective
 */
int MFTLRemeshEnsembleDispersedMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, const double *centroid,
                                        const double *radius);

#endif  // MFTL_REMESH_INTERNAL_H
