/**
 * @file mftl_communication.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Distributed memory: redistribution, ghost synchronisation, the spatial
 *        halo, and the exchange engine they share.
 *
 * Three kinds of neighbour data:
 *
 * | Call | Question | Result |
 * |---|---|---|
 * | MFTLRedistribute() | the mesh moved -- who owns what now? | re-partitioned pools with a vertex-complete topological halo |
 * | MFTLUpdateGhostPositions() | the mesh deformed -- refresh my ghosts | ghost positions pulled from their owners; topology untouched |
 * | MFTLExtendSpatialHaloNearBodies() | what of body B is NEAR my share of body A? | one shadow object per nearby foreign body |
 *
 * **Ownership and halo rule.** A vertex belongs to the rank whose Cartesian
 * cell contains its position, a triangle to the rank whose cell contains its
 * cached centre (MFTLGetOwnerProc()). An owned triangle is additionally sent to
 * the owner of each of its three corners and of each of its three edge
 * neighbours; that surplus IS the ghost layer. It is VERTEX-COMPLETE (every
 * triangle incident to an owned vertex is present locally, which makes vertex
 * normals communication-free) and gives every owned triangle all three edge
 * neighbours.
 *
 * **Wire format.** Only topology, identity and vertex POSITIONS travel
 * (40 B per vertex, 40 B per triangle with its three half-edges). Half-edge ids
 * (\f$3\,t_\mathrm{gid}+k\f$), `next`/face (arithmetic), twins and
 * `out_he_idx` are reconstructed by the receiver.
 *
 * **Postcondition of a redistribution.** Valid topology, valid positions,
 * #MFTL_DERIVED_TOPOLOGY set; `tris.area/normal/center` and `verts.normal`
 * UNSPECIFIED. With redistribution the per-timestep loop is
 *
 *   1. move owned vertices                      #MFTL_FOREACH_OWNED_VERTEX
 *   2. refresh ghost positions                  MFTLUpdateGhostPositions()
 *   3. geometry of ALL triangles, ghosts too    MFTLComputeTriangleGeometry(obj, obj->tris.n)
 *   4. re-partition                             MFTLRedistribute()
 *   5. (if geometry is read afterwards)         MFTLEnsembleUpdate() / MFTLEnsembleComputeTriangleGeometry()
 *
 * Step 2 precedes 3 because a triangle's corners may belong to three ranks; step
 * 3 must cover ghosts because redistribution routes ghosts by their cached
 * centre too, and two ranks disagreeing about a ghost's destination silently
 * cost an owned triangle an edge neighbour (a Debug assertion checks the
 * centres).
 *
 * **Preconditions.** MFTLRegisterMpiTypes() has been called; every rank added
 * the same bodies in the same order (body ids are the names on the wire); the
 * triangle centres of every local object are current, ghosts included
 * (MFTLEnsembleComputeTriangleGeometry()) -- including before the FIRST
 * redistribution of a freshly generated ensemble, whose centres are unset.
 * Stale centres route triangles to wrong owners without any error in Release.
 *
 * Theory: src/communication/README.md.
 */

#ifndef MFTL_COMMUNICATION_H
#define MFTL_COMMUNICATION_H

#include <stdbool.h>
#include <stddef.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Wire format
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_PackedVertex
 * @brief Wire image of a vertex: position and identity, 40 bytes.
 *
 * One run of doubles then one run of ints, so the MPI datatype is two
 * contiguous blocks (pinned by static assertions).
 */
typedef struct
{
  double pos[3];  ///< AUTHORITATIVE position, wrapped into the domain.

  int global_id;  ///< Stable name of this vertex across ranks and reorderings.
  int owner_proc;  ///< The NEW owner, decided by the sender from @c pos.
  int flags;  ///< MFTL_FLAG_BOUNDARY travels with the entity; it cannot be re-derived.
  int body_gid;  ///< #MFTL_Object::global_id of the owning body. See the struct note.
} MFTL_PackedVertex;

/**
 * @struct MFTL_PackedTriangleComplex
 * @brief Wire image of a triangle AND its three half-edges: ten ints, 40 bytes.
 *
 * Slot k is half-edge 3t+k, so its triangle and its successor are implicit and
 * only the target vertex and the flags travel.
 *
 * @note `vertex_gid[k]` is the global id of the TARGET of half-edge 3t+k
 *       (`verts.rec[he[3t+k].vertex_idx].global_id`), NOT `tris.verts[3t+k]`,
 *       which is its ORIGIN.
 */
typedef struct
{
  int global_id;  ///< Stable name of this triangle. Its half-edges' ids are 3*this + k.
  int owner_proc;  ///< The NEW owner, decided by the sender from the cached centre.
  int flags;  ///< MFTL_FLAG_BOUNDARY travels with the entity; it cannot be re-derived.
  int body_gid;  ///< #MFTL_Object::global_id of the owning body. See the struct note.

  int vertex_gid[3];  ///< Global id of the TARGET vertex of half-edge k.
  int he_flags[3];  ///< Half-edge flags; MFTL_FLAG_BOUNDARY is not re-derivable.
} MFTL_PackedTriangleComplex;

/**
 * @struct MFTL_GhostUpdatePayload
 * @brief One per-vertex 3-vector plus the vertex global id, 32 bytes -- the
 *        reply of a ghost synchronisation.
 *
 * Generic on purpose: positions and normals travel in the same format, one
 * vector per exchange. Bundling recomputable fields (normal, area, ...) with the
 * position would send bytes the receiver can compute more cheaply.
 */
typedef struct
{
  double value[3];  ///< The synchronised vector: a position, a normal, ...
  int global_id;  ///< MFTL_ERROR marks a request the owner could not answer.
} MFTL_GhostUpdatePayload;

/* The layouts are part of the contract with MFTLRegisterMpiTypes: the datatypes
 * describe contiguous runs, which is only valid while these hold. */
_Static_assert(sizeof(MFTL_PackedVertex) == 40, "MFTL_PackedVertex layout changed");
_Static_assert(sizeof(MFTL_PackedTriangleComplex) == 40, "MFTL_PackedTriangleComplex layout changed");
_Static_assert(sizeof(MFTL_GhostUpdatePayload) == 32, "MFTL_GhostUpdatePayload layout changed");

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. MPI datatypes
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief MPI datatype for #MFTL_PackedVertex. Valid between MFTLRegisterMpiTypes() and MFTLFreeMpiTypes(). */
extern MPI_Datatype MFTL_MPI_PackedVertex;
/** @brief MPI datatype for #MFTL_PackedTriangleComplex. */
extern MPI_Datatype MFTL_MPI_PackedTriangleComplex;
/** @brief MPI datatype for #MFTL_GhostUpdatePayload. */
extern MPI_Datatype MFTL_MPI_GhostUpdatePayload;

/**
 * @brief Creates and commits the MFTL MPI datatypes (extents pinned to the C
 *        `sizeof`). Call once after `MPI_Init`, before any exchange; idempotent.
 */
void MFTLRegisterMpiTypes(void);

/** @brief Frees the MFTL MPI datatypes. Call before MPI_Finalize. Idempotent. */
void MFTLFreeMpiTypes(void);

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Generic variable-length exchange
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One destination's outgoing elements, as a typed byte blob. */
typedef struct
{
  void *buffer;  ///< malloc'd by the producer, freed by the producer.
  int count;  ///< Number of elements currently in @c buffer.
  int capacity;  ///< Number of elements @c buffer can hold.
} MFTL_SendBuffer;

/** @brief Upper bound on payloads a single MFTLExchangeStart may carry. */
#define MFTL_MAX_EXCHANGE_PAYLOADS 4

/**
 * @struct MFTL_Exchange
 * @brief One variable-length payload: what to send, and where the result lands.
 */
typedef struct
{
  MFTL_SendBuffer *send_bufs;  ///< [nproc] indexed by DESTINATION RANK, even on the neighbour path.
  MPI_Datatype type;  ///< Committed MPI datatype of one element.
  size_t type_size;  ///< sizeof of one element, must match @c type.

  void *recv_buf;  ///< out: allocated here, ownership passes to the caller.
  int recv_count;  ///< out: number of elements received.
} MFTL_Exchange;

/**
 * @struct MFTL_ExchangeCtx
 * @brief In-flight state of an MFTLExchangeStart. Treat the fields as private.
 */
typedef struct
{
  MPI_Request requests[MFTL_MAX_EXCHANGE_PAYLOADS];  ///< One in-flight *alltoallv per payload.
  char *flat_send[MFTL_MAX_EXCHANGE_PAYLOADS];  ///< Internal copies of the send buffers, freed by MFTLExchangeFree().
  int *counts;  ///< 4 * n_payloads * n_targets ints: send/recv counts and displacements.
  int n_payloads;  ///< Payloads this context is carrying.
  int n_targets;  ///< Destinations: `nproc` on the global path, `num_neighbors` on the graph path.
} MFTL_ExchangeCtx;

/**
 * @brief Starts a non-blocking all-to-all-v of up to four independent payloads.
 *
 * One blocking count exchange carrying all payloads' counts
 * (`MPI_Alltoall` / `MPI_Neighbor_alltoall`), then one non-blocking
 * `*alltoallv` per payload, posted back to back. The send buffers are copied
 * into internal storage before return, so the caller may reuse them at once.
 *
 * @param domain          Supplies the communicators, rank count and neighbour list.
 * @param x               [n_payloads] payload descriptors; `recv_buf` /
 *                        `recv_count` are filled in (the caller frees `recv_buf`).
 * @param n_payloads      1..#MFTL_MAX_EXCHANGE_PAYLOADS, identical on all ranks.
 * @param use_global_path `true`: `MPI_Alltoallv` over `comm_cart` (any
 *                        destination); `false`: `MPI_Neighbor_alltoallv` over
 *                        `comm_graph` (26-neighbourhood only). Identical on all
 *                        ranks -- agree on it with a reduction first.
 * @param ctx             Out: pass to MFTLExchangeWait() / MFTLExchangeFree().
 *
 * @collective The payload count and order come from the code path, never the
 *             data, as non-blocking collectives require.
 */
void MFTLExchangeStart(const MFTL_Domain *domain, MFTL_Exchange *x, int n_payloads, bool use_global_path, MFTL_ExchangeCtx *ctx);

/**
 * @brief Blocks until payload @p payload has arrived; its `recv_buf` is then
 *        valid. Safe to call twice.
 *
 * @param ctx     Context filled in by MFTLExchangeStart().
 * @param payload Payload index in `[0, n_payloads)`.
 */
void MFTLExchangeWait(MFTL_ExchangeCtx *ctx, int payload);

/**
 * @brief Waits for anything still outstanding and releases the internal buffers.
 *
 * Does NOT free the payloads' recv_buf -- those belong to the caller.
 *
 * @param ctx Context filled in by MFTLExchangeStart(). Safe to call twice.
 */
void MFTLExchangeFree(MFTL_ExchangeCtx *ctx);

/**
 * @brief Variable-length allgather of doubles over `comm_graph`: this rank's
 *        block, delivered to every graph neighbour.
 *
 * The shape of every sparse per-body publication (holder-route records, halo
 * radii, balancer verdicts): a count allgather plus an allgatherv, payload
 * `O(held data x degree)`. If `comm_graph` has a self-loop this rank's own block
 * comes back once; callers folding their own contribution locally skip it.
 *
 * @param      domain The MPI/physical world.
 * @param      send   `[n_send]` this rank's block.
 * @param      n_send Doubles in @p send. May be zero.
 * @param[out] rcount `[num_neighbors]` doubles received from each neighbour.
 * @param[out] rdispl `[num_neighbors]` offset of each neighbour's block.
 * @param[out] n_recv Total doubles received.
 * @return A newly allocated buffer of @p n_recv doubles in neighbour order
 *         (never #MFTL_NULL); the caller frees it.
 *
 * @collective Over `comm_graph`; a serial run makes no MPI call.
 */
double *MFTLNeighborAllgatherDoubles(const MFTL_Domain *domain, const double *send, int n_send, int *rcount, int *rdispl, int *n_recv);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Redistribution
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * A redistribution also updates the OBJECT LIST: a rank holds an #MFTL_Object
 * exactly for the bodies it holds entities of, so bodies that left are
 * destroyed and bodies that arrived are created. Hence an `MFTL_Object*`, a list
 * slot and `n_local_objects` do NOT survive a redistribution; the body id does:
 *
 *     MFTLRedistribute(ens);
 *     obj = MFTLEnsembleFindLocalObject(ens, body_gid);   // may be MFTL_NULL now
 *
 * A body without entities on any rank ends up held by nobody; its id stays
 * allocated.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Re-partitions every local object with the geometric owner rule, in one
 *        communication round for the whole ensemble.
 *
 * 1. **Analysis.** Every entity's owner is recomputed (MFTLGetOwnerProc()); each
 *    OWNED triangle is destined for its owner plus the owners of its corners and
 *    edge neighbours. Entities this rank keeps are marked, not packed. Ghost
 *    triangles are never forwarded (their owner sends the authoritative copy).
 *    Bodies nobody saw change are left alone (MFTLRedistributeSetSkipUnchangedBodies()).
 * 2. **Route.** One `MPI_Allreduce` (LOR) decides whether any destination lies
 *    outside the 26-neighbourhood; if so the exchange uses `comm_cart`.
 * 3. **Exchange.** Vertices, triangles and one 7-int header per (body,
 *    destination) as three payloads of one MFTLExchangeStart(); all bodies share
 *    the messages, so the message count is independent of the body count.
 * 4. **Rebuild.** Kept entities are compacted in place into the layout
 *    `[kept owned | new owned | kept ghost | new ghost]`; arrivals are
 *    de-duplicated by global id; only half-edges that lost their twin are
 *    re-paired; `out_he_idx` and the derived arrays are emitted inline. Bodies
 *    that arrived are created, bodies left empty destroyed.
 *
 * @param ens The ensemble; every local object is repartitioned in place. Its
 *            triangle centres must be current, ghosts included
 *            (MFTLEnsembleComputeTriangleGeometry(); asserted in Debug).
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble.
 *
 * @collective Ranks holding no bodies must call it too; an ensemble without
 *             bodies is a no-op.
 * @warning Invalidates every pool index, every #MFTL_Object pointer and all
 *          cached geometry except vertex positions (see the file comment).
 */
int MFTLRedistribute(MFTL_Ensemble *ens);

/**
 * @struct MFTL_OwnerMap
 * @brief Caller-supplied destination rank for every entity of one body.
 *
 * Replaces the geometric rule for one redistribution. The distributed remesher
 * uses it twice: the SPANNING strategy shifts the cut planes, the DISPERSED
 * strategy sends each body to the owner of its centroid. Everything downstream
 * (halo surplus, de-duplication, twin repair) is indifferent to where the
 * destinations came from, and a NULL map keeps the production path unchanged
 * (one hoisted branch per body).
 *
 * **Consistency across ranks.** Each rank decides ownership independently, so
 * the maps must name the SAME destination for every entity two ranks both hold
 * -- a rule evaluated on identical coordinates does; a rule reading anything
 * rank-local (a flag, a pool index) does not. Ghost entries matter as much as
 * owned ones. Entries are ranks in `[0, nproc)`; destinations outside the
 * 26-neighbourhood force the global route.
 */
typedef struct MFTL_OwnerMap
{
  const int *tri_owner;  ///< [obj->tris.n]  destination rank per triangle, pool order. GHOSTS INCLUDED.
  const int *vert_owner;  ///< [obj->verts.n] destination rank per vertex, pool order. GHOSTS INCLUDED.
} MFTL_OwnerMap;

/**
 * @brief MFTLRedistribute() with the ownership rule supplied per body.
 *
 * @p maps is indexed by LOCAL SLOT, `maps[o]` describing `ens->local_objects[o]`:
 * the argument never crosses a rank, so the slot needs no global meaning (and a
 * body-id-indexed array would cost `O(bodies in the ensemble)` per call).
 *
 * @param ens  The ensemble.
 * @param maps `[ens->n_local_objects]` owner maps in `local_objects` order, or
 *             #MFTL_NULL for the geometric rule for every body. When non-NULL,
 *             both arrays of every entry must be set (asserted): there is no
 *             per-body fallback to the geometric rule.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid ensemble.
 *
 * @collective The maps must agree on every shared entity (#MFTL_OwnerMap).
 */
int MFTLRedistributeWithOwners(MFTL_Ensemble *ens, const MFTL_OwnerMap *maps);

/**
 * @brief Allows redistribution to skip bodies nobody saw change (default ON).
 *
 * A rank's state after a redistribution is a function of (global topology,
 * owner map) only -- positions decide the owner map but, once it is fixed,
 * moving a vertex within its cell changes nothing about who holds what. A body
 * is skipped when no rank saw any of its entities change owner and nothing
 * stamped it partition-dirty (MFTLObjectMarkPartitionDirty()): all
 * classification, packing and rebuild work for it is omitted. The verdict is
 * agreed per body with one MFTLEnsembleReduceBodies() call.
 *
 * The skip is EXACT, slot for slot: because a rebuild is not idempotent (an
 * owned vertex whose incident triangles are all remote is dropped and
 * re-received at the end of the run), a body is rebuilt once more after it
 * stops changing before it may be skipped (`partition_settled`).
 * `parallel_advection_redistribute_list --digest` and
 * `test_skipping_reproduces_the_full_rebuild` compare set and slot-order
 * digests. Anything that rewrites global ids or flags, or creates/destroys
 * entities outside MFTLObjectSetCounts(), must stamp the object dirty.
 *
 * @param on `true` to allow the skip, `false` to rebuild every body every call.
 *
 * @collective Every rank must pass the SAME value: it changes the number of
 *             collectives (checked by a Debug checksum).
 */
void MFTLRedistributeSetSkipUnchangedBodies(bool on);

/**
 * @brief Reports the current setting of MFTLRedistributeSetSkipUnchangedBodies().
 * @return `true` if unchanged bodies may be skipped.
 * @rank_local
 */
bool MFTLRedistributeGetSkipUnchangedBodies(void);

/**
 * @brief Lets redistribution return outgrown pool capacity (default 0 = off).
 *
 * After each rebuild every touched body is offered to
 * MFTLObjectShrinkToFit(obj, slack). Useful after generation, where rank 0
 * built every body whole (\f$3440\cdot4^r+128\f$ bytes per icosphere) and keeps
 * a sliver of each in a full-size block. A pure memory policy: counts,
 * contents, derived bits and partition stamps are unchanged, and ranks may
 * differ. It does not lower the generation PEAK. Measured: about 8 % less
 * capacity for about 2 % of the redistribution time, hence off by default.
 *
 * @param slack Shrink when `capacity > slack * n`; use #MFTL_SHRINK_SLACK_DEFAULT.
 *              Zero or negative disables.
 *
 * @rank_local
 */
void MFTLRedistributeSetShrinkSlack(double slack);

/**
 * @brief Reports the current setting of MFTLRedistributeSetShrinkSlack().
 * @return The slack, or 0.0 when shrinking is off.
 * @rank_local
 */
double MFTLRedistributeGetShrinkSlack(void);

/**
 * @brief Releases the persistent communication workspace of an ensemble.
 *
 * Called by MFTLEnsembleDestroy(). It holds only working storage, so releasing
 * it early is safe; the next redistribution rebuilds it.
 *
 * @param p_ws Address of `MFTL_Ensemble::comm_workspace`; set to #MFTL_NULL.
 *
 * @rank_local
 */
void MFTLCommWorkspaceDestroy(void **p_ws);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Ghost synchronisation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Refreshes the POSITION of every ghost vertex from its owning rank.
 *
 * Request/reply over `comm_graph`: each rank asks the owners for the
 * (body id, vertex id) pairs it holds as ghosts (8 B each) and the owners answer
 * with an #MFTL_GhostUpdatePayload (32 B); the reply order equals the request
 * order, so replies are applied by direct index. Three neighbourhood
 * collectives (counts, requests, replies). Topology is untouched: call it after
 * moving vertices, and redistribute only when ownership may have changed.
 *
 * A ghost whose owner is not a graph neighbour cannot be served, keeps a STALE
 * value and is reported (asserted in Debug); this happens only on a partition
 * whose halo reaches beyond the 26-neighbourhood.
 *
 * @param ens The ensemble.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if some ghost of THIS rank could not be
 *         served (a local verdict).
 *
 * @collective
 */
int MFTLUpdateGhostPositions(MFTL_Ensemble *ens);

/**
 * @brief Refreshes `verts.normal` of every ghost vertex from its owning rank.
 *
 * The same exchange as MFTLUpdateGhostPositions(). Owned vertices get exact
 * normals locally (vertex-complete halo); ghosts hold partial sums until this
 * call. MFTLEnsembleUpdateVertexNormals() is the one-call form.
 *
 * @note Run MFTLComputeVertexNormals() on every object first; this copies, it
 *       does not compute.
 * @param ens The ensemble.
 * @return As for MFTLUpdateGhostPositions().
 *
 * @collective
 */
int MFTLUpdateGhostVertexNormals(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Spatial halo
 *
 * The decomposition is geometric, so two bodies in contact are cut by the same
 * rank boundaries, and the rank holding one side of a contact may hold nothing
 * of the other body. No topological halo growth can help (the bodies share no
 * edges), so the triangles of body B within a margin of this rank's share of
 * body A are shipped here explicitly.
 *
 * They are delivered as a SHADOW OBJECT per body -- a self-consistent
 * #MFTL_Object with `n_local == 0`, twins linked within the patch, derived
 * arrays built -- and never appended to the body's own pools. Appending would
 * silently change every pass bounded by `tris.n` (the Morton box,
 * MFTLMeanEdgeLength(), the vertex-normal scatter, MFTLRebuildTwins(), which
 * would weld bodies sharing vertex slots), and the arriving vertex ids would
 * break either vertex completeness or local id uniqueness. The patches are not
 * in `ens->local_objects`, so redistribution and ghost syncs never see them.
 *
 * The halo is TRANSIENT: redistribution keeps only owned triangles, so it cannot
 * survive a repartition. Each extension REPLACES earlier patches. Order:
 * redistribute -> MFTLSpatialHaloResolveRadii() -> MFTLExtendSpatialHaloNearBodies()
 * -> contact detection.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_SpatialHaloPair
 * @brief One candidate body pair kept by the broad phase.
 *
 * `body_a < body_b`: the smaller id is the A side, whose OWNED triangles report
 * contacts; only the B side travels. That asymmetry is the "reported once"
 * rule. A SELF pair (`body_a == body_b`) appears only when
 * #MFTL_SpatialHalo::self_pairs is set; the traversal then restores "reported
 * once" by triangle id. A pair is present on this rank iff it holds body A and
 * can see body B (locally, as a patch, or both).
 */
typedef struct MFTL_SpatialHaloPair
{
  int body_a;  ///< Smaller body id. Held by this rank; its owned triangles report.
  int body_b;  ///< Larger body id -- or the SAME id for a self pair. Held here, patched here, or both.
} MFTL_SpatialHaloPair;

/**
 * @struct MFTL_SpatialHalo
 * @brief A sparse, body-id-keyed directory of shadow objects and body radii.
 *
 * `body_gid` is strictly ascending; entry `e` describes body `body_gid[e]`. Use
 * MFTLSpatialHaloPatch() / MFTLSpatialHaloRadius(), which binary-search. The
 * directory holds the bodies this rank HOLDS plus the bodies it RECEIVED A PATCH
 * FOR (the latter are exactly those it holds nothing of); other entries are
 * pruned after every exchange, so its size tracks what is nearby, not the
 * ensemble.
 *
 * `patch[e]` is #MFTL_NULL until something arrives; otherwise a valid object
 * with `n_local == 0`. Patch global ids are the body's real ids (so reported
 * contacts are meaningful to the owner) but are neither dense nor unique against
 * the body's own pool.
 */
typedef struct MFTL_SpatialHalo
{
  int *body_gid;  ///< [num_objects] body ids, STRICTLY ASCENDING. The key.
  MFTL_Object **patch;  ///< [num_objects] shadow object per entry, or MFTL_NULL.
  double *global_radius;  ///< [num_objects] half the LONGEST SIDE of the body's global box.
  int num_objects;  ///< Entries in the directory. NOT the size of the ensemble.
  int capacity;  ///< Entries allocated.

  /**
   * @brief Candidate pairs, ascending and unique. Filled by
   *        MFTLExtendSpatialHaloNearBodies(); the explicit request form leaves
   *        it empty.
   */
  MFTL_SpatialHaloPair *pair;
  int n_pairs;  ///< Entries in `pair`.
  int cap_pairs;  ///< Entries allocated in `pair`.

  /**
   * @brief INPUT to MFTLExtendSpatialHaloNearBodies(): also derive SELF pairs
   *        (default false).
   *
   * When set, a body also travels towards a neighbour's share of the SAME body,
   * and `pair` gains `(B, B)` for every body this rank owns something of -- what
   * a self-proximity query (two sheets of one body across a rank boundary, a
   * neck on a cut plane) needs, in the same exchange.
   */
  bool self_pairs;

  MFTL_Domain *domain;  ///< Borrowed from the ensemble.
} MFTL_SpatialHalo;

/**
 * @struct MFTL_SpatialHaloRequest
 * @brief "Send me the parts of body B within `margin` of my share of body A."
 *
 * Ordered: only B travels, because the rank owning the A-side triangle reports
 * the contact.
 */
typedef struct MFTL_SpatialHaloRequest
{
  int body_a;  ///< Body id (#MFTL_Object::global_id). Only its EXTENT is used.
  int body_b;  ///< Body id of the body whose triangles travel.
  double margin;  ///< Absolute distance, in domain length units.
} MFTL_SpatialHaloRequest;

/**
 * @brief Allocates an empty halo whose directory lists the bodies @p ens holds.
 *
 * Every later collective call re-synchronises the directory with the ensemble,
 * so a halo may be created once and reused across redistributions.
 *
 * @param ens The ensemble (only its local list is read).
 * @return The halo; free with MFTLSpatialHaloDestroy().
 */
MFTL_SpatialHalo *MFTLSpatialHaloCreate(MFTL_Ensemble *ens);

/**
 * @brief Directory slot of body @p body_gid, by binary search.
 *
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return The slot in `body_gid` / `patch` / `global_radius`, or #MFTL_ERROR if
 *         this halo knows nothing about that body.
 */
static inline int MFTLSpatialHaloFindBody(const MFTL_SpatialHalo *halo, int body_gid)
{
  return (halo == MFTL_NULL) ? MFTL_ERROR : MFTLSortedIndexOf(halo->body_gid, halo->num_objects, body_gid);
}

/**
 * @brief Destroys a halo and every patch object it owns. Safe on MFTL_NULL.
 *
 * @param p_halo Address of the halo pointer. Set to #MFTL_NULL on return.
 */
void MFTLSpatialHaloDestroy(MFTL_SpatialHalo **p_halo);

/**
 * @brief Resolves `global_radius` -- half the LONGEST SIDE of the body's
 *        bounding box over all its holders (\f$= R\f$ for a sphere) -- for every
 *        body this rank holds, and for the bodies its graph neighbours hold.
 *
 * The GLOBAL box, so that envelopes scaled by it do not depend on the process
 * count. Each holder contributes the box of its OWNED triangles in two frames,
 * raw and shifted by half a domain, and the tighter extent per axis wins: a body
 * straddling a periodic seam is contiguous in the shifted frame and would
 * otherwise measure as domain-wide. Twelve MIN slots per body go through
 * MFTLEnsembleReduceBodies(). The result is then published one hop along
 * `comm_graph`, because a rank deciding what to SEND needs the size of the body
 * it sends TOWARDS, which it may hold nothing of.
 *
 * @param ens  The ensemble.
 * @param halo The halo whose radii are filled in.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 *
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLSpatialHaloResolveRadii(MFTL_Ensemble *ens, MFTL_SpatialHalo *halo);

/**
 * @brief Exchanges the explicitly requested patches and rebuilds the affected
 *        shadow objects.
 *
 * 1. Each rank publishes one record per body it HOLDS -- `(body id, centre,
 *    half-width, global radius, envelope)`, nine doubles -- to its graph
 *    neighbours (count allgather + non-blocking allgatherv, overlapped with
 *    computing per-triangle extents). Payload `O(local bodies x degree)`.
 * 2. For each request whose B body it holds, a rank sends every owned triangle
 *    of B (and its vertices) whose extent, widened by the margin, is near a
 *    neighbour's published extent of A. Per destination each triangle is packed
 *    once, however many partners want it.
 * 3. One MFTLExchangeStart() with two payloads; one patch per arriving body is
 *    rebuilt with MFTLBuildTopologyFromIndexList(), and the transmitted boundary
 *    flags overwrite the derived ones (a patch's cut rim is not a mesh boundary).
 *
 * **Extents are a centre plus a half-width**, compared by minimum-image
 * distance between centres; a (lo, hi) box would miss contacts across a
 * periodic seam and, for any share containing one seam-straddling triangle,
 * span the whole domain and ship everything (measured at np = 27: 314 triangles
 * with boxes against 56). A rank's own extent is measured in the raw and the
 * half-domain-shifted frame and the tighter kept. The per-triangle test is
 * widened by the margin and so is the body-level pre-filter, which keeps the
 * pre-filter strictly more permissive than the test it short-cuts.
 *
 * @param ens        The ensemble.
 * @param requests   `[n_requests]` ordered pairs with margins (absolute lengths).
 *                   A request whose B body this rank does not hold is ignored.
 *                   The lists need NOT agree across ranks.
 * @param n_requests Number of requests; 0 still enters every collective.
 * @param halo       Filled in. Patches of the requested B bodies are replaced;
 *                   `pair` is left untouched.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments.
 *
 * @collective Over `comm_graph`; the collective sequence is fixed by the
 *             neighbour count, not by the requests.
 */
int MFTLExtendSpatialHalo(MFTL_Ensemble *ens, const MFTL_SpatialHaloRequest *requests, int n_requests, MFTL_SpatialHalo *halo);

/**
 * @brief The same exchange with the candidate pairs DERIVED from the published
 *        extents instead of enumerated -- the form contact detection uses.
 *
 * An all-pairs request list is \f$O(B^2)\f$ per rank (134 million records at
 * 16 384 bodies). Instead the arriving records are binned into a uniform grid
 * whose cell is at least `2 x widest half-width + largest envelope`, so
 * overlapping extents lie in adjacent cells and each query visits 27 cells:
 * \f$O(\text{local bodies} + \text{records})\f$. Body B travels towards body A
 * only when A < B (plus A == B when `self_pairs`), with envelope
 * `MAX(tol_A, tol_B)`.
 *
 * On return `halo->pair` holds, ascending and unique, every `(A, B)` such that
 * this rank holds A and can see B and their extents are within the envelope.
 *
 * @param ens       The ensemble.
 * @param tolerance `[ens->n_local_objects]` contact envelope per LOCAL slot, or
 *                  #MFTL_NULL for zero (overlap only).
 * @param halo      Filled in; EVERY patch is replaced and `pair` rebuilt.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble or halo.
 *
 * @collective Over `comm_graph`.
 */
int MFTLExtendSpatialHaloNearBodies(MFTL_Ensemble *ens, const double *tolerance, MFTL_SpatialHalo *halo);

/**
 * @brief The received patch of body @p body_gid, or #MFTL_NULL.
 *
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return The shadow object (owned by the halo, invalidated by the next
 *         extension or MFTLSpatialHaloDestroy()), or #MFTL_NULL.
 */
static inline const MFTL_Object *MFTLSpatialHaloPatch(const MFTL_SpatialHalo *halo, int body_gid)
{
  const int e = MFTLSpatialHaloFindBody(halo, body_gid);
  return (e == MFTL_ERROR) ? MFTL_NULL : halo->patch[e];
}

/**
 * @brief Half the longest side of body @p body_gid's global bounding box.
 *
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return The radius MFTLSpatialHaloResolveRadii() (or a received record)
 *         provided, or 0.0 if this halo knows nothing about the body.
 */
static inline double MFTLSpatialHaloRadius(const MFTL_SpatialHalo *halo, int body_gid)
{
  const int e = MFTLSpatialHaloFindBody(halo, body_gid);
  return (e == MFTL_ERROR) ? 0.0 : halo->global_radius[e];
}

/**
 * @brief Number of triangles this rank received for body @p body_gid.
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return Triangle count of the patch, or 0 if nothing was received.
 */
static inline int MFTLSpatialHaloCount(const MFTL_SpatialHalo *halo, int body_gid)
{
  const MFTL_Object *const p = MFTLSpatialHaloPatch(halo, body_gid);
  return (p == MFTL_NULL) ? 0 : p->tris.n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Instrumentation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_RedistributeStats
 * @brief Rank-local counters and phase timers of MFTLRedistribute() and the
 *        ghost syncs. Purely diagnostic.
 *
 * Everything accumulates until MFTLRedistributeStatsReset(); nothing takes a
 * collective. The counters are always on. The `t_*` timers are filled only in a
 * build configured with `-DMFTL_COMM_PROFILE=ON` (otherwise they stay 0); they
 * are in seconds and partition the redistribution, so `t_total` minus the other
 * buckets is residue. `t_wait`, `t_route` and `t_skip_vote` absorb load
 * imbalance: a rank arriving early at a collective charges its neighbours'
 * slowness to itself. `n_clock_reads` counts the `MPI_Wtime()` calls the
 * instrumentation made (tens of thousands per step at a thousand bodies), so
 * its own perturbation can be estimated.
 */
typedef struct
{
  // --- Redistribution: the disjoint phase buckets, in call order -------------

  double t_total;  ///< The whole of MFTLRedistribute(), including the serial short-circuit.

  /**
   * Per-call setup: the `nproc`-sized resets of the persistent workspace
   * (neighbour map, send-buffer fill levels, id-list hints), its growth, and
   * the buffer-hint arithmetic. Proportional to the PROCESS COUNT, so the bucket
   * to watch in a weak-scaling sweep.
   */
  double t_setup;

  double t_owners;  ///< Ownership classification. O(N+H) and invariant to how much moves.
  double t_masks;  ///< Clearing the per-object keep flags and de-duplication mask. O(V+T) per body.
  double t_destinations;  ///< Destination analysis: seven candidates per owned triangle.
  /**
   * The neighbour-vs-global route `MPI_Allreduce` (one int): pure latency, but
   * after the vote it may still absorb load imbalance.
   */
  double t_route;

  /**
   * The per-body skip vote (one MFTLEnsembleReduceBodies()). The first
   * collective after the object loop, so it absorbs the load imbalance between
   * ranks; read it together with `t_route`.
   */
  double t_skip_vote;

  double t_pack;  ///< Building the per-destination payloads.
  double t_exchange_counts;  ///< The merged count exchange, (Neighbor_)Alltoall.
  double t_exchange_data;  ///< Posting the payload Alltoallv, plus the flattening copy.
  double t_wait;  ///< Waiting for the payloads. Absorbs load imbalance; see above.
  double t_group;  ///< De-interleaving the arrival stream by body.
  double t_reserve;  ///< Pool resize: growth inside MFTLObjectSetCounts(), plus the phase 5b shrink.
  double t_kept_gather;  ///< Gathering kept entities out of the pool into scratch.
  double t_kept_place;  ///< Writing them back at their new offsets.
  /**
   * Re-linking kept half-edges: vertex remap, twin remap, `out_he_idx`, and the
   * `tris.verts` / `tris.neighbors` entries of every kept triangle (emitted
   * inline instead of by a closing MFTLObjectBuildDerived() pass; about a third
   * of that pass's cost reappears here, docs/analysis.md §2.5).
   */
  double t_relink;
  double t_arrivals;  ///< Classifying and writing what arrived, derived entries included.
  double t_twins;  ///< repair_twins() over the half-edges that lost a partner.

  /**
   * Freeing what the exchange layer allocated for this call (the workspace
   * itself persists).
   */
  double t_teardown;

  // --- Redistribution: counters ---------------------------------------------

  long n_calls;  ///< Redistributions performed, serial short-circuit included.
  long n_verts_scanned;  ///< Vertices classified, owned and ghost, summed over calls.
  long n_tris_scanned;  ///< Triangles classified, owned and ghost.
  long n_verts_owner_changed;  ///< Of those, how many changed hands.
  long n_tris_owner_changed;  ///< Of those, how many changed hands.
  long n_verts_kept;  ///< Kept in the pool rather than re-received.
  long n_tris_kept;  ///< Kept in the pool rather than re-received.
  long n_verts_received;  ///< Arrivals, before de-duplication.
  long n_tris_received;  ///< Arrivals, before de-duplication.
  long n_destinations;  ///< Distinct remote destinations touched, summed over calls.
  long n_messages;  ///< Non-empty per-destination payloads sent.
  long n_bytes_sent;  ///< Payload bytes out, wire format only.
  long n_bytes_recv;  ///< Payload bytes in.
  long n_route_neighbor;  ///< Calls that took the 26-neighbour graph route.
  long n_route_global;  ///< Calls that fell back to the global communicator.
  long n_objects_visited;  ///< Bodies iterated, summed over calls.

  /**
   * Bodies that reached the rebuild with nothing kept and nothing arriving:
   * all their entities left this rank; their pools are released and the object
   * is destroyed at the end of the call.
   */
  long n_objects_empty_local;

  /**
   * Bodies skipped because no rank saw them change.
   * `n_objects_skipped / n_objects_visited` is the body fraction the skip turns
   * on (not an entity fraction). See MFTLRedistributeSetSkipUnchangedBodies().
   */
  long n_objects_skipped;

  /**
   * Bodies classified a second time in pass 2 (another rank saw them change, or
   * they needed their settling rebuild).
   */
  long n_objects_reclassified;

  /**
   * Of those, the ones rebuilt only to reach the layout fixed point (the price
   * of an exact, slot-for-slot skip: one rebuild per dirty-to-clean transition).
   */
  long n_objects_settled;

  /**
   * Bodies whose pools were shrunk (only with MFTLRedistributeSetShrinkSlack()
   * armed). A non-zero count every step means the gate fires on jitter and the
   * slack is too low.
   */
  long n_objects_shrunk;
  long n_bytes_reclaimed;  ///< Capacity bytes released by the shrink. See `n_objects_shrunk`.

  /**
   * Bodies created here on arrival (and, below, destroyed on becoming empty) --
   * the sparse object list's churn, and the non-vacuity measure of any test of
   * it.
   */
  long n_objects_created;
  long n_objects_destroyed;  ///< See n_objects_created.

  long n_allocs;  ///< Allocation calls on the path.
  long n_frees;  ///< Matching frees.
  long n_bytes_allocated;  ///< Bytes requested, summed. Not a peak.
  long n_gid_table_bytes;  ///< Bytes the arrival id tables asked for.

  /**
   * @name Persistent workspace footprint, split by what it scales with
   *
   * Current capacities (not churn): per body HELD, per rank of the job (the
   * `[nproc]` arrays the global fallback needs), and per entity held. In a
   * weak-scaling sweep the per-body and per-entity figures should stay flat; the
   * per-rank one grows with P.
   * @{
   */
  long ws_bytes_per_body;
  long ws_bytes_per_rank;
  long ws_bytes_per_entity;
  /** @} */

  /**
   * Uses of the arrival id tables in direct-address mode (`tag[gid]`, a table as
   * large as the body's id range) and in sparse, open-addressed mode (a table
   * proportional to the entries held). The mode is chosen per body per call by
   * the density of the ids.
   */
  long n_gid_table_direct;
  long n_gid_table_sparse;  ///< See n_gid_table_direct.

  /**
   * Destinations beyond the 32 de-duplication bit slots of one body, whose
   * vertex lists then fall back to a sort. Zero on the neighbour route (at most
   * 26 remote destinations).
   */
  long n_dedup_overflow;

  long n_clock_reads;  ///< MPI_Wtime() calls the instrumentation made. See above.

  // --- Ghost synchronisation -------------------------------------------------

  double g_t_total;  ///< The whole of MFTLUpdateGhostPositions()/VertexNormals().
  double g_t_collect;  ///< Scanning the ghost suffix and building the request lists.
  double g_t_counts;  ///< The count exchange.
  double g_t_requests;  ///< Sending the (body, global id) requests.
  double g_t_table;  ///< Owner-side id table build: a scatter over every owned vertex.
  double g_t_answer;  ///< Resolving requests to positions.
  double g_t_replies;  ///< Sending the payloads back.
  double g_t_apply;  ///< Writing them into the ghost slots.
  double g_t_teardown;  ///< Freeing the per-call request lists, buffers and id table.

  long g_n_calls;  ///< Syncs performed.
  long g_n_ghosts_requested;  ///< Ghost vertices asked for, summed over calls.
  long g_n_orphaned;  ///< Ghosts whose owner was not a graph neighbour, and were SKIPPED.
  long g_n_bytes_sent;  ///< Request bytes out plus reply bytes out.
  long g_n_bytes_recv;  ///< Request bytes in plus reply bytes in.
  long g_n_allocs;  ///< Allocation calls on the path.
  long g_n_bytes_allocated;  ///< Bytes requested, summed.
} MFTL_RedistributeStats;

/**
 * @brief Zeroes the accumulator (e.g. after a benchmark's warm-up steps).
 *
 * @rank_local
 */
void MFTLRedistributeStatsReset(void);

/**
 * @brief Copies the accumulator out.
 *
 * @param[out] out Receives this rank's counters. Must not be #MFTL_NULL.
 * @rank_local
 */
void MFTLRedistributeGetStats(MFTL_RedistributeStats *out);

#endif  // MFTL_COMMUNICATION_H
