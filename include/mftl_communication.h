/**
 * @file mftl_communication.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Distributed-memory redistribution and ghost synchronisation.
 *
 * Two operations live here, and they answer two different questions:
 *
 *   MFTLRedistribute*        "the mesh has moved -- who owns what now?"
 *                            Re-partitions the local mesh: every triangle travels
 *                            to the rank that owns its centre, plus to every rank
 *                            that owns a vertex or an edge neighbour of it (that
 *                            surplus IS the ghost layer).
 *
 *   MFTLUpdateGhostPositions "the mesh has deformed -- refresh my ghosts."
 *                            Topology untouched, only vertex POSITIONS are pulled
 *                            from the owning ranks.
 *
 * ## WHAT THE WIRE CARRIES, AND WHAT IT DELIBERATELY DOES NOT
 *
 * ONLY TOPOLOGY AND IDENTITY TRAVEL. Everything geometric except the vertex
 * position is left behind and recomputed at the receiver, because recomputing it
 * is a streaming pass over contiguous arrays (docs/analysis.md §2.1) while
 * shipping it is bytes on a wire, per entity, per timestep:
 *
 *                            geometry on the wire   here
 *   MFTL_PackedVertex                 96 B            40 B    (pos + 4 ints)
 *   MFTL_PackedTriangleComplex       120 B            40 B    (10 ints)
 *
 * A vertex POSITION is the one geometric quantity that is authoritative
 * (mftl_types.h, invariant 3) -- nothing can recompute it -- so it travels. The
 * vertex normal and the triangle area/normal/centre are all functions of the
 * positions and of the topology, so they do not.
 *
 * Three fields a general half-edge wire format would carry are not sent, because
 * implicit indexing already determines them:
 *
 *   - a half-edge's GLOBAL ID. Half-edge 3t+k of triangle t carries
 *     `3 * triangle_gid + k`, mesh-wide, established by
 *     MFTLBuildTopologyFromIndexList and preserved by MFTLObjectReorderMorton
 *     (which moves half-edge triples as a unit). The receiver reconstructs it.
 *     Guarded by inv_halfedge_gids_are_implicit() in tests/test_mesh_invariants.h.
 *   - a vertex's outgoing half-edge. Re-derived at the receiver in the same pass
 *     that writes the half-edges (origin(h) gets out_he_idx = h).
 *   - `next`, `triangle` and `twin` of a half-edge. The first two are arithmetic;
 *     the third is re-derived from the vertex ids that arrive anyway.
 *
 * ### Postcondition, and the loop order that follows from it
 *
 * A redistributed object comes back with valid topology, valid POSITIONS, and
 * the derived-array bit set (MFTLObjectBuildDerived has run). Its
 * `tris.area/normal/center` and `verts.normal` hold
 * UNSPECIFIED values and must be recomputed before they are read. The
 * per-timestep loop that satisfies this with exactly one geometry pass is:
 *
 *   1. move owned vertices                     MFTL_FOREACH_OWNED_VERTEX
 *   2. pull the vertices the neighbours moved  MFTLUpdateGhostPositions
 *   3. area/normal/CENTRE for ALL triangles    MFTLComputeTriangleGeometry(obj, obj->tris.n)
 *   4. re-decide ownership                     MFTLRedistribute
 *
 * Step 2 before 3 is not optional: a triangle's three vertices can be owned by
 * three different ranks. Step 3 before 4 is not optional either, and it must
 * cover GHOSTS -- redistribution routes by the CACHED centre of ghost triangles
 * as well, which is how a rank knows where its neighbour's triangle is going and
 * therefore which of its own triangles that rank will need as a ghost. Refreshing
 * only the owned ones makes two ranks disagree about a ghost's destination and
 * someone's owned triangle silently loses an edge neighbour. A Debug assertion in
 * the redistribution catches it.
 *
 * A caller that also wants MFTLEnsembleUpdate's quality statistics or vertex
 * normals inserts those after step 4 so they operate on the new partition.
 *
 * "Exactly one geometry pass" is precise, and it is the reason the postcondition
 * above is worth stating so loudly: the loop ENDS on step 4, so between two
 * timesteps the triangle geometry is undefined. That costs nothing here, because
 * step 3 of the next iteration recomputes it from the moved positions anyway.
 * A solver that reads `tris.area` or `tris.normal` AFTER the redistribution --
 * rather than between steps 3 and 4, where the values describe the configuration
 * the forces belong to -- owes one more pass, which is the price of not packing
 * the geometry onto the wire. That pass is 13-15 % of a timestep;
 * examples/parallel_advection_redistribute_list.c runs it as step 5 and times it
 * separately, and docs/analysis.md §2.1 quotes the comparison both
 * ways because both are honest answers to different questions.
 *
 * ## ENTITIES THAT STAY NEVER LEAVE THE POOL
 *
 * Treating "stays here" as just another destination routes EVERY entity through
 * the wire format, including the ~90 % of a partition that does not move: pack
 * into an MFTL_Packed*, radix-sort the merged (kept + received) stream by global
 * id, de-duplicate, unpack, and insert four hash-map entries per triangle. That
 * round trip is the single most expensive part of a redistribution and none of it
 * moves any data anywhere.
 *
 * Here the kept entities are identified during the same destination analysis that
 * decides what to send (this rank is simply one more destination), and are then
 * COMPACTED IN THE POOL by a permutation. Only arrivals go through the wire
 * format, and only arrivals are de-duplicated. Concretely, per redistribution:
 *
 *   - no pack/unpack of kept entities, and no global sort: the merged sort is
 *     replaced by an O(kept) gather plus an O(received) classification;
 *   - no hash maps. Global ids are dense per entity class (see
 *     inv_global_ids_are_dense), so "have I already got global id g" is an
 *     epoch-stamped array probe, not a hash lookup;
 *   - NO FULL TWIN REBUILD. A twin between two kept triangles survives as an
 *     index remap; only half-edges that actually lost their partner are re-paired,
 *     and they are collected for free while the remap runs.
 *
 * ## PRECONDITIONS
 *
 *   - MFTLRegisterMpiTypes() has been called on every rank.
 *   - `tris.center` is current for ALL triangles, ghosts included (see above).
 *   - Every rank added the SAME bodies in the SAME order, so that a body id
 *     (#MFTL_Object::global_id, which is how bodies are named on the wire) means
 *     the same body everywhere. Which bodies a rank HOLDS differs between ranks
 *     by design. A Debug-only checksum reduction inside the redistribution
 *     reports a disagreement about the id space instead of corrupting.
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
 * @brief Serialisable image of a vertex: its position and its identity.
 *
 * 40 bytes, laid out as one run of doubles followed by one run of ints, so the
 * MPI datatype is two contiguous blocks rather than one block per field. There is
 * no padding anywhere in it -- 3 doubles and 4 ints fill 40 bytes exactly.
 *
 * ## WHY `body_gid` AND NOT A LIST POSITION
 *
 * A rank holds an #MFTL_Object only for the bodies it has entities of, so a
 * position in `MFTL_Ensemble::local_objects` means a different body on every
 * rank. #MFTL_Object::global_id is the one body name every rank agrees about,
 * and it is dense (MFTLEnsembleAddObject() hands it out).
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
 * @brief Serialisable image of a triangle AND its three half-edges.
 *
 * Ten ints, 40 bytes, one contiguous block -- the MPI datatype is a plain
 * MPI_Type_contiguous. Keeping the triangle and its half-edges in one transfer
 * unit is what makes the half-edges free of everything except their target
 * vertex: slot k already states that the half-edge belongs to this triangle and
 * that its successor is slot (k+1)%3.
 *
 * @note `vertex_gid[k]` is the global id of the vertex at the TARGET of half-edge
 *       3t+k, i.e. `verts.rec[he[3t+k].vertex_idx].global_id`. It is NOT
 *       `tris.verts[3t+k]`, which is the ORIGIN of that half-edge -- the two
 *       differ by a rotation, and mixing them silently rotates every triangle.
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
 * @brief ONE per-vertex 3-vector, plus the id that identifies whose it is.
 *
 * Deliberately generic rather than named after the position: the same 32 bytes,
 * the same datatype and the same request/reply exchange carry whichever per-vertex
 * vector is being synchronised. Two callers use it today --
 * MFTLUpdateGhostPositions and MFTLUpdateGhostVertexNormals -- and they differ
 * only in which array they read from and write into.
 *
 * ONE vector per message, never a bundle. Shipping position, normal, both
 * curvatures and the area together costs 80 B against 32 B, of which everything
 * but the position is a function of the positions and the topology, so four
 * fifths of it is recomputable at the receiver for less than it costs to send.
 * Anything that genuinely does need a second field sends a second exchange
 * and pays only for that field.
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
 * @brief Creates and commits the MFTL MPI datatypes.
 *
 * Call once per process after MPI_Init and before any redistribution. Idempotent:
 * a second call without an intervening MFTLFreeMpiTypes() does nothing.
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
 * @brief Starts a non-blocking all-to-all-v of up to 4 independent payloads.
 *
 * Two collectives regardless of the payload count: ONE MPI_Alltoall carrying
 * n_payloads counts per target, then one non-blocking *alltoallv per payload, all
 * posted back to back. Merging the count exchange matters at scale -- a 1-int
 * Alltoall is pure latency and does not get cheaper with fewer bytes, so paying
 * it once instead of once per payload is what keeps the small-message cost of a
 * redistribution flat.
 *
 * On return every send buffer has already been copied into internal storage, so
 * the caller may free @c send_bufs immediately without waiting.
 *
 * @param domain        Supplies the communicators, rank count and neighbour list.
 * @param x             [n_payloads] payload descriptors; recv_buf/recv_count are filled in here.
 * @param n_payloads    1..MFTL_MAX_EXCHANGE_PAYLOADS. Must be identical on all ranks.
 * @param use_global_path MFTL_TRUE for MPI_Alltoallv over comm_cart (arbitrary
 *                      destinations), MFTL_FALSE for MPI_Neighbor_alltoallv over
 *                      comm_graph (26-neighbourhood only). Must be identical on
 *                      all ranks -- agree on it with an MPI_Allreduce first.
 * @param ctx           out: hand to MFTLExchangeWait / MFTLExchangeFree.
 *
 * @note Collective. Non-blocking collectives must be initiated in the same order
 *       on every rank, which holds here because @p n_payloads and the payload
 *       order come from the code path, never from the data.
 */
void MFTLExchangeStart(const MFTL_Domain *domain, MFTL_Exchange *x, int n_payloads, bool use_global_path, MFTL_ExchangeCtx *ctx);

/**
 * @brief Blocks until payload @p payload has arrived; its recv_buf is then valid.
 *
 * Waiting per payload rather than for all of them is the point: the caller can
 * consume payload 0 while payload 1 is still on the wire. Safe to call twice.
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
 * The shape every sparse, per-body publication in the library has -- the holder
 * route of MFTLEnsembleReduceBodies(), the spatial halo's radius publish, the
 * dispersed balancer's verdicts -- so it is spelled once. Two collectives and not
 * one: only the sender knows how much it is sending, so the lengths cannot be
 * predicted from anything replicated. The payload is `O(what this rank holds *
 * degree)`, never sized by the body-id space or by `nproc`.
 *
 * The graph may carry a SELF-LOOP (see MFTLDomainNeighborIndex()), in which case
 * this rank's own block comes back once; a caller that also folds its own
 * contribution in locally must skip that slot.
 *
 * @param      domain The MPI/physical world.
 * @param      send   `[n_send]` this rank's block.
 * @param      n_send Doubles in @p send. May be zero.
 * @param[out] rcount `[num_neighbors]` doubles received from each neighbour.
 * @param[out] rdispl `[num_neighbors]` offset of each neighbour's block.
 * @param[out] n_recv Total doubles received.
 * @return A newly allocated buffer of @p n_recv doubles in neighbour order; the
 *         caller frees it. Never #MFTL_NULL, so an empty result frees cleanly.
 *
 * @collective Over `comm_graph`; a serial run makes no MPI call.
 */
double *MFTLNeighborAllgatherDoubles(const MFTL_Domain *domain, const double *send, int n_send, int *rcount, int *rdispl, int *n_recv);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Redistribution
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ═══════════════════════════════════════════════════════════════════════════
 * WHAT A REDISTRIBUTION DOES TO THE OBJECT LIST
 *
 * A rank holds an #MFTL_Object for a body exactly when it holds entities of it,
 * and redistribution is what makes that true: a body whose entities have all left
 * is DESTROYED here, and a body arriving on a rank that held nothing of it is
 * CREATED here. That is the whole of the per-body footprint a rank used to carry
 * for every body in the ensemble whether it held any of it or not.
 *
 * ## AN MFTL_Object POINTER DOES NOT SURVIVE A REDISTRIBUTION
 *
 * Neither does a slot in #MFTL_Ensemble::local_objects, nor the local body count.
 * The handle that does survive is #MFTL_Object::global_id, and the way back is
 * MFTLEnsembleFindLocalObject(), which returns MFTL_NULL when this rank no longer
 * holds anything of the body:
 *
 * @code
 * MFTLRedistribute(ens);
 * obj = MFTLEnsembleFindLocalObject(ens, body_gid);   // may be MFTL_NULL now
 * @endcode
 *
 * This is the same rule global ids and pool indices already follow across a
 * remeshing pass, extended to the object itself. It is stated this loudly because
 * the failure is a use-after-free on a pointer that was valid one line earlier,
 * and because a large body on few ranks can hide it for a long time: only a rank
 * that genuinely ends up holding nothing triggers it.
 *
 * A body that ends up empty on EVERY rank -- one that was added and never given
 * any entities -- therefore ends up held by nobody. Its id stays allocated in the
 * space; nothing else of it remains.
 *
 * Nothing in the library enumerates every body of the ensemble on a rank; work
 * lists are derived from what a rank holds and what its neighbours published
 * (see MFTLExtendSpatialHaloNearBodies()).
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Redistributes every local object of an ensemble in ONE communication round.
 *
 * Phases:
 *   1. ANALYSIS  Every OWNED triangle is assigned to the rank owning its centre,
 *      plus the ranks owning its three vertices and its three edge neighbours.
 *      That surplus is what produces a one-layer, VERTEX-COMPLETE ghost halo for
 *      free -- no separate ghost exchange is needed afterwards, and it is exactly
 *      the halo MFTLComputeVertexNormals needs to be communication-free. Ghost
 *      triangles are never forwarded: their owner sends the authoritative copy.
 *      A body for which nothing changed is left alone entirely (see
 *      MFTLRedistributeSetSkipUnchangedBodies()).
 *   2. DECISION  One MPI_Allreduce(LOR) agrees on whether any destination lies
 *      outside the 26-neighbourhood. If none does, the exchange runs on the graph
 *      communicator; otherwise it falls back to the global one.
 *   3. EXCHANGE  Vertices, triangles and one small header per (body,
 *      destination) go out as three payloads of a single non-blocking exchange
 *      (see MFTLExchangeStart). The streams of all bodies are interleaved, so the
 *      message COUNT is independent of the number of bodies.
 *   4. REBUILD   Kept entities are compacted in place; arrivals are de-duplicated
 *      against them through a global-id table and appended. Twins are repaired
 *      incrementally, `out_he_idx` and the derived arrays are emitted inline. A
 *      body arriving where this rank held nothing is created; a body this rank
 *      no longer holds anything of is destroyed.
 *
 * @param ens The ensemble. Every local object is repartitioned in place.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble.
 *
 * @collective Every rank must call it, including one holding no bodies. An
 *             ensemble with no bodies at all is a no-op, not an error.
 * @note Invalidates every local index, every #MFTL_Object pointer (see above) and
 *       all cached geometry except the vertex positions; see the postcondition at
 *       the top of this header.
 */
int MFTLRedistribute(MFTL_Ensemble *ens);

/**
 * @struct MFTL_OwnerMap
 * @brief An explicit answer to "who owns each entity", supplied by the caller.
 *
 * Redistribution normally derives ownership from geometry, by asking
 * MFTLGetOwnerProc which Cartesian cell each vertex position and each cached
 * triangle centre falls in. That is the right rule for a flow solver and it is
 * the only rule that costs nothing -- one multiply and a table lookup per
 * entity, inlined into a streaming loop.
 *
 * It is not the only rule anyone wants. Distributed remeshing needs the SAME
 * routing machinery under a different assignment twice over:
 *
 *   SPANNING   the cut planes are offset by half a box, so that the frozen band
 *              of one placement lands in the interior of the next. Same plane
 *              test, different planes.
 *   DISPERSED  every triangle of a body goes to the owner of the body's
 *              CENTROID, so the body becomes wholly local and the serial kernel
 *              runs on all of it with no frozen entities at all. Not a plane
 *              test at all.
 *
 * Both are one array of destinations per entity class, computed by the caller,
 * and everything downstream -- the halo surplus, the de-duplication, the twin
 * repair, the two-payload exchange -- is indifferent to where they came from.
 * So the seam is here rather than inside MFTLGetOwnerProc: passing NULL leaves
 * the production path bit-for-bit what it was, and the only cost to it is one
 * pointer test per object, outside every loop.
 *
 * ── What the caller owes ───────────────────────────────────────────────────
 *
 * The maps must be CONSISTENT ACROSS RANKS for every entity two ranks both
 * hold. Ownership is decided independently on each rank -- that is what makes
 * redistribution need no negotiation -- so if rank A routes a ghost triangle
 * somewhere rank B does not, one of them silently ends up owning a triangle
 * with a missing edge neighbour. A geometric rule evaluated on identical
 * coordinates satisfies this by construction; a rule that reads anything local
 * (a flag, a pool index, a per-rank count) does not.
 *
 * Entries must be in [0, nproc). Destinations outside the 26-neighbourhood are
 * legal but push the exchange onto the global communicator, exactly as they do
 * for the geometric rule.
 */
typedef struct MFTL_OwnerMap
{
  const int *tri_owner;  ///< [obj->tris.n]  destination rank per triangle, pool order. GHOSTS INCLUDED.
  const int *vert_owner;  ///< [obj->verts.n] destination rank per vertex, pool order. GHOSTS INCLUDED.
} MFTL_OwnerMap;

/**
 * @brief Redistribution with the ownership rule supplied rather than derived.
 *
 * Identical to MFTLRedistribute in every other respect, including the
 * postcondition. @p maps is indexed by **local slot** --
 * `[ens->n_local_objects]`, entry `o` describing `ens->local_objects[o]` -- or
 * MFTL_NULL for the geometric rule (MFTLRedistribute() is exactly that call).
 *
 * ── WHY A SLOT HERE, WHEN THE WIRE INSISTS ON A GID ────────────────────────
 *
 * It used to be `[ens->n_bodies_global]` keyed by #MFTL_Object::global_id, on
 * the rule everything that crosses a rank boundary follows: a list position is
 * only an identity while every rank holds every body. THIS ARGUMENT NEVER
 * CROSSES ONE. The caller builds it by walking `ens->local_objects` and this
 * function reads it by walking the same list in the same order on the same
 * rank, so slot `o` names the same body at both ends by construction and no
 * identity has to survive anything.
 *
 * Keyed by gid it cost every caller an array with a slot for every body in the
 * ensemble -- two pointers plus a clearing pass per call, on a rank that may
 * hold none of them. That is the last `O(bodies in the ensemble)` term the
 * remesher had, and it sat on the per-PHASE path. The correction is the one
 * MFTLEnsembleReduceBodies() makes to its own caller's buffer: keep the id
 * indexing inside whatever genuinely needs it, or the message shrinks and the
 * per-rank memory does not.
 *
 * Ghost entries matter as much as owned ones: a rank routes its ghosts in order
 * to work out which of its OWN triangles the ghost's owner is about to need, so
 * a map that only fills in `[0, n_local)` produces a halo with holes.
 *
 * @param ens  The ensemble.
 * @param maps `[ens->n_local_objects]` owner maps in `ens->local_objects` order,
 *             or #MFTL_NULL for the geometric rule. An individual entry may also
 *             be #MFTL_NULL, which means "use the geometric rule for this body".
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an invalid ensemble.
 *
 * @collective Every rank must call it with maps that agree on every shared
 *             entity; see #MFTL_OwnerMap.
 */
int MFTLRedistributeWithOwners(MFTL_Ensemble *ens, const MFTL_OwnerMap *maps);

/**
 * @brief Lets redistribution leave a body alone when nothing about it changed.
 *
 * A rank's local state after a redistribution is a pure function of the global
 * topology and the owner map, so a body for which neither input changed already
 * holds the answer: the ownership prefix, the halo, the twins and the derived
 * topology are all reproductions of what is in the pool. With this on, the
 * per-object loop tests both inputs, agrees on the verdict with the body's other
 * holders in one per-body reduction, and skips the destination analysis, the
 * kept gather, the relink, the arrival classification, the twin repair and the
 * topology rebuild for every body that did not move. Vertex motion requires no
 * derived work.
 *
 * It is BEHAVIOUR-NEUTRAL, not an approximation: the resulting pool is
 * bit-identical to the rebuilt one, slot for slot.
 * `examples/parallel_advection_redistribute_list.c --digest` is the check -- it
 * reports a SET digest and a SLOT-ORDER digest and both must match -- and it is
 * the same protocol the module's other behaviour-neutral changes were proven
 * with. `tests/test_redistribution.c` runs it as an assertion.
 *
 * Getting the order half is what costs a body ONE extra rebuild. A rebuild is
 * not idempotent: an owned vertex whose incident triangles all belong to other
 * ranks is dropped and re-received, so it ends up at the end of the owned run
 * rather than where it was, and a second rebuild moves entities the first one
 * left alone. A body is therefore rebuilt once more after it stops changing
 * before it may be skipped -- once per dirty-to-clean transition, so a body that
 * sits still for L steps loses one skip out of L, and one that alternates never
 * skips at all, which is the right answer for it. `n_objects_settled` reports
 * what that cost.
 *
 * The second half of the precondition is not free of the caller, though it is
 * enforced rather than documented: anything that rewrites global ids, writes
 * entity flags, or creates or destroys entities must stamp the object through
 * MFTLObjectMarkPartitionDirty() (mftl_object.h), which MFTLObjectSetCounts()
 * and the remesher already do. Read that function before adding a mesh
 * operation of your own.
 *
 * Default ON.
 *
 * @param on `true` to allow the skip, `false` to rebuild every body every call.
 *
 * @collective In the sense that every rank must pass the SAME value: the setting
 *             decides how many collectives a redistribution makes, so a
 *             disagreement deadlocks. A Debug-only checksum inside the
 *             redistribution reports it instead.
 */
void MFTLRedistributeSetSkipUnchangedBodies(bool on);

/**
 * @brief Reports the current setting of MFTLRedistributeSetSkipUnchangedBodies().
 * @return `true` if unchanged bodies may be skipped.
 * @rank_local
 */
bool MFTLRedistributeGetSkipUnchangedBodies(void);

/**
 * @brief Lets a redistribution give back pool capacity its ranks have outgrown.
 *
 * Pools are grow-only by design (MFTLObjectReserve(), and the reason is in
 * mftl_object.h), so a rank whose share of a body SHRANK keeps the block it held
 * at its high-water mark for the rest of the run. That is not a corner case:
 * MFTLGenerateSphere() builds every body whole on rank 0, and after the opening
 * redistribution rank 0 holds a sliver of each in a block sized for all of it --
 * `3440 * 4^refine + 128` bytes per body, 430 MiB for 512 bodies at refine 4.
 * With this armed, the redistribution offers every body it touched back to
 * MFTLObjectShrinkToFit() once the rebuild is complete.
 *
 * It is a MEMORY policy and nothing else. No count, no content, no derived bit
 * and no partition stamp changes, so the mesh a caller sees is identical either
 * way; `tests/test_redistribution.c` asserts that against both digest halves.
 * It needs no collective and no agreement between ranks -- capacity is purely
 * local state, which is why this setter, unlike
 * MFTLRedistributeSetSkipUnchangedBodies(), is @rank_local.
 *
 * **It does not lower the GENERATION peak**, only the resident footprint. Every
 * body is still built whole on rank 0 before the first redistribution, and this
 * runs inside that redistribution.
 *
 * Default 0.0, i.e. OFF. #MFTL_SHRINK_SLACK_DEFAULT is the value to pass, and
 * its documentation derives why anything below 4 can thrash against the
 * doubling growth ladder.
 *
 * @param slack Waste factor at which a pool is worth reallocating: a pool is
 *              shrunk when `capacity > slack * n`. Zero or negative disables.
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
 * @brief Releases the redistribution workspace hanging off an ensemble.
 *
 * Called by MFTLEnsembleDestroy(). The workspace is created lazily on the first
 * redistribution and holds nothing but working storage, so releasing it early is
 * safe -- the next redistribution simply builds it again.
 *
 * @param p_ws Address of `MFTL_Ensemble::comm_workspace`. Set to #MFTL_NULL on
 *             return. A #MFTL_NULL pointer or a #MFTL_NULL workspace is a no-op.
 *
 * @rank_local
 */
void MFTLCommWorkspaceDestroy(void **p_ws);

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Ghost synchronisation
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Pulls fresh POSITIONS for every ghost vertex from its owning rank.
 *
 * Request-reply over the graph communicator: each rank asks its neighbours for
 * the (body, vertex) pairs it holds as ghosts, and the owners answer with the
 * position. Three neighbour collectives per call, all sized by the actual ghost
 * count -- nothing scales with nproc.
 *
 * Topology is not touched, so this is the per-timestep counterpart to
 * redistribution: call it after advection or smoothing, and redistribute only
 * when ownership can actually have changed.
 *
 * A request names the body by its global id and the vertex by its global id;
 * the owner resolves the body through its sorted local directory and the vertex
 * through an epoch-stamped id table, and the reply comes back in request order,
 * so the requester applies it with a direct indexed write.
 *
 * @param ens The ensemble. All ranks must call collectively.
 * @return MFTL_SUCCESS, or MFTL_ERROR on an invalid ensemble or when some ghost
 *         could not be served (its owner is not a graph neighbour, or does not
 *         hold it) and therefore kept a STALE value. The verdict is this rank's.
 */
int MFTLUpdateGhostPositions(MFTL_Ensemble *ens);

/**
 * @brief Pulls exact NORMALS for every ghost vertex from its owning rank.
 *
 * The same request/reply exchange as MFTLUpdateGhostPositions, over the same
 * communicator, with the same cost -- only the array read and written differs.
 *
 * ── Why this exists at all, when normals need no communication ─────────────
 *
 * MFTLComputeVertexNormals accumulates over every locally held triangle, so an
 * OWNED vertex gets the exact answer with no communication: a vertex-complete
 * halo guarantees every incident face is present. A GHOST vertex has no such
 * guarantee -- its far-side faces live on another rank -- so it is left holding a
 * partial sum, correctly normalised and quietly wrong.
 *
 * That is a trap rather than a limitation, because a partial sum looks exactly
 * like a real normal. This exchange is what closes it: after it, `verts.normal`
 * is exact for every vertex in the pool, and a consumer no longer has to know
 * where the ownership prefix ends. MFTLEnsembleUpdateVertexNormals() is the
 * one-call form.
 *
 * Neither normal nor position synchronisation changes derived connectivity.
 *
 * @note Requires MFTLComputeVertexNormals() to have run on every object first;
 *       it synchronises `verts.normal`, it does not compute it.
 * @param ens The ensemble. All ranks must call collectively.
 * @return MFTL_SUCCESS, or MFTL_ERROR as for MFTLUpdateGhostPositions().
 */
int MFTLUpdateGhostVertexNormals(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Spatial halo
 *
 * A THIRD kind of neighbour data, and it answers a question neither of the two
 * above does:
 *
 *   MFTLRedistribute*        "who owns what now?"          -> topological halo
 *   MFTLUpdateGhostPositions "refresh what I already hold"
 *   MFTLExtendSpatialHalo    "what of body B is NEAR my share of body A?"
 *
 * The decomposition is geometric, so two bodies that touch are cut by the SAME
 * rank boundary. A rank holding the left half of droplet A and none of droplet B
 * cannot see the contact at all, and neither can the rank holding B if the
 * contact is on A's side of the cut. Somebody has to receive a copy, and no
 * amount of topological halo growth produces it -- the two bodies share no
 * edges, so there is no adjacency to grow along.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY THE ARRIVALS GET THEIR OWN OBJECT INSTEAD OF THE BODY'S GHOST SUFFIX
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The obvious design is to append them to body B's pools after the topological
 * ghosts: the ownership prefix is untouched, so every `n_local` loop is
 * unaffected. That much is true. What it misses is the SECOND family of loops,
 * the ones bounded by `tris.n` / `verts.n`, where four things change silently:
 *
 *   - MFTLObjectReorderMorton takes its bounding box over ALL vertices with one
 *     shared range (src/object/derived.c). A partner body on the far side of the
 *     domain stretches that box, so the 21-bit cells get coarser for the OWNED
 *     mesh. Pure locality loss, no test can see it.
 *   - MFTLMeanEdgeLength (src/remeshing/remesh_driver.c) counts twin-less
 *     half-edges, so every appended triangle contributes three. The remeshing
 *     TARGET EDGE LENGTH would then be set partly by another body's geometry.
 *     It carries no ghost assertion, so nothing stops it.
 *   - MFTLComputeVertexNormals scatters over all triangles. Safe only while no
 *     appended corner shares a slot with an existing vertex.
 *   - MFTLRebuildTwins re-pairs purely by (origin, target) vertex SLOT, so a
 *     shared corner can manufacture a twin between two different bodies.
 *
 * And underneath sits a dilemma with no good side: an arriving triangle carries
 * body B's vertex global ids. De-duplicate them against the slots already in the
 * pool and inv_halo_is_vertex_complete breaks (a twin-less half-edge would have
 * OWNED endpoints); give them fresh slots and inv_local_global_ids_are_unique
 * breaks. Both are part of inv_partition_is_sound. Resolving it properly needs
 * the two-way ownership split to become three-way -- one of the three invariants
 * mftl_types.h is built on.
 *
 * So the arrivals go into a SHADOW OBJECT instead: a complete, self-consistent
 * MFTL_Object holding nothing but the received patch. It is not in
 * `ens->local_objects`, so redistribution and the ghost syncs never see it. Nothing
 * that loops over the real body sees it either. And because it is an ordinary
 * object, every existing pass works on it unmodified -- MFTLObjectBuildDerived,
 * MFTLComputeTriangleGeometry, MFTLGetTriangleCorners -- which is
 * what makes the patch reusable by anything that accepts an MFTL_Object, at the
 * price of a consumer having to ask the halo for it rather than finding it
 * already spliced into the body.
 *
 * ── Lifetime ───────────────────────────────────────────────────────────────
 *
 * TRANSIENT, and unavoidably so. Redistribution keeps only owned triangles
 * (`collect_destinations` scans `[0, tris.n_local)`), so a halo cannot survive a
 * repartition even in principle -- and the per-timestep loop redistributes every
 * step. Each MFTLExtendSpatialHalo REPLACES the previous contents rather than
 * accumulating, and the intended order is:
 *
 *   1..4  the usual loop, ending in MFTLRedistribute
 *   5.    MFTLExtendSpatialHalo(...)                      the halo
 *   6.    contact detection
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_SpatialHaloPair
 * @brief One candidate body pair the broad phase kept, ordered by body id.
 *
 * `body_a < body_b`: the smaller id is the A side, the side whose OWNED
 * triangles are the ones reported, and the larger is the side whose triangles
 * travel. That asymmetry is the whole of the "reported once" rule -- see
 * MFTL_SpatialHaloRequest -- so the ordering is a contract and not a
 * convenience. The one exception is a SELF pair, `body_a == body_b`, present
 * only when #MFTL_SpatialHalo::self_pairs asked for it; there the traversal
 * restores "reported once" by triangle id instead (see
 * MFTLEnsembleIntersectObjects()).
 *
 * A pair is present on THIS rank iff this rank holds body A and can see body B,
 * either because it holds part of it or because a patch for it arrived.
 */
typedef struct MFTL_SpatialHaloPair
{
  int body_a;  ///< Smaller body id. Held by this rank; its owned triangles report.
  int body_b;  ///< Larger body id -- or the SAME id for a self pair. Held here, patched here, or both.
} MFTL_SpatialHaloPair;

/**
 * @struct MFTL_SpatialHalo
 * @brief A SPARSE, body-id-keyed directory of shadow objects and body scales.
 *
 * ## IT IS INDEXED BY A SORTED KEY ARRAY, NOT BY BODY ID
 *
 * `body_gid` is strictly ascending and `num_objects` is its length, so entry `e`
 * describes body `body_gid[e]`. A `[n_bodies_global]` array would be the obvious
 * shape and is the one that stops scaling: at a million bodies every rank would
 * hold a million patch pointers and a million radii to describe the handful of
 * bodies it is anywhere near. Use MFTLSpatialHaloPatch() and
 * MFTLSpatialHaloRadius(), which binary-search; the arrays are public only so
 * that a consumer can walk the directory.
 *
 * ## WHICH BODIES ARE IN IT
 *
 * The bodies this rank HOLDS, plus the bodies it RECEIVED A PATCH FOR. The
 * second set is not a subset of the first and that is the point: a rank holding
 * droplet A and nothing of droplet B is exactly the rank that must be told about
 * B, and it has no local object to hang that on.
 *
 * `patch[e]` is MFTL_NULL until body `body_gid[e]` actually receives something.
 * When it is not null it is a fully valid MFTL_Object: twins linked WITHIN the
 * patch, derived connectivity built, and `n_local == 0` because this rank owns
 * none of it.
 *
 * Global ids on a patch are the body's real ids, so a contact reported against a
 * patch triangle names something the owning rank recognises. They are NOT dense
 * and NOT unique against the body's own pool -- a patch is a fragment, and
 * inv_global_ids_are_dense must not be applied to it.
 */
typedef struct MFTL_SpatialHalo
{
  int *body_gid;  ///< [num_objects] body ids, STRICTLY ASCENDING. The key.
  MFTL_Object **patch;  ///< [num_objects] shadow object per entry, or MFTL_NULL.
  double *global_radius;  ///< [num_objects] half the LONGEST SIDE of the body's global box.
  int num_objects;  ///< Entries in the directory. NOT the size of the ensemble.
  int capacity;  ///< Entries allocated.

  /**
   * @brief The candidate pairs the broad phase kept, ascending and unique.
   *
   * Filled by MFTLExtendSpatialHaloNearBodies(); left empty by the explicit
   * request-list form, which is told what to fetch and therefore has no opinion
   * about what to traverse.
   */
  MFTL_SpatialHaloPair *pair;
  int n_pairs;  ///< Entries in `pair`.
  int cap_pairs;  ///< Entries allocated in `pair`.

  /**
   * @brief INPUT to MFTLExtendSpatialHaloNearBodies(): also derive SELF pairs.
   *
   * False by default. When set, a body travels towards a neighbour's share of
   * the SAME body as well as towards smaller ids, and `pair` gains `(B, B)` for
   * every body this rank owns something of. That is the exchange a
   * self-proximity query needs -- two sheets of one body across a rank boundary,
   * a thin neck on a cut plane -- and it rides the same two collectives as the
   * pairwise one rather than costing a second exchange. The explicit request
   * form ignores it: a request `(B, B)` already asks for exactly this.
   */
  bool self_pairs;

  MFTL_Domain *domain;  ///< Borrowed from the ensemble.
} MFTL_SpatialHalo;

/**
 * @struct MFTL_SpatialHaloRequest
 * @brief "Fetch me the parts of body B within @c margin of my share of body A."
 *
 * The pair is ORDERED and only B travels. That follows from how a distributed
 * contact must be attributed: if the rank owning the A-side triangle is the one
 * that reports the pair, then it is the one that needs the B-side copy, and
 * sending A as well would have both ranks report the same contact.
 */
typedef struct MFTL_SpatialHaloRequest
{
  int body_a;  ///< Body id (#MFTL_Object::global_id). Only its EXTENT is used.
  int body_b;  ///< Body id of the body whose triangles travel.
  double margin;  ///< Absolute distance, in domain length units.
} MFTL_SpatialHaloRequest;

/**
 * @brief Allocates a halo seeded with the bodies @p ens currently holds here.
 *
 * The directory is re-synchronised against the ensemble by every later
 * MFTLSpatialHaloResolveRadii() and MFTLExtendSpatialHalo*() call, so a halo
 * created before a redistribution is still usable after one. Seeding it here
 * only saves the first growth.
 *
 * @param ens The ensemble. Only its LOCAL object list is read.
 * @return A halo with every patch #MFTL_NULL. Free with MFTLSpatialHaloDestroy().
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
 * @brief Fills `global_radius` for every body THIS RANK HOLDS: half the longest
 *        side of that body's bounding box over all of its holders.
 *
 * Exists as its own call because a caller that scales anything by the body size
 * needs the number BEFORE it can state a margin, and the number is only
 * obtainable collectively.
 *
 * Why the GLOBAL box and not this rank's: a partition is a fraction of the body,
 * so a margin scaled by a local extent would shrink as ranks are added and the
 * same simulation would detect contact at a different separation depending only
 * on how it was launched.
 *
 * ## IT IS A PER-BODY REDUCTION, SO IT GOES TO THE BODY'S HOLDERS
 *
 * Twelve values per body -- a lower corner and a negated upper corner, in the raw
 * frame and in one displaced by half a domain -- through MFTLEnsembleReduceBodies()
 * rather than through an `MPI_Allreduce` over `comm_world` on a
 * `[12 * n_bodies_global]` staging buffer. The two frames are what stop a droplet
 * ON THE SEAM from reporting a radius of half the box, and they cost payload
 * rather than a second collective.
 *
 * A body this rank holds nothing of gets no radius here, because this rank has
 * nothing to scale by it. Ask MFTLSpatialHaloRadius() and the answer is 0.0 until
 * a patch for that body arrives carrying the owner's number.
 *
 * @note Collective: every rank must call it, including one holding no bodies.
 * @note Reads current positions and uses compact connectivity when available.
 *
 * @param ens  The ensemble. Its local object list is re-synchronised into the
 *             halo directory first, so a halo created before a redistribution
 *             keeps working after one.
 * @param halo The halo whose `global_radius` entries are filled in.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 */
int MFTLSpatialHaloResolveRadii(MFTL_Ensemble *ens, MFTL_SpatialHalo *halo);

/**
 * @brief Exchanges the requested patches and rebuilds every affected shadow object.
 *
 * ── What replaced the brute force ──────────────────────────────────────────
 *
 * The version this is ported from ran, PER OBJECT PAIR, one neighbour alltoall
 * of macro-AABBs, four more for counts, four alltoallvs of payload, and -- for
 * every neighbour that passed the macro test -- a full scan of the triangle list
 * rebuilding each triangle's AABB from its three vertices INSIDE the neighbour
 * loop. That is O(n_neighbours * n_triangles) box constructions and 9
 * collectives, both multiplied by the pair count: an emulsion of 500 droplets
 * pays 124 750 pairs of that per timestep.
 *
 * Here the whole request list costs THREE collectives regardless of its length
 * (the radii reduction, one extent allgather, one variable-length payload), each
 * triangle's box is built once per body and reused for every neighbour and every
 * partner, and the work is indexed by BODY rather than by pair -- which is also
 * what stops a triangle relevant to two different A bodies from being sent, and
 * therefore reported, twice.
 *
 * ── The extent exchange is SPARSE, and that is what makes it scale ─────────
 *
 * A rank publishes one record per body IT HOLDS -- `(body id, centre, half-width,
 * global radius, tolerance)` -- through a neighbourhood allgatherv, so the
 * payload is `O(local bodies * degree)` and not `O(bodies in the ensemble) *
 * degree`. The dense form cost 48 bytes per body per neighbour, i.e. 1.2 kB per
 * body per rank at a full 26-neighbourhood: a gigabyte at a million bodies, to
 * describe droplets that are almost all nowhere near this rank. Requests naming
 * a body this rank does not hold are simply not matched -- there is nothing here
 * to send.
 *
 * ── Extents are CENTRE + HALF-WIDTH, and that is a correctness statement ───
 *
 * Every extent here -- what a rank publishes about its share, and what each
 * triangle contributes -- is a centre on the periodic circle plus a half-width,
 * compared by minimum-image distance. A (lo, hi) box cannot do this job, and
 * fails in both directions at once:
 *
 *   IT MISSES CONTACTS. Two shares meeting across the seam are a domain length
 *   apart in stored coordinates, so min/max comparison reports no overlap for
 *   two surfaces that are touching. MFTL_IntersectObjects tries every periodic
 *   image of B, but only for triangles it HAS -- and this test is what decides
 *   whether they are sent. The miss happens one stage before anything the BVH
 *   can see, and nobody reports the contact.
 *
 *   IT OVER-SENDS ENORMOUSLY. A share holding even one seam-straddling triangle
 *   has a min/max box spanning the whole domain, because MFTLGetTriangleBounds
 *   is seam-consistent and correctly places that triangle's corners just outside
 *   the box on one side. Such a share overlaps everything, so every neighbour
 *   ships its entire share of every partner body. A droplet on the seam turns
 *   the halo into an all-to-all broadcast of the mesh, every timestep. Measured
 *   at np = 27 on a two-sphere contact at the seam: 314 triangles ghosted with
 *   boxes against 56 with intervals -- and 320 against 0 for a placement where
 *   no contact is cross-rank at all.
 *
 * A rank's own extent is measured in two frames, raw and displaced by half a
 * domain, keeping the tighter per axis; that is what makes a straddling share
 * report its true size. The payload is six doubles per body per rank either way.
 * Both the published extents and the margin come from the SAME representation,
 * so the macro pre-filter is strictly more permissive than the per-triangle test
 * it short-cuts -- a pre-filter that is not is a silent miss generator, and the
 * un-widened macro box was exactly that for proximity contacts at a cut.
 *
 * The scheme degrades rather than breaking: a share genuinely wider than half
 * the domain gets a half-width above L/2, the minimum-image test accepts
 * everything, and the behaviour falls back to the old conservative one.
 *
 * ── Blocking or not ───────────────────────────────────────────────────────
 *
 * The extent allgather is non-blocking and is overlapped with building the local
 * per-triangle boxes, the one piece of real work that does not depend on it.
 * Everything after is a strict dependency chain -- a triangle cannot be
 * classified before the extents arrive, packed before it is classified, or
 * unpacked before it is sent -- so the payload goes through the ordinary
 * MFTLExchangeStart path, which already merges the count exchange into the
 * transfer.
 *
 * Deadlock freedom: the sequence of collectives is derived from @p requests and
 * the neighbour count, never from local geometry, so every rank executes the
 * same one. A rank holding nothing of either body contributes zero counts and
 * still enters every call.
 *
 * @param ens      The ensemble. Every rank must call, but the request lists no
 *                 longer have to agree: the sequence of collectives is fixed by
 *                 the neighbour count alone, and a request is only ever a
 *                 statement about what THIS rank is willing to send.
 * @param requests [n_requests] ordered pairs and their margins. A pair naming a
 *                 body this rank does not hold as B is ignored.
 * @param n_requests Zero is a valid no-op that still enters every collective.
 * @param halo     Filled in. Patches of the requested B bodies are DISCARDED
 *                 first, not accumulated. `pair` is left untouched -- an explicit
 *                 request list says what to fetch, not what to traverse.
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments.
 *
 * @note Collective over the ensemble's graph communicator.
 * @note Reads current `verts.pos` and triangle connectivity. No geometry cache
 *       refresh is required after motion.
 */
int MFTLExtendSpatialHalo(MFTL_Ensemble *ens, const MFTL_SpatialHaloRequest *requests, int n_requests, MFTL_SpatialHalo *halo);

/**
 * @brief The same exchange, with the pair list DERIVED from the body extents
 *        instead of enumerated by the caller.
 *
 * ## WHY THIS EXISTS: AN ALL-PAIRS REQUEST LIST IS THE LAST \f$O(B^2)\f$ TERM
 *
 * The caller that wants "every body against every other body" -- contact
 * detection -- used to build exactly that list: `B*(B-1)/2` ordered pairs, on
 * every rank, every timestep. At 16 384 bodies that is 134 million records and
 * two gigabytes of request array before a single triangle is examined, and it
 * grows quadratically in a quantity that is already the one the sparse local
 * object list was introduced to stop paying linearly for.
 *
 * Nothing in the list was information: a pair is worth fetching exactly when the
 * two bodies' extents are within the pair's envelope, and after the extent
 * exchange every rank already holds the extents it needs to decide that. So the
 * list is not built at all. This function bins the neighbours' published extents
 * into a uniform grid whose cell is at least as wide as the widest body plus the
 * largest envelope, and asks each of its own bodies for the 27 cells around it.
 * Cost is `O(local bodies + records)` and the answer is the same set.
 *
 * ## THE PAIR ORDER IS THE "REPORTED ONCE" RULE
 *
 * Body B travels towards body A only when `A < B` by body id, matching
 * MFTL_SpatialHaloRequest: the holder of the smaller id is the rank that reports
 * the pair, so it is the rank that must receive the copy. Sending both ways
 * would have both ranks report the same contact.
 *
 * ## WHAT COMES BACK
 *
 * `halo->pair` is the candidate list for the traversal, ascending and unique:
 * every `(A, B)` with `A < B` such that this rank holds A and can see B -- as a
 * local object, as a patch, or both. A caller with a different definition of
 * "near" than `MAX(tol_A, tol_B)` should use the explicit request form instead.
 *
 * @param ens The ensemble. Collective; every rank must call.
 * @param tolerance `[ens->n_local_objects]` contact envelope per LOCAL SLOT, in
 *                  the order of `ens->local_objects`. MFTL_NULL means zero for
 *                  every body, i.e. overlap-only.
 * @param halo Filled in. EVERY patch is discarded first, and `pair` is rebuilt.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble or halo.
 *
 * @note Collective over the ensemble's graph communicator.
 */
int MFTLExtendSpatialHaloNearBodies(MFTL_Ensemble *ens, const double *tolerance, MFTL_SpatialHalo *halo);

/**
 * @brief The received patch of body @p body_gid, or MFTL_NULL if it received nothing.
 *
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return The shadow object, or #MFTL_NULL. It is owned by the halo and is
 *         invalidated by the next MFTLExtendSpatialHalo() or
 *         MFTLSpatialHaloDestroy().
 */
static inline const MFTL_Object *MFTLSpatialHaloPatch(const MFTL_SpatialHalo *halo, int body_gid)
{
  const int e = MFTLSpatialHaloFindBody(halo, body_gid);
  return (e == MFTL_ERROR) ? MFTL_NULL : halo->patch[e];
}

/**
 * @brief Half the longest side of body @p body_gid's box over ALL ranks.
 *
 * @param halo     The halo, or #MFTL_NULL.
 * @param body_gid The body id.
 * @return The radius MFTLSpatialHaloResolveRadii() resolved, or 0.0 if this halo
 *         knows nothing about that body.
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
 * @brief Where the time inside MFTLRedistribute() and MFTLUpdateGhostPositions()
 *        actually goes. Purely diagnostic.
 *
 * It exists because this module had no instrumentation at all: `git grep
 * MPI_Wtime -- src/` returned exactly one hit, in `remesh_comm.c`. The
 * redistribution is the largest phase of the per-timestep loop
 * (docs/analysis.md §2), and every published statement about
 * WHICH PART of it is expensive was inference from the source rather than a
 * measurement. Architectural decisions about this module were being made on that
 * inference. These counters replace it.
 *
 * ## HOW TO READ IT
 *
 * Everything is RANK-LOCAL and everything ACCUMULATES until
 * MFTLRedistributeStatsReset(). A caller wanting a per-rank mean divides by
 * `n_calls`; a caller wanting a fleet figure reduces the fields it cares about
 * itself. Nothing here takes a collective, which is what makes it safe to leave
 * enabled: a reduction inside the measurement would synchronise the very phases
 * it is trying to time.
 *
 * The `t_*` fields are SECONDS and they PARTITION the redistribution: every
 * bucket except `t_total` is disjoint from every other, so
 * `t_total - (sum of the rest)` is honest residue and should be small. If it is
 * not, a phase is missing a timer.
 *
 * One of them reads differently from the others. `t_wait` overlaps nothing, but
 * it absorbs LOAD IMBALANCE: a rank that arrives early at MFTLExchangeWait()
 * charges its neighbours' slowness to itself. Read it as an imbalance measure,
 * not as communication cost.
 *
 * Allocation is deliberately NOT a bucket. Its sites are spread through every
 * phase, and wrapping individual `malloc` calls in a clock read whose cost is
 * comparable to the call itself measures the instrument rather than the
 * allocator. `t_setup` and `t_teardown` bracket the per-call scratch, which is
 * where the churn actually lives, and `n_allocs` / `n_bytes_allocated` count the
 * rest.
 *
 * ## THE COST OF MEASURING
 *
 * `n_clock_reads` counts every MPI_Wtime() call the instrumentation made. The
 * per-object buckets are read once per body per call, so a 1000-body ensemble
 * reads the clock tens of thousands of times per timestep and that is NOT
 * negligible against a 10 ms phase. Multiply `n_clock_reads` by the measured
 * cost of one MPI_Wtime() on the machine to get the perturbation, and treat the
 * benchmark's un-instrumented wall clock as the authoritative total.
 */
typedef struct
{
  // --- Redistribution: the disjoint phase buckets, in call order -------------

  double t_total;  ///< The whole of MFTLRedistribute(), including the serial short-circuit.

  /**
   * Per-call scratch setup: the eight `nproc`-sized allocations, the `nproc`
   * initialisation loops, and the buffer-hint arithmetic.
   *
   * Invariant to how much of the mesh moves and proportional to the PROCESS
   * COUNT rather than to the local mesh, which is what makes it the bucket to
   * watch in a weak-scaling sweep.
   */
  double t_setup;

  double t_owners;  ///< Ownership classification. O(N+H) and invariant to how much moves.
  double t_masks;  ///< Clearing the per-object keep flags and de-duplication mask. O(V+T) per body.
  double t_destinations;  ///< Destination analysis: seven candidates per owned triangle.
  /**
   * The neighbour-vs-global decision Allreduce. One int, so pure latency -- but
   * it is the FIRST collective after the per-object loop, so like `t_wait` it
   * absorbs load imbalance. On a small mesh where the loop is short it can read
   * as a double-digit share of the phase and still be measuring nothing but the
   * spread between ranks.
   */
  double t_route;

  /**
   * The per-body skip vote: one MFTLEnsembleReduceBodies() over each body's
   * holders, one value per body held.
   *
   * Small by payload, but it is the
   * FIRST collective after the object loop, so it takes over t_route's other job
   * of absorbing the spread between ranks. Read the two together: if this bucket
   * is large and `t_route` has collapsed, the imbalance simply moved and the
   * skip cost one latency. If both are large, it did not.
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
   * `tris.verts` / `tris.neighbors` entries of every kept triangle.
   *
   * THERE IS NO `derived: topology` BUCKET, and its absence is a result rather
   * than an omission. The rebuild used to close with MFTLObjectBuildDerived(),
   * one streaming pass re-deriving two arrays whose inputs this loop already
   * holds in registers; it now emits them as it goes.
   *
   * READ THE TWO BUCKETS TOGETHER, because most of the removed pass reappears
   * here. Isolated at np = 4 on 1000 bodies with the skip off: the pass cost
   * 0.588 ms/step and 0.369 of it came back into this bucket, so the trade is
   * worth about 0.22 ms/step -- roughly a THIRD of what the traffic arithmetic
   * predicts, because a streaming pass the prefetcher handles perfectly is being
   * traded for scattered stores in a loop that is already dependency-bound.
   * docs/analysis.md §2.5 has the measurement.
   */
  double t_relink;
  double t_arrivals;  ///< Classifying and writing what arrived, derived entries included.
  double t_twins;  ///< repair_twins() over the half-edges that lost a partner.

  /**
   * Tearing the per-call scratch down again: every `free` on the path, including
   * the per-destination id lists and the send and receive buffers.
   *
   * `t_setup + t_teardown` is what a persistent workspace would remove, and
   * docs/analysis.md §2.1 identifies that churn as the reason a
   * first benchmark run is slower than every repeat of it. The counters
   * `n_allocs` / `n_bytes_allocated` say how much of it there is; these two say
   * what it costs.
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
   * Bodies that reached the rebuild with nothing kept and nothing arriving.
   *
   * These are bodies whose entities have all left this rank: the rebuild gives
   * their pool blocks back (and the object is destroyed at the end of the call),
   * so the counter measures departure churn rather than wasted work.
   */
  long n_objects_empty_local;

  /**
   * Bodies left entirely alone because no rank saw them change.
   *
   * `n_objects_skipped / n_objects_visited` is THE number this optimisation
   * turns on, and it is a body fraction rather than an entity fraction: a body
   * straddling a cut plane very likely has some entity cross every step, so the
   * ~95 % of entities that do not move does not predict it. See
   * MFTLRedistributeSetSkipUnchangedBodies().
   */
  long n_objects_skipped;

  /**
   * Bodies the second pass of the object loop picked up.
   *
   * Their ownership classification therefore ran twice, which is the price of
   * not storing a per-body owner array between the two passes. Large against
   * `n_objects_skipped` means the vote is mostly being overruled and the split
   * is buying little.
   */
  long n_objects_reclassified;

  /**
   * Of those, the ones rebuilt only to reach a fixed point.
   *
   * A rebuild is not idempotent -- it moves entities that are no longer corners
   * of one of this rank's triangles to the end of their run -- so a body is
   * rebuilt once more after it stops changing before it may be skipped. That is
   * what makes a skipped body's SLOT ORDER identical to a rebuilt one's, not
   * merely its contents. The cost is one rebuild per dirty-to-clean transition:
   * read it against `n_objects_skipped` to see what exactness cost.
   */
  long n_objects_settled;

  /**
   * Bodies whose pool blocks were given back, and how many bytes that recovered.
   *
   * Zero unless MFTLRedistributeSetShrinkSlack() is armed. Read them against
   * `n_objects_visited`: a healthy steady state shrinks nothing, because both
   * resize operators leave capacity within a factor of two of the count and the
   * gate is four. A NON-ZERO count every step means the gate is firing on
   * jitter and the slack is too low -- see #MFTL_SHRINK_SLACK_DEFAULT.
   *
   * `n_bytes_reclaimed` is the drop in requested capacity, not a measured RSS
   * change: whether free() returns pages to the OS is allocator-dependent, so
   * only a resident-set measurement can answer that.
   */
  long n_objects_shrunk;
  long n_bytes_reclaimed;  ///< Capacity bytes released by the shrink. See `n_objects_shrunk`.

  /**
   * Bodies created here on arrival, and destroyed here on becoming empty.
   *
   * A rank holds an #MFTL_Object only for the bodies it has entities of, so a
   * body crossing a rank boundary is created on the receiving rank and destroyed
   * on the one it left. These count that churn -- the per-body footprint a rank
   * used to carry unconditionally, now paid only when a body actually moves.
   *
   * They are also the non-vacuity measure for any test of the sparse list: a
   * fixture in which both stay zero exercised no arrival and no departure, and
   * would pass unchanged with the whole mechanism deleted.
   */
  long n_objects_created;
  long n_objects_destroyed;  ///< See n_objects_created.

  long n_allocs;  ///< Allocation calls on the path.
  long n_frees;  ///< Matching frees.
  long n_bytes_allocated;  ///< Bytes requested, summed. Not a peak.
  long n_gid_table_bytes;  ///< Bytes the arrival id tables asked for.

  /**
   * @name Persistent workspace, split by WHAT it is proportional to
   *
   * High-water marks, not a running sum: what the workspace currently holds,
   * which is what a rank's resident footprint actually is.
   *
   * The split is the point. `n_bytes_allocated` says how much churn there is;
   * these three say which of them can be expected to STAY CONSTANT as a job
   * grows, and that is the only form of the question weak scaling cares about:
   *
   *   ws_bytes_per_body    proportional to the number of bodies THIS RANK
   *                        holds. It used to be sized by the whole ensemble; a
   *                        figure that grows with P in a weak-scaling sweep is
   *                        now a regression.
   *   ws_bytes_per_rank    proportional to nproc. Bounded-degree neighbour
   *                        traffic does not need it; the global Alltoallv
   *                        fallback does.
   *   ws_bytes_per_entity  proportional to the mesh this rank actually holds.
   *                        The only one of the three that is SUPPOSED to be
   *                        constant per rank in a weak sweep, and the yardstick
   *                        the other two are read against.
   *
   * A run whose per-body and per-rank figures are flat as P rises is the
   * acceptance criterion; one where they track P is the defect.
   * @{
   */
  long ws_bytes_per_body;
  long ws_bytes_per_rank;
  long ws_bytes_per_entity;
  /** @} */

  /**
   * How the arrival id tables were indexed, counted per use.
   *
   * Direct addressing is `tag[gid]`: one load, and a table proportional to the
   * BODY's id range. Sparse is hashed with linear probing: a table proportional
   * to what this rank actually holds, at roughly 1.5x the lookup cost. The
   * crossover is a density ratio and the mode is chosen per body per call, so
   * these two say which regime a run was actually in rather than which one it was
   * expected to be in.
   */
  long n_gid_table_direct;
  long n_gid_table_sparse;  ///< See n_gid_table_direct.

  /**
   * Times the per-object de-duplication bit budget was exhausted.
   *
   * Above #MFTL_DEDUP_BITS distinct destinations the vertex lists fall back to a
   * radix sort per destination. Unreachable below 32 remote destinations, so on
   * a 26-neighbour graph this is zero by construction and only the global route
   * can raise it.
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
 * @brief Zeroes the accumulator.
 *
 * Call it after the warm-up steps of a benchmark, so that first-touch faults and
 * the initial scatter do not land in the steady-state figures.
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
