/**
 * @file ghost_sync.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Per-timestep ghost synchronisation: one per-vertex vector, unchanged
 *        topology.
 *
 * Redistribution is the expensive operation and it is only needed when OWNERSHIP
 * can have changed. Between two redistributions the partition is fixed and all a
 * ghost needs is fresh values, which is what this file moves -- 32 bytes per
 * ghost vertex per field, three neighbourhood collectives, nothing that scales
 * with nproc.
 *
 * ### One field per exchange, and why that is not a missed fusion
 *
 * The whole exchange -- who asks whom for which vertex, and where each reply
 * lands -- is independent of the payload. The implementation is parameterised
 * by which per-vertex 3-vector is being moved, and both callers share it:
 *
 *   MFTLUpdateGhostPositions      verts.pos     -- authoritative, cannot be
 *                                                 recomputed anywhere else
 *   MFTLUpdateGhostVertexNormals  verts.normal  -- recomputable for OWNED
 *                                                 vertices, not for ghosts
 *
 * Shipping position, normal, both curvatures and the area in one 80-byte message
 * looks like a saving (one exchange instead of two) and is not: four fifths of
 * that payload is a function of the positions and the topology, so it would be
 * sent to save a receiver-side pass that costs less than the bytes do. Each field
 * costs exactly one exchange here, and a caller
 * that does not need the normals does not send them -- which is the common case,
 * because MFTLComputeVertexNormals is exact for owned vertices with no
 * communication at all (mftl_geometry.h).
 *
 * ### Why a request/reply and not a push
 *
 * Only the HOLDER of a ghost knows that it holds it; the owner has no record of
 * who copied its vertex. The request phase supplies exactly that information on
 * every call. No persistent owner-to-holder state is stored.
 */

#include "mftl_communication.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/**
 * @brief Which per-vertex 3-vector an exchange moves.
 *
 * An enum rather than a pointer-to-member or an array offset, because the two
 * cases share the exchange code and differ only in which array is packed. Resolved
 * once per call, outside every loop.
 */
typedef enum
{
  GHOST_FIELD_POSITION = 0,  ///< `verts.pos`.
  GHOST_FIELD_NORMAL = 1  ///< `verts.normal`.
} ghost_field;

/**
 * @brief The pool array a field names, read side.
 * @param obj The mesh.
 * @param f   Which field.
 * @return Const pointer to the named `[3*capacity]` array.
 */
static const double *ghost_field_read(const MFTL_Object *obj, ghost_field f)
{
  return (f == GHOST_FIELD_POSITION) ? obj->verts.pos : obj->verts.normal;
}

/**
 * @brief The pool array a field names, write side.
 * @param obj The mesh.
 * @param f   Which field.
 * @return Writable pointer to the named `[3*capacity]` array.
 */
static double *ghost_field_write(MFTL_Object *obj, ghost_field f)
{
  return (f == GHOST_FIELD_POSITION) ? obj->verts.pos : obj->verts.normal;
}

/**
 * @brief Answers the requests addressed to one object.
 *
 * The owner has to turn a global id into a pool slot, which is the one place
 * this module would have wanted a hash map. It does not need one: global ids are
 * dense per entity class, so a scatter of the OWNED vertices into an epoch-
 * stamped array (gt_* in comm_internal.h) answers every request for that object
 * with a single indexed read.
 *
 * The scatter is O(owned vertices) and the requests are O(halo), so it is only
 * worth doing per object that actually received a request -- which is why the
 * caller groups the requests by object first. In the many-body case that skips
 * the scatter for every body this rank does not hold, and the table for a body it
 * does hold is a few kilobytes and stays in cache.
 *
 * A request the owner cannot serve is a broken invariant (the requester believes
 * this rank owns a vertex it does not have). Rather than dereferencing garbage,
 * the reply is marked with MFTL_ERROR so the requester can skip it -- one
 * predictable compare on the apply side, and a diagnosable failure.
 *
 * @param      obj        The mesh whose OWNED vertices answer the requests.
 * @param      field      Which per-vertex 3-vector is being synchronised.
 * @param      flat_recv  The received request ids, all senders concatenated.
 * @param      order      Request indices belonging to this object.
 * @param      first      First slot of @p order to serve.
 * @param      last       One past the last slot to serve.
 * @param[out] replies    One payload per request. An unanswerable request gets
 *                        `global_id == MFTL_ERROR`.
 * @param      tbl        Epoch-stamped id-to-slot table, reused across objects.
 * @param[out] n_unserved Incremented once per request this rank could not answer.
 */
static void answer_object_requests(const MFTL_Object *obj, ghost_field field, const int *restrict flat_recv, const int *restrict order, int first, int last,
                                   MFTL_GhostUpdatePayload *restrict replies, mftl_gid_table *tbl, int *n_unserved)
{
  const MFTL_Vertex *const restrict vr = obj->verts.rec;
  const double *const restrict src = ghost_field_read(obj, field);
  const int n_local = obj->verts.n_local;

  CS_BEGIN(ts_tbl);
  /* Entry count and id range: the owned vertices go in, and their ids are drawn
   * from the body's whole range. gt_begin() takes the direct-address path while
   * this rank holds a reasonable share of the body and switches to a sparse table
   * when it does not, which is what stops the table growing with the process
   * count at a fixed problem size. See comm_internal.h. */
  gt_begin(tbl, n_local, MFTL_MAX(obj->next_vertex_gid, 1));
  for (int v = 0; v < n_local; ++v) gt_put(tbl, vr[v].global_id, v);
  CS_END(ts_tbl, g_t_table);

  for (int i = first; i < last; ++i)
  {
    const int k = order[i];
    const int gid = flat_recv[2 * k + 1];
    MFTL_GhostUpdatePayload *const p = &replies[k];

    if (gid < 0 || !gt_has(tbl, gid))
    {
      MFTL_ASSERT(0 && "ghost sync: asked for a vertex this rank does not own");
      memset(p, 0, sizeof *p);
      p->global_id = MFTL_ERROR;
      (*n_unserved)++;
      continue;
    }

    const int v = gt_get(tbl, gid);

    p->value[0] = src[3 * v + 0];
    p->value[1] = src[3 * v + 1];
    p->value[2] = src[3 * v + 2];

    p->global_id = gid;  // echoes the request, which is how the requester matches it
  }
}

/**
 * @brief The shared request/reply exchange behind both public ghost syncs.
 * @param ens   The ensemble.
 * @param field Which per-vertex 3-vector to move.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the exchange failed.
 */
static int ghost_sync_vec3(MFTL_Ensemble *ens, ghost_field field)
{
  if (ens == MFTL_NULL || ens->local_objects == MFTL_NULL) return MFTL_ERROR;
  MFTL_ASSERT(MFTL_MPI_GhostUpdatePayload != MPI_DATATYPE_NULL && "ghost sync: call MFTLRegisterMpiTypes() first");

  MFTL_Domain *const domain = &ens->domain;
  MFTL_Object *const *const objects = ens->local_objects;
  const int num_objects = ens->n_local_objects;
  const int nproc = domain->nproc;
  const int nn = domain->num_neighbors;

  /* The body-count guard is GLOBAL: three neighbourhood collectives follow, so a
   * rank holding nothing must still enter them and exchange zero bytes. `nn` and
   * `nproc` are safe to test locally -- a rank with no graph neighbours has no
   * ghosts to sync and nobody is expecting a message from it, and nproc == 1 is
   * the same number everywhere. */
  if (ens->n_bodies_global == 0 || nn == 0 || nproc == 1) return MFTL_SUCCESS;

  CS_BEGIN(ts_total);
  MFTLCommStats.g_n_calls++;
  CS_BEGIN(ts_collect);

  /* ── 1. Collect every ghost we hold, grouped by its owner ────────────────
   *
   * Two lists per neighbour, filled in lockstep: the REQUEST (object index,
   * vertex gid) that goes on the wire, and the TARGET (object index, local vertex
   * slot) that stays here. Recording the target now is what lets the apply phase
   * be a direct indexed write. Re-deriving it instead costs a hash lookup per
   * ghost, every timestep.
   *
   * Ghosts are an array SUFFIX, so the loop bound is the ownership filter. */
  mftl_int_buffer *const requests = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nn * sizeof(mftl_int_buffer), "ghost request lists");
  mftl_int_buffer *const targets = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nn * sizeof(mftl_int_buffer), "ghost target lists");
  for (int i = 0; i < nn; ++i)
  {
    ib_init(&requests[i], 1024);
    ib_init(&targets[i], 1024);
  }

  int *const rank_to_neigh = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "ghost neighbour index");
  for (int r = 0; r < nproc; ++r) rank_to_neigh[r] = MFTL_ERROR;
  for (int i = 0; i < nn; ++i) rank_to_neigh[domain->neighbors[i]] = i;

  int orphaned = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    const MFTL_Object *const obj = objects[o];
    const MFTL_Vertex *const vr = obj->verts.rec;

    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      const int owner = vr[v].owner_proc;
      const int n_idx = (owner >= 0 && owner < nproc) ? rank_to_neigh[owner] : MFTL_ERROR;
      if (n_idx == MFTL_ERROR)
      {
        /* The owner is not in our 26-neighbourhood, so no message will reach it.
         * Skipping keeps every rank inside the collectives below; aborting here
         * would deadlock the ranks that are still fine.
         *
         * ASSERTED, because the consequence is silent and downstream: the ghost
         * keeps a STALE POSITION, so this rank and the owner compute different
         * centres for the same triangle, disagree about its destination in the
         * next redistribution, and someone's owned triangle quietly loses an edge
         * neighbour. The stderr line below is the Release symptom and it is easy
         * to miss in a job log; in Debug this must stop the run at the cause.
         *
         * MFTLRemeshSyncGhostIds has the same structure and answers it with a
         * global-communicator fallback. The same fix belongs here, and is not
         * done: this function runs once per TIMESTEP rather than once per remesh,
         * so it may not grow an O(nproc) path without a measurement first. */
        MFTL_ASSERT(0 && "ghost sync: ghost vertex whose owner is not a graph neighbour -- its position will not be updated");
        orphaned++;
        continue;
      }

      /* The REQUEST names the body by its global id -- the owner has its own
       * slot for it and no reason for it to be the same one. The TARGET never
       * leaves this rank, so it keeps the local slot and the apply phase stays a
       * direct indexed write. */
      ib_add2(&requests[n_idx], obj->global_id, vr[v].global_id);
      ib_add2(&targets[n_idx], o, v);
      MFTLCommStats.g_n_ghosts_requested++;
    }
  }
  free(rank_to_neigh);
  MFTLCommStats.g_n_orphaned += orphaned;

  if (orphaned > 0)
  {
    fprintf(stderr, "[MFTL rank %d] ghost sync: %d ghost vertices whose owner is not a graph neighbour were not updated\n", domain->rank, orphaned);
  }

  CS_END(ts_collect, g_t_collect);

  /* ── 2. Exchange request counts ──────────────────────────────────────────
   *
   * Non-blocking, and this is the ONE place in this function where it buys
   * anything: assembling the flat request image below is real work that does not
   * depend on the counts. Everything after it is a strict dependency chain --
   * the replies cannot be computed before the requests arrive and cannot be
   * applied before they come back -- so the two alltoallv calls are blocking. */
  int *const n_send = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "ghost send counts");
  int *const n_recv = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "ghost recv counts");
  for (int i = 0; i < nn; ++i) n_send[i] = requests[i].size;

  MPI_Request count_req;
  CS_BEGIN(ts_counts);
  MPI_Ineighbor_alltoall(n_send, 1, MPI_INT, n_recv, 1, MPI_INT, domain->comm_graph, &count_req);

  int *const s_displs = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "ghost send displs");
  int total_send = 0;
  for (int i = 0; i < nn; ++i)
  {
    s_displs[i] = total_send;
    total_send += n_send[i];
  }

  int *const flat_send = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(total_send, 1) * sizeof(int), "ghost request image");
  for (int i = 0; i < nn; ++i)
  {
    if (n_send[i] > 0) memcpy(flat_send + s_displs[i], requests[i].data, (size_t) n_send[i] * sizeof(int));
  }

  MPI_Wait(&count_req, MPI_STATUS_IGNORE);
  CS_END(ts_counts, g_t_counts);

  int *const r_displs = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "ghost recv displs");
  int total_recv = 0;
  for (int i = 0; i < nn; ++i)
  {
    r_displs[i] = total_recv;
    total_recv += n_recv[i];
  }

  /* ── 3. Exchange the requests ────────────────────────────────────────────── */
  CS_BEGIN(ts_req);
  int *const flat_recv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(total_recv, 1) * sizeof(int), "ghost incoming requests");
  MPI_Neighbor_alltoallv(flat_send, n_send, s_displs, MPI_INT, flat_recv, n_recv, r_displs, MPI_INT, domain->comm_graph);
  free(flat_send);
  CS_END(ts_req, g_t_requests);
  MFTLCommStats.g_n_bytes_sent += (long) total_send * (long) sizeof(int);
  MFTLCommStats.g_n_bytes_recv += (long) total_recv * (long) sizeof(int);

  CS_BEGIN(ts_answer);

  /* ── 4. Answer them ──────────────────────────────────────────────────────
   *
   * The incoming stream is grouped by sender and, within a sender, by object.
   * Answering it in arrival order would rebuild the id table once per (sender,
   * object) pair, so the requests are first grouped by OBJECT with a counting
   * sort -- O(requests + objects), no comparisons -- and each object's id table
   * is then built exactly once. The replies are written back at the request's
   * ORIGINAL position, because the requester matches them positionally. */
  const int n_replies = total_recv / 2;
  MFTL_GhostUpdatePayload *const replies =
      (MFTL_GhostUpdatePayload *) MFTLMallocOrDie((size_t) MFTL_MAX(n_replies, 1) * sizeof(MFTL_GhostUpdatePayload), "ghost replies");

  int *const obj_off = (int *) MFTLCallocOrDie((size_t) num_objects + 1, sizeof(int), "ghost request grouping");
  int *const order = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_replies, 1) * sizeof(int), "ghost request order");
  int unserved = 0;

  {
    /* Resolve each request's body id to THIS rank's slot for it, ONCE, in place.
     * Everything downstream then buckets by local slot, so the counting sort and
     * its cursors stay sized by what this rank holds rather than by the
     * ensemble. `flat_recv[2k+1]` -- the vertex id -- is left alone.
     *
     * A request for a body this rank has nothing of is a routing error, not a
     * partition this rank is simply not part of: the requester named this rank
     * as the vertex's OWNER, so it must hold the body. */
    int dropped = 0;
    for (int k = 0; k < n_replies; ++k)
    {
      const int slot = MFTLEnsembleFindLocalSlot(ens, flat_recv[2 * k]);
      flat_recv[2 * k] = slot;
      if (slot != MFTL_ERROR)
        obj_off[slot + 1]++;
      else
        dropped++;
    }
    for (int o = 0; o < num_objects; ++o) obj_off[o + 1] += obj_off[o];

    int *const cursor = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(num_objects, 1) * sizeof(int), "ghost grouping cursors");
    for (int o = 0; o < num_objects; ++o) cursor[o] = obj_off[o];

    for (int k = 0; k < n_replies; ++k)
    {
      const int o = flat_recv[2 * k];
      if (o == MFTL_ERROR)
      {
        MFTL_ASSERT(0 && "ghost sync: request names a body this rank does not hold, but claims this rank owns the vertex");
        memset(&replies[k], 0, sizeof replies[k]);
        replies[k].global_id = MFTL_ERROR;
        continue;
      }
      order[cursor[o]++] = k;
    }
    free(cursor);
    unserved += dropped;
  }

  {
    /* The ensemble's, not a local: this table is sized by a body's id range, and
     * building and releasing one per TIMESTEP was the last per-step allocation on
     * either path. It is retired by an epoch bump inside answer_object_requests,
     * which is what the epoch was always for. */
    mftl_gid_table *const tbl = MFTLCommWorkspaceGhostTable(ens);
    for (int o = 0; o < num_objects; ++o)
    {
      if (obj_off[o + 1] == obj_off[o]) continue;  // nothing asked of this body
      answer_object_requests(objects[o], field, flat_recv, order, obj_off[o], obj_off[o + 1], replies, tbl, &unserved);
    }
  }

  free(order);
  free(obj_off);
  free(flat_recv);
  CS_END(ts_answer, g_t_answer);

  CS_BEGIN(ts_rep);

  /* ── 5. Send the replies back ─────────────────────────────────────────────
   * No extra count exchange: the reply counts are exactly the request counts
   * halved, and both sides already know both. */
  int *const n_send_rep = n_recv;  // what we were asked for, we now answer
  int *const n_recv_rep = n_send;  // what we asked for, we now receive
  for (int i = 0; i < nn; ++i)
  {
    n_send_rep[i] /= 2;
    n_recv_rep[i] /= 2;
    s_displs[i] /= 2;
    r_displs[i] /= 2;
  }

  const int n_incoming = total_send / 2;
  MFTL_GhostUpdatePayload *const incoming =
      (MFTL_GhostUpdatePayload *) MFTLMallocOrDie((size_t) MFTL_MAX(n_incoming, 1) * sizeof(MFTL_GhostUpdatePayload), "ghost updates");

  MPI_Neighbor_alltoallv(replies, n_send_rep, r_displs, MFTL_MPI_GhostUpdatePayload, incoming, n_recv_rep, s_displs, MFTL_MPI_GhostUpdatePayload,
                         domain->comm_graph);
  free(replies);
  CS_END(ts_rep, g_t_replies);
  MFTLCommStats.g_n_bytes_sent += (long) (total_recv / 2) * (long) sizeof(MFTL_GhostUpdatePayload);
  MFTLCommStats.g_n_bytes_recv += (long) (total_send / 2) * (long) sizeof(MFTL_GhostUpdatePayload);
  CS_BEGIN(ts_apply);

  /* ── 6. Apply ─────────────────────────────────────────────────────────────
   * The replies come back in exactly the order the requests went out, so the
   * targets list resolves each one directly. No hashing, no object search. */
  int p = 0;
  for (int i = 0; i < nn; ++i)
  {
    const int *const t = targets[i].data;
    const int count = targets[i].size;

    for (int k = 0; k < count; k += 2, ++p)
    {
      const MFTL_GhostUpdatePayload *const src = &incoming[p];
      if (src->global_id == MFTL_ERROR) continue;  // owner could not serve it

      MFTL_Object *const obj = objects[t[k]];
      const int idx = t[k + 1];
      MFTL_ASSERT(obj->verts.rec[idx].global_id == src->global_id && "ghost reply does not match the request order");

      double *const dst = ghost_field_write(obj, field);
      dst[3 * idx + 0] = src->value[0];
      dst[3 * idx + 1] = src->value[1];
      dst[3 * idx + 2] = src->value[2];
    }
  }

  free(incoming);
  CS_END(ts_apply, g_t_apply);
  CS_BEGIN(ts_free);
  for (int i = 0; i < nn; ++i)
  {
    ib_free(&requests[i]);
    ib_free(&targets[i]);
  }
  free(requests);
  free(targets);
  free(n_send);
  free(n_recv);
  free(s_displs);
  free(r_displs);
  CS_END(ts_free, g_t_teardown);

  CS_END(ts_total, g_t_total);

  /* An orphaned ghost kept a STALE value, so the sync did not do its job even
   * though every collective completed: report it, rather than returning success
   * over ghosts that two ranks now see differently. */
  return (unserved == 0 && orphaned == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}

int MFTLUpdateGhostPositions(MFTL_Ensemble *ens) { return ghost_sync_vec3(ens, GHOST_FIELD_POSITION); }

int MFTLUpdateGhostVertexNormals(MFTL_Ensemble *ens) { return ghost_sync_vec3(ens, GHOST_FIELD_NORMAL); }
