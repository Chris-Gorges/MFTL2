/**
 * @file ghost_sync.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Ghost synchronisation of one per-vertex 3-vector (position or normal)
 *        with the topology unchanged.
 *
 * A request/reply exchange, because only the HOLDER of a ghost knows it holds
 * it; no owner-to-holder state is kept between calls. Per ghost vertex 8 B of
 * request and 32 B of reply; three neighbourhood collectives (counts, requests,
 * replies). One field per exchange: bundling recomputable fields with the
 * position would ship bytes the receiver computes more cheaply.
 *
 * Per call it builds one `[nproc]` rank-to-neighbour lookup (O(nproc) memory
 * and time), the only term here that grows with the process count.
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

/** @brief Which per-vertex 3-vector an exchange moves. */
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
 * The owner turns a vertex global id into a pool slot through the epoch-stamped
 * id table (comm_internal.h), filled with this object's OWNED vertices only for
 * objects that actually received requests. An unanswerable request (the
 * requester believes this rank owns a vertex it does not) is answered with
 * `global_id == MFTL_ERROR`, asserted in Debug and skipped by the requester.
 *
 * @param      obj        The mesh whose OWNED vertices answer the requests.
 * @param      field      Which per-vertex 3-vector is being synchronised.
 * @param      flat_recv  The received requests (local slot, vertex id), all senders.
 * @param      order      Request indices belonging to this object.
 * @param      first      First slot of @p order to serve.
 * @param      last       One past the last slot to serve.
 * @param[out] replies    One payload per request.
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
  /* Owned vertices go in; the mode follows the share of the body held here. */
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

  /* Guards on GLOBAL state (or on a rank with no graph neighbours, which nobody
   * expects a message from): a rank holding nothing must still enter the
   * neighbourhood collectives. */
  if (ens->n_bodies_global == 0 || nn == 0 || nproc == 1) return MFTL_SUCCESS;

  CS_BEGIN(ts_total);
  MFTLCommStats.g_n_calls++;
  CS_BEGIN(ts_collect);

  /* ── 1. Collect every ghost (the pool suffix), grouped by owner: a REQUEST
   * (body id, vertex id) for the wire and a TARGET (local slot, vertex slot)
   * kept here, so replies are applied by direct index. */
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
        /* Owner outside the 26-neighbourhood: unreachable. Skipped (aborting
         * would deadlock the others) but asserted, because the stale position
         * later makes two ranks route a triangle differently. There is no global
         * fallback (unlike MFTLRemeshSyncGhostIds()); a partition whose halo
         * leaves the neighbourhood must not outlive the call that created it. */
        MFTL_ASSERT(0 && "ghost sync: ghost vertex whose owner is not a graph neighbour -- its position will not be updated");
        orphaned++;
        continue;
      }

      /* Body by global id on the wire; local slot in the target. */
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

  /* ── 2. Request counts, non-blocking and overlapped with flattening the
   * requests; everything after is a strict dependency chain. */
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

  /* ── 4. Answer: group the requests by object (counting sort) so each object's
   * id table is built once; replies go back at the request's ORIGINAL position,
   * since the requester matches them positionally. */
  const int n_replies = total_recv / 2;
  MFTL_GhostUpdatePayload *const replies =
      (MFTL_GhostUpdatePayload *) MFTLMallocOrDie((size_t) MFTL_MAX(n_replies, 1) * sizeof(MFTL_GhostUpdatePayload), "ghost replies");

  int *const obj_off = (int *) MFTLCallocOrDie((size_t) num_objects + 1, sizeof(int), "ghost request grouping");
  int *const order = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_replies, 1) * sizeof(int), "ghost request order");
  int unserved = 0;

  {
    /* Resolve each body id to this rank's slot once, in place. A body not held
     * here is a routing error: the requester named this rank as the owner. */
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
    /* The ensemble's persistent table, retired per object by an epoch bump. */
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

  /* ── 5. Replies: counts are the request counts halved, known to both sides. */
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

  /* ── 6. Apply in request order through the target list. */
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

  /* An orphaned or unserved ghost kept a stale value: report it. */
  return (unserved == 0 && orphaned == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}

int MFTLUpdateGhostPositions(MFTL_Ensemble *ens)
{
  /* New ghost positions: geometry computed before this is stale on the ghosts. */
  if (ens != MFTL_NULL && ens->local_objects != MFTL_NULL)
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      ens->local_objects[o]->derived &= ~MFTL_DERIVED_GEOMETRY;
      MFTLObjectMarkChanged(ens->local_objects[o]);
    }
  return ghost_sync_vec3(ens, GHOST_FIELD_POSITION);
}

int MFTLUpdateGhostVertexNormals(MFTL_Ensemble *ens) { return ghost_sync_vec3(ens, GHOST_FIELD_NORMAL); }
