/**
 * @file redistribute.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Redistribution: re-partitioning a distributed mesh over the process grid.
 *
 * Both public entry points run redistribute_core(); MFTLRedistribute() is
 * MFTLRedistributeWithOwners() with the geometric owner rule.
 *
 * Most of a partition does not move, so kept entities never touch the wire
 * format:
 *
 *   kept      -> a PERMUTATION of the pool: records and positions are gathered
 *                into scratch in final order and written back.
 *   arrivals  -> de-duplicated through a global-id table (gt_* in
 *                comm_internal.h) and appended.
 *   twins     -> REPAIRED, not rebuilt: a twin between two kept triangles is an
 *                index remap; only half-edges that lost their partner are
 *                re-paired.
 *
 * **Why the incremental twin repair is complete.** Kept-ness is decided per
 * triangle, so a kept half-edge's twin survives iff the twin's triangle is kept,
 * and then both are remapped consistently. A half-edge whose twin did not
 * survive can only pair with another such half-edge, so the repair set is closed
 * under the twin relation and bucketing it against itself finds every pair a
 * full MFTLRebuildTwins() would. tests/test_redistribution.c re-runs
 * MFTLRebuildTwins() after a redistribution and asserts that nothing changes.
 */

#include "mftl_communication.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Small shared helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Per-destination starting capacity of the id lists: a fifth of this
 *        rank's fair share of the mesh, clamped to [1024, 10^6].
 *
 * @param      n_tris_global  Triangles in the whole mesh.
 * @param      n_verts_global Vertices in the whole mesh.
 * @param      nproc          Ranks in the communicator.
 * @param[out] tri_hint       Starting triangle capacity per destination.
 * @param[out] vert_hint      Starting vertex capacity per destination.
 */
static void calc_buffer_hints(int n_tris_global, int n_verts_global, int nproc, int *tri_hint, int *vert_hint)
{
  /* Assume no single destination takes more than a fifth of this rank's fair
   * share; the buffers grow if that is wrong. */
  const double factor = 0.2;
  const int t = (int) (factor * (double) n_tris_global / (double) nproc);
  const int v = (int) (factor * (double) n_verts_global / (double) nproc);

  *tri_hint = MFTL_MIN(MFTL_MAX(1024, t), 1000000);
  *vert_hint = MFTL_MIN(MFTL_MAX(1024, v), 1000000);
}

/**
 * @brief Grows a send buffer geometrically, in elements of @p elem bytes.
 *
 * @param sb     The send buffer.
 * @param needed Additional elements that must fit.
 * @param elem   Size of one element in bytes.
 */
static void sb_reserve(MFTL_SendBuffer *sb, int needed, size_t elem)
{
  if (sb->count + needed <= sb->capacity) return;

  int cap = (sb->capacity == 0) ? MFTL_MAX(needed, 256) : sb->capacity;
  while (cap < sb->count + needed) cap *= 2;

  sb->buffer = MFTLReallocOrDie(sb->buffer, (size_t) cap * elem, "MFTL redistribution send buffer");
  sb->capacity = cap;
}

/**
 * @brief Grows a scratch int array, discarding its contents.
 *
 * @param p    Address of the array pointer. Replaced if it grows.
 * @param cap  Address of the capacity counter, in elements.
 * @param need Elements required.
 */
static void ensure_ints(int **p, int *cap, int need)
{
  if (need <= *cap) return;
  const int c = MFTLGrowCapacity(*cap, need);
  free(*p);
  *p = (int *) MFTLMallocOrDie((size_t) c * sizeof(int), "MFTL redistribution scratch");
  *cap = c;
}

/**
 * @brief Grows a scratch byte block, discarding its contents.
 *
 * @param p    Address of the block pointer. Replaced if it grows.
 * @param cap  Address of the capacity counter, in bytes.
 * @param need Bytes required.
 */
static void ensure_bytes(void **p, size_t *cap, size_t need)
{
  if (need <= *cap) return;
  size_t c = (*cap > 0) ? *cap : 1024;
  while (c < need) c *= 2;
  free(*p);
  *p = MFTLMallocOrDie(c, "MFTL redistribution scratch");
  *cap = c;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Phase 1 -- ownership and destinations
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Records one ownership classification pass into the diagnostics.
 *
 * @param nv        Vertices classified, ghosts included.
 * @param nt        Triangles classified, ghosts included.
 * @param changed_v Of those vertices, how many changed owner.
 * @param changed_t Of those triangles, how many changed owner.
 */
static void stats_owner_scan(int nv, int nt, long changed_v, long changed_t)
{
  MFTLCommStats.n_verts_scanned += nv;
  MFTLCommStats.n_tris_scanned += nt;
  MFTLCommStats.n_verts_owner_changed += changed_v;
  MFTLCommStats.n_tris_owner_changed += changed_t;
}

/**
 * @brief Assigns every entity (ghosts included) its NEW owner and writes it into
 *        `owner_proc`.
 *
 * Two streams over `verts.pos` and `tris.center` through MFTLGetOwnerProc(), or
 * the caller's #MFTL_OwnerMap (tested once per object, outside the loops). The
 * centre-currency assertion belongs to the geometric rule only. Each loop also
 * counts owner changes from the value it overwrites -- one compare against a
 * word it stores anyway -- and that verdict is half of the skip precondition.
 * Idempotent: a second call rewrites the same values and reports no change.
 *
 * @param      obj          The mesh.
 * @param      map          Explicit owner map, or NULL for the geometric rule.
 * @param[out] tri_owner    `[tris.n]` destination rank per triangle.
 * @param[out] vert_owner   `[verts.n]` destination rank per vertex.
 * @param[out] vert_gid     `[verts.n]` vertex global ids, gathered contiguously
 *                          for the packing loops.
 * @param      record_stats `false` on a re-classification, so counters describe
 *                          each entity once.
 * @return Non-zero if any entity of @p obj changed owner.
 */
static int compute_owners(MFTL_Object *obj, const MFTL_OwnerMap *map, int *restrict tri_owner, int *restrict vert_owner, int *restrict vert_gid,
                          bool record_stats)
{
  const MFTL_Domain *const dom = obj->domain;

  const double *const restrict pos = obj->verts.pos;
  MFTL_Vertex *const restrict vr = obj->verts.rec;
  const int nv = obj->verts.n;

  const double *const restrict ctr = obj->tris.center;
  MFTL_Triangle *const restrict tr = obj->tris.rec;
  const int nt = obj->tris.n;

  if (map != MFTL_NULL)
  {
    const int *const restrict vo = map->vert_owner;
    const int *const restrict to = map->tri_owner;
    MFTL_ASSERT(vo != MFTL_NULL && to != MFTL_NULL && "redistribute: an owner map must cover both entity classes");

    long changed_v = 0, changed_t = 0;

    for (int v = 0; v < nv; ++v)
    {
      const int owner = vo[v];
      MFTL_ASSERT(owner >= 0 && owner < dom->nproc && "redistribute: owner map names a rank outside the communicator");
      vert_owner[v] = owner;
      vert_gid[v] = vr[v].global_id;
      changed_v += (vr[v].owner_proc != owner);
      vr[v].owner_proc = owner;
    }

    for (int t = 0; t < nt; ++t)
    {
      const int owner = to[t];
      MFTL_ASSERT(owner >= 0 && owner < dom->nproc && "redistribute: owner map names a rank outside the communicator");
      tri_owner[t] = owner;
      changed_t += (tr[t].owner_proc != owner);
      tr[t].owner_proc = owner;
    }

    if (record_stats) stats_owner_scan(nv, nt, changed_v, changed_t);
    return (changed_v | changed_t) != 0;
  }

  MFTL_ASSERT(MFTLTriangleCentresAreCurrent(obj) &&
              "redistribute: stale triangle centres -- call MFTLComputeTriangleGeometry(obj, obj->tris.n) (ghosts included) first");

  long changed_v = 0, changed_t = 0;

  for (int v = 0; v < nv; ++v)
  {
    const int owner = MFTLGetOwnerProc(dom, &pos[3 * v]);
    vert_owner[v] = owner;
    vert_gid[v] = vr[v].global_id;
    /* Unconditional: the answer decides whether this body is rebuilt at all. */
    changed_v += (vr[v].owner_proc != owner);
    vr[v].owner_proc = owner;
  }

  for (int t = 0; t < nt; ++t)
  {
    const int owner = MFTLGetOwnerProc(dom, &ctr[3 * t]);
    tri_owner[t] = owner;
    changed_t += (tr[t].owner_proc != owner);
    tr[t].owner_proc = owner;
  }

  if (record_stats) stats_owner_scan(nv, nt, changed_v, changed_t);
  return (changed_v | changed_t) != 0;
}

/**
 * @struct vertex_dedup
 * @brief De-duplicates each REMOTE destination's vertex list on construction.
 *
 * A vertex is a candidate once per incident triangle bound for a destination.
 * One bit per (vertex, destination slot) makes the test-and-set O(1) instead of
 * a later sort + unique. Slots are handed out on first contact; 32 suffice on
 * the neighbour route (at most 26 remote destinations). On the global route a
 * destination beyond 32 gets #MFTL_DEDUP_NO_BIT and its list is sorted instead;
 * correctness never depends on the bound.
 */
/** @brief De-duplication bit slots available, one per remote destination. */
#define MFTL_DEDUP_BITS 32
/** @brief Slot value meaning "no bit available"; that destination falls back to a sort. */
#define MFTL_DEDUP_NO_BIT MFTL_DEDUP_BITS

typedef struct
{
  uint32_t *mask;  ///< [verts.n] zeroed per object.
  int *rank_bit;  ///< [nproc] bit slot per destination, MFTL_ERROR = unassigned.
  int n_bits;  ///< Slots handed out for the current object.
} vertex_dedup;

/**
 * @brief Destination analysis: which ranks each OWNED triangle goes to, and what
 *        this rank keeps.
 *
 * Destinations of triangle t: its owner, the owners of its three corners and
 * the owners of its three edge neighbours (the halo rule). This rank is one
 * destination like any other, recorded as keep flags instead of a copy. Ghost
 * triangles are never forwarded. Interior triangles, whose seven candidates all
 * name one rank, take a register-only fast path (an XOR-OR test); otherwise the
 * candidates are collected in first-seen order, which fixes the wire order and
 * hence the arrival slot order.
 *
 * @param      obj         The mesh.
 * @param      my_rank     This rank.
 * @param      is_neighbor `[nproc]` flags marking the 26-neighbourhood.
 * @param      tri_owner   `[tris.n]` destination rank per triangle.
 * @param      vert_owner  `[verts.n]` destination rank per vertex.
 * @param      dd          Per-destination vertex de-duplication state.
 * @param[out] keep_t      `[tris.n]` flags: triangle stays on this rank.
 * @param[out] keep_v      `[verts.n]` flags: vertex stays on this rank.
 * @param[out] tri_ids     `[nproc]` per-destination triangle index lists.
 * @param[out] vert_ids    `[nproc]` per-destination vertex index lists.
 * @param[out] touched     Destination ranks that received anything.
 * @param[out] n_touched   Number of entries written to @p touched.
 * @return 1 if any destination lies outside the 26-neighbourhood (forcing the
 *         global route).
 */
static int collect_destinations(const MFTL_Object *obj, int my_rank, const char *restrict is_neighbor, const int *restrict tri_owner,
                                const int *restrict vert_owner, vertex_dedup *dd, char *restrict keep_t, char *restrict keep_v, mftl_int_buffer *tri_ids,
                                mftl_int_buffer *vert_ids, int *touched, int *n_touched)
{
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "redistribute: tris.verts/neighbors are stale -- MFTLObjectBuildDerived first");

  const int n_owned = obj->tris.n_local;
  if (n_owned < 1) return 0;

  /* 12 B per triangle each, streaming (48 B from the half-edge records). */
  const int *const restrict tv = obj->tris.verts;
  const int *const restrict tn = obj->tris.neighbors;

  uint32_t *const restrict mask = dd->mask;
  int *const restrict rank_bit = dd->rank_bit;

  int requires_global = 0;

  for (int t = 0; t < n_owned; ++t)
  {
    const int base = 3 * t;
    const int v0 = tv[base + 0], v1 = tv[base + 1], v2 = tv[base + 2];

    /* Fast path: "is any of the seven candidates different from the owner" is
     * the OR of six XORs, answered in registers; cand[] (a variable-index
     * array that cannot live in registers) is built only when the answer is yes.
     * A missing edge neighbour contributes `r`, so its term vanishes. cand[]
     * keeps first-seen order, which fixes the wire and arrival order. */
    const int r = tri_owner[t];
    const int o0 = vert_owner[v0], o1 = vert_owner[v1], o2 = vert_owner[v2];
    const int a0 = tn[base + 0], a1 = tn[base + 1], a2 = tn[base + 2];
    const int r0 = (a0 != MFTL_ERROR) ? tri_owner[a0] : r;
    const int r1 = (a1 != MFTL_ERROR) ? tri_owner[a1] : r;
    const int r2 = (a2 != MFTL_ERROR) ? tri_owner[a2] : r;

    int dests[7];
    int n_dests = 1;
    dests[0] = r;

    if (((o0 ^ r) | (o1 ^ r) | (o2 ^ r) | (r0 ^ r) | (r1 ^ r) | (r2 ^ r)) != 0)
    {
      int cand[7];
      int n_cand = 0;
      cand[n_cand++] = r;
      cand[n_cand++] = o0;
      cand[n_cand++] = o1;
      cand[n_cand++] = o2;
      if (a0 != MFTL_ERROR) cand[n_cand++] = r0;
      if (a1 != MFTL_ERROR) cand[n_cand++] = r1;
      if (a2 != MFTL_ERROR) cand[n_cand++] = r2;
      n_dests = MFTLGetUniqueIntegers(cand, n_cand, dests, 7);
    }

    for (int k = 0; k < n_dests; ++k)
    {
      const int dst = dests[k];

      if (dst == my_rank)
      {
        keep_t[t] = 1;
        keep_v[v0] = 1;
        keep_v[v1] = 1;
        keep_v[v2] = 1;
        continue;
      }

      if (!is_neighbor[dst]) requires_global = 1;

      int b = rank_bit[dst];
      if (b == MFTL_ERROR)
      {
        touched[(*n_touched)++] = dst;  // first contact and first triangle coincide
        MFTL_ASSERT(tri_ids[dst].size == 0 && "collect_destinations: bit slot and buffer state disagree");
        b = (dd->n_bits < MFTL_DEDUP_BITS) ? dd->n_bits++ : MFTL_DEDUP_NO_BIT;
        if (b == MFTL_DEDUP_NO_BIT) MFTLCommStats.n_dedup_overflow++;
        rank_bit[dst] = b;
      }

      ib_add(&tri_ids[dst], t);

      if (b == MFTL_DEDUP_NO_BIT)
      {
        ib_add3(&vert_ids[dst], v0, v1, v2);  // append_destination will sort this one
        continue;
      }

      const uint32_t m = (uint32_t) 1 << b;
      if (!(mask[v0] & m))
      {
        mask[v0] |= m;
        ib_add(&vert_ids[dst], v0);
      }
      if (!(mask[v1] & m))
      {
        mask[v1] |= m;
        ib_add(&vert_ids[dst], v1);
      }
      if (!(mask[v2] & m))
      {
        mask[v2] |= m;
        ib_add(&vert_ids[dst], v2);
      }
    }
  }

  return requires_global;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Packing
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Packs one triangle and its three half-edges into a transfer unit: ten
 *        ints, all of them information the receiver cannot derive.
 *
 * @param obj      The mesh.
 * @param t        Triangle index.
 * @param tv       `obj->tris.verts`.
 * @param vgid     `[verts.n]` vertex global ids, contiguous, from compute_owners().
 * @param[out] dst The wire record.
 * @param body_gid #MFTL_Object::global_id of the owning body.
 */
static inline void pack_triangle(const MFTL_Object *obj, int t, const int *restrict tv, const int *restrict vgid, MFTL_PackedTriangleComplex *dst, int body_gid)
{
  const MFTL_Triangle *const src = &obj->tris.rec[t];
  const MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
  const int base = 3 * t;

  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->flags = src->flags;
  dst->body_gid = body_gid;

  /* tris.verts[3t+k] is the ORIGIN of half-edge k; the wire carries its TARGET,
   * target(k) == origin((k+1)%3). */
  dst->vertex_gid[0] = vgid[tv[base + 1]];
  dst->vertex_gid[1] = vgid[tv[base + 2]];
  dst->vertex_gid[2] = vgid[tv[base + 0]];

  dst->he_flags[0] = he[0].flags;
  dst->he_flags[1] = he[1].flags;
  dst->he_flags[2] = he[2].flags;

  MFTL_ASSERT(dst->vertex_gid[0] == obj->verts.rec[he[0].vertex_idx].global_id && "pack_triangle: half-edge target rotation is wrong");
  MFTL_ASSERT(he[0].global_id == 3 * src->global_id && he[1].global_id == 3 * src->global_id + 1 && he[2].global_id == 3 * src->global_id + 2 &&
              "pack_triangle: half-edge global ids are not 3*triangle_gid + k -- the receiver reconstructs them from that");
}

/**
 * @brief Packs one vertex: its position and its identity, nothing else.
 * @param obj      The mesh.
 * @param v        Vertex index.
 * @param[out] dst The wire record.
 * @param body_gid #MFTL_Object::global_id of the owning body.
 */
static inline void pack_vertex(const MFTL_Object *obj, int v, MFTL_PackedVertex *dst, int body_gid)
{
  const MFTL_Vertex *const src = &obj->verts.rec[v];
  const double *const p = &obj->verts.pos[3 * v];

  dst->pos[0] = p[0];
  dst->pos[1] = p[1];
  dst->pos[2] = p[2];
  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->flags = src->flags;
  dst->body_gid = body_gid;
}

#ifndef NDEBUG
/**
 * @brief Debug-only: is the array strictly increasing (sorted and duplicate-free)?
 *
 * @param a Array to inspect.
 * @param n Number of elements.
 * @return `true` if `a[i] < a[i+1]` throughout.
 */
static bool is_strictly_increasing(const int *a, int n)
{
  for (int i = 1; i < n; ++i)
    if (a[i] <= a[i - 1]) return MFTL_FALSE;
  return MFTL_TRUE;
}
#endif

/**
 * @brief Packs one object's contribution for one REMOTE destination rank,
 *        appending to that destination's send buffers.
 *
 * Triangle lists need no de-duplication (each triangle is appended at most once
 * per distinct destination, in ascending order); vertex lists do only for a
 * destination without a de-duplication bit.
 *
 * @param obj         The mesh.
 * @param body_gid    #MFTL_Object::global_id of the owning body.
 * @param vgid        `[verts.n]` vertex global ids.
 * @param tri_ids     Triangle indices bound for this destination.
 * @param vert_ids    Vertex indices bound for this destination.
 * @param out_tris    Send buffer the packed triangles are appended to.
 * @param out_verts   Send buffer the packed vertices are appended to.
 * @param needs_dedup `true` when the vertex list must be sorted and uniqued here.
 */
static void append_destination(const MFTL_Object *obj, int body_gid, const int *restrict vgid, mftl_int_buffer *tri_ids, mftl_int_buffer *vert_ids,
                               MFTL_SendBuffer *out_tris, MFTL_SendBuffer *out_verts, bool needs_dedup)
{
  /* Already strictly increasing: see collect_destinations(). */
  MFTL_ASSERT(is_strictly_increasing(tri_ids->data, tri_ids->size) && "append_destination: triangle list is not already unique");

  if (tri_ids->size > 0)
  {
    sb_reserve(out_tris, tri_ids->size, sizeof(MFTL_PackedTriangleComplex));
    MFTL_PackedTriangleComplex *const buf = (MFTL_PackedTriangleComplex *) out_tris->buffer;
    const int *const restrict tv = obj->tris.verts;
    for (int i = 0; i < tri_ids->size; ++i) pack_triangle(obj, tri_ids->data[i], tv, vgid, &buf[out_tris->count++], body_gid);
  }

  if (vert_ids->size > 0)
  {
    int n = vert_ids->size;
    if (needs_dedup)
    {
      MFTLSortIds(vert_ids->data, n);
      n = MFTLUniqueIntArray(vert_ids->data, n);
    }

    sb_reserve(out_verts, n, sizeof(MFTL_PackedVertex));
    MFTL_PackedVertex *const buf = (MFTL_PackedVertex *) out_verts->buffer;
    for (int i = 0; i < n; ++i) pack_vertex(obj, vert_ids->data[i], &buf[out_verts->count++], body_gid);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Per-object state carried across the exchange
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct body_state
 * @brief What the analysis phase hands to the rebuild phase, per object.
 *
 * The kept lists are OLD pool indices in final relative order: the owned run
 * first, then the ghost run. Both runs are ascending, which keeps the gather in
 * the rebuild a forward scan over the pool rather than a random walk.
 */
typedef struct
{
  int *kept_v;  ///< [n_kept_v] old vertex indices (arena slice).
  int n_kept_v_owned;  ///< Kept vertices this rank OWNS; they lead the rebuilt prefix.
  int n_kept_v;  ///< Kept vertices in total, owned plus retained ghosts.

  int *kept_t;  ///< [n_kept_t] old triangle indices (arena slice).
  int n_kept_t_owned;  ///< Kept triangles this rank OWNS.
  int n_kept_t;  ///< Kept triangles in total.

  /* Arrival slices: indices into the flat receive buffers, grouped by body. */
  const int *recv_v_idx;  ///< [n_recv_v] arriving vertex slots.
  int n_recv_v;  ///< Vertices arriving for this body.
  const int *recv_t_idx;  ///< [n_recv_t] arriving triangle slots.
  int n_recv_t;  ///< Triangles arriving for this body.
} body_state;

/**
 * @brief Splits one entity class's keep flags into the owned run and the ghost
 *        run, both ascending, in ONE pass.
 *
 * The owned run is filled forward from 0 and the ghost run backward from the
 * end of the body's arena slice (sized for the whole class, so the cursors
 * cannot meet); the ghost run is then reversed and moved down. O(halo) fix-up,
 * byte-identical to two separate scans. Ownership is read from the dense
 * `owner` array written by compute_owners() (4 B per entity) rather than from
 * the AoS records.
 *
 * @param      keep    `[n]` keep flags from collect_destinations().
 * @param      owner   `[n]` destination rank per entity.
 * @param      n       Entities: `verts.n`, or `tris.n_local` (only owned
 *                     triangles can be kept).
 * @param      my_rank This rank.
 * @param[out] out     `[n]` kept indices: owned run then ghost run.
 * @param[out] n_owned Length of the owned run.
 * @return Kept entities in total.
 */
static int split_kept_run(const char *restrict keep, const int *restrict owner, int n, int my_rank, int *restrict out, int *n_owned)
{
  int lo = 0, hi = n;

  for (int i = 0; i < n; ++i)
  {
    if (!keep[i]) continue;
    if (owner[i] == my_rank)
      out[lo++] = i;
    else
      out[--hi] = i;
  }

  *n_owned = lo;
  const int n_ghost = n - hi;

  for (int a = 0, b = n_ghost - 1; a < b; ++a, --b)
  {
    const int tmp = out[hi + a];
    out[hi + a] = out[hi + b];
    out[hi + b] = tmp;
  }
  if (n_ghost > 0 && hi != lo) memmove(out + lo, out + hi, (size_t) n_ghost * sizeof(int));

  return lo + n_ghost;
}

/**
 * @brief Splits the keep flags into the two ascending runs the rebuild wants.
 *
 * @param      obj        The mesh, for the two class sizes.
 * @param      keep_v     `[verts.n]` flags from collect_destinations().
 * @param      keep_t     `[tris.n_local]` flags from collect_destinations().
 * @param      vert_owner `[verts.n]` destination rank per vertex.
 * @param      tri_owner  `[tris.n]` destination rank per triangle.
 * @param      my_rank    This rank.
 * @param[out] bs         Per-body state; its kept lists and counts are filled in.
 */
static void build_kept_lists(const MFTL_Object *obj, const char *restrict keep_v, const char *restrict keep_t, const int *restrict vert_owner,
                             const int *restrict tri_owner, int my_rank, body_state *bs)
{
  bs->n_kept_v = split_kept_run(keep_v, vert_owner, obj->verts.n, my_rank, bs->kept_v, &bs->n_kept_v_owned);
  bs->n_kept_t = split_kept_run(keep_t, tri_owner, obj->tris.n_local, my_rank, bs->kept_t, &bs->n_kept_t_owned);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Rebuild
 *
 * Slots cannot be assigned before the arrivals are de-duplicated, so the rebuild
 * first works in a LOGICAL index space -- `[0, n_kept)` kept (owned run, then
 * ghost run), `[n_kept, n_kept + n_new)` distinct arrivals in arrival order --
 * and `final_of_logical[]` maps it to the final pool layout
 *
 *     [ kept owned | new owned | kept ghost | new ghost ]
 *
 * in which `n_local` remains a prefix.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Scratch reused by every object's rebuild; grown, never shrunk. */
typedef struct
{
  void *v_rec;  ///< kept vertex records
  size_t v_rec_cap;  ///< Its allocated size in bytes.
  void *v_pos;  ///< kept vertex positions
  size_t v_pos_cap;  ///< Its allocated size in bytes.
  void *t_rec;  ///< kept triangle records
  size_t t_rec_cap;  ///< Its allocated size in bytes.
  void *t_he;  ///< kept half-edge triples
  size_t t_he_cap;  ///< Its allocated size in bytes.

  int *v_final_of_old;  ///< Old vertex index -> final slot.
  int v_final_of_old_cap;  ///< Its allocated length.
  int *t_final_of_old;  ///< Old triangle index -> final slot.
  int t_final_of_old_cap;  ///< Its allocated length.
  int *v_final_of_logical;  ///< Logical (kept-then-arrived) vertex index -> final slot.
  int v_final_of_logical_cap;  ///< Its allocated length.
  int *t_final_of_logical;  ///< Logical triangle index -> final slot.
  int t_final_of_logical_cap;  ///< Its allocated length.
  int *recv_v_logical;  ///< Arriving vertex slot -> logical index.
  int recv_v_logical_cap;  ///< Its allocated length.
  int *recv_t_logical;  ///< Arriving triangle slot -> logical index.
  int recv_t_logical_cap;  ///< Its allocated length.
  int *v_head;  ///< twin-repair buckets, indexed by vertex
  int v_head_cap;  ///< Its allocated length.
  int *u_next;  ///< twin-repair chain, indexed by position in the repair list
  int u_next_cap;  ///< Its allocated length.

  mftl_int_buffer u_list;  ///< half-edges that lost their twin
  mftl_gid_table v_tbl;  ///< Vertex global id -> slot, for de-duplicating arrivals.
  mftl_gid_table t_tbl;  ///< Triangle global id -> slot.
} rebuild_scratch;

/**
 * @struct body_run
 * @brief One contiguous run of one body's entities inside the arrival stream.
 *
 * Every rank packs body by body, and the (neighbour) alltoallv concatenates the
 * received blocks in source order, so the arrivals are a sequence of per-body
 * runs whose lengths are shipped in the header payload. A running offset over
 * the headers recovers the runs without sorting the arrivals.
 */
typedef struct
{
  int obj_idx;  ///< Local slot of the body, MFTL_ERROR if unknown. See slot_of_body().
  int tri_start;  ///< First arriving triangle of this run in the flat receive buffer.
  int tri_len;  ///< Triangles in the run.
  int vert_start;  ///< First arriving vertex of this run.
  int vert_len;  ///< Vertices in the run.
} body_run;

/**
 * @brief Resolves a body id off the wire to this call's work-list slot.
 *
 * The one place a wire body id becomes a local index. The work list is sorted
 * for its first @p n_sorted entries (the ensemble's directory) and followed by
 * the bodies arriving on this rank for the first time; binary search, then a
 * scan of the short tail.
 *
 * @param work     The work list: sorted prefix, then arrivals.
 * @param n_sorted Length of the sorted prefix.
 * @param n_work   Total entries in @p work.
 * @param gid      Body id off the wire.
 * @return The slot, or #MFTL_ERROR if this call has no such body yet.
 */
static inline int slot_of_body(MFTL_Object *const *work, int n_sorted, int n_work, int gid)
{
  /* Binary search over the part that came from the sorted directory. */
  int lo = 0, hi = n_sorted - 1;
  while (lo <= hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    const int g = work[mid]->global_id;
    if (g == gid) return mid;
    if (g < gid)
      lo = mid + 1;
    else
      hi = mid - 1;
  }

  /* Then the (short, unsorted) tail of first-time arrivals. */
  for (int i = n_sorted; i < n_work; ++i)
    if (work[i]->global_id == gid) return i;

  return MFTL_ERROR;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5b. The persistent workspace
 *
 * All scratch a redistribution needs, kept on the MFTL_Ensemble between calls
 * and only ever grown, so a steady loop allocates nothing after its first few
 * steps (the per-call allocation it replaced requested 20-39 MB per step at
 * four ranks and re-faulted those pages every time). It lives on the ensemble,
 * not the domain, because the domain is read-only during a redistribution.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Per-call scratch, kept between calls. Grown, never shrunk. */
typedef struct
{
  int nproc;  ///< What the rank-sized arrays below were built for. 0 means "none yet".
  char *is_neighbor;  ///< [nproc] 1 for a graph neighbour.
  mftl_int_buffer *tri_ids;  ///< [nproc] triangles bound for each destination.
  mftl_int_buffer *vert_ids;  ///< [nproc] and their vertices.
  MFTL_SendBuffer *send_tris;  ///< [nproc] packed triangle payloads.
  MFTL_SendBuffer *send_verts;  ///< [nproc] packed vertex payloads.
  MFTL_SendBuffer *send_counts;  ///< [nproc] per-body run lengths, multi-body only.
  int *touched;  ///< [nproc] destinations this body reached.
  int *rank_bit;  ///< [nproc] de-duplication bit handed to each destination.

  int cap_objects;  ///< Bodies the per-body arrays are sized for.

  /**
   * The bodies THIS CALL touches: those this rank held, then first-time
   * arrivals appended. Arrivals are merged into the sorted directory only at the
   * end, because an ordered insert would shift the slots `states`, the arrival
   * CSR and `runs[k].obj_idx` are indexed by.
   */
  MFTL_Object **work;  ///< [cap_objects]
  body_state *states;  ///< [cap_objects]
  int *grp_off_v;  ///< [cap_objects + 1] arrival CSR offsets, vertices.
  int *grp_off_t;  ///< [cap_objects + 1] and triangles.
  int *cursor;  ///< [cap_objects] counting-sort cursors.

  /**
   * Per-body skip votes by LOCAL SLOT (this rank's votes, then the reduced
   * copy), reduced over each body's holders by MFTLEnsembleReduceBodies().
   */
  int cap_votes;  ///< Bodies the vote arrays are sized for.
  double *dirty_votes;  ///< [2 * cap_votes]: this rank's votes, then the reduction.

  int cap_arena_v;  ///< Vertices the kept-list arena holds.
  int *kept_arena_v;  ///< [cap_arena_v]
  int cap_arena_t;  ///< Triangles the kept-list arena holds.
  int *kept_arena_t;  ///< [cap_arena_t]

  int cap_verts;  ///< Largest body's vertex count seen so far.
  int *vert_owner;  ///< [cap_verts]
  int *vert_gid;  ///< [cap_verts]
  char *keep_v;  ///< [cap_verts]
  uint32_t *dedup_mask;  ///< [cap_verts]

  int cap_tris;  ///< Largest body's triangle count seen so far.
  int *tri_owner;  ///< [cap_tris]
  char *keep_t;  ///< [cap_tris]

  int cap_runs;  ///< Arrival runs the run table holds.
  body_run *runs;  ///< [cap_runs]
  int cap_gidx_v;  ///< Arrival indices the vertex CSR holds.
  int *grp_idx_v;  ///< [cap_gidx_v]
  int cap_gidx_t;  ///< Arrival indices the triangle CSR holds.
  int *grp_idx_t;  ///< [cap_gidx_t]

  rebuild_scratch scratch;  ///< The rebuild's own arrays and id tables.

  /**
   * The ghost sync's owner-side id table, kept here so that a table sized by a
   * body's id range is not allocated and freed every timestep.
   */
  mftl_gid_table ghost_tbl;
} comm_workspace;

/**
 * @brief Releases the arrays indexed by destination rank, including the heap each
 *        per-destination buffer owns.
 * @param ws The workspace. Its `[nproc]` arrays are freed; `nproc` is left as is.
 */
static void ws_free_rank_arrays(comm_workspace *ws)
{
  for (int r = 0; r < ws->nproc; ++r)
  {
    ib_free(&ws->tri_ids[r]);
    ib_free(&ws->vert_ids[r]);
    free(ws->send_tris[r].buffer);
    free(ws->send_verts[r].buffer);
    free(ws->send_counts[r].buffer);
  }
  free(ws->is_neighbor);
  free(ws->tri_ids);
  free(ws->vert_ids);
  free(ws->send_tris);
  free(ws->send_verts);
  free(ws->send_counts);
  free(ws->touched);
  free(ws->rank_bit);
}

/** @brief Grows a byte array, discarding its contents. */
static void ws_bytes(void **p, int *cap, int need, size_t elem, const char *what)
{
  if (need <= *cap) return;
  const int c = MFTLGrowCapacity(*cap, need);
  free(*p);
  *p = MFTLMallocOrDie((size_t) c * elem, what);
  *cap = c;
  cs_alloc((size_t) c * elem);
}

/**
 * @brief Grows the per-body arrays, PRESERVING `states` (it must survive growth
 *        within a call when arrivals extend the work list) and zeroing its tail.
 *
 * @param ws The workspace.
 * @param n  Bodies the arrays must hold.
 */
static void ws_bodies_reserve(comm_workspace *ws, int n)
{
  if (n <= ws->cap_objects) return;

  const int c = MFTLGrowCapacity(ws->cap_objects, n);
  const int had = ws->cap_objects;

  MFTL_Object **const work = (MFTL_Object **) MFTLMallocOrDie((size_t) c * sizeof(MFTL_Object *), "redistribution work list");
  body_state *const states = (body_state *) MFTLCallocOrDie((size_t) c, sizeof(body_state), "redistribution body states");

  if (had > 0 && ws->states != MFTL_NULL) memcpy(states, ws->states, (size_t) had * sizeof(body_state));

  free(ws->work);
  free(ws->states);
  free(ws->grp_off_v);
  free(ws->grp_off_t);
  free(ws->cursor);

  ws->work = work;
  ws->states = states;
  ws->grp_off_v = (int *) MFTLMallocOrDie((size_t) (c + 1) * sizeof(int), "redistribution group offsets");
  ws->grp_off_t = (int *) MFTLMallocOrDie((size_t) (c + 1) * sizeof(int), "redistribution group offsets");
  ws->cursor = (int *) MFTLMallocOrDie((size_t) c * sizeof(int), "redistribution run cursors");
  ws->cap_objects = c;
  cs_alloc((size_t) c * (sizeof(MFTL_Object *) + sizeof(body_state) + 3 * sizeof(int)) + 2 * sizeof(int));
}

/**
 * @brief Sizes the workspace for this call; every array only grows. The
 *        `[nproc]` block is built on the first call only.
 *
 * @param ws           The workspace.
 * @param nproc        Ranks in the communicator.
 * @param num_objects  Bodies this rank holds, before any arrival.
 * @param sum_verts    Total vertices over all bodies, for the kept-list arena.
 * @param sum_tris     Total owned triangles over all bodies.
 * @param max_verts    Largest body's vertex count.
 * @param max_tris     Largest body's triangle count.
 * @param tri_hint     First-touch size for a destination's triangle id list.
 * @param vert_hint    Ditto for vertices.
 */
static void ws_reserve(comm_workspace *ws, int nproc, int num_objects, int sum_verts, int sum_tris, int max_verts, int max_tris, int tri_hint, int vert_hint)
{
  if (ws->nproc != nproc)
  {
    /* Only on the first call for this ensemble, when nothing is allocated yet. */
    ws_free_rank_arrays(ws);

    ws->is_neighbor = (char *) MFTLCallocOrDie((size_t) nproc, 1, "redistribution neighbour map");
    ws->tri_ids = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nproc * sizeof(mftl_int_buffer), "redistribution id lists");
    ws->vert_ids = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nproc * sizeof(mftl_int_buffer), "redistribution id lists");
    ws->send_tris = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution send buffers");
    ws->send_verts = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution send buffers");
    ws->send_counts = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution body counts");
    ws->touched = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "redistribution touched ranks");
    ws->rank_bit = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "redistribution dedup bit slots");
    for (int r = 0; r < nproc; ++r)
    {
      ib_init(&ws->tri_ids[r], tri_hint);
      ib_init(&ws->vert_ids[r], vert_hint);
      ws->rank_bit[r] = MFTL_ERROR;
    }
    ws->nproc = nproc;
    cs_alloc((size_t) nproc * (1 + 2 * sizeof(mftl_int_buffer) + 3 * sizeof(MFTL_SendBuffer) + 2 * sizeof(int)));
  }
  else
  {
    /* The hint is derived from the mesh density, which changes as the mesh does.
     * Only untouched lists can adopt a new one; a grown list keeps its capacity. */
    for (int r = 0; r < nproc; ++r)
    {
      ws->tri_ids[r].hint = tri_hint;
      ws->vert_ids[r].hint = vert_hint;
    }
  }

  ws_bodies_reserve(ws, num_objects);

  ws_bytes((void **) &ws->kept_arena_v, &ws->cap_arena_v, MFTL_MAX(sum_verts, 1), sizeof(int), "redistribution kept vertex lists");
  ws_bytes((void **) &ws->kept_arena_t, &ws->cap_arena_t, MFTL_MAX(sum_tris, 1), sizeof(int), "redistribution kept triangle lists");

  /* One block, two slices (local votes, reduced votes), sized by bodies held. */
  ws_bytes((void **) &ws->dirty_votes, &ws->cap_votes, MFTL_MAX(num_objects, 1), 2 * sizeof(double), "redistribution skip votes");

  if (max_verts > ws->cap_verts)
  {
    const int c = MFTLGrowCapacity(ws->cap_verts, max_verts);
    free(ws->vert_owner);
    free(ws->vert_gid);
    free(ws->keep_v);
    free(ws->dedup_mask);
    ws->vert_owner = (int *) MFTLMallocOrDie((size_t) c * sizeof(int), "redistribution vertex owners");
    ws->vert_gid = (int *) MFTLMallocOrDie((size_t) c * sizeof(int), "redistribution vertex gids");
    ws->keep_v = (char *) MFTLMallocOrDie((size_t) c, "redistribution vertex keep flags");
    ws->dedup_mask = (uint32_t *) MFTLMallocOrDie((size_t) c * sizeof(uint32_t), "redistribution vertex dedup mask");
    ws->cap_verts = c;
    cs_alloc((size_t) c * (2 * sizeof(int) + 1 + sizeof(uint32_t)));
  }

  if (max_tris > ws->cap_tris)
  {
    const int c = MFTLGrowCapacity(ws->cap_tris, max_tris);
    free(ws->tri_owner);
    free(ws->keep_t);
    ws->tri_owner = (int *) MFTLMallocOrDie((size_t) c * sizeof(int), "redistribution triangle owners");
    ws->keep_t = (char *) MFTLMallocOrDie((size_t) c, "redistribution triangle keep flags");
    ws->cap_tris = c;
    cs_alloc((size_t) c * (sizeof(int) + 1));
  }

  /* Footprint by what it scales with, read off the capacities. The payload
   * buffers are not counted per rank: they are only touched for destinations
   * that received something. */
  MFTLCommStats.ws_bytes_per_rank = (long) ((size_t) ws->nproc * (1 + 2 * sizeof(mftl_int_buffer) + 3 * sizeof(MFTL_SendBuffer) + 2 * sizeof(int)));

  /* Every per-body array is sized by bodies HELD, so this should stay flat in a
   * weak-scaling sweep. */
  MFTLCommStats.ws_bytes_per_body = (long) ((size_t) ws->cap_objects * (sizeof(MFTL_Object *) + sizeof(body_state) + 3 * sizeof(int)) + 2 * sizeof(int) +
                                            (size_t) ws->cap_votes * 2 * sizeof(double));

  MFTLCommStats.ws_bytes_per_entity = (long) ((size_t) ws->cap_arena_v * sizeof(int) + (size_t) ws->cap_arena_t * sizeof(int) +
                                              (size_t) ws->cap_verts * (2 * sizeof(int) + 1 + sizeof(uint32_t)) + (size_t) ws->cap_tris * (sizeof(int) + 1));
}

/** @brief Releases everything the workspace holds. */
static void ws_free(comm_workspace *ws)
{
  ws_free_rank_arrays(ws);
  free(ws->work);
  free(ws->states);
  free(ws->grp_off_v);
  free(ws->grp_off_t);
  free(ws->cursor);
  free(ws->kept_arena_v);
  free(ws->kept_arena_t);
  free(ws->dirty_votes);
  free(ws->vert_owner);
  free(ws->vert_gid);
  free(ws->keep_v);
  free(ws->dedup_mask);
  free(ws->tri_owner);
  free(ws->keep_t);
  free(ws->runs);
  free(ws->grp_idx_v);
  free(ws->grp_idx_t);

  rebuild_scratch *const s = &ws->scratch;
  free(s->v_rec);
  free(s->v_pos);
  free(s->t_rec);
  free(s->t_he);
  free(s->v_final_of_old);
  free(s->t_final_of_old);
  free(s->v_final_of_logical);
  free(s->t_final_of_logical);
  free(s->recv_v_logical);
  free(s->recv_t_logical);
  free(s->v_head);
  free(s->u_next);
  ib_free(&s->u_list);
  gt_free(&s->v_tbl);
  gt_free(&s->t_tbl);
  gt_free(&ws->ghost_tbl);

  memset(ws, 0, sizeof *ws);
}

mftl_gid_table *MFTLCommWorkspaceGhostTable(MFTL_Ensemble *ens)
{
  if (ens->comm_workspace == MFTL_NULL) ens->comm_workspace = MFTLCallocOrDie(1, sizeof(comm_workspace), "MFTL communication workspace");
  return &((comm_workspace *) ens->comm_workspace)->ghost_tbl;
}

void MFTLCommWorkspaceDestroy(void **p_ws)
{
  if (p_ws == MFTL_NULL || *p_ws == MFTL_NULL) return;
  comm_workspace *const ws = (comm_workspace *) *p_ws;
  ws_free(ws);
  free(ws);
  *p_ws = MFTL_NULL;
}

/**
 * @brief Re-pairs exactly the half-edges that lost their twin.
 *
 * The bucketing of MFTLRebuildTwins() restricted to the repair set (closed under
 * the twin relation, see the file comment); only unlinked candidates are paired,
 * keeping the relation symmetric on non-manifold input. The matching
 * `tris.neighbors` entries are written with each linked pair.
 *
 * @param obj    The mesh being rebuilt.
 * @param u_list Candidate half-edges whose twin is unresolved.
 * @param n_u    Number of entries in @p u_list.
 * @param v_head `[verts.n]` per-vertex bucket heads, scratch.
 * @param u_next `[n_u]` bucket chain links, scratch.
 */
static void repair_twins(MFTL_Object *obj, const int *restrict u_list, int n_u, int *restrict v_head, int *restrict u_next)
{
  MFTL_HalfEdge *const he = obj->tris.he;
  int *const restrict tn = obj->tris.neighbors;

  for (int j = 0; j < n_u; ++j) v_head[he[MFTLHePrevIdx(u_list[j])].vertex_idx] = MFTL_ERROR;

  for (int j = 0; j < n_u; ++j)
  {
    const int origin = he[MFTLHePrevIdx(u_list[j])].vertex_idx;
    u_next[j] = v_head[origin];
    v_head[origin] = j;
  }

  for (int j = 0; j < n_u; ++j)
  {
    const int h = u_list[j];
    if (he[h].twin_idx != MFTL_ERROR) continue;  // already linked by its partner

    const int u = he[MFTLHePrevIdx(h)].vertex_idx;
    const int v = he[h].vertex_idx;

    for (int c = v_head[v]; c != MFTL_ERROR; c = u_next[c])
    {
      const int h2 = u_list[c];
      if (he[h2].vertex_idx != u || he[h2].twin_idx != MFTL_ERROR) continue;

      he[h].twin_idx = h2;
      he[h2].twin_idx = h;
      tn[h] = MFTLHeTriIdx(h2);
      tn[h2] = MFTLHeTriIdx(h);
      break;
    }
  }
}

/**
 * @brief Rebuilds one object from its kept entities and its arrivals (no sort,
 *        no full twin rebuild).
 *
 * Steps: (0) a body with nothing kept and nothing arriving is emptied and its
 * pools released; (1) gather kept records and positions; (2) de-duplicate the
 * arrivals by global id (a vertex arrives once per sender sharing it, and the
 * copies are identical; a triangle has one owner and cannot arrive twice);
 * (3) map logical to final slots; (4) size the pools (geometric pre-growth, so
 * jittering counts do not reallocate every step) and place the kept entities;
 * (5-6) remap kept half-edges, retiring twins that did not survive into the
 * repair list, and emit `out_he_idx` and the derived arrays inline; (7) write
 * the arrivals; (8) repair twins.
 *
 * @param obj     The mesh, rewritten in place. On return: valid topology and
 *                positions, #MFTL_DERIVED_TOPOLOGY set, cached geometry
 *                unspecified, `partition_dirty` cleared.
 * @param bs      Per-body state from build_kept_lists().
 * @param recv_v  Received vertex records.
 * @param recv_t  Received triangle records.
 * @param my_rank This rank.
 * @param s       Reusable scratch, grown monotonically across calls.
 */
static void rebuild_object(MFTL_Object *obj, const body_state *bs, const MFTL_PackedVertex *recv_v, const MFTL_PackedTriangleComplex *recv_t, int my_rank,
                           rebuild_scratch *s)
{
  const int n_old_v = obj->verts.n;
  const int n_old_t = obj->tris.n;

  const int n_kept_v = bs->n_kept_v;
  const int n_kept_t = bs->n_kept_t;

  /* ── 0. Bodies this rank no longer has any part of: emptied here, destroyed
   * at the end of the call. */
  if (n_kept_v == 0 && n_kept_t == 0 && bs->n_recv_v == 0 && bs->n_recv_t == 0)
  {
    MFTLCommStats.n_objects_empty_local++;

    if (n_old_v != 0 || n_old_t != 0)
    {
      /* It has left entirely: release its pools (after generation this recovers
       * most of rank 0's footprint -- measured 80 MB -> 33.5 MB at 256 bodies,
       * refine 3, np = 4). */
      MFTLObjectSetCounts(obj, 0, 0, 0, 0);
      MFTLObjectReleasePools(obj);
    }

    /* An empty mesh is trivially described. */
    obj->derived = MFTL_DERIVED_TOPOLOGY;
    /* This call is the rebuild the dirty stamp asked for. */
    obj->partition_dirty = MFTL_FALSE;
    return;
  }

  CS_TIMELINE(tm);
  MFTLCommStats.n_verts_kept += n_kept_v;
  MFTLCommStats.n_tris_kept += n_kept_t;
  MFTLCommStats.n_verts_received += bs->n_recv_v;
  MFTLCommStats.n_tris_received += bs->n_recv_t;

  /* ── 1. Gather the kept records and positions (geometry is not copied: it is
   * unspecified after a redistribution). Also finds the largest kept id, which
   * gt_begin() needs to choose its indexing mode. */
  ensure_bytes(&s->v_rec, &s->v_rec_cap, (size_t) MFTL_MAX(n_kept_v, 1) * sizeof(MFTL_Vertex));
  ensure_bytes(&s->v_pos, &s->v_pos_cap, (size_t) MFTL_MAX(n_kept_v, 1) * 3 * sizeof(double));
  ensure_bytes(&s->t_rec, &s->t_rec_cap, (size_t) MFTL_MAX(n_kept_t, 1) * sizeof(MFTL_Triangle));
  ensure_bytes(&s->t_he, &s->t_he_cap, (size_t) MFTL_MAX(n_kept_t, 1) * 3 * sizeof(MFTL_HalfEdge));

  MFTL_Vertex *const kv_rec = (MFTL_Vertex *) s->v_rec;
  double *const kv_pos = (double *) s->v_pos;
  MFTL_Triangle *const kt_rec = (MFTL_Triangle *) s->t_rec;
  MFTL_HalfEdge *const kt_he = (MFTL_HalfEdge *) s->t_he;

  int v_range = obj->next_vertex_gid;
  int t_range = obj->next_triangle_gid;

  for (int i = 0; i < n_kept_v; ++i)
  {
    const int o = bs->kept_v[i];
    kv_rec[i] = obj->verts.rec[o];
    if (kv_rec[i].global_id >= v_range) v_range = kv_rec[i].global_id + 1;
    /* Re-derived below; the old value indexes the old pool. */
    kv_rec[i].out_he_idx = MFTL_ERROR;
    kv_pos[3 * i + 0] = obj->verts.pos[3 * o + 0];
    kv_pos[3 * i + 1] = obj->verts.pos[3 * o + 1];
    kv_pos[3 * i + 2] = obj->verts.pos[3 * o + 2];
  }

  for (int i = 0; i < n_kept_t; ++i)
  {
    const int o = bs->kept_t[i];
    kt_rec[i] = obj->tris.rec[o];
    kt_he[3 * i + 0] = obj->tris.he[3 * o + 0];
    kt_he[3 * i + 1] = obj->tris.he[3 * o + 1];
    kt_he[3 * i + 2] = obj->tris.he[3 * o + 2];
    if (kt_rec[i].global_id >= t_range) t_range = kt_rec[i].global_id + 1;
  }

  CS_LAP(tm, t_kept_gather);

  /* ── 2. Classify the arrivals against the kept entities ──────────────────── */

  /* Entry count and id range decide the table's indexing mode (comm_internal.h). */
  for (int i = 0; i < bs->n_recv_v; ++i)
  {
    const int g = recv_v[bs->recv_v_idx[i]].global_id;
    if (g >= v_range) v_range = g + 1;
  }
  for (int i = 0; i < bs->n_recv_t; ++i)
  {
    const int g = recv_t[bs->recv_t_idx[i]].global_id;
    if (g >= t_range) t_range = g + 1;
  }

  gt_begin(&s->v_tbl, n_kept_v + bs->n_recv_v, MFTL_MAX(v_range, 1));
  gt_begin(&s->t_tbl, n_kept_t + bs->n_recv_t, MFTL_MAX(t_range, 1));
  for (int i = 0; i < n_kept_v; ++i) gt_put(&s->v_tbl, kv_rec[i].global_id, i);
  for (int i = 0; i < n_kept_t; ++i) gt_put(&s->t_tbl, kt_rec[i].global_id, i);

  ensure_ints(&s->recv_v_logical, &s->recv_v_logical_cap, MFTL_MAX(bs->n_recv_v, 1));
  ensure_ints(&s->recv_t_logical, &s->recv_t_logical_cap, MFTL_MAX(bs->n_recv_t, 1));

  /* A vertex arrives once per sender sharing it; the copies are bit-identical,
   * so the first wins. */
  int n_new_v = 0, n_new_v_owned = 0;
  for (int i = 0; i < bs->n_recv_v; ++i)
  {
    const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx[i]];
    if (gt_has(&s->v_tbl, p->global_id))
    {
      s->recv_v_logical[i] = MFTL_ERROR;
      continue;
    }
    const int L = n_kept_v + n_new_v++;
    gt_put(&s->v_tbl, p->global_id, L);
    s->recv_v_logical[i] = L;
    if (p->owner_proc == my_rank) n_new_v_owned++;
  }

  /* Triangles cannot duplicate (only the single owner forwards one); the check
   * guards against silent corruption if that is ever violated. */
  int n_new_t = 0, n_new_t_owned = 0;
  for (int i = 0; i < bs->n_recv_t; ++i)
  {
    const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx[i]];
    if (gt_has(&s->t_tbl, p->global_id))
    {
      MFTL_ASSERT(0 && "redistribute: the same triangle arrived twice -- two ranks believe they own it");
      s->recv_t_logical[i] = MFTL_ERROR;
      continue;
    }
    const int L = n_kept_t + n_new_t++;
    gt_put(&s->t_tbl, p->global_id, L);
    s->recv_t_logical[i] = L;
    if (p->owner_proc == my_rank) n_new_t_owned++;
  }

  /* ── 3. Logical -> pool slot ─────────────────────────────────────────────── */
  ensure_ints(&s->v_final_of_logical, &s->v_final_of_logical_cap, MFTL_MAX(n_kept_v + n_new_v, 1));
  ensure_ints(&s->t_final_of_logical, &s->t_final_of_logical_cap, MFTL_MAX(n_kept_t + n_new_t, 1));

  /* Ownership of an arrival is re-read from the payload (same visiting order). */
  {
    const int n_kept_v_ghost = n_kept_v - bs->n_kept_v_owned;
    for (int L = 0; L < bs->n_kept_v_owned; ++L) s->v_final_of_logical[L] = L;
    for (int L = bs->n_kept_v_owned; L < n_kept_v; ++L) s->v_final_of_logical[L] = L + n_new_v_owned;

    int o = bs->n_kept_v_owned;
    int g = bs->n_kept_v_owned + n_new_v_owned + n_kept_v_ghost;
    for (int i = 0; i < bs->n_recv_v; ++i)
    {
      const int L = s->recv_v_logical[i];
      if (L == MFTL_ERROR) continue;
      const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx[i]];
      s->v_final_of_logical[L] = (p->owner_proc == my_rank) ? o++ : g++;
    }
    MFTL_ASSERT(o == bs->n_kept_v_owned + n_new_v_owned && "rebuild_object: owned vertex block overrun");
  }

  {
    const int n_kept_t_ghost = n_kept_t - bs->n_kept_t_owned;
    for (int L = 0; L < bs->n_kept_t_owned; ++L) s->t_final_of_logical[L] = L;
    for (int L = bs->n_kept_t_owned; L < n_kept_t; ++L) s->t_final_of_logical[L] = L + n_new_t_owned;

    int o = bs->n_kept_t_owned;
    int g = bs->n_kept_t_owned + n_new_t_owned + n_kept_t_ghost;
    for (int i = 0; i < bs->n_recv_t; ++i)
    {
      const int L = s->recv_t_logical[i];
      if (L == MFTL_ERROR) continue;
      const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx[i]];
      s->t_final_of_logical[L] = (p->owner_proc == my_rank) ? o++ : g++;
    }
    MFTL_ASSERT(o == bs->n_kept_t_owned + n_new_t_owned && "rebuild_object: owned triangle block overrun");
  }

  const int n_v = n_kept_v + n_new_v;
  const int n_v_local = bs->n_kept_v_owned + n_new_v_owned;
  const int n_t = n_kept_t + n_new_t;
  const int n_t_local = bs->n_kept_t_owned + n_new_t_owned;

  /* ── 4. Size the pools, then place the kept entities ─────────────────────── */
  CS_LAP(tm, t_arrivals);

  /* Grow GEOMETRICALLY before MFTLObjectSetCounts() (whose reserve is exact),
   * so counts jittering under advection do not reallocate both pools every
   * step. This leaves capacity in [n, 2n), which is why the shrink gate in
   * phase 5b is 4 and not 2 (MFTL_SHRINK_SLACK_DEFAULT). */
  if (n_v > obj->verts.capacity || n_t > obj->tris.capacity)
  {
    MFTLObjectReserve(obj, MFTLGrowCapacity(obj->verts.capacity, n_v), MFTLGrowCapacity(obj->tris.capacity, n_t));
  }
  MFTLObjectSetCounts(obj, n_v, n_v_local, n_t, n_t_local);
  CS_LAP(tm, t_reserve);

  {
    const int a = bs->n_kept_v_owned;
    const int b = n_kept_v - a;
    if (a > 0)
    {
      memcpy(&obj->verts.rec[0], &kv_rec[0], (size_t) a * sizeof(MFTL_Vertex));
      memcpy(&obj->verts.pos[0], &kv_pos[0], (size_t) a * 3 * sizeof(double));
    }
    if (b > 0)
    {
      const int dst = a + n_new_v_owned;
      memcpy(&obj->verts.rec[dst], &kv_rec[a], (size_t) b * sizeof(MFTL_Vertex));
      memcpy(&obj->verts.pos[3 * dst], &kv_pos[3 * a], (size_t) b * 3 * sizeof(double));
    }
  }
  {
    const int a = bs->n_kept_t_owned;
    const int b = n_kept_t - a;
    if (a > 0)
    {
      memcpy(&obj->tris.rec[0], &kt_rec[0], (size_t) a * sizeof(MFTL_Triangle));
      memcpy(&obj->tris.he[0], &kt_he[0], (size_t) a * 3 * sizeof(MFTL_HalfEdge));
    }
    if (b > 0)
    {
      const int dst = a + n_new_t_owned;
      memcpy(&obj->tris.rec[dst], &kt_rec[a], (size_t) b * sizeof(MFTL_Triangle));
      memcpy(&obj->tris.he[3 * dst], &kt_he[3 * a], (size_t) b * 3 * sizeof(MFTL_HalfEdge));
    }
  }

  CS_LAP(tm, t_kept_place);

  /* ── 5. Old pool index -> new pool slot, for the kept entities ───────────── */
  ensure_ints(&s->v_final_of_old, &s->v_final_of_old_cap, MFTL_MAX(n_old_v, 1));
  ensure_ints(&s->t_final_of_old, &s->t_final_of_old_cap, MFTL_MAX(n_old_t, 1));

  /* The triangle sentinel is load-bearing (it retires twins whose triangle was
   * not kept); the vertex one is read only by an assertion, because a kept
   * triangle's corners are always kept -- so it is cleared in Debug only. */
#ifndef NDEBUG
  for (int v = 0; v < n_old_v; ++v) s->v_final_of_old[v] = MFTL_ERROR;
#endif
  for (int t = 0; t < n_old_t; ++t) s->t_final_of_old[t] = MFTL_ERROR;
  for (int i = 0; i < n_kept_v; ++i) s->v_final_of_old[bs->kept_v[i]] = s->v_final_of_logical[i];
  for (int i = 0; i < n_kept_t; ++i) s->t_final_of_old[bs->kept_t[i]] = s->t_final_of_logical[i];

  /* ── 6. Re-link the kept half-edges: remap the target vertex, remap or retire
   * the twin (retired twins form the repair list for free), re-derive
   * `out_he_idx`, and emit `tris.verts` / `tris.neighbors` -- whose inputs are
   * in registers here -- instead of a closing MFTLObjectBuildDerived() pass.
   * The rotation is thus spelled in two places; inv_derived_matches_topology()
   * (tests/test_mesh_invariants.h) pins them together on every redistributed
   * mesh. Every slot is written exactly once, by forward streams. */
  ib_reset(&s->u_list);
  MFTL_HalfEdge *const he = obj->tris.he;
  MFTL_Vertex *const vr = obj->verts.rec;
  int *const restrict tv = obj->tris.verts;
  int *const restrict tn = obj->tris.neighbors;

  for (int i = 0; i < n_kept_t; ++i)
  {
    const int slot = s->t_final_of_logical[i];
    MFTL_HalfEdge *const h3 = &he[3 * slot];

    for (int k = 0; k < 3; ++k)
    {
      const int nv_idx = s->v_final_of_old[h3[k].vertex_idx];
      MFTL_ASSERT(nv_idx != MFTL_ERROR && "rebuild_object: a kept triangle references a vertex that was not kept");
      h3[k].vertex_idx = nv_idx;

      const int tw = h3[k].twin_idx;
      const int tw_tri = (tw != MFTL_ERROR) ? s->t_final_of_old[tw / 3] : MFTL_ERROR;
      if (tw_tri != MFTL_ERROR)
      {
        /* Keep the corner h % 3, remap the triangle h / 3. */
        h3[k].twin_idx = 3 * tw_tri + tw % 3;
      }
      else
      {
        h3[k].twin_idx = MFTL_ERROR;
        ib_add(&s->u_list, 3 * slot + k);
      }

      /* twin_idx / 3, or MFTL_ERROR until repair_twins() finds a partner. */
      tn[3 * slot + k] = tw_tri;
    }

    /* origin(3s+k) == target(3s+(k+2)%3), and all three targets are final now. */
    vr[h3[2].vertex_idx].out_he_idx = 3 * slot + 0;
    vr[h3[0].vertex_idx].out_he_idx = 3 * slot + 1;
    vr[h3[1].vertex_idx].out_he_idx = 3 * slot + 2;

    /* Slot k is the ORIGIN of half-edge 3s+k (as in build_topology_arrays()). */
    tv[3 * slot + 0] = h3[2].vertex_idx;
    tv[3 * slot + 1] = h3[0].vertex_idx;
    tv[3 * slot + 2] = h3[1].vertex_idx;
  }

  CS_LAP(tm, t_relink);

  /* ── 7. Write the arrivals ───────────────────────────────────────────────── */
  for (int i = 0; i < bs->n_recv_v; ++i)
  {
    const int L = s->recv_v_logical[i];
    if (L == MFTL_ERROR) continue;

    const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx[i]];
    const int slot = s->v_final_of_logical[L];

    obj->verts.pos[3 * slot + 0] = p->pos[0];
    obj->verts.pos[3 * slot + 1] = p->pos[1];
    obj->verts.pos[3 * slot + 2] = p->pos[2];

    MFTL_Vertex *const rec = &vr[slot];
    rec->global_id = p->global_id;
    rec->owner_proc = p->owner_proc;
    rec->flags = p->flags;
    rec->out_he_idx = MFTL_ERROR;  // filled in by the triangle pass below
  }

  for (int i = 0; i < bs->n_recv_t; ++i)
  {
    const int L = s->recv_t_logical[i];
    if (L == MFTL_ERROR) continue;

    const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx[i]];
    const int slot = s->t_final_of_logical[L];

    MFTL_Triangle *const rec = &obj->tris.rec[slot];
    rec->global_id = p->global_id;
    rec->owner_proc = p->owner_proc;
    rec->flags = p->flags;

    MFTL_HalfEdge *const h3 = &he[3 * slot];
    for (int k = 0; k < 3; ++k)
    {
      MFTL_ASSERT(gt_has(&s->v_tbl, p->vertex_gid[k]) &&
                  "rebuild_object: a received triangle references a vertex that never arrived -- the sender's vertex list is incomplete");
      h3[k].vertex_idx = s->v_final_of_logical[gt_get(&s->v_tbl, p->vertex_gid[k])];
      /* Reconstructed, not transmitted: half-edge 3t+k carries 3*tri_gid + k. */
      h3[k].global_id = 3 * p->global_id + k;
      h3[k].flags = p->he_flags[k];
      h3[k].twin_idx = MFTL_ERROR;
      tn[3 * slot + k] = MFTL_ERROR;  // every arriving twin is resolved by repair_twins
      ib_add(&s->u_list, 3 * slot + k);
    }

    vr[h3[2].vertex_idx].out_he_idx = 3 * slot + 0;
    vr[h3[0].vertex_idx].out_he_idx = 3 * slot + 1;
    vr[h3[1].vertex_idx].out_he_idx = 3 * slot + 2;

    tv[3 * slot + 0] = h3[2].vertex_idx;
    tv[3 * slot + 1] = h3[0].vertex_idx;
    tv[3 * slot + 2] = h3[1].vertex_idx;
  }

  /* ── 8. Repair, then rebuild what is genuinely derived ───────────────────── */
  CS_LAP(tm, t_arrivals);

  if (s->u_list.size > 0)
  {
    ensure_ints(&s->v_head, &s->v_head_cap, MFTL_MAX(n_v, 1));
    ensure_ints(&s->u_next, &s->u_next_cap, s->u_list.size);
    repair_twins(obj, s->u_list.data, s->u_list.size, s->v_head, s->u_next);
  }

#ifndef NDEBUG
  for (int v = 0; v < n_v; ++v)
  {
    MFTL_ASSERT(vr[v].out_he_idx != MFTL_ERROR && "rebuild_object: a vertex ended up with no incident triangle");
  }
#endif

  CS_LAP(tm, t_twins);

  /* Steps 6-8 emitted both derived arrays; restore the bit SetCounts cleared. */
  obj->derived = MFTL_DERIVED_TOPOLOGY;

  /* Cleared last: MFTLObjectSetCounts() set it on the way through. */
  obj->partition_dirty = MFTL_FALSE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Multi-body: separating the interleaved arrival stream
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Ints in the per-(body, destination) HEADER payload.
 *
 * `{gid, n_tris, n_verts}` name and delimit the run; the four global fields
 * (`next_*_gid`, `global_n_*`) let a rank that held nothing of a body create it
 * with correct id counters (arrival tables are sized from them). They are the
 * same on every holder, so a receiver that already has the body asserts them.
 */
#define MFTL_BODY_COUNT_INTS 7

/** @brief Body header slot: which body the run belongs to. */
#define MFTL_BODY_HDR_GID 0
/** @brief Body header slot: triangles in this run. */
#define MFTL_BODY_HDR_N_TRIS 1
/** @brief Body header slot: vertices in this run. */
#define MFTL_BODY_HDR_N_VERTS 2
/** @brief Body header slot: the body's next unused vertex global id. */
#define MFTL_BODY_HDR_NEXT_VGID 3
/** @brief Body header slot: the body's next unused triangle global id. */
#define MFTL_BODY_HDR_NEXT_TGID 4
/** @brief Body header slot: the body's global vertex count. */
#define MFTL_BODY_HDR_GLOBAL_V 5
/** @brief Body header slot: the body's global triangle count. */
#define MFTL_BODY_HDR_GLOBAL_T 6

/**
 * @brief Groups the arrival runs by object with a counting sort (no comparisons);
 *        within an object the arrival order is preserved, which makes "the first
 *        copy of a vertex wins" well defined.
 *
 * @param      runs        Per-sender runs of arriving entities.
 * @param      n_runs      Number of runs.
 * @param      num_objects Objects in the work list.
 * @param      use_tris    `true` to group the triangle runs, `false` for vertices.
 * @param[out] out_off     `[num_objects+1]` CSR offsets into @p out_idx.
 * @param[out] out_idx     Arrival indices, grouped by object.
 * @param      cursor      `[num_objects]` scratch write cursors.
 * @return The number of indices written to @p out_idx.
 */
static int group_runs_by_object(const body_run *runs, int n_runs, int num_objects, bool use_tris, int *out_off, int *out_idx, int *cursor)
{
  for (int o = 0; o <= num_objects; ++o) out_off[o] = 0;

  for (int k = 0; k < n_runs; ++k)
  {
    const int o = runs[k].obj_idx;
    if (o != MFTL_ERROR) out_off[o + 1] += use_tris ? runs[k].tri_len : runs[k].vert_len;
  }

  for (int o = 0; o < num_objects; ++o) out_off[o + 1] += out_off[o];
  for (int o = 0; o < num_objects; ++o) cursor[o] = out_off[o];

  /* Arrival order is preserved within each object's slice. */
  for (int k = 0; k < n_runs; ++k)
  {
    const int o = runs[k].obj_idx;
    if (o == MFTL_ERROR) continue;

    const int start = use_tris ? runs[k].tri_start : runs[k].vert_start;
    const int len = use_tris ? runs[k].tri_len : runs[k].vert_len;
    int p = cursor[o];
    for (int i = 0; i < len; ++i) out_idx[p + i] = start + i;
    cursor[o] = p + len;
  }

  return out_off[num_objects];
}

/* ════════════════════════════════════════════════════════════════════════════
 * 6b. Leaving a body alone
 *
 * A rank's state after a redistribution is a pure function of the global
 * topology and the owner map: read backwards, the halo rule DEFINES the output
 * ("rank r holds t iff r owns t, one of its corners or one of its edge
 * neighbours", plus the corners of what it holds). Positions only decide the
 * owner map. So a body for which neither input changed already holds the
 * answer, and its whole rebuild can be skipped (docs/analysis.md §2.4).
 *
 * The precondition is NOT "did topology change": under the remesher's freeze
 * discipline a topology change touches only unreplicated entities and is
 * invisible to the halo rule. What a skip breaks is GLOBAL IDS and FLAGS of
 * replicated entities -- e.g. renumbering leaves ghost-triangle ids to be
 * repaired by the rebuild. Hence: skip iff no entity changed owner AND the body
 * is not `partition_dirty`.
 *
 * The decision needs agreement (a rank may be the receiver of a change it
 * cannot see), and one collective per body would cost more than it saves. So
 * the object loop is split: pass 1a classifies every body and packs those this
 * rank saw change; ONE per-body MAX reduction over each body's holders agrees
 * the votes; pass 1c packs the bodies dirty elsewhere, re-running the idempotent
 * compute_owners() instead of storing owner arrays between the passes.
 *
 * rebuild_object() is not idempotent: an owned vertex whose incident triangles
 * are all remote is dropped and re-received at the end of the owned run, so a
 * second rebuild of an unchanged body can still move entities. A skip is
 * therefore taken only when the body was clean in this call AND the previous one
 * (D = "some rank saw it change"):
 *
 *     D now  D before   action               layout afterwards
 *       1       -       rebuild              not necessarily a fixed point
 *       0       1       rebuild ("settle")   a fixed point
 *       0       0       SKIP                 unchanged, still a fixed point
 *
 * which makes a skipped pool identical, slot for slot, to a rebuilt one, at the
 * cost of one rebuild per dirty-to-clean transition. "D before" is
 * MFTL_Object::partition_settled, so a body created on arrival starts unsettled
 * without affecting any other body.
 * ════════════════════════════════════════════════════════════════════════════ */

/** @brief Whether unchanged bodies may be left alone. See MFTLRedistributeSetSkipUnchangedBodies(). */
static bool g_skip_unchanged = MFTL_TRUE;

void MFTLRedistributeSetSkipUnchangedBodies(bool on) { g_skip_unchanged = on; }

bool MFTLRedistributeGetSkipUnchangedBodies(void) { return g_skip_unchanged; }

/**
 * @brief Waste factor at which a rebuilt body's pools are given back; 0 = off.
 *
 * Not part of the Debug consistency checksum: it changes no message and no mesh,
 * so ranks may legitimately differ.
 */
static double g_shrink_slack = 0.0;

void MFTLRedistributeSetShrinkSlack(double slack) { g_shrink_slack = (slack > 0.0) ? slack : 0.0; }

double MFTLRedistributeGetShrinkSlack(void) { return g_shrink_slack; }

/**
 * @brief Destination analysis, kept lists and packing for ONE body -- shared by
 *        both passes of the object loop. compute_owners() must have run on
 *        @p obj immediately before.
 *
 * @param      ws           The workspace; supplies every scratch array.
 * @param      obj          The mesh (its global id is the body name on the wire).
 * @param      my_rank      This rank.
 * @param      dd           Per-destination vertex de-duplication state.
 * @param[out] bs           This body's kept lists and counts.
 * @return 1 if any destination lies outside the 26-neighbourhood.
 */
static int analyze_and_pack_body(comm_workspace *ws, MFTL_Object *obj, int my_rank, vertex_dedup *dd, body_state *bs)
{
  CS_TIMELINE(tm);

  memset(ws->keep_v, 0, (size_t) obj->verts.n);
  memset(ws->keep_t, 0, (size_t) obj->tris.n_local);
  memset(dd->mask, 0, (size_t) obj->verts.n * sizeof(uint32_t));
  dd->n_bits = 0;
  CS_LAP(tm, t_masks);

  int n_touched = 0;
  const int requires_global = collect_destinations(obj, my_rank, ws->is_neighbor, ws->tri_owner, ws->vert_owner, dd, ws->keep_t, ws->keep_v, ws->tri_ids,
                                                   ws->vert_ids, ws->touched, &n_touched);

  build_kept_lists(obj, ws->keep_v, ws->keep_t, ws->vert_owner, ws->tri_owner, my_rank, bs);
  CS_LAP(tm, t_destinations);

  MFTLCommStats.n_destinations += n_touched;

  /* Only the ranks this object actually reached (at most 27 on the neighbour
   * route), not all nproc. */
  for (int i = 0; i < n_touched; ++i)
  {
    const int r = ws->touched[i];
    const int tri_before = ws->send_tris[r].count;
    const int vert_before = ws->send_verts[r].count;

    append_destination(obj, obj->global_id, ws->vert_gid, &ws->tri_ids[r], &ws->vert_ids[r], &ws->send_tris[r], &ws->send_verts[r],
                       dd->rank_bit[r] == MFTL_DEDUP_NO_BIT);

    sb_reserve(&ws->send_counts[r], MFTL_BODY_COUNT_INTS, sizeof(int));
    int *const c = (int *) ws->send_counts[r].buffer + ws->send_counts[r].count;
    c[MFTL_BODY_HDR_GID] = obj->global_id;  // the body GID, not this rank's slot for it
    c[MFTL_BODY_HDR_N_TRIS] = ws->send_tris[r].count - tri_before;
    c[MFTL_BODY_HDR_N_VERTS] = ws->send_verts[r].count - vert_before;
    c[MFTL_BODY_HDR_NEXT_VGID] = obj->next_vertex_gid;
    c[MFTL_BODY_HDR_NEXT_TGID] = obj->next_triangle_gid;
    c[MFTL_BODY_HDR_GLOBAL_V] = obj->global_n_vertices;
    c[MFTL_BODY_HDR_GLOBAL_T] = obj->global_n_triangles;
    ws->send_counts[r].count += MFTL_BODY_COUNT_INTS;

    ib_reset(&ws->tri_ids[r]);
    ib_reset(&ws->vert_ids[r]);
    dd->rank_bit[r] = MFTL_ERROR;
  }

  CS_LAP(tm, t_pack);
  return requires_global;
}

#ifndef NDEBUG
/**
 * @brief Debug-only: do all ranks agree on the size of the body-id space and on
 *        the skip switch?
 *
 * Equal id spaces are the observable consequence of "every rank added the same
 * bodies in the same order"; disagreement would merge entities into the wrong
 * body. The skip switch changes the number of collectives, so disagreement
 * would deadlock. `{sig, ~sig}` under one `MPI_MIN` detects a difference in
 * either direction. The local body count legitimately differs and is excluded.
 *
 * @param dom             Supplies the communicator.
 * @param n_bodies_global Size of the body-id space.
 */
static void check_object_lists_agree(const MFTL_Domain *dom, int n_bodies_global)
{

  unsigned sig = (unsigned) n_bodies_global * 2u + (g_skip_unchanged ? 1u : 0u);


  unsigned v[2] = {sig, ~sig};
  MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_UNSIGNED, MPI_MIN, dom->comm_world);
  MFTL_ASSERT(v[0] == sig && v[1] == ~sig &&
              "redistribute: ranks disagree about the body-id space or about the skip switch -- every rank must add the same bodies in the same "
              "order, and MFTLRedistributeSetSkipUnchangedBodies is collective");
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. The driver
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The redistribution implementation behind both public entry points:
 *        setup, pass 1a, skip vote, pass 1c, route decision, exchange, arrival
 *        grouping, rebuild, optional shrink, directory update.
 *
 * @param ens  The ensemble: its domain, local object list and persistent
 *             workspace.
 * @param maps `[ens->n_local_objects]` explicit owner maps in local-slot order,
 *             or NULL for the geometric rule.
 * @return #MFTL_SUCCESS.
 */
static int redistribute_core(MFTL_Ensemble *ens, const MFTL_OwnerMap *maps)
{
  MFTL_ASSERT(MFTL_MPI_PackedVertex != MPI_DATATYPE_NULL && "MFTLRedistribute: call MFTLRegisterMpiTypes() first");

  MFTL_Domain *const domain = &ens->domain;
  MFTL_Object **const objects = ens->local_objects;
  const int num_objects = ens->n_local_objects;

  CS_BEGIN(ts_total);
  MFTLCommStats.n_calls++;
  MFTLCommStats.n_objects_visited += num_objects;

  const int nproc = domain->nproc;
  const int my_rank = domain->rank;

#ifndef NDEBUG
  check_object_lists_agree(domain, ens->n_bodies_global);
#endif

  if (nproc < 2)
  {
    /* One rank: nothing moves; only establish the derived arrays if missing. */
    for (int o = 0; o < num_objects; ++o)
    {
      MFTL_Object *const obj = objects[o];
      MFTL_ASSERT(obj->verts.n_local == obj->verts.n && obj->tris.n_local == obj->tris.n && "redistribute: single-rank object holds ghosts");
      if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    }

    /* Same postcondition as in parallel: an empty body is held by nobody.
     * Walked backwards so an erase does not move a slot still to be visited. */
    for (int o = num_objects - 1; o >= 0; --o)
    {
      MFTL_Object *gone = objects[o];
      if (gone->tris.n > 0 || gone->verts.n > 0) continue;

      MFTLEnsembleEraseLocalSlot(ens, MFTLEnsembleFindLocalSlot(ens, gone->global_id));
      MFTLObjectDestroy(&gone);
      MFTLCommStats.n_objects_destroyed++;
    }

    CS_END(ts_total, t_total);
    return MFTL_SUCCESS;
  }

  /* ── Phase 0: setup ────────────────────────────────────────────────────── */
  CS_BEGIN(ts_setup);

  int global_tris = 0, global_verts = 0;
  int sum_verts = 0, sum_tris = 0, max_verts = 1, max_tris = 1;
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_ASSERT(objects[o]->domain == domain && "redistribute: object belongs to a different domain");
    global_tris += objects[o]->global_n_triangles;
    global_verts += objects[o]->global_n_vertices;
    sum_verts += objects[o]->verts.n;
    sum_tris += objects[o]->tris.n_local;
    max_verts = MFTL_MAX(max_verts, objects[o]->verts.n);
    max_tris = MFTL_MAX(max_tris, objects[o]->tris.n);
  }

  int tri_hint, vert_hint;
  calc_buffer_hints(global_tris, global_verts, nproc, &tri_hint, &vert_hint);

  /* The header payload always travels (it is what lets an arrival create its
   * body), so the payload count -- collective state -- is constant. */
  if (ens->comm_workspace == MFTL_NULL) ens->comm_workspace = MFTLCallocOrDie(1, sizeof(comm_workspace), "MFTL communication workspace");
  comm_workspace *const ws = (comm_workspace *) ens->comm_workspace;

  /* A LOCAL SLOT indexes this call's work list and every workspace array; a
   * BODY ID indexes anything that goes on the wire.
   *
   * The pre-exchange holder test must be taken HERE: compute_owners() below
   * overwrites every `owner_proc` with the DESTINATION rank, after which the
   * pools describe the partition about to exist rather than the one the vote
   * has to reach. */
  const bool holders_fit = MFTLEnsembleBodyHoldersFitTheGraph(ens);

  /* At least one slot: the arrival path grows the arrays from here. */
  ws_reserve(ws, nproc, MFTL_MAX(num_objects, 1), sum_verts, sum_tris, max_verts, max_tris, tri_hint, vert_hint);

  char *const is_neighbor = ws->is_neighbor;
  memset(is_neighbor, 0, (size_t) nproc);
  for (int i = 0; i < domain->num_neighbors; ++i) is_neighbor[domain->neighbors[i]] = 1;

  MFTL_SendBuffer *const send_tris = ws->send_tris;
  MFTL_SendBuffer *const send_verts = ws->send_verts;
  MFTL_SendBuffer *const send_counts = ws->send_counts;
  int *const kept_arena_v = ws->kept_arena_v;
  int *const kept_arena_t = ws->kept_arena_t;
  int *const tri_owner = ws->tri_owner;
  int *const vert_owner = ws->vert_owner;
  int *const vert_gid = ws->vert_gid;

  /* The work list: bodies held now; phase 4 appends first-time arrivals. */
  const int n_pre = num_objects;
  int n_work = n_pre;
  for (int o = 0; o < n_pre; ++o) ws->work[o] = objects[o];

  /* The payload buffers keep their allocations; only the fill level is reset. */
  for (int r = 0; r < nproc; ++r)
  {
    send_tris[r].count = 0;
    send_verts[r].count = 0;
    send_counts[r].count = 0;
  }
  memset(ws->states, 0, (size_t) n_pre * sizeof(body_state));

  vertex_dedup dd;
  dd.mask = ws->dedup_mask;
  dd.rank_bit = ws->rank_bit;
  dd.n_bits = 0;
  CS_END(ts_setup, t_setup);

  /* ── Phase 1a: ownership everywhere, analysis where THIS rank sees change
   * (section 6b). */
  int requires_global = 0;
  int arena_v = 0, arena_t = 0;

  double *const dirty_local = ws->dirty_votes;
  double *const dirty_global = ws->dirty_votes + ws->cap_votes;

  memset(dirty_local, 0, (size_t) n_pre * sizeof(double));

  for (int o = 0; o < n_pre; ++o)
  {
    MFTL_Object *const obj = ws->work[o];
    body_state *const bs = &ws->states[o];

    bs->kept_v = kept_arena_v + arena_v;
    bs->kept_t = kept_arena_t + arena_t;
    arena_v += obj->verts.n;
    arena_t += obj->tris.n_local;

    /* Nothing of this body is here: no opinion. */
    if (obj->tris.n < 1 || obj->verts.n < 1) continue;

    CS_TIMELINE(tm);
    int dirty = compute_owners(obj, maps ? &maps[o] : MFTL_NULL, tri_owner, vert_owner, vert_gid, MFTL_TRUE);
    CS_LAP(tm, t_owners);

    /* The dirty stamp, and a body never given derived arrays, force a rebuild. */
    dirty |= obj->partition_dirty;
    dirty |= !MFTLObjectHasTopology(obj);
    if (!g_skip_unchanged) dirty = 1;

    if (!dirty) continue;

    dirty_local[o] = 1.0;
    requires_global |= analyze_and_pack_body(ws, obj, my_rank, &dd, bs);
  }

  /* ── Phase 1b: one per-body MAX over each body's HOLDERS. Not in place:
   * pass 1c needs this rank's own votes to know what it already did. */
  CS_BEGIN(ts_vote);
  {
    static const MFTL_BodyReduceOp vote_op = MFTL_BODY_REDUCE_MAX;
    memcpy(dirty_global, dirty_local, (size_t) n_pre * sizeof(double));
    MFTLEnsembleReduceBodiesPrescanned(ens, dirty_global, 1, &vote_op, holders_fit);
  }
  CS_END(ts_vote, t_skip_vote);

  /* ── Phase 1c: bodies dirty elsewhere, and clean bodies still settling. */
  for (int o = 0; o < n_pre; ++o)
  {
    MFTL_Object *const obj = ws->work[o];

    if (dirty_local[o] != 0.0) continue;  // already done in pass 1a
    if (dirty_global[o] == 0.0 && obj->partition_settled) continue;  // skipped

    if (obj->tris.n < 1 || obj->verts.n < 1) continue;  // nothing here to send

    /* Idempotent re-classification; counters not updated twice. */
    CS_TIMELINE(tm);
    (void) compute_owners(obj, maps ? &maps[o] : MFTL_NULL, tri_owner, vert_owner, vert_gid, MFTL_FALSE);
    CS_LAP(tm, t_owners);

    requires_global |= analyze_and_pack_body(ws, obj, my_rank, &dd, &ws->states[o]);
    MFTLCommStats.n_objects_reclassified++;
    if (dirty_global[o] == 0.0) MFTLCommStats.n_objects_settled++;
  }



  /* ── Phase 2: agree on the route. Collective by necessity: a non-neighbour
   * receiver cannot know it must listen on the global path. */
  CS_BEGIN(ts_route);
  int global_decision = 0;
  MPI_Allreduce(&requires_global, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);
  CS_END(ts_route, t_route);
  if (global_decision)
    MFTLCommStats.n_route_global++;
  else
    MFTLCommStats.n_route_neighbor++;

  /* ── Phase 3: exchange ─────────────────────────────────────────────────── */
  MFTL_Exchange xfer[3];
  xfer[0].send_bufs = send_verts;
  xfer[0].type = MFTL_MPI_PackedVertex;
  xfer[0].type_size = sizeof(MFTL_PackedVertex);
  xfer[1].send_bufs = send_tris;
  xfer[1].type = MFTL_MPI_PackedTriangleComplex;
  xfer[1].type_size = sizeof(MFTL_PackedTriangleComplex);
  xfer[2].send_bufs = send_counts;
  xfer[2].type = MPI_INT;
  xfer[2].type_size = sizeof(int);

  for (int r = 0; r < nproc; ++r)
  {
    const long bytes = (long) send_verts[r].count * (long) sizeof(MFTL_PackedVertex) + (long) send_tris[r].count * (long) sizeof(MFTL_PackedTriangleComplex) +
                       (long) send_counts[r].count * (long) sizeof(int);
    if (bytes > 0)
    {
      MFTLCommStats.n_bytes_sent += bytes;
      MFTLCommStats.n_messages++;
    }
  }

  MFTL_ExchangeCtx ctx;
  MFTLExchangeStart(domain, xfer, 3, global_decision != 0, &ctx);

  /* The send buffers were copied; they persist for the next call. */

  /* ── Phase 4: reconstruct the per-body arrival slices ───────────────────── */
  CS_BEGIN(ts_w2);
  MFTLExchangeWait(&ctx, 2);
  CS_END(ts_w2, t_wait);

  CS_BEGIN(ts_g2);
  const int *const rc = (const int *) xfer[2].recv_buf;
  const int n_runs = xfer[2].recv_count / MFTL_BODY_COUNT_INTS;

  ws_bytes((void **) &ws->runs, &ws->cap_runs, MFTL_MAX(n_runs, 1), sizeof(body_run), "redistribution body runs");
  body_run *const runs = ws->runs;

  int t_off = 0, v_off = 0;
  for (int k = 0; k < n_runs; ++k)
  {
    const int *const tp = &rc[MFTL_BODY_COUNT_INTS * k];
    const int gid = tp[MFTL_BODY_HDR_GID];

    int slot = slot_of_body(ws->work, n_pre, n_work, gid);
    if (slot == MFTL_ERROR)
    {
      /* A body arriving where this rank held nothing: created and appended to
       * the work list (merged into the directory at the end); it starts
       * unsettled. */
      MFTL_ASSERT(gid >= 0 && gid < ens->n_bodies_global && "redistribute: received a body gid outside the id space");

      ws_bodies_reserve(ws, n_work + 1);
      MFTL_Object *const arrived = MFTLObjectCreate(gid);
      arrived->domain = domain;
      arrived->derived = MFTL_DERIVED_TOPOLOGY;  // an empty mesh is trivially described

      /* The persistent slot may hold the previous call's state: clear it. */
      memset(&ws->states[n_work], 0, sizeof(body_state));

      ws->work[n_work] = arrived;
      slot = n_work++;
      MFTLCommStats.n_objects_created++;
    }

    /* The four global fields: identical on every holder (asserted), applied. */
    MFTL_Object *const target = ws->work[slot];
    MFTL_ASSERT((target->verts.n == 0 && target->tris.n == 0) ||
                (target->next_vertex_gid == tp[MFTL_BODY_HDR_NEXT_VGID] && target->next_triangle_gid == tp[MFTL_BODY_HDR_NEXT_TGID] &&
                 target->global_n_vertices == tp[MFTL_BODY_HDR_GLOBAL_V] && target->global_n_triangles == tp[MFTL_BODY_HDR_GLOBAL_T]) &&
                    "redistribute: two ranks disagree about a body's global counters");

    target->next_vertex_gid = tp[MFTL_BODY_HDR_NEXT_VGID];
    target->next_triangle_gid = tp[MFTL_BODY_HDR_NEXT_TGID];
    target->next_halfedge_gid = 3 * target->next_triangle_gid;
    target->global_n_vertices = tp[MFTL_BODY_HDR_GLOBAL_V];
    target->global_n_triangles = tp[MFTL_BODY_HDR_GLOBAL_T];

    runs[k].obj_idx = slot;
    runs[k].tri_start = t_off;
    runs[k].tri_len = tp[MFTL_BODY_HDR_N_TRIS];
    runs[k].vert_start = v_off;
    runs[k].vert_len = tp[MFTL_BODY_HDR_N_VERTS];

    t_off += tp[MFTL_BODY_HDR_N_TRIS];
    v_off += tp[MFTL_BODY_HDR_N_VERTS];
  }

  CS_END(ts_g2, t_group);

  CS_BEGIN(ts_w01);
  MFTLExchangeWait(&ctx, 0);
  MFTLExchangeWait(&ctx, 1);
  CS_END(ts_w01, t_wait);

  CS_BEGIN(ts_g01);
  MFTL_ASSERT(t_off == xfer[1].recv_count && "redistribute: body triangle counts do not add up to the arrival count");
  MFTL_ASSERT(v_off == xfer[0].recv_count && "redistribute: body vertex counts do not add up to the arrival count");

  ws_bytes((void **) &ws->grp_idx_v, &ws->cap_gidx_v, MFTL_MAX(xfer[0].recv_count, 1), sizeof(int), "redistribution group indices");
  ws_bytes((void **) &ws->grp_idx_t, &ws->cap_gidx_t, MFTL_MAX(xfer[1].recv_count, 1), sizeof(int), "redistribution group indices");
  int *const grp_off_v = ws->grp_off_v;
  int *const grp_off_t = ws->grp_off_t;
  int *const grp_idx_v = ws->grp_idx_v;
  int *const grp_idx_t = ws->grp_idx_t;

  group_runs_by_object(runs, n_runs, n_work, MFTL_FALSE, grp_off_v, grp_idx_v, ws->cursor);
  group_runs_by_object(runs, n_runs, n_work, MFTL_TRUE, grp_off_t, grp_idx_t, ws->cursor);

  for (int o = 0; o < n_work; ++o)
  {
    ws->states[o].recv_v_idx = grp_idx_v + grp_off_v[o];
    ws->states[o].n_recv_v = grp_off_v[o + 1] - grp_off_v[o];
    ws->states[o].recv_t_idx = grp_idx_t + grp_off_t[o];
    ws->states[o].n_recv_t = grp_off_t[o + 1] - grp_off_t[o];
  }

  CS_END(ts_g01, t_group);

  MFTLCommStats.n_bytes_recv += (long) xfer[0].recv_count * (long) sizeof(MFTL_PackedVertex) +
                                (long) xfer[1].recv_count * (long) sizeof(MFTL_PackedTriangleComplex) + (long) xfer[2].recv_count * (long) sizeof(int);

  /* ── Phase 5: rebuild ──────────────────────────────────────────────────── */
  rebuild_scratch *const scratch = &ws->scratch;
  if (scratch->u_list.hint == 0) ib_init(&scratch->u_list, 1024);

  for (int o = 0; o < n_work; ++o)
  {
    MFTL_Object *const obj = ws->work[o];

    /* An arrival is dirty by construction and has no vote slot. */
    const bool clean = (o < n_pre) && (dirty_global[o] == 0.0);

    if (g_skip_unchanged && clean && obj->partition_settled)
    {
      /* Nobody saw it change and it is settled: the pool IS the answer. */
      MFTLCommStats.n_objects_skipped++;

      if (obj->tris.n < 1 || obj->verts.n < 1)
      {
        obj->derived = MFTL_DERIVED_TOPOLOGY;  // an empty mesh is trivially described
      }
      continue;
    }

    rebuild_object(obj, &ws->states[o], (const MFTL_PackedVertex *) xfer[0].recv_buf, (const MFTL_PackedTriangleComplex *) xfer[1].recv_buf, my_rank, scratch);
  }

  /* ── Phase 5b: optional shrink, after every rebuild (no pool pointer is live
   * here). The gate (> slack*n, slack >= 4) never fights the pre-growth above,
   * which leaves capacity in [n, 2n). */
  if (g_shrink_slack > 0.0)
  {
    CS_BEGIN(ts_shrink);
    for (int o = 0; o < n_work; ++o)
    {
      MFTL_Object *const obj = ws->work[o];
      const size_t before = MFTLObjectPoolBytes(obj);

      if (MFTLObjectShrinkToFit(obj, g_shrink_slack))
      {
        MFTLCommStats.n_objects_shrunk++;
        MFTLCommStats.n_bytes_reclaimed += (long) (before - MFTLObjectPoolBytes(obj));
      }
    }
    CS_END(ts_shrink, t_reserve);
  }

  /* "D before" for the next call (section 6b). */
  for (int o = 0; o < n_work; ++o) ws->work[o]->partition_settled = (o < n_pre) && (dirty_global[o] == 0.0);

  /* ── Phase 6: the sorted directory catches up (no slot index is live any
   * more): arrivals are inserted, bodies left empty are destroyed. */
  for (int o = n_pre; o < n_work; ++o)
  {
    MFTL_Object *arrived = ws->work[o];
    if (arrived->tris.n > 0 || arrived->verts.n > 0)
    {
      MFTLEnsembleInsertLocalObject(ens, arrived);
    }
    else
    {
      /* Every arriving entity was a duplicate: nothing to hold. */
      MFTLObjectDestroy(&arrived);
      MFTLCommStats.n_objects_destroyed++;
    }
    ws->work[o] = MFTL_NULL;
  }

  for (int o = 0; o < n_pre; ++o)
  {
    MFTL_Object *gone = ws->work[o];
    if (gone->tris.n > 0 || gone->verts.n > 0) continue;

    /* By GID: the insertions above shifted the slots. */
    const int slot = MFTLEnsembleFindLocalSlot(ens, gone->global_id);
    MFTL_ASSERT(slot != MFTL_ERROR && "redistribute: a body in the work list is not in the directory");
    MFTLEnsembleEraseLocalSlot(ens, slot);
    MFTLObjectDestroy(&gone);
    MFTLCommStats.n_objects_destroyed++;
    ws->work[o] = MFTL_NULL;
  }

  /* Only what the exchange layer allocated for this call; the rest persists. */
  CS_BEGIN(ts_free3);
  free(xfer[0].recv_buf);
  free(xfer[1].recv_buf);
  free(xfer[2].recv_buf);

  MFTLExchangeFree(&ctx);
  MFTLCommStats.n_frees += 3;
  CS_END(ts_free3, t_teardown);

  CS_END(ts_total, t_total);
  return MFTL_SUCCESS;
}

int MFTLRedistributeWithOwners(MFTL_Ensemble *ensemble, const MFTL_OwnerMap *maps)
{
  if (ensemble == MFTL_NULL || ensemble->local_objects == MFTL_NULL) return MFTL_ERROR;

  /* No bodies anywhere: a no-op. The test is GLOBAL -- a rank holding no bodies
   * must still enter the exchange, since another rank may send it one. */
  if (ensemble->n_bodies_global == 0) return MFTL_SUCCESS;

  return redistribute_core(ensemble, maps);
}

int MFTLRedistribute(MFTL_Ensemble *ensemble) { return MFTLRedistributeWithOwners(ensemble, MFTL_NULL); }
