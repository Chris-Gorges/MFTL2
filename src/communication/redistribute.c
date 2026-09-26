/**
 * @file redistribute.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Redistribution: re-partitioning a distributed mesh over the process grid.
 *
 * Both public entry points run the SAME implementation (redistribute_core below);
 * MFTLRedistribute is MFTLRedistributeWithOwners with the geometric owner rule.
 *
 * ## THE ONE IDEA THIS FILE EXISTS TO EXPRESS
 *
 * MOST OF A PARTITION DOES NOT MOVE, SO MOST OF A PARTITION MUST NOT BE TOUCHED
 * BY THE WIRE FORMAT.
 *
 * Treating "stays here" as just another destination runs it through the full
 * protocol: pack every kept entity into an MFTL_Packed*,
 * concatenate kept and received, radix-sort the whole thing by global id,
 * de-duplicate, unpack, and insert one hash entry per vertex and four per
 * triangle. On a steadily advecting mesh that is O(V + T) of pure round trip for
 * a shell of entities that is O(V^(1/2)).
 *
 * Here the destination analysis marks kept entities instead of packing them, and
 * the rebuild is:
 *
 *   kept      -> a PERMUTATION of the pool. Records and positions are gathered
 *                into scratch in final order and written back as two memcpys.
 *                Nothing is sorted, nothing is serialised.
 *   arrivals  -> classified against a dense global-id table (gt_* in
 *                comm_internal.h), which is what replaces khash. Global ids are
 *                dense per class, so "have I got g already" is an array probe.
 *   twins     -> REPAIRED, not rebuilt. A twin between two kept triangles is an
 *                index remap; only half-edges that actually lost their partner
 *                are re-paired, and they cost nothing to collect because the
 *                remap already looks at every one of them.
 *
 * ### Why the incremental twin repair is correct
 *
 * Two half-edges are twins iff they are the two orientations of one undirected
 * edge. Kept-ness is decided per TRIANGLE, so if h is kept its twin is kept iff
 * its triangle is -- and both sides then get remapped consistently. If only one
 * side is kept, that side is set to MFTL_ERROR and its partner does not exist
 * locally any more. Therefore:
 *
 *     a half-edge whose twin survived the remap is FINAL, and
 *     a half-edge whose twin did not can only pair with another such half-edge.
 *
 * The repair set is closed under the twin relation, so bucketing it against
 * itself finds every pair a full MFTLRebuildTwins would have found. Guarded by
 * tests/test_redistribution.c, which re-runs MFTLRebuildTwins after a
 * redistribution and asserts that it changes nothing.
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
 * @brief Per-destination starting capacity, from the mesh density.
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
 * The change counts are accumulated inside compute_owners' two hot loops rather
 * than derived afterwards, because `owner_proc` is overwritten in place and the
 * previous value has no other reader. The cost is one integer compare per entity
 * on a record the loop is already storing into -- the cache line is resident by
 * necessity.
 *
 * They used to be a profile-build extra. They are unconditional now because the
 * VERDICT is load-bearing: "did any entity of this body change owner" is half of
 * the precondition for leaving the body alone entirely (see redistribute_core),
 * so the compare has to happen whether or not anyone is reading the counter. The
 * counters come along for free once it does.
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
 * @brief Assigns every entity to the rank whose sub-domain contains it.
 *
 * Both loops are pure streams over one contiguous array (`verts.pos`,
 * `tris.center`) with one small scattered write each. That is the access pattern
 * those interleaved arrays exist to serve: 24 B per entity instead of a stride
 * through a record.
 *
 * ── The override ───────────────────────────────────────────────────────────
 *
 * @p map is normally MFTL_NULL and the geometric rule above is the whole story.
 * A caller with its own ownership rule (MFTL_OwnerMap, mftl_communication.h --
 * the distributed remesher has two) supplies the destinations instead, and the
 * only thing that changes is where `owner` comes from: the record write, the
 * global-id harvest and everything downstream are identical, because the rest of
 * the redistribution never asked how ownership was decided.
 *
 * The test is per OBJECT, hoisted out of both loops, so the production path
 * costs one predicted branch per body per timestep and keeps the inlined
 * MFTLGetOwnerProc in a loop with nothing conditional in it.
 *
 * The centre check goes with the geometric rule and not with the function: an
 * override does not route by `tris.center`, so a stale centre is not an error
 * for it, and asserting anyway would make a legitimate caller pay a full
 * geometry pass to satisfy a precondition it does not have.
 *
 * ── The verdict ────────────────────────────────────────────────────────────
 *
 * The return value is what lets redistribute_core leave a body alone. `changed`
 * is accumulated in both loops from the value being overwritten -- the previous
 * `owner_proc` has no other reader, so this is the only place the question can
 * be asked at all, and asking it costs one compare against a word the loop is
 * already storing into.
 *
 * It is idempotent, and the second pass over a body relies on that: called again
 * on a pool it has already classified it rewrites the same values and reports
 * 0. @p record_stats is what stops the second call double-counting.
 *
 * @param      obj          The mesh.
 * @param      map          Explicit owner map, or NULL for the geometric rule.
 * @param[out] tri_owner    `[tris.n]` destination rank per triangle.
 * @param[out] vert_owner   `[verts.n]` destination rank per vertex.
 * @param[out] vert_gid     `[verts.n]` vertex global ids, gathered contiguously so
 *                          the packing loops never touch the AoS records again.
 * @param      record_stats `false` on a re-classification, so that the scan and
 *                          change counters describe entities once each.
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
    /* One compare against a word this loop is storing into anyway, in the loop
     * CLAUDE.md records a 1.70x regression on. It is here rather than behind
     * MFTL_COMM_PROFILE because the ANSWER decides whether the rest of the
     * redistribution runs for this body at all. */
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
 * @brief Makes each REMOTE destination's vertex list unique on construction.
 *
 * A vertex is a candidate once per incident triangle bound for the same
 * destination, so the list would otherwise arrive `valence` times over and have
 * to be fixed up afterwards with a sort plus a unique pass -- work to remove
 * duplicates that never had to be created.
 *
 * One bit per (vertex, destination) replaces it: `mask[v] & (1 << b)` is "v is
 * already queued for the destination holding bit slot b", so the test-and-set is
 * O(1) and nothing is ever sorted.
 *
 * Bit slots are handed out on first contact. A REMOTE destination set is bounded
 * by the 26-neighbourhood on the local path, so 32 bits are enough there by
 * construction; the global path (the initial scatter, or an object spanning more
 * than 32 cells) can exceed it, and those ranks get NO_BIT and fall back to the
 * sort. Correctness never depends on the bound.
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
 * @brief Works out which ranks each owned triangle must be sent to, and what stays.
 *
 * A triangle goes to the owner of its centre (that rank will own it), and to the
 * owners of its three vertices and its three edge neighbours (those ranks need it
 * as a ghost). Sending that surplus here is what produces a complete, VERTEX-
 * COMPLETE one-layer halo without a second communication round.
 *
 * THIS RANK is one of those destinations like any other -- it is simply handled
 * by setting a keep flag rather than by packing a copy.
 *
 * Ghost triangles are never considered: their owner sends the authoritative copy,
 * and forwarding stale ghosts is how a partition accumulates entities that no
 * rank actually owns. With ownership as a prefix that filter is the loop bound,
 * not a test.
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
 * @return 1 if any destination fell outside the 26-neighbourhood, which forces
 *         the exchange onto the global communicator.
 */
static int collect_destinations(const MFTL_Object *obj, int my_rank, const char *restrict is_neighbor, const int *restrict tri_owner,
                                const int *restrict vert_owner, vertex_dedup *dd, char *restrict keep_t, char *restrict keep_v,
                                mftl_int_buffer *tri_ids, mftl_int_buffer *vert_ids, int *touched, int *n_touched)
{
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "redistribute: tris.verts/neighbors are stale -- MFTLObjectBuildDerived first");

  const int n_owned = obj->tris.n_local;
  if (n_owned < 1) return 0;

  /* 12 B per triangle out of each, streaming. Reading the same information from
   * the half-edge records would be 48 B of which 36 are fields this loop does not
   * want -- which is the entire argument for keeping these two arrays. */
  const int *const restrict tv = obj->tris.verts;
  const int *const restrict tn = obj->tris.neighbors;

  uint32_t *const restrict mask = dd->mask;
  int *const restrict rank_bit = dd->rank_bit;

  int requires_global = 0;

  for (int t = 0; t < n_owned; ++t)
  {
    const int base = 3 * t;
    const int v0 = tv[base + 0], v1 = tv[base + 1], v2 = tv[base + 2];

    /* ── The stencil, and the question asked of it without writing it down ──
     *
     * Fast path: the whole stencil sits on one rank. That is every triangle in
     * the interior of a partition -- the overwhelming majority -- and it skips
     * the quadratic dedup below entirely.
     *
     * ASKING IT THROUGH A cand[7] ARRAY MAKES THE FAST PATH PAY FOR THE SLOW
     * ONE. `cand[n_cand++]` is a VARIABLE index, so the array cannot live in
     * registers: measured on the -O3 arm64 build, four to seven stack stores per
     * owned triangle followed by three reloads for the all-equal test, on the
     * second-largest bucket of the phase. An XOR-OR reduction answers the same
     * question in registers -- "is any of the seven different from the first" is
     * exactly "is the OR of the six differences non-zero" -- and cand[] is then
     * built only when the answer is yes.
     *
     * A MISSING edge neighbour contributes `r` rather than a sentinel, so its
     * term is identically zero and needs no select. It is NOT inserted into
     * cand[] below, because MFTLGetUniqueIntegers preserves first-seen order:
     * dests[] decides the order this body is appended to each destination's id
     * lists, hence the wire order, hence the arrival order, hence the pool slot
     * order. The candidate list is built exactly as it was. */
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
 * @brief Packs one triangle and its three half-edges into a transfer unit.
 *
 * Ten ints out, and every one of them is something the receiver cannot derive.
 * What is NOT written here is the point: no area, no normal, no centre, no
 * half-edge global ids (they are 3*triangle_gid + k), no `next`/`triangle`/`twin`.
 *
 * @param obj      The mesh.
 * @param t        Triangle index.
 * @param tv       `obj->tris.verts`, streaming past anyway.
 * @param vgid     `[verts.n]` vertex global ids, contiguous, from compute_owners().
 * @param[out] dst The wire record.
 * @param body_gid #MFTL_Object::global_id of the owning body.
 */
static inline void pack_triangle(const MFTL_Object *obj, int t, const int *restrict tv, const int *restrict vgid, MFTL_PackedTriangleComplex *dst,
                                 int body_gid)
{
  const MFTL_Triangle *const src = &obj->tris.rec[t];
  const MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
  const int base = 3 * t;

  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->flags = src->flags;
  dst->body_gid = body_gid;

  /* tris.verts[3t+k] is the ORIGIN of half-edge k; the wire wants its TARGET, and
   * target(k) == origin((k+1)%3). Getting this rotation wrong is silent -- the
   * triangles stay valid, they are just rotated. */
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
 * @brief Debug-only check behind the "triangle lists need no dedup" argument.
 *
 * @param a Array to inspect.
 * @param n Number of elements.
 * @return `true` if `a[i] < a[i+1]` throughout, i.e. sorted AND duplicate-free.
 */
static bool is_strictly_increasing(const int *a, int n)
{
  for (int i = 1; i < n; ++i)
    if (a[i] <= a[i - 1]) return MFTL_FALSE;
  return MFTL_TRUE;
}
#endif

/**
 * @brief Packs one object's contribution for one REMOTE destination rank.
 *
 * Consumes the transient id lists and appends to that destination's send buffers,
 * so several objects accumulate into the same pair of messages.
 *
 * @param obj         The mesh.
 * @param body_gid    #MFTL_Object::global_id of the owning body.
 * @param vgid        `[verts.n]` vertex global ids.
 * @param tri_ids     Triangle indices bound for this destination.
 * @param vert_ids    Vertex indices bound for this destination.
 * @param out_tris    Send buffer the packed triangles are appended to.
 * @param out_verts   Send buffer the packed vertices are appended to.
 * @param needs_dedup `true` when the destination had no de-duplication bit slot,
 *                    so the vertex list must be sorted and uniqued here.
 */
static void append_destination(const MFTL_Object *obj, int body_gid, const int *restrict vgid, mftl_int_buffer *tri_ids, mftl_int_buffer *vert_ids,
                               MFTL_SendBuffer *out_tris, MFTL_SendBuffer *out_verts, bool needs_dedup)
{
  /* Triangles need NO de-duplication. collect_destinations visits each triangle
   * once and appends it at most once per DISTINCT destination, so this list is
   * already strictly increasing. */
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
 * @brief Splits one entity class's keep flags into the two runs, in ONE pass.
 *
 * ── Why two cursors rather than two scans ──────────────────────────────────
 *
 * The rebuild wants the owned entities first (n_local is an array PREFIX) and
 * both runs ascending (so the gather in rebuild_object stays a forward scan).
 * The obvious way to get that is one scan per run, which reads the flags and the
 * ownership twice. Instead the owned run is filled FORWARD from 0 and the ghost
 * run BACKWARD from the top of the body's arena slice -- which is already sized
 * for the whole class, so the two cursors cannot meet: every kept entity takes
 * one slot from one end and there are at most @p n of them.
 *
 * The backward run comes out descending, so it is reversed and moved down
 * against the owned run. Both are O(halo), against O(n) for the pass they
 * replace, and the result is BYTE-IDENTICAL to what two scans produce -- which
 * is what keeps the pool's slot order, and therefore the partition digest,
 * unchanged.
 *
 * Counting the runs first would let the two be written straight to their final
 * offsets and skip the fix-up, and it is the wrong trade: the counter would have
 * to be incremented in collect_destinations, where `keep[i] = 1` is an
 * unconditional store today, and a read-modify-write with a loop-carried
 * dependency in the hottest loop of the phase is not worth an O(halo) memmove
 * here.
 *
 * ── Why the OWNERSHIP comes from the dense array and not from the record ───
 *
 * `owner[i]` and `rec[i].owner_proc` hold the same value -- compute_owners()
 * wrote both, immediately before, and the entry is the NEW owner rather than the
 * one the entity arrived with. The array is 4 B per entity against 16 for
 * MFTL_Vertex and 12 for MFTL_Triangle, it is dense, and it is still hot. This
 * is an AoS record being read for one int; the SoA copy already exists.
 *
 * @param      keep    `[n]` keep flags from collect_destinations().
 * @param      owner   `[n]` destination rank per entity, from compute_owners().
 * @param      n       Entities to consider: `verts.n`, or `tris.n_local` because
 *                     only an owned triangle can ever be kept.
 * @param      my_rank This rank.
 * @param[out] out     `[n]` the kept indices: owned run then ghost run, both
 *                     ascending. Must be the body's full arena slice.
 * @param[out] n_owned Length of the owned run.
 * @return Kept entities in total, owned plus retained ghosts.
 */
static int split_kept_run(const char *restrict keep, const int *restrict owner, int n, int my_rank, int *restrict out, int *n_owned)
{
  int lo = 0, hi = n;

  for (int i = 0; i < n; ++i)
  {
    if (!keep[i]) continue;
    if (owner[i] == my_rank) out[lo++] = i;
    else out[--hi] = i;
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
 * ── The logical index space ────────────────────────────────────────────────
 *
 * Slots cannot be assigned while the arrival counts are unknown, and the arrival
 * counts are unknown until the arrivals have been de-duplicated against the kept
 * entities. The rebuild therefore works in a LOGICAL space first:
 *
 *     [0, n_kept)                 kept, owned run then ghost run
 *     [n_kept, n_kept + n_new)    distinct arrivals, in arrival order
 *
 * and `final_of_logical[]` turns that into a pool slot once both counts exist.
 * The final pool layout is
 *
 *     [ kept owned | new owned | kept ghost | new ghost ]
 *
 * so that `n_local` remains an honest prefix.
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
 * @brief One contiguous run of one body's entities inside the arrival sequence.
 *
 * The multi-body rebuild has to separate the interleaved stream back into one
 * group per object. Deriving the groups at the receiver means extracting the body
 * id from every arrival (a strided read over the packed complexes), radix-sorting
 * all of them, and gathering again -- work to recover information the SENDER
 * already had.
 *
 * It is recoverable for free. Every rank packs body by body in the same object
 * order, and MPI_(Neighbor_)alltoallv concatenates the received blocks in target
 * order, so the arrival sequence is a concatenation of runs whose body order is
 * fixed by the code. Shipping the per-body counts as a third payload replaces the
 * sort with a prefix sum, and because the counts arrive in exactly the same order
 * as the entities they describe, a single running offset reconstructs everything.
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
 * @brief Resolves a body GID off the wire to this call's slot for that body.
 *
 * THE ONE PLACE the wire's body id becomes a local array index. The work list is
 * sorted for its first @p n_sorted entries -- those came from the ensemble's
 * gid-sorted directory -- and unsorted after them, where this call has been
 * appending the bodies arriving on this rank for the first time. So the search is
 * a binary one over the prefix and a scan over the tail, and the tail is short by
 * construction: it holds only bodies crossing a rank boundary this step.
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

  /* Then the arrival tail, which is appended in the order the runs decoded and
   * so is not sorted. It holds only the bodies arriving on this rank for the
   * FIRST time -- bodies crossing a rank boundary this step -- so it is short and
   * a scan beats keeping it ordered. */
  for (int i = n_sorted; i < n_work; ++i)
    if (work[i]->global_id == gid) return i;

  return MFTL_ERROR;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5b. The persistent workspace
 *
 * Everything a redistribution needs and nothing it produces. It exists because
 * the per-call form allocated and freed all of this on EVERY timestep:
 *
 *   - eight `nproc`-sized arrays, plus five `nproc` loops to initialise them;
 *   - one lazily-grown id list per touched destination, each jumping straight to
 *     a density-derived hint of up to a megabyte on first contact;
 *   - one send buffer per touched destination;
 *   - the whole of `rebuild_scratch`, fourteen arrays and two id tables.
 *
 * Measured on the unmodified code at four ranks: 66 to 99 allocations and 20 to
 * 39 MB requested PER TIMESTEP. docs/analysis.md §2.1 identifies
 * that churn as the reason a first benchmark run is slower than every repeat of
 * it -- the pages are handed back and re-faulted each time.
 *
 * It lives on the MFTL_Ensemble and NOT on the MFTL_Domain, which is the
 * distinction deferred issue 6 in CLAUDE.md turns on: the domain is read-only
 * during a redistribution and several ranks' worth of reasoning depends on that
 * staying true. The ensemble is already the mutable handle.
 *
 * It also gives the id tables' epoch mechanism something to amortise over for the
 * first time. The tables were built to be retired in O(1) by an epoch bump, and
 * then held in a stack local that was freed at the end of every call -- so the
 * epoch only ever saved a clear BETWEEN THE BODIES OF ONE CALL, never between
 * timesteps.
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
   * The bodies THIS CALL touches: the ones this rank already held, then the ones
   * it is about to receive for the first time.
   *
   * The ensemble's own list cannot serve, because a body arriving on a rank that
   * held nothing of it has to be created and inserted -- and an ordered insert
   * SHIFTS every slot above it, which would invalidate `states`, the arrival CSR
   * and every `runs[k].obj_idx` computed so far. So arrivals are appended here
   * instead and merged into the sorted directory once, at the very end, when
   * nothing is indexing by slot any more.
   */
  MFTL_Object **work;  ///< [cap_objects]
  body_state *states;  ///< [cap_objects]
  int *grp_off_v;  ///< [cap_objects + 1] arrival CSR offsets, vertices.
  int *grp_off_t;  ///< [cap_objects + 1] and triangles.
  int *cursor;  ///< [cap_objects] counting-sort cursors.

  /**
   * The per-body skip votes, INDEXED BY LOCAL SLOT -- one value per body this
   * rank holds, not one bit per body in the ensemble.
   *
   * It used to be a bitset over the id space, because it was the payload of an
   * MPI_BOR over comm_world and a message's index has to mean the same thing on
   * every rank. One bit per body made that as small as it could be, which is
   * why it was defensible -- and it was still the one array here proportional to
   * the ENSEMBLE rather than to what this rank holds, i.e. the one that grew
   * with P in a weak-scaling sweep.
   *
   * MFTLEnsembleReduceBodies() removes the constraint rather than the array: it
   * takes the vote over each body's HOLDERS, so the caller's buffer is local and
   * the id only appears on the wire, one per record. Doubles because that is the
   * one reduction mechanism for every per-body payload in the library, and a
   * vote is 0 or 1.
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
   * The ghost sync's owner-side id table.
   *
   * It lives here rather than in ghost_sync.c because it has exactly the same
   * problem the rebuild's tables had and the same solution: it was gt_init'd and
   * gt_free'd inside every call, so a table sized by a body's id range was
   * allocated and released once per TIMESTEP. Measured at 3.7 MB per step on one
   * 327 680-triangle body -- the last per-step allocation left on either path.
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
 * @brief Grows the per-body arrays, PRESERVING `states`.
 *
 * Separate from ws_bytes() because `states` is the one workspace array that has
 * to survive a growth WITHIN a call: phase 1 fills it for the bodies this rank
 * already held, and phase 4 then discovers arrivals for bodies it did not, which
 * makes the list longer after the useful contents are already in there. The other
 * two are filled after the growth, so they may be discarded.
 *
 * The tail is zeroed, which is what makes a body created on arrival start with
 * no kept lists and no arrival counts rather than with the previous call's.
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
 * @brief Sizes the workspace for this call. Every array only ever grows.
 *
 * A steady loop reaches its high-water marks in the first few timesteps and then
 * allocates nothing at all, which is the property this whole structure exists
 * for. The `nproc` block is rebuilt only if the communicator size changed, which
 * for a given ensemble it cannot.
 *
 * @param ws           The workspace.
 * @param nproc        Ranks in the communicator.
 * @param num_objects  Bodies this rank holds, before any arrival.
 * @param sum_verts   Total vertices over all bodies, for the kept-list arena.
 * @param sum_tris    Total owned triangles over all bodies.
 * @param max_verts   Largest body's vertex count.
 * @param max_tris    Largest body's triangle count.
 * @param tri_hint    First-touch size for a destination's triangle id list.
 * @param vert_hint   Ditto for vertices.
 */
static void ws_reserve(comm_workspace *ws, int nproc, int num_objects, int sum_verts, int sum_tris, int max_verts, int max_tris, int tri_hint,
                       int vert_hint)
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

  /* One block, two slices, both sized by what THIS RANK holds. Keeping them in
   * a single allocation with a single capacity is what makes "they are the same
   * length" unrepresentable rather than merely true -- and the reduction needs
   * them separate: it works in place, and the local votes have to survive it to
   * tell pass 2 which bodies this rank already did.
   *
   * Neither slice survives the call, so ws_bytes discarding contents on growth is
   * harmless. The one piece of state redistribution DOES carry between calls --
   * whether a body's layout has settled -- lives on the object itself
   * (MFTL_Object::partition_settled), which is why there is no invalidation rule
   * here any more. */
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

  /* ── What the workspace is proportional to ───────────────────────────────
   *
   * Read off the CAPACITIES rather than accumulated at each allocation site, so
   * this is what the rank currently holds rather than what it has ever asked
   * for -- a footprint, not churn. cs_alloc() already counts the churn.
   *
   * The split is the measurement a weak-scaling sweep needs: per-entity and
   * per-body are supposed to stay flat as P rises because the work and the
   * bodies per rank do, and per-rank is the one that does not. See
   * MFTL_RedistributeStats::ws_bytes_per_body.
   *
   * The per-destination payload buffers are deliberately NOT counted as
   * per-rank: they are sized by the traffic that actually flowed, are only
   * touched for destinations this rank really sent to, and so stay bounded by
   * the neighbourhood degree on the neighbour path. The arrays INDEXED by rank
   * are the part that is O(P) whether or not anything was sent. */
  MFTLCommStats.ws_bytes_per_rank = (long) ((size_t) ws->nproc * (1 + 2 * sizeof(mftl_int_buffer) + 3 * sizeof(MFTL_SendBuffer) + 2 * sizeof(int)));

  /* Every per-body array here is now sized by what THIS RANK holds, votes
   * included -- the vote was the last one indexed by the ensemble's id space,
   * and MFTLEnsembleReduceBodies() is what let it stop being. So in a
   * weak-scaling sweep this figure is expected to stay FLAT rather than to grow
   * with P; that expectation is the measurement, see
   * MFTL_RedistributeStats::ws_bytes_per_body. */
  MFTLCommStats.ws_bytes_per_body = (long) ((size_t) ws->cap_objects * (sizeof(MFTL_Object *) + sizeof(body_state) + 3 * sizeof(int)) + 2 * sizeof(int)
                                            + (size_t) ws->cap_votes * 2 * sizeof(double));

  MFTLCommStats.ws_bytes_per_entity = (long) ((size_t) ws->cap_arena_v * sizeof(int) + (size_t) ws->cap_arena_t * sizeof(int)
                                              + (size_t) ws->cap_verts * (2 * sizeof(int) + 1 + sizeof(uint32_t)) + (size_t) ws->cap_tris * (sizeof(int) + 1));
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
 * @brief Repairs the twins of exactly the half-edges that lost theirs.
 *
 * Same bucketing as MFTLRebuildTwins, restricted to the repair set: half-edges
 * are keyed by their ORIGIN vertex through an intrusive linked list, and the twin
 * of (u -> v) is found by walking the bucket of v for the edge that ends at u.
 * The set is closed under the twin relation (see the file header), so nothing
 * outside it can be a partner.
 *
 * Requiring the candidate to be UNLINKED keeps the relation symmetric on
 * non-manifold input, where three or more half-edges share an undirected edge:
 * the surplus is left as boundary, which MFTLCheckMeshIntegrity reports.
 *
 * `tris.neighbors` is written alongside `twin_idx`, because the rebuild emits the
 * derived topology inline rather than in a closing MFTLObjectBuildDerived() pass
 * (see step 8). Every half-edge in @p u_list already carries MFTL_ERROR there,
 * so only the pairs this function actually links need a store -- and it makes
 * exactly two per pair, both into lines it is writing anyway.
 *
 * @param obj    The mesh being rebuilt. `tris.he[].twin_idx` and the matching
 *               entries of `tris.neighbors` are written for every repaired pair.
 * @param u_list Candidate half-edges whose twin is still unresolved.
 * @param n_u    Number of entries in @p u_list.
 * @param v_head `[verts.n]` head of a per-vertex chain, scratch.
 * @param u_next `[n_u]` next-link of that chain, scratch.
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
 * @brief Rebuilds one object from its kept entities and its arrivals.
 *
 * The whole of section 5's contract lives here; see the file header for why none
 * of it involves a sort, a hash map or a full twin rebuild.
 *
 * @param obj     The mesh, rewritten in place from the kept entities plus
 *                everything that arrived. On return it has valid topology, valid
 *                positions and #MFTL_DERIVED_TOPOLOGY set -- emitted inline by
 *                steps 6 to 8 rather than by a closing MFTLObjectBuildDerived();
 *                the cached GEOMETRY is unspecified.
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

  /* ── 0. Bodies this rank no longer has any part of ───────────────────────
   *
   * A body whose entities have all left: nothing kept, nothing arriving. It is
   * emptied here and destroyed at the end of the call (phase 6), so it pays no
   * id-table epoch bump and no rebuild.
   *
   * The early return is placed before the timeline opens rather than after it,
   * because a body with no work is exactly where the instrumentation's own ten
   * clock reads would dominate what they are measuring. */
  if (n_kept_v == 0 && n_kept_t == 0 && bs->n_recv_v == 0 && bs->n_recv_t == 0)
  {
    MFTLCommStats.n_objects_empty_local++;

    if (n_old_v != 0 || n_old_t != 0)
    {
      /* The body was here last step and has now left entirely. Zero the counts,
       * then give the pool blocks back: they are at the high-water mark of
       * everything this body ever held here, and growth is a ratchet.
       *
       * This is the branch that recovers most of a generated ensemble's
       * footprint, and it is worth knowing how much. Rank 0 builds every body
       * whole; after the scatter it holds NOTHING of most of them, and each of
       * those lands here. Measured at 256 bodies, refine 3, np = 4: 80 MB
       * generated on rank 0 against 33.5 MB of pool capacity left there
       * afterwards. What this branch cannot reach is the bodies rank 0 keeps a
       * SLIVER of -- full capacity against a handful of triangles -- and that is
       * MFTLRedistributeSetShrinkSlack()'s half of the job. */
      MFTLObjectSetCounts(obj, 0, 0, 0, 0);
      MFTLObjectReleasePools(obj);
    }

    /* An empty mesh is trivially described. */
    obj->derived = MFTL_DERIVED_TOPOLOGY;
    /* MFTLObjectSetCounts above stamps the body; this call IS the rebuild the
     * stamp was asking for, so it is discharged here rather than by the caller. */
    obj->partition_dirty = MFTL_FALSE;
    return;
  }

  CS_TIMELINE(tm);
  MFTLCommStats.n_verts_kept += n_kept_v;
  MFTLCommStats.n_tris_kept += n_kept_t;
  MFTLCommStats.n_verts_received += bs->n_recv_v;
  MFTLCommStats.n_tris_received += bs->n_recv_t;

  /* ── 1. Gather the kept entities out of the pool ─────────────────────────
   *
   * Records and positions only. Everything else in the pool is either derived
   * (rebuilt below) or cached geometry the caller recomputes -- see the
   * postcondition in mftl_communication.h. Copying `tris.area/normal/center` here
   * would be 56 B per triangle of traffic for values that are overwritten before
   * they are read.
   *
   * It still runs before the id table is sized because it is also what finds the
   * largest kept global id, which is half of what gt_begin() needs to choose its
   * indexing mode. The other half, the entry count, is already known. */
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
    /* Every out_he_idx is re-derived below; a value carried over from the old
     * index space would point into someone else's fan. */
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

  /* Both numbers matter and they answer different questions. The ENTRY COUNT is
   * what a sparse table costs; the ID RANGE is what a direct-address table costs
   * and is also the bound its indexing needs. gt_begin() picks between them --
   * see comm_internal.h, where the measurement behind the ratio is recorded. */
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

  /* A vertex legitimately arrives once per sender that shares it, so the first
   * copy wins and the rest are dropped. The replicas are bit-identical: every
   * sender computed `owner_proc` from the same position with the same function,
   * and the position itself is the one the owner published. */
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

  /* Triangles cannot duplicate: only the OWNER of a triangle forwards it, and a
   * triangle has exactly one owner, so no two senders can carry the same one and
   * a kept triangle was owned by this rank rather than by a sender. The check
   * stays because a duplicate would otherwise corrupt the mesh silently; the
   * assertion states what the check is expected never to find. */
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

  /* The "is this arrival owned" bit is re-read from the payload rather than
   * cached: the arrivals are visited in the same order, so the j-th distinct one
   * is the same entity in both passes. */
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

  /* GEOMETRIC, then exact. MFTLObjectReserve grows to exactly what is asked for
   * -- deliberately, because a mesh of stable size must not re-fault its pages
   * every timestep -- and MFTLObjectSetCounts inherits that. Under advection the
   * counts jitter by a few entities every step, so the exact rule reallocates a
   * whole pool block for a one-entity rise and copies EVERY sub-array forward:
   * 64 B per vertex and 140 B per triangle, almost all of it overwritten before
   * it is read. Asking for a doubled capacity first turns the reserve inside
   * SetCounts into an early return and makes the reallocation O(log n) over the
   * run instead of once per jitter.
   *
   * The policy stays local to this call site on purpose. `MFTLObjectReserve`'s
   * exactness is a contract other callers rely on, and src/object/object.c
   * remains the only file that writes a capacity.
   *
   * THIS IS WHY THE SHRINK IN PHASE 5b HAS A GATE OF FOUR AND NOT TWO. What this
   * line establishes is `capacity` in [n, 2n) -- MFTLGrowCapacity doubles from
   * below n, so its result is under 2n whatever it started from. A shrink gate of
   * `capacity > 2n` is therefore met by ANY decrease, and the two would take
   * turns reallocating both pools every timestep on a body whose count moves by
   * two entities. At four the count must halve first, so neither can see the
   * other. Derivation and the worked sequence: MFTL_SHRINK_SLACK_DEFAULT. */
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

  /* THE TWO CLEARS ARE NOT THE SAME STATEMENT, WHICH IS WHY ONLY ONE OF THEM IS
   * COMPILED IN RELEASE.
   *
   * The TRIANGLE clear is load-bearing: `t_final_of_old[tw / 3] == MFTL_ERROR`
   * is how the relink below retires a twin whose triangle was NOT kept, so its
   * sentinel is read on the production path.
   *
   * The VERTEX clear is only ever read by the assertion in that same loop. A
   * kept triangle's corners are always kept -- collect_destinations() sets
   * keep_v[v0..v2] inside the same `r == my_rank` branch that sets keep_t[t] --
   * so every slot the relink reads was written by the loop below it, and in
   * Release the sentinel has no reader at all. n_old_v * 4 B of stores per
   * rebuilt body per step, for an assert. */
#ifndef NDEBUG
  for (int v = 0; v < n_old_v; ++v) s->v_final_of_old[v] = MFTL_ERROR;
#endif
  for (int t = 0; t < n_old_t; ++t) s->t_final_of_old[t] = MFTL_ERROR;
  for (int i = 0; i < n_kept_v; ++i) s->v_final_of_old[bs->kept_v[i]] = s->v_final_of_logical[i];
  for (int i = 0; i < n_kept_t; ++i) s->t_final_of_old[bs->kept_t[i]] = s->t_final_of_logical[i];

  /* ── 6. Re-link the kept half-edges ──────────────────────────────────────
   *
   * Four things happen in one pass over 48 bytes per kept triangle: the target
   * vertex index is moved into the new index space, the twin is either remapped
   * or retired into the repair list, `out_he_idx` is re-derived, and the two
   * DERIVED arrays are emitted. The repair list therefore costs nothing to build
   * -- the loop already knows which twins did not survive.
   *
   * WHY THE DERIVED ARRAYS ARE WRITTEN HERE AND NOT BY A CLOSING PASS.
   *
   * `tris.verts` and `tris.neighbors` are a fixed function of the half-edge
   * records at the same index (build_topology_arrays, src/object/derived.c):
   *
   *     verts[3t+k]     == he[3t+(k+2)%3].vertex_idx
   *     neighbors[3t+k] == he[3t+k].twin_idx / 3
   *
   * and BOTH are already in registers here -- `tw_tri` IS `twin_idx / 3`, and the
   * three remapped targets are the rotation this loop forms two lines later for
   * `out_he_idx`. A closing MFTLObjectBuildDerived() re-reads 48 B and writes
   * 24 B per triangle to recompute them; emitting them costs 24 B of stores into
   * lines this loop is already sequential over, and removes a whole pass over
   * the pool.
   *
   * Every pool slot is covered exactly once, and every write is a forward
   * stream: the kept-owned run is [0, n_kept_t_owned), the kept-ghost run is
   * contiguous at `L + n_new_t_owned`, and the arrivals below fill the two
   * remaining runs through their own cursors. kept + new == n_t.
   *
   * THE PRICE IS THAT THE ROTATION IS NOW SPELLED IN TWO PLACES, and what stops
   * the two drifting is inv_derived_matches_topology()
   * (tests/test_mesh_invariants.h): it re-derives both arrays from the records
   * and compares element for element, it is inside inv_mesh_is_sound, and
   * tests/test_redistribution.c runs it against every redistributed mesh. */
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
        /* A half-edge index carries its corner slot in the low position, so the
         * remap keeps h % 3 and moves only the triangle part -- which is exactly
         * the implicit-indexing invariant being maintained through the change. */
        h3[k].twin_idx = 3 * tw_tri + tw % 3;
      }
      else
      {
        h3[k].twin_idx = MFTL_ERROR;
        ib_add(&s->u_list, 3 * slot + k);
      }

      /* One store for both branches: `tw_tri` is `twin_idx / 3` on the retained
       * branch and MFTL_ERROR on the other, which is exactly what
       * build_topology_arrays() writes in each case. repair_twins() overwrites
       * the entries whose half-edge finds a new partner. */
      tn[3 * slot + k] = tw_tri;
    }

    /* origin(3s+k) == target(3s+(k+2)%3), and all three targets are final now. */
    vr[h3[2].vertex_idx].out_he_idx = 3 * slot + 0;
    vr[h3[0].vertex_idx].out_he_idx = 3 * slot + 1;
    vr[h3[1].vertex_idx].out_he_idx = 3 * slot + 2;

    /* The same rotation, deliberately spelled the way build_topology_arrays()
     * spells it: slot k is the ORIGIN of half-edge 3s+k. */
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

  /* No rebuild: steps 6 and 7 emitted `tris.verts`/`tris.neighbors` for every
   * slot as they went and repair_twins() finished the neighbours it relinked.
   * MFTLObjectSetCounts() cleared the bit on the way through, so it is restored
   * here by hand -- the one line MFTLObjectBuildDerived() would still contribute. */
  obj->derived = MFTL_DERIVED_TOPOLOGY;

  /* Whatever asked for this rebuild has had it. Cleared LAST, because
   * MFTLObjectSetCounts sets the stamp on the way through and would otherwise
   * leave the body permanently dirty. */
  obj->partition_dirty = MFTL_FALSE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Multi-body: separating the interleaved arrival stream
 * ═══════════════════════════════════════════════════════════════════════════ */


/**
 * @brief The per-destination body HEADER, appended once per body that contributed.
 *
 * `{gid, n_tris, n_verts}` used to be the whole of it: which body the next run of
 * entities belongs to and how long the run is. The four that follow are what a
 * body arriving on a rank that held NOTHING of it cannot otherwise know.
 *
 * `next_*_gid` are not cosmetic. src/communication/ sizes its arrival lookup
 * tables from them, and MFTLUpdateGhostPositions() sizes the owner's id table
 * the same way -- a body created with zero counters would build a table too
 * small for the ids it is holding, which is a wrong answer rather than a slow
 * one. `global_n_*` is the count every consumer of the body reports.
 *
 * All four are GLOBAL properties, so every rank holding the body already agrees
 * about them; a receiver that already had the body therefore ASSERTS them rather
 * than applying them, which turns the payload into an invariant check on the
 * path that does not need it.
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
 * @brief Turns the runs into one grouped index list per object, without sorting.
 *
 * Counting sort over the runs: count each object's entities, prefix-sum into
 * per-object offsets, then scatter the arrival indices. @p out_off is the
 * resulting [num_objects + 1] offset table into @p out_idx.
 *
 * @param      runs        Per-sender runs of arriving entities.
 * @param      n_runs      Number of runs.
 * @param      num_objects Objects in the ensemble.
 * @param      use_tris    `true` to group the triangle runs, `false` for vertices.
 * @param[out] out_off     `[num_objects+1]` CSR offsets into @p out_idx.
 * @param[out] out_idx     Run indices, grouped by object.
 * @param      cursor      `[num_objects]` scratch write cursors.
 * @return The number of run indices written to @p out_idx.
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

  /* Runs of the same object are visited in arrival order, so its slice stays in
   * arrival order too -- which is what makes "the first copy of a vertex wins"
   * a well-defined rule inside rebuild_object. */
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
 * MOST BODIES DO NOT MOVE BETWEEN TWO TIMESTEPS, AND A BODY THAT DID NOT MOVE
 * NEEDS NOTHING DONE TO IT.
 *
 * A rank's local state after a redistribution is a pure function of two things:
 * the global topology, and the owner map. Read the halo rule backwards and it is
 * a definition of the OUTPUT rather than a description of a procedure --
 *
 *     rank r holds triangle t iff r owns t, or owns one of its three corners,
 *     or owns one of its three edge neighbours
 *
 * -- together with the corners of everything it holds. Positions are not an
 * input. They DECIDE the owner map (compute_owners reads verts.pos and
 * tris.center), but once the map is fixed, moving a vertex inside its own cell
 * changes nothing about who holds what.
 *
 * So if neither input changed, the whole of section 5 is a very expensive way of
 * reproducing the pool that is already there. The estimate in
 * docs/analysis.md §2.4 prices what a skip removes at 68 %
 * of the phase: kept gather, destination analysis, relink, topology arrays,
 * arrivals, pool growth, place-back and twin repair. What survives is the
 * ownership scan (which is how the decision is made) and the collectives.
 *
 * ── THE PRECONDITION IS NOT "DID TOPOLOGY CHANGE" ──────────────────────────
 *
 * A topology change under the remesher's freeze discipline is invisible to the
 * halo rule: an unfrozen entity has no ghost corner and no ghost edge neighbour,
 * so every triangle an operator builds out of it has all seven halo clauses
 * resolving to the acting rank, and no other rank's halo can name it. The
 * library already relies on this -- MFTLRemeshMigrateShifted skips an entire
 * migration on one MPI_LOR of "did anything move", on the hot path, every
 * phase 0 of every remeshing call.
 *
 * What a skip DOES break is global ids. The spanning remesher's closing
 * redensification renumbers owned entities and deliberately leaves ghost
 * TRIANGLE ids alone, because redistribution drops and re-receives every ghost
 * triangle and that is what repairs them. A skip removes the repair. Worse, the
 * skip's trigger and the hazard coincide on the most common case rather than a
 * corner one: a converged body whose smoother moved no vertex across a cut plane
 * gets renumbered and then skipped, every step, and the stale ids sit in the
 * pool colliding with fresh ones until something finally reads them.
 *
 * The minimal correct precondition is therefore
 *
 *     no entity of the body changed owner, AND nothing rewrote the global ids
 *     or the flags of entities that other ranks hold copies of
 *
 * and the second clause is `obj->partition_dirty` (MFTLObjectMarkPartitionDirty,
 * mftl_object.h), stamped at the choke point every entity creation and
 * destruction already passes through plus the two renumbering sites and the one
 * flag-writing site. Full derivation in docs/analysis.md §2.4.
 *
 * ── WHY IT TAKES TWO PASSES AND ONE COLLECTIVE ─────────────────────────────
 *
 * The decision cannot be local. A rank with zero local changes is still the
 * RECEIVER of a change made elsewhere -- a triangle whose centre crossed into
 * its cell was owned by a neighbour, and if it was not in this rank's halo, this
 * rank sees nothing at all. So every rank must agree, per body.
 *
 * Agreeing per body inside the object loop would be num_objects collectives per
 * timestep, which at a thousand droplets costs more than the phase it is saving.
 * ONE per-body reduction over the bodies' holders (MFTLEnsembleReduceBodies(),
 * one MAX per body) is the alternative, and it forces the loop in two:
 *
 *   pass 1  classify ownership for EVERY body, and do the full analysis and
 *           packing immediately for every body THIS rank saw change;
 *   vote    one per-body reduction;
 *   pass 2  the bodies that were clean here and dirty somewhere else.
 *
 * Pass 2 does not need pass 1's owner arrays kept alive, which is the thing that
 * would otherwise cost O(sum_verts + sum_tris) of new storage: compute_owners is
 * IDEMPOTENT, so pass 2 simply re-runs it on the few bodies it reaches. That set
 * is exactly the bodies whose motion happened on a neighbour's side of the
 * boundary, and it is small for the same reason the skip pays at all.
 *
 * ── A BODY MUST BE CLEAN TWICE BEFORE IT IS SKIPPED, AND THAT IS WHAT MAKES
 *    THE SKIP EXACT RATHER THAN MERELY EQUIVALENT ─────────────────────────
 *
 * rebuild_object is NOT idempotent. Its output layout is
 *
 *     [ kept owned | new owned | kept ghost | new ghost ]
 *
 * and "kept" means "a corner of one of MY triangles": an owned vertex whose
 * incident triangles all belong to other ranks is not kept, it is dropped and
 * re-received, and it therefore lands at the END of the owned run rather than
 * wherever it was. So running the rebuild twice on an unchanged body moves
 * entities the FIRST run left in place -- the second run is the one that reaches
 * a fixed point.
 *
 * A skip on a layout that has not reached that fixed point is still a correct
 * partition -- same entities, same ownership, same halo, same ids, same
 * positions, same topology -- but it is a different SLOT ORDER than a rebuild
 * would have produced, and slot order is exactly what the acceptance test
 * compares (`parallel_advection_redistribute_list --digest` reports a set digest
 * and an order digest, and the second is there to catch this class of
 * difference). Measured: 8 domain-spanning bodies at np = 4 diverge in the order
 * digest on the FIRST step, because the initial scatter's output is not a fixed
 * point either.
 *
 * One extra bit per body closes it. Let D be "some rank saw this body change":
 *
 *     D now  D before   action              layout afterwards
 *     -----  --------   ------------------  -----------------------------
 *       1       -       rebuild             not necessarily a fixed point
 *       0       1       rebuild ("settle")  a fixed point, by the argument above
 *       0       0       SKIP                unchanged, so still a fixed point
 *
 * -- i.e. skip iff `!D_now && !D_prev`, and the new "before" is simply `D_now`.
 * The cost is ONE extra rebuild per dirty->clean transition, so a body that sits
 * still for a run of L steps loses one skip out of L; a body that alternates
 * never skips, which is the right answer for it anyway. What it buys is that the
 * order digest matches too, and the optimisation becomes invisible rather than
 * merely harmless.
 *
 * "D before" is the only state redistribution carries between calls, and it
 * lives on the object (MFTL_Object::partition_settled) rather than in the
 * workspace: a body created on arrival starts unsettled, which is the
 * conservative verdict, without un-settling any other body.
 * ════════════════════════════════════════════════════════════════════════════ */

/** @brief Whether unchanged bodies may be left alone. See MFTLRedistributeSetSkipUnchangedBodies(). */
static bool g_skip_unchanged = MFTL_TRUE;

void MFTLRedistributeSetSkipUnchangedBodies(bool on) { g_skip_unchanged = on; }

bool MFTLRedistributeGetSkipUnchangedBodies(void) { return g_skip_unchanged; }

/**
 * @brief Waste factor at which a rebuilt body's pools are given back. 0 = off.
 *
 * Deliberately NOT part of the object-list checksum: this changes no
 * communication and no mesh, only how much memory a rank holds, so two ranks
 * disagreeing about it is a legitimate local policy rather than a deadlock.
 */
static double g_shrink_slack = 0.0;

void MFTLRedistributeSetShrinkSlack(double slack) { g_shrink_slack = (slack > 0.0) ? slack : 0.0; }

double MFTLRedistributeGetShrinkSlack(void) { return g_shrink_slack; }

/**
 * @brief Destination analysis, kept lists and packing for ONE body.
 *
 * Everything the per-object loop does after the ownership classification, in one
 * place so that both passes run identical code rather than two copies that can
 * drift. It consumes `ws->tri_owner` / `ws->vert_owner` / `ws->vert_gid`, so
 * compute_owners must have run on @p obj immediately before.
 *
 * @param      ws           The workspace; supplies every scratch array.
 * @param      obj          The mesh. Its #MFTL_Object::global_id is the body id
 *                          on the wire -- the caller's slot for it is not
 *                          transmitted and is not needed here.
 * @param      my_rank      This rank.
 * @param      dd           Per-destination vertex de-duplication state.
 * @param[out] bs           This body's kept lists and counts.
 * @return 1 if any destination fell outside the 26-neighbourhood, which forces
 *         the exchange onto the global communicator.
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

  /* Only the ranks this object actually reached -- typically fewer than 27, not
   * nproc. That difference is the whole point of the touched list. */
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
 * @brief Debug-only: do all ranks agree about the body-id space and the switches?
 *
 * Bodies are named on the wire by #MFTL_Object::global_id, and those ids are
 * handed out by MFTLEnsembleAddObject() in add order. So "the ranks agree about
 * which id names which body" reduces to "every rank added the same bodies in the
 * same order", and the observable consequence of that is the SIZE of the id
 * space. A disagreement there silently merges entities into the wrong body,
 * which is why it is verified rather than merely documented.
 *
 * The check is deliberately CONSTANT SIZE rather than a hash of every id. The
 * per-rank id list is about to stop being the whole ensemble's, so a checksum
 * over it would be asserting a property this design is removing; what survives is
 * the id space and the collective switches. That this rank's own ids are dense in
 * list order is a LOCAL statement and is checked where it is used, in
 * slot_of_body(), which needs no communication to do it.
 *
 * The skip switch rides along for free. It changes how many collectives the call
 * makes, so two ranks disagreeing about it is an immediate deadlock with no
 * diagnostic -- exactly the failure this reduction already exists to report, and
 * one more term costs nothing. MFTLRedistributeSetShrinkSlack() deliberately does
 * NOT ride along: it changes no message and no mesh, only how much memory a rank
 * holds, so disagreeing about it is legitimate local policy.
 *
 * @param dom             Supplies the communicator.
 * @param n_bodies_global Size of the body-id space.
 */
static void check_object_lists_agree(const MFTL_Domain *dom, int n_bodies_global)
{
  /* The LOCAL body count is deliberately not in here. It legitimately differs
   * between ranks -- that is the entire point of a sparse object list -- so
   * folding it in would turn the change this check exists to protect into a
   * failure of the check itself. */
  unsigned sig = (unsigned) n_bodies_global * 2u + (g_skip_unchanged ? 1u : 0u);

  /* {sig, ~sig} under one MPI_MIN: a minimum detects a disagreement in EITHER
   * direction, which a single reduced value cannot. */
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
 * @brief The one redistribution implementation; both public entry points call it.
 *
 * Every local object is repartitioned in ONE communication round: the packed
 * streams of all bodies are interleaved into one pair of messages and separated
 * again at the receiver from a third payload of per-body headers, so the message
 * COUNT is independent of the number of bodies.
 *
 * @param ens  The ensemble: its domain, its local object list and its persistent
 *             workspace.
 * @param maps `[ens->n_local_objects]` explicit owner maps in local-slot order,
 *             or NULL for the geometric rule.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the exchange failed.
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
    /* Nothing can move, so the ownership half of the postcondition is already
     * true: every entity is owned and the prefix covers the pool. Only the
     * derived-array half may still need establishing, and only for an object
     * that never had them built -- checking the bit keeps the serial path O(1)
     * per timestep while keeping the postcondition identical at every nproc. */
    for (int o = 0; o < num_objects; ++o)
    {
      MFTL_Object *const obj = objects[o];
      MFTL_ASSERT(obj->verts.n_local == obj->verts.n && obj->tris.n_local == obj->tris.n && "redistribute: single-rank object holds ghosts");
      if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    }

    /* The same postcondition as the parallel path: a rank has an object for a
     * body exactly when it has entities of it. At one rank "this rank" is every
     * rank, so a body with nothing in it is a body nobody holds. Skipping this
     * would make the invariant true at nproc > 1 and false at nproc == 1, which
     * is the shape of difference that only ever surfaces in someone's serial
     * reproduction of a parallel bug. Walked backwards so an erase does not move
     * a slot this loop has yet to reach. */
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

  /* The per-body header payload ALWAYS travels, even for a single-body
   * ensemble. A rank that holds nothing of a body it is about to receive has no
   * object to put the arrivals in, and the header is what lets an arrival name
   * its body -- which is what makes creation on arrival possible at all. It also
   * keeps the payload count a constant, and the payload count is collective
   * state: two ranks posting a different number of Alltoallv calls deadlock with
   * no diagnostic.
   *
   * The workspace is the ensemble's, so a steady loop reaches its high-water
   * marks in the first few timesteps and then allocates nothing here at all. */
  if (ens->comm_workspace == MFTL_NULL) ens->comm_workspace = MFTLCallocOrDie(1, sizeof(comm_workspace), "MFTL communication workspace");
  comm_workspace *const ws = (comm_workspace *) ens->comm_workspace;

  /* ── Two different body indexes, and the difference is the whole design ──
   *
   * A LOCAL SLOT indexes this call's work list. It is private to this rank and
   * changes whenever a body arrives or leaves. Every per-body array in the
   * workspace is one of these, the skip votes included -- they were the last
   * that was not.
   *
   * A BODY ID indexes anything that is a MESSAGE -- the body headers on the
   * wire, one record of a per-body reduction. Two ranks holding different bodies
   * agree about nothing else. (An explicit owner map is indexed by local slot:
   * it never crosses a rank, see MFTLRedistributeWithOwners().) */

  /* ── Whether the skip vote can stay on the neighbour graph ───────────────
   *
   * TAKEN HERE, BEFORE THE CLASSIFICATION, AND THAT IS NOT A STYLE CHOICE.
   * MFTLEnsembleBodyHoldersFitTheGraph() reads each body's ghost `owner_proc` to
   * find its other holders, and compute_owners() below OVERWRITES every
   * entity's owner_proc with its DESTINATION rank. After pass 1a the pools
   * therefore describe the partition that is about to exist, while the vote has
   * to reach the ranks holding the body NOW. Asking one line later gives a
   * plausible answer to the wrong question -- and one that is wrong only when
   * something moved far, i.e. exactly when the vote matters.
   *
   * The verdict is handed to the reduction at phase 1b, which agrees it together
   * with the certificate it takes after its exchange -- that one reads only the
   * exchanged records, so the rewritten pools cannot mislead it. */
  const bool holders_fit = MFTLEnsembleBodyHoldersFitTheGraph(ens);

  /* At least one slot even with nothing held: the arrival path grows the arrays
   * from here, and a NULL base pointer is not a valid thing to grow from. */
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

  /* The work list starts as what this rank already holds; phase 4 appends the
   * bodies arriving here for the first time. Everything from here on indexes
   * `work`, never the ensemble's list, so an insertion into the sorted directory
   * -- which shifts slots -- can be deferred to the very end. */
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

  /* ── Phase 1a: ownership everywhere, the rest where THIS rank sees change ─
   *
   * See section 6b for why the object loop is in two halves and why the split
   * costs no per-body storage. */
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

    /* Nothing of this body is here, so this rank has no opinion. Another rank
     * that IS holding it votes for both of them. */
    if (obj->tris.n < 1 || obj->verts.n < 1) continue;

    CS_TIMELINE(tm);
    int dirty = compute_owners(obj, maps ? &maps[o] : MFTL_NULL, tri_owner, vert_owner, vert_gid, MFTL_TRUE);
    CS_LAP(tm, t_owners);

    /* The other half of the precondition (section 6b), plus one guard: a body
     * whose derived arrays were never built has no partition to preserve, and
     * collect_destinations' assertion on them must not be lost on the skip
     * path. */
    dirty |= obj->partition_dirty;
    dirty |= !MFTLObjectHasTopology(obj);
    if (!g_skip_unchanged) dirty = 1;

    if (!dirty) continue;

    dirty_local[o] = 1.0;
    requires_global |= analyze_and_pack_body(ws, obj, my_rank, &dd, bs);
  }

  /* ── Phase 1b: agree, per body, over each body's HOLDERS ─────────────────
   *
   * BLOCKING, and there is nothing to overlap it with: pass 1 has finished and
   * pass 2 consumes the answer.
   *
   * IT IS NO LONGER A comm_world REDUCTION OVER THE ID SPACE. It used to be one
   * MPI_BOR of a bitset with one bit per body in the ensemble, which is as small
   * as that shape can be made and still O(bodies) on every rank -- so in a weak
   * sweep, where the body count grows with the rank count, it grew with P for a
   * fixed amount of work per rank. Only the ranks HOLDING a body can have seen
   * it change, and only they will rebuild it, so the vote is exactly a
   * per-holder reduction and MFTLEnsembleReduceBodies() is where that decision
   * lives. On the neighbour route the payload is two doubles per body HELD.
   *
   * The pre-exchange half of the route test was taken before the classification
   * (see the note where `holders_fit` is taken); the reduction agrees it with
   * its own certificate in the one int it costs anyway.
   *
   * Not in place: pass 2 needs to know which bodies THIS rank already did, which
   * is exactly the information a reduction in place would destroy -- so the
   * local votes are copied into the result buffer and the reduction runs on the
   * copy.
   *
   * It is the FIRST collective after the object loop, so it inherits t_route's
   * other job -- absorbing the load imbalance between ranks. The route reduction
   * that follows then finds everyone already synchronised and costs pure
   * latency. t_skip_vote against t_route is the measurement. */
  CS_BEGIN(ts_vote);
  {
    static const MFTL_BodyReduceOp vote_op = MFTL_BODY_REDUCE_MAX;
    memcpy(dirty_global, dirty_local, (size_t) n_pre * sizeof(double));
    MFTLEnsembleReduceBodiesPrescanned(ens, dirty_global, 1, &vote_op, holders_fit);
  }
  CS_END(ts_vote, t_skip_vote);

  /* ── Phase 1c: everything pass 1a left, that is not being skipped ────────
   *
   * Two kinds of body reach this loop: the ones another rank saw change while
   * this one did not, and the ones nobody saw change but whose layout has not
   * reached its fixed point yet (section 6b). */
  for (int o = 0; o < n_pre; ++o)
  {
    MFTL_Object *const obj = ws->work[o];

    if (dirty_local[o] != 0.0) continue;  // already done in pass 1a
    if (dirty_global[o] == 0.0 && obj->partition_settled) continue;  // skipped

    if (obj->tris.n < 1 || obj->verts.n < 1) continue;  // nothing here to send

    /* compute_owners is idempotent, so this rewrites the same values pass 1a
     * wrote and reports no change; the counters it feeds must not be told about
     * the same entities twice. */
    CS_TIMELINE(tm);
    (void) compute_owners(obj, maps ? &maps[o] : MFTL_NULL, tri_owner, vert_owner, vert_gid, MFTL_FALSE);
    CS_LAP(tm, t_owners);

    requires_global |= analyze_and_pack_body(ws, obj, my_rank, &dd, &ws->states[o]);
    MFTLCommStats.n_objects_reclassified++;
    if (dirty_global[o] == 0.0) MFTLCommStats.n_objects_settled++;
  }

  /* Nothing to release: every array above belongs to the workspace and is reused
   * by the next call. The id lists keep their capacity too -- ib_reset() emptied
   * them in the packing loop without giving the memory back. */

  /* ── Phase 2: agree on the communication path ────────────────────────────
   *
   * Not a local decision: if A must send to a non-neighbour B, then B has to be
   * listening on the global path, and B cannot know that by itself.
   *
   * BLOCKING, deliberately. It is one int, so the cost is pure latency, and there
   * is nothing left to overlap it with -- the send buffers are already packed and
   * the rebuild cannot begin before the data arrives. Tearing the old pools down
   * would give it something to hide behind; nothing is torn down here, which is
   * the whole point of section 5. */
  CS_BEGIN(ts_route);
  int global_decision = 0;
  MPI_Allreduce(&requires_global, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);
  CS_END(ts_route, t_route);
  if (global_decision) MFTLCommStats.n_route_global++;
  else MFTLCommStats.n_route_neighbor++;

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

  /* MFTLExchangeStart has already copied everything into its own storage, so the
   * per-destination buffers are dead -- but they are the right size for the next
   * timestep, and this loop used to free them and pay the first-touch faults
   * again on every call. They now persist; only the fill level is reset, at the
   * top of the next call. */

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
      /* ── A body arriving where this rank held nothing of it ─────────────
       *
       * The body is created here and appended to the WORK list -- not inserted
       * into the sorted directory, because an ordered insert shifts every slot
       * above it and phase 5 is about to index by slot. The merge happens once,
       * at the end, when nothing is indexing any more.
       *
       * It starts unsettled (MFTLObjectCreate zeroes `partition_settled`),
       * which is the conservative verdict and exactly what the "clean twice"
       * rule wants for a body this rank has never rebuilt. */
      MFTL_ASSERT(gid >= 0 && gid < ens->n_bodies_global && "redistribute: received a body gid outside the id space");

      ws_bodies_reserve(ws, n_work + 1);
      MFTL_Object *const arrived = MFTLObjectCreate(gid);
      arrived->domain = domain;
      arrived->derived = MFTL_DERIVED_TOPOLOGY;  // an empty mesh is trivially described

      /* Clear the slot. The workspace is persistent and only ever grows, so a
       * slot past this call's `n_pre` still holds whatever the PREVIOUS call
       * put there -- a kept-list pointer into the arena and a non-zero kept
       * count, which rebuild_object() would then read against a body that has
       * no pool at all. The setup memset covers [0, n_pre) only, because that
       * is all this call knew about at the time. */
      memset(&ws->states[n_work], 0, sizeof(body_state));

      ws->work[n_work] = arrived;
      slot = n_work++;
      MFTLCommStats.n_objects_created++;
    }

    /* The four global fields. A body this rank already had must agree with the
     * sender about every one of them -- they are properties of the whole mesh,
     * not of a partition -- so applying them unconditionally is correct, and
     * checking them first is free evidence that it is. */
    MFTL_Object *const target = ws->work[slot];
    MFTL_ASSERT((target->verts.n == 0 && target->tris.n == 0)
                || (target->next_vertex_gid == tp[MFTL_BODY_HDR_NEXT_VGID] && target->next_triangle_gid == tp[MFTL_BODY_HDR_NEXT_TGID]
                    && target->global_n_vertices == tp[MFTL_BODY_HDR_GLOBAL_V] && target->global_n_triangles == tp[MFTL_BODY_HDR_GLOBAL_T])
                && "redistribute: two ranks disagree about a body's global counters");

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

    /* An ARRIVAL has no vote slot and needs none. It is here because an entity
     * of it crossed into this rank's cell, so some rank saw it change and the
     * body is dirty by construction -- and MFTLObjectCreate() leaves
     * `partition_settled` false, so the clause below is already decided. */
    const bool clean = (o < n_pre) && (dirty_global[o] == 0.0);

    if (g_skip_unchanged && clean && obj->partition_settled)
    {
      /* Nobody saw this body change, so the pool it is already holding IS the
       * answer -- see section 6b. The topology bit was checked in pass 1a and
       * vertex motion does not invalidate it, so there is nothing to rebuild. */
      MFTLCommStats.n_objects_skipped++;

      if (obj->tris.n < 1 || obj->verts.n < 1)
      {
        obj->derived = MFTL_DERIVED_TOPOLOGY;  // an empty mesh is trivially described
      }
      continue;
    }

    rebuild_object(obj, &ws->states[o], (const MFTL_PackedVertex *) xfer[0].recv_buf, (const MFTL_PackedTriangleComplex *) xfer[1].recv_buf, my_rank,
                   scratch);
  }

  /* ── Phase 5b: give back capacity this rank has outgrown ─────────────────
   *
   * Pools are grow-only (mftl_object.h), so a rank whose share of a body just
   * SHRANK is still holding the block it needed at its high-water mark. After
   * generation that is the whole mesh: MFTLGenerateSphere() builds every body
   * whole on rank 0, and the scatter above leaves rank 0 with a sliver of each
   * in a block sized for all of it.
   *
   * A separate pass rather than a line inside rebuild_object(): here no pointer
   * into any pool is live, so the reallocation needs no argument about what it
   * dangles, and skipped bodies are covered without one either. They can never
   * fire anyway -- neither their count nor their capacity moved since the
   * rebuild that last offered them -- but not having to know that is the point.
   *
   * The gate composes with the pre-grow in rebuild_object() rather than fighting
   * it: that leaves capacity in [n, 2n), and this fires only above 4n, so a body
   * whose count merely jitters is never touched by either. See
   * MFTL_SHRINK_SLACK_DEFAULT for why 4 is a floor and not a preference. */
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

  /* What the next call needs to know: a body rebuilt because something CHANGED
   * has a layout that is not yet a fixed point, so it gets one more rebuild
   * before it may be skipped. See section 6b.
   *
   * On the object rather than in a workspace bitset, so a body carries its own
   * answer. A body that arrives on this rank for the first time starts unsettled
   * because MFTLObjectCreate() zeroes it, which is exactly the conservative
   * answer -- and it is arrived at without un-settling every other body, which is
   * what a bitset indexed by list position had to do whenever the list changed. */
  for (int o = 0; o < n_work; ++o) ws->work[o]->partition_settled = (o < n_pre) && (dirty_global[o] == 0.0);

  /* ── Phase 6: the local directory catches up ─────────────────────────────
   *
   * Every index into `states`, `runs` and the arrival CSR is dead by now, so this
   * is the first point at which the sorted list may be disturbed -- and it is why
   * arrivals were parked in the work list rather than inserted when they were
   * discovered.
   *
   * A body this rank ends up holding NOTHING of is destroyed outright, not kept
   * as an empty shell. rebuild_object() has already given its pool blocks back;
   * what goes here is the 264-byte object and the pointer to it, which is the
   * whole per-body footprint a rank used to carry for every body in the ensemble
   * whether it held any of it or not.
   *
   * Its skip state goes with it. If the body comes back, it is created unsettled
   * and gets its settling rebuild -- which is correct rather than merely safe:
   * a rank that has just received a body has never rebuilt it here. */
  for (int o = n_pre; o < n_work; ++o)
  {
    MFTL_Object *arrived = ws->work[o];
    if (arrived->tris.n > 0 || arrived->verts.n > 0)
    {
      MFTLEnsembleInsertLocalObject(ens, arrived);
    }
    else
    {
      /* Addressed to this rank and yet empty: every entity of the run was a
       * duplicate this rank kept. Nothing to hold, so nothing to keep. */
      MFTLObjectDestroy(&arrived);
      MFTLCommStats.n_objects_destroyed++;
    }
    ws->work[o] = MFTL_NULL;
  }

  for (int o = 0; o < n_pre; ++o)
  {
    MFTL_Object *gone = ws->work[o];
    if (gone->tris.n > 0 || gone->verts.n > 0) continue;

    /* By GID, not by `o`: the insertions above have already shifted the slots. */
    const int slot = MFTLEnsembleFindLocalSlot(ens, gone->global_id);
    MFTL_ASSERT(slot != MFTL_ERROR && "redistribute: a body in the work list is not in the directory");
    MFTLEnsembleEraseLocalSlot(ens, slot);
    MFTLObjectDestroy(&gone);
    MFTLCommStats.n_objects_destroyed++;
    ws->work[o] = MFTL_NULL;
  }

  /* Only what the exchange layer allocated for this call. Everything else --
   * the rebuild scratch with its two id tables, the run table, the arrival CSRs
   * -- belongs to the workspace and is exactly the right size for the next call.
   * The id tables in particular: they were built to be retired by an epoch bump
   * rather than a clear, and until now they were thrown away every timestep, so
   * that mechanism had nothing to amortise over. */
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

  /* An ensemble with no bodies at all is a no-op, not an error. The test is
   * GLOBAL: a rank that holds no bodies still has to enter the exchange, because
   * another rank may be sending it one. */
  if (ensemble->n_bodies_global == 0) return MFTL_SUCCESS;

  return redistribute_core(ensemble, maps);
}

int MFTLRedistribute(MFTL_Ensemble *ensemble) { return MFTLRedistributeWithOwners(ensemble, MFTL_NULL); }
