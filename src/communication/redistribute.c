/**
 * @file redistribute.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Redistribution: re-partitioning a distributed mesh over the process grid.
 *
 * Both public entry points run the SAME implementation (redistribute_core below);
 * MFTLRedistributeObject is the n == 1 case, which differs only in that it skips
 * the per-body count payload and the run bookkeeping.
 *
 * ## THE ONE IDEA THIS FILE EXISTS TO EXPRESS
 *
 * MOST OF A PARTITION DOES NOT MOVE, SO MOST OF A PARTITION MUST NOT BE TOUCHED
 * BY THE WIRE FORMAT.
 *
 * The previous version treated "stays here" as just another destination and ran
 * it through the full protocol: pack every kept entity into an MFTL_Packed*,
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

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
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

  void *p = realloc(sb->buffer, (size_t) cap * elem);
  if (!p)
  {
    fprintf(stderr, "FATAL ERROR: MFTL redistribution send buffer allocation failed for %lu bytes.\n", (unsigned long) ((size_t) cap * elem));
    exit(EXIT_FAILURE);
  }
  sb->buffer = p;
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

#ifndef NDEBUG
/**
 * @brief Debug check: every triangle's cached centre still matches its vertices.
 *
 * Redistribution routes by the cached centre, for GHOST triangles as well -- that
 * is how a rank knows where its neighbour's triangle is going, and therefore
 * which of its own triangles that rank will need as a ghost. Nothing communicates
 * ghost triangle geometry, so a caller that refreshed only its owned triangles
 * (MFTLEnsembleUpdate rather than MFTLComputeTriangleGeometry(obj, obj->tris.n))
 * leaves those centres one step behind.
 *
 * The consequence is not a crash: two ranks disagree about a ghost triangle's
 * destination, and someone's locally owned triangle silently ends up without one
 * of its edge neighbours. Failing here instead is worth a Debug-only pass.
 *
 * @param obj The mesh.
 * @return `true` if every triangle's `tris.center` matches the centroid its
 *         three vertices imply, to within a loose tolerance.
 */
static bool centres_are_current(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double fresh[3];
    MFTLGetTriangleProperties(obj, t, MFTL_NULL, MFTL_NULL, fresh);

    /* Both are wrapped positions, so compare through the seam, not across it. */
    double d[3];
    MFTLGetMinimumImageVector(obj->domain, d, &obj->tris.center[3 * t], fresh);
    if (fabs(d[0]) > 1e-9 || fabs(d[1]) > 1e-9 || fabs(d[2]) > 1e-9) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}
#endif

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
 * @param      obj        The mesh.
 * @param      map        Explicit owner map, or NULL for the geometric rule.
 * @param[out] tri_owner  `[tris.n]` destination rank per triangle.
 * @param[out] vert_owner `[verts.n]` destination rank per vertex.
 * @param[out] vert_gid   `[verts.n]` vertex global ids, gathered contiguously so
 *                        the packing loops never touch the AoS records again.
 */
static void compute_owners(MFTL_Object *obj, const MFTL_OwnerMap *map, int *restrict tri_owner, int *restrict vert_owner, int *restrict vert_gid)
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

    for (int v = 0; v < nv; ++v)
    {
      const int owner = vo[v];
      MFTL_ASSERT(owner >= 0 && owner < dom->nproc && "redistribute: owner map names a rank outside the communicator");
      vert_owner[v] = owner;
      vert_gid[v] = vr[v].global_id;
      vr[v].owner_proc = owner;
    }

    for (int t = 0; t < nt; ++t)
    {
      const int owner = to[t];
      MFTL_ASSERT(owner >= 0 && owner < dom->nproc && "redistribute: owner map names a rank outside the communicator");
      tri_owner[t] = owner;
      tr[t].owner_proc = owner;
    }
    return;
  }

  MFTL_ASSERT(centres_are_current(obj) &&
              "redistribute: stale triangle centres -- call MFTLComputeTriangleGeometry(obj, obj->tris.n) (ghosts included) first");

  for (int v = 0; v < nv; ++v)
  {
    const int owner = MFTLGetOwnerProc(dom, &pos[3 * v]);
    vert_owner[v] = owner;
    vert_gid[v] = vr[v].global_id;
    vr[v].owner_proc = owner;
  }

  for (int t = 0; t < nt; ++t)
  {
    const int owner = MFTLGetOwnerProc(dom, &ctr[3 * t]);
    tri_owner[t] = owner;
    tr[t].owner_proc = owner;
  }
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

    int cand[7];
    int n_cand = 0;
    cand[n_cand++] = tri_owner[t];
    cand[n_cand++] = vert_owner[v0];
    cand[n_cand++] = vert_owner[v1];
    cand[n_cand++] = vert_owner[v2];
    for (int j = 0; j < 3; ++j)
    {
      const int adj = tn[base + j];
      if (adj != MFTL_ERROR) cand[n_cand++] = tri_owner[adj];
    }

    /* Fast path: the whole stencil sits on one rank. That is every triangle in
     * the interior of a partition -- the overwhelming majority -- and it skips
     * the quadratic dedup below entirely. */
    int dests[7];
    int n_dests = 1;
    dests[0] = cand[0];
    for (int j = 1; j < n_cand; ++j)
    {
      if (cand[j] != cand[0])
      {
        n_dests = MFTLGetUniqueIntegers(cand, n_cand, dests, 7);
        break;
      }
    }

    for (int k = 0; k < n_dests; ++k)
    {
      const int r = dests[k];

      if (r == my_rank)
      {
        keep_t[t] = 1;
        keep_v[v0] = 1;
        keep_v[v1] = 1;
        keep_v[v2] = 1;
        continue;
      }

      if (!is_neighbor[r]) requires_global = 1;

      int b = rank_bit[r];
      if (b == MFTL_ERROR)
      {
        touched[(*n_touched)++] = r;  // first contact and first triangle coincide
        MFTL_ASSERT(tri_ids[r].size == 0 && "collect_destinations: bit slot and buffer state disagree");
        b = (dd->n_bits < MFTL_DEDUP_BITS) ? dd->n_bits++ : MFTL_DEDUP_NO_BIT;
        rank_bit[r] = b;
      }

      ib_add(&tri_ids[r], t);

      if (b == MFTL_DEDUP_NO_BIT)
      {
        ib_add3(&vert_ids[r], v0, v1, v2);  // append_destination will sort this one
        continue;
      }

      const uint32_t m = (uint32_t) 1 << b;
      if (!(mask[v0] & m))
      {
        mask[v0] |= m;
        ib_add(&vert_ids[r], v0);
      }
      if (!(mask[v1] & m))
      {
        mask[v1] |= m;
        ib_add(&vert_ids[r], v1);
      }
      if (!(mask[v2] & m))
      {
        mask[v2] |= m;
        ib_add(&vert_ids[r], v2);
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
 * @param body_idx Index of the owning object in the ensemble's list.
 */
static inline void pack_triangle(const MFTL_Object *obj, int t, const int *restrict tv, const int *restrict vgid, MFTL_PackedTriangleComplex *dst,
                                 int body_idx)
{
  const MFTL_Triangle *const src = &obj->tris.rec[t];
  const MFTL_HalfEdge *const he = &obj->tris.he[3 * t];
  const int base = 3 * t;

  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->flags = src->flags;
  dst->body_idx = body_idx;

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
 * @param body_idx Index of the owning object in the ensemble's list.
 */
static inline void pack_vertex(const MFTL_Object *obj, int v, MFTL_PackedVertex *dst, int body_idx)
{
  const MFTL_Vertex *const src = &obj->verts.rec[v];
  const double *const p = &obj->verts.pos[3 * v];

  dst->pos[0] = p[0];
  dst->pos[1] = p[1];
  dst->pos[2] = p[2];
  dst->global_id = src->global_id;
  dst->owner_proc = src->owner_proc;
  dst->flags = src->flags;
  dst->body_idx = body_idx;
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
 * @param body_idx    Index of the owning object in the ensemble's list.
 * @param vgid        `[verts.n]` vertex global ids.
 * @param tri_ids     Triangle indices bound for this destination.
 * @param vert_ids    Vertex indices bound for this destination.
 * @param out_tris    Send buffer the packed triangles are appended to.
 * @param out_verts   Send buffer the packed vertices are appended to.
 * @param needs_dedup `true` when the destination had no de-duplication bit slot,
 *                    so the vertex list must be sorted and uniqued here.
 */
static void append_destination(const MFTL_Object *obj, int body_idx, const int *restrict vgid, mftl_int_buffer *tri_ids, mftl_int_buffer *vert_ids,
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
    for (int i = 0; i < tri_ids->size; ++i) pack_triangle(obj, tri_ids->data[i], tv, vgid, &buf[out_tris->count++], body_idx);
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
    for (int i = 0; i < n; ++i) pack_vertex(obj, vert_ids->data[i], &buf[out_verts->count++], body_idx);
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

  /* Arrival slices. `*_idx == MFTL_NULL` means the identity [0, n_recv_*), which
   * is the single-body case where no grouping is needed at all. */
  const int *recv_v_idx;  ///< Arriving vertex slots, or MFTL_NULL for the identity.
  int n_recv_v;  ///< Vertices arriving for this body.
  const int *recv_t_idx;  ///< Arriving triangle slots, or MFTL_NULL for the identity.
  int n_recv_t;  ///< Triangles arriving for this body.
} body_state;

/**
 * @brief Splits the keep flags into the two ascending runs the rebuild wants.
 *
 * Owned first, because n_local is defined as an array PREFIX. `owner_proc` on the
 * record is the NEW owner at this point (compute_owners wrote it), so this is the
 * final ownership, not the one the entity arrived with.
 *
 * @param      obj     The mesh.
 * @param      keep_v  `[verts.n]` flags from collect_destinations().
 * @param      keep_t  `[tris.n]` flags from collect_destinations().
 * @param      my_rank This rank.
 * @param[out] bs      Per-body state; its kept lists and counts are filled in.
 */
static void build_kept_lists(const MFTL_Object *obj, const char *restrict keep_v, const char *restrict keep_t, int my_rank, body_state *bs)
{
  const MFTL_Vertex *const vr = obj->verts.rec;
  const int nv = obj->verts.n;
  int n = 0;

  for (int v = 0; v < nv; ++v)
    if (keep_v[v] && vr[v].owner_proc == my_rank) bs->kept_v[n++] = v;
  bs->n_kept_v_owned = n;
  for (int v = 0; v < nv; ++v)
    if (keep_v[v] && vr[v].owner_proc != my_rank) bs->kept_v[n++] = v;
  bs->n_kept_v = n;

  const MFTL_Triangle *const tr = obj->tris.rec;
  const int nt = obj->tris.n_local;  // only owned triangles can ever be kept
  n = 0;

  for (int t = 0; t < nt; ++t)
    if (keep_t[t] && tr[t].owner_proc == my_rank) bs->kept_t[n++] = t;
  bs->n_kept_t_owned = n;
  for (int t = 0; t < nt; ++t)
    if (keep_t[t] && tr[t].owner_proc != my_rank) bs->kept_t[n++] = t;
  bs->n_kept_t = n;
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
 * @param obj    The mesh being rebuilt.
 * @param u_list Candidate half-edges whose twin is still unresolved.
 * @param n_u    Number of entries in @p u_list.
 * @param v_head `[verts.n]` head of a per-vertex chain, scratch.
 * @param u_next `[n_u]` next-link of that chain, scratch.
 */
static void repair_twins(MFTL_Object *obj, const int *restrict u_list, int n_u, int *restrict v_head, int *restrict u_next)
{
  MFTL_HalfEdge *const he = obj->tris.he;

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
 *                positions and both derived bits set; the cached GEOMETRY is
 *                unspecified.
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

  /* ── 1. Gather the kept entities out of the pool ─────────────────────────
   *
   * Records and positions only. Everything else in the pool is either derived
   * (rebuilt below) or cached geometry the caller recomputes -- see the
   * postcondition in mftl_communication.h. Copying `tris.area/normal/center` here
   * would be 56 B per triangle of traffic for values that are overwritten before
   * they are read.
   *
   * The gather runs before the id table is sized because it is also what finds
   * the largest kept global id. */
  ensure_bytes(&s->v_rec, &s->v_rec_cap, (size_t) MFTL_MAX(n_kept_v, 1) * sizeof(MFTL_Vertex));
  ensure_bytes(&s->v_pos, &s->v_pos_cap, (size_t) MFTL_MAX(n_kept_v, 1) * 3 * sizeof(double));
  ensure_bytes(&s->t_rec, &s->t_rec_cap, (size_t) MFTL_MAX(n_kept_t, 1) * sizeof(MFTL_Triangle));
  ensure_bytes(&s->t_he, &s->t_he_cap, (size_t) MFTL_MAX(n_kept_t, 1) * 3 * sizeof(MFTL_HalfEdge));

  MFTL_Vertex *const kv_rec = (MFTL_Vertex *) s->v_rec;
  double *const kv_pos = (double *) s->v_pos;
  MFTL_Triangle *const kt_rec = (MFTL_Triangle *) s->t_rec;
  MFTL_HalfEdge *const kt_he = (MFTL_HalfEdge *) s->t_he;

  int v_range = obj->next_vertex_gid;
  for (int i = 0; i < n_kept_v; ++i)
  {
    const int o = bs->kept_v[i];
    kv_rec[i] = obj->verts.rec[o];
    /* Every out_he_idx is re-derived below; a value carried over from the old
     * index space would point into someone else's fan. */
    kv_rec[i].out_he_idx = MFTL_ERROR;
    kv_pos[3 * i + 0] = obj->verts.pos[3 * o + 0];
    kv_pos[3 * i + 1] = obj->verts.pos[3 * o + 1];
    kv_pos[3 * i + 2] = obj->verts.pos[3 * o + 2];
    if (kv_rec[i].global_id >= v_range) v_range = kv_rec[i].global_id + 1;
  }

  int t_range = obj->next_triangle_gid;
  for (int i = 0; i < n_kept_t; ++i)
  {
    const int o = bs->kept_t[i];
    kt_rec[i] = obj->tris.rec[o];
    kt_he[3 * i + 0] = obj->tris.he[3 * o + 0];
    kt_he[3 * i + 1] = obj->tris.he[3 * o + 1];
    kt_he[3 * i + 2] = obj->tris.he[3 * o + 2];
    if (kt_rec[i].global_id >= t_range) t_range = kt_rec[i].global_id + 1;
  }

  /* ── 2. Classify the arrivals against the kept entities ──────────────────── */
  for (int i = 0; i < bs->n_recv_v; ++i)
  {
    const int g = recv_v[bs->recv_v_idx ? bs->recv_v_idx[i] : i].global_id;
    if (g >= v_range) v_range = g + 1;
  }
  for (int i = 0; i < bs->n_recv_t; ++i)
  {
    const int g = recv_t[bs->recv_t_idx ? bs->recv_t_idx[i] : i].global_id;
    if (g >= t_range) t_range = g + 1;
  }

  gt_begin(&s->v_tbl, MFTL_MAX(v_range, 1));
  gt_begin(&s->t_tbl, MFTL_MAX(t_range, 1));
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
    const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx ? bs->recv_v_idx[i] : i];
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
    const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx ? bs->recv_t_idx[i] : i];
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
      const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx ? bs->recv_v_idx[i] : i];
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
      const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx ? bs->recv_t_idx[i] : i];
      s->t_final_of_logical[L] = (p->owner_proc == my_rank) ? o++ : g++;
    }
    MFTL_ASSERT(o == bs->n_kept_t_owned + n_new_t_owned && "rebuild_object: owned triangle block overrun");
  }

  const int n_v = n_kept_v + n_new_v;
  const int n_v_local = bs->n_kept_v_owned + n_new_v_owned;
  const int n_t = n_kept_t + n_new_t;
  const int n_t_local = bs->n_kept_t_owned + n_new_t_owned;

  /* ── 4. Size the pools, then place the kept entities ─────────────────────── */
  MFTLObjectSetCounts(obj, n_v, n_v_local, n_t, n_t_local);

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

  /* ── 5. Old pool index -> new pool slot, for the kept entities ───────────── */
  ensure_ints(&s->v_final_of_old, &s->v_final_of_old_cap, MFTL_MAX(n_old_v, 1));
  ensure_ints(&s->t_final_of_old, &s->t_final_of_old_cap, MFTL_MAX(n_old_t, 1));
  for (int v = 0; v < n_old_v; ++v) s->v_final_of_old[v] = MFTL_ERROR;
  for (int t = 0; t < n_old_t; ++t) s->t_final_of_old[t] = MFTL_ERROR;
  for (int i = 0; i < n_kept_v; ++i) s->v_final_of_old[bs->kept_v[i]] = s->v_final_of_logical[i];
  for (int i = 0; i < n_kept_t; ++i) s->t_final_of_old[bs->kept_t[i]] = s->t_final_of_logical[i];

  /* ── 6. Re-link the kept half-edges ──────────────────────────────────────
   *
   * Three things happen in one pass over 48 bytes per kept triangle: the target
   * vertex index is moved into the new index space, the twin is either remapped
   * or retired into the repair list, and `out_he_idx` is re-derived. The repair
   * list therefore costs nothing to build -- the loop already knows which twins
   * did not survive. */
  ib_reset(&s->u_list);
  MFTL_HalfEdge *const he = obj->tris.he;
  MFTL_Vertex *const vr = obj->verts.rec;

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
    }

    /* origin(3s+k) == target(3s+(k+2)%3), and all three targets are final now. */
    vr[h3[2].vertex_idx].out_he_idx = 3 * slot + 0;
    vr[h3[0].vertex_idx].out_he_idx = 3 * slot + 1;
    vr[h3[1].vertex_idx].out_he_idx = 3 * slot + 2;
  }

  /* ── 7. Write the arrivals ───────────────────────────────────────────────── */
  for (int i = 0; i < bs->n_recv_v; ++i)
  {
    const int L = s->recv_v_logical[i];
    if (L == MFTL_ERROR) continue;

    const MFTL_PackedVertex *const p = &recv_v[bs->recv_v_idx ? bs->recv_v_idx[i] : i];
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

    const MFTL_PackedTriangleComplex *const p = &recv_t[bs->recv_t_idx ? bs->recv_t_idx[i] : i];
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
      ib_add(&s->u_list, 3 * slot + k);
    }

    vr[h3[2].vertex_idx].out_he_idx = 3 * slot + 0;
    vr[h3[0].vertex_idx].out_he_idx = 3 * slot + 1;
    vr[h3[1].vertex_idx].out_he_idx = 3 * slot + 2;
  }

  /* ── 8. Repair, then rebuild what is genuinely derived ───────────────────── */
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

  MFTLObjectBuildDerived(obj);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Multi-body: separating the interleaved arrival stream
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct body_run
 * @brief One contiguous run of one body's entities inside the arrival sequence.
 *
 * The multi-body rebuild has to separate the interleaved stream back into one
 * group per object. The previous version derived the groups by extracting the
 * body id from every arrival (a strided read over the packed complexes),
 * radix-sorting all of them, and gathering again -- work to recover information
 * the SENDER already had.
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
  int obj_idx;  ///< Index into the caller's object list, MFTL_ERROR if unknown.
  int tri_start;  ///< First arriving triangle of this run in the flat receive buffer.
  int tri_len;  ///< Triangles in the run.
  int vert_start;  ///< First arriving vertex of this run.
  int vert_len;  ///< Vertices in the run.
} body_run;

/** @brief Per-destination triple appended once per body that contributed to it. */
#define MFTL_BODY_COUNT_INTS 3

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

#ifndef NDEBUG
/**
 * @brief Debug check: every rank holds the same object list in the same order.
 *
 * Bodies are named on the wire by their INDEX in that list, which is what removed
 * the previous version's body-id hash map. If the lists disagree, entities are
 * merged into the wrong object and nothing crashes -- so the precondition is
 * verified rather than merely documented.
 *
 * @brief Debug-only: do all ranks hold the same object list in the same order?
 *
 * Bodies are identified ON THE WIRE by their INDEX in the ensemble's list, so a
 * disagreement silently merges entities into the wrong body. A checksum
 * reduction reports it instead.
 *
 * @param dom     Supplies the communicator.
 * @param objects The object list.
 * @param n       Its length.
 */
static void check_object_lists_agree(const MFTL_Domain *dom, MFTL_Object *const *objects, int n)
{
  unsigned sig = (unsigned) n;
  for (int o = 0; o < n; ++o) sig = sig * 1000003u + (unsigned) objects[o]->global_id;

  unsigned v[2] = {sig, ~sig};
  MPI_Allreduce(MPI_IN_PLACE, v, 2, MPI_UNSIGNED, MPI_MIN, dom->comm_world);
  MFTL_ASSERT(v[0] == sig && v[1] == ~sig && "redistribute: ranks disagree about the object list -- bodies are identified by list INDEX");
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. The driver
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The one redistribution implementation; both public entry points call it.
 *
 * @param domain      The MPI/physical world.
 * @param objects     Objects to repartition, all in one communication round.
 * @param num_objects Number of objects.
 * @param maps        `[num_objects]` explicit owner maps, or NULL for the
 *                    geometric rule.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the exchange failed.
 */
static int redistribute_core(MFTL_Domain *domain, MFTL_Object **objects, int num_objects, const MFTL_OwnerMap *maps)
{
  MFTL_ASSERT(MFTL_MPI_PackedVertex != MPI_DATATYPE_NULL && "MFTLRedistribute: call MFTLRegisterMpiTypes() first");

  const int nproc = domain->nproc;
  const int my_rank = domain->rank;

#ifndef NDEBUG
  check_object_lists_agree(domain, objects, num_objects);
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
      if (!MFTLObjectHasTopology(obj) || !MFTLObjectHasFrames(obj)) MFTLObjectBuildDerived(obj);
    }
    return MFTL_SUCCESS;
  }

  /* ── Phase 0: setup ────────────────────────────────────────────────────── */
  char *const is_neighbor = (char *) MFTLCallocOrDie((size_t) nproc, 1, "redistribution neighbour map");
  for (int i = 0; i < domain->num_neighbors; ++i) is_neighbor[domain->neighbors[i]] = 1;

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

  mftl_int_buffer *const tri_ids = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nproc * sizeof(mftl_int_buffer), "redistribution id lists");
  mftl_int_buffer *const vert_ids = (mftl_int_buffer *) MFTLMallocOrDie((size_t) nproc * sizeof(mftl_int_buffer), "redistribution id lists");
  for (int r = 0; r < nproc; ++r)
  {
    ib_init(&tri_ids[r], tri_hint);
    ib_init(&vert_ids[r], vert_hint);
  }

  MFTL_SendBuffer *const send_tris = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution send buffers");
  MFTL_SendBuffer *const send_verts = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution send buffers");

  const bool group_bodies = (num_objects > 1);
  MFTL_SendBuffer *const send_counts =
      group_bodies ? (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "redistribution body counts") : MFTL_NULL;

  int *const touched = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "redistribution touched ranks");

  /* One arena for every object's kept lists rather than two mallocs per object:
   * at a thousand droplets that is two thousand allocations per timestep. */
  int *const kept_arena_v = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(sum_verts, 1) * sizeof(int), "redistribution kept vertex lists");
  int *const kept_arena_t = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(sum_tris, 1) * sizeof(int), "redistribution kept triangle lists");
  body_state *const states = (body_state *) MFTLCallocOrDie((size_t) num_objects, sizeof(body_state), "redistribution body states");

  int *const tri_owner = (int *) MFTLMallocOrDie((size_t) max_tris * sizeof(int), "redistribution triangle owners");
  int *const vert_owner = (int *) MFTLMallocOrDie((size_t) max_verts * sizeof(int), "redistribution vertex owners");
  int *const vert_gid = (int *) MFTLMallocOrDie((size_t) max_verts * sizeof(int), "redistribution vertex gids");
  char *const keep_v = (char *) MFTLMallocOrDie((size_t) max_verts, "redistribution vertex keep flags");
  char *const keep_t = (char *) MFTLMallocOrDie((size_t) max_tris, "redistribution triangle keep flags");

  vertex_dedup dd;
  dd.mask = (uint32_t *) MFTLMallocOrDie((size_t) max_verts * sizeof(uint32_t), "redistribution vertex dedup mask");
  dd.rank_bit = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "redistribution dedup bit slots");
  dd.n_bits = 0;
  for (int r = 0; r < nproc; ++r) dd.rank_bit[r] = MFTL_ERROR;

  /* ── Phase 1: per object, decide destinations, keep or pack ─────────────── */
  int requires_global = 0;
  int arena_v = 0, arena_t = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *const obj = objects[o];
    body_state *const bs = &states[o];

    bs->kept_v = kept_arena_v + arena_v;
    bs->kept_t = kept_arena_t + arena_t;
    arena_v += obj->verts.n;
    arena_t += obj->tris.n_local;

    if (obj->tris.n < 1 || obj->verts.n < 1) continue;

    compute_owners(obj, maps ? &maps[o] : MFTL_NULL, tri_owner, vert_owner, vert_gid);

    memset(keep_v, 0, (size_t) obj->verts.n);
    memset(keep_t, 0, (size_t) obj->tris.n_local);
    memset(dd.mask, 0, (size_t) obj->verts.n * sizeof(uint32_t));
    dd.n_bits = 0;

    int n_touched = 0;
    requires_global |= collect_destinations(obj, my_rank, is_neighbor, tri_owner, vert_owner, &dd, keep_t, keep_v, tri_ids, vert_ids, touched, &n_touched);

    build_kept_lists(obj, keep_v, keep_t, my_rank, bs);

    /* Only the ranks this object actually reached -- typically fewer than 27, not
     * nproc. That difference is the whole point of the touched list. */
    for (int i = 0; i < n_touched; ++i)
    {
      const int r = touched[i];
      const int tri_before = send_tris[r].count;
      const int vert_before = send_verts[r].count;

      append_destination(obj, o, vert_gid, &tri_ids[r], &vert_ids[r], &send_tris[r], &send_verts[r], dd.rank_bit[r] == MFTL_DEDUP_NO_BIT);

      if (group_bodies)
      {
        sb_reserve(&send_counts[r], MFTL_BODY_COUNT_INTS, sizeof(int));
        int *const c = (int *) send_counts[r].buffer + send_counts[r].count;
        c[0] = o;  // the object INDEX; every rank holds the same list
        c[1] = send_tris[r].count - tri_before;
        c[2] = send_verts[r].count - vert_before;
        send_counts[r].count += MFTL_BODY_COUNT_INTS;
      }

      ib_reset(&tri_ids[r]);
      ib_reset(&vert_ids[r]);
      dd.rank_bit[r] = MFTL_ERROR;
    }
  }

  free(tri_owner);
  free(vert_owner);
  free(vert_gid);
  free(keep_v);
  free(keep_t);
  free(dd.mask);
  free(dd.rank_bit);
  free(touched);
  free(is_neighbor);
  for (int r = 0; r < nproc; ++r)
  {
    ib_free(&tri_ids[r]);
    ib_free(&vert_ids[r]);
  }
  free(tri_ids);
  free(vert_ids);

  /* ── Phase 2: agree on the communication path ────────────────────────────
   *
   * Not a local decision: if A must send to a non-neighbour B, then B has to be
   * listening on the global path, and B cannot know that by itself.
   *
   * BLOCKING, deliberately. It is one int, so the cost is pure latency, and there
   * is nothing left to overlap it with -- the send buffers are already packed and
   * the rebuild cannot begin before the data arrives. The previous version hid it
   * behind tearing the old pools down; nothing is torn down here, which is the
   * whole point of section 5. */
  int global_decision = 0;
  MPI_Allreduce(&requires_global, &global_decision, 1, MPI_INT, MPI_LOR, domain->comm_cart);

  /* ── Phase 3: exchange ─────────────────────────────────────────────────── */
  MFTL_Exchange xfer[3];
  xfer[0].send_bufs = send_verts;
  xfer[0].type = MFTL_MPI_PackedVertex;
  xfer[0].type_size = sizeof(MFTL_PackedVertex);
  xfer[1].send_bufs = send_tris;
  xfer[1].type = MFTL_MPI_PackedTriangleComplex;
  xfer[1].type_size = sizeof(MFTL_PackedTriangleComplex);
  if (group_bodies)
  {
    xfer[2].send_bufs = send_counts;
    xfer[2].type = MPI_INT;
    xfer[2].type_size = sizeof(int);
  }

  MFTL_ExchangeCtx ctx;
  MFTLExchangeStart(domain, xfer, group_bodies ? 3 : 2, global_decision != 0, &ctx);

  /* MFTLExchangeStart has already copied everything into its own storage. */
  for (int r = 0; r < nproc; ++r)
  {
    free(send_tris[r].buffer);
    free(send_verts[r].buffer);
    if (group_bodies) free(send_counts[r].buffer);
  }
  free(send_tris);
  free(send_verts);
  free(send_counts);

  /* ── Phase 4: reconstruct the per-body arrival slices ───────────────────── */
  body_run *runs = MFTL_NULL;
  int *grp_off_v = MFTL_NULL, *grp_idx_v = MFTL_NULL;
  int *grp_off_t = MFTL_NULL, *grp_idx_t = MFTL_NULL;

  if (group_bodies)
  {
    MFTLExchangeWait(&ctx, 2);
    const int *const rc = (const int *) xfer[2].recv_buf;
    const int n_runs = xfer[2].recv_count / MFTL_BODY_COUNT_INTS;

    runs = (body_run *) MFTLMallocOrDie((size_t) MFTL_MAX(n_runs, 1) * sizeof(body_run), "redistribution body runs");

    int t_off = 0, v_off = 0;
    for (int k = 0; k < n_runs; ++k)
    {
      const int *const tp = &rc[MFTL_BODY_COUNT_INTS * k];
      const int idx = tp[0];

      runs[k].obj_idx = (idx >= 0 && idx < num_objects) ? idx : MFTL_ERROR;
      MFTL_ASSERT(runs[k].obj_idx != MFTL_ERROR && "redistribute: received a body index outside the object list");
      runs[k].tri_start = t_off;
      runs[k].tri_len = tp[1];
      runs[k].vert_start = v_off;
      runs[k].vert_len = tp[2];

      t_off += tp[1];
      v_off += tp[2];
    }

    MFTLExchangeWait(&ctx, 0);
    MFTLExchangeWait(&ctx, 1);

    MFTL_ASSERT(t_off == xfer[1].recv_count && "redistribute: body triangle counts do not add up to the arrival count");
    MFTL_ASSERT(v_off == xfer[0].recv_count && "redistribute: body vertex counts do not add up to the arrival count");

    int *const cursor = (int *) MFTLMallocOrDie((size_t) num_objects * sizeof(int), "redistribution run cursors");
    grp_off_v = (int *) MFTLMallocOrDie((size_t) (num_objects + 1) * sizeof(int), "redistribution group offsets");
    grp_off_t = (int *) MFTLMallocOrDie((size_t) (num_objects + 1) * sizeof(int), "redistribution group offsets");
    grp_idx_v = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(xfer[0].recv_count, 1) * sizeof(int), "redistribution group indices");
    grp_idx_t = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(xfer[1].recv_count, 1) * sizeof(int), "redistribution group indices");

    group_runs_by_object(runs, n_runs, num_objects, MFTL_FALSE, grp_off_v, grp_idx_v, cursor);
    group_runs_by_object(runs, n_runs, num_objects, MFTL_TRUE, grp_off_t, grp_idx_t, cursor);
    free(cursor);

    for (int o = 0; o < num_objects; ++o)
    {
      states[o].recv_v_idx = grp_idx_v + grp_off_v[o];
      states[o].n_recv_v = grp_off_v[o + 1] - grp_off_v[o];
      states[o].recv_t_idx = grp_idx_t + grp_off_t[o];
      states[o].n_recv_t = grp_off_t[o + 1] - grp_off_t[o];
    }
  }
  else
  {
    MFTLExchangeWait(&ctx, 0);
    MFTLExchangeWait(&ctx, 1);

    states[0].recv_v_idx = MFTL_NULL;  // identity
    states[0].n_recv_v = xfer[0].recv_count;
    states[0].recv_t_idx = MFTL_NULL;
    states[0].n_recv_t = xfer[1].recv_count;
  }

  /* ── Phase 5: rebuild ──────────────────────────────────────────────────── */
  rebuild_scratch scratch;
  memset(&scratch, 0, sizeof scratch);
  ib_init(&scratch.u_list, 1024);
  gt_init(&scratch.v_tbl);
  gt_init(&scratch.t_tbl);

  for (int o = 0; o < num_objects; ++o)
  {
    rebuild_object(objects[o], &states[o], (const MFTL_PackedVertex *) xfer[0].recv_buf, (const MFTL_PackedTriangleComplex *) xfer[1].recv_buf, my_rank,
                   &scratch);
  }

  free(scratch.v_rec);
  free(scratch.v_pos);
  free(scratch.t_rec);
  free(scratch.t_he);
  free(scratch.v_final_of_old);
  free(scratch.t_final_of_old);
  free(scratch.v_final_of_logical);
  free(scratch.t_final_of_logical);
  free(scratch.recv_v_logical);
  free(scratch.recv_t_logical);
  free(scratch.v_head);
  free(scratch.u_next);
  ib_free(&scratch.u_list);
  gt_free(&scratch.v_tbl);
  gt_free(&scratch.t_tbl);

  free(grp_off_v);
  free(grp_off_t);
  free(grp_idx_v);
  free(grp_idx_t);
  free(runs);

  free(xfer[0].recv_buf);
  free(xfer[1].recv_buf);
  if (group_bodies) free(xfer[2].recv_buf);

  MFTLExchangeFree(&ctx);

  free(kept_arena_v);
  free(kept_arena_t);
  free(states);

  return MFTL_SUCCESS;
}

int MFTLRedistributeObject(MFTL_Object *object)
{
  if (object == MFTL_NULL || object->domain == MFTL_NULL) return MFTL_ERROR;
  return redistribute_core(object->domain, &object, 1, MFTL_NULL);
}

int MFTLRedistributeListOfObjects(MFTL_Ensemble *ensemble)
{
  if (ensemble == MFTL_NULL || ensemble->num_objects == 0 || ensemble->objects == MFTL_NULL) return MFTL_ERROR;
  return redistribute_core(&ensemble->domain, ensemble->objects, ensemble->num_objects, MFTL_NULL);
}

int MFTLRedistributeWithOwners(MFTL_Ensemble *ensemble, const MFTL_OwnerMap *maps)
{
  if (ensemble == MFTL_NULL || ensemble->objects == MFTL_NULL) return MFTL_ERROR;
  if (ensemble->num_objects == 0) return MFTL_SUCCESS;

  /* No dispatch of its own: redistribute_core already takes the direct path --
   * no per-body count payload, no run bookkeeping -- when num_objects is 1, which
   * is the whole of what MFTLRedistribute's dispatch buys. */
  return redistribute_core(&ensemble->domain, ensemble->objects, ensemble->num_objects, maps);
}
