/**
 * @file remesh_compact.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief Removing dead entities, and re-densifying global ids.
 *
 * ## WHY A COLLAPSE CANNOT FREE ANYTHING, AND WHAT FOLLOWS
 *
 * A triangle's three half-edges ARE slots 3t..3t+2 of the triangle pool. There
 * is no half-edge free list because there is no half-edge allocation, so an
 * operator that removes a triangle cannot hand anything back -- it can only mark
 * it. Compaction is therefore not an optimisation here, it is the deallocator.
 *
 * Two properties of this library make the pass simpler than a general compaction:
 *
 *   HALF-EDGES MOVE AS TRIPLES. `he[3*t_new + k] = he[3*t_old + k]`, so the
 *   implicit indexing invariant survives by construction and so do the implicit
 *   half-edge global ids (3 * triangle gid + k) -- neither needs a fix-up pass.
 *   Compacting half-edges as an independent array with a map of their own is
 *   what forces `next_idx` and `triangle_idx` to be remapped as well.
 *
 *   ONE ALLOCATION PER POOL. Compacting a pool is a loop over its sub-arrays,
 *   and a sub-array that someone forgot cannot exist -- it would have to have
 *   been allocated separately, and nothing outside src/object/object.c can do
 *   that.
 *
 * ## out_he_idx IS RE-DERIVED, NOT REMAPPED
 *
 * Remapping it would be one array lookup per vertex and would faithfully
 * preserve whatever each operator left behind, INCLUDING an index the operator
 * failed to repair. Re-deriving it from the surviving half-edges is the same
 * single-pass rule MFTLBuildTopologyFromIndexList uses, costs O(H) rather than
 * O(V), and cannot be stale. It gives up nothing: the fan walkers rewind to the
 * start of an open fan themselves (MFTLVertexFanStart), so which outgoing
 * half-edge a vertex names has never mattered.
 *
 * ## COMPACTION IS ALSO WHERE THE OWNERSHIP PREFIX IS RE-ESTABLISHED
 *
 * An operator appends, and an append lands at the end of the pool -- past the
 * ghosts. So during a parallel remeshing sweep the pool is
 *
 *     [ owned ...... | ghosts ...... | entities created this sweep ...... ]
 *                    ^ n_local
 *
 * which violates invariant 2 of mftl_types.h (ownership is a prefix) for exactly
 * as long as the sweep runs. This function is what ends that window, and it is
 * the natural place: it is already a full permutation of both pools with a full
 * index remap behind it, so restoring the prefix costs one more block move and
 * one more term in the map -- not a second pass.
 *
 * The permutation is not a forward compaction any more, because a ghost may have
 * to move RIGHT to make room for the entities created after it. It is a forward
 * compaction FOLLOWED BY a swap of the two trailing blocks, and that is the whole
 * of it: the forward compaction preserves relative order, so the survivors come
 * out as [owned][ghosts][created] and the prefix is restored by exchanging the
 * last two. A general permutation would need cycle-following or a full copy of
 * both pools; two adjacent blocks need one buffer the size of the smaller of
 * them and two memmoves.
 *
 * On a serial object there are no ghosts, an append increments `n_local` with
 * the pool, both trailing blocks are empty and every line below is the code that
 * was here before.
 */

#include "mftl_remeshing.h"

#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_internal.h"

/**
 * @brief Exchanges the two adjacent blocks [first, first+n1) and [first+n1, +n2).
 *
 * @param base    Start of the array.
 * @param elem    Size of one element in bytes.
 * @param first   Index the first run starts at.
 * @param n1      Elements in the first run.
 * @param n2      Elements in the second run, which follows it immediately.
 * @param scratch Buffer of at least `MIN(n1, n2) * elem` bytes.
 */
static void swap_adjacent_blocks(void *base, size_t elem, int first, int n1, int n2, void *scratch)
{
  if (n1 == 0 || n2 == 0) return;

  char *const p = (char *) base + (size_t) first * elem;

  if (n1 <= n2)
  {
    memcpy(scratch, p, (size_t) n1 * elem);
    memmove(p, p + (size_t) n1 * elem, (size_t) n2 * elem);
    memcpy(p + (size_t) n2 * elem, scratch, (size_t) n1 * elem);
  }
  else
  {
    memcpy(scratch, p + (size_t) n1 * elem, (size_t) n2 * elem);
    memmove(p + (size_t) n2 * elem, p, (size_t) n1 * elem);
    memcpy(p, scratch, (size_t) n2 * elem);
  }
}

/**
 * @struct pool_partition
 * @brief The three runs a compacted pool falls into, and the map term they imply.
 *
 * `n_kept_owned` survivors from the ownership prefix, then `n_ghost` surviving
 * ghosts, then `n_created` entities this sweep appended. Swapping the last two
 * restores the prefix, and `partition_remap` is that swap expressed on indices.
 */
typedef struct
{
  int n_kept_owned;  ///< Surviving entities that were owned before the pass.
  int n_ghost;  ///< Surviving ghosts.
  int n_created;  ///< Entities this rank created during the pass, past the old prefix.
} pool_partition;

/**
 * @brief Where compacted index @p i lands once the two trailing blocks swap.
 *
 * @brief Re-establishes the ownership prefix and reports the new index of each entity.
 * @param      p     The three run lengths of the pool.
 * @param[out] i `[n]` new index per old index, or #MFTL_ERROR for a dead entity.
 * @return The new owned count, i.e. the prefix length.
 */
static inline int partition_remap(const pool_partition *p, int i)
{
  if (i < p->n_kept_owned) return i;
  if (i < p->n_kept_owned + p->n_ghost) return i + p->n_created;
  return i - p->n_ghost;
}

int MFTLRemeshCompactObject(MFTL_Object *obj, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int n_v_old = obj->verts.n;
  const int n_t_old = obj->tris.n;

  /* The prefix as it stands BEFORE this call: everything below it is owned,
   * everything above it is a ghost or was created by this sweep, and only
   * `owner_proc` tells those two apart. */
  const int lv_old = obj->verts.n_local;
  const int lt_old = obj->tris.n_local;
  const int my_rank = (obj->domain != MFTL_NULL) ? obj->domain->rank : 0;

  if (n_t_old == 0 && n_v_old == 0) return 0;

  /* One allocation for both maps plus the reference marks: same lifetime, and
   * three separate mallocs of a few hundred kilobytes each is three first-touch
   * fault storms per remeshing iteration. */
  int *const scratch = (int *) MFTLMallocOrDie(sizeof(int) * ((size_t) n_v_old * 2 + (size_t) n_t_old), "remesh compaction maps");
  int *const v_map = scratch;
  int *const v_referenced = scratch + n_v_old;
  int *const t_map = scratch + 2 * (size_t) n_v_old;

  /* ── 1. Which vertices does the surviving topology actually use? ──────────
   *
   * Not the same question as "which are not flagged dead". A vertex can be left
   * with no incident triangle by a sequence of collapses around it without ever
   * being the one retired. It is still a valid record pointing at a half-edge
   * that no longer exists, so every later traversal from it walks into freed
   * territory. Counting these separately is deliberate -- an orphan is a symptom,
   * and a silent removal would hide it. */
  memset(v_referenced, 0, (size_t) n_v_old * sizeof(int));

  for (int t = 0; t < n_t_old; ++t)
  {
    if (tri_is_dead(obj, t)) continue;
    for (int k = 0; k < 3; ++k)
    {
      const int v = obj->tris.he[3 * t + k].vertex_idx;
      MFTL_ASSERT(v >= 0 && v < n_v_old && "MFTLRemeshCompactObject: live half-edge targets a vertex outside the pool");
      v_referenced[v] = 1;
    }
  }

  /* ── 2. Build the maps, and classify the survivors into the three runs ───── */
  pool_partition vpart = {0, 0, 0};
  pool_partition tpart = {0, 0, 0};

  int n_v_new = 0;
  int n_dead_v = 0;
  int n_orphan_v = 0;

  for (int v = 0; v < n_v_old; ++v)
  {
    const bool dead = vert_is_dead(obj, v);

    if (dead || !v_referenced[v])
    {
      MFTL_ASSERT((v < lv_old || obj->verts.rec[v].owner_proc == my_rank) &&
                  "MFTLRemeshCompactObject: a GHOST vertex was retired. Ghosts are frozen for the whole sweep, so "
                  "this means the freeze classification and the operators disagree about what is replicated.");
      v_map[v] = MFTL_ERROR;
      if (dead)
        n_dead_v++;
      else
        n_orphan_v++;
      continue;
    }

    if (v < lv_old)
      vpart.n_kept_owned++;
    else if (obj->verts.rec[v].owner_proc != my_rank)
    {
      vpart.n_ghost++;
      MFTL_ASSERT(vpart.n_created == 0 && "MFTLRemeshCompactObject: a created vertex precedes a ghost -- the pool is not [owned|ghost|created]");
    }
    else
      vpart.n_created++;

    v_map[v] = n_v_new++;
  }

  int n_t_new = 0;
  for (int t = 0; t < n_t_old; ++t)
  {
    if (tri_is_dead(obj, t))
    {
      MFTL_ASSERT((t < lt_old || obj->tris.rec[t].owner_proc == my_rank) && "MFTLRemeshCompactObject: a GHOST triangle was retired -- see the vertex case");
      t_map[t] = MFTL_ERROR;
      continue;
    }

    if (t < lt_old)
      tpart.n_kept_owned++;
    else if (obj->tris.rec[t].owner_proc != my_rank)
    {
      tpart.n_ghost++;
      MFTL_ASSERT(tpart.n_created == 0 && "MFTLRemeshCompactObject: a created triangle precedes a ghost -- the pool is not [owned|ghost|created]");
    }
    else
      tpart.n_created++;

    t_map[t] = n_t_new++;
  }

  /* Both blocks non-empty is exactly "a partitioned object grew this sweep". */
  const bool swap_v = (vpart.n_ghost > 0 && vpart.n_created > 0);
  const bool swap_t = (tpart.n_ghost > 0 && tpart.n_created > 0);

  const int removed = (n_v_old - n_v_new) + (n_t_old - n_t_new);
  if (removed == 0 && !swap_v && !swap_t)
  {
    /* Nothing moved and no block needs exchanging, so the prefix is already
     * right: either the object has no ghosts (and the appends grew `n_local`
     * with the pool) or it grew by nothing. */
    free(scratch);
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    return 0;
  }

  /* ── 3. Compact the pools ────────────────────────────────────────────────
   *
   * Forward scan, so the destination index never exceeds the source index and
   * the moves cannot overlap destructively. */
  MFTL_VertexPool *const vp = &obj->verts;
  for (int v = 0; v < n_v_old; ++v)
  {
    const int nv = v_map[v];
    if (nv == MFTL_ERROR || nv == v) continue;

    vp->rec[nv] = vp->rec[v];
    for (int k = 0; k < 3; ++k)
    {
      vp->pos[3 * nv + k] = vp->pos[3 * v + k];
      vp->normal[3 * nv + k] = vp->normal[3 * v + k];
    }
  }

  MFTL_TrianglePool *const tp = &obj->tris;
  for (int t = 0; t < n_t_old; ++t)
  {
    const int nt = t_map[t];
    if (nt == MFTL_ERROR || nt == t) continue;

    tp->rec[nt] = tp->rec[t];
    for (int k = 0; k < 3; ++k) tp->he[3 * nt + k] = tp->he[3 * t + k];

    tp->area[nt] = tp->area[t];
    for (int k = 0; k < 3; ++k)
    {
      tp->normal[3 * nt + k] = tp->normal[3 * t + k];
      tp->center[3 * nt + k] = tp->center[3 * t + k];
    }
  }

  /* ── 3b. Restore the ownership prefix ─────────────────────────────────────
   *
   * The survivors are now [owned][ghost][created]; exchanging the last two
   * blocks makes the owned entities a prefix again. Every sub-array of a pool is
   * permuted the same way, which is what one allocation per pool buys -- there is
   * no list of arrays here that a later addition could be left out of, because
   * the loop below IS the layout.
   *
   * The derived arrays (`verts`, `neighbors`) are deliberately
   * absent: they are rebuilt at the end of this function, so moving them would be
   * work thrown away. */
  if (swap_v || swap_t)
  {
    const size_t need_v = swap_v ? (size_t) MFTL_MIN(vpart.n_ghost, vpart.n_created) * 3 * sizeof(double) : 0;
    const size_t need_t = swap_t ? (size_t) MFTL_MIN(tpart.n_ghost, tpart.n_created) * 3 * sizeof(MFTL_HalfEdge) : 0;

    void *const tmp = MFTLMallocOrDie(MFTL_MAX(MFTL_MAX(need_v, need_t), (size_t) 1), "remesh partition scratch");

    if (swap_v)
    {
      const int f = vpart.n_kept_owned, a = vpart.n_ghost, b = vpart.n_created;
      swap_adjacent_blocks(vp->rec, sizeof(MFTL_Vertex), f, a, b, tmp);
      swap_adjacent_blocks(vp->pos, 3 * sizeof(double), f, a, b, tmp);
      swap_adjacent_blocks(vp->normal, 3 * sizeof(double), f, a, b, tmp);
    }

    if (swap_t)
    {
      const int f = tpart.n_kept_owned, a = tpart.n_ghost, b = tpart.n_created;
      swap_adjacent_blocks(tp->rec, sizeof(MFTL_Triangle), f, a, b, tmp);
      /* Half-edges move as TRIPLES, which is what preserves both the implicit
       * indexing invariant and the implicit half-edge global ids. Expressing that
       * as an element size of three records rather than as three separate swaps
       * makes it impossible to get the grouping wrong. */
      swap_adjacent_blocks(tp->he, 3 * sizeof(MFTL_HalfEdge), f, a, b, tmp);
      swap_adjacent_blocks(tp->area, sizeof(double), f, a, b, tmp);
      swap_adjacent_blocks(tp->normal, 3 * sizeof(double), f, a, b, tmp);
      swap_adjacent_blocks(tp->center, 3 * sizeof(double), f, a, b, tmp);
    }

    free(tmp);

    /* Fold the exchange into the maps, so step 4 remaps straight to the final
     * index space rather than to the intermediate one. */
    if (swap_v)
    {
      for (int v = 0; v < n_v_old; ++v)
      {
        if (v_map[v] != MFTL_ERROR) v_map[v] = partition_remap(&vpart, v_map[v]);
      }
    }
    if (swap_t)
    {
      for (int t = 0; t < n_t_old; ++t)
      {
        if (t_map[t] != MFTL_ERROR) t_map[t] = partition_remap(&tpart, t_map[t]);
      }
    }
  }

  /* ── 4. Remap the indices the records carry ───────────────────────────────── */
  for (int h = 0; h < 3 * n_t_new; ++h)
  {
    MFTL_HalfEdge *const e = &tp->he[h];

    const int v_new = v_map[e->vertex_idx];
    MFTL_ASSERT(v_new != MFTL_ERROR && "MFTLRemeshCompactObject: live half-edge targets a removed vertex");
    e->vertex_idx = v_new;

    if (e->twin_idx == MFTL_ERROR) continue;

    const int t_new = t_map[MFTLHeTriIdx(e->twin_idx)];
    if (t_new == MFTL_ERROR)
    {
      /* A live half-edge twinned to a dead one means an operator removed a
       * triangle without re-gluing its rim. Turning it into a boundary keeps the
       * structure traversable, but the surface now has a hole that no flag
       * records, so this is an assertion first and a fallback second. */
      MFTL_ASSERT(0 && "MFTLRemeshCompactObject: live half-edge twinned to a removed triangle -- rim was not glued");
      e->twin_idx = MFTL_ERROR;
      e->flags |= MFTL_FLAG_BOUNDARY;
      continue;
    }
    e->twin_idx = 3 * t_new + MFTLHeCorner(e->twin_idx);
  }

  /* ── 5. Clear the transient flags ─────────────────────────────────────────── */
  for (int v = 0; v < n_v_new; ++v) vp->rec[v].flags &= ~(MFTL_FLAG_DEAD | MFTL_FLAG_VISITED);
  for (int t = 0; t < n_t_new; ++t) tp->rec[t].flags &= ~(MFTL_FLAG_DEAD | MFTL_FLAG_VISITED);
  for (int h = 0; h < 3 * n_t_new; ++h) tp->he[h].flags &= ~(MFTL_FLAG_DEAD | MFTL_FLAG_VISITED);

  /* ── 6. Publish the new counts, then re-derive out_he ─────────────────────
   *
   * SetCounts before the derivation, so the loop below is bounded by the counts
   * a later reader will use rather than by the ones that just became wrong.
   *
   * The owned count is the two owned runs, whether or not they had to be moved
   * to become adjacent -- which is also what makes this the point at which the
   * entities created by the sweep stop being past the prefix and start being
   * ordinary owned entities. */
  MFTLObjectSetCounts(obj, n_v_new, vpart.n_kept_owned + vpart.n_created, n_t_new, tpart.n_kept_owned + tpart.n_created);

  for (int h = 0; h < 3 * n_t_new; ++h) vp->rec[MFTLHeOrigin(obj, h)].out_he_idx = h;

  free(scratch);

  if (stats)
  {
    stats->n_vertices_removed += n_dead_v;
    stats->n_orphans_removed += n_orphan_v;
  }

  MFTLObjectBuildDerived(obj);
  return removed;
}

void MFTLRemeshRenumberGlobalIds(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(obj->tris.n_local == obj->tris.n && obj->verts.n_local == obj->verts.n &&
              "MFTLRemeshRenumberGlobalIds: object holds ghosts -- the renumbering is a LOCAL decision about a "
              "mesh-wide namespace and is only sound when this rank holds all of it");

  const int n_v = obj->verts.n;
  const int n_t = obj->tris.n;

  for (int v = 0; v < n_v; ++v) obj->verts.rec[v].global_id = v;

  for (int t = 0; t < n_t; ++t)
  {
    obj->tris.rec[t].global_id = t;
    /* Not a free choice: half-edge 3t+k must carry 3 * (triangle gid) + k, which
     * is what lets redistribution reconstruct half-edge ids instead of sending
     * them (inv_halfedge_gids_are_implicit). */
    for (int k = 0; k < 3; ++k) obj->tris.he[3 * t + k].global_id = 3 * t + k;
  }

  obj->next_vertex_gid = n_v;
  obj->next_triangle_gid = n_t;
  obj->next_halfedge_gid = 3 * n_t;

  /* Every id in the body just moved. The assertion above says this rank holds
   * the whole mesh, so nothing is replicated RIGHT NOW -- but the migration that
   * puts it back on the grid is the one that has to observe the new numbering,
   * and it decides per body whether to bother. Tell it. */
  MFTLObjectMarkPartitionDirty(obj);
}
