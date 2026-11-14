/**
 * @file topology.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_topology
 *
 * @brief Whole-mesh connectivity operations: twin rebuilding, boundary marking,
 *        integrity checking and connected-component labelling. Navigation is
 *        inline in mftl_topology.h.
 */

#include "mftl_topology.h"

#include <stdio.h>
#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"

void MFTLRebuildTwins(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int n_he = 3 * obj->tris.n;
  const int n_vert = obj->verts.n;
  MFTL_HalfEdge *const he = obj->tris.he;

  if (n_he < 1 || n_vert < 1) return;

  /* Half-edges are bucketed by ORIGIN vertex in an intrusive singly linked list:
   *
   *   v_heads[u] = first half-edge whose origin is u
   *   he_next[h] = next half-edge sharing h's origin
   *
   * The twin of (u -> v) is found by walking the bucket of v for an edge ending
   * at u; a bucket is as long as the valence. Both arrays share one allocation. */
  int *scratch = (int *) MFTLMallocOrDie(sizeof(int) * ((size_t) n_vert + (size_t) n_he), "MFTLRebuildTwins");

  int *const v_heads = scratch;
  int *const he_next = scratch + n_vert;

  for (int v = 0; v < n_vert; ++v) v_heads[v] = MFTL_ERROR;

  /* ── Pass 1: bucket by origin, and reset every twin ────────────────────── */
  for (int h = 0; h < n_he; ++h)
  {
    const int u = he[MFTLHePrevIdx(h)].vertex_idx;
    MFTL_ASSERT(u >= 0 && u < n_vert && "MFTLRebuildTwins: half-edge origin out of range");

    he_next[h] = v_heads[u];
    v_heads[u] = h;

    he[h].twin_idx = MFTL_ERROR;  // assume boundary until pass 2 finds a partner
  }

  /* ── Pass 2: pair them off ─────────────────────────────────────────────────
   *
   * Each match links both halves, so already-linked half-edges are skipped.
   * Requiring the candidate to be UNLINKED keeps the relation symmetric on
   * non-manifold input (three or more half-edges on one undirected edge): the
   * surplus stays twin-less for MFTLCheckMeshIntegrity() to report. */
  for (int h = 0; h < n_he; ++h)
  {
    if (he[h].twin_idx != MFTL_ERROR) continue;  // already linked by its partner

    const int u = he[MFTLHePrevIdx(h)].vertex_idx;  // origin
    const int v = he[h].vertex_idx;  // target
    MFTL_ASSERT(v >= 0 && v < n_vert && "MFTLRebuildTwins: half-edge target out of range");

    for (int cand = v_heads[v]; cand != MFTL_ERROR; cand = he_next[cand])
    {
      if (he[cand].vertex_idx != u || he[cand].twin_idx != MFTL_ERROR) continue;

      he[h].twin_idx = cand;
      he[cand].twin_idx = h;
      break;
    }
  }

  free(scratch);

  /* `tris.neighbors` is a function of the twins, so it is stale by definition. */
  MFTLObjectInvalidateDerived(obj);
}

int MFTLMarkMeshBoundary(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(obj->tris.n_local == obj->tris.n &&
              "MFTLMarkMeshBoundary: mesh is partitioned -- the flag can only be derived from a COMPLETE mesh");

  MFTL_Vertex *const vr = obj->verts.rec;
  MFTL_HalfEdge *const he = obj->tris.he;
  const int n_verts = obj->verts.n;
  const int n_he = 3 * obj->tris.n;

  /* Pass 1: clear, so a re-run replaces the previous marking. */
  for (int i = 0; i < n_verts; ++i) vr[i].flags &= ~MFTL_FLAG_BOUNDARY;
  for (int i = 0; i < n_he; ++i) he[i].flags &= ~MFTL_FLAG_BOUNDARY;

  /* Pass 2: set on every twin-less half-edge and both its endpoints. */
  int n_boundary = 0;
  for (int h = 0; h < n_he; ++h)
  {
    if (he[h].twin_idx != MFTL_ERROR) continue;

    he[h].flags |= MFTL_FLAG_BOUNDARY;
    n_boundary++;

    const int target = he[h].vertex_idx;
    const int origin = he[MFTLHePrevIdx(h)].vertex_idx;

    MFTL_ASSERT(target >= 0 && target < n_verts && origin >= 0 && origin < n_verts);
    vr[target].flags |= MFTL_FLAG_BOUNDARY;
    vr[origin].flags |= MFTL_FLAG_BOUNDARY;
  }

  return n_boundary;
}

/**
 * @brief The edge-level half of MFTLCheckMeshIntegrity(): every undirected edge
 *        carries at most two half-edges, in opposite directions and linked as
 *        twins, and no two triangles have the same three corners.
 *
 * Each statement holds for any SUBSET of a valid mesh, so it is checked on a
 * partition with its halo exactly as on a complete mesh: a rank holds each
 * triangle once, a sub-mesh of an oriented 2-manifold has no edge with three
 * half-edges or two in one direction, and redistribution links every pair whose
 * two triangles a rank holds (the twins a full MFTLRebuildTwins() would make). A
 * twin-less half-edge whose opposite is NOT in the pool -- an open surface's rim,
 * a halo's outer edge -- stays legal.
 *
 * Half-edges are bucketed by origin, as in MFTLRebuildTwins(); every edge is
 * reported once, from its lowest half-edge index, and every duplicated pair of
 * triangles once, from the lower one's first half-edge. Half-edges with an
 * out-of-range target or origin were reported by the caller and are skipped.
 *
 * @param obj The mesh.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR after printing every violation.
 */
static int check_edge_incidence(const MFTL_Object *obj)
{
  const MFTL_HalfEdge *const he = obj->tris.he;
  const int n_verts = obj->verts.n;
  const int n_hes = 3 * obj->tris.n;
  if (n_hes < 1 || n_verts < 1) return MFTL_SUCCESS;

  int *const scratch = (int *) MFTLMallocOrDie(sizeof(int) * ((size_t) n_verts + (size_t) n_hes), "MFTLCheckMeshIntegrity");
  int *const v_heads = scratch;  // first half-edge with origin v
  int *const he_next = scratch + n_verts;  // next half-edge with the same origin

  for (int v = 0; v < n_verts; ++v) v_heads[v] = MFTL_ERROR;
  for (int h = n_hes - 1; h >= 0; --h)
  {
    he_next[h] = MFTL_ERROR;
    const int u = he[MFTLHePrevIdx(h)].vertex_idx, v = he[h].vertex_idx;
    if (u < 0 || u >= n_verts || v < 0 || v >= n_verts) continue;
    he_next[h] = v_heads[u];
    v_heads[u] = h;
  }

  int ok = MFTL_SUCCESS;
  for (int h = 0; h < n_hes; ++h)
  {
    const int u = he[MFTLHePrevIdx(h)].vertex_idx, v = he[h].vertex_idx;
    if (u < 0 || u >= n_verts || v < 0 || v >= n_verts) continue;

    const int gu = obj->verts.rec[u].global_id, gv = obj->verts.rec[v].global_id;
    if (u == v)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d starts and ends at vertex %d: its triangle repeats a corner\n", he[h].global_id, gu);
      ok = MFTL_ERROR;
      continue;
    }

    const int t = MFTLHeTriIdx(h);
    const int third = he[MFTLHeNextIdx(h)].vertex_idx;  // the corner of t off this edge
    const int first_of_t = (h == MFTLTriFirstHe(t));  // duplicates are reported from here only

    /* Everything else on the undirected edge {u, v}: same direction in u's
     * bucket, the opposite one in v's. */
    int n_same = 0, n_opp = 0, lowest = h, same = MFTL_ERROR, opp = MFTL_ERROR;
    for (int g = v_heads[u]; g != MFTL_ERROR; g = he_next[g])
    {
      if (g == h || he[g].vertex_idx != v) continue;
      n_same++;
      same = g;
      if (g < lowest) lowest = g;
      const int s = MFTLHeTriIdx(g);
      if (first_of_t && s > t && he[MFTLHeNextIdx(g)].vertex_idx == third)
      {
        fprintf(stderr, "[MeshCheck] Triangles %d and %d have the same three corners, in the same order (a duplicated triangle)\n", obj->tris.rec[t].global_id,
                obj->tris.rec[s].global_id);
        ok = MFTL_ERROR;
      }
    }
    for (int g = v_heads[v]; g != MFTL_ERROR; g = he_next[g])
    {
      if (he[g].vertex_idx != u) continue;
      n_opp++;
      opp = g;
      if (g < lowest) lowest = g;
      const int s = MFTLHeTriIdx(g);
      if (first_of_t && s > t && he[MFTLHeNextIdx(g)].vertex_idx == third)
      {
        fprintf(stderr, "[MeshCheck] Triangles %d and %d have the same three corners, in opposite orders (a two-triangle pillow)\n", obj->tris.rec[t].global_id,
                obj->tris.rec[s].global_id);
        ok = MFTL_ERROR;
      }
    }

    if (h != lowest) continue;  // the edge is reported once, from its lowest half-edge

    const int n_on_edge = 1 + n_same + n_opp;
    if (n_on_edge > 2)
    {
      fprintf(stderr, "[MeshCheck] Edge %d-%d is shared by %d half-edges: a non-manifold edge\n", gu, gv, n_on_edge);
      ok = MFTL_ERROR;
    }
    else if (n_same == 1)
    {
      fprintf(stderr, "[MeshCheck] Half-edges %d and %d both run from vertex %d to %d: their triangles are oriented inconsistently\n", he[h].global_id,
              he[same].global_id, gu, gv);
      ok = MFTL_ERROR;
    }
    else if (n_opp == 1 && (he[h].twin_idx != opp || he[opp].twin_idx != h))
    {
      fprintf(stderr, "[MeshCheck] Half-edges %d and %d run in opposite directions along edge %d-%d but are not each other's twin\n", he[h].global_id,
              he[opp].global_id, gu, gv);
      ok = MFTL_ERROR;
    }
  }

  free(scratch);
  return ok;
}

int MFTLCheckMeshIntegrity(const MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  int ok = MFTL_SUCCESS;
  int n_boundary = 0;

  const MFTL_Vertex *const vr = obj->verts.rec;
  const MFTL_HalfEdge *const he = obj->tris.he;

  const int n_verts = obj->verts.n;
  const int n_tris = obj->tris.n;
  const int n_hes = 3 * n_tris;

  /* The next-cycle, the face back-reference and "every triangle has half-edges"
   * are tautologies under implicit indexing and need no check. */

  /* ── Vertices ─────────────────────────────────────────────────────────── */
  for (int v = 0; v < n_verts; ++v)
  {
    const int oh = vr[v].out_he_idx;

    if (oh == MFTL_ERROR)
    {
      fprintf(stderr, "[MeshCheck] Vertex %d has no outgoing half-edge\n", vr[v].global_id);
      ok = MFTL_ERROR;
    }
    else if (oh < 0 || oh >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Vertex %d has out-of-bounds half-edge index %d (max %d)\n", vr[v].global_id, oh, n_hes);
      ok = MFTL_ERROR;
    }
    else if (he[MFTLHePrevIdx(oh)].vertex_idx != v)
    {
      /* out_he_idx must be OUTGOING from v. A stale value (a topology edit that
       * forgot the vertex) makes every one-ring walk from v traverse another fan. */
      fprintf(stderr, "[MeshCheck] Vertex %d: half-edge %d does not start at it\n", vr[v].global_id, oh);
      ok = MFTL_ERROR;
    }
  }

  /* ── Half-edges ───────────────────────────────────────────────────────── */
  for (int h = 0; h < n_hes; ++h)
  {
    if (he[h].vertex_idx < 0 || he[h].vertex_idx >= n_verts)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has invalid target vertex index %d\n", he[h].global_id, he[h].vertex_idx);
      ok = MFTL_ERROR;
      continue;
    }

    const int tw = he[h].twin_idx;

    if (tw == MFTL_ERROR)
    {
      n_boundary++;
      continue;  // a genuine boundary edge: nothing further to compare
    }

    if (tw < 0 || tw >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has out-of-bounds twin index %d\n", he[h].global_id, tw);
      ok = MFTL_ERROR;
      continue;
    }

    if (he[tw].twin_idx != h)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d twin relation is not symmetric\n", he[h].global_id);
      ok = MFTL_ERROR;
    }

    /* The origin of h must be the target of its twin, and vice versa. */
    const int h_origin = he[MFTLHePrevIdx(h)].vertex_idx;
    const int tw_origin = he[MFTLHePrevIdx(tw)].vertex_idx;

    if (h_origin != he[tw].vertex_idx || he[h].vertex_idx != tw_origin)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d and its twin disagree about their endpoints\n", he[h].global_id);
      ok = MFTL_ERROR;
    }
  }

  if (check_edge_incidence(obj) != MFTL_SUCCESS) ok = MFTL_ERROR;

  if (ok == MFTL_SUCCESS)
  {
    /* Twin-less half-edges are legal (open surfaces); report their count. */
    if (n_boundary == 0)
      printf("[MeshCheck] Mesh is consistent and closed (%d vertices, %d triangles).\n", n_verts, n_tris);
    else
      printf("[MeshCheck] Mesh is consistent and open (%d vertices, %d triangles, %d boundary half-edges).\n", n_verts, n_tris, n_boundary);
  }

  return ok;
}

int MFTLLabelConnectedComponents(const MFTL_Object *obj, int *labels)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int nt = obj->tris.n;
  if (nt <= 0) return 0;

  /* One allocation: the labels when the caller wants none, then the queue. */
  int *const scratch = (int *) MFTLMallocOrDie((size_t) (labels ? nt : 2 * nt) * sizeof(int), "MFTLLabelConnectedComponents");
  int *const lab = labels ? labels : scratch + nt;
  int *const queue = scratch;

  for (int t = 0; t < nt; ++t) lab[t] = MFTL_ERROR;

  const MFTL_HalfEdge *const he = obj->tris.he;
  int n_comp = 0;

  /* Seeds in ascending index: a component's label is the rank of its smallest
   * triangle, a function of the pool alone. */
  for (int seed = 0; seed < nt; ++seed)
  {
    if (lab[seed] != MFTL_ERROR) continue;

    int head = 0, tail = 0;
    queue[tail++] = seed;
    lab[seed] = n_comp;

    while (head < tail)
    {
      const int t = queue[head++];
      for (int k = 0; k < 3; ++k)
      {
        const int tw = he[3 * t + k].twin_idx;
        if (tw == MFTL_ERROR) continue;
        const int u = MFTLHeTriIdx(tw);
        MFTL_ASSERT(u >= 0 && u < nt && "MFTLLabelConnectedComponents: twin outside the pool");
        if (lab[u] != MFTL_ERROR) continue;
        lab[u] = n_comp;
        queue[tail++] = u;
      }
    }
    n_comp++;
  }

  free(scratch);
  return n_comp;
}
