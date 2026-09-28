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
