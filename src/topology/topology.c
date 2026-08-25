/**
 * @file topology.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_topology
 *
 * @brief Half-edge connectivity maintenance and validation.
 *
 * The navigation accessors live inline in mftl_topology.h; this file holds the
 * three operations that need real work.
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

  /* ── Allocation ───────────────────────────────────────────────────────────
   *
   * Half-edges are bucketed by their ORIGIN vertex through an intrusive singly
   * linked list -- an O(V + H) build with no hashing and no per-edge allocation:
   *
   *   v_heads[u] = index of the first half-edge whose origin is u
   *   he_next[h] = index of the next half-edge sharing h's origin
   *
   * Finding the twin of (u -> v) is then a walk of the bucket of v looking for
   * the edge that ends at u. Bucket length is the vertex valence, so this is
   * effectively constant time per half-edge.
   *
   * One allocation for both arrays: same lifetime, and two separate multi-MB
   * mallocs would be two mmap/first-touch fault storms per call.
   *
   * A third array caching each half-edge's ORIGIN would pay for itself if
   * reaching it cost two dependent pointer hops. With implicit indexing it is one
   * arithmetic op and one load from a record in the same 64-byte neighbourhood,
   * so the cache array is more expensive than the thing it avoids. */
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
   * Twins come in PAIRS, so a search that finds (v -> u) for (u -> v) has found
   * both halves of the answer: writing the partner as well and skipping any
   * half-edge that is already linked halves the number of bucket walks.
   *
   * Requiring the candidate to be UNLINKED is what keeps the relation symmetric
   * on non-manifold input, where three or more half-edges share an undirected
   * edge. Pairing them off first-come-first-served leaves the surplus as
   * boundary, which MFTLCheckMeshIntegrity can report; a one-directional search
   * would silently produce twin links that do not point back at each other. */
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

  /* Pass 1: clear. Unconditional, so a re-run after a topological edit retires
   * the previous marking instead of accumulating on top of it. */
  for (int i = 0; i < n_verts; ++i) vr[i].flags &= ~MFTL_FLAG_BOUNDARY;
  for (int i = 0; i < n_he; ++i) he[i].flags &= ~MFTL_FLAG_BOUNDARY;

  /* Pass 2: set. The vertex writes are scattered, but only for boundary edges --
   * O(sqrt(T)) of them on a patch, none at all on a closed surface. */
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

  /* ── What is NOT checked here, because it cannot be wrong ─────────────────
   *
   * An explicitly-indexed half-edge mesh has to validate the next-cycle of every
   * triangle, the triangle <-> half-edge back-reference, and that every triangle
   * has a half-edge at all. Implicit indexing (mftl_topology.h) makes all three
   * tautologies: next(3t+k) is computed, not stored, and half-edge h belongs to
   * triangle h/3 by definition. Three of the six failure modes such a checker
   * looks for are not expressible here. */

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
      /* out_he_idx must be OUTGOING from this vertex, i.e. its origin is v. A
       * stale value here is the classic symptom of a topology edit that moved a
       * half-edge without updating the vertex that pointed at it, and it makes
       * every one-ring walk from this vertex silently traverse someone else's
       * fan. */
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
    /* Reporting the count rather than asserting "closed": an open surface is a
     * legitimate mesh, and the number is what tells a caller whether the mesh it
     * believed to be closed actually is. */
    if (n_boundary == 0)
      printf("[MeshCheck] Mesh is consistent and closed (%d vertices, %d triangles).\n", n_verts, n_tris);
    else
      printf("[MeshCheck] Mesh is consistent and open (%d vertices, %d triangles, %d boundary half-edges).\n", n_verts, n_tris, n_boundary);
  }

  return ok;
}
