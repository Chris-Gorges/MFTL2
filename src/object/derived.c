/**
 * @file derived.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief The derived connectivity arrays, and the Morton renumbering of the pools.
 *
 * Only two derived arrays exist, `tris.verts` and `tris.neighbors`. Everything
 * else a "flattened" half-edge mesh is often given was removed after counting
 * consumers: a second (sometimes unwrapped) copy of the positions and a stored
 * SFC permutation (both replaced by one index space, see
 * MFTLObjectReorderMorton()), per-half-edge vertex pairs (edges are formed in
 * registers), and a vertex-to-triangle CSR (measured as a dead heat against the
 * triangle scatter for vertex normals, docs/analysis.md §1.5).
 */

#include "mftl_object.h"

#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Derived connectivity
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Emits `tris.verts` and `tris.neighbors` in one streaming pass.
 *
 * Both are functions of the half-edge records at the same index:
 *
 *     verts[3t+k]     == he[3t+(k+2)%3].vertex_idx     origin of half-edge 3t+k
 *     neighbors[3t+k] == he[3t+k].twin_idx / 3         triangle across it
 *
 * They are stored for TRAFFIC: a connectivity-only loop streams 12 B per
 * triangle from one of them instead of the 48 B of three half-edge records.
 *
 * `rebuild_object()` in src/communication/redistribute.c emits the same two
 * arrays inside its relink loop; `inv_derived_matches_topology()`
 * (tests/test_mesh_invariants.h) pins both spellings to this convention.
 *
 * @param obj The mesh. `tris.verts` and `tris.neighbors` are rewritten.
 */
static void build_topology_arrays(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  const MFTL_HalfEdge *restrict he = obj->tris.he;
  int *restrict tv = obj->tris.verts;
  int *restrict tn = obj->tris.neighbors;

  for (int t = 0; t < n_tris; ++t)
  {
    const int b = 3 * t;

    /* Slot k is the ORIGIN of half-edge b+k: origin(h) == target(prev(h)), the
     * convention of MFTLGetTriangleVertices(). */
    tv[b + 0] = he[b + 2].vertex_idx;
    tv[b + 1] = he[b + 0].vertex_idx;
    tv[b + 2] = he[b + 1].vertex_idx;

    const int w0 = he[b + 0].twin_idx;
    const int w1 = he[b + 1].twin_idx;
    const int w2 = he[b + 2].twin_idx;
    tn[b + 0] = (w0 == MFTL_ERROR) ? MFTL_ERROR : MFTLHeTriIdx(w0);
    tn[b + 1] = (w1 == MFTL_ERROR) ? MFTL_ERROR : MFTLHeTriIdx(w1);
    tn[b + 2] = (w2 == MFTL_ERROR) ? MFTL_ERROR : MFTLHeTriIdx(w2);
  }
}

void MFTLObjectBuildDerived(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  if (obj->tris.n > 0) build_topology_arrays(obj);

  /* An empty mesh is trivially described, so the bit is set. The geometry bit is
   * left alone: rebuilding connectivity moves nothing, and every edit that does
   * change a triangle's corners has already cleared it. */
  obj->derived = MFTL_DERIVED_TOPOLOGY | (obj->derived & MFTL_DERIVED_GEOMETRY);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Morton renumbering of the pools
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Permutation helpers: gather into scratch in the new order, copy back. O(n),
 * sequential writes and one random read stream. */

/**
 * @brief Permutes an interleaved double array by gather-then-copy-back.
 *
 * @param field      The array, permuted in place.
 * @param scratch    Buffer of at least `n * comp` doubles.
 * @param old_of_new `old_of_new[i]` is the OLD index of the entity that belongs
 *                   at new position `i`.
 * @param n          Number of entities.
 * @param comp       Components per entity: 1 for a scalar, 3 for a vector.
 */
static void permute_doubles(double *field, double *scratch, const int *old_of_new, int n, int comp)
{
  for (int i = 0; i < n; ++i)
  {
    const double *restrict s = &field[(size_t) comp * old_of_new[i]];
    double *restrict d = &scratch[(size_t) comp * i];
    for (int c = 0; c < comp; ++c) d[c] = s[c];
  }
  memcpy(field, scratch, (size_t) n * (size_t) comp * sizeof(double));
}

/**
 * @brief Permutes the vertex AoS records.
 * @param field      The record array, permuted in place.
 * @param scratch    Buffer of at least @p n records.
 * @param old_of_new Old index of the record belonging at each new position.
 * @param n          Number of vertices.
 */
static void permute_vertex_recs(MFTL_Vertex *field, MFTL_Vertex *scratch, const int *old_of_new, int n)
{
  for (int i = 0; i < n; ++i) scratch[i] = field[old_of_new[i]];
  memcpy(field, scratch, (size_t) n * sizeof(MFTL_Vertex));
}

/**
 * @brief Permutes the triangle AoS records.
 * @param field      The record array, permuted in place.
 * @param scratch    Buffer of at least @p n records.
 * @param old_of_new Old index of the record belonging at each new position.
 * @param n          Number of triangles.
 */
static void permute_triangle_recs(MFTL_Triangle *field, MFTL_Triangle *scratch, const int *old_of_new, int n)
{
  for (int i = 0; i < n; ++i) scratch[i] = field[old_of_new[i]];
  memcpy(field, scratch, (size_t) n * sizeof(MFTL_Triangle));
}

/**
 * @brief Moves each triangle's three half-edges as one unit, preserving 3t+k.
 *
 * @param field      The half-edge array, permuted in place.
 * @param scratch    Buffer of at least `3 * n_tris` half-edges.
 * @param old_of_new Old TRIANGLE index belonging at each new triangle position.
 * @param n_tris     Number of triangles.
 *
 * @note Moving the triples as units is what preserves BOTH the implicit
 *       indexing invariant and the implicit half-edge global ids.
 */
static void permute_he_triples(MFTL_HalfEdge *field, MFTL_HalfEdge *scratch, const int *old_of_new, int n_tris)
{
  for (int i = 0; i < n_tris; ++i)
  {
    const MFTL_HalfEdge *restrict s = &field[3 * old_of_new[i]];
    MFTL_HalfEdge *restrict d = &scratch[3 * i];
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
  }
  memcpy(field, scratch, 3 * (size_t) n_tris * sizeof(MFTL_HalfEdge));
}

/**
 * @brief Orders the vertices along a Morton curve and remaps every reference.
 *
 * @param      obj        The mesh.
 * @param[out] old_of_new `[verts.n]` old index per new position.
 * @param[out] new_of_old `[verts.n]` new index per old position.
 * @param      scratch    Buffer large enough for the widest per-vertex array.
 * @return `true` if a permutation was applied (`false` for fewer than 2 vertices).
 */
static bool reorder_vertices(MFTL_Object *obj, int *old_of_new, int *new_of_old, void *scratch)
{
  const int nv = obj->verts.n;
  if (nv < 2) return MFTL_FALSE;

  const double *restrict pos = obj->verts.pos;

  /* Bounding box of THIS object, so the 21 bits per axis are spent where the
   * mesh is. A seam-straddling object gets a domain-wide box and coarser cells:
   * a locality cost only. */
  double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int v = 0; v < nv; ++v)
  {
    for (int k = 0; k < 3; ++k)
    {
      const double c = pos[3 * v + k];
      if (c < lo[k]) lo[k] = c;
      if (c > hi[k]) hi[k] = c;
    }
  }

  /* One scale for all axes keeps the Morton cells cubic. */
  double range = hi[0] - lo[0];
  if (hi[1] - lo[1] > range) range = hi[1] - lo[1];
  if (hi[2] - lo[2] > range) range = hi[2] - lo[2];
  /* The only condition is a FINITE scale; an absolute floor would collapse every
   * key of a small object onto one value. */
  const double inv_range = (range > MFTL_MORTON_MAX / MFTL_DBL_MAX) ? MFTL_MORTON_MAX / range : 0.0;

  uint64_t *keys = (uint64_t *) MFTLMallocOrDie((size_t) nv * sizeof(uint64_t), "MFTLObjectReorderMorton vertex keys");

  for (int v = 0; v < nv; ++v)
  {
    const uint64_t ix = (uint64_t) ((pos[3 * v + 0] - lo[0]) * inv_range);
    const uint64_t iy = (uint64_t) ((pos[3 * v + 1] - lo[1]) * inv_range);
    const uint64_t iz = (uint64_t) ((pos[3 * v + 2] - lo[2]) * inv_range);
    keys[v] = MFTLMorton3D(ix, iy, iz);
    old_of_new[v] = v;
  }

  /* Owned and ghost vertices are sorted SEPARATELY to keep the ownership prefix. */
  const int n_local = obj->verts.n_local;
  MFTLSortPairs64(keys, old_of_new, n_local);
  MFTLSortPairs64(keys + n_local, old_of_new + n_local, nv - n_local);

  free(keys);

  for (int i = 0; i < nv; ++i) new_of_old[old_of_new[i]] = i;

  /* `normal` is computed geometry but is permuted too, so a reorder never
   * discards values the caller just computed. */
  permute_vertex_recs(obj->verts.rec, (MFTL_Vertex *) scratch, old_of_new, nv);
  permute_doubles(obj->verts.pos, (double *) scratch, old_of_new, nv, 3);
  permute_doubles(obj->verts.normal, (double *) scratch, old_of_new, nv, 3);

  /* Every reference INTO the vertex pool. (`out_he_idx` is a half-edge index;
   * it is remapped with the triangles.) */
  MFTL_HalfEdge *restrict he = obj->tris.he;
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h) he[h].vertex_idx = new_of_old[he[h].vertex_idx];

  return MFTL_TRUE;
}

/**
 * @brief Orders the triangles spatially and remaps every reference.
 *
 * The key is the SMALLEST of the triangle's three (already Morton-ordered)
 * vertex indices: spatial by construction, three loads and two compares, no
 * centroid (and no question of what a centroid means across a periodic seam),
 * and independent of which corner is called corner 0.
 *
 * @param      obj        The mesh.
 * @param[out] old_of_new `[tris.n]` old index per new position.
 * @param[out] new_of_old `[tris.n]` new index per old position.
 * @param      scratch    Buffer large enough for the widest per-triangle array.
 * @return `true` if a permutation was applied.
 */
static bool reorder_triangles(MFTL_Object *obj, int *old_of_new, int *new_of_old, void *scratch)
{
  const int nt = obj->tris.n;
  if (nt < 2) return MFTL_FALSE;

  MFTL_HalfEdge *restrict he = obj->tris.he;

  int *keys = (int *) MFTLMallocOrDie((size_t) nt * sizeof(int), "MFTLObjectReorderMorton triangle keys");

  for (int t = 0; t < nt; ++t)
  {
    const int a = he[3 * t + 0].vertex_idx;
    const int b = he[3 * t + 1].vertex_idx;
    const int c = he[3 * t + 2].vertex_idx;
    int m = a;
    if (b < m) m = b;
    if (c < m) m = c;
    keys[t] = m;
    old_of_new[t] = t;
  }

  const int n_local = obj->tris.n_local;
  MFTLSortPairs32(keys, old_of_new, n_local);
  MFTLSortPairs32(keys + n_local, old_of_new + n_local, nt - n_local);

  free(keys);

  for (int i = 0; i < nt; ++i) new_of_old[old_of_new[i]] = i;

  /* `verts` and `neighbors` are not permuted; the caller rebuilds them. */
  permute_triangle_recs(obj->tris.rec, (MFTL_Triangle *) scratch, old_of_new, nt);
  permute_he_triples(obj->tris.he, (MFTL_HalfEdge *) scratch, old_of_new, nt);
  permute_doubles(obj->tris.area, (double *) scratch, old_of_new, nt, 1);
  permute_doubles(obj->tris.normal, (double *) scratch, old_of_new, nt, 3);
  permute_doubles(obj->tris.center, (double *) scratch, old_of_new, nt, 3);

  /* Every reference INTO the half-edge array: keep the corner h % 3, remap the
   * triangle h / 3 (implicit indexing survives the permutation). */
  he = obj->tris.he;
  const int n_he = 3 * nt;
  for (int h = 0; h < n_he; ++h)
  {
    const int tw = he[h].twin_idx;
    if (tw != MFTL_ERROR) he[h].twin_idx = 3 * new_of_old[tw / 3] + tw % 3;
  }

  MFTL_Vertex *restrict vr = obj->verts.rec;
  const int nv = obj->verts.n;
  for (int v = 0; v < nv; ++v)
  {
    const int oh = vr[v].out_he_idx;
    if (oh != MFTL_ERROR) vr[v].out_he_idx = 3 * new_of_old[oh / 3] + oh % 3;
  }

  return MFTL_TRUE;
}

void MFTLObjectReorderMorton(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int nv = obj->verts.n;
  const int nt = obj->tris.n;
  if (nv < 2 && nt < 2) return;

  const int n_max = (nv > nt) ? nv : nt;

  /* One scratch block for every permutation, sized for the widest element moved:
   * a triangle's three half-edge records, 48 bytes. */
  const size_t widest = 48;
  void *scratch = MFTLMallocOrDie((size_t) n_max * widest, "MFTLObjectReorderMorton scratch");

  int *old_of_new = (int *) MFTLMallocOrDie(2 * (size_t) n_max * sizeof(int), "MFTLObjectReorderMorton maps");
  int *new_of_old = old_of_new + n_max;

  reorder_vertices(obj, old_of_new, new_of_old, scratch);
  reorder_triangles(obj, old_of_new, new_of_old, scratch);

  free(old_of_new);
  free(scratch);

  /* The derived arrays were deliberately not permuted; rebuild them instead. */
  MFTLObjectBuildDerived(obj);
  MFTLObjectMarkChanged(obj);
}
