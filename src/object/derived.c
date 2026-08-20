/**
 * @file derived.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief The derived per-triangle arrays, and the Morton renumbering of the pools.
 *
 * ## WHAT SURVIVED FROM THE PREVIOUS VERSION'S "FLATTEN" MODULE, AND WHY
 *
 * That module produced ten arrays. Four are here. The other six were removed
 * after counting their consumers, which is the only honest way to decide:
 *
 *   flat_vert_pos + flat_pos_unwrapped   DELETED. It was a second copy of every
 *     position, kept in a second index space, sometimes unwrapped and sometimes
 *     not depending on whether the object fitted in half a domain -- a mode a
 *     consumer had to query and branch on. Positions are now the authoritative
 *     SoA array `verts.pos`, in the one index space there is. The seam handling
 *     it existed for lives in `tris.edge` instead, which is exact for EVERY
 *     object regardless of size, because a single triangle is always small
 *     enough to place unambiguously even when the whole surface is not.
 *
 *   sfc_perm + sfc_inv                   DELETED. The permutation is applied to
 *     the pools (MFTLObjectReorderMorton below), so there is nothing to store
 *     and no second index space to confuse with the first. The previous
 *     version's notes record a live silent-corruption bug from exactly that
 *     confusion; it is now unrepresentable.
 *
 *   vert_offsets + vert_tris (CSR)       DELETED. Zero consumers outside its own
 *     unit test. Every vertex-neighbourhood operation in the library goes
 *     through the half-edge one-ring walk, which is what the half-edge structure
 *     is for. Bring it back when a kernel exists that streams it.
 *
 *   flat_he_verts                        DELETED. Zero consumers. The three edge
 *     vectors of a triangle are e1, e2 and e2-e1, all in registers inside any
 *     loop over `tris.edge`.
 *
 * The four that stayed are `tris.verts`, `tris.neighbors`, `tris.anchor` and
 * `tris.edge`. The first two are redundant with the half-edge records and are
 * stored anyway; the accounting is in build_topology_arrays() below.
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
 * ── On storing something derivable ─────────────────────────────────────────
 *
 * Both arrays are functions of the half-edge records at the same index:
 *
 *     verts[3t+k]     == he[3t+(k+2)%3].vertex_idx     (a fixed rotation)
 *     neighbors[3t+k] == he[3t+k].twin_idx / 3         (a fixed arithmetic op)
 *
 * They are stored because of TRAFFIC, not because of instructions. A kernel that
 * wants only connectivity -- IO, a triangle-to-vertex scatter, redistribution's
 * halo analysis, a BVH build -- streams 12 bytes per triangle out of one of
 * these arrays. Reading the same information from the half-edge records means
 * touching all three 16-byte records, 48 bytes per triangle, of which 36 are
 * fields it does not want. On a mesh that does not fit in cache that 4x is the
 * whole cost of the loop.
 *
 * The redundancy is real and is the price. It is bounded and cheap to pay
 * because both arrays are rebuilt by this one function, from one pass, and are
 * marked stale by a single bit the moment topology changes.
 *
 * ── Why this loop is fast ──────────────────────────────────────────────────
 *
 * Implicit half-edge indexing means there is no pointer chase at all: triangle
 * t's three half-edges are at 3t, and their records are contiguous. The loop
 * reads a 48-byte block and writes two 12-byte blocks, all sequential.
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

    /* Slot k is the ORIGIN of half-edge b+k, and origin(h) == target(prev(h)).
     * This is the convention MFTLGetTriangleVertices uses; the two must agree or
     * code that mixes them silently rotates its triangles. */
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

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Derived frames
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Rebuilds `tris.anchor` and `tris.edge` from the current positions.
 *
 * The anchor is corner 0's position; the edges are {v1-v0, v2-v0}, made
 * SEAM-CONSISTENT here so that no consumer ever has to be. That is the array
 * that makes a triangle hot loop branch-free, and unlike a set of unwrapped
 * vertex coordinates it stays exactly correct for an object spanning a periodic
 * axis -- a single triangle is always small enough to place unambiguously even
 * when the whole surface is not.
 *
 * Writing the anchor out costs three sequential stores of a value already in a
 * register (both edge vectors are differences against it) and deletes the last
 * indirection from every consumer that needs an absolute position.
 *
 * ── Two loops, not one branch ──────────────────────────────────────────────
 *
 * The periodic correction is loop-invariant in structure but not in value, so it
 * is hoisted as data rather than as a branch: `inv_len[k]` is zero on a
 * non-periodic axis, which makes nearbyint() return zero and the correction
 * vanish without a test. That form is branchless and vectorises.
 *
 * A domain with no periodic axis at all still gets its own loop, because paying
 * three multiplies, three rounds and three FMAs per component to subtract zero
 * is not free -- and the non-periodic case is common enough to deserve the
 * duplication. This is manual loop unswitching; it is written out because the
 * fast loop must vectorise and a call to MFTLGetMinimumImageVector in the body
 * would stop it.
 *
 * @param obj The mesh. `tris.anchor` and `tris.edge` are rewritten from
 *            `verts.pos`, with the minimum-image correction applied once here.
 */
static void build_frames(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  const int *restrict tv = obj->tris.verts;
  const double *restrict pos = obj->verts.pos;
  double *restrict anc = obj->tris.anchor;
  double *restrict ed = obj->tris.edge;

  const MFTL_Domain *dom = obj->domain;
  const bool periodic = (dom != MFTL_NULL) && (dom->is_periodic[0] || dom->is_periodic[1] || dom->is_periodic[2]);

  if (!periodic)
  {
    for (int t = 0; t < n_tris; ++t)
    {
      const double *restrict p0 = &pos[3 * tv[3 * t + 0]];
      const double *restrict p1 = &pos[3 * tv[3 * t + 1]];
      const double *restrict p2 = &pos[3 * tv[3 * t + 2]];

      anc[3 * t + 0] = p0[0];
      anc[3 * t + 1] = p0[1];
      anc[3 * t + 2] = p0[2];

      ed[6 * t + 0] = p1[0] - p0[0];
      ed[6 * t + 1] = p1[1] - p0[1];
      ed[6 * t + 2] = p1[2] - p0[2];
      ed[6 * t + 3] = p2[0] - p0[0];
      ed[6 * t + 4] = p2[1] - p0[1];
      ed[6 * t + 5] = p2[2] - p0[2];
    }
    return;
  }

  /* Zero on a non-periodic axis: nearbyint(d * 0) == 0, so the correction is a
   * no-op there without a branch. */
  double len[3];
  double inv_len[3];
  for (int k = 0; k < 3; ++k)
  {
    len[k] = dom->domain_size[k];
    inv_len[k] = dom->is_periodic[k] ? dom->inv_domain_size[k] : 0.0;
  }

  for (int t = 0; t < n_tris; ++t)
  {
    const double *restrict p0 = &pos[3 * tv[3 * t + 0]];
    const double *restrict p1 = &pos[3 * tv[3 * t + 1]];
    const double *restrict p2 = &pos[3 * tv[3 * t + 2]];

    anc[3 * t + 0] = p0[0];
    anc[3 * t + 1] = p0[1];
    anc[3 * t + 2] = p0[2];

    for (int k = 0; k < 3; ++k)
    {
      const double d1 = p1[k] - p0[k];
      const double d2 = p2[k] - p0[k];
      ed[6 * t + 0 + k] = d1 - len[k] * nearbyint(d1 * inv_len[k]);
      ed[6 * t + 3 + k] = d2 - len[k] * nearbyint(d2 * inv_len[k]);
    }
  }
}

void MFTLObjectBuildDerived(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  if (obj->tris.n > 0)
  {
    build_topology_arrays(obj);
    build_frames(obj);
  }

  /* An empty mesh is trivially described, so both bits are set: a hot loop over
   * zero triangles is correct, not stale. */
  obj->derived = MFTL_DERIVED_TOPOLOGY | MFTL_DERIVED_FRAMES;
}

void MFTLObjectRefreshFrames(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLObjectRefreshFrames: tris.verts is stale -- call MFTLObjectBuildDerived after a topology change");

  if (obj->tris.n > 0) build_frames(obj);

  obj->derived |= MFTL_DERIVED_FRAMES;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Morton renumbering of the pools
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * Permutation helpers. Each gathers `field` into `scratch` in the new order and
 * copies it back, which is O(n) with sequential writes and one random read
 * stream -- the cheap direction. Permuting in place with cycle following would
 * save the scratch buffer at the cost of a visited bitmap and a random access
 * pattern on BOTH sides.
 */

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
 * @brief Sorts one contiguous segment of Morton keys, carrying the old indices.
 *
 * @param keys   The Morton keys of the segment, sorted ascending in place.
 * @param values On entry the old indices of the segment, on exit permuted so
 *               that `values[i]` is the old index that belongs at new position
 *               `i`.
 * @param count  Elements in the segment.
 */
static void sort_segment64(uint64_t *keys, int *values, int count)
{
  if (count > 1) MFTLSortPairs64(keys, values, count);
}

/**
 * @brief The 32-bit counterpart, for the triangle keys.
 *
 * @param keys   The 32-bit spatial keys, sorted in place.
 * @param values Old indices, permuted alongside the keys.
 * @param count  Elements in the segment.
 */
static void sort_segment32(int *keys, int *values, int count)
{
  if (count > 1) MFTLSortPairs32(keys, values, count);
}

/**
 * @brief Orders the vertices along a Morton curve and remaps every reference.
 * @return true if a permutation was applied.
 *
 * @param      obj        The mesh.
 * @param[out] old_of_new `[verts.n]` old index per new position.
 * @param[out] new_of_old `[verts.n]` new index per old position.
 * @param      scratch    Buffer large enough for the widest per-vertex array.
 * @return `true` if a permutation was applied, `false` if the order was already
 *         Morton (or the mesh is empty).
 */
static bool reorder_vertices(MFTL_Object *obj, int *old_of_new, int *new_of_old, void *scratch)
{
  const int nv = obj->verts.n;
  if (nv < 2) return MFTL_FALSE;

  const double *restrict pos = obj->verts.pos;

  /* Bounding box of THIS object, so the 21 bits per axis are spent where the
   * mesh actually is. An object straddling a periodic seam gets a box as wide as
   * the domain and therefore coarser cells; the ordering is a locality
   * heuristic, so that costs a little cache efficiency and nothing else. */
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

  /* One shared scale for all three axes keeps the Morton cells cubic. Scaling
   * per axis would stretch the curve and make locality worse on flat objects. */
  double range = hi[0] - lo[0];
  if (hi[1] - lo[1] > range) range = hi[1] - lo[1];
  if (hi[2] - lo[2] > range) range = hi[2] - lo[2];
  const double inv_range = (range > MFTL_VERYSMALL) ? MFTL_MORTON_MAX / range : 0.0;

  uint64_t *keys = (uint64_t *) MFTLMallocOrDie((size_t) nv * sizeof(uint64_t), "MFTLObjectReorderMorton vertex keys");

  for (int v = 0; v < nv; ++v)
  {
    const uint64_t ix = (uint64_t) ((pos[3 * v + 0] - lo[0]) * inv_range);
    const uint64_t iy = (uint64_t) ((pos[3 * v + 1] - lo[1]) * inv_range);
    const uint64_t iz = (uint64_t) ((pos[3 * v + 2] - lo[2]) * inv_range);
    keys[v] = MFTLMorton3D(ix, iy, iz);
    old_of_new[v] = v;
  }

  /* Owned and ghost vertices are sorted SEPARATELY, so that [0, n_local) still
   * holds exactly the owned vertices afterwards. Sorting the whole pool at once
   * would interleave them and destroy the prefix invariant that every ownership
   * test in the library relies on. */
  const int n_local = obj->verts.n_local;
  sort_segment64(keys, old_of_new, n_local);
  sort_segment64(keys + n_local, old_of_new + n_local, nv - n_local);

  free(keys);

  for (int i = 0; i < nv; ++i) new_of_old[old_of_new[i]] = i;

  /* Records first, then the SoA arrays. `pos` is authoritative and the rest is
   * cached geometry, but all of it is permuted: a reorder must not silently
   * discard values the caller computed one line earlier. */
  permute_vertex_recs(obj->verts.rec, (MFTL_Vertex *) scratch, old_of_new, nv);
  permute_doubles(obj->verts.pos, (double *) scratch, old_of_new, nv, 3);
  permute_doubles(obj->verts.normal, (double *) scratch, old_of_new, nv, 3);
  permute_doubles(obj->verts.area, (double *) scratch, old_of_new, nv, 1);
  permute_doubles(obj->verts.mean_curv, (double *) scratch, old_of_new, nv, 1);
  permute_doubles(obj->verts.gauss_curv, (double *) scratch, old_of_new, nv, 1);

  /* Every reference INTO the vertex pool. `out_he_idx` is a half-edge index and
   * travelled with its record, so it needs nothing here. */
  MFTL_HalfEdge *restrict he = obj->tris.he;
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h) he[h].vertex_idx = new_of_old[he[h].vertex_idx];

  return MFTL_TRUE;
}

/**
 * @brief Orders the triangles spatially and remaps every reference.
 *
 * The key is the SMALLEST of the triangle's three vertex indices, taken after
 * the vertices have been Morton-ordered. That is a spatial key by construction
 * and it costs three loads and two compares -- no second geometric pass, no
 * centroid, and no question about what a centroid means across a periodic seam.
 * Using min() rather than, say, the first corner makes the key independent of
 * which corner the generator happened to call corner 0.
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
  sort_segment32(keys, old_of_new, n_local);
  sort_segment32(keys + n_local, old_of_new + n_local, nt - n_local);

  free(keys);

  for (int i = 0; i < nt; ++i) new_of_old[old_of_new[i]] = i;

  /* Only the AUTHORITATIVE triangle state is permuted. `verts`, `neighbors`,
   * `anchor` and `edge` are derived and are rebuilt by the caller in one pass,
   * which is cheaper than permuting four arrays and exactly as correct. */
  permute_triangle_recs(obj->tris.rec, (MFTL_Triangle *) scratch, old_of_new, nt);
  permute_he_triples(obj->tris.he, (MFTL_HalfEdge *) scratch, old_of_new, nt);
  permute_doubles(obj->tris.area, (double *) scratch, old_of_new, nt, 1);
  permute_doubles(obj->tris.normal, (double *) scratch, old_of_new, nt, 3);
  permute_doubles(obj->tris.center, (double *) scratch, old_of_new, nt, 3);

  /* Every reference INTO the half-edge array. A half-edge index carries its
   * corner slot in the low position, so the remap keeps h % 3 and moves only the
   * triangle part -- which is precisely the implicit-indexing invariant being
   * maintained through the permutation. */
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

  /* One scratch block for every permutation. Sized for the widest element any
   * permute_* call moves: 6 doubles per triangle (`edge`) and 3 half-edge
   * records per triangle both come to 48 bytes, against 24 for a vertex
   * position. Allocating it once keeps the reorder at two mallocs plus the two
   * the sorts make internally. */
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
}
