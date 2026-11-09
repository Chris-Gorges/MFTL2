/**
 * @file remesh_project.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_remeshing
 *
 * @brief Back-projection of smoothed vertices onto the Phong surface of the
 *        mesh a remeshing call received (#MFTL_REMESH_PROJECT_PHONG).
 *
 * ## Why the reference is the call's input, and why it is curved
 *
 * Botsch & Kobbelt project smoothed vertices back onto the ORIGINAL surface,
 * kept as a mesh. In a graphics remesher that is a fixed, usually finer mesh. In
 * front tracking it is the previous step's mesh at the same resolution, and its
 * flat triangles are chords: every call puts the smoothed vertices on them and
 * the next call starts from there (the LeVeque period lost 12.5 % of its volume
 * at refinement 4 that way, against 3.2 % without projection). Its Phong
 * surface passes through the same vertices with their normals and is fourth-order
 * accurate on a sphere, so the projection re-lifts what this call's collapses
 * and flips cut off instead (docs/analysis.md §14.9). A reference rebuilt from the
 * CURRENT mesh does not: it has no record of what the call removed.
 *
 * ## Once per call, after everything else
 *
 * The projection runs once, after the call's last smoothing sweep: serially at
 * the end of MFTLRemeshIterate(), distributed after the spanning driver's closing
 * redistribution. Every owned free vertex is moved to the closest point of the
 * reference's flat triangles within the BRIDGE distance -- half its shortest
 * incident edge, capped at MFTLRemeshProjectionRadius() -- among the triangles
 * whose normal does not point against the vertex's (a thin sheet's other face
 * is never the answer), and then lifted onto the Phong surface there.
 *
 * ## Where the reference lives, and why that is a proof rather than a hope
 *
 * A distributed call projects on the canonical partition, where every owned
 * vertex lies in its rank's cell. MFTLRemeshEnsembleReferenceBuild() gives each
 * rank the input triangles it OWNED when the call started plus every input
 * triangle a neighbour owned within the cap of its cell, shipped once. Every
 * triangle within the bridge distance of an owned vertex is therefore present,
 * however the call moved, split or migrated the mesh in between: no lookup by
 * global id, no walk that can stop at the edge of a rank's copy. The first
 * distributed version -- a per-rank copy with a one-ring halo, searched from the
 * vertex's own reference triangle -- missed about 1 % of the vertices at np = 2-8
 * and lost volume in proportion (docs/analysis.md §5.8).
 *
 * ## The search
 *
 * A uniform grid whose cell is at least the widest triangle box and twice the
 * cap: a triangle is filed under every cell its box overlaps (at most eight), a
 * query visits every cell the box of its ball overlaps (at most eight), and any
 * triangle within the ball shares one of them. The occupied cells are a sorted
 * key array searched by bisection -- no hash map -- each owning a contiguous
 * run of entries that carry the triangle's corner 0 and its box, so the test
 * that rejects most candidates streams instead of gathering. A query meets
 * about 45 entries, 19 pass the box test and 8 of those are distinct triangles
 * (a uniform sphere mesh); each triangle is evaluated once, and only the box,
 * the orientation and the plane can reject one, all against the fixed radius:
 * the candidate set, and with it the answer, is the same in every visiting
 * order. Ties are broken by triangle global id, so the answer does not depend
 * on which rank, or which order, found it.
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_vectoroperations.h"
#include "../object/fields_internal.h"
#include "remesh_comm.h"
#include "remesh_internal.h"

/** @brief Staged vertex record width: global id, position, normal. */
#define REF_VREC 7

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Building one reference
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTLRemeshProjectionRadius(const MFTL_RemeshParameters *params) { return 0.5 * params->max_edge_length; }

void MFTLRemeshReferenceInit(MFTL_RemeshReference *ref, const MFTL_Domain *domain)
{
  MFTL_ASSERT(ref != MFTL_NULL);
  memset(ref, 0, sizeof(*ref));
  ref->domain = domain;
}

/** @brief Room for @p add more staged vertices and @p add_t more staged triangles. */
static void stage_reserve(MFTL_RemeshReference *ref, int add_v, int add_t)
{
  if (ref->sv_n + add_v > ref->sv_cap)
  {
    ref->sv_cap = MFTL_MAX(2 * ref->sv_cap, ref->sv_n + add_v);
    ref->sv = (double *) MFTLReallocOrDie(ref->sv, (size_t) ref->sv_cap * REF_VREC * sizeof(double), "remesh reference staged vertices");
  }
  if (ref->st_n + add_t > ref->st_cap)
  {
    ref->st_cap = MFTL_MAX(2 * ref->st_cap, ref->st_n + add_t);
    ref->st = (int *) MFTLReallocOrDie(ref->st, (size_t) ref->st_cap * 4 * sizeof(int), "remesh reference staged triangles");
  }
}

/** @brief Does vertex @p v of @p obj lift the Phong surface, or is its normal undefined (crease, rim half-fan)? */
static inline bool corner_lifts(const MFTL_Object *obj, int v)
{
  return !(obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER | MFTL_FLAG_BOUNDARY | MFTL_FLAG_DEAD));
}

void MFTLRemeshReferenceAddObject(MFTL_RemeshReference *ref, const MFTL_Object *obj, bool owned_only)
{
  MFTL_ASSERT(ref != MFTL_NULL && obj != MFTL_NULL);
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLRemeshReferenceAddObject: derived topology is stale");

  const int nt = owned_only ? obj->tris.n_local : obj->tris.n;
  const int nv = obj->verts.n;

  /* Every vertex once, whether or not a staged triangle uses it: the finishing
   * pass removes duplicates by id anyway, and an unused vertex costs 56 bytes. */
  stage_reserve(ref, nv, nt);
  for (int v = 0; v < nv; ++v)
  {
    double *e = &ref->sv[(size_t) (ref->sv_n++) * REF_VREC];
    e[0] = (double) obj->verts.rec[v].global_id;
    const bool lift = corner_lifts(obj, v);
    for (int k = 0; k < 3; ++k)
    {
      e[1 + k] = obj->verts.pos[3 * v + k];
      e[4 + k] = lift ? obj->verts.normal[3 * v + k] : 0.0;
    }
  }
  for (int t = 0; t < nt; ++t)
  {
    if (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) continue;
    int *e = &ref->st[(size_t) (ref->st_n++) * 4];
    e[0] = obj->tris.rec[t].global_id;
    for (int k = 0; k < 3; ++k) e[1 + k] = obj->verts.rec[obj->tris.verts[3 * t + k]].global_id;
  }
}

void MFTLRemeshReferencePackTriangle(const MFTL_Object *obj, int t, double *rec)
{
  rec[0] = (double) obj->global_id;
  rec[1] = (double) obj->tris.rec[t].global_id;
  for (int k = 0; k < 3; ++k)
  {
    const int v = obj->tris.verts[3 * t + k];
    double *c = &rec[2 + 7 * k];
    const bool lift = corner_lifts(obj, v);
    c[0] = (double) obj->verts.rec[v].global_id;
    for (int j = 0; j < 3; ++j)
    {
      c[1 + j] = obj->verts.pos[3 * v + j];
      c[4 + j] = lift ? obj->verts.normal[3 * v + j] : 0.0;
    }
  }
}

void MFTLRemeshReferenceAddRecord(MFTL_RemeshReference *ref, const double *rec)
{
  stage_reserve(ref, 3, 1);
  int *e = &ref->st[(size_t) (ref->st_n++) * 4];
  e[0] = (int) rec[1];
  for (int k = 0; k < 3; ++k)
  {
    const double *c = &rec[2 + 7 * k];
    memcpy(&ref->sv[(size_t) (ref->sv_n++) * REF_VREC], c, REF_VREC * sizeof(double));
    e[1 + k] = (int) c[0];
  }
}

/** @brief Corners of reference triangle @p t, consistent with corner 0 across a periodic seam. */
static void ref_corners(const MFTL_RemeshReference *ref, int t, double c[3][3])
{
  const double *f = &ref->frame[9 * t];
  for (int j = 0; j < 3; ++j)
  {
    c[0][j] = f[j];
    c[1][j] = f[j] + f[3 + j];
    c[2][j] = f[j] + f[6 + j];
  }
}

/** @brief The grid cell index of coordinate @p x on axis @p k, wrapped on a periodic axis; -1 outside a bounded grid. */
static inline int cell_of(const MFTL_RemeshReference *ref, int k, double x)
{
  const long long i = (long long) floor((x - ref->origin[k]) / ref->cell[k]);
  if (ref->wrap[k])
  {
    const long long n = ref->ncell[k];
    return (int) (((i % n) + n) % n);
  }
  return (i < 0 || i >= ref->ncell[k]) ? -1 : (int) i;
}

/**
 * @brief The distinct cells a box `[lo, hi]` overlaps.
 * @param      ref  The reference.
 * @param      lo   Lower corner.
 * @param      hi   Upper corner.
 * @param[out] keys Up to 27 keys (at most eight when the cell exceeds the box).
 * @return Number of keys.
 */
static int box_keys(const MFTL_RemeshReference *ref, const double lo[3], const double hi[3], uint64_t keys[27])
{
  int idx[3][3], n[3];
  for (int k = 0; k < 3; ++k)
  {
    n[k] = 0;
    const long long i0 = (long long) floor((lo[k] - ref->origin[k]) / ref->cell[k]);
    const long long i1 = (long long) floor((hi[k] - ref->origin[k]) / ref->cell[k]);
    for (long long i = i0; i <= i1 && n[k] < 3; ++i)
    {
      const int c = cell_of(ref, k, ref->origin[k] + ((double) i + 0.5) * ref->cell[k]);
      if (c < 0) continue;
      bool seen = MFTL_FALSE;
      for (int j = 0; j < n[k]; ++j) seen |= (idx[k][j] == c);
      if (!seen) idx[k][n[k]++] = c;
    }
  }
  int m = 0;
  for (int a = 0; a < n[0]; ++a)
    for (int b = 0; b < n[1]; ++b)
      for (int c = 0; c < n[2]; ++c)
        keys[m++] = ((uint64_t) idx[0][a] * (uint64_t) ref->ncell[1] + (uint64_t) idx[1][b]) * (uint64_t) ref->ncell[2] + (uint64_t) idx[2][c];
  return m;
}

/** @brief Ascending order of staged vertex records by id, for the de-duplication. */
static void finish_vertices(MFTL_RemeshReference *ref)
{
  const int ns = ref->sv_n;
  int *ids = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(2 * ns, 1) * sizeof(int), "remesh reference ids");
  int *slot = ids + ns;
  for (int i = 0; i < ns; ++i) ids[i] = (int) ref->sv[(size_t) i * REF_VREC], slot[i] = i;
  MFTLSortPairs32(ids, slot, ns);

  int nu = 0;
  for (int i = 0; i < ns; ++i)
    if (i == 0 || ids[i] != ids[i - 1]) nu++;

  ref->n_v = nu;
  ref->pos = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nu, 1) * 3 * sizeof(double), "remesh reference positions");
  ref->nrm = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nu, 1) * 3 * sizeof(double), "remesh reference normals");
  ref->vgid = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nu, 1) * sizeof(int), "remesh reference vertex ids");
  int u = -1;
  for (int i = 0; i < ns; ++i)
  {
    if (i > 0 && ids[i] == ids[i - 1]) continue;  // copies of one vertex carry the same bits
    const double *e = &ref->sv[(size_t) slot[i] * REF_VREC];
    ++u;
    ref->vgid[u] = ids[i];
    for (int k = 0; k < 3; ++k)
    {
      ref->pos[3 * u + k] = e[1 + k];
      ref->nrm[3 * u + k] = e[4 + k];
    }
  }
  free(ids);
}

static void build_index(MFTL_RemeshReference *ref, double radius);

/**
 * @brief The largest float not above @p x <= 0: one ulp outward where the
 *        conversion rounded inward (nextafterf() as integer steps, inline --
 *        the library call was half the cost of laying out the grid).
 * @param x A box offset, not positive.
 * @return The float.
 */
static inline float float_below(double x)
{
  float f = (float) x;
  if ((double) f > x)
  {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    b = (f == 0.0f) ? 0x80000001u : b + 1u;  // a negative float grows in magnitude with its bits
    memcpy(&f, &b, sizeof f);
  }
  return f;
}

/**
 * @brief The smallest float not below @p x >= 0 (see float_below()).
 * @param x A box offset, not negative.
 * @return The float.
 */
static inline float float_above(double x)
{
  float f = (float) x;
  if ((double) f < x)
  {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    b = (f == 0.0f) ? 0x00000001u : b + 1u;
    memcpy(&f, &b, sizeof f);
  }
  return f;
}

void MFTLRemeshReferenceFinish(MFTL_RemeshReference *ref, double radius)
{
  MFTL_ASSERT(ref != MFTL_NULL && ref->domain != MFTL_NULL);
  finish_vertices(ref);

  /* Triangles, by id: a triangle can arrive only from its one owner, but the
   * sort makes the index order, and with it every tie, independent of arrival. */
  const int nt0 = ref->st_n;
  int *tid = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(2 * nt0, 1) * sizeof(int), "remesh reference triangle order");
  int *tslot = tid + nt0;
  for (int i = 0; i < nt0; ++i) tid[i] = ref->st[4 * i], tslot[i] = i;
  MFTLSortPairs32(tid, tslot, nt0);

  ref->tv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt0, 1) * 3 * sizeof(int), "remesh reference corners");
  ref->tgid = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt0, 1) * sizeof(int), "remesh reference triangle ids");
  int nt = 0;
  for (int i = 0; i < nt0; ++i)
  {
    if (i > 0 && tid[i] == tid[i - 1]) continue;
    const int *e = &ref->st[4 * tslot[i]];
    bool ok = MFTL_TRUE;
    int c[3];
    for (int k = 0; k < 3; ++k)
    {
      c[k] = MFTLSortedIndexOf(ref->vgid, ref->n_v, e[1 + k]);
      ok &= (c[k] >= 0);
    }
    MFTL_ASSERT(ok && "MFTLRemeshReferenceFinish: a staged triangle names a vertex that was never staged");
    if (!ok) continue;
    ref->tgid[nt] = e[0];
    for (int k = 0; k < 3; ++k) ref->tv[3 * nt + k] = c[k];
    nt++;
  }
  ref->n_t = nt;
  free(tid);
  free(ref->sv);
  free(ref->st);
  ref->sv = MFTL_NULL;
  ref->st = MFTL_NULL;
  ref->sv_n = ref->sv_cap = ref->st_n = ref->st_cap = 0;
  build_index(ref, radius);
}

/**
 * @brief The frames and the grid of a reference whose vertices and triangles
 *        are in place. The index order of either reaches no answer: ties are
 *        broken by triangle id, and the rejections do not depend on the order.
 * @param ref    The reference.
 * @param radius The cap, MFTLRemeshProjectionRadius().
 */
static void build_index(MFTL_RemeshReference *ref, double radius)
{
  const int nt = ref->n_t;

  /* One frame per triangle, so a query takes one minimum image instead of
   * three and every corner is in the same image as corner 0. */
  ref->frame = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * 9 * sizeof(double), "remesh reference frames");
  for (int t = 0; t < nt; ++t)
  {
    double *f = &ref->frame[9 * t];
    const double *p0 = &ref->pos[3 * ref->tv[3 * t]];
    for (int j = 0; j < 3; ++j) f[j] = p0[j];
    MFTLGetMinimumImageVector(ref->domain, &f[3], p0, &ref->pos[3 * ref->tv[3 * t + 1]]);
    MFTLGetMinimumImageVector(ref->domain, &f[6], p0, &ref->pos[3 * ref->tv[3 * t + 2]]);
  }

  /* ── The grid ──────────────────────────────────────────────────────────── */
  const MFTL_Domain *const dom = ref->domain;
  double widest = 0.0, lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX}, hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int t = 0; t < nt; ++t)
  {
    double c[3][3];
    ref_corners(ref, t, c);
    for (int k = 0; k < 3; ++k)
    {
      const double a = MFTL_MIN(c[0][k], MFTL_MIN(c[1][k], c[2][k])), b = MFTL_MAX(c[0][k], MFTL_MAX(c[1][k], c[2][k]));
      widest = MFTL_MAX(widest, b - a);
      lo[k] = MFTL_MIN(lo[k], a);
      hi[k] = MFTL_MAX(hi[k], b);
    }
  }
  const double cell = MFTL_MAX(widest, 2.0 * radius);
  for (int k = 0; k < 3; ++k)
  {
    if (nt == 0 || !(cell > 0.0))
    {
      ref->wrap[k] = MFTL_FALSE;
      ref->origin[k] = 0.0;
      ref->cell[k] = 1.0;
      ref->ncell[k] = 1;
      continue;
    }
    if (dom->is_periodic[k] && dom->domain_size[k] > 0.0)
    {
      const double L = dom->domain_size[k];
      const int n = MFTL_MAX(1, (int) floor(L / cell));
      ref->wrap[k] = MFTL_TRUE;
      ref->origin[k] = dom->global_min[k];
      ref->cell[k] = L / (double) n;
      ref->ncell[k] = n;
    }
    else
    {
      ref->wrap[k] = MFTL_FALSE;
      ref->origin[k] = lo[k] - cell;
      ref->cell[k] = cell;
      ref->ncell[k] = (int) MFTL_MIN(ceil((hi[k] - lo[k]) / cell) + 3.0, 2097151.0);
    }
  }

  /* Up to eight cells per triangle. Collect (cell, triangle) pairs, sort them
   * by cell (stably, so a cell lists its triangles in index order), then lay
   * the entries out cell by cell with what the box test reads beside them. */
  int cap = 8 * MFTL_MAX(nt, 1), n_ent = 0;
  uint64_t *pk = (uint64_t *) MFTLMallocOrDie((size_t) cap * sizeof(uint64_t), "remesh reference keys");
  int *pt = (int *) MFTLMallocOrDie((size_t) cap * sizeof(int), "remesh reference entries");
  for (int t = 0; t < nt; ++t)
  {
    double c[3][3], a[3], b[3];
    ref_corners(ref, t, c);
    for (int k = 0; k < 3; ++k)
    {
      a[k] = MFTL_MIN(c[0][k], MFTL_MIN(c[1][k], c[2][k]));
      b[k] = MFTL_MAX(c[0][k], MFTL_MAX(c[1][k], c[2][k]));
    }
    uint64_t keys[27];
    const int m = box_keys(ref, a, b, keys);
    if (n_ent + m > cap)
    {
      cap = 2 * cap + m;
      pk = (uint64_t *) MFTLReallocOrDie(pk, (size_t) cap * sizeof(uint64_t), "remesh reference keys");
      pt = (int *) MFTLReallocOrDie(pt, (size_t) cap * sizeof(int), "remesh reference entries");
    }
    for (int i = 0; i < m; ++i)
    {
      pk[n_ent] = keys[i];
      pt[n_ent] = t;
      n_ent++;
    }
  }
  MFTLSortPairs64(pk, pt, n_ent);

  int n_cells = 0;
  for (int i = 0; i < n_ent; ++i) n_cells += (i == 0 || pk[i] != pk[i - 1]);
  ref->cell_key = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_cells, 1) * sizeof(uint64_t), "remesh reference cells");
  ref->cell_start = (int *) MFTLMallocOrDie((size_t) (n_cells + 1) * sizeof(int), "remesh reference cells");
  ref->ent_tri = pt;
  ref->ent_c0 = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_ent, 1) * 3 * sizeof(double), "remesh reference entries");
  ref->ent_box = (float *) MFTLMallocOrDie((size_t) MFTL_MAX(n_ent, 1) * 6 * sizeof(float), "remesh reference entries");
  int u = 0;
  for (int i = 0; i < n_ent; ++i)
  {
    if (i == 0 || pk[i] != pk[i - 1])
    {
      ref->cell_key[u] = pk[i];
      ref->cell_start[u] = i;
      u++;
    }
    const double *f = &ref->frame[9 * pt[i]];
    for (int k = 0; k < 3; ++k)
    {
      ref->ent_c0[3 * i + k] = f[k];
      /* Outward, so the float box contains the exact one. */
      const double lo = MFTL_MIN(0.0, MFTL_MIN(f[3 + k], f[6 + k])), hi = MFTL_MAX(0.0, MFTL_MAX(f[3 + k], f[6 + k]));
      ref->ent_box[6 * i + k] = float_below(lo);
      ref->ent_box[6 * i + 3 + k] = float_above(hi);
    }
  }
  ref->cell_start[n_cells] = n_ent;
  free(pk);
  ref->n_cells = n_cells;
  ref->n_ent = n_ent;
  ref->bytes = (double) ref->n_v * (6 * sizeof(double) + sizeof(int)) + (double) ref->n_t * (4 * sizeof(int) + 9 * sizeof(double)) +
               (double) n_cells * (sizeof(uint64_t) + sizeof(int)) + (double) n_ent * (sizeof(int) + 3 * sizeof(double) + 6 * sizeof(float));
}

void MFTLRemeshReferenceFree(MFTL_RemeshReference *ref)
{
  if (ref == MFTL_NULL) return;
  free(ref->pos);
  free(ref->nrm);
  free(ref->vgid);
  free(ref->tv);
  free(ref->tgid);
  free(ref->frame);
  free(ref->sv);
  free(ref->st);
  free(ref->cell_key);
  free(ref->cell_start);
  free(ref->ent_tri);
  free(ref->ent_c0);
  free(ref->ent_box);
  const MFTL_Domain *d = ref->domain;
  memset(ref, 0, sizeof(*ref));
  ref->domain = d;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. One projection
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The occupied cell with key @p key, or -1. */
static inline int find_cell(const MFTL_RemeshReference *ref, uint64_t key)
{
  int lo = 0, hi = ref->n_cells;
  while (lo < hi)
  {
    const int mid = lo + (hi - lo) / 2;
    if (ref->cell_key[mid] < key)
      lo = mid + 1;
    else
      hi = mid;
  }
  return (lo < ref->n_cells && ref->cell_key[lo] == key) ? lo : -1;
}

/** @brief The best candidate of one query so far, and the triangles it has tested. */
typedef struct
{
  int best_t;  ///< -1 before the first candidate.
  double best_d;  ///< Its squared distance; the radius squared before.
  double best_rel[3][3];  ///< Its corners relative to the query point.
  double best_foot[3];  ///< Its closest point, relative to the query point.
  int seen[64];  ///< Triangles evaluated: a triangle filed under several visited cells is met several times.
  int n_seen;
} query_state;

/**
 * @brief Tests the entries of one cell against a query.
 *
 * The rejections, cheapest first, each discard only a triangle no point of
 * which is within @p reach: the box (from the entry alone, so a cell's tests
 * stream), the orientation, the plane. Nothing is tested against the best
 * distance so far, so the triangles that reach the closest point are decided by
 * @p reach alone, whatever the visiting order.
 *
 * @param      ref   The reference.
 * @param      cell  An occupied cell.
 * @param      p     The query point.
 * @param      n_v   The vertex normal; a face pointing against it is never the answer.
 * @param      reach The bridge distance, widened past its rounding.
 * @param[in,out] q  The query's state.
 */
static void scan_cell(const MFTL_RemeshReference *ref, int cell, const double p[3], const double n_v[3], double reach, query_state *q)
{
  const double origin[3] = {0.0, 0.0, 0.0};
  for (int j = ref->cell_start[cell]; j < ref->cell_start[cell + 1]; ++j)
  {
    double rel[3][3];
    MFTLGetMinimumImageVector(ref->domain, rel[0], p, &ref->ent_c0[3 * j]);
    const float *bx = &ref->ent_box[6 * j];
    bool outside = MFTL_FALSE;
    for (int c = 0; c < 3; ++c) outside |= (rel[0][c] + (double) bx[c] > reach) | (rel[0][c] + (double) bx[3 + c] < -reach);
    if (outside) continue;

    /* Once per triangle: a repeat cannot change a minimum with a total
     * tie-break, only its cost (and the list simply stops dropping if full). */
    const int t = ref->ent_tri[j];
    bool repeat = MFTL_FALSE;
    for (int s = 0; s < q->n_seen; ++s) repeat |= (q->seen[s] == t);
    if (repeat) continue;
    if (q->n_seen < (int) (sizeof q->seen / sizeof q->seen[0])) q->seen[q->n_seen++] = t;

    /* A face whose normal points against the vertex's is the other side of a
     * sheet, or a fold: never the surface this vertex belongs to. */
    const double *f = &ref->frame[9 * t], *e1 = &f[3], *e2 = &f[6];
    const double fn[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    if (!(fn[0] * n_v[0] + fn[1] * n_v[1] + fn[2] * n_v[2] > 0.0)) continue;

    /* The plane: no point of the triangle is closer than it. */
    const double h = fn[0] * rel[0][0] + fn[1] * rel[0][1] + fn[2] * rel[0][2];
    if (h * h > reach * reach * (fn[0] * fn[0] + fn[1] * fn[1] + fn[2] * fn[2])) continue;

    for (int c = 0; c < 3; ++c)
    {
      rel[1][c] = rel[0][c] + e1[c];
      rel[2][c] = rel[0][c] + e2[c];
    }
    double foot[3];
    MFTL_TriangleFeature feature;
    const double d = MFTLClosestPointOnTriangle(origin, rel, foot, &feature);
    if (d < q->best_d || (d == q->best_d && (q->best_t < 0 || ref->tgid[t] < ref->tgid[q->best_t])))
    {
      q->best_t = t;
      q->best_d = d;
      memcpy(q->best_rel, rel, sizeof rel);
      memcpy(q->best_foot, foot, sizeof foot);
    }
  }
}

int MFTLRemeshReferenceProjectPoint(const MFTL_RemeshReference *ref, const double p[3], const double n_v[3], double radius, double out[3])
{
  if (ref->n_t == 0 || !(radius > 0.0)) return MFTL_ERROR;

  query_state q;
  q.best_t = -1;
  q.best_d = radius * radius;
  q.n_seen = 0;
  const double reach = radius * (1.0 + 1e-12);

  /* Two passes, and the answer does not depend on the first. (1) The cell
   * holding the point, at the full radius. (2) Every other cell the ball's box
   * overlaps, at the distance (1) found plus a margin far above rounding: the
   * winner is never farther than any compatible candidate, and neither is a
   * triangle tied with it, so both are still tested. A smoothed vertex is
   * usually much nearer the input surface than the radius, so (2) mostly
   * finds no other cell to visit. */
  uint64_t home = UINT64_MAX;
  {
    int idx[3];
    bool inside = MFTL_TRUE;
    for (int k = 0; k < 3; ++k)
    {
      idx[k] = cell_of(ref, k, p[k]);
      inside &= (idx[k] >= 0);
    }
    if (inside)
    {
      home = ((uint64_t) idx[0] * (uint64_t) ref->ncell[1] + (uint64_t) idx[1]) * (uint64_t) ref->ncell[2] + (uint64_t) idx[2];
      const int cell = find_cell(ref, home);
      if (cell >= 0) scan_cell(ref, cell, p, n_v, reach, &q);
    }
  }
  const double reach2 = (q.best_t >= 0) ? MFTL_MIN(reach, sqrt(q.best_d) * (1.0 + 1e-9) + 1e-3 * radius) : reach;
  const double lo[3] = {p[0] - reach2, p[1] - reach2, p[2] - reach2};
  const double hi[3] = {p[0] + reach2, p[1] + reach2, p[2] + reach2};
  uint64_t keys[27];
  const int m = box_keys(ref, lo, hi, keys);
  for (int i = 0; i < m; ++i)
  {
    if (keys[i] == home) continue;
    const int cell = find_cell(ref, keys[i]);
    if (cell >= 0) scan_cell(ref, cell, p, n_v, reach2, &q);
  }
  if (q.best_t < 0) return MFTL_ERROR;
  const int best_t = q.best_t;
  const double(*best_rel)[3] = q.best_rel;
  const double *best_foot = q.best_foot;

  /* ── The Phong lift over the closest point ──────────────────────────── */
  const double *e1 = &ref->frame[9 * best_t + 3], *e2 = &ref->frame[9 * best_t + 6];
  double r[3];
  for (int c = 0; c < 3; ++c) r[c] = best_foot[c] - best_rel[0][c];
  const double d11 = MFTLVectorInProd(e1, e1), d12 = MFTLVectorInProd(e1, e2), d22 = MFTLVectorInProd(e2, e2);
  const double r1 = MFTLVectorInProd(r, e1), r2 = MFTLVectorInProd(r, e2);
  const double den = d11 * d22 - d12 * d12;
  if (!(den > 0.0)) return MFTL_ERROR;

  const double u1 = (d22 * r1 - d12 * r2) / den;
  const double u2 = (d11 * r2 - d12 * r1) / den;
  const double u[3] = {1.0 - u1 - u2, u1, u2};

  double f[3] = {best_foot[0], best_foot[1], best_foot[2]};
  for (int i = 0; i < 3; ++i)
  {
    const double *n = &ref->nrm[3 * ref->tv[3 * best_t + i]];
    const double hgt = (best_foot[0] - best_rel[i][0]) * n[0] + (best_foot[1] - best_rel[i][1]) * n[1] + (best_foot[2] - best_rel[i][2]) * n[2];
    for (int c = 0; c < 3; ++c) f[c] -= 0.5 * u[i] * hgt * n[c];
  }

  for (int c = 0; c < 3; ++c) out[c] = p[c] + f[c];
  MFTLApplyPeriodicity(ref->domain, out);
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Projecting a mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Does @p obj carry a conservative triangle density, whose amounts a moved corner must keep? */
static bool has_density(const MFTL_Object *obj)
{
  if (!obj->fields) return MFTL_FALSE;
  for (int f = 0; f < obj->fields->n; ++f)
    if (obj->fields->entries[f].entity == MFTL_FIELD_TRIANGLE && obj->fields->entries[f].mode == MFTL_FIELD_CONSERVATIVE_DENSITY) return MFTL_TRUE;
  return MFTL_FALSE;
}

/** @brief Area of triangle @p t from seam-consistent corners. */
static double triangle_area(const MFTL_Object *obj, int t)
{
  double T[3][3];
  MFTLGetTriangleCorners(obj, t, T);
  const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
  const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
  const double c[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  return 0.5 * sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
}

double *MFTLRemeshSnapshotDensityAreas(const MFTL_Object *obj)
{
  if (!has_density(obj)) return MFTL_NULL;
  double *a = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(obj->tris.n, 1) * sizeof(double), "remesh projection areas");
  for (int t = 0; t < obj->tris.n; ++t) a[t] = triangle_area(obj, t);
  return a;
}

void MFTLRemeshRescaleDensities(MFTL_Object *obj, const double *area_before)
{
  if (area_before == MFTL_NULL) return;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    const double a1 = triangle_area(obj, t);
    if (a1 == area_before[t] || !(a1 > 0.0)) continue;
    const double factor = area_before[t] / a1;
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *const d = &obj->fields->entries[f];
      if (d->entity != MFTL_FIELD_TRIANGLE || d->mode != MFTL_FIELD_CONSERVATIVE_DENSITY) continue;
      double *const row = MFTLObjectField(obj, f) + (size_t) t * d->ncomp;
      for (int c = 0; c < d->ncomp; ++c) row[c] *= factor;
    }
  }
}

void MFTLRemeshProjectOwned(MFTL_Object *obj, const MFTL_RemeshReference *ref, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st)
{
  const int nvl = obj->verts.n_local;
  if (nvl == 0 || obj->tris.n == 0) return;
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  const double t0 = MFTLRemeshWallTime();
  const double cap = MFTLRemeshProjectionRadius(params);

  /* The vertex's own orientation, for the compatibility test: exact on owned
   * vertices because the halo is vertex-complete. */
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);

  /* Jacobi: every target from the positions the smoother left, then all at
   * once, so the order of the loop cannot reach the answer. */
  double *const out = (double *) MFTLMallocOrDie((size_t) nvl * 3 * sizeof(double), "remesh projection targets");
  memcpy(out, obj->verts.pos, (size_t) nvl * 3 * sizeof(double));
  int n_ok = 0, n_missed = 0;

  /* The shortest incident edge of every owned vertex, in one pass over the
   * half-edges (the halo is vertex-complete, so every incident edge is here):
   * a fan walk per vertex costs twice as much and gives the same bits. */
  double *const min_sq = (double *) MFTLMallocOrDie((size_t) nvl * sizeof(double), "remesh projection edges");
  for (int v = 0; v < nvl; ++v) min_sq[v] = MFTL_DBL_MAX;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    if (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) continue;
    for (int k = 0; k < 3; ++k)
    {
      const int a = obj->tris.he[3 * t + k].vertex_idx, b = obj->tris.he[3 * t + (k + 1) % 3].vertex_idx;
      if (a >= nvl && b >= nvl) continue;
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, &obj->verts.pos[3 * a], &obj->verts.pos[3 * b]);
      const double sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      if (a < nvl) min_sq[a] = MFTL_MIN(min_sq[a], sq);
      if (b < nvl) min_sq[b] = MFTL_MIN(min_sq[b], sq);
    }
  }

  for (int v = 0; v < nvl; ++v)
  {
    if (!corner_lifts(obj, v)) continue;  // free vertices only: a ridge or rim vertex keeps its curve
    if (min_sq[v] == MFTL_DBL_MAX) continue;  // no incident edge
    const double *nv = &obj->verts.normal[3 * v];
    if (!(nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2] > 0.5)) continue;  // degenerate fan: no orientation to compare

    const double radius = MFTL_MIN(0.5 * sqrt(min_sq[v]), cap);
    if (MFTLRemeshReferenceProjectPoint(ref, &obj->verts.pos[3 * v], nv, radius, &out[3 * v]) == MFTL_SUCCESS)
      n_ok++;
    else
      n_missed++;
  }

  memcpy(obj->verts.pos, out, (size_t) nvl * 3 * sizeof(double));
  free(out);
  free(min_sq);
  obj->derived &= ~MFTL_DERIVED_GEOMETRY;
  MFTLObjectMarkChanged(obj);

  if (st)
  {
    st->n_projected += n_ok;
    st->n_projection_missed += n_missed;
    st->t_projection += MFTLRemeshWallTime() - t0;
  }
}

void MFTLRemeshReferenceFromObject(MFTL_RemeshReference *ref, MFTL_Object *obj, const MFTL_RemeshParameters *params)
{
  MFTLRemeshReferenceInit(ref, obj->domain);
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
  /* Angle weights. Max's weights are exact on a sphere and make the lift
   * fourth-order on an IRREGULAR sphere mesh, where angle weights leave it
   * second-order (test_projected_smoothing_stays_on_the_sphere measures both
   * orders' constants), but over the LeVeque period they were worse: d_rms
   * 0.194 against 0.136 % R0 and E_H 0.29 against 0.15 at refinement 4,
   * equal at 5. They weight short edges heavily, which a graded,
   * stretched mesh has plenty of (docs/analysis.md §14.10). */
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE);

  /* One object needs no staging: its own index space, without the
   * de-duplication and the two sorts a reference assembled from several
   * ranks' records takes. */
  const int nv = obj->verts.n, nt = obj->tris.n;
  ref->n_v = nv;
  ref->pos = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nv, 1) * 3 * sizeof(double), "remesh reference positions");
  ref->nrm = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nv, 1) * 3 * sizeof(double), "remesh reference normals");
  memcpy(ref->pos, obj->verts.pos, (size_t) nv * 3 * sizeof(double));
  for (int v = 0; v < nv; ++v)
  {
    const bool lift = corner_lifts(obj, v);
    for (int k = 0; k < 3; ++k) ref->nrm[3 * v + k] = lift ? obj->verts.normal[3 * v + k] : 0.0;
  }
  ref->tv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * 3 * sizeof(int), "remesh reference corners");
  ref->tgid = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(int), "remesh reference triangle ids");
  int m = 0;
  for (int t = 0; t < nt; ++t)
  {
    if (obj->tris.rec[t].flags & MFTL_FLAG_DEAD) continue;
    for (int k = 0; k < 3; ++k) ref->tv[3 * m + k] = obj->tris.verts[3 * t + k];
    ref->tgid[m] = obj->tris.rec[t].global_id;
    m++;
  }
  ref->n_t = m;
  build_index(ref, MFTLRemeshProjectionRadius(params));
}

void MFTLRemeshProjectObject(MFTL_Object *obj, const MFTL_RemeshReference *ref, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st)
{
  double *const area0 = MFTLRemeshSnapshotDensityAreas(obj);
  MFTLRemeshProjectOwned(obj, ref, params, st);
  MFTLRemeshRescaleDensities(obj, area0);
  free(area0);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The distributed reference
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The reference of body @p body in @p set, or MFTL_NULL. */
static MFTL_RemeshReference *set_find(const MFTL_RemeshReferenceSet *set, int body)
{
  int lo = 0, hi = set->n - 1;
  while (lo <= hi)
  {
    const int mid = (lo + hi) / 2;
    if (set->body[mid] == body) return &set->ref[mid];
    if (set->body[mid] < body)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return MFTL_NULL;
}

/** @brief The reference of body @p body, created (empty) if absent; keeps `body` ascending. */
static MFTL_RemeshReference *set_get(MFTL_RemeshReferenceSet *set, int body, const MFTL_Domain *domain)
{
  MFTL_RemeshReference *r = set_find(set, body);
  if (r) return r;
  if (set->n == set->cap)
  {
    set->cap = MFTL_MAX(4, 2 * set->cap);
    set->body = (int *) MFTLReallocOrDie(set->body, (size_t) set->cap * sizeof(int), "remesh reference set");
    set->ref = (MFTL_RemeshReference *) MFTLReallocOrDie(set->ref, (size_t) set->cap * sizeof(MFTL_RemeshReference), "remesh reference set");
  }
  int i = set->n;
  while (i > 0 && set->body[i - 1] > body)
  {
    set->body[i] = set->body[i - 1];
    set->ref[i] = set->ref[i - 1];
    --i;
  }
  set->body[i] = body;
  MFTLRemeshReferenceInit(&set->ref[i], domain);
  set->n++;
  return &set->ref[i];
}

const MFTL_RemeshReference *MFTLRemeshReferenceSetFind(const MFTL_RemeshReferenceSet *set, int body) { return set_find(set, body); }

int MFTLRemeshEnsembleReferenceBuild(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshReferenceSet *set, MFTL_RemeshStats *st)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL && set != MFTL_NULL);
  memset(set, 0, sizeof(*set));
  const double t0 = MFTLRemeshWallTime();
  int status = MFTL_SUCCESS;

  /* Which corners may lift is the OWNER's classification: a ghost's fan is
   * partial, so its local one can be wrong. */
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, MFTL_NULL);
  }
  if (MFTLRemeshSyncGhostClass(ens) != MFTL_SUCCESS) status = MFTL_ERROR;

  /* Exact normals on every copy, ghosts included: a halo triangle's corners are
   * often this rank's ghosts or the sender's. */
  if (MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_ANGLE) != MFTL_SUCCESS) status = MFTL_ERROR;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    MFTL_RemeshReference *r = set_get(set, obj->global_id, &ens->domain);
    MFTLRemeshReferenceAddObject(r, obj, MFTL_TRUE);
  }

  /* Every input triangle a neighbour owns within the cap of this rank's cell. */
  int n_in = 0;
  double *in = MFTL_NULL;
  if (MFTLRemeshReferenceHalo(ens, MFTLRemeshProjectionRadius(params), &in, &n_in) != MFTL_SUCCESS) status = MFTL_ERROR;
  for (int i = 0; i < n_in; ++i)
  {
    const double *rec = &in[(size_t) i * MFTL_REMESH_REF_HALO_WIDTH];
    MFTLRemeshReferenceAddRecord(set_get(set, (int) rec[0], &ens->domain), rec);
  }
  free(in);

  double bytes = 0.0;
  for (int i = 0; i < set->n; ++i)
  {
    MFTLRemeshReferenceFinish(&set->ref[i], MFTLRemeshProjectionRadius(params));
    bytes += set->ref[i].bytes;
  }
  if (st)
  {
    st->t_reference += MFTLRemeshWallTime() - t0;
    st->reference_bytes = MFTL_MAX(st->reference_bytes, bytes);
  }
  return status;
}

int MFTLRemeshEnsembleProject(MFTL_Ensemble *ens, const MFTL_RemeshReferenceSet *set, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st)
{
  MFTL_ASSERT(ens != MFTL_NULL && set != MFTL_NULL);
  const int b = ens->n_local_objects;
  double **area0 = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(double *), "remesh projection area snapshots");

  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    area0[o] = MFTLRemeshSnapshotDensityAreas(obj);
    const MFTL_RemeshReference *ref = set_find(set, obj->global_id);
    if (ref == MFTL_NULL)
    {
      /* A body this rank neither held nor bordered when the call started; the
       * reference cannot be here, so its vertices keep the smoother's place. */
      if (st) st->n_projection_missed += obj->verts.n_local;
      continue;
    }
    MFTLRemeshProjectOwned(obj, ref, params, st);
  }

  /* The ghosts learn their owners' new places; every holder then rescales the
   * densities of every triangle it holds from the same bits. */
  const int status = MFTLUpdateGhostPositions(ens);
  for (int o = 0; o < b; ++o)
  {
    MFTLRemeshRescaleDensities(ens->local_objects[o], area0[o]);
    free(area0[o]);
  }
  free(area0);
  return status;
}

void MFTLRemeshReferenceSetFree(MFTL_RemeshReferenceSet *set)
{
  if (set == MFTL_NULL) return;
  for (int i = 0; i < set->n; ++i) MFTLRemeshReferenceFree(&set->ref[i]);
  free(set->body);
  free(set->ref);
  memset(set, 0, sizeof(*set));
}
