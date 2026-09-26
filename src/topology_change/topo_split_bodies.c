/**
 * @file topo_split_bodies.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief From triangle soup to bodies: distributed component labelling, the id
 *        decision, and the objects themselves.
 *
 * ## LABELS ARE THE SMALLEST TRIANGLE KEY IN THE COMPONENT
 *
 * A marching cubes triangle has a lattice identity, `16 * cell + slot`, that
 * every rank holding it computes identically. A component's label is the
 * smallest such key it contains -- a pure function of the component, so every
 * holder converges on the same number without anybody electing anything.
 *
 * Each round does three things: every LOCAL component takes the smallest label
 * of any triangle in it (MFTLLabelConnectedComponents did the local
 * connectivity); every holder of a GHOST triangle tells its owner the label it
 * has; the owner lowers its component to the smallest it hears and answers with
 * the result. A component spanning k rank boundaries needs at most k + 1 rounds,
 * and the loop ends on a global "nothing changed". No message is sized by the
 * body count or the rank count -- only by the ghost triangles.
 *
 * A body can pass through one rank's box twice with the connection elsewhere --
 * a U whose bend is on the next rank. Locally that is two components; globally
 * one. Objects are therefore built per GLOBAL label, not per local component.
 *
 * ## IDS: PROVISIONAL, THEN DENSE
 *
 * A body's vertex and triangle ids must be dense over all its holders. An owner
 * numbers what it owns from a base MFTLEnsembleExscanBodies() gives it -- the
 * library's own per-body scan over each body's holders -- and a ghost asks its
 * owner for the id of the entity with its lattice key. The closing
 * redistribution then rebuilds the canonical halo from correct ids.
 */

#include "topo_internal.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Keys and walls
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Which domain walls a vertex lies on, as bits `2 * axis + side`.
 *
 * Read off its LATTICE KEY, so a ghost copy knows it exactly as its owner does:
 * a vertex on a lattice edge lying in a wall plane is where the surface crosses
 * the wall, i.e. a rim vertex. Centroid vertices are never on a wall.
 */
static int wall_mask(const MFTL_TopoLattice *lat, long long vkey)
{
  if (vkey < 0) return 0;
  const int axis = (int) (vkey % 3);
  const long long node = vkey / 3;
  const long long nx = lat->n_global[0], ny = lat->n_global[1];
  const int g[3] = {(int) (node % nx), (int) ((node / nx) % ny), (int) (node / (nx * ny))};

  int m = 0;
  for (int k = 0; k < 3; ++k)
  {
    if (k == axis || lat->periodic[k]) continue;
    if (g[k] == 0) m |= 1 << (2 * k);
    if (g[k] == lat->cells[k]) m |= 1 << (2 * k + 1);
  }
  return m;
}

static int cmp_pair_ll(const void *a, const void *b)
{
  const long long x = ((const long long *) a)[0], y = ((const long long *) b)[0];
  return (x > y) - (x < y);
}

/** @brief Value paired with @p key in sorted (key, value) pairs, or -1. */
static long long find_pair(const long long *pairs, int n, long long key)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    const int mid = (lo + hi) / 2;
    const long long k = pairs[2 * mid];
    if (k == key) return pairs[2 * mid + 1];
    if (k < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Pieces: one cluster's local soup as a scratch mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopoBuildPieces(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, MFTL_TopoPiece *pieces)
{
  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_TopoPiece *const p = &pieces[c];
    memset(p, 0, sizeof(*p));
    p->rank = dom->rank;
    const MFTL_LevelSetSurface *const s = &cl[c].surf;
    const int nv = s->n_verts, nt = s->n_tris;

    p->mesh = MFTLObjectCreate(MFTL_ERROR);
    p->tri_owner = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(int), "topology piece triangle owners");
    p->vert_owner = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nv, 1) * sizeof(int), "topology piece vertex owners");
    p->tkey = (long long *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(long long), "topology piece triangle keys");
    p->vkey = (long long *) MFTLMallocOrDie((size_t) MFTL_MAX(nv, 1) * sizeof(long long), "topology piece vertex keys");
    p->label = (long long *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(long long), "topology piece labels");
    p->comp = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(int), "topology piece components");
    p->owned_key = (long long *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(nt, 1) * sizeof(long long), "topology owned keys");
    if (nt == 0) continue;

    MFTL_Object *const m = p->mesh;
    MFTLObjectSetCounts(m, nv, nv, nt, nt);
    for (int v = 0; v < nv; ++v)
    {
      double *const q = &m->verts.pos[3 * v];
      for (int k = 0; k < 3; ++k) q[k] = s->pos[3 * (size_t) v + k];
      MFTLApplyPeriodicity(dom, q);
      p->vert_owner[v] = MFTLGetOwnerProc(dom, q);
      p->vkey[v] = s->vkey[v];
      m->verts.rec[v].global_id = v;
      m->verts.rec[v].owner_proc = p->vert_owner[v];
      m->verts.rec[v].flags = 0;
      m->verts.rec[v].out_he_idx = MFTL_ERROR;
    }
    for (int t = 0; t < nt; ++t)
    {
      p->tkey[t] = 16 * s->tcell[t] + s->tslot[t];
      p->tri_owner[t] = MFTLTopoCellOwner(dom, lat, s->tcell[t]);
      m->tris.rec[t].global_id = t;
      m->tris.rec[t].owner_proc = p->tri_owner[t];
      m->tris.rec[t].flags = 0;
      for (int k = 0; k < 3; ++k)
      {
        m->tris.he[3 * t + k].vertex_idx = s->tri[3 * (size_t) t + (k + 1) % 3];
        m->tris.he[3 * t + k].global_id = 3 * t + k;
        m->tris.he[3 * t + k].flags = 0;
        m->verts.rec[s->tri[3 * (size_t) t + k]].out_he_idx = 3 * t + k;
      }
    }
    /* Twins by vertex slot: two triangles sharing a lattice edge's vertex pair
     * are adjacent, because marching cubes produced ONE vertex per lattice edge
     * in this extraction. */
    MFTLRebuildTwins(m);

    p->n_comp = MFTLLabelConnectedComponents(m, p->comp);
    for (int t = 0; t < nt; ++t) p->label[t] = p->tkey[t];

    int w = 0;
    for (int t = 0; t < nt; ++t)
      if (p->tri_owner[t] == dom->rank)
      {
        p->owned_key[2 * w] = p->tkey[t];
        p->owned_key[2 * w + 1] = t;
        w++;
      }
    p->n_owned = w;
    qsort(p->owned_key, (size_t) w, 2 * sizeof(long long), cmp_pair_ll);
  }
}

void MFTLTopoFreePieces(MFTL_TopoPiece *pieces, int n)
{
  if (pieces == MFTL_NULL) return;
  for (int c = 0; c < n; ++c)
  {
    MFTLObjectDestroy(&pieces[c].mesh);
    free(pieces[c].tri_owner);
    free(pieces[c].vert_owner);
    free(pieces[c].tkey);
    free(pieces[c].vkey);
    free(pieces[c].label);
    free(pieces[c].comp);
    free(pieces[c].owned_key);
  }
}

static inline int piece_tris(const MFTL_TopoPiece *p) { return (p->mesh != MFTL_NULL) ? p->mesh->tris.n : 0; }

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Labelling
 * ═══════════════════════════════════════════════════════════════════════════ */

bool MFTLTopoLabelLocal(MFTL_TopoPiece *pieces, int n_cl)
{
  bool changed = MFTL_FALSE;
  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_TopoPiece *const p = &pieces[c];
    const int nt = piece_tris(p);
    if (nt == 0) continue;

    long long *const best = (long long *) MFTLMallocOrDie((size_t) p->n_comp * sizeof(long long), "topology component minima");
    for (int i = 0; i < p->n_comp; ++i) best[i] = LLONG_MAX;
    for (int t = 0; t < nt; ++t)
      if (p->label[t] < best[p->comp[t]]) best[p->comp[t]] = p->label[t];
    for (int t = 0; t < nt; ++t)
      if (best[p->comp[t]] < p->label[t])
      {
        p->label[t] = best[p->comp[t]];
        changed = MFTL_TRUE;
      }
    free(best);
  }
  return changed;
}

/**
 * @brief Offsets of each owner's block of ghost triangles, in the order the
 *        requests are packed: (cluster, triangle) ascending within an owner.
 */
static void ghost_triangle_offsets(const MFTL_TopoPiece *pieces, int n_cl, int nproc, int *counts, int *offsets)
{
  for (int r = 0; r < nproc; ++r) counts[r] = 0;
  for (int c = 0; c < n_cl; ++c)
  {
    const MFTL_TopoPiece *const p = &pieces[c];
    for (int t = 0; t < piece_tris(p); ++t)
      if (p->tri_owner[t] != p->rank) counts[p->tri_owner[t]]++;
  }
  int total = 0;
  for (int r = 0; r < nproc; ++r)
  {
    offsets[r] = total;
    total += counts[r];
  }
}

void MFTLTopoPackLabelRequests(const MFTL_TopoPiece *pieces, int n_cl, int nproc, int *dest_counts, double **records)
{
  int *const cursor = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "topology label cursors");
  ghost_triangle_offsets(pieces, n_cl, nproc, dest_counts, cursor);

  int total = 0;
  for (int r = 0; r < nproc; ++r) total += dest_counts[r];
  double *const buf = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "topology label requests");

  for (int c = 0; c < n_cl; ++c)
  {
    const MFTL_TopoPiece *const p = &pieces[c];
    for (int t = 0; t < piece_tris(p); ++t)
    {
      const int r = p->tri_owner[t];
      if (r == p->rank) continue;
      double *const rec = &buf[3 * (size_t) cursor[r]++];
      rec[0] = (double) c;
      rec[1] = (double) p->tkey[t];
      rec[2] = (double) p->label[t];
    }
  }
  free(cursor);
  *records = buf;
}

bool MFTLTopoServeLabelRequests(MFTL_TopoPiece *pieces, int n_cl, const double *req, int n_req, double *reply)
{
  bool changed = MFTL_FALSE;

  /* Component minima, starting from the labels this rank already has. */
  long long **const cmin = (long long **) MFTLCallocOrDie((size_t) MFTL_MAX(n_cl, 1), sizeof(long long *), "topology served minima");
  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_TopoPiece *const p = &pieces[c];
    if (piece_tris(p) == 0) continue;
    cmin[c] = (long long *) MFTLMallocOrDie((size_t) p->n_comp * sizeof(long long), "topology served minima");
    for (int i = 0; i < p->n_comp; ++i) cmin[c][i] = LLONG_MAX;
    for (int t = 0; t < piece_tris(p); ++t)
      if (p->label[t] < cmin[c][p->comp[t]]) cmin[c][p->comp[t]] = p->label[t];
  }

  int *const tri_of = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(int), "topology served triangles");
  for (int i = 0; i < n_req; ++i)
  {
    const int c = (int) req[3 * i];
    const long long key = (long long) req[3 * i + 1];
    const long long lab = (long long) req[3 * i + 2];
    tri_of[i] = -1;
    if (c < 0 || c >= n_cl || cmin[c] == MFTL_NULL) continue;
    const int t = (int) find_pair(pieces[c].owned_key, pieces[c].n_owned, key);
    MFTL_ASSERT(t >= 0 && "topology labelling: asked about a triangle this rank does not own");
    if (t < 0) continue;
    tri_of[i] = t;
    long long *const m = &cmin[c][pieces[c].comp[t]];
    if (lab < *m) *m = lab;
  }

  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_TopoPiece *const p = &pieces[c];
    if (cmin[c] == MFTL_NULL) continue;
    for (int t = 0; t < piece_tris(p); ++t)
      if (cmin[c][p->comp[t]] < p->label[t])
      {
        p->label[t] = cmin[c][p->comp[t]];
        changed = MFTL_TRUE;
      }
  }

  for (int i = 0; i < n_req; ++i)
  {
    const int c = (int) req[3 * i];
    reply[i] = (tri_of[i] >= 0) ? (double) pieces[c].label[tri_of[i]] : req[3 * i + 2];
  }

  for (int c = 0; c < n_cl; ++c) free(cmin[c]);
  free(cmin);
  free(tri_of);
  return changed;
}

bool MFTLTopoApplyLabelReplies(MFTL_TopoPiece *pieces, int n_cl, int nproc, const double *reply)
{
  int *const counts = (int *) MFTLMallocOrDie(2 * (size_t) nproc * sizeof(int), "topology reply cursors");
  int *const cursor = counts + nproc;
  ghost_triangle_offsets(pieces, n_cl, nproc, counts, cursor);

  bool changed = MFTL_FALSE;
  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_TopoPiece *const p = &pieces[c];
    for (int t = 0; t < piece_tris(p); ++t)
    {
      const int r = p->tri_owner[t];
      if (r == p->rank) continue;
      const long long lab = (long long) reply[cursor[r]++];
      if (lab < p->label[t])
      {
        p->label[t] = lab;
        changed = MFTL_TRUE;
      }
    }
  }
  free(counts);
  return changed;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Components: records, the decision, the objects
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Whether half-edge 3t+k of a piece lies on a domain wall. */
static bool piece_edge_on_wall(const MFTL_TopoLattice *lat, const MFTL_TopoPiece *p, int t, int k)
{
  const MFTL_HalfEdge *const he = p->mesh->tris.he;
  const int target = he[3 * t + k].vertex_idx;
  const int origin = he[3 * t + (k + 2) % 3].vertex_idx;
  return (wall_mask(lat, p->vkey[origin]) & wall_mask(lat, p->vkey[target])) != 0;
}

static int cmp_record(const void *a, const void *b)
{
  const double *x = (const double *) a, *y = (const double *) b;
  if (x[0] != y[0]) return (x[0] < y[0]) ? -1 : 1;
  if (x[1] != y[1]) return (x[1] < y[1]) ? -1 : 1;
  return 0;
}

int MFTLTopoComponentRecords(const MFTL_TopoLattice *lat, const MFTL_TopoPiece *pieces, int n_cl, double **records)
{
  /* One record per (cluster, OWNED triangle), then merged by label locally:
   * a label seen through two local components is one component. */
  int total = 0;
  for (int c = 0; c < n_cl; ++c) total += pieces[c].n_owned;
  double *const rec = (double *) MFTLMallocOrDie(4 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "topology component records");

  int n = 0;
  for (int c = 0; c < n_cl; ++c)
  {
    const MFTL_TopoPiece *const p = &pieces[c];
    for (int t = 0; t < piece_tris(p); ++t)
    {
      if (p->tri_owner[t] != p->rank) continue;
      bool open = MFTL_FALSE;
      for (int k = 0; k < 3; ++k)
        if (piece_edge_on_wall(lat, p, t, k)) open = MFTL_TRUE;
      double *const r = &rec[4 * (size_t) n++];
      r[0] = (double) c;
      r[1] = (double) p->label[t];
      r[2] = 1.0;
      r[3] = open ? 1.0 : 0.0;
    }
  }

  qsort(rec, (size_t) n, 4 * sizeof(double), cmp_record);
  int w = 0;
  for (int i = 0; i < n; ++i)
  {
    if (w > 0 && rec[4 * (w - 1)] == rec[4 * i] && rec[4 * (w - 1) + 1] == rec[4 * i + 1])
    {
      rec[4 * (w - 1) + 2] += rec[4 * i + 2];
      rec[4 * (w - 1) + 3] = fmax(rec[4 * (w - 1) + 3], rec[4 * i + 3]);
      continue;
    }
    memmove(&rec[4 * (size_t) w], &rec[4 * (size_t) i], 4 * sizeof(double));
    w++;
  }
  *records = rec;
  return w;
}

/** @brief Global size of body @p gid from the listed sizes. */
static long long body_size_of(const int *bodies, const long long *size, int n, int gid)
{
  for (int i = 0; i < n; ++i)
    if (bodies[i] == gid) return size[i];
  return 0;
}

typedef struct
{
  int gid;
  long long size;
} sized_body;

static int cmp_sized_body(const void *a, const void *b)
{
  const sized_body *x = (const sized_body *) a, *y = (const sized_body *) b;
  if (x->size != y->size) return (x->size > y->size) ? -1 : 1;
  return (x->gid > y->gid) - (x->gid < y->gid);
}

static int cmp_comp_size(const void *a, const void *b)
{
  const MFTL_TopoComponent *x = (const MFTL_TopoComponent *) a, *y = (const MFTL_TopoComponent *) b;
  if (x->cluster != y->cluster) return (x->cluster < y->cluster) ? -1 : 1;
  if (x->n_tris != y->n_tris) return (x->n_tris > y->n_tris) ? -1 : 1;
  return (x->label > y->label) - (x->label < y->label);
}

static int cmp_comp_key(const void *a, const void *b)
{
  const MFTL_TopoComponent *x = (const MFTL_TopoComponent *) a, *y = (const MFTL_TopoComponent *) b;
  if (x->cluster != y->cluster) return (x->cluster < y->cluster) ? -1 : 1;
  return (x->label > y->label) - (x->label < y->label);
}

int MFTLTopoDecideComponents(const double *all, int n_all, const MFTL_TopoCluster *cl, int n_cl, const int *bodies, const long long *body_size, int n_bodies,
                             int min_tris, int *n_bodies_global, MFTL_TopoComponent **comps_out, MFTL_TopologyStats *stats)
{
  /* Merge the holders' records: they arrive in rank order, each rank's sorted. */
  double *const rec = (double *) MFTLMallocOrDie(4 * (size_t) MFTL_MAX(n_all, 1) * sizeof(double), "topology merged records");
  memcpy(rec, all, 4 * (size_t) n_all * sizeof(double));
  qsort(rec, (size_t) n_all, 4 * sizeof(double), cmp_record);

  MFTL_TopoComponent *const comps = (MFTL_TopoComponent *) MFTLMallocOrDie((size_t) MFTL_MAX(n_all, 1) * sizeof(MFTL_TopoComponent), "topology components");
  int n = 0;
  for (int i = 0; i < n_all; ++i)
  {
    const int c = (int) rec[4 * i];
    const long long label = (long long) rec[4 * i + 1];
    if (n > 0 && comps[n - 1].cluster == c && comps[n - 1].label == label)
    {
      comps[n - 1].n_tris += (long long) rec[4 * i + 2];
      comps[n - 1].open = comps[n - 1].open || rec[4 * i + 3] > 0.0;
      continue;
    }
    comps[n].cluster = c;
    comps[n].label = label;
    comps[n].n_tris = (long long) rec[4 * i + 2];
    comps[n].open = rec[4 * i + 3] > 0.0;
    comps[n].body = -1;
    n++;
  }
  free(rec);

  /* Per cluster: components largest first, bodies largest first, pair them up. */
  qsort(comps, (size_t) n, sizeof(MFTL_TopoComponent), cmp_comp_size);
  int first = 0;
  for (int c = 0; c < n_cl; ++c)
  {
    if (cl[c].refused) continue;

    int last = first;
    while (last < n && comps[last].cluster == c) last++;

    sized_body *const sb = (sized_body *) MFTLMallocOrDie((size_t) MFTL_MAX(cl[c].n_bodies, 1) * sizeof(sized_body), "topology sized bodies");
    for (int i = 0; i < cl[c].n_bodies; ++i)
    {
      sb[i].gid = cl[c].bodies[i];
      sb[i].size = body_size_of(bodies, body_size, n_bodies, cl[c].bodies[i]);
    }
    qsort(sb, (size_t) cl[c].n_bodies, sizeof(sized_body), cmp_sized_body);

    int kept = 0;
    for (int i = first; i < last; ++i)
    {
      if (!comps[i].open && comps[i].n_tris < min_tris)
      {
        comps[i].body = -1;
        if (stats) stats->n_bodies_discarded++;
        continue;
      }
      if (kept < cl[c].n_bodies)
        comps[i].body = sb[kept].gid;
      else
      {
        comps[i].body = (*n_bodies_global)++;
        if (stats) stats->n_bodies_created++;
      }
      kept++;
    }
    if (stats)
    {
      stats->n_bodies_reconstructed += cl[c].n_bodies;
      stats->n_bodies_out += kept;
      if (kept < cl[c].n_bodies) stats->n_bodies_retired += cl[c].n_bodies - kept;
    }
    free(sb);
    first = last;
  }

  qsort(comps, (size_t) n, sizeof(MFTL_TopoComponent), cmp_comp_key);
  *comps_out = comps;
  return n;
}

const MFTL_TopoComponent *MFTLTopoFindComponent(const MFTL_TopoComponent *comps, int n, int cluster, long long label)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    const int mid = (lo + hi) / 2;
    const MFTL_TopoComponent *const m = &comps[mid];
    if (m->cluster == cluster && m->label == label) return m;
    if (m->cluster < cluster || (m->cluster == cluster && m->label < label))
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return MFTL_NULL;
}

/**
 * @brief Builds ONE body's local object from the piece triangles carrying
 *        @p label: owned entities first, ghosts after, twins remapped.
 *
 * @return The object, or #MFTL_NULL when this rank owns none of it.
 */
static MFTL_Object *build_one(const MFTL_Ensemble *ens, const MFTL_TopoLattice *lat, const MFTL_TopoPiece *p, long long label, int body, long long **vkeys_out,
                              long long **tkeys_out)
{
  const MFTL_Object *const m = p->mesh;
  const int nt_all = m->tris.n, nv_all = m->verts.n;

  /* Which triangles and vertices, and in which order. */
  int *const tmap = (int *) MFTLMallocOrDie((size_t) nt_all * sizeof(int), "topology triangle map");
  int *const vmap = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nv_all, 1) * sizeof(int), "topology vertex map");
  for (int t = 0; t < nt_all; ++t) tmap[t] = -1;
  for (int v = 0; v < nv_all; ++v) vmap[v] = -1;

  int nt = 0, nt_own = 0, nv = 0, nv_own = 0;
  for (int pass = 0; pass < 2; ++pass)  // owned first, ghosts second
    for (int t = 0; t < nt_all; ++t)
    {
      if (p->label[t] != label) continue;
      if ((p->tri_owner[t] == p->rank) != (pass == 0)) continue;
      tmap[t] = nt++;
      if (pass == 0) nt_own++;
    }
  for (int pass = 0; pass < 2; ++pass)
    for (int t = 0; t < nt_all; ++t)
    {
      if (tmap[t] < 0) continue;
      for (int k = 0; k < 3; ++k)
      {
        const int v = m->tris.he[3 * t + k].vertex_idx;
        if (vmap[v] >= 0) continue;
        if ((p->vert_owner[v] == p->rank) != (pass == 0)) continue;
        vmap[v] = nv++;
        if (pass == 0) nv_own++;
      }
    }

  if (nt_own == 0 && nv_own == 0)
  {
    free(tmap);
    free(vmap);
    return MFTL_NULL;
  }

  MFTL_Object *const obj = MFTLObjectCreate(body);
  obj->domain = (MFTL_Domain *) &ens->domain;
  MFTLObjectSetCounts(obj, nv, nv_own, nt, nt_own);

  long long *const vkeys = (long long *) MFTLMallocOrDie((size_t) MFTL_MAX(nv, 1) * sizeof(long long), "topology object vertex keys");
  long long *const tkeys = (long long *) MFTLMallocOrDie((size_t) MFTL_MAX(nt, 1) * sizeof(long long), "topology object triangle keys");

  for (int v = 0; v < nv_all; ++v)
  {
    const int u = vmap[v];
    if (u < 0) continue;
    for (int k = 0; k < 3; ++k) obj->verts.pos[3 * u + k] = m->verts.pos[3 * v + k];
    obj->verts.rec[u].global_id = u;  // provisional
    obj->verts.rec[u].owner_proc = p->vert_owner[v];
    obj->verts.rec[u].flags = wall_mask(lat, p->vkey[v]) ? MFTL_FLAG_BOUNDARY : 0;
    obj->verts.rec[u].out_he_idx = MFTL_ERROR;
    vkeys[u] = p->vkey[v];
  }

  for (int t = 0; t < nt_all; ++t)
  {
    const int u = tmap[t];
    if (u < 0) continue;
    obj->tris.rec[u].global_id = u;  // provisional
    obj->tris.rec[u].owner_proc = p->tri_owner[t];
    obj->tris.rec[u].flags = 0;
    tkeys[u] = p->tkey[t];
    for (int k = 0; k < 3; ++k)
    {
      MFTL_HalfEdge *const h = &obj->tris.he[3 * u + k];
      const MFTL_HalfEdge *const src = &m->tris.he[3 * t + k];
      h->vertex_idx = vmap[src->vertex_idx];
      h->global_id = 3 * u + k;
      const int tw = src->twin_idx;
      h->twin_idx = (tw != MFTL_ERROR && tmap[MFTLHeTriIdx(tw)] >= 0) ? 3 * tmap[MFTLHeTriIdx(tw)] + MFTLHeCorner(tw) : MFTL_ERROR;
      h->flags = piece_edge_on_wall(lat, p, t, k) ? MFTL_FLAG_BOUNDARY : 0;

      const int origin = vmap[m->tris.he[3 * t + (k + 2) % 3].vertex_idx];
      obj->verts.rec[origin].out_he_idx = 3 * u + k;
    }
  }

  MFTLObjectBuildDerived(obj);
  free(tmap);
  free(vmap);
  *vkeys_out = vkeys;
  *tkeys_out = tkeys;
  return obj;
}

int MFTLTopoBuildObjects(MFTL_Ensemble *ens, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, MFTL_TopoPiece *pieces,
                         const MFTL_TopoComponent *comps, int n_comps, int **created_out, long long ***vkeys_out, long long ***tkeys_out)
{
  /* Retire every listed body of every reconstructed cluster on this rank. */
  for (int c = 0; c < n_cl; ++c)
  {
    if (cl[c].refused) continue;
    for (int i = 0; i < cl[c].n_bodies; ++i)
    {
      const int slot = MFTLEnsembleFindLocalSlot(ens, cl[c].bodies[i]);
      if (slot == MFTL_ERROR) continue;
      MFTL_Object *obj = ens->local_objects[slot];
      MFTLEnsembleEraseLocalSlot(ens, slot);
      MFTLObjectDestroy(&obj);
    }
  }

  /* Distinct (cluster, label) pairs present here, each once. */
  int cap = 8, n_new = 0;
  int *created = (int *) MFTLMallocOrDie((size_t) cap * sizeof(int), "topology created bodies");
  long long **vkeys = (long long **) MFTLMallocOrDie((size_t) cap * sizeof(long long *), "topology created vertex keys");
  long long **tkeys = (long long **) MFTLMallocOrDie((size_t) cap * sizeof(long long *), "topology created triangle keys");
  MFTL_Object **objs = (MFTL_Object **) MFTLMallocOrDie((size_t) cap * sizeof(MFTL_Object *), "topology created objects");

  for (int c = 0; c < n_cl; ++c)
  {
    const MFTL_TopoPiece *const p = &pieces[c];
    const int nt = piece_tris(p);
    if (nt == 0) continue;

    long long *const seen = (long long *) MFTLMallocOrDie((size_t) nt * sizeof(long long), "topology seen labels");
    int n_seen = 0;
    for (int t = 0; t < nt; ++t)
    {
      const long long label = p->label[t];
      bool dup = MFTL_FALSE;
      for (int i = 0; i < n_seen && !dup; ++i) dup = (seen[i] == label);
      if (dup) continue;
      seen[n_seen++] = label;

      const MFTL_TopoComponent *const comp = MFTLTopoFindComponent(comps, n_comps, c, label);
      if (comp == MFTL_NULL || comp->body < 0) continue;

      long long *vk = MFTL_NULL, *tk = MFTL_NULL;
      MFTL_Object *const obj = build_one(ens, lat, p, label, comp->body, &vk, &tk);
      if (obj == MFTL_NULL) continue;

      if (n_new == cap)
      {
        cap *= 2;
        created = (int *) MFTLReallocOrDie(created, (size_t) cap * sizeof(int), "topology created bodies");
        vkeys = (long long **) MFTLReallocOrDie(vkeys, (size_t) cap * sizeof(long long *), "topology created vertex keys");
        tkeys = (long long **) MFTLReallocOrDie(tkeys, (size_t) cap * sizeof(long long *), "topology created triangle keys");
        objs = (MFTL_Object **) MFTLReallocOrDie(objs, (size_t) cap * sizeof(MFTL_Object *), "topology created objects");
      }
      created[n_new] = comp->body;
      vkeys[n_new] = vk;
      tkeys[n_new] = tk;
      objs[n_new] = obj;
      n_new++;
    }
    free(seen);
  }

  /* Ascending body id, so the owner side can binary search it; then into the
   * ensemble, which keeps its own list sorted. */
  for (int i = 1; i < n_new; ++i)
  {
    for (int j = i; j > 0 && created[j - 1] > created[j]; --j)
    {
      int tg = created[j];
      created[j] = created[j - 1];
      created[j - 1] = tg;
      long long *tv = vkeys[j];
      vkeys[j] = vkeys[j - 1];
      vkeys[j - 1] = tv;
      long long *tt = tkeys[j];
      tkeys[j] = tkeys[j - 1];
      tkeys[j - 1] = tt;
      MFTL_Object *to = objs[j];
      objs[j] = objs[j - 1];
      objs[j - 1] = to;
    }
  }
  for (int i = 0; i < n_new; ++i) MFTLEnsembleInsertLocalObject(ens, objs[i]);
  free(objs);

  *created_out = created;
  *vkeys_out = vkeys;
  *tkeys_out = tkeys;
  return n_new;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Ghost ids
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Per-owner offsets of the ghost requests of the created objects, in pack order. */
static void ghost_id_offsets(const MFTL_Ensemble *ens, const int *created, int n_created, int nproc, int *counts, int *offsets)
{
  for (int r = 0; r < nproc; ++r) counts[r] = 0;
  for (int i = 0; i < n_created; ++i)
  {
    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, created[i]);
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v) counts[obj->verts.rec[v].owner_proc]++;
    for (int t = obj->tris.n_local; t < obj->tris.n; ++t) counts[obj->tris.rec[t].owner_proc]++;
  }
  int total = 0;
  for (int r = 0; r < nproc; ++r)
  {
    offsets[r] = total;
    total += counts[r];
  }
}

void MFTLTopoPackIdRequests(const MFTL_Ensemble *ens, const int *created, int n_created, long long **vkeys, long long **tkeys, int nproc, int *dest_counts,
                            double **records)
{
  int *const cursor = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "topology id cursors");
  ghost_id_offsets(ens, created, n_created, nproc, dest_counts, cursor);
  int total = 0;
  for (int r = 0; r < nproc; ++r) total += dest_counts[r];
  double *const buf = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "topology id requests");

  for (int i = 0; i < n_created; ++i)
  {
    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, created[i]);
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      double *const rec = &buf[3 * (size_t) cursor[obj->verts.rec[v].owner_proc]++];
      rec[0] = (double) created[i];
      rec[1] = 0.0;
      rec[2] = (double) vkeys[i][v];
    }
    for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
    {
      double *const rec = &buf[3 * (size_t) cursor[obj->tris.rec[t].owner_proc]++];
      rec[0] = (double) created[i];
      rec[1] = 1.0;
      rec[2] = (double) tkeys[i][t];
    }
  }
  free(cursor);
  *records = buf;
}

void MFTLTopoServeIdRequests(const MFTL_Ensemble *ens, const int *created, int n_created, long long **vkeys, long long **tkeys, const double *req, int n_req,
                             double *reply)
{
  /* Sorted (key, index) of the OWNED entities of every created object. */
  long long **vs = (long long **) MFTLCallocOrDie((size_t) MFTL_MAX(n_created, 1), sizeof(long long *), "topology owned vertex keys");
  long long **ts = (long long **) MFTLCallocOrDie((size_t) MFTL_MAX(n_created, 1), sizeof(long long *), "topology owned triangle keys");
  for (int i = 0; i < n_created; ++i)
  {
    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, created[i]);
    const int nv = obj->verts.n_local, nt = obj->tris.n_local;
    vs[i] = (long long *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(nv, 1) * sizeof(long long), "topology owned vertex keys");
    ts[i] = (long long *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(nt, 1) * sizeof(long long), "topology owned triangle keys");
    for (int v = 0; v < nv; ++v)
    {
      vs[i][2 * v] = vkeys[i][v];
      vs[i][2 * v + 1] = v;
    }
    for (int t = 0; t < nt; ++t)
    {
      ts[i][2 * t] = tkeys[i][t];
      ts[i][2 * t + 1] = t;
    }
    qsort(vs[i], (size_t) nv, 2 * sizeof(long long), cmp_pair_ll);
    qsort(ts[i], (size_t) nt, 2 * sizeof(long long), cmp_pair_ll);
  }

  for (int q = 0; q < n_req; ++q)
  {
    const int body = (int) req[3 * q];
    const bool tri = req[3 * q + 1] > 0.5;
    const long long key = (long long) req[3 * q + 2];
    reply[q] = -1.0;

    int lo = 0, hi = n_created - 1, i = -1;
    while (lo <= hi)
    {
      const int mid = (lo + hi) / 2;
      if (created[mid] == body)
      {
        i = mid;
        break;
      }
      if (created[mid] < body)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    MFTL_ASSERT(i >= 0 && "topology ids: asked about a body this rank did not create");
    if (i < 0) continue;

    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, created[i]);
    const long long idx = tri ? find_pair(ts[i], obj->tris.n_local, key) : find_pair(vs[i], obj->verts.n_local, key);
    MFTL_ASSERT(idx >= 0 && "topology ids: asked about an entity this rank does not own");
    if (idx < 0) continue;
    reply[q] = tri ? (double) obj->tris.rec[idx].global_id : (double) obj->verts.rec[idx].global_id;
  }

  for (int i = 0; i < n_created; ++i)
  {
    free(vs[i]);
    free(ts[i]);
  }
  free(vs);
  free(ts);
}

void MFTLTopoApplyIdReplies(MFTL_Ensemble *ens, const int *created, int n_created, int nproc, const double *reply)
{
  int *const counts = (int *) MFTLMallocOrDie(2 * (size_t) nproc * sizeof(int), "topology id reply cursors");
  int *const cursor = counts + nproc;
  ghost_id_offsets(ens, created, n_created, nproc, counts, cursor);

  for (int i = 0; i < n_created; ++i)
  {
    MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, created[i]);
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v) obj->verts.rec[v].global_id = (int) reply[cursor[obj->verts.rec[v].owner_proc]++];
    for (int t = obj->tris.n_local; t < obj->tris.n; ++t) obj->tris.rec[t].global_id = (int) reply[cursor[obj->tris.rec[t].owner_proc]++];
    for (int t = 0; t < obj->tris.n; ++t)
      for (int k = 0; k < 3; ++k) obj->tris.he[3 * t + k].global_id = 3 * obj->tris.rec[t].global_id + k;
  }
  free(counts);
}
