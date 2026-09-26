/*
 * Stage 16B — periodic image counters (plans/16 §3.1–§3.5).
 *
 *  - MFTLCanonicaliseImages: gather the edge graph with its lattice shifts to rank 0, a
 *    deterministic BFS spanning tree per connected component (root = smallest vertex gid),
 *    potentials phi, img' = img - phi (+ one whole-body lattice vector g), cycle residuals ->
 *    winding lattice -> cached per-axis wrap flags; scatter the new counters to every copy.
 *  - MFTLObjectRegaugeImages: whole-body lattice re-gauge that brings every component's root
 *    counter to 0 (keeps counters bounded).
 *  - MFTLCheckWrapFlags: debug check "WRAP_k <=> a cut edge along k" (never a cut edge along an
 *    axis whose flag is clear).
 *  - MFTLObjectReference: the collective body reference point (pos, img) of §3.5.
 *
 * Non-periodic domains: img == 0 everywhere and the flags are 0; canonicalisation is a no-op
 * without communication (bit-identical closed non-periodic behaviour, no cost).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_images.h"

static int mftlCmpInt(const void* a, const void* b)
{
  int x = *(const int*) a, y = *(const int*) b;
  return (x > y) - (x < y);
}

static int mftlCmpRec4(const void* a, const void* b)  // (gid, i0, i1, i2) by gid
{
  const int* x = (const int*) a;
  const int* y = (const int*) b;
  return (x[0] > y[0]) - (x[0] < y[0]);
}

static int mftlCmpEdge(const void* a, const void* b)  // (u, v, s0, s1, s2) by (u, v)
{
  const int* x = (const int*) a;
  const int* y = (const int*) b;
  if (x[0] != y[0]) return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}

static void mftlGathervInts(MPI_Comm comm, int root, const int* send, int n, int** recv, int* nrecv)
{
  int np, me;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &me);
  int* counts = (me == root) ? (int*) malloc((size_t) np * sizeof(int)) : MFTL_NULL;
  int* displs = (me == root) ? (int*) malloc((size_t) np * sizeof(int)) : MFTL_NULL;
  MPI_Gather(&n, 1, MPI_INT, counts, 1, MPI_INT, root, comm);
  int tot = 0;
  if (me == root)
    for (int p = 0; p < np; ++p)
    {
      displs[p] = tot;
      tot += counts[p];
    }
  *recv = (int*) malloc((size_t) (tot > 0 ? tot : 1) * sizeof(int));
  MPI_Gatherv(send, n, MPI_INT, *recv, counts, displs, MPI_INT, root, comm);
  *nrecv = tot;
  free(counts);
  free(displs);
}

int MFTLObjectWrapFlags(const MFTL_Object* obj, int wrap[3])
{
  for (int k = 0; k < 3; ++k) wrap[k] = obj->wrap_valid ? obj->wrap[k] : 0;
  return obj->wrap_valid ? MFTL_SUCCESS : MFTL_ERROR;
}

int MFTLObjectIsCompact(const MFTL_Object* obj)
{
  if (!MFTLDomainAnyPeriodic(obj->domain)) return 1;
  // Unknown winding (never canonicalised, or invalidated by a construction path) is never
  // taken as compact: wrap-sensitive collective entry points call MFTLObjectEnsureWinding first.
  if (!obj->wrap_valid) return 0;
  return !obj->wrap[0] && !obj->wrap[1] && !obj->wrap[2];
}

int MFTLObjectEnsureWinding(MFTL_Object* obj)
{
  if (!MFTLDomainAnyPeriodic(obj->domain))
  {
    obj->wrap[0] = obj->wrap[1] = obj->wrap[2] = 0;
    obj->wrap_valid = 1;
    return MFTL_SUCCESS;  // no communication: closed non-periodic behaviour unchanged
  }
  int loc = obj->wrap_valid ? 1 : 0, glob = 0;
  MPI_Allreduce(&loc, &glob, 1, MPI_INT, MPI_MIN, obj->domain->comm_cart);
  if (glob) return MFTL_SUCCESS;
  return MFTLCanonicaliseImages(obj);
}

int MFTLObjectHasVolume(const MFTL_Object* obj) { return !obj->declared_open && MFTLObjectIsCompact(obj); }

/* ------------------------------------------------------------------------------------------
 * canonicalisation
 * ------------------------------------------------------------------------------------------ */
static int mftlCanonicalise(MFTL_Object* obj, int zero_roots)
{
  MFTL_Domain* dom = obj->domain;
  if (!MFTLDomainAnyPeriodic(dom))
  {
    obj->wrap[0] = obj->wrap[1] = obj->wrap[2] = 0;
    obj->wrap_valid = 1;
    obj->n_wind = 0;
    return MFTL_SUCCESS;
  }
  MPI_Comm comm = dom->comm_cart;
  const int me = dom->rank;

  // --- 1. local records: authoritative (non-ghost) vertices (gid, img) and each edge once
  //        (the canonical half-edge of an authoritative triangle): (u, v, s) with u < v.
  int nv = 0, ne = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
    if (!obj->vertices[i].is_ghost && !(obj->vertices[i].flags & MFTL_FLAG_DEAD)) nv++;
  int* vrec = (int*) malloc((size_t) (4 * (nv > 0 ? nv : 1)) * sizeof(int));
  nv = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD)) continue;
    vrec[4 * nv + 0] = v->global_id;
    vrec[4 * nv + 1] = v->img[0];
    vrec[4 * nv + 2] = v->img[1];
    vrec[4 * nv + 3] = v->img[2];
    nv++;
  }
  int* erec = (int*) malloc((size_t) (5 * (obj->n_half_edges > 0 ? obj->n_half_edges : 1)) * sizeof(int));
  for (int h = 0; h < obj->n_half_edges; ++h)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[h];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    const MFTL_Triangle* t = &obj->triangles[he->triangle_idx];
    if (t->is_ghost || (t->flags & MFTL_FLAG_DEAD)) continue;
    if (MFTLHeEdgeId(obj, he) != he->global_id) continue;  // the other half-edge emits it
    const MFTL_Vertex* u = MFTLHeOrigin(obj, he);
    const MFTL_Vertex* v = MFTLHeTarget(obj, he);
    int s[3];
    MFTLEdgeShift(dom, u, v, s);
    int a = u->global_id, b = v->global_id;
    if (a > b)
    {
      int tmp = a;
      a = b;
      b = tmp;
      s[0] = -s[0];
      s[1] = -s[1];
      s[2] = -s[2];
    }
    erec[5 * ne + 0] = a;
    erec[5 * ne + 1] = b;
    erec[5 * ne + 2] = s[0];
    erec[5 * ne + 3] = s[1];
    erec[5 * ne + 4] = s[2];
    ne++;
  }

  int *V = MFTL_NULL, *E = MFTL_NULL, nV4 = 0, nE5 = 0;
  mftlGathervInts(comm, 0, vrec, 4 * nv, &V, &nV4);
  mftlGathervInts(comm, 0, erec, 5 * ne, &E, &nE5);
  free(vrec);
  free(erec);

  // --- 2. rank 0: BFS spanning trees, potentials, residuals
  int bad = 0, nU = 0;
  int* delta = MFTL_NULL;  // per unique vertex: 3 ints (img' - img)
  int* ugid = MFTL_NULL;
  int flags[3] = {0, 0, 0};
  int ngen = 0, cap_gen = 16;
  int(*gen)[3] = MFTL_NULL;
  if (me == 0)
  {
    const int nVr = nV4 / 4, nEr = nE5 / 5;
    qsort(V, (size_t) nVr, 4 * sizeof(int), mftlCmpRec4);
    // dedup vertices; duplicate copies must agree on img (§3.2: soup pieces)
    for (int i = 0; i < nVr; ++i)
    {
      if (nU > 0 && V[4 * (nU - 1)] == V[4 * i])
      {
        if (memcmp(&V[4 * (nU - 1) + 1], &V[4 * i + 1], 3 * sizeof(int)) != 0)
        {
          if (bad++ < 5) fprintf(stderr, "MFTLCanonicaliseImages: body %d: duplicate copies of vertex %d disagree on img\n", obj->global_id, V[4 * i]);
        }
        continue;
      }
      memmove(&V[4 * nU], &V[4 * i], 4 * sizeof(int));
      nU++;
    }
    qsort(E, (size_t) nEr, 5 * sizeof(int), mftlCmpEdge);
    int nEu = 0;
    for (int i = 0; i < nEr; ++i)
    {
      if (nEu > 0 && E[5 * (nEu - 1)] == E[5 * i] && E[5 * (nEu - 1) + 1] == E[5 * i + 1])
      {
        if (memcmp(&E[5 * (nEu - 1) + 2], &E[5 * i + 2], 3 * sizeof(int)) != 0)
          if (bad++ < 5) fprintf(stderr, "MFTLCanonicaliseImages: body %d: duplicate copies of edge %d-%d disagree on the shift\n", obj->global_id, E[5 * i], E[5 * i + 1]);
        continue;
      }
      memmove(&E[5 * nEu], &E[5 * i], 5 * sizeof(int));
      nEu++;
    }
    ugid = (int*) malloc((size_t) (nU > 0 ? nU : 1) * sizeof(int));
    for (int i = 0; i < nU; ++i) ugid[i] = V[4 * i];
    // adjacency (CSR, both directions)
    int* deg = (int*) calloc((size_t) (nU + 1), sizeof(int));
    int* eu = (int*) malloc((size_t) (nEu > 0 ? nEu : 1) * sizeof(int));
    int* ev = (int*) malloc((size_t) (nEu > 0 ? nEu : 1) * sizeof(int));
    for (int i = 0; i < nEu; ++i)
    {
      int* pu = bsearch(&E[5 * i], ugid, (size_t) nU, sizeof(int), mftlCmpInt);
      int* pv = bsearch(&E[5 * i + 1], ugid, (size_t) nU, sizeof(int), mftlCmpInt);
      if (!pu || !pv)
      {
        if (bad++ < 5) fprintf(stderr, "MFTLCanonicaliseImages: body %d: edge %d-%d references a vertex without an owner copy\n", obj->global_id, E[5 * i], E[5 * i + 1]);
        eu[i] = ev[i] = -1;
        continue;
      }
      eu[i] = (int) (pu - ugid);
      ev[i] = (int) (pv - ugid);
      deg[eu[i] + 1]++;
      deg[ev[i] + 1]++;
    }
    for (int i = 0; i < nU; ++i) deg[i + 1] += deg[i];
    int* adj = (int*) malloc((size_t) (2 * (nEu > 0 ? nEu : 1)) * sizeof(int));  // (neighbour, edge*2 + dir)
    int* fill = (int*) malloc((size_t) (nU > 0 ? nU : 1) * sizeof(int));
    memcpy(fill, deg, (size_t) nU * sizeof(int));
    int* adj_e = (int*) malloc((size_t) (2 * (nEu > 0 ? nEu : 1)) * sizeof(int));
    for (int i = 0; i < nEu; ++i)
    {
      if (eu[i] < 0) continue;
      adj[fill[eu[i]]] = ev[i];
      adj_e[fill[eu[i]]++] = 2 * i;  // u -> v: +s
      adj[fill[ev[i]]] = eu[i];
      adj_e[fill[ev[i]]++] = 2 * i + 1;  // v -> u: -s
    }
    // deterministic order: neighbours by index (gid order)
    for (int i = 0; i < nU; ++i)
    {
      int a = deg[i], b = deg[i + 1];
      for (int x = a + 1; x < b; ++x)  // insertion sort (small degrees)
      {
        int kv = adj[x], ke = adj_e[x], y = x - 1;
        while (y >= a && (adj[y] > kv || (adj[y] == kv && adj_e[y] > ke)))
        {
          adj[y + 1] = adj[y];
          adj_e[y + 1] = adj_e[y];
          y--;
        }
        adj[y + 1] = kv;
        adj_e[y + 1] = ke;
      }
    }
    int* phi = (int*) malloc((size_t) (3 * (nU > 0 ? nU : 1)) * sizeof(int));
    int* seen = (int*) calloc((size_t) (nU > 0 ? nU : 1), sizeof(int));
    int* comp = (int*) malloc((size_t) (nU > 0 ? nU : 1) * sizeof(int));
    int* queue = (int*) malloc((size_t) (nU > 0 ? nU : 1) * sizeof(int));
    int ncomp = 0;
    delta = (int*) malloc((size_t) (3 * (nU > 0 ? nU : 1)) * sizeof(int));
    gen = malloc((size_t) cap_gen * sizeof(*gen));
    for (int r = 0; r < nU; ++r)  // roots in increasing gid order = smallest gid of each component
    {
      if (seen[r]) continue;
      int qh = 0, qt = 0;
      queue[qt++] = r;
      seen[r] = 1;
      comp[r] = ncomp;
      phi[3 * r] = phi[3 * r + 1] = phi[3 * r + 2] = 0;
      while (qh < qt)
      {
        const int u = queue[qh++];
        for (int x = deg[u]; x < deg[u + 1]; ++x)
        {
          const int v = adj[x], ei = adj_e[x] / 2, dir = (adj_e[x] & 1) ? -1 : 1;
          if (seen[v]) continue;
          seen[v] = 1;
          comp[v] = ncomp;
          for (int k = 0; k < 3; ++k) phi[3 * v + k] = phi[3 * u + k] + dir * E[5 * ei + 2 + k];
          queue[qt++] = v;
        }
      }
      // whole-body re-gauge g: root keeps its counter (or goes to 0 when zero_roots)
      int g[3] = {0, 0, 0};
      if (zero_roots)
        for (int k = 0; k < 3; ++k) g[k] = -V[4 * r + 1 + k];
      for (int q = 0; q < qt; ++q)
      {
        const int v = queue[q];
        for (int k = 0; k < 3; ++k) delta[3 * v + k] = -phi[3 * v + k] + g[k];
      }
      ncomp++;
    }
    // residuals of the non-tree edges: r = s - (phi_v - phi_u)
    for (int i = 0; i < nEu; ++i)
    {
      if (eu[i] < 0) continue;
      int r[3], nz = 0;
      for (int k = 0; k < 3; ++k)
      {
        r[k] = E[5 * i + 2 + k] - (phi[3 * ev[i] + k] - phi[3 * eu[i] + k]);
        if (r[k] != 0) nz = 1;
      }
      if (!nz) continue;
      for (int k = 0; k < 3; ++k)
        if (r[k] != 0) flags[k] = 1;
      // store the generator (sign-normalised: first non-zero component positive), dedup
      int sgn = 1;
      for (int k = 0; k < 3; ++k)
        if (r[k] != 0)
        {
          sgn = r[k] > 0 ? 1 : -1;
          break;
        }
      for (int k = 0; k < 3; ++k) r[k] *= sgn;
      int dup = 0;
      for (int q = 0; q < ngen && !dup; ++q)
        if (gen[q][0] == r[0] && gen[q][1] == r[1] && gen[q][2] == r[2]) dup = 1;
      if (!dup)
      {
        if (ngen == cap_gen)
        {
          cap_gen *= 2;
          gen = realloc(gen, (size_t) cap_gen * sizeof(*gen));
        }
        memcpy(gen[ngen++], r, sizeof(r));
      }
    }
    free(deg);
    free(eu);
    free(ev);
    free(adj);
    free(adj_e);
    free(fill);
    free(phi);
    free(seen);
    free(comp);
    free(queue);
  }
  free(V);
  free(E);

  // --- 3. scatter: every rank applies the deltas to all of its copies (owned and ghost)
  int hdr[6] = {bad, nU, flags[0], flags[1], flags[2], ngen};
  MPI_Bcast(hdr, 6, MPI_INT, 0, comm);
  bad = hdr[0];
  nU = hdr[1];
  if (me != 0)
  {
    ugid = (int*) malloc((size_t) (nU > 0 ? nU : 1) * sizeof(int));
    delta = (int*) malloc((size_t) (3 * (nU > 0 ? nU : 1)) * sizeof(int));
    gen = malloc((size_t) (hdr[5] > 0 ? hdr[5] : 1) * sizeof(*gen));
  }
  MPI_Bcast(ugid, nU, MPI_INT, 0, comm);
  MPI_Bcast(delta, 3 * nU, MPI_INT, 0, comm);
  if (hdr[5] > 0) MPI_Bcast(gen, 3 * hdr[5], MPI_INT, 0, comm);
  if (bad)
  {
    free(ugid);
    free(delta);
    free(gen);
    return MFTL_ERROR;
  }
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    int* p = bsearch(&v->global_id, ugid, (size_t) nU, sizeof(int), mftlCmpInt);
    if (!p) continue;  // a stale ghost without an owner copy (dropped by the next redistribution)
    const int j = (int) (p - ugid);
    for (int k = 0; k < 3; ++k) v->img[k] += delta[3 * j + k];
  }
  // Stage 16C (plans/16 §4.4.4): a class transition (winding acquired or lost) is reported
  if (obj->wrap_valid && (obj->wrap[0] != hdr[2] || obj->wrap[1] != hdr[3] || obj->wrap[2] != hdr[4]))
  {
    obj->class_transition = 1;
    if (me == 0)
      fprintf(stderr, "MFTL: body %d changed its winding class: wrap (%d,%d,%d) -> (%d,%d,%d)\n", obj->global_id, obj->wrap[0], obj->wrap[1], obj->wrap[2], hdr[2],
              hdr[3], hdr[4]);
  }
  obj->wrap[0] = hdr[2];
  obj->wrap[1] = hdr[3];
  obj->wrap[2] = hdr[4];
  obj->wrap_valid = 1;
  free(obj->wind_gen);
  obj->wind_gen = gen;
  obj->n_wind = hdr[5];
  free(ugid);
  free(delta);
  return MFTL_SUCCESS;
}

int MFTLCanonicaliseImages(MFTL_Object* obj) { return mftlCanonicalise(obj, 0); }
int MFTLObjectRegaugeImages(MFTL_Object* obj) { return mftlCanonicalise(obj, 1); }

int MFTLCheckWrapFlags(MFTL_Object* obj, int verbose)
{
  MFTL_Domain* dom = obj->domain;
  if (!MFTLDomainAnyPeriodic(dom)) return MFTL_SUCCESS;
  int bad = 0;
  if (obj->wrap_valid)
  {
    for (int h = 0; h < obj->n_half_edges && bad < 1000000; ++h)
    {
      const MFTL_HalfEdge* he = &obj->half_edges[h];
      if (he->flags & MFTL_FLAG_DEAD) continue;
      int s[3];
      MFTLEdgeShift(dom, MFTLHeOrigin(obj, he), MFTLHeTarget(obj, he), s);
      for (int k = 0; k < 3; ++k)
        if (s[k] != 0 && !obj->wrap[k])
        {
          if (verbose && bad < 5)
            fprintf(stderr, "[WrapFlags r%d] body %d: half-edge %d is cut along axis %d, but the body does not wrap %d\n", dom->rank, obj->global_id,
                    he->global_id, k, k);
          bad++;
        }
    }
  }
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, dom->comm_cart);
  return gbad ? MFTL_ERROR : MFTL_SUCCESS;
}

int MFTLCountCutEdges(MFTL_Object* obj, int count[3])
{
  MFTL_Domain* dom = obj->domain;
  int loc[3] = {0, 0, 0};
  for (int h = 0; h < obj->n_half_edges; ++h)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[h];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    const MFTL_Triangle* t = &obj->triangles[he->triangle_idx];
    if (t->is_ghost || (t->flags & MFTL_FLAG_DEAD)) continue;
    int s[3];
    MFTLEdgeShift(dom, MFTLHeOrigin(obj, he), MFTLHeTarget(obj, he), s);
    for (int k = 0; k < 3; ++k)
      if (s[k] != 0) loc[k]++;
  }
  MPI_Allreduce(loc, count, 3, MPI_INT, MPI_SUM, dom->comm_cart);  // half-edges: 2 per interior cut edge
  return MFTL_SUCCESS;
}

int MFTLObjectReference(const MFTL_Object* obj, double pos_ref[3], int img_ref[3])
{
  const MFTL_Domain* dom = obj->domain;
  // the lowest-gid authoritative vertex (§3.5), agreed collectively
  int best = -1, bi = -1;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
    if (best < 0 || v->global_id < best)
    {
      best = v->global_id;
      bi = i;
    }
  }
  int loc[2] = {best < 0 ? 2147483647 : best, dom->rank}, glob[2];
  MPI_Allreduce(loc, glob, 1, MPI_2INT, MPI_MINLOC, dom->comm_cart);
  if (glob[0] == 2147483647)
  {
    pos_ref[0] = pos_ref[1] = pos_ref[2] = 0.0;
    img_ref[0] = img_ref[1] = img_ref[2] = 0;
    return 0;
  }
  double buf[3] = {0, 0, 0};
  int ib[3] = {0, 0, 0};
  if (dom->rank == glob[1])
  {
    memcpy(buf, obj->vertices[bi].pos, sizeof(buf));
    memcpy(ib, obj->vertices[bi].img, sizeof(ib));
  }
  MPI_Bcast(buf, 3, MPI_DOUBLE, glob[1], dom->comm_cart);
  MPI_Bcast(ib, 3, MPI_INT, glob[1], dom->comm_cart);
  memcpy(pos_ref, buf, sizeof(buf));
  memcpy(img_ref, ib, sizeof(ib));
  return 1;
}

int MFTLObjectTakeClassTransition(MFTL_Object* obj)
{
  const int t = obj->class_transition;
  obj->class_transition = 0;
  return t;
}
