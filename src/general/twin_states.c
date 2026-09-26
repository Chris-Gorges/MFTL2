/*
 * Stage 16A — topology and twin-state invariant (plans/16 §2).
 *
 *  - declared class (MFTLObjectDeclareOpen), boundary cache and boundary-vertex convention;
 *  - MFTLVertexRing (CLOSED / OPEN_COMPLETE / INCOMPLETE);
 *  - rank-local checks and the collective MFTLValidateTwinStatesEx (remote liveness);
 *  - GLOBAL discovery mode (lookup of twin gids among the live authoritative half-edges of all
 *    ranks) used by validation and MFTLObjectCommitCopy;
 *  - ghost completion G1/G2 (owned-mode geometric round + mandatory global fallback);
 *  - boundary-loop tagging (builders) and MFTLBoundaryLoops (distributed chaining).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_communication.h"
#include "mftl_twin_internal.h"
#include "mftl_images.h"

/* ========================================================================================
 * small helpers
 * ======================================================================================== */
static inline int mftlOriginIdx(const MFTL_Object* obj, int h)
{
  const int n = obj->half_edges[h].next_idx;
  return obj->half_edges[obj->half_edges[n].next_idx].vertex_idx;
}

static inline int mftlHeLive(const MFTL_Object* obj, int h)
{
  const MFTL_HalfEdge* he = &obj->half_edges[h];
  if (he->flags & MFTL_FLAG_DEAD) return 0;
  if (he->triangle_idx < 0 || (obj->triangles[he->triangle_idx].flags & MFTL_FLAG_DEAD)) return 0;
  return 1;
}

static int mftlCmpInt(const void* a, const void* b)
{
  int x = *(const int*) a, y = *(const int*) b;
  return (x > y) - (x < y);
}

static int mftlCmpInt2(const void* a, const void* b)
{
  const int* x = (const int*) a;
  const int* y = (const int*) b;
  if (x[0] != y[0]) return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}

static void mftlAllgathervBytes(MPI_Comm comm, const void* send, int nsend, void** recv, int* nrecv, int** counts_out, int** displs_out)
{
  int np;
  MPI_Comm_size(comm, &np);
  int* counts = (int*) malloc((size_t) np * sizeof(int));
  int* displs = (int*) malloc((size_t) np * sizeof(int));
  MPI_Allgather(&nsend, 1, MPI_INT, counts, 1, MPI_INT, comm);
  int total = 0;
  for (int p = 0; p < np; ++p)
  {
    displs[p] = total;
    total += counts[p];
  }
  *recv = malloc((size_t) (total > 0 ? total : 1));
  MPI_Allgatherv(send, nsend, MPI_BYTE, *recv, counts, displs, MPI_BYTE, comm);
  *nrecv = total;
  if (counts_out)
    *counts_out = counts;
  else
    free(counts);
  if (displs_out)
    *displs_out = displs;
  else
    free(displs);
}

void MFTLObjectDeclareOpen(MFTL_Object* obj, int open) { obj->declared_open = open ? 1 : 0; }

void MFTLRefreshBoundaryCache(MFTL_Object* obj)
{
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    he->boundary = MFTLHeIsBoundary(he);
  }
  // boundary-vertex convention: out_he_idx is the outgoing BOUNDARY half-edge (plans/16 §2.6)
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (!he->boundary || (he->flags & MFTL_FLAG_DEAD)) continue;
    if (he->next_idx < 0 || obj->half_edges[he->next_idx].next_idx < 0) continue;
    const int o = mftlOriginIdx(obj, i);
    if (o >= 0 && o < obj->n_vertices) obj->vertices[o].out_he_idx = i;
  }
}

/* ========================================================================================
 * rings
 * ======================================================================================== */
int MFTLVertexRing(const MFTL_Object* obj, int v_idx, int* out_he, int max_he, int* n_out)
{
  if (n_out) *n_out = 0;
  const MFTL_Vertex* v = &obj->vertices[v_idx];
  const int start = v->out_he_idx;
  if (start < 0 || start >= obj->n_half_edges) return MFTL_RING_INCOMPLETE;
  const int cap = 4 * obj->n_half_edges + 8;

  // rewind clockwise: curr <- twin(prev(curr)) while prev(curr) is LOCAL
  int curr = start, steps = 0, closed = 0;
  while (1)
  {
    const int prev = obj->half_edges[obj->half_edges[curr].next_idx].next_idx;
    const MFTL_HalfEdge* hp = &obj->half_edges[prev];
    if (!MFTLHeIsLocal(hp))
    {
      if (!MFTLHeIsBoundary(hp)) return MFTL_RING_INCOMPLETE;
      break;
    }
    curr = hp->twin_idx;
    if (curr == start)
    {
      closed = 1;
      break;
    }
    if (++steps > cap) return MFTL_RING_INCOMPLETE;
  }
  // replay counter-clockwise: curr <- next(twin(curr))
  const int first = curr;
  int n = 0;
  steps = 0;
  while (1)
  {
    if (out_he && n < max_he) out_he[n] = curr;
    n++;
    const MFTL_HalfEdge* hc = &obj->half_edges[curr];
    if (!MFTLHeIsLocal(hc))
    {
      if (n_out) *n_out = (out_he && n > max_he) ? max_he : n;
      if (closed || !MFTLHeIsBoundary(hc)) return MFTL_RING_INCOMPLETE;
      return MFTL_RING_OPEN_COMPLETE;
    }
    curr = obj->half_edges[hc->twin_idx].next_idx;
    if (curr == first) break;
    if (++steps > cap) return MFTL_RING_INCOMPLETE;
  }
  if (n_out) *n_out = (out_he && n > max_he) ? max_he : n;
  return closed ? MFTL_RING_CLOSED : MFTL_RING_INCOMPLETE;
}

/* ========================================================================================
 * boundary-loop tagging of directed boundary half-edges (builders)
 * ======================================================================================== */
int MFTLInternalTagBoundaryLoops(int n, const int (*ab)[2], int* tag_out)
{
  if (n == 0) return MFTL_SUCCESS;
  // successor of record r (a->b): the record starting at b
  int(*by_origin)[2] = malloc((size_t) n * sizeof(*by_origin));  // {origin gid, record}
  for (int i = 0; i < n; ++i)
  {
    by_origin[i][0] = ab[i][0];
    by_origin[i][1] = i;
  }
  qsort(by_origin, (size_t) n, sizeof(*by_origin), mftlCmpInt2);
  int ok = 1;
  for (int i = 1; i < n; ++i)
    if (by_origin[i][0] == by_origin[i - 1][0]) ok = 0;  // two boundary half-edges leave one vertex
  int* succ = malloc((size_t) n * sizeof(int));
  for (int i = 0; i < n && ok; ++i)
  {
    int lo = 0, hi = n - 1, found = -1;
    while (lo <= hi)
    {
      int mid = (lo + hi) / 2;
      if (by_origin[mid][0] == ab[i][1])
      {
        found = by_origin[mid][1];
        break;
      }
      if (by_origin[mid][0] < ab[i][1])
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    if (found < 0) ok = 0;
    succ[i] = found;
  }
  int* loop_of = malloc((size_t) n * sizeof(int));
  for (int i = 0; i < n; ++i) loop_of[i] = -1;
  int n_loops = 0;
  int(*loop_min)[2] = malloc((size_t) n * sizeof(*loop_min));  // {min vertex gid, loop}
  for (int i = 0; i < n && ok; ++i)
  {
    if (loop_of[i] >= 0) continue;
    int r = i, len = 0, mn = ab[i][0];
    while (loop_of[r] < 0)
    {
      loop_of[r] = n_loops;
      if (ab[r][0] < mn) mn = ab[r][0];
      r = succ[r];
      if (++len > n) break;
    }
    if (r != i) ok = 0;  // chain did not close on itself
    loop_min[n_loops][0] = mn;
    loop_min[n_loops][1] = n_loops;
    n_loops++;
  }
  if (ok)
  {
    qsort(loop_min, (size_t) n_loops, sizeof(*loop_min), mftlCmpInt2);
    int* rank_of = malloc((size_t) (n_loops > 0 ? n_loops : 1) * sizeof(int));
    for (int k = 0; k < n_loops; ++k) rank_of[loop_min[k][1]] = k;
    for (int i = 0; i < n; ++i) tag_out[i] = rank_of[loop_of[i]];
    free(rank_of);
  }
  free(by_origin);
  free(succ);
  free(loop_of);
  free(loop_min);
  return ok ? MFTL_SUCCESS : MFTL_ERROR;
}

/* Directed duplicates among live half-edges: returns the number of duplicate pairs. */
int MFTLInternalCountDirectedDuplicates(const MFTL_Object* obj, int verbose)
{
  int n = 0;
  int(*rec)[3] = malloc((size_t) (obj->n_half_edges > 0 ? obj->n_half_edges : 1) * sizeof(*rec));
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    if (!mftlHeLive(obj, i)) continue;
    rec[n][0] = obj->vertices[mftlOriginIdx(obj, i)].global_id;
    rec[n][1] = obj->vertices[obj->half_edges[i].vertex_idx].global_id;
    rec[n][2] = obj->half_edges[i].global_id;
    n++;
  }
  qsort(rec, (size_t) n, sizeof(*rec), mftlCmpInt2);
  int dup = 0;
  for (int i = 1; i < n; ++i)
    if (rec[i][0] == rec[i - 1][0] && rec[i][1] == rec[i - 1][1])
    {
      if (verbose && dup < 10)
        fprintf(stderr, "[TwinStates] duplicate directed half-edge %d->%d (half-edges %d, %d): non-manifold edge or inconsistent orientation\n",
                rec[i][0], rec[i][1], rec[i - 1][2], rec[i][2]);
      dup++;
    }
  free(rec);
  return dup;
}

/* ========================================================================================
 * rank-local checks
 * ======================================================================================== */
#define MFTL_TS_REPORT(...)                               \
  do                                                      \
  {                                                       \
    if (verbose && bad < 20) fprintf(stderr, __VA_ARGS__); \
    bad++;                                                \
  } while (0)

int MFTLCheckTwinStatesLocal(const MFTL_Object* obj, int flags, int verbose)
{
  int bad = 0;
  const int rank = obj->domain ? obj->domain->rank : 0;
  const int nh = obj->n_half_edges, nv = obj->n_vertices;

  for (int i = 0; i < nh; ++i)
  {
    if (!mftlHeLive(obj, i)) continue;
    const MFTL_HalfEdge* he = &obj->half_edges[i];
    if (MFTLHeIsLocal(he))
    {
      if (he->twin_idx >= nh)
      {
        MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: twin index out of range\n", rank, he->global_id);
        continue;
      }
      const MFTL_HalfEdge* tw = &obj->half_edges[he->twin_idx];
      if (!mftlHeLive(obj, he->twin_idx)) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: LOCAL twin is dead\n", rank, he->global_id);
      if (tw->twin_idx != i) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: LOCAL twin not symmetric\n", rank, he->global_id);
      if (he->twin_gid != tw->global_id)
        MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: LOCAL twin_gid %d != twin gid %d\n", rank, he->global_id, he->twin_gid, tw->global_id);
      if (mftlOriginIdx(obj, i) != tw->vertex_idx || he->vertex_idx != mftlOriginIdx(obj, he->twin_idx))
        MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: LOCAL twin has inconsistent vertices\n", rank, he->global_id);
      if (he->boundary) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: boundary cache set on a LOCAL half-edge\n", rank, he->global_id);
    }
    else if (MFTLHeIsRemote(he))
    {
      if (he->boundary) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: boundary cache set on a REMOTE half-edge\n", rank, he->global_id);
      if (he->twin_gid == he->global_id) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: REMOTE twin names itself\n", rank, he->global_id);
    }
    else if (MFTLHeIsBoundary(he))
    {
      if (!he->boundary) MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: boundary cache not set on a BOUNDARY half-edge\n", rank, he->global_id);
      if (!obj->declared_open)
        MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: BOUNDARY half-edge (tag %d) on a body not declared open\n", rank, he->global_id,
                       MFTLHeBoundaryTag(he));
    }
    else
    {
      MFTL_TS_REPORT("[TwinStates r%d] half-edge %d: twin state UNKNOWN (twin_gid %d)\n", rank, he->global_id, he->twin_gid);
    }
  }

  if (flags & MFTL_VALIDATE_CHEAP) goto g1;  // Release quiescent points: states + G1 only
  bad += MFTLInternalCountDirectedDuplicates(obj, verbose);

  // per-vertex: out half-edge, boundary-vertex convention, bow-ties, G2
  int* incident = calloc((size_t) (nv > 0 ? nv : 1), sizeof(int));
  for (int i = 0; i < nh; ++i)
    if (mftlHeLive(obj, i) && obj->half_edges[i].vertex_idx >= 0 && obj->half_edges[i].vertex_idx < nv) incident[obj->half_edges[i].vertex_idx]++;
  for (int v = 0; v < nv; ++v)
  {
    const MFTL_Vertex* vx = &obj->vertices[v];
    if (vx->flags & MFTL_FLAG_DEAD) continue;
    if (incident[v] == 0) continue;  // unreferenced (ghost remnant); carries no mesh
    if (vx->out_he_idx < 0 || vx->out_he_idx >= nh || !mftlHeLive(obj, vx->out_he_idx) || mftlOriginIdx(obj, vx->out_he_idx) != v)
    {
      MFTL_TS_REPORT("[TwinStates r%d] vertex %d: invalid outgoing half-edge\n", rank, vx->global_id);
      continue;
    }
    if (vx->is_ghost) continue;
    int n_ring = 0;
    const int cls = MFTLVertexRing(obj, v, MFTL_NULL, 0, &n_ring);
    if (cls == MFTL_RING_INCOMPLETE)
    {
      if (flags & MFTL_VALIDATE_GHOSTS) MFTL_TS_REPORT("[TwinStates r%d] G2: owned vertex %d has an incomplete fan\n", rank, vx->global_id);
      continue;
    }
    if (n_ring != incident[v])
      MFTL_TS_REPORT("[TwinStates r%d] vertex %d: fan has %d of %d incident triangles (bow-tie / non-manifold vertex)\n", rank, vx->global_id, n_ring,
                     incident[v]);
    if (cls == MFTL_RING_OPEN_COMPLETE && !MFTLHeIsBoundary(&obj->half_edges[vx->out_he_idx]))
      MFTL_TS_REPORT("[TwinStates r%d] boundary vertex %d: out half-edge is not its outgoing BOUNDARY half-edge\n", rank, vx->global_id);
  }
  free(incident);

g1:
  if (flags & MFTL_VALIDATE_GHOSTS)
  {
    for (int t = 0; t < obj->n_triangles; ++t)
    {
      const MFTL_Triangle* tri = &obj->triangles[t];
      if (tri->is_ghost || (tri->flags & MFTL_FLAG_DEAD)) continue;
      int h = tri->he_idx;
      for (int k = 0; k < 3; ++k, h = obj->half_edges[h].next_idx)
        if (!MFTLHeIsLocal(&obj->half_edges[h]) && !MFTLHeIsBoundary(&obj->half_edges[h]))
          MFTL_TS_REPORT("[TwinStates r%d] G1: owned triangle %d edge %d is not LOCAL/BOUNDARY\n", rank, tri->global_id, obj->half_edges[h].global_id);
    }
  }
  return bad;
}

/* ========================================================================================
 * GLOBAL discovery mode (lookup only): for every local request (want gid, code), find the live
 * AUTHORITATIVE (non-ghost triangle) holders of `want` on all ranks. Output per request:
 * holder[i] (smallest holding rank, -1 none), match[i] (every holder's twin code of `want`
 * equals code[i]) and conflict[i] (holders' triangle signatures differ).
 * ======================================================================================== */
static uint64_t mftlFnv(uint64_t h, const void* p, size_t n)
{
  const unsigned char* c = (const unsigned char*) p;
  for (size_t i = 0; i < n; ++i)
  {
    h ^= c[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t mftlTriSignature(const MFTL_Object* obj, int t)
{
  uint64_t h = 1469598103934665603ULL;
  const MFTL_Triangle* tri = &obj->triangles[t];
  h = mftlFnv(h, &tri->global_id, sizeof(int));
  int e = tri->he_idx;
  for (int k = 0; k < 3; ++k, e = obj->half_edges[e].next_idx)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[e];
    const MFTL_Vertex* v = &obj->vertices[he->vertex_idx];
    int code = MFTLHeTwinCode(obj, he);
    h = mftlFnv(h, &he->global_id, sizeof(int));
    h = mftlFnv(h, &code, sizeof(int));
    h = mftlFnv(h, &v->global_id, sizeof(int));
    h = mftlFnv(h, v->pos, 3 * sizeof(double));
  }
  return h;
}

typedef struct
{
  int req;  // global request index
  int rank;
  int match;
  int pad;
  uint64_t sig;
} mftlLookupAns;

int MFTLInternalGlobalLookup(MFTL_Object* obj, int n, const int* want, const int* code, int* holder, int* match, int* conflict)
{
  MFTL_Domain* dom = obj->domain;
  MPI_Comm comm = dom->comm_cart;
  int total = 0;
  MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, comm);
  for (int i = 0; i < n; ++i)
  {
    holder[i] = -1;
    match[i] = 1;
    conflict[i] = 0;
  }
  if (total == 0) return 0;

  int* send = malloc((size_t) (2 * (n > 0 ? n : 1)) * sizeof(int));
  for (int i = 0; i < n; ++i)
  {
    send[2 * i] = want[i];
    send[2 * i + 1] = code[i];
  }
  int *all = MFTL_NULL, *counts = MFTL_NULL, *displs = MFTL_NULL, nbytes = 0;
  mftlAllgathervBytes(comm, send, 2 * n * (int) sizeof(int), (void**) &all, &nbytes, &counts, &displs);
  free(send);
  const int nall = nbytes / (int) (2 * sizeof(int));
  const int my_first = displs[dom->rank] / (int) (2 * sizeof(int));
  free(counts);
  free(displs);

  int na = 0, cap = 64;
  mftlLookupAns* ans = malloc((size_t) cap * sizeof(mftlLookupAns));
  for (int k = 0; k < nall; ++k)
  {
    khiter_t it = kh_get(halfedge_map, obj->halfedge_lookup, all[2 * k]);
    if (it == kh_end(obj->halfedge_lookup)) continue;
    const int h = kh_value(obj->halfedge_lookup, it);
    if (h < 0 || h >= obj->n_half_edges || !mftlHeLive(obj, h)) continue;
    const MFTL_HalfEdge* he = &obj->half_edges[h];
    if (he->global_id != all[2 * k]) continue;
    if (obj->triangles[he->triangle_idx].is_ghost) continue;  // not authoritative
    if (na == cap)
    {
      cap *= 2;
      ans = realloc(ans, (size_t) cap * sizeof(mftlLookupAns));
    }
    mftlLookupAns* a = &ans[na++];
    memset(a, 0, sizeof(*a));
    a->req = k;
    a->rank = dom->rank;
    a->match = (MFTLHeTwinCode(obj, he) == all[2 * k + 1]);
    a->sig = mftlTriSignature(obj, he->triangle_idx);
  }
  free(all);

  mftlLookupAns* all_ans = MFTL_NULL;
  int ab = 0;
  mftlAllgathervBytes(comm, ans, na * (int) sizeof(mftlLookupAns), (void**) &all_ans, &ab, MFTL_NULL, MFTL_NULL);
  free(ans);
  const int nans = ab / (int) sizeof(mftlLookupAns);
  uint64_t* sig = calloc((size_t) (n > 0 ? n : 1), sizeof(uint64_t));
  int resolved = 0;
  for (int k = 0; k < nans; ++k)
  {
    const int i = all_ans[k].req - my_first;
    if (i < 0 || i >= n) continue;
    if (holder[i] < 0)
    {
      holder[i] = all_ans[k].rank;
      sig[i] = all_ans[k].sig;
      resolved++;
    }
    else
    {
      if (all_ans[k].rank < holder[i]) holder[i] = all_ans[k].rank;
      if (all_ans[k].sig != sig[i]) conflict[i] = 1;
    }
    if (!all_ans[k].match) match[i] = 0;
  }
  free(sig);
  free(all_ans);
  return resolved;
}

/* ========================================================================================
 * collective validation
 * ======================================================================================== */
int MFTLValidateTwinStatesEx(MFTL_Object* obj, int flags)
{
  MFTL_Domain* dom = obj->domain;
  MPI_Comm comm = dom->comm_cart;
  int bad = MFTLCheckTwinStatesLocal(obj, flags, 1);

  // remote liveness (REMOTE: partner live + names back) and BOUNDARY agreement across copies
  int n = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (mftlHeLive(obj, i) && (MFTLHeIsRemote(&obj->half_edges[i]) || MFTLHeIsBoundary(&obj->half_edges[i]))) n++;
  int* want = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* code = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* kind = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* holder = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* match = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* conflict = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* self = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  n = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    if (!mftlHeLive(obj, i)) continue;
    const MFTL_HalfEdge* he = &obj->half_edges[i];
    if (MFTLHeIsRemote(he))
    {
      want[n] = he->twin_gid;  // the partner must be live and name me back
      code[n] = he->global_id;
      kind[n] = 0;
    }
    else if (MFTLHeIsBoundary(he))
    {
      want[n] = he->global_id;  // every authoritative copy must carry the same code
      code[n] = he->twin_gid;
      kind[n] = 1;
    }
    else
      continue;
    self[n] = i;
    n++;
  }
  MFTLInternalGlobalLookup(obj, n, want, code, holder, match, conflict);
  int printed = 0;
  for (int k = 0; k < n; ++k)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[self[k]];
    int fail = 0;
    if (kind[k] == 0 && holder[k] < 0)
    {
      fail = 1;
      if (printed++ < 20) fprintf(stderr, "[TwinStates r%d] half-edge %d: REMOTE twin %d is not live on any rank\n", dom->rank, he->global_id, he->twin_gid);
    }
    else if (kind[k] == 0 && !match[k])
    {
      fail = 1;
      if (printed++ < 20) fprintf(stderr, "[TwinStates r%d] half-edge %d: REMOTE twin %d does not name it back\n", dom->rank, he->global_id, he->twin_gid);
    }
    else if (kind[k] == 1 && holder[k] >= 0 && !match[k])
    {
      fail = 1;
      if (printed++ < 20) fprintf(stderr, "[TwinStates r%d] half-edge %d: BOUNDARY state differs between copies\n", dom->rank, he->global_id);
    }
    else if (kind[k] == 1 && holder[k] < 0)
    {
      fail = 1;  // a stored boundary half-edge must have an authoritative copy somewhere
      if (printed++ < 20) fprintf(stderr, "[TwinStates r%d] half-edge %d: BOUNDARY half-edge without an authoritative copy\n", dom->rank, he->global_id);
    }
    bad += fail;
  }
  free(want);
  free(code);
  free(kind);
  free(holder);
  free(match);
  free(conflict);
  free(self);

  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, comm);
  if (gbad && dom->rank == 0) fprintf(stderr, "[TwinStates] body %d: %d twin-state violation(s) (details above)\n", obj->global_id, gbad);
  return gbad ? MFTL_ERROR : MFTL_SUCCESS;
}

/* Quiescent-point hook (plans/16 §2.3): the full collective validation in Debug builds or when
   the environment variable MFTL_VALIDATE is set to a non-zero value; otherwise the cheap
   rank-local checks. Returns MFTL_SUCCESS / MFTL_ERROR (collective in the full mode only). */
int MFTLInternalQuiescentCheck(MFTL_Object* obj, int flags)
{
  static int env = -1;
  if (env < 0)
  {
    const char* e = getenv("MFTL_VALIDATE");
    env = (e && atoi(e) != 0) ? 1 : 0;
  }
#ifndef NDEBUG
  const int full = 1;
#else
  const int full = env;
#endif
  if (full)
  {
    int rc = MFTLValidateTwinStatesEx(obj, flags);
    // Stage 16B (§3.3): the cached wrap flags must hold (no cut edge along a compact axis)
    if (MFTLCheckWrapFlags(obj, 1) != MFTL_SUCCESS) rc = MFTL_ERROR;
    return rc;
  }
  return MFTLCheckTwinStatesLocal(obj, flags | MFTL_VALIDATE_CHEAP, 1) ? MFTL_ERROR : MFTL_SUCCESS;
}

/* ========================================================================================
 * MFTLObjectCommitCopy
 * ======================================================================================== */
int MFTLObjectCommitCopy(MFTL_Object* dst, int mode)
{
  MFTL_Domain* dom = dst->domain;
  MPI_Comm comm = dom->comm_cart;
  int bad = 0;

  int n = 0;
  for (int i = 0; i < dst->n_half_edges; ++i)
  {
    if (!mftlHeLive(dst, i)) continue;
    const MFTL_HalfEdge* he = &dst->half_edges[i];
    if (!MFTLHeIsLocal(he) && !MFTLHeIsBoundary(he)) n++;
  }
  int* want = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* code = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* self = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* holder = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* match = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  int* conflict = malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  n = 0;
  for (int i = 0; i < dst->n_half_edges; ++i)
  {
    if (!mftlHeLive(dst, i)) continue;
    const MFTL_HalfEdge* he = &dst->half_edges[i];
    if (MFTLHeIsLocal(he) || MFTLHeIsBoundary(he)) continue;
    want[n] = he->twin_gid;  // -1 (unknown) is looked up too and never found
    code[n] = he->global_id;
    self[n] = i;
    n++;
  }
  MFTLInternalGlobalLookup(dst, n, want, code, holder, match, conflict);

  // one new tag for the cut set of this commit
  int max_tag = -1, gmax_tag = -1;
  for (int i = 0; i < dst->n_half_edges; ++i)
    if (mftlHeLive(dst, i) && MFTLHeIsBoundary(&dst->half_edges[i])) max_tag = MFTL_MAX(max_tag, MFTLHeBoundaryTag(&dst->half_edges[i]));
  MPI_Allreduce(&max_tag, &gmax_tag, 1, MPI_INT, MPI_MAX, comm);
  const int cut_tag = gmax_tag + 1;

  int printed = 0;
  for (int k = 0; k < n; ++k)
  {
    MFTL_HalfEdge* he = &dst->half_edges[self[k]];
    if (holder[k] >= 0)
    {
      if (!match[k])
      {
        bad++;
        if (printed++ < 20) fprintf(stderr, "[CommitCopy r%d] half-edge %d: partner %d does not name it back\n", dom->rank, he->global_id, want[k]);
      }
      if (conflict[k])
      {
        bad++;
        if (printed++ < 20) fprintf(stderr, "[CommitCopy r%d] half-edge %d: duplicate copies of partner %d differ\n", dom->rank, he->global_id, want[k]);
      }
      MFTLHeSetRemote(he, want[k]);
    }
    else if (mode == MFTL_COMMIT_CUT && want[k] >= 0 && dst->declared_open)
    {
      MFTLHeSetBoundary(he, cut_tag);  // the explicit cut set: an existing source twin not copied
    }
    else
    {
      bad++;
      if (printed++ < 20)
        fprintf(stderr, "[CommitCopy r%d] half-edge %d: twin %d unresolved (%s)\n", dom->rank, he->global_id, want[k],
                mode == MFTL_COMMIT_CUT ? (dst->declared_open ? "unknown source twin" : "CUT requires a body declared open") : "ASSEMBLE");
    }
  }
  free(want);
  free(code);
  free(self);
  free(holder);
  free(match);
  free(conflict);

  // duplicate authoritative copies of the SAME local half-edges are compared by signature above
  // for requested partners; the local structure is re-checked here.
  MFTLRefreshBoundaryCache(dst);
  MFTLFlattenConnectivity(dst);
  bad += MFTLCheckTwinStatesLocal(dst, 0, 1);
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, comm);
  if (gbad) return MFTL_ERROR;
  // Stage 16B (§3.3): copies/merges join independently gauged pieces -> canonicalise
  return MFTLCanonicaliseImages(dst);
}

/* ========================================================================================
 * ghost completion (G1/G2)
 * ======================================================================================== */
static int g_force_fallback = 0;
static MFTL_DiscoveryStats g_last_stats;

void MFTLDiscoverySetForceFallback(int force) { g_force_fallback = force; }
void MFTLGetLastDiscoveryStats(MFTL_DiscoveryStats* stats) { *stats = g_last_stats; }

// record: [PackedTriangleComplex][PackedVertex x3][payload]
static size_t mftlRecSize(const MFTL_Object* obj) { return sizeof(PackedTriangleComplex) + 3 * sizeof(PackedVertex) + obj->data_size; }

static void mftlPackRecord(MFTL_Object* obj, int t, char* out)
{
  PackedTriangleComplex* ptc = (PackedTriangleComplex*) out;
  memset(out, 0, mftlRecSize(obj));
  MFTLPackSingleTriangle(obj, t, ptc, MFTL_TRUE);
  PackedVertex* pv = (PackedVertex*) (out + sizeof(PackedTriangleComplex));
  int h = obj->triangles[t].he_idx;
  for (int k = 0; k < 3; ++k, h = obj->half_edges[h].next_idx)
  {
    MFTLPackSingleVertex(obj, obj->half_edges[h].vertex_idx, &pv[k], obj->vertices[obj->half_edges[h].vertex_idx].is_ghost);
  }
  if (obj->data_size > 0) memcpy(out + sizeof(PackedTriangleComplex) + 3 * sizeof(PackedVertex), obj->triangles[t].data, obj->data_size);
}

static double mftlPeriodicBoxDist2(const MFTL_Domain* dom, const double x[3], const double lo[3], const double hi[3])
{
  double d2 = 0.0;
  for (int d = 0; d < 3; ++d)
  {
    double best = 1e300;
    const int ns = dom->is_periodic[d] ? 3 : 1;
    for (int s = 0; s < ns; ++s)
    {
      const double shift = (s == 0) ? 0.0 : (s == 1 ? dom->domain_size[d] : -dom->domain_size[d]);
      const double xx = x[d] + shift;
      double dd = 0.0;
      if (xx < lo[d])
        dd = lo[d] - xx;
      else if (xx > hi[d])
        dd = xx - hi[d];
      if (dd < best) best = dd;
    }
    d2 += best * best;
  }
  return d2;
}

void MFTLInternalRankBox(const MFTL_Domain* dom, int rank, double bmin[3], double bmax[3]);

// Exchange byte records with MPI_Alltoallv; returns the received buffer and its size.
static char* mftlAlltoallBytes(MPI_Comm comm, int np, char** sendbufs, const int* sendsz, int* recv_total, int* recv_counts)
{
  int* scount = malloc((size_t) np * sizeof(int));
  int* rcount = malloc((size_t) np * sizeof(int));
  int* sdisp = malloc((size_t) np * sizeof(int));
  int* rdisp = malloc((size_t) np * sizeof(int));
  int st = 0;
  for (int p = 0; p < np; ++p)
  {
    scount[p] = sendsz[p];
    sdisp[p] = st;
    st += sendsz[p];
  }
  MPI_Alltoall(scount, 1, MPI_INT, rcount, 1, MPI_INT, comm);
  int rt = 0;
  for (int p = 0; p < np; ++p)
  {
    rdisp[p] = rt;
    rt += rcount[p];
  }
  char* sbuf = malloc((size_t) (st > 0 ? st : 1));
  for (int p = 0; p < np; ++p)
    if (sendsz[p] > 0) memcpy(sbuf + sdisp[p], sendbufs[p], (size_t) sendsz[p]);
  char* rbuf = malloc((size_t) (rt > 0 ? rt : 1));
  MPI_Alltoallv(sbuf, scount, sdisp, MPI_BYTE, rbuf, rcount, rdisp, MPI_BYTE, comm);
  if (recv_counts) memcpy(recv_counts, rcount, (size_t) np * sizeof(int));
  free(sbuf);
  free(scount);
  free(rcount);
  free(sdisp);
  free(rdisp);
  *recv_total = rt;
  return rbuf;
}

typedef struct
{
  char* buf;
  int size, cap;
} mftlByteBuf;

static void mftlBBAppend(mftlByteBuf* b, const void* p, int n)
{
  if (b->size + n > b->cap)
  {
    b->cap = MFTL_MAX(2 * b->cap, b->size + n + 256);
    b->buf = realloc(b->buf, (size_t) b->cap);
  }
  memcpy(b->buf + b->size, p, (size_t) n);
  b->size += n;
}

void MFTLInternalRemapAppendedGhostIndices(MFTL_Object* obj, int v_mark, int he_mark);

// Appends received records as ghosts (dedup by gid). Returns the number of new triangles.
static int mftlAppendRecords(MFTL_Object* obj, const char* buf, int nbytes)
{
  const size_t rs = mftlRecSize(obj);
  const int v_mark = obj->n_vertices, he_mark = obj->n_half_edges;
  int added = 0;
  for (size_t off = 0; off + rs <= (size_t) nbytes; off += rs)
  {
    PackedTriangleComplex* ptc = (PackedTriangleComplex*) (buf + off);
    khiter_t kt = kh_get(triangle_map, obj->triangle_lookup, ptc->triangle.global_id);
    if (kt != kh_end(obj->triangle_lookup)) continue;
    PackedVertex* pv = (PackedVertex*) (buf + off + sizeof(PackedTriangleComplex));
    for (int k = 0; k < 3; ++k)
    {
      khiter_t kv = kh_get(vertex_map, obj->vertex_lookup, pv[k].global_id);
      if (kv == kh_end(obj->vertex_lookup)) MFTLAppendGhostVertex(obj, &pv[k]);
    }
    MFTLAppendGhostTriangle(obj, ptc, buf + off + sizeof(PackedTriangleComplex) + 3 * sizeof(PackedVertex));
    added++;
  }
  if (added > 0)
  {
    MFTLInternalRemapAppendedGhostIndices(obj, v_mark, he_mark);
    // a pre-existing vertex whose outgoing half-edge was not stored gets one from the imports
    for (int i = he_mark; i < obj->n_half_edges; ++i)
    {
      const int v = obj->half_edges[i].vertex_idx;
      const int o = obj->vertices[v].out_he_idx;
      if (o < 0 || o >= obj->n_half_edges) obj->vertices[v].out_he_idx = obj->half_edges[i].next_idx;
    }
  }
  return added;
}

// holder test: live half-edge gid of a non-ghost triangle (owned_only: also owner_proc == me)
static int mftlHolderTri(const MFTL_Object* obj, int gid, int owned_only)
{
  khiter_t it = kh_get(halfedge_map, obj->halfedge_lookup, gid);
  if (it == kh_end(obj->halfedge_lookup)) return -1;
  const int h = kh_value(obj->halfedge_lookup, it);
  if (h < 0 || h >= obj->n_half_edges || obj->half_edges[h].global_id != gid || !mftlHeLive(obj, h)) return -1;
  const int t = obj->half_edges[h].triangle_idx;
  if (obj->triangles[t].is_ghost) return -1;
  if (owned_only && obj->triangles[t].owner_proc != obj->domain->rank) return -1;
  return t;
}

// a REMOTE half-edge whose twin must be stored: it belongs to an owned triangle (G1) or is
// incident to an owned vertex (G2)
static int mftlNeedsTwin(const MFTL_Object* obj, int i)
{
  if (!mftlHeLive(obj, i)) return 0;
  const MFTL_HalfEdge* he = &obj->half_edges[i];
  if (!MFTLHeIsRemote(he)) return 0;
  if (!obj->triangles[he->triangle_idx].is_ghost) return 1;
  return !obj->vertices[he->vertex_idx].is_ghost || !obj->vertices[mftlOriginIdx(obj, i)].is_ghost;
}

int MFTLInternalCountGhostRequests(const MFTL_Object* obj)
{
  int n = 0;
  for (int i = 0; i < obj->n_half_edges; ++i) n += mftlNeedsTwin(obj, i);
  return n;
}

int MFTLCompleteGhosts(MFTL_Object* obj, double eps, MFTL_DiscoveryStats* stats)
{
  MFTL_Domain* dom = obj->domain;
  MPI_Comm comm = dom->comm_cart;
  const int np = dom->nproc, me = dom->rank;
  MFTL_DiscoveryStats st;
  memset(&st, 0, sizeof(st));
  const size_t rs = mftlRecSize(obj);
  int rc = MFTL_SUCCESS;
  double lmax = -1.0;

  for (int round = 0; round < 64; ++round)
  {
    st.rounds++;
    // --- requests: REMOTE edges of owned triangles (G1) and incident to owned vertices (G2)
    int n = 0, cap = 64;
    int* req = malloc((size_t) cap * sizeof(int));  // half-edge index
    for (int i = 0; i < obj->n_half_edges; ++i)
    {
      if (!mftlNeedsTwin(obj, i)) continue;
      if (n == cap)
      {
        cap *= 2;
        req = realloc(req, (size_t) cap * sizeof(int));
      }
      req[n++] = i;
    }
    int total = 0;
    MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, comm);
    if (total == 0)
    {
      free(req);
      break;
    }
    st.n_requested += total;
    char* resolved = calloc((size_t) (n > 0 ? n : 1), 1);
    int n_new = 0;

    // --- round 1: owned mode, candidate ranks by geometry
    const int use_round1 = !g_force_fallback && dom->lb_nprocs == 0 && dom->rank_map != MFTL_NULL;
    int any_round1 = 0;
    MPI_Allreduce(&use_round1, &any_round1, 1, MPI_INT, MPI_MIN, comm);
    if (any_round1)
    {
      if (lmax < 0.0)
      {
        double loc = 0.0;
        for (int i = 0; i < obj->n_half_edges; ++i)
        {
          if (!mftlHeLive(obj, i)) continue;
          double e[3];
          MFTLGetMinimumImageVector(dom, e, obj->vertices[mftlOriginIdx(obj, i)].pos, obj->vertices[obj->half_edges[i].vertex_idx].pos);
          loc = fmax(loc, sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]));
        }
        MPI_Allreduce(&loc, &lmax, 1, MPI_DOUBLE, MPI_MAX, comm);
      }
      const double r = lmax / 3.0 + 2.0 * eps;
      const double r2 = (r * (1.0 + 1e-9) + 1e-12) * (r * (1.0 + 1e-9) + 1e-12);
      double(*boxes)[2][3] = malloc((size_t) np * sizeof(*boxes));
      for (int p = 0; p < np; ++p) MFTLInternalRankBox(dom, p, boxes[p][0], boxes[p][1]);
      mftlByteBuf* qb = calloc((size_t) np, sizeof(mftlByteBuf));
      for (int k = 0; k < n; ++k)
      {
        const int i = req[k];
        const double* a = obj->vertices[mftlOriginIdx(obj, i)].pos;
        const double* b = obj->vertices[obj->half_edges[i].vertex_idx].pos;
        double e[3], xm[3];
        MFTLGetMinimumImageVector(dom, e, a, b);
        for (int d = 0; d < 3; ++d) xm[d] = a[d] + 0.5 * e[d];
        for (int p = 0; p < np; ++p)
        {
          if (p == me) continue;
          if (mftlPeriodicBoxDist2(dom, xm, boxes[p][0], boxes[p][1]) <= r2) mftlBBAppend(&qb[p], &obj->half_edges[i].twin_gid, sizeof(int));
        }
      }
      free(boxes);
      char** sb = malloc((size_t) np * sizeof(char*));
      int* ss = malloc((size_t) np * sizeof(int));
      for (int p = 0; p < np; ++p)
      {
        sb[p] = qb[p].buf;
        ss[p] = qb[p].size;
      }
      // the request exchange (counts + gids); rcount gives the per-source request blocks
      int* rcount = malloc((size_t) np * sizeof(int));
      int rt = 0;
      char* rq = mftlAlltoallBytes(comm, np, sb, ss, &rt, rcount);
      for (int p = 0; p < np; ++p) free(qb[p].buf);
      free(qb);
      // answer: records for gids held as live half-edges of OWNED triangles
      mftlByteBuf* ab = calloc((size_t) np, sizeof(mftlByteBuf));
      char* rec = malloc(rs);
      int off = 0;
      for (int p = 0; p < np; ++p)
      {
        for (int q = 0; q < rcount[p]; q += (int) sizeof(int))
        {
          int gid;
          memcpy(&gid, rq + off + q, sizeof(int));
          const int t = mftlHolderTri(obj, gid, 1);
          if (t < 0) continue;
          mftlPackRecord(obj, t, rec);
          mftlBBAppend(&ab[p], rec, (int) rs);
        }
        off += rcount[p];
      }
      free(rq);
      for (int p = 0; p < np; ++p)
      {
        sb[p] = ab[p].buf;
        ss[p] = ab[p].size;
      }
      int nb = 0;
      char* rb = mftlAlltoallBytes(comm, np, sb, ss, &nb, MFTL_NULL);
      for (int p = 0; p < np; ++p) free(ab[p].buf);
      free(ab);
      free(sb);
      free(ss);
      free(rcount);
      free(rec);
      n_new += mftlAppendRecords(obj, rb, nb);
      free(rb);
      // resolution of round-1 requests: the partner half-edge is now stored
      long r1 = 0;
      for (int k = 0; k < n; ++k)
      {
        khiter_t it = kh_get(halfedge_map, obj->halfedge_lookup, obj->half_edges[req[k]].twin_gid);
        if (it != kh_end(obj->halfedge_lookup))
        {
          resolved[k] = 1;
          r1++;
        }
      }
      long g1 = 0;
      MPI_Allreduce(&r1, &g1, 1, MPI_LONG, MPI_SUM, comm);
      st.n_round1 += g1;
    }

    // --- round 2: global fallback (always executed; collective)
    {
      int nu = 0;
      for (int k = 0; k < n; ++k)
        if (!resolved[k]) nu++;
      int* send = malloc((size_t) (nu > 0 ? nu : 1) * sizeof(int));
      nu = 0;
      for (int k = 0; k < n; ++k)
        if (!resolved[k]) send[nu++] = obj->half_edges[req[k]].twin_gid;
      int gnu = 0;
      MPI_Allreduce(&nu, &gnu, 1, MPI_INT, MPI_SUM, comm);
      if (gnu > 0)
      {
        int *all = MFTL_NULL, *counts = MFTL_NULL, *displs = MFTL_NULL, nbytes = 0;
        mftlAllgathervBytes(comm, send, nu * (int) sizeof(int), (void**) &all, &nbytes, &counts, &displs);
        mftlByteBuf* ab = calloc((size_t) np, sizeof(mftlByteBuf));
        char* rec = malloc(rs);
        for (int p = 0; p < np; ++p)
        {
          if (p == me) continue;
          for (int q = 0; q < counts[p] / (int) sizeof(int); ++q)
          {
            const int gid = all[displs[p] / (int) sizeof(int) + q];
            const int t = mftlHolderTri(obj, gid, 0);
            if (t < 0) continue;
            mftlPackRecord(obj, t, rec);
            mftlBBAppend(&ab[p], rec, (int) rs);
          }
        }
        free(rec);
        free(all);
        free(counts);
        free(displs);
        char** sb = malloc((size_t) np * sizeof(char*));
        int* ss = malloc((size_t) np * sizeof(int));
        for (int p = 0; p < np; ++p)
        {
          sb[p] = ab[p].buf;
          ss[p] = ab[p].size;
        }
        int nb = 0;
        char* rb = mftlAlltoallBytes(comm, np, sb, ss, &nb, MFTL_NULL);
        for (int p = 0; p < np; ++p) free(ab[p].buf);
        free(ab);
        free(sb);
        free(ss);
        n_new += mftlAppendRecords(obj, rb, nb);
        free(rb);
        long rf = 0, ru = 0;
        for (int k = 0; k < n; ++k)
        {
          if (resolved[k]) continue;
          khiter_t it = kh_get(halfedge_map, obj->halfedge_lookup, obj->half_edges[req[k]].twin_gid);
          if (it != kh_end(obj->halfedge_lookup))
          {
            resolved[k] = 1;
            rf++;
          }
          else
          {
            ru++;
            if (ru <= 10)
              fprintf(stderr, "[CompleteGhosts r%d] body %d: half-edge %d has no live authoritative twin %d on any rank\n", me, obj->global_id,
                      obj->half_edges[req[k]].global_id, obj->half_edges[req[k]].twin_gid);
          }
        }
        long g[2] = {rf, ru}, gg[2];
        MPI_Allreduce(g, gg, 2, MPI_LONG, MPI_SUM, comm);
        st.n_fallback += gg[0];
        st.n_unresolved += gg[1];
      }
      free(send);
    }
    free(resolved);
    free(req);

    if (n_new > 0)
    {
      MFTLRebuildTwins(obj);
      MFTLFlattenConnectivity(obj);
    }
    if (st.n_unresolved > 0)
    {
      rc = MFTL_ERROR;
      break;
    }
    int gnew = 0;
    MPI_Allreduce(&n_new, &gnew, 1, MPI_INT, MPI_SUM, comm);
    if (gnew == 0)
    {
      // requests existed but nothing could be imported: they are unresolvable
      rc = MFTL_ERROR;
      break;
    }
  }
  g_last_stats = st;
  if (stats) *stats = st;
  if (getenv("MFTL_DEBUG_DISCOVERY") && me == 0 && st.n_requested > 0)
    printf("[CompleteGhosts] body %d: rounds %d requested %ld round1 %ld fallback %ld unresolved %ld\n", obj->global_id, st.rounds, st.n_requested,
           st.n_round1, st.n_fallback, st.n_unresolved);
  return rc;
}

/* ========================================================================================
 * boundary loops
 * ======================================================================================== */
typedef struct
{
  int in_eid, out_eid, vgid, tag;
  int s[3];  // Stage 16B: lattice shift of the outgoing boundary edge
  int pad;
  double len;
} mftlLoopRec;

static int mftlCmpLoopIn(const void* a, const void* b)
{
  const mftlLoopRec* x = (const mftlLoopRec*) a;
  const mftlLoopRec* y = (const mftlLoopRec*) b;
  return (x->in_eid > y->in_eid) - (x->in_eid < y->in_eid);
}

void MFTLFreeBoundaryLoops(MFTL_BoundaryLoopSet* set)
{
  for (int k = 0; k < set->n_loops; ++k)
  {
    free(set->loops[k].edge_ids);
    free(set->loops[k].vertex_gids);
  }
  free(set->loops);
  set->loops = MFTL_NULL;
  set->n_loops = 0;
}

int MFTLBoundaryLoops(MFTL_Object* obj, MFTL_BoundaryLoopSet* out)
{
  MFTL_Domain* dom = obj->domain;
  MPI_Comm comm = dom->comm_cart;
  out->n_loops = 0;
  out->loops = MFTL_NULL;
  int bad = 0, n = 0, cap = 16;
  mftlLoopRec* rec = malloc((size_t) cap * sizeof(mftlLoopRec));
  int* ring = malloc(1024 * sizeof(int));
  char* touches_b = calloc((size_t) (obj->n_vertices > 0 ? obj->n_vertices : 1), 1);
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (mftlHeLive(obj, i) && MFTLHeIsBoundary(&obj->half_edges[i]))
    {
      touches_b[mftlOriginIdx(obj, i)] = 1;
      touches_b[obj->half_edges[i].vertex_idx] = 1;
    }
  for (int v = 0; v < obj->n_vertices; ++v)
  {
    const MFTL_Vertex* vx = &obj->vertices[v];
    if (vx->is_ghost || (vx->flags & MFTL_FLAG_DEAD) || vx->out_he_idx < 0) continue;
    int nr = 0;
    const int cls = MFTLVertexRing(obj, v, ring, 1024, &nr);
    if (cls == MFTL_RING_INCOMPLETE)
    {
      // an owned vertex touching a BOUNDARY half-edge must have a complete fan here (G2)
      if (touches_b[v]) bad++;
      continue;
    }
    if (cls != MFTL_RING_OPEN_COMPLETE) continue;
    const int out_h = ring[nr - 1];
    const int in_h = obj->half_edges[obj->half_edges[ring[0]].next_idx].next_idx;
    if (n == cap)
    {
      cap *= 2;
      rec = realloc(rec, (size_t) cap * sizeof(mftlLoopRec));
    }
    mftlLoopRec* r = &rec[n++];
    r->in_eid = MFTLHeEdgeId(obj, &obj->half_edges[in_h]);
    r->out_eid = MFTLHeEdgeId(obj, &obj->half_edges[out_h]);
    r->vgid = vx->global_id;
    r->tag = MFTLHeBoundaryTag(&obj->half_edges[out_h]);
    double e[3];
    MFTLGetMinimumImageVector(dom, e, vx->pos, obj->vertices[obj->half_edges[out_h].vertex_idx].pos);
    r->len = sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    MFTLEdgeShift(dom, vx, &obj->vertices[obj->half_edges[out_h].vertex_idx], r->s);
    r->pad = 0;
  }
  free(ring);
  free(touches_b);
  mftlLoopRec* all = MFTL_NULL;
  int nb = 0;
  mftlAllgathervBytes(comm, rec, n * (int) sizeof(mftlLoopRec), (void**) &all, &nb, MFTL_NULL, MFTL_NULL);
  free(rec);
  const int N = nb / (int) sizeof(mftlLoopRec);
  qsort(all, (size_t) N, sizeof(mftlLoopRec), mftlCmpLoopIn);
  for (int i = 1; i < N; ++i)
    if (all[i].in_eid == all[i - 1].in_eid) bad++;  // two vertices claim one incoming edge
  int* succ = malloc((size_t) (N > 0 ? N : 1) * sizeof(int));
  for (int i = 0; i < N; ++i)
  {
    mftlLoopRec key;
    memset(&key, 0, sizeof(key));
    key.in_eid = all[i].out_eid;
    mftlLoopRec* f = bsearch(&key, all, (size_t) N, sizeof(mftlLoopRec), mftlCmpLoopIn);
    succ[i] = f ? (int) (f - all) : -1;
    if (!f) bad++;
  }
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, comm);
  if (gbad)
  {
    free(all);
    free(succ);
    if (dom->rank == 0) fprintf(stderr, "[BoundaryLoops] body %d: broken boundary chain (%d)\n", obj->global_id, gbad);
    return MFTL_ERROR;
  }
  char* seen = calloc((size_t) (N > 0 ? N : 1), 1);
  int lcap = 4;
  out->loops = malloc((size_t) lcap * sizeof(MFTL_BoundaryLoop));
  // loops in order of their smallest vertex gid: visit records by increasing vertex gid
  int* order = malloc((size_t) (N > 0 ? N : 1) * sizeof(int) * 2);
  for (int i = 0; i < N; ++i)
  {
    order[2 * i] = all[i].vgid;
    order[2 * i + 1] = i;
  }
  qsort(order, (size_t) N, 2 * sizeof(int), mftlCmpInt2);
  for (int oi = 0; oi < N; ++oi)
  {
    const int s = order[2 * oi + 1];
    if (seen[s]) continue;
    if (out->n_loops == lcap)
    {
      lcap *= 2;
      out->loops = realloc(out->loops, (size_t) lcap * sizeof(MFTL_BoundaryLoop));
    }
    MFTL_BoundaryLoop* L = &out->loops[out->n_loops++];
    memset(L, 0, sizeof(*L));
    int len = 0, r = s;
    do
    {
      seen[r] = 1;
      len++;
      r = succ[r];
    } while (r != s && !seen[r]);
    L->n = len;
    L->edge_ids = malloc((size_t) len * sizeof(int));
    L->vertex_gids = malloc((size_t) len * sizeof(int));
    L->tag = all[s].tag;
    r = s;
    for (int k = 0; k < len; ++k, r = succ[r])
    {
      L->vertex_gids[k] = all[r].vgid;
      L->edge_ids[k] = all[r].out_eid;
      L->length += all[r].len;
      for (int k = 0; k < 3; ++k) L->winding[k] += all[r].s[k];
      if (all[r].tag != L->tag) L->mixed_tags = 1;
      if (all[r].tag < L->tag) L->tag = all[r].tag;
    }
  }
  free(order);
  free(seen);
  free(all);
  free(succ);
  return MFTL_SUCCESS;
}
