/*
 * Stage 16A — topology and twin-state invariant (plans/16-generalized-surfaces.md §2, tests
 * A1–A4, A6, A7 of §10 and the 16A gate):
 *   - three-state check (LOCAL / REMOTE / BOUNDARY) after every tested consistent point, with
 *     the full collective MFTLValidateTwinStatesEx (remote liveness, G1/G2 after redistribution);
 *   - closed-manifold regression (sphere), open patches with one and two boundary loops;
 *   - builder rejections (Mobius, bow-tie, non-manifold edge, closed body with a hole, open body
 *     without a declaration) and accepted declarations (BoundaryFromMesh, explicit edge list);
 *   - builder before ownership (soup with random placement and duplicates);
 *   - copy/commit ASSEMBLE and CUT (explicit cut set, one new tag);
 *   - remote-twin discovery: owned-mode round 1 for fresh positions, forced and stale-position
 *     fallback, stale/dead twin gid detected;
 *   - boundary-adjacent split / collapse / flip;
 *   - redistribution + ghost completion G1/G2, exactly-once ownership, 1- and 2-slab periodic
 *     neighbour layouts;
 *   - randomized invariant fuzz (jitter + remesh + redistribution), closed and open.
 * Every CHECK is collective. Registered at 1, 2, 4 and 8 ranks.
 * Environment: S16_FUZZ_STEPS (default 12) sets the fuzz length.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"

static int g_rank = 0, g_nproc = 1, g_failed = 0, g_checks = 0;

#define CHECK(cond)                                                                                       \
  do                                                                                                      \
  {                                                                                                       \
    int _l = (cond) ? 0 : 1, _g = 0;                                                                      \
    MPI_Allreduce(&_l, &_g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);                                         \
    g_checks++;                                                                                           \
    if (_g)                                                                                               \
    {                                                                                                     \
      if (_l) fprintf(stderr, "[rank %d] CHECK failed: %s (%s:%d)\n", g_rank, #cond, __FILE__, __LINE__); \
      g_failed++;                                                                                         \
    }                                                                                                     \
  } while (0)

static const double GMIN[3] = {-1.5, -1.5, -1.5}, GMAX[3] = {1.5, 1.5, 1.5};

static void grid3(int np, int g[3])
{
  g[0] = np;
  g[1] = 1;
  g[2] = 1;
  if (np == 4)
  {
    g[0] = 2;
    g[1] = 2;
  }
  if (np == 8)
  {
    g[0] = 2;
    g[1] = 2;
    g[2] = 2;
  }
}

static MFTL_Ensemble* make_ens(const int g[3], const bool per[3])
{
  int cuts[3] = {g[0], g[1], g[2]};
  double mn[3] = {GMIN[0], GMIN[1], GMIN[2]}, mx[3] = {GMAX[0], GMAX[1], GMAX[2]};
  bool p[3] = {per[0], per[1], per[2]};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, mn, mx, p, 4);
}

static MFTL_Ensemble* make_default_ens(void)
{
  int g[3];
  grid3(g_nproc, g);
  bool per[3] = {false, false, false};
  return make_ens(g, per);
}

/* --------------------------------------------------------------------------------------- */
/* invariant helpers                                                                        */
/* --------------------------------------------------------------------------------------- */
static int valid(MFTL_Object* obj, int ghosts) { return MFTLValidateTwinStatesEx(obj, ghosts ? MFTL_VALIDATE_GHOSTS : 0) == MFTL_SUCCESS; }

static int cmp_int(const void* a, const void* b)
{
  int x = *(const int*) a, y = *(const int*) b;
  return (x > y) - (x < y);
}

// gathers the gids of the local (non-ghost) entities; returns duplicates across ranks
static int gather_dups(int n, const int* gids, int* total_out)
{
  int* counts = malloc((size_t) g_nproc * sizeof(int));
  int* displs = malloc((size_t) g_nproc * sizeof(int));
  MPI_Allgather(&n, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int p = 0; p < g_nproc; ++p)
  {
    displs[p] = tot;
    tot += counts[p];
  }
  int* all = malloc((size_t) (tot > 0 ? tot : 1) * sizeof(int));
  MPI_Allgatherv(gids, n, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD);
  qsort(all, (size_t) tot, sizeof(int), cmp_int);
  int dup = 0;
  for (int i = 1; i < tot; ++i)
    if (all[i] == all[i - 1]) dup++;
  free(all);
  free(counts);
  free(displs);
  if (total_out) *total_out = tot;
  return dup;
}

// exactly-once ownership of triangles and vertices; every referenced vertex owned somewhere
static int exactly_once(MFTL_Object* obj, int expect_tris)
{
  int bad = 0, nt = 0, nv = 0;
  int* tg = malloc((size_t) (obj->n_triangles + 1) * sizeof(int));
  int* vg = malloc((size_t) (obj->n_vertices + 1) * sizeof(int));
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    if (obj->triangles[t].is_ghost) continue;
    if (obj->triangles[t].owner_proc != g_rank) bad++;
    tg[nt++] = obj->triangles[t].global_id;
  }
  for (int v = 0; v < obj->n_vertices; ++v)
  {
    if (obj->vertices[v].is_ghost) continue;
    if (obj->vertices[v].owner_proc != g_rank) bad++;
    vg[nv++] = obj->vertices[v].global_id;
  }
  int tot_t = 0, tot_v = 0;
  bad += gather_dups(nt, tg, &tot_t);
  bad += gather_dups(nv, vg, &tot_v);
  if (expect_tris >= 0 && tot_t != expect_tris) bad++;
  // every vertex referenced by an owned triangle is owned by exactly one rank: compare the
  // number of distinct referenced vertices with the number of owned vertices
  int* rv = malloc((size_t) (3 * nt + 1) * sizeof(int));
  int nr = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    if (obj->triangles[t].is_ghost) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[t], vv);
    for (int k = 0; k < 3; ++k) rv[nr++] = vv[k]->global_id;
  }
  int* counts = malloc((size_t) g_nproc * sizeof(int));
  int* displs = malloc((size_t) g_nproc * sizeof(int));
  MPI_Allgather(&nr, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int p = 0; p < g_nproc; ++p)
  {
    displs[p] = tot;
    tot += counts[p];
  }
  int* all = malloc((size_t) (tot > 0 ? tot : 1) * sizeof(int));
  MPI_Allgatherv(rv, nr, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD);
  qsort(all, (size_t) tot, sizeof(int), cmp_int);
  int distinct = 0;
  for (int i = 0; i < tot; ++i)
    if (i == 0 || all[i] != all[i - 1]) distinct++;
  if (distinct != tot_v) bad++;
  free(all);
  free(counts);
  free(displs);
  free(rv);
  free(tg);
  free(vg);
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (gbad && getenv("S16_VERBOSE"))
  {
    int wrong = 0;
    for (int t = 0; t < obj->n_triangles; ++t)
      if (!obj->triangles[t].is_ghost && obj->triangles[t].owner_proc != g_rank) wrong++;
    printf("  [r%d] local tris %d (n_local %d), non-ghost with foreign owner %d\n", g_rank, nt, obj->n_local_triangles, wrong);
  }
  if (gbad && getenv("S16_VERBOSE") && g_rank == 0)
    printf("  exactly_once: tris %d (expect %d), owned verts %d, referenced verts %d\n", tot_t, expect_tris, tot_v, distinct);
  return gbad;
}

static int global_owned_tris(MFTL_Object* obj)
{
  int n = 0, gn = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (!obj->triangles[t].is_ghost) n++;
  MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return gn;
}

// BOUNDARY half-edges of owned triangles (each boundary edge counted once globally)
static int global_boundary_edges(MFTL_Object* obj)
{
  int n = 0, gn = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    if (obj->triangles[t].is_ghost) continue;
    MFTL_HalfEdge* h[3];
    MFTLGetTriangleHalfedges(obj, &obj->triangles[t], h);
    for (int k = 0; k < 3; ++k)
      if (MFTLHeIsBoundary(h[k])) n++;
  }
  MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return gn;
}

// A7: boundary edges of owned triangles have distinct edge ids equal to their gid
static int boundary_edge_id_violations(MFTL_Object* obj)
{
  int n = 0, bad = 0;
  int* ids = malloc((size_t) (3 * obj->n_triangles + 1) * sizeof(int));
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    if (obj->triangles[t].is_ghost) continue;
    MFTL_HalfEdge* h[3];
    MFTLGetTriangleHalfedges(obj, &obj->triangles[t], h);
    for (int k = 0; k < 3; ++k)
    {
      if (!MFTLHeIsBoundary(h[k])) continue;
      const int id = MFTLHeEdgeId(obj, h[k]);
      if (id != h[k]->global_id || id < 0) bad++;
      if (MFTLHeTwinGid(obj, h[k]) != MFTL_ERROR) bad++;  // a boundary code is never a gid
      ids[n++] = id;
    }
  }
  bad += gather_dups(n, ids, MFTL_NULL);
  free(ids);
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return gbad;
}

// loops: count, sum of sizes == boundary edges, consistent tags; returns the loop count
static int check_loops_ex(MFTL_Object* obj, int expect_loops, double* lengths_out, int allow_mixed)
{
  MFTL_BoundaryLoopSet L;
  int rc = MFTLBoundaryLoops(obj, &L);
  CHECK(rc == MFTL_SUCCESS);
  if (rc != MFTL_SUCCESS) return -1;
  int sum = 0, mixed = 0;
  for (int k = 0; k < L.n_loops; ++k)
  {
    sum += L.loops[k].n;
    mixed += L.loops[k].mixed_tags;
    if (lengths_out) lengths_out[k] = L.loops[k].length;
  }
  CHECK(sum == global_boundary_edges(obj));
  CHECK(allow_mixed || mixed == 0);
  if (expect_loops >= 0) CHECK(L.n_loops == expect_loops);
  // identical on every rank: compare a checksum of the vertex sequences
  unsigned long cs = 0;  // unsigned: the checksum wraps by design (no signed overflow)
  for (int k = 0; k < L.n_loops; ++k)
    for (int i = 0; i < L.loops[k].n; ++i) cs = cs * 31u + (unsigned long) L.loops[k].vertex_gids[i] + 7u * (unsigned long) L.loops[k].tag;
  unsigned long mn = 0, mx = 0;
  MPI_Allreduce(&cs, &mn, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&cs, &mx, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
  CHECK(mn == mx);
  const int n = L.n_loops;
  MFTLFreeBoundaryLoops(&L);
  return n;
}

static int check_loops(MFTL_Object* obj, int expect_loops, double* lengths_out) { return check_loops_ex(obj, expect_loops, lengths_out, 0); }

// remove every ghost triangle (and its half-edges) — creates G1/G2 holes for discovery tests
static void strip_ghost_triangles(MFTL_Object* obj)
{
  const int nt = obj->n_local_triangles;
  for (int t = nt; t < obj->n_triangles; ++t)
  {
    khiter_t k = kh_get(triangle_map, obj->triangle_lookup, obj->triangles[t].global_id);
    if (k != kh_end(obj->triangle_lookup)) kh_del(triangle_map, obj->triangle_lookup, k);
    int h = obj->triangles[t].he_idx;
    for (int j = 0; j < 3; ++j, h = obj->half_edges[h].next_idx)
    {
      khiter_t kh = kh_get(halfedge_map, obj->halfedge_lookup, obj->half_edges[h].global_id);
      if (kh != kh_end(obj->halfedge_lookup)) kh_del(halfedge_map, obj->halfedge_lookup, kh);
    }
    if (obj->data_size > 0) free(obj->triangles[t].data);
  }
  obj->n_triangles = nt;
  obj->n_half_edges = obj->n_local_half_edges;
  for (int v = 0; v < obj->n_vertices; ++v)
    if (obj->vertices[v].out_he_idx >= obj->n_half_edges) obj->vertices[v].out_he_idx = MFTL_ERROR;
  // vertices keep an outgoing half-edge from the remaining triangles where possible
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    const int nx = obj->half_edges[i].next_idx;
    const int o = obj->half_edges[obj->half_edges[nx].next_idx].vertex_idx;
    if (obj->vertices[o].out_he_idx == MFTL_ERROR) obj->vertices[o].out_he_idx = i;
  }
  MFTLRebuildTwins(obj);
  MFTLFlattenConnectivity(obj);
}

/* --------------------------------------------------------------------------------------- */
/* meshes                                                                                   */
/* --------------------------------------------------------------------------------------- */
static double g_sphere_radius = 1.0;

static MFTL_Object* make_sphere(MFTL_Ensemble* ens, int gid, const double c[3], int refine)
{
  MFTL_Object* obj = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(obj, cc, g_sphere_radius, refine);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistribute(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  return obj;
}

// curved N x N grid patch over [-1.2,1.2]^2 (optionally with a square hole) — triangle list
typedef struct
{
  int nv, nt;
  double (*pos)[3];
  int (*tri)[3];
  int n_outer, n_inner;  // boundary edges of the outer / inner loop
} Patch;

static Patch make_patch(int N, int hole)
{
  Patch P;
  memset(&P, 0, sizeof(P));
  P.nv = N * N;
  P.pos = malloc((size_t) P.nv * sizeof(*P.pos));
  P.tri = malloc((size_t) (2 * (N - 1) * (N - 1)) * sizeof(*P.tri));
  const double h = 2.4 / (N - 1);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      double x = -1.2 + i * h, y = -1.2 + j * h;
      P.pos[j * N + i][0] = x;
      P.pos[j * N + i][1] = y;
      P.pos[j * N + i][2] = 0.1 * sin(1.3 * x) * cos(0.9 * y) + 0.013;
    }
  const int lo = N / 2 - 1, hi = N / 2 + 1;  // hole: cells [lo,hi) x [lo,hi)
  for (int j = 0; j < N - 1; ++j)
    for (int i = 0; i < N - 1; ++i)
    {
      if (hole && i >= lo && i < hi && j >= lo && j < hi) continue;
      int a = j * N + i, b = j * N + i + 1, c = (j + 1) * N + i + 1, d = (j + 1) * N + i;
      if ((i + j) % 2 == 0)
      {
        P.tri[P.nt][0] = a, P.tri[P.nt][1] = b, P.tri[P.nt][2] = c, P.nt++;
        P.tri[P.nt][0] = a, P.tri[P.nt][1] = c, P.tri[P.nt][2] = d, P.nt++;
      }
      else
      {
        P.tri[P.nt][0] = a, P.tri[P.nt][1] = b, P.tri[P.nt][2] = d, P.nt++;
        P.tri[P.nt][0] = b, P.tri[P.nt][1] = c, P.tri[P.nt][2] = d, P.nt++;
      }
    }
  // drop vertices no triangle references (the vertex inside the hole)
  int* map = malloc((size_t) P.nv * sizeof(int));
  for (int i = 0; i < P.nv; ++i) map[i] = -1;
  for (int t = 0; t < P.nt; ++t)
    for (int k = 0; k < 3; ++k) map[P.tri[t][k]] = 0;
  int nk = 0;
  for (int i = 0; i < P.nv; ++i)
    if (map[i] == 0)
    {
      memcpy(P.pos[nk], P.pos[i], sizeof(P.pos[0]));
      map[i] = nk++;
    }
  for (int t = 0; t < P.nt; ++t)
    for (int k = 0; k < 3; ++k) P.tri[t][k] = map[P.tri[t][k]];
  P.nv = nk;
  free(map);
  P.n_outer = 4 * (N - 1);
  P.n_inner = hole ? 4 * (hi - lo) : 0;
  return P;
}

static void free_patch(Patch* P)
{
  free(P->pos);
  free(P->tri);
}

// open patch built on rank 0 (serial builder), then redistributed
static MFTL_Object* make_open_patch(MFTL_Ensemble* ens, int gid, int N, int hole, int* rc_out)
{
  MFTL_Object* obj = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, obj);
  MFTLObjectDeclareOpen(obj, 1);
  Patch P = make_patch(N, hole);
  int rc = MFTL_SUCCESS;
  if (g_rank == 0) rc = MFTLBuildTopologyFromTriples(obj, P.nv, (const double (*)[3]) P.pos, P.nt, (const int (*)[3]) P.tri);
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&obj->next_vertex_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&obj->next_triangle_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&obj->next_halfedge_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free_patch(&P);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistribute(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  if (rc_out) *rc_out = rc;
  return obj;
}

// soup of the patch: triangle t on ranks hash(t) and (for every third one) hash(t)+1
static int build_patch_soup(MFTL_Object* obj, const Patch* P, const MFTL_BoundaryDecl* decl)
{
  int* vg = malloc((size_t) P->nv * sizeof(int));
  for (int i = 0; i < P->nv; ++i) vg[i] = 100 + 2 * i;
  int* tg = malloc((size_t) (P->nt + 1) * sizeof(int));
  int (*tv)[3] = malloc((size_t) (P->nt + 1) * sizeof(*tv));
  int* tw = malloc((size_t) (P->nt + 1) * sizeof(int));
  int n = 0;
  for (int t = 0; t < P->nt; ++t)
  {
    const unsigned hsh = (unsigned) (t * 2654435761u) >> 7;
    const int r0 = (int) (hsh % (unsigned) g_nproc);
    const int dup = (t % 3 == 0) && g_nproc > 1;
    if (r0 != g_rank && !(dup && (r0 + 1) % g_nproc == g_rank)) continue;
    tg[n] = 7 + 5 * t;
    for (int k = 0; k < 3; ++k) tv[n][k] = vg[P->tri[t][k]];
    tw[n] = (int) ((hsh / 7) % (unsigned) g_nproc);  // arbitrary owner label (not the holder)
    n++;
  }
  int rc = MFTLBuildFromSoupEx(obj, P->nv, vg, (const double (*)[3]) P->pos, MFTL_NULL, n, tg, (const int (*)[3]) tv, tw, decl);
  // before ownership the owner labels are arbitrary: centres of EVERY stored triangle
  for (int t = 0; t < obj->n_triangles; ++t)
    MFTLGetTriangleProperties(obj, &obj->triangles[t], &obj->triangles[t].area, obj->triangles[t].normal, obj->triangles[t].center);
  free(vg);
  free(tg);
  free(tv);
  free(tw);
  return rc;
}

/* --------------------------------------------------------------------------------------- */
/* tests                                                                                    */
/* --------------------------------------------------------------------------------------- */
static void test_closed_sphere(void)
{
  MFTL_Ensemble* ens = make_default_ens();
  double c[3] = {0.05, -0.03, 0.02};
  MFTL_Object* obj = make_sphere(ens, 1, c, 3);
  CHECK(valid(obj, 1));
  CHECK(exactly_once(obj, 20 * 64) == 0);
  CHECK(global_boundary_edges(obj) == 0);
  CHECK(check_loops(obj, 0, MFTL_NULL) == 0);
  // no BOUNDARY anywhere, no UNKNOWN; REMOTE only on ghost triangles (G1)
  int bad = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (MFTLHeIsBoundary(he) || MFTLHeIsUnresolved(he)) bad++;
    if (MFTLHeIsRemote(he) && !obj->triangles[he->triangle_idx].is_ghost) bad++;
  }
  CHECK(bad == 0);
  // closed-mesh edge id is min(gid, twin gid), unchanged (Stage 15)
  int bad_id = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    const int tg = MFTLHeTwinGid(obj, he);
    if (tg < 0 || MFTLHeEdgeId(obj, he) != (tg < he->global_id ? tg : he->global_id)) bad_id++;
  }
  CHECK(bad_id == 0);
  // stale / dead twin gid is detected (A3)
  int victim = -1;
  for (int i = 0; i < obj->n_half_edges && victim < 0; ++i)
    if (MFTLHeIsRemote(&obj->half_edges[i])) victim = i;
  int saved = victim >= 0 ? obj->half_edges[victim].twin_gid : 0;
  if (victim >= 0) obj->half_edges[victim].twin_gid = 1 << 29;  // non-existent gid
  int any = victim >= 0, gany = 0;
  MPI_Allreduce(&any, &gany, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (gany)
  {
    if (g_rank == 0) fprintf(stderr, "(expected diagnostics follow: stale twin gid)\n");
    CHECK(!valid(obj, 0));
  }
  if (victim >= 0) obj->half_edges[victim].twin_gid = saved;
  CHECK(valid(obj, 1));
  // an UNKNOWN state is detected
  if (obj->n_half_edges > 0 && g_rank == 0)
  {
    obj->half_edges[0].twin_idx = MFTL_ERROR;
    obj->half_edges[0].twin_gid = MFTL_TWIN_UNKNOWN;
    fprintf(stderr, "(expected diagnostics follow: unknown state)\n");
  }
  CHECK(!valid(obj, 0));
  MFTLRebuildTwins(obj);  // re-pairs locally
  MFTLEnsembleDestroy(&ens);
}

static void test_open_patch(void)
{
  for (int hole = 0; hole <= 1; ++hole)
  {
    MFTL_Ensemble* ens = make_default_ens();
    int rc = 0;
    const int N = 13;
    MFTL_Object* obj = make_open_patch(ens, 2, N, hole, &rc);
    CHECK(rc == MFTL_SUCCESS);
    Patch P = make_patch(N, hole);
    CHECK(global_owned_tris(obj) == P.nt);
    CHECK(valid(obj, 1));
    CHECK(exactly_once(obj, P.nt) == 0);
    CHECK(global_boundary_edges(obj) == P.n_outer + P.n_inner);
    double len[4] = {0, 0, 0, 0};
    CHECK(check_loops(obj, hole ? 2 : 1, len) == (hole ? 2 : 1));
    // loops are ordered by their smallest vertex gid: the outer loop (vertex 0) first; its
    // length is the sum of its edges (the curved border)
    CHECK(len[0] > 4 * 2.4 * 0.999 && len[0] < 4 * 2.4 * 1.05);
    CHECK(boundary_edge_id_violations(obj) == 0);
    // tags: 0 for the outer loop, 1 for the hole
    int badtag = 0;
    for (int i = 0; i < obj->n_half_edges; ++i)
      if (MFTLHeIsBoundary(&obj->half_edges[i]) && (MFTLHeBoundaryTag(&obj->half_edges[i]) < 0 || MFTLHeBoundaryTag(&obj->half_edges[i]) > hole)) badtag++;
    CHECK(badtag == 0);
    // re-decompose (a second redistribution to a different layout) keeps everything
    int g2[3] = {1, g_nproc, 1};
    bool per[3] = {false, false, false};
    CHECK(MFTLEnsembleSetDecomposition(ens, g2, MFTL_NULL, MFTL_NULL, GMIN, GMAX, per) == MFTL_SUCCESS);
    MFTLRedistribute(ens);
    CHECK(valid(obj, 1));
    CHECK(exactly_once(obj, P.nt) == 0);
    CHECK(check_loops(obj, hole ? 2 : 1, MFTL_NULL) == (hole ? 2 : 1));
    free_patch(&P);
    MFTLEnsembleDestroy(&ens);
  }
}

static int serial_build(int declare_open, int nv, const double (*pos)[3], int nt, const int (*tri)[3])
{
  MFTL_Ensemble* ens = make_default_ens();
  MFTL_Object* obj = MFTLObjectCreate(3);
  MFTLEnsembleAddObject(ens, obj);
  MFTLObjectDeclareOpen(obj, declare_open);
  int rc = MFTL_SUCCESS;
  if (g_rank == 0) rc = MFTLBuildTopologyFromTriples(obj, nv, pos, nt, tri);
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MFTLEnsembleDestroy(&ens);
  return rc;
}

static void test_builder_rejections(void)
{
  if (g_rank == 0) fprintf(stderr, "(expected builder diagnostics follow)\n");
  // Mobius strip (n = 6 quads, half twist)
  {
    const int n = 6;
    double pos[12][3];
    int tri[12][3], nt = 0;
    for (int i = 0; i < n; ++i)
    {
      double a = 2.0 * M_PI * i / n;
      pos[i][0] = cos(a), pos[i][1] = sin(a), pos[i][2] = 0.2;  // rail a_i
      pos[n + i][0] = cos(a), pos[n + i][1] = sin(a), pos[n + i][2] = -0.2;  // rail b_i
    }
    for (int i = 0; i < n; ++i)
    {
      int p, q, r, s;
      if (i < n - 1)
        p = i, q = i + 1, r = n + i + 1, s = n + i;
      else
        p = n - 1, q = n + 0, r = 0, s = 2 * n - 1;  // twisted closing quad
      tri[nt][0] = p, tri[nt][1] = q, tri[nt][2] = r, nt++;
      tri[nt][0] = p, tri[nt][1] = r, tri[nt][2] = s, nt++;
    }
    CHECK(serial_build(1, 2 * n, (const double (*)[3]) pos, nt, (const int (*)[3]) tri) == MFTL_ERROR);
  }
  // bow-tie: two triangles sharing only vertex 0
  {
    double pos[5][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {-1, 0, 0}, {-1, -1, 0}};
    int tri[2][3] = {{0, 1, 2}, {0, 3, 4}};
    CHECK(serial_build(1, 5, (const double (*)[3]) pos, 2, (const int (*)[3]) tri) == MFTL_ERROR);
  }
  // non-manifold edge 0-1 with three triangles
  {
    double pos[5][3] = {{0, 0, 0}, {1, 0, 0}, {0.5, 1, 0}, {0.5, -1, 0}, {0.5, 0, 1}};
    int tri[3][3] = {{0, 1, 2}, {1, 0, 3}, {0, 1, 4}};
    CHECK(serial_build(1, 5, (const double (*)[3]) pos, 3, (const int (*)[3]) tri) == MFTL_ERROR);
  }
  // declared closed with a hole / declared open accepted
  {
    Patch P = make_patch(7, 1);
    CHECK(serial_build(0, P.nv, (const double (*)[3]) P.pos, P.nt, (const int (*)[3]) P.tri) == MFTL_ERROR);
    CHECK(serial_build(1, P.nv, (const double (*)[3]) P.pos, P.nt, (const int (*)[3]) P.tri) == MFTL_SUCCESS);
    free_patch(&P);
  }
  if (g_rank == 0) fprintf(stderr, "(end of expected builder diagnostics)\n");
}

static void test_soup_before_ownership(void)
{
  const int N = 11;
  Patch P = make_patch(N, 1);
  // A6: an open soup without a declaration (or on a closed body) is a collective error
  if (g_rank == 0) fprintf(stderr, "(expected soup diagnostics follow)\n");
  for (int mode = 0; mode < 2; ++mode)
  {
    MFTL_Ensemble* ens = make_default_ens();
    MFTL_Object* obj = MFTLObjectCreate(4);
    MFTLEnsembleAddObject(ens, obj);
    MFTLObjectDeclareOpen(obj, mode);  // mode 0: closed body; mode 1: open without decl
    CHECK(build_patch_soup(obj, &P, MFTL_NULL) == MFTL_ERROR);
    MFTLEnsembleDestroy(&ens);
  }
  // explicit list missing one edge / with an extra (interior) edge: errors
  // (built from the unmatched set: the boundary edges of the patch)
  int nb = 0;
  MFTL_BoundaryEdgeDecl* bl = malloc((size_t) (P.n_outer + P.n_inner + 1) * sizeof(*bl));
  {
    // boundary edges of the triangle list = directed edges without reverse
    for (int t = 0; t < P.nt; ++t)
      for (int k = 0; k < 3; ++k)
      {
        int a = P.tri[t][k], b = P.tri[t][(k + 1) % 3], found = 0;
        for (int u = 0; u < P.nt && !found; ++u)
          for (int m = 0; m < 3; ++m)
            if (P.tri[u][m] == b && P.tri[u][(m + 1) % 3] == a) found = 1;
        if (!found)
        {
          bl[nb].vgid_a = 100 + 2 * a;
          bl[nb].vgid_b = 100 + 2 * b;
          bl[nb].tag = (a == 0 || b == 0 || P.pos[a][0] < -1.19 || P.pos[a][0] > 1.19 || P.pos[a][1] < -1.19 || P.pos[a][1] > 1.19) ? 5 : 9;
          nb++;
        }
      }
  }
  CHECK(nb == P.n_outer + P.n_inner);
  for (int variant = 0; variant < 4; ++variant)
  {
    MFTL_Ensemble* ens = make_default_ens();
    MFTL_Object* obj = MFTLObjectCreate(4);
    MFTLEnsembleAddObject(ens, obj);
    MFTLObjectDeclareOpen(obj, 1);
    MFTL_BoundaryDecl d = {0, 0, MFTL_NULL};
    MFTL_BoundaryEdgeDecl extra[1] = {{100 + 2 * (N + 1), 100 + 2 * (N + 2), 3}};  // interior edge
    if (variant == 0) d.from_mesh = 1;  // BoundaryFromMesh
    if (variant == 1)
    {
      d.n_edges = (g_rank == 0) ? nb : 0;  // explicit list (on one rank only: union is used)
      d.edges = bl;
    }
    if (variant == 2)
    {
      d.n_edges = (g_rank == 0) ? nb - 1 : 0;  // one boundary edge missing
      d.edges = bl;
    }
    if (variant == 3)
    {
      d.n_edges = (g_rank == 0) ? 1 : 0;  // wrong: an interior edge only
      d.edges = extra;
    }
    const int rc = build_patch_soup(obj, &P, &d);
    CHECK(rc == (variant <= 1 ? MFTL_SUCCESS : MFTL_ERROR));
    if (variant == 1 && g_rank == 0) fprintf(stderr, "(end of expected soup diagnostics)\n");
    if (rc == MFTL_SUCCESS)
    {
      CHECK(valid(obj, 0));  // before ownership: REMOTE resolved in global mode
      MFTLComputeTriangleProperties(obj);
      MFTLRedistribute(ens);
      CHECK(valid(obj, 1));
      CHECK(exactly_once(obj, P.nt) == 0);
      CHECK(global_boundary_edges(obj) == nb);
      CHECK(check_loops(obj, 2, MFTL_NULL) == 2);
      int badtag = 0;
      for (int i = 0; i < obj->n_half_edges; ++i)
        if (MFTLHeIsBoundary(&obj->half_edges[i]))
        {
          const int tg = MFTLHeBoundaryTag(&obj->half_edges[i]);
          if (variant == 0 && tg != 0 && tg != 1) badtag++;
          if (variant == 1 && tg != 5 && tg != 9) badtag++;
        }
      CHECK(badtag == 0);
    }
    MFTLEnsembleDestroy(&ens);
  }
  free(bl);
  // closed sphere soup, random placement + duplicates: every twin resolved before ownership
  {
    MFTL_Ensemble* ens = make_default_ens();
    MFTL_Object* ref = MFTLObjectCreate(9);
    MFTLEnsembleAddObject(ens, ref);
    double c[3] = {0, 0, 0};
    MFTLGenerateSphere(ref, c, 1.0, 2);  // on rank 0 only; broadcast the triangle list
    int nv = ref->n_vertices, nt = ref->n_triangles;
    MPI_Bcast(&nv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    double (*pos)[3] = malloc((size_t) nv * sizeof(*pos));
    int (*tri)[3] = malloc((size_t) nt * sizeof(*tri));
    if (g_rank == 0)
    {
      for (int i = 0; i < nv; ++i) memcpy(pos[i], ref->vertices[i].pos, sizeof(pos[i]));
      for (int t = 0; t < nt; ++t)
      {
        MFTL_Vertex* vv[3];
        MFTLGetTriangleVertices(ref, &ref->triangles[t], vv);
        for (int k = 0; k < 3; ++k) tri[t][k] = (int) (vv[k] - ref->vertices);
      }
    }
    MPI_Bcast(pos, 3 * nv, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(tri, 3 * nt, MPI_INT, 0, MPI_COMM_WORLD);
    MFTL_Object* obj = MFTLObjectCreate(10);
    MFTLEnsembleAddObject(ens, obj);
    Patch S = {nv, nt, pos, tri, 0, 0};
    CHECK(build_patch_soup(obj, &S, MFTL_NULL) == MFTL_SUCCESS);
    int remote = 0, g_remote = 0;
    for (int i = 0; i < obj->n_half_edges; ++i)
      if (MFTLHeIsRemote(&obj->half_edges[i])) remote++;
    MPI_Allreduce(&remote, &g_remote, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    CHECK(g_nproc == 1 || g_remote > 0);
    CHECK(valid(obj, 0));
    MFTLComputeTriangleProperties(obj);
    MFTLComputeTriangleProperties(ref);
    MFTLRedistribute(ens);
    CHECK(valid(obj, 1));
    CHECK(exactly_once(obj, nt) == 0);
    free(pos);
    free(tri);
    MFTLEnsembleDestroy(&ens);
  }
  free_patch(&P);
}

static void test_copy_commit(void)
{
  if (g_rank == 0) fprintf(stderr, "(expected commit diagnostics follow)\n");
  for (int mode = 0; mode < 5; ++mode)
  {
    // 0: ASSEMBLE whole body; 1: CUT half (open dst); 2: ASSEMBLE half -> error;
    // 3: CUT half into a closed dst -> error; 4: CUT half of an open patch (existing tag kept)
    MFTL_Ensemble* ens = make_default_ens();
    MFTL_Object* src;
    double c[3] = {0.05, -0.03, 0.02};
    if (mode == 4)
      src = make_open_patch(ens, 20, 11, 0, MFTL_NULL);
    else
      src = make_sphere(ens, 20, c, 2);
    MFTL_Object* dst = MFTLObjectCreate(21);
    MFTLEnsembleAddObject(ens, dst);
    MFTLObjectDeclareOpen(dst, mode == 1 || mode == 4);
    int cut_edges = 0;
    for (int t = 0; t < src->n_triangles; ++t)
    {
      MFTL_Triangle* tri = &src->triangles[t];
      if (tri->is_ghost) continue;
      const int take = (mode == 0) || (mode == 4 ? tri->center[0] > 0.05 : tri->center[2] > 0.0);
      if (!take) continue;
      MFTLObjectAppendTriangleCopy(dst, src, t);
      MFTL_HalfEdge* h[3];
      MFTLGetTriangleHalfedges(src, tri, h);
      for (int k = 0; k < 3; ++k)
      {
        if (!MFTLHeIsLocal(h[k])) continue;  // G1: LOCAL or BOUNDARY
        MFTL_Triangle* nb = &src->triangles[src->half_edges[h[k]->twin_idx].triangle_idx];
        const int ntake = (mode == 4 ? nb->center[0] > 0.05 : nb->center[2] > 0.0);
        if (!ntake) cut_edges++;
      }
    }
    int g_cut = 0;
    MPI_Allreduce(&cut_edges, &g_cut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int prior_b = (mode == 4) ? -1 : 0;
    const int rc = MFTLObjectCommitCopy(dst, (mode == 1 || mode == 3 || mode == 4) ? MFTL_COMMIT_CUT : MFTL_COMMIT_ASSEMBLE);
    CHECK(rc == ((mode == 2 || mode == 3) ? MFTL_ERROR : MFTL_SUCCESS));
    if (rc == MFTL_SUCCESS)
    {
      CHECK(valid(dst, 0));
      MFTLComputeTriangleProperties(dst);
      MFTLRedistribute(ens);
      CHECK(valid(dst, 1));
      CHECK(valid(src, 1));
      CHECK(exactly_once(dst, -1) == 0);
      if (mode == 0) CHECK(global_boundary_edges(dst) == 0);
      if (mode == 1)
      {
        CHECK(global_boundary_edges(dst) == g_cut);  // exactly the recorded cut set
        CHECK(check_loops(dst, 1, MFTL_NULL) == 1);
        int badtag = 0;
        for (int i = 0; i < dst->n_half_edges; ++i)
          if (MFTLHeIsBoundary(&dst->half_edges[i]) && MFTLHeBoundaryTag(&dst->half_edges[i]) != 0) badtag++;
        CHECK(badtag == 0);
      }
      if (mode == 4)
      {
        // the copied part of the outer boundary keeps tag 0; the new cut gets tag 1
        int n_new = 0, n_old = 0, bad = 0;
        for (int t = 0; t < dst->n_triangles; ++t)
        {
          if (dst->triangles[t].is_ghost) continue;
          MFTL_HalfEdge* h[3];
          MFTLGetTriangleHalfedges(dst, &dst->triangles[t], h);
          for (int k = 0; k < 3; ++k)
            if (MFTLHeIsBoundary(h[k]))
            {
              const int tg = MFTLHeBoundaryTag(h[k]);
              if (tg == 1)
                n_new++;
              else if (tg == 0)
                n_old++;
              else
                bad++;
            }
        }
        int v[3] = {n_new, n_old, bad}, gv[3];
        MPI_Allreduce(v, gv, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        CHECK(gv[0] == g_cut && gv[1] > 0 && gv[2] == 0);
        CHECK(check_loops_ex(dst, 1, MFTL_NULL, 1) == 1);  // one loop carrying tags 0 and 1
        (void) prior_b;
      }
    }
    MFTLEnsembleDestroy(&ens);
  }
  if (g_rank == 0) fprintf(stderr, "(end of expected commit diagnostics)\n");
}

static void test_discovery(void)
{
  // periodic layouts: default grid, 1-slab and 2-slab in a periodic x direction
  for (int layout = 0; layout < 3; ++layout)
  {
    if (layout == 2 && (g_nproc % 2) != 0) continue;
    int g[3];
    bool per[3] = {false, false, false};
    grid3(g_nproc, g);
    if (layout == 1)
    {
      g[0] = 1, g[1] = g_nproc, g[2] = 1;
      per[0] = true;
    }
    if (layout == 2)
    {
      g[0] = 2, g[1] = g_nproc / 2, g[2] = 1;
      per[0] = true;
    }
    for (int body = 0; body < 2; ++body)
    {
      MFTL_Ensemble* ens = make_ens(g, per);
      MFTL_Object* obj;
      int nt = 0;
      if (body == 0)
      {
        double c[3] = {layout ? 1.3 : 0.05, -0.03, 0.02};  // crosses the periodic x seam
        g_sphere_radius = layout ? 0.6 : 1.0;  // periodic: extent below half a period (Stage 13)
        obj = make_sphere(ens, 30, c, 3);
        g_sphere_radius = 1.0;
        nt = 20 * 64;
      }
      else
      {
        obj = make_open_patch(ens, 31, 13, 1, MFTL_NULL);
        Patch P = make_patch(13, 1);
        nt = P.nt;
        free_patch(&P);
      }
      CHECK(valid(obj, 1));
      CHECK(exactly_once(obj, nt) == 0);
      // A3: fresh positions -> resolved entirely by the owned-mode round
      strip_ghost_triangles(obj);
      int holes = (MFTLCheckTwinStatesLocal(obj, MFTL_VALIDATE_GHOSTS, 0) > 0), g_holes = 0;
      MPI_Allreduce(&holes, &g_holes, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      CHECK(g_nproc == 1 || g_holes);
      MFTL_DiscoveryStats st;
      CHECK(MFTLCompleteGhosts(obj, 0.0, &st) == MFTL_SUCCESS);
      CHECK(st.n_unresolved == 0);
      CHECK(st.n_fallback == 0);
      CHECK(st.n_round1 == st.n_requested);
      CHECK(valid(obj, 1));
      // forced fallback
      strip_ghost_triangles(obj);
      MFTLDiscoverySetForceFallback(1);
      CHECK(MFTLCompleteGhosts(obj, 0.0, &st) == MFTL_SUCCESS);
      MFTLDiscoverySetForceFallback(0);
      CHECK(st.n_round1 == 0 && st.n_fallback == st.n_requested && st.n_unresolved == 0);
      CHECK(valid(obj, 1));
      // stale positions: this rank's copies of rim vertices displaced -> round 1 misses, the
      // mandatory fallback completes
      strip_ghost_triangles(obj);
      const int nv_old = obj->n_vertices;
      double (*save)[3] = malloc((size_t) (nv_old + 1) * sizeof(*save));
      for (int v = 0; v < nv_old; ++v)
      {
        memcpy(save[v], obj->vertices[v].pos, sizeof(save[v]));
        for (int d = 0; d < 3; ++d) obj->vertices[v].pos[d] += 1.4;  // stale by far more than l_max/3
      }
      CHECK(MFTLCompleteGhosts(obj, 0.0, &st) == MFTL_SUCCESS);
      for (int v = 0; v < nv_old; ++v) memcpy(obj->vertices[v].pos, save[v], sizeof(save[v]));
      free(save);
      CHECK(valid(obj, 1));
      CHECK(st.n_unresolved == 0 && st.n_round1 + st.n_fallback == st.n_requested);
      long fb = st.n_fallback;
      CHECK(g_nproc == 1 || st.n_requested == 0 || fb > 0);
      MFTLEnsembleDestroy(&ens);
    }
  }
}

// v and w share an edge (scan; includes the neighbour across an incoming boundary edge)
static int is_neighbour(const MFTL_Object* obj, int v, int w)
{
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[i];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    const int o = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
    if ((o == v && he->vertex_idx == w) || (o == w && he->vertex_idx == v)) return 1;
  }
  return 0;
}

static void test_boundary_operators(void)
{
  // serial operators on a boundary-rich open patch (rank 0 holds everything at np 1; at np > 1
  // the same patch is built in a rank-local ensemble on every rank)
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_SELF, (int[3]){1, 1, 1}, (double[3]){GMIN[0], GMIN[1], GMIN[2]},
                                          (double[3]){GMAX[0], GMAX[1], GMAX[2]}, (bool[3]){false, false, false}, 2);
  MFTL_Object* obj = MFTLObjectCreate(40);
  MFTLEnsembleAddObject(ens, obj);
  MFTLObjectDeclareOpen(obj, 1);
  Patch P = make_patch(9, 1);
  CHECK(MFTLBuildTopologyFromTriples(obj, P.nv, (const double (*)[3]) P.pos, P.nt, (const int (*)[3]) P.tri) == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(obj);
  const int nb0 = P.n_outer + P.n_inner;
  int nb = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (MFTLHeIsBoundary(&obj->half_edges[i])) nb++;
  CHECK(nb == nb0);
  CHECK(MFTLCheckTwinStatesLocal(obj, 0, 1) == 0);

  // split three boundary edges: halves keep the tag; one boundary edge more each
  int splits = 0;
  for (int i = 0; i < obj->n_half_edges && splits < 3; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (!MFTLHeIsBoundary(he) || (he->flags & MFTL_FLAG_VISITED)) continue;
    const int tag = MFTLHeBoundaryTag(he);
    const int sv = obj->n_vertices, st = obj->n_triangles, sh = obj->n_half_edges;
    int vm = MFTLSplitEdge(obj, i);
    MFTLFinalizeRemeshing(obj, sv, st, sh, false);
    for (int k = 0; k < obj->n_half_edges; ++k) obj->half_edges[k].flags &= ~MFTL_FLAG_VISITED;
    for (int k = 0; k < obj->n_vertices; ++k) obj->vertices[k].flags &= ~MFTL_FLAG_VISITED;
    // the new vertex is a boundary vertex with an outgoing BOUNDARY half-edge of the same tag
    CHECK(MFTLVertexRing(obj, vm, MFTL_NULL, 0, MFTL_NULL) == MFTL_RING_OPEN_COMPLETE);
    CHECK(MFTLHeIsBoundary(&obj->half_edges[obj->vertices[vm].out_he_idx]));
    CHECK(MFTLHeBoundaryTag(&obj->half_edges[obj->vertices[vm].out_he_idx]) == tag);
    splits++;
  }
  nb = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (MFTLHeIsBoundary(&obj->half_edges[i])) nb++;
  CHECK(nb == nb0 + 3);
  CHECK(MFTLCheckTwinStatesLocal(obj, 0, 1) == 0);

  // flips of boundary edges are rejected; a flip of an interior edge keeps the states
  MFTL_RemeshParameters params;
  memset(&params, 0, sizeof(params));
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 0.3, 1, MFTL_SMOOTH_QUALITY_DRIVEN, 45.0);
  int flip_b = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (MFTLHeIsBoundary(&obj->half_edges[i]) && FlipApprovedWithParams(obj, i, &params)) flip_b++;
  CHECK(flip_b == 0);
  // an admissible interior flip (both endpoints valence > 3, not creating a duplicate edge)
  int flipped = 0;
  for (int i = 0; i < obj->n_half_edges && !flipped; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (!MFTLHeIsLocal(he) || !FlipApprovedWithParams(obj, i, MFTL_NULL)) continue;
    MFTLFlipEdge(obj, i);
    flipped = 1;
  }
  CHECK(flipped);
  CHECK(MFTLCheckTwinStatesLocal(obj, 0, 1) == 0);

  // collapse of an interior edge with one BOUNDARY wing (mftlGlueWings: the LOCAL wing becomes
  // BOUNDARY with that tag); the edge from a boundary vertex b to an interior vertex c
  int collapsed = 0;
  for (int i = 0; i < obj->n_half_edges && collapsed < 2; ++i)
  {
    MFTL_HalfEdge* hb = &obj->half_edges[i];  // a -> b BOUNDARY
    if (!MFTLHeIsBoundary(hb) || (hb->flags & MFTL_FLAG_DEAD)) continue;
    const int e = hb->next_idx;  // b -> c
    MFTL_HalfEdge* he = &obj->half_edges[e];
    if (!MFTLHeIsLocal(he)) continue;
    const int c = he->vertex_idx;
    if (MFTLVertexRing(obj, c, MFTL_NULL, 0, MFTL_NULL) != MFTL_RING_CLOSED) continue;
    // the other triangle must not have a boundary wing, and b, c share exactly 2 neighbours
    const int tw = he->twin_idx;
    const int w1 = obj->half_edges[tw].next_idx, w2 = obj->half_edges[w1].next_idx;
    if (!MFTLHeIsLocal(&obj->half_edges[w1]) || !MFTLHeIsLocal(&obj->half_edges[w2])) continue;
    const int b = hb->vertex_idx;
    int common = 0;
    for (int x = 0; x < obj->n_vertices; ++x)
      if (x != b && x != c && is_neighbour(obj, b, x) && is_neighbour(obj, c, x)) common++;
    if (common != 2) continue;
    const int tag = MFTLHeBoundaryTag(hb);
    int nb_before = 0;
    for (int k = 0; k < obj->n_half_edges; ++k)
      if (!(obj->half_edges[k].flags & MFTL_FLAG_DEAD) && MFTLHeIsBoundary(&obj->half_edges[k])) nb_before++;
    const int sv = obj->n_vertices, st = obj->n_triangles, sh = obj->n_half_edges;
    MFTLCollapseEdge(obj, e);
    MFTLFinalizeRemeshing(obj, sv, st, sh, true);
    for (int k = 0; k < obj->n_half_edges; ++k) obj->half_edges[k].flags &= ~MFTL_FLAG_VISITED;
    for (int k = 0; k < obj->n_vertices; ++k) obj->vertices[k].flags &= ~MFTL_FLAG_VISITED;
    int nb_after = 0, tag_ok = 1;
    for (int k = 0; k < obj->n_half_edges; ++k)
      if (MFTLHeIsBoundary(&obj->half_edges[k]))
      {
        nb_after++;
        if (MFTLHeBoundaryTag(&obj->half_edges[k]) < 0) tag_ok = 0;
      }
    CHECK(nb_after == nb_before);  // the boundary edge a-b is replaced by a-c
    CHECK(tag_ok);
    (void) tag;
    CHECK(MFTLCheckTwinStatesLocal(obj, 0, 1) == 0);
    collapsed++;
    i = -1;  // indices changed by compaction: restart the scan
  }
  if (collapsed != 2) fprintf(stderr, "collapsed=%d\n", collapsed);
  CHECK(collapsed == 2);
  // a collapse whose removed triangle has two BOUNDARY wings is rejected by the gate: open fan
  // 0-(1,2,3,4), triangle (0,1,2) has the BOUNDARY wings 0->1 and 1->2
  {
    MFTL_Object* fan = MFTLObjectCreate(41);
    MFTLEnsembleAddObject(ens, fan);
    MFTLObjectDeclareOpen(fan, 1);
    double fp[5][3] = {{0, 0, 0}, {1, 0, 0}, {0.3, 1, 0}, {-0.6, 0.8, 0}, {-1, 0, 0}};
    int ft[3][3] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}};
    CHECK(MFTLBuildTopologyFromTriples(fan, 5, (const double (*)[3]) fp, 3, (const int (*)[3]) ft) == MFTL_SUCCESS);
    MFTLComputeTriangleProperties(fan);
    params.min_edge_length_sq = 1e6;  // every edge "short": only topology decides
    params.criterion = MFTL_CRITERION_BOTSCH_KOBBELT;
    params.feature_angle_threshold = 180.0;
    int corner_rejected = 0, corner_seen = 0;
    for (int i = 0; i < fan->n_half_edges; ++i)
    {
      MFTL_HalfEdge* he = &fan->half_edges[i];
      const int n1 = he->next_idx, n2 = fan->half_edges[n1].next_idx;
      if (!MFTLHeIsLocal(he)) continue;
      if (!(MFTLHeIsBoundary(&fan->half_edges[n1]) && MFTLHeIsBoundary(&fan->half_edges[n2]))) continue;
      corner_seen++;
      if (!CollapseApproved(fan, i, &params)) corner_rejected++;
      if (!CollapseApproved(fan, he->twin_idx, &params)) corner_rejected++;  // same edge, other side
    }
    CHECK(corner_seen == 2 && corner_rejected == 2 * corner_seen);
  }
  free_patch(&P);
  MFTLEnsembleDestroy(&ens);
}

// deterministic jitter in [-1,1]^3 by vertex gid and step (identical on every copy)
static void jitter(MFTL_Object* obj, int step, double amp)
{
  for (int v = 0; v < obj->n_vertices; ++v)
  {
    unsigned h = (unsigned) obj->vertices[v].global_id * 2654435761u ^ (unsigned) (step + 1) * 40503u;
    for (int d = 0; d < 3; ++d)
    {
      h = h * 1103515245u + 12345u;
      obj->vertices[v].pos[d] += amp * (((h >> 8) & 0xffff) / 32767.5 - 1.0);
    }
  }
}

static void test_fuzz(void)
{
  int steps = 12;
  const char* e = getenv("S16_FUZZ_STEPS");
  if (e) steps = atoi(e);
  for (int body = 0; body < 2; ++body)
  {
    MFTL_Ensemble* ens = make_default_ens();
    MFTL_Object* obj;
    double c[3] = {0.05, -0.03, 0.02};
    if (body == 0)
      obj = make_sphere(ens, 50, c, 3);
    else
      obj = make_open_patch(ens, 51, 13, 1, MFTL_NULL);
    int n_loops0 = (body == 0) ? 0 : 2;
    int fails = 0;
    for (int s = 0; s < steps; ++s)
    {
      if (getenv("S16_VERBOSE"))
      {
        const int nn = global_owned_tris(obj);  // collective
        if (g_rank == 0) printf("  fuzz body %d step %d: %d owned triangles\n", body, s, nn);
      }
      jitter(obj, s, 0.004);
      MFTLComputeTriangleProperties(obj);
      MFTL_RemeshParameters params;
      memset(&params, 0, sizeof(params));
      MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, (s % 2) ? 0.16 : 0.22, 1, MFTL_SMOOTH_QUALITY_DRIVEN, 45.0);
      params.quiet = 1;
      params.max_smoothing_iter = 1;
      params.mftl1_criteria = (body == 0) && (s % 3 == 2);  // closed: also the parity criteria
      if (body == 1)
      {
        params.skip_volume_recovery = 1;  // open bodies have no enclosed volume
        params.skip_tangential_smoothing = 1;
      }
      MFTLRemeshEnsembleParallel(ens, &params);
      MFTLComputeTriangleProperties(obj);
      MFTLRedistribute(ens);
      int ok = valid(obj, 1);
      ok = ok && exactly_once(obj, -1) == 0;
      if (body == 1) ok = ok && boundary_edge_id_violations(obj) == 0;
      int nl = check_loops(obj, n_loops0, MFTL_NULL);
      ok = ok && nl == n_loops0;
      int tag_bad = 0;
      for (int i = 0; i < obj->n_half_edges; ++i)
        if (MFTLHeIsBoundary(&obj->half_edges[i]) && MFTLHeBoundaryTag(&obj->half_edges[i]) > 1) tag_bad++;
      ok = ok && tag_bad == 0;
      int lf = !ok, gf = 0;
      MPI_Allreduce(&lf, &gf, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
      if (gf)
      {
        fails++;
        if (g_rank == 0) fprintf(stderr, "fuzz body %d step %d: invariant violation\n", body, s);
        break;
      }
    }
    CHECK(fails == 0);
    const int n_end = global_owned_tris(obj);  // collective
    if (g_rank == 0) printf("  fuzz body %d: %d steps, %d owned triangles at the end\n", body, steps, n_end);
    MFTLEnsembleDestroy(&ens);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  setvbuf(stdout, MFTL_NULL, _IONBF, 0);

  struct
  {
    const char* name;
    void (*fn)(void);
  } tests[] = {{"closed_sphere", test_closed_sphere},           {"open_patch", test_open_patch}, {"builder_rejections", test_builder_rejections},
               {"soup_before_ownership", test_soup_before_ownership}, {"copy_commit", test_copy_commit}, {"discovery", test_discovery},
               {"boundary_operators", test_boundary_operators}, {"fuzz", test_fuzz}};
  for (size_t k = 0; k < sizeof(tests) / sizeof(tests[0]); ++k)
  {
    const int before = g_failed;
    const char* only = getenv("S16_ONLY");
    if (only && !strstr(only, tests[k].name)) continue;
    if (g_rank == 0) printf("== %s\n", tests[k].name);
    tests[k].fn();
    if (g_rank == 0) printf("   %s: %s\n", tests[k].name, g_failed == before ? "ok" : "FAILED");
  }
  if (g_rank == 0) printf("Stage16A np%d: %d checks, %d failed\n", g_nproc, g_checks, g_failed);
  MPI_Finalize();
  return g_failed ? 1 : 0;
}
