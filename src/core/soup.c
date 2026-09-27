/*
 * MFTL2 core — explicit-id construction and entity copies.
 *
 *  - MFTLBuildFromSoup / Ex / Img: builds an object from a distributed triangle soup whose
 *    vertices and triangles carry GLOBAL ids (the same entity may be added on several ranks,
 *    e.g. a marching-cubes triangulation of overlapping ghost cells). Half-edge ids are derived
 *    collectively from the global rank of the triangle id (compact and identical on every
 *    rank holding the triangle), twins are paired locally, and every unpaired half-edge is
 *    resolved globally before ownership exists (plans/16 §2.4, global mode): a partner on
 *    another rank makes it REMOTE; an edge unmatched everywhere becomes BOUNDARY only on a
 *    declared-open body with a boundary declaration, and is a collective error otherwise.
 *    Manifoldness and orientation are validated, and the image counters are canonicalised.
 *  - MFTLObjectAppendTriangleCopy: copies one triangle with its half-edges and missing
 *    vertices into another object, deduplicating by global id and linking twins through the
 *    retained twin gids (connected-component extraction); the batch is staged until
 *    MFTLObjectCommitCopy (twin_states.c).
 *  - MFTLObjectClear: removes every entity from an object.
 *
 * Every entity created by a build is stored as owned (is_ghost == false) with the given owner
 * rank: the soup is authoritative until the next redistribution re-derives ownership.
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
#include "mftl_twin_internal.h"
#include "mftl_images.h"

static int mftlCmpInt(const void* a, const void* b)
{
  int x = *(const int*) a, y = *(const int*) b;
  return (x > y) - (x < y);
}

// position of key in a sorted unique array, -1 if absent
static int mftlBsearch(const int* arr, int n, int key)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    int mid = lo + (hi - lo) / 2;
    if (arr[mid] == key) return mid;
    if (arr[mid] < key)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

// rim record: undirected vertex-gid pair, direction and half-edge gid
typedef struct
{
  int lo, hi, forward, he_gid;
} mftlRimRec;

static int mftlCmpRim(const void* a, const void* b)
{
  const mftlRimRec* x = (const mftlRimRec*) a;
  const mftlRimRec* y = (const mftlRimRec*) b;
  if (x->lo != y->lo) return (x->lo > y->lo) - (x->lo < y->lo);
  if (x->hi != y->hi) return (x->hi > y->hi) - (x->hi < y->hi);
  if (x->forward != y->forward) return x->forward - y->forward;
  return (x->he_gid > y->he_gid) - (x->he_gid < y->he_gid);
}

static void mftlAllgathervInts(MPI_Comm comm, const int* send, int nsend, int** recv, int* nrecv)
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
  *recv = (int*) malloc((size_t) (total > 0 ? total : 1) * sizeof(int));
  MPI_Allgatherv(send, nsend, MPI_INT, *recv, counts, displs, MPI_INT, comm);
  *nrecv = total;
  free(counts);
  free(displs);
}

int MFTLBuildFromSoup(MFTL_Object* obj, int nV, const int* vgid, const double (*pos)[3], const int* vowner, int nT, const int* tgid, const int (*tvg)[3],
                      const int* towner)
{
  return MFTLBuildFromSoupEx(obj, nV, vgid, pos, vowner, nT, tgid, tvg, towner, MFTL_NULL);
}

static int mftlCmpDecl(const void* a, const void* b)
{
  const int* x = (const int*) a;
  const int* y = (const int*) b;
  if (x[0] != y[0]) return (x[0] > y[0]) - (x[0] < y[0]);
  return (x[1] > y[1]) - (x[1] < y[1]);
}

static int mftlSoupCore(MFTL_Object* obj, int nV, const int* vgid, const double (*pos)[3], const int (*vimg)[3], const int* vowner, int nT,
                        const int* tgid, const int (*tvg)[3], const int* towner, const MFTL_BoundaryDecl* decl);

int MFTLBuildFromSoupEx(MFTL_Object* obj, int nV, const int* vgid, const double (*pos)[3], const int* vowner, int nT, const int* tgid,
                        const int (*tvg)[3], const int* towner, const MFTL_BoundaryDecl* decl)
{
  return mftlSoupCore(obj, nV, vgid, pos, MFTL_NULL, vowner, nT, tgid, tvg, towner, decl);
}

int MFTLBuildFromSoupImg(MFTL_Object* obj, int nV, const int* vgid, const double (*pos)[3], const int (*vimg)[3], const int* vowner, int nT,
                         const int* tgid, const int (*tvg)[3], const int* towner, const MFTL_BoundaryDecl* decl)
{
  return mftlSoupCore(obj, nV, vgid, pos, vimg, vowner, nT, tgid, tvg, towner, decl);
}

static int mftlSoupCore(MFTL_Object* obj, int nV, const int* vgid, const double (*pos)[3], const int (*vimg)[3], const int* vowner, int nT,
                        const int* tgid, const int (*tvg)[3], const int* towner, const MFTL_BoundaryDecl* decl)
{
  MFTL_ASSERT(obj != MFTL_NULL && obj->domain != MFTL_NULL);
  MFTL_Domain* domain = obj->domain;
  MPI_Comm comm = domain->comm_cart;

  // the object must be empty on every rank (checked collectively before touching it)
  int bad = (obj->n_vertices != 0 || obj->n_triangles != 0 || obj->n_half_edges != 0);
  int gbad_empty = 0;
  MPI_Allreduce(&bad, &gbad_empty, 1, MPI_INT, MPI_MAX, comm);
  if (gbad_empty) return MFTL_ERROR;

  // --- 1. local vertices (dedup by gid, first occurrence wins) -------------------------
  MFTL_RESERVE_ADDITIONAL(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, nV, MFTL_Vertex);
  int nv_loc = 0;
  for (int i = 0; i < nV && !bad; ++i)
  {
    khiter_t k = kh_get(vertex_map, obj->vertex_lookup, vgid[i]);
    if (k != kh_end(obj->vertex_lookup)) continue;
    MFTL_Vertex* v = &obj->vertices[nv_loc];
    memset(v, 0, sizeof(*v));
    v->global_id = vgid[i];
    v->owner_proc = vowner ? vowner[i] : domain->rank;
    v->is_ghost = MFTL_FALSE;
    if (vimg)
    {
      memcpy(v->pos, pos[i], 3 * sizeof(double));  // wrapped position with explicit counters (restart)
      memcpy(v->img, vimg[i], 3 * sizeof(int));
    }
    else
      MFTLWrapContinuous(domain, pos[i], v->pos, v->img);  // continuous input
    v->out_he_idx = MFTL_ERROR;
    int ret;
    k = kh_put(vertex_map, obj->vertex_lookup, v->global_id, &ret);
    kh_value(obj->vertex_lookup, k) = nv_loc;
    nv_loc++;
  }
  obj->n_vertices = nv_loc;
  obj->n_local_vertices = nv_loc;

  // --- 2. global triangle-id ranking -> compact, rank-independent half-edge ids ---------
  int* tsorted = (int*) malloc((size_t) (nT > 0 ? nT : 1) * sizeof(int));
  if (nT > 0) memcpy(tsorted, tgid, (size_t) nT * sizeof(int));
  int* all_t = MFTL_NULL;
  int n_all_t = 0;
  mftlAllgathervInts(comm, tsorted, nT, &all_t, &n_all_t);
  qsort(all_t, (size_t) n_all_t, sizeof(int), mftlCmpInt);
  int n_unique_t = 0;
  for (int i = 0; i < n_all_t; ++i)
    if (n_unique_t == 0 || all_t[i] != all_t[n_unique_t - 1]) all_t[n_unique_t++] = all_t[i];

  int local_max[3] = {-1, -1, obj->next_halfedge_gid - 1}, global_max[3];
  for (int i = 0; i < nv_loc; ++i) local_max[0] = MFTL_MAX(local_max[0], obj->vertices[i].global_id);
  for (int t = 0; t < nT; ++t) local_max[1] = MFTL_MAX(local_max[1], tgid[t]);
  MPI_Allreduce(local_max, global_max, 3, MPI_INT, MPI_MAX, comm);
  const int he_base = global_max[2] + 1;

  // --- 3. local triangles + half-edges (dedup by gid) ----------------------------------
  MFTL_RESERVE_ADDITIONAL(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, nT, MFTL_Triangle);
  MFTL_RESERVE_ADDITIONAL(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 3 * nT, MFTL_HalfEdge);
  int nt_loc = 0;
  for (int t = 0; t < nT && !bad; ++t)
  {
    khiter_t k = kh_get(triangle_map, obj->triangle_lookup, tgid[t]);
    if (k != kh_end(obj->triangle_lookup)) continue;
    int vi[3];
    for (int j = 0; j < 3; ++j)
    {
      khiter_t kv = kh_get(vertex_map, obj->vertex_lookup, tvg[t][j]);
      if (kv == kh_end(obj->vertex_lookup))
      {
        fprintf(stderr, "MFTLBuildFromSoup: triangle %d references unknown vertex %d\n", tgid[t], tvg[t][j]);
        bad = 1;
        break;
      }
      vi[j] = kh_value(obj->vertex_lookup, kv);
    }
    if (bad) break;
    const int trank = mftlBsearch(all_t, n_unique_t, tgid[t]);
    MFTL_Triangle* tri = &obj->triangles[nt_loc];
    memset(tri, 0, sizeof(*tri));
    tri->global_id = tgid[t];
    tri->owner_proc = towner ? towner[t] : domain->rank;
    tri->is_ghost = MFTL_FALSE;
    tri->he_idx = 3 * nt_loc;
    tri->data = (obj->data_size > 0) ? calloc(1, obj->data_size) : MFTL_NULL;
    tri->surface_scalar = 0.0;
    for (int j = 0; j < 3; ++j)
    {
      MFTL_HalfEdge* he = &obj->half_edges[3 * nt_loc + j];
      memset(he, 0, sizeof(*he));
      he->global_id = he_base + 3 * trank + j;
      he->triangle_idx = nt_loc;
      he->vertex_idx = vi[(j + 1) % 3];  // half-edge j: v_j -> v_(j+1)
      he->next_idx = 3 * nt_loc + (j + 1) % 3;
      he->twin_idx = MFTL_ERROR;
      he->twin_gid = MFTL_TWIN_UNKNOWN;
      he->boundary = MFTL_FALSE;
      if (obj->vertices[vi[j]].out_he_idx == MFTL_ERROR) obj->vertices[vi[j]].out_he_idx = 3 * nt_loc + j;
      int ret;
      khiter_t kh = kh_put(halfedge_map, obj->halfedge_lookup, he->global_id, &ret);
      kh_value(obj->halfedge_lookup, kh) = 3 * nt_loc + j;
    }
    int ret;
    k = kh_put(triangle_map, obj->triangle_lookup, tri->global_id, &ret);
    kh_value(obj->triangle_lookup, k) = nt_loc;
    nt_loc++;
  }
  obj->n_triangles = nt_loc;
  obj->n_local_triangles = nt_loc;
  obj->n_half_edges = 3 * nt_loc;
  obj->n_local_half_edges = 3 * nt_loc;
  free(tsorted);
  free(all_t);

  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, comm);
  if (gbad) return MFTL_ERROR;

  // --- 3b. drop vertices no local triangle references (they carry no mesh) ---------------
  {
    int* vmap = (int*) malloc((size_t) (obj->n_vertices > 0 ? obj->n_vertices : 1) * sizeof(int));
    int nkeep = 0;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if (obj->vertices[i].out_he_idx == MFTL_ERROR)
      {
        vmap[i] = MFTL_ERROR;
        continue;
      }
      vmap[i] = nkeep;
      if (i != nkeep) obj->vertices[nkeep] = obj->vertices[i];
      nkeep++;
    }
    if (nkeep != obj->n_vertices)
    {
      for (int i = 0; i < obj->n_half_edges; ++i) obj->half_edges[i].vertex_idx = vmap[obj->half_edges[i].vertex_idx];
      obj->n_vertices = obj->n_local_vertices = nkeep;
      kh_clear(vertex_map, obj->vertex_lookup);
      for (int i = 0; i < nkeep; ++i)
      {
        int ret;
        khiter_t k = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
        kh_value(obj->vertex_lookup, k) = i;
      }
    }
    free(vmap);
  }

  // --- 4. local twins, then the rim twins through one gather of the unpaired half-edges --
  // (GLOBAL discovery mode, plans/16 §2.4: before ownership, every rank's copy counts.)
  int bad4 = (MFTLInternalCountDirectedDuplicates(obj, 1) > 0);  // non-manifold / orientation
  MFTLRebuildTwins(obj);
  int n_rim = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (!MFTLHeIsLocal(&obj->half_edges[i])) n_rim++;
  int* rim_send = (int*) malloc((size_t) (4 * (n_rim > 0 ? n_rim : 1)) * sizeof(int));
  int r = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (MFTLHeIsLocal(he)) continue;
    int a = MFTLHeOrigin(obj, he)->global_id, b = MFTLHeTarget(obj, he)->global_id;
    rim_send[4 * r + 0] = a < b ? a : b;
    rim_send[4 * r + 1] = a < b ? b : a;
    rim_send[4 * r + 2] = a < b;
    rim_send[4 * r + 3] = he->global_id;
    r++;
  }
  int* rim_all = MFTL_NULL;
  int n_rim_all = 0;
  mftlAllgathervInts(comm, rim_send, 4 * n_rim, &rim_all, &n_rim_all);
  n_rim_all /= 4;
  // phase 2: locally paired half-edges on an edge that is on some rank's rim (the same edge may
  // be interior on one rank and rim on another when triangles are duplicated across ranks)
  {
    int(*keys)[2] = malloc((size_t) (n_rim_all > 0 ? n_rim_all : 1) * sizeof(*keys));
    for (int i = 0; i < n_rim_all; ++i)
    {
      keys[i][0] = rim_all[4 * i];
      keys[i][1] = rim_all[4 * i + 1];
    }
    qsort(keys, (size_t) n_rim_all, sizeof(*keys), mftlCmpDecl);
    int n2 = 0, cap2 = 64;
    int* send2 = (int*) malloc((size_t) (4 * cap2) * sizeof(int));
    for (int i = 0; i < obj->n_half_edges && n_rim_all > 0; ++i)
    {
      MFTL_HalfEdge* he = &obj->half_edges[i];
      if (!MFTLHeIsLocal(he)) continue;
      int a = MFTLHeOrigin(obj, he)->global_id, b = MFTLHeTarget(obj, he)->global_id;
      int key[2] = {MFTL_MIN(a, b), MFTL_MAX(a, b)};
      if (!bsearch(key, keys, (size_t) n_rim_all, sizeof(*keys), mftlCmpDecl)) continue;
      if (n2 == cap2)
      {
        cap2 *= 2;
        send2 = (int*) realloc(send2, (size_t) (4 * cap2) * sizeof(int));
      }
      send2[4 * n2 + 0] = key[0];
      send2[4 * n2 + 1] = key[1];
      send2[4 * n2 + 2] = a < b;
      send2[4 * n2 + 3] = he->global_id;
      n2++;
    }
    free(keys);
    int* all2 = MFTL_NULL;
    int n_all2 = 0;
    mftlAllgathervInts(comm, send2, 4 * n2, &all2, &n_all2);
    free(send2);
    if (n_all2 > 0)
    {
      rim_all = (int*) realloc(rim_all, (size_t) (4 * n_rim_all + n_all2) * sizeof(int));
      memcpy(rim_all + 4 * n_rim_all, all2, (size_t) n_all2 * sizeof(int));
      n_rim_all += n_all2 / 4;
    }
    free(all2);
  }
  qsort(rim_all, (size_t) n_rim_all, sizeof(mftlRimRec), mftlCmpRim);
  mftlRimRec* base = (mftlRimRec*) rim_all;
  // drop identical duplicates (the same half-edge held on several ranks); a second half-edge
  // with the same direction on the same vertex pair is a non-manifold edge / orientation error
  int n_u = 0;
  for (int i = 0; i < n_rim_all; ++i)
  {
    if (n_u > 0 && base[i].lo == base[n_u - 1].lo && base[i].hi == base[n_u - 1].hi && base[i].forward == base[n_u - 1].forward)
    {
      if (base[i].he_gid != base[n_u - 1].he_gid)
      {
        if (domain->rank == 0)
          fprintf(stderr, "MFTLBuildFromSoup: duplicate directed edge %d-%d (half-edges %d, %d): non-manifold edge or inconsistent orientation\n",
                  base[i].lo, base[i].hi, base[n_u - 1].he_gid, base[i].he_gid);
        bad4 = 1;
      }
      continue;
    }
    base[n_u++] = base[i];
  }
  // partner of each unique rim record; unmatched ones form the global boundary candidate set U
  int* partner = (int*) malloc((size_t) (n_u > 0 ? n_u : 1) * sizeof(int));
  int n_unm = 0;
  for (int i = 0; i < n_u; ++i)
  {
    mftlRimRec key = {base[i].lo, base[i].hi, !base[i].forward, -2147483647 - 1};
    int lo = 0, hi = n_u;
    while (lo < hi)
    {
      int mid = lo + (hi - lo) / 2;
      if (mftlCmpRim(&base[mid], &key) < 0)
        lo = mid + 1;
      else
        hi = mid;
    }
    partner[i] = (lo < n_u && base[lo].lo == key.lo && base[lo].hi == key.hi && base[lo].forward == key.forward) ? base[lo].he_gid : MFTL_ERROR;
    if (partner[i] == MFTL_ERROR) n_unm++;
  }
  // U: unmatched records {origin gid, target gid, he gid, tag}
  int(*U)[4] = malloc((size_t) (n_unm > 0 ? n_unm : 1) * sizeof(*U));
  n_unm = 0;
  for (int i = 0; i < n_u; ++i)
  {
    if (partner[i] != MFTL_ERROR) continue;
    U[n_unm][0] = base[i].forward ? base[i].lo : base[i].hi;
    U[n_unm][1] = base[i].forward ? base[i].hi : base[i].lo;
    U[n_unm][2] = base[i].he_gid;
    U[n_unm][3] = -1;
    n_unm++;
  }
  if (n_unm > 0)
  {
    // the declaration is the union over ranks (collective decision)
    int dloc[2] = {decl ? decl->from_mesh : 0, decl ? decl->n_edges : 0}, dglob[2];
    MPI_Allreduce(dloc, dglob, 2, MPI_INT, MPI_MAX, comm);
    if (!obj->declared_open || (!dglob[0] && dglob[1] == 0))
    {
      if (domain->rank == 0)
        fprintf(stderr, "MFTLBuildFromSoup: %d edge(s) unmatched on every rank (first %d-%d) on a body %s\n", n_unm, U[0][0], U[0][1],
                obj->declared_open ? "declared open without a boundary declaration" : "not declared open");
      bad4 = 1;
    }
    else if (dglob[0])
    {
      int(*ab)[2] = malloc((size_t) n_unm * sizeof(*ab));
      int* tag = malloc((size_t) n_unm * sizeof(int));
      for (int i = 0; i < n_unm; ++i)
      {
        ab[i][0] = U[i][0];
        ab[i][1] = U[i][1];
      }
      if (MFTLInternalTagBoundaryLoops(n_unm, (const int(*)[2]) ab, tag) != MFTL_SUCCESS)
      {
        if (domain->rank == 0) fprintf(stderr, "MFTLBuildFromSoup: the unmatched edges do not form simple loops\n");
        bad4 = 1;
      }
      else
        for (int i = 0; i < n_unm; ++i) U[i][3] = tag[i];
      free(ab);
      free(tag);
    }
    else
    {
      const int nde = decl ? decl->n_edges : 0;
      int* dl = (int*) malloc((size_t) (3 * (nde > 0 ? nde : 1)) * sizeof(int));
      for (int i = 0; i < nde; ++i)
      {
        dl[3 * i + 0] = MFTL_MIN(decl->edges[i].vgid_a, decl->edges[i].vgid_b);
        dl[3 * i + 1] = MFTL_MAX(decl->edges[i].vgid_a, decl->edges[i].vgid_b);
        dl[3 * i + 2] = decl->edges[i].tag;
      }
      int *dall = MFTL_NULL, nd = 0;
      mftlAllgathervInts(comm, dl, 3 * nde, &dall, &nd);
      nd /= 3;
      free(dl);
      qsort(dall, (size_t) nd, 3 * sizeof(int), mftlCmpDecl);
      char* used = (char*) calloc((size_t) (nd > 0 ? nd : 1), 1);
      for (int i = 0; i < n_unm; ++i)
      {
        int key[3] = {MFTL_MIN(U[i][0], U[i][1]), MFTL_MAX(U[i][0], U[i][1]), 0};
        int* f = (int*) bsearch(key, dall, (size_t) nd, 3 * sizeof(int), mftlCmpDecl);
        if (!f || f[2] < 0)
        {
          if (domain->rank == 0) fprintf(stderr, "MFTLBuildFromSoup: unmatched edge %d-%d is not declared a boundary\n", key[0], key[1]);
          bad4 = 1;
          continue;
        }
        U[i][3] = f[2];
        // mark every (duplicate) declaration of this edge as used
        for (int* g = f; g >= dall && g[0] == key[0] && g[1] == key[1]; g -= 3) used[(g - dall) / 3] = 1;
        for (int* g = f; g < dall + 3 * nd && g[0] == key[0] && g[1] == key[1]; g += 3) used[(g - dall) / 3] = 1;
      }
      for (int k = 0; k < nd; ++k)
        if (!used[k])
        {
          if (domain->rank == 0) fprintf(stderr, "MFTLBuildFromSoup: declared boundary edge %d-%d is not an unmatched edge\n", dall[3 * k], dall[3 * k + 1]);
          bad4 = 1;
        }
      free(used);
      free(dall);
    }
  }
  // apply: unmatched -> BOUNDARY with its tag; partner found -> REMOTE (below)
  {
    int(*Ug)[2] = malloc((size_t) (n_unm > 0 ? n_unm : 1) * sizeof(*Ug));
    for (int i = 0; i < n_unm; ++i)
    {
      Ug[i][0] = U[i][2];
      Ug[i][1] = U[i][3];
    }
    qsort(Ug, (size_t) n_unm, sizeof(*Ug), mftlCmpDecl);
    for (int i = 0; i < obj->n_half_edges; ++i)
    {
      MFTL_HalfEdge* he = &obj->half_edges[i];
      if (MFTLHeIsLocal(he)) continue;
      int gk[2] = {he->global_id, 0};
      int(*f)[2] = bsearch(gk, Ug, (size_t) n_unm, sizeof(*Ug), mftlCmpInt);
      if (f && (*f)[1] >= 0) MFTLHeSetBoundary(he, (*f)[1]);
    }
    free(Ug);
  }
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (MFTLHeIsLocal(he) || MFTLHeIsBoundary(he)) continue;
    int a = MFTLHeOrigin(obj, he)->global_id, b = MFTLHeTarget(obj, he)->global_id;
    mftlRimRec key = {a < b ? a : b, a < b ? b : a, !(a < b), -2147483647 - 1};
    int lo = 0, hi = n_u;  // first record >= key
    while (lo < hi)
    {
      int mid = lo + (hi - lo) / 2;
      if (mftlCmpRim(&base[mid], &key) < 0)
        lo = mid + 1;
      else
        hi = mid;
    }
    if (lo < n_u && base[lo].lo == key.lo && base[lo].hi == key.hi && base[lo].forward == key.forward) MFTLHeSetRemote(he, base[lo].he_gid);
  }
  free(partner);
  free(U);
  free(rim_send);
  free(rim_all);
  MFTLRefreshBoundaryCache(obj);
  bad4 += MFTLCheckTwinStatesLocal(obj, 0, 1);
  int gbad4 = 0;
  MPI_Allreduce(&bad4, &gbad4, 1, MPI_INT, MPI_MAX, comm);
  if (gbad4) return MFTL_ERROR;

  // --- 5. counters and acceleration arrays ---------------------------------------------
  obj->next_vertex_gid = MFTL_MAX(obj->next_vertex_gid, global_max[0] + 1);
  obj->next_triangle_gid = MFTL_MAX(obj->next_triangle_gid, global_max[1] + 1);
  obj->next_halfedge_gid = he_base + 3 * n_unique_t;
  MFTLFlattenConnectivity(obj);
  // assembled pieces may be gauged independently -> canonicalise (plans/16 §3.2/§3.3)
  return MFTLCanonicaliseImages(obj);
}

int MFTLObjectAppendTriangleCopy(MFTL_Object* dst, const MFTL_Object* src, int tri_idx)
{
  dst->wrap_valid = 0;  // staged; MFTLObjectCommitCopy canonicalises
  const MFTL_Triangle* st = &src->triangles[tri_idx];
  khiter_t kt = kh_get(triangle_map, dst->triangle_lookup, st->global_id);
  if (kt != kh_end(dst->triangle_lookup)) return kh_value(dst->triangle_lookup, kt);

  // vertices first (dedup by gid)
  MFTL_HalfEdge* sh[3];
  MFTLGetTriangleHalfedges(src, st, sh);
  int dv[3];
  for (int j = 0; j < 3; ++j)
  {
    const MFTL_Vertex* sv = MFTLHeTarget(src, sh[j]);  // target of half-edge j
    khiter_t kv = kh_get(vertex_map, dst->vertex_lookup, sv->global_id);
    if (kv != kh_end(dst->vertex_lookup))
    {
      dv[j] = kh_value(dst->vertex_lookup, kv);
      continue;
    }
    MFTL_ENSURE_CAPACITY(dst->vertices, &dst->n_vertices, &dst->capacity_vertices, 1, MFTL_Vertex);
    int idx = dst->n_vertices++;
    dst->vertices[idx] = *sv;
    dst->vertices[idx].is_ghost = MFTL_FALSE;
    dst->vertices[idx].out_he_idx = MFTL_ERROR;
    dst->vertices[idx].flags = sv->flags & MFTL_VFLAG_CORNER;  // the corner tag travels with copies
    int ret;
    kv = kh_put(vertex_map, dst->vertex_lookup, sv->global_id, &ret);
    kh_value(dst->vertex_lookup, kv) = idx;
    dv[j] = idx;
  }

  // triangle
  MFTL_ENSURE_CAPACITY(dst->triangles, &dst->n_triangles, &dst->capacity_triangles, 1, MFTL_Triangle);
  MFTL_ENSURE_CAPACITY(dst->half_edges, &dst->n_half_edges, &dst->capacity_half_edges, 3, MFTL_HalfEdge);
  int ti = dst->n_triangles++;
  int h0 = dst->n_half_edges;
  dst->n_half_edges += 3;
  MFTL_Triangle* dt = &dst->triangles[ti];
  *dt = *st;
  dt->is_ghost = MFTL_FALSE;
  dt->he_idx = h0;
  dt->flags = MFTL_FLAG_NONE;
  if (dst->data_size > 0)
  {
    dt->data = calloc(1, dst->data_size);
    if (src->data_size == dst->data_size && st->data) memcpy(dt->data, st->data, dst->data_size);
  }
  else
  {
    dt->data = MFTL_NULL;
  }
  int ret;
  kt = kh_put(triangle_map, dst->triangle_lookup, dt->global_id, &ret);
  kh_value(dst->triangle_lookup, kt) = ti;

  // half-edges: j-th copy keeps the source's j-th half-edge (target dv[j]); twins by gid
  for (int j = 0; j < 3; ++j)
  {
    MFTL_HalfEdge* he = &dst->half_edges[h0 + j];
    memset(he, 0, sizeof(*he));
    he->global_id = sh[j]->global_id;
    he->twin_gid = MFTLHeTwinCode(src, sh[j]);  // staged: source state code until the commit
    he->vertex_idx = dv[j];
    he->next_idx = h0 + (j + 1) % 3;
    he->triangle_idx = ti;
    he->twin_idx = MFTL_ERROR;
    he->boundary = MFTLHeIsBoundary(he);
    he->flags = sh[j]->flags & MFTL_FLAG_FEATURE;
    int origin = dv[(j + 2) % 3];
    if (dst->vertices[origin].out_he_idx == MFTL_ERROR) dst->vertices[origin].out_he_idx = h0 + j;
    khiter_t kh = kh_put(halfedge_map, dst->halfedge_lookup, he->global_id, &ret);
    kh_value(dst->halfedge_lookup, kh) = h0 + j;
    if (MFTLHeIsRemote(he))  // staged: the source twin gid (a BOUNDARY code is kept)
    {
      khiter_t ktw = kh_get(halfedge_map, dst->halfedge_lookup, he->twin_gid);
      if (ktw != kh_end(dst->halfedge_lookup))
      {
        int tw = kh_value(dst->halfedge_lookup, ktw);
        he->twin_idx = tw;
        he->boundary = MFTL_FALSE;
        dst->half_edges[tw].twin_idx = h0 + j;
        dst->half_edges[tw].twin_gid = he->global_id;
        dst->half_edges[tw].boundary = MFTL_FALSE;
      }
    }
  }
  dst->n_local_vertices = dst->n_vertices;
  dst->n_local_triangles = dst->n_triangles;
  dst->n_local_half_edges = dst->n_half_edges;
  dst->next_vertex_gid = MFTL_MAX(dst->next_vertex_gid, src->next_vertex_gid);
  dst->next_triangle_gid = MFTL_MAX(dst->next_triangle_gid, src->next_triangle_gid);
  dst->next_halfedge_gid = MFTL_MAX(dst->next_halfedge_gid, src->next_halfedge_gid);
  return ti;
}

void MFTLObjectClear(MFTL_Object* obj)
{
  obj->wrap_valid = 0;  // no stale winding flags for the next contents
  if (obj->data_size > 0)
    for (int i = 0; i < obj->n_triangles; ++i) free(obj->triangles[i].data);
  obj->n_vertices = obj->n_triangles = obj->n_half_edges = 0;
  obj->n_local_vertices = obj->n_local_triangles = obj->n_local_half_edges = 0;
  kh_clear(vertex_map, obj->vertex_lookup);
  kh_clear(triangle_map, obj->triangle_lookup);
  kh_clear(halfedge_map, obj->halfedge_lookup);
  free(obj->flat_tri_verts);
  free(obj->flat_tri_neighbors);
  free(obj->vert_offsets);
  free(obj->vert_tris);
  obj->flat_tri_verts = obj->flat_tri_neighbors = obj->vert_offsets = obj->vert_tris = MFTL_NULL;
}
