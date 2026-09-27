/*
 * MFTL2 core — half-edge topology construction: twin pairing (MFTLRebuildTwins), global-id ->
 * local-index conversion after unpacking, the structural integrity check, flattened
 * connectivity, the triple builder and the vertex/triangle neighbourhood gathering.
 *
 * All functions are rank-local and act on the STORED entities (owned and ghost).
 */

#include "mftl_topology.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_twin_internal.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// origin vertex of a half-edge (triangles: prev == next->next)
static inline int mftlHeStartVertex(MFTL_HalfEdge *he_base, int he_idx)
{
  int next_idx = he_base[he_idx].next_idx;
  int prev_idx = he_base[next_idx].next_idx;
  return he_base[prev_idx].vertex_idx;
}

void MFTLRebuildTwins(MFTL_Object *obj)
{
  const int n_he = obj->n_half_edges;
  const int n_vert = obj->n_vertices;
  MFTL_HalfEdge *he_base = obj->half_edges;

  // Linked lists of the half-edges leaving each vertex: v_heads[v] = first one, he_next[h] = next
  int *v_heads = (int *) malloc(sizeof(int) * n_vert);
  int *he_next = (int *) malloc(sizeof(int) * n_he);

  assert(v_heads && he_next);

  for (int i = 0; i < n_vert; i++)
  {
    v_heads[i] = MFTL_ERROR;
  }

  for (int i = 0; i < n_he; i++)
  {
    int u = mftlHeStartVertex(he_base, i);
    assert(u >= 0 && u < n_vert);

    he_next[i] = v_heads[u];
    v_heads[u] = i;

    MFTL_HalfEdge *he = &he_base[i];
    he->twin_idx = MFTL_ERROR;  // unpaired half-edges keep their state code (REMOTE / BOUNDARY)
  }

  // the twin of u -> v is a stored half-edge v -> u
  for (int i = 0; i < n_he; i++)
  {
    MFTL_HalfEdge *he = &he_base[i];

    int u = mftlHeStartVertex(he_base, i);
    int v = he->vertex_idx;
    assert(v >= 0 && v < n_vert);

    int candidate = v_heads[v];
    int twin_idx = MFTL_ERROR;

    while (candidate != MFTL_ERROR)
    {
      if (he_base[candidate].vertex_idx == u)
      {
        he->twin_idx = candidate;
        he->twin_gid = he_base[candidate].global_id;  // (a rim half-edge keeps its retained twin_gid)
        break;
      }
      candidate = he_next[candidate];
    }
  }

  free(v_heads);
  free(he_next);

  // boundary cache and boundary-vertex out-half-edge convention (plans/16 §2.6)
  MFTLRefreshBoundaryCache(obj);
}

void MFTLConvertGlobalToLocalIndices(MFTL_Object *obj)
{
  // vertices: out_he_idx (a gid here) -> local index
  MFTL_Vertex *v_base = obj->vertices;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex *v = &v_base[i];

    khiter_t k = kh_get(halfedge_map, obj->halfedge_lookup, v->out_he_idx);
    if (k != kh_end(obj->halfedge_lookup))
    {
      v->out_he_idx = kh_value(obj->halfedge_lookup, k);
    }
    else
    {
      // the packed out half-edge is not stored here (typical for ghosts): repaired below
      v->out_he_idx = MFTL_ERROR;
    }
  }

  // half-edges: vertex_idx (a gid here) -> local index
  MFTL_HalfEdge *he_base = obj->half_edges;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &he_base[i];

    khiter_t k = kh_get(vertex_map, obj->vertex_lookup, he->vertex_idx);

    // every stored half-edge must have its target vertex stored
    if (k == kh_end(obj->vertex_lookup))
    {
      fprintf(stderr, "FATAL ERROR: Half-edge %d references missing vertex GID %d\n", he->global_id, he->vertex_idx);
      he->vertex_idx = MFTL_ERROR;
      exit(EXIT_FAILURE);
    }

    int local_v_idx = kh_value(obj->vertex_lookup, k);
    he->vertex_idx = local_v_idx;

    int next_he_idx = he->next_idx;  // already a local index (set when unpacking)

    // A vertex without a stored out half-edge: 'he' points TO it, so he->next starts FROM it.
    if (v_base[local_v_idx].out_he_idx == MFTL_ERROR)
    {
      v_base[local_v_idx].out_he_idx = next_he_idx;
    }

    // twin_idx is not converted here: MFTLRebuildTwins re-pairs topologically.
  }
}

int MFTLCheckMeshIntegrity(MFTL_Object *object)
{
  int ok = MFTL_SUCCESS;
  int MFTL_ID_INVALID = MFTL_ERROR;

  const MFTL_Vertex *const v_base = object->vertices;
  const MFTL_Triangle *const t_base = object->triangles;
  const MFTL_HalfEdge *const he_base = object->half_edges;

  const int n_verts = object->n_vertices;
  const int n_tris = object->n_triangles;
  const int n_hes = object->n_half_edges;

  // vertices: a valid outgoing half-edge
  for (int i = 0; i < n_verts; ++i)
  {
    const MFTL_Vertex *v = &v_base[i];

    if (v->out_he_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Vertex %d has no outgoing half-edge\n", v->global_id);
      ok = MFTL_ERROR;
    }
    else if (v->out_he_idx < 0 || v->out_he_idx >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Vertex %d has out-of-bounds HE index %d (max: %d)\n", v->global_id, v->out_he_idx, n_hes);
      ok = MFTL_ERROR;
    }
  }

  // triangles: a closed 3-cycle of half-edges pointing back to the triangle
  for (int i = 0; i < n_tris; ++i)
  {
    const MFTL_Triangle *tri = &t_base[i];

    if (tri->he_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d has no half-edge\n", tri->global_id);
      ok = MFTL_ERROR;
      continue;
    }
    if (tri->he_idx < 0 || tri->he_idx >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d has OOB half-edge index %d (max: %d)\n", tri->global_id, tri->he_idx, n_hes);
      ok = MFTL_ERROR;
      continue;
    }

    int idx0 = tri->he_idx;

    int idx1 = he_base[idx0].next_idx;
    if (idx1 == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d broken cycle (he->next is invalid)\n", tri->global_id);
      ok = MFTL_ERROR;
      continue;
    }
    if (idx1 < 0 || idx1 >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d broken cycle (he->next is OOB: %d)\n", tri->global_id, idx1);
      ok = MFTL_ERROR;
      continue;
    }

    int idx2 = he_base[idx1].next_idx;
    if (idx2 == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d broken cycle (he->next->next is invalid)\n", tri->global_id);
      ok = MFTL_ERROR;
      continue;
    }
    if (idx2 < 0 || idx2 >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d broken cycle (he->next->next is OOB: %d)\n", tri->global_id, idx2);
      ok = MFTL_ERROR;
      continue;
    }

    if (he_base[idx2].next_idx != idx0)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d half-edge cycle is broken (next->next->next != self)\n", tri->global_id);
      ok = MFTL_ERROR;
    }

    if (he_base[idx0].triangle_idx != (int) i)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d: he0 does not point back to triangle index %d\n", tri->global_id, i);
      ok = MFTL_ERROR;
    }
  }

  // half-edges: bounds, twin state, twin symmetry and vertex consistency
  for (int i = 0; i < n_hes; ++i)
  {
    const MFTL_HalfEdge *he = &he_base[i];

    if (he->vertex_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has INVALID vertex\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (he->vertex_idx < 0 || he->vertex_idx >= n_verts)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has OOB vertex index %d\n", he->global_id, he->vertex_idx);
      ok = MFTL_ERROR;
    }

    if (he->triangle_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has INVALID triangle\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (he->triangle_idx < 0 || he->triangle_idx >= n_tris)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has OOB triangle index %d\n", he->global_id, he->triangle_idx);
      ok = MFTL_ERROR;
    }

    if (he->next_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has INVALID next\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (he->next_idx < 0 || he->next_idx >= n_hes)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has OOB next index %d\n", he->global_id, he->next_idx);
      ok = MFTL_ERROR;
    }

    if (MFTLHeIsUnresolved(he))
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has no twin and no REMOTE/BOUNDARY state\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (MFTLHeIsBoundary(he) && !object->declared_open)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d is BOUNDARY on a body not declared open\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (he->twin_idx != MFTL_ID_INVALID)
    {
      if (he->twin_idx < 0 || he->twin_idx >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d has OOB twin index %d\n", he->global_id, he->twin_idx);
        ok = MFTL_ERROR;
        continue;
      }

      const MFTL_HalfEdge *twin = &he_base[he->twin_idx];

      if (twin->twin_idx != i)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d twin relation is not symmetric\n", he->global_id);
        ok = MFTL_ERROR;
      }

      // origin(he) == target(twin) and target(he) == origin(twin); every hop bounds-checked
      int he_next = he->next_idx;

      int he_prev = he_base[he_next].next_idx;
      if (he_prev < 0 || he_prev >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d cycle broken at 2nd hop (OOB)\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      int v_start_idx = he_base[he_prev].vertex_idx;
      if (v_start_idx < 0 || v_start_idx >= n_verts)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d start vertex is OOB\n", he->global_id);
        ok = MFTL_ERROR;
      }

      int twin_next = twin->next_idx;
      if (twin_next < 0 || twin_next >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB next index\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      int twin_prev = he_base[twin_next].next_idx;
      if (twin_prev < 0 || twin_prev >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB next->next index\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      int v_twin_start_idx = he_base[twin_prev].vertex_idx;
      if (v_twin_start_idx < 0 || v_twin_start_idx >= n_verts)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB start vertex\n", he->global_id);
        ok = MFTL_ERROR;
      }

      int v_twin_target_idx = twin->vertex_idx;
      int v_target_idx = he->vertex_idx;

      if (v_start_idx == v_twin_target_idx && v_target_idx == v_twin_start_idx)
      {
      }
      else
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d twin does not have consistent vertices\n", he->global_id);
        ok = MFTL_ERROR;
      }
    }
  }

  // success is silent (the check runs inside MFTL_ASSERTs of hot collective routines such as
  // MFTLObjectVolume; failures are reported above)
  return ok;
}

/**
 * @brief Flattened connectivity (see mftl_topology.h): per triangle the half-edge targets and
 * the stored neighbours, and the vertex -> triangle CSR arrays.
 */
void MFTLFlattenConnectivity(MFTL_Object *obj)
{
  assert(obj != MFTL_NULL);

  int n_tris = obj->n_triangles;
  int n_verts = obj->n_vertices;

  if (n_tris < 1 || n_verts < 1) return;

  free(obj->flat_tri_verts);
  free(obj->flat_tri_neighbors);
  free(obj->vert_tris);
  free(obj->vert_offsets);
  obj->flat_tri_verts = (int *) malloc(3 * n_tris * sizeof(int));
  obj->flat_tri_neighbors = (int *) malloc(3 * n_tris * sizeof(int));
  obj->vert_tris = (int *) malloc(3 * n_tris * sizeof(int));
  obj->vert_offsets = (int *) calloc(n_verts + 1, sizeof(int));

  assert(obj->flat_tri_verts && obj->flat_tri_neighbors && obj->vert_tris && obj->vert_offsets);

  MFTL_Triangle *tri_base = obj->triangles;

  int *restrict flat_verts = obj->flat_tri_verts;
  int *restrict flat_neigh = obj->flat_tri_neighbors;
  int *restrict v_offsets = obj->vert_offsets;
  int *restrict v_tris = obj->vert_tris;

  for (int i = 0; i < n_tris; ++i)
  {
    MFTL_Triangle *tri = &tri_base[i];
    MFTL_HalfEdge *he[3];
    MFTLGetTriangleHalfedges(obj, tri, he);

    int base_idx = 3 * i;

    for (int j = 0; j < 3; ++j)
    {
      // target of half-edge j (the corners in a rotation of MFTLGetTriangleVertices' order)
      int v_idx = he[j]->vertex_idx;
      flat_verts[base_idx + j] = v_idx;

      // triangle across half-edge j; -1 without a stored twin
      int neighbor_idx = -1;
      if (MFTLHeIsLocal(he[j]))
      {
        MFTL_Triangle *adj_tri = MFTLHeAdjacentTriangle(obj, he[j]);
        neighbor_idx = (int) (adj_tri - tri_base);
      }
      flat_neigh[base_idx + j] = neighbor_idx;

      v_offsets[v_idx]++;  // valence count
    }
  }

  // CSR: exclusive prefix sum of the counts, fill (advancing the offsets), shift back
  int sum = 0;
  for (int i = 0; i < n_verts; ++i)
  {
    int count = v_offsets[i];
    v_offsets[i] = sum;
    sum += count;
  }
  v_offsets[n_verts] = sum;

  int total_indices = 3 * n_tris;

  for (int i = 0; i < total_indices; ++i)
  {
    int v_idx = flat_verts[i];
    int t_idx = i / 3;

    int pos = v_offsets[v_idx];
    v_tris[pos] = t_idx;

    v_offsets[v_idx]++;
  }

  for (int i = n_verts; i > 0; --i)
  {
    v_offsets[i] = v_offsets[i - 1];
  }
  v_offsets[0] = 0;
}

int MFTLBuildTopologyFromTriples(MFTL_Object* obj, int nV, const double positions[][3], int nF, const int triples[][3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(positions != MFTL_NULL && triples != MFTL_NULL);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLBuildTopologyFromTriples: object must be empty");
  MFTL_Domain* domain = obj->domain;
  // rank-local builder, cannot canonicalise (collective): the winding state is unknown until
  // the next wrap-sensitive collective operation (MFTLObjectEnsureWinding)
  obj->wrap_valid = 0;

  MFTL_RESERVE_ADDITIONAL(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, nV, MFTL_Vertex);
  MFTL_RESERVE_ADDITIONAL(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, nF, MFTL_Triangle);
  MFTL_RESERVE_ADDITIONAL(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 3 * nF, MFTL_HalfEdge);

  obj->n_vertices = nV;
  obj->n_triangles = nF;
  obj->n_half_edges = 3 * nF;

  for (int i = 0; i < nV; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    v->global_id = i;
    v->owner_proc = domain->rank;
    v->is_ghost = MFTL_FALSE;
    memcpy(v->pos, positions[i], 3 * sizeof(double));
    v->out_he_idx = MFTL_ERROR;
    v->flags = MFTL_FLAG_NONE;
    v->img[0] = v->img[1] = v->img[2] = 0;  // callers lift continuous input
  }

  for (int t = 0; t < nF; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    tri->flags = MFTL_FLAG_NONE;
    tri->global_id = t;
    tri->owner_proc = domain->rank;
    tri->is_ghost = MFTL_FALSE;
    tri->he_idx = 3 * t;

    for (int j = 0; j < 3; ++j)
    {
      int a = triples[t][j];
      int b = triples[t][(j + 1) % 3];  // half-edge target = next vertex

      MFTL_HalfEdge* he = &obj->half_edges[3 * t + j];
      he->global_id = 3 * t + j;
      he->triangle_idx = t;
      he->vertex_idx = b;
      he->next_idx = 3 * t + ((j + 1) % 3);
      he->twin_idx = MFTL_ERROR;
      he->twin_gid = MFTL_TWIN_UNKNOWN;
      he->boundary = MFTL_FALSE;
      he->flags = 0;

      obj->vertices[a].out_he_idx = 3 * t + j;
    }
  }

  obj->next_vertex_gid = nV;
  obj->next_triangle_gid = nF;
  obj->next_halfedge_gid = 3 * nF;

  // id maps (gid -> local index), needed by twin discovery / validation
  for (int i = 0; i < nV; ++i)
  {
    int ret;
    khiter_t k = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
    kh_value(obj->vertex_lookup, k) = i;
  }
  for (int t = 0; t < nF; ++t)
  {
    int ret;
    khiter_t k = kh_put(triangle_map, obj->triangle_lookup, obj->triangles[t].global_id, &ret);
    kh_value(obj->triangle_lookup, k) = t;
  }
  for (int h = 0; h < 3 * nF; ++h)
  {
    int ret;
    khiter_t k = kh_put(halfedge_map, obj->halfedge_lookup, obj->half_edges[h].global_id, &ret);
    kh_value(obj->halfedge_lookup, k) = h;
  }
  obj->n_local_vertices = nV;
  obj->n_local_triangles = nF;
  obj->n_local_half_edges = 3 * nF;

  // manifold / orientation check, then pairing and the declared-class policy (plans/16 §2.2)
  if (MFTLInternalCountDirectedDuplicates(obj, 1) > 0)
  {
    fprintf(stderr, "MFTLBuildTopologyFromTriples: non-manifold edge or inconsistent orientation\n");
    return MFTL_ERROR;
  }
  MFTLRebuildTwins(obj);
  int n_open = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (!MFTLHeIsLocal(&obj->half_edges[i])) n_open++;
  if (n_open > 0)
  {
    if (!obj->declared_open)
    {
      fprintf(stderr, "MFTLBuildTopologyFromTriples: %d unmatched half-edge(s) on a body not declared open\n", n_open);
      return MFTL_ERROR;
    }
    int(*ab)[2] = malloc((size_t) n_open * sizeof(*ab));
    int* idx = malloc((size_t) n_open * sizeof(int));
    int* tag = malloc((size_t) n_open * sizeof(int));
    int k = 0;
    for (int i = 0; i < obj->n_half_edges; ++i)
    {
      if (MFTLHeIsLocal(&obj->half_edges[i])) continue;
      ab[k][0] = MFTLHeOrigin(obj, &obj->half_edges[i])->global_id;
      ab[k][1] = MFTLHeTarget(obj, &obj->half_edges[i])->global_id;
      idx[k++] = i;
    }
    int rc = MFTLInternalTagBoundaryLoops(n_open, (const int(*)[2]) ab, tag);
    if (rc == MFTL_SUCCESS)
      for (k = 0; k < n_open; ++k) MFTLHeSetBoundary(&obj->half_edges[idx[k]], tag[k]);
    free(ab);
    free(idx);
    free(tag);
    if (rc != MFTL_SUCCESS)
    {
      fprintf(stderr, "MFTLBuildTopologyFromTriples: boundary does not form simple loops (bow-tie vertex)\n");
      return MFTL_ERROR;
    }
    MFTLRefreshBoundaryCache(obj);
  }
  MFTLFlattenConnectivity(obj);
  if (MFTLCheckTwinStatesLocal(obj, 0, 1) != 0)
  {
    fprintf(stderr, "MFTLBuildTopologyFromTriples: twin-state validation failed\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/**
 * @brief Breadth-first vertex/triangle gathering (see mftl_topology.h). Two frontier queues
 * with one shared capacity are swapped every step; MFTL_FLAG_VISITED marks gathered entities
 * and is cleared at the end.
 */
void MFTLGetNeighborhood(MFTL_Object *restrict obj, int *restrict seed_indices, int n_seeds, bool use_vertex_seeds, int steps,
                         MFTL_Neighborhood *restrict out_result)
{
  if (!obj->flat_tri_verts || !obj->vert_tris) MFTLFlattenConnectivity(obj);

  MFTL_Vertex *restrict const vert_base = obj->vertices;
  MFTL_Triangle *restrict const tri_base = obj->triangles;

  const int *restrict const flat_tri_verts = obj->flat_tri_verts;
  const int *restrict const vert_offsets = obj->vert_offsets;
  const int *restrict const vert_tris = obj->vert_tris;

  out_result->n_verts = 0;
  out_result->cap_verts = 0;
  out_result->vert_indices = MFTL_NULL;
  out_result->n_tris = 0;
  out_result->cap_tris = 0;
  out_result->tri_indices = MFTL_NULL;

  // current / next frontier; one capacity for BOTH queues so they can be swapped
  int cap_q = (n_seeds * 6 < MFTL_INIT_CAPACITY) ? MFTL_INIT_CAPACITY : n_seeds * 6;

  int *q_curr = (int *) malloc(cap_q * sizeof(int));
  int *q_next = (int *) malloc(cap_q * sizeof(int));

  int n_curr = 0;
  int n_next = 0;

  // does q_curr hold vertices (true) or triangles?
  bool is_expanding_vertices = use_vertex_seeds;

  for (int i = 0; i < n_seeds; ++i)
  {
    int idx = seed_indices[i];

    if (use_vertex_seeds)
    {
      MFTL_Vertex *v = &vert_base[idx];
      if (v->flags & MFTL_FLAG_VISITED) continue;
      v->flags |= MFTL_FLAG_VISITED;

      MFTL_ENSURE_CAPACITY(out_result->vert_indices, &out_result->n_verts, &out_result->cap_verts, 1, int);
      out_result->vert_indices[out_result->n_verts++] = idx;
    }
    else
    {
      MFTL_Triangle *t = &tri_base[idx];
      if (t->flags & MFTL_FLAG_VISITED) continue;
      t->flags |= MFTL_FLAG_VISITED;

      MFTL_ENSURE_CAPACITY(out_result->tri_indices, &out_result->n_tris, &out_result->cap_tris, 1, int);
      out_result->tri_indices[out_result->n_tris++] = idx;
    }

    q_curr[n_curr++] = idx;  // cap_q >= n_seeds
  }

  for (int s = 0; s < steps; ++s)
  {
    if (n_curr == 0) break;
    n_next = 0;

    for (int i = 0; i < n_curr; ++i)
    {
      int idx = q_curr[i];

      if (is_expanding_vertices)
      {
        // vertex -> incident triangles (CSR)
        int start = vert_offsets[idx];
        int end = vert_offsets[idx + 1];

        for (int k = start; k < end; ++k)
        {
          int t_idx = vert_tris[k];
          MFTL_Triangle *t = &tri_base[t_idx];

          if (!(t->flags & MFTL_FLAG_VISITED))
          {
            t->flags |= MFTL_FLAG_VISITED;

            MFTL_ENSURE_CAPACITY(out_result->tri_indices, &out_result->n_tris, &out_result->cap_tris, 1, int);
            out_result->tri_indices[out_result->n_tris++] = t_idx;

            if (n_next >= cap_q)
            {
              // both queues grow together (identical capacities)
              cap_q *= 2;
              q_next = (int *) realloc(q_next, cap_q * sizeof(int));
              q_curr = (int *) realloc(q_curr, cap_q * sizeof(int));
            }
            q_next[n_next++] = t_idx;
          }
        }
      }
      else
      {
        // triangle -> corners (flat_tri_verts)
        int base = 3 * idx;
        for (int k = 0; k < 3; ++k)
        {
          int v_idx = flat_tri_verts[base + k];
          MFTL_Vertex *v = &vert_base[v_idx];

          if (!(v->flags & MFTL_FLAG_VISITED))
          {
            v->flags |= MFTL_FLAG_VISITED;

            MFTL_ENSURE_CAPACITY(out_result->vert_indices, &out_result->n_verts, &out_result->cap_verts, 1, int);
            out_result->vert_indices[out_result->n_verts++] = v_idx;

            if (n_next >= cap_q)
            {
              cap_q *= 2;
              q_next = (int *) realloc(q_next, cap_q * sizeof(int));
              q_curr = (int *) realloc(q_curr, cap_q * sizeof(int));
            }
            q_next[n_next++] = v_idx;
          }
        }
      }
    }

    // swap the frontiers (equal capacities) and toggle the expansion type
    int *temp = q_curr;
    q_curr = q_next;
    q_next = temp;

    n_curr = n_next;
    is_expanding_vertices = !is_expanding_vertices;
  }

  free(q_curr);
  free(q_next);

  // clear the marks of the gathered entities only
  for (int i = 0; i < out_result->n_verts; i++) obj->vertices[out_result->vert_indices[i]].flags &= ~MFTL_FLAG_VISITED;
  for (int i = 0; i < out_result->n_tris; i++) obj->triangles[out_result->tri_indices[i]].flags &= ~MFTL_FLAG_VISITED;
}