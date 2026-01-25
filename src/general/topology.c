/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:54:07
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 19:41:02
 */

#include "mftl_topology.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include <stdlib.h>  // for malloc/free
#include <stdio.h>  // for fprintf

// Helper to get the start vertex of a half-edge
// (Assumes triangular mesh where prev == next->next)
static inline mftl_int_t get_start_vertex(MFTL_HalfEdge *he_base, mftl_int_t he_idx)
{
  mftl_int_t next_idx = he_base[he_idx].next_idx;
  mftl_int_t prev_idx = he_base[next_idx].next_idx;
  return he_base[prev_idx].vertex_idx;
}

void MFTLRebuildTwins(MFTL_Object *obj)
{
  const mftl_int_t n_he = obj->n_half_edges;
  const mftl_int_t n_vert = obj->n_vertices;
  MFTL_HalfEdge *he_base = obj->half_edges;

  // --- 1. Allocation ---
  // v_heads: Maps Vertex Index -> Index of the first Half-Edge starting at this vertex
  mftl_int_t *v_heads = (mftl_int_t *) malloc(sizeof(mftl_int_t) * n_vert);

  // he_next: Links Half-Edges sharing the same Start Vertex
  mftl_int_t *he_next = (mftl_int_t *) malloc(sizeof(mftl_int_t) * n_he);

  assert(v_heads && he_next);

  // Initialize heads to INVALID (-1)
  for (mftl_int_t i = 0; i < n_vert; i++)
  {
    v_heads[i] = MFTL_ERROR;
  }

  // --- 2. Build Adjacency List (Pass 1) ---
  for (mftl_int_t i = 0; i < n_he; i++)
  {
    // Find start vertex of edge i
    mftl_int_t u = get_start_vertex(he_base, i);
    assert(u >= 0 && u < n_vert);

    // Insert 'i' at the front of the list for vertex 'u'
    he_next[i] = v_heads[u];
    v_heads[u] = i;

    MFTL_HalfEdge *he = &he_base[i];
    he->twin_idx = MFTL_ERROR;
    he->boundary = MFTL_TRUE;
  }

  // --- 3. Find Twins (Pass 2) ---
  for (mftl_int_t i = 0; i < n_he; i++)
  {
    MFTL_HalfEdge *he = &he_base[i];

    mftl_int_t u = get_start_vertex(he_base, i);  // Start
    mftl_int_t v = he->vertex_idx;  // End
    assert(v >= 0 && v < n_vert);

    // We want to find an edge that goes v -> u.
    // Look in the list of edges STARTING at v.
    mftl_int_t candidate = v_heads[v];
    mftl_int_t twin_idx = MFTL_ERROR;

    while (candidate != MFTL_ERROR)
    {
      // Check where this candidate edge ends
      if (he_base[candidate].vertex_idx == u)
      {
        he->twin_idx = candidate;
        he->boundary = MFTL_FALSE;
        break;  // Found it!
      }
      // Move to next edge starting at v
      candidate = he_next[candidate];
    }
  }

  // --- Cleanup ---
  free(v_heads);
  free(he_next);
}

void MFTLConvertGlobalToLocalIndices(MFTL_Object *obj)
{
  // --- 1. Convert Vertex "Out Half-Edge" Indices ---
  MFTL_Vertex *v_base = obj->vertices;
  for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex *v = &v_base[i];

    // Lookup the GID of the outgoing half-edge
    khiter_t k = kh_get(halfedge_map, obj->halfedge_lookup, v->out_he_idx);  // out_he_idx is currently GID
    if (k != kh_end(obj->halfedge_lookup))
    {
      v->out_he_idx = kh_value(obj->halfedge_lookup, k);  // Found! Replace with Local Index
    }
    else
    {
      // The specific half-edge this vertex pointed to wasn't sent to this rank.
      // This is expected for ghost vertices. We will repair this in step 2.
      v->out_he_idx = MFTL_ERROR;
    }
  }

  // --- 2. Convert Half-Edge "Vertex" Indices ---
  MFTL_HalfEdge *he_base = obj->half_edges;
  for (mftl_int_t i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &he_base[i];

    // Lookup the GID of the vertex
    khiter_t k = kh_get(vertex_map, obj->vertex_lookup, he->vertex_idx);  // vertex_idx is currently GID

    // This assertion MUST hold. If we have an edge, we must have its target vertex.
    // assert(k != kh_end(obj->vertex_lookup) && "Half-edge refers to a missing vertex!");
    if (k == kh_end(obj->vertex_lookup))
    {
      fprintf(stderr, "FATAL ERROR: Half-edge %d references missing vertex GID %d\n", he->global_id, he->vertex_idx);
      // Set to error to prevent OOB write, but mesh is broken
      he->vertex_idx = MFTL_ERROR;
      // Force exit or return error code
      exit(EXIT_FAILURE);
    }

    mftl_int_t local_v_idx = kh_value(obj->vertex_lookup, k);
    he->vertex_idx = local_v_idx;  // Replace GID with Local Index

    // --- REPAIR STRATEGY ---
    // Since we process sequentially, we can simply repair the TARGET vertex of the current HE
    // (which is `local_v_idx`).

    mftl_int_t next_he_idx = he->next_idx;  // This is already a Local Index (set during unpack)

    // Check the vertex 'v' that this edge 'he' points TO.
    if (v_base[local_v_idx].out_he_idx == MFTL_ERROR)
    {
      // This vertex had a missing out-edge.
      // Since `he` points TO `local_v_idx`, `he->next` MUST start FROM `local_v_idx`.
      v_base[local_v_idx].out_he_idx = next_he_idx;
    }

    // Note: We do NOT convert he->twin_idx here.
    // We let MFTLRebuildTwins handle that topologically next.
  }
}

mftl_int_t MFTLCheckMeshIntegrity(MFTL_Object *object)
{
  mftl_int_t ok = MFTL_SUCCESS;
  mftl_int_t MFTL_ID_INVALID = MFTL_ERROR;

  // --- 1. Setup Base Pointers for Speed ---
  const MFTL_Vertex *const v_base = object->vertices;
  const MFTL_Triangle *const t_base = object->triangles;
  const MFTL_HalfEdge *const he_base = object->half_edges;

  const mftl_int_t n_verts = object->n_vertices;
  const mftl_int_t n_tris = object->n_triangles;
  const mftl_int_t n_hes = object->n_half_edges;

  // --- Check Vertices ---
  for (mftl_int_t i = 0; i < n_verts; ++i)
  {
    const MFTL_Vertex *v = &v_base[i];

    // Check if index is valid (equivalent to !v->out_he)
    if (v->out_he_idx == MFTL_ID_INVALID)
    {
      fprintf(stderr, "[MeshCheck] Vertex %d has no outgoing half-edge\n", v->global_id);
      ok = MFTL_ERROR;
    }
    else if (v->out_he_idx < 0 || v->out_he_idx >= n_hes)
    {
      // Extra safety check for out-of-bounds indices
      fprintf(stderr, "[MeshCheck] Vertex %d has out-of-bounds HE index %d (max: %d)\n", v->global_id, v->out_he_idx, n_hes);
      ok = MFTL_ERROR;
    }
  }

  // --- Check Triangles ---
  for (mftl_int_t i = 0; i < n_tris; ++i)
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

    // Cycle Check: HE -> Next -> Next -> Next == HE
    mftl_int_t idx0 = tri->he_idx;

    // Check Hop 1
    mftl_int_t idx1 = he_base[idx0].next_idx;
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

    // Check Hop 2
    mftl_int_t idx2 = he_base[idx1].next_idx;
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

    // Check Cycle Closure
    if (he_base[idx2].next_idx != idx0)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d half-edge cycle is broken (next->next->next != self)\n", tri->global_id);
      ok = MFTL_ERROR;
    }

    // Belonging Check: Does the HE point back to this triangle?
    if (he_base[idx0].triangle_idx != (mftl_int_t) i)
    {
      fprintf(stderr, "[MeshCheck] Triangle %d: he0 does not point back to triangle index %d\n", tri->global_id, i);
      ok = MFTL_ERROR;
    }
  }

  // --- Check Half-Edges ---
  for (mftl_int_t i = 0; i < n_hes; ++i)
  {
    const MFTL_HalfEdge *he = &he_base[i];

    // 1. Basic Validity & Bounds
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

    // 2. Twin Consistency
    if (he->twin_idx == MFTL_ID_INVALID && !he->boundary)
    {
      fprintf(stderr, "[MeshCheck] Half-edge %d has no twin (mesh is not closed)\n", he->global_id);
      ok = MFTL_ERROR;
    }
    else if (he->twin_idx != MFTL_ID_INVALID)
    {
      // Bounds check Twin Index
      if (he->twin_idx < 0 || he->twin_idx >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d has OOB twin index %d\n", he->global_id, he->twin_idx);
        ok = MFTL_ERROR;
        continue;  // Cannot proceed with twin checks
      }

      const MFTL_HalfEdge *twin = &he_base[he->twin_idx];

      // A. Symmetry: Twin's twin must be me
      if (twin->twin_idx != i)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d twin relation is not symmetric\n", he->global_id);
        ok = MFTL_ERROR;
      }

      // B. Geometric Consistency
      // Logic: The vertex at the START of 'he' must be the vertex at the END of 'twin'.
      // This requires traversing prev pointers (next->next). We must bounds check every hop!

      // --- Calculate Start of HE (Origin) ---
      mftl_int_t he_next = he->next_idx;
      // Bounds check for next was done above, so safe to access he_base[he_next]

      mftl_int_t he_prev = he_base[he_next].next_idx;
      if (he_prev < 0 || he_prev >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d cycle broken at 2nd hop (OOB)\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      // Get Start Vertex
      mftl_int_t v_start_idx = he_base[he_prev].vertex_idx;
      if (v_start_idx < 0 || v_start_idx >= n_verts)
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d start vertex is OOB\n", he->global_id);
        ok = MFTL_ERROR;
      }

      // --- Calculate Start of Twin ---
      mftl_int_t twin_next = twin->next_idx;
      if (twin_next < 0 || twin_next >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB next index\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      mftl_int_t twin_prev = he_base[twin_next].next_idx;
      if (twin_prev < 0 || twin_prev >= n_hes)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB next->next index\n", he->global_id);
        ok = MFTL_ERROR;
        continue;
      }

      mftl_int_t v_twin_start_idx = he_base[twin_prev].vertex_idx;
      if (v_twin_start_idx < 0 || v_twin_start_idx >= n_verts)
      {
        fprintf(stderr, "[MeshCheck] Twin of %d has OOB start vertex\n", he->global_id);
        ok = MFTL_ERROR;
      }

      // --- Compare ---
      // Target of Twin
      mftl_int_t v_twin_target_idx = twin->vertex_idx;  // Already checked bounds via loop invariant for other edge, but good to be safe

      // Target of HE
      mftl_int_t v_target_idx = he->vertex_idx;

      if (v_start_idx == v_twin_target_idx && v_target_idx == v_twin_start_idx)
      {
        // Correct orientation
      }
      else
      {
        fprintf(stderr, "[MeshCheck] Half-edge %d twin does not have consistent vertices\n", he->global_id);
        ok = MFTL_ERROR;
      }
    }
  }

  if (ok == MFTL_SUCCESS)
  {
    printf("[MeshCheck] Mesh is consistent and closed.\n");
  }
  return ok;
}

/**
 * @brief Builds internal cache-friendly integer arrays for mesh connectivity.
 * Populates: flat_tri_verts, flat_tri_neighbors, vert_offsets, vert_tris.
 * 1. Triangle Maps (flat_tri_*): Direct lookup of vertices and neighbors for a triangle.
 * 2. Vertex Maps (vert_*): Compressed Sparse Row (CSR) structure for lookup of incident triangles for a vertex.
 */
void MFTLFlattenConnectivity(MFTL_Object *obj)
{
  assert(obj != MFTL_NULL);

  mftl_int_t n_tris = obj->n_triangles;
  mftl_int_t n_verts = obj->n_vertices;

  if (n_tris < 1 || n_verts < 1) return;

  // --- Memory Allocation ---
  free(obj->flat_tri_verts);
  free(obj->flat_tri_neighbors);
  free(obj->vert_tris);
  free(obj->vert_offsets);
  obj->flat_tri_verts = (mftl_int_t *) malloc(3 * n_tris * sizeof(mftl_int_t));
  obj->flat_tri_neighbors = (mftl_int_t *) malloc(3 * n_tris * sizeof(mftl_int_t));
  obj->vert_tris = (mftl_int_t *) malloc(3 * n_tris * sizeof(mftl_int_t));
  obj->vert_offsets = (mftl_int_t *) calloc(n_verts + 1, sizeof(mftl_int_t));

  assert(obj->flat_tri_verts && obj->flat_tri_neighbors && obj->vert_tris && obj->vert_offsets);

  MFTL_Triangle *tri_base = obj->triangles;

  // Using restrict to help compiler optimize
  mftl_int_t *restrict flat_verts = obj->flat_tri_verts;
  mftl_int_t *restrict flat_neigh = obj->flat_tri_neighbors;
  mftl_int_t *restrict v_offsets = obj->vert_offsets;
  mftl_int_t *restrict v_tris = obj->vert_tris;

  for (mftl_int_t i = 0; i < n_tris; ++i)
  {
    MFTL_Triangle *tri = &tri_base[i];
    MFTL_HalfEdge *he[3];
    MFTLGetTriangleHalfedges(obj, tri, he);

    mftl_int_t base_idx = 3 * i;

    for (int j = 0; j < 3; ++j)
    {
      // Note: Verify if you need 'origin' or 'target' vertex here.
      // Assuming target (he->vertex_idx) is desired:
      mftl_int_t v_idx = he[j]->vertex_idx;
      flat_verts[base_idx + j] = v_idx;

      // Flatten Neighbors
      mftl_int_t neighbor_idx = -1;
      if (!he[j]->boundary)
      {
        MFTL_Triangle *adj_tri = MFTLHeAdjacentTriangle(obj, he[j]);
        // PTRDIFF: Ensure adj_tri is within the tri_base array range
        neighbor_idx = (mftl_int_t) (adj_tri - tri_base);
      }
      flat_neigh[base_idx + j] = neighbor_idx;

      // Count Valence
      v_offsets[v_idx]++;
    }
  }

  // --- PASS 2: CSR Construction ---

  // Prefix Sum (Convert counts to Start Offsets)
  mftl_int_t sum = 0;
  for (mftl_int_t i = 0; i < n_verts; ++i)
  {
    mftl_int_t count = v_offsets[i];
    v_offsets[i] = sum;
    sum += count;
  }
  v_offsets[n_verts] = sum;

  mftl_int_t total_indices = 3 * n_tris;

  for (mftl_int_t i = 0; i < total_indices; ++i)
  {
    mftl_int_t v_idx = flat_verts[i];
    mftl_int_t t_idx = i / 3;

    mftl_int_t pos = v_offsets[v_idx];
    v_tris[pos] = t_idx;

    v_offsets[v_idx]++;
  }

  // Restore Offsets
  for (mftl_int_t i = n_verts; i > 0; --i)
  {
    v_offsets[i] = v_offsets[i - 1];
  }
  v_offsets[0] = 0;
}

/**
 * @brief Gather the neighborhood of a set of seeds by traversing the mesh topology.
 *
 * @details This function performs a Breadth-First Search (BFS) starting from the provided
 * seed indices (vertices or triangles). It treats the mesh as a bipartite graph
 * where vertices connect to triangles, and triangles connect to vertices.
 * * It alternates expansion types at each step:
 * - If currently holding **Vertices**: It gathers all incident **Triangles**.
 * - If currently holding **Triangles**: It gathers all incident **Vertices**.
 * * This approach is more robust than purely edge-based traversal as it naturally 
 * captures the full "star" of a vertex or the full set of corners of a triangle.
 *
 * ### Optimization
 * The function uses the flattened connectivity arrays (`vert_offsets` / `vert_tris` for V->T 
 * and `flat_tri_verts` for T->V) to ensure maximum cache coherency and avoid pointer chasing.
 * It uses a double-buffered queue system to minimize memory allocation overhead during the loop.
 *
 * @note    The meaning of `steps` is topological hops, not geometric distance:
 * - **1 Step** from a Vertex gathers its incident Triangles (the "Umbrella").
 * - **2 Steps** from a Vertex gathers incident Triangles AND their other Vertices (the full "1-Ring").
 *
 * @param[in]  obj               The mesh object containing the topology.
 * @param[in]  seed_indices      Array of local indices serving as the starting front.
 * @param[in]  n_seeds           Number of indices in `seed_indices`.
 * @param[in]  use_vertex_seeds  Set to `true` if seeds are Vertex indices, `false` for Triangle indices.
 * @param[in]  steps             Number of bipartite hops to expand. 
 * (e.g., 2 steps = V->T->V).
 * @param[out] out_result        Pointer to a user-allocated `MFTL_Neighborhood` struct 
 * where the results will be stored. Arrays are allocated internally 
 * and must be freed by the user (or `MFTLFreeNeighborhood`).
 */
void MFTLGetNeighborhood(MFTL_Object *restrict obj, mftl_int_t *restrict seed_indices, mftl_int_t n_seeds, bool use_vertex_seeds, int steps,
                         MFTL_Neighborhood *restrict out_result)
{
  // --- 1. Validation & Setup ---
  if (!obj->flat_tri_verts || !obj->vert_tris) MFTLFlattenConnectivity(obj);

  // Use local restrictive pointers for compiler optimization
  MFTL_Vertex *restrict const vert_base = obj->vertices;
  MFTL_Triangle *restrict const tri_base = obj->triangles;

  // Connectivity Arrays (Read-Only)
  const mftl_int_t *restrict const flat_tri_verts = obj->flat_tri_verts;
  const mftl_int_t *restrict const vert_offsets = obj->vert_offsets;
  const mftl_int_t *restrict const vert_tris = obj->vert_tris;

  // --- 2. Initialize Result ---
  out_result->n_verts = 0;
  out_result->cap_verts = 0;
  out_result->vert_indices = MFTL_NULL;
  out_result->n_tris = 0;
  out_result->cap_tris = 0;
  out_result->tri_indices = MFTL_NULL;

  // --- 3. Initialize Synchronized Queues ---
  // We use ONE capacity variable for BOTH queues.
  mftl_int_t cap_q = (n_seeds * 6 < MFTL_INIT_CAPACITY) ? MFTL_INIT_CAPACITY : n_seeds * 6;

  mftl_int_t *q_curr = (mftl_int_t *) malloc(cap_q * sizeof(mftl_int_t));
  mftl_int_t *q_next = (mftl_int_t *) malloc(cap_q * sizeof(mftl_int_t));

  mftl_int_t n_curr = 0;
  mftl_int_t n_next = 0;

  // State toggle: Are the IDs in q_curr Vertices or Triangles?
  bool is_expanding_vertices = use_vertex_seeds;

  for (mftl_int_t i = 0; i < n_seeds; ++i)
  {
    mftl_int_t idx = seed_indices[i];

    if (use_vertex_seeds)
    {
      MFTL_Vertex *v = &vert_base[idx];
      if (v->flags & MFTL_FLAG_VISITED) continue;
      v->flags |= MFTL_FLAG_VISITED;  // Mark as visited

      MFTL_ENSURE_CAPACITY(out_result->vert_indices, &out_result->n_verts, &out_result->cap_verts, 1, mftl_int_t);
      out_result->vert_indices[out_result->n_verts++] = idx;
    }
    else
    {
      MFTL_Triangle *t = &tri_base[idx];
      if (t->flags & MFTL_FLAG_VISITED) continue;
      t->flags |= MFTL_FLAG_VISITED;  // Mark as visited

      MFTL_ENSURE_CAPACITY(out_result->tri_indices, &out_result->n_tris, &out_result->cap_tris, 1, mftl_int_t);
      out_result->tri_indices[out_result->n_tris++] = idx;
    }

    // We already ensured capacity for q_curr above
    q_curr[n_curr++] = idx;
  }

  // --- 5. The Unified BFS Loop ---
  for (int s = 0; s < steps; ++s)
  {
    if (n_curr == 0) break;
    n_next = 0;

    for (mftl_int_t i = 0; i < n_curr; ++i)
    {
      mftl_int_t idx = q_curr[i];

      if (is_expanding_vertices)
      {
        // EXPAND: Vertex -> Triangles (using CSR)
        mftl_int_t start = vert_offsets[idx];
        mftl_int_t end = vert_offsets[idx + 1];

        for (mftl_int_t k = start; k < end; ++k)
        {
          mftl_int_t t_idx = vert_tris[k];
          MFTL_Triangle *t = &tri_base[t_idx];

          if (!(t->flags & MFTL_FLAG_VISITED))
          {
            t->flags |= MFTL_FLAG_VISITED;

            // Add to Results
            MFTL_ENSURE_CAPACITY(out_result->tri_indices, &out_result->n_tris, &out_result->cap_tris, 1, mftl_int_t);
            out_result->tri_indices[out_result->n_tris++] = t_idx;

            // Add to Next Queue (Synchronized Growth)
            if (n_next >= cap_q)
            {
              // If one grows, BOTH grow. This keeps capacities identical.
              cap_q *= 2;
              q_next = (mftl_int_t *) realloc(q_next, cap_q * sizeof(mftl_int_t));
              q_curr = (mftl_int_t *) realloc(q_curr, cap_q * sizeof(mftl_int_t));
            }
            q_next[n_next++] = t_idx;
          }
        }
      }
      else
      {
        // EXPAND: Triangle -> Vertices (using Flat Array)
        mftl_int_t base = 3 * idx;
        for (int k = 0; k < 3; ++k)
        {
          mftl_int_t v_idx = flat_tri_verts[base + k];
          MFTL_Vertex *v = &vert_base[v_idx];

          if (!(v->flags & MFTL_FLAG_VISITED))
          {
            v->flags |= MFTL_FLAG_VISITED;

            // Add to Results
            MFTL_ENSURE_CAPACITY(out_result->vert_indices, &out_result->n_verts, &out_result->cap_verts, 1, mftl_int_t);
            out_result->vert_indices[out_result->n_verts++] = v_idx;

            // Add to Next Queue (Synchronized Growth)
            if (n_next >= cap_q)
            {
              cap_q *= 2;
              q_next = (mftl_int_t *) realloc(q_next, cap_q * sizeof(mftl_int_t));
              q_curr = (mftl_int_t *) realloc(q_curr, cap_q * sizeof(mftl_int_t));
            }
            q_next[n_next++] = v_idx;
          }
        }
      }
    }

    // Swap Queues and Toggle Mode
    // Because capacities are strictly synced, we ONLY swap pointers.
    mftl_int_t *temp = q_curr;
    q_curr = q_next;
    q_next = temp;

    n_curr = n_next;
    is_expanding_vertices = !is_expanding_vertices;
  }

  // --- 6. Cleanup ---
  free(q_curr);
  free(q_next);

  // Clear flags (O(k))
  for (mftl_int_t i = 0; i < out_result->n_verts; i++) obj->vertices[out_result->vert_indices[i]].flags &= ~MFTL_FLAG_VISITED;
  for (mftl_int_t i = 0; i < out_result->n_tris; i++) obj->triangles[out_result->tri_indices[i]].flags &= ~MFTL_FLAG_VISITED;
}