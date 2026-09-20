/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:07:46
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-26 10:30:53
 */

#include <assert.h>  // For assert()
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "khash.h"
#include <stdint.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_vectoroperations.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"

/**
 * @brief Performs one level of 1-to-4 triangle subdivision on the object
 *        using khash-based edge lookups for midpoint vertices and twins.
 *     v1                  v1
 *    /  \                /  \
 *   /    \              m1--m2
 *  /      \   ----->    / \/ \
 * v2------v3          v2--m3--v3
 */
static int tessellate_sphere(MFTL_Object *object)
{
  MFTL_Domain *domain = object->domain;

  // --- 1. Snapshot Old Data (Base Pointers) ---
  const size_t old_nv = object->n_vertices;
  const size_t old_nt = object->n_triangles;
  const size_t old_nhe = object->n_half_edges;

  // We treat old arrays as "Read-Only" sources
  const MFTL_Vertex *old_vertices = object->vertices;
  const MFTL_Triangle *old_triangles = object->triangles;
  const MFTL_HalfEdge *old_half_edges = object->half_edges;

  // Calculate number of unique edges (assuming closed mesh)
  const size_t old_nedges = old_nhe / 2;

  // --- 2. Allocate NEW Buffers ---
  const size_t new_nv = old_nv + old_nedges;  // Old Verts + Midpoints
  const size_t new_nt = old_nt * 4;  // 1 Triangle -> 4 Triangles
  const size_t new_nhe = new_nt * 3;  // 3 HEs per Triangle

  MFTL_Vertex *new_vertices = calloc(new_nv, sizeof(MFTL_Vertex));
  MFTL_Triangle *new_triangles = calloc(new_nt, sizeof(MFTL_Triangle));
  MFTL_HalfEdge *new_half_edges = calloc(new_nhe, sizeof(MFTL_HalfEdge));

  if (!new_vertices || !new_triangles || !new_half_edges)
  {
    fprintf(stderr, "[Error] Allocation failed in tessellate_sphere\n");
    return MFTL_ERROR;
  }

  // --- 3. Copy Old Vertices ---
  // The first 'old_nv' vertices in the new array are identical to the old ones.
  // Their indices [0 ... old_nv-1] remain unchanged!
  memcpy(new_vertices, old_vertices, old_nv * sizeof(MFTL_Vertex));

  // Reset outgoing edge indices for copied vertices (we will re-link them in Step 5)
  for (size_t i = 0; i < old_nv; ++i) new_vertices[i].out_he_idx = MFTL_ERROR;

  // --- 4. Create Midpoints ---

  // A temporary map: old_he_index -> new_midpoint_vertex_index
  int *midpoint_map = malloc(old_nhe * sizeof(int));

  // Initialize with -1
  for (size_t i = 0; i < old_nhe; ++i) midpoint_map[i] = MFTL_ERROR;

  int next_midpoint_idx = (int) old_nv;  // Start creating after old verts

  for (size_t i = 0; i < old_nhe; ++i)
  {
    // If we already visited this edge (via its twin), skip.
    if (midpoint_map[i] != MFTL_ERROR) continue;

    // Get the edge vertices
    const MFTL_HalfEdge *he = &old_half_edges[i];
    const MFTL_Vertex *v1 = &old_vertices[he->vertex_idx];  // Target

    // Find Origin (using twin if available, or manual lookup)
    int twin_idx = he->twin_idx;
    const MFTL_Vertex *v0 = &old_vertices[old_half_edges[twin_idx].vertex_idx];

    // Create Midpoint Vertex
    MFTL_Vertex *mid = &new_vertices[next_midpoint_idx];

    // Geometry: Average position
    mid->pos[0] = 0.5 * (v0->pos[0] + v1->pos[0]);
    mid->pos[1] = 0.5 * (v0->pos[1] + v1->pos[1]);
    mid->pos[2] = 0.5 * (v0->pos[2] + v1->pos[2]);

    // Topology Init
    mid->global_id = object->next_vertex_gid++;
    mid->owner_proc = domain->rank;
    mid->is_ghost = MFTL_FALSE;
    mid->out_he_idx = MFTL_ERROR;  // Will set in Step 5

    // Register in Map (Optimization: Map BOTH sides of the edge)
    midpoint_map[i] = next_midpoint_idx;
    midpoint_map[twin_idx] = next_midpoint_idx;

    next_midpoint_idx++;
  }

  // --- 5. Build New Triangles (1 -> 4) ---
  //
  //      v2
  //     /  \
  //   m2 -- m1
  //   / \  / \
  // v0 -- m0 -- v1

  int new_tri_idx = 0;
  int new_he_idx = 0;

  for (size_t i = 0; i < old_nt; ++i)
  {
    const MFTL_Triangle *old_tri = &old_triangles[i];

    // Get Old Half-Edges indices
    int he0_idx = old_tri->he_idx;
    int he1_idx = old_half_edges[he0_idx].next_idx;
    int he2_idx = old_half_edges[he1_idx].next_idx;

    // Get Vertex Indices (Indices in new array match old array)
    // v0 is Origin of he0 (Target of he2)
    int v0_idx = old_half_edges[he2_idx].vertex_idx;
    int v1_idx = old_half_edges[he0_idx].vertex_idx;
    int v2_idx = old_half_edges[he1_idx].vertex_idx;

    // Get Midpoint Indices (Direct Lookup!)
    int m0_idx = midpoint_map[he0_idx];  // on edge v0-v1
    int m1_idx = midpoint_map[he1_idx];  // on edge v1-v2
    int m2_idx = midpoint_map[he2_idx];  // on edge v2-v0

    // Define the 4 new triangles (CCW winding)
    // Each row is a triangle: {Vert A, Vert B, Vert C}
    int sub_tris[4][3] = {
        {v0_idx, m0_idx, m2_idx},  // Corner 0
        {v1_idx, m1_idx, m0_idx},  // Corner 1
        {v2_idx, m2_idx, m1_idx},  // Corner 2
        {m0_idx, m1_idx, m2_idx}  // Center
    };

    for (int j = 0; j < 4; ++j)
    {
      // 1. Setup Triangle
      MFTL_Triangle *tri = &new_triangles[new_tri_idx];
      tri->global_id = object->next_triangle_gid++;
      tri->owner_proc = domain->rank;
      tri->is_ghost = MFTL_FALSE;
      tri->he_idx = new_he_idx;  // Points to first HE created below

      // 2. Setup 3 Half-Edges
      for (int k = 0; k < 3; ++k)
      {
        MFTL_HalfEdge *he = &new_half_edges[new_he_idx + k];
        int current_v_idx = sub_tris[j][k];
        int next_v_idx = sub_tris[j][(k + 1) % 3];

        he->global_id = object->next_halfedge_gid++;

        he->triangle_idx = new_tri_idx;
        he->vertex_idx = next_v_idx;  // HE points TO next vertex

        // Link within triangle
        he->next_idx = new_he_idx + ((k + 1) % 3);

        // Twin: We leave invalid (-1), will rebuild globally later
        he->twin_idx = MFTL_ERROR;
        he->boundary = MFTL_TRUE;

        // 3. Update Vertex Connectivity
        // Ensure the vertex points to this outgoing half-edge
        new_vertices[current_v_idx].out_he_idx = new_he_idx + k;
      }

      new_tri_idx++;
      new_he_idx += 3;
    }
  }

  // --- 6. Swap and Cleanup ---
  free(object->vertices);
  free(object->triangles);
  free(object->half_edges);
  free(midpoint_map);  // Done with the map!

  object->vertices = new_vertices;
  object->triangles = new_triangles;
  object->half_edges = new_half_edges;

  object->n_vertices = new_nv;
  object->n_triangles = new_nt;
  object->n_half_edges = new_nhe;

  object->capacity_vertices = new_nv;
  object->capacity_triangles = new_nt;
  object->capacity_half_edges = new_nhe;

  // --- 7. Rebuild Twins ---
  // Efficiently relinks everything using the geometric keys
  MFTLRebuildTwins(object);

  return MFTL_SUCCESS;
}

/**
 * @brief Generate a unit icosahedron inside an MFTL_Object.
 *
 * Allocates 12 vertices, 20 triangles, and 60 half-edges, then
 * rebuilds twin relationships automatically via hashing.
 *
 * @param object Mesh object (must be initialized).
 * @return int 0 on success.
 */
static int generate_icosahedron(MFTL_Object *object)
{
  MFTL_Domain *domain = object->domain;

  // --- Constants for icosahedron geometry ---
  const double X = 0.525731112119133606;
  const double Z = 0.850650808352039932;

  const double vertices_data[12][3] = {{-X, 0.0, Z}, {X, 0.0, Z},   {-X, 0.0, -Z}, {X, 0.0, -Z}, {0.0, Z, X},  {0.0, Z, -X},
                                       {0.0, -Z, X}, {0.0, -Z, -X}, {Z, X, 0.0},   {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}};

  const size_t triangles_data[20][3] = {{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1}, {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3}, {2, 7, 3},
                                        {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6}, {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5}, {7, 2, 11}};

  // --- Allocate space for icosahedron (12 vertices, 20 faces, 60 halfedges) ---
  MFTL_RESERVE_ADDITIONAL(object->vertices, &object->n_vertices, &object->capacity_vertices, 12, MFTL_Vertex);
  MFTL_RESERVE_ADDITIONAL(object->triangles, &object->n_triangles, &object->capacity_triangles, 20, MFTL_Triangle);
  MFTL_RESERVE_ADDITIONAL(object->half_edges, &object->n_half_edges, &object->capacity_half_edges, 60, MFTL_HalfEdge);

  // The reserve macros hand back realloc'd, UNINITIALIZED memory. Zero every
  // field first — in particular 'flags': garbage flag bits (e.g. a stray
  // MFTL_FLAG_VISITED) are transported to ghost copies on other ranks and
  // corrupt flag-based traversals such as MFTLGetNeighborhood.
  memset(object->vertices, 0, 12 * sizeof(MFTL_Vertex));
  memset(object->triangles, 0, 20 * sizeof(MFTL_Triangle));
  memset(object->half_edges, 0, 60 * sizeof(MFTL_HalfEdge));

  object->n_vertices = 12;
  object->n_triangles = 20;
  object->n_half_edges = 60;
  object->next_vertex_gid = 12;
  object->next_triangle_gid = 20;
  object->next_halfedge_gid = 60;

  // --- Create vertices ---
  for (int i = 0; i < 12; ++i)
  {
    MFTL_Vertex *v = &object->vertices[i];
    v->global_id = i;
    v->owner_proc = domain->rank;
    v->is_ghost = MFTL_FALSE;
    memcpy(v->pos, vertices_data[i], 3 * sizeof(double));
    v->out_he_idx = -1;  // will be set later
    // int ret;
    // khiter_t k = kh_put(vertex_map, object->vertex_lookup, v->global_id, &ret);
    // kh_value(object->vertex_lookup, k) = base_vertex_gid + i;
  }

  // --- Create triangles and half-edges ---
  for (int i = 0; i < 20; ++i)
  {
    MFTL_Triangle *tri = &object->triangles[i];
    tri->global_id = i;
    tri->owner_proc = domain->rank;
    tri->is_ghost = MFTL_FALSE;
    // int ret;
    // khiter_t k = kh_put(triangle_map, object->triangle_lookup, tri->global_id, &ret);
    // kh_value(object->triangle_lookup, k) = base_triangle_gid + i;

    size_t v_idx[3] = {triangles_data[i][0], triangles_data[i][1], triangles_data[i][2]};

    // Each triangle gets 3 half-edges
    MFTL_HalfEdge *he[3];
    for (int j = 0; j < 3; ++j)
    {
      he[j] = &object->half_edges[i * 3 + j];
      he[j]->global_id = i * 3 + j;
      he[j]->triangle_idx = i;
      he[j]->vertex_idx = v_idx[(j + 1) % 3];  // END vertex
      he[j]->twin_idx = -1;  // will be rebuilt
      he[j]->next_idx = -1;  // will be set later

      //   int ret;
      //   khiter_t k = kh_put(halfedge_map, object->halfedge_lookup, he[j]->global_id, &ret);
      //   kh_value(object->halfedge_lookup, k) = base_he_gid + (i * 3 + j);

      // Register outgoing edge for the starting vertex
      MFTL_Vertex *start_v = &object->vertices[v_idx[j]];
      start_v->out_he_idx = i * 3 + j;
    }

    // Link the cycle of half-edges around the triangle
    he[0]->next_idx = i * 3 + 1;
    he[1]->next_idx = i * 3 + 2;
    he[2]->next_idx = i * 3;

    // Store reference in the triangle
    tri->he_idx = i * 3;
  }

  // --- Rebuild all twin relationships in one pass ---
  MFTLRebuildTwins(object);

  return MFTL_SUCCESS;
}

/**
 * @brief Public function to generate a sphere.
 * @param ensemble The ensemble containing the domain context (MPI, Periodicity).
 * @param object The object to generate the sphere on.
 * @param center The center position of the sphere.
 * @param radius The radius of the sphere.
 * @param refinements The number of refinements for the sphere.
 */
int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements)
{
  assert(object);

  // Access the domain for MPI and Physics context
  MFTL_Domain *domain = object->domain;

  if (domain->rank == 0)
  {
    generate_icosahedron(object);
    for (int i = 0; i < refinements; ++i)
    {
      // printf("Performing refinement level %d...\n", i + 1);
      tessellate_sphere(object);
    }

    for (size_t i = 0; i < object->n_vertices; ++i)
    {
      MFTLNormalizeVector(object->vertices[i].pos);
      object->vertices[i].pos[0] = center[0] + radius * object->vertices[i].pos[0];
      object->vertices[i].pos[1] = center[1] + radius * object->vertices[i].pos[1];
      object->vertices[i].pos[2] = center[2] + radius * object->vertices[i].pos[2];
      MFTLApplyPeriodicity(domain, object->vertices[i].pos);
    }
  }

  MPI_Bcast(&object->next_vertex_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&object->next_triangle_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&object->next_halfedge_gid, 1, MPI_INT, 0, domain->comm_world);

  MFTLObjectUpdateGlobalStats(object);

  MFTLFlattenConnectivity(object);

  return MFTL_SUCCESS;
}
