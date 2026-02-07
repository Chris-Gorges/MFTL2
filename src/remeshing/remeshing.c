/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:54:07
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 19:41:02
 */

// #include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

// void ActionBufferPush(ActionBuffer *buf, MFTL_RemeshAction action)
// {
//   MFTL_ENSURE_CAPACITY(buf->data, &buf->count, &buf->capacity, 1, MFTL_RemeshAction);
//   buf->data[buf->count++] = action;
// }

// // -----------------------------------------------------------------------------
// // MAIN REMESHING FUNCTION
// // -----------------------------------------------------------------------------
// void MFTL_ParallelRemesh(MFTL_Ensemble *ens, MFTL_Object *obj)
// {
//   int my_rank = ens->domain.rank;

//   // --- STEP 1: PREPARE BUFFERS ---
//   // We need one buffer for each neighbor to store the commands we will send.
//   // Map neighbor rank -> ActionBuffer
//   // (Simplification: simpler to just use an array of size nproc, or a hash map if nproc is huge)
//   ActionBuffer *send_buffers = calloc(ens->domain.nproc, sizeof(ActionBuffer));

//   // --- STEP 2: BOUNDARY PASS (The Owner-Arbiter Loop) ---
//   // Iterate all half-edges to find partition boundaries
//   for (mftl_int_t i = 0; i < obj->n_half_edges; i++)
//   {
//     MFTL_HalfEdge *he = &obj->half_edges[i];
//     if (he->flags & MFTL_FLAG_DEAD) continue;  // Skip deleted

//     MFTL_Vertex *v_origin = MFTLHeOrigin(obj, he);
//     MFTL_Vertex *v_target = MFTLHeTarget(obj, he);
//     MFTL_Triangle *tri = MFTLHeTriangle(obj, he);

//     // Check if this is a "Shared Boundary Edge"
//     // A shared edge exists if the triangle is local, but the twin is ghost/remote.
//     // OR if both vertices are on the boundary.

//     bool is_shared = (v_origin->is_ghost || v_target->is_ghost);
//     if (!is_shared) continue;  // Skip interior edges for now

//     // Determine the "Neighbor Rank" sharing this edge.
//     // If v_target is ghost, its owner is the neighbor.
//     // Note: For robust logic, we need to know exactly which rank shares this specific edge.
//     int neighbor_rank = (v_target->is_ghost) ? v_target->owner_proc : v_origin->owner_proc;

//     // --- ARBITRATION ---
//     if (MFTLIsEdgeOwner(my_rank, neighbor_rank))
//     {
//       // I AM THE OWNER. I decide.

//       double len = MFTLGetEdgeLength(obj, he);
//       double target_len = 1.0;  // Placeholder: Get from metric field

//       if (len > 1.5 * target_len)
//       {
//         // --- DECISION: SPLIT ---

//         // 1. Calculate New Geometry
//         double new_pos[3];
//         // Midpoint (simple)
//         for (int k = 0; k < 3; k++) new_pos[k] = 0.5 * (v_origin->pos[k] + v_target->pos[k]);

//         // 2. Generate Deterministic IDs for the new entities
//         // We use our local counters, but we must reserve them globally?
//         // BETTER: Owner generates them. Neighbor adopts them.
//         mftl_int_t new_v_gid = obj->next_vertex_gid++;
//         mftl_int_t new_e_gid = obj->next_halfedge_gid++;

//         // 3. Record Action
//         MFTL_RemeshAction action;
//         action.type = MFTL_OP_SPLIT;
//         action.edge_gid = he->global_id;
//         action.new_vert_gid = new_v_gid;
//         action.new_edge_gid = new_e_gid;
//         memcpy(action.new_pos, new_pos, 3 * sizeof(double));

//         // 4. Buffer for Neighbor
//         ActionBufferPush(&send_buffers[neighbor_rank], action);

//         // 5. Apply Locally immediately (or defer, but immediate is easier for state)
//         MFTL_ApplySplit(obj, he, new_pos, new_v_gid, new_e_gid);
//       }
//       else if (len < 0.5 * target_len)
//       {
//         // --- DECISION: COLLAPSE ---
//         // Similar logic...
//         // Only collapse if topologically safe (Link Condition check required!)
//       }
//     }
//     else
//     {
//       // I AM NOT THE OWNER.
//       // Do nothing. Wait for instructions.
//     }
//   }

//   // --- STEP 3: SYNCHRONIZATION (Exchange Instructions) ---
//   // This requires a custom Alltoall or Neighbor_alltoall implementation
//   // to exchange the ActionBuffers.

//   // ... [MPI Exchange Logic: Send counts, Allocate Recv buffers, Send Data] ...
//   // Let's assume we now have `recv_buffers` from neighbors.

//   // --- STEP 4: REPLAY (The Follower Loop) ---
//   for (int n = 0; n < ens->domain.num_neighbors; n++)
//   {
//     int neighbor = ens->domain.neighbors[n];
//     ActionBuffer *cmds = &recv_buffers[neighbor];  // Psuedo-code

//     for (size_t k = 0; k < cmds->count; k++)
//     {
//       MFTL_RemeshAction act = cmds->data[k];

//       // 1. Find the Local Edge by Global ID
//       khiter_t iter = kh_get(halfedge_map, obj->halfedge_lookup, act.edge_gid);
//       if (iter == kh_end(obj->halfedge_lookup)) continue;  // Error or ghost drift
//       mftl_int_t local_he_idx = kh_value(obj->halfedge_lookup, iter);
//       MFTL_HalfEdge *he = &obj->half_edges[local_he_idx];

//       // 2. Execute blindly
//       if (act.type == MFTL_OP_SPLIT)
//       {
//         MFTL_ApplySplit(obj, he, act.new_pos, act.new_vert_gid, act.new_edge_gid);
//       }
//       // ... handle other types
//     }
//   }

//   // --- STEP 5: INTERIOR PASS ---
//   // Now that the boundary is "Fixed" and "Synced", we treat boundary vertices as locked.
//   // Iterate all edges again. If neither endpoint is boundary/ghost, remesh freely.

//   // Cleanup buffers...
// }

// /**
//  * @brief Splits a half-edge (and its twin) at a given position.
//  * * Logic:
//  * 1. Create new vertex M.
//  * 2. Create new half-edges.
//  * 3. Re-link next/prev pointers.
//  * 4. Split the triangles (1 triangle becomes 2).
//  * * @param obj The mesh object.
//  * @param he The half-edge to split.
//  * @param new_pos Coordinates for the new vertex.
//  * @param new_v_gid Forced Global ID for the new vertex (from Owner).
//  * @param new_he_gid Forced Global ID for the new edge (from Owner).
//  */
// void MFTL_ApplySplit(MFTL_Object *obj, MFTL_HalfEdge *he, double new_pos[3], mftl_int_t new_v_gid, mftl_int_t new_he_gid)
// {
//   // 1. Collect neighborhood pointers BEFORE modifying anything
//   MFTL_HalfEdge *twin = MFTLHeTwin(obj, he);
//   MFTL_Vertex *v_origin = MFTLHeOrigin(obj, he);
//   MFTL_Vertex *v_target = MFTLHeTarget(obj, he);
//   MFTL_Triangle *tri = MFTLHeTriangle(obj, he);
//   MFTL_Triangle *tri_twin = (twin) ? MFTLHeTriangle(obj, twin) : NULL;

//   // 2. Create New Vertex (M)
//   mftl_int_t m_idx = obj->n_vertices++;
//   // (Ensure capacity checks here!)
//   MFTL_Vertex *m = &obj->vertices[m_idx];
//   memcpy(m->pos, new_pos, 3 * sizeof(double));
//   m->global_id = new_v_gid;
//   m->owner_proc = obj->domain->rank;  // Or inherit from parent edge?
//   // Actually, if this is a boundary split, ownership depends on the Arbiter rule.
//   // If I am the Arbiter/Owner of the edge, I own the new vertex.

//   // 3. Create New Half-Edges
//   // We need new HEs to connect M->v_target and v_target->M?
//   // Splitting an edge in a triangle (A,B,C) where edge is AB:
//   // AB becomes AM -> MB.
//   // We need tobisect the triangle C-A-B into C-A-M and C-M-B.

//   // This requires allocating:
//   // - 1 New Vertex
//   // - 3 New HalfEdges (one for the split edge, two for the new internal edge)
//   // - 1 New Triangle

//   // Implementation detail:
//   // This is complex to write out fully in a snippet without a visual reference,
//   // but the key is that YOU MUST UPDATE THE LOOKUP TABLES.

//   kh_put(vertex_map, obj->vertex_lookup, new_v_gid, &ret);
//   // ... map value to m_idx

//   // ... [Topological rewiring] ...

//   // 4. Handle the Twin (if it exists)
//   // If twin is ghost (on other proc), we technically don't see the other triangle.
//   // BUT we must split the 'twin' half-edge so that the topology remains consistent
//   // at the boundary. The 'twin' will point to M now.
// }

// Allocators that return INDEX
static mftl_int_t new_vertex(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);
  mftl_int_t idx = obj->n_vertices++;

  obj->vertices[idx].out_he_idx = MFTL_ERROR;
  return idx;
}

static mftl_int_t new_triangle(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  mftl_int_t idx = obj->n_triangles++;

  obj->triangles[idx].he_idx = MFTL_ERROR;
  return idx;
}

static mftl_int_t new_halfedge(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 1, MFTL_HalfEdge);
  mftl_int_t idx = obj->n_half_edges++;

  obj->half_edges[idx].twin_idx = MFTL_ERROR;
  obj->half_edges[idx].next_idx = MFTL_ERROR;
  obj->half_edges[idx].vertex_idx = MFTL_ERROR;
  obj->half_edges[idx].triangle_idx = MFTL_ERROR;
  return idx;
}

/**
 * @brief Splits a half-edge (and its twin) by inserting a new vertex at the midpoint.
 * * Topologically, this transforms 2 triangles into 4 triangles (for interior edges)
 * or 1 triangle into 2 triangles (for boundary edges).
 *
 * @param obj The mesh object.
 * @param he_ab_idx The index of the half-edge to split (A -> B).
 * @return The index of the newly created vertex M.
 */
/*
      C                   C
     / \                 /|\
    /   \      ->       / | \
   A-----B             A--M--B
    \   /               \ | /
     \ /                 \|/
      D                   D
*/
mftl_int_t MFTLSplitEdge(MFTL_Object *obj, mftl_int_t he_ab_idx)
{
  // --- 1. Gather Initial Context ---
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  mftl_int_t he_ba_idx = he_ab->twin_idx;

  // Vertices of the edge
  mftl_int_t he_next = obj->half_edges[he_ab_idx].next_idx;
  mftl_int_t he_prev = obj->half_edges[he_next].next_idx;
  mftl_int_t v_a_idx = obj->half_edges[he_prev].vertex_idx;
  mftl_int_t v_b_idx = he_ab->vertex_idx;

  // --- 2. Create the New Vertex M ---
  mftl_int_t v_m_idx = new_vertex(obj);
  MFTL_Vertex *v_m = &obj->vertices[v_m_idx];
  MFTL_Vertex *v_a = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b = &obj->vertices[v_b_idx];

  for (int i = 0; i < 3; i++)
  {
    v_m->pos[i] = 0.5 * (v_a->pos[i] + v_b->pos[i]);
  }
  v_m->owner_proc = obj->domain->rank;  // change this later because in parallel this might not be true
  v_m->is_ghost = MFTL_FALSE;  // change this later because in parallel this might not be true

  // assign global id
  v_m->global_id = obj->next_vertex_gid++;  // change this later because in parallel this might not be true

  // --- 3. Split Upper Triangle (T1: ABC) ---
  // We transform T1(A,B,C) into T1(A,M,C) and T2(M,B,C)

  mftl_int_t t1_idx = he_ab->triangle_idx;
  mftl_int_t he_bc_idx = he_ab->next_idx;
  mftl_int_t he_ca_idx = obj->half_edges[he_bc_idx].next_idx;
  mftl_int_t v_c_idx = obj->half_edges[he_bc_idx].vertex_idx;

  mftl_int_t t2_idx = new_triangle(obj);
  mftl_int_t he_mb_idx = new_halfedge(obj);
  mftl_int_t he_mc_idx = new_halfedge(obj);
  mftl_int_t he_cm_idx = new_halfedge(obj);

  // Update T1 (Left: A-M-C)
  obj->triangles[t1_idx].he_idx = he_ab_idx;  // A->M

  // HE A->M (Reuse he_ab)
  obj->half_edges[he_ab_idx].vertex_idx = v_m_idx;
  obj->half_edges[he_ab_idx].next_idx = he_mc_idx;
  // twin_idx for he_am will be set in the bottom half logic
  obj->half_edges[he_ab_idx].twin_idx = MFTL_ERROR;

  // HE M->C
  MFTL_HalfEdge *he_mc = &obj->half_edges[he_mc_idx];
  he_mc->vertex_idx = v_c_idx;
  he_mc->triangle_idx = t1_idx;
  he_mc->next_idx = he_ca_idx;
  he_mc->twin_idx = he_cm_idx;
  he_mc->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

  // HE C->A (Fix next pointer)
  obj->half_edges[he_ca_idx].next_idx = he_ab_idx;

  // Update T2 (Right: M-B-C)
  obj->triangles[t2_idx].he_idx = he_mb_idx;
  obj->triangles[t2_idx].global_id = obj->next_triangle_gid++;  // change this later because in parallel this might not be true
  obj->triangles[t2_idx].owner_proc = obj->domain->rank;  // change this later because in parallel this might not be true
  obj->triangles[t2_idx].is_ghost = MFTL_FALSE;  // change this later because in parallel this might not be true

  // HE M->B
  MFTL_HalfEdge *he_mb = &obj->half_edges[he_mb_idx];
  he_mb->vertex_idx = v_b_idx;
  he_mb->triangle_idx = t2_idx;
  he_mb->next_idx = he_bc_idx;
  // twin_idx for he_mb will be set in the bottom half logic
  he_mb->twin_idx = MFTL_ERROR;
  he_mb->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

  // HE B->C
  obj->half_edges[he_bc_idx].triangle_idx = t2_idx;
  obj->half_edges[he_bc_idx].next_idx = he_cm_idx;

  // HE C->M
  MFTL_HalfEdge *he_cm = &obj->half_edges[he_cm_idx];
  he_cm->vertex_idx = v_m_idx;
  he_cm->triangle_idx = t2_idx;
  he_cm->next_idx = he_mb_idx;
  he_cm->twin_idx = he_mc_idx;
  he_cm->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

  // Mark the split segments as visited so the remeshing loop doesn't split them again
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_VISITED;
  obj->half_edges[he_mb_idx].flags |= MFTL_FLAG_VISITED;

  // --- 4. Split Bottom Triangle (T3: ABD) if not boundary ---
  if (he_ba_idx != MFTL_ERROR)
  {
    mftl_int_t t3_idx = obj->half_edges[he_ba_idx].triangle_idx;
    mftl_int_t he_ad_idx = obj->half_edges[he_ba_idx].next_idx;
    mftl_int_t he_db_idx = obj->half_edges[he_ad_idx].next_idx;
    mftl_int_t v_d_idx = obj->half_edges[he_ad_idx].vertex_idx;

    mftl_int_t t4_idx = new_triangle(obj);
    mftl_int_t he_ma_idx = new_halfedge(obj);
    mftl_int_t he_md_idx = new_halfedge(obj);
    mftl_int_t he_dm_idx = new_halfedge(obj);

    // Update T3 (Right-Bottom: B-M-D)
    obj->triangles[t3_idx].he_idx = he_ba_idx;  // B->M

    // HE B->M (Reuse he_ba)
    obj->half_edges[he_ba_idx].vertex_idx = v_m_idx;
    obj->half_edges[he_ba_idx].next_idx = he_md_idx;
    obj->half_edges[he_ba_idx].twin_idx = he_mb_idx;  // Link to M->B
    obj->half_edges[he_mb_idx].twin_idx = he_ba_idx;  // Back-link

    // HE M->D
    MFTL_HalfEdge *he_md = &obj->half_edges[he_md_idx];
    he_md->vertex_idx = v_d_idx;
    he_md->triangle_idx = t3_idx;
    he_md->next_idx = he_db_idx;
    he_md->twin_idx = he_dm_idx;
    he_md->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

    // HE D->B
    obj->half_edges[he_db_idx].next_idx = he_ba_idx;

    // Update T4 (Left-Bottom: M-A-D)
    obj->triangles[t4_idx].he_idx = he_ma_idx;
    obj->triangles[t4_idx].global_id = obj->next_triangle_gid++;  // change this later because in parallel this might not be true
    obj->triangles[t4_idx].owner_proc = obj->domain->rank;  // change this later because in parallel this might not be true
    obj->triangles[t4_idx].is_ghost = MFTL_FALSE;  // change this later because in parallel this might not be true

    // HE M->A
    MFTL_HalfEdge *he_ma = &obj->half_edges[he_ma_idx];
    he_ma->vertex_idx = v_a_idx;
    he_ma->triangle_idx = t4_idx;
    he_ma->next_idx = he_ad_idx;
    he_ma->twin_idx = he_ab_idx;  // Link to A->M
    obj->half_edges[he_ab_idx].twin_idx = he_ma_idx;  // Back-link
    he_ma->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

    // HE A->D
    obj->half_edges[he_ad_idx].triangle_idx = t4_idx;
    obj->half_edges[he_ad_idx].next_idx = he_dm_idx;

    // HE D->M
    MFTL_HalfEdge *he_dm = &obj->half_edges[he_dm_idx];
    he_dm->vertex_idx = v_m_idx;
    he_dm->triangle_idx = t4_idx;
    he_dm->next_idx = he_ma_idx;
    he_dm->twin_idx = he_md_idx;
    he_dm->global_id = obj->next_halfedge_gid++;  // change this later because in parallel this might not be true

    // Mark the split segments as visited so the remeshing loop doesn't split them again
    obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_VISITED;
    obj->half_edges[he_ma_idx].flags |= MFTL_FLAG_VISITED;
  }

  // --- 5. Set Vertex Outgoing Edges ---
  // These must be valid, non-deleted outgoing edges
  obj->vertices[v_m_idx].out_he_idx = he_mb_idx;
  obj->vertices[v_a_idx].out_he_idx = he_ab_idx;
  obj->vertices[v_b_idx].out_he_idx = he_bc_idx;

  return v_m_idx;
}

// // Check if collapsing he (A->B) is topologically valid
// static bool is_collapse_legal(MFTL_Object *obj, mftl_int_t he_idx)
// {
//   // 1. Boundary check (don't collapse boundary vertices into interior unless careful)
//   // 2. Link Condition:
//   //    Get 1-ring of A. Get 1-ring of B.
//   //    Count duplicates. Should be exactly 2 (C and D) for internal edge.
//   return true;  // Placeholder
// }

// mftl_int_t MFTLCollapseEdge(MFTL_Object *obj, mftl_int_t he_idx)
// {
//   if (!is_collapse_legal(obj, he_idx)) return MFTL_ERROR;

//   // 1. Resolve Indices
//   MFTL_HalfEdge *he_ab = &obj->half_edges[he_idx];
//   mftl_int_t v_a_idx = MFTLHeOrigin(obj, he_ab)->vertex_idx;  // We delete A
//   mftl_int_t v_b_idx = he_ab->vertex_idx;  // We keep B

//   mftl_int_t he_ba_idx = he_ab->twin_idx;

//   // 2. Re-point A's neighborhood to B
//   // Iterate all incoming half-edges to A
//   MFTL_HalfEdge *curr = &obj->half_edges[obj->vertices[v_a_idx].out_he_idx];

//   // We need to iterate the "Umbrella" of A.
//   // For every HE starting at A, we find its twin (pointing to A).
//   // Wait, simpler: Iterate A's outgoing half-edges.
//   // For each outgoing he (A->X), he->twin is (X->A).
//   // Change X->A to X->B.

//   // CAUTION: Iterating while modifying topology is tricky.
//   // Collect all incoming edges to A first?
//   // In Half-Edge, we rotate around A.

//   mftl_int_t start_he = obj->vertices[v_a_idx].out_he_idx;
//   mftl_int_t curr_he = start_he;

//   do
//   {
//     MFTL_HalfEdge *he = &obj->half_edges[curr_he];
//     mftl_int_t twin_idx = he->twin_idx;

//     if (twin_idx != -1)
//     {
//       // Twin points TO A. Make it point TO B.
//       obj->half_edges[twin_idx].vertex_idx = v_b_idx;
//     }

//     // Move to next spoke
//     if (he->prev_idx == -1) break;  // Should not happen in closed mesh
//     mftl_int_t prev_he = he->prev_idx;
//     if (obj->half_edges[prev_he].twin_idx == -1) break;  // Boundary stop

//     curr_he = obj->half_edges[prev_he].twin_idx;
//   } while (curr_he != start_he);

//   // 3. Delete the degenerate triangles (A,B,C) and (B,A,D)
//   // Mark associated edges/faces as DEAD.

//   // (A,B,C) is he_ab->triangle_idx
//   obj->triangles[he_ab->triangle_idx].flags |= MFTL_FLAG_DEAD;
//   // (B,A,D) is obj->half_edges[he_ba_idx].triangle_idx
//   if (he_ba_idx != -1) obj->triangles[obj->half_edges[he_ba_idx].triangle_idx].flags |= MFTL_FLAG_DEAD;

//   // Mark edges A->B and B->A as DEAD
//   obj->half_edges[he_idx].flags |= MFTL_FLAG_DEAD;
//   if (he_ba_idx != -1) obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_DEAD;

//   // Mark vertex A as DEAD
//   obj->vertices[v_a_idx].flags |= MFTL_FLAG_DEAD;

//   // 4. Update B's connectivity
//   // B might point to the deleted edge B->A. Update B->out_he_idx to something safe.
//   // Find a neighbor of B that isn't A.

//   // 5. Glue the "flaps"
//   // The edges AC and BC are now effectively parallel and identical.
//   // You must merge the half-edge AC with BC.
//   // This is the specific logic of "Collapsing":
//   // The twin of (C->A) was (A->C). Now (C->A) points to B (C->B).
//   // There is already a (C->B). We have duplicate edges.
//   // We must remove the "inner" duplicate edges and link the "outer" ones.

//   return v_b_idx;
// }

static inline int SplitApproved(MFTL_Object *obj, mftl_int_t he_idx, MFTL_RemeshParameters *params)
{
  int approved = MFTL_FALSE;
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  if (he->flags & MFTL_FLAG_VISITED) return MFTL_FALSE;

  // Check if edge is too long
  double len_sq = MFTLGetEdgeLengthSquared(obj, he);
  if (len_sq > params->max_edge_length_sq)
  {
    approved = MFTL_TRUE;
  }

  return approved;
}

void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  for (mftl_int_t i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];

    // --- 0. Snapshot Initial Counts ---
    // We use these to identify exactly which elements are "new" after the pass.
    mftl_int_t start_n_vertices = obj->n_vertices;
    mftl_int_t start_n_triangles = obj->n_triangles;
    mftl_int_t start_n_halfedges = obj->n_half_edges;

    // Reset visited flags
    for (mftl_int_t j = 0; j < obj->n_half_edges; ++j)
    {
      obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
    }

    // 1. SPLIT
    // Iterate active edges (loop over HEs, process if index < twin_index)
    // Note: Iterate BACKWARDS or use a separate list, because splitting adds new edges
    // that you don't want to split again in this pass.
    for (mftl_int_t j = obj->n_half_edges - 1; j >= 0; j--)
    {
      if (SplitApproved(obj, j, params))
      {
        MFTLSplitEdge(obj, j);
      }
    }

    // // 2. COLLAPSE
    // // Use a while loop because collapsing changes indices drastically
    // bool collapsed_something = true;
    // while (collapsed_something)
    // {
    //   collapsed_something = false;
    //   for (mftl_int_t i = 0; i < obj->n_half_edges; i++)
    //   {
    //     if (obj->half_edges[i].flags & MFTL_FLAG_DEAD) continue;

    //     // Check Length < min
    //     double len = MFTLGetEdgeLength(obj, &obj->half_edges[i]);
    //     if (len < params->min_edge_length)
    //     {
    //       if (MFTLCollapseEdge(obj, i) != MFTL_ERROR)
    //       {
    //         collapsed_something = true;
    //         // Don't break, continue pass? Or restart?
    //         // Restarting is safer but slower.
    //       }
    //     }
    //   }
    // }

    // 3. SWAP (Valence Optimization)
    // Flip edges to target valence 6

    // 4. SMOOTH (Tangential)
    // Simple Laplacian smoothing to improve quality

    // --- BATCH UPDATE HASH MAPS (The Optimization) ---
    // We only iterate over the elements added during this remeshing pass.

    // A. Vertices
    if (obj->n_vertices > start_n_vertices)
    {
      // 1. Resize once to avoid multiple rehashes
      kh_resize(vertex_map, obj->vertex_lookup, obj->n_vertices);

      // 2. Insert only the new vertices
      for (mftl_int_t v = start_n_vertices; v < obj->n_vertices; ++v)
      {
        // // Skip if dead (relevant if collapse is active)
        // if (obj->vertices[v].flags & MFTL_FLAG_DEAD) continue;

        int ret;
        mftl_int_t gid = obj->vertices[v].global_id;
        khiter_t iter = kh_put(vertex_map, obj->vertex_lookup, gid, &ret);
        kh_value(obj->vertex_lookup, iter) = v;
      }
    }

    // B. Triangles
    if (obj->n_triangles > start_n_triangles)
    {
      kh_resize(triangle_map, obj->triangle_lookup, obj->n_triangles);

      for (mftl_int_t t = start_n_triangles; t < obj->n_triangles; ++t)
      {
        // if (obj->triangles[t].flags & MFTL_FLAG_DEAD) continue;

        int ret;
        mftl_int_t gid = obj->triangles[t].global_id;
        khiter_t iter = kh_put(triangle_map, obj->triangle_lookup, gid, &ret);
        kh_value(obj->triangle_lookup, iter) = t;
      }
    }

    // C. Half-Edges (Optional - only if your library relies on Global ID lookups for HEs)
    // Most half-edge ops rely on local indices, so this might be unnecessary overhead.
    // TODO: Check if this is actually needed. Currently we only need the halfedge lookup after redistribution.
    if (obj->halfedge_lookup && obj->n_half_edges > start_n_halfedges)
    {
      kh_resize(halfedge_map, obj->halfedge_lookup, obj->n_half_edges);

      for (mftl_int_t h = start_n_halfedges; h < obj->n_half_edges; ++h)
      {
        // if (obj->half_edges[h].flags & MFTL_FLAG_DEAD) continue;

        int ret;
        mftl_int_t gid = obj->half_edges[h].global_id;
        khiter_t iter = kh_put(halfedge_map, obj->halfedge_lookup, gid, &ret);
        kh_value(obj->halfedge_lookup, iter) = h;
      }
    }
  }
}
