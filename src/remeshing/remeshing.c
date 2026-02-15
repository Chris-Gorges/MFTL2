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
  obj->vertices[idx].flags = MFTL_FLAG_NONE;
  return idx;
}

static mftl_int_t new_triangle(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  mftl_int_t idx = obj->n_triangles++;

  obj->triangles[idx].he_idx = MFTL_ERROR;
  obj->triangles[idx].flags = MFTL_FLAG_NONE;
  obj->triangles[idx].owner_proc = obj->domain->rank;
  obj->triangles[idx].is_ghost = MFTL_FALSE;
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
  obj->half_edges[idx].flags = MFTL_FLAG_NONE;
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

  obj->vertices[v_m_idx].flags |= MFTL_FLAG_VISITED;

  return v_m_idx;
}

// Helper to link twins and bypass a dead edge
// Connects OuterTwin1 <-> OuterTwin2
void glue_wings(MFTL_Object *o, mftl_int_t edge1, mftl_int_t edge2)
{
  mftl_int_t t1 = o->half_edges[edge1].twin_idx;
  mftl_int_t t2 = o->half_edges[edge2].twin_idx;

  if (t1 != MFTL_ERROR)
  {
    o->half_edges[t1].twin_idx = t2;
    // Safety: Update vertex out pointer
    mftl_int_t v = o->half_edges[t1].vertex_idx;  // Origin of t2
    // Actually origin of t1 is target of edge1.
    // Target of edge1 is the vertex 'C'.
    o->vertices[o->half_edges[edge1].vertex_idx].out_he_idx = t1;
  }
  if (t2 != MFTL_ERROR)
  {
    o->half_edges[t2].twin_idx = t1;
    // Safety: Update vertex out pointer
    o->vertices[o->half_edges[edge2].vertex_idx].out_he_idx = t2;
  }
};

void MFTLCollapseEdge(MFTL_Object *obj, mftl_int_t he_ab_idx)
{
  // --- 1. Gather Context ---
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  mftl_int_t he_ba_idx = he_ab->twin_idx;

  // Identify Vertices
  mftl_int_t prev_he = obj->half_edges[obj->half_edges[he_ab_idx].next_idx].next_idx;
  mftl_int_t v_a_idx = obj->half_edges[prev_he].vertex_idx;
  mftl_int_t v_b_idx = he_ab->vertex_idx;

  // SAFETY: If A or B are already dead (from previous collapses in this pass), ABORT.
  if ((obj->vertices[v_a_idx].flags & MFTL_FLAG_DEAD) || (obj->vertices[v_b_idx].flags & MFTL_FLAG_DEAD))
  {
    printf("Collapse: Vertex %lld or %lld is already dead. ABORTING.\n", (long long) v_a_idx, (long long) v_b_idx);
    return;
  }
  // SAFETY: If edge is already dead
  if (he_ab->flags & MFTL_FLAG_DEAD)
  {
    printf("Collapse: Edge %lld is already dead. ABORTING.\n", (long long) he_ab_idx);
    return;
  }

  // --- 2. Re-point A's Neighborhood to B (Dynamic Loop) ---
  // Instead of a fixed array, we circulate dynamically.
  // We also save one "survivor" edge index to help repair B later.

  mftl_int_t safe_survivor_he = MFTL_ERROR;

  MFTL_HalfEdge *start_he = MFTLVOutHe(obj, &obj->vertices[v_a_idx]);
  MFTL_HalfEdge *curr = start_he;

  // 2a. Rewind to start of fan (Essential for boundaries)
  while (true)
  {
    MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
    MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);
    if (!cw_neighbor || cw_neighbor == start_he) break;
    curr = cw_neighbor;
  }
  start_he = curr;  // Reset start

  // 2b. Replay and Update
  do
  {
    // Capture the outgoing edge index for later safety check
    mftl_int_t he_out_idx = (mftl_int_t) (curr - obj->half_edges);

    // Save a candidate for B's new out_he
    if (he_out_idx != he_ab_idx)
    {
      safe_survivor_he = he_out_idx;
    }

    // UPDATE THE INCOMING EDGE
    // Safety: Verify Next link exists
    if (curr->next_idx != MFTL_ERROR)
    {
      mftl_int_t next_id = curr->next_idx;
      // We need Prev(curr). In a triangle, Prev = Next->Next.
      // We must check if Next->Next is valid.
      mftl_int_t next_next_id = obj->half_edges[next_id].next_idx;

      if (next_next_id != MFTL_ERROR)
      {
        mftl_int_t he_in_idx = next_next_id;
        // CRITICAL CHECK: Does this edge actually point to A?
        // If we are on a boundary loop > 3, Next->Next might point to a different vertex.
        if (obj->half_edges[he_in_idx].vertex_idx == v_a_idx)
        {
          obj->half_edges[he_in_idx].vertex_idx = v_b_idx;
        }
        else
        {
          // We failed to find the incoming edge via Next->Next.
          // This implies a boundary loop or topology issue.
          // We rely on the Fallback Sweep (if present) or manual search?
          // For now, avoid corruption is the priority.
        }
      }
    }

    // Advance
    MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
    if (!twin) break;  // Boundary hit
    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  // --- 2.1 Fallback Safety Sweep (Robustness) ---
  // In case the 1-ring traversal missed any incoming edges (e.g. boundary weirdness),
  // we do a full sweep to ensure NO edge points to A.
  for (mftl_int_t k = 0; k < obj->n_half_edges; ++k)
  {
    if (obj->half_edges[k].flags & MFTL_FLAG_DEAD) continue;
    if (obj->half_edges[k].vertex_idx == v_a_idx)
    {
      obj->half_edges[k].vertex_idx = v_b_idx;
    }
  }

  // --- 3. Remove Degenerate Faces (Gluing) ---
  // (Copy your existing code for Step 3 exactly as is)
  mftl_int_t t1_idx = he_ab->triangle_idx;
  obj->triangles[t1_idx].flags |= MFTL_FLAG_DEAD;
  mftl_int_t he_bc = he_ab->next_idx;
  mftl_int_t he_ca = obj->half_edges[he_bc].next_idx;
  glue_wings(obj, he_ca, he_bc);
  obj->half_edges[he_bc].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ca].flags |= MFTL_FLAG_DEAD;

  if (he_ba_idx != MFTL_ERROR)
  {
    mftl_int_t t2_idx = obj->half_edges[he_ba_idx].triangle_idx;
    obj->triangles[t2_idx].flags |= MFTL_FLAG_DEAD;
    mftl_int_t he_ad = obj->half_edges[he_ba_idx].next_idx;
    mftl_int_t he_db = obj->half_edges[he_ad].next_idx;
    glue_wings(obj, he_ad, he_db);
    obj->half_edges[he_ad].flags |= MFTL_FLAG_DEAD;
    obj->half_edges[he_db].flags |= MFTL_FLAG_DEAD;
  }

  // --- 4. Final Cleanup ---
  obj->vertices[v_a_idx].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_DEAD;
  if (he_ba_idx != MFTL_ERROR) obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_DEAD;

  // Update B's position
  for (int k = 0; k < 3; ++k)
  {
    obj->vertices[v_b_idx].pos[k] = 0.5 * (obj->vertices[v_b_idx].pos[k] + obj->vertices[v_a_idx].pos[k]);
  }

  // --- SAFE POINTER UPDATE (Using the survivor we found in Step 2) ---
  mftl_int_t check_he = obj->vertices[v_b_idx].out_he_idx;

  if (check_he == MFTL_ERROR || (obj->half_edges[check_he].flags & MFTL_FLAG_DEAD))
  {
    // We prioritize the survivor found in Step 2.
    // Since Step 2 updated all neighbors, 'safe_survivor_he' is now an edge B->X.
    if (safe_survivor_he != MFTL_ERROR && !(obj->half_edges[safe_survivor_he].flags & MFTL_FLAG_DEAD))
    {
      obj->vertices[v_b_idx].out_he_idx = safe_survivor_he;
    }
    // Fallback: Try the twin of the collapsed wing (he_ca)
    else if (obj->half_edges[he_ca].twin_idx != MFTL_ERROR)
    {
      obj->vertices[v_b_idx].out_he_idx = obj->half_edges[he_ca].twin_idx;
    }
  }

  // --- MARK SURVIVOR ---
  obj->vertices[v_b_idx].flags |= MFTL_FLAG_VISITED;
}

// Helper to check if vertex is on boundary (uses the 1-ring logic implicitly)
static bool IsBoundaryVertex(const MFTL_Object *obj, mftl_int_t v_idx)
{
  // Fast check: just see if the circulation hits a null twin
  MFTL_HalfEdge *start = &obj->half_edges[obj->vertices[v_idx].out_he_idx];
  MFTL_HalfEdge *curr = start;
  do
  {
    if (curr->twin_idx == MFTL_ERROR) return MFTL_TRUE;
    curr = &obj->half_edges[obj->half_edges[curr->twin_idx].next_idx];
  } while (curr != start);
  return MFTL_FALSE;
}

int CollapseApproved(MFTL_Object *obj, mftl_int_t he_idx, MFTL_RemeshParameters *params)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // 1. Status Checks
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // --- CHECK VERTEX FLAGS ---
  // Prevent chained collapses by checking if vertices are already dirty
  mftl_int_t prev_he = obj->half_edges[obj->half_edges[he_idx].next_idx].next_idx;
  mftl_int_t v_a_idx = obj->half_edges[prev_he].vertex_idx;
  mftl_int_t v_b_idx = he->vertex_idx;

  // If either vertex is DEAD (removed) or VISITED (involved in another op), skip.
  if (obj->vertices[v_a_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // 1.1 Check if the edge is a boundary edge
  // if (he->twin_idx == MFTL_ERROR) return MFTL_FALSE;  // Generally avoid collapsing boundary edges unless robust
  // Boundary Edge Detection
  bool is_boundary_edge = (he->twin_idx == MFTL_ERROR);

  // 2. Length Check (Squared distance optimization)
  double len_sq = MFTLGetEdgeLengthSquared(obj, he);
  if (len_sq > params->min_edge_length_sq) return MFTL_FALSE;

  // 3. Topology: Link Condition (The Intersection Test)
  // We need to ensure A and B share exactly 2 neighbors (for interior edge).

  MFTL_Vertex *v_a_ptr = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b_ptr = &obj->vertices[v_b_idx];

  // Get Rings
  MFTL_Vertex *ring_a[13];
  MFTL_Vertex *ring_b[13];
  mftl_int_t count_a = MFTLGetVertexOneRingVertices(obj, v_a_ptr, ring_a, 13, MFTL_FALSE);  // Check if MFTL_FALSE OR TRUE IS BETTER HERE
  mftl_int_t count_b = MFTLGetVertexOneRingVertices(obj, v_b_ptr, ring_b, 13, MFTL_FALSE);  // Check if MFTL_FALSE OR TRUE IS BETTER HERE

  // Count Common Neighbors
  int common = 0;
  for (int i = 0; i < count_a; ++i)
  {
    for (int j = 0; j < count_b; ++j)
    {
      // Compare IDs (safer than pointers if realloc occurred, though safe here)
      if (ring_a[i] == ring_b[j]) common++;
    }
  }

  // Threshold: Interior edge must share exactly 2 neighbors.
  // (If boundary edge logic is supported, threshold is 1)
  // Threshold
  if (is_boundary_edge)
  {
    if (common != 1) return MFTL_FALSE;
  }
  else
  {
    if (common != 2) return MFTL_FALSE;
  }

  // 4. Boundary Geometry Check
  // Don't collapse boundary vertex into interior vertex
  bool a_bound = IsBoundaryVertex(obj, v_a_idx);
  bool b_bound = IsBoundaryVertex(obj, v_b_idx);
  if (a_bound && !b_bound) return MFTL_FALSE;

  return MFTL_TRUE;
}

static inline int SplitApproved(MFTL_Object *obj, mftl_int_t he_idx, MFTL_RemeshParameters *params)
{
  int approved = MFTL_FALSE;
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // 1. Status Checks (Edge AND Vertices)
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // Check vertices to ensure we don't split an area that is already being modified
  mftl_int_t next = obj->half_edges[he_idx].next_idx;
  mftl_int_t prev = obj->half_edges[next].next_idx;
  mftl_int_t v_a = obj->half_edges[prev].vertex_idx;
  mftl_int_t v_b = he->vertex_idx;

  if (obj->vertices[v_a].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // Check if edge is too long
  double len_sq = MFTLGetEdgeLengthSquared(obj, he);
  if (len_sq > params->max_edge_length_sq)
  {
    approved = MFTL_TRUE;
  }

  return approved;
}

/**
 * @brief Finalizes the object state after remeshing.
 * Smartly chooses between a fast "append-only" update (if only splits happened)
 * or a full "compaction & rebuild" (if collapses happened).
 * * @param obj The mesh object.
 * @param start_num_v The number of vertices BEFORE the remeshing pass.
 * @param start_num_t The number of triangles BEFORE the remeshing pass.
 * @param start_num_h The number of half-edges BEFORE the remeshing pass.
 * @param collapses_occurred Set to true if ANY collapse operation was performed.
 */
void MFTLFinalizeRemeshing(MFTL_Object *obj, mftl_int_t start_num_v, mftl_int_t start_num_t, mftl_int_t start_num_h, bool collapses_occurred)
{
  // Reset visited flags
  for (mftl_int_t j = 0; j < obj->n_half_edges; ++j)
  {
    obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
  }

  for (mftl_int_t j = 0; j < obj->n_vertices; ++j)
  {
    obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;
  }

  // ---------------------------------------------------------
  // PATH A: FAST APPEND (Only Splits - No Compaction Needed)
  // ---------------------------------------------------------
  if (!collapses_occurred)
  {
    // 1. Vertices (Only process the new ones)
    if (obj->n_vertices > start_num_v)
    {
      // Optimal: Resize once to fit the final count
      kh_resize(vertex_map, obj->vertex_lookup, obj->n_vertices);

      for (mftl_int_t i = start_num_v; i < obj->n_vertices; ++i)
      {
        int ret;
        khiter_t iter = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
        kh_value(obj->vertex_lookup, iter) = i;
      }
    }

    // 2. Triangles
    if (obj->n_triangles > start_num_t)
    {
      kh_resize(triangle_map, obj->triangle_lookup, obj->n_triangles);

      for (mftl_int_t i = start_num_t; i < obj->n_triangles; ++i)
      {
        int ret;
        khiter_t iter = kh_put(triangle_map, obj->triangle_lookup, obj->triangles[i].global_id, &ret);
        kh_value(obj->triangle_lookup, iter) = i;
      }
    }

    // 3. Half-Edges (If used)
    if (obj->halfedge_lookup && obj->n_half_edges > start_num_h)
    {
      kh_resize(halfedge_map, obj->halfedge_lookup, obj->n_half_edges);

      for (mftl_int_t i = start_num_h; i < obj->n_half_edges; ++i)
      {
        int ret;
        khiter_t iter = kh_put(halfedge_map, obj->halfedge_lookup, obj->half_edges[i].global_id, &ret);
        kh_value(obj->halfedge_lookup, iter) = i;
      }
    }

    return;  // DONE! (Fast exit)
  }

  // ---------------------------------------------------------
  // PATH B: FULL COMPACTION (Collapses Happened - Move Data)
  // ---------------------------------------------------------

  // Safety Check: ensure no live half-edge points to a dead vertex
  for (mftl_int_t i = 0; i < obj->n_half_edges; ++i)
  {
    if (obj->half_edges[i].flags & MFTL_FLAG_DEAD) continue;
    mftl_int_t v_idx = obj->half_edges[i].vertex_idx;
    if (obj->vertices[v_idx].flags & MFTL_FLAG_DEAD)
    {
      printf("Error: Live HalfEdge %d points to Dead Vertex %d\n", i, v_idx);
      assert(0);
    }
  }

  // 1. Setup Maps
  mftl_int_t *v_map = malloc(obj->n_vertices * sizeof(mftl_int_t));
  mftl_int_t *t_map = malloc(obj->n_triangles * sizeof(mftl_int_t));
  mftl_int_t *h_map = malloc(obj->n_half_edges * sizeof(mftl_int_t));

  assert(v_map && t_map && h_map);

  // 2. Compact Arrays (Standard Logic)
  // --- Vertices ---
  mftl_int_t new_v_count = 0;
  for (mftl_int_t i = 0; i < obj->n_vertices; i++)
  {
    if (obj->vertices[i].flags & MFTL_FLAG_DEAD)
    {
      v_map[i] = MFTL_ERROR;
    }
    else
    {
      v_map[i] = new_v_count;
      if (i != new_v_count) obj->vertices[new_v_count] = obj->vertices[i];
      new_v_count++;
    }
  }
  obj->n_vertices = new_v_count;

  // --- Triangles ---
  mftl_int_t new_t_count = 0;
  for (mftl_int_t i = 0; i < obj->n_triangles; i++)
  {
    if (obj->triangles[i].flags & MFTL_FLAG_DEAD)
    {
      t_map[i] = MFTL_ERROR;
    }
    else
    {
      t_map[i] = new_t_count;
      if (i != new_t_count) obj->triangles[new_t_count] = obj->triangles[i];
      new_t_count++;
    }
  }
  obj->n_triangles = new_t_count;

  // --- Half-Edges ---
  mftl_int_t new_h_count = 0;
  for (mftl_int_t i = 0; i < obj->n_half_edges; i++)
  {
    if (obj->half_edges[i].flags & MFTL_FLAG_DEAD)
    {
      h_map[i] = MFTL_ERROR;
    }
    else
    {
      h_map[i] = new_h_count;
      if (i != new_h_count) obj->half_edges[new_h_count] = obj->half_edges[i];
      new_h_count++;
    }
  }
  obj->n_half_edges = new_h_count;

  // 3. Relink Pointers
  // Vertices
  for (mftl_int_t i = 0; i < obj->n_vertices; i++)
  {
    assert(obj->vertices[i].out_he_idx != MFTL_ERROR);
    obj->vertices[i].out_he_idx = h_map[obj->vertices[i].out_he_idx];
  }
  // Triangles
  for (mftl_int_t i = 0; i < obj->n_triangles; i++)
  {
    assert(obj->triangles[i].he_idx != MFTL_ERROR);
    if (h_map[obj->triangles[i].he_idx] == MFTL_ERROR)
    {
      int old_he = obj->triangles[i].he_idx;
      printf("CRITICAL ERROR: Live Triangle %d (GID %lld) points to Dead HalfEdge %d (GID %lld)\n", i, (long long) obj->triangles[i].global_id, old_he,
             (long long) obj->half_edges[old_he].global_id);
      assert(h_map[obj->triangles[i].he_idx] != MFTL_ERROR);
    }
    obj->triangles[i].he_idx = h_map[obj->triangles[i].he_idx];
  }
  // Half-Edges
  for (mftl_int_t i = 0; i < obj->n_half_edges; i++)
  {
    MFTL_HalfEdge *he = &obj->half_edges[i];

    assert(he->next_idx != MFTL_ERROR);
    he->next_idx = h_map[he->next_idx];
    if (he->twin_idx != MFTL_ERROR)
      he->twin_idx = h_map[he->twin_idx];
    else
      he->twin_idx = MFTL_ERROR;
    assert(he->vertex_idx != MFTL_ERROR);
    he->vertex_idx = v_map[he->vertex_idx];
    assert(he->triangle_idx != MFTL_ERROR);
    he->triangle_idx = t_map[he->triangle_idx];
  }

  // 4. Free Maps
  free(v_map);
  free(t_map);
  free(h_map);

  // 5. Rebuild Hash Maps (The Full Reset)
  // Since indices shifted arbitrarily, we clear and fill from scratch.

  // Vertices
  kh_clear(vertex_map, obj->vertex_lookup);
  kh_resize(vertex_map, obj->vertex_lookup, obj->n_vertices);
  for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
  {
    int ret;
    khiter_t iter = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
    kh_value(obj->vertex_lookup, iter) = i;
  }

  // Triangles
  kh_clear(triangle_map, obj->triangle_lookup);
  kh_resize(triangle_map, obj->triangle_lookup, obj->n_triangles);
  for (mftl_int_t i = 0; i < obj->n_triangles; ++i)
  {
    int ret;
    khiter_t iter = kh_put(triangle_map, obj->triangle_lookup, obj->triangles[i].global_id, &ret);
    kh_value(obj->triangle_lookup, iter) = i;
  }

  // Half-Edges
  if (obj->halfedge_lookup)
  {
    kh_clear(halfedge_map, obj->halfedge_lookup);
    kh_resize(halfedge_map, obj->halfedge_lookup, obj->n_half_edges);
    for (mftl_int_t i = 0; i < obj->n_half_edges; ++i)
    {
      int ret;
      khiter_t iter = kh_put(halfedge_map, obj->halfedge_lookup, obj->half_edges[i].global_id, &ret);
      kh_value(obj->halfedge_lookup, iter) = i;
    }
  }
}

void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  mftl_int_t counter_split = 0;
  mftl_int_t counter_collapse = 0;
  for (int iter = 0; iter < params->iterations; iter++)
  {
    for (mftl_int_t i = 0; i < ens->num_objects; i++)
    {
      MFTL_Object *obj = ens->objects[i];

      // --- 0. Snapshot Initial Counts ---
      // We use these to identify exactly which elements are "new" after the pass.
      mftl_int_t start_n_vertices = obj->n_vertices;
      mftl_int_t start_n_triangles = obj->n_triangles;
      mftl_int_t start_n_halfedges = obj->n_half_edges;

      bool collapses_occurred = MFTL_FALSE;

      // 1. SPLIT
      // Iterate active edges (loop over HEs, process if index < twin_index)
      // Note: Iterate BACKWARDS or use a separate list, because splitting adds new edges
      // that you don't want to split again in this pass.
      for (mftl_int_t j = obj->n_half_edges - 1; j >= 0; j--)
      {
        if (SplitApproved(obj, j, params))
        {
          MFTLSplitEdge(obj, j);
          counter_split++;
        }
      }

      // 2. COLLAPSE
      for (mftl_int_t j = 0; j < obj->n_half_edges; ++j)
      {
        if (CollapseApproved(obj, j, params))
        {
          MFTLCollapseEdge(obj, j);
          collapses_occurred = MFTL_TRUE;
          counter_collapse++;
        }
      }

      // 3. SWAP (Valence Optimization)
      // Flip edges to target valence 6

      // 4. SMOOTH (Tangential)
      // Simple Laplacian smoothing to improve quality

      MFTLFinalizeRemeshing(obj, start_n_vertices, start_n_triangles, start_n_halfedges, collapses_occurred);
    }
  }
  printf("Split %d times\n", counter_split);
  printf("Collapsed %d times\n", counter_collapse);
}
