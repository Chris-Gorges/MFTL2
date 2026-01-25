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
