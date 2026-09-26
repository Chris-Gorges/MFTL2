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
#include "mftl_images.h"
#include "mftl_boundary_internal.h"
#include "mftl_phase_set.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

// Allocators that return INDEX
static int new_vertex(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);
  int idx = obj->n_vertices++;

  obj->vertices[idx].out_he_idx = MFTL_ERROR;
  obj->vertices[idx].flags = MFTL_FLAG_NONE;
  obj->vertices[idx].img[0] = obj->vertices[idx].img[1] = obj->vertices[idx].img[2] = 0;  // set by the operator
  return idx;
}

static int new_triangle(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  int idx = obj->n_triangles++;

  obj->triangles[idx].he_idx = MFTL_ERROR;
  obj->triangles[idx].flags = MFTL_FLAG_NONE;
  obj->triangles[idx].owner_proc = obj->domain->rank;
  obj->triangles[idx].is_ghost = MFTL_FALSE;
  obj->triangles[idx].surface_scalar = 0.0;  // set by the operator's mass remap (Stage 15, L7)
  // Stage 12: user data is NOT repartitioned by split/collapse — new triangles
  // from splits carry a zeroed blob; callers that attach data must impose
  // their own split/merge policy (per Stage 12 plan).
  if (obj->data_size > 0)
    obj->triangles[idx].data = calloc(1, obj->data_size);
  else
    obj->triangles[idx].data = MFTL_NULL;
  return idx;
}

static int new_halfedge(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 1, MFTL_HalfEdge);
  int idx = obj->n_half_edges++;

  obj->half_edges[idx].twin_idx = MFTL_ERROR;
  obj->half_edges[idx].twin_gid = MFTL_TWIN_UNKNOWN;  // set by the operator (Stage 16A)
  obj->half_edges[idx].boundary = MFTL_FALSE;
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
int MFTLSplitEdge(MFTL_Object *obj, int he_ab_idx)
{
  // --- 1. Gather Initial Context ---
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  int he_ba_idx = he_ab->twin_idx;
  // Stage 16A: only LOCAL (interior) or BOUNDARY edges are split; a BOUNDARY edge's two halves
  // inherit its state code (tag).
  MFTL_ASSERT(!MFTLHeIsRemote(he_ab) && "MFTLSplitEdge: REMOTE edge");
  const int parent_code = he_ab->twin_gid;

  // Vertices of the edge
  int he_next = obj->half_edges[he_ab_idx].next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v_a_idx = obj->half_edges[he_prev].vertex_idx;
  int v_b_idx = he_ab->vertex_idx;

  // --- 2. Create the New Vertex M ---
  int v_m_idx = new_vertex(obj);
  MFTL_Vertex *v_m = &obj->vertices[v_m_idx];
  MFTL_Vertex *v_a = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b = &obj->vertices[v_b_idx];

  // Seam-safe midpoint (Stage 13): with wrapped storage the raw average of a
  // seam-crossing edge lands half a domain away. Branch so the non-periodic
  // arithmetic stays bit-identical.
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    double e_ab[3];
    MFTLGetMinimumImageVector(obj->domain, e_ab, v_a->pos, v_b->pos);
    for (int i = 0; i < 3; i++)
    {
      v_m->pos[i] = v_a->pos[i] + 0.5 * e_ab[i];
    }
    // Stage 16B displacement rule: anchor a (edge a->M has zero lattice shift)
    memcpy(v_m->img, v_a->img, 3 * sizeof(int));
    MFTLWrapPosImg(obj->domain, v_m->pos, v_m->img);
  }
  else
  {
    for (int i = 0; i < 3; i++)
    {
      v_m->pos[i] = 0.5 * (v_a->pos[i] + v_b->pos[i]);
    }
  }
  v_m->owner_proc = obj->domain->rank;  // change this later because in parallel this might not be true
  v_m->is_ghost = MFTL_FALSE;  // change this later because in parallel this might not be true

  // assign global id
  v_m->global_id = obj->next_vertex_gid++;  // change this later because in parallel this might not be true

  // --- 3. Split Upper Triangle (T1: ABC) ---
  // We transform T1(A,B,C) into T1(A,M,C) and T2(M,B,C)

  int t1_idx = he_ab->triangle_idx;
  int he_bc_idx = he_ab->next_idx;
  int he_ca_idx = obj->half_edges[he_bc_idx].next_idx;
  int v_c_idx = obj->half_edges[he_bc_idx].vertex_idx;

  int t2_idx = new_triangle(obj);
  int he_mb_idx = new_halfedge(obj);
  int he_mc_idx = new_halfedge(obj);
  int he_cm_idx = new_halfedge(obj);

  // Update T1 (Left: A-M-C)
  obj->triangles[t1_idx].he_idx = he_ab_idx;  // A->M

  // HE A->M (Reuse he_ab)
  obj->half_edges[he_ab_idx].vertex_idx = v_m_idx;
  obj->half_edges[he_ab_idx].next_idx = he_mc_idx;
  // twin_idx for he_am will be set in the bottom half logic
  obj->half_edges[he_ab_idx].twin_idx = MFTL_ERROR;
  obj->half_edges[he_ab_idx].twin_gid = MFTL_TWIN_UNKNOWN;  // A->M: set below (LOCAL pair or BOUNDARY)

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
  he_mc->twin_gid = he_cm->global_id;  // new inner pair: LOCAL
  he_cm->twin_gid = he_mc->global_id;

  // Mark the split segments as visited so the remeshing loop doesn't split them again
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_VISITED;
  obj->half_edges[he_mb_idx].flags |= MFTL_FLAG_VISITED;

  // If the original edge was a feature, the two new sub-edges must also be features!
  if (obj->half_edges[he_ab_idx].flags & MFTL_FLAG_FEATURE)
  {
    obj->half_edges[he_mb_idx].flags |= MFTL_FLAG_FEATURE;
  }

  // --- 4. Split Bottom Triangle (T3: ABD) if not boundary ---
  if (he_ba_idx != MFTL_ERROR)
  {
    int t3_idx = obj->half_edges[he_ba_idx].triangle_idx;
    int he_ad_idx = obj->half_edges[he_ba_idx].next_idx;
    int he_db_idx = obj->half_edges[he_ad_idx].next_idx;
    int v_d_idx = obj->half_edges[he_ad_idx].vertex_idx;

    int t4_idx = new_triangle(obj);
    int he_ma_idx = new_halfedge(obj);
    int he_md_idx = new_halfedge(obj);
    int he_dm_idx = new_halfedge(obj);

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
    obj->half_edges[he_md_idx].twin_gid = he_dm->global_id;  // new inner pair: LOCAL
    he_dm->twin_gid = obj->half_edges[he_md_idx].global_id;
    // the split halves of the interior edge: A->M <-> M->A, M->B <-> B->M
    obj->half_edges[he_ab_idx].twin_gid = obj->half_edges[he_ma_idx].global_id;
    obj->half_edges[he_ma_idx].twin_gid = obj->half_edges[he_ab_idx].global_id;
    obj->half_edges[he_mb_idx].twin_gid = obj->half_edges[he_ba_idx].global_id;
    obj->half_edges[he_ba_idx].twin_gid = obj->half_edges[he_mb_idx].global_id;

    // Mark the split segments as visited so the remeshing loop doesn't split them again
    obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_VISITED;
    obj->half_edges[he_ma_idx].flags |= MFTL_FLAG_VISITED;

    if (obj->half_edges[he_ba_idx].flags & MFTL_FLAG_FEATURE)
    {
      obj->half_edges[he_ma_idx].flags |= MFTL_FLAG_FEATURE;
    }
  }

  else
  {
    // BOUNDARY edge: both halves keep the parent's tag (plans/16 §2.2)
    MFTLHeSetBoundary(&obj->half_edges[he_ab_idx], MFTL_TWIN_NONE - parent_code);
    MFTLHeSetBoundary(&obj->half_edges[he_mb_idx], MFTL_TWIN_NONE - parent_code);
  }

  // --- 5. Set Vertex Outgoing Edges ---
  // These must be valid, non-deleted outgoing edges
  obj->vertices[v_m_idx].out_he_idx = he_mb_idx;
  obj->vertices[v_a_idx].out_he_idx = he_ab_idx;
  // a boundary vertex keeps its outgoing BOUNDARY half-edge (plans/16 §2.6)
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_b_idx].out_he_idx])) obj->vertices[v_b_idx].out_he_idx = he_bc_idx;

  obj->vertices[v_m_idx].flags |= MFTL_FLAG_VISITED;

  return v_m_idx;
}

// Helper to link twins and bypass a dead edge
// Connects OuterTwin1 <-> OuterTwin2. Stage 16A (plans/16 §2.2): the merged edge's state
// follows from the outer half-edges — both LOCAL: a LOCAL pair; one BOUNDARY: the LOCAL one
// becomes BOUNDARY with that tag. A REMOTE outer half-edge or two BOUNDARY ones are rejected by
// the collapse gates before any change.
void glue_wings(MFTL_Object *o, int edge1, int edge2)
{
  int t1 = o->half_edges[edge1].twin_idx;
  int t2 = o->half_edges[edge2].twin_idx;
  MFTL_ASSERT(!MFTLHeIsRemote(&o->half_edges[edge1]) && !MFTLHeIsRemote(&o->half_edges[edge2]) && "glue_wings: REMOTE wing");
  MFTL_ASSERT((t1 != MFTL_ERROR || t2 != MFTL_ERROR) && "glue_wings: two BOUNDARY wings");

  if (t1 != MFTL_ERROR)
  {
    o->half_edges[t1].twin_idx = t2;
    if (t2 != MFTL_ERROR)
    {
      o->half_edges[t1].twin_gid = o->half_edges[t2].global_id;
      o->half_edges[t1].boundary = MFTL_FALSE;
    }
    else
    {
      o->half_edges[t1].twin_gid = o->half_edges[edge2].twin_gid;  // inherits edge2's BOUNDARY code
      o->half_edges[t1].boundary = MFTLHeIsBoundary(&o->half_edges[t1]);
    }
    // Safety: Update vertex out pointer (origin of t1 = target of edge1 = the vertex 'C')
    o->vertices[o->half_edges[edge1].vertex_idx].out_he_idx = t1;
  }
  if (t2 != MFTL_ERROR)
  {
    o->half_edges[t2].twin_idx = t1;
    if (t1 != MFTL_ERROR)
    {
      o->half_edges[t2].twin_gid = o->half_edges[t1].global_id;
      o->half_edges[t2].boundary = MFTL_FALSE;
    }
    else
    {
      o->half_edges[t2].twin_gid = o->half_edges[edge1].twin_gid;  // inherits edge1's BOUNDARY code
      o->half_edges[t2].boundary = MFTLHeIsBoundary(&o->half_edges[t2]);
    }
    // Safety: Update vertex out pointer
    o->vertices[o->half_edges[edge2].vertex_idx].out_he_idx = t2;
  }
};

/* Stage 16A collapse admissibility of the twin states (plans/16 §2.2): the edge is LOCAL or
   BOUNDARY, no wing of a removed triangle is REMOTE, and the two wings of a removed triangle
   are not both BOUNDARY. */
static int mftlCollapseStatesOk(const MFTL_Object *obj, int he_ab_idx)
{
  const MFTL_HalfEdge *he = &obj->half_edges[he_ab_idx];
  if (MFTLHeIsRemote(he) || MFTLHeIsUnresolved(he)) return 0;
  int tri_he[2] = {he_ab_idx, he->twin_idx};
  for (int s = 0; s < 2; ++s)
  {
    if (tri_he[s] == MFTL_ERROR) continue;
    const int w1 = obj->half_edges[tri_he[s]].next_idx, w2 = obj->half_edges[w1].next_idx;
    const MFTL_HalfEdge *a = &obj->half_edges[w1], *b = &obj->half_edges[w2];
    if (MFTLHeIsRemote(a) || MFTLHeIsRemote(b) || MFTLHeIsUnresolved(a) || MFTLHeIsUnresolved(b)) return 0;
    if (MFTLHeIsBoundary(a) && MFTLHeIsBoundary(b)) return 0;
  }
  return 1;
}

void MFTLCollapseEdge(MFTL_Object *obj, int he_ab_idx)
{
  // --- 1. Gather Context ---
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  int he_ba_idx = he_ab->twin_idx;

  // Identify Vertices
  int prev_he = obj->half_edges[obj->half_edges[he_ab_idx].next_idx].next_idx;
  int v_a_idx = obj->half_edges[prev_he].vertex_idx;
  int v_b_idx = he_ab->vertex_idx;

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

  int safe_survivor_he = MFTL_ERROR;

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
    int he_out_idx = (int) (curr - obj->half_edges);

    // Save a candidate for B's new out_he
    if (he_out_idx != he_ab_idx)
    {
      safe_survivor_he = he_out_idx;
    }

    // UPDATE THE INCOMING EDGE
    // Safety: Verify Next link exists
    if (curr->next_idx != MFTL_ERROR)
    {
      int next_id = curr->next_idx;
      // We need Prev(curr). In a triangle, Prev = Next->Next.
      // We must check if Next->Next is valid.
      int next_next_id = obj->half_edges[next_id].next_idx;

      if (next_next_id != MFTL_ERROR)
      {
        int he_in_idx = next_next_id;
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
  // for (int k = 0; k < obj->n_half_edges; ++k)
  // {
  //   if (obj->half_edges[k].flags & MFTL_FLAG_DEAD) continue;
  //   if (obj->half_edges[k].vertex_idx == v_a_idx)
  //   {
  //     obj->half_edges[k].vertex_idx = v_b_idx;
  //   }
  // }

  // --- 3. Remove Degenerate Faces (Gluing) ---
  // (Copy your existing code for Step 3 exactly as is)
  int t1_idx = he_ab->triangle_idx;
  obj->triangles[t1_idx].flags |= MFTL_FLAG_DEAD;
  int he_bc = he_ab->next_idx;
  int he_ca = obj->half_edges[he_bc].next_idx;
  glue_wings(obj, he_ca, he_bc);
  obj->half_edges[he_bc].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ca].flags |= MFTL_FLAG_DEAD;

  if (he_ba_idx != MFTL_ERROR)
  {
    int t2_idx = obj->half_edges[he_ba_idx].triangle_idx;
    obj->triangles[t2_idx].flags |= MFTL_FLAG_DEAD;
    int he_ad = obj->half_edges[he_ba_idx].next_idx;
    int he_db = obj->half_edges[he_ad].next_idx;
    glue_wings(obj, he_ad, he_db);
    obj->half_edges[he_ad].flags |= MFTL_FLAG_DEAD;
    obj->half_edges[he_db].flags |= MFTL_FLAG_DEAD;
  }

  // --- 4. Final Cleanup ---
  obj->vertices[v_a_idx].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_DEAD;
  if (he_ba_idx != MFTL_ERROR) obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_DEAD;

  // Update B's position (seam-safe midpoint; raw average when non-periodic)
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    double e_ba[3];
    MFTLGetMinimumImageVector(obj->domain, e_ba, obj->vertices[v_b_idx].pos, obj->vertices[v_a_idx].pos);
    for (int k = 0; k < 3; ++k)
    {
      obj->vertices[v_b_idx].pos[k] = obj->vertices[v_b_idx].pos[k] + 0.5 * e_ba[k];
    }
    MFTLWrapPosImg(obj->domain, obj->vertices[v_b_idx].pos, obj->vertices[v_b_idx].img);  // Stage 16B: anchor b
  }
  else
  {
    for (int k = 0; k < 3; ++k)
    {
      obj->vertices[v_b_idx].pos[k] = 0.5 * (obj->vertices[v_b_idx].pos[k] + obj->vertices[v_a_idx].pos[k]);
    }
  }

  // --- SAFE POINTER UPDATE (Using the survivor we found in Step 2) ---
  int check_he = obj->vertices[v_b_idx].out_he_idx;

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

/* Stage 16A: the one-ring VERTICES of v including, for a boundary vertex (OPEN_COMPLETE fan),
   the neighbour across the incoming BOUNDARY edge, which MFTLGetVertexOneRingVertices (targets of
   the outgoing half-edges) misses. Identical to MFTLGetVertexOneRingVertices for closed fans.
   16C review: returns -1 (buffer untouched) when the ring does not fit into `max` entries; the
   callers reject the operation. MFTLGetVertexOneRingVertices only asserts its bound, i.e. it
   overflowed the caller's buffer in Release builds once refinement raised a valence past `max`. */
static int mftlRingVerticesFull(const MFTL_Object *obj, MFTL_Vertex *v, MFTL_Vertex **out, int max)
{
  int nfan = 0;
  MFTLVertexRing(obj, (int) (v - obj->vertices), MFTL_NULL, 0, &nfan);  // count only (no buffer)
  if (nfan + 1 > max) return -1;
  int n = MFTLGetVertexOneRingVertices(obj, v, out, max, MFTL_FALSE);
  int ring[64], nr = 0;
  if (MFTLVertexRing(obj, (int) (v - obj->vertices), ring, 64, &nr) == MFTL_RING_OPEN_COMPLETE && nr > 0 && n < max)
  {
    const int in_h = obj->half_edges[obj->half_edges[ring[0]].next_idx].next_idx;  // incoming BOUNDARY
    const int in_prev = obj->half_edges[obj->half_edges[in_h].next_idx].next_idx;
    MFTL_Vertex *w = &obj->vertices[obj->half_edges[in_prev].vertex_idx];  // its origin
    int present = 0;
    for (int i = 0; i < n; ++i)
      if (out[i] == w) present = 1;
    if (!present) out[n++] = w;
  }
  return n;
}

// Helper to check if vertex is on boundary (uses the 1-ring logic implicitly)
static bool IsBoundaryVertex(const MFTL_Object *obj, int v_idx)
{
  // Stage 16A: MFTLVertexRing. An incomplete fan (REMOTE rim) is treated like a boundary, as
  // before; a boundary vertex has an OPEN_COMPLETE fan.
  return MFTLVertexRing(obj, v_idx, MFTL_NULL, 0, MFTL_NULL) != MFTL_RING_CLOSED;
}

/**
 * @brief Counts the number of feature edges (or boundary edges) connected to a vertex.
 */
int CountFeatureEdges(MFTL_Object *obj, int v_idx)
{
  MFTL_Vertex *v = &obj->vertices[v_idx];

  // Safety check: skip dead or isolated vertices
  if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) return 0;

  int feat_count = 0;
  int curr_he_idx = v->out_he_idx;
  int start_he_idx = curr_he_idx;

  // --- 1. REWIND TO START OF FAN ---
  // If the vertex is on a boundary, we must find the leftmost edge.
  while (1)
  {
    // Get the incoming half-edge inside the current triangle
    int next_idx = obj->half_edges[curr_he_idx].next_idx;
    int prev_idx = obj->half_edges[next_idx].next_idx;

    int twin_idx = obj->half_edges[prev_idx].twin_idx;

    // If no twin, we hit the boundary dead-end. If we reach start, it's an interior vertex.
    if (twin_idx == MFTL_ERROR || twin_idx == start_he_idx) break;

    curr_he_idx = twin_idx;
  }

  start_he_idx = curr_he_idx;  // This is the true start of our sweep

  // --- 2. SWEEP FORWARD AND COUNT ---
  while (1)
  {
    MFTL_HalfEdge *he = &obj->half_edges[curr_he_idx];

    // Is this outgoing edge a feature or a boundary?
    if ((he->flags & MFTL_FLAG_FEATURE) || he->twin_idx == MFTL_ERROR)
    {
      feat_count++;
    }

    // Move to the next outgoing half-edge
    if (he->twin_idx == MFTL_ERROR)
    {
      // We reached the rightmost boundary edge. The forward sweep is done.
      break;
    }

    curr_he_idx = obj->half_edges[he->twin_idx].next_idx;

    // If we circled all the way back, we are done (Interior vertex)
    if (curr_he_idx == start_he_idx) break;
  }

  // --- 3. HANDLE THE FINAL INCOMING BOUNDARY EDGE ---
  // If we stopped because of a boundary (twin == ERROR), there is exactly ONE
  // physical edge we missed: the very first incoming boundary edge at the start of the fan.
  int next_idx = obj->half_edges[start_he_idx].next_idx;
  int prev_idx = obj->half_edges[next_idx].next_idx;
  MFTL_HalfEdge *incoming_start_he = &obj->half_edges[prev_idx];

  if (incoming_start_he->twin_idx == MFTL_ERROR)
  {
    // Boundary edges act as features automatically
    feat_count++;
  }

  return feat_count;
}

// Stage 08: unit normal of a triangle from current vertex positions.
static void mftlUnitNormalFromTri(MFTL_Object* obj, int tri_idx, double n[3])
{
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);
  // minimum-image edges: raw differences when non-periodic, seam-safe else
  double e0[3], e1[3];
  MFTLGetMinimumImageVector(obj->domain, e0, vv[0]->pos, vv[1]->pos);
  MFTLGetMinimumImageVector(obj->domain, e1, vv[0]->pos, vv[2]->pos);
  n[0] = e0[1] * e1[2] - e0[2] * e1[1];
  n[1] = e0[2] * e1[0] - e0[0] * e1[2];
  n[2] = e0[0] * e1[1] - e0[1] * e1[0];
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len < MFTL_VERYSMALL)
  {
    n[0] = n[1] = n[2] = 0.0;
    return;
  }
  n[0] /= len;
  n[1] /= len;
  n[2] /= len;
}

// Geometric sharp-edge test: the angle between the two adjacent triangle
// normals exceeds the feature threshold (flag-independent, so protection
// holds even when the caller never ran MFTLDetectFeatures).
static int mftlEdgeIsGeometricallySharp(MFTL_Object* obj, int he_idx, double angle_degrees)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->twin_idx == MFTL_ERROR) return 1;  // boundary: treat as feature
  double n1[3], n2[3];
  mftlUnitNormalFromTri(obj, he->triangle_idx, n1);
  mftlUnitNormalFromTri(obj, obj->half_edges[he->twin_idx].triangle_idx, n2);
  double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
  return dot < MFTL_COS(angle_degrees * MFTL_PI / 180.0);
}

/* number of NON-boundary feature (sharp-dihedral) half-edges leaving v (generalized mode: a
   boundary edge is not a feature; corners are handled by the turning angle) */
static int mftlInteriorFeatureCount(MFTL_Object *obj, int v)
{
  int ring[128], nr = 0, n = 0;
  if (MFTLVertexRing(obj, v, ring, 128, &nr) == MFTL_RING_INCOMPLETE) return 0;
  for (int i = 0; i < nr; ++i)
  {
    const MFTL_HalfEdge *h = &obj->half_edges[ring[i]];
    if (MFTLHeIsLocal(h) && (h->flags & MFTL_FLAG_FEATURE)) n++;
  }
  return n;
}

/* Stage 16C (plans/16 §4.4.1): collapse approval of the GENERALIZED remesher (open bodies and
   closed bodies opted in). The topological test is the augmented link condition only; on top:
   corners are never collapsed, a boundary vertex never collapses into an interior one, the
   survivor of a boundary edge lies on the boundary curve and an interior edge with one boundary
   endpoint keeps the boundary vertex in place (16A S-A5); boundary edges are not features, so
   boundary loops coarsen as well as refine (16A S-A4). */
#define MFTL_GENERALIZED_MAX_VALENCE 16
/* 16C review: splits add a neighbour to each opposite vertex, and flips to the two new edge
   endpoints; refinement around a vertex left coarse by repeated coarsening raised its valence past
   the fixed one-ring buffers (flip gate 32, smoothers 32, fits 64; np-8 cycles). A generalized
   split or flip therefore never raises a valence above this cap (a topological, translation- and
   decomposition-independent rule; flips still reduce higher valences). */
#define MFTL_GENERALIZED_MAX_GROW_VALENCE 24
#define MFTL_FLIP_RING_MAX 128

/* number of neighbour vertices of v (a boundary vertex's full fan plus one), -1 if its fan is not
   completely stored */
static int mftlVertexValence(const MFTL_Object *obj, int v)
{
  int nfan = 0;
  const int cls = MFTLVertexRing(obj, v, MFTL_NULL, 0, &nfan);
  if (cls == MFTL_RING_INCOMPLETE) return -1;
  return cls == MFTL_RING_OPEN_COMPLETE ? nfan + 1 : nfan;
}

/* 1 if splitting he keeps both opposite vertices within the growth cap (complete fans only: an
   opposite vertex whose fan is not stored is not split around) */
int MFTLGeneralizedSplitValenceOk(const MFTL_Object *obj, int he_idx)
{
  const MFTL_HalfEdge *he = &obj->half_edges[he_idx];
  const int val_c = mftlVertexValence(obj, obj->half_edges[he->next_idx].vertex_idx);
  if (val_c < 0 || val_c + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE) return 0;
  if (MFTLHeIsLocal(he))
  {
    const int val_d = mftlVertexValence(obj, obj->half_edges[obj->half_edges[he->twin_idx].next_idx].vertex_idx);
    if (val_d < 0 || val_d + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE) return 0;
  }
  return 1;
}
static int mftlCollapseApprovedGeneralized(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  const int v_a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
  const int v_b = he->vertex_idx;
  if (obj->vertices[v_a].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  const int is_bnd = MFTLHeIsBoundary(he);
  if (!is_bnd && mftlEdgeIsGeometricallySharp(obj, he_idx, params->feature_angle_threshold)) return MFTL_FALSE;

  // criterion (boundary edges: one triangle; curvature-based criteria fall back to the length
  // criterion where the curvature is undefined, i.e. at boundary vertices)
  const int curv_undef = (obj->vertices[v_a].flags | obj->vertices[v_b].flags) & MFTL_VFLAG_CURV_UNDEFINED;
  if (params->criterion == MFTL_CRITERION_TRIANGLE_AREA)
  {
    double a1 = obj->triangles[he->triangle_idx].area, amin = a1;
    if (MFTLHeIsLocal(he)) amin = fmin(a1, obj->triangles[obj->half_edges[he->twin_idx].triangle_idx].area);
    if (amin > params->min_area_threshold) return MFTL_FALSE;
  }
  else if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE && !is_bnd && !curv_undef)
  {
    if (!MFTLBotschKobbeltCurvatureThreshold(obj, he_idx, params, 2)) return MFTL_FALSE;
  }
  else if (MFTLGetEdgeLengthSquared(obj, he) > params->min_edge_length_sq)
    return MFTL_FALSE;

  // complete fans, corners, boundary-into-interior
  const int cls_a = MFTLVertexRing(obj, v_a, MFTL_NULL, 0, MFTL_NULL);
  const int cls_b = MFTLVertexRing(obj, v_b, MFTL_NULL, 0, MFTL_NULL);
  if (cls_a == MFTL_RING_INCOMPLETE || cls_b == MFTL_RING_INCOMPLETE) return MFTL_FALSE;
  const double corner = MFTLBoundaryCornerAngle(params);
  const int ka = MFTLBoundaryVertexKind(obj, v_a, corner), kb = MFTLBoundaryVertexKind(obj, v_b, corner);
  if (ka == 1 || kb == 1) return MFTL_FALSE;  // corners are never collapsed or merged
  if (ka == 0 && kb != 0) return MFTL_FALSE;  // a boundary vertex never moves into the interior
  if (!MFTLAugmentedLinkConditionOk(obj, he_idx)) return MFTL_FALSE;
  // survivor valence cap: repeated coarsening would otherwise pile fans onto one survivor (valence
  // > 30 observed on a coarsened patch) and overflow the fixed one-ring buffers of the smoothers;
  // the survivor keeps at most val(a) + val(b) - 2 neighbours
  {
    int nra = 0, nrb = 0;
    MFTLVertexRing(obj, v_a, MFTL_NULL, 0, &nra);
    MFTLVertexRing(obj, v_b, MFTL_NULL, 0, &nrb);
    if (nra + nrb - 2 > MFTL_GENERALIZED_MAX_VALENCE) return MFTL_FALSE;
  }

  // survivor placement and geometric gates
  double target[3];
  int kind;
  if (MFTLGeneralizedCollapseTarget(obj, he_idx, corner, target, &kind) != MFTL_SUCCESS) return MFTL_FALSE;
  MFTL_Vertex *ring[64];
  const int ends[2] = {v_a, v_b};
  for (int e = 0; e < 2; ++e)
  {
    MFTL_Vertex *ve = &obj->vertices[ends[e]];
    const int cnt = mftlRingVerticesFull(obj, ve, ring, 64);
    if (cnt < 0) return MFTL_FALSE;  // ring larger than the buffer (16C review)
    for (int i = 0; i < cnt; ++i)
    {
      if (ring[i] == &obj->vertices[v_a] || ring[i] == &obj->vertices[v_b]) continue;
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, target, ring[i]->pos);
      if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > params->max_edge_length_sq) return MFTL_FALSE;  // stretch
    }
  }
  if (!MFTLCollapseKeepsOrientation(obj, he_idx, target)) return MFTL_FALSE;
  // interior feature lines (Stage 08 protection, boundary edges excluded)
  const int fa = mftlInteriorFeatureCount(obj, v_a), fb = mftlInteriorFeatureCount(obj, v_b);
  if ((fa > 0) != (fb > 0)) return MFTL_FALSE;
  if (fa > 2 || fb > 2) return MFTL_FALSE;
  return MFTL_TRUE;
}

int CollapseApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // Stage 16A: twin-state admissibility (no REMOTE wing, no two BOUNDARY wings)
  if (!mftlCollapseStatesOk(obj, he_idx)) return MFTL_FALSE;

  // Stage 16C: open bodies and closed bodies opted in use the generalized remesher (never the
  // parity path); closed bodies in the default mode are unchanged
  if (params && MFTLObjectRemeshGeneralized(obj)) return mftlCollapseApprovedGeneralized(obj, he_idx, params);

  // Stage 15: MFTL1 criteria (with the parallel drivers' locks)
  if (params && params->mftl1_criteria) return MFTL1CollapseApproved(obj, he_idx, params, 1);

  // 1. Status Checks
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // 1.2 Geometric feature protection (Stage 08 review): a sharp edge must
  // never be collapsed, regardless of pre-set feature flags.
  if (mftlEdgeIsGeometricallySharp(obj, he_idx, params->feature_angle_threshold)) return MFTL_FALSE;

  // --- CHECK VERTEX FLAGS ---
  // Prevent chained collapses by checking if vertices are already dirty
  int prev_he = obj->half_edges[obj->half_edges[he_idx].next_idx].next_idx;
  int v_a_idx = obj->half_edges[prev_he].vertex_idx;
  int v_b_idx = he->vertex_idx;

  // If either vertex is DEAD (removed) or VISITED (involved in another op), skip.
  if (obj->vertices[v_a_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // 1.1 Check if the edge is a boundary edge
  // if (he->twin_idx == MFTL_ERROR) return MFTL_FALSE;  // Generally avoid collapsing boundary edges unless robust
  // Boundary Edge Detection
  bool is_boundary_edge = MFTLHeIsBoundary(he);

  // 2. Criterion Check (Stage 08): reject unless the criterion says the edge
  // is short enough to collapse.
  if (params->criterion == MFTL_CRITERION_TRIANGLE_AREA)
  {
    double a1 = obj->triangles[he->triangle_idx].area;
    double a2 = 0.0;
    if (he->twin_idx != MFTL_ERROR) a2 = obj->triangles[obj->half_edges[he->twin_idx].triangle_idx].area;
    double amin = a1 < a2 ? a1 : a2;
    if (amin > params->min_area_threshold) return MFTL_FALSE;
  }
  else if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE)
  {
    if (!MFTLBotschKobbeltCurvatureThreshold(obj, he_idx, params, 2)) return MFTL_FALSE;
  }
  else
  {
    double len_sq0 = MFTLGetEdgeLengthSquared(obj, he);
    if (len_sq0 > params->min_edge_length_sq) return MFTL_FALSE;
  }

  // 3. Topology: Link Condition (The Intersection Test)
  // We need to ensure A and B share exactly 2 neighbors (for interior edge).

  MFTL_Vertex *v_a_ptr = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b_ptr = &obj->vertices[v_b_idx];

  // Get Rings
  MFTL_Vertex *ring_a[64];
  MFTL_Vertex *ring_b[64];
  // Stage 16A: full neighbour sets (boundary vertices included), so the link test is exact
  int count_a = mftlRingVerticesFull(obj, v_a_ptr, ring_a, 64);
  int count_b = mftlRingVerticesFull(obj, v_b_ptr, ring_b, 64);
  if (count_a < 0 || count_b < 0) return MFTL_FALSE;  // ring larger than the buffer (16C review)

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
  // Stage 16A: an interior edge joining two boundary vertices would pinch the surface into a
  // non-manifold vertex (link condition with the virtual boundary vertex, plans/16 §4.4.1)
  if (a_bound && b_bound && !is_boundary_edge) return MFTL_FALSE;

  // --- 5. Ping-Pong Prevention (The Stretch Check) ---
  // Calculate proposed midpoint M (seam-safe; raw average when non-periodic)
  double pos_m[3];
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    double e_ba[3];
    MFTLGetMinimumImageVector(obj->domain, e_ba, v_b_ptr->pos, v_a_ptr->pos);
    for (int k = 0; k < 3; ++k) pos_m[k] = v_b_ptr->pos[k] + 0.5 * e_ba[k];
    MFTLApplyPeriodicity(obj->domain, pos_m);
  }
  else
  {
    for (int k = 0; k < 3; ++k)
    {
      pos_m[k] = 0.5 * (v_a_ptr->pos[k] + v_b_ptr->pos[k]);
    }
  }

  // Check neighbors of A
  for (int i = 0; i < count_a; ++i)
  {
    // Don't check B, it's collapsing
    if (ring_a[i] == v_b_ptr) continue;

    double dvec[3];
    MFTLGetMinimumImageVector(obj->domain, dvec, pos_m, ring_a[i]->pos);
    double new_len_sq = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];

    // If the new edge will be longer than max allowed, reject the collapse!
    if (new_len_sq > params->max_edge_length_sq) return MFTL_FALSE;
  }

  // Check neighbors of B
  for (int i = 0; i < count_b; ++i)
  {
    if (ring_b[i] == v_a_ptr) continue;

    double dvec[3];
    MFTLGetMinimumImageVector(obj->domain, dvec, pos_m, ring_b[i]->pos);
    double new_len_sq = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];

    if (new_len_sq > params->max_edge_length_sq) return MFTL_FALSE;
  }

  // Count how many feature edges connect to A and B
  int a_feat_count = CountFeatureEdges(obj, v_a_idx);
  int b_feat_count = CountFeatureEdges(obj, v_b_idx);

  // If A is on a feature line, and B is not, collapsing A into the midpoint
  // will pull the feature line off its trajectory.
  if (a_feat_count > 0 && b_feat_count == 0) return MFTL_FALSE;
  if (b_feat_count > 0 && a_feat_count == 0) return MFTL_FALSE;

  // If A is a hard corner (>= 3 feature edges), you cannot collapse it!
  if (a_feat_count > 2 || b_feat_count > 2) return MFTL_FALSE;

  return MFTL_TRUE;
}

int MFTLSplitApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params)
{
  int approved = MFTL_FALSE;
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // 1. Status Checks (Edge AND Vertices)
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // Check vertices to ensure we don't split an area that is already being modified
  int next = obj->half_edges[he_idx].next_idx;
  int prev = obj->half_edges[next].next_idx;
  int v_a = obj->half_edges[prev].vertex_idx;
  int v_b = he->vertex_idx;

  if (obj->vertices[v_a].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // Stage 16A: a REMOTE edge is never split (its second triangle is not stored)
  if (MFTLHeIsRemote(he) || MFTLHeIsUnresolved(he)) return MFTL_FALSE;

  // Stage 15: MFTL1 criteria (Stage 16C: never for generalized bodies)
  const int generalized = MFTLObjectRemeshGeneralized(obj);
  if (params->mftl1_criteria && !generalized) return MFTL1SplitCriterion(obj, he_idx, params);
  // 16C review: the opposite vertices gain a neighbour; never above the growth cap
  if (generalized && !MFTLGeneralizedSplitValenceOk(obj, he_idx)) return MFTL_FALSE;
  // Stage 16C: a boundary split needs its boundary-curve point (a CONSTRAINED projector may refuse)
  if (generalized && MFTLHeIsBoundary(he))
  {
    double tmp[3];
    if (MFTLGeneralizedSplitTarget(obj, he_idx, MFTLBoundaryCornerAngle(params), tmp) != MFTL_SUCCESS) return MFTL_FALSE;
  }
  // Stage 16C: curvature criteria are undefined at boundary vertices -> length criterion
  if (generalized && params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE &&
      (MFTLHeIsBoundary(he) || ((obj->vertices[v_a].flags | obj->vertices[v_b].flags) & MFTL_VFLAG_CURV_UNDEFINED)))
    return MFTLGetEdgeLengthSquared(obj, he) > params->max_edge_length_sq;

  // 2. Criterion check (Stage 08: MFTL1 RemeshCriteria parity)
  switch (params->criterion)
  {
    case MFTL_CRITERION_TRIANGLE_AREA:
    {
      // Split when an adjacent triangle exceeds the max-area threshold.
      double a1 = obj->triangles[he->triangle_idx].area;
      double a2 = 0.0;
      if (he->twin_idx != MFTL_ERROR) a2 = obj->triangles[obj->half_edges[he->twin_idx].triangle_idx].area;
      approved = (a1 > params->max_area_threshold || a2 > params->max_area_threshold);
      break;
    }
    case MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE:
      approved = MFTLBotschKobbeltCurvatureThreshold(obj, he_idx, params, 1);
      break;
    case MFTL_CRITERION_JIAO:
    {
      // Jiao 2010: relative longness floor + absolute max-length split.
      double len_sq = MFTLGetEdgeLengthSquared(obj, he);
      if (len_sq < params->reference_length * params->reference_length) break;
      approved = (len_sq > params->max_edge_length_sq);
      break;
    }
    case MFTL_CRITERION_BOTSCH_KOBBELT:
    default:
    {
      // Check if edge is too long
      double len_sq = MFTLGetEdgeLengthSquared(obj, he);
      if (len_sq > params->max_edge_length_sq) approved = MFTL_TRUE;
      break;
    }
  }

  return approved;
}

void MFTLFlipEdge(MFTL_Object *obj, int he_idx)
{
  // get the two triangles involved
  int tri1_idx = obj->half_edges[he_idx].triangle_idx;
  int tri2_idx = obj->half_edges[obj->half_edges[he_idx].twin_idx].triangle_idx;  // ensure this is not a boundary edge

  // Stage 15 (L7, MFTL1 edgeflip.c parity): mass and area of the two triangles before the
  // flip; afterwards the mass is split in proportion new_area / (sum of the OLD areas),
  // the second triangle taking the remainder.
  MFTLGetTriangleProperties(obj, &obj->triangles[tri1_idx], &obj->triangles[tri1_idx].area, MFTL_NULL, MFTL_NULL);
  MFTLGetTriangleProperties(obj, &obj->triangles[tri2_idx], &obj->triangles[tri2_idx].area, MFTL_NULL, MFTL_NULL);
  const double ss_mass_sum =
      obj->triangles[tri1_idx].surface_scalar * obj->triangles[tri1_idx].area + obj->triangles[tri2_idx].surface_scalar * obj->triangles[tri2_idx].area;
  const double ss_area_sum = obj->triangles[tri1_idx].area + obj->triangles[tri2_idx].area;

  // get the halfedges involved
  int he_ab_idx = he_idx;
  int he_bc_idx = obj->half_edges[he_ab_idx].next_idx;
  int he_ca_idx = obj->half_edges[he_bc_idx].next_idx;
  int he_ba_idx = obj->half_edges[he_ab_idx].twin_idx;
  int he_ad_idx = obj->half_edges[he_ba_idx].next_idx;
  int he_db_idx = obj->half_edges[he_ad_idx].next_idx;

  // get the vertices involved
  int v_a = obj->half_edges[he_ba_idx].vertex_idx;
  int v_b = obj->half_edges[he_ab_idx].vertex_idx;
  int v_c = obj->half_edges[he_bc_idx].vertex_idx;
  int v_d = obj->half_edges[he_ad_idx].vertex_idx;

  // update the outgoing halfedges of the vertices a and b (simply do it instead of also checking)
  // (a boundary vertex keeps its outgoing BOUNDARY half-edge, which the flip does not touch)
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_a].out_he_idx])) obj->vertices[v_a].out_he_idx = he_ad_idx;
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_b].out_he_idx])) obj->vertices[v_b].out_he_idx = he_bc_idx;

  // update the halfedge for t1 and t2 (simply do it instead of also checking)
  obj->triangles[tri1_idx].he_idx = he_ab_idx;
  obj->triangles[tri2_idx].he_idx = he_ba_idx;

  // update the tris in the halfedges
  obj->half_edges[he_ab_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ca_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ad_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ba_idx].triangle_idx = tri2_idx;
  obj->half_edges[he_db_idx].triangle_idx = tri2_idx;
  obj->half_edges[he_bc_idx].triangle_idx = tri2_idx;

  // update the next pointers
  obj->half_edges[he_ab_idx].next_idx = he_ca_idx;
  obj->half_edges[he_ca_idx].next_idx = he_ad_idx;
  obj->half_edges[he_ad_idx].next_idx = he_ab_idx;
  obj->half_edges[he_ba_idx].next_idx = he_db_idx;
  obj->half_edges[he_db_idx].next_idx = he_bc_idx;
  obj->half_edges[he_bc_idx].next_idx = he_ba_idx;

  // update the vertex indices
  obj->half_edges[he_ab_idx].vertex_idx = v_c;
  obj->half_edges[he_ba_idx].vertex_idx = v_d;

  // Stage 15 (L7): remap the conserved surface scalar (see above).
  MFTL_Triangle *t1 = &obj->triangles[tri1_idx], *t2 = &obj->triangles[tri2_idx];
  MFTLGetTriangleProperties(obj, t1, &t1->area, MFTL_NULL, MFTL_NULL);
  MFTLGetTriangleProperties(obj, t2, &t2->area, MFTL_NULL, MFTL_NULL);
  const double ss_mass1 = (ss_area_sum > 0.0) ? ss_mass_sum * (t1->area / ss_area_sum) : 0.0;
  const double ss_mass2 = ss_mass_sum - ss_mass1;
  t1->surface_scalar = (t1->area > 0.0) ? ss_mass1 / t1->area : 0.0;
  t2->surface_scalar = (t2->area > 0.0) ? ss_mass2 / t2->area : 0.0;
}

// Stage 08 gated flip (defined below; declared here for the driver).
int FlipApprovedWithParams(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params);

int FlipApproved(MFTL_Object *obj, int he_idx)
{
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_idx];
  int he_ba_idx = he_ab->twin_idx;
  if (!MFTLHeIsLocal(he_ab)) return MFTL_FALSE;  // Stage 16A: only interior LOCAL edges flip

  MFTL_HalfEdge *he_ba = &obj->half_edges[he_ba_idx];

  // --- 2. Identify the 4 Vertices (A, B, C, D) ---
  // he_ab is A->B, so target is B. he_ba is B->A, so target is A.
  int v_b = he_ab->vertex_idx;
  int v_a = he_ba->vertex_idx;

  int he_bc_idx = he_ab->next_idx;
  int he_ad_idx = he_ba->next_idx;

  int v_c = obj->half_edges[he_bc_idx].vertex_idx;
  int v_d = obj->half_edges[he_ad_idx].vertex_idx;

  // --- 3. Gather Current Valences ---
  // We use your existing 1-ring function just to get the degree/count
  // 16C review: 32 entries overflowed on generalized bodies (refinement around a coarse vertex
  // raised its valence past 32); the gather is bounded and a ring that does not fit rejects the
  // flip. Identical to the previous buffer for every valence below 32.
  MFTL_Vertex *ring_buffer[MFTL_FLIP_RING_MAX];
  int val_a = mftlRingVerticesFull(obj, &obj->vertices[v_a], ring_buffer, MFTL_FLIP_RING_MAX);
  int val_b = mftlRingVerticesFull(obj, &obj->vertices[v_b], ring_buffer, MFTL_FLIP_RING_MAX);

  // Topological Safety Check 1: Prevent Degenerate Triangles
  // Flipping reduces the valence of A and B by 1. A vertex must have at least valence 3.
  if (val_a <= 3 || val_b <= 3) return MFTL_FALSE;

  // Stage 16A: full neighbour set of C (a boundary vertex's neighbour across its incoming
  // boundary edge included) for the duplicate-edge test below
  int val_c = mftlRingVerticesFull(obj, &obj->vertices[v_c], ring_buffer, MFTL_FLIP_RING_MAX);
  if (val_c < 0) return MFTL_FALSE;

  // Topological Safety Check 2: Prevent Non-Manifold Edges
  // If C and D are ALREADY connected by an edge, flipping AB to CD would create
  // two overlapping edges between C and D, breaking the manifold property.
  // We can check this by seeing if D is already in C's 1-ring.
  for (int i = 0; i < val_c; ++i)
  {
    // ring_buffer is populated with C's neighbors
    if (ring_buffer[i] == &obj->vertices[v_d]) return MFTL_FALSE;
  }

  int val_d = mftlRingVerticesFull(obj, &obj->vertices[v_d], ring_buffer, MFTL_FLIP_RING_MAX);
  if (val_d < 0) return MFTL_FALSE;
  // 16C review: a generalized flip never raises a valence above the split/flip cap (see
  // MFTL_GENERALIZED_MAX_GROW_VALENCE)
  if (MFTLObjectRemeshGeneralized(obj) && (val_c + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE || val_d + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE)) return MFTL_FALSE;

  // --- 4. Determine Target Valences ---
  // Ideal valence is 6 for interior vertices, 4 for boundary vertices
  int opt_a = IsBoundaryVertex(obj, v_a) ? 4 : 6;
  int opt_b = IsBoundaryVertex(obj, v_b) ? 4 : 6;
  int opt_c = IsBoundaryVertex(obj, v_c) ? 4 : 6;
  int opt_d = IsBoundaryVertex(obj, v_d) ? 4 : 6;
  if (MFTLObjectRemeshGeneralized(obj))
  {
    // Stage 16C (plans/16 §4.4.2): boundary valence target round(theta / 60 deg) + 1 (min 2),
    // theta the interior angle; 6 in the interior
    opt_a = MFTLBoundaryValenceTarget(obj, v_a);
    opt_b = MFTLBoundaryValenceTarget(obj, v_b);
    opt_c = MFTLBoundaryValenceTarget(obj, v_c);
    opt_d = MFTLBoundaryValenceTarget(obj, v_d);
  }

  // --- 5. Calculate Current Error (Energy) ---
  int err_current =
      (val_a - opt_a) * (val_a - opt_a) + (val_b - opt_b) * (val_b - opt_b) + (val_c - opt_c) * (val_c - opt_c) + (val_d - opt_d) * (val_d - opt_d);

  // --- 6. Calculate Flipped Error (Energy) ---
  // If flipped, A and B lose an edge, C and D gain an edge
  int val_a_new = val_a - 1;
  int val_b_new = val_b - 1;
  int val_c_new = val_c + 1;
  int val_d_new = val_d + 1;

  int err_flipped = (val_a_new - opt_a) * (val_a_new - opt_a) + (val_b_new - opt_b) * (val_b_new - opt_b) + (val_c_new - opt_c) * (val_c_new - opt_c) +
                    (val_d_new - opt_d) * (val_d_new - opt_d);

  // --- 7. The Golden Rule ---
  // Only approve if the total deviation strictly decreases!
  if (err_flipped < err_current)
  {
    return MFTL_TRUE;
  }

  return MFTL_FALSE;
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
// Stage 15 (L3): re-derive the retained twin gid of every half-edge whose twin is stored.
// Stage 16A: the operators set twin_gid explicitly (split / glue_wings); this only refreshes
// LOCAL pairs after compaction and never touches REMOTE / BOUNDARY codes.
static void mftlRefreshTwinGids(MFTL_Object *obj)
{
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &obj->half_edges[i];
    if (he->twin_idx != MFTL_ERROR) he->twin_gid = obj->half_edges[he->twin_idx].global_id;
  }
}

void MFTLFinalizeRemeshing(MFTL_Object *obj, int start_num_v, int start_num_t, int start_num_h, bool collapses_occurred)
{
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

      for (int i = start_num_v; i < obj->n_vertices; ++i)
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

      for (int i = start_num_t; i < obj->n_triangles; ++i)
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

      for (int i = start_num_h; i < obj->n_half_edges; ++i)
      {
        int ret;
        khiter_t iter = kh_put(halfedge_map, obj->halfedge_lookup, obj->half_edges[i].global_id, &ret);
        kh_value(obj->halfedge_lookup, iter) = i;
      }
    }

    mftlRefreshTwinGids(obj);
    MFTLRefreshBoundaryCache(obj);
    return;  // DONE! (Fast exit)
  }

  // ---------------------------------------------------------
  // PATH B: FULL COMPACTION (Collapses Happened - Move Data)
  // ---------------------------------------------------------

  // Safety Check: ensure no live half-edge points to a dead vertex
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    if (obj->half_edges[i].flags & MFTL_FLAG_DEAD) continue;
    int v_idx = obj->half_edges[i].vertex_idx;
    if (obj->vertices[v_idx].flags & MFTL_FLAG_DEAD)
    {
      printf("Error: Live HalfEdge %d points to Dead Vertex %d\n", i, v_idx);
      assert(0);
    }
  }

  // 1. Setup Maps
  int *v_map = malloc(obj->n_vertices * sizeof(int));
  int *t_map = malloc(obj->n_triangles * sizeof(int));
  int *h_map = malloc(obj->n_half_edges * sizeof(int));

  assert(v_map && t_map && h_map);

  // 2. Compact Arrays (Standard Logic)
  // --- Vertices ---
  int new_v_count = 0;
  for (int i = 0; i < obj->n_vertices; i++)
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
  int new_t_count = 0;
  for (int i = 0; i < obj->n_triangles; i++)
  {
    if (obj->triangles[i].flags & MFTL_FLAG_DEAD)
    {
      // Stage 12: free the dead triangle's user payload. Live triangles move
      // by struct assignment below, so their blob pointer rides along
      // automatically (this is why the payload is a per-triangle heap blob
      // rather than a contiguous pool, which would need an explicit
      // permutation at every reorder/growth site).
      if (obj->data_size > 0) free(obj->triangles[i].data);
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
  int new_h_count = 0;
  for (int i = 0; i < obj->n_half_edges; i++)
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
  for (int i = 0; i < obj->n_vertices; i++)
  {
    assert(obj->vertices[i].out_he_idx != MFTL_ERROR);
    obj->vertices[i].out_he_idx = h_map[obj->vertices[i].out_he_idx];
  }
  // Triangles
  for (int i = 0; i < obj->n_triangles; i++)
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
  for (int i = 0; i < obj->n_half_edges; i++)
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
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    int ret;
    khiter_t iter = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
    kh_value(obj->vertex_lookup, iter) = i;
  }

  // Triangles
  kh_clear(triangle_map, obj->triangle_lookup);
  kh_resize(triangle_map, obj->triangle_lookup, obj->n_triangles);
  for (int i = 0; i < obj->n_triangles; ++i)
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
    for (int i = 0; i < obj->n_half_edges; ++i)
    {
      int ret;
      khiter_t iter = kh_put(halfedge_map, obj->halfedge_lookup, obj->half_edges[i].global_id, &ret);
      kh_value(obj->halfedge_lookup, iter) = i;
    }
  }
  mftlRefreshTwinGids(obj);
  MFTLRefreshBoundaryCache(obj);
}

/**
 * @brief Performs one iteration of Tangential Laplacian Smoothing.
 * @param obj The mesh object.
 * @param mode The smoothing mode (Vertex-based or Triangle-based).
 * @param relaxation Factor between 0.0 and 1.0 (usually 1.0 for remeshing).
 */
void MFTLSmoothTangential(MFTL_Object *obj, MFTL_SmoothMode mode, double relaxation)
{
  if (obj->n_vertices == 0) return;

  // Allocate buffer for simultaneous (Jacobi) updates
  double (*new_pos)[3] = malloc(obj->n_vertices * sizeof(double[3]));
  assert(new_pos != MFTL_NULL);

  MFTL_Vertex *ring_verts[32];
  MFTL_Triangle *ring_tris[32];

  int smooth_iters = 3;
  for (int s = 0; s < smooth_iters; s++)
  {
    // --- PASS 1: Calculate New Positions ---
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &obj->vertices[i];

      // Skip dead or isolated vertices
      if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) continue;

      // Default: no movement
      new_pos[i][0] = v->pos[0];
      new_pos[i][1] = v->pos[1];
      new_pos[i][2] = v->pos[2];

      // ---------------------------------------------------------
      // FEATURE DETECTION: Circulate the 1-ring
      // ---------------------------------------------------------
      int bnd_count = 0;
      int feat_count = 0;
      double line_center[3] = {0, 0, 0};

      // Neighbour contributions are re-imaged into v's periodic image (exact
      // copies when non-periodic), so a ring crossing a seam stays contiguous.
      MFTL_HalfEdge *curr = MFTLVOutHe(obj, v);
      MFTL_HalfEdge *start = curr;

      // Circulate to count boundary and feature edges
      if (curr != MFTL_NULL)
      {
        do
        {
          if (curr->twin_idx == MFTL_ERROR)
          {
            bnd_count++;
            // Add neighbor position
            int nbr_v = obj->half_edges[curr->next_idx].vertex_idx;
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, obj->vertices[nbr_v].pos, v->pos);
            line_center[0] += u[0];
            line_center[1] += u[1];
            line_center[2] += u[2];
          }
          else if (curr->flags & MFTL_FLAG_FEATURE)
          {
            feat_count++;
            int nbr_v = curr->vertex_idx;
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, obj->vertices[nbr_v].pos, v->pos);
            line_center[0] += u[0];
            line_center[1] += u[1];
            line_center[2] += u[2];
          }

          MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
          if (!twin) break;
          curr = MFTLHeNext(obj, twin);
        } while (curr != start);
      }

      // ---------------------------------------------------------
      // APPLY SOTA FEATURE SMOOTHING
      // ---------------------------------------------------------
      if (bnd_count > 0 || feat_count > 2)
      {
        // CORNER OR COMPLEX BOUNDARY: 0D Lock
        // Position is already set to v->pos above, so we do nothing.
      }
      else if (feat_count == 2)
      {
        // RIDGE LINE: 1D Smoothing
        // Smooth only along the feature vector to preserve the sharp edge
        new_pos[i][0] = 0.5 * line_center[0];
        new_pos[i][1] = 0.5 * line_center[1];
        new_pos[i][2] = 0.5 * line_center[2];
      }
      else
      {
        // INTERIOR (feat_count == 0 or 1): 2D Tangential Smoothing
        double center[3] = {0, 0, 0};
        int degree = 0;

        if (mode == MFTL_SMOOTH_VERTEX_BASED)
        {
          degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, ring_verts[k]->pos, v->pos);
            center[0] += u[0];
            center[1] += u[1];
            center[2] += u[2];
          }
        }
        else  // MFTL_SMOOTH_TRIANGLE_BASED
        {
          degree = MFTLGetVertexOneRing(obj, v, ring_tris, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            MFTL_Vertex *t_verts[3];
            MFTLGetTriangleVertices(obj, ring_tris[k], t_verts);
            double u0[3], u1[3], u2[3];
            MFTLUnwrapToAnchor(obj->domain, u0, t_verts[0]->pos, v->pos);
            MFTLUnwrapToAnchor(obj->domain, u1, t_verts[1]->pos, v->pos);
            MFTLUnwrapToAnchor(obj->domain, u2, t_verts[2]->pos, v->pos);
            center[0] += (u0[0] + u1[0] + u2[0]) / 3.0;
            center[1] += (u0[1] + u1[1] + u2[1]) / 3.0;
            center[2] += (u0[2] + u1[2] + u2[2]) / 3.0;
          }
        }

        if (degree > 0)
        {
          center[0] /= degree;
          center[1] /= degree;
          center[2] /= degree;

          // 1. Calculate the displacement vector (Center - Current)
          double disp[3] = {center[0] - v->pos[0], center[1] - v->pos[1], center[2] - v->pos[2]};

          // 2. Compute Vertex Normal (Simple area-weighted or uniform)
          double nx = 0, ny = 0, nz = 0;
          degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            MFTL_Vertex *v1 = ring_verts[k];
            MFTL_Vertex *v2 = ring_verts[(k + 1) % degree];  // CCW neighbor

            double d1[3], d2[3];
            MFTLGetMinimumImageVector(obj->domain, d1, v->pos, v1->pos);
            MFTLGetMinimumImageVector(obj->domain, d2, v->pos, v2->pos);

            nx += (d1[1] * d2[2] - d1[2] * d2[1]);
            ny += (d1[2] * d2[0] - d1[0] * d2[2]);
            nz += (d1[0] * d2[1] - d1[1] * d2[0]);
          }

          // Normalize normal
          double n_len = sqrt(nx * nx + ny * ny + nz * nz);
          if (n_len > 1e-12)
          {
            nx /= n_len;
            ny /= n_len;
            nz /= n_len;
          }

          // 3. Project displacement onto the Tangent Plane
          double dot = disp[0] * nx + disp[1] * ny + disp[2] * nz;

          double tangent_disp[3] = {disp[0] - dot * nx, disp[1] - dot * ny, disp[2] - dot * nz};

          // 4. Apply Tangential Update
          new_pos[i][0] = v->pos[0] + relaxation * tangent_disp[0];
          new_pos[i][1] = v->pos[1] + relaxation * tangent_disp[1];
          new_pos[i][2] = v->pos[2] + relaxation * tangent_disp[2];
        }
      }
    }

    // --- PASS 2: Apply Updates ---
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if ((obj->vertices[i].flags & MFTL_FLAG_DEAD) || obj->vertices[i].out_he_idx == MFTL_ERROR) continue;

      obj->vertices[i].pos[0] = new_pos[i][0];
      obj->vertices[i].pos[1] = new_pos[i][1];
      obj->vertices[i].pos[2] = new_pos[i][2];
      // keep the wrapped-storage invariant (no-op when non-periodic); new_pos = pos + short move
      MFTLWrapPosImg(obj->domain, obj->vertices[i].pos, obj->vertices[i].img);
    }
  }
  free(new_pos);
}

// Adds MFTL_FLAG_FEATURE to half-edges that exceed the angle threshold
void MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees)
{
  // Convert angle to dot-product threshold for fast comparison
  // e.g., 30 degrees -> cos(30) = 0.866. If dot < 0.866, angle > 30.
  double threshold_dot = MFTL_COS(angle_degrees * MFTL_PI / 180.0);

  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &obj->half_edges[i];

    // Skip dead edges, boundaries, or already processed twins
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (he->twin_idx == MFTL_ERROR)
    {
      // It's a boundary! Mark it as a feature to protect it.
      he->flags |= MFTL_FLAG_FEATURE;
      continue;
    }
    if (i > he->twin_idx) continue;

    int t1_idx = he->triangle_idx;
    int t2_idx = obj->half_edges[he->twin_idx].triangle_idx;

    // Get pre-computed triangle normals
    double *n1 = obj->triangles[t1_idx].normal;
    double *n2 = obj->triangles[t2_idx].normal;

    double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];

    // If the angle between normals is larger than the threshold, mark it!
    if (dot < threshold_dot)
    {
      he->flags |= MFTL_FLAG_FEATURE;
      obj->half_edges[he->twin_idx].flags |= MFTL_FLAG_FEATURE;
    }
    else
    {
      // Ensure flag is cleared if geometry smoothed out
      he->flags &= ~MFTL_FLAG_FEATURE;
      obj->half_edges[he->twin_idx].flags &= ~MFTL_FLAG_FEATURE;
    }
  }
}

/* ------------------------------------------------------------------------------------------
 * Stage 16C (plans/16 §4.4.4): class-guarded volume recovery of the remesh drivers
 * ------------------------------------------------------------------------------------------ */
void MFTLRemeshRecoveryTarget(MFTL_Object *obj, MFTL_RemeshParameters *params, MFTL_RecoveryTarget *t)
{
  t->kind = 0;
  t->target = 0.0;
  MFTLObjectEnsureWinding(obj);  // collective; unknown winding never counts as compact
  if (MFTLObjectHasVolume(obj))
  {
    // closed and compact: the body's own volume (Stage-15 behaviour, bit-identical)
    t->kind = 1;
    t->target = MFTLObjectVolume(obj);
    return;
  }
  if (params->skip_volume_recovery) return;
  // open or wrapping: the singleton phase set decides (never MFTLObjectVolume)
  MFTL_PhaseSet *S = MFTLPhaseSetCreate(obj->domain);
  MFTLPhaseSetAddMember(S, obj, 1);
  double V = 0.0;
  if (MFTLPhaseSetVolume(S, &V) == MFTL_DEFINED)
  {
    t->kind = 2;
    t->target = V;
  }
  else
    params->stat_recovery_skipped++;
  MFTLPhaseSetDestroy(&S);
}

void MFTLRemeshRecover(MFTL_Ensemble *ens, MFTL_Object *obj, const MFTL_RecoveryTarget *t)
{
  if (t->kind == 1)
    MFTLVolumeRecovery(obj, t->target);
  else if (t->kind == 2)
  {
    MFTL_PhaseSet *S = MFTLPhaseSetCreate(obj->domain);
    MFTLPhaseSetAddMember(S, obj, 1);
    MFTLPhaseSetVolumeRecovery(S, ens, t->target, MFTL_NULL);
    MFTLPhaseSetDestroy(&S);
  }
}

void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  int iter = 0;

  bool collapses_occurred[ens->num_objects];
  // --- Snapshot Initial Counts ---
  int start_n_vertices[ens->num_objects];
  int start_n_triangles[ens->num_objects];
  int start_n_halfedges[ens->num_objects];
  MFTL_RecoveryTarget start_volume[ens->num_objects];

  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];

    collapses_occurred[i] = MFTL_FALSE;
    start_n_vertices[i] = obj->n_vertices;
    start_n_triangles[i] = obj->n_triangles;
    start_n_halfedges[i] = obj->n_half_edges;
    // Stage 09 recovery target (collective); Stage 16B/16C class guard: the body's own volume only
    // for closed compact bodies, a set-level target for DEFINED singleton phase sets, else none
    MFTLRemeshRecoveryTarget(obj, params, &start_volume[i]);
  }

  // Loop continues IF:
  // 1. Iterations is set to MFTL_ERROR (Run until convergence)
  // 2. OR we haven't reached the user-defined iteration limit
  while (params->iterations == MFTL_ERROR || iter < params->iterations)
  {
    if (!params->quiet) printf("--- Iteration %d ---\n", iter + 1);

    int total_split = 0;
    int total_collapse = 0;
    int total_swap = 0;

    for (int i = 0; i < ens->num_objects; i++)
    {
      MFTL_Object *obj = ens->objects[i];

      int obj_split = 0;
      int obj_collapse = 0;
      int obj_swap = 0;

      // Stage 08: refresh geometry for the criteria, fitting and smoothing,
      // and re-detect features EVERY iteration: the ops change the geometry
      // and feature protection must track it without the caller pre-setting
      // flags. Detection consumes triangle normals, so it runs after the
      // property refresh.
      MFTLComputeTriangleProperties(obj);
      MFTLDetectFeatures(obj, params->feature_angle_threshold);
      if (params->vertex_fit != MFTL_FIT_NONE || params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE ||
          params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        MFTLComputeVertexNormals(obj);
        if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE || params->smooth_mode == MFTL_SMOOTH_CURVATURE_BASED)
          MFTLComputeVertexCurvatures(obj);
      }

      // Stage 16C (plans/16 §4.4.4): the parity path only for closed bodies in the default mode
      const int parity = params->mftl1_criteria && !MFTLObjectRemeshGeneralized(obj);
      if (parity)
      {
        // Stage 15: MFTL1's sweeps (MFTL_Refining / MFTL_Coarsening / MFTL_Flipping)
        MFTL1Sweeps(obj, params, &obj_split, &obj_collapse, &obj_swap);
        if (obj_collapse > 0) collapses_occurred[i] = MFTL_TRUE;
      }

      // 1. SPLIT
      for (int j = obj->n_half_edges - 1; j >= 0 && !params->skip_split && !parity; j--)
      {
        if (MFTLSplitApproved(obj, j, params))
        {
          MFTLSplitEdgeWithFit(obj, j, params);
          obj_split++;
        }
      }

      // 2. COLLAPSE
      for (int j = 0; j < obj->n_half_edges && !params->skip_collapse && !parity; ++j)
      {
        if (CollapseApproved(obj, j, params))
        {
          MFTLCollapseEdgeWithFit(obj, j, params);
          collapses_occurred[i] = MFTL_TRUE;
          obj_collapse++;
        }
      }

      // 3. SWAP / FLIP (Valence Optimization + quality/dihedral gates)
      for (int j = 0; j < obj->n_half_edges && !params->skip_flip && !parity; ++j)
      {
        MFTL_HalfEdge *he = &obj->half_edges[j];

        // Skip dead edges
        if (he->flags & MFTL_FLAG_DEAD) continue;

        // Skip boundary edges
        if (he->twin_idx == MFTL_ERROR) continue;

        // Only process the half-edge with the smaller index (avoid checking twin)
        if (j > he->twin_idx) continue;

        // Skip feature edges
        if (he->flags & MFTL_FLAG_FEATURE) continue;

        // Now evaluate for flip
        if (FlipApprovedWithParams(obj, j, params))
        {
          MFTLFlipEdge(obj, j);
          obj_swap++;
        }
      }

      // 4. SMOOTH (Tangential; mode/iterations from params)
      if (!params->skip_tangential_smoothing) MFTLSmoothTangentialWithParams(obj, params);

      // Reset visited flags
      for (int j = 0; j < obj->n_half_edges; ++j)
      {
        obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
      }

      for (int j = 0; j < obj->n_vertices; ++j)
      {
        obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;
      }

      // Accumulate totals
      params->stat_splits += obj_split;
      params->stat_collapses += obj_collapse;
      params->stat_flips += obj_swap;
      total_split += obj_split;
      total_collapse += obj_collapse;
      total_swap += obj_swap;
    }

    // WE NEED TO UPDATE THE GEOMETRIC PROPERTIES HERE!!!!!
    // NORMALS AREA ETC

    if (!params->quiet) printf("Operations -> Split: %d | Collapsed: %d | Swapped: %d\n", total_split, total_collapse, total_swap);

    // --- CONVERGENCE CHECK ---
    // On a single rank (or a ghost-complete mesh) the local op count is the
    // whole story. When the serial kernel runs on a distributed ensemble
    // (MFTLRemeshEnsembleDispersedCentering gathers to rank 0; other ranks
    // hold empty objects), the decision must be collective — otherwise the
    // empty ranks would exit the loop early and desynchronize the
    // collectives in the Stage-09 stages below.
    int converged = !params->fixed_iterations && (total_split + total_collapse + total_swap) < 10;
    if (ens->domain.nproc > 1)
    {
      int converged_g = 0;
      MPI_Allreduce(&converged, &converged_g, 1, MPI_INT, MPI_MIN, ens->domain.comm_cart);
      converged = converged_g;
    }
    if (converged)
    {
      if (!params->quiet) printf("Remeshing converged successfully after %d iterations.\n", iter + 1);
      break;
    }

    // --- FAILSAFE ---
    // Prevent infinite loops if floating-point drift causes non-convergence
    if (iter >= 15 && params->iterations == MFTL_ERROR)
    {
      if (!params->quiet) printf("Warning: Remeshing reached failsafe limit (15 iterations). Exiting to prevent hang.\n");
      break;
    }

    iter++;
  }

  // Reset the arrays and hash maps and delete the dead elements
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    MFTLFinalizeRemeshing(obj, start_n_vertices[i], start_n_triangles[i], start_n_halfedges[i], collapses_occurred[i]);
  }

  // --- Stage 09: volume-conserving post-remesh stages (finalized mesh) ---
  // PRECONDITION for the edge-local smoothers: a complete one-ring around
  // every processed vertex must be stored (rank-0-local mesh, or ghosts via
  // MFTLExtendGhostLayer + MFTLUpdateGhostPositions); incomplete rings skip.
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    MFTLComputeTriangleProperties(obj);

    if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_KUPRAT)
      MFTLVolumeConservingSmoothing(obj, params);
    else if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_TSUR)
      MFTLTSUR3D(obj, params);

    if (params->roughness_smoothing) MFTLRoughnessSmoothing(obj, params);
  }

  // ALWAYS finish a remesh cycle with global volume recovery against the
  // pre-remesh volume: per-edge local conservation is first-order only and
  // drift accumulates across the split/collapse/flip/smooth pipeline.
  for (int i = 0; i < ens->num_objects; i++)
  {
    MFTL_Object *obj = ens->objects[i];
    if (!params->skip_volume_recovery) MFTLRemeshRecover(ens, obj, &start_volume[i]);
    MFTLComputeTriangleProperties(obj);
  }
  MFTLUpdateGhostPositions(ens, NULL);
}

/* ========================================================================== */
/* Stage 08 — criteria support, gated flips, params-driven smoothing          */
/* ========================================================================== */

// MFTL1 triangle quality: q = (perimeter^2 / area) / 20.78460969.
// 1.0 = equilateral; LARGER is worse.
double MFTL_TriangleQuality(const MFTL_Object* obj, int tri_idx)
{
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);
  // minimum-image edge vectors (raw differences when non-periodic)
  double v01[3], v12[3], v20[3], v02[3];
  MFTLGetMinimumImageVector(obj->domain, v01, vv[0]->pos, vv[1]->pos);
  MFTLGetMinimumImageVector(obj->domain, v12, vv[1]->pos, vv[2]->pos);
  MFTLGetMinimumImageVector(obj->domain, v20, vv[2]->pos, vv[0]->pos);
  MFTLGetMinimumImageVector(obj->domain, v02, vv[0]->pos, vv[2]->pos);
  double e0 = v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2];
  double e1 = v12[0] * v12[0] + v12[1] * v12[1] + v12[2] * v12[2];
  double e2 = v20[0] * v20[0] + v20[1] * v20[1] + v20[2] * v20[2];
  double a = sqrt(e0), b = sqrt(e1), c = sqrt(e2);
  double p = a + b + c;
  double cx = v01[1] * v02[2] - v01[2] * v02[1];
  double cy = v01[2] * v02[0] - v01[0] * v02[2];
  double cz = v01[0] * v02[1] - v01[1] * v02[0];
  double A = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  if (A < MFTL_VERYSMALL) return 1e9;  // degenerate: worst quality
  return (p * p / A) / 20.78460969;
}

// Botsch-Kobbelt curvature-adaptive thresholds (MFTL1 edgesplit.c parity).
// kappa_max = mean + sqrt(mean^2 - gauss) at both endpoints;
// L = sqrt(6*eps/kappa - 3*eps^2) clamped to [min, max] edge length;
// split (flag 1) above 4L/3, collapse (flag 2) below 0.8L.
int MFTLBotschKobbeltCurvatureThreshold(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params, int split_or_collapse)
{
  MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v1 = obj->half_edges[he_prev].vertex_idx;
  int v2 = he->vertex_idx;

  double k1 = obj->vertices[v1].meancurvature + sqrt(MFTL_MAX(0.0, MFTL_SQ(obj->vertices[v1].meancurvature) - obj->vertices[v1].gaussiancurvature));
  double k2 = obj->vertices[v2].meancurvature + sqrt(MFTL_MAX(0.0, MFTL_SQ(obj->vertices[v2].meancurvature) - obj->vertices[v2].gaussiancurvature));
  k1 = MFTL_MAX(k1, MFTL_VERYSMALL);
  k2 = MFTL_MAX(k2, MFTL_VERYSMALL);

  double L1 = sqrt(MFTL_MAX(0.0, (6.0 * params->epsilon / k1) - 3.0 * MFTL_SQ(params->epsilon)));
  double L2 = sqrt(MFTL_MAX(0.0, (6.0 * params->epsilon / k2) - 3.0 * MFTL_SQ(params->epsilon)));
  double L = MFTL_MIN(L1, L2);
  L = MFTL_MIN(L, params->max_edge_length);
  L = MFTL_MAX(L, params->min_edge_length);

  double high = 4.0 * L / 3.0;
  double low = 0.8 * L;
  double edge_length = sqrt(MFTLGetEdgeLengthSquared(obj, he));

  if (split_or_collapse == 1) return edge_length > high;
  return edge_length < low;
}

static inline double mftlTriQualityFromPos(const MFTL_Domain* dom, const double a[3], const double b[3], const double c[3])
{
  // minimum-image edge vectors (raw differences when non-periodic)
  double v01[3], v12[3], v20[3], v02[3];
  MFTLGetMinimumImageVector(dom, v01, a, b);
  MFTLGetMinimumImageVector(dom, v12, b, c);
  MFTLGetMinimumImageVector(dom, v20, c, a);
  MFTLGetMinimumImageVector(dom, v02, a, c);
  double e0 = 0.0, e1 = 0.0, e2 = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    e0 += v01[k] * v01[k];
    e1 += v12[k] * v12[k];
    e2 += v20[k] * v20[k];
  }
  double p = sqrt(e0) + sqrt(e1) + sqrt(e2);
  double cx = v01[1] * v02[2] - v01[2] * v02[1];
  double cy = v01[2] * v02[0] - v01[0] * v02[2];
  double cz = v01[0] * v02[1] - v01[1] * v02[0];
  double A = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  if (A < MFTL_VERYSMALL) return 1e9;
  return (p * p / A) / 20.78460969;
}

// FlipApproved extended with the Stage-08 gate stack: MFTL1 quality threshold
// and the max-angle-between-triangles (dihedral) gate. params == NULL keeps
// the legacy valence-only behaviour.
int FlipApprovedWithParams(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  if (!MFTLHeIsLocal(&obj->half_edges[he_idx])) return MFTL_FALSE;  // Stage 16A: interior LOCAL edges only
  if (params && params->mftl1_criteria && !MFTLObjectRemeshGeneralized(obj)) return MFTL1FlipApproved(obj, he_idx, params);  // Stage 15
  if (!FlipApproved(obj, he_idx)) return MFTL_FALSE;
  if (!params) return MFTL_TRUE;

  MFTL_HalfEdge* he_ab = &obj->half_edges[he_idx];
  int he_ba_idx = he_ab->twin_idx;
  int v_b = he_ab->vertex_idx;
  int v_a = obj->half_edges[he_ba_idx].vertex_idx;
  int v_c = obj->half_edges[he_ab->next_idx].vertex_idx;
  int v_d = obj->half_edges[obj->half_edges[he_ba_idx].next_idx].vertex_idx;

  const double* pa = obj->vertices[v_a].pos;
  const double* pb = obj->vertices[v_b].pos;
  const double* pc = obj->vertices[v_c].pos;
  const double* pd = obj->vertices[v_d].pos;

  // Quality gate: reject when a created triangle would be worse than the
  // configured worst quality.
  double q1 = mftlTriQualityFromPos(obj->domain, pc, pd, pa);
  double q2 = mftlTriQualityFromPos(obj->domain, pc, pd, pb);
  if (q1 > params->max_quality_threshold || q2 > params->max_quality_threshold) return MFTL_FALSE;

  // Stage 16C (generalized bodies): the created triangles are (c, a, d) and (d, b, c) with the
  // orientation of the original pair; the dihedral gate compares their consistently oriented
  // normals, so a flip of a non-convex quad (one created triangle inverted) is always rejected.
  // NOTE: the legacy gate below builds both normals from (c, d, x), which inverts the sign of the
  // dot product (it rejects convex flips and accepts folding ones). It is kept unchanged for closed
  // bodies in the default mode (bit-identity contract, plans/16 §0); see plans/16 §22.
  if (MFTLObjectRemeshGeneralized(obj))
  {
    double ca[3], cd[3], db[3], dc[3];
    MFTLGetMinimumImageVector(obj->domain, ca, pc, pa);
    MFTLGetMinimumImageVector(obj->domain, cd, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, db, pd, pb);
    MFTLGetMinimumImageVector(obj->domain, dc, pd, pc);
    const double n1[3] = {ca[1] * cd[2] - ca[2] * cd[1], ca[2] * cd[0] - ca[0] * cd[2], ca[0] * cd[1] - ca[1] * cd[0]};
    const double n2[3] = {db[1] * dc[2] - db[2] * dc[1], db[2] * dc[0] - db[0] * dc[2], db[0] * dc[1] - db[1] * dc[0]};
    const double l1 = sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]), l2 = sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    if (!(l1 > 0.0) || !(l2 > 0.0)) return MFTL_FALSE;  // degenerate created triangle
    double dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2);
    dot = dot > 1.0 ? 1.0 : (dot < -1.0 ? -1.0 : dot);
    const double max_angle = params->max_angle_between_tris < 179.0 ? params->max_angle_between_tris : 179.0;
    if (acos(dot) * 180.0 / MFTL_PI > max_angle) return MFTL_FALSE;
    // neither created triangle may invert with respect to the original pair (normals from the
    // current positions: cached triangle normals are stale within a pass)
    double ab[3], ac[3], ba[3], bd[3];
    MFTLGetMinimumImageVector(obj->domain, ab, pa, pb);
    MFTLGetMinimumImageVector(obj->domain, ac, pa, pc);
    MFTLGetMinimumImageVector(obj->domain, ba, pb, pa);
    MFTLGetMinimumImageVector(obj->domain, bd, pb, pd);
    const double no1[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
    const double no2[3] = {ba[1] * bd[2] - ba[2] * bd[1], ba[2] * bd[0] - ba[0] * bd[2], ba[0] * bd[1] - ba[1] * bd[0]};
    const double nold[3] = {no1[0] + no2[0], no1[1] + no2[1], no1[2] + no2[2]};
    if (n1[0] * nold[0] + n1[1] * nold[1] + n1[2] * nold[2] <= 0.0) return MFTL_FALSE;
    if (n2[0] * nold[0] + n2[1] * nold[1] + n2[2] * nold[2] <= 0.0) return MFTL_FALSE;
    return MFTL_TRUE;
  }

  // Dihedral gate: reject when the angle between the new triangle normals
  // exceeds max_angle_between_tris (degrees).
  if (params->max_angle_between_tris < 179.0)
  {
    double n1[3], n2[3];
    // minimum-image legs (raw differences when non-periodic)
    double u1[3], u2[3], w1[3], w2[3];
    MFTLGetMinimumImageVector(obj->domain, u1, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, u2, pc, pa);
    MFTLGetMinimumImageVector(obj->domain, w1, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, w2, pc, pb);
    n1[0] = u1[1] * u2[2] - u1[2] * u2[1];
    n1[1] = u1[2] * u2[0] - u1[0] * u2[2];
    n1[2] = u1[0] * u2[1] - u1[1] * u2[0];
    n2[0] = w1[1] * w2[2] - w1[2] * w2[1];
    n2[1] = w1[2] * w2[0] - w1[0] * w2[2];
    n2[2] = w1[0] * w2[1] - w1[1] * w2[0];
    double l1 = sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
    double l2 = sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    if (l1 > MFTL_VERYSMALL && l2 > MFTL_VERYSMALL)
    {
      double dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2);
      if (dot > 1.0) dot = 1.0;
      if (dot < -1.0) dot = -1.0;
      double angle = acos(dot) * 180.0 / MFTL_PI;
      if (angle > params->max_angle_between_tris) return MFTL_FALSE;
    }
  }

  return MFTL_TRUE;
}

// Tangential smoothing driven by params. QUALITY_DRIVEN accepts a move only
// when the local minimum triangle quality improves; CURVATURE_BASED weights
// the centroid by area * L_target(curvature) (MFTL1 tangentialrelaxation.c).
// Feature vertices and corners are left fixed in the Stage-08 modes.
void MFTLSmoothTangentialWithParams(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  if (obj->n_vertices == 0) return;
  double relaxation = 1.0;
  int iters = params && params->max_smoothing_iter > 0 ? params->max_smoothing_iter : 3;
  MFTL_SmoothMode mode = params ? params->smooth_mode : MFTL_SMOOTH_VERTEX_BASED;

  double (*new_pos)[3] = malloc(obj->n_vertices * sizeof(double[3]));
  assert(new_pos != MFTL_NULL);
  MFTL_Vertex* ring_verts[64];
  MFTL_Triangle* ring_tris[32];
  const int generalized = MFTLObjectRemeshGeneralized(obj);  // Stage 16C

  for (int s = 0; s < iters; ++s)
  {
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost) continue;  // Stage 10: ghosts live on their owner rank; never move them locally
      if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) continue;
      new_pos[i][0] = v->pos[0];
      new_pos[i][1] = v->pos[1];
      new_pos[i][2] = v->pos[2];

      // feature / boundary census
      int bnd_count = 0;
      int feat_count = 0;
      int he = v->out_he_idx;
      int start = he;
      do
      {
        MFTL_HalfEdge* curr = &obj->half_edges[he];
        if (curr->twin_idx == MFTL_ERROR)
          bnd_count++;
        else if (curr->flags & MFTL_FLAG_FEATURE)
          feat_count++;
        if (curr->twin_idx == MFTL_ERROR) break;
        he = obj->half_edges[curr->twin_idx].next_idx;
      } while (he != start && he != MFTL_ERROR);

      int locked = (bnd_count > 0 || feat_count > 0);
      // Stage 16C: a generalized fan larger than the one-ring buffers below (refinement around a
      // high-valence vertex) is left in place instead of overflowing them
      if (generalized)
      {
        int nfan = 0;
        MFTLVertexRing(obj, i, MFTL_NULL, 0, &nfan);
        if (nfan >= 32) continue;
      }
      if (mode == MFTL_SMOOTH_QUALITY_DRIVEN || mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        // Stage-08 modes: feature vertices are fixed
        if (locked) continue;
      }
      else if (locked)
      {
        continue;  // legacy modes keep the existing ridge/corner handling elsewhere
      }

      // candidate centroid
      double center[3] = {0, 0, 0};
      double denom = 0.0;
      if (mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        // area * L_target weighted triangle centres (MFTL1 CurvatureBased)
        int nt = MFTLGetVertexOneRing(obj, v, ring_tris, 32, MFTL_TRUE);
        for (int k = 0; k < nt; ++k)
        {
          MFTL_Triangle* t = ring_tris[k];
          double Lavg = 0.0;
          MFTL_Vertex* tv[3];
          MFTLGetTriangleVertices(obj, t, tv);
          for (int j = 0; j < 3; ++j)
          {
            double mc = tv[j]->meancurvature;
            double gc = tv[j]->gaussiancurvature;
            double kmax = mc + sqrt(MFTL_MAX(0.0, MFTL_SQ(mc) - gc));
            kmax = MFTL_MAX(kmax, MFTL_VERYSMALL);
            double Lv = sqrt(MFTL_MAX(0.0, 6.0 * params->epsilon / kmax - 3.0 * MFTL_SQ(params->epsilon)));
            Lv = MFTL_MIN(Lv, params->max_edge_length);
            Lv = MFTL_MAX(Lv, params->min_edge_length);
            Lavg += Lv / 3.0;
          }
          double w = t->area * Lavg;
          // re-image the stored (wrapped) centroid into v's periodic image
          double tc[3];
          MFTLUnwrapToAnchor(obj->domain, tc, t->center, v->pos);
          center[0] += w * tc[0];
          center[1] += w * tc[1];
          center[2] += w * tc[2];
          denom += w;
        }
      }
      else
      {
        int degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 64, MFTL_TRUE);
        for (int k = 0; k < degree; ++k)
        {
          // neighbour re-imaged into v's periodic image (plain copy otherwise)
          double u[3];
          MFTLUnwrapToAnchor(obj->domain, u, ring_verts[k]->pos, v->pos);
          center[0] += u[0];
          center[1] += u[1];
          center[2] += u[2];
        }
        denom = degree;
      }
      if (denom <= 0) continue;
      center[0] /= denom;
      center[1] /= denom;
      center[2] /= denom;

      // vertex normal for the tangential projection
      double nx = 0, ny = 0, nz = 0;
      int degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 64, MFTL_TRUE);
      for (int k = 0; k < degree; ++k)
      {
        MFTL_Vertex* v1 = ring_verts[k];
        MFTL_Vertex* v2 = ring_verts[(k + 1) % degree];
        double d1[3], d2[3];
        MFTLGetMinimumImageVector(obj->domain, d1, v->pos, v1->pos);
        MFTLGetMinimumImageVector(obj->domain, d2, v->pos, v2->pos);
        nx += (d1[1] * d2[2] - d1[2] * d2[1]);
        ny += (d1[2] * d2[0] - d1[0] * d2[2]);
        nz += (d1[0] * d2[1] - d1[1] * d2[0]);
      }
      double n_len = sqrt(nx * nx + ny * ny + nz * nz);
      if (n_len < 1e-12) continue;
      nx /= n_len;
      ny /= n_len;
      nz /= n_len;

      double disp[3] = {center[0] - v->pos[0], center[1] - v->pos[1], center[2] - v->pos[2]};
      double dot = disp[0] * nx + disp[1] * ny + disp[2] * nz;
      double cand[3] = {v->pos[0] + relaxation * (disp[0] - dot * nx), v->pos[1] + relaxation * (disp[1] - dot * ny),
                        v->pos[2] + relaxation * (disp[2] - dot * nz)};

      if (mode == MFTL_SMOOTH_QUALITY_DRIVEN)
      {
        // accept only when the local minimum quality improves
        MFTL_Triangle* tris[32];
        int nt = MFTLGetVertexOneRing(obj, v, tris, 32, MFTL_TRUE);
        double q_before = 0.0, q_after = 0.0;
        for (int k = 0; k < nt; ++k)
        {
          MFTL_Vertex* tv[3];
          MFTLGetTriangleVertices(obj, tris[k], tv);
          double qb = mftlTriQualityFromPos(obj->domain, tv[0]->pos, tv[1]->pos, tv[2]->pos);
          double p0[3], p1[3], p2[3];
          memcpy(p0, tv[0]->pos, sizeof(p0));
          memcpy(p1, tv[1]->pos, sizeof(p1));
          memcpy(p2, tv[2]->pos, sizeof(p2));
          if (tv[0] == v) memcpy(p0, cand, sizeof(p0));
          if (tv[1] == v) memcpy(p1, cand, sizeof(p1));
          if (tv[2] == v) memcpy(p2, cand, sizeof(p2));
          double qa = mftlTriQualityFromPos(obj->domain, p0, p1, p2);
          if (qb > q_before) q_before = qb;
          if (qa > q_after) q_after = qa;
        }
        if (q_after >= q_before) continue;  // no improvement: keep position
      }
      // Stage 16C: generalized bodies never accept a move that inverts a fan triangle
      if (MFTLObjectRemeshGeneralized(obj) && !MFTLMoveKeepsOrientation(obj, i, v->pos, cand)) continue;
      // ... nor a move racing with another rank's move of a fan corner (checked only after the ghost update)
      if (MFTLObjectRemeshGeneralized(obj) && !MFTLVertexFanOwned(obj, i)) continue;

      memcpy(new_pos[i], cand, sizeof(cand));
    }

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if (obj->vertices[i].is_ghost) continue;  // Stage 10: ghosts are synced from their owner, never written here
      if ((obj->vertices[i].flags & MFTL_FLAG_DEAD) || obj->vertices[i].out_he_idx == MFTL_ERROR) continue;
      // Stage 16C: the Jacobi candidates were checked against the OLD neighbour positions; for
      // generalized bodies each move is re-checked against the current ones when it is applied, so
      // simultaneous neighbour moves can never fold the surface
      if (generalized && !MFTLMoveKeepsOrientation(obj, i, obj->vertices[i].pos, new_pos[i])) continue;
      double old_pos[3];
      memcpy(old_pos, obj->vertices[i].pos, 3 * sizeof(double));
      memcpy(obj->vertices[i].pos, new_pos[i], 3 * sizeof(double));
      // keep the wrapped-storage invariant (no-op when non-periodic)
      MFTLApplyPeriodicity(obj->domain, obj->vertices[i].pos);
      MFTLImgFollow(obj->domain, obj->vertices[i].img, old_pos, obj->vertices[i].pos);  // Stage 16B
    }
    // Stage 16C (plans/16 §4.4.2): boundary vertices of a generalized body move only along their
    // boundary curve (arc-length redistribution); corners and frozen tags stay fixed
    if (MFTLObjectRemeshGeneralized(obj)) MFTLSmoothBoundaryVertices(obj, relaxation, MFTLBoundaryCornerAngle(params));
  }
  free(new_pos);
}
