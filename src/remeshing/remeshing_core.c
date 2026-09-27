/*
 * MFTL2 remeshing — the edge split / collapse / flip operators, their approval gates
 * (criteria, link condition, feature and boundary protection) and the compaction after a pass
 * (MFTLFinalizeRemeshing). Smoothing, feature detection and the quality gates are in
 * remesh_smoothing.c; the serial driver and the class-guarded volume recovery in
 * remesh_driver.c.
 *
 * Operator contract: operators act rank-locally on stored entities and keep the twin states
 * (plans/16 §2.2): split/flip create LOCAL pairs, a boundary split keeps the parent's tag,
 * collapse merges the wing states (mftlGlueWings) and is refused for REMOTE wings. New vertices
 * follow the displacement rule (anchored minimum-image midpoints, plans/16 §3.2). Removed
 * entities are only flagged MFTL_FLAG_DEAD during a pass; MFTLFinalizeRemeshing compacts.
 * The parallel driver (remeshing_parallel.c) calls these operators only on edges that are
 * interior to one rank and assigns disjoint gid blocks beforehand.
 */

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
#include "remeshing_internal.h"

// Entity allocators: append one entity and return its local index.
static int mftlNewVertex(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);
  int idx = obj->n_vertices++;

  obj->vertices[idx].out_he_idx = MFTL_ERROR;
  obj->vertices[idx].flags = MFTL_FLAG_NONE;
  obj->vertices[idx].img[0] = obj->vertices[idx].img[1] = obj->vertices[idx].img[2] = 0;  // set by the operator
  return idx;
}

static int mftlNewTriangle(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  int idx = obj->n_triangles++;

  obj->triangles[idx].he_idx = MFTL_ERROR;
  obj->triangles[idx].flags = MFTL_FLAG_NONE;
  obj->triangles[idx].owner_proc = obj->domain->rank;
  obj->triangles[idx].is_ghost = MFTL_FALSE;
  obj->triangles[idx].surface_scalar = 0.0;  // set by the operator's mass remap
  // The opaque payload is NOT repartitioned by split/collapse: new triangles carry a zeroed
  // blob; callers that attach data impose their own split/merge policy (plans/12).
  if (obj->data_size > 0)
    obj->triangles[idx].data = calloc(1, obj->data_size);
  else
    obj->triangles[idx].data = MFTL_NULL;
  return idx;
}

static int mftlNewHalfEdge(MFTL_Object *obj)
{
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 1, MFTL_HalfEdge);
  int idx = obj->n_half_edges++;

  obj->half_edges[idx].twin_idx = MFTL_ERROR;
  obj->half_edges[idx].twin_gid = MFTL_TWIN_UNKNOWN;  // set by the operator
  obj->half_edges[idx].boundary = MFTL_FALSE;
  obj->half_edges[idx].next_idx = MFTL_ERROR;
  obj->half_edges[idx].vertex_idx = MFTL_ERROR;
  obj->half_edges[idx].triangle_idx = MFTL_ERROR;
  obj->half_edges[idx].flags = MFTL_FLAG_NONE;
  return idx;
}

/**
 * @brief Splits the edge of half-edge A->B at its midpoint M: 2 -> 4 triangles for an interior
 * (LOCAL) edge, 1 -> 2 for a BOUNDARY edge (both halves keep the tag). REMOTE edges are not
 * split. M gets the next vertex gid and is anchored at A (displacement rule); the caller fits
 * M onto the surface afterwards (MFTLSplitEdgeWithFit).
 * @return The local index of M.
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
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  int he_ba_idx = he_ab->twin_idx;
  // only LOCAL (interior) or BOUNDARY edges are split; a BOUNDARY edge's two halves inherit its
  // state code (tag)
  MFTL_ASSERT(!MFTLHeIsRemote(he_ab) && "MFTLSplitEdge: REMOTE edge");
  const int parent_code = he_ab->twin_gid;

  int he_next = obj->half_edges[he_ab_idx].next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v_a_idx = obj->half_edges[he_prev].vertex_idx;
  int v_b_idx = he_ab->vertex_idx;

  int v_m_idx = mftlNewVertex(obj);
  MFTL_Vertex *v_m = &obj->vertices[v_m_idx];
  MFTL_Vertex *v_a = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b = &obj->vertices[v_b_idx];

  // Seam-safe midpoint: with wrapped storage the raw average of a seam-crossing edge lands half
  // a domain away. Branch so the non-periodic arithmetic stays bit-identical.
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    double e_ab[3];
    MFTLGetMinimumImageVector(obj->domain, e_ab, v_a->pos, v_b->pos);
    for (int i = 0; i < 3; i++)
    {
      v_m->pos[i] = v_a->pos[i] + 0.5 * e_ab[i];
    }
    // displacement rule: anchor a (edge a->M has zero lattice shift)
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
  // New entities are owned by this rank; ownership is re-derived at the next redistribution. In
  // parallel the driver has set next_*_gid to a block reserved for this rank.
  v_m->owner_proc = obj->domain->rank;
  v_m->is_ghost = MFTL_FALSE;
  v_m->global_id = obj->next_vertex_gid++;

  // Upper triangle T1(A,B,C) -> T1(A,M,C) + T2(M,B,C)

  int t1_idx = he_ab->triangle_idx;
  int he_bc_idx = he_ab->next_idx;
  int he_ca_idx = obj->half_edges[he_bc_idx].next_idx;
  int v_c_idx = obj->half_edges[he_bc_idx].vertex_idx;

  int t2_idx = mftlNewTriangle(obj);
  int he_mb_idx = mftlNewHalfEdge(obj);
  int he_mc_idx = mftlNewHalfEdge(obj);
  int he_cm_idx = mftlNewHalfEdge(obj);

  // T1 = (A, M, C)
  obj->triangles[t1_idx].he_idx = he_ab_idx;  // A->M

  // A->M re-uses he_ab; its twin is set with the lower triangle
  obj->half_edges[he_ab_idx].vertex_idx = v_m_idx;
  obj->half_edges[he_ab_idx].next_idx = he_mc_idx;
  obj->half_edges[he_ab_idx].twin_idx = MFTL_ERROR;
  obj->half_edges[he_ab_idx].twin_gid = MFTL_TWIN_UNKNOWN;  // A->M: set below (LOCAL pair or BOUNDARY)

  // M->C
  MFTL_HalfEdge *he_mc = &obj->half_edges[he_mc_idx];
  he_mc->vertex_idx = v_c_idx;
  he_mc->triangle_idx = t1_idx;
  he_mc->next_idx = he_ca_idx;
  he_mc->twin_idx = he_cm_idx;
  he_mc->global_id = obj->next_halfedge_gid++;

  // C->A
  obj->half_edges[he_ca_idx].next_idx = he_ab_idx;

  // T2 = (M, B, C)
  obj->triangles[t2_idx].he_idx = he_mb_idx;
  obj->triangles[t2_idx].global_id = obj->next_triangle_gid++;
  obj->triangles[t2_idx].owner_proc = obj->domain->rank;
  obj->triangles[t2_idx].is_ghost = MFTL_FALSE;

  // M->B (twin set with the lower triangle)
  MFTL_HalfEdge *he_mb = &obj->half_edges[he_mb_idx];
  he_mb->vertex_idx = v_b_idx;
  he_mb->triangle_idx = t2_idx;
  he_mb->next_idx = he_bc_idx;
  he_mb->twin_idx = MFTL_ERROR;
  he_mb->global_id = obj->next_halfedge_gid++;

  // B->C
  obj->half_edges[he_bc_idx].triangle_idx = t2_idx;
  obj->half_edges[he_bc_idx].next_idx = he_cm_idx;

  // C->M
  MFTL_HalfEdge *he_cm = &obj->half_edges[he_cm_idx];
  he_cm->vertex_idx = v_m_idx;
  he_cm->triangle_idx = t2_idx;
  he_cm->next_idx = he_mb_idx;
  he_cm->twin_idx = he_mc_idx;
  he_cm->global_id = obj->next_halfedge_gid++;
  he_mc->twin_gid = he_cm->global_id;  // new inner pair: LOCAL
  he_cm->twin_gid = he_mc->global_id;

  // the halves are not split again in this pass
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_VISITED;
  obj->half_edges[he_mb_idx].flags |= MFTL_FLAG_VISITED;

  // the halves of a feature edge are feature edges
  if (obj->half_edges[he_ab_idx].flags & MFTL_FLAG_FEATURE)
  {
    obj->half_edges[he_mb_idx].flags |= MFTL_FLAG_FEATURE;
  }

  // Lower triangle T3(B,A,D) -> T3(B,M,D) + T4(M,A,D) (interior edge only)
  if (he_ba_idx != MFTL_ERROR)
  {
    int t3_idx = obj->half_edges[he_ba_idx].triangle_idx;
    int he_ad_idx = obj->half_edges[he_ba_idx].next_idx;
    int he_db_idx = obj->half_edges[he_ad_idx].next_idx;
    int v_d_idx = obj->half_edges[he_ad_idx].vertex_idx;

    int t4_idx = mftlNewTriangle(obj);
    int he_ma_idx = mftlNewHalfEdge(obj);
    int he_md_idx = mftlNewHalfEdge(obj);
    int he_dm_idx = mftlNewHalfEdge(obj);

    // T3 = (B, M, D)
    obj->triangles[t3_idx].he_idx = he_ba_idx;  // B->M

    // B->M re-uses he_ba
    obj->half_edges[he_ba_idx].vertex_idx = v_m_idx;
    obj->half_edges[he_ba_idx].next_idx = he_md_idx;
    obj->half_edges[he_ba_idx].twin_idx = he_mb_idx;
    obj->half_edges[he_mb_idx].twin_idx = he_ba_idx;

    // M->D
    MFTL_HalfEdge *he_md = &obj->half_edges[he_md_idx];
    he_md->vertex_idx = v_d_idx;
    he_md->triangle_idx = t3_idx;
    he_md->next_idx = he_db_idx;
    he_md->twin_idx = he_dm_idx;
    he_md->global_id = obj->next_halfedge_gid++;

    // D->B
    obj->half_edges[he_db_idx].next_idx = he_ba_idx;

    // T4 = (M, A, D)
    obj->triangles[t4_idx].he_idx = he_ma_idx;
    obj->triangles[t4_idx].global_id = obj->next_triangle_gid++;
    obj->triangles[t4_idx].owner_proc = obj->domain->rank;
    obj->triangles[t4_idx].is_ghost = MFTL_FALSE;

    // M->A
    MFTL_HalfEdge *he_ma = &obj->half_edges[he_ma_idx];
    he_ma->vertex_idx = v_a_idx;
    he_ma->triangle_idx = t4_idx;
    he_ma->next_idx = he_ad_idx;
    he_ma->twin_idx = he_ab_idx;
    obj->half_edges[he_ab_idx].twin_idx = he_ma_idx;
    he_ma->global_id = obj->next_halfedge_gid++;

    // A->D
    obj->half_edges[he_ad_idx].triangle_idx = t4_idx;
    obj->half_edges[he_ad_idx].next_idx = he_dm_idx;

    // D->M
    MFTL_HalfEdge *he_dm = &obj->half_edges[he_dm_idx];
    he_dm->vertex_idx = v_m_idx;
    he_dm->triangle_idx = t4_idx;
    he_dm->next_idx = he_ma_idx;
    he_dm->twin_idx = he_md_idx;
    he_dm->global_id = obj->next_halfedge_gid++;
    obj->half_edges[he_md_idx].twin_gid = he_dm->global_id;  // new inner pair: LOCAL
    he_dm->twin_gid = obj->half_edges[he_md_idx].global_id;
    // the split halves of the interior edge: A->M <-> M->A, M->B <-> B->M
    obj->half_edges[he_ab_idx].twin_gid = obj->half_edges[he_ma_idx].global_id;
    obj->half_edges[he_ma_idx].twin_gid = obj->half_edges[he_ab_idx].global_id;
    obj->half_edges[he_mb_idx].twin_gid = obj->half_edges[he_ba_idx].global_id;
    obj->half_edges[he_ba_idx].twin_gid = obj->half_edges[he_mb_idx].global_id;

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

  // outgoing half-edges: live ones
  obj->vertices[v_m_idx].out_he_idx = he_mb_idx;
  obj->vertices[v_a_idx].out_he_idx = he_ab_idx;
  // a boundary vertex keeps its outgoing BOUNDARY half-edge (plans/16 §2.6)
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_b_idx].out_he_idx])) obj->vertices[v_b_idx].out_he_idx = he_bc_idx;

  obj->vertices[v_m_idx].flags |= MFTL_FLAG_VISITED;

  return v_m_idx;
}

// Removes a triangle's two wing half-edges edge1/edge2 from the topology by pairing their
// outer twins with each other (collapse). The merged edge's state follows from the outer
// half-edges (plans/16 §2.2) — both LOCAL: a LOCAL pair; one BOUNDARY: the LOCAL one becomes
// BOUNDARY with that tag. A REMOTE outer half-edge or two BOUNDARY ones are rejected by the
// collapse gates before any change.
void mftlGlueWings(MFTL_Object *o, int edge1, int edge2)
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

/* Collapse admissibility of the twin states (plans/16 §2.2): the edge is LOCAL or
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

// Collapses A->B into B: A's fan is re-pointed to B, the two triangles of the edge are
// removed (mftlGlueWings), B moves to the (anchored) midpoint. Dead entities are compacted by
// MFTLFinalizeRemeshing; the caller fits B afterwards (MFTLCollapseEdgeWithFit).
void MFTLCollapseEdge(MFTL_Object *obj, int he_ab_idx)
{
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_ab_idx];
  int he_ba_idx = he_ab->twin_idx;

  int prev_he = obj->half_edges[obj->half_edges[he_ab_idx].next_idx].next_idx;
  int v_a_idx = obj->half_edges[prev_he].vertex_idx;
  int v_b_idx = he_ab->vertex_idx;

  // refuse entities already removed earlier in this pass
  if ((obj->vertices[v_a_idx].flags & MFTL_FLAG_DEAD) || (obj->vertices[v_b_idx].flags & MFTL_FLAG_DEAD))
  {
    printf("Collapse: Vertex %lld or %lld is already dead. ABORTING.\n", (long long) v_a_idx, (long long) v_b_idx);
    return;
  }
  if (he_ab->flags & MFTL_FLAG_DEAD)
  {
    printf("Collapse: Edge %lld is already dead. ABORTING.\n", (long long) he_ab_idx);
    return;
  }

  // Re-point every half-edge incoming to A to B, walking A's fan; remember one surviving
  // outgoing half-edge as B's new out half-edge.

  int safe_survivor_he = MFTL_ERROR;

  MFTL_HalfEdge *start_he = MFTLVOutHe(obj, &obj->vertices[v_a_idx]);
  MFTL_HalfEdge *curr = start_he;

  // rewind to the start of an open fan
  while (true)
  {
    MFTL_HalfEdge *incoming = MFTLHePrev(obj, curr);
    MFTL_HalfEdge *cw_neighbor = MFTLHeTwin(obj, incoming);
    if (!cw_neighbor || cw_neighbor == start_he) break;
    curr = cw_neighbor;
  }
  start_he = curr;

  do
  {
    int he_out_idx = (int) (curr - obj->half_edges);

    if (he_out_idx != he_ab_idx)
    {
      safe_survivor_he = he_out_idx;
    }

    // the incoming half-edge of the same triangle (prev = next->next) targets A
    if (curr->next_idx != MFTL_ERROR)
    {
      int next_id = curr->next_idx;
      int next_next_id = obj->half_edges[next_id].next_idx;

      if (next_next_id != MFTL_ERROR)
      {
        int he_in_idx = next_next_id;
        // always true on a valid triangle mesh; otherwise the half-edge is left unchanged
        if (obj->half_edges[he_in_idx].vertex_idx == v_a_idx)
        {
          obj->half_edges[he_in_idx].vertex_idx = v_b_idx;
        }
        else
        {
        }
      }
    }

    MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
    if (!twin) break;  // end of an open fan
    curr = MFTLHeNext(obj, twin);

  } while (curr != start_he);

  // remove the triangle(s) of the edge; their outer wings are paired with each other
  int t1_idx = he_ab->triangle_idx;
  obj->triangles[t1_idx].flags |= MFTL_FLAG_DEAD;
  int he_bc = he_ab->next_idx;
  int he_ca = obj->half_edges[he_bc].next_idx;
  mftlGlueWings(obj, he_ca, he_bc);
  obj->half_edges[he_bc].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ca].flags |= MFTL_FLAG_DEAD;

  if (he_ba_idx != MFTL_ERROR)
  {
    int t2_idx = obj->half_edges[he_ba_idx].triangle_idx;
    obj->triangles[t2_idx].flags |= MFTL_FLAG_DEAD;
    int he_ad = obj->half_edges[he_ba_idx].next_idx;
    int he_db = obj->half_edges[he_ad].next_idx;
    mftlGlueWings(obj, he_ad, he_db);
    obj->half_edges[he_ad].flags |= MFTL_FLAG_DEAD;
    obj->half_edges[he_db].flags |= MFTL_FLAG_DEAD;
  }

  obj->vertices[v_a_idx].flags |= MFTL_FLAG_DEAD;
  obj->half_edges[he_ab_idx].flags |= MFTL_FLAG_DEAD;
  if (he_ba_idx != MFTL_ERROR) obj->half_edges[he_ba_idx].flags |= MFTL_FLAG_DEAD;

  // B moves to the midpoint (anchored at B, seam-safe; raw average when non-periodic)
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    double e_ba[3];
    MFTLGetMinimumImageVector(obj->domain, e_ba, obj->vertices[v_b_idx].pos, obj->vertices[v_a_idx].pos);
    for (int k = 0; k < 3; ++k)
    {
      obj->vertices[v_b_idx].pos[k] = obj->vertices[v_b_idx].pos[k] + 0.5 * e_ba[k];
    }
    MFTLWrapPosImg(obj->domain, obj->vertices[v_b_idx].pos, obj->vertices[v_b_idx].img);  // anchor b
  }
  else
  {
    for (int k = 0; k < 3; ++k)
    {
      obj->vertices[v_b_idx].pos[k] = 0.5 * (obj->vertices[v_b_idx].pos[k] + obj->vertices[v_a_idx].pos[k]);
    }
  }

  // B needs a live outgoing half-edge: the survivor of A's fan (now B->X), else the twin of the
  // removed wing
  int check_he = obj->vertices[v_b_idx].out_he_idx;

  if (check_he == MFTL_ERROR || (obj->half_edges[check_he].flags & MFTL_FLAG_DEAD))
  {
    if (safe_survivor_he != MFTL_ERROR && !(obj->half_edges[safe_survivor_he].flags & MFTL_FLAG_DEAD))
    {
      obj->vertices[v_b_idx].out_he_idx = safe_survivor_he;
    }
    else if (obj->half_edges[he_ca].twin_idx != MFTL_ERROR)
    {
      obj->vertices[v_b_idx].out_he_idx = obj->half_edges[he_ca].twin_idx;
    }
  }

  obj->vertices[v_b_idx].flags |= MFTL_FLAG_VISITED;  // no further operation on B in this pass
}

/* The one-ring VERTICES of v including, for a boundary vertex (OPEN_COMPLETE fan),
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

// 1 unless the vertex has a closed stored fan (boundary vertex, or a fan cut by an MPI rim)
static bool mftlIsBoundaryVertex(const MFTL_Object *obj, int v_idx)
{
  // An incomplete fan (REMOTE rim) is treated like a boundary; a boundary vertex has an
  // OPEN_COMPLETE fan.
  return MFTLVertexRing(obj, v_idx, MFTL_NULL, 0, MFTL_NULL) != MFTL_RING_CLOSED;
}

/**
 * @brief Number of feature edges at a vertex; an edge without a stored twin (physical boundary
 * or MPI rim) counts as a feature. Walks the fan once (rewind + forward sweep) and adds the
 * incoming edge at the start of an open fan.
 */
int mftlCountFeatureEdges(MFTL_Object *obj, int v_idx)
{
  MFTL_Vertex *v = &obj->vertices[v_idx];

  if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) return 0;

  int feat_count = 0;
  int curr_he_idx = v->out_he_idx;
  int start_he_idx = curr_he_idx;

  // rewind (twin(prev), counter-clockwise seen from the normal side) to the first outgoing
  // half-edge of an open fan
  while (1)
  {
    int next_idx = obj->half_edges[curr_he_idx].next_idx;
    int prev_idx = obj->half_edges[next_idx].next_idx;

    int twin_idx = obj->half_edges[prev_idx].twin_idx;

    if (twin_idx == MFTL_ERROR || twin_idx == start_he_idx) break;

    curr_he_idx = twin_idx;
  }

  start_he_idx = curr_he_idx;

  // forward sweep over the outgoing half-edges
  while (1)
  {
    MFTL_HalfEdge *he = &obj->half_edges[curr_he_idx];

    if ((he->flags & MFTL_FLAG_FEATURE) || he->twin_idx == MFTL_ERROR)
    {
      feat_count++;
    }

    if (he->twin_idx == MFTL_ERROR)
    {
      break;  // end of an open fan
    }

    curr_he_idx = obj->half_edges[he->twin_idx].next_idx;

    if (curr_he_idx == start_he_idx) break;
  }

  // an open fan has one more edge: the incoming half-edge before the first outgoing one
  int next_idx = obj->half_edges[start_he_idx].next_idx;
  int prev_idx = obj->half_edges[next_idx].next_idx;
  MFTL_HalfEdge *incoming_start_he = &obj->half_edges[prev_idx];

  if (incoming_start_he->twin_idx == MFTL_ERROR)
  {
    feat_count++;
  }

  return feat_count;
}

// unit normal of a triangle from the current vertex positions
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
  if (he->twin_idx == MFTL_ERROR) return 1;  // no stored twin (boundary or rim): treat as feature
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

/* Collapse approval of the GENERALIZED remesher (plans/16 §4.4.1; open bodies and
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
  // interior feature lines (boundary edges excluded)
  const int fa = mftlInteriorFeatureCount(obj, v_a), fb = mftlInteriorFeatureCount(obj, v_b);
  if ((fa > 0) != (fb > 0)) return MFTL_FALSE;
  if (fa > 2 || fb > 2) return MFTL_FALSE;
  return MFTL_TRUE;
}

int CollapseApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params)
{
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // twin-state admissibility (no REMOTE wing, no two BOUNDARY wings)
  if (!mftlCollapseStatesOk(obj, he_idx)) return MFTL_FALSE;

  // open bodies and closed bodies opted in use the generalized remesher (never the parity path);
  // closed bodies in the default mode are unchanged
  if (params && MFTLObjectRemeshGeneralized(obj)) return mftlCollapseApprovedGeneralized(obj, he_idx, params);

  // MFTL1 criteria (with the parallel drivers' locks)
  if (params && params->mftl1_criteria) return MFTL1CollapseApproved(obj, he_idx, params, 1);

  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // geometric feature protection: a sharp edge is never collapsed, regardless of feature flags
  if (mftlEdgeIsGeometricallySharp(obj, he_idx, params->feature_angle_threshold)) return MFTL_FALSE;

  // no chained operations: both vertices must be untouched in this pass
  int prev_he = obj->half_edges[obj->half_edges[he_idx].next_idx].next_idx;
  int v_a_idx = obj->half_edges[prev_he].vertex_idx;
  int v_b_idx = he->vertex_idx;

  if (obj->vertices[v_a_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  bool is_boundary_edge = MFTLHeIsBoundary(he);

  // criterion: the edge must be short (small) enough to collapse
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

  // Link condition: A and B share exactly 2 neighbours (interior edge) or 1 (boundary edge).

  MFTL_Vertex *v_a_ptr = &obj->vertices[v_a_idx];
  MFTL_Vertex *v_b_ptr = &obj->vertices[v_b_idx];

  MFTL_Vertex *ring_a[64];
  MFTL_Vertex *ring_b[64];
  // full neighbour sets (boundary vertices included), so the link test is exact
  int count_a = mftlRingVerticesFull(obj, v_a_ptr, ring_a, 64);
  int count_b = mftlRingVerticesFull(obj, v_b_ptr, ring_b, 64);
  if (count_a < 0 || count_b < 0) return MFTL_FALSE;  // ring larger than the buffer (16C review)

  int common = 0;
  for (int i = 0; i < count_a; ++i)
  {
    for (int j = 0; j < count_b; ++j)
    {
      if (ring_a[i] == ring_b[j]) common++;
    }
  }

  if (is_boundary_edge)
  {
    if (common != 1) return MFTL_FALSE;
  }
  else
  {
    if (common != 2) return MFTL_FALSE;
  }

  // a boundary vertex is never collapsed into an interior vertex
  bool a_bound = mftlIsBoundaryVertex(obj, v_a_idx);
  bool b_bound = mftlIsBoundaryVertex(obj, v_b_idx);
  if (a_bound && !b_bound) return MFTL_FALSE;
  // an interior edge joining two boundary vertices would pinch the surface into a
  // non-manifold vertex (link condition with the virtual boundary vertex, plans/16 §4.4.1)
  if (a_bound && b_bound && !is_boundary_edge) return MFTL_FALSE;

  // Stretch check (prevents split/collapse ping-pong): no edge from the midpoint M to a
  // neighbour may exceed the maximum edge length. M is seam-safe (raw average when non-periodic).
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

  for (int i = 0; i < count_a; ++i)
  {
    if (ring_a[i] == v_b_ptr) continue;

    double dvec[3];
    MFTLGetMinimumImageVector(obj->domain, dvec, pos_m, ring_a[i]->pos);
    double new_len_sq = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];

    if (new_len_sq > params->max_edge_length_sq) return MFTL_FALSE;
  }

  for (int i = 0; i < count_b; ++i)
  {
    if (ring_b[i] == v_a_ptr) continue;

    double dvec[3];
    MFTLGetMinimumImageVector(obj->domain, dvec, pos_m, ring_b[i]->pos);
    double new_len_sq = dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2];

    if (new_len_sq > params->max_edge_length_sq) return MFTL_FALSE;
  }

  // feature lines: both endpoints on a feature line or neither (a collapse would pull the line
  // off its trajectory), and no corner (> 2 feature edges) is collapsed
  int a_feat_count = mftlCountFeatureEdges(obj, v_a_idx);
  int b_feat_count = mftlCountFeatureEdges(obj, v_b_idx);

  if (a_feat_count > 0 && b_feat_count == 0) return MFTL_FALSE;
  if (b_feat_count > 0 && a_feat_count == 0) return MFTL_FALSE;

  if (a_feat_count > 2 || b_feat_count > 2) return MFTL_FALSE;

  return MFTL_TRUE;
}

int MFTLSplitApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params)
{
  int approved = MFTL_FALSE;
  MFTL_HalfEdge *he = &obj->half_edges[he_idx];

  // edge and vertices untouched in this pass
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  int next = obj->half_edges[he_idx].next_idx;
  int prev = obj->half_edges[next].next_idx;
  int v_a = obj->half_edges[prev].vertex_idx;
  int v_b = he->vertex_idx;

  if (obj->vertices[v_a].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->vertices[v_b].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  // a REMOTE edge is never split (its second triangle is not stored)
  if (MFTLHeIsRemote(he) || MFTLHeIsUnresolved(he)) return MFTL_FALSE;

  // MFTL1 criteria (never for generalized bodies)
  const int generalized = MFTLObjectRemeshGeneralized(obj);
  if (params->mftl1_criteria && !generalized) return MFTL1SplitCriterion(obj, he_idx, params);
  // the opposite vertices gain a neighbour; never above the growth cap
  if (generalized && !MFTLGeneralizedSplitValenceOk(obj, he_idx)) return MFTL_FALSE;
  // a boundary split needs its boundary-curve point (a CONSTRAINED projector may refuse)
  if (generalized && MFTLHeIsBoundary(he))
  {
    double tmp[3];
    if (MFTLGeneralizedSplitTarget(obj, he_idx, MFTLBoundaryCornerAngle(params), tmp) != MFTL_SUCCESS) return MFTL_FALSE;
  }
  // curvature criteria are undefined at boundary vertices -> length criterion
  if (generalized && params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE &&
      (MFTLHeIsBoundary(he) || ((obj->vertices[v_a].flags | obj->vertices[v_b].flags) & MFTL_VFLAG_CURV_UNDEFINED)))
    return MFTLGetEdgeLengthSquared(obj, he) > params->max_edge_length_sq;

  // criterion (MFTL1 RemeshCriteria parity)
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
      double len_sq = MFTLGetEdgeLengthSquared(obj, he);
      if (len_sq > params->max_edge_length_sq) approved = MFTL_TRUE;
      break;
    }
  }

  return approved;
}

void MFTLFlipEdge(MFTL_Object *obj, int he_idx)
{
  // interior LOCAL edge (checked by the approval gate)
  int tri1_idx = obj->half_edges[he_idx].triangle_idx;
  int tri2_idx = obj->half_edges[obj->half_edges[he_idx].twin_idx].triangle_idx;

  // Conserved surface scalar (MFTL1 edgeflip.c parity): mass and area of the two triangles before the
  // flip; afterwards the mass is split in proportion new_area / (sum of the OLD areas),
  // the second triangle taking the remainder.
  MFTLGetTriangleProperties(obj, &obj->triangles[tri1_idx], &obj->triangles[tri1_idx].area, MFTL_NULL, MFTL_NULL);
  MFTLGetTriangleProperties(obj, &obj->triangles[tri2_idx], &obj->triangles[tri2_idx].area, MFTL_NULL, MFTL_NULL);
  const double ss_mass_sum =
      obj->triangles[tri1_idx].surface_scalar * obj->triangles[tri1_idx].area + obj->triangles[tri2_idx].surface_scalar * obj->triangles[tri2_idx].area;
  const double ss_area_sum = obj->triangles[tri1_idx].area + obj->triangles[tri2_idx].area;

  int he_ab_idx = he_idx;
  int he_bc_idx = obj->half_edges[he_ab_idx].next_idx;
  int he_ca_idx = obj->half_edges[he_bc_idx].next_idx;
  int he_ba_idx = obj->half_edges[he_ab_idx].twin_idx;
  int he_ad_idx = obj->half_edges[he_ba_idx].next_idx;
  int he_db_idx = obj->half_edges[he_ad_idx].next_idx;

  int v_a = obj->half_edges[he_ba_idx].vertex_idx;
  int v_b = obj->half_edges[he_ab_idx].vertex_idx;
  int v_c = obj->half_edges[he_bc_idx].vertex_idx;
  int v_d = obj->half_edges[he_ad_idx].vertex_idx;

  // A and B get outgoing half-edges that survive the flip (a boundary vertex keeps its outgoing
  // BOUNDARY half-edge, which the flip does not touch)
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_a].out_he_idx])) obj->vertices[v_a].out_he_idx = he_ad_idx;
  if (!MFTLHeIsBoundary(&obj->half_edges[obj->vertices[v_b].out_he_idx])) obj->vertices[v_b].out_he_idx = he_bc_idx;

  // T1 = (A, C, D) along C->A->D, T2 = (B, D, C) along D->B->C; the edge becomes C<->D
  obj->triangles[tri1_idx].he_idx = he_ab_idx;
  obj->triangles[tri2_idx].he_idx = he_ba_idx;

  obj->half_edges[he_ab_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ca_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ad_idx].triangle_idx = tri1_idx;
  obj->half_edges[he_ba_idx].triangle_idx = tri2_idx;
  obj->half_edges[he_db_idx].triangle_idx = tri2_idx;
  obj->half_edges[he_bc_idx].triangle_idx = tri2_idx;

  obj->half_edges[he_ab_idx].next_idx = he_ca_idx;
  obj->half_edges[he_ca_idx].next_idx = he_ad_idx;
  obj->half_edges[he_ad_idx].next_idx = he_ab_idx;
  obj->half_edges[he_ba_idx].next_idx = he_db_idx;
  obj->half_edges[he_db_idx].next_idx = he_bc_idx;
  obj->half_edges[he_bc_idx].next_idx = he_ba_idx;

  obj->half_edges[he_ab_idx].vertex_idx = v_c;
  obj->half_edges[he_ba_idx].vertex_idx = v_d;

  // remap the conserved surface scalar (see above)
  MFTL_Triangle *t1 = &obj->triangles[tri1_idx], *t2 = &obj->triangles[tri2_idx];
  MFTLGetTriangleProperties(obj, t1, &t1->area, MFTL_NULL, MFTL_NULL);
  MFTLGetTriangleProperties(obj, t2, &t2->area, MFTL_NULL, MFTL_NULL);
  const double ss_mass1 = (ss_area_sum > 0.0) ? ss_mass_sum * (t1->area / ss_area_sum) : 0.0;
  const double ss_mass2 = ss_mass_sum - ss_mass1;
  t1->surface_scalar = (t1->area > 0.0) ? ss_mass1 / t1->area : 0.0;
  t2->surface_scalar = (t2->area > 0.0) ? ss_mass2 / t2->area : 0.0;
}

int FlipApprovedWithParams(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params);

// Valence gate of a flip: interior LOCAL edge, no degenerate or duplicate edge, and the summed
// squared deviation from the target valences (6 interior, 4 boundary; generalized bodies:
// plans/16 §4.4.2) strictly decreases.

int mftlFlipApproved(MFTL_Object *obj, int he_idx)
{
  MFTL_HalfEdge *he_ab = &obj->half_edges[he_idx];
  int he_ba_idx = he_ab->twin_idx;
  if (!MFTLHeIsLocal(he_ab)) return MFTL_FALSE;  // only interior LOCAL edges flip

  MFTL_HalfEdge *he_ba = &obj->half_edges[he_ba_idx];

  // A->B with opposite vertices C (upper) and D (lower)
  int v_b = he_ab->vertex_idx;
  int v_a = he_ba->vertex_idx;

  int he_bc_idx = he_ab->next_idx;
  int he_ad_idx = he_ba->next_idx;

  int v_c = obj->half_edges[he_bc_idx].vertex_idx;
  int v_d = obj->half_edges[he_ad_idx].vertex_idx;

  // Current valences. The ring buffer holds MFTL_FLIP_RING_MAX entries (32 overflowed on
  // generalized bodies); a ring that does not fit rejects the flip. Identical to the former
  // 32-entry buffer for every valence below 32.
  MFTL_Vertex *ring_buffer[MFTL_FLIP_RING_MAX];
  int val_a = mftlRingVerticesFull(obj, &obj->vertices[v_a], ring_buffer, MFTL_FLIP_RING_MAX);
  int val_b = mftlRingVerticesFull(obj, &obj->vertices[v_b], ring_buffer, MFTL_FLIP_RING_MAX);

  // the flip lowers the valence of A and B by 1; both must keep at least 3
  if (val_a <= 3 || val_b <= 3) return MFTL_FALSE;

  // full neighbour set of C (a boundary vertex's neighbour across its incoming boundary edge
  // included) for the duplicate-edge test below
  int val_c = mftlRingVerticesFull(obj, &obj->vertices[v_c], ring_buffer, MFTL_FLIP_RING_MAX);
  if (val_c < 0) return MFTL_FALSE;

  // C and D already connected: the flip would duplicate the edge C-D (non-manifold)
  for (int i = 0; i < val_c; ++i)
  {
    if (ring_buffer[i] == &obj->vertices[v_d]) return MFTL_FALSE;
  }

  int val_d = mftlRingVerticesFull(obj, &obj->vertices[v_d], ring_buffer, MFTL_FLIP_RING_MAX);
  if (val_d < 0) return MFTL_FALSE;
  // a generalized flip never raises a valence above the split/flip cap (see
  // MFTL_GENERALIZED_MAX_GROW_VALENCE)
  if (MFTLObjectRemeshGeneralized(obj) && (val_c + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE || val_d + 1 > MFTL_GENERALIZED_MAX_GROW_VALENCE)) return MFTL_FALSE;

  // target valences: 6 interior, 4 boundary (or rim-cut fan)
  int opt_a = mftlIsBoundaryVertex(obj, v_a) ? 4 : 6;
  int opt_b = mftlIsBoundaryVertex(obj, v_b) ? 4 : 6;
  int opt_c = mftlIsBoundaryVertex(obj, v_c) ? 4 : 6;
  int opt_d = mftlIsBoundaryVertex(obj, v_d) ? 4 : 6;
  if (MFTLObjectRemeshGeneralized(obj))
  {
    // generalized bodies (plans/16 §4.4.2): boundary valence target round(theta / 60 deg) + 1
    // (min 2), theta the interior angle; 6 in the interior
    opt_a = MFTLBoundaryValenceTarget(obj, v_a);
    opt_b = MFTLBoundaryValenceTarget(obj, v_b);
    opt_c = MFTLBoundaryValenceTarget(obj, v_c);
    opt_d = MFTLBoundaryValenceTarget(obj, v_d);
  }

  // summed squared valence deviation before and after the flip
  int err_current =
      (val_a - opt_a) * (val_a - opt_a) + (val_b - opt_b) * (val_b - opt_b) + (val_c - opt_c) * (val_c - opt_c) + (val_d - opt_d) * (val_d - opt_d);

  int val_a_new = val_a - 1;
  int val_b_new = val_b - 1;
  int val_c_new = val_c + 1;
  int val_d_new = val_d + 1;

  int err_flipped = (val_a_new - opt_a) * (val_a_new - opt_a) + (val_b_new - opt_b) * (val_b_new - opt_b) + (val_c_new - opt_c) * (val_c_new - opt_c) +
                    (val_d_new - opt_d) * (val_d_new - opt_d);

  if (err_flipped < err_current)
  {
    return MFTL_TRUE;
  }

  return MFTL_FALSE;
}

// Re-derives the twin gid of every half-edge whose twin is stored. The operators set twin_gid
// explicitly (split / mftlGlueWings); this only refreshes LOCAL pairs after compaction and never
// touches REMOTE / BOUNDARY codes.
static void mftlRefreshTwinGids(MFTL_Object *obj)
{
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &obj->half_edges[i];
    if (he->twin_idx != MFTL_ERROR) he->twin_gid = obj->half_edges[he->twin_idx].global_id;
  }
}

/**
 * @brief Ends a remesh pass. Without collapses only the appended entities (index >=
 * start_num_*) are entered into the id maps; after collapses the arrays are compacted (dead
 * entities removed, payloads of dead triangles freed), all indices relinked and the maps
 * rebuilt. Twin gids and the boundary cache are refreshed in both cases.
 */
void MFTLFinalizeRemeshing(MFTL_Object *obj, int start_num_v, int start_num_t, int start_num_h, bool collapses_occurred)
{
  // Path A: splits only — no compaction needed
  if (!collapses_occurred)
  {
    if (obj->n_vertices > start_num_v)
    {
      kh_resize(vertex_map, obj->vertex_lookup, obj->n_vertices);

      for (int i = start_num_v; i < obj->n_vertices; ++i)
      {
        int ret;
        khiter_t iter = kh_put(vertex_map, obj->vertex_lookup, obj->vertices[i].global_id, &ret);
        kh_value(obj->vertex_lookup, iter) = i;
      }
    }

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
    return;
  }

  // Path B: collapses happened — compact, relink, rebuild the maps

  // no live half-edge may point to a dead vertex
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

  // old index -> new index
  int *v_map = malloc(obj->n_vertices * sizeof(int));
  int *t_map = malloc(obj->n_triangles * sizeof(int));
  int *h_map = malloc(obj->n_half_edges * sizeof(int));

  assert(v_map && t_map && h_map);

  // compact the arrays (live entities keep their relative order)
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


  int new_t_count = 0;
  for (int i = 0; i < obj->n_triangles; i++)
  {
    if (obj->triangles[i].flags & MFTL_FLAG_DEAD)
    {
      // Free the dead triangle's user payload. Live triangles move
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

  // relink all indices through the maps
  for (int i = 0; i < obj->n_vertices; i++)
  {
    assert(obj->vertices[i].out_he_idx != MFTL_ERROR);
    obj->vertices[i].out_he_idx = h_map[obj->vertices[i].out_he_idx];
  }
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


  free(v_map);
  free(t_map);
  free(h_map);

  // rebuild the id maps from scratch (indices shifted)

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
