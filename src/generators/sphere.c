/*
 * MFTL2 generators — the icosphere (MFTLGenerateSphere).
 *
 * Collective: rank 0 builds an icosahedron, refines it by 1-to-4 subdivision, canonicalises it
 * to OUTWARD winding (V > 0; the icosahedron table itself is inward-wound), projects onto the
 * sphere and lifts the continuous positions to (pos, img); every rank then receives the gid
 * counters, flattens and canonicalises the image counters. The caller distributes the object
 * (MFTLRedistributeListOfObjects). A mesh violating the periodic minimum-image constraint is
 * refused (MFTL_ERROR on every rank).
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "khash.h"
#include <stdint.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_vectoroperations.h"
#include "mftl_geometry.h"
#include "mftl_images.h"
#include "mftl_topology.h"

/**
 * @brief One level of 1-to-4 subdivision of a closed mesh (rank 0, before scaling): one
 * midpoint per edge (shared by both half-edges), four sub-triangles per triangle with the
 * parent's winding, then MFTLRebuildTwins.
 */
static int mftlTessellateSphere(MFTL_Object *object)
{
  MFTL_Domain *domain = object->domain;

  const size_t old_nv = object->n_vertices;
  const size_t old_nt = object->n_triangles;
  const size_t old_nhe = object->n_half_edges;

  const MFTL_Vertex *old_vertices = object->vertices;
  const MFTL_Triangle *old_triangles = object->triangles;
  const MFTL_HalfEdge *old_half_edges = object->half_edges;

  const size_t old_nedges = old_nhe / 2;  // closed mesh

  const size_t new_nv = old_nv + old_nedges;  // old vertices + one midpoint per edge
  const size_t new_nt = old_nt * 4;
  const size_t new_nhe = new_nt * 3;

  MFTL_Vertex *new_vertices = calloc(new_nv, sizeof(MFTL_Vertex));
  MFTL_Triangle *new_triangles = calloc(new_nt, sizeof(MFTL_Triangle));
  MFTL_HalfEdge *new_half_edges = calloc(new_nhe, sizeof(MFTL_HalfEdge));

  if (!new_vertices || !new_triangles || !new_half_edges)
  {
    fprintf(stderr, "[Error] Allocation failed in tessellate_sphere\n");
    return MFTL_ERROR;
  }

  // old vertices keep their indices; their outgoing half-edges are re-linked below
  memcpy(new_vertices, old_vertices, old_nv * sizeof(MFTL_Vertex));
  for (size_t i = 0; i < old_nv; ++i) new_vertices[i].out_he_idx = MFTL_ERROR;

  // old half-edge index -> midpoint vertex index (both half-edges of an edge)
  int *midpoint_map = malloc(old_nhe * sizeof(int));
  for (size_t i = 0; i < old_nhe; ++i) midpoint_map[i] = MFTL_ERROR;

  int next_midpoint_idx = (int) old_nv;

  for (size_t i = 0; i < old_nhe; ++i)
  {
    if (midpoint_map[i] != MFTL_ERROR) continue;  // edge already split via its twin

    const MFTL_HalfEdge *he = &old_half_edges[i];
    const MFTL_Vertex *v1 = &old_vertices[he->vertex_idx];  // target
    int twin_idx = he->twin_idx;
    const MFTL_Vertex *v0 = &old_vertices[old_half_edges[twin_idx].vertex_idx];  // origin

    MFTL_Vertex *mid = &new_vertices[next_midpoint_idx];

    // unit-sphere coordinates (projection and scaling follow in MFTLGenerateSphere)
    mid->pos[0] = 0.5 * (v0->pos[0] + v1->pos[0]);
    mid->pos[1] = 0.5 * (v0->pos[1] + v1->pos[1]);
    mid->pos[2] = 0.5 * (v0->pos[2] + v1->pos[2]);

    mid->global_id = object->next_vertex_gid++;
    mid->owner_proc = domain->rank;
    mid->is_ghost = MFTL_FALSE;
    mid->out_he_idx = MFTL_ERROR;

    midpoint_map[i] = next_midpoint_idx;
    midpoint_map[twin_idx] = next_midpoint_idx;

    next_midpoint_idx++;
  }

  // Sub-triangles of (v0, v1, v2), same winding as the parent:
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

    int he0_idx = old_tri->he_idx;
    int he1_idx = old_half_edges[he0_idx].next_idx;
    int he2_idx = old_half_edges[he1_idx].next_idx;

    // v0 = origin of he0 (= target of he2)
    int v0_idx = old_half_edges[he2_idx].vertex_idx;
    int v1_idx = old_half_edges[he0_idx].vertex_idx;
    int v2_idx = old_half_edges[he1_idx].vertex_idx;

    int m0_idx = midpoint_map[he0_idx];  // on edge v0-v1
    int m1_idx = midpoint_map[he1_idx];  // on edge v1-v2
    int m2_idx = midpoint_map[he2_idx];  // on edge v2-v0

    int sub_tris[4][3] = {
        {v0_idx, m0_idx, m2_idx},  // Corner 0
        {v1_idx, m1_idx, m0_idx},  // Corner 1
        {v2_idx, m2_idx, m1_idx},  // Corner 2
        {m0_idx, m1_idx, m2_idx}  // Center
    };

    for (int j = 0; j < 4; ++j)
    {
      MFTL_Triangle *tri = &new_triangles[new_tri_idx];
      tri->global_id = object->next_triangle_gid++;
      tri->owner_proc = domain->rank;
      tri->is_ghost = MFTL_FALSE;
      tri->he_idx = new_he_idx;

      for (int k = 0; k < 3; ++k)
      {
        MFTL_HalfEdge *he = &new_half_edges[new_he_idx + k];
        int current_v_idx = sub_tris[j][k];
        int next_v_idx = sub_tris[j][(k + 1) % 3];

        he->global_id = object->next_halfedge_gid++;

        he->triangle_idx = new_tri_idx;
        he->vertex_idx = next_v_idx;  // target vertex
        he->next_idx = new_he_idx + ((k + 1) % 3);

        he->twin_idx = MFTL_ERROR;
        he->twin_gid = MFTL_TWIN_UNKNOWN;  // closed: every half-edge is paired by MFTLRebuildTwins
        he->boundary = MFTL_FALSE;

        new_vertices[current_v_idx].out_he_idx = new_he_idx + k;
      }

      new_tri_idx++;
      new_he_idx += 3;
    }
  }

  free(object->vertices);
  // free per-triangle payloads before replacing the triangle array wholesale (defensive;
  // generation normally precedes MFTLInitializeObjectData)
  if (object->data_size > 0)
    for (size_t i = 0; i < old_nt; ++i) free(object->triangles[i].data);
  free(object->triangles);
  free(object->half_edges);
  free(midpoint_map);

  object->vertices = new_vertices;
  object->triangles = new_triangles;
  object->half_edges = new_half_edges;

  object->n_vertices = new_nv;
  object->n_triangles = new_nt;
  object->n_half_edges = new_nhe;

  object->capacity_vertices = new_nv;
  object->capacity_triangles = new_nt;
  object->capacity_half_edges = new_nhe;

  MFTLRebuildTwins(object);

  return MFTL_SUCCESS;
}

/**
 * @brief Fills an empty object with the unit icosahedron (12 vertices, 20 triangles, 60
 * half-edges; gids = indices) and pairs the twins (MFTLRebuildTwins). The table is
 * inward-wound; MFTLGenerateSphere canonicalises the winding afterwards.
 */
static int mftlGenerateIcosahedron(MFTL_Object *object)
{
  MFTL_Domain *domain = object->domain;

  const double X = 0.525731112119133606;
  const double Z = 0.850650808352039932;

  const double vertices_data[12][3] = {{-X, 0.0, Z}, {X, 0.0, Z},   {-X, 0.0, -Z}, {X, 0.0, -Z}, {0.0, Z, X},  {0.0, Z, -X},
                                       {0.0, -Z, X}, {0.0, -Z, -X}, {Z, X, 0.0},   {-Z, X, 0.0}, {Z, -X, 0.0}, {-Z, -X, 0.0}};

  const size_t triangles_data[20][3] = {{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1}, {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3}, {2, 7, 3},
                                        {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6}, {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5}, {7, 2, 11}};

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

  for (int i = 0; i < 12; ++i)
  {
    MFTL_Vertex *v = &object->vertices[i];
    v->global_id = i;
    v->owner_proc = domain->rank;
    v->is_ghost = MFTL_FALSE;
    memcpy(v->pos, vertices_data[i], 3 * sizeof(double));
    v->img[0] = v->img[1] = v->img[2] = 0;  // lifted after scaling
    v->out_he_idx = -1;
  }

  for (int i = 0; i < 20; ++i)
  {
    MFTL_Triangle *tri = &object->triangles[i];
    tri->global_id = i;
    tri->owner_proc = domain->rank;
    tri->is_ghost = MFTL_FALSE;

    size_t v_idx[3] = {triangles_data[i][0], triangles_data[i][1], triangles_data[i][2]};

    MFTL_HalfEdge *he[3];
    for (int j = 0; j < 3; ++j)
    {
      he[j] = &object->half_edges[i * 3 + j];
      he[j]->global_id = i * 3 + j;
      he[j]->triangle_idx = i;
      he[j]->vertex_idx = v_idx[(j + 1) % 3];  // target vertex
      he[j]->twin_idx = -1;  // paired by MFTLRebuildTwins
      he[j]->twin_gid = MFTL_ERROR;
      he[j]->next_idx = -1;

      MFTL_Vertex *start_v = &object->vertices[v_idx[j]];
      start_v->out_he_idx = i * 3 + j;
    }

    he[0]->next_idx = i * 3 + 1;
    he[1]->next_idx = i * 3 + 2;
    he[2]->next_idx = i * 3;

    tri->he_idx = i * 3;
  }

  MFTLRebuildTwins(object);

  return MFTL_SUCCESS;
}

/**
 * @brief Reverse the orientation of every triangle (flip the whole surface).
 *
 * Swaps the vertex targets of the first two half-edges of each triangle,
 * turning the cycle v0->v1->v2->v0 into v0->v2->v1->v0, then repairs the
 * vertex outgoing-edge assignments and the twin links.
 */
static void mftlReverseTriangleWinding(MFTL_Object *object)
{
  for (int t = 0; t < object->n_triangles; ++t)
  {
    MFTL_HalfEdge *he0 = &object->half_edges[object->triangles[t].he_idx];
    MFTL_HalfEdge *he1 = &object->half_edges[he0->next_idx];

    // Swap the targets of the first two half-edges: this reverses the
    // 3-cycle (verified by construction: afterwards he0 runs v0->v2,
    // he1 runs v2->v1, he2 runs v1->v0).
    int tmp = he0->vertex_idx;
    he0->vertex_idx = he1->vertex_idx;
    he1->vertex_idx = tmp;
  }

  // The reversal changed every half-edge's origin; out_he_idx must point
  // to a half-edge ORIGINATING at the vertex (origin = target of prev(he)).
  for (int i = 0; i < object->n_vertices; ++i) object->vertices[i].out_he_idx = MFTL_ERROR;
  for (int h = 0; h < object->n_half_edges; ++h)
  {
    MFTL_HalfEdge *he = &object->half_edges[h];
    int origin = object->half_edges[object->half_edges[he->next_idx].next_idx].vertex_idx;
    if (object->vertices[origin].out_he_idx == MFTL_ERROR) object->vertices[origin].out_he_idx = h;
  }

  MFTLRebuildTwins(object);
}

/**
 * @brief Canonicalize the surface to OUTWARD winding: compute the signed
 * volume (same 1/6 * v0.(v1 x v2) formula as MFTLObjectVolume) and flip
 * every triangle if it is negative — the exact approach of the GTS reader
 * (src/io/io.c). Idempotent by construction.
 */
static void mftlCanonicalizeOutwardWinding(MFTL_Object *object)
{
  double S = 0.0;
  for (int t = 0; t < object->n_triangles; ++t)
  {
    MFTL_HalfEdge *he0 = &object->half_edges[object->triangles[t].he_idx];
    MFTL_HalfEdge *he1 = &object->half_edges[he0->next_idx];
    MFTL_HalfEdge *he2 = &object->half_edges[he1->next_idx];
    const double *p0 = object->vertices[he2->vertex_idx].pos;
    const double *p1 = object->vertices[he0->vertex_idx].pos;
    const double *p2 = object->vertices[he1->vertex_idx].pos;
    S += p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) + p1[0] * (p2[1] * p0[2] - p2[2] * p0[1]) + p2[0] * (p0[1] * p1[2] - p0[2] * p1[1]);
  }

  if (S < 0.0) mftlReverseTriangleWinding(object);
}

// Id -> index maps of the generated sphere (the generator fills the arrays directly).
static void mftlSphereRebuildLookups(MFTL_Object *object)
{
  int ret;
  kh_clear(vertex_map, object->vertex_lookup);
  for (int i = 0; i < object->n_vertices; ++i)
  {
    khiter_t k = kh_put(vertex_map, object->vertex_lookup, object->vertices[i].global_id, &ret);
    kh_value(object->vertex_lookup, k) = i;
  }
  kh_clear(triangle_map, object->triangle_lookup);
  for (int i = 0; i < object->n_triangles; ++i)
  {
    khiter_t k = kh_put(triangle_map, object->triangle_lookup, object->triangles[i].global_id, &ret);
    kh_value(object->triangle_lookup, k) = i;
  }
  if (object->halfedge_lookup)
  {
    kh_clear(halfedge_map, object->halfedge_lookup);
    for (int i = 0; i < object->n_half_edges; ++i)
    {
      khiter_t k = kh_put(halfedge_map, object->halfedge_lookup, object->half_edges[i].global_id, &ret);
      kh_value(object->halfedge_lookup, k) = i;
    }
  }
}

int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements)
{
  assert(object);

  MFTL_Domain *domain = object->domain;

  if (domain->rank == 0)
  {
    mftlGenerateIcosahedron(object);
    for (int i = 0; i < refinements; ++i)
    {
      mftlTessellateSphere(object);
    }

    // OUTWARD winding (project invariant: outward normals, V > 0); the icosahedron table
    // winds the other way, and the signed-volume check makes this idempotent.
    mftlCanonicalizeOutwardWinding(object);

    for (size_t i = 0; i < object->n_vertices; ++i)
    {
      MFTLNormalizeVector(object->vertices[i].pos);
      object->vertices[i].pos[0] = center[0] + radius * object->vertices[i].pos[0];
      object->vertices[i].pos[1] = center[1] + radius * object->vertices[i].pos[1];
      object->vertices[i].pos[2] = center[2] + radius * object->vertices[i].pos[2];
      // continuous input -> (pos, img), img = floor((X - min) / L)
      double X[3] = {object->vertices[i].pos[0], object->vertices[i].pos[1], object->vertices[i].pos[2]};
      MFTLWrapContinuous(domain, X, object->vertices[i].pos, object->vertices[i].img);
    }
    mftlSphereRebuildLookups(object);
  }

  MPI_Bcast(&object->next_vertex_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&object->next_triangle_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&object->next_halfedge_gid, 1, MPI_INT, 0, domain->comm_world);

  MFTLObjectUpdateGlobalStats(object);

  MFTLFlattenConnectivity(object);
  MFTLCanonicaliseImages(object);  // cached wrap flags

  // refuse periodic meshes that violate the minimum-image constraint (collective; the object
  // is empty on non-root ranks, so all ranks agree)
  if (MFTLCheckMinimumImageConstraint(object) != MFTL_SUCCESS) return MFTL_ERROR;

  return MFTL_SUCCESS;
}
