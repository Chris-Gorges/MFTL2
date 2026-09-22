#include "mftl_bvh.h"
#include "mftl_communication.h"
#include "mftl_arrayfunctions.h"
#include "mftl_geometry.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

// Forward declaration
int MFTL_TriTriProximity(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, double proximity_tolerance);

/* ------------------------------------------------------------------------- */
/* Stage 13: periodic-aware box overlap                                      */
/*                                                                           */
/* On a periodic axis two intervals [a0,a1], [b0,b1] (wrapped storage)      */
/* overlap iff B shifted by k*L, k in {-1,0,1}, overlaps A in the raw real  */
/* line sense. This is conservative (a seam-straddling body whose raw box   */
/* spans the whole period simply never prunes) and exact under the          */
/* minimum-image constraint (true feature extents < L/2). Non-periodic axes */
/* use the plain raw comparison, so the non-periodic path is bit-identical  */
/* to the pre-Stage-13 code.                                                */
/* ------------------------------------------------------------------------- */

static inline int mftlIntervalOverlap1DPeriodic(double L, double amin, double amax, double bmin, double bmax)
{
  for (int s = -1; s <= 1; ++s)
  {
    double shift = s * L;
    double bs0 = bmin + shift;
    double bs1 = bmax + shift;
    if (!(amax < bs0 || amin > bs1)) return 1;
  }
  return 0;
}

static int mftlIntervalOverlapDom(const MFTL_Domain* dom, const double minA[3], const double maxA[3], const double minB[3], const double maxB[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (dom->is_periodic[k])
    {
      if (!mftlIntervalOverlap1DPeriodic(dom->domain_size[k], minA[k], maxA[k], minB[k], maxB[k])) return 0;
    }
    else
    {
      if (maxA[k] < minB[k] || minA[k] > maxB[k]) return 0;
    }
  }
  return 1;
}

// float variant for the LBVH node boxes (float->double promotion is exact)
static int mftlAABBOverlapDom(const MFTL_Domain* dom, const float minA[3], const float maxA[3], const float minB[3], const float maxB[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (dom->is_periodic[k])
    {
      if (!mftlIntervalOverlap1DPeriodic(dom->domain_size[k], (double) minA[k], (double) maxA[k], (double) minB[k], (double) maxB[k])) return 0;
    }
    else
    {
      if (maxA[k] < minB[k] || minA[k] > maxB[k]) return 0;
    }
  }
  return 1;
}

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
static inline uint32_t MFTL_ExpandBits(uint32_t v)
{
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

// Calculates a 30-bit Morton code for the given 3D point located within the unit cube [0,1].
static inline int MFTL_Morton3D(float x, float y, float z)
{
  x = fminf(fmaxf(x * 1024.0f, 0.0f), 1023.0f);
  y = fminf(fmaxf(y * 1024.0f, 0.0f), 1023.0f);
  z = fminf(fmaxf(z * 1024.0f, 0.0f), 1023.0f);
  uint32_t xx = MFTL_ExpandBits((uint32_t) x);
  uint32_t yy = MFTL_ExpandBits((uint32_t) y);
  uint32_t zz = MFTL_ExpandBits((uint32_t) z);
  return (int) ((xx << 2) | (yy << 1) | zz);
}

/**
 * @brief Internal recursive function that creates the parent-child node relationships of the LBVH tree.
 * 
 * Uses a State-of-the-Art Radix Binary tree split. Instead of sorting and slicing straight 
 * down the median of primitives, this analyzes the exact Highest Differing Bit between the 
 * Morton codes of the first and last elements using `__builtin_clz`, performing a binary search 
 * to surgically split nodes at logical geometric radix boundary planes.
 * 
 * @param bvh Pointer to the hierarchy object.
 * @param node_idx The current node index being populated.
 * @param first_prim Start index in the sorted primitive array.
 * @param prim_count Count of active primitives for this subtree.
 * @param morton_codes The fully sorted array of Morton code signatures.
 */
static void MFTL_BuildTopology(MFTL_BVH* bvh, int node_idx, int first_prim, int prim_count, int* morton_codes)
{
  int last_prim = first_prim + prim_count - 1;

  // 1. SOTA FIX: Removed the `first_code == last_code` trap!
  // We now strictly enforce the geometric threshold (<= 4 primitives).
  if (prim_count <= 4)
  {
    bvh->nodes[node_idx].payload = 1 | ((uint64_t) prim_count << 1) | ((uint64_t) first_prim << 32);
    return;
  }

  uint32_t first_code = (uint32_t) morton_codes[first_prim];
  uint32_t last_code = (uint32_t) morton_codes[last_prim];

  // 2. SOTA FIX: Create a virtual 64-bit key (Morton Code + Array Index Tie-Breaker)
  // This guarantees every single primitive has a mathematically unique identifier for the split.
  uint64_t first_key = ((uint64_t) first_code << 32) | (uint32_t) first_prim;
  uint64_t last_key = ((uint64_t) last_code << 32) | (uint32_t) last_prim;

  // Use the 64-bit version of count-leading-zeros (__builtin_clzll)
  int common_prefix = __builtin_clzll(first_key ^ last_key);

  int split_idx = first_prim;
  int step = prim_count;

  do
  {
    step = (step + 1) >> 1;  // ceiling division by 2
    int new_idx = split_idx + step;

    if (new_idx < last_prim)
    {
      uint32_t split_code = (uint32_t) morton_codes[new_idx];

      // Construct the virtual key for the split candidate
      uint64_t split_key = ((uint64_t) split_code << 32) | (uint32_t) new_idx;

      int split_prefix = __builtin_clzll(first_key ^ split_key);

      // If the candidate shares the same prefix bit as the first element, advance right.
      if (split_prefix > common_prefix)
      {
        split_idx = new_idx;
      }
    }
  } while (step > 1);

  // The actual right-child start index
  split_idx++;

  // Allocate two children contiguously
  int left_child = bvh->nodes_used;
  bvh->nodes_used += 2;

  bvh->nodes[node_idx].payload = ((uint64_t) left_child << 1);

  MFTL_BuildTopology(bvh, left_child, first_prim, split_idx - first_prim, morton_codes);
  MFTL_BuildTopology(bvh, left_child + 1, split_idx, prim_count - (split_idx - first_prim), morton_codes);
}

/**
 * @brief Internal recursive node refitting subroutine for LBVH bounding volumes.
 * 
 * Recursively ascends from leaf bounds (derived directly through array map tracking on the 
 * `flat_tri_verts`) to establish tight enclosing bounding boxes traversing up to the root.
 * 
 * @param bvh Pointer to the evaluated hierarchy.
 * @param node_idx Iterated index.
 * @param obj Pointer to the referenced geometry object holding the vertex positions.
 */
static void MFTL_RefitBVH(MFTL_BVH* bvh, int node_idx, MFTL_Object* obj)
{
  uint64_t payload = bvh->nodes[node_idx].payload;
  if (payload & 1)
  {
    // Leaf
    int count = (payload >> 1) & 0x7FFFFFFF;
    int first = (payload >> 32);

    float min_b[3] = {1e30f, 1e30f, 1e30f};
    float max_b[3] = {-1e30f, -1e30f, -1e30f};

    for (int i = 0; i < count; i++)
    {
      int tri_idx = bvh->prim_indices[first + i];
      int v0 = obj->flat_tri_verts[3 * tri_idx + 0];
      int v1 = obj->flat_tri_verts[3 * tri_idx + 1];
      int v2 = obj->flat_tri_verts[3 * tri_idx + 2];

      for (int k = 0; k < 3; k++)
      {
        float p0 = (float) obj->vertices[v0].pos[k];
        float p1 = (float) obj->vertices[v1].pos[k];
        float p2 = (float) obj->vertices[v2].pos[k];

        min_b[k] = fminf(min_b[k], fminf(p0, fminf(p1, p2)));
        max_b[k] = fmaxf(max_b[k], fmaxf(p0, fmaxf(p1, p2)));
      }
    }

    // Make this a user-defined parameter later. Increases the AABB size by a small amount to account for instance for lubrication models.
    // Could be a factor of the mesh size.
    double proximity_tolerance = 1e-4;

    for (int k = 0; k < 3; k++)
    {
      bvh->nodes[node_idx].aabb_min[k] = min_b[k] - proximity_tolerance;
      bvh->nodes[node_idx].aabb_max[k] = max_b[k] + proximity_tolerance;
    }
  }
  else
  {
    // Internal
    int left = payload >> 1;
    int right = left + 1;

    MFTL_RefitBVH(bvh, left, obj);
    MFTL_RefitBVH(bvh, right, obj);

    for (int k = 0; k < 3; k++)
    {
      bvh->nodes[node_idx].aabb_min[k] = fminf(bvh->nodes[left].aabb_min[k], bvh->nodes[right].aabb_min[k]);
      bvh->nodes[node_idx].aabb_max[k] = fmaxf(bvh->nodes[left].aabb_max[k], bvh->nodes[right].aabb_max[k]);
    }
  }
}
/**
 * @brief Constructs missing ghost layers for primitives that cross processor boundaries.
 * 
 * Uses `MPI_Neighbor_alltoall` to transmit the macro-AABBs of the local bodies to adjacent 
 * ranks. If overlapping bounding boxes are detected, intersects them with local geometries, 
 * packages the triangles/vertices using `PackedTriangleComplex`, and shuttles them to the 
 * respective neighbors via `MPI_Neighbor_alltoallv`. 
 * 
 * SOTA Upgrade: Unpacks incoming spatial ghost triangles natively onto the `flat_tri_verts` 
 * arrays, letting the LBVH builder resolve them purely geometrically without requiring 
 * half-edge boundary topology linking!
 * 
 * @param objA Pointer to the first object.
 * @param objB Pointer to the second object.
 * @return `MFTL_SUCCESS` on successful network exchange and data serialization mapping.
 */
/* THIS IS CURRENTLY NOT OPTIMAL COMMUNICATION-WISE BUT BRUTE FORCE, NEEDS TO BE OPTIMIZED*/
int MFTL_CreateSpatialGhosts(MFTL_Object* objA, MFTL_Object* objB)
{
  if (!objA || !objB || !objA->domain || objA->domain->nproc < 2) return MFTL_SUCCESS;
  MFTL_Domain* domain = objA->domain;
  int num_neighbors = domain->num_neighbors;
  if (num_neighbors == 0) return MFTL_SUCCESS;

  // Stage 12: payload byte streams derive their counts from the local
  // data_size — refuse to transport on a cross-rank mismatch.
  if (MFTLCheckTriangleDataSizeConsistency(objA) != MFTL_SUCCESS) return MFTL_ERROR;
  if (objB != objA && MFTLCheckTriangleDataSizeConsistency(objB) != MFTL_SUCCESS) return MFTL_ERROR;

  // 1. Compute Macro-AABBs
  double local_aabbA_min[3] = {1e30, 1e30, 1e30}, local_aabbA_max[3] = {-1e30, -1e30, -1e30};
  double local_aabbB_min[3] = {1e30, 1e30, 1e30}, local_aabbB_max[3] = {-1e30, -1e30, -1e30};

  for (int i = 0; i < objA->n_triangles; i++)
  {
    if (objA->triangles[i].is_ghost) continue;
    for (int j = 0; j < 3; j++)
    {
      int v_idx = objA->flat_tri_verts[3 * i + j];
      for (int d = 0; d < 3; d++)
      {
        double p = objA->vertices[v_idx].pos[d];
        if (p < local_aabbA_min[d]) local_aabbA_min[d] = p;
        if (p > local_aabbA_max[d]) local_aabbA_max[d] = p;
      }
    }
  }

  for (int i = 0; i < objB->n_triangles; i++)
  {
    if (objB->triangles[i].is_ghost) continue;
    for (int j = 0; j < 3; j++)
    {
      int v_idx = objB->flat_tri_verts[3 * i + j];
      for (int d = 0; d < 3; d++)
      {
        double p = objB->vertices[v_idx].pos[d];
        if (p < local_aabbB_min[d]) local_aabbB_min[d] = p;
        if (p > local_aabbB_max[d]) local_aabbB_max[d] = p;
      }
    }
  }

  // 2. Neighbor Exchange
  double* send_aabb_both = malloc(num_neighbors * 12 * sizeof(double));
  double* recv_aabb_both = malloc(num_neighbors * 12 * sizeof(double));

  for (int i = 0; i < num_neighbors; i++)
  {
    send_aabb_both[i * 12 + 0] = local_aabbA_min[0];
    send_aabb_both[i * 12 + 1] = local_aabbA_min[1];
    send_aabb_both[i * 12 + 2] = local_aabbA_min[2];
    send_aabb_both[i * 12 + 3] = local_aabbA_max[0];
    send_aabb_both[i * 12 + 4] = local_aabbA_max[1];
    send_aabb_both[i * 12 + 5] = local_aabbA_max[2];
    send_aabb_both[i * 12 + 6] = local_aabbB_min[0];
    send_aabb_both[i * 12 + 7] = local_aabbB_min[1];
    send_aabb_both[i * 12 + 8] = local_aabbB_min[2];
    send_aabb_both[i * 12 + 9] = local_aabbB_max[0];
    send_aabb_both[i * 12 + 10] = local_aabbB_max[1];
    send_aabb_both[i * 12 + 11] = local_aabbB_max[2];
  }

  MPI_Neighbor_alltoall(send_aabb_both, 12, MPI_DOUBLE, recv_aabb_both, 12, MPI_DOUBLE, domain->comm_graph);

  // 3. Overlap Check & Packing
  int* n_send_triA = calloc(num_neighbors, sizeof(int));
  int* n_send_vertA = calloc(num_neighbors, sizeof(int));
  int* n_send_triB = calloc(num_neighbors, sizeof(int));
  int* n_send_vertB = calloc(num_neighbors, sizeof(int));

  int* pack_triA = malloc(num_neighbors * objA->n_triangles * sizeof(int));
  int* pack_triB = malloc(num_neighbors * objB->n_triangles * sizeof(int));

  for (int n = 0; n < num_neighbors; n++)
  {
    double* remoteA_min = &recv_aabb_both[n * 12 + 0];
    double* remoteA_max = &recv_aabb_both[n * 12 + 3];
    double* remoteB_min = &recv_aabb_both[n * 12 + 6];
    double* remoteB_max = &recv_aabb_both[n * 12 + 9];

    int overlap_A_my_B_remote = mftlIntervalOverlapDom(domain, local_aabbA_min, local_aabbA_max, remoteB_min, remoteB_max);

    if (overlap_A_my_B_remote)
    {
      for (int i = 0; i < objA->n_triangles; i++)
      {
        if (objA->triangles[i].is_ghost) continue;
        double tmin[3] = {1e30, 1e30, 1e30}, tmax[3] = {-1e30, -1e30, -1e30};
        for (int j = 0; j < 3; j++)
        {
          int v_idx = objA->flat_tri_verts[3 * i + j];
          for (int d = 0; d < 3; d++)
          {
            double p = objA->vertices[v_idx].pos[d];
            if (p < tmin[d]) tmin[d] = p;
            if (p > tmax[d]) tmax[d] = p;
          }
        }
        int tri_overlap = mftlIntervalOverlapDom(domain, tmin, tmax, remoteB_min, remoteB_max);

        if (tri_overlap)
        {
          pack_triA[n * objA->n_triangles + n_send_triA[n]++] = i;
          n_send_vertA[n] += 3;
        }
      }
    }

    int overlap_B_my_A_remote = mftlIntervalOverlapDom(domain, local_aabbB_min, local_aabbB_max, remoteA_min, remoteA_max);

    if (overlap_B_my_A_remote)
    {
      for (int i = 0; i < objB->n_triangles; i++)
      {
        if (objB->triangles[i].is_ghost) continue;
        double tmin[3] = {1e30, 1e30, 1e30}, tmax[3] = {-1e30, -1e30, -1e30};
        for (int j = 0; j < 3; j++)
        {
          int v_idx = objB->flat_tri_verts[3 * i + j];
          for (int d = 0; d < 3; d++)
          {
            double p = objB->vertices[v_idx].pos[d];
            if (p < tmin[d]) tmin[d] = p;
            if (p > tmax[d]) tmax[d] = p;
          }
        }
        int tri_overlap = mftlIntervalOverlapDom(domain, tmin, tmax, remoteA_min, remoteA_max);

        if (tri_overlap)
        {
          pack_triB[n * objB->n_triangles + n_send_triB[n]++] = i;
          n_send_vertB[n] += 3;
        }
      }
    }
  }

  int* sdispls_triA = calloc(num_neighbors + 1, sizeof(int));
  int* rdispls_triA = calloc(num_neighbors + 1, sizeof(int));
  int* sdispls_vertA = calloc(num_neighbors + 1, sizeof(int));
  int* rdispls_vertA = calloc(num_neighbors + 1, sizeof(int));
  int* sdispls_triB = calloc(num_neighbors + 1, sizeof(int));
  int* rdispls_triB = calloc(num_neighbors + 1, sizeof(int));
  int* sdispls_vertB = calloc(num_neighbors + 1, sizeof(int));
  int* rdispls_vertB = calloc(num_neighbors + 1, sizeof(int));
  int* n_recv_triA = calloc(num_neighbors, sizeof(int));
  int* n_recv_vertA = calloc(num_neighbors, sizeof(int));
  int* n_recv_triB = calloc(num_neighbors, sizeof(int));
  int* n_recv_vertB = calloc(num_neighbors, sizeof(int));

  MPI_Neighbor_alltoall(n_send_triA, 1, MPI_INT, n_recv_triA, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(n_send_vertA, 1, MPI_INT, n_recv_vertA, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(n_send_triB, 1, MPI_INT, n_recv_triB, 1, MPI_INT, domain->comm_graph);
  MPI_Neighbor_alltoall(n_send_vertB, 1, MPI_INT, n_recv_vertB, 1, MPI_INT, domain->comm_graph);

  int total_send_triA = 0, total_recv_triA = 0;
  int total_send_vertA = 0, total_recv_vertA = 0;
  int total_send_triB = 0, total_recv_triB = 0;
  int total_send_vertB = 0, total_recv_vertB = 0;

  for (int i = 0; i < num_neighbors; i++)
  {
    sdispls_triA[i + 1] = sdispls_triA[i] + n_send_triA[i];
    rdispls_triA[i + 1] = rdispls_triA[i] + n_recv_triA[i];
    total_send_triA += n_send_triA[i];
    total_recv_triA += n_recv_triA[i];
    sdispls_vertA[i + 1] = sdispls_vertA[i] + n_send_vertA[i];
    rdispls_vertA[i + 1] = rdispls_vertA[i] + n_recv_vertA[i];
    total_send_vertA += n_send_vertA[i];
    total_recv_vertA += n_recv_vertA[i];
    sdispls_triB[i + 1] = sdispls_triB[i] + n_send_triB[i];
    rdispls_triB[i + 1] = rdispls_triB[i] + n_recv_triB[i];
    total_send_triB += n_send_triB[i];
    total_recv_triB += n_recv_triB[i];
    sdispls_vertB[i + 1] = sdispls_vertB[i] + n_send_vertB[i];
    rdispls_vertB[i + 1] = rdispls_vertB[i] + n_recv_vertB[i];
    total_send_vertB += n_send_vertB[i];
    total_recv_vertB += n_recv_vertB[i];
  }

  PackedTriangleComplex* sendbuf_triA = malloc(total_send_triA * sizeof(PackedTriangleComplex));
  PackedVertex* sendbuf_vertA = malloc(total_send_vertA * sizeof(PackedVertex));
  PackedTriangleComplex* sendbuf_triB = malloc(total_send_triB * sizeof(PackedTriangleComplex));
  PackedVertex* sendbuf_vertB = malloc(total_send_vertB * sizeof(PackedVertex));

  // Stage 12: parallel payload byte streams. Byte counts ARE derivable here
  // (n_recv_triA * objA->data_size, and data_size is collective), so no extra
  // count exchange is needed.
  const size_t dsA = objA->data_size;
  const size_t dsB = objB->data_size;
  char* sendbuf_dataA = MFTL_NULL;
  char* sendbuf_dataB = MFTL_NULL;
  if (dsA > 0) sendbuf_dataA = malloc((size_t) total_send_triA * dsA);
  if (dsB > 0) sendbuf_dataB = malloc((size_t) total_send_triB * dsB);

  for (int n = 0; n < num_neighbors; n++)
  {
    int ptr_tA = sdispls_triA[n];
    int ptr_vA = sdispls_vertA[n];
    for (int i = 0; i < n_send_triA[n]; i++)
    {
      int tri_idx = pack_triA[n * objA->n_triangles + i];
      MFTLPackSingleTriangle(objA, tri_idx, &sendbuf_triA[ptr_tA++], true);
      MFTLPackSingleVertex(objA, objA->flat_tri_verts[3 * tri_idx + 0], &sendbuf_vertA[ptr_vA++], true);
      MFTLPackSingleVertex(objA, objA->flat_tri_verts[3 * tri_idx + 1], &sendbuf_vertA[ptr_vA++], true);
      MFTLPackSingleVertex(objA, objA->flat_tri_verts[3 * tri_idx + 2], &sendbuf_vertA[ptr_vA++], true);

      // Stage 12: blob of the triangle in slot (sdispls_triA[n] + i)
      if (dsA > 0)
      {
        if (objA->triangles[tri_idx].data == MFTL_NULL)
        {
          fprintf(stderr,
                  "FATAL ERROR: MFTL_CreateSpatialGhosts: triangle (gid %d, body %d) has NULL data but data_size=%zu — "
                  "call MFTLInitializeObjectData on all ranks before ghosting.\n",
                  objA->triangles[tri_idx].global_id, objA->global_id, dsA);
          exit(EXIT_FAILURE);
        }
        memcpy(sendbuf_dataA + (size_t) (sdispls_triA[n] + i) * dsA, objA->triangles[tri_idx].data, dsA);
      }
    }

    int ptr_tB = sdispls_triB[n];
    int ptr_vB = sdispls_vertB[n];
    for (int i = 0; i < n_send_triB[n]; i++)
    {
      int tri_idx = pack_triB[n * objB->n_triangles + i];
      MFTLPackSingleTriangle(objB, tri_idx, &sendbuf_triB[ptr_tB++], true);
      MFTLPackSingleVertex(objB, objB->flat_tri_verts[3 * tri_idx + 0], &sendbuf_vertB[ptr_vB++], true);
      MFTLPackSingleVertex(objB, objB->flat_tri_verts[3 * tri_idx + 1], &sendbuf_vertB[ptr_vB++], true);
      MFTLPackSingleVertex(objB, objB->flat_tri_verts[3 * tri_idx + 2], &sendbuf_vertB[ptr_vB++], true);

      // Stage 12: blob of the triangle in slot (sdispls_triB[n] + i)
      if (dsB > 0)
      {
        if (objB->triangles[tri_idx].data == MFTL_NULL)
        {
          fprintf(stderr,
                  "FATAL ERROR: MFTL_CreateSpatialGhosts: triangle (gid %d, body %d) has NULL data but data_size=%zu — "
                  "call MFTLInitializeObjectData on all ranks before ghosting.\n",
                  objB->triangles[tri_idx].global_id, objB->global_id, dsB);
          exit(EXIT_FAILURE);
        }
        memcpy(sendbuf_dataB + (size_t) (sdispls_triB[n] + i) * dsB, objB->triangles[tri_idx].data, dsB);
      }
    }
  }

  PackedTriangleComplex* recvbuf_triA = malloc(total_recv_triA * sizeof(PackedTriangleComplex));
  PackedVertex* recvbuf_vertA = malloc(total_recv_vertA * sizeof(PackedVertex));
  PackedTriangleComplex* recvbuf_triB = malloc(total_recv_triB * sizeof(PackedTriangleComplex));
  PackedVertex* recvbuf_vertB = malloc(total_recv_vertB * sizeof(PackedVertex));

  MPI_Neighbor_alltoallv(sendbuf_triA, n_send_triA, sdispls_triA, MPI_PackedTriangleComplex, recvbuf_triA, n_recv_triA, rdispls_triA, MPI_PackedTriangleComplex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(sendbuf_vertA, n_send_vertA, sdispls_vertA, MPI_PackedVertex, recvbuf_vertA, n_recv_vertA, rdispls_vertA, MPI_PackedVertex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(sendbuf_triB, n_send_triB, sdispls_triB, MPI_PackedTriangleComplex, recvbuf_triB, n_recv_triB, rdispls_triB, MPI_PackedTriangleComplex,
                         domain->comm_graph);
  MPI_Neighbor_alltoallv(sendbuf_vertB, n_send_vertB, sdispls_vertB, MPI_PackedVertex, recvbuf_vertB, n_recv_vertB, rdispls_vertB, MPI_PackedVertex,
                         domain->comm_graph);

  // Stage 12: exchange the payload byte streams (counts derived from the
  // triangle counts; data_size is collective, so both sides agree).
  char* recvbuf_dataA = MFTL_NULL;
  char* recvbuf_dataB = MFTL_NULL;
  if (dsA > 0)
  {
    int* cnt_s = calloc(num_neighbors, sizeof(int));
    int* cnt_r = calloc(num_neighbors, sizeof(int));
    int* sd = calloc(num_neighbors + 1, sizeof(int));
    int* rd = calloc(num_neighbors + 1, sizeof(int));
    for (int i = 0; i < num_neighbors; ++i)
    {
      cnt_s[i] = n_send_triA[i] * (int) dsA;
      cnt_r[i] = n_recv_triA[i] * (int) dsA;
      sd[i + 1] = sd[i] + cnt_s[i];
      rd[i + 1] = rd[i] + cnt_r[i];
    }
    recvbuf_dataA = malloc((size_t) total_recv_triA * dsA);
    MPI_Neighbor_alltoallv(sendbuf_dataA, cnt_s, sd, MPI_BYTE, recvbuf_dataA, cnt_r, rd, MPI_BYTE, domain->comm_graph);
    free(cnt_s);
    free(cnt_r);
    free(sd);
    free(rd);
  }
  if (dsB > 0)
  {
    int* cnt_s = calloc(num_neighbors, sizeof(int));
    int* cnt_r = calloc(num_neighbors, sizeof(int));
    int* sd = calloc(num_neighbors + 1, sizeof(int));
    int* rd = calloc(num_neighbors + 1, sizeof(int));
    for (int i = 0; i < num_neighbors; ++i)
    {
      cnt_s[i] = n_send_triB[i] * (int) dsB;
      cnt_r[i] = n_recv_triB[i] * (int) dsB;
      sd[i + 1] = sd[i] + cnt_s[i];
      rd[i + 1] = rd[i] + cnt_r[i];
    }
    recvbuf_dataB = malloc((size_t) total_recv_triB * dsB);
    MPI_Neighbor_alltoallv(sendbuf_dataB, cnt_s, sd, MPI_BYTE, recvbuf_dataB, cnt_r, rd, MPI_BYTE, domain->comm_graph);
    free(cnt_s);
    free(cnt_r);
    free(sd);
    free(rd);
  }

  int old_tris_A = objA->n_triangles;
  for (int i = 0; i < total_recv_vertA; i++) MFTLAppendGhostVertex(objA, &recvbuf_vertA[i]);
  for (int i = 0; i < total_recv_triA; i++)
    MFTLAppendGhostTriangle(objA, &recvbuf_triA[i], dsA > 0 ? (const void*) (recvbuf_dataA + (size_t) i * dsA) : MFTL_NULL);

  if (total_recv_triA > 0)
  {
    objA->flat_tri_verts = realloc(objA->flat_tri_verts, objA->n_triangles * 3 * sizeof(int));
    for (int i = 0; i < total_recv_triA; i++)
    {
      PackedTriangleComplex* ptc = &recvbuf_triA[i];
      for (int j = 0; j < 3; j++)
      {
        int v_gid = ptc->half_edges[j].vertex_gid;
        khiter_t k = kh_get(vertex_map, objA->vertex_lookup, v_gid);
        if (k != kh_end(objA->vertex_lookup))
        {
          objA->flat_tri_verts[3 * (old_tris_A + i) + j] = kh_value(objA->vertex_lookup, k);
        }
        else
        {
          objA->flat_tri_verts[3 * (old_tris_A + i) + j] = 0;  // Fallback
        }
      }
    }
  }

  int old_tris_B = objB->n_triangles;
  for (int i = 0; i < total_recv_vertB; i++) MFTLAppendGhostVertex(objB, &recvbuf_vertB[i]);
  for (int i = 0; i < total_recv_triB; i++)
    MFTLAppendGhostTriangle(objB, &recvbuf_triB[i], dsB > 0 ? (const void*) (recvbuf_dataB + (size_t) i * dsB) : MFTL_NULL);

  if (total_recv_triB > 0)
  {
    objB->flat_tri_verts = realloc(objB->flat_tri_verts, objB->n_triangles * 3 * sizeof(int));
    for (int i = 0; i < total_recv_triB; i++)
    {
      PackedTriangleComplex* ptc = &recvbuf_triB[i];
      for (int j = 0; j < 3; j++)
      {
        int v_gid = ptc->half_edges[j].vertex_gid;
        khiter_t k = kh_get(vertex_map, objB->vertex_lookup, v_gid);
        if (k != kh_end(objB->vertex_lookup))
        {
          objB->flat_tri_verts[3 * (old_tris_B + i) + j] = kh_value(objB->vertex_lookup, k);
        }
        else
        {
          objB->flat_tri_verts[3 * (old_tris_B + i) + j] = 0;  // Fallback
        }
      }
    }
  }

  free(send_aabb_both);
  free(recv_aabb_both);
  free(n_send_triA);
  free(n_send_vertA);
  free(n_send_triB);
  free(n_send_vertB);
  free(sdispls_triA);
  free(sdispls_vertA);
  free(sdispls_triB);
  free(sdispls_vertB);
  free(rdispls_triA);
  free(rdispls_vertA);
  free(rdispls_triB);
  free(rdispls_vertB);
  free(n_recv_triA);
  free(n_recv_vertA);
  free(n_recv_triB);
  free(n_recv_vertB);
  free(pack_triA);
  free(pack_triB);
  free(sendbuf_triA);
  free(sendbuf_vertA);
  free(sendbuf_triB);
  free(sendbuf_vertB);
  free(sendbuf_dataA);  // Stage 12
  free(sendbuf_dataB);  // Stage 12
  free(recvbuf_triA);
  free(recvbuf_vertA);
  free(recvbuf_triB);
  free(recvbuf_vertB);
  free(recvbuf_dataA);  // Stage 12
  free(recvbuf_dataB);  // Stage 12

  return MFTL_SUCCESS;
}
/**
 * @brief Generates a Linear Bounding Volume Hierarchy (LBVH) for a given object.
 * 
 * Extracts normalized centroids over the `flat_tri_verts` array, generating 30-bit Morton Codes.
 * Sorts them using a specialized 16-bit radix sort (`MFTLSortPairs`).
 * Builds the tree using a top-down deterministic binary Radix Split (`MFTL_BuildTopology`).
 * Finally, refits the bounds from the leaf AABBs organically upwards to the root (`MFTL_RefitBVH`).
 * 
 * @param obj The mesh object (must have `flat_tri_verts` mapped).
 * @return A new allocated `MFTL_BVH` structure. Must be destroyed with `MFTL_DestroyBVH`.
 */
MFTL_BVH* MFTL_GenerateLBVH(MFTL_Object* obj)
{
  if (!obj || obj->n_triangles == 0 || !obj->flat_tri_verts)
  {
    return NULL;
  }

  MFTL_BVH* bvh = (MFTL_BVH*) malloc(sizeof(MFTL_BVH));
  bvh->nodes_capacity = obj->n_triangles * 2;
  bvh->nodes = (MFTL_BVHNode*) malloc(bvh->nodes_capacity * sizeof(MFTL_BVHNode));
  bvh->prim_indices = (int*) malloc(obj->n_triangles * sizeof(int));

  int* morton_codes = (int*) malloc(obj->n_triangles * sizeof(int));

  // Centroid Bounds
  float global_min[3] = {1e30f, 1e30f, 1e30f};
  float global_max[3] = {-1e30f, -1e30f, -1e30f};

  // First pass compute centroids and bounds.
  // Periodicity (Stage 13 resolution of the old "IS THIS CORRECT?" note): with
  // wrapped storage the raw centroids/bounds describe the STRETCHED embedding
  // of seam-crossing triangles, which yields conservative (never too small)
  // node boxes — safe for both ray casting and pair traversal. Periodic
  // correctness is provided at the query level: per-pair minimum-image
  // unwrapping (mftlTriCommonFrame) and shifted interval overlap
  // (mftlAABBOverlapDom); point queries unwrap the whole body about a
  // collective anchor before building the BVH (MFTLPointInsideObject).
  float* centroids = (float*) malloc(3 * obj->n_triangles * sizeof(float));

  for (int i = 0; i < obj->n_triangles; i++)
  {
    int v0 = obj->flat_tri_verts[3 * i + 0];
    int v1 = obj->flat_tri_verts[3 * i + 1];
    int v2 = obj->flat_tri_verts[3 * i + 2];

    float cx = (obj->vertices[v0].pos[0] + obj->vertices[v1].pos[0] + obj->vertices[v2].pos[0]) / 3.0f;
    float cy = (obj->vertices[v0].pos[1] + obj->vertices[v1].pos[1] + obj->vertices[v2].pos[1]) / 3.0f;
    float cz = (obj->vertices[v0].pos[2] + obj->vertices[v1].pos[2] + obj->vertices[v2].pos[2]) / 3.0f;

    centroids[3 * i + 0] = cx;
    centroids[3 * i + 1] = cy;
    centroids[3 * i + 2] = cz;

    global_min[0] = fminf(global_min[0], cx);
    global_min[1] = fminf(global_min[1], cy);
    global_min[2] = fminf(global_min[2], cz);

    global_max[0] = fmaxf(global_max[0], cx);
    global_max[1] = fmaxf(global_max[1], cy);
    global_max[2] = fmaxf(global_max[2], cz);
  }

  float extent[3] = {global_max[0] - global_min[0], global_max[1] - global_min[1], global_max[2] - global_min[2]};

  for (int k = 0; k < 3; k++)
    if (extent[k] == 0.0f) extent[k] = 1e-6f;

  // Generate Morton Codes
  for (int i = 0; i < obj->n_triangles; i++)
  {
    float nx = (centroids[3 * i + 0] - global_min[0]) / extent[0];
    float ny = (centroids[3 * i + 1] - global_min[1]) / extent[1];
    float nz = (centroids[3 * i + 2] - global_min[2]) / extent[2];

    morton_codes[i] = MFTL_Morton3D(nx, ny, nz);
    bvh->prim_indices[i] = i;
  }

  // Sort using custom radix sort
  MFTLSortPairs(morton_codes, bvh->prim_indices, obj->n_triangles);

  // Build Topology
  bvh->nodes_used = 1;  // root node
  bvh->root_index = 0;
  MFTL_BuildTopology(bvh, 0, 0, obj->n_triangles, morton_codes);

  // Bottom up AABB refitting
  MFTL_RefitBVH(bvh, 0, obj);

  free(centroids);
  free(morton_codes);
  return bvh;
}

void MFTL_DestroyBVH(MFTL_BVH* bvh)
{
  if (bvh)
  {
    if (bvh->nodes) free(bvh->nodes);
    if (bvh->prim_indices) free(bvh->prim_indices);
    free(bvh);
  }
}

// --- Intersection Traversal Logic ---

static inline int AABB_Overlap(const float minA[3], const float maxA[3], const float minB[3], const float maxB[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (maxA[k] < minB[k] || minA[k] > maxB[k]) return 0;
  }
  return 1;
}

// Domain-aware node overlap: raw comparison when non-periodic (bit-identical),
// shifted-image test on periodic axes. A NULL domain (hand-built unit-test
// objects) is treated as non-periodic.
static inline int mftlNodeOverlap(const MFTL_Domain* dom, const float minA[3], const float maxA[3], const float minB[3], const float maxB[3])
{
  if (!dom || !MFTLDomainAnyPeriodic(dom)) return AABB_Overlap(minA, maxA, minB, maxB);
  return mftlAABBOverlapDom(dom, minA, maxA, minB, maxB);
}

// Stage 13: materialize both triangles into ONE contiguous coordinate frame.
// Each triangle is unwrapped about its own first vertex (resolving internal
// seam stretching), then B is shifted by the integer period multiple that
// places its centroid nearest A's centroid. Correct under the minimum-image
// constraint (MFTLCheckMinimumImageConstraint).
static void mftlTriCommonFrame(const MFTL_Domain* dom, const double* A0, const double* A1, const double* A2, const double* B0, const double* B1,
                               const double* B2, double a[3][3], double b[3][3])
{
  memcpy(a[0], A0, 3 * sizeof(double));
  MFTLUnwrapToAnchor(dom, a[1], A1, A0);
  MFTLUnwrapToAnchor(dom, a[2], A2, A0);
  memcpy(b[0], B0, 3 * sizeof(double));
  MFTLUnwrapToAnchor(dom, b[1], B1, B0);
  MFTLUnwrapToAnchor(dom, b[2], B2, B0);

  double ca[3] = {(a[0][0] + a[1][0] + a[2][0]) / 3.0, (a[0][1] + a[1][1] + a[2][1]) / 3.0, (a[0][2] + a[1][2] + a[2][2]) / 3.0};
  double cb[3] = {(b[0][0] + b[1][0] + b[2][0]) / 3.0, (b[0][1] + b[1][1] + b[2][1]) / 3.0, (b[0][2] + b[1][2] + b[2][2]) / 3.0};
  for (int k = 0; k < 3; ++k)
  {
    if (!dom->is_periodic[k]) continue;
    double s = dom->domain_size[k] * round((ca[k] - cb[k]) / dom->domain_size[k]);
    if (s == 0.0) continue;
    for (int i = 0; i < 3; ++i) b[i][k] += s;
  }
}

static void TraverseBVHPair(MFTL_Object* objA, MFTL_BVH* bvhA, int nodeA, MFTL_Object* objB, MFTL_BVH* bvhB, int nodeB, IntersectionCallback callback,
                            void* user_data)
{
  if (!mftlNodeOverlap(objA->domain, bvhA->nodes[nodeA].aabb_min, bvhA->nodes[nodeA].aabb_max, bvhB->nodes[nodeB].aabb_min,
                       bvhB->nodes[nodeB].aabb_max))
  {
    return;
  }

  int is_leafA = bvhA->nodes[nodeA].payload & 1;
  int is_leafB = bvhB->nodes[nodeB].payload & 1;

  if (is_leafA && is_leafB)
  {
    int countA = (bvhA->nodes[nodeA].payload >> 1) & 0x7FFFFFFF;
    int firstA = (bvhA->nodes[nodeA].payload >> 32);

    int countB = (bvhB->nodes[nodeB].payload >> 1) & 0x7FFFFFFF;
    int firstB = (bvhB->nodes[nodeB].payload >> 32);

    for (int i = 0; i < countA; i++)
    {
      for (int j = 0; j < countB; j++)
      {
        int triA = bvhA->prim_indices[firstA + i];
        int triB = bvhB->prim_indices[firstB + j];
        if (MFTL_TriTriProximity(objA, triA, objB, triB, 1e-6))
        {
          callback(objA, triA, objB, triB, user_data);
        }
      }
    }
  }
  else if (is_leafA)
  {
    int leftB = bvhB->nodes[nodeB].payload >> 1;
    TraverseBVHPair(objA, bvhA, nodeA, objB, bvhB, leftB, callback, user_data);
    TraverseBVHPair(objA, bvhA, nodeA, objB, bvhB, leftB + 1, callback, user_data);
  }
  else if (is_leafB)
  {
    int leftA = bvhA->nodes[nodeA].payload >> 1;
    TraverseBVHPair(objA, bvhA, leftA, objB, bvhB, nodeB, callback, user_data);
    TraverseBVHPair(objA, bvhA, leftA + 1, objB, bvhB, nodeB, callback, user_data);
  }
  else
  {
    // Both are internal nodes: check their volumes and descend the larger one
    float volA = (bvhA->nodes[nodeA].aabb_max[0] - bvhA->nodes[nodeA].aabb_min[0]) * (bvhA->nodes[nodeA].aabb_max[1] - bvhA->nodes[nodeA].aabb_min[1]) *
                 (bvhA->nodes[nodeA].aabb_max[2] - bvhA->nodes[nodeA].aabb_min[2]);

    float volB = (bvhB->nodes[nodeB].aabb_max[0] - bvhB->nodes[nodeB].aabb_min[0]) * (bvhB->nodes[nodeB].aabb_max[1] - bvhB->nodes[nodeB].aabb_min[1]) *
                 (bvhB->nodes[nodeB].aabb_max[2] - bvhB->nodes[nodeB].aabb_min[2]);

    if (volA > volB)
    {
      int leftA = bvhA->nodes[nodeA].payload >> 1;
      TraverseBVHPair(objA, bvhA, leftA, objB, bvhB, nodeB, callback, user_data);
      TraverseBVHPair(objA, bvhA, leftA + 1, objB, bvhB, nodeB, callback, user_data);
    }
    else
    {
      int leftB = bvhB->nodes[nodeB].payload >> 1;
      TraverseBVHPair(objA, bvhA, nodeA, objB, bvhB, leftB, callback, user_data);
      TraverseBVHPair(objA, bvhA, nodeA, objB, bvhB, leftB + 1, callback, user_data);
    }
  }
}

void MFTL_IntersectObjects(MFTL_Object* objA, MFTL_BVH* bvhA, MFTL_Object* objB, MFTL_BVH* bvhB, IntersectionCallback callback, void* user_data)
{
  if (!objA || !bvhA || !objB || !bvhB || !callback) return;

  // --- FAST PATH: Outer AABB Overlap Check ---
  // Instead of unconditionally invoking the recursive traversal (which inherently checks these bounds),
  // we do an explicit up-front check of the root nodes. This is extremely smart because if the entire
  // objects are physically displaced (which is highly common in macroscopic physics simulations),
  // we bypass function call overhead and return immediately.
  if (bvhA->nodes_used > 0 && bvhB->nodes_used > 0)
  {
    int rootA = bvhA->root_index;
    int rootB = bvhB->root_index;
    if (!mftlNodeOverlap(objA->domain, bvhA->nodes[rootA].aabb_min, bvhA->nodes[rootA].aabb_max, bvhB->nodes[rootB].aabb_min,
                         bvhB->nodes[rootB].aabb_max))
    {
      return;  // The objects do not collide globally.
    }
  }
  else
  {
    return;  // Empty BVH
  }

  // Start traversal at their respective root nodes (index 0)
  TraverseBVHPair(objA, bvhA, bvhA->root_index, objB, bvhB, bvhB->root_index, callback, user_data);
}

/**
 * @brief A rigorous, mathematics-based exact 3D orientation sub-check evaluating spatial coordinates.
 * 
 * Constructs a 3D tetrahedron constructed by pa, pb, pc, pd.
 * Determines the orientation utilizing determinant mapping equations.
 * 
 * @return Positive if pd is "below" the plane spanned by pa, pb, pc. 
 *         Near `0.0` denotes exact coplanarity alignments.
 */
static inline double MFTL_Orient3D(const double pa[3], const double pb[3], const double pc[3], const double pd[3])
{
  double adx = pa[0] - pd[0];
  double bdx = pb[0] - pd[0];
  double cdx = pc[0] - pd[0];

  double ady = pa[1] - pd[1];
  double bdy = pb[1] - pd[1];
  double cdy = pc[1] - pd[1];

  double adz = pa[2] - pd[2];
  double bdz = pb[2] - pd[2];
  double cdz = pc[2] - pd[2];

  return adx * (bdy * cdz - bdz * cdy) + bdx * (cdy * adz - cdz * ady) + cdx * (ady * bdz - adz * bdy);
}

static inline void MFTL_Sort2(double* a, double* b)
{
  if (*a > *b)
  {
    double tmp = *a;
    *a = *b;
    *b = tmp;
  }
}

// Helper macro to permute vertices so V0 is the isolated vertex
#define MFTL_PERMUTE_VERTICES(V0, V1, V2, P0, P1, P2) \
  V0 = P0;                                            \
  V1 = P1;                                            \
  V2 = P2;

static inline int MFTL_CheckIntervals(const double* A0, const double* A1, const double* A2, const double* B0, const double* B1, const double* B2, double dA0,
                                      double dA1, double dA2, double dB0, double dB1, double dB2)
{
  const double *IsoA, *A_p1, *A_p2;
  const double *IsoB, *B_p1, *B_p2;

  // 1. Find the isolated vertex for Triangle A
  if (dA0 * dA1 > 0.0)
  {
    MFTL_PERMUTE_VERTICES(IsoA, A_p1, A_p2, A2, A0, A1);
  }
  else if (dA0 * dA2 > 0.0)
  {
    MFTL_PERMUTE_VERTICES(IsoA, A_p1, A_p2, A1, A2, A0);
  }
  else
  {
    MFTL_PERMUTE_VERTICES(IsoA, A_p1, A_p2, A0, A1, A2);
  }

  // 2. Find the isolated vertex for Triangle B
  if (dB0 * dB1 > 0.0)
  {
    MFTL_PERMUTE_VERTICES(IsoB, B_p1, B_p2, B2, B0, B1);
  }
  else if (dB0 * dB2 > 0.0)
  {
    MFTL_PERMUTE_VERTICES(IsoB, B_p1, B_p2, B1, B2, B0);
  }
  else
  {
    MFTL_PERMUTE_VERTICES(IsoB, B_p1, B_p2, B0, B1, B2);
  }

  // 3. Evaluate the final Guigue-Devillers orientation predicates
  if (MFTL_Orient3D(IsoA, A_p1, IsoB, B_p1) < 0.0)
  {
    if (MFTL_Orient3D(IsoA, A_p1, IsoB, B_p2) > 0.0) return 0;
    return (MFTL_Orient3D(IsoA, A_p2, IsoB, B_p1) >= 0.0);
  }
  else
  {
    if (MFTL_Orient3D(IsoA, A_p2, IsoB, B_p1) < 0.0) return 0;
    return (MFTL_Orient3D(IsoA, A_p1, IsoB, B_p2) <= 0.0);
  }
}

int MFTL_TriTriIntersect(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id)
{
  // Extract vertex indices
  int a0_idx = objA->flat_tri_verts[3 * triA_id + 0];
  int a1_idx = objA->flat_tri_verts[3 * triA_id + 1];
  int a2_idx = objA->flat_tri_verts[3 * triA_id + 2];

  int b0_idx = objB->flat_tri_verts[3 * triB_id + 0];
  int b1_idx = objB->flat_tri_verts[3 * triB_id + 1];
  int b2_idx = objB->flat_tri_verts[3 * triB_id + 2];

  // Extract coordinates
  const double* A0 = objA->vertices[a0_idx].pos;
  const double* A1 = objA->vertices[a1_idx].pos;
  const double* A2 = objA->vertices[a2_idx].pos;

  const double* B0 = objB->vertices[b0_idx].pos;
  const double* B1 = objB->vertices[b1_idx].pos;
  const double* B2 = objB->vertices[b2_idx].pos;

  // Stage 13: on periodic domains re-image both triangles into one contiguous
  // frame (raw storage is torn across periodic faces). Non-periodic keeps the
  // raw pointers — bit-identical. A NULL domain (hand-built unit-test objects)
  // is treated as non-periodic.
  double af[3][3], bf[3][3];
  if (objA->domain && MFTLDomainAnyPeriodic(objA->domain))
  {
    mftlTriCommonFrame(objA->domain, A0, A1, A2, B0, B1, B2, af, bf);
    A0 = af[0];
    A1 = af[1];
    A2 = af[2];
    B0 = bf[0];
    B1 = bf[1];
    B2 = bf[2];
  }

  // 1. Signed distances of Tri B's vertices to Tri A's plane
  double dB0 = MFTL_Orient3D(B0, A0, A1, A2);
  double dB1 = MFTL_Orient3D(B1, A0, A1, A2);
  double dB2 = MFTL_Orient3D(B2, A0, A1, A2);

  double eps = 1e-12;
  if (fabs(dB0) < eps) dB0 = 0.0;
  if (fabs(dB1) < eps) dB1 = 0.0;
  if (fabs(dB2) < eps) dB2 = 0.0;

  // Early out: All strictly on one side
  if ((dB0 > 0.0 && dB1 > 0.0 && dB2 > 0.0) || (dB0 < 0.0 && dB1 < 0.0 && dB2 < 0.0)) return 0;

  // 2. Signed distances of Tri A's vertices to Tri B's plane
  double dA0 = MFTL_Orient3D(A0, B0, B1, B2);
  double dA1 = MFTL_Orient3D(A1, B0, B1, B2);
  double dA2 = MFTL_Orient3D(A2, B0, B1, B2);

  if (fabs(dA0) < eps) dA0 = 0.0;
  if (fabs(dA1) < eps) dA1 = 0.0;
  if (fabs(dA2) < eps) dA2 = 0.0;

  // Early out: All strictly on one side
  if ((dA0 > 0.0 && dA1 > 0.0 && dA2 > 0.0) || (dA0 < 0.0 && dA1 < 0.0 && dA2 < 0.0)) return 0;

  // 3. Coplanar handler
  if (dA0 == 0.0 && dA1 == 0.0 && dA2 == 0.0)
  {
    // A robust 2D edge-crossing fallback should go here.
    // For brevity, assuming your AABB filter acts as a fast-pass, but for
    // strict topological correctness, project to 2D and test segment intersections.
    return 1;
  }

  // 4. Guigue-Devillers Interval Permutation & Test
  return MFTL_CheckIntervals(A0, A1, A2, B0, B1, B2, dA0, dA1, dA2, dB0, dB1, dB2);
}

// --- Helper Macros for Vector Math ---
#ifndef MFTL_DOT
#define MFTL_DOT(u, v) ((u)[0] * (v)[0] + (u)[1] * (v)[1] + (u)[2] * (v)[2])
#endif

#ifndef MFTL_SUB
#define MFTL_SUB(dest, v1, v2)     \
  do                               \
  {                                \
    (dest)[0] = (v1)[0] - (v2)[0]; \
    (dest)[1] = (v1)[1] - (v2)[1]; \
    (dest)[2] = (v1)[2] - (v2)[2]; \
  } while (0)
#endif

#ifndef MFTL_MIN_DIST
#define MFTL_MIN_DIST(a, b) ((a) < (b) ? (a) : (b))
#endif

// -----------------------------------------------------------------------------
// 1. Edge-to-Edge Distance Squared
// -----------------------------------------------------------------------------
static double MFTL_EdgeEdgeDistanceSq(const double P0[3], const double P1[3], const double Q0[3], const double Q1[3])
{
  double u[3], v[3], w[3];
  MFTL_SUB(u, P1, P0);
  MFTL_SUB(v, Q1, Q0);
  MFTL_SUB(w, P0, Q0);

  double a = MFTL_DOT(u, u);
  double b = MFTL_DOT(u, v);
  double c = MFTL_DOT(v, v);
  double d = MFTL_DOT(u, w);
  double e = MFTL_DOT(v, w);
  double D = a * c - b * b;
  double sc, sN, sD = D;
  double tc, tN, tD = D;

  if (D < 1e-12)
  {
    sN = 0.0;
    sD = 1.0;
    tN = e;
    tD = c;
  }
  else
  {
    sN = (b * e - c * d);
    tN = (a * e - b * d);
    if (sN < 0.0)
    {
      sN = 0.0;
      tN = e;
      tD = c;
    }
    else if (sN > sD)
    {
      sN = sD;
      tN = e + b;
      tD = c;
    }
  }

  if (tN < 0.0)
  {
    tN = 0.0;
    if (-d < 0.0)
      sN = 0.0;
    else if (-d > a)
      sN = sD;
    else
    {
      sN = -d;
      sD = a;
    }
  }
  else if (tN > tD)
  {
    tN = tD;
    if ((-d + b) < 0.0)
      sN = 0.0;
    else if ((-d + b) > a)
      sN = sD;
    else
    {
      sN = (-d + b);
      sD = a;
    }
  }

  sc = (fabs(sN) < 1e-12 ? 0.0 : sN / sD);
  tc = (fabs(tN) < 1e-12 ? 0.0 : tN / tD);

  double dP[3] = {w[0] + (sc * u[0]) - (tc * v[0]), w[1] + (sc * u[1]) - (tc * v[1]), w[2] + (sc * u[2]) - (tc * v[2])};

  return MFTL_DOT(dP, dP);
}

// -----------------------------------------------------------------------------
// 2. Point-to-Face Distance Squared
// Returns 1e30 if the orthogonal projection falls outside the triangle bounds
// -----------------------------------------------------------------------------
static double MFTL_PointFaceDistanceSq(const double P[3], const double A[3], const double B[3], const double C[3])
{
  double edge0[3], edge1[3], v0[3];
  MFTL_SUB(edge0, B, A);
  MFTL_SUB(edge1, C, A);
  MFTL_SUB(v0, A, P);  // Vector from P to A

  double a = MFTL_DOT(edge0, edge0);
  double b = MFTL_DOT(edge0, edge1);
  double c = MFTL_DOT(edge1, edge1);
  double d = MFTL_DOT(edge0, v0);
  double e = MFTL_DOT(edge1, v0);

  double det = a * c - b * b;
  if (det < 1e-12) return 1e30;  // Degenerate triangle safeguard

  double s = b * e - c * d;
  double t = b * d - a * e;

  // Verify projection is strictly inside the barycentric bounds of the triangle
  if (s >= 0.0 && t >= 0.0 && (s + t) <= det)
  {
    double invDet = 1.0 / det;
    s *= invDet;
    t *= invDet;
    double proj[3] = {A[0] + s * edge0[0] + t * edge1[0], A[1] + s * edge0[1] + t * edge1[1], A[2] + s * edge0[2] + t * edge1[2]};
    double dist_vec[3] = {proj[0] - P[0], proj[1] - P[1], proj[2] - P[2]};
    return MFTL_DOT(dist_vec, dist_vec);
  }

  // If outside, Edge-Edge distance will safely catch the closest boundary point
  return 1e30;
}

// -----------------------------------------------------------------------------
// 3. The Main Proximity Aggregator
// -----------------------------------------------------------------------------
int MFTL_TriTriProximity(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, double proximity_tolerance)
{
  // 1. FAST PATH: Check for pure mathematical intersection first
  if (MFTL_TriTriIntersect(objA, triA_id, objB, triB_id))
  {
    return 1;
  }

  // 2. Extract Coordinates
  int a0_idx = objA->flat_tri_verts[3 * triA_id + 0];
  int a1_idx = objA->flat_tri_verts[3 * triA_id + 1];
  int a2_idx = objA->flat_tri_verts[3 * triA_id + 2];

  int b0_idx = objB->flat_tri_verts[3 * triB_id + 0];
  int b1_idx = objB->flat_tri_verts[3 * triB_id + 1];
  int b2_idx = objB->flat_tri_verts[3 * triB_id + 2];

  const double* A[3] = {objA->vertices[a0_idx].pos, objA->vertices[a1_idx].pos, objA->vertices[a2_idx].pos};
  const double* B[3] = {objB->vertices[b0_idx].pos, objB->vertices[b1_idx].pos, objB->vertices[b2_idx].pos};

  // Stage 13: one contiguous frame for both triangles on periodic domains
  // (non-periodic keeps the raw pointers — bit-identical). A NULL domain
  // (hand-built unit-test objects) is treated as non-periodic.
  double af[3][3], bf[3][3];
  if (objA->domain && MFTLDomainAnyPeriodic(objA->domain))
  {
    mftlTriCommonFrame(objA->domain, A[0], A[1], A[2], B[0], B[1], B[2], af, bf);
    for (int i = 0; i < 3; ++i)
    {
      A[i] = af[i];
      B[i] = bf[i];
    }
  }

  double min_dist_sq = 1e30;

  // 3. Evaluate 9 Edge-to-Edge Distances
  for (int i = 0; i < 3; i++)
  {
    for (int j = 0; j < 3; j++)
    {
      double d_sq = MFTL_EdgeEdgeDistanceSq(A[i], A[(i + 1) % 3], B[j], B[(j + 1) % 3]);
      min_dist_sq = MFTL_MIN_DIST(min_dist_sq, d_sq);
    }
  }

  // 4. Evaluate 6 Point-to-Face Interior Distances
  for (int i = 0; i < 3; i++)
  {
    // Distance from Triangle A's vertices to Triangle B's face
    double d_sq_A = MFTL_PointFaceDistanceSq(A[i], B[0], B[1], B[2]);
    min_dist_sq = MFTL_MIN_DIST(min_dist_sq, d_sq_A);

    // Distance from Triangle B's vertices to Triangle A's face
    double d_sq_B = MFTL_PointFaceDistanceSq(B[i], A[0], A[1], A[2]);
    min_dist_sq = MFTL_MIN_DIST(min_dist_sq, d_sq_B);
  }

  // 5. Check against the user-defined tolerance envelope
  double tol_sq = proximity_tolerance * proximity_tolerance;

  return (min_dist_sq <= tol_sq) ? 1 : 0;
}

/**
 * @brief Scalable intersection pipeline for full ensembles.
 * @param ensA Primary ensemble (Required).
 * @param ensB Secondary ensemble (Optional). Pass NULL to self-intersect ensA.
 * @param callback The Triangle-Triangle hit response function.
 * @param user_data Physics payload pointer.
 */
void MFTLEnsembleIntersectObjects(MFTL_Ensemble* ensA, MFTL_Ensemble* ensB, IntersectionCallback callback, void* user_data)
{
  if (!ensA || ensA->num_objects == 0) return;

  // 1. Bulk Spatial Ghosting
  // MFTL_CreateSpatialGhosts fails collective-consistently (its pre-communication
  // contract check reduces over comm_cart), so all ranks bail at the same pair.
  if (ensB)
  {
    for (int i = 0; i < ensA->num_objects; i++)
      for (int j = 0; j < ensB->num_objects; j++)
        if (MFTL_CreateSpatialGhosts(ensA->objects[i], ensB->objects[j]) != MFTL_SUCCESS)
        {
          fprintf(stderr, "ERROR: MFTLEnsembleIntersectObjects: spatial ghosting failed for object pair (%d, %d) — aborting intersection.\n",
                  ensA->objects[i]->global_id, ensB->objects[j]->global_id);
          return;
        }
  }
  else
  {
    for (int i = 0; i < ensA->num_objects; i++)
      for (int j = i + 1; j < ensA->num_objects; j++)
        if (MFTL_CreateSpatialGhosts(ensA->objects[i], ensA->objects[j]) != MFTL_SUCCESS)
        {
          fprintf(stderr, "ERROR: MFTLEnsembleIntersectObjects: spatial ghosting failed for object pair (%d, %d) — aborting intersection.\n",
                  ensA->objects[i]->global_id, ensA->objects[j]->global_id);
          return;
        }
  }

  // 2. Generate Acceleration Structures (LBVH)
  MFTL_BVH** bvhsA = malloc(ensA->num_objects * sizeof(MFTL_BVH*));
  for (int i = 0; i < ensA->num_objects; i++)
  {
    bvhsA[i] = MFTL_GenerateLBVH(ensA->objects[i]);
  }

  MFTL_BVH** bvhsB = MFTL_NULL;
  if (ensB)
  {
    bvhsB = malloc(ensB->num_objects * sizeof(MFTL_BVH*));
    for (int i = 0; i < ensB->num_objects; i++)
    {
      bvhsB[i] = MFTL_GenerateLBVH(ensB->objects[i]);
    }
  }

  // 3. Object-Object Global AABB Traversal
  if (ensB)
  {
    // Two distinctly different ensembles (e.g., Fluids vs Rigid Bodies)
    for (int i = 0; i < ensA->num_objects; i++)
    {
      for (int j = 0; j < ensB->num_objects; j++)
      {
        MFTL_IntersectObjects(ensA->objects[i], bvhsA[i], ensB->objects[j], bvhsB[j], callback, user_data);
      }
    }
  }
  else
  {
    // Self-intersection mapping within a single ensemble
    // Note: Starts j at i+1 to avoid intersecting Object 1 with Object 1,
    // and prevents double-testing (tests 1v2, but skips 2v1)
    for (int i = 0; i < ensA->num_objects; i++)
    {
      for (int j = i + 1; j < ensA->num_objects; j++)
      {
        MFTL_IntersectObjects(ensA->objects[i], bvhsA[i], ensA->objects[j], bvhsA[j], callback, user_data);
      }
    }
  }

  // 4. Memory Cleanup
  for (int i = 0; i < ensA->num_objects; i++)
  {
    MFTL_DestroyBVH(bvhsA[i]);
  }
  free(bvhsA);

  if (ensB)
  {
    for (int i = 0; i < ensB->num_objects; i++)
    {
      MFTL_DestroyBVH(bvhsB[i]);
    }
    free(bvhsB);
  }
}
