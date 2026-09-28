/**
 * @file bvh_build.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Primitive access, and building, refitting and destroying the hierarchy:
 *        Morton keys, stable radix sort, top-down radix split, bottom-up fit.
 */

#include "mftl_bvh.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "bvh_internal.h"
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Primitive access: every consumer reads the mesh through prim_source(), so
 * local and halo primitives are the same thing to it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Resolves a primitive index to its source object and triangle index.
 *
 * @param      bvh   The hierarchy.
 * @param      p     Primitive index in `[0, n_prims)`.
 * @param[out] t_out Receives the triangle index within the returned object.
 * @return `bvh->obj` for a local primitive, `bvh->halo` for a remote one.
 */
static inline const MFTL_Object *prim_source(const MFTL_BVH *bvh, int p, int *t_out)
{
  MFTL_ASSERT(p >= 0 && p < bvh->n_prims && "BVH primitive index out of range");

  if (p >= bvh->n_local_prims)
  {
    *t_out = p - bvh->n_local_prims;
    MFTL_ASSERT(bvh->halo != MFTL_NULL && *t_out < bvh->halo->tris.n && "BVH halo primitive without a patch");
    return bvh->halo;
  }
  *t_out = p;
  return bvh->obj;
}

const MFTL_Object *MFTL_BVHPrimTriangle(const MFTL_BVH *bvh, int p, int *tri)
{
  MFTL_ASSERT(bvh != MFTL_NULL && tri != MFTL_NULL);
  return prim_source(bvh, p, tri);
}

void MFTL_BVHPrimVerts(const MFTL_BVH *bvh, int p, double out[3][3])
{
  int t;
  const MFTL_Object *const src = prim_source(bvh, p, &t);
  MFTLGetTriangleCorners(src, t, out);
}

int MFTL_BVHPrimGlobalId(const MFTL_BVH *bvh, int p)
{
  int t;
  const MFTL_Object *const src = prim_source(bvh, p, &t);
  return src->tris.rec[t].global_id;
}

int MFTL_BVHPrimOwner(const MFTL_BVH *bvh, int p)
{
  int t;
  const MFTL_Object *const src = prim_source(bvh, p, &t);
  return src->tris.rec[t].owner_proc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Topology: the radix split
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Common-prefix length of the 95-bit keys (Morton code, index) at sorted
 *        slots @p i and @p j.
 *
 * A radix split needs UNIQUE keys: equal codes (two triangles in one Morton
 * cell -- common at fine resolution) would never split. Appending the slot index
 * (ascending, since the array is sorted) makes the keys unique without another
 * sort and bounds the tree depth by 95 (Karras 2012):
 * \f$\delta = \clz_{64}(Z_i \oplus Z_j)\f$ if the codes differ, else
 * \f$64 + \clz_{32}(i \oplus j)\f$.
 *
 * @param keys The sorted Morton codes.
 * @param i    First slot.
 * @param j    Second slot; must differ from @p i.
 * @return Common prefix length in bits.
 */
static inline int prefix_len(const uint64_t *restrict keys, int i, int j)
{
  const uint64_t x = keys[i] ^ keys[j];
  if (x != 0u) return __builtin_clzll(x);

  const unsigned d = (unsigned) i ^ (unsigned) j;
  /* i != j at every call site (clz of 0 is undefined). */
  MFTL_ASSERT(d != 0u && "prefix_len: called with i == j");
  return 64 + (int) __builtin_clz(d);
}

/**
 * @brief Populates node @p node_idx and, recursively, its subtree.
 *
 * The split is the last slot still sharing a longer prefix with the first than
 * the range's common prefix, found by binary search (\f$\delta(f,\cdot)\f$ is
 * non-increasing over sorted keys). The children are therefore separated by a
 * dyadic plane of the Morton grid. Recursion depth is bounded by 95.
 *
 * @param bvh        The hierarchy being built.
 * @param node_idx   Node to populate.
 * @param first_prim First primitive slot of this node's range.
 * @param prim_count Primitives in the range.
 * @param keys       The sorted Morton codes.
 * @param leaf_size  Maximum primitives a leaf may hold.
 */
static void build_topology(MFTL_BVH *bvh, int node_idx, int first_prim, int prim_count, const uint64_t *restrict keys, int leaf_size)
{
  if (prim_count <= leaf_size)
  {
    bvh->nodes[node_idx].payload = 1u | ((uint64_t) prim_count << 1) | ((uint64_t) first_prim << 32);
    return;
  }

  const int last_prim = first_prim + prim_count - 1;
  const int common = prefix_len(keys, first_prim, last_prim);

  int split_idx = first_prim;
  int step = prim_count;
  do
  {
    step = (step + 1) >> 1;  // ceiling halving
    const int cand = split_idx + step;
    if (cand < last_prim && prefix_len(keys, first_prim, cand) > common) split_idx = cand;
  } while (step > 1);

  const int left_count = split_idx + 1 - first_prim;

  /* Children as a contiguous pair after the parent (the refit's reverse scan
   * relies on child > parent). */
  const int left_child = bvh->nodes_used;
  bvh->nodes_used += 2;
  MFTL_ASSERT(bvh->nodes_used <= bvh->nodes_capacity && "BVH node capacity exceeded");

  bvh->nodes[node_idx].payload = ((uint64_t) left_child << 1);

  build_topology(bvh, left_child, first_prim, left_count, keys, leaf_size);
  build_topology(bvh, left_child + 1, split_idx + 1, prim_count - left_count, keys, leaf_size);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Refit
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLBVHRefit(MFTL_BVH *bvh)
{
  /* An empty hierarchy still gets a tolerance. */
  if (bvh->nodes_used <= 0)
  {
    bvh->bounding_radius = (bvh->radius_hint > 0.0) ? bvh->radius_hint : 0.0;
    bvh->tolerance = MFTL_MAX(bvh->params.proximity_floor, bvh->params.proximity_factor * bvh->bounding_radius);
    if (!(bvh->tolerance >= 0.0)) bvh->tolerance = 0.0;
    return;
  }

  MFTL_BVHNode *const restrict nodes = bvh->nodes;
  const int *const restrict prims = bvh->prim_indices;

  /* ── Pass 1: reverse scan, uninflated ─────────────────────────────────── */
  for (int i = bvh->nodes_used - 1; i >= 0; --i)
  {
    MFTL_BVHNode *const n = &nodes[i];

    if (MFTL_BVHNodeIsLeaf(n))
    {
      const int first = MFTL_BVHLeafFirst(n);
      const int count = MFTL_BVHLeafCount(n);

      double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
      double hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};

      for (int s = 0; s < count; ++s)
      {
        double tri[3][3];
        MFTL_BVHPrimVerts(bvh, prims[first + s], tri);
        for (int c = 0; c < 3; ++c)
        {
          for (int k = 0; k < 3; ++k)
          {
            if (tri[c][k] < lo[k]) lo[k] = tri[c][k];
            if (tri[c][k] > hi[k]) hi[k] = tri[c][k];
          }
        }
      }

      /* Defensive: an empty leaf gets a box that overlaps nothing. */
      if (count == 0)
      {
        for (int k = 0; k < 3; ++k)
        {
          lo[k] = 0.0;
          hi[k] = -1.0;
        }
      }

      /* Rounded OUTWARD into float (bvh_internal.h). */
      for (int k = 0; k < 3; ++k)
      {
        n->aabb_min[k] = mftl_bvh_floor_f(lo[k]);
        n->aabb_max[k] = mftl_bvh_ceil_f(hi[k]);
      }
    }
    else
    {
      const int l = MFTL_BVHLeftChild(n);
      const MFTL_BVHNode *const a = &nodes[l];
      const MFTL_BVHNode *const b = &nodes[l + 1];
      MFTL_ASSERT(l > i && l + 1 < bvh->nodes_used && "BVH child index must exceed its parent's");

      for (int k = 0; k < 3; ++k)
      {
        n->aabb_min[k] = fminf(a->aabb_min[k], b->aabb_min[k]);
        n->aabb_max[k] = fmaxf(a->aabb_max[k], b->aabb_max[k]);
      }
    }
  }

  /* ── Pass 2: the tolerance, from the root box ─────────────────────────── */
  {
    /* Half the LONGEST SIDE of the root box (R for a sphere), not half the
     * diagonal, which would inflate every envelope by sqrt(3). */
    const MFTL_BVHNode *const root = &nodes[bvh->root_index];
    double longest = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double d = (double) root->aabb_max[k] - (double) root->aabb_min[k];
      if (d > longest) longest = d;
    }
    bvh->bounding_radius = 0.5 * longest;

    /* The GLOBAL radius wins when known (MFTL_BVH::radius_hint). */
    const double radius = (bvh->radius_hint > 0.0) ? bvh->radius_hint : bvh->bounding_radius;

    const double scaled = bvh->params.proximity_factor * radius;
    bvh->tolerance = MFTL_MAX(bvh->params.proximity_floor, scaled);
    if (!(bvh->tolerance >= 0.0)) bvh->tolerance = 0.0;  // NaN-proof
  }

  /* ── Pass 3: inflate every node ───────────────────────────────────────── */
  if (bvh->tolerance > 0.0)
  {
    const float tol = (float) bvh->tolerance;
    for (int i = 0; i < bvh->nodes_used; ++i)
    {
      for (int k = 0; k < 3; ++k)
      {
        nodes[i].aabb_min[k] -= tol;
        nodes[i].aabb_max[k] += tol;
      }
    }
  }
}

int MFTL_RefitLBVH(MFTL_BVH *bvh, const MFTL_Object *obj)
{
  if (bvh == MFTL_NULL || obj == MFTL_NULL) return MFTL_ERROR;

  /* Leaves address primitives by index: after a re-triangulation the indices no
   * longer name the same triangles, and a refit would give WRONG contacts. */
  const int n_local = bvh->params.include_ghosts ? obj->tris.n : obj->tris.n_local;
  const int n_halo = (bvh->halo != MFTL_NULL) ? bvh->halo->tris.n : 0;
  if (n_local != bvh->n_local_prims || n_halo != bvh->n_halo_prims || (n_local + n_halo) != bvh->built_for_n_prims)
  {
    /* Not an assertion: "refit if you can, otherwise rebuild" is supported. */
    return MFTL_ERROR;
  }

  MFTL_ASSERT(obj == bvh->obj && "MFTL_RefitLBVH: must be handed the object the BVH was built from");
  bvh->obj = obj;

  MFTLBVHRefit(bvh);
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Build
 * ═══════════════════════════════════════════════════════════════════════════ */

MFTL_BVH *MFTL_GenerateLBVH(const MFTL_Object *obj, const MFTL_BVHBuildParams *params)
{
  if (obj == MFTL_NULL) return MFTL_NULL;

  const MFTL_BVHBuildParams p = (params != MFTL_NULL) ? *params : MFTL_BVHDefaultParams();
  const int n_local = p.include_ghosts ? obj->tris.n : obj->tris.n_local;

  MFTL_BVH *const bvh = (MFTL_BVH *) MFTLCallocOrDie(1, sizeof(MFTL_BVH), "MFTL_BVH");
  bvh->obj = obj;
  bvh->params = p;
  if (bvh->params.leaf_size < 1) bvh->params.leaf_size = MFTL_BVH_DEFAULT_LEAF_SIZE;

  bvh->n_local_prims = n_local;
  bvh->n_halo_prims = 0;
  bvh->halo = MFTL_NULL;
  bvh->n_prims = n_local;
  bvh->root_index = 0;

  /* A binary tree over n leaves has at most 2n - 1 nodes. */
  bvh->nodes_capacity = MFTL_MAX(2 * bvh->n_prims, 1);
  bvh->nodes = (MFTL_BVHNode *) MFTLAlignedAllocOrDie((size_t) bvh->nodes_capacity * sizeof(MFTL_BVHNode), "BVH nodes");
  bvh->prim_indices = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(bvh->n_prims, 1) * sizeof(int), "BVH primitive indices");

  MFTLBVHBuildFromPrims(bvh);
  return bvh;
}

/* The part of the build that depends only on the primitive soup. Documented on
 * the declaration in bvh_internal.h. */
void MFTLBVHBuildFromPrims(MFTL_BVH *bvh)
{
  const int n = bvh->n_prims;

  /* Empty: no nodes, but a tolerance from the floor / radius hint. */
  if (n <= 0)
  {
    bvh->nodes_used = 0;
    bvh->built_for_n_prims = 0;
    bvh->bounding_radius = (bvh->radius_hint > 0.0) ? bvh->radius_hint : 0.0;
    bvh->tolerance = MFTL_MAX(bvh->params.proximity_floor, bvh->params.proximity_factor * bvh->bounding_radius);
    if (!(bvh->tolerance >= 0.0)) bvh->tolerance = 0.0;
    return;
  }

  /* Centroids of the SEAM-CONSISTENT corners, deliberately NOT wrapped: as a
   * sort key a wrapped centroid would put neighbours across the seam at
   * opposite ends of the key range. */
  double *const centroids = (double *) MFTLMallocOrDie(3 * (size_t) n * sizeof(double), "BVH centroids");

  double lo[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double hi[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};

  for (int i = 0; i < n; ++i)
  {
    double tri[3][3];
    MFTL_BVHPrimVerts(bvh, i, tri);

    for (int k = 0; k < 3; ++k)
    {
      const double c = (tri[0][k] + tri[1][k] + tri[2][k]) * MFTL_ONE_THIRD;
      centroids[3 * i + k] = c;
      if (c < lo[k]) lo[k] = c;
      if (c > hi[k]) hi[k] = c;
    }
  }

  /* MFTLMorton3D (21 bits per axis), one common scale so the cells are cubic --
   * the same key as the pool renumbering. */
  double range = hi[0] - lo[0];
  if (hi[1] - lo[1] > range) range = hi[1] - lo[1];
  if (hi[2] - lo[2] > range) range = hi[2] - lo[2];
  /* Any FINITE scale is valid (no absolute floor). */
  const double inv_range = (range > MFTL_MORTON_MAX / MFTL_DBL_MAX) ? MFTL_MORTON_MAX / range : 0.0;

  uint64_t *const keys = (uint64_t *) MFTLMallocOrDie((size_t) n * sizeof(uint64_t), "BVH Morton keys");

  for (int i = 0; i < n; ++i)
  {
    const uint64_t ix = (uint64_t) ((centroids[3 * i + 0] - lo[0]) * inv_range);
    const uint64_t iy = (uint64_t) ((centroids[3 * i + 1] - lo[1]) * inv_range);
    const uint64_t iz = (uint64_t) ((centroids[3 * i + 2] - lo[2]) * inv_range);
    keys[i] = MFTLMorton3D(ix, iy, iz);
    bvh->prim_indices[i] = i;
  }

  free(centroids);

  /* Stable sort: the tree is a deterministic function of the input. */
  MFTLSortPairs64(keys, bvh->prim_indices, n);

  bvh->nodes_used = 1;
  bvh->root_index = 0;
  build_topology(bvh, 0, 0, n, keys, bvh->params.leaf_size);

  free(keys);

  bvh->built_for_n_prims = n;
  MFTLBVHRefit(bvh);
}

int MFTL_BVHSetHalo(MFTL_BVH *bvh, const MFTL_Object *patch)
{
  if (bvh == MFTL_NULL) return MFTL_ERROR;

  bvh->halo = (patch != MFTL_NULL && patch->tris.n > 0) ? patch : MFTL_NULL;
  bvh->n_halo_prims = (bvh->halo != MFTL_NULL) ? bvh->halo->tris.n : 0;

  const int n = bvh->n_local_prims + bvh->n_halo_prims;
  bvh->n_prims = n;

  const int need_nodes = MFTL_MAX(2 * n, 1);
  if (need_nodes > bvh->nodes_capacity)
  {
    free(bvh->nodes);
    bvh->nodes_capacity = need_nodes;
    bvh->nodes = (MFTL_BVHNode *) MFTLAlignedAllocOrDie((size_t) bvh->nodes_capacity * sizeof(MFTL_BVHNode), "BVH nodes");
  }


  free(bvh->prim_indices);
  bvh->prim_indices = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(int), "BVH primitive indices");

  MFTLBVHBuildFromPrims(bvh);
  return MFTL_SUCCESS;
}

int MFTL_BVHSetGlobalRadius(MFTL_BVH *bvh, double radius)
{
  if (bvh == MFTL_NULL) return MFTL_ERROR;
  bvh->radius_hint = (radius > 0.0) ? radius : 0.0;
  MFTLBVHRefit(bvh);
  return MFTL_SUCCESS;
}

void MFTL_DestroyBVH(MFTL_BVH *bvh)
{
  if (bvh == MFTL_NULL) return;
  free(bvh->nodes);
  free(bvh->prim_indices);
  free(bvh);
}
