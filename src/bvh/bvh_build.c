/**
 * @file bvh_build.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Building, refitting and destroying the hierarchy.
 *
 * The pipeline is the classic linear BVH -- Morton codes, radix sort, top-down
 * split at the highest differing bit, bottom-up AABB fit -- with two changes
 * against the previous version, both of which are about periodicity or scale
 * rather than about the tree.
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
 * 1. Primitive geometry
 *
 * Everything downstream -- centroids, leaf boxes, the narrow phase -- reads the
 * mesh through this one function, which is what lets local and remote
 * primitives be the same thing to every consumer.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Resolves a primitive index to its source object and triangle index.
 *
 * The whole local/remote distinction, in three lines. It used to be more:
 * remote primitives lived in a private array of nine doubles per triangle,
 * which meant a second geometry representation, a second growth path and a
 * second periodic-consistency argument. A halo patch is an ordinary
 * MFTL_Object, so both kinds of primitive are just "triangle t of some object"
 * and everything downstream reads them through MFTLGetTriangleCorners.
 *
 * @param      bvh   The hierarchy.
 * @param      p     Primitive index in `[0, n_prims)`.
 * @param[out] t_out Receives the triangle index within the returned object.
 * @return The owning object: `bvh->obj` for a local primitive, `bvh->halo` for
 *         a remote one.
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
 * @brief Length of the common prefix of the sorted keys at @p i and @p j.
 *
 * ── Why the index takes part in the key ────────────────────────────────────
 *
 * A radix split needs every primitive to have a UNIQUE key: two primitives
 * sharing one code have a common prefix of the full key length, the search for
 * the highest differing bit finds nothing, and the node fails to split -- which
 * in the recursive builder is an infinite descent, and in the previous version's
 * guarded form was a leaf containing an unbounded number of primitives.
 *
 * Duplicate codes are not exotic. Two triangles smaller than one Morton cell
 * share a code, which for a resolved interface mesh is most of the mesh at low
 * bit counts -- the previous version's 10 bits per axis put a 1024^3 grid over
 * the object, and any mesh finer than that in any region collides.
 *
 * Extending the key with the primitive's INDEX makes it unique by construction
 * without another sort: the array is already sorted, so indices are ascending,
 * and comparing (key, index) lexicographically is exactly what a 95-bit radix
 * tree over unique keys does. It also bounds the recursion depth at 95 rather
 * than at the primitive count.
 *
 * Two levels because 63 + 32 does not fit a machine word: equal keys fall
 * through to the index difference, offset by 64 so the ordering is consistent.
 *
 * @param keys The sorted Morton codes.
 * @param i    First slot.
 * @param j    Second slot. Must differ from @p i.
 * @return Common prefix length of the 95-bit (code, index) keys, in bits.
 */
static inline int prefix_len(const uint64_t *restrict keys, int i, int j)
{
  const uint64_t x = keys[i] ^ keys[j];
  if (x != 0u) return __builtin_clzll(x);

  const unsigned d = (unsigned) i ^ (unsigned) j;
  /* i != j is guaranteed by every call site: the split search never compares a
   * slot with itself. A zero here would make __builtin_clz undefined. */
  MFTL_ASSERT(d != 0u && "prefix_len: called with i == j");
  return 64 + (int) __builtin_clz(d);
}

/**
 * @brief Populates node @p node_idx and, recursively, everything below it.
 *
 * The split is a binary search for the last primitive still sharing the range's
 * common prefix -- the radix point of the Morton curve, so the two children are
 * separated by a real axis-aligned plane rather than by a median that happens to
 * fall wherever the primitive count does.
 *
 * Recursion is kept here, unlike in the traversal and the refit. The depth is
 * bounded by the key length (95) because the keys are unique, so there is no
 * stack-overflow case to defend against, and the build runs once per
 * re-triangulation rather than once per timestep -- the two reasons that made
 * the traversal worth converting do not apply.
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

  /* Children are allocated as a contiguous pair, so the parent stores only the
   * left index. Both are greater than the parent's own index, which is what the
   * reverse-scan refit relies on. */
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
  /* An empty hierarchy still has a tolerance, because a ghost exchange may be
   * about to put remote primitives into it. */
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

      /* An empty leaf cannot arise from build_topology, but a defensive box that
       * overlaps nothing is cheaper than a NaN propagating to the root. */
      if (count == 0)
      {
        for (int k = 0; k < 3; ++k)
        {
          lo[k] = 0.0;
          hi[k] = -1.0;
        }
      }

      /* Rounded OUTWARD into float: the broad phase must never exclude geometry
       * the narrow phase would have accepted. See bvh_internal.h. */
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
    /* HALF THE LONGEST SIDE of the root box, not half its diagonal.
     *
     * Both bound the object, but only this one equals R for a sphere of radius
     * R -- and `proximity_factor` is documented as a fraction of the body, so a
     * caller writing 1e-3 for a 1 mm droplet is entitled to get 1 um. Half the
     * diagonal would hand them 1.73 um, a sqrt(3) inflation that is invisible
     * (it just makes contact trigger earlier) and therefore exactly the kind of
     * silent factor that should not be in a physical threshold. */
    const MFTL_BVHNode *const root = &nodes[bvh->root_index];
    double longest = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double d = (double) root->aabb_max[k] - (double) root->aabb_min[k];
      if (d > longest) longest = d;
    }
    bvh->bounding_radius = 0.5 * longest;

    /* The GLOBAL radius wins whenever it is known. Scaling by the partition-local
     * one would make the contact envelope a function of the process count -- see
     * MFTL_BVH::radius_hint. */
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

  /* The leaves address primitives by index and the Morton order was decided from
   * the geometry at build time, so a changed primitive count means the indices
   * no longer name the same triangles. Refusing is the only safe answer: a
   * partial refit would leave a tree whose boxes describe triangles that have
   * been renumbered, which produces wrong contacts rather than missing ones. */
  const int n_local = bvh->params.include_ghosts ? obj->tris.n : obj->tris.n_local;
  const int n_halo = (bvh->halo != MFTL_NULL) ? bvh->halo->tris.n : 0;
  if (n_local != bvh->n_local_prims || n_halo != bvh->n_halo_prims || (n_local + n_halo) != bvh->built_for_n_prims)
  {
    /* Deliberately NOT an assertion. "Refit if you can, otherwise rebuild" is a
     * supported way to drive this -- the header documents MFTL_ERROR as the
     * signal to call MFTL_GenerateLBVH -- so a caller that probes with a refit
     * after a remesh is using the API correctly, and aborting in Debug would
     * make the documented control flow unusable in the build the library is
     * developed in. The object is left untouched, so the fallback is free. */
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

  /* Worst case is one primitive per leaf, i.e. n leaves and 2n-1 nodes. The
   * capacity is not a heuristic that can be undershot, so it is not one. The
   * MFTL_MAX keeps a zero-primitive object from asking the allocator for zero
   * bytes -- an empty hierarchy is a legal thing to hold, see the header. */
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

  /* An empty hierarchy: no nodes, and a tolerance that still reflects whatever
   * absolute floor the caller set, so a later ghost exchange that fills this
   * BVH with remote primitives starts from a sane envelope rather than zero. */
  if (n <= 0)
  {
    bvh->nodes_used = 0;
    bvh->built_for_n_prims = 0;
    bvh->bounding_radius = (bvh->radius_hint > 0.0) ? bvh->radius_hint : 0.0;
    bvh->tolerance = MFTL_MAX(bvh->params.proximity_floor, bvh->params.proximity_factor * bvh->bounding_radius);
    if (!(bvh->tolerance >= 0.0)) bvh->tolerance = 0.0;
    return;
  }

  /* ── Centroids and their bounding box ─────────────────────────────────────
   *
   * The centroid is the mean of the three SEAM-CONSISTENT corners, so for a
   * triangle straddling a periodic face it is the true centroid just outside the
   * box rather than a point in the middle of the domain. This is the second half
   * of the periodic fix; the first is in MFTL_BVHPrimVerts.
   *
   * Note that the centroid is deliberately NOT wrapped back into the domain
   * afterwards, which is where MFTLGetTriangleProperties differs. A wrapped
   * centroid is the right answer for "where is this triangle" -- it is a
   * position, and redistribution routes by it. It is the wrong answer for a
   * spatial SORT KEY, because wrapping puts two triangles that are neighbours
   * across the seam at opposite ends of the key range, which is the corruption
   * the TODO was about. */
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

  /* ── Morton codes ────────────────────────────────────────────────────────
   *
   * MFTLMorton3D from mftl_object.h -- the same 63-bit, 21-bit-per-axis key the
   * pool renumbering uses, rather than the previous version's separate 30-bit
   * implementation. Two reasons to share it: a 1024^3 grid puts many triangles
   * of any resolved interface into one cell (see prefix_len), and a module that
   * carries its own copy of a spatial key will eventually disagree with the one
   * the pools are ordered by.
   *
   * ONE shared scale for all three axes, so the cells stay cubic. Scaling per
   * axis stretches the curve and makes locality worse on a flat object -- the
   * same choice, for the same reason, as reorder_vertices in src/object/derived.c. */
  double range = hi[0] - lo[0];
  if (hi[1] - lo[1] > range) range = hi[1] - lo[1];
  if (hi[2] - lo[2] > range) range = hi[2] - lo[2];
  const double inv_range = (range > MFTL_VERYSMALL) ? MFTL_MORTON_MAX / range : 0.0;

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

  /* Stable radix sort, keys and values moving together. Stability is what makes
   * the tree deterministic for a given input, which is what lets
   * `test_refit_only_matches_full_rebuild` compare two trees node by node. */
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

  /* The index array is reassigned wholesale by the build, so there is nothing to
   * preserve across the growth -- free and allocate rather than realloc. */
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
