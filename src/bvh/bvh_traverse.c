/**
 * @file bvh_traverse.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Dual-tree traversal over an explicit stack, the batched leaf-leaf
 *        narrow phase, and the ensemble contact pipeline.
 *
 * An explicit stack of (nodeA, nodeB) pairs keeps the loop-invariant state in
 * registers, cannot overflow the call stack (a dual traversal is not
 * tail-recursive), and lets leaf-leaf candidates be collected into batches of
 * 64 independent tests -- the shape a future SIMD narrow phase needs, which
 * would then change flush_batch() only. The batch is not vectorised yet.
 *
 * Periodic images: two bodies on opposite faces of a periodic domain are
 * neighbours although a domain length apart in stored coordinates, so the pair
 * is traversed once per image offset in {-L, 0, +L} per periodic axis (at most
 * 27), each gated by one root-box test.
 */

#include "mftl_bvh.h"

#include <stdlib.h>

#include "bvh_internal.h"
#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The pair stack
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One entry of the traversal stack. Eight bytes, so four per cache line pair. */
typedef struct
{
  int a;  ///< Node index in the A-side tree.
  int b;  ///< Node index in the B-side tree.
} node_pair;

/**
 * @brief Initial traversal stack depth; each tree is at most 95 levels deep, so
 *        256 is ample, and the stack grows by doubling if not.
 */
#define MFTL_BVH_STACK_INIT 256

/** @brief Leaf pairs collected before the narrow phase runs. */
#define MFTL_BVH_BATCH_SIZE 64

/** @brief Explicit traversal stack of node pairs. */
typedef struct
{
  node_pair *data;  ///< [capacity] the stack itself.
  int size;  ///< Entries currently pushed.
  int capacity;  ///< Entries allocated. Doubles on overflow.
} pair_stack;

/**
 * @brief Pushes a node pair, growing the stack if it is full.
 * @param s The stack.
 * @param a Node index in the A-side tree.
 * @param b Node index in the B-side tree.
 */
static inline void stack_push(pair_stack *s, int a, int b)
{
  if (s->size == s->capacity)
  {
    s->capacity *= 2;
    node_pair *const grown = (node_pair *) MFTLMallocOrDie((size_t) s->capacity * sizeof(node_pair), "BVH traversal stack");
    for (int i = 0; i < s->size; ++i) grown[i] = s->data[i];
    free(s->data);
    s->data = grown;
  }
  s->data[s->size].a = a;
  s->data[s->size].b = b;
  s->size++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The batched narrow phase
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Everything the narrow phase needs, hoisted out of the traversal loop. */
typedef struct
{
  MFTL_BVH *bvhA;  ///< A-side hierarchy.
  MFTL_BVH *bvhB;  ///< B-side hierarchy.

  double shift[3];  ///< Periodic image offset applied to every B coordinate.
  double tolerance;  ///< The gap at which contact is declared, for this pair.

  IntersectionCallback callback;  ///< Invoked once per surviving pair.
  void *user_data;  ///< Passed to @c callback untouched.

  /** Both trees are over ONE object: a self-proximity query. */
  bool self;
  /** The A side's optional cheap pre-test; see MFTL_BVHBuildParams::pair_filter. */
  bool (*filter)(const MFTL_BVH *, int, const MFTL_BVH *, int, void *);

  int prim_a[MFTL_BVH_BATCH_SIZE];  ///< A-side primitive of each pending pair.
  int prim_b[MFTL_BVH_BATCH_SIZE];  ///< B-side primitive of each pending pair.
  int n_batched;  ///< Pairs pending in the batch.
} narrow_ctx;

/**
 * @brief Self-proximity rules, applied before the exact test: drop pairs that
 *        share a vertex (compared by vertex GLOBAL id, since B may be a patch),
 *        and keep a pair only if `gid(A) < gid(B)` -- one of the two local
 *        orderings, and exactly one of the two owners of a cross-rank pair.
 *
 * @param bvhA A-side hierarchy.
 * @param pa   A-side primitive.
 * @param bvhB B-side hierarchy.
 * @param pb   B-side primitive.
 * @return `true` if the pair should be tested.
 */
static bool self_pair_admissible(const MFTL_BVH *bvhA, int pa, const MFTL_BVH *bvhB, int pb)
{
  int ta, tb;
  const MFTL_Object *const oa = MFTL_BVHPrimTriangle(bvhA, pa, &ta);
  const MFTL_Object *const ob = MFTL_BVHPrimTriangle(bvhB, pb, &tb);
  if (oa->tris.rec[ta].global_id >= ob->tris.rec[tb].global_id) return MFTL_FALSE;

  for (int i = 0; i < 3; ++i)
  {
    const int ga = oa->verts.rec[oa->tris.verts[3 * ta + i]].global_id;
    for (int j = 0; j < 3; ++j)
      if (ga == ob->verts.rec[ob->tris.verts[3 * tb + j]].global_id) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/**
 * @brief Runs the exact test on every batched pair and reports the hits.
 *
 * The SIMD seam: everything above produces a contiguous run of independent
 * tests. The periodic shift is applied to the B corners here, because it
 * belongs to the image being traversed, not to the primitive.
 *
 * @param ctx The narrow-phase context; its batch is emptied.
 */
static void flush_batch(narrow_ctx *ctx)
{
  const double sx = ctx->shift[0], sy = ctx->shift[1], sz = ctx->shift[2];

  for (int i = 0; i < ctx->n_batched; ++i)
  {
    if (ctx->self && !self_pair_admissible(ctx->bvhA, ctx->prim_a[i], ctx->bvhB, ctx->prim_b[i])) continue;
    if (ctx->filter && !ctx->filter(ctx->bvhA, ctx->prim_a[i], ctx->bvhB, ctx->prim_b[i], ctx->user_data)) continue;

    double TA[3][3];
    double TB[3][3];

    MFTL_BVHPrimVerts(ctx->bvhA, ctx->prim_a[i], TA);
    MFTL_BVHPrimVerts(ctx->bvhB, ctx->prim_b[i], TB);

    for (int c = 0; c < 3; ++c)
    {
      TB[c][0] += sx;
      TB[c][1] += sy;
      TB[c][2] += sz;
    }

    if (MFTL_TriTriProximity(TA, TB, ctx->tolerance)) ctx->callback(ctx->bvhA, ctx->prim_a[i], ctx->bvhB, ctx->prim_b[i], ctx->user_data);
  }

  ctx->n_batched = 0;
}

/**
 * @brief Queues one primitive pair, flushing when the batch is full.
 * @param ctx The narrow-phase context.
 * @param pa  A-side primitive index.
 * @param pb  B-side primitive index.
 */
static inline void batch_add(narrow_ctx *ctx, int pa, int pb)
{
  ctx->prim_a[ctx->n_batched] = pa;
  ctx->prim_b[ctx->n_batched] = pb;
  ctx->n_batched++;
  if (ctx->n_batched == MFTL_BVH_BATCH_SIZE) flush_batch(ctx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The traversal
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Traverses one periodic image of the pair. When both nodes are internal
 *        the one with the larger box volume is split (the boxes then converge
 *        faster); leaf pairs with a remote A-side primitive are skipped.
 *
 * @param ctx      The narrow-phase context, including the image shift.
 * @param stack    Scratch stack, reused across images.
 * @param shift_f  The same shift in `float`, for the box tests.
 */
static void traverse_image(narrow_ctx *ctx, pair_stack *stack, const float shift_f[3])
{
  const MFTL_BVHNode *const restrict nodesA = ctx->bvhA->nodes;
  const MFTL_BVHNode *const restrict nodesB = ctx->bvhB->nodes;
  const int *const restrict primsA = ctx->bvhA->prim_indices;
  const int *const restrict primsB = ctx->bvhB->prim_indices;
  const int n_local_a = ctx->bvhA->n_local_prims;

  stack->size = 0;
  stack_push(stack, ctx->bvhA->root_index, ctx->bvhB->root_index);

  while (stack->size > 0)
  {
    const node_pair top = stack->data[--stack->size];
    const MFTL_BVHNode *const na = &nodesA[top.a];
    const MFTL_BVHNode *const nb = &nodesB[top.b];

    if (!mftl_bvh_aabb_overlap(na, nb, shift_f)) continue;

    const bool leaf_a = MFTL_BVHNodeIsLeaf(na);
    const bool leaf_b = MFTL_BVHNodeIsLeaf(nb);

    if (leaf_a && leaf_b)
    {
      const int firstA = MFTL_BVHLeafFirst(na);
      const int countA = MFTL_BVHLeafCount(na);
      const int firstB = MFTL_BVHLeafFirst(nb);
      const int countB = MFTL_BVHLeafCount(nb);

      for (int i = 0; i < countA; ++i)
      {
        const int pa = primsA[firstA + i];

        /* Only the owner of the A-side primitive reports: skip remote A. */
        if (pa >= n_local_a) continue;

        for (int j = 0; j < countB; ++j) batch_add(ctx, pa, primsB[firstB + j]);
      }
      continue;
    }

    if (leaf_a)
    {
      const int l = MFTL_BVHLeftChild(nb);
      stack_push(stack, top.a, l);
      stack_push(stack, top.a, l + 1);
      continue;
    }

    if (leaf_b)
    {
      const int l = MFTL_BVHLeftChild(na);
      stack_push(stack, l, top.b);
      stack_push(stack, l + 1, top.b);
      continue;
    }

    if (mftl_bvh_node_volume(na) > mftl_bvh_node_volume(nb))
    {
      const int l = MFTL_BVHLeftChild(na);
      stack_push(stack, l, top.b);
      stack_push(stack, l + 1, top.b);
    }
    else
    {
      const int l = MFTL_BVHLeftChild(nb);
      stack_push(stack, top.a, l);
      stack_push(stack, top.a, l + 1);
    }
  }

  flush_batch(ctx);
}

void MFTL_IntersectObjects(const MFTL_Object *objA, MFTL_BVH *bvhA, const MFTL_Object *objB, MFTL_BVH *bvhB, IntersectionCallback callback, void *user_data)
{
  if (objA == MFTL_NULL || bvhA == MFTL_NULL || objB == MFTL_NULL || bvhB == MFTL_NULL || callback == MFTL_NULL) return;
  if (bvhA->nodes_used <= 0 || bvhB->nodes_used <= 0) return;
  MFTL_ASSERT(bvhA != bvhB && "MFTL_IntersectObjects: a BVH against itself is not a self-intersection query");

  narrow_ctx ctx;
  ctx.bvhA = bvhA;
  ctx.bvhB = bvhB;
  ctx.callback = callback;
  ctx.user_data = user_data;
  ctx.n_batched = 0;
  ctx.self = (objA == objB);
  ctx.filter = bvhA->params.pair_filter;

  /* MAX(tol_A, tol_B) <= tol_A + tol_B, the separation the inflated boxes
   * admit, so the broad phase cannot cull a contact the narrow phase accepts. */
  ctx.tolerance = MFTL_MAX(bvhA->tolerance, bvhB->tolerance);

  pair_stack stack;
  stack.capacity = MFTL_BVH_STACK_INIT;
  stack.size = 0;
  stack.data = (node_pair *) MFTLMallocOrDie((size_t) stack.capacity * sizeof(node_pair), "BVH traversal stack");

  /* Image offsets per axis: {0}, or {0, -L, +L} on a periodic axis; each image
   * is gated by one root-box test. */
  const MFTL_Domain *const domain = objA->domain;

  double off[3][3];
  int n_off[3];
  for (int k = 0; k < 3; ++k)
  {
    off[k][0] = 0.0;
    n_off[k] = 1;
    if (domain != MFTL_NULL && domain->is_periodic[k] && domain->domain_size[k] > 0.0)
    {
      off[k][1] = -domain->domain_size[k];
      off[k][2] = +domain->domain_size[k];
      n_off[k] = 3;
    }
  }

  const MFTL_BVHNode *const rootA = &bvhA->nodes[bvhA->root_index];
  const MFTL_BVHNode *const rootB = &bvhB->nodes[bvhB->root_index];

  for (int ix = 0; ix < n_off[0]; ++ix)
  {
    for (int iy = 0; iy < n_off[1]; ++iy)
    {
      for (int iz = 0; iz < n_off[2]; ++iz)
      {
        ctx.shift[0] = off[0][ix];
        ctx.shift[1] = off[1][iy];
        ctx.shift[2] = off[2][iz];

        const float shift_f[3] = {(float) ctx.shift[0], (float) ctx.shift[1], (float) ctx.shift[2]};

        if (!mftl_bvh_aabb_overlap(rootA, rootB, shift_f)) continue;

        traverse_image(&ctx, &stack, shift_f);
      }
    }
  }

  free(stack.data);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The ensemble pipeline
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLEnsembleIntersectObjects(MFTL_Ensemble *ens, const MFTL_BVHBuildParams *params, IntersectionCallback callback, void *user_data)
{
  if (ens == MFTL_NULL || callback == MFTL_NULL) return MFTL_ERROR;

  const bool self = (params != MFTL_NULL) && params->self_pairs;

  /* Guard on the GLOBAL body count: everything below is collective. */
  if (ens->n_bodies_global < (self ? 1 : 2)) return MFTL_SUCCESS;

  const int nb = ens->n_local_objects;

  /* One tree per body this rank holds. */
  MFTL_BVH **const bvhs = (MFTL_BVH **) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(MFTL_BVH *), "BVH list");
  for (int o = 0; o < nb; ++o) bvhs[o] = MFTL_GenerateLBVH(ens->local_objects[o], params);

  /* Global radii first (the envelope is a fraction of the body, and this rank's
   * share would make it depend on the process count), then one halo exchange
   * for the whole ensemble. */
  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(ens);
  MFTLSpatialHaloResolveRadii(ens, halo);

  double *const tol = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(double), "BVH pair envelopes");
  for (int o = 0; o < nb; ++o)
  {
    MFTL_BVHSetGlobalRadius(bvhs[o], MFTLSpatialHaloRadius(halo, ens->local_objects[o]->global_id));
    tol[o] = bvhs[o]->tolerance;
  }

  /* The envelopes travel with the extents, so the candidate pairs are derived
   * from them rather than enumerated. */
  halo->self_pairs = self;
  MFTLExtendSpatialHaloNearBodies(ens, tol, halo);
  free(tol);

  /* A tree for every directory entry. A body only PATCHED here gets a tree over
   * a shared empty object: every primitive resolves through the patch. */
  const int ne = halo->num_objects;
  MFTL_BVH **const by_entry = (MFTL_BVH **) MFTLCallocOrDie((size_t) MFTL_MAX(ne, 1), sizeof(MFTL_BVH *), "BVH directory");
  const MFTL_Object **const obj_by_entry = (const MFTL_Object **) MFTLCallocOrDie((size_t) MFTL_MAX(ne, 1), sizeof(MFTL_Object *), "BVH object directory");

  MFTL_Object *empty = MFTLObjectCreate(MFTL_ERROR);
  empty->domain = &ens->domain;

  for (int e = 0; e < ne; ++e)
  {
    const int gid = halo->body_gid[e];
    const int slot = MFTLEnsembleFindLocalSlot(ens, gid);

    if (slot != MFTL_ERROR)
    {
      by_entry[e] = bvhs[slot];
      obj_by_entry[e] = ens->local_objects[slot];
    }
    else if (halo->patch[e] != MFTL_NULL)
    {
      by_entry[e] = MFTL_GenerateLBVH(empty, params);
      obj_by_entry[e] = empty;
    }
    else
    {
      continue;
    }

    /* Attach only a real patch (attaching rebuilds the tree). */
    if (halo->patch[e] != MFTL_NULL) MFTL_BVHSetHalo(by_entry[e], halo->patch[e]);

    /* Radius last, so every rank that can report the pair uses the same
     * envelope (test_tolerance_is_independent_of_the_partition). */
    MFTL_BVHSetGlobalRadius(by_entry[e], MFTLSpatialHaloRadius(halo, gid));
  }

  /* The candidate pairs (body_a < body_b, A held here); the traversal's A-side
   * filter decides which rank reports. */
  for (int i = 0; i < halo->n_pairs; ++i)
  {
    const int ea = MFTLSpatialHaloFindBody(halo, halo->pair[i].body_a);
    const int eb = MFTLSpatialHaloFindBody(halo, halo->pair[i].body_b);
    if (ea == MFTL_ERROR || eb == MFTL_ERROR) continue;
    if (by_entry[ea] == MFTL_NULL || by_entry[eb] == MFTL_NULL) continue;

    if (ea != eb)
    {
      MFTL_IntersectObjects(obj_by_entry[ea], by_entry[ea], obj_by_entry[eb], by_entry[eb], callback, user_data);
      continue;
    }

    /* A body against itself: B is its tree WITH the patch; A is a second,
     * patch-free tree over the same owned triangles (a patch primitive may
     * never report) with a ZERO envelope, so the broad phase admits tol_B
     * rather than tol_A + tol_B -- 129 candidate pairs per triangle otherwise,
     * measured on a 17k-triangle tube at an envelope of 1.5 edges. */
    const MFTL_Object *const obj = obj_by_entry[ea];
    MFTL_BVHBuildParams tight = (params != MFTL_NULL) ? *params : MFTL_BVHDefaultParams();
    tight.proximity_factor = 0.0;
    tight.proximity_floor = 0.0;
    MFTL_BVH *const a_side = MFTL_GenerateLBVH(obj, &tight);
    MFTL_IntersectObjects(obj, a_side, obj, by_entry[eb], callback, user_data);
    MFTL_DestroyBVH(a_side);
  }

  for (int e = 0; e < ne; ++e)
  {
    if (obj_by_entry[e] == empty) MFTL_DestroyBVH(by_entry[e]);
  }
  for (int o = 0; o < nb; ++o) MFTL_DestroyBVH(bvhs[o]);

  MFTLObjectDestroy(&empty);
  free(obj_by_entry);
  free(by_entry);
  free(bvhs);
  MFTLSpatialHaloDestroy(&halo);

  return MFTL_SUCCESS;
}
