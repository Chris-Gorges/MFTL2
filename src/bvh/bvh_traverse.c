/**
 * @file bvh_traverse.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_bvh
 *
 * @brief Dual-tree traversal: an explicit stack of node pairs, and a batched
 *        leaf-leaf narrow phase.
 *
 * ## WHY THE RECURSION WENT
 *
 * The previous version descended with a four-argument recursive call. Replacing
 * it with an explicit stack buys three things, and only the third one is really
 * about speed:
 *
 *   1. THE HOT STATE STAYS IN REGISTERS. Both objects, both node arrays, the
 *      callback and the image shift are loop-invariant, and a recursive call has
 *      to re-establish all of them per node pair. The loop form loads them once.
 *
 *   2. THE "WHAT NEXT" DATA BECOMES CONTIGUOUS. A pair stack is two ints per
 *      entry in one array. A call stack is a frame per entry, interleaved with
 *      spill slots and a return address.
 *
 *   3. LEAF PAIRS CAN BE BATCHED, and this is the one that matters going
 *      forward. A leaf-leaf hit does not run the narrow phase where it is found;
 *      it appends to a fixed-size array of (primA, primB) and the array is
 *      flushed when full. The traversal therefore hands the narrow phase a
 *      CONTIGUOUS RUN OF INDEPENDENT TESTS, which is the shape a vectorised
 *      triangle-triangle kernel needs. Making that kernel SIMD is then a change
 *      to flush_batch() alone -- the traversal above it does not move.
 *
 * The batch is deliberately NOT vectorised yet. MFTL_TriTriIntersect is a
 * decision tree over exact predicates with early-out at every level, so the
 * useful SIMD form is a gather of the first orientation test across a batch, not
 * a naive widening of the whole function; that is a measurement exercise, and
 * the batch exists so it can be done without touching anything else.
 *
 * ## PERIODIC IMAGES
 *
 * Two bodies on opposite faces of a periodic domain are neighbours, and no
 * amount of per-triangle unwrapping can see that -- each body is internally
 * consistent and they are genuinely a domain-length apart in stored
 * coordinates. So the pair is traversed once per periodic IMAGE of B: the
 * offsets {-L, 0, +L} on each periodic axis, up to 27 of them, each admitted or
 * rejected by a single root-AABB test before any descent happens.
 *
 * A non-periodic domain generates exactly one image and pays one root test it
 * was going to pay anyway. A periodic domain typically admits one image and
 * rejects the other 26 in 26 compares -- the bodies would have to span half the
 * domain for two images to survive.
 */

#include "mftl_bvh.h"

#include <stdlib.h>

#include "bvh_internal.h"
#include "mftl_communication.h"
#include "mftl_constants.h"
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
 * @brief Initial stack depth.
 *
 * A dual traversal pushes at most one extra pair per descent, and the depth of
 * each tree is bounded by the Morton key length (95 bits, see prefix_len in
 * bvh_build.c), so 256 covers every non-pathological case without a single
 * growth step. It grows anyway rather than asserting, because "non-pathological"
 * is not something a library should require of its caller.
 */
#define MFTL_BVH_STACK_INIT 256

/** @brief Leaf pairs collected before the narrow phase runs. */
#define MFTL_BVH_BATCH_SIZE 64

/**
 * @brief Explicit traversal stack, replacing the recursion of the original.
 *
 * A dual tree traversal is not tail-recursive in either branch, so the depth is
 * the sum of two tree depths and a pathological pair could exhaust the machine
 * stack with no diagnostic. An explicit stack makes the bound a `malloc` failure
 * instead, and lets the batch below span node pairs.
 */
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
  const MFTL_Object *objA;  ///< Object the A-side tree was built over.
  const MFTL_Object *objB;  ///< Object the B-side tree was built over.
  MFTL_BVH *bvhA;  ///< A-side hierarchy.
  MFTL_BVH *bvhB;  ///< B-side hierarchy.

  double shift[3];  ///< Periodic image offset applied to every B coordinate.
  double tolerance;  ///< The gap at which contact is declared, for this pair.

  IntersectionCallback callback;  ///< Invoked once per surviving pair.
  void *user_data;  ///< Passed to @c callback untouched.

  int prim_a[MFTL_BVH_BATCH_SIZE];  ///< A-side primitive of each pending pair.
  int prim_b[MFTL_BVH_BATCH_SIZE];  ///< B-side primitive of each pending pair.
  int n_batched;  ///< Pairs pending in the batch.
} narrow_ctx;

/**
 * @brief Runs the exact test on every batched pair and reports the hits.
 *
 * THE SIMD SEAM. Everything above this function produces a contiguous run of
 * independent (primA, primB) tests; everything below it is scalar double
 * precision. A vector narrow phase replaces the body of this loop and nothing
 * else in the module changes.
 *
 * The B-side coordinates get the periodic shift applied HERE rather than in
 * MFTL_BVHPrimVerts, because the shift belongs to the pair being traversed and
 * not to the primitive -- the same primitive is tested against several images of
 * the same object when the bodies are large.
 *
 * @param ctx The narrow-phase context. Its batch is emptied.
 */
static void flush_batch(narrow_ctx *ctx)
{
  const double sx = ctx->shift[0], sy = ctx->shift[1], sz = ctx->shift[2];

  for (int i = 0; i < ctx->n_batched; ++i)
  {
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

    if (MFTL_TriTriProximity(TA, TB, ctx->tolerance)) ctx->callback(ctx->objA, ctx->prim_a[i], ctx->objB, ctx->prim_b[i], ctx->user_data);
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
 * @brief Traverses one periodic image of the pair, reporting every contact.
 *
 * The descent rule is the previous version's and is kept: when both nodes are
 * internal, split the one whose box is LARGER. Splitting the smaller one makes
 * the pair of boxes converge more slowly, so the traversal visits more pairs for
 * the same set of leaf tests.
 *
 * @param ctx      The narrow-phase context, including the image shift.
 * @param stack    Scratch stack, reused across images.
 * @param shift_f  The same shift in `float`, for the AABB tests.
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

        /* OWNERSHIP OF THE PAIR. Only the rank that owns the A-side primitive
         * reports the contact, so a pair whose A side arrived as a spatial ghost
         * is skipped -- the rank that owns it is testing the mirror image of
         * this same pair and will report it. Without this, every contact that
         * straddles a rank boundary is reported twice and every contact force
         * built from it is doubled. The test is one compare against a value
         * already in a register, hoisted out of the inner loop. */
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
  ctx.objA = objA;
  ctx.objB = objB;
  ctx.bvhA = bvhA;
  ctx.bvhB = bvhB;
  ctx.callback = callback;
  ctx.user_data = user_data;
  ctx.n_batched = 0;

  /* ── The pair's tolerance ────────────────────────────────────────────────
   *
   * The LARGER of the two envelopes. Each BVH's leaves are already inflated by
   * its own tolerance, so the broad phase admits any pair within
   * tol_A + tol_B -- taking the maximum keeps the narrow-phase threshold inside
   * what the broad phase can deliver, which is the direction that cannot lose a
   * contact. Taking the minimum would let the smaller body's envelope silently
   * govern a contact with a body a thousand times its size, which is exactly the
   * two-length-scale failure the fixed 1e-6 produced. */
  ctx.tolerance = MFTL_MAX(bvhA->tolerance, bvhB->tolerance);

  pair_stack stack;
  stack.capacity = MFTL_BVH_STACK_INIT;
  stack.size = 0;
  stack.data = (node_pair *) MFTLMallocOrDie((size_t) stack.capacity * sizeof(node_pair), "BVH traversal stack");

  /* ── Enumerate the periodic images of B ──────────────────────────────────
   *
   * Offsets are built per axis so a non-periodic domain contributes exactly one
   * value and the product collapses to a single image. The root test inside the
   * loop is what makes trying 27 of them cheap: it is the same six compares the
   * traversal would have done first anyway. */
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
  if (ens == MFTL_NULL || ens->objects == MFTL_NULL || callback == MFTL_NULL) return MFTL_ERROR;

  const int n_obj = ens->num_objects;
  if (n_obj < 2) return MFTL_SUCCESS;  // one body cannot collide with another

  MFTL_BVH **const bvhs = (MFTL_BVH **) MFTLCallocOrDie((size_t) n_obj, sizeof(MFTL_BVH *), "BVH list");
  for (int i = 0; i < n_obj; ++i) bvhs[i] = MFTL_GenerateLBVH(ens->objects[i], params);

  /* ── The body scales, then one halo exchange for every pair ──────────────
   *
   * Two collectives for the whole ensemble, neither of which grows with the
   * number of bodies. The version this replaces ran a full neighbour alltoall
   * PER OBJECT PAIR, so 500 droplets meant 124 750 rounds of collectives per
   * timestep before a single triangle moved.
   *
   * The radii come first because the margin is a fraction of the body and the
   * body's size is only knowable collectively -- see MFTL_BVH::radius_hint for
   * why using this rank's share instead would make detection depend on the
   * process count. */
  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(ens);
  MFTLSpatialHaloResolveRadii(ens, halo);

  for (int i = 0; i < n_obj; ++i) MFTL_BVHSetGlobalRadius(bvhs[i], halo->global_radius[i]);

  {
    int n_req = 0;
    MFTL_SpatialHaloRequest *const reqs =
        (MFTL_SpatialHaloRequest *) MFTLMallocOrDie((size_t) (n_obj * (n_obj - 1) / 2) * sizeof(MFTL_SpatialHaloRequest) + 1, "BVH halo requests");

    /* EVERY pair goes in, unconditionally. The list has to be identical on every
     * rank -- MFTLExtendSpatialHalo is collective and derives its sequence of
     * collectives from it -- so it may depend only on the object list, which all
     * ranks hold in the same order, and never on what this rank happens to hold
     * geometrically. That is exactly why MFTL_GenerateLBVH returns an empty
     * hierarchy for an empty partition instead of MFTL_NULL: a NULL here would
     * make the list rank-dependent and hang the ranks that still had work.
     *
     * The margin is the LARGER of the two bodies' envelopes, matching what
     * MFTL_IntersectObjects will test the pair against -- fetching less than the
     * narrow phase is willing to accept would lose contacts at the rank boundary
     * and nowhere else, which is the worst possible place for a bug to live. */
    for (int i = 0; i < n_obj; ++i)
    {
      for (int j = i + 1; j < n_obj; ++j)
      {
        reqs[n_req].body_a = i;
        reqs[n_req].body_b = j;
        reqs[n_req].margin = MFTL_MAX(bvhs[i]->tolerance, bvhs[j]->tolerance);
        n_req++;
      }
    }

    MFTLExtendSpatialHalo(ens, reqs, n_req, halo);
    free(reqs);
  }

  /* Only the B side of a pair ever receives a patch, and body j is B for every
   * pair (i, j) with i < j -- so attaching every body's patch is correct and the
   * A-side filter in the traversal keeps the halo primitives from being reported
   * as if this rank owned them. */
  for (int i = 0; i < n_obj; ++i) MFTL_BVHSetHalo(bvhs[i], MFTLSpatialHaloPatch(halo, i));

  for (int i = 0; i < n_obj; ++i)
  {
    for (int j = i + 1; j < n_obj; ++j) MFTL_IntersectObjects(ens->objects[i], bvhs[i], ens->objects[j], bvhs[j], callback, user_data);
  }

  for (int i = 0; i < n_obj; ++i) MFTL_DestroyBVH(bvhs[i]);
  free(bvhs);
  MFTLSpatialHaloDestroy(&halo);

  return MFTL_SUCCESS;
}
