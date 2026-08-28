/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for the object pools -- the AoS/SoA synchronisation guarantee.
 *
 * This file is the regression test for the single guarantee the pool layout
 * exists to provide. If the AoS entity pools, the geometry arrays and the derived
 * arrays each had their own capacity and could be grown independently, a growth
 * site that forgot one of the three would produce a heap overflow rather than a
 * wrong number.
 *
 * The tests below therefore do not just check that the counts look right. They
 * WRITE THE LAST ADDRESSABLE ELEMENT OF EVERY ARRAY at the reported capacity, so
 * that any array which was not grown with the rest is a hard failure under ASan
 * rather than a quiet corruption. Run this suite in the Debug (sanitiser) build
 * for that half of its value.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

/* ─────────────────────────────────────────────────────────────────────────────
 * Helpers
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Touches the last element of every array of both pools.
 *
 * The point is the ADDRESS, not the value: if any single array in a pool were
 * sized for fewer entities than `capacity` says, this writes past its end.
 */
static void touch_every_array_at_capacity(MFTL_Object *obj)
{
  const int vc = obj->verts.capacity;
  const int tc = obj->tris.capacity;

  if (vc > 0)
  {
    obj->verts.rec[vc - 1].global_id = 1;
    obj->verts.pos[3 * vc - 1] = 1.0;
    obj->verts.normal[3 * vc - 1] = 1.0;
    obj->verts.area[vc - 1] = 1.0;
    obj->verts.mean_curv[vc - 1] = 1.0;
    obj->verts.gauss_curv[vc - 1] = 1.0;
  }

  if (tc > 0)
  {
    obj->tris.rec[tc - 1].global_id = 1;
    /* Three half-edges per triangle, in the same block. If the half-edge array
     * were sized for `capacity` instead of 3*capacity -- the exact shape of the
     * bug the previous layout allowed -- this is where it shows. */
    obj->tris.he[3 * tc - 1].global_id = 1;
    obj->tris.area[tc - 1] = 1.0;
    obj->tris.normal[3 * tc - 1] = 1.0;
    obj->tris.center[3 * tc - 1] = 1.0;
    obj->tris.verts[3 * tc - 1] = 1;
    obj->tris.neighbors[3 * tc - 1] = 1;
    obj->tris.anchor[3 * tc - 1] = 1.0;
    obj->tris.edge[6 * tc - 1] = 1.0;
  }
}

/** @brief Writes a deterministic pattern into every array of the live prefix. */
static void fill_pattern(MFTL_Object *obj, int seed)
{
  for (int v = 0; v < obj->verts.n; ++v)
  {
    obj->verts.rec[v].global_id = seed + v;
    obj->verts.rec[v].out_he_idx = seed + 2 * v;
    obj->verts.rec[v].owner_proc = seed;
    obj->verts.rec[v].flags = v & 1;
    for (int k = 0; k < 3; ++k)
    {
      obj->verts.pos[3 * v + k] = (double) (seed + 3 * v + k);
      obj->verts.normal[3 * v + k] = (double) (seed - 3 * v - k);
    }
    obj->verts.area[v] = (double) (seed * 2 + v);
    obj->verts.mean_curv[v] = (double) (seed * 3 + v);
    obj->verts.gauss_curv[v] = (double) (seed * 4 + v);
  }

  for (int t = 0; t < obj->tris.n; ++t)
  {
    obj->tris.rec[t].global_id = seed + t;
    obj->tris.rec[t].owner_proc = seed;
    obj->tris.area[t] = (double) (seed + t);
    for (int k = 0; k < 3; ++k)
    {
      obj->tris.normal[3 * t + k] = (double) (seed + 3 * t + k);
      obj->tris.center[3 * t + k] = (double) (seed - 3 * t - k);
      obj->tris.verts[3 * t + k] = seed + 3 * t + k;
      obj->tris.neighbors[3 * t + k] = seed - 3 * t - k;
      obj->tris.anchor[3 * t + k] = (double) (seed * 5 + 3 * t + k);
    }
    for (int k = 0; k < 6; ++k) obj->tris.edge[6 * t + k] = (double) (seed * 6 + 6 * t + k);
    for (int k = 0; k < 3; ++k)
    {
      obj->tris.he[3 * t + k].global_id = seed + 3 * t + k;
      obj->tris.he[3 * t + k].vertex_idx = seed + k;
      obj->tris.he[3 * t + k].twin_idx = seed - k;
      obj->tris.he[3 * t + k].flags = k;
    }
  }
}

/** @brief Verifies the pattern fill_pattern() wrote is still intact. */
static int check_pattern(const MFTL_Object *obj, int seed, int nv, int nt)
{
  for (int v = 0; v < nv; ++v)
  {
    if (obj->verts.rec[v].global_id != seed + v) return 0;
    if (obj->verts.rec[v].out_he_idx != seed + 2 * v) return 0;
    if (obj->verts.rec[v].flags != (v & 1)) return 0;
    for (int k = 0; k < 3; ++k)
    {
      if (obj->verts.pos[3 * v + k] != (double) (seed + 3 * v + k)) return 0;
      if (obj->verts.normal[3 * v + k] != (double) (seed - 3 * v - k)) return 0;
    }
    if (obj->verts.area[v] != (double) (seed * 2 + v)) return 0;
    if (obj->verts.mean_curv[v] != (double) (seed * 3 + v)) return 0;
    if (obj->verts.gauss_curv[v] != (double) (seed * 4 + v)) return 0;
  }

  for (int t = 0; t < nt; ++t)
  {
    if (obj->tris.rec[t].global_id != seed + t) return 0;
    if (obj->tris.area[t] != (double) (seed + t)) return 0;
    for (int k = 0; k < 3; ++k)
    {
      if (obj->tris.normal[3 * t + k] != (double) (seed + 3 * t + k)) return 0;
      if (obj->tris.center[3 * t + k] != (double) (seed - 3 * t - k)) return 0;
      if (obj->tris.verts[3 * t + k] != seed + 3 * t + k) return 0;
      if (obj->tris.neighbors[3 * t + k] != seed - 3 * t - k) return 0;
      if (obj->tris.anchor[3 * t + k] != (double) (seed * 5 + 3 * t + k)) return 0;
      if (obj->tris.he[3 * t + k].global_id != seed + 3 * t + k) return 0;
      if (obj->tris.he[3 * t + k].vertex_idx != seed + k) return 0;
      if (obj->tris.he[3 * t + k].twin_idx != seed - k) return 0;
      if (obj->tris.he[3 * t + k].flags != k) return 0;
    }
    for (int k = 0; k < 6; ++k)
    {
      if (obj->tris.edge[6 * t + k] != (double) (seed * 6 + 6 * t + k)) return 0;
    }
  }
  return 1;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Lifecycle
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_fresh_object_is_empty_and_consistent(void)
{
  MFTL_Object *obj = MFTLObjectCreate(42);
  TEST_ASSERT(obj != MFTL_NULL);

  TEST_ASSERT_EQUAL_INT(42, obj->global_id);
  TEST_ASSERT(obj->domain == MFTL_NULL);

  TEST_ASSERT_EQUAL_INT(0, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(0, obj->verts.n_local);
  TEST_ASSERT_EQUAL_INT(0, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(0, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(0, obj->tris.capacity);

  TEST_ASSERT(obj->verts.block == MFTL_NULL);
  TEST_ASSERT(obj->tris.block == MFTL_NULL);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  // Nothing derived yet.
  TEST_ASSERT(!MFTLObjectHasTopology(obj));
  TEST_ASSERT(!MFTLObjectHasFrames(obj));

  MFTLObjectDestroy(&obj);
  TEST_ASSERT(obj == MFTL_NULL);
  return 0;
}

static int test_destroy_is_null_safe_and_idempotent(void)
{
  MFTLObjectDestroy(MFTL_NULL);

  MFTL_Object *obj = MFTL_NULL;
  MFTLObjectDestroy(&obj);  // already NULL

  obj = MFTLObjectCreate(1);
  MFTLObjectReserve(obj, 10, 20);
  MFTLObjectDestroy(&obj);
  MFTLObjectDestroy(&obj);  // second time is a no-op, not a double free
  TEST_ASSERT(obj == MFTL_NULL);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The synchronisation guarantee
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_reserve_sizes_every_array_of_both_pools(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 37, 61);
  TEST_ASSERT(obj->verts.capacity >= 37);
  TEST_ASSERT(obj->tris.capacity >= 61);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  /* Every array, right up to the capacity the pool reports. Under ASan this is
   * the assertion that no array was left behind by the growth. */
  touch_every_array_at_capacity(obj);

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_capacity_is_shared_across_many_growth_steps(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  /* Grow the two pools by DIFFERENT amounts, in an interleaved order, many
   * times. The previous layout is exactly what this sequence broke: a growth of
   * the triangle pool that did not also grow the triangle geometry arrays. */
  const int steps[8][2] = {{1, 1}, {5, 3}, {5, 400}, {700, 400}, {700, 401}, {2000, 401}, {2000, 5000}, {12345, 5000}};

  for (int s = 0; s < 8; ++s)
  {
    MFTLObjectReserve(obj, steps[s][0], steps[s][1]);
    TEST_ASSERT(obj->verts.capacity >= steps[s][0]);
    TEST_ASSERT(obj->tris.capacity >= steps[s][1]);
    TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));
    touch_every_array_at_capacity(obj);
  }

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_half_edge_storage_is_exactly_three_per_triangle(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLObjectSetCounts(obj, 4, 4, 100, 100);

  // The count relation the whole implicit-indexing design depends on.
  TEST_ASSERT_EQUAL_INT(300, MFTLNumHalfEdges(obj));
  TEST_ASSERT_EQUAL_INT(300, MFTLNumLocalHalfEdges(obj));

  // ...and the storage behind it, addressable to 3*capacity - 1.
  for (int h = 0; h < 3 * obj->tris.capacity; ++h) obj->tris.he[h].global_id = h;
  for (int h = 0; h < 3 * obj->tris.capacity; ++h) TEST_ASSERT_EQUAL_INT(h, obj->tris.he[h].global_id);

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_growth_preserves_every_array(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectSetCounts(obj, 50, 50, 40, 40);
  fill_pattern(obj, 7);
  TEST_ASSERT(check_pattern(obj, 7, 50, 40));

  /* Force a reallocation of both blocks. Contents must survive: the previous
   * version deliberately discarded geometry here, which silently turned any
   * value the caller had just computed into 0.0. */
  MFTLObjectReserve(obj, 100000, 90000);
  TEST_ASSERT(obj->verts.capacity >= 100000);
  TEST_ASSERT(obj->tris.capacity >= 90000);
  TEST_ASSERT(check_pattern(obj, 7, 50, 40));
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_set_counts_reserves_before_it_assigns(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  /* Counts far beyond any capacity ever requested. The postcondition
   * n <= capacity must hold anyway -- that is the property that makes "counts
   * and capacity disagree" unreachable rather than merely unlikely. */
  MFTLObjectSetCounts(obj, 5000, 4000, 6000, 3000);

  TEST_ASSERT_EQUAL_INT(5000, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(4000, obj->verts.n_local);
  TEST_ASSERT_EQUAL_INT(6000, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(3000, obj->tris.n_local);
  TEST_ASSERT(obj->verts.n <= obj->verts.capacity);
  TEST_ASSERT(obj->tris.n <= obj->tris.capacity);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  touch_every_array_at_capacity(obj);

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_set_counts_invalidates_the_derived_arrays(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLObjectSetCounts(obj, 3, 3, 1, 1);

  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(MFTLObjectHasTopology(obj));
  TEST_ASSERT(MFTLObjectHasFrames(obj));

  // A different entity count means the derived arrays describe a different mesh.
  MFTLObjectSetCounts(obj, 6, 6, 2, 2);
  TEST_ASSERT(!MFTLObjectHasTopology(obj));
  TEST_ASSERT(!MFTLObjectHasFrames(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_reserve_never_shrinks(void)
{
  /* MFTLObjectReserve() is grow-only, and that is a contract other callers rely
   * on -- src/communication/redistribute.c pre-grows and then lets the reserve
   * inside MFTLObjectSetCounts() early-return, which only works while this is
   * true. Shrinking lives in MFTLObjectShrinkToFit() and nowhere else. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 5000, 5000);
  const int vc = obj->verts.capacity;
  const int tc = obj->tris.capacity;
  const void *vb = obj->verts.block;
  const void *tb = obj->tris.block;

  // A smaller request must not reallocate: a mesh of stable size should not
  // re-fault its pages every timestep.
  MFTLObjectReserve(obj, 10, 10);
  TEST_ASSERT_EQUAL_INT(vc, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(tc, obj->tris.capacity);
  TEST_ASSERT(obj->verts.block == vb);
  TEST_ASSERT(obj->tris.block == tb);

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_clear_keeps_the_allocation(void)
{
  /* MFTLObjectClear() empties the mesh and keeps the block, so refilling does
   * not re-allocate. Giving it back is the other half, and it is a separate,
   * explicit call: MFTLObjectShrinkToFit() or MFTLObjectReleasePools(). */
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLObjectSetCounts(obj, 100, 100, 80, 80);
  MFTLObjectBuildDerived(obj);

  const int vc = obj->verts.capacity;
  const int tc = obj->tris.capacity;
  const void *vb = obj->verts.block;

  MFTLObjectClear(obj);

  TEST_ASSERT_EQUAL_INT(0, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(0, obj->tris.n);
  TEST_ASSERT_EQUAL_INT(0, obj->next_vertex_gid);
  TEST_ASSERT_EQUAL_INT(0, obj->next_triangle_gid);
  TEST_ASSERT_EQUAL_INT(vc, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(tc, obj->tris.capacity);
  TEST_ASSERT(obj->verts.block == vb);
  TEST_ASSERT(!MFTLObjectHasTopology(obj));
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_reserve_additional_is_geometric(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  int reallocs = 0;
  const void *last_block = MFTL_NULL;

  for (int i = 0; i < 20000; ++i)
  {
    MFTLObjectReserveAdditional(obj, 1, 1);
    if (obj->verts.block != last_block)
    {
      last_block = obj->verts.block;
      reallocs++;
    }
    MFTLObjectSetCounts(obj, obj->verts.n + 1, obj->verts.n + 1, obj->tris.n + 1, obj->tris.n + 1);
  }

  TEST_ASSERT_EQUAL_INT(20000, obj->verts.n);
  TEST_ASSERT(reallocs <= 12);  // O(log n), not O(n)
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Shrinking -- the only way a pool ever gets smaller
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_shrink_to_fit_recovers_the_block(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 100000, 90000);
  MFTLObjectSetCounts(obj, 300, 300, 200, 200);

  const void *vb = obj->verts.block;
  const void *tb = obj->tris.block;

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));

  // MFTLGrowCapacity(0, n): 128 doubled until it covers n.
  TEST_ASSERT_EQUAL_INT(512, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(256, obj->tris.capacity);
  TEST_ASSERT(obj->verts.block != vb);
  TEST_ASSERT(obj->tris.block != tb);

  // Counts are untouched -- this moved storage, not entities.
  TEST_ASSERT_EQUAL_INT(300, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(200, obj->tris.n);

  touch_every_array_at_capacity(obj);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_preserves_every_entity(void)
{
  /* The growth direction is covered by test_growth_preserves_every_array. This
   * is the same statement for the shrink, and it is the test that catches a
   * wrong copy count or a sub-array offset mis-derived at the new capacity. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 40000, 40000);
  MFTLObjectSetCounts(obj, 700, 700, 600, 600);
  fill_pattern(obj, 17);

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT(check_pattern(obj, 17, 700, 600));

  touch_every_array_at_capacity(obj);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_never_drops_entities(void)
{
  /* A slack of 1 asks for the most aggressive shrink the API permits. The
   * target is still floored at the live count: this function moves storage and
   * does not decide what to discard. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 80000, 80000);
  MFTLObjectSetCounts(obj, 5000, 2500, 5000, 2500);

  (void) MFTLObjectShrinkToFit(obj, 1.0);

  TEST_ASSERT(obj->verts.capacity >= obj->verts.n);
  TEST_ASSERT(obj->tris.capacity >= obj->tris.n);
  TEST_ASSERT_EQUAL_INT(5000, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(2500, obj->verts.n_local);
  TEST_ASSERT_EQUAL_INT(5000, obj->tris.n);

  touch_every_array_at_capacity(obj);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_is_idempotent(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 60000, 60000);
  MFTLObjectSetCounts(obj, 900, 900, 900, 900);

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  const int vc = obj->verts.capacity;
  const int tc = obj->tris.capacity;
  const void *vb = obj->verts.block;
  const void *tb = obj->tris.block;

  /* The BLOCK POINTER is the real assertion. A second call that reallocated to
   * the same capacity would satisfy every count check while costing a full
   * memcpy of both pools on every timestep it ran. */
  TEST_ASSERT(!MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT(!MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));

  TEST_ASSERT_EQUAL_INT(vc, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(tc, obj->tris.capacity);
  TEST_ASSERT(obj->verts.block == vb);
  TEST_ASSERT(obj->tris.block == tb);

  /* And the same at a count below MFTL_INIT_CAPACITY, where the gate stays
   * permanently true (capacity 128 against n == 4 satisfies 128 > 16 forever)
   * and only the guard stops the reallocation. */
  MFTLObjectSetCounts(obj, 4, 4, 4, 4);
  (void) MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT);
  const void *vb2 = obj->verts.block;
  TEST_ASSERT(!MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT(obj->verts.block == vb2);
  TEST_ASSERT_EQUAL_INT(MFTL_INIT_CAPACITY, obj->verts.capacity);

  touch_every_array_at_capacity(obj);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_respects_the_slack_gate(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 4096, 4096);
  MFTLObjectSetCounts(obj, 2000, 2000, 2000, 2000);

  const void *vb = obj->verts.block;
  const void *tb = obj->tris.block;

  // 4096 > 4 * 2000 is false: wasting 2x is not worth a reallocation.
  TEST_ASSERT(!MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT_EQUAL_INT(4096, obj->verts.capacity);
  TEST_ASSERT(obj->verts.block == vb);
  TEST_ASSERT(obj->tris.block == tb);

  // 4096 > 4 * 500 is true.
  MFTLObjectSetCounts(obj, 500, 500, 500, 500);
  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT_EQUAL_INT(512, obj->verts.capacity);

  /* And the case rank 0 is actually in for the bodies it keeps whole: capacity
   * equals the count, nothing is wasted, nothing must move. */
  MFTL_Object *tight = MFTLObjectCreate(2);
  MFTLObjectSetCounts(tight, 2562, 2562, 5120, 5120);
  TEST_ASSERT_EQUAL_INT(2562, tight->verts.capacity);
  TEST_ASSERT(!MFTLObjectShrinkToFit(tight, MFTL_SHRINK_SLACK_DEFAULT));
  TEST_ASSERT_EQUAL_INT(2562, tight->verts.capacity);
  TEST_ASSERT_EQUAL_INT(5120, tight->tris.capacity);

  touch_every_array_at_capacity(obj);
  touch_every_array_at_capacity(tight);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(tight));

  MFTLObjectDestroy(&tight);
  MFTLObjectDestroy(&obj);
  return 0;
}

/**
 * @brief Replays a redistribution's exact resize sequence for @p steps steps.
 *
 * `n` alternates between @p lo and @p hi -- the few-entity jitter a body under
 * advection shows every timestep -- and each step does what rebuild_object()
 * does: pre-grow geometrically, set the counts, then offer the pool back.
 *
 * @return Number of times the vertex block was reallocated.
 */
static int count_reallocs_under_jitter(double slack, int lo, int hi, int steps)
{
  MFTL_Object *obj = MFTLObjectCreate(1);

  /* Start ON the doubling ladder, because that is the state the shrink itself
   * produces: its target is MFTLGrowCapacity(0, n). MFTLObjectSetCounts() alone
   * would leave capacity at exactly `lo`, half a rung low, where the gate
   * happens to miss by one and nothing thrashes -- so a harness built that way
   * would report both slacks as safe and prove nothing. */
  MFTLObjectReserve(obj, MFTLGrowCapacity(0, lo), MFTLGrowCapacity(0, lo));
  MFTLObjectSetCounts(obj, lo, lo, lo, lo);

  const void *last = obj->verts.block;
  int reallocs = 0;

  for (int i = 0; i < steps; ++i)
  {
    const int n = (i & 1) ? hi : lo;

    // src/communication/redistribute.c, rebuild_object step 4.
    if (n > obj->verts.capacity) MFTLObjectReserve(obj, MFTLGrowCapacity(obj->verts.capacity, n), MFTLGrowCapacity(obj->tris.capacity, n));
    MFTLObjectSetCounts(obj, n, n, n, n);
    (void) MFTLObjectShrinkToFit(obj, slack);

    if (obj->verts.block != last)
    {
      last = obj->verts.block;
      reallocs++;
    }
  }

  MFTLObjectDestroy(&obj);
  return reallocs;
}

static int test_shrink_to_fit_does_not_thrash_against_the_grow_ladder(void)
{
  /* WHY THE DEFAULT SLACK IS 4 AND NOT 2, as an executable claim.
   *
   * Both resize operators leave capacity in [n, 2n), so a gate of
   * `capacity > 2n` is satisfied by ANY decrease: a body dithering by two
   * entities around 1024 grows to 2048, shrinks to 1024, grows to 2048 ...
   * reallocating both pools every single timestep. At 4 the count has to halve
   * before anything moves. */
  const int lax = count_reallocs_under_jitter(2.0, 1023, 1025, 200);
  const int sane = count_reallocs_under_jitter(MFTL_SHRINK_SLACK_DEFAULT, 1023, 1025, 200);

  TEST_ASSERT(lax > 100);
  TEST_ASSERT(sane <= 2);

  return 0;
}

static int test_shrink_to_fit_at_zero_releases_the_block(void)
{
  /* At n == 0 the target is MFTLGrowCapacity(0, 0) == 0, so "give the block
   * back entirely" needs no special case -- it is the same arithmetic. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 20000, 20000);
  MFTLObjectSetCounts(obj, 0, 0, 0, 0);

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));

  TEST_ASSERT_EQUAL_INT(0, obj->verts.capacity);
  TEST_ASSERT_EQUAL_INT(0, obj->tris.capacity);
  TEST_ASSERT(obj->verts.block == MFTL_NULL);
  TEST_ASSERT(obj->tris.block == MFTL_NULL);
  TEST_ASSERT(obj->verts.rec == MFTL_NULL);
  TEST_ASSERT(obj->tris.he == MFTL_NULL);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  // A released pool must be refillable, exactly like a fresh one.
  MFTLObjectSetCounts(obj, 50, 50, 40, 40);
  fill_pattern(obj, 3);
  TEST_ASSERT(check_pattern(obj, 3, 50, 40));
  touch_every_array_at_capacity(obj);
  TEST_ASSERT(MFTLObjectPoolsAreConsistent(obj));

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_zeroes_the_new_tail(void)
{
  /* Same argument as test_zeroed_on_first_allocation, for the block a shrink
   * produces: slots above the live count must read zero, or a later
   * MFTLObjectSetCounts() hands the caller uninitialised `flags` and a closed
   * mesh claims a boundary. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 30000, 30000);
  MFTLObjectSetCounts(obj, 300, 300, 200, 200);
  fill_pattern(obj, 9);

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));

  for (int v = obj->verts.n; v < obj->verts.capacity; ++v)
  {
    TEST_ASSERT_EQUAL_INT(0, obj->verts.rec[v].flags);
    for (int k = 0; k < 3; ++k) TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->verts.pos[3 * v + k], 0.0);
  }
  for (int h = 3 * obj->tris.n; h < 3 * obj->tris.capacity; ++h)
  {
    TEST_ASSERT_EQUAL_INT(0, obj->tris.he[h].flags);
  }

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_shrink_to_fit_leaves_the_bookkeeping_alone(void)
{
  /* The one test that catches an implementation spelled through
   * MFTLObjectSetCounts(): that stamps partition_dirty and clears `derived`, so
   * every body would be permanently dirty and the redistribution's per-body
   * skip would silently stop firing. Nothing else here would fail. */
  MFTL_Object *obj = MFTLObjectCreate(1);

  MFTLObjectReserve(obj, 50000, 50000);
  MFTLObjectSetCounts(obj, 400, 400, 300, 300);
  MFTLObjectBuildDerived(obj);
  obj->partition_dirty = MFTL_FALSE;

  TEST_ASSERT(MFTLObjectShrinkToFit(obj, MFTL_SHRINK_SLACK_DEFAULT));

  TEST_ASSERT(MFTLObjectHasTopology(obj));
  TEST_ASSERT(MFTLObjectHasFrames(obj));
  TEST_ASSERT(!obj->partition_dirty);
  TEST_ASSERT_EQUAL_INT(400, obj->verts.n);
  TEST_ASSERT_EQUAL_INT(400, obj->verts.n_local);
  TEST_ASSERT_EQUAL_INT(300, obj->tris.n);

  MFTLObjectDestroy(&obj);
  return 0;
}

static int test_zeroed_on_first_allocation(void)
{
  /* Fields no explicit pass writes must start at a defined value -- `flags` in
   * particular, where a stray bit would claim a closed mesh has a boundary. */
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLObjectSetCounts(obj, 64, 64, 64, 64);

  for (int v = 0; v < 64; ++v) TEST_ASSERT_EQUAL_INT(0, obj->verts.rec[v].flags);
  for (int h = 0; h < 192; ++h) TEST_ASSERT_EQUAL_INT(0, obj->tris.he[h].flags);
  for (int v = 0; v < 192; ++v) TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->verts.pos[v], 0.0);
  for (int t = 0; t < 64; ++t) TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->tris.area[t], 0.0);

  MFTLObjectDestroy(&obj);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Global statistics (collective)
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_update_global_stats_counts_the_owned_prefix(void)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, obj);

  // 10 vertices of which 6 are owned; 8 triangles of which 5 are owned.
  MFTLObjectSetCounts(obj, 10, 6, 8, 5);
  MFTLObjectUpdateGlobalStats(obj);

  const int nproc = g_ens->domain.nproc;
  TEST_ASSERT_EQUAL_INT(6 * nproc, obj->global_n_vertices);
  TEST_ASSERT_EQUAL_INT(5 * nproc, obj->global_n_triangles);

  // Ghosts must not be counted twice: the prefix IS the ownership statement.
  TEST_ASSERT(obj->global_n_vertices != 10 * nproc);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 4);

  printf("--- Running MFTL Object Tests ---\n");

  RUN_TEST(test_fresh_object_is_empty_and_consistent);
  RUN_TEST(test_destroy_is_null_safe_and_idempotent);
  RUN_TEST(test_reserve_sizes_every_array_of_both_pools);
  RUN_TEST(test_capacity_is_shared_across_many_growth_steps);
  RUN_TEST(test_half_edge_storage_is_exactly_three_per_triangle);
  RUN_TEST(test_growth_preserves_every_array);
  RUN_TEST(test_set_counts_reserves_before_it_assigns);
  RUN_TEST(test_set_counts_invalidates_the_derived_arrays);
  RUN_TEST(test_reserve_never_shrinks);
  RUN_TEST(test_clear_keeps_the_allocation);
  RUN_TEST(test_reserve_additional_is_geometric);
  RUN_TEST(test_shrink_to_fit_recovers_the_block);
  RUN_TEST(test_shrink_to_fit_preserves_every_entity);
  RUN_TEST(test_shrink_to_fit_never_drops_entities);
  RUN_TEST(test_shrink_to_fit_is_idempotent);
  RUN_TEST(test_shrink_to_fit_respects_the_slack_gate);
  RUN_TEST(test_shrink_to_fit_does_not_thrash_against_the_grow_ladder);
  RUN_TEST(test_shrink_to_fit_at_zero_releases_the_block);
  RUN_TEST(test_shrink_to_fit_zeroes_the_new_tail);
  RUN_TEST(test_shrink_to_fit_leaves_the_bookkeeping_alone);
  RUN_TEST(test_zeroed_on_first_allocation);
  RUN_TEST(test_update_global_stats_counts_the_owned_prefix);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
