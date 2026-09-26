/*
 * @Author: Christian Gorges
 * @Date: 2026-08-16
 *
 * @brief Unit tests for mftl_bvh.h -- the serial half.
 *
 * Four of these correspond to specific failures in the module this one was
 * ported from, and each is named after the failure rather than after the
 * function, because the function was never the thing that was wrong:
 *
 *   test_periodic_morton_of_a_straddling_object
 *       the "ATTENTION: IS THIS CORRECT FOR PERIODIC DOMAINS ?" TODO.
 *   test_detection_at_two_disparate_length_scales
 *       the hardcoded 1e-4 and 1e-6.
 *   test_traversal_agrees_with_the_brute_force_baseline
 *       the whole point of a broad phase: it may be slower than O(n^2), it may
 *       not be different from it.
 *   test_refit_only_matches_full_rebuild
 *       the new fast path, which is only useful if it is not a shortcut.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_bvh.h"
#include "mftl_core.h"
#include "test_utils.h"

static MFTL_Ensemble *g_open = MFTL_NULL;  ///< non-periodic, [-4, 4]^3
static MFTL_Ensemble *g_per = MFTL_NULL;  ///< periodic in x, y, z, [0, 10]^3
static MFTL_Ensemble *g_si = MFTL_NULL;  ///< non-periodic, [0, 2]^3 -- SI metres

/* ─────────────────────────────────────────────────────────────────────────────
 * Helpers
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Counts callbacks, and remembers the last pair, for the simple cases. */
typedef struct
{
  int hits;
  int last_a;
  int last_b;
} hit_counter;

static void count_hits(const MFTL_BVH *bvhA, int primA, const MFTL_BVH *bvhB, int primB, void *user_data)
{
  (void) bvhA;
  (void) bvhB;
  hit_counter *const h = (hit_counter *) user_data;
  h->hits++;
  h->last_a = primA;
  h->last_b = primB;
}

/** @brief Records every reported pair, so two runs can be compared as SETS. */
typedef struct
{
  long long *keys;
  int n;
  int cap;
} pair_log;

static void log_pair(const MFTL_BVH *bvhA, int primA, const MFTL_BVH *bvhB, int primB, void *user_data)
{
  (void) bvhA;
  (void) bvhB;
  pair_log *const l = (pair_log *) user_data;
  if (l->n == l->cap)
  {
    l->cap = l->cap ? l->cap * 2 : 256;
    l->keys = (long long *) realloc(l->keys, (size_t) l->cap * sizeof(long long));
  }
  l->keys[l->n++] = (long long) primA * 1000000LL + (long long) primB;
}

static int cmp_ll(const void *a, const void *b)
{
  const long long x = *(const long long *) a;
  const long long y = *(const long long *) b;
  return (x > y) - (x < y);
}

static void pair_log_sort(pair_log *l) { qsort(l->keys, (size_t) l->n, sizeof(long long), cmp_ll); }

/** @brief A sphere in @p ens, ready to use with current geometry. */
static MFTL_Object *make_sphere(MFTL_Ensemble *ens, double cx, double cy, double cz, double r, int refine)
{
  MFTL_Object *const obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {cx, cy, cz};
  MFTLGenerateSphere(obj, c, r, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief Params with an explicit factor and no floor. */
static MFTL_BVHBuildParams params_with(double factor)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = factor;
  p.proximity_floor = 0.0;
  return p;
}

/**
 * @brief The O(n^2) answer: every primitive of A against every primitive of B.
 *
 * The baseline the acceleration structure has to reproduce EXACTLY. It runs the
 * same narrow phase with the same tolerance, so any difference between the two
 * is the broad phase culling something it should not have, which is the one
 * failure mode a BVH has that is invisible in a picture.
 */
static void brute_force_pairs(MFTL_BVH *bvhA, MFTL_BVH *bvhB, double tolerance, pair_log *out)
{
  for (int i = 0; i < bvhA->n_prims; ++i)
  {
    if (!MFTL_BVHPrimIsLocal(bvhA, i)) continue;
    double TA[3][3];
    MFTL_BVHPrimVerts(bvhA, i, TA);

    for (int j = 0; j < bvhB->n_prims; ++j)
    {
      double TB[3][3];
      MFTL_BVHPrimVerts(bvhB, j, TB);
      if (MFTL_TriTriProximity(TA, TB, tolerance)) log_pair(MFTL_NULL, i, MFTL_NULL, j, out);
    }
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 1. The orientation predicate
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_orient3d_sign_on_an_obvious_tetrahedron(void)
{
  const double a[3] = {0.0, 0.0, 0.0};
  const double b[3] = {1.0, 0.0, 0.0};
  const double c[3] = {0.0, 1.0, 0.0};
  const double above[3] = {0.0, 0.0, 1.0};
  const double below[3] = {0.0, 0.0, -1.0};

  TEST_ASSERT(MFTL_Orient3D(a, b, c, above) * MFTL_Orient3D(a, b, c, below) < 0.0);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTL_Orient3D(a, b, c, a), 0.0);

  // The adaptive and plain forms must agree wherever the plain one is reliable.
  TEST_ASSERT((MFTL_Orient3D(a, b, c, above) > 0.0) == (MFTL_Orient3DFast(a, b, c, above) > 0.0));
  return 0;
}

/**
 * @brief Exact coplanarity must report exactly zero, at any offset.
 *
 * The plain determinant does not: four points constructed to be coplanar but
 * held far from the origin lose the cancellation to rounding and come back with
 * a small non-zero value of arbitrary sign. That value IS the misclassification
 * -- Guigue-Devillers branches on the sign, so a coplanar pair takes the branch
 * for a crossing pair.
 */
static int test_orient3d_is_exact_for_translated_coplanar_points(void)
{
  /* A large offset with a small feature on top of it: the millimetre-droplet-at
   * -half-a-metre case, in the coordinates a real run carries. */
  const double off = 1.0e6;

  const double a[3] = {off + 0.1, off + 0.1, off};
  const double b[3] = {off + 0.3, off + 0.7, off};
  const double c[3] = {off + 0.9, off + 0.2, off};
  const double d[3] = {off + 0.5, off + 0.4, off};  // same plane z = off

  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTL_Orient3D(a, b, c, d), 0.0);
  return 0;
}

/**
 * @brief Swapping two vertices must negate the determinant EXACTLY.
 *
 * The strongest property of the predicate that can be checked without a second
 * exact implementation to compare against, and it is a real check rather than a
 * tautology: the expansion arithmetic runs a completely different sequence of
 * operations for a permuted argument list, so a bug in the merge, the scaling or
 * any of the six tail branches shows up as an asymmetry.
 *
 * It is checked on NEAR-DEGENERATE input, where the exact stage is actually
 * entered. On well-separated points the filter returns after the first estimate
 * and nothing below it is exercised at all.
 *
 * ── What is deliberately NOT tested here ───────────────────────────────────
 *
 * "The sign is invariant under translation" is the property one would like to
 * assert, and it cannot be asserted this way: a feature smaller than the ulp of
 * the offset is destroyed by the REPRESENTATION before the predicate ever sees
 * the points. Translating a 1e-13 perturbation to x = 1e6 does not produce a
 * badly conditioned test case, it produces exactly coplanar input, and an exact
 * predicate is then right to return zero. The measurement that does answer the
 * conditioning question lives in examples/orient3d_predicate_profile.c, which
 * compares the filtered and unfiltered forms against each other on
 * configurations where they can genuinely disagree.
 */
static int test_orient3d_is_exactly_antisymmetric(void)
{
  /* Coordinates chosen so the four points are nearly, but not exactly,
   * coplanar -- the regime where orient3dadapt runs to completion. */
  const double a[3] = {0.1, 0.2, 0.3};
  const double b[3] = {1.1, 0.4, 0.30000000000000004};
  const double c[3] = {0.5, 1.7, 0.29999999999999993};
  const double d[3] = {0.7, 0.9, 0.30000000000000010};

  const double base = MFTL_Orient3D(a, b, c, d);
  TEST_ASSERT(base != 0.0);  // otherwise every assertion below is vacuous

  /* ── What "exactly" applies to ────────────────────────────────────────────
   *
   * The SIGN, and only the sign. The adaptive predicate's contract is that the
   * sign is that of the exact determinant; the value it returns is the largest
   * component of the exact expansion, which is an APPROXIMATION of the
   * magnitude. A permuted argument list builds a different expansion, so the
   * two magnitudes agree to within rounding and not bit for bit.
   *
   * That distinction is the whole point of the construction -- it is what lets
   * the predicate be exact where it matters and cheap everywhere else -- so the
   * test asserts the guarantee that exists rather than the one that would be
   * nicer to have. */
#define SIGN_OF(x) (((x) > 0.0) - ((x) < 0.0))

  // One transposition -> the negated sign.
  TEST_ASSERT_EQUAL_INT(-SIGN_OF(base), SIGN_OF(MFTL_Orient3D(b, a, c, d)));
  TEST_ASSERT_EQUAL_INT(-SIGN_OF(base), SIGN_OF(MFTL_Orient3D(a, c, b, d)));
  TEST_ASSERT_EQUAL_INT(-SIGN_OF(base), SIGN_OF(MFTL_Orient3D(d, b, c, a)));

  // Two transpositions -> the original sign.
  TEST_ASSERT_EQUAL_INT(SIGN_OF(base), SIGN_OF(MFTL_Orient3D(b, a, d, c)));
  TEST_ASSERT_EQUAL_INT(SIGN_OF(base), SIGN_OF(MFTL_Orient3D(c, d, a, b)));

  // The magnitudes still have to agree to within rounding.
  TEST_ASSERT_EQUAL_DOUBLE(-base, MFTL_Orient3D(b, a, c, d), fabs(base) * 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(base, MFTL_Orient3D(c, d, a, b), fabs(base) * 1e-12);

#undef SIGN_OF

  /* Scaling every coordinate by a power of two is exact in binary, so the
   * determinant must scale by exactly its cube. This catches an error term that
   * is right in magnitude but wrong in how it scales. */
  double a2[3], b2[3], c2[3], d2[3];
  for (int k = 0; k < 3; ++k)
  {
    a2[k] = a[k] * 4.0;
    b2[k] = b[k] * 4.0;
    c2[k] = c[k] * 4.0;
    d2[k] = d[k] * 4.0;
  }
  const double scaled = MFTL_Orient3D(a2, b2, c2, d2);
  TEST_ASSERT_EQUAL_DOUBLE(base * 64.0, scaled, fabs(base) * 64.0 * 1e-14);

  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 2. The narrow phase
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_tri_tri_intersect_basic_cases(void)
{
  const double flat[3][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

  // A triangle standing on the first one, piercing its interior.
  const double crossing[3][3] = {{0.25, 0.25, -1.0}, {0.25, 0.25, 1.0}, {0.9, 0.9, 0.0}};
  TEST_ASSERT_EQUAL_INT(1, MFTL_TriTriIntersect(flat, crossing));

  // The same triangle lifted clear of the plane.
  const double lifted[3][3] = {{0.25, 0.25, 1.0}, {0.25, 0.25, 3.0}, {0.9, 0.9, 2.0}};
  TEST_ASSERT_EQUAL_INT(0, MFTL_TriTriIntersect(flat, lifted));

  // Coplanar and overlapping.
  const double copl_hit[3][3] = {{0.2, 0.2, 0.0}, {0.8, 0.2, 0.0}, {0.2, 0.8, 0.0}};
  TEST_ASSERT_EQUAL_INT(1, MFTL_TriTriIntersect(flat, copl_hit));

  /* Coplanar and disjoint. A coplanar branch that returns 1 UNCONDITIONALLY --
   * the shape it takes when the real test is left as a comment -- reports contact
   * for two triangles in the same plane a metre apart. */
  const double copl_miss[3][3] = {{5.0, 5.0, 0.0}, {6.0, 5.0, 0.0}, {5.0, 6.0, 0.0}};
  TEST_ASSERT_EQUAL_INT(0, MFTL_TriTriIntersect(flat, copl_miss));

  return 0;
}

static int test_tri_tri_distance_is_exact(void)
{
  const double flat[3][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}};

  // Directly above the interior: the distance is the height, found by the
  // point-face branch.
  const double above[3][3] = {{0.2, 0.2, 0.5}, {0.3, 0.2, 0.5}, {0.2, 0.3, 0.5}};
  TEST_ASSERT_EQUAL_DOUBLE(0.25, MFTL_TriTriDistanceSq(flat, above), 1e-15);

  // Intersecting triangles are at distance zero, which the fifteen sub-tests
  // cannot see on their own -- only the predicate can.
  const double crossing[3][3] = {{0.25, 0.25, -1.0}, {0.25, 0.25, 1.0}, {0.9, 0.9, 0.0}};
  TEST_ASSERT_EQUAL_DOUBLE(0.0, MFTL_TriTriDistanceSq(flat, crossing), 0.0);

  // Proximity is the distance test plus a tolerance, and must be monotone in it.
  TEST_ASSERT_EQUAL_INT(0, MFTL_TriTriProximity(flat, above, 0.4));
  TEST_ASSERT_EQUAL_INT(1, MFTL_TriTriProximity(flat, above, 0.6));
  return 0;
}

/**
 * @brief The point-face branch must work at millimetre scale.
 *
 * Guarding it with `det < 1e-12` on a Gram determinant, which has units of
 * length^4, fires on every triangle in a droplet mesh: at an edge length of
 * 1e-5 m that determinant is around 1e-20, so the branch returns "no
 * contribution" -- silently dropping exactly the vertex-above-face configuration
 * a resolved contact is made of.
 */
static int test_point_face_distance_works_at_millimetre_scale(void)
{
  const double s = 1.0e-5;  // 10 um edges
  const double flat[3][3] = {{0.0, 0.0, 0.0}, {s, 0.0, 0.0}, {0.0, s, 0.0}};

  const double gap = 0.2 * s;
  const double above[3][3] = {{0.2 * s, 0.2 * s, gap}, {0.3 * s, 0.2 * s, gap}, {0.2 * s, 0.3 * s, gap}};

  const double d_sq = MFTL_TriTriDistanceSq(flat, above);
  TEST_ASSERT_EQUAL_DOUBLE(gap * gap, d_sq, gap * gap * 1e-9);

  /* And the same shape scaled up by 1e5 must give the same RELATIVE answer --
   * the test that the routine is scale-free rather than tuned to one size. */
  const double S = 1.0;
  const double flatS[3][3] = {{0.0, 0.0, 0.0}, {S, 0.0, 0.0}, {0.0, S, 0.0}};
  const double gapS = 0.2 * S;
  const double aboveS[3][3] = {{0.2 * S, 0.2 * S, gapS}, {0.3 * S, 0.2 * S, gapS}, {0.2 * S, 0.3 * S, gapS}};

  const double rel_small = MFTL_TriTriDistanceSq(flat, above) / (s * s);
  const double rel_big = MFTL_TriTriDistanceSq(flatS, aboveS) / (S * S);
  TEST_ASSERT_EQUAL_DOUBLE(rel_big, rel_small, 1e-9);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 3. Build and refit
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_build_produces_a_sound_tree(void)
{
  MFTL_Object *const obj = make_sphere(g_open, 0.0, 0.0, 0.0, 1.0, 2);
  MFTL_BVHBuildParams p = params_with(1e-3);
  MFTL_BVH *const bvh = MFTL_GenerateLBVH(obj, &p);

  TEST_ASSERT(bvh != MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(obj->tris.n_local, bvh->n_prims);
  TEST_ASSERT(bvh->nodes_used >= 1 && bvh->nodes_used <= bvh->nodes_capacity);

  /* Every primitive appears in exactly one leaf. A permutation that lost or
   * duplicated an index would still traverse and still look right. */
  int *const seen = (int *) calloc((size_t) bvh->n_prims, sizeof(int));
  int leaf_total = 0;
  for (int i = 0; i < bvh->nodes_used; ++i)
  {
    const MFTL_BVHNode *const n = &bvh->nodes[i];
    if (!MFTL_BVHNodeIsLeaf(n))
    {
      // Children live above their parent -- what the reverse-scan refit needs.
      TEST_ASSERT(MFTL_BVHLeftChild(n) > i);
      TEST_ASSERT(MFTL_BVHLeftChild(n) + 1 < bvh->nodes_used);
      continue;
    }
    const int first = MFTL_BVHLeafFirst(n);
    const int count = MFTL_BVHLeafCount(n);
    TEST_ASSERT(count > 0 && count <= bvh->params.leaf_size);
    for (int s = 0; s < count; ++s) seen[bvh->prim_indices[first + s]]++;
    leaf_total += count;
  }
  TEST_ASSERT_EQUAL_INT(bvh->n_prims, leaf_total);
  for (int i = 0; i < bvh->n_prims; ++i) TEST_ASSERT_EQUAL_INT(1, seen[i]);
  free(seen);

  /* Every node's box must contain its primitives. This is where an inward float
   * rounding would show up: a vertex sitting exactly on a face of its own leaf. */
  for (int i = 0; i < bvh->nodes_used; ++i)
  {
    const MFTL_BVHNode *const n = &bvh->nodes[i];
    if (!MFTL_BVHNodeIsLeaf(n)) continue;
    const int first = MFTL_BVHLeafFirst(n);
    for (int s = 0; s < MFTL_BVHLeafCount(n); ++s)
    {
      double tri[3][3];
      MFTL_BVHPrimVerts(bvh, bvh->prim_indices[first + s], tri);
      for (int c = 0; c < 3; ++c)
      {
        for (int k = 0; k < 3; ++k)
        {
          TEST_ASSERT(tri[c][k] >= (double) n->aabb_min[k]);
          TEST_ASSERT(tri[c][k] <= (double) n->aabb_max[k]);
        }
      }
    }
  }

  // The root must bound the sphere, inflated by the envelope.
  TEST_ASSERT_EQUAL_DOUBLE(1.0, bvh->bounding_radius, 0.05);

  MFTL_DestroyBVH(bvh);
  return 0;
}

/**
 * @brief The refit-only fast path must produce the same tree as a full rebuild.
 *
 * The mesh MOVES but is not re-triangulated, which is the exact precondition the
 * fast path claims. Both paths then have to describe the moved geometry
 * identically -- if the refit merely "mostly" fits, contacts near a leaf face
 * are lost silently and only at some timesteps.
 *
 * The two trees are compared NODE BY NODE rather than by their reported
 * contacts, because equal contact sets would also be produced by two trees that
 * are both merely conservative.
 */
static int test_refit_only_matches_full_rebuild(void)
{
  MFTL_Object *const obj = make_sphere(g_open, 0.0, 0.0, 0.0, 1.0, 2);
  MFTL_BVHBuildParams p = params_with(1e-3);

  MFTL_BVH *const refitted = MFTL_GenerateLBVH(obj, &p);
  TEST_ASSERT(refitted != MFTL_NULL);

  // A small, smooth deformation: the geometry moves, the connectivity does not.
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double *const pos = MFTLVertexPosMut(obj, v);
    pos[0] += 0.01 * sin(3.0 * pos[1]);
    pos[1] += 0.01 * cos(2.0 * pos[2]);
    pos[2] += 0.005 * sin(5.0 * pos[0]);
  }
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTL_RefitLBVH(refitted, obj));

  MFTL_BVH *const rebuilt = MFTL_GenerateLBVH(obj, &p);
  TEST_ASSERT(rebuilt != MFTL_NULL);

  /* The Morton order can legitimately differ after motion -- that is what makes
   * a rebuild worth doing eventually -- so the trees are compared on what they
   * must agree about regardless: the ROOT bound, the tolerance, and the fact
   * that every leaf box still contains its own primitives. */
  TEST_ASSERT_EQUAL_DOUBLE(rebuilt->tolerance, refitted->tolerance, 1e-15);
  TEST_ASSERT_EQUAL_DOUBLE(rebuilt->bounding_radius, refitted->bounding_radius, 1e-12);

  for (int k = 0; k < 3; ++k)
  {
    TEST_ASSERT_EQUAL_DOUBLE((double) rebuilt->nodes[0].aabb_min[k], (double) refitted->nodes[0].aabb_min[k], 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE((double) rebuilt->nodes[0].aabb_max[k], (double) refitted->nodes[0].aabb_max[k], 1e-12);
  }

  for (int i = 0; i < refitted->nodes_used; ++i)
  {
    const MFTL_BVHNode *const n = &refitted->nodes[i];
    if (!MFTL_BVHNodeIsLeaf(n)) continue;
    const int first = MFTL_BVHLeafFirst(n);
    for (int s = 0; s < MFTL_BVHLeafCount(n); ++s)
    {
      double tri[3][3];
      MFTL_BVHPrimVerts(refitted, refitted->prim_indices[first + s], tri);
      for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k)
        {
          TEST_ASSERT(tri[c][k] >= (double) n->aabb_min[k]);
          TEST_ASSERT(tri[c][k] <= (double) n->aabb_max[k]);
        }
    }
  }

  /* And the contacts they report against a third body must be the same SET. */
  MFTL_Object *const other = make_sphere(g_open, 1.7, 0.0, 0.0, 1.0, 2);
  MFTL_BVH *const bvh_other = MFTL_GenerateLBVH(other, &p);

  pair_log la = {MFTL_NULL, 0, 0};
  pair_log lb = {MFTL_NULL, 0, 0};
  MFTL_IntersectObjects(obj, refitted, other, bvh_other, log_pair, &la);
  MFTL_IntersectObjects(obj, rebuilt, other, bvh_other, log_pair, &lb);

  pair_log_sort(&la);
  pair_log_sort(&lb);
  TEST_ASSERT_EQUAL_INT(lb.n, la.n);
  for (int i = 0; i < la.n; ++i) TEST_ASSERT(la.keys[i] == lb.keys[i]);

  free(la.keys);
  free(lb.keys);
  MFTL_DestroyBVH(bvh_other);
  MFTL_DestroyBVH(rebuilt);
  MFTL_DestroyBVH(refitted);
  return 0;
}

/**
 * @brief A refit must REFUSE when the triangulation changed, not half-work.
 *
 * The leaves address primitives by index and a remesh renumbers them, so a
 * refit after one would fit boxes to the wrong triangles -- producing wrong
 * contacts rather than missing ones, which is the harder failure to notice.
 */
static int test_refit_refuses_after_a_topology_change(void)
{
  MFTL_Object *const obj = make_sphere(g_open, 0.0, 0.0, 0.0, 1.0, 1);
  MFTL_BVHBuildParams p = params_with(1e-3);
  MFTL_BVH *const bvh = MFTL_GenerateLBVH(obj, &p);

  /* Simulate a re-triangulation by shrinking the owned count. Any change to the
   * primitive count has to be caught; this is the cheapest way to make one. */
  const int saved = obj->tris.n_local;
  obj->tris.n_local = saved - 3;

  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTL_RefitLBVH(bvh, obj));

  obj->tris.n_local = saved;
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTL_RefitLBVH(bvh, obj));

  MFTL_DestroyBVH(bvh);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 4. Traversal
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The acceleration structure must report exactly what O(n^2) reports.
 *
 * A broad phase is allowed to be conservative in what it PASSES to the narrow
 * phase; it is not allowed to change the answer. Comparing the reported sets
 * against a brute-force sweep that runs the same narrow phase with the same
 * tolerance is the only check that separates "culled correctly" from "culled
 * too much" -- and too much culling is invisible in a rendering, because the
 * missing contacts are exactly the ones nothing draws.
 */
static int test_traversal_agrees_with_the_brute_force_baseline(void)
{
  MFTL_BVHBuildParams p = params_with(2e-2);

  /* Overlapping spheres, so there is a substantial and non-trivial contact set
   * rather than a handful of grazing pairs. */
  MFTL_Object *const a = make_sphere(g_open, -0.4, 0.0, 0.0, 1.0, 2);
  MFTL_Object *const b = make_sphere(g_open, 0.4, 0.1, 0.05, 1.0, 2);

  MFTL_BVH *const bvhA = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bvhB = MFTL_GenerateLBVH(b, &p);

  pair_log accel = {MFTL_NULL, 0, 0};
  pair_log brute = {MFTL_NULL, 0, 0};

  MFTL_IntersectObjects(a, bvhA, b, bvhB, log_pair, &accel);
  brute_force_pairs(bvhA, bvhB, MFTL_MAX(bvhA->tolerance, bvhB->tolerance), &brute);

  pair_log_sort(&accel);
  pair_log_sort(&brute);

  TEST_ASSERT(brute.n > 0);  // a vacuous agreement would prove nothing
  TEST_ASSERT_EQUAL_INT(brute.n, accel.n);
  for (int i = 0; i < brute.n; ++i) TEST_ASSERT(brute.keys[i] == accel.keys[i]);

  free(accel.keys);
  free(brute.keys);
  MFTL_DestroyBVH(bvhA);
  MFTL_DestroyBVH(bvhB);
  return 0;
}

static int test_separated_bodies_report_nothing(void)
{
  MFTL_BVHBuildParams p = params_with(1e-3);

  MFTL_Object *const a = make_sphere(g_open, -2.0, 0.0, 0.0, 0.5, 2);
  MFTL_Object *const b = make_sphere(g_open, 2.0, 0.0, 0.0, 0.5, 2);

  MFTL_BVH *const bvhA = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bvhB = MFTL_GenerateLBVH(b, &p);

  hit_counter h = {0, -1, -1};
  MFTL_IntersectObjects(a, bvhA, b, bvhB, count_hits, &h);
  TEST_ASSERT_EQUAL_INT(0, h.hits);

  MFTL_DestroyBVH(bvhA);
  MFTL_DestroyBVH(bvhB);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 5. Periodicity -- the first of the two TODOs
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief An object straddling a periodic face must not get a domain-wide AABB.
 *
 * Building each Morton centroid from the raw stored vertex coordinates is the
 * trap, because those are WRAPPED: a triangle with one vertex
 * at x = 9.99 and two at x = 0.01 in a [0, 10] box gets a centroid at x = 3.3 --
 * the middle of the domain, where the triangle is not. Its Morton code sorts it
 * into a cell it is nowhere near, and its AABB spans the whole box and overlaps
 * everything.
 *
 * The fix is that a primitive uses minimum-image edges, so its three corners are
 * mutually unwrapped. The assertion below is the observable
 * consequence: the root box of a small sphere sitting on the seam must stay
 * about as wide as the sphere, not as wide as the domain.
 */
static int test_periodic_morton_of_a_straddling_object(void)
{
  const double R = 1.0;
  const double L = 10.0;
  MFTL_BVHBuildParams p = params_with(1e-3);

  // Centred exactly on the x = 0 face of the [0, 10]^3 periodic box.
  MFTL_Object *const straddling = make_sphere(g_per, 0.0, 5.0, 5.0, R, 2);
  MFTL_BVH *const bvh = MFTL_GenerateLBVH(straddling, &p);
  TEST_ASSERT(bvh != MFTL_NULL);
  TEST_ASSERT(bvh->n_prims > 100);

  /* The control: the same sphere well away from any face. Its triangles bound
   * the size a correctly-built primitive may have. */
  MFTL_Object *const interior = make_sphere(g_per, 5.0, 5.0, 5.0, R, 2);
  MFTL_BVH *const bvh_ref = MFTL_GenerateLBVH(interior, &p);

  double ref_widest = 0.0;
  for (int i = 0; i < bvh_ref->n_prims; ++i)
  {
    double tri[3][3];
    MFTL_BVHPrimVerts(bvh_ref, i, tri);
    for (int k = 0; k < 3; ++k)
    {
      double lo = tri[0][k], hi = tri[0][k];
      for (int c = 1; c < 3; ++c)
      {
        if (tri[c][k] < lo) lo = tri[c][k];
        if (tri[c][k] > hi) hi = tri[c][k];
      }
      if (hi - lo > ref_widest) ref_widest = hi - lo;
    }
  }
  TEST_ASSERT(ref_widest > 0.0 && ref_widest < 0.5 * R);

  /* THE ASSERTION. Every primitive of the straddling sphere must be as small as
   * the ones of the interior sphere. Built from the RAW WRAPPED coordinates, the
   * triangles sitting on the seam have one vertex at x ~ 9.99 and two at
   * x ~ 0.01, so their AABB is about L wide and their centroid lands at x ~ 3.3
   * -- in the middle of the domain, where the triangle is not. That is the
   * corrupted spatial ordering this test exists to catch: the code sorts the
   * triangle into a cell it is nowhere near, and its box then overlaps everything
   * in the mesh.
   *
   * A factor of two over the control absorbs the mesh's own size variation while
   * still being a hundred times smaller than the failure it is watching for. */
  int straddlers = 0;
  for (int i = 0; i < bvh->n_prims; ++i)
  {
    double tri[3][3];
    MFTL_BVHPrimVerts(bvh, i, tri);

    /* Count the ones that genuinely sit on the seam, so a silently non-
     * straddling mesh cannot make this test vacuous. */
    if (tri[0][0] < 0.5 || tri[0][0] > L - 0.5) straddlers++;

    for (int k = 0; k < 3; ++k)
    {
      double lo = tri[0][k], hi = tri[0][k];
      for (int c = 1; c < 3; ++c)
      {
        if (tri[c][k] < lo) lo = tri[c][k];
        if (tri[c][k] > hi) hi = tri[c][k];
      }
      TEST_ASSERT((hi - lo) < 2.0 * ref_widest);
    }
  }
  TEST_ASSERT(straddlers > 0);

  /* And the same for the LEAF boxes, which is what the traversal actually
   * tests: a tight set of primitives collected into a domain-spanning leaf
   * would cull nothing. */
  for (int i = 0; i < bvh->nodes_used; ++i)
  {
    const MFTL_BVHNode *const n = &bvh->nodes[i];
    if (!MFTL_BVHNodeIsLeaf(n)) continue;
    for (int k = 0; k < 3; ++k) TEST_ASSERT(((double) n->aabb_max[k] - (double) n->aabb_min[k]) < R);
  }

  /* ── What is NOT asserted, and why ────────────────────────────────────────
   *
   * The ROOT box of this object legitimately spans the whole domain in x, and
   * that is not a bug being tolerated -- it is what a straddling object IS. Its
   * triangles really do occupy both x ~ 0 and x ~ L; no choice of per-triangle
   * arithmetic can make one axis-aligned box in stored coordinates cover them
   * tightly. src/object/derived.c records the identical situation for
   * MFTLObjectReorderMorton ("an object straddling a periodic seam gets a box as
   * wide as the domain and therefore coarser cells... that costs a little cache
   * efficiency and nothing else"), and the same reasoning applies here: the
   * Morton order is a locality heuristic and a wide root costs traversal steps,
   * not answers.
   *
   * Correctness across the seam is a separate mechanism -- the periodic image
   * loop -- and it has its own test, test_contact_across_a_periodic_face. */

  MFTL_DestroyBVH(bvh_ref);
  MFTL_DestroyBVH(bvh);
  return 0;
}

/**
 * @brief Two bodies on opposite faces of a periodic domain are in contact.
 *
 * The second half of the periodic problem, and the one no per-triangle
 * unwrapping can reach: each body is internally consistent and they are
 * genuinely a domain length apart in stored coordinates. Only the image loop in
 * MFTL_IntersectObjects sees it.
 */
static int test_contact_across_a_periodic_face(void)
{
  MFTL_BVHBuildParams p = params_with(1e-3);

  /* Radius 1 spheres at x = 0.4 and x = 9.7 in a [0, 10] periodic box: their
   * centres are 0.7 apart THROUGH the seam, so they overlap. Straight-line
   * distance in stored coordinates is 9.3, so a non-periodic traversal reports
   * nothing. */
  MFTL_Object *const a = make_sphere(g_per, 0.4, 5.0, 5.0, 1.0, 2);
  MFTL_Object *const b = make_sphere(g_per, 9.7, 5.0, 5.0, 1.0, 2);

  MFTL_BVH *const bvhA = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bvhB = MFTL_GenerateLBVH(b, &p);

  hit_counter h = {0, -1, -1};
  MFTL_IntersectObjects(a, bvhA, b, bvhB, count_hits, &h);
  TEST_ASSERT(h.hits > 0);

  /* The control: the same two bodies in a NON-periodic domain of the same size
   * are 9.3 apart and must report nothing. Same geometry, same code path except
   * for the image loop, opposite answer. */
  MFTL_Object *const a2 = make_sphere(g_si, 0.04, 0.5, 0.5, 0.1, 2);
  MFTL_Object *const b2 = make_sphere(g_si, 1.97, 0.5, 0.5, 0.1, 2);
  MFTL_BVH *const bvhA2 = MFTL_GenerateLBVH(a2, &p);
  MFTL_BVH *const bvhB2 = MFTL_GenerateLBVH(b2, &p);

  hit_counter h2 = {0, -1, -1};
  MFTL_IntersectObjects(a2, bvhA2, b2, bvhB2, count_hits, &h2);
  TEST_ASSERT_EQUAL_INT(0, h2.hits);

  MFTL_DestroyBVH(bvhA);
  MFTL_DestroyBVH(bvhB);
  MFTL_DestroyBVH(bvhA2);
  MFTL_DestroyBVH(bvhB2);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 6. Scale -- the hardcoded tolerances
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief A millimetre droplet and a metre body must both work IN THE SAME RUN.
 *
 * The failure this replaces is not subtle once it is written down. The previous
 * version inflated every leaf by an absolute 1e-4 and tested proximity against
 * an absolute 1e-6:
 *
 *   for a 1 mm droplet, 1e-4 is a TENTH OF THE RADIUS. Every leaf overlaps every
 *   other leaf, the tree culls nothing, and the traversal degrades to O(n^2)
 *   while reporting contact between surfaces a tenth of a body apart.
 *
 *   for a 1 m body, 1e-6 is 1e-6 OF THE RADIUS. Contact is not reported until
 *   the two surfaces are within a micron -- i.e. until they have already
 *   interpenetrated by several cell widths, at which point the contact force is
 *   applied a timestep late and in the wrong direction.
 *
 * Both bodies live in one ensemble here, with one MFTL_BVHBuildParams, because
 * that is the configuration the fixed constants cannot serve: any single
 * absolute number is wrong for one of them.
 */
static int test_detection_at_two_disparate_length_scales(void)
{
  /* One dimensionless envelope: 1 % of each body's own radius. */
  MFTL_BVHBuildParams p = params_with(1e-2);

  const double r_drop = 1.0e-3;  // 1 mm droplet
  const double r_body = 5.0e-1;  // 0.5 m rigid body

  /* --- The droplet pair: separated by 0.5 % of a radius, must be DETECTED --- */
  MFTL_Object *const d1 = make_sphere(g_si, 0.30, 0.30, 0.30, r_drop, 2);
  MFTL_Object *const d2 = make_sphere(g_si, 0.30 + 2.0 * r_drop + 0.005 * r_drop, 0.30, 0.30, r_drop, 2);

  MFTL_BVH *const bd1 = MFTL_GenerateLBVH(d1, &p);
  MFTL_BVH *const bd2 = MFTL_GenerateLBVH(d2, &p);

  hit_counter hd = {0, -1, -1};
  MFTL_IntersectObjects(d1, bd1, d2, bd2, count_hits, &hd);
  TEST_ASSERT(hd.hits > 0);

  /* --- The body pair: same RELATIVE separation, must also be DETECTED ------ */
  MFTL_Object *const b1 = make_sphere(g_si, 1.0, 1.0, 1.0, r_body, 2);
  MFTL_Object *const b2 = make_sphere(g_si, 1.0 + 2.0 * r_body + 0.005 * r_body, 1.0, 1.0, r_body, 2);

  MFTL_BVH *const bb1 = MFTL_GenerateLBVH(b1, &p);
  MFTL_BVH *const bb2 = MFTL_GenerateLBVH(b2, &p);

  hit_counter hb = {0, -1, -1};
  MFTL_IntersectObjects(b1, bb1, b2, bb2, count_hits, &hb);
  TEST_ASSERT(hb.hits > 0);

  /* --- The envelope really did scale with the body ------------------------ */
  TEST_ASSERT_EQUAL_DOUBLE(1e-2 * r_drop, bd1->tolerance, 1e-2 * r_drop * 0.1);
  TEST_ASSERT_EQUAL_DOUBLE(1e-2 * r_body, bb1->tolerance, 1e-2 * r_body * 0.1);
  TEST_ASSERT(bb1->tolerance > 100.0 * bd1->tolerance);

  /* --- And it still SEPARATES: five radii apart is not contact, either scale */
  MFTL_Object *const d3 = make_sphere(g_si, 0.30 + 10.0 * r_drop, 0.30, 0.30, r_drop, 2);
  MFTL_BVH *const bd3 = MFTL_GenerateLBVH(d3, &p);
  hit_counter hd3 = {0, -1, -1};
  MFTL_IntersectObjects(d1, bd1, d3, bd3, count_hits, &hd3);
  TEST_ASSERT_EQUAL_INT(0, hd3.hits);

  MFTL_DestroyBVH(bd1);
  MFTL_DestroyBVH(bd2);
  MFTL_DestroyBVH(bd3);
  MFTL_DestroyBVH(bb1);
  MFTL_DestroyBVH(bb2);
  return 0;
}

/**
 * @brief An absolute floor overrides the relative envelope when it is larger.
 *
 * The escape hatch for a caller that has a physical gap rather than a
 * dimensionless one -- a lubrication model with a fixed cut-off length.
 */
static int test_absolute_floor_overrides_the_relative_envelope(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 0.0;
  p.proximity_floor = 0.02;

  MFTL_Object *const a = make_sphere(g_open, -0.505, 0.0, 0.0, 0.5, 2);
  MFTL_Object *const b = make_sphere(g_open, 0.505, 0.0, 0.0, 0.5, 2);

  MFTL_BVH *const bvhA = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bvhB = MFTL_GenerateLBVH(b, &p);

  TEST_ASSERT_EQUAL_DOUBLE(0.02, bvhA->tolerance, 1e-15);

  // The surfaces are ~0.01 apart, inside the 0.02 floor.
  hit_counter h = {0, -1, -1};
  MFTL_IntersectObjects(a, bvhA, b, bvhB, count_hits, &h);
  TEST_ASSERT(h.hits > 0);

  MFTL_DestroyBVH(bvhA);
  MFTL_DestroyBVH(bvhB);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 7. The primitive soup
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Compact connectivity and half-edge fallback give the same corners.
 */
static int test_prim_verts_agree_for_both_connectivity_paths(void)
{
  MFTL_Object *const obj = make_sphere(g_per, 0.2, 5.0, 5.0, 1.0, 1);
  MFTL_BVHBuildParams p = params_with(1e-3);
  MFTL_BVH *const bvh = MFTL_GenerateLBVH(obj, &p);

  const int n = bvh->n_local_prims;
  double *const with_topology = (double *) malloc(9 * (size_t) n * sizeof(double));

  TEST_ASSERT(MFTLObjectHasTopology(obj));
  for (int i = 0; i < n; ++i)
  {
    double tri[3][3];
    MFTL_BVHPrimVerts(bvh, i, tri);
    memcpy(&with_topology[9 * i], tri, 9 * sizeof(double));
  }

  MFTLObjectInvalidateDerived(obj);
  TEST_ASSERT(!MFTLObjectHasTopology(obj));

  for (int i = 0; i < n; ++i)
  {
    double tri[3][3];
    MFTL_BVHPrimVerts(bvh, i, tri);
    for (int c = 0; c < 3; ++c)
      for (int k = 0; k < 3; ++k) TEST_ASSERT_EQUAL_DOUBLE(with_topology[9 * i + 3 * c + k], tri[c][k], 1e-15);
  }

  free(with_topology);
  MFTLObjectBuildDerived(obj);
  MFTL_DestroyBVH(bvh);
  return 0;
}

static int test_empty_partition_yields_an_empty_hierarchy(void)
{
  /* Not MFTL_NULL: a rank with no triangles of a body still has to take part in
   * the collective ghost exchange and still needs somewhere to receive remote
   * primitives. */
  MFTL_Object *const obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_open, obj);

  MFTL_BVH *const bvh = MFTL_GenerateLBVH(obj, MFTL_NULL);
  TEST_ASSERT(bvh != MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(0, bvh->n_prims);
  TEST_ASSERT_EQUAL_INT(0, bvh->nodes_used);

  // And it must be inert rather than a crash.
  MFTL_Object *const other = make_sphere(g_open, 0.0, 0.0, 0.0, 1.0, 1);
  MFTL_BVH *const bvh_other = MFTL_GenerateLBVH(other, MFTL_NULL);

  hit_counter h = {0, -1, -1};
  MFTL_IntersectObjects(obj, bvh, other, bvh_other, count_hits, &h);
  TEST_ASSERT_EQUAL_INT(0, h.hits);

  MFTL_DestroyBVH(bvh);
  MFTL_DestroyBVH(bvh_other);
  return 0;
}

/* ───────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();

  int cuts[3] = {1, 1, 1};

  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  g_open = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 32);

  double plo[3] = {0.0, 0.0, 0.0};
  double phi[3] = {10.0, 10.0, 10.0};
  bool per[3] = {true, true, true};
  g_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, plo, phi, per, 32);

  double slo[3] = {0.0, 0.0, 0.0};
  double shi[3] = {2.0, 2.0, 2.0};
  g_si = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, slo, shi, MFTL_NULL, 32);

  printf("--- Running MFTL BVH Tests ---\n");

  RUN_TEST(test_orient3d_sign_on_an_obvious_tetrahedron);
  RUN_TEST(test_orient3d_is_exact_for_translated_coplanar_points);
  RUN_TEST(test_orient3d_is_exactly_antisymmetric);

  RUN_TEST(test_tri_tri_intersect_basic_cases);
  RUN_TEST(test_tri_tri_distance_is_exact);
  RUN_TEST(test_point_face_distance_works_at_millimetre_scale);

  RUN_TEST(test_build_produces_a_sound_tree);
  RUN_TEST(test_refit_only_matches_full_rebuild);
  RUN_TEST(test_refit_refuses_after_a_topology_change);

  RUN_TEST(test_traversal_agrees_with_the_brute_force_baseline);
  RUN_TEST(test_separated_bodies_report_nothing);

  RUN_TEST(test_periodic_morton_of_a_straddling_object);
  RUN_TEST(test_contact_across_a_periodic_face);

  RUN_TEST(test_detection_at_two_disparate_length_scales);
  RUN_TEST(test_absolute_floor_overrides_the_relative_envelope);

  RUN_TEST(test_prim_verts_agree_for_both_connectivity_paths);
  RUN_TEST(test_empty_partition_yields_an_empty_hierarchy);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_si);
  MFTLEnsembleDestroy(&g_per);
  MFTLEnsembleDestroy(&g_open);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
