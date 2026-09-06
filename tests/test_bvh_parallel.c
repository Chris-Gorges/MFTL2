/*
 * @Author: Christian Gorges
 * @Date: 2026-08-16
 *
 * @brief Unit tests for the spatial halo and the distributed contact pipeline.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE TEST THAT MATTERS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Distributed contact detection has exactly two failure modes and they pull in
 * opposite directions, so any test that checks only one of them is satisfied by
 * a broken implementation that fails the other:
 *
 *   MISSES     a contact whose two triangles are owned by different ranks, and
 *              which no rank can see because neither holds both sides. Fixed by
 *              the spatial halo. A test that only counts "did we find something"
 *              passes while missing most of the seam.
 *
 *   DUPLICATES the same contact reported by two ranks, because both ended up
 *              holding both sides -- one as owner, one as halo. Every contact
 *              force built from it is then applied twice. Fixed by the reporting
 *              rule (mftl_bvh.h section 4). A test that only checks "every
 *              contact was found" passes while doubling half of them.
 *
 * So the assertion is on the SET, gathered across every rank, compared against
 * the answer computed on one rank while it still held the whole mesh. Equality
 * of that set rules out both at once, and nothing weaker does.
 *
 * Pairs are named by GLOBAL TRIANGLE ID, the only identity that survives
 * redistribution -- a pool index means nothing on another rank and nothing after
 * the next repartition either.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_bvh.h"
#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;
static int g_rank = 0;
static int g_nproc = 1;

/* ─────────────────────────────────────────────────────────────────────────────
 * Collecting contacts as a comparable set
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct
{
  long long *keys;  ///< gidA * 2^20 + gidB, so a pair is one sortable integer.
  int n;
  int cap;
  const MFTL_BVH *bvh_a;
  const MFTL_BVH *bvh_b;
} contact_set;

static void record_contact(const MFTL_Object *objA, int primA, const MFTL_Object *objB, int primB, void *user_data)
{
  (void) objA;
  (void) objB;
  contact_set *const s = (contact_set *) user_data;

  if (s->n == s->cap)
  {
    s->cap = s->cap ? s->cap * 2 : 512;
    s->keys = (long long *) realloc(s->keys, (size_t) s->cap * sizeof(long long));
  }

  const long long ga = MFTL_BVHPrimGlobalId(s->bvh_a, primA);
  const long long gb = MFTL_BVHPrimGlobalId(s->bvh_b, primB);
  s->keys[s->n++] = ga * (1LL << 20) + gb;
}

/** @brief Counts hits and nothing else, for the pipeline comparison. */
static void count_only(const MFTL_Object *objA, int primA, const MFTL_Object *objB, int primB, void *user_data)
{
  (void) objA;
  (void) primA;
  (void) objB;
  (void) primB;
  (*(int *) user_data)++;
}

static int cmp_ll(const void *a, const void *b)
{
  const long long x = *(const long long *) a;
  const long long y = *(const long long *) b;
  return (x > y) - (x < y);
}

/** @brief Gathers every rank's contacts onto rank 0 and sorts them. */
static long long *gather_contacts(const contact_set *local, int *n_out)
{
  int *counts = MFTL_NULL;
  int *displs = MFTL_NULL;
  if (g_rank == 0)
  {
    counts = (int *) malloc((size_t) g_nproc * sizeof(int));
    displs = (int *) malloc((size_t) g_nproc * sizeof(int));
  }

  MPI_Gather(&local->n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  long long *all = MFTL_NULL;
  if (g_rank == 0)
  {
    for (int i = 0; i < g_nproc; ++i)
    {
      displs[i] = total;
      total += counts[i];
    }
    all = (long long *) malloc((size_t) (total > 0 ? total : 1) * sizeof(long long));
  }

  MPI_Gatherv(local->keys, local->n, MPI_LONG_LONG, all, counts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (g_rank == 0)
  {
    qsort(all, (size_t) total, sizeof(long long), cmp_ll);
    free(counts);
    free(displs);
  }

  *n_out = total;
  return all;
}

/** @brief Two spheres overlapping along x, straddling the x = 5 cut. */
static void make_pair(MFTL_Object **a, MFTL_Object **b, double sep, int refine)
{
  *a = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, *a);
  double ca[3] = {5.0 - 0.5 * sep, 5.0, 5.0};
  MFTLGenerateSphere(*a, ca, 1.0, refine);

  *b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, *b);
  double cb[3] = {5.0 + 0.5 * sep, 5.0, 5.0};
  MFTLGenerateSphere(*b, cb, 1.0, refine);
}

/**
 * @brief Body @p gid as an object, or a shared EMPTY one if this rank holds none.
 *
 * A rank holds an MFTL_Object only for the bodies it has entities of, so a
 * pointer cached before a redistribution may be dangling after it. Contact
 * detection reads bodies through the ensemble's dense view for the same reason,
 * and so does this test.
 */
static MFTL_Object *bodyv(int gid) { return MFTLEnsembleAcquireDenseView(g_ens)[gid]; }

/** @brief The per-timestep preamble every consumer of triangle geometry owes. */
static void refresh(MFTL_Ensemble *ens)
{
  for (int i = 0; i < ens->n_local_objects; ++i)
  {
    MFTLComputeTriangleGeometry(ens->local_objects[i], ens->local_objects[i]->tris.n);
  }
}

/**
 * @brief The explicit contact pipeline for the two-body case.
 *
 * The order is load-bearing and is the same one MFTLEnsembleIntersectObjects
 * runs internally: the radii must be resolved BEFORE the margin can be stated,
 * because the margin is a fraction of the body and the body's size is only
 * knowable collectively.
 *
 * @return The halo, which owns the patch objects and must outlive the BVHs.
 */
static MFTL_SpatialHalo *prepare_pair(MFTL_BVH *ba, MFTL_BVH *bb)
{
  MFTL_SpatialHalo *const halo = MFTLSpatialHaloCreate(g_ens);
  MFTLSpatialHaloResolveRadii(g_ens, halo);

  MFTL_BVHSetGlobalRadius(ba, MFTLSpatialHaloRadius(halo, 0));
  MFTL_BVHSetGlobalRadius(bb, MFTLSpatialHaloRadius(halo, 1));

  MFTL_SpatialHaloRequest req;
  req.body_a = 0;
  req.body_b = 1;
  req.margin = MFTL_MAX(ba->tolerance, bb->tolerance);

  MFTLExtendSpatialHalo(g_ens, &req, 1, halo);
  MFTL_BVHSetHalo(bb, MFTLSpatialHaloPatch(halo, 1));
  return halo;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Tests
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The distributed answer must equal the serial one, as a SET.
 *
 * The reference is taken on rank 0 BEFORE redistribution, while
 * MFTLGenerateSphere's rank-0-holds-everything contract means one rank really
 * does have the whole mesh and the serial code path gives the true answer. The
 * mesh is then spread over the process grid and the same query is run through
 * the collective pipeline.
 *
 * Equality of the two sets is simultaneously the no-miss and the no-duplicate
 * assertion; see the file comment for why neither alone would be enough.
 */
static int test_distributed_contacts_match_the_serial_answer(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 1.6, 2);  // radius 1 spheres 1.6 apart: a substantial overlap
  refresh(g_ens);

  /* ── The reference, on the rank that holds everything ─────────────────── */
  int n_ref = 0;
  long long *ref = MFTL_NULL;
  if (g_rank == 0)
  {
    MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
    MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);

    contact_set s = {MFTL_NULL, 0, 0, ba, bb};
    MFTL_IntersectObjects(a, ba, b, bb, record_contact, &s);

    qsort(s.keys, (size_t) s.n, sizeof(long long), cmp_ll);
    ref = s.keys;
    n_ref = s.n;

    MFTL_DestroyBVH(ba);
    MFTL_DestroyBVH(bb);
  }
  MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);

  /* The reference has to be non-empty or every assertion below is vacuous --
   * two separated spheres would "agree" trivially. */
  TEST_ASSERT(agreed(n_ref > 0, MPI_COMM_WORLD));

  /* ── Spread the mesh, then ask again ──────────────────────────────────── */
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  a = bodyv(0);  /* the pointers do not survive a redistribution */
  b = bodyv(1);
  refresh(g_ens);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
  MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

  /* THE HALO MUST HAVE DONE SOMETHING. Without this the whole test degrades
   * quietly the day the decomposition, the sphere radius or the separation
   * changes so that every contact happens to land inside one rank: the sets
   * would still match, and the halo -- the only thing here that is not already
   * covered by the serial tests -- would no longer be exercised at all.
   *
   * The spheres straddle the x = 5 cut precisely so the contact region is cut by
   * a rank boundary at every process count this test is registered at. */
  {
    int local_halo = MFTLSpatialHaloCount(halo, 1);
    int total_halo = 0;
    MPI_Allreduce(&local_halo, &total_halo, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(total_halo > 0, MPI_COMM_WORLD));
  }

  contact_set s = {MFTL_NULL, 0, 0, ba, bb};
  MFTL_IntersectObjects(a, ba, b, bb, record_contact, &s);

  int n_got = 0;
  long long *got = gather_contacts(&s, &n_got);

  int same = 1;
  if (g_rank == 0)
  {
    if (n_got != n_ref)
    {
      printf("\n  [contacts] serial %d, distributed %d\n  ", n_ref, n_got);
      same = 0;
    }
    else
    {
      for (int i = 0; i < n_ref; ++i)
      {
        if (got[i] != ref[i])
        {
          printf("\n  [contacts] first difference at %d: expected %lld, got %lld\n  ", i, ref[i], got[i]);
          same = 0;
          break;
        }
      }
    }
    free(got);
    free(ref);
  }
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));

  free(s.keys);
  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief A halo patch must be a VALID, self-contained MFTL_Object.
 *
 * This is the property the whole shadow-object design exists to provide, and it
 * is the reason the arrivals are not simply nine loose doubles: a patch that
 * satisfies the ordinary mesh invariants can be handed to any pass in the
 * library that accepts an MFTL_Object -- geometry, normals, output --
 * without that pass knowing what a spatial halo is.
 *
 * Three things are checked and each rules out a different mistake:
 *
 *   inv_mesh_is_sound        implicit indexing, twin symmetry, out_he really
 *                            outgoing, derived arrays agreeing with the records,
 *                            ownership prefix. Catches a bad vertex rotation, a
 *                            dangling half-edge, an unbuilt derived array.
 *   n_local == 0             this rank owns none of it. If the patch claimed
 *                            ownership, MFTLEnsembleUpdate-style passes would
 *                            fold another rank's triangles into local statistics.
 *   ordinary passes run      MFTLComputeTriangleGeometry and
 *                            MFTLComputeVertexNormals must simply work on it.
 */
static int test_halo_patch_is_a_valid_object(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 1.6, 2);
  refresh(g_ens);

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = bodyv(0);  /* the pointers do not survive a redistribution */

  b = bodyv(1);
  refresh(g_ens);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
  MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

  const MFTL_Object *const patch = MFTLSpatialHaloPatch(halo, 1);

  int ok = 1;
  int have_patch = (patch != MFTL_NULL && patch->tris.n > 0) ? 1 : 0;

  if (have_patch)
  {
    MFTL_Object *const mut = (MFTL_Object *) patch;

    ok = ok && inv_mesh_is_sound(mut, g_rank);
    ok = ok && (mut->tris.n_local == 0);
    ok = ok && (mut->verts.n_local == 0);
    ok = ok && (mut->domain == &g_ens->domain);

    /* Every triangle must name a body-B triangle that some rank really owns, and
     * must NOT be one this rank already has -- the halo is what this rank is
     * missing, and a copy of something local would be reported twice. */
    for (int t = 0; t < mut->tris.n && ok; ++t)
    {
      const int gid = mut->tris.rec[t].global_id;
      if (gid < 0 || gid >= b->global_n_triangles) ok = 0;
      if (mut->tris.rec[t].owner_proc == g_rank) ok = 0;
      for (int u = 0; u < b->tris.n_local; ++u)
      {
        if (b->tris.rec[u].global_id == gid)
        {
          ok = 0;
          break;
        }
      }
    }

    /* And the ordinary passes must run on it. */
    MFTLComputeTriangleGeometry(mut, mut->tris.n);
    MFTLComputeVertexNormals(mut);
    ok = ok && inv_triangle_normals_are_unit(mut, 1e-9);
  }

  /* At least one rank must actually have a patch, or this test proves nothing. */
  int any = 0;
  MPI_Allreduce(&have_patch, &any, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(any > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief The contact envelope must not depend on the process count.
 *
 * `bounding_radius` is measured from the primitives a rank holds, which is a
 * fraction of the body -- so scaling the envelope by it would make a droplet
 * split over eight ranks use a smaller envelope on each of them than the same
 * droplet on one rank, and the same simulation would then detect contact at a
 * different separation depending only on how it was launched.
 *
 * MFTLSpatialHaloResolveRadii replaces it with the body's global radius. This
 * asserts that every rank ended up with the SAME tolerance, and that it is the
 * one the whole body implies rather than a partition's.
 */
static int test_tolerance_is_independent_of_the_partition(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-2;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 1.6, 2);
  refresh(g_ens);

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = bodyv(0);  /* the pointers do not survive a redistribution */

  b = bodyv(1);
  refresh(g_ens);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
  MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

  double lo = 0.0, hi = 0.0;
  MPI_Allreduce(&ba->tolerance, &lo, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&ba->tolerance, &hi, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  // Same on every rank...
  TEST_ASSERT(agreed(fabs(hi - lo) <= 1e-15 * MFTL_MAX(hi, 1.0), MPI_COMM_WORLD));
  // ...and equal to 1 % of the WHOLE sphere's radius, not of one rank's slice.
  TEST_ASSERT(agreed(fabs(lo - 1.0e-2) < 1.0e-3, MPI_COMM_WORLD));

  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief Separated bodies must produce no contacts and no hang.
 *
 * The zero-traffic path through every collective. A rank whose partition holds
 * nothing of either body still has to enter the reduction, the allgather and the
 * exchange -- which is why MFTL_GenerateLBVH returns an empty hierarchy rather
 * than MFTL_NULL, and why the request list is built from the object list rather
 * than from what a rank happens to hold.
 */
static int test_separated_bodies_are_silent_and_do_not_hang(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-3;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 6.0, 2);  // radius 1, centres 6 apart
  refresh(g_ens);

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = bodyv(0);  /* the pointers do not survive a redistribution */

  b = bodyv(1);
  refresh(g_ens);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
  MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

  contact_set s = {MFTL_NULL, 0, 0, ba, bb};
  MFTL_IntersectObjects(a, ba, b, bb, record_contact, &s);
  const int hits = s.n;
  free(s.keys);

  int total = 0;
  MPI_Allreduce(&hits, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(total == 0, MPI_COMM_WORLD));

  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief The ensemble entry point must agree with the explicit pipeline.
 *
 * MFTLEnsembleIntersectObjects builds the request list itself, one entry per
 * unordered pair, and that list has to be identical on every rank or the
 * collectives inside the exchange desynchronise. This runs it for real at the
 * test's process count -- a mismatch hangs here rather than in production.
 */
static int test_ensemble_pipeline_matches_the_explicit_one(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 1.6, 2);
  refresh(g_ens);

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = bodyv(0);  /* the pointers do not survive a redistribution */

  b = bodyv(1);
  refresh(g_ens);

  int n_explicit = 0;
  {
    MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
    MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
    MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

    contact_set s = {MFTL_NULL, 0, 0, ba, bb};
    MFTL_IntersectObjects(a, ba, b, bb, record_contact, &s);
    n_explicit = s.n;
    free(s.keys);

    MFTL_DestroyBVH(ba);
    MFTL_DestroyBVH(bb);
    MFTLSpatialHaloDestroy(&halo);
  }

  int n_pipeline = 0;
  TEST_ASSERT(agreed(MFTLEnsembleIntersectObjects(g_ens, &p, count_only, &n_pipeline) == MFTL_SUCCESS, MPI_COMM_WORLD));

  int tot_explicit = 0, tot_pipeline = 0;
  MPI_Allreduce(&n_explicit, &tot_explicit, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&n_pipeline, &tot_pipeline, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  TEST_ASSERT(agreed(tot_explicit > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(tot_explicit == tot_pipeline, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief The real bodies must be untouched by the halo exchange.
 *
 * The whole reason the arrivals go into a shadow object is that appending them
 * to the body's own ghost suffix would silently change four passes that loop to
 * `tris.n` -- the Morton bounding box, MFTLMeanEdgeLength, the vertex-normal
 * scatter and MFTLRebuildTwins. This is the assertion that the chosen design
 * actually delivers that: entity counts and the partition invariants must be
 * bit-for-bit what they were before the exchange.
 */
static int test_halo_does_not_touch_the_bodies(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  MFTL_Object *a = MFTL_NULL;
  MFTL_Object *b = MFTL_NULL;
  make_pair(&a, &b, 1.6, 2);
  refresh(g_ens);

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  a = bodyv(0);  /* the pointers do not survive a redistribution */

  b = bodyv(1);
  refresh(g_ens);

  const int va = a->verts.n, ta = a->tris.n, vla = a->verts.n_local, tla = a->tris.n_local;
  const int vb = b->verts.n, tb = b->tris.n, vlb = b->verts.n_local, tlb = b->tris.n_local;

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);
  MFTL_SpatialHalo *halo = prepare_pair(ba, bb);

  TEST_ASSERT(agreed(a->verts.n == va && a->tris.n == ta && a->verts.n_local == vla && a->tris.n_local == tla, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(b->verts.n == vb && b->tris.n == tb && b->verts.n_local == vlb && b->tris.n_local == tlb, MPI_COMM_WORLD));

  /* And both bodies still satisfy every partition invariant -- in particular
   * inv_local_global_ids_are_unique and inv_halo_is_vertex_complete, which are
   * exactly the two an appended spatial ghost would have broken. */
  TEST_ASSERT(agreed(inv_partition_is_sound(a, g_rank), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(inv_partition_is_sound(b, g_rank), MPI_COMM_WORLD));

  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/* ───────────────────────────────────────────────────────────────────────── */

/** @brief A fresh ensemble per test: each one builds its own bodies. */
static void ensemble_reset(void)
{
  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);

  int cuts[3] = {1, 1, 1};
  if (g_nproc == 2)
  {
    cuts[0] = 2;
  }
  else if (g_nproc == 4)
  {
    cuts[0] = 2;
    cuts[1] = 2;
  }
  else if (g_nproc == 8)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 2;
  }

  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {10.0, 10.0, 10.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 8);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("--- Running MFTL BVH Parallel Tests (np = %d) ---\n", g_nproc);

  ensemble_reset();
  RUN_TEST(test_distributed_contacts_match_the_serial_answer);

  ensemble_reset();
  RUN_TEST(test_halo_patch_is_a_valid_object);

  ensemble_reset();
  RUN_TEST(test_halo_does_not_touch_the_bodies);

  ensemble_reset();
  RUN_TEST(test_tolerance_is_independent_of_the_partition);

  ensemble_reset();
  RUN_TEST(test_separated_bodies_are_silent_and_do_not_hang);

  ensemble_reset();
  RUN_TEST(test_ensemble_pipeline_matches_the_explicit_one);

  if (g_rank == 0) TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();

  return (g_tests_failed > 0) ? 1 : 0;
}
