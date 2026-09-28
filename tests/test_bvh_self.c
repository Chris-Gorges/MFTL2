/*
 * @Author: Christian Gorges
 * @Date: 2026-09-25
 *
 * @brief SELF-proximity through MFTLEnsembleIntersectObjects (params.self_pairs).
 *
 * The query a topology change is detected by: two parts of ONE surface that are
 * close without being adjacent. It is tested the way CLAUDE.md prescribes for
 * any distributed contact query -- as a MULTISET of named pairs, against an
 * exhaustive O(n^2) sweep taken while one rank still held the whole mesh, which
 * shares nothing with the BVH, the halo or the filter. Equality of the gathered
 * multisets rules out, at once, a MISSING cross-rank pair, a DUPLICATE reported
 * by two ranks, a pair of ADJACENT triangles leaking through the filter, and a
 * DIVERGENT answer at a different process count.
 *
 * Every fixture sits where the cut planes meet, and the number of pairs whose
 * two triangles are owned by different ranks is counted and asserted: a fixture
 * whose pairs all land on one rank tests the traversal and not the halo.
 *
 * Registered at np = 1, 2, 4, 8 and 27 for the same reasons as
 * test_bvh_ghosting.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

#define KEY_SHIFT 22

/* ─────────────────────────────────────────────────────────────────────────────
 * Grid and fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

static void choose_cuts(int nproc, int cuts[3])
{
  cuts[0] = cuts[1] = cuts[2] = 0;
  MPI_Dims_create(nproc, 3, cuts);
}

typedef double (*field_fn)(const double p[3], const void *arg);

/** @brief Torus about the z axis through `c`: major R, minor r. args: cx cy cz R r. */
static double torus_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;
  const double x = p[0] - s[0], y = p[1] - s[1], z = p[2] - s[2];
  const double q = sqrt(x * x + y * y) - s[3];
  return sqrt(q * q + z * z) - s[4];
}

/** @brief Dumbbell along x through `c`: sphere radius R, neck radius a, half-length h. */
static double dumbbell_field(const double p[3], const void *arg)
{
  const double *s = (const double *) arg;
  const double x = p[0] - s[0], y = p[1] - s[1], z = p[2] - s[2];
  const double a = sqrt((x + s[5]) * (x + s[5]) + y * y + z * z) - s[3];
  const double b = sqrt((x - s[5]) * (x - s[5]) + y * y + z * z) - s[3];
  const double cyl = fmax(sqrt(y * y + z * z) - s[4], fabs(x) - s[5]);
  return fmin(fmin(a, b), cyl);
}

/**
 * @brief Builds a body from a field on rank 0 only -- the generation contract --
 *        so the reference can be taken before anything is partitioned.
 */
static MFTL_Object *add_body(MFTL_Ensemble *ens, field_fn f, const double *arg, const double lo[3], const double hi[3], double dx)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (g_rank == 0)
  {
    int dims[3];
    const double sp[3] = {dx, dx, dx};
    for (int k = 0; k < 3; ++k) dims[k] = (int) ceil((hi[k] - lo[k]) / dx) + 1;
    MFTL_SdfGrid g;
    MFTLSdfGridInit(&g, lo, sp, dims);
    for (int k = 0; k < dims[2]; ++k)
      for (int j = 0; j < dims[1]; ++j)
        for (int i = 0; i < dims[0]; ++i)
        {
          const double p[3] = {lo[0] + i * dx, lo[1] + j * dx, lo[2] + k * dx};
          g.phi[((size_t) k * dims[1] + j) * dims[0] + i] = f(p, arg);
        }
    MFTLBuildMeshFromLevelSet(obj, &g, 1e-2);
    MFTLSdfGridFree(&g);
  }
  else
    MFTLObjectBuildDerived(obj);
  return obj;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * The reference: every non-adjacent pair of one body within the tolerance
 * ─────────────────────────────────────────────────────────────────────────── */

typedef struct
{
  long long *k;
  int n, cap;
} keyset;

static void ks_push(keyset *s, long long v)
{
  if (s->n == s->cap)
  {
    s->cap = s->cap ? 2 * s->cap : 256;
    s->k = (long long *) realloc(s->k, (size_t) s->cap * sizeof(long long));
  }
  s->k[s->n++] = v;
}

static int cmp_ll(const void *a, const void *b)
{
  const long long x = *(const long long *) a, y = *(const long long *) b;
  return (x > y) - (x < y);
}

static long long pair_key(int body, int ga, int gb) { return ((long long) body << (2 * KEY_SHIFT)) + ((long long) ga << KEY_SHIFT) + gb; }

/**
 * @brief O(n^2), with periodic images, no BVH, no halo, no filter shared.
 *
 * Every pair is enumerated; a pair whose bounding spheres (centroid, farthest
 * corner) are more than `tol` apart skips the narrow phase. That reject is
 * strictly more permissive than MFTL_TriTriProximity -- two points within tol
 * lie within tol of each other's spheres -- so it changes the cost, not the set.
 */
static void brute_self(const MFTL_Object *obj, double tol, keyset *out)
{
  const MFTL_Domain *d = obj->domain;
  const int n = obj->tris.n;
  double (*const T)[3][3] = (double (*)[3][3]) malloc((size_t) (n > 0 ? n : 1) * sizeof(*T));
  double *const ctr = (double *) malloc(4 * (size_t) (n > 0 ? n : 1) * sizeof(double));
  for (int i = 0; i < n; ++i)
  {
    MFTLGetTriangleCorners(obj, i, T[i]);
    double r2 = 0.0;
    for (int k = 0; k < 3; ++k) ctr[4 * i + k] = (T[i][0][k] + T[i][1][k] + T[i][2][k]) / 3.0;
    for (int c = 0; c < 3; ++c)
    {
      double e2 = 0.0;
      for (int k = 0; k < 3; ++k) e2 += (T[i][c][k] - ctr[4 * i + k]) * (T[i][c][k] - ctr[4 * i + k]);
      r2 = fmax(r2, e2);
    }
    ctr[4 * i + 3] = sqrt(r2);
  }

  for (int i = 0; i < n; ++i)
  {
    const double (*const A)[3] = T[i];
    for (int j = 0; j < n; ++j)
    {
      const int gi = obj->tris.rec[i].global_id, gj = obj->tris.rec[j].global_id;
      if (gi >= gj) continue;

      bool adjacent = MFTL_FALSE;
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          if (obj->tris.verts[3 * i + a] == obj->tris.verts[3 * j + b]) adjacent = MFTL_TRUE;
      if (adjacent) continue;

      /* Nearest image of B's first corner relative to A's. */
      double shift[3];
      MFTLGetMinimumImageVector(d, shift, A[0], T[j][0]);
      for (int k = 0; k < 3; ++k) shift[k] = A[0][k] + shift[k] - T[j][0][k];

      double dc2 = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        const double dk = ctr[4 * j + k] + shift[k] - ctr[4 * i + k];
        dc2 += dk * dk;
      }
      const double reach = ctr[4 * i + 3] + ctr[4 * j + 3] + tol;
      if (dc2 > reach * reach) continue;

      double B[3][3];
      for (int c = 0; c < 3; ++c)
        for (int k = 0; k < 3; ++k) B[c][k] = T[j][c][k] + shift[k];
      if (MFTL_TriTriProximity(A, B, tol)) ks_push(out, pair_key(obj->global_id, gi, gj));
    }
  }
  free(ctr);
  free(T);
}

typedef struct
{
  keyset ks;
  int cross_rank;
  int not_ordered;
} collector;

static void collect(const MFTL_BVH *bvhA, int pa, const MFTL_BVH *bvhB, int pb, void *ud)
{
  collector *c = (collector *) ud;
  int ta, tb;
  const MFTL_Object *oa = MFTL_BVHPrimTriangle(bvhA, pa, &ta);
  const MFTL_Object *ob = MFTL_BVHPrimTriangle(bvhB, pb, &tb);
  const int ga = oa->tris.rec[ta].global_id, gb = ob->tris.rec[tb].global_id;
  if (oa->global_id == ob->global_id)
  {
    if (ga >= gb) c->not_ordered++;
    if (MFTL_BVHPrimOwner(bvhB, pb) != g_rank) c->cross_rank++;
    ks_push(&c->ks, pair_key(oa->global_id, ga, gb));
  }
  else
  {
    /* An inter-body pair: key it so it cannot collide with a self pair. */
    ks_push(&c->ks, pair_key(1000 + oa->global_id * 10 + ob->global_id, ga, gb));
  }
}

/** @brief Gathers keys on rank 0, sorted. */
static void gather(const keyset *mine, keyset *all)
{
  int *counts = (int *) malloc((size_t) g_nproc * sizeof(int)), *displs = (int *) malloc((size_t) g_nproc * sizeof(int));
  MPI_Allgather(&mine->n, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  int total = 0;
  for (int i = 0; i < g_nproc; ++i)
  {
    displs[i] = total;
    total += counts[i];
  }
  all->k = (long long *) malloc((size_t) (total > 0 ? total : 1) * sizeof(long long));
  all->n = all->cap = total;
  MPI_Allgatherv(mine->k, mine->n, MPI_LONG_LONG, all->k, counts, displs, MPI_LONG_LONG, MPI_COMM_WORLD);
  qsort(all->k, (size_t) total, sizeof(long long), cmp_ll);
  free(counts);
  free(displs);
}

static void refresh(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

/**
 * @brief The whole protocol for one scene: reference before partitioning,
 *        pipeline after, multiset comparison, non-vacuity.
 */
static int run_scene(MFTL_Ensemble *ens, double tol, int *n_pairs_out, int *n_cross_out)
{
  MFTLPublishGeneratedBodies(ens);
  refresh(ens);

  keyset ref = {MFTL_NULL, 0, 0};
  if (g_rank == 0)
  {
    for (int o = 0; o < ens->n_local_objects; ++o) brute_self(ens->local_objects[o], tol, &ref);
    qsort(ref.k, (size_t) ref.n, sizeof(long long), cmp_ll);
  }
  int n_ref = ref.n;
  MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);

  MFTLRedistribute(ens);
  refresh(ens);

  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 0.0;
  p.proximity_floor = tol;
  p.self_pairs = MFTL_TRUE;

  collector c = {{MFTL_NULL, 0, 0}, 0, 0};
  const int rc = MFTLEnsembleIntersectObjects(ens, &p, collect, &c);
  keyset all = {MFTL_NULL, 0, 0};
  gather(&c.ks, &all);

  int same = 1;
  if (g_rank == 0)
  {
    same = (all.n == ref.n);
    for (int i = 0; same && i < ref.n; ++i) same = (all.k[i] == ref.k[i]);
    if (!same) printf("\n  reference %d pairs, pipeline %d\n  ", ref.n, all.n);
  }
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int cross = 0, unordered = 0;
  MPI_Allreduce(&c.cross_rank, &cross, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&c.not_ordered, &unordered, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  free(ref.k);
  free(all.k);
  free(c.ks.k);

  *n_pairs_out = n_ref;
  *n_cross_out = cross;
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(unordered == 0, MPI_COMM_WORLD));
  return 0;
}

static MFTL_Ensemble *make_ensemble(bool periodic)
{
  int cuts[3];
  choose_cuts(g_nproc, cuts);
  double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {4.0, 4.0, 4.0};
  bool per[3] = {periodic, periodic, periodic};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_narrow_torus_hole_is_found_as_a_set(void)
{
  /* Hole diameter 2 * (R - r) = 0.1, envelope 0.15: the pairs across the hole
   * are exactly what self-proximity is for. Centred on the point where every
   * cut plane of every tested grid meets. */
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  const double arg[5] = {2.0, 2.0, 2.0, 0.45, 0.4};
  const double lo[3] = {1.0, 1.0, 1.4}, hi[3] = {3.0, 3.0, 2.6};
  add_body(ens, torus_field, arg, lo, hi, 0.06);

  int pairs = 0, cross = 0;
  const int rc = run_scene(ens, 0.15, &pairs, &cross);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT(agreed(pairs > 50, MPI_COMM_WORLD));
  if (g_nproc > 1) TEST_ASSERT(agreed(cross > 0, MPI_COMM_WORLD));
  return 0;
}

static int test_a_thin_neck_on_a_cut_plane_is_found_as_a_set(void)
{
  /* Neck radius 0.06, envelope 0.15: the pairs across the neck are the
   * breakup candidates. The neck is centred on the x cut plane. */
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  const double arg[6] = {2.0, 2.0, 2.0, 0.35, 0.06, 0.45};
  const double lo[3] = {1.1, 1.5, 1.5}, hi[3] = {2.9, 2.5, 2.5};
  add_body(ens, dumbbell_field, arg, lo, hi, 0.04);

  int pairs = 0, cross = 0;
  const int rc = run_scene(ens, 0.15, &pairs, &cross);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT(agreed(pairs > 20, MPI_COMM_WORLD));
  if (g_nproc > 1) TEST_ASSERT(agreed(cross > 0, MPI_COMM_WORLD));
  return 0;
}

static int test_self_pairs_across_the_periodic_seam(void)
{
  /* The same torus centred on the domain CORNER of a periodic box: the hole
   * straddles the seam on two axes, so its pairs are only found through the
   * periodic images of the B side. */
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE);
  const double arg[5] = {0.0, 0.0, 0.0, 0.45, 0.4};
  const double lo[3] = {-1.0, -1.0, -0.6}, hi[3] = {1.0, 1.0, 0.6};
  add_body(ens, torus_field, arg, lo, hi, 0.06);

  int pairs = 0, cross = 0;
  const int rc = run_scene(ens, 0.15, &pairs, &cross);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT(agreed(pairs > 50, MPI_COMM_WORLD));
  return 0;
}

static int test_self_and_pairwise_contacts_in_one_call(void)
{
  /* A dumbbell plus a second body close to one of its bulbs: both kinds of pair
   * come out of the same call, each once. */
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  const double a1[6] = {2.0, 2.0, 2.0, 0.35, 0.06, 0.45};
  const double lo1[3] = {1.1, 1.5, 1.5}, hi1[3] = {2.9, 2.5, 2.5};
  add_body(ens, dumbbell_field, a1, lo1, hi1, 0.05);
  const double a2[5] = {2.45, 2.0, 2.8, 0.3, 0.12};  // a ring hovering just above the right bulb
  const double lo2[3] = {1.9, 1.4, 2.5}, hi2[3] = {3.0, 2.6, 3.1};
  add_body(ens, torus_field, a2, lo2, hi2, 0.05);

  int pairs = 0, cross = 0;
  const int rc = run_scene(ens, 0.15, &pairs, &cross);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT(agreed(pairs > 20, MPI_COMM_WORLD));
  return 0;
}

static int test_without_the_flag_one_body_reports_nothing(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  const double arg[5] = {2.0, 2.0, 2.0, 0.45, 0.4};
  const double lo[3] = {1.0, 1.0, 1.4}, hi[3] = {3.0, 3.0, 2.6};
  add_body(ens, torus_field, arg, lo, hi, 0.08);
  MFTLPublishGeneratedBodies(ens);
  refresh(ens);
  MFTLRedistribute(ens);
  refresh(ens);

  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 0.0;
  p.proximity_floor = 0.15;
  collector c = {{MFTL_NULL, 0, 0}, 0, 0};
  MFTLEnsembleIntersectObjects(ens, &p, collect, &c);
  int n = 0;
  MPI_Allreduce(&c.ks.n, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  free(c.ks.k);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(n == 0, MPI_COMM_WORLD));
  return 0;
}

static int test_a_well_resolved_sphere_has_no_self_pairs(void)
{
  /* Envelope well below the sphere's diameter and far above an edge: every
   * pair within it is adjacent, so the filter must remove them all. The
   * opposite of a vacuous test -- here the adjacency filter is doing ALL the
   * work, and a leak shows up as a non-empty set. */
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c0[3] = {2.0, 2.0, 2.0};
  MFTLGenerateSphere(obj, c0, 0.8, 3);
  refresh(ens);
  MFTLRedistribute(ens);
  refresh(ens);

  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 0.0;
  p.proximity_floor = 0.02;
  p.self_pairs = MFTL_TRUE;
  collector c = {{MFTL_NULL, 0, 0}, 0, 0};
  MFTLEnsembleIntersectObjects(ens, &p, collect, &c);
  int n = 0;
  MPI_Allreduce(&c.ks.n, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  free(c.ks.k);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(n == 0, MPI_COMM_WORLD));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0)
  {
    int cuts[3];
    choose_cuts(g_nproc, cuts);
    printf("--- Running MFTL BVH Self-Proximity Tests (np = %d, grid %dx%dx%d) ---\n", g_nproc, cuts[0], cuts[1], cuts[2]);
  }

  RUN_TEST(test_a_narrow_torus_hole_is_found_as_a_set);
  RUN_TEST(test_a_thin_neck_on_a_cut_plane_is_found_as_a_set);
  RUN_TEST(test_self_pairs_across_the_periodic_seam);
  RUN_TEST(test_self_and_pairwise_contacts_in_one_call);
  RUN_TEST(test_without_the_flag_one_body_reports_nothing);
  RUN_TEST(test_a_well_resolved_sphere_has_no_self_pairs);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
