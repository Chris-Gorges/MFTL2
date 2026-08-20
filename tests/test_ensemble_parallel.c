/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Multi-rank tests for the ensemble (launched at 8 ranks).
 *
 * At nproc == 1 the collectives are called correctly but almost nothing they do
 * is exercised: neighbour discovery is degenerate, a SUM reduces one value, and
 * a MIN/MAX never has to choose. This file exists for what only shows up at
 * P > 1.
 *
 * EVERY assertion here is made on an MPI_Allreduce'd verdict (see agreed()).
 * TEST_ASSERT returns immediately, so an assertion that fails on only some ranks
 * would leave the others blocked in the next collective -- a hang instead of a
 * clean failure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;
static int g_rank = 0;
static int g_nproc = 1;

/* ─────────────────────────────────────────────────────────────────────────────
 * Domain topology at P > 1
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_every_rank_agrees_on_the_rank_map(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  /* The rank map is a lookup table used to route entities. If two ranks
   * disagreed about a single cell, they would disagree about where a triangle
   * belongs -- and each would be internally consistent about it. */
  int *mine = (int *) malloc((size_t) d->nproc * sizeof(int));
  int *root = (int *) malloc((size_t) d->nproc * sizeof(int));
  for (int i = 0; i < d->nproc; ++i) mine[i] = root[i] = d->rank_map[i];

  MPI_Bcast(root, d->nproc, MPI_INT, 0, d->comm_world);

  int ok = 1;
  for (int i = 0; i < d->nproc; ++i)
  {
    if (mine[i] != root[i]) ok = 0;
  }
  free(mine);
  free(root);

  TEST_ASSERT(agreed(ok, d->comm_world));
  return 0;
}

static int test_owner_lookup_is_a_partition(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  /* Sample the domain on a fine grid and count how many points each rank claims.
   * Summed over ranks that must equal the number of samples exactly: every point
   * belongs to exactly one rank, no gaps and no overlaps. */
  const int N = 20;
  int *claimed = (int *) calloc((size_t) d->nproc, sizeof(int));

  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < N; ++k)
      {
        double p[3] = {d->global_min[0] + (i + 0.5) * d->domain_size[0] / N, d->global_min[1] + (j + 0.5) * d->domain_size[1] / N,
                       d->global_min[2] + (k + 0.5) * d->domain_size[2] / N};
        const int owner = MFTLGetOwnerProc(d, p);
        if (owner < 0 || owner >= d->nproc)
        {
          free(claimed);
          TEST_ASSERT(agreed(0, d->comm_world));
          return 1;
        }
        claimed[owner]++;
      }
    }
  }

  int total = 0;
  int mine = claimed[d->rank];
  for (int r = 0; r < d->nproc; ++r) total += claimed[r];
  free(claimed);

  int ok = (total == N * N * N);
  // With a 2x2x2 grid over a symmetric domain each rank owns exactly an eighth.
  ok = ok && (mine == N * N * N / d->nproc);

  TEST_ASSERT(agreed(ok, d->comm_world));
  return 0;
}

static int test_neighbour_graph_is_symmetric(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  /* MFTL assumes an undirected neighbourhood -- if I send to you, you expect
   * from me. A neighbourhood collective on an asymmetric graph deadlocks, so
   * this is a precondition rather than a nicety. */
  int *is_neighbour = (int *) calloc((size_t) d->nproc, sizeof(int));
  for (int i = 0; i < d->num_neighbors; ++i) is_neighbour[d->neighbors[i]] = 1;

  int *theirs = (int *) malloc((size_t) d->nproc * (size_t) d->nproc * sizeof(int));
  MPI_Allgather(is_neighbour, d->nproc, MPI_INT, theirs, d->nproc, MPI_INT, d->comm_world);

  int ok = 1;
  for (int a = 0; a < d->nproc; ++a)
  {
    for (int b = 0; b < d->nproc; ++b)
    {
      if (theirs[a * d->nproc + b] != theirs[b * d->nproc + a]) ok = 0;
    }
    // Nobody is their own neighbour.
    if (theirs[a * d->nproc + a] != 0) ok = 0;
  }

  free(is_neighbour);
  free(theirs);

  TEST_ASSERT(agreed(ok, d->comm_world));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Reductions with genuinely different values per rank
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_global_counts_sum_unequal_ranks(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){2, 2, 2}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 2);
  MFTL_Object *a = MFTLObjectCreate(1);
  MFTL_Object *b = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(e, a);
  MFTLEnsembleAddObject(e, b);

  /* Deliberately unequal, and with ghosts, so the reduction cannot be right by
   * accident: rank r owns r+1 triangles of object A but holds 100 more as
   * ghosts. Only the owned prefix may be counted, or every shared entity is
   * counted once per rank that sees it. */
  MFTLObjectSetCounts(a, 3 * (g_rank + 1) + 50, 3 * (g_rank + 1), g_rank + 1 + 100, g_rank + 1);
  MFTLObjectSetCounts(b, 10, 10, 4, 4);

  MFTLEnsembleUpdateGlobalCounts(e);

  const int expect_a = g_nproc * (g_nproc + 1) / 2;  // sum of 1..nproc
  int ok = (a->global_n_triangles == expect_a);
  ok = ok && (a->global_n_vertices == 3 * expect_a);
  ok = ok && (b->global_n_triangles == 4 * g_nproc);
  ok = ok && (e->total_global_triangles == expect_a + 4 * g_nproc);

  TEST_ASSERT(agreed(ok, e->domain.comm_world));

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_update_reduces_min_and_max_across_ranks(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){2, 2, 2}, (double[3]){-100, -100, -100}, (double[3]){100, 100, 100}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);

  /* Every rank builds the same square scaled by (rank + 1), so the areas differ
   * by rank and the global min/max have unambiguous, hand-computable values:
   * area = s^2 / 2 with s = rank + 1. */
  const double s = (double) (g_rank + 1);
  const double pos[12] = {0, 0, 0, s, 0, 0, s, s, 0, 0, s, 0};
  const int tv[6] = {0, 1, 2, 0, 2, 3};
  MFTLBuildIndexedMesh(obj, pos, 4, tv, 2);

  MFTLEnsembleUpdate(e);

  const double expect_min = 0.5;  // rank 0, s = 1
  const double expect_max = 0.5 * (double) g_nproc * (double) g_nproc;

  int ok = fabs(obj->quality.min_triangle_area - expect_min) < 1e-9;
  ok = ok && fabs(obj->quality.max_triangle_area - expect_max) < 1e-9;

  /* Quality is scale invariant, so every rank measures the same value and the
   * reduction must not perturb it -- a good check that MIN and MAX are not
   * accidentally reading each other's buffer. */
  ok = ok && fabs(obj->quality.min_aspect_ratio - sqrt(3.0) / 2.0) < 1e-9;
  ok = ok && fabs(obj->quality.max_aspect_ratio - sqrt(3.0) / 2.0) < 1e-9;

  /* The average must be weighted by the GLOBAL triangle count, not averaged over
   * per-rank averages. Here both give the same answer only because every rank
   * holds two triangles, so check the value itself. */
  double sum = 0.0;
  for (int r = 0; r < g_nproc; ++r) sum += 2.0 * 0.5 * (double) (r + 1) * (double) (r + 1);
  ok = ok && fabs(obj->quality.avg_triangle_area - sum / (2.0 * g_nproc)) < 1e-9;

  TEST_ASSERT(agreed(ok, e->domain.comm_world));

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_update_ignores_ranks_that_hold_nothing(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){2, 2, 2}, (double[3]){-100, -100, -100}, (double[3]){100, 100, 100}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);

  /* Only rank 0 holds any of this object -- the state right after generation.
   * The MIN identity contributed by every other rank must not win the
   * reduction, which is why the accumulators start at +/- DBL_MAX rather than 0. */
  if (g_rank == 0)
  {
    const double pos[12] = {0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 2, 0};
    const int tv[6] = {0, 1, 2, 0, 2, 3};
    MFTLBuildIndexedMesh(obj, pos, 4, tv, 2);
  }
  else
  {
    MFTLObjectBuildDerived(obj);
  }

  MFTLEnsembleUpdate(e);

  int ok = fabs(obj->quality.min_triangle_area - 2.0) < 1e-9;
  ok = ok && fabs(obj->quality.max_triangle_area - 2.0) < 1e-9;
  ok = ok && fabs(obj->quality.avg_triangle_area - 2.0) < 1e-9;
  ok = ok && (obj->quality.min_triangle_area < 1e30);  // not a leaked DBL_MAX

  TEST_ASSERT(agreed(ok, e->domain.comm_world));

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_repeated_updates_are_stable(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){2, 2, 2}, (double[3]){-8, -8, -8}, (double[3]){8, 8, 8}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);
  double c[3] = {0, 0, 0};
  MFTLGenerateSphere(obj, c, 1.0, 2);

  /* Three non-blocking collectives per call, waited on together. Calling it in a
   * loop is the shape a solver uses it in, and it is where an unmatched request
   * or a reused buffer would show up as a hang or as drifting values. */
  MFTLEnsembleUpdate(e);
  const double a0 = obj->quality.avg_triangle_area;
  const double q0 = obj->quality.min_aspect_ratio;

  int ok = 1;
  for (int i = 0; i < 20; ++i)
  {
    MFTLEnsembleUpdate(e);
    if (fabs(obj->quality.avg_triangle_area - a0) > 1e-15) ok = 0;
    if (fabs(obj->quality.min_aspect_ratio - q0) > 1e-15) ok = 0;
  }

  TEST_ASSERT(agreed(ok, e->domain.comm_world));

  MFTLEnsembleDestroy(&e);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  int cuts[3] = {2, 2, 2};
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {true, true, true};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

  if (g_ens == MFTL_NULL)
  {
    if (g_rank == 0) fprintf(stderr, "test_ensemble_parallel must be launched with 8 ranks.\n");
    MPI_Finalize();
    return 1;
  }

  if (g_rank == 0) printf("--- Running MFTL Ensemble Tests (%d ranks) ---\n", g_nproc);

  RUN_TEST(test_every_rank_agrees_on_the_rank_map);
  RUN_TEST(test_owner_lookup_is_a_partition);
  RUN_TEST(test_neighbour_graph_is_symmetric);
  RUN_TEST(test_global_counts_sum_unequal_ranks);
  RUN_TEST(test_update_reduces_min_and_max_across_ranks);
  RUN_TEST(test_update_ignores_ranks_that_hold_nothing);
  RUN_TEST(test_repeated_updates_are_stable);

  if (g_rank == 0) TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
