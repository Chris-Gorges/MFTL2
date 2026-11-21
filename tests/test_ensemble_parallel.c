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

/**
 * @brief MFTLDomainNeighborIndex() is ONE table load (`MFTL_Domain::neighbor_slot`),
 *        and every module that routes by rank reads it, so a slot it gets wrong
 *        sends a ghost request, a routed block or a body vote to the wrong
 *        neighbour. Checked against a brute-force scan of `neighbors` (which
 *        shares nothing with the table) for every rank and two out-of-range
 *        values, and `cell` against MPI_Cart_coords() and `rank_map`, on grid
 *        shapes with and without the self-loop and with aliased offsets.
 */
static int test_the_neighbour_slot_table_inverts_the_neighbour_list(void)
{
  if (g_nproc != 8) return 0;
  static const int shapes[4][3] = {{2, 2, 2}, {2, 2, 2}, {8, 1, 1}, {4, 2, 1}};
  static const bool periodic[4] = {false, true, true, true};
  int ok = 1, saw_self_loop = 0, saw_non_neighbour = 0;
  for (int c = 0; c < 4; ++c)
  {
    const bool per[3] = {periodic[c], periodic[c], periodic[c]};
    MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, shapes[c], (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, per, 1);
    if (e == MFTL_NULL)
    {
      ok = 0;
      break;
    }
    const MFTL_Domain *d = &e->domain;
    for (int r = -1; r <= d->nproc; ++r)
    {
      int expect = MFTL_ERROR, hits = 0;
      for (int i = 0; i < d->num_neighbors; ++i)
        if (d->neighbors[i] == r)
        {
          if (hits++ == 0) expect = i;
        }
      ok &= (hits <= 1);  // the list holds each rank once
      ok &= (MFTLDomainNeighborIndex(d, r) == expect);
      if (r == d->rank && expect != MFTL_ERROR) saw_self_loop = 1;
      if (r >= 0 && r < d->nproc && r != d->rank && expect == MFTL_ERROR) saw_non_neighbour = 1;
    }
    int coords[3];
    MPI_Cart_coords(d->comm_cart, d->rank, 3, coords);
    for (int k = 0; k < 3; ++k) ok &= (d->cell[k] == coords[k]);
    ok &= (d->rank_map[d->cell[0] + d->cuts[0] * (d->cell[1] + d->cuts[1] * d->cell[2])] == d->rank);
    MFTLEnsembleDestroy(&e);
  }
  /* Non-vacuity: the periodic {8,1,1} and {4,2,1} carry the self-loop on every
   * rank and leave ranks outside the neighbourhood (on 2x2x2 every rank is a
   * neighbour). */
  int seen[2] = {saw_self_loop, saw_non_neighbour};
  MPI_Allreduce(MPI_IN_PLACE, seen, 2, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  ok &= seen[0] && seen[1];
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Which ranks a region reaches: MFTLDomainRanksOfCells / MFTLDomainRanksTouching
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief Grid shapes for the region queries: closed, periodic, slabs, mixed. */
static const int g_region_shapes[4][3] = {{2, 2, 2}, {2, 2, 2}, {8, 1, 1}, {4, 2, 1}};
static const bool g_region_periodic[4][3] = {{false, false, false}, {true, true, true}, {true, true, true}, {true, false, true}};

/** @brief A small deterministic generator (the queries are rank-local; every rank runs its own stream). */
static unsigned region_rand(unsigned *s)
{
  *s = *s * 1664525u + 1013904223u;
  return *s >> 8;
}

static double region_uniform(unsigned *s) { return (double) region_rand(s) / 16777216.0; }

static int cmp_int(const void *a, const void *b) { return (*(const int *) a > *(const int *) b) - (*(const int *) a < *(const int *) b); }

/**
 * @brief MFTLDomainRanksOfCells() lists the ranks of exactly the cells in the
 *        ranges, each once, ascending, and writes nothing when the capacity is
 *        short. Reference: every cell of the grid tested axis by axis, a
 *        periodic range by walking it cell by cell modulo cuts.
 */
static int test_a_cell_range_reaches_exactly_its_cells_ranks(void)
{
  if (g_nproc != 8) return 0;
  int ok = 1;
  long multi = 0, wrapped = 0, empty = 0;
  unsigned seed = 2026u + (unsigned) g_rank;
  for (int c = 0; c < 4; ++c)
  {
    MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, g_region_shapes[c], (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, g_region_periodic[c], 1);
    if (e == MFTL_NULL)
    {
      ok = 0;
      break;
    }
    const MFTL_Domain *d = &e->domain;
    for (int trial = 0; trial < 500; ++trial)
    {
      int lo[3], hi[3];
      bool leaves = MFTL_FALSE;
      for (int k = 0; k < 3; ++k)
      {
        const int cuts = d->cuts[k];
        lo[k] = (int) (region_rand(&seed) % (unsigned) (4 * cuts + 2)) - 2 * cuts - 1;
        hi[k] = lo[k] + (int) (region_rand(&seed) % (unsigned) (2 * cuts + 3)) - 1;
        leaves |= d->is_periodic[k] && (lo[k] < 0 || hi[k] >= cuts) && hi[k] >= lo[k];
      }

      int expect[8], ne = 0;
      for (int z = 0; z < d->cuts[2]; ++z)
        for (int y = 0; y < d->cuts[1]; ++y)
          for (int x = 0; x < d->cuts[0]; ++x)
          {
            const int cell[3] = {x, y, z};
            bool member = MFTL_TRUE;
            for (int k = 0; k < 3; ++k)
            {
              bool in = MFTL_FALSE;
              if (d->is_periodic[k])
              {
                for (int q = lo[k]; q <= hi[k] && !in; ++q) in = (((q % d->cuts[k]) + d->cuts[k]) % d->cuts[k] == cell[k]);
              }
              else
                in = (lo[k] <= cell[k] && cell[k] <= hi[k]);
              member &= in;
            }
            if (member) expect[ne++] = d->rank_map[x + d->cuts[0] * (y + d->cuts[1] * z)];
          }
      qsort(expect, (size_t) ne, sizeof(int), cmp_int);

      int got[8];
      for (int i = 0; i < 8; ++i) got[i] = -7;
      const int n = MFTLDomainRanksOfCells(d, lo, hi, got, 8);
      ok &= (n == ne);
      for (int i = 0; i < ne && n == ne; ++i) ok &= (got[i] == expect[i]);

      /* Short capacity: the count comes back, nothing is written. */
      if (ne > 1)
      {
        int small[8];
        for (int i = 0; i < 8; ++i) small[i] = -7;
        ok &= (MFTLDomainRanksOfCells(d, lo, hi, small, ne - 1) == ne);
        for (int i = 0; i < 8; ++i) ok &= (small[i] == -7);
      }
      multi += (ne > 1);
      wrapped += leaves && ne > 0;
      empty += (ne == 0);
    }
    MFTLEnsembleDestroy(&e);
  }
  long seen[3] = {multi, wrapped, empty};
  MPI_Allreduce(MPI_IN_PLACE, seen, 3, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  ok &= seen[0] > 0 && seen[1] > 0 && seen[2] > 0;  // non-vacuity on every rank
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief MFTLDomainRanksTouching() is the owner rule applied to a box: the
 *        owner (MFTLGetOwnerProc()) of every sampled point of the box is in the
 *        answer, every rank in it has a cell that meets the box (a wall's
 *        boundary cells reaching to infinity, a periodic box in any image), and a
 *        box shrunk to a point -- on a cell face too -- reaches exactly that
 *        point's owner.
 */
static int test_a_box_reaches_exactly_the_owners_of_its_points(void)
{
  if (g_nproc != 8) return 0;
  const double gmin[3] = {-1.0, -0.5, 0.0}, gmax[3] = {1.0, 1.5, 3.0};
  int ok = 1;
  long multi = 0, beyond = 0, whole = 0, faces = 0;
  unsigned seed = 77u + 13u * (unsigned) g_rank;
  for (int c = 0; c < 4; ++c)
  {
    MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, g_region_shapes[c], gmin, gmax, g_region_periodic[c], 1);
    if (e == MFTL_NULL)
    {
      ok = 0;
      break;
    }
    const MFTL_Domain *d = &e->domain;
    for (int trial = 0; trial < 400; ++trial)
    {
      double lo[3], hi[3];
      const bool point = (trial % 4 == 0);
      bool out = MFTL_FALSE, spans = MFTL_FALSE;
      for (int k = 0; k < 3; ++k)
      {
        const double L = gmax[k] - gmin[k], h = L / d->cuts[k];
        if (point)
        {
          /* Inside the box, half of them exactly on a cell face. */
          const double x = (trial % 8 == 0) ? gmin[k] + h * (double) (region_rand(&seed) % (unsigned) d->cuts[k]) : gmin[k] + L * region_uniform(&seed);
          lo[k] = hi[k] = x;
          continue;
        }
        const double ctr = gmin[k] - 0.5 * L + 2.0 * L * region_uniform(&seed);
        const double hw = 0.7 * L * region_uniform(&seed);
        lo[k] = ctr - hw;
        hi[k] = ctr + hw;
        out |= !d->is_periodic[k] && (lo[k] > gmax[k] || hi[k] < gmin[k]);
        spans |= d->is_periodic[k] && hi[k] - lo[k] >= L;
      }

      int got[8];
      const int n = MFTLDomainRanksTouching(d, lo, hi, got, 8);
      ok &= (n >= 1 && n <= 8);
      if (n < 1 || n > 8) continue;
      for (int i = 1; i < n; ++i) ok &= (got[i - 1] < got[i]);

      /* Complete: the owner of every sampled point. */
      for (int s = 0; s < 125; ++s)
      {
        double p[3];
        const int t[3] = {s % 5, (s / 5) % 5, s / 25};
        for (int k = 0; k < 3; ++k)
        {
          p[k] = lo[k] + 0.25 * t[k] * (hi[k] - lo[k]);
          if (d->is_periodic[k] && (p[k] < gmin[k] || p[k] >= gmax[k]))  // a point inside stays bit for bit
          {
            const double L = gmax[k] - gmin[k];
            p[k] = gmin[k] + fmod(p[k] - gmin[k], L);
            if (p[k] < gmin[k]) p[k] += L;
            if (p[k] >= gmax[k]) p[k] = gmin[k];
          }
        }
        const int owner = MFTLGetOwnerProc(d, p);
        bool found = MFTL_FALSE;
        for (int i = 0; i < n; ++i) found |= (got[i] == owner);
        ok &= found;
        if (point) ok &= (n == 1 && got[0] == owner);
      }

      /* Tight: every rank listed has a cell meeting the box. */
      for (int i = 0; i < n; ++i)
      {
        int cell[3] = {-1, -1, -1};
        for (int q = 0; q < d->nproc; ++q)
          if (d->rank_map[q] == got[i])
          {
            cell[0] = q % d->cuts[0];
            cell[1] = (q / d->cuts[0]) % d->cuts[1];
            cell[2] = q / (d->cuts[0] * d->cuts[1]);
          }
        for (int k = 0; k < 3; ++k)
        {
          const double L = gmax[k] - gmin[k], h = L / d->cuts[k], eps = 1e-12 * L;
          double clo = gmin[k] + cell[k] * h - eps, chi = gmin[k] + (cell[k] + 1) * h + eps;
          bool meets = MFTL_FALSE;
          if (d->is_periodic[k])
          {
            for (int m = -4; m <= 4 && !meets; ++m) meets = (lo[k] + m * L <= chi && hi[k] + m * L >= clo);
          }
          else
          {
            if (cell[k] == 0) clo = -HUGE_VAL;
            if (cell[k] == d->cuts[k] - 1) chi = HUGE_VAL;
            meets = (lo[k] <= chi && hi[k] >= clo);
          }
          ok &= meets;
        }
      }
      multi += (n > 1);
      beyond += out;
      whole += spans;
      faces += point && (trial % 8 == 0);
    }
    MFTLEnsembleDestroy(&e);
  }
  long seen[4] = {multi, beyond, whole, faces};
  MPI_Allreduce(MPI_IN_PLACE, seen, 4, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  ok &= seen[0] > 0 && seen[1] > 0 && seen[2] > 0 && seen[3] > 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
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
  ok = ok && fabs(obj->quality.min_mean_ratio - sqrt(3.0) / 2.0) < 1e-9;
  ok = ok && fabs(obj->quality.max_mean_ratio - sqrt(3.0) / 2.0) < 1e-9;

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
  const double q0 = obj->quality.min_mean_ratio;

  int ok = 1;
  for (int i = 0; i < 20; ++i)
  {
    MFTLEnsembleUpdate(e);
    if (fabs(obj->quality.avg_triangle_area - a0) > 1e-15) ok = 0;
    if (fabs(obj->quality.min_mean_ratio - q0) > 1e-15) ok = 0;
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
  RUN_TEST(test_the_neighbour_slot_table_inverts_the_neighbour_list);
  RUN_TEST(test_a_cell_range_reaches_exactly_its_cells_ranks);
  RUN_TEST(test_a_box_reaches_exactly_the_owners_of_its_points);
  RUN_TEST(test_global_counts_sum_unequal_ranks);
  RUN_TEST(test_update_reduces_min_and_max_across_ranks);
  RUN_TEST(test_update_ignores_ranks_that_hold_nothing);
  RUN_TEST(test_repeated_updates_are_stable);

  if (g_rank == 0) TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
