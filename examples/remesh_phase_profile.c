/*
 * @Author: Christian Gorges
 * @Date: 2026-08-18
 *
 * @brief Where a distributed remeshing pass spends its time, and how much of the
 *        mesh each placement moves.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE QUESTION THIS EXISTS TO ANSWER
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * docs/analysis.md §5.4 records one thing as deliberately
 * not attempted and names its prerequisite:
 *
 *     "A profile that separates sweep time from migration time inside
 *      MFTLRemeshEnsembleParallel is the prerequisite, and it does not exist yet."
 *
 * This is that profile. It is the measurement that decides whether the work-list
 * architecture (docs/analysis.md §5.4) is worth building -- a work
 * list shortens each SWEEP, and if the pass is bound by MIGRATION instead then it
 * buys nothing.
 *
 * It also settles two claims that were argued from structure rather than
 * measured. First, that a shifted placement moves "a band": it does not, it moves
 * a large fraction of the mesh, and `entities migrated` says which. Second, that
 * the geometry pass in front of each migration is negligible: it is O(n) over
 * owned AND ghost triangles and is invariant to how much actually moves, so it is
 * the floor under every optimisation aimed at the wire.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW TO READ THE OUTPUT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Every time is a MAXIMUM OVER RANKS, not a sum and not a mean. The phase loop is
 * barrier-synchronised by the id collective, so the wall clock a caller waits for
 * is set by the slowest rank in each section; a mean would report a comfortable
 * number for a pass that spent half its time with seven ranks idle.
 *
 * `migrated` is RANK-LOCAL on purpose -- it is what THIS rank put on the wire,
 * which is the quantity that pairs with its own `t_migrate`. Rank 0's figure is
 * printed; it is representative, not global.
 *
 * The `skipped` column is the placements recognised as the one already in force
 * and elided. Phase 0 of the schedule asks for the canonical placement and the
 * mesh arrives canonically partitioned, so at least one is always skipped; on an
 * anisotropic grid, where several table entries collapse onto the same placement,
 * there are more.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OPTIONS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   -refine N    icosahedron subdivisions for the sphere      (default 4)
 *   -target L    target edge length; 0 = 0.6 * the mesh mean  (default 0)
 *   -budget N    parallel_phase_budget                        (default 8)
 *   -passes N    remeshing passes to profile                  (default 3)
 *   -coarsen N   parallel_coarsen: 0 off, 1 auto, >1 capped   (default 0)
 *   -radius R    sphere radius                                (default 0.7)
 *   -walls       non-periodic domain                          (default periodic)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

static int g_rank = 0;
static int g_nproc = 1;

/** @brief MPI_Dims_create's answer, so the grid is the library's own choice. */
static void grid_for(int nproc, int cuts[3])
{
  cuts[0] = cuts[1] = cuts[2] = 0;
  MPI_Dims_create(nproc, 3, cuts);
}

static int arg_int(int argc, char **argv, const char *flag, int fallback)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
  return fallback;
}

static double arg_double(int argc, char **argv, const char *flag, double fallback)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return atof(argv[i + 1]);
  return fallback;
}

static bool arg_flag(int argc, char **argv, const char *flag)
{
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return MFTL_TRUE;
  return MFTL_FALSE;
}

/** @brief Owned triangles on the busiest rank against the mean: load imbalance. */
static void triangle_spread(const MFTL_Object *obj, double *out_max, double *out_mean)
{
  double mine = (double) obj->tris.n_local;
  double sum = mine, max = mine;

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  *out_max = max;
  *out_mean = sum / (double) g_nproc;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  const int refine = arg_int(argc, argv, "-refine", 4);
  const int budget = arg_int(argc, argv, "-budget", 8);
  const int passes = arg_int(argc, argv, "-passes", 3);
  const int coarsen = arg_int(argc, argv, "-coarsen", 0);
  const double radius = arg_double(argc, argv, "-radius", 0.7);
  const double target_in = arg_double(argc, argv, "-target", 0.0);
  const bool periodic = !arg_flag(argc, argv, "-walls");

  int cuts[3];
  grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 2);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, centre, radius, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLRedistribute(ens);

  /* COLLECTIVE, not MFTLMeanEdgeLength: the local form returns zero on a rank
   * holding nothing, and a zero-length band refines without bound. */
  const double mean_edge = MFTLEnsembleMeanEdgeLength(ens);
  const double target = (target_in > 0.0) ? target_in : 0.6 * mean_edge;

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, 4, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = 3;
  params.smooth_relaxation = 0.5;
  params.parallel_phase_budget = budget;
  params.parallel_coarsen = coarsen;

  if (g_rank == 0)
  {
    const double h_min = MFTLDomainGetHMin(&ens->domain);
    printf("\n=== Remeshing phase profile ===\n");
    printf("np %d, grid %dx%dx%d, %s, sphere r=%.2f refine %d\n", g_nproc, cuts[0], cuts[1], cuts[2], periodic ? "periodic" : "walls", radius, refine);
    printf("mean edge %.4f -> target %.4f (band [%.4f, %.4f]), budget %d, coarsen %d\n", mean_edge, target, params.min_edge_length,
           params.max_edge_length, budget, coarsen);
    printf("h_min %.4f; the 3*max_edge estimate of delta is %.4f, so the ESTIMATE says h > 4*delta is %s\n", h_min,
           MFTL_REMESH_BAND_EDGES * params.max_edge_length, (h_min > 4.0 * MFTL_REMESH_BAND_EDGES * params.max_edge_length) ? "satisfied" : "VIOLATED");
    printf("           (the measured band is in the table; it runs about a third of that estimate)\n\n");

    printf("pass  tris  ph mig skip  conv  hLONG hSHORT   grp   band  |     total   migrate  geometry    freeze     sweep    reduce  | migrated  maxT/meanT\n");
    printf("-------------------------------------------------------------------------------------------------------------------------------------------\n");
  }

  for (int pass = 0; pass < passes; ++pass)
  {
    MFTL_RemeshStats st;
    if (MFTLRemeshEnsembleParallel(ens, &params, &st) != MFTL_SUCCESS)
    {
      if (g_rank == 0) fprintf(stderr, "remesh failed on pass %d\n", pass);
      break;
    }

    /* RE-RESOLVED, not the pointer MFTLObjectCreate returned: a remeshing pass
     * ends in a redistribution, which destroys the object of a body a rank keeps
     * nothing of and creates one for a body arriving. `obj` above is valid only
     * until the opening MFTLRedistribute. MFTL_NULL means this rank holds none of
     * the body, which is a legitimate state and not an error. */
    const MFTL_Object *const held = MFTLEnsembleFindLocalObject(ens, 0);

    double max_t = 0.0, mean_t = 0.0;
    if (held != MFTL_NULL) triangle_spread(held, &max_t, &mean_t);

    /* The global triangle count is the same number on every holder, so rank 0
     * can only print it if it holds some of the body. It is reduced rather than
     * read, which also covers the case where rank 0 holds none. */
    int n_tri_global = (held != MFTL_NULL) ? held->global_n_triangles : 0;
    MPI_Allreduce(MPI_IN_PLACE, &n_tri_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    if (g_rank == 0)
    {
      printf("%4d %6d %3d %3d %4d %5d %6d %6d %5d %6.3f  | %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f  | %8d  %9.2f\n", pass, n_tri_global,
             st.n_phases, st.n_migrations, st.n_migrations_skipped, st.converged ? 1 : 0, st.n_edges_hidden_long, st.n_edges_hidden_short,
             st.n_coarse_groups, st.band_measured, st.t_total, st.t_migrate, st.t_geometry, st.t_freeze, st.t_sweep, st.t_reduce,
             st.n_entities_migrated, (mean_t > 0.0) ? max_t / mean_t : 0.0);

      /* AS FRACTIONS OF THE ACCOUNTED SUM, NOT OF t_total, and the distinction is
       * not pedantry. Every section is reduced independently as a maximum over
       * ranks, and the maxima are attained on DIFFERENT ranks, so their sum
       * exceeds the maximum of the sum -- i.e. exceeds t_total. Dividing by
       * t_total produces percentages adding to more than a hundred and an
       * "unaccounted" term that goes negative, which is an artefact of the
       * reduction and not a measurement of anything. */
      const double acc = st.t_migrate + st.t_geometry + st.t_freeze + st.t_sweep + st.t_reduce;
      if (acc > 0.0)
      {
        printf("      -> of the critical path: migrate %.0f%%, geometry %.0f%%, freeze %.0f%%, sweep %.0f%%, reduce %.0f%%"
               "   (sum/total %.2f)\n",
               100.0 * st.t_migrate / acc, 100.0 * st.t_geometry / acc, 100.0 * st.t_freeze / acc, 100.0 * st.t_sweep / acc,
               100.0 * st.t_reduce / acc, (st.t_total > 0.0) ? acc / st.t_total : 0.0);
      }
    }
  }

  if (g_rank == 0)
  {
    printf("\nHow to read it:\n");
    printf("  SWEEP dominating   -> the work-list architecture (plan section 7) is worth measuring.\n");
    printf("  MIGRATE dominating -> it is not; the lever is fewer phases, not shorter sweeps.\n");
    printf("  GEOMETRY large     -> the O(n) floor in front of every migration, which no wire\n");
    printf("                        optimisation touches. Fusing it into the redistribution's own\n");
    printf("                        rebuild is deferred issue 7 in CLAUDE.md.\n");
    printf("  maxT/meanT >> 1    -> load imbalance in the PARTITION; with coarsening on, expect it\n");
    printf("                        to approach the group size, because one rank per group sweeps.\n");
    printf("  REDUCE large       -> read it as WAITING, not as bandwidth. The id collective is the\n");
    printf("                        phase's barrier, so the maximum over ranks of the time spent in\n");
    printf("                        it is what the EARLIEST rank waited for the latest -- i.e. it is\n");
    printf("                        a second, sharper measure of imbalance than maxT/meanT, which\n");
    printf("                        only counts triangles and not the work each one costs.\n");
    printf("  sum/total > 1      -> expected. Each section is a maximum over ranks and the maxima\n");
    printf("                        are attained on different ranks, so the sections do not add up\n");
    printf("                        to the wall clock. Compare sections with each other, not with\n");
    printf("                        the total.\n\n");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return 0;
}
