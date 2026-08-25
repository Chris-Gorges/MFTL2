/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Many-body periodic advection: the standard per-timestep MFTL loop.
 *
 * This is the reference for what a front-tracking solver actually does with this
 * library every timestep, and the benchmark for how much of that time goes into
 * communication. N spheres are placed in a periodic box, translated rigidly, and
 * redistributed across the process grid as they cross rank boundaries.
 *
 * ── The loop order is the part worth copying ────────────────────────────────
 *
 *   1. move only the vertices THIS rank owns, and wrap them back into the box
 *   2. MFTLUpdateGhostPositions      -- pull the vertices the neighbours moved
 *   3. MFTLComputeTriangleGeometry   -- areas/normals/CENTRES for ALL triangles
 *   4. MFTLRedistribute              -- ownership is decided from those centres
 *   5. MFTLComputeTriangleGeometry   -- restore the geometry redistribution drops
 *
 * Step 2 before step 3 is not optional: a triangle's three vertices can be owned
 * by three different ranks, so its geometry is only complete after the ghosts
 * have been synchronised. Step 3 before step 4 is not optional either --
 * redistribution routes by the CACHED centre, so a stale centre routes to the
 * wrong rank.
 *
 * And step 3 must cover ALL triangles, not only the owned ones. Redistribution
 * reads the cached centre of GHOST triangles as well -- that is how a rank knows
 * where its neighbour's triangle is going, and therefore where to send the
 * triangles adjacent to it. Refreshing only the owned ones makes two ranks
 * disagree about a ghost's destination, and someone's locally owned triangle
 * silently loses an edge neighbour. Nothing crashes; the ghost halo is just
 * short. A Debug build asserts on it inside the redistribution.
 *
 * Note what is NOT in the loop: no MFTLObjectRefreshFrames. Step 3 reads the
 * half-edge structure, and step 4 rebuilds the derived arrays on its way out, so
 * a frame refresh here would recompute something that is about to be recomputed.
 * --quality adds MFTLEnsembleUpdate AFTER step 4, where the frames are current.
 *
 * ── Step 5, and why the benchmark would lie without it ─────────────────────
 *
 * A redistribution that PACKS the area, the normal and the centroid onto the wire
 * ends a timestep with valid triangle geometry. This one deliberately does not:
 * geometry is recomputable from positions, so dropping it is what buys the 3x
 * smaller messages (see include/mftl_communication.h). The object that comes out
 * of MFTLRedistribute has valid topology and valid positions, and
 * `tris.area/normal/center` hold unspecified values.
 *
 * Timing the redistribution alone therefore credits it with a saving it has not
 * finished paying for. Step 5 pays it: one geometry pass over all triangles,
 * restoring the end-of-step state a geometry-carrying wire format would have
 * delivered. It is timed as its own phase so both readings are available --
 * redistribute alone, and redistribute plus the recompute it defers.
 *
 * `--no-post-geom` removes it, which is legitimate for a solver that consumes
 * triangle geometry between steps 3 and 4 (the natural place: that is the
 * configuration the forces belong to) and never looks at it afterwards. Such a
 * solver genuinely owes nothing, because step 3 of the next timestep recomputes
 * from the moved positions regardless.
 *
 * The pass used is MFTLComputeTriangleGeometry rather than MFTLEnsembleUpdate,
 * because restoring what the wire dropped is geometry only: ALL triangles, no
 * quality statistics and no reduction. examples/ensemble_update_benchmark.c
 * measures what the fused alternative would cost instead.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P parallel_advection_redistribute_list cx cy cz px py pz [options]
 *
 *     cx cy cz     process grid; cx*cy*cz must equal P
 *     px py pz     periodicity per axis (0/1). A non-periodic axis gets zero
 *                  advection velocity, since nothing here reflects at a wall.
 *
 *     --spheres N  number of bodies              (default 64)
 *     --refine N   subdivision level per body    (default 3, = 1280 triangles)
 *     --steps N    timesteps                     (default 100)
 *     --dt X       timestep                      (default 0.1)
 *     --verify     check conservation every step, exit non-zero on violation
 *     --quality    also run MFTLEnsembleUpdate (quality statistics) per step
 *     --normals    also run MFTLEnsembleUpdateVertexNormals per step
 *     --no-post-geom  drop step 5 (see above)
 *     --vtk N      write the surface every N steps
 *     --vtk-mode M  binary (default) | ascii | shared
 *                   binary/ascii write one .vtu per rank plus a .pvtu index;
 *                   shared writes ONE .vtu collectively with MPI-IO
 *
 *   A run big enough for the numbers to mean anything:
 *     mpirun -n 8 ... 4 2 1 1 1 1 --spheres 1000 --refine 4 --steps 500
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/* ── Configuration ──────────────────────────────────────────────────────── */

typedef struct
{
  int cuts[3];
  bool periodic[3];
  int n_spheres;
  int refinements;
  int steps;
  double dt;
  double radius;
  bool verify;
  bool quality;
  bool normals;
  bool post_geom;  ///< Step 5: restore the geometry redistribution drops.
  int vtk_every;  ///< 0 = never.
  MFTL_VtkOptions vtk;
} config;

static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  cfg->n_spheres = 64;
  cfg->refinements = 3;
  cfg->steps = 100;
  cfg->dt = 0.1;
  cfg->radius = 0.1;
  cfg->verify = false;
  cfg->quality = false;
  cfg->normals = false;
  cfg->post_geom = true;
  cfg->vtk_every = 0;
  cfg->vtk = MFTLVtkDefaults();

  if (argc < 7)
  {
    if (rank == 0)
    {
      fprintf(stderr,
              "Usage: %s <cuts_x> <cuts_y> <cuts_z> <per_x> <per_y> <per_z> [options]\n"
              "  --spheres N   number of bodies            (default 64)\n"
              "  --refine N    subdivision level           (default 3)\n"
              "  --steps N     timesteps                   (default 100)\n"
              "  --dt X        timestep                    (default 0.1)\n"
              "  --radius X    sphere radius               (default 0.1)\n"
              "  --verify      check conservation each step\n"
              "  --quality     also run MFTLEnsembleUpdate each step\n"
              "  --normals     also run MFTLEnsembleUpdateVertexNormals each step\n"
              "  --no-post-geom  skip the geometry pass after redistribution\n"
              "  --vtk N       write the surface every N steps\n"
              "  --vtk-mode M  binary (default) | ascii | shared\n",
              argv[0]);
    }
    return MFTL_ERROR;
  }

  for (int d = 0; d < 3; ++d)
  {
    cfg->cuts[d] = atoi(argv[1 + d]);
    cfg->periodic[d] = (atoi(argv[4 + d]) != 0);
  }

  for (int i = 7; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--verify"))
      cfg->verify = true;
    else if (!strcmp(argv[i], "--quality"))
      cfg->quality = true;
    else if (!strcmp(argv[i], "--normals"))
      cfg->normals = true;
    else if (!strcmp(argv[i], "--no-post-geom"))
      cfg->post_geom = false;
    else if (!strcmp(argv[i], "--vtk") && i + 1 < argc)
      cfg->vtk_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--vtk-mode") && i + 1 < argc)
    {
      const char *m = argv[++i];
      if (!strcmp(m, "binary"))
      {
        cfg->vtk.layout = MFTL_IO_FILE_PER_RANK;
        cfg->vtk.encoding = MFTL_VTK_BINARY;
      }
      else if (!strcmp(m, "ascii"))
      {
        cfg->vtk.layout = MFTL_IO_FILE_PER_RANK;
        cfg->vtk.encoding = MFTL_VTK_ASCII;
      }
      else if (!strcmp(m, "shared"))
      {
        cfg->vtk.layout = MFTL_IO_SINGLE_FILE;
        cfg->vtk.encoding = MFTL_VTK_BINARY;
      }
      else
      {
        if (rank == 0) fprintf(stderr, "Unknown --vtk-mode: %s (binary | ascii | shared)\n", m);
        return MFTL_ERROR;
      }
    }
    else if (!strcmp(argv[i], "--spheres") && i + 1 < argc)
      cfg->n_spheres = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--refine") && i + 1 < argc)
      cfg->refinements = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "--radius") && i + 1 < argc)
      cfg->radius = atof(argv[++i]);
    else
    {
      if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
      return MFTL_ERROR;
    }
  }

  if (cfg->cuts[0] * cfg->cuts[1] * cfg->cuts[2] != nproc)
  {
    if (rank == 0) fprintf(stderr, "Process grid %dx%dx%d does not match %d ranks.\n", cfg->cuts[0], cfg->cuts[1], cfg->cuts[2], nproc);
    return MFTL_ERROR;
  }
  if (cfg->n_spheres < 1 || cfg->refinements < 0 || cfg->steps < 0 || cfg->radius <= 0.0)
  {
    if (rank == 0) fprintf(stderr, "spheres must be >= 1, refine and steps >= 0, radius > 0.\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ── Conservation ───────────────────────────────────────────────────────── */

/**
 * @brief Rank-local surface area and entity counts, over OWNED entities only.
 *
 * Ghosts are copies; counting them would count their owner's contribution twice.
 * Ownership is an array PREFIX, so the counts are two reads and the area sum is
 * a contiguous walk -- there is no is_ghost test anywhere in here.
 */
static void local_totals(const MFTL_Ensemble *ens, double out[3])
{
  out[0] = out[1] = out[2] = 0.0;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    for (int i = 0; i < obj->tris.n_local; ++i) out[0] += obj->tris.area[i];
    out[1] += (double) obj->tris.n_local;
    out[2] += (double) obj->verts.n_local;
  }
}

/** @brief Owned and ghost triangle counts, for the halo-fraction report. */
static void local_halo(const MFTL_Ensemble *ens, long out[2])
{
  out[0] = out[1] = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    out[0] += obj->tris.n_local;
    out[1] += obj->tris.n - obj->tris.n_local;
  }
}

/** @brief Bytes MFTLUpdateGhostPositions has to move for this partition. */
static long local_ghost_bytes(const MFTL_Ensemble *ens)
{
  long n = 0;
  for (int o = 0; o < ens->num_objects; ++o) n += ens->objects[o]->verts.n - ens->objects[o]->verts.n_local;
  return n * (long) sizeof(MFTL_GhostUpdatePayload);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  config cfg;
  if (parse_args(argc, argv, rank, nproc, &cfg) != MFTL_SUCCESS)
  {
    MPI_Finalize();
    return 1;
  }

  /* Required once per process before anything communicates mesh entities. */
  MFTLRegisterMpiTypes();

  const double global_min[3] = {-4.0, -2.0, -2.0};
  const double global_max[3] = {4.0, 2.0, 2.0};
  const double radius = cfg.radius;

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, (double *) global_min, (double *) global_max, cfg.periodic, cfg.n_spheres);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0)
  {
    printf("%d ranks in a %dx%dx%d grid | periodic %d%d%d | %d spheres of radius %g, refine %d (%d triangles each) | %d steps\n", nproc, cfg.cuts[0],
           cfg.cuts[1], cfg.cuts[2], (int) cfg.periodic[0], (int) cfg.periodic[1], (int) cfg.periodic[2], cfg.n_spheres, cfg.radius, cfg.refinements,
           20 << (2 * cfg.refinements), cfg.steps);
    printf("wire format: %d B per vertex, %d B per triangle complex, %d B per ghost update\n", (int) sizeof(MFTL_PackedVertex),
           (int) sizeof(MFTL_PackedTriangleComplex), (int) sizeof(MFTL_GhostUpdatePayload));
  }

  /* Fixed seed on every rank: the centres must agree, since MFTLGenerateSphere is
   * collective and only rank 0 keeps the mesh. */
  srand(42);

  for (int o = 0; o < cfg.n_spheres; ++o)
  {
    MFTL_Object *obj = MFTLObjectCreate(o);
    MFTLEnsembleAddObject(ens, obj);

    /* Kept clear of the non-periodic faces; a periodic axis may straddle. */
    double centre[3];
    for (int d = 0; d < 3; ++d)
    {
      const double margin = cfg.periodic[d] ? 0.0 : 2.0 * radius;
      const double lo = global_min[d] + margin;
      const double hi = global_max[d] - margin;
      centre[d] = lo + ((double) rand() / (double) RAND_MAX) * (hi - lo);
    }

    MFTLGenerateSphere(obj, centre, radius, cfg.refinements);

    /* Ownership is decided from the cached centre, and a freshly generated mesh
     * has none. MFTLGenerateSphere already wrapped the positions. */
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  /* Scatter: everything is on rank 0 after generation. */
  MFTLRedistribute(ens);

  /* A non-periodic axis has no wall model here, so nothing may drift out of it. */
  const double velocity[3] = {cfg.periodic[0] ? 1.0 : 0.0, cfg.periodic[1] ? 0.37 : 0.0, cfg.periodic[2] ? 0.61 : 0.0};

  double reference[3] = {0.0, 0.0, 0.0};
  if (cfg.verify)
  {
    /* The pools were just rebuilt, so the areas are not valid yet -- one geometry
     * pass before the reference is taken, exactly as step 3 does in the loop. */
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

    double local[3];
    local_totals(ens, local);
    MPI_Allreduce(local, reference, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("reference: area %.12f, %d triangles, %d vertices\n", reference[0], (int) reference[1], (int) reference[2]);
  }

  double t_advect = 0.0, t_ghost = 0.0, t_geom = 0.0, t_redist = 0.0, t_post = 0.0, t_extra = 0.0, t_vtk = 0.0;
  long ghost_bytes = 0;
  long vtk_writes = 0;
  int failures = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop_start = MPI_Wtime();

  for (int step = 0; step < cfg.steps; ++step)
  {
    /* 1. Move what we own, and only what we own. */
    double t0 = MPI_Wtime();
    for (int o = 0; o < ens->num_objects; ++o)
    {
      MFTL_Object *obj = ens->objects[o];
      MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += velocity[0] * cfg.dt; pos[1] += velocity[1] * cfg.dt; pos[2] += velocity[2] * cfg.dt;
                                MFTLApplyPeriodicity(&ens->domain, pos););
    }
    t_advect += MPI_Wtime() - t0;

    /* 2. Pull in the vertices our neighbours moved. */
    ghost_bytes += local_ghost_bytes(ens);
    t0 = MPI_Wtime();
    MFTLUpdateGhostPositions(ens);
    t_ghost += MPI_Wtime() - t0;

    /* 3. Every triangle now has current vertices: refresh its geometry. ALL of
     *    them, ghosts included -- see the note in the header comment. */
    t0 = MPI_Wtime();
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
    t_geom += MPI_Wtime() - t0;

    if (cfg.verify)
    {
      /* Rigid translation conserves surface area exactly, and redistribution
       * conserves entities exactly. Either one drifting means a triangle
       * straddling a periodic seam is being measured across the box, or an
       * entity was lost or duplicated in the exchange. */
      double local[3], total[3];
      local_totals(ens, local);
      MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      const double rel = fabs(total[0] - reference[0]) / reference[0];
      if (rel > 1e-9 || total[1] != reference[1] || total[2] != reference[2])
      {
        if (rank == 0)
        {
          fprintf(stderr, "step %d: area %.12f (rel dev %.3e), %d triangles, %d vertices -- expected %.12f / %d / %d\n", step, total[0], rel, (int) total[1],
                  (int) total[2], reference[0], (int) reference[1], (int) reference[2]);
        }
        failures++;
      }
    }

    /* 4. Re-decide ownership from the fresh centres. */
    t0 = MPI_Wtime();
    MFTLRedistribute(ens);
    t_redist += MPI_Wtime() - t0;

    /* 5. Restore what the redistribution deliberately did not send. The object
     *    now has valid topology and valid positions; its areas, normals and
     *    centres hold unspecified values until this runs. This is the phase
     *    that makes the per-step total describe a complete end state. */
    if (cfg.post_geom)
    {
      t0 = MPI_Wtime();
      for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
      t_post += MPI_Wtime() - t0;
    }

    /* Optional consumers of the rebuilt derived arrays. Both go AFTER the
     * redistribution: it leaves the frames current, so neither needs a refresh
     * of its own, and both would otherwise operate on the old partition. */
    if (cfg.quality || cfg.normals)
    {
      t0 = MPI_Wtime();
      if (cfg.quality) MFTLEnsembleUpdate(ens);
      /* The ensemble form, not a loop over MFTLComputeVertexNormals: that one
       * leaves every GHOST vertex holding a partial sum -- correctly normalised
       * and quietly wrong. This adds one ghost exchange and makes the field
       * exact for the whole pool. */
      if (cfg.normals) MFTLEnsembleUpdateVertexNormals(ens);
      t_extra += MPI_Wtime() - t0;
    }

    /* The dump goes last, so what lands on disk is the state the step ended in.
     * It is timed separately and excluded from the phase percentages: writing
     * the surface is not part of the timestep, it is what you do to look at it. */
    if (cfg.vtk_every > 0 && step % cfg.vtk_every == 0)
    {
      char prefix[256];
      snprintf(prefix, sizeof prefix, "advection_%04d", step);

      t0 = MPI_Wtime();
      if (MFTLWriteVtu(ens, prefix, &cfg.vtk) != MFTL_SUCCESS)
      {
        if (rank == 0) fprintf(stderr, "step %d: MFTLWriteVtu failed for %s\n", step, prefix);
        failures++;
      }
      t_vtk += MPI_Wtime() - t0;
      vtk_writes++;
    }

    if (rank == 0 && cfg.steps >= 20 && step % (cfg.steps / 10) == 0) printf("  step %d/%d\n", step, cfg.steps);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_total = MPI_Wtime() - t_loop_start;

  /* The phase breakdown is a MEAN over ranks, not a max. Taking the max per phase
   * would pick a different rank for each one, and the parts would then not add up
   * to the whole -- the slowest rank in the exchange is usually not the slowest
   * one in the rebuild. */
  const double mine[8] = {t_total, t_advect, t_ghost, t_geom, t_redist, t_extra, t_post, t_vtk};
  double sum[8];
  MPI_Reduce(mine, sum, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Partition quality, which is what "imbalance" has to mean here: the wall clock
   * is bounded by barriers and is therefore identical on every rank by
   * construction, so max/mean of the TIME would always read 1.00. Max/mean of the
   * owned triangle count is the thing that actually varies. */
  long halo[2];
  local_halo(ens, halo);
  long halo_sum[2] = {0, 0}, owned_max = 0, gbytes = 0;
  MPI_Reduce(halo, halo_sum, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[0], &owned_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ghost_bytes, &gbytes, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0 && cfg.steps > 0)
  {
    for (int i = 0; i < 8; ++i) sum[i] /= (double) nproc;
    const double mean_total = sum[0];
    const double rest = mean_total - sum[1] - sum[2] - sum[3] - sum[4] - sum[5] - sum[6] - sum[7];
    const double owned_mean = (double) halo_sum[0] / (double) nproc;
    const double per_step = (double) cfg.steps;

    printf("\n=== advection benchmark, %d ranks, %d steps ===\n", nproc, cfg.steps);
    printf("  wall clock                 %8.3f s\n", mean_total);
    printf("  owned triangles            %8ld   (%.0f per rank on average)\n", halo_sum[0], owned_mean);
    printf("  ghost triangles            %8ld   (%.1f %% halo overhead)\n", halo_sum[1], 100.0 * (double) halo_sum[1] / (double) halo_sum[0]);
    printf("  partition imbalance        %8.2f   (busiest rank / average)\n", (owned_mean > 0.0) ? (double) owned_max / owned_mean : 1.0);
    printf("  ghost sync volume          %8.2f MB/step (all ranks)\n", (double) gbytes / (1024.0 * 1024.0) / per_step);
    printf("  -- mean per rank ---------------------------\n");
    printf("  redistribute        %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[4], 100.0 * sum[4] / mean_total, 1000.0 * sum[4] / per_step);
    printf("  ghost sync          %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[2], 100.0 * sum[2] / mean_total, 1000.0 * sum[2] / per_step);
    printf("  triangle geometry   %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[3], 100.0 * sum[3] / mean_total, 1000.0 * sum[3] / per_step);
    if (cfg.post_geom)
      printf("  post-redist geometry%8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[6], 100.0 * sum[6] / mean_total, 1000.0 * sum[6] / per_step);
    printf("  advection           %8.3f s   %5.1f %%\n", sum[1], 100.0 * sum[1] / mean_total);
    if (cfg.quality || cfg.normals) printf("  quality + normals   %8.3f s   %5.1f %%\n", sum[5], 100.0 * sum[5] / mean_total);
    if (vtk_writes > 0) printf("  vtk output          %8.3f s   %5.1f %%   (%.1f ms per dump, %ld dumps)\n", sum[7], 100.0 * sum[7] / mean_total,
                               1000.0 * sum[7] / (double) vtk_writes, vtk_writes);
    printf("  verify + rest       %8.3f s   %5.1f %%\n", rest, 100.0 * rest / mean_total);
    printf("  redistribute cost   %8.1f ns per owned triangle per step\n", 1e9 * sum[4] / (per_step * (owned_mean > 0.0 ? owned_mean : 1.0)));

    /* The honest per-step total: the redistribution plus the geometry pass it
     * defers, i.e. the cost of reaching a state with valid triangle geometry. */
    if (cfg.post_geom)
    {
      printf("  redistribute + post %8.3f s              (%.3f ms/step)  <- the honest total\n", sum[4] + sum[6],
             1000.0 * (sum[4] + sum[6]) / per_step);
    }
    printf("==============================================\n");
  }

  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (cfg.verify && rank == 0)
  {
    printf("%s\n", total_failures == 0 ? "verification: OK" : "verification: FAILED");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failures > 0) ? 1 : 0;
}
