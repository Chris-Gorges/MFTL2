/*
 * @Author: Christian Gorges
 * @Date: 2026-08-23
 *
 * @brief Strong, weak and granularity scaling of the distributed advection timestep.
 *
 * docs/performance-analysis.md sec 10 and docs/parallel-remeshing-analysis.md both
 * record the same hole in their own words: nothing in MFTL has been measured above
 * four ranks on real hardware, and there was no instrument that could do it. This
 * is that instrument. It runs the same five-step timestep as
 * examples/parallel_advection_redistribute_list.c -- that example is the reference
 * for what a step COSTS -- and answers the different question of what happens to
 * that cost as P grows.
 *
 * Drive it with run_scaling.py, which sweeps the rank counts, accumulates the JSON
 * this writes, and renders the curves. A single run of this binary is one point.
 *
 * -- Three axes, and only two of them are scaling laws --------------------------
 *
 *   STRONG       total work fixed, P grows.        Ideal: T proportional to 1/P.
 *                --mode strong. The domain and the body count do not depend on P.
 *
 *   WEAK         work PER RANK fixed, P and total work grow together.
 *                Ideal: T flat. --mode weak. The domain is the per-rank box times
 *                the process grid, so every rank owns a constant physical volume
 *                holding a constant number of bodies -- and therefore a constant
 *                halo fraction and a constant neighbour count. Growing the body
 *                count inside a FIXED domain instead would keep the work per rank
 *                constant while making the problem denser, and the resulting curve
 *                would mix "the machinery got slower" with "the problem got
 *                harder". It is not offered.
 *
 *   GRANULARITY  the same triangle count, split differently between bodies and
 *                triangles-per-body. This is NOT a scaling law. It changes the
 *                CONSTANT in both of the above, so it is swept as a family: pass
 *                the rungs of an iso-triangle ladder (4096 x refine 1, 1024 x
 *                refine 2, ... 16 x refine 5 are all 327 680 triangles) as separate
 *                strong-mode runs. In MFTL it is not cosmetic -- many small bodies
 *                against few large ones is the same distinction the remesher
 *                dispatches on, and it drives the halo fraction, the per-object
 *                loop overhead, and the number of distinct redistribution
 *                destinations.
 *
 * -- What is measured, beyond the wall clock ------------------------------------
 *
 * A wall clock alone cannot separate "the exchange got slower" from "one rank had
 * more work and everybody waited in the next collective". So each phase reports
 * mean, max, min and coefficient of variation over ranks, and four labelled totals
 * whose disagreement is itself the measurement:
 *
 *   wall clock (barriered loop bounds)   <- the scaling number
 *   sum of phase means                   <- the breakdown that adds up
 *   sum of phase maxima                  <- an UPPER BOUND on the critical path
 *   unattributed                         <- wall minus sum of means
 *
 * Reporting the per-phase mean is right for the breakdown, for the reason the
 * sibling example gives: per-phase maxima land on different ranks, so they do not
 * sum to the whole. That is an argument against calling sum-of-maxima the critical
 * path, not against printing it. The GAP between it and the barriered wall clock is
 * exactly the imbalance that is not additive across phases.
 *
 * Per-phase imbalance is max/mean. For a COLLECTIVE phase that is a lower bound --
 * the collective has already equalised the finish times -- so the structural
 * imbalance (busiest rank's owned triangles over the average) is printed alongside
 * and is the honest partition metric. Max over mean of a barrier-bounded wall clock
 * always reads 1.00 and measures nothing.
 *
 * -- Four traps this file exists to avoid ---------------------------------------
 *
 * 1. SETUP IS RANK-0-SERIAL AND ITS COST IS RESIDENT, NOT TRANSIENT.
 *    MFTLGenerateSphere builds the whole sphere on rank 0, and MFTL pools NEVER
 *    SHRINK (include/mftl_object.h, "pools never shrink, so a mesh of stable size
 *    does not re-fault its pages every timestep"). Redistribution lowers tris.n and
 *    never tris.capacity, so every body keeps its full generation capacity on rank 0
 *    for the whole run. Generating in batches with a redistribute between them --
 *    the obvious fix -- does not work: it shaves only the packed-buffer transient on
 *    top of a base it cannot touch, and costs O(N^2/K) setup work, because
 *    collect_destinations re-derives the entire halo on every call. So the footprint
 *    is PRINTED and checked against --max-gen-mb instead, and t_setup is timed and
 *    reported separately. In weak mode, where the body count grows with P, setup
 *    time grows LINEARLY WITH P on one rank while the timed loop does not -- it is
 *    the Amdahl ceiling of the harness, not of the library, and it must never be
 *    folded into a weak-scaling total.
 *
 * 2. COUNTING MIGRATION WARMS THE CACHES FOR THE PHASE BEING TIMED.
 *    --migration-stats streams tris.center, tris.verts, tris.neighbors and
 *    verts.pos immediately before MFTLRedistribute -- which is precisely the set of
 *    arrays compute_owners and collect_destinations are about to read. It makes the
 *    redistribution look faster than it is. Default OFF, and it prints a warning
 *    when on. Never quote timings from a run that had it enabled.
 *
 * 3. THE FIRST REDISTRIBUTE IS NOT A REDISTRIBUTE, IT IS THE INITIAL SCATTER.
 *    Rank 0 holds every body after generation and no destination is inside its
 *    26-neighbourhood, so it takes the global MPI_Alltoallv path and moves the
 *    entire mesh. It is a different operation and is excluded from the timed loop
 *    rather than warmed. --warmup defaults to 3 and not 1 for the weaker version of
 *    the same effect: the step after setup still migrates atypically much, because
 *    the advection has not reached a steady state.
 *
 * 4. A TIMER THAT RAN OUT OF BITS LOOKS EXACTLY LIKE A SCALING LIMIT.
 *    At high P the per-rank advection and geometry phases shrink toward MPI_Wtick,
 *    and a strong-scaling curve that flattens for that reason is indistinguishable
 *    from one that flattened physically. Every phase mean below 100 * MPI_Wtick is
 *    flagged in the report and in the JSON. MPI_WTIME_IS_GLOBAL is false on stock
 *    Open MPI and MPICH and that is fine: every quantity here is a difference of two
 *    MPI_Wtime calls ON THE SAME RANK, so a constant per-rank offset cancels exactly.
 *    Timestamps are never compared across ranks; --sync-phases is the legal way to
 *    ask that question.
 *
 * -- Placement and motion, and why they are not the obvious thing ---------------
 *
 * Bodies sit on a JITTERED LATTICE, not at uniform random positions. Random
 * placement has Poisson fluctuations in bodies-per-rank; those fluctuations change
 * with P and with N, so they land inside the weak-scaling curve and cannot be told
 * apart from a real overhead trend. A jittered lattice is stratified sampling:
 * bodies-per-rank is near-constant by construction, and the curve then measures the
 * machinery. The radius is derived from a VOLUME FRACTION rather than fixed, so the
 * density is the same at every rung of the iso-triangle ladder and at every P.
 *
 * Each body carries its OWN velocity direction at a common speed. One global
 * velocity makes every body cross a cut plane on the same step, turning migration
 * traffic into a spike train that is an artefact of the setup. Per-body directions
 * stagger the crossings, and the motion stays a RIGID TRANSLATION PER BODY -- which
 * is what makes --verify exact, since translation conserves surface area to the bit
 * and redistribution conserves entities exactly.
 *
 * All randomness is splitmix64 keyed on the body index, so the placement is a pure
 * function of the body index and is identical on every rank, at every rank count,
 * and on every platform. rand() is none of those things across libcs.
 *
 * -- Usage ----------------------------------------------------------------------
 *
 *   mpirun -n P parallel_advection_scaling [options]
 *
 *     --mode strong|weak    (default strong)
 *     --label NAME          tag copied into the JSON; run_scaling.py owns the
 *                           configuration table, this binary just records the name
 *
 *   problem size
 *     --spheres N           bodies, STRONG mode                  (default 256)
 *     --spheres-per-rank M  bodies per rank, WEAK mode           (default 32)
 *     --refine R            subdivision level, 20*4^R triangles  (default 3)
 *     --domain X Y Z        domain box, STRONG mode              (default 4 4 4)
 *     --box B               per-rank box edge, WEAK mode         (default 1.0)
 *     --phi F               volume fraction, sets the radius     (default 0.05)
 *     --radius R            explicit radius, overrides --phi     (default derived)
 *     --jitter F            lattice jitter, in cell edges        (default 0.35)
 *
 *   decomposition
 *     --cuts X Y Z          process grid; default is MPI_Dims_create
 *     --periodic PX PY PZ   per axis 0/1                         (default 1 1 1)
 *
 *   motion and duration
 *     --speed S             body speed                           (default 1.0)
 *     --dt X                timestep                             (default 0.05)
 *     --steps N             timed steps                          (default 50)
 *     --warmup N            untimed steps before them            (default 3)
 *     --repeat R            re-run the timed loop, report the min (default 1)
 *
 *   optional per-step work
 *     --quality             also MFTLEnsembleUpdate
 *     --normals             also MFTLEnsembleUpdateVertexNormals
 *     --no-post-geom        drop the geometry pass redistribution defers
 *     --vtk N               write the surface every N steps      (default 0, off)
 *     --vtk-mode M          binary (default) | ascii | shared
 *     --vtk-dir D           where the dumps go                   (default ".")
 *
 *   instrumentation
 *     --sync-phases         barrier before each phase; NOT for quoted timings
 *     --migration-stats     count migration volume; NOT for quoted timings
 *     --per-step            put the per-step series in the JSON
 *     --verify              check conservation every step, exit non-zero on drift
 *     --max-gen-mb X        refuse if rank 0 generation exceeds this (default 8192)
 *     --json PATH           write the machine-readable result here
 *     --git HASH            recorded verbatim in the JSON        (default "unknown")
 *
 *   One point of a strong curve, and one of a weak curve:
 *     mpirun -n 4 ./build/parallel_advection_scaling --mode strong --spheres 256 --refine 3
 *     mpirun -n 4 ./build/parallel_advection_scaling --mode weak --spheres-per-rank 32 --refine 3
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mpi.h>

#include "mftl_core.h"

/* ── Phases ─────────────────────────────────────────────────────────────── */

/**
 * @brief The unit operations of one timestep, in the order they run.
 *
 * PH_INSTR is the instrumentation bucket -- conservation checks, migration
 * counting, RSS sampling. It is accumulated so that "wall minus the sum of the
 * phases" stays a small honest residue, and reported on its own line so that it
 * is never read as part of the timestep.
 */
enum
{
  PH_ADVECT = 0,
  PH_GHOST,
  PH_GEOM,
  PH_REDIST,
  PH_POST,
  PH_QUAL,
  PH_NORM,
  PH_VTK,
  PH_INSTR,
  N_PHASE
};

/** @brief JSON keys and report labels for the phases; index with the enum above. */
static const char *const PHASE_KEY[N_PHASE] = {"advect", "ghost_sync", "geometry", "redistribute", "post_geom", "quality", "normals", "vtk", "instrument"};

/** @brief Human-readable phase names, padded so the report table lines up. */
static const char *const PHASE_LABEL[N_PHASE] = {"advection         ", "ghost sync        ", "triangle geometry ", "redistribute      ", "post-redist geom  ",
                                                 "quality stats     ", "vertex normals    ", "vtk output        ", "instrumentation   "};

/* ── Deterministic randomness ───────────────────────────────────────────── */

/**
 * @brief splitmix64, so placement is a pure function of the body index.
 *
 * rand() is not reproducible across libcs, and the sibling example only needs it
 * to agree between RANKS of one run. A scaling study compares a laptop run to a
 * cluster run and an old result to a new one, so the placement has to be
 * identical everywhere. Fixed-width integer arithmetic is; a libc is not.
 *
 * @param s Mutable state, advanced by one step.
 * @return The next 64-bit value of the stream.
 */
static uint64_t splitmix64(uint64_t *s)
{
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/**
 * @brief A uniform double in [0, 1) from the stream.
 * @param s Mutable splitmix64 state.
 * @return The sample.
 */
static double sm_uniform(uint64_t *s)
{
  return (double) (splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * @brief The stream belonging to one body, independent of every other body.
 * @param body_index Index of the body in the ensemble.
 * @return A seeded state.
 */
static uint64_t sm_seed_for_body(int body_index)
{
  return 0x243F6A8885A308D3ULL ^ ((uint64_t) body_index * 0x9E3779B97F4A7C15ULL);
}

/* ── Process-local resources ────────────────────────────────────────────── */

/**
 * @brief Peak resident set size of this process, in bytes.
 *
 * `ru_maxrss` is BYTES on macOS and KILOBYTES on Linux. Normalising here rather
 * than at the call site is what stops a laptop number and a cluster number
 * differing by 1024x without anything looking wrong.
 *
 * @return Peak RSS in bytes, or 0 if the query failed.
 */
static long peak_rss_bytes(void)
{
  struct rusage ru;
  if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#ifdef __APPLE__
  return (long) ru.ru_maxrss;
#else
  return (long) ru.ru_maxrss * 1024L;
#endif
}

/**
 * @brief Size of one file, in bytes, or 0 if it cannot be stat'd.
 * @param path The file.
 * @return Size in bytes.
 */
static long file_size_bytes(const char *path)
{
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return (long) st.st_size;
}

/* ── Configuration ──────────────────────────────────────────────────────── */

/** @brief Which scaling law this run is a point of. */
typedef enum
{
  MODE_STRONG = 0,  ///< Total work fixed, P grows.
  MODE_WEAK = 1  ///< Work per rank fixed; the domain grows with the process grid.
} scaling_mode;

/** @brief Everything the run is parameterised by; filled once by parse_args(). */
typedef struct
{
  scaling_mode mode;
  char label[64];  ///< Tag copied into the JSON; run_scaling.py owns the config table.
  char git[64];  ///< Recorded verbatim, so a number can be traced to a tree.

  int cuts[3];  ///< 0,0,0 until resolved by MPI_Dims_create or --cuts.
  bool cuts_given;
  bool periodic[3];

  int n_spheres;  ///< STRONG mode body count.
  int spheres_per_rank;  ///< WEAK mode body count per rank.
  int refinements;
  double domain[3];  ///< STRONG mode domain box.
  double box;  ///< WEAK mode per-rank box edge.
  double phi;  ///< Volume fraction; sets the radius unless --radius is given.
  double radius;  ///< 0 means "derive from phi".
  double jitter;  ///< Lattice jitter, in cell edges.

  double speed;
  double dt;
  int steps;
  int warmup;
  int repeat;

  bool quality;
  bool normals;
  bool post_geom;
  bool sync_phases;
  bool migration_stats;
  bool per_step;
  bool verify;

  double max_gen_mb;
  int vtk_every;  ///< 0 = never.
  char vtk_dir[192];
  MFTL_VtkOptions vtk;
  char json_path[256];  ///< Empty means "do not write one".
} config;

/**
 * @brief MPI_Dims_create's answer, so the grid is the library's own choice.
 *
 * MFTLEnsembleCreate only VALIDATES that the product equals the comm size; it
 * has no factoriser of its own. A sweep over P cannot carry a hand-written grid
 * per point, so the default has to come from somewhere, and MPI's own balanced
 * factorisation is the least surprising choice. Same helper as
 * examples/remesh_phase_profile.c.
 *
 * @param      nproc Communicator size.
 * @param[out] cuts  The process grid.
 */
static void grid_for(int nproc, int cuts[3])
{
  cuts[0] = cuts[1] = cuts[2] = 0;
  MPI_Dims_create(nproc, 3, cuts);
}

/**
 * @brief Fills @p cfg from argv, or explains what was wrong and fails.
 *
 * Unknown options are a hard error rather than being ignored: a sweep script
 * passes these programmatically, and a silently dropped `--refine` would produce
 * a whole curve of the wrong problem size that still looked plausible.
 *
 * @param argc  From main.
 * @param argv  From main.
 * @param rank  This rank, so only rank 0 writes diagnostics.
 * @param nproc Communicator size, for validating an explicit --cuts.
 * @param[out] cfg Receives the configuration.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR after printing why.
 */
static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  memset(cfg, 0, sizeof *cfg);
  cfg->mode = MODE_STRONG;
  snprintf(cfg->label, sizeof cfg->label, "custom");
  snprintf(cfg->git, sizeof cfg->git, "unknown");
  cfg->periodic[0] = cfg->periodic[1] = cfg->periodic[2] = true;
  cfg->n_spheres = 256;
  cfg->spheres_per_rank = 32;
  cfg->refinements = 3;
  cfg->domain[0] = cfg->domain[1] = cfg->domain[2] = 4.0;
  cfg->box = 1.0;
  cfg->phi = 0.05;
  cfg->radius = 0.0;
  cfg->jitter = 0.35;
  cfg->speed = 1.0;
  cfg->dt = 0.05;
  cfg->steps = 50;
  cfg->warmup = 3;
  cfg->repeat = 1;
  cfg->post_geom = true;
  cfg->max_gen_mb = 8192.0;
  snprintf(cfg->vtk_dir, sizeof cfg->vtk_dir, ".");
  cfg->vtk = MFTLVtkDefaults();

  for (int i = 1; i < argc; ++i)
  {
    const char *a = argv[i];

    if (!strcmp(a, "--mode") && i + 1 < argc)
    {
      const char *m = argv[++i];
      if (!strcmp(m, "strong"))
        cfg->mode = MODE_STRONG;
      else if (!strcmp(m, "weak"))
        cfg->mode = MODE_WEAK;
      else
      {
        if (rank == 0) fprintf(stderr, "Unknown --mode: %s (strong | weak)\n", m);
        return MFTL_ERROR;
      }
    }
    else if (!strcmp(a, "--label") && i + 1 < argc)
      snprintf(cfg->label, sizeof cfg->label, "%s", argv[++i]);
    else if (!strcmp(a, "--git") && i + 1 < argc)
      snprintf(cfg->git, sizeof cfg->git, "%s", argv[++i]);
    else if (!strcmp(a, "--cuts") && i + 3 < argc)
    {
      for (int d = 0; d < 3; ++d) cfg->cuts[d] = atoi(argv[++i]);
      cfg->cuts_given = true;
    }
    else if (!strcmp(a, "--periodic") && i + 3 < argc)
    {
      for (int d = 0; d < 3; ++d) cfg->periodic[d] = (atoi(argv[++i]) != 0);
    }
    else if (!strcmp(a, "--domain") && i + 3 < argc)
    {
      for (int d = 0; d < 3; ++d) cfg->domain[d] = atof(argv[++i]);
    }
    else if (!strcmp(a, "--spheres") && i + 1 < argc)
      cfg->n_spheres = atoi(argv[++i]);
    else if (!strcmp(a, "--spheres-per-rank") && i + 1 < argc)
      cfg->spheres_per_rank = atoi(argv[++i]);
    else if (!strcmp(a, "--refine") && i + 1 < argc)
      cfg->refinements = atoi(argv[++i]);
    else if (!strcmp(a, "--box") && i + 1 < argc)
      cfg->box = atof(argv[++i]);
    else if (!strcmp(a, "--phi") && i + 1 < argc)
      cfg->phi = atof(argv[++i]);
    else if (!strcmp(a, "--radius") && i + 1 < argc)
      cfg->radius = atof(argv[++i]);
    else if (!strcmp(a, "--jitter") && i + 1 < argc)
      cfg->jitter = atof(argv[++i]);
    else if (!strcmp(a, "--speed") && i + 1 < argc)
      cfg->speed = atof(argv[++i]);
    else if (!strcmp(a, "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
    else if (!strcmp(a, "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(a, "--warmup") && i + 1 < argc)
      cfg->warmup = atoi(argv[++i]);
    else if (!strcmp(a, "--repeat") && i + 1 < argc)
      cfg->repeat = atoi(argv[++i]);
    else if (!strcmp(a, "--max-gen-mb") && i + 1 < argc)
      cfg->max_gen_mb = atof(argv[++i]);
    else if (!strcmp(a, "--quality"))
      cfg->quality = true;
    else if (!strcmp(a, "--normals"))
      cfg->normals = true;
    else if (!strcmp(a, "--no-post-geom"))
      cfg->post_geom = false;
    else if (!strcmp(a, "--sync-phases"))
      cfg->sync_phases = true;
    else if (!strcmp(a, "--migration-stats"))
      cfg->migration_stats = true;
    else if (!strcmp(a, "--per-step"))
      cfg->per_step = true;
    else if (!strcmp(a, "--verify"))
      cfg->verify = true;
    else if (!strcmp(a, "--vtk") && i + 1 < argc)
      cfg->vtk_every = atoi(argv[++i]);
    else if (!strcmp(a, "--vtk-dir") && i + 1 < argc)
      snprintf(cfg->vtk_dir, sizeof cfg->vtk_dir, "%s", argv[++i]);
    else if (!strcmp(a, "--json") && i + 1 < argc)
      snprintf(cfg->json_path, sizeof cfg->json_path, "%s", argv[++i]);
    else if (!strcmp(a, "--vtk-mode") && i + 1 < argc)
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
    else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
    {
      if (rank == 0) fprintf(stderr, "See the header comment of %s for the full option list.\n", __FILE__);
      return MFTL_ERROR;
    }
    else
    {
      if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", a);
      return MFTL_ERROR;
    }
  }

  if (!cfg->cuts_given)
    grid_for(nproc, cfg->cuts);
  else if (cfg->cuts[0] * cfg->cuts[1] * cfg->cuts[2] != nproc)
  {
    if (rank == 0) fprintf(stderr, "Process grid %dx%dx%d does not match %d ranks.\n", cfg->cuts[0], cfg->cuts[1], cfg->cuts[2], nproc);
    return MFTL_ERROR;
  }

  if (cfg->refinements < 0 || cfg->steps < 1 || cfg->warmup < 0 || cfg->repeat < 1)
  {
    if (rank == 0) fprintf(stderr, "refine >= 0, steps >= 1, warmup >= 0, repeat >= 1.\n");
    return MFTL_ERROR;
  }
  if (cfg->mode == MODE_STRONG && cfg->n_spheres < 1)
  {
    if (rank == 0) fprintf(stderr, "--spheres must be >= 1.\n");
    return MFTL_ERROR;
  }
  if (cfg->mode == MODE_WEAK && cfg->spheres_per_rank < 1)
  {
    if (rank == 0) fprintf(stderr, "--spheres-per-rank must be >= 1.\n");
    return MFTL_ERROR;
  }
  if (cfg->phi <= 0.0 || cfg->phi >= 1.0 || cfg->box <= 0.0 || cfg->jitter < 0.0 || cfg->jitter > 0.5)
  {
    if (rank == 0) fprintf(stderr, "0 < phi < 1, box > 0, 0 <= jitter <= 0.5.\n");
    return MFTL_ERROR;
  }
  for (int d = 0; d < 3; ++d)
  {
    if (cfg->domain[d] <= 0.0)
    {
      if (rank == 0) fprintf(stderr, "--domain components must be > 0.\n");
      return MFTL_ERROR;
    }
  }
  return MFTL_SUCCESS;
}

/* ── Placement: a jittered lattice, not uniform random ──────────────────── */

/** @brief The stratification grid the bodies are placed on. */
typedef struct
{
  int n[3];  ///< Cells per axis.
  long cells;  ///< n[0]*n[1]*n[2]; always >= the body count.
  double cell[3];  ///< Cell edge per axis.
} lattice;

/**
 * @brief A near-cubic lattice covering the domain with at least @p n_bodies cells.
 *
 * Uniform random placement has Poisson fluctuations in bodies-per-rank. Those
 * fluctuations change with P and with N, so on a weak-scaling curve they are
 * indistinguishable from a real overhead trend -- the curve would be measuring
 * placement luck. Stratifying removes them by construction.
 *
 * @param      size     Domain extent per axis.
 * @param      n_bodies Bodies to be placed.
 * @param[out] lat      Receives the lattice.
 */
static void lattice_build(const double size[3], int n_bodies, lattice *lat)
{
  const double vol = size[0] * size[1] * size[2];
  const double s = pow((double) n_bodies / vol, 1.0 / 3.0);

  for (int d = 0; d < 3; ++d)
  {
    const int nd = (int) ceil(size[d] * s - 1e-9);
    lat->n[d] = (nd < 1) ? 1 : nd;
  }

  /* Grow whichever axis currently has the LONGEST cells, so the cells stay as
   * close to cubic as an integer grid allows. */
  while ((long) lat->n[0] * lat->n[1] * lat->n[2] < (long) n_bodies)
  {
    int best = 0;
    double longest = size[0] / lat->n[0];
    for (int d = 1; d < 3; ++d)
    {
      const double e = size[d] / lat->n[d];
      if (e > longest)
      {
        longest = e;
        best = d;
      }
    }
    lat->n[best]++;
  }

  lat->cells = (long) lat->n[0] * lat->n[1] * lat->n[2];
  for (int d = 0; d < 3; ++d) lat->cell[d] = size[d] / (double) lat->n[d];
}

/**
 * @brief The radius that gives the requested volume fraction.
 *
 * Deriving it rather than fixing it is what keeps the DENSITY invariant across
 * the iso-triangle ladder and across P: 4096 tiny bodies and 16 large ones then
 * occupy the same fraction of the box, so the ladder varies granularity and
 * nothing else. Capped at 0.45 of the smallest cell edge so neighbouring bodies
 * on the lattice cannot interpenetrate at zero jitter.
 *
 * @param size     Domain extent per axis.
 * @param n_bodies Bodies in the domain.
 * @param phi      Target volume fraction.
 * @param lat      The placement lattice, for the cap.
 * @return The radius to generate every body at.
 */
static double derive_radius(const double size[3], int n_bodies, double phi, const lattice *lat)
{
  const double vol = size[0] * size[1] * size[2];
  const double r = cbrt(3.0 * phi * vol / (4.0 * MFTL_PI * (double) n_bodies));
  const double cap = 0.45 * MFTL_MIN(lat->cell[0], MFTL_MIN(lat->cell[1], lat->cell[2]));
  return MFTL_MIN(r, cap);
}

/**
 * @brief Where body @p body_index starts, as a pure function of its index.
 *
 * The bodies are spread evenly over the lattice with `cell = i * cells / N`
 * rather than occupying the first N cells: filling in raster order would leave
 * the last layer empty and make the density non-uniform along the slowest axis,
 * which is exactly the artefact the lattice exists to remove.
 *
 * @param      lat        The placement lattice.
 * @param      gmin       Lower corner of the domain.
 * @param      size       Domain extent per axis.
 * @param      periodic   Per-axis periodicity.
 * @param      body_index Index of the body.
 * @param      n_bodies   Total bodies.
 * @param      jitter     Jitter amplitude, in cell edges.
 * @param      radius     Body radius, for the non-periodic margin.
 * @param[out] out        The centre.
 */
static void body_centre(const lattice *lat, const double gmin[3], const double size[3], const bool periodic[3], int body_index, int n_bodies, double jitter,
                        double radius, double out[3])
{
  const long cell = ((long) body_index * lat->cells) / (long) n_bodies;
  const int c[3] = {(int) (cell % lat->n[0]), (int) ((cell / lat->n[0]) % lat->n[1]), (int) (cell / ((long) lat->n[0] * lat->n[1]))};

  uint64_t s = sm_seed_for_body(body_index);
  for (int d = 0; d < 3; ++d)
  {
    const double j = (sm_uniform(&s) * 2.0 - 1.0) * jitter * lat->cell[d];
    double p = gmin[d] + ((double) c[d] + 0.5) * lat->cell[d] + j;

    if (periodic[d])
    {
      while (p < gmin[d]) p += size[d];
      while (p >= gmin[d] + size[d]) p -= size[d];
    }
    else
    {
      /* Nothing here reflects at a wall, so a body must not be able to reach one. */
      const double lo = gmin[d] + 2.0 * radius;
      const double hi = gmin[d] + size[d] - 2.0 * radius;
      p = MFTL_MIN(MFTL_MAX(p, lo), hi);
    }
    out[d] = p;
  }
}

/**
 * @brief The velocity of body @p body_index: a per-body direction at a common speed.
 *
 * One global velocity makes every body cross a cut plane on the same step, so
 * migration traffic becomes a spike train that is an artefact of the setup rather
 * than a property of the library. Per-body directions stagger the crossings while
 * keeping the motion a rigid translation, which is what makes the conservation
 * check exact. A non-periodic axis gets zero, because nothing here reflects.
 *
 * The direction is rejection-sampled inside the unit ball so it is isotropic; a
 * component-wise uniform draw would bias it toward the cube's corners. The
 * normalisation is the library's: magnitude UNCLAMPED, then a test against zero.
 *
 * @param      body_index Index of the body.
 * @param      periodic   Per-axis periodicity.
 * @param      speed      Common speed.
 * @param[out] out        The velocity.
 */
static void body_velocity(int body_index, const bool periodic[3], double speed, double out[3])
{
  uint64_t s = sm_seed_for_body(body_index) ^ 0xD1B54A32D192ED03ULL;
  double v[3] = {0.0, 0.0, 0.0};
  double n2 = 0.0;

  for (int attempt = 0; attempt < 64; ++attempt)
  {
    for (int d = 0; d < 3; ++d) v[d] = periodic[d] ? (sm_uniform(&s) * 2.0 - 1.0) : 0.0;
    n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    if (n2 > 0.01 && n2 <= 1.0) break;
    n2 = 0.0;
  }

  const double mag = sqrt(n2);
  const double inv = (mag > 0.0) ? speed / mag : 0.0;
  for (int d = 0; d < 3; ++d) out[d] = v[d] * inv;
}

/* ── Rank-local diagnostics ─────────────────────────────────────────────── */

/**
 * @brief Owned surface area and entity counts, over OWNED entities only.
 *
 * Ghosts are copies; counting them would count their owner's contribution twice.
 * Ownership is an array PREFIX, so there is no is_ghost test anywhere in here.
 *
 * @param      ens The ensemble.
 * @param[out] out {area, triangles, vertices}.
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

/**
 * @brief Owned triangles, ghost triangles, and bodies this rank holds a piece of.
 *
 * The third number is what separates the two ways a rank can be underloaded: a
 * small share of many bodies is a partitioning result, a share of NO bodies is an
 * idle rank, and only the second is fatal to a scaling curve.
 *
 * @param      ens The ensemble.
 * @param[out] out {owned triangles, ghost triangles, bodies touched}.
 */
static void local_halo(const MFTL_Ensemble *ens, long out[3])
{
  out[0] = out[1] = out[2] = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    out[0] += obj->tris.n_local;
    out[1] += obj->tris.n - obj->tris.n_local;
    if (obj->tris.n_local > 0) out[2] += 1;
  }
}

/** @brief Bytes MFTLUpdateGhostPositions has to move for this partition.
 *  @param ens The ensemble.
 *  @return Payload bytes for one sync. */
static long local_ghost_bytes(const MFTL_Ensemble *ens)
{
  long n = 0;
  for (int o = 0; o < ens->num_objects; ++o) n += ens->objects[o]->verts.n - ens->objects[o]->verts.n_local;
  return n * (long) sizeof(MFTL_GhostUpdatePayload);
}

/**
 * @brief Ownership churn and triangle wire volume for the redistribution about to run.
 *
 * Two DIFFERENT numbers, and conflating them is the easy mistake:
 *
 *   churn  owned triangles whose destination is not this rank. MFTLGetOwnerProc
 *          is documented as the inner statement of compute_owners, and
 *          compute_owners applies it to tris.center -- which
 *          MFTLComputeTriangleGeometry has already wrapped into the box -- so
 *          this reproduces the routing rule exactly.
 *
 *   sends  distinct REMOTE destinations summed over owned triangles. A
 *          redistribution sends each owned triangle to as many as seven places
 *          (its centre's owner, its three corners' owners, its three edge
 *          neighbours' owners), because the halo is re-materialised from scratch
 *          on every call. On an advecting mesh the churn is a thin rim and the
 *          wire traffic is the whole surface, so only this one predicts bytes.
 *          sends * sizeof(MFTL_PackedTriangleComplex) is exact for triangles.
 *
 * @warning This streams tris.center, tris.verts, tris.neighbors and verts.pos --
 *          precisely the arrays compute_owners and collect_destinations are about
 *          to read. Calling it immediately before MFTLRedistribute WARMS THE
 *          CACHES FOR THE PHASE BEING TIMED. It is behind --migration-stats for
 *          that reason, and no quoted timing may come from a run that had it on.
 *
 * @param      ens The ensemble; requires current centres and derived topology.
 * @param[out] out {churn, sends}.
 */
static void local_migration(const MFTL_Ensemble *ens, long out[2])
{
  const MFTL_Domain *const dom = &ens->domain;
  const int me = dom->rank;
  long churn = 0, sends = 0;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
    const int nt = obj->tris.n_local;
    if (nt < 1) continue;

    const double *const ctr = obj->tris.center;
    const double *const pos = obj->verts.pos;
    const int *const tv = obj->tris.verts;
    const int *const tn = obj->tris.neighbors;

    for (int t = 0; t < nt; ++t)
    {
      const int base = 3 * t;
      const int own = MFTLGetOwnerProc(dom, &ctr[base]);
      if (own != me) ++churn;

      int cand[7];
      int n = 0;
      cand[n++] = own;
      cand[n++] = MFTLGetOwnerProc(dom, &pos[3 * tv[base + 0]]);
      cand[n++] = MFTLGetOwnerProc(dom, &pos[3 * tv[base + 1]]);
      cand[n++] = MFTLGetOwnerProc(dom, &pos[3 * tv[base + 2]]);
      for (int j = 0; j < 3; ++j)
      {
        const int adj = tn[base + j];
        if (adj != MFTL_ERROR) cand[n++] = MFTLGetOwnerProc(dom, &ctr[3 * adj]);
      }

      int uniq[7];
      const int nu = MFTLGetUniqueIntegers(cand, n, uniq, 7);
      for (int k = 0; k < nu; ++k)
        if (uniq[k] != me) ++sends;
    }
  }

  out[0] = churn;
  out[1] = sends;
}

/* ── Phase timing ───────────────────────────────────────────────────────── */

/**
 * @brief Opens a timed phase, optionally synchronising first.
 *
 * With --sync-phases the barrier is charged to @p wait_acc rather than to the
 * phase, which turns "this collective is slow" into "this collective is WAITING".
 * It is off by default because it perturbs what it measures: barriers remove the
 * natural pipelining between phases, so the total goes up. The measured barrier
 * time is itself the measurement of that overhead, which is why it is reported.
 *
 * @param      sync     Whether to barrier.
 * @param[out] wait_acc Accumulates the barrier duration.
 * @return The phase start time.
 */
static inline double phase_begin(bool sync, double *wait_acc)
{
  if (sync)
  {
    const double b0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    *wait_acc += MPI_Wtime() - b0;
  }
  return MPI_Wtime();
}

/* ── One timestep ───────────────────────────────────────────────────────── */

/** @brief Everything one pass over the timed loop accumulates. */
typedef struct
{
  double acc[N_PHASE];  ///< Seconds per phase, this rank.
  double wait[N_PHASE];  ///< Barrier time per phase under --sync-phases.
  long ghost_bytes;  ///< Ghost-sync payload this rank published.
  long churn;  ///< Owned triangles changing owner, summed over steps.
  long sends;  ///< Triangle wire volume, summed over steps.
  long io_bytes;  ///< Bytes this rank wrote.
  long io_dumps;  ///< Dumps written.
  int failures;  ///< Conservation violations seen.
} run_state;

/**
 * @brief The five-step MFTL timestep, instrumented.
 *
 * The order is the sibling example's and is not negotiable: the ghost sync must
 * precede the geometry pass (a triangle's three corners can belong to three
 * ranks), the geometry pass must cover the GHOSTS too (redistribution routes them
 * by their cached centre, and refreshing only the owned ones makes two ranks
 * disagree about a ghost's destination), and the frames are deliberately not
 * refreshed here because the redistribution rebuilds them on its way out.
 *
 * @param      ens        The ensemble.
 * @param      cfg        The configuration.
 * @param      vel        [3*num_objects] per-body velocity.
 * @param[in,out] st      Accumulators.
 * @param      step_id    Step number, for the dump filename.
 * @param      reference  {area, triangles, vertices} to check against, or MFTL_NULL.
 * @param[out] out_phase  [N_PHASE] this step's phase times, or MFTL_NULL.
 */
static void run_step(MFTL_Ensemble *ens, const config *cfg, const double *vel, run_state *st, int step_id, const double *reference, double *out_phase)
{
  const int rank = ens->domain.rank;
  double ph[N_PHASE], wt[N_PHASE];
  for (int p = 0; p < N_PHASE; ++p) ph[p] = wt[p] = 0.0;

  /* 1. Move what we own, and only what we own. */
  double t0 = phase_begin(cfg->sync_phases, &wt[PH_ADVECT]);
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = ens->objects[o];
    const double dx = vel[3 * o + 0] * cfg->dt;
    const double dy = vel[3 * o + 1] * cfg->dt;
    const double dz = vel[3 * o + 2] * cfg->dt;
    MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += dx; pos[1] += dy; pos[2] += dz; MFTLApplyPeriodicity(&ens->domain, pos););
  }
  ph[PH_ADVECT] += MPI_Wtime() - t0;

  /* 2. Pull in the vertices our neighbours moved. */
  st->ghost_bytes += local_ghost_bytes(ens);
  t0 = phase_begin(cfg->sync_phases, &wt[PH_GHOST]);
  MFTLUpdateGhostPositions(ens);
  ph[PH_GHOST] += MPI_Wtime() - t0;

  /* 3. Every triangle now has current vertices. ALL of them, ghosts included. */
  t0 = phase_begin(cfg->sync_phases, &wt[PH_GEOM]);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
  ph[PH_GEOM] += MPI_Wtime() - t0;

  if (reference)
  {
    /* Rigid per-body translation conserves surface area exactly and
     * redistribution conserves entities exactly, so this needs no tolerance
     * beyond round-off: any drift is a real defect. */
    t0 = MPI_Wtime();
    double local[3], total[3];
    local_totals(ens, local);
    MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double rel = fabs(total[0] - reference[0]) / reference[0];
    if (rel > 1e-9 || total[1] != reference[1] || total[2] != reference[2])
    {
      if (rank == 0)
      {
        fprintf(stderr, "step %d: area %.12f (rel dev %.3e), %d triangles, %d vertices -- expected %.12f / %d / %d\n", step_id, total[0], rel, (int) total[1],
                (int) total[2], reference[0], (int) reference[1], (int) reference[2]);
      }
      st->failures++;
    }
    ph[PH_INSTR] += MPI_Wtime() - t0;
  }

  if (cfg->migration_stats)
  {
    t0 = MPI_Wtime();
    long mig[2];
    local_migration(ens, mig);
    st->churn += mig[0];
    st->sends += mig[1];
    ph[PH_INSTR] += MPI_Wtime() - t0;
  }

  /* 4. Re-decide ownership from the fresh centres. */
  t0 = phase_begin(cfg->sync_phases, &wt[PH_REDIST]);
  MFTLRedistribute(ens);
  ph[PH_REDIST] += MPI_Wtime() - t0;

  /* 5. Restore the geometry the redistribution deliberately did not send. */
  if (cfg->post_geom)
  {
    t0 = phase_begin(cfg->sync_phases, &wt[PH_POST]);
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
    ph[PH_POST] += MPI_Wtime() - t0;
  }

  /* Both go AFTER the redistribution, where the frames are current. */
  if (cfg->quality)
  {
    t0 = phase_begin(cfg->sync_phases, &wt[PH_QUAL]);
    MFTLEnsembleUpdate(ens);
    ph[PH_QUAL] += MPI_Wtime() - t0;
  }
  if (cfg->normals)
  {
    /* The ensemble form, not a loop over MFTLComputeVertexNormals: that one
     * leaves every ghost vertex holding a partial sum -- correctly normalised
     * and quietly wrong. */
    t0 = phase_begin(cfg->sync_phases, &wt[PH_NORM]);
    MFTLEnsembleUpdateVertexNormals(ens);
    ph[PH_NORM] += MPI_Wtime() - t0;
  }

  if (cfg->vtk_every > 0 && step_id >= 0 && step_id % cfg->vtk_every == 0)
  {
    char prefix[448];
    snprintf(prefix, sizeof prefix, "%s/scaling_%s_np%d_%05d", cfg->vtk_dir, cfg->label, ens->domain.nproc, step_id);

    t0 = phase_begin(cfg->sync_phases, &wt[PH_VTK]);
    if (MFTLWriteVtu(ens, prefix, &cfg->vtk) != MFTL_SUCCESS)
    {
      if (rank == 0) fprintf(stderr, "step %d: MFTLWriteVtu failed for %s\n", step_id, prefix);
      st->failures++;
    }
    ph[PH_VTK] += MPI_Wtime() - t0;

    /* Stat what landed, so the I/O rate is measured rather than modelled. This
     * is itself I/O, so it is charged to the instrumentation bucket. */
    t0 = MPI_Wtime();
    char path[512];
    if (cfg->vtk.layout == MFTL_IO_FILE_PER_RANK)
    {
      snprintf(path, sizeof path, "%s_%05d.vtu", prefix, rank);
      st->io_bytes += file_size_bytes(path);
      if (rank == 0)
      {
        snprintf(path, sizeof path, "%s.pvtu", prefix);
        st->io_bytes += file_size_bytes(path);
      }
    }
    else if (rank == 0)
    {
      snprintf(path, sizeof path, "%s.vtu", prefix);
      st->io_bytes += file_size_bytes(path);
    }
    ph[PH_INSTR] += MPI_Wtime() - t0;
    st->io_dumps++;
  }

  for (int p = 0; p < N_PHASE; ++p)
  {
    st->acc[p] += ph[p];
    st->wait[p] += wt[p];
    if (out_phase) out_phase[p] = ph[p];
  }
}

/* ── JSON ───────────────────────────────────────────────────────────────── */

/**
 * @brief Writes a JSON string value with the two characters that must be escaped.
 * @param fp Destination.
 * @param s  The string.
 */
/* ═══════════════════════════════════════════════════════════════════════════
 * Redistribution internals
 *
 * The phase table says the redistribution is the expensive phase. These say
 * which PART of it is, which is the question an architecture decision actually
 * turns on -- and the one that had no answer before #MFTL_RedistributeStats
 * existed, because src/communication/ contained no timers at all.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Nanoseconds one MPI_Wtime() call costs on this machine.
 *
 * The redistribution instrumentation reads the clock once per phase boundary per
 * BODY, so the perturbation scales with the body count. At a thousand droplets it
 * is several per cent of the phase, and the breakdown is only honest if it says
 * so -- hence measured rather than quoted, since MPI implementations differ by an
 * order of magnitude in how MPI_Wtime is implemented.
 *
 * Best of five batches rather than a mean: the quantity wanted is the cost of the
 * call itself, and every disturbance can only add to it.
 *
 * @return Cost of one call, in nanoseconds.
 */
static double measure_clock_cost(void)
{
  const int batch = 20000;
  double best = 1e30;

  for (int r = 0; r < 5; ++r)
  {
    const double t0 = MPI_Wtime();
    double sink = 0.0;
    for (int i = 0; i < batch; ++i) sink += MPI_Wtime();
    const double dt = MPI_Wtime() - t0;

    if (sink == 0.0) best = 0.0;  // keeps the loop observable
    if (dt < best) best = dt;
  }
  return 1e9 * best / (double) batch;
}

/** @brief JSON keys for the redistribution sub-phases, in struct order. */
static const char *const RSTAT_KEY[] = {"setup",      "owners",     "masks",     "destinations", "route",    "pack",   "exchange_counts",
                                        "exchange_post", "wait",    "group",     "reserve",      "kept_gather", "kept_place", "relink",
                                        "arrivals",   "twins",      "derived_topology", "derived_frames", "teardown"};

/** @brief Human labels for the same, padded for the printed table. */
static const char *const RSTAT_LABEL[] = {"setup           ", "owner classify  ", "mask clear      ", "destinations    ", "route decision  ",
                                          "pack            ", "exchange counts ", "exchange post   ", "exchange wait   ", "arrival grouping",
                                          "pool growth     ", "kept: gather    ", "kept: place     ", "kept: relink    ", "arrivals        ",
                                          "twin repair     ", "derived topology", "derived FRAMES  ", "teardown        "};

#define N_RSTAT 19

/** @brief JSON keys for the ghost-sync sub-phases, in struct order. */
static const char *const GSTAT_KEY[] = {"collect", "exchange_counts", "exchange_requests", "owner_table", "answer", "exchange_replies", "apply", "teardown"};

/** @brief Human labels for the same. */
static const char *const GSTAT_LABEL[] = {"collect requests", "exchange counts ", "exchange request", "owner id table  ",
                                          "answer          ", "exchange replies", "apply           ", "teardown        "};

#define N_GSTAT 8

/**
 * @brief Flattens #MFTL_RedistributeStats into the arrays the report and the
 *        JSON both read, and reduces them to rank 0.
 *
 * Sums rather than maxima, for the reason the phase table gives: a per-bucket
 * maximum lands on a different rank for each bucket and the parts stop adding up
 * to the whole. Divide by the rank count for a per-rank mean.
 *
 * @param      st  This rank's counters.
 * @param[out] rt  `[N_RSTAT]` redistribution bucket sums.
 * @param[out] gt  `[N_GSTAT]` ghost-sync bucket sums.
 * @param[out] tot `[2]` the two totals, `{redistribute, ghost sync}`.
 * @param[out] ct  `[26]` counter sums, in the order the caller indexes them.
 */
static void rstats_reduce(const MFTL_RedistributeStats *st, double *rt, double *gt, double *tot, long *ct)
{
  const double lt[N_RSTAT] = {st->t_setup,     st->t_owners,          st->t_masks,           st->t_destinations, st->t_route,
                              st->t_pack,      st->t_exchange_counts, st->t_exchange_data,   st->t_wait,         st->t_group,
                              st->t_reserve,   st->t_kept_gather,     st->t_kept_place,      st->t_relink,       st->t_arrivals,
                              st->t_twins,     st->t_derived_topology, st->t_derived_frames, st->t_teardown};
  const double lg[N_GSTAT] = {st->g_t_collect, st->g_t_counts, st->g_t_requests, st->g_t_table, st->g_t_answer, st->g_t_replies, st->g_t_apply,
                              st->g_t_teardown};
  const double ltot[2] = {st->t_total, st->g_t_total};
  const long lc[26] = {st->n_calls,          st->n_verts_scanned,      st->n_tris_scanned,   st->n_verts_owner_changed,
                       st->n_tris_owner_changed, st->n_verts_kept,     st->n_tris_kept,      st->n_verts_received,
                       st->n_tris_received,  st->n_destinations,       st->n_messages,       st->n_bytes_sent,
                       st->n_bytes_recv,     st->n_route_neighbor,     st->n_route_global,   st->n_objects_visited,
                       st->n_objects_empty_local, st->n_allocs,        st->n_frees,          st->n_bytes_allocated,
                       st->n_gid_table_bytes, st->n_dedup_overflow,    st->n_clock_reads,    st->g_n_ghosts_requested,
                       st->g_n_orphaned,     st->g_n_bytes_allocated};

  MPI_Reduce(lt, rt, N_RSTAT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(lg, gt, N_GSTAT, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(ltot, tot, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(lc, ct, 26, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
}

static void json_str(FILE *fp, const char *s)
{
  fputc('"', fp);
  for (const char *p = s; *p; ++p)
  {
    if (*p == '"' || *p == '\\') fputc('\\', fp);
    fputc(*p, fp);
  }
  fputc('"', fp);
}

/**
 * @brief The p-quantile of an already-sorted array.
 * @param sorted Ascending values.
 * @param n      Element count.
 * @param q      Quantile in [0, 1].
 * @return The value, or 0 for an empty array.
 */
static double quantile(const double *sorted, int n, double q)
{
  if (n < 1) return 0.0;
  int i = (int) (q * (double) (n - 1) + 0.5);
  if (i < 0) i = 0;
  if (i >= n) i = n - 1;
  return sorted[i];
}

/** @brief qsort comparator for doubles.
 *  @param a First element.
 *  @param b Second element.
 *  @return -1, 0 or 1. */
static int cmp_double(const void *a, const void *b)
{
  const double x = *(const double *) a, y = *(const double *) b;
  return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/* ── main ───────────────────────────────────────────────────────────────── */

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

  MFTLRegisterMpiTypes();

  /* ---- Problem size and domain, per mode ------------------------------- */

  const int n_bodies = (cfg.mode == MODE_WEAK) ? cfg.spheres_per_rank * nproc : cfg.n_spheres;

  double gmin[3], gmax[3], size[3];
  for (int d = 0; d < 3; ++d)
  {
    /* WEAK: the domain is the per-rank box times the process grid, so every rank
     * owns a constant physical volume holding a constant number of bodies. The
     * halo fraction and the neighbour count are then invariant and the ideal
     * curve is flat. STRONG: fixed, independent of P. */
    size[d] = (cfg.mode == MODE_WEAK) ? cfg.box * (double) cfg.cuts[d] : cfg.domain[d];
    gmin[d] = -0.5 * size[d];
    gmax[d] = 0.5 * size[d];
  }

  lattice lat;
  lattice_build(size, n_bodies, &lat);
  const double radius = (cfg.radius > 0.0) ? cfg.radius : derive_radius(size, n_bodies, cfg.phi, &lat);

  const long tri_per_body = 20L * (1L << (2 * cfg.refinements));
  const long vert_per_body = 10L * (1L << (2 * cfg.refinements)) + 2L;

  /* Rank 0 builds every sphere whole, and MFTL pools NEVER SHRINK, so this
   * footprint is resident for the whole run rather than transient. Batched
   * generation does not reduce it -- only the body count and the refinement do.
   * See trap 1 in the header comment. */
  const double gen_mb = (double) n_bodies * ((double) tri_per_body * 212.0 + (double) vert_per_body * 88.0) / (1024.0 * 1024.0);
  if (gen_mb > cfg.max_gen_mb)
  {
    if (rank == 0)
    {
      fprintf(stderr,
              "refusing: generation would put %.1f MB on rank 0, over --max-gen-mb %.1f.\n"
              "  %d bodies x refine %d (%ld triangles each). Refinement costs 4x per level;\n"
              "  reduce --refine or the body count, or raise --max-gen-mb if the node has the memory.\n",
              gen_mb, cfg.max_gen_mb, n_bodies, cfg.refinements, tri_per_body);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, gmin, gmax, cfg.periodic, n_bodies);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* MPI_Wtime is only guaranteed LOCAL and monotonic. That is enough, because
   * every quantity here is a difference of two calls on the SAME rank, so a
   * constant per-rank offset cancels exactly. Timestamps are never compared
   * across ranks. What does matter is the resolution -- see trap 4. */
  const double tick = MPI_Wtick();
  int wtime_global = 0;
  {
    int *attr = MFTL_NULL, flag = 0;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_WTIME_IS_GLOBAL, &attr, &flag);
    wtime_global = (flag && attr && *attr) ? 1 : 0;
  }

  char host[128] = "unknown";
  gethostname(host, sizeof host - 1);

  if (rank == 0)
  {
    printf("=== advection scaling | %s | label %s | %d ranks (%dx%dx%d) ===\n", (cfg.mode == MODE_WEAK) ? "WEAK" : "STRONG", cfg.label, nproc, cfg.cuts[0],
           cfg.cuts[1], cfg.cuts[2]);
    printf("  bodies             %8d x %ld tri = %ld triangles, radius %.5g (phi %.4g)\n", n_bodies, tri_per_body, (long) n_bodies * tri_per_body, radius,
           cfg.phi);
    printf("  domain             %8.4g x %.4g x %.4g   periodic %d%d%d   lattice %dx%dx%d\n", size[0], size[1], size[2], (int) cfg.periodic[0],
           (int) cfg.periodic[1], (int) cfg.periodic[2], lat.n[0], lat.n[1], lat.n[2]);
    printf("  loop               %8d steps (+%d warmup) x %d repeat, dt %g, speed %g\n", cfg.steps, cfg.warmup, cfg.repeat, cfg.dt, cfg.speed);
    printf("  rank-0 generation  %8.1f MB resident -- serial, and it grows with the body count\n", gen_mb);
    printf("  MPI_Wtick          %8.3g s   MPI_WTIME_IS_GLOBAL %s\n", tick, wtime_global ? "yes" : "no (durations only)");
    if (cfg.migration_stats) printf("  ** --migration-stats is ON: it warms the caches for the redistribute phase. Do not quote these timings. **\n");
    if (cfg.sync_phases) printf("  ** --sync-phases is ON: barriers remove inter-phase pipelining. Do not quote this wall clock. **\n");
    fflush(stdout);
  }

  /* ---- Setup: generate, then scatter off rank 0 ------------------------- */

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_setup0 = MPI_Wtime();

  for (int o = 0; o < n_bodies; ++o)
  {
    MFTL_Object *obj = MFTLObjectCreate(o);
    MFTLEnsembleAddObject(ens, obj);

    double centre[3];
    body_centre(&lat, gmin, size, cfg.periodic, o, n_bodies, cfg.jitter, radius, centre);

    MFTLGenerateSphere(obj, centre, radius, cfg.refinements);

    /* Ownership is decided from the cached centre, and a fresh mesh has none. */
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  /* NOT a redistribution in the steady-state sense: rank 0 holds everything and
   * no destination is in its 26-neighbourhood, so this takes the global
   * Alltoallv path and moves the whole mesh. Excluded from the loop, not warmed. */
  MFTLRedistribute(ens);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_setup = MPI_Wtime() - t_setup0;

  double *const vel = (double *) malloc((size_t) n_bodies * 3 * sizeof(double));
  if (!vel)
  {
    fprintf(stderr, "Rank %d: out of memory for %d velocities.\n", rank, n_bodies);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  for (int o = 0; o < n_bodies; ++o) body_velocity(o, cfg.periodic, cfg.speed, &vel[3 * o]);

  MFTLEnsembleUpdateGlobalCounts(ens);
  long global_tris = 0, global_verts = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    global_tris += ens->objects[o]->global_n_triangles;
    global_verts += ens->objects[o]->global_n_vertices;
  }

  double reference[3] = {0.0, 0.0, 0.0};
  const double *ref_ptr = MFTL_NULL;
  if (cfg.verify)
  {
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
    double local[3];
    local_totals(ens, local);
    MPI_Allreduce(local, reference, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ref_ptr = reference;
    if (rank == 0) printf("  reference          area %.12f, %d triangles, %d vertices\n", reference[0], (int) reference[1], (int) reference[2]);
  }

  /* ---- Warmup, then the timed repeats ---------------------------------- */

  for (int w = 0; w < cfg.warmup; ++w)
  {
    run_state discard;
    memset(&discard, 0, sizeof discard);
    run_step(ens, &cfg, vel, &discard, -1, MFTL_NULL, MFTL_NULL);
  }

  const size_t series_n = (size_t) cfg.steps * N_PHASE;
  double *const series = (double *) malloc(series_n * sizeof(double));
  double *const best_series = (double *) malloc(series_n * sizeof(double));
  if (!series || !best_series)
  {
    fprintf(stderr, "Rank %d: out of memory for the per-step series.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  run_state best;
  memset(&best, 0, sizeof best);
  double best_wall = 0.0;
  bool have_best = false;

  /* The internals are captured PER REPEAT and kept alongside the winning wall
   * clock. The accumulator is global and would otherwise sum every repeat,
   * describing a run that was never reported. */
  MFTL_RedistributeStats best_rstats;
  memset(&best_rstats, 0, sizeof best_rstats);

  for (int rep = 0; rep < cfg.repeat; ++rep)
  {
    run_state st;
    memset(&st, 0, sizeof st);
    MFTLRedistributeStatsReset();

    MPI_Barrier(MPI_COMM_WORLD);
    const double w0 = MPI_Wtime();
    for (int s = 0; s < cfg.steps; ++s) run_step(ens, &cfg, vel, &st, s, ref_ptr, &series[(size_t) s * N_PHASE]);
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall_local = MPI_Wtime() - w0;

    MFTL_RedistributeStats rstats_this;
    MFTLRedistributeGetStats(&rstats_this);

    /* The selection criterion must be bit-identical on every rank, or two ranks
     * would keep different repeats and the reductions below would mix them. */
    double wall = 0.0;
    MPI_Allreduce(&wall_local, &wall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (!have_best || wall < best_wall)
    {
      best = st;
      best_wall = wall;
      best_rstats = rstats_this;
      have_best = true;
      memcpy(best_series, series, series_n * sizeof(double));
    }
    if (rank == 0 && cfg.repeat > 1) fprintf(stderr, "  repeat %d/%d: %.4f s\n", rep + 1, cfg.repeat, wall);
  }

  /* ---- Reductions ------------------------------------------------------ */

  /* Collective: every rank calls it, only rank 0 receives. Placed with the other
   * reductions so the instrument stays out of the timed region entirely. */
  double rstat_t[N_RSTAT], gstat_t[N_GSTAT], rstat_tot[2];
  long rstat_c[26];
  rstats_reduce(&best_rstats, rstat_t, gstat_t, rstat_tot, rstat_c);
  const double clock_ns = measure_clock_cost();

  /* Three reductions for the phase statistics: sum and sum-of-squares in one
   * MPI_SUM array (mean and variance), then MPI_MAX and MPI_MIN. Doing this once
   * at the end rather than per step keeps the instrument out of the measurement. */
  double packed[2 * N_PHASE], sum[2 * N_PHASE], mx[N_PHASE], mn[N_PHASE], wait_sum[N_PHASE];
  for (int p = 0; p < N_PHASE; ++p)
  {
    packed[p] = best.acc[p];
    packed[N_PHASE + p] = best.acc[p] * best.acc[p];
  }
  MPI_Reduce(packed, sum, 2 * N_PHASE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(best.acc, mx, N_PHASE, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(best.acc, mn, N_PHASE, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(best.wait, wait_sum, N_PHASE, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  long halo[3];
  local_halo(ens, halo);
  long halo_sum[3] = {0, 0, 0}, owned_max = 0, owned_min = 0, bodies_min = 0, bodies_max = 0;
  MPI_Reduce(halo, halo_sum, 3, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[0], &owned_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[0], &owned_min, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[2], &bodies_min, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[2], &bodies_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  const long counters[4] = {best.ghost_bytes, best.churn, best.sends, best.io_bytes};
  long counters_sum[4] = {0, 0, 0, 0};
  MPI_Reduce(counters, counters_sum, 4, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  const long rss = peak_rss_bytes();
  long rss_max = 0, rss_sum = 0;
  MPI_Reduce(&rss, &rss_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&rss, &rss_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  double setup_max = 0.0;
  MPI_Reduce(&t_setup, &setup_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  int failures_total = 0;
  MPI_Allreduce(&best.failures, &failures_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  double *series_mean = MFTL_NULL;
  if (rank == 0) series_mean = (double *) malloc(series_n * sizeof(double));
  MPI_Reduce(best_series, series_mean, (int) series_n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* ---- Report and JSON, rank 0 ----------------------------------------- */

  if (rank == 0)
  {
    double mean[N_PHASE], cv[N_PHASE], imb[N_PHASE], wait_mean[N_PHASE];
    double sum_mean = 0.0, sum_max = 0.0;

    for (int p = 0; p < N_PHASE; ++p)
    {
      mean[p] = sum[p] / (double) nproc;
      double var = sum[N_PHASE + p] / (double) nproc - mean[p] * mean[p];
      if (var < 0.0) var = 0.0;
      cv[p] = (mean[p] > 0.0) ? sqrt(var) / mean[p] : 0.0;
      imb[p] = (mean[p] > 0.0) ? mx[p] / mean[p] : 1.0;
      wait_mean[p] = wait_sum[p] / (double) nproc;
      sum_mean += mean[p];
      sum_max += mx[p];
    }

    for (size_t i = 0; i < series_n; ++i) series_mean[i] /= (double) nproc;

    double *const step_total = (double *) malloc((size_t) cfg.steps * sizeof(double));
    double *const step_redist = (double *) malloc((size_t) cfg.steps * sizeof(double));
    for (int s = 0; s < cfg.steps; ++s)
    {
      double tot = 0.0;
      for (int p = 0; p < N_PHASE; ++p) tot += series_mean[(size_t) s * N_PHASE + p];
      step_total[s] = tot;
      step_redist[s] = series_mean[(size_t) s * N_PHASE + PH_REDIST];
    }
    qsort(step_total, (size_t) cfg.steps, sizeof(double), cmp_double);
    qsort(step_redist, (size_t) cfg.steps, sizeof(double), cmp_double);

    const double per_step = (double) cfg.steps;
    const double owned_mean = (double) halo_sum[0] / (double) nproc;
    const double struct_imb = (owned_mean > 0.0) ? (double) owned_max / owned_mean : 1.0;
    const double halo_frac = (halo_sum[0] > 0) ? (double) halo_sum[1] / (double) halo_sum[0] : 0.0;
    const double floor_s = 100.0 * tick;
    const double throughput = (best_wall > 0.0) ? (double) global_tris * per_step / best_wall : 0.0;

    printf("  -- totals ------------------------------------------------------------\n");
    printf("  wall clock (barriered)     %10.4f s   (%.4f ms/step)   <- the scaling number\n", best_wall, 1000.0 * best_wall / per_step);
    printf("  sum of phase means         %10.4f s                        <- the breakdown that adds up\n", sum_mean);
    printf("  sum of phase maxima        %10.4f s                        <- upper bound on the critical path\n", sum_max);
    printf("  unattributed               %10.4f s   (%.1f %%)\n", best_wall - sum_mean, 100.0 * (best_wall - sum_mean) / best_wall);
    printf("  setup (rank-0 serial)      %10.4f s                        <- NOT part of any scaling total\n", setup_max);
    printf("  throughput                 %10.4g owned-triangle updates / s (all ranks)\n", throughput);

    printf("  -- per phase ---------------------------------------------------------\n");
    printf("  %-18s %10s %6s %10s %9s %7s\n", "phase", "mean s", "%", "ms/step", "max/mean", "cv");
    for (int p = 0; p < N_PHASE; ++p)
    {
      if (mean[p] <= 0.0) continue;
      const bool thin = (mean[p] / per_step) < floor_s;
      printf("  %s %10.4f %6.1f %10.4f %9.2f %7.3f%s\n", PHASE_LABEL[p], mean[p], 100.0 * mean[p] / best_wall, 1000.0 * mean[p] / per_step, imb[p], cv[p],
             thin ? "   ! below 100*MPI_Wtick, not trustworthy" : "");
    }
    printf("  per-step redistribute      p50 %.4f ms   p95 %.4f ms   (a growing tail vs a rising floor)\n", 1000.0 * quantile(step_redist, cfg.steps, 0.50),
           1000.0 * quantile(step_redist, cfg.steps, 0.95));
    if (cfg.sync_phases)
    {
      double wtot = 0.0;
      for (int p = 0; p < N_PHASE; ++p) wtot += wait_mean[p];
      printf("  barrier wait (mean)        %10.4f s   (%.1f %% -- this is --sync-phases measuring its own overhead)\n", wtot, 100.0 * wtot / best_wall);
    }
    printf("  NOTE max/mean is a LOWER bound for a collective phase: the collective has already equalised the finish times.\n");

    {
      const double inv_p = 1.0 / (double) nproc;
      const double red_total = rstat_tot[0] * inv_p;
      const double gs_total = rstat_tot[1] * inv_p;
      const double mb = 1.0 / (1024.0 * 1024.0);

      printf("  -- inside the redistribution (mean per rank) --------------------------\n");
      printf("  %s %10.4f s  %5.1f %%  %8.4f ms/step\n", "TOTAL           ", red_total, 100.0 * red_total / best_wall, 1000.0 * red_total / per_step);
      double acc = 0.0;
      for (int i = 0; i < N_RSTAT; ++i)
      {
        const double v = rstat_t[i] * inv_p;
        acc += v;
        if (v <= 0.0) continue;
        printf("  %s %10.4f s  %5.1f %%  %8.4f ms/step\n", RSTAT_LABEL[i], v, (red_total > 0.0) ? 100.0 * v / red_total : 0.0, 1000.0 * v / per_step);
      }
      printf("  %s %10.4f s  %5.1f %%   <- untimed residue; a big one means a phase has no timer\n", "residue         ", red_total - acc,
             (red_total > 0.0) ? 100.0 * (red_total - acc) / red_total : 0.0);

      printf("  -- inside the ghost sync (mean per rank) ------------------------------\n");
      printf("  %s %10.4f s  %5.1f %%  %8.4f ms/step\n", "TOTAL           ", gs_total, 100.0 * gs_total / best_wall, 1000.0 * gs_total / per_step);
      acc = 0.0;
      for (int i = 0; i < N_GSTAT; ++i)
      {
        const double v = gstat_t[i] * inv_p;
        acc += v;
        if (v <= 0.0) continue;
        printf("  %s %10.4f s  %5.1f %%  %8.4f ms/step\n", GSTAT_LABEL[i], v, (gs_total > 0.0) ? 100.0 * v / gs_total : 0.0, 1000.0 * v / per_step);
      }
      printf("  %s %10.4f s  %5.1f %%\n", "residue         ", gs_total - acc, (gs_total > 0.0) ? 100.0 * (gs_total - acc) / gs_total : 0.0);

      printf("  -- what the redistribution moved (all ranks, per step) ----------------\n");
      printf("  entities classified        %10.0f v + %.0f t   (%.2f %% / %.2f %% changed owner)\n", (double) rstat_c[1] / per_step,
             (double) rstat_c[2] / per_step, (rstat_c[1] > 0) ? 100.0 * (double) rstat_c[3] / (double) rstat_c[1] : 0.0,
             (rstat_c[2] > 0) ? 100.0 * (double) rstat_c[4] / (double) rstat_c[2] : 0.0);
      printf("  kept / received            %10.0f / %.0f v      %.0f / %.0f t\n", (double) rstat_c[5] / per_step, (double) rstat_c[7] / per_step,
             (double) rstat_c[6] / per_step, (double) rstat_c[8] / per_step);
      printf("  wire volume                %10.3f MB/step out   %.3f MB/step in\n", (double) rstat_c[11] * mb / per_step,
             (double) rstat_c[12] * mb / per_step);
      printf("  messages / destinations    %10.0f / %.0f per step\n", (double) rstat_c[10] / per_step, (double) rstat_c[9] / per_step);
      printf("  route neighbour / global   %10ld / %ld calls\n", rstat_c[13], rstat_c[14]);
      printf("  bodies visited / EMPTY     %10.0f / %.0f per step   (%.1f %% of the per-body work is on nothing)\n", (double) rstat_c[15] / per_step,
             (double) rstat_c[16] / per_step, (rstat_c[15] > 0) ? 100.0 * (double) rstat_c[16] / (double) rstat_c[15] : 0.0);
      printf("  allocations                %10.0f /step        %.3f MB/step requested\n", (double) rstat_c[17] / per_step,
             (double) rstat_c[19] * mb / per_step);
      printf("  of which the id tables     %10.3f MB/step      <- sized from the GLOBAL id range, not the local one\n",
             (double) rstat_c[20] * mb / per_step);
      if (rstat_c[21] > 0) printf("  dedup budget exhausted     %10ld times\n", rstat_c[21]);
      printf("  clock reads (instrument)   %10.0f /rank/step   %.4f ms/step at %.1f ns each\n", (double) rstat_c[22] / per_step / (double) nproc,
             (double) rstat_c[22] / per_step / (double) nproc * clock_ns * 1e-6, clock_ns);
    }

    printf("  -- partition ---------------------------------------------------------\n");
    printf("  owned triangles            %10ld   (mean %.0f, min %ld, max %ld per rank)\n", halo_sum[0], owned_mean, owned_min, owned_max);
    printf("  structural imbalance       %10.3f   (busiest rank / average owned triangles)\n", struct_imb);
    printf("  halo                       %10ld   (%.2f %% of owned)\n", halo_sum[1], 100.0 * halo_frac);
    printf("  bodies touched per rank    %10ld   (min) .. %ld (max) of %d\n", bodies_min, bodies_max, n_bodies);
    printf("  ghost sync volume          %10.3f MB/step (all ranks)\n", (double) counters_sum[0] / (1024.0 * 1024.0) / per_step);
    if (cfg.migration_stats)
    {
      printf("  ownership churn            %10.1f triangles/step (all ranks)\n", (double) counters_sum[1] / per_step);
      printf("  triangle wire volume       %10.3f MB/step  (%.1f tri/step; vertices bounded by 3x)\n",
             (double) counters_sum[2] * (double) sizeof(MFTL_PackedTriangleComplex) / (1024.0 * 1024.0) / per_step, (double) counters_sum[2] / per_step);
    }
    printf("  peak RSS                   %10.1f MB (busiest rank), %.1f MB (all ranks)\n", (double) rss_max / (1024.0 * 1024.0),
           (double) rss_sum / (1024.0 * 1024.0));
    if (best.io_dumps > 0)
    {
      const double mb = (double) counters_sum[3] / (1024.0 * 1024.0);
      printf("  vtk output                 %10ld dumps, %.1f MB, %.1f MB/s\n", best.io_dumps, mb, (mean[PH_VTK] > 0.0) ? mb / mean[PH_VTK] : 0.0);
    }
    printf("======================================================================\n");

    if (cfg.json_path[0])
    {
      FILE *fp = fopen(cfg.json_path, "w");
      if (!fp)
      {
        fprintf(stderr, "could not open %s for writing\n", cfg.json_path);
      }
      else
      {
        char libver[MPI_MAX_LIBRARY_VERSION_STRING] = "unknown";
        int liblen = 0;
        MPI_Get_library_version(libver, &liblen);
        for (char *q = libver; *q; ++q)
          if (*q == '\n' || *q == '\r')
          {
            *q = '\0';
            break;
          }

        fprintf(fp, "{\n  \"schema\": 1,\n");
        fprintf(fp, "  \"label\": ");
        json_str(fp, cfg.label);
        fprintf(fp, ",\n  \"mode\": ");
        json_str(fp, (cfg.mode == MODE_WEAK) ? "weak" : "strong");
        fprintf(fp, ",\n  \"git\": ");
        json_str(fp, cfg.git);
        fprintf(fp, ",\n  \"hostname\": ");
        json_str(fp, host);
        fprintf(fp, ",\n  \"mpi\": ");
        json_str(fp, libver);
        fprintf(fp, ",\n  \"timestamp\": %lld,\n", (long long) time(MFTL_NULL));
        fprintf(fp, "  \"nranks\": %d,\n  \"cuts\": [%d, %d, %d],\n", nproc, cfg.cuts[0], cfg.cuts[1], cfg.cuts[2]);
        fprintf(fp, "  \"periodic\": [%d, %d, %d],\n", (int) cfg.periodic[0], (int) cfg.periodic[1], (int) cfg.periodic[2]);
        fprintf(fp, "  \"n_bodies\": %d,\n  \"refinements\": %d,\n  \"tris_per_body\": %ld,\n", n_bodies, cfg.refinements, tri_per_body);
        fprintf(fp, "  \"radius\": %.17g,\n  \"phi\": %.17g,\n  \"jitter\": %.17g,\n", radius, cfg.phi, cfg.jitter);
        fprintf(fp, "  \"domain_size\": [%.17g, %.17g, %.17g],\n", size[0], size[1], size[2]);
        fprintf(fp, "  \"lattice\": [%d, %d, %d],\n", lat.n[0], lat.n[1], lat.n[2]);
        fprintf(fp, "  \"steps\": %d,\n  \"warmup\": %d,\n  \"repeat\": %d,\n  \"dt\": %.17g,\n  \"speed\": %.17g,\n", cfg.steps, cfg.warmup, cfg.repeat,
                cfg.dt, cfg.speed);
        fprintf(fp, "  \"sync_phases\": %s,\n  \"migration_stats\": %s,\n", cfg.sync_phases ? "true" : "false", cfg.migration_stats ? "true" : "false");
        fprintf(fp, "  \"quality\": %s,\n  \"normals\": %s,\n  \"post_geom\": %s,\n", cfg.quality ? "true" : "false", cfg.normals ? "true" : "false",
                cfg.post_geom ? "true" : "false");
        fprintf(fp, "  \"global_triangles\": %ld,\n  \"global_vertices\": %ld,\n", global_tris, global_verts);
        fprintf(fp, "  \"gen_mb_rank0\": %.17g,\n  \"t_setup_s\": %.17g,\n", gen_mb, setup_max);
        fprintf(fp, "  \"mpi_wtick\": %.17g,\n  \"wtime_is_global\": %s,\n", tick, wtime_global ? "true" : "false");
        fprintf(fp, "  \"wall_s\": %.17g,\n  \"sum_phase_means_s\": %.17g,\n  \"sum_phase_maxima_s\": %.17g,\n", best_wall, sum_mean, sum_max);
        fprintf(fp, "  \"throughput_tri_per_s\": %.17g,\n", throughput);
        fprintf(fp, "  \"owned_tris\": {\"min\": %ld, \"max\": %ld, \"mean\": %.17g, \"total\": %ld},\n", owned_min, owned_max, owned_mean, halo_sum[0]);
        fprintf(fp, "  \"bodies_touched\": {\"min\": %ld, \"max\": %ld},\n", bodies_min, bodies_max);
        fprintf(fp, "  \"halo_fraction\": %.17g,\n  \"structural_imbalance\": %.17g,\n", halo_frac, struct_imb);
        fprintf(fp, "  \"ghost_bytes_per_step\": %.17g,\n", (double) counters_sum[0] / per_step);
        fprintf(fp, "  \"churn_tris_per_step\": %.17g,\n  \"wire_tris_per_step\": %.17g,\n", (double) counters_sum[1] / per_step,
                (double) counters_sum[2] / per_step);
        fprintf(fp, "  \"maxrss_bytes_max\": %ld,\n  \"maxrss_bytes_sum\": %ld,\n", rss_max, rss_sum);
        fprintf(fp, "  \"redist_p50_s\": %.17g,\n  \"redist_p95_s\": %.17g,\n", quantile(step_redist, cfg.steps, 0.50), quantile(step_redist, cfg.steps, 0.95));
        fprintf(fp, "  \"step_p50_s\": %.17g,\n  \"step_p95_s\": %.17g,\n", quantile(step_total, cfg.steps, 0.50), quantile(step_total, cfg.steps, 0.95));
        fprintf(fp, "  \"io\": {\"dumps\": %ld, \"bytes\": %ld},\n", best.io_dumps, counters_sum[3]);
        fprintf(fp, "  \"timer_floor_s\": %.17g,\n", floor_s);

        fprintf(fp, "  \"phases\": {\n");
        for (int p = 0; p < N_PHASE; ++p)
        {
          fprintf(fp, "    \"%s\": {\"mean_s\": %.17g, \"max_s\": %.17g, \"min_s\": %.17g, \"cv\": %.17g, \"imbalance\": %.17g, \"wait_s\": %.17g}%s\n",
                  PHASE_KEY[p], mean[p], mx[p], mn[p], cv[p], imb[p], wait_mean[p], (p + 1 < N_PHASE) ? "," : "");
        }
        fprintf(fp, "  },\n");

        if (cfg.per_step)
        {
          fprintf(fp, "  \"per_step_s\": [");
          for (int s = 0; s < cfg.steps; ++s)
          {
            double tot = 0.0;
            for (int p = 0; p < N_PHASE; ++p) tot += series_mean[(size_t) s * N_PHASE + p];
            fprintf(fp, "%s%.17g", (s > 0) ? ", " : "", tot);
          }
          fprintf(fp, "],\n");
        }

        fprintf(fp, "  \"redistribute_internals_s\": {\n");
        for (int i = 0; i < N_RSTAT; ++i)
          fprintf(fp, "    \"%s\": %.17g%s\n", RSTAT_KEY[i], rstat_t[i] / (double) nproc, (i + 1 < N_RSTAT) ? "," : "");
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"ghost_sync_internals_s\": {\n");
        for (int i = 0; i < N_GSTAT; ++i)
          fprintf(fp, "    \"%s\": %.17g%s\n", GSTAT_KEY[i], gstat_t[i] / (double) nproc, (i + 1 < N_GSTAT) ? "," : "");
        fprintf(fp, "  },\n");
        fprintf(fp, "  \"redistribute_total_s\": %.17g,\n  \"ghost_sync_total_s\": %.17g,\n", rstat_tot[0] / (double) nproc,
                rstat_tot[1] / (double) nproc);
        fprintf(fp, "  \"redistribute_counters\": {\"calls\": %ld, \"verts_scanned\": %ld, \"tris_scanned\": %ld, \"verts_moved\": %ld, "
                    "\"tris_moved\": %ld, \"verts_kept\": %ld, \"tris_kept\": %ld, \"verts_recv\": %ld, \"tris_recv\": %ld, "
                    "\"destinations\": %ld, \"messages\": %ld, \"bytes_sent\": %ld, \"bytes_recv\": %ld, \"route_neighbor\": %ld, "
                    "\"route_global\": %ld, \"bodies_visited\": %ld, \"bodies_empty\": %ld, \"allocs\": %ld, \"frees\": %ld, "
                    "\"bytes_allocated\": %ld, \"gid_table_bytes\": %ld, \"dedup_overflow\": %ld, \"clock_reads\": %ld, "
                    "\"ghosts_requested\": %ld, \"ghosts_orphaned\": %ld},\n",
                rstat_c[0], rstat_c[1], rstat_c[2], rstat_c[3], rstat_c[4], rstat_c[5], rstat_c[6], rstat_c[7], rstat_c[8], rstat_c[9], rstat_c[10],
                rstat_c[11], rstat_c[12], rstat_c[13], rstat_c[14], rstat_c[15], rstat_c[16], rstat_c[17], rstat_c[18], rstat_c[19], rstat_c[20],
                rstat_c[21], rstat_c[22], rstat_c[23], rstat_c[24]);
        fprintf(fp, "  \"clock_cost_ns\": %.17g,\n", clock_ns);
        fprintf(fp, "  \"failures\": %d\n}\n", failures_total);
        fclose(fp);
      }
    }

    free(step_total);
    free(step_redist);
    free(series_mean);
  }

  if (rank == 0 && cfg.verify) printf("%s\n", (failures_total == 0) ? "VERIFY: conservation held on every step." : "VERIFY: FAILED -- see above.");

  free(series);
  free(best_series);
  free(vel);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (failures_total > 0) ? 1 : 0;
}
