/*
 * @Author: Christian Gorges
 * @Date: 2026-08-18
 *
 * @brief The four parallel remeshing regimes, measured against each other on one
 *        grid, and the strategy rule checked against what is actually faster.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE QUESTION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * MFTLRemeshChooseStrategy decides between the two distributed drivers with ONE
 * ratio -- the largest body radius against 0.4 * h_min -- and nothing measures
 * whether that is the right decision rule. It cannot be checked by a unit test
 * either: a test can assert which strategy was chosen, but "chosen correctly"
 * means "chosen the cheaper one", and cheapness is a measurement.
 *
 * So this runs BOTH drivers on the same fixture, in each of the four regimes the
 * implementation plan names, and prints their costs side by side together with
 * what AUTO picked. A row where AUTO's choice is not the cheaper one is the
 * result worth having.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE FOUR REGIMES, AND WHY THEY ARE FOUR AND NOT TWO
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * There are two STRATEGIES and four REGIMES, and the mismatch is the point:
 *
 *   A  DENSE      N_bodies >> P, R << h.  An emulsion or a spray. Dispersed
 *                 wins outright: two migrations for the whole ensemble, no
 *                 frozen entities, no phases.
 *   B  MATCHED    N_bodies ~ P, R ~ h.    The threshold itself. Either driver is
 *                 defensible and the ratio decides; this is the row where the
 *                 0.4 constant is on trial.
 *   C  SPARSE     N_bodies << P.          Few bodies, many ranks. Dispersed is
 *                 correct and leaves most ranks IDLE -- and no strategy fixes
 *                 that, because there is no work to give them. What the run
 *                 shows is the size of the idleness.
 *   D  LARGE      R >> h.                 A jet or a large interface. Only the
 *                 spanning schedule applies; dispersed detects it and falls back
 *                 at the cost of one wasted measurement.
 *
 * Regime C has a second half that IS fixable and is the one `parallel_coarsen`
 * addresses: raise P far enough and the rank box stops being wider than four
 * frozen bands, at which point the spanning schedule degrades silently. The C
 * row is therefore run twice, with coarsening off and on.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT MAY AND MAY NOT BE COMPARED
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Never entity by entity, and never across rank counts. Remeshing is a sequence
 * of local decisions each taken against the mesh the previous one produced, so
 * two orderings give different, equally valid triangulations. What is compared
 * here is COST (phases, migrations, entities on the wire, wall clock, imbalance)
 * and gross OUTCOME (triangle count, enclosed volume) -- the same rules
 * examples/leveque_remeshing_parallel.c states for itself.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OPTIONS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   -refine N    subdivisions per droplet   (default 2; regime A is N_bodies of them)
 *   -budget N    parallel_phase_budget      (default 8)
 *   -dense N     bodies in regime A         (default 32)
 *   -only X      any subset of the section letters, e.g. -only fgl
 *                a b c d  the four regimes of the implementation plan
 *                f        does the frozen-fraction trigger pay?
 *                g        where is the single-rank gather crossover?
 *                l        what does the dispersed load balancer buy?
 *                (default "abcd" -- the new sections are opt-in because each
 *                 sweeps several configurations and they are slow)
 *   -walls       non-periodic domain
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

static int g_rank = 0;
static int g_nproc = 1;

static int arg_int(int argc, char **argv, const char *flag, int fallback)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return atoi(argv[i + 1]);
  return fallback;
}

static const char *arg_str(int argc, char **argv, const char *flag, const char *fallback)
{
  for (int i = 1; i + 1 < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return argv[i + 1];
  return fallback;
}

static bool arg_flag(int argc, char **argv, const char *flag)
{
  for (int i = 1; i < argc; ++i)
    if (strcmp(argv[i], flag) == 0) return MFTL_TRUE;
  return MFTL_FALSE;
}

static const char *strategy_name(MFTL_RemeshParallelStrategy s)
{
  switch (s)
  {
    case MFTL_REMESH_PARALLEL_SPANNING: return "spanning";
    case MFTL_REMESH_PARALLEL_DISPERSED: return "dispersed";
    default: return "auto";
  }
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief N droplets of radius @p r on the coarsest lattice that holds them.
 *
 * Placed on a lattice rather than at random for one reason that matters here:
 * every rank must build the SAME ensemble, and a lattice is a pure function of
 * (n, domain) while a random placement would need a seed agreed across ranks.
 * Bodies are identified on the wire by their index in the object list, so two
 * ranks disagreeing about the list is not a cosmetic problem.
 */
static MFTL_Ensemble *build(int n_bodies, double r, int refine, bool periodic, int cuts[3])
{
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, MFTL_MAX(n_bodies, 2));

  int side = 1;
  while (side * side * side < n_bodies) side++;

  const double step = 2.0 / (double) side;

  for (int b = 0; b < n_bodies; ++b)
  {
    MFTL_Object *obj = MFTLObjectCreate(b);
    MFTLEnsembleAddObject(ens, obj);

    const int ix = b % side;
    const int iy = (b / side) % side;
    const int iz = (b / (side * side)) % side;

    double c[3] = {-1.0 + ((double) ix + 0.5) * step, -1.0 + ((double) iy + 0.5) * step, -1.0 + ((double) iz + 0.5) * step};

    MFTLGenerateSphere(obj, c, r, refine);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  MFTLRedistribute(ens);
  return ens;
}

/**
 * @brief The same bodies, packed into ONE rank's box.
 *
 * The load balancer's fixture, and the reason it needs a separate one: bodies go
 * where their centroids are, so an ensemble spread over a lattice is already
 * roughly balanced and measuring a balancer on it measures nothing. A real
 * emulsion is not uniform over the process grid either -- droplets collect where
 * the flow puts them -- so this is the shape of the problem rather than a
 * pathological case invented to make a number move.
 *
 * The box is resolved against the grid at run time, not hardcoded: `cuts` is
 * 2x2x2 at np = 8 and 3x3x3 at np = 27, and a coordinate that piles up on one is
 * spread across the other.
 *
 * @param n_bodies Bodies to place.
 * @param r        Radius of each.
 * @param refine   Icosahedron subdivisions.
 * @param periodic Whether the domain wraps.
 * @param cuts     The process grid.
 * @return The ensemble, already redistributed.
 */
static MFTL_Ensemble *build_clustered(int n_bodies, double r, int refine, bool periodic, int cuts[3])
{
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, MFTL_MAX(n_bodies, 2));

  double box[3];
  for (int k = 0; k < 3; ++k) box[k] = 2.0 / (double) cuts[k];

  int side = 1;
  while (side * side * side < n_bodies) side++;

  for (int b = 0; b < n_bodies; ++b)
  {
    MFTL_Object *obj = MFTLObjectCreate(b);
    MFTLEnsembleAddObject(ens, obj);

    /* Inset by one step from both faces so no droplet leaks into a neighbouring
     * box -- which would balance the fixture by accident and quietly weaken it. */
    const int ix = b % side;
    const int iy = (b / side) % side;
    const int iz = (b / (side * side)) % side;

    double c[3] = {-1.0 + box[0] * ((double) ix + 1.0) / ((double) side + 1.0), -1.0 + box[1] * ((double) iy + 1.0) / ((double) side + 1.0),
                   -1.0 + box[2] * ((double) iz + 1.0) / ((double) side + 1.0)};

    MFTLGenerateSphere(obj, c, r, refine);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  MFTLRedistribute(ens);
  return ens;
}

/** @brief Owned triangles on the busiest rank against the mean. 1.0 is perfect. */
static double imbalance(const MFTL_Ensemble *ens)
{
  double mine = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o) mine += (double) ens->local_objects[o]->tris.n_local;

  double sum = mine, max = mine;
  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  return (sum > 0.0) ? max / (sum / (double) g_nproc) : 0.0;
}

/**
 * @brief Owned triangles over the WHOLE ensemble.
 *
 * Reduced, not read off `global_n_triangles`. That field is the count for ONE
 * body and every holder of it carries the same number, so summing it over a
 * rank's local list answers "how big are the bodies this rank happens to hold" --
 * which used to be the whole ensemble, because every rank held every body, and
 * is now a rank-dependent fraction of it. The printed column then halves when
 * the rank count doubles, for no reason connected to the mesh.
 *
 * @param ens The ensemble.
 * @return The total, identical on every rank.
 * @collective
 */
static int global_triangles(const MFTL_Ensemble *ens)
{
  int n = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) n += ens->local_objects[o]->tris.n_local;
  MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return n;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * One measurement
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @struct sweep_config
 * @brief One row's worth of knobs.
 *
 * A struct rather than more positional arguments, because the three features
 * this file now measures each add one and a twelve-argument call is a place
 * where two doubles get swapped and nobody notices for a week. The zero value is
 * the library default for every field that has one, so a row only names what it
 * is varying.
 */
typedef struct
{
  int n_bodies;
  double r;
  int refine;
  int budget;
  int coarsen;  ///< `parallel_coarsen`. 0 disarms, 1 arms, >1 caps.
  double frozen;  ///< `coarsen_frozen_fraction`. 0 means "leave the default", i.e. disabled.
  int gather;  ///< `serial_gather_max_triangles`. 0 disables.
  double slack;  ///< `dispersed_balance_slack`. Negative means "leave the default".
  bool cluster;  ///< Pack every body into ONE rank's box instead of spreading them.
  MFTL_RemeshParallelStrategy forced;
} sweep_config;

static void run_one(const char *regime, const char *note, const sweep_config *cfg, bool periodic, int cuts[3])
{
  const int n_bodies = cfg->n_bodies;
  const double r = cfg->r;

  MFTL_Ensemble *ens = cfg->cluster ? build_clustered(n_bodies, r, cfg->refine, periodic, cuts) : build(n_bodies, r, cfg->refine, periodic, cuts);

  const double mean_edge = MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 0.7 * mean_edge, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = 2;
  params.smooth_relaxation = 0.5;
  params.parallel_phase_budget = cfg->budget;
  params.parallel_coarsen = cfg->coarsen;
  params.parallel_strategy = cfg->forced;

  if (cfg->frozen > 0.0) params.coarsen_frozen_fraction = cfg->frozen;
  if (cfg->gather > 0) params.serial_gather_max_triangles = cfg->gather;
  if (cfg->slack >= 0.0) params.dispersed_balance_slack = cfg->slack;

  const int tris_before = global_triangles(ens);

  MPI_Barrier(MPI_COMM_WORLD);
  const double t0 = MPI_Wtime();

  MFTL_RemeshStats st;
  const int rc = MFTLRemeshEnsemble(ens, &params, &st);

  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t0;

  const double bal = imbalance(ens);
  const int tris_after = global_triangles(ens);

  if (g_rank == 0)
  {
    /* `frz` and `trg` are blank on a row where no pre-pass ran, rather than
     * printed as zero: 0.0 % frozen and "not measured" are different statements
     * and a column that renders them alike is the reason a reader draws the
     * wrong conclusion from a table. */
    char frz[8] = "     -";
    if (st.n_coarse_groups > 1 || st.frozen_fraction > 0.0) snprintf(frz, sizeof(frz), "%5.1f", 100.0 * st.frozen_fraction);

    printf("%-2s %-11s %-9s %6d %6.3f %7d %7d %4d %4d %5d %s %3d %6d %6d %8.4f %7.2f  %s\n", regime, note, strategy_name(st.strategy), n_bodies, r,
           tris_before, tris_after, st.n_phases, st.n_migrations, st.n_coarse_groups, frz, st.coarsen_trigger, st.n_edges_hidden_long,
           st.n_edges_hidden_short, wall, bal, (rc != MFTL_SUCCESS) ? "FAILED" : (st.converged ? "converged" : "NOT converged"));

    /* The dispersed fields are printed on their own line only when they say
     * something: after a spanning pass they are all zero and a second line of
     * zeros per row makes the table unreadable. */
    if (st.strategy == MFTL_REMESH_PARALLEL_DISPERSED)
    {
      printf("   %-11s   balance: imbalance %.2f, %d/%d ranks idle, %d/%d bodies reassigned, busiest rank %d ops\n", "", st.load_imbalance,
             st.n_ranks_idle, g_nproc, st.n_bodies_reassigned, n_bodies, st.n_ops_busiest_rank);
    }
  }

  MFTLEnsembleDestroy(&ens);
}

/** @brief AUTO first, then both drivers forced, so the choice can be judged. */
static void sweep(const char *regime, const char *what, const sweep_config *cfg, bool periodic, int cuts[3])
{
  sweep_config c = *cfg;

  c.forced = MFTL_REMESH_PARALLEL_AUTO;
  run_one(regime, what, &c, periodic, cuts);

  c.forced = MFTL_REMESH_PARALLEL_SPANNING;
  run_one(regime, "  forced", &c, periodic, cuts);

  c.forced = MFTL_REMESH_PARALLEL_DISPERSED;
  run_one(regime, "  forced", &c, periodic, cuts);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  const int refine = arg_int(argc, argv, "-refine", 2);
  const int budget = arg_int(argc, argv, "-budget", 8);
  const int dense = arg_int(argc, argv, "-dense", 32);
  const bool periodic = !arg_flag(argc, argv, "-walls");
  const char *only = arg_str(argc, argv, "-only", "abcd");

  int cuts[3] = {0, 0, 0};
  MPI_Dims_create(g_nproc, 3, cuts);

  /* h_min needs a domain, and a domain needs an ensemble, so build a throwaway
   * one purely to ask. Cheaper than duplicating the arithmetic and then having
   * the two disagree when MFTLDomainGetHMin's degenerate-axis rule changes. */
  double h_min;
  {
    MFTL_Ensemble *probe = build(1, 0.1, 0, periodic, cuts);
    h_min = MFTLDomainGetHMin(&probe->domain);
    MFTLEnsembleDestroy(&probe);
  }

  if (g_rank == 0)
  {
    printf("\n=== The four parallel remeshing regimes (np = %d, grid %dx%dx%d, %s) ===\n", g_nproc, cuts[0], cuts[1], cuts[2],
           periodic ? "periodic" : "walls");
    printf("h_min = %.4f; the strategy threshold is R > %.4f (0.4 * h_min)\n\n", h_min, 0.4 * h_min);
    printf("rg what        strategy  bodies      R  tri_in tri_out   ph  mig  grps  frz%% trg  hLONG hSHORT     wall  imbal  verdict\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");
  }

  /* A: many small bodies. R well under the threshold, so AUTO must say dispersed
   *    and the forced spanning run is the cost of getting it wrong. */
  sweep_config base = {0};
  base.refine = refine;
  base.budget = budget;
  base.slack = -1.0;  // leave dispersed_balance_slack at its default

  if (strchr(only, 'a'))
  {
    sweep_config c = base;
    c.n_bodies = dense;
    c.r = 0.15 * h_min;
    sweep("A", "dense", &c, periodic, cuts);
  }

  /* B: the threshold itself, at 0.4 * h_min from below and above. Two rows,
   *    because the interesting question is whether the answer flips where the
   *    rule says it should AND whether the cost flips at the same place. */
  if (strchr(only, 'b'))
  {
    sweep_config c = base;
    c.n_bodies = g_nproc;

    c.r = 0.35 * h_min;
    sweep("B", "just under", &c, periodic, cuts);

    c.r = 0.45 * h_min;
    sweep("B", "just over", &c, periodic, cuts);
  }

  /* C: one body, many ranks. Correct on either driver; the number to look at is
   *    `imbal`, which is the idleness nothing can fix. The second pair is the
   *    half that CAN be fixed -- a body large enough that the frozen band no
   *    longer fits in a rank box -- run without and with coarsening. */
  if (strchr(only, 'c'))
  {
    sweep_config c = base;
    c.n_bodies = 1;
    c.r = 0.2 * h_min;
    sweep("C", "sparse", &c, periodic, cuts);

    c.r = 1.2 * h_min;
    c.forced = MFTL_REMESH_PARALLEL_SPANNING;
    c.coarsen = 0;
    run_one("C", "band>h/4", &c, periodic, cuts);
    c.coarsen = 1;
    run_one("C", "  +coarsen", &c, periodic, cuts);
  }

  /* D: one body far larger than a rank box. Dispersed must detect it and fall
   *    back rather than serialising the largest object in the run onto one rank,
   *    so the third row of this sweep is a test of the admission check. */
  if (strchr(only, 'd'))
  {
    sweep_config c = base;
    c.n_bodies = 1;
    c.r = 0.7;
    sweep("D", "large", &c, periodic, cuts);
  }

  /* ── F: does the FROZEN trigger pay? ────────────────────────────────────
   *
   * The question this section exists to answer, and it is a trade rather than a
   * bug hunt. `coarsen_frozen_fraction` coarsens when too much of the mesh is
   * frozen, which costs `grps` in parallelism and buys `1/(1-frozen)` in useful
   * work per migration. Whether that is worth it is a property of the run.
   *
   * Read the rows against the FIRST one, which is the trigger disabled. If the
   * wall clock does not fall as `frz%` does, the trigger is not paying, and the
   * default (1.0, i.e. disabled) is right.
   *
   * ── THE RADIUS IS SCALED TO THE DOMAIN, NOT TO h_min ──────────────────
   *
   * Every other section sizes its bodies against the rank box, which is right
   * for a strategy question. It is wrong here. `0.35 * h_min` shrinks with the
   * rank count, so at np = 27 the body fits inside a single box, NOTHING IS
   * FROZEN, and every row reports a frozen share of zero however the threshold
   * is set -- a table of four identical rows that looks like a measurement.
   * A fixed radius spans ranks at every np, which is the only configuration in
   * which this trigger means anything.
   *
   * `refine` is what moves the frozen share: the band is about one edge long, so
   * halving the edge halves delta/h. Sweep it (-refine 2, 3, 4) to walk the
   * column, and watch `trg` -- a row showing 1 or 3 has the CORRECTNESS trigger
   * in it and is not evidence about this one. */
  if (strchr(only, 'f'))
  {
    const double thresholds[] = {1.0, 0.50, 0.33, 0.25};

    sweep_config c = base;
    c.n_bodies = 1;
    c.r = 0.7;
    c.coarsen = 1;
    c.forced = MFTL_REMESH_PARALLEL_SPANNING;

    for (int i = 0; i < 4; ++i)
    {
      char note[16];
      snprintf(note, sizeof(note), "frz>%.0f%%", 100.0 * thresholds[i]);
      c.frozen = thresholds[i];
      run_one("F", note, &c, periodic, cuts);
    }
  }

  /* ── G: where is the GATHER crossover? ──────────────────────────────────
   *
   * `serial_gather_max_triangles` remeshes the whole ensemble on one rank when
   * it is small enough. The threshold is swept ACROSS a fixed mesh rather than
   * the mesh across a fixed threshold, because that is one variable instead of
   * two: 0 is the distributed control and anything above the mesh size is the
   * gather, so the pair of rows brackets the answer at this size.
   *
   * Two columns decide it and they can disagree. `wall` is the reason to gather.
   * `tri_out` and the verdict are the reason not to: the phase driver does about
   * 3.9x the remeshing work, and on an adequately resolved surface that extra
   * work is better relaxation rather than waste. Run it at several `-refine`
   * values before setting a default. */
  if (strchr(only, 'g'))
  {
    const int refines[] = {1, 2, 3};

    for (int i = 0; i < 3; ++i)
    {
      /* Domain-scaled for the same reason as section F: a body that fits inside
       * one rank box is already gathered, and comparing the gather against it
       * would compare a thing with itself. */
      sweep_config c = base;
      c.n_bodies = 1;
      c.r = 0.7;
      c.refine = refines[i];
      c.forced = MFTL_REMESH_PARALLEL_SPANNING;

      char note[16];
      snprintf(note, sizeof(note), "refine %d", refines[i]);

      c.gather = 0;
      run_one("G", note, &c, periodic, cuts);

      c.gather = 10000000;  // certainly above the mesh, so the trigger fires
      run_one("G", "  +gather", &c, periodic, cuts);
    }
  }

  /* ── L: what does the BALANCER buy? ─────────────────────────────────────
   *
   * Two fixtures, and the pair is the measurement. SPREAD is the lattice every
   * other regime uses: the geometric assignment is already near-even, so the
   * balancer should do nothing and the two rows should agree -- which is how
   * this section shows the balancer is not disturbing a case it should leave
   * alone. CLUSTERED packs every body into one rank's box, which is what a real
   * emulsion does locally, and is where the balancer either works or does not.
   *
   * The number to read is on the continuation line: `imbalance` and `ranks
   * idle`. `wall` is secondary here and will be noisy at these sizes -- what is
   * being measured is whether the WORK is spread, and at four performance cores
   * a machine that is oversubscribed anyway will not show it cleanly. */
  if (strchr(only, 'l'))
  {
    const int n = 4 * g_nproc;  // enough bodies that "no rank idle" is reachable

    sweep_config c = base;
    c.n_bodies = n;
    c.r = 0.10 * h_min;
    c.forced = MFTL_REMESH_PARALLEL_DISPERSED;

    c.cluster = MFTL_FALSE;
    c.slack = 0.0;
    run_one("L", "spread", &c, periodic, cuts);
    c.slack = 1.25;
    run_one("L", "  +balance", &c, periodic, cuts);

    c.cluster = MFTL_TRUE;
    c.slack = 0.0;
    run_one("L", "clustered", &c, periodic, cuts);
    c.slack = 1.25;
    run_one("L", "  +balance", &c, periodic, cuts);
  }

  if (g_rank == 0)
  {
    printf("\nHow to read it:\n");
    printf("  Compare the AUTO row against the two forced rows below it. AUTO is right when it\n");
    printf("  matches the cheaper one; a row where it does not is evidence against the 0.4 ratio.\n");
    printf("  `imbal` is max/mean owned triangles per rank: 1.0 is perfect, %.1f would mean one\n", (double) g_nproc);
    printf("  rank holds everything. In regime C it is the idleness that has no fix -- there is\n");
    printf("  no remeshing work to give a rank that holds no body.\n");
    printf("  A forced-dispersed row in regime D reports `spanning`: the admission check refused\n");
    printf("  and handed the ensemble on, which costs one measurement and is the correct answer.\n\n");
  }

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return 0;
}
