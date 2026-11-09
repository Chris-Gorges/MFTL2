/*
 * @Author: Christian Gorges
 * @Date: 2026-11-09
 *
 * @brief Droplets carried through a duct: hundreds of bodies of different sizes
 *        in a swirling flow, remeshed and redistributed over the ranks every step.
 *
 * A square duct of side H, periodic along its length L, with walls on the four
 * sides. N droplets of random radius in [rmin, rmax] are placed at random --
 * without overlap, a margin away from the walls, from a fixed seed so every
 * rank draws the same list -- and carried by a PRESCRIBED velocity field:
 *
 *   u =  U p(y) p(z),                    p(s) = 1 - (2 s / H - 1)^16
 *   v =  a U sin(pi y / H) cos(pi z / H)
 *   w = -a U cos(pi y / H) sin(pi z / H)
 *
 * The streamwise profile is flat in the core and falls to zero in a thin layer
 * at the walls, so a droplet near a wall lags and is sheared while one in the
 * core travels at U -- the flat profile of a turbulent duct rather than a laminar
 * one. The droplets start a margin away from the walls. A profile that sheared
 * the whole section (exponent 4) drew the outer ones into filaments far thinner
 * than any mesh of them, which is not what a droplet with surface tension does,
 * and cost a third of the volume over one pass; at exponent 8 the droplets the
 * swirl carries towards the walls still flattened into sheets.
 *
 * The cross-stream part is one vortex filling the section, as behind a swirler:
 * it turns the core almost rigidly, so the droplets travel down the duct on
 * helices. A vortex whose sense alternated along the duct was tried first; the
 * opposite turning of neighbouring stretches pulled the droplets into sheets.
 * The field is exactly divergence-free -- u depends on y and z only, and
 * dv/dy + dw/dz = 0 -- and has
 * no normal component on the walls, so every droplet keeps its volume and stays
 * inside the duct up to the time integration.
 *
 * There is no momentum equation and no surface tension: a sheared droplet is not
 * pulled back towards a sphere. What the run exercises is what a front-tracking
 * solver asks of the library every step -- move the owned vertices, synchronise
 * the ghosts, remesh hundreds of bodies with the dispersed strategy, and
 * redistribute them as they cross rank boundaries and the periodic seam.
 *
 * Usage:
 *   mpirun -n 8 droplets_in_duct_flow [--drops N] [--rmin R] [--rmax R] [--length L]
 *          [--refine N] [--steps N] [--time T] [--swirl a] [--seed S]
 *          [--vtk N] [--report N] [--verify]
 *
 *   --drops N    number of droplets                      (default 600)
 *                (the README animation: 800 on 8 ranks, --refine 3 --vtk 2)
 *   --rmin/rmax  radius range, in units of H             (default 0.03 .. 0.07)
 *   --length L   duct length, in units of H              (default 4)
 *   --refine N   icosphere level of the generated drops  (default 2)
 *   --steps N    timesteps                               (default 240)
 *   --time T     simulated time, in units of L/U         (default 1: one pass)
 *   --swirl a    cross-stream amplitude, in units of U   (default 0.15: stronger
 *                swirl turns the outer droplets into bananas, since the vortex
 *                turns its core faster than its rim)
 *   --margin m   distance of every droplet from the walls, in units of H (default 0.15)
 *   --sizing S   curvature | uniform                     (default curvature: a small
 *                droplet gets edges as short as its curvature asks for)
 *   --vtk N      write duct_NNNN.pvtu every N steps      (default 0)
 *   --verify     exit non-zero unless every droplet is still a closed sphere
 *                topologically and the total volume moved by less than 1 %
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The flow -- the program's own business, not the library's
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double H, L, U, swirl;
} duct_flow;

static double profile(double s, double H)
{
  const double q = 2.0 * s / H - 1.0, q2 = q * q, q4 = q2 * q2, q8 = q4 * q4;
  return 1.0 - q8 * q8;
}

static void velocity(const duct_flow *f, const double x[3], double t, double u[3])
{
  (void) t;  // steady
  const double A = f->swirl * f->U;
  const double sy = sin(MFTL_PI * x[1] / f->H), cy = cos(MFTL_PI * x[1] / f->H);
  const double sz = sin(MFTL_PI * x[2] / f->H), cz = cos(MFTL_PI * x[2] / f->H);
  u[0] = f->U * profile(x[1], f->H) * profile(x[2], f->H);
  u[1] = A * sy * cz;
  u[2] = -A * cy * sz;
}

/** @brief RK4 over the vertices this rank owns, wrapped along the duct and kept inside the walls. */
static void advect(MFTL_Ensemble *ens, const duct_flow *f, double t, double dt)
{
  const MFTL_Domain *dom = &ens->domain;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    MFTL_FOREACH_OWNED_VERTEX(obj, {
      double k1[3], k2[3], k3[3], k4[3], x[3];
      velocity(f, pos, t, k1);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + 0.5 * dt * k1[d];
      velocity(f, x, t + 0.5 * dt, k2);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + 0.5 * dt * k2[d];
      velocity(f, x, t + 0.5 * dt, k3);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + dt * k3[d];
      velocity(f, x, t + dt, k4);
      for (int d = 0; d < 3; ++d) pos[d] += (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
      MFTLApplyPeriodicity(dom, pos);
      for (int d = 1; d < 3; ++d) pos[d] = fmin(fmax(pos[d], dom->global_min[d]), dom->global_max[d]);
    });
    MFTLObjectMarkChanged(obj);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Placement: the same pseudo-random list on every rank
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief splitmix64: a portable generator, so every rank and platform draws the same droplets. */
static double uniform(uint64_t *state)
{
  uint64_t z = (*state += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  z ^= z >> 31;
  return (double) (z >> 11) * 0x1.0p-53;
}

/**
 * @brief Random non-overlapping droplets: radius uniform in [rmin, rmax], a gap
 *        of rmin between any two (minimum image along the periodic length) and
 *        @p margin from every wall. Rejection sampling; returns how many were placed.
 */
static int place_drops(int n, double rmin, double rmax, double L, double H, double margin, uint64_t seed, double *c, double *r)
{
  uint64_t s = seed;
  const double gap = rmin;
  int placed = 0;
  for (int attempt = 0; placed < n && attempt < 2000 * n; ++attempt)
  {
    const double ri = rmin + (rmax - rmin) * uniform(&s);
    const double lo = margin + ri, span = H - 2.0 * lo;
    const double ci[3] = {L * uniform(&s), lo + span * uniform(&s), lo + span * uniform(&s)};
    bool free_ = span > 0.0;
    for (int j = 0; j < placed && free_; ++j)
    {
      double dx = ci[0] - c[3 * j];
      dx -= L * nearbyint(dx / L);
      const double dy = ci[1] - c[3 * j + 1], dz = ci[2] - c[3 * j + 2];
      const double need = ri + r[j] + gap;
      free_ = dx * dx + dy * dy + dz * dz >= need * need;
    }
    if (!free_) continue;
    for (int d = 0; d < 3; ++d) c[3 * placed + d] = ci[d];
    r[placed++] = ri;
  }
  return placed;
}

/** @brief Cuts for nproc ranks: each prime factor goes to the axis whose rank box is widest. */
static void choose_cuts(int nproc, const double extent[3], int cuts[3])
{
  cuts[0] = cuts[1] = cuts[2] = 1;
  int factors[64], nf = 0, m = nproc;
  for (int p = 2; m > 1 && nf < 64; ++p)
    while (m % p == 0 && nf < 64)
    {
      factors[nf++] = p;
      m /= p;
    }
  for (int i = nf - 1; i >= 0; --i)
  {
    int best = 0;
    for (int k = 1; k < 3; ++k)
      if (extent[k] / cuts[k] > extent[best] / cuts[best]) best = k;
    cuts[best] *= factors[i];
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Measuring
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double volume;  ///< sum over droplets
  int n_bodies;  ///< droplets that exist
  int bad_euler;  ///< droplets whose V - E + F is not 2
  int unmeasurable;  ///< droplets without a defined volume
} duct_stats;

/** @brief Collective: total volume and the topology of every droplet, each counted once. */
static void measure(MFTL_Ensemble *ens, duct_stats *g)
{
  const int b = ens->n_local_objects;
  MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) malloc((size_t) (b > 0 ? b : 1) * sizeof(MFTL_BodyIntegrals));
  double *counts = (double *) malloc((size_t) (2 * b > 0 ? 2 * b : 1) * sizeof(double));
  if (!bi || !counts || MFTLEnsembleBodyIntegrals(ens, bi) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int o = 0; o < b; ++o)
  {
    counts[2 * o] = (double) ens->local_objects[o]->verts.n_local;
    counts[2 * o + 1] = (double) ens->local_objects[o]->tris.n_local;
  }
  const MFTL_BodyReduceOp sum[2] = {MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};
  MFTLEnsembleReduceBodies(ens, counts, 2, sum, MFTL_BODY_REDUCE_SCOPE_AUTO);

  double loc[4] = {0.0, 0.0, 0.0, 0.0};
  for (int o = 0; o < b; ++o)
  {
    if (bi[o].lowest_holder != ens->domain.rank) continue;
    loc[0] += bi[o].volume;
    loc[1] += 1.0;
    const double chi = counts[2 * o] - 0.5 * counts[2 * o + 1];  // V - E + F with E = 3F/2
    if (fabs(chi - 2.0) > 0.5) loc[2] += 1.0;
    if (bi[o].volume_status != MFTL_INTEGRAL_DEFINED || !bi[o].closed) loc[3] += 1.0;
  }
  MPI_Allreduce(MPI_IN_PLACE, loc, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  g->volume = loc[0];
  g->n_bodies = (int) loc[1];
  g->bad_euler = (int) loc[2];
  g->unmeasurable = (int) loc[3];
  free(counts);
  free(bi);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The run
 * ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int n_drops = 600, refine = 2, steps = 240, vtk_every = 0, report = 40;
  double rmin = 0.03, rmax = 0.07, length = 4.0, time_passes = 1.0, swirl = 0.15, margin = 0.15;
  bool curvature = MFTL_TRUE;
  uint64_t seed = 20261109;
  bool verify = MFTL_FALSE;
  for (int i = 1; i < argc; ++i)
  {
    const bool has = i + 1 < argc;
    if (!strcmp(argv[i], "--drops") && has)
      n_drops = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--rmin") && has)
      rmin = atof(argv[++i]);
    else if (!strcmp(argv[i], "--rmax") && has)
      rmax = atof(argv[++i]);
    else if (!strcmp(argv[i], "--length") && has)
      length = atof(argv[++i]);
    else if (!strcmp(argv[i], "--refine") && has)
      refine = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && has)
      steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--time") && has)
      time_passes = atof(argv[++i]);
    else if (!strcmp(argv[i], "--swirl") && has)
      swirl = atof(argv[++i]);
    else if (!strcmp(argv[i], "--margin") && has)
      margin = atof(argv[++i]);
    else if (!strcmp(argv[i], "--sizing") && has)
      curvature = strcmp(argv[++i], "uniform") != 0;
    else if (!strcmp(argv[i], "--seed") && has)
      seed = (uint64_t) strtoull(argv[++i], MFTL_NULL, 10);
    else if (!strcmp(argv[i], "--vtk") && has)
      vtk_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--report") && has)
      report = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--verify"))
      verify = MFTL_TRUE;
    else
    {
      if (rank == 0) fprintf(stderr, "unknown or incomplete option '%s'; see the header of droplets_in_duct_flow.c\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  const duct_flow flow = {1.0, length, 1.0, swirl};
  double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {length, flow.H, flow.H};
  bool periodic[3] = {true, false, false};
  const double extent[3] = {length, flow.H, flow.H};
  int cuts[3];
  choose_cuts(nproc, extent, cuts);

  /* The droplets: the same list on every rank, generated on rank 0 and scattered. */
  double *centre = (double *) malloc(3 * (size_t) n_drops * sizeof(double));
  double *radius = (double *) malloc((size_t) n_drops * sizeof(double));
  if (!centre || !radius) MPI_Abort(MPI_COMM_WORLD, 1);
  const int placed = place_drops(n_drops, rmin, rmax, length, flow.H, margin, seed, centre, radius);

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, placed);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int i = 0; i < placed; ++i)
  {
    MFTL_Object *obj = MFTLObjectCreate(i);
    MFTLEnsembleAddObject(ens, obj);
    if (MFTLGenerateSphereLocal(obj, &centre[3 * i], radius[i], refine) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MFTLPublishGeneratedBodies(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);  // every MFTL_Object* from here on is invalid; the handle is the body id

  const double target = MFTLEnsembleMeanEdgeLength(ens);
  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.projection = MFTL_REMESH_PROJECT_PHONG;
  if (curvature) MFTLRemeshUseCurvatureSizing(&params, 0.02 * target, 0.25 * target, target);

  duct_stats g0;
  measure(ens, &g0);
  MFTLEnsembleUpdateGlobalCounts(ens);
  if (rank == 0)
  {
    printf("=== droplets in a duct: %d rank(s), grid %dx%dx%d ===\n", nproc, cuts[0], cuts[1], cuts[2]);
    printf("  duct %.2f x %.2f x %.2f, periodic along x; %d droplets of radius %.3f .. %.3f (%d requested)\n", length, flow.H, flow.H, placed, rmin, rmax,
           n_drops);
    printf("  %lld triangles, target edge %.5f (%s sizing), swirl %.2f U, %d steps over %.2f L/U\n", (long long) ens->total_global_triangles, target,
           curvature ? "curvature" : "uniform", swirl, steps, time_passes);
    printf("  %6s %8s %10s %10s %9s\n", "step", "t", "triangles", "V/V0", "ms/step");
  }

  const double T = time_passes * length / flow.U, dt = T / (double) steps;
  char name[64];
  if (vtk_every > 0)
  {
    snprintf(name, sizeof name, "duct_%04d", 0);
    MFTLWriteVtu(ens, name, MFTL_NULL);
  }

  double step_max = 0.0, step_sum = 0.0;
  for (int step = 1; step <= steps; ++step)
  {
    const double t = (step - 1) * dt;
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    advect(ens, &flow, t, dt);
    MFTLUpdateGhostPositions(ens);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    MFTL_RemeshStats rs;
    if (MFTLRemeshEnsemble(ens, &params, &rs) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);

    double dt_step = MPI_Wtime() - t0;
    MPI_Allreduce(MPI_IN_PLACE, &dt_step, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    step_sum += dt_step;
    step_max = fmax(step_max, dt_step);

    if (vtk_every > 0 && step % vtk_every == 0)
    {
      snprintf(name, sizeof name, "duct_%04d", step);
      MFTLWriteVtu(ens, name, MFTL_NULL);
    }
    if ((report > 0 && step % report == 0) || step == steps)
    {
      duct_stats g;
      measure(ens, &g);
      MFTLEnsembleUpdateGlobalCounts(ens);
      if (rank == 0) printf("  %6d %8.3f %10lld %10.6f %9.2f\n", step, t + dt, (long long) ens->total_global_triangles, g.volume / g0.volume, 1e3 * dt_step);
    }
  }

  duct_stats g;
  measure(ens, &g);
  const double dv = g.volume / g0.volume - 1.0;
  const bool ok = g.n_bodies == placed && g.bad_euler == 0 && g.unmeasurable == 0 && fabs(dv) < 0.01;
  if (rank == 0)
  {
    printf("  droplets %d of %d, every one a closed sphere: %s; volume %+.3f %%; %.2f ms per step (max %.2f)\n", g.n_bodies, placed,
           (g.bad_euler == 0 && g.unmeasurable == 0) ? "yes" : "NO", 1e2 * dv, 1e3 * step_sum / steps, 1e3 * step_max);
    if (verify) printf("  verify: %s\n", ok ? "ok" : "FAILED");
  }

  free(centre);
  free(radius);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (verify && !ok) ? EXIT_FAILURE : EXIT_SUCCESS;
}
