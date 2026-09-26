/*
 * @Author: Christian Gorges
 * @Date: 2026-09-24
 *
 * @brief Steep, near-breaking deep-water waves travelling through a periodic
 *        boundary on an open surface, remeshed in parallel as they go.
 *
 * A water surface spanning the whole box, carrying waves that march from left to
 * right, leave through the right-hand face and re-enter on the left. The mesh is
 * open -- it has genuine free edges along the two channel walls -- but it is
 * PERIODICALLY STITCHED in the direction of travel, which is what makes the
 * crossing seamless rather than merely repetitive.
 *
 * The wave is driven close to breaking, and that is what makes remeshing
 * necessary rather than cosmetic. Surface vertices are material points, and in a
 * steep wave the water at the crest nearly keeps pace with the crest: vertices
 * bunch up under every crest and spread out in the trough behind it, cyclically,
 * as the crest passes through them. At the default steepness the spacing changes
 * by a factor of three each way -- far outside the remesher's [4/5, 4/3] band --
 * where the MFTL_5 default (ka = 0.15) moved it by a few percent and never
 * needed a remesher at all.
 *
 * ── Why the mesh has to be stitched, not just the field ────────────────────
 *
 * A plain rectangular patch spanning the domain would be a disk with four rims
 * and a velocity field that happens to be periodic: the wave pattern would
 * reappear on the left, but the SURFACE would not -- there would be a free edge
 * at x_min and another at x_max, two rows of vertices on top of each other with
 * no triangles between them, and a remesher working near that seam would have no
 * neighbourhood to work with.
 *
 * So the last column of vertices is not duplicated: cell i connects column i to
 * column (i+1) % nx, and the seam is an ordinary interior edge like any other.
 * The result is topologically an ANNULUS (a cylinder):
 *
 *     vertices        nx * (ny + 1)
 *     triangles       2 * nx * ny
 *     boundary edges  2 * nx            <- TWO loops, one per channel wall
 *     V - E + F       0                 <- annulus, not 1 (disk) and not 2 (sphere)
 *
 * The Euler characteristic is checked at every report, over the DISTRIBUTED and
 * REMESHED mesh: it stays 0 only if the seam is still a pair of linked twins
 * after the remesher has split, collapsed and flipped across it and the
 * redistribution has torn it across ranks.
 *
 * ── The velocity field ─────────────────────────────────────────────────────
 *
 * Deep-water (Airy) kinematics. Per component, with wave vector (k_x, k_y),
 * k = |k|, phase theta = k_x*x + k_y*y - omega*t and omega = sqrt(g*k):
 *
 *     u = A*omega * (k_x/k) * e^{k(z-z_ref)} * cos(theta)
 *     v = A*omega * (k_y/k) * e^{k(z-z_ref)} * cos(theta)
 *     w = A*omega *           e^{k(z-z_ref)} * sin(theta)
 *
 * The depth decay e^{k(z-z_ref)} is what makes it divergence-free, so the
 * surface is stretched, never inflated, and the volume BELOW it is an invariant
 * (the V/V0 column). Every wavelength divides the box exactly (integer mode
 * numbers), which is what makes the field L_x-periodic and the crossing
 * seamless.
 *
 * ── Why the surface starts on a STREAMLINE, not on a cosine ────────────────
 *
 * The MFTL_5 version started the mesh on the linear surface z = A*cos(theta).
 * That is not a state this field keeps: e^{k(z-z_ref)} makes crests travel faster
 * than troughs, so the crest grows and sharpens -- mildly at ka = 0.15, and
 * catastrophically near breaking. Measured on the MFTL_5 two-component sea with
 * the linear start: at an initial crest speed of 0.58 c the surface had
 * overturned after ONE time unit, and at 0.81 c it ran away to the box walls
 * within two.
 *
 * In the frame moving with the primary wave the field is STEADY, with stream
 * function (in units of zeta = k(z - z_ref), theta = k*x - omega*t)
 *
 *     psi = zeta - e^{zeta} * cos(theta)
 *
 * and a surface that lies on a streamline psi = C stays on it forever: the wave
 * then travels with its shape exactly preserved, which is the kinematic
 * counterpart of a steady Stokes wave. So the primary wave's surface is that
 * streamline, found by bisection per vertex at t = 0, with z_ref placed so the
 * mean water level sits mid-box. The family has ONE shape parameter -- the
 * crest height, equivalently the ratio of the crest particle speed to the phase
 * speed, u_crest/c = e^{zeta_crest}:
 *
 *     ka = kH/2      H/lambda   u_crest/c   steepest face   crest spacing
 *     0.37           0.118       0.50       21.8 deg        2.5x compressed
 *     0.42 default   0.134       0.59       25.0 deg        3.0x
 *     0.45           0.143       0.64       27.0 deg        3.5x
 *     0.54           0.172       0.81       33.6 deg        6.6x
 *     0.60           0.191       0.92       38.9 deg        16x
 *
 * Stokes' limiting wave -- the steepest periodic wave that exists -- has
 * ka = pi*H/lambda = 0.443, and the default 0.42 is 95 % of it: "shortly before
 * breaking" in the classical sense. This kinematic family goes further, to a
 * 90-degree crest at u_crest/c = 1 (Stokes' has 120 degrees), which is the
 * kinematic breaking criterion: water at the crest overtaking the crest. Past it
 * the steady family ends; --steepness is capped at 0.6 accordingly.
 *
 * The "crest spacing" column is what the remesher meets. Vertices move along the
 * surface, in the wave frame, at speed c - u: slowest at the crest (c(1 - r)),
 * fastest in the trough. Two vertices a fixed time apart are therefore closest
 * under the crest and farthest apart in the trough, by that ratio each way --
 * edges shrink under the crest and stretch behind it, EVERY period, forever. A
 * remesher that collapses on the way in and splits on the way out will THRASH;
 * the `coll` and `split` columns measure how much. Measured at the defaults over
 * 400 passes: 70 to 120 operations per pass at np = 1, 2 and 4, never settling,
 * while the wave height held to within 1 % and the volume below it to 6e-4.
 *
 * A single-valued kinematic field is a DIFFEOMORPHISM, so the surface can never
 * intersect itself; overturned triangles (n_z <= 0) are counted anyway, and
 * --verify fails on them.
 *
 * --components N adds N-1 shorter, dispersive RIPPLES (omega = sqrt(g*k), so
 * they drift through the primary), each of steepness --ripple, started on their
 * linear surfaces. The sea is then no longer steady: the mesh sees a deformation
 * that never repeats instead of one travelling pattern. A short wave riding on
 * the crest of a steep long one is compressed and amplified there -- a real
 * effect, and a real route to breaking -- so ripples and a near-breaking carrier
 * do not mix. Measured: kA = 0.03 ripples on the default ka = 0.42 carrier broke
 * it by t = 4 with one ripple and by t = 1 with two (one of them oblique), and
 * MFTL_5's relative amplitudes 1/(j+1)^2 broke even a ka = 0.25 carrier at
 * t = 0.3. With ripples the carrier therefore defaults to ka = 0.30, where two
 * and three components (oblique included) ran 800 remeshed steps soundly with
 * faces up to 38 degrees.
 *
 * ── The two rims ───────────────────────────────────────────────────────────
 *
 * Stitching x leaves exactly two boundary loops, one along each channel wall at
 * y_min and y_max, so every rim vertex is a wall vertex and MFTL_FLAG_BOUNDARY
 * alone is the constraint set -- which is also why the constraint survives
 * remeshing: a split of a rim edge creates a rim vertex, and the operators hand
 * it MFTL_FLAG_BOUNDARY. The rims slide: the y velocity is projected out and the
 * y coordinate snapped. For the default sea this is a no-op (k_y = 0); it earns
 * its place under --oblique.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P travelling_waves_open_surface [cx cy cz] [options]
 *
 *     cx cy cz          process grid, cx*cy*cz == P. Defaults to (P,1,1).
 *
 *     --nx N            cells along the periodic direction  (default 128)
 *     --ny N            cells across the channel            (default 32)
 *     --steps N         timesteps                           (default 800)
 *     --dt X            timestep                            (default 0.01)
 *     --mode M          wavelengths of the primary wave across the box (default 2)
 *     --steepness X     ka = k*H/2 of the primary, (0, 0.6] (default 0.42, or
 *                       0.30 with ripples; Stokes' limiting wave has 0.443)
 *     --components N    1 steady wave, 2..3 adds dispersive ripples (default 1)
 *     --ripple X        steepness k*A of each ripple         (default 0.03)
 *     --speed X         phase speed of the primary wave     (default 1.0)
 *     --oblique         give the last ripple a cross-channel wavenumber
 *
 *     --remesh-every N  remesh every N steps, 0 = never     (default 2)
 *     --target X        target edge length                  (default: initial mean)
 *     --iterations N    remeshing iterations per pass       (default 3)
 *     --smooth N        smoothing sweeps per iteration      (default 3)
 *     --budget N        cut-plane placements per pass       (default 4)
 *     --strategy S      auto | spanning | dispersed         (default auto)
 *
 *     --vtk N           write a .pvtu every N steps         (default 25, 0 off)
 *     --report N        quality line every N steps          (default 50)
 *     --out PREFIX      output file prefix                  (default "waves")
 *     --verify          exit non-zero unless halo, seam (chi = 0), rims, volume
 *                       and orientation hold at every report
 *     --require-rim-cut --verify, and fail if at np > 1 no rim triangle ever met
 *                       a partition boundary (what the CTest passes)
 *
 *     --topology        let the surface change topology: a breaking crest whose
 *                       lip comes within --delta-coalescence of the face it
 *                       overturns onto reconnects with it, trapping air, and a
 *                       sheet thinner than --delta-breakup tears
 *     --delta-breakup X     (default 1.5 * target)
 *     --delta-coalescence X (default 1.0 * target)
 *     --topo-dx X       reconstruction lattice spacing (default target)
 *     --topo-every N    detect every N steps (default --remesh-every)
 *     --require-topology --verify, and fail unless the body count or the genus
 *                       actually changed (the --topology CTest's non-vacuity)
 *
 * ── Breaking, with topology changes ────────────────────────────────────────
 *
 * Without --topology an overturned triangle FAILS --verify: a single-valued
 * kinematic field cannot fold the surface through itself, but it can bring the
 * plunging lip arbitrarily close to the face below it, which no mesh resolves.
 * With --topology that is exactly what gets detected and reconnected, so an
 * overturned triangle is a reported quantity rather than a failure, and the
 * annulus check becomes a per-body one: every OPEN body has V-E+F <= 1 and
 * every CLOSED one an even V-E+F <= 2. The volume below the surface is
 * conserved by the reconstruction itself -- it measures the same flux of
 * (z - z_bottom) e_z this example does -- and what is left is booked per call.
 * A breaking sea: --components 2 --steepness 0.42 --ripple 0.03.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

#define WAVE_MAX_COMPONENTS 3

/** @brief ka of Stokes' limiting (steepest) wave, pi * 0.1412. */
#define WAVE_STOKES_LIMIT_KA 0.4436

/**
 * @brief Ceiling on the depth-decay exponent k*(z - z_ref). Runaway guard only.
 *
 * e^{k(z-z_ref)} feeds the surface elevation back into its own velocity, which
 * is a finite-time blowup for a configuration that overturns. Past it positions
 * reach inf, then NaN. Capping bounds the growth to linear so the height check
 * at the next report catches it. On the steady primary surface the exponent is
 * at most ln(u_crest/c) < 0, so the cap never binds there.
 */
#define WAVE_DECAY_EXP_CAP 1.5

/** @brief How close to a channel wall a rim vertex must sit, relative to its width. */
#define WAVE_WALL_TOL 1e-9

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int cuts[3];
  int nx, ny;
  int steps;
  double dt;
  int mode;
  int n_components;
  double steepness;
  double ripple;
  bool steepness_given;
  double speed;
  bool oblique;

  int remesh_every;
  double target;
  int iterations;
  int smooth;
  int budget;
  MFTL_RemeshParallelStrategy strategy;

  int vtk_every;
  int report_every;
  char out[128];
  bool verify;
  bool require_rim_cut;

  bool topology;
  double delta_breakup;
  double delta_coalescence;
  double topo_dx;
  int topo_every;
  bool require_topology;
} config;

static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  cfg->cuts[0] = nproc;
  cfg->cuts[1] = 1;
  cfg->cuts[2] = 1;
  cfg->nx = 128;
  cfg->ny = 32;
  cfg->steps = 800;
  cfg->dt = 0.01;
  cfg->mode = 2;
  cfg->n_components = 1;
  cfg->steepness = 0.42;
  cfg->steepness_given = MFTL_FALSE;
  cfg->ripple = 0.03;
  cfg->speed = 1.0;
  cfg->oblique = MFTL_FALSE;
  cfg->remesh_every = 2;
  cfg->target = 0.0;
  cfg->iterations = 3;
  cfg->smooth = 3;
  cfg->budget = 4;
  cfg->strategy = MFTL_REMESH_PARALLEL_AUTO;
  cfg->vtk_every = 25;
  cfg->report_every = 50;
  cfg->verify = MFTL_FALSE;
  cfg->require_rim_cut = MFTL_FALSE;
  cfg->topology = MFTL_FALSE;
  cfg->delta_breakup = 0.0;
  cfg->delta_coalescence = 0.0;
  cfg->topo_dx = 0.0;
  cfg->topo_every = 0;
  cfg->require_topology = MFTL_FALSE;
  snprintf(cfg->out, sizeof cfg->out, "waves");

  int i = 1;
  if (argc >= 4 && argv[1][0] != '-')
  {
    for (int d = 0; d < 3; ++d) cfg->cuts[d] = atoi(argv[1 + d]);
    i = 4;
  }

  for (; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--oblique"))
      cfg->oblique = MFTL_TRUE;
    else if (!strcmp(argv[i], "--verify"))
      cfg->verify = MFTL_TRUE;
    else if (!strcmp(argv[i], "--topology"))
      cfg->topology = MFTL_TRUE;
    else if (!strcmp(argv[i], "--require-topology"))
      cfg->verify = cfg->require_topology = cfg->topology = MFTL_TRUE;
    else if (!strcmp(argv[i], "--delta-breakup") && i + 1 < argc)
      cfg->delta_breakup = atof(argv[++i]);
    else if (!strcmp(argv[i], "--delta-coalescence") && i + 1 < argc)
      cfg->delta_coalescence = atof(argv[++i]);
    else if (!strcmp(argv[i], "--topo-dx") && i + 1 < argc)
      cfg->topo_dx = atof(argv[++i]);
    else if (!strcmp(argv[i], "--topo-every") && i + 1 < argc)
      cfg->topo_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--require-rim-cut"))
      cfg->verify = cfg->require_rim_cut = MFTL_TRUE;
    else if (!strcmp(argv[i], "--nx") && i + 1 < argc)
      cfg->nx = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--ny") && i + 1 < argc)
      cfg->ny = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "--mode") && i + 1 < argc)
      cfg->mode = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--components") && i + 1 < argc)
      cfg->n_components = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steepness") && i + 1 < argc)
    {
      cfg->steepness = atof(argv[++i]);
      cfg->steepness_given = MFTL_TRUE;
    }
    else if (!strcmp(argv[i], "--ripple") && i + 1 < argc)
      cfg->ripple = atof(argv[++i]);
    else if (!strcmp(argv[i], "--speed") && i + 1 < argc)
      cfg->speed = atof(argv[++i]);
    else if (!strcmp(argv[i], "--remesh-every") && i + 1 < argc)
      cfg->remesh_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--target") && i + 1 < argc)
      cfg->target = atof(argv[++i]);
    else if (!strcmp(argv[i], "--iterations") && i + 1 < argc)
      cfg->iterations = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--smooth") && i + 1 < argc)
      cfg->smooth = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--budget") && i + 1 < argc)
      cfg->budget = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--vtk") && i + 1 < argc)
      cfg->vtk_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--report") && i + 1 < argc)
      cfg->report_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--out") && i + 1 < argc)
      snprintf(cfg->out, sizeof cfg->out, "%s", argv[++i]);
    else if (!strcmp(argv[i], "--strategy") && i + 1 < argc)
    {
      ++i;
      if (!strcmp(argv[i], "auto"))
        cfg->strategy = MFTL_REMESH_PARALLEL_AUTO;
      else if (!strcmp(argv[i], "spanning"))
        cfg->strategy = MFTL_REMESH_PARALLEL_SPANNING;
      else if (!strcmp(argv[i], "dispersed"))
        cfg->strategy = MFTL_REMESH_PARALLEL_DISPERSED;
      else
      {
        if (rank == 0) fprintf(stderr, "--strategy must be auto, spanning or dispersed (got '%s')\n", argv[i]);
        return MFTL_ERROR;
      }
    }
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
  /* nx >= 3 so that wrapping a cell's two corners cannot produce a triangle with
   * a repeated vertex; integer mode >= 1 so the field is exactly box-periodic,
   * which is the whole premise of the seam. */
  if (cfg->nx < 3 || cfg->ny < 1 || cfg->steps < 0 || cfg->mode < 1 || cfg->remesh_every < 0 || cfg->iterations < 1 || cfg->smooth < 0)
  {
    if (rank == 0) fprintf(stderr, "Need nx >= 3, ny >= 1, steps >= 0, mode >= 1, remesh-every >= 0, iterations >= 1, smooth >= 0.\n");
    return MFTL_ERROR;
  }
  /* Ripples break a near-breaking carrier (see the file header), so a rippled
   * sea gets the carrier it was measured to survive unless one was asked for. */
  if (cfg->n_components > 1 && !cfg->steepness_given) cfg->steepness = 0.30;

  if (cfg->n_components < 1 || cfg->n_components > WAVE_MAX_COMPONENTS || cfg->ripple < 0.0)
  {
    if (rank == 0) fprintf(stderr, "--components must lie in 1..%d and --ripple must not be negative.\n", WAVE_MAX_COMPONENTS);
    return MFTL_ERROR;
  }
  if (cfg->steepness <= 0.0 || cfg->steepness > 0.6)
  {
    /* Past ~0.61 there is no steady wave left in this kinematic family: the
     * crest particle would overtake the crest (u_crest/c = 1). */
    if (rank == 0) fprintf(stderr, "--steepness must lie in (0, 0.6]; Stokes' limiting wave has ka = %.3f.\n", WAVE_STOKES_LIMIT_KA);
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The wave field
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double kx, ky;  ///< wave vector
  double k;  ///< its magnitude
  double amp;  ///< orbital amplitude A, measured at z_ref
  double omega;  ///< angular frequency, sqrt(g*k)
  double phase;  ///< phase offset at t = 0
} wave_component;

typedef struct
{
  wave_component comp[WAVE_MAX_COMPONENTS];
  int n;
  double origin[2];  ///< phases are measured from here
  double z_ref;  ///< reference level of the depth decay e^{k(z - z_ref)}
  double gravity;
  double c;  ///< phase speed of the primary

  /* The primary's steady surface: the streamline zeta - e^zeta cos(theta) = C. */
  double psi;  ///< C = ln(r) - r
  double crest_ratio;  ///< r = u_crest / c = e^{zeta_crest}
  double zeta_mean;  ///< mean level of the streamline, in units of zeta
  double height;  ///< crest-to-trough height H of the primary
  double max_slope_deg;  ///< steepest face of the steady primary surface
} wave_field;

/**
 * @brief zeta on the primary's surface streamline at phase theta.
 *
 * Solves zeta - e^zeta cos(theta) = C on the branch BELOW the separatrix, i.e.
 * where d/dzeta = 1 - e^zeta cos(theta) > 0. Bisection rather than Newton: it
 * runs once per vertex at generation time, and on the concave branch (cos > 0)
 * Newton converges only from one side.
 *
 * The bracket: the function is -1 - e^{C-1}cos(theta) < 0 at C - 1 in every
 * case, and positive at -ln(cos theta) (its maximum, where the separatrix would
 * be) for cos > 0, or at C + 1 for cos <= 0.
 */
static double streamline_zeta(double theta, double C)
{
  const double c = cos(theta);
  double lo = C - 1.0;
  double hi = (c > 0.0) ? -log(c) : C + 1.0;

  for (int it = 0; it < 200 && hi - lo > 1e-15; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    if (mid - exp(mid) * c - C < 0.0)
      lo = mid;
    else
      hi = mid;
  }
  return 0.5 * (lo + hi);
}

/** @brief kH/2 of the steady wave with crest ratio @p r. Monotone in r. */
static double steady_ka(double r)
{
  const double C = log(r) - r;
  return 0.5 * (log(r) - streamline_zeta(MFTL_PI, C));
}

static void wave_field_init(wave_field *f, const MFTL_Domain *dom, const config *cfg)
{
  const double lx = dom->domain_size[0];
  const double ly = dom->domain_size[1];
  const double z_mid = 0.5 * (dom->global_min[2] + dom->global_max[2]);

  f->origin[0] = dom->global_min[0];
  f->origin[1] = dom->global_min[1];
  f->n = cfg->n_components;

  /* Gravity is derived rather than given, so that --speed means what it says:
   * the primary travels at exactly that phase speed, and the others follow from
   * the deep-water dispersion relation the same g implies. */
  const double k1 = MFTL_TWO_PI * (double) cfg->mode / lx;
  f->gravity = cfg->speed * cfg->speed * k1;
  f->c = cfg->speed;

  /* ── The primary: the crest ratio r whose steady wave has ka = --steepness.
   * steady_ka is monotone in r, so bisection finds it. */
  double lo = 1e-6, hi = 1.0 - 1e-12;
  for (int it = 0; it < 200; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    if (steady_ka(mid) < cfg->steepness)
      lo = mid;
    else
      hi = mid;
  }
  f->crest_ratio = lo;
  f->psi = log(lo) - lo;

  /* Mean level and steepest face of the streamline, sampled finely enough that
   * a vertex never falls between two samples of a different shape. */
  const int n_samples = 8192;
  double sum = 0.0, max_dz = 0.0, prev = streamline_zeta(-MFTL_TWO_PI / (double) n_samples, f->psi);
  double zeta_min = MFTL_DBL_MAX, zeta_max = -MFTL_DBL_MAX;
  for (int i = 0; i < n_samples; ++i)
  {
    const double zeta = streamline_zeta(MFTL_TWO_PI * (double) i / (double) n_samples, f->psi);
    sum += zeta;
    zeta_min = MFTL_MIN(zeta_min, zeta);
    zeta_max = MFTL_MAX(zeta_max, zeta);
    max_dz = MFTL_MAX(max_dz, fabs(zeta - prev) * (double) n_samples / MFTL_TWO_PI);
    prev = zeta;
  }
  f->zeta_mean = sum / (double) n_samples;
  f->height = (zeta_max - zeta_min) / k1;
  f->max_slope_deg = atan(max_dz) * 180.0 / MFTL_PI;

  /* The mean water level sits mid-box. */
  f->z_ref = z_mid - f->zeta_mean / k1;

  /* In these units u = c * e^{zeta} cos(theta), i.e. A*omega = c at z_ref. */
  f->comp[0].kx = k1;
  f->comp[0].ky = 0.0;
  f->comp[0].k = k1;
  f->comp[0].omega = sqrt(f->gravity * k1);
  f->comp[0].amp = 1.0 / k1;
  f->comp[0].phase = 0.0;

  /* ── The dispersive ripples: steepness --ripple each, measured at the mean
   * water level. */
  for (int j = 1; j < f->n; ++j)
  {
    wave_component *c = &f->comp[j];

    c->kx = MFTL_TWO_PI * (double) (cfg->mode * (j + 1)) / lx;
    c->ky = (cfg->oblique && j == f->n - 1) ? MFTL_TWO_PI / ly : 0.0;
    c->k = sqrt(c->kx * c->kx + c->ky * c->ky);
    c->omega = sqrt(f->gravity * c->k);

    c->amp = (cfg->ripple / c->k) * exp(-c->k * (z_mid - f->z_ref));

    /* Irrational-ish offsets, so the components crest together only now and
     * then rather than once, obviously, at t = 0. */
    c->phase = 0.7 * (double) j;
  }
}

/**
 * @brief Surface elevation at t = 0: the primary's streamline plus the linear
 *        elevation of every extra component.
 */
static double initial_elevation(const wave_field *f, double x, double y)
{
  const double theta = f->comp[0].kx * (x - f->origin[0]) + f->comp[0].phase;
  double z = f->z_ref + streamline_zeta(theta, f->psi) / f->comp[0].k;

  for (int j = 1; j < f->n; ++j)
  {
    const wave_component *c = &f->comp[j];
    const double z_mid = f->z_ref + f->zeta_mean / f->comp[0].k;
    z += c->amp * exp(c->k * (z_mid - f->z_ref)) * cos(c->kx * (x - f->origin[0]) + c->ky * (y - f->origin[1]) + c->phase);
  }
  return z;
}

/**
 * @brief Orbital velocity of the superposed wave train at a point.
 *
 * Analytic and purely local: the advection step needs no neighbour data, so the
 * message profile is exactly the library's own.
 */
static void wave_velocity(const wave_field *f, double t, const double p[3], double out[3])
{
  out[0] = out[1] = out[2] = 0.0;

  const double dz = p[2] - f->z_ref;

  for (int j = 0; j < f->n; ++j)
  {
    const wave_component *c = &f->comp[j];
    const double theta = c->kx * (p[0] - f->origin[0]) + c->ky * (p[1] - f->origin[1]) - c->omega * t + c->phase;
    const double a = c->amp * c->omega * exp(MFTL_MIN(c->k * dz, WAVE_DECAY_EXP_CAP));

    out[0] += a * (c->kx / c->k) * cos(theta);
    out[1] += a * (c->ky / c->k) * cos(theta);
    out[2] += a * sin(theta);
  }
}

/** @brief Where the primary's crest sits right now, wrapped into the box. */
static double crest_position(const wave_field *f, const MFTL_Domain *dom, double t)
{
  const wave_component *c = &f->comp[0];
  const double lx = dom->domain_size[0];

  /* theta = 0 at a crest, so k*(x - x0) = omega*t - phase. */
  double x = f->origin[0] + (c->omega * t - c->phase) / c->kx;

  x = fmod(x - dom->global_min[0], lx);
  if (x < 0.0) x += lx;
  return dom->global_min[0] + x;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The periodically stitched patch
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Builds an open surface that is periodic in x and free at both y edges.
 *
 * Column nx-1 wraps to column 0 instead of a duplicate column existing there, so
 * the seam is an ordinary interior edge and MFTLRebuildTwins links it like any
 * other. Winding (v00,v10,v11) and (v00,v11,v01) with x along u and y along v,
 * so the surface normal points up, out of the water.
 */
static int build_periodic_patch(MFTL_Object *obj, const MFTL_Domain *dom, const wave_field *f, int nx, int ny)
{
  const int n_verts = nx * (ny + 1);
  const int n_tris = 2 * nx * ny;

  double *pos = (double *) malloc(3 * (size_t) n_verts * sizeof(double));
  int *tris = (int *) malloc(3 * (size_t) n_tris * sizeof(int));
  if (!pos || !tris)
  {
    free(pos);
    free(tris);
    return MFTL_ERROR;
  }

  /* nx intervals across nx columns, not nx-1: the last interval is the seam and
   * closes back onto column 0. Getting this wrong leaves a one-cell gap that no
   * topological check would catch -- the mesh is valid, just the wrong shape. */
  const double dx = dom->domain_size[0] / (double) nx;
  const double dy = dom->domain_size[1] / (double) ny;

  for (int j = 0; j <= ny; ++j)
  {
    /* The last row is written as the bound itself: the rim must lie on the
     * channel wall to the bit. */
    const double y = (j == ny) ? dom->global_max[1] : dom->global_min[1] + (double) j * dy;
    for (int i = 0; i < nx; ++i)
    {
      const double x = dom->global_min[0] + (double) i * dx;
      const int v = j * nx + i;

      pos[3 * v + 0] = x;
      pos[3 * v + 1] = y;
      pos[3 * v + 2] = initial_elevation(f, x, y);
    }
  }

  int w = 0;
  for (int j = 0; j < ny; ++j)
  {
    for (int i = 0; i < nx; ++i)
    {
      const int i2 = (i + 1) % nx;  // the wrap: this is the whole trick
      const int v00 = j * nx + i;
      const int v10 = j * nx + i2;
      const int v01 = (j + 1) * nx + i;
      const int v11 = (j + 1) * nx + i2;

      tris[w++] = v00;
      tris[w++] = v10;
      tris[w++] = v11;

      tris[w++] = v00;
      tris[w++] = v11;
      tris[w++] = v01;
    }
  }

  const int rc = MFTLBuildIndexedMesh(obj, pos, n_verts, tris, n_tris);
  free(pos);
  free(tris);
  return rc;
}

/**
 * @brief Walks the boundary loops of the COMPLETE mesh. Rank 0, before scattering.
 *
 * A distributed mesh cannot answer this: MFTLHeNextBoundary rotates around the
 * shared vertex until the fan runs out, and on a partition the fan also runs out
 * at the halo rim. The loop STRUCTURE is a property of the whole surface, so it
 * is measured while the whole surface is in one place.
 */
static int count_boundary_loops(const MFTL_Object *obj, int *shortest, int *longest)
{
  *shortest = 0;
  *longest = 0;

  const int n_he = MFTLNumHalfEdges(obj);
  char *seen = (char *) calloc((size_t) MFTL_MAX(n_he, 1), 1);
  if (!seen) return 0;

  int loops = 0;
  for (int h0 = 0; h0 < n_he; ++h0)
  {
    if (!MFTLHeIsMeshBoundary(obj, h0) || seen[h0]) continue;

    int len = 0;
    int h = h0;
    do
    {
      seen[h] = 1;
      len++;
      h = MFTLHeNextBoundary(obj, h);
    } while (h != MFTL_ERROR && h != h0);

    loops++;
    if (*shortest == 0 || len < *shortest) *shortest = len;
    if (len > *longest) *longest = len;
  }

  free(seen);
  return loops;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Advection and the rim constraint
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Keeps a vertex inside the box: wrap in x, slide the rim along y.
 *
 * Every rim vertex is a channel-wall vertex (stitching x left only the two y
 * rims), so MFTL_FLAG_BOUNDARY is itself the constraint set. Snapping y exactly,
 * rather than trusting the projected velocity, is what keeps a rim on its wall.
 *
 * The clamps on y and z are written as negated comparisons on purpose: `pos <
 * min` and `pos > max` are BOTH false for a NaN, so the natural form passes a NaN
 * straight through to the ownership lookup. This form clamps it instead, and the
 * runaway check at report time says what happened.
 */
static void constrain(const MFTL_Domain *dom, bool on_rim, double pos[3])
{
  if (on_rim)
  {
    const double y_mid = 0.5 * (dom->global_min[1] + dom->global_max[1]);
    pos[1] = (pos[1] < y_mid) ? dom->global_min[1] : dom->global_max[1];
  }

  MFTLApplyPeriodicity(dom, pos);

  for (int d = 1; d < 3; ++d)
  {
    if (!(pos[d] >= dom->global_min[d])) pos[d] = dom->global_min[d];
    if (!(pos[d] <= dom->global_max[d])) pos[d] = dom->global_max[d];
  }
}

/**
 * @brief One RK2 (midpoint) step over every vertex this rank owns.
 *
 * Second order matters more here than in a monotonically stretching field: the
 * orbits are closed, and a first-order integrator turns each into an outward
 * spiral -- the wave would gain height every period from nothing but truncation
 * error, and near breaking that is the difference between a steep wave and an
 * overturned one.
 */
static void advect(MFTL_Ensemble *ens, const wave_field *f, double t, double dt)
{
  const MFTL_Domain *dom = &ens->domain;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];

    MFTL_FOREACH_OWNED_VERTEX(obj, {
      const bool on_rim = (rec->flags & MFTL_FLAG_BOUNDARY) != 0;
      double v1[3], mid[3], v2[3];

      wave_velocity(f, t, pos, v1);
      if (on_rim) v1[1] = 0.0;

      for (int d = 0; d < 3; ++d) mid[d] = pos[d] + 0.5 * dt * v1[d];
      constrain(dom, on_rim, mid);

      wave_velocity(f, t + 0.5 * dt, mid, v2);
      if (on_rim) v2[1] = 0.0;

      for (int d = 0; d < 3; ++d) pos[d] += dt * v2[d];
      constrain(dom, on_rim, pos);
    });
  }
}

/**
 * @brief Re-snaps every rim vertex in the pool, ghosts included, after a remesh.
 *
 * A pure function of the position applied to bit-identical copies, so owner and
 * ghost land on the same value without a message. The operators keep a rim on
 * its wall to the bit (a split midpoint of two points with the same y has that
 * y, and the ridge smoother moves a rim vertex along the chord of its two rim
 * neighbours); this is the guard, not the mechanism.
 */
static void snap_rim(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      if (MFTLVertexIsMeshBoundary(obj, v)) constrain(&ens->domain, MFTL_TRUE, MFTLVertexPosMut(obj, v));
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Diagnostics -- owned entities only, so every reduction is exact
 * ═══════════════════════════════════════════════════════════════════════════ */

enum
{
  S_EDGE_SUM,
  S_EDGE_N,
  S_Q_SUM,
  S_AREA,
  S_VOLUME,  ///< volume BELOW the surface
  S_TRIS,
  S_VERTS,
  S_RIM,  ///< owned mesh-boundary half-edges
  S_RIM_CUT,  ///< owned rim half-edges in triangles another rank holds a copy of
  S_HOLES,  ///< owned twin-less half-edges that are not mesh boundary
  S_OFF_WALL,  ///< owned rim vertices on neither channel wall
  S_INVERTED,  ///< owned triangles facing against every edge neighbour: a fold-over
  S_CREASES,  ///< edges folded by more than 120 degrees: unresolved curvature
  S_OVERTURNED,  ///< n_z <= 0: the surface is no longer single-valued there
  S_N
};

enum
{
  X_MIN_EDGE,
  X_MIN_Q,
  X_MIN_NZ,  ///< steepest face: smallest vertical normal component
  X_Z_MIN,
  X_NEG_MAX_EDGE,
  X_NEG_Z_MAX,
  X_NEG_U_OVER_C,  ///< -max over surface vertices of u_x / c
  X_N
};

typedef struct
{
  double sum[S_N];
  double min[X_N];  ///< every entry is a MIN; maxima are stored negated
} wave_stats;

/**
 * @brief True if another rank holds a copy of owned triangle @p t.
 *
 * The redistribution's own rule: a triangle is sent to the owner of each of its
 * corners and edge neighbours, so it is replicated exactly when one of them is a
 * ghost here. A rim half-edge in such a triangle is a piece of the surface's
 * boundary that meets a PARTITION boundary -- which is what makes a parallel
 * open-surface run test anything at all about parallel rims.
 */
static bool is_replicated(const MFTL_Object *obj, int t)
{
  int v[3], nb[3];
  MFTLGetTriangleVertices(obj, t, v);
  MFTLGetTriangleNeighbors(obj, t, nb);
  for (int k = 0; k < 3; ++k)
  {
    if (v[k] >= obj->verts.n_local || nb[k] >= obj->tris.n_local) return MFTL_TRUE;
  }
  return MFTL_FALSE;
}

/**
 * @brief Classifies triangle @p t against its edge neighbours.
 *
 * INVERTED: its normal faces against EVERY neighbour it has. That is what a
 * fold-over looks like -- a triangle turned over inside its own one-ring -- and it
 * is a remeshing error. CREASE: one edge across which the surface turns by more
 * than 120 degrees. That is not an orientation error at all; it is curvature the
 * mesh no longer resolves, and it is reported rather than failed on.
 *
 * The distinction is the reason this is not the vertex-normal test the LeVeque
 * example uses. A surface curled tighter than one edge length -- a rolled-up
 * mushroom cap, a neck about to pinch -- puts two consistently wound triangles
 * back to back across one edge, and a vertex normal averaged over that fold
 * points away from both. Measured here: every triangle that test flagged faced
 * against exactly one neighbour and agreed with the other two.
 *
 * Neighbours may be ghosts; their positions are current, and the halo is
 * complete, so an owned triangle sees all of them. Each crease is counted once
 * mesh-wide, by the owned triangle of smaller global id.
 */
static void fold_state(const MFTL_Object *obj, int t, const double n[3], double *inverted, double *creases)
{
  int present = 0, against = 0;
  for (int k = 0; k < 3; ++k)
  {
    const int h = 3 * t + k;
    const int tw = MFTLHeTwin(obj, h);
    if (tw == MFTL_ERROR) continue;

    const int u = MFTLHeTriIdx(tw);
    double m[3];
    MFTLGetTriangleNormal(obj, u, m);
    const double d = n[0] * m[0] + n[1] * m[1] + n[2] * m[2];

    present++;
    if (d < 0.0) against++;

    const bool mine = (u < obj->tris.n_local) ? (h < tw) : (obj->tris.rec[t].global_id < obj->tris.rec[u].global_id);
    if (mine && d < -0.5) *creases += 1.0;
  }
  if (present > 0 && against == present) *inverted += 1.0;
}

/**
 * @brief Collective survey of the whole surface.
 *
 * Every length goes through the minimum-image convention -- mandatory on this
 * mesh, where the seam triangles have vertices at opposite ends of the box in
 * stored coordinates. A plain difference across the seam would report a whole
 * domain length and inflate the area by orders of magnitude.
 *
 * The VOLUME BELOW THE SURFACE is V = integral over the surface of
 * (z - z_bottom) n_z dA: the channel walls are vertical (n_z = 0), x is periodic
 * (no wall at all), the bottom has z - z_bottom = 0, and the one-point centroid
 * rule is exact for a flat triangle. The field is solenoidal, so V is an
 * invariant and V/V0 measures what integration and remeshing cost.
 */
static void measure(MFTL_Ensemble *ens, const wave_field *f, double t, wave_stats *g)
{
  const MFTL_Domain *dom = &ens->domain;
  const double z_bottom = dom->global_min[2];

  double s[S_N];
  double x[X_N];
  memset(s, 0, sizeof s);
  for (int k = 0; k < X_N; ++k) x[k] = MFTL_DBL_MAX;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];

    for (int tr = 0; tr < obj->tris.n_local; ++tr)
    {
      double area = 0.0, n[3], c[3];
      MFTLGetTriangleProperties(obj, tr, &area, n, c);

      double sum_sq = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        const int h = 3 * tr + k;
        const double len_sq = MFTLGetEdgeLengthSquared(obj, h);
        const double len = sqrt(len_sq);

        sum_sq += len_sq;
        s[S_EDGE_SUM] += len;
        s[S_EDGE_N] += 1.0;
        x[X_MIN_EDGE] = MFTL_MIN(x[X_MIN_EDGE], len);
        x[X_NEG_MAX_EDGE] = MFTL_MIN(x[X_NEG_MAX_EDGE], -len);

        if (MFTLHeIsMeshBoundary(obj, h))
        {
          s[S_RIM] += 1.0;
          if (is_replicated(obj, tr)) s[S_RIM_CUT] += 1.0;
        }
        else if (MFTLHeIsLocalBoundary(obj, h))
          s[S_HOLES] += 1.0;
      }

      const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
      x[X_MIN_Q] = MFTL_MIN(x[X_MIN_Q], q);
      x[X_MIN_NZ] = MFTL_MIN(x[X_MIN_NZ], n[2]);
      s[S_Q_SUM] += q;
      s[S_AREA] += area;
      s[S_VOLUME] += (c[2] - z_bottom) * n[2] * area;
      if (n[2] <= 0.0) s[S_OVERTURNED] += 1.0;

      fold_state(obj, tr, n, &s[S_INVERTED], &s[S_CREASES]);
    }

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      x[X_Z_MIN] = MFTL_MIN(x[X_Z_MIN], p[2]);
      x[X_NEG_Z_MAX] = MFTL_MIN(x[X_NEG_Z_MAX], -p[2]);

      double u[3];
      wave_velocity(f, t, p, u);
      x[X_NEG_U_OVER_C] = MFTL_MIN(x[X_NEG_U_OVER_C], -u[0] / f->c);

      if (MFTLVertexIsMeshBoundary(obj, v))
      {
        const double tol = WAVE_WALL_TOL * dom->domain_size[1];
        if (fabs(p[1] - dom->global_min[1]) > tol && fabs(p[1] - dom->global_max[1]) > tol) s[S_OFF_WALL] += 1.0;
      }
    }

    s[S_TRIS] += (double) obj->tris.n_local;
    s[S_VERTS] += (double) obj->verts.n_local;
  }

  MPI_Allreduce(s, g->sum, S_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(x, g->min, X_N, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
}

/** @brief V - E + F from owned counts, with E = (3F + B)/2: 0 for an annulus. */
static double euler_of(const wave_stats *g) { return g->sum[S_VERTS] - 0.5 * (3.0 * g->sum[S_TRIS] + g->sum[S_RIM]) + g->sum[S_TRIS]; }

/** @brief Steepest face angle from the vertical normal component, in degrees. */
static double slope_degrees(double min_nz) { return atan2(sqrt(MFTL_MAX(0.0, 1.0 - min_nz * min_nz)), min_nz) * 180.0 / MFTL_PI; }

/**
 * @brief The verdict --verify acts on, identical on every rank.
 *
 * The overturned count is part of it deliberately: at the default steepness the
 * surface must stay single-valued, and a remesher that folds a steep crest over
 * shows up here before it shows up anywhere else.
 */
/**
 * @brief Per body: V-E+F, and whether it has a rim. What --verify's Euler check
 *        reads once topology changes can split the surface into several.
 *
 * Arrays over the body id space, reduced: a handful of bodies here. A solver
 * with thousands would use MFTLEnsembleReduceBodies() instead.
 */
typedef struct
{
  int n_bodies;  ///< bodies that exist anywhere
  int n_open;  ///< of which with a rim
  int bad;  ///< bodies whose V-E+F no orientable surface of their kind can have
  int chi_sum;  ///< V-E+F summed over all bodies
} body_census;

static void census_bodies(const MFTL_Ensemble *ens, body_census *c)
{
  const int n = ens->n_bodies_global;
  long long *a = (long long *) calloc(4 * (size_t) (n > 0 ? n : 1), sizeof(long long));
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    long long *r = &a[4 * obj->global_id];
    r[0] += obj->verts.n_local;
    r[1] += obj->tris.n_local;
    for (int h = 0; h < 3 * obj->tris.n_local; ++h)
      if (MFTLHeIsMeshBoundary(obj, h)) r[2]++;
    r[3] += obj->tris.n_local > 0 ? 1 : 0;
  }
  MPI_Allreduce(MPI_IN_PLACE, a, 4 * n, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  memset(c, 0, sizeof(*c));
  for (int b = 0; b < n; ++b)
  {
    if (a[4 * b + 3] == 0) continue;
    const long long chi = a[4 * b] - (3 * a[4 * b + 1] + a[4 * b + 2]) / 2 + a[4 * b + 1];
    const bool open = a[4 * b + 2] > 0;
    c->n_bodies++;
    c->chi_sum += (int) chi;
    if (open) c->n_open++;
    /* chi = 2 - 2g - (rim loops): at most 1 with a rim, an even at most 2 without. */
    if (open ? chi > 1 : (chi > 2 || (chi & 1) != 0)) c->bad++;
  }
  free(a);
}

static bool verdict(const wave_stats *g, double v0, const body_census *bc, char *why, size_t why_len)
{
  why[0] = '\0';
  if (g->sum[S_HOLES] != 0.0) snprintf(why, why_len, "%.0f twin-less interior half-edges (halo incomplete)", g->sum[S_HOLES]);
  if (bc == MFTL_NULL && fabs(euler_of(g)) > 0.5) snprintf(why, why_len, "V-E+F = %.0f, the seam came apart (an annulus has 0)", euler_of(g));
  if (bc != MFTL_NULL && bc->bad > 0) snprintf(why, why_len, "%d body(ies) with a V-E+F no orientable surface of its kind has", bc->bad);
  if (g->sum[S_OFF_WALL] != 0.0) snprintf(why, why_len, "%.0f rim vertices left their channel wall", g->sum[S_OFF_WALL]);
  if (g->sum[S_INVERTED] != 0.0) snprintf(why, why_len, "%.0f inverted triangles", g->sum[S_INVERTED]);
  if (bc == MFTL_NULL && g->sum[S_OVERTURNED] != 0.0) snprintf(why, why_len, "%.0f overturned triangles -- the wave broke", g->sum[S_OVERTURNED]);
  if (fabs(g->sum[S_VOLUME] / v0 - 1.0) > 0.01) snprintf(why, why_len, "volume below the surface drifted to V/V0 = %.5f", g->sum[S_VOLUME] / v0);
  return why[0] == '\0';
}

typedef struct
{
  long split, collapse, flip;  ///< global, since the last report
  int phases;
  MFTL_RemeshParallelStrategy strategy;
  bool any;
  bool serial;  ///< nproc == 1: MFTLRemeshEnsemble took the serial path, which has no strategy
} remesh_window;

static void print_header(bool remeshing)
{
  printf("\n  %5s %6s %7s %5s | %6s %5s %5s | %7s | %8s %8s %8s | %7s %6s | %7s %8s", "step", "t", "crest_x", "wraps", "height", "u/c", "slope", "ntri",
         "edge min", "mean", "max", "q worst", "mean", "area", "V/V0");
  if (remeshing) printf(" | %6s %6s %6s %3s %4s", "split", "coll", "flip", "ph", "str");
  printf("\n  ");
  for (int i = 0; i < (remeshing ? 150 : 118); ++i) putchar('-');
  printf("\n");
}

static void print_stats(int step, double t, double crest, int wraps, const wave_stats *g, double v0, const remesh_window *w, bool remeshing)
{
  const double mean_edge = (g->sum[S_EDGE_N] > 0.0) ? g->sum[S_EDGE_SUM] / g->sum[S_EDGE_N] : 0.0;
  const double mean_q = (g->sum[S_TRIS] > 0.0) ? g->sum[S_Q_SUM] / g->sum[S_TRIS] : 0.0;

  printf("  %5d %6.2f %7.3f %5d | %6.4f %5.3f %5.1f | %7.0f | %8.5f %8.5f %8.5f | %7.4f %6.4f | %7.4f %8.5f", step, t, crest, wraps,
         -g->min[X_NEG_Z_MAX] - g->min[X_Z_MIN], -g->min[X_NEG_U_OVER_C], slope_degrees(g->min[X_MIN_NZ]), g->sum[S_TRIS], g->min[X_MIN_EDGE], mean_edge,
         -g->min[X_NEG_MAX_EDGE], g->min[X_MIN_Q], mean_q, g->sum[S_AREA], g->sum[S_VOLUME] / v0);
  if (remeshing)
  {
    const char *str = !w->any ? "-" : w->serial ? "ser" : (w->strategy == MFTL_REMESH_PARALLEL_DISPERSED) ? "disp" : "span";
    printf(" | %6ld %6ld %6ld %3d %4s", w->split, w->collapse, w->flip, w->phases, str);
  }
  printf("\n");
  fflush(stdout);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Driver
 * ═══════════════════════════════════════════════════════════════════════════ */

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

  /* A channel four units long: periodic along the direction of travel, solid
   * across it and in the vertical. */
  double global_min[3] = {-2.0, -0.5, -1.0};
  double global_max[3] = {2.0, 0.5, 1.0};
  bool periodic[3] = {MFTL_TRUE, MFTL_FALSE, MFTL_FALSE};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, global_min, global_max, periodic, 1);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  wave_field field;
  wave_field_init(&field, &ens->domain, &cfg);

  MFTL_Object *surface = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, surface);

  const int expected_rim = 2 * cfg.nx;
  const int expected_tris = 2 * cfg.nx * cfg.ny;
  const int expected_verts = cfg.nx * (cfg.ny + 1);

  /* Built on rank 0 only, matching every library generator; one publication
   * then settles the id counters and global counts on every rank. */
  if (rank == 0)
  {
    if (build_periodic_patch(surface, &ens->domain, &field, cfg.nx, cfg.ny) != MFTL_SUCCESS)
    {
      fprintf(stderr, "Failed to build the periodic patch.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int shortest = 0, longest = 0;
    const int loops = count_boundary_loops(surface, &shortest, &longest);
    const int E = (3 * expected_tris + expected_rim) / 2;

    printf("=== near-breaking travelling waves on a periodically stitched open surface ===\n");
    printf("  %d rank(s), grid %dx%dx%d | channel x[%.1f,%.1f] PERIODIC, y[%.1f,%.1f] walls, z[%.1f,%.1f]\n", nproc, cfg.cuts[0], cfg.cuts[1], cfg.cuts[2],
           global_min[0], global_max[0], global_min[1], global_max[1], global_min[2], global_max[2]);
    printf("  mesh: %d x %d stitched patch -> V=%d  E=%d  F=%d  |  V-E+F = %d (annulus; a disk gives 1, a sphere 2)\n", cfg.nx, cfg.ny, expected_verts, E,
           expected_tris, expected_verts - E + expected_tris);
    printf("  boundary: %d half-edges in %d loop(s) of %d and %d -- one per channel wall, the seam is interior\n", expected_rim, loops, shortest, longest);

    int seam_tris = 0;
    for (int t = 0; t < surface->tris.n; ++t)
    {
      int v[3];
      MFTLGetTriangleVertices(surface, t, v);
      for (int a = 0; a < 3; ++a)
      {
        if (fabs(MFTLVertexPos(surface, v[a])[0] - MFTLVertexPos(surface, v[(a + 1) % 3])[0]) > 0.5 * ens->domain.domain_size[0])
        {
          seam_tris++;
          break;
        }
      }
    }
    printf("  %d triangles straddle the periodic seam (expected %d = 2*ny); only the minimum image measures them\n", seam_tris, 2 * cfg.ny);

    const double lambda = MFTL_TWO_PI / field.comp[0].k;
    printf("  primary wave: lambda %.3f, c %.3f, T %.3f, steady streamline with ka %.3f = %.0f %% of Stokes' limiting wave\n", lambda, field.c,
           lambda / field.c, cfg.steepness, 100.0 * cfg.steepness / WAVE_STOKES_LIMIT_KA);
    printf("                H %.4f (H/lambda %.4f), crest particle speed u/c %.3f (1 breaks), steepest face %.1f deg\n", field.height,
           field.height / lambda, field.crest_ratio, field.max_slope_deg);
    printf("                surface spacing under the crest %.1fx tighter than in the trough -- what the remesher meets\n",
           (1.0 + exp(streamline_zeta(MFTL_PI, field.psi))) / (1.0 - field.crest_ratio));
    for (int j = 1; j < field.n; ++j)
    {
      const wave_component *c = &field.comp[j];
      printf("  ripple [%d]: kA %.3f, lambda %.4f, c %.4f, T %.4f%s -- the sea is no longer steady\n", j, cfg.ripple, MFTL_TWO_PI / c->k, c->omega / c->k, MFTL_TWO_PI / c->omega,
             (c->ky != 0.0) ? ", oblique" : "");
    }
    printf("  the primary crest crosses the box every %.2f time units (%.0f steps)\n", ens->domain.domain_size[0] / cfg.speed,
           ens->domain.domain_size[0] / cfg.speed / cfg.dt);

    MFTLCheckMeshIntegrity(surface);
  }
  else
    MFTLObjectBuildDerived(surface);
  MFTLPublishGeneratedBodies(ens);

  /* Scatter. `surface` is dead after this: the local list is re-read below. */
  MFTLComputeTriangleGeometry(surface, surface->tris.n);
  MFTLRedistribute(ens);
  surface = MFTL_NULL;

  const double target = (cfg.target > 0.0) ? cfg.target : MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  /* A smooth interface: feature detection off. The two rims remain features
   * regardless -- a mesh boundary always is -- and slide as ridges. */
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, cfg.iterations, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = cfg.smooth;
  params.parallel_strategy = cfg.strategy;
  params.parallel_phase_budget = cfg.budget;

  const bool remeshing = cfg.remesh_every > 0;

  /* Topology: a lip within one edge of the face below it reconnects, a sheet
   * tears at one and a half, the result is remeshed with the loop's parameters. */
  MFTL_TopologyOptions topt = MFTLTopologyDefaultOptions();
  topt.dx = (cfg.topo_dx > 0.0) ? cfg.topo_dx : target;
  topt.delta_breakup = (cfg.delta_breakup > 0.0) ? cfg.delta_breakup : 1.5 * target;
  topt.delta_coalescence = (cfg.delta_coalescence > 0.0) ? cfg.delta_coalescence : 1.0 * target;
  topt.remesh = remeshing ? &params : MFTL_NULL;
  const int topo_every = (cfg.topo_every > 0) ? cfg.topo_every : (remeshing ? cfg.remesh_every : 2);

  wave_stats g0;
  measure(ens, &field, 0.0, &g0);
  const double v0 = g0.sum[S_VOLUME];

  if (rank == 0)
  {
    if (remeshing)
    {
      printf("  remesh every %d step(s): target %.5f (band %.5f .. %.5f), %d iterations, %d smoothing sweeps, budget %d\n", cfg.remesh_every, target,
             params.min_edge_length, params.max_edge_length, cfg.iterations, cfg.smooth, cfg.budget);
    }
    else
      printf("  remeshing OFF: fixed connectivity, the degradation baseline\n");
    print_header(remeshing);
    remesh_window none = {0, 0, 0, 0, MFTL_REMESH_PARALLEL_SPANNING, MFTL_FALSE, MFTL_FALSE};
    print_stats(0, 0.0, crest_position(&field, &ens->domain, 0.0), 0, &g0, v0, &none, remeshing);
  }

  if (cfg.vtk_every > 0)
  {
    char prefix[256];
    snprintf(prefix, sizeof prefix, "%s_%04d", cfg.out, 0);
    MFTLWriteVtu(ens, prefix, MFTL_NULL);
  }

  /* ── The per-timestep loop ─────────────────────────────────────────────── */
  double t = 0.0;
  double crest_prev = crest_position(&field, &ens->domain, 0.0);
  int wraps = 0;
  int broken = 0;
  double rim_cut_max = 0.0;
  int n_passes = 0, n_not_converged = 0;
  long total_ops = 0;
  double t_remesh = 0.0, max_u_over_c = -g0.min[X_NEG_U_OVER_C], max_slope = slope_degrees(g0.min[X_MIN_NZ]), worst_q = g0.min[X_MIN_Q];
  double min_edge = g0.min[X_MIN_EDGE], max_edge = -g0.min[X_NEG_MAX_EDGE];
  remesh_window win = {0, 0, 0, 0, MFTL_REMESH_PARALLEL_SPANNING, MFTL_FALSE, MFTL_FALSE};
  wave_stats g = g0;

  double v_surgery = 0.0, t_topology = 0.0;
  int n_changes = 0, n_events_total = 0, max_bodies = 1;
  bool genus_changed = MFTL_FALSE;
  body_census bc = {1, 1, 0, 0};

  for (int step = 1; step <= cfg.steps; ++step)
  {
    advect(ens, &field, t, cfg.dt);
    t += cfg.dt;

    /* The crest coordinate jumps backwards exactly when it leaves through the
     * right face, which is the cheapest way to count crossings. */
    const double crest = crest_position(&field, &ens->domain, t);
    if (crest < crest_prev) wraps++;
    crest_prev = crest;

    MFTLUpdateGhostPositions(ens);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);

    if (remeshing && step % cfg.remesh_every == 0)
    {
      MFTL_RemeshStats st;
      memset(&st, 0, sizeof st);

      const double a = MPI_Wtime();
      if (MFTLRemeshEnsemble(ens, &params, &st) != MFTL_SUCCESS)
      {
        if (rank == 0) fprintf(stderr, "remeshing failed at step %d\n", step);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      t_remesh += MPI_Wtime() - a;
      snap_rim(ens);

      long ops[3] = {st.n_split, st.n_collapse, st.n_flip}, gops[3];
      MPI_Allreduce(ops, gops, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      win.split += gops[0];
      win.collapse += gops[1];
      win.flip += gops[2];
      win.phases = st.n_phases;
      win.strategy = st.strategy;
      win.any = MFTL_TRUE;
      win.serial = (nproc == 1);
      total_ops += gops[0] + gops[1] + gops[2];
      n_passes++;
      if (!st.converged) n_not_converged++;
    }
    else if (MFTLRedistribute(ens) != MFTL_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 1);

    /* Topology: detect, and only when something was found, reconstruct -- with
     * the volume below the surface measured around it. */
    if (cfg.topology && step % topo_every == 0)
    {
      const double a = MPI_Wtime();
      MFTL_TopologyEvent *events = MFTL_NULL;
      int n_events = 0;
      if (MFTLTopologyDetectEvents(ens, &topt, &events, &n_events) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      if (n_events > 0)
      {
        wave_stats before, after;
        measure(ens, &field, t, &before);
        int *bodies = (int *) malloc(2 * (size_t) n_events * sizeof(int));
        for (int e = 0; e < n_events; ++e)
        {
          bodies[2 * e] = events[e].body_a;
          bodies[2 * e + 1] = events[e].body_b;
        }
        MFTL_TopologyStats tst;
        if (MFTLEnsembleReconstructBodies(ens, bodies, 2 * n_events, events, n_events, &topt, &tst) != MFTL_SUCCESS)
        {
          if (rank == 0) fprintf(stderr, "topology change failed at step %d\n", step);
          MPI_Abort(MPI_COMM_WORLD, 1);
        }
        free(bodies);
        snap_rim(ens);
        measure(ens, &field, t, &after);
        const int chi_before = bc.chi_sum;
        census_bodies(ens, &bc);
        if (bc.chi_sum != chi_before) genus_changed = MFTL_TRUE;
        max_bodies = MFTL_MAX(max_bodies, bc.n_bodies);
        v_surgery += after.sum[S_VOLUME] - before.sum[S_VOLUME];
        n_changes++;
        n_events_total += n_events;
        if (rank == 0)
          printf(
              "  step %5d t %6.2f: TOPOLOGY -- %d breakup / %d coalescence event(s); %d body(ies) in, %d out; now %d bodies (%d open), "
              "V-E+F %d; volume moved %+.2e\n",
              step, t, tst.n_breakup, tst.n_coalescence, tst.n_bodies_reconstructed, tst.n_bodies_out, bc.n_bodies, bc.n_open, bc.chi_sum,
              after.sum[S_VOLUME] - before.sum[S_VOLUME]);
      }
      free(events);
      t_topology += MPI_Wtime() - a;
    }

    if (cfg.report_every > 0 && (step % cfg.report_every == 0 || step == cfg.steps))
    {
      measure(ens, &field, t, &g);
      if (rank == 0) print_stats(step, t, crest, wraps, &g, v0, &win, remeshing);
      memset(&win, 0, sizeof win);

      max_u_over_c = MFTL_MAX(max_u_over_c, -g.min[X_NEG_U_OVER_C]);
      max_slope = MFTL_MAX(max_slope, slope_degrees(g.min[X_MIN_NZ]));
      worst_q = MFTL_MIN(worst_q, g.min[X_MIN_Q]);
      min_edge = MFTL_MIN(min_edge, g.min[X_MIN_EDGE]);
      max_edge = MFTL_MAX(max_edge, -g.min[X_NEG_MAX_EDGE]);

      rim_cut_max = MFTL_MAX(rim_cut_max, g.sum[S_RIM_CUT]);

      char why[160];
      if (!verdict(&g, v0 + v_surgery, cfg.topology ? &bc : MFTL_NULL, why, sizeof why))
      {
        broken++;
        if (rank == 0) fprintf(stderr, "  step %d: %s\n", step, why);
      }

      /* The runaway guard for the e^{k(z-z_ref)} feedback. A sea that has broken
       * does not stop at breaking: the lip keeps accelerating through the
       * exponential, the remesher then grinds through a curling sheet of ever
       * shorter edges, and the run slows to a crawl long before the height
       * reaches the walls. Past u/c = 1.5 the kinematics describe nothing. */
      if (-g.min[X_NEG_Z_MAX] - g.min[X_Z_MIN] > 0.5 * ens->domain.domain_size[2] || -g.min[X_NEG_U_OVER_C] > 1.5)
      {
        if (rank == 0)
          fprintf(stderr, "\n  step %d: the wave broke and ran away (u/c %.2f, height %.3f) -- lower --steepness or --ripple.\n", step,
                  -g.min[X_NEG_U_OVER_C], -g.min[X_NEG_Z_MAX] - g.min[X_Z_MIN]);
        broken++;
        break;
      }
    }

    if (cfg.vtk_every > 0 && step % cfg.vtk_every == 0)
    {
      char prefix[256];
      snprintf(prefix, sizeof prefix, "%s_%04d", cfg.out, step);
      MFTLWriteVtu(ens, prefix, MFTL_NULL);
    }
  }

  measure(ens, &field, t, &g);
  /* Non-vacuity, for the CTests. A parallel run whose rim never met a partition
   * boundary exercised the serial remesher on a partitioned interior and nothing
   * about parallel RIMS -- it would still pass with the rim handling deleted, and
   * a grid or resolution change can make a test drift into that state without
   * anyone noticing. --require-rim-cut turns the count into a verdict. */
  if (cfg.require_rim_cut && nproc > 1 && rim_cut_max == 0.0)
  {
    broken++;
    if (rank == 0) fprintf(stderr, "  VACUOUS: the rim never met a partition boundary -- choose a grid whose cut planes cross it\n");
  }
  if (cfg.require_topology && max_bodies == 1 && !genus_changed)
  {
    broken++;
    if (rank == 0) fprintf(stderr, "  VACUOUS: neither the body count nor the genus ever changed -- make the sea break harder or run longer\n");
  }

  double t_max = 0.0, t_topo_max = 0.0;
  MPI_Allreduce(&t_remesh, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&t_topology, &t_topo_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (cfg.topology) census_bodies(ens, &bc);

  if (rank == 0)
  {
    if (cfg.topology)
    {
      printf("\n  TOPOLOGY: %d reconstruction(s) from %d event(s); up to %d bodies; now %d (%d open), V-E+F summed %d%s\n", n_changes, n_events_total,
             max_bodies, bc.n_bodies, bc.n_open, bc.chi_sum, genus_changed ? ", genus changed" : "");
      printf("            volume moved by reconstructions %+.3e (V0 %.6f), %.3f s in topology calls (max over ranks)\n", v_surgery, v0, t_topo_max);
    }
    const double flat_area = ens->domain.domain_size[0] * ens->domain.domain_size[1];

    printf("\n  crest travel     %.2f box lengths in %.1f time units, %d seam re-entries observed\n", field.c * t / ens->domain.domain_size[0], t, wraps);
    printf("  near breaking    u_crest/c up to %.3f (1 breaks), steepest face %.1f deg, overturned triangles now %.0f\n", max_u_over_c, max_slope,
           g.sum[S_OVERTURNED]);
    printf("  surface area     %.6f (flat would be %.4f, ratio %.4f)\n", g.sum[S_AREA], flat_area, g.sum[S_AREA] / flat_area);
    printf("  triangles        %.0f -> %.0f   (rim %.0f -> %.0f half-edges)\n", g0.sum[S_TRIS], g.sum[S_TRIS], g0.sum[S_RIM], g.sum[S_RIM]);
    printf("  edge length      %.6f .. %.6f over the run at the reports (started %.6f .. %.6f)\n", min_edge, max_edge, g0.min[X_MIN_EDGE],
           -g0.min[X_NEG_MAX_EDGE]);
    printf("  worst quality    %.4f at the start, %.4f at the worst report, %.4f now\n", g0.min[X_MIN_Q], worst_q, g.min[X_MIN_Q]);
    printf("  volume below     V/V0 = %.6f\n", g.sum[S_VOLUME] / v0);
    if (cfg.topology)
      printf("  Euler V-E+F      %.0f summed over %d bodies, %s\n", euler_of(&g), bc.n_bodies, bc.bad ? "SOME INCONSISTENT" : "each consistent with its kind");
    else
      printf("  Euler V-E+F      %.0f   (0 for an annulus: the seam held)\n", euler_of(&g));
    printf("  orientation      %.0f inverted triangles, %.0f creases sharper than 120 deg\n", g.sum[S_INVERTED], g.sum[S_CREASES]);
    if (remeshing)
    {
      printf("  remeshing        %d passes, %ld operations (%.1f per pass), %d not converged, %.3f s (max over ranks)\n", n_passes, total_ops,
             n_passes ? (double) total_ops / (double) n_passes : 0.0, n_not_converged, t_max);
    }
    printf("  rim across ranks up to %.0f rim half-edges in triangles another rank also holds%s\n", rim_cut_max,
           (nproc > 1) ? "" : " (one rank: none can be)");
    printf("  structure        %s\n", (broken == 0) ? "sound at every report" : "BROKEN -- see above");
    if (cfg.vtk_every > 0) printf("  wrote %s_*.pvtu every %d steps.\n", cfg.out, cfg.vtk_every);
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (cfg.verify && broken > 0) ? 1 : 0;
}
