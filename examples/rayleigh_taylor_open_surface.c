/*
 * @Author: Christian Gorges
 * @Date: 2026-09-24
 *
 * @brief A kinematic Rayleigh-Taylor emulation on an open surface, remeshed in
 *        parallel every few steps.
 *
 * An open surface is generated horizontally across the middle of the box and
 * advected by a PRESCRIBED velocity field whose structure reproduces what a
 * single-mode Rayleigh-Taylor interface looks like: heavy fluid falls in a
 * narrow spike, light fluid rises in broad bubbles, the spike stem thins as
 * material converges onto it from above, and the tip flares outward into a
 * mushroom cap. The surface area grows several-fold, so the mesh is kept at its
 * starting resolution by the DISTRIBUTED remesher (MFTLRemeshEnsemble), which is
 * what this example exists to exercise on a surface with a rim.
 *
 * ── What this is NOT ───────────────────────────────────────────────────────
 *
 * There is no momentum equation, no pressure, no Atwood number, no surface
 * tension and no feedback of the interface on the flow. The field is imposed,
 * so the mode is chosen by the FIELD rather than selected by the instability,
 * and nothing here would reproduce a growth rate, a mixing width or a bubble
 * competition law. It is a kinematic caricature that LOOKS right and, far more
 * usefully, DEFORMS right: it is the interface stretching -- not the physics --
 * that a remesher has to survive.
 *
 * ── The velocity field ─────────────────────────────────────────────────────
 *
 * Per mode (m_x, m_y), with k_x = 2*pi*m_x/L_x, k_y = 2*pi*m_y/L_y, an envelope
 * S(z) in the vertical and zeta = pi*(z-z_mid)/L_z in [-pi/2, pi/2]:
 *
 *     u =  (V / 2k_x) * sin(k_x x) * cos(k_y y) * S'(z)
 *     v =  (V / 2k_y) * cos(k_x x) * sin(k_y y) * S'(z)
 *     w = -V          * cos(k_x x) * cos(k_y y) * S(z)
 *
 *     S(zeta) = cos(zeta) * (1 - b*sin(zeta)),     0 <= b < 1   (--asym)
 *
 * Four properties, each of them load-bearing:
 *
 *   1. EXACTLY divergence-free: du/dx + dv/dy = V*cx*cy*S' = -dw/dz, identically,
 *      FOR ANY S. An interface advected by a solenoidal field stretches without
 *      being artificially compressed or inflated, so the triangle degradation
 *      this example produces is genuine strain rather than an artefact of the
 *      driver -- and the volume BELOW the interface is an invariant, which is
 *      what the V/V0 column measures (see the diagnostics section).
 *
 *   2. Nothing can leave the box. S(+-L_z/2) = 0 kills the vertical velocity at
 *      the top and bottom walls (true for any b); sin(k*(+-L/2)) = sin(+-pi*m)
 *      = 0 kills the wall-normal horizontal velocity at the side walls -- which
 *      is WHY the mode numbers must be integers (see --mode-x). The field is
 *      free-slip on all six walls by construction, not by clamping.
 *
 *   3. The asymmetry b is what makes it RT rather than a standing wave. With
 *      b = 0 the envelope is symmetric about mid-height, so S'(z_mid) = 0 and
 *      there is NO horizontal velocity at the initial interface at all: the
 *      surface just bobs up and down. b > 0 gives S'(0) = -b*pi/L_z < 0, so flow
 *      CONVERGES on the spike axis from t = 0 and the stem thins, and S > 1 below
 *      mid-height makes the spike outrun the bubble -- the classic high-Atwood
 *      asymmetry. b < 1 keeps S >= 0, so the spike never reverses.
 *
 *   4. V(t) = v_max * tanh(growth * t): zero at t = 0 with finite slope, then
 *      saturating at a terminal velocity. The right shape at both ends.
 *
 * ── The mushroom cap needs a vortex ring, and that part is a hack ──────────
 *
 * The modal field produces a falling TONGUE, not a mushroom: its outward
 * velocity near the axis goes as sin(k_x x) ~ k_x x, so on the axis, where the
 * tip is, there is nothing to flare it with. A real mushroom is a
 * Kelvin-Helmholtz roll-up of the shear layer along the spike's flanks, and
 * concentrated vorticity is the one ingredient a smooth modal field cannot
 * supply. So it is supplied explicitly: an axisymmetric VORTEX RING (--roll)
 * riding just behind the descending tip, from a Stokes stream function
 * psi = r*A_phi with A_phi = C*(r/R)*exp(-((r-R)^2 + (z-z_c)^2)/2sigma^2):
 *
 *     u_r = -(1/r) dpsi/dz,      u_z = (1/r) dpsi/dr
 *
 * exactly divergence-free, so the total field still is. The roll-up is IMPOSED
 * at a hand-placed location, not emergent from a shear instability: honest as a
 * geometry generator, dishonest as physics. `--roll 0` leaves the pure modal
 * field. The Gaussian sits ~7 sigma from the nearest wall, so it perturbs the
 * free-slip property by O(1e-9), which the wall snap absorbs.
 *
 * ── The rim ────────────────────────────────────────────────────────────────
 *
 * The patch spans the full cross-section, so its boundary lies ON the four side
 * walls. Two treatments, and they are genuinely different test cases:
 *
 *   --rim slide  (default)  The free-slip wall: the wall-normal velocity
 *                           component is projected out and the coordinate
 *                           snapped back onto the wall. A no-op in exact
 *                           arithmetic (property 2); it earns its place against
 *                           round-off and against the remesher, see below.
 *
 *   --rim pin               Rim vertices never move: a membrane clamped to a
 *                           frame. The HARDER remeshing case -- the pinned ring
 *                           cannot follow the interface and the triangles next
 *                           to it shear without bound -- and the one where the
 *                           volume below the interface is NOT conserved, because
 *                           the rim stops being a material curve.
 *
 * ── Which wall a rim vertex is on is read from its POSITION, and only for ──
 * ── rim vertices ───────────────────────────────────────────────────────────
 *
 * The MFTL_5 version stamped wall-membership bits into `flags` at generation
 * time and argued that a constraint is an identity, not a coordinate. Two things
 * make that wrong here. The bits it used (0x2, 0x4) are MFTL_FLAG_DEAD and
 * MFTL_FLAG_VISITED in this library -- the remesher would read every rim vertex
 * as retired. And more fundamentally, a split of a rim edge creates a NEW rim
 * vertex, and the operators hand it MFTL_FLAG_BOUNDARY and nothing else: an
 * application bit does not survive the first remeshing pass.
 *
 * Restricted to MFTL_FLAG_BOUNDARY vertices the coordinate test is also
 * unambiguous, which is the argument the MFTL_5 comment was really making: every
 * rim vertex of this patch lies on a side wall by construction, and the
 * remeshing operators keep it there -- a split midpoint of two points on one wall
 * is on that wall to the bit, a rim collapse keeps a rim position, the ridge
 * smoother moves a rim vertex along the chord of its two rim neighbours (so the
 * wall coordinate is untouched), and the box corners are CORNER vertices that no
 * operator moves at all. An INTERIOR vertex that wanders onto a wall plane is
 * never asked the question. The snap after every remesh absorbs round-off, and
 * `--verify` checks that no rim vertex has left its wall.
 *
 * ── The loop ───────────────────────────────────────────────────────────────
 *
 *   advect owned vertices -> MFTLUpdateGhostPositions -> triangle geometry
 *   (ghosts too) -> MFTLRemeshEnsemble every --remesh-every steps, plain
 *   MFTLRedistribute otherwise -> re-snap the rim
 *
 * MFTLRemeshEnsemble measures the body and picks its strategy collectively; an
 * interface spanning the whole box is SPANNING (move the cut planes), and the
 * rim then crosses every cut plane it meets -- which is the configuration the
 * spanning driver's freeze classification had never been run on before these
 * examples. `--remesh-every 0` turns the remesher off and reproduces the
 * degradation baseline.
 *
 * Measured at the defaults, np = 1, 2 and 4 alike: the area grows 5.8x and the
 * mesh from 4.6k to ~20.5k triangles, the rim from 192 to ~590 half-edges -- on
 * its walls to the bit throughout -- the volume below the interface holds to
 * 5e-4, and no triangle is ever inverted. From t ~ 2.5 on, creases appear where
 * the vortex ring winds the mushroom cap tighter than one edge; that is the
 * imposed roll-up outrunning the resolution, and the `crs` column counts it.
 * Without the remesher the same run has 48 inverted triangles by t = 3.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P rayleigh_taylor_open_surface [cx cy cz] [options]
 *
 *     cx cy cz           process grid, cx*cy*cz == P. Default: P factored over
 *                        x and y as evenly as possible, one cell in z
 *
 *     --res N            patch resolution per side      (default 48 -> 4608 tris)
 *     --steps N          timesteps                      (default 400)
 *     --dt X             timestep                       (default 0.01)
 *     --vmax X           terminal velocity scale        (default 0.35)
 *     --growth X         acceleration rate              (default 1.5)
 *     --asym X           up/down asymmetry b, [0,1)     (default 0.6)
 *     --roll X           vortex-ring strength           (default 0.8; 0 = none)
 *     --perturb X        initial interface ripple       (default 0.02)
 *     --mode-x M         integer mode number in x       (default 1)
 *     --mode-y M         integer mode number in y       (default 1)
 *     --multimode        add a second, off-symmetry mode (bubble competition)
 *     --rim MODE         slide | pin                    (default slide)
 *
 *     --remesh-every N   remesh every N steps, 0 = never (default 5)
 *     --target X         target edge length             (default: initial mean)
 *     --iterations N     remeshing iterations per pass  (default 3)
 *     --smooth N         smoothing sweeps per iteration (default 3)
 *     --budget N         cut-plane placements per pass  (default 4; see
 *                        MFTL_RemeshParameters.parallel_phase_budget)
 *     --strategy S       auto | spanning | dispersed    (default auto)
 *
 *     --vtk N            write a .pvtu every N steps    (default 20, 0 disables)
 *     --report N         print a line every N steps     (default 20)
 *     --out PREFIX       output file prefix             (default "rt")
 *     --verify           exit non-zero unless the halo, the rim, the Euler
 *                        characteristic, the orientation and (rim slide) the
 *                        volume below the interface all hold at every report
 *     --require-rim-cut  --verify, and fail if at np > 1 no rim triangle ever
 *                        met a partition boundary (what the CTest passes)
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/** @brief Wall membership of a RIM vertex, as a bitmask. Never stored in `flags`. */
#define RT_WALL_X 0x1
#define RT_WALL_Y 0x2

/** @brief How close to a wall a rim vertex must sit, relative to the box width. */
#define RT_WALL_TOL 1e-9

#define RT_MAX_MODES 2

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int cuts[3];
  int res;
  int steps;
  double dt;
  double vmax;
  double growth;
  double asym;
  double roll;
  double perturb;
  int mx, my;
  bool multimode;
  bool pin_rim;

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
} config;

/** @brief cx*cy == nproc as square as possible, cz = 1: the interface is horizontal. */
static void default_grid(int nproc, int cuts[3])
{
  int cy = 1;
  for (int f = 1; f * f <= nproc; ++f)
    if (nproc % f == 0) cy = f;
  cuts[0] = nproc / cy;
  cuts[1] = cy;
  cuts[2] = 1;
}

static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  default_grid(nproc, cfg->cuts);
  cfg->res = 48;
  cfg->steps = 400;
  cfg->dt = 0.01;
  cfg->vmax = 0.35;
  cfg->growth = 1.5;
  cfg->asym = 0.6;
  cfg->roll = 0.8;
  cfg->perturb = 0.02;
  cfg->mx = 1;
  cfg->my = 1;
  cfg->multimode = MFTL_FALSE;
  cfg->pin_rim = MFTL_FALSE;
  cfg->remesh_every = 5;
  cfg->target = 0.0;
  cfg->iterations = 3;
  cfg->smooth = 3;
  cfg->budget = 4;
  cfg->strategy = MFTL_REMESH_PARALLEL_AUTO;
  cfg->vtk_every = 20;
  cfg->report_every = 20;
  cfg->verify = MFTL_FALSE;
  cfg->require_rim_cut = MFTL_FALSE;
  snprintf(cfg->out, sizeof cfg->out, "rt");

  int i = 1;
  if (argc >= 4 && argv[1][0] != '-')
  {
    for (int d = 0; d < 3; ++d) cfg->cuts[d] = atoi(argv[1 + d]);
    i = 4;
  }

  for (; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--multimode"))
      cfg->multimode = MFTL_TRUE;
    else if (!strcmp(argv[i], "--verify"))
      cfg->verify = MFTL_TRUE;
    else if (!strcmp(argv[i], "--require-rim-cut"))
      cfg->verify = cfg->require_rim_cut = MFTL_TRUE;
    else if (!strcmp(argv[i], "--res") && i + 1 < argc)
      cfg->res = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "--vmax") && i + 1 < argc)
      cfg->vmax = atof(argv[++i]);
    else if (!strcmp(argv[i], "--growth") && i + 1 < argc)
      cfg->growth = atof(argv[++i]);
    else if (!strcmp(argv[i], "--asym") && i + 1 < argc)
      cfg->asym = atof(argv[++i]);
    else if (!strcmp(argv[i], "--roll") && i + 1 < argc)
      cfg->roll = atof(argv[++i]);
    else if (!strcmp(argv[i], "--perturb") && i + 1 < argc)
      cfg->perturb = atof(argv[++i]);
    else if (!strcmp(argv[i], "--mode-x") && i + 1 < argc)
      cfg->mx = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--mode-y") && i + 1 < argc)
      cfg->my = atoi(argv[++i]);
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
    else if (!strcmp(argv[i], "--rim") && i + 1 < argc)
    {
      ++i;
      if (!strcmp(argv[i], "pin"))
        cfg->pin_rim = MFTL_TRUE;
      else if (!strcmp(argv[i], "slide"))
        cfg->pin_rim = MFTL_FALSE;
      else
      {
        if (rank == 0) fprintf(stderr, "--rim must be 'slide' or 'pin' (got '%s')\n", argv[i]);
        return MFTL_ERROR;
      }
    }
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
  if (cfg->res < 1 || cfg->steps < 0 || cfg->mx < 1 || cfg->my < 1 || cfg->remesh_every < 0 || cfg->iterations < 1 || cfg->smooth < 0)
  {
    /* Non-integer mode numbers would leave a non-zero wall-normal velocity at
     * the side walls, and the free-slip property this example relies on would
     * quietly stop holding. */
    if (rank == 0) fprintf(stderr, "Need res >= 1, steps >= 0, integer mode numbers >= 1, remesh-every >= 0, iterations >= 1, smooth >= 0.\n");
    return MFTL_ERROR;
  }
  if (cfg->asym < 0.0 || cfg->asym >= 1.0)
  {
    /* At b >= 1 the envelope changes sign near the bottom wall and the spike
     * would reverse direction mid-fall -- not an RT morphology at all. */
    if (rank == 0) fprintf(stderr, "--asym must lie in [0, 1) (got %f).\n", cfg->asym);
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The prescribed velocity field
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double kx, ky;
  double weight;
} rt_mode;

typedef struct
{
  double mid[3];  ///< box centre: the field is written about it
  double inv_lz_pi;  ///< pi / L_z, for the vertical envelope
  rt_mode modes[RT_MAX_MODES];
  int n_modes;
  double vmax;
  double growth;
  double asym;  ///< b in S(zeta) = cos(zeta)*(1 - b*sin(zeta)); 0 = standing wave

  double roll;  ///< vortex-ring strength; 0 disables it
  double ring_radius;  ///< R, the ring's radius about the spike axis
  double ring_sigma;  ///< core thickness; also sets how compact its support is
  double ring_lag;  ///< ring centre sits this many R ABOVE the tip
} rt_field;

static void rt_field_init(rt_field *f, const MFTL_Domain *dom, const config *cfg)
{
  for (int d = 0; d < 3; ++d) f->mid[d] = 0.5 * (dom->global_min[d] + dom->global_max[d]);

  f->inv_lz_pi = MFTL_PI / dom->domain_size[2];
  f->vmax = cfg->vmax;
  f->growth = cfg->growth;
  f->asym = cfg->asym;

  f->modes[0].kx = MFTL_TWO_PI * (double) cfg->mx / dom->domain_size[0];
  f->modes[0].ky = MFTL_TWO_PI * (double) cfg->my / dom->domain_size[1];
  f->modes[0].weight = 1.0;
  f->n_modes = 1;

  if (cfg->multimode)
  {
    /* A shorter, deliberately off-symmetry mode. Integer mode numbers keep the
     * free-slip wall property and a sum of solenoidal fields is solenoidal, so
     * the interface simply grows several competing bubbles of unequal size. */
    f->modes[1].kx = MFTL_TWO_PI * (double) (cfg->mx + 1) / dom->domain_size[0];
    f->modes[1].ky = MFTL_TWO_PI * (double) (cfg->my + 2) / dom->domain_size[1];
    f->modes[1].weight = 0.35;
    f->n_modes = 2;
  }

  /* Sized against the primary wavelength, so the ring stays comparable to the
   * stem it has to curl however the box or the mode changes: the measured stem
   * radius at mid-depth is ~0.13 wavelengths. */
  const double wavelength = dom->domain_size[0] / (double) cfg->mx;
  f->roll = cfg->roll;
  f->ring_radius = 0.13 * wavelength;
  f->ring_sigma = 0.45 * f->ring_radius;
  f->ring_lag = 0.8;
}

/**
 * @brief The modal base field. Analytic, local, no mesh access -- so the
 *        advection step needs no communication at all.
 */
static void rt_base_velocity(const rt_field *f, double t, const double p[3], double out[3])
{
  const double V = f->vmax * tanh(f->growth * t);

  const double zeta = f->inv_lz_pi * (p[2] - f->mid[2]);
  const double sz = sin(zeta), cz = cos(zeta);

  const double s_env = cz * (1.0 - f->asym * sz);
  const double ds_env = f->inv_lz_pi * (-sz - f->asym * (cz * cz - sz * sz));

  const double dx = p[0] - f->mid[0];
  const double dy = p[1] - f->mid[1];

  out[0] = out[1] = out[2] = 0.0;

  for (int m = 0; m < f->n_modes; ++m)
  {
    const double kx = f->modes[m].kx;
    const double ky = f->modes[m].ky;
    const double a = V * f->modes[m].weight;

    const double cx = cos(kx * dx), sx = sin(kx * dx);
    const double cy = cos(ky * dy), sy = sin(ky * dy);

    out[0] += (a / (2.0 * kx)) * sx * cy * ds_env;
    out[1] += (a / (2.0 * ky)) * cx * sy * ds_env;
    out[2] += -a * cx * cy * s_env;
  }
}

/**
 * @brief Adds the vortex ring riding just behind the spike tip.
 *
 *     u_r = C*(r/R)*E*(z-z_c)/sigma^2,     u_z = C*E*(2/R - (r/R)*(r-R)/sigma^2)
 *
 * exactly divergence-free and finite on the axis because of the (r/R) factor.
 * C < 0 curls a FALLING tip outward and back over itself; the strength tracks
 * V(t), so the cap develops only once the spike exists.
 */
static void rt_add_ring(const rt_field *f, double t, double z_tip, const double p[3], double out[3])
{
  if (f->roll <= 0.0) return;

  const double R = f->ring_radius;
  const double inv_s2 = 1.0 / (f->ring_sigma * f->ring_sigma);

  const double dx = p[0] - f->mid[0];
  const double dy = p[1] - f->mid[1];
  const double r = sqrt(dx * dx + dy * dy);
  const double dz = p[2] - (z_tip + f->ring_lag * R);
  const double dr = r - R;

  const double arg = 0.5 * (dr * dr + dz * dz) * inv_s2;
  if (arg > 30.0) return;  // e^-30: below double precision's reach on the sum anyway

  const double e = exp(-arg);
  const double c = -f->roll * f->vmax * tanh(f->growth * t) * R * 0.5;

  const double u_r = c * (r / R) * e * dz * inv_s2;
  out[2] += c * e * (2.0 / R - (r / R) * dr * inv_s2);

  if (r > MFTL_VERYSMALL)
  {
    out[0] += u_r * dx / r;
    out[1] += u_r * dy / r;
  }
}

static void rt_velocity(const rt_field *f, double t, double z_tip, const double p[3], double out[3])
{
  rt_base_velocity(f, t, p, out);
  rt_add_ring(f, t, z_tip, p, out);
}

/**
 * @brief Advances the spike tip, a material point on the axis, by @p dt.
 *
 * The axis is a streamline of both parts of the field, so only its height is
 * integrated. Driven by the BASE field alone: the ring is placed relative to
 * the tip, and letting it also move the tip would close a feedback loop for no
 * physical gain. Every rank integrates the same scalar ODE from the same value
 * with the same arithmetic, so they agree without communicating.
 */
static double advance_tip(const rt_field *f, double z_tip, double t, double dt)
{
  double p[3] = {f->mid[0], f->mid[1], z_tip};
  double v1[3], v2[3];

  rt_base_velocity(f, t, p, v1);
  p[2] = z_tip + 0.5 * dt * v1[2];
  rt_base_velocity(f, t + 0.5 * dt, p, v2);

  return z_tip + dt * v2[2];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Rim constraint
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Which side walls a RIM vertex sits on. Call it for boundary vertices only.
 *
 * See the file header for why this is read off the position rather than off an
 * application bit, and why that is unambiguous for rim vertices. A box corner
 * answers both.
 */
static int wall_mask(const MFTL_Domain *dom, const double pos[3])
{
  int mask = 0;
  for (int d = 0; d < 2; ++d)
  {
    const double tol = RT_WALL_TOL * dom->domain_size[d];
    if (fabs(pos[d] - dom->global_min[d]) < tol || fabs(pos[d] - dom->global_max[d]) < tol) mask |= (d == 0) ? RT_WALL_X : RT_WALL_Y;
  }
  return mask;
}

/** @brief Distance of a rim vertex from the nearest side wall. What --verify bounds. */
static double wall_distance(const MFTL_Domain *dom, const double pos[3])
{
  double best = MFTL_DBL_MAX;
  for (int d = 0; d < 2; ++d)
  {
    best = MFTL_MIN(best, fabs(pos[d] - dom->global_min[d]));
    best = MFTL_MIN(best, fabs(pos[d] - dom->global_max[d]));
  }
  return best;
}

/**
 * @brief Snaps a constrained coordinate exactly back onto its wall, and clamps.
 *
 * The snap is what keeps MFTLGetOwnerProc honest: a rim vertex sitting exactly
 * on global_max needs only one ulp of round-off to leave the box. The clamp
 * covers the other route there -- an explicit integrator overshooting near a
 * wall where the true trajectory merely approaches it.
 */
static void snap_to_domain(const MFTL_Domain *dom, int mask, double pos[3])
{
  for (int d = 0; d < 2; ++d)
  {
    if (!(mask & ((d == 0) ? RT_WALL_X : RT_WALL_Y))) continue;
    pos[d] = (pos[d] < 0.5 * (dom->global_min[d] + dom->global_max[d])) ? dom->global_min[d] : dom->global_max[d];
  }

  for (int d = 0; d < 3; ++d)
  {
    if (!(pos[d] >= dom->global_min[d])) pos[d] = dom->global_min[d];
    if (!(pos[d] <= dom->global_max[d])) pos[d] = dom->global_max[d];
  }
}

/**
 * @brief Re-snaps every rim vertex in the pool -- owned AND ghost.
 *
 * Run after each remeshing pass. Ghosts too, and without communication: the snap
 * is a pure function of the position, and after the pass every ghost is a
 * bit-exact copy of its owner (the spanning driver closes with a plain
 * redistribution), so both copies land on the same value.
 */
static void snap_rim(MFTL_Ensemble *ens)
{
  const MFTL_Domain *dom = &ens->domain;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      if (!MFTLVertexIsMeshBoundary(obj, v)) continue;
      double *p = MFTLVertexPosMut(obj, v);
      snap_to_domain(dom, wall_mask(dom, p), p);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Advection
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Moves every vertex this rank owns, one step, with RK2 (midpoint).
 *
 * Second order because the whole point is the SHAPE: Euler in a strained field
 * of this strength visibly rounds off the mushroom cap. Only owned vertices
 * move; MFTLUpdateGhostPositions refreshes the copies.
 */
static void advect(MFTL_Ensemble *ens, const rt_field *f, double t, double dt, double z_tip, double z_tip_mid, bool pin_rim)
{
  const MFTL_Domain *dom = &ens->domain;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];

    MFTL_FOREACH_OWNED_VERTEX(obj, {
      const bool on_rim = (rec->flags & MFTL_FLAG_BOUNDARY) != 0;
      if (pin_rim && on_rim) continue;

      const int mask = on_rim ? wall_mask(dom, pos) : 0;
      double v1[3], mid[3], v2[3];

      rt_velocity(f, t, z_tip, pos, v1);
      if (mask & RT_WALL_X) v1[0] = 0.0;
      if (mask & RT_WALL_Y) v1[1] = 0.0;

      for (int d = 0; d < 3; ++d) mid[d] = pos[d] + 0.5 * dt * v1[d];
      snap_to_domain(dom, mask, mid);

      /* The ring has moved with the tip by the midpoint stage. */
      rt_velocity(f, t + 0.5 * dt, z_tip_mid, mid, v2);
      if (mask & RT_WALL_X) v2[0] = 0.0;
      if (mask & RT_WALL_Y) v2[1] = 0.0;

      for (int d = 0; d < 3; ++d) pos[d] += dt * v2[d];
      snap_to_domain(dom, mask, pos);
    });
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Diagnostics
 *
 * Everything over OWNED entities only, so every reduction is exact: a ghost is a
 * copy of a triangle another rank owns, and counting it would weight that
 * triangle twice by an amount that grows with the rank count.
 * ═══════════════════════════════════════════════════════════════════════════ */

enum
{
  S_EDGE_SUM,
  S_EDGE_N,
  S_Q_SUM,
  S_AREA,
  S_PERIMETER,
  S_VOLUME,  ///< volume BELOW the interface, see measure()
  S_TRIS,
  S_VERTS,
  S_RIM,  ///< owned mesh-boundary half-edges
  S_RIM_CUT,  ///< owned rim half-edges in triangles another rank holds a copy of
  S_HOLES,  ///< owned twin-less half-edges that are NOT mesh boundary: a short halo
  S_OFF_WALL,  ///< owned rim vertices on no side wall
  S_INVERTED,  ///< owned triangles facing against every edge neighbour: a fold-over
  S_CREASES,  ///< edges folded by more than 120 degrees: unresolved curvature
  S_N
};

enum
{
  X_MIN_EDGE,
  X_MIN_Q,
  X_MIN_Q_RIM,  ///< worst quality among triangles with a rim corner
  X_Z_MIN,
  X_NEG_MAX_EDGE,
  X_NEG_Z_MAX,
  X_NEG_RIM_DIST,  ///< -max distance of a rim vertex from its wall
  X_N
};

typedef struct
{
  double sum[S_N];
  double min[X_N];  ///< every entry is a MIN; maxima are stored negated
} rt_stats;

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
 * @brief Collective survey of the whole interface.
 *
 * The VOLUME BELOW THE INTERFACE is the physical check, and it needs no closing
 * surface: by the divergence theorem on the region between the bottom wall and
 * the interface, V = integral over the interface of (z - z_bottom) n_z dA. The
 * side walls contribute nothing (n_z = 0 there) and the bottom contributes
 * nothing (z - z_bottom = 0), and for a flat triangle the integrand is linear so
 * the one-point centroid rule is exact. A solenoidal, free-slip field conserves
 * V exactly; what drifts is the discretisation -- RK2, and the remesher's
 * midpoint splits and tangential smoothing -- so V/V0 is a direct measurement of
 * how much volume a remeshing pass costs on an open surface.
 *
 * Orientation is judged per triangle against its edge neighbours (fold_state):
 * an inverted triangle fails --verify, a crease is reported. Late in the default
 * run the vortex ring winds the mushroom cap tighter than one edge length and
 * creases appear there -- the imposed roll-up outrunning the resolution, not the
 * remesher folding anything.
 */
static void measure(MFTL_Ensemble *ens, rt_stats *g)
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

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double area = 0.0, n[3], c[3];
      MFTLGetTriangleProperties(obj, t, &area, n, c);

      double sum_sq = 0.0;
      for (int k = 0; k < 3; ++k)
      {
        const int h = 3 * t + k;
        const double len_sq = MFTLGetEdgeLengthSquared(obj, h);
        const double len = sqrt(len_sq);

        sum_sq += len_sq;
        s[S_EDGE_SUM] += len;
        s[S_EDGE_N] += 1.0;
        x[X_MIN_EDGE] = MFTL_MIN(x[X_MIN_EDGE], len);
        x[X_NEG_MAX_EDGE] = MFTL_MIN(x[X_NEG_MAX_EDGE], -len);

        /* The topological flag, not twin_idx: after redistribution a missing
         * twin also means "the neighbour is on another rank". */
        if (MFTLHeIsMeshBoundary(obj, h))
        {
          s[S_RIM] += 1.0;
          if (is_replicated(obj, t)) s[S_RIM_CUT] += 1.0;
          s[S_PERIMETER] += len;
        }
        else if (MFTLHeIsLocalBoundary(obj, h))
          s[S_HOLES] += 1.0;
      }

      int v[3];
      MFTLGetTriangleVertices(obj, t, v);

      const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
      x[X_MIN_Q] = MFTL_MIN(x[X_MIN_Q], q);
      if (MFTLVertexIsMeshBoundary(obj, v[0]) || MFTLVertexIsMeshBoundary(obj, v[1]) || MFTLVertexIsMeshBoundary(obj, v[2])) x[X_MIN_Q_RIM] = MFTL_MIN(x[X_MIN_Q_RIM], q);
      s[S_Q_SUM] += q;
      s[S_AREA] += area;
      s[S_VOLUME] += (c[2] - z_bottom) * n[2] * area;

      fold_state(obj, t, n, &s[S_INVERTED], &s[S_CREASES]);
    }

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      x[X_Z_MIN] = MFTL_MIN(x[X_Z_MIN], p[2]);
      x[X_NEG_Z_MAX] = MFTL_MIN(x[X_NEG_Z_MAX], -p[2]);

      if (!MFTLVertexIsMeshBoundary(obj, v)) continue;
      if (wall_mask(dom, p) == 0) s[S_OFF_WALL] += 1.0;
      x[X_NEG_RIM_DIST] = MFTL_MIN(x[X_NEG_RIM_DIST], -wall_distance(dom, p));
    }

    s[S_TRIS] += (double) obj->tris.n_local;
    s[S_VERTS] += (double) obj->verts.n_local;
  }

  MPI_Allreduce(s, g->sum, S_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(x, g->min, X_N, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
}

/** @brief V - E + F from owned counts, with E = (3F + B)/2: 1 for a disk. */
static double euler_of(const rt_stats *g) { return g->sum[S_VERTS] - 0.5 * (3.0 * g->sum[S_TRIS] + g->sum[S_RIM]) + g->sum[S_TRIS]; }

/**
 * @brief The collective verdict --verify acts on, identical on every rank.
 *
 * Structural statements first, because they are exact: the halo is complete in
 * the open-surface sense, the surface is still a disk, no rim vertex has left its
 * wall and no triangle is folded. Then the one physical statement, with a
 * tolerance: with the rim sliding, the volume below the interface is conserved.
 * Two percent is loose against what the loop actually loses -- it exists to fail
 * a remesher that eats volume at the rim, not to grade RK2.
 */
static bool verdict(const rt_stats *g, double v0, bool pin_rim, char *why, size_t why_len)
{
  why[0] = '\0';
  if (g->sum[S_HOLES] != 0.0) snprintf(why, why_len, "%.0f twin-less interior half-edges (halo incomplete)", g->sum[S_HOLES]);
  if (fabs(euler_of(g) - 1.0) > 0.5) snprintf(why, why_len, "V-E+F = %.0f, the surface is no longer a disk", euler_of(g));
  if (g->sum[S_OFF_WALL] != 0.0) snprintf(why, why_len, "%.0f rim vertices left their wall", g->sum[S_OFF_WALL]);
  if (g->sum[S_INVERTED] != 0.0) snprintf(why, why_len, "%.0f inverted triangles", g->sum[S_INVERTED]);
  if (!pin_rim && fabs(g->sum[S_VOLUME] / v0 - 1.0) > 0.02) snprintf(why, why_len, "volume below the interface drifted to V/V0 = %.5f", g->sum[S_VOLUME] / v0);
  return why[0] == '\0';
}

static void print_header(bool remeshing)
{
  printf("\n  %5s %6s | %7s %7s | %7s | %8s %8s %8s | %7s %6s | %7s %8s | %3s %4s", "step", "t", "spike", "bubble", "ntri", "edge min", "mean", "max",
         "q worst", "mean", "area", "V/V0", "inv", "crs");
  if (remeshing) printf(" | %6s %6s %6s %3s %4s", "split", "coll", "flip", "ph", "str");
  printf("\n  ");
  for (int i = 0; i < (remeshing ? 149 : 117); ++i) putchar('-');
  printf("\n");
}

typedef struct
{
  long split, collapse, flip;  ///< global, since the last report
  int phases;  ///< of the last pass
  MFTL_RemeshParallelStrategy strategy;
  bool any;
  bool serial;  ///< nproc == 1: MFTLRemeshEnsemble took the serial path, which has no strategy
} remesh_window;

static void print_stats(int step, double t, const rt_stats *g, double v0, const remesh_window *w, bool remeshing)
{
  const double mean_edge = (g->sum[S_EDGE_N] > 0.0) ? g->sum[S_EDGE_SUM] / g->sum[S_EDGE_N] : 0.0;
  const double mean_q = (g->sum[S_TRIS] > 0.0) ? g->sum[S_Q_SUM] / g->sum[S_TRIS] : 0.0;

  printf("  %5d %6.2f | %7.3f %7.3f | %7.0f | %8.5f %8.5f %8.5f | %7.4f %6.4f | %7.4f %8.5f | %3.0f %4.0f", step, t, g->min[X_Z_MIN], -g->min[X_NEG_Z_MAX],
         g->sum[S_TRIS], g->min[X_MIN_EDGE], mean_edge, -g->min[X_NEG_MAX_EDGE], g->min[X_MIN_Q], mean_q, g->sum[S_AREA], g->sum[S_VOLUME] / v0,
         g->sum[S_INVERTED], g->sum[S_CREASES]);
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

/** @brief The horizontal patch spanning the box at mid-height, as an indexed list. */
static int build_interface(MFTL_Object *obj, const MFTL_Domain *dom, const rt_field *f, int res, double perturb)
{
  const int n = res + 1;
  const int n_verts = n * n;
  const int n_tris = 2 * res * res;

  double *pos = (double *) malloc(3 * (size_t) n_verts * sizeof(double));
  int *tris = (int *) malloc(3 * (size_t) n_tris * sizeof(int));
  if (!pos || !tris)
  {
    free(pos);
    free(tris);
    return MFTL_ERROR;
  }

  const double z_mid = 0.5 * (dom->global_min[2] + dom->global_max[2]);
  for (int j = 0; j < n; ++j)
  {
    for (int i = 0; i < n; ++i)
    {
      double *p = &pos[3 * (j * n + i)];

      /* The end points are written as the bounds themselves, not as
       * min + res*dx: the rim must lie on the walls to the bit. */
      p[0] = (i == res) ? dom->global_max[0] : dom->global_min[0] + (double) i * dom->domain_size[0] / (double) res;
      p[1] = (j == res) ? dom->global_max[1] : dom->global_min[1] + (double) j * dom->domain_size[1] / (double) res;

      /* Seed ripple in phase with the driving mode, so the interface already dips
       * where the spike will form. Cosmetic with a PRESCRIBED field -- the mode is
       * set by the field and this is simply advected along -- but it is the seed
       * a real RT solver would grow. */
      p[2] = z_mid - perturb * cos(f->modes[0].kx * (p[0] - f->mid[0])) * cos(f->modes[0].ky * (p[1] - f->mid[1]));
    }
  }

  /* (v00, v10, v11) and (v00, v11, v01) with x along u and y along v: the normal
   * points UP, into the heavy fluid. */
  int w = 0;
  for (int j = 0; j < res; ++j)
  {
    for (int i = 0; i < res; ++i)
    {
      const int v00 = j * n + i;
      const int v10 = v00 + 1;
      const int v01 = v00 + n;
      const int v11 = v01 + 1;

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

  /* A tall box, as RT boxes are. Solid on all six sides -- the field is
   * free-slip there by construction, so periodicity would add nothing but a
   * seam to reason about. */
  double global_min[3] = {-0.5, -0.5, -1.5};
  double global_max[3] = {0.5, 0.5, 1.5};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, global_min, global_max, periodic, 1);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  rt_field field;
  rt_field_init(&field, &ens->domain, &cfg);

  /* ── Generate: rank 0 builds, one publication settles ids and counts ────── */
  MFTL_Object *interface = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, interface);

  if (rank == 0)
  {
    if (build_interface(interface, &ens->domain, &field, cfg.res, cfg.perturb) != MFTL_SUCCESS)
    {
      fprintf(stderr, "Failed to build the interface.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MFTLCheckMeshIntegrity(interface);
  }
  else
    MFTLObjectBuildDerived(interface);
  MFTLPublishGeneratedBodies(ens);

  /* Scatter. `interface` is dead after this line: the handle that survives a
   * redistribution is the body id, so everything below walks the local list. */
  MFTLComputeTriangleGeometry(interface, interface->tris.n);
  MFTLRedistribute(ens);
  interface = MFTL_NULL;

  /* The target is the mesh's own starting resolution, measured COLLECTIVELY
   * (MFTLMeanEdgeLength on a partition would give every rank a different band). */
  const double target = (cfg.target > 0.0) ? cfg.target : MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  /* Feature detection off: the interface is smooth, and every edge the detector
   * flagged would be a false positive that locks a vertex the smoother needs.
   * The RIM is a feature regardless of the threshold -- a mesh boundary always
   * is -- and its four box corners classify as CORNER through the turn angle. */
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, cfg.iterations, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = cfg.smooth;
  params.parallel_strategy = cfg.strategy;
  params.parallel_phase_budget = cfg.budget;

  const bool remeshing = cfg.remesh_every > 0;
  const double h_min = MFTLDomainGetHMin(&ens->domain);

  rt_stats g0;
  measure(ens, &g0);
  const double v0 = g0.sum[S_VOLUME];
  const double area0 = g0.sum[S_AREA];
  const double max_edge0 = -g0.min[X_NEG_MAX_EDGE];

  if (rank == 0)
  {
    printf("=== Rayleigh-Taylor emulation on an open surface ===\n");
    printf("  %d rank(s), grid %dx%dx%d | box [%.1f,%.1f]x[%.1f,%.1f]x[%.1f,%.1f]\n", nproc, cfg.cuts[0], cfg.cuts[1], cfg.cuts[2], global_min[0],
           global_max[0], global_min[1], global_max[1], global_min[2], global_max[2]);
    printf("  interface: %d x %d patch -> %.0f triangles, %.0f rim half-edges, V-E+F = %.0f (a disk)\n", cfg.res, cfg.res, g0.sum[S_TRIS], g0.sum[S_RIM],
           euler_of(&g0));
    printf("  field: %s, mode (%d,%d), v_max %.3f, growth %.2f, asym %.2f, roll %.2f%s\n", cfg.multimode ? "two modes" : "single mode", cfg.mx, cfg.my,
           cfg.vmax, cfg.growth, cfg.asym, cfg.roll, (cfg.roll > 0.0) ? "" : " (no mushroom cap)");
    printf("  rim %s | %d steps of dt %.3f\n", cfg.pin_rim ? "PINNED" : "sliding on the side walls", cfg.steps, cfg.dt);
    if (remeshing)
    {
      printf("  remesh every %d step(s): target %.5f (band %.5f .. %.5f), %d iterations, %d smoothing sweeps, budget %d\n", cfg.remesh_every, target,
             params.min_edge_length, params.max_edge_length, cfg.iterations, cfg.smooth, cfg.budget);
      if (nproc > 1)
        printf("  band precondition h_min > 4*3*max_edge: %.4f %s %.4f%s\n", h_min, (h_min > 12.0 * params.max_edge_length) ? ">" : "<=",
             12.0 * params.max_edge_length, (h_min > 12.0 * params.max_edge_length) ? "" : "  (the armed coarsening fallback decides)");
    }
    else
      printf("  remeshing OFF: fixed connectivity, the degradation baseline\n");
    print_header(remeshing);
    remesh_window none = {0, 0, 0, 0, MFTL_REMESH_PARALLEL_SPANNING, MFTL_FALSE, MFTL_FALSE};
    print_stats(0, 0.0, &g0, v0, &none, remeshing);
  }

  if (cfg.vtk_every > 0)
  {
    char prefix[256];
    snprintf(prefix, sizeof prefix, "%s_%04d", cfg.out, 0);
    MFTLWriteVtu(ens, prefix, MFTL_NULL);
  }

  /* ── The per-timestep loop ─────────────────────────────────────────────── */
  const double z_mid = 0.5 * (global_min[2] + global_max[2]);
  double t = 0.0;
  double z_tip = z_mid - cfg.perturb;

  remesh_window win = {0, 0, 0, 0, MFTL_REMESH_PARALLEL_SPANNING, MFTL_FALSE, MFTL_FALSE};
  long total_ops = 0;
  int n_passes = 0, n_not_converged = 0, broken = 0;
  double rim_cut_max = 0.0;
  double t_remesh = 0.0;
  rt_stats g = g0;

  for (int step = 1; step <= cfg.steps; ++step)
  {
    /* 1. move only what this rank owns */
    const double z_tip_mid = advance_tip(&field, z_tip, t, 0.5 * cfg.dt);
    advect(ens, &field, t, cfg.dt, z_tip, z_tip_mid, cfg.pin_rim);
    z_tip = advance_tip(&field, z_tip, t, cfg.dt);
    t += cfg.dt;

    /* 2. pull in the vertices the neighbours moved */
    MFTLUpdateGhostPositions(ens);

    /* 3. ALL triangles, ghosts included: both the remesher's first migration and
     *    a plain redistribution route ghost triangles by their cached centre */
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);

    /* 4. remesh (which re-partitions on the way out), or just re-partition */
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

      /* The rim, back onto its walls to the bit (a no-op unless round-off). */
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

    const bool report_now = cfg.report_every > 0 && (step % cfg.report_every == 0 || step == cfg.steps);
    if (report_now)
    {
      measure(ens, &g);
      if (rank == 0) print_stats(step, t, &g, v0, &win, remeshing);
      memset(&win, 0, sizeof win);

      rim_cut_max = MFTL_MAX(rim_cut_max, g.sum[S_RIM_CUT]);

      char why[160];
      if (!verdict(&g, v0, cfg.pin_rim, why, sizeof why))
      {
        broken++;
        if (rank == 0) fprintf(stderr, "  step %d: %s\n", step, why);
      }
    }

    if (cfg.vtk_every > 0 && step % cfg.vtk_every == 0)
    {
      char prefix[256];
      snprintf(prefix, sizeof prefix, "%s_%04d", cfg.out, step);
      MFTLWriteVtu(ens, prefix, MFTL_NULL);
    }
  }

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

  /* ── Closing summary ────────────────────────────────────────────────────── */
  measure(ens, &g);
  double t_max = 0.0;
  MPI_Allreduce(&t_remesh, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
  {
    printf("\n  interface area   %.6f -> %.6f  (%.2fx)\n", area0, g.sum[S_AREA], g.sum[S_AREA] / area0);
    printf("  triangles        %.0f -> %.0f\n", g0.sum[S_TRIS], g.sum[S_TRIS]);
    printf("  longest edge     %.6f -> %.6f  (%.2fx)\n", max_edge0, -g.min[X_NEG_MAX_EDGE], -g.min[X_NEG_MAX_EDGE] / max_edge0);
    printf("  worst quality    %.4f -> %.4f   (triangles touching the rim: %.4f -> %.4f)\n", g0.min[X_MIN_Q], g.min[X_MIN_Q], g0.min[X_MIN_Q_RIM], g.min[X_MIN_Q_RIM]);
    printf("  rim              %.0f -> %.0f half-edges, perimeter %.6f -> %.6f, max distance from its wall %.2e\n", g0.sum[S_RIM], g.sum[S_RIM],
           g0.sum[S_PERIMETER], g.sum[S_PERIMETER], -g.min[X_NEG_RIM_DIST]);
    printf("  volume below     V/V0 = %.6f%s\n", g.sum[S_VOLUME] / v0, cfg.pin_rim ? "  (not an invariant with a pinned rim)" : "");
    printf("  Euler V-E+F      %.0f   (1 for a disk)\n", euler_of(&g));
    printf("  orientation      %.0f inverted triangles, %.0f creases sharper than 120 deg (curvature below the mesh resolution)\n", g.sum[S_INVERTED],
           g.sum[S_CREASES]);
    if (remeshing)
    {
      printf("  remeshing        %d passes, %ld operations, %d not converged (expected at a small budget), %.3f s (max over ranks)\n", n_passes, total_ops,
             n_not_converged, t_max);
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
