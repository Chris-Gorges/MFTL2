/*
 * @Author: Christian Gorges
 * @Date: 2026-09-24
 *
 * @brief A liquid jet entering the domain continuously through a nozzle, with a
 *        Rayleigh-Plateau perturbation that grows until the jet pinches off --
 *        an open surface that GROWS, remeshed in parallel every few steps.
 *
 * A jet of radius R0 issues from a circular nozzle in the x_min wall and runs
 * along +x at speed U. Its surface is one open mesh: a tube attached to the
 * nozzle lip, closed at the far end by the jet head. The only boundary is the
 * nozzle rim, which is PINNED -- the lip does not move -- while every other
 * vertex travels downstream with the liquid. So the band of triangles next to
 * the lip is stretched at rate U, forever, and the only thing that keeps the
 * mesh intact is the remesher splitting it: new surface enters the domain the
 * way it does in a front-tracking solver, through the remesher at an inflow
 * boundary rather than through a generator.
 *
 * A forced sausage perturbation rides on the jet and grows as it is convected
 * downstream, by the Rayleigh-Plateau instability's growth rate. The necks thin,
 * the swells grow, and eventually a neck is thinner than the mesh can resolve.
 * That is where a real jet breaks up into drops, and it is a TOPOLOGY CHANGE --
 * cutting the ligament and closing both ends -- which this library does not do
 * yet. So the run stops there and reports where and when the jet broke, against
 * Rayleigh's linear estimate of the breakup length. It is the example the
 * topology-change work will be driven by.
 *
 * ── What this is NOT ───────────────────────────────────────────────────────
 *
 * There is no momentum equation and no surface tension. The field is PRESCRIBED
 * -- a kinematic caricature in the spirit of rayleigh_taylor_open_surface.c --
 * and its growth rate is Rayleigh's LINEAR one, applied as though the jet stayed
 * nearly cylindrical. There are no satellite drops, no end-pinching and no
 * nonlinear neck dynamics; a simulated jet would pinch differently and at a
 * different place. What it gets right is what the mesh has to survive: a
 * surface growing through an inflow boundary, a travelling pattern of necks and
 * swells, a neck whose circumference shrinks towards a handful of edges, and a
 * swell that stretches.
 *
 * ── The velocity field ─────────────────────────────────────────────────────
 *
 * Lab frame, x measured from the nozzle, r the distance from the jet axis,
 * xi = x - U t the phase of the convected pattern, phi a fixed forcing phase:
 *
 *     u = U e_x + u'      with      psi' = B(x,t) * g(r)
 *
 *     u'_x =  (1/r) dpsi'/dr = (g'(r)/r) B
 *     u'_r = -(1/r) dpsi'/dx = -(g(r)/r) dB/dx
 *
 *     B(x,t) = -b(x) sin(k xi + phi),   b(x) = sigma * eps(x) * R0 / (k g(R0))
 *
 *     g(r)   = h_sat tanh(h(r)/h_sat),  h(r) = r I1(k r),  h_sat = h(1.5 R0)
 *
 * Four things about it, each load-bearing:
 *
 *   1. It is written as an axisymmetric STOKES STREAM FUNCTION, so it is exactly
 *      divergence-free for ANY B(x,t) and ANY g(r) -- including the x-dependent
 *      amplitude and the radial saturation below, either of which would spoil a
 *      velocity potential. Divergence-free means the jet's volume changes only
 *      by what enters through the nozzle, and that is the physical check this
 *      example makes: V(t) = V0 + A_nozzle * U * t.
 *
 *   2. Inside the jet h << h_sat, g = r I1(k r), and u' is the potential flow of
 *      Rayleigh's sausage mode (I0 and I1 are the modified Bessel functions its
 *      eigenmode is made of). At r = R0 the radial velocity is exactly
 *      sigma * eps(x) * cos(k xi + phi), so a surface point convected at U sees
 *      its displacement grow like e^{sigma t}: the TEMPORAL instability in the
 *      frame of the liquid, which in the lab is a SPATIALLY growing, travelling
 *      wave -- the classical picture of a jet forced at the nozzle. (The dB/dx
 *      term adds a phase shift of order sigma/(kU), a few percent.)
 *
 *   3. OUTSIDE the jet the linear eigenmode is the wrong answer, and a dangerous
 *      one: I1 grows exponentially in r, so a swell that has already doubled its
 *      radius is pushed outward faster still. With the pure Bessel profile the
 *      head flattened into a disc that reached the box corners, r = 7 R0, before
 *      the first neck pinched. The tanh caps g, so beyond ~1.5 R0 the radial
 *      flow decays like 1/r (a line source) and swells grow only slowly past
 *      about drop size -- one wavelength of jet makes a drop of ~1.9 R0.
 *
 *   4. eps(x) is the amplitude a liquid element has reached after travelling x
 *      from the nozzle: eps0 * e^{sigma x/U}, saturated LOGISTICALLY at R0 so the
 *      field stays bounded downstream of the breakup point, where the linear
 *      theory has nothing to say anyway. phi puts the initial head on a swell
 *      crest, so it grows into the bulbous leading drop a starting jet has; on a
 *      neck phase the field instead draws the head out into a long filament of
 *      almost constant radius, and "pinch-off" becomes a dip in that filament.
 *
 * sigma is Rayleigh's inviscid growth rate, in units of U/R0:
 *
 *     sigma^2 = x I1(x) (1 - x^2) / (I0(x) We),     x = k R0,  We = rho U^2 R0 / gamma
 *
 * positive for x < 1 (longer than the circumference: unstable), fastest at
 * x = 0.697, where sigma = 0.343 / sqrt(We). Rayleigh's breakup length follows
 * as L_b = (U / sigma) ln(R0 / eps0).
 *
 * The neck thins more slowly in this field than the linear R0 - eps would say:
 * u'_r carries I1(k r), which falls off towards the axis, so a neck at radius r
 * shrinks at a rate proportional to r and approaches zero exponentially rather
 * than in finite time. Pinch-off therefore happens DOWNSTREAM of L_b -- measured
 * at the defaults, the ligament behind the leading drop reaches 0.25 R0 at
 * x = 26.2 .. 26.8 R0 (np = 1, 2, 4) against L_b = 20.4 R0. That is a property of
 * the caricature, and a benign one: the neck still becomes thinner than any mesh
 * resolves.
 *
 * ── The mesh ───────────────────────────────────────────────────────────────
 *
 * Built here as an indexed triangle list and handed to MFTLBuildIndexedMesh --
 * the library has no open-surface generator, and needs none. Rings of vertices
 * along the axis, every other ring rotated by half a sector so the triangles are
 * near-equilateral, then a hemispherical head whose rings carry fewer vertices
 * the closer they are to the pole, stitched by a zipper that walks two rings of
 * different counts in angle order. Topologically a DISK: one boundary loop (the
 * nozzle lip), V - E + F = 1.
 *
 * ── The nozzle rim ─────────────────────────────────────────────────────────
 *
 * Pinned in the advection step. It is a feature curve to the remesher -- a mesh
 * boundary always is -- and the ridge smoother may slide a lip vertex ALONG the
 * rim to even out the spacing; it does so along the chord of its two rim
 * neighbours, which leaves it slightly inside the circle. So after every remesh
 * each rim vertex, owned and ghost alike, is projected back onto the nozzle
 * circle: a pure function of the position applied to bit-identical copies, so no
 * message is needed. --verify checks the lip never left the circle.
 *
 * ── Which strategy runs ────────────────────────────────────────────────────
 *
 * MFTLRemeshEnsemble measures the body every pass. On the default (P,1,1) grid
 * a short jet is smaller than a rank box, so it starts DISPERSED -- the whole jet
 * is moved to one rank and remeshed there serially, bit-identically to np = 1 --
 * and switches to SPANNING once it is longer than 0.4 of a box; the `str` column
 * shows the switch. On a grid that cuts across the axis (2 2 1) it is SPANNING
 * from the first pass, and the lip itself is shared between ranks.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P liquid_jet_open_surface [cx cy cz] [options]
 *
 *     cx cy cz          process grid, cx*cy*cz == P. Defaults to (P,1,1).
 *
 *     --length L        domain length in R0              (default 40)
 *     --width W         domain width in R0               (default 10)
 *     --stub L0         initial jet length in R0         (default 4)
 *     --sectors N       vertices around the jet          (default 32)
 *     --weber We        Weber number rho U^2 R0 / gamma  (default 4)
 *     --kr X            forced wavenumber k R0, (0, 1)   (default 0.697, the
 *                       fastest-growing mode)
 *     --eps X           perturbation at the nozzle, eps0/R0 (default 0.03;
 *                       0 = unperturbed, the jet never breaks)
 *     --pinch X         neck radius counted as pinch-off, in R0 (default 0.25;
 *                       never below one target edge length, which is where the
 *                       mesh stops resolving the neck)
 *     --steps N         stop after N steps; 0 = run until pinch-off or until the
 *                       head reaches the outlet (default 0)
 *     --dt X            timestep, in R0/U                (default 0.1)
 *
 *     --remesh-every N  remesh every N steps; 0 = never (the lip band then
 *                       stretches without bound)         (default 2)
 *     --target X        target edge length               (default: initial mean)
 *     --iterations N    remeshing iterations per pass    (default 3)
 *     --smooth N        smoothing sweeps per iteration   (default 3)
 *     --budget N        cut-plane placements per pass    (default 4)
 *     --strategy S      auto | spanning | dispersed      (default auto)
 *
 *     --vtk N           write a .pvtu every N steps      (default 20, 0 off)
 *     --report N        print a line every N steps       (default 20)
 *     --out PREFIX      output file prefix               (default "jet")
 *     --verify          exit non-zero unless halo, lip, Euler characteristic,
 *                       orientation and volume balance hold at every report
 *     --require-rim-cut --verify, and fail if at np > 1 no lip triangle ever met
 *                       a partition boundary. Never true on the default
 *                       (P,1,1) grid, where the lip sits inside rank 0's box;
 *                       the CTest runs 2 2 1, which puts a cut plane through
 *                       the axis
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/** @brief How far a lip vertex may sit from the nozzle circle, relative to R0. */
#define JET_LIP_TOL 1e-9

/** @brief The jet's nozzle radius and speed are the units of length and velocity. */
#define R0 1.0
#define U0 1.0

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Configuration
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int cuts[3];
  double length;
  double width;
  double stub;
  int sectors;
  double weber;
  double kr;
  double eps;
  double pinch;
  int steps;
  double dt;

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

static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  cfg->cuts[0] = nproc;
  cfg->cuts[1] = 1;
  cfg->cuts[2] = 1;
  cfg->length = 40.0;
  cfg->width = 10.0;
  cfg->stub = 4.0;
  cfg->sectors = 32;
  cfg->weber = 4.0;
  cfg->kr = 0.697;
  cfg->eps = 0.03;
  cfg->pinch = 0.25;
  cfg->steps = 0;
  cfg->dt = 0.1;
  cfg->remesh_every = 2;
  cfg->target = 0.0;
  cfg->iterations = 3;
  cfg->smooth = 3;
  cfg->budget = 4;
  cfg->strategy = MFTL_REMESH_PARALLEL_AUTO;
  cfg->vtk_every = 20;
  cfg->report_every = 20;
  cfg->verify = MFTL_FALSE;
  cfg->require_rim_cut = MFTL_FALSE;
  snprintf(cfg->out, sizeof cfg->out, "jet");

  int i = 1;
  if (argc >= 4 && argv[1][0] != '-')
  {
    for (int d = 0; d < 3; ++d) cfg->cuts[d] = atoi(argv[1 + d]);
    i = 4;
  }

  for (; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--verify"))
      cfg->verify = MFTL_TRUE;
    else if (!strcmp(argv[i], "--require-rim-cut"))
      cfg->verify = cfg->require_rim_cut = MFTL_TRUE;
    else if (!strcmp(argv[i], "--length") && i + 1 < argc)
      cfg->length = atof(argv[++i]);
    else if (!strcmp(argv[i], "--width") && i + 1 < argc)
      cfg->width = atof(argv[++i]);
    else if (!strcmp(argv[i], "--stub") && i + 1 < argc)
      cfg->stub = atof(argv[++i]);
    else if (!strcmp(argv[i], "--sectors") && i + 1 < argc)
      cfg->sectors = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--weber") && i + 1 < argc)
      cfg->weber = atof(argv[++i]);
    else if (!strcmp(argv[i], "--kr") && i + 1 < argc)
      cfg->kr = atof(argv[++i]);
    else if (!strcmp(argv[i], "--eps") && i + 1 < argc)
      cfg->eps = atof(argv[++i]);
    else if (!strcmp(argv[i], "--pinch") && i + 1 < argc)
      cfg->pinch = atof(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
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
  if (cfg->sectors < 6 || cfg->stub <= 0.0 || cfg->steps < 0 || cfg->dt <= 0.0 || cfg->remesh_every < 0 || cfg->iterations < 1 || cfg->smooth < 0)
  {
    if (rank == 0) fprintf(stderr, "Need sectors >= 6, stub > 0, steps >= 0, dt > 0, remesh-every >= 0, iterations >= 1, smooth >= 0.\n");
    return MFTL_ERROR;
  }
  if (cfg->kr <= 0.0 || cfg->kr >= 1.0 || cfg->weber <= 0.0 || cfg->eps < 0.0 || cfg->eps >= R0)
  {
    /* k R0 >= 1 is a STABLE mode -- shorter than the circumference, surface
     * tension restores it -- and the growth rate below would be imaginary. */
    if (rank == 0) fprintf(stderr, "Need 0 < kr < 1 (k R0 >= 1 is stable), weber > 0 and 0 <= eps < 1.\n");
    return MFTL_ERROR;
  }
  if (cfg->length < cfg->stub + 4.0 * R0 || cfg->width < 4.0 * R0)
  {
    if (rank == 0) fprintf(stderr, "The domain must be longer than the stub plus 4 R0 and at least 4 R0 wide.\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The Rayleigh-Plateau field
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Modified Bessel function I0, by its power series. Exact to round-off for |z| < 20. */
static double bessel_i0(double z)
{
  const double q = 0.25 * z * z;
  double term = 1.0, sum = 1.0;
  for (int m = 1; m < 60 && term > 1e-17 * sum; ++m)
  {
    term *= q / ((double) m * (double) m);
    sum += term;
  }
  return sum;
}

/**
 * @brief I1(z)/z, by its power series: finite at z = 0, where it is 1/2.
 *
 * The ratio rather than I1 itself because the Cartesian radial components are
 * u'_r * y / r, and I1(k r)/r is the factor that makes them regular on the axis.
 */
static double bessel_i1_over_z(double z)
{
  const double q = 0.25 * z * z;
  double term = 0.5, sum = 0.5;
  for (int m = 1; m < 60 && term > 1e-17 * sum; ++m)
  {
    term *= q / ((double) m * (double) (m + 1));
    sum += term;
  }
  return sum;
}

typedef struct
{
  double x_nozzle;  ///< the nozzle plane, global_min[0]
  double k;  ///< forced wavenumber
  double sigma;  ///< Rayleigh's growth rate at k, in U/R0
  double eps0;  ///< perturbation amplitude at the nozzle
  double eps_max;  ///< logistic saturation of the amplitude
  double h_sat;  ///< h(r_sat) = r_sat I1(k r_sat): where the radial profile saturates
  double g_r0;  ///< g(R0), which normalises the amplitude so u'_r(R0) is exactly sigma*eps
  double phase0;  ///< forcing phase: puts the initial head on a swell crest
} jet_field;

static void jet_field_init(jet_field *f, const MFTL_Domain *dom, const config *cfg)
{
  f->x_nozzle = dom->global_min[0];
  f->k = cfg->kr / R0;
  f->eps0 = cfg->eps * R0;
  f->eps_max = R0;

  /* The forcing phase is arbitrary at a real nozzle; here it is chosen so the
   * head starts on a SWELL crest. The head then grows into the bulbous leading
   * drop a starting jet actually has, and the first neck forms behind it. With
   * the head on a neck phase instead, the field draws it out into a long
   * filament of nearly constant radius, and "pinch-off" becomes a dip in that
   * filament -- measured, and not the phenomenon this example is about. */
  f->phase0 = -f->k * (cfg->stub + R0);

  const double x = cfg->kr;
  const double i1 = x * bessel_i1_over_z(x);
  f->sigma = (U0 / R0) * sqrt(x * i1 * (1.0 - x * x) / (bessel_i0(x) * cfg->weber));

  /* The radial profile saturates at 1.5 R0, below the ~1.9 R0 of the drop one
   * wavelength of jet makes: see jet_velocity. */
  const double r_sat = 1.5 * R0;
  f->h_sat = r_sat * (f->k * r_sat) * bessel_i1_over_z(f->k * r_sat);
  f->g_r0 = f->h_sat * tanh(R0 * i1 / f->h_sat);
}

/**
 * @brief The amplitude a liquid element has reached x downstream of the nozzle,
 *        and its derivative.
 *
 * Logistic: eps0 * e^{sigma x/U} while small, saturating at eps_max, with
 * d eps/dx = (sigma/U) eps (1 - eps/eps_max). Both are needed -- the derivative is
 * the dB/dx part of u'_r.
 */
static void jet_amplitude(const jet_field *f, double x, double *eps, double *deps_dx)
{
  if (f->eps0 <= 0.0)
  {
    *eps = *deps_dx = 0.0;
    return;
  }
  const double s = f->sigma * MFTL_MAX(x, 0.0) / U0;
  const double e = f->eps_max / (1.0 + (f->eps_max / f->eps0 - 1.0) * exp(-s));
  *eps = e;
  *deps_dx = (f->sigma / U0) * e * (1.0 - e / f->eps_max);
}

/**
 * @brief The full field at a point. Analytic and local: no mesh access, no
 *        communication.
 *
 * The perturbation is psi' = B(x,t) g(r), divergence-free for ANY g, with
 *
 *     g(r) = h_sat tanh(h(r) / h_sat),       h(r) = r I1(k r)
 *
 * so that
 *
 *     u'_r = -(g(r)/r) dB/dx,                u'_x = (g'(r)/r) B
 *     g'(r)/r = k I0(k r) sech^2(h / h_sat)
 *
 * Inside the jet h << h_sat and this is Rayleigh's eigenmode, g = r I1(k r).
 * Outside it the linear mode is the wrong answer, and a dangerous one: I1 grows
 * EXPONENTIALLY in r, so a swell that has already doubled its radius is pushed
 * outward faster still, and a swelling head flattens into a pancake that reaches
 * the walls -- measured, with the pure Bessel profile, at 7 R0 before the first
 * neck pinched. The tanh caps g at h_sat, beyond which the radial flow decays
 * like 1/r (a line source, still exactly solenoidal) and the swells grow only
 * slowly past the saturation radius. The amplitude is normalised with g(R0)
 * rather than R0 I1(k R0), so the growth rate AT THE JET SURFACE is still
 * exactly sigma.
 */
static void jet_velocity(const jet_field *f, double t, const double p[3], double out[3])
{
  const double x = p[0] - f->x_nozzle;
  const double r2 = p[1] * p[1] + p[2] * p[2];
  const double kr = f->k * sqrt(r2);

  double eps, deps;
  jet_amplitude(f, x, &eps, &deps);

  const double phase = f->k * (x - U0 * t) + f->phase0;
  const double s = sin(phase), c = cos(phase);

  /* B = -b sin(k xi + phi), with b fixed by u'_r(R0) = sigma eps cos. */
  const double b = f->sigma * eps * R0 / (f->k * f->g_r0);
  const double db = f->sigma * deps * R0 / (f->k * f->g_r0);

  /* g(r)/r^2 = k (I1(kr)/(kr)) * tanh(q)/q with q = h/h_sat: finite on the axis,
   * which is what the Cartesian components u'_r * (y, z)/r need. */
  const double i1_over_z = bessel_i1_over_z(kr);
  const double q = f->k * r2 * i1_over_z / f->h_sat;  // h / h_sat, since h = r I1(kr) = k r^2 (I1(kr)/(kr))
  const double tanhc = (q > 1e-8) ? tanh(q) / q : 1.0 - q * q / 3.0;
  const double sech = 1.0 / cosh(q);

  const double ur_over_r = f->k * i1_over_z * tanhc * (db * s + b * f->k * c);

  out[0] = U0 - b * f->k * bessel_i0(kr) * sech * sech * s;
  out[1] = ur_over_r * p[1];
  out[2] = ur_over_r * p[2];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The mesh: tube + hemispherical head, as an indexed triangle list
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One ring: `n` vertices from `first`, at angles phase + 2 pi i / n. */
typedef struct
{
  int first;
  int n;
  double phase;
} ring;

/**
 * @brief Triangulates the band between ring @p a and the next ring @p b towards
 *        the head, whatever their vertex counts.
 *
 * Walks both rings in angle order and always advances the one whose next vertex
 * comes first, so every triangle joins two consecutive vertices of one ring to
 * the nearest of the other: na + nb triangles, none degenerate. Both kinds are
 * wound so that (v1-v0) x (v2-v0) points AWAY from the axis -- around first, then
 * along, which is e_phi x e_x = e_r:
 *
 *     advance a:   (a_i, a_i+1, b_j)
 *     advance b:   (b_j, a_i,   b_j+1)
 */
static int zip_rings(const ring *a, const ring *b, int *tris, int w)
{
  const double da = MFTL_TWO_PI / (double) a->n;
  const double db = MFTL_TWO_PI / (double) b->n;

  /* Start b at its first vertex at or after a's first, so the band opens with
   * the spoke (a_0, b_s) and closes on the same spoke after one lap. */
  const int shift = (int) ceil((a->phase - b->phase) / db - 1e-9);
  const double b0 = b->phase + db * (double) shift;

#define A_IDX(i) (a->first + (i) % a->n)
#define B_IDX(j) (b->first + ((((shift + (j)) % b->n) + b->n) % b->n))

  int i = 0, j = 0;
  while (i < a->n || j < b->n)
  {
    const double next_a = a->phase + da * (double) (i + 1);
    const double next_b = b0 + db * (double) (j + 1);

    if (j == b->n || (i < a->n && next_a < next_b))
    {
      tris[w++] = A_IDX(i);
      tris[w++] = A_IDX(i + 1);
      tris[w++] = B_IDX(j);
      ++i;
    }
    else
    {
      tris[w++] = B_IDX(j);
      tris[w++] = A_IDX(i);
      tris[w++] = B_IDX(j + 1);
      ++j;
    }
  }

#undef A_IDX
#undef B_IDX
  return w;
}

/**
 * @brief Radius of the initial jet at x: R0 at the lip, and the displacement the
 *        field would have grown by the time a liquid element got here.
 */
static double initial_radius(const jet_field *f, double x)
{
  double eps, deps;
  jet_amplitude(f, x, &eps, &deps);
  return R0 + (eps - f->eps0) * cos(f->k * x + f->phase0);
}

/**
 * @brief The jet at t = 0: a tube of length @p stub from the nozzle, closed by a
 *        hemispherical head.
 *
 * Ring spacing along the axis is sqrt(3)/2 of the circumferential spacing and
 * every other ring is rotated by half a sector, which makes the tube's triangles
 * equilateral. The head's rings carry round(sectors * sin(theta)) vertices, so
 * the circumferential spacing stays the same all the way to the pole.
 */
static int build_jet(MFTL_Object *obj, const MFTL_Domain *dom, const jet_field *f, double stub, int sectors)
{
  const double edge = MFTL_TWO_PI * R0 / (double) sectors;
  const double dx_ring = 0.5 * sqrt(3.0) * edge;
  const double x0 = dom->global_min[0];

  const int n_tube = MFTL_MAX(1, (int) lround(stub / dx_ring));
  const double r_end = initial_radius(f, stub);
  const int n_cap = MFTL_MAX(2, (int) lround(0.5 * MFTL_PI * r_end / dx_ring));

  /* Ring layout, nozzle to head, then the pole. */
  const int n_rings = (n_tube + 1) + (n_cap - 1);
  ring *rings = (ring *) malloc((size_t) n_rings * sizeof(ring));
  if (!rings) return MFTL_ERROR;

  int n_verts = 0;
  double phase = 0.0;
  for (int j = 0; j < n_rings; ++j)
  {
    int count = sectors;
    if (j > n_tube)
    {
      const double theta = 0.5 * MFTL_PI * (1.0 - (double) (j - n_tube) / (double) n_cap);
      count = MFTL_MAX(3, (int) lround((double) sectors * sin(theta)));
    }
    rings[j].first = n_verts;
    rings[j].n = count;
    rings[j].phase = phase;
    n_verts += count;
    phase += MFTL_PI / (double) count;  // half a sector of THIS ring: the stagger
  }
  const int pole = n_verts++;

  int n_tris = 0;
  for (int j = 0; j + 1 < n_rings; ++j) n_tris += rings[j].n + rings[j + 1].n;
  n_tris += rings[n_rings - 1].n;

  double *pos = (double *) malloc(3 * (size_t) n_verts * sizeof(double));
  int *tris = (int *) malloc(3 * (size_t) n_tris * sizeof(int));
  if (!pos || !tris)
  {
    free(rings);
    free(pos);
    free(tris);
    return MFTL_ERROR;
  }

  for (int j = 0; j < n_rings; ++j)
  {
    double x, radius;
    if (j <= n_tube)
    {
      x = stub * (double) j / (double) n_tube;
      /* The lip is the nozzle circle exactly; the rest carries the grown
       * perturbation, so the stub is consistent with the field from t = 0. */
      radius = (j == 0) ? R0 : initial_radius(f, x);
    }
    else
    {
      const double theta = 0.5 * MFTL_PI * (1.0 - (double) (j - n_tube) / (double) n_cap);
      x = stub + r_end * cos(theta);
      radius = r_end * sin(theta);
    }

    for (int s = 0; s < rings[j].n; ++s)
    {
      const double phi = rings[j].phase + MFTL_TWO_PI * (double) s / (double) rings[j].n;
      double *p = &pos[3 * (rings[j].first + s)];
      p[0] = (j == 0) ? x0 : x0 + x;  // the lip on the wall to the bit
      p[1] = radius * cos(phi);
      p[2] = radius * sin(phi);
    }
  }
  pos[3 * pole + 0] = x0 + stub + r_end;
  pos[3 * pole + 1] = 0.0;
  pos[3 * pole + 2] = 0.0;

  int w = 0;
  for (int j = 0; j + 1 < n_rings; ++j) w = zip_rings(&rings[j], &rings[j + 1], tris, w);

  const ring *last = &rings[n_rings - 1];
  for (int s = 0; s < last->n; ++s)
  {
    tris[w++] = last->first + s;
    tris[w++] = last->first + (s + 1) % last->n;
    tris[w++] = pole;
  }
  MFTL_ASSERT(w == 3 * n_tris);

  const int rc = MFTLBuildIndexedMesh(obj, pos, n_verts, tris, n_tris);
  free(rings);
  free(pos);
  free(tris);
  return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Advection and the lip
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Clamp into the box. Negated comparisons, so a NaN is clamped too. */
static void clamp_to_domain(const MFTL_Domain *dom, double pos[3])
{
  for (int d = 0; d < 3; ++d)
  {
    if (!(pos[d] >= dom->global_min[d])) pos[d] = dom->global_min[d];
    if (!(pos[d] <= dom->global_max[d])) pos[d] = dom->global_max[d];
  }
}

/**
 * @brief RK2 (midpoint) over the vertices this rank owns; the lip does not move.
 *
 * Pinning the lip is the whole inflow mechanism: every other vertex leaves at U,
 * so the band of triangles attached to the lip stretches at rate U and the
 * remesher refills it.
 */
static void advect(MFTL_Ensemble *ens, const jet_field *f, double t, double dt)
{
  const MFTL_Domain *dom = &ens->domain;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];

    MFTL_FOREACH_OWNED_VERTEX(obj, {
      if (rec->flags & MFTL_FLAG_BOUNDARY) continue;

      double v1[3], mid[3], v2[3];
      jet_velocity(f, t, pos, v1);
      for (int d = 0; d < 3; ++d) mid[d] = pos[d] + 0.5 * dt * v1[d];
      clamp_to_domain(dom, mid);

      jet_velocity(f, t + 0.5 * dt, mid, v2);
      for (int d = 0; d < 3; ++d) pos[d] += dt * v2[d];
      clamp_to_domain(dom, pos);
    });
  }
}

/**
 * @brief Projects every lip vertex in the pool -- owned and ghost -- back onto
 *        the nozzle circle.
 *
 * Run after each remesh. The ridge smoother slides a lip vertex along the chord
 * of its two rim neighbours, which leaves it inside the circle by O(h^2/R0), and
 * a split of a lip edge puts its midpoint on the chord. The projection is a pure
 * function of the position, applied to bit-identical copies, so owner and ghost
 * agree without a message.
 */
static void project_lip(MFTL_Ensemble *ens)
{
  const double x0 = ens->domain.global_min[0];
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      if (!MFTLVertexIsMeshBoundary(obj, v)) continue;
      double *p = MFTLVertexPosMut(obj, v);
      const double r = sqrt(p[1] * p[1] + p[2] * p[2]);
      p[0] = x0;
      if (r > 0.0)
      {
        p[1] *= R0 / r;
        p[2] *= R0 / r;
      }
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
  S_VOLUME,  ///< enclosed by the jet and the nozzle disk
  S_LIP_AREA,  ///< area of the lip polygon: what the inflow passes through
  S_TRIS,
  S_VERTS,
  S_RIM,  ///< owned lip half-edges
  S_RIM_CUT,  ///< owned rim half-edges in triangles another rank holds a copy of
  S_HOLES,  ///< owned twin-less half-edges that are not mesh boundary
  S_OFF_LIP,  ///< owned lip vertices off the nozzle circle
  S_INVERTED,  ///< owned triangles facing against every edge neighbour: a fold-over
  S_CREASES,  ///< edges folded by more than 120 degrees: unresolved curvature
  S_N
};

enum
{
  X_MIN_EDGE,
  X_MIN_Q,
  X_NEG_MAX_EDGE,
  X_NEG_TIP,  ///< -max x: the head
  X_NEG_SWELL,  ///< -max distance from the axis
  X_N
};

typedef struct
{
  double sum[S_N];
  double min[X_N];  ///< every entry is a MIN; maxima are stored negated

  /* The neck, from the slab profile. */
  double neck_radius;
  double neck_x;
} jet_stats;

/**
 * @brief The thinnest section of the jet that has a wider one downstream.
 *
 * The jet is binned into slabs along the axis and each slab records the largest
 * distance from the axis of any vertex in it -- for a single-valued radius
 * profile that is the section's radius, to within a slab. MPI_MAX over ranks
 * makes it the whole jet's profile.
 *
 * "Has a wider section downstream" is what separates a NECK from the head: the
 * hemispherical head tapers to zero radius at its pole, and the thinnest slab of
 * a plain minimum would always be the last one. A neck between two swells, or
 * behind a bulbous head, passes; the taper towards the pole does not.
 *
 * @return The neck radius, R0 if there is no neck (an unperturbed jet).
 */
static double find_neck(MFTL_Ensemble *ens, double slab, int n_slabs, double *profile, double *scratch, double *neck_x)
{
  const double x0 = ens->domain.global_min[0];

  for (int s = 0; s < n_slabs; ++s) scratch[s] = -1.0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    MFTL_FOREACH_VERTEX_POS(obj, obj->verts.n_local, {
      int s = (int) ((px - x0) / slab);
      s = MFTL_MIN(MFTL_MAX(s, 0), n_slabs - 1);
      scratch[s] = MFTL_MAX(scratch[s], sqrt(py * py + pz * pz));
    });
  }
  MPI_Allreduce(scratch, profile, n_slabs, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  double neck = R0;
  *neck_x = x0;
  double widest_downstream = -1.0;
  for (int s = n_slabs - 1; s >= 0; --s)
  {
    if (profile[s] < 0.0) continue;  // empty slab
    if (profile[s] < widest_downstream && profile[s] < neck)
    {
      neck = profile[s];
      *neck_x = x0 + (s + 0.5) * slab;
    }
    widest_downstream = MFTL_MAX(widest_downstream, profile[s]);
  }
  return neck;
}

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
 * @brief Collective survey of the whole jet.
 *
 * The VOLUME is taken by the divergence theorem over the jet surface closed by
 * the nozzle disk: V = integral of (x - x_nozzle) n_x dA over the jet, since the
 * disk has x - x_nozzle = 0 and contributes nothing, and the one-point centroid
 * rule is exact on a flat triangle. The field is divergence-free, so the jet
 * gains exactly what flows through the lip polygon: A_lip * U per unit time.
 */
static void measure(MFTL_Ensemble *ens, jet_stats *g)
{
  const MFTL_Domain *dom = &ens->domain;
  const double x0 = dom->global_min[0];

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

        if (MFTLHeIsMeshBoundary(obj, h))
        {
          /* The lip polygon's area, by the shoelace formula in the nozzle
           * plane. Its sign follows the lip's orientation, fixed by the
           * outward winding; abs() below makes it independent of it. */
          const double *pa = MFTLVertexPos(obj, MFTLHeOrigin(obj, h));
          const double *pb = MFTLVertexPos(obj, MFTLHeTarget(obj, h));
          s[S_LIP_AREA] += 0.5 * (pa[1] * pb[2] - pb[1] * pa[2]);
          s[S_RIM] += 1.0;
          if (is_replicated(obj, t)) s[S_RIM_CUT] += 1.0;
        }
        else if (MFTLHeIsLocalBoundary(obj, h))
          s[S_HOLES] += 1.0;
      }

      const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
      x[X_MIN_Q] = MFTL_MIN(x[X_MIN_Q], q);
      s[S_Q_SUM] += q;
      s[S_AREA] += area;
      s[S_VOLUME] += (c[0] - x0) * n[0] * area;

      fold_state(obj, t, n, &s[S_INVERTED], &s[S_CREASES]);
    }

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      const double r = sqrt(p[1] * p[1] + p[2] * p[2]);
      x[X_NEG_TIP] = MFTL_MIN(x[X_NEG_TIP], -p[0]);
      x[X_NEG_SWELL] = MFTL_MIN(x[X_NEG_SWELL], -r);

      if (MFTLVertexIsMeshBoundary(obj, v) && (fabs(p[0] - x0) > JET_LIP_TOL * R0 || fabs(r - R0) > JET_LIP_TOL * R0)) s[S_OFF_LIP] += 1.0;
    }

    s[S_TRIS] += (double) obj->tris.n_local;
    s[S_VERTS] += (double) obj->verts.n_local;
  }

  MPI_Allreduce(s, g->sum, S_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(x, g->min, X_N, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  g->sum[S_LIP_AREA] = fabs(g->sum[S_LIP_AREA]);
}

/** @brief V - E + F from owned counts, with E = (3F + B)/2: 1 for the jet (a disk). */
static double euler_of(const jet_stats *g) { return g->sum[S_VERTS] - 0.5 * (3.0 * g->sum[S_TRIS] + g->sum[S_RIM]) + g->sum[S_TRIS]; }

/**
 * @brief The verdict --verify acts on, identical on every rank.
 *
 * The volume balance is the physical statement: the jet must hold exactly what
 * came through the lip, V0 + A_lip U t. What the loop actually loses is the
 * remesher's discretisation error on a curved surface, and it converges with
 * resolution -- measured to pinch-off at the defaults: 4.2 % at 24 sectors, 2.5 %
 * at 32, 1.4 % at 48 (edges of 0.26, 0.20 and 0.13 R0). Five percent passes the
 * default and fails a remesher that eats volume at the inflow, where it splits a
 * stretched band every pass; with remeshing OFF the band's long straight
 * triangles cut through the necks and swells and the balance misses by 40 %.
 */
static bool verdict(const jet_stats *g, double v_expected, char *why, size_t why_len)
{
  why[0] = '\0';
  if (g->sum[S_HOLES] != 0.0) snprintf(why, why_len, "%.0f twin-less interior half-edges (halo incomplete)", g->sum[S_HOLES]);
  if (fabs(euler_of(g) - 1.0) > 0.5) snprintf(why, why_len, "V-E+F = %.0f, the jet is no longer a disk", euler_of(g));
  if (g->sum[S_OFF_LIP] != 0.0) snprintf(why, why_len, "%.0f lip vertices left the nozzle circle", g->sum[S_OFF_LIP]);
  if (g->sum[S_INVERTED] != 0.0) snprintf(why, why_len, "%.0f inverted triangles", g->sum[S_INVERTED]);
  if (fabs(g->sum[S_VOLUME] / v_expected - 1.0) > 0.05)
    snprintf(why, why_len, "volume %.5f against %.5f that came through the nozzle", g->sum[S_VOLUME], v_expected);
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
  printf("\n  %5s %6s | %6s %6s %6s %6s | %7s | %8s %8s %8s | %7s %6s | %8s", "step", "t", "tip", "neck", "at x", "swell", "ntri", "edge min", "mean",
         "max", "q worst", "mean", "V/V_in");
  if (remeshing) printf(" | %6s %6s %6s %3s %4s", "split", "coll", "flip", "ph", "str");
  printf("\n  ");
  for (int i = 0; i < (remeshing ? 144 : 112); ++i) putchar('-');
  printf("\n");
}

static void print_stats(int step, double t, const jet_stats *g, double v_expected, const remesh_window *w, bool remeshing)
{
  const double mean_edge = (g->sum[S_EDGE_N] > 0.0) ? g->sum[S_EDGE_SUM] / g->sum[S_EDGE_N] : 0.0;
  const double mean_q = (g->sum[S_TRIS] > 0.0) ? g->sum[S_Q_SUM] / g->sum[S_TRIS] : 0.0;

  printf("  %5d %6.2f | %6.2f %6.3f %6.2f %6.3f | %7.0f | %8.5f %8.5f %8.5f | %7.4f %6.4f | %8.5f", step, t, -g->min[X_NEG_TIP], g->neck_radius, g->neck_x,
         -g->min[X_NEG_SWELL], g->sum[S_TRIS], g->min[X_MIN_EDGE], mean_edge, -g->min[X_NEG_MAX_EDGE], g->min[X_MIN_Q], mean_q, g->sum[S_VOLUME] / v_expected);
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

  /* The nozzle is in the x_min wall, on the axis y = z = 0. Walls everywhere:
   * the jet leaves through nothing -- the run stops before the head arrives. */
  double global_min[3] = {0.0, -0.5 * cfg.width, -0.5 * cfg.width};
  double global_max[3] = {cfg.length, 0.5 * cfg.width, 0.5 * cfg.width};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, global_min, global_max, periodic, 1);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  jet_field field;
  jet_field_init(&field, &ens->domain, &cfg);

  MFTL_Object *jet = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, jet);

  if (rank == 0)
  {
    if (build_jet(jet, &ens->domain, &field, cfg.stub, cfg.sectors) != MFTL_SUCCESS)
    {
      fprintf(stderr, "Failed to build the jet.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MFTLCheckMeshIntegrity(jet);
  }
  else
    MFTLObjectBuildDerived(jet);
  MFTLPublishGeneratedBodies(ens);

  /* Scatter. `jet` is dead after this: the handle that survives is the body id. */
  MFTLComputeTriangleGeometry(jet, jet->tris.n);
  MFTLRedistribute(ens);
  jet = MFTL_NULL;

  const double target = (cfg.target > 0.0) ? cfg.target : MFTLEnsembleMeanEdgeLength(ens);
  const double pinch_radius = MFTL_MAX(cfg.pinch * R0, target);

  MFTL_RemeshParameters params;
  /* A smooth interface: feature detection off. The lip is a feature regardless
   * -- a mesh boundary always is -- and slides as a ridge (see project_lip). */
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, cfg.iterations, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = cfg.smooth;
  params.parallel_strategy = cfg.strategy;
  params.parallel_phase_budget = cfg.budget;

  const bool remeshing = cfg.remesh_every > 0;

  /* The neck profile's slabs: one target edge wide, so every slab of the tube
   * holds vertices of at least one ring. */
  const double slab = target;
  const int n_slabs = (int) ceil(cfg.length / slab) + 1;
  double *profile = (double *) malloc((size_t) n_slabs * sizeof(double));
  double *scratch = (double *) malloc((size_t) n_slabs * sizeof(double));
  if (!profile || !scratch) MPI_Abort(MPI_COMM_WORLD, 1);

  jet_stats g0;
  measure(ens, &g0);
  g0.neck_radius = find_neck(ens, slab, n_slabs, profile, scratch, &g0.neck_x);
  const double v0 = g0.sum[S_VOLUME];
  const double lip_area0 = g0.sum[S_LIP_AREA];

  const double breakup_length = (cfg.eps > 0.0) ? (U0 / field.sigma) * log(R0 / field.eps0) : 0.0;

  if (rank == 0)
  {
    printf("=== liquid jet from a nozzle, Rayleigh-Plateau breakup, on an open surface ===\n");
    printf("  %d rank(s), grid %dx%dx%d | domain [0,%.0f] x [%.1f,%.1f]^2 in units of the nozzle radius R0\n", nproc, cfg.cuts[0], cfg.cuts[1], cfg.cuts[2],
           cfg.length, global_min[1], global_max[1]);
    printf("  jet: %.0f triangles, %.0f lip half-edges, V-E+F = %.0f (a disk), stub %.1f R0 + head, volume %.4f\n", g0.sum[S_TRIS], g0.sum[S_RIM],
           euler_of(&g0), cfg.stub, v0);
    if (cfg.eps > 0.0)
    {
      printf("  Rayleigh-Plateau: k R0 %.3f (lambda %.2f R0), We %.2f -> sigma %.4f U/R0; eps0 %.3f R0\n", cfg.kr, MFTL_TWO_PI / field.k, cfg.weber,
             field.sigma, field.eps0);
      printf("                    Rayleigh's linear breakup length L_b = (U/sigma) ln(R0/eps0) = %.2f R0\n", breakup_length);
    }
    else
      printf("  unperturbed jet (--eps 0): it grows straight and never breaks\n");
    printf("  pinch-off is declared when a neck is thinner than %.3f R0\n", pinch_radius);
    if (remeshing)
    {
      printf("  remesh every %d step(s): target %.4f (band %.4f .. %.4f), %d iterations, %d smoothing sweeps, budget %d\n", cfg.remesh_every, target,
             params.min_edge_length, params.max_edge_length, cfg.iterations, cfg.smooth, cfg.budget);
    }
    else
      printf("  remeshing OFF: the band at the lip stretches without bound\n");
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
  double t = 0.0;
  int broken = 0, n_passes = 0, n_not_converged = 0, step = 0, n_dispersed = 0, n_spanning = 0;
  double rim_cut_max = 0.0;
  long total_ops = 0;
  double t_remesh = 0.0, worst_q = g0.min[X_MIN_Q];
  bool pinched = MFTL_FALSE, reached_outlet = MFTL_FALSE;
  remesh_window win = {0, 0, 0, 0, MFTL_REMESH_PARALLEL_SPANNING, MFTL_FALSE, MFTL_FALSE};
  jet_stats g = g0;

  /* Without --steps the run ends at pinch-off or at the outlet; the cap only
   * stops an unperturbed jet in a domain long enough never to fill. */
  const int max_steps = (cfg.steps > 0) ? cfg.steps : (int) ceil(4.0 * cfg.length / (U0 * cfg.dt));

  for (step = 1; step <= max_steps; ++step)
  {
    advect(ens, &field, t, cfg.dt);
    t += cfg.dt;

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
      project_lip(ens);

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
      if (st.strategy == MFTL_REMESH_PARALLEL_DISPERSED)
        n_dispersed++;
      else
        n_spanning++;
    }
    else if (MFTLRedistribute(ens) != MFTL_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 1);

    /* The head and the neck are checked EVERY step -- one small reduction each
     * -- because stopping one report late at a pinch means stopping with a neck
     * of a fraction of an edge, i.e. with a degenerate mesh. */
    double tip_local = -MFTL_DBL_MAX, tip = 0.0;
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      const MFTL_Object *obj = ens->local_objects[o];
      for (int v = 0; v < obj->verts.n_local; ++v) tip_local = MFTL_MAX(tip_local, obj->verts.pos[3 * v]);
    }
    MPI_Allreduce(&tip_local, &tip, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    double neck_x = 0.0;
    const double neck = find_neck(ens, slab, n_slabs, profile, scratch, &neck_x);

    pinched = (cfg.eps > 0.0) && neck < pinch_radius;
    reached_outlet = tip > global_max[0] - 2.0 * R0;

    const bool last = pinched || reached_outlet || step == max_steps;
    if ((cfg.report_every > 0 && step % cfg.report_every == 0) || last)
    {
      measure(ens, &g);
      g.neck_radius = neck;
      g.neck_x = neck_x;

      const double v_expected = v0 + lip_area0 * U0 * t;
      if (rank == 0) print_stats(step, t, &g, v_expected, &win, remeshing);
      memset(&win, 0, sizeof win);
      worst_q = MFTL_MIN(worst_q, g.min[X_MIN_Q]);

      rim_cut_max = MFTL_MAX(rim_cut_max, g.sum[S_RIM_CUT]);

      char why[160];
      if (!verdict(&g, v_expected, why, sizeof why))
      {
        broken++;
        if (rank == 0) fprintf(stderr, "  step %d: %s\n", step, why);
      }
    }

    if (cfg.vtk_every > 0 && (step % cfg.vtk_every == 0 || last))
    {
      char prefix[256];
      snprintf(prefix, sizeof prefix, "%s_%04d", cfg.out, step);
      MFTLWriteVtu(ens, prefix, MFTL_NULL);
    }

    if (last) break;
  }
  if (step > max_steps) step = max_steps;

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

  double t_max = 0.0;
  MPI_Allreduce(&t_remesh, &t_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
  {
    const double v_expected = v0 + lip_area0 * U0 * t;

    printf("\n");
    if (pinched)
    {
      printf("  PINCH-OFF at t = %.2f R0/U: a neck at x = %.2f R0 is %.3f R0 thin (%.1f edges around).\n", t, g.neck_x, g.neck_radius,
             MFTL_TWO_PI * g.neck_radius / target);
      printf("  This is where the jet breaks into a drop. Cutting the ligament and closing both ends is a\n");
      printf("  topology change, which the library does not do yet -- so the run stops here.\n");
      printf("  breakup length   %.2f R0 measured, %.2f R0 Rayleigh's linear estimate (see the file header\n", g.neck_x - global_min[0], breakup_length);
      printf("                   for why the kinematic neck pinches downstream of it)\n");
    }
    else if (reached_outlet)
      printf("  The head reached the outlet at t = %.2f without a pinch-off%s.\n", t,
             (cfg.eps > 0.0) ? " -- lengthen the domain or raise --eps" : ", as an unperturbed jet does");
    else
      printf("  Stopped after %d steps (t = %.2f), neck %.3f R0 at x = %.2f.\n", step, t, g.neck_radius, g.neck_x);

    printf("  jet              head at x = %.2f, widest swell %.3f R0, surface area %.3f\n", -g.min[X_NEG_TIP], -g.min[X_NEG_SWELL], g.sum[S_AREA]);
    printf("  triangles        %.0f -> %.0f   (lip %.0f -> %.0f half-edges, polygon area %.5f -> %.5f)\n", g0.sum[S_TRIS], g.sum[S_TRIS], g0.sum[S_RIM],
           g.sum[S_RIM], lip_area0, g.sum[S_LIP_AREA]);
    printf("  volume           %.5f, against %.5f = V0 + A_lip U t that came through the nozzle (%.4f)\n", g.sum[S_VOLUME], v_expected,
           g.sum[S_VOLUME] / v_expected);
    printf("  worst quality    %.4f at the start, %.4f at the worst report, %.4f at the end\n", g0.min[X_MIN_Q], worst_q, g.min[X_MIN_Q]);
    printf("  Euler V-E+F      %.0f   (1: still one disk)\n", euler_of(&g));
    printf("  orientation      %.0f inverted triangles, %.0f creases sharper than 120 deg\n", g.sum[S_INVERTED], g.sum[S_CREASES]);
    if (remeshing)
    {
      if (nproc == 1)
        printf("  remeshing        %d serial passes, %ld operations, %.3f s\n", n_passes, total_ops, t_max);
      else
        printf("  remeshing        %d passes (%d dispersed, %d spanning), %ld operations, %d not converged, %.3f s (max over ranks)\n", n_passes,
               n_dispersed, n_spanning, total_ops, n_not_converged, t_max);
    }
    printf("  lip across ranks up to %.0f rim half-edges in triangles another rank also holds%s\n", rim_cut_max,
           (nproc > 1) ? "" : " (one rank: none can be)");
    printf("  structure        %s\n", (broken == 0) ? "sound at every report" : "BROKEN -- see above");
    if (cfg.vtk_every > 0) printf("  wrote %s_*.pvtu every %d steps and at the end.\n", cfg.out, cfg.vtk_every);
  }

  free(profile);
  free(scratch);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (cfg.verify && broken > 0) ? 1 : 0;
}
