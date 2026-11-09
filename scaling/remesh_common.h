/*
 * @Author: Christian Gorges
 * @Date: 2026-11-06
 *
 * @brief What remesh_strong_scaling.c and remesh_weak_scaling.c share: the
 *        flow, the remeshed timestep, the accuracy checks and the report.
 *
 * -- The problem: bubbles in a straining flow, one knob for the regime ------------
 *
 *   Spheres of radius R on a cubic lattice of spacing a in a fully periodic box.
 *   The velocity is a cellular strain that TRAVELS with a uniform drift,
 *
 *     u(x, t) = s(t) [ d + w(x - d tau(t)) ],   s = cos(pi t/T),  tau = (T/pi) sin(pi t/T),
 *     w(xi)   = U ( sin X cos Y cos Z, -1/2 cos X sin Y cos Z, -1/2 cos X cos Y sin Z ),
 *
 *   with X = 2 pi xi_x / a + pi etc.: a Taylor-Green-type cell of period a,
 *   divergence-free, with a stagnation point on every body centre. In the frame
 *   xi = x - d tau(t) the velocity is s(t) w(xi), an autonomous field times a
 *   scalar that integrates to zero over [0, T], so the flow map at t = T is the
 *   IDENTITY: every body is a sphere again, which is what makes the result
 *   checkable at a scale where no serial reference can be run. Each body is
 *   stretched along x into a prolate ellipsoid (log-stretch about S = --strain
 *   at T/2 for a small body) and restored, while the drift carries it
 *   (1.5, 1, 0.5) a * --drift out and back across the cut planes, which is what
 *   exercises the remesher's migrations. Every body has the SAME history, so with
 *   the lattice aligned to the rank boxes every rank does the same work (exactly
 *   so in the weak drivers; the report prints the imbalance).
 *
 *   --regime dispersed   droplets much smaller than a rank box (R/h < 0.4): the
 *                        dispersed driver remeshes each body whole on one rank.
 *                        Its answer is the serial answer to the bit, and every
 *                        other step here is partition-independent, so the final
 *                        DIGEST of a strong sweep must be the same at every P.
 *   --regime spanning    bubbles wider than a rank box (R/h > 0.5): the spanning
 *                        driver moves the cut planes. Both of its modes
 *                        (MFTL_RemeshSpanningMode: once, nested) run in the same
 *                        job, on the same problem, one after the other.
 *
 * -- The timestep -----------------------------------------------------------------
 *
 *   advect   RK4 of the owned vertices, wrapped into the box
 *   ghost    MFTLUpdateGhostPositions
 *   geom     MFTLComputeTriangleGeometry over all triangles
 *   remesh   MFTLRemeshEnsemble, its closing redistribution included (every
 *            --remesh-every steps; MFTLRedistribute + geometry in between)
 *
 *   Every phase is timed per rank and its MAXIMUM over ranks is summed per step,
 *   behind a barrier at the start of each step: a remeshing pass is synchronised
 *   by its own collectives, so the slowest rank sets the step.
 *
 * -- What is checked at the end of every trajectory (t = T) ------------------------
 *
 *   The Euler characteristic of the ensemble (2 per body, exactly); the volume
 *   against the generated one (the sum and the worst body); the distance of
 *   every vertex to its body's exact sphere (rms and max, % of R); minimum and
 *   mean triangle quality; edges outside the length band; passes that did not
 *   converge; vertices the projection could not place; and at T/2 the
 *   deformation reached (the largest and smallest vertex distance to the body
 *   centre, in R). A run whose structure breaks or whose volume moves by more
 *   than --max-dv fails.
 */
#ifndef MFTL_REMESH_SCALING_COMMON_H
#define MFTL_REMESH_SCALING_COMMON_H

/* The redistribution harness's helpers are reused (options, the bounded-batch
 * initialisation, report output, RSS, quantiles); its timestep and report are
 * not, and would otherwise warn as unused. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "scaling_common.h"
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "mftl_remeshing.h"

/** @brief The two regimes. */
enum
{
  RM_DISPERSED = 0,
  RM_SPANNING = 1
};

static const char *const RM_REGIME_NAME[2] = {"dispersed", "spanning"};

/** @brief The body lattice: body `b` sits at `origin + (i, j, k) a`, `b = i + n0 (j + n1 k)`. */
typedef struct
{
  int n[3];
  double a;
  double origin[3];
} rm_lattice;

/** @brief What the driver decided the problem is. */
typedef struct
{
  const char *mode;  ///< "remesh_strong" or "remesh_weak".
  int regime;  ///< RM_DISPERSED or RM_SPANNING.
  int n_bodies;
  double gmin[3];
  double gmax[3];
  double radius;
  rm_lattice lat;
  char placement[200];
} rm_problem;

/**
 * @brief Centre of body @p body at t = 0 (an sc_centre_fn over an rm_lattice).
 * @param      ctx  The rm_lattice.
 * @param      body Body index.
 * @param[out] out  The centre.
 */
static void rm_centre(const void *ctx, int body, double out[3])
{
  const rm_lattice *L = (const rm_lattice *) ctx;
  const int ijk[3] = {body % L->n[0], (body / L->n[0]) % L->n[1], body / (L->n[0] * L->n[1])};
  for (int d = 0; d < 3; ++d) out[d] = L->origin[d] + (double) ijk[d] * L->a;
}

/** @brief Run options. */
typedef struct
{
  sc_options sc;  ///< refine, steps (per trajectory), cuts, init batch, report, label, git.
  int regime;
  double radius;  ///< R / a; 0 = the regime's default.
  double strain;  ///< S: log-stretch along x at T/2 for a small body.
  double drift;  ///< Excursion of the drift, in units of (1.5, 1, 0.5) a.
  int trajectories;  ///< Timed trajectories; the fastest is reported.
  int warmup;  ///< Untimed trajectories first (the first remesh to the band happens there).
  int remesh_every;
  int iterations;
  int smooth;
  int budget;
  int sizing;  ///< MFTL_RemeshSizing.
  int projection;  ///< MFTL_RemeshProjection.
  int modes[2];  ///< Spanning modes to run, in order.
  int n_modes;
  double max_dv;  ///< A run whose volume moved by more than this fraction fails.
  int vtk_every;  ///< Write the first trajectory as .pvtu every this many steps; 0 = never.
} rm_options;

/**
 * @brief Defaults of both drivers.
 * @param[out] o    Receives the defaults.
 * @param      mode "remesh_strong" or "remesh_weak", for the default label.
 */
static void rm_options_defaults(rm_options *o, const char *mode)
{
  memset(o, 0, sizeof *o);
  sc_options_defaults(&o->sc, mode, 100);
  o->sc.refine = -1;  // the regime's default
  o->sc.jitter_given = true;  // a lattice, never jittered
  o->sc.init_batch = 0;  // the regime's default
  o->regime = RM_SPANNING;
  o->strain = 0.6;
  o->drift = 1.0;
  o->trajectories = 2;
  o->warmup = 1;
  o->remesh_every = 1;
  o->iterations = 3;
  o->smooth = 3;
  o->budget = MFTL_REMESH_NUM_PHASES;
  o->sizing = MFTL_REMESH_SIZING_UNIFORM;
  o->projection = MFTL_REMESH_PROJECT_PHONG;
  o->modes[0] = MFTL_REMESH_SPANNING_ONCE;
  o->modes[1] = MFTL_REMESH_SPANNING_NESTED;
  o->n_modes = 2;
  o->max_dv = 0.05;
}

/** @brief Usage of the options both drivers share. */
static const char *const RM_COMMON_USAGE =
    "  --regime R          dispersed | spanning                                (default spanning)\n"
    "  --refine N          icosphere level, 20*4^N triangles per body  (default 3 dispersed, 7 spanning)\n"
    "  --radius F          body radius in lattice spacings a       (default 0.25 dispersed, 0.3 spanning)\n"
    "  --steps N           steps per trajectory; the flow reverses half-way       (default 100)\n"
    "  --trajectories N    timed trajectories; the fastest is reported           (default 2)\n"
    "  --warmup N          untimed trajectories first                            (default 1)\n"
    "  --strain S          log-stretch along x at T/2 (aspect about exp(1.5 S))   (default 0.6)\n"
    "  --drift D           drift excursion in units of (1.5, 1, 0.5) a            (default 1)\n"
    "  --remesh-every N    remesh every N steps                                  (default 1)\n"
    "  --iterations N --smooth N --budget N   MFTL_RemeshParameters           (default 3, 3, 8)\n"
    "  --sizing S          uniform | curvature                                   (default uniform)\n"
    "  --projection P      phong | none                                          (default phong)\n"
    "  --spanning-modes M  once,nested | nested,once | once | nested: the spanning modes this\n"
    "                      job runs, one after the other on the same problem  (default once,nested)\n"
    "  --max-dv X          fail a run whose volume moved by more than X          (default 0.05)\n"
    "  --vtk N             write the first trajectory every N steps as <label>_<mode>_NNNN.pvtu,\n"
    "                      outside the step's clock -- for looking at it, not for timing (default 0)\n"
    "  --cuts X Y Z        process grid\n"
    "  --init-batch N      spheres per initialisation redistribution    (default 64 dispersed, 4 spanning)\n"
    "  --max-gen-mb X      limit the mesh storage of ONE initialisation batch on rank 0\n"
    "  --max-seconds X     soft job budget, checked between trajectories     (default 3000, 0 = off)\n"
    "  --report PATH       write the report here (it is also printed)\n"
    "  --label NAME --git HASH   recorded verbatim\n";

/**
 * @brief Tries to consume one shared option at argv[*i].
 * @param argc From main.
 * @param argv From main.
 * @param[in,out] i Current index; advanced past any value consumed.
 * @param[out] o Receives the value.
 * @return 1 if consumed, 0 if not a shared option, -1 for a missing or invalid value.
 */
static int rm_parse_common(int argc, char **argv, int *i, rm_options *o)
{
  const char *a = argv[*i];
  const bool has1 = (*i + 1 < argc);
#define RM_INT(name, field)                                         \
  if (!strcmp(a, name))                                             \
  {                                                                 \
    if (!has1 || !sc_parse_int(argv[++(*i)], &o->field)) return -1; \
    return 1;                                                       \
  }
#define RM_DBL(name, field)                                            \
  if (!strcmp(a, name))                                                \
  {                                                                    \
    if (!has1 || !sc_parse_double(argv[++(*i)], &o->field)) return -1; \
    return 1;                                                          \
  }
  RM_DBL("--radius", radius)
  RM_DBL("--strain", strain)
  RM_DBL("--drift", drift)
  RM_INT("--trajectories", trajectories)
  RM_INT("--warmup", warmup)
  RM_INT("--remesh-every", remesh_every)
  RM_INT("--iterations", iterations)
  RM_INT("--smooth", smooth)
  RM_INT("--budget", budget)
  RM_DBL("--max-dv", max_dv)
  RM_INT("--vtk", vtk_every)
#undef RM_INT
#undef RM_DBL

  /* Keyword options. */
  if (!strcmp(a, "--regime") || !strcmp(a, "--sizing") || !strcmp(a, "--projection") || !strcmp(a, "--spanning-modes"))
  {
    if (!has1) return -1;
    const char *v = argv[++(*i)];
    if (!strcmp(a, "--regime"))
    {
      if (!strcmp(v, "dispersed"))
        o->regime = RM_DISPERSED;
      else if (!strcmp(v, "spanning"))
        o->regime = RM_SPANNING;
      else
        return -1;
    }
    else if (!strcmp(a, "--sizing"))
    {
      if (!strcmp(v, "uniform"))
        o->sizing = MFTL_REMESH_SIZING_UNIFORM;
      else if (!strcmp(v, "curvature"))
        o->sizing = MFTL_REMESH_SIZING_CURVATURE;
      else
        return -1;
    }
    else if (!strcmp(a, "--projection"))
    {
      if (!strcmp(v, "phong"))
        o->projection = MFTL_REMESH_PROJECT_PHONG;
      else if (!strcmp(v, "none"))
        o->projection = MFTL_REMESH_PROJECT_NONE;
      else
        return -1;
    }
    else
    {
      o->n_modes = 0;
      if (!strcmp(v, "once") || !strcmp(v, "once,nested")) o->modes[o->n_modes++] = MFTL_REMESH_SPANNING_ONCE;
      if (!strcmp(v, "nested") || !strcmp(v, "once,nested") || !strcmp(v, "nested,once")) o->modes[o->n_modes++] = MFTL_REMESH_SPANNING_NESTED;
      if (!strcmp(v, "nested,once")) o->modes[o->n_modes++] = MFTL_REMESH_SPANNING_ONCE;
      if (o->n_modes == 0) return -1;
    }
    return 1;
  }

  /* The redistribution harness's options that mean the same thing here. */
  static const char *const shared[] = {"--refine",      "--steps",  "--cuts",  "--init-batch", "--max-gen-mb",
                                       "--max-seconds", "--report", "--label", "--git",        "--verify-init"};
  for (size_t k = 0; k < sizeof shared / sizeof shared[0]; ++k)
    if (!strcmp(a, shared[k])) return sc_parse_common(argc, argv, i, &o->sc);
  return 0;
}

/**
 * @brief Fills the regime's defaults, validates, and resolves the process grid.
 * @param[in,out] o     The options.
 * @param         nproc Communicator size.
 * @param         rank  This rank; only rank 0 prints.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR after explaining why.
 */
static int rm_options_finish(rm_options *o, int nproc, int rank)
{
  const bool span = (o->regime == RM_SPANNING);
  if (o->sc.refine < 0) o->sc.refine = span ? 7 : 3;
  if (o->sc.init_batch <= 0) o->sc.init_batch = span ? 4 : 64;
  if (o->radius <= 0.0) o->radius = span ? 0.3 : 0.25;
  if (sc_options_finish(&o->sc, nproc, rank) != MFTL_SUCCESS) return MFTL_ERROR;
  if (o->trajectories < 1 || o->warmup < 0 || o->remesh_every < 1 || o->iterations < 1 || o->smooth < 0 || o->budget < 1 || o->strain < 0.0 || o->drift < 0.0 ||
      !(o->max_dv > 0.0) || o->sc.steps < 2 || o->radius >= 0.45)
  {
    if (rank == 0)
      fprintf(stderr,
              "need trajectories >= 1, warmup >= 0, remesh-every >= 1, iterations >= 1, smooth >= 0, budget >= 1, steps >= 2, "
              "strain and drift >= 0, max-dv > 0, radius < 0.45 lattice spacings.\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/**
 * @brief The most cubic process grid whose every axis is a multiple of @p m.
 * @param      nproc Rank count.
 * @param      m     Divisor every cut count must have.
 * @param[out] cuts  The grid, largest axis first.
 * @return true if one exists.
 */
static bool rm_divisible_cuts(int nproc, int m, int cuts[3])
{
  bool found = false;
  double best = 0.0;
  for (int c0 = 1; c0 <= nproc; ++c0)
  {
    if (nproc % c0) continue;
    for (int c1 = 1; c1 <= c0; ++c1)
    {
      if ((nproc / c0) % c1) continue;
      const int c2 = nproc / c0 / c1;
      if (c2 > c1 || c0 % m || c1 % m || c2 % m) continue;
      const double ratio = (double) c2 / (double) c0;  // 1 for a cube
      if (!found || ratio > best)
      {
        found = true;
        best = ratio;
        cuts[0] = c0, cuts[1] = c1, cuts[2] = c2;
      }
    }
  }
  return found;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The flow
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The velocity field and its parameters. */
typedef struct
{
  double period;  ///< T: one trajectory.
  double k;  ///< 2 pi / a.
  double drift[3];  ///< d: peak drift velocity.
  double u;  ///< U: peak strain velocity scale.
  double origin[3];  ///< Body 0's centre at t = 0, a stagnation point of w.
} rm_flow;

/**
 * @brief Resolves the flow from the options.
 * @param      prob The problem (lattice spacing and origin).
 * @param      opt  The options.
 * @param[out] f    The flow.
 */
static void rm_flow_init(const rm_problem *prob, const rm_options *opt, rm_flow *f)
{
  const double a = prob->lat.a;
  f->period = 1.0;
  f->k = 2.0 * MFTL_PI / a;
  /* tau(T/2) = T/pi, so the excursion d T/pi is (1.5, 1, 0.5) a * drift. */
  const double ex[3] = {1.5, 1.0, 0.5};
  for (int d = 0; d < 3; ++d)
  {
    f->drift[d] = opt->drift * ex[d] * a * MFTL_PI / f->period;
    f->origin[d] = prob->lat.origin[d];
  }
  /* At a centre dw_x/dx = U k, so the log-stretch at T/2 is U k T/pi. */
  f->u = opt->strain * MFTL_PI / (f->k * f->period);
}

/**
 * @brief How far the strain cells have travelled, in units of the drift.
 * @param f The flow.
 * @param t Time.
 * @return tau(t) = (T/pi) sin(pi t/T).
 */
static inline double rm_tau(const rm_flow *f, double t) { return f->period / MFTL_PI * sin(MFTL_PI * t / f->period); }

/**
 * @brief Velocity at @p x and time @p t.
 * @param      f The flow.
 * @param      x Position.
 * @param      t Time.
 * @param[out] u Velocity.
 */
static inline void rm_velocity(const rm_flow *f, const double x[3], double t, double u[3])
{
  const double s = cos(MFTL_PI * t / f->period), tau = rm_tau(f, t);
  /* + pi: a centre is where cos X = cos Y = cos Z = -1, the cell's stretching point. */
  const double X = f->k * (x[0] - f->origin[0] - f->drift[0] * tau) + MFTL_PI;
  const double Y = f->k * (x[1] - f->origin[1] - f->drift[1] * tau) + MFTL_PI;
  const double Z = f->k * (x[2] - f->origin[2] - f->drift[2] * tau) + MFTL_PI;
  const double sx = sin(X), cx = cos(X), sy = sin(Y), cy = cos(Y), sz = sin(Z), cz = cos(Z);
  /* dw_x/dx = -U k cosX cosY cosZ = +U k at a centre: stretched along x,
   * compressed half as fast along y and z, divergence-free everywhere. */
  u[0] = s * (f->drift[0] - f->u * sx * cy * cz);
  u[1] = s * (f->drift[1] + 0.5 * f->u * cx * sy * cz);
  u[2] = s * (f->drift[2] + 0.5 * f->u * cx * cy * sz);
}

/**
 * @brief RK4 of every owned vertex over [t, t + dt], wrapped into the box.
 * @param ens The ensemble.
 * @param f   The flow.
 * @param t   Start time.
 * @param dt  Step.
 */
static void rm_advect(MFTL_Ensemble *ens, const rm_flow *f, double t, double dt)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    MFTL_FOREACH_OWNED_VERTEX(obj, {
      double k1[3], k2[3], k3[3], k4[3], x[3];
      rm_velocity(f, pos, t, k1);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + 0.5 * dt * k1[d];
      rm_velocity(f, x, t + 0.5 * dt, k2);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + 0.5 * dt * k2[d];
      rm_velocity(f, x, t + 0.5 * dt, k3);
      for (int d = 0; d < 3; ++d) x[d] = pos[d] + dt * k3[d];
      rm_velocity(f, x, t + dt, k4);
      for (int d = 0; d < 3; ++d) pos[d] += (dt / 6.0) * (k1[d] + 2.0 * k2[d] + 2.0 * k3[d] + k4[d]);
      MFTLApplyPeriodicity(&ens->domain, pos);
    });
    MFTLObjectMarkChanged(obj);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Measuring the mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Global measures of the mesh. */
typedef struct
{
  double tris, verts, edges, oob;  ///< Counts; `oob` = edges outside the length band.
  double volume, worst_dv;  ///< Sum of body volumes; worst |V_b / V0 - 1|.
  double dist_rms, dist_max;  ///< | |x - c| / R - 1 | over vertices.
  double r_max, r_min;  ///< Largest and smallest |x - c| / R: the deformation.
  double qmin, qmean;
  double tris_rank_max, tris_rank_mean;
  unsigned long long digest;  ///< XOR over owned vertices of (body, id, position bits): independent of the partition.
} rm_state;

/**
 * @brief Does this rank count undirected edge @p h, so that the mesh counts it exactly once?
 * @param obj The body.
 * @param h   A half-edge.
 * @return 1 if this rank counts it here.
 */
static int rm_counts_edge(const MFTL_Object *obj, int h)
{
  const int t = MFTLHeTriIdx(h);
  if (t >= obj->tris.n_local) return 0;
  const int tw = obj->tris.he[h].twin_idx;
  if (tw == MFTL_ERROR) return 1;
  const int to = MFTLHeTriIdx(tw);
  if (to < obj->tris.n_local) return h < tw;
  return obj->tris.rec[t].global_id < obj->tris.rec[to].global_id;  // the other side is a ghost: the lower triangle id counts
}

/**
 * @brief Measures the ensemble; collective. Needs current triangle geometry.
 * @param      ens    The ensemble.
 * @param      prob   For each body's start centre and the radius.
 * @param      shift  Where the centres are now, relative to the start.
 * @param      params For the length band.
 * @param      v0     Initial volume of one body (all bodies are alike); 0 skips the volumes.
 * @param[out] s      The state.
 */
static void rm_measure(MFTL_Ensemble *ens, const rm_problem *prob, const double shift[3], const MFTL_RemeshParameters *params, double v0, rm_state *s)
{
  double acc[7] = {0, 0, 0, 0, 0, 0, 0}, mx[5] = {0.0, -1.0, -2.0, -1e300, -1e300};
  unsigned long long dg = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    double c[3];
    rm_centre(&prob->lat, obj->global_id, c);
    for (int d = 0; d < 3; ++d) c[d] += shift[d];
    acc[0] += obj->tris.n_local;
    acc[1] += obj->verts.n_local;
    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!rm_counts_edge(obj, h)) continue;
      acc[2] += 1.0;
      const double l = MFTLGetEdgeLength(obj, h);
      if (l > params->max_edge_length || l < params->min_edge_length) acc[3] += 1.0;
    }
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      double d[3];
      MFTLGetMinimumImageVector(&ens->domain, d, c, &obj->verts.pos[3 * v]);
      const double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) / prob->radius;
      acc[4] += (r - 1.0) * (r - 1.0);
      mx[0] = MFTL_MAX(mx[0], fabs(r - 1.0));
      mx[3] = MFTL_MAX(mx[3], r);
      mx[4] = MFTL_MAX(mx[4], -r);
      unsigned long long h = (unsigned long long) (unsigned) obj->verts.rec[v].global_id * 0x9E3779B97F4A7C15ull ^
                             (unsigned long long) (unsigned) obj->global_id * 0xC2B2AE3D27D4EB4Full;
      for (int k = 0; k < 3; ++k)
      {
        unsigned long long bits;
        memcpy(&bits, &obj->verts.pos[3 * v + k], sizeof bits);
        h ^= bits + 0x165667B19E3779F9ull + (h << 6) + (h >> 2);
      }
      dg ^= h;
    }
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double C[3][3];
      MFTLGetTriangleCorners(obj, t, C);
      double sq = 0.0;
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k)
        {
          const double e = C[(j + 1) % 3][k] - C[j][k];
          sq += e * e;
        }
      const double q = (sq > 0.0) ? MFTL_FOUR_SQRT3 * obj->tris.area[t] / sq : 0.0;
      acc[5] += q;
      mx[2] = MFTL_MAX(mx[2], -q);  // max of -q is -min q
    }
  }

  /* Volumes per body, counted on the body's lowest holder. */
  if (v0 > 0.0)
  {
    const int b = ens->n_local_objects;
    MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) MFTLMallocOrDie((size_t) MFTL_MAX(b, 1) * sizeof(MFTL_BodyIntegrals), "remesh scaling integrals");
    sc_require(MFTLEnsembleBodyIntegrals(ens, bi), "body integrals");
    for (int o = 0; o < b; ++o)
      if (bi[o].lowest_holder == ens->domain.rank)
      {
        acc[6] += bi[o].volume;
        mx[1] = MFTL_MAX(mx[1], fabs(bi[o].volume / v0 - 1.0));
      }
    free(bi);
  }

  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  double g[7], gmax[5], own_max = 0.0;
  unsigned long long gd = 0;
  MPI_Allreduce(acc, g, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(mx, gmax, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&acc[0], &own_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&dg, &gd, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, MPI_COMM_WORLD);

  s->tris = g[0];
  s->verts = g[1];
  s->edges = g[2];
  s->oob = g[3];
  s->dist_rms = sqrt(g[4] / MFTL_MAX(g[1], 1.0));
  s->dist_max = gmax[0];
  s->qmean = g[5] / MFTL_MAX(g[0], 1.0);
  s->qmin = -gmax[2];
  s->volume = g[6];
  s->worst_dv = gmax[1];
  s->r_max = gmax[3];
  s->r_min = -gmax[4];
  s->tris_rank_max = own_max;
  s->tris_rank_mean = g[0] / (double) nproc;
  s->digest = gd;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The run
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Per-step timers: the first RT_LIB are measured here, the rest are the remesher's own. */
enum
{
  RT_STEP,
  RT_ADVECT,
  RT_GHOST,
  RT_GEOM,
  RT_REMESH,
  RT_REDIST,
  RT_LIB,  // ---- from here: MFTL_RemeshStats
  RT_MIGRATE = RT_LIB,
  RT_SWEEP,
  RT_FREEZE,
  RT_REFERENCE,
  RT_PROJECTION,
  RT_REDUCE,
  RT_RGEOM,
  N_RT
};
static const char *const RT_KEY[N_RT] = {"step",  "advect", "ghost",     "geom",       "remesh", "redist", "migrate",
                                         "sweep", "freeze", "reference", "projection", "ids",    "rgeom"};

/** @brief What one trajectory produced. */
typedef struct
{
  double t[N_RT];  ///< Seconds, summed over the trajectory's steps.
  double phases, migrations, frozen, load_imb, idle, ops, calls, not_converged, missed, not_centred, ref_mb;
} rm_traj;

/**
 * @brief The target edge length: the mean edge of body 0 as generated, computed
 *        on rank 0 alone and broadcast, so that it is the same bits at every P
 *        (a reduced mean is not: its summation order follows the partition).
 * @param ens  The ensemble (for its domain).
 * @param prob The problem.
 * @param opt  The options.
 * @return The target.
 */
static double rm_target_length(const MFTL_Ensemble *ens, const rm_problem *prob, const rm_options *opt)
{
  double target = 0.0;
  if (ens->domain.rank == 0)
  {
    int one[3] = {1, 1, 1};
    double gmin[3], gmax[3], c[3];
    bool periodic[3];
    for (int d = 0; d < 3; ++d)
    {
      gmin[d] = prob->gmin[d];
      gmax[d] = prob->gmax[d];
      periodic[d] = true;
    }
    MFTL_Ensemble *self = MFTLEnsembleCreate(MPI_COMM_SELF, one, gmin, gmax, periodic, 1);
    MFTL_Object *body = MFTLObjectCreate(0);
    sc_require(MFTLEnsembleAddObject(self, body), "reference body");
    rm_centre(&prob->lat, 0, c);
    sc_require(MFTLGenerateSphereLocal(body, c, prob->radius, opt->sc.refine), "reference body");
    target = MFTLMeanEdgeLength(body);
    MFTLEnsembleDestroy(&self);
  }
  MPI_Bcast(&target, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return target;
}

/**
 * @brief Generates the problem, runs one spanning mode (or the dispersed driver)
 *        over the warm-up and the timed trajectories, checks, and writes the
 *        report section and its @summary line. Collective.
 * @param prob    The problem.
 * @param opt     The options.
 * @param mode    The spanning mode (ignored by the dispersed driver).
 * @param n_nodes Nodes in the job, for the summary.
 * @param rpn_max Most ranks on one node, for the summary.
 * @return 0, or 1 if a check failed.
 */
static int rm_run_mode(const rm_problem *prob, const rm_options *opt, int mode, int n_nodes, int rpn_max)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  const char *const mode_name = (prob->regime == RM_DISPERSED) ? "dispersed" : (mode == MFTL_REMESH_SPANNING_NESTED ? "nested" : "once");

  bool periodic[3] = {true, true, true};
  int cuts[3] = {opt->sc.cuts[0], opt->sc.cuts[1], opt->sc.cuts[2]};
  double gmin[3], gmax[3], h_min = 1e300;
  for (int d = 0; d < 3; ++d)
  {
    gmin[d] = prob->gmin[d];
    gmax[d] = prob->gmax[d];
    h_min = MFTL_MIN(h_min, (gmax[d] - gmin[d]) / (double) cuts[d]);
  }
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, gmin, gmax, periodic, MFTL_MAX(prob->n_bodies / nproc, 16));
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  /* The redistribution harness's bounded-batch initialisation. */
  sc_problem sp;
  memset(&sp, 0, sizeof sp);
  sp.mode = prob->mode;
  sp.n_bodies = prob->n_bodies;
  memcpy(sp.gmin, prob->gmin, sizeof sp.gmin);
  memcpy(sp.gmax, prob->gmax, sizeof sp.gmax);
  sp.radius = prob->radius;
  sp.spacing = prob->lat.a;
  sp.centre = rm_centre;
  sp.centre_ctx = &prob->lat;
  const sc_setup setup = sc_initialize(ens, &sp, &opt->sc);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLEnsembleUpdateGlobalCounts(ens);

  /* Hold the generated resolution while the bodies deform. */
  const double target = rm_target_length(ens, prob, opt);
  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, opt->iterations, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = opt->smooth;
  params.parallel_phase_budget = opt->budget;
  params.projection = (MFTL_RemeshProjection) opt->projection;
  params.parallel_strategy = (prob->regime == RM_DISPERSED) ? MFTL_REMESH_PARALLEL_DISPERSED : MFTL_REMESH_PARALLEL_SPANNING;
  params.spanning_mode = (MFTL_RemeshSpanningMode) mode;
  if (opt->sizing == MFTL_REMESH_SIZING_CURVATURE) MFTLRemeshUseCurvatureSizing(&params, 0.02 * target, 0.25 * target, target);

  /* The generated volume of one body (all are alike). */
  double v0 = 0.0;
  {
    const int b = ens->n_local_objects;
    MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) MFTLMallocOrDie((size_t) MFTL_MAX(b, 1) * sizeof(MFTL_BodyIntegrals), "remesh scaling integrals");
    sc_require(MFTLEnsembleBodyIntegrals(ens, bi), "body integrals");
    double acc[2] = {0.0, 0.0};
    for (int o = 0; o < b; ++o)
      if (bi[o].lowest_holder == rank) acc[0] += bi[o].volume, acc[1] += 1.0;
    free(bi);
    MPI_Allreduce(MPI_IN_PLACE, acc, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    v0 = acc[0] / MFTL_MAX(acc[1], 1.0);
  }
  const double zero[3] = {0.0, 0.0, 0.0};
  rm_state s0;
  rm_measure(ens, prob, zero, &params, v0, &s0);

  rm_flow flow;
  rm_flow_init(prob, opt, &flow);
  const int ns = opt->sc.steps;
  const double dt = flow.period / (double) ns;

  if (rank == 0)
  {
    printf("   [%s] setup %.1f s; target edge %.5g; R/h %.3f; %.0f triangles, %.0f per rank; warm-up %d trajectory(ies) ...\n", mode_name, setup.wall_s, target,
           prob->radius / h_min, s0.tris, s0.tris / nproc, opt->warmup);
    fflush(stdout);
  }

  rm_traj best;
  memset(&best, 0, sizeof best);
  best.t[RT_STEP] = -1.0;
  rm_state last = s0, mid;
  memset(&mid, 0, sizeof mid);
  double *step_ms = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "remesh scaling steps");
  double *best_step = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "remesh scaling steps");
  int failures = 0, completed = 0;
  bool budget_stop = false;
  const double mode_start = MPI_Wtime();
  const int n_modes = (prob->regime == RM_DISPERSED) ? 1 : opt->n_modes;

  for (int traj = 0; traj < opt->warmup + opt->trajectories; ++traj)
  {
    const bool timed = (traj >= opt->warmup);
    /* Between trajectories only, so a partial trajectory is never reported. The
     * job's budget is split evenly over the modes it runs. */
    if (timed && completed > 0 && opt->sc.max_seconds > 0.0)
    {
      double e = MPI_Wtime() - mode_start;
      MPI_Allreduce(MPI_IN_PLACE, &e, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      if (e * (double) (traj + 1) / (double) traj > opt->sc.max_seconds / (double) n_modes)
      {
        budget_stop = true;
        break;
      }
    }
    rm_traj tr;
    memset(&tr, 0, sizeof tr);
    char vtk_name[160];
    const bool vtk = (opt->vtk_every > 0 && traj == 0);
    if (vtk)
    {
      snprintf(vtk_name, sizeof vtk_name, "%s_%s_%04d", opt->sc.label, mode_name, 0);
      sc_require(MFTLWriteVtu(ens, vtk_name, MFTL_NULL), "vtu output");
    }
    for (int st = 0; st < ns; ++st)
    {
      const double t = st * dt;
      double ph[RT_LIB];
      for (int k = 0; k < RT_LIB; ++k) ph[k] = 0.0;
      MPI_Barrier(MPI_COMM_WORLD);
      const double s_begin = MPI_Wtime();

      double t0 = MPI_Wtime();
      rm_advect(ens, &flow, t, dt);
      ph[RT_ADVECT] = MPI_Wtime() - t0;
      t0 = MPI_Wtime();
      sc_require(MFTLUpdateGhostPositions(ens), "ghost positions");
      ph[RT_GHOST] = MPI_Wtime() - t0;
      t0 = MPI_Wtime();
      for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
      ph[RT_GEOM] = MPI_Wtime() - t0;

      MFTL_RemeshStats rs;
      memset(&rs, 0, sizeof rs);
      const bool remeshed = ((st + 1) % opt->remesh_every == 0);
      t0 = MPI_Wtime();
      if (remeshed)
      {
        sc_require(MFTLRemeshEnsemble(ens, &params, &rs), "remesh");
        ph[RT_REMESH] = MPI_Wtime() - t0;
      }
      else
      {
        sc_require(MFTLRedistribute(ens), "redistribution");
        for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
        ph[RT_REDIST] = MPI_Wtime() - t0;
      }
      ph[RT_STEP] = MPI_Wtime() - s_begin;

      /* Outside the step's clock: maxima over ranks, and the remesher's
       * rank-local counts. Its own timers are maxima already, except the
       * projection's two, which are rank-local. */
      double loc[RT_LIB + 3], glob[RT_LIB + 3];
      for (int k = 0; k < RT_LIB; ++k) loc[k] = ph[k];
      loc[RT_LIB + 0] = rs.t_reference;
      loc[RT_LIB + 1] = rs.t_projection;
      loc[RT_LIB + 2] = rs.reference_bytes;
      double cnt[2] = {(double) rs.n_split + rs.n_collapse + rs.n_flip, (double) rs.n_projection_missed}, gcnt[2];
      MPI_Allreduce(loc, glob, RT_LIB + 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      MPI_Allreduce(cnt, gcnt, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      for (int k = 0; k < RT_LIB; ++k) tr.t[k] += glob[k];
      step_ms[st] = 1e3 * glob[RT_STEP];
      if (remeshed)
      {
        tr.t[RT_MIGRATE] += rs.t_migrate;
        tr.t[RT_SWEEP] += rs.t_sweep;
        tr.t[RT_FREEZE] += rs.t_freeze;
        tr.t[RT_REFERENCE] += glob[RT_LIB + 0];
        tr.t[RT_PROJECTION] += glob[RT_LIB + 1];
        tr.t[RT_REDUCE] += rs.t_reduce;
        tr.t[RT_RGEOM] += rs.t_geometry;
        tr.ref_mb = MFTL_MAX(tr.ref_mb, glob[RT_LIB + 2] / (1024.0 * 1024.0));
        tr.phases += rs.n_phases;
        tr.migrations += rs.n_migrations;
        tr.frozen += rs.frozen_fraction;
        tr.load_imb += rs.load_imbalance;
        tr.idle += rs.n_ranks_idle;
        tr.not_converged += rs.converged ? 0.0 : 1.0;
        tr.not_centred += rs.n_bodies_not_centred;
        tr.ops += gcnt[0];
        tr.missed += gcnt[1];
        tr.calls += 1.0;
      }

      if (vtk && (st + 1) % opt->vtk_every == 0)
      {
        snprintf(vtk_name, sizeof vtk_name, "%s_%s_%04d", opt->sc.label, mode_name, st + 1);
        sc_require(MFTLWriteVtu(ens, vtk_name, MFTL_NULL), "vtu output");
      }

      if (st == ns / 2 - 1)  // the step that ends at T/2
      {
        const double tau = rm_tau(&flow, (st + 1) * dt);
        const double shift[3] = {flow.drift[0] * tau, flow.drift[1] * tau, flow.drift[2] * tau};
        rm_measure(ens, prob, shift, &params, 0.0, &mid);
      }
    }

    rm_state sf;
    rm_measure(ens, prob, zero, &params, v0, &sf);
    last = sf;
    const double chi = sf.verts - sf.edges + sf.tris;
    const bool bad = !isfinite(sf.volume) || fabs(chi - 2.0 * prob->n_bodies) > 0.5 || fabs(sf.volume / s0.volume - 1.0) > opt->max_dv || !(sf.qmin > 0.0);
    if (bad) failures++;
    if (rank == 0)
    {
      printf(
          "   [%s] trajectory %d%s: %.2f ms/step; at T/2 r/R %.3f..%.3f; at T %.0f tris, V/V0-1 %+.4f %% (worst body %.4f %%), "
          "dist rms %.4f %% max %.3f %% of R, q min %.3f mean %.4f%s\n",
          mode_name, traj + 1, timed ? "" : " (warm-up)", 1e3 * tr.t[RT_STEP] / ns, mid.r_min, mid.r_max, sf.tris, 100.0 * (sf.volume / s0.volume - 1.0),
          100.0 * sf.worst_dv, 100.0 * sf.dist_rms, 100.0 * sf.dist_max, sf.qmin, sf.qmean, bad ? "  ** CHECK FAILED **" : "");
      fflush(stdout);
    }
    if (!timed) continue;
    completed++;
    if (best.t[RT_STEP] < 0.0 || tr.t[RT_STEP] < best.t[RT_STEP])
    {
      best = tr;
      memcpy(best_step, step_ms, (size_t) ns * sizeof(double));
    }
  }

  /* ── The report section and the machine-readable line ─────────────────── */
  long rss = sc_peak_rss(), rss_max = 0;
  MPI_Reduce(&rss, &rss_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  const int exit_code = (failures || completed == 0) ? 1 : 0;
  if (rank == 0)
  {
    const double S = (double) ns, C = MFTL_MAX(best.calls, 1.0), ms = 1e3 / S;
    const double chi = last.verts - last.edges + last.tris;
    const char *sizing = (opt->sizing == MFTL_REMESH_SIZING_CURVATURE) ? "curvature" : "uniform";
    const char *proj = (opt->projection == MFTL_REMESH_PROJECT_PHONG) ? "phong" : "none";
    if (completed > 0) qsort(best_step, (size_t) ns, sizeof(double), sc_cmp_double);
    const double p50 = completed ? sc_quantile(best_step, ns, 0.5) : 0.0, p90 = completed ? sc_quantile(best_step, ns, 0.9) : 0.0;
    const double pmax = completed ? best_step[ns - 1] : 0.0;
    sc_out("\n[%s] %s regime, %d bodies, R/h %.3f, refine %d, %s sizing, projection %s, budget %d, iterations %d x %d sweeps, %d of %d timed trajectories\n",
           mode_name, RM_REGIME_NAME[prob->regime], prob->n_bodies, prob->radius / h_min, opt->sc.refine, sizing, proj, opt->budget, opt->iterations,
           opt->smooth, completed, opt->trajectories);
    sc_out("  ms per step, max over ranks   step %.3f   advect %.3f   ghost %.3f   geom %.3f   remesh %.3f   redistribute %.3f\n", best.t[RT_STEP] * ms,
           best.t[RT_ADVECT] * ms, best.t[RT_GHOST] * ms, best.t[RT_GEOM] * ms, best.t[RT_REMESH] * ms, best.t[RT_REDIST] * ms);
    sc_out("  inside the remesh call        migrate %.3f   sweep %.3f   freeze %.3f   reference %.3f   projection %.3f   ids %.3f   geometry %.3f\n",
           best.t[RT_MIGRATE] * ms, best.t[RT_SWEEP] * ms, best.t[RT_FREEZE] * ms, best.t[RT_REFERENCE] * ms, best.t[RT_PROJECTION] * ms,
           best.t[RT_REDUCE] * ms, best.t[RT_RGEOM] * ms);
    sc_out("  per remesh call               %.2f placements, %.2f migrations, frozen share %.1f %%, %.0f operations, load imbalance %.3f, idle ranks %.1f\n",
           best.phases / C, best.migrations / C, 100.0 * best.frozen / C, best.ops / C, best.load_imb / C, best.idle / C);
    sc_out("  step time distribution        p50 %.3f   p90 %.3f   max %.3f ms\n", p50, p90, pmax);
    sc_out("  partition at T                owned triangles per rank mean %.0f, max %.0f (imbalance %.3f); projection reference max %.2f MB/rank\n",
           last.tris_rank_mean, last.tris_rank_max, last.tris_rank_max / MFTL_MAX(last.tris_rank_mean, 1.0), best.ref_mb);
    sc_out("  deformation at T/2            r/R %.3f .. %.3f\n", mid.r_min, mid.r_max);
    sc_out("  accuracy at T                 V/V0 - 1 %+.4f %% (worst body %.4f %%), distance to the sphere rms %.4f %% max %.3f %% of R\n",
           100.0 * (last.volume / s0.volume - 1.0), 100.0 * last.worst_dv, 100.0 * last.dist_rms, 100.0 * last.dist_max);
    sc_out("                                quality min %.4f mean %.5f, edges out of band %.3f %%, triangles %.0f (generated %.0f)\n", last.qmin, last.qmean,
           100.0 * last.oob / MFTL_MAX(last.edges, 1.0), last.tris, s0.tris);
    sc_out(
        "  checks                        Euler %.0f (expected %d) %s, passes not converged %.0f of %.0f, projection missed %.0f, bodies not centred %.0f: %s\n",
        chi, 2 * prob->n_bodies, fabs(chi - 2.0 * prob->n_bodies) < 0.5 ? "ok" : "FAILED", best.not_converged, best.calls, best.missed, best.not_centred,
        exit_code ? "FAILED -- do not use these numbers" : "ok");
    sc_out("  final digest                  %016llx%s\n", last.digest,
           prob->regime == RM_DISPERSED ? "  (dispersed: the same at every rank count of a strong sweep)" : "");
    sc_out("@summary mode=%s regime=%s spanning=%s np=%d nodes=%d rpn=%d cuts=%dx%dx%d bodies=%d refine=%d R_over_h=%.5g tris=%.0f tris_rank=%.0f steps=%d",
           prob->mode, RM_REGIME_NAME[prob->regime], mode_name, nproc, n_nodes, rpn_max, cuts[0], cuts[1], cuts[2], prob->n_bodies, opt->sc.refine,
           prob->radius / h_min, last.tris, last.tris_rank_mean, ns);
    sc_out(" ms_step=%.6g", best.t[RT_STEP] * ms);
    for (int k = 1; k < N_RT; ++k) sc_out(" %s=%.6g", RT_KEY[k], best.t[k] * ms);
    sc_out(" step_p50=%.6g step_p90=%.6g step_max=%.6g", p50, p90, pmax);
    sc_out(" phases=%.4g migrations=%.4g frozen_pct=%.4g ops_call=%.6g load_imb=%.4g idle=%.4g imbalance=%.4g ref_mb=%.4g", best.phases / C,
           best.migrations / C, 100.0 * best.frozen / C, best.ops / C, best.load_imb / C, best.idle / C,
           last.tris_rank_max / MFTL_MAX(last.tris_rank_mean, 1.0), best.ref_mb);
    sc_out(" stretch_max=%.5g stretch_min=%.5g dv_pct=%.6g dv_worst_pct=%.6g dist_rms_pct=%.6g dist_max_pct=%.6g qmin=%.5g qmean=%.6g oob_pct=%.5g", mid.r_max,
           mid.r_min, 100.0 * (last.volume / s0.volume - 1.0), 100.0 * last.worst_dv, 100.0 * last.dist_rms, 100.0 * last.dist_max, last.qmin, last.qmean,
           100.0 * last.oob / MFTL_MAX(last.edges, 1.0));
    sc_out(" euler=%.0f not_converged=%.0f calls=%.0f missed=%.0f not_centred=%.0f digest=%016llx trajectories=%d budget_stop=%d init_s=%.4g rss_rank_mb=%.5g",
           chi, best.not_converged, best.calls, best.missed, best.not_centred, last.digest, completed, budget_stop ? 1 : 0, setup.wall_s,
           (double) rss_max / (1024.0 * 1024.0));
    sc_out(" sizing=%s proj_mode=%s iterations=%d smooth=%d budget=%d strain=%.4g drift=%.4g remesh_every=%d ok=%d\n", sizing, proj, opt->iterations,
           opt->smooth, opt->budget, opt->strain, opt->drift, opt->remesh_every, exit_code ? 0 : 1);
    fflush(stdout);
  }
  free(step_ms);
  free(best_step);
  MFTLEnsembleDestroy(&ens);
  return exit_code;
}

/**
 * @brief Runs every requested mode of one problem; collective.
 * @param prob The problem.
 * @param opt  The resolved options.
 * @return Process exit code: 0 on success, 1 if a check failed, 2 on refusal.
 */
static int rm_run(const rm_problem *prob, const rm_options *opt)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const long tri_per_body = 20L * (1L << (2 * opt->sc.refine));
  const long vert_per_body = 10L * (1L << (2 * opt->sc.refine)) + 2L;
  if ((long long) prob->n_bodies * tri_per_body > INT_MAX / 2)
  {
    if (rank == 0) fprintf(stderr, "refusing: the problem exceeds half the int-sized entity counts (remeshing can grow it).\n");
    return 2;
  }
  const int batch = MFTL_MIN(prob->n_bodies, opt->sc.init_batch);
  const double gen_mb = (double) batch * ((double) tri_per_body * 140.0 + (double) vert_per_body * 64.0) / (1024.0 * 1024.0);
  if (gen_mb > opt->sc.max_gen_mb)
  {
    if (rank == 0) fprintf(stderr, "refusing: initialisation batch %.0f MiB > --max-gen-mb %.0f; lower --init-batch.\n", gen_mb, opt->sc.max_gen_mb);
    return 2;
  }

  int report_error = 0;
  if (rank == 0 && opt->sc.report[0])
  {
    sc_report_fp = fopen(opt->sc.report, "w");
    if (!sc_report_fp)
    {
      fprintf(stderr, "cannot open report %s: %s\n", opt->sc.report, strerror(errno));
      report_error = 1;
    }
  }
  MPI_Bcast(&report_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (report_error) return 2;

  /* Placement on the machine, checked against what was submitted. */
  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
  int node_rank = 0, node_size = 1, n_nodes = 0, rpn_max = 0;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);
  const int leader = (node_rank == 0) ? 1 : 0;
  MPI_Allreduce(&leader, &n_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&node_size, &rpn_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Comm_free(&node_comm);
  const char *expected_ranks = getenv("NRANKS");
  const char *expected_nodes = getenv("MFTL_EXPECT_NODES");
  const char *expected_rpn = getenv("MFTL_EXPECT_RPN");
  if ((expected_ranks && atoi(expected_ranks) != nproc) || (expected_nodes && atoi(expected_nodes) != n_nodes) ||
      (expected_rpn && rpn_max > atoi(expected_rpn)))
  {
    if (rank == 0) fprintf(stderr, "MPI placement does not match the submitted rank/node counts; check site.sh's launcher.\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (rank == 0)
  {
    sc_out("== MFTL remeshing %s scaling, %s regime: %d ranks (%dx%dx%d) on %d node(s), %d bodies x %ld triangles ==\n", prob->mode,
           RM_REGIME_NAME[prob->regime], nproc, opt->sc.cuts[0], opt->sc.cuts[1], opt->sc.cuts[2], n_nodes, prob->n_bodies, tri_per_body);
    sc_out("   placement: %s\n", prob->placement);
    sc_out("   flow: strain %.3g, drift %.3g x (1.5, 1, 0.5) a out and back, %d steps per trajectory, remesh every %d step(s)\n", opt->strain, opt->drift,
           opt->sc.steps, opt->remesh_every);
    sc_out("   label %s, git %s\n", opt->sc.label, opt->sc.git);
    fflush(stdout);
  }

  int code = 0;
  const int n_modes = (prob->regime == RM_DISPERSED) ? 1 : opt->n_modes;
  for (int m = 0; m < n_modes; ++m) code |= rm_run_mode(prob, opt, opt->modes[m], n_nodes, rpn_max);

  if (rank == 0 && sc_report_fp)
  {
    const int write_error = ferror(sc_report_fp);
    const int close_error = fclose(sc_report_fp);
    sc_report_fp = MFTL_NULL;
    if (write_error || close_error)
    {
      fprintf(stderr, "could not finish report %s\n", opt->sc.report);
      code = 2;
    }
    else
      printf("report written to %s\n", opt->sc.report);
  }
  MPI_Bcast(&code, 1, MPI_INT, 0, MPI_COMM_WORLD);
  return code;
}

#endif  // MFTL_REMESH_SCALING_COMMON_H
