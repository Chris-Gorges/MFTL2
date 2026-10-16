/*
 * @Author: Christian Gorges
 * @Date: 2026-10-13
 *
 * @brief What strong_scaling.c and weak_scaling.c share: the instrumented
 *        timestep, the run harness, the redistribution profile and the report.
 *
 * The two drivers differ ONLY in how the problem grows with P -- the body count,
 * the domain and where the bodies start. Everything that is measured, and how,
 * lives here, so a strong point and a weak point are the same instrument.
 *
 * -- The timestep (the documented MFTL loop with redistribution) ----------------
 *
 *   1. advect      move OWNED vertices, wrap into the periodic box
 *   2. ghost       MFTLUpdateGhostPositions          (neighbour exchange)
 *   3. geom        MFTLComputeTriangleGeometry over ALL triangles, ghosts too:
 *                  redistribution routes ghosts by their cached centre
 *   4. redist      MFTLRedistribute                  (the phase under study)
 *   5. postgeom    MFTLComputeTriangleGeometry again: redistribution does not
 *                  ship geometry, so a solver owes this pass afterwards
 *   6. normals     MFTLEnsembleUpdateVertexNormals   (local pass + one ghost
 *                  exchange that makes ghost normals exact)
 *
 * -- The motion, and why the default is a period-locked uniform flow -----------
 *
 *   A scaling curve is only about the library if every point does the same work
 *   per rank. Random motion breaks that: a straight DRIFT in random directions
 *   diffuses the bodies until the count per rank is Poisson (sigma 12 % at 64 per
 *   rank), and the busiest rank -- which every collective waits for -- gets
 *   busier with P by extreme-value statistics alone. Measured at P = 27: drift
 *   1.02 -> 1.25 structural imbalance; a random OSCILLATION 1.02 -> 1.16 at every
 *   half period; even a common flow 1.02 -> 1.16 while the bodies are jittered.
 *
 *   flow (default)  every body translates by (3,2,1) body spacings over each
 *                   trajectory of --steps timesteps; --cycles trajectories form
 *                   one timed repeat. On a full UNJITTERED cubic lattice the
 *                   geometry repeats up to body labels and floating-point error.
 *                   The minimum over repeats estimates attainable performance. In weak mode
 *                   every rank box is an exact copy of every other, so every rank
 *                   does identical work on every step at every P; in strong mode
 *                   the same holds whenever the grid divides the domain into a
 *                   whole number of spacings (the default full-node grids through
 *                   P=576 with the 48^3 domain). The odd ratios 3:2:1 stagger the x,
 *                   y and z plane crossings. Per-step displacement: |(3,2,1)| /
 *                   --steps spacings, so --steps also sets the speed.
 *   osc             each body oscillates about its start along its own direction
 *                   with its own phase (amplitude --amplitude spacings, peak speed
 *                   --speed). Bounded, but periodic imbalance.
 *   drift           straight lines at --speed in random directions: the mixed,
 *                   imbalanced case. Read its curve against the imbalance the
 *                   report prints.
 *
 * -- What is profiled --------------------------------------------------------
 *
 *   * per phase: mean / min / max over ranks. max/mean is the imbalance; for a
 *     phase that ends in a collective it is a LOWER bound, because the collective
 *     equalises finish times -- the waiting is charged to whichever bucket holds
 *     the collective.
 *   * inside MFTLRedistribute: the 18 disjoint buckets of MFTL_RedistributeStats,
 *     split into LOCAL WORK (should fall like 1/P in strong scaling, stay flat in
 *     weak) and COMMUNICATION + SYNC (latency, volume, and absorbed imbalance).
 *     Needs the library built with -DMFTL_COMM_PROFILE=ON; without it every
 *     t_* bucket is zero and the report says so.
 *   * redistribution traffic: entities scanned / kept / received, owner changes,
 *     bytes and messages (mean AND busiest rank), neighbour vs global route,
 *     bodies skipped by the unchanged-body vote, object churn, allocation churn,
 *     and the workspace split by what it scales with (per body, per RANK of the
 *     job, per entity) -- the per-rank term is the one that grows with P.
 *   * inside the ghost syncs: the 8 buckets of the position AND normal exchanges.
 *   * per step: the busiest rank's step time, as p50 / p90 / max over steps, so a
 *     growing tail can be told from a rising floor.
 *   * partition: owned / ghost triangles, bodies held, graph degree, all min /
 *     mean / max over ranks; memory: peak RSS per rank and per NODE.
 *   * the instrument itself: clock reads x measured MPI_Wtime cost, phases below
 *     100 x MPI_Wtick, and whether the build was optimised with profiling on.
 *
 * -- What is checked (a scaling number from a broken run is worthless) ---------
 *
 *   rigid translation conserves the surface area to round-off and redistribution
 *   conserves entities exactly, so after every repeat the global triangle and
 *   vertex counts must be unchanged and the area within 1e-9. Orphaned ghosts and
 *   centres the redistribution had to recompute must both be zero. Any failure
 *   makes the run exit non-zero and is printed in the report.
 *
 * -- Rules kept from examples/parallel_advection_scaling.c ---------------------
 *
 *   setup (batched rank-0 generation + distribution) is reported, never timed into
 *   the loop; warm-up steps run before the clock starts; the loop is repeated and
 *   the FASTEST repeat is reported with all others listed; nothing in the harness
 *   in the timed loop is O(global bodies) per rank (velocities and centres are pure
 *   functions of the body id, splitmix64-keyed, identical on every platform).
 *
 * Must be the FIRST include of a translation unit: it sets the POSIX feature
 * macro the Linux headers need under -std=c11.
 */

#ifndef MFTL_SCALING_COMMON_H
#define MFTL_SCALING_COMMON_H

#if !defined(__APPLE__) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include <math.h>
#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <unistd.h>
#endif

#include <mpi.h>

#include "mftl_core.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Tables: phases, redistribution buckets, counters
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The phases of one timestep, in the order they run. */
enum
{
  PH_ADVECT = 0,
  PH_GHOST,
  PH_GEOM,
  PH_REDIST,
  PH_POST,
  PH_NORMALS,
  N_PHASE
};

/** @brief Summary keys of the phases. */
static const char *const SC_PHASE_KEY[N_PHASE] = {"advect", "ghost", "geom", "redist", "postgeom", "normals"};

/** @brief Report labels of the phases. */
static const char *const SC_PHASE_LABEL[N_PHASE] = {"1 advection", "2 ghost sync", "3 tri geometry (all)", "4 REDISTRIBUTE", "5 tri geometry (post)",
                                                    "6 vertex normals"};

/**
 * @brief The disjoint timer buckets of MFTLRedistribute(), in call order:
 *        X(summary key, struct field, report label, 1 = communication or sync).
 *
 * Communication buckets also absorb load imbalance: a rank that reaches a
 * collective early charges its neighbours' lateness to itself.
 */
#define SC_RBUCKETS(X)                                                    \
  X(setup, t_setup, "setup: O(nproc) workspace resets", 0)                \
  X(owners, t_owners, "owner classification", 0)                         \
  X(masks, t_masks, "keep/dedup mask clearing", 0)                        \
  X(dest, t_destinations, "destination analysis", 0)                      \
  X(route, t_route, "route Allreduce (+imbalance)", 1)                    \
  X(skipvote, t_skip_vote, "skip vote reduction (+imbalance)", 1)         \
  X(pack, t_pack, "pack payloads", 0)                                     \
  X(xcounts, t_exchange_counts, "exchange: counts (+imbalance)", 1)       \
  X(xpost, t_exchange_data, "exchange: post payloads + flatten", 1)       \
  X(wait, t_wait, "exchange: wait for payloads", 1)                       \
  X(group, t_group, "group arrivals by body", 0)                          \
  X(reserve, t_reserve, "pool growth / shrink", 0)                        \
  X(kgather, t_kept_gather, "kept: gather", 0)                            \
  X(kplace, t_kept_place, "kept: place", 0)                               \
  X(relink, t_relink, "kept: relink half-edges", 0)                       \
  X(arrivals, t_arrivals, "write arrivals", 0)                            \
  X(twins, t_twins, "twin repair", 0)                                     \
  X(teardown, t_teardown, "teardown", 0)

/** @brief The timer buckets of the ghost exchanges (positions AND normals). */
#define SC_GBUCKETS(X)                                    \
  X(collect, g_t_collect, "collect requests", 0)          \
  X(counts, g_t_counts, "exchange counts", 1)             \
  X(requests, g_t_requests, "exchange requests", 1)       \
  X(table, g_t_table, "owner id table", 0)                \
  X(answer, g_t_answer, "answer requests", 0)             \
  X(replies, g_t_replies, "exchange replies", 1)          \
  X(apply, g_t_apply, "apply to ghost slots", 0)          \
  X(teardown, g_t_teardown, "teardown", 0)

/** @brief The `long` counters of MFTL_RedistributeStats that the report reads. */
#define SC_RCOUNTERS(X)                                 \
  X(calls, n_calls)                                     \
  X(verts_scanned, n_verts_scanned)                     \
  X(tris_scanned, n_tris_scanned)                       \
  X(verts_moved, n_verts_owner_changed)                 \
  X(tris_moved, n_tris_owner_changed)                   \
  X(verts_kept, n_verts_kept)                           \
  X(tris_kept, n_tris_kept)                             \
  X(verts_recv, n_verts_received)                       \
  X(tris_recv, n_tris_received)                         \
  X(destinations, n_destinations)                       \
  X(messages, n_messages)                               \
  X(bytes_sent, n_bytes_sent)                           \
  X(bytes_recv, n_bytes_recv)                           \
  X(route_neighbor, n_route_neighbor)                   \
  X(route_global, n_route_global)                       \
  X(bodies_visited, n_objects_visited)                  \
  X(bodies_empty, n_objects_empty_local)                \
  X(bodies_skipped, n_objects_skipped)                  \
  X(bodies_reclassified, n_objects_reclassified)        \
  X(bodies_settled, n_objects_settled)                  \
  X(bodies_shrunk, n_objects_shrunk)                    \
  X(bytes_reclaimed, n_bytes_reclaimed)                 \
  X(bodies_created, n_objects_created)                  \
  X(bodies_destroyed, n_objects_destroyed)              \
  X(centres_recomputed, n_objects_centres_recomputed)   \
  X(allocs, n_allocs)                                   \
  X(bytes_allocated, n_bytes_allocated)                 \
  X(gid_table_bytes, n_gid_table_bytes)                 \
  X(ws_per_body, ws_bytes_per_body)                     \
  X(ws_per_rank, ws_bytes_per_rank)                     \
  X(ws_per_entity, ws_bytes_per_entity)                 \
  X(gid_direct, n_gid_table_direct)                     \
  X(gid_sparse, n_gid_table_sparse)                     \
  X(dedup_overflow, n_dedup_overflow)                   \
  X(clock_reads, n_clock_reads)                         \
  X(g_calls, g_n_calls)                                 \
  X(g_requested, g_n_ghosts_requested)                  \
  X(g_orphaned, g_n_orphaned)                           \
  X(g_bytes_sent, g_n_bytes_sent)                       \
  X(g_bytes_recv, g_n_bytes_recv)                       \
  X(g_allocs, g_n_allocs)                               \
  X(g_bytes_allocated, g_n_bytes_allocated)

enum
{
#define SC_X(key, field, label, comm) RB_##key,
  SC_RBUCKETS(SC_X)
#undef SC_X
  N_RB
};

enum
{
#define SC_X(key, field, label, comm) GB_##key,
  SC_GBUCKETS(SC_X)
#undef SC_X
  N_GB
};

enum
{
#define SC_X(key, field) RC_##key,
  SC_RCOUNTERS(SC_X)
#undef SC_X
  N_RC
};

static const size_t SC_RB_OFF[N_RB] = {
#define SC_X(key, field, label, comm) offsetof(MFTL_RedistributeStats, field),
    SC_RBUCKETS(SC_X)
#undef SC_X
};
static const char *const SC_RB_KEY[N_RB] = {
#define SC_X(key, field, label, comm) #key,
    SC_RBUCKETS(SC_X)
#undef SC_X
};
static const char *const SC_RB_LABEL[N_RB] = {
#define SC_X(key, field, label, comm) label,
    SC_RBUCKETS(SC_X)
#undef SC_X
};
static const int SC_RB_COMM[N_RB] = {
#define SC_X(key, field, label, comm) comm,
    SC_RBUCKETS(SC_X)
#undef SC_X
};

static const size_t SC_GB_OFF[N_GB] = {
#define SC_X(key, field, label, comm) offsetof(MFTL_RedistributeStats, field),
    SC_GBUCKETS(SC_X)
#undef SC_X
};
static const char *const SC_GB_LABEL[N_GB] = {
#define SC_X(key, field, label, comm) label,
    SC_GBUCKETS(SC_X)
#undef SC_X
};
static const int SC_GB_COMM[N_GB] = {
#define SC_X(key, field, label, comm) comm,
    SC_GBUCKETS(SC_X)
#undef SC_X
};

static const size_t SC_RC_OFF[N_RC] = {
#define SC_X(key, field) offsetof(MFTL_RedistributeStats, field),
    SC_RCOUNTERS(SC_X)
#undef SC_X
};

/* The flat vector of per-rank timings that is reduced once at the end:
 * [phases | barrier waits | redistribution buckets | redistribution total |
 *  ghost buckets | ghost total]. */
#define TV_PHASE 0
#define TV_WAIT (TV_PHASE + N_PHASE)
#define TV_RB (TV_WAIT + N_PHASE)
#define TV_RTOT (TV_RB + N_RB)
#define TV_GB (TV_RTOT + 1)
#define TV_GTOT (TV_GB + N_GB)
#define N_TV (TV_GTOT + 1)

/** @brief Per-rank end-of-run figures, reduced as sum / max / min. */
enum
{
  PV_DEGREE = 0,
  PV_POOL_BYTES,
  PV_RSS_NOW,
  PV_RSS_PEAK,
  N_PV
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Configuration and problem
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The three motion laws; see the header comment. */
enum
{
  SC_FLOW = 0,
  SC_OSC,
  SC_DRIFT
};

/** @brief Names of the motion laws, for the report and the summary. */
static const char *const SC_MOTION_NAME[3] = {"flow", "osc", "drift"};

/** @brief Run options common to both drivers. */
typedef struct
{
  int refine;  ///< Icosphere subdivision level: 20 * 4^refine triangles per body.
  double phi;  ///< Volume fraction; sets the radius.
  double jitter;  ///< Lattice jitter, in lattice cells.
  double speed;  ///< Drift speed, or the oscillation's peak speed (not used by flow).
  int motion;  ///< SC_FLOW, SC_OSC or SC_DRIFT.
  double amplitude;  ///< Oscillation amplitude, in body spacings.
  bool jitter_given;  ///< --jitter was passed; otherwise flow forces 0.
  double dt;  ///< Timestep.
  int steps;  ///< Steps per flow trajectory.
  int cycles;  ///< Trajectories per repeat; increases duration without changing flow speed.
  int warmup;  ///< Untimed steps before the first repeat.
  int repeat;  ///< Timed repeats; the fastest is reported.
  int cuts[3];  ///< Process grid; {0,0,0} until resolved.
  bool cuts_given;
  bool normals;  ///< Phase 6.
  bool skip;  ///< MFTLRedistributeSetSkipUnchangedBodies().
  bool sync_phases;  ///< Barrier before every phase (diagnostic; not for quoted wall clocks).
  double shrink_slack;  ///< MFTLRedistributeSetShrinkSlack(); 0 = off.
  double max_gen_mb;  ///< Limit one rank-0 staging batch's mesh, not total startup RSS.
  int init_batch;  ///< Only this many new spheres enter each initialization redistribution.
  bool verify_init;  ///< Extra identity/position/connectivity checksum for initialization tests.
  double max_seconds;  ///< Soft job budget, checked outside timed repeats; zero disables it.
  char report[512];  ///< Report file; empty = stdout only.
  char label[64];
  char git[64];
} sc_options;

/** @brief Places body @p body; must be a pure function of the body index. */
typedef void (*sc_centre_fn)(const void *ctx, int body, double out[3]);

/** @brief What the driver decided the problem is. */
typedef struct
{
  const char *mode;  ///< "strong" or "weak".
  int n_bodies;
  double gmin[3];
  double gmax[3];
  double radius;
  double spacing;  ///< Mean centre-to-centre body spacing (cube root of volume per body).
  char placement[160];  ///< One line describing the start configuration.
  sc_centre_fn centre;
  const void *centre_ctx;
} sc_problem;

/* ═══════════════════════════════════════════════════════════════════════════
 * Small utilities
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief splitmix64, so placement and motion are pure functions of the body id.
 * @param s Mutable state, advanced by one step.
 * @return The next 64-bit value.
 */
static uint64_t sc_splitmix64(uint64_t *s)
{
  uint64_t z = (*s += 0x9E3779B97F4A7C15ULL);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

/**
 * @brief A uniform double in [0, 1).
 * @param s Mutable splitmix64 state.
 * @return The sample.
 */
static double sc_uniform(uint64_t *s)
{
  return (double) (sc_splitmix64(s) >> 11) * (1.0 / 9007199254740992.0);
}

/**
 * @brief The random stream of one body.
 * @param body Body index.
 * @param salt Separates independent streams of the same body.
 * @return A seeded state.
 */
static uint64_t sc_seed(int body, uint64_t salt)
{
  return 0x243F6A8885A308D3ULL ^ salt ^ ((uint64_t) body * 0x9E3779B97F4A7C15ULL);
}

/**
 * @brief The direction of body @p body: an isotropic unit vector.
 *
 * Per-body directions (and, for the oscillation, phases) stagger the cut-plane
 * crossings; one global velocity would make every body migrate on the same step
 * and turn the redistribution traffic into a spike train that belongs to the
 * setup, not to the library. Computed on the fly from the id rather than stored,
 * so the harness carries no array sized by the ensemble.
 *
 * @param      body Body global id.
 * @param[out] out  The unit direction.
 */
static void sc_body_direction(int body, double out[3])
{
  uint64_t s = sc_seed(body, 0xD1B54A32D192ED03ULL);
  double v[3] = {1.0, 0.0, 0.0};
  for (int attempt = 0; attempt < 64; ++attempt)
  {
    double w[3];
    for (int d = 0; d < 3; ++d) w[d] = sc_uniform(&s) * 2.0 - 1.0;
    const double n2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
    if (n2 > 0.01 && n2 <= 1.0)
    {
      for (int d = 0; d < 3; ++d) v[d] = w[d];
      break;
    }
  }
  const double mag = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
  for (int d = 0; d < 3; ++d) out[d] = v[d] * inv;
}

/** @brief The motion law, resolved once per run. */
typedef struct
{
  int kind;  ///< SC_FLOW, SC_OSC or SC_DRIFT.
  double flow_step[3];  ///< Flow: displacement per step, (3,2,1) spacings / steps.
  double speed;  ///< Drift speed, or the oscillation's peak speed.
  double amp;  ///< Oscillation amplitude (absolute).
  double omega;  ///< speed / amp.
} sc_motion;

/**
 * @brief Rigid displacement of body @p body over [t, t + dt].
 * @param      body Body global id.
 * @param      m    The motion law.
 * @param      t    Time at the start of the step.
 * @param      dt   Timestep.
 * @param[out] out  The displacement.
 */
static void sc_body_displacement(int body, const sc_motion *m, double t, double dt, double out[3])
{
  if (m->kind == SC_FLOW)
  {
    for (int d = 0; d < 3; ++d) out[d] = m->flow_step[d];
    return;
  }
  double dir[3];
  sc_body_direction(body, dir);
  double s;
  if (m->kind == SC_DRIFT)
    s = m->speed * dt;
  else
  {
    uint64_t st = sc_seed(body, 0x5851F42D4C957F2DULL);
    const double phase = 2.0 * MFTL_PI * sc_uniform(&st);
    s = m->amp * (sin(m->omega * (t + dt) + phase) - sin(m->omega * t + phase));
  }
  for (int d = 0; d < 3; ++d) out[d] = dir[d] * s;
}

/** @brief A near-cubic placement lattice. */
typedef struct
{
  int n[3];  ///< Cells per axis.
  long cells;  ///< n[0]*n[1]*n[2] >= the body count.
  double cell[3];  ///< Cell edge per axis.
} sc_lattice;

/**
 * @brief The most nearly cubic lattice over @p size with at least @p n_bodies cells.
 *
 * Stratified placement keeps bodies-per-rank near constant; uniform random
 * placement has Poisson fluctuations that change with P and land inside a
 * scaling curve as fake overhead.
 *
 * @param      size     Extent per axis.
 * @param      n_bodies Bodies to place.
 * @param[out] lat      The lattice.
 */
static void sc_lattice_build(const double size[3], int n_bodies, sc_lattice *lat)
{
  const double s = cbrt((double) n_bodies / (size[0] * size[1] * size[2]));
  for (int d = 0; d < 3; ++d)
  {
    const int nd = (int) ceil(size[d] * s - 1e-9);
    lat->n[d] = (nd < 1) ? 1 : nd;
  }
  while ((long) lat->n[0] * lat->n[1] * lat->n[2] < (long) n_bodies)
  {
    int best = 0;
    for (int d = 1; d < 3; ++d)
      if (size[d] / lat->n[d] > size[best] / lat->n[best]) best = d;
    lat->n[best]++;
  }
  lat->cells = (long) lat->n[0] * lat->n[1] * lat->n[2];
  for (int d = 0; d < 3; ++d) lat->cell[d] = size[d] / (double) lat->n[d];
}

/**
 * @brief Position of the @p j-th of @p n bodies on @p lat, relative to the
 *        lattice origin, jittered by the stream of body @p body.
 *
 * Bodies are spread with `cell = j * cells / n` rather than filling the first n
 * cells, so the density stays uniform along the slowest axis.
 *
 * @param      lat    The lattice.
 * @param      j      Index of the body on this lattice.
 * @param      n      Bodies on this lattice.
 * @param      body   Global body index, keys the jitter.
 * @param      jitter Jitter amplitude, in cells.
 * @param[out] out    Offset from the lattice origin.
 */
static void sc_lattice_point(const sc_lattice *lat, int j, int n, int body, double jitter, double out[3])
{
  const long cell = ((long) j * lat->cells) / (long) n;
  const long c[3] = {cell % lat->n[0], (cell / lat->n[0]) % lat->n[1], cell / ((long) lat->n[0] * lat->n[1])};
  uint64_t s = sc_seed(body, 0);
  for (int d = 0; d < 3; ++d) out[d] = ((double) c[d] + 0.5 + (sc_uniform(&s) * 2.0 - 1.0) * jitter) * lat->cell[d];
}

/**
 * @brief The radius giving volume fraction @p phi at @p vol_per_body, capped at
 *        0.45 of the smallest lattice cell so lattice neighbours cannot touch.
 * @param vol_per_body Volume available to one body.
 * @param phi          Volume fraction.
 * @param lat          The placement lattice.
 * @return The radius.
 */
static double sc_radius(double vol_per_body, double phi, const sc_lattice *lat)
{
  const double r = cbrt(3.0 * phi * vol_per_body / (4.0 * MFTL_PI));
  const double cap = 0.45 * MFTL_MIN(lat->cell[0], MFTL_MIN(lat->cell[1], lat->cell[2]));
  return MFTL_MIN(r, cap);
}

/**
 * @brief Peak resident set size, in bytes (ru_maxrss is bytes on macOS, kB on Linux).
 * @return Peak RSS, or 0 if the query failed.
 */
static long sc_peak_rss(void)
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
 * @brief Current resident set size, in bytes.
 * @return Current RSS, or 0 if the query failed.
 */
static long sc_current_rss(void)
{
#ifdef __APPLE__
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &info, &count) != KERN_SUCCESS) return 0;
  return (long) info.resident_size;
#else
  FILE *fp = fopen("/proc/self/statm", "r");
  if (fp == MFTL_NULL) return 0;
  long total = 0, resident = 0;
  const int got = fscanf(fp, "%ld %ld", &total, &resident);
  fclose(fp);
  return (got == 2) ? resident * (long) sysconf(_SC_PAGESIZE) : 0;
#endif
}

/**
 * @brief Cost of one MPI_Wtime() call in ns, best of five batches; the
 *        redistribution timers read the clock per body, so this prices them.
 * @return Nanoseconds per call.
 */
static double sc_clock_cost_ns(void)
{
  const int batch = 20000;
  double best = 1e30, sink = 0.0;
  for (int r = 0; r < 5; ++r)
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < batch; ++i) sink += MPI_Wtime();
    const double dt = MPI_Wtime() - t0;
    if (dt < best) best = dt;
  }
  if (sink == 0.0) best = 0.0;  // keeps the loop observable
  return 1e9 * best / (double) batch;
}

/**
 * @brief The q-quantile of an ascending array (nearest rank).
 * @param v Sorted values.
 * @param n Count.
 * @param q Quantile in [0, 1].
 * @return The value, 0 for an empty array.
 */
static double sc_quantile(const double *v, int n, double q)
{
  if (n < 1) return 0.0;
  int i = (int) (q * (double) (n - 1) + 0.5);
  if (i < 0) i = 0;
  if (i >= n) i = n - 1;
  return v[i];
}

/** @brief qsort comparator for doubles.
 *  @param a First.
 *  @param b Second.
 *  @return -1, 0 or 1. */
static int sc_cmp_double(const void *a, const void *b)
{
  const double x = *(const double *) a, y = *(const double *) b;
  return (x < y) ? -1 : ((x > y) ? 1 : 0);
}

/** @brief Field @p off of @p st as a double.
 *  @param st  The stats.
 *  @param off Byte offset of a double field.
 *  @return Its value. */
static double sc_dfield(const MFTL_RedistributeStats *st, size_t off)
{
  double v;
  memcpy(&v, (const char *) st + off, sizeof v);
  return v;
}

/** @brief Field @p off of @p st as a long.
 *  @param st  The stats.
 *  @param off Byte offset of a long field.
 *  @return Its value. */
static long sc_lfield(const MFTL_RedistributeStats *st, size_t off)
{
  long v;
  memcpy(&v, (const char *) st + off, sizeof v);
  return v;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Command line
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Reject partial conversions, overflow and NaN/Inf before entering MPI work. */
static bool sc_parse_int(const char *s, int *out)
{
  char *end;
  errno = 0;
  const long v = strtol(s, &end, 10);
  if (errno || end == s || *end || v < INT_MIN || v > INT_MAX) return false;
  *out = (int) v;
  return true;
}

static bool sc_parse_double(const char *s, double *out)
{
  char *end;
  errno = 0;
  const double v = strtod(s, &end);
  if (errno || end == s || *end || !isfinite(v)) return false;
  *out = v;
  return true;
}

static void sc_require(int rc, const char *operation)
{
  if (rc == MFTL_SUCCESS) return;
  fprintf(stderr, "scaling: %s failed\n", operation);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

/**
 * @brief Defaults shared by both drivers.
 * @param[out] opt   Receives the defaults.
 * @param      mode  "strong" or "weak", for the default label.
 * @param      steps Default steps per flow trajectory.
 */
static void sc_options_defaults(sc_options *opt, const char *mode, int steps)
{
  memset(opt, 0, sizeof *opt);
  opt->refine = 3;
  opt->phi = 0.05;
  opt->jitter = 0.35;
  opt->speed = 1.0;
  opt->motion = SC_FLOW;
  opt->amplitude = 0.5;
  opt->dt = 0.02;
  opt->steps = steps;
  opt->cycles = 1;
  opt->warmup = 5;
  opt->repeat = 3;
  opt->normals = true;
  opt->skip = true;
  opt->shrink_slack = 0.0;
  opt->max_gen_mb = 40000.0;
  opt->init_batch = 512;
  opt->max_seconds = 3000.0;
  snprintf(opt->label, sizeof opt->label, "%s", mode);
  snprintf(opt->git, sizeof opt->git, "unknown");
}

/** @brief Usage text of the shared options. */
static const char *const SC_COMMON_USAGE =
    "  --refine R          icosphere level, 20*4^R triangles per body  (default 3)\n"
    "  --phi F             volume fraction, sets the radius             (default 0.05)\n"
    "  --jitter F          lattice jitter in cells, 0..0.5     (default 0 for flow, else 0.35)\n"
    "  --motion M          flow (default): the unjittered lattice advances (3,2,1)\n"
    "                      spacings per trajectory; equivalent work for a full cubic\n"
    "                      unjittered lattice. Per-step displacement\n"
    "                      |(3,2,1)|/steps spacings.  osc: per-body oscillation.\n"
    "                      drift: straight lines in random directions (imbalanced)\n"
    "  --speed S --dt X    osc/drift only: (peak) speed and timestep     (default 1, 0.02)\n"
    "  --amplitude A       osc only: amplitude in body spacings          (default 0.5)\n"
    "  --steps N           steps per trajectory (also sets flow displacement)\n"
    "  --cycles N          trajectories per repeat; lengthen timing without changing speed (default 1)\n"
    "  --warmup N          untimed steps first                          (default 5)\n"
    "  --repeat R          timed repeats, fastest is reported           (default 3)\n"
    "  --cuts X Y Z        process grid                                 (default MPI_Dims_create)\n"
    "  --no-normals        drop phase 6 (vertex normals)\n"
    "  --no-skip           rebuild every body every step (disable the unchanged-body skip)\n"
    "  --shrink-slack X    MFTLRedistributeSetShrinkSlack               (default 0 = off)\n"
    "  --sync-phases       barrier before every phase: separates waiting from work.\n"
    "                      Diagnostic only -- do not quote that wall clock.\n"
    "  --init-batch N      spheres per initialization redistribution    (default 512)\n"
    "  --verify-init       check initial body/entity IDs, positions and connectivity\n"
    "  --max-gen-mb X      limit mesh storage of ONE initialization batch (default 40000)\n"
    "  --max-seconds X     soft job budget; stop between complete repeats (default 3000, 0=off)\n"
    "  --report PATH       write the report here (it is also printed)\n"
    "  --label NAME --git HASH   recorded verbatim\n";

/**
 * @brief Tries to consume one shared option at argv[*i].
 * @param argc From main.
 * @param argv From main.
 * @param[in,out] i Current index; advanced past any value consumed.
 * @param[out] opt Receives the value.
 * @return 1 if consumed, 0 if not a shared option, -1 for a missing or invalid value.
 */
static int sc_parse_common(int argc, char **argv, int *i, sc_options *opt)
{
  const char *a = argv[*i];
  const bool has1 = (*i + 1 < argc), has3 = (*i + 3 < argc);

#define SC_INT(name, field)                                                                                 \
  if (!strcmp(a, name))                                                                                     \
  {                                                                                                         \
    if (!has1) return -1;                                                                                   \
    if (!sc_parse_int(argv[++(*i)], &opt->field)) return -1;                                                \
    return 1;                                                                                               \
  }
#define SC_DBL(name, field)                                                                                 \
  if (!strcmp(a, name))                                                                                     \
  {                                                                                                         \
    if (!has1) return -1;                                                                                   \
    if (!sc_parse_double(argv[++(*i)], &opt->field)) return -1;                                             \
    return 1;                                                                                               \
  }
#define SC_STR(name, field)                                                                                 \
  if (!strcmp(a, name))                                                                                     \
  {                                                                                                         \
    if (!has1) return -1;                                                                                   \
    if (snprintf(opt->field, sizeof opt->field, "%s", argv[++(*i)]) >= (int) sizeof opt->field) return -1;  \
    return 1;                                                                                               \
  }

  SC_INT("--refine", refine)
  SC_DBL("--phi", phi)
  SC_DBL("--speed", speed)
  SC_DBL("--amplitude", amplitude)
  SC_DBL("--dt", dt)
  SC_INT("--steps", steps)
  SC_INT("--cycles", cycles)
  SC_INT("--warmup", warmup)
  SC_INT("--repeat", repeat)
  SC_DBL("--shrink-slack", shrink_slack)
  SC_DBL("--max-gen-mb", max_gen_mb)
  SC_INT("--init-batch", init_batch)
  SC_DBL("--max-seconds", max_seconds)
  SC_STR("--report", report)
  SC_STR("--label", label)
  SC_STR("--git", git)
#undef SC_INT
#undef SC_DBL
#undef SC_STR

  if (!strcmp(a, "--cuts"))
  {
    if (!has3) return -1;
    for (int d = 0; d < 3; ++d)
      if (!sc_parse_int(argv[++(*i)], &opt->cuts[d])) return -1;
    opt->cuts_given = true;
    return 1;
  }
  if (!strcmp(a, "--jitter"))
  {
    if (!has1) return -1;
    if (!sc_parse_double(argv[++(*i)], &opt->jitter)) return -1;
    opt->jitter_given = true;
    return 1;
  }
  if (!strcmp(a, "--motion"))
  {
    if (!has1) return -1;
    const char *m = argv[++(*i)];
    if (!strcmp(m, "flow"))
      opt->motion = SC_FLOW;
    else if (!strcmp(m, "osc"))
      opt->motion = SC_OSC;
    else if (!strcmp(m, "drift"))
      opt->motion = SC_DRIFT;
    else
      return -1;
    return 1;
  }
  if (!strcmp(a, "--no-normals"))
  {
    opt->normals = false;
    return 1;
  }
  if (!strcmp(a, "--verify-init"))
  {
    opt->verify_init = true;
    return 1;
  }
  if (!strcmp(a, "--no-skip"))
  {
    opt->skip = false;
    return 1;
  }
  if (!strcmp(a, "--sync-phases"))
  {
    opt->sync_phases = true;
    return 1;
  }
  return 0;
}

/**
 * @brief Resolves the process grid and validates the shared options.
 * @param[in,out] opt   The options.
 * @param         nproc Communicator size.
 * @param         rank  This rank; only rank 0 prints.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR after explaining why.
 */
static int sc_options_finish(sc_options *opt, int nproc, int rank)
{
  /* Flow is only period-locked on an unjittered lattice. */
  if (opt->motion == SC_FLOW && !opt->jitter_given) opt->jitter = 0.0;

  if (!opt->cuts_given)
  {
    opt->cuts[0] = opt->cuts[1] = opt->cuts[2] = 0;
    MPI_Dims_create(nproc, 3, opt->cuts);
  }
  else if (opt->cuts[0] < 1 || opt->cuts[1] < 1 || opt->cuts[2] < 1 ||
           (long long) opt->cuts[0] * opt->cuts[1] > nproc ||
           (long long) opt->cuts[0] * opt->cuts[1] * opt->cuts[2] != nproc)
  {
    if (rank == 0) fprintf(stderr, "--cuts %d %d %d does not match %d ranks.\n", opt->cuts[0], opt->cuts[1], opt->cuts[2], nproc);
    return MFTL_ERROR;
  }
  if (opt->refine < 0 || opt->refine > 8 || opt->steps < 1 || opt->warmup < 0 || opt->repeat < 1 || opt->phi <= 0.0 || opt->phi >= 0.5 ||
      opt->jitter < 0.0 || opt->jitter > 0.5 || opt->dt <= 0.0 || opt->speed <= 0.0 || opt->amplitude <= 0.0 ||
      opt->max_gen_mb <= 0.0 || opt->init_batch < 1 || opt->max_seconds < 0.0 ||
      opt->shrink_slack < 0.0 || (opt->shrink_slack > 0.0 && opt->shrink_slack < 1.0))
  {
    if (rank == 0)
      fprintf(stderr, "need 0 <= refine <= 8, steps >= 1, warmup >= 0, repeat >= 1, init-batch >= 1, max-seconds >= 0, 0 < phi < 0.5, 0 <= jitter <= 0.5, dt, speed, amplitude, max-gen-mb > 0; shrink-slack is 0 or >= 1.\n");
    return MFTL_ERROR;
  }
  if (opt->cycles < 1 || opt->cycles > INT_MAX / opt->steps)
  {
    if (rank == 0) fprintf(stderr, "need cycles >= 1 and steps*cycles <= INT_MAX.\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The timestep
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Opens a phase; under --sync-phases a barrier first, charged to @p wait.
 * @param      sync Whether to barrier.
 * @param[out] wait Accumulates the barrier time.
 * @return The phase start time.
 */
static inline double sc_begin(bool sync, double *wait)
{
  if (sync)
  {
    const double b0 = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    *wait += MPI_Wtime() - b0;
  }
  return MPI_Wtime();
}

/**
 * @brief One instrumented timestep.
 * @param      ens  The ensemble.
 * @param      opt  The options.
 * @param      mot  The motion law.
 * @param      t    Time at the start of the step.
 * @param[out] ph   [N_PHASE] this step's phase times on this rank.
 * @param[in,out] wt [N_PHASE] accumulated barrier waits (--sync-phases).
 */
static void sc_step(MFTL_Ensemble *ens, const sc_options *opt, const sc_motion *mot, double t, double ph[N_PHASE], double wt[N_PHASE])
{
  for (int p = 0; p < N_PHASE; ++p) ph[p] = 0.0;

  /* 1. Move what this rank owns, and only that. The handle that survives a
   *    redistribution is global_id, so the motion is keyed on it. */
  double t0 = sc_begin(opt->sync_phases, &wt[PH_ADVECT]);
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    double disp[3];
    sc_body_displacement(obj->global_id, mot, t, opt->dt, disp);
    const double dx = disp[0], dy = disp[1], dz = disp[2];
    MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += dx; pos[1] += dy; pos[2] += dz; MFTLApplyPeriodicity(&ens->domain, pos););
  }
  ph[PH_ADVECT] = MPI_Wtime() - t0;

  /* 2. Pull in the vertices the neighbours moved (also renews the generations). */
  t0 = sc_begin(opt->sync_phases, &wt[PH_GHOST]);
  sc_require(MFTLUpdateGhostPositions(ens), "ghost positions");
  ph[PH_GHOST] = MPI_Wtime() - t0;

  /* 3. Geometry of EVERY triangle: redistribution routes ghosts by their centre. */
  t0 = sc_begin(opt->sync_phases, &wt[PH_GEOM]);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  ph[PH_GEOM] = MPI_Wtime() - t0;

  /* 4. Re-decide ownership from the fresh centres. */
  t0 = sc_begin(opt->sync_phases, &wt[PH_REDIST]);
  sc_require(MFTLRedistribute(ens), "redistribution");
  ph[PH_REDIST] = MPI_Wtime() - t0;

  /* 5. Restore the triangle geometry the redistribution does not ship. */
  t0 = sc_begin(opt->sync_phases, &wt[PH_POST]);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  ph[PH_POST] = MPI_Wtime() - t0;

  /* 6. Vertex normals, exact on ghosts too (one ghost exchange). */
  if (opt->normals)
  {
    t0 = sc_begin(opt->sync_phases, &wt[PH_NORMALS]);
    sc_require(MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_AREA), "vertex normals");
    ph[PH_NORMALS] = MPI_Wtime() - t0;
  }
}

/**
 * @brief Local owned area, triangles and vertices; requires current geometry.
 * @param      ens The ensemble.
 * @param[out] mine {area, triangles, vertices} on this rank.
 */
static void sc_local_totals(const MFTL_Ensemble *ens, double mine[3])
{
  mine[0] = mine[1] = mine[2] = 0.0;
  double correction = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      /* Compensated sum: tens of millions of triangles should not fail the
       * conservation test merely because MPI changes the summation order. */
      const double y = obj->tris.area[t] - correction;
      const double sum = mine[0] + y;
      correction = (sum - mine[0]) - y;
      mine[0] = sum;
    }
    mine[1] += (double) obj->tris.n_local;
    mine[2] += (double) obj->verts.n_local;
  }
}

static void sc_global_totals(const MFTL_Ensemble *ens, double out[3])
{
  double mine[3];
  sc_local_totals(ens, mine);
  MPI_Allreduce(mine, out, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

#include "scaling_init.h"

/**
 * @brief This rank's share right now: owned triangles, ghost triangles, bodies
 *        held. Sampled every timed step, because a snapshot at the end describes
 *        one special configuration (under flow, the start one, which no cut plane
 *        crosses) rather than the run.
 * @param      ens The ensemble.
 * @param[out] out {owned triangles, ghost triangles, bodies held}.
 */
static void sc_partition_sample(const MFTL_Ensemble *ens, double out[3])
{
  long owned = 0, ghost = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    owned += ens->local_objects[o]->tris.n_local;
    ghost += ens->local_objects[o]->tris.n - ens->local_objects[o]->tris.n_local;
  }
  out[0] = (double) owned;
  out[1] = (double) ghost;
  out[2] = (double) ens->n_local_objects;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Report output
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Where rank 0 writes: stdout always, plus the report file if open. */
static FILE *sc_report_fp = MFTL_NULL;

/**
 * @brief printf to stdout and to the report file.
 * @param fmt printf format.
 */
static void sc_out(const char *fmt, ...)
{
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stdout, fmt, ap);
  va_end(ap);
  if (sc_report_fp)
  {
    va_start(ap, fmt);
    vfprintf(sc_report_fp, fmt, ap);
    va_end(ap);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The run
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Generates, scatters, warms up, times, checks and reports one point.
 *
 * Collective. Called by every rank after MPI_Init; does not finalize.
 *
 * @param prob The problem the driver built.
 * @param opt  The resolved options.
 * @return Process exit code: 0 on success, 1 if a check failed, 2 on refusal.
 */
static int sc_run(const sc_problem *prob, const sc_options *opt)
{
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  const double run_start = MPI_Wtime();
  const int nb = prob->n_bodies;
  const long tri_per_body = 20L * (1L << (2 * opt->refine));
  const long vert_per_body = 10L * (1L << (2 * opt->refine)) + 2L;

  /* The current redistribution and publication use int-sized counts. */
  if ((long long) nb * tri_per_body > INT_MAX || nb > INT_MAX / 7)
  {
    if (rank == 0) fprintf(stderr, "refusing: problem exceeds the current int-sized redistribution/publication counts.\n");
    return 2;
  }

  /* Only one batch is generated on rank 0. The final local partition and
   * scatter/workspace allocations are additional to this batch-only estimate. */
  const int batch_bodies = MFTL_MIN(nb, opt->init_batch);
  const double gen_mb = (double) batch_bodies * ((double) tri_per_body * 140.0 + (double) vert_per_body * 64.0) / (1024.0 * 1024.0);
  if (gen_mb > opt->max_gen_mb)
  {
    if (rank == 0)
      fprintf(stderr, "refusing: initialization batch mesh storage %.0f MiB > --max-gen-mb %.0f; lower --init-batch.\n",
              gen_mb, opt->max_gen_mb);
    return 2;
  }

  int report_error = 0;
  if (rank == 0 && opt->report[0])
  {
    sc_report_fp = fopen(opt->report, "w");
    if (!sc_report_fp)
    {
      fprintf(stderr, "cannot open report %s: %s\n", opt->report, strerror(errno));
      report_error = 1;
    }
  }
  MPI_Bcast(&report_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (report_error) return 2;

  /* ---- Placement on the machine ------------------------------------------ */

  MPI_Comm node_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
  int node_rank = 0, node_size = 1;
  MPI_Comm_rank(node_comm, &node_rank);
  MPI_Comm_size(node_comm, &node_size);
  int n_nodes = 0, rpn_max = 0, rpn_min = 0;
  {
    const int leader = (node_rank == 0) ? 1 : 0;
    MPI_Allreduce(&leader, &n_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&node_size, &rpn_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&node_size, &rpn_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  }
  const char *expected_ranks = getenv("NRANKS");
  const char *expected_nodes = getenv("MFTL_EXPECT_NODES");
  const char *expected_rpn = getenv("MFTL_EXPECT_RPN");
  if ((expected_ranks && atoi(expected_ranks) != nproc) ||
      (expected_nodes && atoi(expected_nodes) != n_nodes) ||
      (expected_rpn && rpn_max > atoi(expected_rpn)))
  {
    if (rank == 0) fprintf(stderr, "MPI placement does not match the submitted rank/node counts; check site.sh's launcher.\n");
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  bool periodic[3] = {true, true, true};
  int cuts[3] = {opt->cuts[0], opt->cuts[1], opt->cuts[2]};
  double gmin[3], gmax[3], size[3], rank_box[3];
  for (int d = 0; d < 3; ++d)
  {
    gmin[d] = prob->gmin[d];
    gmax[d] = prob->gmax[d];
    size[d] = gmax[d] - gmin[d];
    rank_box[d] = size[d] / (double) cuts[d];
  }
  const double box_min = MFTL_MIN(rank_box[0], MFTL_MIN(rank_box[1], rank_box[2]));

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, gmin, gmax, periodic, MFTL_MAX(nb / nproc, 16));
  if (!ens)
  {
    fprintf(stderr, "rank %d: MFTLEnsembleCreate failed\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  sc_motion mot;
  mot.kind = opt->motion;
  mot.flow_step[0] = 3.0 * prob->spacing / (double) opt->steps;
  mot.flow_step[1] = 2.0 * prob->spacing / (double) opt->steps;
  mot.flow_step[2] = 1.0 * prob->spacing / (double) opt->steps;
  mot.speed = opt->speed;
  mot.amp = opt->amplitude * prob->spacing;
  mot.omega = opt->speed / mot.amp;
  long step_global = 0;  // time = step_global * dt; keeps the oscillation continuous across repeats

  MFTLRedistributeSetSkipUnchangedBodies(opt->skip);
  MFTLRedistributeSetShrinkSlack(opt->shrink_slack);

  if (rank == 0)
  {
    printf("== MFTL %s scaling: %d ranks (%dx%dx%d) on %d node(s), %d bodies x %ld triangles ==\n", prob->mode, nproc, cuts[0], cuts[1], cuts[2], n_nodes,
           nb, tri_per_body);
    printf("   initializing in batches of at most %d spheres (batch mesh estimate %.0f MiB on rank 0) ...\n", batch_bodies, gen_mb);
    fflush(stdout);
  }

  /* Setup moves only each new batch, never revisits the accumulated mesh. */
  const sc_setup setup = sc_initialize(ens, prob, opt);
  const double t_gen = setup.generation_s, t_scatter = setup.scatter_s;
  const long rss_after_gen = setup.rss_generation, rss_after_scatter = setup.rss_scatter;
  double reference[3];
  memcpy(reference, setup.reference, sizeof reference);
  if (!isfinite(reference[0]) || reference[0] <= 0.0 ||
      reference[1] != (double) nb * tri_per_body || reference[2] != (double) nb * vert_per_body)
  {
    if (rank == 0) fprintf(stderr, "generated mesh does not match the requested sphere counts/area\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  ens->total_global_triangles = (int) reference[1];
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    if (obj->domain != &ens->domain || obj->global_id < 0 || obj->global_id >= nb ||
        (o && ens->local_objects[o - 1]->global_id >= obj->global_id))
    {
      fprintf(stderr, "invalid final body directory after initialization\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
  double scattered[3];
  sc_global_totals(ens, scattered);
  if (!isfinite(scattered[0]) || fabs(scattered[0] - reference[0]) / reference[0] > 1e-9 ||
      scattered[1] != reference[1] || scattered[2] != reference[2])
  {
    if (rank == 0) fprintf(stderr, "batched initialization failed conservation\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (rank == 0)
  {
    printf("   setup: %.2f s total, generation %.2f s, distribution/append %.2f s; warm-up %d steps ...\n",
           setup.wall_s, t_gen, t_scatter, opt->warmup);
    fflush(stdout);
  }

  /* ---- Warm-up, then the timed repeats ------------------------------------ */

  double ph[N_PHASE], wt_discard[N_PHASE];
  MFTLRedistributeStatsReset();
  for (int p = 0; p < N_PHASE; ++p) wt_discard[p] = 0.0;
  for (int w = 0; w < opt->warmup; ++w) sc_step(ens, opt, &mot, (double) step_global++ * opt->dt, ph, wt_discard);
  MFTL_RedistributeStats warm_stats;
  MFTLRedistributeGetStats(&warm_stats);
  /* Check all work, including warm-up and slower repeats, independently of
   * which repeat supplies the performance profile. Initial scatter is excluded. */
  long check_counts[3] = {warm_stats.g_n_orphaned, warm_stats.n_objects_centres_recomputed, warm_stats.n_route_global};

  const int ns = opt->steps * opt->cycles;
  double *const step_tot = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const step_red = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const best_tot = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const best_red = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const rep_wall = (double *) MFTLMallocOrDie((size_t) opt->repeat * sizeof(double), "scaling repeat walls");
  double *const step_own = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const best_own = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double best_part[3] = {0.0, 0.0, 0.0};

  double best_acc[N_PHASE], best_wait[N_PHASE];
  MFTL_RedistributeStats best_rs;
  MFTL_BodyReduceStats best_br;
  memset(&best_rs, 0, sizeof best_rs);
  memset(&best_br, 0, sizeof best_br);
  double best_wall = -1.0;
  int failures = 0;
  double worst_area_dev = 0.0;
  int completed_repeats = 0;
  bool budget_stop = false;

  for (int rep = 0; rep < opt->repeat; ++rep)
  {
    /* Budget decisions happen between full repeats, never inside the measured
     * step loop. Preserve the same trajectory/speed at every scaling point. */
    const double elapsed_local = MPI_Wtime() - run_start;
    double elapsed;
    MPI_Allreduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (opt->max_seconds > 0.0 &&
        elapsed + (rep ? 1.5 * rep_wall[rep - 1] : 0.0) + 30.0 > opt->max_seconds)
    {
      if (rep == 0)
      {
        if (rank == 0) fprintf(stderr, "setup exhausted the soft run budget before any timed repeat\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
      }
      budget_stop = true;
      if (rank == 0) printf("   stopping after %d complete repeats to stay within the %.0f s soft budget\n", rep, opt->max_seconds);
      break;
    }
    double acc[N_PHASE], wait[N_PHASE], part_acc[3] = {0.0, 0.0, 0.0};
    for (int p = 0; p < N_PHASE; ++p) acc[p] = wait[p] = 0.0;
    MFTLRedistributeStatsReset();
    MFTLEnsembleResetBodyReduceStats();

    MPI_Barrier(MPI_COMM_WORLD);
    const double w0 = MPI_Wtime();
    for (int s = 0; s < ns; ++s)
    {
      sc_step(ens, opt, &mot, (double) step_global++ * opt->dt, ph, wait);
      double tot = 0.0;
      for (int p = 0; p < N_PHASE; ++p)
      {
        acc[p] += ph[p];
        tot += ph[p];
      }
      step_tot[s] = tot;
      step_red[s] = ph[PH_REDIST];

      double part[3];
      sc_partition_sample(ens, part);
      step_own[s] = part[0];
      for (int k = 0; k < 3; ++k) part_acc[k] += part[k];
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double wall_local = MPI_Wtime() - w0;

    MFTL_RedistributeStats rs;
    MFTL_BodyReduceStats br;
    MFTLRedistributeGetStats(&rs);
    MFTLEnsembleGetBodyReduceStats(&br);

    /* The selection must be identical on every rank, so it uses a reduced value. */
    double wall = 0.0;
    MPI_Allreduce(&wall_local, &wall, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    rep_wall[rep] = wall;
    completed_repeats++;

    /* Conservation, outside the clock. Phase 5 left current geometry. */
    double now[3];
    sc_global_totals(ens, now);
    const double dev = fabs(now[0] - reference[0]) / reference[0];
    worst_area_dev = MFTL_MAX(worst_area_dev, dev);
    if (!isfinite(now[0]) || dev > 1e-9 || now[1] != reference[1] || now[2] != reference[2])
    {
      if (rank == 0)
        fprintf(stderr, "repeat %d: area rel. dev %.3e, %.0f triangles, %.0f vertices (expected %.0f / %.0f)\n", rep, dev, now[1], now[2], reference[1],
                reference[2]);
      failures++;
    }
    check_counts[0] += rs.g_n_orphaned;
    check_counts[1] += rs.n_objects_centres_recomputed;
    check_counts[2] += rs.n_route_global;
    if (rs.n_calls != ns) failures++;

    if (best_wall < 0.0 || wall < best_wall)
    {
      best_wall = wall;
      best_rs = rs;
      best_br = br;
      for (int p = 0; p < N_PHASE; ++p)
      {
        best_acc[p] = acc[p];
        best_wait[p] = wait[p];
      }
      memcpy(best_tot, step_tot, (size_t) ns * sizeof(double));
      memcpy(best_red, step_red, (size_t) ns * sizeof(double));
      memcpy(best_own, step_own, (size_t) ns * sizeof(double));
      for (int k = 0; k < 3; ++k) best_part[k] = part_acc[k] / (double) ns;
    }
    if (rank == 0)
    {
      printf("   repeat %d/%d: %.4f s (%.3f ms/step)\n", rep + 1, opt->repeat, wall, 1e3 * wall / ns);
      fflush(stdout);
    }
  }

  /* ---- Reductions (all outside the timed region) -------------------------- */

  double tv[N_TV], tv_sum[N_TV], tv_max[N_TV], tv_min[N_TV];
  for (int p = 0; p < N_PHASE; ++p)
  {
    tv[TV_PHASE + p] = best_acc[p];
    tv[TV_WAIT + p] = best_wait[p];
  }
  for (int i = 0; i < N_RB; ++i) tv[TV_RB + i] = sc_dfield(&best_rs, SC_RB_OFF[i]);
  tv[TV_RTOT] = best_rs.t_total;
  for (int i = 0; i < N_GB; ++i) tv[TV_GB + i] = sc_dfield(&best_rs, SC_GB_OFF[i]);
  tv[TV_GTOT] = best_rs.g_t_total;
  MPI_Reduce(tv, tv_sum, N_TV, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(tv, tv_max, N_TV, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(tv, tv_min, N_TV, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  long rc[N_RC], rc_sum[N_RC], rc_max[N_RC], rc_min[N_RC];
  for (int i = 0; i < N_RC; ++i) rc[i] = sc_lfield(&best_rs, SC_RC_OFF[i]);
  MPI_Reduce(rc, rc_sum, N_RC, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(rc, rc_max, N_RC, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(rc, rc_min, N_RC, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

  long br[3] = {best_br.n_holders, best_br.n_global, best_br.n_bytes}, br_sum[3];
  MPI_Reduce(br, br_sum, 3, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  long pv[N_PV], pv_sum[N_PV], pv_max[N_PV], pv_min[N_PV];
  for (int i = 0; i < N_PV; ++i) pv[i] = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) pv[PV_POOL_BYTES] += (long) MFTLObjectPoolBytes(ens->local_objects[o]);
  pv[PV_DEGREE] = ens->domain.num_neighbors;
  pv[PV_RSS_NOW] = sc_current_rss();
  pv[PV_RSS_PEAK] = sc_peak_rss();
  MPI_Reduce(pv, pv_sum, N_PV, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(pv, pv_max, N_PV, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(pv, pv_min, N_PV, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);

  /* Memory per NODE: the number that decides whether a node runs out. */
  long node_rss[2] = {0, 0}, node_rss_max[2] = {0, 0};
  {
    const long mine[2] = {pv[PV_RSS_PEAK], pv[PV_RSS_NOW]};
    MPI_Reduce(mine, node_rss, 2, MPI_LONG, MPI_SUM, 0, node_comm);
    if (node_rank != 0) node_rss[0] = node_rss[1] = 0;
    MPI_Reduce(node_rss, node_rss_max, 2, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  }
  long setup_rss[2] = {rss_after_gen, rss_after_scatter}, setup_rss_max[2];
  MPI_Reduce(setup_rss, setup_rss_max, 2, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  /* Per step: the BUSIEST rank's step and redistribution time. */
  double *const step_tot_max = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  double *const step_red_max = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  MPI_Reduce(best_tot, step_tot_max, ns, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(best_red, step_red_max, ns, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  /* The busiest rank's owned triangles at every step, and each rank's
   * time-averaged share {owned, ghost, bodies held}. */
  double *const step_own_max = (double *) MFTLMallocOrDie((size_t) ns * sizeof(double), "scaling step series");
  MPI_Reduce(best_own, step_own_max, ns, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  double part_sum[3], part_max[3], part_min[3];
  MPI_Reduce(best_part, part_sum, 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(best_part, part_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(best_part, part_min, 3, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

  const double clock_ns = sc_clock_cost_ns();
  double clock_ns_max = 0.0;
  MPI_Reduce(&clock_ns, &clock_ns_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  /* Every rank judged the same reduced totals; MAX makes that an agreement. */
  int failures_total = 0;
  MPI_Allreduce(&failures, &failures_total, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  long check_sum[3];
  MPI_Reduce(check_counts, check_sum, 3, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  /* ---- Report, rank 0 ------------------------------------------------------ */

  int exit_code = 0;
  if (rank == 0)
  {
    const double P = (double) nproc, S = (double) ns, mb = 1.0 / (1024.0 * 1024.0);
    const double tick = MPI_Wtick();
    const double floor_s = 100.0 * tick;
#ifdef MFTL_COMM_PROFILE
    const int profiled = 1;
#else
    const int profiled = 0;
#endif
#if defined(NDEBUG) && defined(__OPTIMIZE__)
    const int optimised = 1;
#else
    const int optimised = 0;
#endif

    char libver[MPI_MAX_LIBRARY_VERSION_STRING] = "unknown";
    int liblen = 0;
    MPI_Get_library_version(libver, &liblen);
    for (char *q = libver; *q; ++q)
      if (*q == '\n' || *q == '\r')
      {
        *q = '\0';
        break;
      }
    char host[MPI_MAX_PROCESSOR_NAME] = "unknown";
    int hostlen = 0;
    MPI_Get_processor_name(host, &hostlen);
    char when[64];
    const time_t now_t = time(MFTL_NULL);
    strftime(when, sizeof when, "%Y-%m-%d %H:%M:%S", localtime(&now_t));

    const long g_tris = (long) reference[1], g_verts = (long) reference[2];
    const double ms_step = 1e3 * best_wall / S;
    /* Structural imbalance as the TIME sees it: the busiest rank's owned
     * triangles over the mean, at every step, averaged over steps; and the worst
     * step. Every phase waits for the busiest rank of that step. */
    const double owned_mean = (double) g_tris / P;
    double struct_imb = 0.0, imb_peak = 0.0;
    for (int k = 0; k < ns; ++k)
    {
      const double r = (owned_mean > 0.0) ? step_own_max[k] / owned_mean : 1.0;
      struct_imb += r / S;
      imb_peak = MFTL_MAX(imb_peak, r);
    }
    const double halo = (part_sum[0] > 0.0) ? part_sum[1] / part_sum[0] : 0.0;
    const double bodies_mean = part_sum[2] / P;
    /* The largest per-step displacement of any body. */
    const double disp = (opt->motion == SC_FLOW) ? sqrt(14.0) * prob->spacing / opt->steps : opt->speed * opt->dt;

    double ph_mean[N_PHASE], sum_mean = 0.0, sum_max = 0.0;
    for (int p = 0; p < N_PHASE; ++p)
    {
      ph_mean[p] = tv_sum[TV_PHASE + p] / P;
      sum_mean += ph_mean[p];
      sum_max += tv_max[TV_PHASE + p];
    }

    double red_comm = 0.0, red_local = 0.0;
    for (int i = 0; i < N_RB; ++i)
    {
      const double v = tv_sum[TV_RB + i] / P;
      if (SC_RB_COMM[i])
        red_comm += v;
      else
        red_local += v;
    }
    const double red_total = tv_sum[TV_RTOT] / P;
    const double gs_total = tv_sum[TV_GTOT] / P;
    double gs_comm = 0.0;
    for (int i = 0; i < N_GB; ++i)
      if (SC_GB_COMM[i]) gs_comm += tv_sum[TV_GB + i] / P;

    /* The communication share from the library's own timers: the exchange and
     * synchronisation buckets of the redistribution and of both ghost exchanges.
     * Counting whole phases instead would charge the redistribution's local
     * rebuild (most of it) to the network. */
    const double comm_pct = 100.0 * (red_comm + gs_comm) / best_wall;

    qsort(step_tot_max, (size_t) ns, sizeof(double), sc_cmp_double);
    qsort(step_red_max, (size_t) ns, sizeof(double), sc_cmp_double);
    double *rw = (double *) MFTLMallocOrDie((size_t) completed_repeats * sizeof(double), "scaling repeat walls");
    memcpy(rw, rep_wall, (size_t) completed_repeats * sizeof(double));
    qsort(rw, (size_t) completed_repeats, sizeof(double), sc_cmp_double);
    const double median_wall = 0.5 * (rw[(completed_repeats - 1) / 2] + rw[completed_repeats / 2]);

    const double per_calls = (rc_sum[RC_calls] > 0) ? (double) rc_sum[RC_calls] / P : S;  // redistributions per rank
    const double owner_pct = (rc_sum[RC_tris_scanned] > 0) ? 100.0 * (double) rc_sum[RC_tris_moved] / (double) rc_sum[RC_tris_scanned] : 0.0;
    const double skip_pct = (rc_sum[RC_bodies_visited] > 0) ? 100.0 * (double) rc_sum[RC_bodies_skipped] / (double) rc_sum[RC_bodies_visited] : 0.0;
    const double perturb_ms = (double) rc_sum[RC_clock_reads] / P / S * clock_ns_max * 1e-6;

    const bool orphaned = check_sum[0] > 0;
    const bool recomputed = check_sum[1] > 0;
    const bool global_route = check_sum[2] > 0;
    if (failures_total > 0 || orphaned || recomputed || global_route) exit_code = 1;

    /* -- header ---------------------------------------------------------- */
    sc_out("==============================================================================\n");
    sc_out(" MFTL %s SCALING  --  %d ranks (%dx%dx%d) on %d node(s), %d..%d ranks/node\n", (!strcmp(prob->mode, "weak")) ? "WEAK" : "STRONG", nproc, cuts[0],
           cuts[1], cuts[2], n_nodes, rpn_min, rpn_max);
    sc_out("==============================================================================\n");
    sc_out("[run]\n");
    sc_out("  label / git            %s / %s\n", opt->label, opt->git);
    sc_out("  date, rank-0 host      %s, %s\n", when, host);
    sc_out("  MPI library            %s\n", libver);
    sc_out("  build                  optimised+NDEBUG: %s   MFTL_COMM_PROFILE: %s\n", optimised ? "yes" : "NO -- do not quote these timings",
           profiled ? "on" : "OFF -- redistribution buckets below are all zero");
    sc_out("  MPI_Wtick / Wtime cost %.3g s / %.1f ns\n", tick, clock_ns_max);
    sc_out("  options                normals %s, body skip %s, shrink slack %g, sync-phases %s\n", opt->normals ? "on" : "off", opt->skip ? "on" : "off",
           opt->shrink_slack, opt->sync_phases ? "ON (diagnostic: wall clock includes barriers)" : "off");

    sc_out("[problem]\n");
    sc_out("  bodies                 %d x %ld triangles (refine %d) = %ld triangles, %ld vertices\n", nb, tri_per_body, opt->refine, g_tris, g_verts);
    sc_out("  per rank (mean)        %.0f triangles, %.1f bodies held\n", owned_mean, bodies_mean);
    sc_out("  domain                 %.4g x %.4g x %.4g, fully periodic\n", size[0], size[1], size[2]);
    sc_out("  rank box               %.4g x %.4g x %.4g\n", rank_box[0], rank_box[1], rank_box[2]);
    sc_out("  placement              %s\n", prob->placement);
    sc_out("  radius / spacing       %.5g / %.5g   (body diameter = %.3f rank boxes)\n", prob->radius, prob->spacing, 2.0 * prob->radius / box_min);
    if (opt->motion == SC_FLOW)
      sc_out("  motion                 FLOW: advances (3,2,1) mean spacings/trajectory; repeats are equivalent only for a full, unjittered cubic lattice\n");
    else if (opt->motion == SC_DRIFT)
      sc_out("  motion                 DRIFT at speed %g, dt %g (bodies diffuse; per-rank load becomes Poisson-imbalanced)\n", opt->speed, opt->dt);
    else
      sc_out("  motion                 OSC about the start: amplitude %.4g (%.2f spacings), peak speed %g, dt %g, period %.0f steps\n", mot.amp,
             opt->amplitude, opt->speed, opt->dt, 2.0 * MFTL_PI / mot.omega / opt->dt);
    sc_out("  per-step displacement  <= %.4g = %.3f %% of the smallest rank box\n", disp, 100.0 * disp / box_min);
    sc_out("  loop                   %d warm-up + %d/%d repeats x %d timed steps\n", opt->warmup, completed_repeats, opt->repeat, ns);
    sc_out("  soft runtime budget    %.0f s (0 = off); stopped early: %s\n", opt->max_seconds, budget_stop ? "yes" : "no");
    sc_out("  trajectory             %d steps x %d cycles per repeat\n", opt->steps, opt->cycles);

    sc_out("[setup]  (batched generation/distribution, NOT part of any scaling number)\n");
    sc_out("  initialization         %10.3f s total, %d batches of <= %d spheres\n", setup.wall_s, setup.batches, batch_bodies);
    sc_out("  generation             %10.3f s   (batch mesh estimate %.1f MiB; measured rank-0 batch pools %.1f MiB)\n",
           t_gen, gen_mb, (double) setup.batch_pool_peak * mb);
    sc_out("  distribution + append  %10.3f s   (only new spheres; batch workspace released each time)\n", t_scatter);
    sc_out("  RSS max over ranks     %10.1f MB at batch generation boundaries, %.1f MB after batch scatters\n",
           (double) setup_rss_max[0] * mb, (double) setup_rss_max[1] * mb);
    if (opt->verify_init) sc_out("  initialization checksum %016llx (body/entity IDs, positions, connectivity verified)\n", setup.signature);

    /* -- the scaling numbers --------------------------------------------- */
    sc_out("[timestep]\n");
    sc_out("  wall clock             %10.4f s  = %.4f ms/step   <-- THE SCALING NUMBER\n", best_wall, ms_step);
    if (best_wall < 1.0) sc_out("  short timing window    consider increasing --cycles at fixed --steps for final measurements\n");
    sc_out("  repeats (min/med/max)  %10.4f / %.4f / %.4f s   spread %.1f %%\n", rw[0], median_wall, rw[completed_repeats - 1],
           100.0 * (rw[completed_repeats - 1] - rw[0]) / rw[0]);
    sc_out("  throughput             %10.4g triangle-steps/s total, %.4g per rank\n", (double) g_tris * S / best_wall, (double) g_tris * S / best_wall / P);
    sc_out("  sum of phase means     %10.4f s   (unattributed %.1f %% of wall)\n", sum_mean, 100.0 * (best_wall - sum_mean) / best_wall);
    sc_out("  sum of phase maxima    %10.4f s   (max of each rank's accumulated phase; NOT a critical-path bound)\n", sum_max);
    if (profiled)
      sc_out("  communication + sync   %10.1f %% of wall (exchange/sync buckets of redistribution + ghost exchanges)\n", comm_pct);
    else
      sc_out("  communication + sync          n/a (needs MFTL_COMM_PROFILE)\n");
    sc_out("  busiest-rank step      p50 %.4f  p90 %.4f  max %.4f ms\n", 1e3 * sc_quantile(step_tot_max, ns, 0.5), 1e3 * sc_quantile(step_tot_max, ns, 0.9),
           1e3 * step_tot_max[ns - 1]);
    sc_out("  busiest-rank redist    p50 %.4f  p90 %.4f  max %.4f ms\n", 1e3 * sc_quantile(step_red_max, ns, 0.5), 1e3 * sc_quantile(step_red_max, ns, 0.9),
           1e3 * step_red_max[ns - 1]);
    sc_out("\n  %-24s %10s %10s %10s %7s %9s%s\n", "phase  [ms/step]", "mean", "min", "max", "% wall", "max/mean", opt->sync_phases ? "   barrier" : "");
    for (int p = 0; p < N_PHASE; ++p)
    {
      if (p == PH_NORMALS && !opt->normals) continue;
      const double m = ph_mean[p] / S * 1e3;
      const bool thin = ph_mean[p] / S < floor_s;
      sc_out("  %-24s %10.4f %10.4f %10.4f %7.1f %9.2f", SC_PHASE_LABEL[p], m, tv_min[TV_PHASE + p] / S * 1e3, tv_max[TV_PHASE + p] / S * 1e3,
             100.0 * ph_mean[p] / best_wall, (ph_mean[p] > 0.0) ? tv_max[TV_PHASE + p] / ph_mean[p] : 1.0);
      if (opt->sync_phases) sc_out(" %9.4f", tv_sum[TV_WAIT + p] / P / S * 1e3);
      sc_out("%s\n", thin ? "  ! < 100 MPI_Wtick" : "");
    }
    sc_out("  (max/mean of a phase ending in a collective is a LOWER bound on its imbalance)\n");
    sc_out("  (per-step maxima/quantiles include deterministic plane crossings, not just noise; steps are not barriered)\n");

    /* -- inside the redistribution ---------------------------------------- */
    sc_out("[redistribution profile]  per rank, ms per step\n");
    if (!profiled) sc_out("  !! library built without -DMFTL_COMM_PROFILE=ON: rebuild with  python3 build_all.py --comm-profile\n");
    if (nproc == 1) sc_out("  (np = 1: MFTLRedistribute takes its serial short-circuit -- nothing below is communication)\n");
    sc_out("  %-36s %4s %9s %9s %9s %7s %8s\n", "bucket", "kind", "mean", "min", "max", "% red.", "max/mean");
    for (int pass = 0; pass < 2; ++pass)
    {
      for (int i = 0; i < N_RB; ++i)
      {
        if (SC_RB_COMM[i] != pass) continue;
        const double v = tv_sum[TV_RB + i] / P;
        sc_out("  %-36s %4s %9.4f %9.4f %9.4f %7.1f %8.2f\n", SC_RB_LABEL[i], pass ? "COMM" : "work", v / S * 1e3, tv_min[TV_RB + i] / S * 1e3,
               tv_max[TV_RB + i] / S * 1e3, (red_total > 0.0) ? 100.0 * v / red_total : 0.0, (v > 0.0) ? tv_max[TV_RB + i] / v : 1.0);
      }
    }
    sc_out("  %-36s %4s %9.4f %9s %9s %7.1f\n", "local work", "", red_local / S * 1e3, "", "", (red_total > 0.0) ? 100.0 * red_local / red_total : 0.0);
    sc_out("  %-36s %4s %9.4f %9s %9s %7.1f\n", "communication + sync", "", red_comm / S * 1e3, "", "", (red_total > 0.0) ? 100.0 * red_comm / red_total : 0.0);
    sc_out("  %-36s %4s %9.4f %9.4f %9.4f\n", "TOTAL (library timer)", "", red_total / S * 1e3, tv_min[TV_RTOT] / S * 1e3, tv_max[TV_RTOT] / S * 1e3);
    sc_out("  %-36s %4s %9.4f   <- a large residue means an untimed region\n", "residue (total - buckets)", "", (red_total - red_local - red_comm) / S * 1e3);
    sc_out("  %-36s %4s %9.4f   (%ld clock reads/rank/step at %.1f ns)\n", "instrument perturbation (est.)", "", perturb_ms,
           (long) ((double) rc_sum[RC_clock_reads] / P / S), clock_ns_max);

    sc_out("[redistribution traffic]  per step\n");
    sc_out("  triangles scanned      %12.0f total   owner changed %.3f %%\n", (double) rc_sum[RC_tris_scanned] / S, owner_pct);
    sc_out("  vertices  scanned      %12.0f total   owner changed %.3f %%\n", (double) rc_sum[RC_verts_scanned] / S,
           (rc_sum[RC_verts_scanned] > 0) ? 100.0 * (double) rc_sum[RC_verts_moved] / (double) rc_sum[RC_verts_scanned] : 0.0);
    sc_out("  kept / received        %12.0f / %.0f triangles,  %.0f / %.0f vertices\n", (double) rc_sum[RC_tris_kept] / S, (double) rc_sum[RC_tris_recv] / S,
           (double) rc_sum[RC_verts_kept] / S, (double) rc_sum[RC_verts_recv] / S);
    sc_out("  bytes sent per rank    %12.4f MB mean, %.4f MB busiest rank\n", (double) rc_sum[RC_bytes_sent] * mb / P / S, (double) rc_max[RC_bytes_sent] * mb / S);
    sc_out("  bytes recv per rank    %12.4f MB mean, %.4f MB busiest rank\n", (double) rc_sum[RC_bytes_recv] * mb / P / S, (double) rc_max[RC_bytes_recv] * mb / S);
    sc_out("  messages per rank      %12.2f mean, %.2f max   (body x destination pairs %.2f mean)\n", (double) rc_sum[RC_messages] / P / S,
           (double) rc_max[RC_messages] / S, (double) rc_sum[RC_destinations] / P / S);
    sc_out("  route neighbour/global %12ld / %ld calls (summed over ranks; global must stay 0 in steady state)\n", rc_sum[RC_route_neighbor],
           rc_sum[RC_route_global]);
    sc_out("  bodies visited/rank    %12.2f   skipped %.1f %%  reclassified %.2f  settled %.2f  empty %.2f\n", (double) rc_sum[RC_bodies_visited] / P / S, skip_pct,
           (double) rc_sum[RC_bodies_reclassified] / P / S, (double) rc_sum[RC_bodies_settled] / P / S, (double) rc_sum[RC_bodies_empty] / P / S);
    sc_out("  objects created/destr. %12.2f / %.2f per rank (sparse object list churn)\n", (double) rc_sum[RC_bodies_created] / P / S,
           (double) rc_sum[RC_bodies_destroyed] / P / S);
    sc_out("  allocations per rank   %12.1f calls, %.3f MB requested (id tables %.3f MB)\n", (double) rc_sum[RC_allocs] / P / S,
           (double) rc_sum[RC_bytes_allocated] * mb / P / S, (double) rc_sum[RC_gid_table_bytes] * mb / P / S);
    sc_out("  id tables direct/sparse%12ld / %ld uses;  dedup overflow %ld\n", rc_sum[RC_gid_direct], rc_sum[RC_gid_sparse], rc_sum[RC_dedup_overflow]);
    sc_out("  workspace min..max/rank  per body %.1f..%.1f kB   per RANK %.1f..%.1f kB   per entity %.1f..%.1f kB\n", (double) rc_min[RC_ws_per_body] / 1024.0,
           (double) rc_max[RC_ws_per_body] / 1024.0, (double) rc_min[RC_ws_per_rank] / 1024.0, (double) rc_max[RC_ws_per_rank] / 1024.0,
           (double) rc_min[RC_ws_per_entity] / 1024.0, (double) rc_max[RC_ws_per_entity] / 1024.0);
    sc_out("  per-body reductions    %12ld on holders, %ld global, %.4f MB/step all ranks\n", br_sum[0], br_sum[1], (double) br_sum[2] * mb / S);

    sc_out("[ghost exchanges]  positions%s, per rank, ms per step\n", opt->normals ? " + vertex normals" : "");
    for (int i = 0; i < N_GB; ++i)
    {
      const double v = tv_sum[TV_GB + i] / P;
      sc_out("  %-36s %4s %9.4f   max %9.4f\n", SC_GB_LABEL[i], SC_GB_COMM[i] ? "COMM" : "work", v / S * 1e3, tv_max[TV_GB + i] / S * 1e3);
    }
    sc_out("  %-36s %4s %9.4f   max %9.4f\n", "TOTAL (library timer)", "", gs_total / S * 1e3, tv_max[TV_GTOT] / S * 1e3);
    sc_out("  ghosts requested/rank  %12.0f per step, %.4f MB sent+recv per rank\n", (double) rc_sum[RC_g_requested] / P / S,
           (double) (rc_sum[RC_g_bytes_sent] + rc_sum[RC_g_bytes_recv]) * mb / P / S);

    sc_out("[partition]  averaged over the timed steps of the reported repeat\n");
    sc_out("  owned triangles/rank   min %.0f  mean %.0f  max %.0f (time averages per rank)\n", part_min[0], owned_mean, part_max[0]);
    sc_out("  structural imbalance   %.3f   busiest rank / mean, per step, averaged (worst step %.3f)\n", struct_imb, imb_peak);
    sc_out("  ghost triangles        %.2f %% of owned (halo)\n", 100.0 * halo);
    sc_out("  bodies held/rank       min %.1f  mean %.1f  max %.1f\n", part_min[2], bodies_mean, part_max[2]);
    sc_out("  graph degree           %ld..%ld distinct neighbour ranks (26 needs >= 3 cuts per axis)\n", pv_min[PV_DEGREE], pv_max[PV_DEGREE]);

    sc_out("[memory]\n");
    sc_out("  peak RSS per rank      max %.1f MB, mean %.1f MB\n", (double) pv_max[PV_RSS_PEAK] * mb, (double) pv_sum[PV_RSS_PEAK] * mb / P);
    sc_out("  RSS peak-sum per node  max %.1f MB (upper bound: rank peaks need not coincide; shared pages counted repeatedly)   current %.1f MB\n", (double) node_rss_max[0] * mb, (double) node_rss_max[1] * mb);
    sc_out("  pool capacity per rank max %.1f MB, mean %.1f MB\n", (double) pv_max[PV_POOL_BYTES] * mb, (double) pv_sum[PV_POOL_BYTES] * mb / P);

    sc_out("[checks]\n");
    sc_out("  conservation           %s (worst area rel. deviation %.2e; triangles and vertices exact)\n", failures_total ? "FAILED" : "ok", worst_area_dev);
    sc_out("  orphaned ghosts        %s (%ld, all repeats + warm-up)\n", orphaned ? "FAILED" : "ok", check_sum[0]);
    sc_out("  centres recomputed     %s (%ld; non-zero = geometry pass missing before redistribution)\n", recomputed ? "FAILED" : "ok",
           check_sum[1]);
    sc_out("  global route           %s (%ld, all repeats + warm-up; this benchmark expects neighbour routing)\n", global_route ? "FAILED" : "ok", check_sum[2]);
    sc_out("  redistributions/rank   %.0f (expected %d)\n", per_calls, ns);
    sc_out("  verdict                %s\n", exit_code ? "FAILED -- do not use these numbers" : "ok");

    /* -- one machine-readable line for collect.sh --------------------------- */
    sc_out("@summary mode=%s motion=%s np=%d nodes=%d rpn=%d cuts=%dx%dx%d bodies=%d refine=%d tris=%ld tris_rank=%.0f steps=%d", prob->mode, SC_MOTION_NAME[opt->motion], nproc, n_nodes, rpn_max,
            cuts[0], cuts[1], cuts[2], nb, opt->refine, g_tris, owned_mean, ns);
    sc_out(" wall_s=%.6g ms_step=%.6g rep_spread=%.4g ms_median=%.6g ms_max=%.6g sync_phases=%d", best_wall, ms_step,
           (rw[completed_repeats - 1] - rw[0]) / rw[0], 1e3 * median_wall / S, 1e3 * rw[completed_repeats - 1] / S, opt->sync_phases);
    sc_out(" cycle_steps=%d cycles=%d", opt->steps, opt->cycles);
    sc_out(" normals=%d skip=%d phi=%.8g jitter=%.8g dt=%.8g speed=%.8g shrink=%.8g warmup=%d repeats=%d", opt->normals,
           opt->skip, opt->phi, opt->jitter, opt->dt, opt->speed, opt->shrink_slack, opt->warmup, completed_repeats);
    sc_out(" requested_repeats=%d budget_stop=%d init_batch=%d init_batches=%d init_batch_pool_mb=%.6g init_s=%.6g init_signature=%016llx",
           opt->repeat, budget_stop, batch_bodies, setup.batches, setup.batch_pool_peak * mb, setup.wall_s, setup.signature);
    for (int p = 0; p < N_PHASE; ++p) sc_out(" %s=%.6g", SC_PHASE_KEY[p], ph_mean[p] / S * 1e3);
    sc_out(" redist_max=%.6g redist_lib=%.6g redist_work=%.6g redist_comm=%.6g", tv_max[TV_PHASE + PH_REDIST] / S * 1e3, red_total / S * 1e3,
            red_local / S * 1e3, red_comm / S * 1e3);
    for (int i = 0; i < N_RB; ++i) sc_out(" rb_%s=%.6g", SC_RB_KEY[i], tv_sum[TV_RB + i] / P / S * 1e3);
    for (int i = 0; i < N_RB; ++i) sc_out(" rbmax_%s=%.6g", SC_RB_KEY[i], tv_max[TV_RB + i] / S * 1e3);
    sc_out(" ghost_lib=%.6g ghost_comm=%.6g comm_pct=%.4g step_p50=%.6g step_p90=%.6g step_max=%.6g", gs_total / S * 1e3, gs_comm / S * 1e3, profiled ? comm_pct : -1.0,
            1e3 * sc_quantile(step_tot_max, ns, 0.5), 1e3 * sc_quantile(step_tot_max, ns, 0.9), 1e3 * step_tot_max[ns - 1]);
    sc_out(" mb_sent_rank=%.6g mb_sent_max=%.6g msgs_rank=%.4g owner_pct=%.4g skip_pct=%.4g route_global=%ld", (double) rc_sum[RC_bytes_sent] * mb / P / S,
            (double) rc_max[RC_bytes_sent] * mb / S, (double) rc_sum[RC_messages] / P / S, owner_pct, skip_pct, rc_sum[RC_route_global]);
    sc_out(" imbalance=%.4g imb_peak=%.4g halo_pct=%.4g bodies_rank=%.4g degree=%ld ws_rank_kb=%.4g", struct_imb, imb_peak, 100.0 * halo, bodies_mean,
            pv_max[PV_DEGREE], (double) rc_max[RC_ws_per_rank] / 1024.0);
    sc_out(" rss_rank_current_mb=%.6g rss_node_current_mb=%.6g pool_rank_mb=%.6g", (double) pv_max[PV_RSS_NOW] * mb,
           (double) node_rss_max[1] * mb, (double) pv_max[PV_POOL_BYTES] * mb);
    sc_out(" rss_rank_mb=%.5g rss_node_mb=%.6g gen_s=%.4g scatter_s=%.4g perturb_ms=%.4g profiled=%d optimised=%d ok=%d\n", (double) pv_max[PV_RSS_PEAK] * mb,
            (double) node_rss_max[0] * mb, t_gen, t_scatter, perturb_ms, profiled, optimised, exit_code ? 0 : 1);
    for (int rep = 0; rep < completed_repeats; ++rep)
      sc_out("@repeat np=%d repeat=%d ms_step=%.9g\n", nproc, rep + 1, 1e3 * rep_wall[rep] / S);

    if (sc_report_fp)
    {
      const int write_error = ferror(sc_report_fp);
      const int close_error = fclose(sc_report_fp);
      sc_report_fp = MFTL_NULL;
      if (write_error || close_error)
      {
        fprintf(stderr, "could not finish report %s\n", opt->report);
        exit_code = 2;
      }
      else printf("report written to %s\n", opt->report);
    }
    free(rw);
  }

  MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);

  free(step_tot);
  free(step_red);
  free(best_tot);
  free(best_red);
  free(step_tot_max);
  free(step_red_max);
  free(step_own);
  free(best_own);
  free(step_own_max);
  free(rep_wall);
  MPI_Comm_free(&node_comm);
  MFTLEnsembleDestroy(&ens);
  return exit_code;
}

#endif  // MFTL_SCALING_COMMON_H
