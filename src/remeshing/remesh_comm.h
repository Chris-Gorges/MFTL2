/**
 * @file remesh_comm.h
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief EVERY MPI CALL THE REMESHING MODULE MAKES, AND NOTHING ELSE.
 *
 * ## WHY THIS FILE EXISTS
 *
 * The first version of the distributed remesher reached the network by growing
 * the production communication primitives:
 *
 *   - MFTLGetOwnerProc gained a per-axis cut-plane offset and a periodicity
 *     branch, so that a shifted decomposition could be expressed by subtracting
 *     from the position. That function is the body of `compute_owners`, i.e. it
 *     runs once per vertex and once per triangle on every redistributed
 *     timestep, forever, to serve a routine that runs a few times per hundred.
 *
 *   - The per-timestep ghost exchange gained a third field, "tell me my new
 *     global id", and with it two `field ==` tests inside the loop that applies
 *     replies to ghost vertices.
 *
 * Neither was large and both were the wrong shape: a feature used once per
 * remesh was paid for once per timestep, inside the two loops the library is
 * most careful about. Both are reverted. What replaces them is this file, plus
 * ONE seam in redistribution -- MFTL_OwnerMap (mftl_communication.h), a NULL
 * pointer on the production path and one predicted branch per body.
 *
 * ## THE PROPERTY THIS FILE IS FOR
 *
 *     Of every source in src/remeshing, only this one may contain an MPI call.
 *
 * remesh_ops.c, remesh_smooth.c, remesh_compact.c, remesh_driver.c,
 * remesh_parallel.c and remesh_dispersed.c contain no MPI at all: the operators
 * and the two drivers are pure local computation over a partition somebody else
 * arranged. Checked by `test_remeshing_mpi_is_confined` in the test suite, and
 * it is what makes the serial path provably unaffected by any of this.
 *
 * ## WHAT IS HERE, AND HOW EACH ONE SCALES
 *
 *   MFTLRemeshMigrateShifted    26-neighbour. One redistribution under moved
 *                               cut planes. Volume is a LARGE FRACTION OF THE
 *                               MESH -- see below -- plus one int of latency for
 *                               the identity test that skips it entirely when
 *                               the placement is the one already in force.
 *   MFTLRemeshMigrateToRanks    26-neighbour when the bodies are smaller than a
 *                               rank box, which is the regime that asks for it;
 *                               redistribution detects the exception itself and
 *                               falls back to the global path.
 *   MFTLRemeshSumInts           1 Allreduce of 3 ints. Pure latency. Once per
 *                               phase, carrying the convergence certificate.
 *   MFTLRemeshSyncGhostIds      3 neighbourhood collectives, 4 bytes per ghost
 *                               vertex. Once per remeshing call. Falls back to
 *                               the GLOBAL communicator when any ghost's owner
 *                               is not a graph neighbour -- which is what the
 *                               coarsened placement makes normal.
 *   MFTLRemeshMeasureBodies     9 SUM/MIN values per LOCAL body over that body's
 *                               holders, then 1 MAX value per local body, then
 *                               one scalar MPI_MAX for the largest body in the
 *                               ensemble. Nothing here is proportional to the id
 *                               space. ONCE per pass, not once per driver:
 *                               MFTLRemeshEnsemble threads the measurement from
 *                               the strategy choice into the dispersed driver.
 *   MFTLRemeshAgreeIntMax       1 Allreduce of n ints, n a small constant. Id
 *                               counters and watermarks are per-body and go
 *                               through MFTLEnsembleReduceBodies() instead.
 *   MFTLRemeshAgreeDoubleMax    1 Allreduce of 8 doubles. The profile and the
 *                               band measurement, once per pass.
 *   MFTLRemeshNeighborGather    26-neighbour, variable length. What one rank
 *                               holds, delivered one hop. The dispersed
 *                               balancer's two publishes: `(body, home, weight)`
 *                               for the bodies a rank speaks for, and
 *                               `(body, destination, weight)` for the ones it
 *                               moves. Three doubles each, and NOTHING sized by
 *                               `n_bodies_global`.
 *   MFTLRemeshNeighborAllgather 26-neighbour, fixed length. One double: the room
 *                               a rank advertises before the negotiation starts.
 *   MFTLRemeshNeighborAlltoall  26-neighbour, DIRECTED, one or two doubles. The
 *                               ask, the grant and the shipping report of one
 *                               balancing round, three times over. Latency, not
 *                               volume -- and it is what a purely local balancer
 *                               costs instead of the `4 * n_bodies_global`
 *                               doubles the replicated one broadcast.
 *
 * ### HOW MUCH A SHIFTED MIGRATION ACTUALLY MOVES
 *
 * This used to read "O(band), not O(mesh)" and that was simply wrong. Between a
 * placement at shift `a` and one at shift `b`, an entity changes owner on axis k
 * exactly when its coordinate falls in [a_k, b_k) modulo h_k, so the moved
 * fraction is
 *
 *     1 - PROD_k (1 - |b_k - a_k| / h_k)
 *
 * and for the actual table that is 87.5 % for (0,0,0) -> (1,1,1), 50 % for a
 * single-axis change and 75 % for two. Measured directly: MFTL_RemeshStats
 * .n_entities_migrated. The band is what gets FROZEN; it is not what moves.
 *
 * Nor is the claim that nothing here allocates or messages O(nproc) true, and it
 * never was: MFTLRemeshSyncGhostIds allocates one int per rank to invert the
 * neighbour list, and every migration goes through redistribute_core, which
 * allocates several arrays sized by nproc and loops over them. At P in the
 * hundreds that is noise; at P in the thousands, times the phase count, it is
 * the wall. Recorded here rather than fixed -- it is deferred issue 6 in
 * CLAUDE.md and it belongs to redistribution, not to this file.
 */

#ifndef MFTL_REMESH_COMM_H
#define MFTL_REMESH_COMM_H

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_remeshing.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Ownership rules -- local, and deliberately not in mftl_geometry.h
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The per-axis offset of the cut planes for one phase of the schedule.
 *
 * @p phase entries are 0 or 1; the eight combinations are MFTL_REMESH_PHASES.
 * An axis with a single cut has no interior plane to move, and shifting it would
 * only rotate the one cell against the periodic seam -- migration for no gain in
 * coverage -- so it is left alone.
 *
 * Every rank must call this with the same @p phase before the migration that
 * follows. It contains no MPI, so nothing detects a disagreement, and a
 * disagreement makes ownership AMBIGUOUS rather than merely suboptimal. What
 * keeps it true is that the phase index comes from a compile-time table and a
 * loop counter, never from data.
 *
 * @param      domain The MPI/physical world.
 * @param      group  Per-axis coarsening factors, from MFTLRemeshCoarseningFactors().
 * @param      phase  Per-axis phase entry, 0 or 1.
 * @param[out] out    Per-axis cut-plane offset in domain length units.
 */
static inline void MFTLRemeshPhaseShift(const MFTL_Domain *domain, const int group[3], const double phase[3], double out[3])
{
  for (int k = 0; k < 3; ++k)
  {
    const int g = (group && group[k] > 1) ? group[k] : 1;

    /* The planes that move are the COARSE ones, so the box to halve is g cells
     * wide -- and an axis with fewer than two coarse cells has no interior plane
     * at all, which is the same test as before with `cuts` replaced by
     * `cuts / g`. Getting this wrong is silent: the shift would be half a FINE
     * cell, the band would not clear it, and the schedule would simply stop
     * covering while every structural check still passed. */
    const int n_groups = domain->cuts[k] / g;
    const double h = (n_groups > 1) ? (double) g * domain->domain_size[k] / (double) domain->cuts[k] : 0.0;
    out[k] = phase[k] * 0.5 * h;
  }
}

/**
 * @brief The rank owning @p pos when every cut plane of axis k sits at
 *        `global_min[k] + shift[k] + j*h_k`.
 *
 * MFTLGetOwnerProc with the planes moved, and the shift is exactly what forces
 * the extra branch. Without it the argument of the truncation is non-negative
 * for any wrapped position, so `(int)` IS `floor` and clamping is the same as
 * wrapping. With a shift of up to h/2 the argument reaches -0.5, where `(int)`
 * truncates TOWARDS ZERO and would report cell 0 for a point belonging to cell
 * `cuts-1` on the far side of the seam. That is a silent half-cell of
 * misattributed ownership right at the seam -- which is precisely where the
 * schedule is trying to put the frozen band.
 *
 * The bias by `cuts` makes the truncation a floor again for one add and one
 * predicated subtract. The non-periodic path keeps the clamp, because there a
 * negative argument really does mean "off the end of the grid".
 *
 * @p pos must be WRAPPED into the domain, exactly as for MFTLGetOwnerProc.
 *
 * @param domain The MPI/physical world.
 * @param k      Axis, 0..2.
 * @param shift  Per-axis cut-plane offset.
 * @param pos    A position, already wrapped into the domain.
 * @return The FINE cell index along axis @p k, clamped to `[0, cuts[k])`.
 */
static inline int MFTLRemeshFineCell(const MFTL_Domain *domain, int k, const double shift[3], const double pos[3])
{
  const int cuts = domain->cuts[k];
  const double s = (pos[k] - domain->global_min[k] - shift[k]) * domain->grid_factors[k];

  int i;
  if (domain->is_periodic[k])
  {
    /* s lies in [-0.5, cuts) for any wrapped position and any shift in
     * [0, h/2], so a single bias-and-subtract is a complete modulo. */
    i = (int) (s + (double) cuts);
    if (i >= cuts) i -= cuts;
  }
  else
  {
    i = (int) s;
  }

  if (i < 0) i = 0;
  if (i >= cuts) i = cuts - 1;
  return i;
}

/**
 * @brief MFTLGetOwnerProc() evaluated against cut planes offset by @p shift.
 *
 * @param domain The MPI/physical world.
 * @param shift  Per-axis cut-plane offset from MFTLRemeshPhaseShift().
 * @param pos    A position, already wrapped into the domain.
 * @return The rank owning @p pos under the shifted decomposition.
 *
 * @warning A shifted plane needs a FLOOR, not a truncation. On a periodic axis
 *          the argument reaches \f$-0.5\f$, where `(int)` truncates *towards
 *          zero* and attributes the sliver at the domain minimum to cell 0
 *          instead of to cell `cuts-1` across the seam. The wrong rule is still
 *          a consistent rule -- every rank computes the same answer and every
 *          structural invariant holds -- so the only symptom is that the SEAM
 *          PLANE NEVER MOVES and the entities on it are frozen in every phase.
 */
static inline int MFTLRemeshOwnerProc(const MFTL_Domain *domain, const double shift[3], const double pos[3])
{
  const int c0 = MFTLRemeshFineCell(domain, 0, shift, pos);
  const int c1 = MFTLRemeshFineCell(domain, 1, shift, pos);
  const int c2 = MFTLRemeshFineCell(domain, 2, shift, pos);
  return domain->rank_map[c0 + c1 * domain->cuts[0] + c2 * domain->cuts[0] * domain->cuts[1]];
}

/**
 * @brief MFTLRemeshOwnerProc on a decomposition COARSENED by @p group per axis.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * REGIME C WITHOUT A SECOND COMMUNICATION SUBSYSTEM
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The shift schedule needs `h > 4*delta`. Raise the rank count on a fixed domain
 * and h shrinks until it does not, and then the scheme DEGRADES SILENTLY: bands
 * along the cut planes that no placement ever offers to anyone, every one of
 * them a perfectly valid piece of mesh. That is the regime the implementation
 * plan calls C, and its answer there was MPI_Comm_split plus a gather onto a
 * group master plus a scatter back -- a whole second communication subsystem,
 * which is why it was never built.
 *
 * It is not needed. Redistribution already accepts an arbitrary destination per
 * entity (MFTL_OwnerMap), which is how the dispersed strategy routes by body
 * rather than by plane. So a coarser decomposition is just a different OWNER
 * FUNCTION: take the fine cell, integer-divide by the group size, map back to a
 * representative fine cell, look up its rank. Every rank still enters every
 * collective; the ones that are not representatives simply receive nothing and
 * contribute zeros -- exactly the situation the dispersed driver is already in
 * whenever there are fewer bodies than ranks.
 *
 * Nothing else in the driver changes. The freeze rule is `index >= n_local` and
 * is therefore partition-agnostic. The covering argument transfers verbatim with
 * `H_k = group[k] * h_k`, and since `delta` is unchanged and H grows, making it
 * hold IS the point. The closing MFTLRedistribute passes no owner map at all, so
 * the fine canonical partition comes back by construction.
 *
 * ── WHAT IT COSTS, STATED PLAINLY ──────────────────────────────────────────
 *
 * `1 / (group[0]*group[1]*group[2])` of the ranks do all of the sweep work. The
 * remesh is 96-99 % of a timestep in the measured case, so this is a CORRECTNESS
 * ESCAPE HATCH, NOT A PERFORMANCE FEATURE. Remeshing to a longer target length,
 * or running on fewer ranks, beats it whenever either is available. And because
 * a destination is now up to `group[k]` cells away, it is outside the
 * 26-neighbourhood, so redistribution falls back to its global exchange path and
 * MFTLRemeshSyncGhostIds falls back to its -- both detect it themselves.
 *
 * The factors are PER AXIS and that is not a detail. The precondition is per
 * axis (`h_k > 4*delta` for every cut axis; MFTLDomainGetHMin takes the minimum
 * precisely because of that), so a domain that is thin in x only needs merging
 * in x only: 2x parallelism lost instead of 8x.
 *
 * @p group entries are >= 1; {1,1,1} is bit-for-bit MFTLRemeshOwnerProc.
 * @p pos must be WRAPPED into the domain.
 *
 * @param domain The MPI/physical world.
 * @param group  Per-axis coarsening factors, each >= 1.
 * @param shift  Per-axis cut-plane offset.
 * @param pos    A position, already wrapped into the domain.
 * @return The rank owning @p pos under the COARSENED, shifted decomposition.
 *         Every rank of a group maps to the same answer, so a group has one
 *         owner and no second communicator is needed.
 */
static inline int MFTLRemeshCoarseOwnerProc(const MFTL_Domain *domain, const int group[3], const double shift[3], const double pos[3])
{
  int c[3];
  for (int k = 0; k < 3; ++k)
  {
    /* The fine cell comes from the SAME function the uncoarsened rule uses. Two
     * copies of the periodic floor is exactly the kind of duplication that
     * drifts, and the thing it would drift away from is a silent half-cell of
     * misattributed ownership at the seam. */
    const int i = MFTLRemeshFineCell(domain, k, shift, pos);
    const int cuts = domain->cuts[k];
    const int g = (group[k] > 1) ? group[k] : 1;

    /* THE REMAINDER GOES INTO THE LAST GROUP, never into a short one of its own.
     * A short group is narrower than the others, so its H is smaller, and if it
     * falls below 4*delta its planes are exactly the ones that never move -- the
     * permanently frozen seam this whole path exists to remove, reintroduced at
     * the far end of every axis. Absorbing the remainder makes the last group up
     * to 2g-1 cells wide, which is only ever wider than it needs to be. */
    const int n_groups = (cuts >= g) ? cuts / g : 1;
    int grp = i / g;
    if (grp >= n_groups) grp = n_groups - 1;

    c[k] = grp * g;
  }
  return domain->rank_map[c[0] + c[1] * domain->cuts[0] + c[2] * domain->cuts[0] * domain->cuts[1]];
}

/**
 * @brief Shrinks @p group until it merges at most @p cap ranks per group.
 *
 * Shared rather than duplicated because BOTH paths that coarsen have to apply
 * the same cap: the spanning driver caps the factors it partitions by, and the
 * dispersed strategy caps the factors it uses as a yardstick for "is a body
 * small enough for one rank". Two copies of this loop would eventually differ,
 * and the symptom would be the chooser electing DISPERSED on a body the driver
 * then rejects -- one wasted migration and a fall back to the spanning path,
 * reported only as `n_bodies_not_centred`.
 *
 * The largest factor is shrunk first, which keeps the merged boxes as close to
 * cubic as the cap allows. A cap below 1 is ignored; @p cap <= 1 means no cap.
 *
 * @param[in,out] group [3] per-axis factors, each >= 1.
 * @param         cap   Maximum `group[0]*group[1]*group[2]`. <= 1 is a no-op.
 * @rank_local Pure arithmetic; every rank must call it with the same inputs.
 */
static inline void MFTLRemeshCapGroups(int group[3], int cap)
{
  if (cap <= 1) return;

  while (group[0] * group[1] * group[2] > cap)
  {
    int worst = 0;
    for (int k = 1; k < 3; ++k)
    {
      if (group[k] > group[worst]) worst = k;
    }
    if (group[worst] <= 1) break;
    group[worst]--;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Migration
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_RemeshMigrator
 * @brief Reusable scratch for the owner maps a remeshing pass builds.
 *
 * A spanning pass migrates up to `parallel_phase_budget` times and each
 * migration needs one destination per entity of every body. Allocating that per
 * phase would repeat the first-touch page faults every time -- which is
 * measured, on the redistribution's own send buffers, as the largest single
 * source of timing noise in the per-timestep loop (deferred issue 6, CLAUDE.md).
 *
 * So the arena is grown geometrically and reused, and it belongs to the CALLER
 * rather than to MFTL_Domain: the domain is read-only during a redistribution,
 * and a phase-local buffer hanging off it would be mutable per-call state on a
 * struct whose whole contract is that it is not.
 *
 * BOTH ARRAYS ARE SIZED BY WHAT THIS RANK HOLDS, never by the id space. `maps`
 * is indexed by local slot because the only thing that reads it walks the same
 * list in the same order on the same rank; keyed by body gid it was the last
 * `O(bodies in the ensemble)` allocation on the per-phase path.
 */
typedef struct MFTL_RemeshMigrator
{
  MFTL_OwnerMap *maps;  ///< [cap_objects] by LOCAL SLOT, handed straight to MFTLRedistributeWithOwners.
  int *arena;  ///< [cap_entities] all bodies' vertex and triangle destinations, back to back.
  int cap_objects;  ///< LOCAL bodies the `maps` array can describe. Never the id space -- see migrator_reserve().
  int cap_entities;  ///< Total entity slots the arena holds. Grown monotonically.
} MFTL_RemeshMigrator;

/**
 * @struct MFTL_RemeshMigrationReport
 * @brief What a migration actually did, for the profile and for the tests.
 *
 * `performed` is false when the requested placement turned out to be the one the
 * mesh already had, so the redistribution was skipped -- which is the normal
 * case for phase 0 of the schedule. It is a GLOBAL verdict: redistribution is
 * collective, so either every rank skips or none does.
 */
typedef struct MFTL_RemeshMigrationReport
{
  int n_moved;  ///< OWNED entities of this rank whose destination was elsewhere.
  bool performed;  ///< False when the migration was the identity and was skipped.
} MFTL_RemeshMigrationReport;

/**
 * @brief Zeroes the handle. Nothing is allocated until the first migration.
 * @param m The migrator handle.
 */
void MFTLRemeshMigratorInit(MFTL_RemeshMigrator *m);

/**
 * @brief Releases the arena. Safe on a zeroed handle and on a used one.
 * @param m The migrator handle.
 */
void MFTLRemeshMigratorFree(MFTL_RemeshMigrator *m);

/**
 * @brief Redistributes the ensemble under cut planes offset by @p shift.
 *
 * Routes vertices by position and triangles by their cached CENTRE, i.e. the
 * same two arrays and the same rule as the production path, evaluated against
 * moved planes. Ghosts are routed too, and must be: a rank routes its ghosts in
 * order to work out which of its own triangles the ghost's owner is about to
 * need, so the halo would otherwise come back with holes.
 *
 * Preconditions are the redistribution's own -- current derived arrays, and
 * centres refreshed over the FULL triangle count including ghosts.
 *
 * A placement that every entity already satisfies is SKIPPED -- see the body for
 * why that is sound and why the test has to be global. Phase 0 of the schedule
 * is exactly that case.
 *
 * @param      m      Migrator handle, reused across calls.
 * @param      ens    The ensemble.
 * @param      group  Per-axis coarsening factors.
 * @param      shift  Per-axis cut-plane offset for this phase.
 * @param[out] report Optional. See #MFTL_RemeshMigrationReport.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the redistribution failed.
 */
int MFTLRemeshMigrateShifted(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens, const int group[3], const double shift[3], MFTL_RemeshMigrationReport *report);

/**
 * @brief Redistributes the ensemble so that local body `o` lands wholly on `dest[o]`.
 *
 * The object-centering migration of the dispersed strategy: every entity of a
 * body -- owned and ghost, vertex and triangle -- is given the same
 * destination, so the body arrives complete on one rank with no ghosts and the
 * untouched serial kernel can run on all of it.
 *
 * @p dest is indexed by LOCAL SLOT, but every HOLDER of a body must still have
 * arrived at the same destination for it -- #MFTL_OwnerMap is undefined, not
 * merely slow, when two ranks name different destinations for one entity. Both
 * of the dispersed driver's paths satisfy that: the unbalanced one applies
 * MFTLGetOwnerProc() to a centroid every holder received identically to the bit
 * from MFTLEnsembleReduceBodies(), and the balanced one decides each body on ONE
 * rank and delivers the verdict to every holder one hop along `comm_graph`. It
 * is a 26-neighbour exchange exactly when each body fits
 * inside a rank box; when it does not, redistribution notices by itself and
 * falls back to the global communicator, so the failure mode is a slower message
 * pattern and not a wrong answer.
 *
 * @param      m      Migrator handle, reused across calls.
 * @param      ens    The ensemble.
 * @param      dest   `[ens->n_local_objects]` destination rank per LOCAL SLOT.
 * @param[out] report Optional. See #MFTL_RemeshMigrationReport.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the redistribution failed.
 * @collective
 */
int MFTLRemeshMigrateToRanks(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens, const int *dest, MFTL_RemeshMigrationReport *report);

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Collectives
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief In-place `MPI_Allreduce(MPI_SUM)` of @p n ints over the world.
 *
 * The convergence certificate and nothing else: three ints, once per phase, of
 * pure latency.
 *
 * ## THE COLLECTIVE THIS REPLACED, AND WHY IT WENT
 *
 * The phase used to hand out its new global ids with an `MPI_Exscan` plus an
 * `MPI_Allreduce` of `2 * n_bodies_global` ints, and these three counters rode
 * along in the same message -- free, because the message was already
 * ensemble-sized. The ids now go through MFTLEnsembleExscanBodies(), which is a
 * per-body scan over that body's HOLDERS and `O(local bodies)`, so appending
 * three ensemble-wide ints to it would put a `comm_world` reduction back on
 * every phase. One small latency is the price of removing an
 * `O(bodies in the ensemble)` payload.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their sums over all ranks.
 * @param n      How many. Less than 1 is a no-op.
 * @collective
 */
void MFTLRemeshSumInts(const MFTL_Domain *domain, int *values, int n);

/**
 * @brief In-place MPI_Allreduce(MPI_SUM) of @p n doubles over the world.
 *
 * The mean-edge-length reduction and nothing else so far. Sum rather than a
 * ready-made mean, because a mean of per-rank means is weighted by RANKS and not
 * by edges -- and a rank holding no part of the mesh would contribute a zero
 * that drags the answer down without being anybody's edge.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their sums over all ranks.
 * @param n      Number of values.
 * @collective
 */
void MFTLRemeshAgreeDouble(const MFTL_Domain *domain, double *values, int n);

/**
 * @brief Debug: do all ranks agree on the parameters that decide what to do?
 *
 * A remeshing pass is a set of independent local decisions taken against ONE
 * length band, and nothing in the algorithm communicates the band -- every rank
 * is simply assumed to have been handed the same one. When that assumption
 * breaks the failure is spectacular and its cause is invisible: a rank whose
 * `max_edge_length` is smaller splits everything it owns, every phase, and the
 * mesh grows without bound until the job is killed. There is no wrong answer to
 * find afterwards, only an enormous one.
 *
 * The realistic way in is not a typo. It is deriving the target from the mesh --
 * `MFTLMeanEdgeLength(obj)`, which is LOCAL -- on a mesh that a collective
 * generator has already distributed, so that a rank holding nothing computes
 * zero. MFTLEnsembleMeanEdgeLength exists to remove that route; this check
 * catches every other one.
 *
 * @param domain The MPI/physical world.
 * @param params The parameter block this rank was handed.
 * @return #MFTL_TRUE if every rank passed bitwise-identical bands.
 * @collective
 */
bool MFTLRemeshParamsAgree(const MFTL_Domain *domain, const MFTL_RemeshParameters *params);

/**
 * @brief In-place MPI_Allreduce(MPI_MAX) of @p n ints over the world.
 *
 * Used where a value is known exactly by ONE rank and is zero or stale on the
 * others -- the id counters of a body that a dispersed pass put wholly on its
 * owner, and the Debug watermark agreement check.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their maxima over all ranks.
 * @param n      Number of values.
 * @collective
 */
void MFTLRemeshAgreeIntMax(const MFTL_Domain *domain, int *values, int n);

/**
 * @brief In-place MPI_Allreduce(MPI_MAX) of @p n doubles over the world.
 *
 * The double-precision twin of MFTLRemeshAgreeIntMax, for the two things a pass
 * measures rather than assumes: the wall-clock profile (where the interesting
 * number is the SLOWEST rank, because the phase loop is barrier-synchronised by
 * the id collective) and the observed width of the frozen band.
 *
 * MPI_MAX on doubles is order-independent, so the result is bit-identical on
 * every rank and anything derived from it stays a pure function of replicated
 * data -- which is what lets a driver branch on it without splitting the ranks.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their maxima over all ranks.
 * @param n      Number of values.
 * @collective
 */
void MFTLRemeshAgreeDoubleMax(const MFTL_Domain *domain, double *values, int n);

/**
 * @brief Replaces the six wall-clock fields of @p st, and up to two further
 *        values, by their maxima over all ranks -- in ONE MPI_Allreduce.
 *
 * Both distributed drivers publish their profile this way at the end of a pass,
 * and the spanning driver appends its band measurement to the same message. The
 * fields are gathered into a local array rather than reduced through
 * `&st->t_total`: that spelling would silently depend on the six fields staying
 * adjacent and in this order, and a member added between them is exactly the
 * kind of edit nobody re-reads a reduction for.
 *
 * UNCONDITIONAL in the drivers, even when the caller passed no stats struct: the
 * collective must be entered by every rank.
 *
 * @param         domain  The MPI/physical world.
 * @param[in,out] st      `t_total`, `t_migrate`, `t_geometry`, `t_freeze`,
 *                        `t_sweep` and `t_reduce` are replaced by their maxima.
 * @param[in,out] extra   `[n_extra]` further values, replaced likewise. May be
 *                        #MFTL_NULL when @p n_extra is 0.
 * @param         n_extra 0, 1 or 2.
 * @collective
 */
void MFTLRemeshAgreeProfile(const MFTL_Domain *domain, MFTL_RemeshStats *st, double *extra, int n_extra);

/**
 * @brief Seconds from an arbitrary fixed origin. MPI_Wtime, and nothing else.
 *
 * It looks like a pointless wrapper and it is not: `test_remeshing_mpi_is_confined`
 * is a source scan for `MPI_*(` outside this file, so a bare MPI_Wtime() in
 * remesh_parallel.c fails the BUILD. The rule is worth more than the indirection
 * -- see the header of this file for what it cost the first time it was broken --
 * and the call is not on any hot path: six per phase, against a migration.
 *
 * @return Seconds from an arbitrary fixed origin.
 */
double MFTLRemeshWallTime(void);

/**
 * @brief Tells every ghost vertex the NEW global id its owner has just given it.
 *
 * Parallel remeshing makes the global ids sparse -- splits hand out ids,
 * collapses retire them -- and has to re-densify them, but a rank may only
 * renumber what it owns. This closes the gap for the ghosts.
 *
 * ── Why a renumbering cannot simply be followed by a redistribution ────────
 *
 * Because redistribution KEEPS ghost vertices. A ghost is retained as a corner
 * of a kept owned triangle, and arrivals are de-duplicated against the kept set
 * BY GLOBAL ID. A kept ghost still carrying its pre-renumbering id therefore
 * fails to match the authoritative copy its owner now sends, both survive, and
 * the twin rebuild -- which pairs by vertex SLOT -- tears the surface along the
 * partition boundary. Ghost TRIANGLES have no such problem: they are dropped and
 * re-received every time, which is why only the vertices need this.
 *
 * ── Why it is here and not a third field of MFTLUpdateGhostPositions ───────
 *
 * It asks the same question of the same set, so sharing that schedule was
 * tempting and was in fact tried. The cost landed in the wrong place: the shared
 * function acquired a payload discriminator that every per-timestep position
 * sync then evaluated once per ghost vertex, to serve a call made once per
 * remesh. Written separately it is also strictly cheaper on the wire -- 4 bytes
 * per ghost instead of the 32-byte MFTL_GhostUpdatePayload, and no registered
 * MPI datatype at all.
 *
 * @param ens             The ensemble.
 * @param vertex_gid_base `[ens->n_local_objects]` first new vertex id this rank was
 *                        allotted per body, or #MFTL_NULL when the ids did not move.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the exchange failed.
 */
int MFTLRemeshSyncGhostIds(MFTL_Ensemble *ens, const int *vertex_gid_base);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Neighbourhood exchange -- the three shapes a bounded-degree pass needs
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * THE POINT OF THESE IS WHAT THEY CANNOT EXPRESS. Every one of them moves a
 * payload proportional to `num_neighbors` -- at most 26, whatever `nproc` is --
 * and to what THIS RANK holds. Nothing here can be handed an array indexed by
 * the body id space or by the rank space, so a call site that needs one has to
 * say so by reaching for MFTLRemeshAgreeDouble() instead, where the cost is
 * visible.
 *
 * They exist for the dispersed load balancer, which used to replicate
 * `4 * n_bodies_global` doubles onto every rank once per remesh so that every
 * rank could compute the same greedy over the same table. It now decides each
 * body on ONE rank and delivers the verdict one hop, which is these three
 * collectives and no replicated table at all.
 *
 * ── The self-loop, which is not a corner case ─────────────────────────────
 *
 * `domain->neighbors` is built from the 26 non-zero cell OFFSETS, so on a grid
 * with a single cut on a periodic axis two opposite offsets wrap onto this rank
 * and `comm_graph` carries a SELF-LOOP -- shapes {2,1,1} and {2,2,1} have one,
 * {2,2,2} and {3,3,3} do not. A gather then delivers this rank's own block back
 * to it, and a caller that also folds in its own contribution locally counts it
 * twice. Every consumer here must skip the slot
 * `MFTLDomainNeighborIndex(domain, domain->rank)` (mftl_ensemble.h) reports, and
 * the same trap doubled every SUM in MFTLEnsembleReduceBodies() before it was
 * found.
 */

/**
 * @brief Variable-length allgatherv of doubles over `comm_graph`.
 *
 * MFTLNeighborAllgatherDoubles() (mftl_communication.h), routed through this
 * file so that remesh_comm.c stays the complete inventory of the module's
 * network use.
 *
 * @param      domain The MPI/physical world.
 * @param      send   `[n_send]` this rank's block, delivered to every neighbour.
 * @param      n_send Doubles in @p send. May be zero.
 * @param[out] rcount `[num_neighbors]` doubles received from each neighbour.
 * @param[out] rdispl `[num_neighbors]` offset of each neighbour's block.
 * @param[out] n_recv Total doubles received.
 * @return A newly allocated buffer of @p n_recv doubles; the caller frees it.
 *         Never #MFTL_NULL, so that an empty result still frees cleanly.
 * @collective Over `comm_graph`, so a rank with no neighbours does not
 *             communicate at all.
 */
double *MFTLRemeshNeighborGather(const MFTL_Domain *domain, const double *send, int n_send, int *rcount, int *rdispl, int *n_recv);

/**
 * @brief Fixed-length allgather of @p n doubles over `comm_graph`.
 *
 * @param      domain The MPI/physical world.
 * @param      send   `[n]` this rank's record.
 * @param      n      Doubles per rank.
 * @param[out] recv   `[num_neighbors * n]`; block `i` came from `neighbors[i]`.
 * @collective
 */
void MFTLRemeshNeighborAllgather(const MFTL_Domain *domain, const double *send, int n, double *recv);

/**
 * @brief Directed @p n-doubles-per-neighbour exchange over `comm_graph`.
 *
 * The shape a NEGOTIATION needs and an allgather cannot express: block `i` of
 * @p send goes to `neighbors[i]` only, and block `i` of @p recv is what
 * `neighbors[i]` sent to this rank. The graph is created with identical source
 * and destination lists (`MPI_Dist_graph_create_adjacent` in
 * `domain_init_comm`), so in-degree and out-degree agree and slot `i` means the
 * same rank in both directions.
 *
 * @param      domain The MPI/physical world.
 * @param      send   `[num_neighbors * n]`, one block per neighbour.
 * @param      n      Doubles per neighbour.
 * @param[out] recv   `[num_neighbors * n]`, one block from each neighbour.
 * @collective
 */
void MFTLRemeshNeighborAlltoall(const MFTL_Domain *domain, const double *send, int n, double *recv);

/* MFTLRemeshMeasureBodies is declared in mftl_remeshing.h -- it is the one entry
 * point here a CALLER has a use for (any ensemble rebalancing is exactly
 * "measure the bodies, then decide"), so it is published rather than internal.
 * Its implementation is in remesh_comm.c with the rest of the module's MPI. */

#endif  // MFTL_REMESH_COMM_H
