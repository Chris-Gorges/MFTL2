/**
 * @file remesh_comm.h
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief Every MPI call the remeshing module makes, and nothing else.
 *
 * ## Why this file exists
 *
 * Of every source in src/remeshing only remesh_comm.c may call MPI
 * (`test_remeshing_mpi_is_confined`, a build-time source check); the operators,
 * the smoother, the compaction and both drivers are local computation over a
 * partition arranged here. The rule exists because the first distributed
 * version grew the production primitives instead: a cut-plane offset inside
 * MFTLGetOwnerProc() (1.70x slower `compute_owners`, which runs per entity per
 * timestep) and a "new global id" field in the per-timestep ghost exchange.
 * Both were reverted in favour of this file plus one seam in redistribution,
 * #MFTL_OwnerMap (NULL on the production path).
 *
 * ## What is here, and how each scales
 *
 *   MFTLRemeshMigrateShifted    26-neighbour redistribution under moved planes;
 *                               skipped (one int, LOR) when it is the identity.
 *   MFTLRemeshMigrateToRanks    26-neighbour while bodies fit a rank box;
 *                               redistribution falls back to its global path
 *                               otherwise.
 *   MFTLRemeshSumInts           Allreduce of 3 ints per phase (certificate).
 *   MFTLRemeshSyncGhostIds      3 neighbourhood collectives, 4 bytes per ghost
 *                               vertex, once per pass; global fallback when a
 *                               ghost's owner is not a graph neighbour (normal
 *                               under coarsening).
 *   MFTLRemeshSyncGhostSizing   the same route, 8 bytes per ghost vertex, twice
 *                               per phase -- only under curvature sizing.
 *   MFTLRemeshMeasureBodies     per-body holder reductions (9 values, then 1)
 *                               plus one scalar MPI_MAX; once per pass.
 *   MFTLRemeshAgree*            small fixed-size Allreduces (the parameter block
 *                               once per call, profile, frozen counts, balancer
 *                               scalars, Debug checks).
 *   MFTLRemeshNeighbor*         the balancer's one-hop exchanges: O(local
 *                               bodies) or O(1) per neighbour, never sized by
 *                               n_bodies_global.
 *
 * A shifted migration moves far more than the band: between shifts a and b an
 * entity changes owner on axis k iff its coordinate lies in [a_k, b_k) modulo
 * h_k, so the moved fraction is 1 - PROD_k (1 - |b_k - a_k| / h_k): 87.5 % for
 * (0,0,0) -> (1,1,1), 50 % for one axis, 75 % for two
 * (MFTL_RemeshStats::n_entities_migrated). The band is what is FROZEN, not what
 * moves. Not everything here is O(degree) either: MFTLRemeshSyncGhostIds() and
 * the redistribution behind every migration allocate arrays sized by nproc,
 * which matters at P in the thousands times the phase count.
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
 * @brief MFTLRemeshOwnerProc() on a decomposition COARSENED by @p group per axis.
 *
 * The shift schedule needs \f$h > 4\delta\f$; when more ranks shrink h below
 * that, bands along the cut planes are silently never remeshed. A coarser
 * decomposition needs no second communicator: redistribution accepts any
 * destination per entity (#MFTL_OwnerMap), so coarsening is just a different
 * owner function -- fine cell, integer-divided by the group size, mapped to a
 * representative fine cell's rank. Every rank still enters every collective;
 * non-representatives receive nothing. The freeze rule (`index >= n_local`) and
 * the covering argument transfer with \f$H_k = g_k h_k\f$, and the closing
 * MFTLRedistribute() restores the fine partition.
 *
 * Cost: \f$1/(g_0g_1g_2)\f$ of the ranks do all the sweeping, and destinations
 * leave the 26-neighbourhood, so redistribution and MFTLRemeshSyncGhostIds()
 * take their global paths. Per-axis factors keep the loss to the axes that need
 * it.
 *
 * @param domain The MPI/physical world.
 * @param group  Per-axis coarsening factors, each >= 1; {1,1,1} is bit-for-bit
 *               MFTLRemeshOwnerProc().
 * @param shift  Per-axis cut-plane offset.
 * @param pos    A position, already wrapped into the domain.
 * @return The rank owning @p pos under the coarsened, shifted decomposition;
 *         all ranks of a group map to the same one.
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
 * @brief Reusable scratch for the owner maps of a remeshing pass.
 *
 * A spanning pass migrates up to `parallel_phase_budget` times, each needing one
 * destination per entity; the arena grows geometrically and is reused so later
 * migrations do not pay first-touch page faults again. It belongs to the
 * caller, not to #MFTL_Domain, which is read-only during a redistribution.
 * Both arrays are sized by what this rank holds; `maps` is indexed by local
 * slot, because its only reader walks the same list in the same order.
 */
typedef struct MFTL_RemeshMigrator
{
  MFTL_OwnerMap *maps;  ///< [cap_objects] by LOCAL SLOT, handed straight to MFTLRedistributeWithOwners.
  int *arena;  ///< [cap_entities] all bodies' vertex and triangle destinations, back to back.
  int cap_objects;  ///< LOCAL bodies the `maps` array can describe. Never the id space -- see migrator_reserve().
  int cap_entities;  ///< Total entity slots the arena holds. Grown monotonically.
  /**
   * MFTLRemeshMigrateShifted() hands every owned vertex's `verts.sizing` to its
   * new owner and then the ghosts theirs, so a curvature field computed ONCE per
   * pass survives the pass's migrations (a redistribution does not keep the
   * array). Set by the spanning driver under curvature sizing; 12 bytes per
   * owned vertex of each migration, self-destined records included.
   */
  bool carry_sizing;
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
 * The spanning driver's convergence certificate (3 ints per phase) and the
 * dispersed driver's gates and idle/reassigned counts. The certificate does not
 * ride the per-body id scan, which goes over each body's holders.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their sums over all ranks.
 * @param n      How many; less than 1 is a no-op.
 * @collective
 */
void MFTLRemeshSumInts(const MFTL_Domain *domain, int *values, int n);

/**
 * @brief In-place `MPI_Allreduce(MPI_SUM)` of @p n doubles over the world.
 *
 * Mean edge length (sum and count), frozen-share counts, the balancer's total
 * weight. Always sums, never per-rank means, which would weight ranks rather
 * than entities.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their sums over all ranks.
 * @param n      Number of values.
 * @collective
 */
void MFTLRemeshAgreeDouble(const MFTL_Domain *domain, double *values, int n);

/**
 * @brief Refuses, on EVERY rank, a parameter block that is invalid on any rank or
 *        not identical on all of them.
 *
 * Nothing in a distributed pass communicates the parameters: every rank is
 * assumed to hold the same block. Some fields decide the sequence of
 * collectives (`parallel_strategy`, `parallel_phase_budget`, the coarsening
 * controls), the rest decide what happens to shared entities. Two realistic
 * violations, neither with a symptom of its own:
 *
 * - **A band that is invalid on some ranks only.** The usual cause is
 *   MFTLMeanEdgeLength() (LOCAL) on a distributed mesh: 0 on a rank holding no
 *   triangle of the body. A per-rank refusal returns on that rank and leaves the
 *   others in the next collective, which is a hang in Release.
 * - **Blocks that differ between ranks.** Different collective sequences hang;
 *   different decisions on shared entities give an inconsistent mesh.
 *
 * One `MPI_Allreduce(MIN)` over every field, every negated field and the local
 * band verdict gives each rank the extremes of every field and the worst
 * verdict, so all ranks reach the same answer; a refusal is useful only if it
 * is unanimous. The band is valid when
 * \f$0 \le \ell_{\min} \le \ell_{\max}\f$ and \f$\ell_{\max} > 0\f$
 * (NaN fails). Fields must agree bitwise. Rank 0 prints the reason on stderr.
 *
 * @param domain The MPI/physical world.
 * @param params This rank's parameter block.
 * @param caller Name of the public entry point, for the diagnostic.
 * @return #MFTL_SUCCESS on every rank, or #MFTL_ERROR on every rank.
 * @collective One Allreduce of 49 doubles; at `nproc == 1` only the band test.
 */
int MFTLRemeshAgreeParams(const MFTL_Domain *domain, const MFTL_RemeshParameters *params, const char *caller);

/**
 * @brief In-place `MPI_Allreduce(MPI_MAX)` of @p n ints over the world.
 *
 * The dispersed driver's busiest-rank operation count and the Debug watermark
 * check. Per-body id counters go through MFTLEnsembleReduceBodies() instead.
 *
 * @param domain The MPI/physical world.
 * @param values `[n]` values, replaced by their maxima over all ranks.
 * @param n      Number of values.
 * @collective
 */
void MFTLRemeshAgreeIntMax(const MFTL_Domain *domain, int *values, int n);

/**
 * @brief In-place `MPI_Allreduce(MPI_MAX)` of @p n doubles over the world.
 *
 * The measured band, the balancer's heaviest body and peak load. `MPI_MAX` is
 * order-independent, so the result is bit-identical everywhere and safe to
 * branch on.
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
 * @brief `MPI_Wtime()`, wrapped so that the drivers contain no MPI call
 *        (`test_remeshing_mpi_is_confined` is a source scan).
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

/**
 * @brief Gives every ghost vertex its owner's curvature target length
 *        (`verts.sizing`, #MFTL_REMESH_SIZING_CURVATURE).
 *
 * The sizing is a one-ring quantity, so an owned vertex computes it exactly
 * (the halo is vertex-complete) and a ghost does not. No admitted operation
 * ever reads a ghost's value -- every vertex a split, collapse, flip or smooth
 * of an unfrozen entity reads is owned -- but the convergence certificate does:
 * `count_unoffered_edges` judges edges whose endpoints may be ghosts, and it is
 * exact only if every copy of a vertex holds the same target.
 * Same route as MFTLRemeshSyncGhostIds(), including its global fallback for a
 * coarsened placement; 8 bytes per ghost vertex.
 *
 * @param ens The ensemble; every local body must have `verts.sizing` enabled.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if some ghost could not be served.
 * @collective
 */
int MFTLRemeshSyncGhostSizing(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Neighbourhood exchange -- the three shapes the dispersed balancer needs
 *
 * Each moves a payload proportional to `num_neighbors` (<= 26) and to what THIS
 * rank holds; none can carry an array indexed by body id or by rank.
 *
 * The self-loop: `domain->neighbors` comes from the 26 cell OFFSETS, so on a
 * grid with a single cut on a periodic axis (shapes {2,1,1}, {2,2,1}) two
 * offsets wrap onto this rank and a gather returns this rank's own block. A
 * caller that also folds in its own contribution locally must skip the slot
 * `MFTLDomainNeighborIndex(domain, domain->rank)` reports, or every SUM doubles.
 * ═══════════════════════════════════════════════════════════════════════════ */

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

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Routed records, classification, positions and the projection halo
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Delivers @p n_rec records of @p width doubles each to rank `dest[i]`:
 *        over `comm_graph` when every destination of every rank is itself or a
 *        graph neighbour, over `comm_world` otherwise (agreed by one LOR).
 *        Records addressed to the calling rank never touch the wire.
 * @param      domain The domain.
 * @param      rec    `[n_rec * width]` records.
 * @param      dest   `[n_rec]` destination ranks.
 * @param      n_rec  Records.
 * @param      width  Doubles per record.
 * @param[out] n_out  Records received, own ones first.
 * @return `[*n_out * width]` records to free().
 * @collective
 */
double *MFTLRemeshExchangeRecords(const MFTL_Domain *domain, const double *rec, const int *dest, int n_rec, int width, int *n_out);

/**
 * @brief Gives every ghost vertex its owner's #MFTL_FLAG_FEATURE / #MFTL_FLAG_CORNER
 *        bits: a ghost's fan is partial, so its own classification can be wrong.
 *        Same route as MFTLRemeshSyncGhostIds(); 4 bytes per ghost vertex.
 * @param ens The ensemble, features detected on every rank.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if a request went unserved.
 * @collective
 */
int MFTLRemeshSyncGhostClass(MFTL_Ensemble *ens);

/**
 * @brief Gives every ghost vertex its owner's position, on any partition the
 *        remesher makes -- unlike MFTLUpdateGhostPositions(), with the global
 *        fallback a coarsened placement needs. Used by the NESTED mode's
 *        smoothing sweeps; 24 bytes per ghost vertex.
 * @param ens The ensemble.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if a request went unserved.
 * @collective
 */
int MFTLRemeshSyncGhostPositions(MFTL_Ensemble *ens);

/**
 * @brief The projection reference's halo: every OWNED triangle whose box,
 *        widened by @p width, meets another rank's canonical cell is sent to that
 *        rank as one record (MFTLRemeshReferencePackTriangle()). On the canonical
 *        partition this gives each rank every input triangle within @p width of
 *        its cell, whatever the stencil (aliased periodic images are sent once).
 * @param      ens   The ensemble, canonically partitioned, angle normals exact
 *                   on every copy, ghost classification synchronised.
 * @param      width The halo width (MFTLRemeshProjectionRadius()).
 * @param[out] out   Records received, #MFTL_REMESH_REF_HALO_WIDTH doubles each; free().
 * @param[out] n_out Records received.
 * @return #MFTL_SUCCESS.
 * @collective
 */
int MFTLRemeshReferenceHalo(MFTL_Ensemble *ens, double width, double **out, int *n_out);

#endif  // MFTL_REMESH_COMM_H
