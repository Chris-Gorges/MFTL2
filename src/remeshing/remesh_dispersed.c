/**
 * @file remesh_dispersed.c
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief Distributed remeshing for bodies SMALLER than a rank box: move the
 *        bodies, not the cut planes.
 *
 * ## THE SAME PROBLEM FROM THE OPPOSITE END
 *
 * An edge whose endpoints belong to two ranks cannot be collapsed by either of
 * them. remesh_parallel.c answers that by moving the CUT PLANES until every
 * entity has been in somebody's interior. This file answers it by observing that
 * for a body that FITS INSIDE A RANK BOX the question need never arise: migrate
 * the whole body onto one rank and there are no split edges at all.
 *
 * Both are exact. Which is cheaper is decided entirely by R_body / h:
 *
 *              R >> h                          R << h
 *   move       the planes                      the bodies
 *   phases     4-17 measured (np = 1..8)       1
 *   migrations one per phase, plus one         exactly 2
 *   frozen     O(delta/h) of the mesh          nothing
 *   id cost    one Exscan+Allreduce per phase  one Allreduce(MAX), total
 *   weakness   phase count at large P          load balance at N_bodies < P
 *
 * The dispersed regime is what a real emulsion, spray or bubbly flow is in, and
 * there the difference is not marginal: a pass over ten thousand droplets costs
 * two migrations instead of thirty-four.
 *
 * ## WHY THE IDS NEED ALMOST NOTHING
 *
 * The spanning driver's most delicate machinery is the per-phase MPI_Exscan that
 * makes concurrently-created global ids disjoint. None of it is needed here, and
 * the reason is worth stating because it is easy to add the machinery back out
 * of habit:
 *
 *   after centering, exactly ONE rank holds any entity of a given body,
 *   so that rank's own counter is already globally unique FOR THAT BODY.
 *
 * Two ranks remeshing two different bodies cannot collide, because ids live in a
 * per-body space (`obj->next_vertex_gid` is per object). So the operators run
 * completely untouched, MFTLRemeshRenumberGlobalIds makes the ids dense on the
 * owner, and the only thing the other ranks need is the resulting COUNTS -- one
 * Allreduce(MAX) over 2*num_objects ints, because a rank holding nothing of a
 * body contributes zero and the owner's value wins.
 *
 * Densifying is not cosmetic: redistribution and the ghost exchanges size their
 * arrival tables from `next_*_gid`, so holes accumulated over a few hundred
 * timesteps would become an unboundedly growing table for a mesh of constant
 * size (CLAUDE.md, deferred issue 15).
 *
 * ## ORDERING, AND THE ONE THING THAT MUST NOT HAPPEN
 *
 * The locality check comes BEFORE any remeshing. If some body did not become
 * local, this pass must not have modified anything yet -- the fallback hands the
 * whole ensemble to the spanning driver, and a half-remeshed ensemble entering
 * it would carry per-rank id blocks that no Exscan agreed on. The ordering costs
 * NOTHING: the radii it tests were already reduced for the strategy decision, so
 * the check is a loop over replicated data.
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "remesh_comm.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Strategy selection
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The box width a body's radius is judged against: the rank box, or the
 *        COARSENED box when the fallback is armed and the fine one is too small.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY THE YARDSTICK MOVES AND THE ROUTING DOES NOT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * `r_max > frac * h_min` sends the whole ensemble to the spanning driver. Raise
 * the rank count on a fixed domain and `h_min` shrinks, so a spray of droplets
 * that are ABSOLUTELY tiny becomes RELATIVELY large and is classified SPANNING
 * -- for a reason that has nothing to do with the bodies and everything to do
 * with how many ranks were asked for. The spanning driver is the wrong answer
 * there by roughly two migrations against thirty-four.
 *
 * What the test is actually asking is "can one rank reasonably hold this body",
 * and that question is answered by whatever box the pass is willing to work in.
 * So when `parallel_coarsen` is armed, the yardstick becomes the coarsened box
 * `group[k] * h_k` -- the same relaxation the spanning driver would apply to
 * itself, capped by the same knob.
 *
 * ── The yardstick moves; the DESTINATIONS do not ──────────────────────────
 *
 * Nothing here changes where a body is sent. Routing through coarse group
 * representatives would restrict destinations to `P/G` ranks and undo the load
 * balancer, which needs every rank to be reachable. The coarse box appears
 * ONLY as the size a rank is deemed able to hold, which is a memory statement
 * and not a routing one.
 *
 * ══ THE RELAXATION IS BOUNDED, AND THE BOUND IS NOT GEOMETRIC ══════════════
 *
 * An unbounded version of this is worse than useless, and it is worth saying
 * exactly how because it is not obvious and it is what the tests caught. The
 * factors clamp at `cuts[k]`, so on a 2x1x1 grid the coarse box is the WHOLE
 * DOMAIN -- every body fits, SPANNING becomes unreachable, and a jet spanning
 * the domain gets serialised onto one rank while reporting itself perfectly
 * centred.
 *
 * The real precondition for relaxing is not about the body at all. Dispersed
 * puts one body on one rank, so it uses `min(N_bodies, P)` ranks and no more --
 * and the relaxation is worth having exactly when that is still all of them.
 * With fewer bodies than ranks the strategy is already leaving ranks idle, the
 * relaxation buys no parallelism, and all it does is take a large body that the
 * spanning schedule would have shared and give it to one rank. So:
 *
 *   N_bodies >= nproc   there are enough bodies to keep the machine busy, and
 *                       the fine box has stopped being a statement about the
 *                       bodies and started being one about the rank count
 *   G < nproc           never merge the entire grid into one box. That is the
 *                       gather, it is a spanning-path decision, and it has its
 *                       own parameter with its own memory bound
 *
 * ── It must give the same answer to the chooser and to the driver ─────────
 *
 * Both call this. If they disagreed, the chooser would elect DISPERSED on a body
 * the driver then rejects, and the pass would pay a wasted measurement and fall
 * back to spanning -- correct, slower, and reported only as
 * `n_bodies_not_centred`.
 *
 * @param dom      The MPI/physical world.
 * @param params   May be #MFTL_NULL, in which case no relaxation is applied.
 * @param r_max    The largest body radius anywhere.
 * @param n_bodies Bodies in the ensemble. Below `nproc` no relaxation happens.
 * @return The box width to compare `r_max / frac` against; 0.0 when no axis is
 *         cut, which the callers already treat as "the question is meaningless".
 * @rank_local Pure arithmetic on replicated data, which is what lets the chooser
 *             use it without a second collective.
 */
static double dispersed_box_min(const MFTL_Domain *dom, const MFTL_RemeshParameters *params, double r_max, int n_bodies)
{
  const double h_min = MFTLDomainGetHMin(dom);
  if (h_min <= 0.0) return h_min;

  if (params == MFTL_NULL || params->parallel_coarsen == 0) return h_min;
  if (n_bodies < dom->nproc) return h_min;  // not enough bodies for the relaxation to buy anything

  const double frac = (params->dispersed_radius_fraction > 0.0) ? params->dispersed_radius_fraction : 0.4;
  if (!(r_max > frac * h_min)) return h_min;  // the fine box is already enough

  /* The smallest per-axis merge that would make the body fit, by the same rule
   * and with the same clamp the spanning fallback uses: never more groups than
   * there are cells, because merging every cell on an axis is the most a
   * coarsening can do and is still a legal decomposition. */
  int group[3] = {1, 1, 1};
  for (int k = 0; k < 3; ++k)
  {
    const int cuts = dom->cuts[k];
    if (cuts < 2 || dom->domain_size[k] <= 0.0) continue;

    const double h = dom->domain_size[k] / (double) cuts;
    const double ratio = r_max / (frac * h);
    if (ratio <= 1.0) continue;

    int g = (int) ceil(ratio);
    if (g > cuts) g = cuts;
    group[k] = g;
  }

  MFTLRemeshCapGroups(group, params->parallel_coarsen);

  /* Merging the whole grid into one box is the GATHER, and the gather is not
   * this function's decision to take: it belongs to the spanning driver, which
   * bounds it by a triangle count because one rank holding everything is a
   * memory question. Refusing here leaves such an ensemble on the spanning
   * path, where `serial_gather_max_triangles` can answer it properly. */
  if (group[0] * group[1] * group[2] >= dom->nproc) return h_min;

  double box = MFTL_DBL_MAX;
  for (int k = 0; k < 3; ++k)
  {
    const int cuts = dom->cuts[k];
    if (cuts < 2 || dom->domain_size[k] <= 0.0) continue;
    box = MFTL_MIN(box, (double) group[k] * dom->domain_size[k] / (double) cuts);
  }

  return (box == MFTL_DBL_MAX) ? 0.0 : box;
}

void MFTLRemeshMeasurementInit(MFTL_RemeshMeasurement *m, const MFTL_Ensemble *ens)
{
  const int b = ens->n_local_objects;
  m->n_local = b;
  m->r_max = 0.0;
  /* One block: a pass takes one measurement, so one malloc. */
  m->centroid = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(6 * b, 1), sizeof(double), "remesh body measurement");
  m->radius = m->centroid + 3 * b;
  m->weight = m->radius + b;
  m->lowest_holder = m->weight + b;
}

void MFTLRemeshMeasurementFree(MFTL_RemeshMeasurement *m)
{
  free(m->centroid);
  m->centroid = MFTL_NULL;
  m->radius = MFTL_NULL;
  m->weight = MFTL_NULL;
  m->lowest_holder = MFTL_NULL;
  m->n_local = 0;
  m->r_max = 0.0;
}

MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategyMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshMeasurement *m)
{
  MFTL_ASSERT(ens != MFTL_NULL && m != MFTL_NULL);

  /* At one rank nothing is split and both strategies degenerate to the serial
   * kernel. Answering SPANNING keeps MFTLRemeshEnsemble's nproc == 1 path the
   * only special case there is. */
  /* GLOBAL body count: MFTLRemeshMeasureBodies() below is collective, and both
   * this function's callers dispatch on its answer -- two ranks returning
   * different strategies is an immediate deadlock. */
  if (ens->domain.nproc < 2 || ens->n_bodies_global == 0) return MFTL_REMESH_PARALLEL_SPANNING;

  const double frac = (params && params->dispersed_radius_fraction > 0.0) ? params->dispersed_radius_fraction : 0.4;

  /* The measurement is taken even when h_min says the answer is already SPANNING,
   * because the caller may be MFTLRemeshEnsemble, which owns the buffers and
   * would otherwise have to know whether they were filled. Filling them
   * unconditionally is one collective on a path that is about to run a dozen. */
  MFTLEnsembleComputeTriangleGeometry(ens);
  MFTLRemeshMeasureBodies(ens, m->centroid, m->radius, m->weight, &m->r_max, m->lowest_holder);

  if (MFTLDomainGetHMin(&ens->domain) <= 0.0) return MFTL_REMESH_PARALLEL_SPANNING;

  /* ONE SCALAR, and it has to be one. `radius` describes this rank's bodies and
   * nothing else, so a maximum over it is a different number on every rank; the
   * dispatch below would then split the ranks between two drivers, which is an
   * immediate deadlock with no diagnostic. `r_max` came out of a scalar `MPI_MAX`
   * inside the measurement and is identical to the bit everywhere.
   *
   * The replicated `[n_bodies_global]` radius array this replaces was, for this
   * consumer, an expensive way of computing one number. */
  const double box = dispersed_box_min(&ens->domain, params, m->r_max, ens->n_bodies_global);
  return (m->r_max > frac * box) ? MFTL_REMESH_PARALLEL_SPANNING : MFTL_REMESH_PARALLEL_DISPERSED;
}

MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategy(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens->domain.nproc < 2 || ens->n_bodies_global == 0) return MFTL_REMESH_PARALLEL_SPANNING;

  MFTL_RemeshMeasurement m;
  MFTLRemeshMeasurementInit(&m, ens);
  const MFTL_RemeshParallelStrategy strategy = MFTLRemeshChooseStrategyMeasured(ens, params, &m);
  MFTLRemeshMeasurementFree(&m);
  return strategy;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1b. Load balancing
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ══ NOBODY REPLICATES ANYTHING; ONE RANK DECIDES AND TELLS THE HOLDERS ══════
 *
 * MFTL_OwnerMap is undefined -- not slow, undefined -- if two ranks name
 * different destinations for the same entity, so the destination of a body has
 * to be AGREED. There are exactly two ways to agree, and which one is chosen is
 * the whole scalability question for this driver:
 *
 *   REPLICATE   give every rank every body's centroid and weight, and have all
 *               of them run the same deterministic greedy. That is what this
 *               file did until now, and it is `O(bodies in the ensemble)` in
 *               payload AND in per-rank memory: 32 bytes of staging per body on
 *               every rank, once per remesh, 32 MB each at a million droplets.
 *
 *   DECIDE ONCE one rank decides and SENDS the verdict to the ranks that need
 *               it. `O(local bodies x degree)`, and the degree is at most 26
 *               whatever `nproc` is.
 *
 * The second is what is below. It was the last `O(bodies in the ensemble)` term
 * in the library (docs/analysis.md §6.4).
 *
 * ── THE ONE PRECONDITION, AND IT IS A LOCAL TEST RATHER THAN AN ARGUMENT ───
 *
 * "Decide once and send it" needs the decider to be able to REACH every holder,
 * and the halo only ever ships one hop along `comm_graph`. Let
 * `home(o) = MFTLGetOwnerProc(centroid(o))` -- the rank that would hold body `o`
 * without any balancing, agreed by every holder to the bit because the centroid
 * came out of a reduction. The precondition is
 *
 *     for every body o and every holder H of o:  home(o) is H or a neighbour of H
 *
 * and the elegant part is that a holder can test its own half of it with no
 * communication at all, so an `MPI_LAND` over the job establishes the whole
 * statement. Two consequences fall straight out of it and they are exactly what
 * the scheme needs:
 *
 *   home(o) CAN BE TOLD ABOUT o. The graph is symmetric, so `lowest_holder(o)`
 *   -- a genuine holder, already reduced by MFTLRemeshMeasureBodies, and the
 *   same rank in everybody's opinion -- is a neighbour of home(o) and its
 *   `(gid, home, weight)` record arrives there.
 *
 *   home(o) CAN TELL EVERY HOLDER. Each holder is a neighbour of home(o), so one
 *   neighbourhood gather of `(gid, dest, weight)` reaches all of them.
 *
 * Note what is NOT required: the holder set does not have to be a clique, no
 * holder has to own an entity (an empty holder still READS the verdict, it just
 * cannot be discovered -- and nothing here needs to discover it), and home(o)
 * does not have to hold the body. `MFTLEnsembleBodyHoldersFitTheGraph()` is a
 * strictly stronger statement and is deliberately not what is tested here.
 *
 * When the precondition fails anywhere the pass reports it and falls back to
 * PURE CENTROID OWNERSHIP -- correct, unbalanced, and visible as
 * `MFTL_RemeshStats::n_bodies_unreachable`. It does not fall back to
 * replicating, because a fallback that is `O(B)` is an `O(B)` term.
 *
 * ── WHY IT IS A NEGOTIATION AND NOT A GREEDY ──────────────────────────────
 *
 * The replicated greedy could see every rank's load, so it could hand a body to
 * the least-loaded rank in the job and know that nobody else was doing the same.
 * A local rule cannot: every overloaded rank sees the same idle neighbour and,
 * left alone, they all dump on it. The fix is two directed exchanges of one
 * double each -- a rank asks its neighbours for room in proportion to the room
 * they advertise, and each neighbour divides the room IT ACTUALLY HAS among the
 * askers in proportion to what they asked:
 *
 *     spare_r  = MAX(0, cap - load_r)                       published to all
 *     dem_sr   = excess_s * spare_r / SUM_r spare_r         s asks r
 *     grant_sr = spare_r * dem_sr / SUM_s dem_sr            r answers s
 *
 * `SUM_s grant_sr == spare_r` exactly, so NO RANK CAN BE PUSHED OVER ITS CAP no
 * matter how many neighbours ship at once -- that is the property a greedy over
 * stale loads does not have. An uncontended sender is granted the whole spare
 * rather than the fraction it asked for, which matters because bodies are
 * indivisible and a grant equal to the request rounds down to fewer bodies than
 * it should.
 *
 * ── WHAT THE ONE HOP COSTS, STATED RATHER THAN HIDDEN ─────────────────────
 *
 * Weight moves at most one cell per pass. A pile-up confined to one rank box is
 * fully fixed whenever the grid is small enough that the 26-neighbourhood is the
 * whole machine (every shape up to 2 cells per axis, i.e. np <= 8) and, at
 * larger `P`, whenever the neighbours have room -- which is the case a dispersed
 * ensemble is actually in, because droplet clustering is a local phenomenon. A
 * pile-up SPANNING many cells is not fully fixed, and the replicated greedy
 * would not have fixed it either: its own locality-first rule tries the 26 cells
 * before anywhere else, and its global fallback is precisely the case where a
 * body leaves the stencil and redistribution drops to its all-to-all path.
 *
 * ── DETERMINISM IS NOW A WEAKER REQUIREMENT, WHICH IS THE OTHER WIN ───────
 *
 * Every body is decided by exactly ONE rank, so two ranks can no longer disagree
 * about it -- the failure mode the replicated version had to defend against with
 * a total ordering on every tie-break, a ban on reading `dom->neighbors`, and a
 * comment explaining that qsort is not stable. Run-to-run reproducibility is
 * still wanted, and the tie-breaks below still deliver it, but a mistake in one
 * is now a different assignment rather than a corrupt mesh.
 */

/**
 * @brief Doubles this rank has staged for the balancer's exchanges.
 *
 * The only observable the scaling property has -- see MFTLRemeshBalancerPayload()
 * in mftl_remeshing.h. Incremented at every point a buffer is handed to
 * remesh_comm.c, so the number is what goes on the wire and not an estimate of
 * it.
 */
static long g_balancer_payload = 0;

long MFTLRemeshBalancerPayload(void) { return g_balancer_payload; }

void MFTLRemeshResetBalancerPayload(void) { g_balancer_payload = 0; }

/** @brief One body's weight and id at the rank that decides it. */
typedef struct
{
  double w;  ///< Weight: the body's global owned-triangle count, floored at one.
  int gid;  ///< Body global id, and the tie-break that makes the order total.
} homed_body;

/**
 * @brief Orders bodies heaviest first, ties by ascending global id.
 *
 * Heaviest first because a large body placed late finds only fragments of the
 * grants left; the id tie-break makes the order TOTAL, so the result does not
 * depend on qsort's (unspecified) stability and one rank's assignment repeats
 * exactly from run to run.
 *
 * @param a First element.
 * @param b Second element.
 * @return Negative, zero or positive in the usual qsort sense.
 */
static int by_weight_desc(const void *a, const void *b)
{
  const homed_body *const x = (const homed_body *) a;
  const homed_body *const y = (const homed_body *) b;

  if (x->w > y->w) return -1;
  if (x->w < y->w) return 1;
  return (x->gid < y->gid) ? -1 : ((x->gid > y->gid) ? 1 : 0);
}

/** @brief A `(gid, dest, weight)` verdict, sorted so a holder can binary-search it. */
typedef struct
{
  double w;  ///< The body's balancing weight, so the receiver can price its inflow.
  int gid;  ///< Body global id.
  int dest;  ///< The rank home(gid) chose.
} verdict;

/**
 * @brief Orders verdicts by ascending body id.
 *
 * @param a First element.
 * @param b Second element.
 * @return Negative, zero or positive in the usual qsort sense.
 */
static int by_gid_asc(const void *a, const void *b)
{
  const verdict *const x = (const verdict *) a;
  const verdict *const y = (const verdict *) b;
  return (x->gid < y->gid) ? -1 : ((x->gid > y->gid) ? 1 : 0);
}

/**
 * @brief A body's weight as the balancer counts it: the global triangle count,
 *        or one unit for a body that has no triangles anywhere.
 *
 * Zero is what a body with no triangles reports, and giving it zero weight would
 * let a rank accumulate any number of them while reporting a perfect balance. It
 * still occupies a rank, so it counts as one.
 *
 * @param w The reduced weight from MFTLRemeshMeasureBodies().
 * @return The weight to balance on, always positive.
 */
static inline double balance_weight(double w) { return (w > 0.0) ? w : 1.0; }

/**
 * @brief How many ask/grant/ship rounds one balancing pass runs.
 *
 * BOUNDED RATHER THAN CONVERGENT, and the difference is the safety argument:
 * running out of rounds leaves an imbalance that the pass reports in
 * `load_imbalance`, never a wrong mesh. There is no fixed point to detect and no
 * collective spent detecting one.
 *
 * Measured on `examples/remesh_regime_sweep -only l`, the clustered and the
 * spread fixture, at np = 4, 8 and 27 -- the resulting imbalance against a slack
 * of 1.25:
 *
 *   rounds   np=4   np=8   np=27 spread   np=27 clustered
 *   1        1.25   1.25   1.50           1.25
 *   2        1.25   1.25   1.25           1.25
 *   3        1.25   1.25   1.25           1.25
 *   4        1.25   1.25   1.25           1.25
 *
 * A lone pile-up settles in ONE round because its ask is uncontended; it is the
 * SPREAD imbalance over a full 26-neighbourhood that needs a second, because
 * every overloaded rank asks the same emptiest neighbours first and only the
 * served ones drop out of the next round's advertisement -- see the ask step.
 *
 * Two is what the measurement needs and three is what ships, because the two
 * errors are not symmetric: one round too few is a silently unbalanced pass --
 * the failure this loop exists to fix -- and one too many is three directed
 * exchanges of one or two doubles per neighbour, which is latency and not
 * volume.
 */
enum
{
  MFTL_DISPERSED_BALANCE_ROUNDS = 3  ///< Ask/grant/ship rounds per balancing pass.
};

/** @brief Everything one balancing pass needs to know about the neighbourhood. */
typedef struct
{
  int d;  ///< `domain->num_neighbors`, at most 26.
  int self;  ///< Slot of the self-loop in `domain->neighbors`, or #MFTL_ERROR.
} nbhood;

/**
 * @brief Resolves rank @p r to a slot in `domain->neighbors`, or "that is me".
 *
 * @param dom The MPI/physical world.
 * @param r   A rank.
 * @return `num_neighbors` when @p r IS this rank, its slot when it is a graph
 *         neighbour, and #MFTL_ERROR otherwise. A linear scan over at most 26
 *         entries (MFTLDomainNeighborIndex()), which is why nothing here
 *         allocates per rank.
 */
static int neighbor_slot(const MFTL_Domain *dom, int r) { return (r == dom->rank) ? dom->num_neighbors : MFTLDomainNeighborIndex(dom, r); }

/**
 * @brief Is neighbour slot @p a preferred to slot @p b at equal usefulness?
 *
 * ══ A TIE-BREAK THAT IS THE SAME ON EVERY RANK STARVES THE SAME RANK ═══════
 *
 * The obvious tie-break -- lowest rank id -- is deterministic, total, and wrong
 * here, because it is the SAME ORDERING ON EVERY SENDER. Four overloaded ranks
 * looking at four equally empty neighbours all pick the same one, then the same
 * second one, and the fourth is never chosen by anybody: measured as 1 of 8
 * ranks left idle on a spread fixture that has thirty-two bodies for eight
 * ranks. Nothing is wrong with the mesh and the imbalance still reports 1.25 --
 * this is a tie-break, not a correctness rule -- but a rank was left with
 * nothing to do for no reason.
 *
 * The fix has to make different senders prefer different targets, and the
 * MEANING is already available: the cell OFFSET of a neighbour is relative to
 * the rank doing the looking, so a nearer destination is both a different choice
 * on every rank and a genuinely cheaper migration -- redistribution's stencil is
 * a box, and a body sent one cell stays inside more senders' halos than one sent
 * across a corner. Manhattan distance first, rank id second so the order stays
 * total.
 *
 * @param dom The MPI/physical world.
 * @param a   A neighbour slot.
 * @param b   A neighbour slot.
 * @return #MFTL_TRUE when @p a should be tried before @p b.
 * @rank_local
 */
static bool nearer_neighbour(const MFTL_Domain *dom, int a, int b)
{
  int da = 0, db = 0;
  for (int k = 0; k < 3; ++k)
  {
    da += (dom->neighbor_offset[a][k] < 0) ? -dom->neighbor_offset[a][k] : dom->neighbor_offset[a][k];
    db += (dom->neighbor_offset[b][k] < 0) ? -dom->neighbor_offset[b][k] : dom->neighbor_offset[b][k];
  }
  if (da != db) return da < db;
  return dom->neighbors[a] < dom->neighbors[b];
}

/**
 * @brief Orders the neighbour slots by DESCENDING @p key, ties by nearness.
 *
 * An insertion sort over at most 26 entries, which is the whole point: the
 * balancer's per-rank work must not grow with `nproc`, and an ordering of the
 * neighbourhood is the largest thing it is allowed to sort. Slot order itself is
 * never used as a tie-break -- it is whatever MPI reported for this rank, and
 * ordering on it would make the assignment depend on that.
 *
 * @param      dom The MPI/physical world.
 * @param      key `[d]` the quantity to order by, per neighbour slot.
 * @param      d   Neighbours, at most 26.
 * @param[out] out `[d]` slot indices, largest key first.
 * @return How many were written, i.e. @p d.
 * @rank_local
 */
static int order_neighbours(const MFTL_Domain *dom, const double *key, int d, int out[26])
{
  for (int i = 0; i < d; ++i)
  {
    int j = i - 1;
    const int v = i;
    while (j >= 0 && (key[out[j]] < key[v] || (key[out[j]] == key[v] && nearer_neighbour(dom, v, out[j]))))
    {
      out[j + 1] = out[j];
      j--;
    }
    out[j + 1] = v;
  }
  return d;
}

/**
 * @brief Every body this rank is HOME to, gathered from its lowest holders.
 *
 * One neighbourhood gather of three doubles per body this rank SPEAKS FOR --
 * `lowest_holder[o] == rank`, so each body is published exactly once in the job
 * however many ranks hold it.
 *
 * @param      ens   The ensemble.
 * @param      m     The local measurement; row `o` describes local slot `o`.
 * @param      nh    The resolved neighbourhood.
 * @param[out] n_out Bodies homed here.
 * @return A newly allocated `[*n_out]` array, sorted heaviest-first. The caller
 *         frees it. Never #MFTL_NULL.
 * @collective One MFTLRemeshNeighborGather().
 */
static homed_body *gather_homed_bodies(MFTL_Ensemble *ens, const MFTL_RemeshMeasurement *m, const nbhood *nh, int *n_out)
{
  const MFTL_Domain *const dom = &ens->domain;

  /* ── What this rank publishes ────────────────────────────────────────────
   *
   * `(gid, home, weight)`. The gid is what a holder matches the verdict against,
   * `home` is the address, and the weight is what home balances on -- it cannot
   * be recomputed there, because home need hold no entity of the body. */
  int n_send = 0;
  double *const send = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(m->n_local, 1) * sizeof(double), "dispersed home publish");

  for (int o = 0; o < m->n_local; ++o)
  {
    if ((int) m->lowest_holder[o] != dom->rank) continue;
    send[n_send + 0] = (double) ens->local_objects[o]->global_id;
    send[n_send + 1] = (double) MFTLGetOwnerProc(dom, &m->centroid[3 * o]);
    send[n_send + 2] = balance_weight(m->weight[o]);
    n_send += 3;
  }

  int *const rcount = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nh->d, 1) * sizeof(int), "dispersed home counts");
  int *const rdispl = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nh->d, 1) * sizeof(int), "dispersed home displacements");

  g_balancer_payload += n_send;

  int n_recv = 0;
  double *const recv = MFTLRemeshNeighborGather(dom, send, n_send, rcount, rdispl, &n_recv);

  /* Capacity: everything this rank published plus everything it received could
   * in principle be homed here. Bounded by `local + degree * their locals`, i.e.
   * by what the neighbourhood holds and never by the ensemble. */
  const int cap = n_send / 3 + n_recv / 3;
  homed_body *const mine = (homed_body *) MFTLMallocOrDie((size_t) MFTL_MAX(cap, 1) * sizeof(homed_body), "dispersed homed bodies");
  int n = 0;

  for (int i = 0; i < n_send; i += 3)
  {
    if ((int) send[i + 1] != dom->rank) continue;
    mine[n].gid = (int) send[i];
    mine[n].w = send[i + 2];
    n++;
  }

  for (int i = 0; i < nh->d; ++i)
  {
    /* THE SELF-LOOP. On a grid with one cut on a periodic axis this rank is its
     * own graph neighbour, so its block comes back to it -- and it was already
     * folded in above. Counting it twice doubles the load of every body homed
     * here, which no MIN or MAX would notice and every SUM would carry. */
    if (i == nh->self) continue;

    for (int j = rdispl[i]; j + 2 < rdispl[i] + rcount[i]; j += 3)
    {
      if ((int) recv[j + 1] != dom->rank) continue;
      mine[n].gid = (int) recv[j];
      mine[n].w = recv[j + 2];
      n++;
    }
  }

  free(recv);
  free(rdispl);
  free(rcount);
  free(send);

  qsort(mine, (size_t) n, sizeof(homed_body), by_weight_desc);

  *n_out = n;
  return mine;
}

/**
 * @brief Chooses a destination rank for every body this rank holds.
 *
 * Six steps, and every payload is bounded by the neighbourhood: publish who is
 * home to what, agree two scalars, advertise room, ask for room, grant room,
 * publish the verdicts. See the section banner for the argument.
 *
 * @param      ens     The ensemble.
 * @param      params  Supplies `dispersed_balance_slack`; `<= 0` reports the
 *                     balance without changing anything.
 * @param      m       The local measurement, describing the CURRENT local list.
 * @param      veto    Non-zero anywhere in the job means the reachability
 *                     precondition failed; balancing is skipped.
 * @param[out] dest    `[m->n_local]` destination rank per LOCAL SLOT.
 * @param[out] st      Receives `n_bodies_reassigned`, `n_ranks_idle` and
 *                     `load_imbalance`.
 * @collective Two neighbourhood gathers (the home publish and the verdict
 *             publish), one neighbourhood allgather of a single double, three
 *             directed exchanges per balancing round, and four scalar
 *             `comm_world` reductions. Nothing is proportional to
 *             `n_bodies_global` or to `nproc`.
 */
static void assign_destinations(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, const MFTL_RemeshMeasurement *m, int veto, int *dest,
                                MFTL_RemeshStats *st)
{
  const MFTL_Domain *const dom = &ens->domain;

  nbhood nh;
  nh.d = dom->num_neighbors;
  nh.self = MFTLDomainNeighborIndex(dom, dom->rank);

  /* The unbalanced answer, and the one every body keeps unless its home rank
   * says otherwise. Every holder computes it from its own reduced centroid, so
   * they agree without being told. */
  for (int o = 0; o < m->n_local; ++o) dest[o] = MFTLGetOwnerProc(dom, &m->centroid[3 * o]);

  /* ── 1. Who is home to what ──────────────────────────────────────────── */
  int n_mine = 0;
  homed_body *const mine = gather_homed_bodies(ens, m, &nh, &n_mine);

  double load = 0.0;
  double heaviest = 0.0;
  for (int i = 0; i < n_mine; ++i)
  {
    load += mine[i].w;
    if (mine[i].w > heaviest) heaviest = mine[i].w;
  }

  /* ── 2. The two scalars a cap needs ──────────────────────────────────────
   *
   * `total` for the mean and `heaviest` for the floor, and both are ONE double.
   * The `[nproc]` load table this replaces was the other thing in this file that
   * did not scale: correct, cheap at 27 ranks, and 800 kB per rank per remesh at
   * a hundred thousand. */
  double total = load;
  MFTLRemeshAgreeDouble(dom, &total, 1);
  MFTLRemeshAgreeDoubleMax(dom, &heaviest, 1);

  const double slack = params->dispersed_balance_slack;
  /* Every term is REPLICATED -- the slack is a checked parameter, the rank count
   * is the world and the veto came out of a reduction -- so every rank enters
   * and leaves the collectives below together. `nh.d` deliberately does NOT
   * appear: it is per rank, and gating a neighbourhood collective on it is a
   * deadlock on any grid where one rank is isolated. (None is, above one rank:
   * `nproc > 1` puts at least two cells on some axis and the +/-1 offset along
   * it is in range, so the degree is at least one.) */
  const bool balancing = (slack > 0.0 && dom->nproc > 1 && veto == 0);

  /* `MAX(mean * slack, heaviest)` -- the second term is not defensive padding. A
   * body is indivisible here, so a cap below the heaviest body asks for an
   * assignment that does not exist and every rank holding that body would try,
   * and fail, to shed it every pass. */
  const double cap = MFTL_MAX(total / (double) dom->nproc * slack, heaviest);

  int reassigned = 0;
  double shipped_out = 0.0;
  double received_in = 0.0;

  /* FIVE ARRAYS, SEVEN SLOTS OF LENGTH `d`, AND NOTHING ELSE. Every buffer the
   * negotiation touches is sized by the graph degree -- at most 26, whatever
   * `nproc` is -- which is the whole difference from the `[nproc]` load table and
   * the `4 * n_bodies_global` staging this replaced. One allocation rather than
   * five because they have the same lifetime and none of them is large. */
  double *const nb_buf = (double *) MFTLCallocOrDie(7 * (size_t) MFTL_MAX(nh.d, 1), sizeof(double), "dispersed neighbourhood scalars");
  double *const spare_nb = nb_buf;  // [d] room each neighbour last advertised
  double *const demand = nb_buf + nh.d;  // [d] what this rank asks of each
  double *const dem_in = nb_buf + 2 * nh.d;  // [d] what each neighbour asks of this rank
  double *const reply = nb_buf + 3 * nh.d;  // [2d] (granted, room still unpromised) to each asker
  double *const got = nb_buf + 5 * nh.d;  // [2d] the same, received

  verdict *moved = MFTL_NULL;
  bool *placed = MFTL_NULL;

  if (balancing)
  {
    moved = (verdict *) MFTLMallocOrDie((size_t) MFTL_MAX(n_mine, 1) * sizeof(verdict), "dispersed verdicts");
    placed = (bool *) MFTLCallocOrDie((size_t) MFTL_MAX(n_mine, 1), sizeof(bool), "dispersed placed flags");

    /* ── 3. Advertise room ────────────────────────────────────────────────
     *
     * Room and not load, because what a sender must respect is the TARGET's cap
     * and the cap is the target's own arithmetic. Once, before the rounds; from
     * then on each grant carries the granter's remaining room back with it, so
     * no further advertisement is needed. */
    double left = MFTL_MAX(cap - load, 0.0);
    /* The whole fixed cost of the negotiation, counted once so that
     * MFTLRemeshBalancerPayload() is the honest total and not just its
     * body-proportional half: one advertisement, then four directed doubles per
     * neighbour per round (ask, two of grant-and-room, ship). */
    g_balancer_payload += 1 + 4 * nh.d * MFTL_DISPERSED_BALANCE_ROUNDS;
    MFTLRemeshNeighborAllgather(dom, &left, 1, spare_nb);
    if (nh.self != MFTL_ERROR) spare_nb[nh.self] = 0.0;  // shipping to oneself moves nothing

    double cur = load;

    for (int round = 0; round < MFTL_DISPERSED_BALANCE_ROUNDS; ++round)
    {
      /* ── 4. Ask FEW NEIGHBOURS FOR A LOT, and ask again next round ───────
       *
       * ══ THE THING A BODY IS THAT A LOAD IS NOT: INDIVISIBLE ════════════
       *
       * The obvious rule -- split the excess over the neighbours in proportion
       * to the room each advertises -- is right for a fluid and wrong here, and
       * the way it fails PASSES EVERY TEST OF THE ANSWER. On a 3x3x3 grid a rank
       * asks each of 26 neighbours for a twenty-sixth of its excess; each of
       * those, contended by several askers, replies with a slice of that; and
       * the grant that comes back is SMALLER THAN ONE BODY. Nothing is
       * placeable, nothing moves, and the pass reports an unchanged imbalance
       * with `n_bodies_reassigned == 0` -- which is exactly what an ensemble
       * that needed no balancing reports. Measured before this was fixed: 108
       * bodies over 27 ranks stayed at an imbalance of 2.00, while the same code
       * balanced a single clustered rank -- whose ask was uncontended -- to 1.25.
       *
       * So the ask is CONCENTRATED: walk the neighbours emptiest first and ask
       * each for everything it advertises until the budget is covered.
       *
       * ── And that is why there are ROUNDS ────────────────────────────────
       *
       * Concentrating creates the opposite failure: every overloaded rank
       * computes the same ordering over the same advertisements and they all ask
       * the SAME emptiest neighbour, which can serve one of them. One round of
       * this balances a lone pile-up perfectly and a spread-out imbalance barely
       * at all. A round is cheap -- three directed exchanges of one or two
       * doubles per neighbour -- and after each one the served target advertises
       * less, so the unserved askers move on. Four is enough for every fixture
       * measured and the loop is bounded rather than convergent by argument: a
       * pass that runs out of rounds leaves an imbalance, never a wrong mesh.
       *
       * ── The budget overshoots by one body, deliberately ─────────────────
       *
       * `cur - cap` is what has to leave, but only whole bodies can leave, so
       * asking for exactly that is routinely one body short of reaching the cap.
       * The heaviest body still unplaced is added, which is the smallest
       * addition that always suffices. It cannot push a TARGET over its cap: a
       * grant is bounded by the target's own room whatever was asked. */
      for (int i = 0; i < nh.d; ++i) demand[i] = 0.0;

      if (cur > cap)
      {
        double biggest = 0.0;
        for (int i = 0; i < n_mine; ++i)
        {
          if (!placed[i])
          {
            biggest = mine[i].w;  // sorted heaviest first, so the first unplaced is the largest
            break;
          }
        }

        int ord[26];
        order_neighbours(dom, spare_nb, nh.d, ord);

        int n_open = 0;
        for (int i = 0; i < nh.d; ++i)
        {
          if (spare_nb[i] > 0.0) n_open++;
        }

        const double budget = (cur - cap) + biggest;

        /* AS MANY NEIGHBOURS AS THE BUDGET CAN GIVE A WHOLE BODY EACH, and no
         * more. Fewer and ranks that could have been given work stay idle;
         * more and every chunk is a fraction of a body and nothing is placeable.
         * `budget / biggest` is exactly that count, clamped to the neighbours
         * that have room and to one. */
        int k = (biggest > 0.0) ? (int) (budget / biggest) : n_open;
        if (k > n_open) k = n_open;
        if (k < 1) k = 1;

        const double share = budget / (double) k;
        for (int j = 0; j < k; ++j)
        {
          const int i = ord[j];
          if (spare_nb[i] <= 0.0) break;  // ordered by descending room, so the rest are full too
          demand[i] = MFTL_MIN(spare_nb[i], share);
        }
      }

      MFTLRemeshNeighborAlltoall(dom, demand, 1, dem_in);
      if (nh.self != MFTL_ERROR) dem_in[nh.self] = 0.0;

      /* ── 5. Grant, neediest first, and never more room than exists ───────
       *
       * `SUM_s granted_sr <= room_r` by construction, so NO RANK CAN BE PUSHED
       * PAST ITS CAP however many neighbours ship at once -- the property a
       * greedy over stale loads does not have, and the reason this is a
       * negotiation rather than a race. Serving in priority order rather than
       * proportionally is the granting half of the indivisibility argument
       * above: a room divided among all askers is a room divided into pieces
       * none of them can use.
       *
       * The second field is what the next round is ordered by, which is why no
       * second advertisement is needed. It is the room left after this round's
       * PROMISES, so it under-states whenever a grant goes unused -- and `left`
       * itself is corrected below from what was actually shipped, so the
       * under-statement costs at most one round of a smaller ask. */
      {
        int ord[26];
        order_neighbours(dom, dem_in, nh.d, ord);

        double unpromised = left;
        for (int j = 0; j < nh.d; ++j)
        {
          const int i = ord[j];
          const double give = MFTL_MIN(dem_in[i], unpromised);
          reply[2 * i] = (give > 0.0) ? give : 0.0;
          unpromised -= reply[2 * i];
        }
        for (int i = 0; i < nh.d; ++i) reply[2 * i + 1] = unpromised;
      }

      MFTLRemeshNeighborAlltoall(dom, reply, 2, got);
      if (nh.self != MFTL_ERROR)
      {
        got[2 * nh.self] = 0.0;
        got[2 * nh.self + 1] = 0.0;
      }

      /* ── 6. Place the bodies, heaviest first ─────────────────────────────
       *
       * Into the neighbour with the most room left, which keeps the largest body
       * still to come placeable, and among equals into the NEAREST -- see
       * nearer_neighbour() for why a tie-break that is identical on every sender
       * leaves a rank idle. */
      for (int i = 0; i < nh.d; ++i) demand[i] = 0.0;  // reused: weight shipped to each neighbour

      for (int i = 0; i < n_mine && cur > cap; ++i)
      {
        if (placed[i]) continue;
        const double w = mine[i].w;

        int best = MFTL_ERROR;
        for (int j = 0; j < nh.d; ++j)
        {
          if (got[2 * j] < w) continue;
          if (best == MFTL_ERROR || got[2 * j] > got[2 * best] || (got[2 * j] == got[2 * best] && nearer_neighbour(dom, j, best))) best = j;
        }
        if (best == MFTL_ERROR) continue;  // no neighbour granted room for one this large

        moved[reassigned].gid = mine[i].gid;
        moved[reassigned].dest = dom->neighbors[best];
        moved[reassigned].w = w;
        reassigned++;
        placed[i] = MFTL_TRUE;

        got[2 * best] -= w;
        demand[best] += w;
        cur -= w;
        shipped_out += w;
      }

      /* ── 7. Say what was actually shipped ────────────────────────────────
       *
       * A promise is not a delivery: a grant of two bodies' worth that fits only
       * one leaves room the granter must get back, or the rounds would leak
       * capacity and the imbalance report would be wrong about the inflow. One
       * more directed double closes both. */
      MFTLRemeshNeighborAlltoall(dom, demand, 1, dem_in);
      if (nh.self != MFTL_ERROR) dem_in[nh.self] = 0.0;

      double inflow = 0.0;
      for (int i = 0; i < nh.d; ++i) inflow += dem_in[i];
      received_in += inflow;
      left = MFTL_MAX(left - inflow, 0.0);

      for (int i = 0; i < nh.d; ++i) spare_nb[i] = got[2 * i + 1];
      if (nh.self != MFTL_ERROR) spare_nb[nh.self] = 0.0;
    }
  }

  /* ── 7. Publish the verdicts, and read everyone else's ───────────────────
   *
   * `(gid, dest, weight)`, three doubles per body that actually moved -- which
   * is a small fraction of a balanced ensemble and exactly zero when the pass
   * changed nothing. Only the HOLDERS need it and they are all one hop from
   * home, which is the precondition this driver checked before it started. */
  if (balancing)
  {
    double *const send = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(reassigned, 1) * sizeof(double), "dispersed verdict publish");
    for (int i = 0; i < reassigned; ++i)
    {
      send[3 * i + 0] = (double) moved[i].gid;
      send[3 * i + 1] = (double) moved[i].dest;
      send[3 * i + 2] = moved[i].w;
    }

    int *const rcount = (int *) MFTLMallocOrDie((size_t) nh.d * sizeof(int), "dispersed verdict counts");
    int *const rdispl = (int *) MFTLMallocOrDie((size_t) nh.d * sizeof(int), "dispersed verdict displacements");

    g_balancer_payload += 3 * reassigned;

    int n_recv = 0;
    double *const recv = MFTLRemeshNeighborGather(dom, send, 3 * reassigned, rcount, rdispl, &n_recv);

    /* Everything this rank must apply: its own verdicts plus its neighbours'. A
     * body appears at most once -- only home publishes it -- so the sorted list
     * is a strictly increasing key and a binary search per local body resolves
     * it in `O(local log moved)`. */
    const int cap_all = reassigned + n_recv / 3;
    verdict *const all = (verdict *) MFTLMallocOrDie((size_t) MFTL_MAX(cap_all, 1) * sizeof(verdict), "dispersed verdict table");
    int n_all = 0;

    for (int i = 0; i < reassigned; ++i) all[n_all++] = moved[i];

    for (int i = 0; i < nh.d; ++i)
    {
      if (i == nh.self) continue;  // own verdicts, already in
      for (int j = rdispl[i]; j + 2 < rdispl[i] + rcount[i]; j += 3)
      {
        all[n_all].gid = (int) recv[j];
        all[n_all].dest = (int) recv[j + 1];
        all[n_all].w = recv[j + 2];
        n_all++;
      }
    }

    qsort(all, (size_t) n_all, sizeof(verdict), by_gid_asc);

    for (int o = 0; o < m->n_local; ++o)
    {
      const int g = ens->local_objects[o]->global_id;
      int lo = 0, hi = n_all - 1;
      while (lo <= hi)
      {
        const int mid = lo + (hi - lo) / 2;
        if (all[mid].gid == g)
        {
          dest[o] = all[mid].dest;
          break;
        }
        if (all[mid].gid < g)
          lo = mid + 1;
        else
          hi = mid - 1;
      }
    }

    free(all);
    free(recv);
    free(rdispl);
    free(rcount);
    free(send);
  }

  /* ── 8. The report, in four scalars ──────────────────────────────────────
   *
   * The load a rank ends with is what it was home to, less what it shed, plus
   * what it was given -- all three known locally by now. `total` is conserved by
   * construction, so only the peak and the idle count have to be reduced. */
  const double post = load - shipped_out + received_in;

  double peak = post;
  MFTLRemeshAgreeDoubleMax(dom, &peak, 1);

  int counts[2] = {(post <= 0.0) ? 1 : 0, reassigned};
  MFTLRemeshSumInts(dom, counts, 2);

  st->n_ranks_idle = counts[0];
  st->n_bodies_reassigned = counts[1];
  st->load_imbalance = (total > 0.0) ? (peak * (double) dom->nproc / total) : 1.0;

  free(placed);
  free(moved);
  free(nb_buf);
  free(mine);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The driver
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLRemeshEnsembleDispersed(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  /* The public entry point owes its own measurement. MFTLRemeshEnsemble does not
   * come through here -- it measured once for the strategy choice and hands that
   * measurement straight to the Measured form below. */
  MFTL_RemeshMeasurement m;
  MFTLRemeshMeasurementInit(&m, ens);

  if (ens->n_bodies_global >= 1)
  {
    MFTLEnsembleComputeTriangleGeometry(ens);
    MFTLRemeshMeasureBodies(ens, m.centroid, m.radius, m.weight, &m.r_max, m.lowest_holder);
  }

  const int rc = MFTLRemeshEnsembleDispersedMeasured(ens, params, stats, &m);
  MFTLRemeshMeasurementFree(&m);
  return rc;
}

int MFTLRemeshEnsembleDispersedMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, const MFTL_RemeshMeasurement *m)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL && m != MFTL_NULL);

  MFTL_Domain *const dom = &ens->domain;

  /* The GLOBAL body count, and only for the guard and for the balancer's
   * replicated table: everything below is collective, so a rank holding nothing
   * must still reach it.
   *
   * There is deliberately NO cached local count. A migration destroys the object
   * of a body a rank keeps nothing of and creates one for a body arriving, so
   * `ens->n_local_objects` and `ens->local_objects` both change under this
   * function -- every loop below re-reads them. */
  const int n = ens->n_bodies_global;

  MFTL_RemeshStats sink;
  MFTL_RemeshStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  st->strategy = MFTL_REMESH_PARALLEL_DISPERSED;
  st->n_coarse_groups = 1;  // no bands, so nothing to coarsen; 1 rather than 0 so the field always reads as a group size

  if (n == 0) return MFTL_SUCCESS;

  /* A zero or negative length band is a caller error that has no cheap symptom:
   * every edge is longer than max_edge_length, so the pass refines without bound
   * until the job is killed. Refusing here turns minutes of silent memory growth
   * into an immediate diagnosable failure. The realistic cause is deriving the
   * target with MFTLMeanEdgeLength on a distributed object -- see the warning on
   * that function. */
  if (!(params->max_edge_length > 0.0) || !(params->min_edge_length >= 0.0) || params->min_edge_length > params->max_edge_length)
  {
    if (dom->rank == 0)
    {
      fprintf(stderr, "[MFTL] %s: refusing to remesh with an edge-length band of [%.6g, %.6g]. Use MFTLEnsembleMeanEdgeLength for a distributed mesh.\n",
              "MFTLRemeshEnsembleDispersed", params->min_edge_length, params->max_edge_length);
    }
    return MFTL_ERROR;
  }

  MFTL_ASSERT(MFTLRemeshParamsAgree(dom, params) &&
              "MFTLRemeshEnsembleDispersed: the ranks were handed different edge-length bands. Nothing communicates the band, so a rank "
              "with a shorter one refines everything it owns, every phase, without bound. The usual cause is "
              "MFTLMeanEdgeLength (LOCAL) on a mesh a collective generator already distributed.");

  int status = MFTL_SUCCESS;

  /* Free-running per-rank accumulation, reduced once on the way out. Same six
   * sections as the spanning driver, so the two are directly comparable -- which
   * is the whole point of measuring them at all. */
  const double t0_call = MFTLRemeshWallTime();
  double t_mark;

  MFTL_RemeshMigrator mig;
  MFTLRemeshMigratorInit(&mig);

  /* ── 1. Where is each body, and how big is it? ───────────────────────────
   *
   * Handed in, not measured here. MFTLRemeshMeasureBodies is two per-body
   * reductions and a scalar, and the caller has already paid them to decide the
   * strategy. Measuring again would double the collectives of the whole pass to
   * recompute an identical answer.
   *
   * The radius is not diagnostic here. It is the ADMISSION TEST, and it has to
   * be taken before the migration rather than after it -- see step 2.
   *
   * The rows describe THIS RANK'S bodies. That is enough for everything except
   * the balancer, which needs the whole ensemble and says so. */
  MFTL_ASSERT(m->n_local == ens->n_local_objects &&
              "MFTLRemeshEnsembleDispersedMeasured: the measurement describes a different local list -- something migrated between the two calls");

  const int n_local = m->n_local;
  int *const dest = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_local, 1) * sizeof(int), "dispersed destinations");

  /* ── 2. Is this strategy applicable at all? ──────────────────────────────
   *
   * ── Why the test is on the RADIUS and not on the result of the migration ──
   *
   * The obvious check -- migrate, then see whether every body came out with no
   * ghosts -- is vacuous, and quietly so. An MFTL_OwnerMap names one destination
   * for every entity of a body, so redistribution puts the body wholly on that
   * rank WHATEVER ITS SIZE: a jet spanning the whole domain would arrive
   * complete on one rank, report itself perfectly local, and remesh correctly.
   * The check would pass and the run would have serialised the largest object in
   * the simulation onto a single rank's memory.
   *
   * So the thing that has to be tested is the precondition itself, and it is
   * geometric: a body wider than a rank box has no rank that can hold it as a
   * matter of design, not of outcome. Two consequences, both real:
   *
   *   MEMORY   one rank holds the entire body while the others hold none of it.
   *   MESSAGES the centering stops being a 26-neighbour exchange, because a
   *            triangle further than one box from the centroid has a
   *            destination outside the stencil, and redistribution correctly
   *            falls back to the GLOBAL communicator -- an all-to-all where the
   *            design promised a neighbourhood.
   *
   * Taking the decision here also means NOTHING HAS MOVED YET when the fallback
   * fires. That matters: the spanning driver's per-phase Exscan assumes every
   * rank entered with the same id watermarks, which a half-migrated,
   * half-remeshed ensemble would not have.
   *
   * The threshold is MFTLRemeshChooseStrategy's, so a caller who let AUTO decide
   * can never reach this branch -- only one who forced the strategy, or whose
   * bodies grew since they measured them. */
  const double frac = (params->dispersed_radius_fraction > 0.0) ? params->dispersed_radius_fraction : 0.4;

  /* THE SAME YARDSTICK THE CHOOSER USED, and it has to be derived the same way
   * rather than passed in: `r_max` is the largest radius, so the box it implies
   * is a property of the ensemble and every rank recomputes it identically from
   * data it already holds. A chooser and a driver disagreeing here would elect
   * DISPERSED and then reject it, costing a measurement and a fall back to
   * spanning -- visible only as `n_bodies_not_centred`. */
  const double h_min = dispersed_box_min(dom, params, m->r_max, n);

  /* Counted over THIS RANK'S bodies and summed, rather than read off a
   * replicated radius array. The verdict `too_large > 0` was already agreed by a
   * collective -- it decides whether the whole ensemble falls back to the
   * spanning driver, so it always was -- and this only changes what is summed
   * into it.
   *
   * ONCE PER BODY, NOT ONCE PER HOLDER. Every holder of a body reduced to the
   * same radius, so an unqualified sum reports a body held by four ranks four
   * times -- which the regime sweep printed as "4 of 1 bodies are wider than a
   * rank box". `lowest_holder` designates one of them and costs no collective;
   * see MFTL_RemeshMeasurement. */
  /* TWO VERDICTS ON ONE MESSAGE, because both are gates and both are sums of a
   * local count. `gate[0]` decides whether this driver runs at all; `gate[1]`
   * decides whether it may BALANCE, and it is the reachability precondition of
   * the neighbourhood balancer -- see the section banner above. Folding the
   * second in here is what makes the balancer's precondition free: it is one
   * more int on a reduction the pass was taking anyway. */
  int gate[2] = {0, 0};
  if (dom->nproc > 1)
  {
    for (int o = 0; o < n_local; ++o)
    {
      if (h_min > 0.0 && m->radius[o] > frac * h_min && (int) m->lowest_holder[o] == dom->rank) gate[0]++;

      /* Counted once per HOLDER that cannot reach the body's home rank, not once
       * per body: that is what makes it a purely local count, and the number is
       * a diagnostic rather than a body count. Zero is the only value that
       * changes what happens. */
      if (neighbor_slot(dom, MFTLGetOwnerProc(dom, &m->centroid[3 * o])) == MFTL_ERROR) gate[1]++;
    }
  }
  MFTLRemeshSumInts(dom, gate, 2);
  const int too_large = gate[0];
  st->n_bodies_not_centred = gate[0];
  st->n_bodies_unreachable = gate[1];

  if (too_large > 0)
  {
    free(dest);
    MFTLRemeshMigratorFree(&mig);

    if (dom->rank == 0)
    {
      fprintf(stderr,
              "[MFTL] dispersed remeshing: %d of %d bodies are wider than %.2f * h_min (%.4g) and cannot be "
              "centred on one rank. Falling back to the spanning driver for the whole ensemble.\n",
              too_large, n, frac, frac * h_min);
    }

    /* The spanning driver zeroes and refills the stats, so the one field that
     * describes THIS decision has to be carried across by hand. Nothing else
     * should survive: the pass being reported is the one that ran. */
    const int rc = MFTLRemeshEnsembleParallel(ens, params, st);
    st->n_bodies_not_centred = too_large;
    return rc;
  }

  /* ── Centroid ownership, then balance ────────────────────────────────────
   *
   * ONE path now, and its payload is bounded by what this rank holds times the
   * graph degree -- never by `n_bodies_global` and never by `nproc`. A body goes
   * to the rank owning its centroid unless that rank is over the cap and a
   * neighbour granted it room; the verdict is decided on ONE rank and delivered
   * one hop, which is what removes the replicated table this used to need.
   *
   * With the balancer off the same call reports the imbalance and changes
   * nothing, so the diagnostic no longer costs a different code path -- and it
   * costs four scalars rather than the `[nproc]` reduction it used to. */
  assign_destinations(ens, params, m, gate[1], dest, st);

  /* ── 3. Move each body whole ─────────────────────────────────────────────── */
  t_mark = MFTLRemeshWallTime();
  MFTL_RemeshMigrationReport mig_report;
  if (MFTLRemeshMigrateToRanks(&mig, ens, dest, &mig_report) != MFTL_SUCCESS) status = MFTL_ERROR;
  st->t_migrate += MFTLRemeshWallTime() - t_mark;
  st->n_entities_migrated += mig_report.n_moved;
  st->n_migrations++;

  free(dest);
  MFTLRemeshMigratorFree(&mig);

  /* One destination per body means no rank can end up holding part of one, so
   * the serial kernel's own precondition is met by construction rather than by
   * inspection. Asserted because it is the hinge of the whole strategy and it is
   * free in Release. */
#ifndef NDEBUG
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    MFTL_ASSERT(obj->verts.n_local == obj->verts.n && obj->tris.n_local == obj->tris.n &&
                "dispersed remeshing: a body kept ghosts after centering -- the owner map did not name one rank");
  }
#endif

  /* ── 4. The untouched serial kernel, on whole bodies ─────────────────────
   *
   * MFTLRemeshObject, not MFTLRemeshIterate: a body with no ghosts is exactly
   * what the serial entry point is specified for, down to its assertion. A rank
   * holding nothing of a body enters with tris.n == 0 and returns immediately,
   * which is why this loop needs no ownership test of its own. */
  t_mark = MFTLRemeshWallTime();
  int local_ops = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (obj->tris.n == 0) continue;

    MFTL_RemeshStats one;
    memset(&one, 0, sizeof(one));

    if (MFTLRemeshObject(obj, params, &one) != MFTL_SUCCESS) status = MFTL_ERROR;
    remesh_stats_accumulate(st, &one);
    local_ops += one.n_split + one.n_collapse + one.n_flip;

    /* Unconditionally, whatever `renumber_global_ids` says. In serial that flag
     * is a caller preference; here density is a REQUIREMENT of the migration
     * that follows, because redistribution sizes its arrival table from
     * next_*_gid and a pass that leaves holes grows that table without bound
     * over a run. Idempotent when MFTLRemeshObject already did it. */
    MFTLRemeshRenumberGlobalIds(obj);
  }
  st->t_sweep += MFTLRemeshWallTime() - t_mark;

  /* ── 5. Make the id counters agree, in one collective ────────────────────
   *
   * Exactly one rank holds any entity of a body, so its counter is the answer
   * and everyone else contributes zero -- MAX picks the owner's value without
   * anyone having to know who the owner is. Contributing the STALE pre-pass
   * value instead would be wrong in the one direction that matters: a body that
   * shrank would keep the larger old count, and the arrival tables would then be
   * sized for ids nobody holds.
   *
   * The convergence counter rides the same message. It is reported rather than
   * looped on -- there is only ever one pass here. */
  /* Re-read, not `n_local`: the migration in step 3 changed which bodies this
   * rank holds, which is the whole point of it. */
  const int n_held = ens->n_local_objects;
  MFTL_BodyReduceOp cops[2] = {MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MAX};
  double *const counters = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(2 * n_held, 1), sizeof(double), "dispersed id counters");

  for (int o = 0; o < n_held; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    counters[2 * o + 0] = (obj->verts.n > 0) ? (double) obj->next_vertex_gid : 0.0;
    counters[2 * o + 1] = (obj->tris.n > 0) ? (double) obj->next_triangle_gid : 0.0;
  }

  /* PER BODY, over that body's holders, and indexed by LOCAL SLOT. The staging
   * buffer this replaces was `2 * n_bodies_global + 1` ints on every rank -- the
   * id space, because an `MPI_Allreduce` aligns element k of every rank's buffer
   * and the body id is the only index two ranks holding different bodies agree
   * about. Moving that indexing inside MFTLEnsembleReduceBodies() is what makes
   * the per-rank memory follow the holding rather than the ensemble.
   *
   * In this driver a body is normally held by exactly ONE rank when this runs --
   * step 3 centred it -- so the reduction is the identity and costs the message
   * and nothing else. It is taken anyway because "normally" is not "always": a
   * rank can still hold an EMPTY object for a body, and such a rank goes on to a
   * redistribution that sizes its arrival table from `next_*_gid`. An empty
   * holder is also exactly what makes MFTLEnsembleBodyHoldersFitTheGraph() veto,
   * so that case takes the global route and gets the right answer there. */
  t_mark = MFTLRemeshWallTime();
  MFTLEnsembleReduceBodies(ens, counters, 2, cops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* One int of pure latency, and it cannot ride the per-body message: it is a
   * maximum over RANKS, not over the holders of any one body. */
  int busiest = local_ops;
  MFTLRemeshAgreeIntMax(dom, &busiest, 1);
  st->t_reduce += MFTLRemeshWallTime() - t_mark;

  for (int o = 0; o < n_held; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    obj->next_vertex_gid = (int) counters[2 * o + 0];
    obj->next_triangle_gid = (int) counters[2 * o + 1];
    obj->next_halfedge_gid = 3 * obj->next_triangle_gid;
  }

  /* MAX rather than SUM, and it is the right reduction here: exactly one rank
   * holds any entity of a body, so no two ranks operate on the same body and the
   * maximum is the busiest rank's count. It is reported as such -- see the
   * locality note on MFTL_RemeshStats. */
  st->n_ops_busiest_rank = busiest;
  free(counters);

  /* One pass over whole bodies IS the serial answer, so there is nothing left
   * for a second placement to reveal. Unlike the shift schedule, this driver has
   * no notion of an entity it was not allowed to touch. */
  st->n_phases = 1;
  st->converged = MFTL_TRUE;
  st->n_edges_hidden = 0;

  /* ── 6. Back to the decomposition the flow solver expects ────────────────── */
  t_mark = MFTLRemeshWallTime();
  MFTLEnsembleComputeTriangleGeometry(ens);
  st->t_geometry += MFTLRemeshWallTime() - t_mark;

  t_mark = MFTLRemeshWallTime();
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) status = MFTL_ERROR;
  st->t_migrate += MFTLRemeshWallTime() - t_mark;
  st->n_migrations++;

  MFTLEnsembleUpdateGlobalCounts(ens);

  /* Redistribution leaves `tris.area/normal/center` unspecified by design, and
   * MFTLRemeshObject promises its caller current triangle geometry. This entry
   * point promises the same thing rather than being a different contract. */
  t_mark = MFTLRemeshWallTime();
  MFTLEnsembleComputeTriangleGeometry(ens);
  st->t_geometry += MFTLRemeshWallTime() - t_mark;

  /* The maximum over ranks, and unconditional -- see the same block at the end
   * of MFTLRemeshEnsembleParallel. Here the imbalance it exposes is the POINT:
   * this strategy sends every body to one rank, so at N_bodies < nproc most
   * ranks contribute a `t_sweep` of zero and the max is one rank's whole cost. */
  st->t_total = MFTLRemeshWallTime() - t0_call;
  MFTLRemeshAgreeProfile(dom, st, MFTL_NULL, 0);

  return status;
}
