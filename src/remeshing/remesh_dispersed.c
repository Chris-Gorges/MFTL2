/**
 * @file remesh_dispersed.c
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief Distributed remeshing for bodies SMALLER than a rank box: move the
 *        bodies, not the cut planes.
 *
 * ## The same problem from the opposite end
 *
 * An edge whose endpoints belong to two ranks can be collapsed by neither.
 * remesh_parallel.c moves the cut planes until every entity has been interior
 * somewhere; for a body that fits in a rank box it is simpler to migrate the
 * whole body onto one rank, where no edge is shared. Both are exact; R/h
 * decides which is cheaper:
 *
 *              R >> h                              R << h
 *   move       the planes                          the bodies
 *   phases     about 4 at budget 4 (np = 2..8)     1
 *   migrations one per placement, plus one         exactly 2
 *   frozen     O(delta/h) of the mesh              nothing
 *   id cost    per-phase holder prefix sums        one per-body MAX, total
 *   weakness   phase count, band precondition      load balance at N_bodies < P
 *
 * ## Why the ids need almost nothing
 *
 * After centering exactly ONE rank holds any entity of a body, and ids live in a
 * per-body space (`next_*_gid` is per object), so that rank's counter is already
 * unique for the body. The operators run untouched, MFTLRemeshRenumberGlobalIds()
 * makes the ids dense on the owner, and the other holders learn the counters by
 * one per-body MAX over the holders (a rank holding nothing contributes zero).
 *
 * ## Ordering
 *
 * Admission is decided BEFORE anything is modified: if a body is too large, the
 * whole ensemble goes to the spanning driver, whose per-phase id scan assumes
 * all holders enter with the same id counters -- which a half-remeshed
 * ensemble would violate. The test uses radii already reduced for the strategy
 * decision, so it costs one small reduction.
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
   * unconditionally is one collective on a path that is about to run a dozen.
   * The caller has computed the geometry it reads. */
  MFTLRemeshMeasureBodies(ens, m->centroid, m->radius, m->weight, &m->r_max, m->lowest_holder);

  if (MFTLDomainGetHMin(&ens->domain) <= 0.0) return MFTL_REMESH_PARALLEL_SPANNING;

  /* ONE SCALAR, identical to the bit on every rank (a scalar MPI_MAX inside the
   * measurement). A maximum over this rank's own radii would differ between
   * ranks and split them between two drivers -- a deadlock. */
  const double box = dispersed_box_min(&ens->domain, params, m->r_max, ens->n_bodies_global);
  return (m->r_max > frac * box) ? MFTL_REMESH_PARALLEL_SPANNING : MFTL_REMESH_PARALLEL_DISPERSED;
}

MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategy(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens->domain.nproc < 2 || ens->n_bodies_global == 0) return MFTL_REMESH_PARALLEL_SPANNING;

  MFTL_RemeshMeasurement m;
  MFTLRemeshMeasurementInit(&m, ens);
  MFTLEnsembleComputeTriangleGeometry(ens);
  const MFTL_RemeshParallelStrategy strategy = MFTLRemeshChooseStrategyMeasured(ens, params, &m);
  MFTLRemeshMeasurementFree(&m);
  return strategy;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1b. Load balancing
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * ## One rank decides each body and tells the holders
 *
 * MFTL_OwnerMap is undefined if two ranks name different destinations for one
 * entity, so a body's destination must be AGREED. Replicating every body's
 * centroid and weight on every rank and running one deterministic greedy would
 * cost O(bodies in the ensemble) payload and memory per rank. Instead one rank
 * decides and sends the verdict: O(local bodies x degree), degree <= 26.
 *
 * ## The precondition is a local test
 *
 * Let home(o) = MFTLGetOwnerProc(centroid(o)), identical on every holder because
 * the centroid is reduced. Required:
 *
 *     for every body o and every holder H of o:  home(o) is H or a neighbour of H
 *
 * Each holder tests its own half without communication; one reduction agrees
 * it. It gives exactly the two messages needed: home can be told about o
 * (lowest_holder(o), a genuine holder, is its neighbour; the graph is
 * symmetric), and home can tell every holder. Not required: a clique of holders,
 * a holder owning an entity, or home holding the body
 * (MFTLEnsembleBodyHoldersFitTheGraph() is strictly stronger and deliberately
 * not used). On failure the pass reports n_bodies_unreachable and uses plain
 * centroid ownership -- never an O(B) replicated fallback.
 *
 * ## A negotiation, because a local rule cannot see the global load
 *
 * Every overloaded rank sees the same idle neighbour. So each rank advertises
 * its room (cap - load); an overloaded rank asks a FEW neighbours, emptiest
 * first, for all of their room until its excess (plus one body) is covered; each
 * neighbour grants its real room to its askers neediest first, so
 * SUM(grants) <= room and nobody is pushed past the cap however many ship at
 * once. Two failure modes shaped this, both silent: a proportional ask over 26
 * neighbours yields grants smaller than one body, so nothing moves (108 bodies
 * over 27 ranks stayed at imbalance 2.00); and a tie-break identical on every
 * rank starves one target, so ties break on the neighbour's cell offset
 * relative to the asker. Rounds are bounded (MFTL_DISPERSED_BALANCE_ROUNDS):
 * running out leaves an imbalance that is reported, never a wrong mesh.
 *
 * ## What one hop costs
 *
 * Weight moves at most one cell per pass: a pile-up in one rank box is fixed
 * when its neighbours have room; one spanning many cells only partly.
 *
 * Every body is decided by exactly one rank, so a flawed tie-break yields a
 * different assignment, never a corrupt mesh; the tie-breaks are total anyway,
 * for run-to-run reproducibility.
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
 * @brief Ask/grant/ship rounds per balancing pass: bounded, not convergent.
 *
 * Imbalance reached against a slack of 1.25 (`examples/remesh_regime_sweep
 * -only l`):
 *
 *   rounds   np=4   np=8   np=27 spread   np=27 clustered
 *   1        1.25   1.25   1.50           1.25
 *   2        1.25   1.25   1.25           1.25
 *   3, 4     1.25   1.25   1.25           1.25
 *
 * A lone pile-up settles in one round (its ask is uncontended); a spread
 * imbalance over a full 26-neighbourhood needs two. Three ship: one round too
 * few is a silently unbalanced pass, one too many costs three directed
 * exchanges of one or two doubles per neighbour.
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

  /* ── 2. The two scalars a cap needs: `total` for the mean and `heaviest` for
   * the floor -- no per-rank load table. */
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

  /* Every buffer of the negotiation is sized by the graph degree (at most 26),
   * in one allocation. */
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
      /* ── 4. Ask FEW neighbours for a lot, and ask again next round ─────────
       *
       * Bodies are indivisible: splitting the excess over all neighbours in
       * proportion to their room returns grants smaller than one body, nothing
       * is placeable, and the pass reports an unchanged imbalance with
       * `n_bodies_reassigned == 0` -- indistinguishable from an ensemble that
       * needed nothing (measured: 108 bodies over 27 ranks stuck at 2.00). So
       * the ask is CONCENTRATED: neighbours emptiest first, each asked for all
       * it advertises, until the budget is covered.
       *
       * Concentrating makes every overloaded rank ask the same emptiest
       * neighbour, which can serve one of them; hence ROUNDS -- served targets
       * advertise less next time and the others move on.
       *
       * The budget is `cur - cap` plus the heaviest unplaced body, because only
       * whole bodies leave and asking for exactly the excess is routinely one
       * body short. A target cannot be pushed over its cap: a grant is bounded
       * by its own room. */
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
  return MFTLRemeshDispersedPass(ens, params, stats, "MFTLRemeshEnsembleDispersed");
}

int MFTLRemeshDispersedPass(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, const char *agree_as)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  /* This form owes its own measurement. MFTLRemeshEnsemble's AUTO path does not
   * come through here -- it measured once for the strategy choice and hands that
   * measurement straight to the Measured form below. */
  MFTL_RemeshMeasurement m;
  MFTLRemeshMeasurementInit(&m, ens);

  if (ens->n_bodies_global >= 1)
  {
    MFTLEnsembleComputeTriangleGeometry(ens);

    /* The parameter check at the first collective, after the local work before
     * it (see MFTLRemeshSpanningPass()). */
    if (agree_as != MFTL_NULL && MFTLRemeshAgreeParams(&ens->domain, params, agree_as) != MFTL_SUCCESS)
    {
      MFTLRemeshMeasurementFree(&m);
      if (stats)
      {
        memset(stats, 0, sizeof(*stats));
        stats->strategy = MFTL_REMESH_PARALLEL_DISPERSED;
        stats->n_coarse_groups = 1;
      }
      return MFTL_ERROR;
    }
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

  /* The GLOBAL body count for the guard: everything below is collective, so a
   * rank holding nothing must still reach it.
   *
   * No cached local count: migrations destroy and create local objects, so
   * `ens->n_local_objects` and `ens->local_objects` change under this function
   * and every loop re-reads them. */
  const int n = ens->n_bodies_global;

  MFTL_RemeshStats sink;
  MFTL_RemeshStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  st->strategy = MFTL_REMESH_PARALLEL_DISPERSED;
  st->n_coarse_groups = 1;  // no bands, so nothing to coarsen; 1 rather than 0 so the field always reads as a group size

  if (n == 0) return MFTL_SUCCESS;

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
   * Handed in by the caller, who already measured to choose the strategy.
   * The radius is the ADMISSION TEST (step 2); the rows describe this rank's
   * bodies, which is all the driver and the balancer need. */
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
   * fires. That matters: the spanning driver's per-phase id scan assumes every
   * holder entered with the same id counters, which a half-migrated,
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

  /* Counted over THIS RANK'S bodies, once per body rather than once per holder:
   * every holder reduced to the same radius, so an unqualified sum reports a
   * body held by four ranks four times. `lowest_holder` designates one. */
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
    const int rc = MFTLRemeshSpanningPass(ens, params, st, MFTL_NULL);
    st->n_bodies_not_centred = too_large;
    return rc;
  }

  /* ── Centroid ownership, then balance ────────────────────────────────────
   *
   * A body goes to the rank owning its centroid unless that rank is over the
   * cap and a neighbour granted room; decided on ONE rank, delivered one hop.
   * With the balancer off the same call only reports the imbalance. */
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
   * what the serial entry point is specified for, down to its assertion, and
   * its canonical ordering is what undoes the arrival order of the migration
   * (kept entities first, then arrivals by source rank). A rank
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

  /* ── 5. Make the id counters agree ───────────────────────────────────────
   *
   * Exactly one rank holds entities of a body, so its counter is the answer
   * and every other holder contributes zero: a per-body MAX over the holders
   * picks it without anyone knowing who the owner is. Contributing the stale
   * pre-pass value would be wrong for a body that shrank. */
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

  /* Per body, over its holders, indexed by LOCAL SLOT. Normally one holder per
   * body at this point, but a rank can still hold an EMPTY object for a body
   * and size its next arrival table from `next_*_gid`; such an empty holder
   * vetoes the neighbourhood route, and the reduction takes the global one. */
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

  /* One pass over whole bodies IS the serial answer -- to the bit, because
   * MFTLRemeshObject() puts the body it is handed into global-id order before it
   * sweeps, and the migration above changes the pool order and nothing else
   * (test_a_dispersed_remesh_is_the_serial_remesh_to_the_bit). So there is
   * nothing left for a second placement to reveal. Unlike the shift schedule,
   * this driver has no notion of an entity it was not allowed to touch. */
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
