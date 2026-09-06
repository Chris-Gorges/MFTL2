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

/** @brief Defined below; the measurement needs current triangle geometry. */
static void prepare_for_migration(MFTL_Ensemble *ens);

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
  prepare_for_migration(ens);
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
 * THE ASSIGNMENT IS A PURE FUNCTION OF REPLICATED DATA, AND EVERYTHING ELSE
 * HERE FOLLOWS FROM THAT.
 *
 * Every rank runs this loop and every rank must reach the same answer, because
 * MFTL_OwnerMap is undefined -- not slow, undefined -- if two ranks name
 * different destinations for the same entity. There is no reduction anywhere
 * below: the centroids and the weights came out of MFTLRemeshMeasureBodies'
 * Allreduce and are already identical to the bit, and `rank_map` is replicated
 * by construction.
 *
 * That is also why `dom->neighbors` must not be used. It is THIS rank's
 * neighbour list and differs from every other rank's, so a candidate ordering
 * derived from it would split the ranks silently. The 26 cells around a body
 * are re-derived from the body's own centroid instead, which is replicated.
 *
 * Determinism extends to the tie-breaks. Two ranks with equal load are ordered
 * by rank id and two bodies of equal weight by body index, so the comparator is
 * a total order and the result does not depend on qsort's stability.
 */

/**
 * @brief The distinct ranks owning the up-to-27 cells around @p pos.
 *
 * Mirrors the enumeration MFTLEnsembleCreate uses to build the neighbour graph
 * -- periodic axes wrap, walled axes reject -- and de-duplicates linearly, which
 * matters for the same reason it does there: on a thin grid several offsets map
 * to the same rank, and a duplicate would be considered twice as a destination.
 *
 * The centre cell comes first, so a body that fits under the cap on its own home
 * rank is never moved by a search that starts here.
 *
 * @param      dom The MPI/physical world.
 * @param      pos A position, already wrapped into the domain.
 * @param[out] out [27] distinct ranks, centre first.
 * @return How many were written.
 * @rank_local Pure arithmetic on replicated data.
 */
static int cells_around(const MFTL_Domain *dom, const double pos[3], int out[27])
{
  int base[3];
  for (int k = 0; k < 3; ++k)
  {
    int i = (int) ((pos[k] - dom->global_min[k]) * dom->grid_factors[k]);
    if (i < 0) i = 0;
    if (i >= dom->cuts[k]) i = dom->cuts[k] - 1;
    base[k] = i;
  }

  int n = 0;
  for (int d = 0; d < 27; ++d)
  {
    /* Offsets enumerated so that d == 0 is (0,0,0): the home cell has to be the
     * first candidate, or a balanced ensemble would still shuffle every body. */
    const int dx = (d == 0) ? 0 : (d % 3) - 1;
    const int dy = (d == 0) ? 0 : ((d / 3) % 3) - 1;
    const int dz = (d == 0) ? 0 : (d / 9) - 1;
    if (d != 0 && dx == 0 && dy == 0 && dz == 0) continue;

    int c[3];
    bool ok = MFTL_TRUE;
    const int off[3] = {dx, dy, dz};
    for (int k = 0; k < 3 && ok; ++k)
    {
      int i = base[k] + off[k];
      if (i < 0 || i >= dom->cuts[k])
      {
        if (!dom->is_periodic[k])
        {
          ok = MFTL_FALSE;
          break;
        }
        i = (i + dom->cuts[k]) % dom->cuts[k];
      }
      c[k] = i;
    }
    if (!ok) continue;

    const int r = dom->rank_map[c[0] + c[1] * dom->cuts[0] + c[2] * dom->cuts[0] * dom->cuts[1]];

    bool seen = MFTL_FALSE;
    for (int j = 0; j < n && !seen; ++j)
    {
      if (out[j] == r) seen = MFTL_TRUE;
    }
    if (!seen) out[n++] = r;
  }

  return n;
}

/** @brief One body's weight and index, for the heaviest-first ordering. */
typedef struct
{
  double w;  ///< Global owned-triangle count of the body.
  int o;  ///< Body index, and the tie-break that makes the order total.
} weighted_body;

/**
 * @brief Orders bodies heaviest first, ties by ascending index.
 *
 * The index tie-break is what makes this a TOTAL order, which is what makes the
 * result independent of qsort's (unspecified) stability -- and therefore
 * identical on every rank and on every libc. Two ranks ordering equal-weight
 * bodies differently would assign them to different destinations.
 *
 * @param a First element.
 * @param b Second element.
 * @return Negative, zero or positive in the usual qsort sense.
 */
static int by_weight_desc(const void *a, const void *b)
{
  const weighted_body *const x = (const weighted_body *) a;
  const weighted_body *const y = (const weighted_body *) b;

  if (x->w > y->w) return -1;
  if (x->w < y->w) return 1;
  return (x->o < y->o) ? -1 : ((x->o > y->o) ? 1 : 0);
}

/**
 * @brief Chooses a destination rank for every body: locality first, then load.
 *
 * ── The policy, and why it is that way round ──────────────────────────────
 *
 * A body stays on the rank owning its centroid unless that rank is over the cap.
 * Object-centering is a 26-neighbour exchange exactly while every destination is
 * within a box of the body, so the bodies that DO NOT MOVE are the ones that
 * keep the stencil -- and in any ensemble worth balancing that is most of them.
 * A body that must move prefers the least-loaded of the 26 cells around it, and
 * only falls back to the globally least-loaded rank when none of those has room.
 *
 * ── Why it terminates ─────────────────────────────────────────────────────
 *
 * Two guarantees, and both are needed. A body is only ever moved to a rank where
 * it still FITS under the cap, so a rank that is under the cap stays under it and
 * is never processed; and the cap is at least the heaviest single body, so a rank
 * holding one body is never over it. A body therefore moves at most once, and the
 * outer loop settles one rank per iteration.
 *
 * `cap = MAX(mean * slack, heaviest)` -- the second term is not defensive
 * padding. A body is indivisible here, so a cap below the heaviest body asks for
 * an assignment that does not exist, and the loop would spin looking for it.
 *
 * @param      dom      The MPI/physical world.
 * @param      params   Supplies `dispersed_balance_slack`. <= 0 disables.
 * @param      centroid [3*n] REPLICATED body centroids, indexed by body id.
 * @param      weight   [n] replicated per-body global triangle counts, or
 *                      #MFTL_NULL for one unit each.
 * @param      n        Bodies in the ensemble.
 * @param[out] dest     [n] destination rank per body, by body id.
 * @param[out] st       Receives `n_bodies_reassigned`, `n_ranks_idle` and
 *                      `load_imbalance`.
 * @rank_local No MPI at all -- see the section banner for why it must be so.
 */
static void assign_destinations(const MFTL_Domain *dom, const MFTL_RemeshParameters *params, const double *centroid, const double *weight, int n, int *dest,
                                MFTL_RemeshStats *st)
{
  const int nproc = dom->nproc;

  double *const load = (double *) MFTLCallocOrDie((size_t) nproc, sizeof(double), "dispersed rank loads");
  bool *const settled = (bool *) MFTLCallocOrDie((size_t) nproc, sizeof(bool), "dispersed settled ranks");
  weighted_body *const order = (weighted_body *) MFTLMallocOrDie((size_t) n * sizeof(weighted_body), "dispersed body order");

  double total = 0.0;
  double heaviest = 0.0;

  for (int o = 0; o < n; ++o)
  {
    /* The PRODUCTION ownership rule, on a position every rank agrees about. A
     * centroid that came out of an Allreduce is identical everywhere to the bit,
     * so this is a pure function of replicated data and every rank picks the same
     * home -- which is what MFTL_OwnerMap requires, balanced or not. */
    dest[o] = MFTLGetOwnerProc(dom, &centroid[3 * o]);

    /* A weightless body still occupies a rank, so it counts as one unit rather
     * than as nothing. Zero is what a body with no triangles anywhere reports,
     * and giving it zero weight would let a rank accumulate any number of them
     * while reporting a perfect balance. */
    const double w = (weight != MFTL_NULL && weight[o] > 0.0) ? weight[o] : 1.0;
    order[o].w = w;
    order[o].o = o;

    load[dest[o]] += w;
    total += w;
    if (w > heaviest) heaviest = w;
  }

  const double slack = params->dispersed_balance_slack;
  if (slack > 0.0 && nproc > 1 && n > 1)
  {
    const double mean = total / (double) nproc;
    const double cap = MFTL_MAX(mean * slack, heaviest);

    qsort(order, (size_t) n, sizeof(weighted_body), by_weight_desc);

    for (int iter = 0; iter < nproc; ++iter)
    {
      int worst = MFTL_ERROR;
      for (int r = 0; r < nproc; ++r)
      {
        if (settled[r] || load[r] <= cap) continue;
        if (worst == MFTL_ERROR || load[r] > load[worst]) worst = r;
      }
      if (worst == MFTL_ERROR) break;

      bool moved_any = MFTL_FALSE;

      for (int i = 0; i < n && load[worst] > cap; ++i)
      {
        const int o = order[i].o;
        if (dest[o] != worst) continue;

        const double w = order[i].w;

        int cand[27];
        const int n_cand = cells_around(dom, &centroid[3 * o], cand);

        int target = MFTL_ERROR;
        for (int j = 0; j < n_cand; ++j)
        {
          const int r = cand[j];
          if (r == worst || load[r] + w > cap) continue;
          /* Least-loaded of the neighbourhood, ties by rank id so that two ranks
           * evaluating this loop cannot pick differently. */
          if (target == MFTL_ERROR || load[r] < load[target] || (load[r] == load[target] && r < target)) target = r;
        }

        if (target == MFTL_ERROR)
        {
          /* Nothing adjacent has room. Anywhere is better than leaving the rank
           * over the cap -- this is the body that pays for the balance by
           * leaving the 26-neighbourhood, and `n_bodies_reassigned` counts it. */
          for (int r = 0; r < nproc; ++r)
          {
            if (r == worst || load[r] + w > cap) continue;
            if (target == MFTL_ERROR || load[r] < load[target] || (load[r] == load[target] && r < target)) target = r;
          }
        }

        if (target == MFTL_ERROR) continue;  // this body fits nowhere; try the next

        dest[o] = target;
        load[worst] -= w;
        load[target] += w;
        moved_any = MFTL_TRUE;
      }

      /* Settled either way: if nothing could leave, nothing ever will -- the
       * loads only rise from here. Without this the outer loop would pick the
       * same rank forever. */
      if (!moved_any || load[worst] <= cap) settled[worst] = MFTL_TRUE;
    }
  }

  int reassigned = 0;
  for (int o = 0; o < n; ++o)
  {
    if (dest[o] != MFTLGetOwnerProc(dom, &centroid[3 * o])) reassigned++;
  }

  int idle = 0;
  double peak = 0.0;
  for (int r = 0; r < nproc; ++r)
  {
    if (load[r] <= 0.0) idle++;
    if (load[r] > peak) peak = load[r];
  }

  st->n_bodies_reassigned = reassigned;
  st->n_ranks_idle = idle;
  st->load_imbalance = (total > 0.0) ? (peak * (double) nproc / total) : 1.0;

  free(order);
  free(settled);
  free(load);
}

/**
 * @brief Broadcasts the local measurement rows into replicated per-body arrays.
 *
 * ══ THE ONE `O(bodies)` TERM THIS MODULE STILL HAS, AND IT IS NOT AN OVERSIGHT
 *
 * assign_destinations() is a greedy over RANK LOADS, and a rank's load is a sum
 * over bodies that may be held anywhere in the job. Every rank has to reach the
 * same assignment -- MFTL_OwnerMap is undefined, not slow, if two ranks name
 * different destinations for one entity -- so every rank has to see every body.
 * No communicator choice fixes that; it is a property of the algorithm, and
 * replacing it needs a different balancer rather than a different route.
 *
 * What IS fixed here is that nothing else pays for it. The measurement itself,
 * the strategy ratio, the admission test and the destination of a body this rank
 * holds are all answered from local rows and two scalars; this function is
 * entered only when `dispersed_balance_slack > 0`, and a caller who turns the
 * balancer off pays no per-body collective at all.
 *
 * ── MAX, and the rows are already identical ───────────────────────────────
 *
 * Every holder of a body came out of MFTLEnsembleReduceBodies() with the same
 * value TO THE BIT -- that is the guarantee its ascending-source-rank fold
 * exists to give -- so a maximum over the holders is that value, and a body no
 * rank holds keeps the identity and is recognisable as such. A SUM would have
 * multiplied a shared body by its holder count.
 *
 * @param      ens      The ensemble.
 * @param      m        The local measurement.
 * @param[out] centroid `[3 * n_bodies_global]` replicated centroids.
 * @param[out] weight   `[n_bodies_global]` replicated triangle counts.
 * @collective One MFTLRemeshAgreeDoubleMax() of `4 * n_bodies_global` doubles.
 */
static void replicate_for_balancer(MFTL_Ensemble *ens, const MFTL_RemeshMeasurement *m, double *centroid, double *weight)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int n = ens->n_bodies_global;

  double *const glo = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(4 * n, 1) * sizeof(double), "dispersed balancer staging");

  for (int i = 0; i < 4 * n; ++i) glo[i] = -MFTL_DBL_MAX;

  for (int o = 0; o < m->n_local; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    MFTL_ASSERT(g >= 0 && g < n && "dispersed balancer: body id outside the id space");
    glo[4 * g + 0] = m->centroid[3 * o + 0];
    glo[4 * g + 1] = m->centroid[3 * o + 1];
    glo[4 * g + 2] = m->centroid[3 * o + 2];
    glo[4 * g + 3] = m->weight[o];
  }

  MFTLRemeshAgreeDoubleMax(dom, glo, 4 * n);

  for (int g = 0; g < n; ++g)
  {
    if (glo[4 * g] == -MFTL_DBL_MAX)
    {
      /* Nobody holds it. It has no entities to route, so any destination is as
       * good as any other -- but it must be the SAME one on every rank, and it
       * must not be a coordinate no cell contains. */
      for (int k = 0; k < 3; ++k) centroid[3 * g + k] = dom->global_min[k];
      weight[g] = 0.0;
      continue;
    }
    for (int k = 0; k < 3; ++k) centroid[3 * g + k] = glo[4 * g + k];
    weight[g] = glo[4 * g + 3];
  }

  free(glo);
}

/**
 * @brief The load table and the imbalance statistics, without replicating anything.
 *
 * The path taken when the balancer is off: a body stays on the rank owning its
 * centroid, which every holder can decide for itself from its own row. What is
 * still global is the REPORT -- `n_ranks_idle` and `load_imbalance` are sums over
 * bodies held elsewhere -- and that costs `nproc` doubles rather than
 * `n_bodies_global`, i.e. it scales with the machine and not with the
 * simulation.
 *
 * A body is counted by every holder, so the contribution is divided among them:
 * that is what `share` is, and it is why the weight is not simply added by each.
 *
 * @param      ens  The ensemble.
 * @param      m    The local measurement.
 * @param[out] dest `[m->n_local]` destination rank per LOCAL SLOT.
 * @param[out] st   Receives `n_bodies_reassigned` (always 0 here), `n_ranks_idle`
 *                  and `load_imbalance`.
 * @collective One MFTLRemeshAgreeDouble() of `nproc` doubles.
 */
static void assign_destinations_local(MFTL_Ensemble *ens, const MFTL_RemeshMeasurement *m, int *dest, MFTL_RemeshStats *st)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int nproc = dom->nproc;

  double *const load = (double *) MFTLCallocOrDie((size_t) nproc, sizeof(double), "dispersed rank loads");

  for (int o = 0; o < m->n_local; ++o)
  {
    dest[o] = MFTLGetOwnerProc(dom, &m->centroid[3 * o]);

    /* EACH HOLDER CONTRIBUTES ITS OWN SHARE, so the sum over ranks is the body's
     * global weight and not that weight once per holder. `m->weight[o]` is the
     * reduced owned-triangle count of the whole body and `tris.n_local` is this
     * rank's part of it, so the parts add up to the whole by construction.
     *
     * A body with no triangles anywhere still occupies a rank, so it counts as
     * one unit; it is credited by its lowest-numbered holder, for the same
     * reason the admission test below is. */
    const int mine = ens->local_objects[o]->tris.n_local;
    load[dest[o]] += (m->weight[o] > 0.0) ? (double) mine : (((int) m->lowest_holder[o] == dom->rank) ? 1.0 : 0.0);
  }

  MFTLRemeshAgreeDouble(dom, load, nproc);

  double total = 0.0, peak = 0.0;
  int idle = 0;
  for (int r = 0; r < nproc; ++r)
  {
    total += load[r];
    if (load[r] > peak) peak = load[r];
    if (load[r] <= 0.0) idle++;
  }

  st->n_bodies_reassigned = 0;
  st->n_ranks_idle = idle;
  st->load_imbalance = (total > 0.0) ? (peak * (double) nproc / total) : 1.0;

  free(load);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The driver
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Accumulates one object's stats into the ensemble total.
 *
 * @param[out] dst The caller's statistics block.
 * @param      src One object's statistics from this pass.
 */
static void accumulate_stats(MFTL_RemeshStats *dst, const MFTL_RemeshStats *src)
{
  dst->n_split += src->n_split;
  dst->n_collapse += src->n_collapse;
  dst->n_flip += src->n_flip;
  dst->n_collapse_rejected_link += src->n_collapse_rejected_link;
  dst->n_collapse_rejected_fold += src->n_collapse_rejected_fold;
  dst->n_collapse_rejected_feature += src->n_collapse_rejected_feature;
  dst->n_flip_rejected_fold += src->n_flip_rejected_fold;
  dst->n_feature_edges += src->n_feature_edges;
  dst->n_corner_vertices += src->n_corner_vertices;
  dst->n_ridge_vertices += src->n_ridge_vertices;
  dst->n_free_vertices += src->n_free_vertices;
  dst->n_vertices_removed += src->n_vertices_removed;
  dst->n_orphans_removed += src->n_orphans_removed;
  dst->iterations = MFTL_MAX(dst->iterations, src->iterations);
}

/**
 * @brief Refreshes centres over the FULL triangle count and repairs the derived
 *        arrays -- the two preconditions of any migration.
 *
 * The centres must cover ghosts as well as owned triangles even here, where the
 * destination does not depend on them: `MFTL_OwnerMap` bypasses the geometric
 * rule, but the redistribution still uses the centres for its own bookkeeping in
 * Debug, and the migration BACK at the end is the geometric one.
 *
 * @param ens The ensemble. Rebuilds the derived arrays and refreshes the
 *            triangle centres over the FULL count including ghosts.
 */
static void prepare_for_migration(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

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
    prepare_for_migration(ens);
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
  int too_large = 0;
  if (dom->nproc > 1 && h_min > 0.0)
  {
    for (int o = 0; o < n_local; ++o)
    {
      if (m->radius[o] > frac * h_min && (int) m->lowest_holder[o] == dom->rank) too_large++;
    }
  }
  MFTLRemeshSumInts(dom, &too_large, 1);
  st->n_bodies_not_centred = too_large;

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
   * Two paths, and which one runs is what decides whether this driver has an
   * `O(bodies in the ensemble)` term at all.
   *
   * BALANCER OFF. A body goes to the rank owning its centroid, which every
   * holder decides from its own measurement row. Nothing is replicated and
   * nothing is `O(bodies)`; the only collective is `nproc` doubles for the
   * imbalance report.
   *
   * BALANCER ON (the default). The greedy is over RANK LOADS, and a rank's load
   * sums bodies held anywhere in the job, so every rank has to see every body or
   * two of them will name different destinations for one entity -- which
   * MFTL_OwnerMap does not merely handle badly, it leaves undefined. The
   * replication is therefore explicit, priced and attributable to one switch.
   * See replicate_for_balancer(). */
  if (params->dispersed_balance_slack > 0.0 && dom->nproc > 1 && n > 1)
  {
    double *const rep = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(4 * n, 1) * sizeof(double), "dispersed replicated measurement");
    double *const rep_centroid = rep;
    double *const rep_weight = rep + 3 * n;
    int *const rep_dest = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(int), "dispersed replicated destinations");

    replicate_for_balancer(ens, m, rep_centroid, rep_weight);
    assign_destinations(dom, params, rep_centroid, rep_weight, n, rep_dest, st);

    for (int o = 0; o < n_local; ++o) dest[o] = rep_dest[ens->local_objects[o]->global_id];

    free(rep_dest);
    free(rep);
  }
  else
  {
    assign_destinations_local(ens, m, dest, st);
  }

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
    accumulate_stats(st, &one);
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
  prepare_for_migration(ens);
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
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
  st->t_geometry += MFTLRemeshWallTime() - t_mark;

  /* The maximum over ranks, and unconditional -- see the same block at the end
   * of MFTLRemeshEnsembleParallel. Here the imbalance it exposes is the POINT:
   * this strategy sends every body to one rank, so at N_bodies < nproc most
   * ranks contribute a `t_sweep` of zero and the max is one rank's whole cost. */
  st->t_total = MFTLRemeshWallTime() - t0_call;

  double prof[6] = {st->t_total, st->t_migrate, st->t_geometry, st->t_freeze, st->t_sweep, st->t_reduce};
  MFTLRemeshAgreeDoubleMax(dom, prof, 6);
  st->t_total = prof[0];
  st->t_migrate = prof[1];
  st->t_geometry = prof[2];
  st->t_freeze = prof[3];
  st->t_sweep = prof[4];
  st->t_reduce = prof[5];

  return status;
}
