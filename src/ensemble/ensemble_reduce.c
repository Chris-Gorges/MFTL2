/**
 * @file ensemble_reduce.c
 * @author Christian Gorges
 * @date 2026-09-04
 * @ingroup mftl_ensemble
 *
 * @brief Per-body reductions taken over the ranks that HOLD each body.
 *
 * Three per-timestep collectives in this library reduce one small record per
 * body -- the quality statistics, the global entity counts and the
 * redistribution skip vote. All three used to do it with an `MPI_Allreduce`
 * over `comm_world` on a buffer indexed by body id, which is the obvious shape
 * and the one that stops scaling first: at a million bodies every rank
 * allocates, fills with identities and reduces a million slots in order to
 * learn about the hundred it holds.
 *
 * This file is the one place that choice is made. The contract and the
 * exactness argument are on MFTLEnsembleReduceBodies() in mftl_ensemble.h; what
 * is here is the implementation and the three things about it that are easy to
 * get wrong:
 *
 *   1. THE CALLER'S BUFFER IS INDEXED BY LOCAL SLOT. Moving the id indexing
 *      inside is most of the saving -- otherwise the collective shrinks and the
 *      per-rank memory does not.
 *   2. CONTRIBUTIONS ARE COMBINED IN ASCENDING SOURCE-RANK ORDER. Floating-point
 *      addition is not associative and the neighbour list is in an MPI-chosen
 *      order that differs from rank to rank, so receive order would leave two
 *      holders of one body disagreeing in the last ulp.
 *   3. THE ROUTE IS A COLLECTIVE DECISION. It changes which communicator is
 *      used, so it is agreed with one int rather than concluded locally.
 */

#include "mftl_ensemble.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_types.h"

/** @brief This rank's route counters. See MFTLEnsembleGetBodyReduceStats(). */
static MFTL_BodyReduceStats g_stats;

void MFTLEnsembleGetBodyReduceStats(MFTL_BodyReduceStats *out)
{
  if (out != MFTL_NULL) *out = g_stats;
}

void MFTLEnsembleResetBodyReduceStats(void) { memset(&g_stats, 0, sizeof(g_stats)); }

/* ═══════════════════════════════════════════════════════════════════════════
 * Slot arithmetic
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The value that leaves a reduction unchanged.
 *
 * @param op The operator.
 * @return \f$+\infty\f$, \f$-\infty\f$ or 0.
 */
static inline double op_identity(MFTL_BodyReduceOp op)
{
  if (op == MFTL_BODY_REDUCE_MIN) return MFTL_DBL_MAX;
  if (op == MFTL_BODY_REDUCE_MAX) return -MFTL_DBL_MAX;
  return 0.0;
}

/**
 * @brief Folds one contribution into an accumulator, slot by slot.
 *
 * @param[in,out] acc  Accumulator row, @p n values.
 * @param         in   Contribution row, @p n values.
 * @param         n    Values per body.
 * @param         ops  [n] operators.
 */
static inline void op_combine(double *acc, const double *in, int n, const MFTL_BodyReduceOp *ops)
{
  for (int j = 0; j < n; ++j)
  {
    switch (ops[j])
    {
      case MFTL_BODY_REDUCE_MIN: acc[j] = fmin(acc[j], in[j]); break;
      case MFTL_BODY_REDUCE_MAX: acc[j] = fmax(acc[j], in[j]); break;
      default: acc[j] += in[j]; break;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Is the neighbourhood enough?
 *
 * The whole test, and it is deliberately local: no geometry, no body-global
 * measurement, and nothing that has to be kept up to date between calls. See
 * the derivation on MFTLEnsembleReduceBodies() -- statements 1 and 2 there are
 * `not a neighbour` and the offset span below.
 *
 * It reads only the GHOST SUFFIX of each pool. Every entity in [0, n_local) is
 * owned here by the ownership-prefix invariant, so it contributes the offset
 * (0,0,0) and nothing else; the halo is what names the other holders. On a body
 * this rank holds whole -- the common case in the regime this route exists for
 * -- there is no ghost suffix and the scan does nothing at all.
 *
 * ── A HOLDER THAT OWNS NOTHING IS INVISIBLE, SO IT VETOES ──────────────────
 *
 * The derivation starts from "a rank holds an entity of a body only if it OWNS
 * one", which is what makes the ghost `owner_proc` a complete list of the other
 * holders. A rank holding an EMPTY object breaks it from the other side: it owns
 * nothing, so it appears in nobody's halo, so no other holder can discover it --
 * and it would sit out the exchange and keep whatever was in its buffer.
 *
 * It cannot be detected from outside, so it reports itself: a rank finding one of
 * its own bodies empty votes the whole ensemble onto the global route. That is
 * not a corner case, it is the state a generated ensemble is in before its first
 * redistribution -- MFTLGenerateSphere() builds every body on rank 0 and every
 * other rank holds the shell. After a redistribution no empty object exists at
 * all (inv_local_directory_is_sound), so in the steady state this costs nothing.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Running state of the holder-set test for ONE body. */
typedef struct
{
  const MFTL_Domain *dom;  ///< For the neighbour list and its offsets.
  int seen[27];  ///< Distinct owners met so far; at most self plus 26 neighbours.
  int n_seen;  ///< Entries in `seen`.
  int last;  ///< Last owner met, so a run of ghosts from one rank costs one compare.
  int lo[3];  ///< Smallest cell offset met, per axis.
  int hi[3];  ///< Largest cell offset met, per axis.
} holder_probe;

/**
 * @brief Starts a probe with this rank -- offset (0,0,0) -- already in it.
 *
 * @param[out] p   The probe.
 * @param      dom The MPI/physical world.
 */
static inline void probe_begin(holder_probe *p, const MFTL_Domain *dom)
{
  p->dom = dom;
  p->seen[0] = dom->rank;
  p->n_seen = 1;
  p->last = dom->rank;
  for (int k = 0; k < 3; ++k)
  {
    p->lo[k] = 0;
    p->hi[k] = 0;
  }
}

/**
 * @brief Adds one owner to the probe.
 *
 * @param[in,out] p     The probe.
 * @param         owner The rank owning some entity of this body.
 * @return #MFTL_FALSE as soon as the holder set provably does not fit inside one
 *         26-neighbourhood, #MFTL_TRUE while it still might.
 */
static bool probe_owner(holder_probe *p, int owner)
{
  if (owner == p->last) return MFTL_TRUE;
  p->last = owner;

  for (int i = 0; i < p->n_seen; ++i)
    if (p->seen[i] == owner) return MFTL_TRUE;

  const MFTL_Domain *const dom = p->dom;

  /* Statement 1: a holder this rank can SEE and cannot TALK TO on the graph.
   * Nothing further to learn -- the neighbourhood route would silently drop that
   * rank's contribution. */
  int nb = MFTL_ERROR;
  for (int nn = 0; nn < dom->num_neighbors; ++nn)
  {
    if (dom->neighbors[nn] != owner) continue;
    nb = nn;
    break;
  }
  if (nb == MFTL_ERROR) return MFTL_FALSE;

  MFTL_ASSERT(p->n_seen < 27 && "a rank has at most 26 distinct neighbours");
  p->seen[p->n_seen++] = owner;

  /* Statement 2: two holders one cell either side of this one are two cells
   * apart from each other, so they are not in each other's neighbourhood even
   * though both are in this rank's. */
  for (int k = 0; k < 3; ++k)
  {
    const int off = dom->neighbor_offset[nb][k];
    if (off < p->lo[k]) p->lo[k] = off;
    if (off > p->hi[k]) p->hi[k] = off;
    if (p->hi[k] - p->lo[k] > 1) return MFTL_FALSE;
  }

  return MFTL_TRUE;
}

bool MFTLEnsembleBodyHoldersFitTheGraph(const MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL) return MFTL_FALSE;

  const MFTL_Domain *const dom = &ens->domain;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    /* Owns nothing here, so no other holder can see this rank. See the banner. */
    if (obj->verts.n_local < 1 && obj->tris.n_local < 1) return MFTL_FALSE;

    holder_probe p;
    probe_begin(&p, dom);

    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
      if (!probe_owner(&p, obj->verts.rec[v].owner_proc)) return MFTL_FALSE;

    for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
      if (!probe_owner(&p, obj->tris.rec[t].owner_proc)) return MFTL_FALSE;
  }

  return MFTL_TRUE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Route A: the holders, over the 26-neighbour graph
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One `(body id, payload)` allgatherv over `comm_graph`.
 *
 * The half of the holder route that is the same whichever way the records are
 * then combined. It is two collectives and not one for the reason the ghost sync
 * is: only the holder of a body knows it holds it, so the lengths cannot be
 * predicted from the id space.
 *
 * @param      ens    The ensemble.
 * @param      values `[n_local_objects * n]` this rank's contribution.
 * @param      n      Values per body.
 * @param[out] recv   Receives the concatenated neighbour blocks. Caller frees.
 * @param[out] rcount `[num_neighbors]` doubles received per neighbour.
 * @param[out] rdispl `[num_neighbors]` offsets into @p recv.
 */
static void holder_gather(MFTL_Ensemble *ens, const double *values, int n, double **recv, int *rcount, int *rdispl)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int d = dom->num_neighbors;
  const int b = ens->n_local_objects;
  const int rec = n + 1; /* the body id rides in front of the payload */

  double *const send = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * rec, 1) * sizeof(double), "MFTLEnsembleReduceBodies send");
  for (int o = 0; o < b; ++o)
  {
    send[(size_t) o * rec] = (double) ens->local_objects[o]->global_id;
    memcpy(&send[(size_t) o * rec + 1], &values[(size_t) o * n], (size_t) n * sizeof(double));
  }

  const int scount = b * rec;
  MPI_Neighbor_allgather(&scount, 1, MPI_INT, rcount, 1, MPI_INT, dom->comm_graph);

  size_t total = 0;
  for (int i = 0; i < d; ++i)
  {
    rdispl[i] = (int) total;
    total += (size_t) rcount[i];
  }
  /* MPI counts and displacements are ints. Reaching this needs of the order of
   * 10^7 doubles arriving from 26 neighbours at once, i.e. millions of bodies
   * held on ONE rank -- far past where the per-body payload is the problem. */
  MFTL_ASSERT(total <= (size_t) 0x7fffffff && "MFTLEnsembleReduceBodies: neighbour payload exceeds an MPI int count");

  *recv = (double *) MFTLMallocOrDie(MFTL_MAX(total, (size_t) 1) * sizeof(double), "MFTLEnsembleReduceBodies receive");
  MPI_Neighbor_allgatherv(send, scount, MPI_DOUBLE, *recv, rcount, rdispl, MPI_DOUBLE, dom->comm_graph);

  g_stats.n_bytes += (long) (scount + total) * (long) sizeof(double);
  free(send);
}

/**
 * @brief The neighbour block indices in ASCENDING SOURCE-RANK order, self spliced in.
 *
 * ── WHY THE ORDER IS PART OF THE CONTRACT ────────────────────────────────
 *
 * `MFTL_Domain::neighbors` is in whatever order MPI reported it, which differs
 * between ranks, so folding the blocks in receive order would make two holders
 * of one body sum the same values in different orders -- and two holders
 * disagreeing in the last ulp about `quality` is two ranks entering different
 * collectives when a solver branches on it. At most 27 entries, so an insertion
 * sort is the whole algorithm.
 *
 * THIS RANK IS DROPPED FROM THE NEIGHBOUR BLOCKS, and that is not defensive
 * tidiness. domain_init_comm() skips the (0,0,0) OFFSET, not the own RANK: on an
 * axis with a single cut and periodicity every offset along it wraps back onto
 * this cell, so the 26-stencil legitimately resolves to this rank and comm_graph
 * carries a self-loop. An exchange does not care -- MPI delivers the block and
 * the data is right either way -- but a REDUCTION does: the self block would be
 * folded in a second time, and MIN and MAX would hide it while SUM silently
 * doubled. Grids of the shape {2,1,1} and {2,2,1} have the self-loop and {2,2,2}
 * and {3,3,3} do not, which is exactly the set of rank counts that failed.
 *
 * @param      dom   The MPI/physical world.
 * @param[out] order `[27]` block indices; the value `num_neighbors` stands for
 *                   this rank's own contribution.
 * @return How many entries were written.
 */
static int holder_source_order(const MFTL_Domain *dom, int order[27])
{
  const int d = dom->num_neighbors;
  MFTL_ASSERT(d <= 26 && "the 26-neighbour graph has at most 26 neighbours");

  int n_src = 0;
  for (int i = 0; i < d; ++i)
  {
    if (dom->neighbors[i] == dom->rank) continue;
    order[n_src++] = i;
  }
  order[n_src++] = d; /* the sentinel index standing for self */

  for (int i = 1; i < n_src; ++i)
  {
    const int key = order[i];
    const int key_rank = (key == d) ? dom->rank : dom->neighbors[key];
    int j = i - 1;
    while (j >= 0 && ((order[j] == d) ? dom->rank : dom->neighbors[order[j]]) > key_rank)
    {
      order[j + 1] = order[j];
      --j;
    }
    order[j + 1] = key;
  }
  return n_src;
}

/**
 * @brief One `(body id, payload)` exchange over `comm_graph`, combined locally.
 *
 * @param      ens    The ensemble.
 * @param[in,out] values `[n_local_objects * n]` contribution in, reduction out.
 * @param      n      Values per body.
 * @param      ops    `[n]` operators.
 * @return #MFTL_SUCCESS.
 */
static int reduce_over_holders(MFTL_Ensemble *ens, double *values, int n, const MFTL_BodyReduceOp *ops)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int d = dom->num_neighbors;
  const int b = ens->n_local_objects;
  const int rec = n + 1; /* the body id rides in front of the payload */

  /* This rank's contribution has to survive the identity fill below, because it
   * is folded back in at its own position in the rank order rather than being
   * the accumulator's starting value. That ordering is what makes the answer
   * bitwise identical on every holder -- see the file banner. */
  double *const mine = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * n, 1) * sizeof(double), "MFTLEnsembleReduceBodies own contribution");
  if (b > 0) memcpy(mine, values, (size_t) b * n * sizeof(double));

  int *const rcount = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleReduceBodies counts");
  int *const rdispl = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleReduceBodies displacements");
  double *recv = MFTL_NULL;

  holder_gather(ens, mine, n, &recv, rcount, rdispl);

  /* Ascending source rank, self spliced in -- see holder_source_order(). */
  int order[27];
  const int n_src = holder_source_order(dom, order);

  for (int o = 0; o < b; ++o)
    for (int j = 0; j < n; ++j) values[(size_t) o * n + j] = op_identity(ops[j]);

  for (int s = 0; s < n_src; ++s)
  {
    const int src = order[s];

    if (src == d)
    {
      for (int o = 0; o < b; ++o) op_combine(&values[(size_t) o * n], &mine[(size_t) o * n], n, ops);
      continue;
    }

    /* A MERGE, not a search per record: the sender's local list is sorted by
     * body id and so is this rank's, so one cursor over each is enough. That is
     * the third thing the directory's sort order buys (mftl_ensemble.h). */
    const double *const blk = &recv[rdispl[src]];
    const int n_rec = rcount[src] / rec;
    int slot = 0;

    for (int k = 0; k < n_rec; ++k)
    {
      const int gid = (int) blk[(size_t) k * rec];
      while (slot < b && ens->local_objects[slot]->global_id < gid) ++slot;
      if (slot >= b) break; /* the rest of this block is about bodies not held here */
      if (ens->local_objects[slot]->global_id != gid) continue;
      op_combine(&values[(size_t) slot * n], &blk[(size_t) k * rec + 1], n, ops);
    }
  }

  free(recv);
  free(rdispl);
  free(rcount);
  free(mine);

  g_stats.n_holders++;
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The exclusive prefix sum, on the same records
 *
 * A REDUCTION IS NOT ENOUGH FOR HANDING OUT IDS, and that is the whole reason
 * this exists next to MFTLEnsembleReduceBodies() rather than inside it. Two
 * ranks that both split an edge of the same body each produce a new vertex, and
 * the two need DIFFERENT global ids -- so each holder needs to know how many
 * entities the holders BEFORE it created, which is an exclusive scan and not a
 * combination. `MPI_Exscan` gives it over `comm_world`; over a body's holders it
 * has to be arranged, and arranging it is exactly one more pass over the records
 * the reduction already gathers.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Per-body exclusive prefix sum over the holders, on the 26-neighbour graph.
 *
 * @param      ens    The ensemble.
 * @param[in,out] values `[n_local_objects * n]` contribution in, prefix out.
 * @param[out] total  `[n_local_objects * n]` sum over every holder, or MFTL_NULL.
 * @param      n      Values per body.
 * @return #MFTL_SUCCESS.
 */
static int exscan_over_holders(MFTL_Ensemble *ens, double *values, double *total, int n)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int d = dom->num_neighbors;
  const int b = ens->n_local_objects;
  const int rec = n + 1;

  double *const mine = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * n, 1) * sizeof(double), "MFTLEnsembleExscanBodies own contribution");
  if (b > 0) memcpy(mine, values, (size_t) b * n * sizeof(double));

  int *const rcount = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleExscanBodies counts");
  int *const rdispl = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleExscanBodies displacements");
  double *recv = MFTL_NULL;

  holder_gather(ens, mine, n, &recv, rcount, rdispl);

  int order[27];
  const int n_src = holder_source_order(dom, order);

  double *const sum = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b * n, 1), sizeof(double), "MFTLEnsembleExscanBodies running sum");

  for (int o = 0; o < b * n; ++o) values[o] = 0.0;

  /* One sweep in ascending source rank. Everything folded in BEFORE this rank's
   * own position is the prefix; everything, including this rank, is the total.
   * Splitting them costs one extra array and no extra pass. */
  bool past_self = MFTL_FALSE;

  for (int s = 0; s < n_src; ++s)
  {
    const int src = order[s];

    if (src == d)
    {
      /* The prefix is frozen here: this rank contributes to the total but not to
       * its own offset, which is what makes the ranges disjoint rather than
       * overlapping by this rank's own count. */
      if (b > 0) memcpy(values, sum, (size_t) b * n * sizeof(double));
      for (int o = 0; o < b * n; ++o) sum[o] += mine[o];
      past_self = MFTL_TRUE;
      continue;
    }

    /* A MERGE, not a search per record: the sender's local list is sorted by
     * body id and so is this rank's, so one cursor over each is enough. */
    const double *const blk = &recv[rdispl[src]];
    const int n_rec = rcount[src] / rec;
    int slot = 0;

    for (int k = 0; k < n_rec; ++k)
    {
      const int gid = (int) blk[(size_t) k * rec];
      while (slot < b && ens->local_objects[slot]->global_id < gid) ++slot;
      if (slot >= b) break;
      if (ens->local_objects[slot]->global_id != gid) continue;
      for (int j = 0; j < n; ++j) sum[(size_t) slot * n + j] += blk[(size_t) k * rec + 1 + j];
    }
  }

  MFTL_ASSERT(past_self && "MFTLEnsembleExscanBodies: this rank is missing from its own source order");
  (void) past_self;

  if (total != MFTL_NULL && b > 0) memcpy(total, sum, (size_t) b * n * sizeof(double));

  free(sum);
  free(recv);
  free(rdispl);
  free(rcount);
  free(mine);

  g_stats.n_holders++;
  return MFTL_SUCCESS;
}

/**
 * @brief The `MPI_Exscan` + `MPI_Allreduce` this function exists to stop taking.
 *
 * @param      ens    The ensemble.
 * @param[in,out] values `[n_local_objects * n]` contribution in, prefix out.
 * @param[out] total  `[n_local_objects * n]` sum over every rank, or MFTL_NULL.
 * @param      n      Values per body.
 * @return #MFTL_SUCCESS.
 */
static int exscan_over_comm_world(MFTL_Ensemble *ens, double *values, double *total, int n)
{
  const int nb = ens->n_bodies_global;
  const int b = ens->n_local_objects;
  const size_t block = (size_t) nb * (size_t) n;

  double *const buf = (double *) MFTLCallocOrDie(3 * MFTL_MAX(block, (size_t) 1), sizeof(double), "MFTLEnsembleExscanBodies staging");
  double *const loc = buf;
  double *const pre = buf + block;
  double *const tot = buf + 2 * block;

  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    MFTL_ASSERT(g >= 0 && g < nb && "MFTLEnsembleExscanBodies: body id outside the id space");
    for (int j = 0; j < n; ++j) loc[(size_t) g * n + j] = values[(size_t) o * n + j];
  }

  /* Two collectives rather than one because they are different reductions of the
   * same input, and they are independent: both are posted before either is
   * waited on, so the two O(log P) latencies overlap instead of adding. */
  MPI_Request req[2];
  MPI_Iexscan(loc, pre, (int) block, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world, &req[0]);
  MPI_Iallreduce(loc, tot, (int) block, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

  g_stats.n_bytes += 2 * (long) block * (long) sizeof(double);

  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    for (int j = 0; j < n; ++j)
    {
      values[(size_t) o * n + j] = pre[(size_t) g * n + j];
      if (total != MFTL_NULL) total[(size_t) o * n + j] = tot[(size_t) g * n + j];
    }
  }

  free(buf);

  g_stats.n_global++;
  return MFTL_SUCCESS;
}

int MFTLEnsembleExscanBodies(MFTL_Ensemble *ens, double *values, double *total, int n_per_body, MFTL_BodyReduceScope scope)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL || n_per_body < 0) return MFTL_ERROR;
  if (n_per_body == 0) return MFTL_SUCCESS; /* identical on every rank, so no rank is left in a collective */
  MFTL_ASSERT((values != MFTL_NULL) || ens->n_local_objects == 0);

  const int b = ens->n_local_objects;

  /* One rank IS the holder set of every body, so the prefix is zero and the
   * total is this rank's own contribution. Taken before anything collective, so
   * a serial run makes no MPI call at all. */
  if (ens->domain.nproc < 2)
  {
    if (total != MFTL_NULL && b > 0) memcpy(total, values, (size_t) b * n_per_body * sizeof(double));
    for (int i = 0; i < b * n_per_body; ++i) values[i] = 0.0;
    return MFTL_SUCCESS;
  }

  bool use_holders;
  if (scope == MFTL_BODY_REDUCE_SCOPE_AUTO)
  {
    int mine = MFTLEnsembleBodyHoldersFitTheGraph(ens) ? 1 : 0;
    int agreed = 0;
    MPI_Allreduce(&mine, &agreed, 1, MPI_INT, MPI_LAND, ens->domain.comm_world);
    use_holders = (agreed != 0);
  }
  else
  {
    use_holders = (scope == MFTL_BODY_REDUCE_SCOPE_HOLDERS);
  }

  return use_holders ? exscan_over_holders(ens, values, total, n_per_body) : exscan_over_comm_world(ens, values, total, n_per_body);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Route B: comm_world, staged by body id
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The `MPI_Allreduce` this function exists to stop taking, kept exact.
 *
 * Correct for any ensemble, and `O(bodies in the ensemble)` per rank in both
 * payload and staging memory. The slots are grouped by OPERATOR so that each
 * group is one contiguous range and the call count is the number of DISTINCT
 * operators -- at most three, whatever `n` is.
 *
 * @param ens        The ensemble.
 * @param[in,out] values [n_local_objects * n] contribution in, reduction out.
 * @param n          Values per body.
 * @param ops        [n] operators.
 * @return #MFTL_SUCCESS.
 */
static int reduce_over_comm_world(MFTL_Ensemble *ens, double *values, int n, const MFTL_BodyReduceOp *ops)
{
  const int nb = ens->n_bodies_global;
  const int b = ens->n_local_objects;

  /* Group the slots: `col[j]` is where source slot j lands, `base[op]` is where
   * that operator's group starts and `cnt[op]` is how wide it is. */
  int cnt[3] = {0, 0, 0};
  for (int j = 0; j < n; ++j)
  {
    MFTL_ASSERT((int) ops[j] >= 0 && (int) ops[j] < 3 && "MFTLEnsembleReduceBodies: unknown operator");
    cnt[(int) ops[j]]++;
  }

  int base[3];
  base[0] = 0;
  base[1] = cnt[0];
  base[2] = cnt[0] + cnt[1];

  int fill[3] = {base[0], base[1], base[2]};
  int *const col = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(int), "MFTLEnsembleReduceBodies columns");
  for (int j = 0; j < n; ++j) col[j] = fill[(int) ops[j]]++;

  /* [local | global], each laid out group-major so that group `op` is the
   * contiguous range [base[op]*nb, (base[op]+cnt[op])*nb). A body-major layout
   * would put three disjoint segments in every row and cost one Allreduce per
   * body. */
  const size_t block = (size_t) nb * (size_t) n;
  double *const buf = (double *) MFTLMallocOrDie(2 * MFTL_MAX(block, (size_t) 1) * sizeof(double), "MFTLEnsembleReduceBodies staging");
  double *const loc = buf;
  double *const glo = buf + block;

  for (int op = 0; op < 3; ++op)
  {
    const double id = op_identity((MFTL_BodyReduceOp) op);
    double *const grp = loc + (size_t) base[op] * (size_t) nb;
    for (size_t i = 0; i < (size_t) cnt[op] * (size_t) nb; ++i) grp[i] = id;
  }

  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    MFTL_ASSERT(g >= 0 && g < nb && "MFTLEnsembleReduceBodies: body id outside the id space");
    for (int j = 0; j < n; ++j)
    {
      loc[(size_t) col[j] * (size_t) nb + (size_t) g] = values[(size_t) o * n + j];
    }
  }

  static const MPI_Op mpi_op[3] = {MPI_MIN, MPI_MAX, MPI_SUM};
  MPI_Request req[3];
  int n_req = 0;
  for (int op = 0; op < 3; ++op)
  {
    if (cnt[op] == 0) continue;
    const size_t off = (size_t) base[op] * (size_t) nb;
    MPI_Iallreduce(loc + off, glo + off, cnt[op] * nb, MPI_DOUBLE, mpi_op[op], ens->domain.comm_world, &req[n_req++]);
    g_stats.n_bytes += (long) cnt[op] * (long) nb * (long) sizeof(double);
  }
  MPI_Waitall(n_req, req, MPI_STATUSES_IGNORE);

  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    for (int j = 0; j < n; ++j)
    {
      values[(size_t) o * n + j] = glo[(size_t) col[j] * (size_t) nb + (size_t) g];
    }
  }

  free(buf);
  free(col);

  g_stats.n_global++;
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The entry point
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLEnsembleReduceBodies(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, MFTL_BodyReduceScope scope)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL || n_per_body < 0) return MFTL_ERROR;
  if (n_per_body == 0) return MFTL_SUCCESS; /* identical on every rank, so no rank is left in a collective */
  MFTL_ASSERT(ops != MFTL_NULL);
  if (ops == MFTL_NULL) return MFTL_ERROR;
  MFTL_ASSERT((values != MFTL_NULL) || ens->n_local_objects == 0);

  /* One rank IS the holder set of every body, so both routes are the identity
   * and the values are already the answer. Taken before anything collective so
   * that a serial run makes no MPI call at all. */
  if (ens->domain.nproc < 2) return MFTL_SUCCESS;

  bool use_holders;

  if (scope == MFTL_BODY_REDUCE_SCOPE_AUTO)
  {
    /* THE ROUTE IS A COLLECTIVE DECISION even though the test is local: the two
     * branches call different communicators, so a rank concluding on its own
     * evidence would be alone in whichever it picked. The property is a
     * conjunction over ranks, so MPI_LAND of the local answers IS the answer.
     *
     * One int, so this is latency and not volume -- which is the whole trade,
     * against the `O(bodies)` payload it decides whether to send. A caller that
     * already knows says so and pays neither. */
    int mine = MFTLEnsembleBodyHoldersFitTheGraph(ens) ? 1 : 0;
    int agreed = 0;
    MPI_Allreduce(&mine, &agreed, 1, MPI_INT, MPI_LAND, ens->domain.comm_world);
    use_holders = (agreed != 0);
  }
  else
  {
    use_holders = (scope == MFTL_BODY_REDUCE_SCOPE_HOLDERS);
  }

  return use_holders ? reduce_over_holders(ens, values, n_per_body, ops) : reduce_over_comm_world(ens, values, n_per_body, ops);
}
