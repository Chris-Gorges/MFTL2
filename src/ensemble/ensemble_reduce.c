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
 *   3. THE ROUTE IS DECIDED AFTER THE EXCHANGE, NOT BEFORE IT. What a rank can
 *      read in its own pools before the exchange is necessary and NOT
 *      sufficient; what makes the holder route exact is only visible once every
 *      holder has heard from its neighbours. Both verdicts ride on one int, so
 *      the route costs the same single MPI_Allreduce it always did.
 */

#include "mftl_ensemble.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
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
 * @param         ops  [n] operators, or #MFTL_NULL for a plain sum.
 */
static inline void op_combine(double *acc, const double *in, int n, const MFTL_BodyReduceOp *ops)
{
  for (int j = 0; j < n; ++j)
  {
    switch ((ops != MFTL_NULL) ? ops[j] : MFTL_BODY_REDUCE_SUM)
    {
      case MFTL_BODY_REDUCE_MIN: acc[j] = fmin(acc[j], in[j]); break;
      case MFTL_BODY_REDUCE_MAX: acc[j] = fmax(acc[j], in[j]); break;
      default: acc[j] += in[j]; break;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Before the exchange: what a rank's own pools can rule out
 *
 * Statements 1 and 2 of the derivation on MFTLEnsembleReduceBodies(), read off
 * the GHOST SUFFIX of each pool. Every entity in [0, n_local) is owned here by
 * the ownership-prefix invariant, so it contributes the offset (0,0,0) and
 * nothing else; the halo is what names the other holders. On a body this rank
 * holds whole -- the common case in the regime this route exists for -- there is
 * no ghost suffix and the scan does nothing at all.
 *
 * What it establishes is that a body's holders are CONNECTED on the graph, and
 * that is all. A bent tube whose holders form a chain around a corner passes it
 * on every rank while its two ends are two cells apart and never hear from each
 * other; the certificate after the exchange is what closes that.
 *
 * ── A HOLDER THAT OWNS NOTHING IS DISCONNECTED, SO IT VETOES ───────────────
 *
 * The derivation starts from "a rank holds an entity of a body only if it OWNS
 * one", which is what links every holder to another through an entity they
 * share. A rank holding an EMPTY object shares nothing with anyone, so nothing
 * connects it to the others and the certificate cannot vouch for it.
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

  /* Statement 2: a holder this rank can SEE and cannot TALK TO on the graph.
   * Nothing further to learn -- the neighbourhood route would silently drop that
   * rank's contribution. */
  const int nb = MFTLDomainNeighborIndex(dom, owner);
  if (nb == MFTL_ERROR) return MFTL_FALSE;

  MFTL_ASSERT(p->n_seen < 27 && "a rank has at most 26 distinct neighbours");
  p->seen[p->n_seen++] = owner;

  /* Two visible holders one cell either side of this one are two cells apart
   * from each other. The certificate after the exchange would refuse the same
   * pair, since both are neighbours that hold the body; refusing it here costs
   * nothing and keeps this predicate's answer meaningful on its own. */
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

    /* Statement 1: owns nothing here, so nothing connects this rank to the other
     * holders. See the banner. */
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
 * The holder route's records, over the 26-neighbour graph
 *
 * The reduction and the exclusive scan below gather the SAME records -- one
 * `(body id, payload)` record per body this rank holds, published one hop along
 * `comm_graph` -- and walk them in the SAME order. Only what they do with each
 * record differs, so the gathering, the ordering and the merge are shared.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Everything the holder route gathers before it combines anything. */
typedef struct
{
  double *mine;  ///< [b * n] this rank's contribution, kept past the in-place overwrite.
  double *recv;  ///< The concatenated neighbour blocks.
  int *rcount;  ///< [num_neighbors] doubles received per neighbour.
  int *rdispl;  ///< [num_neighbors] offset of each neighbour's block in `recv`.
  int order[27];  ///< Block indices in ascending source rank; `num_neighbors` stands for this rank.
  int n_src;  ///< Entries in `order`.
} holder_records;

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
 * @brief One `(body id, payload)` allgatherv over `comm_graph`, plus the order to
 *        fold the blocks in.
 *
 * Two collectives and not one for the reason the ghost sync takes two: only the
 * holder of a body knows it holds it, so the lengths cannot be predicted from
 * the id space. Release with holder_records_free().
 *
 * @param      ens    The ensemble.
 * @param      values `[n_local_objects * n]` this rank's contribution.
 * @param      n      Values per body.
 * @param[out] h      The gathered records.
 */
static void holder_records_gather(MFTL_Ensemble *ens, const double *values, int n, holder_records *h)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int d = dom->num_neighbors;
  const int b = ens->n_local_objects;
  const int rec = n + 1; /* the body id rides in front of the payload */

  /* The contribution has to survive the caller overwriting `values`, because it
   * is folded back in at this rank's own position in the rank order rather than
   * being the accumulator's starting value. That ordering is what makes the
   * answer bitwise identical on every holder -- see the file banner. */
  h->mine = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * n, 1) * sizeof(double), "MFTLEnsembleReduceBodies own contribution");
  if (b > 0) memcpy(h->mine, values, (size_t) b * n * sizeof(double));

  double *const send = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * rec, 1) * sizeof(double), "MFTLEnsembleReduceBodies send");
  for (int o = 0; o < b; ++o)
  {
    send[(size_t) o * rec] = (double) ens->local_objects[o]->global_id;
    memcpy(&send[(size_t) o * rec + 1], &h->mine[(size_t) o * n], (size_t) n * sizeof(double));
  }

  h->rcount = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleReduceBodies counts");
  h->rdispl = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(d, 1) * sizeof(int), "MFTLEnsembleReduceBodies displacements");

  const int scount = b * rec;
  int n_recv = 0;
  h->recv = MFTLNeighborAllgatherDoubles(dom, send, scount, h->rcount, h->rdispl, &n_recv);
  free(send);

  g_stats.n_bytes += (long) (scount + n_recv) * (long) sizeof(double);

  h->n_src = holder_source_order(dom, h->order);
}

/**
 * @brief Releases what holder_records_gather() allocated.
 * @param h The gathered records.
 */
static void holder_records_free(holder_records *h)
{
  free(h->recv);
  free(h->rdispl);
  free(h->rcount);
  free(h->mine);
}

/**
 * @brief Folds one neighbour's records into @p acc, for the bodies held here.
 *
 * A MERGE, not a search per record: the sender's local list is sorted by body id
 * and so is this rank's, so one cursor over each is enough. That is the third
 * thing the directory's sort order buys (mftl_ensemble.h).
 *
 * @param         ens The ensemble.
 * @param         blk The neighbour's block: `(body id, n values)` records.
 * @param         n_rec Records in @p blk.
 * @param         n   Values per body.
 * @param[in,out] acc `[n_local_objects * n]` accumulator, by local slot.
 * @param         ops `[n]` operators, or #MFTL_NULL for a plain sum.
 */
static void fold_neighbour_block(const MFTL_Ensemble *ens, const double *blk, int n_rec, int n, double *acc, const MFTL_BodyReduceOp *ops)
{
  const int b = ens->n_local_objects;
  const int rec = n + 1;
  int slot = 0;

  for (int k = 0; k < n_rec; ++k)
  {
    const int gid = (int) blk[(size_t) k * rec];
    while (slot < b && ens->local_objects[slot]->global_id < gid) ++slot;
    if (slot >= b) break; /* the rest of this block is about bodies not held here */
    if (ens->local_objects[slot]->global_id != gid) continue;
    op_combine(&acc[(size_t) slot * n], &blk[(size_t) k * rec + 1], n, ops);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * After the exchange: the certificate
 *
 * Statement 3 of the derivation on MFTLEnsembleReduceBodies(). Every holder of
 * a body publishes a record for it, so once the records are in, a rank knows
 * EVERY holder of every body it holds that is a graph neighbour -- not only the
 * ones its halo happens to name. If those, with this rank, span at most one cell
 * per axis on every holder, a connected holder set is a clique and the fold is
 * exact; the proof is on MFTLEnsembleReduceBodies().
 *
 * The test is on the records alone: no pool is read, so it is equally valid for
 * redistribution's vote, taken after `owner_proc` has been rewritten.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Whether, for every body held here, this rank and the neighbours that
 *        published a record for it span at most one cell per axis.
 *
 * The offsets are `MFTL_Domain::neighbor_offset`, i.e. the first of possibly
 * several aliases on a thin periodic axis. That choice cannot make the test
 * unsound -- any alias of a rank locates its cell -- and it cannot make it
 * spuriously strict either: on an axis with at most two cuts every first alias
 * is -1 or 0, and on one with three or more cuts every alias is unique.
 *
 * @param ens The ensemble.
 * @param h   The gathered records.
 * @param n   Values per body.
 * @return #MFTL_TRUE if no held body's neighbouring holders span two cells.
 */
static bool holders_span_one_cell(const MFTL_Ensemble *ens, const holder_records *h, int n)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int d = dom->num_neighbors;
  const int b = ens->n_local_objects;
  const int rec = n + 1;
  if (b == 0) return MFTL_TRUE;

  /* [b][6]: lo[3] then hi[3], starting at this rank's own offset (0,0,0). */
  signed char *const win = (signed char *) MFTLCallocOrDie((size_t) b * 6, sizeof(signed char), "MFTLEnsembleReduceBodies offset windows");
  bool ok = MFTL_TRUE;

  for (int s = 0; s < h->n_src && ok; ++s)
  {
    const int src = h->order[s];
    if (src == d) continue; /* this rank: already the (0,0,0) every window starts from */

    const signed char *const off = dom->neighbor_offset[src];
    const double *const blk = &h->recv[h->rdispl[src]];
    const int n_rec = h->rcount[src] / rec;

    /* The same two-cursor merge fold_neighbour_block() walks. */
    int slot = 0;
    for (int k = 0; k < n_rec && ok; ++k)
    {
      const int gid = (int) blk[(size_t) k * rec];
      while (slot < b && ens->local_objects[slot]->global_id < gid) ++slot;
      if (slot >= b) break;
      if (ens->local_objects[slot]->global_id != gid) continue;

      signed char *const lo = &win[(size_t) slot * 6];
      signed char *const hi = lo + 3;
      for (int a = 0; a < 3; ++a)
      {
        if (off[a] < lo[a]) lo[a] = off[a];
        if (off[a] > hi[a]) hi[a] = off[a];
        if (hi[a] - lo[a] > 1) ok = MFTL_FALSE;
      }
    }
  }

  free(win);
  return ok;
}

/**
 * @brief Agrees, collectively, whether the gathered records may be folded.
 *
 * THE ROUTE IS A COLLECTIVE DECISION even though both tests are local: the
 * fallback runs on `comm_world`, so a rank concluding on its own evidence would
 * be alone in it. The property is a conjunction over ranks, so MPI_LAND of the
 * local answers IS the answer. One int, which is latency and not volume -- the
 * same single collective the route cost when it was decided before the exchange,
 * now taken where it is sound.
 *
 * @param ens    The ensemble.
 * @param h      The gathered records.
 * @param n      Values per body.
 * @param pre_ok This rank's MFTLEnsembleBodyHoldersFitTheGraph() verdict.
 * @return `true` on every rank if the holder route is exact, `false` on every
 *         rank otherwise.
 */
static bool holder_route_certified(MFTL_Ensemble *ens, const holder_records *h, int n, bool pre_ok)
{
  int mine = (pre_ok && holders_span_one_cell(ens, h, n)) ? 1 : 0;
  int agreed = 0;
  MPI_Allreduce(&mine, &agreed, 1, MPI_INT, MPI_LAND, ens->domain.comm_world);
  return agreed != 0;
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

  /* MPI leaves the exclusive-scan receive buffer UNDEFINED on rank 0; the calloc
   * above is no guarantee after the call. These prefixes become global-id
   * offsets, so an implementation that writes rank 0's buffer would hand out
   * overlapping ids. Rank 0's prefix is zero by definition. */
  if (ens->domain.rank == 0) memset(pre, 0, block * sizeof(double));

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

/* ═══════════════════════════════════════════════════════════════════════════
 * Route A: the holders, with the certificate and the fallback
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The per-body reduction: over the holders when certified, over
 *        `comm_world` otherwise.
 *
 * @param      ens    The ensemble.
 * @param[in,out] values `[n_local_objects * n]` contribution in, reduction out.
 * @param      n      Values per body.
 * @param      ops    `[n]` operators.
 * @param      pre_ok This rank's MFTLEnsembleBodyHoldersFitTheGraph() verdict.
 * @return #MFTL_SUCCESS.
 */
static int reduce_bodies(MFTL_Ensemble *ens, double *values, int n, const MFTL_BodyReduceOp *ops, bool pre_ok)
{
  const int d = ens->domain.num_neighbors;
  const int b = ens->n_local_objects;

  holder_records h;
  holder_records_gather(ens, values, n, &h);

  /* Nothing has been written to `values` yet, so a refusal needs no restore. */
  if (!holder_route_certified(ens, &h, n, pre_ok))
  {
    holder_records_free(&h);
    return reduce_over_comm_world(ens, values, n, ops);
  }

  for (int o = 0; o < b; ++o)
    for (int j = 0; j < n; ++j) values[(size_t) o * n + j] = op_identity(ops[j]);

  for (int s = 0; s < h.n_src; ++s)
  {
    const int src = h.order[s];
    if (src == d)
    {
      for (int o = 0; o < b; ++o) op_combine(&values[(size_t) o * n], &h.mine[(size_t) o * n], n, ops);
      continue;
    }
    fold_neighbour_block(ens, &h.recv[h.rdispl[src]], h.rcount[src] / (n + 1), n, values, ops);
  }

  holder_records_free(&h);
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
 * @brief Per-body exclusive prefix sum: over the holders when certified, over
 *        `comm_world` otherwise.
 *
 * @param      ens    The ensemble.
 * @param[in,out] values `[n_local_objects * n]` contribution in, prefix out.
 * @param[out] total  `[n_local_objects * n]` sum over every holder, or MFTL_NULL.
 * @param      n      Values per body.
 * @param      pre_ok This rank's MFTLEnsembleBodyHoldersFitTheGraph() verdict.
 * @return #MFTL_SUCCESS.
 */
static int exscan_bodies(MFTL_Ensemble *ens, double *values, double *total, int n, bool pre_ok)
{
  const int d = ens->domain.num_neighbors;
  const int b = ens->n_local_objects;

  holder_records h;
  holder_records_gather(ens, values, n, &h);

  if (!holder_route_certified(ens, &h, n, pre_ok))
  {
    holder_records_free(&h);
    return exscan_over_comm_world(ens, values, total, n);
  }

  double *const sum = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b * n, 1), sizeof(double), "MFTLEnsembleExscanBodies running sum");

  for (int o = 0; o < b * n; ++o) values[o] = 0.0;

  /* One sweep in ascending source rank. Everything folded in BEFORE this rank's
   * own position is the prefix; everything, including this rank, is the total.
   * Splitting them costs one extra array and no extra pass. */
  bool past_self = MFTL_FALSE;

  for (int s = 0; s < h.n_src; ++s)
  {
    const int src = h.order[s];

    if (src == d)
    {
      /* The prefix is frozen here: this rank contributes to the total but not to
       * its own offset, which is what makes the ranges disjoint rather than
       * overlapping by this rank's own count. */
      if (b > 0) memcpy(values, sum, (size_t) b * n * sizeof(double));
      for (int o = 0; o < b * n; ++o) sum[o] += h.mine[o];
      past_self = MFTL_TRUE;
      continue;
    }

    fold_neighbour_block(ens, &h.recv[h.rdispl[src]], h.rcount[src] / (n + 1), n, sum, MFTL_NULL);
  }

  MFTL_ASSERT(past_self && "MFTLEnsembleExscanBodies: this rank is missing from its own source order");
  (void) past_self;

  if (total != MFTL_NULL && b > 0) memcpy(total, sum, (size_t) b * n * sizeof(double));

  free(sum);
  holder_records_free(&h);

  g_stats.n_holders++;
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The entry points
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The argument checks both reduction entry points share.
 *
 * Everything that ends the call early is decided identically on every rank --
 * an argument that must be the same everywhere, or the process count -- so an
 * early return leaves no rank alone in a collective.
 *
 * @param      ens    The ensemble.
 * @param      values The caller's buffer.
 * @param      n      Values per body.
 * @param      ops    `[n]` operators.
 * @param[out] status What the entry point returns when this returns `true`.
 * @return `true` if there is nothing to reduce, `false` if the reduction has to run.
 */
static bool reduce_is_trivial(const MFTL_Ensemble *ens, const double *values, int n, const MFTL_BodyReduceOp *ops, int *status)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(n <= 0 || ops != MFTL_NULL);
  *status = MFTL_ERROR;
  if (ens == MFTL_NULL || n < 0 || (n > 0 && ops == MFTL_NULL)) return MFTL_TRUE;

  *status = MFTL_SUCCESS;
  if (n == 0) return MFTL_TRUE;
  MFTL_ASSERT((values != MFTL_NULL) || ens->n_local_objects == 0);
  (void) values;

  /* One rank IS the holder set of every body, so both routes are the identity
   * and the values are already the answer. Taken before anything collective so
   * that a serial run makes no MPI call at all. */
  return ens->domain.nproc < 2;
}

int MFTLEnsembleReduceBodiesPrescanned(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, bool holders_fit)
{
  int status;
  if (reduce_is_trivial(ens, values, n_per_body, ops, &status)) return status;
  return reduce_bodies(ens, values, n_per_body, ops, holders_fit);
}

int MFTLEnsembleReduceBodies(MFTL_Ensemble *ens, double *values, int n_per_body, const MFTL_BodyReduceOp *ops, MFTL_BodyReduceScope scope)
{
  int status;
  if (reduce_is_trivial(ens, values, n_per_body, ops, &status)) return status;
  if (scope == MFTL_BODY_REDUCE_SCOPE_GLOBAL) return reduce_over_comm_world(ens, values, n_per_body, ops);
  return reduce_bodies(ens, values, n_per_body, ops, MFTLEnsembleBodyHoldersFitTheGraph(ens));
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

  if (scope == MFTL_BODY_REDUCE_SCOPE_GLOBAL) return exscan_over_comm_world(ens, values, total, n_per_body);
  return exscan_bodies(ens, values, total, n_per_body, MFTLEnsembleBodyHoldersFitTheGraph(ens));
}
