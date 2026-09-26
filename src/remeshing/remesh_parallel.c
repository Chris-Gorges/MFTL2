/**
 * @file remesh_parallel.c
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief Distributed remeshing: the shift/freeze phase schedule.
 *
 * ## THE PROBLEM, AND THE ONE IDEA THAT REMOVES IT
 *
 * An edge whose two endpoints are owned by different ranks cannot be collapsed
 * by either of them: each holds half of the one-ring that the link condition is
 * a statement about, and each would have to rewrite records the other owns. The
 * obvious answer is a protocol -- lock the edge, ask the neighbour, agree, act --
 * and it is the wrong one. It puts a round trip inside the innermost loop of a
 * remesher, and it makes the RESULT depend on message arrival order, so the same
 * mesh remeshes differently on the same hardware twice in a row.
 *
 * The idea that removes the problem instead of managing it:
 *
 *     DON'T MOVE THE MESH TO SUIT THE CUT PLANES. MOVE THE CUT PLANES.
 *
 * Run the untouched serial kernel on the INTERIOR of each rank's share, where
 * the whole modified neighbourhood is exclusively that rank's; refuse every
 * operation whose footprint reaches the partition boundary; then MOVE THE PLANES
 * BY HALF A BOX and run it again. What was on the boundary is now in the middle
 * of somebody's share, and gets its turn. Nothing is negotiated because nothing
 * overlaps, and the answer does not depend on timing because no two ranks ever
 * touch the same entity.
 *
 * ## THE GEOMETRIC GUARANTEE, AND WHAT IT COSTS
 *
 * Let delta be the geometric half-width of the frozen band and h the rank box
 * width. A point frozen in the canonical placement is at distance <= delta from
 * a plane, hence at distance >= h/2 - delta from the nearest plane of the
 * half-shifted placement. It is interior there iff
 *
 *     h/2 - delta > delta   <=>   delta < h/4   <=>   h > 4 delta
 *
 * which is the hard precondition of the whole scheme, checked in
 * MFTLRemeshEnsembleParallel and reported rather than assumed.
 *
 * That is the PER-AXIS statement. In 3D an entity can sit near planes of one,
 * two or three axes at once, and the diagonal shift (1,1,1) only rescues the
 * first case:
 *
 *   SLAB   near 1 axis    measure O(delta/h)     2 phases: (0,0,0), (1,1,1)
 *   TUBE   near 2 axes    measure O((delta/h)^2) 4 phases: the even-parity set
 *   CORNER near 3 axes    measure O((delta/h)^3) 8 phases: all of {0,h/2}^3
 *
 * because an entity near the x and y planes but not z is still near the shifted
 * x and y planes after (1,1,1) -- both moved, and it was between them. Only a
 * shift that moves ONE of the two rescues it, which is what the odd-parity
 * phases are for. The table below is ordered so that the phases that matter most
 * run first: two phases handle all but O((delta/h)^2) of the mesh, and phases
 * 3-8 typically find a handful of entities each and exit almost immediately.
 *
 * ## "FROZEN" IS DERIVED, NOT COMMUNICATED, AND IT IS EXACT
 *
 * The frozen set is defined by an equivalence rather than by a distance:
 *
 *     an owned entity is frozen  <=>  some other rank holds a copy of it
 *
 * and this rank can evaluate the right-hand side ALONE, because it is the same
 * rule the redistribution used when it decided what to send. An owned triangle
 * is sent to the owner of each of its three vertices and of each of its three
 * edge neighbours (src/communication/redistribute.c, collect_destinations), so
 *
 *     triangle t is replicated  <=>  a corner of t is a ghost
 *                                    OR an edge neighbour of t is a ghost
 *
 * and with ownership as a prefix, "is a ghost" is `index >= n_local`. The
 * classification is therefore two streaming reads (`tris.verts`,
 * `tris.neighbors`, 12 bytes each per triangle) and six integer compares, with
 * no distance test, no tolerance, and no message. See MFTLRemeshMarkFrozen.
 *
 * Being exact rather than conservative is what makes the DILATION the plan
 * called for unnecessary, and the argument is short enough to state in full:
 *
 *   1. A vertex is frozen if ANY incident triangle is frozen. So an unfrozen
 *      vertex has every incident triangle unfrozen -- owned, and replicated
 *      nowhere.
 *   2. An unfrozen triangle has no ghost corner, so all three of its corners
 *      are OWNED. Combined with 1: every vertex an operator reads while judging
 *      an operation on unfrozen entities is owned by this rank.
 *   3. The redistribution halo is VERTEX-COMPLETE (inv_halo_is_vertex_complete),
 *      so every triangle incident to an owned vertex is present locally. With 2,
 *      every fan walk an approval predicate performs -- the link condition, the
 *      wing valences, the stretch bound, the fold-over test -- traverses a
 *      COMPLETE fan and gets the same answer it would get on the whole mesh.
 *
 * The converse direction is what makes the phase safe rather than merely
 * self-consistent, and it is worth stating separately because it is what buys
 * the missing communication:
 *
 *   4. Every ghost in this rank's pool is frozen ON ITS OWNER. A ghost triangle
 *      is here because this rank owns one of its corners or one of its edge
 *      neighbours -- which, seen from the owner, IS the definition of replicated.
 *      So no neighbour may rewrite it, delete it, or move any of its corners.
 *
 * Therefore this rank's ghost copies are still EXACT when the sweeps finish, and
 * the phase needs no ghost position sync before the next migration -- the one
 * per-phase collective the design might have been expected to pay.
 *
 * ## GLOBAL IDS: SPARSE DURING THE PASS, DENSE AT THE END
 *
 * Splits hand out ids and collapses retire them, and in parallel the ids also
 * have to be unique across ranks. The rank-encoded 64-bit id that suggests
 * itself is the wrong answer HERE, whatever its merits elsewhere: redistribution
 * and the ghost sync size their arrival lookup tables from `next_*_gid`
 * (redistribute.c, ghost_sync.c), so they need ids DENSE, and a rank-encoded id
 * is sparse by construction. It would turn a few hundred kilobytes per body into
 * a table indexed by rank << 40.
 *
 * So ids stay 32-bit and dense, in two steps:
 *
 *   PER PHASE   the operators keep handing out ids from the object's own
 *               counter, which makes them locally unique and globally
 *               ambiguous. One MPI_Exscan afterwards turns the local block into
 *               a globally disjoint one -- entities created this phase are
 *               exactly those with `global_id >= the watermark recorded before
 *               the sweeps`, so nothing in the operators had to change. Ids are
 *               unique but no longer dense; the sparsity is bounded by the
 *               creations of one remeshing call.
 *
 *   AT THE END  one MPI_Exscan over the OWNED counts renumbers every owned
 *               entity to `base + slot`, which is dense over the whole mesh by
 *               construction. Ghost vertices learn their new ids from
 *               MFTLRemeshSyncGhostIds, and ghost triangles do not need to:
 *               the closing redistribution drops and re-receives them.
 *
 * The order of those last three is forced and each ordering constraint is silent
 * when violated -- see redensify_global_ids().
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
#include "mftl_topology.h"
#include "remesh_comm.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The phase schedule
 * ═══════════════════════════════════════════════════════════════════════════ */

const double MFTL_REMESH_PHASES[MFTL_REMESH_NUM_PHASES][3] = {
    /* The workhorses: between them every SLAB conflict is interior once. */
    {0.0, 0.0, 0.0},
    {1.0, 1.0, 1.0},
    /* Completing the even-parity orthogonal array: every TUBE conflict. */
    {0.0, 1.0, 1.0},
    {1.0, 0.0, 1.0},
    /* The odd-parity set: every CORNER conflict. Measure O((delta/h)^3), so
     * these normally perform a handful of operations and cost one migration. */
    {1.0, 1.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Freeze classification
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshMarkFrozen(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(MFTLObjectHasTopology(obj) &&
              "MFTLRemeshMarkFrozen: tris.verts/neighbors are stale. The classification IS those two arrays -- "
              "call MFTLObjectBuildDerived first (a redistribution leaves them current).");

  /* This writes entity FLAGS without changing any entity count, which is one of
   * the three things MFTLRedistribute's per-body skip has to be told about: a
   * kept ghost vertex retains its LOCAL record across a redistribution, so a
   * flag written here would otherwise survive a skip and diverge from its
   * owner's. Once per phase, not once per timestep -- it costs nothing here and
   * it removes the one class of divergence the skip cannot see. */
  MFTLObjectMarkPartitionDirty(obj);

  const int nv = obj->verts.n;
  const int nvl = obj->verts.n_local;
  const int nt = obj->tris.n;
  const int ntl = obj->tris.n_local;

  MFTL_Vertex *const restrict vr = obj->verts.rec;
  MFTL_Triangle *const restrict tr = obj->tris.rec;

  for (int v = 0; v < nv; ++v) vr[v].flags &= ~MFTL_FLAG_FROZEN;
  for (int t = 0; t < nt; ++t) tr[t].flags &= ~MFTL_FLAG_FROZEN;

  /* No ghosts means nothing is replicated, which is the serial case -- and the
   * cleared flags above are then the whole answer, so the gates in the sweeps
   * are exactly the no-ops they need to be for the serial results to be
   * unchanged. */
  if (nvl == nv && ntl == nt) return;

  const int *const restrict tv = obj->tris.verts;
  const int *const restrict tn = obj->tris.neighbors;

  /* ── Owned triangles: frozen iff some neighbour was sent a copy ───────────
   *
   * Six compares against two loop-invariant scalars over two contiguous 12 B/tri
   * streams, with a branchless flag update -- the classification never asks a
   * question that needs a pointer chase. MFTL_ERROR (-1) is below every count,
   * so a triangle at a genuine mesh boundary is not accidentally frozen by its
   * missing neighbour. */
  for (int t = 0; t < ntl; ++t)
  {
    const int b = 3 * t;
    const int replicated =
        (tv[b] >= nvl) | (tv[b + 1] >= nvl) | (tv[b + 2] >= nvl) | (tn[b] >= ntl) | (tn[b + 1] >= ntl) | (tn[b + 2] >= ntl);

    tr[t].flags |= replicated ? MFTL_FLAG_FROZEN : 0;
  }

  for (int t = ntl; t < nt; ++t) tr[t].flags |= MFTL_FLAG_FROZEN;

  /* ── Vertices: frozen iff ANY incident triangle is ────────────────────────
   *
   * Scattered, but only over the band: the branch skips every interior triangle,
   * and the frozen set is O(delta/h) of the mesh plus the ghosts. Doing it the
   * other way round -- per vertex, walking its fan -- would be a dependent load
   * chain over the WHOLE pool to answer a question about a thin shell.
   *
   * This direction is also what makes the vertex predicate strong enough to
   * gate a collapse on its own: "unfrozen vertex" now implies "every incident
   * triangle is mine and unreplicated", so freeze_blocks_collapse() needs two
   * flag tests where the modified set is an entire fan. */
  for (int t = 0; t < nt; ++t)
  {
    if (!(tr[t].flags & MFTL_FLAG_FROZEN)) continue;

    const int b = 3 * t;
    vr[tv[b]].flags |= MFTL_FLAG_FROZEN;
    vr[tv[b + 1]].flags |= MFTL_FLAG_FROZEN;
    vr[tv[b + 2]].flags |= MFTL_FLAG_FROZEN;
  }
}

void MFTLDilateFrozenBand(MFTL_Object *obj, int rings)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  if (rings <= 0) return;

  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLDilateFrozenBand: tris.verts is stale");

  const int nt = obj->tris.n;
  const int ntl = obj->tris.n_local;

  MFTL_Vertex *const restrict vr = obj->verts.rec;
  MFTL_Triangle *const restrict tr = obj->tris.rec;
  const int *const restrict tv = obj->tris.verts;

  /* One ring is triangle-then-vertex, and that ordering is what makes it exactly
   * one ring with no scratch bit. The first half reads VERTEX flags and writes
   * TRIANGLE flags; the second reads triangle flags and writes vertex flags.
   * Neither half can see its own output, so neither can propagate within itself
   * -- which is the failure a same-array vertex-to-vertex sweep has, where the
   * reach of a single pass depends on the order the half-edges happen to be in. */
  for (int r = 0; r < rings; ++r)
  {
    for (int t = 0; t < ntl; ++t)
    {
      if (tr[t].flags & MFTL_FLAG_FROZEN) continue;

      const int b = 3 * t;
      if ((vr[tv[b]].flags | vr[tv[b + 1]].flags | vr[tv[b + 2]].flags) & MFTL_FLAG_FROZEN) tr[t].flags |= MFTL_FLAG_FROZEN;
    }

    for (int t = 0; t < nt; ++t)
    {
      if (!(tr[t].flags & MFTL_FLAG_FROZEN)) continue;

      const int b = 3 * t;
      vr[tv[b]].flags |= MFTL_FLAG_FROZEN;
      vr[tv[b + 1]].flags |= MFTL_FLAG_FROZEN;
      vr[tv[b + 2]].flags |= MFTL_FLAG_FROZEN;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2b. The convergence certificate
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Does this rank count the undirected edge at @p h, mesh-wide exactly once?
 *
 * The edge is attributed to the owner of the incident triangle of SMALLER GLOBAL
 * ID. Attributing it to "whoever owns a triangle here" instead double-counts
 * every cut edge, once on each side -- by an amount that GROWS WITH THE RANK
 * COUNT, which is the one shape of error a certificate about partitioning must
 * not have.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` if this rank is the one that should COUNT this undirected edge
 *         -- exactly one rank must, or the certificate double-counts.
 */
static bool owns_edge_count(const MFTL_Object *obj, int h)
{
  const int t = MFTLHeTriIdx(h);
  if (t >= obj->tris.n_local) return MFTL_FALSE;  // the ghost's owner counts it

  const int tw = obj->tris.he[h].twin_idx;
  if (tw == MFTL_ERROR) return MFTL_TRUE;  // mesh boundary: one triangle, and it is mine

  const int to = MFTLHeTriIdx(tw);
  if (to < obj->tris.n_local) return h < tw;  // both mine: the smaller half-edge

  return obj->tris.rec[t].global_id < obj->tris.rec[to].global_id;
}

/**
 * @brief Out-of-band edges that THIS PLACEMENT of the cut planes hides from
 *        every rank at once, split by the OPERATOR that would fix them.
 *
 * The certificate the driver converges on, and it is deliberately not the
 * question "how many operations did the gate refuse". That one cannot be
 * answered locally -- see the comment above the sweeps in remesh_driver.c, where
 * a rank's partial view of a neighbour's one-ring invents blocked work that does
 * not exist. This one can, because it needs no fan walk:
 *
 *   an edge is OFFERABLE  <=>  the operator that would fix it is not gated
 *
 * and the two operators have DIFFERENT gates, which is why this returns two
 * numbers rather than one. A split is gated by freeze_blocks_pair -- the two
 * incident TRIANGLES -- while a collapse is gated by freeze_blocks_collapse --
 * the two endpoint VERTICES. The vertex-frozen set strictly contains the
 * triangle-frozen one (a vertex is frozen when ANY incident triangle is, and
 * that triangle need not touch this edge), so testing a long edge with the
 * vertex rule, which is what this function used to do, OVER-REPORTS: it declares
 * hidden an edge that the very next sweep splits. An over-report suppresses the
 * fast exit, and suppressing the fast exit costs a full covering set of
 * migrations.
 *
 * -- Why the two counts mean different things --------------------------------
 *
 * `hidden_long` is EXACT, and that is worth stating as a theorem because the
 * driver rests on it. Suppose it is zero and no rank performed an operation.
 * Zero under the pair gate means every long edge had both incident triangles
 * unfrozen on the rank that counts it. If one of them were a ghost it would be
 * frozen (every ghost is), so both are owned; sweep_split therefore reaches the
 * canonical half-edge, and MFTLSplitApproved refuses only for DEAD, VISITED or
 * non-canonical -- never on merit, since a split needs no link condition and no
 * fold test. DEAD and VISITED are both clear at measurement time (the compaction
 * and clear_visited run before the phase ends). So the edge would have been
 * split, so ops > 0, contradiction. There were no long edges.
 *
 * `hidden_short` is NOT exact, and cannot be made so locally. A collapse can be
 * refused on its merits by four separate rules -- the link condition, the wing
 * valence floor, the stretch bound, and the DIRECTIONAL feature/boundary/corner
 * rules (sweep_collapse only ever offers the canonical half-edge, so a short
 * edge collapsible in the other direction only is collapsed by nobody). Such an
 * edge is out of band forever and counts as hidden forever. That is the false
 * alarm the streak exit exists to absorb.
 *
 * The one thing neither count can say is anything about the mesh HANDED BACK:
 * this runs before the sweeps, and MFTLRemeshIterate smooths after them, so the
 * statement is always "at the start of the certifying phase".
 *
 * The alarm the precondition needs survives both: an out-of-band edge that no
 * placement ever offers -- the h <= 4*delta failure -- is counted in every phase.
 *
 * @note Called BEFORE the sweeps, so it measures what the placement hides going
 *       in. In the converged regime -- which is the only regime whose value is
 *       ever read -- the sweeps change nothing, so before and after coincide.
 *       Reading it after would also be wrong for a duller reason: the entities
 *       the sweeps create carry provisional ids until assign_phase_gids runs,
 *       and the attribution rule above compares ids across ranks.
 *
 * @brief The convergence certificate: out-of-band edges with a frozen endpoint.
 *
 * Needs no fan walk, so a rank holding only a partial view cannot give it a
 * wrong answer -- which is why it replaced "approved, then refused by the gate".
 *
 * @param      obj          The mesh.
 * @param      params       Supplies the length band.
 * @param[out] hidden_long  Out-of-band LONG edges with a frozen endpoint.
 * @param[out] hidden_short Out-of-band SHORT edges with a frozen endpoint.
 */
static void count_hidden_edges(const MFTL_Object *obj, const MFTL_RemeshParameters *params, int *hidden_long, int *hidden_short)
{
  const int n_he = 3 * obj->tris.n;

  for (int h = 0; h < n_he; ++h)
  {
    if (!owns_edge_count(obj, h)) continue;

    const double l2 = MFTLGetEdgeLengthSquared(obj, h);

    if (l2 > params->max_edge_length_sq)
    {
      if (freeze_blocks_pair(obj, h)) (*hidden_long)++;
    }
    else if (l2 < params->min_edge_length_sq)
    {
      if (freeze_blocks_collapse(obj, h)) (*hidden_short)++;
    }
  }
}

/**
 * @brief Removes the transient classification. Called once, on the way out.
 *
 * @param ens The mesh. #MFTL_FLAG_FROZEN is cleared on every entity, so the
 *            serial closing pass sees an unconstrained mesh.
 */
static void clear_frozen(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v) obj->verts.rec[v].flags &= ~MFTL_FLAG_FROZEN;
    for (int t = 0; t < obj->tris.n; ++t) obj->tris.rec[t].flags &= ~MFTL_FLAG_FROZEN;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Global ids
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct gid_scratch
 * @brief The four O(num_objects) int arrays every id collective needs.
 *
 * One allocation, because they share a lifetime and none of them is ever large:
 * four ints per body, against a message of two.
 */
/**
 * @brief The convergence certificate, summed over ranks once per phase.
 *
 * A 3-int Allreduce is pure latency and does not get cheaper for being small, so
 * everything the driver needs summed over ranks shares one. It used to ride the
 * id Allreduce instead, which was free while the ids were themselves a
 * `comm_world` reduction of `2 * bodies` ints; now that they go over the
 * 26-neighbour graph, appending three ensemble-wide ints to them would put a
 * `comm_world` reduction back on every phase. The array is read back IN PLACE by
 * the driver, by name.
 */
/** @brief Trailing slot: operations performed this phase, summed over ranks. */
#define MFTL_REMESH_EXTRA_OPS 0
/** @brief Trailing slot: out-of-band LONG edges with a frozen endpoint. */
#define MFTL_REMESH_EXTRA_HIDDEN_LONG 1
/** @brief Trailing slot: out-of-band SHORT edges with a frozen endpoint. */
#define MFTL_REMESH_EXTRA_HIDDEN_SHORT 2
/** @brief Number of trailing slots the convergence certificate rides along in. */
#define MFTL_REMESH_EXTRA_INTS 3

typedef struct
{
  double *local;  ///< [2*b] what this rank contributes: {vertices, triangles} per LOCAL SLOT.
  double *prefix;  ///< [2*b] exclusive prefix over the body's holders: this rank's offset.
  double *total;  ///< [2*b] sum over the body's holders: the block length.
  int *base;  ///< [b]   per-LOCAL-SLOT vertex id base, as MFTLRemeshSyncGhostIds wants it.
  int *watermark;  ///< [2*b] the id counters as they stood BEFORE a phase's sweeps.
  int cap;  ///< Local slots the arrays are sized for.
} gid_scratch;

/**
 * @brief Sizes the scratch for @p b local bodies, never shrinking.
 *
 * ── IT IS INDEXED BY LOCAL SLOT, AND THAT IS THE POINT ────────────────────
 *
 * The version this replaces was `11 * n_bodies_global + 2E` ints, allocated once
 * per pass on every rank -- 44 MB at a million droplets, to hand out ids for the
 * handful a rank holds. Every one of those arrays was the payload of an
 * `MPI_Exscan` or an `MPI_Allreduce` over `comm_world`, which is the only place
 * the body id has to be the index: `MPI_Exscan` aligns element k of every rank's
 * buffer. MFTLEnsembleExscanBodies() moves that indexing inside itself, and only
 * on the route that still needs it.
 *
 * A ratchet, and re-sized per PHASE rather than per pass, because a phase
 * migrates and a migration changes how many bodies this rank holds. Growth only:
 * the counts move by a few percent between phases, so after the first one every
 * later phase reuses warm pages.
 *
 * @param g The scratch handle.
 * @param b Local bodies this phase.
 */
static void gid_scratch_reserve(gid_scratch *g, int b)
{
  /* At least one slot even for a rank holding nothing: under a coarsened
   * placement that is the common case, and the arrays are still HANDED to a
   * collective (MFTLRemeshSyncGhostIds) that has no use for a null pointer it
   * will never dereference. */
  if (b < 1) b = 1;
  if (b <= g->cap) return;

  int cap = (g->cap > 0) ? g->cap : 8;
  while (cap < b) cap *= 2;

  free(g->local);
  free(g->base);

  /* 2 + 2 + 2 doubles and 1 + 2 ints per slot, as two blocks so that a phase
   * takes two mallocs however many bodies it holds. */
  g->local = (double *) MFTLCallocOrDie((size_t) (6 * cap), sizeof(double), "remesh id scratch");
  g->prefix = g->local + 2 * cap;
  g->total = g->prefix + 2 * cap;

  g->base = (int *) MFTLCallocOrDie((size_t) (3 * cap), sizeof(int), "remesh id watermarks");
  g->watermark = g->base + cap;

  g->cap = cap;
}

/**
 * @brief Zeroes the handle. @param g The scratch handle.
 */
static void gid_scratch_init(gid_scratch *g)
{
  g->local = MFTL_NULL;
  g->prefix = MFTL_NULL;
  g->total = MFTL_NULL;
  g->base = MFTL_NULL;
  g->watermark = MFTL_NULL;
  g->cap = 0;
}

/**
 * @brief Releases the two blocks.
 * @param g The scratch handle.
 */
static void gid_scratch_free(gid_scratch *g)
{
  free(g->local);
  free(g->base);
  gid_scratch_init(g);
}

/**
 * @brief The per-body exclusive prefix sum, plus the total.
 *
 * ── THE SCOPE IS AUTO, AND IT IS NOT A DETAIL ─────────────────────────────
 *
 * #MFTL_BODY_REDUCE_SCOPE_AUTO lets MFTLEnsembleExscanBodies() take the route
 * the partition actually permits: the body's HOLDERS over the 26-neighbour
 * graph when every holder set fits inside one neighbourhood, and `comm_world`
 * when it does not -- which is exactly what a COARSENED placement makes normal,
 * because a coarse representative is `group` fine cells away and no local test
 * would call it a neighbour. The same condition MFTLRemeshSyncGhostIds() falls
 * back on, decided by the same one int.
 *
 * Pinning this to #MFTL_BODY_REDUCE_SCOPE_GLOBAL is a scaling regression THAT NO
 * TEST OF THE MESH CAN SEE. The two routes agree to the bit by construction, so
 * every digest in docs/analysis.md §5.3 reproduces either way; the
 * cost is `3 * n_bodies_global * 2` doubles of staging and two `comm_world`
 * collectives over a third of that, ONCE PER PHASE, on a rank that may hold a
 * hundred of a million bodies. `test_a_droplet_remesh_takes_no_global_reduction`
 * (tests/test_body_reduce.c) counts the routes rather than the answer, and is
 * the only thing in the suite that would notice.
 *
 * @param      ens The ensemble.
 * @param      g   The scratch handle; `local` must be filled in.
 * @collective
 */
static void gid_reduce(MFTL_Ensemble *ens, gid_scratch *g)
{
  const int b = ens->n_local_objects;
  if (b > 0) memcpy(g->prefix, g->local, (size_t) (2 * b) * sizeof(double));
  MFTLEnsembleExscanBodies(ens, g->prefix, g->total, 2, MFTL_BODY_REDUCE_SCOPE_AUTO);
}

/**
 * @brief Turns the phase's locally-numbered new entities into globally unique ones.
 *
 * The operators never learned about MPI: they keep drawing ids from the object's
 * own counter, so two ranks that both split an edge produce the same id. What
 * makes that harmless is that NOTHING COMMUNICATES between the sweeps and this
 * function -- the ambiguity exists only inside one rank's phase, and this closes
 * it before the next migration can observe it.
 *
 * Identifying what is new needs no bookkeeping either: @p watermark is each
 * object's id counter as it stood before the sweeps, and every id at or above it
 * was handed out by them.
 *
 * `g->watermark` holds {next_vertex_gid, next_triangle_gid} per body as recorded
 * before the sweeps. @p extra is MFTL_REMESH_EXTRA_INTS convergence counters
 * riding the same Allreduce, which is why they are an argument here rather than
 * a collective of their own.
 *
 * @param ens   The ensemble.
 * @param g     The scratch handle, holding the Exscan offsets.
 * @param extra `[MFTL_REMESH_EXTRA_INTS]` convergence counters to carry along in
 *              the same reduction, or NULL.
 */
static void assign_phase_gids(MFTL_Ensemble *ens, gid_scratch *g, int *extra)
{
  const int b = ens->n_local_objects;
  const int *const watermark = g->watermark;

  gid_scratch_reserve(g, b);

  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    int cv = 0, ct = 0;

    for (int v = 0; v < obj->verts.n_local; ++v)
      if (obj->verts.rec[v].global_id >= watermark[2 * o + 0]) cv++;
    for (int t = 0; t < obj->tris.n_local; ++t)
      if (obj->tris.rec[t].global_id >= watermark[2 * o + 1]) ct++;

    g->local[2 * o + 0] = (double) cv;
    g->local[2 * o + 1] = (double) ct;
  }

  gid_reduce(ens, g);

  /* ── The convergence counters no longer ride along, and that is a TRADE ───
   *
   * They used to be appended to the id Allreduce, so the phase paid no third
   * latency for them -- a good trade while the id payload was `2 * bodies` ints
   * on `comm_world` and three more cost nothing. It is the wrong trade now: the
   * ids go over the 26-neighbour graph and are `O(local bodies)`, so appending
   * three ensemble-wide ints to them would put a `comm_world` reduction back on
   * the phase. Three ints of pure latency, once per phase, buys removing an
   * `O(bodies in the ensemble)` payload from every phase. */
  MFTLRemeshSumInts(&ens->domain, extra, MFTL_REMESH_EXTRA_INTS);

  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    const int wv = watermark[2 * o + 0];
    const int wt = watermark[2 * o + 1];

    int next_v = wv + (int) g->prefix[2 * o + 0];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      if (obj->verts.rec[v].global_id >= wv) obj->verts.rec[v].global_id = next_v++;
    }

    int next_t = wt + (int) g->prefix[2 * o + 1];
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      if (obj->tris.rec[t].global_id < wt) continue;

      const int gid = next_t++;
      obj->tris.rec[t].global_id = gid;
      /* Not a free choice: a half-edge id is 3 * (triangle id) + k mesh-wide,
       * which is what lets redistribution reconstruct them instead of sending
       * them (inv_halfedge_gids_are_implicit). */
      for (int k = 0; k < 3; ++k) obj->tris.he[3 * t + k].global_id = 3 * gid + k;
    }

    /* The watermark advances by the GLOBAL creation count on every holder, so it
     * stays a value all of them agree on -- which is what the next phase's
     * "created this phase" test and redistribution's table sizing both rest on. */
    obj->next_vertex_gid = wv + (int) g->total[2 * o + 0];
    obj->next_triangle_gid = wt + (int) g->total[2 * o + 1];
    obj->next_halfedge_gid = 3 * obj->next_triangle_gid;
  }
}

/**
 * @brief Makes the ids of both classes a dense permutation of [0, N) again.
 *
 * ── The three steps are ordered, and every ordering constraint is silent ───
 *
 * 1. REDUCE first. The Exscan of the owned counts is what "base + slot" means,
 *    and the ghost exchange in step 2 answers with it.
 * 2. EXCHANGE before overwriting. The request is keyed on the OLD id, so the
 *    owner must still be holding old ids when it answers. Renumbering first
 *    would make every request unanswerable -- and the reply path treats an
 *    unanswerable request as a skip, so the ghosts would quietly keep stale ids
 *    instead of failing.
 * 3. OVERWRITE last, by SLOT. Any other rule would need a second table on the
 *    owner to answer step 2; numbering by slot makes the answer arithmetic.
 *
 * Ghost TRIANGLES are deliberately not exchanged. Redistribution keeps only
 * owned triangles and re-receives every ghost, so a ghost triangle's id is
 * replaced wholesale by the caller's closing migration -- which is also why that
 * migration has to come AFTER this function and not before it.
 *
 * @param ens The ensemble.
 * @param g   The scratch handle.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the ghost id exchange failed.
 */
static int redensify_global_ids(MFTL_Ensemble *ens, gid_scratch *g)
{
  const int b = ens->n_local_objects;

  gid_scratch_reserve(g, b);

  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    g->local[2 * o + 0] = (double) obj->verts.n_local;
    g->local[2 * o + 1] = (double) obj->tris.n_local;
  }

  gid_reduce(ens, g);

  /* `base` is indexed by LOCAL SLOT, as MFTLRemeshSyncGhostIds() reads it. */
  for (int o = 0; o < b; ++o) g->base[o] = (int) g->prefix[2 * o + 0];

  const int status = MFTLRemeshSyncGhostIds(ens, g->base);

  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    const int base_v = (int) g->prefix[2 * o + 0];
    const int base_t = (int) g->prefix[2 * o + 1];

    for (int v = 0; v < obj->verts.n_local; ++v) obj->verts.rec[v].global_id = base_v + v;

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      const int gid = base_t + t;
      obj->tris.rec[t].global_id = gid;
      for (int k = 0; k < 3; ++k) obj->tris.he[3 * t + k].global_id = 3 * gid + k;
    }

    obj->next_vertex_gid = (int) g->total[2 * o + 0];
    obj->next_triangle_gid = (int) g->total[2 * o + 1];
    obj->next_halfedge_gid = 3 * obj->next_triangle_gid;

    /* The ids of entities other ranks hold copies of have just been rewritten,
     * and the ghost TRIANGLES above were deliberately left on the previous
     * numbering. The caller's closing migration is what repairs them -- by
     * dropping and re-receiving every one of them -- so it must not be allowed
     * to decide the body is unchanged and skip it. Without this stamp the skip
     * fires on exactly the most common case: a converged body whose smoother
     * moved no vertex across a cut plane. */
    MFTLObjectMarkPartitionDirty(obj);
  }

  return status;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Preconditions
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The band this partition ACTUALLY has, and an upper bound on it.
 *
 * ══ WHY THE ESTIMATE IS NOT GOOD ENOUGH, WITH NUMBERS ══════════════════════
 *
 * `frozen_band_width` is 3 * max_edge_length, derived from the reach of the
 * freeze rule: one triangle to the plane, one more to a triangle that reaches
 * it, one more because a vertex is frozen when ANY incident triangle is. It is
 * an upper bound and it is a loose one. Measured on a partitioned sphere in
 * [-1,1]^3 across a sweep of target lengths, the band that actually occurs is
 *
 *     0.84 * max_edge_length
 *
 * -- about ONE edge, not three, and consistently so from target 0.06 to 0.50.
 * The consequence is not academic: `h > 4*delta` computed from the estimate
 * refuses configurations that work perfectly. At a target of 0.12 on that
 * fixture the estimate says 4*delta = 1.92 against h = 1.0, i.e. "this cannot
 * work", and the pass converges with nothing hidden at every rank count. A
 * precondition that cries wolf by a factor of three is a precondition nobody
 * reads.
 *
 * So the band is measured. Two numbers, because they answer different questions:
 *
 *   out[0]  band_measured -- max over this rank's OWNED FROZEN vertices of the
 *           distance to the nearest cut plane. What the band IS, right now.
 *
 *   out[1]  the longest owned edge, which times MFTL_REMESH_BAND_EDGES is a
 *           measured upper bound for the band later in the pass. Note it can
 *           legitimately exceed `params->max_edge_length`: a collapse can
 *           lengthen an edge past the band, and the split sweep's snapshot bound
 *           defers the correction by one iteration.
 *
 * ── The trap in out[0], which the obvious spelling walks into ──────────────
 *
 * The distance has to be the MINIMUM over axes, then the maximum over vertices.
 * A vertex frozen because of a y plane sits about h/2 from every x plane, so a
 * maximum over axes reports h/2 on the first sample and the measurement is
 * worthless. Axes with cuts < 2 are skipped for the same reason they are skipped
 * in MFTLDomainGetHMin: they have no interior plane to be near.
 *
 * LOCAL, no MPI -- the caller reduces both with one MFTLRemeshAgreeDoubleMax.
 *
 * @param      obj   The mesh.
 * @param      dom   The MPI/physical world.
 * @param      shift Per-axis cut-plane offset of the current phase.
 * @param[out] out   {largest frozen distance from a plane, largest owned edge}.
 */
static void local_band_probe(const MFTL_Object *obj, const MFTL_Domain *dom, const double shift[3], double out[2])
{
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    if (!MFTLVertexIsFrozen(obj, v)) continue;

    double nearest = MFTL_DBL_MAX;
    for (int k = 0; k < 3; ++k)
    {
      const int cuts = dom->cuts[k];
      if (cuts < 2 || dom->domain_size[k] <= 0.0) continue;

      const double h = dom->domain_size[k] / (double) cuts;

      /* THE PLANES OF THIS PHASE, not the canonical ones. Measuring against the
       * canonical planes during a shifted phase reports h/2 for every frozen
       * vertex -- the band is at the shifted planes, which is exactly h/2 away --
       * so the running maximum saturates at h/2 and says nothing at all. Same
       * arithmetic as MFTLRemeshOwnerProc, kept in step with it deliberately. */
      double sc = (obj->verts.pos[3 * v + k] - dom->global_min[k] - shift[k]) / h;
      if (dom->is_periodic[k])
      {
        sc += (double) cuts;
        while (sc >= (double) cuts) sc -= (double) cuts;
      }

      int i = (int) floor(sc);
      if (i < 0) i = 0;
      if (i >= cuts) i = cuts - 1;

      const double frac = sc - (double) i;

      /* On a WALLED axis the domain faces are not cut planes, so the first and
       * last cells have only one bounding plane each. Offering them the face
       * would report a spuriously small distance for every vertex near the wall
       * and understate the band. */
      if (dom->is_periodic[k] || i > 0) nearest = MFTL_MIN(nearest, frac * h);
      if (dom->is_periodic[k] || i + 1 < cuts) nearest = MFTL_MIN(nearest, (1.0 - frac) * h);
    }

    if (nearest < MFTL_DBL_MAX && nearest > out[0]) out[0] = nearest;
  }

  const int n_he = 3 * obj->tris.n_local;
  for (int h = 0; h < n_he; ++h)
  {
    const double len = MFTLGetEdgeLength(obj, h);
    if (len > out[1]) out[1] = len;
  }
}

/**
 * @brief Accumulates this rank's owned and frozen-owned triangle counts.
 *
 * ── Why TRIANGLES and not vertices ─────────────────────────────────────────
 *
 * Because the quantity being estimated is the share of a MIGRATION that buys
 * nothing, and the migration's payload is dominated by triangles: a closed mesh
 * has twice as many of them as vertices, and each carries three half-edges. A
 * vertex-based share would be measuring a different, smaller thing and comparing
 * it against a threshold calibrated on this one.
 *
 * ── Why it is a running sum and not a fraction ─────────────────────────────
 *
 * The ratio has to be taken AFTER the reduction. Averaging per-rank fractions
 * weights a rank holding four triangles the same as one holding forty thousand,
 * and at the partition boundary those are the ranks that differ most -- which is
 * precisely the population the measurement is about.
 *
 * LOCAL, no MPI. The caller reduces both with one MFTLRemeshAgreeDouble (SUM).
 *
 * @param      obj The mesh, with #MFTL_FLAG_FROZEN already current.
 * @param[out] out {owned triangles, frozen owned triangles}, accumulated into.
 */
static void local_frozen_count(const MFTL_Object *obj, double out[2])
{
  const int nt = obj->tris.n_local;
  int frozen = 0;

  for (int t = 0; t < nt; ++t)
  {
    if (MFTLTriangleIsFrozen(obj, t)) frozen++;
  }

  out[0] += (double) nt;
  out[1] += (double) frozen;
}

/**
 * @brief The band that would hold the frozen share at or under @p f_target.
 *
 * ── One formula, two constants ─────────────────────────────────────────────
 *
 * The correctness rule is `g[k] = ceil(4*delta / h_k)` and the efficiency rule
 * is `g[k] = ceil(2*delta / ((1-u) * h_k))`, so they differ only by a constant
 * and the second can be expressed as the first applied to an INFLATED BAND.
 * That is why this returns a band rather than a factor: it lets both triggers go
 * through the one published MFTLRemeshCoarseningFactors, which is tested, clamps
 * to `cuts[k]`, and skips uncut axes -- three behaviours a second implementation
 * would have to reproduce and would eventually stop reproducing.
 *
 * ── Where u comes from ─────────────────────────────────────────────────────
 *
 * A box loses `2*delta` per cut axis, so the unfrozen share is the product over
 * cut axes of `(1 - 2*delta/h_k)`. Requiring each axis to contribute at least
 * `u = (1 - f_target)^(1/n_cut)` makes the product at least `1 - f_target`, and
 * is the only split of the requirement across axes that does not privilege one.
 * Solving `1 - 2*delta/(g*h) >= u` for g gives the rule above.
 *
 * Check it against the degenerate case: with ONE cut axis, `u = 1 - f_target`,
 * so the band inflates by `1/(2*f_target)` and at a third that is `h >= 6*delta`
 * -- which is `f = 2*delta/h <= 1/3` read straight off, with no model in between.
 *
 * @param dom      The MPI/physical world; supplies which axes are cut.
 * @param delta    The measured band half-width.
 * @param f_target The frozen share the caller is willing to accept, in (0, 1).
 * @return The band to hand MFTLRemeshCoarseningFactors, or @p delta unchanged
 *         when the target is not a constraint.
 * @rank_local Pure arithmetic on replicated data.
 */
static double band_for_frozen_target(const MFTL_Domain *dom, double delta, double f_target)
{
  if (!(f_target > 0.0) || f_target >= 1.0 || !(delta > 0.0)) return delta;

  int n_cut = 0;
  for (int k = 0; k < 3; ++k)
  {
    if (dom->cuts[k] >= 2 && dom->domain_size[k] > 0.0) n_cut++;
  }
  if (n_cut == 0) return delta;

  const double u = pow(1.0 - f_target, 1.0 / (double) n_cut);

  /* u == 1 is unreachable for f_target in (0,1), but the division is guarded
   * anyway: the alternative to a guard here is an infinity propagating into a
   * ceil() and out as a coarsening factor of INT_MIN. */
  if (!(u < 1.0)) return delta;

  const double inflated = delta / (2.0 * (1.0 - u));
  return MFTL_MAX(delta, inflated);
}

/**
 * @brief The geometric half-width of the frozen band, in domain length units.
 *
 * The band is "one triangle plus one adjacency hop" either side of a cut plane:
 * a triangle is replicated when a corner or an edge neighbour belongs to another
 * rank, so the deepest an entity can be and still be frozen is one triangle from
 * the plane plus one more from its neighbour. Two edges is therefore the honest
 * estimate, and `max_edge_length` is the only bound on an edge the remesher
 * guarantees -- using the TARGET length instead would understate it by a third
 * at exactly the moment the mesh is worst.
 *
 * A dilation ring adds one more triangle to the reach, so it is counted.
 *
 * @param params Supplies `max_edge_length` and `freeze_dilation_rings`.
 * @return The conservative band half-width, \f$(\texttt{BAND\_EDGES} + \texttt{rings})
 *         \cdot \texttt{max\_edge\_length}\f$. Roughly three times the measured
 *         reach, which is why both are reported.
 */
static double frozen_band_width(const MFTL_RemeshParameters *params)
{
  return (MFTL_REMESH_BAND_EDGES + (double) MFTL_MAX(params->freeze_dilation_rings, 0)) * params->max_edge_length;
}

void MFTLRemeshCoarseningFactors(const MFTL_Domain *domain, const MFTL_RemeshParameters *params, double band, int out_group[3])
{
  MFTL_ASSERT(domain != MFTL_NULL && params != MFTL_NULL && out_group != MFTL_NULL);

  const double delta = (band > 0.0) ? band : frozen_band_width(params);

  for (int k = 0; k < 3; ++k)
  {
    out_group[k] = 1;

    const int cuts = domain->cuts[k];
    if (cuts < 2 || domain->domain_size[k] <= 0.0 || delta <= 0.0) continue;

    const double h = domain->domain_size[k] / (double) cuts;
    const double ratio = 4.0 * delta / h;
    if (ratio <= 1.0) continue;

    int g = (int) ceil(ratio);

    /* Never more groups than there are cells: merging every cell on an axis
     * leaves one box spanning the domain, which is the most this can do and is
     * still a legal decomposition. Asking for more would produce zero groups. */
    if (g > cuts) g = cuts;
    out_group[k] = g;
  }
}

/**
 * @brief Reports, ONCE per process, that the rank boxes are too small for the
 *        shift schedule.
 *
 * ── Why this warns and does not assert ─────────────────────────────────────
 *
 * Because h <= 4*delta makes the scheme DEGRADE, not break. The mesh that comes
 * out is structurally perfect; it simply has bands along the cut planes that no
 * placement ever offered to anyone, so they keep whatever triangles they had.
 * `MFTL_RemeshStats.converged` says so, and the caller can act on it.
 *
 * An assertion would abort a running simulation over a mesh-QUALITY shortfall,
 * which is the wrong trade at 3 a.m. on a cluster. It is also wrong for a duller
 * reason that only showed up under test: `freeze_dilation_rings` is a documented,
 * supported parameter that WIDENS delta, so a configuration the API invites the
 * caller to try would abort in Debug and run in Release. A knob whose whole
 * purpose is to be conservative must not be the thing that kills the process.
 *
 * The one-shot latch is not cosmetic either. A solver remeshes every N
 * timesteps, so a per-call warning is thousands of identical lines -- and the
 * reliable effect of that is that nobody reads any of them.
 *
 * @param dom           The MPI/physical world.
 * @param params        Supplies the coarsening cap.
 * @param band_measured The observed band half-width.
 * @param band_bound    The conservative bound from frozen_band_width().
 *
 * Reports on rank 0 when `h > 4*delta` no longer holds, which is the one
 * precondition whose violation is otherwise completely silent.
 */
static void check_band_fits(const MFTL_Domain *dom, const MFTL_RemeshParameters *params, double band_measured, double band_bound)
{
  static bool warned = MFTL_FALSE;

  const double h_min = MFTLDomainGetHMin(dom);

  /* THE MEASURED band, not the estimate. The estimate exceeds the truth by about
   * a factor of three (see local_band_probe), so warning on it fired on
   * configurations that then converged perfectly -- and a warning that is wrong
   * two times in three is a warning that gets filtered out of the log. This one
   * fires when the partition genuinely cannot offer everything to somebody.
   *
   * The conservative bound is still printed, because it is what the coarsening
   * factor is computed from and a caller comparing the two is entitled to see
   * both. */
  if (dom->nproc == 1 || h_min <= 0.0 || band_measured <= 0.0 || h_min > 4.0 * band_measured) return;
  if (warned || dom->rank != 0) return;

  /* The factor quoted is the one the driver would ACTUALLY use -- per axis, from
   * the measured band. A single factor cubed from the estimate comes out about
   * nine times larger, which in a message that says "group this many ranks" is
   * not a safety margin, it is a wrong instruction. */
  int group[3];
  MFTLRemeshCoarseningFactors(dom, params, band_measured, group);

  warned = MFTL_TRUE;
  fprintf(stderr,
          "[MFTL] parallel remeshing: rank box h_min=%.4g is not > 4*delta=%.4g, where delta=%.4g is the MEASURED "
          "frozen band (the %.1f*max_edge_length bound would give %.4g). Entities inside the permanently frozen "
          "bands may never be offered to any rank; MFTL_RemeshStats.converged and n_edges_hidden_long report "
          "whether the pass got there anyway.\n"
          "       Ways out, cheapest first: use the DISPERSED strategy if the bodies fit in a rank box (it has no "
          "band and therefore no precondition); set MFTL_RemeshParameters.parallel_coarsen = 1, which would merge "
          "%dx%dx%d = %d ranks per group here; or remesh to a longer target length. Reported once.\n",
          h_min, 4.0 * band_measured, band_measured, MFTL_REMESH_BAND_EDGES + (double) MFTL_MAX(params->freeze_dilation_rings, 0), band_bound,
          group[0], group[1], group[2], group[0] * group[1] * group[2]);
}

/**
 * @brief Reports, ONCE per process, that the fallback merged rank boxes.
 *
 * ── Why this is announced at all ───────────────────────────────────────────
 *
 * Because arming the fallback by default converts one silent failure into
 * another. Before, a too-fine partition quietly produced un-remeshed bands.
 * Armed, it quietly produces a run in which `1/(g0*g1*g2)` of the ranks do all
 * of the sweeping and the timestep grows by the same factor -- correct, and
 * mystifying if the first you hear of it is a profile. `n_coarse_groups` and
 * `coarsen_trigger` carry the same information for a caller that reads the
 * stats; this is for the one that does not.
 *
 * The one-shot latch is the same discipline as check_band_fits, for the same
 * reason: a solver remeshes thousands of times, and the reliable effect of
 * thousands of identical lines is that nobody reads any of them.
 *
 * @param dom   The MPI/physical world.
 * @param st    Supplies the trigger bitmask and the measured frozen share.
 * @param group The per-axis factors actually adopted.
 */
static void report_fallback(const MFTL_Domain *dom, const MFTL_RemeshStats *st, const int group[3])
{
  static bool announced = MFTL_FALSE;

  if (announced || dom->rank != 0) return;
  announced = MFTL_TRUE;

  const int ranks_per_group = group[0] * group[1] * group[2];

  if (st->coarsen_trigger & MFTL_REMESH_COARSEN_GATHER)
  {
    fprintf(stderr,
            "[MFTL] parallel remeshing: the ensemble fits under serial_gather_max_triangles, so the whole of it "
            "is being remeshed on ONE rank (%d ranks per group, one placement, two migrations). This is a "
            "deliberate trade -- the phase driver does about 3.9x the remeshing work and produces a measurably "
            "better mesh on an adequately resolved surface. Reported once.\n",
            ranks_per_group);
    return;
  }

  /* BAND and FROZEN are reported differently on purpose. One is the price of a
   * correct answer and the other is a trade with two cheaper alternatives the
   * library cannot take on its own, and a message that treated them alike would
   * be wrong for whichever one it was not written for. */
  if (st->coarsen_trigger & MFTL_REMESH_COARSEN_BAND)
  {
    fprintf(stderr,
            "[MFTL] parallel remeshing: the rank boxes are narrower than four frozen bands, so the coarsening "
            "fallback merged %dx%dx%d = %d ranks per group. Without it this pass would have left a lattice of "
            "bands that no placement ever offers to any rank. The cost is that 1/%d of the ranks do all the "
            "sweep work; remeshing to a longer target length, or the DISPERSED strategy if the bodies fit in a "
            "rank box, avoids it. Reported once.\n",
            group[0], group[1], group[2], ranks_per_group, ranks_per_group);
    return;
  }

  fprintf(stderr,
          "[MFTL] parallel remeshing: %.0f %% of the mesh was frozen at the start of the pass, above "
          "coarsen_frozen_fraction, so the fallback merged %dx%dx%d = %d ranks per group. THIS ONE IS OPTIONAL: "
          "the pass would have converged without it, just spending that share of every migration on entities it "
          "could not touch. It costs 1/%d of the parallelism to buy it back. Set coarsen_frozen_fraction to 1.0 "
          "to disable this trigger and keep the correctness one. Reported once.\n",
          100.0 * st->frozen_fraction, group[0], group[1], group[2], ranks_per_group, ranks_per_group);
}

/**
 * @brief Reports, ONCE per process, that `parallel_coarsen`'s cap bound below
 *        what the band precondition required.
 *
 * The failure this makes visible has no other symptom at all. A caller who set
 * `parallel_coarsen = 4` to bound the parallelism loss, on a partition that
 * needed 8, gets a pass that coarsens, costs the parallelism, AND still leaves
 * permanently frozen bands -- the worst of both, reported by `converged` coming
 * back false exactly as it would have with the fallback disarmed.
 *
 * @param dom      The MPI/physical world.
 * @param params   Supplies the cap that bound.
 * @param required The factors the band precondition asked for.
 * @param adopted  The factors the cap allowed.
 */
static void report_cap_bound(const MFTL_Domain *dom, const MFTL_RemeshParameters *params, const int required[3], const int adopted[3])
{
  static bool announced = MFTL_FALSE;

  if (announced || dom->rank != 0) return;
  announced = MFTL_TRUE;

  fprintf(stderr,
          "[MFTL] parallel remeshing: parallel_coarsen = %d capped the fallback at %dx%dx%d = %d ranks per group, "
          "but the measured band needs %dx%dx%d = %d. The pass is paying for coarsening AND keeping permanently "
          "frozen bands; MFTL_RemeshStats.converged and n_edges_hidden_long will report the second half. Raise the "
          "cap, or set parallel_coarsen = 1 to let it choose. Reported once.\n",
          params->parallel_coarsen, adopted[0], adopted[1], adopted[2], adopted[0] * adopted[1] * adopted[2], required[0], required[1], required[2],
          required[0] * required[1] * required[2]);
}

#ifndef NDEBUG
/**
 * @brief Debug check: every rank agrees on the id watermark of every body.
 *
 * The per-phase id blocks are carved out of `next_*_gid`, so a rank whose
 * watermark differs hands out ids that overlap another rank's block -- and two
 * entities with one id are merged by the next redistribution's de-duplication
 * into a single, silently wrong entity. Cheaper to find here than there.
 *
 * @param ens The ensemble.
 * @return `true` if every rank entered the phase with the same id counters.
 *         Debug only.
 */
static bool watermarks_agree(MFTL_Ensemble *ens)
{
  if (ens->n_bodies_global == 0) return MFTL_TRUE;

  /* Per body, over that body's HOLDERS -- the only ranks that have a counter to
   * agree about. A rank holding none of a body is not part of the question, and
   * the check is exactly "does my value survive a MAX over the others". */
  const int b = ens->n_local_objects;
  MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MAX};

  double *const mine = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(4 * b, 1), sizeof(double), "remesh watermark check");
  double *const agreed = mine + 2 * b;

  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    mine[2 * o + 0] = (double) obj->next_vertex_gid;
    mine[2 * o + 1] = (double) obj->next_triangle_gid;
  }

  if (b > 0) memcpy(agreed, mine, (size_t) (2 * b) * sizeof(double));
  MFTLEnsembleReduceBodies(ens, agreed, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  bool ok = MFTL_TRUE;
  for (int o = 0; o < b; ++o)
  {
    if (mine[2 * o + 0] != agreed[2 * o + 0] || mine[2 * o + 1] != agreed[2 * o + 1]) ok = MFTL_FALSE;
  }

  free(mine);
  return ok;
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. The driver
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  MFTL_Domain *const dom = &ens->domain;

  MFTL_RemeshStats sink;
  MFTL_RemeshStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  st->strategy = MFTL_REMESH_PARALLEL_SPANNING;

  /* GLOBAL: everything below is collective. */
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  MFTL_ASSERT(watermarks_agree(ens) &&
              "MFTLRemeshEnsembleParallel: the ranks disagree about an object's next_*_gid. New ids are carved out "
              "of it, so the blocks would overlap and two entities would end up sharing one global id.");

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
              "MFTLRemeshEnsembleParallel", params->min_edge_length, params->max_edge_length);
    }
    return MFTL_ERROR;
  }

  MFTL_ASSERT(MFTLRemeshParamsAgree(dom, params) &&
              "MFTLRemeshEnsembleParallel: the ranks were handed different edge-length bands. Nothing communicates the band, so a rank "
              "with a shorter one refines everything it owns, every phase, without bound. The usual cause is "
              "MFTLMeanEdgeLength (LOCAL) on a mesh a collective generator already distributed.");

  /* ── Regime C: do the rank boxes need merging before anything starts? ────
   *
   * Decided ONCE, before the loop, and from a MEASUREMENT rather than from the
   * estimate. The pre-pass costs one freeze classification and one Allreduce of
   * two doubles on a partition that is already canonical -- no migration, no
   * sweeps -- and it is worth every bit of that, because the estimate overstates
   * the band by about three (see local_band_probe) and the factor enters as a
   * VOLUME: coarsening on the estimate would merge 27 times more ranks per group
   * than the mesh actually requires.
   *
   * The group has to be a pure function of replicated data, and it is: `domain`
   * and `params` are replicated by contract, and the band comes out of an
   * Allreduce(MAX), which is order-independent and therefore bit-identical
   * everywhere. Two ranks computing different groups would partition against
   * different planes with nothing to detect it.
   *
   * If the band later GROWS -- which a coarsening pass does, since it lengthens
   * edges -- the group is then slightly under-conservative. That is not silent:
   * it is exactly what `n_edges_hidden_long` and `converged` report. */
  int group[3] = {1, 1, 1};
  if ((params->parallel_coarsen != 0 || params->serial_gather_max_triangles > 0) && dom->nproc > 1)
  {
    MFTLEnsembleComputeTriangleGeometry(ens);

    double probe[2] = {0.0, 0.0};
    double counts[2] = {0.0, 0.0};
    const double no_shift[3] = {0.0, 0.0, 0.0};
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTLRemeshMarkFrozen(ens->local_objects[o]);
      MFTLDilateFrozenBand(ens->local_objects[o], params->freeze_dilation_rings);
      local_band_probe(ens->local_objects[o], dom, no_shift, probe);
      local_frozen_count(ens->local_objects[o], counts);
    }
    MFTLRemeshAgreeDoubleMax(dom, probe, 2);

    /* A SECOND reduction, and a different OPERATOR -- which is why it is a
     * second one and not two more slots on the first. The band is a maximum
     * (the worst place any rank found) and the counts are sums (how much mesh
     * there is, and how much of it is frozen). Two doubles through an
     * MPI_SUM, once per pass, against several migrations. */
    MFTLRemeshAgreeDouble(dom, counts, 2);

    st->frozen_fraction = (counts[0] > 0.0) ? (counts[1] / counts[0]) : 0.0;

    /* THE BAND THIS PASS WILL HAVE, not the one it starts with. The pre-pass
     * measures the input mesh, and for a REFINING pass that is the worst case --
     * edges only get shorter. For a COARSENING pass it is the best case and
     * predicts nothing: on the [-1,1]^3 fixture at target 0.40 the input band is
     * 0.216 and the pass drives it to 0.44, so a decision taken on 0.216 does
     * not coarsen and the pass then fails to offer work to anybody.
     *
     * So take the larger of what is there and what the target implies. One
     * max-length edge is the right predictor for the second term: the freeze
     * rule's reach argument counts three, but the band that actually occurs is
     * 0.84 * max_edge_length, measured across a sweep from target 0.06 to 0.50.
     * Using the 3-edge bound here would coarsen by 3x per axis, i.e. 27x in
     * ranks per group, to buy margin the measurement says is not needed. */
    const double delta_predicted = MFTL_MAX(probe[0], params->max_edge_length);

    /* ── THE SIZE TRIGGER SHORT-CIRCUITS BOTH OTHERS, AND HAS TO ────────────
     *
     * A gather is `group == cuts`, which is the LARGEST factor either of the
     * other triggers could ever ask for -- MFTLRemeshCoarseningFactors clamps
     * there. So there is nothing for them to add, and evaluating them anyway
     * would only produce a `coarsen_trigger` naming three reasons for one
     * decision. The threshold is a triangle count because ONE RANK HOLDS THE
     * WHOLE ENSEMBLE for the duration of the sweep, and a count is the only
     * form of the question that bounds that. */
    const int gather_limit = params->serial_gather_max_triangles;
    if (gather_limit > 0 && counts[0] <= (double) gather_limit)
    {
      for (int k = 0; k < 3; ++k) group[k] = dom->cuts[k];
      st->coarsen_trigger |= MFTL_REMESH_COARSEN_GATHER;
    }
    else if (params->parallel_coarsen != 0)
    {
      /* ── The two triggers, evaluated as one call on the larger band ───────
       *
       * They are asked separately because `coarsen_trigger` has to report which
       * one fired, and they are ACTED on together because the answer is the
       * larger of the two factors and MFTLRemeshCoarseningFactors is monotone
       * in its band. Running it twice and taking a per-axis maximum would give
       * the same numbers for twice the code. */
      int band_group[3];
      MFTLRemeshCoarseningFactors(dom, params, delta_predicted, band_group);
      if (band_group[0] * band_group[1] * band_group[2] > 1) st->coarsen_trigger |= MFTL_REMESH_COARSEN_BAND;

      double band_used = delta_predicted;
      if (st->frozen_fraction > params->coarsen_frozen_fraction)
      {
        const double band_eff = band_for_frozen_target(dom, delta_predicted, params->coarsen_frozen_fraction);
        if (band_eff > band_used) band_used = band_eff;
      }

      MFTLRemeshCoarseningFactors(dom, params, band_used, group);
      if (group[0] * group[1] * group[2] > band_group[0] * band_group[1] * band_group[2])
      {
        st->coarsen_trigger |= MFTL_REMESH_COARSEN_FROZEN;
      }

      /* A positive value is a CAP on the ranks per group, so a caller who knows
       * how much parallelism they are willing to lose can say so. Applied by
       * shrinking the largest factor first, which keeps the boxes as close to
       * cubic as the cap allows. */
      if (params->parallel_coarsen > 1)
      {
        MFTLRemeshCapGroups(group, params->parallel_coarsen);

        /* THE CAP CAN BIND BELOW WHAT CORRECTNESS NEEDED, and that outcome is
         * indistinguishable from a healthy partition in every other field: the
         * pass simply fails to certify, exactly as it would have with the
         * fallback disarmed. Say so once, or the caller reads `parallel_coarsen
         * = 4` as a guarantee it never was. */
        if (group[0] * group[1] * group[2] < band_group[0] * band_group[1] * band_group[2])
        {
          report_cap_bound(dom, params, band_group, group);
        }
      }
    }

    if (group[0] * group[1] * group[2] > 1) report_fallback(dom, st, group);
  }
  st->n_coarse_groups = group[0] * group[1] * group[2];

  gid_scratch gid;
  /* Indexed by LOCAL SLOT and re-sized per phase, because a phase migrates and a
   * migration changes how many bodies this rank holds. MFTLEnsembleExscanBodies()
   * is what makes that possible: the body-id indexing an `MPI_Exscan` needs now
   * lives inside the collective and only on the route that still requires it. */
  gid_scratch_init(&gid);

  /* One arena for every migration this pass makes, grown once and reused: a
   * pass migrates up to `budget` times, and re-allocating a destination array
   * per entity per phase would pay the first-touch faults every phase. */
  MFTL_RemeshMigrator mig;
  MFTLRemeshMigratorInit(&mig);

  /* ── How many of the eight placements are actually DIFFERENT? ────────────
   *
   * An axis with fewer than two (coarse) cells has no interior plane, so
   * MFTLRemeshPhaseShift leaves it at zero and two table entries that differ
   * only on such an axis ARE THE SAME PLACEMENT. On a 2x1x1 grid all eight
   * entries are two placements; on 2x2x1 they are four.
   *
   * That matters twice, and neither was accounted for before. A repeated
   * placement migrates nothing (which the identity check now catches) but still
   * pays a full set of sweeps and a collective for a phase that cannot discover
   * anything new -- the frozen set is identical. And the convergence streak
   * demanded eight consecutive silent placements, i.e. eight proofs of a
   * statement that two placements already prove on a slab decomposition.
   *
   * So the table is deduplicated, in its existing order, and the streak length
   * becomes the number of distinct entries. Keeping first occurrences preserves
   * the ordering the table was built with -- slabs first, then the even-parity
   * set, then the rest -- which is what makes a small budget spend itself on the
   * placements that matter most. */
  int distinct[MFTL_REMESH_NUM_PHASES];
  int n_distinct = 0;
  {
    bool active[3];
    for (int k = 0; k < 3; ++k)
    {
      const int g = (group[k] > 1) ? group[k] : 1;
      active[k] = (dom->cuts[k] / g) > 1;
    }

    unsigned seen = 0;
    for (int e = 0; e < MFTL_REMESH_NUM_PHASES; ++e)
    {
      unsigned sig = 0;
      for (int k = 0; k < 3; ++k)
        if (active[k] && MFTL_REMESH_PHASES[e][k] != 0.0) sig |= 1u << k;

      if (seen & (1u << sig)) continue;
      seen |= 1u << sig;
      distinct[n_distinct++] = e;
    }
  }
  st->n_distinct_placements = n_distinct;

  const int budget = (params->parallel_phase_budget > 0) ? params->parallel_phase_budget : 4 * MFTL_REMESH_NUM_PHASES;

  int clean_streak = 0;
  int status = MFTL_SUCCESS;

  /* Free-running per-rank accumulation; reduced ONCE, on the way out. Timing
   * every phase collectively would cost more than it measures and would add a
   * barrier the algorithm does not otherwise have. */
  const double t0_call = MFTLRemeshWallTime();
  double t_mark;

  /* {widest frozen band seen, longest owned edge seen}, local running maxima.
   * See local_band_probe. */
  double band_probe[2] = {0.0, 0.0};

  for (int p = 0; p < budget; ++p)
  {
    st->n_phases = p + 1;

    /* ── 1. Move the planes, then move the mesh to match ────────────────────
     *
     * The phase index comes from a compile-time table and the loop counter,
     * never from geometry -- which is what guarantees every rank places the
     * planes identically. MFTLRemeshPhaseShift contains no MPI, so nothing could
     * detect a disagreement, and a disagreement makes ownership ambiguous rather
     * than merely suboptimal.
     *
     * The shift reaches the redistribution as an OWNER MAP and not as a mutated
     * domain: `MFTL_Domain` is read-only during a redistribution, and the
     * production ownership rule (MFTLGetOwnerProc) has no idea any of this
     * exists. See remesh_comm.h. */
    double shift[3];
    MFTLRemeshPhaseShift(dom, group, MFTL_REMESH_PHASES[distinct[p % n_distinct]], shift);

    t_mark = MFTLRemeshWallTime();
    MFTLEnsembleComputeTriangleGeometry(ens);
    st->t_geometry += MFTLRemeshWallTime() - t_mark;

    t_mark = MFTLRemeshWallTime();
    MFTL_RemeshMigrationReport mig_report;
    if (MFTLRemeshMigrateShifted(&mig, ens, group, shift, &mig_report) != MFTL_SUCCESS)
    {
      /* Every later phase would run on a partition nobody agrees about, and the
       * closing sequence would renumber ids against it. Stop here and report. */
      status = MFTL_ERROR;
      break;
    }
    st->t_migrate += MFTLRemeshWallTime() - t_mark;
    st->n_entities_migrated += mig_report.n_moved;
    if (mig_report.performed)
      st->n_migrations++;
    else
      st->n_migrations_skipped++;

    /* ── 2. Who may this rank touch? ────────────────────────────────────────
     * Local, exact, and read straight off the partition the migration produced.
     * Nothing here communicates. */
    t_mark = MFTLRemeshWallTime();
    int extra[MFTL_REMESH_EXTRA_INTS] = {0};
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTLRemeshMarkFrozen(ens->local_objects[o]);
      MFTLDilateFrozenBand(ens->local_objects[o], params->freeze_dilation_rings);
      count_hidden_edges(ens->local_objects[o], params, &extra[MFTL_REMESH_EXTRA_HIDDEN_LONG], &extra[MFTL_REMESH_EXTRA_HIDDEN_SHORT]);
    }

    /* A RUNNING MAXIMUM over every phase, accumulated locally and reduced once on
     * the way out. Measuring only the first phase was the obvious cheap thing and
     * it is wrong in one direction that matters: a COARSENING pass makes the mesh
     * longer-edged as it runs, so the band grows, and phase 0 understates it. On
     * the [-1,1]^3 fixture at target 0.40 the first phase reports 0.216 and the
     * band actually reaches 0.375 -- which is the difference between "the
     * precondition holds" and the pass genuinely failing to offer everything.
     *
     * Costing nothing extra is what makes the running maximum affordable: the
     * two doubles ride the profile reduction at the end of the call, so the pass
     * pays no collective for them at all. */
    for (int o = 0; o < ens->n_local_objects; ++o) local_band_probe(ens->local_objects[o], dom, shift, band_probe);
    st->t_freeze += MFTLRemeshWallTime() - t_mark;

    /* The id counters BEFORE the sweeps -- everything at or above these was
     * created by this phase, which is how step 4 finds them without the
     * operators having to keep a list. */
    gid_scratch_reserve(&gid, ens->n_local_objects);
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      const MFTL_Object *const obj = ens->local_objects[o];
      gid.watermark[2 * o + 0] = obj->next_vertex_gid;
      gid.watermark[2 * o + 1] = obj->next_triangle_gid;
    }

    /* ── 3. The untouched serial kernel, on the interior ────────────────────
     *
     * The FULL iteration count, not one iteration. Within a phase the frozen set
     * is fixed, so the interior can be converged completely before there is any
     * reason to pay for another migration -- which is what keeps the migration
     * count proportional to the number of PHASES rather than to the number of
     * remeshing iterations. */
    t_mark = MFTLRemeshWallTime();
    int local_ops = 0;

    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTL_RemeshStats one;
      memset(&one, 0, sizeof(one));

      local_ops += MFTLRemeshIterate(ens->local_objects[o], params, &one, params->iterations);
      remesh_stats_accumulate(st, &one);
    }
    st->t_sweep += MFTLRemeshWallTime() - t_mark;
    extra[MFTL_REMESH_EXTRA_OPS] = local_ops;

    /* ── 4. Globally unique ids for what the sweeps created, and the verdict ─
     *
     * One Exscan and one Allreduce, overlapped, carrying both -- the ids because
     * the next migration identifies entities by them, the counters because the
     * loop cannot end without them. Merging the two is worth stating: the
     * convergence test is otherwise a 2-int Allreduce, which is pure latency and
     * does not get cheaper for being small. */
    t_mark = MFTLRemeshWallTime();
    assign_phase_gids(ens, &gid, extra);
    st->t_reduce += MFTLRemeshWallTime() - t_mark;

    const int global_ops = extra[MFTL_REMESH_EXTRA_OPS];
    const int global_hidden_long = extra[MFTL_REMESH_EXTRA_HIDDEN_LONG];
    const int global_hidden_short = extra[MFTL_REMESH_EXTRA_HIDDEN_SHORT];
    const int global_hidden = global_hidden_long + global_hidden_short;

    /* Reported as the GLOBAL residual of the last phase, not as a sum over
     * phases: the useful question is "what is this placement still hiding", and
     * a sum answers "was anything ever late", which is normal and
     * uninformative. */
    st->n_edges_hidden = global_hidden;
    st->n_edges_hidden_long = global_hidden_long;
    st->n_edges_hidden_short = global_hidden_short;

    /* ── 5. Converge ────────────────────────────────────────────────────────
     *
     * TWO exits, and the second is what makes the first safe to be fast.
     *
     * FAST, and a complete proof on its own: nothing was performed anywhere AND
     * no out-of-band edge had a frozen endpoint. The second half is the
     * interesting one -- it says every edge outside the length band was OFFERED
     * to a rank that was allowed to act on it, and the first half says that rank
     * declined. Declining is the approval predicates doing their job (a short
     * edge the link condition may not collapse stays short), so there is nothing
     * left for another placement to reveal.
     *
     * SLOW, for when the fast one keeps missing: a full table of consecutive
     * silent placements. `hidden_short` has false-alarm modes -- an out-of-band
     * edge that no operator would fix anyway, sitting on this placement's band --
     * and such an edge would otherwise keep the driver running to its budget and
     * then report a convergence failure that is not one. `n_distinct` silent
     * placements is a covering set by construction: with h > 4*delta every
     * entity is interior in at least one of them, and it was silent there. It is
     * n_distinct rather than eight because two table entries that differ only on
     * an uncut axis are the same placement and prove the same thing twice.
     *
     * The cost of the slow exit is up to n_distinct quiet phases, and a quiet
     * phase is cheap by the same token -- MFTLRemeshIterate breaks out after its
     * first iteration when that iteration performs nothing, and the migration in
     * front of it is skipped when the placement is the one already in force. */
    if (global_ops == 0 && global_hidden == 0)
    {
      st->converged = MFTL_TRUE;
      break;
    }

    if (global_ops == 0)
    {
      /* The streak may only certify while the EXACT half of the certificate is
       * clean. `hidden_long` has no false alarms -- a long edge offered to any
       * rank is split, unconditionally -- so a non-zero value is a placement
       * genuinely failing to reach work, and eight silent phases say nothing
       * about it. `hidden_short` is the one that legitimately never clears, and
       * it is the one the streak exists to absorb.
       *
       * Before this test the streak absorbed both, which is how the driver could
       * report converged alongside dozens of permanently hidden edges. */
      if (global_hidden_long != 0)
      {
        /* Silent, but with work this placement is known to be hiding. Not
         * evidence, and it RESETS rather than merely failing to count: the
         * covering argument is about CONSECUTIVE placements, so a streak
         * assembled from phases either side of a blocked one would not cover. */
        clean_streak = 0;
      }
      else if (++clean_streak >= n_distinct)
      {
        st->converged = MFTL_TRUE;
        break;
      }
    }
    else
    {
      clean_streak = 0;
    }
  }

  /* ── 6. Hand the mesh back the way the flow solver expects it ─────────────
   *
   * Canonical planes, dense ids, one migration, and the geometry the
   * postcondition promises. The order of the middle two is forced: the
   * renumbering makes every ghost TRIANGLE id stale, and the migration is
   * exactly what replaces them -- so it has to come after. Ghost VERTICES are
   * kept by a migration rather than re-received, which is why they need the
   * explicit exchange inside redensify_global_ids and cannot rely on this. */
  t_mark = MFTLRemeshWallTime();
  if (redensify_global_ids(ens, &gid) != MFTL_SUCCESS) status = MFTL_ERROR;
  st->t_reduce += MFTLRemeshWallTime() - t_mark;

  t_mark = MFTLRemeshWallTime();
  MFTLEnsembleComputeTriangleGeometry(ens);
  st->t_geometry += MFTLRemeshWallTime() - t_mark;

  /* The canonical placement is the PRODUCTION rule, so the closing migration
   * goes through MFTLRedistribute and not through the shifted path. That is not
   * a tidiness point: it is what makes "the mesh the flow solver gets back is
   * partitioned exactly as MFTLRedistribute would partition it" true by
   * construction rather than by an argument about a zero shift. */
  t_mark = MFTLRemeshWallTime();
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) status = MFTL_ERROR;
  st->t_migrate += MFTLRemeshWallTime() - t_mark;
  st->n_migrations++;

  clear_frozen(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  /* Redistribution leaves `tris.area/normal/center` unspecified by design
   * (mftl_communication.h). MFTLRemeshObject promises its caller current
   * triangle geometry, so this one promises the same thing rather than making
   * the parallel entry point a different contract. */
  t_mark = MFTLRemeshWallTime();
  MFTLEnsembleComputeTriangleGeometry(ens);
  st->t_geometry += MFTLRemeshWallTime() - t_mark;

  /* ── 7. Publish the profile ───────────────────────────────────────────────
   *
   * The MAXIMUM over ranks, not the sum and not the mean. The phase loop is
   * barrier-synchronised by the id collective, so the wall clock a caller
   * actually waits is set by the slowest rank in each section; a mean would
   * report a comfortable number for a pass that spent half its time with seven
   * ranks idle, which is the failure this measurement exists to expose.
   *
   * UNCONDITIONAL, even when the caller passed no stats struct: `st` is the
   * local sink in that case, but the collective must still be entered by every
   * rank. Gating a collective on a caller's argument is a deadlock waiting for
   * the one caller who passes NULL. */
  st->t_total = MFTLRemeshWallTime() - t0_call;
  MFTLRemeshAgreeProfile(dom, st, band_probe, 2);

  st->band_measured = band_probe[0];
  st->band_bound = (MFTL_REMESH_BAND_EDGES + (double) MFTL_MAX(params->freeze_dilation_rings, 0)) * band_probe[1];

  /* Warned about AFTER the fact, and that is not a compromise. The band is a
   * property of the mesh as the pass actually saw it, so the accurate number
   * only exists now; and the verdict a caller acts on is `converged` plus
   * `n_edges_hidden_long`, which this call cannot change. Warning early from an
   * estimate is what produced a message that was wrong two times in three. */
  /* NOT when the fallback already handled it. check_band_fits compares against
   * the FINE box, which is by construction still too narrow on a pass that
   * coarsened precisely because it was -- so leaving it unguarded makes the
   * successful repair announce itself as the failure it repaired. The band
   * argument holds against `group[k] * h_k` after coarsening, and
   * report_fallback has already said so with the factors quoted. */
  if (st->n_coarse_groups <= 1) check_band_fits(dom, params, st->band_measured, st->band_bound);

  gid_scratch_free(&gid);
  MFTLRemeshMigratorFree(&mig);
  return status;
}
