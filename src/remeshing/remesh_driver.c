/**
 * @file remesh_driver.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief The remeshing pass: sweep order, convergence, and what is refreshed
 *        when.
 *
 * ## The order of the sweeps
 *
 *   DETECT -> SPLIT -> COLLAPSE -> COMPACT -> FLIP -> DETECT -> SMOOTH
 *
 * - SPLIT before COLLAPSE: splitting cannot fail (no link condition, no fold
 *   test), so the collapse sweep then sees edges that are genuinely short rather
 *   than short relative to over-long neighbours.
 * - COMPACT right after COLLAPSE: flip and smoothing never see a dead entity, so
 *   the smoother can use MFTLComputeVertexNormals() over the whole pool without
 *   accumulating dead triangles into live normals.
 * - FLIP before SMOOTH: the valence criterion reads the connectivity the
 *   previous sweeps produced; smoothing does not change it.
 * - DETECT twice per iteration: splits and collapses change which edges exist,
 *   and a vertex the smoother wrongly believes free (e.g. the midpoint of a
 *   split rim edge) slides off its curve.
 *
 * Each sweep bounds its loop by the half-edge count taken BEFORE it: the halves
 * of a split edge are not re-examined in the same sweep (a long edge would
 * otherwise burst into many tiny ones), but in the next iteration, after
 * smoothing.
 */

#include "mftl_remeshing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_comm.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Sweeps
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Has the spanning pass already offered this edge, and has nothing moved
 *        at it since? The half-edge AND both its ends carry #MFTL_FLAG_DONE.
 *
 * Only the spanning driver sets the flag, so a serial sweep never takes the
 * branch; the extra test is three loads of records the sweep reads anyway. Both
 * halves are needed. The edge's own flag: two DONE ends may have been unfrozen
 * in two different placements, and the edge between them was then never
 * offered. The ends' flags: a vertex reopened for smoothing (reopen_touched())
 * moves, and every edge at it changes length -- with the edge flag alone a
 * one-shot refinement at np = 4 ended with 84 out-of-band edges after its
 * second placement against 69.
 *
 * @param obj  The mesh.
 * @param h    Half-edge index.
 * @param gate Whether the pass applies the rule at all.
 * @return `true` if the edge must not be offered again.
 */
static inline bool done_blocks(const MFTL_Object *obj, int h, bool gate)
{
  if (!gate) return MFTL_FALSE;
  return (obj->tris.he[h].flags & obj->verts.rec[MFTLHeOrigin(obj, h)].flags & obj->verts.rec[obj->tris.he[h].vertex_idx].flags & MFTL_FLAG_DONE) != 0;
}

/**
 * @brief Splits every approved edge, once per undirected edge.
 *
 * The parallel freeze gate is tested first (two flag tests, against an
 * approval's fan walk). Refusals are NOT counted as blocked work: a rank sees
 * ghosts with incomplete one-rings, approves a collapse the owner would reject,
 * and would report phantom blocked operations (3 on a converged np = 4 sphere).
 * Convergence is instead certified by `count_unoffered_edges` in remesh_parallel.c,
 * which needs no fan walk.
 *
 * @param obj    The mesh.
 * @param params Supplies `max_edge_length_sq`.
 * @param gate   Apply the spanning pass's DONE rule (done_blocks()).
 * @return Splits performed.
 */

static int sweep_split(MFTL_Object *obj, const MFTL_RemeshParameters *params, bool gate)
{
  const int n_he = 3 * obj->tris.n;  // snapshot: see the file header
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_pair(obj, h) || done_blocks(obj, h, gate)) continue;
    if (!MFTLSplitApproved(obj, h, params)) continue;
    if (MFTLSplitEdge(obj, h, params) != MFTL_ERROR) count++;
  }
  return count;
}

/**
 * @brief Collapses every approved edge, once per undirected edge.
 * @param      obj    The mesh.
 * @param      params Supplies `min_edge_length_sq` and the safety thresholds.
 * @param[out] stats  May be #MFTL_NULL. The rejection counters are accumulated here.
 * @param      gate   Apply the spanning pass's DONE rule (done_blocks()).
 * @return Collapses performed.
 */
static int sweep_collapse(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, bool gate)
{
  const int n_he = 3 * obj->tris.n;
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_collapse(obj, h) || done_blocks(obj, h, gate)) continue;

    int reason = MFTL_REMESH_REJECT_NONE;

    if (MFTLCollapseApproved(obj, h, params, &reason))
    {
      if (MFTLCollapseEdge(obj, h, params) == MFTL_SUCCESS) count++;
      continue;
    }

    if (!stats) continue;

    switch (reason)
    {
      case MFTL_REMESH_REJECT_LINK: stats->n_collapse_rejected_link++; break;
      case MFTL_REMESH_REJECT_FOLD: stats->n_collapse_rejected_fold++; break;
      case MFTL_REMESH_REJECT_FEATURE: stats->n_collapse_rejected_feature++; break;
      default: break;
    }
  }
  return count;
}

/**
 * @brief Flips every edge whose flip strictly lowers the valence deviation.
 * @param      obj    The mesh.
 * @param      params Supplies `max_normal_deviation_cos`.
 * @param[out] stats  May be #MFTL_NULL.
 * @param      gate   Apply the spanning pass's DONE rule (done_blocks()).
 * @return Flips performed.
 */
static int sweep_flip(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, bool gate)
{
  const int n_he = 3 * obj->tris.n;
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_pair(obj, h) || done_blocks(obj, h, gate)) continue;

    int reason = MFTL_REMESH_REJECT_NONE;

    if (MFTLFlipApproved(obj, h, params, &reason))
    {
      if (MFTLFlipEdge(obj, h) == MFTL_SUCCESS) count++;
    }
    else if (stats && reason == MFTL_REMESH_REJECT_FOLD)
    {
      stats->n_flip_rejected_fold++;
    }
  }
  return count;
}

/**
 * @brief Clears the per-sweep flags so the next iteration starts from scratch.
 *
 * @param obj The mesh. #MFTL_FLAG_VISITED is cleared on every entity, so the
 *            next sweep starts from a clean slate.
 */
static void clear_visited(MFTL_Object *obj)
{
  const int n_he = 3 * obj->tris.n;
  for (int h = 0; h < n_he; ++h) obj->tris.he[h].flags &= ~MFTL_FLAG_VISITED;
  for (int v = 0; v < obj->verts.n; ++v) obj->verts.rec[v].flags &= ~MFTL_FLAG_VISITED;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTLMeanEdgeLength(const MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int n_he = 3 * obj->tris.n;
  if (n_he == 0) return 0.0;

  double sum = 0.0;
  int count = 0;

  /* Over UNDIRECTED edges. Averaging half-edges gives the same answer on a closed
   * mesh and a wrong one on an open surface, where a boundary edge has one half
   * and an interior edge has two -- so the boundary would be weighted half. */
  for (int h = 0; h < n_he; ++h)
  {
    const int tw = obj->tris.he[h].twin_idx;
    if (tw != MFTL_ERROR && h > tw) continue;

    sum += MFTLGetEdgeLength(obj, h);
    count++;
  }

  return (count > 0) ? sum / (double) count : 0.0;
}

double MFTLEnsembleMeanEdgeLength(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);

  double acc[2] = {0.0, 0.0};

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const int n_he = 3 * obj->tris.n;

    for (int h = 0; h < n_he; ++h)
    {
      const int t = MFTLHeTriIdx(h);
      if (t >= obj->tris.n_local) continue;  // a ghost's owner counts it

      const int tw = obj->tris.he[h].twin_idx;
      if (tw != MFTL_ERROR)
      {
        const int to = MFTLHeTriIdx(tw);

        /* Both mine: the smaller half-edge counts, as in the serial form. Split
         * across ranks: the smaller triangle global id counts, which is a rule
         * both sides evaluate to the same answer without asking each other. */
        if (to < obj->tris.n_local)
        {
          if (h > tw) continue;
        }
        else if (obj->tris.rec[t].global_id > obj->tris.rec[to].global_id)
        {
          continue;
        }
      }

      acc[0] += MFTLGetEdgeLength(obj, h);
      acc[1] += 1.0;
    }
  }

  MFTLRemeshAgreeDouble(&ens->domain, acc, 2);
  return (acc[1] > 0.0) ? acc[0] / acc[1] : 0.0;
}

/**
 * @brief Corners of every triangle an operation of this sweep set touched lose
 *        #MFTL_FLAG_DONE, so the ONCE pass smooths them again and offers every
 *        edge at them again (done_blocks() needs both ends DONE).
 *
 * The operators mark what they rewrote with #MFTL_FLAG_VISITED (a split its
 * new vertex and triangles, a collapse its survivor, a flip its two triangles).
 * Without this, an operation a later placement performs next to finished
 * vertices would stay unsmoothed until the next call: min q fell to 0.28-0.41
 * in the identity study.
 *
 * @param obj The mesh, derived topology current.
 */
static void reopen_touched(MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    int vv[3];
    MFTLGetTriangleVertices(obj, t, vv);
    const int hf = obj->tris.he[3 * t].flags | obj->tris.he[3 * t + 1].flags | obj->tris.he[3 * t + 2].flags;
    const int vf = obj->verts.rec[vv[0]].flags | obj->verts.rec[vv[1]].flags | obj->verts.rec[vv[2]].flags;
    if ((hf | vf) & MFTL_FLAG_VISITED)
      for (int k = 0; k < 3; ++k) obj->verts.rec[vv[k]].flags &= ~MFTL_FLAG_DONE;
  }
}

/**
 * @brief Detect, split, collapse, compact, flip: the edge half of one iteration.
 * @param         obj    The mesh.
 * @param         params The parameter block.
 * @param[in,out] st     Statistics.
 * @param         gate   The spanning DONE rule.
 * @param[out]    ops    {splits, collapses, flips}.
 */
static void edge_sweeps(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st, bool gate, int ops[3])
{
  ops[0] = sweep_split(obj, params, gate);
  ops[1] = sweep_collapse(obj, params, st, gate);

  /* Unconditional, not "if anything died": the split sweep left the derived
   * arrays invalid too, and MFTLRemeshCompactObject rebuilds them on the way
   * out whether or not it moved anything. */
  MFTLRemeshCompactObject(obj, st);

  ops[2] = sweep_flip(obj, params, st, gate);
  if (ops[2] > 0) MFTLObjectBuildDerived(obj);
}

/**
 * @brief The iteration loop of MFTLRemeshIterate() and of the spanning ONCE mode.
 * @param         obj        The mesh.
 * @param         params     The parameter block.
 * @param[in,out] st         Statistics.
 * @param         iterations Outer iterations.
 * @param         project    Project once after the last smoothing (the serial call).
 * @param         once       The spanning DONE rules.
 * @return Operations performed.
 */
static int iterate(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st, int iterations, bool project, bool once)
{
  int total_ops = 0;

  if (obj->tris.n == 0) return 0;

  /* Every operator reads and writes the half-edge records; the derived arrays are
   * rebuilt by the compaction that follows the collapse sweep. Building them here
   * makes the object valid for the very first MFTLDetectFeatures. */
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  /* Curvature sizing is computed by the CALLER (MFTLRemeshPrepareSizing()),
   * because the spanning driver computes it once per pass and carries it. */
  MFTL_ASSERT((params->sizing != MFTL_REMESH_SIZING_CURVATURE || obj->verts.has_sizing) &&
              "MFTLRemeshIterate: curvature sizing without verts.sizing -- MFTLRemeshPrepareSizing first");

  /* The projection's reference: the mesh as THIS call received it, copied once,
   * after the first feature detection, which decides which corners may lift;
   * used once, after the last smoothing sweep (remesh_project.c). */
  MFTL_RemeshReference ref;
  bool have_ref = MFTL_FALSE;
  project = project && params->projection == MFTL_REMESH_PROJECT_PHONG;

  for (int iter = 0; iter < iterations; ++iter)
  {
    st->iterations = iter + 1;

    MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, st);

    if (iter == 0 && project)
    {
      const double t0 = MFTLRemeshWallTime();
      MFTLRemeshReferenceFromObject(&ref, obj, params);
      have_ref = MFTL_TRUE;
      st->t_reference += MFTLRemeshWallTime() - t0;
      st->reference_bytes = MFTL_MAX(st->reference_bytes, ref.bytes);
    }

    int ops[3];
    edge_sweeps(obj, params, st, once, ops);
    if (once) reopen_touched(obj);

    clear_visited(obj);

    /* Detected AGAIN, against the mesh that is about to be smoothed rather than
     * the one the sweeps started from. The three sweeps in between create
     * vertices that the first classification never saw: a split of a rim edge
     * produces a midpoint that IS on the rim, and a vertex the smoother believes
     * to be free gets the 2D tangent-plane treatment and slides straight off it.
     * The surface then grows a serrated notch, which is invisible to every
     * structural check -- the result is a perfectly valid manifold with a
     * perfectly valid boundary, just not the boundary it started with.
     *
     * The cost is one geometry pass and two linear passes per iteration. Handing
     * the classification forward from the first call instead would mean every
     * operator maintaining it incrementally, which is the arrangement that leaves
     * a smoother three sweeps out of date. */
    MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, st);
    MFTLRemeshSmooth(obj, params, params->smooth_iterations, st);

    st->n_split += ops[0];
    st->n_collapse += ops[1];
    st->n_flip += ops[2];
    const int n_ops = ops[0] + ops[1] + ops[2];
    total_ops += n_ops;

    if (n_ops <= params->min_convergence_ops) break;
  }

  /* ONCE, after the call's last smoothing sweep: the smoother redistributes the
   * vertices over all its sweeps and the projection then puts every free one
   * back on the reference surface. Projecting after every sweep measured the
   * same within the run-to-run spread at three times the cost
   * (docs/analysis.md §14.10). */
  if (have_ref)
  {
    MFTLRemeshProjectObject(obj, &ref, params, st);
    MFTLRemeshReferenceFree(&ref);
  }
  return total_ops;
}

int MFTLRemeshIterate(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st, int iterations)
{
  return iterate(obj, params, st, iterations, MFTL_TRUE, MFTL_FALSE);
}

int MFTLRemeshIterateSpanning(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st, int iterations, bool once)
{
  return iterate(obj, params, st, iterations, MFTL_FALSE, once);
}

int MFTLRemeshEdgeSweeps(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st)
{
  if (obj->tris.n == 0) return 0;
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
  MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, st);
  int ops[3];
  edge_sweeps(obj, params, st, MFTL_TRUE, ops);
  clear_visited(obj);
  st->n_split += ops[0];
  st->n_collapse += ops[1];
  st->n_flip += ops[2];
  return ops[0] + ops[1] + ops[2];
}

int MFTLRemeshObject(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);
  MFTL_ASSERT(obj->tris.n_local == obj->tris.n && obj->verts.n_local == obj->verts.n &&
              "MFTLRemeshObject: object holds ghosts. Collapsing an edge whose far endpoint is owned elsewhere "
              "needs agreement, so a 'serial' remesh of a partitioned object silently diverges between ranks. "
              "The distributed entry point is MFTLRemeshEnsemble.");

  /* The stats describe THIS call, so they are zeroed here rather than left to the
   * caller. Accumulating into whatever the caller's struct happened to hold would
   * make a second pass over the same mesh report the sum of both. */
  MFTL_RemeshStats local;
  MFTL_RemeshStats *const st = stats ? stats : &local;
  memset(st, 0, sizeof(*st));

  if (obj->tris.n == 0) return MFTL_SUCCESS;

  /* The sweeps are greedy in pool order and every fan sum is rounded in the
   * order its walk takes from `out_he_idx`, so the same mesh in two pool orders
   * remeshes to two meshes. Ordering by global id first makes the result a
   * function of the mesh alone -- which is what lets the dispersed driver, whose
   * migration hands a body over in arrival order, reproduce this call to the bit.
   * After the first call the pools are already sorted (the renumbering below
   * sets id == slot), and this is two read-only scans and one `out_he_idx` pass. */
  MFTLObjectCanonicalOrder(obj);

  /* The curvature sizing field, ONCE per call, on the mesh as handed in; the
   * operators carry it from here (remesh_sizing.c). A no-op otherwise. */
  if (MFTLRemeshPrepareSizing(obj, params) != MFTL_SUCCESS) return MFTL_ERROR;

  MFTLRemeshIterate(obj, params, st, params->iterations);

  if (params->renumber_global_ids) MFTLRemeshRenumberGlobalIds(obj);

  /* Postcondition: the caller gets an object it can stream immediately. The
   * derived topology is already current -- compaction and the flip sweep rebuild
   * it, and neither smoothing nor the id renumbering changes connectivity -- so
   * only `tris.area/normal/center` are left to fill; the operators leave them
   * stale by design because none of them read them. */
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  return MFTL_SUCCESS;
}

int MFTLRemeshEnsemble(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  /* One rank stays on the path below, and not merely as an optimisation: the
   * phase schedule migrates, renumbers ids collectively and rebuilds the halo,
   * all of which are identities at nproc == 1 but none of which are bit-level
   * identities -- a redistribution permutes the pools. Keeping the serial answer
   * reproducible is worth the branch. */
  if (ens->domain.nproc > 1)
  {
    /* The parameters are agreed ONCE per call, at the first collective of
     * whichever path runs (MFTLRemeshAgreeParams()). Every path's first
     * collective is that check, so two ranks that read different strategies
     * still meet in it and both refuse. */
    /* AUTO resolves COLLECTIVELY -- MFTLRemeshChooseStrategy measures the bodies
     * with two Allreduces and every rank therefore reads the same ratio. That is
     * the load-bearing property of this branch, not the choice itself: two ranks
     * entering different drivers is an immediate deadlock, and nothing further
     * down would report it. */
    MFTL_RemeshParallelStrategy strategy = params->parallel_strategy;

    if (strategy != MFTL_REMESH_PARALLEL_AUTO)
    {
      if (strategy == MFTL_REMESH_PARALLEL_DISPERSED) return MFTLRemeshDispersedPass(ens, params, stats, "MFTLRemeshEnsemble");
      return MFTLRemeshSpanningPass(ens, params, stats, "MFTLRemeshEnsemble");
    }

    /* AUTO measures the bodies to decide, and the dispersed driver needs exactly
     * the same measurement to route them -- so it is taken ONCE here and handed
     * on. Doing it the obvious way (choose, then call the public driver) makes
     * the pass pay four per-body reductions where two do, and the first caller
     * then discards the centroids the second one recomputes.
     *
     * GLOBAL body count for the guard: everything below is collective. The
     * measurement itself is indexed by LOCAL SLOT -- see MFTL_RemeshMeasurement. */
    if (ens->n_bodies_global < 1) return MFTLRemeshSpanningPass(ens, params, stats, MFTL_NULL);

    MFTL_RemeshMeasurement measure;
    MFTLRemeshMeasurementInit(&measure, ens);

    MFTLEnsembleComputeTriangleGeometry(ens);
    if (MFTLRemeshAgreeParams(&ens->domain, params, "MFTLRemeshEnsemble") != MFTL_SUCCESS)
    {
      MFTLRemeshMeasurementFree(&measure);
      if (stats) memset(stats, 0, sizeof(*stats));
      return MFTL_ERROR;
    }
    strategy = MFTLRemeshChooseStrategyMeasured(ens, params, &measure);

    const int rc = (strategy == MFTL_REMESH_PARALLEL_DISPERSED) ? MFTLRemeshEnsembleDispersedMeasured(ens, params, stats, &measure)
                                                                : MFTLRemeshSpanningPass(ens, params, stats, MFTL_NULL);
    MFTLRemeshMeasurementFree(&measure);
    return rc;
  }

  /* One rank: the band test alone, no MPI. */
  if (ens->n_bodies_global > 0 && MFTLRemeshAgreeParams(&ens->domain, params, "MFTLRemeshEnsemble") != MFTL_SUCCESS)
  {
    if (stats) memset(stats, 0, sizeof(*stats));
    return MFTL_ERROR;
  }

  if (stats)
  {
    memset(stats, 0, sizeof(*stats));

    /* A serial pass IS converged, by the only definition the field has: the mesh
     * is as converged as the serial remesher would have made it, because it is
     * what the serial remesher made. Leaving the flag false because "no phase
     * driver ran" would make every nproc == 1 run report a convergence failure,
     * which is what a solver logs and acts on. */
    stats->converged = MFTL_TRUE;
    stats->n_phases = 1;
    stats->strategy = MFTL_REMESH_PARALLEL_SPANNING;
    stats->n_coarse_groups = 1;
    stats->n_distinct_placements = 1;  // one rank, one placement, and it is the canonical one
  }

  int status = MFTL_SUCCESS;

  for (int i = 0; i < ens->n_local_objects; ++i)
  {
    MFTL_RemeshStats one;
    memset(&one, 0, sizeof(one));

    if (MFTLRemeshObject(ens->local_objects[i], params, &one) != MFTL_SUCCESS) status = MFTL_ERROR;

    if (!stats) continue;

    remesh_stats_accumulate(stats, &one);
  }

  MFTLEnsembleUpdateGlobalCounts(ens);
  return status;
}
