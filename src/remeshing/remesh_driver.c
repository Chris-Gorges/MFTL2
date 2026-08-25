/**
 * @file remesh_driver.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief The remeshing pass: sweep order, convergence, and what is refreshed when.
 *
 * ## THE ORDER OF THE SWEEPS IS NOT ARBITRARY
 *
 *   DETECT   -> SPLIT -> COLLAPSE -> COMPACT -> FLIP -> SMOOTH
 *
 * SPLIT before COLLAPSE, because splitting is the operation that cannot fail:
 * it needs no link condition and no fold test, so doing it first gives the
 * collapse sweep a mesh where the short edges are genuinely short rather than
 * merely short relative to their over-long neighbours.
 *
 * COMPACT between COLLAPSE and FLIP, which is the change from the previous
 * version. It compacted once at the very end of the whole call, so the flip and
 * smoothing sweeps ran over a pool containing dead entities and had to defend
 * against them individually. Here they never see one, and the payoff is not
 * tidiness: the smoother can call MFTLObjectRefreshFrames() and
 * MFTLComputeVertexNormals(), which stream the whole pool and would otherwise
 * accumulate dead triangles into live vertices' normals.
 *
 * FLIP before SMOOTH, because the valence criterion is combinatorial: it reads
 * the connectivity the previous two sweeps produced, and smoothing does not
 * change connectivity. Running it after would just delay its effect by an
 * iteration.
 *
 * DETECT at the top of every iteration rather than once per call, because splits
 * and collapses change which edges exist. Detecting once per call means that from
 * the first split onwards the smoother classifies vertices against a feature set
 * describing a mesh that no longer exists.
 *
 * ## WHY EVERY SWEEP TAKES A FRESH SNAPSHOT OF THE HALF-EDGE COUNT
 *
 * MFTLSplitEdge appends triangles, so 3 * tris.n grows DURING the split sweep.
 * The bound is captured once, before the loop, on purpose: the half-edges the
 * sweep creates are the two halves of an edge it has just decided was too long,
 * and re-examining them in the same sweep is how a remesher turns one long edge
 * into a burst of tiny ones. They are examined on the next iteration, where they
 * are measured against a mesh that has been smoothed in between.
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
 * THE PARALLEL GATE IS CHEAP AND IS TESTED FIRST; THE CONVERGENCE CERTIFICATE IS
 * MEASURED SOMEWHERE ELSE ENTIRELY, AND THAT SEPARATION IS DELIBERATE.
 *
 * The obvious design is to count here: run the approval, and if the gate then
 * refuses, record "the band hid a real operation". It does not work, and the way
 * it fails is instructive. A rank examines every half-edge in its pool, GHOSTS
 * INCLUDED, and an approval predicate walks fans -- but a fan is only guaranteed
 * complete around an OWNED vertex (inv_halo_is_vertex_complete). Deep inside a
 * neighbour's territory this rank sees a partial one-ring, approves a collapse
 * the owner correctly rejects on the link condition, and reports blocked work
 * that does not exist. Measured: 3 phantom blocked operations that never clear,
 * on a np = 4 sphere whose mesh was in fact fully converged.
 *
 * So the gate here is only a gate, and it goes FIRST because it is two flag
 * tests against an approval's fan walk. What the driver actually measures is
 * `count_hidden_edges` in remesh_parallel.c: out-of-band edges that no rank
 * could act on under this placement of the cut planes. That question needs no
 * fan and no approval, so a partial view cannot invent an answer to it.
 *
 * @brief Splits every approved edge, once per undirected edge.
 * @param obj    The mesh.
 * @param params Supplies `max_edge_length_sq`.
 * @return Splits performed.
 */

static int sweep_split(MFTL_Object *obj, const MFTL_RemeshParameters *params)
{
  const int n_he = 3 * obj->tris.n;  // snapshot: see the file header
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_pair(obj, h)) continue;
    if (!MFTLSplitApproved(obj, h, params)) continue;
    if (MFTLSplitEdge(obj, h) != MFTL_ERROR) count++;
  }
  return count;
}

/**
 * @brief Collapses every approved edge, once per undirected edge.
 * @param      obj    The mesh.
 * @param      params Supplies `min_edge_length_sq` and the safety thresholds.
 * @param[out] stats  May be #MFTL_NULL. The rejection counters are accumulated here.
 * @return Collapses performed.
 */
static int sweep_collapse(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  const int n_he = 3 * obj->tris.n;
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_collapse(obj, h)) continue;

    int reason = MFTL_REMESH_REJECT_NONE;

    if (MFTLCollapseApproved(obj, h, params, &reason))
    {
      if (MFTLCollapseEdge(obj, h) == MFTL_SUCCESS) count++;
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
 * @return Flips performed.
 */
static int sweep_flip(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats)
{
  const int n_he = 3 * obj->tris.n;
  int count = 0;

  for (int h = 0; h < n_he; ++h)
  {
    if (freeze_blocks_pair(obj, h)) continue;

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

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
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

int MFTLRemeshIterate(MFTL_Object *obj, const MFTL_RemeshParameters *params, MFTL_RemeshStats *st, int iterations)
{
  int total_ops = 0;

  if (obj->tris.n == 0) return 0;

  /* Every operator reads and writes the half-edge records; the derived arrays are
   * rebuilt by the compaction that follows the collapse sweep. Building them here
   * makes the object valid for the very first MFTLDetectFeatures. */
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  for (int iter = 0; iter < iterations; ++iter)
  {
    st->iterations = iter + 1;

    MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, st);

    const int n_split = sweep_split(obj, params);
    const int n_collapse = sweep_collapse(obj, params, st);

    /* Unconditional, not "if anything died": the split sweep left the derived
     * arrays invalid too, and MFTLRemeshCompactObject rebuilds them on the way
     * out whether or not it moved anything. */
    MFTLRemeshCompactObject(obj, st);

    const int n_flip = sweep_flip(obj, params, st);
    if (n_flip > 0) MFTLObjectBuildDerived(obj);

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

    MFTLSmoothTangential(obj, params, params->smooth_iterations);

    st->n_split += n_split;
    st->n_collapse += n_collapse;
    st->n_flip += n_flip;
    total_ops += n_split + n_collapse + n_flip;

    if (n_split + n_collapse + n_flip <= params->min_convergence_ops) break;
  }

  return total_ops;
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

  MFTLRemeshIterate(obj, params, st, params->iterations);

  if (params->renumber_global_ids) MFTLRemeshRenumberGlobalIds(obj);

  /* Postcondition: the caller gets an object it can stream immediately. Both
   * derived bits are already set (compaction, then the smoother's closing
   * refresh); this fills `tris.area/normal/center`, which the operators leave
   * stale by design because none of them read it. */
  MFTLObjectBuildDerived(obj);
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
    /* AUTO resolves COLLECTIVELY -- MFTLRemeshChooseStrategy measures the bodies
     * with two Allreduces and every rank therefore reads the same ratio. That is
     * the load-bearing property of this branch, not the choice itself: two ranks
     * entering different drivers is an immediate deadlock, and nothing further
     * down would report it. */
    MFTL_RemeshParallelStrategy strategy = params->parallel_strategy;

    if (strategy != MFTL_REMESH_PARALLEL_AUTO)
    {
      if (strategy == MFTL_REMESH_PARALLEL_DISPERSED) return MFTLRemeshEnsembleDispersed(ens, params, stats);
      return MFTLRemeshEnsembleParallel(ens, params, stats);
    }

    /* AUTO measures the bodies to decide, and the dispersed driver needs exactly
     * the same measurement to route them -- so it is taken ONCE here and handed
     * on. Doing it the obvious way (choose, then call the public driver) makes
     * the pass pay four Allreduces where two do, two of them carrying
     * 7*num_objects doubles that the first caller then discards. */
    const int n_bodies = ens->num_objects;
    if (n_bodies < 1) return MFTLRemeshEnsembleParallel(ens, params, stats);

    double *const measure = (double *) MFTLMallocOrDie((size_t) (5 * n_bodies) * sizeof(double), "remesh strategy measurement");
    double *const centroid = measure;
    double *const radius = measure + 3 * n_bodies;
    double *const weight = measure + 4 * n_bodies;

    strategy = MFTLRemeshChooseStrategyMeasured(ens, params, centroid, radius, weight);

    const int rc = (strategy == MFTL_REMESH_PARALLEL_DISPERSED) ? MFTLRemeshEnsembleDispersedMeasured(ens, params, stats, centroid, radius, weight)
                                                                : MFTLRemeshEnsembleParallel(ens, params, stats);
    free(measure);
    return rc;
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

  for (int i = 0; i < ens->num_objects; ++i)
  {
    MFTL_RemeshStats one;
    memset(&one, 0, sizeof(one));

    if (MFTLRemeshObject(ens->objects[i], params, &one) != MFTL_SUCCESS) status = MFTL_ERROR;

    if (!stats) continue;

    stats->iterations = MFTL_MAX(stats->iterations, one.iterations);
    stats->n_split += one.n_split;
    stats->n_collapse += one.n_collapse;
    stats->n_flip += one.n_flip;
    stats->n_collapse_rejected_link += one.n_collapse_rejected_link;
    stats->n_collapse_rejected_fold += one.n_collapse_rejected_fold;
    stats->n_collapse_rejected_feature += one.n_collapse_rejected_feature;
    stats->n_flip_rejected_fold += one.n_flip_rejected_fold;
    stats->n_feature_edges += one.n_feature_edges;
    stats->n_corner_vertices += one.n_corner_vertices;
    stats->n_ridge_vertices += one.n_ridge_vertices;
    stats->n_free_vertices += one.n_free_vertices;
    stats->n_vertices_removed += one.n_vertices_removed;
    stats->n_orphans_removed += one.n_orphans_removed;
  }

  MFTLEnsembleUpdateGlobalCounts(ens);
  return status;
}
