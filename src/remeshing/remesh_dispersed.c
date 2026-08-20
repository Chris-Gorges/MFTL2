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

MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategyMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, double *centroid, double *radius)
{
  MFTL_ASSERT(ens != MFTL_NULL && centroid != MFTL_NULL && radius != MFTL_NULL);

  /* At one rank nothing is split and both strategies degenerate to the serial
   * kernel. Answering SPANNING keeps MFTLRemeshEnsemble's nproc == 1 path the
   * only special case there is. */
  if (ens->domain.nproc < 2 || ens->num_objects == 0) return MFTL_REMESH_PARALLEL_SPANNING;

  const double frac = (params && params->dispersed_radius_fraction > 0.0) ? params->dispersed_radius_fraction : 0.4;
  const double h_min = MFTLDomainGetHMin(&ens->domain);

  /* The measurement is taken even when h_min says the answer is already SPANNING,
   * because the caller may be MFTLRemeshEnsemble, which owns the buffers and
   * would otherwise have to know whether they were filled. Filling them
   * unconditionally is one collective on a path that is about to run a dozen. */
  prepare_for_migration(ens);
  MFTLRemeshMeasureBodies(ens, centroid, radius);

  if (h_min <= 0.0) return MFTL_REMESH_PARALLEL_SPANNING;

  double r_max = 0.0;
  for (int o = 0; o < ens->num_objects; ++o) r_max = MFTL_MAX(r_max, radius[o]);

  /* The reduction inside MFTLRemeshMeasureBodies already made every rank agree
   * on every radius, so this comparison is a pure function of replicated data
   * and cannot split the ranks between the two drivers. That property is the
   * whole reason the measurement is collective rather than local-and-reduced-
   * later: two ranks entering different drivers is an immediate deadlock. */
  return (r_max > frac * h_min) ? MFTL_REMESH_PARALLEL_SPANNING : MFTL_REMESH_PARALLEL_DISPERSED;
}

MFTL_RemeshParallelStrategy MFTLRemeshChooseStrategy(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens->domain.nproc < 2 || ens->num_objects == 0) return MFTL_REMESH_PARALLEL_SPANNING;

  const int n = ens->num_objects;
  double *const scratch = (double *) MFTLMallocOrDie((size_t) (4 * n) * sizeof(double), "remesh strategy measurement");
  const MFTL_RemeshParallelStrategy strategy = MFTLRemeshChooseStrategyMeasured(ens, params, scratch, scratch + 3 * n);
  free(scratch);
  return strategy;
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
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *const obj = ens->objects[o];
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
  const int n_bodies = ens->num_objects;
  if (n_bodies < 1) return MFTLRemeshEnsembleDispersedMeasured(ens, params, stats, MFTL_NULL, MFTL_NULL);

  double *const scratch = (double *) MFTLMallocOrDie((size_t) (4 * n_bodies) * sizeof(double), "dispersed body measurements");
  prepare_for_migration(ens);
  MFTLRemeshMeasureBodies(ens, scratch, scratch + 3 * n_bodies);

  const int rc = MFTLRemeshEnsembleDispersedMeasured(ens, params, stats, scratch, scratch + 3 * n_bodies);
  free(scratch);
  return rc;
}

int MFTLRemeshEnsembleDispersedMeasured(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, MFTL_RemeshStats *stats, const double *centroid_in,
                                        const double *radius_in)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  MFTL_Domain *const dom = &ens->domain;
  const int n = ens->num_objects;

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
   * Handed in, not measured here. MFTLRemeshMeasureBodies is two Allreduces for
   * the whole ensemble -- 7*num_objects doubles and then num_objects -- and the
   * caller has already paid them to decide the strategy. Measuring again would
   * double the collectives of the whole pass to recompute an identical answer.
   *
   * The radius is not diagnostic here. It is the ADMISSION TEST, and it has to
   * be taken before the migration rather than after it -- see step 2. */
  MFTL_ASSERT(centroid_in != MFTL_NULL && radius_in != MFTL_NULL &&
              "MFTLRemeshEnsembleDispersedMeasured: the caller owes a replicated measurement -- see remesh_internal.h");

  const double *const centroid = centroid_in;
  const double *const radius = radius_in;
  int *const dest = (int *) MFTLMallocOrDie((size_t) n * sizeof(int), "dispersed destinations");

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
  const double h_min = MFTLDomainGetHMin(dom);

  int too_large = 0;
  if (dom->nproc > 1 && h_min > 0.0)
  {
    for (int o = 0; o < n; ++o)
    {
      if (radius[o] > frac * h_min) too_large++;
    }
  }
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

  /* The PRODUCTION ownership rule, on a position every rank agrees about. A
   * centroid that came out of an Allreduce is identical everywhere to the bit,
   * so this is a pure function of replicated data and every rank picks the same
   * destination -- which is exactly what MFTL_OwnerMap requires. */
  for (int o = 0; o < n; ++o) dest[o] = MFTLGetOwnerProc(dom, &centroid[3 * o]);

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
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
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
  for (int o = 0; o < n; ++o)
  {
    MFTL_Object *const obj = ens->objects[o];
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
  int *const counters = (int *) MFTLCallocOrDie((size_t) (2 * n + 1), sizeof(int), "dispersed id counters");
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
    counters[2 * o + 0] = (obj->verts.n > 0) ? obj->next_vertex_gid : 0;
    counters[2 * o + 1] = (obj->tris.n > 0) ? obj->next_triangle_gid : 0;
  }
  /* The operation count rides the same message so that the reduction is one
   * collective rather than two, and it is read back below into the stats. */
  counters[2 * n] = local_ops;

  t_mark = MFTLRemeshWallTime();
  MFTLRemeshAgreeIntMax(dom, counters, 2 * n + 1);
  st->t_reduce += MFTLRemeshWallTime() - t_mark;

  for (int o = 0; o < n; ++o)
  {
    MFTL_Object *const obj = ens->objects[o];
    obj->next_vertex_gid = counters[2 * o + 0];
    obj->next_triangle_gid = counters[2 * o + 1];
    obj->next_halfedge_gid = 3 * obj->next_triangle_gid;
  }

  /* MAX rather than SUM, and it is the right reduction here: exactly one rank
   * holds any entity of a body, so no two ranks operate on the same body and the
   * maximum is the busiest rank's count. It is reported as such -- see the
   * locality note on MFTL_RemeshStats. */
  st->n_ops_busiest_rank = counters[2 * n];
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
  for (int o = 0; o < n; ++o)
  {
    MFTL_Object *const obj = ens->objects[o];
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
