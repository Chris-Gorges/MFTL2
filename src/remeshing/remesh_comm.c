/**
 * @file remesh_comm.c
 * @author Christian Gorges
 * @date 2026-08-17
 * @ingroup mftl_remeshing
 *
 * @brief The remeshing module's own MPI. See remesh_comm.h for why it is its own
 *        file and what each entry point costs.
 */

#include "remesh_comm.h"

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Migration
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshMigratorInit(MFTL_RemeshMigrator *m)
{
  m->maps = MFTL_NULL;
  m->arena = MFTL_NULL;
  m->cap_objects = 0;
  m->cap_entities = 0;
  m->carry_sizing = MFTL_FALSE;
}

void MFTLRemeshMigratorFree(MFTL_RemeshMigrator *m)
{
  free(m->maps);
  free(m->arena);
  const bool carry = m->carry_sizing;
  MFTLRemeshMigratorInit(m);
  m->carry_sizing = carry;
}

/**
 * @brief Grows the scratch to hold one destination per entity of every body and
 *        points each body's MFTL_OwnerMap at its slice.
 *
 * Geometric growth, never shrinking: a remeshing pass migrates once per phase
 * and the entity counts move by a few percent between phases, so after the first
 * migration every later one reuses warm, already-faulted pages. That is the
 * whole reason the arena is a caller-owned handle rather than a local.
 *
 * @param m   The migrator handle.
 * @param ens The ensemble; its object count and pool sizes fix the arena size.
 */
static void migrator_reserve(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens)
{
  const int n = ens->n_local_objects;

  /* ── The map array is indexed by LOCAL SLOT ─────────────────────────────
   *
   * It is not on a wire: MFTLRedistributeWithOwners() walks the same list in the
   * same order on this rank, so slot `o` names the same body at both ends. Keyed
   * by body gid it would be O(bodies in the ensemble) per rank per migration.
   *
   * At least one slot even for a rank holding nothing, so `m->maps` is never
   * NULL: NULL means "use the geometric rule", which a migration under moved
   * planes must not get. */
  const int n_slots = MFTL_MAX(n, 1);

  int need = 0;
  for (int o = 0; o < n; ++o) need += ens->local_objects[o]->verts.n + ens->local_objects[o]->tris.n;

  if (n_slots > m->cap_objects)
  {
    free(m->maps);
    m->cap_objects = MFTLGrowCapacityFrom(m->cap_objects, n_slots, 1);
    m->maps = (MFTL_OwnerMap *) MFTLMallocOrDie((size_t) m->cap_objects * sizeof(MFTL_OwnerMap), "remesh owner maps");
  }

  /* Every slot in [0, n) is overwritten below, so this only ever clears the
   * padding slot of a rank holding nothing. Kept anyway: NULL is the entry
   * MFTLRedistributeWithOwners() cannot misread, and a stale pointer from the
   * previous phase -- when this rank held a different set of bodies -- is
   * exactly what would be there otherwise. */
  for (int o = 0; o < n_slots; ++o)
  {
    m->maps[o].vert_owner = MFTL_NULL;
    m->maps[o].tri_owner = MFTL_NULL;
  }

  /* `m->arena == MFTL_NULL` is part of the condition, not a redundant guard: a
   * rank holding NOTHING of any body has `need == 0`, would skip the allocation,
   * and would then form `MFTL_NULL + off` below -- undefined behaviour that
   * UBSan flags and that today is simply unreachable. Under the coarsening path
   * it is the common case for most ranks. */
  if (need > m->cap_entities || m->arena == MFTL_NULL)
  {
    free(m->arena);
    m->cap_entities = MFTLGrowCapacityFrom(m->cap_entities, MFTL_MAX(need, 1), 1024);
    m->arena = (int *) MFTLMallocOrDie((size_t) m->cap_entities * sizeof(int), "remesh owner arena");
  }

  int off = 0;
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    m->maps[o].vert_owner = m->arena + off;
    off += obj->verts.n;
    m->maps[o].tri_owner = m->arena + off;
    off += obj->tris.n;
  }
}

/**
 * @brief The redistribution's carry bits for this migrator.
 *
 * The curvature field is computed once per pass (the spanning driver), so it
 * must outlive the pass's migrations; it rides in each vertex's record, which
 * hands owned vertices and ghosts alike their owner's value.
 *
 * @param m The migrator.
 * @return #MFTL_REDISTRIBUTE_CARRY_SIZING under MFTL_RemeshMigrator::carry_sizing, else 0.
 */
static unsigned migrator_carry(const MFTL_RemeshMigrator *m) { return m->carry_sizing ? MFTL_REDISTRIBUTE_CARRY_SIZING : 0u; }

int MFTLRemeshMigrateShifted(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens, const int group[3], const double shift[3], MFTL_RemeshMigrationReport *report)
{
  MFTL_ASSERT(m != MFTL_NULL && ens != MFTL_NULL && group != MFTL_NULL && shift != MFTL_NULL);
  if (report)
  {
    report->n_moved = 0;
    report->performed = MFTL_FALSE;
  }
  /* GLOBAL: an MPI_Allreduce and a collective redistribution follow, so a rank
   * holding no bodies must still reach them. */
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  const MFTL_Domain *const dom = &ens->domain;
  const int my_rank = dom->rank;
  int moved = 0;

  migrator_reserve(m, ens);

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    /* The check `compute_owners` makes for the geometric rule, repeated here
     * because an owner map turns that one off: a map need not route by geometry
     * at all (the dispersed rule does not), but the shifted rule does. */
    MFTL_ASSERT(MFTLTriangleCentresAreCurrent(obj) &&
                "MFTLRemeshMigrateShifted: stale triangle centres -- call MFTLComputeTriangleGeometry(obj, obj->tris.n) "
                "(ghosts included) first. Triangles are routed by their centre here, exactly as in the production rule.");

    /* Two streams over one contiguous array each, with the same access pattern
     * `compute_owners` has and for the same reason: `verts.pos` and
     * `tris.center` are interleaved so that a pass wanting one 3-vector per
     * entity reads 24 contiguous bytes rather than striding a record.
     *
     * The cast away from const is the arena's own storage; MFTL_OwnerMap
     * publishes it as const because the redistribution must not write it. */
    int *const restrict vo = (int *) m->maps[o].vert_owner;
    int *const restrict to = (int *) m->maps[o].tri_owner;

    /* `moved` counts OWNED entities whose destination is somebody else, which is
     * the wire volume this rank is about to pay for. Two integer compares in a
     * loop that already runs, and it is what turns "a shifted placement moves a
     * band" from an assumption into a measurement -- see MFTL_RemeshStats. */
    const double *const restrict pos = obj->verts.pos;
    const int nv = obj->verts.n;
    const int nvl = obj->verts.n_local;
    for (int v = 0; v < nv; ++v)
    {
      vo[v] = MFTLRemeshCoarseOwnerProc(dom, group, shift, &pos[3 * v]);
      moved += (v < nvl) & (vo[v] != my_rank);
    }

    const double *const restrict ctr = obj->tris.center;
    const int nt = obj->tris.n;
    const int ntl = obj->tris.n_local;
    for (int t = 0; t < nt; ++t)
    {
      to[t] = MFTLRemeshCoarseOwnerProc(dom, group, shift, &ctr[3 * t]);
      moved += (t < ntl) & (to[t] != my_rank);
    }
  }

  if (report) report->n_moved = moved;

  /* ── The migration that does not have to happen ──────────────────────────
   *
   * Phase 0 of the schedule is the CANONICAL placement, and a remeshing pass is
   * entered on a mesh that MFTLRedistribute has just partitioned -- by the
   * documented per-timestep order, and by this driver's own closing step. So the
   * owners computed above are, entity for entity, the owners the mesh already
   * has: nothing to pack, nothing to send, nothing to rebuild. Before this test
   * that phase still paid a full pack/exchange/rebuild for a permutation that
   * was the identity.
   *
   * The test has to be GLOBAL. A rank whose own share happens not to move cannot
   * skip on its own -- redistribution is collective, and a rank that returned
   * early would leave every other rank blocked in the exchange. One int through
   * an LOR is the whole cost, and it is the same shape as the decision
   * redistribution already makes about its own message pattern.
   *
   * Skipping is sound because a redistribution that moves nothing changes
   * nothing: the ownership prefix, the halo and the derived arrays are all
   * functions of the assignment, and the assignment is what did not change. The
   * halo in particular is still vertex-complete, because a neighbour's sweeps
   * can only have created entities whose corners it owns -- an unfrozen edge has
   * all of its corners on the rank operating on it -- so nothing incident to one
   * of THIS rank's owned vertices can have appeared elsewhere. */
  int anyone_moved = (moved > 0);
  MFTLAllreduceInts(dom->comm_world, &anyone_moved, 1, MFTL_REDUCE_MAX);
  if (!anyone_moved) return MFTL_SUCCESS;

  if (report) report->performed = MFTL_TRUE;
  return MFTLRedistributeWithOwnersCarrying(ens, m->maps, migrator_carry(m));
}

int MFTLRemeshMigrateToRanks(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens, const int *dest, MFTL_RemeshMigrationReport *report)
{
  MFTL_ASSERT(m != MFTL_NULL && ens != MFTL_NULL && dest != MFTL_NULL);
  if (report)
  {
    report->n_moved = 0;
    report->performed = MFTL_TRUE;  // the dispersed migration is never the identity in practice
  }
  /* GLOBAL: MFTLRedistributeWithOwners() below is collective. */
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  const int my_rank = ens->domain.rank;
  int moved = 0;

  migrator_reserve(m, ens);

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    /* `dest` is indexed by LOCAL SLOT. Every holder of a body must still have
     * arrived at the SAME destination for it -- that is what MFTL_OwnerMap
     * requires, and it is the caller's obligation, not this loop's. What changed
     * is only where the answer is stored: a rank keeps the destinations of the
     * bodies it holds instead of a copy of every body's. */
    const int r = dest[o];
    MFTL_ASSERT(r >= 0 && r < ens->domain.nproc && "MFTLRemeshMigrateToRanks: destination outside the communicator");

    int *const vo = (int *) m->maps[o].vert_owner;
    int *const to = (int *) m->maps[o].tri_owner;

    for (int v = 0; v < obj->verts.n; ++v) vo[v] = r;
    for (int t = 0; t < obj->tris.n; ++t) to[t] = r;

    /* One destination for the whole body, so the owned count IS the moved count
     * when the body is going somewhere else, and zero when it is staying. */
    if (r != my_rank) moved += obj->verts.n_local + obj->tris.n_local;
  }

  if (report) report->n_moved = moved;
  return MFTLRedistributeWithOwnersCarrying(ens, m->maps, migrator_carry(m));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The profile and the parameter check
 *
 * Both are one of the library's small reductions (MFTLAllreduceDoubles())
 * over a record only the remesher knows how to build.
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshAgreeProfile(const MFTL_Domain *domain, MFTL_RemeshStats *st, double *extra, int n_extra)
{
  MFTL_ASSERT(n_extra >= 0 && n_extra <= 2 && "MFTLRemeshAgreeProfile: at most two extra values");

  double prof[8] = {st->t_total, st->t_migrate, st->t_geometry, st->t_freeze, st->t_sweep, st->t_reduce, 0.0, 0.0};
  for (int i = 0; i < n_extra; ++i) prof[6 + i] = extra[i];

  MFTLAllreduceDoubles(domain->comm_world, prof, 6 + n_extra, MFTL_REDUCE_MAX);

  st->t_total = prof[0];
  st->t_migrate = prof[1];
  st->t_geometry = prof[2];
  st->t_freeze = prof[3];
  st->t_sweep = prof[4];
  st->t_reduce = prof[5];
  for (int i = 0; i < n_extra; ++i) extra[i] = prof[6 + i];
}

/**
 * @brief Every field of #MFTL_RemeshParameters, in declaration order. A field
 *        missing here is a field the ranks may silently disagree about, so a new
 *        field goes here too. The two pointers that close the struct cannot be
 *        cast to a double: `rim_project` is compared by presence below, and
 *        `rim_project_user` is the caller's own data.
 */
#define MFTL_REMESH_AGREED_FIELDS(X)                                                                                                                           \
  X(type)                                                                                                                                                      \
  X(smooth_mode) X(iterations) X(min_convergence_ops) X(target_length) X(min_edge_length) X(max_edge_length) X(min_edge_length_sq) X(max_edge_length_sq)       \
      X(sizing) X(curvature_tolerance) X(min_target_length) X(split_placement) X(collapse_placement) X(collapse_preserves_volume) X(fit_radius) X(projection)  \
          X(feature_angle_threshold) X(feature_corner_angle) X(smooth_iterations) X(smooth_relaxation) X(smooth_max_step_fraction) X(max_normal_deviation_cos) \
              X(renumber_global_ids) X(parallel_phase_budget) X(freeze_dilation_rings) X(parallel_strategy) X(dispersed_radius_fraction) X(parallel_coarsen)   \
                  X(coarsen_frozen_fraction) X(serial_gather_max_triangles) X(dispersed_balance_slack) X(spanning_mode)

#define MFTL_REMESH_FIELD_NAME(f) #f,
#define MFTL_REMESH_FIELD_VALUE(f) (double) params->f,

int MFTLRemeshAgreeParams(const MFTL_Domain *domain, const MFTL_RemeshParameters *params, const char *caller)
{
  /* The rim projector is a function pointer: its PRESENCE is what can be
   * compared, and it decides whether rim operations may be refused -- i.e.
   * which entities two ranks create. */
  static const char *const names[] = {MFTL_REMESH_AGREED_FIELDS(MFTL_REMESH_FIELD_NAME) "rim_project"};
  enum
  {
    N = (int) (sizeof(names) / sizeof(names[0]))
  };

  /* Agreed BIT FOR BIT, -0.0 read as 0.0, by MFTLAgreeWords(): exact, and the
   * verdict is reduced, so it fires on every rank or on none. Broadcasting rank
   * 0's block and comparing would let it fire on one rank only. */
  const double local[N] = {MFTL_REMESH_AGREED_FIELDS(MFTL_REMESH_FIELD_VALUE)(params->rim_project != MFTL_NULL) ? 1.0 : 0.0};
  uint64_t bits[N];
  for (int i = 0; i < N; ++i)
  {
    const double x = (local[i] == 0.0) ? 0.0 : local[i];
    memcpy(&bits[i], &x, sizeof x);
  }
  const int differs = MFTLAgreeWords(domain->comm_world, bits, N, MFTL_NULL);
  if (differs >= 0)
  {
    /* The failure path only, entered by every rank: the field's range. */
    double range[2] = {local[differs], -local[differs]};
    MFTLAllreduceDoubles(domain->comm_world, range, 2, MFTL_REDUCE_MIN);
    if (domain->rank == 0)
      fprintf(stderr, "[MFTL] %s: refusing to remesh: the ranks were handed different parameter blocks (%s ranges over [%.17g, %.17g]).\n", caller,
              names[differs], range[0], -range[1]);
    return MFTL_ERROR;
  }

  /* One block on every rank, so every verdict below is the same everywhere
   * without a reduction. NaN fails every comparison, so it fails these tests;
   * a NaN in any other field is refused by name. */
  const bool band_ok = (params->max_edge_length > 0.0) && (params->min_edge_length >= 0.0) && (params->min_edge_length <= params->max_edge_length);
  const char *const sizing_bad = MFTLRemeshSizingProblem(params);
  int nan_field = -1;
  for (int i = 0; i < N && nan_field < 0; ++i)
    if (local[i] != local[i]) nan_field = i;

  if (band_ok && sizing_bad == MFTL_NULL && nan_field < 0) return MFTL_SUCCESS;

  if (domain->rank == 0)
  {
    if (sizing_bad != MFTL_NULL)
      fprintf(stderr, "[MFTL] %s: refusing to remesh: %s.\n", caller, sizing_bad);
    else if (!band_ok)
      fprintf(stderr,
              "[MFTL] %s: refusing to remesh: the edge-length band [%.6g, %.6g] is invalid (valid is 0 <= min <= max, max > 0). Use "
              "MFTLEnsembleMeanEdgeLength for a distributed mesh.\n",
              caller, params->min_edge_length, params->max_edge_length);
    else
      fprintf(stderr, "[MFTL] %s: refusing to remesh: %s is not a number.\n", caller, names[nan_field]);
  }
  return MFTL_ERROR;
}

#undef MFTL_REMESH_FIELD_VALUE
#undef MFTL_REMESH_FIELD_NAME
#undef MFTL_REMESH_AGREED_FIELDS

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Body extents
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief How many times MFTLRemeshMeasureBodies() has run in this process: a
 *        test hook asserting that a pass measures once.
 */
static int g_measurement_count = 0;

int MFTLRemeshMeasurementCount(void) { return g_measurement_count; }

void MFTLRemeshResetMeasurementCount(void) { g_measurement_count = 0; }

void MFTLRemeshMeasureBodies(MFTL_Ensemble *ens, double *centroid, double *radius, double *weight, double *r_max, double *lowest_holder)
{
  const MFTL_Domain *const dom = &ens->domain;

  if (r_max != MFTL_NULL) *r_max = 0.0;

  /* GLOBAL: the reductions below are collective, so a rank holding nothing still
   * has to enter them. */
  if (ens->n_bodies_global < 1) return;

  g_measurement_count++;

  const int b = ens->n_local_objects;

  /* Layout per body: [w, s0, c0, s1, c1, s2, c2, ntri, lowest holder]. A
   * periodic axis uses the (sin, cos) pair, a non-periodic one puts its plain
   * weighted sum in the `s` slot and leaves `c` zero. The owned TRIANGLE COUNT
   * rides along because the dispersed balancer needs it -- the alternative is
   * not a shorter message, it is a SECOND reduction over the same bodies.
   *
   * INDEXED BY LOCAL SLOT: the measurement is a per-body reduction like any
   * other, so it goes through the one place that chooses a communicator, and the
   * caller's buffers describe what this rank holds. The strategy ratio, the
   * admission test and the destination of a body this rank holds are all
   * answered from the local rows and one scalar (`r_max`).
   */
  enum
  {
    STRIDE = 9
  };

  MFTL_BodyReduceOp ops[STRIDE];
  for (int j = 0; j < STRIDE; ++j) ops[j] = MFTL_BODY_REDUCE_SUM;

  /* ── The ninth slot: WHO SPEAKS FOR THIS BODY ─────────────────────────────
   *
   * A MIN of the contributing rank, i.e. the lowest-numbered holder, and it is
   * what makes a per-body quantity COUNTABLE exactly once across the job. Every
   * holder of a body comes out of the reduction with the same radius, so summing
   * "is this body too large" over ranks counts a shared body once per holder --
   * and reports "4 of 1 bodies", which is how this slot came to exist.
   * Designating one holder fixes it without a second collective and without
   * anyone having to know the holder set.
   *
   * It rides THIS reduction rather than the radius one because the radius pass
   * is skipped when the caller wants neither radius nor r_max, and who speaks
   * for a body is not conditional on that. */
  ops[8] = MFTL_BODY_REDUCE_MIN;

  double *const acc = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b * STRIDE, 1), sizeof(double), "remesh body accumulators");

  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    double *const a = acc + (size_t) STRIDE * o;
    a[8] = (double) dom->rank;

    /* OWNED triangles only. A ghost is another rank's triangle and it is
     * contributing to the same reduction from there, so counting it here would
     * weight the seam of every partition twice -- by an amount that grows with
     * the rank count, which is the one shape of error a global measurement must
     * not have. */
    const int nt = obj->tris.n_local;

    /* The weight is a COUNT, and it is accumulated outside the loop because it
     * does not depend on the geometry -- only on how many owned triangles this
     * rank has of this body. Exactly the same ownership rule as the area sum
     * below, for exactly the same reason: a ghost is counted by its owner. */
    a[7] += (double) nt;

    for (int t = 0; t < nt; ++t)
    {
      const double w = obj->tris.area[t];
      const double *const c = &obj->tris.center[3 * t];

      a[0] += w;
      for (int k = 0; k < 3; ++k)
      {
        if (dom->is_periodic[k] && dom->domain_size[k] > 0.0)
        {
          const double th = MFTL_TWO_PI * (c[k] - dom->global_min[k]) / dom->domain_size[k];
          a[1 + 2 * k] += w * sin(th);
          a[2 + 2 * k] += w * cos(th);
        }
        else
        {
          a[1 + 2 * k] += w * c[k];
        }
      }
    }
  }

  MFTLEnsembleReduceBodies(ens, acc, STRIDE, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  /* Written before the centroids, because it needs nothing from them: the slot
   * came back from the same reduction and is already the global count. */
  if (weight != MFTL_NULL)
  {
    for (int o = 0; o < b; ++o) weight[o] = acc[(size_t) STRIDE * o + 7];
  }

  if (lowest_holder != MFTL_NULL)
  {
    for (int o = 0; o < b; ++o) lowest_holder[o] = acc[(size_t) STRIDE * o + 8];
  }

  /* Scratch for the centroid even when the caller does not want it -- the radius
   * is a distance to it. */
  double *const ctr = (centroid != MFTL_NULL) ? centroid : (double *) MFTLMallocOrDie((size_t) MFTL_MAX(3 * b, 1) * sizeof(double), "remesh centroids");

  for (int o = 0; o < b; ++o)
  {
    const double *const s = acc + (size_t) STRIDE * o;
    const double w = s[0];
    const double inv = (w > 0.0) ? 1.0 / w : 0.0;

    for (int k = 0; k < 3; ++k)
    {
      double x;
      if (dom->is_periodic[k] && dom->domain_size[k] > 0.0)
      {
        /* atan2 of the weighted mean direction. Its branch cut is at theta = pi,
         * i.e. the middle of the domain and not the seam, which is the point:
         * the seam is the one place a min/max or an arithmetic mean fails. */
        double th = atan2(s[1 + 2 * k], s[2 + 2 * k]);
        if (th < 0.0) th += MFTL_TWO_PI;
        x = dom->global_min[k] + th * dom->domain_size[k] / MFTL_TWO_PI;
      }
      else
      {
        x = s[1 + 2 * k] * inv;
      }
      ctr[3 * o + k] = x;
    }
    MFTLApplyPeriodicity(dom, &ctr[3 * o]);
  }

  /* The second reduction. Whether it is taken is a property of the CALL and not
   * of this rank's data, so two ranks cannot disagree about entering it. */
  if (radius != MFTL_NULL || r_max != MFTL_NULL)
  {
    double *const rad = (radius != MFTL_NULL) ? radius : (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b, 1) * sizeof(double), "remesh body radii");
    MFTL_BodyReduceOp rop = MFTL_BODY_REDUCE_MAX;

    for (int o = 0; o < b; ++o)
    {
      const MFTL_Object *const obj = ens->local_objects[o];
      double best = 0.0;

      /* Corners rather than centres: a triangle sticking out of the body is part
       * of it, and the difference is exactly one edge length -- the same order as
       * the band widths this radius is later compared against. */
      const int nt = obj->tris.n_local;
      for (int t = 0; t < nt; ++t)
      {
        double corner[3][3];
        MFTLGetTriangleCorners(obj, t, corner);

        for (int j = 0; j < 3; ++j)
        {
          double d[3];
          MFTLGetMinimumImageVector(dom, d, &ctr[3 * o], corner[j]);
          const double l2 = MFTLVectorNormSq(d);
          if (l2 > best) best = l2;
        }
      }
      rad[o] = sqrt(best);
    }

    MFTLEnsembleReduceBodies(ens, rad, 1, &rop, MFTL_BODY_REDUCE_SCOPE_AUTO);

    /* ── The largest body in the ENSEMBLE, as ONE scalar ────────────────────
     *
     * The strategy ratio and the admission test are taken against it and must
     * agree on every rank, or the ranks split between two drivers (deadlock).
     * One double through MPI_MAX. */
    if (r_max != MFTL_NULL)
    {
      *r_max = 0.0;
      for (int o = 0; o < b; ++o) *r_max = MFTL_MAX(*r_max, rad[o]);
      MFTLAllreduceDoubles(dom->comm_world, r_max, 1, MFTL_REDUCE_MAX);
    }

    if (radius == MFTL_NULL) free(rad);
  }

  if (centroid == MFTL_NULL) free(ctr);
  free(acc);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Neighbourhood exchange
 *
 * Two fixed-length shapes (the variable-length one is the library's
 * MFTLNeighborAllgatherDoubles()), none of which can carry a payload indexed by
 * the body id space or by the rank space -- see remesh_comm.h for why that is
 * the point.
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshNeighborAllgather(const MFTL_Domain *domain, const double *send, int n, double *recv)
{
  const int d = domain->num_neighbors;
  if (domain->nproc < 2 || d < 1 || n < 1) return;

  MPI_Neighbor_allgather(send, n, MPI_DOUBLE, recv, n, MPI_DOUBLE, domain->comm_graph);
}

void MFTLRemeshNeighborAlltoall(const MFTL_Domain *domain, const double *send, int n, double *recv)
{
  const int d = domain->num_neighbors;
  if (domain->nproc < 2 || d < 1 || n < 1) return;

  MPI_Neighbor_alltoall(send, n, MPI_DOUBLE, recv, n, MPI_DOUBLE, domain->comm_graph);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Ghost exchanges: payloads on the library's one engine
 *
 * Every ghost refresh the remesher needs is a payload on MFTLGhostPull()
 * (mftl_communication.h) over #MFTL_ROUTE_AUTO, the route with the global
 * fallback a coarsened placement needs. They are listed here because this file
 * is the module's inventory of network use; none of them calls MPI. Each is a
 * pair of block callbacks (once per (body, pool), never per ghost), so the
 * per-timestep position sync, which shares the engine, has no payload test in
 * any loop of its own (see remesh_comm.h for why that matters).
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Owner side of the id exchange: `base[slot] + v`, or #MFTL_ERROR. */
static void serve_ghost_ids(const MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, const int *offset, int n, unsigned char *reply, void *ctx)
{
  (void) obj;
  (void) pool;
  /* Owned entities occupy [0, n_local) and are renumbered `base + slot`, so the
   * answer is arithmetic. That is the whole reason the re-densification numbers
   * BY SLOT rather than by any other rule -- any other rule would need a second
   * table here. `base` is keyed by LOCAL SLOT: it came out of a per-body scan
   * over that body's holders. */
  const int base = (slot != MFTL_ERROR) ? ((const int *) ctx)[slot] : 0;
  for (int j = 0; j < n; ++j)
  {
    const int id = (idx[j] == MFTL_ERROR) ? MFTL_ERROR : base + idx[j];
    memcpy(reply + offset[j], &id, sizeof id);
  }
}

/**
 * @brief Requester side of an exchange of one int per ghost vertex that
 *        #MFTL_ERROR marks unserved.
 * @param obj    The body.
 * @param idx    `[n]` ghost indices.
 * @param n      Replies.
 * @param rows   `[n]` ints.
 * @param id_out Writes the global id (true) or merges the feature bits (false);
 *               a literal at both call sites, so inlining folds the test away.
 * @return Replies that were the sentinel.
 */
static inline int apply_ints(MFTL_Object *obj, const int *idx, int n, const unsigned char *rows, bool id_out)
{
  int skipped = 0;
  for (int j = 0; j < n; ++j)
  {
    int value;
    memcpy(&value, rows + (size_t) j * sizeof value, sizeof value);
    if (value == MFTL_ERROR)
    {
      skipped++;  // owner could not serve it: the ghost keeps its value
      continue;
    }
    MFTL_Vertex *const v = &obj->verts.rec[idx[j]];
    if (id_out)
      v->global_id = value;
    else
      v->flags = (v->flags & ~(MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER)) | value;
  }
  return skipped;
}

/** @brief Requester side of the id exchange. */
static int apply_ghost_ids(MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, int n, const unsigned char *rows, void *ctx)
{
  (void) slot;
  (void) pool;
  (void) ctx;
  return apply_ints(obj, idx, n, rows, MFTL_TRUE);
}

/**
 * @brief A one-int-per-ghost-vertex pull on the remesher's route.
 * @param ens   The ensemble.
 * @param serve Owner side.
 * @param apply Requester side.
 * @param ctx   Their context.
 * @return As MFTLGhostPull().
 */
static int pull_vertex_ints(MFTL_Ensemble *ens, MFTL_GhostServeFn serve, MFTL_GhostApplyFn apply, void *ctx)
{
  MFTL_GhostPull pull;
  pull.row_bytes[MFTL_FIELD_VERTEX] = (int) sizeof(int);
  pull.row_bytes[MFTL_FIELD_TRIANGLE] = 0;
  pull.route = MFTL_ROUTE_AUTO;
  pull.keys[0] = pull.keys[1] = MFTL_NULL;
  pull.serve = serve;
  pull.apply = apply;
  pull.ctx = ctx;
  return MFTLGhostPull(ens, &pull);
}

int MFTLRemeshSyncGhostIds(MFTL_Ensemble *ens, const int *vertex_gid_base)
{
  MFTL_ASSERT(ens != MFTL_NULL && vertex_gid_base != MFTL_NULL);
  const int rc = pull_vertex_ints(ens, serve_ghost_ids, apply_ghost_ids, (void *) vertex_gid_base);

  /* A failure means a request could not be answered -- the requester believes a
   * rank owns a vertex it does not have. That is a disagreement about
   * ownership, not a routing problem, and it leaves the ghost with a stale id
   * that the closing redistribution will duplicate rather than match. Loud in
   * Debug, reported in Release. */
  MFTL_ASSERT(rc == MFTL_SUCCESS && "remesh ghost ids: some ghost kept a stale global id -- the closing redistribution will duplicate it");
  return rc;
}

int MFTLRemeshSyncGhostSizing(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  const MFTL_GhostColumn c = {MFTL_GHOST_COLUMN_SIZING, 0};
  return MFTLUpdateGhostColumns(ens, &c, 1, MFTL_ROUTE_AUTO);
}

/** @brief Owner side of the classification exchange: its FEATURE/CORNER bits, or #MFTL_ERROR. */
static void serve_ghost_class(const MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, const int *offset, int n, unsigned char *reply,
                              void *ctx)
{
  (void) slot;
  (void) pool;
  (void) ctx;
  for (int j = 0; j < n; ++j)
  {
    const int bits = (idx[j] == MFTL_ERROR) ? MFTL_ERROR : (obj->verts.rec[idx[j]].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER));
    memcpy(reply + offset[j], &bits, sizeof bits);
  }
}

/** @brief Requester side of the classification exchange. */
static int apply_ghost_class(MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, int n, const unsigned char *rows, void *ctx)
{
  (void) slot;
  (void) pool;
  (void) ctx;
  return apply_ints(obj, idx, n, rows, MFTL_FALSE);
}

int MFTLRemeshSyncGhostClass(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  return pull_vertex_ints(ens, serve_ghost_class, apply_ghost_class, MFTL_NULL);
}

int MFTLRemeshSyncGhostPositions(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  /* MFTLUpdateGhostPositions() on the route with the global fallback: the same
   * payload, the same invalidation of the derived geometry. */
  const MFTL_GhostColumn c = {MFTL_GHOST_COLUMN_POSITION, 0};
  return MFTLUpdateGhostColumns(ens, &c, 1, MFTL_ROUTE_AUTO);
}
/* ═══════════════════════════════════════════════════════════════════════════
 * 6. The projection halo
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLRemeshReferenceHalo(MFTL_Ensemble *ens, double width, double **out, int *n_out)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int me = dom->rank;

  /* This rank's canonical cell. */
  double h[3], centre[3];
  for (int k = 0; k < 3; ++k)
  {
    h[k] = dom->domain_size[k] / (double) dom->cuts[k];
    centre[k] = dom->global_min[k] + ((double) dom->cell[k] + 0.5) * h[k];
  }
  /* A margin far below any length the mesh resolves, so a cell face that a
   * box only touches in exact arithmetic still counts. */
  const double reach = width * (1.0 + 1e-9);

  int n_rec = 0, cap = 0;
  double *rec = MFTL_NULL;
  int *dst = MFTL_NULL;
  int cap_ranks = 27;
  int *ranks = (int *) MFTLMallocOrDie((size_t) cap_ranks * sizeof(int), "remesh reference halo ranks");

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double c[3][3];
      MFTLGetTriangleCorners(obj, t, c);
      /* The image of the triangle nearest this cell. */
      double g[3], d[3], lo[3], hi[3];
      for (int k = 0; k < 3; ++k) g[k] = (c[0][k] + c[1][k] + c[2][k]) / 3.0;
      MFTLGetMinimumImageVector(dom, d, centre, g);
      for (int k = 0; k < 3; ++k)
      {
        const double sh = centre[k] + d[k] - g[k];
        lo[k] = MFTL_MIN(c[0][k], MFTL_MIN(c[1][k], c[2][k])) + sh - reach;
        hi[k] = MFTL_MAX(c[0][k], MFTL_MAX(c[1][k], c[2][k])) + sh + reach;
      }
      /* Every rank whose cell the widened box reaches, by the owner rule (a
       * box past a wall belongs to the boundary cells), with no bound on how
       * many: a triangle wider than several rank cells still reaches them all. */
      int nr = MFTLDomainRanksTouching(dom, lo, hi, ranks, cap_ranks);
      if (nr > cap_ranks)
      {
        cap_ranks = nr;
        ranks = (int *) MFTLReallocOrDie(ranks, (size_t) cap_ranks * sizeof(int), "remesh reference halo ranks");
        nr = MFTLDomainRanksTouching(dom, lo, hi, ranks, cap_ranks);
      }
      if (n_rec + nr > cap)
      {
        cap = MFTLGrowCapacity(cap, n_rec + nr);
        rec = (double *) MFTLReallocOrDie(rec, (size_t) cap * MFTL_REMESH_REF_HALO_WIDTH * sizeof(double), "remesh reference halo");
        dst = (int *) MFTLReallocOrDie(dst, (size_t) cap * sizeof(int), "remesh reference halo destinations");
      }
      for (int j = 0; j < nr; ++j)
      {
        if (ranks[j] == me) continue;
        MFTLRemeshReferencePackTriangle(obj, t, &rec[(size_t) n_rec * MFTL_REMESH_REF_HALO_WIDTH]);
        dst[n_rec++] = ranks[j];
      }
    }
  }

  *out = (double *) MFTLExchangeRecordsTo(dom, MFTL_REMESH_REF_HALO_WIDTH * sizeof(double), rec, dst, n_rec, n_out);
  free(ranks);
  free(rec);
  free(dst);
  return MFTL_SUCCESS;
}
