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
    m->cap_objects = (m->cap_objects > 0) ? m->cap_objects : 1;
    while (m->cap_objects < n_slots) m->cap_objects *= 2;
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
    m->cap_entities = (m->cap_entities > 0) ? m->cap_entities : 1024;
    while (m->cap_entities < need) m->cap_entities *= 2;
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
  if (dom->nproc > 1) MPI_Allreduce(MPI_IN_PLACE, &anyone_moved, 1, MPI_INT, MPI_LOR, dom->comm_world);
  if (!anyone_moved) return MFTL_SUCCESS;

  if (report) report->performed = MFTL_TRUE;
  if (!m->carry_sizing) return MFTLRedistributeWithOwners(ens, m->maps);

  /* ── The curvature field rides along ────────────────────────────────────
   * Computed once per pass (the spanning driver), so it must outlive the pass's
   * migrations; a redistribution keeps positions and topology, not this array.
   * Every owned vertex sends (body, id, target) to the rank the owner map names,
   * itself included, and the ghosts then ask their owners. Ids are unique
   * during the pass, so the lookup on arrival is exact. */
  int n_rec = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (ens->local_objects[o]->verts.has_sizing) n_rec += ens->local_objects[o]->verts.n_local;
  double *rec = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(3 * n_rec, 1) * sizeof(double), "remesh sizing carry");
  int *dst = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_rec, 1) * sizeof(int), "remesh sizing carry destinations");
  int r = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    if (!obj->verts.has_sizing) continue;
    for (int v = 0; v < obj->verts.n_local; ++v, ++r)
    {
      rec[3 * r + 0] = (double) obj->global_id;
      rec[3 * r + 1] = (double) obj->verts.rec[v].global_id;
      rec[3 * r + 2] = obj->verts.sizing[v];
      dst[r] = m->maps[o].vert_owner[v];
    }
  }
  int n_in = 0;
  double *in = MFTLRemeshExchangeRecords(dom, rec, dst, n_rec, 3, &n_in);
  free(rec);
  free(dst);

  int status = MFTLRedistributeWithOwners(ens, m->maps);

  /* Owned vertices by id: one sorted id list per held body. */
  const int b = ens->n_local_objects;
  int **ids = (int **) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1), sizeof(int *), "remesh sizing carry ids");
  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    MFTLObjectEnableSizing(obj);
    const int nl = obj->verts.n_local;
    ids[o] = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(2 * nl, 1) * sizeof(int), "remesh sizing carry ids");
    for (int v = 0; v < nl; ++v) ids[o][v] = obj->verts.rec[v].global_id, ids[o][nl + v] = v;
    MFTLSortPairs32(ids[o], ids[o] + nl, nl);
  }
  int missed = 0;
  for (int i = 0; i < n_in; ++i)
  {
    const int body = (int) in[3 * i], gid = (int) in[3 * i + 1];
    /* The local list is ordered by body id, so the slot is a bisection. */
    int o = -1, lo = 0, hi = b - 1;
    while (lo <= hi)
    {
      const int mid = (lo + hi) / 2, g = ens->local_objects[mid]->global_id;
      if (g == body)
      {
        o = mid;
        break;
      }
      if (g < body)
        lo = mid + 1;
      else
        hi = mid - 1;
    }
    MFTL_Object *const obj = (o >= 0) ? ens->local_objects[o] : MFTL_NULL;
    const int k = obj ? MFTLSortedIndexOf(ids[o], obj->verts.n_local, gid) : -1;
    if (k < 0)
    {
      missed++;
      continue;
    }
    obj->verts.sizing[ids[o][obj->verts.n_local + k]] = in[3 * i + 2];
  }
  for (int o = 0; o < b; ++o) free(ids[o]);
  free(ids);
  free(in);
  MFTL_ASSERT(missed == 0 && "MFTLRemeshMigrateShifted: a carried target length found no owned vertex on arrival");
  if (missed) status = MFTL_ERROR;

  if (MFTLRemeshSyncGhostSizing(ens) != MFTL_SUCCESS) status = MFTL_ERROR;
  return status;
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
  return MFTLRedistributeWithOwners(ens, m->maps);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Small collectives
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshSumInts(const MFTL_Domain *domain, int *values, int n)
{
  if (domain->nproc < 2 || n < 1) return;
  MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_INT, MPI_SUM, domain->comm_world);
}

void MFTLRemeshAgreeIntMax(const MFTL_Domain *domain, int *values, int n)
{
  if (domain->nproc < 2 || n < 1) return;
  MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_INT, MPI_MAX, domain->comm_world);
}

void MFTLRemeshAgreeDouble(const MFTL_Domain *domain, double *values, int n)
{
  if (domain->nproc < 2 || n < 1) return;
  MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_DOUBLE, MPI_SUM, domain->comm_world);
}

void MFTLRemeshAgreeDoubleMax(const MFTL_Domain *domain, double *values, int n)
{
  if (domain->nproc < 2 || n < 1) return;
  MPI_Allreduce(MPI_IN_PLACE, values, n, MPI_DOUBLE, MPI_MAX, domain->comm_world);
}

void MFTLRemeshAgreeProfile(const MFTL_Domain *domain, MFTL_RemeshStats *st, double *extra, int n_extra)
{
  MFTL_ASSERT(n_extra >= 0 && n_extra <= 2 && "MFTLRemeshAgreeProfile: at most two extra values");

  double prof[8] = {st->t_total, st->t_migrate, st->t_geometry, st->t_freeze, st->t_sweep, st->t_reduce, 0.0, 0.0};
  for (int i = 0; i < n_extra; ++i) prof[6 + i] = extra[i];

  MFTLRemeshAgreeDoubleMax(domain, prof, 6 + n_extra);

  st->t_total = prof[0];
  st->t_migrate = prof[1];
  st->t_geometry = prof[2];
  st->t_freeze = prof[3];
  st->t_sweep = prof[4];
  st->t_reduce = prof[5];
  for (int i = 0; i < n_extra; ++i) extra[i] = prof[6 + i];
}

double MFTLRemeshWallTime(void) { return MPI_Wtime(); }

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

  /* NaN fails every comparison, so it fails the band test too. */
  const bool band_ok = (params->max_edge_length > 0.0) && (params->min_edge_length >= 0.0) && (params->min_edge_length <= params->max_edge_length);
  const char *const sizing_bad = MFTLRemeshSizingProblem(params);

  /* [values | negated values | band verdict]. A MIN of -x is a MAX of x, so one
   * reduction yields both extremes of every field, and equal extremes mean every
   * rank passed the same value. Broadcasting rank 0's block and comparing would
   * let the check fire on one rank only, which is the failure being fixed. */
  const double local[N] = {MFTL_REMESH_AGREED_FIELDS(MFTL_REMESH_FIELD_VALUE)(params->rim_project != MFTL_NULL) ? 1.0 : 0.0};
  double v[2 * N + 2];
  for (int i = 0; i < N; ++i)
  {
    v[i] = local[i];
    v[N + i] = -local[i];
  }
  v[2 * N] = band_ok ? 1.0 : 0.0;
  v[2 * N + 1] = sizing_bad ? 0.0 : 1.0;

  if (domain->nproc > 1) MPI_Allreduce(MPI_IN_PLACE, v, 2 * N + 2, MPI_DOUBLE, MPI_MIN, domain->comm_world);

  int differs = -1;
  for (int i = 0; i < N && differs < 0; ++i)
  {
    if (!(v[i] == -v[N + i])) differs = i;
  }

  if (v[2 * N] == 1.0 && v[2 * N + 1] == 1.0 && differs < 0) return MFTL_SUCCESS;

  if (domain->rank == 0)
  {
    if (v[2 * N + 1] != 1.0)
    {
      /* Rank 0's own reason when it has one; otherwise another rank's block was
       * the bad one, which the field comparison below would also name. */
      fprintf(stderr, "[MFTL] %s: refusing to remesh: %s.\n", caller,
              sizing_bad ? sizing_bad : "the sizing parameters are invalid on at least one rank");
    }
    else if (v[2 * N] != 1.0)
    {
      fprintf(stderr,
              "[MFTL] %s: refusing to remesh: the edge-length band is invalid on at least one rank (this rank: [%.6g, %.6g]; valid is "
              "0 <= min <= max, max > 0). Use MFTLEnsembleMeanEdgeLength for a distributed mesh.\n",
              caller, params->min_edge_length, params->max_edge_length);
    }
    else
    {
      fprintf(stderr, "[MFTL] %s: refusing to remesh: the ranks were handed different parameter blocks (%s ranges over [%.17g, %.17g]).\n",
              caller, names[differs], v[differs], -v[N + differs]);
    }
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
          const double l2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
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
      double mine = 0.0;
      for (int o = 0; o < b; ++o) mine = MFTL_MAX(mine, rad[o]);

      if (dom->nproc > 1)
        MPI_Allreduce(&mine, r_max, 1, MPI_DOUBLE, MPI_MAX, dom->comm_world);
      else
        *r_max = mine;
    }

    if (radius == MFTL_NULL) free(rad);
  }

  if (centroid == MFTL_NULL) free(ctr);
  free(acc);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Neighbourhood exchange
 *
 * Three shapes, none of which can carry a payload indexed by the body id space
 * or by the rank space -- see remesh_comm.h for why that is the point.
 * ═══════════════════════════════════════════════════════════════════════════ */

double *MFTLRemeshNeighborGather(const MFTL_Domain *domain, const double *send, int n_send, int *rcount, int *rdispl, int *n_recv)
{
  return MFTLNeighborAllgatherDoubles(domain, send, n_send, rcount, rdispl, n_recv);
}

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
 * 5. The ghost global-id exchange
 *
 * Request/reply over comm_graph, the same schedule shape as
 * MFTLUpdateGhostPositions and deliberately a separate implementation of it --
 * see remesh_comm.h. Four bytes per ghost vertex, no registered datatype.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The 2-int-at-a-time growable list the request/target pairs go into. */
typedef struct
{
  int *data;  ///< [cap] interleaved pairs.
  int size;  ///< INTS written, i.e. twice the pair count.
  int cap;  ///< Ints allocated.
} pair_list;

/**
 * @brief Initialises the list with its starting capacity.
 * @param b The list.
 */
static void pl_init(pair_list *b)
{
  b->cap = 256;
  b->size = 0;
  b->data = (int *) MFTLMallocOrDie((size_t) b->cap * sizeof(int), "remesh ghost id list");
}

/**
 * @brief Appends one pair.
 * @param b The list.
 * @param a First member.
 * @param c Second member.
 */
static void pl_add2(pair_list *b, int a, int c)
{
  MFTL_ENSURE_CAPACITY(b->data, &b->size, &b->cap, 2, int);
  b->data[b->size++] = a;
  b->data[b->size++] = c;
}

/**
 * @brief Releases the list.
 * @param b The list.
 */
static void pl_free(pair_list *b) { free(b->data); }

/**
 * @brief What the owner sends back for one ghost vertex: writes `width` bytes
 *        to @p out. Called with @p v = #MFTL_ERROR for a request it cannot
 *        serve, and must then write a sentinel its ghost_apply_fn skips.
 */
typedef void (*ghost_answer_fn)(const MFTL_Object *obj, int slot, int v, void *out, const void *ctx);

/** @brief Applies one reply (`width` bytes at @p in) to ghost vertex @p v of @p obj. */
typedef void (*ghost_apply_fn)(MFTL_Object *obj, int v, const void *in);

/**
 * @brief Answers every request addressed to one body.
 *
 * The owner has to turn a global id into a pool slot, which is the one lookup
 * in this module that is not already an index. It needs no hash map for the
 * same reason the ghost sync does not: ids are DENSE per entity class, so a
 * scatter of the owned vertices into an array sized by `next_vertex_gid` answers
 * every request with one indexed read.
 *
 * O(owned) to build and O(requests) to use, so it is built once per body that
 * was actually asked something -- which is what the counting sort in the caller
 * is for.
 *
 * @param      obj         The mesh whose OWNED vertices answer the requests.
 * @param      slot        Its local slot, handed to @p answer.
 * @param      flat_recv   The received request ids, all senders concatenated.
 * @param      order       Request indices belonging to this body.
 * @param      first       First slot of @p order to serve.
 * @param      last        One past the last slot to serve.
 * @param[out] reply       `width` bytes per request, written by @p answer.
 * @param      width       Bytes per reply.
 * @param      answer      Writes one reply.
 * @param      ctx         Passed to @p answer.
 * @param      slot_of_gid Scatter table from global id to pool slot.
 * @param      table_len   Addressable length of @p slot_of_gid.
 * @param[out] n_unserved  Incremented once per unanswerable request.
 */
static void answer_body(const MFTL_Object *obj, int slot, const int *restrict flat_recv, const int *restrict order, int first, int last,
                        unsigned char *restrict reply, size_t width, ghost_answer_fn answer, const void *ctx, int *restrict slot_of_gid,
                        int table_len, int *n_unserved)
{
  for (int i = 0; i < table_len; ++i) slot_of_gid[i] = MFTL_ERROR;

  const MFTL_Vertex *const restrict vr = obj->verts.rec;
  const int n_local = obj->verts.n_local;
  for (int v = 0; v < n_local; ++v)
  {
    const int g = vr[v].global_id;
    MFTL_ASSERT(g >= 0 && g < table_len && "remesh ghost ids: owned vertex outside the body's id range");
    if (g >= 0 && g < table_len) slot_of_gid[g] = v;
  }

  for (int i = first; i < last; ++i)
  {
    const int k = order[i];
    const int gid = flat_recv[2 * k + 1];
    const int v = (gid >= 0 && gid < table_len) ? slot_of_gid[gid] : MFTL_ERROR;

    if (v == MFTL_ERROR)
    {
      /* The requester believes this rank owns a vertex it does not have. Report
       * it rather than dereferencing garbage: the requester skips the reply and
       * keeps its stale value, which the caller turns into MFTL_ERROR. */
      MFTL_ASSERT(0 && "remesh ghost exchange: asked for a vertex this rank does not own");
      (*n_unserved)++;
    }
    answer(obj, slot, v, reply + (size_t) k * width, ctx);
  }
}

/**
 * @brief Owner-to-ghost request/reply over the ghost VERTICES of every local
 *        body: each ghost asks its owner, by (body gid, vertex gid), and the
 *        owner's @p answer comes back @p width bytes per ghost.
 *
 * The one route both remesher exchanges take -- the new global ids after a
 * phase (MFTLRemeshSyncGhostIds()) and the curvature sizing after a migration
 * (MFTLRemeshSyncGhostSizing()) -- so the global fallback a coarsened placement
 * needs exists once. Requests are two ints, replies opaque bytes.
 *
 * @param ens    The ensemble; every rank calls.
 * @param width  Bytes per reply.
 * @param answer Owner side, per request.
 * @param ctx    Passed to @p answer.
 * @param apply  Requester side, per reply.
 * @param what   Name for the diagnostics.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR when some request could not be served
 *         (the ghost keeps its old value).
 * @collective
 */
static int ghost_vertex_exchange(MFTL_Ensemble *ens, size_t width, ghost_answer_fn answer, const void *ctx, ghost_apply_fn apply, const char *what)
{

  MFTL_Domain *const dom = &ens->domain;
  MFTL_Object *const *const objects = ens->local_objects;
  const int n_local = ens->n_local_objects;
  const int nproc = dom->nproc;
  const int nn = dom->num_neighbors;

  /* GLOBAL: an MPI_Allreduce decides the route below, so a rank holding no
   * bodies must still reach it. */
  if (ens->n_bodies_global == 0 || nproc == 1) return MFTL_SUCCESS;

  /* ── 0. Can the 26-neighbourhood carry this exchange at all? ─────────────
   *
   * THE FAILURE THIS REMOVES IS SILENT AND IT TEARS THE SURFACE. The graph
   * communicator reaches the 26 cells around this one, and for the production
   * decomposition that is where every ghost's owner is. It is not where the
   * owner is under a COARSENED placement (the representative of an adjacent
   * coarse block is `group` fine cells away), nor when a body is large enough
   * that redistribution had to use the global path in the first place.
   *
   * What used to happen then: the request was dropped, the ghost kept its OLD
   * global id, and the closing redistribution -- which de-duplicates arrivals by
   * global id -- failed to match it against the authoritative copy. Both
   * survived, and repair_twins, which pairs by vertex SLOT, welded the surface
   * to itself along the partition boundary. Nothing asserts, nothing crashes,
   * and MFTLCheckMeshIntegrity reports a perfectly valid mesh.
   *
   * So: detect it, agree about it in one int, and fall back to the global
   * communicator. The fallback is not cheap -- it is an Alltoall over nproc --
   * but it is entered only when the neighbourhood provably cannot deliver, and
   * the alternative is a wrong answer. Everything below is written against
   * `n_slots` rather than against `nn` so that the two paths share it all. */
  int *const rank_to_slot = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "remesh ghost slot index");
  for (int r = 0; r < nproc; ++r) rank_to_slot[r] = MFTL_ERROR;
  for (int i = 0; i < nn; ++i) rank_to_slot[dom->neighbors[i]] = i;

  int needs_global = 0;
  for (int o = 0; o < n_local && !needs_global; ++o)
  {
    const MFTL_Object *const obj = objects[o];
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      const int owner = obj->verts.rec[v].owner_proc;
      if (owner < 0 || owner >= nproc || rank_to_slot[owner] == MFTL_ERROR)
      {
        needs_global = 1;
        break;
      }
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &needs_global, 1, MPI_INT, MPI_LOR, dom->comm_world);

  if (!needs_global && nn == 0)
  {
    free(rank_to_slot);
    return MFTL_SUCCESS;
  }

  const int n_slots = needs_global ? nproc : nn;
  if (needs_global)
  {
    for (int r = 0; r < nproc; ++r) rank_to_slot[r] = r;
  }

  /* ── 1. Every ghost we hold, grouped by owner ────────────────────────────
   * Two lists filled in lockstep: the REQUEST (body, old gid) that goes on the
   * wire, and the TARGET (body, local slot) that stays here so that the apply
   * phase is a direct indexed write. Ghosts are an array suffix, so the loop
   * bound is the ownership filter. */
  pair_list *const requests = (pair_list *) MFTLMallocOrDie((size_t) n_slots * sizeof(pair_list), "remesh ghost requests");
  pair_list *const targets = (pair_list *) MFTLMallocOrDie((size_t) n_slots * sizeof(pair_list), "remesh ghost targets");
  for (int i = 0; i < n_slots; ++i)
  {
    pl_init(&requests[i]);
    pl_init(&targets[i]);
  }

  int unserved = 0;
  for (int o = 0; o < n_local; ++o)
  {
    const MFTL_Object *const obj = objects[o];
    const MFTL_Vertex *const vr = obj->verts.rec;

    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
    {
      const int owner = vr[v].owner_proc;
      const int idx = (owner >= 0 && owner < nproc) ? rank_to_slot[owner] : MFTL_ERROR;
      if (idx == MFTL_ERROR)
      {
        /* Unreachable now that the global path exists -- an owner outside
         * [0, nproc) is a corrupt record, not a routing problem. Counted rather
         * than dereferenced, and asserted because it means something upstream is
         * already wrong. */
        MFTL_ASSERT(0 && "remesh ghost ids: ghost vertex with an owner_proc outside the communicator");
        unserved++;
        continue;
      }
      /* The REQUEST names the body by its global id -- the owner has its own
       * slot for it and no reason to have the same one. The TARGET never leaves
       * this rank, so it keeps the local slot and the apply phase stays a direct
       * indexed write. */
      pl_add2(&requests[idx], obj->global_id, vr[v].global_id);
      pl_add2(&targets[idx], o, v);
    }
  }
  free(rank_to_slot);

  /* ── 2. Counts, overlapped with packing the request image ────────────────
   * The one place non-blocking buys anything here: assembling the flat image is
   * real work that does not depend on the counts. Everything after it is a
   * strict dependency chain -- replies cannot be computed before the requests
   * arrive, nor applied before they come back -- so the two alltoallv calls are
   * blocking, which is the cheaper spelling of a dependency. */
  int *const n_send = (int *) MFTLMallocOrDie((size_t) (4 * n_slots) * sizeof(int), "remesh ghost counts");
  int *const n_recv = n_send + n_slots;
  int *const s_displs = n_send + 2 * n_slots;
  int *const r_displs = n_send + 3 * n_slots;

  for (int i = 0; i < n_slots; ++i) n_send[i] = requests[i].size;

  MPI_Request count_req;
  if (needs_global)
    MPI_Ialltoall(n_send, 1, MPI_INT, n_recv, 1, MPI_INT, dom->comm_world, &count_req);
  else
    MPI_Ineighbor_alltoall(n_send, 1, MPI_INT, n_recv, 1, MPI_INT, dom->comm_graph, &count_req);

  int total_send = 0;
  for (int i = 0; i < n_slots; ++i)
  {
    s_displs[i] = total_send;
    total_send += n_send[i];
  }

  int *const flat_send = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(total_send, 1) * sizeof(int), "remesh ghost request image");
  for (int i = 0; i < n_slots; ++i)
  {
    if (n_send[i] > 0) memcpy(flat_send + s_displs[i], requests[i].data, (size_t) n_send[i] * sizeof(int));
  }

  MPI_Wait(&count_req, MPI_STATUS_IGNORE);

  int total_recv = 0;
  for (int i = 0; i < n_slots; ++i)
  {
    r_displs[i] = total_recv;
    total_recv += n_recv[i];
  }

  /* ── 3. Requests out ─────────────────────────────────────────────────────── */
  int *const flat_recv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(total_recv, 1) * sizeof(int), "remesh ghost incoming requests");
  if (needs_global)
    MPI_Alltoallv(flat_send, n_send, s_displs, MPI_INT, flat_recv, n_recv, r_displs, MPI_INT, dom->comm_world);
  else
    MPI_Neighbor_alltoallv(flat_send, n_send, s_displs, MPI_INT, flat_recv, n_recv, r_displs, MPI_INT, dom->comm_graph);
  free(flat_send);

  /* ── 4. Answer them, one id table per body ───────────────────────────────
   * The stream arrives grouped by sender and, within a sender, by body, so
   * answering in arrival order would rebuild a body's table once per SENDER. A
   * counting sort by body -- O(requests + bodies), no comparisons -- makes it
   * once per body instead. Replies go back at the request's original position,
   * because the requester matches them positionally. */
  const int n_req = total_recv / 2;
  unsigned char *const reply = (unsigned char *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * width, "remesh ghost replies");
  int *const obj_off = (int *) MFTLCallocOrDie((size_t) n_local + 1, sizeof(int), "remesh ghost grouping");
  int *const order = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(int), "remesh ghost order");

  /* Resolve each request's body id to THIS rank's slot for it, ONCE, in place.
   * Everything downstream then buckets by local slot, so the counting sort and
   * its cursors are sized by what this rank holds rather than by the ensemble.
   * `flat_recv[2k+1]` -- the vertex id answer_body() actually reads -- is left
   * alone. A request for a body this rank has nothing of is a routing error, not
   * a partition this rank simply is not part of: the requester named this rank
   * as the vertex's OWNER. */
  for (int k = 0; k < n_req; ++k)
  {
    const int slot = MFTLEnsembleFindLocalSlot(ens, flat_recv[2 * k]);
    flat_recv[2 * k] = slot;
    if (slot == MFTL_ERROR)
    {
      MFTL_ASSERT(0 && "remesh ghost ids: request names a body this rank does not hold, but claims this rank owns the vertex");
      unserved++;
      continue;
    }
    obj_off[slot + 1]++;
  }
  for (int o = 0; o < n_local; ++o) obj_off[o + 1] += obj_off[o];

  {
    int *const cursor = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_local, 1) * sizeof(int), "remesh ghost cursors");
    for (int o = 0; o < n_local; ++o) cursor[o] = obj_off[o];

    for (int k = 0; k < n_req; ++k)
    {
      const int o = flat_recv[2 * k];
      if (o == MFTL_ERROR)
      {
        answer(MFTL_NULL, MFTL_ERROR, MFTL_ERROR, reply + (size_t) k * width, ctx);
        continue;
      }
      order[cursor[o]++] = k;
    }
    free(cursor);
  }

  {
    int table_len = 0;
    for (int o = 0; o < n_local; ++o)
    {
      if (obj_off[o + 1] > obj_off[o]) table_len = MFTL_MAX(table_len, objects[o]->next_vertex_gid);
    }

    int *const slot_of_gid = (table_len > 0) ? (int *) MFTLMallocOrDie((size_t) table_len * sizeof(int), "remesh ghost id table") : MFTL_NULL;

    for (int o = 0; o < n_local; ++o)
    {
      if (obj_off[o + 1] == obj_off[o]) continue;  // nothing asked of this body

      /* A request only arrives here when it names this rank as a vertex's
       * owner, so every body asked about is one this rank holds -- and the
       * answer can be keyed by LOCAL SLOT. */
      answer_body(objects[o], o, flat_recv, order, obj_off[o], obj_off[o + 1], reply, width, answer, ctx, slot_of_gid, objects[o]->next_vertex_gid,
                  &unserved);
    }
    free(slot_of_gid);
  }

  free(order);
  free(obj_off);
  free(flat_recv);

  /* ── 5. Replies back ─────────────────────────────────────────────────────
   * No second count exchange: one reply per request, so the reply counts are the
   * request counts halved and both sides already know both. Counted in BYTES,
   * so the id exchange still sends four per ghost. */
  const int w = (int) width;
  for (int i = 0; i < n_slots; ++i)
  {
    n_recv[i] = (n_recv[i] / 2) * w;
    n_send[i] = (n_send[i] / 2) * w;
    s_displs[i] = (s_displs[i] / 2) * w;
    r_displs[i] = (r_displs[i] / 2) * w;
  }

  const int n_in = total_send / 2;
  unsigned char *const incoming = (unsigned char *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * width, "remesh ghost replies in");
  if (needs_global)
    MPI_Alltoallv(reply, n_recv, r_displs, MPI_BYTE, incoming, n_send, s_displs, MPI_BYTE, dom->comm_world);
  else
    MPI_Neighbor_alltoallv(reply, n_recv, r_displs, MPI_BYTE, incoming, n_send, s_displs, MPI_BYTE, dom->comm_graph);
  free(reply);

  /* ── 6. Apply ────────────────────────────────────────────────────────────
   * The replies come back in exactly the order the requests went out, so the
   * target list resolves each one directly -- no hashing, no body search. */
  int p = 0;
  for (int i = 0; i < n_slots; ++i)
  {
    const int *const t = targets[i].data;
    const int count = targets[i].size;

    for (int k = 0; k < count; k += 2, ++p) apply(objects[t[k]], t[k + 1], incoming + (size_t) p * width);
  }

  free(incoming);
  for (int i = 0; i < n_slots; ++i)
  {
    pl_free(&requests[i]);
    pl_free(&targets[i]);
  }
  free(requests);
  free(targets);
  free(n_send);

  /* A non-zero count here means a request could not be answered -- the requester
   * believes a rank owns a vertex it does not have. That is a disagreement about
   * ownership, not a routing problem, and it leaves the ghost with a stale id
   * that the closing redistribution will duplicate rather than match. Loud in
   * Debug, reported in Release. */
  (void) what;
  return (unserved == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}

/** @brief Owner side of the id exchange: `base[slot] + v`, or #MFTL_ERROR. */
static void answer_ghost_id(const MFTL_Object *obj, int slot, int v, void *out, const void *ctx)
{
  (void) obj;
  /* Owned entities occupy [0, n_local) and are renumbered `base + slot`, so
   * the answer is arithmetic. That is the whole reason the re-densification
   * numbers BY SLOT rather than by any other rule -- any other rule would need
   * a second table here. `base` is keyed by LOCAL SLOT: it came out of a
   * per-body scan over that body's holders. */
  const int id = (v == MFTL_ERROR) ? MFTL_ERROR : ((const int *) ctx)[slot] + v;
  memcpy(out, &id, sizeof id);
}

/** @brief Requester side of the id exchange. */
static void apply_ghost_id(MFTL_Object *obj, int v, const void *in)
{
  int id;
  memcpy(&id, in, sizeof id);
  if (id != MFTL_ERROR) obj->verts.rec[v].global_id = id;  // owner could not serve it otherwise
}

int MFTLRemeshSyncGhostIds(MFTL_Ensemble *ens, const int *vertex_gid_base)
{
  MFTL_ASSERT(ens != MFTL_NULL && vertex_gid_base != MFTL_NULL);
  const int rc = ghost_vertex_exchange(ens, sizeof(int), answer_ghost_id, vertex_gid_base, apply_ghost_id, "MFTLRemeshSyncGhostIds");

  /* A failure means a request could not be answered -- the requester believes a
   * rank owns a vertex it does not have. That is a disagreement about
   * ownership, not a routing problem, and it leaves the ghost with a stale id
   * that the closing redistribution will duplicate rather than match. Loud in
   * Debug, reported in Release. */
  MFTL_ASSERT(rc == MFTL_SUCCESS && "remesh ghost ids: some ghost kept a stale global id -- the closing redistribution will duplicate it");
  return rc;
}

/** @brief Owner side of the sizing exchange: the owner's target length, or -1. */
static void answer_ghost_sizing(const MFTL_Object *obj, int slot, int v, void *out, const void *ctx)
{
  (void) slot;
  (void) ctx;
  /* -1 is the "not served" sentinel: every real target length is positive. */
  const double size = (v == MFTL_ERROR || !obj->verts.has_sizing) ? -1.0 : obj->verts.sizing[v];
  memcpy(out, &size, sizeof size);
}

/** @brief Requester side of the sizing exchange. */
static void apply_ghost_sizing(MFTL_Object *obj, int v, const void *in)
{
  double size;
  memcpy(&size, in, sizeof size);
  if (size > 0.0 && obj->verts.has_sizing) obj->verts.sizing[v] = size;
}

int MFTLRemeshSyncGhostSizing(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  return ghost_vertex_exchange(ens, sizeof(double), answer_ghost_sizing, MFTL_NULL, apply_ghost_sizing, "MFTLRemeshSyncGhostSizing");
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Routed records, classification, positions and the projection halo
 * ═══════════════════════════════════════════════════════════════════════════ */

double *MFTLRemeshExchangeRecords(const MFTL_Domain *domain, const double *rec, const int *dest, int n_rec, int width, int *n_out)
{
  const int nproc = domain->nproc, me = domain->rank, nn = domain->num_neighbors;

  int n_self = 0;
  for (int i = 0; i < n_rec; ++i) n_self += (dest[i] == me);

  if (nproc == 1)
  {
    double *out = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_self * width, 1) * sizeof(double), "remesh records");
    if (n_self > 0) memcpy(out, rec, (size_t) n_self * width * sizeof(double)); /* rec may be NULL when empty */
    *n_out = n_self;
    return out;
  }

  /* The route: the graph when every destination is a neighbour, agreed. */
  int *const rank_to_slot = (int *) MFTLMallocOrDie((size_t) nproc * sizeof(int), "remesh records slot index");
  for (int r = 0; r < nproc; ++r) rank_to_slot[r] = MFTL_ERROR;
  for (int i = 0; i < nn; ++i) rank_to_slot[domain->neighbors[i]] = i;
  int needs_global = 0;
  for (int i = 0; i < n_rec && !needs_global; ++i)
    if (dest[i] != me && (dest[i] < 0 || dest[i] >= nproc || rank_to_slot[dest[i]] == MFTL_ERROR)) needs_global = 1;
  MPI_Allreduce(MPI_IN_PLACE, &needs_global, 1, MPI_INT, MPI_LOR, domain->comm_world);

  const int n_slots = needs_global ? nproc : nn;
  if (needs_global)
    for (int r = 0; r < nproc; ++r) rank_to_slot[r] = r;

  int *const cnt = (int *) MFTLCallocOrDie((size_t) 4 * MFTL_MAX(n_slots, 1), sizeof(int), "remesh records counts");
  int *const rcnt = cnt + n_slots, *const sdis = cnt + 2 * n_slots, *const rdis = cnt + 3 * n_slots;
  for (int i = 0; i < n_rec; ++i)
    if (dest[i] != me) cnt[rank_to_slot[dest[i]]] += width;
  int stot = 0;
  for (int k = 0; k < n_slots; ++k) sdis[k] = stot, stot += cnt[k];
  double *const sbuf = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(stot, 1) * sizeof(double), "remesh records send");
  int *const fill = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(n_slots, 1), sizeof(int), "remesh records fill");
  for (int i = 0; i < n_rec; ++i)
  {
    if (dest[i] == me) continue;
    const int k = rank_to_slot[dest[i]];
    memcpy(&sbuf[sdis[k] + fill[k]], &rec[(size_t) i * width], (size_t) width * sizeof(double));
    fill[k] += width;
  }
  free(fill);

  if (needs_global)
    MPI_Alltoall(cnt, 1, MPI_INT, rcnt, 1, MPI_INT, domain->comm_world);
  else
    MPI_Neighbor_alltoall(cnt, 1, MPI_INT, rcnt, 1, MPI_INT, domain->comm_graph);
  int rtot = 0;
  for (int k = 0; k < n_slots; ++k) rdis[k] = rtot, rtot += rcnt[k];

  double *const out = (double *) MFTLMallocOrDie((size_t) MFTL_MAX((size_t) n_self * width + rtot, 1) * sizeof(double), "remesh records received");
  int w = 0;
  for (int i = 0; i < n_rec; ++i)
    if (dest[i] == me) memcpy(&out[(size_t) (w++) * width], &rec[(size_t) i * width], (size_t) width * sizeof(double));
  double *const rbuf = out + (size_t) n_self * width;
  if (needs_global)
    MPI_Alltoallv(sbuf, cnt, sdis, MPI_DOUBLE, rbuf, rcnt, rdis, MPI_DOUBLE, domain->comm_world);
  else
    MPI_Neighbor_alltoallv(sbuf, cnt, sdis, MPI_DOUBLE, rbuf, rcnt, rdis, MPI_DOUBLE, domain->comm_graph);

  free(sbuf);
  free(cnt);
  free(rank_to_slot);
  *n_out = n_self + rtot / width;
  return out;
}

/** @brief Owner side of the classification exchange: its FEATURE/CORNER bits, or -1. */
static void answer_ghost_class(const MFTL_Object *obj, int slot, int v, void *out, const void *ctx)
{
  (void) slot;
  (void) ctx;
  const int bits = (v == MFTL_ERROR) ? -1 : (obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER));
  memcpy(out, &bits, sizeof bits);
}

/** @brief Requester side of the classification exchange. */
static void apply_ghost_class(MFTL_Object *obj, int v, const void *in)
{
  int bits;
  memcpy(&bits, in, sizeof bits);
  if (bits < 0) return;
  obj->verts.rec[v].flags = (obj->verts.rec[v].flags & ~(MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER)) | bits;
}

int MFTLRemeshSyncGhostClass(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  return ghost_vertex_exchange(ens, sizeof(int), answer_ghost_class, MFTL_NULL, apply_ghost_class, "MFTLRemeshSyncGhostClass");
}

/** @brief Owner side of the position exchange. */
static void answer_ghost_position(const MFTL_Object *obj, int slot, int v, void *out, const void *ctx)
{
  (void) slot;
  (void) ctx;
  double p[4] = {0.0, 0.0, 0.0, -1.0};
  if (v != MFTL_ERROR)
  {
    for (int k = 0; k < 3; ++k) p[k] = obj->verts.pos[3 * v + k];
    p[3] = 1.0;
  }
  memcpy(out, p, sizeof p);
}

/** @brief Requester side of the position exchange. */
static void apply_ghost_position(MFTL_Object *obj, int v, const void *in)
{
  double p[4];
  memcpy(p, in, sizeof p);
  if (p[3] < 0.0) return;
  for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] = p[k];
}

int MFTLRemeshSyncGhostPositions(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  const int rc = ghost_vertex_exchange(ens, 4 * sizeof(double), answer_ghost_position, MFTL_NULL, apply_ghost_position, "MFTLRemeshSyncGhostPositions");
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    ens->local_objects[o]->derived &= ~MFTL_DERIVED_GEOMETRY;
    MFTLObjectMarkChanged(ens->local_objects[o]);
  }
  return rc;
}

int MFTLRemeshReferenceHalo(MFTL_Ensemble *ens, double width, double **out, int *n_out)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int me = dom->rank;

  /* This rank's canonical cell. */
  int my_cell[3] = {0, 0, 0};
  for (int c = 0; c < dom->nproc; ++c)
    if (dom->rank_map[c] == me)
    {
      my_cell[0] = c % dom->cuts[0];
      my_cell[1] = (c / dom->cuts[0]) % dom->cuts[1];
      my_cell[2] = c / (dom->cuts[0] * dom->cuts[1]);
      break;
    }
  double h[3], centre[3];
  for (int k = 0; k < 3; ++k)
  {
    h[k] = dom->domain_size[k] / (double) dom->cuts[k];
    centre[k] = dom->global_min[k] + ((double) my_cell[k] + 0.5) * h[k];
  }
  /* A margin far below any length the mesh resolves, so a cell face that a
   * box only touches in exact arithmetic still counts. */
  const double reach = width * (1.0 + 1e-9);

  int n_rec = 0, cap = 0;
  double *rec = MFTL_NULL;
  int *dst = MFTL_NULL;

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
      int i0[3], i1[3];
      for (int k = 0; k < 3; ++k)
      {
        i0[k] = (int) floor((lo[k] - dom->global_min[k]) / h[k]);
        i1[k] = (int) floor((hi[k] - dom->global_min[k]) / h[k]);
      }
      int sent[64], n_sent = 0;
      for (int a = i0[0]; a <= i1[0]; ++a)
        for (int b = i0[1]; b <= i1[1]; ++b)
          for (int e = i0[2]; e <= i1[2]; ++e)
          {
            int q[3] = {a, b, e};
            bool inside = MFTL_TRUE;
            for (int k = 0; k < 3; ++k)
            {
              if (dom->is_periodic[k])
                q[k] = ((q[k] % dom->cuts[k]) + dom->cuts[k]) % dom->cuts[k];
              else if (q[k] < 0 || q[k] >= dom->cuts[k])
                inside = MFTL_FALSE;
            }
            if (!inside) continue;
            const int r = dom->rank_map[q[0] + dom->cuts[0] * (q[1] + dom->cuts[1] * q[2])];
            if (r == me) continue;
            bool dup = MFTL_FALSE;
            for (int j = 0; j < n_sent; ++j) dup |= (sent[j] == r);
            if (dup || n_sent == 64) continue;
            sent[n_sent++] = r;
            if (n_rec == cap)
            {
              cap = MFTL_MAX(64, 2 * cap);
              rec = (double *) MFTLReallocOrDie(rec, (size_t) cap * MFTL_REMESH_REF_HALO_WIDTH * sizeof(double), "remesh reference halo");
              dst = (int *) MFTLReallocOrDie(dst, (size_t) cap * sizeof(int), "remesh reference halo destinations");
            }
            MFTLRemeshReferencePackTriangle(obj, t, &rec[(size_t) n_rec * MFTL_REMESH_REF_HALO_WIDTH]);
            dst[n_rec++] = r;
          }
    }
  }

  *out = MFTLRemeshExchangeRecords(dom, rec, dst, n_rec, MFTL_REMESH_REF_HALO_WIDTH, n_out);
  free(rec);
  free(dst);
  return MFTL_SUCCESS;
}
