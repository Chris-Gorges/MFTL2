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

#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Migration
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLRemeshMigratorInit(MFTL_RemeshMigrator *m)
{
  m->maps = MFTL_NULL;
  m->arena = MFTL_NULL;
  m->cap_objects = 0;
  m->cap_entities = 0;
}

void MFTLRemeshMigratorFree(MFTL_RemeshMigrator *m)
{
  free(m->maps);
  free(m->arena);
  MFTLRemeshMigratorInit(m);
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
  return MFTLRedistributeWithOwners(ens, m->maps);
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

bool MFTLRemeshParamsAgree(const MFTL_Domain *domain, const MFTL_RemeshParameters *params)
{
  if (domain->nproc < 2) return MFTL_TRUE;

  /* One MIN over the values AND their negations, rather than broadcasting rank
   * 0's copy and testing equality: a minimum of -x is a maximum of x, so the two
   * halves recover both extremes from a single reduction, and every rank learns
   * that the ranks disagree. A broadcast-and-compare fires on one rank while the
   * others walk into the next collective and hang. */
  double v[4];
  v[0] = params->min_edge_length;
  v[1] = params->max_edge_length;
  v[2] = -params->min_edge_length;
  v[3] = -params->max_edge_length;
  MPI_Allreduce(MPI_IN_PLACE, v, 4, MPI_DOUBLE, MPI_MIN, domain->comm_world);

  return (v[0] == -v[2]) && (v[1] == -v[3]);
}

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
 * @brief Answers every request addressed to one body.
 *
 * The owner has to turn an old global id into a pool slot, which is the one
 * lookup in this module that is not already an index. It needs no hash map for
 * the same reason the ghost sync does not: ids are DENSE per entity class, so a
 * scatter of the owned vertices into an array sized by `next_vertex_gid` answers
 * every request with one indexed read.
 *
 * O(owned) to build and O(requests) to use, so it is built once per body that
 * was actually asked something -- which is what the counting sort in the caller
 * is for.
 *
 * @param      obj         The mesh whose OWNED vertices answer the requests.
 * @param      gid_base    First NEW id this rank was allotted for this body.
 * @param      flat_recv   The received request ids, all senders concatenated.
 * @param      order       Request indices belonging to this body.
 * @param      first       First slot of @p order to serve.
 * @param      last        One past the last slot to serve.
 * @param[out] reply       One new id per request; #MFTL_ERROR when unanswerable.
 * @param      slot_of_gid Scatter table from OLD global id to pool slot.
 * @param      table_len   Addressable length of @p slot_of_gid.
 * @param[out] n_unserved  Incremented once per unanswerable request.
 */
static void answer_body(const MFTL_Object *obj, int gid_base, const int *restrict flat_recv, const int *restrict order, int first, int last,
                        int *restrict reply, int *restrict slot_of_gid, int table_len, int *n_unserved)
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
    const int slot = (gid >= 0 && gid < table_len) ? slot_of_gid[gid] : MFTL_ERROR;

    if (slot == MFTL_ERROR)
    {
      /* The requester believes this rank owns a vertex it does not have. Report
       * it rather than dereferencing garbage: the requester skips the reply and
       * keeps its stale id, which the caller turns into MFTL_ERROR. */
      MFTL_ASSERT(0 && "remesh ghost ids: asked for a vertex this rank does not own");
      reply[k] = MFTL_ERROR;
      (*n_unserved)++;
      continue;
    }

    /* Owned entities occupy [0, n_local) and are renumbered `base + slot`, so
     * the answer is arithmetic. That is the whole reason the re-densification
     * numbers BY SLOT rather than by any other rule -- any other rule would need
     * a second table here. */
    reply[k] = gid_base + slot;
  }
}

int MFTLRemeshSyncGhostIds(MFTL_Ensemble *ens, const int *vertex_gid_base)
{
  MFTL_ASSERT(ens != MFTL_NULL && vertex_gid_base != MFTL_NULL);

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
  int *const reply = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(int), "remesh ghost replies");
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
        reply[k] = MFTL_ERROR;
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

      /* `vertex_gid_base` is keyed by LOCAL SLOT. It came out of a per-body scan
       * over that body's holders, so it describes the bodies THIS RANK holds and
       * nothing else -- which is all this rank can ever be asked about, because a
       * request only arrives here when it names this rank as a vertex's owner. */
      answer_body(objects[o], vertex_gid_base[o], flat_recv, order, obj_off[o], obj_off[o + 1], reply, slot_of_gid, objects[o]->next_vertex_gid, &unserved);
    }
    free(slot_of_gid);
  }

  free(order);
  free(obj_off);
  free(flat_recv);

  /* ── 5. Replies back ─────────────────────────────────────────────────────
   * No second count exchange: one reply per request, so the reply counts are the
   * request counts halved and both sides already know both. */
  for (int i = 0; i < n_slots; ++i)
  {
    n_recv[i] /= 2;
    n_send[i] /= 2;
    s_displs[i] /= 2;
    r_displs[i] /= 2;
  }

  const int n_in = total_send / 2;
  int *const incoming = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * sizeof(int), "remesh ghost new ids");
  if (needs_global)
    MPI_Alltoallv(reply, n_recv, r_displs, MPI_INT, incoming, n_send, s_displs, MPI_INT, dom->comm_world);
  else
    MPI_Neighbor_alltoallv(reply, n_recv, r_displs, MPI_INT, incoming, n_send, s_displs, MPI_INT, dom->comm_graph);
  free(reply);

  /* ── 6. Apply ────────────────────────────────────────────────────────────
   * The replies come back in exactly the order the requests went out, so the
   * target list resolves each one directly -- no hashing, no body search. */
  int p = 0;
  for (int i = 0; i < n_slots; ++i)
  {
    const int *const t = targets[i].data;
    const int count = targets[i].size;

    for (int k = 0; k < count; k += 2, ++p)
    {
      if (incoming[p] == MFTL_ERROR) continue;  // owner could not serve it
      objects[t[k]]->verts.rec[t[k + 1]].global_id = incoming[p];
    }
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
  MFTL_ASSERT(unserved == 0 && "remesh ghost ids: some ghost kept a stale global id -- the closing redistribution will duplicate it");
  return (unserved == 0) ? MFTL_SUCCESS : MFTL_ERROR;
}
