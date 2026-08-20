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
  const int n = ens->num_objects;

  int need = 0;
  for (int o = 0; o < n; ++o) need += ens->objects[o]->verts.n + ens->objects[o]->tris.n;

  if (n > m->cap_objects)
  {
    free(m->maps);
    m->cap_objects = (m->cap_objects > 0) ? m->cap_objects : 1;
    while (m->cap_objects < n) m->cap_objects *= 2;
    m->maps = (MFTL_OwnerMap *) MFTLMallocOrDie((size_t) m->cap_objects * sizeof(MFTL_OwnerMap), "remesh owner maps");
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
    const MFTL_Object *const obj = ens->objects[o];
    m->maps[o].vert_owner = m->arena + off;
    off += obj->verts.n;
    m->maps[o].tri_owner = m->arena + off;
    off += obj->tris.n;
  }
}

#ifndef NDEBUG
/**
 * @brief Debug check: every triangle's cached centre still matches its vertices.
 *
 * The same check `compute_owners` makes for the geometric rule, repeated here
 * because an owner map turns that one off -- a map is not obliged to route by
 * geometry at all, and the dispersed rule does not. The shifted rule DOES, so it
 * owes the check itself.
 *
 * It covers GHOSTS as well as owned triangles, and that is the whole point. A
 * rank routes its ghosts in order to work out which of its OWN triangles the
 * ghost's owner is about to need; if only the owned centres were refreshed, two
 * ranks would disagree about a ghost's destination and someone's owned triangle
 * would silently end up without one of its edge neighbours. Nothing crashes.
 *
 * @param obj The mesh.
 * @return `true` if every triangle's cached centre still matches its vertices.
 *         Debug only -- a migration routes ghosts by that centre, so a stale one
 *         makes two ranks disagree about a destination.
 */
static bool centres_are_current(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double fresh[3];
    MFTLGetTriangleProperties(obj, t, MFTL_NULL, MFTL_NULL, fresh);

    /* Both are wrapped positions, so compare THROUGH the seam, not across it. */
    double d[3];
    MFTLGetMinimumImageVector(obj->domain, d, &obj->tris.center[3 * t], fresh);
    if (fabs(d[0]) > 1e-9 || fabs(d[1]) > 1e-9 || fabs(d[2]) > 1e-9) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}
#endif

int MFTLRemeshMigrateShifted(MFTL_RemeshMigrator *m, MFTL_Ensemble *ens, const int group[3], const double shift[3], MFTL_RemeshMigrationReport *report)
{
  MFTL_ASSERT(m != MFTL_NULL && ens != MFTL_NULL && group != MFTL_NULL && shift != MFTL_NULL);
  if (report)
  {
    report->n_moved = 0;
    report->performed = MFTL_FALSE;
  }
  if (ens->num_objects == 0) return MFTL_SUCCESS;

  const MFTL_Domain *const dom = &ens->domain;
  const int my_rank = dom->rank;
  int moved = 0;

  migrator_reserve(m, ens);

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];

    MFTL_ASSERT(centres_are_current(obj) &&
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
  if (ens->num_objects == 0) return MFTL_SUCCESS;

  const int my_rank = ens->domain.rank;
  int moved = 0;

  migrator_reserve(m, ens);

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
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

void MFTLRemeshIdReduce(const MFTL_Domain *domain, const int *send, int *prefix, int *total, int n_counts, int n_extra)
{
  if (domain->nproc < 2)
  {
    memset(prefix, 0, (size_t) n_counts * sizeof(int));
    memcpy(total, send, (size_t) (n_counts + n_extra) * sizeof(int));
    return;
  }

  memset(prefix, 0, (size_t) n_counts * sizeof(int));  // MPI_Exscan leaves rank 0's recvbuf untouched

  MPI_Request req[2];
  MPI_Iexscan(send, prefix, n_counts, MPI_INT, MPI_SUM, domain->comm_world, &req[0]);
  MPI_Iallreduce(send, total, n_counts + n_extra, MPI_INT, MPI_SUM, domain->comm_world, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
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
 * @brief How many times MFTLRemeshMeasureBodies has run in this process.
 *
 * Not a statistic anybody tunes on -- it exists so that a test can assert the
 * number of measurements a pass takes, which is the only way the "measure once"
 * arrangement stays true. It was two before, and the second call recomputed an
 * identical answer at 7*num_objects doubles a time.
 */
static int g_measurement_count = 0;

int MFTLRemeshMeasurementCount(void) { return g_measurement_count; }

void MFTLRemeshResetMeasurementCount(void) { g_measurement_count = 0; }

void MFTLRemeshMeasureBodies(MFTL_Ensemble *ens, double *centroid, double *radius)
{
  const MFTL_Domain *const dom = &ens->domain;
  const int n = ens->num_objects;
  if (n < 1) return;

  g_measurement_count++;

  /* Layout per body: [w, s0, c0, s1, c1, s2, c2]. A periodic axis uses the
   * (sin, cos) pair, a non-periodic one puts its plain weighted sum in the `s`
   * slot and leaves `c` zero.
   *
   * SEVEN, not eight. The eighth slot was padding, justified by making the
   * per-body stride a power of two -- which buys an address calculation nothing
   * a reduction cares about, and costs 12.5 % of the message. At the ensemble
   * size this collective is FOR (an emulsion of ten thousand droplets is 560 kB
   * per rank per pass, not 640) that is the only figure that matters. */
  enum
  {
    STRIDE = 7
  };
  double *const acc = (double *) MFTLCallocOrDie((size_t) (2 * STRIDE * n), sizeof(double), "remesh body accumulators");
  double *const sum = acc + STRIDE * n;

  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
    double *const a = acc + STRIDE * o;

    /* OWNED triangles only. A ghost is another rank's triangle and it is
     * contributing to the same reduction from there, so counting it here would
     * weight the seam of every partition twice -- by an amount that grows with
     * the rank count, which is the one shape of error a global measurement must
     * not have. */
    const int nt = obj->tris.n_local;
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

  if (dom->nproc > 1)
    MPI_Allreduce(acc, sum, STRIDE * n, MPI_DOUBLE, MPI_SUM, dom->comm_world);
  else
    memcpy(sum, acc, (size_t) (STRIDE * n) * sizeof(double));

  /* Scratch for the centroid even when the caller does not want it -- the radius
   * is a distance to it. */
  double *const ctr = centroid ? centroid : (double *) MFTLMallocOrDie((size_t) (3 * n) * sizeof(double), "remesh centroids");

  for (int o = 0; o < n; ++o)
  {
    const double *const s = sum + STRIDE * o;
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

  if (radius)
  {
    double *const rmax = (double *) MFTLCallocOrDie((size_t) (2 * n), sizeof(double), "remesh body radii");
    double *const rsum = rmax + n;

    for (int o = 0; o < n; ++o)
    {
      const MFTL_Object *const obj = ens->objects[o];
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
      rmax[o] = sqrt(best);
    }

    if (dom->nproc > 1)
      MPI_Allreduce(rmax, rsum, n, MPI_DOUBLE, MPI_MAX, dom->comm_world);
    else
      memcpy(rsum, rmax, (size_t) n * sizeof(double));

    memcpy(radius, rsum, (size_t) n * sizeof(double));
    free(rmax);
  }

  if (!centroid) free(ctr);
  free(acc);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The ghost global-id exchange
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
  MFTL_Object *const *const objects = ens->objects;
  const int num_objects = ens->num_objects;
  const int nproc = dom->nproc;
  const int nn = dom->num_neighbors;

  if (num_objects == 0 || nproc == 1) return MFTL_SUCCESS;

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
  for (int o = 0; o < num_objects && !needs_global; ++o)
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
  for (int o = 0; o < num_objects; ++o)
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
      pl_add2(&requests[idx], o, vr[v].global_id);
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
  int *const obj_off = (int *) MFTLCallocOrDie((size_t) num_objects + 1, sizeof(int), "remesh ghost grouping");
  int *const order = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(int), "remesh ghost order");

  for (int k = 0; k < n_req; ++k)
  {
    const int o = flat_recv[2 * k];
    if (o >= 0 && o < num_objects)
      obj_off[o + 1]++;
    else
      unserved++;
  }
  for (int o = 0; o < num_objects; ++o) obj_off[o + 1] += obj_off[o];

  {
    int *const cursor = (int *) MFTLMallocOrDie((size_t) num_objects * sizeof(int), "remesh ghost cursors");
    for (int o = 0; o < num_objects; ++o) cursor[o] = obj_off[o];

    for (int k = 0; k < n_req; ++k)
    {
      const int o = flat_recv[2 * k];
      if (o < 0 || o >= num_objects)
      {
        MFTL_ASSERT(0 && "remesh ghost ids: request names a body outside the list -- ranks disagree about the object list");
        reply[k] = MFTL_ERROR;
        continue;
      }
      order[cursor[o]++] = k;
    }
    free(cursor);
  }

  {
    int table_len = 0;
    for (int o = 0; o < num_objects; ++o)
    {
      if (obj_off[o + 1] > obj_off[o]) table_len = MFTL_MAX(table_len, objects[o]->next_vertex_gid);
    }

    int *const slot_of_gid = (table_len > 0) ? (int *) MFTLMallocOrDie((size_t) table_len * sizeof(int), "remesh ghost id table") : MFTL_NULL;

    for (int o = 0; o < num_objects; ++o)
    {
      if (obj_off[o + 1] == obj_off[o]) continue;  // nothing asked of this body
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
