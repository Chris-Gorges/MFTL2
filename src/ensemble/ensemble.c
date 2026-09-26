/**
 * @file ensemble.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief Ensemble lifecycle: the MPI/physical domain plus its list of objects.
 */

#include "mftl_ensemble.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/**
 * @brief Builds the MPI topology of a domain: Cartesian grid, rank map, neighbour graph.
 *
 * All three are set up ONCE here so that nothing on a hot path has to call an
 * MPI topology function again:
 *
 *   comm_cart   the Cartesian communicator, which is what defines the mapping
 *               from grid coordinates to ranks;
 *   rank_map    that mapping materialised as a flat array, because
 *               MPI_Cart_rank is a function call with internal bookkeeping and
 *               owner lookup happens once per triangle per redistribution;
 *   comm_graph  a distributed graph over the (up to) 26 surrounding ranks, which
 *               is what makes neighbourhood collectives available. Those are the
 *               right primitive for halo exchange: the message count per rank is
 *               bounded by the neighbourhood degree rather than by P, so the
 *               communication cost does not grow when the job does.
 *
 * On failure the domain is left with whatever communicators were created so far
 * set back to MPI_COMM_NULL, so MFTLEnsembleDestroy() can clean up safely.
 *
 * @param domain The domain whose `comm_cart`, `comm_graph`, `rank_map`,
 *               `neighbors` and `num_neighbors` are established. `comm_world` must
 *               already be set.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the Cartesian or graph communicator
 *         could not be created.
 */
static int domain_init_comm(MFTL_Domain *domain)
{
  if (!domain) return MFTL_ERROR;

  // --- 1. Cartesian communicator ---
  int periods[3] = {(int) domain->is_periodic[0], (int) domain->is_periodic[1], (int) domain->is_periodic[2]};
  // Reordering is disabled so ranks keep the same order as the input communicator.
  const int reorder = 0;

  if (MPI_Cart_create(domain->comm_world, 3, domain->cuts, periods, reorder, &domain->comm_cart) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create Cartesian communicator.\n", domain->rank);
    domain->comm_cart = MPI_COMM_NULL;
    return MFTL_ERROR;
  }

  // --- 2. Rank map and grid factors ---
  // grid_factors turns the per-axis division in MFTLGetOwnerProc into a multiply.
  for (int d = 0; d < 3; ++d)
  {
    domain->grid_factors[d] = (domain->domain_size[d] > 0.0) ? (double) domain->cuts[d] / domain->domain_size[d] : 0.0;
  }

  domain->rank_map = (int *) MFTLMallocOrDie((size_t) domain->nproc * sizeof(int), "MFTL domain rank_map");

  for (int k = 0; k < domain->cuts[2]; ++k)
  {
    for (int j = 0; j < domain->cuts[1]; ++j)
    {
      for (int i = 0; i < domain->cuts[0]; ++i)
      {
        int coords[3] = {i, j, k};
        int rank_out;
        MPI_Cart_rank(domain->comm_cart, coords, &rank_out);
        domain->rank_map[i + j * domain->cuts[0] + k * domain->cuts[0] * domain->cuts[1]] = rank_out;
      }
    }
  }

  // --- 3. Collect the (up to) 26 surrounding neighbours ---
  int my_coords[3];
  MPI_Cart_coords(domain->comm_cart, domain->rank, 3, my_coords);

  int unique_neighbors[26];
  signed char unique_offsets[26][3];
  int num_unique = 0;

  for (int dz = -1; dz <= 1; ++dz)
  {
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        if (dx == 0 && dy == 0 && dz == 0) continue;  // skip myself

        int nc[3] = {my_coords[0] + dx, my_coords[1] + dy, my_coords[2] + dz};
        bool valid = MFTL_TRUE;

        for (int i = 0; i < 3; ++i)
        {
          if (periods[i])
          {
            // Periodic wrap-around; the + cuts handles negative results correctly.
            nc[i] = (nc[i] + domain->cuts[i]) % domain->cuts[i];
          }
          else if (nc[i] < 0 || nc[i] >= domain->cuts[i])
          {
            valid = MFTL_FALSE;
            break;
          }
        }

        if (!valid) continue;

        int neighbor_rank;
        MPI_Cart_rank(domain->comm_cart, nc, &neighbor_rank);

        /* De-duplicate as we go. With a thin decomposition (e.g. cuts = {2,1,1})
         * periodic wrap-around maps several offsets onto the same rank, and a
         * distributed graph must not list a neighbour twice. The list is at most
         * 26 entries, so the linear scan is cheaper than any hash. */
        bool present = MFTL_FALSE;
        for (int n = 0; n < num_unique; ++n)
        {
          if (unique_neighbors[n] == neighbor_rank)
          {
            present = MFTL_TRUE;
            break;
          }
        }
        if (!present)
        {
          /* The OFFSET is kept with the rank, not the wrapped cell coordinate:
           * it is what MFTLEnsembleReduceBodies() tests a body's holder set
           * against, and it needs no seam correction. The first offset wins
           * where wrap-around aliases several onto one rank -- see
           * MFTL_Domain::neighbor_offset. */
          unique_offsets[num_unique][0] = (signed char) dx;
          unique_offsets[num_unique][1] = (signed char) dy;
          unique_offsets[num_unique][2] = (signed char) dz;
          unique_neighbors[num_unique++] = neighbor_rank;
        }
      }
    }
  }

  // --- 4. Distributed graph communicator ---
  // For an adjacent graph the sources of incoming messages are the same as the
  // destinations of outgoing messages.
  if (MPI_Dist_graph_create_adjacent(domain->comm_world, num_unique, unique_neighbors, MPI_UNWEIGHTED, num_unique, unique_neighbors, MPI_UNWEIGHTED,
                                     MPI_INFO_NULL, reorder, &domain->comm_graph) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create distributed graph communicator.\n", domain->rank);
    domain->comm_graph = MPI_COMM_NULL;
    return MFTL_ERROR;
  }

  // --- 5. Read the neighbour list back from MPI ---
  int indegree = 0, outdegree = 0, weighted = 0;
  MPI_Dist_graph_neighbors_count(domain->comm_graph, &indegree, &outdegree, &weighted);
  MFTL_ASSERT(indegree == outdegree && "MFTL assumes a symmetric neighbour graph");

  domain->num_neighbors = indegree;
  domain->neighbors = (int *) MFTLMallocOrDie((size_t) (indegree > 0 ? indegree : 1) * sizeof(int), "MFTL domain neighbours");
  int *dest = (int *) MFTLMallocOrDie((size_t) (outdegree > 0 ? outdegree : 1) * sizeof(int), "MFTL domain neighbours (dest)");

  MPI_Dist_graph_neighbors(domain->comm_graph, indegree, domain->neighbors, MPI_UNWEIGHTED, outdegree, dest, MPI_UNWEIGHTED);
  free(dest);

  /* MPI is free to report the neighbours in an order of its own, so the offsets
   * are matched back by RANK rather than by position. 26 entries against 26 is
   * the same linear scan the de-duplication above uses, for the same reason. */
  for (int n = 0; n < indegree; ++n)
  {
    domain->neighbor_offset[n][0] = 0;
    domain->neighbor_offset[n][1] = 0;
    domain->neighbor_offset[n][2] = 0;
    for (int u = 0; u < num_unique; ++u)
    {
      if (unique_neighbors[u] != domain->neighbors[n]) continue;
      domain->neighbor_offset[n][0] = unique_offsets[u][0];
      domain->neighbor_offset[n][1] = unique_offsets[u][1];
      domain->neighbor_offset[n][2] = unique_offsets[u][2];
      break;
    }
  }

  return MFTL_SUCCESS;
}

MFTL_Ensemble *MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3], int initial_object_capacity)
{
  if (!cuts || !global_min || !global_max) return MFTL_NULL;

  MFTL_Ensemble *ens = (MFTL_Ensemble *) calloc(1, sizeof(MFTL_Ensemble));
  if (!ens) return MFTL_NULL;

  MFTL_Domain *dom = &ens->domain;

  /* calloc zeroes the struct, but MPI_COMM_NULL is not guaranteed to be 0 (in
   * Open MPI it is a pointer to a static object). The handles must therefore be
   * set explicitly, otherwise the cleanup path would call MPI_Comm_free on a
   * zeroed handle. */
  dom->comm_world = MPI_COMM_NULL;
  dom->comm_cart = MPI_COMM_NULL;
  dom->comm_graph = MPI_COMM_NULL;

  // --- 1. Duplicate the world communicator so the library owns its own ---
  if (MPI_Comm_dup(comm, &dom->comm_world) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error: Failed to duplicate the MPI communicator.\n");
    dom->comm_world = MPI_COMM_NULL;
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }
  MPI_Comm_size(dom->comm_world, &dom->nproc);
  MPI_Comm_rank(dom->comm_world, &dom->rank);

  // --- 2. Validate the process grid and the bounds ---
  /* The product in 64 bits: three int cuts can overflow an int before the
   * comparison, and an overflow that happens to equal nproc would be accepted. */
  if (cuts[0] < 1 || cuts[1] < 1 || cuts[2] < 1 || (long long) cuts[0] * cuts[1] * cuts[2] != (long long) dom->nproc)
  {
    if (dom->rank == 0)
    {
      fprintf(stderr, "MFTL Error: Cuts {%d, %d, %d} do not match MPI size %d.\n", cuts[0], cuts[1], cuts[2], dom->nproc);
    }
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  /* A zero-length axis is legal (a flat decomposition), an inverted or non-finite
   * one is not: it would silently produce negative sizes, a negative reciprocal
   * and owner lookups that clamp everything into one cell. `!(a >= b)` also
   * rejects NaN. */
  for (int i = 0; i < 3; ++i)
  {
    if (!(global_max[i] >= global_min[i]) || !isfinite(global_min[i]) || !isfinite(global_max[i]))
    {
      if (dom->rank == 0) fprintf(stderr, "MFTL Error: invalid domain bounds on axis %d: [%g, %g].\n", i, global_min[i], global_max[i]);
      MFTLEnsembleDestroy(&ens);
      return MFTL_NULL;
    }
  }

  // --- 3. Copy bounds and flags ---
  for (int i = 0; i < 3; ++i)
  {
    dom->cuts[i] = cuts[i];
    dom->global_min[i] = global_min[i];
    dom->global_max[i] = global_max[i];
    dom->domain_size[i] = global_max[i] - global_min[i];
    dom->inv_domain_size[i] = (dom->domain_size[i] > 0.0) ? 1.0 / dom->domain_size[i] : 0.0;
    dom->is_periodic[i] = is_periodic ? is_periodic[i] : MFTL_FALSE;
  }

  // --- 4. Build the MPI topology ---
  if (domain_init_comm(dom) != MFTL_SUCCESS)
  {
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  // --- 5. Set up the object list ---
  ens->capacity_objects = (initial_object_capacity > 0) ? initial_object_capacity : 1;
  ens->local_objects = (MFTL_Object **) MFTLCallocOrDie((size_t) ens->capacity_objects, sizeof(MFTL_Object *), "MFTL ensemble object list");
  ens->n_local_objects = 0;

  return ens;
}

void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens)
{
  if (!p_ens || !(*p_ens)) return;
  MFTL_Ensemble *ens = *p_ens;

  // 1. Destroy every object the ensemble owns
  for (int i = 0; i < ens->n_local_objects; ++i) MFTLObjectDestroy(&ens->local_objects[i]);
  free(ens->local_objects);

  // 1a. And the communication module's scratch, if a redistribution built one
  MFTLCommWorkspaceDestroy(&ens->comm_workspace);

  // 2. Release the domain resources
  MFTL_Domain *dom = &ens->domain;
  if (dom->comm_graph != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_graph);
  if (dom->comm_cart != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_cart);
  if (dom->comm_world != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_world);
  free(dom->neighbors);
  free(dom->rank_map);

  // 3. Free the ensemble itself
  free(ens);
  *p_ens = MFTL_NULL;
}

int MFTLEnsembleAddObject(MFTL_Ensemble *ensemble, MFTL_Object *object)
{
  if (!ensemble || !object) return MFTL_ERROR;

  object->domain = &ensemble->domain;

  /* ── The ensemble hands out the body id, the caller does not ─────────────
   *
   * Whatever MFTLObjectCreate() was given is provisional and is overwritten
   * here, so `global_id` is DENSE in [0, n_bodies_global) by construction. Two
   * things follow and both are the point:
   *
   * A dense id is an ARRAY INDEX. Every keyed per-body structure -- a vote
   * bitset, an owner map, an arrival lookup -- can be a plain array without a
   * hash, which is the same argument that keeps vertex and triangle ids dense.
   *
   * And it is the identity bodies are travelling under. While every rank holds
   * the same list in the same order, `global_id == ` the list position, so
   * nothing observable changes; what changes is which of the two the code NAMES,
   * and the position is the one that cannot survive a rank holding only some of
   * the bodies.
   *
   * Every rank must therefore add the same bodies in the same order -- which was
   * already required, and is now checked by the redistribution's Debug
   * checksum rather than merely assumed. */
  object->global_id = ensemble->n_bodies_global++;

  /* Ids are handed out in add order and the list is appended to, so it stays
   * sorted -- which is the directory invariant, not a coincidence. */
  return MFTLEnsembleInsertLocalObject(ensemble, object);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The local object directory
 *
 * `local_objects` is SORTED BY ASCENDING global_id, strictly increasing, and
 * three things depend on that rather than one:
 *
 *   1. `gid -> slot` is a BINARY SEARCH -- an array, not a hash. That is what
 *      "there are no hash maps" asks for, and the justification it asks for is
 *      that the id space is dense and global while the holding is sparse, so the
 *      only array form of the lookup is O(bodies in the ensemble).
 *
 *   2. Iterating the local list visits bodies in ascending gid, i.e. in the
 *      GLOBAL order restricted to this rank. Every deterministic tie-break that
 *      says "bodies in order" therefore keeps meaning the same thing on a rank
 *      that holds only some of them.
 *
 *   3. Arrival runs arrive grouped, so they join against the directory with a
 *      merge rather than a search per record.
 *
 * Insert and erase are ORDERED -- a memmove of at most `n_local_objects`
 * pointers. Swap-with-tail would be O(1) and is wrong: it destroys the order and
 * with it points 1 and 2.
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLEnsembleFindLocalSlot(const MFTL_Ensemble *ens, int body_gid)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL) return MFTL_ERROR;

  int lo = 0, hi = ens->n_local_objects - 1;
  while (lo <= hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    const int g = ens->local_objects[mid]->global_id;
    if (g == body_gid) return mid;
    if (g < body_gid)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return MFTL_ERROR;
}

MFTL_Object *MFTLEnsembleFindLocalObject(const MFTL_Ensemble *ens, int body_gid)
{
  const int slot = MFTLEnsembleFindLocalSlot(ens, body_gid);
  return (slot == MFTL_ERROR) ? MFTL_NULL : ens->local_objects[slot];
}

int MFTLEnsembleInsertLocalObject(MFTL_Ensemble *ens, MFTL_Object *object)
{
  if (ens == MFTL_NULL || object == MFTL_NULL) return MFTL_ERROR;

  /* GEOMETRIC, not exact. Bodies are added one at a time, so the exact form
   * reallocates once per add: N adds past the initial capacity are N reallocs
   * and N copies of the whole list, i.e. O(N^2) bytes moved to build an ensemble.
   * The slack a doubling schedule leaves behind is one pointer per body, against
   * the 264 B object it points at. */
  MFTL_ENSURE_CAPACITY(ens->local_objects, &ens->n_local_objects, &ens->capacity_objects, 1, MFTL_Object *);

  /* Scan from the END. Both callers append in ascending id -- MFTLEnsembleAddObject
   * hands out the next id, and a body arriving in a redistribution is resolved
   * from a run stream that is itself ascending -- so the common case is zero
   * iterations and the memmove below moves nothing. */
  int i = ens->n_local_objects;
  while (i > 0 && ens->local_objects[i - 1]->global_id > object->global_id) --i;

  MFTL_ASSERT((i == 0 || ens->local_objects[i - 1]->global_id != object->global_id) && "MFTLEnsembleInsertLocalObject: that body is already here");

  if (i < ens->n_local_objects) memmove(&ens->local_objects[i + 1], &ens->local_objects[i], (size_t) (ens->n_local_objects - i) * sizeof(MFTL_Object *));

  ens->local_objects[i] = object;
  ens->n_local_objects++;
  return MFTL_SUCCESS;
}

void MFTLEnsembleEraseLocalSlot(MFTL_Ensemble *ens, int slot)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL || slot < 0 || slot >= ens->n_local_objects) return;

  if (slot + 1 < ens->n_local_objects)
  {
    memmove(&ens->local_objects[slot], &ens->local_objects[slot + 1], (size_t) (ens->n_local_objects - slot - 1) * sizeof(MFTL_Object *));
  }
  ens->n_local_objects--;
}

void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble)
{
  if (!ensemble) return;

  /* GLOBAL, not local: collectives follow, so the guard has to be a number every
   * rank agrees about. Returning on the LOCAL count would have a rank holding
   * nothing skip a collective the others are inside. */
  if (ensemble->n_bodies_global == 0)
  {
    ensemble->total_global_triangles = 0;
    return;
  }

  const int n_local = ensemble->n_local_objects;

  /* ── Two sums per body, over the body's HOLDERS ──────────────────────────
   *
   * Indexed by LOCAL SLOT, so the buffer is what this rank holds rather than
   * the id space. That is the whole change: the reduction used to be an
   * `MPI_Allreduce` of `2 * bodies` ints over comm_world on every rank, so a
   * rank learned two numbers about a hundred bodies by reducing two million.
   * MFTLEnsembleReduceBodies() decides which communicator that becomes.
   *
   * Doubles rather than ints because the reduction is one mechanism for every
   * per-body payload in the library, and an entity count is exact in a double
   * far past any mesh that fits in memory. */
  static const MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};

  double *const counts = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(2 * n_local, 1) * sizeof(double), "MFTLEnsembleUpdateGlobalCounts");

  /* The local half is two array reads per body held. Scanning every vertex and
   * every triangle for an `is_ghost` flag first is O(V + T) of pure memory
   * traffic to recover a number the ownership prefix already carries. */
  long local_tris = 0;
  for (int o = 0; o < n_local; ++o)
  {
    MFTL_Object *const obj = ensemble->local_objects[o];
    MFTL_ASSERT(obj != MFTL_NULL);
    counts[2 * o + 0] = (double) obj->tris.n_local;
    counts[2 * o + 1] = (double) obj->verts.n_local;
    local_tris += obj->tris.n_local;
  }

  MFTLEnsembleReduceBodies(ensemble, counts, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  for (int o = 0; o < n_local; ++o)
  {
    MFTL_Object *const obj = ensemble->local_objects[o];
    obj->global_n_triangles = (int) counts[2 * o + 0];
    obj->global_n_vertices = (int) counts[2 * o + 1];
  }

  free(counts);

  /* The ensemble total is a genuinely ensemble-wide scalar, so it gets a scalar
   * collective -- ONE long, whatever the body count. Summing the reduced
   * per-body counts instead would need every body's answer on every rank, which
   * is precisely the O(bodies) buffer the reduction above stopped allocating. */
  long total = 0;
  MPI_Allreduce(&local_tris, &total, 1, MPI_LONG, MPI_SUM, ensemble->domain.comm_world);
  ensemble->total_global_triangles = (int) total;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Decomposition metrics
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTLDomainGetHMin(const MFTL_Domain *domain)
{
  MFTL_ASSERT(domain != MFTL_NULL);

  double h_min = MFTL_DBL_MAX;
  for (int k = 0; k < 3; ++k)
  {
    /* A degenerate axis (zero extent, or a single cut) is not a constraint on
     * the frozen band, so it must not be allowed to become the minimum -- an
     * ensemble that is effectively 2D would otherwise report h_min = 0 and fail
     * every geometric precondition the shift scheme has. */
    if (domain->cuts[k] < 2 || domain->domain_size[k] <= 0.0) continue;
    h_min = MFTL_MIN(h_min, domain->domain_size[k] / (double) domain->cuts[k]);
  }

  return (h_min == MFTL_DBL_MAX) ? 0.0 : h_min;
}
