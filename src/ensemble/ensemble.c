/**
 * @file ensemble.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief Ensemble lifecycle: the MPI/physical domain plus its list of objects.
 */

#include "mftl_ensemble.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
        if (!present) unique_neighbors[num_unique++] = neighbor_rank;
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

  // --- 2. Validate the process grid ---
  if (cuts[0] < 1 || cuts[1] < 1 || cuts[2] < 1 || cuts[0] * cuts[1] * cuts[2] != dom->nproc)
  {
    if (dom->rank == 0)
    {
      fprintf(stderr, "MFTL Error: Cuts {%d, %d, %d} do not match MPI size %d.\n", cuts[0], cuts[1], cuts[2], dom->nproc);
    }
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  // --- 3. Copy bounds and flags ---
  for (int i = 0; i < 3; ++i)
  {
    dom->cuts[i] = cuts[i];
    dom->global_min[i] = global_min[i];
    dom->global_max[i] = global_max[i];
    dom->domain_size[i] = global_max[i] - global_min[i];
    dom->domain_half_size[i] = dom->domain_size[i] * 0.5;
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
  ens->objects = (MFTL_Object **) MFTLCallocOrDie((size_t) ens->capacity_objects, sizeof(MFTL_Object *), "MFTL ensemble object list");
  ens->num_objects = 0;

  return ens;
}

void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens)
{
  if (!p_ens || !(*p_ens)) return;
  MFTL_Ensemble *ens = *p_ens;

  // 1. Destroy every object the ensemble owns
  for (int i = 0; i < ens->num_objects; ++i) MFTLObjectDestroy(&ens->objects[i]);
  free(ens->objects);

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

  // Grow exactly by one -- objects are added one at a time.
  MFTL_RESERVE_ADDITIONAL(ensemble->objects, &ensemble->num_objects, &ensemble->capacity_objects, 1, MFTL_Object *);
  ensemble->objects[ensemble->num_objects++] = object;

  return MFTL_SUCCESS;
}

void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble)
{
  if (!ensemble) return;

  const int num_objects = ensemble->num_objects;
  if (num_objects == 0)
  {
    ensemble->total_global_triangles = 0;
    return;
  }

  /* ── One reduction for the whole ensemble ────────────────────────────────
   *
   * Layout: [obj0_tris, obj0_verts, obj1_tris, obj1_verts, ...]. The message
   * count is 1 regardless of how many objects there are, which is the property
   * that matters at scale -- an Allreduce per object would put O(num_objects)
   * latencies in series.
   *
   * Blocking: the buffer is 2 ints per object, so this is pure latency, and the
   * caller has nothing to overlap it with (the counts are what it asked for).
   *
   * The local half is now two array reads per object. The previous version
   * scanned every vertex and every triangle testing an `is_ghost` flag first,
   * which is O(V + T) of pure memory traffic to recover a number the ownership
   * prefix already carries. */
  int *local_counts = (int *) MFTLMallocOrDie(2 * (size_t) num_objects * sizeof(int), "MFTLEnsembleUpdateGlobalCounts");
  int *global_counts = (int *) MFTLMallocOrDie(2 * (size_t) num_objects * sizeof(int), "MFTLEnsembleUpdateGlobalCounts");

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    MFTL_ASSERT(obj != MFTL_NULL);
    local_counts[2 * o + 0] = obj->tris.n_local;
    local_counts[2 * o + 1] = obj->verts.n_local;
  }

  // Assumes every rank holds the same object list in the same order.
  MPI_Allreduce(local_counts, global_counts, 2 * num_objects, MPI_INT, MPI_SUM, ensemble->domain.comm_world);

  int total = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    obj->global_n_triangles = global_counts[2 * o + 0];
    obj->global_n_vertices = global_counts[2 * o + 1];
    total += obj->global_n_triangles;
  }
  ensemble->total_global_triangles = total;

  free(local_counts);
  free(global_counts);
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
