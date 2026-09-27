/*
 * MFTL2 parallel — ensembles and their domain: MPI Cartesian / rectilinear decomposition,
 * the slab -> rank map, the 26-neighbour distributed-graph communicator, object membership,
 * and temporary domains for load-balanced remeshing.
 *
 * Periodic axes wrap the neighbour search, so on a 1- or 2-slab periodic axis a rank can be
 * its own neighbour or the same rank can appear on both sides; the neighbour list is
 * de-duplicated. The committed MPI datatypes are registered with the first live ensemble and
 * released with the last one (reference count).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_memory.h"

// number of live ensembles; the MPI datatypes exist while it is > 0
static int g_mftl_ensemble_ref_count = 0;

// Builds the Cartesian communicator, the slab -> rank map (unless supplied), the neighbour
// list and the distributed-graph communicator of a domain whose bounds/cuts are set.
static int mftlDomainInitComm(MFTL_Domain* domain)
{
  if (!domain) return MFTL_ERROR;

  int periods[3] = {(int) domain->is_periodic[0], (int) domain->is_periodic[1], (int) domain->is_periodic[2]};
  // no reordering: ranks keep their order in the input communicator
  int reorder = 0;

  if (MPI_Cart_create(domain->comm_world, 3, domain->cuts, periods, reorder, &domain->comm_cart) != MPI_SUCCESS)
  {
    return MFTL_ERROR;
  }

  for (int d = 0; d < 3; d++)
  {
    domain->grid_factors[d] = domain->cuts[d] / domain->domain_size[d];
  }

  // A caller-supplied slab -> rank map (MFTLEnsembleCreateRectilinear) is kept; otherwise the
  // MPI Cartesian order is used.
  if (domain->rank_map == MFTL_NULL)
  {
    domain->rank_map = (int*) malloc(domain->nproc * sizeof(int));

    int coords[3];
    for (int k = 0; k < domain->cuts[2]; ++k)
    {
      for (int j = 0; j < domain->cuts[1]; ++j)
      {
        for (int i = 0; i < domain->cuts[0]; ++i)
        {
          coords[0] = i;
          coords[1] = j;
          coords[2] = k;
          int rank_out;
          MPI_Cart_rank(domain->comm_cart, coords, &rank_out);

          int linear_idx = i + (j * domain->cuts[0]) + (k * domain->cuts[0] * domain->cuts[1]);
          domain->rank_map[linear_idx] = rank_out;
        }
      }
    }
  }

  // Neighbours: the up to 26 slabs of the 3x3x3 block around this rank's slab. The slab
  // coordinates come from the rank map (identical to MPI_Cart_coords for the Cartesian map).
  int my_coords[3] = {0, 0, 0};
  for (int idx = 0; idx < domain->nproc; ++idx)
  {
    if (domain->rank_map[idx] != domain->rank) continue;
    my_coords[0] = idx % domain->cuts[0];
    my_coords[1] = (idx / domain->cuts[0]) % domain->cuts[1];
    my_coords[2] = idx / (domain->cuts[0] * domain->cuts[1]);
    break;
  }

  int neighbor_ranks[26];
  int num_neighbors = 0;

  for (int dz = -1; dz <= 1; ++dz)
  {
    for (int dy = -1; dy <= 1; ++dy)
    {
      for (int dx = -1; dx <= 1; ++dx)
      {
        if (dx == 0 && dy == 0 && dz == 0)
        {
          continue;
        }

        int neighbor_coords[3] = {my_coords[0] + dx, my_coords[1] + dy, my_coords[2] + dz};
        bool is_valid_neighbor = MFTL_TRUE;

        for (int i = 0; i < 3; ++i)
        {
          if (periods[i])
          {
            neighbor_coords[i] = (neighbor_coords[i] + domain->cuts[i]) % domain->cuts[i];
          }
          else
          {
            if (neighbor_coords[i] < 0 || neighbor_coords[i] >= domain->cuts[i])
            {
              is_valid_neighbor = MFTL_FALSE;
              break;
            }
          }
        }

        if (is_valid_neighbor)
        {
          int nidx = neighbor_coords[0] + neighbor_coords[1] * domain->cuts[0] + neighbor_coords[2] * domain->cuts[0] * domain->cuts[1];
          neighbor_ranks[num_neighbors++] = domain->rank_map[nidx];
        }
      }
    }
  }

  // periodic wrap on 1- or 2-slab axes repeats ranks (possibly this rank itself)
  int unique_neighbor_ranks[26];
  int num_unique_neighbors = 0;

  for (int j = 0; j < num_neighbors; ++j)
  {
    int d = neighbor_ranks[j];
    int found = 0;
    for (int k = 0; k < num_unique_neighbors; ++k)
    {
      if (unique_neighbor_ranks[k] == d)
      {
        found = 1;
        break;
      }
    }

    if (!found)
    {
      unique_neighbor_ranks[num_unique_neighbors++] = d;
    }
  }

  // symmetric adjacency: sources == destinations
  if (MPI_Dist_graph_create_adjacent(domain->comm_world,
                                     num_unique_neighbors,  // indegree
                                     unique_neighbor_ranks,  // sources
                                     MPI_UNWEIGHTED,  // source weights
                                     num_unique_neighbors,  // outdegree
                                     unique_neighbor_ranks,  // destinations
                                     MPI_UNWEIGHTED,  // destination weights
                                     MPI_INFO_NULL,  // info
                                     reorder,  // allow reordering
                                     &domain->comm_graph) != MPI_SUCCESS)
  {
    fprintf(stderr, "MFTL Error (Rank %d): Failed to create distributed graph communicator.\n", domain->rank);
    MPI_Comm_free(&domain->comm_cart);
    return MFTL_ERROR;
  }

  int weighted;
  MPI_Dist_graph_neighbors_count(domain->comm_graph, &domain->num_neighbors, &domain->num_neighbors, &weighted);
  domain->neighbors = (int*) malloc(domain->num_neighbors * sizeof(int));

  int* temp_neighbors = (int*) malloc(domain->num_neighbors * sizeof(int));
  MPI_Dist_graph_neighbors(domain->comm_graph, domain->num_neighbors, domain->neighbors, MPI_UNWEIGHTED, domain->num_neighbors, temp_neighbors, MPI_UNWEIGHTED);
  free(temp_neighbors);

  return MFTL_SUCCESS;
}

MFTL_Ensemble* MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3], int initial_object_capacity)
{
  MFTL_Ensemble* ens = (MFTL_Ensemble*) calloc(1, sizeof(MFTL_Ensemble));
  if (!ens) return MFTL_NULL;

  MFTL_Domain* dom = &ens->domain;

  MPI_Comm_dup(comm, &dom->comm_world);
  MPI_Comm_size(dom->comm_world, &dom->nproc);
  MPI_Comm_rank(dom->comm_world, &dom->rank);

  if (cuts[0] * cuts[1] * cuts[2] != dom->nproc)
  {
    if (dom->rank == 0) fprintf(stderr, "MFTL Error: Cuts do not match MPI Size.\n");
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  for (int i = 0; i < 3; ++i)
  {
    dom->cuts[i] = cuts[i];
    dom->global_min[i] = global_min[i];
    dom->global_max[i] = global_max[i];
    dom->domain_size[i] = global_max[i] - global_min[i];
    dom->domain_half_size[i] = dom->domain_size[i] * 0.5;
    if (is_periodic) dom->is_periodic[i] = is_periodic[i];
  }

  if (mftlDomainInitComm(dom) != MFTL_SUCCESS)
  {
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  ens->capacity_objects = (initial_object_capacity > 0) ? initial_object_capacity : 1;
  ens->objects = (MFTL_Object**) calloc(ens->capacity_objects, sizeof(MFTL_Object*));
  ens->num_objects = 0;

  if (g_mftl_ensemble_ref_count == 0)
  {
    MFTLRegisterMpiTypes();
  }
  g_mftl_ensemble_ref_count++;

  return ens;
}

void MFTLEnsembleDestroy(MFTL_Ensemble** p_ens)
{
  if (!p_ens || !(*p_ens)) return;
  MFTL_Ensemble* ens = *p_ens;

  for (int i = 0; i < ens->num_objects; ++i)
  {
    MFTLObjectDestroy(&ens->objects[i]);
  }
  free(ens->objects);

  MFTL_Domain* dom = &ens->domain;
  if (dom->comm_graph != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_graph);
  if (dom->comm_cart != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_cart);
  if (dom->comm_world != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_world);
  if (dom->neighbors) free(dom->neighbors);
  if (dom->rank_map) free(dom->rank_map);
  if (dom->lb_xcuts) free(dom->lb_xcuts);
  for (int d = 0; d < 3; ++d) free(dom->axis_cuts[d]);

  free(ens);
  *p_ens = MFTL_NULL;

  g_mftl_ensemble_ref_count--;
  if (g_mftl_ensemble_ref_count <= 0)
  {
    g_mftl_ensemble_ref_count = 0;
    MFTLFreeMpiTypes();
  }
}

int MFTLEnsembleAddObject(MFTL_Ensemble* ensemble, MFTL_Object* object)
{
  object->domain = &ensemble->domain;
  MFTL_RESERVE_ADDITIONAL(ensemble->objects, &ensemble->num_objects, &ensemble->capacity_objects, 1, MFTL_Object*);
  ensemble->objects[ensemble->num_objects++] = object;
  return MFTL_SUCCESS;
}

/* ========================================================================== */
/* Rectilinear decompositions                                                 */
/* ========================================================================== */

// Copies the rectilinear description into an (already comm-initialised) domain and builds
// the communicators. Returns MFTL_ERROR on an inconsistent description.
static int mftlDomainSetupRectilinear(MFTL_Domain* dom, const int cuts[3], const double* const axis_cuts[3], const int* rank_map, const double global_min[3],
                                      const double global_max[3], const bool is_periodic[3])
{
  if (cuts[0] * cuts[1] * cuts[2] != dom->nproc)
  {
    if (dom->rank == 0) fprintf(stderr, "MFTL Error: Cuts do not match MPI Size.\n");
    return MFTL_ERROR;
  }
  for (int d = 0; d < 3; ++d)
  {
    dom->cuts[d] = cuts[d];
    dom->global_min[d] = global_min[d];
    dom->global_max[d] = global_max[d];
    dom->domain_size[d] = global_max[d] - global_min[d];
    dom->domain_half_size[d] = dom->domain_size[d] * 0.5;
    dom->is_periodic[d] = is_periodic ? is_periodic[d] : MFTL_FALSE;
    dom->axis_cuts[d] = MFTL_NULL;
    if (axis_cuts && axis_cuts[d])
    {
      for (int i = 0; i < cuts[d]; ++i)
        if (!(axis_cuts[d][i] <= axis_cuts[d][i + 1]))
        {
          if (dom->rank == 0) fprintf(stderr, "MFTL Error: axis cuts along %d are not ascending.\n", d);
          return MFTL_ERROR;
        }
      dom->axis_cuts[d] = (double*) malloc((size_t) (cuts[d] + 1) * sizeof(double));
      memcpy(dom->axis_cuts[d], axis_cuts[d], (size_t) (cuts[d] + 1) * sizeof(double));
    }
  }
  dom->rank_map = MFTL_NULL;
  if (rank_map)
  {
    // must be a bijection slab -> rank
    char* seen = (char*) calloc((size_t) dom->nproc, 1);
    int ok = 1;
    for (int i = 0; i < dom->nproc; ++i)
    {
      if (rank_map[i] < 0 || rank_map[i] >= dom->nproc || seen[rank_map[i]]) ok = 0;
      if (ok) seen[rank_map[i]] = 1;
    }
    free(seen);
    if (!ok)
    {
      if (dom->rank == 0) fprintf(stderr, "MFTL Error: the slab -> rank map is not a bijection.\n");
      return MFTL_ERROR;
    }
    dom->rank_map = (int*) malloc((size_t) dom->nproc * sizeof(int));
    memcpy(dom->rank_map, rank_map, (size_t) dom->nproc * sizeof(int));
  }
  dom->lb_nprocs = 0;
  dom->lb_xmin = dom->lb_xmax = 0.0;
  dom->lb_xcuts = MFTL_NULL;
  return mftlDomainInitComm(dom);
}

MFTL_Ensemble* MFTLEnsembleCreateRectilinear(MPI_Comm comm, const int cuts[3], const double* const axis_cuts[3], const int* rank_map,
                                             const double global_min[3], const double global_max[3], const bool is_periodic[3], int initial_object_capacity)
{
  MFTL_Ensemble* ens = (MFTL_Ensemble*) calloc(1, sizeof(MFTL_Ensemble));
  if (!ens) return MFTL_NULL;
  MFTL_Domain* dom = &ens->domain;
  dom->comm_cart = MPI_COMM_NULL;
  dom->comm_graph = MPI_COMM_NULL;
  MPI_Comm_dup(comm, &dom->comm_world);
  MPI_Comm_size(dom->comm_world, &dom->nproc);
  MPI_Comm_rank(dom->comm_world, &dom->rank);

  // same bookkeeping as MFTLEnsembleCreate (the MPI types are registered before any exit
  // path can reach MFTLEnsembleDestroy, which releases them by reference count)
  if (g_mftl_ensemble_ref_count == 0) MFTLRegisterMpiTypes();
  g_mftl_ensemble_ref_count++;

  if (mftlDomainSetupRectilinear(dom, cuts, axis_cuts, rank_map, global_min, global_max, is_periodic) != MFTL_SUCCESS)
  {
    MFTLEnsembleDestroy(&ens);
    return MFTL_NULL;
  }

  ens->capacity_objects = (initial_object_capacity > 0) ? initial_object_capacity : 1;
  ens->objects = (MFTL_Object**) calloc(ens->capacity_objects, sizeof(MFTL_Object*));
  ens->num_objects = 0;
  return ens;
}

int MFTLEnsembleSetDecomposition(MFTL_Ensemble* ens, const int cuts[3], const double* const axis_cuts[3], const int* rank_map, const double global_min[3],
                                 const double global_max[3], const bool is_periodic[3])
{
  MFTL_Domain* dom = &ens->domain;
  if (dom->comm_graph != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_graph);
  if (dom->comm_cart != MPI_COMM_NULL) MPI_Comm_free(&dom->comm_cart);
  free(dom->neighbors);
  dom->neighbors = MFTL_NULL;
  dom->num_neighbors = 0;
  free(dom->rank_map);
  free(dom->lb_xcuts);
  for (int d = 0; d < 3; ++d)
  {
    free(dom->axis_cuts[d]);
    dom->axis_cuts[d] = MFTL_NULL;
  }
  return mftlDomainSetupRectilinear(dom, cuts, axis_cuts, rank_map, global_min, global_max, is_periodic);
}

/* ========================================================================== */
/* Detaching / transferring objects                                           */
/* ========================================================================== */

int MFTLEnsembleRemoveObject(MFTL_Ensemble* ensemble, MFTL_Object* object)
{
  for (int i = 0; i < ensemble->num_objects; ++i)
  {
    if (ensemble->objects[i] != object) continue;
    memmove(&ensemble->objects[i], &ensemble->objects[i + 1], (size_t) (ensemble->num_objects - i - 1) * sizeof(MFTL_Object*));
    ensemble->num_objects--;
    object->domain = MFTL_NULL;
    return MFTL_SUCCESS;
  }
  return MFTL_ERROR;
}

int MFTLEnsembleTransferObject(MFTL_Ensemble* dst, MFTL_Ensemble* src, MFTL_Object* object)
{
  if (MFTLEnsembleRemoveObject(src, object) != MFTL_SUCCESS) return MFTL_ERROR;
  return MFTLEnsembleAddObject(dst, object);
}

/* ========================================================================== */
/* Temporary domains (x-only load-balancing layout on 3-D grids)              */
/* ========================================================================== */

int MFTLDomainCreateTemporary(const MFTL_Domain* base, const int cuts[3], MFTL_Domain* tmp)
{
  memset(tmp, 0, sizeof(*tmp));
  tmp->comm_world = base->comm_world;  // borrowed: NOT freed by MFTLDomainFreeTemporary
  tmp->rank = base->rank;
  tmp->nproc = base->nproc;
  tmp->comm_cart = MPI_COMM_NULL;
  tmp->comm_graph = MPI_COMM_NULL;
  if (cuts[0] * cuts[1] * cuts[2] != base->nproc) return MFTL_ERROR;
  for (int d = 0; d < 3; ++d)
  {
    tmp->cuts[d] = cuts[d];
    tmp->global_min[d] = base->global_min[d];
    tmp->global_max[d] = base->global_max[d];
    tmp->domain_size[d] = base->domain_size[d];
    tmp->domain_half_size[d] = base->domain_half_size[d];
    tmp->is_periodic[d] = base->is_periodic[d];
  }
  tmp->rank_map = (int*) malloc((size_t) base->nproc * sizeof(int));
  for (int i = 0; i < base->nproc; ++i) tmp->rank_map[i] = i;
  return mftlDomainInitComm(tmp);
}

void MFTLDomainFreeTemporary(MFTL_Domain* tmp)
{
  if (tmp->comm_graph != MPI_COMM_NULL) MPI_Comm_free(&tmp->comm_graph);
  if (tmp->comm_cart != MPI_COMM_NULL) MPI_Comm_free(&tmp->comm_cart);
  free(tmp->neighbors);
  free(tmp->rank_map);
  free(tmp->lb_xcuts);
  for (int d = 0; d < 3; ++d) free(tmp->axis_cuts[d]);
  memset(tmp, 0, sizeof(*tmp));
}
