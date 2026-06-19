/*
 * @Author: Christian Gorges
 * @Date: 2026-06-17 11:03:52
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2026-06-17 11:03:52
 */

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

/* ══════════════════════════════════════════════════════════════════════
 * MFTLEnsembleGetTypicalRadius
 *
 * Returns the maximum bounding-box width (largest dimension) across all
 * objects in the ensemble, reduced globally with MPI_MAX so every rank
 * sees the same value.
 *
 * "Typical radius" here means the widest per-axis extent of the widest
 * object in the ensemble.  A bounding-box width is used (rather than a
 * sphere radius) because surface-mesh objects can be highly elongated —
 * jets, sheets, or flattened drops — for which a sphere radius severely
 * underestimates the true size in the critical axis.
 *
 * Returns      : Global max bounding-box width in physical length units.
 *                Returns 0.0 if the ensemble has no live vertices.
 *
 * Complexity : O(N_local_vertices) locally + one MPI_Allreduce.
 * ══════════════════════════════════════════════════════════════════════ */
double MFTLEnsembleGetTypicalRadius(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL && "MFTLEnsembleGetTypicalRadius: ensemble is NULL\n");

  MFTL_Domain *dom = &ens->domain;
  double local_max = 0.0;

  MFTL_Object *object_base = ens->objects;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *obj = &object_base[o];
    MFTL_ASSERT(obj != MFTL_NULL && "MFTLEnsembleGetTypicalRadius: object is NULL\n");

    double bbox_min[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
    double bbox_max[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};

    MFTL_Vertex *vertex_base = obj->vertices;

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &vertex_base[i];

      if (v->is_ghost) continue;

      for (int j = 0; j < 3; j++)
      {
        bbox_min[j] = MFTL_MIN(bbox_min[j], v->pos[j]);
        bbox_max[j] = MFTL_MAX(bbox_max[j], v->pos[j]);
      }
    }

    /* Largest extent in any single axis.
         * The per-axis maximum matches the 26-neighbour reachability
         * criterion (which is also per-axis), and correctly reflects
         * elongated objects that are wide in one axis but thin in others. */
    double obj_width[3];
    MFTLGetMinimumImageVector(dom, obj_width, bbox_min, bbox_max);
    local_max = MFTL_MAX(MFTL_MAX(local_max, obj_width[0]), MFTL_MAX(obj_width[1], obj_width[2]));
  }

  /* Global reduction: every rank must agree on the algorithm, so we
     * take the MAX across all ranks in the ensemble communicator.
     * MPI_AVG would mask the rare large-object case; MPI_MAX is correct. */
  double global_max = 0.0;
  MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, dom->comm_graph);

  return global_max;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLSelectRemeshAlgorithm
 *
 * Determines the optimal remeshing algorithm for this ensemble at this
 * remesh event and stores the result in ens->last_algorithm.
 *
 * Decision logic
 * ──────────────
 *
 *  SPANNING ensemble  →  always MFTL_ALGO_SHIFT_SCHEME
 *    Objects are by construction larger than rank boxes; centering
 *    migration would require non-neighbour communication and could
 *    exceed per-rank memory.
 *
 *  DISPERSED ensemble with  W_typical < 0.5 * h_min
 *                         →  MFTL_ALGO_CENTERING
 *    All objects fit comfortably within the 26-neighbour shell of their
 *    centroid rank.  Centering migration is strictly cheaper than the
 *    shift scheme: one migration round vs. 2–8 phases, and the serial
 *    kernel runs on fully-local, ungated meshes.
 *
 *  DISPERSED ensemble with  W_typical >= 0.5 * h_min
 *                         →  MFTL_ALGO_SHIFT_SCHEME
 *    Extreme strong scaling: bubbles have grown to rank-box scale, or
 *    MFTLEnsembleRebalance() has not yet promoted oversized objects.
 *    The shift scheme is universally correct for any W/h ratio.
 *
 * Collective: all ranks in ens->domain.comm_cart MUST call this
 * function simultaneously (MFTLEnsembleGetTypicalRadius issues one
 * MPI_Allreduce internally).
 *
 * Complexity : O(N_local_vertices) + one MPI_Allreduce.
 * ══════════════════════════════════════════════════════════════════════ */
MFTL_RemeshAlgorithm MFTLSelectRemeshAlgorithm(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(params != MFTL_NULL);

  /* ── SPANNING: shift scheme unconditionally ─────────────────────── */
  if (ens->type == MFTL_ENSEMBLE_SPANNING)
  {
    ens->last_algorithm = MFTL_ALGO_SHIFT_SCHEME;
    return MFTL_ALGO_SHIFT_SCHEME;
  }

  /* ── DISPERSED: decide from object size vs. rank-box width ─────── *
     *
     * MFTLEnsembleGetTypicalRadius() already performs the MPI_Allreduce,
     * so w_typical is globally consistent on all ranks after this call.  *
     * ─────────────────────────────────────────────────────────────────── */
  double h_min = MFTLDomainGetHMin(&ens->domain);
  double w_typical = MFTLEnsembleGetTypicalRadius(ens);
  double threshold = MFTL_CENTERING_WIDTH_FACTOR * h_min;

  if (w_typical < threshold)
  {
    /* Objects are small relative to rank boxes.                      *
         * All triangles of any single object lie within the 26-neighbour *
         * shell of its centroid rank → centering migration is safe and   *
         * uses only MPI_Neighbor_alltoallv (no non-local communication). */
    ens->last_algorithm = MFTL_ALGO_CENTERING;
  }
  else
  {
    /* Objects have reached rank-box scale.                           *
         * Centering would require non-neighbour communication or exceed  *
         * single-rank memory.  The shift scheme handles this correctly   *
         * regardless of the W/h ratio.                                   *
         *                                                                *
         * Caller (MFTLRemeshEnsemble) has already verified               *
         *   h_min > 4 * delta                                            *
         * via MFTLComputeCoarseningFactor, so the shift scheme's hard    *
         * geometric constraint is known to hold at this point.           */
    ens->last_algorithm = MFTL_ALGO_SHIFT_SCHEME;
  }

  return ens->last_algorithm;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLComputeCoarseningFactor
 *
 * Returns G = cbrt_G^3, the smallest perfect-cube coarsening factor such
 * that  h_min * G^(1/3) > 4 * delta  AND  cbrt_G divides cuts[k] in all
 * three axes (so the coarsened grid is exactly commensurate with the
 * fine grid).
 *
 * Returns 1  if no coarsening is needed  (h_min > 4*delta already).
 * Returns MFTL_ERROR  (-1) if no valid cbrt_G exists within the range
 * [required_cbrt_G, min(cuts[k])].  This signals the caller to either
 * fall back to serial remeshing or abort with a configuration error.
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLComputeCoarseningFactor(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(params != MFTL_NULL);

  double h_min = MFTLDomainGetHMin(&ens->domain);
  double delta = 3.0 * params->max_edge_length;

  /* No freeze band (degenerate input) or constraint already satisfied. */
  if (delta <= 0.0 || h_min > 4.0 * delta) return MFTL_SUCCESS;

  /* ratio = 4δ/h_min > 1.  We need cbrt_G > ratio (strict inequality). */
  double ratio = (4.0 * delta) / h_min;
  int cbrt_G = (int) floor(ratio) + 1; /* guarantees cbrt_G > ratio */

  /* Minimum cuts[k] sets the upper limit on cbrt_G. */
  int min_cuts = MFTL_MIN(ens->domain.cuts[0], MFTL_MIN(ens->domain.cuts[1], ens->domain.cuts[2]));

  /* Search upward for the smallest cbrt_G that divides all cuts[k].    *
   * For well-configured decompositions (cuts[k] = power-of-2 or cube)  *
   * this loop typically executes at most 2-3 iterations.               */
  int valid = 0;

  while (!valid)
  {
    if (cbrt_G > min_cuts)
    {
      return MFTL_ERROR;
    }

    valid = 1;
    for (int k = 0; k < 3; ++k)
    {
      if (ens->domain.cuts[k] % cbrt_G != 0)
      {
        valid = 0;
        break;
      }
    }

    if (!valid)
    {
      ++cbrt_G;
    }
  }

  return cbrt_G * cbrt_G * cbrt_G;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRemeshWithCoarsening
 *
 * Gather and scatter are both implemented by temporarily reconfiguring
 * ens->domain and calling the existing MFTLRedistributeListOfObjects:
 *
 *   GATHER  (all → group master):
 *     Temporarily set cuts={1,1,1}, rank_map={all cells → 0},
 *     comm_cart=group_comm.  MFTLRedistributeListOfObjects sends every
 *     object to rank 0 of group_comm (the master for this group).
 *
 *   SCATTER (master → fine-grid owners within group):
 *     Restore fine cuts, set rank_map[c] = group_pos(c) computed from
 *     the fine cell's local coordinates within the cbrt_G×cbrt_G×cbrt_G
 *     sub-cube, comm_cart=group_comm.  MFTLRedistributeListOfObjects
 *     maps each triangle to its proper fine-grid owner within the group.
 *
 * Non-master ranks participate in both redistributions (with empty
 * ensembles for gather; receiving only for scatter). No deadlock because
 * group_comm provides the matching collective context.
 *
 * The two MPI_Comm_split calls (group_comm, remesh_comm) are collective
 * on flow_comm, so all flow ranks call them in the same order.
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRemeshWithCoarsening(MFTL_Ensemble *ens, MFTL_RemeshParameters *params, int G)
{
  MFTL_ASSERT(ens != NULL);
  MFTL_ASSERT(params != NULL);
  MFTL_ASSERT(G > 1);

  /* Verify G is a perfect cube. */
  int cbrt_G = (int) round(cbrt((double) G));
  MFTL_ASSERT(cbrt_G * cbrt_G * cbrt_G == G && "MFTLRemeshWithCoarsening: G must be a perfect cube");

  MPI_Comm flow_comm = ens->domain.comm_cart;
  int flow_rank, world_size;
  MPI_Comm_rank(flow_comm, &flow_rank);
  MPI_Comm_size(flow_comm, &world_size);

  MFTL_ASSERT(world_size % G == 0 && "MFTLRemeshWithCoarsening: world_size must be divisible by G");

  const int nx = ens->domain.cuts[0];
  const int ny = ens->domain.cuts[1];

  /* ── Cartesian group assignment ──────────────────────────────────
   * Group ranks into cbrt_G × cbrt_G × cbrt_G Cartesian cubes.      */
  int ix = flow_rank % nx;
  int iy = (flow_rank / nx) % ny;
  int iz = flow_rank / (nx * ny);

  int coarse_nx = nx / cbrt_G;
  int coarse_ny = ny / cbrt_G;

  int gcx = ix / cbrt_G;
  int gcy = iy / cbrt_G;
  int gcz = iz / cbrt_G;

  int group_id = gcx + gcy * coarse_nx + gcz * coarse_nx * coarse_ny;

  int lx = ix % cbrt_G;
  int ly = iy % cbrt_G;
  int lz = iz % cbrt_G;

  int group_pos = lx + ly * cbrt_G + lz * cbrt_G * cbrt_G;

  /* ── Step 1: form group communicator (all ranks in one cube) ─────  */
  MPI_Comm group_comm;
  MPI_Comm_split(flow_comm, group_id, group_pos, &group_comm);

  /* Save the original fine domain before any modifications. */
  MFTL_Domain fine_domain = ens->domain;

  /* ── Step 2: GATHER – migrate all objects to group master ────────
   * Single-partition domain: every position maps to rank 0.
   * MFTLRedistributeListOfObjects runs on group_comm (G ranks).       */
  {
    int *gather_map = (int *) calloc(1, sizeof(int)); /* {0} */
    MFTL_ASSERT(gather_map != NULL);

    ens->domain.cuts[0] = 1;
    ens->domain.cuts[1] = 1;
    ens->domain.cuts[2] = 1;
    ens->domain.grid_factors[0] = 1.0 / fine_domain.domain_size[0];
    ens->domain.grid_factors[1] = 1.0 / fine_domain.domain_size[1];
    ens->domain.grid_factors[2] = 1.0 / fine_domain.domain_size[2];
    ens->domain.rank_map = gather_map;
    ens->domain.comm_cart = group_comm;

    MFTLRedistributeListOfObjects(ens);
    /* group master (group_pos==0): holds all group objects.
     * other group members: num_objects == 0.                        */
    free(gather_map);
  }

  /* ── Step 3: form remesh communicator (masters only) ─────────────  */
  MPI_Comm remesh_comm = MPI_COMM_NULL;
  MPI_Comm_split(flow_comm, (group_pos == 0) ? 0 : MPI_UNDEFINED, group_id, &remesh_comm);

  int ret = MFTL_SUCCESS;

  /* ── Step 4: masters – configure coarse domain and remesh ────────  */
  if (remesh_comm != MPI_COMM_NULL)
  {
    /* Build coarse rank_map: coarse cell c → rank c in remesh_comm.  */
    int coarse_nz = fine_domain.cuts[2] / cbrt_G;
    int n_coarse = coarse_nx * coarse_ny * coarse_nz;
    int *coarse_map = (int *) malloc((size_t) n_coarse * sizeof(int));
    assert(coarse_map != NULL);
    for (int c = 0; c < n_coarse; ++c) coarse_map[c] = c;

    ens->domain.cuts[0] = coarse_nx;
    ens->domain.cuts[1] = coarse_ny;
    ens->domain.cuts[2] = coarse_nz;
    ens->domain.grid_factors[0] = (double) coarse_nx / fine_domain.domain_size[0];
    ens->domain.grid_factors[1] = (double) coarse_ny / fine_domain.domain_size[1];
    ens->domain.grid_factors[2] = (double) coarse_nz / fine_domain.domain_size[2];
    ens->domain.rank_map = coarse_map;
    ens->domain.comm_cart = remesh_comm;
    ens->domain.decomp_shift[0] = 0.0;
    ens->domain.decomp_shift[1] = 0.0;
    ens->domain.decomp_shift[2] = 0.0;

    /* Distribute objects to their coarse-grid owners before remeshing. */
    MFTLRedistributeListOfObjects(ens);

    /* Full shift-scheme remeshing on the coarsened decomposition.    *
         * h_min_coarse = h_min_fine * cbrt_G > 4*delta by construction. */
    ret = MFTLRemeshEnsembleShiftScheme(ens, params);

    free(coarse_map);
    MPI_Comm_free(&remesh_comm);
  }

  /* ── Step 5: SCATTER – distribute remeshed objects to fine-grid
   *            owners within the group.
   *
   * scatter_rank_map maps each fine cell to the group_pos of its owner
   * (rank within group_comm).  For a fine cell at Cartesian coords
   * (ix, iy, iz), group_pos = (ix%cbrt_G) + (iy%cbrt_G)*cbrt_G + (iz%cbrt_G)*cbrt_G^2.
   *
   * All group members (master and non-masters) execute this step,
   * making MFTLRedistributeListOfObjects a proper collective.          */
  {
    int n_fine = fine_domain.cuts[0] * fine_domain.cuts[1] * fine_domain.cuts[2];
    int *scatter_map = (int *) malloc((size_t) n_fine * sizeof(int));
    assert(scatter_map != NULL);

    const int fnx = fine_domain.cuts[0];
    const int fny = fine_domain.cuts[1];
    for (int c = 0; c < n_fine; ++c)
    {
      int fix = c % fnx;
      int fiy = (c / fnx) % fny;
      int fiz = c / (fnx * fny);

      scatter_map[c] = (fix % cbrt_G) + (fiy % cbrt_G) * cbrt_G + (fiz % cbrt_G) * (cbrt_G * cbrt_G);
    }

    /* Restore fine cuts and grid_factors; only rank_map and comm change. */
    ens->domain = fine_domain; /* restores fine geometry  */
    ens->domain.rank_map = scatter_map; /* override to group_pos   */
    ens->domain.comm_cart = group_comm;
    ens->domain.decomp_shift[0] = 0.0;
    ens->domain.decomp_shift[1] = 0.0;
    ens->domain.decomp_shift[2] = 0.0;

    MFTLRedistributeListOfObjects(ens);
    /* Each group member now holds exactly the fine-grid triangles
     * belonging to its Cartesian sub-cell.                          */
    free(scatter_map);
  }

  /* ── Step 6: broadcast return value from master, free group_comm ──  */
  MPI_Bcast(&ret, 1, MPI_INT, 0, group_comm);
  MPI_Comm_free(&group_comm);

  /* ── Step 7: restore canonical fine domain, rebuild ghost layer ───  */
  ens->domain = fine_domain; /* original rank_map, flow_comm  */
  MFTLEnsembleCreateSpatialGhosts(ens, params->max_edge_length);

  return ret;
}

/* ══════════════════════════════════════════════════════════════════════
 * MFTLRemeshEnsemble
 *
 * Top-level dispatch for a single ensemble.  Decision tree:
 *
 *   1. G = MFTLComputeCoarseningFactor():
 *      G == MFTL_ERROR → fatal configuration error, abort.
 *      G >  1          → coarsen and remesh on subcommunicator.
 *      G == 1          → proceed to algorithm selection.
 *
 *   2. MFTLSelectRemeshAlgorithm() [collective, one Allreduce]:
 *      CENTERING    → object-centering migration + ungated serial kernel.
 *      SHIFT_SCHEME → spatial shift scheme with phase schedule.
 *
 * When SHIFT_SCHEME is selected after G==1, the constraint
 *   h_min > 4 * delta
 * is already verified by step 1 (would have triggered coarsening
 * otherwise), so MFTLRemeshEnsembleShiftScheme's precondition holds.
 *
 * Collective: all ranks in ens->domain.comm_cart must call this.
 * ══════════════════════════════════════════════════════════════════════ */
int MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(params != MFTL_NULL);

  /* HERE A FUNCTION SHOULD BE CALLED WHICH CHECKS IF REMESHING IS NEEDED AT ALL. IM UNSURE ABOUT A FAST IMPLEMENTATION HERE.
  DOES THIS REALLY NEED TO LOOP OVER ALL OBJECTS UNTIL IT FINDS A EDGE THAT NEEDS REMESHING OR IS THERE SOMETHING FASTER? */

  /* HERE A FUNCTION SHOULD BE CALLED THAT CHECKS IF ITS OPTIMAL TO MOVE OBJECTS TO DIFFERENT KIND OF ENSEMBLES */

  /* ── Step 1: check if coarsening is required ────────────────────── */
  int G = MFTLComputeCoarseningFactor(ens, params);

  if (G == MFTL_ERROR)
  {
    /* No commensurate coarsening factor found.  This is a fatal   *
     * configuration error: either cuts[k] must be chosen to be    *
     * divisible by the required cbrt_G, or max_edge_length /      *
     * k_ghost must be reduced.  We surface it as a return code    *
     * so the caller can decide between abort and fall-back.       */
    /* HERE SHOULD BE A FALLBACK IMPLEMENTED. FOR INSTANCE A COARSER 1D DECOMPOSITION INSTEAD OF THE CUBE ONE. */
    return MFTL_ERROR;
  }

  if (G > 1)
  {
    /* Coarsen: group G flow ranks → 1 remesh rank, remesh on the
     * subcommunicator, scatter back.                               */
    MFTLRemeshWithCoarsening(ens, params, G);
  }

  /* ── Step 2: select algorithm (collective; one MPI_Allreduce) ────── */
  MFTL_RemeshAlgorithm algo = MFTLSelectRemeshAlgorithm(ens, params);

  /* ── Step 3: dispatch ───────────────────────────────────────────── */
  switch (algo)
  {
    case MFTL_ALGO_CENTERING:
      /* Objects are small (R < 0.5 * h_min): gather each object  *
       * to its centroid rank, run ungated serial kernel, done in  *
       * one pass.  MFTLRemeshEnsembleDispersedCentering issues    *
       * its own MPI_Allreduce for convergence.                    */
      return MFTLRemeshEnsembleDispersedCentering(ens, params);

    case MFTL_ALGO_SHIFT_SCHEME:
      /* Universal path: shift scan planes through {0,h/2}^3,     *
       * freeze band, gated serial kernel, convergence certificate.*
       * Precondition h_min > 4*delta is guaranteed by G==1 above. */
      return MFTLRemeshEnsembleShiftScheme(ens, params);

    default: assert(0 && "MFTLRemeshEnsemble: unknown algorithm"); return MFTL_ERROR;
  }
}
