/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Multi-rank tests for mesh generation (launched at 2 ranks).
 *
 * Generation is rank-0-only by contract, which makes it the one place where
 * ranks legitimately hold DIFFERENT state and the collective bookkeeping has to
 * bridge the gap. Everything below is about that bridge.
 *
 * Every assertion is made on an MPI_Allreduce'd verdict; see the note in
 * test_ensemble_parallel.c.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;
static int g_rank = 0;
static int g_nproc = 1;

static MFTL_Object *make_sphere(int refinements)
{
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, obj);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 1.0, refinements);
  return obj;
}

static int test_only_rank_zero_holds_the_mesh(void)
{
  MFTL_Object *obj = make_sphere(2);

  int ok;
  if (g_rank == 0)
  {
    ok = (obj->verts.n == 162) && (obj->tris.n == 320) && (MFTLNumHalfEdges(obj) == 960);
    ok = ok && inv_mesh_is_sound(obj, g_rank);
    ok = ok && inv_mesh_is_closed(obj);
  }
  else
  {
    /* Empty, but STRUCTURALLY VALID: the pools are consistent and the (vacuous)
     * derived arrays are marked current, so a collective caller that immediately
     * runs a hot loop does not trip the staleness assertion on a non-root rank. */
    ok = (obj->verts.n == 0) && (obj->tris.n == 0);
    ok = ok && MFTLObjectPoolsAreConsistent(obj);
    ok = ok && MFTLObjectHasTopology(obj) && MFTLObjectHasFrames(obj);
  }

  TEST_ASSERT(agreed(ok, g_ens->domain.comm_world));
  return 0;
}

static int test_id_counters_are_broadcast_to_every_rank(void)
{
  MFTL_Object *obj = make_sphere(2);

  /* Every rank must continue the SAME numbering scheme, or a rank that later
   * creates an entity collides with an ID rank 0 already used -- and global IDs
   * are what identity across ranks is built on now that there are no hash maps. */
  int ok = (obj->next_vertex_gid == 162) && (obj->next_triangle_gid == 320) && (obj->next_halfedge_gid == 960);

  // ...and they must be identical everywhere, not merely correct on rank 0.
  int mine[3] = {obj->next_vertex_gid, obj->next_triangle_gid, obj->next_halfedge_gid};
  int root[3] = {mine[0], mine[1], mine[2]};
  MPI_Bcast(root, 3, MPI_INT, 0, g_ens->domain.comm_world);
  for (int k = 0; k < 3; ++k)
  {
    if (mine[k] != root[k]) ok = 0;
  }

  TEST_ASSERT(agreed(ok, g_ens->domain.comm_world));
  return 0;
}

static int test_global_stats_see_the_whole_mesh_from_every_rank(void)
{
  MFTL_Object *obj = make_sphere(3);

  /* The counts are global, so a rank holding nothing still knows how big the
   * object is. Generation runs MFTLObjectUpdateGlobalStats itself, so this must
   * already hold on return. */
  int ok = (obj->global_n_vertices == 642) && (obj->global_n_triangles == 1280);

  // Calling it again must not double-count.
  MFTLObjectUpdateGlobalStats(obj);
  ok = ok && (obj->global_n_vertices == 642) && (obj->global_n_triangles == 1280);

  TEST_ASSERT(agreed(ok, g_ens->domain.comm_world));
  return 0;
}

static int test_ensemble_update_tolerates_empty_ranks(void)
{
  MFTL_Object *obj = make_sphere(3);

  /* The state immediately after generation: rank 0 has everything, everyone else
   * has nothing. This is the first collective a solver runs, and it is where an
   * unguarded MIN identity would make the global minimum DBL_MAX. */
  MFTLEnsembleUpdate(g_ens);

  const double exact = 4.0 * MFTL_PI;  // unit sphere
  int ok = (obj->quality.min_triangle_area > 0.0);
  ok = ok && (obj->quality.max_triangle_area < 1.0);
  ok = ok && (obj->quality.avg_triangle_area * 1280.0 < exact);
  ok = ok && (obj->quality.avg_triangle_area * 1280.0 > 0.99 * exact);
  ok = ok && (obj->quality.min_aspect_ratio > 0.5);  // icosphere triangles are near-equilateral
  ok = ok && (obj->quality.max_aspect_ratio <= 1.0 + 1e-12);

  TEST_ASSERT(agreed(ok, g_ens->domain.comm_world));
  return 0;
}

static int test_reorder_is_a_local_operation(void)
{
  MFTL_Object *obj = make_sphere(2);

  /* MFTLObjectReorderMorton is documented as local: each rank orders its own
   * partition and there is no collective inside it. Calling it on ONE rank only
   * must therefore be safe -- if it contained a hidden collective this test
   * would hang, which is the failure mode worth catching early. */
  if (g_rank == 0) MFTLObjectReorderMorton(obj);

  MPI_Barrier(g_ens->domain.comm_world);

  int ok = 1;
  if (g_rank == 0)
  {
    ok = inv_mesh_is_sound(obj, g_rank) && inv_mesh_is_closed(obj) && inv_global_ids_are_dense(obj);
  }

  // The global counts are unchanged by a renumbering.
  MFTLObjectUpdateGlobalStats(obj);
  ok = ok && (obj->global_n_vertices == 162) && (obj->global_n_triangles == 320);

  TEST_ASSERT(agreed(ok, g_ens->domain.comm_world));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  int cuts[3] = {g_nproc, 1, 1};
  double lo[3] = {-4.0, -4.0, -4.0};
  double hi[3] = {4.0, 4.0, 4.0};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 16);

  if (g_ens == MFTL_NULL)
  {
    if (g_rank == 0) fprintf(stderr, "test_meshgen_parallel: failed to create the ensemble.\n");
    MPI_Finalize();
    return 1;
  }

  if (g_rank == 0) printf("--- Running MFTL Mesh Generation Tests (%d ranks) ---\n", g_nproc);

  RUN_TEST(test_only_rank_zero_holds_the_mesh);
  RUN_TEST(test_id_counters_are_broadcast_to_every_rank);
  RUN_TEST(test_global_stats_see_the_whole_mesh_from_every_rank);
  RUN_TEST(test_ensemble_update_tolerates_empty_ranks);
  RUN_TEST(test_reorder_is_a_local_operation);

  if (g_rank == 0) TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
