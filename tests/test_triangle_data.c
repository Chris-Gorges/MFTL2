/*
 * Stage 12 — opaque per-triangle user data transport.
 *
 * Covers (per plans/12-opaque-triangle-data-transport.md):
 *   - byte-exact transport of a flat POD payload across
 *     MFTLRedistributeListOfObjects (owned AND ghost copies, keyed by gid)
 *   - scalar-mass conservation (sum scalar*area) across redistribution
 *   - ghost payload refresh via MFTLScatterTriangleData
 *   - payload transport through MFTLExtendGhostLayer
 *   - data_size == 0 no-op regression path
 *   - MFTLInitializeObjectData idempotence / re-init rejection
 *   - collective data_size mismatch refusal at every transport entry (negative test)
 *
 * Registered at np 1/2/4/8 (cuts {nproc,1,1}, domain [-2,2]^3, sphere r=1 at
 * the origin, refinement 3).
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include <mpi.h>
#include <math.h>
#include <string.h>

static int g_rank = -1;
static int g_nproc = -1;

// Flat POD payload (no embedded pointers — the Stage-12 contract).
typedef struct
{
  int gid;
  double scalar;
  double vec[3];
  unsigned char tag[16];
} TriData;

// Deterministic fill from the triangle global id. The memset pins the struct
// padding bytes so memcmp comparisons are byte-exact across ranks.
static void fill_tridata(int gid, TriData* d)
{
  memset(d, 0, sizeof(TriData));
  d->gid = gid;
  d->scalar = 0.25 + 1e-3 * gid;
  for (int k = 0; k < 3; ++k) d->vec[k] = cos(0.01 * gid * (k + 1));
  for (int j = 0; j < 16; ++j) d->tag[j] = (unsigned char) (gid * 31 + j);
}

// Build the sphere ensemble (NOT yet redistributed, NOT yet data-initialized).
static MFTL_Ensemble* build_sphere_ensemble(MFTL_Object** out_obj)
{
  int cuts[3] = {g_nproc, 1, 1};
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};

  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
  if (!ens) return NULL;

  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  double origin[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLComputeTriangleProperties(obj);

  *out_obj = obj;
  return ens;
}

// Attach the payload (collective) and fill all OWNED triangles from their gid.
static void init_and_fill(MFTL_Object* obj)
{
  MFTLInitializeObjectData(obj, sizeof(TriData));
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->is_ghost) continue;
    TriData d;
    fill_tridata(t->global_id, &d);
    MFTLSetTriangleData(obj, t, &d);
  }
}

// Snapshot this rank's owned (gid, TriData) pairs and gather the GLOBAL map
// (a triangle generated on rank 0 may be owned by any rank afterwards).
static int gather_global_snapshot(MFTL_Object* obj, int** out_gids, TriData** out_data)
{
  int n_local = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
    if (!obj->triangles[i].is_ghost) n_local++;

  int* l_gids = malloc((n_local > 0 ? n_local : 1) * sizeof(int));
  TriData* l_data = malloc((n_local > 0 ? n_local : 1) * sizeof(TriData));
  int w = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->is_ghost) continue;
    l_gids[w] = t->global_id;
    memcpy(&l_data[w], t->data, sizeof(TriData));
    w++;
  }

  int* counts = malloc(g_nproc * sizeof(int));
  MPI_Allgather(&n_local, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);

  int* displs = malloc((g_nproc + 1) * sizeof(int));
  displs[0] = 0;
  for (int r = 0; r < g_nproc; ++r) displs[r + 1] = displs[r] + counts[r];
  int total = displs[g_nproc];

  int* gids = malloc((total > 0 ? total : 1) * sizeof(int));
  TriData* data = malloc((total > 0 ? total : 1) * sizeof(TriData));

  MPI_Allgatherv(l_gids, n_local, MPI_INT, gids, counts, displs, MPI_INT, MPI_COMM_WORLD);

  // TriData rides as raw bytes (it is a flat POD).
  int* bcounts = malloc(g_nproc * sizeof(int));
  int* bdispls = malloc((g_nproc + 1) * sizeof(int));
  bdispls[0] = 0;
  for (int r = 0; r < g_nproc; ++r)
  {
    bcounts[r] = counts[r] * (int) sizeof(TriData);
    bdispls[r + 1] = bdispls[r] + bcounts[r];
  }
  MPI_Allgatherv(l_data, n_local * (int) sizeof(TriData), MPI_BYTE, data, bcounts, bdispls, MPI_BYTE, MPI_COMM_WORLD);

  free(l_gids);
  free(l_data);
  free(counts);
  free(displs);
  free(bcounts);
  free(bdispls);

  *out_gids = gids;
  *out_data = data;
  return total;
}

static int snapshot_find(const int* gids, int n, int gid)
{
  for (int j = 0; j < n; ++j)
    if (gids[j] == gid) return j;
  return -1;
}

// -----------------------------------------------------------------------------

int test_data_survives_redistribution()
{
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  TEST_ASSERT(MFTLInitializeObjectData(obj, sizeof(TriData)) == MFTL_SUCCESS);
  // Idempotent for the same size; a different non-zero size is rejected.
  TEST_ASSERT(MFTLInitializeObjectData(obj, sizeof(TriData)) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLInitializeObjectData(obj, sizeof(TriData) + 8) == MFTL_ERROR);
  TEST_ASSERT(obj->data_size == sizeof(TriData));

  init_and_fill(obj);

  int* snap_gids = NULL;
  TriData* snap_data = NULL;
  int n_snap = gather_global_snapshot(obj, &snap_gids, &snap_data);
  TEST_ASSERT(n_snap == obj->global_n_triangles);

  TEST_ASSERT(MFTLRedistributeListOfObjects(ens) == MFTL_SUCCESS);
  MFTLEnsembleUpdateGlobalCounts(ens);

  int local_fail = 0, owned_found = 0, ghost_checked = 0, ghost_skipped = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    int s = snapshot_find(snap_gids, n_snap, t->global_id);

    if (!t->is_ghost)
    {
      // Every owned triangle MUST be in the snapshot and byte-identical.
      if (s < 0 || memcmp(t->data, &snap_data[s], sizeof(TriData)) != 0)
        local_fail++;
      else
        owned_found++;
    }
    else
    {
      // Ghosts: verify against the snapshot when the gid was local somewhere
      // pre-redistribution (always true here), else skip.
      if (s < 0)
      {
        ghost_skipped++;
        continue;
      }
      if (memcmp(t->data, &snap_data[s], sizeof(TriData)) != 0)
        local_fail++;
      else
        ghost_checked++;
    }
  }

  int g_fail = 0, g_owned_found = 0, g_ghost_checked = 0;
  MPI_Allreduce(&local_fail, &g_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&owned_found, &g_owned_found, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&ghost_checked, &g_ghost_checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0) printf("  fails=%d owned_found=%d/%d ghost_checked=%d ghost_skipped=%d\n", g_fail, g_owned_found, n_snap, g_ghost_checked, ghost_skipped);

  TEST_ASSERT_EQUAL_INT(0, g_fail);
  // Every originally-owned triangle is still owned by exactly one rank, data intact.
  TEST_ASSERT_EQUAL_INT(obj->global_n_triangles, g_owned_found);
  if (g_nproc > 1) TEST_ASSERT(g_ghost_checked > 0);  // validates the ghost copies redistribution itself ships

  free(snap_gids);
  free(snap_data);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_scalar_mass_conservation()
{
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  init_and_fill(obj);

  int* snap_gids = NULL;
  TriData* snap_data = NULL;
  int n_snap = gather_global_snapshot(obj, &snap_gids, &snap_data);

  double M0 = 0.0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->is_ghost) continue;
    const TriData* d = MFTLGetTriangleData(obj, t);
    M0 += d->scalar * t->area;
  }
  double M0g = 0.0;
  MPI_Allreduce(&M0, &M0g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // Single-object route: MFTLRedistribute dispatches to MFTLRedistributeObject
  // here (1 object), so this test also covers that path's data transport.
  TEST_ASSERT(MFTLRedistribute(ens) == MFTL_SUCCESS);

  double M1 = 0.0;
  int local_fail = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    int s = snapshot_find(snap_gids, n_snap, t->global_id);
    if (s < 0 || memcmp(t->data, &snap_data[s], sizeof(TriData)) != 0) local_fail++;
    if (t->is_ghost) continue;
    const TriData* d = MFTLGetTriangleData(obj, t);
    M1 += d->scalar * t->area;
  }
  double M1g = 0.0;
  MPI_Allreduce(&M1, &M1g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  int g_fail = 0;
  MPI_Allreduce(&local_fail, &g_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0) printf("  M0=%.17g M1=%.17g |dM|=%.3e byte_fails=%d\n", M0g, M1g, fabs(M0g - M1g), g_fail);

  // 1e-12 relative bounds MPI reduction-order drift; the strict byte-exactness
  // check is test 1. (The plan's 1e-14 refers to MFTL1's split/merge mass
  // error, not to reduction order.)
  TEST_ASSERT(fabs(M0g - M1g) <= 1e-12 * fmax(1.0, fabs(M0g)));
  TEST_ASSERT_EQUAL_INT(0, g_fail);

  free(snap_gids);
  free(snap_data);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_scatter_refreshes_ghosts()
{
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  init_and_fill(obj);
  TEST_ASSERT(MFTLRedistributeListOfObjects(ens) == MFTL_SUCCESS);
  MFTLEnsembleUpdateGlobalCounts(ens);

  // Mutate every owned blob; ghosts still hold the pre-bump bytes.
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->is_ghost) continue;
    TriData* d = MFTLGetTriangleData(obj, t);
    d->scalar += 1.0;
  }

  TEST_ASSERT(MFTLScatterTriangleData(ens, MFTL_NULL) == MFTL_SUCCESS);

  int local_fail = 0, checked = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (!t->is_ghost) continue;

    TriData expected;
    fill_tridata(t->global_id, &expected);
    expected.scalar += 1.0;

    if (memcmp(t->data, &expected, sizeof(TriData)) != 0)
      local_fail++;
    else
      checked++;
  }

  int g_fail = 0, g_checked = 0;
  MPI_Allreduce(&local_fail, &g_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&checked, &g_checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0) printf("  ghost payloads refreshed: %d (fails=%d)\n", g_checked, g_fail);

  TEST_ASSERT_EQUAL_INT(0, g_fail);
  if (g_nproc > 1) TEST_ASSERT(g_checked > 0);
  // np==1: no ghosts exist; MFTLScatterTriangleData must no-op cleanly.

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_ghost_layer_extension_carries_data()
{
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  init_and_fill(obj);
  TEST_ASSERT(MFTLRedistributeListOfObjects(ens) == MFTL_SUCCESS);
  MFTLEnsembleUpdateGlobalCounts(ens);

  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->is_ghost) continue;
    TriData* d = MFTLGetTriangleData(obj, t);
    d->scalar += 1.0;
  }
  TEST_ASSERT(MFTLScatterTriangleData(ens, MFTL_NULL) == MFTL_SUCCESS);

  TEST_ASSERT(MFTLExtendGhostLayer(ens, 1) == MFTL_SUCCESS);

  // ALL ghost triangles (old and newly pulled) must carry the bumped values.
  int local_fail = 0, checked = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (!t->is_ghost) continue;

    TriData expected;
    fill_tridata(t->global_id, &expected);
    expected.scalar += 1.0;

    if (memcmp(t->data, &expected, sizeof(TriData)) != 0)
      local_fail++;
    else
      checked++;
  }

  int g_fail = 0, g_checked = 0;
  MPI_Allreduce(&local_fail, &g_fail, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&checked, &g_checked, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0) printf("  ghost payloads after layer extension: %d (fails=%d)\n", g_checked, g_fail);

  TEST_ASSERT_EQUAL_INT(0, g_fail);
  if (g_nproc > 1) TEST_ASSERT(g_checked > 0);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_zero_data_size_noop()
{
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  // Do NOT call MFTLInitializeObjectData: data_size stays 0.
  TEST_ASSERT(obj->data_size == 0);

  TEST_ASSERT(MFTLRedistributeListOfObjects(ens) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLExtendGhostLayer(ens, 1) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLScatterTriangleData(ens, MFTL_NULL) == MFTL_SUCCESS);
  MFTLEnsembleUpdateGlobalCounts(ens);

  for (int i = 0; i < obj->n_triangles; ++i)
  {
    TEST_ASSERT(obj->triangles[i].data == MFTL_NULL);
    TEST_ASSERT(MFTLGetTriangleData(obj, &obj->triangles[i]) == MFTL_NULL);
  }

  int owned = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
    if (!obj->triangles[i].is_ghost) owned++;

  int total_owned = 0;
  MPI_Allreduce(&owned, &total_owned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT_EQUAL_INT(obj->global_n_triangles, total_owned);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// Deliberate contract violation: rank 0 initializes a different data_size than
// all other ranks. Every triangle-data transport entry must refuse loudly
// (MFTL_ERROR) and leave the mesh untouched. RUN LAST — nothing may depend on
// post-violation state.
int test_data_size_mismatch_detected()
{
  if (g_nproc < 2) return 0;  // a cross-rank mismatch needs at least 2 ranks

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(&obj);
  TEST_ASSERT(ens != NULL);

  if (g_rank == 0) printf("  (intentional contract violation — loud FATAL stderr lines expected below)\n");

  size_t bad_size = (g_rank == 0) ? sizeof(TriData) : sizeof(TriData) + 8;
  TEST_ASSERT(MFTLInitializeObjectData(obj, bad_size) == MFTL_SUCCESS);

  int owned_before = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
    if (!obj->triangles[i].is_ghost) owned_before++;

  int rc_single = MFTLRedistribute(ens);  // 1 object -> MFTLRedistributeObject
  int rc_list = MFTLRedistributeListOfObjects(ens);
  int rc_scatter = MFTLScatterTriangleData(ens, MFTL_NULL);
  int rc_extend = MFTLExtendGhostLayer(ens, 1);

  // Every path must fail on EVERY rank (Allreduce confirms uniformity).
  int local_bad = (rc_single != MFTL_ERROR) + (rc_list != MFTL_ERROR) + (rc_scatter != MFTL_ERROR) + (rc_extend != MFTL_ERROR);
  int g_bad = 0;
  MPI_Allreduce(&local_bad, &g_bad, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0)
    printf("  refused: single=%d list=%d scatter=%d extend=%d | non-error returns across all ranks: %d\n", rc_single, rc_list, rc_scatter,
           rc_extend, g_bad);

  TEST_ASSERT_EQUAL_INT(0, g_bad);

  // Mesh must be untouched by the refused transports.
  int owned_after = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
    if (!obj->triangles[i].is_ghost) owned_after++;
  TEST_ASSERT_EQUAL_INT(owned_before, owned_after);

  MFTLEnsembleDestroy(&ens);  // rank-local destroy must not crash/leak
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_data_survives_redistribution);
  RUN_TEST(test_scalar_mass_conservation);
  RUN_TEST(test_scatter_refreshes_ghosts);
  RUN_TEST(test_ghost_layer_extension_carries_data);
  RUN_TEST(test_zero_data_size_noop);
  RUN_TEST(test_data_size_mismatch_detected);  // LAST: deliberate contract violation

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
