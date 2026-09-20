/*
 * Stage 04 — object generators (cylinder, box, cube, face).
 *
 * Covers (per plans/04-generators.md):
 *   - box: closed (Euler chi = 2), integrity pass, exact analytic volume/area
 *   - cube: refine 0/2, volume and area exact under flat refinement
 *   - cylinder: capped (chi = 2), POSITIVE outward volume, convergence in dr,
 *     ellipticity convention, centroid at (0,0,L/2)
 *   - face: open patch counts and area; integrity of a non-closed surface
 *   - rank-independence: identical counts and volume on 1/2/4/8 ranks
 *
 * MFTL1 note: MFTL1's own cylinder is an OPEN tube (no caps, chi = 0), so it
 * has no enclosed volume; the capped closed form here follows the MFTL2 plan's
 * acceptance criteria. The ellipticity convention (x scaled by sqrt(1-e^2)
 * for e >= 0) is reproduced from MFTL1's generateCylinder.c.
 *
 * MPI safety: generator meshes exist only on rank 0 until redistribution, so
 * per-mesh checks (counts, Euler, integrity) are rank-0-only and every check
 * accumulates into a flag. All collectives (integrals, redistribution,
 * destroy) complete BEFORE any TEST_ASSERT can early-return — an early
 * return on one rank only would straddle a collective and deadlock.
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include <mpi.h>
#include <math.h>
#include <string.h>

static int g_rank = -1;
static int g_nproc = -1;

static MFTL_Ensemble* make_ensemble(double bmin[3], double bmax[3], bool per[3], MFTL_Object** out_obj)
{
  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
  if (!ens) return NULL;
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  *out_obj = obj;
  return ens;
}

// Euler characteristic from the half-edge wiring: unique edges = boundary
// half-edges + one per twin pair, so chi = V - E + F.
static double euler_chi(MFTL_Object* obj)
{
  int n_boundary_he = 0;
  int n_twinned_he = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    if (obj->half_edges[i].twin_idx == MFTL_ERROR)
      n_boundary_he++;
    else
      n_twinned_he++;
  }
  double E = (double) n_boundary_he + (double) n_twinned_he / 2.0;
  return (double) obj->n_vertices - E + (double) obj->n_triangles;
}

#define CHECK(cond)                                              \
  do                                                             \
  {                                                              \
    if (!(cond))                                                 \
    {                                                            \
      printf("Rank %d: CHECK failed: (%s) at %s:%d\n", g_rank, #cond, __FILE__, __LINE__); \
      fail = 1;                                                  \
    }                                                            \
  } while (0)

int test_box()
{
  double bmin[3] = {-4, -4, -4}, bmax[3] = {4, 4, 4};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  double lo[3] = {-1.0, -2.0, -3.0};
  double hi[3] = {1.0, 2.0, 3.0};
  MFTLGenerateBox(obj, lo, hi);
  MFTLComputeTriangleProperties(obj);

  // Collective integrals on the rank-0-local closed mesh.
  double cm[3] = {0, 0, 0};
  double A = MFTLSurfaceCenter(obj, cm);
  double V = MFTLObjectVolume(obj);

  // Per-mesh checks on the rank-0-local (full) mesh, BEFORE redistribution.
  int fail = 0;
  if (g_rank == 0)
  {
    CHECK(obj->n_vertices == 8);
    CHECK(obj->n_triangles == 12);
    CHECK(obj->n_half_edges == 36);
    CHECK(fabs(euler_chi(obj) - 2.0) < 1e-12);
    CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  }

  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  double A_post = MFTLSurfaceCenter(obj, cm);
  double V_post = MFTLObjectVolume(obj);

  // Exact analytic volume and area (planar faces -> no discretization error).
  CHECK(V > 0.0);  // outward winding
  CHECK(fabs(V - 48.0) < 1e-12);
  CHECK(fabs(A - 88.0) < 1e-12);
  CHECK(fabs(cm[0]) < 1e-12 && fabs(cm[1]) < 1e-12 && fabs(cm[2]) < 1e-12);
  CHECK(fabs(V_post - V) < 1e-12 && fabs(A_post - A) < 1e-12);
  if (g_rank == 0) CHECK(ens->total_global_triangles == 12);

  MFTLEnsembleDestroy(&ens);  // collective: all ranks must arrive here
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_cube_refinement()
{
  double bmin[3] = {-4, -4, -4}, bmax[3] = {4, 4, 4};
  bool per[3] = {false, false, false};

  for (int pass = 0; pass < 2; ++pass)
  {
    int refine = pass == 0 ? 0 : 2;
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    MFTLGenerateCube(obj, refine);
    MFTLComputeTriangleProperties(obj);

    // Expected counts for this refinement level (pure arithmetic). gids use
    // the CONTINUING-counter convention of MFTL1's subdivision (sphere.c and
    // the flat tessellate both assign next_*_gid++), so the counters are not
    // re-based to zero: per level, F quadruples and the triangle/he counters
    // advance by the number of newly created entities.
    int expect_F = 12;
    int expect_V = 8;
    int expect_nextF = 12;    // box leaves next_triangle_gid == 12
    int expect_nextHE = 36;   // ... and next_halfedge_gid == 36
    for (int r = 0; r < refine; ++r)
    {
      int E = 3 * expect_F / 2;
      expect_V += E;
      int newF = 4 * expect_F;
      expect_nextF += newF;
      expect_nextHE += 3 * newF;
      expect_F = newF;
    }

    double cm[3] = {0, 0, 0};
    double A = MFTLSurfaceCenter(obj, cm);
    double V = MFTLObjectVolume(obj);

    // gid counters must be refinement-consistent on EVERY rank: refinement
    // advances them on rank 0 and the generator re-broadcasts them.
    int fail = 0;
    CHECK(obj->next_vertex_gid == expect_V);
    CHECK(obj->next_triangle_gid == expect_nextF);
    CHECK(obj->next_halfedge_gid == expect_nextHE);

    if (g_rank == 0)
    {
      printf("  refine %d: V=%d F=%d (expected %d/%d)\n", refine, obj->n_vertices, obj->n_triangles, expect_V, expect_F);
      CHECK(obj->n_vertices == expect_V);
      CHECK(obj->n_triangles == expect_F);
      CHECK(fabs(euler_chi(obj) - 2.0) < 1e-12);
      CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);

      // gids must be unique across the whole generated mesh
      for (int a = 0; a < obj->n_triangles; ++a)
        for (int b = a + 1; b < obj->n_triangles; ++b)
          if (obj->triangles[a].global_id == obj->triangles[b].global_id)
          {
            printf("Rank 0: duplicate triangle gid %d\n", obj->triangles[a].global_id);
            fail = 1;
          }
    }
    // Flat refinement keeps the geometry exactly: volume and area unchanged.
    CHECK(V > 0.0);
    CHECK(fabs(V - 8.0) < 1e-12);
    CHECK(fabs(A - 24.0) < 1e-12);
    CHECK(fabs(cm[0]) + fabs(cm[1]) + fabs(cm[2]) < 1e-12);

    // Counter consistency must survive redistribution (global totals correct).
    MFTLRedistributeListOfObjects(ens);
    MFTLEnsembleUpdateGlobalCounts(ens);
    double V_post = MFTLObjectVolume(obj);
    CHECK(fabs(V_post - 8.0) < 1e-12);
    if (g_rank == 0) CHECK(ens->total_global_triangles == expect_F);

    MFTLEnsembleDestroy(&ens);
    TEST_ASSERT(fail == 0);
  }
  return 0;
}

int test_cylinder()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};

  // Convergence study: coarse vs fine dr (circular cross-section), then the
  // testCylinder-style ellipticity case (e = 0.5 squeezes x by sqrt(1-e^2)).
  double V_coarse = 0.0, V_fine = 0.0, ellip_V = 0.0;
  const double ellip_expected = MFTL_PI * sqrt(1.0 - 0.5 * 0.5);  // pi r (r sqrt(1-e^2)) L

  for (int pass = 0; pass < 3; ++pass)
  {
    double dr = pass == 0 ? 0.02 : 0.002;
    double ellipticity = pass == 2 ? 0.5 : 0.0;

    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    MFTLGenerateCylinder(obj, dr, 1.0, 1.0, ellipticity);
    MFTLComputeTriangleProperties(obj);

    double cm[3] = {0, 0, 0};
    double A = MFTLSurfaceCenter(obj, cm);
    double V = MFTLObjectVolume(obj);

    int fail = 0;
    if (g_rank == 0)
    {
      CHECK(fabs(euler_chi(obj) - 2.0) < 1e-12);
      CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
      printf("  pass %d: V=%.17g A=%.17g cm=(%.12g,%.12g,%.12g)\n", pass, V, A, cm[0], cm[1], cm[2]);
    }
    CHECK(V > 0.0);  // outward winding
    CHECK(fabs(cm[0]) < 1e-12 && fabs(cm[1]) < 1e-12 && fabs(cm[2] - 0.5) < 1e-12);

    MFTLEnsembleDestroy(&ens);
    TEST_ASSERT(fail == 0);

    if (pass == 0) V_coarse = V;
    if (pass == 1) V_fine = V;
    if (pass == 2) ellip_V = V;
  }

  // Cross-pass comparisons (no collectives remain — rank 0 only).
  if (g_rank == 0)
  {
    printf("  |V_fine - pi| = %.3e (coarse %.3e);  e=0.5: V=%.17g (expected %.17g)\n", fabs(V_fine - MFTL_PI), fabs(V_coarse - MFTL_PI), ellip_V,
           ellip_expected);
  }
  int fail = 0;
  CHECK(fabs(V_fine - MFTL_PI) < 1e-2);                          // fine dr close to pi r^2 L
  CHECK(fabs(V_fine - MFTL_PI) < fabs(V_coarse - MFTL_PI));      // converges
  CHECK(fabs(ellip_V - ellip_expected) / ellip_expected < 1e-2);  // ellipticity convention
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_face()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  double v[4][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}};
  MFTLGenerateFace(obj, v);
  MFTLComputeTriangleProperties(obj);

  // Area of the unit quad via the surface-center reduction (collective).
  double cm[3] = {0, 0, 0};
  double A = MFTLSurfaceCenter(obj, cm);

  int fail = 0;
  if (g_rank == 0)
  {
    CHECK(obj->n_vertices == 4);
    CHECK(obj->n_triangles == 2);
    CHECK(obj->n_half_edges == 6);
    CHECK(fabs(euler_chi(obj) - 1.0) < 1e-12);  // open patch: V - E + F = 4 - 5 + 2 = 1
    // Open surface: boundary edges tolerated by the integrity check.
    CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  }
  CHECK(fabs(A - 1.0) < 1e-12);
  CHECK(fabs(cm[0] - 0.5) < 1e-12 && fabs(cm[1] - 0.5) < 1e-12);

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_rank_independence()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateBox(obj, (double[]) {-1, -1, -1}, (double[]) {1, 1, 1});
  MFTLComputeTriangleProperties(obj);

  int F_pre = obj->n_triangles;
  double cm[3] = {0, 0, 0};
  double A_pre = MFTLSurfaceCenter(obj, cm);
  double Vol_pre = MFTLObjectVolume(obj);

  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  double A_post = MFTLSurfaceCenter(obj, cm);
  double Vol_post = MFTLObjectVolume(obj);

  if (g_rank == 0) printf("  nproc=%d  pre: F=%d Vol=%.17g  post: Vol=%.17g tris=%d\n", g_nproc, F_pre, Vol_pre, Vol_post, ens->total_global_triangles);

  // All integral values below are collective results, identical on every rank,
  // and all collectives have completed — a simultaneous early return is safe.
  int fail = 0;
  CHECK(fabs(Vol_post - Vol_pre) <= 1e-12);
  CHECK(fabs(A_post - A_pre) <= 1e-12);
  if (g_rank == 0) CHECK(ens->total_global_triangles == 12);

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_box);
  RUN_TEST(test_cube_refinement);
  RUN_TEST(test_cylinder);
  RUN_TEST(test_face);
  RUN_TEST(test_rank_independence);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
