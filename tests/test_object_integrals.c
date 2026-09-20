/*
 * Stage 02 — object integral quantities (volume / surface center / moment of inertia).
 *
 * Covers (per plans/02-object-integrals.md):
 *   - sphere r=1 refinements 4/5 vs analytic 4/3*pi and 4*pi, centroid at origin
 *   - inertia of the unit sphere vs the analytic tensor 2/5*M*R^2 (limiting case)
 *   - rank-independence: identical integrals before/after redistribution,
 *     registered at 1/2/4/8 ranks (reduction-order drift bounded by 1e-9)
 *   - periodic seam invariance: sphere straddling X/Y/Z/XY/XYZ faces
 *   - no double-counting: owned triangles counted == total_global_triangles
 *
 * The MFTL1 test1.gts regression is intentionally absent: GTS I/O is Stage 05.
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

// Build a sphere ensemble (cuts {nproc,1,1}, domain [bmin,bmax]), generate, redistribute.
static MFTL_Ensemble* build_sphere_ensemble(const double center[3], double radius, int ref, const double bmin[3], const double bmax[3],
                                            const bool per[3], MFTL_Object** out_obj)
{
  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, (double*) bmin, (double*) bmax, (bool*) per, 2);
  if (!ens) return NULL;

  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  MFTLGenerateSphere(obj, (double*) center, radius, ref);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  *out_obj = obj;
  return ens;
}

// Exact recount (same guard as the library functions) for the no-double-count check.
static int count_owned_triangles(MFTL_Ensemble* ens, int rank)
{
  int n = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object* obj = ens->objects[o];
    for (int t = 0; t < obj->n_triangles; ++t)
      if (!obj->triangles[t].is_ghost && obj->triangles[t].owner_proc == rank) n++;
  }
  return n;
}

// -----------------------------------------------------------------------------

int test_analytic_sphere()
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};
  double origin[3] = {0.0, 0.0, 0.0};

  const double V_true = 4.0 / 3.0 * MFTL_PI;  // 4.1887902047863905
  const double A_true = 4.0 * MFTL_PI;        // 12.566370614359172

  double V4 = 0.0, err4 = 0.0;

  for (int pass = 0; pass < 2; ++pass)
  {
    int ref = pass == 0 ? 4 : 5;
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = build_sphere_ensemble(origin, 1.0, ref, bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    double cm[3] = {0.0, 0.0, 0.0};
    double A = MFTLSurfaceCenter(obj, cm);
    double V = MFTLObjectVolume(obj);

    // The current sphere generator winds clockwise viewed from outside, so
    // the MFTL1 formula yields a NEGATIVE volume for these meshes. Pin the
    // sign (outward-oriented surfaces assert V > 0 — see generator/IO tests)
    // so a winding flip is caught rather than absorbed by fabs checks.
    TEST_ASSERT(V < 0.0);

    // The triangulated sphere has its vertices exactly on the sphere but flat
    // chords as facets, so volume/area approach the analytic values from below.
    double err = fabs(fabs(V) - V_true);
    double cm_err = fabs(cm[0]) + fabs(cm[1]) + fabs(cm[2]);

    if (g_rank == 0) printf("  ref %d: V=%.17g (|V-4/3 pi|=%.3e)  A=%.17g (|A-4pi|=%.3e)  |cm|=%.3e\n", ref, V, err, A, fabs(A - A_true), cm_err);

    // Discretization tolerances for the chord-based sphere (facets inside the
    // true surface; errors shrink ~4x per refinement level).
    TEST_ASSERT(fabs(A - A_true) < 3e-2);
    TEST_ASSERT(cm_err < 1e-10);

    if (ref == 4)
    {
      V4 = V;
      err4 = err;
      TEST_ASSERT(err < 2e-2);
    }
    else
    {
      // Tolerance decreases with refinement, and the approximation improves monotonically.
      TEST_ASSERT(err < 3e-3);
      TEST_ASSERT(err < err4);
      TEST_ASSERT(fabs(fabs(V) - V_true) < fabs(fabs(V4) - V_true));
      // (cm accuracy is round-off limited at both refinements; no monotonicity.)
    }

    MFTLEnsembleDestroy(&ens);
  }

  return 0;
}

int test_inertia_sphere()
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};
  double origin[3] = {0.0, 0.0, 0.0};

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere_ensemble(origin, 1.0, 5, bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  double cm[3] = {0.0, 0.0, 0.0};
  MFTLSurfaceCenter(obj, cm);
  double V = MFTLObjectVolume(obj);

  const double density = 2.5;
  double I[3][3];
  MFTLMomentOfInertia(obj, I, cm, density);

  // Cross-check against the analytic SOLID-sphere tensor about its center:
  //   I_diag_exact = 2/5 * M * R^2  (R = 1).
  // The MFTL1 LUMPED form (all mass at each tetrahedron centroid, apex at the
  // center) is NOT the exact tetrahedron-covariance integral: in the limit it
  // converges to (2/3)*(3/4)^2 = 3/8 * M * R^2 (tet-centroid radius 3/4 R,
  // tangential factor 2/3), i.e. 6.25% below 2/5. The refined sphere must
  // match the lumped limit tightly; the 2/5 comparison is a documented
  // ballpark cross-check, not an equality.
  double M = density * fabs(V);
  double I_diag = 2.0 / 5.0 * M;
  double I_lumped_limit = 3.0 / 8.0 * M;

  if (g_rank == 0)
  {
    printf("  I = [%.12g %.12g %.12g; %.12g %.12g %.12g; %.12g %.12g %.12g], 2/5 M = %.12g, lumped limit 3/8 M = %.12g\n", I[0][0], I[0][1], I[0][2],
           I[1][0], I[1][1], I[1][2], I[2][0], I[2][1], I[2][2], I_diag, I_lumped_limit);
  }

  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      if (i == j)
      {
        TEST_ASSERT(fabs(I[i][j] - I_lumped_limit) < 5e-3 * I_lumped_limit);  // lumped limit
        TEST_ASSERT(fabs(I[i][j] - I_diag) < 0.08 * I_diag);                  // exact-form ballpark cross-check
      }
      else
      {
        TEST_ASSERT(fabs(I[i][j]) < 1e-6 * I_diag);  // isotropy: off-diagonals ~ 0
      }
    }
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_rank_independence()
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {false, false, false};
  double origin[3] = {0.0, 0.0, 0.0};

  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
  TEST_ASSERT(ens != NULL);
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  // Reference pass on the (rank-0-local) closed serial mesh, before distribution.
  double cm_pre[3] = {0.0, 0.0, 0.0};
  double A_pre = MFTLSurfaceCenter(obj, cm_pre);
  double V_pre = MFTLObjectVolume(obj);
  double I_pre[3][3];
  MFTLMomentOfInertia(obj, I_pre, cm_pre, 1.0);

  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  double cm_post[3] = {0.0, 0.0, 0.0};
  double A_post = MFTLSurfaceCenter(obj, cm_post);
  double V_post = MFTLObjectVolume(obj);
  double I_post[3][3];
  MFTLMomentOfInertia(obj, I_post, cm_post, 1.0);

  if (g_rank == 0)
    printf("  nproc=%d  V_pre=%.17g V_post=%.17g  cm_pre=(%.17g,%.17g,%.17g) cm_post=(%.17g,%.17g,%.17g)\n", g_nproc, V_pre, V_post, cm_pre[0], cm_pre[1],
           cm_pre[2], cm_post[0], cm_post[1], cm_post[2]);

  const double tol = 1e-9;
  TEST_ASSERT(fabs(V_post - V_pre) <= tol * fmax(1.0, fabs(V_pre)));
  TEST_ASSERT(V_pre < 0.0 && V_post < 0.0);  // sign pinned: sphere winds inward (see analytic test)
  TEST_ASSERT(fabs(A_post - A_pre) <= tol * fmax(1.0, A_pre));
  for (int k = 0; k < 3; ++k) TEST_ASSERT(fabs(cm_post[k] - cm_pre[k]) <= tol);

  double I_scale = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) I_scale = fmax(I_scale, fabs(I_pre[i][j]));
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) TEST_ASSERT(fabs(I_post[i][j] - I_pre[i][j]) <= tol * fmax(1.0, I_scale));

  // No double-counting: the integrals see exactly total_global_triangles owned triangles.
  int owned = count_owned_triangles(ens, g_rank);
  int total_owned = 0;
  MPI_Allreduce(&owned, &total_owned, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT_EQUAL_INT(ens->total_global_triangles, total_owned);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_periodic_seam_invariance()
{
  const double L = 4.0;
  double bmin[3] = {-L / 2.0, -L / 2.0, -L / 2.0};
  double bmax[3] = {L / 2.0, L / 2.0, L / 2.0};
  bool per[3] = {true, true, true};

  // Sphere radius 0.9 (diameter 1.8 < L/2 = 2) straddling each periodic face.
  const char* names[6] = {"mid", "X", "Y", "Z", "XY", "XYZ"};
  double offsets[6][3] = {
      {0.0, 0.0, 0.0},   // mid-domain reference
      {1.6, 0.3, -0.2},  // X
      {-0.3, 1.6, 0.2},  // Y
      {0.2, -0.3, 1.6},  // Z
      {1.6, 1.5, 0.1},   // XY
      {1.6, 1.6, 1.6},   // XYZ
  };

  double V0 = 0.0, A0 = 0.0, cm0[3] = {0.0, 0.0, 0.0}, I0[3][3] = {{0.0}};

  for (int cse = 0; cse < 6; ++cse)
  {
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = build_sphere_ensemble(offsets[cse], 0.9, 4, bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    double cm[3] = {0.0, 0.0, 0.0};
    double A = MFTLSurfaceCenter(obj, cm);
    double V = MFTLObjectVolume(obj);
    double I[3][3];
    MFTLMomentOfInertia(obj, I, cm, 1.0);

    if (g_rank == 0) printf("  case %-3s V=%.17g A=%.17g cm=(%.12g,%.12g,%.12g)\n", names[cse], V, A, cm[0], cm[1], cm[2]);

    if (cse == 0)
    {
      V0 = V;
      A0 = A;
      memcpy(cm0, cm, 3 * sizeof(double));
      memcpy(I0, I, sizeof(I0));
      TEST_ASSERT(fabs(cm[0]) + fabs(cm[1]) + fabs(cm[2]) < 1e-10);
      TEST_ASSERT(V0 < 0.0);  // sphere winding: negative volume (see analytic test)
    }
    else
    {
      const double tol = 1e-9;
      TEST_ASSERT(fabs(V - V0) <= tol * fmax(1.0, fabs(V0)));
      TEST_ASSERT(fabs(A - A0) <= tol * fmax(1.0, A0));

      // Centroid equals the mid-domain centroid shifted by the offset, modulo the period.
      for (int k = 0; k < 3; ++k)
      {
        double expected = cm0[k] + offsets[cse][k];
        double d = cm[k] - expected;
        d -= L * round(d / L);
        TEST_ASSERT(fabs(d) < 1e-9);
      }

      double I_scale = 0.0;
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) I_scale = fmax(I_scale, fabs(I0[i][j]));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) TEST_ASSERT(fabs(I[i][j] - I0[i][j]) <= tol * fmax(1.0, I_scale));
    }

    MFTLEnsembleDestroy(&ens);
  }

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_analytic_sphere);
  RUN_TEST(test_inertia_sphere);
  RUN_TEST(test_rank_independence);
  RUN_TEST(test_periodic_seam_invariance);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
