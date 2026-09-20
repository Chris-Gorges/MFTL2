/*
 * Stage 03 — rigid-body transforms (scale / translate / rotate).
 *
 * Covers (per plans/03-rigid-body-transforms.md):
 *   - volume & area invariance under rotation and translation (1e-9)
 *   - isotropic scale by 2 -> volume x8, area x4
 *   - rotate(theta) then rotate(-theta) round-trip, vertices back to 1e-12
 *   - rank-independence of post-transform integrals on 1/2/4/8 ranks
 *   - MFTL1 test1.c regression sequence: sphere ref4 -> scale {2,1,1}
 *     -> rotate 30 deg about z -> translate {0.5,0,0}
 *
 * Note: MFTL1's printed reference values are unavailable (MFTL1 not built
 * here); the regression asserts analytically-derived values for the same
 * sequence instead (spheroid area 21.4785 for a=2,b=c=1; centre (0.5,0,0)).
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
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

static void rotz(double R[3][3], double angle)
{
  double c = cos(angle), s = sin(angle);
  memset(R, 0, 9 * sizeof(double));
  R[0][0] = c;
  R[0][1] = -s;
  R[1][0] = s;
  R[1][1] = c;
  R[2][2] = 1.0;
}

// Fixed pseudo-random orthonormal rotation: R = Rz(a) * Ry(b) * Rx(g).
static void make_rotation(double R[3][3])
{
  const double a = 0.731, b = -0.412, g = 1.123;
  double ca = cos(a), sa = sin(a), cb = cos(b), sb = sin(b), cg = cos(g), sg = sin(g);

  double Rz[3][3] = {{ca, -sa, 0}, {sa, ca, 0}, {0, 0, 1}};
  double Ry[3][3] = {{cb, 0, sb}, {0, 1, 0}, {-sb, 0, cb}};
  double Rx[3][3] = {{1, 0, 0}, {0, cg, -sg}, {0, sg, cg}};

  double tmp[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) tmp[i][j] = Rz[i][0] * Ry[0][j] + Rz[i][1] * Ry[1][j] + Rz[i][2] * Ry[2][j];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) R[i][j] = tmp[i][0] * Rx[0][j] + tmp[i][1] * Rx[1][j] + tmp[i][2] * Rx[2][j];
}

int test_rigid_invariance()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistributeListOfObjects(ens);

  double cm[3] = {0, 0, 0};
  double A0 = MFTLSurfaceCenter(obj, cm);
  double V0 = MFTLObjectVolume(obj);
  TEST_ASSERT(V0 < 0.0);  // sphere generator winds inward: volume sign pinned

  // Random rotation about a non-trivial centre
  double R[3][3];
  make_rotation(R);
  double centre[3] = {0.3, -0.2, 0.1};
  MFTLRotateObject(obj, R, centre);
  MFTLUpdateGhostPositions(ens, NULL);

  double A1 = MFTLSurfaceCenter(obj, cm);
  double V1 = MFTLObjectVolume(obj);
  TEST_ASSERT(fabs(fabs(V1) - fabs(V0)) <= 1e-9 * fmax(1.0, fabs(V0)));
  TEST_ASSERT(fabs(A1 - A0) <= 1e-9 * fmax(1.0, A0));

  // Translation (centroid follows: cm = (I-R)*centre + d, integrals invariant)
  double d[3] = {0.7, -1.2, 0.4};
  MFTLTranslateObject(obj, d);
  MFTLUpdateGhostPositions(ens, NULL);

  double A2 = MFTLSurfaceCenter(obj, cm);
  double V2 = MFTLObjectVolume(obj);
  TEST_ASSERT(fabs(fabs(V2) - fabs(V0)) <= 1e-9 * fmax(1.0, fabs(V0)));
  TEST_ASSERT(fabs(A2 - A0) <= 1e-9 * fmax(1.0, A0));
  for (int k = 0; k < 3; ++k)
  {
    double expected = d[k];
    for (int j = 0; j < 3; ++j) expected += ((k == j ? 1.0 : 0.0) - R[k][j]) * centre[j];
    TEST_ASSERT(fabs(cm[k] - expected) < 1e-10);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_scale_laws()
{
  double bmin[3] = {-4, -4, -4}, bmax[3] = {4, 4, 4};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  double cm[3] = {0, 0, 0};
  double A0 = MFTLSurfaceCenter(obj, cm);
  double V0 = MFTLObjectVolume(obj);

  double s[3] = {2.0, 2.0, 2.0};
  MFTLScaleObject(obj, s);

  double A1 = MFTLSurfaceCenter(obj, cm);
  double V1 = MFTLObjectVolume(obj);
  TEST_ASSERT(V0 < 0.0 && V1 < 0.0);  // sign preserved under scaling

  if (g_rank == 0) printf("  V0=%.17g V1=%.17g (ratio %.12g)  A0=%.17g A1=%.17g (ratio %.12g)\n", V0, V1, V1 / V0, A0, A1, A1 / A0);

  TEST_ASSERT(fabs(fabs(V1) / fabs(V0) - 8.0) < 1e-9);
  TEST_ASSERT(fabs(A1 / A0 - 4.0) < 1e-9);
  TEST_ASSERT(fabs(cm[0]) < 1e-12 && fabs(cm[1]) < 1e-12 && fabs(cm[2]) < 1e-12);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_rotation_roundtrip()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLComputeTriangleProperties(obj);

  // Snapshot owned vertex positions
  int n = obj->n_vertices;
  double* snapshot = malloc(n * 3 * sizeof(double));
  TEST_ASSERT(snapshot != NULL);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) snapshot[3 * i + k] = obj->vertices[i].pos[k];

  const double theta = 0.37;
  double R[3][3], Rt[3][3];
  rotz(R, theta);
  rotz(Rt, -theta);
  double centre[3] = {0.15, -0.4, 0.55};

  MFTLRotateObject(obj, R, centre);
  MFTLRotateObject(obj, Rt, centre);

  double max_err = 0.0;
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) max_err = fmax(max_err, fabs(obj->vertices[i].pos[k] - snapshot[3 * i + k]));

  if (g_rank == 0) printf("  round-trip max vertex error: %.3e\n", max_err);
  TEST_ASSERT(max_err < 1e-12);

  free(snapshot);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_rank_independence()
{
  double bmin[3] = {-3, -3, -3}, bmax[3] = {3, 3, 3};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  double cm_pre[3] = {0, 0, 0};
  double A_pre = MFTLSurfaceCenter(obj, cm_pre);
  double V_pre = MFTLObjectVolume(obj);
  TEST_ASSERT(V_pre < 0.0);  // sphere winding: sign pinned

  // Transform BEFORE redistribution: rank 0 moves the whole body locally.
  double R[3][3];
  make_rotation(R);
  double d[3] = {0.6, -0.3, 0.9};
  MFTLRotateObject(obj, R, origin);
  MFTLTranslateObject(obj, d);

  MFTLRedistributeListOfObjects(ens);
  MFTLUpdateGhostPositions(ens, NULL);

  double cm_post[3] = {0, 0, 0};
  double A_post = MFTLSurfaceCenter(obj, cm_post);
  double V_post = MFTLObjectVolume(obj);

  if (g_rank == 0)
    printf("  nproc=%d  V_pre=%.17g V_post=%.17g  cm_post=(%.17g,%.17g,%.17g)\n", g_nproc, V_pre, V_post, cm_post[0], cm_post[1], cm_post[2]);

  TEST_ASSERT(fabs(fabs(V_post) - fabs(V_pre)) <= 1e-9 * fmax(1.0, fabs(V_pre)));
  TEST_ASSERT(fabs(A_post - A_pre) <= 1e-9 * fmax(1.0, A_pre));
  for (int k = 0; k < 3; ++k) TEST_ASSERT(fabs(cm_post[k] - d[k]) < 1e-9);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int test_mftl1_test1_regression()
{
  double bmin[3] = {-5, -5, -5}, bmax[3] = {5, 5, 5};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 4);

  // test1.c step 1: MFTLScaleSurface({2,1,1}) -> prolate spheroid a=2, b=c=1.
  double s[3] = {2.0, 1.0, 1.0};
  MFTLScaleObject(obj, s);

  double cm[3] = {0, 0, 0};
  double A = MFTLSurfaceCenter(obj, cm);
  // Analytic spheroid surface area: 2 pi b^2 (1 + a/(b e) asin(e)), e = sqrt(1 - b^2/a^2).
  const double e = sqrt(3.0) / 2.0;
  const double A_exact = 2.0 * MFTL_PI * 1.0 * (1.0 + 2.0 / e * asin(e));  // 21.4785...
  if (g_rank == 0) printf("  spheroid A=%.17g (analytic %.17g, rel err %.3e)\n", A, A_exact, fabs(A - A_exact) / A_exact);
  TEST_ASSERT(fabs(A - A_exact) / A_exact < 5e-3);
  TEST_ASSERT(fabs(cm[0]) + fabs(cm[1]) + fabs(cm[2]) < 1e-10);

  // test1.c step 2: MFTLRotateBody({0,0,0.523599}, origin) then MFTLMoveBody({0.5,0,0}).
  double R[3][3];
  rotz(R, 0.523599);
  MFTLRotateObject(obj, R, origin);
  double d[3] = {0.5, 0.0, 0.0};
  MFTLTranslateObject(obj, d);

  double A2 = MFTLSurfaceCenter(obj, cm);
  TEST_ASSERT(fabs(A2 - A) <= 1e-9 * fmax(1.0, A));
  TEST_ASSERT(fabs(cm[0] - 0.5) < 1e-12 && fabs(cm[1]) < 1e-12 && fabs(cm[2]) < 1e-12);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_rigid_invariance);
  RUN_TEST(test_scale_laws);
  RUN_TEST(test_rotation_roundtrip);
  RUN_TEST(test_rank_independence);
  RUN_TEST(test_mftl1_test1_regression);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
