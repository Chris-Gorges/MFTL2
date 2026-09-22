/*
 * Stage 11 — overlap-volume quantification tests (plans/11).
 *
 *   1. Analytic lens: two unit spheres at centre distance d < 2R -> the
 *      summed prism overlap volume matches pi*(4R+d)*(2R-d)^2/12 within
 *      discretization tolerance, and converges with refinement.
 *   2. Symmetry: the reversed pair (B,A) gives the same volume (tol 1e-9).
 *   3. Periodic: a pair meeting only across the X seam gives the same volume
 *      as the identical pair placed mid-domain.
 *   4. Detection: sphere vs square face (MFTL1 test5 geometry), sphere vs
 *      ellipsoid across a z-cut (test6 geometry) — hit counts > 0.
 *   5. Rank independence: the lens volume is identical on 1/2/4/8 ranks.
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_overlap.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>

static int g_rank, g_nproc;
static MFTL_Ensemble* g_ens = NULL;

static double analytic_lens(double R, double d) { return M_PI * (4.0 * R + d) * (2.0 * R - d) * (2.0 * R - d) / 12.0; }

static MFTL_Ensemble* make_ensemble(double L, int periodic)
{
  int cuts[3] = {g_nproc, 1, 1};
  double min[3] = {-L, -L, -L};
  double max[3] = {L, L, L};
  bool per[3] = {(bool) periodic, (bool) periodic, (bool) periodic};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, min, max, per, 8);
}

// Build a two-sphere ensemble at the given centres; returns the overlap
// volume sum through the public driver.
static double two_sphere_overlap(double L, int periodic, const double cA[3], const double cB[3], double radius, int ref,
                                 double* out_norm_sign)
{
  MFTL_Ensemble* ens = make_ensemble(L, periodic);
  TEST_ASSERT(ens != NULL);
  MFTL_Object* a = MFTLObjectCreate(0);
  MFTL_Object* b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, a);
  MFTLEnsembleAddObject(ens, b);
  MFTLGenerateSphere(a, (double*) cA, radius, ref);
  MFTLGenerateSphere(b, (double*) cB, radius, ref);
  MFTLComputeTriangleProperties(a);
  MFTLComputeTriangleProperties(b);
  MFTLRedistributeListOfObjects(ens);

  MFTL_Overlap* recs = NULL;
  int n = 0;
  int rc = MFTLEnsembleCalculateOverlap(ens, NULL, &recs, &n);
  TEST_ASSERT(rc == MFTL_SUCCESS);

  double V = 0.0;
  for (int i = 0; i < n; ++i) V += recs[i].Volume;
  if (out_norm_sign && n > 0)
  {
    // AverageSurfaceNorm = (N_A - N_B)/2 with outward normals: at the
    // contact N_A ~ +x (A->B direction) and N_B ~ -x, so the x component
    // must be positive.
    double sx = 0.0;
    for (int i = 0; i < n; ++i) sx += recs[i].AverageSurfaceNorm[0];
    *out_norm_sign = sx / (double) n;
  }

  free(recs);
  MFTLEnsembleDestroy(&ens);
  return V;
}

static int test_analytic_lens(void)
{
  double cA[3] = {0.0, 0.0, 0.0};
  double cB[3] = {0.5, 0.0, 0.0};
  double R = 1.0, d = 0.5;
  double expect = analytic_lens(R, d);

  double v4 = two_sphere_overlap(2.0, 0, cA, cB, R, 4, NULL);
  double v5 = two_sphere_overlap(2.0, 0, cA, cB, R, 5, NULL);

  if (g_rank == 0) printf("  lens: ref4 V=%.6f ref5 V=%.6f analytic=%.6f\n", v4, v5, expect);
  TEST_ASSERT(v4 > 0.0);
  TEST_ASSERT(fabs(v4 - expect) < 0.10 * expect);   // ~10% discretization slack
  TEST_ASSERT(fabs(v5 - expect) < 0.05 * expect);   // converges with refinement
  TEST_ASSERT(fabs(v5 - expect) < fabs(v4 - expect));
  return 0;
}

static int test_symmetry(void)
{
  // (A,B) at (0,0,0)/(0.5,0,0) vs the reversed body order (B,A).
  double cA[3] = {0.0, 0.0, 0.0};
  double cB[3] = {0.5, 0.0, 0.0};
  double v_ab = two_sphere_overlap(2.0, 0, cA, cB, 1.0, 4, NULL);
  double v_ba = two_sphere_overlap(2.0, 0, cB, cA, 1.0, 4, NULL);
  if (g_rank == 0) printf("  symmetry: V(A,B)=%.12f  V(B,A)=%.12f\n", v_ab, v_ba);
  TEST_ASSERT(v_ab > 0.0);
  TEST_ASSERT(fabs(v_ab - v_ba) < 1e-9 * fmax(1.0, v_ab));
  return 0;
}

static int test_periodic_seam(void)
{
  double L = 2.0;
  // Across the X seam (both bodies straddle it; their minimum-image
  // distance is 0.3), vs the identical pair mid-domain.
  double cA[3] = {-1.85, 0.0, 0.0};
  double cB[3] = {1.85, 0.0, 0.0};
  double mA[3] = {-0.15, 0.0, 0.0};
  double mB[3] = {0.15, 0.0, 0.0};

  double v_seam = two_sphere_overlap(L, 1, cA, cB, 0.5, 4, NULL);
  double v_mid = two_sphere_overlap(L, 1, mA, mB, 0.5, 4, NULL);
  if (g_rank == 0) printf("  periodic: seam V=%.12f  mid V=%.12f\n", v_seam, v_mid);
  TEST_ASSERT(v_seam > 0.0);
  TEST_ASSERT(fabs(v_seam - v_mid) < 1e-9 * fmax(1.0, v_mid));
  return 0;
}

// Counting callback for the detection regressions.
typedef struct
{
  int sphere_face;
  int sphere_ell;
} DetectCounts;

static void detect_cb(MFTL_Object* objA, int triA, MFTL_Object* objB, int triB, void* user_data)
{
  (void) triA;
  (void) triB;
  DetectCounts* dc = (DetectCounts*) user_data;
  int ids = objA->global_id + objB->global_id;
  if (ids == 1) dc->sphere_face++;   // objects 0 (sphere) and 1 (face)
  if (ids == 2) dc->sphere_ell++;    // objects 0 (sphere) and 2 (ellipsoid)
}

// Detection regressions in the spirit of MFTL1 test5/test6: sphere vs flat
// square face, and sphere vs stretched/rotated ellipsoid across a z-cut.
static int test_detection_regressions(void)
{
  // Domain sized to contain the transformed ellipsoid (x extent ~2.75).
  MFTL_Ensemble* ens = make_ensemble(4.0, 0);
  TEST_ASSERT(ens != NULL);

  // test5-like: unit sphere at the origin vs the unit square face [0,1]^2 in
  // the z=0 plane (the sphere's equator circle passes through the square).
  // NOTE: a face that lies fully INSIDE the ball (e.g. centred at z=0.3 with
  // extent 0.5) produces no triangle crossings at all — the surface never
  // reaches it — so the geometry must actually pierce the sphere.
  MFTL_Object* sphere = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, sphere);
  double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(sphere, c, 1.0, 3);

  MFTL_Object* face = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, face);
  {
    double pos[4][3] = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {1.0, 1.0, 0.0}, {0.0, 1.0, 0.0}};
    int tri[2][3] = {{0, 1, 2}, {0, 2, 3}};
    MFTLBuildTopologyFromTriples(face, 4, pos, 2, tri);
    MFTLComputeTriangleProperties(face);
  }

  // test6-like: sphere vs an ellipsoid stretched in x and rotated about z.
  MFTL_Object* ell = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(ens, ell);
  {
    double c2[3] = {0.25, -0.1, 0.0};
    MFTLGenerateSphere(ell, c2, 1.0, 3);
    double s[3] = {2.0, 0.5, 0.5};
    MFTLScaleObject(ell, s);
    double R[3][3] = {{cos(0.523599), -sin(0.523599), 0.0}, {sin(0.523599), cos(0.523599), 0.0}, {0.0, 0.0, 1.0}};
    double origin[3] = {0.0, 0.0, 0.0};
    MFTLRotateObject(ell, R, origin);
    double d[3] = {0.25, -0.1, 0.0};
    MFTLTranslateObject(ell, d);
    MFTLComputeTriangleProperties(ell);
  }
  MFTLComputeTriangleProperties(sphere);

  MFTLRedistributeListOfObjects(ens);

  DetectCounts dc = {0, 0};
  MFTLEnsembleIntersectObjects(ens, NULL, detect_cb, &dc);
  int sf = dc.sphere_face, se = dc.sphere_ell;
  MPI_Allreduce(MPI_IN_PLACE, &sf, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &se, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  detection: sphere-face hits=%d  sphere-ellipsoid hits=%d\n", sf, se);
  TEST_ASSERT(sf > 0);
  TEST_ASSERT(se > 0);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// parallel_intersection_spheres as an assertion: 0 collisions before
// contact, > 0 once the spheres touch. Two unit spheres are disjoint only
// for d > 2R = 2 (d = 1.9 still overlaps by 0.1), so "far" uses d = 2.05.
static void count_cb(MFTL_Object* a, int ta, MFTL_Object* b, int tb, void* ud)
{
  (void) a; (void) ta; (void) b; (void) tb;
  (*(int*) ud)++;
}

static int count_hits(double d, int ref)
{
  MFTL_Ensemble* ens = make_ensemble(4.0, 0);
  MFTL_Object* a = MFTLObjectCreate(0);
  MFTL_Object* b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, a);
  MFTLEnsembleAddObject(ens, b);
  double cA[3] = {0.0, 0.0, 0.0}, cB[3] = {d, 0.0, 0.0};
  MFTLGenerateSphere(a, cA, 1.0, ref);
  MFTLGenerateSphere(b, cB, 1.0, ref);
  MFTLComputeTriangleProperties(a);
  MFTLComputeTriangleProperties(b);
  MFTLRedistributeListOfObjects(ens);
  int hits = 0;
  MFTLEnsembleIntersectObjects(ens, NULL, count_cb, &hits);
  MPI_Allreduce(MPI_IN_PLACE, &hits, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MFTLEnsembleDestroy(&ens);
  return hits;
}

static int test_contact_assertion(void)
{
  double cA[3] = {0.0, 0.0, 0.0};

  // Domain sized to contain the d = 2.05 sphere (x extent ~3.05).
  double far = two_sphere_overlap(4.0, 0, cA, (double[]) {2.05, 0.0, 0.0}, 1.0, 3, NULL);
  double near = two_sphere_overlap(4.0, 0, cA, (double[]) {1.9, 0.0, 0.0}, 1.0, 3, NULL);
  int hits_far = count_hits(2.05, 3);
  int hits_near = count_hits(1.9, 3);
  if (g_rank == 0)
    printf("  contact: far V=%.3e (%d hits)  near V=%.6f (%d hits, analytic %.6f)\n", far, hits_far, near, hits_near, analytic_lens(1.0, 1.9));
  TEST_ASSERT(fabs(far) < 1e-12);
  TEST_ASSERT(hits_far == 0);
  TEST_ASSERT(near > 0.0);
  TEST_ASSERT(hits_near > 0);
  TEST_ASSERT(fabs(near - analytic_lens(1.0, 1.9)) < 0.15 * analytic_lens(1.0, 1.9));
  return 0;
}

// AverageSurfaceNorm = (N_A - N_B)/2 with OUTWARD normals: at the contact of
// two spheres along +x, N_A ~ +x (out of A toward B) and N_B ~ -x, so the
// average points +x (the A->B direction, MFTL1's convention).
static int test_normal_direction(void)
{
  double cA[3] = {0.0, 0.0, 0.0};
  double cB[3] = {0.5, 0.0, 0.0};
  double sign = 0.0;
  double v = two_sphere_overlap(2.0, 0, cA, cB, 1.0, 4, &sign);
  TEST_ASSERT(v > 0.0);
  TEST_ASSERT(sign > 0.0);
  return 0;
}

// Public 2D convex-hull helpers (Graham scan): square + interior points.
static int test_convex_hull(void)
{
  MFTL_Point2D pts[8] = {{0.0, 0.0}, {1.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {0.5, 0.5}, {0.25, 0.75}, {0.75, 0.25}, {0.5, 0.9}};
  int n = 8;
  int rc = MFTLConvexHull2D(pts, &n);
  TEST_ASSERT(rc == 0 || rc == 1);  // implementation-defined success code
  TEST_ASSERT(n == 4);
  double area = MFTLPolygonArea2D(pts, n);
  double cen[2];
  MFTLPolygonCentroid2D(pts, n, cen);
  if (g_rank == 0) printf("  hull: n=%d area=%.6f centre=(%.6f,%.6f)\n", n, area, cen[0], cen[1]);
  TEST_ASSERT(fabs(fabs(area) - 1.0) < 1e-12);
  TEST_ASSERT(fabs(cen[0] - 0.5) < 1e-12 && fabs(cen[1] - 0.5) < 1e-12);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_analytic_lens);
  RUN_TEST(test_symmetry);
  RUN_TEST(test_periodic_seam);
  RUN_TEST(test_detection_regressions);
  RUN_TEST(test_contact_assertion);
  RUN_TEST(test_normal_direction);
  RUN_TEST(test_convex_hull);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
