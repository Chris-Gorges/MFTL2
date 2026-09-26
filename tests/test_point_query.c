/*
 * Stage 07 — point queries: point-in-object (ray casting), point-triangle
 * distance, AABB containment.
 *
 * Covers (per plans/07-point-queries-and-obb.md):
 *   - unit sphere: origin inside, (2,0,0) outside, facet-centroid on surface
 *     (-1), just inside/outside classified
 *   - point-triangle distance vs analytic values for the vertex/edge/
 *     interior nearest-feature cases (1e-12)
 *   - rank-independence: identical verdicts on 1/2/4/8 ranks, including a
 *     mirror of MFTL1 test3.c's probe battery (r=0.75 sphere at (0.5,0.5,0))
 *   - AABB containment incl. MFTLPointInsideObjectBox parity behaviour
 *
 * All MFTLPointInsideObject calls are collective and made by every rank.
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

#define CHECK(cond)                                                                              \
  do                                                                                             \
  {                                                                                              \
    if (!(cond))                                                                                 \
    {                                                                                            \
      printf("Rank %d: CHECK failed: (%s) at %s:%d\n", g_rank, #cond, __FILE__, __LINE__);       \
      fail = 1;                                                                                  \
    }                                                                                            \
  } while (0)

int test_point_triangle_distance()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  // Single unit right triangle in the z=0 plane.
  double pos[3][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  int triples[1][3] = {{0, 1, 2}};
  MFTLObjectDeclareOpen(obj, 1);  // Stage 16A: an open surface
  if (g_rank == 0) MFTLBuildTopologyFromTriples(obj, 3, (const double (*)[3]) pos, 1, (const int (*)[3]) triples);
  MFTLComputeTriangleProperties(obj);

  int fail = 0;
  if (g_rank == 0)
  {
    struct
    {
      double p[3];
      double expected;
      const char* region;
    } cases[] = {
        {{0.25, 0.25, 1.0}, 1.0, "interior projection (plane distance)"},
        {{0.25, 0.25, -2.0}, 2.0, "interior projection, negative side"},
        {{2.0, 0.0, 0.0}, 1.0, "edge V1 zone (beyond vertex (1,0,0))"},
        {{0.0, 2.0, 0.0}, 1.0, "edge V2 zone (beyond vertex (0,1,0))"},
        {{-1.0, -1.0, 0.0}, sqrt(2.0), "vertex V0 zone"},
        {{0.5, -1.0, 0.0}, 1.0, "edge V0V1 zone"},
        {{-0.5, 0.25, 0.0}, 0.5, "edge V0V2 zone"},
        {{1.0, 1.0, 0.0}, sqrt(0.5), "vertex V1/V2 corner"},
    };
    int n = (int) (sizeof(cases) / sizeof(cases[0]));
    for (int c = 0; c < n; ++c)
    {
      double d = MFTLPointTriangleDistance(obj, 0, cases[c].p);
      double err = fabs(d - cases[c].expected);
      printf("  case %-38s d=%.15g expected %.15g err %.3e\n", cases[c].region, d, cases[c].expected, err);
      CHECK(err < 1e-12);
    }
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_sphere_inside_outside()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  int fail = 0;

  // On-surface probe: centroid of an arbitrary facet (exactly on the plane).
  double on_surface[3] = {0, 0, 0};
  if (g_rank == 0)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[0], vv);
    for (int k = 0; k < 3; ++k) on_surface[k] = (vv[0]->pos[k] + vv[1]->pos[k] + vv[2]->pos[k]) / 3.0;
  }
  MPI_Bcast(on_surface, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int rc_surface = MFTLPointInsideObject(obj, on_surface);  // collective
  if (g_rank == 0) printf("  on-surface verdict: %d (expect -1)\n", rc_surface);
  CHECK(rc_surface == -1);

  struct
  {
    double p[3];
    int expect;
    const char* what;
  } probes[] = {
      {{0, 0, 0}, 1, "origin inside"},
      {{2, 0, 0}, 0, "(2,0,0) outside"},
      {{0.97, 0.1, 0}, 1, "just inside"},
      {{1.03, 0.1, 0}, 0, "just outside"},
      {{0.5, 0.5, 0.5}, 1, "interior diagonal"},
      {{-0.9, -0.9, -0.9}, 0, "outside corner"},
  };
  int n = (int) (sizeof(probes) / sizeof(probes[0]));
  for (int c = 0; c < n; ++c)
  {
    int rc = MFTLPointInsideObject(obj, probes[c].p);  // collective
    if (g_rank == 0) printf("  %-18s (%g,%g,%g) -> %d (expect %d)\n", probes[c].what, probes[c].p[0], probes[c].p[1], probes[c].p[2], rc, probes[c].expect);
    CHECK(rc == probes[c].expect);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_rank_independence_battery()
{
  // Mirror of MFTL1 test3.c: sphere scaled 0.75, moved to (0.5, 0.5, 0),
  // redistributed; a battery of probes must classify identically on any rank
  // count. Expected verdicts derived from |p - centre| vs r = 0.75.
  double bmin[3] = {-5, -5, -5}, bmax[3] = {5, 5, 5};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  double s[3] = {0.75, 0.75, 0.75};
  MFTLScaleObject(obj, s);
  double dmove[3] = {0.5, 0.5, 0.0};
  MFTLTranslateObject(obj, dmove);
  MFTLRedistributeListOfObjects(ens);

  struct
  {
    double p[3];
    int expect;
  } probes[] = {
      {{-3.0, 3.0, 0.0}, 0},           // test3 point 1 (outside)
      {{0.0, -0.7, 0.7}, 0},           // test3 point 2 (outside)
      {{0.624856, 0.406454, -0.053072}, 1},  // test3 point 3 (inside)
      {{0.0, 0.0, 0.0}, 1},            // test3 point 4 (inside)
      {{0.5, 0.5, 0.0}, 1},            // centre
      {{0.5, 1.3, 0.0}, 0},            // above
      {{0.5, 0.5, -0.7}, 1},           // inside (0.7 < 0.75)
      {{0.5, 0.5, -0.8}, 0},           // outside (0.8 > 0.75)
  };
  int n = (int) (sizeof(probes) / sizeof(probes[0]));
  int fail = 0;
  for (int c = 0; c < n; ++c)
  {
    int rc = MFTLPointInsideObject(obj, probes[c].p);  // collective
    if (g_rank == 0) printf("  probe (%g,%g,%g) -> %d (expect %d)\n", probes[c].p[0], probes[c].p[1], probes[c].p[2], rc, probes[c].expect);
    CHECK(rc == probes[c].expect);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_aabb()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLRedistributeListOfObjects(ens);

  double c[3], d[3];
  MFTLObjectAABB(obj, c, d);  // collective

  int fail = 0;
  if (g_rank == 0)
  {
    printf("  AABB centre (%g,%g,%g) dims (%g,%g,%g)\n", c[0], c[1], c[2], d[0], d[1], d[2]);
    CHECK(fabs(c[0]) < 1e-12 && fabs(c[1]) < 1e-12 && fabs(c[2]) < 1e-12);
    CHECK(d[0] <= 1.0 + 1e-9 && d[0] > 0.99);
    CHECK(fabs(d[0] - d[1]) < 1e-12 && fabs(d[1] - d[2]) < 1e-12);

    CHECK(MFTLPointInsideAABB(c, d, (double[]) {0.9, 0.9, 0.9}) == 1);
    CHECK(MFTLPointInsideAABB(c, d, (double[]) {1.5, 0, 0}) == 0);
    CHECK(MFTLPointInsideAABB(c, d, (double[]) {0, -1.5, 0}) == 0);
    // slack: just outside by less than MFTL_VERYSMALL still counts as inside
    CHECK(MFTLPointInsideAABB(c, d, (double[]) {d[0] + 1e-16, 0, 0}) == 1);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_bvh_built_once()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  // Battery of probes. The ENGINEERED probe is constructed so that the
  // segment to RayEnd passes exactly through sphere vertex 0 (fixed RNG
  // seed 0): the first ray flags the grazing vertex as degenerate and ALL
  // ranks retry with the next direction, so this query deterministically
  // needs >= 2 ray-casting attempts. Pre-fix (BVH rebuilt per attempt) the
  // build count exceeds the query count; post-fix it must equal it.
  double vpos[3] = {0, 0, 0};
  if (g_rank == 0) memcpy(vpos, obj->vertices[0].pos, 3 * sizeof(double));
  MPI_Bcast(vpos, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  int fail = 0;
  double p_vertex[3] = {-0.73282041429574329, 0.25097965928725896, 0.80513537442031691};

  struct
  {
    double p[3];
    int min_attempts;
  } probes[] = {
      {{0, 0, 0}, 1},
      {{1.05, 0, 0}, 1},
      {{0.97, 0.1, 0}, 1},
      {{p_vertex[0], p_vertex[1], p_vertex[2]}, 2},  // ray passes exactly through vertex 0 -> degenerate retry
      {{0.5, 0.5, 0.5}, 1},
      {{-0.9, -0.9, -0.9}, 1},
      {{0.3, -0.95, 0.1}, 1},
  };
  int n = (int) (sizeof(probes) / sizeof(probes[0]));

  int builds_before = MFTLDebugPointQueryBVHBuilds();
  int retried = 0;
  for (int c = 0; c < n; ++c)
  {
    int rc = MFTLPointInsideObject(obj, probes[c].p);  // collective
    int attempts = MFTLDebugPointQueryLastIterations();
    if (g_rank == 0)
      printf("  probe %d (%g,%g,%g) -> %d in %d attempt(s)\n", c, probes[c].p[0], probes[c].p[1], probes[c].p[2], rc, attempts);
    if (attempts >= probes[c].min_attempts && probes[c].min_attempts > 1) retried++;
    if (attempts < probes[c].min_attempts)
    {
      printf("Rank %d: probe %d used %d attempts, expected >= %d\n", g_rank, c, attempts, probes[c].min_attempts);
      fail = 1;
    }
    (void) rc;
  }
  int builds_used = MFTLDebugPointQueryBVHBuilds() - builds_before;

  if (g_rank == 0) printf("  BVH builds: %d for %d queries (retrying probes: %d)\n", builds_used, n, retried);
  CHECK(builds_used == n);  // exactly one BVH build per query, however many retries
  CHECK(retried >= 1);      // the engineered probe exercised the retry path

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_point_triangle_distance);
  RUN_TEST(test_sphere_inside_outside);
  RUN_TEST(test_rank_independence_battery);
  RUN_TEST(test_aabb);
  RUN_TEST(test_bvh_built_once);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
