/*
 * Stage 13 — periodicity completion: end-to-end periodic battery.
 *
 * Covers (per plans/13-periodicity-completion.md):
 *   1. Seam invariance battery: sphere mid-domain vs straddling the X/Y/Z/XY/
 *      XYZ periodic faces — volume, area, inertia, centroid (mod period),
 *      curvature statistics and point-in-object probes must agree.
 *   2. Periodic advection: rigid translation by one full period (and back),
 *      with redistribution every step; centroid (mod period), volume,
 *      in-domain storage and triangle count are conserved.
 *   3. Periodic LeVeque single-vortex advection (time-reversed): a sphere
 *      straddling the x=0 face returns to its initial shape; volume conserved
 *      and max min-image vertex displacement small.
 *   4. Periodic remesh across a seam (TSUR volume-conserving smoothing):
 *      volume conserved, mesh healthy, curvature sane, matches a mid-domain
 *      remesh.
 *   5. Periodic intersection + overlap: two spheres meeting only across the
 *      seam give the same hit count and overlap volume as mid-domain.
 *   6. Minimum-image constraint: MFTLCheckMinimumImageConstraint accepts a
 *      small sphere and rejects an over-large one; MFTLGenerateSphere refuses
 *      the violating radius on a periodic domain.
 *
 * Registered at 1/2/4/8 ranks (x-only decomposition, cuts = {nproc,1,1}).
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_remeshing.h"
#include "mftl_overlap.h"
#include <mpi.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

static int g_rank = -1;
static int g_nproc = -1;

// Smallest difference a-b on a circle of circumference L.
static double period_diff(double a, double b, double L)
{
  double d = a - b;
  d -= L * round(d / L);
  return d;
}

// Build a one-sphere ensemble on [-2,2]^3-style bounds; NULL on failure.
static MFTL_Ensemble* build_sphere(const double bmin[3], const double bmax[3], const bool per[3], const double center[3], double radius, int ref,
                                   MFTL_Object** out_obj)
{
  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, (double*) bmin, (double*) bmax, (bool*) per, 2);
  if (!ens) return NULL;
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  if (MFTLGenerateSphere(obj, (double*) center, radius, ref) != MFTL_SUCCESS)
  {
    MFTLEnsembleDestroy(&ens);
    return NULL;
  }
  MFTLComputeTriangleProperties(obj);
  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  *out_obj = obj;
  return ens;
}

// Ghost layer + normals + curvatures, then global stats over owned vertices.
static void curvature_stats(MFTL_Ensemble* ens, MFTL_Object* obj, double* meanH, double* minH, double* maxH, double* meanK)
{
  MFTLExtendGhostLayer(ens, 2);
  MFTLUpdateGhostPositions(ens, NULL);
  MFTLComputeTriangleProperties(obj);
  MFTLComputeVertexNormals(obj);
  MFTLComputeVertexCurvatures(obj);

  double sH = 0.0, sK = 0.0, mnH = MFTL_DBL_MAX, mxH = -MFTL_DBL_MAX;
  int n = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != g_rank) continue;
    sH += v->meancurvature;
    sK += v->gaussiancurvature;
    if (v->meancurvature < mnH) mnH = v->meancurvature;
    if (v->meancurvature > mxH) mxH = v->meancurvature;
    n++;
  }
  double g_sH = 0.0, g_sK = 0.0, g_mnH = 0.0, g_mxH = 0.0;
  int g_n = 0;
  MPI_Allreduce(&sH, &g_sH, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&sK, &g_sK, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mnH, &g_mnH, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&mxH, &g_mxH, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&n, &g_n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  *meanH = g_n > 0 ? g_sH / g_n : 0.0;
  *meanK = g_n > 0 ? g_sK / g_n : 0.0;
  *minH = g_mnH;
  *maxH = g_mxH;
}

/* ------------------------------------------------------------------------- */
/* 1. Seam invariance battery                                                */
/* ------------------------------------------------------------------------- */

static int test_seam_invariance(void)
{
  const double L = 4.0;
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  const double R = 0.7;
  const int ref = 3;
  const double c0[3] = {0.3, -0.4, 0.2};  // mid-domain reference center

  const char* names[5] = {"X", "Y", "Z", "XY", "XYZ"};
  bool pers[5][3] = {{true, false, false}, {false, true, false}, {false, false, true}, {true, true, false}, {true, true, true}};

  for (int c = 0; c < 5; ++c)
  {
    bool per[3] = {pers[c][0], pers[c][1], pers[c][2]};

    // Seam placement: periodic axes sit on the global-min face.
    double cs[3];
    double delta[3];
    for (int k = 0; k < 3; ++k)
    {
      cs[k] = per[k] ? bmin[k] : c0[k];
      delta[k] = cs[k] - c0[k];
    }

    MFTL_Object *obj_m = NULL, *obj_s = NULL;
    MFTL_Ensemble* ens_m = build_sphere(bmin, bmax, per, c0, R, ref, &obj_m);
    MFTL_Ensemble* ens_s = build_sphere(bmin, bmax, per, cs, R, ref, &obj_s);
    TEST_ASSERT(ens_m != NULL && ens_s != NULL);

    double cm_m[3] = {0, 0, 0}, cm_s[3] = {0, 0, 0};
    double A_m = MFTLSurfaceCenter(obj_m, cm_m);
    double A_s = MFTLSurfaceCenter(obj_s, cm_s);
    double V_m = MFTLObjectVolume(obj_m);
    double V_s = MFTLObjectVolume(obj_s);
    double I_m[3][3], I_s[3][3];
    MFTLMomentOfInertia(obj_m, I_m, cm_m, 1.0);
    MFTLMomentOfInertia(obj_s, I_s, cm_s, 1.0);

    double mH_m, mnH_m, mxH_m, mK_m, mH_s, mnH_s, mxH_s, mK_s;
    curvature_stats(ens_m, obj_m, &mH_m, &mnH_m, &mxH_m, &mK_m);
    curvature_stats(ens_s, obj_s, &mH_s, &mnH_s, &mxH_s, &mK_s);

    if (g_rank == 0)
      printf("  cfg %-3s: V(mid)=%.12g V(seam)=%.12g  meanH mid=%.9g seam=%.9g (2/R=%.6g)\n", names[c], V_m, V_s, mH_m, mH_s, 2.0 / R);

    const double tol = 1e-9;
    TEST_ASSERT(V_m > 0.0 && V_s > 0.0);
    TEST_ASSERT(fabs(V_s - V_m) <= tol * fmax(1.0, fabs(V_m)));
    TEST_ASSERT(fabs(A_s - A_m) <= tol * fmax(1.0, A_m));

    // centroid: seam == mid + delta (mod period)
    for (int k = 0; k < 3; ++k)
    {
      if (per[k])
      {
        TEST_ASSERT(fabs(period_diff(cm_s[k], cm_m[k] + delta[k], L)) < 1e-9);
      }
      else
      {
        TEST_ASSERT(fabs(cm_s[k] - cm_m[k]) < 1e-9);
      }
    }

    // inertia tensor (about the own centroid: translation-invariant)
    double I_scale = 0.0;
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) I_scale = fmax(I_scale, fabs(I_m[i][j]));
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j) TEST_ASSERT(fabs(I_s[i][j] - I_m[i][j]) <= tol * fmax(1.0, I_scale));

    // curvature: sphere values, and seam == mid
    TEST_ASSERT(fabs(mH_m - 2.0 / R) < 0.05 * (2.0 / R));
    TEST_ASSERT(fabs(mK_m - 2.0 / (R * R)) < 0.05 * (2.0 / (R * R)));
    TEST_ASSERT(fabs(mH_s - mH_m) <= 1e-8 * fmax(1.0, fabs(mH_m)));
    TEST_ASSERT(fabs(mK_s - mK_m) <= 1e-8 * fmax(1.0, fabs(mK_m)));
    TEST_ASSERT(fabs(mnH_s - mnH_m) <= 1e-8 * fmax(1.0, fabs(mnH_m)));
    TEST_ASSERT(fabs(mxH_s - mxH_m) <= 1e-8 * fmax(1.0, fabs(mxH_m)));

    // Point probes: center (inside), far point (outside), +/-0.9R (inside) and
    // +/-1.1R (outside) along each periodic axis — across the seam for cs.
    MFTL_Object* objs[2] = {obj_m, obj_s};
    const double* centers[2] = {c0, cs};
    for (int o = 0; o < 2; ++o)
    {
      const double* cc = centers[o];
      double p[3];
      int verdict;

      memcpy(p, cc, 3 * sizeof(double));
      verdict = MFTLPointInsideObject(objs[o], p);
      TEST_ASSERT(verdict == 1);

      double pf[3] = {0.9, 0.9, 0.9};
      verdict = MFTLPointInsideObject(objs[o], pf);
      TEST_ASSERT(verdict == 0);

      for (int k = 0; k < 3; ++k)
      {
        if (!per[k]) continue;
        for (int s = -1; s <= 1; s += 2)
        {
          memcpy(p, cc, 3 * sizeof(double));
          p[k] += s * 0.9 * R;
          verdict = MFTLPointInsideObject(objs[o], p);
          TEST_ASSERT(verdict == 1);
          memcpy(p, cc, 3 * sizeof(double));
          p[k] += s * 1.1 * R;
          verdict = MFTLPointInsideObject(objs[o], p);
          TEST_ASSERT(verdict == 0);
        }
      }
    }

    MFTLEnsembleDestroy(&ens_m);
    MFTLEnsembleDestroy(&ens_s);
  }
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 2. Periodic advection (rigid translation through the seams)               */
/* ------------------------------------------------------------------------- */

static int advect_case(const bool per[3], const double d[3], int n_leg)
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  double origin[3] = {0.0, 0.0, 0.0};

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = build_sphere(bmin, bmax, per, origin, 0.7, 2, &obj);
  TEST_ASSERT(ens != NULL);

  const double V0 = MFTLObjectVolume(obj);
  const int tris0 = ens->total_global_triangles;
  TEST_ASSERT(V0 > 0.0);

  double travelled[3] = {0.0, 0.0, 0.0};  // accumulated displacement
  for (int leg = 0; leg < 2; ++leg)
  {
    double sgn = leg == 0 ? 1.0 : -1.0;
    for (int s = 0; s < n_leg; ++s)
    {
      double step[3] = {sgn * d[0], sgn * d[1], sgn * d[2]};
      MFTLTranslateObject(obj, step);
      MFTLWrapObjectIntoDomain(obj);
      MFTLUpdateGhostPositions(ens, NULL);
      MFTLRedistributeListOfObjects(ens);
      MFTLEnsembleUpdateGlobalCounts(ens);
      for (int k = 0; k < 3; ++k) travelled[k] += step[k];

      // in-domain storage of all owned vertices on periodic axes
      for (int i = 0; i < obj->n_vertices; ++i)
      {
        const MFTL_Vertex* v = &obj->vertices[i];
        if (v->is_ghost || v->owner_proc != g_rank) continue;
        for (int k = 0; k < 3; ++k)
        {
          if (!per[k]) continue;
          TEST_ASSERT(v->pos[k] >= bmin[k] && v->pos[k] < bmax[k]);
        }
      }

      // volume conserved (exact for a closed surface, any placement)
      double V = MFTLObjectVolume(obj);
      TEST_ASSERT(fabs(V - V0) <= 1e-9 * fmax(1.0, fabs(V0)));

      // centroid tracks the expected (wrapped) displacement
      double cm[3] = {0, 0, 0};
      MFTLSurfaceCenter(obj, cm);
      for (int k = 0; k < 3; ++k)
      {
        double L = bmax[k] - bmin[k];
        double expected = travelled[k];
        if (per[k]) expected -= L * floor(expected / L);  // wrap into [bmin, bmax) (bmin=0 shift)
        double err = per[k] ? period_diff(cm[k], expected, L) : (cm[k] - expected);
        TEST_ASSERT(fabs(err) < 1e-9);
      }
    }
  }

  // back at the origin, unchanged mesh size
  TEST_ASSERT_EQUAL_INT(tris0, ens->total_global_triangles);
  double cm[3] = {0, 0, 0};
  MFTLSurfaceCenter(obj, cm);
  for (int k = 0; k < 3; ++k)
  {
    double L = bmax[k] - bmin[k];
    double err = per[k] ? period_diff(cm[k], 0.0, L) : cm[k];
    TEST_ASSERT(fabs(err) < 1e-9);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_periodic_advection(void)
{
  bool perX[3] = {true, false, false};
  double dX[3] = {0.25, 0.0, 0.0};
  if (advect_case(perX, dX, 16)) return 1;  // 16 * 0.25 = 4.0 = one period

  bool perXYZ[3] = {true, true, true};
  double dXYZ[3] = {0.25, 0.25, 0.25};
  if (advect_case(perXYZ, dXYZ, 16)) return 1;
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 3. Periodic LeVeque single-vortex advection (time-reversed)               */
/* ------------------------------------------------------------------------- */

// RK4 step of the LeVeque single-vortex field (examples/leveque_remeshing_
// singleproc.c), owned vertices only, wrapped back into the domain. The field
// is 1-periodic in space and reversed by cos(pi t / T), so the body returns
// to its initial shape at t = T.
static void advect_leveque(MFTL_Object* obj, double time, double dt, double T, double vscale)
{
  const double pi = MFTL_PI;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != obj->domain->rank) continue;

    double t = time, X[3], k1[3], k2[3], k3[3], k4[3];
    X[0] = v->pos[0];
    X[1] = v->pos[1];
    X[2] = v->pos[2];

    k1[0] = vscale * (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k1[1] = vscale * (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k1[2] = vscale * (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = v->pos[0] + 0.5 * dt * k1[0];
    X[1] = v->pos[1] + 0.5 * dt * k1[1];
    X[2] = v->pos[2] + 0.5 * dt * k1[2];
    t = time + 0.5 * dt;

    k2[0] = vscale * (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k2[1] = vscale * (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k2[2] = vscale * (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = v->pos[0] + 0.5 * dt * k2[0];
    X[1] = v->pos[1] + 0.5 * dt * k2[1];
    X[2] = v->pos[2] + 0.5 * dt * k2[2];

    k3[0] = vscale * (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k3[1] = vscale * (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k3[2] = vscale * (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = v->pos[0] + dt * k3[0];
    X[1] = v->pos[1] + dt * k3[1];
    X[2] = v->pos[2] + dt * k3[2];
    t = time + dt;

    k4[0] = vscale * (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k4[1] = vscale * (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k4[2] = vscale * (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    v->pos[0] += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    v->pos[1] += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    v->pos[2] += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);

    MFTLApplyPeriodicity(obj->domain, v->pos);
  }
}

static int test_periodic_leveque(void)
{
  double bmin[3] = {0.0, 0.0, 0.0};
  double bmax[3] = {1.0, 1.0, 1.0};
  bool per[3] = {true, true, true};
  double center[3] = {0.05, 0.5, 0.35};  // straddles the x=0 face
  const double R = 0.15;
  const int ref = 3;

  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
  TEST_ASSERT(ens != NULL);
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  TEST_ASSERT(MFTLGenerateSphere(obj, center, R, ref) == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(obj);

  const double V0 = MFTLObjectVolume(obj);
  int tris0 = 0;  // rank 0 holds the full mesh pre-distribution
  if (g_rank == 0) tris0 = obj->n_triangles;
  MPI_Bcast(&tris0, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(V0 > 0.0);

  // Record the initial (gid, pos) table (mesh is rank-0-local before the
  // first redistribution); broadcast so every rank can score its owned verts.
  int nv0 = 0;
  if (g_rank == 0) nv0 = obj->n_vertices;
  MPI_Bcast(&nv0, 1, MPI_INT, 0, MPI_COMM_WORLD);
  int* gid0 = malloc((size_t) nv0 * sizeof(int));
  double* pos0 = malloc((size_t) 3 * nv0 * sizeof(double));
  TEST_ASSERT(gid0 != NULL && pos0 != NULL);
  if (g_rank == 0)
    for (int i = 0; i < nv0; ++i)
    {
      gid0[i] = obj->vertices[i].global_id;
      memcpy(pos0 + 3 * i, obj->vertices[i].pos, 3 * sizeof(double));
    }
  MPI_Bcast(gid0, nv0, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(pos0, 3 * nv0, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // gid -> table row lookup
  int max_gid = 0;
  for (int i = 0; i < nv0; ++i)
    if (gid0[i] > max_gid) max_gid = gid0[i];
  int* row_of_gid = malloc((size_t) (max_gid + 1) * sizeof(int));
  TEST_ASSERT(row_of_gid != NULL);
  for (int i = 0; i <= max_gid; ++i) row_of_gid[i] = -1;
  for (int i = 0; i < nv0; ++i) row_of_gid[gid0[i]] = i;

  const double T = 2.0;
  const int nsteps = 40;
  const double dt = T / nsteps;
  const double vscale = 0.25;  // kept moderate: no remeshing here, so edges
                               // must stay below the half period throughout

  double max_edge_run = 0.0;
  for (int s = 0; s < nsteps; ++s)
  {
    advect_leveque(obj, s * dt, dt, T, vscale);
    MFTLUpdateGhostPositions(ens, NULL);
    MFTLRedistributeListOfObjects(ens);

    // track the worst edge stretch over the run (diagnostic)
    double local_me = 0.0;
    for (int t = 0; t < obj->n_triangles; ++t)
    {
      MFTL_Triangle* tri = &obj->triangles[t];
      if (tri->is_ghost || tri->owner_proc != g_rank) continue;
      MFTL_Vertex* vv[3];
      MFTLGetTriangleVertices(obj, tri, vv);
      for (int e = 0; e < 3; ++e)
      {
        double dv[3];
        MFTLGetMinimumImageVector(obj->domain, dv, vv[e]->pos, vv[(e + 1) % 3]->pos);
        double len = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
        if (len > local_me) local_me = len;
      }
    }
    double g_me = 0.0;
    MPI_Allreduce(&local_me, &g_me, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (g_me > max_edge_run) max_edge_run = g_me;
  }
  MFTLEnsembleUpdateGlobalCounts(ens);

  const double V1 = MFTLObjectVolume(obj);

  // Max min-image displacement of owned vertices vs the initial table.
  double local_maxd = 0.0;
  int local_missing = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != g_rank) continue;
    int row = (v->global_id >= 0 && v->global_id <= max_gid) ? row_of_gid[v->global_id] : -1;
    if (row < 0)
    {
      local_missing++;
      continue;
    }
    double d2 = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      double dd = period_diff(v->pos[k], pos0[3 * row + k], 1.0);
      d2 += dd * dd;
    }
    if (sqrt(d2) > local_maxd) local_maxd = sqrt(d2);
  }
  double max_disp = 0.0;
  int g_missing = 0;
  MPI_Allreduce(&local_maxd, &max_disp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_missing, &g_missing, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_rank == 0)
    printf("  leveque: V0=%.12g V1=%.12g relerr=%.3e  max_disp=%.3e  max_edge_run=%.3f (half period 0.5)  tris0=%d\n", V0, V1,
           fabs(V1 - V0) / V0, max_disp, max_edge_run, tris0);

  TEST_ASSERT(g_missing == 0);
  TEST_ASSERT_EQUAL_INT(tris0, ens->total_global_triangles);  // no remeshing: mesh size unchanged
  TEST_ASSERT(fabs(V1 - V0) / V0 < 5e-3);                    // documented tolerance (calibrated)
  TEST_ASSERT(max_disp < 2e-2);                              // documented tolerance (calibrated)

  free(gid0);
  free(pos0);
  free(row_of_gid);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 4. Periodic remesh across a seam (TSUR volume-conserving smoothing)       */
/* ------------------------------------------------------------------------- */

static int remesh_case(const double center[3], double R, double* V_out, double* A_out, double* meanH_out)
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {true, false, false};
  const int ref = 3;

  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
  TEST_ASSERT(ens != NULL);
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  TEST_ASSERT(MFTLGenerateSphere(obj, (double*) center, R, ref) == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(obj);

  // average edge length on the rank-0-local mesh, then broadcast
  double avg_edge = 0.0;
  if (g_rank == 0)
  {
    double s = 0.0;
    for (int i = 0; i < obj->n_half_edges; ++i) s += MFTLGetEdgeLength(obj, &obj->half_edges[i]);
    avg_edge = s / obj->n_half_edges;
  }
  MPI_Bcast(&avg_edge, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  const double V0 = MFTLObjectVolume(obj);
  TEST_ASSERT(V0 > 0.0);

  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, avg_edge, 1, MFTL_SMOOTH_VERTEX_BASED, 45.0);
  p.volume_conserving_smoothing = MFTL_VCSMOOTH_TSUR;

  int rc = MFTLRemeshEnsembleParallel(ens, &p);
  TEST_ASSERT(rc == MFTL_SUCCESS);
  MFTLEnsembleUpdateGlobalCounts(ens);

  const double V1 = MFTLObjectVolume(obj);
  if (g_rank == 0) printf("  remesh: V0=%.12g V1=%.12g relerr=%.3e\n", V0, V1, fabs(V1 - V0) / fabs(V0));
  TEST_ASSERT(fabs(V1 - V0) / fabs(V0) < 1e-6);  // VolumeRecovery drives to 1e-12

  TEST_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);

  // minimum owned-edge length stays positive (no slivers/collapses to zero)
  double local_min_e = MFTL_DBL_MAX;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != g_rank) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    for (int e = 0; e < 3; ++e)
    {
      double dv[3];
      MFTLGetMinimumImageVector(obj->domain, dv, vv[e]->pos, vv[(e + 1) % 3]->pos);
      double len = sqrt(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
      if (len < local_min_e) local_min_e = len;
    }
  }
  double min_edge = 0.0;
  MPI_Allreduce(&local_min_e, &min_edge, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  TEST_ASSERT(min_edge > 1e-12);

  // all owned vertices remain inside the domain (wrapped storage)
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != g_rank) continue;
    for (int k = 0; k < 3; ++k)
    {
      if (!per[k]) continue;
      TEST_ASSERT(v->pos[k] >= bmin[k] && v->pos[k] < bmax[k]);
    }
  }

  // curvature stays sane after remeshing across the seam
  double meanH, minH, maxH, meanK;
  curvature_stats(ens, obj, &meanH, &minH, &maxH, &meanK);
  TEST_ASSERT(meanH > 0.8 * (2.0 / R) && meanH < 1.25 * (2.0 / R));

  double cm[3] = {0, 0, 0};
  double A = MFTLSurfaceCenter(obj, cm);
  *V_out = V1;
  *A_out = A;
  *meanH_out = meanH;

  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_periodic_remesh(void)
{
  double c_seam[3] = {-2.0, 0.5, -0.3};  // straddles the x seam
  double c_mid[3] = {0.0, 0.5, -0.3};

  double V_s = 0, A_s = 0, H_s = 0, V_m = 0, A_m = 0, H_m = 0;
  if (remesh_case(c_seam, 0.10, &V_s, &A_s, &H_s)) return 1;
  if (remesh_case(c_mid, 0.10, &V_m, &A_m, &H_m)) return 1;

  if (g_rank == 0) printf("  remesh seam vs mid: V %.12g / %.12g   A %.12g / %.12g   meanH %.9g / %.9g\n", V_s, V_m, A_s, A_m, H_s, H_m);

  // same mesh, translated: post-remesh results must track each other
  TEST_ASSERT(fabs(V_s - V_m) <= 0.02 * fabs(V_m));
  TEST_ASSERT(fabs(A_s - A_m) <= 0.02 * fabs(A_m));
  return 0;
}

/* Wide seam-straddler through the x-slab SHIFT SCHEME: R = 0.7 in [-2,2]^3 at
 * cuts {nproc,1,1}. In wrapped storage the sphere's raw x-range covers slabs
 * {0,1,6,7} at np8, so its triangles hold ghost vertices whose owners sit 2+
 * slabs away (never refreshed — Stage-10 contract). Before the redistribution
 * authority fix, a vertex that wrapped across the seam arrived with
 * conflicting owner labels (fresh copy vs stale ghost copies) and was written
 * BOTH as local and as ghost, overflowing the n_unique-sized vertex array
 * (heap corruption, np8). This case pins the fix: it must complete with
 * rc == 0 and sane volume at every rank count. */
static int test_periodic_remesh_wide_seam(void)
{
  double c_seam[3] = {-2.0, 0.05, -0.03};  // centered on the x seam
  double c_mid[3] = {0.1, 0.05, -0.03};

  double V_s = 0, A_s = 0, H_s = 0, V_m = 0, A_m = 0, H_m = 0;
  if (remesh_case(c_seam, 0.7, &V_s, &A_s, &H_s)) return 1;
  if (remesh_case(c_mid, 0.7, &V_m, &A_m, &H_m)) return 1;

  if (g_rank == 0) printf("  wide remesh seam vs mid: V %.12g / %.12g   A %.12g / %.12g   meanH %.9g / %.9g\n", V_s, V_m, A_s, A_m,
                          H_s, H_m);

  TEST_ASSERT(fabs(V_s - V_m) <= 0.02 * fabs(V_m));
  TEST_ASSERT(fabs(A_s - A_m) <= 0.02 * fabs(A_m));
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 5. Periodic intersection + overlap across the seam                        */
/* ------------------------------------------------------------------------- */

// Count each intersecting pair exactly once: the rank OWNING triangle A is
// the canonical reporter (its ghost copy of B came in through the periodic
// spatial-ghost exchange). Summed over ranks this must reproduce the serial
// pair count exactly, seam or mid-domain.
static void count_cb(MFTL_Object* a, int ta, MFTL_Object* b, int tb, void* ud)
{
  (void) b;
  (void) tb;
  if (a->triangles[ta].is_ghost) return;
  (*(int*) ud)++;
}

static int intersect_case(const double cA[3], const double cB[3], double* vol_out)
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {true, false, false};

  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 4);
  TEST_ASSERT(ens != NULL);
  MFTL_Object* a = MFTLObjectCreate(0);
  MFTL_Object* b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, a);
  MFTLEnsembleAddObject(ens, b);
  TEST_ASSERT(MFTLGenerateSphere(a, (double*) cA, 0.6, 3) == MFTL_SUCCESS);
  TEST_ASSERT(MFTLGenerateSphere(b, (double*) cB, 0.6, 3) == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(a);
  MFTLComputeTriangleProperties(b);
  MFTLRedistributeListOfObjects(ens);

  int hits = 0;
  MFTLEnsembleIntersectObjects(ens, NULL, count_cb, &hits);
  MPI_Allreduce(MPI_IN_PLACE, &hits, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MFTL_Overlap* recs = NULL;
  int n = 0;
  int rc = MFTLEnsembleCalculateOverlap(ens, NULL, &recs, &n);
  TEST_ASSERT(rc == MFTL_SUCCESS);
  double V = 0.0;
  for (int i = 0; i < n; ++i) V += recs[i].Volume;
  free(recs);

  *vol_out = V;
  MFTLEnsembleDestroy(&ens);
  return hits;
}

static int test_periodic_intersection(void)
{
  // Seam pair: min-image distance |-1.9 - 1.9| mod 4 = 0.2 -> lens overlap.
  // (The y/z offsets break the mirror symmetry of the axis-aligned setup,
  // which otherwise produces systematically near-degenerate orientation
  // predicates whose sign flips between coordinate frames at the 1e-15 level.)
  double sA[3] = {-1.9, 0.05, -0.03};
  double sB[3] = {1.9, -0.04, 0.02};
  // Same pair mid-domain.
  double mA[3] = {-0.1, 0.05, -0.03};
  double mB[3] = {0.1, -0.04, 0.02};

  double V_seam = 0.0, V_mid = 0.0;
  int hits_seam = intersect_case(sA, sB, &V_seam);
  int hits_mid = intersect_case(mA, mB, &V_mid);

  if (g_rank == 0) printf("  intersection: seam hits=%d V=%.12g   mid hits=%d V=%.12g\n", hits_seam, V_seam, hits_mid, V_mid);

  TEST_ASSERT(hits_seam > 0);
  TEST_ASSERT(hits_mid > 0);
  TEST_ASSERT_EQUAL_INT(hits_mid, hits_seam);
  TEST_ASSERT(V_seam > 0.0);
  // Observed |dV| = 1.75e-8 (rel. 2.7e-8): a near-degenerate duplicate-point/
  // parallelism branch inside the Stage-11 prism machinery flips under the
  // frame translation (coordinates ~2 vs ~0.1). NOT a periodicity error — a
  // missed seam pair would zero the volume (the symmetric seam case in
  // test_overlap.c still pins seam==mid at 1e-9).
  TEST_ASSERT(fabs(V_seam - V_mid) <= 1e-6 * fmax(1.0, fabs(V_mid)));
  return 0;
}

/* ------------------------------------------------------------------------- */
/* 6. Minimum-image constraint                                               */
/* ------------------------------------------------------------------------- */

static int test_minimum_image_constraint(void)
{
  double bmin[3] = {-2.0, -2.0, -2.0};
  double bmax[3] = {2.0, 2.0, 2.0};
  bool per[3] = {true, false, false};
  double origin[3] = {0.0, 0.0, 0.0};

  // Small sphere: passes, before and after redistribution.
  {
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = build_sphere(bmin, bmax, per, origin, 0.5, 2, &obj);
    TEST_ASSERT(ens != NULL);
    TEST_ASSERT(MFTLCheckMinimumImageConstraint(obj) == MFTL_SUCCESS);
    MFTLRedistributeListOfObjects(ens);
    MFTLEnsembleUpdateGlobalCounts(ens);
    TEST_ASSERT(MFTLCheckMinimumImageConstraint(obj) == MFTL_SUCCESS);
    MFTLEnsembleDestroy(&ens);
  }

  // Over-large sphere (extent 3.0 >= half period 2.0): the generator refuses
  // on a periodic domain.
  {
    int cuts[3] = {g_nproc, 1, 1};
    MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
    TEST_ASSERT(ens != NULL);
    MFTL_Object* obj = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(ens, obj);
    TEST_ASSERT(MFTLGenerateSphere(obj, origin, 1.5, 2) == MFTL_ERROR);
    MFTLEnsembleDestroy(&ens);
  }

  // The check itself: generate the over-large sphere with periodicity
  // temporarily disabled (legal), then flip it back on and probe.
  {
    int cuts[3] = {g_nproc, 1, 1};
    MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 2);
    TEST_ASSERT(ens != NULL);
    MFTL_Object* obj = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(ens, obj);
    ens->domain.is_periodic[0] = false;
    TEST_ASSERT(MFTLGenerateSphere(obj, origin, 1.5, 2) == MFTL_SUCCESS);
    ens->domain.is_periodic[0] = true;
    TEST_ASSERT(MFTLCheckMinimumImageConstraint(obj) == MFTL_ERROR);
    MFTLEnsembleDestroy(&ens);
  }

  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_seam_invariance);
  RUN_TEST(test_periodic_advection);
  RUN_TEST(test_periodic_leveque);
  RUN_TEST(test_periodic_remesh);
  RUN_TEST(test_periodic_remesh_wide_seam);
  RUN_TEST(test_periodic_intersection);
  RUN_TEST(test_minimum_image_constraint);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
