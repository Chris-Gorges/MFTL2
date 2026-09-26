/*
 * Stage 06 — vertex normals, curvatures, roughness.
 *
 * Covers (per plans/06-normals-curvature-roughness.md):
 *   - unit sphere ref4/5: mean curvature -> 2/R, Gaussian -> 2/R^2 at every
 *     vertex (tolerance decreasing with refinement; limiting case)
 *   - planar grid: interior vertices mean ~ 0 and Gaussian ~ 0
 *   - cylinder lateral surface: interior side vertices mean ~ 1/R, Gaussian ~ 0
 *   - rank-independence: owned-vertex curvature identical to the serial
 *     reference on 1/2/4/8 ranks (tol 1e-7)
 *   - roughness: smooth sphere ~ 0; a radially perturbed vertex spikes
 *
 * MPI safety: normals/curvatures are pure local passes; the global geometric
 * Laplacian in MFTLComputeRoughness contains an Allreduce and is called by
 * every rank. Rank-0-only checks accumulate into flags; all collectives
 * (extension, ghost sync, Gatherv, destroy) complete before TEST_ASSERT.
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

#define CHECK(cond)                                                                              \
  do                                                                                             \
  {                                                                                              \
    if (!(cond))                                                                                 \
    {                                                                                            \
      printf("Rank %d: CHECK failed: (%s) at %s:%d\n", g_rank, #cond, __FILE__, __LINE__);       \
      fail = 1;                                                                                  \
    }                                                                                            \
  } while (0)

int test_sphere_curvature()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};

  double err_mean4 = 0.0, err_gauss4 = 0.0;
  double gl_prev = 0.0;

  for (int pass = 0; pass < 3; ++pass)
  {
    int ref = pass == 0 ? 3 : (pass == 1 ? 4 : 5);
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    MFTLGenerateSphere(obj, origin, 1.0, ref);
    MFTLComputeTriangleProperties(obj);
    MFTLComputeVertexNormals(obj);
    MFTLComputeVertexCurvatures(obj);
    double gl = MFTLComputeRoughness(obj);  // collective

    int fail = 0;
    if (g_rank == 0)
    {
      double max_mean = 0.0, max_gauss = 0.0, max_dih = 0.0;
      int nan_found = 0;
      for (int i = 0; i < obj->n_vertices; ++i)
      {
        const MFTL_Vertex* v = &obj->vertices[i];
        double em = fabs(v->meancurvature - 2.0);
        double eg = fabs(v->gaussiancurvature - 2.0);
        if (em > max_mean) max_mean = em;
        if (eg > max_gauss) max_gauss = eg;
        if (v->roughness_mean > max_dih) max_dih = v->roughness_mean;
        if (!(v->meancurvature == v->meancurvature) || !(v->gaussiancurvature == v->gaussiancurvature)) nan_found = 1;
      }
      printf("  ref %d: max |mean-2|=%.3e  max |gauss-2|=%.3e  GL=%.3e  max dihedral=%.3e\n", ref, max_mean, max_gauss, gl, max_dih);
      CHECK(!nan_found);

      // The chord sphere is planar per facet: roughness is O(h^2) and must
      // decrease with refinement (the "smooth sphere" limiting case).
      CHECK(gl < 5e-2);
      CHECK(max_dih < 5e-2);
      if (pass > 0) CHECK(gl < gl_prev);
      gl_prev = gl;

      if (ref == 4)
      {
        err_mean4 = max_mean;
        err_gauss4 = max_gauss;
        CHECK(max_mean < 5e-3);
        CHECK(max_gauss < 5e-2);
      }
      else if (ref == 5)
      {
        CHECK(max_mean < 2e-3);
        CHECK(max_gauss < 2e-2);
        CHECK(max_mean < err_mean4);    // curvatures converge with refinement
        CHECK(max_gauss < err_gauss4);
      }
    }

    MFTLEnsembleDestroy(&ens);
    TEST_ASSERT(fail == 0);
  }
  return 0;
}

int test_flat_patch()
{
  double bmin[3] = {-1, -1, -1}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  // 5x5 planar grid, z=0, quads split along one diagonal.
  const int N = 5;
  double (*pos)[3] = malloc((size_t)(N * N) * sizeof(*pos));
  int (*triples)[3] = malloc((size_t)(2 * (N - 1) * (N - 1)) * sizeof(*triples));
  int t = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      pos[j * N + i][0] = (double) i;
      pos[j * N + i][1] = (double) j;
      pos[j * N + i][2] = 0.0;
    }
  for (int j = 0; j < N - 1; ++j)
    for (int i = 0; i < N - 1; ++i)
    {
      int a = j * N + i, b = j * N + i + 1, c = (j + 1) * N + i + 1, d = (j + 1) * N + i;
      triples[t][0] = a;
      triples[t][1] = b;
      triples[t][2] = c;
      t++;
      triples[t][0] = a;
      triples[t][1] = c;
      triples[t][2] = d;
      t++;
    }
  MFTLObjectDeclareOpen(obj, 1);  // Stage 16A: an open surface
  if (g_rank == 0) MFTLBuildTopologyFromTriples(obj, N * N, (const double (*)[3]) pos, t, (const int (*)[3]) triples);
  free(pos);
  free(triples);

  MFTLComputeTriangleProperties(obj);
  MFTLComputeVertexCurvatures(obj);
  double gl_flat = MFTLComputeRoughness(obj);  // collective

  int fail = 0;
  if (g_rank == 0)
  {
    // interior vertices: full ring, planar -> LB vector 0 and angle sum 2*pi
    for (int j = 1; j < N - 1; ++j)
      for (int i = 1; i < N - 1; ++i)
      {
        const MFTL_Vertex* v = &obj->vertices[j * N + i];
        CHECK(fabs(v->meancurvature) < 1e-12);
        CHECK(fabs(v->gaussiancurvature) < 1e-12);
      }
    // a genuinely smooth (planar) surface has exactly zero roughness
    double max_dih = 0.0;
    for (int i = 0; i < obj->n_vertices; ++i)
      if (obj->vertices[i].roughness_mean > max_dih) max_dih = obj->vertices[i].roughness_mean;
    printf("  planar grid: GL=%.3e  max dihedral=%.3e (interior curvatures ~ 0)\n", gl_flat, max_dih);
    CHECK(gl_flat < 1e-14);
    CHECK(max_dih < 1e-14);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_cylinder_curvature()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  // r = 1: lateral mean curvature k1+k2 = 1/R = 1, gaussian 2*k1*k2 = 0.
  MFTLGenerateCylinder(obj, 0.005, 1.0, 1.0, 0.0);
  MFTLComputeTriangleProperties(obj);
  MFTLComputeVertexCurvatures(obj);

  int fail = 0;
  if (g_rank == 0)
  {
    double max_mean = 0.0, max_gauss = 0.0, max_area = 0.0;
    int checked = 0;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      const MFTL_Vertex* v = &obj->vertices[i];
      if (v->pos[2] < 0.1 || v->pos[2] > 0.9) continue;  // away from the caps
      double em = fabs(v->meancurvature - 1.0);
      double eg = fabs(v->gaussiancurvature);
      if (em > max_mean) max_mean = em;
      if (eg > max_gauss) max_gauss = eg;
      if (v->area > max_area) max_area = v->area;
      checked++;
    }
    printf("  cylinder interior: n=%d  max |mean-1|=%.3e  max |gauss|=%.3e\n", checked, max_mean, max_gauss);
    CHECK(checked > 0);
    CHECK(max_mean < 5e-3);
    CHECK(max_gauss < 5e-3);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_rank_independence()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};

  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 4);
  MFTLComputeTriangleProperties(obj);

  // Serial reference on the rank-0-local closed mesh.
  int n_ref = 0;
  double *ref_mean = NULL, *ref_gauss = NULL, *ref_normal = NULL;
  int* ref_gid = NULL;
  if (g_rank == 0)
  {
    MFTLComputeVertexNormals(obj);
    MFTLComputeVertexCurvatures(obj);
    n_ref = obj->n_vertices;
    ref_mean = malloc((size_t) n_ref * sizeof(double));
    ref_gauss = malloc((size_t) n_ref * sizeof(double));
    ref_normal = malloc((size_t) 3 * n_ref * sizeof(double));
    ref_gid = malloc((size_t) n_ref * sizeof(int));
    for (int i = 0; i < n_ref; ++i)
    {
      ref_gid[i] = obj->vertices[i].global_id;
      ref_mean[i] = obj->vertices[i].meancurvature;
      ref_gauss[i] = obj->vertices[i].gaussiancurvature;
      memcpy(&ref_normal[3 * i], obj->vertices[i].normal, 3 * sizeof(double));
    }
  }

  // Distribute; guarantee complete one-rings; sync ghost positions.
  MFTLRedistributeListOfObjects(ens);
  MFTLExtendGhostLayer(ens, 2);
  MFTLUpdateGhostPositions(ens, NULL);
  MFTLComputeTriangleProperties(obj);
  MFTLComputeVertexNormals(obj);
  MFTLComputeVertexCurvatures(obj);

  // Ship owned (gid, mean, gauss, normal) to rank 0.
  int owned = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
    if (!obj->vertices[i].is_ghost && obj->vertices[i].owner_proc == g_rank) owned++;

  double* vals = malloc((size_t) owned * 6 * sizeof(double));
  int* gids = malloc((size_t) owned * sizeof(int));
  int k = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != g_rank) continue;
    gids[k] = v->global_id;
    vals[6 * k + 0] = v->meancurvature;
    vals[6 * k + 1] = v->gaussiancurvature;
    vals[6 * k + 2] = v->normal[0];
    vals[6 * k + 3] = v->normal[1];
    vals[6 * k + 4] = v->normal[2];
    vals[6 * k + 5] = v->area;
    k++;
  }

  int* gid_counts = NULL;
  int* gid_displs = NULL;
  int* dbl_counts = NULL;
  int* dbl_displs = NULL;
  double* all_vals = NULL;
  int* all_gids = NULL;
  if (g_rank == 0)
  {
    gid_counts = malloc((size_t) g_nproc * sizeof(int));
    gid_displs = malloc((size_t) g_nproc * sizeof(int));
    dbl_counts = malloc((size_t) g_nproc * sizeof(int));
    dbl_displs = malloc((size_t) g_nproc * sizeof(int));
  }
  MPI_Gather(&owned, 1, MPI_INT, gid_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (g_rank == 0)
  {
    int acc = 0;
    for (int r = 0; r < g_nproc; ++r)
    {
      gid_displs[r] = acc;
      dbl_displs[r] = 6 * acc;
      gid_counts[r] = gid_counts[r];      // per-record
      dbl_counts[r] = gid_counts[r] * 6;  // 6 doubles per record
      acc += gid_counts[r];
    }
    total = acc;
    all_vals = malloc((size_t) total * 6 * sizeof(double));
    all_gids = malloc((size_t) total * sizeof(int));
  }
  MPI_Gatherv(vals, owned * 6, MPI_DOUBLE, all_vals, dbl_counts, dbl_displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(gids, owned, MPI_INT, all_gids, gid_counts, gid_displs, MPI_INT, 0, MPI_COMM_WORLD);

  int fail = 0;
  if (g_rank == 0)
  {
    // reference lookup by gid
    double max_dmean = 0.0, max_dgauss = 0.0, max_dnormal = 0.0;
    int missing = 0;
    for (int r = 0; r < total; ++r)
    {
      int gid = all_gids[r];
      int found = -1;
      for (int q = 0; q < n_ref; ++q)
        if (ref_gid[q] == gid)
        {
          found = q;
          break;
        }
      if (found < 0)
      {
        missing++;
        continue;
      }
      double dm = fabs(all_vals[6 * r + 0] - ref_mean[found]);
      double dg = fabs(all_vals[6 * r + 1] - ref_gauss[found]);
      double dn = fabs(all_vals[6 * r + 2] - ref_normal[3 * found + 0]) + fabs(all_vals[6 * r + 3] - ref_normal[3 * found + 1]) +
                  fabs(all_vals[6 * r + 4] - ref_normal[3 * found + 2]);
      if (dm > max_dmean) max_dmean = dm;
      if (dg > max_dgauss) max_dgauss = dg;
      if (dn > max_dnormal) max_dnormal = dn;
    }
    printf("  nproc=%d  max |dmean|=%.3e  max |dgauss|=%.3e  max |dnormal|=%.3e  missing=%d\n", g_nproc, max_dmean, max_dgauss, max_dnormal,
           missing);
    CHECK(missing == 0);
    CHECK(max_dmean < 1e-7);
    CHECK(max_dgauss < 1e-7);
    CHECK(max_dnormal < 1e-7);
  }

  free(vals);
  free(gids);
  free(ref_mean);
  free(ref_gauss);
  free(ref_normal);
  free(ref_gid);
  free(gid_counts);
  free(gid_displs);
  free(dbl_counts);
  free(dbl_displs);
  free(all_vals);
  free(all_gids);

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_roughness()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);
  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLComputeTriangleProperties(obj);

  double gl_smooth = MFTLComputeRoughness(obj);  // collective

  // Perturb one owned vertex radially (rank 0, pre-redistribution).
  int perturbed_gid = -1;
  if (g_rank == 0)
  {
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      double len = sqrt(v->pos[0] * v->pos[0] + v->pos[1] * v->pos[1] + v->pos[2] * v->pos[2]);
      if (len > 0.5)
      {
        v->pos[0] *= 1.25;
        v->pos[1] *= 1.25;
        v->pos[2] *= 1.25;
        perturbed_gid = v->global_id;
        break;
      }
    }
  }
  MPI_Bcast(&perturbed_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MFTLComputeTriangleProperties(obj);
  double gl_rough = MFTLComputeRoughness(obj);  // collective

  int fail = 0;
  if (g_rank == 0)
  {
    double perturbed_dih = -1.0;
    for (int i = 0; i < obj->n_vertices; ++i)
      if (obj->vertices[i].global_id == perturbed_gid) perturbed_dih = obj->vertices[i].roughness_mean;
    printf("  sphere ref3: GL smooth=%.3e -> perturbed=%.3e  perturbed-vertex dihedral mean=%.3e\n", gl_smooth, gl_rough, perturbed_dih);
    CHECK(gl_rough > gl_smooth + 1e-6);
    CHECK(perturbed_dih > 1e-3);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_sphere_curvature);
  RUN_TEST(test_flat_patch);
  RUN_TEST(test_cylinder_curvature);
  RUN_TEST(test_rank_independence);
  RUN_TEST(test_roughness);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
