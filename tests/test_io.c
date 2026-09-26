/*
 * Stage 05 — mesh I/O: GTS reader/writer + PVTU collection.
 *
 * Covers (per plans/05-mesh-io.md):
 *   - read MFTL1 GTS fixtures (integrity, twin symmetry via integrity, counts)
 *   - read -> MFTLWriteGTS -> read round-trip: identical counts, bit-exact
 *     coordinates, volume preserved to 1e-12
 *   - volume regression on mftl/examples/test1.gts vs the MFTL1 VolumeFace
 *     value (computed with the identical double-precision formula)
 *   - parallel read: every rank owns triangles after redistribution; global
 *     volume rank-independent (np2/np4 via registration)
 *   - PVTU smoke: .pvtu exists, is well-formed XML, references N .vtu pieces
 *   - Stage 16D: open GTS patch across the periodic seams (declared open only)
 *
 * MPI safety: per-mesh checks are rank-0-only (the mesh lives on rank 0 until
 * redistribution) and accumulate into flags; every collective completes
 * before TEST_ASSERT can early-return.
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_io.h"
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef MFTL_GTS_DIR
#define MFTL_GTS_DIR "/home/berend/git/Code/mftl"
#endif

#define GTS_PATH(name) MFTL_GTS_DIR "/" name

static int g_rank = -1;
static int g_nproc = -1;

static MFTL_Ensemble* make_ensemble(double bmin[3], double bmax[3], bool per[3], MFTL_Object** out_obj)
{
  int cuts[3] = {g_nproc, 1, 1};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, bmin, bmax, per, 4);
  if (!ens) return NULL;
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);
  *out_obj = obj;
  return ens;
}

static int dbl_cmp(const void* a, const void* b)
{
  const double* x = *(const double (*)[3]) a;
  const double* y = *(const double (*)[3]) b;
  for (int k = 0; k < 3; ++k)
  {
    if (x[k] < y[k]) return -1;
    if (x[k] > y[k]) return 1;
  }
  return 0;
}

#define CHECK(cond)                                                                                  \
  do                                                                                                 \
  {                                                                                                  \
    if (!(cond))                                                                                     \
    {                                                                                                \
      printf("Rank %d: CHECK failed: (%s) at %s:%d\n", g_rank, #cond, __FILE__, __LINE__);           \
      fail = 1;                                                                                      \
    }                                                                                                \
  } while (0)

int test_read_fixtures()
{
  const char* files[2] = {"meshes/sphere_locally_refined.gts", "meshes/disco.gts"};
  const int expectV[2] = {6728, 162};
  const int expectF[2] = {13452, 320};

  for (int f = 0; f < 2; ++f)
  {
    double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};  // fixtures are small meshes
    bool per[3] = {false, false, false};
    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);

    char path[1200];
    snprintf(path, sizeof(path), "%s/%s", MFTL_GTS_DIR, files[f]);
    int rc = MFTLReadGTS(obj, path);
    TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

    double cm[3] = {0, 0, 0};
    double A = MFTLSurfaceCenter(obj, cm);
    double V = MFTLObjectVolume(obj);

    int fail = 0;
    if (g_rank == 0)
    {
      printf("  %s: V=%.17g A=%.17g\n", files[f], V, A);      CHECK(obj->n_vertices == expectV[f]);
      CHECK(obj->n_triangles == expectF[f]);
      CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
      // twin symmetry is part of the integrity check; closedness too
      CHECK(V > 0.0);  // canonicalized outward winding
    }

    // The read mesh must be immediately redistributable with conserved volume.
    MFTLRedistributeListOfObjects(ens);
    double V_post = MFTLObjectVolume(obj);
    CHECK(fabs(V_post - V) <= 1e-9 * fmax(1e-3, fabs(V)));

    MFTLEnsembleDestroy(&ens);
    TEST_ASSERT(fail == 0);
  }
  return 0;
}

int test_roundtrip()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  const char* src = GTS_PATH("meshes/disco.gts");
  const char* tmp = "io_roundtrip.gts";

  int rc = MFTLReadGTS(obj, src);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

  double V1 = MFTLObjectVolume(obj);

  int fail = 0;
  if (g_rank == 0)
  {
    rc = MFTLWriteGTS(obj, tmp);
    CHECK(rc == MFTL_SUCCESS);
  }

  // Rank 0 must have written the file before rank 0 reads it back; no other
  // rank participates. A barrier keeps the file visible before rank-0 reads.
  MPI_Barrier(MPI_COMM_WORLD);

  MFTL_Object* obj2 = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(ens, obj2);
  rc = MFTLReadGTS(obj2, tmp);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

  double V2 = MFTLObjectVolume(obj2);

  if (g_rank == 0)
  {
    printf("  roundtrip: V1=%.17g V2=%.17g (diff %.3e)\n", V1, V2, fabs(V2 - V1));
    CHECK(obj->n_vertices == obj2->n_vertices);
    CHECK(obj->n_triangles == obj2->n_triangles);

    // Coordinates must be BIT-EXACT (the writer prints %.17g); the writer
    // renumbers vertices, so compare sorted position multisets.
    int n = obj->n_vertices;
    double (*p1)[3] = malloc((size_t) n * sizeof(*p1));
    double (*p2)[3] = malloc((size_t) n * sizeof(*p2));
    for (int i = 0; i < n; ++i)
    {
      memcpy(p1[i], obj->vertices[i].pos, 3 * sizeof(double));
      memcpy(p2[i], obj2->vertices[i].pos, 3 * sizeof(double));
    }
    qsort(p1, n, sizeof(*p1), dbl_cmp);
    qsort(p2, n, sizeof(*p2), dbl_cmp);
    CHECK(memcmp(p1, p2, (size_t) n * sizeof(*p1)) == 0);
    free(p1);
    free(p2);
  }

  CHECK(fabs(V2 - V1) <= 1e-12 * fmax(1e-3, fabs(V1)));

  MFTLEnsembleDestroy(&ens);
  if (g_rank == 0) remove(tmp);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_volume_regression_test1()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  int rc = MFTLReadGTS(obj, GTS_PATH("examples/test1.gts"));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

  double V = MFTLObjectVolume(obj);

  // MFTL1's value for examples/test1.gts: produced by running the prebuilt
  // MFTL1 library (mftl/lib/libMFTL.a) with MFTLGTSFileRead + MFTLObjectVolume
  // on this exact fixture: V = 0.0020007879660460201 (centre (0.499, 0.75,
  // 0.5), area 0.080971471135152573). Note: the file's stored edge orientations
  // are arbitrary; MFTL2's reader canonicalizes the winding consistently
  // (plans/05: "inconsistent winding -> canonicalize"), yielding the same
  // consistently-oriented surface as MFTL1's reader.
  const double V_mftl1 = 0.0020007879660460201;

  int fail = 0;
  if (g_rank == 0) printf("  test1.gts: V=%.17g  MFTL1 ref |V|=%.17g\n", V, V_mftl1);
  CHECK(V > 0.0);  // reader canonicalizes the winding
  CHECK(fabs(V - V_mftl1) <= 1e-9 * V_mftl1);

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_parallel_read()
{
  // The fixture spans exactly [0.15, 0.45]^3: match the domain to it so every
  // x-cut slice contains mesh and each rank owns triangles at any np.
  double bmin[3] = {0.15, 0.15, 0.15}, bmax[3] = {0.45, 0.45, 0.45};
  bool per[3] = {false, false, false};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  int rc = MFTLReadGTS(obj, GTS_PATH("meshes/sphere_locally_refined.gts"));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, rc);

  double V_serial = MFTLObjectVolume(obj);

  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  double V_parallel = MFTLObjectVolume(obj);

  // Every rank must own at least one triangle.
  int owned = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (!obj->triangles[t].is_ghost && obj->triangles[t].owner_proc == g_rank) owned++;
  int min_owned = 0;
  MPI_Allreduce(&owned, &min_owned, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  if (g_rank == 0) printf("  nproc=%d  V_serial=%.17g V_parallel=%.17g  min owned tris=%d\n", g_nproc, V_serial, V_parallel, min_owned);

  int fail = 0;
  CHECK(min_owned > 0);
  CHECK(fabs(V_parallel - V_serial) <= 1e-9 * fmax(1e-3, fabs(V_serial)));

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_pvtu()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistributeListOfObjects(ens);

  const char* stem = "io_smoke";
  MFTLWritePvtu(ens, stem);
  MPI_Barrier(MPI_COMM_WORLD);

  int fail = 0;
  if (g_rank == 0)
  {
    char path[1100];
    snprintf(path, sizeof(path), "%s.pvtu", stem);
    FILE* fp = fopen(path, "r");
    CHECK(fp != NULL);
    if (fp)
    {
      char buf[4096];
      size_t total = fread(buf, 1, sizeof(buf) - 1, fp);
      buf[total] = '\0';
      fclose(fp);

      CHECK(strstr(buf, "<VTKFile type=\"PUnstructuredGrid\"") != NULL);
      CHECK(strstr(buf, "</VTKFile>") != NULL);

      // Piece count must equal the rank count.
      int pieces = 0;
      for (const char* p = buf; (p = strstr(p, "<Piece Source=")) != NULL; ++p) pieces++;
      CHECK(pieces == g_nproc);

      // Every referenced piece file must exist and be a well-formed VTU.
      for (int r = 0; r < g_nproc; ++r)
      {
        char piece[1100];
        snprintf(piece, sizeof(piece), "%s_%05d.vtu", stem, r);
        FILE* pf = fopen(piece, "r");
        CHECK(pf != NULL);
        if (pf)
        {
          char pbuf[1024];
          size_t n = fread(pbuf, 1, sizeof(pbuf) - 1, pf);
          pbuf[n] = '\0';
          fclose(pf);
          CHECK(strstr(pbuf, "<VTKFile type=\"UnstructuredGrid\"") != NULL);
        }
        remove(piece);
      }
      remove(path);
    }
    if (fail == 0) printf("  pvtu well-formed: %d pieces referenced and present\n", g_nproc);
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}

int test_write_rejects_distributed_surface()
{
  double bmin[3] = {-2, -2, -2}, bmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  double origin[3] = {0, 0, 0};
  MFTL_Object* obj = NULL;
  MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
  TEST_ASSERT(ens != NULL);

  MFTLGenerateSphere(obj, origin, 1.0, 3);
  MFTLComputeTriangleProperties(obj);

  const char* tmp = "io_partial_reject.gts";
  int fail = 0;

  // Sanity: the serial (pre-redistribution) write succeeds.
  if (g_rank == 0)
  {
    int rc = MFTLWriteGTS(obj, tmp);
    CHECK(rc == MFTL_SUCCESS);
    remove(tmp);
  }

  // After redistribution rank 0 owns only a slice: the write must FAIL with
  // a clear error instead of silently emitting a partial surface. (At np1
  // redistribution is a no-op and the surface stays whole — nothing to
  // reject; the positive path above already ran.)
  MFTLRedistributeListOfObjects(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  if (g_rank == 0 && g_nproc > 1)
  {
    int rc = MFTLWriteGTS(obj, tmp);
    if (rc != MFTL_ERROR)
    {
      printf("Rank 0: expected MFTL_ERROR from MFTLWriteGTS on a distributed surface, got %d\n", rc);
      fail = 1;
      remove(tmp);
    }
    FILE* fp = fopen(tmp, "r");
    if (fp != NULL)
    {
      printf("Rank 0: partial GTS file was created despite the error\n");
      fclose(fp);
      remove(tmp);
      fail = 1;
    }
    if (fail == 0) printf("  write-after-redistribute correctly rejected, no file written\n");
  }

  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(fail == 0);
  return 0;
}


/* Stage 16D: an open GTS patch (8 x 8 quads split in two) crossing the x and y seams of the
   periodic box [0,2]^3, in continuous coordinates. Rejected on a body not declared open;
   read on a declared-open body with its file orientation (+z normals) and one boundary loop. */
static void write_open_patch(const char* path, int flip)
{
  const int n = 9;
  FILE* f = fopen(path, "w");
  int nE = 0, (*E)[2] = malloc(sizeof(int[2]) * 3 * n * n), (*F)[3] = malloc(sizeof(int[3]) * 2 * n * n), nF = 0;
  for (int j = 0; j < n - 1; ++j)
    for (int i = 0; i < n - 1; ++i)
    {
      const int v0 = i + j * n, v1 = v0 + 1, v2 = v1 + n, v3 = v0 + n;
      const int tri[2][3] = {{v0, v1, v2}, {v0, v2, v3}};
      for (int q = 0; q < 2; ++q)
      {
        int tv[3] = {tri[q][0], flip ? tri[q][2] : tri[q][1], flip ? tri[q][1] : tri[q][2]};
        for (int k = 0; k < 3; ++k)
        {
          const int a = tv[k], b = tv[(k + 1) % 3];
          int e = -1;
          for (int m = 0; m < nE; ++m)
            if ((E[m][0] == a && E[m][1] == b) || (E[m][0] == b && E[m][1] == a)) e = m;
          if (e < 0)
          {
            E[nE][0] = a;
            E[nE][1] = b;
            e = nE++;
          }
          F[nF][k] = e;
        }
        nF++;
      }
    }
  fprintf(f, "%d %d %d GtsSurface GtsFace GtsEdge GtsVertex\n", n * n, nE, nF);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) fprintf(f, "%.17g %.17g %.17g\n", 1.6 + 0.1 * i, 1.6 + 0.1 * j, 1.0 + 0.02 * sin(0.7 * i + 0.3 * j));
  for (int m = 0; m < nE; ++m) fprintf(f, "%d %d\n", E[m][0] + 1, E[m][1] + 1);
  for (int m = 0; m < nF; ++m) fprintf(f, "%d %d %d\n", F[m][0] + 1, F[m][1] + 1, F[m][2] + 1);
  fclose(f);
  free(E);
  free(F);
}

int test_read_open_gts()
{
  double bmin[3] = {0, 0, 0}, bmax[3] = {2, 2, 2};
  bool per[3] = {true, true, true};
  char path[256];
  snprintf(path, sizeof(path), "test_io_open_patch_np%d.gts", g_nproc);
  int fail = 0;
  for (int flip = 0; flip < 2; ++flip)
  {
    if (g_rank == 0) write_open_patch(path, flip);
    MPI_Barrier(MPI_COMM_WORLD);

    MFTL_Object* obj = NULL;
    MFTL_Ensemble* ens = make_ensemble(bmin, bmax, per, &obj);
    TEST_ASSERT(ens != NULL);
    CHECK(MFTLReadGTS(obj, path) == MFTL_ERROR);  // undeclared: boundary edges rejected
    MFTLEnsembleDestroy(&ens);

    ens = make_ensemble(bmin, bmax, per, &obj);
    MFTLObjectDeclareOpen(obj, 1);
    const int rc = MFTLReadGTS(obj, path);
    CHECK(rc == MFTL_SUCCESS);
    if (rc == MFTL_SUCCESS)
    {
      MFTLRedistributeListOfObjects(ens);
      MFTLEnsembleUpdateGlobalCounts(ens);
      CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
      int wrong = 0, owned = 0, outside = 0;
      for (int t = 0; t < obj->n_triangles; ++t)
      {
        const MFTL_Triangle* tri = &obj->triangles[t];
        if (tri->is_ghost || tri->owner_proc != g_rank) continue;
        owned++;
        MFTL_Vertex* vv[3];
        MFTLGetTriangleVertices(obj, tri, vv);
        double a[3], b[3];
        MFTLGetMinimumImageVector(obj->domain, a, vv[0]->pos, vv[1]->pos);
        MFTLGetMinimumImageVector(obj->domain, b, vv[0]->pos, vv[2]->pos);
        const double nz = a[0] * b[1] - a[1] * b[0];
        if ((nz > 0.0) != (flip == 0)) wrong++;  // the file orientation is kept
      }
      for (int v = 0; v < obj->n_vertices; ++v)
        for (int k = 0; k < 3; ++k) outside += obj->vertices[v].pos[k] < 0.0 || obj->vertices[v].pos[k] >= 2.0;
      int sums[3] = {wrong, owned, outside}, tot[3];
      MPI_Allreduce(sums, tot, 3, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      MFTL_BoundaryLoopSet L;
      CHECK(MFTLBoundaryLoops(obj, &L) == MFTL_SUCCESS);
      if (g_rank == 0)
        printf("  flip=%d: %d triangles, wrong orientation %d, positions outside %d, %d loop(s) of %d edges, length %.15g\n", flip, tot[1], tot[0],
               tot[2], L.n_loops, L.n_loops ? L.loops[0].n : 0, L.n_loops ? L.loops[0].length : 0.0);
      CHECK(tot[1] == 128 && tot[0] == 0 && tot[2] == 0);
      CHECK(L.n_loops == 1 && L.loops[0].n == 32 && L.loops[0].winding[0] == 0 && L.loops[0].winding[1] == 0);
      MFTLFreeBoundaryLoops(&L);
    }
    MFTLEnsembleDestroy(&ens);
  }
  if (g_rank == 0) remove(path);
  TEST_ASSERT(fail == 0);
  return 0;
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  RUN_TEST(test_read_fixtures);
  RUN_TEST(test_roundtrip);
  RUN_TEST(test_volume_regression_test1);
  RUN_TEST(test_parallel_read);
  RUN_TEST(test_write_rejects_distributed_surface);
  RUN_TEST(test_pvtu);
  RUN_TEST(test_read_open_gts);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
