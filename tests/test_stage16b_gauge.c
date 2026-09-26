/*
 * Stage 16B — independent-review tests (plans/16 §21): lattice-gauge invariance and winding
 * validity. Every CHECK is collective. Registered at 1, 2, 4 and 8 ranks.
 *   gauge    : rotation about a pivot, non-uniform scaling and translation of one seam-crossing
 *              body in three lattice gauges (img shifted by 0, (3,-2,1), (1e6,-1e6,7)): identical
 *              physical result (§3.2: transforms relative to the body reference point);
 *   winding  : bodies built by the public serial builder MFTLBuildTopologyFromTriples (never
 *              canonicalised): a seam-crossing sphere gets the right volume, a wrapping film is
 *              never taken as compact (MFTLObjectEnsureWinding);
 *   film     : transforms of a wrapping film — lattice-preserving 90-degree rotation and z-scaling
 *              accepted, rotations/scalings that break the winding lattice rejected;
 *   clear    : stale flags after MFTLObjectClear; a native v1 (legacy) file of a seam-crossing
 *              body read on this rank count;
 *   phase    : analytic slab-with-bubble volume/centre/inside with members re-gauged by random
 *              lattice vectors; matrix phase; inconsistent disjoint members.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_images.h"
#include "mftl_phase_set.h"
#include "mftl_topology.h"
#include "mftl_io.h"
#include "mftl_remeshing.h"

static int g_rank, g_np, g_fail, g_chk;
#define CHECK(c)                                                                           \
  do                                                                                       \
  {                                                                                        \
    int _l = (c) ? 0 : 1, _g;                                                              \
    MPI_Allreduce(&_l, &_g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);                          \
    g_chk++;                                                                               \
    if (_g)                                                                                \
    {                                                                                      \
      if (g_rank == 0) fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #c, __FILE__, __LINE__); \
      g_fail++;                                                                            \
    }                                                                                      \
  } while (0)

static MFTL_Ensemble* ens_new(void)
{
  int g[3] = {g_np, 1, 1};
  if (g_np == 4) g[0] = 2, g[1] = 2;
  if (g_np == 8) g[0] = 2, g[1] = 2, g[2] = 2;
  double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  bool per[3] = {true, true, true};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, g, mn, mx, per, 4);
}

/* UV sphere (closed, outward) with continuous vertex coordinates */
static void uv_sphere(const double c[3], double r, int n, int* nv, double (**P)[3], int* nt, int (**T)[3])
{
  const int m = 2 * n;
  *nv = 2 + (n - 1) * m;
  *P = malloc((size_t) *nv * sizeof(**P));
  *T = malloc((size_t) (2 * m * (n - 1)) * sizeof(**T));
  (*P)[0][0] = c[0], (*P)[0][1] = c[1], (*P)[0][2] = c[2] + r;
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < m; ++j)
    {
      double th = M_PI * i / n, ph = 2 * M_PI * j / m;
      double* q = (*P)[1 + (i - 1) * m + j];
      q[0] = c[0] + r * sin(th) * cos(ph), q[1] = c[1] + r * sin(th) * sin(ph), q[2] = c[2] + r * cos(th);
    }
  const int S = *nv - 1;
  (*P)[S][0] = c[0], (*P)[S][1] = c[1], (*P)[S][2] = c[2] - r;
#define ID(i, j) (1 + ((i) - 1) * m + ((j) % m))
  int k = 0;
  for (int j = 0; j < m; ++j) (*T)[k][0] = 0, (*T)[k][1] = ID(1, j), (*T)[k][2] = ID(1, j + 1), k++;
  for (int i = 1; i < n - 1; ++i)
    for (int j = 0; j < m; ++j)
    {
      int a = ID(i, j), b = ID(i, j + 1), cc = ID(i + 1, j + 1), d = ID(i + 1, j);
      (*T)[k][0] = a, (*T)[k][1] = d, (*T)[k][2] = cc, k++;
      (*T)[k][0] = a, (*T)[k][1] = cc, (*T)[k][2] = b, k++;
    }
  for (int j = 0; j < m; ++j) (*T)[k][0] = ID(n - 1, j), (*T)[k][1] = S, (*T)[k][2] = ID(n - 1, j + 1), k++;
#undef ID
  *nt = k;
}

/* closed film z = z0 + a sin(2 pi x) wrapping x and y (a torus-topology sheet), N x N */
static void film(double z0, double a, int N, int* nv, double (**P)[3], int* nt, int (**T)[3])
{
  *nv = N * N;
  *P = malloc((size_t) *nv * sizeof(**P));
  *T = malloc((size_t) (2 * N * N) * sizeof(**T));
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      double* q = (*P)[j * N + i];
      q[0] = (double) i / N, q[1] = (double) j / N, q[2] = z0 + a * sin(2 * M_PI * q[0]);
    }
  int k = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      int p = j * N + i, q = j * N + (i + 1) % N, r = ((j + 1) % N) * N + (i + 1) % N, s = ((j + 1) % N) * N + i;
      (*T)[k][0] = p, (*T)[k][1] = q, (*T)[k][2] = r, k++;
      (*T)[k][0] = p, (*T)[k][1] = r, (*T)[k][2] = s, k++;
    }
  *nt = k;
}

/* serial builder on rank 0 with WRAPPED positions (the documented input of the function) */
static MFTL_Object* serial_body(MFTL_Ensemble* ens, int gid, int nv, double (*P)[3], int nt, int (*T)[3])
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  int rc = MFTL_SUCCESS;
  if (g_rank == 0)
  {
    double(*W)[3] = malloc((size_t) nv * sizeof(*W));
    for (int i = 0; i < nv; ++i)
    {
      memcpy(W[i], P[i], sizeof(W[i]));
      MFTLApplyPeriodicity(&ens->domain, W[i]);
    }
    rc = MFTLBuildTopologyFromTriples(o, nv, (const double(*)[3]) W, nt, (const int(*)[3]) T);
    free(W);
  }
  MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
  CHECK(rc == MFTL_SUCCESS);
  MPI_Bcast(&o->next_vertex_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&o->next_triangle_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&o->next_halfedge_gid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  return o;
}

static double mesh_volume(int nt, double (*P)[3], int (*T)[3])
{
  double v = 0;
  for (int t = 0; t < nt; ++t)
  {
    double *a = P[T[t][0]], *b = P[T[t][1]], *c = P[T[t][2]];
    v += a[0] * (b[1] * c[2] - b[2] * c[1]) + b[0] * (c[1] * a[2] - c[2] * a[1]) + c[0] * (a[1] * b[2] - a[2] * b[1]);
  }
  return v / 6.0;
}

static void test_winding(void)
{
  // (1) compact seam-crossing sphere through the serial builder (never canonicalised)
  MFTL_Ensemble* ens = ens_new();
  int nv, nt;
  double(*P)[3];
  int(*T)[3];
  double c[3] = {0.97, 0.02, 0.5};
  uv_sphere(c, 0.2, 16, &nv, &P, &nt, &T);
  const double Vexact = mesh_volume(nt, P, T);
  MFTL_Object* o = serial_body(ens, 1, nv, P, nt, T);
  double V = MFTLObjectVolume(o);
  if (g_rank == 0) printf("  serial-builder seam sphere: volume %.15g (mesh %.15g), valid %d\n", V, Vexact, o->wrap_valid);
  CHECK(fabs(V - Vexact) < 1e-12);
  int w[3];
  CHECK(MFTLObjectWrapFlags(o, w) == MFTL_SUCCESS && !w[0] && !w[1] && !w[2]);
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  double Vps = 0;
  int st = MFTLPhaseSetVolume(S, &Vps);
  if (g_rank == 0) printf("  serial-builder seam sphere: phase set status %d V %.15g\n", st, Vps);
  CHECK(st == MFTL_DEFINED && fabs(Vps - Vexact) < 1e-12);
  MFTLPhaseSetDestroy(&S);
  free(P), free(T);
  MFTLEnsembleDestroy(&ens);

  // (2) film wrapping x and y through the serial builder: must never be taken as compact
  ens = ens_new();
  film(0.4, 0.05, 16, &nv, &P, &nt, &T);
  o = serial_body(ens, 2, nv, P, nt, T);
  int compact_before = MFTLObjectIsCompact(o), hasvol_before = MFTLObjectHasVolume(o);
  if (g_rank == 0) printf("  serial-builder film: IsCompact %d HasVolume %d (valid %d)\n", compact_before, hasvol_before, o->wrap_valid);
  CHECK(!compact_before && !hasvol_before);
  S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  st = MFTLPhaseSetVolume(S, &Vps);
  if (g_rank == 0) printf("  serial-builder film: phase set status %d\n", st);
  CHECK(st == MFTL_UNDEFINED);  // Phi_z = +-A
  MFTLPhaseSetDestroy(&S);
  free(P), free(T);
  MFTLEnsembleDestroy(&ens);
}

/* gather (gid -> wrapped pos) of owned vertices */
static int gather_pos(MFTL_Object* o, double** out)
{
  int n = 0;
  for (int i = 0; i < o->n_vertices; ++i)
    if (!o->vertices[i].is_ghost) n++;
  double* loc = malloc((size_t) (4 * (n > 0 ? n : 1)) * sizeof(double));
  n = 0;
  for (int i = 0; i < o->n_vertices; ++i)
    if (!o->vertices[i].is_ghost)
    {
      loc[4 * n] = o->vertices[i].global_id;
      memcpy(&loc[4 * n + 1], o->vertices[i].pos, 3 * sizeof(double));
      n++;
    }
  int* cnt = malloc(g_np * sizeof(int));
  int* dsp = malloc(g_np * sizeof(int));
  int m = 4 * n;
  MPI_Allgather(&m, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int p = 0; p < g_np; ++p) dsp[p] = tot, tot += cnt[p];
  double* all = malloc((size_t) tot * sizeof(double));
  MPI_Allgatherv(loc, m, MPI_DOUBLE, all, cnt, dsp, MPI_DOUBLE, MPI_COMM_WORLD);
  int N = tot / 4;
  double* res = calloc((size_t) (4 * N), sizeof(double));
  for (int i = 0; i < N; ++i)
  {
    int g = (int) all[4 * i];
    if (g >= 0 && g < N) memcpy(&res[4 * g], &all[4 * i], 4 * sizeof(double));
  }
  free(loc), free(cnt), free(dsp), free(all);
  *out = res;
  return N;
}

static double torus_dist(const double* a, const double* b)
{
  double d2 = 0;
  for (int k = 0; k < 3; ++k)
  {
    double d = a[k] - b[k];
    d -= round(d);
    d2 += d * d;
  }
  return sqrt(d2);
}

static MFTL_Object* soup_sphere(MFTL_Ensemble* ens, int gid, const double c[3], double r)
{
  int nv, nt;
  double(*P)[3];
  int(*T)[3];
  uv_sphere(c, r, 12, &nv, &P, &nt, &T);
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  int* vg = malloc(nv * sizeof(int));
  for (int i = 0; i < nv; ++i) vg[i] = i;
  int* tg = malloc(nt * sizeof(int));
  int(*tv)[3] = malloc(nt * sizeof(*tv));
  int n = 0;
  for (int t = 0; t < nt; ++t)
  {
    if (t % g_np != g_rank) continue;
    tg[n] = t;
    memcpy(tv[n], T[t], sizeof(tv[n]));
    n++;
  }
  int rc = MFTLBuildFromSoup(o, nv, vg, (const double(*)[3]) P, NULL, n, tg, (const int(*)[3]) tv, NULL);
  CHECK(rc == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  free(vg), free(tg), free(tv), free(P), free(T);
  return o;
}

static void test_gauge(void)
{
  const double c[3] = {0.95, 0.03, 0.5};
  const double piv[3] = {0.9, 0.1, 0.45};  // pivot near the body
  const double ang = 0.7;
  double R[3][3] = {{cos(ang), -sin(ang), 0}, {sin(ang), cos(ang), 0}, {0, 0, 1}};
  const int shifts[3][3] = {{0, 0, 0}, {3, -2, 1}, {1000000, -1000000, 7}};
  double* ref = NULL;
  for (int op = 0; op < 3; ++op)  // 0 rotation, 1 non-uniform scale, 2 translation
  {
    double* first = NULL;
    int N = 0;
    for (int g = 0; g < 3; ++g)
    {
      MFTL_Ensemble* ens = ens_new();
      MFTL_Object* o = soup_sphere(ens, 5, c, 0.2);
      for (int i = 0; i < o->n_vertices; ++i)  // whole-body lattice re-gauge
        for (int k = 0; k < 3; ++k) o->vertices[i].img[k] += shifts[g][k];
      if (op == 0) MFTLRotateObject(o, (const double(*)[3]) R, piv);
      if (op == 1)
      {
        double s[3] = {1.1, 0.9, 1.0};
        MFTLScaleObject(o, s);
      }
      if (op == 2)
      {
        double d[3] = {0.31, -0.77, 2.4};
        MFTLTranslateObject(o, d);
      }
      MFTLUpdateGhostPositions(ens, NULL);
      double* pos;
      N = gather_pos(o, &pos);
      if (!first)
        first = pos;
      else
      {
        double worst = 0;
        for (int i = 0; i < N; ++i) worst = fmax(worst, torus_dist(&first[4 * i + 1], &pos[4 * i + 1]));
        if (g_rank == 0) printf("  %s, gauge %d vs gauge 0: worst physical difference %.3e\n", op == 0 ? "rotation" : (op == 1 ? "scaling" : "translation"), g, worst);
        CHECK(worst < 1e-9);
        free(pos);
      }
      int cut[3];
      MFTLCountCutEdges(o, cut);
      CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);
      MFTLEnsembleDestroy(&ens);
    }
    free(first);
  }
  (void) ref;
}


/* film via the soup builder (canonical) */
static MFTL_Object* soup_mesh(MFTL_Ensemble* ens, int gid, int nv, double (*P)[3], int nt, int (*T)[3])
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  int* vg = malloc(nv * sizeof(int));
  for (int i = 0; i < nv; ++i) vg[i] = i;
  int* tg = malloc(nt * sizeof(int));
  int(*tv)[3] = malloc(nt * sizeof(*tv));
  int n = 0;
  for (int t = 0; t < nt; ++t)
  {
    if (t % g_np != g_rank) continue;
    tg[n] = t;
    memcpy(tv[n], T[t], sizeof(tv[n]));
    n++;
  }
  int rc = MFTLBuildFromSoup(o, nv, vg, (const double(*)[3]) P, NULL, n, tg, (const int(*)[3]) tv, NULL);
  CHECK(rc == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  free(vg), free(tg), free(tv);
  return o;
}

static void test_film_transforms(void)
{
  int nv, nt;
  double(*P)[3];
  int(*T)[3];
  film(0.4, 0.05, 16, &nv, &P, &nt, &T);
  for (int op = 0; op < 4; ++op)
  {
    MFTL_Ensemble* ens = ens_new();
    MFTL_Object* o = soup_mesh(ens, 3, nv, P, nt, T);
    int w[3];
    MFTLObjectWrapFlags(o, w);
    CHECK(w[0] && w[1] && !w[2]);
    int rc = 0, expect = 0;
    const char* what = "";
    if (op == 0)
    {
      double R[3][3] = {{0, -1, 0}, {1, 0, 0}, {0, 0, 1}};
      double c[3] = {0.3, 0.3, 0.4};
      rc = MFTLRotateObject(o, (const double(*)[3]) R, c), expect = MFTL_SUCCESS, what = "rotate film 90 deg about z";
    }
    if (op == 1)
    {
      double a = 0.5;
      double R[3][3] = {{cos(a), -sin(a), 0}, {sin(a), cos(a), 0}, {0, 0, 1}};
      double c[3] = {0.3, 0.3, 0.4};
      rc = MFTLRotateObject(o, (const double(*)[3]) R, c), expect = MFTL_ERROR, what = "rotate film 0.5 rad about z";
    }
    if (op == 2)
    {
      double s[3] = {1.0, 1.0, 1.5};
      rc = MFTLScaleObject(o, s), expect = MFTL_SUCCESS, what = "scale film along z";
    }
    if (op == 3)
    {
      double s[3] = {1.2, 1.0, 1.0};
      rc = MFTLScaleObject(o, s), expect = MFTL_ERROR, what = "scale film along x";
    }
    MFTLUpdateGhostPositions(ens, NULL);
    if (g_rank == 0) printf("  %s: rc %d (expected %d)\n", what, rc, expect);
    CHECK(rc == expect);
    MFTLRedistribute(ens);
    CHECK(MFTLCheckWrapFlags(o, 1) == MFTL_SUCCESS);
    MFTLObjectEnsureWinding(o);
    MFTLObjectWrapFlags(o, w);
    CHECK(w[0] && w[1] && !w[2]);
    int cut[3];
    MFTLCountCutEdges(o, cut);
    CHECK(cut[2] == 0);
    MFTLEnsembleDestroy(&ens);
  }
  free(P), free(T);
}

static void test_clear_and_v1(void)
{
  // stale flags after MFTLObjectClear: a film, cleared, then a seam sphere through the serial builder
  MFTL_Ensemble* ens = ens_new();
  int nv, nt;
  double(*P)[3];
  int(*T)[3];
  film(0.4, 0.05, 16, &nv, &P, &nt, &T);
  MFTL_Object* o = soup_mesh(ens, 4, nv, P, nt, T);
  free(P), free(T);
  MFTLObjectClear(o);
  double c[3] = {0.02, 0.98, 0.99};
  uv_sphere(c, 0.25, 16, &nv, &P, &nt, &T);
  const double Vexact = mesh_volume(nt, P, T);
  if (g_rank == 0)
  {
    double(*W)[3] = malloc((size_t) nv * sizeof(*W));
    for (int i = 0; i < nv; ++i)
    {
      memcpy(W[i], P[i], sizeof(W[i]));
      MFTLApplyPeriodicity(&ens->domain, W[i]);
    }
    MFTLBuildTopologyFromTriples(o, nv, (const double(*)[3]) W, nt, (const int(*)[3]) T);
    free(W);
  }
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  double V = MFTLObjectVolume(o);
  if (g_rank == 0) printf("  cleared + rebuilt seam sphere: V %.15g (mesh %.15g)\n", V, Vexact);
  CHECK(fabs(V - Vexact) < 1e-12);
  // v1 native file of the seam sphere (legacy format, no counters), read back on this rank count
  MFTL_SerialBracket br;
  MFTLEnsembleBeginSerial(ens, &br);
  int wrc = (g_rank == 0) ? MFTLWriteNativeMesh(o, "stage16b_gauge_v1.mesh") : MFTL_SUCCESS;
  CHECK(wrc == MFTL_SUCCESS);
  MFTLEnsembleEndSerial(ens, &br, 1);
  MPI_Barrier(MPI_COMM_WORLD);
  MFTL_Ensemble* e2 = ens_new();
  MFTL_Object* r = MFTLObjectCreate(9);
  MFTLEnsembleAddObject(e2, r);
  CHECK(MFTLReadNativeMesh(r, "stage16b_gauge_v1.mesh") == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(r);
  MFTLRedistribute(e2);
  double V2 = MFTLObjectVolume(r);
  if (g_rank == 0) printf("  v1 restart of the seam sphere: V %.15g\n", V2);
  CHECK(fabs(V2 - Vexact) < 1e-12);
  int cut[3];
  MFTLCountCutEdges(r, cut);
  CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);
  free(P), free(T);
  MFTLEnsembleDestroy(&e2);
  MFTLEnsembleDestroy(&ens);
}

/* analytic phase-set checks, members re-gauged by random lattice vectors */
static void regauge(MFTL_Object* o, int a, int b, int c)
{
  for (int i = 0; i < o->n_vertices; ++i) o->vertices[i].img[0] += a, o->vertices[i].img[1] += b, o->vertices[i].img[2] += c;
}

static void flat_sheet(double z, int up, int N, int* nv, double (**P)[3], int* nt, int (**T)[3])
{
  film(z, 0.0, N, nv, P, nt, T);
  if (!up)
    for (int t = 0; t < *nt; ++t)
    {
      int x = (*T)[t][1];
      (*T)[t][1] = (*T)[t][2];
      (*T)[t][2] = x;
    }
}

static void test_phase_analytic(void)
{
  srand(12345);
  for (int seed = 0; seed < 4; ++seed)
  {
    MFTL_Ensemble* ens = ens_new();
    int nv, nt;
    double(*P)[3];
    int(*T)[3];
    flat_sheet(0.2, 0, 8, &nv, &P, &nt, &T);
    MFTL_Object* fl = soup_mesh(ens, 10, nv, P, nt, T);
    free(P), free(T);
    flat_sheet(0.7, 1, 8, &nv, &P, &nt, &T);
    MFTL_Object* ce = soup_mesh(ens, 11, nv, P, nt, T);
    free(P), free(T);
    double cb[3] = {0.5, 0.5, 0.45};
    uv_sphere(cb, 0.1, 12, &nv, &P, &nt, &T);
    const double Vb = mesh_volume(nt, P, T);
    const double Cb = 0.45;  // centroid of the (symmetric) sphere mesh along z
    MFTL_Object* bu = soup_mesh(ens, 12, nv, P, nt, T);
    free(P), free(T);
    int r[9];
    for (int k = 0; k < 9; ++k) r[k] = (rand() % 7) - 3;
    regauge(fl, r[0], r[1], r[2]);
    regauge(ce, r[3], r[4], r[5]);
    regauge(bu, r[6], r[7], r[8]);
    MFTLUpdateGhostPositions(ens, NULL);
    MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, fl, 1);
    MFTLPhaseSetAddMember(S, ce, 1);
    MFTLPhaseSetAddMember(S, bu, -1);  // a bubble of the other phase inside the slab
    double V = 0, C[3];
    MFTL_PhaseStatus cs[3];
    int st = MFTLPhaseSetVolume(S, &V);
    int stc = MFTLPhaseSetCentre(S, C, cs);
    const double Vt = 0.5 - Vb, Ct = (0.5 * (0.49 - 0.04) - Cb * Vb) / Vt;
    double dz = C[2] - Ct;
    dz -= round(dz);
    if (g_rank == 0)
      printf("  slab - bubble (re-gauge %d..): V %.15g (%.15g) Cz %.15g (%.15g) centre status %d/%d/%d\n", r[0], V, Vt, C[2], Ct, cs[0], cs[1], cs[2]);
    CHECK(st == MFTL_DEFINED && fabs(V - Vt) < 1e-12);
    CHECK(stc == MFTL_DEFINED && cs[2] == MFTL_DEFINED && fabs(dz) < 1e-12);
    CHECK(cs[0] == MFTL_UNDEFINED && cs[1] == MFTL_UNDEFINED);  // the slab wraps x and y
    int where = -1;
    double q1[3] = {0.5, 0.5, 0.45}, q2[3] = {0.1, 0.1, 0.5}, q3[3] = {0.1, 0.1, 0.9};
    MFTLPhaseSetInside(S, q1, &where);
    CHECK(where == MFTL_PHASE_OUTSIDE);
    MFTLPhaseSetInside(S, q2, &where);
    CHECK(where == MFTL_PHASE_INSIDE);
    MFTLPhaseSetInside(S, q3, &where);
    CHECK(where == MFTL_PHASE_OUTSIDE);
    MFTLPhaseSetDestroy(&S);
    // the same bubble alone with sigma -1: the matrix is the phase (fully periodic)
    S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, bu, -1);
    st = MFTLPhaseSetVolume(S, &V);
    CHECK(st == MFTL_DEFINED && fabs(V - (1.0 - Vb)) < 1e-12);
    MFTLPhaseSetDestroy(&S);
    // bubble -1 plus a disjoint drop +1 outside it: inconsistent -> UNDEFINED
    uv_sphere((double[3]){0.1, 0.1, 0.9}, 0.05, 10, &nv, &P, &nt, &T);
    MFTL_Object* dr = soup_mesh(ens, 13, nv, P, nt, T);
    free(P), free(T);
    S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, bu, -1);
    MFTLPhaseSetAddMember(S, dr, 1);
    st = MFTLPhaseSetVolume(S, &V);
    CHECK(st == MFTL_UNDEFINED);
    MFTLPhaseSetDestroy(&S);
    MFTLEnsembleDestroy(&ens);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
  setvbuf(stdout, NULL, _IONBF, 0);
  const char* only = getenv("S16G_ONLY");
  if (!only || strstr(only, "gauge"))
  {
    if (g_rank == 0) printf("== gauge\n");
    test_gauge();
  }
  if (!only || strstr(only, "winding"))
  {
    if (g_rank == 0) printf("== winding\n");
    test_winding();
  }
  if (!only || strstr(only, "film"))
  {
    if (g_rank == 0) printf("== film transforms\n");
    test_film_transforms();
  }
  if (!only || strstr(only, "clear"))
  {
    if (g_rank == 0) printf("== clear + v1\n");
    test_clear_and_v1();
  }
  if (!only || strstr(only, "phase"))
  {
    if (g_rank == 0) printf("== phase analytic\n");
    test_phase_analytic();
  }
  if (g_rank == 0) printf("Stage16B-gauge np%d: %d checks, %d failed\n", g_np, g_chk, g_fail);
  MPI_Finalize();
  return g_fail ? 1 : 0;
}
