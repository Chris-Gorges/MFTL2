/*
 * Stage 16B — periodic image representation and phase-boundary sets (plans/16 §3, tests T-B of
 * §10). Every CHECK is collective. Modes:
 *   test_stage16b [cases-file]         all tests; the phase-set cases (B11/B14/B15/B16: every case
 *                                      of §3.6.10 and of the review §19.3, each also rotated so that
 *                                      its evaluation axis is x and y) are read from cases-file
 *                                      (tools/phase_set_prototype/export_cases.py);
 *   test_stage16b write <file>         restart: write a native v2 mesh set (+ reference dump);
 *   test_stage16b read <file>          restart: read it back on another rank count and compare.
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
#include "mftl_io.h"
#include "mftl_phase_set.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"

static int g_rank = 0, g_nproc = 1, g_failed = 0, g_checks = 0;

#define CHECK(cond)                                                                                       \
  do                                                                                                      \
  {                                                                                                       \
    int _l = (cond) ? 0 : 1, _g = 0;                                                                      \
    MPI_Allreduce(&_l, &_g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);                                         \
    g_checks++;                                                                                           \
    if (_g)                                                                                               \
    {                                                                                                     \
      if (_l) fprintf(stderr, "[rank %d] CHECK failed: %s (%s:%d)\n", g_rank, #cond, __FILE__, __LINE__); \
      g_failed++;                                                                                         \
    }                                                                                                     \
  } while (0)

static void grid3(int np, int g[3])
{
  g[0] = np;
  g[1] = 1;
  g[2] = 1;
  if (np == 4) g[0] = 2, g[1] = 2;
  if (np == 8) g[0] = 2, g[1] = 2, g[2] = 2;
  // S16B_SLABS=1|2: one or two slabs along x (every rank is its own / its twin's periodic x
  // neighbour: the 1- and 2-slab neighbour layouts of §2.4/§10), the rest along y
  const char* sl = getenv("S16B_SLABS");
  if (sl && (atoi(sl) == 1 || (atoi(sl) == 2 && np % 2 == 0)))
  {
    g[0] = atoi(sl);
    g[1] = np / g[0];
    g[2] = 1;
  }
}

static MFTL_Ensemble* make_ens(const double mn[3], const double mx[3], const bool per[3])
{
  int g[3];
  grid3(g_nproc, g);
  double a[3] = {mn[0], mn[1], mn[2]}, b[3] = {mx[0], mx[1], mx[2]};
  bool p[3] = {per[0], per[1], per[2]};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, g, a, b, p, 8);
}

/* body from a continuous triangle soup, triangles spread over the ranks (t % np, every 5th also
   on the next rank: builder before ownership), then redistributed */
static int g_id_base = 0;  // vertex / triangle / half-edge id offset of the next soup body

static MFTL_Object* soup_body(MFTL_Ensemble* ens, int gid, int nv, const double (*X)[3], int nt, const int (*T)[3], int open)
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  if (open) MFTLObjectDeclareOpen(o, 1);
  o->next_halfedge_gid = g_id_base;  // disjoint id ranges between bodies (merge test)
  int* vg = malloc((size_t) (nv > 0 ? nv : 1) * sizeof(int));
  for (int i = 0; i < nv; ++i) vg[i] = g_id_base + i;
  int* tg = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(int));
  int(*tv)[3] = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(*tv));
  int n = 0;
  for (int t = 0; t < nt; ++t)
  {
    const int r0 = t % g_nproc;
    if (r0 != g_rank && !((t % 5 == 0) && ((r0 + 1) % g_nproc == g_rank))) continue;
    tg[n] = g_id_base + t;
    for (int c = 0; c < 3; ++c) tv[n][c] = g_id_base + T[t][c];
    n++;
  }
  const MFTL_BoundaryDecl fm = {1, 0, NULL};
  int rc = MFTLBuildFromSoupEx(o, nv, vg, X, NULL, n, tg, (const int (*)[3]) tv, NULL, open ? &fm : NULL);
  CHECK(rc == MFTL_SUCCESS);
  for (int t = 0; t < o->n_triangles; ++t)
    MFTLGetTriangleProperties(o, &o->triangles[t], &o->triangles[t].area, o->triangles[t].normal, o->triangles[t].center);
  g_id_base = 0;
  free(vg);
  free(tg);
  free(tv);
  return o;
}

/* ------------------------------------------------------------------------------------------
 * phase-set cases from the executable model
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  char tag[16];
  int per_z, nclos;
  char clos[2][8];
  char st[16], cst[16];
  double V, Cz;
  int nph;
  int nq;
  double (*Q)[3];
  int* qa;
  int nm;
  int* sigma;
  char (*closure)[8];
  int *nv, *nt;
  double (**X)[3];
  int (**T)[3];
} Case;

static int read_case(FILE* fp, Case* c)
{
  memset(c, 0, sizeof(*c));
  char w[32];
  if (fscanf(fp, "%31s", w) != 1) return 0;
  if (strcmp(w, "case")) return -1;
  if (fscanf(fp, "%15s %d %d", c->tag, &c->per_z, &c->nclos) != 3) return -1;
  for (int i = 0; i < c->nclos; ++i)
    if (fscanf(fp, "%7s", c->clos[i]) != 1) return -1;
  if (fscanf(fp, "%31s %15s %lf %lf %15s %d", w, c->st, &c->V, &c->Cz, c->cst, &c->nph) != 6) return -1;
  if (fscanf(fp, "%31s %d", w, &c->nq) != 2) return -1;
  c->Q = malloc((size_t) (c->nq + 1) * sizeof(*c->Q));
  c->qa = malloc((size_t) (c->nq + 1) * sizeof(int));
  for (int q = 0; q < c->nq; ++q)
    if (fscanf(fp, "%lf %lf %lf %d", &c->Q[q][0], &c->Q[q][1], &c->Q[q][2], &c->qa[q]) != 4) return -1;
  if (fscanf(fp, "%31s %d", w, &c->nm) != 2) return -1;
  c->sigma = malloc((size_t) c->nm * sizeof(int));
  c->closure = malloc((size_t) c->nm * sizeof(*c->closure));
  c->nv = malloc((size_t) c->nm * sizeof(int));
  c->nt = malloc((size_t) c->nm * sizeof(int));
  c->X = malloc((size_t) c->nm * sizeof(*c->X));
  c->T = malloc((size_t) c->nm * sizeof(*c->T));
  for (int m = 0; m < c->nm; ++m)
  {
    if (fscanf(fp, "%31s %d %7s %d %d", w, &c->sigma[m], c->closure[m], &c->nv[m], &c->nt[m]) != 5) return -1;
    c->X[m] = malloc((size_t) c->nv[m] * sizeof(*c->X[m]));
    c->T[m] = malloc((size_t) c->nt[m] * sizeof(*c->T[m]));
    for (int v = 0; v < c->nv[m]; ++v)
      if (fscanf(fp, "%lf %lf %lf", &c->X[m][v][0], &c->X[m][v][1], &c->X[m][v][2]) != 3) return -1;
    for (int t = 0; t < c->nt[m]; ++t)
      if (fscanf(fp, "%d %d %d", &c->T[m][t][0], &c->T[m][t][1], &c->T[m][t][2]) != 3) return -1;
  }
  return 1;
}

static void free_case(Case* c)
{
  for (int m = 0; m < c->nm; ++m)
  {
    free(c->X[m]);
    free(c->T[m]);
  }
  free(c->X);
  free(c->T);
  free(c->sigma);
  free(c->closure);
  free(c->nv);
  free(c->nt);
  free(c->Q);
  free(c->qa);
}

static int status_of(const char* s)
{
  if (!strcmp(s, "DEFINED")) return MFTL_DEFINED;
  if (!strcmp(s, "UNDEFINED")) return MFTL_UNDEFINED;
  return MFTL_UNSUPPORTED;
}

/* model frame (evaluation axis z) -> test frame with the model's z on axis a (cyclic) */
static void rot(int a, const double in[3], double out[3])
{
  out[(a + 1) % 3] = in[0];
  out[(a + 2) % 3] = in[1];
  out[a] = in[2];
}

static int g_case_fail = 0;

static void run_case(const Case* c, int a)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  bool per[3] = {true, true, true};
  per[a] = c->per_z ? true : false;
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTL_Object** bodies = malloc((size_t) c->nm * sizeof(*bodies));
  for (int m = 0; m < c->nm; ++m)
  {
    double(*X)[3] = malloc((size_t) c->nv[m] * sizeof(*X));
    for (int v = 0; v < c->nv[m]; ++v) rot(a, c->X[m][v], X[v]);
    bodies[m] = soup_body(ens, 100 + m, c->nv[m], (const double (*)[3]) X, c->nt[m], (const int (*)[3]) c->T[m], strcmp(c->closure[m], "none") != 0);
    free(X);
  }
  MFTLRedistributeListOfObjects(ens);
  for (int m = 0; m < c->nm; ++m) MFTLPhaseSetAddMember(S, bodies[m], c->sigma[m]);
  for (int q = 0; q < c->nclos; ++q) MFTLPhaseSetAddWallClosure(S, a, strcmp(c->clos[q], "zmin") == 0 ? MFTL_WALL_MIN : MFTL_WALL_MAX);
  MFTL_PhaseSetInfo info;
  MFTL_PhaseStatus st = MFTLPhaseSetClassify(S, &info);
  int bad = (int) st != status_of(c->st);
  double V = 0.0, C[3] = {0, 0, 0};
  MFTL_PhaseStatus cst[3] = {MFTL_UNSUPPORTED, MFTL_UNSUPPORTED, MFTL_UNSUPPORTED};
  double dV = 0.0, dC = 0.0;
  if (!bad && st == MFTL_DEFINED)
  {
    MFTLPhaseSetVolume(S, &V);
    dV = fabs(V - c->V);
    if (dV > 1e-10) bad = 1;
    if (info.n_phase_components != c->nph) bad = 1;
    MFTLPhaseSetCentre(S, C, cst);
    if ((int) cst[a] != status_of(c->cst)) bad = 1;
    if (cst[a] == MFTL_DEFINED)
    {
      double d = C[a] - c->Cz;
      if (c->per_z) d -= round(d);
      dC = fabs(d);
      if (dC > 1e-9) bad = 1;
    }
    for (int q = 0; q < c->nq; ++q)
    {
      double Qr[3];
      rot(a, c->Q[q], Qr);
      int where = -1;
      MFTL_PhaseStatus qs = MFTLPhaseSetInside(S, Qr, &where);
      if (qs != MFTL_DEFINED || where != c->qa[q]) bad = 1;
    }
  }
  if (g_rank == 0)
    printf("  %s %-5s axis %c: expect %-11s got %-11s %s%s V=%.12f dV=%.1e dC=%.1e nph=%d tries=%d\n", bad ? "FAIL" : "ok  ", c->tag, "xyz"[a], c->st,
           st == MFTL_DEFINED ? "DEFINED" : (st == MFTL_UNDEFINED ? "UNDEFINED" : "UNSUPPORTED"), st == MFTL_DEFINED ? "" : info.reason, "", V, dV, dC,
           info.n_phase_components, info.member_line_tries_max);
  CHECK(!bad);
  g_case_fail += bad;
  free(bodies);
  MFTLPhaseSetDestroy(&S);
  MFTLEnsembleDestroy(&ens);
}

static void test_phase_set_cases(const char* path)
{
  FILE* fp = fopen(path, "r");
  int ok = fp != NULL;
  CHECK(ok);
  if (!ok) return;
  Case c;
  int n = 0;
  const char* axes = getenv("S16B_AXES");
  while (read_case(fp, &c) == 1)
  {
    for (int a = 2; a >= 0; --a)
    {
      if (axes && !strchr(axes, "xyz"[a])) continue;
      run_case(&c, a);
    }
    free_case(&c);
    n++;
  }
  fclose(fp);
  if (g_rank == 0) printf("  %d cases, %d failures (all axes)\n", n, g_case_fail);
}

/* ------------------------------------------------------------------------------------------
 * representation tests
 * ------------------------------------------------------------------------------------------ */
static void sphere_soup(const double c[3], double r, int n, double (**X)[3], int* nv, int (**T)[3], int* nt)
{
  const int m = 2 * n;
  *nv = 2 + (n - 1) * m;
  *nt = 2 * m + 2 * m * (n - 2);
  *X = malloc((size_t) *nv * sizeof(**X));
  *T = malloc((size_t) *nt * sizeof(**T));
  int v = 0;
  (*X)[v][0] = c[0], (*X)[v][1] = c[1], (*X)[v][2] = c[2] + r, v++;
  for (int i = 1; i < n; ++i)
    for (int j = 0; j < m; ++j)
    {
      const double th = M_PI * i / n, ph = 2 * M_PI * j / m;
      (*X)[v][0] = c[0] + r * sin(th) * cos(ph);
      (*X)[v][1] = c[1] + r * sin(th) * sin(ph);
      (*X)[v][2] = c[2] + r * cos(th);
      v++;
    }
  (*X)[v][0] = c[0], (*X)[v][1] = c[1], (*X)[v][2] = c[2] - r;
  const int last = v;
#define IDX(i, j) (1 + ((i) - 1) * m + (((j) % m + m) % m))
  int t = 0;
  for (int j = 0; j < m; ++j)
  {
    (*T)[t][0] = 0, (*T)[t][1] = IDX(1, j), (*T)[t][2] = IDX(1, j + 1), t++;
  }
  for (int i = 1; i < n - 1; ++i)
    for (int j = 0; j < m; ++j)
    {
      int a = IDX(i, j), b = IDX(i, j + 1), cc = IDX(i + 1, j + 1), d = IDX(i + 1, j);
      (*T)[t][0] = a, (*T)[t][1] = d, (*T)[t][2] = cc, t++;
      (*T)[t][0] = a, (*T)[t][1] = cc, (*T)[t][2] = b, t++;
    }
  for (int j = 0; j < m; ++j)
  {
    (*T)[t][0] = IDX(n - 1, j), (*T)[t][1] = last, (*T)[t][2] = IDX(n - 1, j + 1), t++;
  }
#undef IDX
  *nt = t;
  // outward orientation (signed volume positive)
  double vol = 0;
  for (int q = 0; q < t; ++q)
  {
    const double *p1 = (*X)[(*T)[q][0]], *p2 = (*X)[(*T)[q][1]], *p3 = (*X)[(*T)[q][2]];
    vol += p1[0] * (p2[1] * p3[2] - p2[2] * p3[1]) + p2[0] * (p3[1] * p1[2] - p3[2] * p1[1]) + p3[0] * (p1[1] * p2[2] - p1[2] * p2[1]);
  }
  if (vol < 0)
    for (int q = 0; q < t; ++q)
    {
      int s = (*T)[q][1];
      (*T)[q][1] = (*T)[q][2];
      (*T)[q][2] = s;
    }
}

static void body_props(MFTL_Object* o, double* V, double* A, double cm[3])
{
  *V = MFTLObjectVolume(o);
  *A = MFTLSurfaceCenter(o, cm);
}

/* B1/B2/B9/B10/B12: compact sphere away from and across seams (x, xy, xyz), dyadic */
static void test_seam_sphere(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  double Vr = 0, Ar = 0, cr[3] = {0, 0, 0};
  const double centres[4][3] = {{0.5, 0.5, 0.5}, {0.96875, 0.5, 0.5}, {0.96875, 0.03125, 0.5}, {0.96875, 0.03125, 0.9375}};
  for (int s = 0; s < 4; ++s)
  {
    MFTL_Ensemble* ens = make_ens(mn, mx, per);
    double(*X)[3];
    int(*T)[3], nv, nt;
    sphere_soup(centres[s], 0.25, 16, &X, &nv, &T, &nt);
    MFTL_Object* o = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
    MFTLRedistributeListOfObjects(ens);
    int wr[3];
    MFTLObjectWrapFlags(o, wr);
    CHECK(wr[0] == 0 && wr[1] == 0 && wr[2] == 0);
    int cut[3];
    MFTLCountCutEdges(o, cut);
    CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);  // canonical: no cut edge on a compact body
    CHECK(MFTLCheckWrapFlags(o, 1) == MFTL_SUCCESS);
    double V, A, cm[3];
    body_props(o, &V, &A, cm);
    MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, o, 1);
    double Vp = 0, C[3];
    MFTL_PhaseStatus cst[3];
    CHECK(MFTLPhaseSetVolume(S, &Vp) == MFTL_DEFINED);
    CHECK(MFTLPhaseSetCentre(S, C, cst) == MFTL_DEFINED);
    CHECK(cst[0] == MFTL_DEFINED && cst[1] == MFTL_DEFINED && cst[2] == MFTL_DEFINED);
    CHECK(fabs(Vp - V) < 1e-14);
    if (s == 0)
    {
      Vr = V;
      Ar = A;
      memcpy(cr, cm, sizeof(cr));
    }
    else
    {
      CHECK(fabs(V - Vr) < 1e-14 && fabs(A - Ar) < 1e-13);
      for (int k = 0; k < 3; ++k)
      {
        double d = (C[k] - centres[s][k]) - (cr[k] - centres[0][k]);
        d -= round(d);
        CHECK(fabs(d) < 1e-13);  // continuous centroid (mod period), same body shape
      }
    }
    if (g_rank == 0) printf("  seam sphere %d: V=%.15f A=%.15f C=(%.6f %.6f %.6f)\n", s, V, A, C[0], C[1], C[2]);
    MFTLPhaseSetDestroy(&S);
    free(X);
    free(T);
    MFTLEnsembleDestroy(&ens);
  }
  // B2: a large compact sphere (extent 0.8 > half period) is accepted with the right volume
  {
    MFTL_Ensemble* ens = make_ens(mn, mx, per);
    double(*X)[3];
    int(*T)[3], nv, nt;
    const double c[3] = {0.9, 0.1, 0.5};
    sphere_soup(c, 0.4, 24, &X, &nv, &T, &nt);
    MFTL_Object* o = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
    MFTLRedistributeListOfObjects(ens);
    double vol = 0;
    for (int q = 0; q < nt; ++q)
    {
      const double *p1 = X[T[q][0]], *p2 = X[T[q][1]], *p3 = X[T[q][2]];
      vol += p1[0] * (p2[1] * p3[2] - p2[2] * p3[1]) + p2[0] * (p3[1] * p1[2] - p3[2] * p1[1]) + p3[0] * (p1[1] * p2[2] - p1[2] * p2[1]);
    }
    vol /= 6.0;
    const double V = MFTLObjectVolume(o);
    if (g_rank == 0) printf("  large compact sphere: V=%.15f exact %.15f\n", V, vol);
    CHECK(fabs(V - vol) < 1e-13);
    free(X);
    free(T);
    MFTLEnsembleDestroy(&ens);
  }
}

/* B10: counters around 1e6: continuous quantities equal the img = 0 case; re-gauge bounds them */
static void test_large_counters(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  double(*X)[3];
  int(*T)[3], nv, nt;
  const double c[3] = {0.97, 0.5, 0.02};
  sphere_soup(c, 0.2, 16, &X, &nv, &T, &nt);
  MFTL_Object* o = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  MFTLRedistributeListOfObjects(ens);
  double V0, A0, c0[3];
  body_props(o, &V0, &A0, c0);
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  double C0[3], C1[3];
  MFTL_PhaseStatus cs[3];
  MFTLPhaseSetCentre(S, C0, cs);
  // a whole-body translation by 1e6 periods: every counter += (1e6, -1e6, 1e6)
  const int big[3] = {1000000, -1000000, 1000000};
  for (int i = 0; i < o->n_vertices; ++i)
    for (int k = 0; k < 3; ++k) o->vertices[i].img[k] += big[k];
  double V1, A1, c1[3];
  body_props(o, &V1, &A1, c1);
  MFTLPhaseSetCentre(S, C1, cs);
  if (g_rank == 0) printf("  img ~ 1e6: V %.17g / %.17g  A %.17g / %.17g\n", V0, V1, A0, A1);
  CHECK(fabs(V1 - V0) <= 1e-13 * fabs(V0));
  CHECK(fabs(A1 - A0) <= 1e-13 * fabs(A0));
  for (int k = 0; k < 3; ++k)
  {
    double d = c1[k] - c0[k];
    d -= round(d);  // the surface centre is continuous (a lattice vector away), mod period equal
    CHECK(fabs(d) < 1e-12);
    double e = C1[k] - C0[k];
    e -= round(e);
    CHECK(fabs(e) < 1e-12);
  }
  // whole-body re-gauge: counters bounded again, shape unchanged
  CHECK(MFTLObjectRegaugeImages(o) == MFTL_SUCCESS);
  int mx_img = 0, gmx = 0;
  for (int i = 0; i < o->n_vertices; ++i)
    for (int k = 0; k < 3; ++k) mx_img = abs(o->vertices[i].img[k]) > mx_img ? abs(o->vertices[i].img[k]) : mx_img;
  MPI_Allreduce(&mx_img, &gmx, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  CHECK(gmx <= 1);
  double V2, A2, c2[3];
  body_props(o, &V2, &A2, c2);
  CHECK(fabs(V2 - V0) <= 1e-14 && fabs(A2 - A0) <= 1e-14);
  MFTLPhaseSetDestroy(&S);
  free(X);
  free(T);
  MFTLEnsembleDestroy(&ens);
}

/* continuous coordinate X = pos + img L of vertex gid 0 (collective) */
static void vertex0_continuous(MFTL_Object* o, double X[3])
{
  double loc[3] = {-1e300, -1e300, -1e300};
  for (int i = 0; i < o->n_vertices; ++i)
    if (!o->vertices[i].is_ghost && o->vertices[i].global_id == 0)
      for (int k = 0; k < 3; ++k) loc[k] = o->vertices[i].pos[k] + o->vertices[i].img[k] * o->domain->domain_size[k];
  MPI_Allreduce(loc, X, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

/* B8: counter lifecycle */
static void test_lifecycle(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  double(*X)[3];
  int(*T)[3], nv, nt;
  const double c[3] = {0.75, 0.375, 0.625};  // after 100 steps: (32.0, -21.5, 47.5) -> on the x seam
  sphere_soup(c, 0.15, 12, &X, &nv, &T, &nt);
  MFTL_Object* o = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  MFTLRedistributeListOfObjects(ens);
  double V0, A0, c0[3];
  body_props(o, &V0, &A0, c0);
  double X0[3];
  vertex0_continuous(o, X0);
  // advect across seams 100 times (translation d, wrap counted into img)
  const double d[3] = {0.3125, -0.21875, 0.46875};
  for (int step = 0; step < 100; ++step)
  {
    MFTLTranslateObject(o, d);
    MFTLWrapObjectIntoDomain(o);
    MFTLUpdateGhostPositions(ens, NULL);
    if (step % 10 == 9) MFTLRedistributeListOfObjects(ens);
  }
  int cut[3];
  MFTLCountCutEdges(o, cut);
  CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);  // img followed every wrap
  double V1, A1, c1[3];
  body_props(o, &V1, &A1, c1);
  CHECK(fabs(V1 - V0) < 1e-12 && fabs(A1 - A0) < 1e-12);
  double X1[3];
  vertex0_continuous(o, X1);
  for (int k = 0; k < 3; ++k)
  {
    CHECK(fabs(X1[k] - (X0[k] + 100 * d[k])) < 1e-9);  // continuous coordinate: no jump, exact counters
    double e = c1[k] - (c0[k] + 100 * d[k]);
    e -= round(e);
    CHECK(fabs(e) < 1e-9);  // surface centre, modulo the period
  }
  // exact counters: the reference vertex moved by 100 d continuously
  // canonicalising wrapped input (all img = 0) reproduces the counters up to one lattice shift
  int* saved = malloc((size_t) (3 * (o->n_vertices + 1)) * sizeof(int));
  for (int i = 0; i < o->n_vertices; ++i) memcpy(&saved[3 * i], o->vertices[i].img, 3 * sizeof(int));
  for (int i = 0; i < o->n_vertices; ++i) o->vertices[i].img[0] = o->vertices[i].img[1] = o->vertices[i].img[2] = 0;
  MFTLCountCutEdges(o, cut);
  CHECK(cut[0] + cut[1] + cut[2] > 0);  // artificial cuts of the wrapped input
  CHECK(MFTLCanonicaliseImages(o) == MFTL_SUCCESS);
  MFTLCountCutEdges(o, cut);
  CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);
  int shift_ok = 1, have = 0, g[3] = {0, 0, 0};
  for (int i = 0; i < o->n_vertices; ++i)
  {
    int dd[3];
    for (int k = 0; k < 3; ++k) dd[k] = saved[3 * i + k] - o->vertices[i].img[k];
    if (!have)
    {
      memcpy(g, dd, sizeof(g));
      have = 1;
    }
    else if (dd[0] != g[0] || dd[1] != g[1] || dd[2] != g[2])
      shift_ok = 0;
  }
  // one lattice shift per rank; also identical across ranks
  int glo[3], ghi[3], gmin[3], gmax[3];  // ranks without vertices do not vote
  for (int k = 0; k < 3; ++k)
  {
    glo[k] = have ? g[k] : 2147483647;
    ghi[k] = have ? g[k] : -2147483647;
  }
  MPI_Allreduce(glo, gmin, 3, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(ghi, gmax, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  CHECK(shift_ok && gmin[0] == gmax[0] && gmin[1] == gmax[1] && gmin[2] == gmax[2]);
  // idempotent
  for (int i = 0; i < o->n_vertices; ++i) memcpy(&saved[3 * i], o->vertices[i].img, 3 * sizeof(int));
  CHECK(MFTLCanonicaliseImages(o) == MFTL_SUCCESS);
  int same = 1;
  for (int i = 0; i < o->n_vertices; ++i)
    if (memcmp(&saved[3 * i], o->vertices[i].img, 3 * sizeof(int))) same = 0;
  CHECK(same);
  free(saved);
  free(X);
  free(T);
  MFTLEnsembleDestroy(&ens);
}

/* B12: two drops merged across a seam, one given as wrapped input without counters */
static void test_merge(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  double(*X)[3];
  int(*T)[3], nv, nt;
  const double ca[3] = {0.3, 0.5, 0.5}, cb[3] = {0.98, 0.5, 0.97};
  sphere_soup(ca, 0.15, 12, &X, &nv, &T, &nt);
  MFTL_Object* A = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  free(X);
  free(T);
  sphere_soup(cb, 0.1, 12, &X, &nv, &T, &nt);
  g_id_base = 1000000;
  MFTL_Object* B = soup_body(ens, 2, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  free(X);
  free(T);
  MFTLRedistributeListOfObjects(ens);
  const double VA = MFTLObjectVolume(A), VB = MFTLObjectVolume(B);
  // B as wrapped input without counters: artificial cut edges
  for (int i = 0; i < B->n_vertices; ++i) B->vertices[i].img[0] = B->vertices[i].img[1] = B->vertices[i].img[2] = 0;
  int cut[3];
  MFTLCountCutEdges(B, cut);
  CHECK(cut[0] > 0 && cut[2] > 0);
  CHECK(MFTLCheckWrapFlags(B, 0) == MFTL_ERROR);  // the regression check detects them
  // merge: copies of B into A (gids offset by the builder id ranges) + ASSEMBLE commit
  // (canonicalises)
  for (int t = 0; t < B->n_triangles; ++t)
    if (!B->triangles[t].is_ghost) MFTLObjectAppendTriangleCopy(A, B, t);
  CHECK(MFTLObjectCommitCopy(A, MFTL_COMMIT_ASSEMBLE) == MFTL_SUCCESS);
  MFTLRedistributeListOfObjects(ens);
  MFTLCountCutEdges(A, cut);
  CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);
  int wr[3];
  MFTLObjectWrapFlags(A, wr);
  CHECK(wr[0] == 0 && wr[1] == 0 && wr[2] == 0);
  const double V = MFTLObjectVolume(A);
  if (g_rank == 0) printf("  merge: V=%.15f  VA+VB=%.15f\n", V, VA + VB);
  CHECK(fabs(V - (VA + VB)) < 1e-14);
  MFTLEnsembleDestroy(&ens);
}

/* ribbon (open strip) along direction dir (lattice vector), width along w, in plane normal nrm */
static void ribbon(MFTL_Ensemble* ens, int gid, const int dir[3], int w, double off, MFTL_Object** out)
{
  const int N = 32, M = 4;
  double(*X)[3] = malloc((size_t) ((N + 1) * (M + 1)) * sizeof(*X));
  int(*T)[3] = malloc((size_t) (2 * N * M) * sizeof(*T));
  int nv = 0, nt = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j <= M; ++j)
    {
      const double u = (double) i / N, v = 0.3 + 0.2 * j / M;
      for (int k = 0; k < 3; ++k) X[nv][k] = off + u * dir[k];
      X[nv][w] = v;
      nv++;
    }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
    {
      int a = i * (M + 1) + j, b = ((i + 1) % N) * (M + 1) + j, c = ((i + 1) % N) * (M + 1) + j + 1, d = i * (M + 1) + j + 1;
      T[nt][0] = a, T[nt][1] = b, T[nt][2] = c, nt++;
      T[nt][0] = a, T[nt][1] = c, T[nt][2] = d, nt++;
    }
  // the continuous coordinates of the closing column (i = N) are those of i = 0 plus one lattice
  // vector: the soup builder receives i = 0 (identified vertices); every edge is short
  *out = soup_body(ens, gid, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 1);
  free(X);
  free(T);
}

/* B3/B13: x-, y-, z-winding and diagonal ribbons: flags, loop windings, area, phase-set status */
static void test_ribbons(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  const int dirs[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 0}};
  const int widths[4] = {1, 2, 0, 2};
  for (int r = 0; r < 4; ++r)
  {
    MFTL_Ensemble* ens = make_ens(mn, mx, per);
    MFTL_Object* o;
    ribbon(ens, 1, dirs[r], widths[r], 0.1, &o);
    MFTLRedistributeListOfObjects(ens);
    int wr[3];
    MFTLObjectWrapFlags(o, wr);
    for (int k = 0; k < 3; ++k) CHECK(wr[k] == (dirs[r][k] != 0));
    CHECK(MFTLValidateTwinStatesEx(o, MFTL_VALIDATE_GHOSTS) == MFTL_SUCCESS);
    MFTL_BoundaryLoopSet L;
    CHECK(MFTLBoundaryLoops(o, &L) == MFTL_SUCCESS);
    CHECK(L.n_loops == 2);
    int wind_ok = 1;
    for (int l = 0; l < L.n_loops; ++l)
    {
      int s = 0;
      for (int k = 0; k < 3; ++k)
        if (L.loops[l].winding[k] != 0) s = L.loops[l].winding[k] > 0 ? 1 : -1;
      for (int k = 0; k < 3; ++k)
        if (L.loops[l].winding[k] != s * dirs[r][k]) wind_ok = 0;
    }
    CHECK(wind_ok);
    MFTLFreeBoundaryLoops(&L);
    // area: width 0.2 x length |dir|
    double A = 0.0, gA = 0.0;
    for (int t = 0; t < o->n_triangles; ++t)
      if (!o->triangles[t].is_ghost)
      {
        double a;
        MFTLGetTriangleProperties(o, &o->triangles[t], &a, NULL, NULL);
        A += a;
      }
    MPI_Allreduce(&A, &gA, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    const double len = sqrt((double) (dirs[r][0] * dirs[r][0] + dirs[r][1] * dirs[r][1] + dirs[r][2] * dirs[r][2]));
    CHECK(fabs(gA - 0.2 * len) < 1e-12);
    // an open ribbon is not closed by a declared wall: UNDEFINED (never a value)
    MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, o, 1);
    double V = -1;
    CHECK(MFTLPhaseSetVolume(S, &V) == MFTL_UNDEFINED && V == -1);
    MFTLPhaseSetDestroy(&S);
    if (g_rank == 0) printf("  ribbon dir (%d,%d,%d): wrap (%d,%d,%d) area %.15f\n", dirs[r][0], dirs[r][1], dirs[r][2], wr[0], wr[1], wr[2], gA);
    MFTLEnsembleDestroy(&ens);
  }
}

/* B4: an xy film in an xyz-periodic domain: WRAP_x ∧ WRAP_y, volume UNDEFINED (Φ_z ≠ 0) */
static void test_film(void)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  const int N = 16;
  double(*X)[3] = malloc((size_t) (N * N) * sizeof(*X));
  int(*T)[3] = malloc((size_t) (2 * N * N) * sizeof(*T));
  int nt = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      X[j * N + i][0] = (double) i / N;
      X[j * N + i][1] = (double) j / N;
      X[j * N + i][2] = 0.4 + 0.05 * sin(2 * M_PI * i / N);
    }
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      int a = j * N + i, b = j * N + (i + 1) % N, c = ((j + 1) % N) * N + (i + 1) % N, d = ((j + 1) % N) * N + i;
      T[nt][0] = a, T[nt][1] = b, T[nt][2] = c, nt++;
      T[nt][0] = a, T[nt][1] = c, T[nt][2] = d, nt++;
    }
  MFTL_Object* o = soup_body(ens, 1, N * N, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  MFTLRedistributeListOfObjects(ens);
  int wr[3];
  MFTLObjectWrapFlags(o, wr);
  CHECK(wr[0] == 1 && wr[1] == 1 && wr[2] == 0);
  CHECK(!MFTLObjectHasVolume(o));
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  MFTL_PhaseSetInfo info;
  CHECK(MFTLPhaseSetClassify(S, &info) == MFTL_UNDEFINED);
  CHECK(fabs(fabs(info.Phi[2]) - 1.0) < 1e-12);
  MFTLPhaseSetDestroy(&S);
  free(X);
  free(T);
  MFTLEnsembleDestroy(&ens);
}

/* ------------------------------------------------------------------------------------------
 * restart on a different rank count (A5 / §3.2 native I/O v2)
 * ------------------------------------------------------------------------------------------ */
static void restart_bodies(MFTL_Ensemble* ens, MFTL_Object** drop, MFTL_Object** rib)
{
  double(*X)[3];
  int(*T)[3], nv, nt;
  const double c[3] = {0.97, 0.03, 0.5};
  sphere_soup(c, 0.2, 12, &X, &nv, &T, &nt);
  // stored three periods away along z: counters (0,0,3)
  for (int v = 0; v < nv; ++v) X[v][2] += 3.0;
  *drop = soup_body(ens, 1, nv, (const double (*)[3]) X, nt, (const int (*)[3]) T, 0);
  free(X);
  free(T);
  const int dir[3] = {1, 0, 0};
  ribbon(ens, 2, dir, 1, 0.1, rib);
  for (MFTL_Object* o = *drop; o; o = (o == *drop ? *rib : NULL))
    for (int t = 0; t < o->n_triangles; ++t) o->triangles[t].surface_scalar = 0.25 + 1e-3 * o->triangles[t].global_id;
  MFTLRedistributeListOfObjects(ens);
}

typedef struct
{
  int gid, img[3];
  double pos[3];
} VRec;

static int cmp_vrec(const void* a, const void* b)
{
  const VRec *x = a, *y = b;
  return (x->gid > y->gid) - (x->gid < y->gid);
}

static int gather_vrecs(MFTL_Object* o, VRec** out)
{
  int n = 0;
  VRec* loc = malloc((size_t) (o->n_vertices + 1) * sizeof(VRec));
  for (int i = 0; i < o->n_vertices; ++i)
    if (!o->vertices[i].is_ghost)
    {
      loc[n].gid = o->vertices[i].global_id;
      memcpy(loc[n].img, o->vertices[i].img, sizeof(loc[n].img));
      memcpy(loc[n].pos, o->vertices[i].pos, sizeof(loc[n].pos));
      n++;
    }
  int nb = n * (int) sizeof(VRec), *cnt = malloc((size_t) g_nproc * sizeof(int)), *dsp = malloc((size_t) g_nproc * sizeof(int));
  MPI_Allgather(&nb, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int p = 0; p < g_nproc; ++p)
  {
    dsp[p] = tot;
    tot += cnt[p];
  }
  VRec* all = malloc((size_t) (tot > 0 ? tot : 1));
  MPI_Allgatherv(loc, nb, MPI_BYTE, all, cnt, dsp, MPI_BYTE, MPI_COMM_WORLD);
  qsort(all, (size_t) (tot / (int) sizeof(VRec)), sizeof(VRec), cmp_vrec);
  free(loc);
  free(cnt);
  free(dsp);
  *out = all;
  return tot / (int) sizeof(VRec);
}

static double scalar_sum(MFTL_Object* o)
{
  double s = 0, g = 0;
  for (int t = 0; t < o->n_triangles; ++t)
    if (!o->triangles[t].is_ghost) s += o->triangles[t].surface_scalar * (1.0 + o->triangles[t].global_id);
  MPI_Allreduce(&s, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

static void restart_write(const char* path)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  MFTL_Object *drop, *rib;
  restart_bodies(ens, &drop, &rib);
  MFTL_Object* bodies[2] = {drop, rib};
  for (int b = 0; b < 2; ++b)
  {
    VRec* v;
    int n = gather_vrecs(bodies[b], &v);
    const double ss = scalar_sum(bodies[b]);
    MFTL_SerialBracket br;
    MFTLEnsembleBeginSerial(ens, &br);
    char fn[512];
    snprintf(fn, sizeof(fn), "%s.body%d", path, b);
    int wrc = MFTL_SUCCESS;
    if (g_rank == 0) wrc = MFTLWriteNativeMeshV2(bodies[b], fn);
    CHECK(wrc == MFTL_SUCCESS);  // collective check outside the rank-0 branch
    MFTLEnsembleEndSerial(ens, &br, 1);
    if (g_rank == 0)
    {
      snprintf(fn, sizeof(fn), "%s.body%d.ref", path, b);
      FILE* fp = fopen(fn, "w");
      fprintf(fp, "%d %.17g\n", n, ss);
      for (int i = 0; i < n; ++i) fprintf(fp, "%d %d %d %d %.17g %.17g %.17g\n", v[i].gid, v[i].img[0], v[i].img[1], v[i].img[2], v[i].pos[0], v[i].pos[1], v[i].pos[2]);
      fclose(fp);
    }
    free(v);
  }
  MFTLEnsembleDestroy(&ens);
}

static void restart_read(const char* path)
{
  const double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  const bool per[3] = {true, true, true};
  MFTL_Ensemble* ens = make_ens(mn, mx, per);
  for (int b = 0; b < 2; ++b)
  {
    MFTL_Object* o = MFTLObjectCreate(1 + b);
    MFTLEnsembleAddObject(ens, o);
    char fn[512];
    snprintf(fn, sizeof(fn), "%s.body%d", path, b);
    CHECK(MFTLReadNativeMesh(o, g_rank == 0 ? fn : NULL) == MFTL_SUCCESS);
    MFTLRedistributeListOfObjects(ens);
    CHECK(MFTLValidateTwinStatesEx(o, MFTL_VALIDATE_GHOSTS) == MFTL_SUCCESS);
    VRec* v;
    int n = gather_vrecs(o, &v);
    snprintf(fn, sizeof(fn), "%s.body%d.ref", path, b);
    FILE* fp = fopen(fn, "r");
    int nr = 0;
    double ss = 0;
    int ok = fp && fscanf(fp, "%d %lf", &nr, &ss) == 2 && nr == n;
    for (int i = 0; ok && i < n; ++i)
    {
      VRec r;
      if (fscanf(fp, "%d %d %d %d %lf %lf %lf", &r.gid, &r.img[0], &r.img[1], &r.img[2], &r.pos[0], &r.pos[1], &r.pos[2]) != 7) ok = 0;
      if (r.gid != v[i].gid || memcmp(r.img, v[i].img, sizeof(r.img)) || memcmp(r.pos, v[i].pos, sizeof(r.pos))) ok = 0;
    }
    if (fp) fclose(fp);
    CHECK(ok);  // identical counters and positions (exact), on any rank count
    CHECK(fabs(scalar_sum(o) - ss) <= 1e-15 * fabs(ss));
    CHECK(o->declared_open == (b == 1));
    if (b == 1)
    {
      MFTL_BoundaryLoopSet L;
      CHECK(MFTLBoundaryLoops(o, &L) == MFTL_SUCCESS);
      CHECK(L.n_loops == 2);
      int tags_ok = 1;
      for (int l = 0; l < L.n_loops; ++l)
        if (L.loops[l].tag != l || L.loops[l].winding[0] == 0) tags_ok = 0;
      CHECK(tags_ok);
      MFTLFreeBoundaryLoops(&L);
    }
    free(v);
    if (g_rank == 0) printf("  restart body %d: %d vertices, counters and positions identical on %d rank(s)\n", b, n, g_nproc);
  }
  MFTLEnsembleDestroy(&ens);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc >= 3 && !strcmp(argv[1], "write"))
    restart_write(argv[2]);
  else if (argc >= 3 && !strcmp(argv[1], "read"))
    restart_read(argv[2]);
  else
  {
    const char* only = getenv("S16B_ONLY");
    struct
    {
      const char* name;
      void (*fn)(void);
    } tests[] = {{"seam_sphere", test_seam_sphere}, {"large_counters", test_large_counters}, {"lifecycle", test_lifecycle},
                 {"merge", test_merge},             {"ribbons", test_ribbons},               {"film", test_film}};
    for (size_t k = 0; k < sizeof(tests) / sizeof(tests[0]); ++k)
    {
      if (only && !strstr(only, tests[k].name)) continue;
      const int before = g_failed;
      if (g_rank == 0) printf("== %s\n", tests[k].name);
      tests[k].fn();
      if (g_rank == 0) printf("   %s: %s\n", tests[k].name, g_failed == before ? "ok" : "FAILED");
    }
    if (argc >= 2 && (!only || strstr(only, "cases")))
    {
      const int before = g_failed;
      if (g_rank == 0) printf("== phase_set_cases (%s)\n", argv[1]);
      test_phase_set_cases(argv[1]);
      if (g_rank == 0) printf("   phase_set_cases: %s\n", g_failed == before ? "ok" : "FAILED");
    }
  }
  if (g_rank == 0) printf("Stage16B np%d: %d checks, %d failed\n", g_nproc, g_checks, g_failed);
  MPI_Finalize();
  return g_failed ? 1 : 0;
}
