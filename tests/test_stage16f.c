/*
 * Stage 16F — final validation of the wrapping (non-contractible) topology (plans/16 §28, Part F).
 * Every CHECK is collective. Registered on 1, 2, 4 and 8 ranks (plus restart and abort-contract
 * tests, tests/CMakeLists.txt).
 *
 *   test_stage16f                   invariance of the winding class of
 *                                     - open ribbons winding once along x, y, z, (1,1,0), (1,0,1),
 *                                       (0,1,1) and (1,1,1),
 *                                     - closed films wrapping xy, yz and zx,
 *                                     - a closed tube wrapping x,
 *                                     - a compact sphere across the xyz corner (must stay COMPACT),
 *                                   under redistribution, a whole-body lattice re-gauge (every
 *                                   counter shifted by (2,-1,3), then MFTLObjectRegaugeImages),
 *                                   generalized / parity remeshing (4 cycles) and the rank count:
 *                                   wrap flags, MFTLCheckWrapFlags (cut edges), boundary-loop count
 *                                   and windings, twin states, area; and the checked queries
 *                                   (HasVolume, singleton phase-set volume / centre statuses);
 *   test_stage16f write <path>      the same bodies after the operations, native v2 + reference;
 *   test_stage16f read <path>       read on another rank count: identical counters and positions
 *                                   (the state is written in the canonical image gauge, which the
 *                                   reader re-establishes), class, windings and loops;
 *   test_stage16f unchecked <what>  the unchecked single-body calls on a body they do not apply to
 *                                   must abort with their documented message (ctest matches it):
 *                                   volume_tube, inertia_ribbon (wrapping), volume_patch,
 *                                   inertia_patch (open, compact).
 */
#include <math.h>
#include <mpi.h>
#include <stdbool.h>
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

static int g_rank = 0, g_np = 1, g_failed = 0, g_checks = 0;

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

static MFTL_Ensemble* ens_new(int periodic)
{
  int g[3] = {g_np, 1, 1};
  if (g_np == 4) g[0] = 2, g[1] = 2;
  if (g_np == 8) g[0] = 2, g[1] = 2, g[2] = 2;
  double mn[3] = {0, 0, 0}, mx[3] = {1, 1, 1};
  bool per[3] = {periodic != 0, periodic != 0, periodic != 0};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, g, mn, mx, per, 8);
}

typedef struct
{
  int nv, nt, open;
  double (*X)[3];
  int (*T)[3];
} Soup;

static void soup_alloc(Soup* s, int nv, int nt, int open)
{
  s->nv = nv, s->nt = nt, s->open = open;
  s->X = malloc((size_t) nv * sizeof(*s->X));
  s->T = malloc((size_t) nt * sizeof(*s->T));
}

static void soup_free(Soup* s)
{
  free(s->X);
  free(s->T);
}

/* body from a continuous soup; triangles spread over the ranks (builder before ownership) */
static MFTL_Object* build(MFTL_Ensemble* ens, int gid, const Soup* s)
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  if (s->open) MFTLObjectDeclareOpen(o, 1);
  int* vg = malloc((size_t) s->nv * sizeof(int));
  for (int i = 0; i < s->nv; ++i) vg[i] = i;
  int* tg = malloc((size_t) s->nt * sizeof(int));
  int(*tv)[3] = malloc((size_t) s->nt * sizeof(*tv));
  int n = 0;
  for (int t = 0; t < s->nt; ++t)
  {
    if (t % g_np != g_rank) continue;
    tg[n] = t;
    memcpy(tv[n], s->T[t], sizeof(tv[n]));
    n++;
  }
  const MFTL_BoundaryDecl fm = {1, 0, NULL};
  CHECK(MFTLBuildFromSoupEx(o, s->nv, vg, (const double(*)[3]) s->X, NULL, n, tg, (const int(*)[3]) tv, NULL, s->open ? &fm : NULL) == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  free(vg);
  free(tg);
  free(tv);
  return o;
}

/* open ribbon winding once along the lattice vector d, width 0.2 along the unit vector e;
   N segments along d, the closing column identified with the first (every edge is short) */
static Soup ribbon(const int d[3], const double e[3], int N)
{
  const int M = 4;
  Soup s;
  soup_alloc(&s, N * (M + 1), 2 * N * M, 1);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j <= M; ++j)
      for (int k = 0; k < 3; ++k) s.X[i * (M + 1) + j][k] = 0.13 + (double) i / N * d[k] + 0.2 * j / M * e[k];
  int nt = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < M; ++j)
    {
      const int a = i * (M + 1) + j, b = ((i + 1) % N) * (M + 1) + j, c = ((i + 1) % N) * (M + 1) + j + 1, q = i * (M + 1) + j + 1;
      s.T[nt][0] = a, s.T[nt][1] = b, s.T[nt][2] = c, nt++;
      s.T[nt][0] = a, s.T[nt][1] = c, s.T[nt][2] = q, nt++;
    }
  return s;
}

/* closed film normal to axis `ax`, periodic in the two other axes, height 0.4 + 0.05 sin */
static Soup film(int ax, int N)
{
  const int a1 = (ax + 1) % 3, a2 = (ax + 2) % 3;
  Soup s;
  soup_alloc(&s, N * N, 2 * N * N, 0);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      double* x = s.X[j * N + i];
      x[a1] = (double) i / N, x[a2] = (double) j / N, x[ax] = 0.4 + 0.05 * sin(2 * M_PI * i / N);
    }
  int nt = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      const int a = j * N + i, b = j * N + (i + 1) % N, c = ((j + 1) % N) * N + (i + 1) % N, q = ((j + 1) % N) * N + i;
      s.T[nt][0] = a, s.T[nt][1] = b, s.T[nt][2] = c, nt++;
      s.T[nt][0] = a, s.T[nt][1] = c, s.T[nt][2] = q, nt++;
    }
  return s;
}

/* closed tube along x (radius 0.15 about y = z = 0.5), outward normals */
static Soup tube(int N, int R)
{
  Soup s;
  soup_alloc(&s, N * R, 2 * N * R, 0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < R; ++j)
    {
      const double ph = 2 * M_PI * (j + 0.5 * (i % 2)) / R;
      double* x = s.X[i * R + j];
      x[0] = (double) i / N, x[1] = 0.5 + 0.15 * cos(ph), x[2] = 0.5 + 0.15 * sin(ph);
    }
  int nt = 0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < R; ++j)
    {
      const int i1 = (i + 1) % N;
      int a = i * R + j, b = i * R + (j + 1) % R, c, q;
      if (i % 2 == 0)
        c = i1 * R + j, q = i1 * R + (j - 1 + R) % R;
      else
        c = i1 * R + (j + 1) % R, q = i1 * R + j;
      // (a, b, c) and (a, c, q): outward for a tube along +x with (y, z) = (cos, sin)
      s.T[nt][0] = a, s.T[nt][1] = b, s.T[nt][2] = c, nt++;
      s.T[nt][0] = a, s.T[nt][1] = c, s.T[nt][2] = q, nt++;
    }
  return s;
}

/* closed UV sphere (radius r about c, continuous coordinates), outward normals */
static Soup sphere(const double c[3], double r, int nth, int nph)
{
  Soup s;
  soup_alloc(&s, 2 + (nth - 1) * nph, 2 * nph * (nth - 1), 0);
  s.X[0][0] = c[0], s.X[0][1] = c[1], s.X[0][2] = c[2] + r;
  for (int i = 1; i < nth; ++i)
    for (int j = 0; j < nph; ++j)
    {
      const double th = M_PI * i / nth, ph = 2 * M_PI * (j + 0.5 * (i % 2)) / nph;
      double* x = s.X[1 + (i - 1) * nph + j];
      x[0] = c[0] + r * sin(th) * cos(ph), x[1] = c[1] + r * sin(th) * sin(ph), x[2] = c[2] + r * cos(th);
    }
  const int S = 1 + (nth - 1) * nph;
  s.X[S][0] = c[0], s.X[S][1] = c[1], s.X[S][2] = c[2] - r;
  int nt = 0;
  for (int j = 0; j < nph; ++j) s.T[nt][0] = 0, s.T[nt][1] = 1 + j, s.T[nt][2] = 1 + (j + 1) % nph, nt++;
  for (int i = 1; i < nth - 1; ++i)
    for (int j = 0; j < nph; ++j)
    {
      const int a = 1 + (i - 1) * nph + j, b = 1 + (i - 1) * nph + (j + 1) % nph, p = 1 + i * nph + j, q = 1 + i * nph + (j + 1) % nph;
      if (i % 2)
        s.T[nt][0] = a, s.T[nt][1] = p, s.T[nt][2] = q, nt++, s.T[nt][0] = a, s.T[nt][1] = q, s.T[nt][2] = b, nt++;
      else
        s.T[nt][0] = a, s.T[nt][1] = p, s.T[nt][2] = b, nt++, s.T[nt][0] = b, s.T[nt][1] = p, s.T[nt][2] = q, nt++;
    }
  for (int j = 0; j < nph; ++j) s.T[nt][0] = S, s.T[nt][1] = 1 + (nth - 2) * nph + (j + 1) % nph, s.T[nt][2] = 1 + (nth - 2) * nph + j, nt++;
  s.nt = nt;
  // orient outward
  for (int t = 0; t < nt; ++t)
  {
    const double *p0 = s.X[s.T[t][0]], *p1 = s.X[s.T[t][1]], *p2 = s.X[s.T[t][2]];
    double e1[3], e2[3], n[3], g[3];
    for (int k = 0; k < 3; ++k) e1[k] = p1[k] - p0[k], e2[k] = p2[k] - p0[k], g[k] = (p0[k] + p1[k] + p2[k]) / 3 - c[k];
    n[0] = e1[1] * e2[2] - e1[2] * e2[1], n[1] = e1[2] * e2[0] - e1[0] * e2[2], n[2] = e1[0] * e2[1] - e1[1] * e2[0];
    if (n[0] * g[0] + n[1] * g[1] + n[2] * g[2] < 0)
    {
      const int tmp = s.T[t][1];
      s.T[t][1] = s.T[t][2], s.T[t][2] = tmp;
    }
  }
  return s;
}

static Soup planar_patch(void)
{
  const int n = 9;
  Soup s;
  soup_alloc(&s, n * n, 2 * (n - 1) * (n - 1), 1);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) s.X[j * n + i][0] = 0.3 + 0.4 * i / (n - 1), s.X[j * n + i][1] = 0.3 + 0.4 * j / (n - 1), s.X[j * n + i][2] = 0.5;
  int nt = 0;
  for (int j = 0; j < n - 1; ++j)
    for (int i = 0; i < n - 1; ++i)
    {
      const int a = j * n + i, b = a + 1, c = a + n + 1, q = a + n;
      s.T[nt][0] = a, s.T[nt][1] = b, s.T[nt][2] = c, nt++;
      s.T[nt][0] = a, s.T[nt][1] = c, s.T[nt][2] = q, nt++;
    }
  return s;
}

/* ------------------------------------------------------------------------------------------ */
typedef struct
{
  int wrap[3], compact, has_volume, nloops, wind[4][3], valid, wrapchk;
  double area;
} Snap;

static double area_of(MFTL_Object* o)
{
  double A = 0.0, g = 0.0;
  for (int t = 0; t < o->n_triangles; ++t)
  {
    const MFTL_Triangle* tri = &o->triangles[t];
    if (tri->is_ghost || (tri->flags & MFTL_FLAG_DEAD)) continue;
    double a;
    MFTLGetTriangleProperties(o, &o->triangles[t], &a, NULL, NULL);
    A += a;
  }
  MPI_Allreduce(&A, &g, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

static void snap(MFTL_Object* o, Snap* s)
{
  memset(s, 0, sizeof(*s));
  MFTLObjectEnsureWinding(o);
  MFTLObjectWrapFlags(o, s->wrap);
  s->compact = MFTLObjectIsCompact(o);
  s->has_volume = MFTLObjectHasVolume(o);
  s->valid = MFTLValidateTwinStatesEx(o, MFTL_VALIDATE_GHOSTS) == MFTL_SUCCESS;
  s->wrapchk = MFTLCheckWrapFlags(o, 1) == MFTL_SUCCESS;
  s->area = area_of(o);
  MFTL_BoundaryLoopSet L;
  if (MFTLBoundaryLoops(o, &L) == MFTL_SUCCESS)
  {
    s->nloops = L.n_loops;
    // windings up to sign, sorted by tag (tags are persistent)
    for (int l = 0; l < L.n_loops && l < 4; ++l)
    {
      int sg = 0;
      for (int k = 0; k < 3; ++k)
        if (!sg && L.loops[l].winding[k]) sg = L.loops[l].winding[k] > 0 ? 1 : -1;
      for (int k = 0; k < 3; ++k) s->wind[l][k] = sg ? sg * L.loops[l].winding[k] : 0;
    }
    MFTLFreeBoundaryLoops(&L);
  }
  else
    s->nloops = -1;
}

static int same_class(const Snap* a, const Snap* b)
{
  if (memcmp(a->wrap, b->wrap, sizeof(a->wrap)) || a->compact != b->compact || a->has_volume != b->has_volume || a->nloops != b->nloops) return 0;
  for (int l = 0; l < a->nloops && l < 4; ++l)
    if (memcmp(a->wind[l], b->wind[l], sizeof(a->wind[l]))) return 0;
  return a->valid && b->valid && a->wrapchk && b->wrapchk;
}

/* whole-body lattice re-gauge: every counter (owned and ghost, identically) shifted, then the
   library re-gauge; the geometry and the winding class must not change */
static void regauge(MFTL_Object* o)
{
  const int shift[3] = {2, -1, 3};
  for (int i = 0; i < o->n_vertices; ++i)
    for (int k = 0; k < 3; ++k) o->vertices[i].img[k] += shift[k];
  CHECK(MFTLObjectRegaugeImages(o) == MFTL_SUCCESS);
}

static void remesh_cycles(MFTL_Ensemble* ens, MFTL_Object* o, int generalized, int* fails)
{
  if (generalized) MFTLObjectSetRemeshMode(o, MFTL_REMESH_MODE_GENERALIZED);
  for (int cyc = 0; cyc < 4; ++cyc)
  {
    MFTL_RemeshParameters p;
    memset(&p, 0, sizeof(p));
    MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, (cyc % 2) ? 0.03 : 0.07, 1, MFTL_SMOOTH_QUALITY_DRIVEN, 45.0);
    p.quiet = 1;
    p.max_smoothing_iter = 1;
    p.mftl1_criteria = !generalized && !o->declared_open;
    p.skip_volume_recovery = 0;
    MFTLComputeTriangleProperties(o);
    MFTLRemeshEnsembleParallel(ens, &p);
    MFTLComputeTriangleProperties(o);
    MFTLRedistribute(ens);
    if (MFTLValidateTwinStatesEx(o, MFTL_VALIDATE_GHOSTS) != MFTL_SUCCESS || MFTLCheckWrapFlags(o, 1) != MFTL_SUCCESS) (*fails)++;
  }
}

typedef struct
{
  const char* name;
  int kind;  // 0 ribbon, 1 film, 2 tube, 3 corner sphere
  int d[3];
  double e[3];
  int ax;
} Case;

static const double S3 = 0.70710678118654752;
static const Case g_cases[] = {
    {"ribbon x", 0, {1, 0, 0}, {0, 1, 0}, 0},        {"ribbon y", 0, {0, 1, 0}, {0, 0, 1}, 0},     {"ribbon z", 0, {0, 0, 1}, {1, 0, 0}, 0},
    {"ribbon (1,1,0)", 0, {1, 1, 0}, {0, 0, 1}, 0},  {"ribbon (1,0,1)", 0, {1, 0, 1}, {0, 1, 0}, 0}, {"ribbon (0,1,1)", 0, {0, 1, 1}, {1, 0, 0}, 0},
    {"ribbon (1,1,1)", 0, {1, 1, 1}, {S3, -S3, 0}, 0}, {"film xy", 1, {0}, {0}, 2},                {"film yz", 1, {0}, {0}, 0},
    {"film zx", 1, {0}, {0}, 1},                     {"tube x", 2, {0}, {0}, 0},                   {"sphere xyz corner", 3, {0}, {0}, 0}};
#define NCASES ((int) (sizeof(g_cases) / sizeof(g_cases[0])))

static Soup make(const Case* c)
{
  switch (c->kind)
  {
    case 0: return ribbon(c->d, c->e, (c->d[0] + c->d[1] + c->d[2]) > 1 ? 48 : 32);
    case 1: return film(c->ax, 16);
    case 2: return tube(32, 16);
    default:
    {
      const double ctr[3] = {0.98, 0.02, 0.97};
      return sphere(ctr, 0.2, 12, 24);
    }
  }
}

static void expected(const Case* c, Snap* x)
{
  memset(x, 0, sizeof(*x));
  x->valid = x->wrapchk = 1;
  if (c->kind == 0)
  {
    for (int k = 0; k < 3; ++k) x->wrap[k] = c->d[k] != 0;
    x->nloops = 2;
    for (int l = 0; l < 2; ++l)
      for (int k = 0; k < 3; ++k) x->wind[l][k] = c->d[k];
  }
  else if (c->kind == 1)
    x->wrap[(c->ax + 1) % 3] = x->wrap[(c->ax + 2) % 3] = 1;
  else if (c->kind == 2)
    x->wrap[0] = 1;
  else
    x->compact = x->has_volume = 1;
}

static void test_invariance(void)
{
  for (int ci = 0; ci < NCASES; ++ci)
  {
    const Case* c = &g_cases[ci];
    MFTL_Ensemble* ens = ens_new(1);
    Soup s = make(c);
    MFTL_Object* o = build(ens, 1, &s);
    soup_free(&s);
    Snap ex, s0, s1, s2;
    expected(c, &ex);
    snap(o, &s0);
    CHECK(same_class(&s0, &ex));
    MFTLRedistributeListOfObjects(ens);
    snap(o, &s1);
    CHECK(same_class(&s1, &s0) && fabs(s1.area - s0.area) <= 1e-13 * s0.area);
    regauge(o);
    MFTLRedistributeListOfObjects(ens);
    snap(o, &s1);
    CHECK(same_class(&s1, &s0) && fabs(s1.area - s0.area) <= 1e-13 * s0.area);
    // checked queries: a value only for a defined phase (never for a wrapping singleton)
    MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
    MFTLPhaseSetAddMember(S, o, 1);
    double V = -1.0, C[3];
    MFTL_PhaseStatus cs[3];
    const MFTL_PhaseStatus vs = MFTLPhaseSetVolume(S, &V);
    const MFTL_PhaseStatus ctr = MFTLPhaseSetCentre(S, C, cs);
    MFTLPhaseSetDestroy(&S);
    if (c->kind == 0 || c->kind == 1)
      CHECK(vs != MFTL_DEFINED && V == -1.0 && ctr != MFTL_DEFINED);  // open ribbon / flux-carrying film
    else if (c->kind == 2)
      CHECK(vs == MFTL_DEFINED && fabs(V - M_PI * 0.15 * 0.15) < 0.02 * V && cs[0] == MFTL_UNDEFINED);  // C10: set-level volume, x-centre undefined
    else
      CHECK(vs == MFTL_DEFINED && fabs(V - MFTLObjectVolume(o)) <= 1e-13 * V && ctr == MFTL_DEFINED);
    int fails = 0;
    remesh_cycles(ens, o, c->kind == 0, &fails);
    snap(o, &s2);
    CHECK(fails == 0 && same_class(&s2, &s0));
    if (g_rank == 0)
      printf("  %-18s wrap (%d,%d,%d) compact %d loops %d wind (%d,%d,%d) area %.12f -> remeshed %.12f, class kept: %s\n", c->name, s0.wrap[0], s0.wrap[1],
             s0.wrap[2], s0.compact, s0.nloops, s0.wind[0][0], s0.wind[0][1], s0.wind[0][2], s0.area, s2.area, same_class(&s2, &s0) ? "yes" : "NO");
    MFTLEnsembleDestroy(&ens);
  }
}

/* the area-based surface centre is defined for an open compact patch (an unchecked call that
   stays valid); the volume-based calls on it abort (mode `unchecked`) */
static void test_open_centre(void)
{
  MFTL_Ensemble* ens = ens_new(0);
  Soup s = planar_patch();
  MFTL_Object* o = build(ens, 1, &s);
  soup_free(&s);
  double cm[3];
  const double A = MFTLSurfaceCenter(o, cm);
  CHECK(fabs(A - 0.16) < 1e-13 && fabs(cm[0] - 0.5) < 1e-13 && fabs(cm[1] - 0.5) < 1e-13 && fabs(cm[2] - 0.5) < 1e-13);
  CHECK(!MFTLObjectHasVolume(o));
  MFTLEnsembleDestroy(&ens);
}

/* ------------------------------------------------------------------------------------------
 * restart on another rank count (native v2)
 * ------------------------------------------------------------------------------------------ */
static void after_ops(MFTL_Ensemble* ens, MFTL_Object* o, const Case* c)
{
  regauge(o);
  int fails = 0;
  remesh_cycles(ens, o, c->kind == 0, &fails);
  CHECK(fails == 0);
}

typedef struct
{
  int gid, img[3];
  double pos[3];
} VRec;

static int cmp_vrec(const void* a, const void* b) { return ((const VRec*) a)->gid - ((const VRec*) b)->gid; }

static int gather_vrecs(MFTL_Object* o, VRec** out)
{
  int n = 0;
  VRec* v = malloc((size_t) (o->n_vertices + 1) * sizeof(VRec));
  for (int i = 0; i < o->n_vertices; ++i)
  {
    const MFTL_Vertex* x = &o->vertices[i];
    if (x->is_ghost) continue;
    v[n].gid = x->global_id;
    memcpy(v[n].img, x->img, sizeof(v[n].img));
    memcpy(v[n].pos, x->pos, sizeof(v[n].pos));
    n++;
  }
  int* cnt = malloc((size_t) g_np * sizeof(int));
  int* dsp = malloc((size_t) g_np * sizeof(int));
  int nb = n * (int) sizeof(VRec), tot = 0;
  MPI_Allgather(&nb, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < g_np; ++r) dsp[r] = tot, tot += cnt[r];
  VRec* all = malloc((size_t) tot + 1);
  MPI_Allgatherv(v, nb, MPI_BYTE, all, cnt, dsp, MPI_BYTE, MPI_COMM_WORLD);
  const int N = tot / (int) sizeof(VRec);
  qsort(all, (size_t) N, sizeof(VRec), cmp_vrec);
  free(v);
  free(cnt);
  free(dsp);
  *out = all;
  return N;
}

static void restart_write(const char* path)
{
  for (int ci = 0; ci < NCASES; ++ci)
  {
    const Case* c = &g_cases[ci];
    MFTL_Ensemble* ens = ens_new(1);
    Soup s = make(c);
    MFTL_Object* o = build(ens, 1, &s);
    soup_free(&s);
    after_ops(ens, o, c);
    // the reader canonicalises the image gauge (io.c): a canonical state is restored exactly; a
    // non-canonical one (e.g. fresh from remeshing) comes back in its canonical gauge (same
    // positions, class and windings; the cut edges of a wrapping body may move)
    Snap sn0, sn;
    snap(o, &sn0);
    CHECK(MFTLCanonicaliseImages(o) == MFTL_SUCCESS);
    snap(o, &sn);
    CHECK(same_class(&sn, &sn0) && fabs(sn.area - sn0.area) <= 1e-13 * sn0.area);
    VRec* v;
    const int n = gather_vrecs(o, &v);
    MFTL_SerialBracket br;
    MFTLEnsembleBeginSerial(ens, &br);
    char fn[512];
    snprintf(fn, sizeof(fn), "%s.case%d", path, ci);
    int wrc = MFTL_SUCCESS;
    if (g_rank == 0) wrc = MFTLWriteNativeMeshV2(o, fn);
    CHECK(wrc == MFTL_SUCCESS);
    MFTLEnsembleEndSerial(ens, &br, 1);
    if (g_rank == 0)
    {
      snprintf(fn, sizeof(fn), "%s.case%d.ref", path, ci);
      FILE* fp = fopen(fn, "w");
      fprintf(fp, "%d %.17g %d %d %d %d %d\n", n, sn.area, sn.wrap[0], sn.wrap[1], sn.wrap[2], sn.compact, sn.nloops);
      for (int l = 0; l < sn.nloops && l < 4; ++l) fprintf(fp, "%d %d %d\n", sn.wind[l][0], sn.wind[l][1], sn.wind[l][2]);
      for (int i = 0; i < n; ++i) fprintf(fp, "%d %d %d %d %.17g %.17g %.17g\n", v[i].gid, v[i].img[0], v[i].img[1], v[i].img[2], v[i].pos[0], v[i].pos[1], v[i].pos[2]);
      fclose(fp);
    }
    free(v);
    MFTLEnsembleDestroy(&ens);
  }
  if (g_rank == 0) printf("  wrote %d bodies (after re-gauge and remeshing) on %d ranks\n", NCASES, g_np);
}

static void restart_read(const char* path)
{
  for (int ci = 0; ci < NCASES; ++ci)
  {
    const Case* c = &g_cases[ci];
    MFTL_Ensemble* ens = ens_new(1);
    MFTL_Object* o = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(ens, o);
    char fn[512];
    snprintf(fn, sizeof(fn), "%s.case%d", path, ci);
    CHECK(MFTLReadNativeMesh(o, g_rank == 0 ? fn : NULL) == MFTL_SUCCESS);
    MFTLRedistributeListOfObjects(ens);
    Snap sn, ref;
    snap(o, &sn);
    memset(&ref, 0, sizeof(ref));
    ref.valid = ref.wrapchk = 1;
    VRec* v;
    const int n = gather_vrecs(o, &v);
    snprintf(fn, sizeof(fn), "%s.case%d.ref", path, ci);
    FILE* fp = fopen(fn, "r");
    int nr = 0, ok = fp != NULL;
    double A = 0;
    ok = ok && fscanf(fp, "%d %lf %d %d %d %d %d", &nr, &A, &ref.wrap[0], &ref.wrap[1], &ref.wrap[2], &ref.compact, &ref.nloops) == 7 && nr == n;
    for (int l = 0; ok && l < ref.nloops && l < 4; ++l) ok = fscanf(fp, "%d %d %d", &ref.wind[l][0], &ref.wind[l][1], &ref.wind[l][2]) == 3;
    ref.has_volume = ref.compact && !o->declared_open;
    for (int i = 0; ok && i < n; ++i)
    {
      VRec r;
      if (fscanf(fp, "%d %d %d %d %lf %lf %lf", &r.gid, &r.img[0], &r.img[1], &r.img[2], &r.pos[0], &r.pos[1], &r.pos[2]) != 7) ok = 0;
      if (r.gid != v[i].gid || memcmp(r.img, v[i].img, sizeof(r.img)) || memcmp(r.pos, v[i].pos, sizeof(r.pos)))
      {
        if (ok && g_rank == 0 && getenv("S16F_DEBUG"))
          printf("  first difference: gid %d/%d img (%d %d %d)/(%d %d %d) pos (%.17g %.17g %.17g)/(%.17g %.17g %.17g)\n", r.gid, v[i].gid, r.img[0], r.img[1],
                 r.img[2], v[i].img[0], v[i].img[1], v[i].img[2], r.pos[0], r.pos[1], r.pos[2], v[i].pos[0], v[i].pos[1], v[i].pos[2]);
        ok = 0;
      }
    }
    if (fp) fclose(fp);
    CHECK(ok);  // identical counters and positions (exact)
    CHECK(same_class(&sn, &ref) && fabs(sn.area - A) <= 1e-13 * A);
    CHECK(o->declared_open == (c->kind == 0));
    free(v);
    if (g_rank == 0) printf("  restart %-18s: %d vertices, counters/positions/class/windings identical on %d rank(s)\n", c->name, n, g_np);
    MFTLEnsembleDestroy(&ens);
  }
}

/* the documented contract of the unchecked single-body calls: abort with a message */
static void unchecked(const char* what)
{
  const int wrapping = !strcmp(what, "volume_tube") || !strcmp(what, "inertia_ribbon");
  MFTL_Ensemble* ens = ens_new(wrapping);
  Soup s;
  if (!strcmp(what, "volume_tube"))
    s = tube(32, 16);
  else if (!strcmp(what, "inertia_ribbon"))
  {
    const int d[3] = {1, 0, 0};
    const double e[3] = {0, 1, 0};
    s = ribbon(d, e, 32);
  }
  else
    s = planar_patch();
  MFTL_Object* o = build(ens, 1, &s);
  soup_free(&s);
  if (!strncmp(what, "volume", 6))
  {
    const double V = MFTLObjectVolume(o);
    if (g_rank == 0) printf("NOT ABORTED: volume %.17g\n", V);
  }
  else
  {
    double I[3][3];
    const double c[3] = {0.5, 0.5, 0.5};
    MFTLMomentOfInertia(o, I, c, 1.0);
    if (g_rank == 0) printf("NOT ABORTED: inertia %.17g\n", I[0][0]);
  }
  MFTLEnsembleDestroy(&ens);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
  if (argc >= 3 && !strcmp(argv[1], "write"))
    restart_write(argv[2]);
  else if (argc >= 3 && !strcmp(argv[1], "read"))
    restart_read(argv[2]);
  else if (argc >= 3 && !strcmp(argv[1], "unchecked"))
    unchecked(argv[2]);
  else
  {
    test_invariance();
    test_open_centre();
  }
  if (g_rank == 0) printf("Stage16F (np %d): %d checks, %d failed\n", g_np, g_checks, g_failed);
  MPI_Finalize();
  return g_failed ? 1 : 0;
}
