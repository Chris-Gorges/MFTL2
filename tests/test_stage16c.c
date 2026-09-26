/*
 * Stage 16C — generalized geometry kernels and boundary-aware remeshing (plans/16 §4, tests T-C
 * of §10). Every CHECK is collective. Registered at 1, 2, 4 and 8 ranks.
 *   planar    C1: open planar patch — 4(n-1) boundary edges, 1 loop, chi = 1, area and normals
 *                 exact, interior curvature 0, boundary vertices flagged CURV_UNDEFINED, corners;
 *   gb        C2: hemisphere refinement — loop length -> 2 pi R, area -> 2 pi R^2 (rate >= 1.9),
 *                 interior curvature -> 2/R, discrete Gauss-Bonnet (sphere, hemisphere, open
 *                 tube, film), extrapolated boundary curvature, boundary-edge conormals;
 *   link      C3: augmented link condition — one-triangle strip (pinches rejected), ear (legal)
 *                 and isolated triangle (rejected), two holes one edge apart (merge rejected),
 *                 triangular hole (closing rejected), corners (never collapsed);
 *   cycles    C4: 20 refine/coarsen cycles (serial and parallel) on a planar patch (boundary on
 *                 the square to 1e-12, corners bit-fixed, area exact, boundary coarsens and
 *                 refines: 16A S-A4) and on a perturbed hemisphere (REPRESENTED: <= C h^3;
 *                 CONSTRAINED circle projector: 1e-12); loops, tags; validation every cycle
 *                 (no bow-tie / pinch); an interior collapse keeps its boundary survivor (S-A5);
 *   params        16D review: MFTLInitRemeshParams defines boundary_corner_angle and
 *                 stat_recovery_skipped (were indeterminate);
 *   valence       16C review: an open wheel of 30 spokes refined: no valence growth above the
 *                 initial 30 (generalized split/flip cap 24; bounded one-ring gathers);
 *   seam      C5: open patch across the x and y seams vs translated away (dyadic): loops, tags,
 *                 areas, normals, curvature flags/values identical; serial remesh pass identical
 *                 (np1) / statistically equal (np > 1);
 *   scales    C6: C1, C2 (Gauss-Bonnet), C4 (planar) and B1 at geometry scales 1e-4 and 1e4;
 *   periodic      wrapping ribbon (loop windings preserved), closed film (parity and
 *                 generalized), closed seam-crossing sphere under parity remeshing and its
 *                 dyadic translation, closed tube wrapping x (set-level volume recovery), volume
 *                 recovery skipped where the volume is undefined; proximity / thin-region search
 *                 across seams.
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
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_boundary_internal.h"

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

/* ------------------------------------------------------------------------------------------
 * ensembles and bodies
 * ------------------------------------------------------------------------------------------ */
static MFTL_Ensemble* ens_new(double lo, double hi, int px, int py, int pz)
{
  int g[3] = {g_np, 1, 1};
  if (g_np == 4) g[0] = 2, g[1] = 2;
  if (g_np == 8) g[0] = 2, g[1] = 2, g[2] = 2;
  double mn[3] = {lo, lo, lo}, mx[3] = {hi, hi, hi};
  bool per[3] = {px != 0, py != 0, pz != 0};
  MFTL_Ensemble* e = MFTLEnsembleCreate(MPI_COMM_WORLD, g, mn, mx, per, 4);
  e->type = MFTL_ENSEMBLE_SPANNING;  // the parallel remesher runs its distributed pass
  return e;
}

typedef struct
{
  int nv, nt;
  double (*P)[3];
  int (*T)[3];
} Mesh;

static void mesh_free(Mesh* m)
{
  free(m->P);
  free(m->T);
  m->P = NULL;
  m->T = NULL;
}

/* soup build: triangles spread over the ranks, continuous input, open bodies from the mesh */
static MFTL_Object* body(MFTL_Ensemble* ens, int gid, const Mesh* M, int open)
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  if (open) MFTLObjectDeclareOpen(o, 1);
  int* vg = malloc((size_t) M->nv * sizeof(int));
  for (int i = 0; i < M->nv; ++i) vg[i] = i;
  int* tg = malloc((size_t) M->nt * sizeof(int));
  int(*tv)[3] = malloc((size_t) M->nt * sizeof(*tv));
  int n = 0;
  for (int t = 0; t < M->nt; ++t)
  {
    if (t % g_np != g_rank) continue;
    tg[n] = t;
    memcpy(tv[n], M->T[t], sizeof(tv[n]));
    n++;
  }
  const MFTL_BoundaryDecl fm = {1, 0, NULL};
  int rc = MFTLBuildFromSoupEx(o, M->nv, vg, (const double(*)[3]) M->P, NULL, n, tg, (const int(*)[3]) tv, NULL, open ? &fm : NULL);
  CHECK(rc == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  free(vg);
  free(tg);
  free(tv);
  return o;
}

/* grid patch (nx x ny vertices) over [x0, x0 + w] x [y0, y0 + h], z = f(x, y); optional wrap
   along x (ribbon: nx columns, the last connected to the first) and optional cell holes */
typedef double (*ZFun)(double x, double y);
static Mesh grid(int nx, int ny, double x0, double y0, double w, double h, ZFun f, int wrap_x, const int* holes, int nholes)
{
  Mesh M;
  M.nv = nx * ny;
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  const int ncx = wrap_x ? nx : nx - 1;
  M.T = malloc((size_t) (2 * ncx * (ny - 1)) * sizeof(*M.T));
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i)
    {
      double x = x0 + w * i / (wrap_x ? nx : (nx - 1)), y = y0 + h * j / (ny - 1);
      M.P[j * nx + i][0] = x, M.P[j * nx + i][1] = y, M.P[j * nx + i][2] = f(x, y);
    }
  M.nt = 0;
  for (int j = 0; j < ny - 1; ++j)
    for (int i = 0; i < ncx; ++i)
    {
      int hole = 0;
      for (int k = 0; k < nholes; ++k)
        if (holes[2 * k] == i && holes[2 * k + 1] == j) hole = 1;
      if (hole) continue;
      const int i1 = (i + 1) % nx;
      int a = j * nx + i, b = j * nx + i1, c = (j + 1) * nx + i1, d = (j + 1) * nx + i;
      if ((i + j) % 2 == 0)
      {
        M.T[M.nt][0] = a, M.T[M.nt][1] = b, M.T[M.nt][2] = c, M.nt++;
        M.T[M.nt][0] = a, M.T[M.nt][1] = c, M.T[M.nt][2] = d, M.nt++;
      }
      else
      {
        M.T[M.nt][0] = a, M.T[M.nt][1] = b, M.T[M.nt][2] = d, M.nt++;
        M.T[M.nt][0] = b, M.T[M.nt][1] = c, M.T[M.nt][2] = d, M.nt++;
      }
    }
  return M;
}

/* UV sphere (closed) or upper cap (theta <= theta_max, open) about c, outward */
static Mesh uv_sphere(const double c[3], double r, int n, int cap)
{
  Mesh M;
  const int m = 2 * n, rings = cap ? n / 2 : n - 1;
  M.nv = 1 + rings * m + (cap ? 0 : 1);
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) (2 * m * n) * sizeof(*M.T));
  M.P[0][0] = c[0], M.P[0][1] = c[1], M.P[0][2] = c[2] + r;
  for (int i = 1; i <= rings; ++i)
    for (int j = 0; j < m; ++j)
    {
      double th = M_PI * i / n, ph = 2 * M_PI * j / m;
      double* q = M.P[1 + (i - 1) * m + j];
      q[0] = c[0] + r * sin(th) * cos(ph), q[1] = c[1] + r * sin(th) * sin(ph), q[2] = c[2] + r * cos(th);
      if (cap && i == rings) q[2] = c[2];  // the rim exactly in the plane
    }
#define ID(i, j) (1 + ((i) - 1) * m + ((j) % m))
  M.nt = 0;
  for (int j = 0; j < m; ++j) M.T[M.nt][0] = 0, M.T[M.nt][1] = ID(1, j), M.T[M.nt][2] = ID(1, j + 1), M.nt++;
  for (int i = 1; i < rings; ++i)
    for (int j = 0; j < m; ++j)
    {
      int a = ID(i, j), b = ID(i, j + 1), cc = ID(i + 1, j + 1), d = ID(i + 1, j);
      M.T[M.nt][0] = a, M.T[M.nt][1] = d, M.T[M.nt][2] = cc, M.nt++;
      M.T[M.nt][0] = a, M.T[M.nt][1] = cc, M.T[M.nt][2] = b, M.nt++;
    }
  if (!cap)
  {
    const int S = M.nv - 1;
    M.P[S][0] = c[0], M.P[S][1] = c[1], M.P[S][2] = c[2] - r;
    for (int j = 0; j < m; ++j) M.T[M.nt][0] = ID(n - 1, j), M.T[M.nt][1] = S, M.T[M.nt][2] = ID(n - 1, j + 1), M.nt++;
  }
#undef ID
  // outward orientation: the first triangle's normal must point away from c
  double *p0 = M.P[M.T[0][0]], *p1 = M.P[M.T[0][1]], *p2 = M.P[M.T[0][2]];
  double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
  double nz = e1[0] * e2[1] - e1[1] * e2[0];
  if (nz < 0)
    for (int t = 0; t < M.nt; ++t)
    {
      int x = M.T[t][1];
      M.T[t][1] = M.T[t][2];
      M.T[t][2] = x;
    }
  return M;
}

/* tube around the x direction: closed (wraps x) or open (x in [x0, x1], two rims) */
static Mesh tube(double yc, double zc, double r, int nx, int m, int wrap, double x0, double x1)
{
  Mesh M;
  const int cols = wrap ? nx : nx + 1;
  M.nv = cols * m;
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) (2 * nx * m) * sizeof(*M.T));
  for (int i = 0; i < cols; ++i)
    for (int j = 0; j < m; ++j)
    {
      double th = 2 * M_PI * j / m;
      double* q = M.P[i * m + j];
      q[0] = x0 + (x1 - x0) * i / nx, q[1] = yc + r * cos(th), q[2] = zc + r * sin(th);
    }
  M.nt = 0;
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < m; ++j)
    {
      int i1 = wrap ? (i + 1) % nx : i + 1;
      int a = i * m + j, b = i1 * m + j, c = i1 * m + (j + 1) % m, d = i * m + (j + 1) % m;
      M.T[M.nt][0] = a, M.T[M.nt][1] = c, M.T[M.nt][2] = b, M.nt++;
      M.T[M.nt][0] = a, M.T[M.nt][1] = d, M.T[M.nt][2] = c, M.nt++;
    }
  return M;
}

/* ------------------------------------------------------------------------------------------
 * collective measurements
 * ------------------------------------------------------------------------------------------ */
typedef struct
{
  long V, F, B;  // owned vertices, owned triangles, owned boundary half-edges
  double area;
} Counts;

static Counts counts(MFTL_Object* o)
{
  long loc[3] = {0, 0, 0};
  double a = 0.0;
  for (int i = 0; i < o->n_vertices; ++i)
    if (!o->vertices[i].is_ghost && !(o->vertices[i].flags & MFTL_FLAG_DEAD) && o->vertices[i].out_he_idx >= 0) loc[0]++;
  for (int t = 0; t < o->n_triangles; ++t)
  {
    const MFTL_Triangle* T = &o->triangles[t];
    if (T->is_ghost || (T->flags & MFTL_FLAG_DEAD)) continue;
    loc[1]++;
    double P[3][3];
    MFTLTriangleUnwrapped(o, T, P);
    double e1[3] = {P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]}, e2[3] = {P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]};
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    a += 0.5 * sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    MFTL_HalfEdge* h[3];
    MFTLGetTriangleHalfedges(o, (MFTL_Triangle*) T, h);
    for (int k = 0; k < 3; ++k)
      if (MFTLHeIsBoundary(h[k])) loc[2]++;
  }
  Counts c;
  long g[3];
  MPI_Allreduce(loc, g, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&a, &c.area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  c.V = g[0], c.F = g[1], c.B = g[2];
  return c;
}

static long euler(const Counts* c) { return c->V - (3 * c->F + c->B) / 2 + c->F; }

/* Gauss-Bonnet sum over owned vertices: interior 2 pi - sum, boundary pi - sum */
static double gauss_bonnet(MFTL_Object* o, int* bad)
{
  double s = 0.0;
  int b = 0;
  for (int i = 0; i < o->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &o->vertices[i];
    if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
    double sum;
    int cls = MFTLVertexAngleSum(o, i, &sum);
    if (cls == MFTL_RING_CLOSED)
      s += 2 * M_PI - sum;
    else if (cls == MFTL_RING_OPEN_COMPLETE)
      s += M_PI - sum;
    else
      b++;
  }
  double gs;
  MPI_Allreduce(&s, &gs, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int gb;
  MPI_Allreduce(&b, &gb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  *bad = gb;
  return gs;
}

static int valid(MFTL_Object* o) { return MFTLValidateTwinStatesEx(o, MFTL_VALIDATE_GHOSTS) == MFTL_SUCCESS; }

static int loops(MFTL_Object* o, MFTL_BoundaryLoopSet* L)
{
  if (MFTLBoundaryLoops(o, L) != MFTL_SUCCESS)
  {
    L->n_loops = -1;
    L->loops = NULL;
    return -1;
  }
  return L->n_loops;
}

static void remesh_params(MFTL_RemeshParameters* p, double target)
{
  memset(p, 0, sizeof(*p));
  MFTLInitRemeshParams(p, MFTL_REMESH_ISOTROPIC, target, 1, MFTL_SMOOTH_QUALITY_DRIVEN, 45.0);
  p->quiet = 1;
  p->max_smoothing_iter = 1;
}

static void remesh(MFTL_Ensemble* ens, MFTL_RemeshParameters* p)
{
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleProperties(ens->objects[o]);
  MFTLRemeshEnsembleParallel(ens, p);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleProperties(ens->objects[o]);
  MFTLRedistribute(ens);
}

/* ------------------------------------------------------------------------------------------
 * C1: planar patch (+ C6 scales)
 * ------------------------------------------------------------------------------------------ */
static double g_zplane = 0.5;
static double zplane(double x, double y)
{
  (void) x, (void) y;
  return g_zplane;
}

static void test_planar(double s)
{
  MFTL_Ensemble* ens = ens_new(0.0, s, 0, 0, 0);
  g_zplane = 0.5 * s;
  const int n = 9;
  Mesh M = grid(n, n, 0.1 * s, 0.1 * s, 0.8 * s, 0.8 * s, zplane, 0, NULL, 0);
  MFTL_Object* o = body(ens, 1, &M, 1);
  mesh_free(&M);
  CHECK(valid(o));
  Counts c = counts(o);
  CHECK(c.B == 4 * (n - 1));
  CHECK(euler(&c) == 1);
  CHECK(fabs(c.area - 0.64 * s * s) <= 1e-14 * s * s);
  MFTL_BoundaryLoopSet L;
  CHECK(loops(o, &L) == 1);
  if (L.n_loops == 1) CHECK(fabs(L.loops[0].length - 3.2 * s) <= 1e-13 * s && L.loops[0].tag == 0);
  MFTLFreeBoundaryLoops(&L);
  MFTLComputeVertexNormals(o);
  MFTLComputeVertexCurvatures(o);
  int bad_n = 0, bad_k = 0, flagged = 0, corners = 0, bnd = 0;
  for (int i = 0; i < o->n_vertices; ++i)
  {
    MFTL_Vertex* v = &o->vertices[i];
    if (v->is_ghost) continue;
    if (fabs(v->normal[0]) > 1e-14 || fabs(v->normal[1]) > 1e-14 || fabs(v->normal[2] - 1.0) > 1e-14) bad_n++;
    double th;
    if (MFTLVertexTurningAngle(o, i, &th) == MFTL_SUCCESS)
    {
      bnd++;
      if (!(v->flags & MFTL_VFLAG_CURV_UNDEFINED)) flagged--;
      if (fabs(th) > M_PI / 6) corners++;
      if (fabs(th) > M_PI / 6 && fabs(th - M_PI / 2) > 1e-12) bad_k++;
      if (fabs(th) <= M_PI / 6 && fabs(th) > 1e-12) bad_k++;
    }
    else
    {
      if (v->flags & MFTL_VFLAG_CURV_UNDEFINED) flagged--;
      if (v->meancurvature * s > 1e-12) bad_k++;
    }
  }
  int loc[4] = {bad_n, bad_k, flagged, corners}, g[4];
  MPI_Allreduce(loc, g, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(g[0] == 0);  // normals exact, boundary included
  CHECK(g[1] == 0);  // interior curvature 0; turning angles 0 or 90 degrees
  CHECK(g[2] == 0);  // boundary vertices flagged, interior not
  CHECK(g[3] == 4);  // four corners
  int gbad;
  double gb = gauss_bonnet(o, &gbad);
  CHECK(gbad == 0 && fabs(gb - 2 * M_PI) < 1e-12);
  // boundary-edge geometry: conormals point out of the square, lengths sum to the perimeter
  double per = 0.0, badnu = 0.0;
  for (int h = 0; h < o->n_half_edges; ++h)
  {
    if (!MFTLHeIsBoundary(&o->half_edges[h]) || o->triangles[o->half_edges[h].triangle_idx].is_ghost) continue;
    double len, t[3], nu[3], X[3];
    if (MFTLBoundaryEdgeGeometry(o, h, &len, t, nu, X) != MFTL_SUCCESS) continue;
    per += len;
    const double cx = X[0] - 0.5 * s, cy = X[1] - 0.5 * s;  // midpoint relative to the centre
    if (nu[0] * cx + nu[1] * cy <= 0.0 || fabs(nu[2]) > 1e-14) badnu += 1.0;
  }
  double gper, gbadnu;
  MPI_Allreduce(&per, &gper, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&badnu, &gbadnu, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  CHECK(fabs(gper - 3.2 * s) <= 1e-13 * s && gbadnu == 0.0);
  if (g_rank == 0) printf("  planar patch (scale %g): B %ld chi %ld area %.15g corners %d GB %.3e\n", s, c.B, euler(&c), c.area, g[3], gb - 2 * M_PI);
  MFTLEnsembleDestroy(&ens);
}

/* ------------------------------------------------------------------------------------------
 * C2: hemisphere convergence, Gauss-Bonnet for several topologies (+ C6)
 * ------------------------------------------------------------------------------------------ */
static double film_z(double x, double y) { return 0.5 + 0.03 * sin(2 * M_PI * x) * sin(2 * M_PI * y); }

/* hemisphere z >= c_z from a subdivided octahedron (near-equilateral triangles, the rim is an
   exact great circle in the plane z = c_z); level l gives 4 * 2^l rim edges */
static Mesh octa_cap(const double c[3], double r, int level)
{
  int cap_v = 4096 * 16, cap_t = 4 * (1 << (2 * level)) + 8;
  double(*U)[3] = malloc((size_t) cap_v * sizeof(*U));
  int nu = 0;
  int(*F)[3] = malloc((size_t) cap_t * sizeof(*F));
  int nf = 0;
  const double base[5][3] = {{1, 0, 0}, {0, 1, 0}, {-1, 0, 0}, {0, -1, 0}, {0, 0, 1}};
  for (int i = 0; i < 5; ++i) memcpy(U[nu++], base[i], sizeof(U[0]));
  for (int i = 0; i < 4; ++i) F[nf][0] = i, F[nf][1] = (i + 1) % 4, F[nf][2] = 4, nf++;
  for (int l = 0; l < level; ++l)
  {
    int(*G)[3] = malloc((size_t) (4 * nf) * sizeof(*G));
    int ng = 0;
    int(*emap)[3] = malloc((size_t) (3 * nf) * sizeof(*emap));  // (a, b, mid)
    int ne = 0;
    for (int f = 0; f < nf; ++f)
    {
      int mid[3];
      for (int k = 0; k < 3; ++k)
      {
        int a = F[f][k], b = F[f][(k + 1) % 3], lo = a < b ? a : b, hi = a < b ? b : a, found = -1;
        for (int q = 0; q < ne; ++q)
          if (emap[q][0] == lo && emap[q][1] == hi) found = emap[q][2];
        if (found < 0)
        {
          double m[3] = {U[a][0] + U[b][0], U[a][1] + U[b][1], U[a][2] + U[b][2]};
          double len = sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]);
          for (int d = 0; d < 3; ++d) U[nu][d] = m[d] / len;
          if (fabs(U[a][2]) < 1e-15 && fabs(U[b][2]) < 1e-15) U[nu][2] = 0.0;
          found = nu++;
          emap[ne][0] = lo, emap[ne][1] = hi, emap[ne][2] = found, ne++;
        }
        mid[k] = found;
      }
      int a = F[f][0], b = F[f][1], cc = F[f][2];
      G[ng][0] = a, G[ng][1] = mid[0], G[ng][2] = mid[2], ng++;
      G[ng][0] = mid[0], G[ng][1] = b, G[ng][2] = mid[1], ng++;
      G[ng][0] = mid[2], G[ng][1] = mid[1], G[ng][2] = cc, ng++;
      G[ng][0] = mid[0], G[ng][1] = mid[1], G[ng][2] = mid[2], ng++;
    }
    free(F);
    free(emap);
    F = G;
    nf = ng;
  }
  Mesh M;
  M.nv = nu;
  M.nt = nf;
  M.P = malloc((size_t) nu * sizeof(*M.P));
  M.T = malloc((size_t) nf * sizeof(*M.T));
  for (int i = 0; i < nu; ++i)
    for (int k = 0; k < 3; ++k) M.P[i][k] = c[k] + r * U[i][k];
  memcpy(M.T, F, (size_t) nf * sizeof(*F));
  free(U);
  free(F);
  return M;
}

static Mesh film_mesh(int N)
{
  Mesh M;
  M.nv = N * N;
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) (2 * N * N) * sizeof(*M.T));
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      double x = (double) i / N, y = (double) j / N;
      M.P[j * N + i][0] = x, M.P[j * N + i][1] = y, M.P[j * N + i][2] = film_z(x, y);
    }
  M.nt = 0;
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      int a = j * N + i, b = j * N + (i + 1) % N, c = ((j + 1) % N) * N + (i + 1) % N, d = ((j + 1) % N) * N + i;
      M.T[M.nt][0] = a, M.T[M.nt][1] = b, M.T[M.nt][2] = c, M.nt++;
      M.T[M.nt][0] = a, M.T[M.nt][1] = c, M.T[M.nt][2] = d, M.nt++;
    }
  return M;
}

static void test_gb(double s)
{
  const double R = 0.3 * s;
  double errL[4], errA[4], errK[4];
  int levels[4] = {2, 3, 4, 5};  // octahedral cap: 16, 32, 64, 128 rim edges
  for (int l = 0; l < 4; ++l)
  {
    MFTL_Ensemble* ens = ens_new(0.0, s, 0, 0, 0);
    double c[3] = {0.5 * s, 0.5 * s, 0.2 * s};
    Mesh M = octa_cap(c, R, levels[l]);
    MFTL_Object* o = body(ens, 1, &M, 1);
    mesh_free(&M);
    Counts cn = counts(o);
    CHECK(euler(&cn) == 1);
    int bad;
    double gb = gauss_bonnet(o, &bad);
    // exact identity; round-off of the angle sums grows with the vertex count
    CHECK(bad == 0 && fabs(gb - 2 * M_PI) < 1e-12 + 1e-15 * (double) cn.V);
    MFTL_BoundaryLoopSet L;
    CHECK(loops(o, &L) == 1);
    errL[l] = (L.n_loops == 1) ? fabs(L.loops[0].length - 2 * M_PI * R) / (2 * M_PI * R) : 1.0;
    MFTLFreeBoundaryLoops(&L);
    errA[l] = fabs(cn.area - 2 * M_PI * R * R) / (2 * M_PI * R * R);
    MFTLComputeVertexCurvatures(o);
    MFTLComputeExtrapolatedCurvatures(o);
    double ek = 0.0, ex = 0.0, ex_area = 0.0;
    int nk = 0, nx = 0, flag_bad = 0;
    for (int i = 0; i < o->n_vertices; ++i)
    {
      MFTL_Vertex* v = &o->vertices[i];
      if (v->is_ghost) continue;
      double th;
      if (MFTLVertexTurningAngle(o, i, &th) == MFTL_SUCCESS)
      {
        if (!(v->flags & MFTL_VFLAG_CURV_UNDEFINED)) flag_bad++;
        if (v->flags & MFTL_VFLAG_CURV_EXTRAPOLATED)
        {
          ex = fmax(ex, fabs(v->curv_extrap * R / 2.0 - 1.0));
          nx++;
        }
        continue;
      }
      // interior away from the rim: area-weighted mean (pointwise cotan curvature does not
      // converge on UV-sphere meshes; the Voronoi-area-weighted integral does)
      double dz = v->pos[2] - c[2];
      if (dz < 0.3 * R) continue;
      ek += v->meancurvature * v->area;
      nk++;
      ex_area += v->area;
    }
    double lk[2] = {ek, ex_area}, gk[2];
    MPI_Allreduce(lk, gk, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    errK[l] = fabs(gk[0] / fmax(gk[1], 1e-300) * R / 2.0 - 1.0);
    (void) nk;
    int gfb, gnx;
    double gex;
    MPI_Allreduce(&flag_bad, &gfb, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&nx, &gnx, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&ex, &gex, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    CHECK(gfb == 0);
    CHECK(gnx == 4 * (1 << levels[l]));  // every rim vertex has an extrapolated curvature
    if (l == 3) CHECK(gex < 0.05);  // |2H| -> 2/R at the rim too
    if (g_rank == 0) printf("  hemisphere level=%d (scale %g): loop err %.3e area err %.3e mean-curv err %.3e rim extrapolated err %.3e\n", levels[l], s, errL[l], errA[l], errK[l], gex);
    MFTLEnsembleDestroy(&ens);
  }
  for (int l = 1; l < 4; ++l)
  {
    const double rL = log(errL[l - 1] / errL[l]) / log(2.0), rA = log(errA[l - 1] / errA[l]) / log(2.0);
    CHECK(rL >= 1.9 && rA >= 1.9);
  }
  CHECK(errK[3] < 0.01);
  // Gauss-Bonnet for other topologies (chi from V - E + F): closed sphere 2, open tube 0, film 0
  for (int t = 0; t < 3; ++t)
  {
    MFTL_Ensemble* ens = ens_new(0.0, s, t == 2, t == 2, t == 2);
    Mesh M;
    int open = 0;
    if (t == 0)
    {
      double c[3] = {0.5 * s, 0.5 * s, 0.5 * s};
      M = uv_sphere(c, 0.3 * s, 16, 0);
    }
    else if (t == 1)
    {
      M = tube(0.5 * s, 0.5 * s, 0.2 * s, 12, 16, 0, 0.2 * s, 0.8 * s);
      open = 1;
    }
    else
    {
      M = film_mesh(24);  // closed film wrapping x and y
      for (int i = 0; i < M.nv; ++i)
        for (int k = 0; k < 3; ++k) M.P[i][k] *= s;
    }
    MFTL_Object* o = body(ens, 1, &M, open);
    mesh_free(&M);
    Counts cn = counts(o);
    const long chi = euler(&cn);
    CHECK(chi == (t == 0 ? 2 : 0));
    int bad;
    double gb = gauss_bonnet(o, &bad);
    CHECK(bad == 0 && fabs(gb - 2 * M_PI * chi) < 1e-11);
    MFTLEnsembleDestroy(&ens);
  }
}

/* ------------------------------------------------------------------------------------------
 * C3: augmented link condition (rank 0 builds; every result is broadcast)
 * ------------------------------------------------------------------------------------------ */
static MFTL_Object* local_mesh(MFTL_Ensemble* ens, const Mesh* M, int open)
{
  MFTL_Object* o = MFTLObjectCreate(90);
  MFTLEnsembleAddObject(ens, o);
  if (open) MFTLObjectDeclareOpen(o, 1);
  if (g_rank == 0)
  {
    int rc = MFTLBuildTopologyFromTriples(o, M->nv, (const double(*)[3]) M->P, M->nt, (const int(*)[3]) M->T);
    if (rc != MFTL_SUCCESS) fprintf(stderr, "local_mesh: build failed\n");
    MFTLComputeTriangleProperties(o);
    MFTLDetectFeatures(o, 45.0);
  }
  return o;
}

static void permissive(MFTL_RemeshParameters* p)
{
  remesh_params(p, 1.0);
  p->criterion = MFTL_CRITERION_BOTSCH_KOBBELT;
  p->min_edge_length = 1e3, p->min_edge_length_sq = 1e6;  // every edge is "short"
  p->max_edge_length = 1e3, p->max_edge_length_sq = 1e6;  // no stretch rejection
}

/* counts approved collapses among the half-edges selected by `sel`, then collapses the first one
   (if any) and checks the result's twin states */
typedef int (*Sel)(MFTL_Object* o, int h);
static int sel_all(MFTL_Object* o, int h)
{
  (void) o, (void) h;
  return 1;
}
static int sel_interior(MFTL_Object* o, int h) { return MFTLHeIsLocal(&o->half_edges[h]); }
static int sel_boundary(MFTL_Object* o, int h) { return MFTLHeIsBoundary(&o->half_edges[h]); }

static void link_case(const char* name, const Mesh* M, Sel sel, int expect_min, int expect_max, int do_collapse, double corner_angle)
{
  MFTL_Ensemble* ens = ens_new(-10, 10, 0, 0, 0);
  MFTL_Object* o = local_mesh(ens, M, 1);
  MFTL_RemeshParameters p;
  permissive(&p);
  p.boundary_corner_angle = corner_angle;
  int approved = 0, first = -1, ok = 1;
  if (g_rank == 0)
  {
    for (int h = 0; h < o->n_half_edges; ++h)
      if (sel(o, h) && CollapseApproved(o, h, &p))
      {
        approved++;
        if (first < 0) first = h;
      }
    if (do_collapse && first >= 0)
    {
      MFTLCollapseEdgeWithFit(o, first, &p);
      MFTLFinalizeRemeshing(o, o->n_vertices, o->n_triangles, o->n_half_edges, true);
      ok = MFTLCheckTwinStatesLocal(o, 0, 1) == 0;
    }
  }
  int buf[2] = {approved, ok};
  MPI_Bcast(buf, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  link %-38s approved %d (expected %d..%d)%s\n", name, buf[0], expect_min, expect_max, do_collapse ? (buf[1] ? ", result valid" : ", RESULT INVALID") : "");
  CHECK(buf[0] >= expect_min && buf[0] <= expect_max);
  CHECK(buf[1]);
  MFTLEnsembleDestroy(&ens);
}

static void test_link(void)
{
  // one-triangle-wide strip: 2 rows of 7 vertices
  Mesh S = grid(7, 2, 0.0, 0.0, 6.0, 1.0, zplane, 0, NULL, 0);
  link_case("strip: interior (pinching) edges", &S, sel_interior, 0, 0, 0, 0.0);
  link_case("strip: boundary edges (legal)", &S, sel_boundary, 1, 1000, 1, 0.0);
  mesh_free(&S);
  // ear: square split into (a,b,c), (a,c,d): collapsing ab (ab, bc boundary, ca interior) is legal
  Mesh E;
  E.nv = 4;
  E.nt = 2;
  E.P = malloc(4 * sizeof(*E.P));
  E.T = malloc(2 * sizeof(*E.T));
  double q[4][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
  memcpy(E.P, q, sizeof(q));
  E.T[0][0] = 0, E.T[0][1] = 1, E.T[0][2] = 2;
  E.T[1][0] = 0, E.T[1][1] = 2, E.T[1][2] = 3;
  // every vertex of the square is a 90-degree corner: raise the corner threshold so that the
  // topological rule alone is tested (corners are tested separately below)
  link_case("ear triangles (boundary edges legal)", &E, sel_boundary, 1, 8, 1, 179.0);
  E.nt = 1;  // isolated triangle
  E.nv = 3;
  link_case("isolated triangle (all rejected)", &E, sel_all, 0, 0, 0, 179.0);
  mesh_free(&E);
  // two single-cell holes one cell column apart: the edges joining them are rejected
  int holes2[4] = {1, 1, 3, 1};
  Mesh H = grid(6, 4, 0.0, 0.0, 5.0, 3.0, zplane, 0, holes2, 2);
  MFTL_Ensemble* ens = ens_new(-10, 10, 0, 0, 0);
  MFTL_Object* o = local_mesh(ens, &H, 1);
  MFTL_RemeshParameters p;
  permissive(&p);
  int bridge = 0, bridge_ok = 0;
  if (g_rank == 0)
    for (int h = 0; h < o->n_half_edges; ++h)
    {
      if (!MFTLHeIsLocal(&o->half_edges[h])) continue;
      const MFTL_Vertex *a = MFTLHeOrigin(o, &o->half_edges[h]), *b = MFTLHeTarget(o, &o->half_edges[h]);
      // horizontal edges x = 2 -> 3 at y = 1 or 2 join the two holes' loops
      if (fabs(a->pos[1] - b->pos[1]) < 1e-12 && (fabs(a->pos[1] - 1) < 1e-12 || fabs(a->pos[1] - 2) < 1e-12) &&
          fmin(a->pos[0], b->pos[0]) > 1.5 && fmax(a->pos[0], b->pos[0]) < 3.5)
      {
        bridge++;
        if (!CollapseApproved(o, h, &p)) bridge_ok++;
      }
    }
  int bb[2] = {bridge, bridge_ok};
  MPI_Bcast(bb, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  link %-38s %d bridge half-edges, %d rejected\n", "two holes one edge apart", bb[0], bb[1]);
  CHECK(bb[0] == 4 && bb[1] == 4);
  MFTLEnsembleDestroy(&ens);
  mesh_free(&H);
  // triangular hole: remove one triangle of a grid interior; its three edges never collapse
  Mesh G = grid(6, 6, 0.0, 0.0, 5.0, 5.0, zplane, 0, NULL, 0);
  int hole_t = -1;
  for (int t = 0; t < G.nt && hole_t < 0; ++t)
  {
    double cx = (G.P[G.T[t][0]][0] + G.P[G.T[t][1]][0] + G.P[G.T[t][2]][0]) / 3, cy = (G.P[G.T[t][0]][1] + G.P[G.T[t][1]][1] + G.P[G.T[t][2]][1]) / 3;
    if (cx > 2 && cx < 3 && cy > 2 && cy < 3) hole_t = t;
  }
  G.T[hole_t][0] = G.T[G.nt - 1][0], G.T[hole_t][1] = G.T[G.nt - 1][1], G.T[hole_t][2] = G.T[G.nt - 1][2];
  G.nt--;
  ens = ens_new(-10, 10, 0, 0, 0);
  o = local_mesh(ens, &G, 1);
  int hole_edges = 0, hole_rej = 0;
  if (g_rank == 0)
    for (int h = 0; h < o->n_half_edges; ++h)
    {
      if (!MFTLHeIsBoundary(&o->half_edges[h])) continue;
      const MFTL_Vertex* a = MFTLHeOrigin(o, &o->half_edges[h]);
      if (a->pos[0] < 1.5 || a->pos[0] > 3.5 || a->pos[1] < 1.5 || a->pos[1] > 3.5) continue;  // outer loop
      hole_edges++;
      if (!CollapseApproved(o, h, &p)) hole_rej++;
    }
  int hh[2] = {hole_edges, hole_rej};
  MPI_Bcast(hh, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  link %-38s %d hole edges, %d rejected\n", "3-edge triangular hole", hh[0], hh[1]);
  CHECK(hh[0] == 3 && hh[1] == 3);
  MFTLEnsembleDestroy(&ens);
  mesh_free(&G);
  // corners: no collapse of an edge incident to a corner of a square patch
  Mesh C = grid(5, 5, 0.0, 0.0, 4.0, 4.0, zplane, 0, NULL, 0);
  ens = ens_new(-10, 10, 0, 0, 0);
  o = local_mesh(ens, &C, 1);
  int corner_edges = 0, corner_rej = 0;
  if (g_rank == 0)
    for (int h = 0; h < o->n_half_edges; ++h)
    {
      const MFTL_Vertex *a = MFTLHeOrigin(o, &o->half_edges[h]), *b = MFTLHeTarget(o, &o->half_edges[h]);
      const MFTL_Vertex* vv[2] = {a, b};
      int corner = 0;
      for (int k = 0; k < 2; ++k)
        if ((fabs(vv[k]->pos[0]) < 1e-12 || fabs(vv[k]->pos[0] - 4) < 1e-12) && (fabs(vv[k]->pos[1]) < 1e-12 || fabs(vv[k]->pos[1] - 4) < 1e-12)) corner = 1;
      if (!corner) continue;
      corner_edges++;
      if (!CollapseApproved(o, h, &p)) corner_rej++;
    }
  int cc[2] = {corner_edges, corner_rej};
  MPI_Bcast(cc, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  link %-38s %d corner half-edges, %d rejected\n", "corners", cc[0], cc[1]);
  CHECK(cc[0] > 0 && cc[1] == cc[0]);
  MFTLEnsembleDestroy(&ens);
  mesh_free(&C);
}

/* ------------------------------------------------------------------------------------------
 * C4: boundary-preserving remeshing cycles
 * ------------------------------------------------------------------------------------------ */
static double g_R = 0.3, g_zc = 0.2, g_xc = 0.5, g_yc = 0.5;
static int circle_projector(int tag, const double X_in[3], double X_out[3], void* user)
{
  (void) tag, (void) user;
  double dx = X_in[0] - g_xc, dy = X_in[1] - g_yc, r = sqrt(dx * dx + dy * dy);
  if (!(r > 0.0)) return MFTL_ERROR;
  X_out[0] = g_xc + g_R * dx / r, X_out[1] = g_yc + g_R * dy / r, X_out[2] = g_zc;
  return MFTL_SUCCESS;
}

/* max over owned boundary vertices of the distance to the square boundary / the circle */
static double square_dev(MFTL_Object* o, double lo, double hi, double z, int* corners)
{
  double dev = 0.0;
  int nc = 0;
  for (int i = 0; i < o->n_vertices; ++i)
  {
    MFTL_Vertex* v = &o->vertices[i];
    if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
    int pr, nx;
    if (MFTLBoundaryNeighbours(o, i, &pr, &nx) != MFTL_SUCCESS) continue;
    const double x = v->pos[0], y = v->pos[1];
    double d = fmin(fmin(fabs(x - lo), fabs(x - hi)), fmin(fabs(y - lo), fabs(y - hi)));
    if (x < lo - 1e-15 || x > hi + 1e-15 || y < lo - 1e-15 || y > hi + 1e-15) d = 1.0;
    dev = fmax(dev, fmax(d, fabs(v->pos[2] - z)));
    if ((x == lo || x == hi) && (y == lo || y == hi)) nc++;  // bit-exact corners
  }
  double g;
  MPI_Allreduce(&dev, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nc, corners, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return g;
}

static double circle_dev(MFTL_Object* o)
{
  double dev = 0.0;
  for (int i = 0; i < o->n_vertices; ++i)
  {
    MFTL_Vertex* v = &o->vertices[i];
    if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
    int pr, nx;
    if (MFTLBoundaryNeighbours(o, i, &pr, &nx) != MFTL_SUCCESS) continue;
    double dx = v->pos[0] - g_xc, dy = v->pos[1] - g_yc;
    dev = fmax(dev, fmax(fabs(sqrt(dx * dx + dy * dy) - g_R), fabs(v->pos[2] - g_zc)));
  }
  double g;
  MPI_Allreduce(&dev, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return g;
}

static void test_cycles_planar(double s)
{
  MFTL_Ensemble* ens = ens_new(0.0, s, 0, 0, 0);
  g_zplane = 0.5 * s;
  Mesh M = grid(9, 9, 0.1 * s, 0.1 * s, 0.8 * s, 0.8 * s, zplane, 0, NULL, 0);
  MFTL_Object* o = body(ens, 1, &M, 1);
  mesh_free(&M);
  // an application corner tag on a straight edge (x = 0.5 on y = 0.1): never moved or collapsed
  const double tagged[3] = {0.5 * s, 0.1 * s, 0.5 * s};
  for (int i = 0; i < o->n_vertices; ++i)
    if (fabs(o->vertices[i].pos[0] - tagged[0]) <= 1e-15 * s && fabs(o->vertices[i].pos[1] - tagged[1]) <= 1e-15 * s) o->vertices[i].flags |= MFTL_VFLAG_CORNER;
  const long B0 = counts(o).B;
  long Bmin = B0, Bmax = B0;
  int fails = 0, last_tris = 0;
  for (int cyc = 0; cyc < 20; ++cyc)
  {
    MFTL_RemeshParameters p;
    remesh_params(&p, ((cyc % 4) < 2 ? 0.05 : 0.25) * s);
    p.skip_volume_recovery = 0;  // open body: the class guard must skip recovery (no abort)
    remesh(ens, &p);
    Counts c = counts(o);
    Bmin = c.B < Bmin ? c.B : Bmin;
    Bmax = c.B > Bmax ? c.B : Bmax;
    int corners;
    const double dev = square_dev(o, 0.1 * s, 0.9 * s, 0.5 * s, &corners);
    MFTL_BoundaryLoopSet L;
    const int nl = loops(o, &L);
    const int tag_ok = (nl == 1) && L.loops[0].tag == 0 && !L.loops[0].mixed_tags;
    MFTLFreeBoundaryLoops(&L);
    int tag_here = 0, gtag = 0;
    for (int i = 0; i < o->n_vertices; ++i)
      if (!o->vertices[i].is_ghost && !(o->vertices[i].flags & MFTL_FLAG_DEAD) && o->vertices[i].out_he_idx >= 0 && (o->vertices[i].flags & MFTL_VFLAG_CORNER) &&
          memcmp(o->vertices[i].pos, tagged, sizeof(tagged)) == 0)
        tag_here++;
    MPI_Allreduce(&tag_here, &gtag, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const int ok = valid(o) && dev <= 1e-12 * s && corners == 4 && gtag == 1 && fabs(c.area - 0.64 * s * s) <= 1e-12 * s * s && nl == 1 && tag_ok &&
                   euler(&c) == 1;
    if (!ok)
    {
      fails++;
      if (g_rank == 0) fprintf(stderr, "planar cycle %d (scale %g): dev %.3e corners %d tagged %d area %.15g loops %d chi %ld\n", cyc, s, dev, corners, gtag, c.area, nl, euler(&c));
    }
    last_tris = (int) c.F;
  }
  if (g_rank == 0) printf("  planar cycles (scale %g, np%d): boundary edges %ld -> [%ld, %ld], final triangles %d, fails %d\n", s, g_np, B0, Bmin, Bmax, last_tris, fails);
  CHECK(fails == 0);
  CHECK(Bmin < B0 && Bmax > B0);  // boundary loops coarsen and refine (16A S-A4)
  MFTLEnsembleDestroy(&ens);
}

static void test_cycles_hemisphere(int constrained)
{
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
  double c[3] = {g_xc, g_yc, g_zc};
  Mesh M = uv_sphere(c, g_R, 16, 1);
  // perturb the interior radially (1 %), deterministic
  for (int i = 0; i < M.nv; ++i)
  {
    if (fabs(M.P[i][2] - g_zc) < 1e-15) continue;
    double f = 1.0 + 0.01 * sin(12.9898 * i);
    for (int k = 0; k < 3; ++k) M.P[i][k] = c[k] + f * (M.P[i][k] - c[k]);
  }
  MFTL_Object* o = body(ens, 1, &M, 1);
  mesh_free(&M);
  if (constrained) CHECK(MFTLObjectSetBoundaryCurve(o, 0, MFTL_BCURVE_CONSTRAINED, circle_projector, NULL, 0) == MFTL_SUCCESS);
  const double h0 = 2 * M_PI * g_R / 32.0;
  int fails = 0;
  double worst = 0.0;
  for (int cyc = 0; cyc < 20; ++cyc)
  {
    MFTL_RemeshParameters p;
    remesh_params(&p, (cyc % 2) ? 0.04 : 0.07);
    remesh(ens, &p);
    const double dev = circle_dev(o);
    worst = fmax(worst, dev);
    if (g_rank == 0 && getenv("S16C_VERBOSE")) printf("    cycle %d rim deviation %.3e\n", cyc, dev);
    MFTL_BoundaryLoopSet L;
    const int nl = loops(o, &L);
    MFTLFreeBoundaryLoops(&L);
    if (!valid(o) || nl != 1) fails++;
  }
  // REPRESENTED: interpolation error C h^3 with h the longest boundary edge the cycles produce
  // (Botsch-Kobbelt: 4/3 of the coarsest target) and C = 1 / R^2
  const double hmax = 4.0 / 3.0 * 0.07;
  const double bound = constrained ? 1e-12 : hmax * hmax * hmax / (g_R * g_R);
  (void) h0;
  if (g_rank == 0) printf("  hemisphere cycles (%s, np%d): worst rim deviation %.3e (bound %.3e), fails %d\n", constrained ? "CONSTRAINED" : "REPRESENTED", g_np, worst, bound, fails);
  CHECK(fails == 0);
  CHECK(worst <= bound);
  MFTLEnsembleDestroy(&ens);
}

/* S-A5: an interior collapse with a boundary survivor keeps the survivor in place (rank 0) */
static void test_sa5(void)
{
  Mesh G = grid(5, 5, 0.0, 0.0, 4.0, 4.0, zplane, 0, NULL, 0);
  MFTL_Ensemble* ens = ens_new(-10, 10, 0, 0, 0);
  MFTL_Object* o = local_mesh(ens, &G, 1);
  MFTL_RemeshParameters p;
  permissive(&p);
  int done = 0, kept = 0;
  if (g_rank == 0)
    for (int h = 0; h < o->n_half_edges && !done; ++h)
    {
      if (!MFTLHeIsLocal(&o->half_edges[h])) continue;
      MFTL_Vertex* b = MFTLHeTarget(o, &o->half_edges[h]);
      int pr, nx;
      if (MFTLBoundaryNeighbours(o, (int) (b - o->vertices), &pr, &nx) != MFTL_SUCCESS) continue;
      if (!CollapseApproved(o, h, &p)) continue;
      double before[3];
      memcpy(before, b->pos, sizeof(before));
      const int vb = (int) (b - o->vertices);
      MFTLCollapseEdgeWithFit(o, h, &p);
      kept = memcmp(before, o->vertices[vb].pos, sizeof(before)) == 0;
      done = 1;
    }
  int bb[2] = {done, kept};
  MPI_Bcast(bb, 2, MPI_INT, 0, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  S-A5 interior collapse onto a boundary survivor: performed %d, survivor bit-fixed %d\n", bb[0], bb[1]);
  CHECK(bb[0] == 1 && bb[1] == 1);
  MFTLEnsembleDestroy(&ens);
  mesh_free(&G);
}

/* ------------------------------------------------------------------------------------------
 * C5: seam-crossing open patch vs translated (dyadic)
 * ------------------------------------------------------------------------------------------ */
static double seam_z(double x, double y)
{
  // dyadic-friendly smooth height, evaluated on the untranslated coordinates
  double u = x - 0.5, v = y - 0.5;
  return 0.5 + 0.25 * (u * u - 0.5 * v * v);
}

typedef struct
{
  int n;
  double* d;  // per gid: 12 doubles (normal, meancurv, flags, turning, area share...)
} PerGid;

static void per_gid(MFTL_Object* o, int nmax, double* out)
{
  // out[gid*6 + k]: normal (3), meancurvature, curvature flag, turning angle (0 interior)
  double* loc = calloc((size_t) (6 * nmax), sizeof(double));
  for (int i = 0; i < o->n_vertices; ++i)
  {
    MFTL_Vertex* v = &o->vertices[i];
    if (v->is_ghost || v->global_id < 0 || v->global_id >= nmax) continue;
    double* r = &loc[6 * v->global_id];
    memcpy(r, v->normal, 3 * sizeof(double));
    r[3] = v->meancurvature;
    r[4] = (v->flags & MFTL_VFLAG_CURV_UNDEFINED) ? 1.0 : 0.0;
    double th = 0.0;
    MFTLVertexTurningAngle(o, i, &th);
    r[5] = th;
  }
  MPI_Allreduce(loc, out, 6 * nmax, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  free(loc);
}

static void test_seam(void)
{
  const int n = 17;
  double shift[2][2] = {{0.0, 0.0}, {0.5, 0.5}};
  double *pg[2] = {NULL, NULL}, len[2], area[2];
  long ops[2][3], tris[2];
  int nl[2];
  for (int s = 0; s < 2; ++s)
  {
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
    Mesh M = grid(n, n, 0.25, 0.25, 0.5, 0.5, seam_z, 0, NULL, 0);
    for (int i = 0; i < M.nv; ++i) M.P[i][0] += shift[s][0], M.P[i][1] += shift[s][1];
    MFTL_Object* o = body(ens, 1, &M, 1);
    mesh_free(&M);
    CHECK(valid(o));
    MFTL_BoundaryLoopSet L;
    nl[s] = loops(o, &L);
    len[s] = nl[s] == 1 ? L.loops[0].length : -1;
    if (nl[s] == 1) CHECK(L.loops[0].winding[0] == 0 && L.loops[0].winding[1] == 0 && L.loops[0].winding[2] == 0);
    MFTLFreeBoundaryLoops(&L);
    area[s] = counts(o).area;
    MFTLComputeVertexNormals(o);
    MFTLComputeVertexCurvatures(o);
    pg[s] = malloc((size_t) (6 * n * n) * sizeof(double));
    per_gid(o, n * n, pg[s]);
    // one remesh pass through the serial kernel (gathered: rank-independent ordering on np1)
    MFTL_RemeshParameters p;
    remesh_params(&p, 0.02);
    p.skip_volume_recovery = 1;
    if (g_np == 1)
      MFTLRemeshEnsemble(ens, &p);
    else
      for (int it = 0; it < 12; ++it)  // parallel sweeps: compare converged remeshing (C7)
      {
        MFTLRemeshEnsembleParallel(ens, &p);
        MFTLComputeTriangleProperties(o);
        MFTLRedistribute(ens);
      }
    MFTLRedistribute(ens);
    CHECK(valid(o));
    long lo[3] = {p.stat_splits, p.stat_collapses, p.stat_flips}, go[3];
    MPI_Allreduce(lo, go, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    memcpy(ops[s], go, sizeof(go));
    tris[s] = counts(o).F;
    MFTLEnsembleDestroy(&ens);
  }
  CHECK(nl[0] == 1 && nl[1] == 1);
  CHECK(fabs(len[0] - len[1]) <= 1e-14 && fabs(area[0] - area[1]) <= 1e-14);
  double worst = 0.0;
  for (int k = 0; k < 6 * n * n; ++k) worst = fmax(worst, fabs(pg[0][k] - pg[1][k]));
  if (g_rank == 0)
    printf("  seam patch: loop length %.15g / %.15g, area diff %.1e, per-vertex normal/curvature/flag/turning max diff %.1e, ops (%ld,%ld,%ld) / (%ld,%ld,%ld), triangles %ld / %ld\n",
           len[0], len[1], fabs(area[0] - area[1]), worst, ops[0][0], ops[0][1], ops[0][2], ops[1][0], ops[1][1], ops[1][2], tris[0], tris[1]);
  CHECK(worst <= 1e-12);
  if (g_np == 1)
    CHECK(ops[0][0] == ops[1][0] && ops[0][1] == ops[1][1] && ops[0][2] == ops[1][2] && tris[0] == tris[1]);
  else
    CHECK(labs(tris[0] - tris[1]) <= 5 * tris[0] / 100);  // statistical (different decompositions; observed <= 3.5 %)
  free(pg[0]);
  free(pg[1]);
}

/* ------------------------------------------------------------------------------------------
 * periodic bodies: ribbon, film, seam sphere (parity), tube (set-level recovery), proximity
 * ------------------------------------------------------------------------------------------ */
static double ribbon_z(double x, double y)
{
  (void) y;
  return 0.5 + 0.05 * sin(2 * M_PI * x);
}

static void test_ribbon(void)
{
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
  Mesh M = grid(24, 7, 0.0, 0.3, 1.0, 0.4, ribbon_z, 1, NULL, 0);
  MFTL_Object* o = body(ens, 1, &M, 1);
  mesh_free(&M);
  int w[3];
  MFTLObjectWrapFlags(o, w);
  CHECK(w[0] == 1 && w[1] == 0 && w[2] == 0);
  int fails = 0;
  for (int cyc = 0; cyc < 10; ++cyc)
  {
    MFTL_RemeshParameters p;
    remesh_params(&p, (cyc % 2) ? 0.03 : 0.08);
    p.skip_volume_recovery = 0;
    remesh(ens, &p);
    MFTL_BoundaryLoopSet L;
    const int nl = loops(o, &L);
    int wind_ok = (nl == 2);
    for (int k = 0; k < nl && wind_ok; ++k) wind_ok = abs(L.loops[k].winding[0]) == 1 && L.loops[k].winding[1] == 0 && L.loops[k].winding[2] == 0;
    if (nl == 2) wind_ok = wind_ok && (L.loops[0].winding[0] == -L.loops[1].winding[0]);
    MFTLFreeBoundaryLoops(&L);
    int cut[3];
    MFTLCountCutEdges(o, cut);
    if (!valid(o) || !wind_ok || cut[1] || cut[2] || MFTLCheckWrapFlags(o, 1) != MFTL_SUCCESS) fails++;
  }
  if (g_rank == 0) printf("  wrapping ribbon: 10 cycles, loop windings +-(1,0,0) kept, fails %d\n", fails);
  CHECK(fails == 0);
  MFTLEnsembleDestroy(&ens);
}

static void test_film(void)
{
  for (int mode = 0; mode < 2; ++mode)  // 0: default (parity), 1: generalized opt-in
  {
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
    Mesh M = film_mesh(20);
    MFTL_Object* o = body(ens, 1, &M, 0);
    mesh_free(&M);
    if (mode) MFTLObjectSetRemeshMode(o, MFTL_REMESH_MODE_GENERALIZED);
    int fails = 0;
    long skipped = 0;
    for (int cyc = 0; cyc < 6; ++cyc)
    {
      MFTL_RemeshParameters p;
      remesh_params(&p, (cyc % 2) ? 0.035 : 0.07);
      p.mftl1_criteria = (mode == 0);
      p.skip_volume_recovery = 0;  // undefined volume (Phi_z != 0): skipped, never MFTLObjectVolume
      remesh(ens, &p);
      skipped += p.stat_recovery_skipped;
      int w[3];
      MFTLObjectWrapFlags(o, w);
      int cut[3];
      MFTLCountCutEdges(o, cut);
      if (!valid(o) || !(w[0] && w[1] && !w[2]) || cut[2] || MFTLCheckWrapFlags(o, 1) != MFTL_SUCCESS) fails++;
    }
    long gsk;
    MPI_Allreduce(&skipped, &gsk, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    if (g_rank == 0) printf("  closed film (%s): 6 cycles, recovery skipped %ld times, fails %d\n", mode ? "generalized" : "parity", gsk, fails);
    CHECK(fails == 0 && gsk > 0);
    MFTLEnsembleDestroy(&ens);
  }
}

static void test_seam_sphere_parity(void)
{
  // a sphere at mid-domain and its dyadic translation (0.5, 0.5, 0.5) onto the xyz corner, under
  // parity remeshing (closed body, default mode). exact = 1: vertex coordinates on a 2^-40 grid and
  // midpoint placement (no fits), so every decision is bitwise reproducible (plans/16 §10
  // conventions); exact = 0: with vertex fits (statistical comparison).
  for (int exact = 1; exact >= 0; --exact)
  {
    double V[2], A[2];
    long T[2], ops[2][3];
    for (int s = 0; s < 2; ++s)
    {
      MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
      double c[3] = {0.5, 0.5, 0.5};
      Mesh M = uv_sphere(c, 0.25, 16, 0);
      for (int i = 0; i < M.nv; ++i)
        for (int k = 0; k < 3; ++k)
        {
          if (exact) M.P[i][k] = ldexp(nearbyint(ldexp(M.P[i][k], 40)), -40);
          if (s) M.P[i][k] += 0.5;
        }
      MFTL_Object* o = body(ens, 1, &M, 0);
      mesh_free(&M);
      const double V0 = MFTLObjectVolume(o);
      MFTL_RemeshParameters p;
      remesh_params(&p, 0.05);
      p.mftl1_criteria = 1;
      p.skip_volume_recovery = 0;
      if (exact) p.vertex_fit = MFTL_FIT_NONE, p.skip_tangential_smoothing = 1;
      if (g_np == 1)
        MFTLRemeshEnsemble(ens, &p);
      else
        for (int it = 0; it < 4; ++it)
        {
          MFTLRemeshEnsembleParallel(ens, &p);
          MFTLComputeTriangleProperties(o);
          MFTLRedistribute(ens);
        }
      MFTLRedistribute(ens);
      CHECK(valid(o));
      int cut[3];
      MFTLCountCutEdges(o, cut);
      CHECK(cut[0] == 0 && cut[1] == 0 && cut[2] == 0);
      V[s] = MFTLObjectVolume(o);
      // recovered (serial 1e-10; the parallel parity path leaves a pre-existing residual of up to a
      // few 1e-7, bit-identical to Stage 16B: plans/16 §22)
      CHECK(fabs(V[s] - V0) <= (g_np == 1 ? 1e-10 : 1e-6) * V0);
      A[s] = counts(o).area;
      T[s] = counts(o).F;
      long lo[3] = {p.stat_splits, p.stat_collapses, p.stat_flips};
      MPI_Allreduce(lo, ops[s], 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      MFTLEnsembleDestroy(&ens);
    }
    if (g_rank == 0)
      printf("  seam sphere (parity, %s): V %.15g / %.15g, area %.15g / %.15g, triangles %ld / %ld, ops (%ld,%ld,%ld) / (%ld,%ld,%ld)\n",
             exact ? "dyadic, midpoints" : "with fits", V[0], V[1], A[0], A[1], T[0], T[1], ops[0][0], ops[0][1], ops[0][2], ops[1][0], ops[1][1], ops[1][2]);
    if (exact && g_np == 1)
    {
      CHECK(T[0] == T[1] && ops[0][0] == ops[1][0] && ops[0][1] == ops[1][1] && ops[0][2] == ops[1][2]);
      CHECK(fabs(V[0] - V[1]) <= 1e-13 * V[0] && fabs(A[0] - A[1]) <= 1e-13 * A[0]);
    }
    else
    {
      CHECK(labs(T[0] - T[1]) <= 5 * T[0] / 100 + 2);  // statistical (observed <= 3.1 %)
      CHECK(fabs(V[0] - V[1]) <= (g_np == 1 ? 1e-10 : 1e-6) * V[0] && fabs(A[0] - A[1]) <= 1e-3 * A[0]);
    }
  }
}

static void test_tube_recovery(void)
{
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
  Mesh M = tube(0.5, 0.5, 0.15, 24, 24, 1, 0.0, 1.0);
  MFTL_Object* o = body(ens, 1, &M, 0);
  mesh_free(&M);
  CHECK(!MFTLObjectHasVolume(o));  // wraps x
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  double V0 = 0.0, V1 = 0.0;
  CHECK(MFTLPhaseSetVolume(S, &V0) == MFTL_DEFINED);
  MFTL_RemeshParameters p;
  remesh_params(&p, 0.03);
  p.skip_volume_recovery = 0;
  remesh(ens, &p);
  CHECK(valid(o));
  CHECK(MFTLPhaseSetVolume(S, &V1) == MFTL_DEFINED);
  if (g_rank == 0) printf("  closed tube wrapping x: set-level recovery V %.15g -> %.15g (rel %.1e)\n", V0, V1, fabs(V1 - V0) / V0);
  CHECK(fabs(V1 - V0) <= 1e-10 * V0);
  MFTLPhaseSetDestroy(&S);
  MFTLEnsembleDestroy(&ens);
}

static void test_proximity(void)
{
  // two drops facing each other across the x seam; and a thin sheet pair across the z seam
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
  double ca[3] = {0.9, 0.5, 0.5}, cb[3] = {0.1, 0.5, 0.5};
  Mesh A = uv_sphere(ca, 0.08, 8, 0), Bm = uv_sphere(cb, 0.08, 8, 0);
  MFTL_Object* a = body(ens, 1, &A, 0);
  MFTL_Object* b = body(ens, 2, &Bm, 0);
  mesh_free(&A);
  mesh_free(&Bm);
  // the searches are rank-local (as the adapter's): bring both bodies to rank 0 first
  MFTL_SerialBracket br;
  MFTLEnsembleBeginSerial(ens, &br);
  int *pairs = NULL, np = 0;
  MFTLFindProximityPairs(a, b, 0.06, &pairs, &np);  // gap across the seam: 0.2 - 0.16 = 0.04
  free(pairs);
  int gnp;
  MPI_Allreduce(&np, &gnp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MFTLFindProximityPairs(a, b, 0.02, &pairs, &np);
  free(pairs);
  int gnp2;
  MPI_Allreduce(&np, &gnp2, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MFTLEnsembleEndSerial(ens, &br, 1);
  if (g_rank == 0) printf("  proximity across the x seam: %d pairs within 0.06, %d within 0.02\n", gnp, gnp2);
  CHECK(gnp > 0 && gnp2 == 0);
  MFTLEnsembleDestroy(&ens);
  // thin region: a closed film pinched across z (two sheets 0.02 apart through the z seam)
  ens = ens_new(0.0, 1.0, 1, 1, 1);
  double c[3] = {0.5, 0.5, 0.0};
  Mesh D = uv_sphere(c, 0.2, 16, 0);
  for (int i = 0; i < D.nv; ++i) D.P[i][2] = c[2] + 0.05 * (D.P[i][2] - c[2]);  // flattened drop around z = 0 (0.02 thick)
  MFTL_Object* d = body(ens, 3, &D, 0);
  mesh_free(&D);
  MFTLFindThinRegionPairs(d, 0.03, &pairs, &np);
  free(pairs);
  MPI_Allreduce(&np, &gnp, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("  thin region across the z seam: %d pairs\n", gnp);
  CHECK(gnp > 0);
  MFTLEnsembleDestroy(&ens);
}

/* B1 at scales: volume of a seam-crossing sphere */
static void test_b1_scale(double s)
{
  MFTL_Ensemble* ens = ens_new(0.0, s, 1, 1, 1);
  double c[3] = {0.95 * s, 0.02 * s, 0.5 * s};
  Mesh M = uv_sphere(c, 0.2 * s, 16, 0);
  double V = 0;
  for (int t = 0; t < M.nt; ++t)
  {
    double *p = M.P[M.T[t][0]], *q = M.P[M.T[t][1]], *r = M.P[M.T[t][2]];
    V += p[0] * (q[1] * r[2] - q[2] * r[1]) + q[0] * (r[1] * p[2] - r[2] * p[1]) + r[0] * (p[1] * q[2] - p[2] * q[1]);
  }
  V /= 6.0;
  MFTL_Object* o = body(ens, 1, &M, 0);
  mesh_free(&M);
  const double Vo = MFTLObjectVolume(o);
  if (g_rank == 0) printf("  B1 seam sphere at scale %g: V rel err %.1e\n", s, fabs(Vo - V) / V);
  CHECK(fabs(Vo - V) <= 1e-12 * V);
  MFTLEnsembleDestroy(&ens);
}


/* Kuprat / TSUR never move boundary vertices (plans/16 §4.4.2); class transitions are reported */
static double hill_z(double x, double y) { return 0.5 + 0.1 * exp(-20.0 * ((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5))); }

static void test_vcs_and_class(void)
{
  for (int mode = 0; mode < 2; ++mode)
  {
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
    Mesh M = grid(13, 13, 0.2, 0.2, 0.6, 0.6, hill_z, 0, NULL, 0);
    for (int i = 0; i < M.nv; ++i)  // roughen the interior
      if (M.P[i][0] > 0.21 && M.P[i][0] < 0.79 && M.P[i][1] > 0.21 && M.P[i][1] < 0.79) M.P[i][2] += 0.01 * sin(7.3 * i);
    MFTL_Object* o = body(ens, 1, &M, 1);
    mesh_free(&M);
    const int nv = o->n_vertices;
    double(*before)[3] = malloc((size_t) (nv > 0 ? nv : 1) * sizeof(*before));
    int* isb = calloc((size_t) (nv > 0 ? nv : 1), sizeof(int));
    for (int i = 0; i < nv; ++i)
    {
      memcpy(before[i], o->vertices[i].pos, sizeof(before[i]));
      int pr, nx;
      isb[i] = MFTLBoundaryNeighbours(o, i, &pr, &nx) == MFTL_SUCCESS;
    }
    MFTL_RemeshParameters p;
    remesh_params(&p, 0.05);
    p.max_smoothing_laplacian_iter = 3;
    p.laplacian_smoothing_under_relaxation = 1.0;
    p.feature_angle_threshold = 80.0;
    MFTLComputeTriangleProperties(o);
    if (mode == 0)
      MFTLVolumeConservingSmoothingParallel(o, &p);
    else
      MFTLTSUR3DParallel(o, &p);
    int moved_b = 0, moved_i = 0;
    for (int i = 0; i < nv; ++i)
    {
      if (o->vertices[i].is_ghost) continue;
      const int m = memcmp(before[i], o->vertices[i].pos, sizeof(before[i])) != 0;
      if (isb[i] && m) moved_b++;
      if (!isb[i] && m) moved_i++;
    }
    int g[2], l[2] = {moved_b, moved_i};
    MPI_Allreduce(l, g, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("  %s on an open patch: %d boundary vertices moved, %d interior vertices moved\n", mode ? "TSUR" : "Kuprat", g[0], g[1]);
    CHECK(g[0] == 0 && g[1] > 0);
    free(before);
    free(isb);
    MFTLEnsembleDestroy(&ens);
  }
  // class transition: a wrap-flag change found by a canonicalisation is reported once
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 1);
  Mesh T = tube(0.5, 0.5, 0.15, 12, 12, 1, 0.0, 1.0);
  MFTL_Object* o = body(ens, 1, &T, 0);
  mesh_free(&T);
  CHECK(MFTLObjectTakeClassTransition(o) == 0);
  o->wrap[0] = 0;  // pretend the previous state was compact
  MFTLCanonicaliseImages(o);
  CHECK(MFTLObjectTakeClassTransition(o) == 1);
  CHECK(MFTLObjectTakeClassTransition(o) == 0);
  MFTLEnsembleDestroy(&ens);
}

/* 16C review: valence growth. An open wheel of 30 spokes: every split of a rim (boundary) edge
   adds a neighbour to the centre (valence 30). Before the review fix, refinement raised it past
   the flip gate's 32-entry ring buffer (assertion in Debug, stack overflow in Release; first
   observed in 16-cycle coarsen/refine runs on 8 ranks). Generalized splits and flips never raise
   a valence above 24, and flips still reduce higher ones. */
static void test_valence_growth(void)
{
  const int N = 30;
  Mesh M;
  M.nv = N + 1;
  M.nt = N;
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) M.nt * sizeof(*M.T));
  M.P[0][0] = 0.5, M.P[0][1] = 0.5, M.P[0][2] = 0.5;
  for (int i = 0; i < N; ++i)
  {
    M.P[1 + i][0] = 0.5 + 0.3 * cos(2 * M_PI * i / N);
    M.P[1 + i][1] = 0.5 + 0.3 * sin(2 * M_PI * i / N);
    M.P[1 + i][2] = 0.5;
    M.T[i][0] = 0, M.T[i][1] = 1 + i, M.T[i][2] = 1 + (i + 1) % N;
  }
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
  MFTL_Object* o = body(ens, 1, &M, 1);
  mesh_free(&M);
  int fails = 0, vmax_seen = 0;
  for (int pass = 0; pass < 6; ++pass)
  {
    MFTL_RemeshParameters p;
    remesh_params(&p, 0.03);
    remesh(ens, &p);
    int vmax = 0;
    for (int i = 0; i < o->n_vertices; ++i)
    {
      const MFTL_Vertex* v = &o->vertices[i];
      if (v->is_ghost || (v->flags & MFTL_FLAG_DEAD) || v->out_he_idx < 0) continue;
      int nfan = 0;
      const int cls = MFTLVertexRing(o, i, NULL, 0, &nfan);
      const int val = cls == MFTL_RING_OPEN_COMPLETE ? nfan + 1 : nfan;
      if (val > vmax) vmax = val;
    }
    int g;
    MPI_Allreduce(&vmax, &g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (g > vmax_seen) vmax_seen = g;
    MFTL_BoundaryLoopSet L;
    const int nl = loops(o, &L);
    if (nl > 0) MFTLFreeBoundaryLoops(&L);
    if (!valid(o) || nl != 1 || g > N) fails++;
  }
  if (g_rank == 0) printf("  valence growth (wheel of %d spokes, np%d): max valence over 6 passes %d, fails %d\n", N, g_np, vmax_seen, fails);
  CHECK(fails == 0 && vmax_seen <= N);
  MFTLEnsembleDestroy(&ens);
}

/* 16D review: MFTLInitRemeshParams defines every Stage-16C field. boundary_corner_angle was left
   indeterminate, so a caller's stack garbage >= 90 deg (or NaN) disabled the corner protection of
   the generalized remesher (a square patch lost corners in the parallel shift scheme on 4 ranks). */
static void test_params_init(void)
{
  MFTL_RemeshParameters p;
  memset(&p, 0xFF, sizeof(p)); /* NaN doubles, -1 integers */
  p.boundary_corner_angle = 1e300;
  p.stat_recovery_skipped = 12345;
  MFTLInitRemeshParams(&p, MFTL_REMESH_CUSTOM, 0.1, 1, MFTL_SMOOTH_QUALITY_DRIVEN, 45.0);
  CHECK(p.boundary_corner_angle == 0.0);
  CHECK(MFTLBoundaryCornerAngle(&p) == MFTL_BOUNDARY_CORNER_ANGLE_DEFAULT);
  CHECK(p.stat_recovery_skipped == 0);
  CHECK(p.stat_splits == 0 && p.stat_collapses == 0 && p.stat_flips == 0);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
  setvbuf(stdout, NULL, _IONBF, 0);
  const char* only = getenv("S16C_ONLY");
#define RUN(name, call)                                                                     \
  if (!only || strstr(only, name))                                                          \
  {                                                                                         \
    const int before = g_failed;                                                            \
    if (g_rank == 0) printf("== %s\n", name);                                               \
    call;                                                                                   \
    if (g_rank == 0) printf("   %s: %s\n", name, g_failed == before ? "ok" : "FAILED");     \
  }
  RUN("params", test_params_init());
  RUN("planar", test_planar(1.0));
  RUN("gb", test_gb(1.0));
  RUN("link", test_link());
  RUN("sa5", test_sa5());
  RUN("valence", test_valence_growth());
  RUN("cycles", (test_cycles_planar(1.0), test_cycles_hemisphere(0), test_cycles_hemisphere(1)));
  RUN("seam", test_seam());
  RUN("vcs", test_vcs_and_class());
  RUN("scales", (test_planar(1e-4), test_planar(1e4), test_gb(1e-4), test_gb(1e4), test_cycles_planar(1e-4), test_cycles_planar(1e4), test_b1_scale(1e-4),
                 test_b1_scale(1e4)));
  RUN("periodic", (test_ribbon(), test_film(), test_seam_sphere_parity(), test_tube_recovery(), test_proximity()));
  if (g_rank == 0) printf("Stage16C np%d: %d checks, %d failed\n", g_np, g_checks, g_failed);
  MPI_Finalize();
  (void) argc;
  (void) argv;
  return g_failed ? 1 : 0;
}
