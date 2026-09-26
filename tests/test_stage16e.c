/*
 * Stage 16E — geometry used by the open-surface boundary models of multiflow (plans/16 §6, §7,
 * tests E2/E3 of §10, library part). MFTL2 provides geometry only; the physical models live in
 * multiflow. Every CHECK is collective. Registered at 1, 2, 4 and 8 ranks.
 *
 *   planar   open planar patch: the Frenet element force (sum over edges of L t x n_edge, n_edge
 *            the normalised sum of the adjacent triangle normals, the multiflow EdgePlanar
 *            construction) vanishes on every triangle; the boundary line force sum L nu of the
 *            loop vanishes (a closed planar loop); total element force = boundary line force;
 *   cap      pinned spherical cap (polar angle alpha): the summed element force equals the
 *            boundary line force sum L nu (interior edges cancel) to round-off, and converges to
 *            the analytic ring force -2 pi r sin(alpha) e_z (r = R sin alpha): second order for
 *            the hemisphere, first order for a general cap (the conormal of a boundary edge uses
 *            its one-sided triangle normal); the loop's sum L nu is the discrete conormal
 *            integral;
 *   seam     the same cap translated across the x and y seams (dyadic shift): the same resultant
 *            to round-off;
 *   reverse  the cap with reversed triangle orientation: the same resultant (orientation
 *            invariance of nu = t x n);
 *   delta    MFTLBoundaryEdgeCentroidDistance: delta_e and h_T agree with a direct evaluation,
 *            and delta_e >= h_T / 3 on every boundary edge (the property that made the original
 *            Dirichlet floor delta_min = 0.1 h_T inactive; corrected in the 16E review to a floor
 *            on the tangential scale 0.1 L_e, multiflow), also on a sliver strip;
 *   closure  a hemisphere on a z-min wall closure is DEFINED; flattened onto the wall it becomes
 *            UNSUPPORTED (degenerate compact), with the rim lifted off the wall UNDEFINED (the
 *            statuses on which multiflow stops a wall-closed run after remeshing).
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_phase_set.h"
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

static MFTL_Ensemble* ens_new(double lo, double hi, int px, int py, int pz)
{
  int g[3] = {g_np, 1, 1};
  if (g_np == 4) g[0] = 2, g[1] = 2;
  if (g_np == 8) g[0] = 2, g[1] = 2, g[2] = 2;
  double mn[3] = {lo, lo, lo}, mx[3] = {hi, hi, hi};
  bool per[3] = {px != 0, py != 0, pz != 0};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, g, mn, mx, per, 4);
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
}

/* soup build (triangles spread over the ranks, continuous input); open bodies: BoundaryFromMesh */
static MFTL_Object* body(MFTL_Ensemble* ens, int gid, const Mesh* M)
{
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  MFTLObjectDeclareOpen(o, 1);
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
  int rc = MFTLBuildFromSoupEx(o, M->nv, vg, (const double(*)[3]) M->P, NULL, n, tg, (const int(*)[3]) tv, NULL, &fm);
  CHECK(rc == MFTL_SUCCESS);
  MFTLComputeTriangleProperties(o);
  MFTLRedistribute(ens);
  free(vg);
  free(tg);
  free(tv);
  return o;
}

/* planar n x n grid patch in z = z0 */
static Mesh plane(int n, double x0, double y0, double w, double z0)
{
  Mesh M;
  M.nv = (n + 1) * (n + 1);
  M.nt = 2 * n * n;
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) M.nt * sizeof(*M.T));
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
    {
      double* q = M.P[j * (n + 1) + i];
      q[0] = x0 + w * i / n, q[1] = y0 + w * j / n, q[2] = z0;
    }
  int t = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
    {
      int a = j * (n + 1) + i, b = a + 1, c = a + n + 2, d = a + n + 1;
      M.T[t][0] = a, M.T[t][1] = b, M.T[t][2] = c, t++;
      M.T[t][0] = a, M.T[t][1] = c, M.T[t][2] = d, t++;
    }
  return M;
}

/* spherical cap theta <= alpha about c (radius R), nr rings of m vertices, outward normals; the
   rim ring lies exactly at theta = alpha */
static Mesh cap(const double c[3], double R, double alpha, int nr, int m, int reverse)
{
  Mesh M;
  M.nv = 1 + nr * m;
  M.nt = m + 2 * m * (nr - 1);
  M.P = malloc((size_t) M.nv * sizeof(*M.P));
  M.T = malloc((size_t) M.nt * sizeof(*M.T));
  M.P[0][0] = c[0], M.P[0][1] = c[1], M.P[0][2] = c[2] + R;
  for (int i = 1; i <= nr; ++i)
    for (int j = 0; j < m; ++j)
    {
      const double th = alpha * i / nr, ph = 2.0 * M_PI * (j + 0.5 * (i % 2)) / m;
      double* q = M.P[1 + (i - 1) * m + j];
      q[0] = c[0] + R * sin(th) * cos(ph), q[1] = c[1] + R * sin(th) * sin(ph), q[2] = c[2] + R * cos(th);
    }
#define ID(i, j) (1 + ((i) - 1) * m + (((j) % m + m) % m))
  int t = 0;
  for (int j = 0; j < m; ++j) M.T[t][0] = 0, M.T[t][1] = ID(1, j), M.T[t][2] = ID(1, j + 1), t++;
  for (int i = 1; i < nr; ++i)
    for (int j = 0; j < m; ++j)
    {
      // ring i is rotated by half a step when i is odd: the strip between rings i and i+1 is
      // triangulated between the staggered vertices
      const int s = i % 2;
      const int a = ID(i, j), b = ID(i, j + 1), o0 = ID(i + 1, j), o1 = ID(i + 1, j + 1);
      if (s)
      {
        // ring i offset 0.5, ring i+1 offset 0: a at j+0.5, b at j+1.5; outer o1 at j+1
        M.T[t][0] = a, M.T[t][1] = o1, M.T[t][2] = b, t++;
        M.T[t][0] = a, M.T[t][1] = o0, M.T[t][2] = o1, t++;
      }
      else
      {
        // ring i offset 0, ring i+1 offset 0.5: a at j, b at j+1; outer o0 at j+0.5
        M.T[t][0] = a, M.T[t][1] = o0, M.T[t][2] = b, t++;
        M.T[t][0] = b, M.T[t][1] = o0, M.T[t][2] = o1, t++;
      }
    }
#undef ID
  M.nt = t;
  // orientation: outward (away from c) unless reversed
  for (int k = 0; k < M.nt; ++k)
  {
    double *p0 = M.P[M.T[k][0]], *p1 = M.P[M.T[k][1]], *p2 = M.P[M.T[k][2]];
    double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
    double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    double g[3] = {(p0[0] + p1[0] + p2[0]) / 3 - c[0], (p0[1] + p1[1] + p2[1]) / 3 - c[1], (p0[2] + p1[2] + p2[2]) / 3 - c[2]};
    const int out = n[0] * g[0] + n[1] * g[1] + n[2] * g[2] > 0.0;
    if (out == (reverse != 0))
    {
      int x = M.T[k][1];
      M.T[k][1] = M.T[k][2];
      M.T[k][2] = x;
    }
  }
  return M;
}

static void tri_normal(const MFTL_Object* o, int t, double n[3])
{
  double P[3][3];
  MFTLTriangleUnwrapped(o, &o->triangles[t], P);
  double e1[3] = {P[1][0] - P[0][0], P[1][1] - P[0][1], P[1][2] - P[0][2]};
  double e2[3] = {P[2][0] - P[0][0], P[2][1] - P[0][1], P[2][2] - P[0][2]};
  n[0] = e1[1] * e2[2] - e1[2] * e2[1], n[1] = e1[2] * e2[0] - e1[0] * e2[2], n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  for (int k = 0; k < 3; ++k) n[k] /= l;
}

/* Frenet element forces (sigma = 1) of the owned triangles, summed collectively: F_elem; the
   boundary line force sum L nu of the owned triangles' boundary half-edges (per tag 0 and total):
   F_line; the largest |element force| (max_elem) */
static void forces(const MFTL_Object* o, double F_elem[3], double F_line[3], double* max_elem, int* n_boundary)
{
  double fe[3] = {0, 0, 0}, fl[3] = {0, 0, 0}, mx = 0.0;
  int nb = 0, bad = 0;
  for (int t = 0; t < o->n_triangles; ++t)
  {
    const MFTL_Triangle* tri = &o->triangles[t];
    if (tri->is_ghost || tri->owner_proc != o->domain->rank || (tri->flags & MFTL_FLAG_DEAD)) continue;
    double nT[3], f[3] = {0, 0, 0};
    tri_normal(o, t, nT);
    int h = tri->he_idx;
    for (int k = 0; k < 3; ++k, h = o->half_edges[h].next_idx)
    {
      const MFTL_HalfEdge* he = &o->half_edges[h];
      double d[3], ne[3];
      MFTLEdgeVector(o, he, d);
      const double L = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      memcpy(ne, nT, sizeof(ne));
      if (MFTLHeIsBoundary(he))
      {
        double len, tt[3], nu[3];
        if (MFTLBoundaryEdgeGeometry(o, h, &len, tt, nu, NULL) != MFTL_SUCCESS) bad++;
        for (int a = 0; a < 3; ++a) fl[a] += len * nu[a];
        nb++;
      }
      else
      {
        if (!MFTLHeIsLocal(he)) bad++;  // G1: every edge of an owned triangle is LOCAL or BOUNDARY
        double nN[3];
        tri_normal(o, o->half_edges[he->twin_idx].triangle_idx, nN);
        for (int a = 0; a < 3; ++a) ne[a] += nN[a];
        const double l = sqrt(ne[0] * ne[0] + ne[1] * ne[1] + ne[2] * ne[2]);
        for (int a = 0; a < 3; ++a) ne[a] /= l;
      }
      // conormal of this triangle's edge: t x n_edge (the triangle lies to the left of t)
      const double nu[3] = {d[1] * ne[2] - d[2] * ne[1], d[2] * ne[0] - d[0] * ne[2], d[0] * ne[1] - d[1] * ne[0]};
      for (int a = 0; a < 3; ++a) f[a] += nu[a] * (L > 0 ? 1.0 : 0.0);  // |d| = L: L * (d/L x n)
    }
    for (int a = 0; a < 3; ++a) fe[a] += f[a];
    const double m = sqrt(f[0] * f[0] + f[1] * f[1] + f[2] * f[2]);
    if (m > mx) mx = m;
  }
  MPI_Allreduce(fe, F_elem, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(fl, F_line, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mx, max_elem, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&nb, n_boundary, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(bad == 0);
}

static double norm3(const double a[3]) { return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

/* ------------------------------------------------------------------------------------------ */
static void test_planar(void)
{
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
  Mesh M = plane(12, 0.2, 0.3, 0.5, 0.45);
  MFTL_Object* o = body(ens, 1, &M);
  double Fe[3], Fl[3], mx;
  int nb;
  forces(o, Fe, Fl, &mx, &nb);
  CHECK(nb == 48);
  CHECK(mx < 1e-14);  // every triangle: zero element force (flat)
  CHECK(norm3(Fl) < 1e-14);  // closed planar loop: sum L nu = 0
  CHECK(norm3(Fe) < 1e-14);
  if (g_rank == 0) printf("planar: max |f_T| = %.2e, |sum L nu| = %.2e, |F_elem| = %.2e\n", mx, norm3(Fl), norm3(Fe));
  mesh_free(&M);
  MFTLEnsembleDestroy(&ens);
}

/* resultant of the cap with polar angle alpha; returns F_elem (z) */
static void cap_forces(double shift_x, double shift_y, int reverse, int nr, int m, double alpha, double Fe[3], double Fl[3], int periodic)
{
  const double R = 0.25;
  const double c[3] = {0.5 + shift_x, 0.5 + shift_y, 0.3};
  MFTL_Ensemble* ens = ens_new(0.0, 1.0, periodic, periodic, 0);
  Mesh M = cap(c, R, alpha, nr, m, reverse);
  MFTL_Object* o = body(ens, 1, &M);
  double mx;
  int nb;
  forces(o, Fe, Fl, &mx, &nb);
  CHECK(nb == m);
  MFTL_BoundaryLoopSet L;
  CHECK(MFTLBoundaryLoops(o, &L) == MFTL_SUCCESS && L.n_loops == 1);
  MFTLFreeBoundaryLoops(&L);
  mesh_free(&M);
  MFTLEnsembleDestroy(&ens);
}

static void test_cap(void)
{
  const double R = 0.25;
  const double alphas[2] = {0.5 * M_PI, 0.35 * M_PI};
  for (int ia = 0; ia < 2; ++ia)
  {
    const double alpha = alphas[ia], r = R * sin(alpha), Fz = -2.0 * M_PI * r * sin(alpha);
    double err[4];
    int ms[4] = {16, 32, 64, 128};
    for (int k = 0; k < 4; ++k)
    {
      const int m = ms[k], nr = m / 4;
      double Fe[3], Fl[3];
      cap_forces(0, 0, 0, nr, m, alpha, Fe, Fl, 0);
      // interior edges cancel: total element force = boundary line force (round-off)
      double d[3] = {Fe[0] - Fl[0], Fe[1] - Fl[1], Fe[2] - Fl[2]};
      CHECK(norm3(d) < 1e-13 * fabs(Fz));
      CHECK(fabs(Fe[0]) < 1e-13 * fabs(Fz) && fabs(Fe[1]) < 1e-13 * fabs(Fz));
      err[k] = fabs(Fe[2] - Fz) / fabs(Fz);
      if (g_rank == 0)
        printf("cap alpha=%.3f m=%3d: F_z = %.12f (analytic %.12f), rel err %.3e, |F_elem - F_line| = %.1e\n", alpha, m, Fe[2], Fz, err[k], norm3(d));
    }
    for (int k = 1; k < 4; ++k)
    {
      const double rate = log(err[k - 1] / err[k]) / log(2.0);
      if (g_rank == 0) printf("  rate %.2f\n", rate);
      /* hemisphere: the rim conormal is exact to O(h^2); a general cap: the boundary edge's
         conormal uses its (one-sided) triangle normal, taken half a ring above the rim, so the
         line-force direction and the resultant converge at first order (the Frenet element
         model's order at an open boundary) */
      CHECK(rate > (ia == 0 ? 1.9 : 0.9));
    }
    CHECK(err[3] < (ia == 0 ? 1e-3 : 1e-2));
  }
}

static void test_seam_reverse(void)
{
  const double alpha = 0.4 * M_PI;
  double F0[3], L0[3], F1[3], L1[3], F2[3], L2[3], F3[3], L3[3];
  cap_forces(0.0, 0.0, 0, 8, 32, alpha, F0, L0, 1);  // periodic domain, away from the seams
  cap_forces(0.5, 0.5, 0, 8, 32, alpha, F1, L1, 1);  // centred on the xy corner (both seams)
  cap_forces(0.0, 0.0, 1, 8, 32, alpha, F2, L2, 0);  // reversed orientation
  cap_forces(0.5, 0.5, 1, 8, 32, alpha, F3, L3, 1);  // reversed and across the seams
  const double s = norm3(F0);
  for (int a = 0; a < 3; ++a)
  {
    CHECK(fabs(F1[a] - F0[a]) < 1e-13 * s);
    CHECK(fabs(L1[a] - L0[a]) < 1e-13 * s);
    CHECK(fabs(F2[a] - F0[a]) < 1e-13 * s);
    CHECK(fabs(L2[a] - L0[a]) < 1e-13 * s);
    CHECK(fabs(F3[a] - F0[a]) < 1e-13 * s);
  }
  if (g_rank == 0)
    printf("seam/reverse: F_z %.15f | seam %.15f | reversed %.15f | reversed+seam %.15f\n", F0[2], F1[2], F2[2], F3[2]);
}

/* delta_e and h_T against a direct evaluation; delta_e >= h_T / 3 */
static void delta_check(MFTL_Object* o, double* min_ratio, int* n)
{
  double mr = 1e300;
  int cnt = 0, bad = 0;
  for (int h = 0; h < o->n_half_edges; ++h)
  {
    const MFTL_HalfEdge* he = &o->half_edges[h];
    if (!MFTLHeIsBoundary(he) || (he->flags & MFTL_FLAG_DEAD)) continue;
    const MFTL_Triangle* tri = &o->triangles[he->triangle_idx];
    if (tri->is_ghost || tri->owner_proc != o->domain->rank) continue;
    double delta, hT;
    if (MFTLBoundaryEdgeCentroidDistance(o, h, &delta, &hT) != MFTL_SUCCESS) bad++;
    // direct: all corners in the frame of the edge origin
    const MFTL_Vertex* a = MFTLHeOrigin(o, he);
    double pb[3], pc[3];
    MFTLGetMinimumImageVector(o->domain, pb, a->pos, MFTLHeTarget(o, he)->pos);
    const MFTL_Vertex* cv = &o->vertices[o->half_edges[he->next_idx].vertex_idx];
    MFTLGetMinimumImageVector(o->domain, pc, a->pos, cv->pos);
    double g[3], e[3];
    for (int k = 0; k < 3; ++k) g[k] = (pb[k] + pc[k]) / 3.0 - 0.5 * pb[k];
    const double L = norm3(pb);
    e[0] = pb[1] * pc[2] - pb[2] * pc[1], e[1] = pb[2] * pc[0] - pb[0] * pc[2], e[2] = pb[0] * pc[1] - pb[1] * pc[0];
    if (!(fabs(delta - norm3(g)) < 1e-12 * L)) bad++;
    if (!(fabs(hT - norm3(e) / L) < 1e-12 * L)) bad++;
    if (delta / hT < mr) mr = delta / hT;
    cnt++;
  }
  MPI_Allreduce(&mr, min_ratio, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&cnt, n, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  CHECK(bad == 0);
}

static void test_delta(void)
{
  {
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 1, 1, 0);
    const double c[3] = {0.98, 0.02, 0.3};  // across the x and y seams
    Mesh M = cap(c, 0.25, 0.4 * M_PI, 6, 24, 0);
    MFTL_Object* o = body(ens, 1, &M);
    double mr;
    int n;
    delta_check(o, &mr, &n);
    CHECK(n == 24);
    CHECK(mr >= 1.0 / 3.0 - 1e-12);
    if (g_rank == 0) printf("delta: cap across seams, %d boundary edges, min delta/h_T = %.4f\n", n, mr);
    mesh_free(&M);
    MFTLEnsembleDestroy(&ens);
  }
  {
    // sliver strip: one row of needle triangles with the long edge on the boundary (h/L = 1e-4)
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
    const int n = 8;
    Mesh M;
    M.nv = 2 * (n + 1);
    M.nt = 2 * n;
    M.P = malloc((size_t) M.nv * sizeof(*M.P));
    M.T = malloc((size_t) M.nt * sizeof(*M.T));
    for (int i = 0; i <= n; ++i)
    {
      M.P[i][0] = 0.1 + 0.8 * i / n, M.P[i][1] = 0.5, M.P[i][2] = 0.5;
      M.P[n + 1 + i][0] = 0.1 + 0.8 * i / n + 0.05, M.P[n + 1 + i][1] = 0.5 + 1e-5, M.P[n + 1 + i][2] = 0.5;
    }
    int t = 0;
    for (int i = 0; i < n; ++i)
    {
      M.T[t][0] = i, M.T[t][1] = i + 1, M.T[t][2] = n + 1 + i, t++;
      M.T[t][0] = i + 1, M.T[t][1] = n + 2 + i, M.T[t][2] = n + 1 + i, t++;
    }
    MFTL_Object* o = body(ens, 1, &M);
    double mr;
    int nb;
    delta_check(o, &mr, &nb);
    CHECK(nb == 2 * n + 2);
    CHECK(mr >= 1.0 / 3.0 - 1e-9);
    if (g_rank == 0) printf("delta: sliver strip (h/L = 1e-4), %d boundary edges, min delta/h_T = %.4f\n", nb, mr);
    mesh_free(&M);
    MFTLEnsembleDestroy(&ens);
  }
}

/* 16E review: a wall-closed set that is DEFINED stops being DEFINED when the geometry changes
   (multiflow re-classifies after every remeshing and stops the run on any status other than
   DEFINED): a hemisphere on the z-min closure flattened onto the wall (degenerate), and the same cap with its rim lifted off
   the wall (a loop on no declared closure). Every vertex (owned and ghost) gets the same
   deterministic position, so no ghost update is needed. */
static MFTL_PhaseStatus closure_status(MFTL_Ensemble* ens, MFTL_Object* o, double* V)
{
  MFTL_PhaseSet* S = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(S, o, 1);
  MFTLPhaseSetAddWallClosure(S, 2, MFTL_WALL_MIN);
  const MFTL_PhaseStatus st = MFTLPhaseSetVolume(S, V);
  MFTLPhaseSetDestroy(&S);
  return st;
}

static void test_closure_transition(void)
{
  const double c[3] = {0.5, 0.5, 0.0}, R = 0.25;
  for (int kind = 0; kind < 2; ++kind)
  {
    MFTL_Ensemble* ens = ens_new(0.0, 1.0, 0, 0, 0);
    Mesh M = cap(c, R, M_PI / 2, 8, 48, 0);
    MFTL_Object* o = body(ens, 1, &M);
    double V = 0.0;
    CHECK(closure_status(ens, o, &V) == MFTL_DEFINED);
    CHECK(fabs(V - 2.0 * M_PI * R * R * R / 3.0) < 0.02 * V);  // discrete hemisphere
    for (int i = 0; i < o->n_vertices; ++i)
    {
      double* x = o->vertices[i].pos;
      if (kind == 0)
        x[2] *= 1e-14;  // flattened onto the wall: V ~ 1e-14 R^3, below the degeneracy tolerance
      else
        x[2] += 0.05;  // lifted: the loop is 0.05 off the closure wall
    }
    MFTLComputeTriangleProperties(o);
    // flattened: a degenerate compact component (|v_c| <= tau_V) is UNSUPPORTED, which precedes
    // the degenerate-volume UNDEFINED (§3.6.3, erratum E2); lifted: the loop is on no closure
    CHECK(closure_status(ens, o, &V) == (kind == 0 ? MFTL_UNSUPPORTED : MFTL_UNDEFINED));
    mesh_free(&M);
    MFTLEnsembleDestroy(&ens);
  }
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
  test_planar();
  test_cap();
  test_seam_reverse();
  test_delta();
  test_closure_transition();
  if (g_rank == 0) printf("Stage16E (np %d): %d checks, %d failed\n", g_np, g_checks, g_failed);
  MPI_Finalize();
  return g_failed ? 1 : 0;
}
