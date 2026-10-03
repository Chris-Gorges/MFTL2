/*
 * @Author: Christian Gorges
 * @Date: 2026-10-03
 *
 * @brief Tests for MFTLEnsembleBodyIntegrals: area, volume and centroid of every
 *        body, closed or closed by the domain walls.
 *
 * EVERY REFERENCE IS COMPUTED BEFORE PARTITIONING, ON RANK 0, BY CODE THAT SHARES
 * NOTHING WITH THE LIBRARY'S. The library integrates fluxes over the surface with
 * minimum-image corners lifted about a reduced centre; the references are the
 * polyhedral volume and centroid of tetrahedra from the generated vertex list
 * (unwrapped by construction), or a height field's prisms from its own grid. Both
 * are EXACT for the mesh, so the comparison is to round-off, not to a
 * discretisation error -- which is what lets a 1e-12 tolerance tell a wrong
 * lift, a lost triangle or a double-counted holder from a right answer.
 *
 * What each fixture exists for:
 *
 *   a sphere and an ellipsoid, the latter across the periodic seam and wider
 *     than half the period (the extent is exact to 3L/4): closed bodies;
 *   a flat sheet with its rim on the four side walls (V = h A exactly), a
 *     dome on the floor and one hanging from the ceiling (the reference wall
 *     chosen from where the rims are), a wavy height field periodic in x with
 *     rims on the y walls (the travelling-wave shape): wall-closed bodies;
 *   the statuses: a rim off the walls, rims on both walls of every
 *     non-periodic axis, a centroid along an axis the body wraps, a closed sheet
 *     across a fully periodic box;
 *   bit-identical answers on every holder, and bodies with several holders
 *     (asserted), at np = 1, 2, 4, 8 and 27.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

#define MAX_B 16

/** @brief What a body should measure, computed on rank 0 before partitioning and broadcast. */
typedef struct
{
  double area, volume, centroid[3];
} reference;

static MFTL_Ensemble *make_ens(const double lo[3], const double hi[3], const bool per[3])
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double l[3] = {lo[0], lo[1], lo[2]}, h[3] = {hi[0], hi[1], hi[2]};
  bool p[3] = {per[0], per[1], per[2]};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, l, h, p, 8);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * References: tetrahedra and prisms, from UNWRAPPED vertex lists
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Volume and centroid of the region a triangle list bounds together with
 *        a plane through @p apex, as signed tetrahedra (apex, a, b, c). For a
 *        closed surface the apex is arbitrary; for one closed by a planar wall,
 *        an apex ON that wall makes the wall's own (planar) cap contribute
 *        nothing. Positions must be unwrapped (one image).
 */
static void tet_reference(const double *x, const int *tri, int nt, const double apex[3], reference *r)
{
  double v = 0.0, m[3] = {0.0, 0.0, 0.0}, area = 0.0;
  for (int t = 0; t < nt; ++t)
  {
    const double *a = &x[3 * tri[3 * t]], *b = &x[3 * tri[3 * t + 1]], *c = &x[3 * tri[3 * t + 2]];
    double u[3], w[3], q[3];
    for (int k = 0; k < 3; ++k)
    {
      u[k] = a[k] - apex[k];
      w[k] = b[k] - apex[k];
      q[k] = c[k] - apex[k];
    }
    const double vt = (u[0] * (w[1] * q[2] - w[2] * q[1]) - u[1] * (w[0] * q[2] - w[2] * q[0]) + u[2] * (w[0] * q[1] - w[1] * q[0])) / 6.0;
    v += vt;
    for (int k = 0; k < 3; ++k) m[k] += vt * (apex[k] + a[k] + b[k] + c[k]) / 4.0;
    double e1[3], e2[3];
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = b[k] - a[k];
      e2[k] = c[k] - a[k];
    }
    const double cr[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
    area += 0.5 * sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
  }
  r->area = area;
  r->volume = v;
  for (int k = 0; k < 3; ++k) r->centroid[k] = m[k] / v;
}

static void bcast_reference(reference *r) { MPI_Bcast(r, 5, MPI_DOUBLE, 0, MPI_COMM_WORLD); }

/* ═════════════════════════════════════════════════════════════════════════════
 * Fixtures
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief An icosphere about @p c, stretched by @p s per axis, built on rank 0;
 *        @p r receives its tetrahedral reference from the UNWRAPPED positions,
 *        which are then wrapped into the domain.
 */
static void add_ellipsoid(MFTL_Ensemble *ens, const double c[3], const double s[3], int refine, reference *r)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(obj, cc, 1.0, refine);
  if (g_rank == 0)
  {
    const MFTL_Domain *dom = &ens->domain;
    double *x = (double *) malloc((size_t) obj->verts.n * 3 * sizeof(double));
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double d[3];
      MFTLGetMinimumImageVector(dom, d, c, &obj->verts.pos[3 * v]);
      for (int k = 0; k < 3; ++k)
      {
        x[3 * v + k] = c[k] + s[k] * d[k];
        obj->verts.pos[3 * v + k] = x[3 * v + k];
      }
      MFTLApplyPeriodicity(dom, &obj->verts.pos[3 * v]);
    }
    int *tri = (int *) malloc((size_t) obj->tris.n * 3 * sizeof(int));
    for (int t = 0; t < obj->tris.n; ++t) MFTLGetTriangleVertices(obj, t, &tri[3 * t]);
    tet_reference(x, tri, obj->tris.n, c, r);
    free(tri);
    free(x);
  }
  bcast_reference(r);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
}

/**
 * @brief A height field z = z0 + f(x, y) over the box's whole x-y extent on an
 *        (nx+1) x (ny+1) grid, or nx columns (ny rows) when x (y) is periodic:
 *        the last column (row) joins the first across the seam. Normals point to +z. With
 *        @p flip, hangs DOWN from the ceiling instead: z = zmax - f, normals to
 *        -z. The reference is the prisms between the surface and the floor (or
 *        ceiling), integrated exactly on the grid's own projected triangles.
 */
typedef double (*height_fn)(double x, double y, const double *a);

static void add_height_field(MFTL_Ensemble *ens, height_fn f, const double *a, int nx, int ny, bool flip, reference *r)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const MFTL_Domain *dom = &ens->domain;
  const bool px = dom->is_periodic[0], py = dom->is_periodic[1];
  const int ncol = px ? nx : nx + 1, nrow = py ? ny : ny + 1;
  const double dx = dom->domain_size[0] / nx, dy = dom->domain_size[1] / ny;
  const double wall = flip ? dom->global_max[2] : dom->global_min[2];

  if (g_rank == 0)
  {
    double *pos = (double *) malloc((size_t) ncol * nrow * 3 * sizeof(double));
    int *tri = (int *) malloc((size_t) 2 * nx * ny * 3 * sizeof(int));
    double vol = 0.0, my = 0.0, mz = 0.0, mx = 0.0;
    for (int j = 0; j < nrow; ++j)
      for (int i = 0; i < ncol; ++i)
      {
        const int v = j * ncol + i;
        const double x = dom->global_min[0] + i * dx, y = dom->global_min[1] + j * dy;
        const double h = f(x, y, a);
        pos[3 * v + 0] = x;
        pos[3 * v + 1] = y;
        pos[3 * v + 2] = flip ? wall - h : wall + h;
      }
    int t = 0;
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
      {
        const int i1 = px ? (i + 1) % ncol : i + 1;
        const int j1 = py ? (j + 1) % nrow : j + 1;
        const int v00 = j * ncol + i, v10 = j * ncol + i1, v01 = j1 * ncol + i, v11 = j1 * ncol + i1;
        int q[2][3] = {{v00, v10, v11}, {v00, v11, v01}};
        for (int s = 0; s < 2; ++s, ++t)
        {
          int *T = &tri[3 * t];
          T[0] = q[s][0];
          T[1] = flip ? q[s][2] : q[s][1];
          T[2] = flip ? q[s][1] : q[s][2];
          /* The prism over the PROJECTED triangle: its x-y corners from the
           * grid indices (unwrapped across the seam), heights from the field. */
          const int ii[2][3] = {{i, i + 1, i + 1}, {i, i + 1, i}}, jj[2][3] = {{j, j, j + 1}, {j, j + 1, j + 1}};
          double X[3], Y[3], H[3];
          for (int k = 0; k < 3; ++k)
          {
            X[k] = dom->global_min[0] + ii[s][k] * dx;
            Y[k] = dom->global_min[1] + jj[s][k] * dy;
            H[k] = f(X[k] - (px && ii[s][k] == nx ? dom->domain_size[0] : 0.0), Y[k] - (py && jj[s][k] == ny ? dom->domain_size[1] : 0.0), a);
          }
          const double A = 0.5 * fabs((X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]));
          const double sh = H[0] + H[1] + H[2], sy = Y[0] + Y[1] + Y[2], sx = X[0] + X[1] + X[2];
          vol += A * sh / 3.0;
          my += A / 12.0 * (Y[0] * H[0] + Y[1] * H[1] + Y[2] * H[2] + sy * sh);
          mx += A / 12.0 * (X[0] * H[0] + X[1] * H[1] + X[2] * H[2] + sx * sh);
          mz += A / 12.0 * (H[0] * H[0] + H[1] * H[1] + H[2] * H[2] + sh * sh) / 2.0;
        }
      }
    MFTLBuildIndexedMesh(obj, pos, ncol * nrow, tri, t);
    r->volume = vol;
    r->centroid[0] = mx / vol;
    r->centroid[1] = my / vol;
    r->centroid[2] = flip ? wall - mz / vol : wall + mz / vol;
    r->area = 0.0;
    for (int tt = 0; tt < obj->tris.n; ++tt) r->area += MFTLGetTriangleArea(obj, tt);
    free(tri);
    free(pos);
  }
  else
    MFTLObjectBuildDerived(obj);
  bcast_reference(r);
}

static double h_flat(double x, double y, const double *a)
{
  (void) x;
  (void) y;
  return a[0];
}

static double h_dome(double x, double y, const double *a)
{
  /* a: cx cy R peak -- a paraboloid cap, zero outside the disk is NOT used:
   * the dome fixtures are built as disks, see add_cap. */
  const double r2 = ((x - a[0]) * (x - a[0]) + (y - a[1]) * (y - a[1])) / (a[2] * a[2]);
  return a[3] * (1.0 - r2);
}

static double h_wave(double x, double y, const double *a)
{
  /* a: mean amplitude Lx Ly */
  return a[0] + a[1] * sin(2.0 * MFTL_PI * x / a[2]) * (1.0 + 0.3 * cos(MFTL_PI * y / a[3]));
}

/**
 * @brief A paraboloid cap with its circular rim on the floor (or hanging from
 *        the ceiling with @p flip): centre + n rings of 6i vertices.
 */
static void add_cap(MFTL_Ensemble *ens, const double *a, int n, bool flip, reference *r)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const MFTL_Domain *dom = &ens->domain;
  const double wall = flip ? dom->global_max[2] : dom->global_min[2];
  if (g_rank == 0)
  {
    const int nv = 1 + 3 * n * (n + 1);
    double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
    double *ang = (double *) malloc((size_t) nv * sizeof(double));
    int *tri = (int *) malloc((size_t) 6 * n * n * 3 * sizeof(int));
    int *first = (int *) malloc((size_t) (n + 1) * sizeof(int));
    int v = 0;
    for (int i = 0; i <= n; ++i)
    {
      first[i] = v;
      const int m = (i == 0) ? 1 : 6 * i;
      for (int j = 0; j < m; ++j, ++v)
      {
        const double th = (i == 0) ? 0.0 : 2.0 * MFTL_PI * j / m, rr = a[2] * (double) i / n;
        ang[v] = th;
        pos[3 * v + 0] = a[0] + rr * cos(th);
        pos[3 * v + 1] = a[1] + rr * sin(th);
        const double s = (double) i / n;
        const double h = a[3] * (1.0 - s * s);
        pos[3 * v + 2] = flip ? wall - h : wall + h;
      }
    }
    int t = 0;
    for (int i = 1; i <= n; ++i)
    {
      const int ma = (i == 1) ? 1 : 6 * (i - 1), mb = 6 * i;
      int ja = 0, jb = 0;
      while (ja < ma || jb < mb)
      {
        const double next_a = (ja + 1 < ma) ? ang[first[i - 1] + ja + 1] : 2.0 * MFTL_PI;
        const double next_b = (jb + 1 < mb) ? ang[first[i] + jb + 1] : 2.0 * MFTL_PI;
        const int a0 = first[i - 1] + ja % ma, b0 = first[i] + jb % mb;
        int c2;
        if (jb < mb && (ja == ma || next_b <= next_a || ma == 1))
        {
          c2 = first[i] + (jb + 1) % mb;
          jb++;
        }
        else
        {
          c2 = first[i - 1] + (ja + 1) % ma;
          ja++;
        }
        tri[3 * t + 0] = a0;
        tri[3 * t + 1] = flip ? c2 : b0;
        tri[3 * t + 2] = flip ? b0 : c2;
        t++;
        if (ma == 1 && jb == mb) break;
      }
    }
    MFTLBuildIndexedMesh(obj, pos, nv, tri, t);
    const double apex[3] = {a[0], a[1], wall};
    tet_reference(pos, tri, t, apex, r);
    free(first);
    free(tri);
    free(ang);
    free(pos);
  }
  else
    MFTLObjectBuildDerived(obj);
  bcast_reference(r);
}

/**
 * @brief A tube of @p nsec-gon cross-section, radius @p R, along axis @p ax
 *        (0: x, 2: z) through (c1, c2) in the other two coordinates. Along a
 *        PERIODIC axis it closes on itself across the seam (a closed tube
 *        wrapping the axis); along a walled one it runs wall to wall, its two
 *        rims on the two walls. Outward normals. Built on rank 0.
 */
static void add_tube(MFTL_Ensemble *ens, int ax, double c1, double c2, double R, int nsec, int nlen)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const MFTL_Domain *dom = &ens->domain;
  const bool wrap = dom->is_periodic[ax];
  const int nring = wrap ? nlen : nlen + 1;
  /* (u, v) are the cross-section's axes, so that (u, v, ax) is right-handed. */
  const int u = (ax + 1) % 3, v = (ax + 2) % 3;
  if (g_rank == 0)
  {
    double *pos = (double *) malloc((size_t) nring * nsec * 3 * sizeof(double));
    int *tri = (int *) malloc((size_t) 2 * nlen * nsec * 3 * sizeof(int));
    const double cu = (u == (ax + 1) % 3) ? c1 : c2, cv = c2;
    for (int i = 0; i < nring; ++i)
      for (int j = 0; j < nsec; ++j)
      {
        double *p = &pos[3 * (i * nsec + j)];
        const double th = 2.0 * MFTL_PI * j / nsec;
        p[ax] = dom->global_min[ax] + dom->domain_size[ax] * (double) i / nlen;
        if (!wrap && i == nlen) p[ax] = dom->global_max[ax];
        p[u] = cu + R * cos(th);
        p[v] = cv + R * sin(th);
      }
    int t = 0;
    for (int i = 0; i < nlen; ++i)
      for (int j = 0; j < nsec; ++j)
      {
        const int i1 = wrap ? (i + 1) % nring : i + 1, j1 = (j + 1) % nsec;
        const int a = i * nsec + j, b = i * nsec + j1, c = i1 * nsec + j1, d = i1 * nsec + j;
        tri[3 * t + 0] = a;
        tri[3 * t + 1] = b;
        tri[3 * t + 2] = c;
        t++;
        tri[3 * t + 0] = a;
        tri[3 * t + 1] = c;
        tri[3 * t + 2] = d;
        t++;
      }
    MFTLBuildIndexedMesh(obj, pos, nring * nsec, tri, t);
    free(tri);
    free(pos);
  }
  else
    MFTLObjectBuildDerived(obj);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Checking
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Bodies held by two or more ranks: the non-vacuity measure. */
static int shared_bodies(MFTL_Ensemble *ens)
{
  const int b = ens->n_local_objects;
  double *r = (double *) malloc((size_t) (2 * b + 1) * sizeof(double));
  const MFTL_BodyReduceOp ops[2] = {MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MAX};
  for (int o = 0; o < b; ++o) r[2 * o] = r[2 * o + 1] = (double) g_rank;
  MFTLEnsembleReduceBodies(ens, r, 2, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
  int n = 0;
  for (int o = 0; o < b; ++o)
    if ((int) r[2 * o] == g_rank && r[2 * o + 1] > r[2 * o]) n++;
  free(r);
  int total = 0;
  MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return total;
}

/**
 * @brief Integrates, then gathers every body's record by gid -- MIN and MAX over
 *        the holders, which must be the same bits -- into @p rec[gid].
 */
static int integrate(MFTL_Ensemble *ens, MFTL_BodyIntegrals *rec, int *identical)
{
  const int b = ens->n_local_objects;
  MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) calloc((size_t) MFTL_MAX(b, 1), sizeof(MFTL_BodyIntegrals));
  const int rc = MFTLEnsembleBodyIntegrals(ens, bi);

  enum
  {
    F = 13
  };
  double lo[MAX_B * F], hi[MAX_B * F];
  for (int i = 0; i < MAX_B * F; ++i)
  {
    lo[i] = 1e300;
    hi[i] = -1e300;
  }
  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    const double v[F] = {bi[o].area, bi[o].volume, bi[o].centroid[0], bi[o].centroid[1], bi[o].centroid[2], bi[o].volume_status, bi[o].centroid_status[0],
                         bi[o].centroid_status[1], bi[o].centroid_status[2], bi[o].axis, bi[o].lowest_holder, bi[o].closed ? 1.0 : 0.0,
                         bi[o].reference};
    for (int k = 0; k < F; ++k)
    {
      lo[g * F + k] = v[k];
      hi[g * F + k] = v[k];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, lo, MAX_B * F, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, hi, MAX_B * F, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  *identical = 1;
  for (int g = 0; g < ens->n_bodies_global && g < MAX_B; ++g)
  {
    for (int k = 0; k < F; ++k)
      if (memcmp(&lo[g * F + k], &hi[g * F + k], sizeof(double)) != 0) *identical = 0;
    rec[g].area = lo[g * F + 0];
    rec[g].volume = lo[g * F + 1];
    for (int k = 0; k < 3; ++k) rec[g].centroid[k] = lo[g * F + 2 + k];
    rec[g].volume_status = (MFTL_IntegralStatus) lo[g * F + 5];
    for (int k = 0; k < 3; ++k) rec[g].centroid_status[k] = (MFTL_IntegralStatus) lo[g * F + 6 + k];
    rec[g].axis = (int) lo[g * F + 9];
    rec[g].lowest_holder = (int) lo[g * F + 10];
    rec[g].closed = lo[g * F + 11] > 0.0;
    rec[g].reference = lo[g * F + 12];
  }
  free(bi);
  return rc;
}

/** @brief |a - b| relative to a scale. */
static bool near(double a, double b, double scale, double tol) { return fabs(a - b) <= tol * scale; }

/** @brief Body @p g against its reference, on the components marked DEFINED in @p want. */
static int matches(const MFTL_BodyIntegrals *got, const reference *ref, const bool want_c[3], double length, const char *what)
{
  int ok = got->volume_status == MFTL_INTEGRAL_DEFINED && near(got->area, ref->area, ref->area, 1e-12) && near(got->volume, ref->volume, fabs(ref->volume), 1e-11);
  for (int k = 0; k < 3; ++k)
  {
    if (!want_c[k]) continue;
    ok = ok && got->centroid_status[k] == MFTL_INTEGRAL_DEFINED && near(got->centroid[k], ref->centroid[k], length, 1e-11);
  }
  if (!ok && g_rank == 0)
    printf("\n  %s: status %d, A %.15g / %.15g, V %.15g / %.15g, C (%.12g %.12g %.12g) / (%.12g %.12g %.12g) ", what, got->volume_status, got->area,
           ref->area, got->volume, ref->volume, got->centroid[0], got->centroid[1], got->centroid[2], ref->centroid[0], ref->centroid[1],
           ref->centroid[2]);
  return ok;
}

/** @brief The reference centroid moved into the box on the periodic axes, as the library reports it. */
static void wrap_reference(const MFTL_Domain *dom, reference *r) { MFTLApplyPeriodicity(dom, r->centroid); }

/* ═════════════════════════════════════════════════════════════════════════════
 * Tests
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_closed_bodies_match_their_polyhedra(void)
{
  const double lo[3] = {-1.3, -0.4, 0.7}, hi[3] = {2.7, 3.6, 4.7};
  const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  reference ref[4];
  /* A sphere; an ellipsoid across the x seam; one across the y seam AND wider
   * than half the period along it (2.4 of 4); a small one near a corner. */
  const double c[4][3] = {{0.7, 1.6, 2.7}, {-1.25, 1.1, 2.0}, {1.2, -0.35, 3.1}, {2.6, 3.5, 4.6}};
  const double s[4][3] = {{0.6, 0.6, 0.6}, {0.9, 0.5, 0.4}, {0.5, 1.2, 0.45}, {0.2, 0.3, 0.25}};
  for (int i = 0; i < 4; ++i)
  {
    add_ellipsoid(ens, c[i], s[i], 4, &ref[i]);
    wrap_reference(&ens->domain, &ref[i]);
  }
  MFTLRedistribute(ens);
  const int shared = shared_bodies(ens);

  MFTL_BodyIntegrals rec[MAX_B];
  int identical = 0;
  const int rc = integrate(ens, rec, &identical);
  int ok = 1;
  const bool all[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  for (int i = 0; i < 4; ++i) ok = ok && matches(&rec[i], &ref[i], all, 4.0, "closed") && rec[i].closed;
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(identical, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || shared > 0, MPI_COMM_WORLD));

  /* And the mesh converges to the sphere at second order: refine 2, 3, 4. */
  if (g_rank == 0)
  {
    double err[3];
    for (int q = 2; q <= 4; ++q)
    {
      MFTL_Ensemble *solo = MFTLEnsembleCreate(MPI_COMM_SELF, (int[3]){1, 1, 1}, (double[3]){-2, -2, -2}, (double[3]){2, 2, 2}, (bool[3]){1, 1, 1}, 1);
      MFTL_Object *o = MFTLObjectCreate(0);
      MFTLEnsembleAddObject(solo, o);
      MFTLGenerateSphere(o, (double[3]){0.3, -0.2, 0.1}, 1.0, q);
      MFTL_BodyIntegrals bi;
      MFTLEnsembleBodyIntegrals(solo, &bi);
      err[q - 2] = fabs(bi.volume - 4.0 / 3.0 * MFTL_PI) / (4.0 / 3.0 * MFTL_PI);
      MFTLEnsembleDestroy(&solo);
    }
    printf("\n    sphere volume error at refine 2/3/4: %.2e %.2e %.2e (ratios %.2f %.2f) ", err[0], err[1], err[2], err[0] / err[1], err[1] / err[2]);
    ok = (err[0] / err[1] > 3.5 && err[1] / err[2] > 3.5);
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(ok);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_wall_closed_bodies_match_their_regions(void)
{
  const double lo[3] = {0.0, 0.0, -0.5}, hi[3] = {2.0, 1.5, 1.5};
  const bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  reference ref[3];

  /* 0: a flat sheet at height 0.62 above the floor, rims on the four side walls:
   *    V = 0.62 * 3 exactly. */
  const double flat[1] = {0.62};
  add_height_field(ens, h_flat, flat, 12, 9, MFTL_FALSE, &ref[0]);
  /* 1: a cap on the floor, rim a circle of radius 0.4. 2: the same hanging
   *    from the ceiling -- the reference wall comes from where the rim is. */
  const double cap[4] = {0.9, 0.7, 0.4, 0.3};
  add_cap(ens, cap, 7, MFTL_FALSE, &ref[1]);
  add_cap(ens, cap, 7, MFTL_TRUE, &ref[2]);
  MFTLPublishGeneratedBodies(ens);
  MFTLRedistribute(ens);
  const int shared = shared_bodies(ens);

  MFTL_BodyIntegrals rec[MAX_B];
  int identical = 0;
  integrate(ens, rec, &identical);
  const bool all[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  int ok = matches(&rec[0], &ref[0], all, 2.0, "sheet") && near(rec[0].volume, 0.62 * 3.0, 1.0, 1e-13);
  ok = ok && matches(&rec[1], &ref[1], all, 2.0, "floor cap") && matches(&rec[2], &ref[2], all, 2.0, "ceiling cap");
  ok = ok && rec[0].axis == 2 && rec[1].axis == 2 && rec[2].axis == 2 && !rec[0].closed;
  if (g_rank == 0 && !ok)
    for (int i = 0; i < 3; ++i)
      printf("\n  body %d: axis %d ref %.6g closed %d status %d V %.15g (ref %.15g) A %.15g (ref %.15g)", i, rec[i].axis, rec[i].reference, rec[i].closed,
             rec[i].volume_status, rec[i].volume, ref[i].volume, rec[i].area, ref[i].area);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(identical, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || shared > 0, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_periodic_wave_has_no_centroid_along_its_period(void)
{
  /* The travelling-wave shape: periodic in x, rims on the two y walls. The
   * volume below it and its y and z centroids are defined; x is not -- the body
   * wraps it. */
  const double lo[3] = {-2.0, -0.5, -1.0}, hi[3] = {2.0, 0.5, 1.0};
  const bool per[3] = {MFTL_TRUE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  reference ref;
  const double a[4] = {0.83, 0.17, 4.0, 1.0};
  add_height_field(ens, h_wave, a, 48, 12, MFTL_FALSE, &ref);
  MFTLPublishGeneratedBodies(ens);
  MFTLRedistribute(ens);

  MFTL_BodyIntegrals rec[MAX_B];
  int identical = 0;
  integrate(ens, rec, &identical);
  const bool yz[3] = {MFTL_FALSE, MFTL_TRUE, MFTL_TRUE};
  int ok = matches(&rec[0], &ref, yz, 2.0, "wave") && rec[0].centroid_status[0] == MFTL_INTEGRAL_UNDEFINED && rec[0].axis == 2;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(identical, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_statuses_say_what_cannot_be_measured(void)
{
  /* A flat disk floating mid-box -- a rim off every wall: UNDEFINED. A closed
   * sheet across a FULLY periodic box: UNSUPPORTED. The same sheet with walls
   * in z: measured from the floor, no centroid in x or y. Then a bridge and a
   * wrapping tube, below. */
  int ok = 1;
  {
    const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {2.0, 2.0, 2.0};
    const bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
    MFTL_Ensemble *ens = make_ens(lo, hi, per);
    reference r;
    const double cap[4] = {1.0, 1.0, 0.5, 0.0};
    add_cap(ens, cap, 5, MFTL_FALSE, &r);
    if (g_rank == 0)
    {
      /* lift it off the floor */
      MFTL_Object *o = ens->local_objects[0];
      for (int v = 0; v < o->verts.n; ++v) o->verts.pos[3 * v + 2] += 0.9;
      MFTLComputeTriangleGeometry(o, o->tris.n);  // the redistribution routes by the centres
    }
    MFTLPublishGeneratedBodies(ens);
    MFTLRedistribute(ens);
    MFTL_BodyIntegrals rec[MAX_B];
    int identical = 0;
    integrate(ens, rec, &identical);
    if (g_rank == 0) printf("\n  floating disk: status %d closed %d area %g", rec[0].volume_status, rec[0].closed, rec[0].area);
    ok = ok && identical && rec[0].volume_status == MFTL_INTEGRAL_UNDEFINED && rec[0].area > 0.0;
    for (int k = 0; k < 3; ++k) ok = ok && rec[0].centroid_status[k] == MFTL_INTEGRAL_UNDEFINED;
    MFTLEnsembleDestroy(&ens);
  }
  {
    const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {2.0, 2.0, 2.0};
    const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
    MFTL_Ensemble *ens = make_ens(lo, hi, per);
    reference r;
    const double a[4] = {0.9, 0.1, 2.0, 2.0};
    add_height_field(ens, h_wave, a, 12, 12, MFTL_FALSE, &r);  // periodic in x and y: closed on the torus
    MFTLPublishGeneratedBodies(ens);
    MFTLRedistribute(ens);
    MFTL_BodyIntegrals rec[MAX_B];
    int identical = 0;
    integrate(ens, rec, &identical);
    if (g_rank == 0) printf("\n  torus sheet: status %d closed %d axis %d", rec[0].volume_status, rec[0].closed, rec[0].axis);
    ok = ok && identical && rec[0].closed && rec[0].volume_status == MFTL_INTEGRAL_UNSUPPORTED;
    MFTLEnsembleDestroy(&ens);
  }
  {
    const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {2.0, 2.0, 2.0};
    const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_FALSE};
    MFTL_Ensemble *ens = make_ens(lo, hi, per);
    /* A closed sheet across x and y with walls in z: DEFINED, measured from the
     * floor; its x and y centroids UNDEFINED (it wraps both). */
    reference r;
    const double a[4] = {0.9, 0.1, 2.0, 2.0};
    add_height_field(ens, h_wave, a, 12, 12, MFTL_FALSE, &r);
    MFTLPublishGeneratedBodies(ens);
    MFTLRedistribute(ens);
    MFTL_BodyIntegrals rec[MAX_B];
    int identical = 0;
    integrate(ens, rec, &identical);
    const bool z_only[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_TRUE};
    if (g_rank == 0)
      printf("\n  walled sheet: status %d closed %d axis %d V %.15g (ref %.15g) cs %d %d %d", rec[0].volume_status, rec[0].closed, rec[0].axis, rec[0].volume,
             r.volume, rec[0].centroid_status[0], rec[0].centroid_status[1], rec[0].centroid_status[2]);
    ok = ok && identical && rec[0].closed && matches(&rec[0], &r, z_only, 2.0, "closed sheet") && rec[0].axis == 2;
    ok = ok && rec[0].centroid_status[0] == MFTL_INTEGRAL_UNDEFINED && rec[0].centroid_status[1] == MFTL_INTEGRAL_UNDEFINED;
    MFTLEnsembleDestroy(&ens);
  }
  {
    /* A liquid bridge floor to ceiling in a box periodic in x and y: rims on
     * both walls of the only walled axis -- UNSUPPORTED. And a closed tube
     * wrapping x in a box walled in y and z: it ENCLOSES (zero area vector),
     * volume = polygon area x period, centroid in y and z, none in x. */
    const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {2.0, 2.0, 2.0};
    const bool per[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_FALSE};
    MFTL_Ensemble *ens = make_ens(lo, hi, per);
    add_tube(ens, 2, 0.9, 1.1, 0.3, 24, 8);
    MFTLPublishGeneratedBodies(ens);
    MFTLRedistribute(ens);
    MFTL_BodyIntegrals rec[MAX_B];
    int identical = 0;
    integrate(ens, rec, &identical);
    if (g_rank == 0) printf("\n  bridge: status %d closed %d", rec[0].volume_status, rec[0].closed);
    ok = ok && identical && !rec[0].closed && rec[0].volume_status == MFTL_INTEGRAL_UNSUPPORTED;
    MFTLEnsembleDestroy(&ens);

    const double lo2[3] = {-1.0, 0.0, 0.0}, hi2[3] = {1.6, 2.0, 2.0};
    const bool per2[3] = {MFTL_TRUE, MFTL_FALSE, MFTL_FALSE};
    ens = make_ens(lo2, hi2, per2);
    add_tube(ens, 0, 0.8, 1.2, 0.35, 24, 20);
    MFTLPublishGeneratedBodies(ens);
    MFTLRedistribute(ens);
    integrate(ens, rec, &identical);
    reference r;
    r.volume = 0.5 * 24 * 0.35 * 0.35 * sin(2.0 * MFTL_PI / 24) * 2.6;
    r.area = rec[0].area;  // checked elsewhere
    r.centroid[0] = 0.0;
    r.centroid[1] = 0.8;
    r.centroid[2] = 1.2;
    const bool yz[3] = {MFTL_FALSE, MFTL_TRUE, MFTL_TRUE};
    if (g_rank == 0)
      printf("\n  wrapping tube: status %d closed %d axis %d V %.15g (ref %.15g) cs %d %d %d", rec[0].volume_status, rec[0].closed, rec[0].axis,
             rec[0].volume, r.volume, rec[0].centroid_status[0], rec[0].centroid_status[1], rec[0].centroid_status[2]);
    ok = ok && identical && rec[0].closed && matches(&rec[0], &r, yz, 2.0, "wrapping tube") && rec[0].centroid_status[0] == MFTL_INTEGRAL_UNDEFINED;
    MFTLEnsembleDestroy(&ens);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

static int test_no_bodies_and_bad_arguments(void)
{
  const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
  const bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  MFTL_Ensemble *ens = make_ens(lo, hi, per);
  TEST_ASSERT(agreed(MFTLEnsembleBodyIntegrals(ens, MFTL_NULL) == MFTL_SUCCESS, MPI_COMM_WORLD));  // nothing to fill
  TEST_ASSERT(agreed(MFTLEnsembleBodyIntegrals(MFTL_NULL, MFTL_NULL) == MFTL_ERROR, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Body integrals (np = %d) ===\n", g_nproc);

  RUN_TEST(test_closed_bodies_match_their_polyhedra);
  RUN_TEST(test_wall_closed_bodies_match_their_regions);
  RUN_TEST(test_a_periodic_wave_has_no_centroid_along_its_period);
  RUN_TEST(test_statuses_say_what_cannot_be_measured);
  RUN_TEST(test_no_bodies_and_bad_arguments);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
