/*
 * @Author: Christian Gorges
 * @Date: 2026-10-03
 *
 * @brief Tests for the rim curve: where the remesher puts a vertex of an open
 *        surface's rim, and the application's rim constraint.
 *
 * A split, a collapse or a smoothing move on the rim used to take the CHORD of
 * two rim vertices: exact on a straight rim and on a planar wall, and inside
 * a curved rim by h^2/(8R) every time -- which a remesher does every pass. The
 * rim is now the average of the circular arcs through the rim's own vertices.
 * What is asserted, and why each needs its own fixture:
 *
 *   a CIRCULAR rim keeps its radius TO ROUND-OFF through twenty passes that
 *     keep splitting and collapsing it (a disk alternately refined and
 *     coarsened, so the spacing is uneven all the time), with and without a
 *     projector onto the circle;
 *   a rim ON A WALL keeps its wall coordinate to the bit through fifty passes,
 *     on a dome whose interior leaves the wall plane;
 *   a STRAIGHT rim stays straight and its CORNERS never move;
 *   UNEVEN spacing (10 : 1 : 1) does not overshoot the edge -- the uniform
 *     parameterisation does, and that is the regression;
 *   a split next to a SLIVER, whose apex lies between the chord and the arc,
 *     falls back to the chord rather than turn two triangles over;
 *   a projector that REFUSES stops every rim operation, and a CLOSED surface
 *     never reaches the rim path at all;
 *   distributed, the rim stays on its constraint where it crosses ranks, and
 *     ranks that disagree about having a projector refuse to remesh.
 *
 * Registered at np = 1, 2 and 4: the serial fixtures run on every rank on
 * MPI_COMM_SELF; the distributed ones on the whole job.
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

/* ═════════════════════════════════════════════════════════════════════════════
 * Fixtures
 * ═════════════════════════════════════════════════════════════════════════════ */

/** A non-periodic box; the disk and dome sit with their rim on z = Z_WALL, its floor. */
#define Z_WALL (-0.63)

static MFTL_Ensemble *make_ens(MPI_Comm comm)
{
  int np = 1;
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double lo[3] = {-1.7, -1.6, Z_WALL}, hi[3] = {1.9, 1.8, Z_WALL + 2.0};
  bool per[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  return MFTLEnsembleCreate(comm, cuts, lo, hi, per, 4);
}

/**
 * @brief A polar disk of radius @p R about (0.1, 0.05): a centre and @p n rings,
 *        ring i holding 6i vertices, zipped by angle. Height z0 + dome (1 -
 *        r^2/R^2), so the rim lies on z = z0 exactly and @p dome = 0 is flat.
 *        Normals point to +z. Built on rank 0 of @p ens only.
 */
static MFTL_Object *add_disk(MFTL_Ensemble *ens, double R, int n, double z0, double dome)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (ens->domain.rank != 0)
  {
    MFTLObjectBuildDerived(obj);
    return obj;
  }

  const int nv = 1 + 3 * n * (n + 1);
  const int nt = 6 * n * n;
  double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
  double *ang = (double *) malloc((size_t) nv * sizeof(double));
  int *tri = (int *) malloc((size_t) nt * 3 * sizeof(int));
  int *first = (int *) malloc((size_t) (n + 1) * sizeof(int));

  const double cx = 0.1, cy = 0.05;
  pos[0] = cx;
  pos[1] = cy;
  pos[2] = z0 + dome;
  ang[0] = 0.0;
  first[0] = 0;
  int v = 1;
  for (int i = 1; i <= n; ++i)
  {
    first[i] = v;
    const int m = 6 * i;
    const double r = R * (double) i / (double) n;
    for (int j = 0; j < m; ++j, ++v)
    {
      const double a = 2.0 * MFTL_PI * (double) j / (double) m;
      ang[v] = a;
      pos[3 * v + 0] = cx + r * cos(a);
      pos[3 * v + 1] = cy + r * sin(a);
      const double s = (double) i / (double) n;
      pos[3 * v + 2] = z0 + dome * (1.0 - s * s);
    }
  }

  int t = 0;
  for (int j = 0; j < 6; ++j)
  {
    tri[3 * t + 0] = 0;
    tri[3 * t + 1] = first[1] + j;
    tri[3 * t + 2] = first[1] + (j + 1) % 6;
    t++;
  }
  for (int i = 2; i <= n; ++i)
  {
    const int ma = 6 * (i - 1), mb = 6 * i;
    int ja = 0, jb = 0;
    while (ja < ma || jb < mb)
    {
      const double next_a = (ja + 1 < ma) ? ang[first[i - 1] + ja + 1] : 2.0 * MFTL_PI;
      const double next_b = (jb + 1 < mb) ? ang[first[i] + jb + 1] : 2.0 * MFTL_PI;
      const int a0 = first[i - 1] + ja % ma, b0 = first[i] + jb % mb;
      if (jb < mb && (ja == ma || next_b <= next_a))
      {
        tri[3 * t + 0] = a0;
        tri[3 * t + 1] = b0;
        tri[3 * t + 2] = first[i] + (jb + 1) % mb;
        jb++;
      }
      else
      {
        tri[3 * t + 0] = a0;
        tri[3 * t + 1] = b0;
        tri[3 * t + 2] = first[i - 1] + (ja + 1) % ma;
        ja++;
      }
      t++;
    }
  }

  MFTLBuildIndexedMesh(obj, pos, nv, tri, t);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  free(first);
  free(tri);
  free(ang);
  free(pos);
  return obj;
}

/** @brief A flat open strip: rows y = 0 and y = 1 over the given x stations, z = 0. */
static MFTL_Object *add_strip(MFTL_Ensemble *ens, const double *x, int n)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double *pos = (double *) malloc((size_t) 2 * n * 3 * sizeof(double));
  int *tri = (int *) malloc((size_t) 2 * (n - 1) * 3 * sizeof(int));
  for (int i = 0; i < n; ++i)
  {
    pos[3 * i + 0] = x[i];
    pos[3 * i + 1] = 0.0;
    pos[3 * i + 2] = 0.0;
    pos[3 * (n + i) + 0] = x[i];
    pos[3 * (n + i) + 1] = 1.0;
    pos[3 * (n + i) + 2] = 0.0;
  }
  int t = 0;
  for (int i = 0; i + 1 < n; ++i)
  {
    const int b0 = i, b1 = i + 1, t0 = n + i, t1 = n + i + 1;
    tri[3 * t + 0] = b0;
    tri[3 * t + 1] = b1;
    tri[3 * t + 2] = t1;
    t++;
    tri[3 * t + 0] = b0;
    tri[3 * t + 1] = t1;
    tri[3 * t + 2] = t0;
    t++;
  }
  MFTLBuildIndexedMesh(obj, pos, 2 * n, tri, t);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  free(tri);
  free(pos);
  return obj;
}

/** @brief The rim's distance from the disk axis, over every live boundary vertex. */
static void rim_radius(const MFTL_Object *obj, int n_verts, double *rmin, double *rmax, int *n_rim)
{
  *rmin = 1e300;
  *rmax = -1e300;
  *n_rim = 0;
  for (int v = 0; v < n_verts; ++v)
  {
    if (!MFTLVertexIsMeshBoundary(obj, v) || (obj->verts.rec[v].flags & MFTL_FLAG_DEAD)) continue;
    const double *p = MFTLVertexPos(obj, v);
    const double r = hypot(p[0] - 0.1, p[1] - 0.05);
    *rmin = fmin(*rmin, r);
    *rmax = fmax(*rmax, r);
    (*n_rim)++;
  }
}

/** @brief Every rim half-edge is twin-less, and the rim is one loop: as many rim vertices as rim edges. */
static bool rim_is_one_clean_loop(const MFTL_Object *obj)
{
  int n_edges = 0, n_verts = 0;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
    if (obj->tris.he[h].twin_idx == MFTL_ERROR)
    {
      if (!MFTLHeIsMeshBoundary(obj, h)) return MFTL_FALSE;
      n_edges++;
    }
  for (int v = 0; v < obj->verts.n; ++v)
    if (MFTLVertexIsMeshBoundary(obj, v)) n_verts++;
  return n_edges > 0 && n_edges == n_verts;
}

/** @brief Projector state: the circle, and how often it was asked. */
typedef struct
{
  double R;
  long calls;
  bool refuse;
} circle_user;

/** @brief Onto the circle of radius R about the disk axis, in its plane (a pure function). */
static int onto_circle(const MFTL_Domain *dom, double p[3], void *user)
{
  (void) dom;
  circle_user *u = (circle_user *) user;
  u->calls++;
  if (u->refuse) return MFTL_ERROR;
  const double dx = p[0] - 0.1, dy = p[1] - 0.05;
  const double r = hypot(dx, dy);
  if (r > 0.0)
  {
    p[0] = 0.1 + dx * (u->R / r);
    p[1] = 0.05 + dy * (u->R / r);
  }
  return MFTL_SUCCESS;
}

static MFTL_RemeshParameters params_at(double target)
{
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, target, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  p.smooth_iterations = 2;
  return p;
}

/** @brief Fewest and most rim vertices seen after any pass of the last churn(): its non-vacuity measure. */
static int g_rim_lo, g_rim_hi;

/** @brief Remeshes @p passes times, alternating a fine and a coarse target: the rim is split and collapsed every pass. */
static void churn(MFTL_Object *obj, int passes, double fine, double coarse, const circle_user *hook)
{
  g_rim_lo = 1 << 30;
  g_rim_hi = 0;
  for (int pass = 0; pass < passes; ++pass)
  {
    MFTL_RemeshParameters p = params_at((pass % 2) ? coarse : fine);
    if (hook != MFTL_NULL)
    {
      p.rim_project = onto_circle;
      p.rim_project_user = (void *) hook;
    }
    MFTL_RemeshStats st;
    MFTLRemeshObject(obj, &p, &st);
    double r0, r1;
    int n_rim;
    rim_radius(obj, obj->verts.n, &r0, &r1, &n_rim);
    g_rim_lo = MFTL_MIN(g_rim_lo, n_rim);
    g_rim_hi = MFTL_MAX(g_rim_hi, n_rim);
  }
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Serial
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_circular_rim_keeps_its_radius(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = add_disk(ens, 1.0, 8, 0.37, 0.0);
  MFTLRemeshResetRimFallbacks();

  const double coarse = 0.2;
  churn(obj, 20, 0.07, coarse, MFTL_NULL);

  double rmin, rmax;
  int n_rim;
  rim_radius(obj, obj->verts.n, &rmin, &rmax, &n_rim);
  /* One chord split of an edge of length h sits h^2 / (8R) inside the circle. */
  const double one_chord = coarse * coarse / 8.0;
  bool flat = MFTL_TRUE;
  for (int v = 0; v < obj->verts.n; ++v) flat = flat && obj->verts.pos[3 * v + 2] == 0.37;

  printf("\n    circular rim after 20 passes: r in 1 + [%.1e, %.1e], rim %d .. %d vertices; one chord split is %.1e inside ", rmin - 1.0, rmax - 1.0,
         g_rim_lo, g_rim_hi, one_chord);
  TEST_ASSERT(rim_is_one_clean_loop(obj));
  TEST_ASSERT(g_rim_lo < 40 && g_rim_hi > 80);  // the rim was coarsened AND refined: non-vacuity of the rim operations
  /* Every circle through three points of a circle is that circle: the rim
   * stays on it to round-off, against 5e-3 for ONE chord split. */
  TEST_ASSERT(fabs(rmin - 1.0) < 1e-12 && fabs(rmax - 1.0) < 1e-12);
  TEST_ASSERT(flat);  // a planar surface stays in its plane to the bit
  TEST_ASSERT(MFTLRemeshRimFallbacks() == 0);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_circular_rim_with_its_projector_stays_on_the_circle(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = add_disk(ens, 1.0, 8, 0.37, 0.0);
  circle_user hook = {1.0, 0, MFTL_FALSE};
  churn(obj, 20, 0.07, 0.2, &hook);

  double rmin, rmax;
  int n_rim;
  rim_radius(obj, obj->verts.n, &rmin, &rmax, &n_rim);
  printf("\n    with the projector: r in 1 + [%.1e, %.1e], %ld projections ", rmin - 1.0, rmax - 1.0, hook.calls);
  TEST_ASSERT(hook.calls > 100);
  TEST_ASSERT(fabs(rmin - 1.0) < 1e-14 && fabs(rmax - 1.0) < 1e-14);
  TEST_ASSERT(rim_is_one_clean_loop(obj));
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_rim_on_a_wall_keeps_the_wall_to_the_bit(void)
{
  /* A dome: the interior rises off the floor, the rim lies on it. Fifty passes
   * of splits, collapses, flips and smoothing must not move one rim vertex off
   * z = Z_WALL by a single bit. */
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = add_disk(ens, 1.0, 8, Z_WALL, 0.45);
  churn(obj, 50, 0.08, 0.18, MFTL_NULL);

  int n_rim = 0, off_wall = 0, interior_on_wall = 0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    if (obj->verts.rec[v].flags & MFTL_FLAG_DEAD) continue;
    const double z = obj->verts.pos[3 * v + 2];
    if (MFTLVertexIsMeshBoundary(obj, v))
    {
      n_rim++;
      if (z != Z_WALL) off_wall++;
    }
    else if (z == Z_WALL)
      interior_on_wall++;
  }
  double rmin, rmax;
  int nr;
  rim_radius(obj, obj->verts.n, &rmin, &rmax, &nr);
  printf("\n    dome rim after 50 passes: %d rim vertices (%d .. %d), %d off the wall, r in 1 + [%.1e, %.1e] ", n_rim, g_rim_lo, g_rim_hi, off_wall,
         rmin - 1.0, rmax - 1.0);
  TEST_ASSERT(g_rim_lo < g_rim_hi);
  TEST_ASSERT(n_rim > 0 && off_wall == 0);
  TEST_ASSERT(interior_on_wall == 0);
  TEST_ASSERT(fabs(rmin - 1.0) < 1e-12 && fabs(rmax - 1.0) < 1e-12);
  TEST_ASSERT(rim_is_one_clean_loop(obj));
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_straight_rim_stays_straight_and_its_corners_stay(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  const double x[6] = {0.0, 0.2, 0.4, 0.6, 0.8, 1.0};
  MFTL_Object *obj = add_strip(ens, x, 6);
  const double corner[4][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};
  churn(obj, 30, 0.06, 0.15, MFTL_NULL);

  int bad = 0, corners_found = 0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    if (obj->verts.rec[v].flags & MFTL_FLAG_DEAD) continue;
    const double *p = MFTLVertexPos(obj, v);
    if (p[2] != 0.0) bad++;
    if (!MFTLVertexIsMeshBoundary(obj, v)) continue;
    /* Exactly on one of the four sides. */
    if (!(p[0] == 0.0 || p[0] == 1.0 || p[1] == 0.0 || p[1] == 1.0)) bad++;
    for (int c = 0; c < 4; ++c)
      if (p[0] == corner[c][0] && p[1] == corner[c][1]) corners_found++;
  }
  printf("\n    straight rim: %d vertices off their side, %d of 4 corners in place ", bad, corners_found);
  TEST_ASSERT(bad == 0 && corners_found == 4);
  TEST_ASSERT(rim_is_one_clean_loop(obj));
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_uneven_spacing_does_not_overshoot(void)
{
  /* Rim stations 0, 1, 1.1, 1.2: spacing 10 : 1 : 1, collinear. A uniform
   * Catmull-Rom point of the middle edge is (-P0 + 9 P1 + 9 P2 - P3) / 16 =
   * 1.10625, past the edge's own end at 1.1 -- a parameterised spline has to be
   * told about spacing. Arcs through collinear points are the line, so the
   * split lands on the chord midpoint, 1.05, exactly. */
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  const double x[5] = {0.0, 1.0, 1.1, 1.2, 1.3};
  MFTL_Object *obj = add_strip(ens, x, 5);

  int h = MFTL_ERROR;
  for (int k = 0; k < 3 * obj->tris.n && h == MFTL_ERROR; ++k)
  {
    if (obj->tris.he[k].twin_idx != MFTL_ERROR) continue;
    const double *a = MFTLVertexPos(obj, MFTLHeOrigin(obj, k));
    const double *b = MFTLVertexPos(obj, MFTLHeTarget(obj, k));
    if (a[1] == 0.0 && b[1] == 0.0 && fmin(a[0], b[0]) == 1.0 && fmax(a[0], b[0]) == 1.1) h = k;
  }
  TEST_ASSERT(h != MFTL_ERROR);
  const int m = MFTLSplitEdge(obj, h, MFTL_NULL);
  TEST_ASSERT(m != MFTL_ERROR);
  const double *p = MFTLVertexPos(obj, m);
  printf("\n    10:1:1 split at x = %.17g (uniform Catmull-Rom: 1.10625) ", p[0]);
  TEST_ASSERT(p[0] > 1.0 && p[0] < 1.1);
  TEST_ASSERT(fabs(p[0] - 1.05) < 1e-15);
  TEST_ASSERT(p[1] == 0.0 && p[2] == 0.0);
  MFTLObjectBuildDerived(obj);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A strip on the outside of a circle of radius 1: the circle is its rim
 *        p-a-b-s (a hole's edge), and the rim triangle over a-b has its apex
 *        at radius @p rc on the bisector. Returns the half-edge between a and b.
 */
static int add_hole_edge(MFTL_Ensemble *ens, double rc, MFTL_Object **out)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const double th = 0.3;
  const double ang[4] = {-1.5 * th, -0.5 * th, 0.5 * th, 1.5 * th};
  double pos[3 * 8];
  for (int i = 0; i < 4; ++i)
  {
    pos[3 * i + 0] = cos(ang[i]);
    pos[3 * i + 1] = sin(ang[i]);
    pos[3 * i + 2] = 0.2;
  }
  const double cpos[4][2] = {{rc, 0.0}, {1.6 * cos(-1.5 * th), 1.6 * sin(-1.5 * th)}, {1.6, 0.0}, {1.6 * cos(1.5 * th), 1.6 * sin(1.5 * th)}};
  for (int i = 0; i < 4; ++i)
  {
    pos[3 * (4 + i) + 0] = cpos[i][0];
    pos[3 * (4 + i) + 1] = cpos[i][1];
    pos[3 * (4 + i) + 2] = 0.2;
  }
  enum
  {
    I0,
    I1,
    I2,
    I3,
    C,
    U0,
    U1,
    U2
  };
  int tri[6][3] = {{I0, I1, U0}, {I1, U1, U0}, {I1, C, U1}, {I1, I2, C}, {I2, U1, C}, {I2, U2, U1}};
  int all[7][3];
  for (int t = 0; t < 6; ++t)
    for (int k = 0; k < 3; ++k) all[t][k] = tri[t][k];
  all[6][0] = I2;
  all[6][1] = I3;
  all[6][2] = U2;
  for (int t = 0; t < 7; ++t)
  {
    /* Every triangle counter-clockwise seen from +z: one consistent orientation. */
    const double *a = &pos[3 * all[t][0]], *b = &pos[3 * all[t][1]], *c = &pos[3 * all[t][2]];
    if ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.0)
    {
      const int x = all[t][1];
      all[t][1] = all[t][2];
      all[t][2] = x;
    }
  }
  MFTLBuildIndexedMesh(obj, pos, 8, &all[0][0], 7);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  *out = obj;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (obj->tris.he[h].twin_idx != MFTL_ERROR) continue;
    const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
    if ((a == I1 && b == I2) || (a == I2 && b == I1)) return h;
  }
  return MFTL_ERROR;
}

/** @brief Every live triangle's normal points to +z: nothing turned over. */
static bool all_face_up(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double n[3];
    MFTLGetTriangleNormal(obj, t, n);
    if (!(n[2] > 0.0)) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

static int test_a_rim_split_next_to_a_sliver_falls_back_to_the_chord(void)
{
  /* The arc over a-b reaches radius 1 at the bisector; the chord midpoint is at
   * cos(0.15) = 0.98877. An apex at 0.99214 -- between the two -- is a sliver
   * the arc point would pass: both new triangles would face down. */
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = MFTL_NULL;
  const double chord = cos(0.15);
  const int h = add_hole_edge(ens, chord + 0.3 * (1.0 - chord), &obj);
  TEST_ASSERT(h != MFTL_ERROR && all_face_up(obj));
  const int m = MFTLSplitEdge(obj, h, MFTL_NULL);
  TEST_ASSERT(m != MFTL_ERROR);
  const double *p = MFTLVertexPos(obj, m);
  printf("\n    sliver: split at (%.17g, %.3g), chord midpoint %.17g ", p[0], p[1], chord);
  TEST_ASSERT(fabs(p[0] - chord) < 1e-15 && fabs(p[1]) < 1e-15);
  TEST_ASSERT(all_face_up(obj));

  /* The control: an apex well clear of the arc, and the same split lands ON the circle. */
  MFTL_Object *obj2 = MFTL_NULL;
  const int h2 = add_hole_edge(ens, 1.3, &obj2);
  const int m2 = MFTLSplitEdge(obj2, h2, MFTL_NULL);
  const double *q = MFTLVertexPos(obj2, m2);
  printf("clear: split at radius %.17g ", hypot(q[0], q[1]));
  TEST_ASSERT(fabs(hypot(q[0], q[1]) - 1.0) < 1e-15 && all_face_up(obj2));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_refusing_projector_freezes_the_rim(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = add_disk(ens, 1.0, 8, 0.37, 0.0);

  /* The rim, as a sorted list of positions. */
  double before[256][2];
  int nb = 0;
  for (int v = 0; v < obj->verts.n; ++v)
    if (MFTLVertexIsMeshBoundary(obj, v) && nb < 256)
    {
      before[nb][0] = obj->verts.pos[3 * v];
      before[nb][1] = obj->verts.pos[3 * v + 1];
      nb++;
    }

  circle_user hook = {1.0, 0, MFTL_TRUE};
  churn(obj, 6, 0.07, 0.2, &hook);

  int n_rim = 0, moved = 0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    if (!MFTLVertexIsMeshBoundary(obj, v) || (obj->verts.rec[v].flags & MFTL_FLAG_DEAD)) continue;
    n_rim++;
    bool found = MFTL_FALSE;
    for (int i = 0; i < nb && !found; ++i) found = (before[i][0] == obj->verts.pos[3 * v] && before[i][1] == obj->verts.pos[3 * v + 1]);
    if (!found) moved++;
  }
  printf("\n    refused: %ld refusals, rim %d -> %d vertices, %d moved ", hook.calls, nb, n_rim, moved);
  TEST_ASSERT(hook.calls > 0);
  TEST_ASSERT(n_rim == nb && moved == 0);
  TEST_ASSERT(inv_mesh_is_sound(obj, 0));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_closed_surface_never_reaches_the_rim(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.1, 0.1, 0.4};
  MFTLGenerateSphere(obj, c, 0.8, 3);
  circle_user hook = {1.0, 0, MFTL_FALSE};
  MFTLRemeshResetRimFallbacks();
  churn(obj, 4, 0.08, 0.2, &hook);
  TEST_ASSERT(hook.calls == 0 && MFTLRemeshRimFallbacks() == 0);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Distributed
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_distributed_rim_stays_on_its_circle(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD);
  add_disk(ens, 1.0, 8, 0.37, 0.0);
  MFTLPublishGeneratedBodies(ens);
  MFTLRedistribute(ens);
  MFTLRemeshResetRimFallbacks();

  circle_user hook = {1.0, 0, MFTL_FALSE};
  int rim_shared = 0;
  for (int pass = 0; pass < 6; ++pass)
  {
    MFTL_RemeshParameters p = params_at((pass % 2) ? 0.2 : 0.07);
    p.parallel_phase_budget = 8;
    p.rim_project = onto_circle;
    p.rim_project_user = &hook;
    MFTL_RemeshStats st;
    TEST_ASSERT(agreed(MFTLRemeshEnsemble(ens, &p, &st) == MFTL_SUCCESS, MPI_COMM_WORLD));

    /* Non-vacuity: a rim half-edge in a GHOST triangle means the rim crosses ranks here. */
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      const MFTL_Object *obj = ens->local_objects[o];
      for (int h = 3 * obj->tris.n_local; h < 3 * obj->tris.n; ++h)
        if (MFTLHeIsMeshBoundary(obj, h)) rim_shared = 1;
    }
  }

  int ok = 1;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    double rmin, rmax;
    int n_rim;
    rim_radius(obj, obj->verts.n_local, &rmin, &rmax, &n_rim);
    if (n_rim > 0 && (fabs(rmin - 1.0) > 1e-14 || fabs(rmax - 1.0) > 1e-14)) ok = 0;
    for (int v = 0; v < obj->verts.n; ++v)
      if (obj->verts.pos[3 * v + 2] != 0.37) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(inv_partition_is_sound(test_body(ens, 0), g_rank), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLRemeshRimFallbacks() == 0, MPI_COMM_WORLD));
  TEST_ASSERT(g_nproc == 1 || anywhere(rim_shared, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_ranks_disagreeing_about_the_projector_refuse_to_remesh(void)
{
  if (g_nproc < 2) return 0;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD);
  add_disk(ens, 1.0, 6, 0.37, 0.0);
  MFTLPublishGeneratedBodies(ens);
  MFTLRedistribute(ens);

  circle_user hook = {1.0, 0, MFTL_FALSE};
  MFTL_RemeshParameters p = params_at(0.1);
  if (g_rank == 0)
  {
    p.rim_project = onto_circle;
    p.rim_project_user = &hook;
  }
  MFTL_RemeshStats st;
  const int rc = MFTLRemeshEnsemble(ens, &p, &st);
  TEST_ASSERT(agreed(rc == MFTL_ERROR, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== Rim curves (np = %d) ===\n", g_nproc);

  RUN_TEST(test_a_circular_rim_keeps_its_radius);
  RUN_TEST(test_a_circular_rim_with_its_projector_stays_on_the_circle);
  RUN_TEST(test_a_rim_on_a_wall_keeps_the_wall_to_the_bit);
  RUN_TEST(test_a_straight_rim_stays_straight_and_its_corners_stay);
  RUN_TEST(test_uneven_spacing_does_not_overshoot);
  RUN_TEST(test_a_rim_split_next_to_a_sliver_falls_back_to_the_chord);
  RUN_TEST(test_a_refusing_projector_freezes_the_rim);
  RUN_TEST(test_a_closed_surface_never_reaches_the_rim);
  RUN_TEST(test_a_distributed_rim_stays_on_its_circle);
  RUN_TEST(test_ranks_disagreeing_about_the_projector_refuse_to_remesh);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
