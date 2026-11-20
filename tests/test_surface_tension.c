/**
 * @file test_surface_tension.c
 * @brief The surface-tension force on a distributed front: both Frenet forms
 *        with both contour normals (MFTLEnsembleComputeSurfaceTension()), the
 *        distributed curvature (MFTLEnsembleComputeCurvature()) and the
 *        per-body totals (MFTLEnsembleSurfaceTensionStats()).
 *
 * What each test is for:
 *
 *   partition    every owned force and curvature equals what ONE rank computes
 *                on the whole body before it is partitioned (by global id, to
 *                rounding: the scatter order changes), every ghost curvature
 *                its owner's to the bit, the net force of a closed body zero
 *                and, for the area gradient, its net torque; the totals call
 *                agrees with the test's own sums. Non-vacuous: owned vertices
 *                whose one-ring holds a ghost are counted;
 *   Marangoni    a flat sheet between two walls with sigma linear along it: no
 *                normal force anywhere, the total is delta sigma times the
 *                width in both forms, and every vertex of the vertex forms
 *                carries grad sigma times its barycentric area -- with the
 *                sigma ghosts deliberately poisoned, so a missing exchange
 *                fails; free rims sum to zero, excluded rims are zero;
 *   walls        a spherical cap on the floor presses on it with the Laplace
 *                pressure times its base (2 sigma pi a^2 / R) in every variant,
 *                and with free rims feels no net force;
 *   refusals     bad parameters and fields fail on every rank; an ensemble
 *                with no bodies returns without a collective out of step.
 *
 * Run at np = 1, 2, 4, 8 and 27 (tests/CMakeLists.txt); every assertion is an
 * agreed() verdict.
 */

#include <math.h>
#include <stdint.h>
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

/** Registered on every ensemble, in this order, before the first body. */
typedef struct
{
  int sigma;  // vertex, 1
  int force_v;  // vertex, 3
  int force_t;  // triangle, 3
  int curv;  // vertex, MFTL_CURVATURE_COMPONENTS
} Fields;

static MFTL_Ensemble *make_ens(const double lo[3], const double hi[3], const bool periodic[3], Fields *f)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 4);
  f->sigma = MFTLEnsembleRegisterField(ens, "sigma", MFTL_FIELD_VERTEX, 1, MFTL_FIELD_RECOMPUTED);
  f->force_v = MFTLEnsembleRegisterField(ens, "force_v", MFTL_FIELD_VERTEX, 3, MFTL_FIELD_RECOMPUTED);
  f->force_t = MFTLEnsembleRegisterField(ens, "force_t", MFTL_FIELD_TRIANGLE, 3, MFTL_FIELD_RECOMPUTED);
  f->curv = MFTLEnsembleRegisterField(ens, "curv", MFTL_FIELD_VERTEX, MFTL_CURVATURE_COMPONENTS, MFTL_FIELD_RECOMPUTED);
  return ens;
}

static double noise(uint64_t id, uint64_t salt)
{
  uint64_t z = id * 0x9E3779B97F4A7C15ull + salt * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * ((double) (z >> 11) / 9007199254740992.0) - 1.0;
}

static double dot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

/** @brief Build an indexed mesh on rank 0 (others stay empty) as body 0 of @p ens. */
static MFTL_Object *add_indexed(MFTL_Ensemble *ens, const double *pos, int nv, const int *tri, int nt)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (g_rank == 0)
    MFTLBuildIndexedMesh(obj, pos, nv, tri, nt);
  else
    MFTLObjectBuildDerived(obj);
  return obj;
}

/** @brief The four force variants: form x contour normal. */
static MFTL_SurfaceTensionParameters variant(int k)
{
  MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
  p.form = (k < 2) ? MFTL_TENSION_FRENET_VERTEX : MFTL_TENSION_FRENET_ELEMENT;
  p.contour_normal = (k % 2) ? MFTL_CONTOUR_NORMAL_VERTICES : MFTL_CONTOUR_NORMAL_TRIANGLES;
  return p;
}

static const char *const g_variant_name[4] = {"vertex/triangles", "vertex/vertices", "element/triangles", "element/vertices"};

/** @brief Owned vertices of @p obj with a ghost in their one-ring: what makes a distributed run non-vacuous here. */
static int count_cut_rings(const MFTL_Object *obj)
{
  int n = 0;
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    int ring[MFTL_MAX_VALENCE + 1];
    const int nr = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_MAX_VALENCE + 1, MFTL_FALSE);
    for (int i = 0; i < nr; ++i)
      if (ring[i] >= obj->verts.n_local)
      {
        n++;
        break;
      }
  }
  return n;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. Partition independence, conservation, totals
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_the_distributed_force_and_curvature_are_the_serial_ones(void)
{
  const double lo[3] = {-1.0, -1.0, -1.0}, hi[3] = {1.0, 1.0, 1.0};
  const bool per[3] = {false, false, false};
  Fields fd;
  MFTL_Ensemble *ens = make_ens(lo, hi, per, &fd);

  /* A noisy closed body centred where every grid's cut planes meet. */
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 0.6, 3);
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double s = 1.0 + 0.15 * noise((uint64_t) obj->verts.rec[v].global_id, 1);
    for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] *= s;
  }
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLObjectMarkChanged(obj);

  /* The serial reference: rank 0 holds the whole body before it is partitioned. */
  int counts[2] = {obj->next_vertex_gid, obj->next_triangle_gid};
  MPI_Bcast(counts, 2, MPI_INT, 0, MPI_COMM_WORLD);
  const int NV = counts[0], NT = counts[1];
  double *ref_f[4], *ref_c[2];
  for (int k = 0; k < 4; ++k) ref_f[k] = (double *) calloc((size_t) 3 * (k < 2 ? NV : NT), sizeof(double));
  for (int m = 0; m < 2; ++m) ref_c[m] = (double *) calloc((size_t) MFTL_CURVATURE_COMPONENTS * NV, sizeof(double));
  if (g_rank == 0)
  {
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_FIT);
    double *tmp = (double *) malloc((size_t) MFTL_CURVATURE_COMPONENTS * (NV > NT ? NV : NT) * sizeof(double));
    for (int k = 0; k < 4; ++k)
    {
      const MFTL_SurfaceTensionParameters p = variant(k);
      MFTLComputeSurfaceTension(obj, &p, tmp);
      const int rows = (k < 2) ? obj->verts.n : obj->tris.n;
      for (int i = 0; i < rows; ++i)
      {
        const int g = (k < 2) ? obj->verts.rec[i].global_id : obj->tris.rec[i].global_id;
        memcpy(&ref_f[k][3 * g], &tmp[3 * i], 3 * sizeof(double));
      }
    }
    for (int m = 0; m < 2; ++m)
    {
      MFTLComputeVertexCurvature(obj, (MFTL_CurvatureMethod) m, MFTL_CURVATURE_COMPONENTS, tmp, MFTL_NULL);
      for (int v = 0; v < obj->verts.n; ++v)
        memcpy(&ref_c[m][MFTL_CURVATURE_COMPONENTS * obj->verts.rec[v].global_id], &tmp[MFTL_CURVATURE_COMPONENTS * v],
               MFTL_CURVATURE_COMPONENTS * sizeof(double));
    }
    free(tmp);
  }
  for (int k = 0; k < 4; ++k) MPI_Bcast(ref_f[k], 3 * (k < 2 ? NV : NT), MPI_DOUBLE, 0, MPI_COMM_WORLD);
  for (int m = 0; m < 2; ++m) MPI_Bcast(ref_c[m], MFTL_CURVATURE_COMPONENTS * NV, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MFTLRedistribute(ens);
  int ok = 1;

  /* Forces: owned rows against the reference, conservation, totals. */
  for (int k = 0; k < 4; ++k)
  {
    const MFTL_SurfaceTensionParameters p = variant(k);
    const int field = (k < 2) ? fd.force_v : fd.force_t;
    ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, field) == MFTL_SUCCESS;

    double fmax_ref = 0.0;
    for (int i = 0; i < 3 * (k < 2 ? NV : NT); ++i) fmax_ref = fmax(fmax_ref, fabs(ref_f[k][i]));
    double sums[7] = {0, 0, 0, 0, 0, 0, 0}, worst = 0.0;  // F, x x F, sum |F|
    double sumx = 0.0;
    MFTL_Object *o = MFTLEnsembleFindLocalObject(ens, 0);
    if (o != MFTL_NULL)
    {
      const double *F = MFTLObjectField(o, field);
      const int n_own = (k < 2) ? o->verts.n_local : o->tris.n_local;
      for (int i = 0; i < n_own; ++i)
      {
        const int g = (k < 2) ? o->verts.rec[i].global_id : o->tris.rec[i].global_id;
        for (int d = 0; d < 3; ++d) worst = fmax(worst, fabs(F[3 * i + d] - ref_f[k][3 * g + d]));
        double x[3];
        if (k < 2)
          memcpy(x, &o->verts.pos[3 * i], sizeof(x));
        else
          MFTLGetTriangleCenter(o, i, x);
        const double tq[3] = {x[1] * F[3 * i + 2] - x[2] * F[3 * i + 1], x[2] * F[3 * i + 0] - x[0] * F[3 * i + 2], x[0] * F[3 * i + 1] - x[1] * F[3 * i + 0]};
        for (int d = 0; d < 3; ++d) sums[d] += F[3 * i + d], sums[3 + d] += tq[d];
        const double fa = sqrt(dot3(&F[3 * i], &F[3 * i]));
        sums[6] += fa;
        sumx += fa * sqrt(dot3(x, x));
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, sums, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &sumx, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    const double net = sqrt(dot3(sums, sums)) / sums[6], torque = sqrt(dot3(&sums[3], &sums[3])) / sumx;
    if (g_rank == 0) printf("\n    %-18s owned vs serial %.1e, net force %.1e, net torque %.1e", g_variant_name[k], worst / fmax_ref, net, torque);
    ok &= worst <= 1e-11 * fmax_ref;
    ok &= net <= 1e-13;
    if (k == 0) ok &= torque <= 1e-13;  // the area gradient is torque-free

    /* The library's totals: the same net force, and E = sigma A. */
    MFTL_SurfaceTensionStats st[4];
    memset(st, 0, sizeof(st));
    ok &= MFTLEnsembleSurfaceTensionStats(ens, &p, field, st) == MFTL_SUCCESS;
    if (o != MFTL_NULL)
    {
      const int slot = MFTLEnsembleFindLocalSlot(ens, 0);
      for (int d = 0; d < 3; ++d) ok &= fabs(st[slot].net_force[d] - sums[d]) <= 1e-14 * sums[6];
      ok &= fabs(st[slot].energy - p.sigma * st[slot].area) <= 1e-14 * st[slot].area && st[slot].area > 0.0;
    }
  }

  /* Curvature: owned rows against the reference, ghosts their owners' to the bit. */
  for (int m = 0; m < 2; ++m)
  {
    int fb = -1;
    ok &= MFTLEnsembleComputeCurvature(ens, (MFTL_CurvatureMethod) m, fd.curv, &fb) == MFTL_SUCCESS;
    double *tab = (double *) calloc((size_t) MFTL_CURVATURE_COMPONENTS * NV, sizeof(double));
    double worst = 0.0;
    MFTL_Object *o = MFTLEnsembleFindLocalObject(ens, 0);
    if (o != MFTL_NULL)
    {
      const double *C = MFTLObjectField(o, fd.curv);
      for (int v = 0; v < o->verts.n_local; ++v)
      {
        const int g = o->verts.rec[v].global_id;
        memcpy(&tab[MFTL_CURVATURE_COMPONENTS * g], &C[MFTL_CURVATURE_COMPONENTS * v], MFTL_CURVATURE_COMPONENTS * sizeof(double));
        for (int s = 0; s < 4; ++s)  // the scalar slots; the direction is arbitrary near umbilics
          worst = fmax(worst, fabs(C[MFTL_CURVATURE_COMPONENTS * v + s] - ref_c[m][MFTL_CURVATURE_COMPONENTS * g + s]));
      }
    }
    MPI_Allreduce(MPI_IN_PLACE, tab, MFTL_CURVATURE_COMPONENTS * NV, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    int ghosts = 1;
    if (o != MFTL_NULL)
    {
      const double *C = MFTLObjectField(o, fd.curv);
      for (int v = o->verts.n_local; v < o->verts.n; ++v)
        ghosts &= memcmp(&C[MFTL_CURVATURE_COMPONENTS * v], &tab[MFTL_CURVATURE_COMPONENTS * o->verts.rec[v].global_id],
                         MFTL_CURVATURE_COMPONENTS * sizeof(double)) == 0;
    }
    if (g_rank == 0) printf("\n    curvature %-5s  owned vs serial %.1e (kappa ~ 3.3)", m == 0 ? "COTAN" : "FIT", worst);
    ok &= worst <= 1e-10 && ghosts && fb == 0;
    free(tab);
  }

  MFTL_Object *o = MFTLEnsembleFindLocalObject(ens, 0);
  const int cut = (o != MFTL_NULL) ? count_cut_rings(o) : 0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || anywhere(cut > 0, MPI_COMM_WORLD), MPI_COMM_WORLD));

  for (int k = 0; k < 4; ++k) free(ref_f[k]);
  for (int m = 0; m < 2; ++m) free(ref_c[m]);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. Marangoni on a flat sheet between two walls
 * ═════════════════════════════════════════════════════════════════════════════ */

enum
{
  SHEET_NX = 24,  // cells along x, between the walls x = 0 and x = 1
  SHEET_NY = 20  // rows along the periodic y
};

/** @brief The sheet z = 0.37 across [0, 1] x [0, 1), rims on the walls x = 0 and x = 1, closed across the periodic y. */
static void add_sheet(MFTL_Ensemble *ens)
{
  const int nvx = SHEET_NX + 1, nv = nvx * SHEET_NY, nt = 2 * SHEET_NX * SHEET_NY;
  double *pos = (double *) malloc((size_t) 3 * nv * sizeof(double));
  int *tri = (int *) malloc((size_t) 3 * nt * sizeof(int));
  for (int j = 0; j < SHEET_NY; ++j)
    for (int i = 0; i < nvx; ++i)
    {
      const int v = j * nvx + i;
      /* An irregular grid: interior columns jittered, the wall columns exact. */
      const double jx = (i > 0 && i < SHEET_NX) ? 0.3 * noise((uint64_t) v, 5) : 0.0;
      pos[3 * v + 0] = (i + jx) / SHEET_NX;
      pos[3 * v + 1] = (j + 0.3 * noise((uint64_t) v, 6) + 0.5) / SHEET_NY;
      pos[3 * v + 2] = 0.37;
    }
  int t = 0;
  for (int j = 0; j < SHEET_NY; ++j)
    for (int i = 0; i < SHEET_NX; ++i)
    {
      const int a = j * nvx + i, b = a + 1, c = ((j + 1) % SHEET_NY) * nvx + i + 1, d = ((j + 1) % SHEET_NY) * nvx + i;
      tri[3 * t + 0] = a, tri[3 * t + 1] = b, tri[3 * t + 2] = c, ++t;
      tri[3 * t + 0] = a, tri[3 * t + 1] = c, tri[3 * t + 2] = d, ++t;
    }
  add_indexed(ens, pos, nv, tri, nt);
  free(pos);
  free(tri);
}

static int test_a_linear_sigma_on_a_flat_sheet_is_the_marangoni_force(void)
{
  const double lo[3] = {0.0, 0.0, 0.0}, hi[3] = {1.0, 1.0, 1.0};
  const bool per[3] = {false, true, false};
  Fields fd;
  MFTL_Ensemble *ens = make_ens(lo, hi, per, &fd);
  add_sheet(ens);
  MFTLRedistribute(ens);

  const double s0 = 1.0, g = 0.3;  // sigma = s0 + g x: delta sigma 0.3 across a width of 1
  int ok = 1, cut = 0;
  for (int pass = 0; pass < 2; ++pass)  // uniform sigma, then the linear field
    for (int k = 0; k < 4; ++k)
      for (int rim = 0; rim < 3; ++rim)
      {
        MFTL_SurfaceTensionParameters p = variant(k);
        p.rim = (MFTL_RimForce) rim;
        p.sigma = s0;
        p.sigma_field = pass ? fd.sigma : -1;
        const int field = (k < 2) ? fd.force_v : fd.force_t;
        MFTL_Object *o = MFTLEnsembleFindLocalObject(ens, 0);
        if (o != MFTL_NULL && pass)
        {
          /* Owned values only; ghosts poisoned, so the call must exchange them. */
          double *sg = MFTLObjectField(o, fd.sigma);
          for (int v = 0; v < o->verts.n; ++v) sg[v] = (v < o->verts.n_local) ? s0 + g * o->verts.pos[3 * v] : NAN;
        }
        ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, field) == MFTL_SUCCESS;

        double sums[4] = {0, 0, 0, 0}, worst_cell = 0.0, worst_z = 0.0, worst_rim = 0.0;
        if (o != MFTL_NULL)
        {
          const double *F = MFTLObjectField(o, field);
          const int n_own = (k < 2) ? o->verts.n_local : o->tris.n_local;
          double *bary = (double *) calloc((size_t) o->verts.n, sizeof(double));
          for (int t = 0; t < o->tris.n; ++t)
          {
            int v[3];
            MFTLGetTriangleVertices(o, t, v);
            for (int c = 0; c < 3; ++c) bary[v[c]] += MFTLGetTriangleArea(o, t) / 3.0;
          }
          for (int i = 0; i < n_own; ++i)
          {
            for (int d = 0; d < 3; ++d) sums[d] += F[3 * i + d];
            sums[3] += sqrt(dot3(&F[3 * i], &F[3 * i]));
            worst_z = fmax(worst_z, fabs(F[3 * i + 2]));
            bool on_rim = (k < 2) ? (o->verts.rec[i].flags & MFTL_FLAG_BOUNDARY) != 0
                                  : ((o->tris.he[3 * i].flags | o->tris.he[3 * i + 1].flags | o->tris.he[3 * i + 2].flags) & MFTL_FLAG_BOUNDARY) != 0;
            if (on_rim && rim == MFTL_RIM_FORCE_EXCLUDE) worst_rim = fmax(worst_rim, sqrt(dot3(&F[3 * i], &F[3 * i])));
            /* Each vertex: grad sigma times its barycentric area (zero for uniform sigma). */
            if (k < 2 && rim == MFTL_RIM_FORCE_WALL)
            {
              const double want[3] = {pass ? g * bary[i] : 0.0, 0.0, 0.0};
              for (int d = 0; d < 3; ++d) worst_cell = fmax(worst_cell, fabs(F[3 * i + d] - want[d]) / (g * bary[i]));
            }
          }
          free(bary);
          cut = count_cut_rings(o);
        }
        MPI_Allreduce(MPI_IN_PLACE, sums, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double w[3] = {worst_cell, worst_z, worst_rim};
        MPI_Allreduce(MPI_IN_PLACE, w, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        ok &= w[1] == 0.0;  // in the plane, exactly
        if (rim == MFTL_RIM_FORCE_WALL)
        {
          const double want = pass ? g : 0.0;  // delta sigma times the width
          ok &= fabs(sums[0] - want) <= 1e-13 && fabs(sums[1]) <= 1e-13;
          ok &= w[0] <= 1e-11;
          if (g_rank == 0 && pass) printf("\n    %-18s wall: total x %.17f (want %.1f), worst cell %.1e", g_variant_name[k], sums[0], want, w[0]);
        }
        else if (rim == MFTL_RIM_FORCE_FREE)
          ok &= sqrt(dot3(sums, sums)) <= 1e-13 * sums[3];
        else
          ok &= w[2] == 0.0;
      }

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || anywhere(cut > 0, MPI_COMM_WORLD), MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. A cap on the floor
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief A spherical cap of radius @p R and base radius R sin(psi) standing on the floor z = lo: a centre and @p n rings. */
static void add_cap(MFTL_Ensemble *ens, double R, double psi, int n)
{
  const double floor_z = ens->domain.global_min[2], d = R * cos(psi);
  const int nv = 1 + 3 * n * (n + 1);
  double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
  int *tri = (int *) malloc((size_t) 6 * n * n * 3 * sizeof(int));
  int *first = (int *) malloc((size_t) (n + 1) * sizeof(int));
  int v = 0;
  for (int i = 0; i <= n; ++i)
  {
    first[i] = v;
    const int m = (i == 0) ? 1 : 6 * i;
    const double a = psi * (double) i / n;
    for (int j = 0; j < m; ++j, ++v)
    {
      const double th = (i == 0) ? 0.0 : 2.0 * MFTL_PI * j / m;
      pos[3 * v + 0] = R * sin(a) * cos(th);
      pos[3 * v + 1] = R * sin(a) * sin(th);
      pos[3 * v + 2] = (i == n) ? floor_z : floor_z + (R * cos(a) - d);
    }
  }
  /* Ring i-1 (ma vertices) to ring i (mb = 6i): walk both by angle. */
  int t = 0;
  for (int i = 1; i <= n; ++i)
  {
    const int ma = (i == 1) ? 1 : 6 * (i - 1), mb = 6 * i;
    int ja = 0, jb = 0;
    while (jb < mb || (ma > 1 && ja < ma))
    {
      const double next_a = (ma == 1) ? 1e9 : 2.0 * MFTL_PI * (ja + 1) / ma, next_b = 2.0 * MFTL_PI * (jb + 1) / mb;
      const int a0 = first[i - 1] + ja % ma, b0 = first[i] + jb % mb;
      int c2;
      if (jb < mb && (ja >= ma || next_b <= next_a))
        c2 = first[i] + (jb + 1) % mb, jb++;
      else
        c2 = first[i - 1] + (ja + 1) % ma, ja++;
      tri[3 * t + 0] = a0, tri[3 * t + 1] = b0, tri[3 * t + 2] = c2, t++;
    }
  }
  add_indexed(ens, pos, nv, tri, t);
  free(first);
  free(tri);
  free(pos);
}

static int test_a_cap_presses_on_its_wall_with_the_laplace_pressure(void)
{
  const double lo[3] = {-1.0, -1.0, -1.0}, hi[3] = {1.0, 1.0, 1.0};
  const bool per[3] = {false, false, false};
  const double R = 0.8, psi = MFTL_PI / 3.0, a = R * sin(psi), sigma = 0.07;
  /* The upward normal: a pressure on the floor is a force along -z. */
  const double want = -2.0 * sigma * MFTL_PI * a * a / R;
  double err[2][4];
  int ok = 1;
  for (int level = 0; level < 2; ++level)
  {
    Fields fd;
    MFTL_Ensemble *ens = make_ens(lo, hi, per, &fd);
    add_cap(ens, R, psi, level ? 24 : 12);
    MFTLRedistribute(ens);
    for (int k = 0; k < 4; ++k)
      for (int rim = 0; rim < 2; ++rim)
      {
        MFTL_SurfaceTensionParameters p = variant(k);
        p.sigma = sigma;
        p.rim = (MFTL_RimForce) rim;
        const int field = (k < 2) ? fd.force_v : fd.force_t;
        ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, field) == MFTL_SUCCESS;
        MFTL_SurfaceTensionStats st[2];
        memset(st, 0, sizeof(st));
        ok &= MFTLEnsembleSurfaceTensionStats(ens, &p, field, st) == MFTL_SUCCESS;
        double net[4] = {0, 0, 0, 0};
        const int slot = MFTLEnsembleFindLocalSlot(ens, 0);
        if (slot >= 0) memcpy(net, st[slot].net_force, 3 * sizeof(double)), net[3] = 1.0;
        MPI_Allreduce(MPI_IN_PLACE, net, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);  // the holders agree: average them
        for (int d = 0; d < 3; ++d) net[d] /= net[3];
        if (rim == MFTL_RIM_FORCE_WALL)
        {
          err[level][k] = net[2] / want - 1.0;
          ok &= fabs(net[0]) <= 1e-12 * fabs(want) && fabs(net[1]) <= 1e-12 * fabs(want);
        }
        else
          ok &= sqrt(dot3(net, net)) <= 1e-13 * fabs(want);
      }
    MFTLEnsembleDestroy(&ens);
  }
  /* The rim's conormal comes from a one-sided normal (a triangle's, or a fit on
   * half a ring): first order, so halving the edges halves the error. */
  for (int k = 0; k < 4; ++k)
  {
    if (g_rank == 0) printf("\n    %-18s wall: net z relative error %+.2e (12 rings), %+.2e (24 rings)", g_variant_name[k], err[0][k], err[1][k]);
    ok &= fabs(err[1][k]) < 0.6 * fabs(err[0][k]) && fabs(err[1][k]) < 1.5e-2;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Refusals
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_bad_parameters_are_refused_everywhere(void)
{
  const double lo[3] = {-1.0, -1.0, -1.0}, hi[3] = {1.0, 1.0, 1.0};
  const bool per[3] = {false, false, false};
  Fields fd;
  MFTL_Ensemble *ens = make_ens(lo, hi, per, &fd);

  /* No bodies yet: nothing to do, no collective out of step. */
  MFTL_SurfaceTensionParameters p = MFTLSurfaceTensionDefaultParameters();
  int ok = MFTLEnsembleComputeSurfaceTension(ens, &p, fd.force_v) == MFTL_SUCCESS;
  ok &= MFTLEnsembleComputeCurvature(ens, MFTL_CURVATURE_FIT, fd.curv, MFTL_NULL) == MFTL_SUCCESS;
  ok &= MFTLEnsembleSurfaceTensionStats(ens, &p, fd.force_v, MFTL_NULL) == MFTL_SUCCESS;

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const double c[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, c, 0.5, 2);
  MFTLRedistribute(ens);

  ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, fd.force_t) == MFTL_ERROR;  // a triangle field for the vertex form
  ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, fd.curv) == MFTL_ERROR;  // seven components
  ok &= MFTLEnsembleComputeSurfaceTension(ens, MFTL_NULL, fd.force_v) == MFTL_ERROR;
  p.sigma_field = fd.force_v;  // three components
  ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, fd.force_v) == MFTL_ERROR;
  p = MFTLSurfaceTensionDefaultParameters();
  p.contour_normal = (MFTL_ContourNormal) 5;
  ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, fd.force_v) == MFTL_ERROR;
  ok &= MFTLEnsembleComputeCurvature(ens, MFTL_CURVATURE_COTAN, fd.force_v, MFTL_NULL) == MFTL_ERROR;  // 3 components
  ok &= MFTLEnsembleComputeCurvature(ens, MFTL_CURVATURE_COTAN, fd.force_t, MFTL_NULL) == MFTL_ERROR;  // a triangle field
  ok &= MFTLEnsembleComputeCurvature(ens, (MFTL_CurvatureMethod) 4, fd.curv, MFTL_NULL) == MFTL_ERROR;
  ok &= MFTLEnsembleComputeCurvature(MFTL_NULL, MFTL_CURVATURE_COTAN, fd.curv, MFTL_NULL) == MFTL_ERROR;

  /* A rank holding the body that passes no output still takes part, then refuses. */
  p = MFTLSurfaceTensionDefaultParameters();
  ok &= MFTLEnsembleComputeSurfaceTension(ens, &p, fd.force_v) == MFTL_SUCCESS;
  const int rc = MFTLEnsembleSurfaceTensionStats(ens, &p, fd.force_v, MFTL_NULL);
  ok &= rc == ((ens->n_local_objects > 0) ? MFTL_ERROR : MFTL_SUCCESS);

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  RUN_TEST(test_the_distributed_force_and_curvature_are_the_serial_ones);
  RUN_TEST(test_a_linear_sigma_on_a_flat_sheet_is_the_marangoni_force);
  RUN_TEST(test_a_cap_presses_on_its_wall_with_the_laplace_pressure);
  RUN_TEST(test_bad_parameters_are_refused_everywhere);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  if (g_rank == 0) TEST_REPORT();
  return (g_tests_failed > 0) ? 1 : 0;
}
