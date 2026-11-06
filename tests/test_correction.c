/*
 * @Author: Christian Gorges
 * @Date: 2026-11-05
 *
 * @brief Tests for the front corrections: TSUR3D undulation removal
 *        (MFTLEnsembleRemoveUndulations) and the volume correction
 *        (MFTLEnsembleCorrectVolumes) with the reference volume that travels
 *        with a body.
 *
 * What each test is for:
 *
 *   one edge         the operation itself, isolated by fixing every other
 *                    vertex: the volume to round-off, both ends moved along ONE
 *                    direction, the edge left perpendicular to it, neither end
 *                    moved further than the edge's own kink;
 *   noise            the purpose: a radially jittered sphere and a jittered
 *                    periodic sheet go back towards a sphere and a plane, with
 *                    the volume conserved to round-off; a flat sheet is a fixed
 *                    point to the bit;
 *   the seam         the same jittered sphere centred on the corner of a
 *                    periodic box ends where the unshifted one does;
 *   fields           a conservative density keeps its amount per triangle, a
 *                    vertex field its values;
 *   distributed      the volume at np = 1 ... 27, ghost copies equal to their
 *                    owners to the bit, the colour rounds non-vacuous, and the
 *                    smoothing as good as on one rank;
 *   volume           a shrunken sphere and a sunk cap on the floor restored to
 *                    round-off in three Newton steps, the cap's rim still on the
 *                    wall, at every rank count;
 *   reference        it reaches ranks that never held the body, and two bodies
 *                    that coalesce hand their sum to the one they become.
 *
 * Every volume is MFTLEnsembleBodyIntegrals()'s, whose own references are
 * tetrahedra built from unwrapped vertex lists (tests/test_body_integrals.c).
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
 * Fixtures and measurements
 * ═════════════════════════════════════════════════════════════════════════════ */

static MFTL_Ensemble *make_ens(MPI_Comm comm, double lo, double hi, bool periodic)
{
  int np = 1;
  MPI_Comm_size(comm, &np);
  int cuts[3];
  test_grid_for(np, cuts);
  double l[3] = {lo, lo, lo}, h[3] = {hi, hi, hi};
  bool p[3] = {periodic, periodic, periodic};
  return MFTLEnsembleCreate(comm, cuts, l, h, p, 4);
}

/** @brief A uniform number in [-1, 1] from an id: the same on every rank, whatever the partition. */
static double noise(uint64_t id, uint64_t salt)
{
  uint64_t z = id * 0x9E3779B97F4A7C15ull + salt * 0xD1B54A32D192ED03ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * ((double) (z >> 11) / 9007199254740992.0) - 1.0;
}

/** @brief An icosphere about @p c, each vertex moved radially by amp * R * noise(gid). */
static void add_noisy_sphere(MFTL_Ensemble *ens, const double c[3], double R, int refine, double amp)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double cc[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(obj, cc, R, refine);
  const MFTL_Domain *dom = &ens->domain;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    double d[3];
    MFTLGetMinimumImageVector(dom, d, c, &obj->verts.pos[3 * v]);
    const double s = 1.0 + amp * noise((uint64_t) obj->verts.rec[v].global_id, 1);
    for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] = c[k] + s * d[k];
    MFTLApplyPeriodicity(dom, &obj->verts.pos[3 * v]);
  }
  /* The vertices moved: the centres a redistribution routes by must follow. */
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  MFTLObjectMarkChanged(obj);
}

/** @brief A sheet z = z0 + amp * noise across a box periodic in x and y, built on rank 0. */
static void add_noisy_sheet(MFTL_Ensemble *ens, int nx, double z0, double amp)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const MFTL_Domain *dom = &ens->domain;
  int rank = 0;
  MPI_Comm_rank(dom->comm_world, &rank);
  if (rank == 0)
  {
    const double dx = dom->domain_size[0] / nx, dy = dom->domain_size[1] / nx;
    double *pos = (double *) malloc((size_t) nx * nx * 3 * sizeof(double));
    int *tri = (int *) malloc((size_t) 2 * nx * nx * 3 * sizeof(int));
    for (int j = 0; j < nx; ++j)
      for (int i = 0; i < nx; ++i)
      {
        const int v = j * nx + i;
        pos[3 * v + 0] = dom->global_min[0] + (i + 0.5 * (j % 2)) * dx;
        pos[3 * v + 1] = dom->global_min[1] + j * dy;
        pos[3 * v + 2] = z0 + amp * noise((uint64_t) v, 2);
        MFTLApplyPeriodicity(dom, &pos[3 * v]);
      }
    int t = 0;
    for (int j = 0; j < nx; ++j)
      for (int i = 0; i < nx; ++i)
      {
        const int i1 = (i + 1) % nx, j1 = (j + 1) % nx;
        const int v00 = j * nx + i, v10 = j * nx + i1, v01 = j1 * nx + i, v11 = j1 * nx + i1;
        if (j % 2 == 0)
        {
          const int q[2][3] = {{v00, v10, v01}, {v10, v11, v01}};
          memcpy(&tri[3 * t], q, sizeof(q));
        }
        else
        {
          const int q[2][3] = {{v00, v11, v01}, {v00, v10, v11}};
          memcpy(&tri[3 * t], q, sizeof(q));
        }
        t += 2;
      }
    MFTLBuildIndexedMesh(obj, pos, nx * nx, tri, t);
    free(tri);
    free(pos);
  }
  else
    MFTLObjectBuildDerived(obj);
}

/** @brief A paraboloid cap of rim radius R and peak height H standing on the floor: centre + n rings. */
static void add_cap(MFTL_Ensemble *ens, double cx, double cy, double R, double H, int n)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  const MFTL_Domain *dom = &ens->domain;
  int rank = 0;
  MPI_Comm_rank(dom->comm_world, &rank);
  if (rank == 0)
  {
    const double wall = dom->global_min[2];
    const int nv = 1 + 3 * n * (n + 1);
    double *pos = (double *) malloc((size_t) nv * 3 * sizeof(double));
    int *tri = (int *) malloc((size_t) 6 * n * n * 3 * sizeof(int));
    int *first = (int *) malloc((size_t) (n + 1) * sizeof(int));
    int v = 0;
    for (int i = 0; i <= n; ++i)
    {
      first[i] = v;
      const int m = (i == 0) ? 1 : 6 * i;
      for (int j = 0; j < m; ++j, ++v)
      {
        const double th = (i == 0) ? 0.0 : 2.0 * MFTL_PI * j / m, rr = R * (double) i / n, s = (double) i / n;
        pos[3 * v + 0] = cx + rr * cos(th);
        pos[3 * v + 1] = cy + rr * sin(th);
        pos[3 * v + 2] = wall + H * (1.0 - s * s);
      }
    }
    /* Ring i-1 (m_a vertices) to ring i (m_b = 6i): walk both by angle. */
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
        tri[3 * t + 1] = b0;
        tri[3 * t + 2] = c2;
        t++;
      }
    }
    MFTLBuildIndexedMesh(obj, pos, nv, tri, t);
    free(first);
    free(tri);
    free(pos);
  }
  else
    MFTLObjectBuildDerived(obj);
}

/** @brief Body @p gid's volume, the same on every rank (its holders agree; MAX over the job). */
static double body_volume(MFTL_Ensemble *ens, int gid)
{
  MFTL_BodyIntegrals bi[16];
  MFTLEnsembleBodyIntegrals(ens, bi);
  double v = -HUGE_VAL;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (ens->local_objects[o]->global_id == gid) v = bi[o].volume;
  MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, ens->domain.comm_world);
  return v;
}

/** @brief RMS over owned vertices of |x - c| - R, with R the radius of the sphere of volume @p V. */
static double radial_rms(MFTL_Ensemble *ens, const double c[3], double V)
{
  const double R = cbrt(3.0 * V / (4.0 * MFTL_PI));
  double s[2] = {0.0, 0.0};
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      double d[3];
      MFTLGetMinimumImageVector(&ens->domain, d, c, &obj->verts.pos[3 * v]);
      const double e = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) - R;
      s[0] += e * e;
      s[1] += 1.0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, s, 2, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world);
  return sqrt(s[0] / s[1]);
}

/** @brief RMS of z about its mean over owned vertices. */
static double height_rms(MFTL_Ensemble *ens)
{
  double s[3] = {0.0, 0.0, 0.0};
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double z = obj->verts.pos[3 * v + 2];
      s[0] += z;
      s[1] += z * z;
      s[2] += 1.0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, s, 3, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world);
  const double m = s[0] / s[2];
  return sqrt(fmax(s[1] / s[2] - m * m, 0.0));
}

/**
 * @brief Every ghost vertex position equals its owner's to the bit, for every
 *        body: owned positions scattered into a table by global id and summed
 *        over the job (exact: each slot has one non-zero contribution).
 */
static int ghosts_agree(MFTL_Ensemble *ens)
{
  int ok = 1;
  for (int gid = 0; gid < ens->n_bodies_global; ++gid)
  {
    MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, gid);
    int n = obj ? obj->next_vertex_gid : 0;
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_MAX, ens->domain.comm_world);
    double *tab = (double *) calloc((size_t) MFTL_MAX(3 * n, 1), sizeof(double));
    if (obj)
      for (int v = 0; v < obj->verts.n_local; ++v) memcpy(&tab[3 * obj->verts.rec[v].global_id], &obj->verts.pos[3 * v], 3 * sizeof(double));
    MPI_Allreduce(MPI_IN_PLACE, tab, 3 * n, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world);
    if (obj)
      for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
        if (memcmp(&tab[3 * obj->verts.rec[v].global_id], &obj->verts.pos[3 * v], 3 * sizeof(double)) != 0) ok = 0;
    free(tab);
  }
  return ok;
}

/** @brief The published method: every edge, whatever its kink (the library's default skips kinks up to 15 degrees). */
static MFTL_UndulationParameters every_edge(int sweeps)
{
  MFTL_UndulationParameters p = MFTLUndulationDefaultParameters();
  p.min_angle = 0.0;
  p.sweeps = sweeps;
  return p;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. The edge operation
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One TSUR3D operation, isolated: every vertex but two neighbours is
 *        flagged as a feature, so exactly one edge is eligible.
 */
static int test_one_edge_conserves_the_volume_and_flattens_the_edge(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  const double c[3] = {0.1, -0.05, 0.2};
  add_noisy_sphere(ens, c, 1.0, 3, 0.03);
  MFTL_Object *obj = ens->local_objects[0];

  /* The edge: half-edge 0's ends. */
  const int a = MFTLHeOrigin(obj, 0), b = MFTLHeTarget(obj, 0);
  for (int v = 0; v < obj->verts.n; ++v)
    if (v != a && v != b) obj->verts.rec[v].flags |= MFTL_FLAG_FEATURE;
  double pa0[3], pb0[3];
  memcpy(pa0, &obj->verts.pos[3 * a], sizeof(pa0));
  memcpy(pb0, &obj->verts.pos[3 * b], sizeof(pb0));
  double *before = (double *) malloc((size_t) obj->verts.n * 3 * sizeof(double));
  memcpy(before, obj->verts.pos, (size_t) obj->verts.n * 3 * sizeof(double));
  const double V0 = body_volume(ens, 0);

  MFTL_UndulationStats st;
  const MFTL_UndulationParameters p1 = every_edge(1);
  ok = ok && MFTLRemoveUndulationsObject(obj, &p1, &st) == MFTL_SUCCESS && st.n_moved == 1;
  const double V1 = body_volume(ens, 0);

  int others_still = 1;
  for (int v = 0; v < obj->verts.n; ++v)
    if (v != a && v != b && memcmp(&before[3 * v], &obj->verts.pos[3 * v], 3 * sizeof(double)) != 0) others_still = 0;
  double da[3], db[3], e1[3], e0[3];
  for (int k = 0; k < 3; ++k)
  {
    da[k] = obj->verts.pos[3 * a + k] - pa0[k];
    db[k] = obj->verts.pos[3 * b + k] - pb0[k];
    e0[k] = pb0[k] - pa0[k];
    e1[k] = obj->verts.pos[3 * b + k] - obj->verts.pos[3 * a + k];
  }
  const double la = sqrt(MFTLVectorInProd(da, da)), lb = sqrt(MFTLVectorInProd(db, db));
  double cr[3];
  MFTLVectorCrossVector(cr, da, db);
  const double par = sqrt(MFTLVectorInProd(cr, cr)) / (la * lb);  // sine of the angle between the moves
  double u[3] = {da[0] / la, da[1] / la, da[2] / la};
  const double perp = fabs(MFTLVectorInProd(e1, u)) / sqrt(MFTLVectorInProd(e1, e1));
  const double kink = fabs(MFTLVectorInProd(e0, u));
  /* The direction is the smoothed normal, which on a sphere is nearly radial at the midpoint. */
  double m[3];
  for (int k = 0; k < 3; ++k) m[k] = 0.5 * (pa0[k] + pb0[k]) - c[k];
  const double radial = fabs(MFTLVectorInProd(m, u)) / sqrt(MFTLVectorInProd(m, m));
  if (g_rank == 0)
    printf("\n    dV/V %.1e, moves %.2e %.2e (kink %.2e), sin(between) %.1e, edge.n %.1e, n.radial %.4f ", fabs(V1 - V0) / V0, la, lb, kink, par, perp,
           radial);
  ok = ok && others_still && fabs(V1 - V0) <= 1e-15 * V0 && par < 1e-12 && perp < 1e-12 && la <= kink && lb <= kink && la > 0.0 && radial > 0.99;
  free(before);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. What it is for: noise
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_jittered_sphere_goes_back_to_a_sphere_at_constant_volume(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  const double c[3] = {0.1, -0.05, 0.2};
  add_noisy_sphere(ens, c, 1.0, 4, 0.01);
  const double V0 = body_volume(ens, 0);
  const double r0 = radial_rms(ens, c, V0);

  /* The clean sphere's own discretisation error is the floor smoothing cannot go below. */
  MFTL_Ensemble *clean = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  add_noisy_sphere(clean, c, 1.0, 4, 0.0);
  const double floor_rms = radial_rms(clean, c, body_volume(clean, 0));
  MFTL_UndulationStats sc;
  const MFTL_UndulationParameters pc = every_edge(1);
  MFTLEnsembleRemoveUndulations(clean, &pc, &sc);
  const double h = 2.0 * MFTL_PI / (5.0 * 16.0);  // icosphere edge length at refine 4, roughly

  MFTL_UndulationParameters p = every_edge(1);
  double rms[6];
  rms[0] = r0;
  double worst_dv = 0.0;
  for (int s = 1; s <= 5; ++s)
  {
    MFTL_UndulationStats st;
    ok = ok && MFTLEnsembleRemoveUndulations(ens, &p, &st) == MFTL_SUCCESS && st.n_moved > 0 && st.n_refused == 0;
    const double V = body_volume(ens, 0);
    worst_dv = fmax(worst_dv, fabs(V - V0) / V0);
    rms[s] = radial_rms(ens, c, V);
  }
  if (g_rank == 0)
    printf("\n    radial rms %.2e -> %.2e %.2e %.2e %.2e %.2e (clean sphere %.2e), worst dV/V %.1e, clean sphere moved %.1e h ", rms[0], rms[1], rms[2],
           rms[3], rms[4], rms[5], floor_rms, worst_dv, sc.max_displacement / h);
  /* The noise above the clean sphere's own discretisation error: TSUR3D damps
   * the grid-scale modes in one sweep, the smoother ones more slowly. */
  double noise_rms[6];
  for (int s = 0; s <= 5; ++s) noise_rms[s] = sqrt(fmax(rms[s] * rms[s] - floor_rms * floor_rms, 0.0));
  if (g_rank == 0) printf("\n    noise above it %.2e -> %.2e after one sweep, %.2e after five ", noise_rms[0], noise_rms[1], noise_rms[5]);
  ok = ok && worst_dv < 1e-13 && noise_rms[1] < 0.5 * noise_rms[0] && noise_rms[5] < 0.25 * noise_rms[0];
  for (int s = 1; s <= 5; ++s) ok = ok && rms[s] <= rms[s - 1];
  /* A resolved sphere is nearly a fixed point: the correction is third order. */
  ok = ok && sc.max_displacement < 0.01 * h;
  MFTLEnsembleDestroy(&clean);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

static int test_a_jittered_sheet_flattens_at_its_mean_height(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_SELF, (int[3]){1, 1, 1}, (double[3]){0.0, 0.0, 0.0}, (double[3]){1.0, 1.0, 1.0},
                                          (bool[3]){MFTL_TRUE, MFTL_TRUE, MFTL_FALSE}, 1);
  add_noisy_sheet(ens, 24, 0.4, 0.004);
  MFTLPublishGeneratedBodies(ens);
  const double V0 = body_volume(ens, 0);
  const double z0 = height_rms(ens);
  MFTL_UndulationParameters p = every_edge(10);
  MFTL_UndulationStats st;
  ok = ok && MFTLEnsembleRemoveUndulations(ens, &p, &st) == MFTL_SUCCESS && st.n_moved > 0;
  const double V1 = body_volume(ens, 0), z1 = height_rms(ens);
  if (g_rank == 0) printf("\n    height rms %.2e -> %.2e, mean height %.15f -> %.15f ", z0, z1, V0, V1);
  ok = ok && fabs(V1 - V0) < 1e-13 * V0 && z1 < 0.25 * z0;

  /* A flat sheet is a fixed point, bit for bit: nothing has a kink. */
  MFTL_Ensemble *flat = MFTLEnsembleCreate(MPI_COMM_SELF, (int[3]){1, 1, 1}, (double[3]){0.0, 0.0, 0.0}, (double[3]){1.0, 1.0, 1.0},
                                           (bool[3]){MFTL_TRUE, MFTL_TRUE, MFTL_FALSE}, 1);
  add_noisy_sheet(flat, 12, 0.4, 0.0);
  MFTLPublishGeneratedBodies(flat);
  MFTL_Object *fo = flat->local_objects[0];
  double *before = (double *) malloc((size_t) fo->verts.n * 3 * sizeof(double));
  memcpy(before, fo->verts.pos, (size_t) fo->verts.n * 3 * sizeof(double));
  MFTLEnsembleRemoveUndulations(flat, &p, &st);
  if (g_rank == 0) printf("flat: %lld moved, %lld flat ", st.n_moved, st.n_flat);
  ok = ok && st.n_moved == 0 && st.n_flat > 0 && memcmp(before, fo->verts.pos, (size_t) fo->verts.n * 3 * sizeof(double)) == 0;
  free(before);
  MFTLEnsembleDestroy(&flat);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/** @brief The jittered sphere centred on the corner of a periodic box ends where the unshifted one does. */
static int test_a_sphere_on_the_periodic_seam_is_smoothed_as_anywhere_else(void)
{
  int ok = 1;
  const double away[3] = {0.0, 0.0, 0.0}, corner[3] = {2.0, 2.0, 2.0};
  MFTL_Ensemble *e0 = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_TRUE);
  MFTL_Ensemble *e1 = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_TRUE);
  add_noisy_sphere(e0, away, 0.8, 4, 0.01);
  add_noisy_sphere(e1, corner, 0.8, 4, 0.01);
  const double V0 = body_volume(e0, 0), V1 = body_volume(e1, 0);
  MFTL_UndulationParameters p = every_edge(3);
  MFTL_UndulationStats s0, s1;
  MFTLEnsembleRemoveUndulations(e0, &p, &s0);
  MFTLEnsembleRemoveUndulations(e1, &p, &s1);
  const double W0 = body_volume(e0, 0), W1 = body_volume(e1, 0);
  const double r0 = radial_rms(e0, away, W0), r1 = radial_rms(e1, corner, W1);
  /* Vertex by vertex, as displacements from the centre. */
  double worst = 0.0;
  const MFTL_Object *o0 = e0->local_objects[0], *o1 = e1->local_objects[0];
  for (int v = 0; v < o0->verts.n; ++v)
  {
    double d0[3], d1[3];
    MFTLGetMinimumImageVector(&e0->domain, d0, away, &o0->verts.pos[3 * v]);
    MFTLGetMinimumImageVector(&e1->domain, d1, corner, &o1->verts.pos[3 * v]);
    for (int k = 0; k < 3; ++k) worst = fmax(worst, fabs(d0[k] - d1[k]));
  }
  if (g_rank == 0) printf("\n    rms %.3e vs %.3e on the seam, largest vertex difference %.1e, moves %lld vs %lld ", r0, r1, worst, s0.n_moved, s1.n_moved);
  ok = fabs(W0 - V0) < 1e-13 * V0 && fabs(W1 - V1) < 1e-13 * V1 && fabs(V0 - V1) < 1e-12 * V0 && worst < 1e-12 && s0.n_moved == s1.n_moved;
  MFTLEnsembleDestroy(&e0);
  MFTLEnsembleDestroy(&e1);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief What the 15-degree default is for: a resolved surface is left alone
 *        to the bit, and a grid-scale zigzag on it is still removed.
 *
 * A clean icosphere at refinement 4 has kinks third order in the edge length,
 * far below 15 degrees, so the default must not move a single vertex; the same
 * sphere jittered at 5 % of the radius (kinks of tens of degrees, as velocity
 * noise makes them) must still lose more than half of its noise, at constant
 * volume, down to the level the threshold leaves (measured 0.41x in 5 sweeps).
 */
static int test_the_default_spares_a_resolved_surface_and_still_removes_noise(void)
{
  int ok = 1;
  const MFTL_UndulationParameters def = MFTLUndulationDefaultParameters();
  ok = ok && def.min_angle == 15.0;
  const double c[3] = {0.1, -0.05, 0.2};

  MFTL_Ensemble *clean = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  add_noisy_sphere(clean, c, 1.0, 4, 0.0);
  MFTL_Object *co = clean->local_objects[0];
  double *before = (double *) malloc((size_t) co->verts.n * 3 * sizeof(double));
  memcpy(before, co->verts.pos, (size_t) co->verts.n * 3 * sizeof(double));
  MFTL_UndulationStats sc;
  MFTLEnsembleRemoveUndulations(clean, MFTL_NULL, &sc);
  ok = ok && sc.n_moved == 0 && sc.n_flat > 0 && memcmp(before, co->verts.pos, (size_t) co->verts.n * 3 * sizeof(double)) == 0;
  free(before);

  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  add_noisy_sphere(ens, c, 1.0, 4, 0.05);
  const double floor_rms = radial_rms(clean, c, body_volume(clean, 0));
  const double V0 = body_volume(ens, 0);
  const double n0 = sqrt(fmax(pow(radial_rms(ens, c, V0), 2) - floor_rms * floor_rms, 0.0));
  MFTL_UndulationParameters p = MFTLUndulationDefaultParameters();
  p.sweeps = 5;
  MFTL_UndulationStats st;
  MFTLEnsembleRemoveUndulations(ens, &p, &st);
  const double V1 = body_volume(ens, 0);
  const double n1 = sqrt(fmax(pow(radial_rms(ens, c, V1), 2) - floor_rms * floor_rms, 0.0));
  if (g_rank == 0)
    printf("\n    clean sphere: %lld moved, %lld flat; 5 %% jitter: noise %.2e -> %.2e in 5 sweeps (%lld moves), dV/V %.1e ", sc.n_moved, sc.n_flat, n0, n1,
           st.n_moved, fabs(V1 - V0) / V0);
  /* The threshold bounds what is left: kinks under 15 degrees on edges of length
   * h leave about tan(15) h / 2 of height noise, ~1 % R at refinement 4. */
  const double h = 2.0 * MFTL_PI / (5.0 * 16.0);
  ok = ok && st.n_moved > 0 && n1 < 0.5 * n0 && n1 < 0.6 * tan(15.0 * MFTL_PI / 180.0) * h && fabs(V1 - V0) < 1e-13 * V0;
  MFTLEnsembleDestroy(&ens);
  MFTLEnsembleDestroy(&clean);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. Fields
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_a_density_keeps_its_amount_per_triangle(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_SELF, -2.0, 2.0, MFTL_FALSE);
  const int fd = MFTLEnsembleRegisterField(ens, "surfactant", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_CONSERVATIVE_DENSITY);
  const int fv = MFTLEnsembleRegisterField(ens, "temperature", MFTL_FIELD_VERTEX, 2, MFTL_FIELD_INTERPOLATED);
  const double c[3] = {0.0, 0.0, 0.0};
  add_noisy_sphere(ens, c, 1.0, 3, 0.03);
  MFTL_Object *obj = ens->local_objects[0];
  const int nt = obj->tris.n, nv = obj->verts.n;
  double *gam = MFTLObjectField(obj, fd), *tv = MFTLObjectField(obj, fv);
  double *m0 = (double *) malloc((size_t) nt * sizeof(double));
  double total0 = 0.0;
  for (int t = 0; t < nt; ++t)
  {
    double ctr[3];
    MFTLGetTriangleCenter(obj, t, ctr);
    gam[t] = 1.0 + 0.5 * sin(3.0 * ctr[0]) * cos(2.0 * ctr[2]);
    m0[t] = gam[t] * MFTLGetTriangleArea(obj, t);
    total0 += m0[t];
  }
  for (int v = 0; v < nv; ++v)
  {
    tv[2 * v] = obj->verts.pos[3 * v];
    tv[2 * v + 1] = 7.0;
  }
  double *tv0 = (double *) malloc((size_t) nv * 2 * sizeof(double));
  memcpy(tv0, tv, (size_t) nv * 2 * sizeof(double));

  MFTL_UndulationParameters p = every_edge(3);
  MFTL_UndulationStats st;
  MFTLEnsembleRemoveUndulations(ens, &p, &st);
  obj = ens->local_objects[0];
  gam = MFTLObjectField(obj, fd);
  tv = MFTLObjectField(obj, fv);
  double worst = 0.0, total1 = 0.0;
  int changed = 0;
  for (int t = 0; t < nt; ++t)
  {
    const double m1 = gam[t] * MFTLGetTriangleArea(obj, t);
    worst = fmax(worst, fabs(m1 - m0[t]) / m0[t]);
    total1 += m1;
    if (m1 != m0[t] || gam[t] != gam[t]) changed++;
  }
  if (g_rank == 0) printf("\n    amount per triangle to %.1e, total to %.1e, %lld moves ", worst, fabs(total1 - total0) / total0, st.n_moved);
  ok = st.n_moved > 0 && worst < 1e-14 && fabs(total1 - total0) < 1e-14 * total0 && memcmp(tv0, tv, (size_t) nv * 2 * sizeof(double)) == 0;
  free(tv0);
  free(m0);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Distributed
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_the_distributed_sweep_is_exact_and_agrees(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, -1.0, 1.0, MFTL_TRUE);
  const int fd = MFTLEnsembleRegisterField(ens, "surfactant", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_CONSERVATIVE_DENSITY);
  const double c[3] = {0.03, -0.02, 0.01};
  add_noisy_sphere(ens, c, 0.6, 4, 0.01);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    double *g = MFTLObjectField(obj, fd);
    for (int t = 0; t < obj->tris.n; ++t) g[t] = 1.0 + 0.25 * noise((uint64_t) obj->tris.rec[t].global_id, 3);
  }
  double amount0 = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    const double *g = MFTLObjectField(obj, fd);
    for (int t = 0; t < obj->tris.n_local; ++t) amount0 += g[t] * MFTLGetTriangleArea(obj, t);
  }
  MPI_Allreduce(MPI_IN_PLACE, &amount0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  const double V0 = body_volume(ens, 0);

  MFTL_UndulationParameters p = every_edge(3);
  MFTL_UndulationStats st;
  ok = MFTLEnsembleRemoveUndulations(ens, &p, &st) == MFTL_SUCCESS;
  const double V1 = body_volume(ens, 0);
  const double rms = radial_rms(ens, c, V1);

  double amount1 = 0.0;
  int dens_agree = 1;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    const double *g = MFTLObjectField(obj, fd);
    for (int t = 0; t < obj->tris.n_local; ++t) amount1 += g[t] * MFTLGetTriangleArea(obj, t);
  }
  MPI_Allreduce(MPI_IN_PLACE, &amount1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  /* Ghost copies of the density equal their owners', by triangle global id. */
  {
    MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, 0);
    int n = obj ? obj->next_triangle_gid : 0;
    MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    double *tab = (double *) calloc((size_t) MFTL_MAX(n, 1), sizeof(double));
    if (obj)
    {
      const double *g = MFTLObjectField(obj, fd);
      for (int t = 0; t < obj->tris.n_local; ++t) tab[obj->tris.rec[t].global_id] = g[t];
    }
    MPI_Allreduce(MPI_IN_PLACE, tab, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (obj)
    {
      const double *g = MFTLObjectField(obj, fd);
      for (int t = obj->tris.n_local; t < obj->tris.n; ++t)
        if (tab[obj->tris.rec[t].global_id] != g[t]) dens_agree = 0;
    }
    free(tab);
  }

  /* The same sweeps on one rank, before any partition. */
  double rms_serial = 0.0;
  if (g_rank == 0)
  {
    MFTL_Ensemble *solo = make_ens(MPI_COMM_SELF, -1.0, 1.0, MFTL_TRUE);
    add_noisy_sphere(solo, c, 0.6, 4, 0.01);
    MFTLEnsembleRemoveUndulations(solo, &p, MFTL_NULL);
    rms_serial = radial_rms(solo, c, body_volume(solo, 0));
    MFTLEnsembleDestroy(&solo);
  }
  MPI_Bcast(&rms_serial, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  const int ghosts = ghosts_agree(ens);
  MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, 0);
  const int sound = obj ? inv_partition_is_sound(obj, g_rank) : 1;
  if (g_rank == 0)
    printf("\n    np %d: dV/V %.1e, amount %.1e, %lld moves (%lld in %d colour rounds, %lld cross-rank edges), rms %.3e vs %.3e on one rank ", g_nproc,
           fabs(V1 - V0) / V0, fabs(amount1 - amount0) / amount0, st.n_moved, st.n_moved_band, st.n_rounds, st.n_cross_rank, rms, rms_serial);
  ok = ok && fabs(V1 - V0) < 1e-13 * V0 && fabs(amount1 - amount0) < 1e-13 * amount0 && rms < 1.15 * rms_serial;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(dens_agree, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(sound, MPI_COMM_WORLD));
  /* Non-vacuity: the band is real and was smoothed in the colour rounds. */
  TEST_ASSERT(agreed(g_nproc == 1 || (st.n_moved_band > 0 && st.n_rounds > 0 && st.n_cross_rank > 0), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc > 1 || (st.n_moved_band == 0 && st.n_rounds == 0), MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 5. Volume correction
 * ═════════════════════════════════════════════════════════════════════════════ */

static void scale_about(MFTL_Ensemble *ens, const double c[3], const double s[3])
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double d[3];
      MFTLGetMinimumImageVector(&ens->domain, d, c, &obj->verts.pos[3 * v]);
      for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] = c[k] + s[k] * d[k];
      MFTLApplyPeriodicity(&ens->domain, &obj->verts.pos[3 * v]);
    }
    MFTLObjectMarkChanged(obj);
  }
}

static int test_a_shrunken_sphere_is_restored_in_three_newton_steps(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, -1.0, 1.0, MFTL_TRUE);
  const double c[3] = {0.03, -0.02, 0.01};
  add_noisy_sphere(ens, c, 0.6, 4, 0.0);
  MFTLRedistribute(ens);
  ok = MFTLEnsembleSetReferenceVolumes(ens) == MFTL_SUCCESS;
  const double V0 = body_volume(ens, 0);
  MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, 0);
  ok = ok && (obj == MFTL_NULL || obj->reference_volume == V0);
  const double rms0 = radial_rms(ens, c, V0);

  const double s[3] = {0.97, 0.97, 0.97};
  double err[4];
  for (int n = 0; n <= 3; ++n)
  {
    scale_about(ens, c, s);
    MFTL_VolumeCorrectionParameters p = MFTLVolumeCorrectionDefaultParameters();
    p.iterations = n;
    MFTL_VolumeCorrectionStats st;
    ok = ok && MFTLEnsembleCorrectVolumes(ens, &p, &st) == MFTL_SUCCESS && st.n_bodies == 1;
    err[n] = fabs(body_volume(ens, 0) / V0 - 1.0);
    ok = ok && fabs(st.max_error_after - err[n]) <= 1e-12 + 1e-6 * err[n];
    /* Back to the reference for the next count, the same way. */
    MFTLEnsembleCorrectVolumes(ens, MFTL_NULL, MFTL_NULL);
  }
  const double rms1 = radial_rms(ens, c, body_volume(ens, 0));
  if (g_rank == 0) printf("\n    from -8.7 %%: |V/V0 - 1| after 0/1/2/3 steps %.1e %.1e %.1e %.1e; shape rms %.3e -> %.3e ", err[0], err[1], err[2], err[3], rms0, rms1);
  /* Quadratic: e_{n+1} ~ e_n^2 / 3 on a sphere, whose V(d) is a cubic in d. */
  ok = ok && err[0] > 0.08 && err[1] < 4e-3 && err[2] < 5e-6 && err[3] < 1e-11 && fabs(rms1 - rms0) < 0.05 * rms0;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ghosts_agree(ens), MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_a_sunk_cap_on_the_floor_is_restored_with_its_rim_on_the_wall(void)
{
  int ok = 1;
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, (double[3]){-1.0, -1.0, 0.0}, (double[3]){1.0, 1.0, 1.0},
                                          (bool[3]){MFTL_FALSE, MFTL_FALSE, MFTL_FALSE}, 4);
  add_cap(ens, 0.02, -0.03, 0.7, 0.4, 14);
  MFTLPublishGeneratedBodies(ens);
  MFTLRedistribute(ens);
  MFTLEnsembleSetReferenceVolumes(ens);
  const double V0 = body_volume(ens, 0);
  const double c[3] = {0.0, 0.0, 0.0}, s[3] = {1.0, 1.0, 0.9};
  scale_about(ens, c, s);  // heights only: the rim stays on the floor
  MFTL_VolumeCorrectionStats st;
  ok = MFTLEnsembleCorrectVolumes(ens, MFTL_NULL, &st) == MFTL_SUCCESS && st.n_bodies == 1;
  const double V1 = body_volume(ens, 0);
  int rim_on_wall = 1, n_rim = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
      if (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY)
      {
        n_rim++;
        if (obj->verts.pos[3 * v + 2] != 0.0) rim_on_wall = 0;
      }
  }
  if (g_rank == 0) printf("\n    cap: %.1e before, %.1e after ", st.max_error_before, fabs(V1 / V0 - 1.0));
  ok = ok && st.max_error_before > 0.09 && fabs(V1 / V0 - 1.0) < 1e-12;
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(rim_on_wall, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n_rim > 0 || g_nproc > 1, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 6. The reference volume
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Translated half a period, a body lands on ranks that never held it; they learn its reference from the wire. */
static int test_the_reference_volume_reaches_new_holders(void)
{
  int ok = 1;
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, -1.0, 1.0, MFTL_TRUE);
  const double c[3] = {-0.5, -0.5, -0.5};
  add_noisy_sphere(ens, c, 0.2, 3, 0.0);
  MFTLRedistribute(ens);
  MFTLEnsembleSetReferenceVolumes(ens);
  const double ref = body_volume(ens, 0);
  const int held_before = MFTLEnsembleFindLocalObject(ens, 0) != MFTL_NULL;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n; ++v)
    {
      for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] += 1.0;
      MFTLApplyPeriodicity(&ens->domain, &obj->verts.pos[3 * v]);
    }
    MFTLObjectMarkChanged(obj);
  }
  MFTLEnsembleComputeTriangleGeometry(ens);
  MFTLRedistribute(ens);
  MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, 0);
  const int newly = obj != MFTL_NULL && !held_before;
  ok = obj == MFTL_NULL || obj->reference_volume == ref;
  int n_new = newly;
  MPI_Allreduce(MPI_IN_PLACE, &n_new, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (g_rank == 0) printf("\n    %d rank(s) newly holding the body ", n_new);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(g_nproc == 1 || n_new > 0, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Two spheres that coalesce hand the SUM of their references to the body they become. */
static int test_a_coalescence_hands_the_sum_of_the_references_on(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, 0.0, 4.0, MFTL_FALSE);
  const double a[3] = {2.0 - 0.53, 2.0, 2.0}, b[3] = {2.0 + 0.53, 2.0, 2.0};
  add_noisy_sphere(ens, a, 0.5, 4, 0.0);
  add_noisy_sphere(ens, b, 0.5, 4, 0.0);
  MFTLRedistribute(ens);
  MFTLEnsembleSetReferenceVolumes(ens);
  /* Make the references differ from the volumes, so the sum cannot be a re-measurement. */
  for (int o = 0; o < ens->n_local_objects; ++o) ens->local_objects[o]->reference_volume *= (ens->local_objects[o]->global_id == 0) ? 1.01 : 0.98;
  double sum = 0.0;
  for (int gid = 0; gid < 2; ++gid)
  {
    double r = -HUGE_VAL;
    const MFTL_Object *o = MFTLEnsembleFindLocalObject(ens, gid);
    if (o) r = o->reference_volume;
    MPI_Allreduce(MPI_IN_PLACE, &r, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    sum += r;
  }
  MFTL_TopologyOptions opt = MFTLTopologyDefaultOptions();
  opt.dx = 4.0 / 60.0;
  opt.delta_breakup = 0.02;
  opt.delta_coalescence = 0.1;
  MFTL_TopologyStats ts;
  const int rc = MFTLEnsembleApplyTopologyChanges(ens, &opt, &ts);
  int live = 0, ok = rc == MFTL_SUCCESS;
  double got = -HUGE_VAL;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    if (obj->global_n_triangles > 0) live = 1;
    if (got == -HUGE_VAL)
      got = obj->reference_volume;
    else if (memcmp(&got, &obj->reference_volume, sizeof(double)) != 0)
      ok = 0;
  }
  double g = got;
  MPI_Allreduce(MPI_IN_PLACE, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  int n_bodies = 0;
  for (int gid = 0; gid < ens->n_bodies_global; ++gid)
  {
    int h = MFTLEnsembleFindLocalObject(ens, gid) != MFTL_NULL;
    MPI_Allreduce(MPI_IN_PLACE, &h, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    n_bodies += h;
  }
  (void) live;
  if (g_rank == 0) printf("\n    %d coalescence events, %d body: reference %.15g, sum %.15g ", ts.n_coalescence, n_bodies, g, sum);
  ok = ok && (got == -HUGE_VAL || got == g);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ts.n_coalescence > 0 && n_bodies == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(g - sum) <= 1e-14 * sum, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_bad_parameters_are_refused_everywhere(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, -1.0, 1.0, MFTL_TRUE);
  MFTL_UndulationParameters p = MFTLUndulationDefaultParameters();
  p.relaxation = 0.0;
  const int a = MFTLEnsembleRemoveUndulations(ens, &p, MFTL_NULL);
  p = MFTLUndulationDefaultParameters();
  p.min_angle = 90.0;
  const int b = MFTLEnsembleRemoveUndulations(ens, &p, MFTL_NULL);
  MFTL_VolumeCorrectionParameters q = MFTLVolumeCorrectionDefaultParameters();
  q.iterations = -1;
  const int c = MFTLEnsembleCorrectVolumes(ens, &q, MFTL_NULL);
  /* An ensemble with no bodies: nothing to do, and no collective is unmatched. */
  const int d = MFTLEnsembleRemoveUndulations(ens, MFTL_NULL, MFTL_NULL) == MFTL_SUCCESS && MFTLEnsembleCorrectVolumes(ens, MFTL_NULL, MFTL_NULL) == MFTL_SUCCESS;
  TEST_ASSERT(agreed(a == MFTL_ERROR && b == MFTL_ERROR && c == MFTL_ERROR && d, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  RUN_TEST(test_one_edge_conserves_the_volume_and_flattens_the_edge);
  RUN_TEST(test_a_jittered_sphere_goes_back_to_a_sphere_at_constant_volume);
  RUN_TEST(test_a_jittered_sheet_flattens_at_its_mean_height);
  RUN_TEST(test_a_sphere_on_the_periodic_seam_is_smoothed_as_anywhere_else);
  RUN_TEST(test_the_default_spares_a_resolved_surface_and_still_removes_noise);
  RUN_TEST(test_a_density_keeps_its_amount_per_triangle);
  RUN_TEST(test_the_distributed_sweep_is_exact_and_agrees);
  RUN_TEST(test_a_shrunken_sphere_is_restored_in_three_newton_steps);
  RUN_TEST(test_a_sunk_cap_on_the_floor_is_restored_with_its_rim_on_the_wall);
  RUN_TEST(test_the_reference_volume_reaches_new_holders);
  RUN_TEST(test_a_coalescence_hands_the_sum_of_the_references_on);
  RUN_TEST(test_bad_parameters_are_refused_everywhere);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  TEST_REPORT();
}
