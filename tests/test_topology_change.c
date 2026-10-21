/*
 * @Author: Christian Gorges
 * @Date: 2026-09-25
 *
 * @brief Topology changes (src/topology_change/): detection, clustering, the
 *        distributed reconstruction, body separation and ids.
 *
 * ## HOW EACH SCENARIO IS JUDGED
 *
 * Every scenario is judged twice, and the two judgements answer different
 * questions:
 *
 *   PHYSICS. What the result must BE, whatever produced it: how many bodies,
 *   each one's Euler characteristic, its volume against the analytic one, rims
 *   on walls, a sound partition, dense ids per body.
 *
 *   PARTITION INDEPENDENCE. The same scenario is run a second time, inside this
 *   test, on MPI_COMM_SELF on rank 0 -- a genuine np = 1 run with no
 *   decomposition at all -- and the distributed result must reproduce it: the
 *   same bodies with the same ids, the same entity counts, the same vertex
 *   positions to 1e-12. That is the one statement no single-rank-count test can
 *   make, and it covers the whole pipeline: detection through the halo, the
 *   overlap exchange, the ghost-layer extraction, the labelling and the ids.
 *   The lattice spacing divides the domain into 60 cells, which every tested
 *   grid (1, 2, 3 cuts per axis) divides, so both runs use the same lattice.
 *
 * Registered at np = 1, 2, 4, 8 and 27; every fixture is centred where the cut
 * planes meet, so at np > 1 every event is cut by at least one rank boundary.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

#define L_BOX 4.0
#define DX (L_BOX / 60.0)

/* ═════════════════════════════════════════════════════════════════════════════
 * 1. Fields
 * ═════════════════════════════════════════════════════════════════════════════ */

typedef double (*field_fn)(const double p[3], const double *a);

/** a: cx cy cz R */
static double f_sphere(const double p[3], const double *a)
{
  return sqrt((p[0] - a[0]) * (p[0] - a[0]) + (p[1] - a[1]) * (p[1] - a[1]) + (p[2] - a[2]) * (p[2] - a[2])) - a[3];
}

/** a: cx cy cz R -- the INSIDE of the sphere is the outside of the body: a bubble. */
static double f_bubble(const double p[3], const double *a) { return -f_sphere(p, a); }

/** a: cx cy cz R neck_radius half_separation -- along x. */
static double f_dumbbell(const double p[3], const double *a)
{
  const double s1[4] = {a[0] - a[5], a[1], a[2], a[3]}, s2[4] = {a[0] + a[5], a[1], a[2], a[3]};
  const double y = p[1] - a[1], z = p[2] - a[2];
  const double cyl = fmax(sqrt(y * y + z * z) - a[4], fabs(p[0] - a[0]) - a[5]);
  return fmin(fmin(f_sphere(p, s1), f_sphere(p, s2)), cyl);
}

/** a: cx cy cz R_small neck half_separation R_big -- a dumbbell with unequal bulbs. */
static double f_lopsided(const double p[3], const double *a)
{
  const double s1[4] = {a[0] - a[5], a[1], a[2], a[6]}, s2[4] = {a[0] + a[5], a[1], a[2], a[3]};
  const double y = p[1] - a[1], z = p[2] - a[2];
  const double cyl = fmax(sqrt(y * y + z * z) - a[4], fabs(p[0] - a[0]) - a[5]);
  return fmin(fmin(f_sphere(p, s1), f_sphere(p, s2)), cyl);
}

/** a: cx cy cz R r -- about the z axis. */
static double f_torus(const double p[3], const double *a)
{
  const double x = p[0] - a[0], y = p[1] - a[1], z = p[2] - a[2];
  const double q = sqrt(x * x + y * y) - a[3];
  return sqrt(q * q + z * z) - a[4];
}

/** a: level  dcx dcy dcz dR neck -- liquid below z = level, and a drop above tied to it by a neck. */
static double f_sheet_with_drop(const double p[3], const double *a)
{
  const double drop[4] = {a[1], a[2], a[3], a[4]};
  const double x = p[0] - a[1], y = p[1] - a[2];
  const double neck = fmax(sqrt(x * x + y * y) - a[5], fmax(a[0] - p[2], p[2] - a[3]));
  return fmin(fmin(p[2] - a[0], f_sphere(p, drop)), neck);
}

/** a: cx cy cz R thickness -- a flat disk-shaped patch: an OPEN surface with a free rim. */

/* ═════════════════════════════════════════════════════════════════════════════
 * 2. Ensembles and scenes
 * ═════════════════════════════════════════════════════════════════════════════ */

/** Lower corner of the next ensembles' box; the box is always L_BOX wide. */
static double g_origin = 0.0;

static MFTL_Ensemble *make_ens_axes(MPI_Comm comm, const bool per[3])
{
  int np = 1;
  MPI_Comm_size(comm, &np);
  int cuts[3] = {0, 0, 0};
  MPI_Dims_create(np, 3, cuts);
  double lo[3] = {g_origin, g_origin, g_origin}, hi[3] = {g_origin + L_BOX, g_origin + L_BOX, g_origin + L_BOX};
  bool p[3] = {per[0], per[1], per[2]};
  return MFTLEnsembleCreate(comm, cuts, lo, hi, p, 4);
}

static MFTL_Ensemble *make_ens(MPI_Comm comm, bool periodic)
{
  const bool per[3] = {periodic, periodic, periodic};
  return make_ens_axes(comm, per);
}

/** @brief A body from a field, built on the ensemble's rank 0 only -- the generation contract. */
static void add_body(MFTL_Ensemble *ens, field_fn f, const double *a, const double lo[3], const double hi[3], double dx)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (ens->domain.rank != 0)
  {
    MFTLObjectBuildDerived(obj);
    return;
  }
  int dims[3];
  const double sp[3] = {dx, dx, dx};
  for (int k = 0; k < 3; ++k) dims[k] = (int) ceil((hi[k] - lo[k]) / dx) + 1;
  MFTL_SdfGrid g;
  MFTLSdfGridInit(&g, lo, sp, dims);
  for (int k = 0; k < dims[2]; ++k)
    for (int j = 0; j < dims[1]; ++j)
      for (int i = 0; i < dims[0]; ++i)
      {
        const double p[3] = {lo[0] + i * dx, lo[1] + j * dx, lo[2] + k * dx};
        g.phi[((size_t) k * dims[1] + j) * dims[0] + i] = f(p, a);
      }
  MFTLBuildMeshFromLevelSet(obj, &g, 1e-2);
  MFTLSdfGridFree(&g);
}

static void refresh(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

/** @brief Publish, scatter, and bring every cache up to date. */
static void settle(MFTL_Ensemble *ens)
{
  MFTLPublishGeneratedBodies(ens);
  refresh(ens);
  MFTLRedistribute(ens);
  refresh(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
}

typedef void (*scene_fn)(MFTL_Ensemble *ens);

/* ═════════════════════════════════════════════════════════════════════════════
 * 3. Measuring the result
 * ═════════════════════════════════════════════════════════════════════════════ */

#define MAX_BODIES 16

typedef struct
{
  int n_live;  ///< Bodies with at least one triangle anywhere.
  int gid[MAX_BODIES];
  long long V[MAX_BODIES], T[MAX_BODIES], B[MAX_BODIES];
  double volume[MAX_BODIES];
  int chi[MAX_BODIES];
  int rim_off_wall;  ///< Boundary half-edges not on a wall, summed.
  int flag_mismatch;  ///< Rim vertices without MFTL_FLAG_BOUNDARY, or vice versa, summed.
  int partition_ok;
  int ids_dense;
  double *pos;  ///< Rank 0: sorted (gid, x, y, z) of every owned vertex.
  int n_pos;
} census;

static bool on_wall(const MFTL_Domain *d, const double *x)
{
  for (int k = 0; k < 3; ++k)
  {
    if (d->is_periodic[k]) continue;
    if (fabs(x[k] - d->global_min[k]) < 1e-9 || fabs(x[k] - d->global_max[k]) < 1e-9) return MFTL_TRUE;
  }
  return MFTL_FALSE;
}

/**
 * @brief Order rows by body, then by coordinates QUANTISED to 1e-8.
 *
 * Not by the raw coordinates: marching-cubes vertices on one lattice plane
 * share that coordinate exactly, so the raw order is decided by the next one --
 * and the volume correction's offset comes from a reduction whose summation
 * order depends on the process count, so two runs agree to 1e-16 and not to
 * the bit. A tie broken by that last bit sorts the rows differently and the
 * element-wise comparison then pairs unrelated vertices. Quantised keys make
 * the order a property of the geometry; the values are still compared to
 * 1e-12 afterwards.
 */
static int cmp_row4(const void *a, const void *b)
{
  const double *x = (const double *) a, *y = (const double *) b;
  if (x[0] != y[0]) return (x[0] < y[0]) ? -1 : 1;
  for (int k = 1; k < 4; ++k)
  {
    const long long qx = llround(x[k] * 1e8), qy = llround(y[k] * 1e8);
    if (qx != qy) return (qx < qy) ? -1 : 1;
  }
  return 0;
}

static int cmp_int(const void *a, const void *b)
{
  const int x = *(const int *) a, y = *(const int *) b;
  return (x > y) - (x < y);
}

/**
 * @brief Everything checked about a result, reduced over the ensemble's
 *        communicator so every rank holds the same verdicts.
 */
static void take_census(MFTL_Ensemble *ens, census *c)
{
  memset(c, 0, sizeof(*c));
  const MFTL_Domain *d = &ens->domain;
  MPI_Comm comm = d->comm_world;
  int np = 1, rank = 0;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &rank);

  /* Live bodies: every rank's local ids with owned triangles, gathered. */
  int mine[MAX_BODIES], n_mine = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (ens->local_objects[o]->tris.n_local > 0 && n_mine < MAX_BODIES) mine[n_mine++] = ens->local_objects[o]->global_id;
  int *counts = (int *) malloc((size_t) np * sizeof(int)), *displs = (int *) malloc((size_t) np * sizeof(int));
  MPI_Allgather(&n_mine, 1, MPI_INT, counts, 1, MPI_INT, comm);
  int total = 0;
  for (int r = 0; r < np; ++r)
  {
    displs[r] = total;
    total += counts[r];
  }
  int *all = (int *) malloc((size_t) (total > 0 ? total : 1) * sizeof(int));
  MPI_Allgatherv(mine, n_mine, MPI_INT, all, counts, displs, MPI_INT, comm);
  qsort(all, (size_t) total, sizeof(int), cmp_int);
  for (int i = 0; i < total; ++i)
    if ((c->n_live == 0 || c->gid[c->n_live - 1] != all[i]) && c->n_live < MAX_BODIES) c->gid[c->n_live++] = all[i];
  free(all);

  /* Per body: an ANCHOR every rank agrees on -- the vertex with global id 0,
   * which exactly one rank owns -- so that every triangle can be unwrapped into
   * one frame around the body. A fixed point like the domain centre splits a
   * body that straddles the periodic corner across two images. */
  double anchor[MAX_BODIES][3];
  memset(anchor, 0, sizeof anchor);
  for (int b = 0; b < c->n_live; ++b)
  {
    const MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, c->gid[b]);
    if (obj == MFTL_NULL) continue;
    for (int v = 0; v < obj->verts.n_local; ++v)
      if (obj->verts.rec[v].global_id == 0)
        for (int k = 0; k < 3; ++k) anchor[b][k] = obj->verts.pos[3 * v + k];
  }
  MPI_Allreduce(MPI_IN_PLACE, anchor, 3 * MAX_BODIES, MPI_DOUBLE, MPI_SUM, comm);

  /* Per body: counts, rim, volume about its anchor with minimum images. */
  double sums[MAX_BODIES][4];
  memset(sums, 0, sizeof sums);
  int bad[3] = {0, 0, 1};
  for (int b = 0; b < c->n_live; ++b)
  {
    const MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, c->gid[b]);
    if (obj == MFTL_NULL) continue;
    sums[b][0] = obj->verts.n_local;
    sums[b][1] = obj->tris.n_local;
    const double *const ref = anchor[b];
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double T[3][3];
      MFTLGetTriangleCorners(obj, t, T);
      double s[3];
      MFTLGetMinimumImageVector(d, s, ref, T[0]);
      for (int k = 0; k < 3; ++k) s[k] = ref[k] + s[k] - T[0][k];
      double q[3][3];
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 3; ++k) q[j][k] = T[j][k] + s[k] - ref[k];
      sums[b][3] += (q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1]) - q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0]) +
                     q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0])) /
                    6.0;
      for (int k = 0; k < 3; ++k)
      {
        const int h = 3 * t + k;
        if (obj->tris.he[h].twin_idx != MFTL_ERROR) continue;
        sums[b][2] += 1.0;
        const double *x0 = &obj->verts.pos[3 * obj->tris.he[h].vertex_idx];
        const double *x1 = &obj->verts.pos[3 * obj->tris.he[MFTLHePrevIdx(h)].vertex_idx];
        if (!on_wall(d, x0) || !on_wall(d, x1)) bad[0]++;
        if (!(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY)) bad[1]++;
      }
    }
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const bool flagged = (obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY) != 0;
      if (flagged != on_wall(d, &obj->verts.pos[3 * v])) bad[1]++;
    }
  }
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (!inv_partition_is_sound(ens->local_objects[o], rank)) bad[2] = 0;

  MPI_Allreduce(MPI_IN_PLACE, sums, 4 * MAX_BODIES, MPI_DOUBLE, MPI_SUM, comm);
  int red[3];
  MPI_Allreduce(bad, red, 2, MPI_INT, MPI_SUM, comm);
  MPI_Allreduce(&bad[2], &red[2], 1, MPI_INT, MPI_MIN, comm);
  c->rim_off_wall = red[0];
  c->flag_mismatch = red[1];
  c->partition_ok = red[2];
  for (int b = 0; b < c->n_live; ++b)
  {
    c->V[b] = (long long) sums[b][0];
    c->T[b] = (long long) sums[b][1];
    c->B[b] = (long long) sums[b][2];
    c->volume[b] = sums[b][3];
    c->chi[b] = (int) (c->V[b] - (3 * c->T[b] + c->B[b]) / 2 + c->T[b]);
  }

  /* Dense ids per body, and the positions, gathered on rank 0. */
  int n_rows = 0;
  for (int b = 0; b < c->n_live; ++b)
  {
    const MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, c->gid[b]);
    if (obj) n_rows += obj->verts.n_local;
  }
  double *rows = (double *) malloc((size_t) (n_rows > 0 ? n_rows : 1) * 6 * sizeof(double));
  int w = 0;
  for (int b = 0; b < c->n_live; ++b)
  {
    const MFTL_Object *obj = MFTLEnsembleFindLocalObject(ens, c->gid[b]);
    if (!obj) continue;
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      double *r = &rows[6 * w++];
      r[0] = c->gid[b];
      r[1] = obj->verts.pos[3 * v];
      r[2] = obj->verts.pos[3 * v + 1];
      r[3] = obj->verts.pos[3 * v + 2];
      r[4] = obj->verts.rec[v].global_id;
      r[5] = 0.0;
    }
  }
  const int n_doubles = 6 * n_rows;
  MPI_Gather(&n_doubles, 1, MPI_INT, counts, 1, MPI_INT, 0, comm);
  total = 0;
  if (rank == 0)
    for (int r = 0; r < np; ++r)
    {
      displs[r] = total;
      total += counts[r];
    }
  double *gathered = (double *) malloc((size_t) (total > 0 ? total : 1) * sizeof(double));
  MPI_Gatherv(rows, n_doubles, MPI_DOUBLE, gathered, counts, displs, MPI_DOUBLE, 0, comm);
  free(rows);

  int dense = 1;
  if (rank == 0)
  {
    const int n = total / 6;
    /* Ids: per body, the sorted owned vertex ids are exactly 0..V-1. */
    for (int b = 0; b < c->n_live && dense; ++b)
    {
      int *ids = (int *) malloc((size_t) (c->V[b] > 0 ? c->V[b] : 1) * sizeof(int));
      int m = 0;
      for (int i = 0; i < n; ++i)
        if ((int) gathered[6 * i] == c->gid[b]) ids[m++] = (int) gathered[6 * i + 4];
      qsort(ids, (size_t) m, sizeof(int), cmp_int);
      for (int i = 0; i < m; ++i)
        if (ids[i] != i) dense = 0;
      if (m != c->V[b]) dense = 0;
      free(ids);
    }
    c->pos = (double *) malloc((size_t) (n > 0 ? n : 1) * 4 * sizeof(double));
    for (int i = 0; i < n; ++i)
      for (int k = 0; k < 4; ++k) c->pos[4 * i + k] = gathered[6 * i + k];
    qsort(c->pos, (size_t) n, 4 * sizeof(double), cmp_row4);
    c->n_pos = n;
  }
  MPI_Bcast(&dense, 1, MPI_INT, 0, comm);
  c->ids_dense = dense;
  free(gathered);
  free(counts);
  free(displs);
}

static void free_census(census *c) { free(c->pos); }

/** @brief Rank 0: are two censuses the same result? Positions to 1e-12. */
static bool same_result(const census *a, const census *b)
{
  if (a->n_live != b->n_live) return MFTL_FALSE;
  for (int i = 0; i < a->n_live; ++i)
    if (a->gid[i] != b->gid[i] || a->V[i] != b->V[i] || a->T[i] != b->T[i] || a->B[i] != b->B[i]) return MFTL_FALSE;
  if (a->n_pos != b->n_pos) return MFTL_FALSE;
  double worst = 0.0;
  for (int i = 0; i < a->n_pos; ++i)
    for (int k = 0; k < 4; ++k) worst = fmax(worst, fabs(a->pos[4 * i + k] - b->pos[4 * i + k]));
  if (worst > 1e-12) printf("\n  largest position difference against the np = 1 run: %.3e\n  ", worst);
  return worst <= 1e-12;
}

/**
 * @brief Runs a scene distributed and as a genuine np = 1 reference, and
 *        checks the two agree. The distributed census is returned.
 */
/** Periodicity of the next run_scene(); reset to "all or nothing" after each. */
static bool g_axes[3];
static bool g_axes_set = MFTL_FALSE;

static MFTL_Ensemble *scene_ens(MPI_Comm comm, bool periodic)
{
  if (g_axes_set) return make_ens_axes(comm, g_axes);
  return make_ens(comm, periodic);
}

static int run_scene(scene_fn scene, bool periodic, const MFTL_TopologyOptions *opts, bool reconstruct_all, census *out, MFTL_TopologyStats *st_out)
{
  MFTL_TopologyStats st;

  MFTL_Ensemble *ens = scene_ens(MPI_COMM_WORLD, periodic);
  scene(ens);
  settle(ens);
  int rc;
  if (reconstruct_all)
  {
    int bodies[MAX_BODIES];
    for (int i = 0; i < ens->n_bodies_global; ++i) bodies[i] = i;
    rc = MFTLEnsembleReconstructBodies(ens, bodies, ens->n_bodies_global, MFTL_NULL, 0, opts, &st);
  }
  else
    rc = MFTLEnsembleApplyTopologyChanges(ens, opts, &st);
  take_census(ens, out);
  MFTLEnsembleDestroy(&ens);
  if (st_out) *st_out = st;

  /* The reference, on rank 0 alone. */
  int same = 1;
  if (g_rank == 0 && g_nproc > 1)
  {
    MFTL_Ensemble *ref = scene_ens(MPI_COMM_SELF, periodic);
    scene(ref);
    settle(ref);
    MFTL_TopologyStats rst;
    if (reconstruct_all)
    {
      int bodies[MAX_BODIES];
      for (int i = 0; i < ref->n_bodies_global; ++i) bodies[i] = i;
      MFTLEnsembleReconstructBodies(ref, bodies, ref->n_bodies_global, MFTL_NULL, 0, opts, &rst);
    }
    else
      MFTLEnsembleApplyTopologyChanges(ref, opts, &rst);
    census rc_ref;
    take_census(ref, &rc_ref);
    same = same_result(out, &rc_ref) && rst.n_events == st.n_events;
    if (!same)
      for (int b = 0; b < MFTL_MAX(out->n_live, rc_ref.n_live); ++b)
        printf("\n  body slot %d: distributed gid %d V %lld T %lld vol %.9f | reference gid %d V %lld T %lld vol %.9f", b, out->gid[b], out->V[b], out->T[b],
               out->volume[b], rc_ref.gid[b], rc_ref.V[b], rc_ref.T[b], rc_ref.volume[b]);
    if (!same) printf("\n  reference: %d bodies, %d events; distributed: %d bodies, %d events\n  ", rc_ref.n_live, rst.n_events, out->n_live, st.n_events);
    free_census(&rc_ref);
    MFTLEnsembleDestroy(&ref);
  }
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);

  TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(out->partition_ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(out->ids_dense, MPI_COMM_WORLD));
  return 0;
}

static MFTL_TopologyOptions base_options(double delta_b, double delta_c)
{
  MFTL_TopologyOptions o = MFTLTopologyDefaultOptions();
  o.delta_breakup = delta_b;
  o.delta_coalescence = delta_c;
  o.dx = DX;
  return o;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 4. Scenes -- every one centred where the cut planes of every tested grid meet
 * ═════════════════════════════════════════════════════════════════════════════ */

static const double C = 0.5 * L_BOX;

static void scene_sphere(MFTL_Ensemble *ens)
{
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {C + 0.013, C - 0.021, C + 0.007};
  MFTLGenerateSphere(obj, c, 0.9, 4);
}

static void scene_dumbbell(MFTL_Ensemble *ens)
{
  /* Bulbs of radius 0.45, a neck of radius 0.05 -- a diameter of 0.1 against a
   * breakup threshold of 0.12 -- centred ON the x cut plane. */
  const double a[6] = {C, C + 0.01, C - 0.01, 0.45, 0.05, 0.6};
  const double lo[3] = {C - 1.2, C - 0.6, C - 0.6}, hi[3] = {C + 1.2, C + 0.6, C + 0.6};
  add_body(ens, f_dumbbell, a, lo, hi, 0.04);
}

static void scene_two_spheres(MFTL_Ensemble *ens)
{
  /* A gap of 0.06 against a coalescence threshold of 0.1, on the x cut plane. */
  const double a[4] = {C - 0.53, C, C, 0.5}, b[4] = {C + 0.53, C, C, 0.5};
  const double lo_a[3] = {C - 1.1, C - 0.6, C - 0.6}, hi_a[3] = {C + 0.05, C + 0.6, C + 0.6};
  const double lo_b[3] = {C - 0.05, C - 0.6, C - 0.6}, hi_b[3] = {C + 1.1, C + 0.6, C + 0.6};
  add_body(ens, f_sphere, a, lo_a, hi_a, 0.05);
  add_body(ens, f_sphere, b, lo_b, hi_b, 0.05);
}

static void scene_two_nearly_touching_spheres(MFTL_Ensemble *ens)
{
  /* A gap of 0.005 -- a twentieth of the coalescence threshold, far below the
   * lattice -- on the x cut plane: two drops the detection meets only once they
   * have almost touched, as after a fast approach between two calls. */
  const double a[4] = {C - 0.5025, C, C, 0.5}, b[4] = {C + 0.5025, C, C, 0.5};
  const double lo_a[3] = {C - 1.1, C - 0.6, C - 0.6}, hi_a[3] = {C + 0.05, C + 0.6, C + 0.6};
  const double lo_b[3] = {C - 0.05, C - 0.6, C - 0.6}, hi_b[3] = {C + 1.1, C + 0.6, C + 0.6};
  add_body(ens, f_sphere, a, lo_a, hi_a, 0.05);
  add_body(ens, f_sphere, b, lo_b, hi_b, 0.05);
}

static void scene_three_in_a_row(MFTL_Ensemble *ens)
{
  const double r = 0.35;
  for (int i = -1; i <= 1; ++i)
  {
    const double a[4] = {C + i * (2 * r + 0.05), C, C, r};
    const double lo[3] = {a[0] - r - 0.1, C - r - 0.1, C - r - 0.1}, hi[3] = {a[0] + r + 0.1, C + r + 0.1, C + r + 0.1};
    add_body(ens, f_sphere, a, lo, hi, 0.05);
  }
}

static void scene_torus(MFTL_Ensemble *ens)
{
  /* Hole diameter 2 * (0.5 - 0.44) = 0.12 against a coalescence threshold of 0.2. */
  const double a[5] = {C, C, C, 0.5, 0.44};
  const double lo[3] = {C - 1.05, C - 1.05, C - 0.55}, hi[3] = {C + 1.05, C + 1.05, C + 0.55};
  add_body(ens, f_torus, a, lo, hi, 0.04);
}

static void scene_drop_with_bubble(MFTL_Ensemble *ens)
{
  /* A bubble of radius 0.4 inside a drop of radius 1, leaving a liquid film of
   * 0.06 at the top: a BREAKUP event between two different bodies. */
  const double drop[4] = {C, C, C, 1.0}, bub[4] = {C, C, C + 0.54, 0.4};
  const double lo[3] = {C - 1.1, C - 1.1, C - 1.1}, hi[3] = {C + 1.1, C + 1.1, C + 1.1};
  const double blo[3] = {C - 0.5, C - 0.5, C + 0.04}, bhi[3] = {C + 0.5, C + 0.5, C + 1.04};
  add_body(ens, f_sphere, drop, lo, hi, 0.05);
  add_body(ens, f_bubble, bub, blo, bhi, 0.04);
}

static void scene_sheet_with_drop(MFTL_Ensemble *ens)
{
  /* Liquid below z = 1.5 across the whole box -- its rim on the four side
   * walls -- and a drop above, tied to it by a neck of radius 0.05. */
  const double a[6] = {1.5, C + 0.01, C - 0.01, 2.4, 0.4, 0.05};
  const double lo[3] = {0.0, 0.0, 1.0}, hi[3] = {L_BOX, L_BOX, 3.0};
  add_body(ens, f_sheet_with_drop, a, lo, hi, 0.05);
}

static void scene_lopsided(MFTL_Ensemble *ens)
{
  /* A big bulb and a satellite a few lattice cells across, tied by a long thin
   * neck: the satellite survives the bias but is far below the big one. */
  const double a[7] = {C, C, C, 0.16, 0.03, 0.6, 0.45};
  const double lo[3] = {C - 1.15, C - 0.55, C - 0.55}, hi[3] = {C + 0.85, C + 0.55, C + 0.55};
  add_body(ens, f_lopsided, a, lo, hi, 0.03);
}

static void scene_seam_dumbbell(MFTL_Ensemble *ens)
{
  /* The dumbbell centred on the domain CORNER of a periodic box: the neck is on
   * the seam of all three axes. */
  const double a[6] = {0.0, 0.01, -0.01, 0.45, 0.05, 0.6};
  const double lo[3] = {-1.2, -0.6, -0.6}, hi[3] = {1.2, 0.6, 0.6};
  add_body(ens, f_dumbbell, a, lo, hi, 0.04);
}

static void scene_breakup_and_merge(MFTL_Ensemble *ens)
{
  /* A dumbbell low in the box and two close spheres high in it: one call must
   * split one body and merge two others. */
  const double a[6] = {C, C, 1.0, 0.4, 0.05, 0.55};
  const double lo[3] = {C - 1.1, C - 0.5, 0.5}, hi[3] = {C + 1.1, C + 0.5, 1.5};
  add_body(ens, f_dumbbell, a, lo, hi, 0.04);
  const double s1[4] = {C - 0.43, C, 3.0, 0.4}, s2[4] = {C + 0.43, C, 3.0, 0.4};
  const double lo1[3] = {C - 0.9, C - 0.5, 2.5}, hi1[3] = {C + 0.05, C + 0.5, 3.5};
  const double lo2[3] = {C - 0.05, C - 0.5, 2.5}, hi2[3] = {C + 0.9, C + 0.5, 3.5};
  add_body(ens, f_sphere, s1, lo1, hi1, 0.04);
  add_body(ens, f_sphere, s2, lo2, hi2, 0.04);
}

static void scene_straddling_sphere(MFTL_Ensemble *ens)
{
  /* 2.3 wide in a periodic box of 4, centred at a quarter of it: on every axis
   * the sphere straddles BOTH the seam and the middle of the box. */
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.25 * L_BOX + 0.013, 0.25 * L_BOX - 0.021, 0.25 * L_BOX + 0.007};
  MFTLGenerateSphere(obj, c, 1.15, 4);
}

static void scene_wide_sphere(MFTL_Ensemble *ens)
{
  /* 3.2 wide in a periodic box of 4: compact, but past three quarters of it. */
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {C + 0.013, C - 0.021, C + 0.007};
  MFTLGenerateSphere(obj, c, 1.6, 4);
}

/** a: level amplitude -- liquid below a wavy surface; the domain decides what is periodic. */
static double f_wavy_sheet(const double p[3], const double *a)
{
  return p[2] - (a[0] + a[1] * sin(2.0 * MFTL_PI * p[0] / L_BOX) * cos(0.5 * MFTL_PI * p[1] / L_BOX));
}

/** @brief A wavy sheet across the whole box: rims on the y walls, periodic in x. */
static void scene_annulus(MFTL_Ensemble *ens)
{
  /* Built directly on the periodic lattice of the domain: nx columns wrap. */
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (ens->domain.rank != 0)
  {
    MFTLObjectBuildDerived(obj);
    return;
  }
  const int nx = 48, ny = 49, nz = 49;
  const double dx = L_BOX / nx, dy = L_BOX / (ny - 1), dz = L_BOX / (nz - 1);
  const double origin[3] = {g_origin, g_origin, g_origin}, sp[3] = {dx, dy, dz};
  const int n_global[3] = {nx, ny, nz}, lo[3] = {0, 0, 0}, dims[3] = {nx, ny, nz};
  const bool per[3] = {true, false, false};
  MFTL_SdfGrid g;
  MFTLSdfGridInitWindow(&g, origin, sp, n_global, per, lo, dims);
  const double a[2] = {C + 0.11, 0.3};  // f_wavy_sheet is evaluated in box-relative coordinates
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i)
      {
        const double p[3] = {i * dx, j * dy, k * dz};  // relative to the origin: the field is the same shape wherever the box is
        g.phi[((size_t) k * ny + j) * nx + i] = f_wavy_sheet(p, a);
      }
  MFTLBuildMeshFromLevelSet(obj, &g, 1e-2);
  MFTLSdfGridFree(&g);
}

/* ═════════════════════════════════════════════════════════════════════════════
 * 5. Tests
 * ═════════════════════════════════════════════════════════════════════════════ */

static int test_the_lattice_divides_every_axis_per_rank(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
  double sp[3];
  int cells[3];
  MFTLTopologyLattice(&ens->domain, DX, sp, cells);
  int ok = 1;
  for (int k = 0; k < 3; ++k)
  {
    if (cells[k] % ens->domain.cuts[k] != 0) ok = 0;
    if (fabs(sp[k] * cells[k] - L_BOX) > 1e-12) ok = 0;
    if (cells[k] != 60) ok = 0;  // 60 is divisible by every cut count this file runs at
  }
  /* A target far larger than a rank box still gives one cell per rank. */
  MFTLTopologyLattice(&ens->domain, 100.0, sp, cells);
  for (int k = 0; k < 3; ++k)
    if (cells[k] != ens->domain.cuts[k]) ok = 0;
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  return 0;
}

static int test_reconstructing_a_sphere_without_events_is_faithful(void)
{
  /* No events and no thresholds: the pipeline alone. A closed sphere comes back
   * as one closed sphere with the same id, its volume within the lattice's
   * second-order error, and identical to the np = 1 answer. */
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_sphere, MFTL_FALSE, &o, MFTL_TRUE, &c, &st));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.chi[0] == 2 && c.B[0] == 0, MPI_COMM_WORLD));
  const double exact = 4.0 / 3.0 * MFTL_PI * 0.9 * 0.9 * 0.9;
  TEST_ASSERT(agreed(fabs(c.volume[0] - exact) / exact < 0.01, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_bodies_created == 0 && st.n_bodies_retired == 0, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/**
 * @brief Rank 0: how many census vertices lie ON the icosphere of scene_sphere()
 *        to round-off, and how many of them within / beyond a radius of @p e.
 *
 * Brute force against a freshly generated copy of the original mesh -- shares
 * nothing with the rasteriser or the extractor.
 */
static void count_on_the_original(const census *c, const double e[3], double r_in, double r_out, int *on_in, int *n_in, int *on_out, int *n_out)
{
  *on_in = *n_in = *on_out = *n_out = 0;
  int one[3] = {1, 1, 1};
  double lo[3] = {g_origin, g_origin, g_origin}, hi[3] = {g_origin + L_BOX, g_origin + L_BOX, g_origin + L_BOX};
  MFTL_Ensemble *e0 = MFTLEnsembleCreate(MPI_COMM_SELF, one, lo, hi, MFTL_NULL, 2);
  MFTL_Object *ref = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(e0, ref);
  double ctr[3] = {C + 0.013, C - 0.021, C + 0.007};
  MFTLGenerateSphere(ref, ctr, 0.9, 4);
  for (int i = 0; i < c->n_pos; ++i)
  {
    const double *x = &c->pos[4 * i + 1];
    const double de = sqrt((x[0] - e[0]) * (x[0] - e[0]) + (x[1] - e[1]) * (x[1] - e[1]) + (x[2] - e[2]) * (x[2] - e[2]));
    if (de >= r_in && de <= r_out) continue;
    double best = MFTL_DBL_MAX;
    for (int t = 0; t < ref->tris.n; ++t)
    {
      double T[3][3], q[3];
      MFTLGetTriangleCorners(ref, t, T);
      best = fmin(best, MFTLClosestPointOnTriangle(x, T, q, MFTL_NULL));
    }
    const int on = sqrt(best) <= 1e-10;
    if (de < r_in)
    {
      (*n_in)++;
      *on_in += on;
    }
    else
    {
      (*n_out)++;
      *on_out += on;
    }
  }
  MFTLEnsembleDestroy(&e0);
}

/**
 * @brief A re-gridded body keeps its vertices ON the surface that went in: each
 *        marching-cubes vertex is put on its lattice edge's exact crossing of
 *        the old mesh, identically at every process count.
 *
 * Linear interpolation of the exact distance puts a vertex O(dx^2 kappa) off a
 * curved surface -- and inwards on a convex one. Without the volume correction,
 * so the raw placement is what is judged: measured, 96.7 % of the vertices on
 * the original polyhedron to 1e-10, against 13.9 % interpolated; the rest are
 * the ones the snap keeps off a lattice node.
 */
static int test_a_regridded_body_keeps_its_vertices_on_the_old_surface(void)
{
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  o.conserve_volume = MFTL_FALSE;
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_sphere, MFTL_FALSE, &o, MFTL_TRUE, &c, &st));
  int ok = 1;
  if (g_rank == 0)
  {
    const double far[3] = {1e9, 1e9, 1e9};
    int on_in, n_in, on_out, n_out;
    count_on_the_original(&c, far, 0.0, 0.0, &on_in, &n_in, &on_out, &n_out);
    ok = n_out > 1000 && on_out > 0.9 * n_out;
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/** @brief One breakup event on the surface of scene_sphere(), on its +x pole. */
static MFTL_TopologyEvent g_pole_event;

static int test_a_biased_edge_follows_the_bias_not_the_old_surface(void)
{
  /* The bias moves the level set inwards around the event: there the exact
   * crossing of the OLD surface is the wrong place, and every edge with a
   * biased end must interpolate the biased field instead. Judged on both sides:
   * no vertex well inside the bump lies on the original polyhedron, and away
   * from it the vertices still do. Dropping the invalidation puts the tail of
   * the bump back onto the old surface. */
  MFTL_TopologyOptions o = base_options(0.12, 0.0);
  o.conserve_volume = MFTL_FALSE;
  o.bias_radius_factor = 3.0;  // a wide tail, where the bias is small and the old surface close
  const double ctr[3] = {C + 0.013, C - 0.021, C + 0.007};
  g_pole_event.body_a = g_pole_event.body_b = 0;
  g_pole_event.kind = MFTL_TOPOLOGY_BREAKUP;
  g_pole_event.gap = 0.1;
  g_pole_event.point[0] = ctr[0] + 0.9;
  g_pole_event.point[1] = ctr[1];
  g_pole_event.point[2] = ctr[2];

  int ok = 1, rc = MFTL_SUCCESS, same = 1;
  census c;
  MFTL_TopologyStats st;
  for (int pass = 0; pass < 2; ++pass)  // distributed, then the np = 1 reference on rank 0
  {
    if (pass == 1 && (g_rank != 0 || g_nproc == 1)) break;
    MFTL_Ensemble *ens = make_ens(pass ? MPI_COMM_SELF : MPI_COMM_WORLD, MFTL_FALSE);
    scene_sphere(ens);
    settle(ens);
    int body = 0;
    const int r = MFTLEnsembleReconstructBodies(ens, &body, 1, &g_pole_event, 1, &o, &st);
    census here;
    take_census(ens, &here);
    MFTLEnsembleDestroy(&ens);
    if (pass == 0)
    {
      rc = r;
      c = here;
    }
    else
    {
      same = same_result(&c, &here);
      free_census(&here);
    }
  }
  if (g_rank == 0)
  {
    const double dx_max = DX;
    const double rho = o.bias_radius_factor * fmax(o.delta_breakup, dx_max);
    int on_in, n_in, on_out, n_out;
    /* An axis edge holding a vertex within rho - dx of the event has both ends
     * strictly inside the bump, i.e. biased; beyond rho + D nothing is. */
    count_on_the_original(&c, g_pole_event.point, rho - dx_max, rho + 1.8 * dx_max, &on_in, &n_in, &on_out, &n_out);
    printf("\n    inside the bump: %d of %d on the old surface; outside: %d of %d ", on_in, n_in, on_out, n_out);
    ok = n_in > 10 && on_in == 0 && on_out > 0.9 * n_out;
  }
  MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS && same && ok, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_thin_neck_breaks_into_two_bodies(void)
{
  MFTL_TopologyOptions o = base_options(0.12, 0.02);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_dumbbell, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_breakup > 0 && st.n_coalescence == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 2, MPI_COMM_WORLD));
  /* The larger piece keeps id 0; the other is new, and they are the two bulbs. */
  TEST_ASSERT(agreed(c.gid[0] == 0 && c.gid[1] == 1 && st.n_bodies_created == 1, MPI_COMM_WORLD));
  /* Each bulb loses the neck stub and what the bias erodes around it: 3.0 %
   * measured at this lattice. 5 % is the bound. */
  const double bulb = 4.0 / 3.0 * MFTL_PI * 0.45 * 0.45 * 0.45;
  for (int b = 0; b < 2; ++b)
  {
    TEST_ASSERT(agreed(c.chi[b] == 2 && c.B[b] == 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(c.volume[b] - bulb) / bulb < 0.05, MPI_COMM_WORLD));
  }
  free_census(&c);
  return 0;
}

static int test_breakup_is_not_undone_by_the_next_call(void)
{
  /* The bias must open the neck wider than the COALESCENCE threshold, or the
   * next call sees two caps facing each other across a thin gap and merges them
   * back. Run twice: the second call must find nothing. */
  MFTL_TopologyOptions o = base_options(0.12, 0.1);
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
  scene_dumbbell(ens);
  settle(ens);
  MFTL_TopologyStats s1, s2;
  MFTLEnsembleApplyTopologyChanges(ens, &o, &s1);
  MFTLEnsembleApplyTopologyChanges(ens, &o, &s2);
  census c;
  take_census(ens, &c);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(s1.n_breakup > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(s2.n_events == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 2, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_two_close_spheres_coalesce(void)
{
  MFTL_TopologyOptions o = base_options(0.02, 0.1);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_two_spheres, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_coalescence > 0 && st.n_breakup == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_bodies_retired == 1 && st.n_bodies_created == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.chi[0] == 2, MPI_COMM_WORLD));
  /* The bridge adds volume; it must not be more than a few percent. */
  const double two = 2.0 * 4.0 / 3.0 * MFTL_PI * 0.125;
  TEST_ASSERT(agreed(c.volume[0] > 0.97 * two && c.volume[0] < 1.06 * two, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/**
 * @brief Two drops almost touching coalesce into ONE sphere-like body, with no
 *        breakup event anywhere.
 *
 * Near contact, a pair of triangles offset sideways across the gap has a
 * centroid difference whose lateral part, projected on the tilted normals,
 * outweighs the gap: read from centroids, each lies "behind" the other and the
 * pair is a BREAKUP. Its bias then opens the very neck the coalescence events
 * close, and the merged drop came out as a torus at a gap of 0.005, with genus
 * two when touching and an entrained bubble at 0.01 -- while every gap of 0.02
 * or more came out right. The side is read along the closest-point segment and
 * only for a pair facing each other across it, and this test fails without that.
 */
static int test_two_nearly_touching_drops_merge_into_one(void)
{
  MFTL_TopologyOptions o = base_options(0.12, 0.1);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_two_nearly_touching_spheres, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_coalescence > 0 && st.n_breakup == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0 && c.chi[0] == 2 && c.B[0] == 0, MPI_COMM_WORLD));
  const double both = 2.0 * 4.0 / 3.0 * MFTL_PI * 0.125;
  TEST_ASSERT(agreed(fabs(c.volume[0] - both) / both < 0.02, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_chain_of_three_merges_into_one(void)
{
  MFTL_TopologyOptions o = base_options(0.02, 0.1);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_three_in_a_row, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_clusters == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.chi[0] == 2, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_bodies_retired == 2, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_narrow_torus_hole_closes(void)
{
  /* Coalescence WITHIN one body changes its genus: the torus becomes a sphere
   * topologically, and keeps its id. */
  MFTL_TopologyOptions o = base_options(0.02, 0.2);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_torus, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_coalescence > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0 && c.chi[0] == 2, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_bubble_bursts_through_a_thin_film(void)
{
  /* The film between a drop and a bubble inside it is liquid: a BREAKUP event
   * between two different bodies. Only the union of the two fields opens it --
   * each alone would erode the film from one side -- and the result is the drop
   * with a crater: one body, genus zero, the bubble's id retired. */
  MFTL_TopologyOptions o = base_options(0.12, 0.02);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_drop_with_bubble, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_breakup > 0 && st.n_clusters == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0 && c.chi[0] == 2, MPI_COMM_WORLD));
  const double v = 4.0 / 3.0 * MFTL_PI * (1.0 - 0.064);
  TEST_ASSERT(agreed(fabs(c.volume[0] - v) / v < 0.03, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_drop_pinches_off_an_open_sheet_whose_rim_stays_on_the_walls(void)
{
  MFTL_TopologyOptions o = base_options(0.12, 0.02);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_sheet_with_drop, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(c.n_live == 2, MPI_COMM_WORLD));
  /* Body 0 is the sheet: a disk, its rim entirely on the walls; body 1 the drop. */
  TEST_ASSERT(agreed(c.chi[0] == 1 && c.B[0] > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.chi[1] == 2 && c.B[1] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.rim_off_wall == 0 && c.flag_mismatch == 0, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_a_satellite_below_the_size_floor_is_discarded(void)
{
  MFTL_TopologyOptions o = base_options(0.1, 0.02);
  o.min_body_triangles = 400;
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_lopsided, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_bodies_discarded >= 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.gid[0] == 0 && c.chi[0] == 2, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/**
 * The floor is taken on the marching-cubes surface AND on what the remesher
 * leaves. Without a remesh the satellite clears it as a few hundred slivers;
 * remeshed at a target edge length near its radius it comes out as a handful of
 * triangles -- in the Rayleigh-Taylor example, a tetrahedron, all four faces
 * facing against all three neighbours -- and must go, its volume to the bulb.
 * Not compared against a single-rank reference: the distributed remesher is
 * deterministic per rank count, not across rank counts.
 */
static int test_a_satellite_the_remesher_collapses_is_discarded(void)
{
  MFTL_TopologyOptions o = base_options(0.1, 0.02);
  o.min_body_triangles = 16;

  census plain, before, after;
  MFTL_TopologyStats st;
  {
    MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
    scene_lopsided(ens);
    settle(ens);
    MFTLEnsembleApplyTopologyChanges(ens, &o, &st);
    take_census(ens, &plain);
    MFTLEnsembleDestroy(&ens);
  }
  TEST_ASSERT(agreed(plain.n_live == 2 && st.n_bodies_created == 1 && st.n_bodies_discarded == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(plain.T[0] >= o.min_body_triangles && plain.T[1] >= o.min_body_triangles, MPI_COMM_WORLD));
  free_census(&plain);

  MFTL_RemeshParameters rp;
  MFTLInitRemeshParams(&rp, MFTL_REMESH_ISOTROPIC, 0.15, 5, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  rp.parallel_phase_budget = 8;
  o.remesh = &rp;
  {
    MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
    scene_lopsided(ens);
    settle(ens);
    take_census(ens, &before);
    TEST_ASSERT(agreed(MFTLEnsembleApplyTopologyChanges(ens, &o, &st) == MFTL_SUCCESS, MPI_COMM_WORLD));
    take_census(ens, &after);
    MFTLEnsembleDestroy(&ens);
  }
  if (g_rank == 0)
    printf("\n  bulb %lld triangles after the remesh; V %.8f -> %.8f (stats: before %.8f raw %.8f after %.8f)\n  ", after.T[0], before.volume[0],
           after.n_live > 0 ? after.volume[0] : 0.0, st.volume_before, st.volume_raw, st.volume_after);
  TEST_ASSERT(agreed(st.n_bodies_discarded == 1 && st.n_bodies_created == 0 && st.n_bodies_out == 1 && st.n_bodies_retired == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.n_live == 1 && after.gid[0] == 0 && after.chi[0] == 2 && after.T[0] >= o.min_body_triangles, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(after.partition_ok && after.ids_dense, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(after.volume[0] - before.volume[0]) <= 1e-5 * before.volume[0], MPI_COMM_WORLD));
  free_census(&before);
  free_census(&after);
  return 0;
}

static int test_every_piece_below_the_floor_leaves_no_body(void)
{
  /* The floor above both halves of a pinched dumbbell: nothing survives, the
   * original id is retired, and the ensemble is empty but consistent. */
  MFTL_TopologyOptions o = base_options(0.12, 0.02);
  o.min_body_triangles = 1000000;
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_dumbbell, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(c.n_live == 0 && st.n_bodies_discarded == 2 && st.n_bodies_retired == 1, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_an_annulus_periodic_in_x_keeps_its_rims_on_the_walls(void)
{
  /* The travelling-wave topology: a sheet across the box, periodic in x, its
   * two rims on the y walls. Reconstructed with no event it must come back an
   * annulus (V-E+F = 0), its rims on the walls and flagged, identical to the
   * np = 1 answer -- at every grid, including those that cut the periodic
   * axis, where the seam is itself a partition boundary. */
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  census c;
  MFTL_TopologyStats st;
  g_axes[0] = MFTL_TRUE;
  g_axes[1] = g_axes[2] = MFTL_FALSE;
  g_axes_set = MFTL_TRUE;
  const int rc = run_scene(scene_annulus, MFTL_FALSE, &o, MFTL_TRUE, &c, &st);
  g_axes_set = MFTL_FALSE;
  TEST_ASSERT_EQUAL_INT(0, rc);
  if (g_rank == 0 && !(c.n_live == 1 && c.chi[0] == 0)) printf("\n  bodies %d, chi %d, B %lld\n  ", c.n_live, c.chi[0], c.B[0]);
  TEST_ASSERT(agreed(c.n_live == 1 && c.chi[0] == 0 && c.B[0] > 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.rim_off_wall == 0 && c.flag_mismatch == 0, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_seam_vertices_get_one_owner_on_every_rank(void)
{
  /* The same annulus in the travelling-wave example's own box -- x in [-2, 2],
   * 54 lattice cells over the period -- where `-2 + 54 * (4/54) - 4` is NOT
   * the bits of `-2`. A marching-cubes vertex on the seam was positioned from
   * the node index of the cell that created it, so the two ranks either side of
   * the seam put it a last bit apart, MFTLGetOwnerProc gave it two owners, and
   * the redistribution tore the surface along the seam: 28 twin-less half-edges
   * in the example at np = 2. Vertices now come from WRAPPED nodes. The origin
   * of 0 and 60 cells the other tests use rounds consistently and hid it. */
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  o.dx = L_BOX / 54.0;
  census c;
  MFTL_TopologyStats st;
  g_axes[0] = MFTL_TRUE;
  g_axes[1] = g_axes[2] = MFTL_FALSE;
  g_axes_set = MFTL_TRUE;
  g_origin = -2.0;
  const int rc = run_scene(scene_annulus, MFTL_FALSE, &o, MFTL_TRUE, &c, &st);
  g_origin = 0.0;
  g_axes_set = MFTL_FALSE;
  TEST_ASSERT_EQUAL_INT(0, rc);
  TEST_ASSERT(agreed(st.lattice_cells[0] == 54, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.n_live == 1 && c.chi[0] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.rim_off_wall == 0 && c.flag_mismatch == 0 && c.partition_ok, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_breakup_across_the_periodic_seam(void)
{
  MFTL_TopologyOptions o = base_options(0.12, 0.02);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_seam_dumbbell, MFTL_TRUE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(c.n_live == 2, MPI_COMM_WORLD));
  const double bulb = 4.0 / 3.0 * MFTL_PI * 0.45 * 0.45 * 0.45;
  for (int b = 0; b < 2; ++b)
  {
    TEST_ASSERT(agreed(c.chi[b] == 2, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(c.volume[b] - bulb) / bulb < 0.05, MPI_COMM_WORLD));
  }
  free_census(&c);
  return 0;
}

/**
 * @brief A body that straddles both the seam and the middle of the box on
 *        every axis is MEASURED, so its volume is corrected.
 *
 * The cluster extent used to be taken in two frames, cut at the seam and at
 * L/2. This sphere contains both cuts on every axis, so both frames measured it
 * as wide as the box, every axis was declared covered, no flux axis was left,
 * and the volume correction was SKIPPED -- silently: volume_before came out 0
 * and the marching-cubes deficit stayed in the mesh. Four frames measure every
 * body narrower than 3L/4 exactly. The census volume is not used: it unwraps
 * about one vertex, which a body wider than L/2 defeats.
 */
static int test_a_body_straddling_both_half_frames_keeps_its_volume(void)
{
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_straddling_sphere, MFTL_TRUE, &o, MFTL_TRUE, &c, &st));
  const double exact = 4.0 / 3.0 * MFTL_PI * 1.15 * 1.15 * 1.15;
  if (g_rank == 0) printf("\n  V %.10f -> raw %.10f -> %.10f (exact %.10f)\n  ", st.volume_before, st.volume_raw, st.volume_after, exact);
  TEST_ASSERT(agreed(c.n_live == 1 && c.chi[0] == 2 && c.B[0] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_clusters == 1 && st.n_clusters_unmeasured == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(st.volume_before - exact) / exact < 0.01, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(st.volume_raw - st.volume_before) >= 1e-4 * exact, MPI_COMM_WORLD));  // there was a deficit to correct
  TEST_ASSERT(agreed(fabs(st.volume_after - st.volume_before) <= 1e-9 * exact, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/**
 * @brief A compact body at least 3L/4 wide on every periodic axis is treated as
 *        covering them all: reconstructed, left uncorrected, and COUNTED.
 *
 * Far outside the periodic extent rule. The library cannot tell it from a body
 * wrapping all three axes at every placement, so it does not try at any: the
 * reconstruction is still valid, and n_clusters_unmeasured says that no volume
 * correction was applied.
 */
static int test_a_body_covering_every_axis_is_counted_as_unmeasured(void)
{
  MFTL_TopologyOptions o = base_options(0.0, 0.0);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_wide_sphere, MFTL_TRUE, &o, MFTL_TRUE, &c, &st));
  TEST_ASSERT(agreed(c.n_live == 1 && c.chi[0] == 2 && c.B[0] == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_clusters == 1 && st.n_clusters_unmeasured == 1, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.volume_before == 0.0 && st.volume_after == 0.0, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

static int test_breakup_and_coalescence_in_one_call(void)
{
  MFTL_TopologyOptions o = base_options(0.12, 0.12);
  census c;
  MFTL_TopologyStats st;
  TEST_ASSERT_EQUAL_INT(0, run_scene(scene_breakup_and_merge, MFTL_FALSE, &o, MFTL_FALSE, &c, &st));
  TEST_ASSERT(agreed(st.n_breakup > 0 && st.n_coalescence > 0 && st.n_clusters == 2, MPI_COMM_WORLD));
  /* The dumbbell (0) splits: 0 and a new id 3; the spheres 1 and 2 merge into 1. */
  TEST_ASSERT(agreed(c.n_live == 3, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(c.gid[0] == 0 && c.gid[1] == 1 && c.gid[2] == 3, MPI_COMM_WORLD));
  for (int b = 0; b < 3; ++b) TEST_ASSERT(agreed(c.chi[b] == 2, MPI_COMM_WORLD));
  free_census(&c);
  return 0;
}

/**
 * @brief Liquid is neither destroyed nor created by a topology change.
 *
 * With `conserve_volume` on (the default) the enclosed volume of every
 * reconstructed cluster is restored after marching cubes. Checked against the
 * volume BEFORE the call, over a breakup, a coalescence and a bubble bursting
 * through a film -- and the raw marching-cubes volume must differ from it, or
 * the correction had nothing to do and this proves nothing.
 */
static int test_volume_is_conserved_through_every_kind_of_change(void)
{
  struct
  {
    scene_fn scene;
    double db, dc;
  } cases[] = {{scene_dumbbell, 0.12, 0.02}, {scene_two_spheres, 0.02, 0.1}, {scene_drop_with_bubble, 0.12, 0.02}};

  for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
  {
    MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
    cases[i].scene(ens);
    settle(ens);
    census before, after;
    take_census(ens, &before);
    MFTL_TopologyOptions o = base_options(cases[i].db, cases[i].dc);
    MFTL_TopologyStats st;
    MFTLEnsembleApplyTopologyChanges(ens, &o, &st);
    take_census(ens, &after);
    MFTLEnsembleDestroy(&ens);

    double v0 = 0.0, v1 = 0.0;
    for (int b = 0; b < before.n_live; ++b) v0 += before.volume[b];
    for (int b = 0; b < after.n_live; ++b) v1 += after.volume[b];
    free_census(&before);
    free_census(&after);

    if (g_rank == 0)
      printf("\n  case %d: V %.8f -> %.8f (rel %.2e); raw marching cubes %.8f (rel %.2e)\n  ", (int) i, v0, v1, (v1 - v0) / v0, st.volume_raw,
             (st.volume_raw - st.volume_before) / st.volume_before);
    TEST_ASSERT(agreed(st.n_events > 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(st.volume_before - v0) <= 1e-12 * v0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(v1 - v0) <= 1e-5 * v0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(fabs(st.volume_raw - v0) >= 1e-3 * v0, MPI_COMM_WORLD));
  }
  return 0;
}

static int test_no_event_changes_nothing(void)
{
  /* A well-resolved sphere: nothing to detect. The pools must be untouched --
   * same ids, same positions -- and no collective beyond the detection may run. */
  MFTL_TopologyOptions o = base_options(0.05, 0.05);
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
  scene_sphere(ens);
  settle(ens);
  census before, after;
  take_census(ens, &before);
  MFTL_TopologyStats st;
  const int rc = MFTLEnsembleApplyTopologyChanges(ens, &o, &st);
  take_census(ens, &after);
  MFTLEnsembleDestroy(&ens);
  int same = 1;
  if (g_rank == 0) same = same_result(&before, &after);
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free_census(&before);
  free_census(&after);
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS && st.n_events == 0 && same, MPI_COMM_WORLD));
  return 0;
}

static int test_a_rim_off_the_walls_is_refused(void)
{
  /* An open patch with a FREE rim in the middle of the domain, pinched: a level
   * set cannot represent the rim, so the cluster is refused and the body is
   * left exactly as it was. */
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  if (g_rank == 0)
  {
    /* A folded strip: two parallel squares 0.05 apart joined along one edge,
     * facing each other -- a coalescence gap -- with free rims elsewhere. */
    const int n = 12;
    double *pos = (double *) malloc((size_t) 2 * (n + 1) * (n + 1) * 3 * sizeof(double));
    int *tri = (int *) malloc((size_t) 2 * 2 * n * n * 3 * sizeof(int));
    int nv = 0, nt = 0;
    for (int s = 0; s < 2; ++s)
      for (int j = 0; j <= n; ++j)
        for (int i = 0; i <= n; ++i)
        {
          pos[3 * nv] = C - 0.5 + i / (double) n;
          pos[3 * nv + 1] = C - 0.5 + j / (double) n;
          pos[3 * nv + 2] = C + (s ? 0.025 : -0.025);
          nv++;
        }
    for (int s = 0; s < 2; ++s)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
        {
          const int o0 = s * (n + 1) * (n + 1), a = o0 + j * (n + 1) + i, b = a + 1, c2 = a + n + 1, d = c2 + 1;
          if (s == 0)
          {  // lower sheet faces up: normal +z
            tri[3 * nt] = a, tri[3 * nt + 1] = b, tri[3 * nt + 2] = d, nt++;
            tri[3 * nt] = a, tri[3 * nt + 1] = d, tri[3 * nt + 2] = c2, nt++;
          }
          else
          {  // upper sheet faces down: normal -z
            tri[3 * nt] = a, tri[3 * nt + 1] = d, tri[3 * nt + 2] = b, nt++;
            tri[3 * nt] = a, tri[3 * nt + 1] = c2, tri[3 * nt + 2] = d, nt++;
          }
        }
    MFTLBuildIndexedMesh(obj, pos, nv, tri, nt);
    free(pos);
    free(tri);
  }
  else
    MFTLObjectBuildDerived(obj);
  settle(ens);

  census before, after;
  take_census(ens, &before);
  MFTL_TopologyOptions o = base_options(0.02, 0.1);
  MFTL_TopologyStats st;
  const int rc = MFTLEnsembleApplyTopologyChanges(ens, &o, &st);
  take_census(ens, &after);
  MFTLEnsembleDestroy(&ens);

  int same = 1;
  if (g_rank == 0) same = same_result(&before, &after);
  MPI_Bcast(&same, 1, MPI_INT, 0, MPI_COMM_WORLD);
  free_census(&before);
  free_census(&after);
  TEST_ASSERT(agreed(rc == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st.n_coalescence > 0 && st.n_clusters_refused == 1 && st.n_clusters == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  return 0;
}

static int test_invalid_arguments_are_refused_everywhere(void)
{
  MFTL_Ensemble *ens = make_ens(MPI_COMM_WORLD, MFTL_FALSE);
  scene_sphere(ens);
  settle(ens);
  MFTL_TopologyOptions o = base_options(0.1, 0.1);
  o.dx = 0.0;
  int body = 0;
  const int rc1 = MFTLEnsembleReconstructBodies(ens, &body, 1, MFTL_NULL, 0, &o, MFTL_NULL);
  const int rc2 = MFTLEnsembleApplyTopologyChanges(MFTL_NULL, &o, MFTL_NULL);
  /* Zero thresholds detect nothing and change nothing. */
  MFTL_TopologyOptions z = base_options(0.0, 0.0);
  MFTL_TopologyStats st;
  const int rc3 = MFTLEnsembleApplyTopologyChanges(ens, &z, &st);
  /* An empty body list is a successful no-op. */
  const int rc4 = MFTLEnsembleReconstructBodies(ens, MFTL_NULL, 0, MFTL_NULL, 0, &z, MFTL_NULL);
  MFTLEnsembleDestroy(&ens);
  TEST_ASSERT(agreed(rc1 == MFTL_ERROR && rc2 == MFTL_ERROR && rc3 == MFTL_SUCCESS && st.n_events == 0 && rc4 == MFTL_SUCCESS, MPI_COMM_WORLD));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0)
  {
    int cuts[3] = {0, 0, 0};
    MPI_Dims_create(g_nproc, 3, cuts);
    printf("--- Running MFTL Topology Change Tests (np = %d, grid %dx%dx%d) ---\n", g_nproc, cuts[0], cuts[1], cuts[2]);
  }

  RUN_TEST(test_the_lattice_divides_every_axis_per_rank);
  RUN_TEST(test_reconstructing_a_sphere_without_events_is_faithful);
  RUN_TEST(test_a_regridded_body_keeps_its_vertices_on_the_old_surface);
  RUN_TEST(test_a_biased_edge_follows_the_bias_not_the_old_surface);
  RUN_TEST(test_a_thin_neck_breaks_into_two_bodies);
  RUN_TEST(test_breakup_is_not_undone_by_the_next_call);
  RUN_TEST(test_two_close_spheres_coalesce);
  RUN_TEST(test_two_nearly_touching_drops_merge_into_one);
  RUN_TEST(test_a_chain_of_three_merges_into_one);
  RUN_TEST(test_a_narrow_torus_hole_closes);
  RUN_TEST(test_a_bubble_bursts_through_a_thin_film);
  RUN_TEST(test_a_drop_pinches_off_an_open_sheet_whose_rim_stays_on_the_walls);
  RUN_TEST(test_a_satellite_below_the_size_floor_is_discarded);
  RUN_TEST(test_a_satellite_the_remesher_collapses_is_discarded);
  RUN_TEST(test_every_piece_below_the_floor_leaves_no_body);
  RUN_TEST(test_breakup_across_the_periodic_seam);
  RUN_TEST(test_a_body_straddling_both_half_frames_keeps_its_volume);
  RUN_TEST(test_a_body_covering_every_axis_is_counted_as_unmeasured);
  RUN_TEST(test_an_annulus_periodic_in_x_keeps_its_rims_on_the_walls);
  RUN_TEST(test_seam_vertices_get_one_owner_on_every_rank);
  RUN_TEST(test_breakup_and_coalescence_in_one_call);
  RUN_TEST(test_volume_is_conserved_through_every_kind_of_change);
  RUN_TEST(test_no_event_changes_nothing);
  RUN_TEST(test_a_rim_off_the_walls_is_refused);
  RUN_TEST(test_invalid_arguments_are_refused_everywhere);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
