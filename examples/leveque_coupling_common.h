/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief What the two coupled LeVeque examples share: the velocity field, the
 *        coupled and the direct RK4 step, and the two passes that measure them.
 *
 * THE SETUP. LeVeque's time-reversed vortex in the unit cube: a sphere of
 * radius 0.15 at (0.35, 0.35, 0.35) is wound into a thin filament by t = T/2
 * and unwound back onto itself by t = T. The exact solution at T is the
 * initial sphere, so the state at T measures everything that went wrong on
 * the way.
 *
 * THE COUPLING. The velocity exists ONLY on the solver's samples (a Cartesian
 * MAC grid, or the cells of an unstructured tetrahedral mesh). Each RK4 stage
 * binds the markers at the stage positions inside one window (no new request),
 * fetches the field at the stage time (three fetches per step: t, t + dt/2,
 * t + dt) and interpolates. The DIRECT run evaluates the analytic field at the
 * vertices -- the same RK4, the same dt, no coupler: the difference between
 * the two is the coupling error and nothing else.
 *
 * TWO PASSES.
 *   A  Lagrangian: no remeshing, so every vertex keeps its identity and its
 *      position can be compared with its own exact-velocity trajectory, which
 *      every rank integrates for every vertex (a reference that never touches
 *      the coupler). Position error, interpolated-velocity error, volume.
 *   B  Front tracking: remeshed every step, the way a solver runs it. The
 *      coupled and the direct ensemble are remeshed identically; the volume,
 *      the area and the distance to the exact sphere at T are compared.
 */

#ifndef LEVEQUE_COUPLING_COMMON_H
#define LEVEQUE_COUPLING_COMMON_H

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_coupling.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The field
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief The spatial part of LeVeque's field; the time factor is cos(pi t / T). */
static void lv_spatial(const double x[3], double u[3])
{
  const double sx = sin(MFTL_PI * x[0]), sy = sin(MFTL_PI * x[1]), sz = sin(MFTL_PI * x[2]);
  u[0] = 2.0 * sx * sx * sin(2.0 * MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[2]);
  u[1] = -sy * sy * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[2]);
  u[2] = -sz * sz * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]);
}

static double lv_time_factor(double t, double T) { return cos(MFTL_PI * t / T); }

static void lv_velocity(const double x[3], double t, double T, double u[3])
{
  lv_spatial(x, u);
  const double s = lv_time_factor(t, T);
  u[0] *= s;
  u[1] *= s;
  u[2] *= s;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The solver, as the coupler sees it
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A fake solver: three velocity fields on registered samples, which it
 *        can bring to any time. A real solver would advance its own equations;
 *        here the field is the analytic one sampled at the solver's DOFs.
 */
typedef struct lv_solver
{
  MFTL_SolverGrid *grid;
  MFTL_SolverField field[3];
  void (*set_time)(struct lv_solver *s, double t);
  void *impl;
  double T;
  uint64_t version;
} lv_solver;

/* ═══════════════════════════════════════════════════════════════════════════
 * Small parallel helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

static int lv_rank(void)
{
  int r;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

static int lv_nproc(void)
{
  int n;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}

static void lv_grid_for(int nproc, int cuts[3])
{
  static const int table[][4] = {{1, 1, 1, 1}, {2, 2, 1, 1},  {3, 3, 1, 1},  {4, 2, 2, 1}, {6, 3, 2, 1},
                                 {8, 2, 2, 2}, {12, 3, 2, 2}, {16, 4, 2, 2}, {27, 3, 3, 3}};
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i)
    if (table[i][0] == nproc)
    {
      cuts[0] = table[i][1];
      cuts[1] = table[i][2];
      cuts[2] = table[i][3];
      return;
    }
  cuts[0] = nproc;
  cuts[1] = cuts[2] = 1;
}

static MFTL_Ensemble *lv_make_ensemble(int refine)
{
  int cuts[3];
  lv_grid_for(lv_nproc(), cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {false, false, false};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 1);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.35, 0.35, 0.35};
  MFTLGenerateSphere(obj, c, 0.15, refine);
  MFTLEnsembleComputeTriangleGeometry(ens);
  MFTLRedistribute(ens);
  MFTLEnsembleComputeTriangleGeometry(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  return ens;
}

/** @brief After positions moved: ghosts, geometry, ownership, geometry. */
static void lv_settle(MFTL_Ensemble *ens)
{
  MFTLUpdateGhostPositions(ens);
  MFTLEnsembleComputeTriangleGeometry(ens);
  MFTLRedistribute(ens);
  MFTLEnsembleComputeTriangleGeometry(ens);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The two RK4 steps
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Direct: the analytic field at the owned vertices. */
static void lv_direct_step(MFTL_Ensemble *ens, double t, double dt, double T)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    MFTL_FOREACH_OWNED_VERTEX(obj, {
      double k1[3], k2[3], k3[3], k4[3], x[3];
      lv_velocity(pos, t, T, k1);
      for (int k = 0; k < 3; ++k) x[k] = pos[k] + 0.5 * dt * k1[k];
      lv_velocity(x, t + 0.5 * dt, T, k2);
      for (int k = 0; k < 3; ++k) x[k] = pos[k] + 0.5 * dt * k2[k];
      lv_velocity(x, t + 0.5 * dt, T, k3);
      for (int k = 0; k < 3; ++k) x[k] = pos[k] + dt * k3[k];
      lv_velocity(x, t + dt, T, k4);
      for (int k = 0; k < 3; ++k) pos[k] += (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    });
  }
  MFTLEnsembleMarkChanged(ens);
}

/** @brief What the coupled stepping keeps between steps, and what it counts. */
typedef struct
{
  MFTL_Coupler *c;
  MFTL_CouplingWindow *win;
  double skin[3];
  long n_steps, n_windows, n_restarts, n_fetches;
  double err_max, err_sq, err_n;  ///< velocity interpolation error at stage 1, against the analytic field
  double t_coupling;  ///< wall time in coupling calls
  double t_window, t_bind, t_fetch, t_interp;  ///< its parts
} lv_coupled;

static int lv_new_window(lv_coupled *cp)
{
  MFTLCouplerReleaseWindow(&cp->win);
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  ms.interpolation_only = true;
  MFTL_WindowSpec ws = {{cp->skin[0], cp->skin[1], cp->skin[2]}};
  MFTL_SupportRequest *req = MFTL_NULL;
  const double t0 = MPI_Wtime();
  int st = MFTLCouplerRequestSupport(cp->c, &ms, &ws, &req);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerPrepareWindow(cp->c, req, &cp->win);
  MFTLCouplerReleaseRequest(&req);
  cp->t_window += MPI_Wtime() - t0;
  cp->n_windows++;
  return st;
}

/** @brief Fetch the solver's field at time @p t. */
static int lv_fetch(lv_coupled *cp, lv_solver *s, double t, MFTL_CouplingFetch **f)
{
  s->set_time(s, t);
  cp->n_fetches++;
  const double t0 = MPI_Wtime();
  const int st = MFTLCouplerFetch(cp->c, cp->win, 3, s->field, f);
  cp->t_fetch += MPI_Wtime() - t0;
  return st;
}

/**
 * @brief One RK4 step through the coupler: a window requested at the step's
 *        start positions with a skin covering every stage of the step, four
 *        binds inside it, three fetches. Should a stage still leave it (on any
 *        rank -- the STALE is agreed), the step restarts on a fresh window.
 */
static int lv_coupled_step(lv_coupled *cp, lv_solver *s, MFTL_Ensemble *ens, double t, double dt, double T)
{
  const double t0 = MPI_Wtime();
  MFTL_MarkerSpec ms;
  memset(&ms, 0, sizeof ms);
  ms.kind = MFTL_MARKERS_VERTICES;
  ms.interpolation_only = true;
  int st = MFTL_COUPLING_STALE;
  for (int attempt = 0; attempt < 3 && st == MFTL_COUPLING_STALE; ++attempt)
  {
    if (attempt > 0) cp->n_restarts++;
    st = lv_new_window(cp);
    if (st != MFTL_COUPLING_OK) return st;
    MFTL_CouplingBinding *b0 = MFTL_NULL, *bs = MFTL_NULL;
    MFTL_CouplingFetch *f0 = MFTL_NULL, *fh = MFTL_NULL, *f1 = MFTL_NULL;
    double tb = MPI_Wtime();
    st = MFTLCouplerBind(cp->c, &ms, cp->win, &b0);
    cp->t_bind += MPI_Wtime() - tb;
    if (st != MFTL_COUPLING_OK) continue;
    const int64_t n = MFTLCouplerBindingMarkerCount(b0);
    const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b0);
    const double *x0 = MFTLCouplerBindingPositions(b0);
    double *k = (double *) malloc(4 * 3 * (size_t) (n > 0 ? n : 1) * sizeof(double));
    double *xs = (double *) malloc(3 * (size_t) (n > 0 ? n : 1) * sizeof(double));
    double *k1 = k, *k2 = k + 3 * n, *k3 = k + 6 * n, *k4 = k + 9 * n;

    st = lv_fetch(cp, s, t, &f0);
    if (st == MFTL_COUPLING_OK) MFTLCouplerInterpolateFetched(b0, f0, k1);
    /* the interpolation error, against the field the samples were taken from */
    for (int64_t i = 0; i < n && st == MFTL_COUPLING_OK; ++i)
    {
      double u[3];
      lv_velocity(&x0[3 * i], t, T, u);
      for (int q = 0; q < 3; ++q)
      {
        const double e = fabs(k1[3 * i + q] - u[q]);
        if (e > cp->err_max) cp->err_max = e;
        cp->err_sq += e * e;
        cp->err_n += 1.0;
      }
    }
    const double coef[3] = {0.5, 0.5, 1.0};
    double *kin[3] = {k1, k2, k3}, *kout[3] = {k2, k3, k4};
    for (int stage = 0; stage < 3 && st == MFTL_COUPLING_OK; ++stage)
    {
      for (int64_t i = 0; i < 3 * n; ++i) xs[i] = x0[i] + coef[stage] * dt * kin[stage][i];
      MFTL_StageGeometry sg = {xs, MFTL_NULL};
      MFTLCouplerReleaseBinding(&bs);
      tb = MPI_Wtime();
      st = MFTLCouplerBindStage(cp->c, b0, &sg, &bs);
      cp->t_bind += MPI_Wtime() - tb;
      if (st != MFTL_COUPLING_OK) break;
      MFTL_CouplingFetch **f = (stage < 2) ? &fh : &f1;
      if (*f == MFTL_NULL) st = lv_fetch(cp, s, t + coef[stage] * dt, f);
      if (st == MFTL_COUPLING_OK) MFTLCouplerInterpolateFetched(bs, *f, kout[stage]);
    }
    if (st == MFTL_COUPLING_OK)
    {
      for (int64_t i = 0; i < n; ++i)
      {
        MFTL_Object *obj = ens->local_objects[ref[i].slot];
        double *p = &obj->verts.pos[3 * ref[i].entity];
        for (int q = 0; q < 3; ++q) p[q] = x0[3 * i + q] + (dt / 6.0) * (k1[3 * i + q] + 2.0 * k2[3 * i + q] + 2.0 * k3[3 * i + q] + k4[3 * i + q]);
      }
    }
    MFTLCouplerReleaseFetch(&f0);
    MFTLCouplerReleaseFetch(&fh);
    MFTLCouplerReleaseFetch(&f1);
    MFTLCouplerReleaseBinding(&bs);
    MFTLCouplerReleaseBinding(&b0);
    free(k);
    free(xs);
  }
  if (st == MFTL_COUPLING_OK) MFTLEnsembleMarkChanged(ens);
  cp->n_steps++;
  cp->t_coupling += MPI_Wtime() - t0;
  return st;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Measurements
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double volume, area;
  double sphere_max, sphere_rms;  ///< | |x - c| - R | over the vertices
  double n_tri, n_vert, n_edge, min_q;
} lv_shape;

static void lv_measure(MFTL_Ensemble *ens, lv_shape *s)
{
  memset(s, 0, sizeof *s);
  const double c[3] = {0.35, 0.35, 0.35}, R = 0.15;
  double a[6] = {0, 0, 0, 0, 0, 0}, mx[2] = {0.0, -1.0};
  const int b = ens->n_local_objects;
  MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) malloc((size_t) (b > 0 ? b : 1) * sizeof(MFTL_BodyIntegrals));
  MFTLEnsembleBodyIntegrals(ens, bi);
  double minq = 1.0;
  for (int o = 0; o < b; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    if (bi[o].lowest_holder == ens->domain.rank)
    {
      a[0] += bi[o].volume;
      a[1] += bi[o].area;
    }
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double *p = &obj->verts.pos[3 * v];
      const double e = fabs(sqrt((p[0] - c[0]) * (p[0] - c[0]) + (p[1] - c[1]) * (p[1] - c[1]) + (p[2] - c[2]) * (p[2] - c[2])) - R);
      mx[0] = fmax(mx[0], e);
      a[2] += e * e;
      a[3] += 1.0;
    }
    a[4] += obj->tris.n_local;
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double cr[3][3], area = 0.0, ss = 0.0;
      MFTLGetTriangleCorners(obj, t, cr);
      MFTLGetTriangleProperties(obj, t, &area, MFTL_NULL, MFTL_NULL);
      for (int e = 0; e < 3; ++e)
        for (int k = 0; k < 3; ++k)
        {
          const double d = cr[(e + 1) % 3][k] - cr[e][k];
          ss += d * d;
        }
      const double q = (ss > 0.0) ? MFTL_FOUR_SQRT3 * area / ss : 0.0;
      if (q < minq) minq = q;
    }
  }
  free(bi);
  MPI_Allreduce(MPI_IN_PLACE, a, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &minq, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  s->volume = a[0];
  s->area = a[1];
  s->sphere_max = mx[0];
  s->sphere_rms = (a[3] > 0.0) ? sqrt(a[2] / a[3]) : 0.0;
  s->n_tri = a[4];
  s->n_vert = a[3];
  s->n_edge = 1.5 * a[4];
  s->min_q = minq;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pass A: Lagrangian, against per-vertex exact-velocity trajectories
 * ═══════════════════════════════════════════════════════════════════════════ */

#define LV_CHECKPOINTS 4

typedef struct
{
  double t[LV_CHECKPOINTS];
  double pos_max[LV_CHECKPOINTS], pos_rms[LV_CHECKPOINTS];  ///< |x_coupled - x_direct| over the vertices
  double vol_err[LV_CHECKPOINTS];  ///< V_coupled / V_direct - 1
  double vol_direct_err;  ///< V_direct(T) / V0 - 1
  double vol_coupled_err;  ///< V_coupled(T) / V0 - 1
  double sphere_direct, sphere_coupled;  ///< max | |x - c| - R | at T
  double interp_max, interp_rms;
  long windows, restarts, steps, fetches;
  double t_coupling, t_window, t_bind, t_fetch;
  MFTL_CouplerStats stats;
} lv_pass_a;

/** @brief Signed volume of a closed triangle list, positions by vertex id. */
static double lv_volume(const double *x, const int *tri, int nt)
{
  double v = 0.0;
  for (int t = 0; t < nt; ++t)
  {
    const double *a = &x[3 * tri[3 * t]], *b = &x[3 * tri[3 * t + 1]], *c = &x[3 * tri[3 * t + 2]];
    v += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
  }
  return v / 6.0;
}

/**
 * @brief Pass A. Every rank gathers the initial vertex positions and the
 *        triangle list (by vertex id) once, and integrates EVERY vertex with the
 *        exact field -- the direct run, with no ensemble and no coupler at all.
 */
static int lv_run_pass_a(lv_solver *s, const MFTL_CouplerParams *P, int refine, double T, double dt, int max_steps, lv_pass_a *out)
{
  memset(out, 0, sizeof *out);
  MFTL_Ensemble *ens = lv_make_ensemble(refine);
  const MFTL_Object *o0 = (ens->n_local_objects > 0) ? ens->local_objects[0] : MFTL_NULL;
  /* the initial state, everywhere */
  const int nv_loc = o0 ? o0->verts.n_local : 0, nt_loc = o0 ? o0->tris.n_local : 0;
  double *vl = (double *) malloc((size_t) (4 * nv_loc + 1) * sizeof(double));
  double *tl = (double *) malloc((size_t) (3 * nt_loc + 1) * sizeof(double));
  for (int v = 0; v < nv_loc; ++v)
  {
    vl[4 * v] = o0->verts.rec[v].global_id;
    memcpy(&vl[4 * v + 1], &o0->verts.pos[3 * v], 3 * sizeof(double));
  }
  for (int t = 0; t < nt_loc; ++t)
  {
    int vv[3];
    MFTLGetTriangleVertices(o0, t, vv);
    for (int c = 0; c < 3; ++c) tl[3 * t + c] = o0->verts.rec[vv[c]].global_id;
  }
  int np = lv_nproc(), *cnt = (int *) malloc((size_t) np * sizeof(int)), *dsp = (int *) malloc((size_t) np * sizeof(int));
  int m = 4 * nv_loc, tot = 0;
  MPI_Allgather(&m, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < np; ++r)
  {
    dsp[r] = tot;
    tot += cnt[r];
  }
  double *va = (double *) malloc((size_t) (tot + 1) * sizeof(double));
  MPI_Allgatherv(vl, m, MPI_DOUBLE, va, cnt, dsp, MPI_DOUBLE, MPI_COMM_WORLD);
  const int nv = tot / 4;
  m = 3 * nt_loc;
  tot = 0;
  MPI_Allgather(&m, 1, MPI_INT, cnt, 1, MPI_INT, MPI_COMM_WORLD);
  for (int r = 0; r < np; ++r)
  {
    dsp[r] = tot;
    tot += cnt[r];
  }
  double *ta = (double *) malloc((size_t) (tot + 1) * sizeof(double));
  MPI_Allgatherv(tl, m, MPI_DOUBLE, ta, cnt, dsp, MPI_DOUBLE, MPI_COMM_WORLD);
  const int nt = tot / 3;
  double *xr = (double *) malloc(3 * (size_t) (nv + 1) * sizeof(double)), *x0 = (double *) malloc(3 * (size_t) (nv + 1) * sizeof(double));
  int *tri = (int *) malloc(3 * (size_t) (nt + 1) * sizeof(int));
  for (int i = 0; i < nv; ++i) memcpy(&xr[3 * (int) va[4 * i]], &va[4 * i + 1], 3 * sizeof(double));
  memcpy(x0, xr, 3 * (size_t) nv * sizeof(double));
  for (int i = 0; i < 3 * nt; ++i) tri[i] = (int) ta[i];
  free(vl);
  free(tl);
  free(va);
  free(ta);
  free(cnt);
  free(dsp);
  const double V0 = lv_volume(x0, tri, nt);

  lv_coupled cp;
  memset(&cp, 0, sizeof cp);
  int st = MFTLCouplerCreate(ens, s->grid, P, &cp.c);
  const double umax[3] = {2.0, 1.0, 1.0};
  for (int k = 0; k < 3; ++k) cp.skin[k] = 1.1 * dt * umax[k];

  const int n_steps = (int) llround(T / dt);
  const int last = (max_steps > 0 && max_steps < n_steps) ? max_steps : n_steps;
  int targets[LV_CHECKPOINTS];
  for (int i = 0; i < LV_CHECKPOINTS; ++i)
  {
    targets[i] = (int) llround((i + 1) * n_steps / (double) LV_CHECKPOINTS);
    if (targets[i] > last || i == LV_CHECKPOINTS - 1) targets[i] = last;
  }
  for (int step = 0; step < last && st == MFTL_COUPLING_OK; ++step)
  {
    const double t = step * dt;
    st = lv_coupled_step(&cp, s, ens, t, dt, T);
    if (st != MFTL_COUPLING_OK) break;
    lv_settle(ens);
    for (int i = 0; i < nv; ++i)
    {
      double *p = &xr[3 * i], k1[3], k2[3], k3[3], k4[3], x[3];
      lv_velocity(p, t, T, k1);
      for (int k = 0; k < 3; ++k) x[k] = p[k] + 0.5 * dt * k1[k];
      lv_velocity(x, t + 0.5 * dt, T, k2);
      for (int k = 0; k < 3; ++k) x[k] = p[k] + 0.5 * dt * k2[k];
      lv_velocity(x, t + 0.5 * dt, T, k3);
      for (int k = 0; k < 3; ++k) x[k] = p[k] + dt * k3[k];
      lv_velocity(x, t + dt, T, k4);
      for (int k = 0; k < 3; ++k) p[k] += (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
    }
    /* checkpoints at T/4, T/2, 3T/4 and the last step (several may coincide in a short run) */
    bool hit = false;
    for (int i = 0; i < LV_CHECKPOINTS; ++i)
      if (step + 1 == targets[i]) hit = true;
    if (hit)
    {
      double e[3] = {0, 0, 0}, emax = 0.0;
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        const MFTL_Object *obj = ens->local_objects[o];
        for (int v = 0; v < obj->verts.n_local; ++v)
        {
          const double *p = &obj->verts.pos[3 * v], *q = &xr[3 * obj->verts.rec[v].global_id];
          const double d = sqrt((p[0] - q[0]) * (p[0] - q[0]) + (p[1] - q[1]) * (p[1] - q[1]) + (p[2] - q[2]) * (p[2] - q[2]));
          emax = fmax(emax, d);
          e[0] += d * d;
          e[1] += 1.0;
        }
      }
      MPI_Allreduce(MPI_IN_PLACE, e, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE, &emax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
      lv_shape sh;
      lv_measure(ens, &sh);
      for (int i = 0; i < LV_CHECKPOINTS; ++i)
      {
        if (step + 1 != targets[i]) continue;
        out->t[i] = (step + 1) * dt;
        out->pos_max[i] = emax;
        out->pos_rms[i] = sqrt(e[0] / fmax(e[1], 1.0));
        out->vol_err[i] = sh.volume / lv_volume(xr, tri, nt) - 1.0;
      }
      if (step + 1 == last)
      {
        out->vol_coupled_err = sh.volume / V0 - 1.0;
        out->vol_direct_err = lv_volume(xr, tri, nt) / V0 - 1.0;
        out->sphere_coupled = sh.sphere_max;
        double dm = 0.0;
        for (int v = 0; v < nv; ++v)
        {
          const double *p = &xr[3 * v];
          const double r = sqrt((p[0] - 0.35) * (p[0] - 0.35) + (p[1] - 0.35) * (p[1] - 0.35) + (p[2] - 0.35) * (p[2] - 0.35));
          dm = fmax(dm, fabs(r - 0.15));
        }
        out->sphere_direct = dm;
      }
    }
  }
  double ev[2] = {cp.err_sq, cp.err_n};
  MPI_Allreduce(MPI_IN_PLACE, ev, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&cp.err_max, &out->interp_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  out->interp_rms = sqrt(ev[0] / fmax(ev[1], 1.0));
  out->windows = cp.n_windows;
  out->restarts = cp.n_restarts;
  out->steps = cp.n_steps;
  out->fetches = cp.n_fetches;
  MPI_Allreduce(&cp.t_coupling, &out->t_coupling, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double tp[3] = {cp.t_window, cp.t_bind, cp.t_fetch};
  MPI_Allreduce(MPI_IN_PLACE, tp, 3, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  out->t_window = tp[0];
  out->t_bind = tp[1];
  out->t_fetch = tp[2];
  if (cp.c) MFTLCouplerReduceStats(cp.c, &out->stats);
  MFTLCouplerReleaseWindow(&cp.win);
  MFTLCouplerDestroy(&cp.c);
  free(xr);
  free(x0);
  free(tri);
  MFTLEnsembleDestroy(&ens);
  return st;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Pass B: front tracking with remeshing, coupled against direct
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  lv_shape direct0, direct, coupled, direct_half, coupled_half;
  long windows, restarts, steps;
  double t_coupling, t_total;
  int failed_remesh;
} lv_pass_b;

static int lv_run_pass_b(lv_solver *s, const MFTL_CouplerParams *P, int refine, double T, double dt, int max_steps, lv_pass_b *out)
{
  memset(out, 0, sizeof *out);
  MFTL_Ensemble *ed = lv_make_ensemble(refine), *ec = lv_make_ensemble(refine);
  const double target = MFTLEnsembleMeanEdgeLength(ed);
  MFTL_RemeshParameters rp;
  MFTLInitRemeshParams(&rp, MFTL_REMESH_ISOTROPIC, target, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  rp.smooth_iterations = 3;
  rp.parallel_phase_budget = 4;
  rp.parallel_strategy = MFTL_REMESH_PARALLEL_AUTO;
  lv_measure(ed, &out->direct0);

  lv_coupled cp;
  memset(&cp, 0, sizeof cp);
  int st = MFTLCouplerCreate(ec, s->grid, P, &cp.c);
  const double umax[3] = {2.0, 1.0, 1.0};
  for (int k = 0; k < 3; ++k) cp.skin[k] = 1.1 * dt * umax[k];
  const int n_steps = (int) llround(T / dt);
  const int last = (max_steps > 0 && max_steps < n_steps) ? max_steps : n_steps;
  const double t0 = MPI_Wtime();
  for (int step = 0; step < last && st == MFTL_COUPLING_OK; ++step)
  {
    const double t = step * dt;
    lv_direct_step(ed, t, dt, T);
    st = lv_coupled_step(&cp, s, ec, t, dt, T);
    if (st != MFTL_COUPLING_OK) break;
    MFTL_Ensemble *both[2] = {ed, ec};
    for (int e = 0; e < 2; ++e)
    {
      MFTLUpdateGhostPositions(both[e]);
      MFTLEnsembleComputeTriangleGeometry(both[e]);
      MFTL_RemeshStats rs;
      if (MFTLRemeshEnsemble(both[e], &rp, &rs) != MFTL_SUCCESS) out->failed_remesh++;
      MFTLEnsembleComputeTriangleGeometry(both[e]);
    }
    if (step + 1 == last / 2)
    {
      lv_measure(ed, &out->direct_half);
      lv_measure(ec, &out->coupled_half);
    }
  }
  out->t_total = MPI_Wtime() - t0;
  lv_measure(ed, &out->direct);
  lv_measure(ec, &out->coupled);
  out->windows = cp.n_windows;
  out->restarts = cp.n_restarts;
  out->steps = cp.n_steps;
  MPI_Allreduce(&cp.t_coupling, &out->t_coupling, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MFTLCouplerReleaseWindow(&cp.win);
  MFTLCouplerDestroy(&cp.c);
  MFTLEnsembleDestroy(&ed);
  MFTLEnsembleDestroy(&ec);
  return st;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Reporting
 * ═══════════════════════════════════════════════════════════════════════════ */

static void lv_print_pass_a_header(void)
{
  printf("#\n# PASS A -- Lagrangian (no remeshing): coupled vertices against their own exact-velocity trajectories\n");
  printf("#   |dx|: distance of each coupled vertex from its own exact-velocity trajectory, at T/2 and at the end\n");
  printf("#   |du|: interpolated minus exact velocity at the markers, every step; order: of rms|dx| at T/2, in h\n");
  printf("#   (the end state is NOT a coupling measure: the interpolated field cos(pi t/T) W u(x) is exactly time-reversible too,\n");
  printf("#    so its trajectories return to their start as the exact ones do, and both end at the RK4 error)\n");
  printf("# %8s %9s | %10s %10s | %10s %10s | %10s %10s | %10s %10s | %6s\n", "label", "h", "max|dx|T/2", "rms|dx|T/2", "max|dx|end", "rms|dx|end",
         "|du|max", "|du|rms", "V_c/V_d-1", "V_c/V0-1", "order");
}

static void lv_print_pass_a_row(const char *label, double h, const lv_pass_a *a, const lv_pass_a *prev, double hprev)
{
  double order = 0.0;
  if (prev && hprev > 0.0 && a->pos_rms[1] > 0.0) order = log(prev->pos_rms[1] / a->pos_rms[1]) / log(hprev / h);
  printf("  %8s %9.5f | %10.3e %10.3e | %10.3e %10.3e | %10.3e %10.3e | %+10.2e %+10.2e | %6.2f\n", label, h, a->pos_max[1], a->pos_rms[1],
         a->pos_max[LV_CHECKPOINTS - 1], a->pos_rms[LV_CHECKPOINTS - 1], a->interp_max, a->interp_rms, a->vol_err[1], a->vol_coupled_err, order);
}

static void lv_print_pass_b(const char *label, const lv_pass_b *b)
{
  const double V0 = b->direct0.volume, A0 = b->direct0.area;
  printf("#\n# PASS B -- front tracking, remeshed every step: %s against the direct (exact-velocity) run\n", label);
  printf("#   %-34s %14s %14s\n", "", "direct", "coupled");
  printf("    %-34s %14.6f %14.6f\n", "V/V0 at T/2", b->direct_half.volume / V0, b->coupled_half.volume / V0);
  printf("    %-34s %14.6f %14.6f\n", "V/V0 at T", b->direct.volume / V0, b->coupled.volume / V0);
  printf("    %-34s %14.6f %14.6f\n", "A/A0 at T/2", b->direct_half.area / A0, b->coupled_half.area / A0);
  printf("    %-34s %14.6f %14.6f\n", "A/A0 at T", b->direct.area / A0, b->coupled.area / A0);
  printf("    %-34s %14.3e %14.3e\n", "max | |x-c| - R | at T", b->direct.sphere_max, b->coupled.sphere_max);
  printf("    %-34s %14.3e %14.3e\n", "rms | |x-c| - R | at T", b->direct.sphere_rms, b->coupled.sphere_rms);
  printf("    %-34s %14.0f %14.0f\n", "triangles at T", b->direct.n_tri, b->coupled.n_tri);
  printf("    %-34s %14.4f %14.4f\n", "min triangle quality at T", b->direct.min_q, b->coupled.min_q);
  printf("    %-34s %14d %14ld\n", "Euler characteristic V-E+F at T", (int) (b->direct.n_vert - b->direct.n_edge + b->direct.n_tri),
         (long) (b->coupled.n_vert - b->coupled.n_edge + b->coupled.n_tri));
  printf("    coupling: %ld windows for %ld steps (%ld restarts), %.3f s of %.3f s\n", b->windows, b->steps, b->restarts, b->t_coupling, b->t_total);
}

#endif  // LEVEQUE_COUPLING_COMMON_H
