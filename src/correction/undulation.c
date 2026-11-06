/**
 * @file undulation.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief TSUR3D: volume-conserving removal of grid-scale undulations.
 *
 * ## The edge operation
 *
 * For an edge a-b: n_a, n_b are the area-weighted normals of the two fans with
 * BOTH ends moved to the centroids of their one-rings (an end's own noise must
 * not tilt its own normal), n the normalised sum of the two unit normals. With
 * the midpoint m as origin the ends sit at heights h_a = (a - m).n = -h_b. Moving
 * a by d n changes the enclosed volume by d (n . S_a)/3 exactly -- S_a the
 * vector area of a's fan, which depends only on its link -- and no cross term
 * appears when both ends move, because det(n, n, .) = 0 for the two triangles
 * that contain both. With the projected fan areas P = n . S, the common height
 *
 *     h = (h_a P_a + h_b P_b) / (P_a + P_b)
 *
 * therefore conserves the volume exactly. It is the published
 * h = (V_1 + V_2) / (V_1/h_1 + V_2/h_2) with V_i / h_i = P_i / 3 cancelled
 * analytically: the published form is 0/0 on an edge that is already flat (mftl
 * returned NaN and skipped it), this one is simply zero. It is a convex
 * combination, so neither end moves further than |h_a - h_b|. A fan folded
 * against n (P <= 0) is refused, and so is any move that would turn a fan
 * triangle over.
 *
 * ## Who may move what
 *
 * A vertex is FIXED (not owned, or flagged boundary, feature or corner),
 * PRIVATE (owned, and every incident triangle owned and held by no other rank:
 * all its corners and edge neighbours owned -- the rule redistribution SENT
 * by, as in MFTLRemeshMarkFrozen()) or SHARED (owned, some incident triangle
 * is held elsewhere too). A private vertex's whole fan, corners included, is
 * owned here and read and written by nobody else. So:
 *
 *   pass 0, every rank at once   edges with both ends private
 *   pass 1, one colour at a time edges with both ends owned and free, one shared
 *
 * Two ranks of one colour are never 26-neighbours (per-axis parity, a third
 * colour for the last cell of an odd periodic axis), and a triangle holding a
 * vertex of each would make them neighbours -- so within a round no two ranks
 * touch, or read, each other's vertices. Each round ends with a ghost position
 * exchange, so the next colour reads exact copies. An edge whose ends have
 * different owners is never moved; each end still moves through its other edges.
 */

#include "correction_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/** @cond INTERNAL */

/** Largest valence an end may have; beyond it the edge is refused. */
#define MFTL_UND_RING_MAX 64

enum
{
  CLS_FIXED = 0,
  CLS_PRIVATE = 1,
  CLS_SHARED = 2
};

enum
{
  OP_MOVED = 1,
  OP_FLAT = 0,
  OP_REFUSED = -1
};

typedef struct
{
  long long moved, moved_band, flat, refused, cross;
  double max_disp;
} und_counts;

/* ── Small vector helpers ──────────────────────────────────────────────────── */

static inline void cross3(double out[3], const double u[3], const double v[3])
{
  out[0] = u[1] * v[2] - u[2] * v[1];
  out[1] = u[2] * v[0] - u[0] * v[2];
  out[2] = u[0] * v[1] - u[1] * v[0];
}

static inline double dot3(const double u[3], const double v[3]) { return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]; }

/* ── Classification ────────────────────────────────────────────────────────── */

static void classify(const MFTL_Object *obj, unsigned char *cls)
{
  const int nv = obj->verts.n, nl = obj->verts.n_local, tl = obj->tris.n_local;
  const unsigned mask = MFTL_FLAG_BOUNDARY | MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER;
  for (int v = 0; v < nv; ++v) cls[v] = (v < nl && !(obj->verts.rec[v].flags & mask)) ? CLS_PRIVATE : CLS_FIXED;

  for (int t = 0; t < obj->tris.n; ++t)
  {
    const int *const c = &obj->tris.verts[3 * t];
    bool held_elsewhere = t >= tl;
    for (int k = 0; k < 3 && !held_elsewhere; ++k)
    {
      const int nb = obj->tris.neighbors[3 * t + k];
      held_elsewhere = c[k] >= nl || nb >= tl;
    }
    if (!held_elsewhere) continue;
    for (int k = 0; k < 3; ++k)
      if (c[k] < nl && cls[c[k]] == CLS_PRIVATE) cls[c[k]] = CLS_SHARED;
  }
}

/* The outgoing half-edges of v, and whether its fan is closed. */
static int closed_fan(const MFTL_Object *obj, int v, int *fan)
{
  const int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return 0;
  int n = 0, h = start;
  do
  {
    if (n >= MFTL_UND_RING_MAX) return 0;
    fan[n++] = h;
    const int twin = obj->tris.he[h].twin_idx;
    if (twin == MFTL_ERROR) return 0;
    h = MFTLHeNextIdx(twin);
  } while (h != start);
  return n;
}

/* ── The edge operation ────────────────────────────────────────────────────── */

/* Position of x relative to `origin`, by minimum image. */
static inline void rel(const MFTL_Object *obj, const double *origin, int x, double out[3])
{
  MFTLGetMinimumImageVector(obj->domain, out, origin, &obj->verts.pos[3 * x]);
}

/* Sum of the fan normals of the vertex whose outgoing half-edges are `fan`,
 * with the vertex at P0 and the other end `other` at Q (all relative to a). */
static void fan_normal(const MFTL_Object *obj, const double *pa, const int *fan, int n, const double P0[3], int other, const double Q[3], double out[3])
{
  out[0] = out[1] = out[2] = 0.0;
  for (int j = 0; j < n; ++j)
  {
    const int x = MFTLHeTarget(obj, fan[j]), y = MFTLHeTarget(obj, MFTLHeNextIdx(fan[j]));
    double X[3], Y[3], e1[3], e2[3], c[3];
    if (x == other)
      memcpy(X, Q, sizeof(X));
    else
      rel(obj, pa, x, X);
    if (y == other)
      memcpy(Y, Q, sizeof(Y));
    else
      rel(obj, pa, y, Y);
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = X[k] - P0[k];
      e2[k] = Y[k] - P0[k];
    }
    cross3(c, e1, e2);
    for (int k = 0; k < 3; ++k) out[k] += c[k];
  }
}

/* Whether moving the fan's apex from P to P + dP n, and `other` from Q to
 * Q + dQ n, turns any triangle of the fan over. */
static bool fan_folds(const MFTL_Object *obj, const double *pa, const int *fan, int n, const double P[3], double dP, int other, const double Q[3], double dQ,
                      const double nrm[3])
{
  for (int j = 0; j < n; ++j)
  {
    const int x = MFTLHeTarget(obj, fan[j]), y = MFTLHeTarget(obj, MFTLHeNextIdx(fan[j]));
    double X[3], Y[3], X1[3], Y1[3], P1[3], e1[3], e2[3], c0[3], c1[3];
    if (x == other)
      memcpy(X, Q, sizeof(X));
    else
      rel(obj, pa, x, X);
    if (y == other)
      memcpy(Y, Q, sizeof(Y));
    else
      rel(obj, pa, y, Y);
    for (int k = 0; k < 3; ++k)
    {
      P1[k] = P[k] + dP * nrm[k];
      X1[k] = X[k] + (x == other ? dQ * nrm[k] : 0.0);
      Y1[k] = Y[k] + (y == other ? dQ * nrm[k] : 0.0);
    }
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = X[k] - P[k];
      e2[k] = Y[k] - P[k];
    }
    cross3(c0, e1, e2);
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = X1[k] - P1[k];
      e2[k] = Y1[k] - P1[k];
    }
    cross3(c1, e1, e2);
    if (!(dot3(c0, c1) > 0.0)) return MFTL_TRUE;
  }
  return MFTL_FALSE;
}

/* Vector area of a closed fan: half the sum of (X - P) x (Y - P) over its link edges. */
static void fan_area_vector(const MFTL_Object *obj, const double *pa, const int *fan, int n, const double P[3], double out[3])
{
  out[0] = out[1] = out[2] = 0.0;
  for (int j = 0; j < n; ++j)
  {
    double X[3], Y[3], e1[3], e2[3], c[3];
    rel(obj, pa, MFTLHeTarget(obj, fan[j]), X);
    rel(obj, pa, MFTLHeTarget(obj, MFTLHeNextIdx(fan[j])), Y);
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = X[k] - P[k];
      e2[k] = Y[k] - P[k];
    }
    cross3(c, e1, e2);
    for (int k = 0; k < 3; ++k) out[k] += 0.5 * c[k];
  }
}

static int tsur_edge(MFTL_Object *obj, int h, double tan_min, double relaxation, double *disp)
{
  const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
  int fa[MFTL_UND_RING_MAX], fb[MFTL_UND_RING_MAX];
  const int na = closed_fan(obj, a, fa), nb = closed_fan(obj, b, fb);
  if (na < 3 || nb < 3) return OP_REFUSED;

  double *const pa = &obj->verts.pos[3 * a];
  const double A[3] = {0.0, 0.0, 0.0};
  double B[3];
  rel(obj, pa, b, B);

  /* The one-ring centroids. A closed fan's link vertices are exactly the
   * targets of its outgoing half-edges, each once. */
  double cA[3] = {0.0, 0.0, 0.0}, cB[3] = {0.0, 0.0, 0.0};
  for (int j = 0; j < na; ++j)
  {
    double X[3];
    rel(obj, pa, MFTLHeTarget(obj, fa[j]), X);
    for (int k = 0; k < 3; ++k) cA[k] += X[k];
  }
  for (int j = 0; j < nb; ++j)
  {
    double X[3];
    rel(obj, pa, MFTLHeTarget(obj, fb[j]), X);
    for (int k = 0; k < 3; ++k) cB[k] += X[k];
  }
  for (int k = 0; k < 3; ++k)
  {
    cA[k] /= (double) na;
    cB[k] /= (double) nb;
  }

  /* The smoothed normals, both ends at their centroids. */
  double nA[3], nB[3], n[3];
  fan_normal(obj, pa, fa, na, cA, b, cB, nA);
  fan_normal(obj, pa, fb, nb, cB, a, cA, nB);
  const double lA = sqrt(dot3(nA, nA)), lB = sqrt(dot3(nB, nB));
  if (!(lA > 0.0) || !(lB > 0.0)) return OP_REFUSED;
  for (int k = 0; k < 3; ++k) n[k] = nA[k] / lA + nB[k] / lB;
  const double ln = sqrt(dot3(n, n));
  if (!(ln > 0.0)) return OP_REFUSED;
  for (int k = 0; k < 3; ++k) n[k] /= ln;

  /* Heights about the midpoint, and the kink. */
  const double bn = dot3(B, n);
  double bt[3];
  for (int k = 0; k < 3; ++k) bt[k] = B[k] - bn * n[k];
  if (!(fabs(bn) > tan_min * sqrt(dot3(bt, bt)))) return OP_FLAT;
  const double hA = -0.5 * bn, hB = 0.5 * bn;

  /* Projected fan areas; positive unless a fan is folded against n. */
  double SA[3], SB[3];
  fan_area_vector(obj, pa, fa, na, A, SA);
  fan_area_vector(obj, pa, fb, nb, B, SB);
  const double PA = dot3(n, SA), PB = dot3(n, SB);
  if (!(PA > 0.0) || !(PB > 0.0)) return OP_REFUSED;

  const double hc = (hA * PA + hB * PB) / (PA + PB);
  const double dA = relaxation * (hc - hA), dB = relaxation * (hc - hB);
  if (!isfinite(dA) || !isfinite(dB)) return OP_REFUSED;
  if (fan_folds(obj, pa, fa, na, A, dA, b, B, dB, n) || fan_folds(obj, pa, fb, nb, B, dB, a, A, dA, n)) return OP_REFUSED;

  double *const pb = &obj->verts.pos[3 * b];
  for (int k = 0; k < 3; ++k)
  {
    pa[k] += dA * n[k];
    pb[k] += dB * n[k];
  }
  MFTLApplyPeriodicity(obj->domain, pa);
  MFTLApplyPeriodicity(obj->domain, pb);
  *disp = fmax(fabs(dA), fabs(dB));
  return OP_MOVED;
}

/* ── One pass over the edges ───────────────────────────────────────────────── */

/* pass 0: both ends private. pass 1: both free, at least one shared. */
static void sweep_pass(MFTL_Object *obj, const unsigned char *cls, int pass, double tan_min, double relaxation, und_counts *c)
{
  const int nl = obj->verts.n_local;
  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    const int twin = obj->tris.he[h].twin_idx;
    if (twin == MFTL_ERROR || twin < h) continue;
    const int a = MFTLHeOrigin(obj, h), b = MFTLHeTarget(obj, h);
    const int ca = cls[a], cb = cls[b];
    if (pass == 0 && (a < nl) != (b < nl))
    {
      /* Counted once in the job: on the rank owning the end with the smaller gid. */
      const int own = (a < nl) ? a : b, other = (a < nl) ? b : a;
      if (obj->verts.rec[own].global_id < obj->verts.rec[other].global_id) c->cross++;
      continue;
    }
    const bool go = (pass == 0) ? (ca == CLS_PRIVATE && cb == CLS_PRIVATE)
                                : (ca != CLS_FIXED && cb != CLS_FIXED && (ca == CLS_SHARED || cb == CLS_SHARED));
    if (!go) continue;
    double disp = 0.0;
    const int r = tsur_edge(obj, h, tan_min, relaxation, &disp);
    if (r == OP_MOVED)
    {
      c->moved++;
      if (pass == 1) c->moved_band++;
      if (disp > c->max_disp) c->max_disp = disp;
    }
    else if (r == OP_FLAT)
      c->flat++;
    else
      c->refused++;
  }
}

/* ── Parameters ────────────────────────────────────────────────────────────── */

MFTL_UndulationParameters MFTLUndulationDefaultParameters(void)
{
  MFTL_UndulationParameters p;
  p.sweeps = 1;
  p.min_angle = 15.0;  // docs/analysis.md §15.2: 0 smooths away under-resolved geometry
  p.relaxation = 1.0;
  return p;
}

static bool params_ok(const MFTL_UndulationParameters *p)
{
  return p->sweeps >= 0 && p->min_angle >= 0.0 && p->min_angle < 90.0 && p->relaxation > 0.0 && p->relaxation <= 1.0;
}

static void to_stats(const und_counts *c, MFTL_UndulationStats *st)
{
  st->n_moved = c->moved;
  st->n_moved_band = c->moved_band;
  st->n_flat = c->flat;
  st->n_refused = c->refused;
  st->n_cross_rank = c->cross;
  st->max_displacement = c->max_disp;
}

/* ── The local form ────────────────────────────────────────────────────────── */

int MFTLRemoveUndulationsObject(MFTL_Object *obj, const MFTL_UndulationParameters *p, MFTL_UndulationStats *stats)
{
  const MFTL_UndulationParameters def = MFTLUndulationDefaultParameters();
  if (p == MFTL_NULL) p = &def;
  if (obj == MFTL_NULL || obj->domain == MFTL_NULL || !params_ok(p)) return MFTL_ERROR;
  if (stats) memset(stats, 0, sizeof(*stats));
  if (obj->verts.n == 0) return MFTL_SUCCESS;
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  const double t0 = MFTLCorrectionTime();
  const double tan_min = tan(p->min_angle * MFTL_PI / 180.0);
  unsigned char *const cls = (unsigned char *) MFTLMallocOrDie((size_t) obj->verts.n, "undulation: vertex classes");
  classify(obj, cls);
  double *const area0 = MFTLCorrectionSnapshotAreas(obj);

  und_counts c = {0};
  for (int s = 0; s < p->sweeps; ++s) sweep_pass(obj, cls, 0, tan_min, p->relaxation, &c);

  MFTLCorrectionRescaleDensities(obj, area0);
  free(area0);
  free(cls);
  if (c.moved > 0)
  {
    obj->derived &= ~MFTL_DERIVED_GEOMETRY;
    MFTLObjectMarkChanged(obj);
  }
  if (stats)
  {
    to_stats(&c, stats);
    stats->t_total = MFTLCorrectionTime() - t0;
  }
  return MFTL_SUCCESS;
}

/* ── The distributed form ──────────────────────────────────────────────────── */

/* Colour of a cell coordinate on one axis, and the number of colours there. */
static int axis_colour(int coord, int cuts, bool periodic, int *n_colours)
{
  if (cuts <= 1)
  {
    *n_colours = 1;
    return 0;
  }
  if (!periodic || cuts % 2 == 0)
  {
    *n_colours = 2;
    return coord % 2;
  }
  *n_colours = 3;  // an odd cycle needs a third colour
  return (coord == cuts - 1) ? 2 : coord % 2;
}

int MFTLEnsembleRemoveUndulations(MFTL_Ensemble *ens, const MFTL_UndulationParameters *p, MFTL_UndulationStats *stats)
{
  const MFTL_UndulationParameters def = MFTLUndulationDefaultParameters();
  if (p == MFTL_NULL) p = &def;
  MFTL_UndulationStats sink;
  MFTL_UndulationStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  if (ens == MFTL_NULL || !params_ok(p)) return MFTL_ERROR;  // agreed: the parameters are
  if (ens->n_bodies_global == 0 || p->sweeps == 0) return MFTL_SUCCESS;

  MFTL_Domain *const dom = &ens->domain;
  const double t0 = MFTLCorrectionTime();
  const double tan_min = tan(p->min_angle * MFTL_PI / 180.0);

  MFTLUpdateGhostPositions(ens);

  /* Classes and area snapshots: topology does not change inside the call. */
  const int nb = ens->n_local_objects;
  unsigned char **const cls = (unsigned char **) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(unsigned char *), "undulation: class tables");
  double **const area0 = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(double *), "undulation: area snapshots");
  bool any_shared = MFTL_FALSE;
  for (int o = 0; o < nb; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    cls[o] = (unsigned char *) MFTLMallocOrDie((size_t) MFTL_MAX(obj->verts.n, 1), "undulation: vertex classes");
    classify(obj, cls[o]);
    for (int v = 0; v < obj->verts.n_local && !any_shared; ++v) any_shared = cls[o][v] == CLS_SHARED;
    area0[o] = MFTLCorrectionSnapshotAreas(obj);
  }

  /* The colours, and which of them have work anywhere: one Allreduce per call. */
  int n_col = 1, my_col = 0;
  long long work[27] = {0};
  if (dom->nproc > 1)
  {
    int coords[3], nc[3], cc[3];
    MFTLCorrectionCoords(dom, coords);
    for (int k = 0; k < 3; ++k) cc[k] = axis_colour(coords[k], dom->cuts[k], dom->is_periodic[k], &nc[k]);
    n_col = nc[0] * nc[1] * nc[2];
    my_col = cc[0] + nc[0] * (cc[1] + nc[1] * cc[2]);
    work[my_col] = any_shared ? 1 : 0;
    MFTLCorrectionAllreduceLongLong(dom, work, n_col, 1);
  }

  und_counts c = {0};
  for (int s = 0; s < p->sweeps; ++s)
  {
    for (int o = 0; o < nb; ++o) sweep_pass(ens->local_objects[o], cls[o], 0, tan_min, p->relaxation, &c);
    if (dom->nproc == 1) continue;
    for (int col = 0; col < n_col; ++col)
    {
      if (!work[col]) continue;  // agreed: the flags were reduced
      if (col == my_col)
        for (int o = 0; o < nb; ++o) sweep_pass(ens->local_objects[o], cls[o], 1, tan_min, p->relaxation, &c);
      MFTLUpdateGhostPositions(ens);
      st->n_rounds++;
    }
  }

  for (int o = 0; o < nb; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    MFTLCorrectionRescaleDensities(obj, area0[o]);
    free(area0[o]);
    free(cls[o]);
    obj->derived &= ~MFTL_DERIVED_GEOMETRY;
  }
  free(area0);
  free(cls);
  MFTLEnsembleMarkChanged(ens);

  long long counts[5] = {c.moved, c.moved_band, c.flat, c.refused, c.cross};
  MFTLCorrectionAllreduceLongLong(dom, counts, 5, 2);
  double mx[2] = {c.max_disp, MFTLCorrectionTime() - t0};
  MFTLCorrectionAllreduceDouble(dom, mx, 2, 1);
  const int rounds = st->n_rounds;
  c.moved = counts[0];
  c.moved_band = counts[1];
  c.flat = counts[2];
  c.refused = counts[3];
  c.cross = counts[4];
  c.max_disp = mx[0];
  to_stats(&c, st);
  st->n_rounds = rounds;
  st->t_total = mx[1];
  return MFTL_SUCCESS;
}

/** @endcond */
