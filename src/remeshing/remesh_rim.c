/**
 * @file remesh_rim.c
 * @author Christian Gorges
 * @date 2026-10-03
 * @ingroup mftl_remeshing
 *
 * @brief The rim curve: where a split, a collapse and the smoother put a vertex
 *        of an open surface's rim.
 *
 * The chord midpoint of two rim vertices lies on the rim only if the rim is
 * straight. On a curved rim it cuts inside by \f$h^2\kappa/8\f$ per operation,
 * and a remesher that splits, collapses and smooths every pass accumulates that
 * into a rim that visibly shrinks: the jet's nozzle lip, a contact line curving
 * along a side wall. The rim's own vertices already describe the curve.
 *
 * Over the edge a-b the curve is the AVERAGE of two circular arcs: the circle
 * through pred(a), a, b and the circle through a, b, succ(b). Three properties
 * decide it, and the spline it replaced had only the last two:
 *
 * - EXACT ON A CIRCLE AT ANY SPACING. Every circle through three points of a
 *   circle is that circle. A remeshed rim is unevenly spaced by construction
 *   (half-edges next to whole ones after a split), and there the centripetal
 *   Catmull-Rom spline -- exact enough on a regular polygon, 7e-6 inside at
 *   h = 0.13 -- was 4e-4 inside: O(h^2), five times better than the chord and
 *   still a shrinking rim (measured: r = 0.994 after eight refine/coarsen passes
 *   of a unit disk). The arcs leave it at 1 to round-off.
 * - NO OVERSHOOT. Every point of an arc over a-b projects INTO the chord, and
 *   the curve breaks (the corner rule below) before an arc can exceed a quarter
 *   circle, so the sagitta is at most 0.21 L.
 * - THE CHORD ON A STRAIGHT RIM, and a planar wall kept to the bit: every term
 *   is a combination of offsets from a, so a coordinate in which a, b and the
 *   third point agree stays exactly zero. A wall both a and b lie on is
 *   additionally held exactly, so a curve bending round a box corner cannot
 *   pull a point off the wall it belongs to.
 *
 * The curve breaks at a CORNER: a vertex flagged #MFTL_FLAG_CORNER, or one the
 * rim turns at by more than the corner angle (MFTLDetectFeatures()'s own test,
 * so an unclassified mesh breaks at the same places). That side's arc is then
 * not used; with neither, the point is the chord's.
 */

#include "remesh_internal.h"

#include <math.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"

/** @brief Rim placements that fell back to the chord: see MFTLRemeshRimFallbacks(). */
static long g_rim_fallbacks = 0;

long MFTLRemeshRimFallbacks(void) { return g_rim_fallbacks; }

void MFTLRemeshResetRimFallbacks(void) { g_rim_fallbacks = 0; }

int MFTLRemeshRimPrevHe(const MFTL_Object *obj, int h)
{
  /* Incoming half-edges of the origin, rotating across interior edges until the
   * one with no twin: prev(h) ends at the origin, and prev(twin(i)) is the next
   * incoming half-edge round the fan. */
  int i = MFTLHePrevIdx(h);
  for (int step = 0; step < MFTL_REMESH_RING_MAX; ++step)
  {
    const int tw = obj->tris.he[i].twin_idx;
    if (tw == MFTL_ERROR) return MFTLHeIsMeshBoundary(obj, i) ? i : MFTL_ERROR;
    i = MFTLHePrevIdx(tw);
  }
  return MFTL_ERROR;
}

int MFTLRemeshRimNextHe(const MFTL_Object *obj, int h)
{
  /* Outgoing half-edges of the target: next(h) leaves it, and next(twin(o)) is
   * the next outgoing half-edge round the fan. */
  int o = MFTLHeNextIdx(h);
  for (int step = 0; step < MFTL_REMESH_RING_MAX; ++step)
  {
    const int tw = obj->tris.he[o].twin_idx;
    if (tw == MFTL_ERROR) return MFTLHeIsMeshBoundary(obj, o) ? o : MFTL_ERROR;
    o = MFTLHeNextIdx(tw);
  }
  return MFTL_ERROR;
}

double MFTLRemeshRimStraightDot(const MFTL_RemeshParameters *params)
{
  const double angle = (params != MFTL_NULL) ? params->feature_corner_angle : 45.0;
  return -MFTL_COS(angle * MFTL_PI / 180.0);
}

/** @brief Squared length. */
static inline double len2(const double d[3]) { return d[0] * d[0] + d[1] * d[1] + d[2] * d[2]; }

/**
 * @brief Does the rim curve continue straight enough through a vertex, given its
 *        two edge directions OUT of it? MFTLDetectFeatures()'s own test.
 */
static bool passes_through(const double d1[3], const double d2[3], double straight_dot)
{
  const double l = len2(d1) * len2(d2);
  if (!(l > 0.0)) return MFTL_FALSE;
  return (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / MFTL_SQRT(l) <= straight_dot;
}

/**
 * @brief The point at fraction @p f of the circular arc from a to b that, carried
 *        on, passes through a third rim vertex q -- as an offset from a.
 *
 * With v = b - a (length L) and u = q - a: the arc's curvature is that of the
 * circumcircle, \f$\kappa = 2|u\times v| / (|u|\,|v|\,|u-v|)\f$, its half-angle
 * \f$\varphi = \arcsin(\kappa L/2)\f$, and it bulges AWAY from q's side of the
 * chord (the circle continues past a or b back towards the centre). At angle
 * \f$\psi = \varphi(2f-1)\f$ the point is
 * \f[ \tfrac{v}{2} + \tfrac{L}{2}\Big(\frac{\sin\psi}{\sin\varphi}\,\hat c
 *     + \frac{\cos\psi-\cos\varphi}{\sin\varphi}\,\hat n\Big), \f]
 * the chord point itself when q is collinear. Every term is a combination of u
 * and v, so a coordinate in which a, b and q agree stays exactly zero.
 */
static void arc_offset(const double u[3], const double v[3], double f, double out[3])
{
  const double L2 = len2(v), u2 = len2(u);
  const double w[3] = {u[0] - v[0], u[1] - v[1], u[2] - v[2]};
  const double cr[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
  const double cr2 = len2(cr);
  const double den = MFTL_SQRT(u2 * L2 * len2(w));

  double along = 2.0 * f - 1.0, across = 0.0;  // in units of L/2: the chord
  double n[3] = {0.0, 0.0, 0.0};
  if (L2 > 0.0 && cr2 > 0.0 && den > 0.0)
  {
    /* sin(phi) = kappa L / 2 = |u x v| L / (|u| |v| |u - v|) = |u x v| / (|u| |u - v|). */
    double s_phi = MFTL_SQRT(cr2) * MFTL_SQRT(L2) / den;
    if (s_phi > 1.0) s_phi = 1.0;
    const double phi = asin(s_phi);
    const double psi = phi * (2.0 * f - 1.0);
    along = sin(psi) / s_phi;
    across = 2.0 * sin(0.5 * (phi + psi)) * sin(0.5 * (phi - psi)) / s_phi;

    /* n: unit, in the plane of the three, perpendicular to v, away from q. */
    const double t = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / L2;
    const double q_side[3] = {u[0] - t * v[0], u[1] - t * v[1], u[2] - t * v[2]};
    const double ql = MFTL_SQRT(len2(q_side));
    if (ql > 0.0)
      for (int k = 0; k < 3; ++k) n[k] = -q_side[k] / ql;
    else
      across = 0.0;
  }
  for (int k = 0; k < 3; ++k) out[k] = 0.5 * v[k] + 0.5 * along * v[k] + 0.5 * MFTL_SQRT(L2) * across * n[k];
}

bool MFTLRemeshRimCurvePoint(const MFTL_Object *obj, int h, double f, double straight_dot, double out[3])
{
  const MFTL_Domain *const dom = obj->domain;
  const int a = MFTLHeOrigin(obj, h);
  const int b = MFTLHeTarget(obj, h);
  const double *const pa = MFTLVertexPos(obj, a);
  const double *const pb = MFTLVertexPos(obj, b);

  double v[3];
  MFTLGetMinimumImageVector(dom, v, pa, pb);

  const int hp = MFTLRemeshRimPrevHe(obj, h);
  const int hn = MFTLRemeshRimNextHe(obj, h);
  const bool ok = (hp != MFTL_ERROR && hn != MFTL_ERROR);

  /* The two arcs over a-b: through the predecessor of a, and through the
   * successor of b. Each is used unless the rim breaks at the vertex it would
   * continue through (#MFTL_FLAG_CORNER, or a turn sharper than the corner
   * angle); with neither, the chord. */
  double C[3] = {0.0, 0.0, 0.0};
  int n_arcs = 0;
  if (ok)
  {
    double up[3], us_b[3];
    MFTLGetMinimumImageVector(dom, up, pa, MFTLVertexPos(obj, MFTLHeOrigin(obj, hp)));
    MFTLGetMinimumImageVector(dom, us_b, pb, MFTLVertexPos(obj, MFTLHeTarget(obj, hn)));
    const double back[3] = {-v[0], -v[1], -v[2]};

    if (!(obj->verts.rec[a].flags & MFTL_FLAG_CORNER) && passes_through(up, v, straight_dot))
    {
      double d[3];
      arc_offset(up, v, f, d);
      for (int k = 0; k < 3; ++k) C[k] += d[k];
      n_arcs++;
    }
    if (!(obj->verts.rec[b].flags & MFTL_FLAG_CORNER) && passes_through(back, us_b, straight_dot))
    {
      const double us[3] = {v[0] + us_b[0], v[1] + us_b[1], v[2] + us_b[2]};  // s - a
      double d[3];
      arc_offset(us, v, f, d);
      for (int k = 0; k < 3; ++k) C[k] += d[k];
      n_arcs++;
    }
  }
  else
  {
    g_rim_fallbacks++;
  }

  if (n_arcs == 0)
    for (int k = 0; k < 3; ++k) C[k] = f * v[k];
  else if (n_arcs == 2)
    for (int k = 0; k < 3; ++k) C[k] *= 0.5;

  for (int k = 0; k < 3; ++k)
  {
    out[k] = pa[k] + C[k];

    /* Both ends on one domain wall: the point stays on it, exactly. */
    if (dom != MFTL_NULL && !dom->is_periodic[k] && pa[k] == pb[k])
    {
      const double tol = 1e-8 * dom->domain_size[k];
      if (fabs(pa[k] - dom->global_min[k]) <= tol || fabs(pa[k] - dom->global_max[k]) <= tol) out[k] = pa[k];
    }
  }
  if (dom != MFTL_NULL) MFTLApplyPeriodicity(dom, out);
  return ok;
}
