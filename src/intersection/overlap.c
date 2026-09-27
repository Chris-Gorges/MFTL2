/*
 * MFTL2 intersection — overlap-volume quantification (see include/mftl_overlap.h for
 * the pipeline description and the documented MFTL1 deviations).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_overlap.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_images.h"
#include "mftl_topology.h"

// MFTL1 overlap.c: near-degenerate in-plane gradient normal system.
#define MFTL_OVERLAP_GRAD_DEGEN 1.0e-8
// MFTL1 CounterClockwiseOrder: duplicate-point culling (relative to span).
#define MFTL_OVERLAP_DUP_POINT 5.0e-15

/* ── small vector helpers ── */
static inline void mftlOverlapSub3(double d[3], const double a[3], const double b[3])
{
  d[0] = a[0] - b[0];
  d[1] = a[1] - b[1];
  d[2] = a[2] - b[2];
}
static inline double mftlOverlapDot3(const double a[3], const double b[3]) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
static inline double mftlOverlapNorm3(const double a[3]) { return sqrt(mftlOverlapDot3(a, a)); }

/* ── one projected triangle ── */
typedef struct
{
  double P[3], Q[3];    // 2D projected vertex coords
  double h[3];          // out-of-plane heights x.w
  double centreP, centreQ, centreH;
  double GradP, GradQ;  // least-squares height-gradient over the 2D triangle
  double GlobalRadiusCurvature;
  double Span;  // max distance centre->vertex (2D)
} MFTL_ProjectedTri;

// Gram-Schmidt orthonormal frame from vecAB (MFTL1 GS_OrthoNormalise).
// The projection axis runs from A to B (w = +vecAB/|vecAB|): inside the
// overlap lens A's sheet lies ABOVE B's (h_A - h_B ~= 2R - |AB| > 0), so
// the prism integrand selects exactly the overlap region and every
// cross-pair (far caps) lands on the sign-rejected side. u, v = the first
// two coordinate axes well-separated from w, orthonormalized (MFTL1 picks
// rejection > 1/3).
static int mftlOverlapFrame(const double vecAB[3], double u[3], double v[3], double w[3])
{
  double n = mftlOverlapNorm3(vecAB);
  if (n * n <= 1.0e-20) return MFTL_ERROR;
  for (int k = 0; k < 3; ++k) w[k] = vecAB[k] / n;

  static const double axes[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  int found_u = 0;
  for (int a = 0; a < 3 && !found_u; ++a)
  {
    double d[3];
    double dot = mftlOverlapDot3(axes[a], w);
    for (int k = 0; k < 3; ++k) d[k] = axes[a][k] - dot * w[k];
    if (mftlOverlapDot3(d, d) > 1.0 / 3.0)
    {
      double dn = mftlOverlapNorm3(d);
      for (int k = 0; k < 3; ++k) u[k] = d[k] / dn;
      found_u = 1;
    }
  }
  if (!found_u) return MFTL_ERROR;  // unreachable for unit w

  // v = w x u (unit, completes the right-handed frame)
  v[0] = w[1] * u[2] - w[2] * u[1];
  v[1] = w[2] * u[0] - w[0] * u[2];
  v[2] = w[0] * u[1] - w[1] * u[0];
  return MFTL_SUCCESS;
}

// Project one triangle + fit its in-plane height gradient and osculating
// radius (MFTL1 GetProjectedTriangle / AddProjectedGrad / AddProjectedCurv /
// AddProjectedSpan).
static void mftlOverlapProjectTriData(const double pos_in[3][3], const double normal[3], const double u[3], const double v[3],
                                      const double w[3], MFTL_ProjectedTri* pt)
{
  pt->centreP = pt->centreQ = pt->centreH = 0.0;
  double pos[3][3];
  for (int i = 0; i < 3; ++i)
  {
    memcpy(pos[i], pos_in[i], 3 * sizeof(double));
    pt->P[i] = mftlOverlapDot3(pos[i], u);
    pt->Q[i] = mftlOverlapDot3(pos[i], v);
    pt->h[i] = mftlOverlapDot3(pos[i], w);
    pt->centreP += pt->P[i] / 3.0;
    pt->centreQ += pt->Q[i] / 3.0;
    pt->centreH += pt->h[i] / 3.0;
  }

  // Span (quick-reject radius)
  pt->Span = 0.0;
  for (int i = 0; i < 3; ++i)
  {
    double dr = pt->P[i] - pt->centreP, ds = pt->Q[i] - pt->centreQ;
    double d2 = dr * dr + ds * ds;
    if (d2 > pt->Span) pt->Span = d2;
  }
  pt->Span = sqrt(pt->Span);

  // Least-squares height gradient over the 2D triangle (2x2 normal system
  // about the projected centroid).
  double Err = 0.0, Ess = 0.0, Ers = 0.0, Epr = 0.0, Eps = 0.0;
  for (int i = 0; i < 3; ++i)
  {
    double r = pt->P[i] - pt->centreP, s = pt->Q[i] - pt->centreQ, dh = pt->h[i] - pt->centreH;
    Err += r * r;
    Ess += s * s;
    Ers += r * s;
    Epr += dh * r;
    Eps += dh * s;
  }
  double D = Err * Ess - Ers * Ers;
  if (D * D < MFTL_OVERLAP_GRAD_DEGEN * MFTL_OVERLAP_GRAD_DEGEN * MFTL_MAX(1.0, Err * Ess) * MFTL_MAX(1.0, Err * Ess) || fabs(D) < MFTL_VERYSMALL)
  {
    pt->GradP = pt->GradQ = 0.0;  // near-degenerate in-plane triangle
  }
  else
  {
    pt->GradP = (Ess * Epr - Ers * Eps) / D;
    pt->GradQ = (Err * Eps - Ers * Epr) / D;
  }

  // Osculating radius: paraboloid height ~ r^2/(2R), equal-weighted over
  // the three vertices (MFTL1 AddProjectedCurv sign convention).
  double dot_nw = mftlOverlapDot3(normal, w);
  double sign = (dot_nw <= 0.0) ? 1.0 : -1.0;
  double radius_sum = 0.0;
  int radius_n = 0;
  for (int i = 0; i < 3; ++i)
  {
    double dr = pt->P[i] - pt->centreP, ds = pt->Q[i] - pt->centreQ;
    double r2 = dr * dr + ds * ds;
    double denom = pt->h[i] * sign;
    if (fabs(denom) < MFTL_VERYSMALL) continue;  // vertex on the w-equator
    radius_sum += r2 / denom;
    radius_n++;
  }
  pt->GlobalRadiusCurvature = (radius_n > 0) ? fabs(radius_sum / (double) radius_n) : 0.0;
}

// Extrapolated sheet heights (along +w) at a 2D point.
static inline double mftlOverlapHeightA(const MFTL_ProjectedTri* A, double P, double Q)
{
  return A->centreH + A->GradP * (P - A->centreP) + A->GradQ * (Q - A->centreQ);
}
static inline double mftlOverlapHeightB(const MFTL_ProjectedTri* B, double P, double Q)
{
  return B->centreH + B->GradP * (P - B->centreP) + B->GradQ * (Q - B->centreQ);
}

/* ── 2D clipping (MFTL1 TessilatedOverlap, fan gate fixed) ── */

// MFTL1 TwoDim_Intersetion: segment AB vs segment PQ. Returns 1 and sets
// (Ix,Iy) on a single crossing inside both segments, 2 on collinear overlap
// (first endpoint), 0 on none/disjoint.
static int mftlTwoDimIntersection(double Ax, double Ay, double Bx, double By, double Px, double Py, double Qx, double Qy, double* Ix,
                                  double* Iy)
{
  double eps = 1.0e-2 * MFTL_REL_TOLERANCE * MFTL_REL_TOLERANCE * ((Ax - Bx) * (Ax - Bx) + (Ay - By) * (Ay - By) + (Px - Qx) * (Px - Qx) + (Py - Qy) * (Py - Qy));

  double Coef_I = (Ay - Py) * (Ax - Bx) - (Ax - Px) * (Ay - By);
  double Coef_J = (Ay - Py) * (Qx - Px) - (Ax - Px) * (Qy - Py);

  if (fabs(Coef_I - Coef_J) > eps) /* non-parallel */
  {
    double i, j;
    if (fabs(Coef_I) < fabs(Coef_J))
    {
      double ratio = -Coef_I / Coef_J; /* j = ratio*i */
      if (fabs(Ax - Px) >= fabs(Ay - Py)) i = (Ax - Px) / ((Ax - Bx) + ratio * (Qx - Px));
      else i = (Ay - Py) / ((Ay - By) + ratio * (Qy - Py));
      j = ratio * i;
    }
    else
    {
      double ratio = -Coef_J / Coef_I; /* i = ratio*j */
      if (fabs(Ax - Px) >= fabs(Ay - Py)) j = (Ax - Px) / ((Ax - Bx) * ratio + (Qx - Px));
      else j = (Ay - Py) / ((Ay - By) * ratio + (Qy - Py));
      i = ratio * j;
    }
    if ((0.0 <= i) && (1.0 >= i) && (0.0 <= j) && (1.0 >= j))
    {
      *Ix = Px + j * (Qx - Px);
      *Iy = Py + j * (Qy - Py);
      return 1;
    }
    return 0;
  }

  /* parallel: collinear? */
  if (fabs((Px - Bx) * (Ay - Qy) - (Py - By) * (Ax - Qx)) >= eps) return 0;

  /* Collinear overlap (MFTL1 TwoDim_Intersetion): the overlap endpoints are
   * always EXISTING segment endpoints — return the first one ("closest to P
   * first"). Only endpoint coordinates are ever returned; interpolating
   * along the dominant axis mixes coordinates of different endpoints and
   * injects off-edge garbage points into the clip polygon (observed: ~50%
   * area inflation for identical triangulations shifted along the projection
   * axis, where corresponding edges are exactly collinear). */
  {
    int A_Between = 0, B_Between = 0, P_Between = 0, Q_Between = 0;
    if (fabs((Bx - Ax) * (Qx - Px)) < fabs((By - Ay) * (Qy - Py)))
    {
      /* dy > dx: use y */
      if ((Ay <= Py && Ay >= Qy) || (Ay >= Py && Ay <= Qy)) A_Between = 1;
      if ((By <= Py && By >= Qy) || (By >= Py && By <= Qy)) B_Between = 1;
      if ((Py <= Ay && Py >= By) || (Py >= Ay && Py <= By)) P_Between = 1;
      if ((Qy <= Ay && Qy >= By) || (Qy >= Ay && Qy <= By)) Q_Between = 1;
    }
    else
    {
      /* dx >= dy: use x */
      if ((Ax <= Px && Ax >= Qx) || (Ax >= Px && Ax <= Qx)) A_Between = 1;
      if ((Bx <= Px && Bx >= Qx) || (Bx >= Px && Bx <= Qx)) B_Between = 1;
      if ((Px <= Ax && Px >= Bx) || (Px >= Ax && Px <= Bx)) P_Between = 1;
      if ((Qx <= Ax && Qx >= Bx) || (Qx >= Ax && Qx <= Bx)) Q_Between = 1;
    }

    if (P_Between && Q_Between)
    {
      *Ix = Px;
      *Iy = Py;
      return 2;
    }
    if (P_Between)
    {
      if (A_Between || B_Between)
      {
        *Ix = Px;
        *Iy = Py;
        return 2;
      }
      return 0; /* MFTL1: "can have this" (-1); treated as no corner */
    }
    if (Q_Between)
    {
      if (A_Between)
      {
        *Ix = Ax;
        *Iy = Ay;
        return 2;
      }
      if (B_Between)
      {
        *Ix = Bx;
        *Iy = By;
        return 2;
      }
      return 0; /* MFTL1: "can have this" (-1); treated as no corner */
    }
    if (A_Between && B_Between)
    {
      if ((By - Ay) * (Qy - Py) > 0.0)
      {
        *Ix = Ax;
        *Iy = Ay;
      }
      else
      {
        *Ix = Bx;
        *Iy = By;
      }
      return 2;
    }
    return 0; /* disjoint or the impossible A/B-only case */
  }
}

// Per-sub-prism integration (MFTL1 OverlapFromVertices with the exact
// zero-crossing fix). VP/VQ: 3 fan corners (2D). Fills out->Volume, the
// projected centre, the sheet areas and the smooth radius.
static void mftlOverlapFromVertices(const MFTL_ProjectedTri* A, const MFTL_ProjectedTri* B, double VP[3], double VQ[3],
                                    MFTL_Overlap* out)
{
  out->EffectiveRadiusOfCurvature = 1.0 / (1.0 / A->GlobalRadiusCurvature + 1.0 / B->GlobalRadiusCurvature);

  // RelHeight[v][0] = -hA(p), RelHeight[v][1] = -hB(p) (MFTL1 sign).
  double RelHeight[3][2];
  for (int v = 0; v < 3; ++v)
  {
    RelHeight[v][0] = -1.0 * mftlOverlapHeightA(A, VP[v], VQ[v]);
    RelHeight[v][1] = -1.0 * mftlOverlapHeightB(B, VP[v], VQ[v]);
  }

  // Projected fan area (Heron, strict radicand guard: degenerate -> empty).
  double e[3];
  e[0] = hypot(VP[1] - VP[0], VQ[1] - VQ[0]);
  e[1] = hypot(VP[2] - VP[1], VQ[2] - VQ[1]);
  e[2] = hypot(VP[0] - VP[2], VQ[0] - VQ[2]);
  double s = 0.5 * (e[0] + e[1] + e[2]);
  double sq = s * (s - e[0]) * (s - e[1]) * (s - e[2]);
  if (!(sq > 0.0))
  {
    out->Volume = 0.0;
    out->AreaTriA = out->AreaTriB = 0.0;
    out->RealCentreOfVolume[0] = (VP[0] + VP[1] + VP[2]) / 3.0;
    out->RealCentreOfVolume[1] = (VQ[0] + VQ[1] + VQ[2]) / 3.0;
    out->RealCentreOfVolume[2] = 0.0;
    return;
  }
  double area_tot = sqrt(sq);

  double volume_tot = (1.0 / 3.0) * area_tot * ((RelHeight[0][1] + RelHeight[1][1] + RelHeight[2][1]) - (RelHeight[0][0] + RelHeight[1][0] + RelHeight[2][0]));

  // Sheet areas: the fan lifted by each sheet's heights (Heron, non-strict
  // guard -> skewed subtriangles contribute 0, MFTL1 parity).
  double AreaRatio[2];
  for (int sheet = 0; sheet < 2; ++sheet)
  {
    double e3[3];
    for (int edge = 0; edge < 3; ++edge)
    {
      int a = edge, b = (edge + 1) % 3;
      double d0 = VP[b] - VP[a], d1 = VQ[b] - VQ[a], d2 = RelHeight[b][sheet] - RelHeight[a][sheet];
      e3[edge] = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    }
    double ss = 0.5 * (e3[0] + e3[1] + e3[2]);
    double sq3 = ss * (ss - e3[0]) * (ss - e3[1]) * (ss - e3[2]);
    double area3d = (0.0 <= sq3) ? sqrt(sq3) : 0.0;
    AreaRatio[sheet] = (area_tot > MFTL_VERYSMALL) ? area3d / area_tot : 0.0;
  }

  // Classify vertices by diff = hA - hB >= 0; `i` is the ODD ONE OUT
  // (positive for num_intersect == 1, negative for num_intersect == 2).
  double diff[3];
  int num_intersect = 0, i = -1;
  for (int v = 0; v < 3; ++v)
  {
    diff[v] = RelHeight[v][1] - RelHeight[v][0]; /* = hA - hB */
    if (diff[v] >= 0.0) num_intersect++;
    else i = v;
  }
  if (num_intersect == 1)
  {
    i = -1;
    for (int v = 0; v < 3; ++v)
      if (diff[v] >= 0.0) i = v;
  }

  double centreP = (VP[0] + VP[1] + VP[2]) / 3.0;
  double centreQ = (VQ[0] + VQ[1] + VQ[2]) / 3.0;
  double centreZ = (1.0 / 3.0) * (0.5 * (RelHeight[0][0] + RelHeight[0][1]) + 0.5 * (RelHeight[1][0] + RelHeight[1][1]) +
                                  0.5 * (RelHeight[2][0] + RelHeight[2][1]));

  if (num_intersect == 3)
  {
    out->Volume = volume_tot;
    out->AreaTriA = area_tot * AreaRatio[0];
    out->AreaTriB = area_tot * AreaRatio[1];
    out->RealCentreOfVolume[0] = centreP;
    out->RealCentreOfVolume[1] = centreQ;
    out->RealCentreOfVolume[2] = centreZ;
    return;
  }

  if (num_intersect == 0)
  {
    out->Volume = 0.0;
    out->AreaTriA = out->AreaTriB = 0.0;
    out->RealCentreOfVolume[0] = centreP;
    out->RealCentreOfVolume[1] = centreQ;
    out->RealCentreOfVolume[2] = centreZ;
    return;
  }

  // 1 or 2 intersections: exact linear zero-crossings on the two edges from i.
  int x = (i + 1) % 3, y = (i + 2) % 3;
  double t_j = diff[i] / (diff[i] - diff[y]); /* crossing on edge i->y */
  double t_k = diff[i] / (diff[i] - diff[x]); /* crossing on edge i->x */

  double jP = VP[i] + t_j * (VP[y] - VP[i]), jQ = VQ[i] + t_j * (VQ[y] - VQ[i]);
  double kP = VP[i] + t_k * (VP[x] - VP[i]), kQ = VQ[i] + t_k * (VQ[x] - VQ[i]);
  double h_j = RelHeight[i][0] + t_j * (RelHeight[y][0] - RelHeight[i][0]);
  double h_k = RelHeight[i][0] + t_k * (RelHeight[x][0] - RelHeight[i][0]);

  // Sub-prism near i (triangle i, j, k).
  double se[3];
  se[0] = hypot(jP - VP[i], jQ - VQ[i]);
  se[1] = hypot(kP - jP, kQ - jQ);
  se[2] = hypot(VP[i] - kP, VQ[i] - kQ);
  double ss2 = 0.5 * (se[0] + se[1] + se[2]);
  double sq2 = ss2 * (ss2 - se[0]) * (ss2 - se[1]) * (ss2 - se[2]);
  double area_sub = (0.0 <= sq2) ? sqrt(sq2) : 0.0;
  double volume_sub = (1.0 / 3.0) * area_sub * diff[i];
  double centre_subP = (VP[i] + jP + kP) / 3.0;
  double centre_subQ = (VQ[i] + jQ + kQ) / 3.0;
  double centre_subZ = (1.0 / 3.0) * (0.5 * (RelHeight[i][0] + RelHeight[i][1]) + h_j + h_k);

  double volume, centre_vP, centre_vQ, centre_vZ;
  if (num_intersect == 1)
  {
    volume = volume_sub;
    centre_vP = centre_subP;
    centre_vQ = centre_subQ;
    centre_vZ = centre_subZ;
  }
  else
  {
    volume = volume_tot - volume_sub; /* volume_sub < 0 here */
    double denom = volume_tot - volume_sub;
    if (fabs(denom) < MFTL_VERYSMALL)
    {
      volume = 0.0;
      centre_vP = centreP;
      centre_vQ = centreQ;
      centre_vZ = centreZ;
    }
    else
    {
      centre_vP = (centreP * volume_tot - centre_subP * volume_sub) / denom;
      centre_vQ = (centreQ * volume_tot - centre_subQ * volume_sub) / denom;
      centre_vZ = (centreZ * volume_tot - centre_subZ * volume_sub) / denom;
    }
  }

  out->Volume = volume;
  out->AreaTriA = ((num_intersect == 1) ? area_sub : (area_tot - area_sub)) * AreaRatio[0];
  out->AreaTriB = ((num_intersect == 1) ? area_sub : (area_tot - area_sub)) * AreaRatio[1];
  out->RealCentreOfVolume[0] = centre_vP;
  out->RealCentreOfVolume[1] = centre_vQ;
  out->RealCentreOfVolume[2] = centre_vZ;
}

// Clip two projected triangles and integrate the prism volume over the
// clipped polygon (MFTL1 TessilatedOverlap, fan gate fixed for arbitrary
// corner counts). Returns the number of non-empty sub-prisms integrated.
static int mftlTessilatedOverlap(const MFTL_ProjectedTri* A, const MFTL_ProjectedTri* B, MFTL_Overlap* out)
{
  out->Volume = 0.0;
  out->AreaTriA = out->AreaTriB = 0.0;
  out->EffectiveRadiusOfCurvature = 0.0;

  // Quick reject: the projected triangles cannot overlap.
  {
    double dP = A->centreP - B->centreP, dQ = A->centreQ - B->centreQ;
    double span = A->Span + B->Span;
    if (span * span < dP * dP + dQ * dQ) return 0;
  }

  // Edge normals, midpoints and inward orientation (MFTL1).
  double edge_norm[2][3][2], edge_centre[2][3][2], dot_opp[2][3];
  double length_scale = 0.0;
  const MFTL_ProjectedTri* tris[2] = {A, B};
  for (int t = 0; t < 2; ++t)
  {
    for (int e = 0; e < 3; ++e)
    {
      int p = (e + 2) % 3, q = (e + 1) % 3;
      edge_norm[t][e][0] = tris[t]->Q[p] - tris[t]->Q[q];
      edge_norm[t][e][1] = tris[t]->P[q] - tris[t]->P[p];
      edge_centre[t][e][0] = 0.5 * (tris[t]->P[p] + tris[t]->P[q]);
      edge_centre[t][e][1] = 0.5 * (tris[t]->Q[p] + tris[t]->Q[q]);
      double dx = tris[t]->P[e] - edge_centre[t][e][0], dy = tris[t]->Q[e] - edge_centre[t][e][1];
      dot_opp[t][e] = edge_norm[t][e][0] * dx + edge_norm[t][e][1] * dy;
      length_scale += fabs(dot_opp[t][e]);
    }
  }
  double slack = MFTL_REL_TOLERANCE * MFTL_REL_TOLERANCE * length_scale * length_scale;

  // Interior vertices (inside ALL half-planes of the other triangle).
  double cornersP[16], cornersQ[16];
  int nc = 0;
  for (int t = 0; t < 2; ++t)
  {
    const MFTL_ProjectedTri* self = tris[t];
    const MFTL_ProjectedTri* other = tris[1 - t];
    for (int vv = 0; vv < 3; ++vv)
    {
      double testP = self->P[vv], testQ = self->Q[vv];
      int is_out = 0;
      for (int e = 0; e < 3; ++e)
      {
        double dot_tst = edge_norm[1 - t][e][0] * (testP - edge_centre[1 - t][e][0]) + edge_norm[1 - t][e][1] * (testQ - edge_centre[1 - t][e][1]);
        if (0.0 > dot_tst * dot_opp[1 - t][e] + slack) is_out = 1;
      }
      if (!is_out && nc < 16)
      {
        cornersP[nc] = testP;
        cornersQ[nc] = testQ;
        nc++;
      }
    }
  }

  // Edge-edge intersections.
  for (int ea = 0; ea < 3; ++ea)
  {
    int pa = (ea + 2) % 3, qa = (ea + 1) % 3;
    double Ax = A->P[qa], Ay = A->Q[qa], Bx = A->P[pa], By = A->Q[pa];
    for (int eb = 0; eb < 3; ++eb)
    {
      int pb = (eb + 2) % 3, qb = (eb + 1) % 3;
      double Px = B->P[qb], Py = B->Q[qb], Qx = B->P[pb], Qy = B->Q[pb];
      double Ix = 0.0, Iy = 0.0;
      int rc = mftlTwoDimIntersection(Ax, Ay, Bx, By, Px, Py, Qx, Qy, &Ix, &Iy);
      if ((rc == 1 || rc == 2) && nc < 16)
      {
        cornersP[nc] = Ix;
        cornersQ[nc] = Iy;
        nc++;
      }
    }
  }

  if (nc < 3) return 0;

  // Cull near-duplicate points (MFTL1 CounterClockwiseOrder culling).
  double cx = 0.0, cy = 0.0;
  for (int c = 0; c < nc; ++c)
  {
    cx += cornersP[c];
    cy += cornersQ[c];
  }
  cx /= (double) nc;
  cy /= (double) nc;
  double span2 = 0.0;
  for (int c = 0; c < nc; ++c)
  {
    double dx = cornersP[c] - cx, dy = cornersQ[c] - cy;
    double d2 = dx * dx + dy * dy;
    if (d2 > span2) span2 = d2;
  }
  double dup_eps = MFTL_OVERLAP_DUP_POINT * span2;
  double ordP[16], ordQ[16];
  int no = 0;
  for (int c = 0; c < nc; ++c)
  {
    int dup = 0;
    for (int d = 0; d < no; ++d)
    {
      double dx = ordP[d] - cornersP[c], dy = ordQ[d] - cornersQ[c];
      if (dx * dx + dy * dy <= dup_eps) dup = 1;
    }
    if (!dup)
    {
      ordP[no] = cornersP[c];
      ordQ[no] = cornersQ[c];
      no++;
    }
  }
  if (no < 3) return 0;

  // CCW ordering by polar angle about the centroid (replaces MFTL1's
  // quadrant scan; identical result for a simple polygon).
  {
    int order[16];
    double ang[16];
    for (int c = 0; c < no; ++c)
    {
      order[c] = c;
      ang[c] = atan2(ordQ[c] - cy, ordP[c] - cx);
    }
    for (int a = 0; a < no; ++a)
      for (int b = a + 1; b < no; ++b)
        if (ang[order[b]] < ang[order[a]])
        {
          int tmp = order[a];
          order[a] = order[b];
          order[b] = tmp;
        }
    double srtP[16], srtQ[16];
    for (int c = 0; c < no; ++c)
    {
      srtP[c] = ordP[order[c]];
      srtQ[c] = ordQ[order[c]];
    }
    memcpy(ordP, srtP, (size_t) no * sizeof(double));
    memcpy(ordQ, srtQ, (size_t) no * sizeof(double));
  }

  // Ensure the winding is CCW (positive signed area); flip if needed so the
  // fan integration and centre of volume are orientation-consistent.
  {
    double a = 0.0;
    for (int c = 0; c < no; ++c)
    {
      int d = (c + 1) % no;
      a += ordP[c] * ordQ[d] - ordQ[c] * ordP[d];
    }
    if (a < 0.0)
      for (int c = 0; c < no / 2; ++c)
      {
        double t1 = ordP[c], t2 = ordQ[c];
        ordP[c] = ordP[no - 1 - c];
        ordQ[c] = ordQ[no - 1 - c];
        ordP[no - 1 - c] = t1;
        ordQ[no - 1 - c] = t2;
      }
  }

  // Fan-triangulate from the polygon centroid (MFTL1 gate `3 >= n` fixed:
  // any n >= 3 contributes).
  double cenP = 0.0, cenQ = 0.0;
  for (int c = 0; c < no; ++c)
  {
    cenP += ordP[c];
    cenQ += ordQ[c];
  }
  cenP /= (double) no;
  cenQ /= (double) no;

  double vol_sum = 0.0, wa_sum = 0.0, wb_sum = 0.0;
  double cvP = 0.0, cvQ = 0.0, cvZ = 0.0;
  double radius_smooth = 0.0;
  int nsub = 0;
  for (int c = 0; c < no; ++c)
  {
    int d = (c + 1) % no;
    double VP[3] = {cenP, ordP[c], ordP[d]};
    double VQ[3] = {cenQ, ordQ[c], ordQ[d]};
    MFTL_Overlap sub;
    mftlOverlapFromVertices(A, B, VP, VQ, &sub);
    radius_smooth = sub.EffectiveRadiusOfCurvature;
    if (fabs(sub.Volume) > MFTL_VERYSMALL || sub.AreaTriA > MFTL_VERYSMALL || sub.AreaTriB > MFTL_VERYSMALL)
    {
      vol_sum += sub.Volume;
      wa_sum += sub.AreaTriA;
      wb_sum += sub.AreaTriB;
      cvP += sub.RealCentreOfVolume[0] * sub.Volume;
      cvQ += sub.RealCentreOfVolume[1] * sub.Volume;
      cvZ += sub.RealCentreOfVolume[2] * sub.Volume;
      nsub++;
    }
  }

  out->Volume = vol_sum;
  out->AreaTriA = wa_sum;
  out->AreaTriB = wb_sum;
  out->EffectiveRadiusOfCurvature = radius_smooth;
  if (fabs(vol_sum) > MFTL_VERYSMALL)
  {
    out->RealCentreOfVolume[0] = cvP / vol_sum;
    out->RealCentreOfVolume[1] = cvQ / vol_sum;
    out->RealCentreOfVolume[2] = cvZ / vol_sum;
  }
  else
  {
    out->RealCentreOfVolume[0] = cenP;
    out->RealCentreOfVolume[1] = cenQ;
    out->RealCentreOfVolume[2] = 0.0;
  }
  return nsub;
}

// Record-based core: the overlap pipeline from raw triangle geometry
// (positions + outward normal + gids). Used by the public object API and by
// the ensemble driver on gathered candidate records.
static int mftlOverlapCompute(const double posA[3][3], const double nA[3], int gidA, const double posB[3][3], const double nB[3], int gidB,
                              const double vecAB[3], MFTL_Overlap* out)
{
  memset(out, 0, sizeof(*out));

  double u[3], v[3], w[3];
  if (mftlOverlapFrame(vecAB, u, v, w) != MFTL_SUCCESS) return 0;

  MFTL_ProjectedTri PA, PB;
  mftlOverlapProjectTriData(posA, nA, u, v, w, &PA);
  mftlOverlapProjectTriData(posB, nB, u, v, w, &PB);

  int nsub = mftlTessilatedOverlap(&PA, &PB, out);
  if (nsub <= 0) return 0;

  // Real-space centre of volume and surface normal (MFTL1 FinaliseOverlap).
  // Note: the projected centre's third component is in RelHeight = -h units
  // (MFTL1 sign convention), so the real w-coordinate is its NEGATIVE.
  double P = out->RealCentreOfVolume[0], Q = out->RealCentreOfVolume[1], OP = out->RealCentreOfVolume[2];
  out->RealCentreOfVolume[0] = P * u[0] + Q * v[0] - OP * w[0];
  out->RealCentreOfVolume[1] = P * u[1] + Q * v[1] - OP * w[1];
  out->RealCentreOfVolume[2] = P * u[2] + Q * v[2] - OP * w[2];

  for (int k = 0; k < 3; ++k) out->AverageSurfaceNorm[k] = 0.5 * (nA[k] - nB[k]);
  out->ID_A = gidA;
  out->ID_B = gidB;
  out->id = -1;  // assigned deterministically by the ensemble driver
  return nsub;
}

int MFTLCalculateOverlap(MFTL_Object* objA, int triA, MFTL_Object* objB, int triB, const double vecAB[3], MFTL_Overlap* out)
{
  double posA[3][3], posB[3][3], nA[3], nB[3];
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(objA, &objA->triangles[triA], vv);
  for (int i = 0; i < 3; ++i) memcpy(posA[i], vv[i]->pos, 3 * sizeof(double));
  memcpy(nA, objA->triangles[triA].normal, 3 * sizeof(double));
  MFTLGetTriangleVertices(objB, &objB->triangles[triB], vv);
  for (int i = 0; i < 3; ++i) memcpy(posB[i], vv[i]->pos, 3 * sizeof(double));
  memcpy(nB, objB->triangles[triB].normal, 3 * sizeof(double));
  return mftlOverlapCompute(posA, nA, objA->triangles[triA].global_id, posB, nB, objB->triangles[triB].global_id, vecAB, out);
}

/* ══════════════════════════════════════════════════════════════════════
 * Ensemble driver
 * ══════════════════════════════════════════════════════════════════════ */

typedef struct
{
  const double* vecAB;
  int my_rank;
  MFTL_Overlap* records;
  int count;
  int capacity;
} MFTL_OverlapAccum;

static void mftlOverlapCallback(MFTL_Object* objA, int triA, MFTL_Object* objB, int triB, void* user_data)
{
  MFTL_OverlapAccum* acc = (MFTL_OverlapAccum*) user_data;
  // Each pair is computed exactly once: on the rank OWNING triangle A, with
  // the ghost copy of B that CreateSpatialGhosts installed.
  if (objA->triangles[triA].is_ghost || objA->triangles[triA].owner_proc != acc->my_rank) return;

  MFTL_Overlap rec;
  if (MFTLCalculateOverlap(objA, triA, objB, triB, acc->vecAB, &rec) <= 0) return;
  if (acc->count >= acc->capacity)
  {
    acc->capacity = acc->capacity > 0 ? acc->capacity * 2 : 64;
    acc->records = (MFTL_Overlap*) realloc(acc->records, (size_t) acc->capacity * sizeof(MFTL_Overlap));
  }
  acc->records[acc->count++] = rec;
}

// One gathered candidate triangle: positions + outward normal + gid
// (MFTL1 MFTL_MPI_Triangle, reduced to what the overlap pipeline needs).
typedef struct
{
  double pos[3][3];
  double normal[3];
  int gid;
} MFTL_OverlapCandidate;

// Per-vertex transform into the common unwrapped frame of the pair query
// (MFTL1 minimum-image handling): unwrap about the body's own surface
// centre, then apply the additive shift (for body B: the minimum image of
// cmB - cmA). Applied to PACKED CANDIDATE DATA only — the mesh itself is
// never mutated, so ghost/ownership state cannot become inconsistent.
// Requires body extent << period (same class as MFTL1).
static void mftlOverlapTransformVertex(MFTL_Domain* dom, const double cm[3], const double shift[3], const double in[3], double out[3])
{
  for (int k = 0; k < 3; ++k)
  {
    out[k] = in[k];
    if (!dom->is_periodic[k]) continue;
    double L = dom->domain_size[k];
    out[k] = in[k] - L * floor((in[k] - cm[k]) / L + 0.5) + shift[k];
  }
}

// Pack this rank's OWNED candidate triangles whose (transformed) AABB
// overlaps the other body's (transformed) big box (MFTL1
// MFTL_BodyOverlap_OneSide filter).
static int mftlPackCandidates(MFTL_Object* obj, const double cm[3], const double shift[3], const double bigMin[3], const double bigMax[3],
                              MFTL_OverlapCandidate** out)
{
  MFTL_Domain* dom = obj->domain;
  int n = 0, cap = 0;
  MFTL_OverlapCandidate* list = NULL;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || (tri->flags & MFTL_FLAG_DEAD)) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double pos[3][3];
    double tmin[3] = {1e30, 1e30, 1e30}, tmax[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < 3; ++i)
    {
      mftlOverlapTransformVertex(dom, cm, shift, vv[i]->pos, pos[i]);
      for (int k = 0; k < 3; ++k)
      {
        if (pos[i][k] < tmin[k]) tmin[k] = pos[i][k];
        if (pos[i][k] > tmax[k]) tmax[k] = pos[i][k];
      }
    }
    int hit = 1;
    for (int k = 0; k < 3; ++k)
      if (tmax[k] < bigMin[k] || tmin[k] > bigMax[k]) hit = 0;
    if (!hit) continue;

    if (n >= cap)
    {
      cap = cap > 0 ? cap * 2 : 256;
      list = (MFTL_OverlapCandidate*) realloc(list, (size_t) cap * sizeof(MFTL_OverlapCandidate));
    }
    for (int i = 0; i < 3; ++i) memcpy(list[n].pos[i], pos[i], 3 * sizeof(double));
    memcpy(list[n].normal, tri->normal, 3 * sizeof(double));
    list[n].gid = tri->global_id;
    n++;
  }
  *out = list;
  return n;
}

// Allgatherv a candidate list so every rank holds the complete set.
static int mftlGatherCandidates(MFTL_Domain* dom, MFTL_OverlapCandidate* local, int n_local, MFTL_OverlapCandidate** out_all)
{
  int nproc = dom->nproc;
  int* counts = (int*) malloc((size_t) nproc * sizeof(int));
  int* displs = (int*) malloc((size_t) (nproc + 1) * sizeof(int));
  if (nproc > 1)
    MPI_Allgather(&n_local, 1, MPI_INT, counts, 1, MPI_INT, dom->comm_cart);
  else
    counts[0] = n_local;

  displs[0] = 0;
  int total = 0;
  for (int r = 0; r < nproc; ++r)
  {
    total += counts[r];
    displs[r + 1] = displs[r] + counts[r];
  }

  MFTL_OverlapCandidate* all = NULL;
  if (total > 0) all = (MFTL_OverlapCandidate*) malloc((size_t) total * sizeof(MFTL_OverlapCandidate));
  if (nproc > 1)
  {
    // MPI_BYTE transport: counts and displacements must be in BYTES.
    int* bcounts = (int*) malloc((size_t) nproc * sizeof(int));
    int* bdispls = (int*) malloc((size_t) nproc * sizeof(int));
    for (int r = 0; r < nproc; ++r)
    {
      bcounts[r] = counts[r] * (int) sizeof(MFTL_OverlapCandidate);
      bdispls[r] = displs[r] * (int) sizeof(MFTL_OverlapCandidate);
    }
    MPI_Allgatherv(local, (int) ((size_t) n_local * sizeof(MFTL_OverlapCandidate)), MPI_BYTE, all, bcounts, bdispls, MPI_BYTE, dom->comm_cart);
    free(bcounts);
    free(bdispls);
  }
  else if (total > 0)
    memcpy(all, local, (size_t) total * sizeof(MFTL_OverlapCandidate));

  free(counts);
  free(displs);
  *out_all = all;
  return total;
}

// Sum the per-body overlap records of one object pair.
//
// Candidate selection mirrors MFTL1 MFTL_BodyOverlap: owned triangles whose
// AABB meets the other body's big box are gathered to every rank (the
// "brute force" exchange accepted in plans/11), because the
// volume prism spans triangle pairs whose SHEETS are far apart in 3D but
// overlap in the projection plane — a proximity/BVH or 26-neighbour ghost
// filter cannot see those pairs (verified: only sliver volumes result).
//
// Periodic handling: candidate positions are packed in a COMMON UNWRAPPED
// FRAME (unwrap each body about its own surface centre, then shift B by the
// minimum image of cmB - cmA). Positions are replaced only temporarily and restored, so ghost
// and ownership state stay consistent on any rank count.
static int mftlOverlapObjectPair(MFTL_Object* objA, MFTL_Object* objB, MFTL_Overlap** out_records, int* out_count)
{
  MFTL_Domain* dom = objA->domain;

  // Body separation for the projection frame (minimum image -> periodic).
  double cmA[3], cmB[3];
  MFTLSurfaceCenter(objA, cmA);
  MFTLSurfaceCenter(objB, cmB);
  double vecAB[3];
  MFTLGetMinimumImageVector(dom, vecAB, cmA, cmB);

  MFTLComputeTriangleProperties(objA);
  MFTLComputeTriangleProperties(objB);

  // Per-body packing transforms: A is only unwrapped; B is unwrapped and
  // shifted by the minimum image of (cmB - cmA).
  double zero3[3] = {0.0, 0.0, 0.0};
  double imgB[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k)
  {
    if (!dom->is_periodic[k]) continue;
    double L = dom->domain_size[k];
    imgB[k] = -L * floor((cmB[k] - cmA[k]) / L + 0.5);
  }

  // Big boxes over TRANSFORMED positions (owned geometry, globally reduced).
  double bbAmin[3] = {1e30, 1e30, 1e30}, bbAmax[3] = {-1e30, -1e30, -1e30};
  double bbBmin[3] = {1e30, 1e30, 1e30}, bbBmax[3] = {-1e30, -1e30, -1e30};
  for (int t = 0; t < objA->n_triangles; ++t)
  {
    if (objA->triangles[t].is_ghost) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(objA, &objA->triangles[t], vv);
    for (int i = 0; i < 3; ++i)
    {
      double p[3];
      mftlOverlapTransformVertex(dom, cmA, zero3, vv[i]->pos, p);
      for (int k = 0; k < 3; ++k)
      {
        if (p[k] < bbAmin[k]) bbAmin[k] = p[k];
        if (p[k] > bbAmax[k]) bbAmax[k] = p[k];
      }
    }
  }
  for (int t = 0; t < objB->n_triangles; ++t)
  {
    if (objB->triangles[t].is_ghost) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(objB, &objB->triangles[t], vv);
    for (int i = 0; i < 3; ++i)
    {
      double p[3];
      mftlOverlapTransformVertex(dom, cmB, imgB, vv[i]->pos, p);
      for (int k = 0; k < 3; ++k)
      {
        if (p[k] < bbBmin[k]) bbBmin[k] = p[k];
        if (p[k] > bbBmax[k]) bbBmax[k] = p[k];
      }
    }
  }
  double red[3];
  MPI_Allreduce(bbAmin, red, 3, MPI_DOUBLE, MPI_MIN, dom->comm_cart);
  memcpy(bbAmin, red, 3 * sizeof(double));
  MPI_Allreduce(bbAmax, red, 3, MPI_DOUBLE, MPI_MAX, dom->comm_cart);
  memcpy(bbAmax, red, 3 * sizeof(double));
  MPI_Allreduce(bbBmin, red, 3, MPI_DOUBLE, MPI_MIN, dom->comm_cart);
  memcpy(bbBmin, red, 3 * sizeof(double));
  MPI_Allreduce(bbBmax, red, 3, MPI_DOUBLE, MPI_MAX, dom->comm_cart);
  memcpy(bbBmax, red, 3 * sizeof(double));

  // Candidates + brute-force gather (every rank ends with both full sets).
  MFTL_OverlapCandidate* mineA = NULL;
  int mineA_n = mftlPackCandidates(objA, cmA, zero3, bbBmin, bbBmax, &mineA);
  MFTL_OverlapCandidate* candA = NULL;
  int candA_n = mftlGatherCandidates(dom, mineA, mineA_n, &candA);

  MFTL_OverlapCandidate* mineB = NULL;
  int mineB_n = mftlPackCandidates(objB, cmB, imgB, bbAmin, bbAmax, &mineB);
  MFTL_OverlapCandidate* candB = NULL;
  int candB_n = mftlGatherCandidates(dom, mineB, mineB_n, &candB);

  MFTL_Overlap* records = NULL;
  int nrec = 0, reccap = 0;

  double u[3], v[3], w[3];
  if (mftlOverlapFrame(vecAB, u, v, w) == MFTL_SUCCESS && candA_n > 0 && candB_n > 0)
  {
    // Project the B candidates once; per-B 2D AABB + centre + Span for the
    // cheap reject (the full per-pair projection runs only for survivors).
    double* bminP = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bminQ = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bmaxP = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bmaxQ = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bcenP = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bcenQ = (double*) malloc((size_t) candB_n * sizeof(double));
    double* bspan = (double*) malloc((size_t) candB_n * sizeof(double));
    for (int j = 0; j < candB_n; ++j)
    {
      double p0 = mftlOverlapDot3(candB[j].pos[0], u), q0 = mftlOverlapDot3(candB[j].pos[0], v);
      double p1 = mftlOverlapDot3(candB[j].pos[1], u), q1 = mftlOverlapDot3(candB[j].pos[1], v);
      double p2 = mftlOverlapDot3(candB[j].pos[2], u), q2 = mftlOverlapDot3(candB[j].pos[2], v);
      bminP[j] = MFTL_MIN(p0, MFTL_MIN(p1, p2));
      bmaxP[j] = MFTL_MAX(p0, MFTL_MAX(p1, p2));
      bminQ[j] = MFTL_MIN(q0, MFTL_MIN(q1, q2));
      bmaxQ[j] = MFTL_MAX(q0, MFTL_MAX(q1, q2));
      bcenP[j] = (p0 + p1 + p2) / 3.0;
      bcenQ[j] = (q0 + q1 + q2) / 3.0;
      bspan[j] = 0.0;
      bspan[j] = MFTL_MAX(bspan[j], hypot(p0 - bcenP[j], q0 - bcenQ[j]));
      bspan[j] = MFTL_MAX(bspan[j], hypot(p1 - bcenP[j], q1 - bcenQ[j]));
      bspan[j] = MFTL_MAX(bspan[j], hypot(p2 - bcenP[j], q2 - bcenQ[j]));
    }

    // This rank computes the pairs for the candidates IT owned (each pair
    // exactly once globally).
    for (int ia = 0; ia < mineA_n; ++ia)
    {
      double ap[3], aq[3];
      double acp = 0.0, acq = 0.0, aspan = 0.0;
      for (int i = 0; i < 3; ++i)
      {
        ap[i] = mftlOverlapDot3(mineA[ia].pos[i], u);
        aq[i] = mftlOverlapDot3(mineA[ia].pos[i], v);
      }
      acp = (ap[0] + ap[1] + ap[2]) / 3.0;
      acq = (aq[0] + aq[1] + aq[2]) / 3.0;
      double aminP = MFTL_MIN(ap[0], MFTL_MIN(ap[1], ap[2])), amaxP = MFTL_MAX(ap[0], MFTL_MAX(ap[1], ap[2]));
      double aminQ = MFTL_MIN(aq[0], MFTL_MIN(aq[1], aq[2])), amaxQ = MFTL_MAX(aq[0], MFTL_MAX(aq[1], aq[2]));
      aspan = MFTL_MAX(aspan, hypot(ap[0] - acp, aq[0] - acq));
      aspan = MFTL_MAX(aspan, hypot(ap[1] - acp, aq[1] - acq));
      aspan = MFTL_MAX(aspan, hypot(ap[2] - acp, aq[2] - acq));

      for (int jb = 0; jb < candB_n; ++jb)
      {
        // 2D AABB reject, then the span reject (MFTL1 quick reject).
        if (amaxP < bminP[jb] || aminP > bmaxP[jb] || amaxQ < bminQ[jb] || aminQ > bmaxQ[jb]) continue;
        double dP = acp - bcenP[jb], dQ = acq - bcenQ[jb];
        double span = aspan + bspan[jb];
        if (span * span < dP * dP + dQ * dQ) continue;

        MFTL_Overlap ov;
        if (mftlOverlapCompute(mineA[ia].pos, mineA[ia].normal, mineA[ia].gid, candB[jb].pos, candB[jb].normal, candB[jb].gid, vecAB, &ov) <= 0)
          continue;
        if (nrec >= reccap)
        {
          reccap = reccap > 0 ? reccap * 2 : 256;
          records = (MFTL_Overlap*) realloc(records, (size_t) reccap * sizeof(MFTL_Overlap));
        }
        records[nrec++] = ov;
      }
    }

    free(bminP);
    free(bminQ);
    free(bmaxP);
    free(bmaxQ);
    free(bcenP);
    free(bcenQ);
    free(bspan);
  }

  free(mineA);
  free(candA);
  free(mineB);
  free(candB);
  *out_records = records;
  *out_count = nrec;
  return MFTL_SUCCESS;
}

int MFTLEnsembleCalculateOverlap(MFTL_Ensemble* ensA, MFTL_Ensemble* ensB, MFTL_Overlap** out_records, int* out_count)
{
  if (out_records) *out_records = NULL;
  if (out_count) *out_count = 0;
  if (!ensA || ensA->num_objects == 0) return MFTL_ERROR;

  MFTL_Domain* dom = &ensA->domain;
  int nobjA = ensA->num_objects;
  int nobjB = ensB ? ensB->num_objects : nobjA;  // self-intersection pairs within ensA

  // The overlap query unwraps each body about its surface centre, which exists only
  // for bodies compact along every periodic axis. A wrapping body is an explicit error here
  // (collective: the flags are identical on all ranks), not an abort inside MFTLSurfaceCenter.
  for (int e = 0; e < (ensB ? 2 : 1); ++e)
  {
    MFTL_Ensemble* en = e ? ensB : ensA;
    for (int o = 0; o < en->num_objects; ++o)
    {
      MFTLObjectEnsureWinding(en->objects[o]);
      if (!MFTLObjectIsCompact(en->objects[o]))
      {
        if (dom->rank == 0)
          fprintf(stderr, "MFTLEnsembleCalculateOverlap: body %d wraps a periodic direction; the overlap query requires bodies compact along every periodic axis\n",
                  en->objects[o]->global_id);
        return MFTL_ERROR;
      }
    }
  }

  MFTL_Overlap* mine = NULL;
  int mine_n = 0, mine_cap = 0;

  // Object-pair loop (i<j for self-intersection, matching the detection
  // driver). Bodies are remeshed-free here: this is a pure query.
  for (int i = 0; i < nobjA; ++i)
  {
    int j_start = ensB ? 0 : i + 1;
    for (int j = j_start; j < nobjB; ++j)
    {
      MFTL_Object* objA = ensA->objects[i];
      MFTL_Object* objB = ensB ? ensB->objects[j] : ensA->objects[j];
      if (!ensB && j == i) continue;

      // Periodic: place BOTH bodies in a common unwrapped frame before the
      // query. Shifting B alone is not enough when A itself straddles the
      // seam: A's wrapped vertices would then sit a full period away from
      // the shifted B and contribute spurious cross-period prisms.
      //
      //   vA' = cmA + wrap(vA - cmA)          (unwrap A about its centre)
      //   vB' = cmB + wrap(vB - cmB) + img    (unwrap B, then min-image shift)
      //   img = -L*round((cmB - cmA)/L)
      //
      // The candidate gather reads OWNED geometry only, so no ghost refresh
      // is needed inside the window; positions are restored exactly (saved
      // copy) and ghosts refreshed once afterwards. Bodies must be smaller
      // than the period in each periodic direction (same restriction class
      // as MFTL1's minimum-image handling).
      double cmA[3], cmB[3];
      MFTLSurfaceCenter(objA, cmA);
      MFTLSurfaceCenter(objB, cmB);
      int any_periodic = dom->is_periodic[0] || dom->is_periodic[1] || dom->is_periodic[2];

      double* saveA = NULL;
      int *imgA = NULL, *imgB = NULL;  // saved image counters
      double* saveB = NULL;
      int nA_save = 0, nB_save = 0;
      if (any_periodic)
      {
        double img[3] = {0.0, 0.0, 0.0};
        for (int k = 0; k < 3; ++k)
        {
          if (!dom->is_periodic[k]) continue;
          double L = dom->domain_size[k];
          img[k] = -L * floor((cmB[k] - cmA[k]) / L + 0.5);
        }
        for (int vi = 0; vi < objA->n_vertices; ++vi)
          nA_save++;  // ghosts too (the counters of every copy are held at 0 below)
        for (int vi = 0; vi < objB->n_vertices; ++vi)
          nB_save++;
        saveA = (double*) malloc((size_t) (nA_save > 0 ? nA_save : 1) * 3 * sizeof(double));
        saveB = (double*) malloc((size_t) (nB_save > 0 ? nB_save : 1) * 3 * sizeof(double));
        imgA = (int*) malloc((size_t) (nA_save > 0 ? nA_save : 1) * 3 * sizeof(int));
        imgB = (int*) malloc((size_t) (nB_save > 0 ? nB_save : 1) * 3 * sizeof(int));
        int s = 0;
        for (int vi = 0; vi < objA->n_vertices; ++vi)
        {
          MFTL_Vertex* vtx = &objA->vertices[vi];
          memcpy(saveA + 3 * s, vtx->pos, 3 * sizeof(double));
          memcpy(imgA + 3 * s, vtx->img, 3 * sizeof(int));  // the temporary positions are
          vtx->img[0] = vtx->img[1] = vtx->img[2] = 0;      // continuous; counters restored below
          s++;
          for (int k = 0; k < 3; ++k)
          {
            if (!dom->is_periodic[k]) continue;
            double L = dom->domain_size[k];
            vtx->pos[k] -= L * floor((vtx->pos[k] - cmA[k]) / L + 0.5);
          }
        }
        s = 0;
        for (int vi = 0; vi < objB->n_vertices; ++vi)
        {
          MFTL_Vertex* vtx = &objB->vertices[vi];
          memcpy(saveB + 3 * s, vtx->pos, 3 * sizeof(double));
          memcpy(imgB + 3 * s, vtx->img, 3 * sizeof(int));
          vtx->img[0] = vtx->img[1] = vtx->img[2] = 0;
          s++;
          for (int k = 0; k < 3; ++k)
          {
            if (!dom->is_periodic[k]) continue;
            double L = dom->domain_size[k];
            vtx->pos[k] += img[k] - L * floor((vtx->pos[k] - cmB[k]) / L + 0.5);
          }
        }
      }

      MFTL_Overlap* pair_records = NULL;
      int pair_n = 0;
      mftlOverlapObjectPair(objA, objB, &pair_records, &pair_n);

      if (any_periodic)
      {
        int s = 0;
        for (int vi = 0; vi < objA->n_vertices; ++vi)
        {
          MFTL_Vertex* vtx = &objA->vertices[vi];
          memcpy(vtx->pos, saveA + 3 * s, 3 * sizeof(double));
          memcpy(vtx->img, imgA + 3 * s, 3 * sizeof(int));
          s++;
        }
        s = 0;
        for (int vi = 0; vi < objB->n_vertices; ++vi)
        {
          MFTL_Vertex* vtx = &objB->vertices[vi];
          memcpy(vtx->pos, saveB + 3 * s, 3 * sizeof(double));
          memcpy(vtx->img, imgB + 3 * s, 3 * sizeof(int));
          s++;
        }
        free(saveA);
        free(saveB);
        free(imgA);
        free(imgB);
        MFTL_Ensemble* sync_ens = (ensB && ensB != ensA) ? ensB : ensA;
        MFTLUpdateGhostPositions(sync_ens, NULL);
      }

    if (pair_n > 0)
    {
      if (mine_n + pair_n > mine_cap)
      {
        mine_cap = (mine_cap > 0) ? mine_cap : 64;
        while (mine_n + pair_n > mine_cap) mine_cap *= 2;
        mine = (MFTL_Overlap*) realloc(mine, (size_t) mine_cap * sizeof(MFTL_Overlap));
      }
      memcpy(mine + mine_n, pair_records, (size_t) pair_n * sizeof(MFTL_Overlap));
      mine_n += pair_n;
    }
    free(pair_records);
    }
  }

  // Gather all records to every rank (each pair was computed exactly once).
  int my_n = mine_n;
  int* all_n = NULL;
  if (dom->nproc > 1)
  {
    all_n = (int*) malloc((size_t) dom->nproc * sizeof(int));
    MPI_Allgather(&my_n, 1, MPI_INT, all_n, 1, MPI_INT, dom->comm_cart);
  }
  else
  {
    all_n = (int*) malloc(sizeof(int));
    all_n[0] = my_n;
  }

  int total = 0;
  int* displs = (int*) malloc((size_t) (dom->nproc + 1) * sizeof(int));
  displs[0] = 0;
  for (int r = 0; r < dom->nproc; ++r)
  {
    total += all_n[r];
    displs[r + 1] = displs[r] + all_n[r];
  }

  MFTL_Overlap* all = NULL;
  if (total > 0) all = (MFTL_Overlap*) malloc((size_t) total * sizeof(MFTL_Overlap));
  if (dom->nproc > 1)
  {
    // MPI_BYTE transport: counts and displacements must be in BYTES.
    int* bcounts = (int*) malloc((size_t) dom->nproc * sizeof(int));
    int* bdispls = (int*) malloc((size_t) dom->nproc * sizeof(int));
    for (int r = 0; r < dom->nproc; ++r)
    {
      bcounts[r] = all_n[r] * (int) sizeof(MFTL_Overlap);
      bdispls[r] = displs[r] * (int) sizeof(MFTL_Overlap);
    }
    MPI_Allgatherv(mine, my_n * (int) sizeof(MFTL_Overlap), MPI_BYTE, all, bcounts, bdispls, MPI_BYTE, dom->comm_cart);
    free(bcounts);
    free(bdispls);
  }
  else if (total > 0)
  {
    memcpy(all, mine, (size_t) total * sizeof(MFTL_Overlap));
  }

  // Deterministic ids in gather order.
  for (int r = 0; r < total; ++r) all[r].id = r;

  free(all_n);
  free(displs);
  free(mine);
  if (out_records) *out_records = all;
  else free(all);
  if (out_count) *out_count = total;
  return MFTL_SUCCESS;
}
