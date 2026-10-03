/**
 * @file body_integrals.c
 * @author Christian Gorges
 * @date 2026-10-03
 * @ingroup mftl_geometry
 *
 * @brief The local half of a body's area, volume and centroid: per-triangle
 *        flux integrals over one rank's OWNED triangles, and where the rims lie.
 *
 * Every quantity is the flux of a polynomial field through the surface, so the
 * divergence theorem turns it into a volume integral of the region the surface
 * bounds -- alone for a closed body, together with the domain walls its rims lie
 * on for a wall-closed one. With y = x - r (r a reference, the body lifted into
 * one periodic image about its centre) and S_t the triangle's area vector,
 * exactly, for a flat triangle with linear integrands:
 *
 *   V      = sum_t (ybar_k) S_t,k                                  F = y_k e_k
 *   Q_kj   = sum_t S_t,k / 12 (sum_i y_k,i y_j,i + sum_i y_k,i sum_i y_j,i)
 *
 * where Q_kk = 2 V (C_k - r_k) (field y_k^2 / 2 e_k) and, for j != k,
 * Q_kj = V (C_j - r_j) (field y_k y_j e_k). The collective assembly and the
 * rules for which axis and which reference are MFTLEnsembleBodyIntegrals().
 *
 * The VOLUME is accumulated with exactly the arithmetic the topology change's
 * volume correction always used, so that module can call it and stay
 * bit-identical.
 */

#include "mftl_geometry.h"

#include <math.h>

#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_types.h"

int MFTLBodyRimWalls(const MFTL_Object *obj, bool *off_wall, int *n_rim)
{
  const MFTL_Domain *const dom = obj->domain;
  int mask = 0, count = 0;
  bool off = MFTL_FALSE;
  const int nh = 3 * obj->tris.n_local;
  for (int h = 0; h < nh; ++h)
  {
    if (!(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY)) continue;
    count++;
    const double *const a = MFTLVertexPos(obj, obj->tris.he[h].vertex_idx);
    const double *const b = MFTLVertexPos(obj, obj->tris.he[MFTLHePrevIdx(h)].vertex_idx);

    /* Which walls the edge's TARGET lies on, and whether both ends lie on one. */
    bool on_wall = MFTL_FALSE;
    for (int k = 0; k < 3; ++k)
    {
      if (dom == MFTL_NULL || dom->is_periodic[k]) continue;
      const double tol = MFTL_BODY_WALL_TOL * dom->domain_size[k];
      const bool a_lo = fabs(a[k] - dom->global_min[k]) <= tol, a_hi = fabs(a[k] - dom->global_max[k]) <= tol;
      if (a_lo) mask |= 1 << (2 * k);
      if (a_hi) mask |= 1 << (2 * k + 1);
      if (a_lo && fabs(b[k] - dom->global_min[k]) <= tol) on_wall = MFTL_TRUE;
      if (a_hi && fabs(b[k] - dom->global_max[k]) <= tol) on_wall = MFTL_TRUE;
    }
    if (!on_wall) off = MFTL_TRUE;
  }
  if (off_wall != MFTL_NULL) *off_wall = off;
  if (n_rim != MFTL_NULL) *n_rim = count;
  return mask;
}

void MFTLBodyShape(const MFTL_Object *obj, double extent[MFTL_PERIODIC_EXTENT_SLOTS], double S[3], double *area)
{
  double s[3] = {0.0, 0.0, 0.0}, a = 0.0;
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    MFTLPeriodicExtentAddTriangle(obj->domain, T, extent);
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    const double n[3] = {0.5 * (e1[1] * e2[2] - e1[2] * e2[1]), 0.5 * (e1[2] * e2[0] - e1[0] * e2[2]), 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])};
    a += sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    for (int k = 0; k < 3; ++k) s[k] += n[k];
  }
  for (int k = 0; k < 3; ++k) S[k] += s[k];
  *area += a;
}

void MFTLBodyFluxVolume(const MFTL_Object *obj, int k, double ref, const double centre[3], double *volume, double *area)
{
  const MFTL_Domain *const dom = obj->domain;
  double v = 0.0, a = 0.0;

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    const double n[3] = {0.5 * (e1[1] * e2[2] - e1[2] * e2[1]), 0.5 * (e1[2] * e2[0] - e1[0] * e2[2]), 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])};
    a += sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (k < 0) continue;

    double x = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
    if (dom != MFTL_NULL && dom->is_periodic[k])
    {
      /* The image nearest the body's centre: a body crossing the seam on this
       * axis is measured in one piece. */
      const double L = dom->domain_size[k];
      x -= L * nearbyint((x - centre[k]) / L);
    }
    v += (x - ref) * n[k];
  }
  *volume = v;
  *area = a;
}

void MFTLBodyFluxMoments(const MFTL_Object *obj, int k, double ref, const double centre[3], double out[MFTL_BODY_MOMENT_SLOTS])
{
  const MFTL_Domain *const dom = obj->domain;
  double q[MFTL_BODY_MOMENT_SLOTS] = {0.0, 0.0, 0.0, 0.0};

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    const double Sk = 0.5 * (e1[(k + 1) % 3] * e2[(k + 2) % 3] - e1[(k + 2) % 3] * e2[(k + 1) % 3]);

    /* y = x - r with the triangle moved rigidly into the image nearest the
     * centre; r = ref on the flux axis, the centre on the others. */
    double y[3][3];
    double sum[3];
    for (int j = 0; j < 3; ++j)
    {
      const double r = (j == k) ? ref : centre[j];
      double shift = 0.0;
      if (dom != MFTL_NULL && dom->is_periodic[j])
      {
        const double L = dom->domain_size[j];
        const double xbar = (T[0][j] + T[1][j] + T[2][j]) / 3.0;
        shift = -L * nearbyint((xbar - centre[j]) / L);
      }
      sum[j] = 0.0;
      for (int i = 0; i < 3; ++i)
      {
        y[i][j] = (T[i][j] + shift) - r;
        sum[j] += y[i][j];
      }
    }

    q[0] += (sum[k] / 3.0) * Sk;
    for (int j = 0; j < 3; ++j)
    {
      const double dot = y[0][k] * y[0][j] + y[1][k] * y[1][j] + y[2][k] * y[2][j];
      q[1 + j] += Sk / 12.0 * (dot + sum[k] * sum[j]);
    }
  }
  for (int i = 0; i < MFTL_BODY_MOMENT_SLOTS; ++i) out[i] = q[i];
}
