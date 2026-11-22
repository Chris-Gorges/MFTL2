/**
 * @file geometry_internal.h
 * @author Christian Gorges
 * @date 2026-11-21
 * @ingroup mftl_geometry
 *
 * @brief What the vertex-normal and curvature kernels share, written once:
 *        the corner angle, a face's term in a vertex-normal sum, the tangent
 *        frame of a normal and the one-ring samples of a height fit.
 *
 * vertex_geometry.c (the normals) and vertex_curvature.c (COTAN and FIT) both
 * scatter faces into vertex sums and both fit a paraboloid to a one-ring; each
 * piece below is the one implementation both use. All inline: they sit inside
 * the triangle scatters, where a call would cost what the fusion saves.
 */

#ifndef MFTL_GEOMETRY_INTERNAL_H
#define MFTL_GEOMETRY_INTERNAL_H

#include <math.h>
#include <stdbool.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_types.h"
#include "mftl_vectoroperations.h"

/** @cond INTERNAL */

/**
 * @brief The interior angle of a corner whose two edges have cross product of
 *        length @p mag and dot product @p dot: `atan2(mag, dot)`.
 *
 * Both arguments carry the same factor |e1||e2|, so the edges need no
 * normalising, and `atan2` is well conditioned for every angle, unlike `acos`
 * of the normalised dot product near 0 and \f$\pi\f$. The ANGLE vertex normal
 * and the Gaussian curvature's angle deficit both use it.
 *
 * @param mag \f$\norm{\vect{e}_1\times\vect{e}_2}\f$, positive.
 * @param dot \f$\vect{e}_1\cdot\vect{e}_2\f$.
 * @return The angle in \f$(0, \pi)\f$.
 */
static inline double MFTLGeomCornerAngle(double mag, double dot) { return atan2(mag, dot); }

/**
 * @brief The dot products of the two edges AT each corner of the face
 *        (p0, p0 + e1, p0 + e2).
 *
 * Each from the corner's own edges -- corner 1 has -e1 and c = e2 - e1, corner
 * 2 has -e2 and -c -- never as a difference of dot products (d11 - d12), which
 * cancels at a sliver's near-right corners. Negative means obtuse there: an
 * exact sign.
 *
 * @param      e1 p1 - p0.
 * @param      e2 p2 - p0.
 * @param[out] d  The three corner dot products.
 */
static inline void MFTLGeomCornerDots(const double e1[3], const double e2[3], double d[3])
{
  const double c[3] = {e2[0] - e1[0], e2[1] - e1[1], e2[2] - e1[2]};
  d[0] = MFTLVectorInProd(e1, e2);
  d[1] = -MFTLVectorInProd(e1, c);
  d[2] = MFTLVectorInProd(e2, c);
}

/**
 * @brief What one face adds to the normal sums of its three corners: THE
 *        triangle branch of #MFTL_NormalWeighting, for every scatter.
 *
 * AREA adds the cross product itself, \f$\vect{e}_1\times\vect{e}_2 =
 * 2A\hat{\vect{n}}\f$, at every corner (a zero-area face adds zero). ANGLE adds
 * the unit normal times each corner's angle (MFTLGeomCornerAngle(),
 * MFTLGeomCornerDots()); a zero-area face has no unit normal and adds nothing.
 * MFTLComputeVertexNormals() and the area-weighted normal of the COTAN
 * curvature both scatter with it, so the two are the same sum to the bit.
 *
 * @param         angle ANGLE weights, else AREA.
 * @param         e1    p1 - p0.
 * @param         e2    p2 - p0.
 * @param[in,out] n     On entry \f$\vect{e}_1\times\vect{e}_2\f$; on return the
 *                      vector to add.
 * @param[out]    w     Its weight at corners 0, 1, 2.
 * @return `false` when the face adds nothing.
 */
static inline bool MFTLGeomFaceNormalTerm(bool angle, const double e1[3], const double e2[3], double n[3], double w[3])
{
  w[0] = w[1] = w[2] = 1.0;
  if (!angle) return MFTL_TRUE;
  const double mag = MFTL_SQRT(MFTLVectorNormSq(n));
  if (!(mag > 0.0)) return MFTL_FALSE;
  const double inv = 1.0 / mag;
  n[0] *= inv;
  n[1] *= inv;
  n[2] *= inv;
  double d[3];
  MFTLGeomCornerDots(e1, e2, d);
  for (int k = 0; k < 3; ++k) w[k] = MFTLGeomCornerAngle(mag, d[k]);
  return MFTL_TRUE;
}

/**
 * @brief Adds a face's term into the sums of its corners @p v0, @p v1, @p v2:
 *        `sum[stride * v_k + j] += w[k] * n[j]`.
 *
 * @param[in,out] sum    Per-vertex records, the normal at offset 0.
 * @param         stride Doubles per vertex record.
 * @param         v0     Corner 0.
 * @param         v1     Corner 1.
 * @param         v2     Corner 2.
 * @param         n      The vector (MFTLGeomFaceNormalTerm()).
 * @param         w      Its weight per corner.
 */
static inline void MFTLGeomAddFaceNormal(double *restrict sum, int stride, int v0, int v1, int v2, const double n[3], const double w[3])
{
  double *const s0 = &sum[(size_t) stride * v0], *const s1 = &sum[(size_t) stride * v1], *const s2 = &sum[(size_t) stride * v2];
  for (int j = 0; j < 3; ++j)
  {
    s0[j] += w[0] * n[j];
    s1[j] += w[1] * n[j];
    s2[j] += w[2] * n[j];
  }
}

/**
 * @brief An orthonormal tangent frame for the unit normal @p n: @p t1 is the
 *        coordinate axis @p n is least aligned with, made orthogonal to it,
 *        and @p t2 = n x t1. Depends on the direction of @p n only.
 *
 * @param      n  A unit normal.
 * @param[out] t1 First tangent.
 * @param[out] t2 Second tangent.
 */
static inline void MFTLGeomTangentFrame(const double n[3], double t1[3], double t2[3])
{
  int a = 0;
  if (fabs(n[1]) < fabs(n[a])) a = 1;
  if (fabs(n[2]) < fabs(n[a])) a = 2;
  t1[0] = -n[a] * n[0];
  t1[1] = -n[a] * n[1];
  t1[2] = -n[a] * n[2];
  t1[a] += 1.0;
  MFTLNormalizeVector(t1);
  MFTLVectorCrossVector(t2, n, t1);
}

/**
 * @brief The vertex @p v and its one-ring @p ring as the samples of a height
 *        fit in the frame (t1, t2, n), in units of the ring's RMS edge length.
 *
 * Sample 0 is the vertex at the origin, sample i + 1 is `ring[i]` at its
 * minimum-image offset from @p v, projected onto t1, t2 and n. Dividing by the
 * RMS edge h makes heights and slopes both O(1), so one weight serves both
 * kinds of row (MFTLFitHeightParaboloid()), and makes the fit independent of
 * the length unit. The FIT normal and the FIT curvature both sample with it.
 *
 * @param      obj  The mesh.
 * @param      v    The vertex.
 * @param      n    Unit frame normal.
 * @param      t1   First tangent (MFTLGeomTangentFrame()).
 * @param      t2   Second tangent.
 * @param      ring The one-ring (MFTLGetVertexOneRingVertices()).
 * @param      nr   Its length.
 * @param[out] pts  `[3 (nr + 1)]` sample coordinates.
 * @return 1/h, or 0 when the ring has no extent (no fit possible).
 */
static inline double MFTLGeomRingSamples(const MFTL_Object *obj, int v, const double n[3], const double t1[3], const double t2[3], const int *ring, int nr,
                                         double *pts)
{
  pts[0] = pts[1] = pts[2] = 0.0;
  const double *const xv = MFTLVertexPos(obj, v);
  double h2 = 0.0;
  for (int i = 0; i < nr; ++i)
  {
    double d[3];
    const double *const xj = MFTLVertexPos(obj, ring[i]);
    if (obj->domain != MFTL_NULL)
      MFTLGetMinimumImageVector(obj->domain, d, xv, xj);
    else
      d[0] = xj[0] - xv[0], d[1] = xj[1] - xv[1], d[2] = xj[2] - xv[2];
    h2 += MFTLVectorNormSq(d);
    double *const p = &pts[3 * (i + 1)];
    p[0] = MFTLVectorInProd(d, t1);
    p[1] = MFTLVectorInProd(d, t2);
    p[2] = MFTLVectorInProd(d, n);
  }
  if (!(h2 > 0.0)) return 0.0;
  const double hinv = 1.0 / MFTL_SQRT(h2 / (double) nr);
  for (int i = 3; i < 3 * (nr + 1); ++i) pts[i] *= hinv;
  return hinv;
}

/** @endcond */

#endif  // MFTL_GEOMETRY_INTERNAL_H
