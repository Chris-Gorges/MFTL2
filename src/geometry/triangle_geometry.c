/**
 * @file triangle_geometry.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_geometry
 *
 * @brief Per-triangle geometry for a whole mesh, and the closest point on a
 *        triangle.
 *
 * MFTLComputeTriangleGeometry() has two bit-identical paths chosen by the
 * object's state: the steady-state path streams `tris.verts` with periodicity
 * unswitched outside the loop; the fallback walks the half-edge records and is
 * valid before derived connectivity has been built.
 */

#include "mftl_geometry.h"

#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_types.h"

/**
 * @brief The fallback path: one MFTLGetTriangleProperties() per triangle.
 *
 * That inline function is the definition of a triangle's geometry, so it is
 * reused rather than re-spelled; outputs are stored straight into the arrays.
 *
 * @param obj   The mesh. `tris.area`, `tris.normal` and `tris.center` are written.
 * @param count Triangles to process, from index 0.
 */
static void geometry_by_walk(MFTL_Object *obj, int count)
{
  for (int t = 0; t < count; ++t)
  {
    MFTLGetTriangleProperties(obj, t, &obj->tris.area[t], &obj->tris.normal[3 * t], &obj->tris.center[3 * t]);
  }
}

/**
 * @brief The steady-state path: compact indices, gathered positions,
 *        periodicity unswitched (#MFTL_FOREACH_TRIANGLE_GEOMETRY).
 *
 * The centroid \f$\vect{p}_0 + (\vect{e}_1+\vect{e}_2)/3\f$ is a true centroid
 * but not necessarily inside the box, so it is wrapped (redistribution feeds it
 * to MFTLGetOwnerProc()).
 *
 * @param obj   The mesh. Requires #MFTL_DERIVED_TOPOLOGY.
 * @param count Triangles to process, from index 0.
 */
static void geometry_by_vertices(MFTL_Object *obj, int count)
{
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, count,
    /* Unnormalised normal; |n| = 2A, so one sqrt serves area and normal. */
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;

    /* Unclamped magnitude, reciprocal guarded against zero -- the same
     * expression as MFTLGetTriangleProperties(), which
     * test_triangle_geometry_paths_are_bit_identical compares with memcmp. */
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;

    out_area[t] = 0.5 * mag;
    out_normal[3 * t + 0] = nx * inv;
    out_normal[3 * t + 1] = ny * inv;
    out_normal[3 * t + 2] = nz * inv;

    double c[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
    MFTLApplyPeriodicity(dom, c);

    out_center[3 * t + 0] = c[0];
    out_center[3 * t + 1] = c[1];
    out_center[3 * t + 2] = c[2];);
}

void MFTLComputeTriangleGeometry(MFTL_Object *obj, int count)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(count <= obj->tris.n && "MFTLComputeTriangleGeometry: count exceeds the triangle pool");

  if (count <= 0) return;

  if ((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u)
    geometry_by_vertices(obj, count);
  else
    geometry_by_walk(obj, count);

  /* Only a pass over every triangle makes the ghosts' geometry current too. */
  if (count == obj->tris.n) obj->derived |= MFTL_DERIVED_GEOMETRY;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Closest points
 *
 * Each query below is a small linear least-squares problem: the foot of a
 * perpendicular. Solving it through its 2x2 NORMAL EQUATIONS -- a Gram
 * determinant a c - b^2 and numerators that are differences of products of dot
 * products -- squares the problem's condition number. Two nearly parallel
 * segments, or a point over a sliver, then lose twice the digits the geometry
 * costs: a gap of 1e-9 between edges at an angle of 2e-8 came out as 1e-8, and
 * the "parallel" threshold sin^2 <= 1e-15 that guarded the noise discarded the
 * interior answer outright (docs/analysis.md §7).
 *
 * Here the segment determinant is a cross product, |u x v|^2 (Lagrange's
 * identity, without the cancellation), its numerators triple products, and a
 * triangle's in-plane coordinates come from a Gram-Schmidt (QR) frame pivoted on
 * its longest edge. Every distance is evaluated from LOCAL differences, never
 * from a returned point, so a configuration far from the origin keeps its
 * accuracy. The result is the distance to rounding: a few ulps of the
 * configuration's size, at every angle and aspect ratio
 * (test_segment_distance_is_accurate_at_every_angle,
 * test_point_face_distance_is_accurate_on_slivers).
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Clamps @p x to [0, 1]; a NaN becomes 0.
 * @param x Value to clamp.
 * @return The clamped value.
 */
static inline double clamp01(double x) { return fmin(fmax(x, 0.0), 1.0); }

/**
 * @brief Closest point on the SEGMENT [a, b] to @p p.
 *
 * @param      p       Query point.
 * @param      a       Segment start.
 * @param      b       Segment end.
 * @param[out] out     Closest point; an endpoint is returned bit-exactly.
 * @param[out] where   0 at @p a, 1 at @p b, 2 strictly inside.
 * @return Squared distance, from local differences.
 */
static double closest_on_segment(const double p[3], const double a[3], const double b[3], double out[3], int *where)
{
  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  const double proj = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];

  double s;
  if (proj <= 0.0 || len_sq <= 0.0)
  {
    s = 0.0;
    *where = 0;
  }
  else if (proj >= len_sq)
  {
    s = 1.0;
    *where = 1;
  }
  else
  {
    s = proj / len_sq;
    *where = 2;
  }

  double d_sq = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    out[k] = (*where == 0) ? a[k] : (*where == 1) ? b[k] : a[k] + s * ab[k];
    const double d = (*where == 1) ? p[k] - b[k] : ap[k] - s * ab[k];
    d_sq += d * d;
  }
  return d_sq;
}

/**
 * @brief The degenerate-triangle answer: the nearest of the three edges (ties to
 *        the lower edge index).
 *
 * @param      p       Query point.
 * @param      tri     The three corners.
 * @param[out] closest Closest point.
 * @param[out] feature Feature of the closest point.
 * @return Squared distance.
 */
static double closest_on_degenerate(const double p[3], const double tri[3][3], double closest[3], MFTL_TriangleFeature *feature)
{
  double best = MFTL_DBL_MAX;
  for (int e = 0; e < 3; ++e)
  {
    double q[3];
    int where = 0;
    const double d_sq = closest_on_segment(p, tri[e], tri[(e + 1) % 3], q, &where);
    if (d_sq < best)
    {
      best = d_sq;
      closest[0] = q[0];
      closest[1] = q[1];
      closest[2] = q[2];
      if (where == 0)
        *feature = (MFTL_TriangleFeature) (MFTL_TRI_FEATURE_VERTEX0 + e);
      else if (where == 1)
        *feature = (MFTL_TriangleFeature) (MFTL_TRI_FEATURE_VERTEX0 + (e + 1) % 3);
      else
        *feature = (MFTL_TriangleFeature) (MFTL_TRI_FEATURE_EDGE0 + e);
    }
  }
  return best;
}

/**
 * @brief A triangle's plane as an orthogonalised frame on its longest edge.
 *
 * With p0 -> p1 the longest edge (the lowest index on a tie) and p2 the third
 * corner, `e = p1 - p0` and `g = (p2 - p0) - c e` with
 * `c = (p2 - p0).e / e.e` is one Gram-Schmidt step, i.e. a QR factorisation of
 * the edge matrix with the longest column first: `g` is the apex's offset
 * perpendicular to `e`, obtained by a subtraction instead of through a 2x2
 * determinant. Pivoting on the longest edge keeps `c` in [0, 1] (the foot of the
 * altitude onto the longest side lies on it).
 */
typedef struct
{
  const double *p0;  ///< Start of the longest edge.
  double e[3];  ///< The longest edge, p1 - p0.
  double g[3];  ///< Apex offset perpendicular to `e`.
  double c;  ///< Apex foot parameter along `e`.
  double ee;  ///< |e|^2.
  double gg;  ///< |g|^2; zero when the corners are numerically collinear.
  int k;  ///< Corner index of p0; p1 is corner k+1, the apex corner k+2 (mod 3).
} plane_frame;

/**
 * @brief Builds the frame of @p tri.
 *
 * @param      tri The three corners.
 * @param[out] f   The frame.
 * @return `true` if the frame spans a plane (`gg > 0`); `false` for numerically
 *         collinear corners, whose point set is covered by their edges.
 */
static bool frame_init(const double tri[3][3], plane_frame *f)
{
  double len[3];
  for (int i = 0; i < 3; ++i)
  {
    const double *a = tri[i], *b = tri[(i + 1) % 3];
    const double d[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    len[i] = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
  }
  int k = 0;
  if (len[1] > len[k]) k = 1;
  if (len[2] > len[k]) k = 2;

  const double *const p0 = tri[k], *const p1 = tri[(k + 1) % 3], *const p2 = tri[(k + 2) % 3];
  f->p0 = p0;
  f->k = k;
  double fv[3];
  for (int j = 0; j < 3; ++j)
  {
    f->e[j] = p1[j] - p0[j];
    fv[j] = p2[j] - p0[j];
  }
  f->ee = f->e[0] * f->e[0] + f->e[1] * f->e[1] + f->e[2] * f->e[2];
  if (!(f->ee > 0.0))
  {
    f->gg = 0.0;
    return MFTL_FALSE;
  }
  f->c = (fv[0] * f->e[0] + fv[1] * f->e[1] + fv[2] * f->e[2]) / f->ee;
  for (int j = 0; j < 3; ++j) f->g[j] = fv[j] - f->c * f->e[j];
  f->gg = f->g[0] * f->g[0] + f->g[1] * f->g[1] + f->g[2] * f->g[2];
  return f->gg > 0.0;
}

/**
 * @brief Where @p p projects in a frame: its coordinates and barycentric weights.
 *
 * `p - p0 = s e + t g + off`, with `s` from the long edge and `t` measured on
 * the residual `r = (p - p0) - s e` alone, so neither coordinate is a difference
 * of products. The barycentric weights are `t` for the apex, `s - c t` for p1
 * and `1 - s - (1 - c) t` for p0. A frame that cannot resolve the tilt of the
 * plane about its long edge (a sliver) moves the distance only to second order,
 * because the offset (frame_offset()) is measured from the long edge, within the
 * sliver's width of the projection.
 *
 * @param      f    A frame for which frame_init() returned `true`.
 * @param      p    Query point.
 * @param[out] st   (s, t).
 * @param[out] r    The residual `(p - p0) - s e`.
 * @param[out] bary Barycentric weight of each corner, indexed by corner.
 */
static inline void frame_locate(const plane_frame *f, const double p[3], double st[2], double r[3], double bary[3])
{
  const double *const p0 = f->p0;
  const double d[3] = {p[0] - p0[0], p[1] - p0[1], p[2] - p0[2]};
  const double s = (d[0] * f->e[0] + d[1] * f->e[1] + d[2] * f->e[2]) / f->ee;
  for (int j = 0; j < 3; ++j) r[j] = d[j] - s * f->e[j];
  const double t = (r[0] * f->g[0] + r[1] * f->g[1] + r[2] * f->g[2]) / f->gg;

  st[0] = s;
  st[1] = t;
  bary[(f->k + 2) % 3] = t;
  bary[(f->k + 1) % 3] = s - f->c * t;
  bary[f->k] = 1.0 - s - (1.0 - f->c) * t;
}

/**
 * @brief `p` minus its projection, from the residual of frame_locate(), and its
 *        squared length.
 *
 * @param      f   The frame.
 * @param      t   The apex coordinate from frame_locate().
 * @param      r   The residual from frame_locate().
 * @param[out] off The offset vector.
 * @return |off|^2.
 */
static inline double frame_offset(const plane_frame *f, double t, const double r[3], double off[3])
{
  for (int j = 0; j < 3; ++j) off[j] = r[j] - t * f->g[j];
  return off[0] * off[0] + off[1] * off[1] + off[2] * off[2];
}

/**
 * @brief The projection itself, `p0 + s e + t g`, on the plane.
 *
 * @param      f  The frame.
 * @param      st (s, t) from frame_locate().
 * @param[out] q  The point.
 */
static inline void frame_point(const plane_frame *f, const double st[2], double q[3])
{
  for (int j = 0; j < 3; ++j) q[j] = f->p0[j] + st[0] * f->e[j] + st[1] * f->g[j];
}

double MFTLClosestPointOnTriangle(const double p[3], const double tri[3][3], double closest[3], MFTL_TriangleFeature *feature)
{
  double q_sink[3];
  MFTL_TriangleFeature f_sink;
  double *const q = (closest != MFTL_NULL) ? closest : q_sink;
  MFTL_TriangleFeature *const f = (feature != MFTL_NULL) ? feature : &f_sink;

  const double *const a = tri[0];
  const double *const b = tri[1];
  const double *const c = tri[2];

  const double ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};

  /* Degeneracy, scale-free: |ab x ac|^2 = |ab|^2 |ac|^2 sin^2(angle at a), so
   * the test is on sin^2 alone. Below it the face normal a caller would sign
   * with is unreliable, so the triangle is answered as its three edges. */
  const double cx = ab[1] * ac[2] - ab[2] * ac[1];
  const double cy = ab[2] * ac[0] - ab[0] * ac[2];
  const double cz = ab[0] * ac[1] - ab[1] * ac[0];
  const double cross_sq = cx * cx + cy * cy + cz * cz;
  const double ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  const double ac_sq = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
  if (!(cross_sq > MFTL_VERYVERYSMALL * ab_sq * ac_sq)) return closest_on_degenerate(p, tri, q, f);

  /* Ericson's region test (Real-Time Collision Detection, §5.1.5). The vertex
   * and edge-end conditions are single dot products and well conditioned; the
   * three that ask "which side of an edge's line" are the barycentric signs,
   * taken from the frame rather than from Ericson's va, vb, vc (differences of
   * products of dot products, the normal equations in disguise). */
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
  const double d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];

  double off[3];  // p - q, from local differences
  if (d1 <= 0.0 && d2 <= 0.0)
  {
    *f = MFTL_TRI_FEATURE_VERTEX0;
    for (int k = 0; k < 3; ++k) q[k] = a[k], off[k] = ap[k];
  }
  else
  {
    const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
    const double d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    const double d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];

    if (d3 >= 0.0 && d4 <= d3)
    {
      *f = MFTL_TRI_FEATURE_VERTEX1;
      for (int k = 0; k < 3; ++k) q[k] = b[k], off[k] = bp[k];
    }
    else
    {
      const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
      const double d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
      const double d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];

      plane_frame fr;
      if (!frame_init(tri, &fr)) return closest_on_degenerate(p, tri, q, f);
      double bary[3], st[2], r[3];
      frame_locate(&fr, p, st, r, bary);

      if (bary[2] <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
      {
        const double w = d1 / (d1 - d3);
        *f = MFTL_TRI_FEATURE_EDGE0;
        for (int k = 0; k < 3; ++k) q[k] = a[k] + w * ab[k], off[k] = ap[k] - w * ab[k];
      }
      else if (d6 >= 0.0 && d5 <= d6)
      {
        *f = MFTL_TRI_FEATURE_VERTEX2;
        for (int k = 0; k < 3; ++k) q[k] = c[k], off[k] = cp[k];
      }
      else if (bary[1] <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
      {
        const double w = d2 / (d2 - d6);
        *f = MFTL_TRI_FEATURE_EDGE2;
        for (int k = 0; k < 3; ++k) q[k] = a[k] + w * ac[k], off[k] = ap[k] - w * ac[k];
      }
      else if (bary[0] <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
      {
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        *f = MFTL_TRI_FEATURE_EDGE1;
        for (int k = 0; k < 3; ++k)
        {
          const double bc = c[k] - b[k];
          q[k] = b[k] + w * bc;
          off[k] = bp[k] - w * bc;
        }
      }
      else
      {
        *f = MFTL_TRI_FEATURE_FACE;
        frame_offset(&fr, st[1], r, off);
        frame_point(&fr, st, q);
      }
    }
  }

  return off[0] * off[0] + off[1] * off[1] + off[2] * off[2];
}

/**
 * @brief The segment kernel on precomputed edge vectors: the squared distance
 *        and the parameters of the closest pair.
 *
 * Split from MFTLClosestPointsOnSegments() so the triangle pair, which uses
 * each edge in three pairs, computes edge vectors and lengths once and builds a
 * witness only for a candidate that improves. Same arithmetic, same bits.
 *
 * @param      p0 Start of P.
 * @param      u  P's direction, p1 - p0.
 * @param      a  |u|^2.
 * @param      q0 Start of Q.
 * @param      v  Q's direction, q1 - q0.
 * @param      c  |v|^2.
 * @param[out] s_out Parameter on P, in [0, 1].
 * @param[out] t_out Parameter on Q, in [0, 1].
 * @return The squared distance, evaluated from local differences.
 */
static inline double segments_kernel(const double p0[3], const double u[3], double a, const double q0[3], const double v[3], double c, double *s_out,
                                     double *t_out)
{
  const double w[3] = {p0[0] - q0[0], p0[1] - q0[1], p0[2] - q0[2]};
  const double d = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
  const double e = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];

  double s = 0.0, t = 0.0;
  if (a > 0.0 && c > 0.0)
  {
    const double b = u[0] * v[0] + u[1] * v[1] + u[2] * v[2];

    /* The stationary point of |w + s u - t v|^2 by Cramer's rule, with
     * D = ac - b^2 = |u x v|^2 and the numerator b e - c d = (u x v).(v x w).
     * No threshold: wherever D is too small to be trusted, so is the valley
     * the error runs along, and the clamped cascade below costs at most a few
     * ulps of the configuration. Exactly parallel segments (D = 0) pin s = 0;
     * the cascade then finds the closest point of their overlap. */
    const double n[3] = {u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    const double D = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    if (D > 0.0)
    {
      const double vw[3] = {v[1] * w[2] - v[2] * w[1], v[2] * w[0] - v[0] * w[2], v[0] * w[1] - v[1] * w[0]};
      s = clamp01((n[0] * vw[0] + n[1] * vw[1] + n[2] * vw[2]) / D);
    }

    /* Ericson's cascade (§5.1.9): the best t for this s; if it leaves [0, 1],
     * the minimum lies on that side of the square and s is re-solved there. */
    t = (b * s + e) / c;
    if (t < 0.0)
    {
      t = 0.0;
      s = clamp01(-d / a);
    }
    else if (t > 1.0)
    {
      t = 1.0;
      s = clamp01((b - d) / a);
    }
  }
  else if (a > 0.0)
    s = clamp01(-d / a);  // Q is a point
  else if (c > 0.0)
    t = clamp01(e / c);  // P is a point

  double d_sq = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    const double dk = w[k] + s * u[k] - t * v[k];
    d_sq += dk * dk;
  }
  *s_out = s;
  *t_out = t;
  return d_sq;
}

/**
 * @brief The point at parameter @p s of the segment from @p p0 to @p p1, an
 *        endpoint bit-exactly (as a corner of MFTLClosestPointOnTriangle()).
 *
 * @param      p0  Start.
 * @param      p1  End.
 * @param      u   p1 - p0.
 * @param      s   Parameter in [0, 1].
 * @param[out] out The point.
 */
static inline void segment_point(const double p0[3], const double p1[3], const double u[3], double s, double out[3])
{
  for (int k = 0; k < 3; ++k) out[k] = (s == 0.0) ? p0[k] : (s == 1.0) ? p1[k] : p0[k] + s * u[k];
}

double MFTLClosestPointsOnSegments(const double p0[3], const double p1[3], const double q0[3], const double q1[3], double c1[3], double c2[3])
{
  const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
  const double v[3] = {q1[0] - q0[0], q1[1] - q0[1], q1[2] - q0[2]};
  const double a = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
  const double c = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];

  double s, t;
  const double d_sq = segments_kernel(p0, u, a, q0, v, c, &s, &t);
  if (c1 != MFTL_NULL) segment_point(p0, p1, u, s, c1);
  if (c2 != MFTL_NULL) segment_point(q0, q1, v, t, c2);
  return d_sq;
}

/**
 * @brief Writes @p src into @p dst.
 * @param[out] dst Destination.
 * @param      src Source.
 */
static inline void copy3(double dst[3], const double src[3])
{
  dst[0] = src[0];
  dst[1] = src[1];
  dst[2] = src[2];
}

double MFTLClosestPointsBetweenTriangles(const double A[3][3], const double B[3][3], double pa[3], double pb[3])
{
  /* Two disjoint closed triangles meet their minimum distance either between
   * two edges (nine segment pairs) or at a corner of one over the INTERIOR of
   * the other (six face projections; a projection onto the boundary is one of
   * the segment pairs). The face terms take every triangle with a plane at all
   * -- no sliver threshold, which would cost up to the sliver's width. The
   * candidates are visited in a fixed order and only a STRICTLY smaller one
   * replaces the witness, so the result is a function of (A, B). */
  double best = MFTL_DBL_MAX;

  plane_frame fa, fb;
  const bool has_a = frame_init(A, &fa);
  const bool has_b = frame_init(B, &fb);

  for (int i = 0; i < 3; ++i)
  {
    double bary[3], st[2], r[3], off[3];
    if (has_b)
    {
      frame_locate(&fb, A[i], st, r, bary);
      if (bary[0] > 0.0 && bary[1] > 0.0 && bary[2] > 0.0)
      {
        const double d_sq = frame_offset(&fb, st[1], r, off);
        if (d_sq < best)
        {
          best = d_sq;
          copy3(pa, A[i]);
          frame_point(&fb, st, pb);
        }
      }
    }
    if (has_a)
    {
      frame_locate(&fa, B[i], st, r, bary);
      if (bary[0] > 0.0 && bary[1] > 0.0 && bary[2] > 0.0)
      {
        const double d_sq = frame_offset(&fa, st[1], r, off);
        if (d_sq < best)
        {
          best = d_sq;
          frame_point(&fa, st, pa);
          copy3(pb, B[i]);
        }
      }
    }
  }

  /* Each edge serves three pairs: its vector and length once. */
  double ue[3][3], ve[3][3], ua[3], vc[3];
  for (int i = 0; i < 3; ++i)
  {
    const double *a0 = A[i], *a1 = A[(i + 1) % 3], *b0 = B[i], *b1 = B[(i + 1) % 3];
    for (int k = 0; k < 3; ++k)
    {
      ue[i][k] = a1[k] - a0[k];
      ve[i][k] = b1[k] - b0[k];
    }
    ua[i] = ue[i][0] * ue[i][0] + ue[i][1] * ue[i][1] + ue[i][2] * ue[i][2];
    vc[i] = ve[i][0] * ve[i][0] + ve[i][1] * ve[i][1] + ve[i][2] * ve[i][2];
  }
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      double s, t;
      const double d_sq = segments_kernel(A[i], ue[i], ua[i], B[j], ve[j], vc[j], &s, &t);
      if (d_sq < best)
      {
        best = d_sq;
        segment_point(A[i], A[(i + 1) % 3], ue[i], s, pa);
        segment_point(B[j], B[(j + 1) % 3], ve[j], t, pb);
      }
    }
  return best;
}
