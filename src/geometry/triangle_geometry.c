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
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Point-triangle closest point
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Closest point on the SEGMENT [a, b] to @p p.
 *
 * @param      p       Query point.
 * @param      a       Segment start.
 * @param      b       Segment end.
 * @param[out] out     Closest point.
 * @param[out] where   0 at @p a, 1 at @p b, 2 strictly inside.
 * @return Squared distance.
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
    out[k] = (*where == 1) ? b[k] : a[k] + s * ab[k];
    const double d = p[k] - out[k];
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
   * the test is on sin^2 alone. Below it the face formula's denominator is
   * noise and the triangle is treated as its three edges. */
  const double cx = ab[1] * ac[2] - ab[2] * ac[1];
  const double cy = ab[2] * ac[0] - ab[0] * ac[2];
  const double cz = ab[0] * ac[1] - ab[1] * ac[0];
  const double cross_sq = cx * cx + cy * cy + cz * cz;
  const double ab_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2];
  const double ac_sq = ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2];
  if (!(cross_sq > MFTL_VERYVERYSMALL * ab_sq * ac_sq)) return closest_on_degenerate(p, tri, q, f);

  /* Vertex region A. */
  const double ap[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  const double d1 = ab[0] * ap[0] + ab[1] * ap[1] + ab[2] * ap[2];
  const double d2 = ac[0] * ap[0] + ac[1] * ap[1] + ac[2] * ap[2];

  double w_ab = 0.0, w_ac = 0.0;  // closest = a + w_ab*ab + w_ac*ac

  if (d1 <= 0.0 && d2 <= 0.0)
  {
    *f = MFTL_TRI_FEATURE_VERTEX0;
  }
  else
  {
    /* Vertex region B. */
    const double bp[3] = {p[0] - b[0], p[1] - b[1], p[2] - b[2]};
    const double d3 = ab[0] * bp[0] + ab[1] * bp[1] + ab[2] * bp[2];
    const double d4 = ac[0] * bp[0] + ac[1] * bp[1] + ac[2] * bp[2];

    /* Vertex region C. */
    const double cp[3] = {p[0] - c[0], p[1] - c[1], p[2] - c[2]};
    const double d5 = ab[0] * cp[0] + ab[1] * cp[1] + ab[2] * cp[2];
    const double d6 = ac[0] * cp[0] + ac[1] * cp[1] + ac[2] * cp[2];

    const double vc = d1 * d4 - d3 * d2;
    const double vb = d5 * d2 - d1 * d6;
    const double va = d3 * d6 - d5 * d4;

    if (d3 >= 0.0 && d4 <= d3)
    {
      *f = MFTL_TRI_FEATURE_VERTEX1;
      w_ab = 1.0;
    }
    else if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
      *f = MFTL_TRI_FEATURE_EDGE0;
      w_ab = d1 / (d1 - d3);
    }
    else if (d6 >= 0.0 && d5 <= d6)
    {
      *f = MFTL_TRI_FEATURE_VERTEX2;
      w_ac = 1.0;
    }
    else if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
      *f = MFTL_TRI_FEATURE_EDGE2;
      w_ac = d2 / (d2 - d6);
    }
    else if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
      /* On BC: b + s*(c - b) = a + (1-s)*ab + s*ac. */
      const double s = (d4 - d3) / ((d4 - d3) + (d5 - d6));
      *f = MFTL_TRI_FEATURE_EDGE1;
      w_ab = 1.0 - s;
      w_ac = s;
    }
    else
    {
      const double denom = va + vb + vc;
      if (!(denom > 0.0)) return closest_on_degenerate(p, tri, q, f);
      const double inv = 1.0 / denom;
      *f = MFTL_TRI_FEATURE_FACE;
      w_ab = vb * inv;
      w_ac = vc * inv;
    }
  }

  /* Corners are returned bit-exactly, not as a + 1.0*ab (which would round). */
  switch (*f)
  {
    case MFTL_TRI_FEATURE_VERTEX0: q[0] = a[0], q[1] = a[1], q[2] = a[2]; break;
    case MFTL_TRI_FEATURE_VERTEX1: q[0] = b[0], q[1] = b[1], q[2] = b[2]; break;
    case MFTL_TRI_FEATURE_VERTEX2: q[0] = c[0], q[1] = c[1], q[2] = c[2]; break;
    default:
      for (int k = 0; k < 3; ++k) q[k] = a[k] + w_ab * ab[k] + w_ac * ac[k];
      break;
  }

  double d_sq = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    const double d = p[k] - q[k];
    d_sq += d * d;
  }
  return d_sq;
}
