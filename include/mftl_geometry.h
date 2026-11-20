/**
 * @file mftl_geometry.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_geometry
 *
 * @brief Periodic-aware geometry: minimum image, ownership lookup, triangle
 *        and vertex geometry, closest points, and signed distance on a lattice.
 *
 * Everything here is LOCAL (no MPI); the collective wrappers are in
 * mftl_ensemble.h. Positions are reached by index (MFTLVertexPos()), the
 * geometry arrays are written directly (`obj->tris.normal[3*t+k]`), and every
 * difference of two positions is taken under the minimum-image convention of
 * the object's domain. Theory: src/geometry/README.md.
 */

#ifndef MFTL_GEOMETRY_H
#define MFTL_GEOMETRY_H

#include <math.h>

#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

// -----------------------------------------------------------------------------
// 1. POSITION ACCESS
// -----------------------------------------------------------------------------

/**
 * @brief Read-only position of vertex @p v: three contiguous doubles.
 * @param obj The mesh.
 * @param v   Vertex index. Bounds-asserted in Debug.
 * @return Pointer to `{x, y, z}` inside the pool's SoA array. This is the
 *         AUTHORITATIVE position -- there is no second copy anywhere.
 */
static inline const double *MFTLVertexPos(const MFTL_Object *obj, int v)
{
  MFTL_ASSERT(v >= 0 && v < obj->verts.n);
  return &obj->verts.pos[3 * v];
}

/**
 * @brief Writable position of vertex @p v.
 *
 * @param obj The mesh.
 * @param v   Vertex index. Bounds-asserted in Debug.
 * @return Writable pointer to `{x, y, z}`.
 */
static inline double *MFTLVertexPosMut(MFTL_Object *obj, int v)
{
  MFTL_ASSERT(v >= 0 && v < obj->verts.n);
  return &obj->verts.pos[3 * v];
}

// -----------------------------------------------------------------------------
// 2. PERIODICITY
// -----------------------------------------------------------------------------

/**
 * @brief The shortest vector from @p p1 to @p p2 under the minimum-image convention.
 *
 * On each periodic axis
 *
 * \f[
 *   \Delta_k = (p_2 - p_1)_k - L_k\,\mathrm{nearbyint}\!\left(\frac{(p_2-p_1)_k}{L_k}\right),
 * \f]
 *
 * so \f$|\Delta_k| \le L_k/2\f$ for ANY displacement (the comparison form
 * `if (d > L/2) d -= L` is only correct for \f$|d| < 3L/2\f$). Ties at
 * \f$\pm L/2\f$ round to even. One multiply, one round and one fused
 * multiply-subtract per axis.
 *
 * @param      domain  Supplies the per-axis periodicity, sizes and reciprocals.
 * @param[out] vec_out The shortest displacement from @p p1 to @p p2.
 * @param      p1      Start position.
 * @param      p2      End position.
 */
static inline void MFTLGetMinimumImageVector(const MFTL_Domain *domain, double vec_out[3], const double p1[3], const double p2[3])
{
  vec_out[0] = p2[0] - p1[0];
  vec_out[1] = p2[1] - p1[1];
  vec_out[2] = p2[2] - p1[2];

  for (int i = 0; i < 3; ++i)
  {
    if (domain->is_periodic[i]) vec_out[i] -= domain->domain_size[i] * nearbyint(vec_out[i] * domain->inv_domain_size[i]);
  }
}

/**
 * @brief Wraps a position into the periodic domain box by at most one period.
 *
 * Correct when the position is less than one period outside the box, which a
 * position wrapped at every step always is. The compare-and-shift form beats a
 * branchless one (measured 1.34-1.37x) because the branches are perfectly
 * predicted and non-periodic axes do no work.
 *
 * @param domain Supplies the per-axis periodicity and bounds.
 * @param pos    Position, wrapped in place into \f$[\min, \max)\f$ on periodic
 *               axes. Non-periodic axes are untouched.
 */
static inline void MFTLApplyPeriodicity(const MFTL_Domain *domain, double pos[3])
{
  for (int i = 0; i < 3; ++i)
  {
    if (domain->is_periodic[i])
    {
      if (pos[i] >= domain->global_max[i])
        pos[i] -= domain->domain_size[i];
      else if (pos[i] < domain->global_min[i])
        pos[i] += domain->domain_size[i];
    }
  }
}

/**
 * @brief The rank whose Cartesian cell contains @p pos.
 *
 * Cell \f$i_k = \lfloor (x_k - \min_k)\,\mathrm{cuts}_k / L_k \rfloor\f$, clamped to
 * \f$[0, \mathrm{cuts}_k)\f$, then a `rank_map` lookup. @p pos must be WRAPPED:
 * coordinates outside the box are clamped to the boundary cell, a plausible
 * wrong answer on a periodic axis.
 *
 * This is the inner statement of redistribution's owner classification (once
 * per vertex and per triangle per redistributed step), so it carries nothing
 * conditional. The remesher's shifted cut planes deliberately do NOT live here --
 * adding the shift and its periodicity branch measured 1.70x slower (1.79 ->
 * 3.05 ns/entity); they reach redistribution through #MFTL_OwnerMap instead.
 *
 * @param domain The MPI/physical world.
 * @param pos    A position, already wrapped into the domain box.
 * @return The rank owning the cell containing @p pos.
 */
static inline int MFTLGetOwnerProc(const MFTL_Domain *domain, const double pos[3])
{
  int c[3];
  for (int k = 0; k < 3; ++k)
  {
    int i = (int) ((pos[k] - domain->global_min[k]) * domain->grid_factors[k]);
    if (i < 0) i = 0;
    if (i >= domain->cuts[k]) i = domain->cuts[k] - 1;
    c[k] = i;
  }
  return domain->rank_map[c[0] + c[1] * domain->cuts[0] + c[2] * domain->cuts[0] * domain->cuts[1]];
}

// -----------------------------------------------------------------------------
// 3. EDGE QUERIES
// -----------------------------------------------------------------------------

/**
 * @brief The seam-consistent vector along half-edge @p h, origin -> target.
 * @param      obj The mesh.
 * @param      h   Half-edge index.
 * @param[out] out The minimum-image difference \f$\vect{p}_{\mathrm{target}} -
 *                 \vect{p}_{\mathrm{origin}}\f$.
 */
static inline void MFTLGetEdgeVector(const MFTL_Object *obj, int h, double out[3])
{
  MFTLGetMinimumImageVector(obj->domain, out, MFTLVertexPos(obj, MFTLHeOrigin(obj, h)), MFTLVertexPos(obj, MFTLHeTarget(obj, h)));
}

/**
 * @brief Squared length of half-edge @p h. Prefer this wherever the sqrt can be avoided.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return \f$\norm{\vect{e}_h}^2\f$, seam-consistent. The remeshing length band
 *         is stored squared for exactly this reason.
 */
static inline double MFTLGetEdgeLengthSquared(const MFTL_Object *obj, int h)
{
  double e[3];
  MFTLGetEdgeVector(obj, h, e);
  return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
}

/**
 * @brief Length of half-edge @p h.
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return \f$\norm{\vect{e}_h}\f$, seam-consistent.
 */
static inline double MFTLGetEdgeLength(const MFTL_Object *obj, int h) { return MFTL_SQRT(MFTLGetEdgeLengthSquared(obj, h)); }

// -----------------------------------------------------------------------------
// 4. SINGLE-TRIANGLE PROPERTIES
// -----------------------------------------------------------------------------

/**
 * @brief Area, unit normal and centroid of triangle @p t.
 *
 * With minimum-image edges \f$\vect{e}_1 = \vect{p}_1 - \vect{p}_0\f$,
 * \f$\vect{e}_2 = \vect{p}_2 - \vect{p}_0\f$ and
 * \f$\vect{n} = \vect{e}_1 \times \vect{e}_2\f$:
 * \f$A = \tfrac12\norm{\vect{n}}\f$, \f$\hat{\vect{n}} = \vect{n}/\norm{\vect{n}}\f$
 * (zero for a degenerate triangle), \f$\vect{c} = \vect{p}_0 +
 * \tfrac13(\vect{e}_1+\vect{e}_2)\f$ wrapped into the box. Reads `tris.verts`
 * when current, the half-edge records otherwise. Loops over many triangles
 * should use #MFTL_FOREACH_TRIANGLE_GEOMETRY or MFTLComputeTriangleGeometry().
 *
 * @param      obj        The mesh.
 * @param      t          Triangle index.
 * @param[out] area_out   Area, or #MFTL_NULL.
 * @param[out] normal_out Unit normal (zero if degenerate), or #MFTL_NULL.
 * @param[out] center_out Centroid, WRAPPED into the domain; the true centroid
 *                        even for a triangle straddling a periodic face. Or
 *                        #MFTL_NULL.
 */
static inline void MFTLGetTriangleProperties(const MFTL_Object *obj, int t, double *area_out, double normal_out[3], double center_out[3])
{
  int v[3];
  if ((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u)
  {
    v[0] = obj->tris.verts[3 * t + 0];
    v[1] = obj->tris.verts[3 * t + 1];
    v[2] = obj->tris.verts[3 * t + 2];
  }
  else
  {
    MFTLGetTriangleVertices(obj, t, v);
  }

  const double *p0 = MFTLVertexPos(obj, v[0]);

  double e1[3];
  double e2[3];
  MFTLGetMinimumImageVector(obj->domain, e1, p0, MFTLVertexPos(obj, v[1]));
  MFTLGetMinimumImageVector(obj->domain, e2, p0, MFTLVertexPos(obj, v[2]));

  /* |e1 x e2| is twice the area. The magnitude is UNCLAMPED and the reciprocal
   * guarded by a test against zero, so every triangle with a representable
   * non-zero cross product gets an exactly unit normal and the true area (a
   * clamp would floor the area and shorten small normals; see
   * MFTLNormalizeVector()). The expression must be identical in every geometry
   * kernel: test_triangle_geometry_paths_are_bit_identical compares them. */
  double cr[3];
  cr[0] = e1[1] * e2[2] - e1[2] * e2[1];
  cr[1] = e1[2] * e2[0] - e1[0] * e2[2];
  cr[2] = e1[0] * e2[1] - e1[1] * e2[0];

  const double mag = MFTL_SQRT(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);

  if (area_out) *area_out = 0.5 * mag;

  if (normal_out)
  {
    const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
    normal_out[0] = cr[0] * inv;
    normal_out[1] = cr[1] * inv;
    normal_out[2] = cr[2] * inv;
  }

  if (center_out)
  {
    center_out[0] = p0[0] + (e1[0] + e2[0]) * MFTL_ONE_THIRD;
    center_out[1] = p0[1] + (e1[1] + e2[1]) * MFTL_ONE_THIRD;
    center_out[2] = p0[2] + (e1[2] + e2[2]) * MFTL_ONE_THIRD;
    MFTLApplyPeriodicity(obj->domain, center_out);
  }
}

/**
 * @brief The three corners of triangle @p t, mutually UNWRAPPED.
 *
 * Corner 0 is the stored (wrapped) position; corners 1 and 2 are corner 0 plus
 * the minimum-image edges, so they may lie just outside the domain box. A
 * triangle at \f$x = L-\epsilon\f$ and \f$x = +\epsilon\f$ thus comes out small
 * rather than domain-spanning. The input of the BVH, the spatial halo and the
 * signed-distance rasteriser.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index. Bounds-asserted in Debug.
 * @param[out] out Row k is corner k.
 */
static inline void MFTLGetTriangleCorners(const MFTL_Object *obj, int t, double out[3][3])
{
  MFTL_ASSERT(t >= 0 && t < obj->tris.n && "MFTLGetTriangleCorners: triangle index out of range");

  int v[3];
  if ((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u)
  {
    v[0] = obj->tris.verts[3 * t + 0];
    v[1] = obj->tris.verts[3 * t + 1];
    v[2] = obj->tris.verts[3 * t + 2];
  }
  else
  {
    MFTLGetTriangleVertices(obj, t, v);
  }

  const double *const p0 = MFTLVertexPos(obj, v[0]);
  double e1[3];
  double e2[3];
  MFTLGetMinimumImageVector(obj->domain, e1, p0, MFTLVertexPos(obj, v[1]));
  MFTLGetMinimumImageVector(obj->domain, e2, p0, MFTLVertexPos(obj, v[2]));

  for (int k = 0; k < 3; ++k)
  {
    out[0][k] = p0[k];
    out[1][k] = p0[k] + e1[k];
    out[2][k] = p0[k] + e2[k];
  }
}

/**
 * @brief Tight axis-aligned bounds of triangle @p t, from its unwrapped corners.
 *
 * No tolerance is added; callers inflate by their own. The box of a
 * seam-straddling triangle correctly extends outside the domain.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] lo  Lower corner of the box.
 * @param[out] hi  Upper corner of the box.
 */
static inline void MFTLGetTriangleBounds(const MFTL_Object *obj, int t, double lo[3], double hi[3])
{
  double c[3][3];
  MFTLGetTriangleCorners(obj, t, c);

  for (int k = 0; k < 3; ++k)
  {
    lo[k] = c[0][k];
    hi[k] = c[0][k];
  }
  for (int j = 1; j < 3; ++j)
  {
    for (int k = 0; k < 3; ++k)
    {
      if (c[j][k] < lo[k]) lo[k] = c[j][k];
      if (c[j][k] > hi[k]) hi[k] = c[j][k];
    }
  }
}

/**
 * @brief Area of triangle @p t.
 * @param obj The mesh.
 * @param t   Triangle index.
 * @return \f$\tfrac12 \norm{\vect{e}_1 \times \vect{e}_2}\f$, computed on the fly.
 *         For a whole-mesh pass use MFTLComputeTriangleGeometry() and read
 *         `tris.area`.
 */
static inline double MFTLGetTriangleArea(const MFTL_Object *obj, int t)
{
  double a = 0.0;
  MFTLGetTriangleProperties(obj, t, &a, MFTL_NULL, MFTL_NULL);
  return a;
}

/**
 * @brief Unit normal of triangle @p t.
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] out The unit normal, or the zero vector if the triangle is degenerate.
 */
static inline void MFTLGetTriangleNormal(const MFTL_Object *obj, int t, double out[3]) { MFTLGetTriangleProperties(obj, t, MFTL_NULL, out, MFTL_NULL); }

/**
 * @brief Wrapped centroid of triangle @p t.
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] out The centroid, wrapped back into the domain box.
 */
static inline void MFTLGetTriangleCenter(const MFTL_Object *obj, int t, double out[3]) { MFTLGetTriangleProperties(obj, t, MFTL_NULL, MFTL_NULL, out); }

/**
 * @brief Whether triangle @p t's cached `tris.center` still matches its vertices.
 *
 * Stale means off by more than \f$10^{-12}(|c_k| + L_k)\f$ on some axis k,
 * compared through the seam, with \f$c\f$ the recomputed centre and \f$L_k\f$
 * the domain length. The tolerance is relative, never absolute: rounding
 * differences between two evaluations are ulps of \f$|c_k|\f$, and a centre
 * error below \f$10^{-12}L_k\f$ can change the owner only of a triangle that
 * close to a cut plane. A NaN or garbage centre is stale.
 *
 * @param obj The mesh; needs a domain.
 * @param t   Triangle index.
 * @return `true` if the cached centre is current.
 */
static inline bool MFTLTriangleCentreIsCurrent(const MFTL_Object *obj, int t)
{
  double fresh[3];
  MFTLGetTriangleCenter(obj, t, fresh);

  /* Both are wrapped positions, so compare THROUGH the seam, not across it. */
  double d[3];
  MFTLGetMinimumImageVector(obj->domain, d, &obj->tris.center[3 * t], fresh);
  for (int k = 0; k < 3; ++k)
  {
    if (!(fabs(d[k]) <= 1e-12 * (fabs(fresh[k]) + obj->domain->domain_size[k]))) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/**
 * @brief Whether every triangle's cached `tris.center`, ghosts included, still
 *        matches its vertices (MFTLTriangleCentreIsCurrent()).
 *
 * The precondition of every migration that routes by centre (MFTLRedistribute(),
 * the remesher's shifted placements): a rank routes its ghosts too, and a stale
 * ghost centre makes two ranks disagree about a destination. A full geometry
 * pass -- use it inside `MFTL_ASSERT` only.
 *
 * @param obj The mesh.
 * @return `true` if every centre is current.
 */
static inline bool MFTLTriangleCentresAreCurrent(const MFTL_Object *obj)
{
  for (int t = 0; t < obj->tris.n; ++t)
  {
    if (!MFTLTriangleCentreIsCurrent(obj, t)) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

// -----------------------------------------------------------------------------
// 5. POINT QUERIES
// -----------------------------------------------------------------------------

/**
 * @brief Which feature of a triangle a closest point lies on (its Voronoi region).
 *
 * Corner k is `tri[k]`; edge k runs from corner k to corner (k+1) mod 3, i.e. it
 * is half-edge 3t+k of an MFTL triangle. A sign decided from a closest point on
 * an edge or a vertex needs the pseudo-normal of that feature (Bærentzen &
 * Aanæs 2005), not the face normal.
 */
typedef enum MFTL_TriangleFeature
{
  MFTL_TRI_FEATURE_FACE = 0,  ///< Strictly inside the triangle.
  MFTL_TRI_FEATURE_VERTEX0 = 1,  ///< Corner 0. Corner k is `MFTL_TRI_FEATURE_VERTEX0 + k`.
  MFTL_TRI_FEATURE_VERTEX1 = 2,  ///< Corner 1.
  MFTL_TRI_FEATURE_VERTEX2 = 3,  ///< Corner 2.
  MFTL_TRI_FEATURE_EDGE0 = 4,  ///< Interior of edge 0 (corner 0 -> 1). Edge k is `MFTL_TRI_FEATURE_EDGE0 + k`.
  MFTL_TRI_FEATURE_EDGE1 = 5,  ///< Interior of edge 1 (corner 1 -> 2).
  MFTL_TRI_FEATURE_EDGE2 = 6  ///< Interior of edge 2 (corner 2 -> 0).
} MFTL_TriangleFeature;

/**
 * @brief Closest point of a (closed, solid) triangle to a point, and its feature.
 *
 * Ericson's region test (*Real-Time Collision Detection*, §5.1.5): dot products
 * select one of the seven Voronoi regions and only that region's formula is
 * evaluated; corners are returned bit-exactly. The three "which side of an
 * edge's line" decisions and the face point come from a frame orthogonalised on
 * the longest edge (src/geometry/README.md, @ref geom_closest_accuracy), not
 * from Ericson's 2x2 determinants of dot products, so the distance is accurate
 * to rounding -- a few ulps of the triangle's size -- however thin the triangle.
 *
 * A triangle with \f$|\vect{ab}\times\vect{ac}|^2 \le 10^{-15}|\vect{ab}|^2|\vect{ac}|^2\f$
 * (a scale-free \f$\sin^2\f$ test at corner 0) has no face region and is
 * answered as the nearest of its three edges: its face normal, which a signed
 * distance would read, is not trustworthy there. That answer can exceed the
 * true distance by up to the triangle's height over its longest edge, at most
 * \f$\sqrt{10^{-15}}\f$ of its two edges at corner 0;
 * MFTLClosestPointsBetweenTriangles() has no such threshold.
 *
 * @param      p       Query point.
 * @param      tri     The three corners, in one periodic image (the caller
 *                     applies any minimum-image shift).
 * @param[out] closest The closest point, or #MFTL_NULL.
 * @param[out] feature Its feature, or #MFTL_NULL.
 * @return The SQUARED distance from @p p to the triangle, evaluated from local
 *         differences (not from @p closest), so it keeps its accuracy far from
 *         the origin.
 *
 * @rank_local Pure function.
 */
double MFTLClosestPointOnTriangle(const double p[3], const double tri[3][3], double closest[3], MFTL_TriangleFeature *feature);

/**
 * @brief Closest points of two closed segments, and their squared distance.
 *
 * The stationary point of \f$\lVert P_0 + s\vect{u} - Q_0 - t\vect{v}\rVert^2\f$
 * by Cramer's rule with the determinant written as \f$|\vect{u}\times\vect{v}|^2\f$
 * and the numerator as a triple product, then Ericson's clamping cascade
 * (§5.1.9). The Gram form \f$ac - b^2\f$ cancels catastrophically for nearly
 * parallel segments and is not used; nor is a parallelism threshold. The
 * distance is accurate to rounding (a few ulps of the configuration's size) at
 * every angle, and a zero-length segment is a point. This is the one
 * segment-segment routine of the library: MFTLClosestPointsBetweenTriangles()
 * and the contact predicates (MFTLTriTriProximity()) both use it.
 *
 * @param      p0 First endpoint of segment P.
 * @param      p1 Second endpoint of segment P.
 * @param      q0 First endpoint of segment Q.
 * @param      q1 Second endpoint of segment Q.
 * @param[out] c1 The witness on P (an endpoint bit-exactly), or #MFTL_NULL.
 * @param[out] c2 The witness on Q, or #MFTL_NULL.
 * @return The squared distance, evaluated from local differences.
 * @rank_local Pure function.
 */
double MFTLClosestPointsOnSegments(const double p0[3], const double p1[3], const double q0[3], const double q1[3], double c1[3], double c2[3]);

/**
 * @brief Closest points of two DISJOINT closed triangles, and their squared
 *        distance.
 *
 * The minimum is attained between two edges or at a corner of one triangle over
 * the interior of the other: nine MFTLClosestPointsOnSegments() terms and six
 * orthogonal projections (in the frame of MFTLClosestPointOnTriangle(), but with
 * no sliver threshold -- a triangle is skipped as a face only when its corners
 * are numerically collinear, and its edges then cover it). The distance is
 * accurate to rounding for every shape, including slivers and zero-area
 * triangles. When several pairs are equally close -- two parallel faces -- the
 * first in a fixed order is returned, so the result is a pure function of the
 * arguments. MFTLTriTriDistanceSq() and MFTLTriTriProximity() are this
 * function behind MFTLTriTriIntersect().
 *
 * For triangles that INTERSECT the result is the closest pair of boundary
 * points, not zero: test MFTLTriTriIntersect() first when that matters.
 *
 * @param      A  Corners of the first triangle.
 * @param      B  Corners of the second, in the same periodic image.
 * @param[out] pa The witness on @p A.
 * @param[out] pb The witness on @p B.
 * @return \f$|p_b - p_a|^2\f$, evaluated from local differences.
 * @rank_local Pure function.
 */
double MFTLClosestPointsBetweenTriangles(const double A[3][3], const double B[3][3], double pa[3], double pb[3]);

// -----------------------------------------------------------------------------
// 6. BULK GEOMETRY PASSES
// -----------------------------------------------------------------------------

/**
 * @brief Writes `tris.area`, `tris.normal` and `tris.center` for the first
 *        @p count triangles.
 *
 * With current `tris.verts` it streams compact indices with periodicity
 * unswitched outside the loop; before derived connectivity exists it walks the
 * half-edge records. The two paths are bit-identical
 * (`test_triangle_geometry_paths_are_bit_identical`).
 *
 * @param obj   The mesh.
 * @param count `obj->tris.n_local` for owned triangles, `obj->tris.n` to include
 *              ghosts; only the latter sets #MFTL_DERIVED_GEOMETRY.
 *
 * @warning Before a redistribution the count must be `obj->tris.n`: ghosts are
 *          routed by their cached centre, and a stale ghost centre makes two
 *          ranks disagree about a destination (an owned triangle then silently
 *          loses an edge neighbour). MFTLRedistribute() recomputes a body whose
 *          bit is clear rather than trust it.
 *
 * @rank_local MFTLEnsembleUpdate() is the collective counterpart that also
 *             reduces quality statistics.
 */
void MFTLComputeTriangleGeometry(MFTL_Object *obj, int count);

/**
 * @brief Which vertex normal MFTLComputeVertexNormals() and
 *        MFTLComputeVertexNormalAt() compute.
 *
 * One choice on two levels (src/geometry/README.md, "Vertex normals"):
 *
 * - FROM THE TRIANGLES, a weighted sum of the face normals around the vertex:
 *   #MFTL_NORMAL_WEIGHT_AREA or #MFTL_NORMAL_WEIGHT_ANGLE. Cheap, and
 *   first-order accurate on an irregular mesh.
 * - FITTED, #MFTL_NORMAL_WEIGHT_FIT: the normal of a one-ring paraboloid,
 *   second-order accurate; one small least-squares solve per vertex.
 *
 * Where the library uses which: AREA wherever a vertex MOVES (the remesher's
 * smoother, its placement fit and its projection's orientation test, the
 * volume corrections),
 * ANGLE wherever inside and outside are DECIDED (the signed distance, topology
 * change) and for the remesher's Phong reference, FIT for curvature and the
 * fitted-normal surface tension. Max's (1999) weights were measured and are
 * not offered (docs/analysis.md §16.9).
 */
typedef enum MFTL_NormalWeighting
{
  /**
   * Weight each face by its AREA: \f$\vect{N}_v \propto \sum_{f \ni v}
   * (\vect{e}_1\times\vect{e}_2)_f = 2\sum_f A_f\hat{\vect{n}}_f\f$, free given
   * the cross product (6-7 ns per vertex). On a closed fan this is the
   * direction of the gradient of the enclosed volume, \f$\partial
   * V/\partial\vect{x}_v = \tfrac16\sum_f(\vect{e}_1\times\vect{e}_2)_f\f$:
   * the direction in which the vertex changes the volume fastest, and the
   * normal of the plane in which it moves without changing the volume to first
   * order. Hence the volume corrections move along it and the remesher's
   * smoother moves in that plane; any other normal there makes the smoother
   * leak volume into noise (LeVeque, uniform sizing, refinement 4: angle
   * normals end 25 % of the radius off the exact surface,
   * src/remeshing/README.md). It is also the partner \f$\vect{G}\f$ of the
   * default surface-tension force in a balanced-force curvature
   * (src/geometry/README.md, "Surface tension"). Its pseudo-normal sign is
   * NOT exact at a spiky vertex: use ANGLE for inside/outside.
   */
  MFTL_NORMAL_WEIGHT_AREA = 0,

  /**
   * Weight each face by its INTERIOR ANGLE at the vertex (Thürmer & Wüthrich
   * 1998): \f$\vect{N}_v \propto \sum_f \alpha_f(v)\,\hat{\vect{n}}_f\f$,
   * \f$\alpha = \operatorname{atan2}(\norm{\vect{e}_1\times\vect{e}_2},
   * \vect{e}_1\cdot\vect{e}_2)\f$ (well conditioned at every angle, unlike
   * `acos`). Invariant under re-triangulation of the fan, and the only
   * weighting for which \f$\sign((\vect{x}-\vect{v})\cdot\vect{N}_v)\f$ is
   * provably the correct inside/outside test when \f$\vect{v}\f$ is the
   * closest surface point to \f$\vect{x}\f$ (Bærentzen & Aanæs 2005): the
   * signed distance and topology change use it, and the remesher's Phong
   * reference, where it measured best. Three `atan2` per face (40-66 ns per
   * vertex); degenerate faces contribute nothing.
   */
  MFTL_NORMAL_WEIGHT_ANGLE = 1,

  /**
   * Not a weighting of face normals: the normal at the vertex of the
   * least-squares paraboloid through the vertex and its one-ring POSITIONS,
   * in the frame of the area-weighted normal (MFTLFitHeightParaboloid(),
   * lengths in units of the ring's RMS edge), \f$\vect{N}_v \propto \vect{n} -
   * c_1\vect{t}_1 - c_2\vect{t}_2\f$. SECOND-order accurate on an irregular
   * mesh, where any weighting of face normals is first-order: 3-27 times
   * more accurate than the best weighting measured on remeshed, graded and
   * marching-cubes ellipsoids and tori (src/geometry/README.md, "Vertex
   * normals"). Which weighting gives the frame does not change the result
   * (three digits), so it is the cheapest. The normals #MFTL_CURVATURE_FIT is
   * meant to read. A vertex with fewer than three neighbours, a fit that
   * cannot determine its slope, or a fitted normal more than 75 degrees from
   * the frame keeps the area normal. One small QR per vertex (about 240 ns).
   */
  MFTL_NORMAL_WEIGHT_FIT = 2
} MFTL_NormalWeighting;

/**
 * @brief Unit vertex normals for every vertex of the pool, into `verts.normal`.
 *
 * One flat pass over ALL triangles (ghosts included) scatters each weighted face
 * normal into its three corners, then one pass normalises; a vertex without
 * incident triangles gets the zero vector. Requires current `tris.verts`.
 *
 * On a distributed mesh an OWNED vertex is exact without communication,
 * because the redistribution halo is vertex-complete (every triangle incident
 * to an owned vertex is present locally). A GHOST vertex misses its far-side
 * faces and holds a normalised PARTIAL sum -- plausible-looking and wrong.
 * MFTLEnsembleUpdateVertexNormals() adds the exchange that makes ghosts exact.
 *
 * #MFTL_NORMAL_WEIGHT_FIT is a one-ring quantity too, so the same holds for it.
 *
 * @param obj       The mesh; `verts.normal` is overwritten for every vertex.
 * @param weighting Which normal (#MFTL_NormalWeighting).
 *
 * @rank_local
 */
void MFTLComputeVertexNormals(MFTL_Object *obj, MFTL_NormalWeighting weighting);

/**
 * @brief The unit normal of ONE vertex, by the rule of
 *        MFTLComputeVertexNormals(), from a walk of its fan. Writes nothing to
 *        the pool.
 *
 * For a caller that edits the mesh between two normals (the remesher's
 * placement fit): a whole-pool pass is stale after the first edit, and a fan
 * walk for every vertex is 3.4 times slower than the whole-pool scatter. The
 * weights and the fit are the same code; the result differs from
 * MFTLComputeVertexNormals()'s only by the order of the sum. The walk goes both
 * ways from `out_he_idx`, so an open fan (a mesh rim, or a ghost's partial
 * fan) is summed as far as it is present. Needs the half-edges and
 * `out_he_idx`, not `tris.verts`.
 *
 * @param      obj       The mesh.
 * @param      v         The vertex, `[0, verts.n)`.
 * @param      weighting Which normal (#MFTL_NormalWeighting).
 * @param[out] n         The unit normal; zero for a vertex without area.
 * @return `true` if the fan closed, i.e. the normal is that of the whole fan.
 *
 * @rank_local
 */
bool MFTLComputeVertexNormalAt(const MFTL_Object *obj, int v, MFTL_NormalWeighting weighting, double n[3]);

// -----------------------------------------------------------------------------
// 7. SIGNED DISTANCE ON A LATTICE  (see src/geometry/signed_distance.c)
// -----------------------------------------------------------------------------

/**
 * @brief Value of a lattice node no triangle came within the cutoff of: sign and
 *        distance unknown.
 *
 * The largest double, so it loses every MFTLSdfMerge() comparison; marching
 * cubes skips any cell with such a corner. No cell crossed by the surface can
 * have one when the cutoff exceeds the cell diagonal
 * (MFTLRasterizeSignedDistance()).
 */
#define MFTL_SDF_UNKNOWN MFTL_DBL_MAX

/**
 * @struct MFTL_SdfGrid
 * @brief A dense WINDOW onto a (possibly periodic) Cartesian lattice of signed
 *        distances.
 *
 * The LATTICE is global: node g on axis k sits at `origin[k] + g*spacing[k]`,
 * and on a periodic axis g and g + `n_global[k]` are the same node. Integer node
 * coordinates are therefore an identity every rank computes identically, which
 * lets a distributed marching cubes name a surface vertex by its lattice EDGE
 * with no negotiation.
 *
 * The WINDOW is what one caller stores: `dims[k]` consecutive nodes from global
 * index `lo[k]` (which may be negative or past the end on a periodic axis;
 * indices are taken modulo `n_global`, and `dims <= n_global`).
 * MFTLSdfGridInit() makes the window the whole lattice, non-periodic.
 *
 * `phi` is negative INSIDE (opposite the outward normals), positive outside,
 * #MFTL_SDF_UNKNOWN where nothing was rasterised; x varies fastest.
 *
 * `cross` is OPTIONAL (MFTLSdfGridEnableCrossings()): for every node and axis
 * k, where the lattice edge from that node to its neighbour along +k crosses
 * the rasterised surface (MFTLRasterizeEdgeCrossings()), as a fraction of the
 * edge, or #MFTL_SDF_CROSS_NONE / #MFTL_SDF_CROSS_MANY. Marching cubes puts a
 * vertex there instead of interpolating `phi` (MFTLExtractLevelSet()).
 */
typedef struct MFTL_SdfGrid
{
  double origin[3];  ///< Coordinate of GLOBAL lattice node 0 on each axis.
  double spacing[3];  ///< Node spacing per axis. May differ between axes.
  int n_global[3];  ///< Nodes in the whole lattice per axis; the wrap period on a periodic axis.
  bool periodic[3];  ///< Whether global indices wrap modulo `n_global` on this axis.
  int lo[3];  ///< Global (unwrapped) index of the window's node 0.
  int dims[3];  ///< Nodes the window stores per axis.
  double *phi;  ///< [dims[0]*dims[1]*dims[2]] values, x fastest.
  double *cross;  ///< #MFTL_NULL, or [3 * nodes] edge crossings, slot `3 * node + k` (MFTLSdfCrossMerge()).
} MFTL_SdfGrid;

/** @brief An edge crossing slot no triangle crossed. */
#define MFTL_SDF_CROSS_NONE (-1.0)

/** @brief An edge crossing slot two or more triangles crossed: no single crossing to use. */
#define MFTL_SDF_CROSS_MANY (-2.0)

/**
 * @brief Combines two contributions to one edge-crossing slot.
 *
 * NONE is the identity, two crossings make MANY, and MANY absorbs everything:
 * the slot holds a fraction in [0, 1] exactly when ONE triangle crossed the
 * edge. Commutative and associative, so the merged slots do not depend on the
 * order in which triangles or ranks contribute -- the property MFTLSdfMerge()
 * gives `phi`. A triangle met twice (a lattice line through an edge both its
 * triangles count) becomes MANY, which falls back to interpolation: never a
 * wrong crossing, at worst no crossing.
 *
 * @param current   The slot so far.
 * @param candidate A new contribution: a fraction, NONE or MANY.
 * @return The merged slot.
 */
static inline double MFTLSdfCrossMerge(double current, double candidate)
{
  if (candidate == MFTL_SDF_CROSS_NONE) return current;
  if (current == MFTL_SDF_CROSS_NONE) return candidate;
  return MFTL_SDF_CROSS_MANY;
}

/**
 * @brief A standalone grid: `dims` nodes from `origin`, no periodicity, every
 *        value #MFTL_SDF_UNKNOWN.
 *
 * @param[out] grid    The grid. Release with MFTLSdfGridFree().
 * @param      origin  Coordinate of node (0,0,0).
 * @param      spacing Node spacing per axis, each > 0.
 * @param      dims    Nodes per axis, each >= 2.
 * @rank_local
 */
void MFTLSdfGridInit(MFTL_SdfGrid *grid, const double origin[3], const double spacing[3], const int dims[3]);

/**
 * @brief A window onto a global lattice, every value #MFTL_SDF_UNKNOWN.
 *
 * @param[out] grid     The grid. Release with MFTLSdfGridFree().
 * @param      origin   Coordinate of GLOBAL node 0 per axis.
 * @param      spacing  Node spacing per axis, each > 0.
 * @param      n_global Nodes in the whole lattice per axis (the period, when periodic).
 * @param      periodic Per-axis wrap, or #MFTL_NULL for none.
 * @param      lo       Global index of the window's first node per axis.
 * @param      dims     Nodes stored per axis; at most `n_global` on a periodic axis.
 * @rank_local
 */
void MFTLSdfGridInitWindow(MFTL_SdfGrid *grid, const double origin[3], const double spacing[3], const int n_global[3], const bool periodic[3], const int lo[3],
                           const int dims[3]);

/**
 * @brief Releases the values. Safe on a zeroed or already-freed grid.
 * @param grid The grid; `phi` and `cross` are freed and set to #MFTL_NULL.
 */
void MFTLSdfGridFree(MFTL_SdfGrid *grid);

/**
 * @brief Gives the grid its edge-crossing slots, every one #MFTL_SDF_CROSS_NONE.
 *        Does nothing if it has them.
 * @param grid The grid.
 * @rank_local
 */
void MFTLSdfGridEnableCrossings(MFTL_SdfGrid *grid);

/**
 * @brief Window index of global node @p g on axis @p k, or -1 if the window
 *        does not hold it.
 * @param grid The grid.
 * @param k    Axis.
 * @param g    Global node index, unwrapped: any integer on a periodic axis.
 * @return The local index in `[0, dims[k])`, or -1.
 */
static inline int MFTLSdfGridLocal(const MFTL_SdfGrid *grid, int k, int g)
{
  int l = g - grid->lo[k];
  if (grid->periodic[k])
  {
    const int n = grid->n_global[k];
    l %= n;
    if (l < 0) l += n;
  }
  return (l >= 0 && l < grid->dims[k]) ? l : -1;
}

/**
 * @brief Coordinate of global node @p g on axis @p k, in the frame @p g lives in.
 *
 * Unwrapped on purpose: node `g` and node `g + n_global` have coordinates one
 * period apart, which is what lets a triangle near the seam measure its distance
 * to a node in its OWN periodic image.
 *
 * @param grid The grid.
 * @param k    Axis.
 * @param g    Global node index, unwrapped.
 * @return `origin[k] + g * spacing[k]`.
 */
static inline double MFTLSdfGridCoord(const MFTL_SdfGrid *grid, int k, int g) { return grid->origin[k] + (double) g * grid->spacing[k]; }

/**
 * @brief Combines two contributions to one node: the smaller magnitude wins, a
 *        tie goes to the NEGATIVE value.
 *
 * A total order, hence commutative and associative: the merged field does not
 * depend on the order in which triangles -- or ranks -- contribute, which makes
 * a distributed rasterisation bit-identical to a serial one. #MFTL_SDF_UNKNOWN
 * is the identity.
 *
 * @param current   The node's value so far.
 * @param candidate A new contribution.
 * @return The merged value.
 */
static inline double MFTLSdfMerge(double current, double candidate)
{
  const double ac = fabs(current), an = fabs(candidate);
  if (an < ac) return candidate;
  if (an == ac && candidate < current) return candidate;
  return current;
}

/**
 * @brief Rasterises the signed distance to triangles `[0, n_tris)` onto every
 *        window node within @p cutoff of one of them.
 *
 * For each triangle T and each node \f$\vect{x}\f$ in T's box dilated by the
 * cutoff, the closest point \f$\vect{q}\f$ and its feature are found with
 * MFTLClosestPointOnTriangle(); the value is
 * \f$\pm\norm{\vect{x}-\vect{q}}\f$ with the sign of
 * \f$(\vect{x}-\vect{q})\cdot\vect{n}\f$, \f$\vect{n}\f$ the pseudo-normal of
 * the feature: the face normal; for an edge the sum of the two incident unit
 * face normals (T's alone at a mesh rim); for a vertex `verts.normal`, which
 * must hold ANGLE-weighted normals. Values are merged with MFTLSdfMerge(). For a
 * closed, consistently oriented, non-self-intersecting surface this yields the
 * exact signed distance (Bærentzen & Aanæs 2005); area-weighted vertex
 * normals can give the wrong sign near thin, sharply folded sheets.
 *
 * **Cutoff.** Nodes farther than @p cutoff from every triangle stay
 * #MFTL_SDF_UNKNOWN. Every corner of a lattice cell crossed by the surface lies
 * within one cell diagonal of it, so a cutoff above the diagonal
 * (\f$\sqrt3\,\Delta x\f$ on a cubic lattice) provides every value marching
 * cubes needs, with no flood fill and no global pass.
 *
 * **Periodicity.** Triangles are unwrapped around corner 0 and reach nodes by
 * UNWRAPPED global index, so each distance is measured in the triangle's own
 * image and the window's modular indexing stores it where it belongs.
 *
 * @param grid    The grid; values are MERGED, so several objects or ranks may
 *                rasterise into one.
 * @param obj     The mesh: current `tris.verts`, and angle-weighted
 *                `verts.normal` for every corner of the triangles used, ghosts
 *                included (MFTLEnsembleUpdateVertexNormals() with
 *                #MFTL_NORMAL_WEIGHT_ANGLE on a distributed mesh).
 * @param n_tris  Rasterise `[0, n_tris)`; `tris.n_local` in a distributed
 *                rasterisation, so no triangle is counted by two ranks.
 * @param cutoff  Distance within which nodes are written; > 0.
 * @return The number of node updates that changed a value.
 * @rank_local No MPI.
 */
long MFTLRasterizeSignedDistance(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris, double cutoff);

/**
 * @brief MFTLRasterizeSignedDistance() with a WIDER cutoff on marked nodes:
 *        every node takes the merge over exactly the triangles within its own
 *        cutoff.
 *
 * A node flagged in @p wide_nodes is written by every triangle within
 * @p wide_cutoff of it, any other node by every triangle within @p cutoff. Only
 * triangles flagged in @p wide_tris look past @p cutoff, so the caller must flag
 * EVERY triangle within @p wide_cutoff of a flagged node -- flagging more costs
 * time and changes nothing. With that, each node's value is the one the plain
 * rasterisation at its own cutoff would give, bit for bit: the nearest triangle
 * of a node that gets a value is always among those it is merged over.
 *
 * Why a second band: a cutoff exists to give marching cubes every corner of
 * every cell the zero level set crosses. Where nothing will be added to the
 * field one cell diagonal suffices; where a bias of up to \f$\varepsilon\f$ will
 * move the level set, the band must be \f$\varepsilon\f$ wider. Paying the wide
 * band only near the bias is what MFTLEnsembleReconstructBodies() uses this for.
 *
 * @param grid        The grid; values are MERGED as by MFTLRasterizeSignedDistance().
 * @param obj         The mesh, with the same requirements.
 * @param n_tris      Rasterise `[0, n_tris)`.
 * @param cutoff      Distance within which every node is written; > 0.
 * @param wide_cutoff Distance within which a flagged node is written; >= @p cutoff.
 * @param wide_nodes  `[dims[0]*dims[1]*dims[2]]` nonzero where @p wide_cutoff
 *                    applies, x fastest; #MFTL_NULL flags every node.
 * @param wide_tris   `[n_tris]` nonzero for every triangle that may come within
 *                    @p wide_cutoff of a flagged node; #MFTL_NULL flags none, which
 *                    is MFTLRasterizeSignedDistance() at @p cutoff.
 * @return The number of node updates that changed a value.
 * @rank_local No MPI.
 */
long MFTLRasterizeSignedDistanceBanded(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris, double cutoff, double wide_cutoff,
                                       const unsigned char *wide_nodes, const unsigned char *wide_tris);

/**
 * @brief Records where the lattice edges cross triangles `[0, n_tris)`, merged
 *        into `grid->cross` with MFTLSdfCrossMerge().
 *
 * For each triangle and axis k, every lattice line along k through the
 * triangle's projected box is tested against the triangle in the other two
 * coordinates; a line inside it crosses it at one coordinate along k, which
 * names the edge it lies on (its lower node) and the fraction along it. The
 * slot of the edge's lower node is written when that node is in the window.
 *
 * Marching cubes interpolates `phi` linearly along an edge, which puts a vertex
 * O(dx^2 kappa) off a curved surface, and inwards on a convex one; a recorded
 * crossing is ON the rasterised surface (Wojtan et al.'s type-1 vertices). A
 * triangle parallel to the axis has no crossing; a line through a shared edge
 * or corner may be counted by both triangles, which makes the slot MANY and the
 * vertex fall back to interpolation. Triangles are unwrapped around corner 0
 * and lines reached by unwrapped index, as in MFTLRasterizeSignedDistance().
 *
 * @param grid   The grid; MFTLSdfGridEnableCrossings() is called if needed.
 * @param obj    The mesh, with current `tris.verts`.
 * @param n_tris Use `[0, n_tris)`; `tris.n_local` in a distributed run.
 * @return The number of crossings recorded in the window.
 * @rank_local No MPI.
 */
long MFTLRasterizeEdgeCrossings(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris);

// -----------------------------------------------------------------------------
// 8. EXTENT ON A PERIODIC AXIS  (see src/geometry/periodic_extent.c)
// -----------------------------------------------------------------------------

/**
 * @brief Doubles one body's extent occupies in a per-body reduction: four frames
 *        times three axes times (lower bound, NEGATED upper bound), every slot a
 *        MIN.
 *
 * Frame f is cut at `global_min + {0, L/2, L/4, 3L/4}[f]` on each periodic axis;
 * slots `[6f, 6f+3)` hold the lower bounds and `[6f+3, 6f+6)` the negated upper
 * bounds. A non-periodic axis uses frame 0 only and leaves the rest at the MIN
 * identity.
 */
#define MFTL_PERIODIC_EXTENT_SLOTS 24

/**
 * @brief Fraction of the period from which an extent counts as COVERING the
 *        axis: a body at least this wide is treated as wrapping it.
 *
 * With four equally spaced cuts, a compact body narrower than \f$3L/4\f$ always
 * leaves one cut in its gap and is measured exactly; a wider one is measured
 * exactly at some placements only. Treating every extent of \f$3L/4\f$ or more as
 * covering makes the verdict depend on the size alone, never on the placement,
 * and keeps it a quarter period away from the round-off at \f$L\f$, where an
 * exact cover and a sliver of a gap differ in the last bit.
 */
#define MFTL_PERIODIC_EXTENT_COVER_FRACTION 0.75

/**
 * @brief Fraction of the period a body that does not wrap a periodic axis must
 *        stay below: THE PERIODIC EXTENT RULE.
 *
 * On every periodic axis a body either wraps the axis or is narrower than half
 * the period along it. Physically, a body wider than \f$L/2\f$ is closer to its
 * own periodic image than to its own far side. Numerically, the library
 * measures every body exactly up to #MFTL_PERIODIC_EXTENT_COVER_FRACTION, so the
 * band \f$[L/2, 3L/4)\f$ is where a growing body is still SEEN breaking the rule
 * before it becomes indistinguishable from a wrapping one.
 * MFTLEnsembleCheckPeriodicExtents() enforces it.
 */
#define MFTL_PERIODIC_EXTENT_RULE_FRACTION 0.5

/**
 * @struct MFTL_PeriodicExtent
 * @brief A body's extent as a centre and a half-width per axis, measured on the
 *        circle on a periodic axis.
 *
 * A `(lo, hi)` box cannot describe a body straddling a periodic seam: its stored
 * coordinates span the whole domain. A centre and a half-width can, and the
 * centre is the point every vertex of the body is within half a period of, so
 * `x - L nearbyint((x - centre) / L)` lifts the whole body into one image.
 */
typedef struct MFTL_PeriodicExtent
{
  double centre[3];  ///< Centre per axis; may lie outside the domain box on a periodic axis.
  double half[3];  ///< Half-width per axis.
  bool covers[3];  ///< Periodic axis covered: the body wraps it, or is at least #MFTL_PERIODIC_EXTENT_COVER_FRACTION of it wide.
  bool empty;  ///< No triangle contributed.
} MFTL_PeriodicExtent;

/**
 * @brief Sets an accumulator to the identity of its MIN reduction.
 *
 * @param[out] acc #MFTL_PERIODIC_EXTENT_SLOTS doubles, every one #MFTL_DBL_MAX.
 * @rank_local
 */
void MFTLPeriodicExtentInit(double acc[MFTL_PERIODIC_EXTENT_SLOTS]);

/**
 * @brief Adds one triangle to an extent accumulator.
 *
 * Frame 0 takes the corners as they are. Frame f > 0 moves the triangle
 * RIGIDLY by the period multiple that brings corner 0, shifted by the frame's
 * offset, into the domain box -- so a triangle is never torn, and a body
 * contiguous in that frame is measured by its true extent.
 *
 * @param      dom Supplies the periodicity, bounds and sizes.
 * @param      T   The corners, in one image (MFTLGetTriangleCorners()).
 * @param[in,out] acc The accumulator.
 * @rank_local
 */
void MFTLPeriodicExtentAddTriangle(const MFTL_Domain *dom, const double T[3][3], double acc[MFTL_PERIODIC_EXTENT_SLOTS]);

/**
 * @brief Adds every OWNED triangle of @p obj to an extent accumulator.
 *
 * Owned only, so that a reduction over the ranks holding the body sees each
 * triangle once; a ghost is contributed by its owner.
 *
 * @param      obj The mesh; needs no derived state.
 * @param[in,out] acc The accumulator.
 * @rank_local
 */
void MFTLPeriodicExtentAddObject(const MFTL_Object *obj, double acc[MFTL_PERIODIC_EXTENT_SLOTS]);

/**
 * @brief Turns a (reduced) accumulator into a centre, a half-width and a
 *        covering flag per axis.
 *
 * On a periodic axis the frames cut at 0 and \f$L/2\f$ are compared first and
 * the tighter wins (frame 0 on a tie); only when neither measures less than
 * #MFTL_PERIODIC_EXTENT_COVER_FRACTION are the quarter frames consulted. A cut
 * lying INSIDE a body's projection makes that frame measure at least \f$L\f$, so
 * the smallest frame extent below \f$L\f$ is always the exact one. A body is
 * `covers[k]` when no frame measures it narrower than
 * #MFTL_PERIODIC_EXTENT_COVER_FRACTION; its centre and half-width are then
 * those of the first comparison and carry no meaning.
 *
 * @param      dom Supplies the periodicity and sizes.
 * @param      acc An accumulator filled by MFTLPeriodicExtentAddTriangle(),
 *                 typically MIN-reduced over the ranks holding the body.
 * @param[out] out The extent.
 * @rank_local Pure function: bit-identical on every rank given identical input.
 */
void MFTLPeriodicExtentResolve(const MFTL_Domain *dom, const double acc[MFTL_PERIODIC_EXTENT_SLOTS], MFTL_PeriodicExtent *out);

// -----------------------------------------------------------------------------
// 9. BODY FLUX INTEGRALS  (see src/geometry/body_integrals.c)
// -----------------------------------------------------------------------------
//
// The local half of MFTLEnsembleBodyIntegrals(): sums over one rank's OWNED
// triangles, to be added up over a body's holders. A body's volume and centroid
// are fluxes of polynomial fields through its surface; the region they measure
// is the one the surface bounds alone (closed) or together with the walls its
// rims lie on.

/**
 * @brief How far, relative to the domain size, a rim vertex may lie from a wall
 *        plane and still be ON it: the tolerance the topology change and the
 *        examples snap rims to.
 */
#define MFTL_BODY_WALL_TOL 1e-8

/** @brief Values MFTLBodyFluxMoments() produces: the volume and three moments. */
#define MFTL_BODY_MOMENT_SLOTS 4

/**
 * @brief Where the rim of @p obj lies: which domain walls, and whether every rim
 *        edge lies on one.
 *
 * Over the mesh-boundary half-edges of OWNED triangles. A rim edge is on a wall
 * when both its ends are within #MFTL_BODY_WALL_TOL of the same non-periodic
 * wall plane; the wall mask records the walls the edges' target vertices touch.
 *
 * @param      obj      The mesh.
 * @param[out] off_wall Set when some rim edge lies on no wall, or #MFTL_NULL.
 * @param[out] n_rim    Mesh-boundary half-edges counted, or #MFTL_NULL.
 * @return Bit `2k` for the wall `x_k = min`, bit `2k+1` for `x_k = max`.
 * @rank_local
 */
int MFTLBodyRimWalls(const MFTL_Object *obj, bool *off_wall, int *n_rim);

/**
 * @brief The extent accumulator and the area vector of the OWNED triangles of
 *        @p obj, from one gather of each triangle's corners.
 *
 * The total area vector \f$\sum_t \vect{S}_t\f$ of a surface that encloses a
 * region is zero; a sheet that separates a periodic box has one the size of
 * the cross-section.
 *
 * @param      obj    The mesh.
 * @param[in,out] extent An extent accumulator (MFTLPeriodicExtentInit()), added to.
 * @param[in,out] S    Summed area vectors, added to.
 * @param[in,out] area Summed areas, added to.
 * @rank_local
 */
void MFTLBodyShape(const MFTL_Object *obj, double extent[MFTL_PERIODIC_EXTENT_SLOTS], double S[3], double *area);

/**
 * @brief This rank's share of a body's volume as the flux of
 *        \f$(x_k - \mathrm{ref})\,\vect{e}_k\f$, and its area, over OWNED triangles.
 *
 * Each triangle's centroid coordinate is taken in the image nearest
 * `centre[k]` on a periodic axis. With @p k < 0 only the area is summed.
 *
 * @param      obj    The mesh.
 * @param      k      The flux axis, or -1.
 * @param      ref    The reference coordinate on it.
 * @param      centre The body's centre (MFTLPeriodicExtentResolve()).
 * @param[out] volume \f$\sum_t (\bar x_k - \mathrm{ref})\,S_{t,k}\f$.
 * @param[out] area   \f$\sum_t \lVert\vect{S}_t\rVert\f$.
 * @rank_local The arithmetic of the topology change's volume correction, which
 *             calls it.
 */
void MFTLBodyFluxVolume(const MFTL_Object *obj, int k, double ref, const double centre[3], double *volume, double *area);

/**
 * @brief This rank's share of a body's volume and first moments along flux axis
 *        @p k, over OWNED triangles.
 *
 * With every triangle moved rigidly into the image nearest @p centre on the
 * periodic axes, \f$y_k = x_k - \mathrm{ref}\f$ and \f$y_j = x_j - c_j\f$ for
 * \f$j \ne k\f$, exactly for flat triangles and linear integrands:
 * `out[0]` \f$= V = \sum_t \bar y_k S_{t,k}\f$ and `out[1+j]`
 * \f$= \sum_t \frac{S_{t,k}}{12}\big(\sum_i y_{k,i}y_{j,i} + \sum_i y_{k,i}\sum_i y_{j,i}\big)\f$,
 * i.e. \f$2V(C_k - \mathrm{ref})\f$ for \f$j = k\f$ and \f$V(C_j - c_j)\f$
 * otherwise, \f$C\f$ the centroid of the region measured.
 *
 * @param      obj    The mesh.
 * @param      k      The flux axis.
 * @param      ref    The reference coordinate on it.
 * @param      centre The body's centre: the lift on periodic axes and the
 *                    reference on the others.
 * @param[out] out    #MFTL_BODY_MOMENT_SLOTS sums.
 * @rank_local
 */
void MFTLBodyFluxMoments(const MFTL_Object *obj, int k, double ref, const double centre[3], double out[MFTL_BODY_MOMENT_SLOTS]);

/**
 * @brief The displacement field of a volume correction, left in `verts.normal`,
 *        and this rank's share of the derivative of the enclosed volume along it.
 *
 * The field \f$\vect{u}\f$ is the unit direction of the volume gradient,
 * \f$\partial V/\partial\vect{x}_v \propto \sum_{T\ni v}(\vect{e}_1\times\vect{e}_2)_T\f$
 * (#MFTL_NORMAL_WEIGHT_AREA): every vertex moves in the
 * direction in which it changes the volume fastest. A vertex
 * flagged #MFTL_FLAG_BOUNDARY loses the component normal to every wall it lies
 * on (within #MFTL_BODY_WALL_TOL), so it slides in the wall and the rim stays
 * there. Moving every OWNED vertex by \f$d\,\vect{u}\f$ changes the enclosed
 * volume by \f$d\,\sum_t \vect{S}_t\cdot(\vect{u}_0+\vect{u}_1+\vect{u}_2)/3
 * + O(d^2)\f$ (\f$\vect{S}_t\f$ the area vector), and the return value is
 * that sum over the (triangle, corner) pairs whose corner this rank OWNS: with a
 * vertex-complete halo every such pair is present exactly once in the job, so
 * the per-body SUM over holders is the exact first variation. Corners are
 * gathered seam-consistently. It holds for a closed body and for one closed by
 * the walls alike, because the walls do not move. Dividing by it rather than by
 * the area is what makes a correction Newton's method.
 *
 * @param obj The mesh; `verts.normal` is overwritten for every vertex (ghost
 *            entries hold partial sums, see MFTLComputeVertexNormals()).
 * @return The owned share of \f$dV/dd\f$.
 * @rank_local The arithmetic of MFTLEnsembleCorrectVolumes() and of the
 *             topology change's correction, which both call it.
 */
double MFTLBodyVolumeDisplacement(MFTL_Object *obj);

// -----------------------------------------------------------------------------
// 10. LEAST SQUARES AND LOCAL HEIGHT FITS  (see src/geometry/surface_fit.c)
// -----------------------------------------------------------------------------

/** @brief Most unknowns MFTLLeastSquaresQRCP() solves for (a quadratic in two variables). */
#define MFTL_LSQ_MAX_COLS 6

/**
 * @brief \f$\min_x \lVert A x - b\rVert_2\f$ by Householder QR with column
 *        pivoting inside two GROUPS of columns; the library's replacement for
 *        LAPACK's `dgelsy` at the sizes a local fit has.
 *
 * Columns `[0, n_lead)` are factored first and pivot only among themselves,
 * then columns `[n_lead, n)`. Within a group the column of largest remaining
 * norm goes next (Businger & Golub 1965); a group stops at the first pivot with
 * \f$|R_{kk}| \le \tau |R_{00}|\f$, and its remaining columns are DROPPED: their
 * coefficients are set to zero and the solution is the basic solution of the
 * accepted columns. With the lead group the constant and linear terms of a
 * height fit and the trailing group its curvature terms, a stencil that
 * cannot determine a curvature component gets that component zero -- a lower
 * degree in that direction -- instead of the minimum-norm solution `dgelsy`
 * returns, which trades the height against the curvature terms whenever the
 * null space couples them.
 *
 * The test is relative, so @p tau is dimensionless only if the columns are of
 * comparable magnitude: scale the data first (MFTLFitHeightParaboloid() takes
 * coordinates in units of the local edge length). A uniform scaling of @p A and
 * @p b by a power of two changes no pivot decision and no bit of @p x.
 *
 * @param         m      Rows (any; fewer rows than columns limits the rank).
 * @param         n      Columns, at most #MFTL_LSQ_MAX_COLS.
 * @param         n_lead Size of the first group, in [0, n].
 * @param[in,out] A      Column-major `m x n`, leading dimension @p m; destroyed
 *                       (the Householder vectors and R are left in it).
 * @param[in,out] b      `[m]` right-hand side; destroyed (becomes Q^T b).
 * @param         tau    Relative rank tolerance, e.g. 1e-3.
 * @param[out]    x      `[n]` the solution, zeros on dropped columns.
 * @param[out]    kept   `[n]` or #MFTL_NULL: 1 where the column was accepted.
 * @return The numerical rank (the number of accepted columns).
 * @rank_local Pure function, no allocation.
 */
int MFTLLeastSquaresQRCP(int m, int n, int n_lead, double *A, double *b, double tau, double *x, unsigned char *kept);

/**
 * @brief Weighted least-squares height paraboloid
 *        \f$z = c_0 + c_1x + c_2y + c_3x^2 + c_4xy + c_5y^2\f$ through samples
 *        given in a LOCAL frame, optionally with their slopes.
 *
 * Each sample \f$j\f$ contributes the row
 * \f$\sqrt{w_j}\,(1, x_j, y_j, x_j^2, x_jy_j, y_j^2)\,c = \sqrt{w_j}\,z_j\f$ and,
 * where its slopes are given, the two rows
 * \f$\sqrt{w_j}\,(0, 1, 0, 2x_j, y_j, 0)\,c = \sqrt{w_j}\,\partial_x z\f$ and
 * \f$\sqrt{w_j}\,(0, 0, 1, 0, x_j, 2y_j)\,c = \sqrt{w_j}\,\partial_y z\f$ -- a
 * unit normal \f$(n_x, n_y, n_z)\f$ gives
 * \f$\partial_x z = -n_x/n_z\f$, \f$\partial_y z = -n_y/n_z\f$. Solved by
 * MFTLLeastSquaresQRCP() with {1, x, y} as the lead group and the curvature
 * terms as the trailing one, never through the normal equations. Pass
 * coordinates scaled to O(1) (in units of the local edge length): heights and
 * slopes are then both O(1) and one weight serves both kinds of row.
 *
 * @param      n         Samples, at most 64.
 * @param      pts       `[3n]` local (x, y, z).
 * @param      slopes    `[2n]` (dz/dx, dz/dy), or #MFTL_NULL for positions only.
 * @param      use_slope `[n]` nonzero where sample j's slopes enter, or
 *                       #MFTL_NULL for every sample (ignored without @p slopes).
 * @param      w         `[n]` non-negative weights.
 * @param      tau       Relative rank tolerance (MFTLLeastSquaresQRCP()).
 * @param[out] c         `[6]` the coefficients; zero where undetermined.
 * @param[out] kept      `[6]` or #MFTL_NULL: which coefficients were determined.
 * @return The rank. \f$c_0\f$ is meaningful only if `kept[0]`.
 * @rank_local Pure function, no allocation.
 */
int MFTLFitHeightParaboloid(int n, const double *pts, const double *slopes, const unsigned char *use_slope, const double *w, double tau, double c[6],
                            unsigned char kept[6]);

// -----------------------------------------------------------------------------
// 11. SIGNED CURVATURE  (see src/geometry/vertex_curvature.c)
//
// THE CONVENTION, stated once: kappa = kappa_1 + kappa_2 = div_s n, with n the
// OUTWARD normal (corners counter-clockwise seen from outside). A sphere of
// radius R has kappa = 2/R, the Young-Laplace jump is p_in - p_out = sigma
// kappa, and the surface-tension force density -sigma kappa n points inward.
// The remesher's curvature sizing reads max(|kappa_1|, |kappa_2|) off the COTAN
// record.
// -----------------------------------------------------------------------------

/**
 * @brief How MFTLComputeVertexCurvature() estimates the curvature of a vertex.
 *
 * Both are one-ring estimates, so both are exact on an owned vertex of a
 * distributed mesh given what the vertex-complete halo holds; they differ in
 * what they assume between the vertices.
 */
typedef enum MFTL_CurvatureMethod
{
  /**
   * Meyer, Desbrun, Schröder & Barr (2003), every operator of their paper:
   * the mean-curvature normal \f$\vect{K}_i = \nabla_i A / A_i\f$ (the
   * cotangent formula over the MIXED area \f$A_i\f$), signed by projection on
   * the area-weighted vertex normal, \f$\kappa_i = \vect{K}_i\cdot\hat{\vect{N}}_i\f$
   * (their eq. 12); the Gaussian curvature by the angle deficit
   * \f$(2\pi - \sum_f\theta_f)/A_i\f$ (eq. 9); the principal curvatures from
   * both (eqs. 10, 11); the principal directions from their least-squares fit
   * of the curvature tensor to the edge normal curvatures, with the trace
   * fixed to \f$\kappa_i\f$ (section 5.3). Needs nothing but positions.
   * Converges in an integral sense, NOT pointwise on an irregular mesh
   * (src/geometry/README.md, "Signed curvature").
   */
  MFTL_CURVATURE_COTAN = 0,

  /**
   * A weighted least-squares height paraboloid in the frame of the vertex
   * normal, through the vertex and its one-ring: three rows per sample (the
   * height and the two slopes its normal gives), solved by
   * MFTLFitHeightParaboloid(). Curvatures and the principal direction are those
   * of the fitted surface at the vertex (shape operator \f$I^{-1}I\!I\f$). Reads
   * `verts.normal` of the vertex AND of its neighbours: a ghost neighbour's
   * normal must be its owner's (MFTLEnsembleUpdateVertexNormals()). The fit is
   * only as good as those normals: with #MFTL_NORMAL_WEIGHT_FIT normals it
   * converges on every mesh measured (second order on spheres and the
   * ellipsoid, first order on an irregular torus), with area- or
   * angle-weighted ones it stalls at a few percent on irregular meshes. A
   * vertex with fewer than three neighbours, fewer than three neighbours whose
   * normal is within 75 degrees of its own, or a rank-deficient system gets
   * the COTAN values and is counted.
   */
  MFTL_CURVATURE_FIT = 1
} MFTL_CurvatureMethod;

/** @brief Record slot: the signed curvature \f$\kappa = \kappa_1 + \kappa_2\f$ (2/R on a sphere). */
#define MFTL_CURVATURE_KAPPA 0
/** @brief Record slot: the Gaussian curvature \f$\kappa_G = \kappa_1\kappa_2\f$. */
#define MFTL_CURVATURE_GAUSS 1
/** @brief Record slot: the larger principal curvature \f$\kappa_1\f$. */
#define MFTL_CURVATURE_K1 2
/** @brief Record slot: the smaller principal curvature \f$\kappa_2\f$. */
#define MFTL_CURVATURE_K2 3
/** @brief Record slots 4-6: a unit tangent along the principal direction of \f$\kappa_1\f$ (a line: its sign is arbitrary). */
#define MFTL_CURVATURE_DIR1 4
/** @brief Components of a full curvature record. 1, 2, 4 and 7 are the valid prefixes. */
#define MFTL_CURVATURE_COMPONENTS 7

/**
 * @brief Signed curvature of every vertex: \f$\kappa\f$, and on request the
 *        Gaussian curvature, the principal curvatures and the principal
 *        direction.
 *
 * Writes a RECORD of @p ncomp doubles per vertex, interleaved, in the slot
 * order #MFTL_CURVATURE_KAPPA, #MFTL_CURVATURE_GAUSS, #MFTL_CURVATURE_K1,
 * #MFTL_CURVATURE_K2, #MFTL_CURVATURE_DIR1 -- the layout of a vertex field of
 * @p ncomp components, so `MFTLObjectField(obj, id)` can be passed directly.
 * Asking for a shorter prefix skips the work it does not need (the
 * principal-direction fit is a per-vertex gather).
 *
 * Exact on OWNED vertices `[0, verts.n_local)`: the cotangent sums are a
 * scatter over every triangle the rank holds, and the halo is vertex-complete.
 * A GHOST vertex gets a value from its partial neighbourhood -- plausible and
 * wrong, as for a ghost's vertex normal; MFTLEnsembleComputeCurvature() replaces
 * it with the owner's. On a RIM vertex (#MFTL_FLAG_BOUNDARY) the cotangent
 * estimate is one-sided and not a curvature; the fit is.
 *
 * Nothing compares a dimensional quantity against a constant: a mesh scaled by
 * \f$2^{-20}\f$ gives every curvature times \f$2^{20}\f$ (the Gaussian times
 * \f$2^{40}\f$) bit for bit.
 *
 * @param      obj        The mesh; needs current `tris.verts`, and for
 *                        #MFTL_CURVATURE_FIT current `verts.normal` on every
 *                        vertex of the one-rings it evaluates.
 * @param      method     #MFTL_CURVATURE_COTAN or #MFTL_CURVATURE_FIT.
 * @param      ncomp      Record length: 1, 2, 4 or #MFTL_CURVATURE_COMPONENTS.
 * @param[out] out        `[ncomp * verts.n]` records.
 * @param[out] n_fallback OWNED vertices for which the fit fell back to the
 *                        cotangent values, or #MFTL_NULL. Always 0 for COTAN.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for a NULL object or output, an
 *         unknown method or an invalid @p ncomp.
 *
 * @rank_local MFTLEnsembleComputeCurvature() adds the normal and the result
 *             exchanges.
 */
int MFTLComputeVertexCurvature(const MFTL_Object *obj, MFTL_CurvatureMethod method, int ncomp, double *out, int *n_fallback);

// -----------------------------------------------------------------------------
// 12. SURFACE TENSION  (see src/geometry/surface_tension.c)
//
// The INTEGRATED force on a piece of surface, sigma times the contour integral
// of the outward conormal around it (Tryggvason et al. 2011, eqs. 7.52-7.53):
// what a front-tracking solver spreads. Same convention as section 11: on a
// closed body with outward normals it is -sigma kappa n A, pointing inward.
// -----------------------------------------------------------------------------

/**
 * @brief Which piece of surface the force is integrated over, and where it
 *        is applied.
 */
typedef enum MFTL_SurfaceTensionForm
{
  /**
   * FRENET VERTEX (Tryggvason et al. 2011, eqs. 7.54-7.59): the pull on the
   * boundary of each vertex's barycentric cell, which inside a triangle runs
   * from an edge midpoint to the centroid to the next edge midpoint. With
   * the triangle's own normal on it (#MFTL_CONTOUR_NORMAL_TRIANGLES, the
   * book's eq. 7.59) the two segments add up to half the pull on the opposite
   * edge, \f$\vect{F}_v = \sum_{T\ni v}\tfrac12\sigma_T(\vect{x}_k -
   * \vect{x}_j)\times\hat{\vect{n}}_T\f$ for \f$(v, j, k)\f$ counter-clockwise:
   * exactly \f$-\partial E/\partial\vect{x}_v\f$ of \f$E = \sum_T\sigma_T A_T\f$,
   * so the net force AND torque of a closed body vanish to round-off. For
   * vertex markers.
   */
  MFTL_TENSION_FRENET_VERTEX = 0,

  /**
   * FRENET ELEMENT (Tryggvason et al. 2011, eq. 7.53 on a triangle): the pull
   * of the neighbours across each edge, \f$\vect{F}_T = \sum_{e\in T}
   * \sigma_e L_e \hat{\vect{m}}_e\f$, \f$\hat{\vect{m}}_e\f$ the unit outward
   * conormal on the edge. The triangle's own normal on all three edges would
   * cancel exactly, so the normal on an edge comes from both sides
   * (#MFTL_ContourNormal). The two triangles of an edge receive opposite
   * pulls, so the net force of a closed body vanishes to round-off; the net
   * torque does not (the force is applied at centroids). For centroid
   * markers.
   */
  MFTL_TENSION_FRENET_ELEMENT = 1
} MFTL_SurfaceTensionForm;

/**
 * @brief The normal the conormal \f$\hat{\vect{m}} = \hat{\vect{t}}\times
 *        \hat{\vect{n}}\f$ is built from along the contour.
 *
 * What decides POINTWISE accuracy (src/geometry/README.md, "Surface tension";
 * measured by examples/curvature_comparison.c). Both choices conserve the
 * net force exactly, because each piece of contour is shared by two cells and
 * evaluated once.
 */
typedef enum MFTL_ContourNormal
{
  /**
   * The TRIANGLES' normals, as printed in the book: in the vertex form each
   * triangle's own (the area gradient: torque-free and energy-consistent),
   * in the element form the mean of the two triangles of the edge. Needs no
   * vertex normals and no communication. Integrally consistent but NOT
   * pointwise on an irregular mesh: the vertex form is \f$\sigma\f$ times
   * Meyer's cotangent operator times the mixed area, with its errors (17 %
   * at the worst vertex of a remeshed sphere).
   */
  MFTL_CONTOUR_NORMAL_TRIANGLES = 0,

  /**
   * The FITTED vertex normals (#MFTL_NORMAL_WEIGHT_FIT, in `verts.normal`),
   * interpolated linearly along the contour; the book's "higher order
   * approximations" (p. 173). Both forms then converge pointwise on irregular
   * meshes -- the vertex form to \f$-\sigma\kappa\vect{n}\f$ times the
   * vertex's BARYCENTRIC area, which is the coupler's marker weight. The net
   * torque is then zero only to truncation error. A ghost vertex must carry
   * its owner's normal; the ensemble call computes and exchanges them.
   */
  MFTL_CONTOUR_NORMAL_VERTICES = 1
} MFTL_ContourNormal;

/**
 * @brief What acts on the RIM of an open surface (#MFTL_FLAG_BOUNDARY).
 *
 * The rim's line force is \f$\sigma\hat{\vect{m}}\,dl\f$ along the outward
 * conormal. A wall is a non-periodic domain face; a rim edge is on it when both
 * ends are within #MFTL_BODY_WALL_TOL of it (the MFTLBodyRimWalls() rule).
 */
typedef enum MFTL_RimForce
{
  /**
   * A FREE EDGE: nothing beyond the rim pulls, so the rim is pulled into the
   * surface. Both forms then give a total of zero on every body.
   */
  MFTL_RIM_FORCE_FREE = 0,

  /**
   * The WALL REACTS: as FREE, plus the wall-normal component of the line
   * force on every rim edge lying on a wall, which the wall supplies. A flat
   * sheet meeting its walls at a right angle feels no force, and with a
   * linear \f$\sigma\f$ (vertex field) the total over the sheet is the
   * Marangoni force \f$\Delta\sigma\,W\f$ exactly, in both forms. A rim edge on
   * no wall is treated as FREE.
   */
  MFTL_RIM_FORCE_WALL = 1,

  /** Rim vertices (vertex form) and triangles with a rim edge (element form) get zero force. */
  MFTL_RIM_FORCE_EXCLUDE = 2
} MFTL_RimForce;

/**
 * @struct MFTL_SurfaceTensionParameters
 * @brief Everything MFTLComputeSurfaceTension() and
 *        MFTLEnsembleComputeSurfaceTension() need. Start from
 *        MFTLSurfaceTensionDefaultParameters().
 */
typedef struct MFTL_SurfaceTensionParameters
{
  MFTL_SurfaceTensionForm form;  ///< Frenet vertex (default) or Frenet element.

  /** The uniform coefficient, used when `sigma_field` is negative. Any unit; the force is linear in it. */
  double sigma;

  /**
   * A registered one-component field holding \f$\sigma\f$, or -1 for the
   * uniform `sigma`. A TRIANGLE field is \f$\sigma_T\f$, on an edge the mean of
   * its two triangles. A VERTEX field is the linear interpolant: on a triangle
   * the mean of its corners (the vertex form with triangle normals), along a
   * piece of contour the mean of its ends. Ghost values must be the owners'
   * (the ensemble call exchanges them).
   */
  int sigma_field;

  MFTL_ContourNormal contour_normal;  ///< The normal along the contour; #MFTL_CONTOUR_NORMAL_TRIANGLES by default.

  MFTL_RimForce rim;  ///< What acts on the rim of an open surface.
} MFTL_SurfaceTensionParameters;

/**
 * @brief The defaults: Frenet vertex, \f$\sigma = 1\f$ uniform, the
 *        triangles' normals along the contour (the book's formula), free rims.
 *        That force is the exact gradient of the discrete surface energy:
 *        net force, net torque and the work it does on any velocity are
 *        exact, it needs no vertex normals and no communication, and it costs
 *        9 ns per vertex. For a solver that spreads the force plainly it is
 *        as good as any variant measured (src/geometry/README.md, "Surface
 *        tension"). #MFTL_CONTOUR_NORMAL_VERTICES (fitted normals, about
 *        thirty times dearer) converges per vertex, which is what a
 *        balanced-force curvature needs.
 * @return The default parameter block.
 * @rank_local
 */
MFTL_SurfaceTensionParameters MFTLSurfaceTensionDefaultParameters(void);

/**
 * @brief The integrated surface-tension force on every vertex (Frenet vertex)
 *        or every triangle (Frenet element).
 *
 * Rows `[0, n_local)` are exact on a distributed mesh without communication,
 * given current ghost values of what the form reads (sigma, and with
 * #MFTL_CONTOUR_NORMAL_VERTICES the vertex normals): the vertex form is a
 * scatter over every triangle the rank holds (the halo is vertex-complete), and
 * every owned triangle has all three edge neighbours. Ghost rows are computed
 * from a partial neighbourhood and are not forces. The force is integrated (a
 * force, not a density): pass the owned prefix to the coupler with
 * #MFTL_SPREAD_INTEGRATED and #MFTL_MARKERS_VERTICES or
 * #MFTL_MARKERS_CENTROIDS.
 *
 * Reads positions and `verts.normal`, never the cached `tris.normal`. Scaling a
 * mesh by \f$2^{-20}\f$ scales every force by \f$2^{-20}\f$, bit for bit.
 *
 * @param      obj   The mesh; needs current `tris.verts` and `tris.neighbors`,
 *                   and for #MFTL_CONTOUR_NORMAL_VERTICES current
 *                   `verts.normal` (MFTLComputeVertexNormals() with
 *                   #MFTL_NORMAL_WEIGHT_FIT; this call does not compute them).
 * @param      p     The parameters.
 * @param[out] force `[3 * verts.n]` (vertex form) or `[3 * tris.n]` (element
 *                   form), xyz interleaved, overwritten.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for a NULL argument, an unknown
 *         enumerator, or a `sigma_field` that is not a one-component field of
 *         this object.
 *
 * @rank_local MFTLEnsembleComputeSurfaceTension() adds the normals and the
 *             exchanges.
 */
int MFTLComputeSurfaceTension(const MFTL_Object *obj, const MFTL_SurfaceTensionParameters *p, double *force);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_GEOMETRY_H
