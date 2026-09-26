/**
 * @file mftl_geometry.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_geometry
 *
 * @brief Periodic-aware geometric queries over the SoA geometry arrays.
 *
 * Two properties of this module follow from the layout in mftl_types.h rather
 * than from anything geometric:
 *
 *   1. POSITIONS ARE REACHED BY INDEX, NOT THROUGH A RECORD. `MFTLVertexPos(obj,
 *      v)` returns a pointer into `verts.pos`, three contiguous doubles. There is
 *      no `MFTL_Vertex.pos` any more, and therefore no second copy of any
 *      coordinate anywhere in the library.
 *
 *   2. THERE ARE NO Load/Store HELPERS. MFTLLoadTriNormal / MFTLStoreTriNormal
 *      are what a geometry array needs when it is a bolted-on side channel that
 *      call sites are not supposed to know about. It is the primary storage here,
 *      so `obj->tris.normal[3*t+k]` is the normal way to write it and a helper
 *      that copies three doubles onto the stack only gets in the way -- routing
 *      the geometry kernel through a store helper measures 25 % slower.
 *
 * Everything here is inline and, apart from the two Compute* passes, side-effect
 * free. All distance and vector computations honour the domain's periodicity via
 * the minimum image convention.
 */

#ifndef MFTL_GEOMETRY_H
#define MFTL_GEOMETRY_H

#include <math.h>

#include "mftl_constants.h"
#include "mftl_topology.h"
#include "mftl_types.h"

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
 * @brief The shortest vector from @p p1 to @p p2 under the minimum image convention.
 *
 * The core of all periodic geometry here.
 *
 * The correction is a multiply, a round (one instruction on both AArch64 and
 * SSE4.1+) and an FMA per axis. That is measurably cheaper in a hot loop than
 * the two compare-multiply-add pairs of the branchless comparison form, and it
 * is correct for ANY displacement rather than only for |d| < 1.5 L, which the
 * comparison form silently required. Ties at exactly \f$\pm L/2\f$ round to even.
 *
 * \f[
 *   \Delta_k = (p_2 - p_1)_k - L_k\,\mathrm{round}\!\left(\frac{(p_2-p_1)_k}{L_k}\right)
 * \f]
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
 * @brief Wraps a position back into the periodic domain.
 *
 * Assumes the displacement since the last wrap is smaller than the domain, which
 * holds for any stable simulation. Comparisons rather than fmod(): the branches
 * are perfectly predicted, and the "branchless" alternative measures 1.34-1.37x
 * SLOWER in the per-timestep kernel because it does unconditional work on
 * non-periodic axes too.
 *
 * @param domain Supplies the per-axis periodicity and bounds.
 * @param pos    Position, wrapped in place. Non-periodic axes are untouched.
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
 * A multiply and a table lookup: `grid_factors` and `rank_map` are precomputed
 * at ensemble creation precisely so that this is not a division and not an
 * MPI_Cart_rank call. @p pos must be WRAPPED into the domain; coordinates
 * outside it are clamped to the boundary cell rather than wrapped, so a caller
 * that passes an unwrapped position gets a plausible wrong answer.
 *
 * This is the inner statement of `compute_owners` (src/communication/
 * redistribute.c), i.e. it runs once per vertex and once per triangle on every
 * timestep the mesh is redistributed. Nothing conditional belongs in it.
 *
 * ── Why there is no cut-plane offset here ─────────────────────────────────
 *
 * The distributed remesher moves the cut planes (src/remeshing/remesh_comm.c),
 * and the tempting way to express that is to subtract a per-axis shift from `pos`
 * right here. That puts a subtract and a periodicity branch per axis into the
 * production ownership loop to serve a routine that runs a few times per hundred
 * timesteps -- measured at 1.70x on this function alone. The shifted rule lives
 * with the remesher as MFTLRemeshOwnerProc() instead, and reaches redistribution
 * through the owner-map override (MFTL_OwnerMap, mftl_communication.h) rather
 * than through this function -- which is also what the DISPERSED strategy needs,
 * and it needs a rule that is not a plane test at all.
 *
 * @param domain The MPI/physical world.
 * @param pos    A position, ALREADY WRAPPED into the domain box.
 * @return The rank owning the Cartesian cell containing @p pos.
 *
 * @measured{Adding the shift and its periodicity branch here cost **1.70x** --
 *           1.79 to 3.05 ns/entity, 42 to 89 instructions for the same loop.}
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
 * @brief Area, unit normal and centroid of triangle @p t, in one pass.
 *
 * Uses compact `tris.verts` when current and falls back to the half-edge records
 * during construction, before derived connectivity has been built. Any output
 * may be MFTL_NULL. A loop over many triangles should use
 * MFTL_FOREACH_TRIANGLE_GEOMETRY (mftl_object.h), which unswitches periodicity
 * outside the loop.
 *
 * @param      obj        The mesh.
 * @param      t          Triangle index.
 * @param[out] area_out   Receives \f$\tfrac12\norm{\vect{e}_1\times\vect{e}_2}\f$.
 *                        May be #MFTL_NULL.
 * @param[out] normal_out Receives the UNIT normal, or the zero vector for a
 *                        degenerate triangle. May be #MFTL_NULL.
 * @param[out] center_out Receives the centroid WRAPPED back into the domain,
 *                        because that is what a position is expected to be. The
 *                        offset it is built from is seam-consistent, so this is
 *                        the true centroid even for a triangle straddling a
 *                        periodic face. May be #MFTL_NULL.
 *
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

  /* The cross product is perpendicular to the triangle; its magnitude is twice
   * the area.
   *
   * ── Where the degeneracy guard belongs ──────────────────────────────────
   *
   * The guard exists to keep a degenerate triangle from producing a NaN that
   * poisons a reduction, and the only operation that can produce one is the
   * RECIPROCAL used to normalise. So the floor goes on the DIVISOR, not on the
   * magnitude.
   *
   * Clamping the squared magnitude before the square root -- which is what this
   * did -- floors the reported AREA at 0.5*sqrt(1e-12) = 5e-7 instead of letting
   * it reach zero, and that is not a rounding detail:
   *
   *   - `quality.min_triangle_area` can then never report a degenerate triangle;
   *   - the mean-ratio quality q = 4*sqrt(3)*A / (a^2+b^2+c^2) becomes UNBOUNDED
   *     ABOVE for small triangles, for a measure whose maximum is 1 at the
   *     equilateral triangle. An average q of 1.023 was measured on a degenerate
   *     mesh (docs/remeshing-quality-analysis.md);
   *   - and the NORMAL of any triangle smaller than that floor came out
   *     under-normalised rather than unit, because it was divided by the clamp
   *     instead of by its own length.
   *
   * ── Why the guard is a test against zero and not a clamp ────────────────
   *
   * `1.0 / MFTL_MAX(mag, tiny)` keeps the instruction count identical to the
   * broken version (one max, one sqrt, one divide) but only MOVES the floor: any
   * triangle whose |e1 x e2| falls below `tiny` still gets a short normal. With
   * `tiny = MFTL_VERYVERYSMALL` that is an area below 5e-16 -- reachable in a
   * simulation carrying SI units on small-scale geometry, and the failure is the
   * silent kind again.
   *
   * The test against zero has no such cliff: every triangle with a representable
   * non-zero cross product gets an exactly unit normal, and one whose cross
   * product is exactly zero gets the zero normal, which is the same detectable
   * sentinel normalise_vertex_normals() uses -- a unit normal never has zero
   * length, so a consumer can tell "not computable" from "computed".
   *
   * It costs one instruction per triangle: `fcmp`+`fcsel` against `fmax`.
   * The retired frame-backed ensemble benchmark measured this guard at +1.1%
   * on the fused geometry+quality pass. That historical result is recorded in
   * docs/performance-analysis.md; the exactness argument remains independent of
   * the triangle input layout. */
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
 * The representation anything geometric about a whole triangle wants: three
 * absolute positions that are consistent with each other even when the triangle
 * straddles a periodic face. Corner 0 is a wrapped position; corners 1 and 2 are
 * corner 0 plus the seam-corrected edge vectors, so they may lie just outside
 * the domain box. That is the point -- a triangle at x = L-eps and x = +eps comes
 * out as a small triangle rather than a domain-spanning sliver.
 *
 * Used by the BVH (primitive geometry and leaf boxes) and by the spatial halo
 * (per-triangle bounds before packing). It is here rather than in either of them
 * because two modules needing the same nine doubles is the definition of a
 * geometry query.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index. Bounds-asserted in Debug.
 * @param[out] out Row \f$k\f$ is corner \f$k\f$. Corner 0 is wrapped; corners
 *                 1 and 2 may lie just outside the domain box, which is the
 *                 point.
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
 * @brief Axis-aligned bounds of triangle @p t, from its unwrapped corners.
 *
 * @param      obj The mesh.
 * @param      t   Triangle index.
 * @param[out] lo  Lower corner of the box.
 * @param[out] hi  Upper corner of the box.
 *
 * @note Tight: no tolerance is added here, because every caller inflates by its
 *       own and adding a second margin invisibly would make the two impossible
 *       to reason about together.
 *
 * @warning Because the corners are mutually unwrapped, the box of a
 *          seam-straddling triangle correctly extends OUTSIDE the domain. That
 *          is what makes it a valid box; it is also why a min/max box is the
 *          wrong primitive for the spatial halo, which must use centre plus
 *          half-width instead.
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

// -----------------------------------------------------------------------------
// 5. POINT QUERIES
// -----------------------------------------------------------------------------

/**
 * @brief Which part of a triangle a closest point lies on.
 *
 * The seven Voronoi regions of a triangle, named in the half-edge convention:
 * corner \f$k\f$ is `tri[k]`, and EDGE \f$k\f$ runs from corner \f$k\f$ to
 * corner \f$(k+1) \bmod 3\f$ -- exactly half-edge `3t+k` of an MFTL triangle.
 * So an edge feature names its half-edge directly and needs no lookup table.
 *
 * The distinction matters to anything that decides a SIGN from the closest
 * point. A face interior has one normal; an edge or a vertex is shared, and the
 * only normal there that gets the inside/outside test right is the
 * angle-weighted pseudo-normal of the whole neighbourhood (Baerentzen &
 * Aanaes, 2005). A caller has to know which case it is in to pick it.
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
 * @brief Closest point of a closed triangle to a point, and which feature it is on.
 *
 * Ericson's region test (Real-Time Collision Detection, section 5.1.5): six dot
 * products decide which of the seven Voronoi regions the query point projects
 * into, and only the region's own formula is evaluated. It is exact in the
 * sense that matters here -- the returned point IS on the triangle and the
 * feature it reports is the one the returned point lies on -- and it has no
 * square root and no division outside the region that needs one.
 *
 * ── Degenerate triangles ───────────────────────────────────────────────────
 *
 * A collinear or collapsed triangle has no face region: the barycentric
 * denominator of the face case is zero. Such a triangle is answered as the
 * closest of its three edges, which is what it geometrically is, rather than
 * with a NaN. A triangle collapsed to a point reports VERTEX0.
 *
 * @param      p       Query point.
 * @param      tri     The three corners, unwrapped into one periodic image.
 *                     The caller applies any minimum-image shift.
 * @param[out] closest The closest point on the triangle. May be #MFTL_NULL.
 * @param[out] feature Where @p closest lies. May be #MFTL_NULL.
 * @return The SQUARED distance from @p p to the triangle.
 *
 * @rank_local Pure function; no MFTL state is read.
 */
double MFTLClosestPointOnTriangle(const double p[3], const double tri[3][3], double closest[3], MFTL_TriangleFeature *feature);

// -----------------------------------------------------------------------------
// 6. BULK GEOMETRY PASSES
// -----------------------------------------------------------------------------

/**
 * @brief Writes area, normal and centroid for the first @p count triangles.
 *
 * Correct on any object, whatever state its derived connectivity is in. With
 * current `tris.verts`, the steady-state kernel streams compact indices, gathers
 * authoritative positions and unswitches periodicity outside the triangle loop.
 * Before that array exists it falls back to implicit half-edge indexing. The two
 * paths are bit-identical and covered by
 * `test_triangle_geometry_paths_are_bit_identical`.
 *
 * @param obj   The mesh. `tris.area`, `tris.normal` and `tris.center` are
 *              overwritten for the first @p count triangles.
 * @param count Pass `obj->tris.n_local` for owned only, `obj->tris.n` to include
 *              ghosts.
 *
 * @warning Redistribution routes ghost triangles by their CACHED CENTRE, so
 *          anything preceding a redistribution must use the full count. If only
 *          the owned centres are refreshed, two ranks disagree about where a
 *          ghost sits and one of them silently ends up with a triangle that is
 *          missing an edge neighbour.
 *
 * @rank_local
 * @note MFTLEnsembleUpdate() is the ensemble-wide counterpart, which
 *       additionally produces and reduces the quality statistics.
 */
void MFTLComputeTriangleGeometry(MFTL_Object *obj, int count);

/**
 * @brief How MFTLComputeVertexNormals() weights the faces around a vertex.
 *
 * Two weightings, because there are two genuinely different questions a vertex
 * normal answers, and each weighting is the right answer to exactly one of them.
 */
typedef enum MFTL_NormalWeighting
{
  /**
   * Weight each incident face by its AREA. The un-normalised cross product IS
   * twice the area times the unit normal, so this costs nothing beyond the cross
   * product. It is the gradient of the enclosed volume with respect to the
   * vertex, which is the normal a volume-conserving front-tracking scheme wants.
   * The default for everything that moves vertices.
   */
  MFTL_NORMAL_WEIGHT_AREA = 0,

  /**
   * Weight each incident face by its INTERIOR ANGLE at the vertex
   * (Thurmer-Wuthrich). Invariant under re-triangulation of the fan -- split a
   * face into two and neither the angle sum nor the normal moves -- and it is the
   * one weighting for which the inside/outside test
   * `sign((x - v) . n(v))` is PROVABLY correct for a point x whose closest
   * surface point is the vertex v (Baerentzen & Aanaes, "Signed distance
   * computation using the angle weighted pseudonormal", 2005). Area weighting
   * has no such guarantee and does fail it, on exactly the thin, sharply folded
   * configurations a topology change is detected on. One `atan2` per corner.
   */
  MFTL_NORMAL_WEIGHT_ANGLE = 1
} MFTL_NormalWeighting;

/**
 * @brief Vertex normals for the whole mesh, into `verts.normal`.
 *
 * The per-vertex counterpart of MFTLEnsembleUpdate: one flat pass gathers each
 * triangle's current positions and scatters its weighted face normal into its
 * three vertices, then one streaming pass normalises. Implemented in
 * src/geometry/vertex_geometry.c,
 * where the choice of loop shape is argued and measured.
 *
 * ── The weighting ──────────────────────────────────────────────────────────
 *
 * #MFTL_NORMAL_WEIGHT_AREA:
 *
 * n(v) = normalise( SUM over incident faces of e1 x e2 )
 *
 * The un-normalised cross product IS twice the face area times the unit face
 * normal, so summing raw cross products is area weighting with no extra work --
 * no square root, no trigonometry, nothing per face beyond the cross product the
 * loop already needs. It is also the direction of the enclosed-volume gradient
 * at that vertex, which is the normal a volume-conserving front-tracking scheme
 * wants: moving a vertex along it changes the enclosed volume at the maximum
 * rate, and the discrete divergence theorem stays consistent with it.
 *
 * #MFTL_NORMAL_WEIGHT_ANGLE:
 *
 * n(v) = normalise( SUM over incident faces of alpha_f(v) * n_f )
 *
 * with alpha_f(v) the interior angle of face f at v and n_f its unit normal.
 * Less sensitive to a fan of very unequal triangles, independent of how the fan
 * is triangulated, and the pseudo-normal a signed distance needs: the
 * topology-change module rasterises its signed distance field with it. A
 * degenerate face has no unit normal and contributes nothing.
 *
 * Area weighting is the right default for a remeshed surface, where the fan is
 * kept near-equilateral by construction and the two agree closely.
 *
 * ── Preconditions ──────────────────────────────────────────────────────────
 *
 * Needs current `tris.verts`: call MFTLObjectBuildDerived() after a topology
 * change. Vertex motion needs no derived refresh. The triangle loop is
 * unswitched by domain periodicity and computes minimum-image edges in registers.
 *
 * ── What it computes on a distributed mesh ─────────────────────────────────
 *
 * The accumulation runs over ALL triangles, ghosts included, and this is what
 * makes the function need NO COMMUNICATION AT ALL. A vertex's normal is a
 * function of its one-ring, so an owned vertex gets the exact answer as long as
 * every triangle touching it is present locally -- which is exactly what a
 * vertex-complete ghost halo guarantees.
 *
 * GHOST vertices are a different matter: their far-side triangles are not here,
 * so `verts.normal` at index >= n_local holds a PARTIAL sum, correctly
 * normalised but taken over too few faces. It must be overwritten by the ghost
 * exchange, not consumed. Nothing in the library reads it yet.
 *
 * @param obj The mesh. `verts.normal` is overwritten for EVERY vertex, owned
 *            and ghost.
 * @param weighting #MFTL_NORMAL_WEIGHT_AREA or #MFTL_NORMAL_WEIGHT_ANGLE.
 *
 * @rank_local Every rank does its own partition.
 *
 * @warning Exact on an owned vertex only when the halo is VERTEX-COMPLETE.
 *          MFTLEnsembleUpdateVertexNormals() adds the one exchange that also
 *          makes the ghosts exact.
 */
void MFTLComputeVertexNormals(MFTL_Object *obj, MFTL_NormalWeighting weighting);

// -----------------------------------------------------------------------------
// 7. SIGNED DISTANCE ON A LATTICE  (see src/geometry/signed_distance.c)
// -----------------------------------------------------------------------------

/**
 * @brief Value of a lattice node that no triangle came within the cutoff of.
 *
 * Neither sign is known there, so it is not a distance at all -- which is why it
 * is the largest double rather than an infinity: MFTLSdfMerge() needs it to lose
 * every comparison, and marching cubes skips any cell that has one as a corner.
 * A cell the surface crosses can never have such a corner, as long as the
 * cutoff exceeds the cell diagonal (see MFTLRasterizeSignedDistance()).
 */
#define MFTL_SDF_UNKNOWN MFTL_DBL_MAX

/**
 * @struct MFTL_SdfGrid
 * @brief A dense WINDOW onto a (possibly periodic) Cartesian lattice of signed
 *        distances.
 *
 * ## THE LATTICE AND THE WINDOW ARE DIFFERENT THINGS
 *
 * The LATTICE is global: node `g` on axis `k` sits at `origin[k] + g *
 * spacing[k]`, and on a periodic axis `g` and `g + n_global[k]` are the same
 * node. A node's integer coordinates are therefore an identity that every rank
 * computes the same way from the same numbers -- which is what lets a
 * distributed marching cubes name a surface vertex by the lattice EDGE it lies
 * on, with no negotiation and no stitching.
 *
 * The WINDOW is what one caller stores: `dims[k]` consecutive nodes starting at
 * global index `lo[k]`. On a periodic axis the window may start anywhere
 * (`lo` may be negative, or past the end) and indices are taken modulo
 * `n_global`, so a window can straddle the seam; it may not exceed one period
 * (`dims <= n_global`). On a non-periodic axis there is nothing to wrap.
 *
 * A serial caller that just wants a box of nodes uses MFTLSdfGridInit(), which
 * makes the window the whole lattice and switches periodicity off.
 *
 * ## THE VALUES
 *
 * `phi` is negative INSIDE -- on the side opposite to the triangles' outward
 * normals -- positive outside, and #MFTL_SDF_UNKNOWN where nothing has been
 * rasterised. x varies fastest.
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
} MFTL_SdfGrid;

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
 * @param grid The grid; `phi` is freed and set to #MFTL_NULL.
 */
void MFTLSdfGridFree(MFTL_SdfGrid *grid);

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
 * @brief The rule two contributions to one node are combined by: the smaller
 *        magnitude wins, and a tie goes to the NEGATIVE one.
 *
 * Commutative and associative, so the result does not depend on the order in
 * which triangles -- or, in a distributed rasterisation, ranks -- contribute.
 * That is what makes the merged field bit-identical at every process count. The
 * tie rule is arbitrary but must be SOME total order: "keep the first" would
 * make a node exactly equidistant from a sheet's two faces take whichever sign
 * arrived first.
 *
 * #MFTL_SDF_UNKNOWN loses every comparison, so it is also the identity.
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
 * @brief Rasterises the signed distance to the first @p n_tris triangles onto
 *        every window node within @p cutoff of one of them.
 *
 * ## THE SIGN: ANGLE-WEIGHTED PSEUDO-NORMALS
 *
 * For each node x near a triangle T the closest point q on T is found with
 * MFTLClosestPointOnTriangle(), and the sign is that of `(x - q) . n` where n is
 * the pseudo-normal of the FEATURE q lies on:
 *
 *   face interior   the unit face normal
 *   edge interior   the sum of the two incident unit face normals (the twin's
 *                   triangle across half-edge 3t+k; just T's at a mesh rim)
 *   vertex          `verts.normal` -- which the caller must have filled with
 *                   #MFTL_NORMAL_WEIGHT_ANGLE
 *
 * With angle weighting at the vertices this is provably the correct
 * inside/outside test for a closed, consistently oriented, non-self-intersecting
 * surface (Baerentzen & Aanaes, 2005). Area weighting is not, and fails on
 * exactly the configurations a topology change is detected on: two sheets a
 * fraction of an edge apart, one of them sharply curved.
 *
 * Each node keeps the contribution of smallest magnitude, combined with
 * MFTLSdfMerge(); ties between triangles sharing the closest feature carry the
 * same pseudo-normal and therefore the same sign.
 *
 * ## THE CUTOFF, AND WHY ONLY ONE BOUND ON IT MATTERS
 *
 * A node with no triangle within @p cutoff keeps #MFTL_SDF_UNKNOWN. Marching
 * cubes needs the sign only at the corners of cells the surface crosses, and
 * every corner of such a cell lies within one cell DIAGONAL of the surface. So a
 * cutoff above the cell diagonal (sqrt(3) * spacing on a cubic lattice) is
 * enough, and nodes left unknown are exactly nodes no surface cell needs -- no
 * sign propagation, no flood fill, no global pass.
 *
 * ## PERIODICITY
 *
 * Triangles are taken with MFTLGetTriangleCorners(), i.e. unwrapped around
 * corner 0, and the nodes they reach are addressed by UNWRAPPED global index,
 * so the distance is always measured in the triangle's own periodic image and
 * the window's modular indexing puts the result where it belongs.
 *
 * @param grid    The grid; values are merged into, not overwritten, so several
 *                objects (or several ranks' shares) may be rasterised into one.
 * @param obj     The mesh. Needs current `tris.verts`, and `verts.normal` holding
 *                ANGLE-weighted normals for every corner of the triangles used --
 *                ghosts included (MFTLEnsembleUpdateVertexNormals() with
 *                #MFTL_NORMAL_WEIGHT_ANGLE on a distributed mesh).
 * @param n_tris  Rasterise triangles `[0, n_tris)`: `tris.n_local` for owned
 *                only, which is what a distributed rasterisation must pass so
 *                that no triangle is counted by two ranks.
 * @param cutoff  Distance within which nodes are written. > 0.
 * @return The number of node updates that changed a value, for diagnostics.
 * @rank_local No MPI.
 */
long MFTLRasterizeSignedDistance(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris, double cutoff);

#endif  // MFTL_GEOMETRY_H
