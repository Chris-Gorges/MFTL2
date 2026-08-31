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
// 5. BULK GEOMETRY PASSES
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
 * @brief Area-weighted vertex normals for the whole mesh, into `verts.normal`.
 *
 * The per-vertex counterpart of MFTLEnsembleUpdate: one flat pass gathers each
 * triangle's current positions and scatters its face normal into its three
 * vertices, then one streaming pass normalises. Implemented in
 * src/geometry/vertex_geometry.c,
 * where the choice of loop shape is argued and measured.
 *
 * ── The weighting ──────────────────────────────────────────────────────────
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
 * The alternative worth knowing about is angle weighting (Thurmer-Wuthrich),
 * which is less sensitive to a fan of very unequal triangles but needs an
 * inverse trigonometric function per corner. Area weighting is the right default
 * for a remeshed surface, where the fan is kept near-equilateral by
 * construction.
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
 *
 * @rank_local Every rank does its own partition.
 *
 * @warning Exact on an owned vertex only when the halo is VERTEX-COMPLETE.
 *          MFTLEnsembleUpdateVertexNormals() adds the one exchange that also
 *          makes the ghosts exact.
 */
void MFTLComputeVertexNormals(MFTL_Object *obj);

#endif  // MFTL_GEOMETRY_H
