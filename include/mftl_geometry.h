/*
 * @Author: Christian Gorges
 * @Date: 2025-08-10 09:12:55
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-21 15:33:57
 */

#ifndef MFTL_GEOMETRY_H
#define MFTL_GEOMETRY_H

#include "mftl_topology.h"
#include "mftl_core.h"
#include "mftl_constants.h"

/**
 * @brief Computes the shortest vector between two points in a periodic-aware
 * manner.
 *
 * This is the core function for all periodic geometry calculations. It applies
 * the "minimum image convention". 
 * @param domain The domain containing periodicity information.
 * @param vec_out The resulting shortest vector (p2 - p1).
 * @param p1 The starting point coordinates.
 * @param p2 The ending point coordinates.  
 */
static inline void MFTLGetMinimumImageVector(const MFTL_Domain *domain, double vec_out[3], const double p1[3], const double p2[3])
{
  // Start with the simple difference
  vec_out[0] = p2[0] - p1[0];
  vec_out[1] = p2[1] - p1[1];
  vec_out[2] = p2[2] - p1[2];

  // Apply periodic correction for each dimension
  for (int i = 0; i < 3; ++i)
  {
    if (domain->is_periodic[i])
    {
      vec_out[i] -= ((vec_out[i] > domain->domain_half_size[i]) * domain->domain_size[i]);
      vec_out[i] += ((vec_out[i] < -domain->domain_half_size[i]) * domain->domain_size[i]);
    }
  }
}

/**
 * @brief Wraps a position vector back into the periodic domain.
 *
 * This optimized version assumes the displacement in a single timestep is less
 * than the domain size, which is true for most stable simulations. It uses
 * simple comparisons which are faster than the fmod() function.
 *
 * @param domain The domain containing periodicity information.
 * @param pos The position vector to be wrapped (modified in-place).
 */
static inline void MFTLApplyPeriodicity(const MFTL_Domain *domain, double pos[3])
{
  for (int i = 0; i < 3; ++i)
  {
    if (domain->is_periodic[i])
    {
      if (pos[i] >= domain->global_max[i])
      {
        pos[i] -= domain->domain_size[i];
      }
      else if (pos[i] < domain->global_min[i])
      {
        pos[i] += domain->domain_size[i];
      }
    }
  }
}

/* ---- Stage 16B: image counters (plans/16 §3.1–§3.3) ------------------------------------ */

/* Wrap a vertex position that was displaced by less than one period from a wrapped state and
   count the wrap into its image counters (the continuous coordinate pos + img*L is unchanged).
   The position arithmetic is exactly that of MFTLApplyPeriodicity (bit-identical positions). */
static inline void MFTLWrapPosImg(const MFTL_Domain *domain, double pos[3], int img[3])
{
  for (int i = 0; i < 3; ++i)
  {
    if (domain->is_periodic[i])
    {
      if (pos[i] >= domain->global_max[i])
      {
        pos[i] -= domain->domain_size[i];
        img[i] += 1;
      }
      else if (pos[i] < domain->global_min[i])
      {
        pos[i] += domain->domain_size[i];
        img[i] -= 1;
      }
    }
  }
}

/* Displacement rule for a position that was REPLACED by an independently wrapped value new_pos
   (fits, smoothing results): the move old -> new is the minimum-image displacement, so the
   counters follow by the number of periods the wrapped difference jumps. */
static inline void MFTLImgFollow(const MFTL_Domain *domain, int img[3], const double old_pos[3], const double new_pos[3])
{
  for (int i = 0; i < 3; ++i)
  {
    if (!domain->is_periodic[i]) continue;
    const double diff = new_pos[i] - old_pos[i];
    if (diff > domain->domain_half_size[i])
      img[i] -= 1;
    else if (diff < -domain->domain_half_size[i])
      img[i] += 1;
  }
}

/* Continuous coordinate of v relative to a reference (pos_ref, img_ref), formed from the exact
   integer image difference (§3.5): Y = (img - img_ref) L + (pos - pos_ref). */
static inline void MFTLVertexRelative(const MFTL_Domain *domain, const MFTL_Vertex *v, const double pos_ref[3], const int img_ref[3], double Y[3])
{
  for (int i = 0; i < 3; ++i)
  {
    Y[i] = v->pos[i] - pos_ref[i];
    if (domain->is_periodic[i]) Y[i] += (double) (v->img[i] - img_ref[i]) * domain->domain_size[i];
  }
}

/* Edge lattice shift s(u,v) = round((X_v - X_u - d(u,v)) / L) (§3.3), exact in integers:
   img_v - img_u + j, where j is the wrap of the minimum-image displacement d. */
static inline void MFTLEdgeShift(const MFTL_Domain *domain, const MFTL_Vertex *u, const MFTL_Vertex *v, int s[3])
{
  for (int i = 0; i < 3; ++i)
  {
    s[i] = 0;
    if (!domain->is_periodic[i]) continue;
    const double diff = v->pos[i] - u->pos[i];
    const int j = (diff > domain->domain_half_size[i]) ? 1 : ((diff < -domain->domain_half_size[i]) ? -1 : 0);
    s[i] = v->img[i] - u->img[i] + j;
  }
}

/* Wrap a continuous input coordinate X into (pos, img) with img = floor((X - min) / L) on the
   periodic axes (continuous input, §3.2). */
static inline void MFTLWrapContinuous(const MFTL_Domain *domain, const double X[3], double pos[3], int img[3])
{
  for (int i = 0; i < 3; ++i)
  {
    pos[i] = X[i];
    img[i] = 0;
    if (!domain->is_periodic[i]) continue;
    const double k = floor((X[i] - domain->global_min[i]) / domain->domain_size[i]);
    if (k != 0.0)
    {
      pos[i] = X[i] - k * domain->domain_size[i];
      img[i] = (int) k;
    }
    if (pos[i] >= domain->global_max[i])  // round-off at the upper face
    {
      pos[i] -= domain->domain_size[i];
      img[i] += 1;
    }
    else if (pos[i] < domain->global_min[i])
    {
      pos[i] += domain->domain_size[i];
      img[i] -= 1;
    }
  }
}

/**
 * @brief True when the domain has at least one periodic axis.
 *
 * Use this to branch between the raw (non-periodic) arithmetic and a
 * periodic-safe formula when the two differ in floating point: the
 * non-periodic path must stay bit-identical to the pre-Stage-13 behaviour.
 */
static inline int MFTLDomainAnyPeriodic(const MFTL_Domain *domain)
{
  return domain->is_periodic[0] || domain->is_periodic[1] || domain->is_periodic[2];
}

/**
 * @brief Re-images v into the periodic image containing anchor:
 * out[k] = v[k] + round((anchor[k]-v[k])/L)*L on periodic axes, a plain copy
 * otherwise (so the non-periodic path is bit-identical to using v directly).
 *
 * Correct whenever the true distance |v - anchor| is < L/2 per periodic axis
 * (the minimum-image constraint, see MFTLCheckMinimumImageConstraint).
 */
static inline void MFTLUnwrapToAnchor(const MFTL_Domain *domain, double out[3], const double v[3], const double anchor[3])
{
  for (int k = 0; k < 3; ++k)
  {
    out[k] = v[k];
    if (domain->is_periodic[k]) out[k] += round((anchor[k] - v[k]) / domain->domain_size[k]) * domain->domain_size[k];
  }
}

/*
 * @brief Calculate an edge's length.
 *
 * @param mesh The mesh object.
 * @param he The half-edge whose length is to be calculated.
 * @return The length of the edge.
 */
static inline double MFTLGetEdgeLength(MFTL_Object *object, MFTL_HalfEdge *he)
{
  // Calculate Edge Vector for a specific HalfEdge 'he'
  MFTL_Vertex *v_start = MFTLHeOrigin(object, he);
  MFTL_Vertex *v_end = MFTLHeTarget(object, he);

  double edge_vec[3];
  MFTLGetMinimumImageVector(object->domain, edge_vec, v_start->pos, v_end->pos);

  // Now calculate the magnitude of the corrected vector
  return MFTL_SQRT(edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2]);
}

/*
 * @brief Calculate an edge's length.
 *
 * @param mesh The mesh object.
 * @param he The half-edge whose length is to be calculated.
 * @return The length of the edge.
 */
static inline double MFTLGetEdgeLengthSquared(MFTL_Object *object, MFTL_HalfEdge *he)
{
  // Calculate Edge Vector for a specific HalfEdge 'he'
  MFTL_Vertex *v_start = MFTLHeOrigin(object, he);
  MFTL_Vertex *v_end = MFTLHeTarget(object, he);

  double edge_vec[3];
  MFTLGetMinimumImageVector(object->domain, edge_vec, v_start->pos, v_end->pos);

  // Now calculate the magnitude of the corrected vector
  return edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2];
}

/**
 * @brief Calculates the fundamental geometric properties of a triangle in one
 * pass.
 *
 * This is the core function. It computes the area, normal, and
 * centroid of a triangle in a single pass to avoid redundant calculations.
 * Any of the output pointers can be NULL if that property is not needed.
 *
 * @param object The mftl object.
 * @param triangle The triangle to be processed.
 * @param area_out Pointer to a double where the calculated area will be stored.
 * Can be NULL.
 * @param normal_out Array where the normalized normal vector will be stored.
 * Can be NULL.
 * @param center_out Array where the centroid coordinates will be stored. Can be
 * NULL.
 */
static inline void MFTLGetTriangleProperties(const MFTL_Object *object, const MFTL_Triangle *triangle, double *area_out, double normal_out[3],
                                             double center_out[3])
{
  if (!triangle) return;

  // Get the three vertices of the triangle using the half-edge loop
  MFTL_Vertex *verts[3];
  MFTLGetTriangleVertices(object, triangle, verts);

  // --- Part 1: Calculate Normal and Area ---
  // Get the two edge vectors using our periodic-aware function
  double vec_a[3];  // v1 - v0
  double vec_b[3];  // v2 - v0
  MFTLGetMinimumImageVector(object->domain, vec_a, verts[0]->pos, verts[1]->pos);
  MFTLGetMinimumImageVector(object->domain, vec_b, verts[0]->pos, verts[2]->pos);

  // Compute the cross product. This vector is perpendicular to the triangle.
  // Its magnitude is twice the triangle's area.
  double cross_prod[3];
  cross_prod[0] = vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1];
  cross_prod[1] = vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2];
  cross_prod[2] = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0];

  // Calculate the magnitude of the cross product vector. Stage 15: the area is NOT floored
  // (the former floor sqrt(MFTL_VERYSMALL) = 1e-6 on |cross| silently clamped every triangle
  // below 5e-7 in area and de-normalised its normal — i.e. typical front-tracking meshes);
  // only the normalisation divides by max(|cross|, 1e-15), exactly as MFTL1 did.
  const double mag = MFTL_SQRT(cross_prod[0] * cross_prod[0] + cross_prod[1] * cross_prod[1] + cross_prod[2] * cross_prod[2]);
  if (area_out)
  {
    *area_out = 0.5 * mag;
  }
  if (normal_out)
  {
    const double inv_mag = 1.0 / MFTL_MAX(MFTL_VERYVERYSMALL, mag);
    normal_out[0] = cross_prod[0] * inv_mag;
    normal_out[1] = cross_prod[1] * inv_mag;
    normal_out[2] = cross_prod[2] * inv_mag;
  }

  // --- Centroid (Periodic Safe) ---
  if (center_out)
  {
    // 1. Calculate relative to v0 (preserves correct periodic shape)
    double offset[3];
    offset[0] = (vec_a[0] + vec_b[0]) * MFTL_ONE_THIRD;  // mult is faster than div
    offset[1] = (vec_a[1] + vec_b[1]) * MFTL_ONE_THIRD;
    offset[2] = (vec_a[2] + vec_b[2]) * MFTL_ONE_THIRD;

    center_out[0] = verts[0]->pos[0] + offset[0];
    center_out[1] = verts[0]->pos[1] + offset[1];
    center_out[2] = verts[0]->pos[2] + offset[2];

    // 2. Wrap it (preserves correct array bounds)
    MFTLApplyPeriodicity(object->domain, center_out);
  }
}

/**
 * @brief Calculates the area of a triangle. (Wrapper for the optimized
 * function)
 */
static inline double MFTLGetTriangleArea(const MFTL_Object *object, const MFTL_Triangle *triangle)
{
  double area;
  MFTLGetTriangleProperties(object, triangle, &area, NULL, NULL);
  return area;
}

/**
 * @brief Calculates the normal vector of a triangle. (Wrapper for the optimized
 * function)
 */
static inline void MFTLGetTriangleNormal(const MFTL_Object *object, const MFTL_Triangle *triangle, double normal_out[3])
{
  MFTLGetTriangleProperties(object, triangle, NULL, normal_out, NULL);
}

/**
 * @brief Calculates the geometric center (centroid) of a triangle. (Wrapper for
 * the optimized function)
 */
static inline void MFTLGetTriangleCenter(const MFTL_Object *object, const MFTL_Triangle *triangle, double center_out[3])
{
  MFTLGetTriangleProperties(object, triangle, NULL, NULL, center_out);
}

/**
 * @brief Calculates the geometric properties of all triangles and store them in each triangle.
 */
static inline void MFTLComputeTriangleProperties(const MFTL_Object *object)
{
  MFTL_Triangle *tri_base = object->triangles;
  for (int i = 0; i < object->n_triangles; i++)
  {
    MFTL_Triangle *tri = &tri_base[i];
    if (tri->flags & MFTL_FLAG_DEAD) continue;  // remeshing leftovers until finalization
    if (tri->owner_proc == object->domain->rank) MFTLGetTriangleProperties(object, tri, &tri->area, tri->normal, tri->center);
  }
}

// =============================================================================
// OBJECT INTEGRAL QUANTITIES (see src/general/geometry_object.c)
// MFTL1-compatible definitions; parallel (Allreduce over comm_cart); owned
// triangles only (owner_proc == rank), ghosts never counted. Periodic-aware:
// vertices are re-imaged into one contiguous image (anchored at a globally
// agreed body vertex) before integrating — exact for bodies smaller than half
// the domain. The SIGN of the volume follows the mesh winding convention.
// =============================================================================

/**
 * @brief Enclosed volume of a closed object: 1/6 * sum_owned v1.(v2 x v3).
 * Requires a closed, consistently-oriented surface (MFTLCheckMeshIntegrity is
 * asserted in debug builds). Returns 0 for an empty object. A body that wraps a
 * periodic direction or is declared open aborts with a message (collective;
 * use the phase-set API, plans/16 §3.6). MFTLMomentOfInertia likewise.
 */
double MFTLObjectVolume(MFTL_Object* obj);

/**
 * @brief Area-weighted SURFACE centroid (NOT the volume centroid):
 * sum_owned Area*(v1+v2+v3) / (3*total_area). Returns the total surface area.
 * The centroid is expressed in the periodic image anchored at the object's
 * first owned vertex (may lie outside the domain for seam-straddling bodies).
 */
double MFTLSurfaceCenter(MFTL_Object* obj, double cm[3]);

/**
 * @brief MFTL1 lumped-tetrahedron moment of inertia about `center` with
 * uniform `density` (mass lumped at each tetrahedron centroid; NOT the exact
 * tetrahedron-covariance formula). `center` must be expressed in the same
 * periodic image as the mesh — e.g. the centroid returned by
 * MFTLSurfaceCenter.
 */
void MFTLMomentOfInertia(MFTL_Object* obj, double inertia[3][3], const double center[3], double density);

/**
 * @brief Minimum-image constraint check (Stage 13): on a periodic domain the
 * unambiguous "nearest image" requires the period on each periodic axis to
 * exceed twice the largest feature extent. This collective routine verifies:
 *   1. the longest (minimum-image) owned edge is < min over periodic axes of
 *      domain_half_size[k]  (unambiguous min-image per edge), AND
 *   2. the per-periodic-axis unwrapped body extent (about a collective
 *      anchor) is < domain_half_size[k]  (the single-anchor unwraps used by
 *      the integrals and point queries stay valid).
 * No-op (MFTL_SUCCESS) when no axis is periodic. On violation a message is
 * printed on rank 0 and MFTL_ERROR is returned (identically on all ranks).
 * Never asserts: callers may probe.
 */
int MFTLCheckMinimumImageConstraint(const MFTL_Object* obj);

// =============================================================================
// Stage 16C: GENERALIZED GEOMETRY KERNELS (plans/16 §4.1-§4.2; src/general/kernels.c)
// Local frames: minimum-image differences from an anchor vertex (edges < half a period).
// =============================================================================
/** @brief Minimum-image displacement b - a. */
void MFTLVertexDelta(const MFTL_Object *obj, const MFTL_Vertex *a, const MFTL_Vertex *b, double d[3]);
/** @brief Minimum-image vector origin -> target of a half-edge. */
void MFTLEdgeVector(const MFTL_Object *obj, const MFTL_HalfEdge *he, double d[3]);
/** @brief Continuous coordinate X = pos + img L (for large counters prefer MFTLVertexRelative). */
void MFTLVertexContinuous(const MFTL_Object *obj, const MFTL_Vertex *v, double X[3]);
/** @brief Triangle corners in the periodic image of its first corner (half-edge order). */
void MFTLTriangleUnwrapped(const MFTL_Object *obj, const MFTL_Triangle *tri, double P[3][3]);
/** @brief BOUNDARY half-edge geometry: length, unit tangent t (along the half-edge), in-surface
 *  outward conormal nu = t x n (n: unit triangle normal) and the continuous midpoint (may be
 *  NULL). MFTL_ERROR for a non-boundary or degenerate half-edge. */
int MFTLBoundaryEdgeGeometry(const MFTL_Object *obj, int he_idx, double *len, double t[3], double nu[3], double mid_X[3]);
/** @brief Stage 16E (plans/16 §6): for a BOUNDARY half-edge, the in-surface distance delta from
 *  its triangle's centroid to the edge midpoint and the triangle height h_T = 2 A_T / L_e over the
 *  edge (local frame). Note delta >= h_T / 3 for every non-degenerate triangle. MFTL_ERROR for a
 *  non-boundary or degenerate half-edge. */
int MFTLBoundaryEdgeCentroidDistance(const MFTL_Object *obj, int he_idx, double *delta, double *h_T);
/** @brief Sum of the corner angles at v over its stored fan; returns the MFTL_RING_* class (the sum
 *  is valid for CLOSED and OPEN_COMPLETE). Angle defect of an interior vertex: 2 pi - sum. */
int MFTLVertexAngleSum(const MFTL_Object *obj, int v, double *sum);
/** @brief Geodesic turning angle pi - sum of corner angles of a boundary (OPEN_COMPLETE) vertex. */
int MFTLVertexTurningAngle(const MFTL_Object *obj, int v, double *theta);
/** @brief Loop neighbours of a boundary vertex (prev = -1 if not stored); MFTL_ERROR when v has no
 *  stored outgoing BOUNDARY half-edge. */
int MFTLBoundaryNeighbours(const MFTL_Object *obj, int v, int *prev, int *next);
/** @brief Thin-region pairs: stored triangles with opposing normals, minimum-image centre distance
 *  below threshold, no shared vertex; *pairs (malloc'ed) holds (gid, gid) pairs. Rank-local. */
int MFTLFindThinRegionPairs(MFTL_Object *obj, double threshold, int **pairs, int *npairs);
/** @brief Proximity pairs between stored triangles of o0 and o1 whose local-frame boxes are within
 *  `distance` in the minimum image; *pairs holds (gid in o0, gid in o1). Rank-local. */
int MFTLFindProximityPairs(MFTL_Object *o0, MFTL_Object *o1, double distance, int **pairs, int *npairs);

// =============================================================================
// RIGID-BODY TRANSFORMS (see src/general/transform.c)
// Operate on LOCAL (owned, non-ghost) vertices only; refresh ghosts afterwards
// via MFTLUpdateGhostPositions. Each transform recomputes triangle properties.
// Storage stays unwrapped — use MFTLWrapObjectIntoDomain explicitly if needed.
// =============================================================================

/** @brief In-place per-axis scale about the origin: pos[d] *= s[d] (MFTL1 MFTLScaleSurface).
 * Stage 16B, periodic domains (collective): continuous coordinates relative to the body
 * reference point, about the image of the origin nearest to it (lattice-gauge invariant, §3.2);
 * MFTL_ERROR (body unchanged) if the scaling maps a winding vector off the periodic lattice. */
int MFTLScaleObject(MFTL_Object* obj, const double s[3]);

/** @brief In-place rigid translation: pos += d (MFTL1 MFTLMoveBody). */
void MFTLTranslateObject(MFTL_Object* obj, const double d[3]);

/** @brief In-place rotation about `centre`: pos = centre + R.(pos - centre).
 * R must be orthonormal (asserted in debug builds, det R = 1).
 * Stage 16B, periodic domains (collective): about the image of `centre` nearest to the body
 * reference point, in continuous coordinates relative to it (lattice-gauge invariant, §3.2);
 * MFTL_ERROR (body unchanged) if R maps a winding vector off the periodic lattice. */
int MFTLRotateObject(MFTL_Object* obj, const double R[3][3], const double centre[3]);

/** @brief Explicitly wrap local vertex positions back into the periodic domain. */
void MFTLWrapObjectIntoDomain(MFTL_Object* obj);

// =============================================================================
// OBJECT GENERATORS (see src/general/generators.c)
// Mirror MFTLGenerateSphere's lifecycle: rank 0 builds the mesh, the gid
// counters are broadcast, the caller redistributes. Closed generators
// (cylinder/box/cube) are outward-wound: MFTLObjectVolume is positive, the
// surface is closed (Euler chi = 2). MFTLGenerateFace is an OPEN patch.
// =============================================================================

/**
 * @brief Capped cylinder along +Z, z in [0, L], cross-section in the XY plane.
 * MFTL1 parameterization: Np = round(radius^0.9 / (pi*dr)) circumferential
 * segments, Nz = round(L/(10*dr) + 1) rings; ellipticity in [0,1) scales the
 * x semi-axis by sqrt(1-e^2), (-1,0) scales y (MFTL1 convention). Flat cap
 * fans close both ends (MFTL1's own cylinder is an open tube; the closed form
 * is per plans/04-generators.md).
 */
void MFTLGenerateCylinder(MFTL_Object* obj, double dr, double radius, double L, double ellipticity);

/** @brief Closed axis-aligned box with corners lo and hi (2 triangles per face). */
void MFTLGenerateBox(MFTL_Object* obj, const double lo[3], const double hi[3]);

/** @brief Closed cube [-1,1]^3, refined `refine` times by flat 1-to-4 subdivision. */
void MFTLGenerateCube(MFTL_Object* obj, int refine);

/** @brief Open quad patch (2 triangles, diagonal v0-v3). Volume undefined. */
void MFTLGenerateFace(MFTL_Object* obj, const double v[4][3]);

// =============================================================================
// VERTEX NORMALS / CURVATURES / ROUGHNESS (see src/general/curvature.c)
// One-ring quantities in MFTL1's conventions: mean curvature 2/R, Gaussian
// 2/R^2 on a sphere. PRECONDITION: every owned vertex's 1-ring of triangles
// must be stored (after redistribution: MFTLUpdateGhostPositions +
// MFTLExtendGhostLayer(ens, >=1)); triangle properties must be current.
// Ghost copies of normal/curvature/area refresh via MFTLUpdateGhostPositions.
// =============================================================================

/** @brief Angle/edge-length-weighted average of incident triangle normals (unit length). */
void MFTLComputeVertexNormals(MFTL_Object* obj);

/** @brief Cotangent Laplace-Beltrami curvatures + mixed-Voronoi area per owned vertex. */
void MFTLComputeVertexCurvatures(MFTL_Object* obj);

/** @brief Stage 16C (plans/16 §4.2): extrapolated curvature |2H| of every owned boundary vertex
 *  (OPEN_COMPLETE fan) from a least-squares paraboloid over its open two-ring stencil (>= 6
 *  points), stored in curv_extrap with MFTL_VFLAG_CURV_EXTRAPOLATED (cleared when the fit is not
 *  possible). MFTLComputeVertexCurvatures flags every owned vertex without a closed fan with
 *  MFTL_VFLAG_CURV_UNDEFINED (no silent skip). */
void MFTLComputeExtrapolatedCurvatures(MFTL_Object* obj);

/* Stage 15 (L5): single-vertex variants (MFTL1 MFTL_OneRingVertexNormal /
   MFTL_VertexLaplaceBeltrami). They work on ANY stored vertex (owned or ghost) whose stored
   one-ring is closed, recompute the ring triangles' cached normals (normal) or areas
   (curvature) from the current positions first, and return MFTL_ERROR (vertex untouched)
   when the ring is not closed. */
int MFTLComputeVertexNormalAt(MFTL_Object* obj, int vi);
int MFTLComputeVertexCurvatureAt(MFTL_Object* obj, int vi);

/**
 * @brief Surface roughness: returns the GLOBAL geometric-Laplacian average
 * (MFTL1 parity); stores per-vertex dihedral mean/variance in
 * roughness_mean / roughness_variance.
 */
double MFTLComputeRoughness(MFTL_Object* obj);

// =============================================================================
// POINT QUERIES (see src/general/point_query.c)
// Point-in-object is COLLECTIVE (broadcast ray + Allreduce); requires a
// closed surface and current triangle properties. No OBB is provided: the
// only consumer box (multiflow moveIBM) is axis-aligned (see MFTLObjectAABB)
// and candidate acceleration uses the existing per-object LBVH.
// =============================================================================

/**
 * @brief Exact point-triangle distance (6-region Voronoi classification,
 * MFTL1 port): nearest-feature distance for vertex/edge/interior regions.
 */
double MFTLPointTriangleDistance(const MFTL_Object* obj, int tri_idx, const double p[3]);

/**
 * @brief Collective point-in-object classification by ray casting (MFTL1
 * parity): 1 inside, 0 outside, -1 on the surface. One random direction is
 * broadcast from rank 0; grazing vertex/edge hits make ALL ranks retry (max
 * 100). On periodic domains the probe is re-imaged next to the object.
 */
int MFTLPointInsideObject(MFTL_Object* obj, const double p[3]);

/* Stage 15 (L2): caller-held point query. MFTLPointQueryCreate builds the object's LBVH
   ONCE (collective); MFTLPointQueryInside / MFTLPointQueryInsideMany then classify points
   against it with exactly the MFTLPointInsideObject semantics (same shared ray sequence,
   same retry rule, 1 inside / 0 outside / -1 on the surface) without rebuilding anything.
   InsideMany classifies n points (the SAME n points on every rank) with O(retries)
   collectives in total instead of O(n). The query is valid until the object is changed
   structurally or its vertices move. On periodic domains the query falls back to
   MFTLPointInsideObject per point (the unwrap-to-anchor step cannot be cached). */
typedef struct MFTL_PointQuery MFTL_PointQuery;
MFTL_PointQuery* MFTLPointQueryCreate(MFTL_Object* obj);
void MFTLPointQueryDestroy(MFTL_PointQuery** q);
int MFTLPointQueryInside(MFTL_PointQuery* q, const double p[3]);
int MFTLPointQueryInsideMany(MFTL_PointQuery* q, int n, const double* xyz, int* result);

/** @brief Orthogonal containment test with MFTL_VERYSMALL slack (MFTL1 MFTLPointInsideObjectBox parity). */
int MFTLPointInsideAABB(const double center[3], const double dims[3], const double p[3]);

/** @brief Collective object axis-aligned bounding box: center + half-dimensions over owned vertices. */
void MFTLObjectAABB(MFTL_Object* obj, double center[3], double dims[3]);

// Test/debug instrumentation for the point-in-object BVH lifecycle.
int MFTLDebugPointQueryBVHBuilds(void);      // BVH builds performed by MFTLPointInsideObject so far
int MFTLDebugPointQueryLastIterations(void); // ray-casting attempts in the most recent call

#endif