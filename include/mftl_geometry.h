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

  // Calculate the magnitude of the cross product vector
  const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, cross_prod[0] * cross_prod[0] + cross_prod[1] * cross_prod[1] + cross_prod[2] * cross_prod[2]));
  if (area_out)
  {
    *area_out = 0.5 * mag;
  }
  if (normal_out)
  {
    const double inv_mag = 1.0 / mag;
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
  for (size_t i = 0; i < object->n_triangles; i++)
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
 * asserted in debug builds). Returns 0 for an empty object.
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
void MFTLMomentOfInertia(MFTL_Object* obj, double I[3][3], const double center[3], double density);

// =============================================================================
// RIGID-BODY TRANSFORMS (see src/general/transform.c)
// Operate on LOCAL (owned, non-ghost) vertices only; refresh ghosts afterwards
// via MFTLUpdateGhostPositions. Each transform recomputes triangle properties.
// Storage stays unwrapped — use MFTLWrapObjectIntoDomain explicitly if needed.
// =============================================================================

/** @brief In-place per-axis scale about the origin: pos[d] *= s[d] (MFTL1 MFTLScaleSurface). */
void MFTLScaleObject(MFTL_Object* obj, const double s[3]);

/** @brief In-place rigid translation: pos += d (MFTL1 MFTLMoveBody). */
void MFTLTranslateObject(MFTL_Object* obj, const double d[3]);

/** @brief In-place rotation about `centre`: pos = centre + R.(pos - centre).
 * R must be orthonormal (asserted in debug builds, det R = 1). */
void MFTLRotateObject(MFTL_Object* obj, const double R[3][3], const double centre[3]);

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

/** @brief Orthogonal containment test with MFTL_VERYSMALL slack (MFTL1 MFTLPointInsideObjectBox parity). */
int MFTLPointInsideAABB(const double center[3], const double dims[3], const double p[3]);

/** @brief Collective object axis-aligned bounding box: center + half-dimensions over owned vertices. */
void MFTLObjectAABB(MFTL_Object* obj, double center[3], double dims[3]);

// Test/debug instrumentation for the point-in-object BVH lifecycle.
int MFTLDebugPointQueryBVHBuilds(void);      // BVH builds performed by MFTLPointInsideObject so far
int MFTLDebugPointQueryLastIterations(void); // ray-casting attempts in the most recent call

#endif