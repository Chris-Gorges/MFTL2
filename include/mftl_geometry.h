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
  for (mftl_int_t i = 0; i < 3; ++i)
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
  for (mftl_int_t i = 0; i < 3; ++i)
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
    if (tri->owner_proc == object->domain->rank) MFTLGetTriangleProperties(object, tri, &tri->area, tri->normal, tri->center);
  }
}

#endif