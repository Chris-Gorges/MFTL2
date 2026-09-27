/*
 * MFTL2 geometry — periodic primitives (minimum image, wrapping, image counters, edge lattice
 * shifts), triangle properties, and the public geometry API: single-body integrals, local-frame
 * kernels, transforms, generators, normals/curvature/roughness and point queries.
 *
 * Periodic storage convention (plans/16 §3.1): pos is WRAPPED into [min, max) on periodic axes
 * and the integer image counters img[3] record the periodic image, so X = pos + img ⊙ L is the
 * continuous coordinate. Every edge is shorter than half a period, so the minimum-image
 * difference of two vertices is their true edge vector. On a non-periodic domain every helper
 * here reduces to the plain arithmetic (bit-identical).
 */

#ifndef MFTL_GEOMETRY_H
#define MFTL_GEOMETRY_H

#include "mftl_topology.h"
#include "mftl_core.h"
#include "mftl_constants.h"

/**
 * @brief Minimum-image difference vec_out = p2 - p1: on each periodic axis one period is added
 * or subtracted when |p2 - p1| exceeds half the period. Exact for wrapped positions whose true
 * separation is below half a period (every mesh edge).
 */
static inline void MFTLGetMinimumImageVector(const MFTL_Domain *domain, double vec_out[3], const double p1[3], const double p2[3])
{
  vec_out[0] = p2[0] - p1[0];
  vec_out[1] = p2[1] - p1[1];
  vec_out[2] = p2[2] - p1[2];

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
 * @brief Wraps a point that lies less than one period outside the box back into [min, max) on
 * the periodic axes (single comparison per axis, no fmod).
 *
 * This wraps a bare point only. A VERTEX position must be wrapped with MFTLWrapPosImg, which
 * also counts the wrap into the image counters (plans/16 §3.2).
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

/* ---- Image counters (plans/16 §3.1–§3.3) ------------------------------------------------ */

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
 * Used to branch between the plain (non-periodic) arithmetic and a periodic-safe formula when
 * the two differ in floating point: the non-periodic path must stay bit-identical to the plain
 * formula.
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

/* Length of the edge of a half-edge (minimum image). */
static inline double MFTLGetEdgeLength(MFTL_Object *object, MFTL_HalfEdge *he)
{
  MFTL_Vertex *v_start = MFTLHeOrigin(object, he);
  MFTL_Vertex *v_end = MFTLHeTarget(object, he);

  double edge_vec[3];
  MFTLGetMinimumImageVector(object->domain, edge_vec, v_start->pos, v_end->pos);
  return MFTL_SQRT(edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2]);
}

/* Squared length of the edge of a half-edge (minimum image). */
static inline double MFTLGetEdgeLengthSquared(MFTL_Object *object, MFTL_HalfEdge *he)
{
  MFTL_Vertex *v_start = MFTLHeOrigin(object, he);
  MFTL_Vertex *v_end = MFTLHeTarget(object, he);

  double edge_vec[3];
  MFTLGetMinimumImageVector(object->domain, edge_vec, v_start->pos, v_end->pos);
  return edge_vec[0] * edge_vec[0] + edge_vec[1] * edge_vec[1] + edge_vec[2] * edge_vec[2];
}

/**
 * @brief Area, unit normal and centroid of a triangle from its current vertex positions (any
 * output may be NULL).
 *
 * The two edge vectors are minimum-image differences from the first vertex, so seam-straddling
 * triangles are exact; the normal is (v1 - v0) x (v2 - v0) normalised (outward for an outward
 * wound body); the centroid is formed relative to v0 and then wrapped into the box.
 */
static inline void MFTLGetTriangleProperties(const MFTL_Object *object, const MFTL_Triangle *triangle, double *area_out, double normal_out[3],
                                             double center_out[3])
{
  if (!triangle) return;

  MFTL_Vertex *verts[3];
  MFTLGetTriangleVertices(object, triangle, verts);

  double vec_a[3];  // v1 - v0
  double vec_b[3];  // v2 - v0
  MFTLGetMinimumImageVector(object->domain, vec_a, verts[0]->pos, verts[1]->pos);
  MFTLGetMinimumImageVector(object->domain, vec_b, verts[0]->pos, verts[2]->pos);

  // |cross| = twice the area
  double cross_prod[3];
  cross_prod[0] = vec_a[1] * vec_b[2] - vec_a[2] * vec_b[1];
  cross_prod[1] = vec_a[2] * vec_b[0] - vec_a[0] * vec_b[2];
  cross_prod[2] = vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0];

  // The area is NOT floored (a former floor sqrt(MFTL_VERYSMALL) = 1e-6 on |cross| silently
  // clamped every triangle below 5e-7 in area and de-normalised its normal — i.e. typical
  // front-tracking meshes); only the normalisation divides by max(|cross|, 1e-15), exactly as
  // MFTL1 did.
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

  if (center_out)
  {
    // relative to v0 (seam-safe), then wrapped into the box
    double offset[3];
    offset[0] = (vec_a[0] + vec_b[0]) * MFTL_ONE_THIRD;
    offset[1] = (vec_a[1] + vec_b[1]) * MFTL_ONE_THIRD;
    offset[2] = (vec_a[2] + vec_b[2]) * MFTL_ONE_THIRD;

    center_out[0] = verts[0]->pos[0] + offset[0];
    center_out[1] = verts[0]->pos[1] + offset[1];
    center_out[2] = verts[0]->pos[2] + offset[2];
    MFTLApplyPeriodicity(object->domain, center_out);
  }
}

/** @brief Triangle area (MFTLGetTriangleProperties). */
static inline double MFTLGetTriangleArea(const MFTL_Object *object, const MFTL_Triangle *triangle)
{
  double area;
  MFTLGetTriangleProperties(object, triangle, &area, NULL, NULL);
  return area;
}

/** @brief Triangle unit normal (MFTLGetTriangleProperties). */
static inline void MFTLGetTriangleNormal(const MFTL_Object *object, const MFTL_Triangle *triangle, double normal_out[3])
{
  MFTLGetTriangleProperties(object, triangle, NULL, normal_out, NULL);
}

/** @brief Triangle centroid, wrapped into the box (MFTLGetTriangleProperties). */
static inline void MFTLGetTriangleCenter(const MFTL_Object *object, const MFTL_Triangle *triangle, double center_out[3])
{
  MFTLGetTriangleProperties(object, triangle, NULL, NULL, center_out);
}

/**
 * @brief Stores area, normal and centre of every OWNED live triangle (ghost copies receive
 * theirs from the owner by communication; dead remeshing leftovers are skipped).
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
// SINGLE-BODY INTEGRAL QUANTITIES (see src/geometry/geometry_object.c)
// MFTL1-compatible definitions; collective (Allreduce over comm_cart); owned
// triangles only (owner_proc == rank), ghosts never counted. Periodic domains:
// continuous coordinates in the image of one collectively agreed anchor vertex
// (exact from the image counters, bodies of any extent). The body must be
// COMPACT on every periodic axis, and volume/inertia need a closed body; outside
// that contract the call aborts collectively (use mftl_phase_set.h). The SIGN of
// the volume follows the winding: V > 0 for the outward convention of MFTL2.
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
 * Defined for open bodies too. On a periodic domain the centroid is a continuous
 * coordinate in the image of the anchor vertex (it may lie outside the box).
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
 * @brief Minimum-image constraint check (collective): the longest minimum-image
 * edge of the owned triangles must be shorter than the smallest half-period
 * (every edge vector is then unambiguous, plans/16 §3.5). There is no condition
 * on the body extent (the image counters carry every vertex's image).
 * No-op (MFTL_SUCCESS) when no axis is periodic. On violation a message is
 * printed on rank 0 and MFTL_ERROR is returned (identically on all ranks).
 * Never asserts: callers may probe.
 */
int MFTLCheckMinimumImageConstraint(const MFTL_Object* obj);

// =============================================================================
// LOCAL-FRAME GEOMETRY KERNELS (plans/16 §4.1-§4.2; src/geometry/kernels.c)
// Minimum-image differences from an anchor vertex (edges < half a period), so
// seam-crossing, wrapping and open surfaces are handled alike.
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
/** @brief For a BOUNDARY half-edge (plans/16 §6), the in-surface distance delta from
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
// TRANSFORMS (see src/geometry/transform.c)
// Operate on owned vertices only; refresh ghosts afterwards via
// MFTLUpdateGhostPositions. Each transform recomputes triangle properties.
// MFTLTranslateObject does not wrap (pos leaves the box, img unchanged): call
// MFTLWrapObjectIntoDomain to wrap pos and count the wraps into img. Scaling and
// rotation on a periodic domain store wrapped positions with updated counters.
// =============================================================================

/** @brief In-place per-axis scale about the origin: pos[d] *= s[d] (MFTL1 MFTLScaleSurface).
 * Periodic domains (collective): continuous coordinates relative to the body
 * reference point, about the image of the origin nearest to it (lattice-gauge invariant, §3.2);
 * MFTL_ERROR (body unchanged) if the scaling maps a winding vector off the periodic lattice. */
int MFTLScaleObject(MFTL_Object* obj, const double s[3]);

/** @brief In-place rigid translation: pos += d, no wrapping (MFTL1 MFTLMoveBody). Rank-local. */
void MFTLTranslateObject(MFTL_Object* obj, const double d[3]);

/** @brief In-place rotation about `centre`: pos = centre + R.(pos - centre).
 * R must be orthonormal (asserted in debug builds, det R = 1).
 * Periodic domains (collective): about the image of `centre` nearest to the body
 * reference point, in continuous coordinates relative to it (lattice-gauge invariant, §3.2);
 * MFTL_ERROR (body unchanged) if R maps a winding vector off the periodic lattice. */
int MFTLRotateObject(MFTL_Object* obj, const double R[3][3], const double centre[3]);

/** @brief Wraps owned vertex positions into the box on periodic axes, counting every wrap
 *  (any number of periods) into img; the continuous coordinates are unchanged. */
void MFTLWrapObjectIntoDomain(MFTL_Object* obj);

// =============================================================================
// OBJECT GENERATORS (see src/generators/generators.c)
// Mirror MFTLGenerateSphere's lifecycle: rank 0 builds the mesh, the gid
// counters are broadcast, the caller redistributes. Closed generators
// (cylinder/box/cube) are outward-wound: MFTLObjectVolume is positive, the
// surface is closed (Euler chi = 2). MFTLGenerateFace is an OPEN patch and
// is declared open (one boundary loop, tag 0).
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

/** @brief Open quad patch (2 triangles, diagonal v0-v3), declared open. Volume undefined. */
void MFTLGenerateFace(MFTL_Object* obj, const double v[4][3]);

// =============================================================================
// VERTEX NORMALS / CURVATURES / ROUGHNESS (see src/geometry/curvature.c)
// One-ring quantities in MFTL1's conventions: mean curvature 2/R, Gaussian
// 2/R^2 on a sphere. PRECONDITION: the fan of every owned vertex is stored
// (guaranteed after MFTLRedistribute* by ghost completion G2), ghost positions
// and triangle properties are current. Ghost copies of normal/curvature/area
// refresh via MFTLUpdateGhostPositions. Boundary vertices: fan normal; curvature
// flagged MFTL_VFLAG_CURV_UNDEFINED (see MFTLComputeExtrapolatedCurvatures).
// =============================================================================

/** @brief Angle/edge-length-weighted average of incident triangle normals (unit length). */
void MFTLComputeVertexNormals(MFTL_Object* obj);

/** @brief Cotangent Laplace-Beltrami curvatures + mixed-Voronoi area per owned vertex. */
void MFTLComputeVertexCurvatures(MFTL_Object* obj);

/** @brief Extrapolated curvature |2H| of every owned boundary vertex (plans/16 §4.2)
 *  (OPEN_COMPLETE fan) from a least-squares paraboloid over its open two-ring stencil (>= 6
 *  points), stored in curv_extrap with MFTL_VFLAG_CURV_EXTRAPOLATED (cleared when the fit is not
 *  possible). MFTLComputeVertexCurvatures flags every owned vertex without a closed fan with
 *  MFTL_VFLAG_CURV_UNDEFINED (no silent skip). */
void MFTLComputeExtrapolatedCurvatures(MFTL_Object* obj);

/* Single-vertex variants (MFTL1 MFTL_OneRingVertexNormal /
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
// POINT QUERIES (see src/geometry/point_query.c)
// Point-in-object is COLLECTIVE (broadcast ray + Allreduce); requires a
// closed surface and current triangle properties; on a periodic domain the body
// must be compact (a wrapping body aborts: use MFTLPhaseSetInside). No oriented
// box is provided: the only consumer box (multiflow moveIBM) is axis-aligned
// (see MFTLObjectAABB) and candidate gathering uses an LBVH.
// =============================================================================

/**
 * @brief Exact point-triangle distance (6-region Voronoi classification,
 * MFTL1 port): nearest-feature distance for vertex/edge/interior regions,
 * plus MFTL1's 1e-15 offset. Uses the stored positions as given (no minimum
 * image).
 */
double MFTLPointTriangleDistance(const MFTL_Object* obj, int tri_idx, const double p[3]);

/**
 * @brief Collective point-in-object classification by ray casting (MFTL1
 * parity): 1 inside, 0 outside, -1 on the surface. One random direction is
 * broadcast from rank 0; grazing vertex/edge hits make ALL ranks retry (max
 * 100). On periodic domains the probe is re-imaged next to the object.
 */
int MFTLPointInsideObject(MFTL_Object* obj, const double p[3]);

/* Caller-held point query. MFTLPointQueryCreate builds the object's LBVH
   ONCE (collective); MFTLPointQueryInside / MFTLPointQueryInsideMany then classify points
   against it with exactly the MFTLPointInsideObject semantics (same shared ray sequence,
   same retry rule, 1 inside / 0 outside / -1 on the surface) without rebuilding anything.
   InsideMany classifies n points (the SAME n points on every rank) with O(retries)
   collectives in total instead of O(n). The query is valid until the object is changed
   structurally or its vertices move. On periodic domains the query falls back to
   MFTLPointInsideObject per point (the unwrap-to-anchor step cannot be cached).
   KNOWN DEFECT (deferred): on a non-periodic domain a rank that stores no triangle of the
   object builds no LBVH and takes that per-point fallback while the other ranks do not; the
   collectives then mismatch and the call deadlocks. Use the query only when every rank stores
   at least one triangle of the object (MFTLPointInsideObject has no such restriction). */
typedef struct MFTL_PointQuery MFTL_PointQuery;
MFTL_PointQuery* MFTLPointQueryCreate(MFTL_Object* obj);
void MFTLPointQueryDestroy(MFTL_PointQuery** q);
int MFTLPointQueryInside(MFTL_PointQuery* q, const double p[3]);
int MFTLPointQueryInsideMany(MFTL_PointQuery* q, int n, const double* xyz, int* result);

/** @brief Orthogonal containment test with MFTL_VERYSMALL slack (MFTL1 MFTLPointInsideObjectBox parity). */
int MFTLPointInsideAABB(const double center[3], const double dims[3], const double p[3]);

/** @brief Collective object axis-aligned bounding box: center + half-dimensions over owned vertices. */
void MFTLObjectAABB(MFTL_Object* obj, double center[3], double dims[3]);

// Test instrumentation for the point-in-object BVH lifecycle.
int MFTLDebugPointQueryBVHBuilds(void);      // BVH builds performed by MFTLPointInsideObject so far
int MFTLDebugPointQueryLastIterations(void); // ray-casting attempts in the most recent call

#endif