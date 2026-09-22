/*
 * Point queries: point-in-object (ray casting), point-triangle distance, and
 * axis-aligned bounding-box containment — MFTL1 parity per
 * plans/07-point-queries-and-obb.md.
 *
 *  - MFTLPointTriangleDistance: verbatim port of MFTL1's 6-region Voronoi
 *    classification (serial/pointtriangledistance.c) — two rotations into a
 *    canonical frame, mirror normalization, then zone tests giving the
 *    vertex/edge/interior nearest feature.
 *  - MFTLPointInsideObject: collective ray casting (generalfunctions/point.c):
 *    one random direction broadcast from rank 0, candidate triangles gathered
 *    by traversing the object's EXISTING LBVH (mftl_bvh.c) with a slab test
 *    (node AABBs inflated to absorb their float precision), crossings counted
 *    on owned triangles only, MPI_Allreduce(SUM) of the stab count and of the
 *    degeneracy flag; odd => inside, even => outside, flag < 0 => on surface
 *    (-1). Grazing vertex/edge hits (SameSide == 2) flag degeneracy and all
 *    ranks retry together with a fresh direction (max 100, MFTL1).
 *  - MFTLPointInsideAABB / MFTLObjectAABB: the ORTHOGONAL box multiflow's
 *    moveIBM uses (MFTLPointInsideObjectBox parity) plus its construction.
 *
 * OBB decision (per the plan): no oriented covariance box is provided. The
 * only in-tree consumer (multiflow moveIBM) builds an ORTHOGONAL box, which
 * the AABB helpers cover; acceleration for candidate gathering is the
 * existing per-object LBVH. An oriented box would add the covariance+Jacobi
 * machinery with no current consumer.
 *
 * Periodicity (Stage 13): on a periodic domain the whole local mesh is
 * temporarily re-imaged into one contiguous image about a collective anchor
 * (first owned vertex of the lowest-rank owner, broadcast), the probe is
 * re-imaged next to it, and the wrapped storage is restored before returning.
 * Exact under the minimum-image constraint (MFTLCheckMinimumImageConstraint).
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_bvh.h"

// MFTL1's MFTL_VERYSMALL is 1e-15 == MFTL2's MFTL_VERYVERYSMALL.
#define MFTL_PQ_TOL MFTL_VERYVERYSMALL

// MFTL1's MFTL_MultiplyTensorVector uses the COLUMN convention:
// out[i] = sum_j R[j][i] * x[j]. Computed into temporaries first (callers
// rotate in place).
static inline void mftlTensorVector(double out[3], const double R[3][3], const double x[3])
{
  double t0 = R[0][0] * x[0] + R[1][0] * x[1] + R[2][0] * x[2];
  double t1 = R[0][1] * x[0] + R[1][1] * x[1] + R[2][1] * x[2];
  double t2 = R[0][2] * x[0] + R[1][2] * x[1] + R[2][2] * x[2];
  out[0] = t0;
  out[1] = t1;
  out[2] = t2;
}

/* ------------------------------------------------------------------------- */
/* Point-triangle distance (verbatim port of the 6-region Voronoi method)    */
/* ------------------------------------------------------------------------- */

static inline double mftlPointPointDistance(const double a[3], const double b[3])
{
  double d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
  return sqrt(d0 * d0 + d1 * d1 + d2 * d2);
}

static inline double mftlPointLineDistance(const double a[3], const double b[3], const double p[3])
{
  double v1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double v2[3] = {p[0] - a[0], p[1] - a[1], p[2] - a[2]};
  double c[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
  double lv1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  if (lv1 < MFTL_PQ_TOL) return mftlPointPointDistance(p, a);
  return sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) / lv1;
}

double MFTLPointTriangleDistance(const MFTL_Object* obj, int tri_idx, const double p[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(tri_idx >= 0 && tri_idx < obj->n_triangles);

  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);

  double Point[3][3];
  double P1[3];
  double translate[3];

  memcpy(translate, vv[0]->pos, 3 * sizeof(double));
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k) Point[i][k] = vv[i]->pos[k] - translate[k];
  for (int k = 0; k < 3; ++k) P1[k] = p[k] - translate[k];

  // Rotate the triangle so that V0V1 aligns with the z axis
  double xy, xyz;
  double R[3][3];
  if (fabs(Point[1][0]) > MFTL_PQ_TOL || fabs(Point[1][1]) > MFTL_PQ_TOL)
  {
    xy = sqrt(Point[1][0] * Point[1][0] + Point[1][1] * Point[1][1]) + MFTL_PQ_TOL;
    xyz = sqrt(Point[1][0] * Point[1][0] + Point[1][1] * Point[1][1] + Point[1][2] * Point[1][2]) + MFTL_PQ_TOL;
    R[0][0] = Point[1][1] / xy;
    R[1][0] = -Point[1][0] / xy;
    R[2][0] = 0.0;
    R[0][1] = Point[1][0] * Point[1][2] / (xy * xyz);
    R[1][1] = Point[1][1] * Point[1][2] / (xy * xyz);
    R[2][1] = -xy / xyz;
    R[0][2] = Point[1][0] / xyz;
    R[1][2] = Point[1][1] / xyz;
    R[2][2] = Point[1][2] / xyz;
    for (int i = 0; i < 3; ++i) mftlTensorVector(Point[i], R, Point[i]);
    mftlTensorVector(P1, R, P1);
  }

  // Rotate so that V2 lies on the yz plane
  if (fabs(Point[2][0]) > MFTL_PQ_TOL)
  {
    xy = sqrt(Point[2][0] * Point[2][0] + Point[2][1] * Point[2][1]) + MFTL_PQ_TOL;
    R[0][0] = Point[2][1] / xy;
    R[1][0] = -Point[2][0] / xy;
    R[2][0] = 0.0;
    R[0][1] = Point[2][0] / xy;
    R[1][1] = Point[2][1] / xy;
    R[2][1] = 0.0;
    R[0][2] = 0.0;
    R[1][2] = 0.0;
    R[2][2] = 1.0;
    for (int i = 0; i < 3; ++i) mftlTensorVector(Point[i], R, Point[i]);
    mftlTensorVector(P1, R, P1);
  }

  // Mirror into the first quarter (P1.y>0 side convention of MFTL1)
  if (Point[1][2] >= 0.0)
  {
    if (Point[2][1] < 0.0)
    {
      Point[2][1] *= -1.0;
      P1[1] *= -1.0;
    }
  }
  else
  {
    Point[1][2] *= -1.0;
    Point[2][2] *= -1.0;
    P1[2] *= -1.0;
    if (Point[2][1] < 0.0)
    {
      Point[2][1] *= -1.0;
      P1[1] *= -1.0;
    }
  }

  // Six Voronoi zones (edge order V0V2, V1V2, V0V1)
  double distance;
  if (((P1[1] - Point[0][1]) * (Point[2][2] - Point[0][2]) - (P1[2] - Point[0][2]) * (Point[2][1] - Point[0][1])) > 0.0)
  {
    if ((-(P1[1] - Point[0][1]) * (Point[2][1] - Point[0][1]) - (P1[2] - Point[0][2]) * (Point[2][2] - Point[0][2])) >= 0.0)
      distance = mftlPointPointDistance(P1, Point[0]);  // zone 1: closest to V0
    else if ((-(P1[1] - Point[2][1]) * (Point[2][1] - Point[0][1]) - (P1[2] - Point[2][2]) * (Point[2][2] - Point[0][2])) <= 0.0)
      distance = mftlPointPointDistance(P1, Point[2]);  // zone 3: closest to V2
    else
      distance = mftlPointLineDistance(Point[0], Point[2], P1);  // zone 2: edge V0V2
  }
  else if (((P1[1] - Point[1][1]) * (Point[2][2] - Point[1][2]) - (P1[2] - Point[1][2]) * (Point[2][1] - Point[1][1])) < 0.0)
  {
    if ((-(P1[1] - Point[2][1]) * (Point[2][1] - Point[1][1]) - (P1[2] - Point[2][2]) * (Point[2][2] - Point[1][2])) <= 0.0)
      distance = mftlPointPointDistance(P1, Point[2]);  // zone 3: closest to V2
    else if ((-(P1[1] - Point[1][1]) * (Point[2][1] - Point[1][1]) - (P1[2] - Point[1][2]) * (Point[2][2] - Point[1][2])) >= 0.0)
      distance = mftlPointPointDistance(P1, Point[1]);  // zone 5: closest to V1
    else
      distance = mftlPointLineDistance(Point[1], Point[2], P1);  // zone 4: edge V1V2
  }
  else if (((P1[1] - Point[0][1]) * (Point[1][2] - Point[0][2]) - (P1[2] - Point[1][2]) * (Point[1][1] - Point[0][1])) < 0.0)
  {
    if ((-(P1[1] - Point[1][1]) * (Point[1][1] - Point[0][1]) - (P1[2] - Point[1][2]) * (Point[1][2] - Point[0][2])) <= 0.0)
      distance = mftlPointPointDistance(P1, Point[1]);  // zone 3: closest to V1
    else if ((-(P1[1] - Point[0][1]) * (Point[1][1] - Point[0][1]) - (P1[2] - Point[0][2]) * (Point[1][2] - Point[0][2])) >= 0.0)
      distance = mftlPointPointDistance(P1, Point[0]);  // zone 1: closest to V0
    else
      distance = mftlPointLineDistance(Point[0], Point[1], P1);  // zone 6: edge V0V1
  }
  else
    distance = fabs(P1[0]);  // interior: distance to the triangle plane

  return distance + MFTL_PQ_TOL;
}

/* ------------------------------------------------------------------------- */
/* Ray casting (collective)                                                  */
/* ------------------------------------------------------------------------- */

// SameSide: 1 strictly same side, 2 on the edge/vertex, 0 opposite side.
static int mftlSameSide(const double p1[3], const double p2[3], const double a[3], const double b[3])
{
  double v1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  double v2[3] = {p1[0] - a[0], p1[1] - a[1], p1[2] - a[2]};
  double cp1[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
  v2[0] = p2[0] - a[0];
  v2[1] = p2[1] - a[1];
  v2[2] = p2[2] - a[2];
  double cp2[3] = {v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]};
  double d = cp1[0] * cp2[0] + cp1[1] * cp2[1] + cp1[2] * cp2[2];
  if (d > MFTL_PQ_TOL)
    return 1;
  else if (d >= -MFTL_PQ_TOL)
    return 2;
  else
    return 0;
}

static int mftlPointInTriangle(const double p[3], const double a[3], const double b[3], const double c[3], int* degenerate)
{
  int s0 = mftlSameSide(p, a, b, c);
  int s1 = mftlSameSide(p, b, a, c);
  int s2 = mftlSameSide(p, c, a, b);
  if (s0 && s1 && s2)
  {
    // grazing a vertex/edge: flag degeneracy unless an on-surface verdict
    // (-999) has already been recorded
    if ((s0 == 2 || s1 == 2 || s2 == 2) && *degenerate >= 0) *degenerate = 1;
    return 1;
  }
  return 0;
}

// Returns 1 when the segment (p, rayEnd) crosses the triangle plane with
// 0 < t < 1 (InterPoint filled), 2 when p lies on the plane, 0 otherwise.
static int mftlRayPlaneIntersection(const double v0[3], const double normal[3], const double p[3], const double ray_end[3],
                                    double inter[3])
{
  double D = v0[0] * normal[0] + v0[1] * normal[1] + v0[2] * normal[2];
  double ray[3] = {ray_end[0] - p[0], ray_end[1] - p[1], ray_end[2] - p[2]};
  double numerator = D - (p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2]);
  double denumerator = ray[0] * normal[0] + ray[1] * normal[1] + ray[2] * normal[2];

  if (fabs(denumerator) < MFTL_PQ_TOL)  // ray parallel to the triangle
  {
    if (fabs(numerator) < MFTL_PQ_TOL)  // p on the triangle plane
      return 2;
    return 0;
  }

  double t = numerator / denumerator;
  for (int k = 0; k < 3; ++k) inter[k] = p[k] + t * ray[k];
  if (t > 0.0 && t < 1.0)
    return 1;  // proper crossing within the segment
  if (fabs(numerator) < MFTL_PQ_TOL)
    return 2;  // p lies on the triangle plane
  return 0;
}

// Port of MFTL_StabbedTriangle: count a crossing of one owned triangle,
// set degenerate to 1 on grazing hits and to -999 when p itself is on the
// triangle (on-surface verdict).
static int mftlStabbedTriangle(const MFTL_Object* obj, int tri_idx, const double p[3], int* degenerate, const double ray_end[3])
{
  const MFTL_Triangle* tri = &obj->triangles[tri_idx];
  if (tri->is_ghost || tri->owner_proc != obj->domain->rank) return 0;

  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, tri, vv);
  double inter[3];

  int code = mftlRayPlaneIntersection(vv[0]->pos, tri->normal, p, ray_end, inter);
  if (code == 2)
  {
    if (mftlPointInTriangle(p, vv[0]->pos, vv[1]->pos, vv[2]->pos, degenerate))
    {
      *degenerate = -999;  // p is on the surface
      return 1;
    }
    return 0;
  }
  if (code == 1) return mftlPointInTriangle(inter, vv[0]->pos, vv[1]->pos, vv[2]->pos, degenerate);
  return 0;
}

// Slab test of the segment (p, rayEnd) against an AABB, inflated by eps to
// absorb the float precision of the LBVH node boxes.
static int mftlSegmentHitsAABB(const float bmin[3], const float bmax[3], double eps, const double p[3], const double ray_end[3])
{
  double tmin = 0.0, tmax = 1.0;
  for (int k = 0; k < 3; ++k)
  {
    double d = ray_end[k] - p[k];
    double lo = (double) bmin[k] - eps;
    double hi = (double) bmax[k] + eps;
    if (fabs(d) < MFTL_PQ_TOL)
    {
      if (p[k] < lo || p[k] > hi) return 0;
    }
    else
    {
      double inv = 1.0 / d;
      double t0 = (lo - p[k]) * inv;
      double t1 = (hi - p[k]) * inv;
      if (t0 > t1)
      {
        double tmp = t0;
        t0 = t1;
        t1 = tmp;
      }
      if (t0 > tmin) tmin = t0;
      if (t1 < tmax) tmax = t1;
      if (tmin > tmax) return 0;
    }
  }
  return 1;
}

// Traverse the LBVH, counting crossings on owned triangles (MFTL1's
// MFTL_StabbedTriangles with the OBB tree replaced by the LBVH).
static int mftlStabbedTrianglesBVH(const MFTL_Object* obj, const MFTL_BVH* bvh, const double p[3], int* degenerate,
                                   const double ray_end[3])
{
  int stabbed = 0;
  if (!bvh || bvh->root_index < 0) return 0;

  double eps = 1e-4;  // float-AABB inflation (domain-scale queries)
  int* stack = malloc((size_t) bvh->nodes_used * sizeof(int));
  int sp = 0;
  stack[sp++] = bvh->root_index;
  while (sp > 0)
  {
    int ni = stack[--sp];
    const MFTL_BVHNode* node = &bvh->nodes[ni];
    if (!mftlSegmentHitsAABB(node->aabb_min, node->aabb_max, eps, p, ray_end)) continue;

    if (node->payload & 1ull)
    {
      // leaf
      int prim_count = (int) ((node->payload >> 1) & 0x7FFFFFFFu);
      khint64_t first = node->payload >> 32;
      for (int q = 0; q < prim_count; ++q)
      {
        int tri_idx = bvh->prim_indices[(int) first + q];
        stabbed += mftlStabbedTriangle(obj, tri_idx, p, degenerate, ray_end);
      }
    }
    else
    {
      khint64_t left = node->payload >> 1;
      stack[sp++] = (int) left;
      stack[sp++] = (int) left + 1;
    }
  }
  free(stack);
  return stabbed;
}

// Deterministic non-zero components in [-1, 1) (rank 0 only; broadcast).
static void mftlRandomRayLocal(int iteration, double dir[3])
{
  unsigned int s = 0x9E3779B9u + 0x85EBCA6Bu * (unsigned int) (iteration + 1);
  for (int k = 0; k < 3; ++k)
  {
    do
    {
      s ^= s << 13;
      s ^= s >> 17;
      s ^= s << 5;
    } while (s == 0u);
    double r = 2.0 * ((double) s / 4294967296.0) - 1.0;
    if (fabs(r) < MFTL_PQ_TOL) r = (r < 0.0) ? -0.5 : 0.5;
    dir[k] = r;
  }
}

// Test/debug instrumentation: counts BVH builds performed by
// MFTLPointInsideObject (must be one per call, independent of retries).
static int g_mftl_pq_bvh_builds = 0;
static int g_mftl_pq_last_iterations = 0;

int MFTLDebugPointQueryBVHBuilds(void)
{
  return g_mftl_pq_bvh_builds;
}

int MFTLDebugPointQueryLastIterations(void)
{
  return g_mftl_pq_last_iterations;
}

// Collective unwrap anchor (same protocol as MFTLObjectUnwrapAnchor in
// geometry_object.c): the wrapped position of the first owned triangle vertex
// on the lowest-rank owner, broadcast so every rank re-images into the SAME
// image. Returns 0 when the object has no owned triangles anywhere.
static int mftlPQUnwrapAnchor(MFTL_Object* obj, double anchor[3])
{
  MFTL_Domain* dom = obj->domain;
  int rank = dom->rank;

  double local_anchor[3] = {0.0, 0.0, 0.0};
  int have_local = 0;
  for (int t = 0; t < obj->n_triangles && !have_local; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != rank) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    memcpy(local_anchor, vv[0]->pos, 3 * sizeof(double));
    have_local = 1;
  }

  int local_root = have_local ? rank : dom->nproc;
  int root = dom->nproc;
  MPI_Allreduce(&local_root, &root, 1, MPI_INT, MPI_MIN, dom->comm_cart);
  if (root == dom->nproc) return 0;  // empty object: nothing to stab

  if (rank == root) memcpy(anchor, local_anchor, 3 * sizeof(double));
  MPI_Bcast(anchor, 3, MPI_DOUBLE, root, dom->comm_cart);
  return 1;
}

int MFTLPointInsideObject(MFTL_Object* obj, const double p_in[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(p_in != MFTL_NULL);
  // Odd/even parity requires a closed surface.
  MFTL_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);

  MFTL_Domain* domain = obj->domain;
  int rank = domain->rank;
  int nproc = domain->nproc;

  double p[3] = {p_in[0], p_in[1], p_in[2]};

  // Stage 13: on a periodic domain the wrapped storage of a seam-straddling
  // body is torn across the domain faces, so neither its AABB nor the raw ray
  // tests see the true geometry. Temporarily re-image ALL local vertex
  // positions (owned and ghost) into one contiguous image about a collective
  // anchor, re-image the probe next to it, ray-cast against the unwrapped
  // mesh, then restore the wrapped storage. Exact under the minimum-image
  // constraint (MFTLCheckMinimumImageConstraint). Non-periodic domains keep
  // the legacy path untouched.
  double* saved = NULL;
  if (MFTLDomainAnyPeriodic(domain))
  {
    double anchor[3];
    if (mftlPQUnwrapAnchor(obj, anchor))  // collective
    {
      saved = malloc((size_t) obj->n_vertices * 3 * sizeof(double));
      MFTL_ASSERT(saved != MFTL_NULL || obj->n_vertices == 0);
      for (int i = 0; i < obj->n_vertices; ++i)
      {
        MFTL_Vertex* v = &obj->vertices[i];
        memcpy(saved + 3 * i, v->pos, 3 * sizeof(double));
        MFTLUnwrapToAnchor(domain, v->pos, v->pos, anchor);
      }
      // re-image the probe next to the anchor
      for (int k = 0; k < 3; ++k)
      {
        if (!domain->is_periodic[k]) continue;
        p[k] += round((anchor[k] - p[k]) / domain->domain_size[k]) * domain->domain_size[k];
      }
    }
  }
  else
  {
    // Non-periodic: legacy re-image next to the object's AABB centre
    // (a no-op without periodic axes; kept for collective-call parity).
    double c[3], d[3];
    MFTLObjectAABB(obj, c, d);
    for (int k = 0; k < 3; ++k)
    {
      if (!domain->is_periodic[k]) continue;
      double half = 0.5 * domain->domain_size[k];
      while (p[k] - c[k] > half) p[k] -= domain->domain_size[k];
      while (p[k] - c[k] < -half) p[k] += domain->domain_size[k];
    }
  }

  // Ray length: 50x the domain diagonal (MFTL1) guarantees exit.
  double domain_size[3] = {domain->domain_size[0], domain->domain_size[1], domain->domain_size[2]};
  double ray_length = 50.0 * sqrt(domain_size[0] * domain_size[0] + domain_size[1] * domain_size[1] + domain_size[2] * domain_size[2]);

  // The mesh is static during ray casting: build the candidate BVH ONCE and
  // reuse it across all retry iterations (was: rebuilt per attempt).
  MFTL_BVH* bvh = MFTL_GenerateLBVH(obj);
  g_mftl_pq_bvh_builds++;

  int maxiter = 100;
  int i = 0;
  int flag = 0;
  int stabbed_glob = 0;
  while (i++ < maxiter)
  {
    int degenerate = 0;
    double dir[3];

    // One shared random direction: rank 0 draws, everyone receives.
    if (rank == 0) mftlRandomRayLocal(i - 1, dir);
    MPI_Bcast(dir, 3, MPI_DOUBLE, 0, domain->comm_cart);

    double ray_end[3] = {p[0] + dir[0] * ray_length, p[1] + dir[1] * ray_length, p[2] + dir[2] * ray_length};

    int stabbed_loc = mftlStabbedTrianglesBVH(obj, bvh, p, &degenerate, ray_end);

    MPI_Allreduce(&stabbed_loc, &stabbed_glob, 1, MPI_INT, MPI_SUM, domain->comm_cart);
    flag = 0;
    MPI_Allreduce(&degenerate, &flag, 1, MPI_INT, MPI_SUM, domain->comm_cart);
    if (flag == 0 || flag < 0) break;  // clean verdict, or on-surface everywhere
  }
  g_mftl_pq_last_iterations = i;

  MFTL_DestroyBVH(bvh);

  // restore the wrapped storage
  if (saved)
  {
    for (int i2 = 0; i2 < obj->n_vertices; ++i2) memcpy(obj->vertices[i2].pos, saved + 3 * i2, 3 * sizeof(double));
    free(saved);
  }

  if (i > maxiter) return 1;  // MFTL1: after 100 retries assume inside
  if (flag < 0) return -1;    // on the surface
  return stabbed_glob % 2;    // odd => inside
}

/* ------------------------------------------------------------------------- */
/* Axis-aligned bounding boxes (multiflow moveIBM's ortho box)               */
/* ------------------------------------------------------------------------- */

void MFTLObjectAABB(MFTL_Object* obj, double center[3], double dims[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_Domain* domain = obj->domain;

  double lmin[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double lmax[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != domain->rank) continue;
    for (int k = 0; k < 3; ++k)
    {
      if (v->pos[k] < lmin[k]) lmin[k] = v->pos[k];
      if (v->pos[k] > lmax[k]) lmax[k] = v->pos[k];
    }
  }

  double gmin[3], gmax[3];
  MPI_Allreduce(lmin, gmin, 3, MPI_DOUBLE, MPI_MIN, domain->comm_cart);
  MPI_Allreduce(lmax, gmax, 3, MPI_DOUBLE, MPI_MAX, domain->comm_cart);

  for (int k = 0; k < 3; ++k)
  {
    if (gmin[k] > gmax[k])
    {
      center[k] = 0.0;  // empty object
      dims[k] = 0.0;
    }
    else
    {
      center[k] = 0.5 * (gmin[k] + gmax[k]);
      dims[k] = 0.5 * (gmax[k] - gmin[k]);
    }
  }
}

int MFTLPointInsideAABB(const double center[3], const double dims[3], const double p[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (p[k] < center[k] - dims[k] - MFTL_VERYSMALL) return 0;
    if (p[k] > center[k] + dims[k] + MFTL_VERYSMALL) return 0;
  }
  return 1;
}

