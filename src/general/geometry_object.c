/*
 * Object-level integral quantities: enclosed volume, area-weighted surface
 * centroid, and lumped-tetra moment of inertia.
 *
 * Definitions reproduce MFTL1 exactly:
 *   - Volume      : 1/6 * sum_owned v1.(v2 x v3)          (objectvolume.c:VolumeFace)
 *   - Surf. center: sum_owned Area*(v1+v2+v3) / (3*Area)  (objectcentre.c:CenterOfAreaFace)
 *   - Inertia     : lumped tetrahedron sum about a center (momentofinertia.c)
 *
 * Periodicity: stored vertex positions may wrap at periodic faces, which
 * corrupts integrals computed from absolute positions. Before integrating,
 * every owned triangle's vertices are re-imaged into ONE contiguous image
 * anchored at a globally agreed body vertex (the first owned vertex of the
 * lowest-rank owner, broadcast). This is exact for bodies whose extent is
 * strictly smaller than half the domain in every periodic axis. Re-imaging
 * is done per vertex via round((anchor - v)/L)*L, which keeps whole bodies in
 * a single image — a per-triangle "minimum image relative to v1" is NOT
 * sufficient for absolute-position sums (centroid, tetra apex distance).
 *
 * The moment-of-inertia caller must pass `center` expressed in the same
 * image (e.g. the centroid returned by MFTLSurfaceCenter).
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

/**
 * @brief Determines the global unwrap anchor for an object: the (wrapped)
 * position of its first owned vertex on the lowest-rank owner, broadcast so
 * every rank re-images into the SAME image.
 * @return 0 if the object has no owned triangles anywhere (all sums are 0).
 */
static int MFTLObjectUnwrapAnchor(const MFTL_Object* obj, double anchor[3])
{
  const MFTL_Domain* dom = obj->domain;
  int rank = dom->rank;

  double local_anchor[3] = {0.0, 0.0, 0.0};
  int have_local = 0;
  for (int t = 0; t < obj->n_triangles && !have_local; ++t)
  {
    const MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != rank) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    memcpy(local_anchor, vv[0]->pos, 3 * sizeof(double));
    have_local = 1;
  }

  int local_root = have_local ? rank : dom->nproc;
  int root = dom->nproc;
  MPI_Allreduce(&local_root, &root, 1, MPI_INT, MPI_MIN, dom->comm_cart);
  if (root == dom->nproc) return 0;  // empty object: nothing to integrate

  // On the root, seed the broadcast buffer from the local scan; on the other
  // ranks MPI_Bcast fills it. (Without the seed, the root would integrate
  // towards an uninitialized anchor.)
  if (rank == root) memcpy(anchor, local_anchor, 3 * sizeof(double));
  MPI_Bcast(anchor, 3, MPI_DOUBLE, root, dom->comm_cart);
  return 1;
}

/**
 * @brief Re-images v into the periodic image containing anchor.
 * Identity for non-periodic axes. Requires |anchor - v| < L/2 (guaranteed
 * when the body extent is strictly smaller than half the domain).
 */
static inline void mftlUnwrapToAnchor(const MFTL_Domain* dom, double out[3], const double v[3], const double anchor[3])
{
  for (int k = 0; k < 3; ++k)
  {
    out[k] = v[k];
    if (dom->is_periodic[k]) out[k] += round((anchor[k] - v[k]) / dom->domain_size[k]) * dom->domain_size[k];
  }
}

double MFTLObjectVolume(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_Domain* dom = obj->domain;
  MFTL_ASSERT(dom != MFTL_NULL);

  // Volume is only defined for a closed, consistently-oriented surface.
  MFTL_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);

  double anchor[3];
  if (!MFTLObjectUnwrapAnchor(obj, anchor)) return 0.0;

  double local = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0]->pos, anchor);
    mftlUnwrapToAnchor(dom, p2, vv[1]->pos, anchor);
    mftlUnwrapToAnchor(dom, p3, vv[2]->pos, anchor);

    // MFTL1 VolumeFace: v1.(v2 x v3), expanded cyclically.
    local += p1[0] * (p2[1] * p3[2] - p2[2] * p3[1]) + p2[0] * (p3[1] * p1[2] - p3[2] * p1[1]) +
             p3[0] * (p1[1] * p2[2] - p1[2] * p2[1]);
  }

  double total = 0.0;
  MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
  return total / 6.0;
}

double MFTLSurfaceCenter(MFTL_Object* obj, double cm[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_Domain* dom = obj->domain;
  MFTL_ASSERT(dom != MFTL_NULL);

  cm[0] = cm[1] = cm[2] = 0.0;

  double anchor[3];
  if (!MFTLObjectUnwrapAnchor(obj, anchor)) return 0.0;

  double local[4] = {0.0, 0.0, 0.0, 0.0};
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0]->pos, anchor);
    mftlUnwrapToAnchor(dom, p2, vv[1]->pos, anchor);
    mftlUnwrapToAnchor(dom, p3, vv[2]->pos, anchor);

    // Area from the re-imaged (true) edges: 0.5 * |(p2-p1) x (p3-p1)|.
    double x[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    double y[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
    double nx = x[1] * y[2] - x[2] * y[1];
    double ny = x[2] * y[0] - x[0] * y[2];
    double nz = x[0] * y[1] - x[1] * y[0];
    double area = 0.5 * sqrt(nx * nx + ny * ny + nz * nz);

    local[0] += area * (p1[0] + p2[0] + p3[0]);
    local[1] += area * (p1[1] + p2[1] + p3[1]);
    local[2] += area * (p1[2] + p2[2] + p3[2]);
    local[3] += area;
  }

  double total[4] = {0.0, 0.0, 0.0, 0.0};
  double area = 0.0;
  MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
  MPI_Allreduce(&local[3], &area, 1, MPI_DOUBLE, MPI_SUM, dom->comm_cart);

  if (area != 0.0)
  {
    cm[0] = total[0] / (3.0 * area);
    cm[1] = total[1] / (3.0 * area);
    cm[2] = total[2] / (3.0 * area);
  }

  return area;
}

void MFTLMomentOfInertia(MFTL_Object* obj, double inertia[3][3], const double center[3], double density)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_Domain* dom = obj->domain;
  MFTL_ASSERT(dom != MFTL_NULL);

  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) inertia[i][j] = 0.0;

  double anchor[3];
  if (!MFTLObjectUnwrapAnchor(obj, anchor)) return;

  double Ilocal[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0]->pos, anchor);
    mftlUnwrapToAnchor(dom, p2, vv[1]->pos, anchor);
    mftlUnwrapToAnchor(dom, p3, vv[2]->pos, anchor);

    // MFTL1 lumped-tetrahedron form (momentofinertia.c).
    double a[3] = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    double b[3] = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
    double c[3] = {center[0] - p1[0], center[1] - p1[1], center[2] - p1[2]};

    double dV = fabs((a[0] * b[1] * c[2] + a[1] * b[2] * c[0] + a[2] * b[0] * c[1] - a[2] * b[1] * c[0] - a[1] * b[0] * c[2] - a[0] * b[2] * c[1]) / 6.0);

    double cm[3] = {(p1[0] + p2[0] + p3[0] + center[0]) / 4.0, (p1[1] + p2[1] + p3[1] + center[1]) / 4.0,
                    (p1[2] + p2[2] + p3[2] + center[2]) / 4.0};
    double r[3] = {cm[0] - center[0], cm[1] - center[1], cm[2] - center[2]};

    Ilocal[0][0] += density * (r[1] * r[1] + r[2] * r[2]) * dV;
    Ilocal[1][1] += density * (r[0] * r[0] + r[2] * r[2]) * dV;
    Ilocal[2][2] += density * (r[0] * r[0] + r[1] * r[1]) * dV;

    Ilocal[0][1] += -density * r[0] * r[1] * dV;
    Ilocal[1][0] += -density * r[0] * r[1] * dV;
    Ilocal[0][2] += -density * r[0] * r[2] * dV;
    Ilocal[2][0] += -density * r[0] * r[2] * dV;
    Ilocal[1][2] += -density * r[1] * r[2] * dV;
    Ilocal[2][1] += -density * r[1] * r[2] * dV;
  }

  // One Allreduce per matrix row (MFTL1 layout).
  MPI_Allreduce(Ilocal[0], inertia[0], 3, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
  MPI_Allreduce(Ilocal[1], inertia[1], 3, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
  MPI_Allreduce(Ilocal[2], inertia[2], 3, MPI_DOUBLE, MPI_SUM, dom->comm_cart);
}

int MFTLCheckMinimumImageConstraint(const MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  const MFTL_Domain* dom = obj->domain;
  MFTL_ASSERT(dom != MFTL_NULL);

  if (!MFTLDomainAnyPeriodic(dom)) return MFTL_SUCCESS;

  double half_min = MFTL_DBL_MAX;  // smallest half-period over periodic axes
  for (int k = 0; k < 3; ++k)
    if (dom->is_periodic[k] && dom->domain_half_size[k] < half_min) half_min = dom->domain_half_size[k];

  // --- 1. Longest minimum-image edge over owned triangles ---
  double local_max_edge = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    const MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank || tri->is_ghost) continue;
    if (tri->flags & MFTL_FLAG_DEAD) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    for (int e = 0; e < 3; ++e)
    {
      double d[3];
      MFTLGetMinimumImageVector(dom, d, vv[e]->pos, vv[(e + 1) % 3]->pos);
      double len = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (len > local_max_edge) local_max_edge = len;
    }
  }
  double max_edge = 0.0;
  MPI_Allreduce(&local_max_edge, &max_edge, 1, MPI_DOUBLE, MPI_MAX, dom->comm_cart);

  if (max_edge >= half_min)
  {
    if (dom->rank == 0)
      fprintf(stderr,
              "ERROR: MFTLCheckMinimumImageConstraint: object %d violates the minimum-image constraint: "
              "longest edge %.6g >= smallest half-period %.6g. Increase the domain or coarsen/shrink the body.\n",
              obj->global_id, max_edge, half_min);
    return MFTL_ERROR;
  }

  // --- 2. Unwrapped body extent per periodic axis ---
  double anchor[3];
  if (!MFTLObjectUnwrapAnchor(obj, anchor)) return MFTL_SUCCESS;  // no owned triangles anywhere: nothing to check

  double lmin[3] = {MFTL_DBL_MAX, MFTL_DBL_MAX, MFTL_DBL_MAX};
  double lmax[3] = {-MFTL_DBL_MAX, -MFTL_DBL_MAX, -MFTL_DBL_MAX};
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost || v->owner_proc != dom->rank) continue;
    if (v->flags & MFTL_FLAG_DEAD) continue;
    double u[3];
    mftlUnwrapToAnchor(dom, u, v->pos, anchor);
    for (int k = 0; k < 3; ++k)
    {
      if (u[k] < lmin[k]) lmin[k] = u[k];
      if (u[k] > lmax[k]) lmax[k] = u[k];
    }
  }
  double gmin[3], gmax[3];
  MPI_Allreduce(lmin, gmin, 3, MPI_DOUBLE, MPI_MIN, dom->comm_cart);
  MPI_Allreduce(lmax, gmax, 3, MPI_DOUBLE, MPI_MAX, dom->comm_cart);

  for (int k = 0; k < 3; ++k)
  {
    if (!dom->is_periodic[k]) continue;
    double extent = gmax[k] - gmin[k];
    if (extent >= dom->domain_half_size[k])
    {
      if (dom->rank == 0)
        fprintf(stderr,
                "ERROR: MFTLCheckMinimumImageConstraint: object %d violates the minimum-image constraint: "
                "axis %d extent %.6g >= half-period %.6g. Single-anchor unwraps (integrals, point queries) are "
                "ambiguous for bodies this large — increase the domain or shrink the body.\n",
                obj->global_id, k, extent, dom->domain_half_size[k]);
      return MFTL_ERROR;
    }
  }

  return MFTL_SUCCESS;
}
