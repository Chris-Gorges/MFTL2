/*
 * MFTL2 geometry — single-body integrals: enclosed volume, area-weighted surface centroid,
 * lumped-tetrahedron moment of inertia, and the runtime minimum-image check.
 *
 * Definitions reproduce MFTL1 exactly:
 *   - Volume      : 1/6 * sum_owned v1.(v2 x v3)          (objectvolume.c:VolumeFace)
 *   - Surf. center: sum_owned Area*(v1+v2+v3) / (3*Area)  (objectcentre.c:CenterOfAreaFace)
 *   - Inertia     : lumped tetrahedron sum about a center (momentofinertia.c)
 * Sums run over OWNED triangles only and are combined with MPI_Allreduce(SUM), so the result
 * does not depend on the ghost layers.
 *
 * Periodicity (plans/16 §3.5): stored positions are wrapped, so absolute-position sums use
 * continuous coordinates in the image of one collectively agreed anchor vertex (the first
 * vertex of the first owned triangle on the lowest rank that owns a triangle). They are formed
 * from the exact integer image difference, anchor + (img - img_anchor) L + (pos - anchor), so
 * bodies of any extent are exact; there is no object-extent restriction. On non-periodic axes
 * the arithmetic is the plain position (bit-identical to the non-periodic formula).
 *
 * Validity: these single-body calls are defined only for a body that is COMPACT on every
 * periodic axis; volume and inertia additionally need a closed body (not declared open). A
 * call outside that contract aborts collectively with a message; the phase-set API
 * (mftl_phase_set.h) is the general, status-returning interface. The inertia caller must pass
 * `center` in the same image (e.g. the centroid returned by MFTLSurfaceCenter).
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
#include "mftl_images.h"
#include "geometry_internal.h"

/**
 * @brief Collective anchor of the integrals: (pos, img) of the first vertex of the first owned
 * triangle on the lowest rank owning a triangle, broadcast so every rank evaluates in the SAME
 * image.
 * @return 0 if the object has no owned triangles anywhere (all sums are 0).
 */
int mftlObjectUnwrapAnchor(const MFTL_Object* obj, double anchor[3], int anchor_img[3])
{
  const MFTL_Domain* dom = obj->domain;
  int rank = dom->rank;

  double local_anchor[3] = {0.0, 0.0, 0.0};
  int local_img[3] = {0, 0, 0};
  int have_local = 0;
  for (int t = 0; t < obj->n_triangles && !have_local; ++t)
  {
    const MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != rank) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    memcpy(local_anchor, vv[0]->pos, 3 * sizeof(double));
    memcpy(local_img, vv[0]->img, 3 * sizeof(int));
    have_local = 1;
  }

  int local_root = have_local ? rank : dom->nproc;
  int root = dom->nproc;
  MPI_Allreduce(&local_root, &root, 1, MPI_INT, MPI_MIN, dom->comm_cart);
  if (root == dom->nproc) return 0;  // empty object: nothing to integrate

  // the root seeds the broadcast buffer from its own scan
  if (rank == root)
  {
    memcpy(anchor, local_anchor, 3 * sizeof(double));
    memcpy(anchor_img, local_img, 3 * sizeof(int));
  }
  MPI_Bcast(anchor, 3, MPI_DOUBLE, root, dom->comm_cart);
  MPI_Bcast(anchor_img, 3, MPI_INT, root, dom->comm_cart);
  return 1;
}

/* Integrals over a periodic domain need a body that is compact along every periodic axis
   (volume, centroid and inertia of a wrapping body are not defined; use the phase-set API,
   plans/16 §3.5). Collective-safe: the flags are identical on all ranks. */
static void mftlRequireCompact(MFTL_Object* obj, const char* what)
{
  MFTLObjectEnsureWinding(obj);  // collective; an unknown winding state is never taken as compact
  if (MFTLObjectIsCompact(obj)) return;
  if (obj->domain->rank == 0)
    fprintf(stderr, "MFTL: %s of body %d is not defined: the body wraps a periodic direction (use MFTLPhaseSetVolume/Centre)\n", what,
            obj->global_id);
  MPI_Barrier(obj->domain->comm_world);  // collective check: the message is out before any rank aborts
  MPI_Abort(obj->domain->comm_world, 1);
}

/* Volume-based quantities (volume, moment of inertia) also need a CLOSED surface: a declared-open
   body encloses no volume (plans/16 §3.6, §12 "mass properties derived from open surfaces"); the
   integrity assertion alone only acts in debug builds. The area-based surface centre
   stays defined for open bodies. */
static void mftlRequireEnclosed(MFTL_Object* obj, const char* what)
{
  mftlRequireCompact(obj, what);
  if (!obj->declared_open) return;
  if (obj->domain->rank == 0)
    fprintf(stderr, "MFTL: %s of body %d is not defined: the body is an open surface (use MFTLPhaseSetVolume with wall closures)\n", what,
            obj->global_id);
  MPI_Barrier(obj->domain->comm_world);
  MPI_Abort(obj->domain->comm_world, 1);
}

/**
 * @brief Continuous coordinate of v in the image of the anchor (plans/16 §3.5):
 * anchor + (img - img_anchor) L + (pos - anchor), formed from the exact integer image
 * difference. Identity (a plain copy) on non-periodic axes, so the non-periodic arithmetic
 * is bit-identical. No extent restriction (the counters carry the image).
 */
static inline void mftlUnwrapToAnchor(const MFTL_Domain* dom, double out[3], const MFTL_Vertex* v, const double anchor[3], const int anchor_img[3])
{
  for (int k = 0; k < 3; ++k)
  {
    out[k] = v->pos[k];
    if (dom->is_periodic[k]) out[k] = anchor[k] + ((double) (v->img[k] - anchor_img[k]) * dom->domain_size[k] + (v->pos[k] - anchor[k]));
  }
}

double MFTLObjectVolume(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_Domain* dom = obj->domain;
  MFTL_ASSERT(dom != MFTL_NULL);

  // Volume is only defined for a closed, consistently-oriented surface.
  MFTL_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  mftlRequireEnclosed(obj, "the volume");

  double anchor[3];
  int anchor_img[3] = {0, 0, 0};
  if (!mftlObjectUnwrapAnchor(obj, anchor, anchor_img)) return 0.0;

  double local = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p2, vv[1], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p3, vv[2], anchor, anchor_img);

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
  mftlRequireCompact(obj, "the surface centre");

  double anchor[3];
  int anchor_img[3] = {0, 0, 0};
  if (!mftlObjectUnwrapAnchor(obj, anchor, anchor_img)) return 0.0;

  double local[4] = {0.0, 0.0, 0.0, 0.0};
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p2, vv[1], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p3, vv[2], anchor, anchor_img);

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
  mftlRequireEnclosed(obj, "the moment of inertia");

  double anchor[3];
  int anchor_img[3] = {0, 0, 0};
  if (!mftlObjectUnwrapAnchor(obj, anchor, anchor_img)) return;

  double Ilocal[3][3] = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->owner_proc != dom->rank) continue;  // owned only: never count ghosts

    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    double p1[3], p2[3], p3[3];
    mftlUnwrapToAnchor(dom, p1, vv[0], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p2, vv[1], anchor, anchor_img);
    mftlUnwrapToAnchor(dom, p3, vv[2], anchor, anchor_img);

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

  // longest minimum-image edge over owned triangles
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

  // There is no object-extent condition: the image counters carry every vertex's image, so
  // bodies of any extent are evaluated exactly (plans/16 §3.5).
  return MFTL_SUCCESS;
}
