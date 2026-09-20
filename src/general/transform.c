/*
 * In-place rigid-body transforms of an object: uniform/per-axis scale,
 * translation, and rotation about a centre — the MFTL2 equivalents of
 * MFTL1's MFTLScaleSurface / MFTLMoveBody / MFTLRotateBody.
 *
 * Contract (per plans/03-rigid-body-transforms.md):
 *   - Only LOCAL (owned, non-ghost) vertices are transformed; ghost copies
 *     are refreshed afterwards by the caller via MFTLUpdateGhostPositions
 *     (ensemble-level, needs the owning ranks' moved positions).
 *   - Every transform recomputes MFTLComputeTriangleProperties so centres,
 *     normals and areas never go stale for downstream ownership/integrals.
 *   - Storage stays UNWRAPPED: no periodic wrapping happens implicitly.
 *     MFTLWrapObjectIntoDomain is an explicit, optional call to bound
 *     coordinate growth for long-circulating bodies (Stage 13 semantics).
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"

// Transform predicate: local = owned by this rank and not a ghost copy.
static inline int mftlIsLocalVertex(const MFTL_Vertex* v, int rank)
{
  return !v->is_ghost && v->owner_proc == rank;
}

void MFTLScaleObject(MFTL_Object* obj, const double s[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(s != MFTL_NULL);
  int rank = obj->domain->rank;

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    v->pos[0] *= s[0];
    v->pos[1] *= s[1];
    v->pos[2] *= s[2];
  }

  MFTLComputeTriangleProperties(obj);
}

void MFTLTranslateObject(MFTL_Object* obj, const double d[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(d != MFTL_NULL);
  int rank = obj->domain->rank;

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    v->pos[0] += d[0];
    v->pos[1] += d[1];
    v->pos[2] += d[2];
  }

  MFTLComputeTriangleProperties(obj);
}

void MFTLRotateObject(MFTL_Object* obj, const double R[3][3], const double centre[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(R != MFTL_NULL);
  MFTL_ASSERT(centre != MFTL_NULL);
  int rank = obj->domain->rank;

  // Guard against a non-orthonormal rotation matrix (shape distortion).
  // Caller supplies R; in debug builds require R^T R = I and det R = 1.
  double RtR[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) RtR[i][j] = R[0][i] * R[0][j] + R[1][i] * R[1][j] + R[2][i] * R[2][j];
  double ortho_err = 0.0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) ortho_err = fmax(ortho_err, fabs(RtR[i][j] - (i == j ? 1.0 : 0.0)));
  double det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
               R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  MFTL_ASSERT(ortho_err < 1e-12 && "MFTLRotateObject: R is not orthonormal");
  MFTL_ASSERT(fabs(det - 1.0) < 1e-12 && "MFTLRotateObject: det(R) != 1");

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;

    double x = v->pos[0] - centre[0];
    double y = v->pos[1] - centre[1];
    double z = v->pos[2] - centre[2];

    v->pos[0] = centre[0] + R[0][0] * x + R[0][1] * y + R[0][2] * z;
    v->pos[1] = centre[1] + R[1][0] * x + R[1][1] * y + R[1][2] * z;
    v->pos[2] = centre[2] + R[2][0] * x + R[2][1] * y + R[2][2] * z;
  }

  MFTLComputeTriangleProperties(obj);
}

void MFTLWrapObjectIntoDomain(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  int rank = obj->domain->rank;

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    MFTLApplyPeriodicity(obj->domain, v->pos);
  }

  MFTLComputeTriangleProperties(obj);
}
