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
#include "mftl_images.h"

// Transform predicate: local = owned by this rank and not a ghost copy.
static inline int mftlIsLocalVertex(const MFTL_Vertex* v, int rank)
{
  return !v->is_ghost && v->owner_proc == rank;
}


/* Stage 16B (plans/16 §3.2, §3.5): rotation and scaling on a periodic domain act on continuous
   coordinates RELATIVE TO THE BODY REFERENCE POINT X_ref (the lowest-gid authoritative vertex,
   agreed collectively), about the image of the given centre that is nearest to X_ref:

     Y = X - X_ref (exact integer image difference),  P = minimg(centre - X_ref),
     Y' = P + M (Y - P),  X' = X_ref + Y'  -> (pos, img) with img = img_ref + wrap.

   Gauge invariance: a whole-body lattice re-gauge X -> X + nL moves X_ref by nL, leaves Y and P
   unchanged (P is a minimum image), so X' moves by nL: the same physical result. Every rank uses
   the same X_ref and P. |Y| is the body extent, so no pos + img L is formed with large img.
   A wrapping body is transformed only if M maps every winding lattice vector D = g ⊙ L onto a
   lattice vector (otherwise the result is not a periodic surface): MFTL_ERROR, body unchanged.
   If M changes a winding vector (e.g. a 90-degree rotation of a film), the cached winding state
   is invalidated and recomputed by the next wrap-sensitive collective operation (after the
   caller's ghost refresh, as for every transform). */
static int mftlPeriodicLinear(MFTL_Object* obj, const double M[3][3], const double centre[3], const char* what)
{
  MFTL_Domain* dom = obj->domain;
  MFTLObjectEnsureWinding(obj);  // collective: canonical counters, known winding lattice
  int lattice_changed = 0;
  for (int g = 0; g < obj->n_wind; ++g)
  {
    double D[3], MD[3], nD = 0.0;
    for (int k = 0; k < 3; ++k) D[k] = obj->wind_gen[g][k] * dom->domain_size[k];
    for (int k = 0; k < 3; ++k) nD += D[k] * D[k];
    int ok = 1;
    for (int k = 0; k < 3; ++k)
    {
      MD[k] = M[k][0] * D[0] + M[k][1] * D[1] + M[k][2] * D[2];
      const double tol = 1e-12 * sqrt(nD);
      if (!dom->is_periodic[k])
        ok = ok && fabs(MD[k]) <= tol;
      else
      {
        const double n = round(MD[k] / dom->domain_size[k]);
        ok = ok && fabs(MD[k] - n * dom->domain_size[k]) <= tol;
        if (n != (double) obj->wind_gen[g][k]) lattice_changed = 1;
      }
    }
    if (!ok)
    {
      if (dom->rank == 0)
        fprintf(stderr, "MFTL: %s of body %d rejected: it maps the winding lattice vector (%d,%d,%d) off the periodic lattice\n", what, obj->global_id,
                obj->wind_gen[g][0], obj->wind_gen[g][1], obj->wind_gen[g][2]);
      return MFTL_ERROR;
    }
  }
  double pos_ref[3];
  int img_ref[3];
  if (!MFTLObjectReference(obj, pos_ref, img_ref))  // collective; empty body
  {
    MFTLComputeTriangleProperties(obj);
    return MFTL_SUCCESS;
  }
  double P[3];
  for (int k = 0; k < 3; ++k)
  {
    P[k] = centre[k] - pos_ref[k];
    if (dom->is_periodic[k]) P[k] -= round(P[k] / dom->domain_size[k]) * dom->domain_size[k];
  }
  const int rank = dom->rank;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    double Y[3], Z[3], X[3];
    MFTLVertexRelative(dom, v, pos_ref, img_ref, Y);
    for (int k = 0; k < 3; ++k) Z[k] = Y[k] - P[k];
    for (int k = 0; k < 3; ++k) X[k] = pos_ref[k] + P[k] + M[k][0] * Z[0] + M[k][1] * Z[1] + M[k][2] * Z[2];
    int wrap[3];
    MFTLWrapContinuous(dom, X, v->pos, wrap);
    for (int k = 0; k < 3; ++k) v->img[k] = dom->is_periodic[k] ? img_ref[k] + wrap[k] : 0;
  }
  if (lattice_changed) obj->wrap_valid = 0;  // new winding lattice M.Lambda: recomputed on demand
  MFTLComputeTriangleProperties(obj);
  return MFTL_SUCCESS;
}

int MFTLScaleObject(MFTL_Object* obj, const double s[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(s != MFTL_NULL);
  int rank = obj->domain->rank;

  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    // Stage 16B: about the image of the origin nearest to the body reference point (collective)
    const double M[3][3] = {{s[0], 0.0, 0.0}, {0.0, s[1], 0.0}, {0.0, 0.0, s[2]}};
    const double origin[3] = {0.0, 0.0, 0.0};
    return mftlPeriodicLinear(obj, M, origin, "scaling");
  }
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    v->pos[0] *= s[0];
    v->pos[1] *= s[1];
    v->pos[2] *= s[2];
  }

  MFTLComputeTriangleProperties(obj);
  return MFTL_SUCCESS;
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

int MFTLRotateObject(MFTL_Object* obj, const double R[3][3], const double centre[3])
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

  if (MFTLDomainAnyPeriodic(obj->domain)) return mftlPeriodicLinear(obj, R, centre, "rotation");  // Stage 16B (collective)

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
  return MFTL_SUCCESS;
}

void MFTLWrapObjectIntoDomain(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  int rank = obj->domain->rank;

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlIsLocalVertex(v, rank)) continue;
    // Stage 16B: the wrap is counted into img (continuous coordinate unchanged); any number of
    // periods (a translation may exceed one), single wraps use MFTLApplyPeriodicity's arithmetic
    for (int k = 0; k < 3; ++k)
    {
      if (!obj->domain->is_periodic[k]) continue;
      while (v->pos[k] >= obj->domain->global_max[k])
      {
        v->pos[k] -= obj->domain->domain_size[k];
        v->img[k] += 1;
      }
      while (v->pos[k] < obj->domain->global_min[k])
      {
        v->pos[k] += obj->domain->domain_size[k];
        v->img[k] -= 1;
      }
    }
  }

  MFTLComputeTriangleProperties(obj);
}
