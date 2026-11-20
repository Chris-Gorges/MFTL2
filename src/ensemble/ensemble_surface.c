/**
 * @file ensemble_surface.c
 * @author Christian Gorges
 * @date 2026-11-15
 * @ingroup mftl_ensemble
 *
 * @brief The collective forms of the signed curvature and the surface-tension
 *        force: the local kernels of src/geometry/ on every object, plus
 *        exactly the exchanges their inputs and outputs need.
 *
 * | call | before the kernels | after |
 * |---|---|---|
 * | curvature, COTAN | -- | ghost rows of the result |
 * | curvature, FIT | vertex normals, exact on ghosts | ghost rows of the result |
 * | force, uniform sigma, triangle normals on the contour | -- | -- |
 * | force, sigma field | its ghost rows | -- |
 * | force, vertex normals on the contour | vertex normals, exact on ghosts | -- |
 *
 * Every branch is taken on arguments that are identical on every rank (the
 * caller's parameters and the replicated field registry), so all ranks enter
 * the same collectives. Validation reads only those, so a refusal is the same
 * verdict on every rank and returns before any collective.
 */

#include "mftl_ensemble.h"

#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_types.h"

/** @brief Rebuilds stale derived connectivity of every local object (the kernels and the normal pass need it). */
static void ensure_topology(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (!MFTLObjectHasTopology(ens->local_objects[o])) MFTLObjectBuildDerived(ens->local_objects[o]);
}

/** @brief The descriptor of field @p id, or #MFTL_NULL if no such field is registered. */
static const MFTL_FieldDescriptor *field_of(const MFTL_Ensemble *ens, int id)
{
  if (id < 0 || id >= ens->fields.n) return MFTL_NULL;
  return &ens->fields.entries[id];
}

int MFTLEnsembleComputeCurvature(MFTL_Ensemble *ens, MFTL_CurvatureMethod method, int field_id, int *n_fallback)
{
  if (n_fallback != MFTL_NULL) *n_fallback = 0;
  if (ens == MFTL_NULL) return MFTL_ERROR;
  if (method != MFTL_CURVATURE_COTAN && method != MFTL_CURVATURE_FIT) return MFTL_ERROR;
  const MFTL_FieldDescriptor *const fd = field_of(ens, field_id);
  if (fd == MFTL_NULL || fd->entity != MFTL_FIELD_VERTEX) return MFTL_ERROR;
  const int ncomp = fd->ncomp;
  if (ncomp != 1 && ncomp != 2 && ncomp != 4 && ncomp != MFTL_CURVATURE_COMPONENTS) return MFTL_ERROR;

  /* Guard on the GLOBAL count: everything below is collective. */
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  ensure_topology(ens);
  int rc = MFTL_SUCCESS;
  if (method == MFTL_CURVATURE_FIT && MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_FIT) != MFTL_SUCCESS) rc = MFTL_ERROR;

  int fallbacks = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (obj->verts.n == 0) continue;
    int nf = 0;
    if (MFTLComputeVertexCurvature(obj, method, ncomp, MFTLObjectField(obj, field_id), &nf) != MFTL_SUCCESS) rc = MFTL_ERROR;
    fallbacks += nf;
  }

  if (MFTLUpdateGhostFields(ens, &field_id, 1, MFTL_FALSE) != MFTL_SUCCESS) rc = MFTL_ERROR;
  if (n_fallback != MFTL_NULL) *n_fallback = fallbacks;
  return rc;
}

/** @brief Whether @p p and @p force_field are acceptable; reads only replicated data. */
static bool tension_arguments_valid(const MFTL_Ensemble *ens, const MFTL_SurfaceTensionParameters *p, int force_field)
{
  if (p->form != MFTL_TENSION_FRENET_VERTEX && p->form != MFTL_TENSION_FRENET_ELEMENT) return MFTL_FALSE;
  if (p->contour_normal != MFTL_CONTOUR_NORMAL_TRIANGLES && p->contour_normal != MFTL_CONTOUR_NORMAL_VERTICES) return MFTL_FALSE;
  if (p->rim != MFTL_RIM_FORCE_FREE && p->rim != MFTL_RIM_FORCE_WALL && p->rim != MFTL_RIM_FORCE_EXCLUDE) return MFTL_FALSE;

  const MFTL_FieldDescriptor *const ff = field_of(ens, force_field);
  const MFTL_FieldEntity want = (p->form == MFTL_TENSION_FRENET_VERTEX) ? MFTL_FIELD_VERTEX : MFTL_FIELD_TRIANGLE;
  if (ff == MFTL_NULL || ff->entity != want || ff->ncomp != 3) return MFTL_FALSE;

  if (p->sigma_field >= 0)
  {
    const MFTL_FieldDescriptor *const sf = field_of(ens, p->sigma_field);
    if (sf == MFTL_NULL || sf->ncomp != 1 || p->sigma_field == force_field) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

int MFTLEnsembleComputeSurfaceTension(MFTL_Ensemble *ens, const MFTL_SurfaceTensionParameters *p, int force_field)
{
  if (ens == MFTL_NULL || p == MFTL_NULL) return MFTL_ERROR;
  if (!tension_arguments_valid(ens, p, force_field)) return MFTL_ERROR;
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  ensure_topology(ens);
  int rc = MFTL_SUCCESS;
  if (p->sigma_field >= 0 && MFTLUpdateGhostFields(ens, &p->sigma_field, 1, MFTL_FALSE) != MFTL_SUCCESS) rc = MFTL_ERROR;
  if (p->contour_normal == MFTL_CONTOUR_NORMAL_VERTICES && MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_FIT) != MFTL_SUCCESS) rc = MFTL_ERROR;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    const int rows = (p->form == MFTL_TENSION_FRENET_VERTEX) ? obj->verts.n : obj->tris.n;
    if (rows == 0) continue;
    if (MFTLComputeSurfaceTension(obj, p, MFTLObjectField(obj, force_field)) != MFTL_SUCCESS) rc = MFTL_ERROR;
  }
  return rc;
}

/** Slots of the per-body reduction. */
enum
{
  STAT_FX = 0,
  STAT_FY,
  STAT_FZ,
  STAT_ENERGY,
  STAT_AREA,
  STAT_N
};

int MFTLEnsembleSurfaceTensionStats(MFTL_Ensemble *ens, const MFTL_SurfaceTensionParameters *p, int force_field, MFTL_SurfaceTensionStats *out)
{
  if (ens == MFTL_NULL || p == MFTL_NULL) return MFTL_ERROR;
  if (!tension_arguments_valid(ens, p, force_field)) return MFTL_ERROR;
  if (ens->n_bodies_global == 0) return MFTL_SUCCESS;

  const int b = ens->n_local_objects;
  double *const rec = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(b, 1) * STAT_N, sizeof(double), "surface tension stats");

  for (int o = 0; o < b; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    double *const r = &rec[(size_t) STAT_N * o];
    const double *const F = MFTLObjectField(obj, force_field);
    const int n_own = (p->form == MFTL_TENSION_FRENET_VERTEX) ? obj->verts.n_local : obj->tris.n_local;
    for (int i = 0; i < n_own && F != MFTL_NULL; ++i)
    {
      r[STAT_FX] += F[3 * i + 0];
      r[STAT_FY] += F[3 * i + 1];
      r[STAT_FZ] += F[3 * i + 2];
    }

    const double *const sig = (p->sigma_field >= 0) ? MFTLObjectField(obj, p->sigma_field) : MFTL_NULL;
    const bool per_vertex = p->sigma_field >= 0 && ens->fields.entries[p->sigma_field].entity == MFTL_FIELD_VERTEX;
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      const double a = MFTLGetTriangleArea(obj, t);
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      double s = p->sigma;
      if (sig != MFTL_NULL) s = per_vertex ? (sig[v[0]] + sig[v[1]] + sig[v[2]]) / 3.0 : sig[t];
      r[STAT_ENERGY] += s * a;
      r[STAT_AREA] += a;
    }
  }

  const MFTL_BodyReduceOp ops[STAT_N] = {MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};
  const int rc = MFTLEnsembleReduceBodies(ens, rec, STAT_N, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  if (out != MFTL_NULL)
    for (int o = 0; o < b; ++o)
    {
      const double *const r = &rec[(size_t) STAT_N * o];
      out[o].net_force[0] = r[STAT_FX];
      out[o].net_force[1] = r[STAT_FY];
      out[o].net_force[2] = r[STAT_FZ];
      out[o].energy = r[STAT_ENERGY];
      out[o].area = r[STAT_AREA];
    }
  free(rec);
  if (b > 0 && out == MFTL_NULL) return MFTL_ERROR;
  return rc;
}
