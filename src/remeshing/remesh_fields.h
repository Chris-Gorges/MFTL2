/** @file remesh_fields.h
 * @brief Private field transfer kernels used by the four local operators.
 */
#ifndef MFTL_REMESH_FIELDS_H
#define MFTL_REMESH_FIELDS_H
#include "remesh_internal.h"
#include "../object/fields_internal.h"

/** @cond INTERNAL */
/* Negative corner ids designate point; otherwise positions come from the mesh. */
double MFTLRemeshFieldArea(const MFTL_Object *obj, const int corners[3], const double point[3]);
double MFTLRemeshFieldEdgeParameter(const MFTL_Object *obj, int a, int b, const double point[3]);
/* True when an operator owes MFTLRemeshFieldVertex(): `verts.sizing` or a persistent vertex field. */
static inline bool MFTLRemeshCarriesVertexValues(const MFTL_Object *obj) { return obj->verts.has_sizing || MFTLFieldsStride(obj, MFTL_FIELD_VERTEX) != 0; }
/* Every value an operator carries to the vertex it made or moved (the split's new
 * vertex, a collapse's survivor): `verts.sizing` -- CARRIED, not recomputed, since
 * dest lies on the chord -- and every persistent vertex field, each the values of
 * a and b interpolated at fraction along a->b. */
void MFTLRemeshFieldVertex(MFTL_Object *obj, int a, int b, int dest, double fraction);
/* Prepare all persistent triangle rows BEFORE any geometry/topology mutation.
 * Negative corners designate point. parent maps new rows to old parents for a
 * split; otherwise NULL. keep_survivors requires matching old/new slots, with
 * zero new area for removed rows. NULL rejects the edit without field writes. */
double *MFTLRemeshFieldPrepare(const MFTL_Object *obj, const int *old_ids, const double *old_area, int nold, const int (*corners)[3],
                             const double point[3], const double *new_area, int nnew, const int *parent, bool keep_survivors);
/* Reacquire pool pointers, commit staged rows and free values. */
void MFTLRemeshFieldCommit(MFTL_Object *obj, const int *new_ids, const double *new_area, int nnew, double *values);
/* Called before the smoother applies candidate positions. False rejects sweep. */
bool MFTLRemeshFieldSmooth(MFTL_Object *obj, const double *new_pos);
/** @endcond */
#endif
