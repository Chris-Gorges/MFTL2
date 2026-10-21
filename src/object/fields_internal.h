/** @file fields_internal.h
 * @brief Private optional-field storage and record movement helpers.
 */
#ifndef MFTL_FIELDS_INTERNAL_H
#define MFTL_FIELDS_INTERNAL_H

#include "mftl_object.h"
#include <string.h>

/** @brief Bind once, relaying generated pools if necessary.
 * @param obj Mesh.
 * @param fields Ensemble-owned registry.
 */
void MFTLObjectAttachFields(MFTL_Object *obj, MFTL_FieldRegistry *fields);

/** @brief Persistent record width.
 * @param obj Mesh.
 * @param entity Entity class.
 * @return Double count.
 */
static inline int MFTLFieldsStride(const MFTL_Object *obj, int entity) { return obj->fields ? obj->fields->stride[entity] : 0; }

/** @brief Pack/unpack one persistent record without an auxiliary index space.
 * @param obj Mesh.
 * @param entity Entity class.
 * @param slot Pool index.
 * @param wire Packed doubles.
 * @param unpack True writes the mesh, false writes the wire.
 */
static inline void MFTLFieldsRecord(const MFTL_Object *obj, int entity, int slot, double *wire, bool unpack)
{
  if (!obj->fields) return;
  for (int f = 0; f < obj->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
    if ((int) d->entity != entity || d->mode == MFTL_FIELD_RECOMPUTED) continue;
    double *row = obj->field_data[f] + (size_t) slot * d->ncomp;
    if (unpack)
      memcpy(row, wire, (size_t) d->ncomp * sizeof(double));
    else
      memcpy(wire, row, (size_t) d->ncomp * sizeof(double));
    wire += d->ncomp;
  }
}

/** @brief Maximum persistent row width, for existing permutation scratch.
 * @param obj Mesh.
 * @param minimum Built-in row width in bytes.
 * @return Required element width in bytes.
 */
static inline size_t MFTLFieldsWidth(const MFTL_Object *obj, size_t minimum)
{
  if (obj->fields)
    for (int f = 0; f < obj->fields->n; ++f)
      if (obj->fields->entries[f].mode != MFTL_FIELD_RECOMPUTED)
      {
        size_t bytes = (size_t) obj->fields->entries[f].ncomp * sizeof(double);
        if (bytes > minimum) minimum = bytes;
      }
  return minimum;
}
#endif
