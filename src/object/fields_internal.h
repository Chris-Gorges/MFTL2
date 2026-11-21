/** @file fields_internal.h
 * @brief Private pool-row iteration, optional-field storage and record movement helpers.
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

/** @brief One sub-array of a pool as raw rows: what a permutation or a compaction moves. */
typedef struct MFTL_PoolRows
{
  unsigned char *base;  ///< Row 0.
  size_t row_bytes;  ///< Bytes per entity: 3 half-edges for `tris.he`, `ncomp` doubles for a field.
} MFTL_PoolRows;

/**
 * @brief Iterates the sub-arrays of @p pool whose rows belong to their entity.
 *
 * Every built-in array the pool lays out except the rebuilt caches
 * (`tris.verts`, `tris.neighbors`), `verts.sizing` when present, then every
 * persistent registered field -- from the list src/object/object.c lays the
 * pool out from, so a permutation or compaction that walks it cannot leave a
 * sub-array behind.
 *
 *     MFTL_PoolRows r;
 *     for (int it = 0; MFTLObjectNextMovedRows(obj, MFTL_FIELD_VERTEX, &it, &r);) ...
 *
 * @param      obj    Mesh.
 * @param      pool   #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE.
 * @param[in,out] cursor Start at 0; advanced past the sub-array returned.
 * @param[out] rows   The next sub-array.
 * @return `false` when there is none left.
 */
bool MFTLObjectNextMovedRows(const MFTL_Object *obj, MFTL_FieldEntity pool, int *cursor, MFTL_PoolRows *rows);

/** @brief Widest row MFTLObjectNextMovedRows() returns for @p pool, in bytes: a permutation's scratch per entity.
 * @param obj Mesh.
 * @param pool Entity class.
 * @return Bytes.
 */
static inline size_t MFTLObjectWidestMovedRow(const MFTL_Object *obj, MFTL_FieldEntity pool)
{
  size_t widest = 0;
  MFTL_PoolRows r;
  for (int it = 0; MFTLObjectNextMovedRows(obj, pool, &it, &r);)
    if (r.row_bytes > widest) widest = r.row_bytes;
  return widest;
}

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
