/**
 * @file ghost_sync.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_communication
 *
 * @brief Ghost refreshes of per-entity double arrays with the topology
 *        unchanged: the COLUMN payload of MFTLGhostPull() (ghost_pull.c), and
 *        the public syncs built on it.
 *
 * Positions, normals, the remesher's target lengths and registered fields are
 * all "copy the owner's row of a double array", so they are one payload: a
 * reply row is the doubles of the pulled columns followed by the entity's
 * global id (echoed, or #MFTL_ERROR when the owner could not answer) and 4
 * bytes of padding. A row of one 3-vector is therefore exactly an
 * #MFTL_GhostUpdatePayload, 32 B, as before. Per ghost: 8 B of request and
 * `8 * (components + 1)` B of reply; three exchanges.
 *
 * One quantity per call is the default: bundling recomputable fields with the
 * position would ship bytes the receiver computes more cheaply.
 */

#include "mftl_communication.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/** @brief Bytes after a row's doubles: the echoed id and 4 bytes of padding. */
#define GHOST_TAG_BYTES 8

/** @brief One pulled column, placed in its pool's reply row. */
typedef struct
{
  MFTL_GhostColumnKind kind;  ///< Which array.
  int field;  ///< Field id for #MFTL_GHOST_COLUMN_FIELD.
  int ncomp;  ///< Doubles per entity.
  int at;  ///< First double of the column in the row.
} ghost_column;

/** @brief The column payload's context: the columns of each pool and a scratch. */
typedef struct
{
  ghost_column *col[2];  ///< `[n[p]]` the columns of pool p, in the caller's order.
  int n[2];  ///< Columns per pool.
  int width[2];  ///< Doubles per row of pool p, before the tag.
  double **arr;  ///< `[n_columns]` the arrays of the body being served or applied.
} ghost_columns;

/**
 * @brief The array behind a column on one body.
 * @param obj The body.
 * @param c   The column.
 * @return The `[ncomp * capacity]` array, or #MFTL_NULL if the body lacks it.
 */
static double *column_array(const MFTL_Object *obj, const ghost_column *c)
{
  switch (c->kind)
  {
    case MFTL_GHOST_COLUMN_POSITION: return obj->verts.pos;
    case MFTL_GHOST_COLUMN_NORMAL: return obj->verts.normal;
    case MFTL_GHOST_COLUMN_SIZING: return obj->verts.has_sizing ? obj->verts.sizing : MFTL_NULL;
    case MFTL_GHOST_COLUMN_FIELD: return (obj->field_data != MFTL_NULL) ? obj->field_data[c->field] : MFTL_NULL;
  }
  return MFTL_NULL;
}

/**
 * @brief Resolves every column of pool @p p on @p obj into the scratch.
 * @param cs  The payload.
 * @param obj The body, or #MFTL_NULL.
 * @param p   The pool.
 * @return True if the body has every column.
 */
static bool bind_columns(ghost_columns *cs, const MFTL_Object *obj, int p)
{
  if (obj == MFTL_NULL) return MFTL_FALSE;
  for (int c = 0; c < cs->n[p]; ++c)
  {
    cs->arr[c] = column_array(obj, &cs->col[p][c]);
    if (cs->arr[c] == MFTL_NULL) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/** @brief Owner side (#MFTL_GhostServeFn): the owned rows, tagged with their ids. */
static void serve_columns(const MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, const int *offset, int n, unsigned char *reply, void *ctx)
{
  (void) slot;
  ghost_columns *const cs = (ghost_columns *) ctx;
  const int p = (int) pool;
  const int nc = cs->n[p];
  const int width = cs->width[p];
  const ghost_column *const col = cs->col[p];
  const double *const *const arr = (const double *const *) cs->arr;
  const bool whole = bind_columns(cs, obj, p);
  const int padding = 0;

  for (int j = 0; j < n; ++j)
  {
    unsigned char *const row = reply + offset[j];
    const int i = idx[j];
    int tag = MFTL_ERROR;
    if (i == MFTL_ERROR || !whole)
    {
      memset(row, 0, (size_t) width * sizeof(double));
    }
    else
    {
      for (int c = 0; c < nc; ++c)
      {
        const int m = col[c].ncomp;
        double *const restrict out = (double *) row + col[c].at;
        const double *const restrict in = arr[c] + (size_t) i * m;
        for (int k = 0; k < m; ++k) out[k] = in[k];
      }
      tag = (p == MFTL_FIELD_VERTEX) ? obj->verts.rec[i].global_id : obj->tris.rec[i].global_id;
    }
    memcpy(row + (size_t) width * sizeof(double), &tag, sizeof tag);
    memcpy(row + (size_t) width * sizeof(double) + sizeof tag, &padding, sizeof padding);  // defined bytes on the wire
  }
}

/** @brief Requester side (#MFTL_GhostApplyFn): copies every tagged row; a sentinel keeps the old value. */
static int apply_columns(MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, int n, const unsigned char *rows, void *ctx)
{
  (void) slot;
  ghost_columns *const cs = (ghost_columns *) ctx;
  const int p = (int) pool;
  const int nc = cs->n[p];
  const int width = cs->width[p];
  const size_t row_bytes = (size_t) width * sizeof(double) + GHOST_TAG_BYTES;
  const ghost_column *const col = cs->col[p];
  if (!bind_columns(cs, obj, p)) return n;  // nowhere to put them
  double *const *const arr = cs->arr;

  int skipped = 0;
  for (int j = 0; j < n; ++j)
  {
    const unsigned char *const row = rows + (size_t) j * row_bytes;
    int tag;
    memcpy(&tag, row + (size_t) width * sizeof(double), sizeof tag);
    if (tag == MFTL_ERROR)
    {
      skipped++;  // the owner could not serve it: keep the stale value, report
      continue;
    }
    const int i = idx[j];
    MFTL_ASSERT(tag == ((p == MFTL_FIELD_VERTEX) ? obj->verts.rec[i].global_id : obj->tris.rec[i].global_id) && "ghost reply does not match the request order");
    for (int c = 0; c < nc; ++c)
    {
      const int m = col[c].ncomp;
      const double *const restrict in = (const double *) row + col[c].at;
      double *const restrict out = arr[c] + (size_t) i * m;
      for (int k = 0; k < m; ++k) out[k] = in[k];
    }
  }
  return skipped;
}

int MFTLUpdateGhostColumns(MFTL_Ensemble *ens, const MFTL_GhostColumn *columns, int n_columns, MFTL_Route route)
{
  if (ens == MFTL_NULL || columns == MFTL_NULL || n_columns < 1) return MFTL_ERROR;

  ghost_columns cs;
  ghost_column *const all = (ghost_column *) MFTLMallocOrDie(2 * (size_t) n_columns * sizeof(ghost_column), "ghost columns");
  cs.col[0] = all;
  cs.col[1] = all + n_columns;
  cs.n[0] = cs.n[1] = 0;
  cs.width[0] = cs.width[1] = 0;
  cs.arr = (double **) MFTLMallocOrDie((size_t) n_columns * sizeof(double *), "ghost column arrays");

  /* Validated from arguments every rank shares, so a refusal is unanimous. */
  bool valid = MFTL_TRUE, positions = MFTL_FALSE;
  for (int j = 0; j < n_columns && valid; ++j)
  {
    const MFTL_GhostColumn *const c = &columns[j];
    int pool = MFTL_FIELD_VERTEX, ncomp = 0;
    switch (c->kind)
    {
      case MFTL_GHOST_COLUMN_POSITION:
      case MFTL_GHOST_COLUMN_NORMAL: ncomp = 3; break;
      case MFTL_GHOST_COLUMN_SIZING: ncomp = 1; break;
      case MFTL_GHOST_COLUMN_FIELD:
        if (c->field < 0 || c->field >= ens->fields.n)
        {
          valid = MFTL_FALSE;
          break;
        }
        pool = (int) ens->fields.entries[c->field].entity;
        ncomp = ens->fields.entries[c->field].ncomp;
        break;
      default: valid = MFTL_FALSE; break;
    }
    if (!valid || ncomp < 1) break;
    for (int k = 0; k < j; ++k)
      if (columns[k].kind == c->kind && (c->kind != MFTL_GHOST_COLUMN_FIELD || columns[k].field == c->field)) valid = MFTL_FALSE;
    if (cs.width[pool] > (INT_MAX - GHOST_TAG_BYTES) / (int) sizeof(double) - ncomp) valid = MFTL_FALSE;  // the row must fit an int
    if (!valid) break;

    ghost_column *const g = &cs.col[pool][cs.n[pool]++];
    g->kind = c->kind;
    g->field = c->field;
    g->ncomp = ncomp;
    g->at = cs.width[pool];
    cs.width[pool] += ncomp;
    if (c->kind == MFTL_GHOST_COLUMN_POSITION) positions = MFTL_TRUE;
  }

  int rc = MFTL_ERROR;
  if (valid)
  {
    /* New ghost positions: geometry computed before this is stale on the
     * ghosts, and the surface changed. */
    if (positions)
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        ens->local_objects[o]->derived &= ~MFTL_DERIVED_GEOMETRY;
        MFTLObjectMarkChanged(ens->local_objects[o]);
      }

    MFTL_GhostPull pull;
    for (int p = 0; p < 2; ++p)
    {
      pull.row_bytes[p] = (cs.width[p] > 0) ? cs.width[p] * (int) sizeof(double) + GHOST_TAG_BYTES : 0;
      pull.keys[p] = MFTL_NULL;
    }
    pull.route = route;
    pull.serve = serve_columns;
    pull.apply = apply_columns;
    pull.ctx = &cs;
    rc = MFTLGhostPull(ens, &pull);
  }

  free(cs.arr);
  free(all);
  return rc;
}

int MFTLUpdateGhostPositions(MFTL_Ensemble *ens)
{
  const MFTL_GhostColumn c = {MFTL_GHOST_COLUMN_POSITION, 0};
  return MFTLUpdateGhostColumns(ens, &c, 1, MFTL_ROUTE_GRAPH);
}

int MFTLUpdateGhostVertexNormals(MFTL_Ensemble *ens)
{
  const MFTL_GhostColumn c = {MFTL_GHOST_COLUMN_NORMAL, 0};
  return MFTLUpdateGhostColumns(ens, &c, 1, MFTL_ROUTE_GRAPH);
}

int MFTLUpdateGhostFields(MFTL_Ensemble *ens, const int *ids, int nfields, bool with_positions)
{
  if (!ens) return MFTL_ERROR;
  if (nfields == 0) return with_positions ? MFTLUpdateGhostPositions(ens) : MFTL_SUCCESS;

  /* Agreed before any rank takes a different path: one rank's bad list must
   * not leave the others inside the pull. */
  int valid = nfields > 0 && ids != MFTL_NULL;
  for (int j = 0; valid && j < nfields; ++j)
  {
    if (ids[j] < 0 || ids[j] >= ens->fields.n) valid = 0;
    for (int k = 0; k < j; ++k)
      if (ids[k] == ids[j]) valid = 0;
  }
  int agreed = 0;
  MPI_Allreduce(&valid, &agreed, 1, MPI_INT, MPI_MIN, ens->domain.comm_world);
  if (!agreed) return MFTL_ERROR;

  MFTL_GhostColumn *const cols = (MFTL_GhostColumn *) MFTLMallocOrDie(((size_t) nfields + 1) * sizeof(MFTL_GhostColumn), "field ghost columns");
  int n = 0;
  if (with_positions) cols[n++] = (MFTL_GhostColumn){MFTL_GHOST_COLUMN_POSITION, 0};
  for (int j = 0; j < nfields; ++j) cols[n++] = (MFTL_GhostColumn){MFTL_GHOST_COLUMN_FIELD, ids[j]};
  int ok = MFTLUpdateGhostColumns(ens, cols, n, MFTL_ROUTE_GRAPH) == MFTL_SUCCESS;
  free(cols);

  /* The same verdict everywhere: an unreachable ghost on one rank is a failure
   * of the call, not of that rank. */
  MPI_Allreduce(&ok, &agreed, 1, MPI_INT, MPI_MIN, ens->domain.comm_world);
  return agreed ? MFTL_SUCCESS : MFTL_ERROR;
}
