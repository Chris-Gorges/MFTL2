/**
 * @file object.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief Object lifecycle and the single allocation authority for both pools.
 *
 * Each pool's sub-arrays are ONE list (MFTL_VERTEX_POOL_COLUMNS,
 * MFTL_TRIANGLE_POOL_COLUMNS). `pool_layout` decides where each starts inside the
 * pool's one block, `pool_bind` publishes those addresses, and `pool_resize` is
 * the only function that allocates, copies or frees a block (`pool_reserve` is
 * its grow-only wrapper, MFTLObjectReleasePools() is a resize to zero, and
 * MFTLObjectShrinkToFit() is the only way down). Hence the AoS records and every
 * SoA array of a pool always share one capacity: they are the same `malloc`.
 * Code elsewhere that moves rows walks the same list through
 * MFTLObjectNextMovedRows().
 */

#include "mftl_object.h"
#include "fields_internal.h"
#include <limits.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

/** @brief Size, copy and bind optional arrays appended to a pool.
 * @param obj Mesh and registry.
 * @param entity Vertex or triangle entity class.
 * @param cap Pool capacity.
 * @param cursor Byte offset following the built-in arrays.
 * @param block New allocation, or NULL to size only.
 * @param n Live rows to copy.
 * @return Total pool bytes including optional arrays.
 */
static size_t field_layout(MFTL_Object *obj, int entity, int cap, size_t cursor, unsigned char *block, int n)
{
  if (!obj->fields) return cursor;
  for (int f = 0; f < obj->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
    if ((int) d->entity != entity) continue;
    const size_t width = (size_t) d->ncomp * sizeof(double);
    if (cursor > SIZE_MAX - MFTL_CACHE_LINE || (size_t) cap > (SIZE_MAX - cursor - MFTL_CACHE_LINE) / width)
    {
      fprintf(stderr, "MFTL field pool size overflow\n");
      abort();
    }
    const size_t off = MFTLLayoutAdd(&cursor, (size_t) cap, width);
    if (block)
    {
      double *dst = (double *) (void *) (block + off);
      if (n > 0 && obj->field_data[f]) memcpy(dst, obj->field_data[f], (size_t) n * width);
      obj->field_data[f] = dst;
    }
  }
  return cursor;
}

/** @brief Clear optional pointers when releasing a pool.
 * @param obj Mesh.
 * @param entity Released entity class.
 */
static void field_release(MFTL_Object *obj, int entity)
{
  if (obj->fields)
    for (int f = 0; f < obj->fields->n; ++f)
      if ((int) obj->fields->entries[f].entity == entity) obj->field_data[f] = NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The sub-arrays of a pool: ONE list each
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Every built-in sub-array is one line below: member, bytes per entity, how its
 * rows move, and whether it is laid out. The layout, the binding of the
 * pointers, the resize copy, the consistency check, the byte count and
 * MFTLObjectNextMovedRows() -- the iterator compaction and both permutations
 * walk -- are all expanded from that line, so a sub-array that one of them
 * forgets cannot exist. Registered fields follow the built-ins
 * (field_layout()).
 *
 *   MOVED    the row belongs to its entity: a permutation, a compaction or the
 *            ownership swap carries it. A triangle's row of `he` is its three
 *            half-edges, so they move as one unit and both the implicit
 *            indexing and the implicit half-edge ids survive by construction.
 *            `normal`, `area` and `center` are computed, but moved so a reorder
 *            never discards values the caller just computed.
 *   REBUILT  a cache re-derived after any of those (MFTLObjectBuildDerived()):
 *            copied by a resize, never moved.
 *
 * Layout order: `rec` and `pos`, which every pass touches, first; the optional
 * `sizing` last among the built-ins, so a pool without it is laid out exactly
 * as before it existed.
 */
#define MFTL_VERTEX_POOL_COLUMNS(X)                 \
  X(rec, sizeof(MFTL_Vertex), COLUMN_MOVED, true)   \
  X(pos, 3 * sizeof(double), COLUMN_MOVED, true)    \
  X(normal, 3 * sizeof(double), COLUMN_MOVED, true) \
  X(sizing, sizeof(double), COLUMN_MOVED, p->has_sizing)

#define MFTL_TRIANGLE_POOL_COLUMNS(X)                  \
  X(rec, sizeof(MFTL_Triangle), COLUMN_MOVED, true)    \
  X(he, 3 * sizeof(MFTL_HalfEdge), COLUMN_MOVED, true) \
  X(area, sizeof(double), COLUMN_MOVED, true)          \
  X(normal, 3 * sizeof(double), COLUMN_MOVED, true)    \
  X(center, 3 * sizeof(double), COLUMN_MOVED, true)    \
  X(verts, 3 * sizeof(int), COLUMN_REBUILT, true)      \
  X(neighbors, 3 * sizeof(int), COLUMN_REBUILT, true)

/** @brief How a built-in sub-array's rows move (see the lists above). */
typedef enum
{
  COLUMN_MOVED,  ///< Carried by every permutation, compaction and ownership swap.
  COLUMN_REBUILT  ///< Re-derived instead; only a resize copies it.
} column_motion;

/** @brief One built-in sub-array of a pool, as raw rows. */
typedef struct
{
  unsigned char *base;  ///< Current address; NULL when not laid out, or laid out for the first time.
  size_t row;  ///< Bytes per entity.
  column_motion motion;  ///< How its rows move.
  bool present;  ///< Part of the layout.
} pool_column;

/** @brief Upper bound on the built-in sub-arrays of either pool. */
#define MAX_POOL_COLUMNS 8

#define COLUMN_COUNT(m, row, motion, present) +1
MFTL_STATIC_ASSERT(0 MFTL_VERTEX_POOL_COLUMNS(COLUMN_COUNT) <= MAX_POOL_COLUMNS, "MAX_POOL_COLUMNS is too small for the vertex pool");
MFTL_STATIC_ASSERT(0 MFTL_TRIANGLE_POOL_COLUMNS(COLUMN_COUNT) <= MAX_POOL_COLUMNS, "MAX_POOL_COLUMNS is too small for the triangle pool");
#undef COLUMN_COUNT

/**
 * @brief The built-in sub-arrays of one pool, in layout order.
 *
 * @param      obj  The mesh.
 * @param      pool #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE.
 * @param[out] c    Receives one entry per sub-array.
 * @return The number of entries written.
 */
static int pool_columns(const MFTL_Object *obj, MFTL_FieldEntity pool, pool_column c[MAX_POOL_COLUMNS])
{
  int k = 0;
#define COLUMN(m, row_bytes, how, laid_out) c[k++] = (pool_column){(unsigned char *) (void *) p->m, (row_bytes), (how), (laid_out)};
  if (pool == MFTL_FIELD_VERTEX)
  {
    const MFTL_VertexPool *const p = &obj->verts;
    MFTL_VERTEX_POOL_COLUMNS(COLUMN)
  }
  else
  {
    const MFTL_TrianglePool *const p = &obj->tris;
    MFTL_TRIANGLE_POOL_COLUMNS(COLUMN)
  }
#undef COLUMN
  return k;
}

/**
 * @brief Lays the built-in sub-arrays out in one block of @p cap entities.
 *
 * @param      c   The sub-arrays, from pool_columns().
 * @param      nc  Their number.
 * @param      cap Entities the block must hold.
 * @param[out] off Byte offset of each present sub-array (0 for an absent one).
 * @return Bytes of the built-in part; registered fields follow it.
 */
static size_t pool_layout(const pool_column *c, int nc, int cap, size_t *off)
{
  size_t cursor = 0;
  for (int k = 0; k < nc; ++k) off[k] = c[k].present ? MFTLLayoutAdd(&cursor, (size_t) cap, c[k].row) : 0;
  return cursor;
}

/**
 * @brief Points every built-in sub-array of @p pool into @p blk (NULL when not
 *        laid out). The one place a pool pointer is written.
 *
 * @param obj  The mesh.
 * @param pool #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE.
 * @param blk  Base of the allocation.
 * @param off  Offsets from pool_layout().
 */
static void pool_bind(MFTL_Object *obj, MFTL_FieldEntity pool, unsigned char *blk, const size_t *off)
{
  int k = 0;
#define BIND(m, row_bytes, how, laid_out)                  \
  p->m = (laid_out) ? (void *) (blk + off[k]) : MFTL_NULL; \
  ++k;
  if (pool == MFTL_FIELD_VERTEX)
  {
    MFTL_VertexPool *const p = &obj->verts;
    MFTL_VERTEX_POOL_COLUMNS(BIND)
  }
  else
  {
    MFTL_TrianglePool *const p = &obj->tris;
    MFTL_TRIANGLE_POOL_COLUMNS(BIND)
  }
#undef BIND
}

/**
 * @brief Resizes one pool to exactly @p capacity entities, up or down.
 *
 * Lays out a new block, zeroes it (so flags and never-written slots start
 * defined), copies the live prefix of every sub-array and frees the old block --
 * one code path for growth, shrink and both pools. Capacity 0 releases the
 * block and zeroes the struct (NULL pointers, not a dangling allocation of
 * nothing).
 *
 * @param obj      The mesh owning the pool.
 * @param pool     #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE (which also holds every half-edge).
 * @param capacity Entities it must hold afterwards; at least the pool's `n`
 *                 (asserted: this moves storage and never discards entities).
 * @param force    Relayout even at unchanged capacity (a registry or `sizing` attached).
 */
static void pool_resize(MFTL_Object *obj, MFTL_FieldEntity pool, int capacity, bool force)
{
  const bool vertex = (pool == MFTL_FIELD_VERTEX);
  const int n = vertex ? obj->verts.n : obj->tris.n;
  int *const cap = vertex ? &obj->verts.capacity : &obj->tris.capacity;
  void **const block = vertex ? &obj->verts.block : &obj->tris.block;
  MFTL_ASSERT(capacity >= n && "pool_resize: would discard live entities");

  if (capacity == *cap && !force) return;

  /* Zero capacity is a release: free, and zero the struct so every pointer is
   * NULL (the state MFTLObjectPoolsAreConsistent() expects of an empty pool). */
  if (capacity == 0)
  {
    field_release(obj, pool);
    free(*block);
    if (vertex)
      memset(&obj->verts, 0, sizeof obj->verts);
    else
      memset(&obj->tris, 0, sizeof obj->tris);
    return;
  }

  pool_column c[MAX_POOL_COLUMNS];
  size_t off[MAX_POOL_COLUMNS];
  const int nc = pool_columns(obj, pool, c);
  const size_t built_in = pool_layout(c, nc, capacity, off);
  const size_t bytes = field_layout(obj, pool, capacity, built_in, NULL, 0);
  unsigned char *blk = (unsigned char *) MFTLAlignedAllocOrDie(bytes, vertex ? "MFTL vertex pool" : "MFTL triangle pool");

  /* Zeroed so never-written fields (flags in particular) start defined. */
  memset(blk, 0, bytes);

  /* A present sub-array whose pointer is still NULL is being laid out for the
   * first time (MFTLObjectEnableSizing()): nothing to carry over, and the zeroed
   * block is what it starts as. */
  if (n > 0)
    for (int k = 0; k < nc; ++k)
      if (c[k].present && c[k].base != MFTL_NULL) memcpy(blk + off[k], c[k].base, (size_t) n * c[k].row);

  field_layout(obj, pool, capacity, built_in, blk, n);
  free(*block);
  *block = blk;
  pool_bind(obj, pool, blk, off);
  *cap = capacity;
}

/**
 * @brief Grows one pool to @p capacity entities; a no-op if it already holds
 *        that many (the ratchet MFTLObjectReserve() publishes).
 *
 * @param obj      The mesh owning the pool.
 * @param pool     #MFTL_FIELD_VERTEX or #MFTL_FIELD_TRIANGLE.
 * @param capacity Entities the pool must hold afterwards.
 */
static void pool_reserve(MFTL_Object *obj, MFTL_FieldEntity pool, int capacity)
{
  if (capacity <= (pool == MFTL_FIELD_VERTEX ? obj->verts.capacity : obj->tris.capacity)) return;
  pool_resize(obj, pool, capacity, false);
}

bool MFTLObjectNextMovedRows(const MFTL_Object *obj, MFTL_FieldEntity pool, int *cursor, MFTL_PoolRows *rows)
{
  pool_column c[MAX_POOL_COLUMNS];
  const int nc = pool_columns(obj, pool, c);
  const int nf = obj->fields ? obj->fields->n : 0;

  /* The cursor counts the built-ins, then the registered fields. */
  while (*cursor < nc + nf)
  {
    const int k = (*cursor)++;
    if (k < nc)
    {
      if (c[k].motion != COLUMN_MOVED || !c[k].present) continue;
      rows->base = c[k].base;
      rows->row_bytes = c[k].row;
      return MFTL_TRUE;
    }
    const MFTL_FieldDescriptor *const d = &obj->fields->entries[k - nc];
    if (d->entity != pool || d->mode == MFTL_FIELD_RECOMPUTED) continue;
    rows->base = (unsigned char *) (void *) obj->field_data[k - nc];
    rows->row_bytes = (size_t) d->ncomp * sizeof(double);
    return MFTL_TRUE;
  }
  return MFTL_FALSE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Source of #MFTL_Object::generation stamps; process-wide so a stamp is never reused. */
static uint64_t g_generation_counter = 0;

void MFTLObjectMarkChanged(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  obj->generation = ++g_generation_counter;
}

MFTL_Object *MFTLObjectCreate(int global_id)
{
  /* calloc: every count, capacity and pointer starts at 0/NULL, which is exactly
   * the empty-pool state the reserve functions expect. */
  MFTL_Object *object = (MFTL_Object *) calloc(1, sizeof(MFTL_Object));
  if (!object)
  {
    fprintf(stderr, "MFTL Error: Failed to allocate memory for MFTL_Object struct.\n");
    return MFTL_NULL;
  }

  object->global_id = global_id;
  object->domain = MFTL_NULL;  // set by MFTLEnsembleAddObject
  object->derived = 0u;
  /* Never partitioned, so no current partition: the first redistribution must
   * rebuild it. */
  object->partition_dirty = MFTL_TRUE;
  object->reference_volume = NAN;  // nobody has said what it should enclose
  MFTLObjectMarkChanged(object);

  return object;
}

void MFTLObjectDestroy(MFTL_Object **p_object)
{
  if (!p_object || !(*p_object)) return;
  MFTL_Object *object = *p_object;

  free(object->verts.block);
  free(object->tris.block);
  free(object->field_data);
  free(object);

  *p_object = MFTL_NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: size
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLObjectReserve(MFTL_Object *obj, int n_vertices, int n_triangles)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(n_vertices >= 0 && n_triangles >= 0);

  pool_reserve(obj, MFTL_FIELD_VERTEX, n_vertices);
  pool_reserve(obj, MFTL_FIELD_TRIANGLE, n_triangles);
}

void MFTLObjectReserveAdditional(MFTL_Object *obj, int extra_vertices, int extra_triangles)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(extra_vertices >= 0 && extra_triangles >= 0);

  pool_reserve(obj, MFTL_FIELD_VERTEX, MFTLGrowCapacity(obj->verts.capacity, obj->verts.n + extra_vertices));
  pool_reserve(obj, MFTL_FIELD_TRIANGLE, MFTLGrowCapacity(obj->tris.capacity, obj->tris.n + extra_triangles));
}

void MFTLObjectSetCounts(MFTL_Object *obj, int n_vertices, int n_local_vertices, int n_triangles, int n_local_triangles)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(n_vertices >= 0 && n_triangles >= 0);
  MFTL_ASSERT(n_local_vertices >= 0 && n_local_vertices <= n_vertices && "MFTLObjectSetCounts: owned vertices must be a prefix of the pool");
  MFTL_ASSERT(n_local_triangles >= 0 && n_local_triangles <= n_triangles && "MFTLObjectSetCounts: owned triangles must be a prefix of the pool");

  /* Reserve BEFORE assigning: this ordering is what makes "a count larger than
   * its capacity" unreachable, no matter what the caller passes. */
  MFTLObjectReserve(obj, n_vertices, n_triangles);

  if (obj->fields)
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
      const int old = d->entity == MFTL_FIELD_VERTEX ? obj->verts.n : obj->tris.n;
      const int next = d->entity == MFTL_FIELD_VERTEX ? n_vertices : n_triangles;
      if (next > old) memset(obj->field_data[f] + (size_t) old * d->ncomp, 0, (size_t) (next - old) * d->ncomp * sizeof(double));
    }

  obj->verts.n = n_vertices;
  obj->verts.n_local = n_local_vertices;
  obj->tris.n = n_triangles;
  obj->tris.n_local = n_local_triangles;

  /* The derived arrays describe a different entity count than they did a moment
   * ago, so whatever they hold is no longer a description of this mesh. */
  obj->derived = 0u;

  /* Nor does the partition. Every entity creation or destruction passes through
   * here, so the redistribution's skip of unchanged bodies cannot miss one. */
  obj->partition_dirty = MFTL_TRUE;
  MFTLObjectMarkChanged(obj);
}

void MFTLObjectClear(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  obj->verts.n = 0;
  obj->verts.n_local = 0;
  obj->tris.n = 0;
  obj->tris.n_local = 0;

  obj->next_vertex_gid = 0;
  obj->next_triangle_gid = 0;
  obj->next_halfedge_gid = 0;

  obj->global_n_vertices = 0;
  obj->global_n_triangles = 0;

  obj->derived = 0u;
  obj->partition_dirty = MFTL_TRUE;
  MFTLObjectMarkChanged(obj);
}

size_t MFTLObjectPoolBytes(const MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  /* Re-derived from the layout functions, so it cannot drift from them. */
  size_t bytes = 0;

  for (int e = MFTL_FIELD_VERTEX; e <= MFTL_FIELD_TRIANGLE; ++e)
  {
    const MFTL_FieldEntity pool = (MFTL_FieldEntity) e;
    const int cap = (pool == MFTL_FIELD_VERTEX) ? obj->verts.capacity : obj->tris.capacity;
    if (cap == 0) continue;
    pool_column c[MAX_POOL_COLUMNS];
    size_t off[MAX_POOL_COLUMNS];
    const int nc = pool_columns(obj, pool, c);
    bytes += field_layout((MFTL_Object *) obj, pool, cap, pool_layout(c, nc, cap, off), NULL, 0);
  }

  return bytes;
}

bool MFTLObjectShrinkToFit(MFTL_Object *obj, double slack)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(slack >= 1.0 && "MFTLObjectShrinkToFit: a slack below 1 would ask for a capacity under the count");

  bool moved = MFTL_FALSE;

  /* Target: back on the growth ladder. At n == 0 MFTLGrowCapacity() returns 0,
   * so a full release is the same arithmetic reaching *_pool_resize(p, 0). */
  const int v_target = MFTLGrowCapacity(0, obj->verts.n);
  const int t_target = MFTLGrowCapacity(0, obj->tris.n);

  /* GATE (policy): wasteful enough to reallocate? GUARD (idempotence): would
   * the target actually be smaller? For n <= 128 the target is a flat 128, so
   * without the guard a capacity-128 pool with n = 4 would "shrink" to itself
   * on every call. */
  if ((double) obj->verts.capacity > slack * (double) obj->verts.n && v_target < obj->verts.capacity)
  {
    pool_resize(obj, MFTL_FIELD_VERTEX, v_target, false);
    moved = MFTL_TRUE;
  }

  if ((double) obj->tris.capacity > slack * (double) obj->tris.n && t_target < obj->tris.capacity)
  {
    pool_resize(obj, MFTL_FIELD_TRIANGLE, t_target, false);
    moved = MFTL_TRUE;
  }

  /* Deliberately not stamped partition-dirty: only storage moved. */
  return moved;
}

void MFTLObjectReleasePools(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(obj->verts.n == 0 && obj->tris.n == 0 && "MFTLObjectReleasePools: the object still holds entities");

  /* A release is a resize to capacity 0, which also re-NULLs every sub-array
   * pointer, so the object is indistinguishable from a fresh one. */
  pool_resize(obj, MFTL_FIELD_VERTEX, 0, false);
  pool_resize(obj, MFTL_FIELD_TRIANGLE, 0, false);
}

bool MFTLObjectPoolsAreConsistent(const MFTL_Object *obj)
{
  if (obj == MFTL_NULL) return MFTL_FALSE;

  const MFTL_VertexPool *vp = &obj->verts;
  const MFTL_TrianglePool *tp = &obj->tris;

  /* Counts inside capacity, owned inside active. */
  if (vp->n < 0 || vp->n > vp->capacity) return MFTL_FALSE;
  if (tp->n < 0 || tp->n > tp->capacity) return MFTL_FALSE;
  if (vp->n_local < 0 || vp->n_local > vp->n) return MFTL_FALSE;
  if (tp->n_local < 0 || tp->n_local > tp->n) return MFTL_FALSE;

  /* An empty pool owns nothing and points nowhere; otherwise every pointer must
   * sit exactly where the layout for this capacity puts it -- the strongest form
   * of "all arrays were sized by the same call". `cursor` is where each pool's
   * registered fields begin. */
  size_t cursor[2] = {0, 0};
  for (int e = MFTL_FIELD_VERTEX; e <= MFTL_FIELD_TRIANGLE; ++e)
  {
    const MFTL_FieldEntity pool = (MFTL_FieldEntity) e;
    const int cap = (pool == MFTL_FIELD_VERTEX) ? vp->capacity : tp->capacity;
    const unsigned char *const blk = (const unsigned char *) ((pool == MFTL_FIELD_VERTEX) ? vp->block : tp->block);
    pool_column c[MAX_POOL_COLUMNS];
    size_t off[MAX_POOL_COLUMNS];
    const int nc = pool_columns(obj, pool, c);
    cursor[e] = pool_layout(c, nc, cap, off);
    if ((cap == 0) != (blk == MFTL_NULL)) return MFTL_FALSE;
    for (int k = 0; k < nc; ++k)
    {
      const unsigned char *const want = (cap > 0 && c[k].present) ? blk + off[k] : MFTL_NULL;
      if (c[k].base != want) return MFTL_FALSE;
    }
  }

  if (obj->fields && obj->fields->n)
  {
    if (!obj->field_data) return MFTL_FALSE;
    for (int f = 0; f < obj->fields->n; ++f)
    {
      const MFTL_FieldDescriptor *d = &obj->fields->entries[f];
      const int cap = d->entity == MFTL_FIELD_VERTEX ? vp->capacity : tp->capacity;
      const unsigned char *block = d->entity == MFTL_FIELD_VERTEX ? vp->block : tp->block;
      const size_t offset = MFTLLayoutAdd(&cursor[d->entity], (size_t) cap, (size_t) d->ncomp * sizeof(double));
      if (cap == 0 ? obj->field_data[f] != NULL : (const unsigned char *) obj->field_data[f] != block + offset) return MFTL_FALSE;
    }
  }
  return MFTL_TRUE;
}

void MFTLObjectEnableSizing(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  if (obj->verts.has_sizing) return;

  /* Through the one resize path, so the new array is cut from the same block as
   * every other vertex array and grows with it from here on. An empty pool only
   * records the flag; its first reserve lays the array out. */
  obj->verts.has_sizing = MFTL_TRUE;
  if (obj->verts.capacity > 0) pool_resize(obj, MFTL_FIELD_VERTEX, obj->verts.capacity, true);
}

void MFTLObjectAttachFields(MFTL_Object *obj, MFTL_FieldRegistry *fields)
{
  if (obj->fields == fields) return;
  MFTL_ASSERT(!obj->fields && "cannot move an object between field registries");
  fields->locked = true;
  if (!fields->n) return;
  obj->fields = fields;
  obj->field_data = MFTLCallocOrDie((size_t) fields->n, sizeof(double *), "field pointers");
  if (obj->verts.capacity) pool_resize(obj, MFTL_FIELD_VERTEX, obj->verts.capacity, true);
  if (obj->tris.capacity) pool_resize(obj, MFTL_FIELD_TRIANGLE, obj->tris.capacity, true);
}

double *MFTLObjectField(MFTL_Object *obj, int field_id)
{
  if (!obj || !obj->fields || field_id < 0 || field_id >= obj->fields->n) return NULL;
  return obj->field_data[field_id];
}

int MFTLObjectFieldComponents(const MFTL_Object *obj, int field_id)
{
  if (!obj || !obj->fields || field_id < 0 || field_id >= obj->fields->n) return MFTL_ERROR;
  return obj->fields->entries[field_id].ncomp;
}
