/**
 * @file object.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief Object lifecycle and the single allocation authority for both pools.
 *
 * `*_layout_compute` decides where each sub-array starts inside a pool's one
 * block, `*_pool_bind` publishes those addresses, and `*_pool_resize` is the
 * only function that allocates, copies or frees a block (`*_pool_reserve` is
 * its grow-only wrapper, MFTLObjectReleasePools() is a resize to zero, and
 * MFTLObjectShrinkToFit() is the only way down). Hence the AoS records and every
 * SoA array of a pool always share one capacity: they are the same `malloc`.
 */

#include "mftl_object.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Vertex pool
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Byte offsets of every vertex sub-array inside one block. */
typedef struct
{
  size_t rec;  ///< [capacity]   MFTL_Vertex records.
  size_t pos;  ///< [3*capacity] AUTHORITATIVE positions, xyz interleaved.
  size_t normal;  ///< [3*capacity] vertex normals, xyz interleaved.
} vertex_layout;

/**
 * @brief Lays out a vertex block for @p cap vertices: `rec`, `pos`, `normal`.
 *
 * `rec` and `pos`, which every pass touches, come first.
 *
 * @param[out] l   Receives one byte offset per field.
 * @param      cap Vertices the block must hold.
 * @return Total block size in bytes.
 */
static size_t vertex_layout_compute(vertex_layout *l, int cap)
{
  const size_t n = (size_t) cap;
  size_t cursor = 0;
  l->rec = MFTLLayoutAdd(&cursor, n, sizeof(MFTL_Vertex));
  l->pos = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->normal = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  return cursor;
}

/**
 * @brief Points every field of @p p into @p blk according to @p l.
 *
 * @param[out] p   The pool whose pointers are set.
 * @param      blk Base of the allocation.
 * @param      l   Offsets from vertex_layout_compute().
 */
static void vertex_pool_bind(MFTL_VertexPool *p, unsigned char *blk, const vertex_layout *l)
{
  p->rec = (MFTL_Vertex *) (void *) (blk + l->rec);
  p->pos = (double *) (void *) (blk + l->pos);
  p->normal = (double *) (void *) (blk + l->normal);
}

/**
 * @brief Resizes the vertex pool to exactly @p capacity vertices, up or down.
 *
 * Lays out a new block, zeroes it (so flags and never-written slots start
 * defined), copies the live prefix of every array and frees the old block --
 * one code path for growth and shrink. Capacity 0 releases the block and zeroes
 * the struct (NULL pointers, not a dangling allocation of nothing).
 *
 * @param p        The pool.
 * @param capacity Vertices it must hold afterwards; at least `p->n` (asserted:
 *                 this moves storage and never discards entities).
 */
static void vertex_pool_resize(MFTL_VertexPool *p, int capacity)
{
  MFTL_ASSERT(capacity >= p->n && "vertex_pool_resize: would discard live entities");

  if (capacity == p->capacity) return;

  /* Zero capacity is a release: free, and zero the struct so every pointer is
   * NULL (the state MFTLObjectPoolsAreConsistent() expects of an empty pool). */
  if (capacity == 0)
  {
    free(p->block);
    memset(p, 0, sizeof *p);
    return;
  }

  vertex_layout l;
  const size_t bytes = vertex_layout_compute(&l, capacity);
  unsigned char *blk = (unsigned char *) MFTLAlignedAllocOrDie(bytes, "MFTL vertex pool");

  /* Zeroed so never-written fields (flags in particular) start defined. */
  memset(blk, 0, bytes);

  const size_t n = (size_t) p->n;
  if (n > 0)
  {
    memcpy(blk + l.rec, p->rec, n * sizeof(MFTL_Vertex));
    memcpy(blk + l.pos, p->pos, 3 * n * sizeof(double));
    memcpy(blk + l.normal, p->normal, 3 * n * sizeof(double));
  }

  free(p->block);
  p->block = blk;
  vertex_pool_bind(p, blk, &l);
  p->capacity = capacity;
}

/**
 * @brief Grows the vertex pool to @p capacity vertices; a no-op if it already
 *        holds that many (the ratchet MFTLObjectReserve() publishes).
 *
 * @param p        The pool.
 * @param capacity Vertices the pool must hold afterwards.
 */
static void vertex_pool_reserve(MFTL_VertexPool *p, int capacity)
{
  if (capacity <= p->capacity) return;
  vertex_pool_resize(p, capacity);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Triangle pool (which also holds every half-edge)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Byte offsets of every triangle/half-edge sub-array inside one block. */
typedef struct
{
  size_t rec;  ///< [capacity]   MFTL_Triangle records.
  size_t he;  ///< [3*capacity] MFTL_HalfEdge records. Three per triangle, always.
  size_t area;  ///< [capacity]   triangle areas.
  size_t normal;  ///< [3*capacity] unit normals, xyz interleaved.
  size_t center;  ///< [3*capacity] centroids, wrapped into the domain.
  size_t verts;  ///< [3*capacity] derived: vertex index per corner.
  size_t neighbors;  ///< [3*capacity] derived: triangle across each half-edge.
} triangle_layout;

/**
 * @brief Lays out a triangle block for @p cap triangles and 3 @p cap half-edges:
 *        `rec`, `he`, `area`, `normal`, `center`, `verts`, `neighbors`.
 *
 * The half-edges live in this block because a triangle has exactly three, at
 * 3t..3t+2; a separate half-edge capacity would describe an impossible mesh.
 *
 * @param[out] l   Receives one byte offset per field.
 * @param      cap Triangles the block must hold.
 * @return Total block size in bytes.
 */
static size_t triangle_layout_compute(triangle_layout *l, int cap)
{
  const size_t n = (size_t) cap;
  size_t cursor = 0;
  l->rec = MFTLLayoutAdd(&cursor, n, sizeof(MFTL_Triangle));
  l->he = MFTLLayoutAdd(&cursor, 3 * n, sizeof(MFTL_HalfEdge));
  l->area = MFTLLayoutAdd(&cursor, n, sizeof(double));
  l->normal = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->center = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->verts = MFTLLayoutAdd(&cursor, 3 * n, sizeof(int));
  l->neighbors = MFTLLayoutAdd(&cursor, 3 * n, sizeof(int));
  return cursor;
}

/**
 * @brief Points every field of @p p into @p blk according to @p l.
 *
 * @param[out] p   The pool whose pointers are set.
 * @param      blk Base of the allocation.
 * @param      l   Offsets from triangle_layout_compute().
 */
static void triangle_pool_bind(MFTL_TrianglePool *p, unsigned char *blk, const triangle_layout *l)
{
  p->rec = (MFTL_Triangle *) (void *) (blk + l->rec);
  p->he = (MFTL_HalfEdge *) (void *) (blk + l->he);
  p->area = (double *) (void *) (blk + l->area);
  p->normal = (double *) (void *) (blk + l->normal);
  p->center = (double *) (void *) (blk + l->center);
  p->verts = (int *) (void *) (blk + l->verts);
  p->neighbors = (int *) (void *) (blk + l->neighbors);
}

/**
 * @brief Resizes the triangle pool (and its half-edges) to exactly @p capacity
 *        triangles, up or down. Same contract as vertex_pool_resize().
 *
 * @param p        The pool.
 * @param capacity Triangles it must hold afterwards; at least `p->n`.
 */
static void triangle_pool_resize(MFTL_TrianglePool *p, int capacity)
{
  MFTL_ASSERT(capacity >= p->n && "triangle_pool_resize: would discard live entities");

  if (capacity == p->capacity) return;

  if (capacity == 0)
  {
    free(p->block);
    memset(p, 0, sizeof *p);
    return;
  }

  triangle_layout l;
  const size_t bytes = triangle_layout_compute(&l, capacity);
  unsigned char *blk = (unsigned char *) MFTLAlignedAllocOrDie(bytes, "MFTL triangle pool");
  memset(blk, 0, bytes);

  const size_t n = (size_t) p->n;
  if (n > 0)
  {
    memcpy(blk + l.rec, p->rec, n * sizeof(MFTL_Triangle));
    memcpy(blk + l.he, p->he, 3 * n * sizeof(MFTL_HalfEdge));
    memcpy(blk + l.area, p->area, n * sizeof(double));
    memcpy(blk + l.normal, p->normal, 3 * n * sizeof(double));
    memcpy(blk + l.center, p->center, 3 * n * sizeof(double));
    memcpy(blk + l.verts, p->verts, 3 * n * sizeof(int));
    memcpy(blk + l.neighbors, p->neighbors, 3 * n * sizeof(int));
  }

  free(p->block);
  p->block = blk;
  triangle_pool_bind(p, blk, &l);
  p->capacity = capacity;
}

/**
 * @brief Grows the triangle pool to @p capacity triangles; never shrinks.
 *
 * @param p        The pool.
 * @param capacity Triangles the pool must hold afterwards.
 */
static void triangle_pool_reserve(MFTL_TrianglePool *p, int capacity)
{
  if (capacity <= p->capacity) return;
  triangle_pool_resize(p, capacity);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

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

  return object;
}

void MFTLObjectDestroy(MFTL_Object **p_object)
{
  if (!p_object || !(*p_object)) return;
  MFTL_Object *object = *p_object;

  free(object->verts.block);
  free(object->tris.block);
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

  vertex_pool_reserve(&obj->verts, n_vertices);
  triangle_pool_reserve(&obj->tris, n_triangles);
}

void MFTLObjectReserveAdditional(MFTL_Object *obj, int extra_vertices, int extra_triangles)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(extra_vertices >= 0 && extra_triangles >= 0);

  vertex_pool_reserve(&obj->verts, MFTLGrowCapacity(obj->verts.capacity, obj->verts.n + extra_vertices));
  triangle_pool_reserve(&obj->tris, MFTLGrowCapacity(obj->tris.capacity, obj->tris.n + extra_triangles));
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
}

size_t MFTLObjectPoolBytes(const MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  /* Re-derived from the layout functions, so it cannot drift from them. */
  size_t bytes = 0;

  if (obj->verts.capacity > 0)
  {
    vertex_layout vl;
    bytes += vertex_layout_compute(&vl, obj->verts.capacity);
  }
  if (obj->tris.capacity > 0)
  {
    triangle_layout tl;
    bytes += triangle_layout_compute(&tl, obj->tris.capacity);
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
    vertex_pool_resize(&obj->verts, v_target);
    moved = MFTL_TRUE;
  }

  if ((double) obj->tris.capacity > slack * (double) obj->tris.n && t_target < obj->tris.capacity)
  {
    triangle_pool_resize(&obj->tris, t_target);
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
  vertex_pool_resize(&obj->verts, 0);
  triangle_pool_resize(&obj->tris, 0);
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

  /* An empty pool owns nothing and points nowhere. */
  if (vp->capacity == 0)
  {
    if (vp->block != MFTL_NULL || vp->rec != MFTL_NULL || vp->pos != MFTL_NULL) return MFTL_FALSE;
  }
  else
  {
    /* Every pointer must sit exactly where the layout for this capacity puts
     * it -- the strongest form of "all arrays were sized by the same call". */
    vertex_layout l;
    (void) vertex_layout_compute(&l, vp->capacity);
    const unsigned char *blk = (const unsigned char *) vp->block;
    if (blk == MFTL_NULL) return MFTL_FALSE;
    if ((const unsigned char *) vp->rec != blk + l.rec) return MFTL_FALSE;
    if ((const unsigned char *) vp->pos != blk + l.pos) return MFTL_FALSE;
    if ((const unsigned char *) vp->normal != blk + l.normal) return MFTL_FALSE;
  }

  if (tp->capacity == 0)
  {
    if (tp->block != MFTL_NULL || tp->rec != MFTL_NULL || tp->he != MFTL_NULL) return MFTL_FALSE;
  }
  else
  {
    triangle_layout l;
    (void) triangle_layout_compute(&l, tp->capacity);
    const unsigned char *blk = (const unsigned char *) tp->block;
    if (blk == MFTL_NULL) return MFTL_FALSE;
    if ((const unsigned char *) tp->rec != blk + l.rec) return MFTL_FALSE;
    if ((const unsigned char *) tp->he != blk + l.he) return MFTL_FALSE;
    if ((const unsigned char *) tp->area != blk + l.area) return MFTL_FALSE;
    if ((const unsigned char *) tp->normal != blk + l.normal) return MFTL_FALSE;
    if ((const unsigned char *) tp->center != blk + l.center) return MFTL_FALSE;
    if ((const unsigned char *) tp->verts != blk + l.verts) return MFTL_FALSE;
    if ((const unsigned char *) tp->neighbors != blk + l.neighbors) return MFTL_FALSE;
  }

  return MFTL_TRUE;
}
