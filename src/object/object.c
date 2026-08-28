/**
 * @file object.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_object
 *
 * @brief Object lifecycle and the ONE allocation authority for both pools.
 *
 * ## WHY THIS FILE IS THE ONLY PLACE THAT MAY TOUCH A POOL POINTER
 *
 * A pool is one allocation. `vertex_layout_compute` decides where each
 * sub-array starts inside it, `*_pool_bind` publishes those addresses into the
 * struct, and `*_pool_reserve` is the only function that ever calls either. The
 * consequence is the property the whole rewrite is organised around:
 *
 *     THE AoS RECORDS AND EVERY SoA ARRAY OF A POOL HAVE THE SAME CAPACITY,
 *     ALWAYS, BECAUSE THEY ARE THE SAME malloc.
 *
 * The arrangement this replaces is three capacity fields (`capacity_*`,
 * `geom_cap_*`, `flat_cap_*`), three reserve functions and a rule written in a
 * comment: "every site that raises capacity_triangles must call
 * MFTLObjectReserveTriangleGeometry, and a site that forgets does not produce a
 * wrong number -- it produces a heap overflow the next time the geometry pass
 * runs". That is an accurate description of any design that depends on everyone
 * remembering something. Here there is nothing to remember: growing the block
 * grows all of it, and a caller cannot reach the individual arrays' sizes
 * because there are none.
 *
 * Two secondary benefits worth naming, since they were not the goal:
 *
 *   - Destroying an object is two free() calls, not fifteen.
 *   - Every sub-array starts on a cache line (MFTLLayoutAdd), so a kernel
 *     writing `tris.area` cannot false-share with one reading `tris.normal`,
 *     and no streaming loop needs a vectoriser peel prologue.
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
  size_t area;  ///< [capacity]   Voronoi areas.
  size_t mean_curv;  ///< [capacity]   mean curvatures.
  size_t gauss_curv;  ///< [capacity]   Gaussian curvatures.
} vertex_layout;

/**
 * @brief Lays out a vertex block for @p cap vertices.
 *
 * Field order is not arbitrary: `rec` and `pos` come first because they are the
 * two arrays every pass touches, so they share the front of the block and the
 * TLB entries that go with it. Everything after them is written once per
 * timestep by a geometry pass and read by whoever wants curvature.
 *
 * @return Total block size in bytes.
 *
 * @param[out] l   Receives one byte offset per field.
 * @param      cap Vertices the block must hold.
 */
static size_t vertex_layout_compute(vertex_layout *l, int cap)
{
  const size_t n = (size_t) cap;
  size_t cursor = 0;
  l->rec = MFTLLayoutAdd(&cursor, n, sizeof(MFTL_Vertex));
  l->pos = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->normal = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->area = MFTLLayoutAdd(&cursor, n, sizeof(double));
  l->mean_curv = MFTLLayoutAdd(&cursor, n, sizeof(double));
  l->gauss_curv = MFTLLayoutAdd(&cursor, n, sizeof(double));
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
  p->area = (double *) (void *) (blk + l->area);
  p->mean_curv = (double *) (void *) (blk + l->mean_curv);
  p->gauss_curv = (double *) (void *) (blk + l->gauss_curv);
}

/**
 * @brief Grows the vertex pool to @p capacity entities. Never shrinks.
 *
 * Contents are preserved: the live prefix of every array is copied into the new
 * block. Discarding the geometry on resize -- on the argument that the caller is
 * about to recompute it -- is true of the callers that exist and false in
 * general, and its failure mode is a silent 0.0 rather than a crash. A memcpy at
 * memory bandwidth, O(log n) times over the mesh's life, is not worth that class
 * of bug.
 *
 * @param p        The pool. Its `block` is replaced and the old one freed.
 * @param capacity Vertices the pool must hold afterwards.
 */
static void vertex_pool_reserve(MFTL_VertexPool *p, int capacity)
{
  if (capacity <= p->capacity) return;

  vertex_layout l;
  const size_t bytes = vertex_layout_compute(&l, capacity);
  unsigned char *blk = (unsigned char *) MFTLAlignedAllocOrDie(bytes, "MFTL vertex pool");

  /* Zeroed rather than left dirty: fields no explicit pass writes (flags in
   * particular, which carries MFTL_FLAG_BOUNDARY) must start at a defined
   * value, and a slot read before it is written yields 0.0 instead of whatever
   * the allocator last had there. */
  memset(blk, 0, bytes);

  const size_t n = (size_t) p->n;
  if (n > 0)
  {
    memcpy(blk + l.rec, p->rec, n * sizeof(MFTL_Vertex));
    memcpy(blk + l.pos, p->pos, 3 * n * sizeof(double));
    memcpy(blk + l.normal, p->normal, 3 * n * sizeof(double));
    memcpy(blk + l.area, p->area, n * sizeof(double));
    memcpy(blk + l.mean_curv, p->mean_curv, n * sizeof(double));
    memcpy(blk + l.gauss_curv, p->gauss_curv, n * sizeof(double));
  }

  free(p->block);
  p->block = blk;
  vertex_pool_bind(p, blk, &l);
  p->capacity = capacity;
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
  size_t anchor;  ///< [3*capacity] derived: position of corner 0.
  size_t edge;  ///< [6*capacity] derived: {e1, e2}, seam-consistent.
} triangle_layout;

/**
 * @brief Lays out a triangle block for @p cap triangles and 3*@p cap half-edges.
 *
 * The half-edge array being here, rather than in a pool of its own, is what
 * removes the third independent capacity a separate half-edge pool would need.
 * It is not a packaging choice: a triangle has exactly three half-edges at 3t..3t+2
 * (mftl_topology.h), so a half-edge capacity that could differ from 3x the
 * triangle capacity would describe a mesh that cannot exist.
 *
 * @return Total block size in bytes.
 *
 * @param[out] l   Receives one byte offset per field.
 * @param      cap Triangles the block must hold. Half-edges get 3x that.
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
  l->anchor = MFTLLayoutAdd(&cursor, 3 * n, sizeof(double));
  l->edge = MFTLLayoutAdd(&cursor, 6 * n, sizeof(double));
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
  p->anchor = (double *) (void *) (blk + l->anchor);
  p->edge = (double *) (void *) (blk + l->edge);
}

/**
 * @brief Grows the triangle pool to @p capacity triangles. Never shrinks.
 *
 * @param p        The pool. Its `block` is replaced and the old one freed.
 * @param capacity Triangles the pool must hold afterwards.
 */
static void triangle_pool_reserve(MFTL_TrianglePool *p, int capacity)
{
  if (capacity <= p->capacity) return;

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
    memcpy(blk + l.anchor, p->anchor, 3 * n * sizeof(double));
    memcpy(blk + l.edge, p->edge, 6 * n * sizeof(double));
  }

  free(p->block);
  p->block = blk;
  triangle_pool_bind(p, blk, &l);
  p->capacity = capacity;
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
  /* A mesh that has never been partitioned cannot have a partition that is still
   * current, so the conservative value is the one that forces the first
   * redistribution to do the work. */
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

  /* And neither does the partition. This is THE choke point for entity creation
   * and destruction, which is what makes the redistribution's per-body skip
   * independent of an unenforceable convention: a caller that splits an edge or
   * collapses one cannot get here without stamping, whatever it forgot to do.
   * See MFTLObjectMarkPartitionDirty() for the whole list. */
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

void MFTLObjectReleasePools(MFTL_Object *obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(obj->verts.n == 0 && obj->tris.n == 0 && "MFTLObjectReleasePools: the object still holds entities");

  free(obj->verts.block);
  free(obj->tris.block);

  /* Re-zeroed through the same path MFTLObjectCreate uses, so a released object
   * is indistinguishable from a freshly created one and the next
   * MFTLObjectReserve() simply allocates. Rebinding the sub-array pointers to
   * NULL matters: MFTLObjectPoolsAreConsistent derives them from `block`, and a
   * dangling pointer here would pass every count-based check. */
  memset(&obj->verts, 0, sizeof obj->verts);
  memset(&obj->tris, 0, sizeof obj->tris);
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
    /* Every pointer must be exactly where the layout for this capacity puts it.
     * That is the strongest available statement of "the AoS and the SoA arrays
     * were sized by the same call": if any of them had been grown separately, it
     * would not be inside this block at this offset. */
    vertex_layout l;
    (void) vertex_layout_compute(&l, vp->capacity);
    const unsigned char *blk = (const unsigned char *) vp->block;
    if (blk == MFTL_NULL) return MFTL_FALSE;
    if ((const unsigned char *) vp->rec != blk + l.rec) return MFTL_FALSE;
    if ((const unsigned char *) vp->pos != blk + l.pos) return MFTL_FALSE;
    if ((const unsigned char *) vp->normal != blk + l.normal) return MFTL_FALSE;
    if ((const unsigned char *) vp->area != blk + l.area) return MFTL_FALSE;
    if ((const unsigned char *) vp->mean_curv != blk + l.mean_curv) return MFTL_FALSE;
    if ((const unsigned char *) vp->gauss_curv != blk + l.gauss_curv) return MFTL_FALSE;
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
    if ((const unsigned char *) tp->anchor != blk + l.anchor) return MFTL_FALSE;
    if ((const unsigned char *) tp->edge != blk + l.edge) return MFTL_FALSE;
  }

  return MFTL_TRUE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public: global statistics
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLObjectUpdateGlobalStats(MFTL_Object *object)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(object->domain != MFTL_NULL && "MFTLObjectUpdateGlobalStats: object has no domain -- add it to an ensemble first");

  /* Ownership is a prefix, so the local counts ARE the answer -- no scan over
   * the entities testing a flag, which is what an `is_ghost` field would need.
   *
   * Blocking rather than non-blocking: two integers, latency-bound, and the
   * caller wants the result immediately. There is nothing to overlap it with. */
  const int local_counts[2] = {object->tris.n_local, object->verts.n_local};
  int global_counts[2] = {0, 0};
  MPI_Allreduce(local_counts, global_counts, 2, MPI_INT, MPI_SUM, object->domain->comm_world);

  object->global_n_triangles = global_counts[0];
  object->global_n_vertices = global_counts[1];
}
