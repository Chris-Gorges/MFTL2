/*
 * @Author: Christian Gorges
 *
 * @brief Periodic advection of many spheres with the per-vertex and
 *        per-triangle work on the GPU -- the second example of examples/gpu,
 *        one driver behind sp_backend.h: CUDA, OpenACC and OpenMP offload.
 *
 * The physics is that of parallel_advection_redistribute_list.c with
 * --quality --normals: N spheres at random centres in the periodic box
 * [-4,4] x [-2,2] x [-2,2], rigidly translated by v = (1, 0.37, 0.61) and
 * repartitioned every step, and after every step the triangle geometry, the
 * per-body quality statistics (MFTLEnsembleUpdate()) and the area-weighted
 * vertex normals (MFTLEnsembleUpdateVertexNormals()). No remeshing: a body's
 * pools change only when the partition does. Where LeVeque is one big body and
 * a remesh, this is many small bodies and almost nothing but geometry.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT RUNS WHERE, ONE STEP
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The device holds every body of the rank in fixed slots (sp_backend.h), and
 * its positions are THE positions: the host copy of a body nobody else holds
 * is stale until the redistribution needs it.
 *
 *   device   sp_advect                   every owned vertex
 *   D -> H   owned positions of the bodies split across ranks
 *   host     MFTLUpdateGhostPositions    MPI
 *   H -> D   their ghost positions
 *   device   sp_classify                 the owner of every entity: MFTLGetOwnerProc() on
 *                                        the vertex and the wrapped centroid
 *   D -> H   the owners of the bodies that changed, and the positions of the
 *            unsplit bodies the redistribution will send
 *   host     MFTLRedistributeWithOwners  MPI and pointer-heavy; routes by the device's owners
 *   H -> D   the bodies that arrived or were rebuilt
 *   device   sp_geometry                 area, normal, centroid of every triangle +
 *                                        9 statistics per body
 *   host     MFTLEnsembleReduceBodies    MPI: the per-body reduction of MFTLEnsembleUpdate()
 *   device   sp_normals                  vertex normals
 *   D -> H, host, H -> D                 the ghost normal exchange of the split bodies
 *
 * On one rank only the device lines run, and nothing crosses PCIe but 9
 * doubles per body. The owner map is what keeps the centroids on the device:
 * MFTLRedistribute() would route by `tris.center`, i.e. a download of every
 * triangle every step; the device sends one int per entity, and only for the
 * bodies in which an owner changed.
 *
 * Slots: one per body, sized to the whole body (no remeshing: a rank's share
 * never exceeds it), 1.5 x the bodies a rank holds after the setup plus 64, or
 * -slots. The device memory is allocated once (sp_setup) and never again; a
 * body that leaves frees its slot for the next one to arrive.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW IT CHECKS ITSELF
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   -verify        every step: area and entity counts conserved; with a device
 *                  also every device result against the library on the SAME
 *                  positions -- the quality statistics, the triangle geometry
 *                  and the vertex normals, ghosts included. The device values
 *                  are kept, so the run is not perturbed.
 *   -save / -compare  the final state -- area, counts, coordinate sums,
 *                  normals, quality range -- against a -cpu run, to round-off.
 *                  The partition does not enter it: any rank count may be
 *                  compared with any other.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   mpirun -n P spheres_gpu_<backend> [options]
 *
 *     -spheres N     bodies                              (default 1000)
 *     -refine N      sphere subdivision level            (default 3: 1280 triangles)
 *     -radius r      sphere radius                       (default 0.1)
 *     -dt step       timestep                            (default 0.1)
 *     -steps N       timesteps                           (default 100)
 *     -slots N       device slots per rank               (default 1.5 x the bodies held + 64)
 *     -cpu           the library on the host, no device: the reference
 *     -verify        check every step (see above)
 *     -save FILE     write the final state
 *     -compare FILE  require the final state of FILE (a -save run) to round-off
 *     -quiet         summary only
 *
 * Exit status 77 when the backend finds no device (CTest's skip code).
 */

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "sp_backend.h"

enum
{
  T_ADVECT,
  T_OWNERS,
  T_GEOM,
  T_NORMALS,
  T_D2H,
  T_H2D,
  T_GHOST,
  T_REDIST,
  T_N
};
static const char *const timer_name[T_N] = {"advect", "owners", "geometry", "normals", "D->H", "H->D", "ghost sync", "redistribute"};

/** @brief A row of MFTLEnsembleUpdate(): the SP_N_STATS statistics, the triangle and the half-edge count. */
#define SP_ROW 11
static const MFTL_BodyReduceOp row_ops[SP_ROW] = {MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MAX,
                                                  MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM,
                                                  MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};

static void *xrealloc(void *p, size_t bytes)
{
  void *q = realloc(p, bytes > 0 ? bytes : 1);
  if (!q)
  {
    fprintf(stderr, "spheres_gpu: out of host memory (%zu bytes)\n", bytes);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return q;
}

static double *g_scratch = NULL;
static size_t g_scratch_n = 0;

static double *scratch(size_t n)
{
  if (n > g_scratch_n)
  {
    g_scratch_n = n + n / 2;
    g_scratch = (double *) xrealloc(g_scratch, g_scratch_n * sizeof(double));
  }
  return g_scratch;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The slot table: which body sits where on the device
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int S, nvs, nts;  ///< slots; vertices and triangles per slot = per body
  int *slot_of;  ///< [n_bodies_global] body id -> slot, -1 when not here
  int *body;  ///< [S] slot -> body id, -1 when free
  int *free_list, n_free;
  long *seen, epoch;  ///< departures: a slot whose body was not seen in this epoch is freed
  uint64_t *stamp;  ///< [S] the body's generation after the ghost sync: a rebuild renews it
  unsigned char *split;  ///< [S] another rank holds part of the body (this step)
  unsigned char *needed;  ///< [S] its host positions were made current for the redistribution
  int *counts;  ///< [SP_N_COUNTS S] mirror of the device's
  int counts_dirty;
  int *vown, *town;  ///< [S nvs], [S nts] the owner map: mirror of the device's, in pool order
  int *changed;  ///< [S]
  double *stats;  ///< [SP_N_STATS S]
  sp_segment *seg;  ///< [S] one transfer's segments ...
  MFTL_Object **seg_obj;  ///< ... and their bodies
  MFTL_Object **list;  ///< [S] scratch body list
} slot_table;

static slot_table tab;

static void table_create(int n_ids, int S, int nvs, int nts)
{
  memset(&tab, 0, sizeof tab);
  tab.S = S;
  tab.nvs = nvs;
  tab.nts = nts;
  tab.slot_of = (int *) xrealloc(NULL, (size_t) n_ids * sizeof(int));
  for (int i = 0; i < n_ids; ++i) tab.slot_of[i] = -1;
  tab.body = (int *) xrealloc(NULL, (size_t) S * sizeof(int));
  tab.free_list = (int *) xrealloc(NULL, (size_t) S * sizeof(int));
  for (int s = 0; s < S; ++s)
  {
    tab.body[s] = -1;
    tab.free_list[s] = S - 1 - s;  // popped from the back: slots fill from 0
  }
  tab.n_free = S;
  tab.seen = (long *) calloc((size_t) S, sizeof(long));
  tab.stamp = (uint64_t *) calloc((size_t) S, sizeof(uint64_t));
  tab.split = (unsigned char *) calloc((size_t) S, 1);
  tab.needed = (unsigned char *) calloc((size_t) S, 1);
  tab.counts = (int *) calloc((size_t) SP_N_COUNTS * S, sizeof(int));
  tab.vown = (int *) xrealloc(NULL, (size_t) S * nvs * sizeof(int));
  tab.town = (int *) xrealloc(NULL, (size_t) S * nts * sizeof(int));
  tab.changed = (int *) calloc((size_t) S, sizeof(int));
  tab.stats = (double *) calloc((size_t) SP_N_STATS * S, sizeof(double));
  tab.seg = (sp_segment *) xrealloc(NULL, (size_t) S * sizeof(sp_segment));
  tab.seg_obj = (MFTL_Object **) xrealloc(NULL, (size_t) S * sizeof(MFTL_Object *));
  tab.list = (MFTL_Object **) xrealloc(NULL, (size_t) S * sizeof(MFTL_Object *));
  if (!tab.seen || !tab.stamp || !tab.split || !tab.needed || !tab.counts || !tab.changed || !tab.stats) MPI_Abort(MPI_COMM_WORLD, 1);
}

static void table_destroy(void)
{
  free(tab.slot_of), free(tab.body), free(tab.free_list), free(tab.seen), free(tab.stamp), free(tab.split), free(tab.needed), free(tab.counts),
      free(tab.vown), free(tab.town), free(tab.changed), free(tab.stats), free(tab.seg), free(tab.seg_obj), free(tab.list);
  memset(&tab, 0, sizeof tab);
}

static int slot_of(const MFTL_Object *obj) { return tab.slot_of[obj->global_id]; }

static int slot_acquire(int id)
{
  if (tab.n_free == 0)
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "spheres_gpu: rank %d holds more than %d bodies -- rerun with a larger -slots\n", rank, tab.S);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const int s = tab.free_list[--tab.n_free];
  tab.slot_of[id] = s;
  tab.body[s] = id;
  return s;
}

static void slot_release(int s)
{
  tab.slot_of[tab.body[s]] = -1;
  tab.body[s] = -1;
  memset(&tab.counts[SP_N_COUNTS * s], 0, SP_N_COUNTS * sizeof(int));
  tab.counts_dirty = 1;
  tab.free_list[tab.n_free++] = s;
}

/** @brief Another rank holds part of the body: its owned vertices are someone's ghosts, or the reverse. */
static int is_split(const MFTL_Object *obj) { return obj->verts.n_local < tab.nvs || obj->tris.n_local < tab.nts; }

/* ═══════════════════════════════════════════════════════════════════════════
 * Transfers: the library's pools <-> the device slots
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum
{
  RANGE_OWNED,
  RANGE_GHOST,
  RANGE_ALL
} entity_range;

static int field_width(sp_field f) { return (f == SP_AREA || f == SP_VOWNER || f == SP_TOWNER) ? 1 : 3; }
static int field_on_tris(sp_field f) { return f == SP_AREA || f == SP_NORMAL || f == SP_CENTER || f == SP_TRI_VERTS || f == SP_TOWNER; }
static int field_is_int(sp_field f) { return f == SP_TRI_VERTS || f == SP_VOWNER || f == SP_TOWNER; }

/** @brief The host side of a double field: the library's own arrays. */
static double *host_doubles(MFTL_Object *obj, sp_field f)
{
  switch (f)
  {
    case SP_POS: return obj->verts.pos;
    case SP_VNORMAL: return obj->verts.normal;
    case SP_AREA: return obj->tris.area;
    case SP_NORMAL: return obj->tris.normal;
    case SP_CENTER: return obj->tris.center;
    default: abort();
  }
}

/** @brief The host side of an int field: the corners, or the owner map cache of the body's slot. */
static int *host_ints(MFTL_Object *obj, sp_field f)
{
  switch (f)
  {
    case SP_TRI_VERTS: return obj->tris.verts;
    case SP_VOWNER: return tab.vown + (size_t) slot_of(obj) * tab.nvs;
    case SP_TOWNER: return tab.town + (size_t) slot_of(obj) * tab.nts;
    default: abort();
  }
}

/** @brief One segment per body of @p list with entities in @p r; returns their number, total in @p n_ent. */
static int make_segments(MFTL_Object *const *list, int n, int on_tris, entity_range r, long *n_ent)
{
  int k = 0;
  long off = 0;
  for (int i = 0; i < n; ++i)
  {
    MFTL_Object *obj = list[i];
    const int all = on_tris ? obj->tris.n : obj->verts.n, owned = on_tris ? obj->tris.n_local : obj->verts.n_local;
    const int first = (r == RANGE_GHOST) ? owned : 0;
    const int count = (r == RANGE_OWNED) ? owned : (r == RANGE_GHOST ? all - owned : all);
    if (count <= 0) continue;
    tab.seg[k] = (sp_segment){slot_of(obj), first, count, (int) off};
    tab.seg_obj[k] = obj;
    off += count;
    ++k;
  }
  *n_ent = off;
  return k;
}

/** @brief Device -> host: field @p f of the bodies in @p list over @p r. */
static void pull(sp_field f, entity_range r, MFTL_Object *const *list, int n)
{
  long n_ent = 0;
  const int n_seg = make_segments(list, n, field_on_tris(f), r, &n_ent);
  if (n_seg == 0) return;
  const int w = field_width(f);
  const void *st = sp_download(f, n_seg, tab.seg);

  for (int k = 0; k < n_seg; ++k)
  {
    const sp_segment *sg = &tab.seg[k];
    const size_t m = (size_t) w * sg->count;
    if (field_is_int(f))
      memcpy(host_ints(tab.seg_obj[k], f) + (size_t) w * sg->first, (const int *) st + (size_t) w * sg->offset, m * sizeof(int));
    else
      memcpy(host_doubles(tab.seg_obj[k], f) + (size_t) w * sg->first, (const double *) st + (size_t) w * sg->offset, m * sizeof(double));
  }
}

/** @brief Host -> device: field @p f of the bodies in @p list over @p r. */
static void push(sp_field f, entity_range r, MFTL_Object *const *list, int n)
{
  long n_ent = 0;
  const int n_seg = make_segments(list, n, field_on_tris(f), r, &n_ent);
  if (n_seg == 0) return;
  const int w = field_width(f);

  if (field_is_int(f))
  {
    int *st = sp_stage_int(w * n_ent);
    for (int k = 0; k < n_seg; ++k)
    {
      const sp_segment *sg = &tab.seg[k];
      memcpy(st + (size_t) w * sg->offset, host_ints(tab.seg_obj[k], f) + (size_t) w * sg->first, (size_t) w * sg->count * sizeof(int));
    }
  }
  else
  {
    double *st = sp_stage_double(w * n_ent);
    for (int k = 0; k < n_seg; ++k)
    {
      const sp_segment *sg = &tab.seg[k];
      memcpy(st + (size_t) w * sg->offset, host_doubles(tab.seg_obj[k], f) + (size_t) w * sg->first, (size_t) w * sg->count * sizeof(double));
    }
  }
  sp_upload(f, n_seg, tab.seg);
}

/** @brief Whole bodies, host -> device: positions, corners, owners, counts. Their slots are assigned. */
static void upload_bodies(MFTL_Object *const *list, int n)
{
  for (int i = 0; i < n; ++i)
  {
    MFTL_Object *obj = list[i];
    const int s = slot_of(obj);
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    if (obj->verts.n > tab.nvs || obj->tris.n > tab.nts)
    {
      fprintf(stderr, "spheres_gpu: body %d has %d vertices / %d triangles here, a slot holds %d / %d\n", obj->global_id, obj->verts.n, obj->tris.n,
              tab.nvs, tab.nts);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *vo = tab.vown + (size_t) s * tab.nvs, *to = tab.town + (size_t) s * tab.nts;
    for (int v = 0; v < obj->verts.n; ++v) vo[v] = obj->verts.rec[v].owner_proc;
    for (int t = 0; t < obj->tris.n; ++t) to[t] = obj->tris.rec[t].owner_proc;

    int *c = &tab.counts[SP_N_COUNTS * s];
    c[SP_NV] = obj->verts.n;
    c[SP_NV_OWNED] = obj->verts.n_local;
    c[SP_NT] = obj->tris.n;
    c[SP_NT_OWNED] = obj->tris.n_local;
    tab.counts_dirty = 1;
  }
  push(SP_POS, RANGE_ALL, list, n);
  push(SP_TRI_VERTS, RANGE_ALL, list, n);
  push(SP_VOWNER, RANGE_ALL, list, n);
  push(SP_TOWNER, RANGE_ALL, list, n);
}

static void flush_counts(void)
{
  if (!tab.counts_dirty) return;
  sp_set_counts(tab.counts);
  tab.counts_dirty = 0;
}

/** @brief The bodies of the rank another rank holds part of, into tab.list. */
static int split_bodies(const MFTL_Ensemble *ens)
{
  int n = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (is_split(ens->local_objects[o])) tab.list[n++] = ens->local_objects[o];
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The per-body statistics: MFTLEnsembleUpdate() with the device's rows
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief unpack_slot() of ensemble_update.c: averages over the GLOBAL counts, zeros for an empty body. */
static void unpack_quality(MFTL_ObjectQualityStatistics *qs, const double *r)
{
  const double n_tris = r[9], n_edges = r[10];
  const int empty = n_tris <= 0.0;
  qs->avg_triangle_area = empty ? 0.0 : r[6] / n_tris;
  qs->avg_aspect_ratio = empty ? 0.0 : r[7] / n_tris;
  qs->avg_edge_length_sq = (n_edges > 0.0) ? r[8] / n_edges : 0.0;
  qs->min_triangle_area = empty ? 0.0 : r[0];
  qs->max_triangle_area = empty ? 0.0 : r[3];
  qs->min_aspect_ratio = empty ? 0.0 : r[1];
  qs->max_aspect_ratio = empty ? 0.0 : r[4];
  qs->min_edge_length_sq = empty ? 0.0 : r[2];
  qs->max_edge_length_sq = empty ? 0.0 : r[5];
}

/* Neumaier-compensated sum: the plain running sum over ~10^5 similar terms
 * drifts by ~1e-14 relative and depends on the order (the rank count), which
 * would hide the device's actual agreement (~1e-16) in the checks. */
typedef struct
{
  double s, c;
} ksum;

static inline void ksum_add(ksum *k, double x)
{
  const double t = k->s + x;
  k->c += (fabs(k->s) >= fabs(x)) ? (k->s - t) + x : (x - t) + k->s;
  k->s = t;
}

static inline double ksum_value(ksum k) { return k.s + k.c; }

/** @brief Device statistics -> per-body rows -> the reduction over each body's holders -> obj->quality. Returns the owned area. */
static double device_quality(MFTL_Ensemble *ens)
{
  sp_geometry(tab.stats);

  const int n = ens->n_local_objects;
  double *rows = scratch((size_t) SP_ROW * (n > 0 ? n : 1));
  ksum area = {0.0, 0.0};
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    double *r = rows + (size_t) SP_ROW * o;
    memcpy(r, tab.stats + (size_t) SP_N_STATS * slot_of(obj), SP_N_STATS * sizeof(double));
    r[9] = obj->tris.n_local;
    r[10] = 3.0 * obj->tris.n_local;
    ksum_add(&area, r[6]);
  }
  MFTLEnsembleReduceBodies(ens, rows, SP_ROW, row_ops, MFTL_BODY_REDUCE_SCOPE_AUTO);
  for (int o = 0; o < n; ++o) unpack_quality(&ens->local_objects[o]->quality, rows + (size_t) SP_ROW * o);
  return ksum_value(area);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Checks
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double area;  ///< max |A_dev - A_lib| / A_lib, every triangle
  double normal;  ///< max |n_dev - n_lib|, every triangle
  double center;  ///< max |c_dev - c_lib| (minimum image), every triangle
  double vnormal;  ///< max |n_dev - n_lib|, every vertex
  double quality;  ///< max relative difference of the nine statistics, every body
  double conserve;  ///< max |A(t) - A(0)| / A(0); counts must be exact
} errors;

/* Largest values measured (refine 2-3, np 1-4, every backend): 4e-16, 4e-16,
 * 9e-16, 4e-16, 1e-14, 3e-15. The quality's 1e-14 is |e|^2 of the longest
 * edge: a minimum-image difference of O(1) coordinates over an edge of
 * h ~ 0.016 is good to ~2 eps / h. Each tolerance is about 100x the value. */
static const errors verify_tol = {1e-13, 1e-13, 1e-13, 1e-13, 1e-12, 1e-12};

static double rel_diff(double a, double b) { return fabs(a - b) / fmax(fabs(b), 1e-300); }

/** @brief Owned area, owned triangles and owned vertices of the rank, from the host pools. */
static void host_totals(const MFTL_Ensemble *ens, double out[3])
{
  ksum area = {0.0, 0.0};
  out[1] = out[2] = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t) ksum_add(&area, obj->tris.area[t]);
    out[1] += obj->tris.n_local;
    out[2] += obj->verts.n_local;
  }
  out[0] = ksum_value(area);
}

/**
 * @brief Every device result of this step against the library, on the device's positions.
 *        Collective. The device values are written back where the step left them.
 */
static void verify_device(MFTL_Ensemble *ens, errors *e)
{
  const int n = ens->n_local_objects;
  MFTL_Object **list = ens->local_objects;
  const double *L = ens->domain.domain_size, *invL = ens->domain.inv_domain_size;

  /* The device holds the positions; the host's are stale on an unsplit body. */
  pull(SP_POS, RANGE_ALL, list, n);

  /* Quality: the device's (unpacked by device_quality) against MFTLEnsembleUpdate(). */
  MFTL_ObjectQualityStatistics *dev_q = (MFTL_ObjectQualityStatistics *) xrealloc(NULL, (size_t) (n > 0 ? n : 1) * sizeof *dev_q);
  for (int o = 0; o < n; ++o) dev_q[o] = list[o]->quality;
  MFTLEnsembleUpdate(ens);
  for (int o = 0; o < n; ++o)
  {
    const double *a = &dev_q[o].min_triangle_area, *b = &list[o]->quality.min_triangle_area;
    for (int k = 0; k < 9; ++k) e->quality = fmax(e->quality, rel_diff(a[k], b[k]));
    list[o]->quality = dev_q[o];
  }
  free(dev_q);

  /* Triangle geometry, ghosts included. */
  size_t nt = 0, nv = 0;
  for (int o = 0; o < n; ++o)
  {
    MFTLComputeTriangleGeometry(list[o], list[o]->tris.n);
    nt += (size_t) list[o]->tris.n;
    nv += (size_t) list[o]->verts.n;
  }
  double *ref = scratch(7 * nt + 3 * nv);
  size_t at = 0;
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *obj = list[o];
    const size_t m = (size_t) obj->tris.n;
    memcpy(ref + at, obj->tris.area, m * sizeof(double));
    memcpy(ref + nt + 3 * at, obj->tris.normal, 3 * m * sizeof(double));
    memcpy(ref + 4 * nt + 3 * at, obj->tris.center, 3 * m * sizeof(double));
    at += m;
  }
  pull(SP_AREA, RANGE_ALL, list, n);
  pull(SP_NORMAL, RANGE_ALL, list, n);
  pull(SP_CENTER, RANGE_ALL, list, n);
  at = 0;
  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *obj = list[o];
    for (int t = 0; t < obj->tris.n; ++t, ++at)
    {
      e->area = fmax(e->area, rel_diff(obj->tris.area[t], ref[at]));
      for (int k = 0; k < 3; ++k)
      {
        e->normal = fmax(e->normal, fabs(obj->tris.normal[3 * t + k] - ref[nt + 3 * at + k]));
        double d = obj->tris.center[3 * t + k] - ref[4 * nt + 3 * at + k];
        d -= L[k] * nearbyint(d * invL[k]);  // a centroid on the seam may wrap either way
        e->center = fmax(e->center, fabs(d));
      }
    }
  }

  /* Vertex normals, ghosts included: the library's after its exchange, the device's after the driver's. */
  MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_AREA);
  double *vref = ref + 7 * nt;
  at = 0;
  for (int o = 0; o < n; ++o)
  {
    memcpy(vref + at, list[o]->verts.normal, 3 * (size_t) list[o]->verts.n * sizeof(double));
    at += 3 * (size_t) list[o]->verts.n;
  }
  pull(SP_VNORMAL, RANGE_ALL, list, n);
  at = 0;
  for (int o = 0; o < n; ++o)
    for (int k = 0; k < 3 * list[o]->verts.n; ++k, ++at) e->vnormal = fmax(e->vnormal, fabs(list[o]->verts.normal[k] - vref[at]));
}

/** @brief The final state, independent of the partition: what -save writes and -compare checks. */
typedef struct
{
  double area, n_tri, n_vert;  ///< owned, summed over ranks
  double sx, sy, sz;  ///< sum of (x - global_min) over the owned vertices: positive, so a relative tolerance applies
  double sn;  ///< sum of |n_x| + |n_y| + |n_z| over the owned vertex normals
  double min_q, max_q, min_esq, max_esq;  ///< over every body
} final_state;
#define FS_N 11

static final_state measure_final(const MFTL_Ensemble *ens)
{
  ksum acc[7] = {{0.0, 0.0}};
  double sums[7], mins[2] = {DBL_MAX, DBL_MAX}, maxs[2] = {-DBL_MAX, -DBL_MAX};
  const double *lo = ens->domain.global_min;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t) ksum_add(&acc[0], obj->tris.area[t]);
    ksum_add(&acc[1], obj->tris.n_local);
    ksum_add(&acc[2], obj->verts.n_local);
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      for (int k = 0; k < 3; ++k) ksum_add(&acc[3 + k], obj->verts.pos[3 * v + k] - lo[k]);
      ksum_add(&acc[6], fabs(obj->verts.normal[3 * v]) + fabs(obj->verts.normal[3 * v + 1]) + fabs(obj->verts.normal[3 * v + 2]));
    }
    if (obj->tris.n_local > 0)
    {
      mins[0] = fmin(mins[0], obj->quality.min_aspect_ratio);
      mins[1] = fmin(mins[1], obj->quality.min_edge_length_sq);
      maxs[0] = fmax(maxs[0], obj->quality.max_aspect_ratio);
      maxs[1] = fmax(maxs[1], obj->quality.max_edge_length_sq);
    }
  }
  for (int k = 0; k < 7; ++k) sums[k] = ksum_value(acc[k]);
  double gs[7], gmin[2], gmax[2];
  MPI_Allreduce(sums, gs, 7, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(mins, gmin, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(maxs, gmax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  return (final_state){gs[0], gs[1], gs[2], gs[3], gs[4], gs[5], gs[6], gmin[0], gmax[0], gmin[1], gmax[1]};
}

static const char *const fs_name[FS_N] = {"area", "triangles", "vertices", "sum x", "sum y", "sum z", "sum |n|_1", "min q", "max q", "min |e|^2", "max |e|^2"};

static int save_final(const char *path, const final_state *fs)
{
  FILE *f = fopen(path, "w");
  if (!f) return 1;
  const double *v = &fs->area;
  for (int i = 0; i < FS_N; ++i) fprintf(f, "%s%.17g", i ? " " : "", v[i]);
  fprintf(f, "\n");
  return fclose(f) != 0;
}

/** @brief Rank 0: the final state against FILE. Counts exact, the rest to 1e-12 relative. */
static int compare_final(const char *path, const final_state *fs)
{
  FILE *f = fopen(path, "r");
  double ref[FS_N];
  int n = 0;
  if (f)
  {
    while (n < FS_N && fscanf(f, "%lf", &ref[n]) == 1) ++n;
    fclose(f);
  }
  if (n != FS_N)
  {
    printf("COMPARE: cannot read %d values from %s -> FAILED\n", FS_N, path);
    return 1;
  }
  const double *v = &fs->area;
  int ok = 1;
  printf("COMPARE final state against %s:\n", path);
  for (int i = 0; i < FS_N; ++i)
  {
    const int exact = (i == 1 || i == 2);
    const double d = exact ? fabs(v[i] - ref[i]) : rel_diff(v[i], ref[i]);
    const int good = exact ? d == 0.0 : d <= 1e-12;
    ok &= good;
    printf("    %-10s %22.15e   %s %.2e %s\n", fs_name[i], v[i], exact ? "abs" : "rel", d, good ? "ok" : "FAILED");
  }
  return !ok;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

static void grid_for(int nproc, int cuts[3])
{
  static const int table[][4] = {{1, 1, 1, 1}, {2, 2, 1, 1}, {3, 3, 1, 1}, {4, 2, 2, 1}, {6, 3, 2, 1},
                                 {8, 2, 2, 2}, {12, 3, 2, 2}, {16, 4, 2, 2}, {24, 4, 3, 2}, {27, 3, 3, 3}, {32, 4, 4, 2}};
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i)
    if (table[i][0] == nproc)
    {
      cuts[0] = table[i][1];
      cuts[1] = table[i][2];
      cuts[2] = table[i][3];
      return;
    }
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
}

static void usage_error(int rank, const char *msg, const char *arg)
{
  if (rank == 0) fprintf(stderr, "spheres_gpu: %s%s%s\n", msg, arg ? ": " : "", arg ? arg : "");
  MPI_Finalize();
  exit(2);
}

static double parse_double(int rank, const char *s, double lo)
{
  char *end = NULL;
  const double x = strtod(s, &end);
  if (!s[0] || *end || !isfinite(x) || x <= lo) usage_error(rank, "invalid number", s);
  return x;
}

static int parse_int(int rank, const char *s, long lo, long hi)
{
  char *end = NULL;
  const long x = strtol(s, &end, 10);
  if (!s[0] || *end || x < lo || x > hi) usage_error(rank, "invalid integer", s);
  return (int) x;
}

static void geometry_all(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int n_spheres = 1000, refinements = 3, n_steps = 100, n_slots = 0, use_cpu = 0, verify = 0, quiet = 0;
  double radius = 0.1, dt = 0.1;
  const char *save_path = NULL, *compare_path = NULL;

  for (int i = 1; i < argc; ++i)
  {
    const char *a = argv[i];
    const int has_value = i + 1 < argc;
    if (!strcmp(a, "-spheres") && has_value)
      n_spheres = parse_int(rank, argv[++i], 1, 10000000);
    else if (!strcmp(a, "-refine") && has_value)
      refinements = parse_int(rank, argv[++i], 0, 6);
    else if (!strcmp(a, "-radius") && has_value)
      radius = parse_double(rank, argv[++i], 0.0);
    else if (!strcmp(a, "-dt") && has_value)
      dt = parse_double(rank, argv[++i], 0.0);
    else if (!strcmp(a, "-steps") && has_value)
      n_steps = parse_int(rank, argv[++i], 0, 10000000);
    else if (!strcmp(a, "-slots") && has_value)
      n_slots = parse_int(rank, argv[++i], 1, 100000000);
    else if (!strcmp(a, "-save") && has_value)
      save_path = argv[++i];
    else if (!strcmp(a, "-compare") && has_value)
      compare_path = argv[++i];
    else if (!strcmp(a, "-cpu"))
      use_cpu = 1;
    else if (!strcmp(a, "-verify"))
      verify = 1;
    else if (!strcmp(a, "-quiet"))
      quiet = 1;
    else
      usage_error(rank, "unknown or incomplete option (see the head of spheres_gpu.c)", a);
  }

  /* Every rank on one node shares its devices round-robin. */
  int local_rank = 0;
  {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
  }
  if (!use_cpu)
  {
    int status = sp_init(local_rank), worst = 0;
    MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (worst != 0)
    {
      if (rank == 0) printf("spheres_gpu (%s): no device on some rank -- skipped\n", sp_backend_name());
      MFTLFreeMpiTypes();
      MPI_Finalize();
      return 77;
    }
  }
  const char *backend = use_cpu ? "cpu (library)" : sp_backend_name();
  /* The *-host builds run the same pragmas on CPU threads: no device, no copies. */
  const int on_device = !use_cpu && !strstr(backend, "host");

  /* --- Setup: parallel_advection_redistribute_list.c, all axes periodic ----- */
  int cuts[3];
  grid_for(nproc, cuts);
  bool periodic[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  double gmin[3] = {-4.0, -2.0, -2.0}, gmax[3] = {4.0, 2.0, 2.0};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, gmin, gmax, periodic, n_spheres);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  /* Fixed seed on every rank: the centres must agree, generation is collective. */
  srand(42);
  for (int b = 0; b < n_spheres; ++b)
  {
    MFTL_Object *obj = MFTLObjectCreate(b);
    MFTLEnsembleAddObject(ens, obj);
    double centre[3];
    for (int k = 0; k < 3; ++k) centre[k] = gmin[k] + ((double) rand() / (double) RAND_MAX) * (gmax[k] - gmin[k]);
    MFTLGenerateSphereLocal(obj, centre, radius, refinements);
  }
  MFTLPublishGeneratedBodies(ens);

  /* Scatter from rank 0, then once more without motion: every body settled,
   * so the first step does not rebuild them all. */
  for (int pass = 0; pass < 2; ++pass)
  {
    geometry_all(ens);
    if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  }
  geometry_all(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);

  const int nt_body = 20 << (2 * refinements), nv_body = nt_body / 2 + 2;
  const double velocity[3] = {1.0, 0.37, 0.61};

  double reference[3], local3[3];
  host_totals(ens, local3);
  MPI_Allreduce(local3, reference, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* --- The device: slots, box, the bodies of this rank --------------------- */
  double device_bytes = 0.0;
  if (!use_cpu)
  {
    const int held = ens->n_local_objects;
    int S = n_slots > 0 ? n_slots : held + held / 2 + 64;
    if (S > n_spheres) S = n_spheres;
    if (S < held) S = held;
    table_create(n_spheres, S, nv_body, nt_body);

    const MFTL_Domain *d = &ens->domain;
    sp_box box;
    for (int k = 0; k < 3; ++k)
    {
      box.min[k] = d->global_min[k];
      box.max[k] = d->global_max[k];
      box.L[k] = d->domain_size[k];
      box.invL[k] = d->is_periodic[k] ? d->inv_domain_size[k] : 0.0;
      box.gf[k] = d->grid_factors[k];
      box.cuts[k] = d->cuts[k];
      box.periodic[k] = d->is_periodic[k] ? 1 : 0;
    }
    sp_setup(S, nv_body, nt_body, &box, d->rank_map, nproc);
    device_bytes = sp_device_bytes();

    for (int o = 0; o < held; ++o)
    {
      slot_acquire(ens->local_objects[o]->global_id);
      tab.list[o] = ens->local_objects[o];
    }
    upload_bodies(tab.list, held);
    flush_counts();
  }

  if (rank == 0 && !quiet)
  {
    printf("# %d spheres, periodic advection + per-step geometry, quality, vertex normals; device work on: %s\n", n_spheres, backend);
    printf("#   %d ranks on a %dx%dx%d grid   r=%g, refine %d -> %d triangles / %d vertices each   dt=%g, %d steps\n", nproc, cuts[0], cuts[1], cuts[2],
           radius, refinements, nt_body, nv_body, dt, n_steps);
    printf("#   A0 = %.12e   %.0f triangles, %.0f vertices\n", reference[0], reference[1], reference[2]);
  }

  /* --- The loop ------------------------------------------------------------ */
  double tm[T_N] = {0.0}, t_verify = 0.0;
  double h2d0 = 0.0, d2h0 = 0.0;
  long copies0 = 0;
  if (!use_cpu) sp_traffic(&h2d0, &d2h0, &copies0);
  long n_split = 0, n_changed = 0, n_needed = 0, n_uploaded = 0;
  errors err = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  int failures = 0;

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();

  for (int step = 0; step < n_steps; ++step)
  {
    double a, owned_area = 0.0;

    if (use_cpu)
    {
      a = MPI_Wtime();
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        MFTL_Object *obj = ens->local_objects[o];
        MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += velocity[0] * dt; pos[1] += velocity[1] * dt; pos[2] += velocity[2] * dt;
                                  MFTLApplyPeriodicity(&ens->domain, pos););
      }
      tm[T_ADVECT] += MPI_Wtime() - a;

      if (nproc > 1)
      {
        a = MPI_Wtime();
        MFTLUpdateGhostPositions(ens);
        tm[T_GHOST] += MPI_Wtime() - a;

        /* What MFTLRedistribute() routes by: every centroid, ghosts included. */
        a = MPI_Wtime();
        geometry_all(ens);
        tm[T_OWNERS] += MPI_Wtime() - a;

        a = MPI_Wtime();
        MFTLRedistribute(ens);
        tm[T_REDIST] += MPI_Wtime() - a;
      }

      /* The device's end state: geometry of every triangle (MFTLEnsembleUpdate
       * covers the owned ones), statistics, vertex normals. */
      a = MPI_Wtime();
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        MFTL_Object *obj = ens->local_objects[o];
        if (obj->tris.n > obj->tris.n_local) MFTLComputeTriangleGeometry(obj, obj->tris.n);
      }
      MFTLEnsembleUpdate(ens);
      tm[T_GEOM] += MPI_Wtime() - a;

      a = MPI_Wtime();
      for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeVertexNormals(ens->local_objects[o], MFTL_NORMAL_WEIGHT_AREA);
      tm[T_NORMALS] += MPI_Wtime() - a;

      if (nproc > 1)
      {
        a = MPI_Wtime();
        MFTLUpdateGhostVertexNormals(ens);
        tm[T_GHOST] += MPI_Wtime() - a;
      }

      if (verify)
      {
        host_totals(ens, local3);
        owned_area = local3[0];
      }
    }
    else
    {
      a = MPI_Wtime();
      sp_advect(velocity[0], velocity[1], velocity[2], dt);
      tm[T_ADVECT] += MPI_Wtime() - a;

      if (nproc > 1)
      {
        /* 1. Ghost positions: only a split body has any, and only its owned
         *    vertices are anybody's ghosts. */
        const int ns = split_bodies(ens);
        n_split += ns;
        for (int o = 0; o < ens->n_local_objects; ++o) tab.split[slot_of(ens->local_objects[o])] = (unsigned char) is_split(ens->local_objects[o]);

        a = MPI_Wtime();
        pull(SP_POS, RANGE_OWNED, tab.list, ns);
        tm[T_D2H] += MPI_Wtime() - a;

        a = MPI_Wtime();
        MFTLUpdateGhostPositions(ens);
        tm[T_GHOST] += MPI_Wtime() - a;

        a = MPI_Wtime();
        push(SP_POS, RANGE_GHOST, tab.list, ns);
        tm[T_H2D] += MPI_Wtime() - a;

        /* The ghost sync renewed every generation: the stamp a rebuild will change. */
        for (int o = 0; o < ens->n_local_objects; ++o) tab.stamp[slot_of(ens->local_objects[o])] = ens->local_objects[o]->generation;

        /* 2. Owners on the device; the host mirror follows where one changed. */
        a = MPI_Wtime();
        sp_classify(tab.changed);
        tm[T_OWNERS] += MPI_Wtime() - a;

        int nc = 0;
        for (int o = 0; o < ens->n_local_objects; ++o)
          if (tab.changed[slot_of(ens->local_objects[o])]) tab.list[nc++] = ens->local_objects[o];
        n_changed += nc;

        a = MPI_Wtime();
        pull(SP_VOWNER, RANGE_ALL, tab.list, nc);
        pull(SP_TOWNER, RANGE_ALL, tab.list, nc);

        /* 3. The bodies the redistribution will send from this rank: dirty
         *    (changed, stamped, no topology) or not yet settled. A split body's
         *    host positions are current already; an unsplit one's come down. */
        int nn = 0;
        for (int o = 0; o < ens->n_local_objects; ++o)
        {
          MFTL_Object *obj = ens->local_objects[o];
          const int s = slot_of(obj);
          tab.needed[s] = (unsigned char) (!tab.split[s] && (tab.changed[s] || obj->partition_dirty || !MFTLObjectHasTopology(obj) || !obj->partition_settled));
          if (tab.needed[s]) tab.list[nn++] = obj;
        }
        n_needed += nn;
        pull(SP_POS, RANGE_ALL, tab.list, nn);
        tm[T_D2H] += MPI_Wtime() - a;

        /* 4. Redistribution by the device's owners. */
        a = MPI_Wtime();
        const int n_pre = ens->n_local_objects;
        MFTL_OwnerMap *maps = (MFTL_OwnerMap *) xrealloc(NULL, (size_t) (n_pre > 0 ? n_pre : 1) * sizeof *maps);
        for (int o = 0; o < n_pre; ++o)
        {
          const int s = slot_of(ens->local_objects[o]);
          maps[o].tri_owner = tab.town + (size_t) s * tab.nts;
          maps[o].vert_owner = tab.vown + (size_t) s * tab.nvs;
        }
        MFTLRedistributeWithOwners(ens, maps);
        free(maps);

        /* Departures free their slots first, then arrivals take one. */
        const long epoch = ++tab.epoch;
        for (int o = 0; o < ens->n_local_objects; ++o)
        {
          const int s = slot_of(ens->local_objects[o]);
          if (s >= 0) tab.seen[s] = epoch;
        }
        for (int s = 0; s < tab.S; ++s)
          if (tab.body[s] >= 0 && tab.seen[s] != epoch) slot_release(s);

        int nu = 0;
        for (int o = 0; o < ens->n_local_objects; ++o)
        {
          MFTL_Object *obj = ens->local_objects[o];
          const int s = slot_of(obj);
          if (s < 0)
          {
            slot_acquire(obj->global_id);
            tab.list[nu++] = obj;
          }
          else if (obj->generation != tab.stamp[s])
          {
            /* Rebuilt from the host pools: they had better have been current. */
            if (!tab.split[s] && !tab.needed[s])
            {
              fprintf(stderr, "spheres_gpu: rank %d step %d: body %d was rebuilt from stale host positions\n", rank, step, obj->global_id);
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
            tab.list[nu++] = obj;
          }
        }
        n_uploaded += nu;
        tm[T_REDIST] += MPI_Wtime() - a;

        a = MPI_Wtime();
        upload_bodies(tab.list, nu);
        flush_counts();
        tm[T_H2D] += MPI_Wtime() - a;
      }

      /* 5. Geometry and statistics. */
      a = MPI_Wtime();
      owned_area = device_quality(ens);
      tm[T_GEOM] += MPI_Wtime() - a;

      /* 6. Vertex normals; the ghosts' come from their owners. */
      a = MPI_Wtime();
      sp_normals();
      tm[T_NORMALS] += MPI_Wtime() - a;

      if (nproc > 1)
      {
        const int ns = split_bodies(ens);
        a = MPI_Wtime();
        pull(SP_VNORMAL, RANGE_OWNED, tab.list, ns);
        tm[T_D2H] += MPI_Wtime() - a;

        a = MPI_Wtime();
        MFTLUpdateGhostVertexNormals(ens);
        tm[T_GHOST] += MPI_Wtime() - a;

        a = MPI_Wtime();
        push(SP_VNORMAL, RANGE_GHOST, tab.list, ns);
        tm[T_H2D] += MPI_Wtime() - a;
      }
    }

    if (verify)
    {
      a = MPI_Wtime();
      /* Rigid translation conserves the area exactly, redistribution the entities. */
      double local[3], total[3];
      local[0] = owned_area;
      local[1] = local[2] = 0.0;
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        local[1] += ens->local_objects[o]->tris.n_local;
        local[2] += ens->local_objects[o]->verts.n_local;
      }
      MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      const double rel = fabs(total[0] - reference[0]) / reference[0];
      err.conserve = fmax(err.conserve, rel);
      if (total[1] != reference[1] || total[2] != reference[2])
      {
        if (rank == 0)
          fprintf(stderr, "step %d: %.0f triangles, %.0f vertices -- expected %.0f / %.0f\n", step, total[1], total[2], reference[1], reference[2]);
        failures++;
      }
      if (!use_cpu) verify_device(ens, &err);
      t_verify += MPI_Wtime() - a;
    }

    if (rank == 0 && !quiet && n_steps >= 10 && (step + 1) % (n_steps / 10) == 0) printf("  step %d/%d\n", step + 1, n_steps);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t_start - t_verify;

  /* --- Summary --------------------------------------------------------------- */
  double h2d = 0.0, d2h = 0.0;
  long copies = 0;
  if (!use_cpu)
  {
    sp_traffic(&h2d, &d2h, &copies);
    h2d -= h2d0, d2h -= d2h0, copies -= copies0;

    /* The final state is on the device. */
    pull(SP_POS, RANGE_OWNED, ens->local_objects, ens->n_local_objects);
    pull(SP_VNORMAL, RANGE_OWNED, ens->local_objects, ens->n_local_objects);
    pull(SP_AREA, RANGE_OWNED, ens->local_objects, ens->n_local_objects);
  }
  const final_state fs = measure_final(ens);

  double tmax[T_N], traffic[3] = {h2d, d2h, (double) copies}, traffic_max[3], dev_max = 0.0;
  MPI_Reduce(tm, tmax, T_N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(traffic, traffic_max, 3, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&device_bytes, &dev_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  long counters[4] = {n_split, n_changed, n_needed, n_uploaded}, counters_sum[4];
  MPI_Reduce(counters, counters_sum, 4, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  errors gerr;
  MPI_Reduce(&err, &gerr, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  int gfail = 0;
  MPI_Reduce(&failures, &gfail, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  int failed = 0;
  if (rank == 0)
  {
    const double per = 1000.0 / (n_steps > 0 ? n_steps : 1), steps = n_steps > 0 ? n_steps : 1;
    const double offload = tmax[T_ADVECT] + tmax[T_OWNERS] + tmax[T_GEOM] + tmax[T_NORMALS] + tmax[T_D2H] + tmax[T_H2D];
    printf("==============================================\n");
    printf("  spheres on %s, %d ranks: %d spheres x %d triangles, %d steps\n", backend, nproc, n_spheres, nt_body, n_steps);
    printf("  final  A/A0-1 %.3e   q in [%.6f, %.6f]   |e|^2 in [%.6e, %.6e]\n", fs.area / reference[0] - 1.0, fs.min_q, fs.max_q, fs.min_esq, fs.max_esq);
    printf("  TIME per step (max over ranks)            ms/step   share\n");
    for (int i = 0; i < T_N; ++i)
    {
      const char *where = (i <= T_NORMALS) ? (on_device ? "device" : "host  ") : (i <= T_H2D ? (on_device ? "PCIe  " : "host  ") : "host  ");
      printf("    %-12s %s %10.4f %6.1f %%\n", timer_name[i], where, per * tmax[i], 100.0 * tmax[i] / wall);
    }
    printf("    %-19s %10.4f\n", "wall", per * wall);
    printf("    %-19s %10.4f %6.1f %%   (advect + owners + geometry + normals + transfers)\n", "offloadable part", per * offload, 100.0 * offload / wall);
    if (use_cpu && nproc > 1) printf("    (-cpu: 'owners' is the geometry pass MFTLRedistribute routes by; 'geometry' includes the body reduction)\n");
    if (!use_cpu)
    {
      if (nproc > 1)
        printf("    per step and rank: %.1f bodies split, %.1f with an owner change, %.1f positions sent down, %.1f bodies re-uploaded\n",
               counters_sum[0] / steps / nproc, counters_sum[1] / steps / nproc, counters_sum[2] / steps / nproc, counters_sum[3] / steps / nproc);
      if (on_device)
        printf("    transfers (max over ranks): H->D %.3f MB/step, D->H %.3f MB/step, %.1f copies/step\n", 1e-6 * traffic_max[0] / steps,
               1e-6 * traffic_max[1] / steps, traffic_max[2] / steps);
      printf("    device memory: %d slots, %.1f MB per rank (max), allocated once\n", tab.S, 1e-6 * dev_max);
    }

    if (verify)
    {
      const int conserve_ok = gerr.conserve <= verify_tol.conserve && gfail == 0;
      printf("VERIFY conservation: max |A/A0 - 1| %.3e, entity counts %s -> %s\n", gerr.conserve, gfail ? "WRONG" : "exact", conserve_ok ? "ok" : "FAILED");
      failed |= !conserve_ok;
      if (!use_cpu)
      {
        const double *e = &gerr.area, *tol = &verify_tol.area;
        static const char *const what[5] = {"area (rel)", "tri normals", "centroids", "vertex normals", "quality (rel)"};
        printf("VERIFY device vs library, every step, max over ranks:\n");
        for (int i = 0; i < 5; ++i)
        {
          const int ok = e[i] <= tol[i];
          failed |= !ok;
          printf("    %-15s %10.3e   (tol %.0e) %s\n", what[i], e[i], tol[i], ok ? "ok" : "FAILED");
        }
      }
    }
    if (save_path && save_final(save_path, &fs))
    {
      printf("SAVE: cannot write %s\n", save_path);
      failed = 1;
    }
    if (compare_path && compare_final(compare_path, &fs)) failed = 1;
  }
  MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (!use_cpu)
  {
    sp_finalize();
    table_destroy();
  }
  free(g_scratch);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
