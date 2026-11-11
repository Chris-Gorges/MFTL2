/*
 * @Author: Christian Gorges
 *
 * @brief Many-spheres GPU example, the directive backend: ONE source for
 *        OpenACC, OpenMP offload, OpenMP threads and serial C (gpu_pragmas.h).
 *
 * Every loop is the host loop nest of the library -- for each body, for each
 * vertex or triangle -- with GPU_OUTER on the first loop and GPU_INNER on the
 * second. On a GPU that is one gang / team per body and its entities over the
 * vector lanes / threads; on CPU threads it is one body per thread, serial
 * inside, which is why the normal scatter needs no atomic there.
 *
 * The slot arrays are host arrays mapped once in sp_setup() and never touched
 * on the host by the GPU builds, so they cost address space, not memory. All
 * traffic goes through the mapped staging buffers and one `update` per call.
 */
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gpu_pragmas.h"
#include "sp_backend.h"
#include "sp_kernels.h"

#ifndef GPU_BACKEND_NAME
#define GPU_BACKEND_NAME GPU_MODEL
#endif

/* 1 for the GPU builds: refuse to run when there is no GPU instead of quietly using the host. */
#ifndef GPU_REQUIRE_DEVICE
#define GPU_REQUIRE_DEVICE 0
#endif

static int S = 0, NVS = 0, NTS = 0, NPROC = 0;  // slots, vertices and triangles per slot, ranks

static sp_box *box;  // [1]: the loops read the box from the device like any other array
static int *rank_map;  // [NPROC]
static double *pos, *vn;  // [3 S NVS]
static double *area, *normal, *center;  // [S NTS], [3 S NTS], [3 S NTS]
static int *tv;  // [3 S NTS]
static int *vown, *town;  // [S NVS], [S NTS]
static int *cnt;  // [SP_N_COUNTS S]
static int *changed;  // [S]
static double *stats;  // [SP_N_STATS S]

static double *dstage = NULL;  // the staging buffers and the segment table, mapped and grown on demand
static int *istage = NULL, *segs = NULL;
static long dstage_n = 0, istage_n = 0, segs_n = 0;

static double bytes_h2d = 0.0, bytes_d2h = 0.0;
static long n_copies = 0;

const char *sp_backend_name(void) { return GPU_BACKEND_NAME; }

int sp_init(int local_rank) { return gpu_select_device(local_rank, GPU_REQUIRE_DEVICE); }

static void *host_alloc(size_t bytes)
{
  void *p = calloc(bytes > 0 ? bytes : 1, 1);
  if (!p)
  {
    fprintf(stderr, "sp_backend: out of host memory (%zu bytes)\n", bytes);
    abort();
  }
  return p;
}

void sp_setup(int n_slots, int nv_slot, int nt_slot, const sp_box *b, const int *rm, int nproc)
{
  S = n_slots;
  NVS = nv_slot;
  NTS = nt_slot;
  NPROC = nproc;
  const long nv = (long) S * NVS, nt = (long) S * NTS, ns = S, np = nproc;

  box = host_alloc(sizeof *box);
  *box = *b;
  rank_map = host_alloc((size_t) np * sizeof(int));
  memcpy(rank_map, rm, (size_t) np * sizeof(int));
  pos = host_alloc((size_t) (3 * nv) * sizeof(double));
  vn = host_alloc((size_t) (3 * nv) * sizeof(double));
  area = host_alloc((size_t) nt * sizeof(double));
  normal = host_alloc((size_t) (3 * nt) * sizeof(double));
  center = host_alloc((size_t) (3 * nt) * sizeof(double));
  tv = host_alloc((size_t) (3 * nt) * sizeof(int));
  vown = host_alloc((size_t) nv * sizeof(int));
  town = host_alloc((size_t) nt * sizeof(int));
  cnt = host_alloc((size_t) (SP_N_COUNTS * ns) * sizeof(int));
  changed = host_alloc((size_t) ns * sizeof(int));
  stats = host_alloc((size_t) (SP_N_STATS * ns) * sizeof(double));

  sp_box *bx = box;
  int *rmap = rank_map, *t3 = tv, *vo = vown, *to = town, *c = cnt, *ch = changed;
  double *p = pos, *n = vn, *ar = area, *nr = normal, *ce = center, *st = stats;
  (void) bx, (void) rmap, (void) t3, (void) vo, (void) to, (void) c, (void) ch, (void) p, (void) n, (void) ar, (void) nr, (void) ce, (void) st;

  GPU_ENTER_CREATE(bx[0 : 1], rmap[0 : np], p[0 : 3 * nv], n[0 : 3 * nv], ar[0 : nt], nr[0 : 3 * nt], ce[0 : 3 * nt], t3[0 : 3 * nt], vo[0 : nv],
                   to[0 : nt], c[0 : SP_N_COUNTS * ns], ch[0 : ns], st[0 : SP_N_STATS * ns])
  GPU_UPDATE_DEVICE(bx[0 : 1], rmap[0 : np], c[0 : SP_N_COUNTS * ns])
  GPU_WAIT
}

double sp_device_bytes(void)
{
  const double nv = (double) S * NVS, nt = (double) S * NTS;
  return 8.0 * (6.0 * nv + 7.0 * nt) + 4.0 * (nv + 4.0 * nt) + (4.0 * (SP_N_COUNTS + 1) + 8.0 * SP_N_STATS) * S;
}

void sp_finalize(void)
{
  if (S == 0) return;
  const long nv = (long) S * NVS, nt = (long) S * NTS, ns = S, np = NPROC;
  sp_box *bx = box;
  int *rmap = rank_map, *t3 = tv, *vo = vown, *to = town, *c = cnt, *ch = changed;
  double *p = pos, *n = vn, *ar = area, *nr = normal, *ce = center, *st = stats;
  (void) bx, (void) rmap, (void) t3, (void) vo, (void) to, (void) c, (void) ch, (void) p, (void) n, (void) ar, (void) nr, (void) ce, (void) st, (void) np,
      (void) nv, (void) nt, (void) ns;

  GPU_EXIT_DELETE(bx[0 : 1], rmap[0 : np], p[0 : 3 * nv], n[0 : 3 * nv], ar[0 : nt], nr[0 : 3 * nt], ce[0 : 3 * nt], t3[0 : 3 * nt], vo[0 : nv],
                  to[0 : nt], c[0 : SP_N_COUNTS * ns], ch[0 : ns], st[0 : SP_N_STATS * ns])

  double *d = dstage;
  int *i = istage, *g = segs;
  const long dn = dstage_n, in = istage_n, gn = segs_n;
  (void) dn, (void) in, (void) gn;
  if (d)
  {
    GPU_EXIT_DELETE(d[0 : dn])
  }
  if (i)
  {
    GPU_EXIT_DELETE(i[0 : in])
  }
  if (g)
  {
    GPU_EXIT_DELETE(g[0 : gn])
  }

  free(box), free(rank_map), free(pos), free(vn), free(area), free(normal), free(center), free(tv), free(vown), free(town), free(cnt), free(changed),
      free(stats), free(dstage), free(istage), free(segs);
  dstage = NULL, istage = NULL, segs = NULL;
  dstage_n = istage_n = segs_n = 0;
  S = 0;
}

/* --- Staging ----------------------------------------------------------------- */

/* A grown buffer is a new host array, so its mapping is renewed with it. */
#define SP_GROW(type, buf, cap, need)                  \
  do                                                   \
  {                                                    \
    if ((need) > (cap))                                \
    {                                                  \
      type *old_ = (buf);                              \
      const long old_n_ = (cap);                       \
      (void) old_n_;                                   \
      if (old_)                                        \
      {                                                \
        GPU_EXIT_DELETE(old_[0 : old_n_])              \
      }                                                \
      free(old_);                                      \
      (cap) = (need) + (need) / 2 + 1024;              \
      (buf) = host_alloc((size_t) (cap) * sizeof(type)); \
      type *new_ = (buf);                              \
      const long new_n_ = (cap);                       \
      (void) new_, (void) new_n_;                      \
      GPU_ENTER_CREATE(new_[0 : new_n_])               \
    }                                                  \
  } while (0)

double *sp_stage_double(long n)
{
  SP_GROW(double, dstage, dstage_n, n);
  return dstage;
}

int *sp_stage_int(long n)
{
  SP_GROW(int, istage, istage_n, n);
  return istage;
}

typedef struct
{
  double *d;  // the slot array of a double field, or NULL
  int *i;  // ... of an int field
  int width;  // values per entity
  int per_slot;  // entities per slot
} field_ref;

static field_ref field_of(sp_field f)
{
  switch (f)
  {
    case SP_POS: return (field_ref){pos, NULL, 3, NVS};
    case SP_VNORMAL: return (field_ref){vn, NULL, 3, NVS};
    case SP_AREA: return (field_ref){area, NULL, 1, NTS};
    case SP_NORMAL: return (field_ref){normal, NULL, 3, NTS};
    case SP_CENTER: return (field_ref){center, NULL, 3, NTS};
    case SP_TRI_VERTS: return (field_ref){NULL, tv, 3, NTS};
    case SP_VOWNER: return (field_ref){NULL, vown, 1, NVS};
    case SP_TOWNER: return (field_ref){NULL, town, 1, NTS};
    default: abort();
  }
}

/** @brief Uploads the segment table; returns the staged values it covers. */
static long put_segments(int n_seg, const sp_segment *seg, int width)
{
  SP_GROW(int, segs, segs_n, 4L * n_seg);
  long total = 0;
  for (int g = 0; g < n_seg; ++g)
  {
    segs[4 * g + 0] = seg[g].slot;
    segs[4 * g + 1] = seg[g].first;
    segs[4 * g + 2] = seg[g].count;
    segs[4 * g + 3] = seg[g].offset;
    if (seg[g].offset + seg[g].count > total) total = seg[g].offset + seg[g].count;
  }
  int *sg = segs;
  const long n4 = 4L * n_seg;
  (void) sg, (void) n4;
  GPU_UPDATE_DEVICE(sg[0 : n4])
  bytes_h2d += 4.0 * n4;
  return total * width;
}

void sp_upload(sp_field field, int n_seg, const sp_segment *seg)
{
  if (n_seg <= 0) return;
  const field_ref f = field_of(field);
  const long n = put_segments(n_seg, seg, f.width);
  const int *restrict sg = segs;
  const int w = f.width, per = f.per_slot;

  if (f.d)
  {
    double *restrict dst = f.d;
    const double *restrict st = dstage;
    GPU_UPDATE_DEVICE(st[0 : n])
    GPU_OUTER()
    for (int g = 0; g < n_seg; ++g)
    {
      const long to = ((long) sg[4 * g] * per + sg[4 * g + 1]) * w, from = (long) sg[4 * g + 3] * w;
      const int m = sg[4 * g + 2] * w;
      GPU_INNER()
      for (int k = 0; k < m; ++k) dst[to + k] = st[from + k];
    }
    bytes_h2d += 8.0 * n;
  }
  else
  {
    int *restrict dst = f.i;
    const int *restrict st = istage;
    GPU_UPDATE_DEVICE(st[0 : n])
    GPU_OUTER()
    for (int g = 0; g < n_seg; ++g)
    {
      const long to = ((long) sg[4 * g] * per + sg[4 * g + 1]) * w, from = (long) sg[4 * g + 3] * w;
      const int m = sg[4 * g + 2] * w;
      GPU_INNER()
      for (int k = 0; k < m; ++k) dst[to + k] = st[from + k];
    }
    bytes_h2d += 4.0 * n;
  }
  GPU_WAIT
  n_copies += 2;
}

const void *sp_download(sp_field field, int n_seg, const sp_segment *seg)
{
  if (n_seg <= 0) return NULL;
  const field_ref f = field_of(field);
  long total = 0;
  for (int g = 0; g < n_seg; ++g)
    if (seg[g].offset + seg[g].count > total) total = seg[g].offset + seg[g].count;
  const long n = total * f.width;
  if (f.d)
    sp_stage_double(n);
  else
    sp_stage_int(n);
  put_segments(n_seg, seg, f.width);

  const int *restrict sg = segs;
  const int w = f.width, per = f.per_slot;
  const void *out;

  if (f.d)
  {
    const double *restrict src = f.d;
    double *restrict st = dstage;
    GPU_OUTER()
    for (int g = 0; g < n_seg; ++g)
    {
      const long from = ((long) sg[4 * g] * per + sg[4 * g + 1]) * w, to = (long) sg[4 * g + 3] * w;
      const int m = sg[4 * g + 2] * w;
      GPU_INNER()
      for (int k = 0; k < m; ++k) st[to + k] = src[from + k];
    }
    GPU_UPDATE_HOST(st[0 : n])
    bytes_d2h += 8.0 * n;
    out = st;
  }
  else
  {
    const int *restrict src = f.i;
    int *restrict st = istage;
    GPU_OUTER()
    for (int g = 0; g < n_seg; ++g)
    {
      const long from = ((long) sg[4 * g] * per + sg[4 * g + 1]) * w, to = (long) sg[4 * g + 3] * w;
      const int m = sg[4 * g + 2] * w;
      GPU_INNER()
      for (int k = 0; k < m; ++k) st[to + k] = src[from + k];
    }
    GPU_UPDATE_HOST(st[0 : n])
    bytes_d2h += 4.0 * n;
    out = st;
  }
  GPU_WAIT
  n_copies += 2;
  return out;
}

void sp_set_counts(const int *counts)
{
  int *c = cnt;
  const long n = (long) SP_N_COUNTS * S;
  memcpy(c, counts, (size_t) n * sizeof(int));
  GPU_UPDATE_DEVICE(c[0 : n])
  GPU_WAIT
  bytes_h2d += 4.0 * n;
  n_copies += 1;
}

void sp_traffic(double *h2d, double *d2h, long *n)
{
  *h2d = bytes_h2d;
  *d2h = bytes_d2h;
  *n = n_copies;
}

/* --- The loops ---------------------------------------------------------------- */

void sp_advect(double vx, double vy, double vz, double dt)
{
  const sp_box *restrict b = box;
  const int *restrict c = cnt;
  double *restrict p = pos;
  const int ns = S, nvs = NVS;

  GPU_OUTER()
  for (int s = 0; s < ns; ++s)
  {
    double *restrict ps = p + 3L * s * nvs;
    const int n = c[SP_N_COUNTS * s + SP_NV_OWNED];
    GPU_INNER()
    for (int v = 0; v < n; ++v) sp_advect_one(b, ps + 3 * v, vx, vy, vz, dt);
  }
  GPU_WAIT
}

void sp_classify(int *out)
{
  const sp_box *restrict b = box;
  const int *restrict rm = rank_map, *restrict c = cnt, *restrict t3 = tv;
  const double *restrict p = pos;
  int *restrict vo = vown, *restrict to = town, *restrict ch = changed;
  const int ns = S, nvs = NVS, nts = NTS;

  GPU_OUTER()
  for (int s = 0; s < ns; ++s)
  {
    const double *restrict ps = p + 3L * s * nvs;
    const int *restrict ts = t3 + 3L * s * nts;
    int *restrict vos = vo + (long) s * nvs, *restrict tos = to + (long) s * nts;
    const int nv = c[SP_N_COUNTS * s + SP_NV], nt = c[SP_N_COUNTS * s + SP_NT];
    int n_changed = 0;

    GPU_INNER(reduction(+ : n_changed))
    for (int v = 0; v < nv; ++v)
    {
      const int o = sp_owner(b, rm, ps + 3 * v);
      n_changed += (o != vos[v]);
      vos[v] = o;
    }

    GPU_INNER(reduction(+ : n_changed))
    for (int t = 0; t < nt; ++t)
    {
      double ctr[3];
      sp_triangle_center(b, ps, ts, t, ctr);
      const int o = sp_owner(b, rm, ctr);
      n_changed += (o != tos[t]);
      tos[t] = o;
    }
    ch[s] = n_changed;
  }
  GPU_UPDATE_HOST(ch[0 : ns])
  GPU_WAIT

  memcpy(out, changed, (size_t) S * sizeof(int));
  bytes_d2h += 4.0 * S;
  n_copies += 1;
}

void sp_geometry(double *out)
{
  const sp_box *restrict b = box;
  const int *restrict c = cnt, *restrict t3 = tv;
  const double *restrict p = pos;
  double *restrict ar = area, *restrict nr = normal, *restrict ce = center, *restrict st = stats;
  const int ns = S, nvs = NVS, nts = NTS;

  GPU_OUTER()
  for (int s = 0; s < ns; ++s)
  {
    const double *restrict ps = p + 3L * s * nvs;
    const int *restrict ts = t3 + 3L * s * nts;
    double *restrict ars = ar + (long) s * nts, *restrict nrs = nr + 3L * s * nts, *restrict ces = ce + 3L * s * nts;
    const int nt = c[SP_N_COUNTS * s + SP_NT], nt_owned = c[SP_N_COUNTS * s + SP_NT_OWNED];

    double min_a = DBL_MAX, min_q = DBL_MAX, min_e = DBL_MAX;
    double max_a = -DBL_MAX, max_q = -DBL_MAX, max_e = -DBL_MAX;
    double sum_a = 0.0, sum_q = 0.0, sum_e = 0.0;

    GPU_INNER(reduction(min : min_a, min_q, min_e) reduction(max : max_a, max_q, max_e) reduction(+ : sum_a, sum_q, sum_e))
    for (int t = 0; t < nt; ++t)
    {
      const sp_tri_stats r = sp_triangle_geometry(b, ps, ts, t, ars, nrs, ces);
      if (t < nt_owned)
      {
        min_a = fmin(min_a, r.area);
        max_a = fmax(max_a, r.area);
        sum_a += r.area;
        min_q = fmin(min_q, r.q);
        max_q = fmax(max_q, r.q);
        sum_q += r.q;
        min_e = fmin(min_e, r.esq_min);
        max_e = fmax(max_e, r.esq_max);
        sum_e += r.esq_sum;
      }
    }

    double *restrict row = st + (long) SP_N_STATS * s;
    row[0] = min_a, row[1] = min_q, row[2] = min_e;
    row[3] = max_a, row[4] = max_q, row[5] = max_e;
    row[6] = sum_a, row[7] = sum_q, row[8] = sum_e;
  }
  GPU_UPDATE_HOST(st[0 : SP_N_STATS * ns])
  GPU_WAIT

  memcpy(out, stats, (size_t) (SP_N_STATS * S) * sizeof(double));
  bytes_d2h += 8.0 * SP_N_STATS * S;
  n_copies += 1;
}

void sp_normals(void)
{
  const sp_box *restrict b = box;
  const int *restrict c = cnt, *restrict t3 = tv;
  const double *restrict p = pos;
  double *restrict n = vn;
  const int ns = S, nvs = NVS, nts = NTS;

  GPU_OUTER()
  for (int s = 0; s < ns; ++s)
  {
    const double *restrict ps = p + 3L * s * nvs;
    const int *restrict ts = t3 + 3L * s * nts;
    double *restrict vns = n + 3L * s * nvs;
    const int nv = c[SP_N_COUNTS * s + SP_NV], nt = c[SP_N_COUNTS * s + SP_NT];

    GPU_INNER()
    for (int k = 0; k < 3 * nv; ++k) vns[k] = 0.0;

    GPU_INNER()
    for (int t = 0; t < nt; ++t)
    {
      double f[3];
      sp_face_normal(b, ps, ts, t, f);
      for (int corner = 0; corner < 3; ++corner)
      {
        const int v = ts[3 * t + corner];
        for (int k = 0; k < 3; ++k)
        {
          GPU_INNER_ATOMIC
          vns[3 * v + k] += f[k];
        }
      }
    }

    GPU_INNER()
    for (int v = 0; v < nv; ++v) sp_normalise_one(vns, v);
  }
  GPU_WAIT
}
