/*
 * @Author: Christian Gorges
 *
 * @brief LeVeque GPU example, the directive backend: ONE source for OpenACC,
 *        OpenMP offload, OpenMP threads and serial C.
 *
 * The loops are the host loops with one directive each, spelled through
 * gpu_pragmas.h, so the compiler flags alone choose the model:
 *
 *   nvc -acc=gpu / -acc=multicore      OpenACC on the GPU / on CPU cores
 *   nvc -mp=gpu  / -mp=multicore       OpenMP target on the GPU / on CPU cores
 *   gcc -fopenmp -DGPU_OMP_HOST_THREADS, gcc     OpenMP threads, serial
 *
 * The device copies are keyed by the library's own pool pointers and sized by
 * the pool CAPACITIES. They stay mapped across the remesh and are reallocated
 * only when a pool moved or grew; otherwise an attach is one `update device` of
 * the live prefix of each array -- the "mirror sized by capacity" of a hand-
 * written CUDA buffer, without the hand-written buffer.
 */
#include <math.h>

#include "gpu_pragmas.h"
#include "lv_backend.h"
#include "lv_kernels.h"

#ifndef GPU_BACKEND_NAME
#define GPU_BACKEND_NAME GPU_MODEL
#endif

/* 1 for the GPU builds: refuse to run when there is no GPU instead of quietly using the host. */
#ifndef GPU_REQUIRE_DEVICE
#define GPU_REQUIRE_DEVICE 0
#endif

static lv_mesh m;  // the attached mesh
static lv_mesh mapped;  // the pools the device copies belong to: host pointers and capacities
static int is_mapped = 0;
static long n_allocations = 0;

const char *lv_backend_name(void) { return GPU_BACKEND_NAME; }

int lv_init(int local_rank) { return gpu_select_device(local_rank, GPU_REQUIRE_DEVICE); }

long lv_device_allocations(void) { return n_allocations; }

static void map_pools(const lv_mesh *p)
{
  double *pos = p->pos, *vn = p->vnormal, *area = p->area, *normal = p->normal, *center = p->center;
  const int *tv = p->tri_verts, *nbr = p->tri_nbr, *rec = p->tri_rec;
  const int nvc = p->nv_capacity, ntc = p->nt_capacity;
  (void) pos, (void) vn, (void) area, (void) normal, (void) center, (void) tv, (void) nbr, (void) rec, (void) nvc, (void) ntc;

  GPU_ENTER_CREATE(pos [0:3 * nvc], vn [0:3 * nvc], tv [0:3 * ntc], nbr [0:3 * ntc], rec [0:3 * ntc], area [0:ntc], normal [0:3 * ntc], center [0:3 * ntc])
}

static void unmap_pools(const lv_mesh *p)
{
  double *pos = p->pos, *vn = p->vnormal, *area = p->area, *normal = p->normal, *center = p->center;
  const int *tv = p->tri_verts, *nbr = p->tri_nbr, *rec = p->tri_rec;
  const int nvc = p->nv_capacity, ntc = p->nt_capacity;
  (void) pos, (void) vn, (void) area, (void) normal, (void) center, (void) tv, (void) nbr, (void) rec, (void) nvc, (void) ntc;

  GPU_EXIT_DELETE(pos [0:3 * nvc], vn [0:3 * nvc], tv [0:3 * ntc], nbr [0:3 * ntc], rec [0:3 * ntc], area [0:ntc], normal [0:3 * ntc], center [0:3 * ntc])
}

static int same_pools(const lv_mesh *a, const lv_mesh *b)
{
  return a->pos == b->pos && a->vnormal == b->vnormal && a->tri_verts == b->tri_verts && a->tri_nbr == b->tri_nbr && a->tri_rec == b->tri_rec &&
         a->area == b->area && a->normal == b->normal && a->center == b->center && a->nv_capacity == b->nv_capacity && a->nt_capacity == b->nt_capacity;
}

void lv_finalize(void)
{
  if (is_mapped) unmap_pools(&mapped);
  is_mapped = 0;
}

void lv_attach(const lv_mesh *mesh)
{
  m = *mesh;

  /* The remesher may have freed a pool and allocated a larger one, possibly
   * overlapping the old range: the stale mapping goes first. */
  if (!is_mapped || !same_pools(&mapped, &m))
  {
    if (is_mapped) unmap_pools(&mapped);
    map_pools(&m);
    mapped = m;
    is_mapped = 1;
    ++n_allocations;
  }

  double *pos = m.pos;
  const int *tv = m.tri_verts, *nbr = m.tri_nbr, *rec = m.tri_rec;
  const int nv = m.nv, nt = m.nt;
  (void) pos, (void) tv, (void) nbr, (void) rec, (void) nv, (void) nt;
  GPU_UPDATE_DEVICE(pos [0:3 * nv], tv [0:3 * nt], nbr [0:3 * nt], rec [0:3 * nt])
  GPU_WAIT
}

void lv_detach(void) {}  // the mapping stays: lv_attach decides whether it still fits

void lv_advect(double t, double dt, double T)
{
  double *restrict pos = m.pos;
  const int n = m.nv_owned;

  GPU_LOOP()
  for (int v = 0; v < n; ++v) lv_rk4(pos + 3 * v, t, dt, T);
  GPU_WAIT
}

void lv_get_owned_positions(void)
{
  double *pos = m.pos;
  const int n = m.nv_owned;
  (void) pos, (void) n;
  GPU_UPDATE_HOST(pos [0:3 * n])
  GPU_WAIT
}

void lv_put_ghost_positions(void)
{
  double *pos = m.pos;
  const int first = 3 * m.nv_owned, count = 3 * (m.nv - m.nv_owned);
  (void) pos, (void) first;
  if (count > 0)
  {
    GPU_UPDATE_DEVICE(pos [first:count])
    GPU_WAIT
  }
}

void lv_triangle_geometry(void)
{
  const double *restrict pos = m.pos;
  const int *restrict tv = m.tri_verts;
  double *restrict area = m.area, *restrict normal = m.normal, *restrict center = m.center;
  const int nt = m.nt;

  GPU_LOOP()
  for (int t = 0; t < nt; ++t) lv_triangle_geometry_one(pos, tv, t, area, normal, center);
  GPU_WAIT
}

void lv_get_triangle_geometry(void)
{
  double *area = m.area, *normal = m.normal, *center = m.center;
  const int nt = m.nt;
  (void) area, (void) normal, (void) center, (void) nt;
  GPU_UPDATE_HOST(area [0:nt], normal [0:3 * nt], center [0:3 * nt])
  GPU_WAIT
}

void lv_measure(double min_len, double max_len, double vol_ref, lv_measure_out *out)
{
  const double *restrict pos = m.pos;
  double *restrict vn = m.vnormal;
  const int *restrict tv = m.tri_verts, *restrict nbr = m.tri_nbr, *restrict rec = m.tri_rec;
  const int nv = m.nv, nt = m.nt, nt_owned = m.nt_owned;

  GPU_LOOP()
  for (int i = 0; i < 3 * nv; ++i) vn[i] = 0.0;

  GPU_LOOP()
  for (int t = 0; t < nt; ++t)
  {
    double e1[3], e2[3], cr[3];
    lv_triangle_edges(pos, tv, t, e1, e2, cr);
    for (int c = 0; c < 3; ++c)
    {
      const int v = tv[3 * t + c];
      for (int k = 0; k < 3; ++k)
      {
        GPU_ATOMIC
        vn[3 * v + k] += cr[k];
      }
    }
  }

  GPU_LOOP()
  for (int v = 0; v < nv; ++v) lv_normalise_one(vn, v);

  /* One array, not six scalars: each reduction variable is its own copy back. */
  double sum[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, min_q = 1.0;
  GPU_LOOP(GPU_REDUCE_ARRAY(+, sum [0:6]) reduction(min : min_q))
  for (int t = 0; t < nt_owned; ++t)
  {
    const lv_tri_stats s = lv_triangle_stats(pos, vn, tv, nbr, rec, t, nt_owned, min_len, max_len, vol_ref);
    sum[0] += s.n_edge;
    sum[1] += s.n_out_of_band;
    sum[2] += s.area;
    sum[3] += s.q;
    sum[4] += s.inverted;
    sum[5] += s.volume;
    min_q = fmin(min_q, s.q);
  }
  GPU_WAIT

  out->n_edge = sum[0];
  out->n_out_of_band = sum[1];
  out->area = sum[2];
  out->q_sum = sum[3];
  out->n_inverted = sum[4];
  out->volume = sum[5];
  out->min_q = min_q;
}
