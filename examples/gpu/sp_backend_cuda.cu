/*
 * @Author: Christian Gorges
 *
 * @brief Many-spheres GPU example, CUDA backend.
 *
 * The same loop nests as sp_backend_directives.c, written out: one thread
 * block per body slot (blockIdx.x), its entities strided over the block's
 * threads -- the gang/vector mapping nvc chooses for the directives, made
 * explicit. What CUDA asks for on top of the shared sp_kernels.h:
 *
 *   - the allocations, and pinned staging buffers for the transfers;
 *   - the block reductions (warp shuffles + shared memory) that a directive
 *     writes as reduction(min : ...);
 *   - __syncthreads() between the phases of the normal kernel, which the
 *     directive models imply at the end of each inner loop.
 */
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "sp_backend.h"
#include "sp_kernels.h"

#define SP_CUDA_CHECK(call)                                                                                      \
  do                                                                                                             \
  {                                                                                                              \
    const cudaError_t err_ = (call);                                                                             \
    if (err_ != cudaSuccess)                                                                                     \
    {                                                                                                            \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err_)); \
      std::abort();                                                                                              \
    }                                                                                                            \
  } while (0)

namespace
{
constexpr int kBlock = 128;  // threads per slot, the vector length nvc picks for the directives
constexpr int kWarps = kBlock / 32;

template <typename T>
T *device_alloc(size_t n)
{
  void *p = nullptr;
  SP_CUDA_CHECK(cudaMalloc(&p, (n > 0 ? n : 1) * sizeof(T)));
  return static_cast<T *>(p);
}

/** @brief A grow-only buffer pair: pinned on the host, plain on the device. */
struct Staging
{
  void *host = nullptr, *dev = nullptr;
  size_t bytes = 0;

  void ensure(size_t need)
  {
    if (need <= bytes) return;
    if (host) SP_CUDA_CHECK(cudaFreeHost(host));
    if (dev) SP_CUDA_CHECK(cudaFree(dev));
    bytes = need + need / 2 + 4096;
    SP_CUDA_CHECK(cudaMallocHost(&host, bytes));
    SP_CUDA_CHECK(cudaMalloc(&dev, bytes));
  }
  void release()
  {
    if (host) SP_CUDA_CHECK(cudaFreeHost(host));
    if (dev) SP_CUDA_CHECK(cudaFree(dev));
    host = dev = nullptr;
    bytes = 0;
  }
};

struct State
{
  int S = 0, NVS = 0, NTS = 0;
  sp_box box = {};
  int *rank_map = nullptr;
  double *pos = nullptr, *vn = nullptr, *area = nullptr, *normal = nullptr, *center = nullptr, *stats = nullptr;
  int *tv = nullptr, *vown = nullptr, *town = nullptr, *cnt = nullptr, *changed = nullptr;
  Staging dstage, istage, segs;
  double h2d = 0.0, d2h = 0.0;
  long n_copies = 0;
} g;

void copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
  if (bytes == 0) return;
  SP_CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
  if (kind == cudaMemcpyHostToDevice)
    g.h2d += static_cast<double>(bytes);
  else
    g.d2h += static_cast<double>(bytes);
  ++g.n_copies;
}

/** @brief Every API call returns with its work done, so host timers measure it. */
void finish()
{
  SP_CUDA_CHECK(cudaGetLastError());
  SP_CUDA_CHECK(cudaDeviceSynchronize());
}

struct FieldRef
{
  void *data;
  int width, per_slot;
  bool is_int;
};

FieldRef field_of(sp_field f)
{
  switch (f)
  {
    case SP_POS: return {g.pos, 3, g.NVS, false};
    case SP_VNORMAL: return {g.vn, 3, g.NVS, false};
    case SP_AREA: return {g.area, 1, g.NTS, false};
    case SP_NORMAL: return {g.normal, 3, g.NTS, false};
    case SP_CENTER: return {g.center, 3, g.NTS, false};
    case SP_TRI_VERTS: return {g.tv, 3, g.NTS, true};
    case SP_VOWNER: return {g.vown, 1, g.NVS, true};
    case SP_TOWNER: return {g.town, 1, g.NTS, true};
    default: std::abort();
  }
}

// --- Kernels: one block per slot ---------------------------------------------

template <typename T>
__global__ void gather_kernel(const T *__restrict__ src, T *__restrict__ stage, const int *__restrict__ seg, int width, int per_slot)
{
  const int *sg = seg + 4 * blockIdx.x;
  const long from = (static_cast<long>(sg[0]) * per_slot + sg[1]) * width, to = static_cast<long>(sg[3]) * width;
  const int m = sg[2] * width;
  for (int k = threadIdx.x; k < m; k += blockDim.x) stage[to + k] = src[from + k];
}

template <typename T>
__global__ void scatter_kernel(T *__restrict__ dst, const T *__restrict__ stage, const int *__restrict__ seg, int width, int per_slot)
{
  const int *sg = seg + 4 * blockIdx.x;
  const long to = (static_cast<long>(sg[0]) * per_slot + sg[1]) * width, from = static_cast<long>(sg[3]) * width;
  const int m = sg[2] * width;
  for (int k = threadIdx.x; k < m; k += blockDim.x) dst[to + k] = stage[from + k];
}

__global__ void advect_kernel(__grid_constant__ const sp_box box, double *__restrict__ pos, const int *__restrict__ cnt, int nvs, double vx, double vy,
                              double vz, double dt)
{
  const int s = blockIdx.x;
  double *ps = pos + 3L * s * nvs;
  const int n = cnt[SP_N_COUNTS * s + SP_NV_OWNED];
  for (int v = threadIdx.x; v < n; v += blockDim.x) sp_advect_one(&box, ps + 3 * v, vx, vy, vz, dt);
}

__global__ void classify_kernel(__grid_constant__ const sp_box box, const int *__restrict__ rank_map, const double *__restrict__ pos,
                                const int *__restrict__ tv, const int *__restrict__ cnt, int nvs, int nts, int *__restrict__ vown, int *__restrict__ town,
                                int *__restrict__ changed)
{
  const int s = blockIdx.x;
  const double *ps = pos + 3L * s * nvs;
  const int *ts = tv + 3L * s * nts;
  int *vos = vown + static_cast<long>(s) * nvs, *tos = town + static_cast<long>(s) * nts;
  const int nv = cnt[SP_N_COUNTS * s + SP_NV], nt = cnt[SP_N_COUNTS * s + SP_NT];

  int any = 0;
  for (int v = threadIdx.x; v < nv; v += blockDim.x)
  {
    const int o = sp_owner(&box, rank_map, ps + 3 * v);
    any |= (o != vos[v]);
    vos[v] = o;
  }
  for (int t = threadIdx.x; t < nt; t += blockDim.x)
  {
    double c[3];
    sp_triangle_center(&box, ps, ts, t, c);
    const int o = sp_owner(&box, rank_map, c);
    any |= (o != tos[t]);
    tos[t] = o;
  }
  any = __syncthreads_or(any);
  if (threadIdx.x == 0) changed[s] = any;
}

__device__ double warp_min(double x)
{
  for (int o = 16; o > 0; o >>= 1) x = fmin(x, __shfl_down_sync(0xffffffffu, x, o));
  return x;
}
__device__ double warp_max(double x)
{
  for (int o = 16; o > 0; o >>= 1) x = fmax(x, __shfl_down_sync(0xffffffffu, x, o));
  return x;
}
__device__ double warp_sum(double x)
{
  for (int o = 16; o > 0; o >>= 1) x += __shfl_down_sync(0xffffffffu, x, o);
  return x;
}

/** @brief Reduces v[0..2] by min, v[3..5] by max, v[6..8] by sum over the block; thread 0 holds the result. */
__device__ void block_stats(double v[SP_N_STATS])
{
  __shared__ double partial[SP_N_STATS][kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

  for (int k = 0; k < 3; ++k) v[k] = warp_min(v[k]);
  for (int k = 3; k < 6; ++k) v[k] = warp_max(v[k]);
  for (int k = 6; k < 9; ++k) v[k] = warp_sum(v[k]);
  if (lane == 0)
    for (int k = 0; k < SP_N_STATS; ++k) partial[k][warp] = v[k];
  __syncthreads();

  if (warp != 0) return;
  for (int k = 0; k < 3; ++k) v[k] = warp_min(lane < kWarps ? partial[k][lane] : DBL_MAX);
  for (int k = 3; k < 6; ++k) v[k] = warp_max(lane < kWarps ? partial[k][lane] : -DBL_MAX);
  for (int k = 6; k < 9; ++k) v[k] = warp_sum(lane < kWarps ? partial[k][lane] : 0.0);
}

__global__ void geometry_kernel(__grid_constant__ const sp_box box, const double *__restrict__ pos, const int *__restrict__ tv, const int *__restrict__ cnt,
                                int nvs, int nts, double *__restrict__ area, double *__restrict__ normal, double *__restrict__ center,
                                double *__restrict__ stats)
{
  const int s = blockIdx.x;
  const double *ps = pos + 3L * s * nvs;
  const int *ts = tv + 3L * s * nts;
  double *ars = area + static_cast<long>(s) * nts, *nrs = normal + 3L * s * nts, *ces = center + 3L * s * nts;
  const int nt = cnt[SP_N_COUNTS * s + SP_NT], nt_owned = cnt[SP_N_COUNTS * s + SP_NT_OWNED];

  double v[SP_N_STATS] = {DBL_MAX, DBL_MAX, DBL_MAX, -DBL_MAX, -DBL_MAX, -DBL_MAX, 0.0, 0.0, 0.0};
  for (int t = threadIdx.x; t < nt; t += blockDim.x)
  {
    const sp_tri_stats r = sp_triangle_geometry(&box, ps, ts, t, ars, nrs, ces);
    if (t < nt_owned)
    {
      v[0] = fmin(v[0], r.area), v[1] = fmin(v[1], r.q), v[2] = fmin(v[2], r.esq_min);
      v[3] = fmax(v[3], r.area), v[4] = fmax(v[4], r.q), v[5] = fmax(v[5], r.esq_max);
      v[6] += r.area, v[7] += r.q, v[8] += r.esq_sum;
    }
  }
  block_stats(v);
  if (threadIdx.x == 0)
    for (int k = 0; k < SP_N_STATS; ++k) stats[SP_N_STATS * s + k] = v[k];
}

__global__ void normals_kernel(__grid_constant__ const sp_box box, const double *__restrict__ pos, const int *__restrict__ tv, const int *__restrict__ cnt,
                               int nvs, int nts, double *__restrict__ vn)
{
  const int s = blockIdx.x;
  const double *ps = pos + 3L * s * nvs;
  const int *ts = tv + 3L * s * nts;
  double *vns = vn + 3L * s * nvs;
  const int nv = cnt[SP_N_COUNTS * s + SP_NV], nt = cnt[SP_N_COUNTS * s + SP_NT];

  for (int k = threadIdx.x; k < 3 * nv; k += blockDim.x) vns[k] = 0.0;
  __syncthreads();

  for (int t = threadIdx.x; t < nt; t += blockDim.x)
  {
    double f[3];
    sp_face_normal(&box, ps, ts, t, f);
    for (int corner = 0; corner < 3; ++corner)
    {
      const int v = ts[3 * t + corner];
      for (int k = 0; k < 3; ++k) atomicAdd(&vns[3 * v + k], f[k]);
    }
  }
  __syncthreads();

  for (int v = threadIdx.x; v < nv; v += blockDim.x) sp_normalise_one(vns, v);
}

/** @brief Uploads the segment table; returns the staged entries it covers (entities x width). */
long put_segments(int n_seg, const sp_segment *seg, int width)
{
  g.segs.ensure(4 * sizeof(int) * static_cast<size_t>(n_seg));
  int *sg = static_cast<int *>(g.segs.host);
  long total = 0;
  for (int i = 0; i < n_seg; ++i)
  {
    sg[4 * i + 0] = seg[i].slot;
    sg[4 * i + 1] = seg[i].first;
    sg[4 * i + 2] = seg[i].count;
    sg[4 * i + 3] = seg[i].offset;
    if (seg[i].offset + seg[i].count > total) total = seg[i].offset + seg[i].count;
  }
  copy(g.segs.dev, g.segs.host, 4 * sizeof(int) * static_cast<size_t>(n_seg), cudaMemcpyHostToDevice);
  return total * width;
}

}  // namespace

// --- The API ----------------------------------------------------------------

extern "C" const char *sp_backend_name(void) { return "cuda"; }

extern "C" int sp_init(int local_rank)
{
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;
  SP_CUDA_CHECK(cudaSetDevice(local_rank % n));
  SP_CUDA_CHECK(cudaFree(nullptr));  // create the context now, not inside the first timed call
  return 0;
}

extern "C" void sp_setup(int n_slots, int nv_slot, int nt_slot, const sp_box *box, const int *rank_map, int nproc)
{
  g.S = n_slots;
  g.NVS = nv_slot;
  g.NTS = nt_slot;
  g.box = *box;
  const size_t nv = static_cast<size_t>(n_slots) * nv_slot, nt = static_cast<size_t>(n_slots) * nt_slot, ns = n_slots;

  g.rank_map = device_alloc<int>(nproc);
  g.pos = device_alloc<double>(3 * nv);
  g.vn = device_alloc<double>(3 * nv);
  g.area = device_alloc<double>(nt);
  g.normal = device_alloc<double>(3 * nt);
  g.center = device_alloc<double>(3 * nt);
  g.tv = device_alloc<int>(3 * nt);
  g.vown = device_alloc<int>(nv);
  g.town = device_alloc<int>(nt);
  g.cnt = device_alloc<int>(SP_N_COUNTS * ns);
  g.changed = device_alloc<int>(ns);
  g.stats = device_alloc<double>(SP_N_STATS * ns);

  copy(g.rank_map, rank_map, nproc * sizeof(int), cudaMemcpyHostToDevice);
  SP_CUDA_CHECK(cudaMemset(g.cnt, 0, SP_N_COUNTS * ns * sizeof(int)));
}

extern "C" double sp_device_bytes(void)
{
  const double nv = static_cast<double>(g.S) * g.NVS, nt = static_cast<double>(g.S) * g.NTS;
  return 8.0 * (6.0 * nv + 7.0 * nt) + 4.0 * (nv + 4.0 * nt) + (4.0 * (SP_N_COUNTS + 1) + 8.0 * SP_N_STATS) * g.S;
}

extern "C" void sp_finalize(void)
{
  void *all[] = {g.rank_map, g.pos, g.vn, g.area, g.normal, g.center, g.tv, g.vown, g.town, g.cnt, g.changed, g.stats};
  for (void *p : all)
    if (p) SP_CUDA_CHECK(cudaFree(p));
  g.dstage.release();
  g.istage.release();
  g.segs.release();
  g = State();
}

extern "C" double *sp_stage_double(long n)
{
  g.dstage.ensure(static_cast<size_t>(n) * sizeof(double));
  return static_cast<double *>(g.dstage.host);
}

extern "C" int *sp_stage_int(long n)
{
  g.istage.ensure(static_cast<size_t>(n) * sizeof(int));
  return static_cast<int *>(g.istage.host);
}

extern "C" void sp_upload(sp_field field, int n_seg, const sp_segment *seg)
{
  if (n_seg <= 0) return;
  const FieldRef f = field_of(field);
  const long n = put_segments(n_seg, seg, f.width);
  const int *sg = static_cast<const int *>(g.segs.dev);

  if (f.is_int)
  {
    copy(g.istage.dev, g.istage.host, n * sizeof(int), cudaMemcpyHostToDevice);
    scatter_kernel<int><<<n_seg, kBlock>>>(static_cast<int *>(f.data), static_cast<const int *>(g.istage.dev), sg, f.width, f.per_slot);
  }
  else
  {
    copy(g.dstage.dev, g.dstage.host, n * sizeof(double), cudaMemcpyHostToDevice);
    scatter_kernel<double><<<n_seg, kBlock>>>(static_cast<double *>(f.data), static_cast<const double *>(g.dstage.dev), sg, f.width, f.per_slot);
  }
  finish();
}

extern "C" const void *sp_download(sp_field field, int n_seg, const sp_segment *seg)
{
  if (n_seg <= 0) return nullptr;
  const FieldRef f = field_of(field);
  long total = 0;
  for (int i = 0; i < n_seg; ++i)
    if (seg[i].offset + seg[i].count > total) total = seg[i].offset + seg[i].count;
  const long n = total * f.width;
  if (f.is_int)
    sp_stage_int(n);
  else
    sp_stage_double(n);
  put_segments(n_seg, seg, f.width);
  const int *sg = static_cast<const int *>(g.segs.dev);

  if (f.is_int)
  {
    gather_kernel<int><<<n_seg, kBlock>>>(static_cast<const int *>(f.data), static_cast<int *>(g.istage.dev), sg, f.width, f.per_slot);
    copy(g.istage.host, g.istage.dev, n * sizeof(int), cudaMemcpyDeviceToHost);
    finish();
    return g.istage.host;
  }
  gather_kernel<double><<<n_seg, kBlock>>>(static_cast<const double *>(f.data), static_cast<double *>(g.dstage.dev), sg, f.width, f.per_slot);
  copy(g.dstage.host, g.dstage.dev, n * sizeof(double), cudaMemcpyDeviceToHost);
  finish();
  return g.dstage.host;
}

extern "C" void sp_set_counts(const int *counts) { copy(g.cnt, counts, SP_N_COUNTS * sizeof(int) * static_cast<size_t>(g.S), cudaMemcpyHostToDevice); }

extern "C" void sp_traffic(double *h2d, double *d2h, long *n_copies)
{
  *h2d = g.h2d;
  *d2h = g.d2h;
  *n_copies = g.n_copies;
}

extern "C" void sp_advect(double vx, double vy, double vz, double dt)
{
  advect_kernel<<<g.S, kBlock>>>(g.box, g.pos, g.cnt, g.NVS, vx, vy, vz, dt);
  finish();
}

extern "C" void sp_classify(int *changed)
{
  classify_kernel<<<g.S, kBlock>>>(g.box, g.rank_map, g.pos, g.tv, g.cnt, g.NVS, g.NTS, g.vown, g.town, g.changed);
  copy(changed, g.changed, sizeof(int) * static_cast<size_t>(g.S), cudaMemcpyDeviceToHost);
  finish();
}

extern "C" void sp_geometry(double *stats)
{
  geometry_kernel<<<g.S, kBlock>>>(g.box, g.pos, g.tv, g.cnt, g.NVS, g.NTS, g.area, g.normal, g.center, g.stats);
  copy(stats, g.stats, SP_N_STATS * sizeof(double) * static_cast<size_t>(g.S), cudaMemcpyDeviceToHost);
  finish();
}

extern "C" void sp_normals(void)
{
  normals_kernel<<<g.S, kBlock>>>(g.box, g.pos, g.tv, g.cnt, g.NVS, g.NTS, g.vn);
  finish();
}
