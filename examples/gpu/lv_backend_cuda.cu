/*
 * @Author: Christian Gorges
 *
 * @brief LeVeque GPU example, CUDA backend.
 *
 * Explicit device buffers, one kernel per loop, a hand-written block reduction.
 * The per-element maths is lv_kernels.h, shared with the directive backend;
 * everything in this file is what CUDA asks for on top of it.
 *
 * The device buffers mirror the host pools: sized by the pool CAPACITY and
 * reallocated only when it grows, so an attach is normally a plain copy -- the
 * same rule the directive backend follows with its persistent mapping.
 */
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "lv_backend.h"
#include "lv_kernels.h"

#define LV_CUDA_CHECK(call)                                                                                      \
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
constexpr int kBlock = 256;
constexpr int kWarps = kBlock / 32;

bool g_grew = false;  // some buffer was reallocated during the current attach

/** @brief A grow-only device allocation: the mesh changes size every step, cudaMalloc should not. */
struct DeviceBuffer
{
  void *ptr = nullptr;
  size_t capacity = 0;

  template <typename T>
  T *ensure(size_t count)
  {
    const size_t bytes = count * sizeof(T);
    if (bytes > capacity)
    {
      if (ptr) LV_CUDA_CHECK(cudaFree(ptr));
      capacity = bytes;
      LV_CUDA_CHECK(cudaMalloc(&ptr, capacity));
      g_grew = true;
    }
    return static_cast<T *>(ptr);
  }
  void release()
  {
    if (ptr) LV_CUDA_CHECK(cudaFree(ptr));
    ptr = nullptr;
    capacity = 0;
  }
};

struct State
{
  lv_mesh host = {};  // the host pools of the attached mesh
  DeviceBuffer pos, vnormal, tri_verts, tri_nbr, tri_rec, area, normal, center, reduction;
  int max_blocks = 1;
  long n_allocations = 0;
} g;

int blocks_for(int n) { return n > 0 ? (n + kBlock - 1) / kBlock : 1; }

/** @brief cudaMemcpy plus the error check, for the six transfers below. */
void copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind)
{
  if (bytes > 0) LV_CUDA_CHECK(cudaMemcpy(dst, src, bytes, kind));
}

/** @brief Every API call returns with its work done, so host timers measure it. */
void finish()
{
  LV_CUDA_CHECK(cudaGetLastError());
  LV_CUDA_CHECK(cudaDeviceSynchronize());
}

// --- Kernels ----------------------------------------------------------------

__global__ void advect_kernel(double *__restrict__ pos, int n, double t, double dt, double T)
{
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) lv_rk4(pos + 3 * v, t, dt, T);
}

__global__ void geometry_kernel(const double *__restrict__ pos, const int *__restrict__ tri_verts, int nt, double *__restrict__ area,
                                double *__restrict__ normal, double *__restrict__ center)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t < nt) lv_triangle_geometry_one(pos, tri_verts, t, area, normal, center);
}

__global__ void scatter_normals_kernel(const double *__restrict__ pos, const int *__restrict__ tri_verts, int nt, double *__restrict__ vn)
{
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nt) return;

  double e1[3], e2[3], cr[3];
  lv_triangle_edges(pos, tri_verts, t, e1, e2, cr);
  for (int c = 0; c < 3; ++c)
  {
    const int v = tri_verts[3 * t + c];
    for (int k = 0; k < 3; ++k) atomicAdd(&vn[3 * v + k], cr[k]);
  }
}

__global__ void normalise_kernel(double *__restrict__ vn, int nv)
{
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < nv) lv_normalise_one(vn, v);
}

__device__ double warp_sum(double x)
{
  for (int offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffffu, x, offset);
  return x;
}

__device__ double warp_min(double x)
{
  for (int offset = 16; offset > 0; offset >>= 1) x = fmin(x, __shfl_down_sync(0xffffffffu, x, offset));
  return x;
}

/**
 * @brief Six sums and one minimum over the owned triangles.
 *
 * Grid-stride loop into registers, warp shuffles, one shared-memory pass over
 * the warps, then ONE atomic per block and quantity into out[7]. The minimum
 * goes through atomicMin on the bit pattern, which orders like the value for
 * non-negative doubles (a quality is never negative).
 */
__global__ void stats_kernel(const double *__restrict__ pos, const double *__restrict__ vn, const int *__restrict__ tri_verts, const int *__restrict__ tri_nbr,
                             const int *__restrict__ tri_rec, int nt_owned, double min_len, double max_len, double vol_ref, double *__restrict__ out)
{
  double acc[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  double min_q = 1.0;

  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < nt_owned; t += gridDim.x * blockDim.x)
  {
    const lv_tri_stats s = lv_triangle_stats(pos, vn, tri_verts, tri_nbr, tri_rec, t, nt_owned, min_len, max_len, vol_ref);
    acc[0] += s.n_edge;
    acc[1] += s.n_out_of_band;
    acc[2] += s.area;
    acc[3] += s.q;
    acc[4] += s.inverted;
    acc[5] += s.volume;
    min_q = fmin(min_q, s.q);
  }

  __shared__ double partial[7][kWarps];
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;

  for (int i = 0; i < 6; ++i) acc[i] = warp_sum(acc[i]);
  min_q = warp_min(min_q);
  if (lane == 0)
  {
    for (int i = 0; i < 6; ++i) partial[i][warp] = acc[i];
    partial[6][warp] = min_q;
  }
  __syncthreads();

  if (warp != 0) return;
  for (int i = 0; i < 6; ++i) acc[i] = warp_sum(lane < kWarps ? partial[i][lane] : 0.0);
  min_q = warp_min(lane < kWarps ? partial[6][lane] : 1.0);
  if (lane == 0)
  {
    for (int i = 0; i < 6; ++i) atomicAdd(&out[i], acc[i]);
    atomicMin(reinterpret_cast<unsigned long long *>(&out[6]), static_cast<unsigned long long>(__double_as_longlong(min_q)));
  }
}

}  // namespace

// --- The API ----------------------------------------------------------------

extern "C" const char *lv_backend_name(void) { return "cuda"; }

extern "C" int lv_init(int local_rank)
{
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 1;
  LV_CUDA_CHECK(cudaSetDevice(local_rank % n));
  LV_CUDA_CHECK(cudaFree(nullptr));  // create the context now, not inside the first timed call

  int device = 0, n_sm = 1;
  LV_CUDA_CHECK(cudaGetDevice(&device));
  LV_CUDA_CHECK(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device));
  g.max_blocks = 8 * n_sm;
  return 0;
}

extern "C" void lv_finalize(void)
{
  DeviceBuffer *all[] = {&g.pos, &g.vnormal, &g.tri_verts, &g.tri_nbr, &g.tri_rec, &g.area, &g.normal, &g.center, &g.reduction};
  for (DeviceBuffer *b : all) b->release();
}

extern "C" void lv_attach(const lv_mesh *m)
{
  g.host = *m;
  const size_t nv = static_cast<size_t>(m->nv), nt = static_cast<size_t>(m->nt);
  const size_t nvc = static_cast<size_t>(m->nv_capacity), ntc = static_cast<size_t>(m->nt_capacity);

  g_grew = false;
  copy(g.pos.ensure<double>(3 * nvc), m->pos, 3 * nv * sizeof(double), cudaMemcpyHostToDevice);
  copy(g.tri_verts.ensure<int>(3 * ntc), m->tri_verts, 3 * nt * sizeof(int), cudaMemcpyHostToDevice);
  copy(g.tri_nbr.ensure<int>(3 * ntc), m->tri_nbr, 3 * nt * sizeof(int), cudaMemcpyHostToDevice);
  copy(g.tri_rec.ensure<int>(3 * ntc), m->tri_rec, 3 * nt * sizeof(int), cudaMemcpyHostToDevice);
  g.vnormal.ensure<double>(3 * nvc);
  g.area.ensure<double>(ntc);
  g.normal.ensure<double>(3 * ntc);
  g.center.ensure<double>(3 * ntc);
  g.reduction.ensure<double>(7);
  if (g_grew) ++g.n_allocations;
  finish();
}

extern "C" long lv_device_allocations(void) { return g.n_allocations; }

extern "C" void lv_detach(void) { g.host = lv_mesh{}; }  // the buffers stay, to be reused at the next attach

extern "C" void lv_advect(double t, double dt, double T)
{
  const int n = g.host.nv_owned;
  advect_kernel<<<blocks_for(n), kBlock>>>(static_cast<double *>(g.pos.ptr), n, t, dt, T);
  finish();
}

extern "C" void lv_get_owned_positions(void) { copy(g.host.pos, g.pos.ptr, 3 * static_cast<size_t>(g.host.nv_owned) * sizeof(double), cudaMemcpyDeviceToHost); }

extern "C" void lv_put_ghost_positions(void)
{
  const size_t first = 3 * static_cast<size_t>(g.host.nv_owned), count = 3 * static_cast<size_t>(g.host.nv - g.host.nv_owned);
  copy(static_cast<double *>(g.pos.ptr) + first, g.host.pos + first, count * sizeof(double), cudaMemcpyHostToDevice);
  finish();
}

extern "C" void lv_triangle_geometry(void)
{
  const int nt = g.host.nt;
  geometry_kernel<<<blocks_for(nt), kBlock>>>(static_cast<const double *>(g.pos.ptr), static_cast<const int *>(g.tri_verts.ptr), nt,
                                              static_cast<double *>(g.area.ptr), static_cast<double *>(g.normal.ptr), static_cast<double *>(g.center.ptr));
  finish();
}

extern "C" void lv_get_triangle_geometry(void)
{
  const size_t nt = static_cast<size_t>(g.host.nt);
  copy(g.host.area, g.area.ptr, nt * sizeof(double), cudaMemcpyDeviceToHost);
  copy(g.host.normal, g.normal.ptr, 3 * nt * sizeof(double), cudaMemcpyDeviceToHost);
  copy(g.host.center, g.center.ptr, 3 * nt * sizeof(double), cudaMemcpyDeviceToHost);
}

extern "C" void lv_measure(double min_len, double max_len, double vol_ref, lv_measure_out *out)
{
  const int nv = g.host.nv, nt = g.host.nt, nt_owned = g.host.nt_owned;
  const double *pos = static_cast<const double *>(g.pos.ptr);
  const int *tri_verts = static_cast<const int *>(g.tri_verts.ptr);
  double *vn = static_cast<double *>(g.vnormal.ptr);
  double *red = static_cast<double *>(g.reduction.ptr);

  LV_CUDA_CHECK(cudaMemsetAsync(vn, 0, 3 * static_cast<size_t>(nv) * sizeof(double)));
  scatter_normals_kernel<<<blocks_for(nt), kBlock>>>(pos, tri_verts, nt, vn);
  normalise_kernel<<<blocks_for(nv), kBlock>>>(vn, nv);

  const double init[7] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0};
  copy(red, init, sizeof init, cudaMemcpyHostToDevice);
  const int blocks = blocks_for(nt_owned) < g.max_blocks ? blocks_for(nt_owned) : g.max_blocks;
  stats_kernel<<<blocks, kBlock>>>(pos, vn, tri_verts, static_cast<const int *>(g.tri_nbr.ptr), static_cast<const int *>(g.tri_rec.ptr), nt_owned, min_len,
                                   max_len, vol_ref, red);
  LV_CUDA_CHECK(cudaGetLastError());

  double r[7];
  copy(r, red, sizeof r, cudaMemcpyDeviceToHost);
  out->n_edge = r[0];
  out->n_out_of_band = r[1];
  out->area = r[2];
  out->q_sum = r[3];
  out->n_inverted = r[4];
  out->volume = r[5];
  out->min_q = r[6];
}
