/*
 * @Author: Christian Gorges
 *
 * @brief The per-element mathematics of the LeVeque GPU example, written ONCE
 *        and compiled by every backend: by gcc for the host reference, by nvcc
 *        for CUDA, by nvc for OpenACC and OpenMP.
 *
 * Each function works on one vertex or one triangle and knows nothing about
 * threads, so the backends differ only in how they loop over elements, reduce
 * and move data. The formulas are those of the library (and of
 * leveque_remeshing_parallel.c), so the device results agree with the host ones
 * to round-off -- not to the bit: nvcc and nvc contract a*b+c into FMAs, and the
 * device sin/cos are not glibc's.
 *
 * The three programming models each want their own annotation on a function
 * that runs on the device; gpu_pragmas.h spells all three, and they expand to
 * nothing for gcc:
 *
 *   CUDA      __host__ __device__ on every function           (GPU_FN)
 *   OpenACC   #pragma acc routine seq before every function   (GPU_ROUTINE)
 *   OpenMP    one declare target region around them all       (GPU_BEGIN/END_DEVICE)
 *
 * Pointer parameters are GPU_RESTRICT: the arrays never alias, and saying so
 * lets nvcc route the read-only ones through the non-coherent load path.
 */
#ifndef LV_KERNELS_H
#define LV_KERNELS_H

#include <math.h>

#include "gpu_pragmas.h"

#define LV_PI 3.14159265358979323846
#define LV_FOUR_SQRT3 6.9282032302755088
#define LV_ONE_THIRD 0.33333333333333333

GPU_BEGIN_DEVICE

/** @brief The time-reversed LeVeque (1996) deformation field of period T. */
GPU_ROUTINE GPU_FN void lv_velocity(const double x[3], double t, double T, double u[3])
{
  const double s = cos(LV_PI * t / T);
  const double sx = sin(LV_PI * x[0]), sy = sin(LV_PI * x[1]), sz = sin(LV_PI * x[2]);
  const double s2x = sin(2.0 * LV_PI * x[0]), s2y = sin(2.0 * LV_PI * x[1]), s2z = sin(2.0 * LV_PI * x[2]);

  u[0] = 2.0 * sx * sx * s2y * s2z * s;
  u[1] = -sy * sy * s2x * s2z * s;
  u[2] = -sz * sz * s2x * s2y * s;
}

/** @brief One classical RK4 step of the point @p p, in place. */
GPU_ROUTINE GPU_FN void lv_rk4(double p[3], double t, double dt, double T)
{
  double k1[3], k2[3], k3[3], k4[3], x[3];

  lv_velocity(p, t, T, k1);
  for (int k = 0; k < 3; ++k) x[k] = p[k] + 0.5 * dt * k1[k];
  lv_velocity(x, t + 0.5 * dt, T, k2);
  for (int k = 0; k < 3; ++k) x[k] = p[k] + 0.5 * dt * k2[k];
  lv_velocity(x, t + 0.5 * dt, T, k3);
  for (int k = 0; k < 3; ++k) x[k] = p[k] + dt * k3[k];
  lv_velocity(x, t + dt, T, k4);

  for (int k = 0; k < 3; ++k) p[k] += (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
}

/** @brief e1 = p1 - p0, e2 = p2 - p0 and cr = e1 x e2 (twice the area times the unit normal). */
GPU_ROUTINE GPU_FN void lv_triangle_edges(const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tri_verts, int t, double e1[3], double e2[3], double cr[3])
{
  const double *p0 = pos + 3 * tri_verts[3 * t + 0];
  const double *p1 = pos + 3 * tri_verts[3 * t + 1];
  const double *p2 = pos + 3 * tri_verts[3 * t + 2];

  for (int k = 0; k < 3; ++k)
  {
    e1[k] = p1[k] - p0[k];
    e2[k] = p2[k] - p0[k];
  }
  cr[0] = e1[1] * e2[2] - e1[2] * e2[1];
  cr[1] = e1[2] * e2[0] - e1[0] * e2[2];
  cr[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

/** @brief MFTLComputeTriangleGeometry() for one triangle (non-periodic domain). */
GPU_ROUTINE GPU_FN void lv_triangle_geometry_one(const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tri_verts, int t, double *GPU_RESTRICT area,
                                                double *GPU_RESTRICT normal, double *GPU_RESTRICT center)
{
  double e1[3], e2[3], cr[3];
  lv_triangle_edges(pos, tri_verts, t, e1, e2, cr);

  const double mag = sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
  const double *p0 = pos + 3 * tri_verts[3 * t];

  area[t] = 0.5 * mag;
  for (int k = 0; k < 3; ++k)
  {
    normal[3 * t + k] = cr[k] * inv;
    center[3 * t + k] = p0[k] + (e1[k] + e2[k]) * LV_ONE_THIRD;
  }
}

/** @brief Scales vertex normal @p v to unit length; a zero sum stays zero. */
GPU_ROUTINE GPU_FN void lv_normalise_one(double *vn, int v)
{
  const double x = vn[3 * v + 0], y = vn[3 * v + 1], z = vn[3 * v + 2];
  const double len_sq = x * x + y * y + z * z;
  const double inv = (len_sq > 0.0) ? 1.0 / sqrt(len_sq) : 0.0;

  vn[3 * v + 0] = x * inv;
  vn[3 * v + 1] = y * inv;
  vn[3 * v + 2] = z * inv;
}

/** @brief What one owned triangle contributes to the diagnostics. */
typedef struct lv_tri_stats
{
  double n_edge, n_out_of_band, area, q, inverted, volume;
} lv_tri_stats;

/**
 * @brief The diagnostics of owned triangle @p t: the body of measure() in
 *        leveque_remeshing_parallel.c.
 *
 * An edge is counted by exactly one triangle mesh-wide: always at a boundary,
 * by the lower index when both sides are owned here, and by the lower GLOBAL id
 * when the other side is a ghost (whose owner applies the same rule).
 */
GPU_ROUTINE GPU_FN lv_tri_stats lv_triangle_stats(const double *GPU_RESTRICT pos, const double *GPU_RESTRICT vn, const int *GPU_RESTRICT tri_verts,
                                                  const int *GPU_RESTRICT tri_nbr, const int *GPU_RESTRICT tri_rec, int t,
                                                int nt_owned, double min_len, double max_len, double vol_ref)
{
  lv_tri_stats s = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  const int v[3] = {tri_verts[3 * t + 0], tri_verts[3 * t + 1], tri_verts[3 * t + 2]};

  double sum_sq = 0.0;
  for (int j = 0; j < 3; ++j)
  {
    const double *a = pos + 3 * v[j];
    const double *b = pos + 3 * v[(j + 1) % 3];
    const double dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
    const double len_sq = dx * dx + dy * dy + dz * dz;
    sum_sq += len_sq;

    const int nb = tri_nbr[3 * t + j];
    const int counts = (nb < 0) || ((nb < nt_owned) ? (t < nb) : (tri_rec[3 * t] < tri_rec[3 * nb]));
    if (counts)
    {
      const double len = sqrt(len_sq);
      s.n_edge += 1.0;
      if (len > max_len || len < min_len) s.n_out_of_band += 1.0;
    }
  }

  double e1[3], e2[3], cr[3];
  lv_triangle_edges(pos, tri_verts, t, e1, e2, cr);
  const double mag = sqrt(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);
  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;

  s.area = 0.5 * mag;
  s.q = (sum_sq > 0.0) ? LV_FOUR_SQRT3 * s.area / sum_sq : 0.0;

  double dot = 0.0;
  for (int k = 0; k < 3; ++k) dot += cr[k] * inv * (vn[3 * v[0] + k] + vn[3 * v[1] + k] + vn[3 * v[2] + k]);
  if (dot <= 0.0) s.inverted = 1.0;

  const double x = (pos[3 * v[0]] + pos[3 * v[1]] + pos[3 * v[2]]) / 3.0;
  s.volume = (x - vol_ref) * 0.5 * cr[0];
  return s;
}

GPU_END_DEVICE

#endif
