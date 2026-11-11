/*
 * @Author: Christian Gorges
 *
 * @brief The per-entity mathematics of the many-spheres GPU example, written
 *        ONCE for every backend (annotations as in lv_kernels.h).
 *
 * The formulas are the library's, in the library's order, so the device
 * results agree with the host ones to round-off:
 *
 *   sp_advect_one        the advection loop of parallel_advection_redistribute_list.c
 *                        plus MFTLApplyPeriodicity()
 *   sp_owner             MFTLGetOwnerProc()
 *   sp_triangle_*        MFTL_FOREACH_TRIANGLE_GEOMETRY with the periodic minimum
 *                        image, and the body of update_kernel() (ensemble_update.c)
 *   sp_face_normal,      MFTLComputeVertexNormals(MFTL_NORMAL_WEIGHT_AREA)
 *   sp_normalise_one
 *
 * Every pointer is relative to ONE slot: `pos` is the slot's first vertex and
 * `tv` its first triangle, whose corners are slot-local vertex indices.
 */
#ifndef SP_KERNELS_H
#define SP_KERNELS_H

#include <math.h>

#include "gpu_pragmas.h"
#include "sp_backend.h"

#define SP_FOUR_SQRT3 6.9282032302755088  // MFTL_FOUR_SQRT3
#define SP_ONE_THIRD 0.33333333333333333  // MFTL_ONE_THIRD

/** @brief Per-triangle contribution to the slot statistics. */
typedef struct sp_tri_stats
{
  double area, q, esq_min, esq_max, esq_sum;
} sp_tri_stats;

GPU_BEGIN_DEVICE

/** @brief MFTLApplyPeriodicity(): one shift into [min, max) on each periodic axis. */
GPU_ROUTINE GPU_FN void sp_wrap(const sp_box *GPU_RESTRICT box, double p[3])
{
  for (int k = 0; k < 3; ++k)
    if (box->periodic[k])
    {
      if (p[k] >= box->max[k])
        p[k] -= box->L[k];
      else if (p[k] < box->min[k])
        p[k] += box->L[k];
    }
}

/** @brief One explicit Euler step of a rigid translation, then the wrap. */
GPU_ROUTINE GPU_FN void sp_advect_one(const sp_box *GPU_RESTRICT box, double p[3], double vx, double vy, double vz, double dt)
{
  p[0] += vx * dt;
  p[1] += vy * dt;
  p[2] += vz * dt;
  sp_wrap(box, p);
}

/** @brief MFTLGetOwnerProc(): the rank of the process-grid cell containing @p p. */
GPU_ROUTINE GPU_FN int sp_owner(const sp_box *GPU_RESTRICT box, const int *GPU_RESTRICT rank_map, const double p[3])
{
  int c[3];
  for (int k = 0; k < 3; ++k)
  {
    int i = (int) ((p[k] - box->min[k]) * box->gf[k]);
    if (i < 0) i = 0;
    if (i >= box->cuts[k]) i = box->cuts[k] - 1;
    c[k] = i;
  }
  return rank_map[c[0] + c[1] * box->cuts[0] + c[2] * box->cuts[0] * box->cuts[1]];
}

/** @brief p0 and the seam-consistent edges e1 = p1 - p0, e2 = p2 - p0 of triangle @p t. */
GPU_ROUTINE GPU_FN void sp_triangle_edges(const sp_box *GPU_RESTRICT box, const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tv, int t, double p0[3],
                                          double e1[3], double e2[3])
{
  const double *a = pos + 3 * tv[3 * t + 0];
  const double *b = pos + 3 * tv[3 * t + 1];
  const double *c = pos + 3 * tv[3 * t + 2];

  for (int k = 0; k < 3; ++k)
  {
    p0[k] = a[k];
    const double d1 = b[k] - a[k], d2 = c[k] - a[k];
    e1[k] = d1 - box->L[k] * nearbyint(d1 * box->invL[k]);
    e2[k] = d2 - box->L[k] * nearbyint(d2 * box->invL[k]);
  }
}

/** @brief The wrapped centroid p0 + (e1 + e2) / 3: what redistribution routes a triangle by. */
GPU_ROUTINE GPU_FN void sp_triangle_center(const sp_box *GPU_RESTRICT box, const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tv, int t, double c[3])
{
  double p0[3], e1[3], e2[3];
  sp_triangle_edges(box, pos, tv, t, p0, e1, e2);
  for (int k = 0; k < 3; ++k) c[k] = p0[k] + (e1[k] + e2[k]) * SP_ONE_THIRD;
  sp_wrap(box, c);
}

/**
 * @brief Area, unit normal and centroid of triangle @p t into the slot's
 *        arrays, and its contribution to the statistics (update_kernel()).
 */
GPU_ROUTINE GPU_FN sp_tri_stats sp_triangle_geometry(const sp_box *GPU_RESTRICT box, const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tv, int t,
                                                     double *GPU_RESTRICT area, double *GPU_RESTRICT normal, double *GPU_RESTRICT center)
{
  double p0[3], e1[3], e2[3];
  sp_triangle_edges(box, pos, tv, t, p0, e1, e2);

  const double nx = e1[1] * e2[2] - e1[2] * e2[1];
  const double ny = e1[2] * e2[0] - e1[0] * e2[2];
  const double nz = e1[0] * e2[1] - e1[1] * e2[0];
  const double mag = sqrt(nx * nx + ny * ny + nz * nz);

  const double e3x = e2[0] - e1[0], e3y = e2[1] - e1[1], e3z = e2[2] - e1[2];
  const double a_sq = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2];
  const double b_sq = e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2];
  const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;
  const double denom = a_sq + b_sq + c_sq;

  sp_tri_stats s;
  s.area = 0.5 * mag;
  s.q = (denom > 0.0) ? SP_FOUR_SQRT3 * s.area / denom : 0.0;
  s.esq_min = fmin(a_sq, fmin(b_sq, c_sq));
  s.esq_max = fmax(a_sq, fmax(b_sq, c_sq));
  s.esq_sum = denom;

  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
  area[t] = s.area;
  normal[3 * t + 0] = nx * inv;
  normal[3 * t + 1] = ny * inv;
  normal[3 * t + 2] = nz * inv;

  double c[3] = {p0[0] + (e1[0] + e2[0]) * SP_ONE_THIRD, p0[1] + (e1[1] + e2[1]) * SP_ONE_THIRD, p0[2] + (e1[2] + e2[2]) * SP_ONE_THIRD};
  sp_wrap(box, c);
  center[3 * t + 0] = c[0];
  center[3 * t + 1] = c[1];
  center[3 * t + 2] = c[2];
  return s;
}

/** @brief e1 x e2 = 2 A n: accumulated raw into the corners, it IS the area weighting. */
GPU_ROUTINE GPU_FN void sp_face_normal(const sp_box *GPU_RESTRICT box, const double *GPU_RESTRICT pos, const int *GPU_RESTRICT tv, int t, double n[3])
{
  double p0[3], e1[3], e2[3];
  sp_triangle_edges(box, pos, tv, t, p0, e1, e2);
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];
}

/** @brief Unit length in place; a vertex without incident triangle stays exactly zero. */
GPU_ROUTINE GPU_FN void sp_normalise_one(double *GPU_RESTRICT vn, int v)
{
  const double x = vn[3 * v + 0], y = vn[3 * v + 1], z = vn[3 * v + 2];
  const double len_sq = x * x + y * y + z * z;
  const double inv = (len_sq > 0.0) ? 1.0 / sqrt(len_sq) : 0.0;
  vn[3 * v + 0] = x * inv;
  vn[3 * v + 1] = y * inv;
  vn[3 * v + 2] = z * inv;
}

GPU_END_DEVICE

#endif
