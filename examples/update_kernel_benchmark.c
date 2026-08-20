/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Is the fused per-timestep update kernel actually optimal?
 *
 * src/ensemble/ensemble_update.c makes three claims that were inherited from the
 * previous version rather than re-measured on this one:
 *
 *   1. FUSING geometry and quality statistics into one pass beats two passes,
 *      because the statistics come from vectors already in registers.
 *   2. The loop does NOT vectorise, and the nine-way statistics reduction is
 *      what stops it.
 *   3. The output layout barely matters.
 *
 * Claim 2 is the interesting one, because "it does not vectorise" is a statement
 * about the compiler, not about the hardware. If the reduction is what blocks
 * vectorisation, then splitting the reduction into several independent partial
 * accumulators should unblock it -- that is the standard transformation, and it
 * is the "last bit of vectorisation" this benchmark exists to price.
 *
 * Eight variants, all writing the same three arrays and producing the same nine
 * statistics on the same mesh in the same binary:
 *
 *   U0   the shipping kernel, written with MFTL_FOREACH_TRIANGLE_FRAME
 *   U1   geometry only, no statistics          -- what do the statistics cost?
 *   U2   statistics only, no geometry writes   -- what does the geometry cost?
 *   U3   U1 followed by U2, as two passes      -- is fusing worth anything?
 *   U4   fused, but with SPLIT output arrays (normal_x/y/z, center_x/y/z)
 *   U5   fused, 1 accumulator set, via a step function -- U6/U7's baseline
 *   U6   fused, 2 independent accumulator sets  -- breaks the reduction chain
 *   U7   fused, 4 independent accumulator sets
 *
 * U5 exists so that U6 and U7 are compared against the same abstraction they are
 * built on. Otherwise a difference between U0 and U7 could be the unrolling or
 * could be the step function, and there would be no way to tell which.
 *
 * Build Release for anything you intend to quote:
 *     python3 build_all.py && ./build/update_kernel_benchmark
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"

#define REP 20
#define TRIALS 9

/** @brief The nine statistics every variant must agree on. */
typedef struct
{
  double min_area, max_area, sum_area;
  double min_q, max_q, sum_q;
  double min_esq, max_esq, sum_esq;
} acc_t;

static void acc_init(acc_t *a)
{
  a->min_area = MFTL_DBL_MAX;
  a->max_area = -MFTL_DBL_MAX;
  a->sum_area = 0.0;
  a->min_q = MFTL_DBL_MAX;
  a->max_q = -MFTL_DBL_MAX;
  a->sum_q = 0.0;
  a->min_esq = MFTL_DBL_MAX;
  a->max_esq = -MFTL_DBL_MAX;
  a->sum_esq = 0.0;
}

/** @brief Folds @p b into @p a. Used to join the partial accumulators of U6/U7. */
static void acc_merge(acc_t *a, const acc_t *b)
{
  a->min_area = fmin(a->min_area, b->min_area);
  a->max_area = fmax(a->max_area, b->max_area);
  a->sum_area += b->sum_area;
  a->min_q = fmin(a->min_q, b->min_q);
  a->max_q = fmax(a->max_q, b->max_q);
  a->sum_q += b->sum_q;
  a->min_esq = fmin(a->min_esq, b->min_esq);
  a->max_esq = fmax(a->max_esq, b->max_esq);
  a->sum_esq += b->sum_esq;
}

/** @brief Largest relative disagreement between two statistic sets. */
static double acc_max_rel_diff(const acc_t *a, const acc_t *b)
{
  const double *x = (const double *) a;
  const double *y = (const double *) b;
  double worst = 0.0;
  for (int i = 0; i < 9; ++i)
  {
    const double scale = fmax(fabs(x[i]), fabs(y[i]));
    const double d = (scale > 0.0) ? fabs(x[i] - y[i]) / scale : 0.0;
    if (d > worst) worst = d;
  }
  return worst;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Context
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  MFTL_Object *obj;
  acc_t acc;

  /* Split output arrays, for U4 only. */
  double *nx, *ny, *nz;
  double *cx, *cy, *cz;
  void *block;
} upd_ctx;

/* ═══════════════════════════════════════════════════════════════════════════
 * U0 -- the shipping kernel, copied verbatim from src/ensemble/ensemble_update.c
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u0_shipping(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;

  double min_area = MFTL_DBL_MAX, max_area = -MFTL_DBL_MAX, sum_area = 0.0;
  double min_q = MFTL_DBL_MAX, max_q = -MFTL_DBL_MAX, sum_q = 0.0;
  double min_esq = MFTL_DBL_MAX, max_esq = -MFTL_DBL_MAX, sum_esq = 0.0;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
      const double area = 0.5 * mag;

      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;

      const double inv_mag = 1.0 / mag;
      out_area[t] = area;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
      MFTLApplyPeriodicity(dom, cc);
      out_center[3 * t + 0] = cc[0];
      out_center[3 * t + 1] = cc[1];
      out_center[3 * t + 2] = cc[2];);

  c->acc.min_area = min_area;
  c->acc.max_area = max_area;
  c->acc.sum_area = sum_area;
  c->acc.min_q = min_q;
  c->acc.max_q = max_q;
  c->acc.sum_q = sum_q;
  c->acc.min_esq = min_esq;
  c->acc.max_esq = max_esq;
  c->acc.sum_esq = sum_esq;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U1 -- geometry only.  U2 -- statistics only.  U3 -- the two in sequence.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u1_geometry_only(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
      const double inv_mag = 1.0 / mag;

      out_area[t] = 0.5 * mag;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
      MFTLApplyPeriodicity(dom, cc);
      out_center[3 * t + 0] = cc[0];
      out_center[3 * t + 1] = cc[1];
      out_center[3 * t + 2] = cc[2];);
}

static void u2_stats_only(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;

  double min_area = MFTL_DBL_MAX, max_area = -MFTL_DBL_MAX, sum_area = 0.0;
  double min_q = MFTL_DBL_MAX, max_q = -MFTL_DBL_MAX, sum_q = 0.0;
  double min_esq = MFTL_DBL_MAX, max_esq = -MFTL_DBL_MAX, sum_esq = 0.0;

  /* EDGE, not FRAME: nothing here needs an absolute position, so this variant
   * also shows what dropping tris.anchor from the read set is worth. */
  MFTL_FOREACH_TRIANGLE_EDGE(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double area = 0.5 * MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));

      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;);

  c->acc.min_area = min_area;
  c->acc.max_area = max_area;
  c->acc.sum_area = sum_area;
  c->acc.min_q = min_q;
  c->acc.max_q = max_q;
  c->acc.sum_q = sum_q;
  c->acc.min_esq = min_esq;
  c->acc.max_esq = max_esq;
  c->acc.sum_esq = sum_esq;
}

static void u3_two_passes(upd_ctx *c)
{
  u1_geometry_only(c);
  u2_stats_only(c);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U4 -- fused, but with the outputs SPLIT into one array per component
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u4_split_outputs(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  double *const restrict out_area = obj->tris.area;
  double *const restrict onx = c->nx;
  double *const restrict ony = c->ny;
  double *const restrict onz = c->nz;
  double *const restrict ocx = c->cx;
  double *const restrict ocy = c->cy;
  double *const restrict ocz = c->cz;
  const MFTL_Domain *const dom = obj->domain;

  double min_area = MFTL_DBL_MAX, max_area = -MFTL_DBL_MAX, sum_area = 0.0;
  double min_q = MFTL_DBL_MAX, max_q = -MFTL_DBL_MAX, sum_q = 0.0;
  double min_esq = MFTL_DBL_MAX, max_esq = -MFTL_DBL_MAX, sum_esq = 0.0;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
      const double area = 0.5 * mag;

      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;

      const double inv_mag = 1.0 / mag;
      out_area[t] = area;
      onx[t] = nx * inv_mag;
      ony[t] = ny * inv_mag;
      onz[t] = nz * inv_mag;

      double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
      MFTLApplyPeriodicity(dom, cc);
      ocx[t] = cc[0];
      ocy[t] = cc[1];
      ocz[t] = cc[2];);

  c->acc.min_area = min_area;
  c->acc.max_area = max_area;
  c->acc.sum_area = sum_area;
  c->acc.min_q = min_q;
  c->acc.max_q = max_q;
  c->acc.sum_q = sum_q;
  c->acc.min_esq = min_esq;
  c->acc.max_esq = max_esq;
  c->acc.sum_esq = sum_esq;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U5 / U6 / U7 -- one, two and four independent accumulator sets
 *
 * The transformation under test. Nine reductions, each carrying a loop-carried
 * dependency through an FP unit: sum_area, sum_q and sum_esq are ~3-4 cycle
 * adds, and the six fmin/fmax are ~2-3 cycles each. They are independent OF EACH
 * OTHER, so out-of-order execution overlaps them -- but each individual chain
 * still admits only one triangle every add-latency. N independent accumulator
 * sets cut that floor by N, at the cost of an N-way unrolled loop, a tail loop,
 * and a merge.
 *
 * It also changes the SUMMATION ORDER, which changes the last bits of the
 * averages. That is not a correctness problem here (the result is a quality
 * statistic and the partial-sum form is in fact slightly MORE accurate), but it
 * would be if any of these numbers fed a convergence test.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One triangle: geometry to the output arrays, statistics into @p a. */
static inline void step(acc_t *restrict a, const double *restrict ed, const double *restrict an, int t, double *restrict out_area,
                        double *restrict out_normal, double *restrict out_center, const MFTL_Domain *dom)
{
  const double e1x = ed[6 * t + 0], e1y = ed[6 * t + 1], e1z = ed[6 * t + 2];
  const double e2x = ed[6 * t + 3], e2y = ed[6 * t + 4], e2z = ed[6 * t + 5];
  const double p0x = an[3 * t + 0], p0y = an[3 * t + 1], p0z = an[3 * t + 2];

  const double nx = e1y * e2z - e1z * e2y;
  const double ny = e1z * e2x - e1x * e2z;
  const double nz = e1x * e2y - e1y * e2x;

  const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
  const double area = 0.5 * mag;

  const double e3x = e2x - e1x;
  const double e3y = e2y - e1y;
  const double e3z = e2z - e1z;

  const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
  const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
  const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

  const double denom = a_sq + b_sq + c_sq;
  const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

  a->min_area = fmin(a->min_area, area);
  a->max_area = fmax(a->max_area, area);
  a->sum_area += area;
  a->min_q = fmin(a->min_q, q);
  a->max_q = fmax(a->max_q, q);
  a->sum_q += q;
  a->min_esq = fmin(a->min_esq, fmin(a_sq, fmin(b_sq, c_sq)));
  a->max_esq = fmax(a->max_esq, fmax(a_sq, fmax(b_sq, c_sq)));
  a->sum_esq += denom;

  const double inv_mag = 1.0 / mag;
  out_area[t] = area;
  out_normal[3 * t + 0] = nx * inv_mag;
  out_normal[3 * t + 1] = ny * inv_mag;
  out_normal[3 * t + 2] = nz * inv_mag;

  double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
  MFTLApplyPeriodicity(dom, cc);
  out_center[3 * t + 0] = cc[0];
  out_center[3 * t + 1] = cc[1];
  out_center[3 * t + 2] = cc[2];
}

/** @brief @p WAY independent accumulator sets over the same loop. */
#define DEFINE_UNROLLED(NAME, WAY)                                                        \
  static void NAME(upd_ctx *c)                                                            \
  {                                                                                       \
    MFTL_Object *obj = c->obj;                                                            \
    const double *const restrict ed = obj->tris.edge;                                     \
    const double *const restrict an = obj->tris.anchor;                                   \
    double *const restrict oa = obj->tris.area;                                           \
    double *const restrict on = obj->tris.normal;                                         \
    double *const restrict oc = obj->tris.center;                                         \
    const MFTL_Domain *const dom = obj->domain;                                           \
    const int n = obj->tris.n;                                                            \
                                                                                          \
    acc_t a[WAY];                                                                         \
    for (int i = 0; i < (WAY); ++i) acc_init(&a[i]);                                       \
                                                                                          \
    int t = 0;                                                                            \
    for (; t + (WAY) -1 < n; t += (WAY))                                                  \
    {                                                                                     \
      for (int i = 0; i < (WAY); ++i) step(&a[i], ed, an, t + i, oa, on, oc, dom);        \
    }                                                                                     \
    for (; t < n; ++t) step(&a[0], ed, an, t, oa, on, oc, dom);                           \
                                                                                          \
    for (int i = 1; i < (WAY); ++i) acc_merge(&a[0], &a[i]);                              \
    c->acc = a[0];                                                                        \
  }

DEFINE_UNROLLED(u5_step_x1, 1)
DEFINE_UNROLLED(u6_step_x2, 2)
DEFINE_UNROLLED(u7_step_x4, 4)

/* ═══════════════════════════════════════════════════════════════════════════
 * U8 -- the unrolling test done WITHOUT the struct
 *
 * U5..U7 hold their accumulators in an acc_t[WAY], and an address-taken array is
 * exactly the thing that can stop clang from keeping the nine values in
 * registers. U5 is 15 % slower than the identical U0, which says the abstraction
 * costs something -- so U6/U7's gain over U5 might be nothing more than clawing
 * that back.
 *
 * U8 removes the doubt: the same 2-way unroll written with eighteen plain
 * scalars, in the same shape as the shipping kernel. If breaking the reduction
 * chain is worth anything at all, it shows up here as U8 < U0.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One triangle's worth of work, accumulating into the named scalars. */
#define UPD_BODY(T, MINA, MAXA, SUMA, MINQ, MAXQ, SUMQ, MINE, MAXE, SUME)                                                   \
  do                                                                                                                        \
  {                                                                                                                         \
    const int t_ = (T);                                                                                                     \
    const double e1x = ed[6 * t_ + 0], e1y = ed[6 * t_ + 1], e1z = ed[6 * t_ + 2];                                          \
    const double e2x = ed[6 * t_ + 3], e2y = ed[6 * t_ + 4], e2z = ed[6 * t_ + 5];                                          \
    const double p0x = an[3 * t_ + 0], p0y = an[3 * t_ + 1], p0z = an[3 * t_ + 2];                                          \
                                                                                                                            \
    const double nx = e1y * e2z - e1z * e2y;                                                                                \
    const double ny = e1z * e2x - e1x * e2z;                                                                                \
    const double nz = e1x * e2y - e1y * e2x;                                                                                \
                                                                                                                            \
    const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));                                    \
    const double area = 0.5 * mag;                                                                                          \
                                                                                                                            \
    const double e3x = e2x - e1x, e3y = e2y - e1y, e3z = e2z - e1z;                                                         \
    const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;                                                                  \
    const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;                                                                  \
    const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;                                                                  \
                                                                                                                            \
    const double denom = a_sq + b_sq + c_sq;                                                                                \
    const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;                                       \
                                                                                                                            \
    MINA = fmin(MINA, area);                                                                                                \
    MAXA = fmax(MAXA, area);                                                                                                \
    SUMA += area;                                                                                                           \
    MINQ = fmin(MINQ, q);                                                                                                   \
    MAXQ = fmax(MAXQ, q);                                                                                                   \
    SUMQ += q;                                                                                                              \
    MINE = fmin(MINE, fmin(a_sq, fmin(b_sq, c_sq)));                                                                        \
    MAXE = fmax(MAXE, fmax(a_sq, fmax(b_sq, c_sq)));                                                                        \
    SUME += denom;                                                                                                          \
                                                                                                                            \
    const double inv_mag = 1.0 / mag;                                                                                       \
    oa[t_] = area;                                                                                                          \
    on[3 * t_ + 0] = nx * inv_mag;                                                                                          \
    on[3 * t_ + 1] = ny * inv_mag;                                                                                          \
    on[3 * t_ + 2] = nz * inv_mag;                                                                                          \
                                                                                                                            \
    double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD}; \
    MFTLApplyPeriodicity(dom, cc);                                                                                          \
    oc[3 * t_ + 0] = cc[0];                                                                                                 \
    oc[3 * t_ + 1] = cc[1];                                                                                                 \
    oc[3 * t_ + 2] = cc[2];                                                                                                 \
  } while (0)

static void u8_scalar_x2(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  const double *const restrict ed = obj->tris.edge;
  const double *const restrict an = obj->tris.anchor;
  double *const restrict oa = obj->tris.area;
  double *const restrict on = obj->tris.normal;
  double *const restrict oc = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;
  const int n = obj->tris.n;

  double mina0 = MFTL_DBL_MAX, maxa0 = -MFTL_DBL_MAX, suma0 = 0.0;
  double minq0 = MFTL_DBL_MAX, maxq0 = -MFTL_DBL_MAX, sumq0 = 0.0;
  double mine0 = MFTL_DBL_MAX, maxe0 = -MFTL_DBL_MAX, sume0 = 0.0;

  double mina1 = MFTL_DBL_MAX, maxa1 = -MFTL_DBL_MAX, suma1 = 0.0;
  double minq1 = MFTL_DBL_MAX, maxq1 = -MFTL_DBL_MAX, sumq1 = 0.0;
  double mine1 = MFTL_DBL_MAX, maxe1 = -MFTL_DBL_MAX, sume1 = 0.0;

  int t = 0;
  for (; t + 1 < n; t += 2)
  {
    UPD_BODY(t + 0, mina0, maxa0, suma0, minq0, maxq0, sumq0, mine0, maxe0, sume0);
    UPD_BODY(t + 1, mina1, maxa1, suma1, minq1, maxq1, sumq1, mine1, maxe1, sume1);
  }
  for (; t < n; ++t) UPD_BODY(t, mina0, maxa0, suma0, minq0, maxq0, sumq0, mine0, maxe0, sume0);

  c->acc.min_area = fmin(mina0, mina1);
  c->acc.max_area = fmax(maxa0, maxa1);
  c->acc.sum_area = suma0 + suma1;
  c->acc.min_q = fmin(minq0, minq1);
  c->acc.max_q = fmax(maxq0, maxq1);
  c->acc.sum_q = sumq0 + sumq1;
  c->acc.min_esq = fmin(mine0, mine1);
  c->acc.max_esq = fmax(maxe0, maxe1);
  c->acc.sum_esq = sume0 + sume1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U9 -- a DIAGNOSTIC PROBE, not a candidate. It computes WRONG NUMBERS.
 *
 * Identical to U1 (geometry only) except that the square root and the reciprocal
 * are deleted: `mag` becomes the squared magnitude and `inv_mag` becomes `mag`.
 * Every other instruction, load and store is the same.
 *
 * The difference U1 - U9 is therefore the price of one sqrt and one divide per
 * triangle, and nothing else. That is worth knowing before reaching for
 * vectorisation: if the kernel is sitting on the divide/sqrt unit's throughput,
 * then unrolling the reductions cannot help it and only a WIDER sqrt could.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u9_probe_no_sqrt(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double mag = MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz); /* no sqrt */
      const double inv_mag = mag;                                               /* no divide */

      out_area[t] = 0.5 * mag;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      double cc[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
      MFTLApplyPeriodicity(dom, cc);
      out_center[3 * t + 0] = cc[0];
      out_center[3 * t + 1] = cc[1];
      out_center[3 * t + 2] = cc[2];);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U10 -- the fused kernel with the periodicity wrap LOOP-UNSWITCHED away
 *
 * A real candidate, not a probe. src/object/derived.c already splits its frame
 * build into a periodic and a non-periodic loop, on the argument that paying for
 * a correction that is identically zero is not free. The update kernel does not
 * do that: it calls MFTLApplyPeriodicity unconditionally, which on a
 * non-periodic domain is three perfectly-predicted, never-taken branches per
 * triangle.
 *
 * This variant is what the non-periodic half of that unswitching would look
 * like. On a non-periodic domain it must be BIT-IDENTICAL to U0, which is
 * checked, so U0 - U10 is exactly the price of the wrap.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u10_no_wrap(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  double min_area = MFTL_DBL_MAX, max_area = -MFTL_DBL_MAX, sum_area = 0.0;
  double min_q = MFTL_DBL_MAX, max_q = -MFTL_DBL_MAX, sum_q = 0.0;
  double min_esq = MFTL_DBL_MAX, max_esq = -MFTL_DBL_MAX, sum_esq = 0.0;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, obj->tris.n,

      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
      const double area = 0.5 * mag;

      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;

      const double inv_mag = 1.0 / mag;
      out_area[t] = area;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      /* No MFTLApplyPeriodicity: on a non-periodic domain it is a no-op. */
      out_center[3 * t + 0] = p0x + (e1x + e2x) * MFTL_ONE_THIRD;
      out_center[3 * t + 1] = p0y + (e1y + e2y) * MFTL_ONE_THIRD;
      out_center[3 * t + 2] = p0z + (e1z + e2z) * MFTL_ONE_THIRD;);

  c->acc.min_area = min_area;
  c->acc.max_area = max_area;
  c->acc.sum_area = sum_area;
  c->acc.min_q = min_q;
  c->acc.max_q = max_q;
  c->acc.sum_q = sum_q;
  c->acc.min_esq = min_esq;
  c->acc.max_esq = max_esq;
  c->acc.sum_esq = sum_esq;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * U11 -- the cost model, overruled
 *
 * Asked directly, clang says of the shipping kernel:
 *
 *     mpicc -O3 -DNDEBUG -Iinclude -c src/ensemble/ensemble_update.c \
 *           -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
 *     ensemble_update.c:94:3: remark: the cost-model indicates that
 *                             vectorization is not beneficial
 *
 * That is NOT "the loop cannot be vectorised". The loop is legal to vectorise --
 * the reductions are recognised, the stores are analysable -- and clang costed it
 * and declined. Which makes the question empirical rather than a matter of
 * removing blockers: a pragma overrules the cost model, and then the hardware
 * settles it.
 *
 * The loop is written out here rather than through MFTL_FOREACH_TRIANGLE_FRAME
 * because the pragma has to sit immediately in front of the `for`.
 * ═══════════════════════════════════════════════════════════════════════════ */

static void u11_forced_vector(upd_ctx *c)
{
  MFTL_Object *obj = c->obj;
  const double *const restrict ed = obj->tris.edge;
  const double *const restrict an = obj->tris.anchor;
  double *const restrict oa = obj->tris.area;
  double *const restrict on = obj->tris.normal;
  double *const restrict oc = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;
  const int n = obj->tris.n;

  double mina = MFTL_DBL_MAX, maxa = -MFTL_DBL_MAX, suma = 0.0;
  double minq = MFTL_DBL_MAX, maxq = -MFTL_DBL_MAX, sumq = 0.0;
  double mine = MFTL_DBL_MAX, maxe = -MFTL_DBL_MAX, sume = 0.0;

#if defined(__clang__)
#pragma clang loop vectorize(enable) vectorize_width(2)
#elif defined(__GNUC__)
#pragma GCC ivdep
#endif
  for (int t = 0; t < n; ++t) UPD_BODY(t, mina, maxa, suma, minq, maxq, sumq, mine, maxe, sume);

  c->acc.min_area = mina;
  c->acc.max_area = maxa;
  c->acc.sum_area = suma;
  c->acc.min_q = minq;
  c->acc.max_q = maxq;
  c->acc.sum_q = sumq;
  c->acc.min_esq = mine;
  c->acc.max_esq = maxe;
  c->acc.sum_esq = sume;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Harness
 * ═══════════════════════════════════════════════════════════════════════════ */

static double best_ns(void (*fn)(upd_ctx *), upd_ctx *c, double elements)
{
  fn(c); /* warm-up */

  double best = 1e30;
  for (int r = 0; r < TRIALS; ++r)
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < REP; ++i) fn(c);
    const double dt = (MPI_Wtime() - t0) / (REP * elements) * 1e9;
    if (dt < best) best = dt;
  }
  return best;
}

/** @brief Snapshot of the three geometry arrays, for cross-variant comparison. */
static double *geom_snapshot(const MFTL_Object *obj)
{
  const int nt = obj->tris.n;
  double *s = (double *) MFTLMallocOrDie(7 * (size_t) nt * sizeof(double), "geometry snapshot");
  for (int t = 0; t < nt; ++t)
  {
    s[7 * t] = obj->tris.area[t];
    for (int k = 0; k < 3; ++k)
    {
      s[7 * t + 1 + k] = obj->tris.normal[3 * t + k];
      s[7 * t + 4 + k] = obj->tris.center[3 * t + k];
    }
  }
  return s;
}

static double geom_max_diff(const MFTL_Object *obj, const double *s)
{
  const int nt = obj->tris.n;
  double worst = 0.0;
  for (int t = 0; t < nt; ++t)
  {
    double d = fabs(s[7 * t] - obj->tris.area[t]);
    if (d > worst) worst = d;
    for (int k = 0; k < 3; ++k)
    {
      d = fabs(s[7 * t + 1 + k] - obj->tris.normal[3 * t + k]);
      if (d > worst) worst = d;
      d = fabs(s[7 * t + 4 + k] - obj->tris.center[3 * t + k]);
      if (d > worst) worst = d;
    }
  }
  return worst;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
  {
    MPI_Finalize();
    return 0;
  }

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0};
  double hi[3] = {2.0, 2.0, 2.0};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 8);

  printf("The fused per-timestep update kernel, taken apart\n");
  printf("ns per TRIANGLE, best of %d runs of %d repetitions. Lower is better.\n\n", TRIALS, REP);

  printf("%-4s %9s | %7s %7s %7s %7s %7s | %7s %7s %7s %7s %8s %8s | %8s | %9s\n", "ref", "tris", "U0 ship", "U1 geom", "U2 stat", "U3 1+2", "U4 splt",
         "U5 x1", "U6 x2", "U7 x4", "U8 x2s", "U10 nowrp", "U11 vec", "U9 probe", "worst dev");
  printf("--------------------------------------------------------------------------------------------------------------------------\n");

  for (int ref = 4; ref <= 8; ++ref)
  {
    MFTL_Object *obj = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(ens, obj);
    double centre[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(obj, centre, 1.0, ref);
    MFTLObjectReorderMorton(obj);

    const int nt = obj->tris.n;

    upd_ctx c;
    memset(&c, 0, sizeof(c));
    c.obj = obj;

    size_t cur = 0;
    size_t off[6];
    for (int i = 0; i < 6; ++i) off[i] = MFTLLayoutAdd(&cur, (size_t) nt, sizeof(double));
    unsigned char *blk = (unsigned char *) MFTLAlignedAllocOrDie(cur, "split outputs");
    memset(blk, 0, cur);
    c.block = blk;
    c.nx = (double *) (void *) (blk + off[0]);
    c.ny = (double *) (void *) (blk + off[1]);
    c.nz = (double *) (void *) (blk + off[2]);
    c.cx = (double *) (void *) (blk + off[3]);
    c.cy = (double *) (void *) (blk + off[4]);
    c.cz = (double *) (void *) (blk + off[5]);

    /* ── Correctness: every variant must agree with U0 ────────────────────── */
    u0_shipping(&c);
    const acc_t ref_acc = c.acc;
    double *ref_geom = geom_snapshot(obj);

    double worst = 0.0;

    void (*writers[])(upd_ctx *) = {u1_geometry_only, u3_two_passes, u5_step_x1, u6_step_x2, u7_step_x4, u8_scalar_x2, u10_no_wrap, u11_forced_vector};
    for (int i = 0; i < 8; ++i)
    {
      acc_init(&c.acc);
      writers[i](&c);
      const double dg = geom_max_diff(obj, ref_geom);
      if (dg > worst) worst = dg;
      /* U1 writes no statistics, so only the others are compared on them. */
      if (i > 0)
      {
        const double da = acc_max_rel_diff(&ref_acc, &c.acc);
        if (da > worst) worst = da;
      }
    }

    acc_init(&c.acc);
    u2_stats_only(&c);
    const double d2 = acc_max_rel_diff(&ref_acc, &c.acc);
    if (d2 > worst) worst = d2;

    /* U4 writes to its own split arrays; check those against the reference. */
    acc_init(&c.acc);
    u4_split_outputs(&c);
    for (int t = 0; t < nt; ++t)
    {
      const double d[6] = {fabs(ref_geom[7 * t + 1] - c.nx[t]), fabs(ref_geom[7 * t + 2] - c.ny[t]), fabs(ref_geom[7 * t + 3] - c.nz[t]),
                           fabs(ref_geom[7 * t + 4] - c.cx[t]), fabs(ref_geom[7 * t + 5] - c.cy[t]), fabs(ref_geom[7 * t + 6] - c.cz[t])};
      for (int k = 0; k < 6; ++k)
        if (d[k] > worst) worst = d[k];
    }

    free(ref_geom);

    /* ── Timing ───────────────────────────────────────────────────────────── */
    const double e = (double) nt;
    const double t0 = best_ns(u0_shipping, &c, e);
    const double t1 = best_ns(u1_geometry_only, &c, e);
    const double t2 = best_ns(u2_stats_only, &c, e);
    const double t3 = best_ns(u3_two_passes, &c, e);
    const double t4 = best_ns(u4_split_outputs, &c, e);
    const double t5 = best_ns(u5_step_x1, &c, e);
    const double t6 = best_ns(u6_step_x2, &c, e);
    const double t7 = best_ns(u7_step_x4, &c, e);
    const double t8 = best_ns(u8_scalar_x2, &c, e);
    const double t10 = best_ns(u10_no_wrap, &c, e);
    const double t11 = best_ns(u11_forced_vector, &c, e);
    const double t9 = best_ns(u9_probe_no_sqrt, &c, e);

    printf("%-4d %9d | %7.3f %7.3f %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f %7.3f %8.3f %8.3f | %8.3f | %9.1e\n", ref, nt, t0, t1, t2, t3, t4, t5, t6, t7, t8,
           t10, t11, t9, worst);

    /* U9 wrote nonsense into the geometry arrays -- restore them, so that the
     * next refinement level does not start from a poisoned state and so that
     * nothing downstream can mistake a probe's output for a measurement. */
    u0_shipping(&c);

    free(blk);
  }

  printf("\nHow to read this\n");
  printf("  U0  the shipping kernel. Everything else is measured against it.\n");
  printf("  U1  geometry without statistics: U0 - U1 is what the nine reductions cost.\n");
  printf("  U2  statistics without geometry, and reading tris.edge only (no anchor).\n");
  printf("  U3  U1 then U2, i.e. NOT fused. U3 - U0 is what fusing is worth, and it is\n");
  printf("      also the answer to 'should quality statistics be split off from\n");
  printf("      area/normal/center?' -- if U3 is slower, splitting them costs time.\n");
  printf("  U4  fused with one output array per component instead of xyz interleaved.\n");
  printf("  U5  U0 rewritten through an inlined step function, 1 accumulator set. It is\n");
  printf("      the honest baseline for U6/U7: any difference between U0 and U7 could\n");
  printf("      otherwise be the abstraction rather than the unrolling.\n");
  printf("  U6  2 independent accumulator sets, U7 4 -- the standard way to break a\n");
  printf("      loop-carried reduction chain, and the last vectorisation lever this\n");
  printf("      kernel has.\n");
  printf("  U8  the same 2-way unroll written with eighteen plain scalars instead of an\n");
  printf("      acc_t[2]. This is the honest version of the unrolling question: U8 vs U0\n");
  printf("      is unrolling against the shipping kernel with nothing else changed.\n");
  printf("  U10 the fused kernel with the periodicity wrap removed, i.e. the non-periodic\n");
  printf("      half of a loop unswitching. Bit-identical to U0 on this domain.\n");
  printf("  U11 the same loop with clang's cost model OVERRULED by a vectorize pragma.\n");
  printf("      This is the direct answer to 'is there anything left in vectorisation':\n");
  printf("      clang reports the loop as legal but not profitable, and U11 is what\n");
  printf("      happens when you insist anyway.\n");
  printf("  U9  A PROBE, NOT A CANDIDATE -- it computes wrong numbers on purpose. It is\n");
  printf("      U1 with the sqrt and the reciprocal deleted and nothing else touched, so\n");
  printf("      U1 - U9 prices exactly one sqrt plus one divide per triangle.\n");
  printf("\n");
  printf("  'worst dev' is the largest deviation of any variant from U0, absolute for\n");
  printf("  the geometry arrays and relative for the statistics. The unrolled variants\n");
  printf("  sum in a different order, so they are expected to differ in the last bits.\n");

  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return 0;
}
