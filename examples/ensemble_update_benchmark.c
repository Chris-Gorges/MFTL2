/*
 * @Author: Christian Gorges
 * @Date: 2026-08-15
 *
 * @brief MFTLEnsembleUpdate against the naive triangle-properties equivalent.
 *
 * The per-timestep geometry update is the loop a front-tracking solver runs more
 * than any other, so it is worth knowing what the fused pass actually buys. Four
 * implementations of overlapping work, on the same mesh, in the same binary:
 *
 *   A  naive     What it looks like written against the public half-edge API:
 *                MFTLGetTriangleProperties per triangle, then a second loop over
 *                half-edges with MFTLGetEdgeLengthSquared for the statistics.
 *                Every value is reached by walking triangle -> half-edge ->
 *                vertex, and every edge gets its own minimum-image correction --
 *                the second loop recomputes edge vectors the first loop already
 *                had in registers and threw away.
 *                Produces GEOMETRY + STATISTICS.
 *
 *   B  general   MFTLComputeTriangleGeometry: the library's always-valid path,
 *                which needs no derived arrays. Same traversal as A, but no
 *                statistics at all, so it does strictly LESS work than D. It is
 *                here to keep the comparison honest -- if the fused pass beats
 *                this, it beats the current per-timestep cost while computing
 *                more, and if it does not, the headline number against A is
 *                measuring the second loop rather than the layout.
 *                Produces GEOMETRY only.
 *
 *   C  frames    The same geometry from MFTL_FOREACH_TRIANGLE_FRAME: two
 *                contiguous arrays in, no indirection, no periodic correction in
 *                the loop. Not a library function -- it exists here so that
 *                D - C isolates what the statistics cost the fused way, against
 *                A - B for the naive way.
 *                Produces GEOMETRY only.
 *
 *   D  fused     MFTLEnsembleUpdate. One streaming pass producing geometry AND
 *                all nine quality statistics from vectors already in registers.
 *                Produces GEOMETRY + STATISTICS, and reduces them across ranks.
 *
 * ── Correctness first ──────────────────────────────────────────────────────
 *
 * A faster answer is worth nothing if it is a different answer, so each mesh
 * size begins by checking that A and D agree on every triangle's area, normal
 * and centroid, and on all nine statistics. The largest deviation is printed.
 * Expect floating-point round-off: the two evaluate the same expressions, but
 * the fused path differences coordinates that the seam correction may have
 * shifted by exactly one domain length, and that shift is not bit-reversible.
 *
 * ── Reading the numbers ────────────────────────────────────────────────────
 *
 * Variants run INTERLEAVED and the best time of each is reported. Running them
 * in blocks would hand whichever went last a warm cache and make the comparison
 * meaningless.
 *
 * Both a NON-PERIODIC and a PERIODIC domain are measured, and the difference
 * between them is the point rather than a detail: on a periodic domain the naive
 * path pays a minimum-image correction per edge per pass, while the fused path
 * paid it once when the frames were built. The periodic case is what a real
 * front-tracking run has.
 *
 * D includes its three non-blocking MPI reductions, because that is what calling
 * the function costs. At one rank they are a few microseconds -- visible at the
 * smallest mesh here, invisible at the largest.
 *
 * Build Release for anything you intend to quote:
 *     python3 build_all.py && ./build/ensemble_update_benchmark
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"

/* Refinement levels: 20 * 4^r triangles each. */
#define MIN_REFINE 4
#define MAX_REFINE 7

/* ═══════════════════════════════════════════════════════════════════════════
 * The variants
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Variant A: geometry and statistics through the public half-edge API.
 *
 * Two passes, because that is how it comes out when the statistics are added to
 * an existing geometry loop: nothing in the API hands the second loop the edge
 * vectors the first one computed, so it recovers them from the vertices again.
 * That is the cost the fused kernel removes, and it is a bigger effect than the
 * indirection everyone expects to be the problem.
 */
static void variant_naive(MFTL_Object *obj, MFTL_ObjectQualityStatistics *out)
{
  const int n = obj->tris.n_local;

  for (int t = 0; t < n; ++t) MFTLGetTriangleProperties(obj, t, &obj->tris.area[t], &obj->tris.normal[3 * t], &obj->tris.center[3 * t]);

  double min_area = MFTL_DBL_MAX, max_area = -MFTL_DBL_MAX, sum_area = 0.0;
  double min_q = MFTL_DBL_MAX, max_q = -MFTL_DBL_MAX, sum_q = 0.0;
  double min_esq = MFTL_DBL_MAX, max_esq = -MFTL_DBL_MAX, sum_esq = 0.0;

  for (int t = 0; t < n; ++t)
  {
    const double a_sq = MFTLGetEdgeLengthSquared(obj, 3 * t + 0);
    const double b_sq = MFTLGetEdgeLengthSquared(obj, 3 * t + 1);
    const double c_sq = MFTLGetEdgeLengthSquared(obj, 3 * t + 2);

    const double area = obj->tris.area[t];
    const double denom = a_sq + b_sq + c_sq;
    const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

    min_area = fmin(min_area, area);
    max_area = fmax(max_area, area);
    sum_area += area;

    min_q = fmin(min_q, q);
    max_q = fmax(max_q, q);
    sum_q += q;

    min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq)));
    max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq)));
    sum_esq += denom;
  }

  const double nt = (double) n;
  out->min_triangle_area = (n > 0) ? min_area : 0.0;
  out->max_triangle_area = (n > 0) ? max_area : 0.0;
  out->avg_triangle_area = (n > 0) ? sum_area / nt : 0.0;
  out->min_aspect_ratio = (n > 0) ? min_q : 0.0;
  out->max_aspect_ratio = (n > 0) ? max_q : 0.0;
  out->avg_aspect_ratio = (n > 0) ? sum_q / nt : 0.0;
  out->min_edge_length_sq = (n > 0) ? min_esq : 0.0;
  out->max_edge_length_sq = (n > 0) ? max_esq : 0.0;
  out->avg_edge_length_sq = (n > 0) ? sum_esq / (3.0 * nt) : 0.0;
}

/**
 * @brief Variant B: MFTLComputeTriangleGeometry's general path, open-coded.
 *
 * Not a call to MFTLComputeTriangleGeometry, deliberately. That function now
 * dispatches on `obj->derived`, and the frames are current throughout this
 * benchmark -- so calling it would give B the streaming kernel and make it
 * identical to C, quietly turning the most interesting row of the table into a
 * duplicate of the row below it.
 *
 * This is that function's slow-path body verbatim, which is what B is supposed
 * to be measuring: the always-valid walk a generator or an I/O reader gets.
 */
static void variant_general(MFTL_Object *obj)
{
  const int n = obj->tris.n_local;
  for (int t = 0; t < n; ++t)
  {
    MFTLGetTriangleProperties(obj, t, &obj->tris.area[t], &obj->tris.normal[3 * t], &obj->tris.center[3 * t]);
  }
}

/** @brief Variant C: the geometry half of the fused kernel, on its own. */
static void variant_frames(MFTL_Object *obj)
{
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_FRAME(obj, obj->tris.n_local,
                              const double nx = e1y * e2z - e1z * e2y;
                              const double ny = e1z * e2x - e1x * e2z;
                              const double nz = e1x * e2y - e1y * e2x;

                              const double mag = MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, nx * nx + ny * ny + nz * nz));
                              const double inv = 1.0 / mag;

                              out_area[t] = 0.5 * mag;
                              out_normal[3 * t + 0] = nx * inv;
                              out_normal[3 * t + 1] = ny * inv;
                              out_normal[3 * t + 2] = nz * inv;

                              double c[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
                              MFTLApplyPeriodicity(dom, c);
                              out_center[3 * t + 0] = c[0];
                              out_center[3 * t + 1] = c[1];
                              out_center[3 * t + 2] = c[2];);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Comparison
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief area, normal and centre of every triangle, so two variants can be compared. */
static double *snapshot(const MFTL_Object *obj)
{
  const int n = obj->tris.n_local;
  double *s = (double *) malloc(7u * (size_t) n * sizeof(double));
  for (int t = 0; t < n; ++t)
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

static double max_geometry_diff(const MFTL_Object *obj, const double *s)
{
  const int n = obj->tris.n_local;
  double worst = 0.0;
  for (int t = 0; t < n; ++t)
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

static double max_stats_diff(const MFTL_ObjectQualityStatistics *a, const MFTL_ObjectQualityStatistics *b)
{
  const double x[9] = {a->min_triangle_area, a->max_triangle_area,  a->avg_triangle_area, a->min_aspect_ratio,  a->max_aspect_ratio,
                       a->avg_aspect_ratio,  a->min_edge_length_sq, a->max_edge_length_sq, a->avg_edge_length_sq};
  const double y[9] = {b->min_triangle_area, b->max_triangle_area,  b->avg_triangle_area, b->min_aspect_ratio,  b->max_aspect_ratio,
                       b->avg_aspect_ratio,  b->min_edge_length_sq, b->max_edge_length_sq, b->avg_edge_length_sq};

  double worst = 0.0;
  for (int i = 0; i < 9; ++i)
  {
    /* Relative, because the nine live on wildly different scales: an area of
     * 1e-5 and a squared edge length of 1e-2 in the same comparison. */
    const double d = fabs(x[i] - y[i]) / (fabs(y[i]) + 1e-300);
    if (d > worst) worst = d;
  }
  return worst;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * One mesh size
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double naive, general, frames, fused;
} timings;

static void run_one(MFTL_Ensemble *ens, MFTL_Object *obj, int refine, bool periodic)
{
  const int n = obj->tris.n_local;

  /* ── Correctness ──────────────────────────────────────────────────────── */
  MFTL_ObjectQualityStatistics naive_stats;
  variant_naive(obj, &naive_stats);
  double *ref = snapshot(obj);

  MFTLEnsembleUpdate(ens);

  const double geo_dev = max_geometry_diff(obj, ref);
  const double stat_dev = max_stats_diff(&naive_stats, &obj->quality);
  free(ref);

  /* ── Timing ───────────────────────────────────────────────────────────── */
  int reps = (int) (2.0e7 / (double) n);
  if (reps < 3) reps = 3;
  if (reps > 50) reps = 50;

  timings best = {1e30, 1e30, 1e30, 1e30};

  for (int r = 0; r < reps; ++r)
  {
    double t0 = MPI_Wtime();
    MFTL_ObjectQualityStatistics tmp;
    variant_naive(obj, &tmp);
    double dt = MPI_Wtime() - t0;
    if (dt < best.naive) best.naive = dt;

    t0 = MPI_Wtime();
    variant_general(obj);
    dt = MPI_Wtime() - t0;
    if (dt < best.general) best.general = dt;

    t0 = MPI_Wtime();
    variant_frames(obj);
    dt = MPI_Wtime() - t0;
    if (dt < best.frames) best.frames = dt;

    t0 = MPI_Wtime();
    MFTLEnsembleUpdate(ens);
    dt = MPI_Wtime() - t0;
    if (dt < best.fused) best.fused = dt;
  }

  const double ns = 1e9 / (double) n;

  printf("  %-9s refine %d, %8d triangles, %2d reps\n", periodic ? "PERIODIC" : "flat", refine, n, reps);
  printf("    A naive   (geom+stats, half-edge walk)  %9.3f ms   %6.2f ns/tri\n", 1e3 * best.naive, ns * best.naive);
  printf("    B general (geom only,  half-edge walk)  %9.3f ms   %6.2f ns/tri\n", 1e3 * best.general, ns * best.general);
  printf("    C frames  (geom only,  streaming)       %9.3f ms   %6.2f ns/tri\n", 1e3 * best.frames, ns * best.frames);
  printf("    D fused   (geom+stats, streaming)       %9.3f ms   %6.2f ns/tri\n", 1e3 * best.fused, ns * best.fused);
  printf("      D vs A  %5.2fx   (same work: the headline)\n", best.naive / best.fused);
  printf("      D vs B  %5.2fx   (D does strictly MORE: it also produces the statistics)\n", best.general / best.fused);
  printf("      statistics cost  +%6.1f %% the naive way (A-B),  +%5.1f %% the fused way (D-C)\n", 100.0 * (best.naive - best.general) / best.general,
         100.0 * (best.fused - best.frames) / best.frames);
  printf("      max deviation A vs D: geometry %.3e absolute, statistics %.3e relative\n\n", geo_dev, stat_dev);
}

int main(int argc, char *argv[])
{
  (void) argv;
  MPI_Init(&argc, NULL);

  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1)
  {
    fprintf(stderr, "ensemble_update_benchmark is a single-rank benchmark (got %d ranks).\n", nproc);
    MPI_Finalize();
    return 1;
  }

  printf("\n=== MFTLEnsembleUpdate vs the naive triangle-properties loop ===\n\n");
  printf("A = MFTLGetTriangleProperties + a second half-edge pass for the statistics\n");
  printf("B = MFTLComputeTriangleGeometry's WALK    (geometry only, does LESS than D)\n");
  printf("C = MFTL_FOREACH_TRIANGLE_FRAME           (geometry only, streaming)\n");
  printf("D = MFTLEnsembleUpdate                    (geometry + statistics, streaming)\n\n");

  for (int pass = 0; pass < 2; ++pass)
  {
    const bool periodic = (pass == 1);
    bool per[3] = {periodic, periodic, periodic};

    int cuts[3] = {1, 1, 1};
    double lo[3] = {-2.0, -2.0, -2.0};
    double hi[3] = {2.0, 2.0, 2.0};

    for (int refine = MIN_REFINE; refine <= MAX_REFINE; ++refine)
    {
      MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 1);
      MFTL_Object *obj = MFTLObjectCreate(0);
      MFTLEnsembleAddObject(ens, obj);

      double centre[3] = {0.0, 0.0, 0.0};
      MFTLGenerateSphere(obj, centre, 1.0, refine);

      /* The frames three of the four variants read. Built once; nothing here
       * moves a vertex, so they stay current for the whole measurement. */
      MFTLObjectBuildDerived(obj);

      run_one(ens, obj, refine, periodic);

      MFTLEnsembleDestroy(&ens);
    }
  }

  printf("Note: D includes its three MPI reductions. At one rank they are a few\n");
  printf("microseconds -- a visible share of the smallest mesh, nothing at the largest.\n");

  MPI_Finalize();
  return 0;
}
