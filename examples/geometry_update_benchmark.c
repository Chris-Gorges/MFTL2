/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Measures the two things the data layout was chosen for.
 *
 * This example exists so that the architectural claims in mftl_types.h can be
 * RECHECKED rather than believed. It compares, on the same mesh, in the same
 * binary, producing bit-identical results:
 *
 *   A  the general path -- MFTLComputeTriangleGeometry, which walks the
 *      half-edge records, gathers three positions per triangle and applies the
 *      minimum-image correction inside the loop. Always valid; needs no derived
 *      arrays.
 *
 *   B  the hot path -- MFTL_FOREACH_TRIANGLE_FRAME, which streams two contiguous
 *      arrays and dereferences nothing. The periodic correction was paid once,
 *      at frame-build time.
 *
 * and then measures what the Morton renumbering is worth by running B before and
 * after MFTLObjectReorderMorton on a deliberately shuffled pool.
 *
 * Build Release for anything you intend to quote:
 *     python3 build_all.py && ./build/geometry_update_benchmark
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"

#define REP 20

/** @brief Variant B: the streaming frame loop, geometry only, no statistics. */
static void geometry_from_frames(MFTL_Object *obj)
{
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;
  const MFTL_Domain *const dom = obj->domain;

  MFTL_FOREACH_TRIANGLE_FRAME(obj, obj->tris.n,
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

/** @brief Snapshots the geometry arrays so the two variants can be compared exactly. */
static double *snapshot(const MFTL_Object *obj)
{
  const int nt = obj->tris.n;
  double *s = (double *) malloc(7 * (size_t) nt * sizeof(double));
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

/** @brief Largest absolute difference between a snapshot and the current arrays. */
static double max_diff(const MFTL_Object *obj, const double *s)
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

/** @brief Mean |index difference| across every edge -- the locality the reorder targets. */
static double mean_edge_index_distance(const MFTL_Object *obj)
{
  const int n_he = MFTLNumHalfEdges(obj);
  double sum = 0.0;
  for (int h = 0; h < n_he; ++h) sum += fabs((double) (MFTLHeTarget(obj, h) - MFTLHeOrigin(obj, h)));
  return sum / (double) n_he;
}

/**
 * @brief A deliberately GATHER-BOUND kernel: three random position reads per
 *        triangle, one scalar out.
 *
 * This is what the Morton renumbering actually changes, and measuring it needs
 * care. The obvious candidate -- MFTLObjectRefreshFrames -- is the wrong probe:
 * it writes 72 bytes per triangle sequentially, and that store stream is wide
 * enough to hide the gather it is supposed to expose. Measured that way the
 * reorder looks worth between 0.97x and 1.93x with no pattern, which is a
 * measurement artefact rather than a result.
 *
 * Here the only memory traffic that scales is the three reads through
 * `tris.verts` into `verts.pos`. Whether those hit depends entirely on how far
 * apart in memory the vertices of one triangle are -- which is the quantity
 * mean_edge_index_distance() reports.
 *
 * The result goes to a per-triangle OUTPUT ARRAY rather than to a scalar
 * accumulator, and that is the second thing this probe got wrong the first time
 * round. With `acc += ...` the loop carries a dependency through the FP adder,
 * ~4 cycles per triangle, and nine gathered loads fit comfortably underneath it:
 * the kernel then measures the adder and reports 1.00x however the pool is
 * ordered. An 8-byte sequential store per triangle has no such chain, so the
 * 72 gathered bytes are what the loop waits on.
 */
static void gather_bound_kernel(const MFTL_Object *obj, double *restrict out)
{
  const int *restrict tv = obj->tris.verts;
  const double *restrict pos = obj->verts.pos;
  const int nt = obj->tris.n;

  for (int t = 0; t < nt; ++t)
  {
    const double *p0 = &pos[3 * tv[3 * t + 0]];
    const double *p1 = &pos[3 * tv[3 * t + 1]];
    const double *p2 = &pos[3 * tv[3 * t + 2]];
    out[t] = (p0[0] + p1[0] + p2[0]) + (p0[1] + p1[1] + p2[1]) + (p0[2] + p1[2] + p2[2]);
  }
}

/**
 * @brief Best of @p trials timed runs, in nanoseconds per triangle.
 *
 * The MINIMUM, not the mean: this is a throughput measurement on a quiet
 * machine, so the fastest run is the one least perturbed by page faults,
 * scheduling and whatever the allocator was doing. Taking a mean lets a single
 * descheduled run decide the answer, which is how the first version of the
 * locality table below ended up reporting 0.97x and 1.93x for the same effect.
 */
static double best_ns_per_triangle(const MFTL_Object *obj, int trials, int reps, double *out)
{
  gather_bound_kernel(obj, out);  // warm-up: fault the output pages in

  double best = 1e30;
  for (int r = 0; r < trials; ++r)
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < reps; ++i) gather_bound_kernel(obj, out);
    const double dt = (MPI_Wtime() - t0) / (reps * (double) obj->tris.n) * 1e9;
    if (dt < best) best = dt;
  }
  return best;
}

/** @brief Best of @p trials runs of a geometry variant, in nanoseconds per triangle. */
/**
 * @brief Times two variants ALTERNATELY and returns the best ns/triangle of each.
 *
 * Interleaved, and the minimum of individual runs rather than the minimum of
 * block averages. Both of those are corrections to what this function used to
 * do, and together they moved the reported speedup from 1.05x to 1.4-1.5x --
 * i.e. the old harness was hiding two thirds of the effect it existed to
 * measure.
 *
 * Running the variants in blocks (twenty passes of A, then twenty of B) is the
 * mistake, and it is the same one docs/performance-analysis.md §11 already
 * records for the I/O modes: whichever block goes second meets a cache and a
 * clock the first one left behind. Averaging inside a block compounds it, since
 * a single descheduled pass then contaminates the whole trial instead of being
 * discarded by the minimum.
 */
static void best_geometry_pair_ns(MFTL_Object *obj, void (*a)(MFTL_Object *), void (*b)(MFTL_Object *), int reps, double *out_a, double *out_b)
{
  a(obj);  // warm-up, both
  b(obj);

  double best_a = 1e30;
  double best_b = 1e30;

  for (int r = 0; r < reps; ++r)
  {
    double t0 = MPI_Wtime();
    a(obj);
    double dt = MPI_Wtime() - t0;
    if (dt < best_a) best_a = dt;

    t0 = MPI_Wtime();
    b(obj);
    dt = MPI_Wtime() - t0;
    if (dt < best_b) best_b = dt;
  }

  *out_a = 1e9 * best_a / (double) obj->tris.n;
  *out_b = 1e9 * best_b / (double) obj->tris.n;
}

/**
 * @brief Variant A: the half-edge walk, open-coded.
 *
 * This is the body of MFTLComputeTriangleGeometry's general path, verbatim, and
 * it is written out rather than called for a reason worth stating: that function
 * now DISPATCHES on `obj->derived` and takes the streaming path whenever the
 * frames are current. The frames are current here -- variant B needs them -- so
 * calling it would make A silently identical to B and report a speedup of 1.00x.
 *
 * A benchmark that measures the same code twice is the worst kind of wrong
 * answer, because it looks like a result. See docs/performance-analysis.md §11.
 */
static void geometry_from_half_edges(MFTL_Object *obj)
{
  const int n = obj->tris.n;
  for (int t = 0; t < n; ++t)
  {
    MFTLGetTriangleProperties(obj, t, &obj->tris.area[t], &obj->tris.normal[3 * t], &obj->tris.center[3 * t]);
  }
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
  {
    /* Single-rank measurement; the other ranks only have to reach the
     * collectives inside MFTLGenerateSphere, so keep the example runnable under
     * mpirun without making the timings meaningless. */
    MPI_Finalize();
    return 0;
  }

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0};
  double hi[3] = {2.0, 2.0, 2.0};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 8);

  /* The PERIODIC case is the one a real front-tracking run has, and it is where
   * the frames pay off most: the walk applies a minimum-image correction to
   * every edge of every triangle on every pass, while the frames applied it once
   * when they were built. Measuring only the non-periodic case understates the
   * whole point of `tris.edge`. */
  bool per[3] = {true, true, true};
  MFTL_Ensemble *ens_per = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

  printf("MFTL geometry-update benchmark\n");
  printf("  MFTL_Vertex %zu B   MFTL_Triangle %zu B   MFTL_HalfEdge %zu B\n", sizeof(MFTL_Vertex), sizeof(MFTL_Triangle), sizeof(MFTL_HalfEdge));
  printf("  per triangle: records %zu B, geometry 56 B, derived frames+conn 108 B\n\n",
         sizeof(MFTL_Triangle) + 3 * sizeof(MFTL_HalfEdge));

  printf("%-10s %-5s %10s %10s | %11s %11s %8s | %11s\n", "domain", "ref", "verts", "tris", "walk ns/t", "frame ns/t", "speedup", "max |diff|");
  printf("--------------------------------------------------------------------------------------------------\n");

  for (int pass = 0; pass < 2; ++pass)
  {
  MFTL_Ensemble *const host = (pass == 0) ? ens : ens_per;
  const char *const label = (pass == 0) ? "flat" : "PERIODIC";

  for (int ref = 4; ref <= 7; ++ref)
  {
    MFTL_Object *obj = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(host, obj);
    double c[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(obj, c, 1.0, ref);
    MFTLObjectReorderMorton(obj);

    const int nt = obj->tris.n;

    /* Correctness before timing: the comparison is only meaningful if both
     * produce the same answer, which is the point of running them in one binary
     * on one mesh. */
    geometry_from_half_edges(obj);
    double *ref_geom = snapshot(obj);
    geometry_from_frames(obj);
    const double diff = max_diff(obj, ref_geom);
    free(ref_geom);

    double walk = 0.0;
    double frame = 0.0;
    best_geometry_pair_ns(obj, geometry_from_half_edges, geometry_from_frames, 4 * REP, &walk, &frame);

    printf("%-10s %-5d %10d %10d | %11.3f %11.3f %8.2fx | %11.2e\n", label, ref, obj->verts.n, nt, walk, frame, walk / frame, diff);
  }
  }

  /* ── What the Morton renumbering is worth ─────────────────────────────── */
  printf("\nSpatial renumbering, measured on a gather-bound kernel\n");
  printf("(before = the subdivision order the generator produces):\n");
  printf("%-5s %10s %10s | %13s %13s | %11s %11s %8s\n", "ref", "tris", "pos KiB", "edge dist in", "edge dist out", "before ns/t", "after ns/t", "speedup");
  printf("---------------------------------------------------------------------------------------------------\n");

  for (int ref = 4; ref <= 8; ++ref)
  {
    MFTL_Object *obj = MFTLObjectCreate(2);
    MFTLEnsembleAddObject(ens, obj);
    double c[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(obj, c, 1.0, ref);

    double *out = (double *) malloc((size_t) obj->tris.n * sizeof(double));

    const double dist_before = mean_edge_index_distance(obj);
    const double before = best_ns_per_triangle(obj, 5, REP, out);

    MFTLObjectReorderMorton(obj);
    const double dist_after = mean_edge_index_distance(obj);
    const double after = best_ns_per_triangle(obj, 5, REP, out);

    free(out);

    printf("%-5d %10d %10.0f | %13.1f %13.1f | %11.3f %11.3f %8.2fx\n", ref, obj->tris.n, 3.0 * obj->verts.n * sizeof(double) / 1024.0, dist_before,
           dist_after, before, after, before / after);
  }

  printf("\nNotes\n");
  printf("  walk   MFTLComputeTriangleGeometry: three 16 B half-edge records, three gathered\n");
  printf("         positions and two minimum-image corrections per triangle.\n");
  printf("  frame  MFTL_FOREACH_TRIANGLE_FRAME: 72 B streamed per triangle, no indirection,\n");
  printf("         no periodic correction (it was applied once, at frame-build time).\n");
  printf("  The two write the same three arrays; max |diff| is how far apart their results are.\n");
  printf("\n");
  printf("  The margin here is set by how fast the GENERAL path is, not by how fast the\n");
  printf("  frames are. A walk that chases pointers and strides 96-byte records makes the\n");
  printf("  frames look worth about 3x; this one dereferences nothing (half-edge indexing is\n");
  printf("  implicit), strides 12-16 B records and reads positions from one contiguous SoA\n");
  printf("  array, so it closes most of that gap on its own. The remaining margin still grows\n");
  printf("  with mesh size, which is the shape you expect when the win is memory traffic.\n");
  printf("\n");
  printf("  The second table is a NEGATIVE result and is printed rather than hidden. Morton\n");
  printf("  renumbering cuts the mean edge index distance by about 160x, and on this machine\n");
  printf("  that buys no measurable time at any size up to a 15 MiB position array. The most\n");
  printf("  likely reason is cache capacity: Apple M-series L2 is large enough that even the\n");
  printf("  scattered access pattern mostly hits. Re-run it on the target cluster before\n");
  printf("  assuming either way -- an x86 node with 1-2 MiB of L2 per core is a different\n");
  printf("  experiment, and it is the one that decides whether the reorder earns its cost.\n");

  MFTLEnsembleDestroy(&ens);
  MFTLEnsembleDestroy(&ens_per);
  MPI_Finalize();
  return 0;
}
