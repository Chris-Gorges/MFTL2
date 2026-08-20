/**
 * @file ensemble_update.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief The fused per-timestep geometry and quality update.
 *
 * One streaming pass over the triangle frames produces area, normal and centroid
 * AND nine quality accumulators, from the same loads. Four properties of the
 * loop are what make it cheap, and only the last is a micro-optimisation:
 *
 *   1. ONE PASS, NOT TWO. Area/normal/centroid and the edge-length statistics
 *      read the same edges. The three edges of a triangle are e1, e2 and e2-e1,
 *      so the statistics cost three subtractions and three dot products on
 *      vectors already in registers -- against a second pass that re-gathered
 *      eighteen doubles per triangle to recompute them.
 *
 *   2. NO MINIMUM-IMAGE WORK IN THE LOOP. `tris.edge` carries the seam
 *      correction already applied (src/object/derived.c), so it is paid once per
 *      frame rebuild instead of nine comparisons and eighteen fused operations
 *      per triangle per timestep.
 *
 *   3. NO GATHER AT ALL. Two contiguous arrays in (`tris.edge`, `tris.anchor`),
 *      three contiguous arrays out (`tris.area`, `tris.normal`, `tris.center`).
 *      Nothing in the loop dereferences anything.
 *
 *   4. THE OWNED PREFIX, NOT AN OWNERSHIP TEST. Owned triangles are an array
 *      prefix, so the loop bound IS the ownership filter.
 */

#include "mftl_ensemble.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_types.h"

/**
 * @name Reduction slot counts
 *
 * Per object, kept in one contiguous block so that the whole ensemble's
 * statistics are THREE `MPI_Allreduce` calls -- one per operator -- rather than
 * three per object. At \f$B\f$ bodies that is 3 collectives instead of
 * \f$3B\f$, and the message is \f$11B\f$ doubles either way.
 * @{
 */
/** @brief MIN slots per object: area, quality, squared edge length. */
#define MFTL_UPD_N_MIN 3
/** @brief MAX slots per object: area, quality, squared edge length. */
#define MFTL_UPD_N_MAX 3
/** @brief SUM slots per object: area, quality, squared edge length, triangle count, edge count. */
#define MFTL_UPD_N_SUM 5
/** @} */

/**
 * @brief The nine local accumulators, so the kernel has one out-parameter.
 *
 * Passed by pointer and written back ONCE, at the end of the kernel. The nine
 * values stay in registers across the loop, which is the whole reason the kernel
 * is a file-static function here rather than in `src/geometry/`: crossing a
 * translation-unit boundary would force them into an address-taken struct, and
 * that was measured at 15 %.
 */
typedef struct
{
  double min_area;  ///< Smallest triangle area seen.
  double max_area;  ///< Largest triangle area seen.
  double sum_area;  ///< Total area.
  double min_q;  ///< Worst mean-ratio quality seen.
  double max_q;  ///< Best mean-ratio quality seen.
  double sum_q;  ///< Summed quality.
  double min_esq;  ///< Shortest squared edge length seen.
  double max_esq;  ///< Longest squared edge length seen.
  double sum_esq;  ///< Summed squared edge length.
} update_accumulators;

/**
 * @brief The hot loop: geometry into the triangle pool, statistics into @p acc.
 *
 * The accumulators use fmin/fmax rather than the obvious `if (a < lo) lo = a;`,
 * and that is a measured choice. Checked with -Rpass-analysis=loop-vectorize on
 * the identical kernel in the previous version: with the if-form (and with a
 * ternary) clang reports "value that could not be identified as reduction", i.e.
 * the loop is disqualified before anything else is considered. With fmin/fmax
 * that objection disappears.
 *
 * Caveat for the cluster: on AArch64 fmin/fmax lower to single fminnm/fmaxnm
 * instructions; on x86-64 they need extra NaN fixup around minsd/maxsd, so the
 * 4-5 % measured is not portable and is worth re-measuring there.
 *
 * @param      obj    The mesh. `tris.area`, `tris.normal` and `tris.center` are
 *                    written for the first @p n_tris triangles.
 * @param      n_tris Triangles to process, from index 0.
 * @param[out] acc    The nine running accumulators, written back once at the end.
 */
static void update_kernel(MFTL_Object *obj, int n_tris, update_accumulators *acc)
{
  MFTL_ASSERT(MFTLObjectHasFrames(obj) && "MFTLEnsembleUpdate: stale frames -- MFTLObjectBuildDerived or MFTLObjectRefreshFrames first");
  MFTL_ASSERT(n_tris <= obj->tris.n && "MFTLEnsembleUpdate: count exceeds the triangle pool");

  /* Three contiguous output streams. In the previous version these were fields
   * of MFTL_Triangle and the loop was a 96-byte-stride scatter that clang
   * refused to vectorise with "cannot identify array bounds". */
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  /* The wrap stays the BRANCHY library call, and that is measured, not assumed.
   * A hoisted branchless form (multiply by a per-axis length that is zero off a
   * periodic axis) is the standard advice and it was 1.34-1.37x SLOWER here: the
   * three branches are perfectly predicted, while the branchless form does
   * unconditional work on all three axes including non-periodic ones. It would
   * be the right choice if this loop vectorised -- it does not, see the note at
   * the bottom of this file. */
  const MFTL_Domain *const dom = obj->domain;

  double min_area = acc->min_area, max_area = acc->max_area, sum_area = 0.0;
  double min_q = acc->min_q, max_q = acc->max_q, sum_q = 0.0;
  double min_esq = acc->min_esq, max_esq = acc->max_esq, sum_esq = 0.0;

  MFTL_FOREACH_TRIANGLE_FRAME(
      obj, n_tris,

      /* Unnormalised normal; its magnitude is twice the area. */
      const double nx = e1y * e2z - e1z * e2y;
      const double ny = e1z * e2x - e1x * e2z;
      const double nz = e1x * e2y - e1y * e2x;

      /* No clamp on the magnitude -- see MFTLGetTriangleProperties. Flooring it
       * would floor `area` at 5e-7, which is precisely the quantity this kernel
       * exists to reduce: min_triangle_area could then never report a degenerate
       * triangle, and the quality below would exceed its own maximum of 1. The
       * unit-normal guarantee costs this loop +1.1 %, measured; that is the whole
       * of what the exactness is priced at. */
      const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
      const double area = 0.5 * mag;

      /* The three edges are e1, e2 and (e2 - e1). */
      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      /* Mean-ratio quality: 1 for an equilateral triangle, 0 for a degenerate one. */
      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > MFTL_VERYSMALL) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;

      const double inv_mag = (mag > 0.0) ? 1.0 / mag : 0.0; /* exact at every scale; see MFTLGetTriangleProperties */
      out_area[t] = area;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      /* Centroid = corner0 + (e1 + e2)/3. The offset is seam-consistent, so this
       * is the true centroid even for a triangle straddling a periodic face --
       * but the result must be wrapped back into the box, because redistribution
       * feeds it to MFTLGetOwnerProc. */
      double c[3] = {p0x + (e1x + e2x) * MFTL_ONE_THIRD, p0y + (e1y + e2y) * MFTL_ONE_THIRD, p0z + (e1z + e2z) * MFTL_ONE_THIRD};
      MFTLApplyPeriodicity(dom, c);
      out_center[3 * t + 0] = c[0];
      out_center[3 * t + 1] = c[1];
      out_center[3 * t + 2] = c[2];);

  acc->min_area = min_area;
  acc->max_area = max_area;
  acc->sum_area = sum_area;
  acc->min_q = min_q;
  acc->max_q = max_q;
  acc->sum_q = sum_q;
  acc->min_esq = min_esq;
  acc->max_esq = max_esq;
  acc->sum_esq = sum_esq;
}

void MFTLEnsembleUpdate(MFTL_Ensemble *ensemble)
{
  MFTL_ASSERT(ensemble != MFTL_NULL);
  if (ensemble == MFTL_NULL || ensemble->num_objects == 0) return;

  const int n_obj = ensemble->num_objects;

  /* One allocation for six logical arrays. Layout:
   *   [ local_min | local_max | local_sum | global_min | global_max | global_sum ]
   * so each reduction reads one contiguous range and writes another. */
  const size_t block = (size_t) n_obj * (MFTL_UPD_N_MIN + MFTL_UPD_N_MAX + MFTL_UPD_N_SUM);
  double *buf = (double *) MFTLMallocOrDie(2 * block * sizeof(double), "MFTLEnsembleUpdate accumulators");

  double *const local_min = buf;
  double *const local_max = local_min + (size_t) n_obj * MFTL_UPD_N_MIN;
  double *const local_sum = local_max + (size_t) n_obj * MFTL_UPD_N_MAX;
  double *const global_min = local_sum + (size_t) n_obj * MFTL_UPD_N_SUM;
  double *const global_max = global_min + (size_t) n_obj * MFTL_UPD_N_MIN;
  double *const global_sum = global_max + (size_t) n_obj * MFTL_UPD_N_MAX;

  /* ── The hot loop, once per object ─────────────────────────────────────── */
  for (int o = 0; o < n_obj; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    MFTL_ASSERT(obj != MFTL_NULL);

    const int n_tris = obj->tris.n_local;

    /* Identities for MIN and MAX, so a rank holding none of this object
     * contributes nothing to the reduction rather than a zero that would win
     * every minimum. */
    update_accumulators acc = {MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0, MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0, MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0};

    if (n_tris > 0) update_kernel(obj, n_tris, &acc);

    local_min[o * MFTL_UPD_N_MIN + 0] = acc.min_area;
    local_min[o * MFTL_UPD_N_MIN + 1] = acc.min_q;
    local_min[o * MFTL_UPD_N_MIN + 2] = acc.min_esq;

    local_max[o * MFTL_UPD_N_MAX + 0] = acc.max_area;
    local_max[o * MFTL_UPD_N_MAX + 1] = acc.max_q;
    local_max[o * MFTL_UPD_N_MAX + 2] = acc.max_esq;

    local_sum[o * MFTL_UPD_N_SUM + 0] = acc.sum_area;
    local_sum[o * MFTL_UPD_N_SUM + 1] = acc.sum_q;
    local_sum[o * MFTL_UPD_N_SUM + 2] = acc.sum_esq;
    local_sum[o * MFTL_UPD_N_SUM + 3] = (double) n_tris;
    local_sum[o * MFTL_UPD_N_SUM + 4] = (double) (3 * n_tris); /* edges visited */
  }

  /* ── Three reductions, issued together ──────────────────────────────────
   *
   * Every object of the ensemble is inside each of them, so the MESSAGE COUNT is
   * 3 whether the ensemble holds one droplet or ten thousand. That is the number
   * that decides how this scales; the volume (11 doubles per object) is
   * negligible next to the latency at any realistic P.
   *
   * Non-blocking, and this is the one place the previous version left something
   * on the table. The three reductions are independent -- different operators
   * over disjoint buffers -- so issuing them back to back as blocking calls puts
   * three O(log P) latencies in series for no reason. MPI_Iallreduce lets the
   * progress engine pipeline them, so the cost tends towards one collective's
   * latency rather than three. At P = 8 that is not measurable; at P = 10k, where
   * an Allreduce is tens of microseconds and this runs every timestep, it is the
   * difference between 3x and 1x that cost.
   *
   * There is deliberately no attempt to overlap them with the geometry loop
   * itself: the loop is what produces the values being reduced, so the only
   * available overlap would be per-object (start object o's reduction while
   * computing object o+1), and that trades the message count -- the thing that
   * actually scales -- for a little latency hiding. Wrong trade. */
  MPI_Comm comm = ensemble->domain.comm_world;
  MPI_Request reqs[3];
  MPI_Iallreduce(local_min, global_min, n_obj * MFTL_UPD_N_MIN, MPI_DOUBLE, MPI_MIN, comm, &reqs[0]);
  MPI_Iallreduce(local_max, global_max, n_obj * MFTL_UPD_N_MAX, MPI_DOUBLE, MPI_MAX, comm, &reqs[1]);
  MPI_Iallreduce(local_sum, global_sum, n_obj * MFTL_UPD_N_SUM, MPI_DOUBLE, MPI_SUM, comm, &reqs[2]);
  MPI_Waitall(3, reqs, MPI_STATUSES_IGNORE);

  /* ── Unpack ─────────────────────────────────────────────────────────────── */
  for (int o = 0; o < n_obj; ++o)
  {
    const double n_tris_g = global_sum[o * MFTL_UPD_N_SUM + 3];
    const double n_edges_g = global_sum[o * MFTL_UPD_N_SUM + 4];

    MFTL_ObjectQualityStatistics *qs = &ensemble->objects[o]->quality;

    /* Averages against the GLOBAL counts, never an average of per-rank averages:
     * ranks hold wildly different triangle counts and the two differ. */
    qs->avg_triangle_area = (n_tris_g > 0.0) ? global_sum[o * MFTL_UPD_N_SUM + 0] / n_tris_g : 0.0;
    qs->avg_aspect_ratio = (n_tris_g > 0.0) ? global_sum[o * MFTL_UPD_N_SUM + 1] / n_tris_g : 0.0;
    qs->avg_edge_length_sq = (n_edges_g > 0.0) ? global_sum[o * MFTL_UPD_N_SUM + 2] / n_edges_g : 0.0;

    /* An object with no triangles anywhere leaves the identities in place; zero
     * them so a caller does not read DBL_MAX as a measurement. */
    const bool empty = (n_tris_g <= 0.0);
    qs->min_triangle_area = empty ? 0.0 : global_min[o * MFTL_UPD_N_MIN + 0];
    qs->max_triangle_area = empty ? 0.0 : global_max[o * MFTL_UPD_N_MAX + 0];
    qs->min_aspect_ratio = empty ? 0.0 : global_min[o * MFTL_UPD_N_MIN + 1];
    qs->max_aspect_ratio = empty ? 0.0 : global_max[o * MFTL_UPD_N_MAX + 1];
    qs->min_edge_length_sq = empty ? 0.0 : global_min[o * MFTL_UPD_N_MIN + 2];
    qs->max_edge_length_sq = empty ? 0.0 : global_max[o * MFTL_UPD_N_MAX + 2];
  }

  free(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The per-vertex counterpart
 *
 * MFTLEnsembleUpdate is to triangles what this is to vertices, and the two are
 * deliberately the same shape: iterate the object list, run the local kernel
 * from src/geometry/, then own whatever collective the result needs. What
 * differs is WHICH collective, and that difference is the interesting part.
 *
 *   triangles   the statistics are global by definition -- a minimum over one
 *               rank's partition is not the mesh's minimum -- so the collective
 *               is a REDUCTION, and it is the reason the kernel and the
 *               orchestration live in the same file (see the note at the very
 *               bottom of this file).
 *
 *   vertices    the normals need no reduction at all: an owned vertex's normal
 *               is a function of its one-ring, which a vertex-complete halo makes
 *               entirely local. What IS missing is the ghost half, so the
 *               collective is an EXCHANGE.
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLEnsembleUpdateVertexNormals(MFTL_Ensemble *ensemble)
{
  MFTL_ASSERT(ensemble != MFTL_NULL);
  if (ensemble == MFTL_NULL) return MFTL_ERROR;
  if (ensemble->num_objects == 0) return MFTL_SUCCESS;

  /* Local, communication-free, exact for every OWNED vertex. Ghost vertices come
   * out of this holding a partial sum -- correctly normalised and quietly wrong,
   * which is precisely why the exchange below is not optional. */
  for (int o = 0; o < ensemble->num_objects; ++o) MFTLComputeVertexNormals(ensemble->objects[o]);

  /* One request/reply over the graph communicator, 32 B per ghost vertex. It is
   * blocking inside, and there is nothing here to overlap it with: every object
   * has to be accumulated before any reply can be served, because a ghost on
   * this rank may be owned by a rank whose own accumulation is not finished
   * either. Issuing it per object would multiply the message COUNT by the number
   * of bodies while moving the same bytes -- the exchange takes the whole
   * ensemble at once for exactly that reason. */
  return MFTLUpdateGhostVertexNormals(ensemble);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vectorisation: asked, forced, and priced
 *
 * Everything below is from examples/update_kernel_benchmark.c, which rebuilds
 * this kernel eleven different ways on one mesh in one binary and checks that
 * they all produce the same three arrays and the same nine statistics. Numbers
 * are ns per triangle, best of nine runs, Apple M-series, Release + LTO, and
 * they are flat from 5 120 to 1 310 720 triangles -- so this loop is NOT memory
 * bound at any size that fits in RAM. Re-measure on the cluster.
 *
 * ── 1. Does the loop vectorise?  No, and not for the reason believed ───────
 *
 *   mpicc -std=c11 -O3 -DNDEBUG -Iinclude \
 *         -c src/ensemble/ensemble_update.c -o /dev/null \
 *         -Rpass=loop-vectorize -Rpass-missed=loop-vectorize \
 *         -Rpass-analysis=loop-vectorize
 *
 * says of the loop below: "the cost-model indicates that vectorization is not
 * beneficial". Note -Rpass-MISSED: without it clang reports nothing at all for
 * this loop and only remarks on the unpack loop at the bottom of
 * MFTLEnsembleUpdate, which is easy to misread as "no diagnostic, so it
 * vectorised".
 *
 * The previous version's note attributed that verdict to the nine-way statistics
 * reduction. THAT ATTRIBUTION WAS WRONG, and two independent measurements say
 * so:
 *
 *   - splitting the reduction into 2 independent accumulator sets -- the
 *     standard fix, written out with eighteen plain scalars so nothing is hidden
 *     in a struct -- gives 3.52 against this kernel's 3.45. A wash, very
 *     slightly worse. Four sets is no better.
 *   - overruling the cost model with `#pragma clang loop vectorize(enable)`
 *     produces "loop not vectorized: the optimizer was unable to perform the
 *     requested transformation", and times identically to this kernel. So the
 *     loop is not merely judged unprofitable, it is not vectorisable in this
 *     shape at all -- and the profitable-looking transformation is not.
 *
 * ── 2. Where the time actually goes ────────────────────────────────────────
 *
 *   this kernel, fused                        3.45
 *   geometry only, no statistics              2.26   <- statistics cost 1.19
 *   geometry only, sqrt and divide DELETED    1.36   <- they cost 0.90, i.e. 26 %
 *
 * One sqrt and one reciprocal per triangle are a quarter of the whole pass, and
 * neither is algebraically removable: the kernel needs both |n| (for the area)
 * and n/|n| (for the unit normal). That is the floor this loop sits on, and it
 * is a throughput limit on one functional unit -- which is precisely why
 * unrolling the reductions changes nothing.
 *
 * ── 3. Three traps for anyone editing this loop, all measured ──────────────
 *
 *   - DO NOT SPLIT THE STATISTICS OUT INTO THEIR OWN PASS. Geometry then
 *     statistics as two loops costs 4.33 against 3.45 fused -- 25 % slower, even
 *     though the second pass re-reads data the first left in cache. The
 *     statistics ride along on vectors already in registers; separating them
 *     means recomputing the cross product and the three edge lengths.
 *   - KEEP THE WRAP BRANCHY, AND DO NOT UNSWITCH IT EITHER. The branchless form
 *     is the textbook advice and was 1.34-1.37x SLOWER. Hoisting a separate
 *     non-periodic loop the way src/object/derived.c does for the frames is also
 *     not worth it here: that variant measures 3.64 against 3.45, i.e. nominally
 *     SLOWER while doing strictly less work, which puts the wrap below this
 *     machine's code-alignment noise floor.
 *   - THE OUTPUT LAYOUT BARELY MATTERS. One array per component measures 3.41
 *     against 3.45 -- 1.3 %, and not worth breaking `tris.normal` into three
 *     arrays for. The layout is chosen for the CONSUMERS, not for this loop:
 *     redistribution wants centres and nothing else, and gets 24 B per triangle
 *     instead of 56.
 * ═══════════════════════════════════════════════════════════════════════════ */
