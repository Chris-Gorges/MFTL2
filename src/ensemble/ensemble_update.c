/**
 * @file ensemble_update.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief The fused per-timestep geometry and quality update.
 *
 * One gathered pass over compact triangle connectivity produces area, normal
 * and centroid AND nine quality accumulators from the same position loads.
 *
 *   1. ONE PASS, NOT TWO. Area/normal/centroid and the edge-length statistics
 *      read the same edges. The three edges of a triangle are e1, e2 and e2-e1,
 *      so the statistics cost three subtractions and three dot products on
 *      vectors already in registers -- against a second pass that re-gathered
 *      eighteen doubles per triangle to recompute them.
 *
 *   2. PERIODICITY IS UNSWITCHED. The periodic-domain decision is outside the
 *      triangle loop. Non-periodic meshes pay no correction instructions;
 *      periodic meshes use branchless minimum-image arithmetic.
 *
 *   3. THE OWNED PREFIX, NOT AN OWNERSHIP TEST. Owned triangles are an array
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
 * The eleven values one body contributes, kept in one contiguous record so that
 * the whole ensemble's statistics are ONE call to MFTLEnsembleReduceBodies()
 * rather than one per body. Mixing the three operators inside one record is what
 * lets the holder route send a single message; see the banner above
 * unpack_slot().
 * @{
 */
/** @brief MIN slots per object: area, quality, squared edge length. */
#define MFTL_UPD_N_MIN 3
/** @brief MAX slots per object: area, quality, squared edge length. */
#define MFTL_UPD_N_MAX 3
/** @brief SUM slots per object: area, quality, squared edge length, triangle count, edge count. */
#define MFTL_UPD_N_SUM 5
/** @brief Slots in one body's reduction record: the three groups, laid out MIN, MAX, SUM. */
#define MFTL_UPD_N_SLOTS (MFTL_UPD_N_MIN + MFTL_UPD_N_MAX + MFTL_UPD_N_SUM)
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
 * the identical kernel: with the if-form (and with a ternary) clang reports
 * "value that could not be identified as reduction", i.e.
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
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLEnsembleUpdate: stale tris.verts -- call MFTLObjectBuildDerived first");
  MFTL_ASSERT(n_tris <= obj->tris.n && "MFTLEnsembleUpdate: count exceeds the triangle pool");

  /* Three contiguous output streams. As fields of MFTL_Triangle they would make
   * this loop a 96-byte-stride scatter, which clang refuses to vectorise with
   * "cannot identify array bounds". */
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

  MFTL_FOREACH_TRIANGLE_GEOMETRY(
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

/* ═══════════════════════════════════════════════════════════════════════════
 * The statistics are PER BODY, and the reduction is over that body's HOLDERS
 *
 * Eleven doubles per body, every timestep. An ensemble-wide aggregate would be
 * eleven doubles full stop, and it is the wrong answer: per-body quality is what
 * a solver reads to decide whether THIS droplet needs remeshing. A single number
 * over every triangle of every body cannot answer that -- one degenerating
 * interface among a thousand healthy ones moves the ensemble minimum and nothing
 * else, so the caller learns that something is wrong and not which body it is.
 *
 * So the payload stays per body. What does NOT have to stay is the set of ranks
 * each body's eleven doubles are reduced over, and that is where this used to
 * cost: three MPI_Allreduce calls over comm_world on a buffer indexed by BODY ID,
 * on every rank, whether or not it held any of that body. The message COUNT was 3
 * regardless of the body count, which is the property that made it look fine --
 * but the VOLUME and the staging memory were both O(bodies in the ensemble) per
 * rank, and in a weak-scaling sweep the body count grows with the rank count.
 *
 * Only the holders of a body have anything to contribute to it, so the reduction
 * goes through MFTLEnsembleReduceBodies(), whose buffer is indexed by LOCAL SLOT
 * and whose collective is a neighbourhood one whenever every body's holders fit
 * inside one 26-neighbourhood. Three consequences worth stating:
 *
 *   - the eleven slots carry THREE DIFFERENT OPERATORS and that is now free. A
 *     neighbourhood exchange combines in local memory, so mixing min, max and sum
 *     costs one message rather than three; the comm_world fallback still needs
 *     one Allreduce per operator, which is exactly what this used to be.
 *   - the identity fill moved inside the reduction. It used to be O(bodies) of
 *     DBL_MAX on every rank so that a body it did not hold contributed nothing to
 *     a minimum; on the holder route no such slot exists to fill.
 *   - a rank holding no bodies still enters every collective, and the guard below
 *     therefore reads n_bodies_global rather than the local count.
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Turns one reduced slot into the nine public statistics.
 *
 * @param[out] qs  Destination.
 * @param      mn  MFTL_UPD_N_MIN reduced minima for this body.
 * @param      mx  MFTL_UPD_N_MAX reduced maxima for this body.
 * @param      sm  MFTL_UPD_N_SUM reduced sums for this body.
 */
static void unpack_slot(MFTL_ObjectQualityStatistics *qs, const double *mn, const double *mx, const double *sm)
{
  const double n_tris_g = sm[3];
  const double n_edges_g = sm[4];

  /* Averages against the GLOBAL counts, never an average of per-rank averages:
   * ranks hold wildly different triangle counts and the two differ. */
  qs->avg_triangle_area = (n_tris_g > 0.0) ? sm[0] / n_tris_g : 0.0;
  qs->avg_aspect_ratio = (n_tris_g > 0.0) ? sm[1] / n_tris_g : 0.0;
  qs->avg_edge_length_sq = (n_edges_g > 0.0) ? sm[2] / n_edges_g : 0.0;

  /* A body with no triangles anywhere leaves the identities in place; zero them
   * so a caller does not read DBL_MAX as a measurement. */
  const bool empty = (n_tris_g <= 0.0);
  qs->min_triangle_area = empty ? 0.0 : mn[0];
  qs->max_triangle_area = empty ? 0.0 : mx[0];
  qs->min_aspect_ratio = empty ? 0.0 : mn[1];
  qs->max_aspect_ratio = empty ? 0.0 : mx[1];
  qs->min_edge_length_sq = empty ? 0.0 : mn[2];
  qs->max_edge_length_sq = empty ? 0.0 : mx[2];
}

void MFTLEnsembleUpdate(MFTL_Ensemble *ensemble)
{
  MFTL_ASSERT(ensemble != MFTL_NULL);
  if (ensemble == MFTL_NULL) return;

  /* GLOBAL, not local. A collective follows, so the guard has to be a number
   * every rank agrees about: returning on the local body count would have a rank
   * holding nothing skip a reduction the others are inside. */
  if (ensemble->n_bodies_global == 0) return;

  const int n_local = ensemble->n_local_objects;

  /* One record per LOCAL body, eleven slots wide:
   *
   *   [0..2]  MIN  area, quality, squared edge length
   *   [3..5]  MAX  the same three
   *   [6..10] SUM  area, quality, squared edge length, triangle count, edge count
   *
   * Indexed by this rank's slot, not by body id, which is the entire scaling
   * argument -- see the banner above unpack_slot(). */
  static const MFTL_BodyReduceOp ops[MFTL_UPD_N_SLOTS] = {
      MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MIN, MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MAX, MFTL_BODY_REDUCE_MAX,
      MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM, MFTL_BODY_REDUCE_SUM};

  double *const rows =
      (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_local * MFTL_UPD_N_SLOTS, 1) * sizeof(double), "MFTLEnsembleUpdate accumulators");

  /* ── The hot loop, once per LOCAL object ───────────────────────────────── */
  for (int o = 0; o < n_local; ++o)
  {
    MFTL_Object *obj = ensemble->local_objects[o];
    MFTL_ASSERT(obj != MFTL_NULL);

    const int n_tris = obj->tris.n_local;

    update_accumulators acc = {MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0, MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0, MFTL_DBL_MAX, -MFTL_DBL_MAX, 0.0};

    /* A body with no owned triangles here keeps the kernel's identities, so it
     * contributes nothing to its own reduction rather than a zero that would win
     * every minimum. There is no longer a slot for a body this rank does not
     * hold at all -- that is the point -- so this is the only such case left. */
    if (n_tris > 0) update_kernel(obj, n_tris, &acc);

    double *const r = &rows[(size_t) o * MFTL_UPD_N_SLOTS];
    r[0] = acc.min_area;
    r[1] = acc.min_q;
    r[2] = acc.min_esq;
    r[3] = acc.max_area;
    r[4] = acc.max_q;
    r[5] = acc.max_esq;
    r[6] = acc.sum_area;
    r[7] = acc.sum_q;
    r[8] = acc.sum_esq;
    r[9] = (double) n_tris;
    r[10] = (double) (3 * n_tris); /* edges visited */
  }

  /* ── One reduction, over each body's holders ─────────────────────────────
   *
   * BLOCKING, and there is nothing left to overlap it with: the loop above is
   * what produces the values, and the unpack below consumes them. The three
   * MPI_Iallreduce calls this replaces were non-blocking for a reason that no
   * longer applies -- they were three independent comm_world reductions whose
   * latencies would otherwise have been in series. Mixing the operators is free
   * on the holder route because it combines in local memory, so there is one
   * message rather than three; the comm_world fallback inside still issues one
   * Iallreduce per operator and waits on them together, which is the old
   * behaviour exactly.
   *
   * There is deliberately no attempt to overlap it with the geometry loop
   * itself: the only available overlap would be per body (start body o's
   * reduction while computing body o+1), and that trades the message count --
   * the thing that actually scales -- for a little latency hiding. Wrong trade. */
  MFTLEnsembleReduceBodies(ensemble, rows, MFTL_UPD_N_SLOTS, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  for (int o = 0; o < n_local; ++o)
  {
    const double *const r = &rows[(size_t) o * MFTL_UPD_N_SLOTS];
    unpack_slot(&ensemble->local_objects[o]->quality, &r[0], &r[3], &r[6]);
  }

  free(rows);
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
  /* GLOBAL, not local: MFTLUpdateGhostVertexNormals() below is collective, so a
   * rank holding no bodies must still reach it. */
  if (ensemble->n_bodies_global == 0) return MFTL_SUCCESS;

  /* Local, communication-free, exact for every OWNED vertex. Ghost vertices come
   * out of this holding a partial sum -- correctly normalised and quietly wrong,
   * which is precisely why the exchange below is not optional. */
  for (int o = 0; o < ensemble->n_local_objects; ++o) MFTLComputeVertexNormals(ensemble->local_objects[o]);

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
 * Historical vectorisation evidence: asked, forced, and priced
 *
 * Everything below predates frame removal and comes from the retired
 * update-kernel benchmark, which rebuilt the frame-backed kernel eleven ways on
 * one mesh in one binary and checked identical results. It explains retained
 * arithmetic choices, but is not a current frame-vs-gather measurement. Current
 * end-to-end measurements live in docs/redistribution_optimizations_review.md.
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
 * The obvious diagnosis is the nine-way statistics reduction. THAT ATTRIBUTION IS
 * WRONG, and two independent measurements say so:
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
 *     separate non-periodic loop is also not worth it here: that variant measures 3.64 against 3.45, i.e. nominally
 *     SLOWER while doing strictly less work, which puts the wrap below this
 *     machine's code-alignment noise floor.
 *   - THE OUTPUT LAYOUT BARELY MATTERS. One array per component measures 3.41
 *     against 3.45 -- 1.3 %, and not worth breaking `tris.normal` into three
 *     arrays for. The layout is chosen for the CONSUMERS, not for this loop:
 *     redistribution wants centres and nothing else, and gets 24 B per triangle
 *     instead of 56.
 * ═══════════════════════════════════════════════════════════════════════════ */
