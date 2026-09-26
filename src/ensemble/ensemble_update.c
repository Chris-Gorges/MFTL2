/**
 * @file ensemble_update.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_ensemble
 *
 * @brief The fused per-timestep geometry + quality pass, and the collective
 *        vertex-normal pass.
 *
 * One gathered pass over the owned prefix of `tris.verts` writes area, normal
 * and centroid and accumulates nine statistics from the same registers (the
 * three edges are e1, e2 and e2-e1); periodicity is decided outside the loop.
 * The kernel is file-static here, not in src/geometry/, so its accumulators stay
 * in registers.
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
 * @name Reduction record layout
 *
 * One body contributes eleven values -- MIN, MAX, SUM groups -- so the whole
 * ensemble's statistics are ONE MFTLEnsembleReduceBodies() call; mixed operators
 * cost nothing on the holder route.
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
 * @brief The nine local accumulators, written back once at the end of the kernel.
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
 * Min/max use `fmin`/`fmax`: with an if-form clang rejects the loop as a
 * reduction before anything else ("value that could not be identified as
 * reduction"). They lower to single instructions on AArch64 but need NaN fix-up
 * on x86-64, so the measured 4-5 % is platform dependent.
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

  /* Three contiguous output streams (SoA). */
  double *const restrict out_area = obj->tris.area;
  double *const restrict out_normal = obj->tris.normal;
  double *const restrict out_center = obj->tris.center;

  /* The centroid wrap stays the branchy MFTLApplyPeriodicity(): a branchless
   * form measured 1.34-1.37x slower in this (non-vectorised) loop. */
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

      /* Unclamped, as in MFTLGetTriangleProperties(): a clamp would floor the
       * area and let q exceed 1. */
      const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
      const double area = 0.5 * mag;

      /* The three edges are e1, e2 and (e2 - e1). */
      const double e3x = e2x - e1x;
      const double e3y = e2y - e1y;
      const double e3z = e2z - e1z;

      const double a_sq = e1x * e1x + e1y * e1y + e1z * e1z;
      const double b_sq = e2x * e2x + e2y * e2y + e2z * e2z;
      const double c_sq = e3x * e3x + e3y * e3y + e3z * e3z;

      /* Mean-ratio quality, dimensionless: the only scale-free degeneracy test
       * is against zero. */
      const double denom = a_sq + b_sq + c_sq;
      const double q = (denom > 0.0) ? MFTL_FOUR_SQRT3 * area / denom : 0.0;

      min_area = fmin(min_area, area); max_area = fmax(max_area, area); sum_area += area;
      min_q = fmin(min_q, q); max_q = fmax(max_q, q); sum_q += q;
      min_esq = fmin(min_esq, fmin(a_sq, fmin(b_sq, c_sq))); max_esq = fmax(max_esq, fmax(a_sq, fmax(b_sq, c_sq))); sum_esq += denom;

      const double inv_mag = (mag > 0.0) ? 1.0 / mag : 0.0; /* exact at every scale; see MFTLGetTriangleProperties */
      out_area[t] = area;
      out_normal[3 * t + 0] = nx * inv_mag;
      out_normal[3 * t + 1] = ny * inv_mag;
      out_normal[3 * t + 2] = nz * inv_mag;

      /* Centroid from seam-consistent offsets, then wrapped (MFTLGetOwnerProc()). */
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

/**
 * @brief Turns one reduced record into the nine public statistics.
 *
 * Averages divide by GLOBAL counts (never an average of per-rank averages). A
 * body with no triangles anywhere reports zeros instead of the identities.
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

  /* Averages against the GLOBAL counts. */
  qs->avg_triangle_area = (n_tris_g > 0.0) ? sm[0] / n_tris_g : 0.0;
  qs->avg_aspect_ratio = (n_tris_g > 0.0) ? sm[1] / n_tris_g : 0.0;
  qs->avg_edge_length_sq = (n_edges_g > 0.0) ? sm[2] / n_edges_g : 0.0;

  /* No triangles anywhere: report zeros, not the DBL_MAX identities. */
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

  /* Guard on the GLOBAL count: a collective follows. */
  if (ensemble->n_bodies_global == 0) return;

  const int n_local = ensemble->n_local_objects;

  /* One record per LOCAL body, indexed by slot:
   *
   *   [0..2]  MIN  area, quality, squared edge length
   *   [3..5]  MAX  the same three
   *   [6..10] SUM  area, quality, squared edge length, triangle count, half-edge count */
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

    /* No owned triangles here: contribute the identities, not zeros. */
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

  /* One blocking reduction over each body's holders for the whole ensemble
   * (per-body overlap would multiply the message count). */
  MFTLEnsembleReduceBodies(ensemble, rows, MFTL_UPD_N_SLOTS, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  for (int o = 0; o < n_local; ++o)
  {
    const double *const r = &rows[(size_t) o * MFTL_UPD_N_SLOTS];
    unpack_slot(&ensemble->local_objects[o]->quality, &r[0], &r[3], &r[6]);
  }

  free(rows);
}

void MFTLEnsembleComputeTriangleGeometry(MFTL_Ensemble *ensemble)
{
  MFTL_ASSERT(ensemble != MFTL_NULL);

  for (int o = 0; o < ensemble->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ensemble->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

/* The per-vertex counterpart needs an EXCHANGE, not a reduction: owned normals
 * are exact locally (vertex-complete halo); only the ghost half is missing. */

int MFTLEnsembleUpdateVertexNormals(MFTL_Ensemble *ensemble, MFTL_NormalWeighting weighting)
{
  MFTL_ASSERT(ensemble != MFTL_NULL);
  if (ensemble == MFTL_NULL) return MFTL_ERROR;
  /* Guard on the GLOBAL count: the exchange below is collective. */
  if (ensemble->n_bodies_global == 0) return MFTL_SUCCESS;

  /* Exact on owned vertices; partial sums on ghosts until the exchange. */
  for (int o = 0; o < ensemble->n_local_objects; ++o) MFTLComputeVertexNormals(ensemble->local_objects[o], weighting);

  /* One request/reply exchange for the whole ensemble (per object would
   * multiply the message count for the same bytes). */
  return MFTLUpdateGhostVertexNormals(ensemble);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Why this loop does not vectorise (docs/analysis.md §1.6)
 *
 * clang's cost model rejects it (-Rpass-missed=loop-vectorize), and forcing it
 * with `#pragma clang loop vectorize(enable)` fails outright. The nine-way
 * reduction is NOT the cause: splitting it into independent accumulator sets
 * changed nothing. One sqrt and one reciprocal per triangle are about a quarter
 * of the pass and neither is removable (the kernel needs both |n| and n/|n|).
 * Measured traps: splitting the statistics into a second pass is 25 % slower;
 * a per-component output layout gains 1.3 %, not worth breaking `tris.normal`.
 * ═══════════════════════════════════════════════════════════════════════════ */
