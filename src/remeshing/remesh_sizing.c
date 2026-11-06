/**
 * @file remesh_sizing.c
 * @author Christian Gorges
 * @date 2026-10-23
 * @ingroup mftl_remeshing
 *
 * @brief The DECISION MAKERS of the two sizing modes: what length an edge
 *        should have, and whether a split or collapse is wanted.
 *
 * ## One switch, read in one place
 *
 * MFTL_RemeshParameters::sizing is read by the predicates in this file and by
 * the smoother's targets, and by nothing else. Every caller that judges an edge
 * -- the split and collapse predicates, the collapse's stretch clause, and the
 * spanning driver's convergence certificate -- goes through
 * MFTLRemeshEdgeWantsSplit() / MFTLRemeshEdgeWantsCollapse(), so the operators,
 * the sweeps and the certificate cannot disagree about what "too long" means.
 * The UNIFORM branches are the expressions the remesher had before sizing
 * existed, verbatim, which is what keeps every uniform digest bit-identical.
 *
 * ## Curvature (Dunyach et al. 2013)
 *
 * A per-vertex target L(v) is computed ONCE per call of the remeshing kernel,
 * on the mesh that call is handed (MFTLRemeshPrepareSizing()), from the paper's
 * own curvature operators, and is then CARRIED: a split interpolates it to the
 * new vertex, a collapse to the survivor, compaction permutes it. Recomputing it
 * after every operation (what the old mftl did) feeds the chord midpoints the
 * operators create back into the curvature, and costs a fan walk per test.
 *
 * Jiao et al.'s anisotropic mode (2010) lived here too, with quadric-error
 * vertex placement. Both were measured over the LeVeque period and removed
 * (studies/leveque_remeshing/REPORT.md, docs/analysis.md §14.9).
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_comm.h"
#include "remesh_fields.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Validity
 * ═══════════════════════════════════════════════════════════════════════════ */

const char *MFTLRemeshSizingProblem(const MFTL_RemeshParameters *params)
{
  /* Negated comparisons throughout, so a NaN anywhere is refused rather than
   * admitted. Placement and projection are checked here too: this is the one
   * verdict every entry point and the distributed agreement read. */
  if (params->split_placement < MFTL_REMESH_PLACE_MIDPOINT || params->split_placement > MFTL_REMESH_PLACE_PARABOLOID_NORMALS) return "unknown split_placement";
  if (params->collapse_placement < MFTL_REMESH_PLACE_MIDPOINT || params->collapse_placement > MFTL_REMESH_PLACE_PARABOLOID_NORMALS)
    return "unknown collapse_placement";
  if ((params->split_placement != MFTL_REMESH_PLACE_MIDPOINT || params->collapse_placement != MFTL_REMESH_PLACE_MIDPOINT) && !(params->fit_radius > 1.0))
    return "a paraboloid placement needs fit_radius > 1";
  if (params->projection < MFTL_REMESH_PROJECT_NONE || params->projection > MFTL_REMESH_PROJECT_PHONG) return "unknown projection";

  switch (params->sizing)
  {
    case MFTL_REMESH_SIZING_UNIFORM: return MFTL_NULL;

    case MFTL_REMESH_SIZING_CURVATURE:
      if (!(params->curvature_tolerance > 0.0)) return "curvature sizing needs curvature_tolerance > 0 (a length)";
      if (!(params->min_target_length > 0.0 && params->min_target_length <= params->target_length))
        return "curvature sizing needs 0 < min_target_length <= target_length";
      break;

    default: return "unknown sizing mode";
  }

  /* The sized smoother weights incident TRIANGLES (Dunyach Eq. 6); it has no
   * vertex-based form. */
  if (params->smooth_mode != MFTL_SMOOTH_TRIANGLE_BASED) return "curvature sizing needs smooth_mode = triangle";
  return MFTL_NULL;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. The curvature sizing field
 * ═══════════════════════════════════════════════════════════════════════════ */

double MFTLRemeshCurvatureSizing(double kappa, const MFTL_RemeshParameters *params)
{
  const double eps = params->curvature_tolerance;
  const double l_min = params->min_target_length;
  const double l_max = params->target_length;

  /* L^2 = 6 eps / kappa - 3 eps^2, compared as kappa L^2 = 6 eps - 3 eps^2 kappa
   * so nothing is divided before the clamps have decided: a flat vertex
   * (kappa = 0) is the coarsest target, a vertex past kappa = 2 / eps (where the
   * formula's L^2 turns negative) the finest. No NaN can come out. */
  if (!(kappa > 0.0)) return l_max;
  const double k_l2 = 6.0 * eps - 3.0 * eps * eps * kappa;
  if (!(k_l2 > l_min * l_min * kappa)) return l_min;
  if (k_l2 >= l_max * l_max * kappa) return l_max;
  return MFTL_SQRT(k_l2 / kappa);
}

/**
 * @brief Does @p v take its target from its neighbours rather than from its own
 *        curvature?
 *
 * A vertex on a feature curve or corner, and a vertex on an open surface's rim:
 * the discrete curvature there is a crease (unbounded) or a half fan (not a
 * curvature at all), and taking it at face value would refine every feature
 * line to the floor (Dunyach et al., Sect. 3, item 1).
 *
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` for a feature, corner or boundary vertex.
 */
static inline bool sizing_from_neighbours(const MFTL_Object *obj, int v)
{
  return (obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER | MFTL_FLAG_BOUNDARY)) != 0;
}

int MFTLRemeshSizingBegin(MFTL_Object *obj, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);
  if (MFTLRemeshSizingProblem(params) != MFTL_NULL) return MFTL_ERROR;
  if (params->sizing != MFTL_REMESH_SIZING_CURVATURE) return MFTL_SUCCESS;

  MFTLObjectEnableSizing(obj);
  const int n_v = obj->verts.n;
  if (n_v == 0 || obj->tris.n == 0) return MFTL_SUCCESS;

  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  /* The feature classification decides which vertices are sized by their
   * neighbours, so it has to describe this mesh -- the same reason the kernel
   * re-detects before every sweep that reads it. */
  MFTLDetectFeatures(obj, params->feature_angle_threshold, params->feature_corner_angle, MFTL_NULL);

  double *const hk = (double *) MFTLMallocOrDie(2 * (size_t) n_v * sizeof(double), "remesh sizing curvature");
  MFTLComputeVertexCurvatures(obj, hk, hk + n_v);

  /* The RAW maximum absolute curvature, parked in the sizing array until
   * MFTLRemeshSizingFinish() turns it into a length: the second half reads
   * neighbours' curvatures, and on a partitioned mesh those must have been
   * exchanged first -- the array is what the exchange carries. */
  double *const kappa = obj->verts.sizing;
  for (int v = 0; v < n_v; ++v)
  {
    const double H = hk[v];
    const double K = hk[n_v + v];
    /* kappa = H + sqrt(H^2 - K) (Dunyach Eq. 5). The two discrete estimates are
     * independent, so H^2 - K can come out negative; it is clamped, not NaN. */
    const double disc = H * H - K;
    kappa[v] = H + ((disc > 0.0) ? MFTL_SQRT(disc) : 0.0);
  }

  free(hk);
  return MFTL_SUCCESS;
}

void MFTLRemeshSizingFinish(MFTL_Object *obj, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);
  if (params->sizing != MFTL_REMESH_SIZING_CURVATURE || !obj->verts.has_sizing) return;

  const int n_v = obj->verts.n;
  if (n_v == 0 || obj->tris.n == 0) return;

  /* One Jacobi pass over the one-ring, on the curvature, before the target is
   * taken from it -- what pmp-library's reference implementation of the paper
   * does (one smoothing iteration of the curvature tensor), and for a reason
   * the drivers make visible. An edge's target is the MINIMUM of its ends', and
   * the minimum of two noisy estimates is biased low: unsmoothed, a remeshed
   * mesh reads ~6 % more curvature than the mesh it came from, so every pass that
   * recomputes the field (each timestep; each phase of the spanning driver)
   * refines a little further. A feature, corner or rim vertex has no curvature
   * of its own to keep: it takes the mean of its non-feature neighbours
   * (Dunyach et al., Sect. 3, item 1).
   *
   * Jacobi -- every value from the RAW copy -- so the result does not depend on
   * the pool order. */
  double *const size = obj->verts.sizing;
  double *const raw = (double *) MFTLMallocOrDie((size_t) n_v * sizeof(double), "remesh sizing raw curvature");
  memcpy(raw, size, (size_t) n_v * sizeof(double));
  int ring[MFTL_REMESH_RING_MAX];

  for (int v = 0; v < n_v; ++v)
  {
    const bool feature = sizing_from_neighbours(obj, v);
    double sum = feature ? 0.0 : raw[v];
    int count = feature ? 0 : 1;

    if (obj->verts.rec[v].out_he_idx != MFTL_ERROR)
    {
      const int n = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_REMESH_RING_MAX, MFTL_FALSE);
      for (int i = 0; i < n; ++i)
      {
        if (sizing_from_neighbours(obj, ring[i])) continue;
        sum += raw[ring[i]];
        count++;
      }
    }

    /* A feature vertex with no smooth neighbour (a thin strip of features) has
     * nothing to borrow from: zero curvature, the coarsest target, which does not
     * refine a crease to the floor. */
    size[v] = MFTLRemeshCurvatureSizing((count > 0) ? sum / (double) count : 0.0, params);
  }

  free(raw);
}

int MFTLRemeshPrepareSizing(MFTL_Object *obj, const MFTL_RemeshParameters *params)
{
  if (MFTLRemeshSizingBegin(obj, params) != MFTL_SUCCESS) return MFTL_ERROR;
  MFTLRemeshSizingFinish(obj, params);
  return MFTL_SUCCESS;
}

int MFTLRemeshEnsemblePrepareSizing(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(ens != MFTL_NULL && params != MFTL_NULL);

  /* The mode and the validity verdict are functions of the block alone, so
   * every rank handed the same block takes the same branch here -- including a
   * rank holding no body, which still has to enter the two exchanges. */
  if (MFTLRemeshSizingProblem(params) != MFTL_NULL) return MFTL_ERROR;
  if (params->sizing != MFTL_REMESH_SIZING_CURVATURE) return MFTL_SUCCESS;

  int status = MFTL_SUCCESS;

  /* Owned vertices are exact after each local half (the halo is
   * vertex-complete); each exchange then gives the ghost copies their owners'
   * values. Twice, because the second half -- feature vertices averaging their
   * neighbours -- reads neighbours, which may be ghosts. */
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (MFTLRemeshSizingBegin(ens->local_objects[o], params) != MFTL_SUCCESS) status = MFTL_ERROR;
  if (MFTLRemeshSyncGhostSizing(ens) != MFTL_SUCCESS) status = MFTL_ERROR;

  for (int o = 0; o < ens->n_local_objects; ++o) MFTLRemeshSizingFinish(ens->local_objects[o], params);
  if (MFTLRemeshSyncGhostSizing(ens) != MFTL_SUCCESS) status = MFTL_ERROR;

  return status;
}

/**
 * @brief The curvature target of edge a-b: the smaller of its ends'
 *        (Dunyach et al.: "conservatively").
 * @param obj    The mesh, with `verts.sizing` prepared.
 * @param a      First endpoint.
 * @param b      Second endpoint.
 * @param params Supplies the fallback when the field is missing.
 * @return \f$L(e) = \min(L(a), L(b))\f$.
 */
static inline double edge_target(const MFTL_Object *obj, int a, int b, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(obj->verts.has_sizing && "curvature sizing: verts.sizing is missing -- MFTLRemeshPrepareSizing first");
  if (!obj->verts.has_sizing) return params->target_length;
  return MFTL_MIN(obj->verts.sizing[a], obj->verts.sizing[b]);
}

double MFTLRemeshMergedTarget(const MFTL_Object *obj, int a, int b, const double pm[3])
{
  const double f = MFTLRemeshFieldEdgeParameter(obj, a, b, pm);
  return (1.0 - f) * obj->verts.sizing[a] + f * obj->verts.sizing[b];
}

void MFTLRemeshSizingCarry(MFTL_Object *obj, int a, int b, int dest, double fraction)
{
  if (!obj->verts.has_sizing) return;
  double *const size = obj->verts.sizing;
  size[dest] = (1.0 - fraction) * size[a] + fraction * size[b];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The predicates
 * ═══════════════════════════════════════════════════════════════════════════ */

bool MFTLRemeshEdgeWantsSplit(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params)
{
  switch (params->sizing)
  {
    case MFTL_REMESH_SIZING_CURVATURE:
    {
      const double L = MFTL_REMESH_SPLIT_RATIO * edge_target(obj, MFTLHeOrigin(obj, h), MFTLHeTarget(obj, h), params);
      return MFTLGetEdgeLengthSquared(obj, h) > L * L;
    }
    case MFTL_REMESH_SIZING_UNIFORM:
    default: return MFTLGetEdgeLengthSquared(obj, h) > params->max_edge_length_sq;
  }
}

bool MFTLRemeshEdgeWantsCollapse(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params)
{
  switch (params->sizing)
  {
    case MFTL_REMESH_SIZING_CURVATURE:
    {
      const double L = MFTL_REMESH_COLLAPSE_RATIO * edge_target(obj, MFTLHeOrigin(obj, h), MFTLHeTarget(obj, h), params);
      return MFTLGetEdgeLengthSquared(obj, h) < L * L;
    }
    case MFTL_REMESH_SIZING_UNIFORM:
    default: return MFTLGetEdgeLengthSquared(obj, h) < params->min_edge_length_sq;
  }
}

bool MFTLRemeshCollapseStretches(const MFTL_Object *obj, double length_sq, double merged_target, int ring_vertex, const MFTL_RemeshParameters *params)
{
  if (params->sizing == MFTL_REMESH_SIZING_CURVATURE)
  {
    const double L = MFTL_REMESH_SPLIT_RATIO * MFTL_MIN(merged_target, obj->verts.sizing[ring_vertex]);
    return length_sq > L * L;
  }
  /* Uniform: the split threshold, the expression from before sizing existed. */
  return length_sq > params->max_edge_length_sq;
}
