/*
 * @Author: Christian Gorges
 * @Date: 2025-08-10 10:00:56
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-06 15:43:58
 */

#ifndef MFTL_REMESHING_H
#define MFTL_REMESHING_H

#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"

typedef enum
{
  MFTL_OP_NONE = 0,
  MFTL_OP_SPLIT,
  MFTL_OP_COLLAPSE,
  MFTL_OP_SWAP
} MFTL_RemeshOpType;

/* ── Centering threshold ────────────────────────────────────────────────
 *
 * Object-centering migration uses ONLY 26-neighbour communication.
 * This is safe when every triangle of an object can reach the object's
 * centroid rank by a single MPI_Neighbor_alltoallv hop.
 *
 * Derivation:
 *   The 26-neighbour shell of the centroid rank extends exactly one
 *   rank-box width h_min beyond its boundary in each axis.  An object
 *   of bounding-box width W whose centroid sits at the worst-case
 *   position (a rank-box corner) has its farthest triangle at distance W
 *   from the rank boundary.  Strict 26-neighbour reachability requires:
 *
 *       W  <  h_min
 *
 *   The factor 0.5 provides a safety margin for:
 *     – centroid not coinciding with the object's geometric centre
 *     – discrete half-edge stencil extending slightly beyond the
 *       geometric bounding box
 * ─────────────────────────────────────────────────────────────────── */
#define MFTL_CENTERING_WIDTH_FACTOR 0.5

typedef enum
{
  MFTL_REMESH_ISOTROPIC,  // Botsch & Kobbelt (4/5 and 4/3 limits)
  MFTL_REMESH_ANISOTROPIC,  // Metric-driven remeshing
  MFTL_REMESH_CUSTOM  // Manual user-defined limits
} MFTL_RemeshType;

typedef enum
{
  MFTL_SMOOTH_VERTEX_BASED,
  MFTL_SMOOTH_TRIANGLE_BASED,
  MFTL_SMOOTH_QUALITY_DRIVEN,  // Stage 08: move only if local quality improves
  MFTL_SMOOTH_CURVATURE_BASED  // Stage 08: curvature-scaled tangential move
} MFTL_SmoothMode;

// Remeshing criteria (MFTL1 RemeshCriteria parity, mftl-constants.h).
typedef enum
{
  MFTL_CRITERION_TRIANGLE_AREA = 0,
  MFTL_CRITERION_BOTSCH_KOBBELT = 1,
  MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE = 2,
  MFTL_CRITERION_JIAO = 3
} MFTL_RemeshCriterion;

// Vertex fitting mode for new/moved vertices (MFTL1 MFTL_VertexFit parity).
typedef enum
{
  MFTL_FIT_NONE = 0,
  MFTL_FIT_PARABOLOID = 1,
  MFTL_FIT_BARYCENTRIC = 2
} MFTL_VertexFitMode;

// Stage 09: volume-conserving smoothing mode (MFTL1 VertexSmoothing parity).
typedef enum
{
  MFTL_VCSMOOTH_NONE = 0,
  MFTL_VCSMOOTH_KUPRAT = 1,  // volume-conserving Laplacian (Kuprat Alg. 4)
  MFTL_VCSMOOTH_TSUR = 2  // Jiao TSUR3D null-space smoothing
} MFTL_VCSmoothMode;

typedef struct
{
  MFTL_RemeshType type;
  MFTL_SmoothMode smooth_mode;
  int iterations;

  // Length targets
  double target_length;
  double min_edge_length;
  double max_edge_length;
  double min_edge_length_sq;
  double max_edge_length_sq;

  double feature_angle_threshold;

  // --- Stage 08: criteria, thresholds, fitting, smoothing ---
  MFTL_RemeshCriterion criterion;   // which split/collapse criterion drives the pass
  MFTL_VertexFitMode vertex_fit;    // fitting for new/moved vertices (feature edges excluded)
  double epsilon;                   // curvature criterion: max deviation from the surface
  double reference_length;          // Jiao/standard relative reference length
  double min_area_threshold;        // triangle-area criterion
  double max_area_threshold;
  double max_angle_between_tris;    // flip gate: max dihedral (degrees) of created triangles
  double min_quality_threshold;     // smoothing quality window (MFTL1 quality, 1 = perfect)
  double max_quality_threshold;     // flip/smoothing gate: reject worse-than-this quality
  int max_smoothing_iter;           // tangential smoothing iterations per pass

  // --- Stage 09: volume-conserving smoothing (MFTL1 parity) ---
  MFTL_VCSmoothMode volume_conserving_smoothing;  // post-remesh smoothing stage
  int roughness_smoothing;                        // per-edge roughness-triggered smoothing (0/1)
  double laplacian_smoothing_under_relaxation;    // Kuprat omega
  int max_smoothing_laplacian_iter;               // Kuprat/TSUR iteration cap
  double max_roughness;                           // dihedral-roughness trigger
  double max_geometric_laplacian;                 // geometric-Laplacian trigger

  // --- Stage 10: load balancing (MFTL1 RemeshParameter parity) ---
  int min_triangles_per_proc;  // MinNoTriangles: lower bound for slab splitting (0 = use all ranks)
  int max_balance_iter;        // maxBalanceIter: slab-width optimizer iteration cap
  double size_threshold;       // Sizethreshold: minimum slab width; retry with fewer slabs below
  double damping_factor;       // dampingfactor: slab-width optimizer damping in (0,1]
  int load_balance;            // enable x-only load balancing + two-pass boundary shift
  int boundary_cleanup;        // MFTL_BoundaryCleanup: gather-to-rank-0 serial pass at the
                               // end, remeshing slivers the two-pass shift cannot cover
                               // (edges straddling both the old and the shifted cut)

  // --- Stage 15: MFTL1 driver-parity switches. Every switch is ZERO by default, which keeps
  //     the MFTL2 pipeline unchanged; multiflow's front tracking sets them to reproduce
  //     MFTL1's MFTL_Remeshing (optional operators, fixed iteration count, no volume
  //     recovery inside remeshing, quiet). ---
  int skip_split;  // 1: no edge splits      (MFTL1 MFTL_Refine = NO)
  int skip_collapse;  // 1: no edge collapses   (MFTL1 MFTL_Coarse = NO)
  int skip_flip;  // 1: no edge flips       (MFTL1 MFTL_Flip = NO)
  int skip_tangential_smoothing;  // 1: no tangential smoothing (MFTL1 MFTL_VertexSmoothing = NO)
  int skip_volume_recovery;  // 1: no global volume recovery inside the remesh drivers
  int fixed_iterations;  // 1: run exactly `iterations` passes (no early convergence exit)
  int skip_split_cleanup;  // 1: no final split-cleanup passes in the parallel driver
  int quiet;  // 1: no per-iteration progress output on stdout
  // 1: MFTL1 edge criteria (Stage 15 parity): MFTL1's split / collapse / flip tests
  //    (edgesplit.c, edgecollapse.c, edgeflip.c) replace the MFTL2 ones, the serial driver
  //    sweeps splits longest-first and collapses shortest-first (stopping at the reference
  //    length for the JIAO / BOTSCH_KOBBELT criteria), and no vertex locking or feature-flag
  //    protection is applied in that sweep (MFTL1 has neither).
  int mftl1_criteria;
  // Stage 16C (plans/16 §4.3): boundary corner threshold in degrees (<= 0: default 30). Used by
  // the generalized remesher (open bodies, closed bodies opted in with
  // MFTLObjectSetRemeshMode(obj, MFTL_REMESH_MODE_GENERALIZED)).
  double boundary_corner_angle;
  // Stage 16C (plans/16 §4.4.4), outputs accumulated by the drivers: bodies whose volume recovery
  // was skipped because their phase set is not DEFINED (open without closure, wrapping, ...)
  long stat_recovery_skipped;
  // Outputs: LOCAL operation counts of this rank, ACCUMULATED by every remesh driver call
  // (the caller resets them).
  long stat_splits, stat_collapses, stat_flips;

  // ... any other parameters like smoothing weights, boundary locks, etc.
} MFTL_RemeshParameters;

/**
 * @brief Initializes remeshing parameters based on standard literature values.
 */
static inline void MFTLInitRemeshParams(MFTL_RemeshParameters *params, MFTL_RemeshType type, double target_length, int iterations, MFTL_SmoothMode smooth_mode,
                                        double feature_angle_threshold)
{
  params->type = type;
  params->target_length = target_length;
  params->iterations = iterations;
  params->smooth_mode = smooth_mode;
  params->feature_angle_threshold = feature_angle_threshold;

  switch (type)
  {
    case MFTL_REMESH_ISOTROPIC:
      // Standard Botsch & Kobbelt isotropic limits
      params->min_edge_length = (4.0 / 5.0) * target_length;
      params->max_edge_length = (4.0 / 3.0) * target_length;
      break;

    case MFTL_REMESH_ANISOTROPIC:
      // For metric-driven anisotropic remeshing, lengths are evaluated
      // in metric space where the target length is always exactly 1.0.
      // We set the physical target_length to 1.0 to reflect metric space.
      params->target_length = 1.0;
      params->min_edge_length = (4.0 / 5.0);
      params->max_edge_length = (4.0 / 3.0);
      // Note: Your distance functions will need to be updated to compute
      // metric distance instead of Euclidean distance if type == ANISOTROPIC.
      break;

    case MFTL_REMESH_CUSTOM:
      // Do nothing. The user must manually set min/max.
      // (You could set defaults here if you wanted).
      params->min_edge_length = target_length;
      params->max_edge_length = target_length;
      break;
  }

  // Pre-calculate squared values for fast distance checks
  params->min_edge_length_sq = params->min_edge_length * params->min_edge_length;
  params->max_edge_length_sq = params->max_edge_length * params->max_edge_length;

  // --- Stage 08 defaults (MFTL1 RemeshParameter parity) ---
  params->criterion = MFTL_CRITERION_BOTSCH_KOBBELT;
  params->vertex_fit = MFTL_FIT_PARABOLOID;
  params->epsilon = 0.02 * target_length;                 // deviation bound; ~good default
  params->reference_length = target_length;
  // Triangle-area criterion: equilateral area for the target length.
  params->min_area_threshold = 0.25 * sqrt(3.0) * params->min_edge_length * params->min_edge_length;
  params->max_area_threshold = 0.25 * sqrt(3.0) * params->max_edge_length * params->max_edge_length;
  params->max_angle_between_tris = feature_angle_threshold;  // degrees
  // MFTL1 quality q = (p^2 / A) / 20.78460969: 1.0 = equilateral, larger = worse.
  params->min_quality_threshold = 1.2;
  params->max_quality_threshold = 2.0;
  params->max_smoothing_iter = 3;

  // --- Stage 09 defaults (MFTL1 RemeshParameter parity) ---
  params->volume_conserving_smoothing = MFTL_VCSMOOTH_NONE;
  params->roughness_smoothing = 0;
  params->laplacian_smoothing_under_relaxation = 0.5;
  params->max_smoothing_laplacian_iter = 10;
  params->max_roughness = 0.05;
  params->max_geometric_laplacian = 0.05;

  // --- Stage 10 defaults (MFTL1 RemeshParameter parity) ---
  params->min_triangles_per_proc = 0;    // 0: use all ranks (MFTL1 MinNoTriangles disabled)
  params->max_balance_iter = 10;         // MFTL1 Iterationsthreshold
  // A slab should hold at least one full edge plus margin: redistribution
  // and ghost updates assume mesh edges do not span a whole slab (owners
  // stay graph neighbours). Transient slab-spanning edges during split-
  // heavy passes are tolerated (their ghost updates are skipped until the
  // next redistribution), so this is a soft floor, not a hard invariant.
  params->size_threshold = 1.5 * params->max_edge_length;
  params->damping_factor = 0.5;          // MFTL1 dampingfactor (0,1]
  params->load_balance = 1;              // MFTL1 MFTL_LoadBalance = MFTL_YES
  params->boundary_cleanup = 1;          // MFTL1 MFTL_BoundaryCleanup = MFTL_YES

  // --- Stage 15 switches/outputs: off / zero (the MFTL2 pipeline) ---
  params->skip_split = 0;
  params->skip_collapse = 0;
  params->skip_flip = 0;
  params->skip_tangential_smoothing = 0;
  params->skip_volume_recovery = 0;
  params->fixed_iterations = 0;
  params->skip_split_cleanup = 0;
  params->quiet = 0;
  params->mftl1_criteria = 0;
  params->stat_splits = params->stat_collapses = params->stat_flips = 0;

  // --- Stage 16C fields (16D review: previously left uninitialised, so the generalized
  //     remesher's corner protection read an indeterminate threshold) ---
  params->boundary_corner_angle = 0.0;  // <= 0: MFTL_BOUNDARY_CORNER_ANGLE_DEFAULT (30 deg)
  params->stat_recovery_skipped = 0;
}

// --- Core Topological Operators ---
// These return the index of the central vertex (split) or remaining vertex (collapse)
// or MFTL_ERROR on failure.

int MFTLSplitEdge(MFTL_Object *obj, int he_ab_idx);
void MFTLCollapseEdge(MFTL_Object *obj, int he_ab_idx);
int MFTLSwapEdge(MFTL_Object *obj, int he_idx);
/** @brief In-place edge flip (valence optimization; no new entities). */
void MFTLFlipEdge(MFTL_Object *obj, int he_idx);
/** @brief Stage-08 flip gate stack: valence + quality + dihedral (params == NULL keeps the legacy valence-only behaviour). */
int FlipApprovedWithParams(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params);

/**
 * @brief Compacts the object after a remesh pass (append-only fast path when
 * only splits happened; full compaction + khash rebuild after collapses).
 * Must run before any redistribution, which would otherwise ship dead
 * entities.
 */
void MFTLFinalizeRemeshing(MFTL_Object *obj, int start_num_v, int start_num_t, int start_num_h, bool collapses_occurred);

// --- Stage 08: criteria, fitting, smoothing ---

/**
 * @brief MFTL1 triangle quality q = (perimeter^2 / area) / 20.78460969.
 * 1.0 = equilateral; LARGER is worse. Used by the flip and smoothing gates.
 */
double MFTL_TriangleQuality(const MFTL_Object* obj, int tri_idx);

/**
 * @brief Botsch-Kobbelt curvature-adaptive edge threshold (MFTL1 parity):
 * kappa_max = mean + sqrt(mean^2 - gauss) per endpoint; L = sqrt(6*eps/kappa
 * - 3*eps^2) clamped to [min_edge_length, max_edge_length]; split above
 * 4L/3 (flag 1), collapse below 0.8L (flag 2). Requires current curvatures.
 */
int MFTLBotschKobbeltCurvatureThreshold(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params, int split_or_collapse);

/**
 * @brief Reposition vertex v_idx (created on he_idx, or the surviving vertex
 * of a collapse) onto the fitted surface. Dispatches on params->vertex_fit:
 * paraboloid least squares (LAPACK dgelsy when MFTL_HAVE_LAPACK; without it,
 * or when the fit is rank-deficient, the barycentric butterfly; ultimate
 * fallback keeps the midpoint). SKIPPED for feature edges.
 */
void MFTLFitVertex(MFTL_Object* obj, int v_idx, int he_idx, MFTL_RemeshParameters* params);

/** @brief MFTLSplitEdge followed by MFTLFitVertex on the new midpoint. */
int MFTLSplitEdgeWithFit(MFTL_Object* obj, int he_ab_idx, MFTL_RemeshParameters* params);

/** @brief MFTLCollapseEdge followed by MFTLFitVertex on the surviving vertex. */
void MFTLCollapseEdgeWithFit(MFTL_Object* obj, int he_ab_idx, MFTL_RemeshParameters* params);

/** @brief MFTLSmoothTangential driven by params (mode, max_smoothing_iter, quality gates). */
void MFTLSmoothTangentialWithParams(MFTL_Object* obj, MFTL_RemeshParameters* params);

/** @brief Flag half-edges whose adjacent triangle normals differ by more than angle_degrees as MFTL_FLAG_FEATURE. */
void MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees);

/**
 * @brief Collapse gate (MFTL1 thresholds + Stage 08 geometric feature
 * protection): status/one-touch checks, criterion length test, link
 * condition, boundary and stretch guards, feature-endpoint rules.
 */
int CollapseApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params);

// --- Stage 09: volume-conserving smoothing (see src/remeshing/volume_conserving.c) ---

/**
 * @brief Global volume recovery (MFTL1 VolumeRecovery): shift every owned
 * vertex along its one-ring normal by h = (V - target)/Area, iterated until
 * the relative volume error is < 1e-12 or the iteration cap (5) is hit.
 * Returns the final relative volume error. Collective (volume/area
 * reductions over comm_cart); caller refreshes ghost positions afterwards.
 * On distributed meshes the effective Area weights each owned triangle by
 * its owned-corner fraction (boundary triangles share corners with ghost
 * copies), so the first-order correction stays exact in parallel.
 */
double MFTLVolumeRecovery(MFTL_Object* obj, double target_volume);

/** @brief Kuprat Algorithm-4 volume-conserving smoothing of one edge (exact first-order local volume correction). */
void MFTLEdgeVolumeConservingSmoothing(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params);

/** @brief Kuprat driver: per-edge volume-conserving Laplacian for max_smoothing_laplacian_iter iterations. */
void MFTLVolumeConservingSmoothing(MFTL_Object* obj, MFTL_RemeshParameters* params);

/** @brief Jiao TSUR3D null-space smoothing of one edge (volume-conserving tangent-plane projection). Returns MFTL_SUCCESS or MFTL_ERROR (degenerate/NaN). */
int MFTLEdgeTSUR(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params);

/** @brief TSUR3D driver over all interior edges. */
void MFTLTSUR3D(MFTL_Object* obj, MFTL_RemeshParameters* params);

/**
 * @brief Roughness-triggered smoothing (MFTL1 Object_Roughness_Smoothing):
 * per edge, when the dihedral roughness or geometric Laplacian exceeds the
 * thresholds, dispatch to Kuprat or TSUR per params->volume_conserving_smoothing.
 * Recomputes the roughness data (MFTLComputeRoughness) on entry.
 */
void MFTLRoughnessSmoothing(MFTL_Object* obj, MFTL_RemeshParameters* params);

// --- High Level Routine ---
/* ---- Stage 16C: remesh-mode selection and boundary curves (plans/16 §4.3, §4.4.4) ---- */
/** @brief Per-body remesh mode (identical on all ranks): MFTL_REMESH_MODE_DEFAULT (closed bodies:
 *  the Stage-15 path, i.e. MFTL1 parity when params->mftl1_criteria, periodic or not) or
 *  MFTL_REMESH_MODE_GENERALIZED (explicit opt-in of a closed body). Open bodies always use the
 *  generalized, boundary-aware remesher (never the parity path). */
void MFTLObjectSetRemeshMode(MFTL_Object *obj, int mode);
/** @brief 1 if the body is remeshed by the generalized remesher (open, or opted in). */
int MFTLObjectRemeshGeneralized(const MFTL_Object *obj);
/** @brief Boundary-curve mode of boundary tag `tag` (identical on all ranks): REPRESENTED
 *  (default; interpolating Catmull-Rom spline through the loop's vertices, corners are
 *  breakpoints) or CONSTRAINED (new and moved boundary vertices are projected by `project`).
 *  freeze = 1: boundary vertices of the tag are never moved by smoothing (e.g. PINNED). */
int MFTLObjectSetBoundaryCurve(MFTL_Object *obj, int tag, int mode, MFTL_BoundaryProjector project, void *user, int freeze);
/** @brief 1 (and clears the flag) if the last canonicalisation changed the body's wrap flags. */
int MFTLObjectTakeClassTransition(MFTL_Object *obj);

void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

// --- Stage 10: parallel remeshing + load balancing (MFTL1 parity) ---

/**
 * @brief Split predicate (Stage 08), exposed for the parallel driver: the
 * parallel split phase counts candidates on every rank and assigns disjoint
 * per-rank GID blocks BEFORE applying any split, so block assignment needs
 * this predicate outside remeshing_core.c.
 */
int MFTLSplitApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params);

/* Stage 15: MFTL1 edge criteria (mftl1_criteria.c; see MFTL_RemeshParameters.mftl1_criteria).
   The approval functions above delegate to them when mftl1_criteria is set; the serial driver
   runs MFTL_Remeshing's three sweeps with MFTL1's edge order (operation counts returned). */
int MFTL1SplitCriterion(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params);
int MFTL1CollapseApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params, int honour_locks);
int MFTL1FlipApproved(MFTL_Object *obj, int he_idx, MFTL_RemeshParameters *params);
void MFTL1Sweeps(MFTL_Object *obj, MFTL_RemeshParameters *params, int *n_split, int *n_collapse, int *n_flip);

/**
 * @brief Parallel dispatcher (all ranks of ens->domain.comm_cart must call).
 * Selects the remesh algorithm (centering for small dispersed objects, the
 * MFTL1 two-pass shift scheme otherwise; serial kernel on a single rank) and
 * returns MFTL_SUCCESS / MFTL_ERROR.
 */
int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

/**
 * @brief MFTL1's x-only load balancing with the two-pass boundary shift.
 * Requires an x-only Cartesian decomposition (cuts[1] == cuts[2] == 1);
 * on other decompositions it degrades to interior-only remeshing (no load
 * balancing, no boundary shift) and prints a note on rank 0.
 */
int MFTLRemeshEnsembleShiftScheme(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

/**
 * @brief Gather the ensemble to rank 0 (temporary single-partition domain),
 * run the ungated serial remesh kernel, restore the domain and redistribute.
 * Used for dispersed objects small enough to center (W < 0.5 * h_min).
 */
int MFTLRemeshEnsembleDispersedCentering(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

/**
 * @brief Stage 15: serial bracket. MFTLEnsembleBeginSerial gathers every object of the
 * ensemble onto rank 0 (single-partition domain, no ghosts, gid counters synced);
 * MFTLEnsembleEndSerial restores the native domain and optionally redistributes back.
 * Collective. Used for serial kernels on distributed ensembles (TSUR3D, volume-conserving
 * smoothing, restart output, ...).
 */
typedef struct
{
  MFTL_Domain native;
} MFTL_SerialBracket;
int MFTLEnsembleBeginSerial(MFTL_Ensemble *ens, MFTL_SerialBracket *bracket);
int MFTLEnsembleEndSerial(MFTL_Ensemble *ens, MFTL_SerialBracket *bracket, int redistribute);

/**
 * @brief MFTL1 MFTLRedistributeObjectLoadBalanced, ensemble-wide: computes an
 * x-only slab decomposition (see MFTL_Domain.lb_*) that equalizes the OWNED
 * triangle count per slab and redistributes. The optimizer state is built
 * from Allreduced quantities only, so every rank computes identical cuts.
 * Requires cuts[1] == cuts[2] == 1. Returns the number of active slabs.
 */
int MFTLRedistributeEnsembleLoadBalanced(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

/**
 * @brief MFTL1 ShiftOwnershipRanges: translate the x-cuts right by
 * max(0.2 * min slab width, 3 * reference_length) so triangles that sat on a
 * slab boundary in pass 1 become slab-interior for the second remeshing
 * pass, then redistribute. Returns MFTL_ERROR when a shift is impossible
 * (fewer than 2 active slabs) and the caller must skip pass 2.
 */
int MFTLShiftOwnershipRanges(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

/**
 * @brief Synchronize every object's next_*_gid counters across
 * ens->domain.comm_cart (MPI_Allreduce MAX) and optionally store the common
 * maximum on all ranks. Called by the parallel remesh driver after every
 * redistribution so later GID allocation can never collide with entities
 * created on other ranks.
 */
void MFTLSyncEnsembleGidCounters(MFTL_Ensemble *ens);

// --- Stage 10: parallel-safe edge-local smoothers (src/remeshing/volume_conserving.c) ---
// The serial drivers MFTLVolumeConservingSmoothing / MFTLTSUR3D /
// MFTLRoughnessSmoothing fail loudly when called on a distributed mesh.
// These variants gate every edge to the MFTL1 parallel-interior rule (both
// endpoints owned by this rank, full local ghost-free fan), which makes the
// per-edge steps exactly local; ghost copies are refreshed by the driver.
void MFTLVolumeConservingSmoothingParallel(MFTL_Object *obj, MFTL_RemeshParameters *params);
void MFTLTSUR3DParallel(MFTL_Object *obj, MFTL_RemeshParameters *params);
void MFTLRoughnessSmoothingParallel(MFTL_Object *obj, MFTL_RemeshParameters *params);

/* ══════════════════════════════════════════════════════════════════════
 * MFTLDomainGetHMin
 *
 * Returns the minimum rank-box width across all three domain axes.
 *
 *   h_k  =  domain_size[k] / cuts[k]        (box width in axis k)
 *   h_min = min(h_0, h_1, h_2)
 *
 * This is the quantity that appears in the shift-scheme hard constraint
 *   δ  <  h_min / 4
 * and in the centering-threshold test
 *   W_typical  <  MFTL_CENTERING_WIDTH_FACTOR * h_min.
 *
 * Complexity : O(1), no communication.
 * ══════════════════════════════════════════════════════════════════════ */
static inline double MFTLDomainGetHMin(MFTL_Domain *dom)
{
  MFTL_ASSERT(dom != MFTL_NULL);
  MFTL_ASSERT(dom->cuts[0] > 0 && dom->cuts[1] > 0 && dom->cuts[2] > 0 && "cuts must be positive");
  MFTL_ASSERT(dom->domain_size[0] > 0.0 && dom->domain_size[1] > 0.0 && dom->domain_size[2] > 0.0 && "domain_size must be positive");

  double h_min = MFTL_DBL_MAX;

  for (int k = 0; k < 3; ++k)
  {
    double h = dom->domain_size[k] / (double) dom->cuts[k];
    if (h < h_min) h_min = h;
  }

  MFTL_ASSERT(h_min < MFTL_DBL_MAX); /* unreachable if assertions above pass */
  return h_min;
}


#endif
