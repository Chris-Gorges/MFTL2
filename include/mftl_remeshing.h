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

// The "Command" packet sent from Owner to Neighbor
typedef struct
{
  int8_t type;  // MFTL_RemeshOpType
  int edge_gid;  // The global ID of the edge being operated on

  // Payload for Split:
  double new_pos[3];  // Exact position of the new vertex
  int new_vert_gid;  // Pre-assigned GID for the new vertex (deterministic!)
  int new_edge_gid;  // Pre-assigned GID for the new halfedge

  // Payload for Collapse:
  int keep_vert_gid;  // The GID of the vertex to keep

} MFTL_RemeshAction;

// Dynamic array (vector) helper for buffering actions
typedef struct
{
  MFTL_RemeshAction *data;
  size_t count;
  size_t capacity;
} ActionBuffer;

// // Helper to determine edge ownership deterministically
// // Rule: The rank with the lower ID owns the shared edge.
// static inline bool MFTLIsEdgeOwner(int my_rank, int neighbor_rank) { return my_rank < neighbor_rank; }

/* ── Per-entity flag additions ────────────────────────────────────────── */
#define MFTL_FLAG_FROZEN (1 << 4) /* in freeze band; do not modify  */
#define MFTL_FLAG_FROZEN_NEXT (1 << 5) /* staging flag for dilation pass */
#define MFTL_FLAG_NEVER_INTERIOR (1 << 6) /* set at pass-start; cleared     */
/* when entity becomes unfrozen   */
/* even once — convergence cert.  */

#define MFTL_FLAG_IMMUTABLE (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED | MFTL_FLAG_FROZEN)

/* ── Phase schedule ────────────────────────────────────────────────────
 *
 * MFTL_SHIFT_PHASES[p][k]:
 *   0  →  no shift in axis k for phase p
 *   1  →  shift cut planes by h_k/2 in axis k for phase p
 *
 * The ordering maximises useful work in the fewest phases:
 *
 *   Phases 0–1  :  resolve O(δ/h)       slab   conflicts   (~95–99 %)
 *   Phases 2–4  :  resolve O((δ/h)²)    tube   conflicts   (even OA)
 *   Phases 5–7  :  resolve O((δ/h)³)    corner conflicts   (odd set)
 *
 * All 8 phases together form {0, h/2}³, which guarantees that every
 * entity is strictly interior (unfrozen) in at least one phase,
 * provided the hard constraint  δ < h_min/4  holds.
 * ─────────────────────────────────────────────────────────────────── */
static const int MFTL_SHIFT_PHASES[8][3] = {
    {0, 0, 0}, /* phase 0 – canonical / baseline                      */
    {1, 1, 1}, /* phase 1 – full diagonal; resolves all slab conflicts */
    {0, 1, 1}, /* phase 2 – even orthogonal-array member 1            */
    {1, 0, 1}, /* phase 3 – even orthogonal-array member 2            */
    {1, 1, 0}, /* phase 4 – even orthogonal-array member 3            */
    {1, 0, 0}, /* phase 5 – odd set member 1                          */
    {0, 1, 0}, /* phase 6 – odd set member 2                          */
    {0, 0, 1} /* phase 7 – odd set member 3                          */
};
#define MFTL_NUM_SHIFT_PHASES 8

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
}

// --- Core Topological Operators ---
// These return the index of the central vertex (split) or remaining vertex (collapse)
// or MFTL_ERROR on failure.

int MFTLSplitEdge(MFTL_Object *obj, int he_ab_idx);
void MFTLCollapseEdge(MFTL_Object *obj, int he_ab_idx);
int MFTLSwapEdge(MFTL_Object *obj, int he_idx);

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

// --- High Level Routine ---
void MFTLRemeshEnsemble(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

// --- Parallel Remeshing (WIP — Stage 10 completes the implementation) ---
// Parallel dispatcher. Deliberately distinct from the serial MFTLRemeshEnsemble:
// the serial entry point (remeshing_core.c) keeps the void signature used by
// existing examples; the parallel dispatcher reports success/failure via its
// return code.
int MFTLRemeshEnsembleParallel(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);
// Shift-scheme kernel: spatial shift / freeze / migrate loop.
// Declared here so the WIP dispatcher type-checks; implemented in Stage 10.
int MFTLRemeshEnsembleShiftScheme(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);
// Centering kernel: object-center migration + ungated serial kernel.
// Declared here so the WIP dispatcher type-checks; implemented in Stage 10.
int MFTLRemeshEnsembleDispersedCentering(MFTL_Ensemble *ens, MFTL_RemeshParameters *params);

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

/* ══════════════════════════════════════════════════════════════════════
 * MFTLDomainSetShiftPhase
 *
 * Sets dom->decomp_shift[k] for each axis k according to the selected
 * phase from the MFTL_SHIFT_PHASES table:
 *
 *   decomp_shift[k] = MFTL_SHIFT_PHASES[phase_idx][k]  *  (h_k / 2)
 *
 * After this call, MFTLGetOwnerProc uses the updated decomp_shift to
 * determine which rank owns a given triangle centroid position.
 *
 * The shift causes all cut planes in axis k to translate by h_k/2
 * (when the table entry is 1), which is the core geometric operation
 * of the freeze/migrate scheme.
 *
 * Precondition : 0 <= phase_idx < MFTL_NUM_SHIFT_PHASES (= 8)
 * Postcondition: dom->decomp_shift is updated; no entity data is moved.
 *                The caller must follow this with
 *                MFTLRedistributeListOfObjects() to actually migrate
 *                entities to their new owning ranks.
 *
 * Complexity : O(1), no communication.
 * ══════════════════════════════════════════════════════════════════════ */
static inline void MFTLDomainSetShiftPhase(MFTL_Domain *dom, int phase_idx)
{
  MFTL_ASSERT(dom != MFTL_NULL);
  MFTL_ASSERT(phase_idx >= 0 && phase_idx < MFTL_NUM_SHIFT_PHASES);

  for (int k = 0; k < 3; ++k)
  {
    double h = dom->domain_size[k] / (double) dom->cuts[k];

    /* Ternary keeps branch-free in the common case. */
    dom->decomp_shift[k] = MFTL_SHIFT_PHASES[phase_idx][k] ? 0.5 * h : 0.0;
  }
}

#endif