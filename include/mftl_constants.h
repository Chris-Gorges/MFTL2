/*
 * MFTL2 core — return codes, numerical constants, entity flag bits and the enumerations shared
 * by every header (twin states, ring classes, commit modes, remesh and boundary-curve modes).
 *
 * Tolerances: MFTL_VERYSMALL / MFTL_VERYVERYSMALL are fixed absolute guards (denominators,
 * degenerate-geometry tests); scale-dependent decisions use mesh-relative tolerances in the
 * modules that make them.
 */
#ifndef MFTL_CONSTANTS_H
#define MFTL_CONSTANTS_H

#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <float.h> /* DBL_MAX   */

#define MFTL_ASSERT(condition) assert(condition)

#define MFTL_SUCCESS 0
#define MFTL_ERROR -1

#define MFTL_NULL (NULL)

#define MFTL_TRUE (true)
#define MFTL_FALSE (false)

#define MFTL_PI 3.14159265358979323846
#define MFTL_TWO_PI 6.28318530717958647692

// absolute tolerances
#define MFTL_EPSILON 1e-9
#define MFTL_VERYSMALL 1e-12
#define MFTL_VERYVERYSMALL 1e-15

#define MFTL_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MFTL_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MFTL_SQ(x) ((x) * (x))
#define MFTL_SQRT(x) sqrt(x)
#define MFTL_ABS(x) (((x) < 0) ? -(x) : (x))
#define MFTL_COS(x) cos(x)
#define MFTL_SIN(x) sin(x)

#define MFTL_ONE_THIRD 0.33333333333333333333

#define MFTL_DBL_MAX DBL_MAX
#define MFTL_DBL_MIN DBL_MIN

#define MFTL_MAX_VALENCE 32

// first capacity of a growing entity array (MFTL_ENSURE_CAPACITY)
#define MFTL_INIT_CAPACITY 128

// =============================================================================
// ENSEMBLE CLASSIFICATION & PARALLEL REMESHING
// (Defined here, not in mftl_remeshing.h, because MFTL_Ensemble in
// mftl_core.h needs these types and must not include mftl_remeshing.h —
// that would create a circular include.)
// =============================================================================
typedef enum
{
  MFTL_ENSEMBLE_DISPERSED = 0, /* bubbles/droplets; R_obj expected << h  */
  MFTL_ENSEMBLE_SPANNING = 1 /* jets, films, large interfaces; R >> h  */
} MFTL_EnsembleType;

typedef enum
{
  MFTL_ALGO_CENTERING = 0, /* gather each body to one rank + serial kernel */
  MFTL_ALGO_SHIFT_SCHEME = 1 /* distributed two-pass shift scheme          */
} MFTL_RemeshAlgorithm;

// =============================================================================
// ENTITY FLAGS (bitmask; the flags travel with the entity through MPI packing)
// =============================================================================
#define MFTL_FLAG_NONE 0
#define MFTL_FLAG_DEAD (1 << 0)  // deleted inside a remesh pass, compacted away at finalisation
#define MFTL_FLAG_MODIFIED (1 << 1)  // touched in the current pass (incremental updates)
#define MFTL_FLAG_BOUNDARY (1 << 2)  // reserved; the physical boundary is the BOUNDARY twin state
#define MFTL_FLAG_VISITED (1 << 3)  // evaluated in the current remeshing pass / traversal
#define MFTL_FLAG_FEATURE (1 << 4)  // sharp feature (not smoothed across, not collapsed)

// vertex flag bits (plans/16 §4.2/§4.3)
#define MFTL_VFLAG_CURV_UNDEFINED (1 << 8)  // curvature not computed: open (boundary) or incomplete fan
#define MFTL_VFLAG_CURV_EXTRAPOLATED (1 << 9)  // curv_extrap holds an open-stencil paraboloid estimate
#define MFTL_VFLAG_CORNER (1 << 10)  // boundary corner (input/application tag): never moved/merged/collapsed

// per-body remesh mode (plans/16 §4.4.4)
#define MFTL_REMESH_MODE_DEFAULT 0  // closed: the closed-surface remesher (MFTL1 parity with mftl1_criteria); open: generalized
#define MFTL_REMESH_MODE_GENERALIZED 1  // explicit opt-in of a closed body into the generalized remesher

// boundary-curve modes per boundary tag (plans/16 §4.3)
#define MFTL_BCURVE_REPRESENTED 0  // interpolating Catmull-Rom spline through the loop's vertices (default)
#define MFTL_BCURVE_CONSTRAINED 1  // application projector: new/moved boundary vertices are projected
#define MFTL_BOUNDARY_CORNER_ANGLE_DEFAULT 30.0  // degrees (plans/16 §4.3)

// =============================================================================
// Half-edge twin states (plans/16 §2.1)
// =============================================================================
// Every live half-edge of a QUIESCENT mesh is exactly one of
//   LOCAL    : twin_idx >= 0, twin_gid == half_edges[twin_idx].global_id
//   REMOTE   : twin_idx == -1, twin_gid >= 0 (live twin on another rank, names h back)
//   BOUNDARY : twin_idx == -1, twin_gid == MFTL_TWIN_NONE - tag (tag >= 0)
// MFTL_TWIN_UNKNOWN (== MFTL_ERROR) is legal only inside an operation (remesh pass,
// staged copy before MFTLObjectCommitCopy) and is rejected by MFTLValidateTwinStates.
#define MFTL_TWIN_UNKNOWN (-1)
#define MFTL_TWIN_NONE (-2)

// MFTLVertexRing classification
#define MFTL_RING_CLOSED 0  // interior vertex, closed fan
#define MFTL_RING_OPEN_COMPLETE 1  // fan runs BOUNDARY -> BOUNDARY (boundary vertex)
#define MFTL_RING_INCOMPLETE 2  // the fan walk reaches a REMOTE half-edge (or is malformed)

// MFTLObjectCommitCopy modes (plans/16 §2.2)
#define MFTL_COMMIT_ASSEMBLE 0  // every non-LOCAL half-edge must resolve to a live partner
#define MFTL_COMMIT_CUT 1  // unresolved copies of existing source twins become BOUNDARY (new tag)

// MFTLValidateTwinStatesEx flags
#define MFTL_VALIDATE_GHOSTS (1 << 0)  // also check G1/G2 (after a redistribution)
#define MFTL_VALIDATE_CHEAP (1 << 1)  // local: states, LOCAL symmetry and G1 only (Release quiescent points)

#endif