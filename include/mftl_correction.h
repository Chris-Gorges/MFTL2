/**
 * @file mftl_correction.h
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_correction
 *
 * @brief Volume-conserving corrections of a front between flow steps: removal of
 *        grid-scale undulations (TSUR3D) and restoring each body's volume.
 *
 * Both act on vertex POSITIONS only, along the surface normal, and both are
 * separate calls a solver makes in its own time loop -- never inside the
 * remesher. That is deliberate: the remesher's Phong back-projection
 * (#MFTL_REMESH_PROJECT_PHONG) moves every smoothed vertex back onto the surface
 * the call received, so a normal correction made INSIDE a remesh call would be
 * undone by the same call. Made before or after it, the projection keeps it.
 *
 * - MFTLEnsembleRemoveUndulations() is TSUR3D (de Sousa et al., J. Comput. Phys.
 *   198 (2004) 469-499): for each edge, both ends are moved along a common
 *   smoothed normal onto one plane, at the height that keeps the volume under
 *   their two fans. Each edge operation conserves the enclosed volume EXACTLY
 *   (to round-off), so a sweep does too, at any process count.
 * - MFTLEnsembleCorrectVolumes() restores each body's #MFTL_Object::reference_volume
 *   by a uniform offset along the area-weighted vertex normals, in Newton steps on
 *   the exact first variation of the volume -- the correction the topology change
 *   applies to its clusters, callable on its own.
 *
 * Optional fields (#MFTL_FieldMode) follow the motion: a CONSERVATIVE_DENSITY
 * keeps its AMOUNT per triangle (density times area), because a normal motion
 * carries no material across an edge; vertex fields keep their values, because
 * the vertex is the same material point.
 *
 * All of this module's MPI is in src/correction/correction_comm.c
 * (`test_correction_mpi_is_confined`). Theory, the distributed schedule and the
 * measurements: src/correction/README.md.
 */
#ifndef MFTL_CORRECTION_H
#define MFTL_CORRECTION_H

#include "mftl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Undulation removal (TSUR3D)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_UndulationParameters
 * @brief What one MFTLEnsembleRemoveUndulations() call does.
 */
typedef struct MFTL_UndulationParameters
{
  /** @brief Passes over the edges per call (Gauss-Seidel, pool order). Default 1. */
  int sweeps;

  /**
   * @brief Edges whose kink is at most this angle, in degrees, are left alone.
   *
   * The kink is the angle between the edge and the plane normal to its smoothed
   * normal, i.e. exactly what the operation removes. On a smooth surface it is
   * third order in the edge length, so a threshold leaves a resolved surface
   * untouched and still catches a grid-scale zigzag. Default 15: on the LeVeque
   * deformation (docs/analysis.md §15.2) treating every edge (0, the published
   * method) smoothed away the under-resolved filament (56.7 % R0 off the exact
   * sphere at refinement 4, against 2.2 % without TSUR3D), while 15 was neutral
   * in position and improved the curvature -- and with velocity noise improved
   * both (3.67 -> 2.66 % R0 and E_H 14.2 -> 4.4 at refinement 5). On a resolved,
   * noisy interface every edge removes more noise (0.51 against 0.80 % R0 on a
   * rotating sphere): set 0 there, or a few degrees.
   */
  double min_angle;

  /**
   * @brief Fraction of the way to the common height each end moves, in (0, 1].
   *
   * Any fraction conserves the volume exactly. Default 1 (TSUR3D).
   */
  double relaxation;
} MFTL_UndulationParameters;

/**
 * @struct MFTL_UndulationStats
 * @brief What one call did. Every field is GLOBAL, identical on every rank.
 */
typedef struct MFTL_UndulationStats
{
  long long n_moved;  ///< Edge operations applied, over all sweeps.
  long long n_moved_band;  ///< Of which, in the coloured rounds: edges whose fans other ranks hold too. The distributed schedule's non-vacuity measure.
  long long n_flat;  ///< Edges not moved because their kink was at most `min_angle`.
  long long n_refused;  ///< Edges not moved because a fan is folded against the normal or the move would fold a triangle.
  long long n_cross_rank;  ///< Edges whose two ends have different owners, counted once per sweep: never moved (each end still moves through its other edges).
  int n_rounds;  ///< Coloured rounds run, over all sweeps (0 on one rank).
  double max_displacement;  ///< Largest single vertex move of one operation.
  double t_total;  ///< Seconds in the call (max over ranks).
} MFTL_UndulationStats;

/**
 * @brief The defaults: one sweep, every edge, full relaxation.
 * @return The parameter set to start from.
 */
MFTL_UndulationParameters MFTLUndulationDefaultParameters(void);

/**
 * @brief Removes grid-scale undulations from every body by TSUR3D, conserving
 *        each body's enclosed volume exactly.
 *
 * Per edge a-b with both ends free (not #MFTL_FLAG_BOUNDARY, #MFTL_FLAG_FEATURE
 * or #MFTL_FLAG_CORNER) and closed fans: each end's normal is the area-weighted
 * normal of its fan with BOTH ends moved to the centroid of their one-rings,
 * which is what keeps an end's own noise out of its normal; n is their
 * normalised sum, h_a = -h_b the ends' heights above the plane through the
 * midpoint, and P_a, P_b the areas of the two fans projected onto that plane.
 * Both ends move along n to the common height
 * \f$h = (h_a P_a + h_b P_b)/(P_a + P_b)\f$. Moving an end along n changes the
 * volume by its displacement times P/3, with no cross term because both moves
 * are parallel, so this h conserves the volume exactly. It is the published
 * \f$(V_1+V_2)/(V_1/h_1+V_2/h_2)\f$ with \f$V_i/h_i = P_i/3\f$ written out --
 * which is 0/0 when the edge is already flat, and here is simply no move.
 *
 * Distributed exactly, with no gather: every rank first treats the edges whose
 * fans no other rank holds; then the ranks take turns in colours no two
 * neighbouring ranks share, each treating its remaining edges with both ends
 * owned, with a ghost position exchange after each turn. An edge whose ends
 * have different owners is never moved. Ghost positions are refreshed first and
 * are exact on return; triangle geometry is stale (the geometry bit is clear)
 * and `verts.normal` is unspecified.
 *
 * @param ens   The ensemble.
 * @param p     Parameters; #MFTL_NULL for MFTLUndulationDefaultParameters().
 * @param stats What was done, or #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on every rank for a NULL ensemble or
 *         parameters out of range.
 *
 * @collective Every rank, including ranks holding no body. The parameters must
 *             agree: they decide how many collectives run.
 */
int MFTLEnsembleRemoveUndulations(MFTL_Ensemble *ens, const MFTL_UndulationParameters *p, MFTL_UndulationStats *stats);

/**
 * @brief TSUR3D on one object, moving only vertices no other rank holds a copy
 *        of -- on one rank, or on a mesh no partition touches, every free vertex.
 *
 * The local half of MFTLEnsembleRemoveUndulations(): on a partitioned mesh it
 * leaves the band near the cut planes alone, and moves no ghost, so the call
 * needs no exchange afterwards. Conservative densities are rescaled.
 *
 * @param obj   The mesh; derived topology is built if missing.
 * @param p     Parameters; #MFTL_NULL for the defaults.
 * @param stats Rank-local counts of this call, or #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for NULL or out-of-range input.
 * @rank_local
 */
int MFTLRemoveUndulationsObject(MFTL_Object *obj, const MFTL_UndulationParameters *p, MFTL_UndulationStats *stats);

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Volume restoration
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @struct MFTL_VolumeCorrectionParameters
 * @brief What one MFTLEnsembleCorrectVolumes() call does.
 */
typedef struct MFTL_VolumeCorrectionParameters
{
  /**
   * @brief Newton steps. Default 3.
   *
   * A FIXED count, so every rank runs the same collectives: a stopping test on a
   * reduced double is two ranks disagreeing in the last ulp about whether to
   * enter another. Quadratic: from a sphere shrunk by 8.7 %, 2.8e-3 / 2.6e-6 /
   * 2.3e-12 after one / two / three steps; from the per-step error of an
   * advected front (below 1e-3), one step already lands below 1e-7.
   */
  int iterations;

  /**
   * @brief Optional constraint for rim vertices, applied after every step: the
   *        same contract as #MFTL_RemeshParameters::rim_project. #MFTL_NULL for
   *        none -- a rim vertex then only slides within the wall it lies on.
   */
  int (*rim_project)(const MFTL_Domain *dom, double pos[3], void *user);
  void *rim_project_user;  ///< Passed to `rim_project`.
} MFTL_VolumeCorrectionParameters;

/**
 * @struct MFTL_VolumeCorrectionStats
 * @brief What one call did. Every field is GLOBAL, identical on every rank.
 */
typedef struct MFTL_VolumeCorrectionStats
{
  int n_bodies;  ///< Bodies corrected: a reference volume and a defined volume.
  int n_bodies_unmeasured;  ///< Bodies with a reference volume but no defined volume (MFTL_BodyIntegrals::volume_status).
  double max_error_before;  ///< Largest |V / V_ref - 1| on entry.
  double max_error_after;  ///< Largest |V / V_ref - 1| on return.
  double max_offset;  ///< Largest |total normal offset| applied to one body.
  double t_total;  ///< Seconds in the call (max over ranks).
} MFTL_VolumeCorrectionStats;

/**
 * @brief The defaults: three Newton steps, no rim constraint.
 * @return The parameter set to start from.
 */
MFTL_VolumeCorrectionParameters MFTLVolumeCorrectionDefaultParameters(void);

/**
 * @brief Sets every body's #MFTL_Object::reference_volume to its volume now.
 *
 * The volume is MFTLEnsembleBodyIntegrals()'s, so closed bodies and open ones
 * closed by the walls alike; a body whose volume is undefined gets NaN. The
 * value is bit-identical on every holder.
 *
 * @param ens The ensemble.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for a NULL ensemble.
 * @collective Ranks holding no bodies must call it too.
 */
int MFTLEnsembleSetReferenceVolumes(MFTL_Ensemble *ens);

/**
 * @brief Restores the volume of every body that has a reference volume, by a
 *        uniform offset along its area-weighted vertex normals.
 *
 * The region is MFTLEnsembleBodyIntegrals()'s. Each step moves every owned
 * vertex by \f$d\,\vect{u}\f$ with \f$\vect{u}\f$ from MFTLBodyVolumeDisplacement()
 * and \f$d = (V_\mathrm{ref} - V)/\sum_t \vect{S}_t\cdot\bar{\vect{u}}_t\f$ --
 * Newton's method on the exact first variation of the volume of the field
 * actually applied. Dividing by the AREA instead (mftl's VolumeRecovery) is
 * exact only when the vertex normals are the face normals, and leaves an error
 * of the order of the edge length times the curvature in every step. A rim
 * vertex slides within its wall, then `rim_project` runs. Ghost positions are
 * refreshed first and after every step, so they are exact on return; triangle
 * geometry is stale and `verts.normal` is unspecified.
 *
 * A uniform offset is the correction that changes a shape least for a given
 * volume (a sphere stays a sphere), but it is GLOBAL: it puts back volume a
 * remesh or an advection lost locally by moving the whole surface, and a
 * deforming flow carries the misplacement. Measured over the LeVeque period
 * (docs/analysis.md §15.1): without back-projection it improves the position and
 * leaves the curvature alone; with the Phong projection, which already holds the
 * volume to 0.18 %, it doubles the curvature roughness at the end. Once per step,
 * after the remesh, is enough -- a second call after the advection changed
 * nothing measurable.
 *
 * @param ens   The ensemble.
 * @param p     Parameters; #MFTL_NULL for MFTLVolumeCorrectionDefaultParameters().
 * @param stats What was done, or #MFTL_NULL.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on every rank for a NULL ensemble or
 *         `iterations < 0`.
 *
 * @collective Every rank, including ranks holding no body; `iterations` must agree.
 */
int MFTLEnsembleCorrectVolumes(MFTL_Ensemble *ens, const MFTL_VolumeCorrectionParameters *p, MFTL_VolumeCorrectionStats *stats);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_CORRECTION_H
