/*
 * MFTL2 intersection — object–object overlap-volume quantification (MFTL1 parity, plans/11)
 * and 2-D convex-hull / polygon helpers.
 *
 * Reimplements MFTL1's overlap.c pipeline per triangle pair:
 *   orthonormal frame from the A->B separation (GS_OrthoNormalise) ->
 *   project both triangles into the plane normal to vecAB -> per-triangle
 *   in-plane height-gradient fit + osculating-radius fit -> 2D clip of the
 *   two projected triangles (interior vertices with MFTL_REL_TOLERANCE
 *   slack + edge-edge intersections) -> CCW ordering -> fan triangulation
 *   -> per-fan prism volume 1/3*area*sum(hA - hB) with exact height
 *   zero-crossings -> map the centre of volume back to real space,
 *   AverageSurfaceNorm = 0.5*(N_A - N_B) with OUTWARD normals (project
 *   invariant), EffectiveRadiusOfCurvature = the blended osculating radius.
 *
 * Deviations from MFTL1 (required by the overlap acceptance tests, recorded
 * in plans/11):
 *   1. Fan-triangulation integrates clip polygons with ANY corner count
 *      (MFTL1's `if (3 >= dis_corners)` gate silently dropped the generic
 *      4-6-corner triangle intersection regions).
 *   2. The 1/2-intersect cut-point ratio uses the exact linear zero-
 *      crossing diff(i)/(diff(i) - diff(x)) (MFTL1 used diff(i)/(-diff(x))).
 *   3. The CCW ordering is a plain angular sort (MFTL1's quadrant scan had
 *      a partial-termination path that returned unordered prefixes).
 *   4. The projection axis is w = +vecAB/|vecAB| (A->B): the lens integrand
 *      h_A - h_B >= 0 then selects exactly the overlap region and every
 *      far-side cross-pair is sign-rejected (verified against the analytic
 *      two-sphere lens, converging under refinement).
 */

#ifndef MFTL_OVERLAP_H
#define MFTL_OVERLAP_H

#include "mftl_core.h"
#include "mftl_bvh.h"

// MFTL1 parity (overlap.c): additive slack in the half-plane/edge tests
// (used squared) and the 2D intersection near-parallel classification.
#define MFTL_REL_TOLERANCE 5.0e-7

/**
 * @struct MFTL_Overlap
 * @brief Overlap-volume record for one intersecting triangle pair.
 * Fields mirror MFTL1's struct MFTL_Overlap (uthash handle dropped).
 */
typedef struct MFTL_Overlap
{
  double Volume;
  double AreaTriA;  // overlap area mapped back onto sheet A
  double AreaTriB;  // ... and onto sheet B

  double AverageSurfaceNorm[3];   // (Normal_A - Normal_B)/2, outward normals
  double RealCentreOfVolume[3];   // centre of the overlap prism in real space
  double EffectiveRadiusOfCurvature;

  int ID_A;  // global triangle id of the overlapping pair
  int ID_B;

  int id;  // running counter, assigned deterministically after the gather
} MFTL_Overlap;

/**
 * @brief Compute the overlap volume between one triangle of objA and one of
 * objB (MFTL1 MFTL_CalculateOverlap, pair-grained).
 *
 * The frame is built from vecAB (the A->B separation; for periodic domains
 * pass the minimum-image vector so cross-seam pairs are handled like
 * mid-domain pairs). Triangle normals must be current
 * (MFTLComputeTriangleProperties) — outward normals per the project
 * invariant.
 *
 * @return the number of sub-volumes integrated (0 = no volume overlap,
 *         e.g. the pair only touches); the summed record is in `out`.
 */
int MFTLCalculateOverlap(MFTL_Object* objA, int triA, MFTL_Object* objB, int triB, const double vecAB[3], MFTL_Overlap* out);

/**
 * @brief Ensemble-level overlap driver (collective): for each object pair (i<j for
 * ensB==NULL, all i,j otherwise), gather the candidate triangles of both
 * bodies to every rank (MFTL1 MFTL_BodyOverlap cross-proc exchange: owned
 * triangles whose AABB meets the other body's big box) and compute the
 * per-pair overlap records, each exactly once globally (on the rank owning
 * the body-A candidate).
 *
 * The brute-force candidate gather (accepted in plans/11) is used
 * instead of the proximity BVH because the volume prism spans triangle pairs
 * whose sheets are far apart in 3D but overlap in the projection plane —
 * MFTL_IntersectObjects only sees genuinely touching pairs (slivers).
 *
 * For periodic domains the packed candidate positions are expressed in a
 * common unwrapped frame (each body unwrapped about its own surface centre,
 * body B additionally shifted by the minimum image of cmB - cmA); positions
 * are replaced only temporarily and restored, so ownership/ghost state stays
 * consistent on any rank count. Every body must be compact (a wrapping body is
 * rejected with MFTL_ERROR) and lie within half a period of its surface centre
 * along each periodic axis (an overlap-specific restriction, as in MFTL1).
 *
 * Records are Allgatherv'd so every rank returns the full, identical,
 * deterministically ordered list.
 *
 * @param ensA primary ensemble (required).
 * @param ensB secondary ensemble, or NULL to overlap ensA with itself
 *             (object pairs i < j, matching MFTLEnsembleIntersectObjects).
 * @param out_records malloc'd array of records (caller frees); identical on
 *                    every rank.
 * @param out_count number of records.
 */
int MFTLEnsembleCalculateOverlap(MFTL_Ensemble* ensA, MFTL_Ensemble* ensB, MFTL_Overlap** out_records, int* out_count);

/* ── 2D convex hull helpers (Graham scan; public, plans/11) ── */

typedef struct
{
  double x, y;
} MFTL_Point2D;

/**
 * @brief Convex hull of a 2D point set (Graham scan). On return `pts`
 * holds `*n` hull vertices ordered CCW, starting at the lowest-rightmost
 * point. Input may be modified (sorted in place); interior points are
 * removed. Degenerate inputs (collinear / coincident) return 0 with *n <= 2.
 */
int MFTLConvexHull2D(MFTL_Point2D* pts, int* n);

/** @brief Signed area of a CCW-ordered polygon (negative if CW). */
double MFTLPolygonArea2D(const MFTL_Point2D* pts, int n);

/** @brief Centroid of a CCW-ordered polygon (fan triangulation). */
void MFTLPolygonCentroid2D(const MFTL_Point2D* pts, int n, double centre[2]);

#endif  // MFTL_OVERLAP_H
