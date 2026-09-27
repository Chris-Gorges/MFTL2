/*
 * MFTL2 remeshing (internal) — helpers of the generalized (boundary-aware) remesher
 * (src/remeshing/boundary_remesh.c; plans/16 §4.3, §4.4). Not part of the public API; it stays in
 * include/ (and is installed) because the test suite uses it (tests/test_stage16c.c).
 */
#ifndef MFTL_BOUNDARY_INTERNAL_H
#define MFTL_BOUNDARY_INTERNAL_H

#include "mftl_core.h"
#include "mftl_remeshing.h"

/* corner threshold of the remesh parameters (degrees; default 30) */
double MFTLBoundaryCornerAngle(const MFTL_RemeshParameters *params);
/* 1 corner (or unknown), 0 regular boundary vertex, -1 interior vertex */
int MFTLBoundaryVertexKind(const MFTL_Object *obj, int v, double corner_angle_deg);
/* boundary-curve point at t on the boundary half-edge he (a -> b), in a's local frame */
int MFTLBoundaryCurvePoint(const MFTL_Object *obj, int he_ab, double t, double corner_angle_deg, double out[3]);
/* set v's position to p (a point in some local frame near v): minimum-image move, counters follow */
void MFTLSetVertexPosFromFrame(MFTL_Object *obj, int v, const double p[3]);
/* augmented (virtual-vertex) link condition of the collapse of he */
int MFTLAugmentedLinkConditionOk(const MFTL_Object *obj, int he_idx);
/* survivor target of a generalized collapse (b's frame); kind 0 free, 1 kept, 2 boundary curve */
int MFTLGeneralizedCollapseTarget(const MFTL_Object *obj, int he_idx, double corner_angle_deg, double target[3], int *kind);
/* 1 if no surviving fan triangle of the collapse inverts with the survivor at target */
int MFTLCollapseKeepsOrientation(const MFTL_Object *obj, int he_idx, const double target[3]);
/* boundary split: new vertex on the boundary curve (a's frame); MFTL_ERROR for interior edges */
int MFTLGeneralizedSplitTarget(const MFTL_Object *obj, int he_idx, double corner_angle_deg, double target[3]);
/* 1 if moving v from `from` to `to` (v's local frame) inverts no stored fan triangle of v */
int MFTLVertexFanOwned(const MFTL_Object *obj, int v);
int MFTLMoveKeepsOrientation(const MFTL_Object *obj, int v, const double from[3], const double to[3]);
/* valence target of a boundary vertex: max(2, round(theta / 60 deg) + 1) (6 otherwise) */
int MFTLBoundaryValenceTarget(const MFTL_Object *obj, int v);
/* 16C review: 1 if a generalized split of he keeps the opposite vertices' valence within the
   growth cap (remeshing_core.c) */
int MFTLGeneralizedSplitValenceOk(const MFTL_Object *obj, int he_idx);
/* one Jacobi sweep of boundary-vertex smoothing along the boundary curve; returns moves */
int MFTLSmoothBoundaryVertices(MFTL_Object *obj, double relaxation, double corner_angle_deg);

/* Class-guarded volume recovery of the remesh drivers (plans/16 §4.4.4). kind 0: no
   recovery (phase set not DEFINED, or skipped), 1: the body's own volume (closed and compact; the
   closed-surface path, unchanged), 2: set-level recovery of the singleton phase set (e.g. a closed body
   that wraps a periodic direction). */
typedef struct
{
  int kind;
  double target;
} MFTL_RecoveryTarget;
/* collective per body; counts the skipped bodies in params->stat_recovery_skipped */
void MFTLRemeshRecoveryTarget(MFTL_Object *obj, MFTL_RemeshParameters *params, MFTL_RecoveryTarget *t);
/* collective per body */
void MFTLRemeshRecover(struct MFTL_Ensemble *ens, MFTL_Object *obj, const MFTL_RecoveryTarget *t);

#endif
