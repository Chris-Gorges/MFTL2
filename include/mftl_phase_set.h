/*
 * MFTL2 phase sets — closed-phase quantities of periodic, open and wall-closed surfaces
 * (plans/16 §3.6, revision 3.2, frozen).
 *
 * A phase-boundary set S is a list of (body, orientation sign σ = ±1) entries plus optional wall
 * closures on ONE non-periodic axis. Every closed-phase query returns one of three statuses:
 *   MFTL_DEFINED      meaningful and evaluated;
 *   MFTL_UNDEFINED    the topology/physics does not define it (Φ(Ŝ) ≠ 0, inconsistent
 *                     orientation, open member not closed by a declared wall, degenerate volume);
 *   MFTL_UNSUPPORTED  mathematically defined, but this implementation cannot evaluate it (no
 *                     admissible axis, |Φ_k(c)| ∉ {0, A_k}, winding wall loops, ambiguous closure,
 *                     no generic line, ...).
 * No query ever returns a value with a non-DEFINED status.
 *
 * Every function is COLLECTIVE over the domain communicator: the set's triangles are gathered to
 * every rank and the §3.6 evaluation runs deterministically and identically on all ranks.
 */
#ifndef MFTL_PHASE_SET_H
#define MFTL_PHASE_SET_H

#include "mftl_core.h"

typedef enum
{
  MFTL_DEFINED = 0,
  MFTL_UNDEFINED = 1,
  MFTL_UNSUPPORTED = 2
} MFTL_PhaseStatus;

#define MFTL_PHASE_OUTSIDE 0
#define MFTL_PHASE_INSIDE 1
#define MFTL_PHASE_ON_BOUNDARY 2

#define MFTL_WALL_MIN 0
#define MFTL_WALL_MAX 1

typedef struct MFTL_PhaseSet MFTL_PhaseSet;

typedef struct MFTL_PhaseSetInfo
{
  MFTL_PhaseStatus status;
  char reason[200];  ///< why a non-DEFINED status was returned (empty when DEFINED)
  double Phi[3];  ///< Φ(Ŝ) of the members (+ selected wall patches)
  int axis;  ///< evaluation axis k (−1 if none)
  int n_components;  ///< components of Ŝ
  int n_layers;  ///< layer components along k
  int n_compact;  ///< compact components along k
  int lambda_E;  ///< exterior level (−1 when layers exist)
  int n_phase_components;  ///< N_ph (§3.6.7)
  int wall_n[2];  ///< selected full-wall multiplicities n_W (min, max), −1 if no closure
  double volume;  ///< V when DEFINED
  int member_line_tries_max;  ///< largest number of member-line candidates used by a compact
} MFTL_PhaseSetInfo;

/** @brief Empty set on the domain of its members (collective creation not required). */
MFTL_PhaseSet *MFTLPhaseSetCreate(MFTL_Domain *domain);
void MFTLPhaseSetDestroy(MFTL_PhaseSet **S);

/** @brief Adds a body with orientation sign sigma (+1: the phase is on the side into which its
 *  stored normals do NOT point). Every connected component of the body is a member. */
int MFTLPhaseSetAddMember(MFTL_PhaseSet *S, MFTL_Object *body, int sigma);

/** @brief Declares a wall closure on the (non-periodic) axis, side MFTL_WALL_MIN / MAX. */
int MFTLPhaseSetAddWallClosure(MFTL_PhaseSet *S, int axis, int side);

/** @brief Full §3.6 classification (collective). */
MFTL_PhaseStatus MFTLPhaseSetClassify(MFTL_PhaseSet *S, MFTL_PhaseSetInfo *info);

/** @brief Phase volume (collective). *V is set only when DEFINED. */
MFTL_PhaseStatus MFTLPhaseSetVolume(MFTL_PhaseSet *S, double *V);

/** @brief Phase centre per axis (collective): status[a] per axis; C[a] (modulo the period on a
 *  periodic axis, in [min, max)) only where status[a] is DEFINED. Returns the set status (the
 *  volume's status). */
MFTL_PhaseStatus MFTLPhaseSetCentre(MFTL_PhaseSet *S, double C[3], MFTL_PhaseStatus status[3]);

/** @brief Inside/outside of a point (collective; Q is the same on every rank). *where is one of
 *  MFTL_PHASE_OUTSIDE / INSIDE / ON_BOUNDARY when DEFINED. */
MFTL_PhaseStatus MFTLPhaseSetInside(MFTL_PhaseSet *S, const double Q[3], int *where);

/** @brief Set-level volume recovery (collective; plans/16 §4.4.4). Every member moves
 *  by a uniform displacement delta along its phase-outward vertex normal (sigma * stored normal;
 *  boundary vertices on wall closures stay fixed), delta = (V_target - V) / A(S), iterated (at
 *  most 8 steps) until |V - V_target| <= 1e-12 V_cell. `ens` holds the members (ghost refresh).
 *  Returns the status of the set; *V_out (may be NULL) receives the final volume when DEFINED. */
MFTL_PhaseStatus MFTLPhaseSetVolumeRecovery(MFTL_PhaseSet *S, struct MFTL_Ensemble *ens, double V_target, double *V_out);

/** @brief Evaluation parameters (tolerances of §3.6; defaults as the executable model p36.py). */
typedef struct MFTL_PhaseSetTolerances
{
  double eta_b, eta_n;  ///< genericity: min barycentric, min |n_k|/|n|
  double tau_phi_rel;  ///< τ_Φ / A_k
  double tau_v_rel;  ///< τ_V / V_cell
  double tau_d_rel;  ///< ON_BOUNDARY distance / max period
  int n_try;  ///< candidates per line kind
} MFTL_PhaseSetTolerances;
void MFTLPhaseSetGetTolerances(const MFTL_PhaseSet *S, MFTL_PhaseSetTolerances *t);
void MFTLPhaseSetSetTolerances(MFTL_PhaseSet *S, const MFTL_PhaseSetTolerances *t);

#endif
