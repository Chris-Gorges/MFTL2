/*
 * MFTL2 remeshing (internal) — declarations shared by the serial remesher sources
 * (remeshing_core.c, remesh_smoothing.c). Not part of the public API and not installed.
 */
#ifndef MFTL_REMESHING_INTERNAL_H
#define MFTL_REMESHING_INTERNAL_H

#include "mftl_core.h"

// Valence gate of an edge flip (remeshing_core.c); FlipApprovedWithParams adds the quality and
// dihedral gates.
int mftlFlipApproved(MFTL_Object *obj, int he_idx);

#endif  // MFTL_REMESHING_INTERNAL_H
