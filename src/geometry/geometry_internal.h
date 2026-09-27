/*
 * MFTL2 geometry (internal) — helpers shared by the geometry sources (geometry_object.c,
 * point_query.c). Not part of the public API and not installed.
 */
#ifndef MFTL_GEOMETRY_INTERNAL_H
#define MFTL_GEOMETRY_INTERNAL_H

#include "mftl_core.h"

/* Collective anchor of the single-body integrals and point queries (geometry_object.c): (pos,
   img) of the first vertex of the first owned triangle on the lowest rank owning a triangle,
   broadcast so every rank evaluates in the SAME periodic image. Returns 0 (outputs untouched)
   when the object has no owned triangle anywhere. */
int mftlObjectUnwrapAnchor(const MFTL_Object* obj, double anchor[3], int anchor_img[3]);

#endif  // MFTL_GEOMETRY_INTERNAL_H
