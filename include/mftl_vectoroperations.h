/*
 * MFTL2 core — small 3-vector helpers (normalisation, dot and cross products).
 */

#ifndef MFTL_VECTOROPERATIONS_H
#define MFTL_VECTOROPERATIONS_H

#include <math.h>

#include "mftl_constants.h"

/**
 * @brief Normalises a 3-vector in place; the squared length is floored at MFTL_VERYSMALL, so a
 * (near-)zero vector stays (near-)zero instead of dividing by zero.
 */
static inline void MFTLNormalizeVector(double vec[3])
{
  const double inv_mag = 1.0 / MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, MFTL_SQ(vec[0]) + MFTL_SQ(vec[1]) + MFTL_SQ(vec[2])));
  vec[0] *= inv_mag;
  vec[1] *= inv_mag;
  vec[2] *= inv_mag;
}

/** @brief Dot product V1 . V2. */
static inline double MFTLVectorInProd(double V1[3], double V2[3]) { return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]); }

/** @brief Cross product Vr = V1 x V2 (Vr must not alias V1 or V2). */
static inline void MFTLVectorCrossVector(double *Vr, double *V1, double *V2)
{
  Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
  Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
  Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
}
#endif