/*
 * @Author: Christian Gorges
 * @Date: 2025-08-10 10:04:03
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-08-10 14:09:06
 */

#ifndef MFTL_VECTOROPERATIONS_H
#define MFTL_VECTOROPERATIONS_H

#include <math.h>

#include "mftl_constants.h"

/**
 * @brief Normalizes a 3D vector in-place.
 * @param vec The vector to be normalized.
 */
static inline void MFTLNormalizeVector(double vec[3])
{
  // Avoid division by zero and sqrt on tiny numbers
  const double inv_mag = 1.0 / MFTL_SQRT(MFTL_MAX(MFTL_VERYSMALL, MFTL_SQ(vec[0]) + MFTL_SQ(vec[1]) + MFTL_SQ(vec[2])));
  vec[0] *= inv_mag;
  vec[1] *= inv_mag;
  vec[2] *= inv_mag;
}

/**
 * @brief Calculates the dot product of two vectors.
 * @param V1 Vector 1.
 * @param V2 Vector 2.
 */
static inline double MFTLVectorInProd(double V1[3], double V2[3]) { return (V1[0] * V2[0] + V1[1] * V2[1] + V1[2] * V2[2]); }

/**
 * @brief Calculates the crossproduct of two vectors.
 * @param V1 Vector 1.
 * @param V2 Vector 2.
 * @param Vr The resulting crossproduct.
 */
static inline void MFTLVectorCrossVector(double *Vr, double *V1, double *V2)
{
  Vr[0] = V1[1] * V2[2] - V1[2] * V2[1];
  Vr[1] = V1[2] * V2[0] - V1[0] * V2[2];
  Vr[2] = V1[0] * V2[1] - V1[1] * V2[0];
}
#endif