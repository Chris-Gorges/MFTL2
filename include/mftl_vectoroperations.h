/**
 * @file mftl_vectoroperations.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_vectorops
 *
 * @brief Small fixed-size vector helpers.
 *
 * Layer 0: plain doubles, no MFTL types. Meant for scalar code paths; streaming
 * kernels write the arithmetic inline (see the `MFTL_FOREACH_*` iterators in
 * mftl_object.h) so the compiler sees the whole loop body.
 */

#ifndef MFTL_VECTOROPERATIONS_H
#define MFTL_VECTOROPERATIONS_H

#include <math.h>

#include "mftl_constants.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Normalises a 3-vector in place: the result has unit length, or is zero.
 *
 * \f$\vect{v} \leftarrow \vect{v}/\norm{\vect{v}}\f$, with the magnitude computed
 * UNCLAMPED and the zero vector mapped to itself:
 * `inv = (mag > 0) ? 1/mag : 0`. The result is unit to rounding whenever
 * \f$\norm{\vect{v}}^2\f$ is a normal double, i.e. for
 * \f$1.5\cdot10^{-154} \lesssim \norm{\vect{v}} \lesssim 1.3\cdot10^{154}\f$;
 * outside it the squared sum is subnormal, zero or infinite and the result loses
 * precision or becomes the zero vector. Zero is a detectable sentinel: a unit
 * vector never has zero length.
 *
 * Every normalisation in the library uses this same expression. The two
 * alternatives are both wrong for small vectors: clamping \f$\norm{\vect{v}}^2\f$
 * before the square root floors the divisor and returns vectors SHORTER than
 * unit (a vector of length \f$10^{-9}\f$ came out at \f$10^{-3}\f$), and
 * `1/max(mag, tiny)` merely moves that floor. A short normal still points the
 * right way, so every sign test passes while it silently scales whatever it
 * multiplies. The select costs one instruction (+1.1 % in the per-triangle
 * kernel). Guarded by `test_normalize_is_unit_across_magnitudes`.
 *
 * @param vec The vector, normalised in place.
 */
static inline void MFTLNormalizeVector(double vec[3])
{
  const double mag = MFTL_SQRT(MFTL_SQ(vec[0]) + MFTL_SQ(vec[1]) + MFTL_SQ(vec[2]));
  const double inv_mag = (mag > 0.0) ? 1.0 / mag : 0.0;
  vec[0] *= inv_mag;
  vec[1] *= inv_mag;
  vec[2] *= inv_mag;
}

/**
 * @brief Dot product of two 3-vectors.
 * @param v1 First vector.
 * @param v2 Second vector.
 * @return \f$\vect{v}_1 \cdot \vect{v}_2\f$.
 */
static inline double MFTLVectorInProd(const double v1[3], const double v2[3]) { return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]; }

/**
 * @brief Squared Euclidean length of a 3-vector.
 *
 * Prefer it to MFTLVectorNorm() wherever lengths are only compared.
 *
 * @param v The vector.
 * @return \f$\norm{\vect{v}}^2\f$.
 */
static inline double MFTLVectorNormSq(const double v[3]) { return v[0] * v[0] + v[1] * v[1] + v[2] * v[2]; }

/**
 * @brief Euclidean length of a 3-vector.
 * @param v The vector.
 * @return \f$\norm{\vect{v}}\f$.
 */
static inline double MFTLVectorNorm(const double v[3]) { return MFTL_SQRT(MFTLVectorNormSq(v)); }

/**
 * @brief Cross product, \f$\vect{out} = \vect{v}_1 \times \vect{v}_2\f$.
 *
 * @param[out] out Result. **Must not alias** @p v1 or @p v2 -- component 0 is
 *                 written before components 1 and 2 are read.
 * @param      v1  Left operand.
 * @param      v2  Right operand.
 */
static inline void MFTLVectorCrossVector(double out[3], const double v1[3], const double v2[3])
{
  out[0] = v1[1] * v2[2] - v1[2] * v2[1];
  out[1] = v1[2] * v2[0] - v1[0] * v2[2];
  out[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

#ifdef __cplusplus
}
#endif

#endif  // MFTL_VECTOROPERATIONS_H
