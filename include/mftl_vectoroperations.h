/**
 * @file mftl_vectoroperations.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_vectorops
 *
 * @brief Small fixed-size vector helpers.
 *
 * Layer 0: plain doubles in, plain doubles out, no MFTL types. All of these are
 * meant to be used on scalar, one-at-a-time code paths; a streaming kernel
 * should write the arithmetic out inline so the compiler sees the whole loop
 * body (see the MFTL_FOREACH_* iterators in mftl_object.h).
 */

#ifndef MFTL_VECTOROPERATIONS_H
#define MFTL_VECTOROPERATIONS_H

#include <math.h>

#include "mftl_constants.h"

/**
 * @brief Normalizes a 3D vector in-place. The result is UNIT LENGTH or zero.
 *
 * Exact for every input whose sum of squares is representable -- roughly
 * |v| in (1e-162, 7e153) -- and exactly zero outside it, which is a detectable
 * sentinel: a unit vector never has zero length, so "not normalisable" stays
 * distinguishable from "normalised".
 *
 * ── Why this is not the guard the triangle kernels use ─────────────────────
 *
 * Until 2026-08-15 this clamped the SQUARED length before the square root, which
 * floors the magnitude at sqrt(MFTL_VERYSMALL) = 1e-6: any shorter vector was
 * divided by the clamp rather than by its own length and came out proportionally
 * SHORT. A vector of length 1e-9 normalised to length 1e-3.
 *
 * That is the most dangerous shape of wrong in a physics kernel. The direction is
 * still right, so every sign test, every orthogonality check and every picture
 * still looks correct, while the magnitude silently scales whatever the vector
 * multiplies -- a surface tension force, an advection step, a projection.
 *
 * MFTLGetTriangleProperties and the two geometry kernels carried the same defect.
 * They were first "fixed" with `1.0 / MFTL_MAX(mag, MFTL_VERYVERYSMALL)`, which
 * keeps the instruction count identical (one max, one sqrt, one divide) but only
 * MOVES the floor: a triangle with 1e-8 edges still came out with a normal of
 * length 0.112. All of them now use the same unclamped test against zero that is
 * written here, so the expression is identical in every kernel in the library --
 * which is what `test_triangle_geometry_paths_are_bit_identical` checks.
 *
 * The select is one instruction more than a max. Measured in the per-triangle
 * kernel at **7.33 -> 7.41 ns/tri, +1.1 %** (best of six over two rounds), and
 * the floor disappears entirely rather than moving from 1e-6 to 1e-15. Pay it.
 *
 * @param vec The vector to normalize, updated in place. On return it has length
 *            1, or length 0 if it arrived as the zero vector (or with a
 *            non-representable sum of squares).
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
 * Preferred over MFTLVectorNorm() wherever the square root can be avoided --
 * comparing two lengths, or testing a length against a threshold, both work on
 * squares.
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

#endif  // MFTL_VECTOROPERATIONS_H
