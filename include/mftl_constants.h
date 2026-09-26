/**
 * @file mftl_constants.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_constants
 *
 * @brief Scalar constants, math macros and the assertion policy.
 *
 * Layer 0: no MFTL types; included by every other header.
 */

#ifndef MFTL_CONSTANTS_H
#define MFTL_CONSTANTS_H

#include <float.h> /* DBL_MAX */
#include <math.h>
#include <stdbool.h>

/**
 * @name Diagnostics
 * @{
 */

#ifdef NDEBUG
/**
 * @brief Debug-only assertion. Compiles to nothing in Release.
 *
 * Keyed off `NDEBUG`, which `CMakeLists.txt` defines for every Release build.
 *
 * @warning Never put a side effect inside it: in Release the argument is not
 *          evaluated at all.
 *
 * @param condition Expression that must hold. Not evaluated in Release.
 */
#define MFTL_ASSERT(condition) ((void) 0)
#else
#include <assert.h>
/**
 * @brief Debug-only assertion. Compiles to nothing in Release.
 * @param condition Expression that must hold. Not evaluated in Release.
 */
#define MFTL_ASSERT(condition) assert(condition)
#endif

/** @} */

/**
 * @name Return codes and sentinels
 *
 * Every MFTL function that can fail returns `int`. #MFTL_ERROR doubles as the
 * "no such entity" index sentinel; being @c -1 it is never a valid pool index.
 * @{
 */

/** @brief Function completed successfully. */
#define MFTL_SUCCESS 0

/**
 * @brief Function failed, **or** "no such entity".
 *
 * As an index it means: no twin across this half-edge, no adjacent triangle, no
 * outgoing half-edge (an isolated vertex), no owner. Being negative it fails every
 * Debug bounds assertion, so an escaped sentinel is caught rather than read.
 */
#define MFTL_ERROR -1

/** @brief Null pointer, spelled the library's way. */
#define MFTL_NULL (NULL)

/** @} */

/**
 * @name Boolean aliases
 * @{
 */
/** @brief `true`, spelled the library's way. */
#define MFTL_TRUE (true)
/** @brief `false`, spelled the library's way. */
#define MFTL_FALSE (false)
/** @} */

/**
 * @name Mathematical constants
 * @{
 */
/** @brief \f$\pi\f$. */
#define MFTL_PI 3.14159265358979323846
/** @brief \f$2\pi\f$. */
#define MFTL_TWO_PI 6.28318530717958647692
/**
 * @brief \f$4\sqrt{3}\f$, the numerator of the mean-ratio triangle quality.
 *
 * \f[
 *   q = \frac{4\sqrt{3}\,A}{a^2 + b^2 + c^2}
 * \f]
 *
 * with \f$A\f$ the area and \f$a,b,c\f$ the edge lengths: \f$q = 1\f$ exactly for
 * an equilateral triangle, \f$0 < q < 1\f$ otherwise, \f$q \to 0\f$ as it degenerates.
 */
#define MFTL_FOUR_SQRT3 6.9282032302755088
/** @brief \f$1/3\f$, so a centroid is a multiply rather than a divide. */
#define MFTL_ONE_THIRD 0.33333333333333333
/** @} */

/**
 * @name Tolerances
 *
 * The library itself uses only #MFTL_VERYVERYSMALL, and only as a
 * **dimensionless** threshold -- always multiplied by the product of the
 * quantities it compares, e.g. \f$|\vect{a}\times\vect{b}|^2 >
 * \varepsilon\,|\vect{a}|^2|\vect{b}|^2\f$, a test on \f$\sin^2\theta\f$
 * alone. No library decision compares a dimensional quantity against an absolute
 * constant, so results are invariant under a change of length unit
 * (`test_remeshing_is_independent_of_the_length_unit` checks this bit for bit).
 * #MFTL_EPSILON and #MFTL_VERYSMALL are absolute and are provided for
 * application code and tests only.
 *
 * @warning None of them may be used to clamp a vector magnitude before
 *          normalisation: that floors the magnitude and returns vectors that are
 *          shorter than unit. See MFTLNormalizeVector().
 * @{
 */
/** @brief Absolute comparison tolerance, \f$10^{-9}\f$. Tests and applications only. */
#define MFTL_EPSILON 1e-9
/** @brief Absolute small-quantity tolerance, \f$10^{-12}\f$. Tests and applications only. */
#define MFTL_VERYSMALL 1e-12
/** @brief Dimensionless degeneracy threshold, \f$10^{-15}\f$ (about \f$5\,\varepsilon_\mathrm{mach}\f$), applied to squared sines. */
#define MFTL_VERYVERYSMALL 1e-15
/** @} */

/**
 * @name Math macros
 *
 * Function-like macros: arguments may be evaluated twice, so never pass an
 * expression with a side effect.
 * @{
 */
/** @brief Smaller of @p a and @p b. Both are evaluated twice. */
#define MFTL_MIN(a, b) (((a) < (b)) ? (a) : (b))
/** @brief Larger of @p a and @p b. Both are evaluated twice. */
#define MFTL_MAX(a, b) (((a) > (b)) ? (a) : (b))
/** @brief @p x squared. @p x is evaluated twice. */
#define MFTL_SQ(x) ((x) * (x))
/** @brief Square root of @p x. */
#define MFTL_SQRT(x) sqrt(x)
/** @brief Cosine of @p x, in radians. */
#define MFTL_COS(x) cos(x)
/** @brief Largest finite `double`. Used as the identity of a running minimum. */
#define MFTL_DBL_MAX DBL_MAX
/** @brief Smallest positive normalised `double`. */
#define MFTL_DBL_MIN DBL_MIN
/** @} */

/**
 * @brief Upper bound on the valence handled by the one-ring walkers.
 *
 * Fan walks write into caller-supplied stack buffers of this size and the
 * rotation loops are bounded by it, so a corrupt (non-manifold) fan terminates.
 * Overflow asserts in Debug and truncates in Release; the remesher's collapse and
 * flip approvals refuse any operation that would exceed it (the regular valence
 * of a triangulated surface is 6).
 */
#define MFTL_MAX_VALENCE 32

/**
 * @brief Alignment, in bytes, of every sub-array of a pool block.
 *
 * MFTLLayoutAdd() rounds each sub-array start up to this boundary, so two
 * arrays that different kernels write (e.g. `tris.area` and `tris.normal`) never
 * share a cache line, and every sub-array starts on a vector-aligned address.
 * 64 B is the line size of current x86-64 and AArch64 cores; a 128-byte-line
 * target would only weaken the separation, not break anything.
 */
#define MFTL_CACHE_LINE 64u

/**
 * @brief Smallest capacity a geometric growth step produces (MFTLGrowCapacity()).
 *
 * A pool grown geometrically from empty follows the ladder \f$128 \cdot 2^k\f$,
 * so incremental growth costs \f$O(\log n)\f$ reallocations.
 */
#define MFTL_INIT_CAPACITY 128

/**
 * @brief Default waste factor above which MFTLObjectShrinkToFit() reallocates.
 *
 * A pool is shrunk when `capacity > slack * n`, to the smallest ladder capacity
 * \f$128\cdot2^k \ge n\f$. Both growth and shrink leave \f$C \in [n, 2n)\f$
 * (for \f$n \ge 128\f$), so a gate of 2 would fire on ANY decrease and a count
 * dithering around a power of two would reallocate every step. A gate of 4 is the
 * product of the two factor-2 headrooms: after a resize at \f$n_0\f$,
 * \f$C < 2n_0\f$, so \f$C > 4n_1\f$ requires \f$n_1 < n_0/2\f$, and the number
 * of reallocations over a run is \f$O(\log(n_{max}/n_{min}))\f$.
 *
 * Pinned by `test_shrink_to_fit_does_not_thrash_against_the_grow_ladder`.
 */
#define MFTL_SHRINK_SLACK_DEFAULT 4.0

#endif  // MFTL_CONSTANTS_H
