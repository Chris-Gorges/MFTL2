/**
 * @file mftl_constants.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_constants
 *
 * @brief Scalar constants, math macros and the assertion policy.
 *
 * Layer 0: knows nothing about MFTL types, included by everything.
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
 * Keyed off `NDEBUG`, which `CMakeLists.txt` defines unconditionally for a
 * Release build so the policy cannot be half-applied.
 *
 * @warning Never put a side effect inside it. In Release the expression is not
 *          merely un-checked, it is not evaluated at all -- the macro expands to
 *          `((void) 0)` and the argument disappears with it.
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
 * "no such entity" index sentinel, which is why it is @c -1 rather than any
 * other non-zero value: it is never a valid pool index, so one constant serves
 * both jobs and a caller cannot confuse a failed lookup with a valid slot 0.
 * @{
 */

/** @brief Function completed successfully. */
#define MFTL_SUCCESS 0

/**
 * @brief Function failed, **or** "no such entity".
 *
 * As an index it means: no twin across this half-edge, no adjacent triangle, no
 * outgoing half-edge (an isolated vertex), no owner. Being negative, it also
 * fails any `0 <= i < n` bounds assertion, so a sentinel that escapes into an
 * index expression is caught in Debug rather than reading backwards off an array.
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
/** @brief \f$2\pi\f$. Precomputed because the angle sum of a closed fan is compared against it. */
#define MFTL_TWO_PI 6.28318530717958647692
/**
 * @brief \f$4\sqrt{3}\f$, the numerator of the mean-ratio triangle quality.
 *
 * \f[
 *   q = \frac{4\sqrt{3}\,A}{a^2 + b^2 + c^2}
 * \f]
 *
 * which is 1 for an equilateral triangle and tends to 0 as one degenerates.
 */
#define MFTL_FOUR_SQRT3 6.9282032302755088
/** @brief \f$1/3\f$, so a centroid is a multiply rather than a divide. */
#define MFTL_ONE_THIRD 0.33333333333333333
/** @} */

/**
 * @name Geometric tolerances
 *
 * Three of them, and they are **not** interchangeable. Each answers a different
 * question, and the scale of the answer follows from what is being compared:
 *
 * | Macro | Question |
 * |---|---|
 * | #MFTL_EPSILON | "are these two *lengths* the same length?" |
 * | #MFTL_VERYSMALL | "is this *squared* quantity zero?" |
 * | #MFTL_VERYVERYSMALL | "is this divisor zero?" |
 *
 * @warning None of them is used to clamp the magnitude of a vector before
 *          normalisation. Clamping \f$\lVert v \rVert^2\f$ against
 *          #MFTL_VERYSMALL floors the magnitude at \f$10^{-6}\f$, and clamping
 *          the divisor against #MFTL_VERYVERYSMALL merely moves that floor. See
 *          MFTLNormalizeVector() for the full argument and the measured cost of
 *          doing it properly.
 * @{
 */
/** @brief Length-scale comparison tolerance, \f$10^{-9}\f$. */
#define MFTL_EPSILON 1e-9
/** @brief Squared-quantity zero test, \f$10^{-12}\f$ -- i.e. #MFTL_EPSILON squared in spirit. */
#define MFTL_VERYSMALL 1e-12
/** @brief Divisor zero test, \f$10^{-15}\f$, near the double-precision epsilon. */
#define MFTL_VERYVERYSMALL 1e-15
/** @} */

/**
 * @name Math macros
 *
 * Function-like macros, so every argument is evaluated more than once in the
 * comparison forms. Do not pass an expression with a side effect.
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
/** @brief Absolute value of @p x. @p x is evaluated twice. */
#define MFTL_ABS(x) (((x) < 0) ? -(x) : (x))
/** @brief Cosine of @p x, in radians. */
#define MFTL_COS(x) cos(x)
/** @brief Sine of @p x, in radians. */
#define MFTL_SIN(x) sin(x)
/** @brief Largest finite `double`. Used as the identity of a running minimum. */
#define MFTL_DBL_MAX DBL_MAX
/** @brief Smallest positive normalised `double`. */
#define MFTL_DBL_MIN DBL_MIN
/** @} */

/**
 * @brief Upper bound on the vertices, triangles or half-edges in a one-ring.
 *
 * Every fan walk writes into a caller-supplied array and stops at its capacity,
 * so this is the size those arrays are given on the stack. A regular
 * triangulation has valence 6; 32 leaves room for the irregular vertices a
 * subdivided icosahedron and a remeshed surface actually produce, while keeping
 * the buffer small enough to live in registers-plus-L1 rather than being
 * allocated.
 *
 * A fan longer than this is not silently truncated in Debug -- the one-ring
 * helpers assert on overflow.
 */
#define MFTL_MAX_VALENCE 32

/**
 * @brief Alignment, in bytes, that every SoA sub-array of a pool block starts at.
 *
 * Not a tuning knob. It is the granularity MFTLLayoutAdd() rounds to, and two
 * consequences make it load-bearing rather than cosmetic:
 *
 * - A streaming kernel over one component never shares a line with a
 *   neighbouring component, so a write to `tris.area` cannot false-share with a
 *   read of `tris.normal`.
 * - 64 bytes is a whole number of every element type in the pools (8 `double`,
 *   16 `int`, 4 #MFTL_Vertex records), so an aligned sub-array start is also an
 *   aligned *element* start and the vectoriser never needs a peel loop.
 *
 * 64 is the line size on x86-64 and on AArch64's normal configurations. A target
 * with 128-byte lines would want it raised; nothing breaks if it is not, the
 * separation guarantee just weakens.
 */
#define MFTL_CACHE_LINE 64u

/**
 * @brief Smallest pool capacity a geometric growth step will produce.
 *
 * MFTLGrowCapacity() starts here rather than at 1, so a mesh built entity by
 * entity performs \f$O(\log n)\f$ reallocations from a sensible floor instead of
 * seven doublings through capacities no real mesh stops at.
 */
#define MFTL_INIT_CAPACITY 128

/**
 * @brief Waste factor at which MFTLObjectShrinkToFit() gives a pool block back.
 *
 * A pool is shrunk when `capacity > MFTL_SHRINK_SLACK_DEFAULT * n`. Four is a
 * floor, not a taste: it is the PRODUCT of the two headroom factors, and
 * anything below it thrashes.
 *
 * Both resize operators leave `capacity` in \f$[n, 2n)\f$ -- MFTLGrowCapacity()
 * doubles from below its @p needed argument, so the result is under `2n`
 * whatever it started from, and the shrink targets the same ladder. A gate of
 * `capacity > 2n` is therefore satisfied by ANY decrease at all. Concretely,
 * with `capacity = 1024` and a body whose count dithers 1023 <-> 1025 under
 * advection: the step at 1025 grows to 2048, the step back at 1023 sees
 * `2048 > 2046` and shrinks to 1024, the next grows again -- a whole pool
 * reallocation and a memcpy of every sub-array, every timestep, for a body that
 * moved by two entities. That is exactly what the pre-grow in
 * src/communication/redistribute.c exists to prevent, and it is silent: nothing
 * fails, only the timing moves.
 *
 * At 4 the gate re-fires only if the count HALVES -- after a resize at
 * \f$n_0\f$ we have \f$C < 2 n_0\f$, so \f$C > 4 n_1\f$ needs
 * \f$n_1 < n_0 / 2\f$ -- which keeps reallocations
 * \f$O(\log(n_{max}/n_{min}))\f$ over a run. Nothing is lost on the case the
 * shrink exists for: a rank holding a sliver of a generated body wastes 25x,
 * not 3x.
 *
 * Pinned by `test_shrink_to_fit_does_not_thrash_against_the_grow_ladder`.
 */
#define MFTL_SHRINK_SLACK_DEFAULT 4.0

#endif  // MFTL_CONSTANTS_H
