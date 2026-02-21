/*
 * @Author: Christian Gorges 
 * @Date: 2025-08-09 14:35:27 
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
 */
#ifndef MFTL_CONSTANTS_H
#define MFTL_CONSTANTS_H

#include <stdbool.h>
#include <math.h>

#define MFTL_SUCCESS 0
#define MFTL_ERROR -1

#define MFTL_NULL (NULL)

// Boolean aliases if <stdbool.h> is not enough or for preference
#define MFTL_TRUE (true)
#define MFTL_FALSE (false)

#define MFTL_PI 3.14159265358979323846
#define MFTL_TWO_PI 6.28318530717958647692

// Tolerance for geometric checks
#define MFTL_EPSILON 1e-9
#define MFTL_VERYSMALL 1e-12
#define MFTL_VERYVERYSMALL 1e-15

// Math Macros
#define MFTL_MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MFTL_MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MFTL_SQ(x) ((x) * (x))
#define MFTL_SQRT(x) sqrt(x)
#define MFTL_ABS(x) (((x) < 0) ? -(x) : (x))
#define MFTL_COS(x) cos(x)
#define MFTL_SIN(x) sin(x)

#define MFTL_ONE_THIRD 0.33333333333333333333

#define MFTL_MAX_VALENCE 32

// Memory Management
#define MFTL_INIT_CAPACITY 128

// =============================================================================
// MESH FLAGS (Bitmask values - must be powers of 2)
// =============================================================================
#define MFTL_FLAG_NONE 0  // 0000 0000
#define MFTL_FLAG_DEAD (1 << 0)  // 0000 0001 - Element is deleted/scheduled for garbage collection
#define MFTL_FLAG_MODIFIED (1 << 1)  // 0000 0010 - Element was touched (for next pass/incremental updates)
#define MFTL_FLAG_BOUNDARY (1 << 2)  // 0000 0100 - Element is on a geometric boundary
#define MFTL_FLAG_VISITED (1 << 3)  // 0000 1000 - Element evaluated in current remeshing pass
#define MFTL_FLAG_FEATURE (1 << 4)  // 0001 0000 - Element is a sharp feature (do not smooth across)

#endif