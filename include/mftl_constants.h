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

// Remeshing flags
#define MFTL_FLAG_NONE 0
#define MFTL_FLAG_DEAD (1 << 0)
#define MFTL_FLAG_BOUNDARY (1 << 1)

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

#define MFTL_ONE_THIRD 0.33333333333333333333

#define MFTL_MAX_VALENCE 32
#define MFTL_FLAG_VISITED (1 << 2)

// Memory Management
#define MFTL_INIT_CAPACITY 128

#endif