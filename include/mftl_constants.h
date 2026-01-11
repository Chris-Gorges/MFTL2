/*
 * @Author: Christian Gorges
 * @Date: 2025-08-10 10:00:56
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-06 15:43:58
 */

#ifndef MFTL_CONSTANTS_H
#define MFTL_CONSTANTS_H

#include <math.h>
#include <stdlib.h>

//------------------------------------------------------------------------------
// Macros and Constants
//------------------------------------------------------------------------------

#define MFTL_NULL (NULL)
#define MFTL_TRUE (true)
#define MFTL_FALSE (false)
#define MFTL_VERYSMALL (1.0e-15)

#define MFTL_SUCCESS 0
#define MFTL_ERROR -1

/* Generic definitions */
#define MFTL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MFTL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MFTL_SQ(x) ((x) * (x))
#define MFTL_SQRT(x) (sqrt(x))
#define MFTL_ABS(x) (((x) < 0) ? -(x) : (x))

#define MFTL_ONE_THIRD 0.33333333333333333333

#endif