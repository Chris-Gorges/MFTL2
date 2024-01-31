/***************************************************
 * mftl-constants.h
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/
#ifndef MFTLCONSTANTS_H_
#define MFTLCONSTANTS_H_

#include <math.h>
#include <stdlib.h>
#include <mpi.h>

typedef int MFTL_INT;
typedef double MFTL_DOUBLE;
typedef long MFTL_LONG;
typedef unsigned char MFTL_BYTE;
typedef int MFTL_ErrorCode;
typedef int ObjectIDType;
typedef short MFTL_BOOL;

extern MPI_Datatype MFTL_MPI_DOUBLE;
extern MPI_Datatype MFTL_MPI_INT;

#define MFTL_STRINGLENGTH (600)
#define MFTL_SHORTSTRING (20)

#define MFTL_NULL (NULL)

#define MFTL_YES (1)
#define MFTL_NO (0)

#define MFTL_PREALLOCATION (100)

/* Maximum number of vertices per processor */
#define MF_PROC_JUMP (1000000)

/* Some "general" sizes */
#define MFTL_SMALL (1.0e-7)
#define MFTL_VERYSMALL (1.0e-15)
#define MFTL_VERYVERYSMALL (1.0e-60)
#define MFTL_LARGE (1.0e+10)
#define MFTL_VERYLARGE (1.0e+13)
#define MFTL_LARGE_INT (100000000)
#define MFTL_ONETHIRD (0.3333333333333333e0)
#define MFTL_TWOTHIRD (0.6666666666666667e0)
#define MFTL_ONESIXTH (0.1666666666666667e0)

/* Generic definitions */
#define MFTL_MAX(a, b) ((a) > (b) ? (a) : (b))
#define MFTL_MIN(a, b) ((a) < (b) ? (a) : (b))
#define MFTL_SQ(x) ((x) * (x))
#define MFTL_SQRT(x) (sqrt(x))
#define MFTL_ABS(x) (((x) < 0) ? -(x) : (x))

/* Pi-related constants up to double precision*/
#define MFTL_PI (3.1415926535897932e0)
#define MFTL_TWO_PI (6.2831853071795864e0)
#define MFTL_HALF_PI (1.5707963267948966e0)
#define MFTL_RAD_TO_DEG (57.2957795130823201e0)
#define MFTL_DEG_TO_RAD (1.7453292519943296e-2)

#define MFTL_CENTER_DISTRIBUTED (1)
#define MFTL_VERTEX_DISTRIBUTED (2)

// Periodic B.C.
#define MFTL_PERNONE (0)  // (0000 0000)
#define MFTL_PERPX (1)  // (0000 0001)
#define MFTL_PERMX (2)  // (0000 0010)
#define MFTL_PERPY (4)  // (0000 0100)
#define MFTL_PERMY (8)  // (0000 1000)
#define MFTL_PERPZ (16)  // (0001 0000)
#define MFTL_PERMZ (32)  // (0010 0000)

enum MFTL_DIAGNOSIS
{
  // General error/success values
  MFTL_SUCCESS = 0,
  MFTL_VALENCE3_SUCCESS = 0,
  MFTL_ERROR = -100000,
  MFTL_SKIP,
  MFTL_VALENCE3,

  // remeshing-related diagnosis values
  MFTL_VERTEX_ANGLE_THRESHOLD_FAIL,  // angle exceeds threshold value (above
  // or below)
  MFTL_ELEMENT_ANGLE_THRESHOLD_FAIL,  // angle exceeds threshold value (above
  // or below)
  MFTL_VALENCE_FAIL,  // vertex exceeds the number of neighbouring vertices
  MFTL_SLAVE_PROC,  // perform remeshing tasks of ghost edges
  MFTL_MASTER_PROC,  // waits for the hardworker to send the processed ghost
  // edges and cells

  MFTL_EDGE_LENGTH_FAIL,  // edge length does not meet the requirements for a
  // given operation

  MFTL_GHOST_VERTEX_FAIL,  // a vertex has a ghost cell linked to it

  MFTL_NNEIGHBOURS_FAIL,

  // edge classification
  MFTL_BDRY_EDGE,  // edge located at the boundary of the domain; applies to
  // physical- or processor- boundaries.
  MFTL_INNER_EDGE,  // edge is not located at any boundary.
  MFTL_GHOST_EDGE,  // edge crosses an inter-processor boundary.

  // triangle classification
  MFTL_GHOST_TRIANGLE,  // triangle has one or more ghost vertices
  MFTL_LOCAL_TRIANGLE,  // triangle has only local vertices
  MFTL_REFINE,
  MFTL_COARSE,

};

enum MFTL_MASTER_RANK
{
  MFTL_MASTER_RANK_ZERO = 0,
  MFTL_MASTER_RANK_HAS_LESS_CELLS = -1,
  MFTL_MASTER_RANK_HAS_MORE_CELLS = 1
};

enum MFTL_TRIANGLE_QUALITY
{
  MFTL_TRIANGLE_SHAPE = -100000,  // compare to an equilateral triangle. Default option
  MFTL_TRIANGLE_MAX_ANGLE,  // return the smallest angle in the triangle
  MFTL_TRIANGLE_MIN_ANGLE,  // return the largest angle in the triangle
  MFTL_TRIANGLE_ASPECT_RATIO,  // return the largest-to-smallest edge ratio
  MFTL_EDGE_LENGTH
};

enum MFTL_REMESH_CRITERIA
{
  // triangle data pointer classification
  MFTL_FRONTTRACKING = 1,

  MFTL_NO_FIT = 0,
  MFTL_PARABOLIC_FIT = 1,
  MFTL_BARYCENTRIC_FIT = 2,

  TRIANGLE_AREA_BASED_REMESHING = 0,
  BOTSCH_KOBBELT_STANDARD_REMESHING = 1,
  BOTSCH_KOBBELT_CURVATURE_REMESHING = 2,
  JIAO_REMESHING = 3,

  VOLUME_CONSERVING_LAPSMOOTHING = 1,
  TSUR3D = 2
};

#endif