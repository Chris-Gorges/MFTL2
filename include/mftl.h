/***************************************************
 * mftl.h
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/
#ifndef MFTL_H_
#define MFTL_H_

#include "mftl-constants.h"
#include "uthash.h"
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* define a point in space */
struct MFTL_Point
{
  MFTL_DOUBLE x[3];
  MFTL_INT processor;
  MFTL_INT body;
  MFTL_INT address;
  MFTL_INT addresslocal;
  MFTL_BYTE Periodicity;

  /* For hashtable */
  ObjectIDType key;
  UT_hash_handle hh;
};

struct MFTL_Vertex
{
  MFTL_DOUBLE x[3];

  MFTL_INT processor;
  short move;
  char pbc;

  /* For hashtable */
  ObjectIDType key;
  UT_hash_handle hh;
};

/* Define the structure for an Edge: a connection between two vertices. An edge
 * is also the barrier between two triangles and contains two pointers to the
 * triangles
 */
struct MFTL_Edge
{
  struct MFTL_Vertex *Vertex[2];
  MFTL_INT VertexID[2];
  struct MFTL_Triangle *Triangle[2];
  MFTL_INT TriangleID[2];

  /* The processor which has the original copy of this vertex */
  MFTL_INT processor;
  /* Processor number of triangles' processors of which the edge belong to */
#ifdef MFTL_DEV
  int procNo;
#endif

  /* For hashtable */
  ObjectIDType key;
  UT_hash_handle hh;
};

/* Defines the structure of a triangle. A triangle consists of 3 edges and 3
 * vertices */
struct MFTL_Triangle
{
  struct MFTL_Edge *Edge[3];
  struct MFTL_Vertex *Vertex[3];

  void *Data;

  /* The processor which has dominance over this triangle */
  MFTL_INT mainprocessor;

  /* The processor which owns the center point of this triangle */
  MFTL_INT centerproc;

  MFTL_DOUBLE center[3];
  MFTL_DOUBLE normal[3];
  MFTL_DOUBLE area;

  /* For hashtable */
  ObjectIDType key;
  UT_hash_handle hh;
};

struct MFTL_Object_size
{
  long unsigned int MFTL_arrayindex[3];
  long unsigned int MFTL_sizeofobject;
};

/* An object contains a number of triangles and also has pointers to the edges
 * and vertices */
struct MFTL_Object
{
  /* Communicator on object */
  MPI_Comm *OBJECT_COMM_WORLD;

  /* Sizes of one part in the object */
  struct MFTL_Object_size Size;

  struct MFTL_Triangle *Triangles;
  struct MFTL_Edge *Edges;
  struct MFTL_Vertex *Vertexes;

  /* Current processor */
  MFTL_INT rank;

  /* Total number of processors */
  MFTL_INT nproc;

  /* Min, Max location of object */
  MFTL_DOUBLE globalxmin[3], globalxmax[3];

  /* Number of "cuts" in each direction */
  MFTL_DOUBLE Xcut, Ycut, Zcut;

  MFTL_DOUBLE *ownershiprangemin[3], *ownershiprangemax[3];

  MFTL_DOUBLE SizeOfData;
  /* For hashtable */
  ObjectIDType key;
  UT_hash_handle hh;
};

typedef struct MFTL_Point MFTL_Point;
typedef struct MFTL_Triangle MFTL_Triangle;
typedef struct MFTL_Vertex MFTL_Vertex;
typedef struct MFTL_Edge MFTL_Edge;
typedef struct MFTL_Object MFTL_Object;
typedef struct MFTL_Object_size MFTL_Object_size;

#endif