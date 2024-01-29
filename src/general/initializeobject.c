/***************************************************
 * finalizeobject.c
 * Created on Mon Jan 29 2024
 *
 *     Description: 1. This routine initialises an object, without inserting any elements in it.
 *                  2. This routine finalises the object and clears all elements.
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include <stdio.h>
#include "mftl.h"

MFTL_ErrorCode MFTL_InitializeObject(MFTL_Object *Object, MFTL_INT M[3], MFTL_DOUBLE xmin[3], MFTL_DOUBLE xmax[3])
{
  MFTL_INT ix[3];
  MFTL_DOUBLE dx[3];

  if (sizeof(MFTL_DOUBLE) == sizeof(double))
    MFTL_MPI_DOUBLE = MPI_DOUBLE;
  else
  {
    char error[] = "Accuracy of MPI_DOUBLE and MFTL_MPI_DOUBLE could not be determined. Needs a fix.";
    MFTLThrowError(1, error);
  }
  if (sizeof(MFTL_INT) == sizeof(int))
    MFTL_MPI_INT = MPI_INT;
  else if (sizeof(MFTL_INT) == sizeof(long))
    MFTL_MPI_INT = MPI_LONG;
  else
  {
    char error[] = "Accuracy of MPI_INT and MFTL_MPI_INT could not be determined. Needs a fix.";
    MFTLThrowError(1, error);
  }

  // Set the size of the object
  Object->Size.MFTL_arrayindex[0] = 0;
  Object->Size.MFTL_arrayindex[1] = sizeof(MFTL_Triangle);
  Object->Size.MFTL_arrayindex[2] = sizeof(MFTL_Triangle) + 3 * sizeof(MFTL_Edge);
  Object->Size.MFTL_sizeofobject = sizeof(MFTL_Triangle) + 3 * sizeof(MFTL_Edge) + 3 * sizeof(MFTL_Vertex);
  Object->SizeOfData = 0;

  // Initialise the inner lists to null
  Object->Edges = MFTL_NULL;
  Object->Triangles = MFTL_NULL;
  Object->Vertexes = MFTL_NULL;

  MFTL_AllocateMemory(sizeof(MPI_Comm), (void **) &Object->OBJECT_COMM_WORLD);

  Object->OBJECT_COMM_WORLD[0] = MPI_COMM_WORLD;
  MPI_Comm_size(Object->OBJECT_COMM_WORLD[0], &Object->nproc);
  MPI_Comm_rank(Object->OBJECT_COMM_WORLD[0], &Object->rank);

  Object->Xcut = M[0];
  Object->Ycut = M[1];
  Object->Zcut = M[2];

  for (MFTL_INT i = 0; i < 3; i++)
  {
    Object->globalxmin[i] = xmin[i];
    Object->globalxmax[i] = xmax[i];
  }

  if (Object->Xcut * Object->Ycut * Object->Zcut != Object->nproc)
  {
    char error[] = "Mesh parallelisation does not fit number of processors.";
    MFTLThrowError(1, error);
  }

  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[0]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[1]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[2]);

  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[0]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[1]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[2]);

  for (MFTL_INT i = 0; i < 3; i++)
  {
    dx[i] = (xmax[i] - xmin[i]) / ((double) M[i]);
  }

  for (MFTL_INT n = 0; n < Object->nproc; n++)
  {
    ix[2] = n / (M[0] * M[1]);
    ix[1] = (n - ix[2] * M[0] * M[1]) / M[0];
    ix[0] = n - ix[2] * M[0] * M[1] - ix[1] * M[0];

    for (MFTL_INT i = 0; i < 3; i++)
    {
      Object->ownershiprangemin[i][n] = xmin[i] + dx[i] * ((double) ix[i]);
      Object->ownershiprangemax[i][n] = xmin[i] + dx[i] * ((double) ix[i] + 1);
    }
  }

  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_InitializeObjectWithComm(MFTL_Object *Object, int M[3], double xmin[3], double xmax[3], MPI_Comm ObjectComm)
{
  MFTL_INT ix[3];
  MFTL_DOUBLE dx[3];
  int result;

  MPI_Comm_compare(MPI_COMM_WORLD, ObjectComm, &result);

  if (sizeof(MFTL_DOUBLE) == sizeof(double))
    MFTL_MPI_DOUBLE = MPI_DOUBLE;
  else
  {
    char error[] = "Accuracy of MPI_DOUBLE and MFTL_MPI_DOUBLE could not be determined. Needs a fix.";
    MFTLThrowError(1, error);
  }
  if (sizeof(MFTL_INT) == sizeof(int))
    MFTL_MPI_INT = MPI_INT;
  else if (sizeof(MFTL_INT) == sizeof(long))
    MFTL_MPI_INT = MPI_LONG;
  else
  {
    char error[] = "Accuracy of MPI_INT and MFTL_MPI_INT could not be determined. Needs a fix.";
    MFTLThrowError(1, error);
  }
  // Set the size of the object
  Object->Size.MFTL_arrayindex[0] = 0;
  Object->Size.MFTL_arrayindex[1] = sizeof(MFTL_Triangle);
  Object->Size.MFTL_arrayindex[2] = sizeof(MFTL_Triangle) + 3 * sizeof(MFTL_Edge);
  Object->Size.MFTL_sizeofobject = sizeof(MFTL_Triangle) + 3 * sizeof(MFTL_Edge) + 3 * sizeof(MFTL_Vertex);
  Object->SizeOfData = 0;

  // Initialise the inner lists to null
  Object->Edges = MFTL_NULL;
  Object->Triangles = MFTL_NULL;
  Object->Vertexes = MFTL_NULL;

  if (result == MPI_IDENT)
  {
    MFTL_AllocateMemory(sizeof(MPI_Comm), (void **) &Object->OBJECT_COMM_WORLD);
    Object->OBJECT_COMM_WORLD[0] = MPI_COMM_WORLD;
  }
  else
  {
    MFTL_AllocateMemory(sizeof(MPI_Comm), (void **) &Object->OBJECT_COMM_WORLD);
    MPI_Comm_dup(ObjectComm, Object->OBJECT_COMM_WORLD);
  }
  MPI_Comm_size(Object->OBJECT_COMM_WORLD[0], &Object->nproc);
  MPI_Comm_rank(Object->OBJECT_COMM_WORLD[0], &Object->rank);

  Object->Xcut = M[0];
  Object->Ycut = M[1];
  Object->Zcut = M[2];

  for (MFTL_INT i = 0; i < 3; i++)
  {
    Object->globalxmin[i] = xmin[i];
    Object->globalxmax[i] = xmax[i];
  }

  if (Object->Xcut * Object->Ycut * Object->Zcut != Object->nproc)
  {
    char error[] = "Mesh parallelisation does not fit number of processors.";
    MFTLThrowError(1, error);
  }

  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[0]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[1]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemin[2]);

  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[0]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[1]);
  MFTL_AllocateMemory(Object->nproc * sizeof(double), (void **) &Object->ownershiprangemax[2]);

  for (MFTL_INT i = 0; i < 3; i++)
  {
    dx[i] = (xmax[i] - xmin[i]) / ((double) M[i]);
  }

  for (MFTL_INT n = 0; n < Object->nproc; n++)
  {
    ix[2] = n / (M[0] * M[1]);
    ix[1] = (n - ix[2] * M[0] * M[1]) / M[0];
    ix[0] = n - ix[2] * M[0] * M[1] - ix[1] * M[0];

    for (MFTL_INT i = 0; i < 3; i++)
    {
      Object->ownershiprangemin[i][n] = xmin[i] + dx[i] * ((double) ix[i]);
      Object->ownershiprangemax[i][n] = xmin[i] + dx[i] * ((double) ix[i] + 1);
    }
  }

  return (MFTL_SUCCESS);
}
