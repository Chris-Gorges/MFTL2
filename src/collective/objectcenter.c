/***************************************************
 * objectcenter.c
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"
#include "mftl-functions.h"

MFTL_ErrorCode MFTL_CenterOfAreaFace(MFTL_Triangle *Triangle, MFTL_DOUBLE data[])
{
  MFTL_Vertex *v1, *v2, *v3;
  MFTL_DOUBLE Area;

  Area = MFTL_TriangleArea(Triangle);

  v1 = Triangle->Vertex[0];
  v2 = Triangle->Vertex[1];
  v3 = Triangle->Vertex[2];

  data[0] += Area * (v1->x[0] + v2->x[0] + v3->x[0]);
  data[1] += Area * (v1->x[1] + v2->x[1] + v3->x[1]);
  data[2] += Area * (v1->x[2] + v2->x[2] + v3->x[2]);

  data[3] += Area;

  return (MFTL_SUCCESS);
}

/***************************************************************************************/

MFTL_DOUBLE MFTL_SurfaceCenter(MFTL_Object *Object, MFTL_DOUBLE cm[])
{
  MFTL_Triangle *Triangle;
  MFTL_DOUBLE data[4], area = 0;
  int ierr;

  cm[0] = cm[1] = cm[2] = 0.0;
  for (MFTL_INT k = 0; k < 4; k++) data[k] = 0.0;

  for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    if (Triangle->mainprocessor == Object->rank) CenterOfAreaFace(Triangle, data);
  }

  ierr = MPI_Allreduce(data, cm, 3, MFTL_MPI_DOUBLE, MPI_SUM, Object->OBJECT_COMM_WORLD[0]);
  ierr = MPI_Allreduce(&data[3], &area, 1, MFTL_MPI_DOUBLE, MPI_SUM, Object->OBJECT_COMM_WORLD[0]);

  if (ierr != 0) printf("%d\n", ierr);

  if (area != 0.)
  {
    cm[0] /= 3. * area;
    cm[1] /= 3. * area;
    cm[2] /= 3. * area;
  }

  return (area);  // returns the volume of the body
}
