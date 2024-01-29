/***************************************************
 * scalesurface.c
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"

MFTL_ErrorCode MFTL_ScaleSurface(MFTL_Object *Object, MFTL_DOUBLE scale[])
{
  MFTL_Vertex *Vertex;

  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    Vertex->x[0] *= scale[0];
    Vertex->x[1] *= scale[1];
    Vertex->x[2] *= scale[2];
  }

  return (MFTL_SUCCESS);
}
