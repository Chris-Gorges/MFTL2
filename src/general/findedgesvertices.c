/***************************************************
 * findedgesvertices.c
 * Created on Wed Jan 31 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/
#include "mftl.h"
#include "mftl-functions.h"

MFTL_ErrorCode MFTL_FindTriangleEdgesandVertices(MFTL_Object *Object)
{
  MFTL_Triangle *TriangleItem;
  MFTL_Edge *EdgeItem;
  MFTL_INT n;

  for (EdgeItem = Object->Edges; EdgeItem != MFTL_NULL; EdgeItem = EdgeItem->hh.next)
  {
    EdgeItem->Triangle[0] = MFTL_NULL;
    EdgeItem->Triangle[1] = MFTL_NULL;
    EdgeItem->TriangleID[0] = MFTL_ERROR;
    EdgeItem->TriangleID[1] = MFTL_ERROR;
  }

  /* Loop over all triangles in the object */
  for (TriangleItem = Object->Triangles; TriangleItem != MFTL_NULL; TriangleItem = TriangleItem->hh.next)
  {
    /* Link edges to the triangle they have */
    for (n = 0; n < 3; n++)
    {
      if (TriangleItem->Edge[n]->Triangle[0] == MFTL_NULL)
      {
        TriangleItem->Edge[n]->Triangle[0] = TriangleItem;
        TriangleItem->Edge[n]->TriangleID[0] = TriangleItem->id;
      }
      else
      {
        TriangleItem->Edge[n]->Triangle[1] = TriangleItem;
        TriangleItem->Edge[n]->TriangleID[1] = TriangleItem->id;
      }
    }

    /* Find the 3 vertexes of each triangle */
    TriangleItem->Vertex[0] = TriangleItem->Edge[0]->Vertex[0];
    TriangleItem->mainprocessor = TriangleItem->Vertex[0]->processor;
    if (TriangleItem->Vertex[0] == TriangleItem->Edge[1]->Vertex[0])
    {
      TriangleItem->Vertex[1] = TriangleItem->Edge[1]->Vertex[1];
      TriangleItem->Vertex[2] = TriangleItem->Edge[0]->Vertex[1];
    }
    else if (TriangleItem->Edge[0]->Vertex[1] == TriangleItem->Edge[1]->Vertex[1])
    {
      TriangleItem->Vertex[1] = TriangleItem->Edge[0]->Vertex[1];
      TriangleItem->Vertex[2] = TriangleItem->Edge[1]->Vertex[0];
    }
    else if (TriangleItem->Vertex[0] == TriangleItem->Edge[1]->Vertex[1])
    {
      TriangleItem->Vertex[1] = TriangleItem->Edge[1]->Vertex[0];
      TriangleItem->Vertex[2] = TriangleItem->Edge[0]->Vertex[1];
    }
    else
    {
      TriangleItem->Vertex[1] = TriangleItem->Edge[0]->Vertex[1];
      TriangleItem->Vertex[2] = TriangleItem->Edge[1]->Vertex[1];
    }
  }

  return (MFTL_SUCCESS);
}
