/*=====================================================================================
 *
 *        Filename:  sphere.c
 *
 *     Description:  Generate a unit isocahedron and refine it k times, to obtain unit sphere
 *
 *          Author:  Berend van Wachem (BvW), berend.van.wachem@gmail.com
 *       Copyright:  (c) Berend van Wachem.
 *
 * =====================================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mftl.h"
#include "mftl-functions.h"

#define MFTL_M_ICOSAHEDRON_X /* sqrt(sqrt(5)+1)/sqrt(2*sqrt(5)) */ 0.850650808352039932181540497063011072240401406
#define MFTL_M_ICOSAHEDRON_Y /* sqrt(2)/sqrt(5+sqrt(5))         */ 0.525731112119133606025669084847876607285497935
#define MFTL_M_ICOSAHEDRON_Z 0.0

MFTL_ErrorCode MFTL_GenerateSphere(MFTL_Object *Object, MFTL_INT refine)
{
  // First generate a unit isocahedron
  MFTL_GenerateIsocahedron(Object);
  MFTL_FindTriangleEdgesandVertices(Object);

  // Refine the sphere "k" times, each refinement increases number of triangles 4 times
  for (MFTL_INT i = 0; i < refine; i++)
  {
    MFTL_TessalateSphere(Object);
    MFTL_FindTriangleEdgesandVertices(Object);
  }

  return (MFTL_SUCCESS);
}
/**********************************************************************/
MFTL_ErrorCode MFTL_GenerateIsocahedron(MFTL_Object *Object)
{
  MFTL_Vertex *NewVertex;
  MFTL_Edge *NewEdge;
  MFTL_Triangle *NewTriangle;

  //define the coordinates of the isocahedron
  MFTL_DOUBLE coordinates[12][3] = {
      {MFTL_M_ICOSAHEDRON_Z, MFTL_M_ICOSAHEDRON_X, -MFTL_M_ICOSAHEDRON_Y},  {MFTL_M_ICOSAHEDRON_X, MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z},
      {MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z, -MFTL_M_ICOSAHEDRON_X},  {MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z, MFTL_M_ICOSAHEDRON_X},
      {MFTL_M_ICOSAHEDRON_X, -MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z},  {MFTL_M_ICOSAHEDRON_Z, MFTL_M_ICOSAHEDRON_X, MFTL_M_ICOSAHEDRON_Y},
      {-MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z, MFTL_M_ICOSAHEDRON_X},  {MFTL_M_ICOSAHEDRON_Z, -MFTL_M_ICOSAHEDRON_X, -MFTL_M_ICOSAHEDRON_Y},
      {-MFTL_M_ICOSAHEDRON_X, MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z},  {-MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z, -MFTL_M_ICOSAHEDRON_X},
      {-MFTL_M_ICOSAHEDRON_X, -MFTL_M_ICOSAHEDRON_Y, MFTL_M_ICOSAHEDRON_Z}, {MFTL_M_ICOSAHEDRON_Z, -MFTL_M_ICOSAHEDRON_X, MFTL_M_ICOSAHEDRON_Y}};

  MFTL_INT vertnum[30][2] = {{0, 1}, {2, 1}, {2, 0}, {3, 4},  {1, 4},  {3, 1}, {5, 6},   {3, 6},  {5, 3},  {7, 2},  {2, 4},  {7, 4},  {5, 8}, {6, 8},  {7, 9},
                             {2, 9}, {5, 0}, {0, 8}, {7, 10}, {9, 10}, {5, 1}, {11, 10}, {11, 7}, {11, 6}, {6, 10}, {11, 3}, {8, 10}, {9, 8}, {11, 4}, {0, 9}};

  MFTL_INT edgenum[20][3] = {{0, 1, 2},    {3, 4, 5},    {6, 7, 8},    {9, 10, 11}, {12, 13, 6},  {14, 15, 9}, {16, 17, 12},
                             {18, 19, 14}, {20, 0, 16},  {21, 18, 22}, {8, 5, 20},  {23, 24, 21}, {25, 7, 23}, {19, 26, 27},
                             {28, 3, 25},  {13, 26, 24}, {22, 11, 28}, {1, 4, 10},  {29, 27, 17}, {2, 15, 29}};

  // Create the table with vertices
  for (MFTL_INT k = 0; k < 12; k++)
  {
    MFTL_AllocateMemory(sizeof(MFTL_Vertex), &NewVertex);
    NewVertex->key = k;
    NewVertex->processor = Object->rank;

    for (MFTL_INT i = 0; i < 3; i++) NewVertex->x[i] = coordinates[k][i];

    HASH_ADD(hh, Object->Vertexes, key, sizeof(ObjectIDType), NewVertex);
  }

  // Create the table with edges
  for (MFTL_INT k = 0; k < 30; k++)
  {
    MFTL_AllocateMemory(sizeof(MFTL_Edge), &NewEdge);

    for (MFTL_INT i = 0; i < 2; i++)
    {
      HASH_FIND(hh, Object->Vertexes, &vertnum[k][i], sizeof(ObjectIDType), NewEdge->Vertex[i]);
      NewEdge->VertexID[i] = NewEdge->Vertex[i]->key;
    }

    NewEdge->key = k;
    NewEdge->processor = Object->rank;

    HASH_ADD(hh, Object->Edges, key, sizeof(ObjectIDType), NewEdge);
  }

  //create the table with triangles
  for (MFTL_INT k = 0; k < 20; k++)
  {
    MFTL_AllocateMemory(sizeof(MFTL_Triangle, &NewTriangle));
    for (MFTL_INT i = 0; i < 3; i++)
    {
      HASH_FIND(hh, Object->Edges, &edgenum[k][i], sizeof(ObjectIDType), NewTriangle->Edge[i]);
    }

    NewTriangle->key = k;
    NewTriangle->mainprocessor = Object->rank;

    HASH_ADD(hh, Object->Triangles, key, sizeof(ObjectIDType), NewTriangle);
  }

  return (MFTL_SUCCESS);
}

/**********************************************************************/
MFTL_ErrorCode MFTL_TessalateEdge_Sphere(MFTL_Vertex *Vertex, MFTL_Edge *Edge)
{
  // Finds the coordinates of the new vertex of a unit sphere
  MFTL_DOUBLE x[3];

  x[0] = 0.5 * (Edge->Vertex[0]->x[0] + Edge->Vertex[1]->x[0]);
  x[1] = 0.5 * (Edge->Vertex[0]->x[1] + Edge->Vertex[1]->x[1]);
  x[2] = 0.5 * (Edge->Vertex[0]->x[2] + Edge->Vertex[1]->x[2]);

  MFTL_NormalizeVector(x);

  MFTL_CopyVector(Vertex->x, x);

  return (MFTL_SUCCESS);
}

/**********************************************************************/

MFTL_ErrorCode MFTL_TessalateSphere(MFTL_Object *Object)
{
  // MFTL Structures
  MFTL_Triangle *Triangle, *TMPTriangle = NULL, *NewTriangle, *NewTriList = NULL;
  MFTL_Edge *Edge, *TMPEdge, *NewEdgeList = NULL;
  MFTL_Vertex *NewVertex;

  // Id of new vertices
  MFTL_INT nvert[3];

  MFTL_INT ParentTriangle, m;

  // Objects counters
  MFTL_INT newvert = 0, newedge = 0, newtri = 0, OldTri;

  // ID's of the edges and vertices in current triangle
  MFTL_INT TriEdgeID[9], TriEdgeVertID[9][2];
  MFTL_INT d1[3] = {0, 1, 0}, d2[3] = {1, 2, 2};

  // Array to store the address of already generated edges
  MFTL_INT *EdgeId;

  // Which edges are used for the current face
  MFTL_INT TriEdge[4][3];

  // Find the current number of vertices & triangles
  newvert = HASH_COUNT(Object->Vertexes);
  OldTri = HASH_COUNT(Object->Triangles);

  // Allocate the memory to store the addresses of refined edges
  MFTL_AllocateMemory((OldTri + 1) * 6 * sizeof(MFTL_INT), &EdgeId);

  // Loop over Triangles of the surface
  for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    for (MFTL_INT i = 0; i < 3; i++)
    {
      // Check if the "i" edge has already been refined
      if (Triangle->Edge[i]->processor < 0)  // If yes use the already generated edges
      {
        // Find the triangle where it has been refined
        if (Triangle->Edge[i]->TriangleID[0] == Triangle->key)
          ParentTriangle = Triangle->Edge[i]->TriangleID[1];
        else
          ParentTriangle = Triangle->Edge[i]->TriangleID[0];

        HASH_FIND(hh, Object->Triangles, &(ParentTriangle), sizeof(ObjectIDType), TMPTriangle);
        // Check which edge it was in the parent triangle
        if (Triangle->Edge[i]->key == TMPTriangle->Edge[0]->key)
          m = 0;
        else if (Triangle->Edge[i]->key == TMPTriangle->Edge[1]->key)
          m = 1;
        else if (Triangle->Edge[i]->key == TMPTriangle->Edge[2]->key)
          m = 2;
        else
        {
          char error[] = "Error during tessalation - missaligned edges.";
          MFTLThrowError(1, error);
        }

        // Use the refined edges
        for (MFTL_INT k = 0; k < 2; k++)
        {
          HASH_FIND(hh, NewEdgeList, &EdgeId[6 * ParentTriangle + 2 * m + k], sizeof(ObjectIDType), TMPEdge);
          TriEdgeID[2 * i + k] = TMPEdge->key;
          TriEdgeVertID[2 * i + k][0] = TMPEdge->Vertex[0]->key;
          TriEdgeVertID[2 * i + k][1] = TMPEdge->Vertex[1]->key;
        }

        // Find the id of the new vertex on that edge
        if (TriEdgeVertID[2 * i][0] == TriEdgeVertID[2 * i + 1][0])
          nvert[i] = TriEdgeVertID[2 * i][0];
        else if (TriEdgeVertID[2 * i][0] == TriEdgeVertID[2 * i + 1][1])
          nvert[i] = TriEdgeVertID[2 * i][0];
        else if (TriEdgeVertID[2 * i][1] == TriEdgeVertID[2 * i + 1][0])
          nvert[i] = TriEdgeVertID[2 * i][1];
        else
          nvert[i] = TriEdgeVertID[2 * i][1];
      }
      else  // generate new edges
      {
        // First create a refined vertex
        MFTL_AllocateMemory(sizeof(MFTL_Vertex), &NewVertex);
        NewVertex->key = newvert;
        nvert[i] = newvert;
        NewVertex->processor = Object->rank;

        MFTL_TessalateEdge_Sphere(NewVertex, Triangle->Edge[i]);

        // Add the newly created vertex to the Object->vertex hash table
        HASH_ADD(hh, Object->Vertexes, key, sizeof(ObjectIDType), NewVertex);
        newvert++;

        //Create new Edges for that vertex
        for (MFTL_INT k = 0; k < 2; k++)
        {
          MFTL_AllocateMemory(sizeof(MFTL_Edge), &Edge);
          HASH_FIND(hh, Object->Vertexes, &(Triangle->Edge[i]->Vertex[k]->key), sizeof(ObjectIDType), Edge->Vertex[0]);
          HASH_FIND(hh, Object->Vertexes, &nvert[i], sizeof(ObjectIDType), Edge->Vertex[1]);

          Edge->key = newedge;
          Edge->processor = Object->rank;
          Edge->VertexID[0] = Edge->Vertex[0]->key;
          Edge->VertexID[1] = Edge->Vertex[1]->key;

          TriEdgeID[2 * i + k] = Edge->key;
          TriEdgeVertID[2 * i + k][0] = Edge->Vertex[0]->key;
          TriEdgeVertID[2 * i + k][1] = Edge->Vertex[1]->key;

          // Add the edge to the new edge hash table
          HASH_ADD(hh, NewEdgeList, key, sizeof(ObjectIDType), Edge);
          EdgeId[6 * Triangle->key + 2 * i + k] = Edge->key;

          newedge++;
        }

        // Mark the parent edge as refined
        Triangle->Edge[i]->processor = MFTL_ERROR;
      }
    }

    // Next create 3 edges based on new vertices
    for (MFTL_INT i = 0; i < 3; i++)
    {
      MFTL_AllocateMemory(sizeof(MFTL_Edge), &Edge);
      HASH_FIND(hh, Object->Vertexes, &nvert[d1[i]], sizeof(ObjectIDType), Edge->Vertex[0]);
      HASH_FIND(hh, Object->Vertexes, &nvert[d2[i]], sizeof(ObjectIDType), Edge->Vertex[1]);

      Edge->key = newedge;
      Edge->processor = Object->rank;
      Edge->VertexID[0] = Edge->Vertex[0]->key;
      Edge->VertexID[1] = Edge->Vertex[1]->key;

      TriEdgeID[i + 6] = Edge->key;
      TriEdgeVertID[i + 6][0] = Edge->Vertex[0]->key;
      TriEdgeVertID[i + 6][1] = Edge->Vertex[1]->key;

      // Add the edge to the hash table

      HASH_ADD(hh, NewEdgeList, key, sizeof(ObjectIDType), Edge);
      EdgeId[6 * Triangle->key + i + 6] = Edge->key;

      newedge++;
    }

    // Find the edges for new faces with edge 6
    if (TriEdgeVertID[0][0] == TriEdgeVertID[2][0])
    {
      TriEdge[0][0] = 0;
      TriEdge[0][1] = 2;
      TriEdge[0][2] = 6;
    }
    else if (TriEdgeVertID[0][0] == TriEdgeVertID[3][0])
    {
      TriEdge[0][0] = 0;
      TriEdge[0][1] = 3;
      TriEdge[0][2] = 6;
    }
    else if (TriEdgeVertID[1][0] == TriEdgeVertID[2][0])
    {
      TriEdge[0][0] = 1;
      TriEdge[0][1] = 2;
      TriEdge[0][2] = 6;
    }
    else
    {
      TriEdge[0][0] = 1;
      TriEdge[0][1] = 3;
      TriEdge[0][2] = 6;
    }

    // with edge 7
    if (TriEdgeVertID[3][0] == TriEdgeVertID[5][0])
    {
      TriEdge[1][0] = 3;
      TriEdge[1][1] = 5;
      TriEdge[1][2] = 7;
    }
    else if (TriEdgeVertID[2][0] == TriEdgeVertID[5][0])
    {
      TriEdge[1][0] = 2;
      TriEdge[1][1] = 5;
      TriEdge[1][2] = 7;
    }
    else if (TriEdgeVertID[3][0] == TriEdgeVertID[4][0])
    {
      TriEdge[1][0] = 3;
      TriEdge[1][1] = 4;
      TriEdge[1][2] = 7;
    }
    else
    {
      TriEdge[1][0] = 2;
      TriEdge[1][1] = 4;
      TriEdge[1][2] = 7;
    }
    // With edge 8
    if (TriEdgeVertID[0][0] == TriEdgeVertID[4][0])
    {
      TriEdge[2][0] = 4;
      TriEdge[2][1] = 0;
      TriEdge[2][2] = 8;
    }
    else if (TriEdgeVertID[0][0] == TriEdgeVertID[5][0])
    {
      TriEdge[2][0] = 5;
      TriEdge[2][1] = 0;
      TriEdge[2][2] = 8;
    }
    else if (TriEdgeVertID[1][0] == TriEdgeVertID[4][0])
    {
      TriEdge[2][0] = 4;
      TriEdge[2][1] = 1;
      TriEdge[2][2] = 8;
    }
    else
    {
      TriEdge[2][0] = 5;
      TriEdge[2][1] = 1;
      TriEdge[2][2] = 8;
    }
    // With Edges 6,7,8
    TriEdge[3][0] = 6;
    TriEdge[3][1] = 7;
    TriEdge[3][2] = 8;

    // Create the new triangles, and add them to the NewTlist
    for (MFTL_INT k = 0; k < 4; k++)
    {
      MFTL_AllocateMemory(sizeof(MFTL_Triangle), &NewTriangle);

      for (MFTL_INT i = 0; i < 3; i++)
      {
        HASH_FIND(hh, NewEdgeList, &(TriEdgeID[TriEdge[k][i]]), sizeof(ObjectIDType), NewTriangle->Edge[i]);
      }

      NewTriangle->key = newtri;
      NewTriangle->mainprocessor = Object->rank;

      HASH_ADD(hh, NewTriList, key, sizeof(ObjectIDType), NewTriangle);
      newtri++;
    }
  }

  // Cleanup old edge and triangle lists from the Object
  while (Object->Triangles)
  {
    Triangle = Object->Triangles;
    HASH_DEL(Object->Triangles, Triangle);
    MFTL_FreeMemory(Triangle);
  }

  while (Object->Edges)
  {
    Edge = Object->Edges;
    HASH_DEL(Object->Edges, Edge);
    MFTL_FreeMemory(Edge);
  }

  // Replace the old lists with the newly generated ones
  Object->Triangles = NewTriList;
  Object->Edges = NewEdgeList;

  MFTL_FreeMemory(EdgeId);
  return (MFTL_SUCCESS);
}
