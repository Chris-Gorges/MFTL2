/***************************************************
 * showsummary.c
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"

/*=====================================================================================
 *
 *        Filename:  showsummary.c
 *
 *     Description:  This routine shows a summary of an object divided over all processors.
 *
 *          Author:  Berend van Wachem (BvW), berend.van.wachem@gmail.com
 *       Copyright:  (c) Berend van Wachem.
 *
 *
 * =====================================================================================*/
#include <stdio.h>
#include <math.h>

#include "mftl.h"

MFTL_ErrorCode MFTL_ShowSummary(MFTL_Object *Object)
{
  MFTL_Triangle *Triangle;
  MFTL_Edge *Edge;
  MFTL_Vertex *Vertex;

  MFTL_INT numoftriangles = 0, numofghosttriangles = 0;
  MFTL_INT numofedges = 0, numofghostedges = 0;
  MFTL_INT numofvertexes = 0, numofghostvertexes = 0;

  for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    numoftriangles++;
    if (Triangle->mainprocessor != Object->rank) numofghosttriangles++;
  }

  for (Edge = Object->Edges; Edge != MFTL_NULL; Edge = Edge->hh.next)
  {
    numofedges++;
    if (Edge->processor != Object->rank) numofghostedges++;
  }

  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    numofvertexes++;
    if (Vertex->processor != Object->rank) numofghostvertexes++;
  }

  if (Object->rank == 0) printf("-----------------------------\n");
  MPI_Barrier(Object->OBJECT_COMM_WORLD[0]);

  printf("Processor %i has %i triangles (of which %i ghost), %i edges (of which %i ghost) and %i vertexes (of which %i ghost)\n", (int) Object->rank,
         (int) numoftriangles, (int) numofghosttriangles, (int) numofedges, (int) numofghostedges, (int) numofvertexes, (int) numofghostvertexes);

  MPI_Barrier(Object->OBJECT_COMM_WORLD[0]);
  if (Object->rank == 0) printf("-----------------------------\n");

  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_VertexShowSummary(MFTL_Vertex *V, FILE *fp, const char *msg)
{
  if (V == MFTL_NULL)
  {
    fprintf(fp, "%s::: VERTEX IS NULL\n", msg);
    fflush(fp);
    return;
  }
  fprintf(fp, "%s:", msg);
  if (V->move == MFTL_YES)
  {
    fprintf(fp, "V.id[%d] V.proc[%d] V.move = MFTL_YES V.x [%6.6e %6.6e %6.6e]\n", (int) V->key, (int) V->processor, (double) V->x[0], (double) V->x[1],
            (double) V->x[2]);
  }
  else
  {
    fprintf(fp, "V.id[%d] V.proc[%d] V.move = MFTL_NO V.x [%6.6e %6.6e %6.6e]\n", (int) V->key, (int) V->processor, (double) V->x[0], (double) V->x[1],
            (double) V->x[2]);
  }
  fflush(fp);
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_EdgeShowSummary(MFTL_Edge *E, FILE *fp, const char *msg)
{
  if (E == MFTL_NULL)
  {
    fprintf(fp, "%s::: EDGE IS NULL\n", msg);
    fflush(fp);
    return;
  }
  MFTL_DOUBLE elength = MFTL_EdgeLength(E);
  fprintf(fp, "%s:", msg);
  fflush(fp);
  fprintf(fp, "E.id[%5d] E.T[%5d %5d] E.V[%5d:%d  %5d:%5d] E.length = %e\n", (int) E->key, (int) E->TriangleID[0], (int) E->TriangleID[1], (int) E->VertexID[0],
          (int) E->Vertex[0]->processor, (int) E->VertexID[1], (int) E->Vertex[1]->processor, (double) elength);
  fflush(fp);
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_TriangleShowSummary(MFTL_Triangle *T, FILE *fp, const char *msg)
{
  if (T == MFTL_NULL)
  {
    fprintf(fp, "%s::: TRIANGLE IS NULL\n", msg);
    fflush(fp);
    return (MFTL_ERROR);
  }

  fprintf(fp, "%s:", msg);
  fflush(fp);
  fprintf(fp, "T.id = %d, T.mainproc = %d\nT.E = [%d %d %d]   T.V = [%d %d %d]   T.V.proc = [%d %d %d]\n", (int) T->key, (int) T->mainprocessor,
          (int) T->Edge[0]->key, (int) T->Edge[1]->key, (int) T->Edge[2]->key, (int) T->Vertex[0]->key, (int) T->Vertex[1]->key, (int) T->Vertex[2]->key,
          (int) T->Vertex[0]->processor, (int) T->Vertex[1]->processor, (int) T->Vertex[2]->processor);
  fflush(fp);

  MFTLEdgeShowSummary(T->Edge[0], fp, "E0:");
  fflush(fp);
  MFTLEdgeShowSummary(T->Edge[1], fp, "E1:");
  fflush(fp);
  MFTLEdgeShowSummary(T->Edge[2], fp, "E2:");
  fflush(fp);
  fprintf(fp, "\n");
  fflush(fp);
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_IntArrayShowSummary(MFTL_INT *array, MFTL_INT n, FILE *fp, const char *msg)
{
  fprintf(fp, "%s:", msg);
  for (MFTL_INT i = 0; i < n; i++)
  {
    fprintf(fp, "%d\t", (int) array[i]);
    if ((i + 1) % 5 == 0) fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
  fflush(fp);
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_DoubleArrayShowSummary(MFTL_DOUBLE *array, MFTL_INT n, FILE *fp, const char *msg)
{
  fprintf(fp, "%s:", msg);
  for (i = 0; i < n; i++)
  {
    fprintf(fp, "%12.10G\t", (double) array[i]);
    if ((i + 1) % 6 == 0) fprintf(fp, "\n");
  }
  fprintf(fp, "\n");
  fflush(fp);
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_ObjectDomainShowSummary(MFTL_Object *Object, FILE *fp, const char *msg)
{
  fprintf(fp, "%s", msg);
  for (MFTL_INT i = 0; i < Object->nproc; i++)
  {
    fprintf(fp, "[%lf %lf %lf] - [%lf %lf %lf]\n", (int) Object->ownershiprangemin[0][i], (int) Object->ownershiprangemin[1][i],
            (int) Object->ownershiprangemin[2][i], (int) Object->ownershiprangemax[0][i], (int) Object->ownershiprangemax[1][i],
            (int) Object->ownershiprangemax[2][i]);
  }
  return (MFTL_SUCCESS);
}

MFTL_ErrorCode MFTL_StatShowSummary(MFTL_Stats *s, FILE *fp, const char *msg)
{
  fprintf(fp, "%s", msg);
  fprintf(fp, "min:    %12.10e\nmax:    %12.10e\nmean:   %12.10e\nstddev: %12.10e\n", (double) s->min, (double) s->max, (double) s->mean, (double) s->stddev);
  return (MFTL_SUCCESS);
}
