/***************************************************
 * redistributeobject.c
 * Created on Wed Jan 31 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"
#include "mftl-functions.h"

/* This routine makes a parallel distribution based upon the parallel cut */
MFTL_ErrorCode MFTL_RedistributeObject(MFTL_Object *Object, MFTL_BOOL LayerOfGhostTriangles)
{
  MFTL_Vertex *Vertex;
  MFTL_Triangle *Triangle, *NextTriangle;
  MFTL_Edge *Edge;
  MFTL_INT LocalProcessor;
  MFTL_INT *NumToSend;
  char **TriangleListToSendToProc;
  MFTL_INT toproc;
  MFTL_INT hadprocs[3];
  MFTL_INT q, t; /* Processor source and target */
  MPI_Status status;
  MFTL_BOOL RemovedElements = MFTL_NO;

  NumToSend = (MFTL_INT *) malloc(Object->nproc * sizeof(MFTL_INT));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));

  MFTL_IntArraySetValue(NumToSend, Object->nproc, 0);

  /* First, loop over vertices and find out if vertex needs to be moved. */
  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    if ((LocalProcessor = MFTL_ProcOfVertex(Object, Vertex)) < 0)
    {
      char error[MFTL_STRINGLENGTH];
      snprintf(error, MFTL_STRINGLENGTH,
               "[rank = %d]MFTLRedistributeObject::FATAL ERROR: Processor out "
               "of range; vertex has left BB.\n",
               Object->rank);
      MFTL_VertexShowSummary(Vertex, stdout, error);
      MFTL_ThrowError(1, error);
    }

    if (LocalProcessor != Vertex->processor || (LocalProcessor == Vertex->processor && LocalProcessor != Object->rank))
    /* This vertex is not on a correct processor or it is a ghost vertex on current processor  */
    {
      Vertex->move = MFTL_YES;
      Vertex->processor = LocalProcessor;
    }
    else if (LocalProcessor == Object->rank && Object->rank != Vertex->processor)  // Vertex on the current proc
        // but labelled wrongly; just update the processor label
    {
      Vertex->move = MFTL_NO;
      Vertex->processor = LocalProcessor;
    }
    else
    {
      Vertex->move = MFTL_NO;
    }
  }

  /* Loop over all the edges to set appropriate processor based on vertex[0] */
  for (Edge = Object->Edges; Edge != NULL; Edge = Edge->hh.next)
  {
    Edge->processor = Edge->Vertex[0]->processor;
  }

  /* Loop over all triangles. If they contain a vertex that will be moved, the complete triangle will be moved */
  Triangle = Object->Triangles;
  while (Triangle != MFTL_NULL)
  {
    NextTriangle = Triangle->hh.next;

    if (Triangle->mainprocessor != Object->rank)  // remove ghost triangles
    {
      MFTL_RemoveTriangle(Object, Triangle);
      RemovedElements = MFTL_YES;
    }
    else
    {
      Triangle->mainprocessor = Triangle->Vertex[0]->processor;
      Triangle->centerproc = MFTL_ProcOfTriangleCenter(Object, Triangle);
      if (Triangle->Vertex[0]->move == MFTL_YES || Triangle->Vertex[1]->move == MFTL_YES || Triangle->Vertex[2]->move == MFTL_YES)
      {
        hadprocs[0] = -1;
        hadprocs[1] = -1;
        hadprocs[2] = -1;
        if (Triangle->Vertex[0]->move == MFTL_YES)
        {
          toproc = Triangle->Vertex[0]->processor;
          hadprocs[0] = toproc;
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;
        }
        if (Triangle->Vertex[1]->move == MFTL_YES && hadprocs[0] != Triangle->Vertex[1]->processor)
        {
          toproc = Triangle->Vertex[1]->processor;
          hadprocs[1] = toproc;
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;
        }
        if (Triangle->Vertex[2]->move == MFTL_YES && hadprocs[0] != Triangle->Vertex[2]->processor && hadprocs[1] != Triangle->Vertex[2]->processor)
        {
          toproc = Triangle->Vertex[2]->processor;
          hadprocs[2] = toproc;
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;
        }
      }

      /* If all 3 vertexes have to be moved, the complete triangle may be
       * deleted from this processor. */
      if (Triangle->Vertex[0]->move == MFTL_YES && Triangle->Vertex[1]->move == MFTL_YES && Triangle->Vertex[2]->move == MFTL_YES)
      {
        MFTL_RemoveTriangle(Object, Triangle);
        RemovedElements = MFTL_YES;
      }
    }
    Triangle = NextTriangle;
  } /* End of looping over all triangles */

  /* Send and receive triangle lists to all processors */
  for (q = 0; q < Object->nproc; q++) /* q is the sending processor */
  {
    if (q == Object->rank) /* The q processor sends to the t processor */
    {
      for (t = 0; t < Object->nproc; t++) /* t are the receiving processors */
      {
        if (q != t) /* Do not send to myself */
        {
          /* Send */
          MPI_Send(&NumToSend[t], 1, MFTL_MPI_INT, t, t, Object->OBJECT_COMM_WORLD[0]);
          if (NumToSend[t] > 0)
          {
            MPI_Send(TriangleListToSendToProc[t], NumToSend[t] * Object->Size.MFTL_sizeofobject, MPI_BYTE, t, t + 1, Object->OBJECT_COMM_WORLD[0]);
            MFTL_FreeMemory(TriangleListToSendToProc[t]);
          }
        }
        else if (NumToSend[t] > 0)
          MFTL_FreeMemory(TriangleListToSendToProc[t]);
      }
    }
    else /* Receive from q */
    {
      MFTL_INT num;
      /* Receive */
      MPI_Recv(&num, 1, MFTL_MPI_INT, q, Object->rank, Object->OBJECT_COMM_WORLD[0], &status);
      if (num > 0)
      {
        char *temporarylist = (char *) malloc(num * Object->Size.MFTL_sizeofobject);

        /* Receive the lists with contents */
        MPI_Recv(temporarylist, num * Object->Size.MFTL_sizeofobject, MPI_BYTE, q, Object->rank + 1, Object->OBJECT_COMM_WORLD[0], &status);

        MFTL_ArrayToTriangles(Object, temporarylist, num);
        MFTL_FreeMemory(temporarylist);
      }
    }
  }
  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  if (RemovedElements == MFTL_YES)
  {
    MFTL_RemoveOrphanVertexes(Object);
    MFTL_RemoveOrphanEdges(Object);
  }

  if (LayerOfGhostTriangles == MFTL_YES)
  {
    MFTL_CreateTriangleGhostPattern(Object);
  }
  else
  {
    MFTL_FindTriangleEdgesandVertices(Object);
  }
  // MFTL_FindTriangleEdgesandVertices(Object);  // more rigorous if this is done also with triangle ghost pattern (is now directly in GhostPattern Function)

  if (RemovedElements == MFTL_YES)
  {
    MFTL_RemoveOrphanVertexes(Object);
    MFTL_RemoveOrphanEdges(Object);
  }

  return (MFTL_SUCCESS);
}

/* This routine makes a parallel distribution based upon the parallel cut */
MFTL_ErrorCode MFTL_RedistributeObjectBasedOnTriangleCenter(MFTL_Object *Object, MFTL_BOOL LayerOfGhostTriangles)
{
  MFTL_Triangle *Triangle = MFTL_NULL, *NextTriangle = MFTL_NULL;
  MFTL_Vertex *Vertex = MFTL_NULL;
  MFTL_Edge *Edge = MFTL_NULL;
  MFTL_INT LocalProcessor = -1;
  MFTL_INT *NumToSend = MFTL_NULL;
  char **TriangleListToSendToProc = MFTL_NULL;
  MFTL_INT q, t; /* Processor source and target */
  MPI_Status status;
  MFTL_BOOL RemovedElements = MFTL_NO;

  NumToSend = (MFTL_INT *) malloc(Object->nproc * sizeof(MFTL_INT));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));

  MFTL_IntArraySetValue(NumToSend, Object->nproc, 0);

  /* First, loop over vertices and find out if vertex needs to be moved. */
  for (Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  {
    if ((LocalProcessor = MFTL_ProcOfVertex(Object, Vertex)) < 0)
    {
      char error[MFTL_STRINGLENGTH];
      snprintf(error, MFTL_STRINGLENGTH,
               "[rank = %d]MFTLRedistributeObject::FATAL ERROR: Processor out "
               "of range; vertex has left BB.\n",
               Object->rank);
      MFTL_VertexShowSummary(Vertex, stdout, error);
      MFTL_ThrowError(1, error);
    }

    if (LocalProcessor != Vertex->processor || (LocalProcessor == Vertex->processor && LocalProcessor != Object->rank))
    /* This vertex is not on a correct processor or it is a ghost vertex on
       current processor  */
    {
      Vertex->move = MFTL_YES;
      Vertex->processor = LocalProcessor;
    }
    else if (LocalProcessor == Object->rank && Object->rank != Vertex->processor)  // Vertex on the current proc
        // but labelled wrongly
    {
      Vertex->move = MFTL_NO;
      Vertex->processor = LocalProcessor;
    }
    else
    {
      Vertex->move = MFTL_NO;
    }
  }

  /* Loop over all the edges to set appropriate processor based on vertex[0] */
  for (Edge = Object->Edges; Edge != NULL; Edge = Edge->hh.next)
  {
    Edge->processor = Edge->Vertex[0]->processor;
  }

  /* Loop over all triangles. If they contain a vertex that will be moved, the
   * complete triangle will be moved */
  Triangle = Object->Triangles;
  while (Triangle != MFTL_NULL)
  {
    NextTriangle = Triangle->hh.next;

    if ((LocalProcessor = MFTL_ProcOfTriangleCenter(Object, Triangle)) < 0 || Triangle->mainprocessor != Object->rank)
    {
      MFTL_RemoveTriangle(Object, Triangle);
      RemovedElements = MFTL_YES;
    }
    else if (LocalProcessor != Triangle->mainprocessor)
    {
      MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[LocalProcessor], NumToSend[LocalProcessor], 1);
      MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[LocalProcessor][Object->Size.MFTL_sizeofobject * NumToSend[LocalProcessor]]);
      NumToSend[LocalProcessor]++;
      MFTL_RemoveTriangle(Object, Triangle);
      RemovedElements = MFTL_YES;
    }
    Triangle = NextTriangle;
  } /* End of looping over all triangles */

  /* Send and receive triangle lists to all processors */
  for (q = 0; q < Object->nproc; q++) /* q is the sending processor */
  {
    if (q == Object->rank) /* The q processor sends to the t processor */
    {
      for (t = 0; t < Object->nproc; t++) /* t are the receiving processors */
      {
        if (q != t) /* Do not send to myself */
        {
          /* Send */
          MPI_Send(&NumToSend[t], 1, MFTL_MPI_INT, t, t, Object->OBJECT_COMM_WORLD[0]);
          if (NumToSend[t] > 0)
          {
            MPI_Send(TriangleListToSendToProc[t], NumToSend[t] * Object->Size.MFTL_sizeofobject, MPI_BYTE, t, t + 1, Object->OBJECT_COMM_WORLD[0]);
            MFTL_FreeMemory(TriangleListToSendToProc[t]);
          }
        }
        else if (NumToSend[t] > 0)
          MFTL_FreeMemory(TriangleListToSendToProc[t]);
      }
    }
    else /* Receive from q */
    {
      MFTL_INT num;
      /* Receive */
      MPI_Recv(&num, 1, MFTL_MPI_INT, q, Object->rank, Object->OBJECT_COMM_WORLD[0], &status);
      if (num > 0)
      {
        char *temporarylist = (char *) malloc(num * Object->Size.MFTL_sizeofobject);

        /* Receive the lists with contents */
        MPI_Recv(temporarylist, num * Object->Size.MFTL_sizeofobject, MPI_BYTE, q, Object->rank + 1, Object->OBJECT_COMM_WORLD[0], &status);

        MFTL_ArrayToTriangles(Object, temporarylist, num);
        MFTL_FreeMemory(temporarylist);
      }
    }
  }
  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    Triangle->mainprocessor = MFTL_ProcOfTriangleCenter(Object, Triangle);
  }

  if (RemovedElements == MFTL_YES)
  {
    MFTL_RemoveOrphanVertexes(Object);
    MFTL_RemoveOrphanEdges(Object);
  }

  if (LayerOfGhostTriangles == MFTL_YES)
  {
    MFTL_CreateTriangleGhostPattern(Object);
    for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
    {
      Triangle->mainprocessor = MFTL_ProcOfTriangleCenter(Object, Triangle);
    }
  }
  else
  {
    MFTL_FindTriangleEdgesandVertices(Object);
  }

  if (RemovedElements == MFTL_YES)
  {
    MFTL_RemoveOrphanVertexes(Object);
    MFTL_RemoveOrphanEdges(Object);
  }

  return (MFTL_SUCCESS);
}

/* Function to send triangles as ghosts from current proc to the proc owning
 * their centres
  MFTL_YES is for forth and MFTL_NO is for back sending
  */
MFTL_ErrorCode MFTL_SendTrianglesToProcWCenter(MFTL_Object *Object, MFTL_BOOL ForthOrBack)
{
  MFTL_Triangle *Triangle;
  MFTL_INT toproc;
  MFTL_INT *NumToSend;
  char **TriangleListToSendToProc;
  MFTL_INT t, q;
  MPI_Status status;

  NumToSend = (MFTL_INT *) malloc(Object->nproc * sizeof(int));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));

  MFTL_IntArraySetValue(NumToSend, Object->nproc, 0);

  // Loop over triangles and populate the send lists
  for (Triangle = Object->Triangles; Triangle != NULL; Triangle = Triangle->hh.next)
  {
    // If triangle centre is on different proc, send a copy there
    Triangle->centerproc = MFTL_ProcOfTriangleCenter(Object, Triangle);
    if (ForthOrBack == MFTL_YES)
    {
      if ((Triangle->mainprocessor != Triangle->centerproc) && (Object->rank == Triangle->mainprocessor))
      {
        toproc = Triangle->centerproc;

        MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
        MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
        NumToSend[toproc]++;
        /* You also need to send all its 3 neighbours */
        for (MFTL_INT i = 0; i < 3; i++)
        {
          MFTL_Triangle *Ttmp = MFTL_EdgeComplementaryTriangle(Object, Triangle->Edge[i], Triangle->key);
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Ttmp, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;
        }
      }
    }
    else
    {
      if ((Triangle->mainprocessor != Triangle->centerproc) && (Object->rank == Triangle->centerproc))
      {
        toproc = Triangle->mainprocessor;
        MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
        MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
        NumToSend[toproc]++;
      }
    }
  }

  /* Send and receive triangle lists to all processors */
  for (q = 0; q < Object->nproc; q++) /* q is the sending processor */
  {
    if (q == Object->rank) /* The q processor sends to the t processor */
    {
      for (t = 0; t < Object->nproc; t++) /* t are the receiving processors */
      {
        if (q != t) /* Do not send to myself */
        {
          /* Send */
          MPI_Send(&NumToSend[t], 1, MFTL_MPI_INT, t, t, Object->OBJECT_COMM_WORLD[0]);
          if (NumToSend[t] > 0)
          {
            MPI_Send(TriangleListToSendToProc[t], NumToSend[t] * Object->Size.MFTL_sizeofobject, MPI_BYTE, t, t + 1, Object->OBJECT_COMM_WORLD[0]);
            MFTL_FreeMemory(TriangleListToSendToProc[t]);
          }
        }
        else if (NumToSend[t] > 0)
          MFTL_FreeMemory(TriangleListToSendToProc[t]);
      }
    }
    else /* Receive from q */
    {
      MFTL_INT num;
      /* Receive */
      MPI_Recv(&num, 1, MFTL_MPI_INT, q, Object->rank, Object->OBJECT_COMM_WORLD[0], &status);
      if (num > 0)
      {
        char *temporarylist = (char *) malloc(num * Object->Size.MFTL_sizeofobject);

        /* Receive the lists with contents */
        MPI_Recv(temporarylist, num * Object->Size.MFTL_sizeofobject, MPI_BYTE, q, Object->rank + 1, Object->OBJECT_COMM_WORLD[0], &status);

        MFTL_ArrayToTriangles(Object, temporarylist, num);
        MFTL_FreeMemory(temporarylist);
      }
    }
  }
  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  return (MFTL_SUCCESS);
}

MFTL_INT MFTL_ProcOfTriangleCenter(MFTL_Object *Object, MFTL_Triangle *T)
{
  MFTL_INT i;

  MFTL_TriangleCentre(T);

  for (i = 0; i < Object->nproc; i++)
  {
    if (T->center[0] >= Object->ownershiprangemin[0][i] && T->center[0] < Object->ownershiprangemax[0][i] && T->center[1] >= Object->ownershiprangemin[1][i] &&
        T->center[1] < Object->ownershiprangemax[1][i] && T->center[2] >= Object->ownershiprangemin[2][i] && T->center[2] < Object->ownershiprangemax[2][i])
    {
      return (i);
    }
  }
  return (-1);
}

MFTL_INT MFTL_ProcOfVertex(MFTL_Object *Object, MFTL_Vertex *V)
{
  MFTL_INT i;

  for (i = 0; i < Object->nproc; i++)
  {
    if (V->x[0] >= Object->ownershiprangemin[0][i] && V->x[0] < Object->ownershiprangemax[0][i] && V->x[1] >= Object->ownershiprangemin[1][i] &&
        V->x[1] < Object->ownershiprangemax[1][i] && V->x[2] >= Object->ownershiprangemin[2][i] && V->x[2] < Object->ownershiprangemax[2][i])
    {
      return (i);
    }
  }
  return (-1);
}