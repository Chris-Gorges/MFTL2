/***************************************************
 * createtriangleghostpattern.c
 * Created on Wed Jan 31 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#include "mftl.h"
#include "mftl-functions.h"

/* Function to send triangles as ghosts from current proc to the proc that
 * potentially owns their neighbour */
MFTL_ErrorCode MFTL_CreateTriangleGhostPattern(MFTL_Object *Object)
{
  MFTL_Triangle *Triangle, *TriangleNeighbour;
  MFTL_INT toproc;
  MFTL_INT NumofTrianglesToSend = 0;
  MFTL_INT *NumTrianglesFromProcs;
  MFTL_INT NumofTotalTrianglesToSend = 0;
  MFTL_INT *displacement;
  MFTL_INT *TriangleIDsfromProcs;
  MFTL_INT *ListToReceive = MFTL_NULL;
  MFTL_INT t, q;
  MFTL_INT i, j;
  MFTL_INT *NumToSend;
  char **TriangleListToSendToProc;
  MFTL_INT TriangleIDNum;
  MPI_Status status;

  NumTrianglesFromProcs = (MFTL_INT *) calloc(Object->nproc, sizeof(MFTL_INT));
  NumToSend = (MFTL_INT *) calloc(Object->nproc, sizeof(MFTL_INT));
  displacement = (MFTL_INT *) calloc(Object->nproc, sizeof(MFTL_INT));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));

  for (Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    if (Triangle->mainprocessor == Object->rank)
    {
      for (i = 0; i < 3; i++)
      {
        for (j = 0; j < 2; j++)
        {
          if (Triangle->Edge[i]->TriangleID[j] != Triangle->key)
          {
            TriangleIDNum = Triangle->Edge[i]->TriangleID[j];
            HASH_FIND(hh, Object->Triangles, &TriangleIDNum, sizeof(ObjectIDType), TriangleNeighbour);
            if (TriangleNeighbour == MFTL_NULL)
            {
              if (!MFTL_InArrayList(Triangle->Edge[i]->TriangleID[j], ListToReceive, NumofTotalTrianglesToSend))
              {
                ListToReceive = MFTL_DynamicallyArrayGrow(ListToReceive, NumofTrianglesToSend, 1, sizeof(MFTL_INT));
                ListToReceive[NumofTrianglesToSend] = Triangle->Edge[i]->TriangleID[j];
                NumofTrianglesToSend++;
              }
            }
          }
        }
      }
    }
  }

  /* Now send the list around */
  MPI_Allgather(&NumofTrianglesToSend, 1, MFTL_MPI_INT, NumTrianglesFromProcs, 1, MFTL_MPI_INT, Object->OBJECT_COMM_WORLD[0]);
  for (i = 0; i < Object->nproc; i++)
  {
    NumofTotalTrianglesToSend += NumTrianglesFromProcs[i];
  }

  TriangleIDsfromProcs = (MFTL_INT *) calloc(NumofTotalTrianglesToSend, sizeof(MFTL_INT));

  displacement[0] = 0;
  for (i = 1; i < Object->nproc; i++)
  {
    displacement[i] = NumTrianglesFromProcs[i - 1] + displacement[i - 1];
  }

  MPI_Allgatherv(ListToReceive, NumofTrianglesToSend, MFTL_MPI_INT, TriangleIDsfromProcs, NumTrianglesFromProcs, displacement, MFTL_MPI_INT,
                 Object->OBJECT_COMM_WORLD[0]);

  if (NumofTrianglesToSend > 0)
  {
    MFTL_FreeMemory(ListToReceive);
  }

  /* Now loop over all the triangle ID's and see if this current processor owns
   * it */
  for (i = 0; i < NumofTotalTrianglesToSend; i++)
  {
    HASH_FIND(hh, Object->Triangles, &TriangleIDsfromProcs[i], sizeof(ObjectIDType), Triangle);
    if (Triangle != MFTL_NULL)
    {
      if (Triangle->mainprocessor == Object->rank) /* We own this triangle! */
      {
        toproc = MFTL_FindIntervalArray(i, displacement, Object->nproc);

        MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
        MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
        NumToSend[toproc]++;
      }
    }
  }
  MFTL_FreeMemory(NumTrianglesFromProcs);
  MFTL_FreeMemory(TriangleIDsfromProcs);
  MFTL_FreeMemory(displacement);

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
            if (NumToSend[t] > 0)
            {
              MFTL_FreeMemory(TriangleListToSendToProc[t]);
            }
          }
        }
        else if (NumToSend[t] > 0)
        {
          MFTL_FreeMemory(TriangleListToSendToProc[t]);
        }
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
        if (num > 0)
        {
          MFTL_FreeMemory(temporarylist);
        }
      }
    }
  }
  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  MFTL_FindTriangleEdgesandVertices(Object);

  return (MFTL_SUCCESS);
}

/* Function to send triangles as ghosts from owning proc to the owner procs of their vertices */
MFTL_ErrorCode MFTL_AssureOneRingTriangleGhostNeighbourhood(MFTL_Object *Object)
{
  char **TriangleListToSendToProc;
  MFTL_INT *NumToSend;
  MPI_Status status;

  NumToSend = (MFTL_INT *) malloc(Object->nproc * sizeof(MFTL_INT));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));
  MFTL_IntArraySetValue(NumToSend, Object->nproc, 0);

  for (MFTL_Triangle *Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    if (Triangle->mainprocessor == Object->rank)
    {
      for (MFTL_INT i = 0; i < 3; i++)
      {
        if (Triangle->Vertex[i]->processor != Object->rank)
        {
          /* Another processor potentially needs this triangle. */
          MFTL_INT toproc = Triangle->Vertex[i]->processor;
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;
        }
      }
    }
  }

  /* Send and receive triangle lists to all processors */
  for (MFTL_INT q = 0; q < Object->nproc; q++) /* q is the sending processor */
  {
    if (q == Object->rank) /* The q processor sends to the t processor */
    {
      for (MFTL_INT t = 0; t < Object->nproc; t++) /* t are the receiving processors */
      {
        if (q != t) /* Do not send to myself */
        {
          if (q != t) /* Do not send to myself */
          {
            /* Send */
            MPI_Send(&NumToSend[t], 1, MFTL_MPI_INT, t, t, Object->OBJECT_COMM_WORLD[0]);
            if (NumToSend[t] > 0)
            {
              MPI_Send(TriangleListToSendToProc[t], NumToSend[t] * Object->Size.MFTL_sizeofobject, MPI_BYTE, t, t + 1, Object->OBJECT_COMM_WORLD[0]);
              if (NumToSend[t] > 0)
              {
                MFTL_FreeMemory(TriangleListToSendToProc[t]);
              }
            }
          }
          else if (NumToSend[t] > 0)
          {
            MFTL_FreeMemory(TriangleListToSendToProc[t]);
          }
        }
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
        if (num > 0)
        {
          MFTL_FreeMemory(temporarylist);
        }
      }
    }
  }

  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  MFTL_FindTriangleEdgesandVertices(Object);

  return (MFTL_SUCCESS);
}

/* Function to send triangles as ghosts from owning proc to the owner procs of their vertices */
MFTL_ErrorCode MFTL_AssureTwoRingTriangleGhostNeighbourhood(MFTL_Object *Object)
{
  char **TriangleListToSendToProc;
  MFTL_INT *NumToSend;
  MPI_Status status;

  NumToSend = (MFTL_INT *) malloc(Object->nproc * sizeof(MFTL_INT));
  TriangleListToSendToProc = (char **) malloc(Object->nproc * sizeof(void *));
  MFTL_IntArraySetValue(NumToSend, Object->nproc, 0);

  for (MFTL_Triangle *Triangle = Object->Triangles; Triangle != MFTL_NULL; Triangle = Triangle->hh.next)
  {
    if (Triangle->mainprocessor == Object->rank)
    {
      for (MFTL_INT i = 0; i < 3; i++)
      {
        if (Triangle->Vertex[i]->processor != Object->rank)
        {
          /* Another processor potentially needs this triangle. */
          MFTL_INT toproc = Triangle->Vertex[i]->processor;
          MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
          MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
          NumToSend[toproc]++;

          /* Send also all available triangles connected to both other vertices */
          /* Expensive!!! Maybe better method possible??? */
          MFTL_INT vn = (i + 1) % 3;
          MFTL_INT vnn = (i + 2) % 3;
          for (MFTL_Triangle *Triangle2 = Object->Triangles; Triangle2 != MFTL_NULL; Triangle2 = Triangle->hh.next)
          {
            if (Triangle2->key == Triangle->key) continue;

            for (MFTL_INT j = 0; j < 3; j++)
            {
              if (Triangle2->Vertex[j]->key == Triangle->Vertex[vn]->key || Triangle2->Vertex[j]->key == Triangle->Vertex[vnn]->key)
              {
                /* Another processor potentially needs this triangle. */
                MFTL_DynamicallyTriangleArrayGrow(Object, &TriangleListToSendToProc[toproc], NumToSend[toproc], 1);
                MFTL_TriangleToArray(Object, Triangle, &TriangleListToSendToProc[toproc][Object->Size.MFTL_sizeofobject * NumToSend[toproc]]);
                NumToSend[toproc]++;
              }
            }
          }
        }
      }
    }
  }

  /* Send and receive triangle lists to all processors */
  for (MFTL_INT q = 0; q < Object->nproc; q++) /* q is the sending processor */
  {
    if (q == Object->rank) /* The q processor sends to the t processor */
    {
      for (MFTL_INT t = 0; t < Object->nproc; t++) /* t are the receiving processors */
      {
        if (q != t) /* Do not send to myself */
        {
          if (q != t) /* Do not send to myself */
          {
            /* Send */
            MPI_Send(&NumToSend[t], 1, MFTL_MPI_INT, t, t, Object->OBJECT_COMM_WORLD[0]);
            if (NumToSend[t] > 0)
            {
              MPI_Send(TriangleListToSendToProc[t], NumToSend[t] * Object->Size.MFTL_sizeofobject, MPI_BYTE, t, t + 1, Object->OBJECT_COMM_WORLD[0]);
              if (NumToSend[t] > 0)
              {
                MFTL_FreeMemory(TriangleListToSendToProc[t]);
              }
            }
          }
          else if (NumToSend[t] > 0)
          {
            MFTL_FreeMemory(TriangleListToSendToProc[t]);
          }
        }
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
        if (num > 0)
        {
          MFTL_FreeMemory(temporarylist);
        }
      }
    }
  }

  MFTL_FreeMemory(TriangleListToSendToProc);
  MFTL_FreeMemory(NumToSend);

  MFTL_FindTriangleEdgesandVertices(Object);

  return (MFTL_SUCCESS);
}
