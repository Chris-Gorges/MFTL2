/***************************************************
 * mftl-functions.h
 * Created on Mon Jan 29 2024
 *
 *
 *
 * (C) Berend van Wachem, Berend.van.Wachem@gmail.com
 ***************************************************/

#ifndef MFTLFUNCTIONS_H_
#define MFTLFUNCTIONS_H_
#include "mftl.h"
#include "mftl-constants.h"

/*****************************************************************************/
/* General memory handling                                                   */
/*****************************************************************************/
MFTL_ErrorCode MFTL_AllocateMemory(MFTL_INT size, void **NewPointer);
MFTL_ErrorCode MFTL_FreeMemory(void *Pointer);

/*****************************************************************************/
/* General INT array handling                                                */
/*****************************************************************************/
MFTL_ErrorCode MFTL_IntArraySetValue(MFTL_INT *A, MFTL_INT size, MFTL_INT value);
MFTL_INT MFTL_FindIntIntervalArray(MFTL_INT num, MFTL_INT *A, MFTL_INT size);
MFTL_BOOL MFTL_InIntArrayList(MFTL_INT num, MFTL_INT *A, MFTL_INT size);

/*****************************************************************************/
/* Initialisation/Finalisation                                               */
/*****************************************************************************/
MFTL_ErrorCode MFTL_InitializeObject(MFTL_Object *Object, MFTL_INT M[3], MFTL_DOUBLE xmin[3], MFTL_DOUBLE xmax[3]);
MFTL_ErrorCode MFTL_InitializeObjectWithComm(MFTL_Object *Object, MFTL_INT M[3], MFTL_DOUBLE xmin[3], MFTL_DOUBLE xmax[3], MPI_Comm ObjectComm);

/*****************************************************************************/
/* General Object generation/manipulation                                    */
/*****************************************************************************/
MFTL_ErrorCode MFTL_GenerateSphere(MFTL_Object *Object, MFTL_INT refine);
MFTL_ErrorCode MFTL_ScaleSurface(MFTL_Object *Object, MFTL_DOUBLE scale[]);
MFTL_ErrorCode MFTL_GenerateIsocahedron(MFTL_Object *Object);
MFTL_ErrorCode MFTL_TessalateEdge_Sphere(MFTL_Vertex *Vertex, MFTL_Edge *Edge);
MFTL_ErrorCode MFTL_TessalateSphere(MFTL_Object *Object);
MFTL_ErrorCode MFTL_FindTriangleEdgesandVertices(MFTL_Object *Object);

/* Moving, rotating objects and Quaternions */
VecD VecD_Add(VecD I_A, VecD I_B);
MFTL_DOUBLE Norm_VecD(VecD I);
MFTL_DOUBLE Norm_QuadD(QuatD I);
QuatD QuatD_Multiply_ByScalar(QuatD I_q, MFTL_DOUBLE scalar);
MFTL_ErrorCode Multipy_pQuatD_ByScalar(QuatD *target, MFTL_DOUBLE scalar);
QuatD QuatD_Multiply(QuatD I_pre, QuatD I_post);
MFTL_ErrorCode ReNorm_QuatD(QuatD *target);
QuatD QuatD_Conj(QuatD I);
QuatD QuatD_Inv(QuatD I);
QuatD QuatD_FromW(VecD w_in, MFTL_DOUBLE f_timestep);
VecD ReframeUI(QuatD Q, QuatD Q_inv, VecD vector);
QuatD QuatD_From_angles(VecD angles);
MFTL_ErrorCode MFTL_MoveVertex(MFTL_Vertex *Vertex, MFTL_DOUBLE move[3]);
MFTL_ErrorCode MFTL_MoveBody(MFTL_Object *Object, MFTL_DOUBLE move[3]);
MFTL_ErrorCode MFTL_RotateVertex(MFTL_Vertex *v, QuatD Q_rot, QuatD Q_inv, VecD Centre);
MFTL_ErrorCode MFTL_RotateBody(MFTL_Object *Object, MFTL_DOUBLE angle[3], MFTL_DOUBLE centre_of_rotation[3]);

/*****************************************************************************/
/* Input/Output functions                                                    */
/*****************************************************************************/
MFTL_ErrorCode MFTL_ShowSummary(MFTL_Object *Object);
MFTL_ErrorCode MFTL_VertexShowSummary(MFTL_Vertex *V, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_EdgeShowSummary(MFTL_Edge *E, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_TriangleShowSummary(MFTL_Triangle *T, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_IntArrayShowSummary(MFTL_INT *array, MFTL_INT n, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_DoubleArrayShowSummary(MFTL_DOUBLE *array, MFTL_INT n, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_ObjectDomainShowSummary(MFTL_Object *Object, FILE *fp, const char *msg);
MFTL_ErrorCode MFTL_StatShowSummary(MFTL_Stats *s, FILE *fp, const char *msg);

/*****************************************************************************/
/* Collective/Parallel operations                                            */
/*****************************************************************************/
MFTL_ErrorCode MFTL_RedistributeObject(MFTL_Object *Object, MFTL_BOOL LayerOfGhostTriangles);
MFTL_ErrorCode MFTL_RedistributeObjectBasedOnTriangleCenter(MFTL_Object *Object, MFTL_BOOL LayerOfGhostTriangles);
MFTL_ErrorCode MFTL_SendTrianglesToProcWCenter(MFTL_Object *Object, MFTL_BOOL ForthOrBack);
MFTL_INT MFTL_ProcOfTriangleCenter(MFTL_Object *Object, MFTL_Triangle *T);
MFTL_INT MFTL_ProcOfVertex(MFTL_Object *Object, MFTL_Vertex *V);
MFTL_ErrorCode MFTL_CreateTriangleGhostPattern(MFTL_Object *Object);
MFTL_ErrorCode MFTL_AssureOneRingTriangleGhostNeighbourhood(MFTL_Object *Object);
MFTL_ErrorCode MFTL_AssureTwoRingTriangleGhostNeighbourhood(MFTL_Object *Object);

#endif