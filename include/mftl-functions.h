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
/* Initialisation/Finalisation                                               */
/*****************************************************************************/
MFTL_ErrorCode MFTL_InitializeObject(MFTL_Object *Object, MFTL_INT M[3], MFTL_DOUBLE xmin[3], MFTL_DOUBLE xmax[3]);
MFTL_ErrorCode MFTL_InitializeObjectWithComm(MFTL_Object *Object, int M[3], double xmin[3], double xmax[3], MPI_Comm ObjectComm);

/*****************************************************************************/
/* General Object generation/manipulation                                    */
/*****************************************************************************/
MFTL_ErrorCode MFTL_GenerateSphere(MFTL_Object *Object, MFTL_INT refine);
MFTL_ErrorCode MFTL_ScaleSurface(MFTL_Object *Object, MFTL_DOUBLE scale[]);
MFTL_ErrorCode MFTL_GenerateIsocahedron(MFTL_Object *Object);
MFTL_ErrorCode MFTL_TessalateEdge_Sphere(MFTL_Vertex *Vertex, MFTL_Edge *Edge);
MFTL_ErrorCode MFTL_TessalateSphere(MFTL_Object *Object);

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

#endif