/*=====================================================================================
 *
 *        Filename:  test1.c
 *
 *     Description:  Function that tests basic workflow of MFTL:
 *                   1. Generate a refined sphere
 *                   2. Scale it to an ellipsoid 2x1x1
 *                   3. Rotate by 30 deg around z-axis, move to (0.5,0,0)
 *                   4. Redistribute over processors
 *                   5. Calculate properties before/after each step
 *                   6. Save to GTS and vtk.
 *
 *          Author:  Berend van Wachem (BvW), berend.van.wachem@gmail.com
 *       Copyright:  (c) Berend van Wachem.

 * =====================================================================================*/

#include "mftl.h"
#include <stdio.h>

int main(int argc, char *argv[])
{
  MFTL_Object MyObject;

  // Domain size
  MFTL_DOUBLE xmin[] = {-5.0, -5.0, -5.0};
  MFTL_DOUBLE xmax[] = {5.0, 5.0, 5.0};

  // Number of processors in each direction
  MFTL_INT M[] = {4, 2, 1};

  // Auxiliary variables: Scale, Translation, Rotation (radians)
  MFTL_DOUBLE scale[3] = {2.0, 1.0, 1.0};
  MFTL_DOUBLE move[3] = {0.5, 0.0, 0.0};
  MFTL_DOUBLE angle[3] = {0.0, 0.0, 0.523599};

  // Object Volume and centre
  MFTL_DOUBLE Volume = 0.0, Center[3];

  MPI_Init(&argc, &argv);
  MFTL_InitializeObject(&Object, M, xmin, xmax);

  // Generate sphere
  /**********************************************/
  if (Object.rank == 0)
  {
    MFTLGenerateSphere(&Object, 4);
    MFTLScaleSurface(&Object, scale);
  }
  /**********************************************/

  // Print the data about the object (distribution over processors, volume,
  // centre)
  MFTLShowSummary(&Object);
  Volume = MFTLSurfaceCenter(&Object, Center);
  printf("Proc %d: Object Characteristics: Volume %lf, Centre: %e %e %e\n", Object.rank, Volume, Center[0], Center[1], Center[2]);
  MPI_Barrier(Object.OBJECT_COMM_WORLD[0]);

  //  // Rotate and move the body
  double angle1[3] = {0.0, 0.0, 0.523599};
  double centre_of_rotation[3] = {0., 0., 0.};
  MFTLRotateBody(&Object, angle1, centre_of_rotation);

  MFTLMoveBody(&Object, move);

  MFTLShowSummary(&Object);
  Volume = MFTLSurfaceCenter(&Object, Center);
  printf("Proc %d: Object Characteristics: Volume %lf, Centre: %e %e %e\n", Object.rank, Volume, Center[0], Center[1], Center[2]);
  MPI_Barrier(Object.OBJECT_COMM_WORLD[0]);

  // Redistribute object over processors
  MFTLRedistributeObject(&Object, MFTL_NO);

  MFTLShowSummary(&Object);
  Volume = MFTLSurfaceCenter(&Object, Center);
  printf("Proc %d: Object Characteristics: Volume %lf, Centre: %e %e %e\n", Object.rank, Volume, Center[0], Center[1], Center[2]);
  MPI_Barrier(Object.OBJECT_COMM_WORLD[0]);

  // Find the triangle properties: centres and normals
  MFTLFindTriangleCentres(&Object);
  MFTLFindTriangleNormals(&Object);

  // Write Object to *.vtk file
  char name[10] = "test1";
  MFTLWriteObjectToVTK(&Object, name);
  MFTLWriteObjectToGTS(&Object, name);

  // Cleanup memory
  MFTLFinalizeObject(&Object);

  MPI_Finalize();
  return (0);
}
