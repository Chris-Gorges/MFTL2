/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:35:27
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
 */

#ifndef MFTL_OBJECT_H
#define MFTL_OBJECT_H

#include <mpi.h>  // For MPI types like MPI_Comm
#include "mftl_types.h"
#include "mftl_constants.h"

//------------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------------
/**
 * @brief Creates and initializes a new MFTL_Object mesh.
 * @param comm The MPI communicator to be used for this mesh.
 * @return A pointer to the newly allocated object, or NULL on failure.
 */
MFTL_Object *MFTLObjectCreate(MPI_Comm comm);

/**
 * @brief Frees all memory associated with an MFTL_Mesh object.
 * @param mesh A pointer to the mesh to be destroyed.
 */
void MFTLObjectDestroy(MFTL_Object **mesh);

/**
 * @brief Initializes the Cartesian domain decomposition for the mesh.
 *
 * This function calculates the spatial region (bounding box) owned by each
 * MPI rank based on a specified number of "cuts" in each dimension.
 *
 * @param mesh The mesh object to initialize.
 * @param cuts An array of 3 integers specifying the number of divisions in x,
 * y, z.
 * @param global_min The minimum coordinates of the global simulation domain.
 * @param global_max The maximum coordinates of the global simulation domain.
 * @param is_periodic An array of 3 booleans indicating periodicity in x,y,z.
 * @return MFTL_SUCCESS on success, MFTL_ERROR on failure.
 */
mftl_int_t MFTLObjectInitDomain(MFTL_Object *object, const mftl_int_t cuts[3], const double global_min[3], const double global_max[3],
                                const bool is_periodic[3]);

/**
 * @brief Public function to generate a sphere.
 * @param center The center position of the sphere.
 * @param radius The radius of the sphere.
 * @param refinements The number of refinements for the sphere.
 */
mftl_int_t MFTLGenerateSphere(MFTL_Object *object, const double center[3], double radius, mftl_int_t refinements);

void MFTLUpdateObjectGlobalCounts(MFTL_Object **objects, mftl_int_t num_objects);

#endif  // MFTL_OBJECT_H