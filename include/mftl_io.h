#ifndef MFTL_IO_H
#define MFTL_IO_H

#include "mftl_types.h"

//------------------------------------------------------------------------------
// Function Prototypes
//------------------------------------------------------------------------------
/**
 * @brief Writes the object from a single process to a legacy VTK file.
 *
 * This function is for simple, serial visualization. It writes the entire
 * object data held by the calling process to a single .vtk file.
 *
 * @param object The object object to write.
 * @param base_filename The base name for the output file (e.g.,
 * "output/sphere"). .vtk will be appended.
 * @return 0 on success, -1 on error.
 */
mftl_int_t MFTLWriteVtk(const MFTL_Object *object, const char *base_filename);

/**
 * @brief Writes ALL MFTL objects to a SINGLE parallel VTK file set (.pvtu + .vtu).
 * * Features:
 * - Aggregates multiple independent meshes (spheres) into one UnstructuredGrid.
 * - Maintains "Smart Periodic Unwrapping" for each object.
 * - Adds "BodyID" data array to distinguish between objects.
 * * @param objects Array of pointers to MFTL_Object.
 * @param num_objects Number of objects in the array.
 * @param filename_prefix Base path/name for output files.
 */
void MFTLWriteVtu(MFTL_Object **objects, int num_objects, const char *filename_prefix);

#endif  // MFTL_IO_H