#ifndef MFTL_IO_H
#define MFTL_IO_H

#include "mftl_core.h"

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
 * - Supports "Smart Periodic Unwrapping" for each object.
 * * @param ensemble The ensemble object containing the objects to write.
 * @param filename_prefix Base path/name for output files.
 */
void MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix);

#endif  // MFTL_IO_H