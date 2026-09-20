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
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename);

/**
 * @brief Writes ALL MFTL objects to a SINGLE parallel VTK file set (.pvtu + .vtu).
 * * Features:
 * - Aggregates multiple independent meshes (spheres) into one UnstructuredGrid.
 * - Supports "Smart Periodic Unwrapping" for each object.
 * * @param ensemble The ensemble object containing the objects to write.
 * @param filename_prefix Base path/name for output files.
 */
void MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix);

/**
 * @brief Reads a GTS surface file into the object (Stage 05).
 *
 * Token-based parser for the MFTL1 GTS layout: "nV nE nT GtsSurface GtsFace
 * GtsEdge GtsVertex" header, then vertex coordinates, 1-based edge vertex
 * pairs, and triangles as 1-based edge triples. Rank 0 parses; the gid
 * counters are broadcast so the read mesh matches MFTLGenerateSphere's
 * post-generation distribution and is immediately redistributable via
 * MFTLRedistributeListOfObjects.
 *
 * Triangle windings are canonicalized so the enclosed signed volume is
 * POSITIVE. Open / non-manifold input is rejected with MFTL_ERROR.
 *
 * @return MFTL_SUCCESS on success, MFTL_ERROR on error (all ranks agree).
 */
int MFTLReadGTS(MFTL_Object* obj, const char* path);

/**
 * @brief Writes the object's owned mesh to a GTS file (Stage 05).
 *
 * Single-file serial semantics: rank 0 writes the triangles it owns, with
 * vertices and edges renumbered in first-appearance order and coordinates
 * printed at %.17g (bit-exact round-trip through MFTLReadGTS). Call BEFORE
 * redistribution (or on a serial mesh). If the surface has been distributed
 * (rank 0 no longer owns all of the object's triangles), the function prints
 * an error and returns MFTL_ERROR rather than writing a partial file. The
 * ownership check uses the object's global counts, so refresh them after
 * redistribution (MFTLEnsembleUpdateGlobalCounts).
 *
 * @return MFTL_SUCCESS on success, MFTL_ERROR on error.
 */
int MFTLWriteGTS(MFTL_Object* obj, const char* path);

/**
 * @brief Writes the ensemble as a parallel VTK collection (Stage 05):
 * per-rank <stem>_%05d.vtu pieces plus the rank-0 <stem>.pvtu index.
 * Thin wrapper over MFTLWriteVtu (which produces the whole set).
 */
void MFTLWritePvtu(MFTL_Ensemble* ens, const char* stem);

#endif  // MFTL_IO_H