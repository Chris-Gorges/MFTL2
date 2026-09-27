/*
 * MFTL2 I/O — visualisation output (legacy VTK, VTU/PVTU), GTS surface files and the native
 * restart mesh (v1: MFTL1 `.mesh` format; v2: class, period box, exact positions with image
 * counters and boundary tags).
 */
#ifndef MFTL_IO_H
#define MFTL_IO_H

#include "mftl_core.h"

/**
 * @brief Writes the entities stored on the calling rank (owned and ghost) to a legacy ASCII
 * VTK polydata file <base_filename>.vtk, with the stored (wrapped) positions. Rank-local
 * visualisation output.
 * @return MFTL_SUCCESS, or MFTL_ERROR if the file cannot be written.
 */
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename);

/**
 * @brief Writes all objects of the ensemble as one parallel VTK set (collective): every rank
 * writes its stored entities (owned and ghost) to <prefix>_%05d.vtu, rank 0 writes the index
 * <prefix>.pvtu. A triangle crossing a periodic seam is drawn with its own copies of its
 * corners in the image of its first corner (minimum image), so it is not stretched across the
 * box. Point data: VertexOwner; cell data: TriangleOwner, BodyID.
 */
void MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix);

/**
 * @brief Reads a GTS surface file into an empty object (collective).
 *
 * Token-based parser for the MFTL1 GTS layout: "nV nE nT GtsSurface GtsFace
 * GtsEdge GtsVertex" header, then vertex coordinates, 1-based edge vertex
 * pairs, and triangles as 1-based edge triples. Rank 0 parses and builds; the gid
 * counters are broadcast, so the object is immediately redistributable via
 * MFTLRedistributeListOfObjects (as after MFTLGenerateSphere).
 *
 * The triangles are oriented consistently; a closed body is then canonicalised to OUTWARD
 * winding (V > 0). The coordinates are continuous input and are lifted to (pos, img). An edge
 * with one triangle is accepted only on a body declared open before the call (its loops become
 * tagged boundaries); non-manifold input is rejected.
 *
 * @return MFTL_SUCCESS on success, MFTL_ERROR on error (all ranks agree).
 */
int MFTLReadGTS(MFTL_Object* obj, const char* path);

/**
 * @brief Writes the object's owned mesh to a GTS file.
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
 * @brief Writes the ensemble as a parallel VTK collection: per-rank <stem>_%05d.vtu pieces
 * plus the rank-0 <stem>.pvtu index (same as MFTLWriteVtu).
 */
void MFTLWritePvtu(MFTL_Ensemble* ens, const char* stem);

/**
 * @brief Native restart mesh v1 in the MFTL1 `.mesh` format (surface scalar with %.15e).
 * MFTLWriteNativeMesh writes the LOCALLY stored entities of obj to `path` (gather the object
 * to one rank first, e.g. MFTLEnsembleBeginSerial). MFTLReadNativeMesh builds an EMPTY object
 * from a v1 or v2 file (collective: ranks passing path == NULL contribute nothing; call
 * MFTLRedistribute afterwards). A v1 file is read as closed (or with its boundary derived from
 * the mesh when the object is declared open) with img = 0, then canonicalised.
 */
int MFTLWriteNativeMesh(const MFTL_Object* obj, const char* path);
/** @brief Native mesh v2 (declared class, period box, exact %.17g positions with image
 *  counters, boundary tags, explicit vertex triples); read back by MFTLReadNativeMesh on any
 *  rank count (then MFTLRedistribute). The reader re-establishes the canonical image gauge,
 *  so a state written in the canonical gauge is restored exactly. */
int MFTLWriteNativeMeshV2(const MFTL_Object* obj, const char* path);
int MFTLReadNativeMesh(MFTL_Object* obj, const char* path);

#endif  // MFTL_IO_H