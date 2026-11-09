/**
 * @file mftl_io.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_io
 *
 * @brief Mesh output in VTK formats: a serial legacy debug dump and a parallel
 *        VTK XML (`.vtu`) writer.
 *
 * Nothing is read back by the library. Four decisions define MFTLWriteVtu():
 *
 * 1. **Raw binary by default.** Decimal formatting of a double costs about
 *    100 ns and ~17 bytes, so an ASCII writer is bound by `printf`, not by the
 *    disk; the appended raw section is a memcpy (23-32x faster, docs/analysis.md
 *    §3). ASCII is kept for inspection.
 * 2. **Two file layouts.** One piece per rank plus a `.pvtu` index (no
 *    contention, but P+1 files per dump -- what overloads a parallel file
 *    system's metadata server first), or one shared file written collectively
 *    with MPI-IO. On one local SSD the per-rank layout was faster (the shared
 *    file: 1.0x at P = 1, 0.45x at P = 4); re-measure on the target system.
 * 3. **Only owned entities are written.** A ghost is a copy of a triangle another
 *    rank owns; writing it would count every halo triangle once per holder and
 *    bias every integral a visualisation tool computes (10-30 % in a typical
 *    run). `include_ghosts` exists for inspecting the halo.
 * 4. **Periodic unwrapping at write time.** A triangle straddling a periodic face
 *    gets three private points at its unwrapped corners; the mesh in memory is
 *    untouched.
 *
 * Theory: src/io/README.md.
 */

#ifndef MFTL_IO_H
#define MFTL_IO_H

#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

// -----------------------------------------------------------------------------
// 1. OPTIONS
// -----------------------------------------------------------------------------

/** @brief How the numbers are encoded in a .vtu file. */
typedef enum
{
  /** Raw binary in an appended section, in the host byte order (declared in
   *  the header). The default: 23-32x faster to write than ASCII and about half
   *  the size; read natively by ParaView and VisIt. */
  MFTL_VTK_BINARY = 0,

  /** Decimal text inside the DataArray elements: human-readable and diffable.
   *  Not available with #MFTL_IO_SINGLE_FILE. */
  MFTL_VTK_ASCII = 1
} MFTL_VtkEncoding;

/** @brief How the output is spread over files. */
typedef enum
{
  /** `<prefix>_%05d.vtu` from every rank plus `<prefix>.pvtu` from rank 0.
   *  No shared file; P+1 files per dump. */
  MFTL_IO_FILE_PER_RANK = 0,

  /** One `<prefix>.vtu` holding a single Piece, written collectively with
   *  MPI-IO. Implies MFTL_VTK_BINARY. */
  MFTL_IO_SINGLE_FILE = 1
} MFTL_IoLayout;

/**
 * @struct MFTL_VtkOptions
 * @brief What to write and how; start from MFTLVtkDefaults().
 *
 * The optional arrays are identity and ownership, which are always
 * well-defined. There is deliberately no option for `tris.area` or
 * `tris.normal`: nothing records whether they are current and a redistribution
 * leaves them unspecified, so a file containing them could be stale while
 * looking right. Compute and write them yourself if needed.
 */
typedef struct MFTL_VtkOptions
{
  MFTL_IoLayout layout;  ///< Default MFTL_IO_FILE_PER_RANK.
  MFTL_VtkEncoding encoding;  ///< Default MFTL_VTK_BINARY.

  /** Write ghost entities too. Default false. The merged dataset then contains
   *  every halo triangle once per rank holding it, so use it to LOOK at the
   *  halo, never to integrate over the surface. */
  bool include_ghosts;

  /** `VertexOwner` / `TriangleOwner` (Int32): which rank owns each entity.
   *  Default true -- this is how you audit a domain decomposition visually. */
  bool write_owner;

  /** `BodyID` (Int32): the object's `global_id`, per cell. Default true; it is
   *  what lets a visualisation tool re-separate the bodies after aggregation. */
  bool write_body_id;

  /** `VertexGID` / `TriangleGID` (Int32): the entity global ids. Default false.
   *  Useful for following one entity across timesteps or across a
   *  redistribution; 8 bytes per triangle of file you usually do not want. */
  bool write_global_id;
} MFTL_VtkOptions;

/**
 * @brief The parameter set a caller should start from.
 * @return Binary encoding, one file per rank, owned entities only, owner and
 *         body id written, global ids not.
 */
MFTL_VtkOptions MFTLVtkDefaults(void);

// -----------------------------------------------------------------------------
// 2. WRITERS
// -----------------------------------------------------------------------------

/**
 * @brief Writes the calling rank's pool of one object to a legacy ASCII `.vtk`
 *        file: the raw debug dump.
 *
 * Every entity the rank holds, ghosts included, with no filtering, no periodic
 * unwrapping and no communication. It also writes `VertexOwner`,
 * `TriangleOwner`, `verts.normal` and `tris.area` UNVALIDATED -- its contract is
 * to show what is in memory, including arrays a redistribution left undefined.
 *
 * Strictly single-rank: several ranks writing the same path race. Call it from
 * one rank, or give every rank its own name.
 *
 * @param object        The object (valid topology required).
 * @param base_filename Base path; ".vtk" is appended.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument or an I/O failure.
 */
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename);

/**
 * @brief Writes every local object of an ensemble as one VTK XML unstructured
 *        grid (all bodies in one point/cell list per piece).
 *
 * Points are the vertices referenced by the written triangles (owned triangles
 * may reference ghost vertices), plus three private points per triangle that
 * straddles a periodic face; a seam vertex is therefore written by every rank
 * with a triangle on it. Cells carry `TriangleOwner`, `BodyID` and optionally
 * `TriangleGID`; points `VertexOwner` and optionally `VertexGID`. Uses compact
 * connectivity when current, the half-edge records otherwise; no geometry
 * precondition.
 *
 * - #MFTL_IO_FILE_PER_RANK writes `<prefix>_%05d.vtu` per rank and
 *   `<prefix>.pvtu` on rank 0.
 * - #MFTL_IO_SINGLE_FILE writes one `<prefix>.vtu` with one Piece: one
 *   non-blocking `MPI_Exscan` + `MPI_Allreduce` pair places every block, rank 0
 *   writes header, length prefixes and footer, and every rank writes its slice of
 *   each block with `MPI_File_write_at_all` (enabling two-phase collective
 *   buffering). The file is truncated to its exact size. Requires
 *   #MFTL_VTK_BINARY and at most \f$2^{31}-1\f$ points and
 *   \f$(2^{31}-1)/3\f$ cells (Int32 connectivity and offsets).
 *
 * @param ensemble        The ensemble; zero objects yield a valid empty dataset.
 * @param filename_prefix Base path/name; suffixes are appended.
 * @param options         #MFTL_NULL for MFTLVtkDefaults().
 * @return #MFTL_SUCCESS if every rank succeeded, #MFTL_ERROR otherwise -- the same
 *         value on every rank.
 *
 * @collective In both layouts (the return code is reduced).
 */
int MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix, const MFTL_VtkOptions *options);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_IO_H
