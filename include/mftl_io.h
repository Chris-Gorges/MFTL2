/**
 * @file mftl_io.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_io
 *
 * @brief Mesh output in VTK formats: a serial debug dump and a parallel writer.
 *
 * Nothing here is read back by the library -- this is the module every other
 * module's correctness is ultimately JUDGED through, because it is how a human
 * looks at what the rest of the library produced.
 *
 * ## THREE DECISIONS THAT DECIDE WHAT THIS MODULE COSTS AND WHAT IT MEANS
 *
 * 1. THE DEFAULT ENCODING IS RAW BINARY, NOT ASCII.
 *
 *    The straightforward writer emits one `fprintf("%.10e %.10e %.10e\n")` per
 *    point and one per cell record. Decimal formatting of a double costs on the order
 *    of a hundred nanoseconds and produces ~25 bytes where the value occupies 8,
 *    so an ASCII writer is CPU-bound on `printf` long before it is bound by the
 *    disk. VTK's XML format has carried a raw appended-binary section since 2003
 *    precisely for this; using it turns the emit phase into a memcpy.
 *
 *    ASCII is kept (MFTL_VTK_ASCII) because a text file you can `less` is worth
 *    a great deal when a mesh is wrong, and because it is the honest baseline the
 *    binary path is measured against.
 *
 * 2. THERE ARE NOW TWO FILE LAYOUTS, AND THE CHOICE IS A SCALING DECISION.
 *
 *    MFTL_IO_FILE_PER_RANK  P pieces plus a .pvtu index. No shared file, so the
 *                           ranks never contend; but it creates P+1 files per
 *                           dump, and on a parallel filesystem the metadata
 *                           server -- not the bandwidth -- is what falls over
 *                           first.
 *
 *    MFTL_IO_SINGLE_FILE    ONE .vtu written collectively with MPI-IO. Every
 *                           rank contributes a slice of each data array at an
 *                           offset it computes from one MPI_Exscan, and
 *                           MPI_File_write_at_all lets the MPI implementation's
 *                           two-phase collective buffering merge those slices
 *                           into a few large, stripe-aligned writes issued by a
 *                           subset of aggregator ranks.
 *
 *    Which is faster is a property of the MACHINE, not of the code, and this
 *    library will not pretend otherwise: on a laptop with one local SSD the
 *    per-rank layout usually wins, because there is no aggregation to be had and
 *    a shared file only adds contention. On a Lustre/GPFS system with P in the
 *    thousands the single file wins by a margin that grows with P, and the
 *    per-rank layout eventually stops working at all. docs/performance-analysis.md
 *    carries the measurement on the development machine; RE-RUN IT on the target.
 *
 * 3. ONLY OWNED ENTITIES ARE WRITTEN.
 *
    Writing every triangle the rank holds, ghosts included, is the obvious loop
 *    and it is wrong. A ghost is a COPY of a triangle another rank owns, so the
 *    merged dataset then contains every halo triangle two or more times:
 *    invisible when you look at the surface, but any integral ParaView computes
 *    over it (total area, enclosed volume) is silently wrong by the halo fraction
 *    -- 10-30 % in a typical run. Ownership is an array prefix here, so writing
 *    owned-only costs nothing but a loop bound. `include_ghosts` asks for the
 *    whole pool deliberately, for when the halo itself is what you want to look
 *    at.
 *
 * 4. THE PERIODIC UNWRAPPING IS FREE.
 *
 *    A triangle straddling a periodic face has corners stored at opposite ends of
 *    the box; drawn naively it stretches across the whole domain. The previous
 *    version detected this by calling MFTLGetMinimumImageVector twice per
 *    triangle inside a helper it then called FOUR TIMES per triangle per file
 *    (its own src/io/README.md §3.3 records the redundancy).
 *
 *    Here `tris.anchor` and `tris.edge` already hold exactly what that helper
 *    computed -- corner 0 and the two seam-consistent edge vectors -- so the
 *    unwrapped corners are p0, p0+e1, p0+e2 with no periodic arithmetic at all,
 *    and the analysis is a single streaming pass over two contiguous arrays. See
 *    src/io/vtu_writer.c §2 for the detection criterion and why it is the safe
 *    one to get wrong.
 */

#ifndef MFTL_IO_H
#define MFTL_IO_H

#include "mftl_types.h"

// -----------------------------------------------------------------------------
// 1. OPTIONS
// -----------------------------------------------------------------------------

/** @brief How the numbers are encoded in a .vtu file. */
typedef enum
{
  /** Raw little-endian binary in an appended section. The default: ~10x faster
   *  to write and ~2.5x smaller than ASCII, and read natively by ParaView/VisIt. */
  MFTL_VTK_BINARY = 0,

  /** Decimal text inside the DataArray elements. Human-readable and diffable,
   *  and the only encoding whose file you can inspect without a tool. Costs
   *  roughly an order of magnitude in write time. Not available with
   *  MFTL_IO_SINGLE_FILE -- see MFTLWriteVtu(). */
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
 * @brief What to write and how. Get a filled-in default from MFTLVtkDefaults().
 *
 * The optional arrays are all things that are ALWAYS well-defined: identity and
 * ownership. Note what is deliberately absent -- there is no option to write
 * `tris.area` or `tris.normal`. Nothing on the object records whether those are
 * current (`derived` tracks the topology and the frames, not the geometry), and
 * a redistribution leaves them holding unspecified values by design. A writer
 * that emitted them would produce a file that looks right and is stale, which is
 * the exact failure mode this library spends its architecture avoiding. Compute
 * them and write your own array if you need them.
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
 * @brief Writes the object held by the CALLING rank to a legacy ASCII .vtk file.
 *
 * The debug dump: every entity the rank currently holds, ghosts included, with
 * no filtering, no periodic unwrapping and no communication. What you see is
 * exactly the object's raw in-memory state, which is the point -- when a mesh is
 * broken you want the broken thing on screen, not a cleaned-up rendering of it.
 *
 * It writes `verts.normal` and `tris.area` UNVALIDATED, which is the one place
 * in this module that does. MFTLWriteVtu refuses to, because nothing on the
 * object says whether they are current and a stale-but-plausible production file
 * is a trap; here showing you an array a redistribution left undefined is how
 * you find out that it is undefined.
 *
 * Strictly single-rank. It takes no domain and issues no MPI call, so calling it
 * from several ranks with the SAME path is a filesystem-level race. Either call
 * it from rank 0 on a rank-0-resident mesh (what examples/sphere_creation_serial.c
 * does) or give every rank a distinct name.
 *
 * @param object        The object to write. Must have valid topology (half-edges).
 * @param base_filename Base path; ".vtk" is appended.
 * @return MFTL_SUCCESS, or MFTL_ERROR on a NULL argument or an unwritable path.
 */
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename);

/**
 * @brief Writes every object of an ensemble as one VTK XML unstructured grid.
 *
 * All objects are aggregated into a single point/cell list per piece, with each
 * object's connectivity shifted by a running vertex offset so that body 2's
 * vertex 0 does not collide with body 1's.
 *
 * ── Collective ─────────────────────────────────────────────────────────────
 *
 * Every rank of the ensemble must call it, at both layouts:
 *
 *   MFTL_IO_SINGLE_FILE     genuinely collective -- MPI_Exscan, MPI_Allreduce and
 *                           MPI_File_*_all, all on the domain communicator.
 *   MFTL_IO_FILE_PER_RANK   the piece writes are independent, but rank 0's .pvtu
 *                           index promises one piece file per rank, and the
 *                           return code is MPI_Allreduce'd so that a rank which
 *                           failed to open its file is not the only one that
 *                           knows. Both make it collective in practice.
 *
 * ── Preconditions ──────────────────────────────────────────────────────────
 *
 * None on the geometry: positions are the only thing read, and they are always
 * authoritative. Current derived arrays (MFTLObjectHasTopology / HasFrames) are
 * not required but ARE used when present -- they turn the periodic-unwrap
 * analysis from a gathering pass into a streaming one. A freshly generated mesh
 * whose frames were never built writes correctly, just a little slower.
 *
 * ── Restrictions ───────────────────────────────────────────────────────────
 *
 * - MFTL_IO_SINGLE_FILE with MFTL_VTK_ASCII returns MFTL_ERROR. A single shared
 *   file exists to be fast, and ASCII text of variable width in a shared file is
 *   a contradiction; refusing it is clearer than silently promoting it.
 * - Point indices are Int32, so a single file holding more than INT32_MAX points
 *   is rejected rather than written wrong. That is ~48 GB of coordinates; a
 *   visualisation that large wants a different tool, not a wider index.
 *
 * @param ensemble        The ensemble. Zero objects is legal and produces a
 *                        valid, empty dataset.
 * @param filename_prefix Base path/name; the suffixes are appended.
 * @param options         MFTL_NULL for MFTLVtkDefaults().
 * @return MFTL_SUCCESS if EVERY rank succeeded, MFTL_ERROR otherwise (the same
 *         value on every rank).
 */
int MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix, const MFTL_VtkOptions *options);

#endif  // MFTL_IO_H
