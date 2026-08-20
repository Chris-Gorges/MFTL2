/**
 * @file vtu_writer.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_io
 *
 * @brief The parallel VTK XML writer: selection, periodic unwrapping, block
 *        assembly, and the two output backends (POSIX per rank, MPI-IO shared).
 *
 * ## THE SHAPE OF THE THING
 *
 * A .vtu file is a short XML header followed by a handful of flat arrays. So the
 * writer is three phases, and every one of them is separable from the others:
 *
 *   ANALYSE   decide which cells go in the file, which points they reference,
 *             and which cells straddle a periodic face. Per rank, no MPI.
 *   ASSEMBLE  build each DataArray as one contiguous byte block in memory.
 *             Per rank, no MPI.
 *   EMIT      put those blocks on disk. THIS is the only phase that differs
 *             between the encodings and the layouts.
 *
 * That split is what lets one code path serve ascii/binary x per-rank/shared:
 * ANALYSE and ASSEMBLE never learn which backend they are feeding, and the
 * backends never learn what a triangle is.
 *
 * ## WHAT COSTS WHAT, MEASURED RATHER THAN ASSUMED
 *
 * The previous version's writer spent its time in printf, not in the kernel or
 * the disk. Formatting one double as "%.10e" is ~100 ns and 17 bytes for an
 * 8-byte value; ASSEMBLE for the binary path is a memcpy. That is the single
 * biggest factor and it is entirely independent of MPI.
 *
 * MPI-IO changes something else: WHERE the bytes go. See §6.
 */

#include "mftl_io.h"

/**
 * @def ASCII_ARRAY
 * @brief Emits one enabled block as an ASCII `<DataArray>` with its numbers inline.
 *
 * Defined inside write_piece_file(), where the buffer and the stream it writes to
 * are in scope.
 *
 * @param idx Block index in the `blocks` table.
 */

/**
 * @def DECL
 * @brief Fills in one row of the block table.
 *
 * Defined inside MFTLWriteVtu(), where the table is in scope.
 *
 * @param idx Block index.
 * @param nm  VTK array name.
 * @param ty  VTK type string.
 * @param kd  Element #scalar_kind.
 * @param nc  Components per tuple.
 * @param pc  `true` if the block is sized by cells rather than points.
 * @param en  `true` if the block is written at all.
 * @param un  Bytes per point (or per cell).
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/** VTK cell type code for a triangle (VTK_TRIANGLE). */
#define MFTL_VTK_TRIANGLE 5

MFTL_VtkOptions MFTLVtkDefaults(void)
{
  MFTL_VtkOptions o;
  o.layout = MFTL_IO_FILE_PER_RANK;
  o.encoding = MFTL_VTK_BINARY;
  o.include_ghosts = false;
  o.write_owner = true;
  o.write_body_id = true;
  o.write_global_id = false;
  return o;
}

/**
 * @brief "LittleEndian" or "BigEndian" for the VTKFile element.
 *
 * The raw appended section is written with plain memcpy, so the file's byte
 * order is the machine's and the header has to say so. Detected rather than
 * assumed: the library is expected to run on whatever the cluster has.
 *
 * @return The string literal `"LittleEndian"` or `"BigEndian"`.
 */
static const char *host_byte_order(void)
{
  const unsigned int one = 1u;
  unsigned char first;
  memcpy(&first, &one, 1);
  return first ? "LittleEndian" : "BigEndian";
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. A growable text buffer, for the XML header only
 *
 * The header is a few kilobytes whatever the mesh size, so this is not a hot
 * path -- it exists so that the header can be BUILT before its length is needed.
 * In the shared-file layout every rank builds the identical header text and uses
 * its length to place the appended section; that is the whole reason it is a
 * string in memory rather than a sequence of fprintf calls.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Growable text buffer, used for the XML header only.
 *
 * The header is a few kilobytes whatever the mesh size, so this is not a hot
 * path. It exists so the header can be BUILT before its length is needed: in the
 * shared-file layout every rank builds the identical header text and uses its
 * length to place the appended section.
 */
typedef struct
{
  char *s;  ///< NUL-terminated text.
  size_t len;  ///< Characters written, excluding the terminator.
  size_t cap;  ///< Bytes allocated.
} sbuf;

/**
 * @brief Initialises an empty text buffer.
 * @param b The buffer.
 */
static void sb_init(sbuf *b)
{
  b->cap = 4096;
  b->s = (char *) MFTLMallocOrDie(b->cap, "MFTL vtu header");
  b->s[0] = '\0';
  b->len = 0;
}

/**
 * @brief Releases the buffer.
 * @param b The buffer.
 */
static void sb_free(sbuf *b)
{
  free(b->s);
  b->s = MFTL_NULL;
  b->len = b->cap = 0;
}

/**
 * @brief Appends `printf`-formatted text, growing the buffer as needed.
 * @param b   The buffer.
 * @param fmt `printf` format string; the arguments follow.
 */
static void sb_addf(sbuf *b, const char *fmt, ...)
{
  va_list ap, ap2;
  va_start(ap, fmt);
  va_copy(ap2, ap);

  const int need = vsnprintf(MFTL_NULL, 0, fmt, ap);
  va_end(ap);

  if (need > 0)
  {
    if (b->len + (size_t) need + 1u > b->cap)
    {
      while (b->cap < b->len + (size_t) need + 1u) b->cap *= 2u;
      char *grown = (char *) realloc(b->s, b->cap);
      if (!grown)
      {
        fprintf(stderr, "FATAL ERROR: MFTL vtu header allocation failed.\n");
        exit(EXIT_FAILURE);
      }
      b->s = grown;
    }
    vsnprintf(b->s + b->len, b->cap - b->len, fmt, ap2);
    b->len += (size_t) need;
  }
  va_end(ap2);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. ANALYSE: which cells, which points, which of them straddle a seam
 *
 * ── Cell selection ─────────────────────────────────────────────────────────
 *
 * Owned triangles only, by default. Ownership is an array PREFIX, so "owned
 * only" is the loop bound `tris.n_local` and costs nothing. The previous version
 * wrote all `n_triangles`, ghosts included, so a triangle owned by rank A and
 * held as a ghost by ranks B and C appeared THREE times in the merged dataset.
 * Nothing about the picture changes -- the copies are coincident -- but every
 * integral the visualisation tool computes over the surface is then wrong by the
 * halo fraction, silently, and the halo fraction is 10-30 % in a normal run.
 *
 * ── Point selection ────────────────────────────────────────────────────────
 *
 * A written triangle can reference a GHOST vertex (its three corners may belong
 * to three different ranks), so the point list is not the owned prefix: it is
 * "every vertex some written triangle references". One marking pass over
 * `tris.verts` -- 12 contiguous bytes per triangle -- and one prefix scan turn
 * that into a compact per-piece index. Vertices referenced by nothing (they
 * exist: a vertex owned here whose entire one-ring is owned elsewhere) are
 * dropped rather than written as an unused point.
 *
 * The duplication this leaves is intentional and standard: a seam vertex is
 * written by every rank that has a triangle on it, so the surface is "cracked"
 * in the point topology and seamless in the picture.
 *
 * ── Periodic unwrapping ────────────────────────────────────────────────────
 *
 * Vertex positions are wrapped into the primary cell, so a triangle straddling a
 * periodic face has corners at opposite ends of the box and, drawn naively,
 * stretches across the entire domain. The fix is per-triangle and write-time:
 * emit THREE PRIVATE points holding unwrapped coordinates and point that one
 * triangle's connectivity at them. Other triangles sharing those vertices are
 * unaffected, and the mesh in memory is never touched.
 *
 * The previous version computed the unwrapped corners with two
 * MFTLGetMinimumImageVector calls inside a helper it then invoked FOUR TIMES per
 * triangle per file (once to size the point list, once per data block). Here
 * `tris.anchor` and `tris.edge` ALREADY hold corner 0 and the two
 * seam-consistent edge vectors -- that is what the frames are -- so:
 *
 *     unwrapped corners = p0, p0 + e1, p0 + e2
 *
 * with no periodic arithmetic at all, from two contiguous streaming arrays.
 *
 * ── The detection criterion, and why it is the safe one to get wrong ────────
 *
 * A triangle straddles iff a reconstructed corner falls OUTSIDE the domain box
 * on a periodic axis. If no minimum-image correction applied on that axis then
 * e[d] is exactly p1[d]-p0[d] (build_frames subtracts a literal zero, which is
 * exact), so p0+e reproduces an in-box position to within one rounding. If a
 * correction did apply, the reconstruction sits a FULL domain length outside the
 * box -- not a close call.
 *
 * So the only way this errs is a false POSITIVE, for a corner within one ulp of
 * a box face, and a false positive costs three extra points that carry the
 * correct coordinates and render identically. A false NEGATIVE would put a
 * triangle across the whole domain on screen. The test is therefore written with
 * NO tolerance: any slack traded for fewer false positives buys them at the
 * price of false negatives, which is the wrong direction.
 *
 * (The previous version compared the edge vector against a freshly recomputed
 * raw difference instead, which is exact in both directions -- but needs the two
 * far corner positions, i.e. a gather per triangle. That form survives here as
 * the fallback for a mesh whose frames were never built.)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Growable double buffer for the unwrapped corner coordinates. */
typedef struct
{
  double *data;  ///< [cap] the coordinates.
  int n;  ///< Doubles written.
  int cap;  ///< Doubles allocated.
} dbuf;

/**
 * @brief Initialises an empty coordinate buffer. Allocates nothing.
 * @param b The buffer.
 */
static void db_init(dbuf *b)
{
  b->data = MFTL_NULL;
  b->n = 0;
  b->cap = 0;
}

/**
 * @brief Releases the buffer.
 * @param b The buffer.
 */
static void db_free(dbuf *b)
{
  free(b->data);
  db_init(b);
}

/**
 * @brief Appends one triangle's three unwrapped corners (9 doubles).
 * @param b  The buffer.
 * @param p0 Corner 0.
 * @param e1 Edge vector to corner 1, seam-consistent.
 * @param e2 Edge vector to corner 2, seam-consistent.
 */
static void db_add_corners(dbuf *b, const double p0[3], const double e1[3], const double e2[3])
{
  MFTL_ENSURE_CAPACITY(b->data, &b->n, &b->cap, 9, double);
  double *o = b->data + b->n;
  for (int k = 0; k < 3; ++k)
  {
    o[0 + k] = p0[k];
    o[3 + k] = p0[k] + e1[k];
    o[6 + k] = p0[k] + e2[k];
  }
  b->n += 9;
}

/** @brief Everything the emit phase needs to know about one object on this rank. */
typedef struct
{
  int n_cells;  ///< Triangles written: tris.n_local, or tris.n with ghosts.
  int n_marked;  ///< Vertices referenced by them.
  int n_wrapped;  ///< How many of n_cells straddle a periodic face.
  int n_points;  ///< n_marked + 3 * n_wrapped.
  int *remap;  ///< [verts.n] compact point index, or MFTL_ERROR if unreferenced.
  unsigned char *wflag;  ///< [n_cells] 1 if the triangle straddles.
  int wpos_off;  ///< Offset of this object's slice in the shared corner buffer.
} piece;

/**
 * @brief Fills @p p for one object and appends its unwrapped corners to @p wpos.
 *
 * Three loop shapes rather than one with branches in it, because the periodic
 * test is loop-invariant in structure and the non-periodic case is common:
 *
 *   (a) non-periodic          mark only; nothing can straddle
 *   (b) periodic + frames     streaming: tris.verts + anchor + edge, no gather
 *   (c) periodic, no frames   gathering: the previous version's exact test
 *
 * @param      obj            The mesh.
 * @param      include_ghosts Count ghost triangles too. Off in production.
 * @param[out] p              Piece description: which vertices become shared
 *                            points, which triangles wrap, and the two totals.
 * @param[out] wpos           Receives the unwrapped corners of the wrapping
 *                            triangles, nine doubles each.
 */
static void analyze_piece(const MFTL_Object *obj, bool include_ghosts, piece *p, dbuf *wpos)
{
  const int nv = obj->verts.n;
  const int nt = include_ghosts ? obj->tris.n : obj->tris.n_local;
  const MFTL_Domain *const dom = obj->domain;
  const bool periodic = (dom != MFTL_NULL) && (dom->is_periodic[0] || dom->is_periodic[1] || dom->is_periodic[2]);

  p->n_cells = nt;
  p->n_wrapped = 0;
  p->wpos_off = wpos->n;

  for (int v = 0; v < nv; ++v) p->remap[v] = MFTL_ERROR;

  const bool streaming = MFTLObjectHasTopology(obj) && MFTLObjectHasFrames(obj);

  if (!periodic)
  {
    /* (a) Marking only. With the derived array this is a flat pass over 12 bytes
     * per triangle; without it, three loads with a 16-byte stride. */
    if (streaming)
    {
      const int *restrict tv = obj->tris.verts;
      for (int t = 0; t < nt; ++t)
      {
        p->remap[tv[3 * t + 0]] = 0;
        p->remap[tv[3 * t + 1]] = 0;
        p->remap[tv[3 * t + 2]] = 0;
      }
    }
    else
    {
      for (int t = 0; t < nt; ++t)
      {
        int v[3];
        MFTLGetTriangleVertices(obj, t, v);
        p->remap[v[0]] = 0;
        p->remap[v[1]] = 0;
        p->remap[v[2]] = 0;
      }
    }
    if (nt > 0) memset(p->wflag, 0, (size_t) nt);
  }
  else if (streaming)
  {
    /* (b) The streaming path. Reads three contiguous arrays and dereferences
     * nothing; the wrap test is six compares against the box. */
    const int *restrict tv = obj->tris.verts;
    const double *restrict anc = obj->tris.anchor;
    const double *restrict ed = obj->tris.edge;

    for (int t = 0; t < nt; ++t)
    {
      p->remap[tv[3 * t + 0]] = 0;
      p->remap[tv[3 * t + 1]] = 0;
      p->remap[tv[3 * t + 2]] = 0;

      const double *const a = &anc[3 * t];
      const double *const e = &ed[6 * t];

      int wrapped = 0;
      for (int k = 0; k < 3; ++k)
      {
        if (!dom->is_periodic[k]) continue;
        const double c1 = a[k] + e[k];
        const double c2 = a[k] + e[3 + k];
        if (c1 < dom->global_min[k] || c1 >= dom->global_max[k] || c2 < dom->global_min[k] || c2 >= dom->global_max[k]) wrapped = 1;
      }

      p->wflag[t] = (unsigned char) wrapped;
      if (wrapped)
      {
        db_add_corners(wpos, a, e, e + 3);
        p->n_wrapped++;
      }
    }
  }
  else
  {
    /* (c) No frames: recompute them for this triangle and use the exact test.
     * Correct on a mesh that was just generated or just read in. */
    const double *const pos = obj->verts.pos;
    for (int t = 0; t < nt; ++t)
    {
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      p->remap[v[0]] = 0;
      p->remap[v[1]] = 0;
      p->remap[v[2]] = 0;

      const double *const p0 = &pos[3 * v[0]];
      const double *const p1 = &pos[3 * v[1]];
      const double *const p2 = &pos[3 * v[2]];

      double e1[3], e2[3];
      MFTLGetMinimumImageVector(dom, e1, p0, p1);
      MFTLGetMinimumImageVector(dom, e2, p0, p2);

      int wrapped = 0;
      for (int k = 0; k < 3; ++k)
      {
        if (e1[k] != p1[k] - p0[k] || e2[k] != p2[k] - p0[k]) wrapped = 1;
      }

      p->wflag[t] = (unsigned char) wrapped;
      if (wrapped)
      {
        db_add_corners(wpos, p0, e1, e2);
        p->n_wrapped++;
      }
    }
  }

  /* Prefix scan: turn the marks into compact indices. */
  int next = 0;
  for (int v = 0; v < nv; ++v)
  {
    if (p->remap[v] >= 0) p->remap[v] = next++;
  }
  p->n_marked = next;
  p->n_points = next + 3 * p->n_wrapped;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. ASSEMBLE: one contiguous byte block per DataArray
 *
 * Every block is described the same way, which is what makes the header
 * generator, the ascii formatter and the MPI-IO offset arithmetic each a single
 * loop over an array of descriptors rather than nine special cases.
 *
 * A block's size is `unit` bytes times either the piece's point count or its
 * cell count -- there is no third possibility -- so ONE exclusive scan of the
 * pair {points, cells} places every block of the shared file. That is why the
 * collective cost of this writer is O(1) messages in the number of arrays.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Element type of one output block. */
typedef enum
{
  SC_F64,  ///< `double`, written as VTK `Float64`.
  SC_I32,  ///< `int`, written as VTK `Int32`.
  SC_U8  ///< `unsigned char`, written as VTK `UInt8` (the cell-type array).
} scalar_kind;

/**
 * @brief One output array, described uniformly so both backends share the code.
 *
 * A block's size is `unit` bytes times EITHER the piece's point count OR its
 * cell count -- there is no third possibility -- so one exclusive scan of the
 * pair {points, cells} places every block of the shared file. That is why the
 * collective cost of this writer is \f$O(1)\f$ messages in the number of arrays.
 */
typedef struct
{
  const char *name;  ///< VTK array name, as it appears in the XML.
  const char *vtk_type;  ///< VTK type string, e.g. "Float64".
  scalar_kind kind;  ///< Element type, for the size and the ASCII formatting.
  int ncomp;  ///< Components per tuple; only Points has 3.
  bool per_cell;  ///< Sized by cells rather than by points.
  bool enabled;  ///< Written at all. Driven by MFTL_VtkOptions.
  size_t unit;  ///< Bytes per point (or per cell).

  void *data;  ///< This rank's payload.
  size_t bytes;  ///< Its size.
  size_t my_offset;  ///< Byte offset of this rank's slice inside the global block.
  size_t global_bytes;  ///< The block's total size over all ranks.
} io_block;

/** @brief Indices into the fixed-size block table. */
enum
{
  BLK_POINTS = 0,  ///< Point coordinates, 3 doubles per point.
  BLK_CONN,  ///< Cell connectivity, 3 ints per triangle.
  BLK_OFFSETS,  ///< Cell offsets, 1 int per triangle.
  BLK_TYPES,  ///< VTK cell type, 1 byte per triangle. Always 5 (VTK_TRIANGLE).
  BLK_VOWNER,  ///< Per-point owning rank.
  BLK_VGID,  ///< Per-point global id.
  BLK_TOWNER,  ///< Per-cell owning rank.
  BLK_BODYID,  ///< Per-cell body (object) id.
  BLK_TGID,  ///< Per-cell global id.
  BLK_COUNT  ///< Number of blocks. Not a block.
};

/**
 * @brief Bytes of one element of the given kind.
 * @param k The element type.
 * @return `sizeof` the corresponding C type.
 */
static size_t scalar_size(scalar_kind k) { return (k == SC_F64) ? sizeof(double) : (k == SC_I32) ? sizeof(int) : 1u; }

/**
 * @brief Points of one object: the referenced vertices, then the private
 *        unwrapped corners.
 *
 * The order is load-bearing -- emit_conn() addresses the private points as
 * `n_marked + 3*w`, so they must follow all the shared ones.
 *
 * @param      obj  The mesh.
 * @param      p    Piece description from analyze_piece().
 * @param      wpos Unwrapped corners collected by analyze_piece().
 * @param[out] out  Destination coordinate array, `3 * p->n_points` doubles.
 */
static void emit_points(const MFTL_Object *obj, const piece *p, const dbuf *wpos, double *out)
{
  const int nv = obj->verts.n;
  const double *const pos = obj->verts.pos;

  if (p->n_marked == nv)
  {
    /* Every vertex is referenced -- the common case for a closed surface whose
     * halo came from a redistribution. The compaction is then the identity, so
     * skip it and move the array at memory bandwidth. */
    memcpy(out, pos, (size_t) (3 * nv) * sizeof(double));
    out += 3 * nv;
  }
  else
  {
    for (int v = 0; v < nv; ++v)
    {
      if (p->remap[v] < 0) continue;
      *out++ = pos[3 * v + 0];
      *out++ = pos[3 * v + 1];
      *out++ = pos[3 * v + 2];
    }
  }

  if (p->n_wrapped > 0) memcpy(out, wpos->data + p->wpos_off, (size_t) (9 * p->n_wrapped) * sizeof(double));
}

/**
 * @brief Connectivity of one object, shifted into the file's point numbering.
 *
 * @param      obj        The mesh.
 * @param      p          Piece description from analyze_piece().
 * @param      point_base Point index this object's block starts at.
 * @param[out] out        Destination connectivity array, `3 * p->n_cells` ints.
 */
static void emit_conn(const MFTL_Object *obj, const piece *p, int point_base, int *out)
{
  const int nt = p->n_cells;
  const int priv0 = point_base + p->n_marked;
  int w = 0;

  if (MFTLObjectHasTopology(obj))
  {
    const int *restrict tv = obj->tris.verts;
    for (int t = 0; t < nt; ++t)
    {
      if (p->wflag[t])
      {
        out[3 * t + 0] = priv0 + 3 * w + 0;
        out[3 * t + 1] = priv0 + 3 * w + 1;
        out[3 * t + 2] = priv0 + 3 * w + 2;
        ++w;
      }
      else
      {
        out[3 * t + 0] = point_base + p->remap[tv[3 * t + 0]];
        out[3 * t + 1] = point_base + p->remap[tv[3 * t + 1]];
        out[3 * t + 2] = point_base + p->remap[tv[3 * t + 2]];
      }
    }
  }
  else
  {
    for (int t = 0; t < nt; ++t)
    {
      if (p->wflag[t])
      {
        out[3 * t + 0] = priv0 + 3 * w + 0;
        out[3 * t + 1] = priv0 + 3 * w + 1;
        out[3 * t + 2] = priv0 + 3 * w + 2;
        ++w;
      }
      else
      {
        int v[3];
        MFTLGetTriangleVertices(obj, t, v);
        out[3 * t + 0] = point_base + p->remap[v[0]];
        out[3 * t + 1] = point_base + p->remap[v[1]];
        out[3 * t + 2] = point_base + p->remap[v[2]];
      }
    }
  }
}

/**
 * @brief A per-point Int32 array: owner rank or global id.
 *
 * The one loop in this writer that reads the AoS records with a stride. It is
 * left that way deliberately -- see the note at the end of this file.
 *
 * @param      obj      The mesh.
 * @param      p        Piece description from analyze_piece().
 * @param      want_gid `true` for the global id, `false` for the owning rank.
 * @param[out] out      Destination int array, `p->n_points` entries.
 */
static void emit_point_ints(const MFTL_Object *obj, const piece *p, bool want_gid, int *out)
{
  const int nv = obj->verts.n;
  const MFTL_Vertex *const rec = obj->verts.rec;

  for (int v = 0; v < nv; ++v)
  {
    if (p->remap[v] < 0) continue;
    *out++ = want_gid ? rec[v].global_id : rec[v].owner_proc;
  }

  /* The private corners inherit the values of the vertices they stand in for. */
  const int nt = p->n_cells;
  for (int t = 0; t < nt; ++t)
  {
    if (!p->wflag[t]) continue;
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    for (int k = 0; k < 3; ++k) *out++ = want_gid ? rec[v[k]].global_id : rec[v[k]].owner_proc;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The XML header
 *
 * Built as a string rather than fprintf'd, because the shared-file backend needs
 * its LENGTH before it can place anything: the appended data begins at exactly
 * `strlen(header)`. Every rank builds it from global counts only, so all ranks
 * produce byte-identical text and agree on that length without communicating it.
 *
 * The `offset` attributes are relative to the first byte after the `_` marker,
 * NOT absolute in the file, so they do not depend on the header's own length.
 * That is what keeps this non-circular and is why no fixed-width padding trick
 * is needed here.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @param h          Header buffer the tag is appended to.
 * @param b          The block being described.
 * @param binary     `true` for the appended-raw form, `false` for inline ASCII.
 * @param rel_offset Byte offset of this block inside the appended section.
 */
static void emit_dataarray_tag(sbuf *h, const io_block *b, bool binary, unsigned long long rel_offset)
{
  sb_addf(h, "        <DataArray type=\"%s\" Name=\"%s\"", b->vtk_type, b->name);
  if (b->ncomp != 1) sb_addf(h, " NumberOfComponents=\"%d\"", b->ncomp);
  if (binary)
    sb_addf(h, " format=\"appended\" offset=\"%llu\"/>\n", rel_offset);
  else
    sb_addf(h, " format=\"ascii\">\n");
}

/**
 * @brief Names of the enabled arrays in a group, for the Scalars= hint.
 *
 * @param blocks The block table.
 * @param ids   Candidate block indices.
 * @param n      Length of @p list.
 * @return The first enabled block index, or -1 if none is.
 */
static const char *first_enabled(const io_block *blocks, const int *ids, int n)
{
  for (int i = 0; i < n; ++i)
  {
    if (blocks[ids[i]].enabled) return blocks[ids[i]].name;
  }
  return MFTL_NULL;
}

/**
 * @brief Writes everything up to and including the `_` of the appended section
 *        (binary), or everything up to the first DataArray (ascii, where the
 *        caller interleaves data and closing tags itself).
 *
 * @param[out] h           Receives the complete XML header text.
 * @param      blocks      The block table.
 * @param      rel_offsets Byte offset of each block inside the appended section.
 * @param      n_points    Points in this piece.
 * @param      n_cells     Cells in this piece.
 * @param      binary      `true` for raw appended data, `false` for inline ASCII.
 */
static void build_header(sbuf *h, const io_block *blocks, const unsigned long long *rel_offsets, long long n_points, long long n_cells, bool binary)
{
  static const int point_arrays[] = {BLK_VOWNER, BLK_VGID};
  static const int cell_arrays[] = {BLK_TOWNER, BLK_BODYID, BLK_TGID};

  sb_addf(h, "<?xml version=\"1.0\"?>\n");
  sb_addf(h, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"%s\" header_type=\"UInt64\">\n", host_byte_order());
  sb_addf(h, "  <UnstructuredGrid>\n");
  sb_addf(h, "    <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n", n_points, n_cells);

  /* In ascii mode the data lives inside the tags, so this function only emits
   * the binary skeleton; the ascii backend calls the tag helper itself. */
  if (!binary) return;

  sb_addf(h, "      <Points>\n");
  emit_dataarray_tag(h, &blocks[BLK_POINTS], true, rel_offsets[BLK_POINTS]);
  sb_addf(h, "      </Points>\n");

  sb_addf(h, "      <Cells>\n");
  emit_dataarray_tag(h, &blocks[BLK_CONN], true, rel_offsets[BLK_CONN]);
  emit_dataarray_tag(h, &blocks[BLK_OFFSETS], true, rel_offsets[BLK_OFFSETS]);
  emit_dataarray_tag(h, &blocks[BLK_TYPES], true, rel_offsets[BLK_TYPES]);
  sb_addf(h, "      </Cells>\n");

  const char *ps = first_enabled(blocks, point_arrays, 2);
  sb_addf(h, "      <PointData%s%s%s>\n", ps ? " Scalars=\"" : "", ps ? ps : "", ps ? "\"" : "");
  for (int i = 0; i < 2; ++i)
  {
    if (blocks[point_arrays[i]].enabled) emit_dataarray_tag(h, &blocks[point_arrays[i]], true, rel_offsets[point_arrays[i]]);
  }
  sb_addf(h, "      </PointData>\n");

  const char *cs = first_enabled(blocks, cell_arrays, 3);
  sb_addf(h, "      <CellData%s%s%s>\n", cs ? " Scalars=\"" : "", cs ? cs : "", cs ? "\"" : "");
  for (int i = 0; i < 3; ++i)
  {
    if (blocks[cell_arrays[i]].enabled) emit_dataarray_tag(h, &blocks[cell_arrays[i]], true, rel_offsets[cell_arrays[i]]);
  }
  sb_addf(h, "      </CellData>\n");

  sb_addf(h, "    </Piece>\n");
  sb_addf(h, "  </UnstructuredGrid>\n");
  sb_addf(h, "  <AppendedData encoding=\"raw\">\n   _");
}

/** @brief The XML tail that closes the appended section and the file. */
static const char *k_footer = "\n  </AppendedData>\n</VTKFile>\n";

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. Backend A: one file per rank (POSIX)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Formats one block as decimal text, one tuple per line, in chunks.
 * @param fp Destination stream.
 * @param b  The block.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a write failure.
 */
static int ascii_emit(FILE *fp, const io_block *b)
{
  const size_t n = b->bytes / scalar_size(b->kind);
  char buf[8192];
  size_t len = 0;

  for (size_t i = 0; i < n; ++i)
  {
    if (len + 64u > sizeof buf)
    {
      if (fwrite(buf, 1, len, fp) != len) return MFTL_ERROR;
      len = 0;
    }
    int w = 0;
    switch (b->kind)
    {
      case SC_F64: w = snprintf(buf + len, sizeof buf - len, "%.10e", ((const double *) b->data)[i]); break;
      case SC_I32: w = snprintf(buf + len, sizeof buf - len, "%d", ((const int *) b->data)[i]); break;
      case SC_U8: w = snprintf(buf + len, sizeof buf - len, "%u", (unsigned) ((const unsigned char *) b->data)[i]); break;
    }
    if (w < 0) return MFTL_ERROR;
    len += (size_t) w;
    buf[len++] = (((i + 1u) % (size_t) b->ncomp) == 0u) ? '\n' : ' ';
  }

  if (len > 0 && fwrite(buf, 1, len, fp) != len) return MFTL_ERROR;
  return MFTL_SUCCESS;
}

/**
 * @brief Writes one block's raw bytes, preceded by its UInt64 length.
 *
 * @param fp Destination stream.
 * @param b  The block.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a write failure.
 */
static int binary_emit(FILE *fp, const io_block *b)
{
  const unsigned long long n = (unsigned long long) b->bytes;
  if (fwrite(&n, sizeof n, 1, fp) != 1) return MFTL_ERROR;
  if (b->bytes > 0 && fwrite(b->data, 1, b->bytes, fp) != b->bytes) return MFTL_ERROR;
  return MFTL_SUCCESS;
}

/**
 * @brief Backend A: this rank writes its own `.vtu`, with no collective at all.
 * @param prefix   File name prefix; the rank number and extension are appended.
 * @param rank     This rank, used in the file name.
 * @param blocks   The block table, fully populated.
 * @param n_points Points in this piece.
 * @param n_cells  Cells in this piece.
 * @param binary   `true` for raw appended data, `false` for inline ASCII.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on any I/O failure.
 */
static int write_piece_file(const char *prefix, int rank, io_block *blocks, long long n_points, long long n_cells, bool binary)
{
  char path[1024];
  const int np = snprintf(path, sizeof path, "%s_%05d.vtu", prefix, rank);
  if (np < 0 || (size_t) np >= sizeof path) return MFTL_ERROR;

  FILE *fp = fopen(path, binary ? "wb" : "w");
  if (!fp)
  {
    fprintf(stderr, "[Rank %d] MFTLWriteVtu: could not open %s.\n", rank, path);
    return MFTL_ERROR;
  }

  char *const iobuf = (char *) malloc(1u << 20);
  if (iobuf) setvbuf(fp, iobuf, _IOFBF, 1u << 20);

  int status = MFTL_SUCCESS;

  unsigned long long rel[BLK_COUNT];
  unsigned long long acc = 0;
  for (int i = 0; i < BLK_COUNT; ++i)
  {
    rel[i] = acc;
    if (blocks[i].enabled) acc += 8u + (unsigned long long) blocks[i].bytes;
  }

  sbuf h;
  sb_init(&h);
  build_header(&h, blocks, rel, n_points, n_cells, binary);

  if (fwrite(h.s, 1, h.len, fp) != h.len) status = MFTL_ERROR;

  if (binary)
  {
    for (int i = 0; i < BLK_COUNT && status == MFTL_SUCCESS; ++i)
    {
      if (blocks[i].enabled) status = binary_emit(fp, &blocks[i]);
    }
    if (status == MFTL_SUCCESS && fwrite(k_footer, 1, strlen(k_footer), fp) != strlen(k_footer)) status = MFTL_ERROR;
  }
  else
  {
    /* ASCII: the numbers live inside the tags, so the structure is interleaved.
     * Same block descriptors, same order, different punctuation. */
    static const int point_arrays[] = {BLK_VOWNER, BLK_VGID};
    static const int cell_arrays[] = {BLK_TOWNER, BLK_BODYID, BLK_TGID};
    sbuf t;
    sb_init(&t);

/* Emits one enabled block as an ASCII <DataArray> with its numbers inline.
 * Documented with @def near the top of this file, because a doc comment inside a
 * function body is attributed to the FUNCTION rather than to the macro. */
#define ASCII_ARRAY(idx)                              \
  do                                                  \
  {                                                   \
    t.len = 0;                                        \
    t.s[0] = '\0';                                    \
    emit_dataarray_tag(&t, &blocks[idx], false, 0);   \
    if (fwrite(t.s, 1, t.len, fp) != t.len) status = MFTL_ERROR; \
    if (status == MFTL_SUCCESS) status = ascii_emit(fp, &blocks[idx]); \
    if (status == MFTL_SUCCESS && fprintf(fp, "        </DataArray>\n") < 0) status = MFTL_ERROR; \
  } while (0)

    fprintf(fp, "      <Points>\n");
    ASCII_ARRAY(BLK_POINTS);
    fprintf(fp, "      </Points>\n      <Cells>\n");
    ASCII_ARRAY(BLK_CONN);
    ASCII_ARRAY(BLK_OFFSETS);
    ASCII_ARRAY(BLK_TYPES);
    fprintf(fp, "      </Cells>\n");

    const char *ps = first_enabled(blocks, point_arrays, 2);
    fprintf(fp, "      <PointData%s%s%s>\n", ps ? " Scalars=\"" : "", ps ? ps : "", ps ? "\"" : "");
    for (int i = 0; i < 2; ++i)
    {
      if (blocks[point_arrays[i]].enabled) ASCII_ARRAY(point_arrays[i]);
    }
    fprintf(fp, "      </PointData>\n");

    const char *cs = first_enabled(blocks, cell_arrays, 3);
    fprintf(fp, "      <CellData%s%s%s>\n", cs ? " Scalars=\"" : "", cs ? cs : "", cs ? "\"" : "");
    for (int i = 0; i < 3; ++i)
    {
      if (blocks[cell_arrays[i]].enabled) ASCII_ARRAY(cell_arrays[i]);
    }
    fprintf(fp, "      </CellData>\n");
    fprintf(fp, "    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
#undef ASCII_ARRAY

    sb_free(&t);
  }

  sb_free(&h);

  if (ferror(fp)) status = MFTL_ERROR;
  if (fclose(fp) != 0) status = MFTL_ERROR;
  free(iobuf);
  return status;
}

/**
 * @brief Rank 0's .pvtu index: the piece list plus the array declarations.
 *
 * @brief Writes the `.pvtu` index that names every piece file.
 * @param prefix File name prefix.
 * @param nproc  Number of piece files to reference.
 * @param blocks The block table, for the array declarations.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an I/O failure.
 */
static int write_pvtu(const char *prefix, int nproc, const io_block *blocks)
{
  static const int point_arrays[] = {BLK_VOWNER, BLK_VGID};
  static const int cell_arrays[] = {BLK_TOWNER, BLK_BODYID, BLK_TGID};

  char path[1024];
  const int np = snprintf(path, sizeof path, "%s.pvtu", prefix);
  if (np < 0 || (size_t) np >= sizeof path) return MFTL_ERROR;

  FILE *fp = fopen(path, "w");
  if (!fp)
  {
    fprintf(stderr, "MFTLWriteVtu: could not open %s.\n", path);
    return MFTL_ERROR;
  }

  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"%s\" header_type=\"UInt64\">\n", host_byte_order());
  fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n");
  fprintf(fp, "    <PPoints>\n      <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>\n    </PPoints>\n");

  const char *ps = first_enabled(blocks, point_arrays, 2);
  fprintf(fp, "    <PPointData%s%s%s>\n", ps ? " Scalars=\"" : "", ps ? ps : "", ps ? "\"" : "");
  for (int i = 0; i < 2; ++i)
  {
    if (blocks[point_arrays[i]].enabled) fprintf(fp, "      <PDataArray type=\"%s\" Name=\"%s\"/>\n", blocks[point_arrays[i]].vtk_type, blocks[point_arrays[i]].name);
  }
  fprintf(fp, "    </PPointData>\n");

  const char *cs = first_enabled(blocks, cell_arrays, 3);
  fprintf(fp, "    <PCellData%s%s%s>\n", cs ? " Scalars=\"" : "", cs ? cs : "", cs ? "\"" : "");
  for (int i = 0; i < 3; ++i)
  {
    if (blocks[cell_arrays[i]].enabled) fprintf(fp, "      <PDataArray type=\"%s\" Name=\"%s\"/>\n", blocks[cell_arrays[i]].vtk_type, blocks[cell_arrays[i]].name);
  }
  fprintf(fp, "    </PCellData>\n");

  /* The piece paths are relative to the .pvtu, so they must not carry the
   * directory component the prefix may have. */
  const char *slash = strrchr(prefix, '/');
  const char *base = slash ? slash + 1 : prefix;
  for (int r = 0; r < nproc; ++r) fprintf(fp, "    <Piece Source=\"%s_%05d.vtu\"/>\n", base, r);

  fprintf(fp, "  </PUnstructuredGrid>\n</VTKFile>\n");

  const int err = ferror(fp);
  return (fclose(fp) != 0 || err) ? MFTL_ERROR : MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Backend B: one shared file (MPI-IO)
 *
 * ── The file layout ────────────────────────────────────────────────────────
 *
 *   [0, H)                          the XML header, written by rank 0
 *   H + rel[b]                      UInt64 length of block b, written by rank 0
 *   H + rel[b] + 8 + my_offset[b]   THIS rank's slice of block b
 *   H + total                       the footer, written by rank 0
 *
 * Every rank knows H because every rank builds the same header text, and knows
 * my_offset because it did the exclusive scan. So there is no second round of
 * communication to distribute offsets -- which is the mistake that makes naive
 * shared-file writers scale badly.
 *
 * ── Why the two reductions are non-blocking and the writes are not ─────────
 *
 * The Exscan (where do my bytes go) and the Allreduce (how big is the file) are
 * INDEPENDENT reductions over the same two integers, so they are issued together
 * as MPI_Iexscan + MPI_Iallreduce and waited on once: two latencies overlap
 * instead of adding. Both are O(log P) messages of 16 bytes and everything after
 * them depends on both, so there is no computation left to hide behind them --
 * the blocks were assembled before the scan precisely because assembling them
 * does not need it.
 *
 * The data writes use MPI_File_write_at_all, and THAT is the entire point of
 * this backend. The independent MPI_File_write_at would produce the same file
 * and would be the classic mistake: each rank's slice of a block is a small,
 * non-contiguous piece of the file, and only the COLLECTIVE call lets the MPI
 * implementation run two-phase I/O -- exchange the pieces among a subset of
 * aggregator ranks so the filesystem sees a few large stripe-aligned writes
 * instead of P small ones. On a parallel filesystem that is the difference
 * between scaling and not.
 *
 * Deadlock freedom: every rank calls write_at_all once per ENABLED block, and
 * the enabled set is derived from the options struct, which is an argument all
 * ranks pass identically. A rank with nothing to contribute calls it with count
 * zero rather than skipping it.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The one round of communication this writer needs: where my slice goes
 *        and how big the file is.
 *
 * Two INDEPENDENT reductions over the same pair {points, cells}: an exclusive
 * scan for this rank's base indices and an all-reduce for the totals the header
 * quotes. Issued together as non-blocking collectives and waited on once, so the
 * two latencies overlap rather than add -- the same reason MFTLEnsembleUpdate
 * issues its three reductions that way.
 *
 * They are not overlapped with COMPUTATION, and that is a deliberate call rather
 * than an omission: both are O(log P) messages of 16 bytes, on the order of tens
 * of microseconds, against an operation that will spend milliseconds in memcpy
 * and tens of milliseconds in the filesystem. Splitting the assembly loop into a
 * scan-independent half and a scan-dependent half to hide that would be real
 * complexity bought for a fraction of a percent.
 *
 * @param      comm      The communicator.
 * @param      my_points This rank's point count.
 * @param      my_cells  This rank's cell count.
 * @param[out] base      Exclusive-scan result: {points, cells} before this rank.
 * @param[out] total     Totals over all ranks.
 */
static void scan_placement(MPI_Comm comm, long long my_points, long long my_cells, long long base[2], long long total[2])
{
  long long mine[2] = {my_points, my_cells};
  base[0] = base[1] = 0;  /* MPI_Exscan leaves rank 0's receive buffer untouched. */
  total[0] = total[1] = 0;

  MPI_Request req[2];
  MPI_Iexscan(mine, base, 2, MPI_LONG_LONG, MPI_SUM, comm, &req[0]);
  MPI_Iallreduce(mine, total, 2, MPI_LONG_LONG, MPI_SUM, comm, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
}

/**
 * @brief Backend B: every rank writes into ONE `.vtu` through MPI-IO.
 * @param comm   The communicator.
 * @param rank   This rank.
 * @param prefix File name prefix.
 * @param blocks The block table, fully populated.
 * @param base   Exclusive-scan offsets from scan_placement().
 * @param total  Global totals from scan_placement().
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on any I/O failure.
 */
static int write_shared_file(MPI_Comm comm, int rank, const char *prefix, io_block *blocks, const long long base[2], const long long total[2])
{
  if (total[0] > 2147483647LL)
  {
    if (rank == 0) fprintf(stderr, "MFTLWriteVtu: %lld points exceeds the Int32 connectivity limit.\n", total[0]);
    return MFTL_ERROR;
  }

  /* Place every block. `unit` is bytes per point or per cell, so this is the
   * only arithmetic the scan result feeds. */
  unsigned long long rel[BLK_COUNT];
  unsigned long long acc = 0;
  for (int i = 0; i < BLK_COUNT; ++i)
  {
    rel[i] = acc;
    if (!blocks[i].enabled) continue;
    const long long n_glob = blocks[i].per_cell ? total[1] : total[0];
    const long long n_base = blocks[i].per_cell ? base[1] : base[0];
    blocks[i].global_bytes = blocks[i].unit * (size_t) n_glob;
    blocks[i].my_offset = blocks[i].unit * (size_t) n_base;
    acc += 8u + (unsigned long long) blocks[i].global_bytes;

    if (blocks[i].bytes > 2147483647u)
    {
      fprintf(stderr, "[Rank %d] MFTLWriteVtu: block '%s' is %llu bytes, above the MPI count limit.\n", rank, blocks[i].name, (unsigned long long) blocks[i].bytes);
      return MFTL_ERROR;
    }
  }

  sbuf h;
  sb_init(&h);
  build_header(&h, blocks, rel, total[0], total[1], true);
  const MPI_Offset header_len = (MPI_Offset) h.len;
  const MPI_Offset footer_at = header_len + (MPI_Offset) acc;

  char path[1024];
  const int np = snprintf(path, sizeof path, "%s.vtu", prefix);
  if (np < 0 || (size_t) np >= sizeof path)
  {
    sb_free(&h);
    return MFTL_ERROR;
  }

  MPI_File fh;
  /* File error handlers default to MPI_ERRORS_RETURN, so the return codes are
   * meaningful without installing one. */
  if (MPI_File_open(comm, path, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
  {
    if (rank == 0) fprintf(stderr, "MFTLWriteVtu: MPI_File_open failed for %s.\n", path);
    sb_free(&h);
    return MFTL_ERROR;
  }

  int status = MFTL_SUCCESS;

  /* Truncate to the exact final size. Without this, overwriting a longer file
   * from a previous, larger step leaves trailing bytes that make the XML
   * unparseable -- a failure that only shows up when the mesh SHRINKS. */
  if (MPI_File_set_size(fh, footer_at + (MPI_Offset) strlen(k_footer)) != MPI_SUCCESS) status = MFTL_ERROR;

  if (rank == 0 && status == MFTL_SUCCESS)
  {
    if (MPI_File_write_at(fh, 0, h.s, (int) h.len, MPI_BYTE, MPI_STATUS_IGNORE) != MPI_SUCCESS) status = MFTL_ERROR;

    /* The per-block length prefixes: eight bytes each, at most nine of them. */
    for (int i = 0; i < BLK_COUNT && status == MFTL_SUCCESS; ++i)
    {
      if (!blocks[i].enabled) continue;
      const unsigned long long n = (unsigned long long) blocks[i].global_bytes;
      if (MPI_File_write_at(fh, header_len + (MPI_Offset) rel[i], &n, 8, MPI_BYTE, MPI_STATUS_IGNORE) != MPI_SUCCESS) status = MFTL_ERROR;
    }

    if (status == MFTL_SUCCESS && MPI_File_write_at(fh, footer_at, (void *) k_footer, (int) strlen(k_footer), MPI_BYTE, MPI_STATUS_IGNORE) != MPI_SUCCESS)
      status = MFTL_ERROR;
  }

  /* Collective, once per enabled block, on every rank -- including ranks whose
   * count is zero. Skipping those would hang the others. */
  for (int i = 0; i < BLK_COUNT; ++i)
  {
    if (!blocks[i].enabled) continue;
    const MPI_Offset at = header_len + (MPI_Offset) rel[i] + 8 + (MPI_Offset) blocks[i].my_offset;
    if (MPI_File_write_at_all(fh, at, blocks[i].data, (int) blocks[i].bytes, MPI_BYTE, MPI_STATUS_IGNORE) != MPI_SUCCESS) status = MFTL_ERROR;
  }

  if (MPI_File_close(&fh) != MPI_SUCCESS) status = MFTL_ERROR;
  sb_free(&h);
  return status;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. MFTLWriteVtu
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix, const MFTL_VtkOptions *options)
{
  if (!ensemble || !filename_prefix) return MFTL_ERROR;

  const MFTL_VtkOptions opt = options ? *options : MFTLVtkDefaults();
  const MFTL_Domain *const dom = &ensemble->domain;
  const int rank = dom->rank;
  const int nproc = dom->nproc;
  const int n_obj = ensemble->num_objects;
  const bool binary = (opt.encoding == MFTL_VTK_BINARY);
  const bool shared = (opt.layout == MFTL_IO_SINGLE_FILE);

  if (shared && !binary)
  {
    if (rank == 0) fprintf(stderr, "MFTLWriteVtu: MFTL_IO_SINGLE_FILE requires MFTL_VTK_BINARY.\n");
    return MFTL_ERROR;
  }

  /* ── ANALYSE ──────────────────────────────────────────────────────────── */

  piece *pieces = (piece *) MFTLCallocOrDie((size_t) (n_obj > 0 ? n_obj : 1), sizeof(piece), "MFTL vtu pieces");
  dbuf wpos;
  db_init(&wpos);

  long long my_points = 0, my_cells = 0;
  for (int o = 0; o < n_obj; ++o)
  {
    const MFTL_Object *obj = ensemble->objects[o];
    const int nv = obj->verts.n;
    const int nt = opt.include_ghosts ? obj->tris.n : obj->tris.n_local;

    pieces[o].remap = (int *) MFTLMallocOrDie((size_t) (nv > 0 ? nv : 1) * sizeof(int), "MFTL vtu point remap");
    pieces[o].wflag = (unsigned char *) MFTLMallocOrDie((size_t) (nt > 0 ? nt : 1), "MFTL vtu wrap flags");

    analyze_piece(obj, opt.include_ghosts, &pieces[o], &wpos);

    my_points += pieces[o].n_points;
    my_cells += pieces[o].n_cells;
  }

  /* ── ASSEMBLE ─────────────────────────────────────────────────────────── */

  io_block blocks[BLK_COUNT];
  memset(blocks, 0, sizeof blocks);

/* Fills in one row of the block table. Documented with @def near the top of this
 * file; see ASCII_ARRAY for why. */
#define DECL(idx, nm, ty, kd, nc, pc, en, un) \
  blocks[idx].name = (nm);                    \
  blocks[idx].vtk_type = (ty);                \
  blocks[idx].kind = (kd);                    \
  blocks[idx].ncomp = (nc);                   \
  blocks[idx].per_cell = (pc);                \
  blocks[idx].enabled = (en);                 \
  blocks[idx].unit = (un)

  DECL(BLK_POINTS, "Points", "Float64", SC_F64, 3, false, true, 3 * sizeof(double));
  DECL(BLK_CONN, "connectivity", "Int32", SC_I32, 1, true, true, 3 * sizeof(int));
  DECL(BLK_OFFSETS, "offsets", "Int32", SC_I32, 1, true, true, sizeof(int));
  DECL(BLK_TYPES, "types", "UInt8", SC_U8, 1, true, true, 1);
  DECL(BLK_VOWNER, "VertexOwner", "Int32", SC_I32, 1, false, opt.write_owner, sizeof(int));
  DECL(BLK_VGID, "VertexGID", "Int32", SC_I32, 1, false, opt.write_global_id, sizeof(int));
  DECL(BLK_TOWNER, "TriangleOwner", "Int32", SC_I32, 1, true, opt.write_owner, sizeof(int));
  DECL(BLK_BODYID, "BodyID", "Int32", SC_I32, 1, true, opt.write_body_id, sizeof(int));
  DECL(BLK_TGID, "TriangleGID", "Int32", SC_I32, 1, true, opt.write_global_id, sizeof(int));
#undef DECL

  for (int i = 0; i < BLK_COUNT; ++i)
  {
    if (!blocks[i].enabled) continue;
    blocks[i].bytes = blocks[i].unit * (size_t) (blocks[i].per_cell ? my_cells : my_points);
    blocks[i].data = MFTLMallocOrDie(blocks[i].bytes > 0 ? blocks[i].bytes : 1, "MFTL vtu data block");
  }

  /* In the shared layout there is ONE Piece spanning every rank, so connectivity
   * and offsets are numbered across the whole file and need this rank's base
   * indices before the arrays can be filled. In the per-rank layout each piece
   * is numbered from zero, so the bases are zero and no communication happens.
   * Same assembly code either way -- only the constant differs. */
  long long base[2] = {0, 0}, total[2] = {0, 0};
  if (shared) scan_placement(dom->comm_world, my_points, my_cells, base, total);

  {
    long long p_off = base[0];  /* running point base of the current object */
    long long c_off = base[1];

    for (int o = 0; o < n_obj; ++o)
    {
      const MFTL_Object *obj = ensemble->objects[o];
      const piece *p = &pieces[o];
      const long long lp = p_off - base[0];  /* this object's slice within my blocks */
      const long long lc = c_off - base[1];

      emit_points(obj, p, &wpos, (double *) blocks[BLK_POINTS].data + 3 * lp);
      emit_conn(obj, p, (int) p_off, (int *) blocks[BLK_CONN].data + 3 * lc);

      int *const off = (int *) blocks[BLK_OFFSETS].data + lc;
      for (int t = 0; t < p->n_cells; ++t) off[t] = (int) (3 * (c_off + t + 1));

      memset((unsigned char *) blocks[BLK_TYPES].data + lc, MFTL_VTK_TRIANGLE, (size_t) p->n_cells);

      if (blocks[BLK_VOWNER].enabled) emit_point_ints(obj, p, false, (int *) blocks[BLK_VOWNER].data + lp);
      if (blocks[BLK_VGID].enabled) emit_point_ints(obj, p, true, (int *) blocks[BLK_VGID].data + lp);

      if (blocks[BLK_TOWNER].enabled)
      {
        int *const d = (int *) blocks[BLK_TOWNER].data + lc;
        for (int t = 0; t < p->n_cells; ++t) d[t] = obj->tris.rec[t].owner_proc;
      }
      if (blocks[BLK_BODYID].enabled)
      {
        int *const d = (int *) blocks[BLK_BODYID].data + lc;
        for (int t = 0; t < p->n_cells; ++t) d[t] = obj->global_id;
      }
      if (blocks[BLK_TGID].enabled)
      {
        int *const d = (int *) blocks[BLK_TGID].data + lc;
        for (int t = 0; t < p->n_cells; ++t) d[t] = obj->tris.rec[t].global_id;
      }

      p_off += p->n_points;
      c_off += p->n_cells;
    }
  }

  /* ── EMIT ─────────────────────────────────────────────────────────────── */

  int status;
  if (shared)
  {
    status = write_shared_file(dom->comm_world, rank, filename_prefix, blocks, base, total);
  }
  else
  {
    status = write_piece_file(filename_prefix, rank, blocks, my_points, my_cells, binary);
    if (rank == 0 && write_pvtu(filename_prefix, nproc, blocks) != MFTL_SUCCESS) status = MFTL_ERROR;
  }

  for (int i = 0; i < BLK_COUNT; ++i) free(blocks[i].data);
  for (int o = 0; o < n_obj; ++o)
  {
    free(pieces[o].remap);
    free(pieces[o].wflag);
  }
  free(pieces);
  db_free(&wpos);

  /* One rank failing to open its piece file must not leave the others believing
   * the dump succeeded -- rank 0's .pvtu already promised that piece exists. One
   * int, and the call is collective in both layouts anyway. */
  int worst = status;
  MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MIN, dom->comm_world);
  return worst;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * A NOTE ON THE ONE AoS READ IN HERE, SINCE IT LOOKS LIKE AN OVERSIGHT
 *
 * emit_point_ints() walks `verts.rec[v].owner_proc` with a 16-byte stride and
 * uses 4 of every 16 bytes it touches -- exactly the access pattern the SoA half
 * of this library exists to avoid, and the obvious "fix" is an SoA owner array.
 *
 * It is deliberately not fixed. The array is optional, written once every N
 * timesteps, and the pass that produces it is followed immediately by a disk
 * write of the same data that is three orders of magnitude slower. Adding
 * `verts.owner[]` would cost 4 bytes per vertex of permanent residency in the
 * pool block, a new field for every growth and permutation site to carry, and a
 * second place for ownership to be recorded -- the exact "SoA island" that
 * mftl_types.h argues against -- to speed up a debug output path by an amount no
 * measurement would resolve. The record stays the single home of ownership.
 * ═══════════════════════════════════════════════════════════════════════════ */
