/**
 * @file vtu_writer.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_io
 *
 * @brief The parallel VTK XML writer: selection, periodic unwrapping, block
 *        assembly, and the two output backends (POSIX per rank, MPI-IO shared).
 *
 * Three separable phases: ANALYSE (which cells and points, which triangles
 * straddle a seam; per rank, no MPI), ASSEMBLE (one contiguous byte block per
 * DataArray; per rank) and EMIT (the only phase that differs between encodings
 * and layouts). The time of an ASCII writer is spent in `printf`; the binary
 * path's assembly is a memcpy.
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

/**
 * @brief Growable text buffer for the XML header. In the shared layout every
 *        rank builds the identical header and uses its LENGTH to place the
 *        appended data, so it must exist in memory before anything is written.
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
 * Cells: owned triangles (`[0, tris.n_local)`), or all with `include_ghosts`.
 * Points: every vertex a written triangle references (possibly a ghost), found
 * by one marking pass and a prefix scan; unreferenced vertices are dropped. A
 * seam vertex is thus written by every rank with a triangle on it -- cracked in
 * the point topology, seamless in the picture.
 *
 * Periodic unwrapping: vertices are stored wrapped, so a triangle straddling a
 * periodic face would be drawn across the whole domain. It gets three PRIVATE
 * points at its unwrapped corners p0, p0 + e1, p0 + e2 (minimum-image edges),
 * which may lie outside the box. A triangle wraps iff some minimum-image edge
 * differs from its raw difference, i.e. iff a periodic correction was applied:
 * `d - L*nearbyint(d/L)` returns `d` bit-exactly when the rounding is 0, and
 * differs by about L otherwise. The test is exact, with no tolerance.
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
 * Two loop shapes, because periodicity is loop-invariant: (a) non-periodic --
 * mark connectivity only; (b) periodic -- also compute each triangle's
 * minimum-image edges once and apply the exact wrap test.
 *
 * @param      obj            The mesh.
 * @param      include_ghosts Count ghost triangles too.
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

  const bool compact = MFTLObjectHasTopology(obj);

  if (!periodic)
  {
    /* (a) Marking only. With the derived array this is a flat pass over 12 bytes
     * per triangle; without it, three loads with a 16-byte stride. */
    if (compact)
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
  else
  {
    /* (b) Compute each triangle's seam-consistent edges once during analysis. */
    const double *const pos = obj->verts.pos;
    const int *const tv = compact ? obj->tris.verts : MFTL_NULL;
    for (int t = 0; t < nt; ++t)
    {
      int v[3];
      if (compact)
      {
        v[0] = tv[3 * t + 0];
        v[1] = tv[3 * t + 1];
        v[2] = tv[3 * t + 2];
      }
      else
      {
        MFTLGetTriangleVertices(obj, t, v);
      }
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
 * 3. ASSEMBLE: one contiguous byte block per DataArray, all described by the
 * same descriptor (io_block), so the header, the ASCII formatter and the MPI-IO
 * offsets are each one loop over the table.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Element type of one output block. */
typedef enum
{
  SC_F64,  ///< `double`, written as VTK `Float64`.
  SC_I32,  ///< `int`, written as VTK `Int32`.
  SC_U8  ///< `unsigned char`, written as VTK `UInt8` (the cell-type array).
} scalar_kind;

/**
 * @brief One output array. Its size is `unit` bytes times EITHER the piece's
 *        point count OR its cell count, so one exclusive scan of the pair
 *        {points, cells} places every block of the shared file.
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
 *        unwrapped corners (emit_conn() addresses them as `n_marked + 3*w`, so
 *        the order is load-bearing).
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
    /* Every vertex referenced (the common case): the compaction is the identity. */
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
 * @brief A per-point Int32 array: owner rank or global id. Private corners
 *        inherit the values of the vertices they stand in for.
 *
 * Reads the AoS records with a 16-byte stride; an SoA owner array would cost
 * 4 B per vertex permanently for an output path dominated by the disk write.
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
 * 4. The XML header, built from GLOBAL counts only, so every rank produces
 * byte-identical text. The appended-data `offset` attributes are relative to the
 * byte after the `_` marker, so they do not depend on the header's own length.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Appends one `<DataArray>` tag (self-closing for appended data, open for
 *        inline ASCII).
 *
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
 * @brief Name of the first enabled array of a group, for the `Scalars=` hint.
 *
 * @param blocks The block table.
 * @param ids    Candidate block indices.
 * @param n      Length of @p ids.
 * @return The array name, or #MFTL_NULL if none is enabled.
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
 * @brief Writes rank 0's `.pvtu` index: array declarations and one piece file
 *        per rank (paths relative to the index).
 *
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
 *   [0, H)                          XML header                       (rank 0)
 *   H + rel[b]                      UInt64 length of block b         (rank 0)
 *   H + rel[b] + 8 + my_offset[b]   this rank's slice of block b     (every rank)
 *   H + total                       footer                           (rank 0)
 *
 * Every rank knows H (identical header text) and its offsets (one exclusive
 * scan), so no offsets are distributed. The slices are written with
 * MPI_File_write_at_all, once per enabled block on every rank (count zero if
 * nothing to contribute), so the MPI library can aggregate the many small
 * non-contiguous slices into a few large stripe-aligned writes (two-phase
 * collective buffering) -- the independent MPI_File_write_at would not.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The one round of communication of the shared layout: an exclusive scan
 *        of {points, cells} (where this rank's slices go) and an all-reduce (the
 *        totals the header quotes), posted together as non-blocking collectives.
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
  total[0] = total[1] = 0;

  MPI_Request req[2];
  MPI_Iexscan(mine, base, 2, MPI_LONG_LONG, MPI_SUM, comm, &req[0]);
  MPI_Iallreduce(mine, total, 2, MPI_LONG_LONG, MPI_SUM, comm, &req[1]);
  MPI_Waitall(2, req, MPI_STATUSES_IGNORE);

  /* MPI leaves rank 0's exscan result UNDEFINED; its prefix is zero. */
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0) base[0] = base[1] = 0;
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
  /* Int32 limits on the GLOBAL totals (identical everywhere, so returning is a
   * collective decision): point ids, and offsets = 3 x cumulative cells. */
  if (total[0] > 2147483647LL || 3 * total[1] > 2147483647LL)
  {
    if (rank == 0) fprintf(stderr, "MFTLWriteVtu: %lld points / %lld cells exceed the Int32 connectivity and offset limits.\n", total[0], total[1]);
    return MFTL_ERROR;
  }

  /* Place every block. `unit` is bytes per point or per cell, so this is the
   * only arithmetic the scan result feeds. */
  unsigned long long rel[BLK_COUNT];
  unsigned long long acc = 0;
  int local_bad = 0;
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
      local_bad = 1;
    }
  }

  char path[1024];
  const int np = snprintf(path, sizeof path, "%s.vtu", prefix);
  if (np < 0 || (size_t) np >= sizeof path) local_bad = 1;

  /* Local checks, agreed before the collective MPI_File_open. */
  int any_bad = 0;
  MPI_Allreduce(&local_bad, &any_bad, 1, MPI_INT, MPI_MAX, comm);
  if (any_bad) return MFTL_ERROR;

  sbuf h;
  sb_init(&h);
  build_header(&h, blocks, rel, total[0], total[1], true);
  const MPI_Offset header_len = (MPI_Offset) h.len;
  const MPI_Offset footer_at = header_len + (MPI_Offset) acc;

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

  /* Truncate: overwriting a longer file would leave trailing bytes. */
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

  /* Collective, once per enabled block, on every rank (zero counts included). */
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
  const int n_obj = ensemble->n_local_objects;
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
    const MFTL_Object *obj = ensemble->local_objects[o];
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

  /* Shared layout: one Piece, so numbering needs this rank's base indices.
   * Per-rank layout: bases are zero. */
  long long base[2] = {0, 0}, total[2] = {0, 0};
  if (shared) scan_placement(dom->comm_world, my_points, my_cells, base, total);

  {
    long long p_off = base[0];  /* running point base of the current object */
    long long c_off = base[1];

    for (int o = 0; o < n_obj; ++o)
    {
      const MFTL_Object *obj = ensemble->local_objects[o];
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

  /* Agree on the outcome (the .pvtu promised every piece). */
  int worst = status;
  MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MIN, dom->comm_world);
  return worst;
}
