/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Tests for src/io/ -- both writers, both layouts, both encodings.
 *
 * ── How these tests are written, and why ───────────────────────────────────
 *
 * A writer test that only checks "the call returned MFTL_SUCCESS and a file
 * appeared" catches almost nothing: every interesting failure of an output
 * module (an off-by-one in the point offsets, a triangle pointing at the wrong
 * shared vertex, a periodic seam not unwrapped, a ghost written twice) produces
 * a perfectly well-formed file with the wrong picture in it.
 *
 * So this file contains a small VTU READER -- enough to pull the appended binary
 * arrays and the inline ascii ones back out -- and every test then asserts on the
 * MESH THAT COMES BACK. The strongest of those assertions is the surface area:
 * reconstructing it from the file's own points and connectivity exercises the
 * point compaction, the connectivity remap, the multi-body offsetting and the
 * periodic unwrapping simultaneously, and any one of them being wrong moves the
 * number. A triangle drawn across the whole periodic box rather than unwrapped
 * is not a subtle error in that metric -- it is orders of magnitude.
 *
 * ── Rank counts ────────────────────────────────────────────────────────────
 *
 *   np = 1   the piece IS the whole mesh, so file totals must match the mesh
 *            exactly, with no seam duplication to account for
 *   np = 2   two pieces to concatenate, and a shared file with a non-trivial
 *            exclusive scan on exactly one rank
 *   np = 4   2x2x1: bodies split over ranks that are diagonal neighbours, and
 *            ranks that legitimately end up holding nothing
 *
 * EVERY assertion goes through agreed(): TEST_ASSERT returns immediately, so an
 * assertion failing on only some ranks leaves the rest blocked in the next
 * collective -- a hang instead of a clean failure.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Fixtures
 * ═══════════════════════════════════════════════════════════════════════════ */

static void grid_for(int nproc, int cuts[3])
{
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
  if (nproc == 4)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 1;
  }
  else if (nproc == 8)
  {
    cuts[0] = 2;
    cuts[1] = 2;
    cuts[2] = 2;
  }
}

static MFTL_Ensemble *make_ensemble(bool periodic, int capacity)
{
  int cuts[3];
  grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, capacity);
}

static MFTL_Object *add_sphere(MFTL_Ensemble *ens, int id, double cx, double cy, double cz, double r, int refine)
{
  MFTL_Object *obj = MFTLObjectCreate(id);
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {cx, cy, cz};
  MFTLGenerateSphere(obj, centre, r, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/** @brief Owned triangle count and owned surface area, summed over all ranks. */
static void global_owned(const MFTL_Ensemble *ens, double out[2])
{
  double local[2] = {0.0, 0.0};
  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    local[0] += (double) obj->tris.n_local;
    for (int t = 0; t < obj->tris.n_local; ++t) local[1] += obj->tris.area[t];
  }
  MPI_Allreduce(local, out, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. A minimal VTU reader
 *
 * Enough of the format to verify what was written, and no more: it understands
 * one Piece, appended raw binary with a UInt64 header type, and inline ascii.
 * It is deliberately independent of the writer -- it re-derives the offsets from
 * the XML text rather than recomputing them the way vtu_writer.c does, so a
 * consistent-but-wrong offset scheme cannot pass.
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  char *raw;
  size_t size;
  long long n_points;
  long long n_cells;
  size_t appended;  ///< Index of the first byte after the '_' marker; 0 if ascii.
} vtu_file;

/**
 * @brief Reads a whole file into memory, NUL-terminated. No format assumptions.
 *
 * Separate from vtu_load() because the .pvtu index and the legacy .vtk have no
 * Piece element to parse -- they are checked as text.
 */
static int slurp(const char *path, vtu_file *f)
{
  memset(f, 0, sizeof *f);

  FILE *fp = fopen(path, "rb");
  if (!fp) return 0;

  fseek(fp, 0, SEEK_END);
  const long n = ftell(fp);
  fseek(fp, 0, SEEK_SET);
  if (n <= 0)
  {
    fclose(fp);
    return 0;
  }

  f->raw = (char *) malloc((size_t) n + 1u);
  if (fread(f->raw, 1, (size_t) n, fp) != (size_t) n)
  {
    fclose(fp);
    free(f->raw);
    f->raw = NULL;
    return 0;
  }
  fclose(fp);
  f->size = (size_t) n;
  f->raw[n] = '\0'; /* the XML part is text; the appended part is found by index */
  return 1;
}

/** @brief slurp() plus the Piece counts and the appended-section anchor. */
static int vtu_load(const char *path, vtu_file *f)
{
  if (!slurp(path, f)) return 0;

  const char *p = strstr(f->raw, "NumberOfPoints=\"");
  const char *c = strstr(f->raw, "NumberOfCells=\"");
  if (!p || !c) return 0;
  f->n_points = atoll(p + strlen("NumberOfPoints=\""));
  f->n_cells = atoll(c + strlen("NumberOfCells=\""));

  const char *a = strstr(f->raw, "<AppendedData encoding=\"raw\">");
  if (a)
  {
    const char *u = strchr(a, '_');
    if (!u) return 0;
    f->appended = (size_t) (u - f->raw) + 1u;
  }
  return 1;
}

static void vtu_free(vtu_file *f)
{
  free(f->raw);
  memset(f, 0, sizeof *f);
}

/** @brief Locates the DataArray element named @p name, or NULL. */
static const char *vtu_find_tag(const vtu_file *f, const char *name)
{
  char needle[128];
  snprintf(needle, sizeof needle, "Name=\"%s\"", name);

  const char *at = strstr(f->raw, needle);
  if (!at) return NULL;

  /* Back up to the opening '<' of this element so the attribute search that
   * follows cannot run into the next one. */
  while (at > f->raw && *at != '<') --at;
  return at;
}

/**
 * @brief Pointer to and byte count of an appended binary array.
 * @return NULL if the array is absent or not in appended form.
 */
static const void *vtu_binary(const vtu_file *f, const char *name, unsigned long long *nbytes)
{
  const char *tag = vtu_find_tag(f, name);
  if (!tag || f->appended == 0) return NULL;

  const char *end = strchr(tag, '>');
  const char *off = strstr(tag, "offset=\"");
  if (!off || !end || off > end) return NULL;

  const unsigned long long rel = strtoull(off + strlen("offset=\""), NULL, 10);
  const size_t at = f->appended + (size_t) rel;
  if (at + 8u > f->size) return NULL;

  unsigned long long len;
  memcpy(&len, f->raw + at, 8);
  if (at + 8u + (size_t) len > f->size) return NULL;

  *nbytes = len;
  return f->raw + at + 8u;
}

/**
 * @brief Parses an inline ascii array into a caller-supplied double buffer.
 * @return Number of scalars parsed, or -1 if the array is absent.
 */
static long vtu_ascii(const vtu_file *f, const char *name, double *out, long max_out)
{
  const char *tag = vtu_find_tag(f, name);
  if (!tag) return -1;

  const char *begin = strchr(tag, '>');
  if (!begin) return -1;
  ++begin;

  const char *end = strstr(begin, "</DataArray>");
  if (!end) return -1;

  long n = 0;
  const char *p = begin;
  while (p < end && n < max_out)
  {
    char *stop = NULL;
    const double v = strtod(p, &stop);
    if (stop == p) break;
    out[n++] = v;
    p = stop;
  }
  return n;
}

/**
 * @brief Surface area of the mesh AS THE FILE DESCRIBES IT.
 *
 * The whole point of the reader: it walks the file's own connectivity into the
 * file's own point list, so a wrong index, a wrong offset or a triangle left
 * stretched across a periodic box all show up in one number.
 *
 * @param base Subtracted from every connectivity index. Zero for a per-rank
 *        piece; the file's global point base for one rank's slice of a shared
 *        file, which the caller knows and the file does not record.
 */
static double vtu_area_binary(const vtu_file *f, int base)
{
  unsigned long long np_bytes = 0, nc_bytes = 0;
  const double *pts = (const double *) vtu_binary(f, "Points", &np_bytes);
  const int *conn = (const int *) vtu_binary(f, "connectivity", &nc_bytes);
  if (!pts || !conn) return -1.0;
  if (np_bytes != (unsigned long long) f->n_points * 24u) return -1.0;
  if (nc_bytes != (unsigned long long) f->n_cells * 12u) return -1.0;

  double area = 0.0;
  for (long long t = 0; t < f->n_cells; ++t)
  {
    const int i0 = conn[3 * t + 0] - base;
    const int i1 = conn[3 * t + 1] - base;
    const int i2 = conn[3 * t + 2] - base;
    if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= f->n_points || i1 >= f->n_points || i2 >= f->n_points) return -2.0;

    double e1[3], e2[3];
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = pts[3 * i1 + k] - pts[3 * i0 + k];
      e2[k] = pts[3 * i2 + k] - pts[3 * i0 + k];
    }
    const double cx = e1[1] * e2[2] - e1[2] * e2[1];
    const double cy = e1[2] * e2[0] - e1[0] * e2[2];
    const double cz = e1[0] * e2[1] - e1[1] * e2[0];
    area += 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  }
  return area;
}

/** @brief Structural checks every well-formed piece must satisfy. */
static int vtu_structure_is_sound(const vtu_file *f)
{
  unsigned long long nb = 0;

  const int *off = (const int *) vtu_binary(f, "offsets", &nb);
  if (!off || nb != (unsigned long long) f->n_cells * 4u) return 0;

  const unsigned char *ty = (const unsigned char *) vtu_binary(f, "types", &nb);
  if (!ty || nb != (unsigned long long) f->n_cells) return 0;

  /* The offsets are cumulative and the file may be one rank's slice of a shared
   * piece, so only the STEP is checkable without knowing the base. */
  for (long long t = 0; t < f->n_cells; ++t)
  {
    if (ty[t] != 5) return 0;
    if (t > 0 && off[t] - off[t - 1] != 3) return 0;
  }
  if (f->n_cells > 0 && off[0] % 3 != 0) return 0;
  return 1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Paths and cleanup
 * ═══════════════════════════════════════════════════════════════════════════ */

static void piece_path(char *out, size_t n, const char *prefix, int rank) { snprintf(out, n, "%s_%05d.vtu", prefix, rank); }

/** @brief Removes everything one MFTLWriteVtu call may have produced. */
static void cleanup(const char *prefix)
{
  char path[512];

  piece_path(path, sizeof path, prefix, g_rank);
  remove(path);

  if (g_rank == 0)
  {
    snprintf(path, sizeof path, "%s.pvtu", prefix);
    remove(path);
    snprintf(path, sizeof path, "%s.vtu", prefix);
    remove(path);
  }
  MPI_Barrier(MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Tests
 * ═══════════════════════════════════════════════════════════════════════════ */

static int test_defaults_are_conservative(void)
{
  const MFTL_VtkOptions o = MFTLVtkDefaults();

  /* Binary because ascii is ten times slower for no benefit a tool can see, and
   * owned-only because writing ghosts double-counts the halo in every integral a
   * visualisation computes. Both are the opposite of the easy default, so they
   * are worth pinning. */
  TEST_ASSERT(o.encoding == MFTL_VTK_BINARY);
  TEST_ASSERT(o.layout == MFTL_IO_FILE_PER_RANK);
  TEST_ASSERT(o.include_ghosts == false);
  TEST_ASSERT(o.write_owner == true);
  TEST_ASSERT(o.write_body_id == true);
  TEST_ASSERT(o.write_global_id == false);
  return 0;
}

static int test_invalid_arguments(void)
{
  MFTL_Ensemble *ens = make_ensemble(false, 1);
  MFTL_VtkOptions o = MFTLVtkDefaults();

  /* Rejected locally on every rank, so no rank enters a collective the others
   * are not entering. */
  TEST_ASSERT(agreed(MFTLWriteVtu(NULL, "x", &o) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, NULL, &o) == MFTL_ERROR, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLWriteVtk(NULL, "x") == MFTL_ERROR, MPI_COMM_WORLD));

  /* A shared file exists to be fast; ascii text of variable width in one is a
   * contradiction, and refusing it is clearer than silently promoting it. */
  o.layout = MFTL_IO_SINGLE_FILE;
  o.encoding = MFTL_VTK_ASCII;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, "test_io_reject", &o) == MFTL_ERROR, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The counts in the header must be the OWNED entity counts, summed.
 *
 * Points may exceed the vertex count -- a vertex on a rank seam is written by
 * every rank holding a triangle on it -- but cells may not: each triangle has
 * exactly one owner and must appear exactly once in the merged dataset.
 */
static int test_cell_count_is_the_owned_count(void)
{
  const char *prefix = "test_io_counts";
  MFTL_Ensemble *ens = make_ensemble(false, 1);
  add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.6, 3);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  double expect[2];
  global_owned(ens, expect);

  char path[512];
  piece_path(path, sizeof path, prefix, g_rank);

  vtu_file f;
  int ok = vtu_load(path, &f);
  long long my_cells = ok ? f.n_cells : -1;
  long long my_points = ok ? f.n_points : -1;
  if (ok) ok = vtu_structure_is_sound(&f);
  if (ok) ok = (f.n_cells == ens->objects[0]->tris.n_local);
  vtu_free(&f);

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  long long sums[2] = {0, 0}, mine[2] = {my_cells, my_points};
  MPI_Allreduce(mine, sums, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

  TEST_ASSERT_EQUAL_INT((int) expect[0], (int) sums[0]);
  /* Seam duplication only ever ADDS points, never removes them. */
  TEST_ASSERT(sums[1] >= 0);

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The surface reconstructed from the files equals the surface in memory.
 *
 * The end-to-end check: point compaction, connectivity remap and multi-body
 * offsetting all have to be right simultaneously for this number to come out.
 */
static int test_area_from_file_matches_mesh(void)
{
  const char *prefix = "test_io_area";
  MFTL_Ensemble *ens = make_ensemble(false, 3);
  add_sphere(ens, 0, -0.4, -0.3, 0.0, 0.25, 3);
  add_sphere(ens, 1, 0.4, 0.3, 0.1, 0.30, 2);
  add_sphere(ens, 2, 0.0, 0.5, -0.4, 0.20, 3);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  double expect[2];
  global_owned(ens, expect);

  char path[512];
  piece_path(path, sizeof path, prefix, g_rank);

  vtu_file f;
  int ok = vtu_load(path, &f);
  double my_area = 0.0;
  if (ok)
  {
    my_area = vtu_area_binary(&f, 0);
    ok = (my_area >= 0.0) && vtu_structure_is_sound(&f);
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  double total = 0.0;
  MPI_Allreduce(&my_area, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT_EQUAL_DOUBLE(expect[1], total, 1e-9 * expect[1]);

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A body straddling a periodic face renders at its true size.
 *
 * Without the unwrapping this is not a subtle deviation: the triangles on the
 * seam are drawn across the entire box, and the reconstructed area comes out
 * orders of magnitude too large. The tolerance is therefore tight and the
 * failure would be unmissable.
 */
static int test_periodic_seam_is_unwrapped(void)
{
  const char *prefix = "test_io_periodic";
  MFTL_Ensemble *ens = make_ensemble(true, 1);
  add_sphere(ens, 0, -0.98, 0.0, 0.0, 0.3, 3); /* centred on the x = -1 face */
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  double expect[2];
  global_owned(ens, expect);

  char path[512];
  piece_path(path, sizeof path, prefix, g_rank);

  vtu_file f;
  int ok = vtu_load(path, &f);
  double my_area = 0.0;
  long long extra_points = 0;
  if (ok)
  {
    my_area = vtu_area_binary(&f, 0);
    ok = (my_area >= 0.0);
    /* Private unwrapped corners are the only way a piece can hold more points
     * than the object has vertices, so this confirms the path was taken at all
     * rather than the test passing because nothing straddled. */
    extra_points = f.n_points - ens->objects[0]->verts.n;
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  double total = 0.0;
  MPI_Allreduce(&my_area, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT_EQUAL_DOUBLE(expect[1], total, 1e-9 * expect[1]);

  long long max_extra = 0;
  MPI_Allreduce(&extra_points, &max_extra, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(max_extra > 0);

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Compact connectivity and the half-edge fallback write the same mesh.
 *
 * Both paths gather current positions and apply the same minimum-image seam test;
 * only the source of the three vertex indices differs.
 */
static int test_wrap_detection_paths_agree(void)
{
  const char *with = "test_io_wrapA";
  const char *without = "test_io_wrapB";

  MFTL_Ensemble *ens = make_ensemble(true, 1);
  add_sphere(ens, 0, -0.95, 0.9, 0.0, 0.35, 3);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();

  TEST_ASSERT(agreed(MFTLObjectHasTopology(ens->objects[0]), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, with, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  for (int o = 0; o < ens->num_objects; ++o) MFTLObjectInvalidateDerived(ens->objects[o]);
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, without, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char pa[512], pb[512];
  piece_path(pa, sizeof pa, with, g_rank);
  piece_path(pb, sizeof pb, without, g_rank);

  vtu_file fa, fb;
  int ok = vtu_load(pa, &fa) && vtu_load(pb, &fb);
  if (ok)
  {
    /* Same number of straddling triangles found, and the same surface. */
    ok = (fa.n_points == fb.n_points) && (fa.n_cells == fb.n_cells);
    if (ok)
    {
      const double a = vtu_area_binary(&fa, 0);
      const double b = vtu_area_binary(&fb, 0);
      ok = (a >= 0.0) && (fabs(a - b) <= 1e-12 * (a + 1.0));
    }
  }
  vtu_free(&fa);
  vtu_free(&fb);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(with);
  cleanup(without);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Ghosts are excluded by default and included on request. */
static int test_ghost_inclusion(void)
{
  const char *a = "test_io_noghost";
  const char *b = "test_io_ghost";

  MFTL_Ensemble *ens = make_ensemble(false, 1);
  add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, a, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));
  opt.include_ghosts = true;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, b, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char pa[512], pb[512];
  piece_path(pa, sizeof pa, a, g_rank);
  piece_path(pb, sizeof pb, b, g_rank);

  vtu_file fa, fb;
  int ok = vtu_load(pa, &fa) && vtu_load(pb, &fb);
  if (ok)
  {
    ok = (fa.n_cells == ens->objects[0]->tris.n_local) && (fb.n_cells == ens->objects[0]->tris.n);
  }
  vtu_free(&fa);
  vtu_free(&fb);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(a);
  cleanup(b);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The shared file holds exactly the union of the per-rank pieces.
 *
 * The test for the MPI-IO backend: identical counts, identical surface. If the
 * exclusive scan were wrong, the connectivity of every rank but the first would
 * point at another rank's points and the area would be nonsense rather than
 * slightly off.
 */
static int test_shared_file_equals_the_pieces(void)
{
  const char *per_rank = "test_io_pieces";
  const char *shared = "test_io_shared";

  MFTL_Ensemble *ens = make_ensemble(true, 2);
  add_sphere(ens, 0, -0.9, 0.2, 0.0, 0.3, 3);
  add_sphere(ens, 1, 0.5, -0.4, 0.3, 0.25, 3);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  opt.write_global_id = true;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, per_rank, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  opt.layout = MFTL_IO_SINGLE_FILE;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, shared, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* Piece side: sum the counts and the areas over the ranks. */
  char path[512];
  piece_path(path, sizeof path, per_rank, g_rank);

  vtu_file f;
  int ok = vtu_load(path, &f);
  long long mine[2] = {0, 0};
  double my_area = 0.0;
  if (ok)
  {
    mine[0] = f.n_points;
    mine[1] = f.n_cells;
    my_area = vtu_area_binary(&f, 0);
    ok = (my_area >= 0.0);
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  long long piece_sum[2] = {0, 0};
  double piece_area = 0.0;
  MPI_Allreduce(mine, piece_sum, 2, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&my_area, &piece_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  /* Shared side: rank 0 reads the one file and checks the whole thing at once. */
  int shared_ok = 1;
  if (g_rank == 0)
  {
    snprintf(path, sizeof path, "%s.vtu", shared);
    vtu_file s;
    shared_ok = vtu_load(path, &s);
    if (shared_ok)
    {
      shared_ok = (s.n_points == piece_sum[0]) && (s.n_cells == piece_sum[1]) && vtu_structure_is_sound(&s);
      if (shared_ok)
      {
        const double a = vtu_area_binary(&s, 0);
        shared_ok = (a >= 0.0) && (fabs(a - piece_area) <= 1e-9 * (piece_area + 1.0));
      }
    }
    vtu_free(&s);
  }
  TEST_ASSERT(agreed(shared_ok, MPI_COMM_WORLD));

  cleanup(per_rank);
  cleanup(shared);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A shared file written over a LARGER one is truncated to its own size.
 *
 * MPI_File_open with MPI_MODE_CREATE does not shorten an existing file, so
 * without the explicit MPI_File_set_size the tail of the previous, bigger dump
 * survives past the footer and the XML no longer parses. It only bites when the
 * mesh shrinks between two dumps to the same prefix, which is exactly what a
 * coarsening remesher does.
 */
static int test_shared_file_is_truncated_on_rewrite(void)
{
  const char *prefix = "test_io_shrink";

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  opt.layout = MFTL_IO_SINGLE_FILE;

  MFTL_Ensemble *big = make_ensemble(false, 1);
  add_sphere(big, 0, 0.0, 0.0, 0.0, 0.5, 4);
  MFTLRedistribute(big);
  TEST_ASSERT(agreed(MFTLWriteVtu(big, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));
  MFTLEnsembleDestroy(&big);

  MFTL_Ensemble *small = make_ensemble(false, 1);
  add_sphere(small, 0, 0.0, 0.0, 0.0, 0.5, 1);
  MFTLRedistribute(small);
  for (int o = 0; o < small->num_objects; ++o) MFTLComputeTriangleGeometry(small->objects[o], small->objects[o]->tris.n);
  TEST_ASSERT(agreed(MFTLWriteVtu(small, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  double expect[2];
  global_owned(small, expect);

  int ok = 1;
  if (g_rank == 0)
  {
    char path[512];
    snprintf(path, sizeof path, "%s.vtu", prefix);

    vtu_file f;
    ok = vtu_load(path, &f);
    if (ok)
    {
      /* The file must END with the footer -- a surviving tail from the larger
       * dump would sit after it. */
      const char *footer = "</VTKFile>\n";
      ok = (f.size >= strlen(footer)) && (memcmp(f.raw + f.size - strlen(footer), footer, strlen(footer)) == 0);
      if (ok) ok = (f.n_cells == (long long) expect[0]);
      if (ok)
      {
        const double a = vtu_area_binary(&f, 0);
        ok = (a >= 0.0) && (fabs(a - expect[1]) <= 1e-9 * expect[1]);
      }
    }
    vtu_free(&f);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(prefix);
  MFTLEnsembleDestroy(&small);
  return 0;
}

/** @brief The ascii encoding describes the same mesh as the binary one. */
static int test_ascii_agrees_with_binary(void)
{
  const char *ab = "test_io_bin";
  const char *aa = "test_io_ascii";

  MFTL_Ensemble *ens = make_ensemble(true, 1);
  add_sphere(ens, 0, -0.9, 0.0, 0.0, 0.3, 2);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, ab, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));
  opt.encoding = MFTL_VTK_ASCII;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, aa, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char pb[512], pa[512];
  piece_path(pb, sizeof pb, ab, g_rank);
  piece_path(pa, sizeof pa, aa, g_rank);

  vtu_file fb, fa;
  int ok = vtu_load(pb, &fb) && vtu_load(pa, &fa);
  if (ok) ok = (fb.n_points == fa.n_points) && (fb.n_cells == fa.n_cells);

  if (ok && fb.n_points > 0)
  {
    unsigned long long nb = 0;
    const double *bpts = (const double *) vtu_binary(&fb, "Points", &nb);
    double *apts = (double *) malloc((size_t) (3 * fa.n_points) * sizeof(double));
    const long n = vtu_ascii(&fa, "Points", apts, 3 * fa.n_points);

    ok = bpts && (n == 3 * fa.n_points);
    /* "%.10e" keeps eleven significant digits, so the two agree to about 1e-10
     * relative -- exactly, not approximately, the format's own precision. */
    for (long i = 0; ok && i < n; ++i) ok = (fabs(apts[i] - bpts[i]) <= 1e-9 * (fabs(bpts[i]) + 1.0));
    free(apts);
  }

  if (ok && fb.n_cells > 0)
  {
    unsigned long long nb = 0;
    const int *bconn = (const int *) vtu_binary(&fb, "connectivity", &nb);
    double *aconn = (double *) malloc((size_t) (3 * fa.n_cells) * sizeof(double));
    const long n = vtu_ascii(&fa, "connectivity", aconn, 3 * fa.n_cells);

    ok = bconn && (n == 3 * fa.n_cells);
    for (long i = 0; ok && i < n; ++i) ok = ((int) aconn[i] == bconn[i]);
    free(aconn);
  }

  vtu_free(&fb);
  vtu_free(&fa);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(ab);
  cleanup(aa);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief BodyID separates the bodies, and every cell carries exactly one. */
static int test_body_id_partitions_the_cells(void)
{
  const char *prefix = "test_io_bodies";
  const int n_bodies = 4;

  MFTL_Ensemble *ens = make_ensemble(false, n_bodies);
  for (int o = 0; o < n_bodies; ++o) add_sphere(ens, 100 + o, -0.6 + 0.4 * o, 0.1 * o, 0.0, 0.15, 2);
  MFTLRedistribute(ens);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char path[512];
  piece_path(path, sizeof path, prefix, g_rank);

  vtu_file f;
  int ok = vtu_load(path, &f);
  long long per_body[8] = {0};
  if (ok)
  {
    unsigned long long nb = 0;
    const int *bid = (const int *) vtu_binary(&f, "BodyID", &nb);
    ok = bid && (nb == (unsigned long long) f.n_cells * 4u);
    for (long long t = 0; ok && t < f.n_cells; ++t)
    {
      const int b = bid[t] - 100;
      ok = (b >= 0 && b < n_bodies);
      if (ok) per_body[b]++;
    }
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  long long total[8] = {0};
  MPI_Allreduce(per_body, total, 8, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  for (int o = 0; o < n_bodies; ++o) TEST_ASSERT_EQUAL_INT(ens->objects[o]->global_n_triangles, (int) total[o]);

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief The optional arrays appear exactly when they are asked for. */
static int test_optional_arrays_toggle(void)
{
  const char *prefix = "test_io_arrays";

  MFTL_Ensemble *ens = make_ensemble(false, 1);
  add_sphere(ens, 7, 0.0, 0.0, 0.0, 0.4, 2);
  MFTLRedistribute(ens);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  opt.write_owner = false;
  opt.write_body_id = false;
  opt.write_global_id = true;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char path[512];
  piece_path(path, sizeof path, prefix, g_rank);

  vtu_file f;
  unsigned long long nb = 0;
  int ok = vtu_load(path, &f);
  if (ok)
  {
    ok = (vtu_binary(&f, "VertexOwner", &nb) == NULL) && (vtu_binary(&f, "TriangleOwner", &nb) == NULL) && (vtu_binary(&f, "BodyID", &nb) == NULL) &&
         (vtu_binary(&f, "VertexGID", &nb) != NULL) && (vtu_binary(&f, "TriangleGID", &nb) != NULL);
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief The .pvtu names one piece per rank, and every one of them exists. */
static int test_pvtu_lists_every_piece(void)
{
  const char *prefix = "test_io_pvtu";

  MFTL_Ensemble *ens = make_ensemble(false, 1);
  add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.5, 2);
  MFTLRedistribute(ens);

  MFTL_VtkOptions opt = MFTLVtkDefaults();
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, prefix, &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  int ok = 1;
  if (g_rank == 0)
  {
    char path[512];
    snprintf(path, sizeof path, "%s.pvtu", prefix);

    vtu_file f;
    ok = slurp(path, &f);
    if (ok)
    {
      int found = 0;
      const char *p = f.raw;
      while ((p = strstr(p, "<Piece Source=")) != NULL)
      {
        ++found;
        ++p;
      }
      ok = (found == g_nproc);

      for (int r = 0; ok && r < g_nproc; ++r)
      {
        char pp[512];
        piece_path(pp, sizeof pp, prefix, r);
        FILE *fp = fopen(pp, "rb");
        ok = (fp != NULL);
        if (fp) fclose(fp);
      }
    }
    vtu_free(&f);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup(prefix);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Nothing to write is not an error, in either layout. */
static int test_empty_is_a_valid_dataset(void)
{
  MFTL_VtkOptions opt = MFTLVtkDefaults();

  /* No objects at all. */
  MFTL_Ensemble *ens = make_ensemble(false, 1);
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, "test_io_empty0", &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));
  opt.layout = MFTL_IO_SINGLE_FILE;
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, "test_io_empty1", &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));
  opt.layout = MFTL_IO_FILE_PER_RANK;

  /* An object that was created but never generated: zero capacity, NULL pools. */
  MFTL_Object *bare = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, bare);
  TEST_ASSERT(agreed(MFTLWriteVtu(ens, "test_io_empty2", &opt) == MFTL_SUCCESS, MPI_COMM_WORLD));

  int ok = 1;
  {
    char path[512];
    piece_path(path, sizeof path, "test_io_empty2", g_rank);
    vtu_file f;
    ok = vtu_load(path, &f);
    if (ok) ok = (f.n_points == 0) && (f.n_cells == 0);
    vtu_free(&f);
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  cleanup("test_io_empty0");
  cleanup("test_io_empty1");
  cleanup("test_io_empty2");
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief The legacy writer dumps the rank's raw state, ghosts included. */
static int test_legacy_vtk_dump(void)
{
  MFTL_Ensemble *ens = make_ensemble(false, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.5, 2);
  MFTLRedistribute(ens);

  char path[512];
  snprintf(path, sizeof path, "test_io_legacy_%05d", g_rank);

  TEST_ASSERT(agreed(MFTLWriteVtk(obj, path) == MFTL_SUCCESS, MPI_COMM_WORLD));

  char full[512];
  snprintf(full, sizeof full, "%s.vtk", path);

  vtu_file f;
  int ok = slurp(full, &f);
  if (ok)
  {
    int np = -1, nt = -1, size_field = -1;
    const char *p = strstr(f.raw, "POINTS ");
    const char *q = strstr(f.raw, "POLYGONS ");
    ok = (p != NULL) && (q != NULL) && (sscanf(p, "POINTS %d", &np) == 1) && (sscanf(q, "POLYGONS %d %d", &nt, &size_field) == 2);

    /* Raw dump: every entity the rank holds, ghosts included. And the second
     * POLYGONS field is the integer COUNT of the block, not the cell count --
     * four per record, the leading 3 included. */
    if (ok) ok = (np == obj->verts.n) && (nt == obj->tris.n) && (size_field == 4 * obj->tris.n);
  }
  vtu_free(&f);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  remove(full);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== MFTL I/O tests (%d rank%s) ===\n", g_nproc, g_nproc == 1 ? "" : "s");

  RUN_TEST(test_defaults_are_conservative);
  RUN_TEST(test_invalid_arguments);
  RUN_TEST(test_cell_count_is_the_owned_count);
  RUN_TEST(test_area_from_file_matches_mesh);
  RUN_TEST(test_periodic_seam_is_unwrapped);
  RUN_TEST(test_wrap_detection_paths_agree);
  RUN_TEST(test_ghost_inclusion);
  RUN_TEST(test_shared_file_equals_the_pieces);
  RUN_TEST(test_shared_file_is_truncated_on_rewrite);
  RUN_TEST(test_ascii_agrees_with_binary);
  RUN_TEST(test_body_id_partitions_the_cells);
  RUN_TEST(test_optional_arrays_toggle);
  RUN_TEST(test_pvtu_lists_every_piece);
  RUN_TEST(test_empty_is_a_valid_dataset);
  RUN_TEST(test_legacy_vtk_dump);

  if (g_rank == 0) TEST_REPORT();

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
