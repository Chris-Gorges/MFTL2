/**
 * @file vtk_legacy.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_io
 *
 * @brief The single-rank legacy .vtk debug dump.
 *
 * Deliberately the dumbest writer in the library: no filtering, no periodic
 * unwrapping, no communication, no options. It puts the calling rank's raw
 * in-memory state on screen, which is what you want when the question is "what
 * did the remesher actually do to this mesh" rather than "what does the surface
 * look like". MFTLWriteVtu() is the production writer.
 */

#include "mftl_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_topology.h"

/**
 * @brief The legacy POLYDATA grammar, in full:
 *
 *     POINTS <nv> float
 *     <x> <y> <z>                    x nv
 *     POLYGONS <nt> <4*nt>
 *     3 <v0> <v1> <v2>               x nt
 *
 * The second POLYGONS field is the total INTEGER COUNT the block occupies, not
 * the triangle count -- each record contributes four integers, the leading 3
 * included. Getting that wrong produces a file every reader rejects, which is
 * why it is spelled out here rather than left as `4 * n`.
 *
 * The vertex indices are 0-based into the POINTS block just written, and since
 * both blocks come from the same unmodified pool in the same order, the pool
 * index IS the file index. No remapping, no offsets.
 */
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename)
{
  if (!object || !base_filename)
  {
    fprintf(stderr, "MFTLWriteVtk: NULL object or filename.\n");
    return MFTL_ERROR;
  }

  char filename[1024];
  const int n = snprintf(filename, sizeof filename, "%s.vtk", base_filename);
  if (n < 0 || (size_t) n >= sizeof filename)
  {
    fprintf(stderr, "MFTLWriteVtk: filename too long.\n");
    return MFTL_ERROR;
  }

  FILE *fp = fopen(filename, "w");
  if (!fp)
  {
    fprintf(stderr, "MFTLWriteVtk: could not open %s for writing.\n", filename);
    return MFTL_ERROR;
  }

  /* A large stdio buffer is worth more here than any formatting trick: the
   * default is typically 4 KB and this writer issues one fprintf per line, so
   * the syscall count is what dominates. Not static -- that would make the
   * function non-reentrant to save one allocation per file. */
  char *const iobuf = (char *) malloc(1u << 20);
  if (iobuf) setvbuf(fp, iobuf, _IOFBF, 1u << 20);

  const int nv = object->verts.n;
  const int nt = object->tris.n;

  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "MFTL object %d\n", object->global_id);
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET POLYDATA\n");

  fprintf(fp, "POINTS %d double\n", nv);
  const double *const pos = object->verts.pos;
  for (int v = 0; v < nv; ++v) fprintf(fp, "%.10e %.10e %.10e\n", pos[3 * v + 0], pos[3 * v + 1], pos[3 * v + 2]);

  fprintf(fp, "\nPOLYGONS %d %d\n", nt, 4 * nt);
  for (int t = 0; t < nt; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(object, t, v);
    fprintf(fp, "3 %d %d %d\n", v[0], v[1], v[2]);
  }

  /* Ownership is the one thing a raw dump cannot show geometrically, and it is
   * usually the thing you opened the file to check.
   *
   * The geometry arrays go out too, UNVALIDATED. MFTLWriteVtu deliberately
   * refuses to write them, because nothing on the object records whether they
   * are current and a stale-but-plausible production file is a trap. Here the
   * contract is the opposite one -- this writer exists to show the raw state, so
   * showing an area array that a redistribution left undefined is the correct
   * behaviour and is how you SEE that it is undefined. */
  fprintf(fp, "\nPOINT_DATA %d\n", nv);
  fprintf(fp, "SCALARS VertexOwner int 1\nLOOKUP_TABLE default\n");
  for (int v = 0; v < nv; ++v) fprintf(fp, "%d\n", object->verts.rec[v].owner_proc);

  /* On a sphere every vertex normal should point straight out from the centre,
   * which makes a sign error in the winding or in the accumulation visible at a
   * glance in the viewer rather than only in a printed statistic. */
  const double *const vn = object->verts.normal;
  fprintf(fp, "NORMALS vertex_normals double\n");
  for (int v = 0; v < nv; ++v) fprintf(fp, "%.10e %.10e %.10e\n", vn[3 * v + 0], vn[3 * v + 1], vn[3 * v + 2]);

  fprintf(fp, "\nCELL_DATA %d\n", nt);
  fprintf(fp, "SCALARS TriangleOwner int 1\nLOOKUP_TABLE default\n");
  for (int t = 0; t < nt; ++t) fprintf(fp, "%d\n", object->tris.rec[t].owner_proc);

  fprintf(fp, "SCALARS area double 1\nLOOKUP_TABLE default\n");
  for (int t = 0; t < nt; ++t) fprintf(fp, "%.10e\n", object->tris.area[t]);

  const int err = ferror(fp);
  const int closed = fclose(fp);
  free(iobuf); /* only after fclose: stdio uses it until then */
  if (closed != 0 || err)
  {
    fprintf(stderr, "MFTLWriteVtk: write error on %s.\n", filename);
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}
