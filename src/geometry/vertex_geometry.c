/**
 * @file vertex_geometry.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_geometry
 *
 * @brief Per-vertex geometry: the area- and angle-weighted vertex normal passes.
 *
 * ## SCATTER OVER TRIANGLES, NOT GATHER OVER VERTICES
 *
 * A vertex normal is a sum over the vertex's incident faces, so there are two
 * ways to compute it and they are not close:
 *
 *   GATHER   for each vertex, walk its one-ring and accumulate in registers.
 *            One sequential write stream, no zeroing pass, no read-modify-write
 *            -- but the inner loop has a data-dependent trip count and every
 *            step of the fan is a dependent load (he -> twin -> next).
 *
 *   SCATTER  for each triangle, gather three current positions through the
 *            contiguous `tris.verts` stream and add its face normal into its
 *            three vertices. The loop is flat and has no pointer chasing, but
 *            pays a read-modify-write into three indexed vertex slots, plus a
 *            zeroing pass before and a normalisation pass after.
 *
 * The historical frame-backed comparison was measured on the same mesh in the
 * same binary and checked to agree to rounding (the retired benchmark and its
 * results are documented in docs/performance-analysis.md):
 *
 *     scatter over triangles       5.2 - 5.5 ns per vertex, flat in mesh size
 *     gather by fan walk          17.8 - 21.6 ns per vertex, i.e. 3.4x slower
 *
 * and the fan walk is measured WITH its fast path, skipping the rewind on every
 * vertex not flagged MFTL_FLAG_BOUNDARY -- without that it is 8x slower, because
 * MFTLVertexFanStart walks a whole extra lap to discover that a closed fan is
 * closed. What is left after that is the dependent loads: each hop is
 * he -> twin -> next and cannot issue its address until the previous hop has
 * returned, so a valence-6 fan serialises six round trips that the scatter never
 * makes.
 *
 * ### The CSR question, settled
 *
 * src/object/derived.c records the vertex->triangle CSR (`vert_offsets` /
 * `vert_tris`) as deleted for want of a consumer, and asks for it to be
 * reconsidered when one appears. This is that consumer, so it was
 * measured rather than assumed: a CSR gather removes the fan walk's dependent
 * loads entirely and lands at 5.2 - 5.7 ns per vertex -- a dead heat with the
 * scatter, not a win.
 *
 * It loses on everything else. It needs 16 bytes per triangle of permanent
 * state, and a rebuild costing another 5.5 - 5.8 ns per vertex -- as much as the
 * whole normal pass -- on every topology change. On a remeshed surface that is
 * paid every step. THE CSR STAYS DELETED, and now for a measured reason rather
 * than an absent consumer.
 *
 * ## WHY THE SCATTER TARGET STAYS INTERLEAVED
 *
 * `verts.normal` is xyz-interleaved, and for THIS loop that is measurably the
 * right layout rather than an inherited one. The read-modify-write of three
 * contiguous doubles compiles to a 128-bit load, a 2-wide vector add and a
 * 128-bit store for x,y plus one scalar triple for z -- four memory operations
 * per vertex. Against three separate component arrays it would be six, because
 * three separate load-add-stores cannot pair. examples/layout_benchmark.c
 * measures that at 1.20-1.25x in favour of interleaved, consistently and at
 * every size.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_types.h"

/**
 * @brief Zeroes the accumulator and scatters every face normal into it.
 *
 * The inner k-loop is written out rather than left as a loop so that the three
 * read-modify-write chains are visibly independent: they touch three different
 * vertices, and the compiler may not assume that in a loop it has to prove
 * something about.
 *
 * The position and normal arrays share the vertex allocation but occupy
 * disjoint aligned sub-arrays; the topology indices live in the triangle pool.
 *
 * @param obj The mesh. `verts.normal` is accumulated into, having been zeroed
 *            by the caller.
 */
static void scatter_face_normals(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  const int n_verts = obj->verts.n;

  double *restrict vn = obj->verts.normal;

  /* Every vertex is written by this pass, so the whole live prefix is cleared
   * rather than only the owned part -- a ghost that kept last step's value would
   * be a stale number that looks like a fresh one. */
  memset(vn, 0, 3 * (size_t) n_verts * sizeof(double));

  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, n_tris,
    /* e1 x e2 is twice the area times the unit normal, so accumulating it raw
     * IS the area weighting. No sqrt here; the only one in the whole pass is the
     * single normalisation per vertex below. */
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;

    vn[3 * v0 + 0] += nx;
    vn[3 * v0 + 1] += ny;
    vn[3 * v0 + 2] += nz;

    vn[3 * v1 + 0] += nx;
    vn[3 * v1 + 1] += ny;
    vn[3 * v1 + 2] += nz;

    vn[3 * v2 + 0] += nx;
    vn[3 * v2 + 1] += ny;
    vn[3 * v2 + 2] += nz;);
}

/**
 * @brief The angle-weighted scatter: each face's UNIT normal times its interior
 *        angle at each corner.
 *
 * The three corner angles come from one cross product and three dot products:
 * every corner sees the same |e1 x e2| (twice the area), so
 *
 *     alpha_0 = atan2(|e1 x e2|,  e1.e2)
 *     alpha_1 = atan2(|e1 x e2|,  e1.e1 - e1.e2)
 *     alpha_2 = atan2(|e1 x e2|,  e2.e2 - e1.e2)
 *
 * `atan2` rather than `acos` of a normalised dot product, for the reason
 * CLAUDE.md gives about comparing unit vectors: `acos` has an infinite
 * derivative at +-1, so a nearly flat corner would carry an angle error of
 * ~1e-8 rad whatever the inputs; `atan2` is well conditioned everywhere.
 *
 * A face with zero area has no unit normal. It is skipped rather than given the
 * zero vector's angle sum, which is the same answer and avoids three `atan2`.
 *
 * @param obj The mesh. `verts.normal` is accumulated into after being zeroed.
 */
static void scatter_angle_weighted_normals(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  const int n_verts = obj->verts.n;

  double *restrict vn = obj->verts.normal;
  memset(vn, 0, 3 * (size_t) n_verts * sizeof(double));

  MFTL_FOREACH_TRIANGLE_GEOMETRY(obj, n_tris, const double nx = e1y * e2z - e1z * e2y; const double ny = e1z * e2x - e1x * e2z;
                                 const double nz = e1x * e2y - e1y * e2x; const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz); if (!(mag > 0.0)) continue;
                                 const double inv = 1.0 / mag; const double ux = nx * inv, uy = ny * inv, uz = nz * inv;

                                 const double d12 = e1x * e2x + e1y * e2y + e1z * e2z; const double d11 = e1x * e1x + e1y * e1y + e1z * e1z;
                                 const double d22 = e2x * e2x + e2y * e2y + e2z * e2z; const double a0 = atan2(mag, d12);
                                 const double a1 = atan2(mag, d11 - d12); const double a2 = atan2(mag, d22 - d12);

                                 vn[3 * v0 + 0] += a0 * ux; vn[3 * v0 + 1] += a0 * uy; vn[3 * v0 + 2] += a0 * uz;

                                 vn[3 * v1 + 0] += a1 * ux; vn[3 * v1 + 1] += a1 * uy; vn[3 * v1 + 2] += a1 * uz;

                                 vn[3 * v2 + 0] += a2 * ux; vn[3 * v2 + 1] += a2 * uy; vn[3 * v2 + 2] += a2 * uz;);
}

/**
 * @brief Scales every accumulated vector to unit length, in place.
 *
 * A pure streaming pass: 24 bytes in, 24 bytes out, one sqrt per vertex, no
 * indirection and no branch that depends on anything but the vector itself.
 *
 * A vertex with no incident triangles -- an isolated vertex, or a ghost whose
 * faces all live on another rank -- accumulates exactly zero and is left at
 * zero. That is a detectable sentinel: a unit normal never has zero length, so a
 * consumer can tell "not computed" from "computed", which a normalised arbitrary
 * direction would not allow.
 *
 * The test is against ZERO, not against a small constant. It used to be
 * `len_sq > MFTL_VERYSMALL`, the clamp CLAUDE.md records as removed from every
 * other normalisation on 2026-08-15 and missed here: with area weighting the
 * accumulated vector has the magnitude of an AREA, so every vertex of a mesh
 * whose triangles are smaller than about 1e-6 on a side silently came out with
 * a zero normal -- the "not computed" sentinel -- on a perfectly valid mesh.
 *
 * @param obj The mesh. Each `verts.normal` entry is scaled to unit length in place.
 */
static void normalise_vertex_normals(MFTL_Object *obj)
{
  const int n_verts = obj->verts.n;
  double *restrict vn = obj->verts.normal;

  for (int v = 0; v < n_verts; ++v)
  {
    const double x = vn[3 * v + 0];
    const double y = vn[3 * v + 1];
    const double z = vn[3 * v + 2];

    const double len_sq = x * x + y * y + z * z;
    const double inv = (len_sq > 0.0) ? 1.0 / MFTL_SQRT(len_sq) : 0.0;

    vn[3 * v + 0] = x * inv;
    vn[3 * v + 1] = y * inv;
    vn[3 * v + 2] = z * inv;
  }
}

void MFTLComputeVertexNormals(MFTL_Object *obj, MFTL_NormalWeighting weighting)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  MFTL_ASSERT((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u && "MFTLComputeVertexNormals: tris.verts is stale -- MFTLObjectBuildDerived first");

  if (obj->verts.n == 0) return;

  if (weighting == MFTL_NORMAL_WEIGHT_ANGLE)
    scatter_angle_weighted_normals(obj);
  else
    scatter_face_normals(obj);
  normalise_vertex_normals(obj);
}
