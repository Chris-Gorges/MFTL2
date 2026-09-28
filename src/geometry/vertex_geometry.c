/**
 * @file vertex_geometry.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_geometry
 *
 * @brief Area- and angle-weighted vertex normals.
 *
 * A SCATTER over triangles: each triangle gathers its three current positions
 * through `tris.verts` and adds its weighted face normal into its three
 * vertices; one streaming pass then normalises. The alternative GATHER (walk
 * each vertex's fan) serialises dependent loads (he -> twin -> next) and
 * measured 3.4x slower; a vertex-to-triangle CSR gather measured a dead heat
 * with the scatter while costing 16 B/triangle and a rebuild per topology
 * change (docs/analysis.md §1.5). `verts.normal` stays xyz-interleaved: the
 * read-modify-write of three contiguous doubles is four memory operations per
 * vertex against six for split arrays (1.20-1.25x in examples/layout_benchmark.c).
 */

#include "mftl_geometry.h"

#include <math.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_types.h"

/**
 * @brief Scatters every face's unnormalised normal \f$\vect{e}_1\times\vect{e}_2
 *        = 2A\hat{\vect{n}}\f$ into its three vertices (area weighting).
 *
 * @param obj The mesh. `verts.normal` is accumulated into; the caller zeroes it.
 */
static void scatter_face_normals(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  double *restrict vn = obj->verts.normal;

  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, n_tris,
    /* e1 x e2 = 2 A n_hat: accumulating it raw IS area weighting. */
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
 * @brief Scatters each face's UNIT normal times its interior angle at each corner.
 *
 * All three corners share \f$s = \norm{\vect{e}_1\times\vect{e}_2}\f$, so with
 * \f$d_{ij} = \vect{e}_i\cdot\vect{e}_j\f$:
 *
 *     alpha_0 = atan2(s, d12),  alpha_1 = atan2(s, d11 - d12),  alpha_2 = atan2(s, d22 - d12)
 *
 * (\f$\vect{e}_1\cdot(\vect{e}_1-\vect{e}_2)\f$ is the dot product of the two
 * edges at corner 1, and similarly at corner 2). `atan2` is well conditioned
 * for every angle, unlike `acos` near 0 and \f$\pi\f$. Zero-area faces have no
 * unit normal and are skipped.
 *
 * @param obj The mesh. `verts.normal` is accumulated into; the caller zeroes it.
 */
static void scatter_angle_weighted_normals(MFTL_Object *obj)
{
  const int n_tris = obj->tris.n;
  double *restrict vn = obj->verts.normal;

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, n_tris,
    const double nx = e1y * e2z - e1z * e2y;
    const double ny = e1z * e2x - e1x * e2z;
    const double nz = e1x * e2y - e1y * e2x;
    const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
    if (!(mag > 0.0)) continue;

    const double inv = 1.0 / mag;
    const double ux = nx * inv, uy = ny * inv, uz = nz * inv;

    const double d12 = e1x * e2x + e1y * e2y + e1z * e2z;
    const double d11 = e1x * e1x + e1y * e1y + e1z * e1z;
    const double d22 = e2x * e2x + e2y * e2y + e2z * e2z;
    const double a0 = atan2(mag, d12);
    const double a1 = atan2(mag, d11 - d12);
    const double a2 = atan2(mag, d22 - d12);

    vn[3 * v0 + 0] += a0 * ux;
    vn[3 * v0 + 1] += a0 * uy;
    vn[3 * v0 + 2] += a0 * uz;

    vn[3 * v1 + 0] += a1 * ux;
    vn[3 * v1 + 1] += a1 * uy;
    vn[3 * v1 + 2] += a1 * uz;

    vn[3 * v2 + 0] += a2 * ux;
    vn[3 * v2 + 1] += a2 * uy;
    vn[3 * v2 + 2] += a2 * uz;);
  // clang-format on
}

/**
 * @brief Scales every accumulated vector to unit length, in place.
 *
 * A vertex with no incident triangle stays exactly zero -- a detectable "not
 * computed" sentinel. The guard is a test against ZERO, not a small constant:
 * area-weighted sums scale as a length squared, so any absolute threshold
 * would zero the normals of a valid mesh with small enough triangles.
 *
 * @param obj The mesh. Each `verts.normal` entry is normalised in place.
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

  /* Clear every vertex, ghosts included: a ghost keeping last step's value
   * would look fresh. */
  memset(obj->verts.normal, 0, 3 * (size_t) obj->verts.n * sizeof(double));

  if (weighting == MFTL_NORMAL_WEIGHT_ANGLE)
    scatter_angle_weighted_normals(obj);
  else
    scatter_face_normals(obj);
  normalise_vertex_normals(obj);
}
