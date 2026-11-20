/**
 * @file vertex_geometry.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_geometry
 *
 * @brief Vertex normals, one rule for the whole pool and for one vertex: an
 *        area- or angle-weighted sum of the face normals, or the normal of a
 *        one-ring paraboloid fit (#MFTL_NormalWeighting).
 *
 * The whole pool (MFTLComputeVertexNormals()) is a SCATTER over triangles: each
 * triangle gathers its three current positions through `tris.verts` and adds
 * its weighted face normal into its three vertices; one streaming pass then
 * normalises. The alternative GATHER (walk each vertex's fan) serialises
 * dependent loads (he -> twin -> next) and measured 3.4x slower; a
 * vertex-to-triangle CSR gather measured a dead heat with the scatter while
 * costing 16 B/triangle and a rebuild per topology change (docs/analysis.md
 * §1.5). `verts.normal` stays xyz-interleaved: the read-modify-write of three
 * contiguous doubles is four memory operations per vertex against six for
 * split arrays (1.20-1.25x in examples/layout_benchmark.c).
 *
 * One vertex (MFTLComputeVertexNormalAt()) is that gather, for a caller that
 * edits the mesh between two normals and would find a whole-pool pass stale
 * after its first edit (the remesher's placement fit). Both take a corner's
 * angle from corner_angle() and the fit from fit_normal(), so each rule is
 * written once; the two differ only in the order of the sum.
 *
 * The FIT normal gathers in both: a least-squares system is per vertex by
 * nature. Its frame is the area-weighted normal, the cheapest: which weighting
 * the frame comes from does not change the fitted normal (three digits,
 * docs/analysis.md §16.8), and the fit is second-order accurate where every
 * weighting is first-order (src/geometry/README.md, "Vertex normals").
 */

#include "mftl_geometry.h"

#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_types.h"

/** Longest fan a walk follows: the remesher's ring bound, since a vertex under edit can exceed a finished mesh's valence. */
#define FAN_WALK_MAX (2 * MFTL_MAX_VALENCE)

/**
 * @brief The interior angle of a corner whose two edges have cross product of
 *        length @p mag and dot product @p dot: `atan2(mag, dot)`.
 *
 * Both arguments carry the same factor |e1||e2|, so the edges need no
 * normalising, and `atan2` is well conditioned for every angle, unlike `acos`
 * of the normalised dot product near 0 and \f$\pi\f$.
 *
 * @param mag \f$\norm{\vect{e}_1\times\vect{e}_2}\f$, positive.
 * @param dot \f$\vect{e}_1\cdot\vect{e}_2\f$.
 * @return The angle in \f$(0, \pi)\f$.
 */
static inline double corner_angle(double mag, double dot) { return atan2(mag, dot); }

/**
 * @brief Scatters every face's weighted normal into its three corners.
 *
 * AREA adds the unnormalised normal \f$\vect{e}_1\times\vect{e}_2 =
 * 2A\hat{\vect{n}}\f$ itself. ANGLE adds \f$\alpha\hat{\vect{n}}\f$; all three
 * corners share \f$s = \norm{\vect{e}_1\times\vect{e}_2}\f$, and with
 * \f$d_{ij} = \vect{e}_i\cdot\vect{e}_j\f$ the dot products of the two edges
 * at corners 0, 1, 2 are \f$d_{12}\f$, \f$d_{11} - d_{12}\f$ and
 * \f$d_{22} - d_{12}\f$. A zero-area face has no unit normal and is skipped
 * (under AREA it adds zero).
 *
 * @param obj       The mesh. `verts.normal` is accumulated into; the caller zeroes it.
 * @param weighting #MFTL_NORMAL_WEIGHT_AREA or #MFTL_NORMAL_WEIGHT_ANGLE.
 */
static void scatter_face_normals(MFTL_Object *obj, MFTL_NormalWeighting weighting)
{
  const int n_tris = obj->tris.n;
  double *restrict vn = obj->verts.normal;
  const bool angle = (weighting == MFTL_NORMAL_WEIGHT_ANGLE);

  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, n_tris,
    double nx = e1y * e2z - e1z * e2y;
    double ny = e1z * e2x - e1x * e2z;
    double nz = e1x * e2y - e1y * e2x;
    double a0 = 1.0, a1 = 1.0, a2 = 1.0;
    if (angle)
    {
      const double mag = MFTL_SQRT(nx * nx + ny * ny + nz * nz);
      if (!(mag > 0.0)) continue;
      const double inv = 1.0 / mag;
      nx *= inv, ny *= inv, nz *= inv;
      const double d12 = e1x * e2x + e1y * e2y + e1z * e2z;
      const double d11 = e1x * e1x + e1y * e1y + e1z * e1z;
      const double d22 = e2x * e2x + e2y * e2y + e2z * e2z;
      a0 = corner_angle(mag, d12);
      a1 = corner_angle(mag, d11 - d12);
      a2 = corner_angle(mag, d22 - d12);
    }

    vn[3 * v0 + 0] += a0 * nx;
    vn[3 * v0 + 1] += a0 * ny;
    vn[3 * v0 + 2] += a0 * nz;

    vn[3 * v1 + 0] += a1 * nx;
    vn[3 * v1 + 1] += a1 * ny;
    vn[3 * v1 + 2] += a1 * nz;

    vn[3 * v2 + 0] += a2 * nx;
    vn[3 * v2 + 1] += a2 * ny;
    vn[3 * v2 + 2] += a2 * nz;);
  // clang-format on
}

/**
 * @brief Adds what the face (v, a, b) contributes to the normal of its corner
 *        v, by the rule of scatter_face_normals().
 *
 * @param obj   The mesh.
 * @param p     Position of v.
 * @param a     The corner after v.
 * @param b     The corner after a.
 * @param angle Angle weights (else area weights).
 * @param n     The sum, accumulated into.
 */
static void add_face(const MFTL_Object *obj, const double p[3], int a, int b, bool angle, double n[3])
{
  double d1[3], d2[3];
  const double *const pa = MFTLVertexPos(obj, a), *const pb = MFTLVertexPos(obj, b);
  if (obj->domain != MFTL_NULL)
  {
    MFTLGetMinimumImageVector(obj->domain, d1, p, pa);
    MFTLGetMinimumImageVector(obj->domain, d2, p, pb);
  }
  else
    for (int k = 0; k < 3; ++k) d1[k] = pa[k] - p[k], d2[k] = pb[k] - p[k];

  double c[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]};
  double s = 1.0;
  if (angle)
  {
    const double mag = MFTL_SQRT(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]);
    if (!(mag > 0.0)) return;
    const double inv = 1.0 / mag;
    c[0] *= inv, c[1] *= inv, c[2] *= inv;
    s = corner_angle(mag, d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]);
  }
  n[0] += s * c[0];
  n[1] += s * c[1];
  n[2] += s * c[2];
}

/**
 * @brief Scales a vector to unit length in place; the zero vector stays zero.
 *
 * The guard is a test against ZERO, not a small constant, so that no length
 * unit can zero the normals of a valid mesh.
 *
 * @param n The vector.
 */
static inline void normalise(double n[3])
{
  const double len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  const double inv = (len_sq > 0.0) ? 1.0 / MFTL_SQRT(len_sq) : 0.0;
  n[0] *= inv;
  n[1] *= inv;
  n[2] *= inv;
}

/** Relative rank tolerance of the fit (the placement's value, src/remeshing/remesh_fit.c). */
#define FIT_NORMAL_TAU 1e-3

/** A fitted normal must stay within acos(0.25), about 75 degrees, of the frame normal it was fitted in. */
#define FIT_NORMAL_MIN_COS 0.25

/**
 * @brief Replaces the unit frame normal @p n of vertex @p v by its fitted
 *        normal: \f$z = c_0 + c_1x + c_2y + \dots\f$ through the vertex and its
 *        one-ring POSITIONS in the frame of @p n, then \f$\vect{N} \propto
 *        \vect{n} - c_1\vect{t}_1 - c_2\vect{t}_2\f$. Leaves @p n unchanged
 *        where the fit cannot be trusted (see #MFTL_NORMAL_WEIGHT_FIT).
 *
 * Reads no other vertex's normal, so a whole pool can be fitted in place.
 *
 * @param obj The mesh.
 * @param v   The vertex.
 * @param n   On entry the unit frame normal, on return the fitted one.
 */
static void fit_normal(const MFTL_Object *obj, int v, double n[3])
{
  if (!(n[0] * n[0] + n[1] * n[1] + n[2] * n[2] > 0.0)) return;

  int ring[MFTL_MAX_VALENCE + 1];
  const int nr = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_MAX_VALENCE + 1, MFTL_FALSE);
  if (nr < 3) return;

  /* Tangent frame from the axis n is least aligned with: depends on n only. */
  int a = 0;
  if (fabs(n[1]) < fabs(n[a])) a = 1;
  if (fabs(n[2]) < fabs(n[a])) a = 2;
  double t1[3] = {-n[a] * n[0], -n[a] * n[1], -n[a] * n[2]};
  t1[a] += 1.0;
  normalise(t1);
  const double t2[3] = {n[1] * t1[2] - n[2] * t1[1], n[2] * t1[0] - n[0] * t1[2], n[0] * t1[1] - n[1] * t1[0]};

  double pts[3 * (MFTL_MAX_VALENCE + 2)], w[MFTL_MAX_VALENCE + 2];
  pts[0] = pts[1] = pts[2] = 0.0;
  w[0] = 1.0;
  const double *const xv = MFTLVertexPos(obj, v);
  double h2 = 0.0;
  for (int i = 0; i < nr; ++i)
  {
    double d[3];
    const double *const xj = MFTLVertexPos(obj, ring[i]);
    if (obj->domain != MFTL_NULL)
      MFTLGetMinimumImageVector(obj->domain, d, xv, xj);
    else
      d[0] = xj[0] - xv[0], d[1] = xj[1] - xv[1], d[2] = xj[2] - xv[2];
    h2 += d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    pts[3 * (i + 1) + 0] = d[0] * t1[0] + d[1] * t1[1] + d[2] * t1[2];
    pts[3 * (i + 1) + 1] = d[0] * t2[0] + d[1] * t2[1] + d[2] * t2[2];
    pts[3 * (i + 1) + 2] = d[0] * n[0] + d[1] * n[1] + d[2] * n[2];
    w[i + 1] = 1.0;
  }
  if (!(h2 > 0.0)) return;
  const double hinv = 1.0 / MFTL_SQRT(h2 / (double) nr);
  for (int i = 3; i < 3 * (nr + 1); ++i) pts[i] *= hinv;

  double c[6];
  unsigned char kept[6];
  MFTLFitHeightParaboloid(nr + 1, pts, MFTL_NULL, MFTL_NULL, w, FIT_NORMAL_TAU, c, kept);
  if (!kept[1] || !kept[2]) return;  // the slope is not determined

  /* Slopes are dimensionless: the scaling of the coordinates does not reach them. */
  double m[3] = {n[0] - c[1] * t1[0] - c[2] * t2[0], n[1] - c[1] * t1[1] - c[2] * t2[1], n[2] - c[1] * t1[2] - c[2] * t2[2]};
  normalise(m);
  if (!(m[0] * n[0] + m[1] * n[1] + m[2] * n[2] > FIT_NORMAL_MIN_COS)) return;  // also refuses a zero m
  n[0] = m[0];
  n[1] = m[1];
  n[2] = m[2];
}

void MFTLComputeVertexNormals(MFTL_Object *obj, MFTL_NormalWeighting weighting)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  MFTL_ASSERT((obj->derived & MFTL_DERIVED_TOPOLOGY) != 0u && "MFTLComputeVertexNormals: tris.verts is stale -- MFTLObjectBuildDerived first");

  if (obj->verts.n == 0) return;

  /* Clear every vertex, ghosts included: a ghost keeping last step's value
   * would look fresh. */
  double *const vn = obj->verts.normal;
  memset(vn, 0, 3 * (size_t) obj->verts.n * sizeof(double));

  /* FIT starts from the area weighting: its frame. */
  scatter_face_normals(obj, (weighting == MFTL_NORMAL_WEIGHT_ANGLE) ? MFTL_NORMAL_WEIGHT_ANGLE : MFTL_NORMAL_WEIGHT_AREA);
  for (int v = 0; v < obj->verts.n; ++v) normalise(&vn[3 * v]);
  if (weighting == MFTL_NORMAL_WEIGHT_FIT)
    for (int v = 0; v < obj->verts.n; ++v) fit_normal(obj, v, &vn[3 * v]);
}

bool MFTLComputeVertexNormalAt(const MFTL_Object *obj, int v, MFTL_NormalWeighting weighting, double n[3])
{
  MFTL_ASSERT(obj != MFTL_NULL && v >= 0 && v < obj->verts.n);
  n[0] = n[1] = n[2] = 0.0;
  const int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return MFTL_FALSE;

  const bool angle = (weighting == MFTL_NORMAL_WEIGHT_ANGLE);
  const double *const p = MFTLVertexPos(obj, v);
  bool closed = MFTL_FALSE;

  /* Forward: the outgoing v->w1 of the face (v, w1, w2), then the outgoing
   * v->w2 = twin(prev). */
  int h = start;
  for (int guard = 0; guard < FAN_WALK_MAX; ++guard)
  {
    add_face(obj, p, MFTLHeTarget(obj, h), MFTLHeTarget(obj, MFTLHeNextIdx(h)), angle, n);
    h = obj->tris.he[MFTLHePrevIdx(h)].twin_idx;
    if (h == start)
    {
      closed = MFTL_TRUE;
      break;
    }
    if (h == MFTL_ERROR) break;
  }

  /* Backward from the start when the fan is open: the face before is the one
   * across the start's twin. */
  if (!closed)
  {
    int g = obj->tris.he[start].twin_idx;
    for (int guard = 0; g != MFTL_ERROR && guard < FAN_WALK_MAX; ++guard)
    {
      const int out = MFTLHeNextIdx(g);  // v -> x in the face of g
      add_face(obj, p, MFTLHeTarget(obj, out), MFTLHeTarget(obj, MFTLHeNextIdx(out)), angle, n);
      g = obj->tris.he[out].twin_idx;
    }
  }

  normalise(n);
  if (weighting == MFTL_NORMAL_WEIGHT_FIT) fit_normal(obj, v, n);
  return closed;
}
