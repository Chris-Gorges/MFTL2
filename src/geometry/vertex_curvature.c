/**
 * @file vertex_curvature.c
 * @author Christian Gorges
 * @date 2026-10-23
 * @ingroup mftl_geometry
 *
 * @brief Discrete curvature of a triangulated surface: Meyer et al. (2003)
 *        and a one-ring position-plus-normal fit.
 *
 * One entry point, MFTLComputeVertexCurvature(): the SIGNED curvature
 * kappa = k1 + k2 (2/R on a sphere with outward normals) and on request the
 * Gaussian curvature, the principal curvatures and the principal direction,
 * by Meyer's operators (COTAN) or by a height paraboloid fitted to the
 * one-ring (FIT). The remesher's curvature sizing reads max(|k1|, |k2|) off
 * the COTAN record.
 *
 * Every estimate is a ONE-RING quantity, which is what makes it exact on an
 * owned vertex of a distributed mesh with no communication: the redistribution
 * halo is vertex-complete, so every triangle incident to an owned vertex is
 * present locally. A ghost vertex receives a PARTIAL sum -- plausible and
 * wrong, exactly like a ghost's vertex normal. The fit additionally reads the
 * neighbours' normals, which must be the owners' on ghosts.
 *
 * The sums are a scatter over triangles (MFTL_FOREACH_TRIANGLE_GEOMETRY), for
 * the same reason MFTLComputeVertexNormals() is one: a per-vertex fan walk
 * serialises dependent loads. Only the least-squares parts (Meyer's tensor fit
 * and the paraboloid) gather a vertex's ring, because a least-squares system
 * is per vertex by nature.
 *
 * Nothing here compares a dimensional quantity against a constant: degenerate
 * triangles are those whose cross product is exactly zero, and the fit works
 * in units of the ring's own edge length. Scaling a mesh by a power of two
 * therefore scales every output by the matching power, bit for bit.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#include "geometry_internal.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_types.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The cotangent scatter
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Per-vertex accumulator record: one cache line. */
enum
{
  ACC_LAP = 0,  // [3] sum (cot alpha + cot beta)(x_i - x_j) = 2 grad A
  ACC_NRM = 3,  // [3] the AREA vertex normal, unnormalised: MFTLComputeVertexNormals()'s sum, to the bit
  ACC_ANGLE = 6,  // sum of the corner angles
  ACC_AREA = 7,  // the MIXED area
  ACC_STRIDE = 8
};

/**
 * @brief One pass over every triangle the rank holds: the cotangent
 *        Laplacian, the area-weighted normal, the angle sum and the mixed area
 *        of every vertex.
 *
 * The normal and the corner angles are the vertex normals' rules
 * (MFTLGeomFaceNormalTerm(), MFTLGeomCornerAngle(), geometry_internal.h),
 * fused into this pass because a second scatter over the triangles for the
 * normal alone costs a third of the kappa-only pass.
 *
 * @param obj    The mesh; current `tris.verts`.
 * @param angles Whether to accumulate the angle sums (three `atan2` per
 *               triangle, most of the pass): only the Gaussian curvature reads
 *               them.
 * @param acc    `[ACC_STRIDE * verts.n]`, zeroed by the caller, accumulated into.
 */
static void scatter_cotan(const MFTL_Object *obj, bool angles, double *restrict acc)
{
  // clang-format off
  MFTL_FOREACH_TRIANGLE_GEOMETRY(
      obj, obj->tris.n,
    /* Corner 0 at p0, edges e1 = p1 - p0 and e2 = p2 - p0; c = p2 - p1 is the
     * edge opposite corner 0. Every corner shares |e1 x e2| = 2A, so a
     * cotangent is a dot product over it. */
    const double e1[3] = {e1x, e1y, e1z};
    const double e2[3] = {e2x, e2y, e2z};
    const double cx = e2x - e1x, cy = e2y - e1y, cz = e2z - e1z;
    double nrm[3];
    MFTLVectorCrossVector(nrm, e1, e2);
    const double mag = MFTLVectorNorm(nrm);
    if (!(mag > 0.0)) continue;  // a degenerate face has no angles to give

    const double l1 = e1x * e1x + e1y * e1y + e1z * e1z;  // |p1 - p0|^2
    const double l2 = e2x * e2x + e2y * e2y + e2z * e2z;  // |p2 - p0|^2
    const double lc = cx * cx + cy * cy + cz * cz;  // |p2 - p1|^2

    /* Dot products of the two edges AT each corner. */
    double dc[3];
    MFTLGeomCornerDots(e1, e2, dc);
    const double d0 = dc[0], d1 = dc[1], d2 = dc[2];

    const double inv = 1.0 / mag;
    const double cot0 = d0 * inv, cot1 = d1 * inv, cot2 = d2 * inv;

    double *const a0 = &acc[ACC_STRIDE * v0];
    double *const a1 = &acc[ACC_STRIDE * v1];
    double *const a2 = &acc[ACC_STRIDE * v2];

    /* (cot alpha + cot beta)(x_i - x_j), one triangle's share per edge:
     * edge v1-v2 carries cot0, v0-v2 cot1, v0-v1 cot2. */
    a0[ACC_LAP + 0] += -cot1 * e2x - cot2 * e1x;
    a0[ACC_LAP + 1] += -cot1 * e2y - cot2 * e1y;
    a0[ACC_LAP + 2] += -cot1 * e2z - cot2 * e1z;
    a1[ACC_LAP + 0] += -cot0 * cx + cot2 * e1x;
    a1[ACC_LAP + 1] += -cot0 * cy + cot2 * e1y;
    a1[ACC_LAP + 2] += -cot0 * cz + cot2 * e1z;
    a2[ACC_LAP + 0] += cot0 * cx + cot1 * e2x;
    a2[ACC_LAP + 1] += cot0 * cy + cot1 * e2y;
    a2[ACC_LAP + 2] += cot0 * cz + cot1 * e2z;

    /* The AREA vertex normal's own term and scatter. */
    double w[3];
    MFTLGeomFaceNormalTerm(MFTL_FALSE, 3, e1, e2, nrm, w);
    MFTLGeomAddFaceNormal(acc + ACC_NRM, ACC_STRIDE, v0, v1, v2, nrm, w);

    if (angles)
    {
      a0[ACC_ANGLE] += MFTLGeomCornerAngle(mag, d0);
      a1[ACC_ANGLE] += MFTLGeomCornerAngle(mag, d1);
      a2[ACC_ANGLE] += MFTLGeomCornerAngle(mag, d2);
    }

    /* The MIXED area (Meyer et al., Fig. 4): the Voronoi share where the
     * triangle is not obtuse, else half the area to the obtuse corner and a
     * quarter to each of the others. The Voronoi shares of one triangle sum to
     * its area, so the mixed areas tile the surface either way. */
    const double A = 0.5 * mag;
    if (d0 >= 0.0 && d1 >= 0.0 && d2 >= 0.0)
    {
      a0[ACC_AREA] += 0.125 * (l2 * cot1 + l1 * cot2);
      a1[ACC_AREA] += 0.125 * (l1 * cot2 + lc * cot0);
      a2[ACC_AREA] += 0.125 * (lc * cot0 + l2 * cot1);
    }
    else
    {
      a0[ACC_AREA] += (d0 < 0.0) ? 0.5 * A : 0.25 * A;
      a1[ACC_AREA] += (d1 < 0.0) ? 0.5 * A : 0.25 * A;
      a2[ACC_AREA] += (d2 < 0.0) ? 0.5 * A : 0.25 * A;
    });
  // clang-format on
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Signed curvature: shared pieces
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Relative rank tolerance of both least-squares fits (the placement's value, src/remeshing/remesh_fit.c). */
#define CURVATURE_FIT_TAU 1e-3

/**
 * Below this cosine between a neighbour's normal and the frame normal (about
 * 75 degrees) the neighbour's slope rows are left out: dz/dx = -n_x/n_z
 * diverges as the normal turns into the tangent plane. Dimensionless.
 */
#define CURVATURE_FIT_MIN_COS 0.25

/**
 * @brief Writes the record from kappa and the Gaussian curvature: the
 *        principal curvatures k1,2 = kappa/2 +- sqrt(max(kappa^2/4 - K, 0))
 *        (Meyer et al., eqs. 10-11; a negative discriminant is rounding at an
 *        umbilic and is clamped). The direction slots are left to the caller.
 */
static void write_scalars(double *rec, int ncomp, double kappa, double gauss)
{
  rec[MFTL_CURVATURE_KAPPA] = kappa;
  if (ncomp < 2) return;
  rec[MFTL_CURVATURE_GAUSS] = gauss;
  if (ncomp < 4) return;
  const double h = 0.5 * kappa;
  double disc = h * h - gauss;
  if (!(disc > 0.0)) disc = 0.0;
  const double r = MFTL_SQRT(disc);
  rec[MFTL_CURVATURE_K1] = h + r;
  rec[MFTL_CURVATURE_K2] = h - r;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * COTAN: Meyer et al. (2003)
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Meyer's principal direction (section 5.3): the curvature tensor
 *        whose normal curvature along each one-ring edge best matches the
 *        edge's own estimate \f$\kappa^N_{ij} = 2(\vect{x}_i -
 *        \vect{x}_j)\cdot\hat{\vect{n}}/|\vect{x}_i - \vect{x}_j|^2\f$.
 *
 * With the trace fixed to kappa (their constraint a + c = 2 kappa_H) the
 * tensor is \f$B = \tfrac{\kappa}{2} I + \begin{pmatrix} p & q \\ q & -p
 * \end{pmatrix}\f$ in the tangent frame, and an edge at angle theta predicts
 * \f$\kappa/2 + p\cos2\theta + q\sin2\theta\f$: a two-unknown linear least
 * squares problem, solved by QR. Each triangle of the fan contributes one row
 * per edge it has at the vertex, weighted by its half of Meyer's weight
 * \f$\tfrac18(\cot\alpha + \cot\beta)|\vect{x}_i - \vect{x}_j|^2\f$ (a negative
 * half, behind an obtuse angle, is dropped); the rows of one edge from its two
 * triangles add up to Meyer's single row. The larger eigenvalue's eigenvector
 * is at angle \f$\tfrac12\operatorname{atan2}(q, p)\f$.
 *
 * @param obj   The mesh.
 * @param v     The vertex.
 * @param n     Its unit normal.
 * @param kappa Its signed curvature.
 * @param dir   Out: unit tangent along the larger principal curvature.
 */
static void cotan_direction(const MFTL_Object *obj, int v, const double n[3], double kappa, double dir[3])
{
  double t1[3], t2[3];
  MFTLGeomTangentFrame(n, t1, t2);
  dir[0] = t1[0];
  dir[1] = t1[1];
  dir[2] = t1[2];

  int ring[MFTL_MAX_VALENCE];
  const int nt = MFTLGetVertexOneRingHalfedges(obj, v, ring, MFTL_MAX_VALENCE, MFTL_FALSE, MFTL_NULL);

  enum
  {
    MAX_ROWS = 2 * MFTL_MAX_VALENCE
  };
  double cs[MAX_ROWS][2], rhs[MAX_ROWS];
  int m = 0;
  const MFTL_Domain *const dom = obj->domain;
  const double *const xv = MFTLVertexPos(obj, v);

  for (int i = 0; i < nt; ++i)
  {
    /* The face of the outgoing half-edge v -> j is (v, j, k). */
    const int j = MFTLHeTarget(obj, ring[i]), k = MFTLHeTarget(obj, MFTLHeNextIdx(ring[i]));

    double e[2][3];
    MFTLGeomCornerEdges(dom, xv, MFTLVertexPos(obj, j), MFTLVertexPos(obj, k), e[0], e[1]);
    double cr[3];
    MFTLVectorCrossVector(cr, e[0], e[1]);
    const double mag = MFTLVectorNorm(cr);
    if (!(mag > 0.0)) continue;

    const double ljj = MFTLVectorNormSq(e[0]);
    const double lkk = MFTLVectorNormSq(e[1]);
    /* Corner 1 is j and corner 2 is k; the angle at k faces the edge v-j. Each
     * cotangent from its corner's own edges (MFTLGeomCornerDots()), never as
     * lkk - ljk, which cancels at a sliver's near-right corners, and over
     * |e_j x e_k| as the cotangent scatter takes it: by its reciprocal. */
    double dc[3];
    MFTLGeomCornerDots(e[0], e[1], dc);
    const double inv = 1.0 / mag;
    const double cot_j = dc[1] * inv;
    const double cot_k = dc[2] * inv;
    const double w[2] = {cot_k * ljj, cot_j * lkk};

    for (int s = 0; s < 2; ++s)
    {
      if (!(w[s] > 0.0)) continue;
      const double *const d = e[s];
      const double l2 = (s == 0) ? ljj : lkk;
      const double dn = MFTLVectorInProd(d, n);
      const double kn = -2.0 * dn / l2;  // normal curvature along the edge
      const double a = MFTLVectorInProd(d, t1);
      const double b = MFTLVectorInProd(d, t2);
      const double ab = a * a + b * b;
      if (!(ab > 0.0)) continue;
      const double sw = MFTL_SQRT(w[s]);
      cs[m][0] = sw * (a * a - b * b) / ab;
      cs[m][1] = sw * (2.0 * a * b) / ab;
      rhs[m] = sw * (kn - 0.5 * kappa);
      ++m;
    }
  }
  if (m < 2) return;

  double A[2 * MAX_ROWS], x[2];
  for (int r = 0; r < m; ++r)
  {
    A[r] = cs[r][0];
    A[m + r] = cs[r][1];
  }
  MFTLLeastSquaresQRCP(m, 2, 2, A, rhs, CURVATURE_FIT_TAU, x, MFTL_NULL);
  const double phi = 0.5 * atan2(x[1], x[0]);
  const double cp = cos(phi), sp = sin(phi);
  for (int k = 0; k < 3; ++k) dir[k] = cp * t1[k] + sp * t2[k];
}

/**
 * @brief The COTAN record of vertex @p v from its scatter accumulator.
 *
 * \f$\vect{K} = \nabla A / A_\text{mixed}\f$ (the accumulated Laplacian is
 * twice the area gradient), \f$\kappa = \vect{K}\cdot\hat{\vect{N}}\f$ with the
 * area-weighted normal, \f$\kappa_G = (2\pi - \sum\theta)/A_\text{mixed}\f$.
 */
static void cotan_record(const MFTL_Object *obj, int v, const double *r, int ncomp, double *rec)
{
  const double a = r[ACC_AREA];
  const double *const N = &r[ACC_NRM];
  const double nm = MFTLVectorNorm(N);
  if (!(a > 0.0) || !(nm > 0.0))
  {
    for (int c = 0; c < ncomp; ++c) rec[c] = 0.0;
    return;
  }
  const double inv = 1.0 / nm;
  const double n[3] = {N[0] * inv, N[1] * inv, N[2] * inv};
  const double *const L = &r[ACC_LAP];
  const double kappa = 0.5 * MFTLVectorInProd(L, n) / a;
  const double gauss = (MFTL_TWO_PI - r[ACC_ANGLE]) / a;
  write_scalars(rec, ncomp, kappa, gauss);
  if (ncomp >= MFTL_CURVATURE_COMPONENTS) cotan_direction(obj, v, n, kappa, &rec[MFTL_CURVATURE_DIR1]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FIT: a height paraboloid through the one-ring positions and normals
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The FIT record of vertex @p v.
 *
 * In the frame (t1, t2, n) of the vertex normal, every sample -- the vertex
 * itself at the origin and each one-ring neighbour -- gives its height and,
 * from its normal, its two slopes \f$\partial_x z = -n_x/n_z\f$,
 * \f$\partial_y z = -n_y/n_z\f$; coordinates are divided by the ring's RMS
 * edge length so that heights and slopes are both O(1) and one weight serves
 * both kinds of row (MFTLFitHeightParaboloid()). With \f$z = c_0 + c_1x + c_2y
 * + c_3x^2 + c_4xy + c_5y^2\f$ the first fundamental form at the vertex is
 * \f$I = \begin{pmatrix}1+c_1^2 & c_1c_2\\ c_1c_2 & 1+c_2^2\end{pmatrix}\f$,
 * the second \f$I\!I = -\tfrac1W\begin{pmatrix}2c_3 & c_4\\ c_4 & 2c_5
 * \end{pmatrix}\f$ with \f$W = \sqrt{1 + c_1^2 + c_2^2}\f$ (signed so that a
 * sphere is positive with an outward frame normal), and the shape operator is
 * \f$S = I^{-1}I\!I\f$: \f$\kappa = \operatorname{tr}S\f$,
 * \f$\kappa_G = \det S\f$, the principal direction its eigenvector.
 *
 * @return `true` on success; `false` (the caller falls back to COTAN) for a
 *         zero normal, fewer than three neighbours or fewer than three whose
 *         normals are within 75 degrees of the vertex's, or a rank-deficient
 *         fit.
 */
static bool fit_record(const MFTL_Object *obj, int v, int ncomp, double *rec)
{
  const double *const nv = &obj->verts.normal[3 * v];
  const double nm = MFTLVectorNorm(nv);
  if (!(nm > 0.0)) return MFTL_FALSE;
  const double ninv = 1.0 / nm;
  const double n[3] = {nv[0] * ninv, nv[1] * ninv, nv[2] * ninv};

  int ring[MFTL_MAX_VALENCE + 1];
  const int nr = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_MAX_VALENCE + 1, MFTL_FALSE);
  if (nr < 3) return MFTL_FALSE;

  double t1[3], t2[3];
  MFTLGeomTangentFrame(n, t1, t2);

  enum
  {
    MAX_SAMPLES = MFTL_MAX_VALENCE + 2
  };
  double pts[3 * MAX_SAMPLES], slopes[2 * MAX_SAMPLES], w[MAX_SAMPLES];
  unsigned char use[MAX_SAMPLES];

  /* Sample 0: the vertex, on its own tangent plane; positions in units of the
   * ring's RMS edge, so heights and slopes are both O(1). */
  const double hinv = MFTLGeomRingSamples(obj, v, n, t1, t2, ring, nr, pts);
  if (!(hinv > 0.0)) return MFTL_FALSE;
  slopes[0] = slopes[1] = 0.0;
  use[0] = 1;
  w[0] = 1.0;

  int n_slopes = 0;
  for (int i = 0; i < nr; ++i)
  {
    const double *const nj = &obj->verts.normal[3 * ring[i]];
    const double nz = MFTLVectorInProd(nj, n);
    const double nn = MFTLVectorNorm(nj);
    use[i + 1] = (unsigned char) (nz > CURVATURE_FIT_MIN_COS * nn);
    if (use[i + 1])
    {
      ++n_slopes;
      slopes[2 * (i + 1) + 0] = -MFTLVectorInProd(nj, t1) / nz;
      slopes[2 * (i + 1) + 1] = -MFTLVectorInProd(nj, t2) / nz;
    }
    else
      slopes[2 * (i + 1) + 0] = slopes[2 * (i + 1) + 1] = 0.0;
    w[i + 1] = 1.0;
  }
  /* Neighbours whose normals turn away from this vertex's by more than 75
   * degrees say the frame is wrong (a fold), not that the surface is curved. */
  if (n_slopes < 3) return MFTL_FALSE;

  double c[6];
  unsigned char kept[6];
  if (MFTLFitHeightParaboloid(nr + 1, pts, slopes, use, w, CURVATURE_FIT_TAU, c, kept) < 6) return MFTL_FALSE;

  const double hx = c[1], hy = c[2], hxx = 2.0 * c[3], hxy = c[4], hyy = 2.0 * c[5];
  const double E = 1.0 + hx * hx, F = hx * hy, G = 1.0 + hy * hy;
  const double W2 = 1.0 + hx * hx + hy * hy, W = MFTL_SQRT(W2);
  const double Lf = -hxx / W, Mf = -hxy / W, Nf = -hyy / W;

  /* S = I^{-1} II, in units of 1/hs. */
  const double S11 = (G * Lf - F * Mf) / W2, S12 = (G * Mf - F * Nf) / W2;
  const double S21 = (E * Mf - F * Lf) / W2, S22 = (E * Nf - F * Mf) / W2;
  const double kappa_s = S11 + S22;
  const double gauss_s = S11 * S22 - S12 * S21;

  write_scalars(rec, ncomp, kappa_s * hinv, gauss_s * hinv * hinv);
  if (ncomp < MFTL_CURVATURE_COMPONENTS) return MFTL_TRUE;

  /* The k1 eigenvector: (S - k2 I) maps every vector into it (Cayley-Hamilton);
   * take its larger column. Both columns vanish only at an umbilic. */
  const double h = 0.5 * kappa_s;
  double disc = h * h - gauss_s;
  if (!(disc > 0.0)) disc = 0.0;
  const double k2 = h - MFTL_SQRT(disc);
  const double a1 = S11 - k2, b1 = S21, a2 = S12, b2 = S22 - k2;
  double u = 1.0, q = 0.0;
  if (a1 * a1 + b1 * b1 >= a2 * a2 + b2 * b2)
  {
    if (a1 * a1 + b1 * b1 > 0.0) u = a1, q = b1;
  }
  else
    u = a2, q = b2;

  /* Parameter direction (u, q) to a tangent vector u x_u + q x_v. */
  double *const dir = &rec[MFTL_CURVATURE_DIR1];
  for (int k = 0; k < 3; ++k) dir[k] = u * t1[k] + q * t2[k] + (u * hx + q * hy) * n[k];
  const double dm = MFTL_SQRT(MFTLVectorNormSq(dir));  // not MFTLVectorNorm: here it turned the branch below into a slower select
  const double dinv = (dm > 0.0) ? 1.0 / dm : 0.0;
  for (int k = 0; k < 3; ++k) dir[k] *= dinv;
  return MFTL_TRUE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Entry point
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLComputeVertexCurvature(const MFTL_Object *obj, MFTL_CurvatureMethod method, int ncomp, double *out, int *n_fallback)
{
  if (n_fallback != MFTL_NULL) *n_fallback = 0;
  if (obj == MFTL_NULL) return MFTL_ERROR;
  if (method != MFTL_CURVATURE_COTAN && method != MFTL_CURVATURE_FIT) return MFTL_ERROR;
  if (ncomp != 1 && ncomp != 2 && ncomp != 4 && ncomp != MFTL_CURVATURE_COMPONENTS) return MFTL_ERROR;

  const int n_v = obj->verts.n;
  if (n_v == 0) return MFTL_SUCCESS;
  if (out == MFTL_NULL) return MFTL_ERROR;
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLComputeVertexCurvature: tris.verts is stale -- MFTLObjectBuildDerived first");

  double *const acc = (double *) MFTLCallocOrDie((size_t) n_v * ACC_STRIDE, sizeof(double), "vertex curvature accumulators");
  scatter_cotan(obj, ncomp >= 2, acc);

  int fallbacks = 0;
  for (int v = 0; v < n_v; ++v)
  {
    double *const rec = &out[(size_t) ncomp * v];
    if (method == MFTL_CURVATURE_FIT)
    {
      if (fit_record(obj, v, ncomp, rec)) continue;
      if (v < obj->verts.n_local) ++fallbacks;
    }
    cotan_record(obj, v, &acc[ACC_STRIDE * v], ncomp, rec);
  }

  free(acc);
  if (n_fallback != MFTL_NULL) *n_fallback = fallbacks;
  return MFTL_SUCCESS;
}
