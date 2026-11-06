/**
 * @file remesh_fit.c
 * @author Christian Gorges
 * @date 2026-11-05
 * @ingroup mftl_remeshing
 *
 * @brief Where a split or collapse puts its new vertex under a paraboloid
 *        `split_placement` or `collapse_placement`: above the edge midpoint, on a weighted least-squares
 *        height paraboloid through the vertices of the edge's two-ring.
 *
 * ## What the old mftl fit did, and what changed
 *
 * mftl fitted \f$z = Ax^2 + Bxy + Cy^2 + Dx + Ey + F\f$ to the 4 + 4 vertices of
 * the edge's two triangles and the four across their outer edges, unweighted,
 * in unscaled coordinates, solved by LAPACK `dgelsy` with `rcond = 1e-10`, the
 * frame normal summed over the locally OWNED vertices only. Each of those is a
 * defect here:
 *
 * - **8 samples for 6 unknowns** nearly interpolates: every sample's noise
 *   enters the result at full weight. The two-ring has 20 to 30 samples.
 * - **No weights**: a far sample counts as much as an endpoint. Here Wendland's
 *   C2 function of the distance from the midpoint, support `fit_radius` local
 *   edge lengths (the kernel Jibben et al.'s facet fit uses).
 * - **Unscaled coordinates** give the columns norms of order 1, h and h^2, so a
 *   relative rank tolerance of 1e-10 drops the curvature columns once h^2 falls
 *   below it: at edge lengths under about 1e-5 in the units of the coordinates
 *   the fit silently becomes a plane through the chord. Here coordinates are in
 *   units of the local edge length and the tolerance is 1e-3 of the first pivot.
 * - **The frame depended on the partition** (owned normals only). Here it uses
 *   the endpoints' normals, whose fans are complete wherever an operation is
 *   allowed.
 *
 * Its coordinate transformation was right: the "inverse" it formed is the
 * cofactor matrix over the determinant, i.e. the inverse TRANSPOSED, which for an
 * orthonormal frame is the frame itself -- the rotation it needed.
 *
 * ## Samples on the vertices, not on the facets
 *
 * A fit to facet integrals (Jibben et al.) reproduces the facets' volume-
 * equivalent surface, which on a mesh whose vertices lie on the interface sits a
 * mean sagitta below them: an edge midpoint placed on it lands at the chord
 * (studies/leveque_remeshing/checks/facet_vs_vertex_paraboloid.py). A front
 * tracker's vertices are the samples of the interface, so they are what is
 * fitted.
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_vectoroperations.h"
#include "remesh_comm.h"
#include "remesh_internal.h"

/** @brief Most stencil vertices: the two-ring of an edge at valence 6 has about 22. */
#define MFTL_REMESH_FIT_MAX_STENCIL 64

/** @brief Relative rank tolerance of the fit (scaled columns, so dimensionless). */
#define MFTL_REMESH_FIT_RANK_TOL 1e-3

/** @brief A sample whose normal is more than 60 degrees from the frame's z is not on the height function's sheet. */
#define MFTL_REMESH_FIT_NORMAL_COS 0.5

/** @brief Largest admitted height above the chord, as a fraction of the edge (a 105-degree arc). */
#define MFTL_REMESH_FIT_MAX_HEIGHT 0.25

static long g_fit_counts[4] = {0, 0, 0, 0};
static double g_fit_time = 0.0;

void MFTLRemeshGetFitCounters(MFTL_RemeshFitCounters *out)
{
  out->n_fits = g_fit_counts[0];
  out->n_rank_deficient = g_fit_counts[1];
  out->n_fallbacks = g_fit_counts[2];
  out->n_with_ghosts = g_fit_counts[3];
  out->t_fit = g_fit_time;
}

void MFTLRemeshResetFitCounters(void)
{
  g_fit_counts[0] = g_fit_counts[1] = g_fit_counts[2] = g_fit_counts[3] = 0;
  g_fit_time = 0.0;
}

void MFTLRemeshFitNoteFallback(void) { g_fit_counts[2]++; }

/**
 * @brief Unit vertex normal from Max's weights (Max 1999), exact for vertices on
 *        a sphere: \f$\sum_f (\vect{e}_1\times\vect{e}_2)/(|\vect{e}_1|^2|\vect{e}_2|^2)\f$.
 *
 * Walks the fan both ways from `out_he_idx`, so an open fan (mesh rim, halo
 * rim) is summed as far as it is present.
 *
 * @param      obj The mesh.
 * @param      v   The vertex.
 * @param[out] n   The unit normal (zero if the fan has no area).
 * @return `true` if the fan closed (the normal is that of the whole fan).
 */
static bool max_normal(const MFTL_Object *obj, int v, double n[3])
{
  n[0] = n[1] = n[2] = 0.0;
  const int start = obj->verts.rec[v].out_he_idx;
  if (start == MFTL_ERROR) return MFTL_FALSE;

  const double *const p = MFTLVertexPos(obj, v);
  bool closed = MFTL_FALSE;

  /* Forward: outgoing v->w1 in (v, w1, w2), then the outgoing v->w2 = twin(prev). */
  int h = start;
  for (int guard = 0; guard < MFTL_REMESH_RING_MAX; ++guard)
  {
    double d1[3], d2[3], c[3];
    MFTLGetMinimumImageVector(obj->domain, d1, p, MFTLVertexPos(obj, MFTLHeTarget(obj, h)));
    MFTLGetMinimumImageVector(obj->domain, d2, p, MFTLVertexPos(obj, MFTLHeTarget(obj, MFTLHeNextIdx(h))));
    MFTLVectorCrossVector(c, d1, d2);
    const double w = MFTLVectorNormSq(d1) * MFTLVectorNormSq(d2);
    if (w > 0.0)
      for (int k = 0; k < 3; ++k) n[k] += c[k] / w;
    h = obj->tris.he[MFTLHePrevIdx(h)].twin_idx;
    if (h == start)
    {
      closed = MFTL_TRUE;
      break;
    }
    if (h == MFTL_ERROR) break;
  }

  /* Backward from the start when the fan is open: the triangle before is the
   * one across the start's twin. */
  if (!closed)
  {
    int g = obj->tris.he[start].twin_idx;
    for (int guard = 0; g != MFTL_ERROR && guard < MFTL_REMESH_RING_MAX; ++guard)
    {
      const int out = MFTLHeNextIdx(g);  // v -> x in the triangle of g
      double d1[3], d2[3], c[3];
      MFTLGetMinimumImageVector(obj->domain, d1, p, MFTLVertexPos(obj, MFTLHeTarget(obj, out)));
      MFTLGetMinimumImageVector(obj->domain, d2, p, MFTLVertexPos(obj, MFTLHeTarget(obj, MFTLHeNextIdx(out))));
      MFTLVectorCrossVector(c, d1, d2);
      const double w = MFTLVectorNormSq(d1) * MFTLVectorNormSq(d2);
      if (w > 0.0)
        for (int k = 0; k < 3; ++k) n[k] += c[k] / w;
      g = obj->tris.he[out].twin_idx;
    }
  }

  const double len = MFTLVectorNorm(n);
  const double inv = (len > 0.0) ? 1.0 / len : 0.0;
  for (int k = 0; k < 3; ++k) n[k] *= inv;
  return closed;
}

/**
 * @brief Appends @p v to @p list unless present.
 * @param list The stencil.
 * @param n    Its length.
 * @param v    A vertex.
 * @return The new length (unchanged if present or full).
 */
static inline int stencil_add(int *list, int n, int v)
{
  for (int i = 0; i < n; ++i)
    if (list[i] == v) return n;
  if (n >= MFTL_REMESH_FIT_MAX_STENCIL) return n;
  list[n] = v;
  return n + 1;
}

/**
 * @brief Wendland's C2 function, \f$(1-q)^4(1+4q)\f$ on [0, 1), zero beyond.
 * @param q Distance over support radius.
 * @return The weight.
 */
static inline double wendland_c2(double q)
{
  if (!(q < 1.0)) return 0.0;
  const double t = 1.0 - q;
  return t * t * t * t * (1.0 + 4.0 * q);
}

/**
 * @brief The fit proper; MFTLRemeshFitEdgePoint() times and counts it.
 * @param      obj    The mesh.
 * @param      h      The edge.
 * @param      params    The parameter block.
 * @param      placement Which paraboloid placement.
 * @param[out] out       The point.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for a fallback.
 */
static int fit_edge_point(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, MFTL_RemeshPlacement placement, double out[3])
{
  const int a = MFTLHeOrigin(obj, h);
  const int b = MFTLHeTarget(obj, h);
  const double *const pa = MFTLVertexPos(obj, a);
  const double *const pb = MFTLVertexPos(obj, b);

  /* ── The stencil: the edge's two-ring ───────────────────────────────── */
  int stencil[MFTL_REMESH_FIT_MAX_STENCIL];
  int ns = 0;
  ns = stencil_add(stencil, ns, a);
  ns = stencil_add(stencil, ns, b);

  double h_sum = 0.0;
  int h_count = 0;
  for (int e = 0; e < 2; ++e)
  {
    const int v = e ? b : a;
    int ring[MFTL_REMESH_RING_MAX];
    const int nr = MFTLGetVertexOneRingVertices(obj, v, ring, MFTL_REMESH_RING_MAX, MFTL_FALSE);
    if (nr <= 0 || nr >= MFTL_REMESH_RING_MAX) goto fail;
    for (int i = 0; i < nr; ++i)
    {
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, MFTLVertexPos(obj, v), MFTLVertexPos(obj, ring[i]));
      h_sum += MFTLVectorNorm(d);
      h_count++;
      ns = stencil_add(stencil, ns, ring[i]);
    }
  }
  const int n_one_ring = ns;
  for (int i = 2; i < n_one_ring; ++i)
  {
    int ring[MFTL_REMESH_RING_MAX];
    const int nr = MFTLGetVertexOneRingVertices(obj, stencil[i], ring, MFTL_REMESH_RING_MAX, MFTL_FALSE);
    for (int j = 0; j < nr && j < MFTL_REMESH_RING_MAX; ++j) ns = stencil_add(stencil, ns, ring[j]);
  }
  const double hl = h_sum / (double) MFTL_MAX(h_count, 1);
  if (!(hl > 0.0)) goto fail;

  /* ── The frame ──────────────────────────────────────────────────────── */
  double na[3], nb[3], nz[3];
  max_normal(obj, a, na);
  max_normal(obj, b, nb);
  for (int k = 0; k < 3; ++k) nz[k] = na[k] + nb[k];
  if (!(MFTLVectorNormSq(nz) > 0.0)) goto fail;
  MFTLNormalizeVector(nz);

  double ab[3], ex[3], ey[3];
  MFTLGetMinimumImageVector(obj->domain, ab, pa, pb);
  const double ab_len = MFTLVectorNorm(ab);
  const double along = MFTLVectorInProd(ab, nz);
  for (int k = 0; k < 3; ++k) ex[k] = ab[k] - along * nz[k];
  if (!(MFTLVectorNormSq(ex) > 0.0)) goto fail;
  MFTLNormalizeVector(ex);
  MFTLVectorCrossVector(ey, nz, ex);

  double m[3];
  midpoint_periodic(obj->domain, pa, pb, m);

  /* ── Samples, in units of the local edge length ─────────────────────── */
  const bool with_normals = (placement == MFTL_REMESH_PLACE_PARABOLOID_NORMALS);
  const double inv_h = 1.0 / hl;
  const double rho = params->fit_radius;

  double pts[3 * MFTL_REMESH_FIT_MAX_STENCIL];
  double slopes[2 * MFTL_REMESH_FIT_MAX_STENCIL];
  double w[MFTL_REMESH_FIT_MAX_STENCIL];
  unsigned char use_slope[MFTL_REMESH_FIT_MAX_STENCIL];
  int n = 0;
  bool ghost_sample = MFTL_FALSE;

  for (int i = 0; i < ns; ++i)
  {
    const int v = stencil[i];
    double d[3];
    MFTLGetMinimumImageVector(obj->domain, d, m, MFTLVertexPos(obj, v));
    for (int k = 0; k < 3; ++k) d[k] *= inv_h;

    const double wi = wendland_c2(MFTLVectorNorm(d) / rho);
    if (!(wi > 0.0)) continue;

    double nv[3];
    const bool closed = max_normal(obj, v, nv);
    const double nzl = MFTLVectorInProd(nv, nz);
    if (!(nzl >= MFTL_REMESH_FIT_NORMAL_COS)) continue;

    if (v >= obj->verts.n_local) ghost_sample = MFTL_TRUE;
    pts[3 * n + 0] = MFTLVectorInProd(d, ex);
    pts[3 * n + 1] = MFTLVectorInProd(d, ey);
    pts[3 * n + 2] = MFTLVectorInProd(d, nz);
    w[n] = wi;

    /* Slopes only from a normal that describes the whole fan of an owned,
     * smooth vertex; a ghost's fan is partial. */
    const int f = obj->verts.rec[v].flags;
    use_slope[n] = with_normals && closed && v < obj->verts.n_local && !(f & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER | MFTL_FLAG_BOUNDARY));
    if (use_slope[n])
    {
      slopes[2 * n + 0] = -MFTLVectorInProd(nv, ex) / nzl;
      slopes[2 * n + 1] = -MFTLVectorInProd(nv, ey) / nzl;
    }
    ++n;
  }
  if (ghost_sample) g_fit_counts[3]++;
  if (n < 6) goto fail;

  double c[6];
  unsigned char kept[6];
  const int rank = MFTLFitHeightParaboloid(n, pts, with_normals ? slopes : MFTL_NULL, use_slope, w, MFTL_REMESH_FIT_RANK_TOL, c, kept);
  if (!kept[0] || rank < 3) goto fail;
  if (rank < 6) g_fit_counts[1]++;  // placed, with a lower degree in some direction

  const double z0 = c[0] * hl;
  if (!(fabs(z0) <= MFTL_REMESH_FIT_MAX_HEIGHT * ab_len)) goto fail;

  for (int k = 0; k < 3; ++k) out[k] = m[k] + z0 * nz[k];
  MFTLApplyPeriodicity(obj->domain, out);
  return MFTL_SUCCESS;

fail:
  return MFTL_ERROR;
}

int MFTLRemeshFitEdgePoint(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, MFTL_RemeshPlacement placement, double out[3])
{
  const double t0 = MFTLRemeshWallTime();
  const int rc = fit_edge_point(obj, h, params, placement, out);
  g_fit_time += MFTLRemeshWallTime() - t0;
  g_fit_counts[0]++;
  if (rc != MFTL_SUCCESS) g_fit_counts[2]++;
  return rc;
}
