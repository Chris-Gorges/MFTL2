/**
 * @file signed_distance.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_geometry
 *
 * @brief Bounded rasterisation of a triangle mesh's signed distance onto a
 *        lattice window, signed by angle-weighted pseudo-normals.
 *
 * A SCATTER over triangles: each triangle visits only the nodes inside its box
 * dilated by the cutoff and keeps, per node, the contribution of smallest
 * magnitude (MFTLSdfMerge()). Cost \f$O(T\,(h+2r)^3/\Delta x^3)\f$ for triangle
 * size h and cutoff r, independent of the rest of the mesh. The sign comes from
 * the pseudo-normal of the closest FEATURE (face, edge or vertex) -- see
 * MFTLRasterizeSignedDistance().
 *
 * Most of a triangle's box is out of its reach or already claimed by a nearer
 * triangle, so the closest-point query is guarded by two exact lower bounds
 * (plane, bounding sphere); MFTLRasterizeSignedDistanceBanded() gives a marked
 * zone a wider cutoff. Neither changes a single bit of the field.
 */

#include "mftl_geometry.h"

#include <math.h>
#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_types.h"

void MFTLSdfGridInitWindow(MFTL_SdfGrid *grid, const double origin[3], const double spacing[3], const int n_global[3], const bool periodic[3], const int lo[3],
                           const int dims[3])
{
  MFTL_ASSERT(grid != MFTL_NULL);

  size_t n = 1;
  for (int k = 0; k < 3; ++k)
  {
    MFTL_ASSERT(spacing[k] > 0.0 && dims[k] >= 1 && "MFTLSdfGridInitWindow: empty axis or non-positive spacing");
    grid->origin[k] = origin[k];
    grid->spacing[k] = spacing[k];
    grid->n_global[k] = n_global[k];
    grid->periodic[k] = (periodic != MFTL_NULL) ? periodic[k] : MFTL_FALSE;
    grid->lo[k] = lo[k];
    grid->dims[k] = dims[k];
    MFTL_ASSERT((!grid->periodic[k] || dims[k] <= n_global[k]) && "MFTLSdfGridInitWindow: a periodic window may not exceed one period");
    n *= (size_t) dims[k];
  }

  grid->phi = (double *) MFTLMallocOrDie(n * sizeof(double), "SDF grid");
  for (size_t i = 0; i < n; ++i) grid->phi[i] = MFTL_SDF_UNKNOWN;
  grid->cross = MFTL_NULL;
}

void MFTLSdfGridEnableCrossings(MFTL_SdfGrid *grid)
{
  MFTL_ASSERT(grid != MFTL_NULL);
  if (grid->cross != MFTL_NULL || grid->phi == MFTL_NULL) return;
  const size_t n = 3 * (size_t) grid->dims[0] * grid->dims[1] * grid->dims[2];
  grid->cross = (double *) MFTLMallocOrDie(n * sizeof(double), "SDF grid edge crossings");
  for (size_t i = 0; i < n; ++i) grid->cross[i] = MFTL_SDF_CROSS_NONE;
}

void MFTLSdfGridInit(MFTL_SdfGrid *grid, const double origin[3], const double spacing[3], const int dims[3])
{
  const int lo[3] = {0, 0, 0};
  MFTLSdfGridInitWindow(grid, origin, spacing, dims, MFTL_NULL, lo, dims);
}

void MFTLSdfGridFree(MFTL_SdfGrid *grid)
{
  if (grid == MFTL_NULL) return;
  free(grid->phi);
  free(grid->cross);
  grid->phi = MFTL_NULL;
  grid->cross = MFTL_NULL;
}

/**
 * @brief Unit normal of a triangle from its corners, or zero if it has no area.
 *
 * Recomputed rather than read from `tris.normal`, which a redistribution leaves
 * unspecified.
 *
 * @param      c The three corners, as MFTLGetTriangleCorners() returns them.
 * @param[out] n The unit normal.
 */
static void face_normal(const double c[3][3], double n[3])
{
  const double e1[3] = {c[1][0] - c[0][0], c[1][1] - c[0][1], c[1][2] - c[0][2]};
  const double e2[3] = {c[2][0] - c[0][0], c[2][1] - c[0][1], c[2][2] - c[0][2]};
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const double mag = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  const double inv = (mag > 0.0) ? 1.0 / mag : 0.0;
  n[0] *= inv;
  n[1] *= inv;
  n[2] *= inv;
}

long MFTLRasterizeSignedDistance(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris, double cutoff)
{
  return MFTLRasterizeSignedDistanceBanded(grid, obj, n_tris, cutoff, cutoff, MFTL_NULL, MFTL_NULL);
}

long MFTLRasterizeSignedDistanceBanded(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris, double cutoff, double wide_cutoff,
                                       const unsigned char *wide_nodes, const unsigned char *wide_tris)
{
  MFTL_ASSERT(grid != MFTL_NULL && obj != MFTL_NULL);
  MFTL_ASSERT(cutoff > 0.0 && wide_cutoff >= cutoff);
  MFTL_ASSERT(n_tris <= obj->tris.n);
  MFTL_ASSERT((n_tris == 0 || MFTLObjectHasTopology(obj)) && "MFTLRasterizeSignedDistance: stale tris.verts -- MFTLObjectBuildDerived first");

  if (grid->phi == MFTL_NULL || n_tris <= 0) return 0;

  const double cut_sq = cutoff * cutoff, wide_sq = wide_cutoff * wide_cutoff;
  const int nx = grid->dims[0], ny = grid->dims[1];
  long n_updates = 0;

  for (int t = 0; t < n_tris; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);

    /* A flagged triangle reaches the wide band, and there only; every other
     * node -- and every node of an unflagged triangle -- takes the narrow one. */
    const bool wide = (wide_tris != MFTL_NULL && wide_tris[t]);
    const double reach = wide ? wide_cutoff : cutoff;
    const unsigned char *const band = wide ? wide_nodes : MFTL_NULL;
    const double reach_sq = wide ? wide_sq : cut_sq;

    /* ── The node box this triangle can reach ───────────────────────────── */
    int g_lo[3], g_hi[3];
    bool empty = MFTL_FALSE;
    for (int k = 0; k < 3; ++k)
    {
      const double mn = fmin(T[0][k], fmin(T[1][k], T[2][k])) - reach;
      const double mx = fmax(T[0][k], fmax(T[1][k], T[2][k])) + reach;
      g_lo[k] = (int) ceil((mn - grid->origin[k]) / grid->spacing[k]);
      g_hi[k] = (int) floor((mx - grid->origin[k]) / grid->spacing[k]);
      if (g_hi[k] < g_lo[k]) empty = MFTL_TRUE;
    }
    if (empty) continue;

    /* Pseudo-normals: the face normal always; each edge's lazily. Edge k of
     * MFTLClosestPointOnTriangle() is half-edge 3t+k, and corner k is
     * tris.verts[3t+k]. */
    double n_face[3];
    face_normal(T, n_face);
    double n_edge[3][3];
    bool have_edge[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};

    /* Two LOWER bounds on a node's distance to T, a few flops each: to T's
     * plane, and to its bounding sphere (centroid, farthest corner). A node is
     * skipped only when a bound exceeds what would reject it anyway -- its
     * cutoff, or the nearer value it already holds -- by a margin covering the
     * rounding of both the bound and the exact distance (a few ulps of the
     * largest coordinate involved). Every skip is therefore one the exact test
     * below would also have made, and the field is the same bits; what is saved
     * is the closest-point query for a node outside T's reach, or one some
     * nearer triangle already claimed, which inside the band is most of them. */
    double ctr[3], rad_sq = 0.0, mag = 0.0;
    for (int k = 0; k < 3; ++k) ctr[k] = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
    for (int j = 0; j < 3; ++j)
    {
      const double ex = T[j][0] - ctr[0], ey = T[j][1] - ctr[1], ez = T[j][2] - ctr[2];
      rad_sq = fmax(rad_sq, ex * ex + ey * ey + ez * ez);
      for (int k = 0; k < 3; ++k) mag = fmax(mag, fabs(T[j][k]));
    }
    const double rad = sqrt(rad_sq);
    const double slack = 1e-12 * (mag + reach + fabs(grid->origin[0]) + fabs(grid->origin[1]) + fabs(grid->origin[2]));
    const double plane = n_face[0] * T[0][0] + n_face[1] * T[0][1] + n_face[2] * T[0][2];

    for (int gz = g_lo[2]; gz <= g_hi[2]; ++gz)
    {
      const int lz = MFTLSdfGridLocal(grid, 2, gz);
      if (lz < 0) continue;
      const double z = MFTLSdfGridCoord(grid, 2, gz);

      for (int gy = g_lo[1]; gy <= g_hi[1]; ++gy)
      {
        const int ly = MFTLSdfGridLocal(grid, 1, gy);
        if (ly < 0) continue;
        const double y = MFTLSdfGridCoord(grid, 1, gy);

        for (int gx = g_lo[0]; gx <= g_hi[0]; ++gx)
        {
          const int lx = MFTLSdfGridLocal(grid, 0, gx);
          if (lx < 0) continue;

          const double x[3] = {MFTLSdfGridCoord(grid, 0, gx), y, z};
          const size_t idx = ((size_t) lz * ny + ly) * nx + lx;
          double *const node = &grid->phi[idx];

          /* The distance that would still be written here: within this node's
           * cutoff, and not beaten by what it holds. */
          const double limit = fmin((band != MFTL_NULL && !band[idx]) ? cutoff : reach, fabs(*node)) + slack;
          if (fabs(n_face[0] * x[0] + n_face[1] * x[1] + n_face[2] * x[2] - plane) > limit) continue;
          const double cx = x[0] - ctr[0], cy = x[1] - ctr[1], cz = x[2] - ctr[2];
          if (cx * cx + cy * cy + cz * cz > (rad + limit) * (rad + limit)) continue;

          double q[3];
          MFTL_TriangleFeature f;
          const double d_sq = MFTLClosestPointOnTriangle(x, T, q, &f);
          if (d_sq > reach_sq) continue;
          if (band != MFTL_NULL && !band[idx] && d_sq > cut_sq) continue;
          const double d = sqrt(d_sq);
          /* A node already holding something strictly nearer cannot change. */
          if (d > fabs(*node)) continue;

          const double *n = n_face;
          if (f >= MFTL_TRI_FEATURE_EDGE0)
          {
            const int e = f - MFTL_TRI_FEATURE_EDGE0;
            if (!have_edge[e])
            {
              const int tw = obj->tris.he[3 * t + e].twin_idx;
              double nt[3] = {0.0, 0.0, 0.0};
              if (tw != MFTL_ERROR)
              {
                double Tw[3][3];
                MFTLGetTriangleCorners(obj, MFTLHeTriIdx(tw), Tw);
                face_normal(Tw, nt);
              }
              for (int k = 0; k < 3; ++k) n_edge[e][k] = n_face[k] + nt[k];
              have_edge[e] = MFTL_TRUE;
            }
            n = n_edge[e];
          }
          else if (f >= MFTL_TRI_FEATURE_VERTEX0)
          {
            n = &obj->verts.normal[3 * obj->tris.verts[3 * t + (f - MFTL_TRI_FEATURE_VERTEX0)]];
          }

          const double dot = (x[0] - q[0]) * n[0] + (x[1] - q[1]) * n[1] + (x[2] - q[2]) * n[2];
          const double value = (dot < 0.0) ? -d : d;
          const double merged = MFTLSdfMerge(*node, value);
          if (merged != *node)
          {
            *node = merged;
            n_updates++;
          }
        }
      }
    }
  }
  return n_updates;
}

long MFTLRasterizeEdgeCrossings(MFTL_SdfGrid *grid, const MFTL_Object *obj, int n_tris)
{
  MFTL_ASSERT(grid != MFTL_NULL && obj != MFTL_NULL);
  MFTL_ASSERT(n_tris <= obj->tris.n);
  if (grid->phi == MFTL_NULL) return 0;
  MFTLSdfGridEnableCrossings(grid);
  if (n_tris <= 0) return 0;

  const int nx = grid->dims[0], ny = grid->dims[1];
  long n_recorded = 0;

  for (int t = 0; t < n_tris; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);

    for (int a = 0; a < 3; ++a)
    {
      const int b = (a + 1) % 3, c = (a + 2) % 3;

      /* The triangle projected along a; twice its signed area. A triangle
       * containing the axis direction has none and no isolated crossing. */
      const double u0 = T[0][b], v0 = T[0][c], u1 = T[1][b], v1 = T[1][c], u2 = T[2][b], v2 = T[2][c];
      const double area2 = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0);
      if (area2 == 0.0) continue;

      const int gb_lo = (int) ceil((fmin(u0, fmin(u1, u2)) - grid->origin[b]) / grid->spacing[b]);
      const int gb_hi = (int) floor((fmax(u0, fmax(u1, u2)) - grid->origin[b]) / grid->spacing[b]);
      const int gc_lo = (int) ceil((fmin(v0, fmin(v1, v2)) - grid->origin[c]) / grid->spacing[c]);
      const int gc_hi = (int) floor((fmax(v0, fmax(v1, v2)) - grid->origin[c]) / grid->spacing[c]);

      for (int gc = gc_lo; gc <= gc_hi; ++gc)
      {
        const int lc = MFTLSdfGridLocal(grid, c, gc);
        if (lc < 0) continue;
        const double yc = MFTLSdfGridCoord(grid, c, gc);
        for (int gb = gb_lo; gb <= gb_hi; ++gb)
        {
          const int lb = MFTLSdfGridLocal(grid, b, gb);
          if (lb < 0) continue;
          const double yb = MFTLSdfGridCoord(grid, b, gb);

          /* Edge functions of the line's foot against the projected edges,
           * each opposite the corner it weights. Inside, all have the sign of
           * the area; on an edge one is zero and the neighbour across it may
           * count the line too, which only makes the slot MANY. */
          const double w0 = (u2 - u1) * (yc - v1) - (v2 - v1) * (yb - u1);
          const double w1 = (u0 - u2) * (yc - v2) - (v0 - v2) * (yb - u2);
          const double w2 = (u1 - u0) * (yc - v0) - (v1 - v0) * (yb - u0);
          const bool inside = (area2 > 0.0) ? (w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0) : (w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0);
          if (!inside) continue;

          const double w = w0 + w1 + w2;
          if (w == 0.0) continue;
          const double xa = (w0 * T[0][a] + w1 * T[1][a] + w2 * T[2][a]) / w;
          const int ga = (int) floor((xa - grid->origin[a]) / grid->spacing[a]);
          const int la = MFTLSdfGridLocal(grid, a, ga);
          if (la < 0) continue;
          const double f = fmin(fmax((xa - MFTLSdfGridCoord(grid, a, ga)) / grid->spacing[a], 0.0), 1.0);

          int l[3];
          l[a] = la;
          l[b] = lb;
          l[c] = lc;
          double *const slot = &grid->cross[3 * (((size_t) l[2] * ny + l[1]) * nx + l[0]) + a];
          *slot = MFTLSdfCrossMerge(*slot, f);
          n_recorded++;
        }
      }
    }
  }
  return n_recorded;
}
