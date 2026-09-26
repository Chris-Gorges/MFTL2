/**
 * @file signed_distance.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_geometry
 *
 * @brief Bounded rasterisation of a triangle mesh's signed distance onto a
 *        lattice window, with the sign from angle-weighted pseudo-normals.
 *
 * ## SCATTER OVER TRIANGLES, BOUNDED BY A CUTOFF
 *
 * The same argument vertex_geometry.c makes for normals applies here with more
 * force. The GATHER form -- for each node, find the nearest triangle -- needs a
 * spatial search per node. The SCATTER form visits each triangle once, walks
 * only the nodes inside its box dilated by the cutoff, and keeps the smallest
 * magnitude per node. Cost is `O(T * (h + 2 r)^3 / dx^3)` with h the triangle
 * size and r the cutoff: a few hundred nodes per triangle at the resolutions
 * the topology change uses, independent of how far the rest of the mesh is.
 *
 * ## THE SIGN
 *
 * A node's distance is decided by the nearest triangle, but its sign cannot be:
 * nearest to a shared edge or vertex, the face normal of whichever triangle was
 * visited first is simply the wrong question on a folded surface. The
 * pseudo-normal of the FEATURE the closest point lies on is the right one --
 * see MFTLRasterizeSignedDistance() in mftl_geometry.h.
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
  grid->phi = MFTL_NULL;
}

/**
 * @brief Unit normal of a triangle from its corners, or zero if it has no area.
 *
 * Recomputed from the current corners rather than read from `tris.normal`,
 * which nothing guarantees is current -- redistribution leaves it unspecified by
 * design.
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
  MFTL_ASSERT(grid != MFTL_NULL && obj != MFTL_NULL);
  MFTL_ASSERT(cutoff > 0.0);
  MFTL_ASSERT(n_tris <= obj->tris.n);
  MFTL_ASSERT((n_tris == 0 || MFTLObjectHasTopology(obj)) && "MFTLRasterizeSignedDistance: stale tris.verts -- MFTLObjectBuildDerived first");

  if (grid->phi == MFTL_NULL || n_tris <= 0) return 0;

  const double cut_sq = cutoff * cutoff;
  const int nx = grid->dims[0], ny = grid->dims[1];
  long n_updates = 0;

  for (int t = 0; t < n_tris; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);

    /* ── The node box this triangle can reach ───────────────────────────── */
    int g_lo[3], g_hi[3];
    bool empty = MFTL_FALSE;
    for (int k = 0; k < 3; ++k)
    {
      const double mn = fmin(T[0][k], fmin(T[1][k], T[2][k])) - cutoff;
      const double mx = fmax(T[0][k], fmax(T[1][k], T[2][k])) + cutoff;
      g_lo[k] = (int) ceil((mn - grid->origin[k]) / grid->spacing[k]);
      g_hi[k] = (int) floor((mx - grid->origin[k]) / grid->spacing[k]);
      if (g_hi[k] < g_lo[k]) empty = MFTL_TRUE;
    }
    if (empty) continue;

    /* ── The three pseudo-normals this triangle can need ────────────────────
     *
     * Computed lazily per triangle, once: most triangles never produce an edge
     * or vertex closest point on any node, but the face normal is always used.
     * The corner indices come from `tris.verts`, whose slot k is the ORIGIN of
     * half-edge 3t+k -- the same convention MFTLClosestPointOnTriangle's edge
     * numbering follows. */
    double n_face[3];
    face_normal(T, n_face);
    double n_edge[3][3];
    bool have_edge[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};

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
          double q[3];
          MFTL_TriangleFeature f;
          const double d_sq = MFTLClosestPointOnTriangle(x, T, q, &f);
          if (d_sq > cut_sq) continue;

          double *const node = &grid->phi[((size_t) lz * ny + ly) * nx + lx];
          const double d = sqrt(d_sq);
          /* A node that already holds something strictly nearer cannot change,
           * whatever the sign here would be -- skip the pseudo-normal. */
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
