/**
 * @file marching_cubes.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_meshgen
 *
 * @brief Marching cubes with a 256-case table GENERATED from one face rule, and
 *        an extractor that names every vertex by the lattice edge it lies on.
 *
 * **Face rule.** Walk a face's four corners counter-clockwise as seen from
 * outside the cell; the sign changes at 0, 2 or 4 of its edges. Connect every
 * crossing where the walk ENTERS the negative region to the next crossing where
 * it LEAVES it. With two crossings that is the only segment; on an ambiguous
 * face (negative corners on a diagonal) it cuts each negative corner off
 * separately. The rule depends only on the face's four signs, and the
 * neighbouring cell -- walking the face in the opposite direction -- produces
 * the same segments with opposite orientation, so the surface cannot crack
 * between cells. Every segment has the negative region on its right, so the
 * chained loops are wound with normals from inside (negative) to outside.
 *
 * A tabulated table assembled from symmetry classes and complements can resolve
 * the same ambiguous face differently from its two sides -- the classic
 * marching-cubes hole. Generating every case from the face rule avoids it, and a
 * typo in a hand-written row cannot exist.
 *
 * **Triangulating a loop.** A loop may pass through one face twice; a fan from
 * an arbitrary vertex could then create a triangle lying IN that face, which the
 * neighbouring cell may duplicate. Each loop is fanned from an apex with no
 * diagonal to a crossing on a shared face; the tests assert such an apex exists
 * for every loop of every case (the centroid fallback is never used).
 */

#include "mftl_meshgen.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Cube geometry
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Cube edge between two corners that differ in exactly one bit.
 * @param p First corner.
 * @param q Second corner.
 * @return The edge index in [0, 12).
 */
static int cube_edge(int p, int q)
{
  const int diff = p ^ q;
  const int a = (diff == 1) ? 0 : (diff == 2) ? 1 : 2;
  const int lower = p & q;
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  return 4 * a + ((lower >> b) & 1) + 2 * ((lower >> c) & 1);
}

/**
 * @brief The lower corner of cube edge @p e.
 * @param e Edge index.
 * @return Corner index in [0, 8).
 */
static int edge_lower_corner(int e)
{
  const int a = e / 4;
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  return ((e & 1) << b) | (((e >> 1) & 1) << c);
}

/**
 * @brief Bit mask of the two faces cube edge @p e lies on (face = 2*axis + side).
 * @param e Edge index.
 * @return Mask with two of the six face bits set.
 */
static int edge_faces(int e)
{
  const int a = e / 4;
  const int b = (a + 1) % 3, c = (a + 2) % 3;
  return (1 << (2 * b + (e & 1))) | (1 << (2 * c + ((e >> 1) & 1)));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Table generation
 * ═══════════════════════════════════════════════════════════════════════════ */

static MFTL_MarchingCubesCase g_cases[256];
static int g_cases_ready = 0;

/**
 * @brief Fans loop @p poly from an apex with no in-face diagonal, or from its
 *        centroid when there is none.
 *
 * @param cs    The case being built; triangles are appended.
 * @param poly  The loop's cube edges, in winding order.
 * @param n     Its length, >= 3.
 * @param p     The loop's index within the case, for the centroid vertex.
 */
static void triangulate_loop(MFTL_MarchingCubesCase *cs, const signed char *poly, int n, int p)
{
  if (n == 3)
  {
    cs->tri[cs->n_tris][0] = poly[0];
    cs->tri[cs->n_tris][1] = poly[1];
    cs->tri[cs->n_tris][2] = poly[2];
    cs->n_tris++;
    return;
  }

  for (int apex = 0; apex < n; ++apex)
  {
    bool ok = MFTL_TRUE;
    for (int j = 0; j < n && ok; ++j)
    {
      if (j == apex || j == (apex + 1) % n || j == (apex + n - 1) % n) continue;
      if (edge_faces(poly[apex]) & edge_faces(poly[j])) ok = MFTL_FALSE;
    }
    if (!ok) continue;

    for (int j = 1; j < n - 1; ++j)
    {
      cs->tri[cs->n_tris][0] = poly[apex];
      cs->tri[cs->n_tris][1] = poly[(apex + j) % n];
      cs->tri[cs->n_tris][2] = poly[(apex + j + 1) % n];
      cs->n_tris++;
    }
    return;
  }

  /* No valid apex: a centroid vertex, one triangle per loop edge. */
  for (int j = 0; j < n; ++j)
  {
    cs->tri[cs->n_tris][0] = (signed char) (MFTL_MC_CENTROID + p);
    cs->tri[cs->n_tris][1] = poly[j];
    cs->tri[cs->n_tris][2] = poly[(j + 1) % n];
    cs->n_tris++;
  }
  cs->n_centroids++;
}

/**
 * @brief Builds case @p mask from the face rule (see the file comment).
 * @param      mask Sign pattern, bit c set when corner c is inside.
 * @param[out] cs   The case.
 */
static void build_case(int mask, MFTL_MarchingCubesCase *cs)
{
  memset(cs, 0, sizeof(*cs));

  int next[12];
  for (int e = 0; e < 12; ++e) next[e] = -1;

  for (int f = 0; f < 6; ++f)
  {
    const int a = f / 2, side = f % 2;
    const int b = (a + 1) % 3, c = (a + 2) % 3;

    /* (b, c) offsets counter-clockwise about +a; reversed on the -a face so the
     * walk is counter-clockwise as seen from outside either way. */
    static const int ccw[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
    int cyc[4];
    for (int i = 0; i < 4; ++i)
    {
      const int *o = side ? ccw[i] : ccw[(4 - i) % 4];
      cyc[i] = (side << a) | (o[0] << b) | (o[1] << c);
    }

    for (int i = 0; i < 4; ++i)
    {
      const int s0 = (mask >> cyc[i]) & 1, s1 = (mask >> cyc[(i + 1) % 4]) & 1;
      if (!(s0 == 0 && s1 == 1)) continue;  // not an ENTERING crossing

      for (int j = 1; j < 4; ++j)
      {
        const int u = (i + j) % 4;
        const int t0 = (mask >> cyc[u]) & 1, t1 = (mask >> cyc[(u + 1) % 4]) & 1;
        if (t0 == 1 && t1 == 0)
        {
          const int from = cube_edge(cyc[i], cyc[(i + 1) % 4]);
          const int to = cube_edge(cyc[u], cyc[(u + 1) % 4]);
          MFTL_ASSERT(next[from] == -1 && "marching cubes: a crossing starts two segments");
          next[from] = to;
          break;
        }
      }
    }
  }

  bool used[12] = {MFTL_FALSE};
  for (int e = 0; e < 12; ++e)
  {
    if (next[e] < 0 || used[e]) continue;

    const int p = cs->n_polys++;
    MFTL_ASSERT(p < MFTL_MC_MAX_CASE_POLYGONS);
    int n = 0;
    for (int u = e; !used[u]; u = next[u])
    {
      used[u] = MFTL_TRUE;
      cs->poly[p][n++] = (signed char) u;
      MFTL_ASSERT(next[u] >= 0 && "marching cubes: an open loop");
    }
    cs->poly_len[p] = n;
    triangulate_loop(cs, cs->poly[p], n, p);
  }
}

const MFTL_MarchingCubesCase *MFTLMarchingCubesCase(int mask)
{
  MFTL_ASSERT(mask >= 0 && mask < 256);
  /* Lazy, idempotent initialisation (one thread per rank). */
  if (!g_cases_ready)
  {
    for (int m = 0; m < 256; ++m) build_case(m, &g_cases[m]);
    g_cases_ready = 1;
  }
  return &g_cases[mask & 255];
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Extraction
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Wraps global node index @p g into [0, n_global) on a periodic axis.
 * @param grid The lattice window.
 * @param k    Axis.
 * @param g    Global node index, unwrapped.
 * @return The wrapped index (unchanged on a non-periodic axis).
 */
static inline int wrap_node(const MFTL_SdfGrid *grid, int k, int g)
{
  if (!grid->periodic[k]) return g;
  const int n = grid->n_global[k];
  int w = g % n;
  if (w < 0) w += n;
  return w;
}

/**
 * @brief Linear lattice key of the WRAPPED node (gx, gy, gz).
 * @param grid The lattice window.
 * @param gx   Global node index along x.
 * @param gy   Global node index along y.
 * @param gz   Global node index along z.
 * @return `(z * N_y + y) * N_x + x` over the wrapped indices.
 */
static inline long long node_key(const MFTL_SdfGrid *grid, int gx, int gy, int gz)
{
  const long long x = wrap_node(grid, 0, gx), y = wrap_node(grid, 1, gy), z = wrap_node(grid, 2, gz);
  return (z * grid->n_global[1] + y) * grid->n_global[0] + x;
}

/**
 * @brief Grows the surface's arrays so @p more_verts and @p more_tris more fit.
 * @param s          The surface.
 * @param more_verts Additional vertices.
 * @param more_tris  Additional triangles.
 */
static void surface_reserve(MFTL_LevelSetSurface *s, int more_verts, int more_tris)
{
  if (s->n_verts + more_verts > s->cap_verts)
  {
    int cap = MFTLGrowCapacity(s->cap_verts, s->n_verts + more_verts);
    s->pos = (double *) MFTLReallocOrDie(s->pos, 3 * (size_t) cap * sizeof(double), "level-set surface positions");
    s->vkey = (long long *) MFTLReallocOrDie(s->vkey, (size_t) cap * sizeof(long long), "level-set surface vertex keys");
    s->cap_verts = cap;
  }
  if (s->n_tris + more_tris > s->cap_tris)
  {
    int cap = MFTLGrowCapacity(s->cap_tris, s->n_tris + more_tris);
    s->tri = (int *) MFTLReallocOrDie(s->tri, 3 * (size_t) cap * sizeof(int), "level-set surface triangles");
    s->tcell = (long long *) MFTLReallocOrDie(s->tcell, (size_t) cap * sizeof(long long), "level-set surface cells");
    s->tslot = (signed char *) MFTLReallocOrDie(s->tslot, (size_t) cap * sizeof(signed char), "level-set surface slots");
    s->cap_tris = cap;
  }
}

void MFTLLevelSetSurfaceFree(MFTL_LevelSetSurface *s)
{
  if (s == MFTL_NULL) return;
  free(s->pos);
  free(s->vkey);
  free(s->tri);
  free(s->tcell);
  free(s->tslot);
  memset(s, 0, sizeof(*s));
}

int MFTLExtractLevelSet(const MFTL_SdfGrid *grid, const int cell_lo[3], const int cell_hi[3], double snap, MFTL_LevelSetSurface *out)
{
  MFTL_ASSERT(grid != MFTL_NULL && out != MFTL_NULL && grid->phi != MFTL_NULL);

  for (int k = 0; k < 3; ++k)
  {
    if (cell_hi[k] < cell_lo[k]) return 0;
    MFTL_ASSERT((!grid->periodic[k] || cell_hi[k] - cell_lo[k] < grid->n_global[k]) &&
                "MFTLExtractLevelSet: a cell range longer than one period would extract a cell twice");
  }

  const int nx = grid->dims[0], ny = grid->dims[1], nz = grid->dims[2];
  const size_t n_nodes = (size_t) nx * ny * nz;
  const double snap_abs = snap * fmin(grid->spacing[0], fmin(grid->spacing[1], grid->spacing[2]));

  /* One vertex slot per (axis, local lower node): hash-free vertex sharing. */
  int *const edge_vid = (int *) MFTLMallocOrDie(3 * n_nodes * sizeof(int), "marching cubes edge table");
  for (size_t i = 0; i < 3 * n_nodes; ++i) edge_vid[i] = -1;

  const int first_tri = out->n_tris;

  for (int gz = cell_lo[2]; gz <= cell_hi[2]; ++gz)
  {
    for (int gy = cell_lo[1]; gy <= cell_hi[1]; ++gy)
    {
      for (int gx = cell_lo[0]; gx <= cell_hi[0]; ++gx)
      {
        /* The eight corners, all of which must be in the window and known. */
        double v[8];
        int lidx[8];
        bool usable = MFTL_TRUE;
        for (int c = 0; c < 8 && usable; ++c)
        {
          const int g[3] = {gx + (c & 1), gy + ((c >> 1) & 1), gz + ((c >> 2) & 1)};
          const int l0 = MFTLSdfGridLocal(grid, 0, g[0]);
          const int l1 = MFTLSdfGridLocal(grid, 1, g[1]);
          const int l2 = MFTLSdfGridLocal(grid, 2, g[2]);
          if (l0 < 0 || l1 < 0 || l2 < 0)
          {
            usable = MFTL_FALSE;
            break;
          }
          lidx[c] = (l2 * ny + l1) * nx + l0;
          double val = grid->phi[lidx[c]];
          if (val == MFTL_SDF_UNKNOWN)
          {
            usable = MFTL_FALSE;
            break;
          }
          if (fabs(val) < snap_abs || val == 0.0) val = (val < 0.0) ? -snap_abs : snap_abs;
          if (val == 0.0) val = MFTL_DBL_MIN;  // snap disabled: zero counts as outside
          v[c] = val;
        }
        if (!usable) continue;

        int mask = 0;
        for (int c = 0; c < 8; ++c)
          if (v[c] < 0.0) mask |= 1 << c;
        if (mask == 0 || mask == 255) continue;

        const MFTL_MarchingCubesCase *cs = MFTLMarchingCubesCase(mask);
        const long long cell = node_key(grid, gx, gy, gz);

        int vid[12 + MFTL_MC_MAX_CASE_POLYGONS];
        for (int i = 0; i < 12 + MFTL_MC_MAX_CASE_POLYGONS; ++i) vid[i] = -1;

        surface_reserve(out, 12 + MFTL_MC_MAX_CASE_POLYGONS, cs->n_tris);

        /* Edge vertices of every loop, deduplicated through the edge table. */
        for (int p = 0; p < cs->n_polys; ++p)
        {
          for (int i = 0; i < cs->poly_len[p]; ++i)
          {
            const int e = cs->poly[p][i];
            const int a = e / 4;
            const int c0 = edge_lower_corner(e), c1 = c0 | (1 << a);
            int *const slot = &edge_vid[3 * (size_t) lidx[c0] + a];
            if (*slot < 0)
            {
              /* From the WRAPPED node: two ranks reaching one lattice edge as
               * node 0 and node n across a seam must compute identical bits,
               * or the vertex gets two owners and the surface tears. Where the
               * grid recorded exactly one crossing of this edge, the vertex is
               * put ON it -- clamped by the same snap the values get, so it
               * never lands on a node -- instead of where the linear
               * interpolant of phi vanishes, which on a curved surface is
               * O(dx^2 kappa) off it. */
              double t = v[c0] / (v[c0] - v[c1]);
              if (grid->cross != MFTL_NULL)
              {
                const double x = grid->cross[3 * (size_t) lidx[c0] + a];
                if (x >= 0.0) t = fmin(fmax(x, snap), 1.0 - snap);
              }
              const int g0[3] = {wrap_node(grid, 0, gx + (c0 & 1)), wrap_node(grid, 1, gy + ((c0 >> 1) & 1)), wrap_node(grid, 2, gz + ((c0 >> 2) & 1))};
              double *q = &out->pos[3 * (size_t) out->n_verts];
              for (int k = 0; k < 3; ++k) q[k] = MFTLSdfGridCoord(grid, k, g0[k]);
              q[a] += t * grid->spacing[a];
              out->vkey[out->n_verts] = 3 * node_key(grid, g0[0], g0[1], g0[2]) + a;
              *slot = out->n_verts++;
            }
            vid[e] = *slot;
          }
        }

        /* Centroid vertices, never shared: they belong to one cell's loop. */
        for (int p = 0; p < cs->n_polys && cs->n_centroids > 0; ++p)
        {
          bool needed = MFTL_FALSE;
          for (int t = 0; t < cs->n_tris; ++t)
            for (int j = 0; j < 3; ++j)
              if (cs->tri[t][j] == MFTL_MC_CENTROID + p) needed = MFTL_TRUE;
          if (!needed) continue;

          /* Averaged in one periodic image (wrapped vertices may be a period apart). */
          double *q = &out->pos[3 * (size_t) out->n_verts];
          const double *const first = &out->pos[3 * (size_t) vid[cs->poly[p][0]]];
          q[0] = q[1] = q[2] = 0.0;
          for (int i = 0; i < cs->poly_len[p]; ++i)
            for (int k = 0; k < 3; ++k)
            {
              double x = out->pos[3 * (size_t) vid[cs->poly[p][i]] + k];
              if (grid->periodic[k])
              {
                const double period = grid->n_global[k] * grid->spacing[k];
                x -= period * nearbyint((x - first[k]) / period);
              }
              q[k] += x;
            }
          for (int k = 0; k < 3; ++k) q[k] /= (double) cs->poly_len[p];
          out->vkey[out->n_verts] = -(16 * cell + p) - 2;
          vid[MFTL_MC_CENTROID + p] = out->n_verts++;
        }

        for (int t = 0; t < cs->n_tris; ++t)
        {
          int *const dst = &out->tri[3 * (size_t) out->n_tris];
          for (int j = 0; j < 3; ++j) dst[j] = vid[(int) cs->tri[t][j]];
          out->tcell[out->n_tris] = cell;
          out->tslot[out->n_tris] = (signed char) t;
          out->n_tris++;
        }
      }
    }
  }

  free(edge_vid);
  return out->n_tris - first_tri;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The serial generator
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLBuildMeshFromLevelSet(MFTL_Object *obj, const MFTL_SdfGrid *grid, double snap)
{
  MFTL_ASSERT(obj != MFTL_NULL && grid != MFTL_NULL);
  if (obj == MFTL_NULL || grid == MFTL_NULL || grid->phi == MFTL_NULL) return MFTL_ERROR;
  if (obj->verts.n != 0 || obj->tris.n != 0) return MFTL_ERROR;

  /* Every cell of the window, including the seam-closing cell on a periodic
   * axis whose window is exactly one period. */
  int lo[3], hi[3];
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = grid->lo[k];
    const bool full_period = grid->periodic[k] && grid->dims[k] == grid->n_global[k];
    hi[k] = grid->lo[k] + (full_period ? grid->dims[k] - 1 : grid->dims[k] - 2);
  }

  MFTL_LevelSetSurface s;
  memset(&s, 0, sizeof s);
  MFTLExtractLevelSet(grid, lo, hi, snap, &s);

  int status = MFTL_SUCCESS;
  if (s.n_tris == 0)
  {
    MFTLObjectBuildDerived(obj);
  }
  else
  {
    if (obj->domain != MFTL_NULL)
      for (int v = 0; v < s.n_verts; ++v) MFTLApplyPeriodicity(obj->domain, &s.pos[3 * (size_t) v]);
    status = MFTLBuildIndexedMesh(obj, s.pos, s.n_verts, s.tri, s.n_tris);
  }

  MFTLLevelSetSurfaceFree(&s);
  return status;
}
