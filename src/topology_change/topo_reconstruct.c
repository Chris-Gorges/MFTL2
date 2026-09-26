/**
 * @file topo_reconstruct.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief The field and the surface: rasterisation, the overlap exchange, the
 *        bias, and marching cubes on this rank's cells.
 *
 * ## WHY EACH RANK RASTERISES ONLY WHAT IT OWNS
 *
 * The value at a lattice node is the minimum over EVERY triangle within the
 * cutoff, and those triangles may be owned by any of up to eight ranks around
 * the node. Two ways to get it:
 *
 *   PULL the triangles: grow a spatial halo of every neighbour's triangles near
 *   this box and rasterise everything locally. Triangles near a corner are
 *   shipped to seven ranks and rasterised by eight.
 *
 *   PUSH the values: each rank rasterises the triangles it OWNS -- each exactly
 *   once in the job -- onto a padded window, and sends every node another rank
 *   needs. MFTLSdfMerge() is commutative and associative, so the merged value is
 *   the same whatever order the contributions arrive in, and bit-identical to a
 *   serial rasterisation of the whole mesh.
 *
 * PUSH is what a CFD code does with a ghost-cell halo, and it moves one double
 * per boundary node instead of a triangle per nearby triangle.
 *
 * ## WHICH NODES A RANK NEEDS
 *
 * Its own box of nodes and one more layer on every side: marching cubes runs on
 * its own cells AND one layer of the neighbours' cells, so that the triangles
 * around every vertex it owns are present -- the vertex-complete halo, built by
 * the extraction rather than by a later exchange.
 */

#include "topo_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Lattice arithmetic
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline int wrap_index(const MFTL_TopoLattice *lat, int k, int g)
{
  if (!lat->periodic[k]) return g;
  const int n = lat->n_global[k];
  int w = g % n;
  if (w < 0) w += n;
  return w;
}

int MFTLTopoCellOwner(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, long long cell)
{
  const long long nx = lat->n_global[0], ny = lat->n_global[1];
  const int ci = (int) (cell % nx), cj = (int) ((cell / nx) % ny), ck = (int) (cell / (nx * ny));
  const int c[3] = {ci / lat->per_rank[0], cj / lat->per_rank[1], ck / lat->per_rank[2]};
  return dom->rank_map[c[0] + c[1] * dom->cuts[0] + c[2] * dom->cuts[0] * dom->cuts[1]];
}

double MFTLTopoMaxBias(const MFTL_TopologyOptions *opts) { return opts->bias_factor * 0.5 * MFTL_MAX(opts->delta_breakup, opts->delta_coalescence); }

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Rasterisation and the overlap exchange
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopoRasterise(const MFTL_Ensemble *ens, MFTL_TopoCluster *cl, int n_cl, const MFTL_TopoLattice *lat)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const int c = MFTLTopoClusterOf(cl, n_cl, obj->global_id);
    if (c < 0 || !cl[c].has_window) continue;
    /* OWNED triangles only: each triangle is rasterised by exactly one rank in
     * the job, so the merge sees every contribution exactly once. */
    MFTLRasterizeSignedDistance(&cl[c].grid, obj, obj->tris.n_local, lat->r_cut);
  }
}

/**
 * @brief The rank-cells along one axis whose NEED range -- box nodes plus one
 *        layer -- contains wrapped node @p g.
 * @return How many were written to @p out (at most 4).
 */
static int needing_cells(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, int k, int g, int out[4])
{
  const int n = lat->per_rank[k], cuts = dom->cuts[k], N = lat->cells[k];
  int m = 0;
  for (int c = g / n - 2; c <= g / n + 2 && m < 4; ++c)
  {
    int cw = c;
    if (lat->periodic[k])
    {
      cw %= cuts;
      if (cw < 0) cw += cuts;
    }
    else if (c < 0 || c >= cuts)
      continue;

    const int lo = cw * n - 1, hi = (cw + 1) * n + 1;  // inclusive need range, unwrapped
    bool in;
    if (lat->periodic[k])
    {
      int d = (g - lo) % N;
      if (d < 0) d += N;
      in = d <= hi - lo || (hi - lo + 1 >= N);
    }
    else
      in = g >= lo && g <= hi;
    if (!in) continue;

    bool dup = MFTL_FALSE;
    for (int i = 0; i < m; ++i)
      if (out[i] == cw) dup = MFTL_TRUE;
    if (!dup) out[m++] = cw;
  }
  return m;
}

/** @brief Whether @p r is this rank or one of its graph neighbours. */
static bool is_neighbour(const MFTL_Domain *dom, int r)
{
  if (r == dom->rank) return MFTL_TRUE;
  for (int i = 0; i < dom->num_neighbors; ++i)
    if (dom->neighbors[i] == r) return MFTL_TRUE;
  return MFTL_FALSE;
}

bool MFTLTopoPackNodeExchange(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, const MFTL_TopoCluster *cl, int n_cl, int *dest_counts, double **records)
{
  const int np = dom->nproc;
  for (int r = 0; r < np; ++r) dest_counts[r] = 0;
  bool far = MFTL_FALSE;

  /* Two passes over the known nodes: count per destination, then fill. The
   * node lists are short -- a band of r_cut around the surface -- so walking
   * twice beats a growable buffer per destination. */
  int *cursor = MFTL_NULL;
  double *buf = MFTL_NULL;

  for (int pass = 0; pass < 2; ++pass)
  {
    for (int c = 0; c < n_cl; ++c)
    {
      if (!cl[c].has_window) continue;
      const MFTL_SdfGrid *const g = &cl[c].grid;
      const int nx = g->dims[0], ny = g->dims[1], nz = g->dims[2];

      for (int lz = 0; lz < nz; ++lz)
        for (int ly = 0; ly < ny; ++ly)
          for (int lx = 0; lx < nx; ++lx)
          {
            const double v = g->phi[((size_t) lz * ny + ly) * nx + lx];
            if (v == MFTL_SDF_UNKNOWN) continue;

            const int gw[3] = {wrap_index(lat, 0, g->lo[0] + lx), wrap_index(lat, 1, g->lo[1] + ly), wrap_index(lat, 2, g->lo[2] + lz)};
            int cx[4], cy[4], cz[4];
            const int mx = needing_cells(dom, lat, 0, gw[0], cx);
            const int my = needing_cells(dom, lat, 1, gw[1], cy);
            const int mz = needing_cells(dom, lat, 2, gw[2], cz);

            const double key = (double) (((long long) gw[2] * lat->n_global[1] + gw[1]) * lat->n_global[0] + gw[0]);

            /* At most 64 (cell, cell, cell) combinations, few of them distinct
             * ranks; de-duplicate so a rank is sent a node once. */
            int sent_to[64];
            int n_sent = 0;
            for (int a = 0; a < mx; ++a)
              for (int b = 0; b < my; ++b)
                for (int d = 0; d < mz; ++d)
                {
                  const int r = dom->rank_map[cx[a] + cy[b] * dom->cuts[0] + cz[d] * dom->cuts[0] * dom->cuts[1]];
                  if (r == dom->rank) continue;
                  bool dup = MFTL_FALSE;
                  for (int i = 0; i < n_sent; ++i)
                    if (sent_to[i] == r) dup = MFTL_TRUE;
                  if (dup) continue;
                  sent_to[n_sent++] = r;

                  if (pass == 0)
                  {
                    dest_counts[r]++;
                    if (!is_neighbour(dom, r)) far = MFTL_TRUE;
                  }
                  else
                  {
                    double *const rec = &buf[3 * (size_t) cursor[r]++];
                    rec[0] = (double) c;
                    rec[1] = key;
                    rec[2] = v;
                  }
                }
          }
    }

    if (pass == 0)
    {
      int total = 0;
      cursor = (int *) MFTLMallocOrDie((size_t) np * sizeof(int), "topology node cursors");
      for (int r = 0; r < np; ++r)
      {
        cursor[r] = total;
        total += dest_counts[r];
      }
      buf = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(total, 1) * sizeof(double), "topology node records");
    }
  }

  free(cursor);
  *records = buf;
  return far;
}

void MFTLTopoMergeNodes(const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, const double *records, int n_records)
{
  const long long nx = lat->n_global[0], ny = lat->n_global[1];
  for (int i = 0; i < n_records; ++i)
  {
    const int c = (int) records[3 * i];
    MFTL_ASSERT(c >= 0 && c < n_cl);
    if (c < 0 || c >= n_cl || !cl[c].has_window) continue;

    const long long key = (long long) records[3 * i + 1];
    const int g[3] = {(int) (key % nx), (int) ((key / nx) % ny), (int) (key / (nx * ny))};
    MFTL_SdfGrid *const grid = &cl[c].grid;
    const int l0 = MFTLSdfGridLocal(grid, 0, g[0]), l1 = MFTLSdfGridLocal(grid, 1, g[1]), l2 = MFTLSdfGridLocal(grid, 2, g[2]);
    if (l0 < 0 || l1 < 0 || l2 < 0) continue;  // outside this rank's window: no cell of its needs it

    double *const node = &grid->phi[((size_t) l2 * grid->dims[1] + l1) * grid->dims[0] + l0];
    *node = MFTLSdfMerge(*node, records[3 * i + 2]);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The bias
 *
 * A breakup event opens the thread it sits in: +eps_b on the field, which moves
 * the zero level set INTO the liquid by eps_b and cuts any thread thinner than
 * 2 * eps_b. A coalescence event closes the film: -eps_c, which fills any gap
 * thinner than 2 * eps_c. The bump is 1 within half its radius of the event
 * point and falls smoothly to 0 at the radius, so the reconstructed surface
 * rejoins the unbiased one without a step.
 *
 * Overlapping bumps of one sign are combined by MAX rather than summed: two
 * events a lattice cell apart describe the same feature and must not open it
 * twice as wide.
 * ═══════════════════════════════════════════════════════════════════════════ */

static inline double bump(double s)
{
  if (s <= 0.5) return 1.0;
  if (s >= 1.0) return 0.0;
  const double t = (s - 0.5) / 0.5;
  return 1.0 - t * t * (3.0 - 2.0 * t);
}

void MFTLTopoApplyBias(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, const MFTL_TopologyEvent *events,
                       const MFTL_TopologyOptions *opts)
{
  const double dx_max = fmax(lat->spacing[0], fmax(lat->spacing[1], lat->spacing[2]));

  for (int c = 0; c < n_cl; ++c)
  {
    if (!cl[c].has_window || cl[c].n_events == 0) continue;
    MFTL_SdfGrid *const g = &cl[c].grid;
    const size_t n = (size_t) g->dims[0] * g->dims[1] * g->dims[2];

    double *const open = (double *) MFTLCallocOrDie(n, sizeof(double), "topology bias open");
    double *const close = (double *) MFTLCallocOrDie(n, sizeof(double), "topology bias close");

    for (int i = 0; i < cl[c].n_events; ++i)
    {
      const MFTL_TopologyEvent *const e = &events[cl[c].events[i]];
      const bool breakup = (e->kind == MFTL_TOPOLOGY_BREAKUP);
      const double delta = breakup ? opts->delta_breakup : opts->delta_coalescence;
      const double eps = opts->bias_factor * 0.5 * delta;
      const double rho = opts->bias_radius_factor * fmax(delta, dx_max);
      double *const acc = breakup ? open : close;

      /* Every image of the event point that reaches the window. */
      int lo[3], hi[3];
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = (int) ceil((e->point[k] - rho - lat->origin[k]) / lat->spacing[k]);
        hi[k] = (int) floor((e->point[k] + rho - lat->origin[k]) / lat->spacing[k]);
      }
      for (int gz = lo[2]; gz <= hi[2]; ++gz)
      {
        const int l2 = MFTLSdfGridLocal(g, 2, gz);
        if (l2 < 0) continue;
        const double z = MFTLSdfGridCoord(g, 2, gz) - e->point[2];
        for (int gy = lo[1]; gy <= hi[1]; ++gy)
        {
          const int l1 = MFTLSdfGridLocal(g, 1, gy);
          if (l1 < 0) continue;
          const double y = MFTLSdfGridCoord(g, 1, gy) - e->point[1];
          for (int gx = lo[0]; gx <= hi[0]; ++gx)
          {
            const int l0 = MFTLSdfGridLocal(g, 0, gx);
            if (l0 < 0) continue;
            const double x = MFTLSdfGridCoord(g, 0, gx) - e->point[0];
            const double w = bump(sqrt(x * x + y * y + z * z) / rho);
            const size_t idx = ((size_t) l2 * g->dims[1] + l1) * g->dims[0] + l0;
            if (eps * w > acc[idx]) acc[idx] = eps * w;
          }
        }
      }
    }

    for (size_t i = 0; i < n; ++i)
      if (g->phi[i] != MFTL_SDF_UNKNOWN) g->phi[i] += open[i] - close[i];

    free(open);
    free(close);
  }
  (void) dom;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Marching cubes on this rank's cells plus one layer
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopoExtract(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl, double snap)
{
  (void) dom;
  int lo[3], hi[3];
  for (int k = 0; k < 3; ++k)
  {
    const int n = lat->per_rank[k], N = lat->cells[k];
    lo[k] = lat->coords[k] * n - 1;
    hi[k] = (lat->coords[k] + 1) * n;  // inclusive: own cells plus one on each side
    if (lat->periodic[k])
    {
      if (hi[k] - lo[k] + 1 > N)
      {
        /* One rank across the whole periodic axis: every cell is its own, and
         * the layer would wrap onto them. */
        lo[k] = lat->coords[k] * n;
        hi[k] = lo[k] + N - 1;
      }
    }
    else
    {
      lo[k] = MFTL_MAX(lo[k], 0);
      hi[k] = MFTL_MIN(hi[k], N - 1);
    }
  }

  for (int c = 0; c < n_cl; ++c)
  {
    if (!cl[c].has_window) continue;
    MFTLExtractLevelSet(&cl[c].grid, lo, hi, snap, &cl[c].surf);
  }
}
