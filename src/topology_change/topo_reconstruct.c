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
 * per boundary node instead of a triangle per nearby triangle -- plus one per
 * lattice edge a triangle crosses there, the slot that lets marching cubes put
 * its vertex on the old surface (MFTLRasterizeEdgeCrossings(), merged with
 * MFTLSdfCrossMerge(), which is order-free in the same way).
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

/** @brief Bias magnitude of event @p e: half its threshold times the bias factor. */
static inline double event_bias(const MFTL_TopologyEvent *e, const MFTL_TopologyOptions *opts)
{
  const double delta = (e->kind == MFTL_TOPOLOGY_BREAKUP) ? opts->delta_breakup : opts->delta_coalescence;
  return opts->bias_factor * 0.5 * delta;
}

/** @brief Radius of event @p e's bias bump: nonzero strictly inside it. */
static inline double event_radius(const MFTL_TopologyEvent *e, const MFTL_TopologyOptions *opts, double dx_max)
{
  const double delta = (e->kind == MFTL_TOPOLOGY_BREAKUP) ? opts->delta_breakup : opts->delta_coalescence;
  return opts->bias_radius_factor * fmax(delta, dx_max);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Rasterisation and the overlap exchange
 *
 * ## TWO BANDS, AND WHY THE RESULT IS THE SAME BITS AS ONE
 *
 * The cutoff exists so that marching cubes finds every corner of every cell
 * the BIASED zero level set crosses. A node x left unknown is harmless exactly
 * when every cell with x as a corner has one biased sign at all eight corners,
 * and those corners lie within one cell diagonal D of x. So:
 *
 *   away from every bias, D + dx/2 suffices: a corner y of such a cell is
 *   farther than dx/2 from the surface, on x's side of it, and unbiased;
 *
 *   where a bias of up to eps can reach a corner, the band must be eps wider,
 *   D + eps + dx/2 (lat->r_cut), or the bias could flip a corner it was never
 *   told about.
 *
 * "Can reach" is the ZONE: nodes within rho_e + D of an event point e, rho_e
 * being its bump radius -- every cell with a biased corner has all its corners
 * in it. Inside the zone the wide band, elsewhere the narrow one, and a node's
 * value is the merge over exactly the triangles within its own cutoff
 * (MFTLRasterizeSignedDistanceBanded()). Its nearest triangle is always one of
 * them, so every node that gets a value gets the value one band of r_cut gave,
 * to the bit, and a node it no longer gets lies only in cells of one sign that
 * marching cubes would have left empty. The surface is unchanged; what changes
 * is that a triangle away from every event visits ((h + 2 r_far)/(h + 2 r_cut))^3
 * of the nodes -- about a third at the default bias -- and the exchange ships a
 * thinner shell. Rasterisation was 91 % of the reconstruction phase.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The wide-band zone of one cluster's window: a byte per node, nonzero
 *        within rho_e + D of one of its events (every image that reaches the
 *        window), walked exactly as MFTLTopoApplyBias() walks the bump.
 */
static unsigned char *wide_zone(const MFTL_TopoLattice *lat, const MFTL_TopoCluster *c, const MFTL_TopologyEvent *events, const MFTL_TopologyOptions *opts,
                                double dx_max, double diag)
{
  const MFTL_SdfGrid *const g = &c->grid;
  const size_t n = (size_t) g->dims[0] * g->dims[1] * g->dims[2];
  unsigned char *const zone = (unsigned char *) MFTLCallocOrDie(n, 1, "topology wide-band zone");

  for (int i = 0; i < c->n_events; ++i)
  {
    const MFTL_TopologyEvent *const e = &events[c->events[i]];
    /* A hundredth of a cell of slack: the zone may only ever be too wide. */
    const double r = event_radius(e, opts, dx_max) + diag + 0.01 * dx_max, r_sq = r * r;
    int lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = (int) floor((e->point[k] - r - lat->origin[k]) / lat->spacing[k]);
      hi[k] = (int) ceil((e->point[k] + r - lat->origin[k]) / lat->spacing[k]);
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
          if (x * x + y * y + z * z <= r_sq) zone[((size_t) l2 * g->dims[1] + l1) * g->dims[0] + l0] = 1;
        }
      }
    }
  }
  return zone;
}

/**
 * @brief Flags every owned triangle of @p obj that can come within the wide
 *        cutoff of a zone node: its centroid within rho_e + D + r_cut + its own
 *        bounding radius of an event, minimum image.
 *
 * Only the efficiency depends on how tight this is: a triangle flagged without
 * need visits the wide box and writes the narrow band outside the zone anyway.
 *
 * @return `[tris.n_local]` flags, or #MFTL_NULL when no triangle needs one.
 */
static unsigned char *wide_triangles(const MFTL_Domain *dom, const MFTL_Object *obj, const MFTL_TopoCluster *c, const MFTL_TopologyEvent *events,
                                     const MFTL_TopologyOptions *opts, double dx_max, double diag, double r_cut)
{
  const int nt = obj->tris.n_local;
  if (nt == 0 || c->n_events == 0) return MFTL_NULL;
  unsigned char *const flag = (unsigned char *) MFTLCallocOrDie((size_t) nt, 1, "topology wide-band triangles");
  bool any = MFTL_FALSE;

  for (int t = 0; t < nt; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    double ctr[3], rad_sq = 0.0;
    for (int k = 0; k < 3; ++k) ctr[k] = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
    for (int j = 0; j < 3; ++j)
    {
      const double dx = T[j][0] - ctr[0], dy = T[j][1] - ctr[1], dz = T[j][2] - ctr[2];
      rad_sq = fmax(rad_sq, dx * dx + dy * dy + dz * dz);
    }
    const double reach = sqrt(rad_sq) + diag + r_cut + 0.01 * dx_max;

    for (int i = 0; i < c->n_events; ++i)
    {
      const MFTL_TopologyEvent *const e = &events[c->events[i]];
      const double r = reach + event_radius(e, opts, dx_max);
      double d[3];
      MFTLGetMinimumImageVector(dom, d, ctr, e->point);
      if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] <= r * r)
      {
        flag[t] = 1;
        any = MFTL_TRUE;
        break;
      }
    }
  }
  if (!any)
  {
    free(flag);
    return MFTL_NULL;
  }
  return flag;
}

void MFTLTopoRasterise(const MFTL_Ensemble *ens, MFTL_TopoCluster *cl, int n_cl, const MFTL_TopoLattice *lat, const MFTL_TopologyEvent *events,
                       const MFTL_TopologyOptions *opts)
{
  const double dx_max = fmax(lat->spacing[0], fmax(lat->spacing[1], lat->spacing[2]));
  const double diag = sqrt(lat->spacing[0] * lat->spacing[0] + lat->spacing[1] * lat->spacing[1] + lat->spacing[2] * lat->spacing[2]);

  unsigned char **const zone = (unsigned char **) MFTLCallocOrDie((size_t) MFTL_MAX(n_cl, 1), sizeof(unsigned char *), "topology wide-band zones");
  for (int c = 0; c < n_cl; ++c)
  {
    if (!cl[c].has_window) continue;
    MFTLSdfGridEnableCrossings(&cl[c].grid);  // every window: a rank with no triangle of the cluster still receives crossings
    if (cl[c].n_events > 0) zone[c] = wide_zone(lat, &cl[c], events, opts, dx_max, diag);
  }

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const int c = MFTLTopoClusterOf(cl, n_cl, obj->global_id);
    if (c < 0 || !cl[c].has_window) continue;
    /* OWNED triangles only: each triangle is rasterised by exactly one rank in
     * the job, so the merge sees every contribution exactly once. */
    unsigned char *const wide = (zone[c] != MFTL_NULL) ? wide_triangles(&ens->domain, obj, &cl[c], events, opts, dx_max, diag, lat->r_cut) : MFTL_NULL;
    MFTLRasterizeSignedDistanceBanded(&cl[c].grid, obj, obj->tris.n_local, lat->r_far, lat->r_cut, zone[c], wide);
    MFTLRasterizeEdgeCrossings(&cl[c].grid, obj, obj->tris.n_local);
    free(wide);
  }

  for (int c = 0; c < n_cl; ++c) free(zone[c]);
  free(zone);
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
            const size_t idx = ((size_t) lz * ny + ly) * nx + lx;
            const double v = g->phi[idx];
            const double *const cx3 = (g->cross != MFTL_NULL) ? &g->cross[3 * idx] : MFTL_NULL;
            const bool has_cross = cx3 != MFTL_NULL && (cx3[0] != MFTL_SDF_CROSS_NONE || cx3[1] != MFTL_SDF_CROSS_NONE || cx3[2] != MFTL_SDF_CROSS_NONE);
            if (v == MFTL_SDF_UNKNOWN && !has_cross) continue;

            const int gw[3] = {wrap_index(lat, 0, g->lo[0] + lx), wrap_index(lat, 1, g->lo[1] + ly), wrap_index(lat, 2, g->lo[2] + lz)};
            int cx[4], cy[4], cz[4];
            const int mx = needing_cells(dom, lat, 0, gw[0], cx);
            const int my = needing_cells(dom, lat, 1, gw[1], cy);
            const int mz = needing_cells(dom, lat, 2, gw[2], cz);

            /* Records carry a TAGGED key, 4 * node + kind: kind 0 is the
             * node's value, kind 1 + k the crossing slot of its edge along k.
             * An edge travels with its lower node, whose need range contains
             * that of every rank extracting a cell of the edge. */
            const long long key = ((long long) gw[2] * lat->n_global[1] + gw[1]) * lat->n_global[0] + gw[0];
            int n_rec = (v != MFTL_SDF_UNKNOWN);
            if (has_cross)
              for (int k = 0; k < 3; ++k) n_rec += (cx3[k] != MFTL_SDF_CROSS_NONE);

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
                    dest_counts[r] += n_rec;
                    if (!MFTLTopoIsNeighbour(dom, r)) far = MFTL_TRUE;
                  }
                  else
                  {
                    for (int kind = 0; kind < 4; ++kind)
                    {
                      const double val = (kind == 0) ? v : (has_cross ? cx3[kind - 1] : MFTL_SDF_CROSS_NONE);
                      if (kind == 0 ? (val == MFTL_SDF_UNKNOWN) : (val == MFTL_SDF_CROSS_NONE)) continue;
                      double *const rec = &buf[3 * (size_t) cursor[r]++];
                      rec[0] = (double) c;
                      rec[1] = (double) (4 * key + kind);
                      rec[2] = val;
                    }
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

    const long long tagged = (long long) records[3 * i + 1];
    const int kind = (int) (tagged % 4);
    const long long key = tagged / 4;
    const int g[3] = {(int) (key % nx), (int) ((key / nx) % ny), (int) (key / (nx * ny))};
    MFTL_SdfGrid *const grid = &cl[c].grid;
    const int l0 = MFTLSdfGridLocal(grid, 0, g[0]), l1 = MFTLSdfGridLocal(grid, 1, g[1]), l2 = MFTLSdfGridLocal(grid, 2, g[2]);
    if (l0 < 0 || l1 < 0 || l2 < 0) continue;  // outside this rank's window: no cell of its needs it

    const size_t idx = ((size_t) l2 * grid->dims[1] + l1) * grid->dims[0] + l0;
    if (kind == 0)
      grid->phi[idx] = MFTLSdfMerge(grid->phi[idx], records[3 * i + 2]);
    else if (grid->cross != MFTL_NULL)
      grid->cross[3 * idx + kind - 1] = MFTLSdfCrossMerge(grid->cross[3 * idx + kind - 1], records[3 * i + 2]);
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
      const double eps = event_bias(e, opts);
      const double rho = event_radius(e, opts, dx_max);  // the zone of MFTLTopoRasterise() is this plus a diagonal
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

    /* A biased edge's zero is wherever the bias moved it, not where the old
     * surface crosses: every edge with a biased end interpolates. Read off the
     * bias arrays, not off phi, so every rank holding the edge decides alike. */
    if (g->cross != MFTL_NULL)
    {
      const size_t stride[3] = {1, (size_t) g->dims[0], (size_t) g->dims[0] * g->dims[1]};
      for (int lz = 0; lz < g->dims[2]; ++lz)
        for (int ly = 0; ly < g->dims[1]; ++ly)
          for (int lx = 0; lx < g->dims[0]; ++lx)
          {
            const size_t i = lz * stride[2] + ly * stride[1] + (size_t) lx;
            if (open[i] == 0.0 && close[i] == 0.0) continue;
            const int l[3] = {lx, ly, lz};
            for (int k = 0; k < 3; ++k)
            {
              g->cross[3 * i + k] = MFTL_SDF_CROSS_MANY;  // the edge it is the lower end of
              /* ... and the edge it is the upper end of, wrapping only when the
               * window holds the whole period on this axis. */
              if (l[k] > 0)
                g->cross[3 * (i - stride[k]) + k] = MFTL_SDF_CROSS_MANY;
              else if (g->periodic[k] && g->dims[k] == g->n_global[k])
                g->cross[3 * (i + (size_t) (g->dims[k] - 1) * stride[k]) + k] = MFTL_SDF_CROSS_MANY;
            }
          }
    }

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
