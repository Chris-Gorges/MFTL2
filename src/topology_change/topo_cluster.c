/**
 * @file topo_cluster.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief The reconstruction lattice, the clusters, and each rank's window.
 *
 * ## ONE LATTICE, ALIGNED TO EVERYTHING
 *
 * The lattice divides every axis into `cuts[k] * m` cells. Two alignments
 * follow, and every other file relies on them:
 *
 *   A NODE PLANE ON EVERY WALL. A surface meeting a wall -- the rim of an open
 *   interface -- is cut by marching cubes exactly in the wall plane, so the
 *   reconstructed rim lies ON the wall rather than a fraction of a cell away.
 *
 *   A NODE PLANE ON EVERY CUT PLANE. Every lattice cell lies inside exactly one
 *   rank box, so "who owns this cell" is integer arithmetic and every triangle
 *   marching cubes produces has exactly one owner, whatever rank computed it.
 *
 * ## A WINDOW, NOT A GRID
 *
 * No rank stores the lattice. It stores a WINDOW per cluster: its own box, plus
 * a pad wide enough that every owned triangle's rasterisation footprint fits,
 * intersected with the cluster's extent. That is the original design's "component grid"
 * per interacting cluster, cut into per-rank pieces -- memory follows the part
 * of the cluster near this rank, never the domain.
 */

#include "topo_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The lattice
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopologyLattice(const MFTL_Domain *domain, double dx, double spacing[3], int cells[3])
{
  MFTL_ASSERT(domain != MFTL_NULL && dx > 0.0);
  for (int k = 0; k < 3; ++k)
  {
    const double L = domain->domain_size[k];
    const int cuts = domain->cuts[k];
    int per = (int) floor(L / ((double) cuts * dx) + 0.5);
    if (per < 1) per = 1;
    cells[k] = per * cuts;
    spacing[k] = L / (double) cells[k];
  }
}

void MFTLTopoLatticeInit(MFTL_TopoLattice *lat, const MFTL_Domain *dom, double dx)
{
  memset(lat, 0, sizeof(*lat));
  MFTLTopologyLattice(dom, dx, lat->spacing, lat->cells);
  for (int k = 0; k < 3; ++k)
  {
    lat->origin[k] = dom->global_min[k];
    lat->periodic[k] = dom->is_periodic[k];
    lat->n_global[k] = lat->periodic[k] ? lat->cells[k] : lat->cells[k] + 1;
    lat->per_rank[k] = lat->cells[k] / dom->cuts[k];
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Clusters
 * ═══════════════════════════════════════════════════════════════════════════ */

static int uf_find(int *parent, int i)
{
  while (parent[i] != i)
  {
    parent[i] = parent[parent[i]];
    i = parent[i];
  }
  return i;
}

int MFTLTopoBuildClusters(const int *bodies_in, int n_in, const MFTL_TopologyEvent *events, int n_events, MFTL_TopoCluster **out)
{
  /* Sorted, unique body list: the cluster order must not depend on the order
   * the caller listed them in. */
  int *const b = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * sizeof(int), "topology bodies");
  memcpy(b, bodies_in, (size_t) n_in * sizeof(int));
  MFTLSortIds(b, n_in);
  const int nb = MFTLUniqueIntArray(b, n_in);

  int *const parent = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nb, 1) * sizeof(int), "topology union-find");
  for (int i = 0; i < nb; ++i) parent[i] = i;

  /* ANY event between two bodies joins them, whatever its kind. Coalescence is
   * the obvious case; the other is a liquid film between two surfaces that bound
   * the same liquid from opposite sides -- a drop and a bubble inside it. Each
   * body's field alone would erode the film from its own side only, and the
   * film opens only in the union. Union by the smaller index keeps the root
   * deterministic. */
  for (int e = 0; e < n_events; ++e)
  {
    if (events[e].body_a == events[e].body_b) continue;
    const int ia = MFTLSortedIndexOf(b, nb, events[e].body_a), ib = MFTLSortedIndexOf(b, nb, events[e].body_b);
    if (ia < 0 || ib < 0) continue;
    const int ra = uf_find(parent, ia), rb = uf_find(parent, ib);
    if (ra < rb)
      parent[rb] = ra;
    else if (rb < ra)
      parent[ra] = rb;
  }

  /* Clusters in order of their smallest body, which is their root. */
  int *const cl_of = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nb, 1) * sizeof(int), "topology cluster index");
  int n_cl = 0;
  for (int i = 0; i < nb; ++i) cl_of[i] = (uf_find(parent, i) == i) ? n_cl++ : -1;
  for (int i = 0; i < nb; ++i) cl_of[i] = cl_of[uf_find(parent, i)];

  MFTL_TopoCluster *const cl = (MFTL_TopoCluster *) MFTLCallocOrDie((size_t) MFTL_MAX(n_cl, 1), sizeof(MFTL_TopoCluster), "topology clusters");
  for (int i = 0; i < nb; ++i) cl[cl_of[i]].n_bodies++;
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].bodies = (int *) MFTLMallocOrDie((size_t) cl[c].n_bodies * sizeof(int), "topology cluster bodies");
    cl[c].n_bodies = 0;
  }
  for (int i = 0; i < nb; ++i) cl[cl_of[i]].bodies[cl[cl_of[i]].n_bodies++] = b[i];

  /* Events to the cluster of their first body. */
  int *const ev_cl = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_events, 1) * sizeof(int), "topology event clusters");
  for (int e = 0; e < n_events; ++e)
  {
    const int ia = MFTLSortedIndexOf(b, nb, events[e].body_a);
    ev_cl[e] = (ia >= 0) ? cl_of[ia] : -1;
    if (ev_cl[e] >= 0) cl[ev_cl[e]].n_events++;
  }
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].events = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(cl[c].n_events, 1) * sizeof(int), "topology cluster events");
    cl[c].n_events = 0;
  }
  for (int e = 0; e < n_events; ++e)
    if (ev_cl[e] >= 0) cl[ev_cl[e]].events[cl[ev_cl[e]].n_events++] = e;

  free(ev_cl);
  free(cl_of);
  free(parent);
  free(b);
  *out = cl;
  return n_cl;
}

void MFTLTopoFreeClusters(MFTL_TopoCluster *cl, int n)
{
  if (cl == MFTL_NULL) return;
  for (int c = 0; c < n; ++c)
  {
    free(cl[c].bodies);
    free(cl[c].events);
    MFTLSdfGridFree(&cl[c].grid);
    MFTLLevelSetSurfaceFree(&cl[c].surf);
  }
  free(cl);
}

int MFTLTopoClusterOf(const MFTL_TopoCluster *cl, int n, int gid)
{
  for (int c = 0; c < n; ++c)
    if (MFTLSortedIndexOf(cl[c].bodies, cl[c].n_bodies, gid) >= 0) return c;
  return -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Extents: centre and half-width on the circle
 *
 * The four-frame measurement of src/geometry/periodic_extent.c: a cluster
 * straddling a periodic seam has a raw min/max box spanning the whole domain,
 * and in a frame whose cut lies in its gap it does not. Every frame is reduced
 * -- as a MIN of the lower bound and a MIN of the NEGATED upper bound, so one
 * reduction does all -- and MFTLPeriodicExtentResolve() picks the exact one.
 * A cluster narrower than 3L/4 is always measured exactly; a wider one is
 * treated as covering the axis, which costs it nothing but a whole-axis window
 * and that axis as a flux axis (MFTLTopoChooseFlux()).
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopoLocalExtents(const MFTL_Ensemble *ens, const MFTL_TopoCluster *cl, int n_cl, double *out)
{
  for (int c = 0; c < n_cl; ++c) MFTLPeriodicExtentInit(&out[MFTL_PERIODIC_EXTENT_SLOTS * c]);

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const int c = MFTLTopoClusterOf(cl, n_cl, obj->global_id);
    if (c < 0) continue;
    MFTLPeriodicExtentAddObject(obj, &out[MFTL_PERIODIC_EXTENT_SLOTS * c]);
  }
}

void MFTLTopoResolveExtents(const MFTL_Domain *dom, MFTL_TopoCluster *cl, int n_cl, const double *reduced)
{
  for (int c = 0; c < n_cl; ++c)
  {
    MFTL_PeriodicExtent x;
    MFTLPeriodicExtentResolve(dom, &reduced[MFTL_PERIODIC_EXTENT_SLOTS * c], &x);
    cl[c].empty = x.empty;
    for (int k = 0; k < 3; ++k)
    {
      cl[c].centre[k] = x.centre[k];
      cl[c].half[k] = x.half[k];
      cl[c].full[k] = x.covers[k];
    }
  }
}

double MFTLTopoLocalOverhang(const MFTL_Ensemble *ens, const MFTL_TopoCluster *cl, int n_cl, const MFTL_TopoLattice *lat)
{
  const MFTL_Domain *const dom = &ens->domain;
  double worst = 0.0;

  double box_lo[3], box_hi[3];
  for (int k = 0; k < 3; ++k)
  {
    box_lo[k] = lat->origin[k] + (double) (lat->coords[k] * lat->per_rank[k]) * lat->spacing[k];
    box_hi[k] = box_lo[k] + (double) lat->per_rank[k] * lat->spacing[k];
  }

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    if (MFTLTopoClusterOf(cl, n_cl, obj->global_id) < 0) continue;

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double T[3][3];
      MFTLGetTriangleCorners(obj, t, T);
      for (int j = 0; j < 3; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          /* The corner in the image nearest the box centre. */
          double x = T[j][k];
          if (dom->is_periodic[k])
          {
            const double L = dom->domain_size[k], mid = 0.5 * (box_lo[k] + box_hi[k]);
            x -= L * nearbyint((x - mid) / L);
          }
          const double out = fmax(box_lo[k] - x, x - box_hi[k]);
          if (out > worst) worst = out;
        }
      }
    }
  }
  return worst;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Windows
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLTopoMakeWindows(const MFTL_Domain *dom, const MFTL_TopoLattice *lat, MFTL_TopoCluster *cl, int n_cl)
{
  (void) dom;
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].has_window = MFTL_FALSE;
    if (cl[c].refused || cl[c].empty) continue;

    int lo[3], dims[3];
    bool empty = MFTL_FALSE;

    for (int k = 0; k < 3 && !empty; ++k)
    {
      const int n = lat->per_rank[k], N = lat->cells[k];
      int wlo = lat->coords[k] * n - lat->pad;
      int whi = (lat->coords[k] + 1) * n + lat->pad;  // inclusive node index

      if (lat->periodic[k])
      {
        if (whi - wlo + 1 >= N)
        {
          wlo = lat->coords[k] * n;
          whi = wlo + N - 1;
        }
        else if (!cl[c].full[k])
        {
          /* The cluster's node range, moved to the image nearest the window. */
          const double margin = lat->r_cut + lat->spacing[k];
          int clo = (int) floor((cl[c].centre[k] - cl[c].half[k] - margin - lat->origin[k]) / lat->spacing[k]);
          int chi = (int) ceil((cl[c].centre[k] + cl[c].half[k] + margin - lat->origin[k]) / lat->spacing[k]);
          if ((chi - clo + 1) + (whi - wlo + 1) <= N)
          {
            const int shift = (int) floor(((0.5 * (wlo + whi)) - 0.5 * (clo + chi)) / N + 0.5);
            clo += shift * N;
            chi += shift * N;
            wlo = MFTL_MAX(wlo, clo);
            whi = MFTL_MIN(whi, chi);
            if (whi < wlo) empty = MFTL_TRUE;
          }
        }
      }
      else
      {
        wlo = MFTL_MAX(wlo, 0);
        whi = MFTL_MIN(whi, N);
        const double margin = lat->r_cut + lat->spacing[k];
        const int clo = (int) floor((cl[c].centre[k] - cl[c].half[k] - margin - lat->origin[k]) / lat->spacing[k]);
        const int chi = (int) ceil((cl[c].centre[k] + cl[c].half[k] + margin - lat->origin[k]) / lat->spacing[k]);
        wlo = MFTL_MAX(wlo, clo);
        whi = MFTL_MIN(whi, chi);
        if (whi < wlo) empty = MFTL_TRUE;
      }
      lo[k] = wlo;
      dims[k] = whi - wlo + 1;
    }
    if (empty) continue;

    MFTLSdfGridInitWindow(&cl[c].grid, lat->origin, lat->spacing, lat->n_global, lat->periodic, lo, dims);
    cl[c].has_window = MFTL_TRUE;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. The one shape a level set cannot reconstruct
 * ═══════════════════════════════════════════════════════════════════════════ */

bool MFTLTopoRimsOnWalls(const MFTL_Object *obj)
{
  const MFTL_Domain *const dom = obj->domain;
  const int nh = 3 * obj->tris.n_local;
  for (int h = 0; h < nh; ++h)
  {
    if (!(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY)) continue;
    const double *const a = MFTLVertexPos(obj, obj->tris.he[h].vertex_idx);
    const double *const b = MFTLVertexPos(obj, obj->tris.he[MFTLHePrevIdx(h)].vertex_idx);

    /* On a wall: both ends on the same non-periodic boundary plane, to the
     * tolerance the examples snap rims to. */
    bool on_wall = MFTL_FALSE;
    for (int k = 0; k < 3 && !on_wall; ++k)
    {
      if (dom == MFTL_NULL || dom->is_periodic[k]) continue;
      const double tol = 1e-8 * dom->domain_size[k];
      if (fabs(a[k] - dom->global_min[k]) <= tol && fabs(b[k] - dom->global_min[k]) <= tol) on_wall = MFTL_TRUE;
      if (fabs(a[k] - dom->global_max[k]) <= tol && fabs(b[k] - dom->global_max[k]) <= tol) on_wall = MFTL_TRUE;
    }
    if (!on_wall) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Volume, for the correction that keeps a reconstruction from eating it
 *
 * A reconstructed surface loses volume: marching cubes cuts chords across a
 * convex surface, O(dx^2 * curvature) per unit area, and on a whole jet that is
 * about one percent PER reconstruction -- measured at -60 of a final 81 over a
 * run of 45 of them before this correction existed. The liquid a cluster
 * encloses is therefore measured before and after and restored.
 *
 * The measure is the flux of F = (x_k - ref) e_k, which has divergence one, so
 * for a closed surface it IS the enclosed volume. For an open surface whose rim
 * lies on walls it is the volume enclosed together with those walls, provided
 * no wall contributes: a wall perpendicular to another axis has F . n = 0, and
 * the wall x_k = ref has F = 0. So k is a non-periodic axis whose far wall
 * carries no rim -- the jet's nozzle wall is x_min and k = x; a sheet with its
 * rim on the four side walls takes k = z. A closed cluster takes any axis it
 * does not wrap around, with x_k unwrapped about its centre.
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLTopoRimWalls(const MFTL_Object *obj)
{
  const MFTL_Domain *const dom = obj->domain;
  int mask = 0;
  const int nh = 3 * obj->tris.n_local;
  for (int h = 0; h < nh; ++h)
  {
    if (!(obj->tris.he[h].flags & MFTL_FLAG_BOUNDARY)) continue;
    const double *const a = MFTLVertexPos(obj, obj->tris.he[h].vertex_idx);
    for (int k = 0; k < 3; ++k)
    {
      if (dom == MFTL_NULL || dom->is_periodic[k]) continue;
      const double tol = 1e-8 * dom->domain_size[k];
      if (fabs(a[k] - dom->global_min[k]) <= tol) mask |= 1 << (2 * k);
      if (fabs(a[k] - dom->global_max[k]) <= tol) mask |= 1 << (2 * k + 1);
    }
  }
  return mask;
}

void MFTLTopoChooseFlux(const MFTL_Domain *dom, MFTL_TopoCluster *cl, int n_cl)
{
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].flux_axis = -1;
    cl[c].flux_ref = 0.0;

    if (cl[c].rim_walls == 0)
    {
      /* Closed: any axis the cluster does not wrap, measured about its centre. */
      for (int k = 0; k < 3 && cl[c].flux_axis < 0; ++k)
        if (!cl[c].full[k])
        {
          cl[c].flux_axis = k;
          cl[c].flux_ref = cl[c].centre[k];
        }
      continue;
    }

    /* Open: a non-periodic axis with no rim on its FAR wall. The near wall, if
     * a rim is on it, is where F vanishes. */
    for (int k = 0; k < 3 && cl[c].flux_axis < 0; ++k)
    {
      if (dom->is_periodic[k]) continue;
      if (cl[c].rim_walls & (1 << (2 * k + 1))) continue;
      cl[c].flux_axis = k;
      cl[c].flux_ref = dom->global_min[k];
    }
  }
}

void MFTLTopoObjectVolume(const MFTL_Object *obj, const MFTL_TopoCluster *c, double *volume, double *area)
{
  const MFTL_Domain *const dom = obj->domain;
  const int k = c->flux_axis;
  double v = 0.0, a = 0.0;

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    const double n[3] = {0.5 * (e1[1] * e2[2] - e1[2] * e2[1]), 0.5 * (e1[2] * e2[0] - e1[0] * e2[2]), 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])};
    a += sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (k < 0) continue;

    double x = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
    if (dom != MFTL_NULL && dom->is_periodic[k])
    {
      /* The image nearest the cluster centre: a closed cluster that crosses the
       * seam on this axis is measured in one piece. */
      const double L = dom->domain_size[k];
      x -= L * nearbyint((x - c->centre[k]) / L);
    }
    v += (x - c->flux_ref) * n[k];
  }
  *volume = v;
  *area = a;
}
