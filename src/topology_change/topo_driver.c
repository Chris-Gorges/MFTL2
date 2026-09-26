/**
 * @file topo_driver.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief The two public entry points, in collective order.
 *
 * Every collective below is entered by every rank, in this order, whatever the
 * rank holds. The only branches around a collective are on values every rank
 * computed identically -- the event list, the cluster list, the options -- and
 * never on a local count.
 */

#include "mftl_topology_change.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_communication.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_remeshing.h"
#include "topo_internal.h"

MFTL_TopologyOptions MFTLTopologyDefaultOptions(void)
{
  MFTL_TopologyOptions o;
  memset(&o, 0, sizeof o);
  o.normal_tol_deg = 60.0;
  o.bias_factor = 1.5;
  o.bias_radius_factor = 1.5;
  o.mc_snap = 0.02;
  o.min_body_triangles = 16;
  o.conserve_volume = MFTL_TRUE;
  o.remesh = MFTL_NULL;
  return o;
}

/** @brief Record layout of an event on the wire. */
#define EVENT_STRIDE 7

static bool is_neighbour(const MFTL_Domain *dom, int r)
{
  if (r == dom->rank) return MFTL_TRUE;
  for (int i = 0; i < dom->num_neighbors; ++i)
    if (dom->neighbors[i] == r) return MFTL_TRUE;
  return MFTL_FALSE;
}

/** @brief Whether any destination with a non-zero count is outside the neighbourhood. */
static bool any_far(const MFTL_Domain *dom, const int *counts)
{
  for (int r = 0; r < dom->nproc; ++r)
    if (counts[r] > 0 && !is_neighbour(dom, r)) return MFTL_TRUE;
  return MFTL_FALSE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Detection
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLTopologyDetectEvents(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopologyEvent **events, int *n_events)
{
  if (ens == MFTL_NULL || opts == MFTL_NULL || events == MFTL_NULL || n_events == MFTL_NULL) return MFTL_ERROR;
  *events = MFTL_NULL;
  *n_events = 0;
  const double dmax = MFTL_MAX(opts->delta_breakup, opts->delta_coalescence);
  if (!(dmax > 0.0)) return MFTL_SUCCESS;  // nothing can be detected, on every rank alike

  const MFTL_Domain *const dom = &ens->domain;
  for (int o = 0; o < ens->n_local_objects; ++o)
    if (!MFTLObjectHasTopology(ens->local_objects[o])) MFTLObjectBuildDerived(ens->local_objects[o]);

  MFTL_TopoEventList local = {MFTL_NULL, 0, 0};
  if (MFTLTopoDetectLocal(ens, opts, &local) != MFTL_SUCCESS)
  {
    free(local.e);
    return MFTL_ERROR;
  }

  /* A quarter of the bias bump's scale, max(delta, dx): events closer than that
   * describe one feature and add nothing to the bias but its cost. Keyed on the
   * smaller active threshold ALONE this shredded one neck into 779 events when
   * the other threshold was ten times smaller. */
  double quantum = dmax;
  if (opts->delta_breakup > 0.0) quantum = fmin(quantum, opts->delta_breakup);
  if (opts->delta_coalescence > 0.0) quantum = fmin(quantum, opts->delta_coalescence);
  quantum = 0.25 * fmax(quantum, opts->dx);

  MFTLTopoCanonicaliseEvents(dom, quantum, &local);

  double *const mine = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(local.n, 1) * EVENT_STRIDE * sizeof(double), "topology event records");
  for (int i = 0; i < local.n; ++i)
  {
    double *const r = &mine[EVENT_STRIDE * i];
    r[0] = local.e[i].body_a;
    r[1] = local.e[i].body_b;
    r[2] = local.e[i].kind;
    r[3] = local.e[i].point[0];
    r[4] = local.e[i].point[1];
    r[5] = local.e[i].point[2];
    r[6] = local.e[i].gap;
  }
  free(local.e);

  int n_all = 0;
  double *const all = MFTLTopoCommAllgather(dom, mine, local.n, EVENT_STRIDE, &n_all);
  free(mine);

  MFTL_TopoEventList list = {MFTL_NULL, 0, 0};
  list.cap = MFTL_MAX(n_all, 1);
  list.e = (MFTL_TopologyEvent *) MFTLMallocOrDie((size_t) list.cap * sizeof(MFTL_TopologyEvent), "topology events");
  for (int i = 0; i < n_all; ++i)
  {
    const double *const r = &all[EVENT_STRIDE * i];
    MFTL_TopologyEvent *const e = &list.e[list.n++];
    e->body_a = (int) r[0];
    e->body_b = (int) r[1];
    e->kind = (int) r[2];
    e->point[0] = r[3];
    e->point[1] = r[4];
    e->point[2] = r[5];
    e->gap = r[6];
  }
  free(all);

  /* Once more over the whole list: now identical on every rank. */
  MFTLTopoCanonicaliseEvents(dom, quantum, &list);

  if (list.n == 0)
  {
    free(list.e);
    return MFTL_SUCCESS;
  }
  *events = list.e;
  *n_events = list.n;
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Reconstruction
 * ═══════════════════════════════════════════════════════════════════════════ */

static int cmp_int_asc(const void *a, const void *b)
{
  const int x = *(const int *) a, y = *(const int *) b;
  return (x > y) - (x < y);
}

static int created_index(const int *created, int n, int gid)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    const int mid = (lo + hi) / 2;
    if (created[mid] == gid) return mid;
    if (created[mid] < gid)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return -1;
}

/** @brief The cluster a body id came out of, or -1 if it is not a new body. */
static int came_from(const MFTL_TopoComponent *comps, int n, int gid)
{
  for (int j = 0; j < n; ++j)
    if (comps[j].body == gid) return comps[j].cluster;
  return -1;
}

/**
 * @brief The size floor again, on what the remesher left: a closed body whose
 *        marching-cubes surface cleared it can still be collapsed below it.
 *
 * A satellite a couple of lattice cells across comes out of marching cubes with
 * a few dozen slivers, and a target edge length comparable to its diameter
 * collapses that to a tetrahedron -- which the collapse approval then refuses to
 * dissolve, so it survives as four triangles every one of which faces against
 * all three of its neighbours. Measured: the pinch-off in the Rayleigh-Taylor
 * example at np = 4. The count is the owned triangles summed over ranks, one
 * value per component of THIS reconstruction (the list every rank already
 * agrees on), so every holder takes the same decision. A deleted body's volume
 * goes to its cluster's surviving bodies through the correction that follows.
 */
static void discard_collapsed(MFTL_Ensemble *ens, MFTL_TopoComponent *comps, int n_comps, int min_tris, int first_new, MFTL_TopologyStats *st)
{
  if (min_tris <= 0 || n_comps == 0) return;
  double *const n = (double *) MFTLCallocOrDie((size_t) n_comps, sizeof(double), "topology remeshed sizes");
  for (int j = 0; j < n_comps; ++j)
  {
    if (comps[j].body < 0 || comps[j].open) continue;
    const int slot = MFTLEnsembleFindLocalSlot(ens, comps[j].body);
    if (slot != MFTL_ERROR) n[j] = ens->local_objects[slot]->tris.n_local;
  }
  MFTLTopoCommAllreduceDouble(&ens->domain, n, n_comps, 2);

  for (int j = 0; j < n_comps; ++j)
  {
    if (comps[j].body < 0 || comps[j].open || n[j] >= (double) min_tris) continue;
    const int slot = MFTLEnsembleFindLocalSlot(ens, comps[j].body);
    if (slot != MFTL_ERROR)
    {
      MFTL_Object *obj = ens->local_objects[slot];
      MFTLEnsembleEraseLocalSlot(ens, slot);
      MFTLObjectDestroy(&obj);
    }
    st->n_bodies_discarded++;
    st->n_bodies_out--;
    if (comps[j].body >= first_new)
      st->n_bodies_created--;  // its id was handed out and is never used
    else
      st->n_bodies_retired++;
    comps[j].body = -1;
  }
  free(n);
}

/**
 * @brief The volume correction's displacement field, left in `verts.normal`,
 *        and the derivative of the enclosed volume along it.
 *
 * u is the area-weighted vertex normal; a rim vertex loses the component normal
 * to the wall it is on, so it slides in the wall and the rim stays there. The
 * return value is this rank's share of dV/dd = sum_t S_t . (u_0 + u_1 + u_2)/3
 * (S_t the triangle's area vector), taken over the (triangle, corner) pairs whose
 * corner this rank OWNS: the halo is vertex-complete, so every such pair is
 * present exactly once in the job. Corners are gathered seam-consistently, so
 * it holds on a periodic domain. It is the first variation of the volume for a
 * closed body and for one closed by the walls alike, because the walls do not
 * move.
 */
static double displacement_field(MFTL_Object *obj)
{
  const MFTL_Domain *const dom = obj->domain;
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    if (!(obj->verts.rec[v].flags & MFTL_FLAG_BOUNDARY)) continue;
    const double *const p = &obj->verts.pos[3 * v];
    for (int k = 0; k < 3; ++k)
    {
      if (dom->is_periodic[k]) continue;
      const double tol = 1e-8 * dom->domain_size[k];
      if (fabs(p[k] - dom->global_min[k]) <= tol || fabs(p[k] - dom->global_max[k]) <= tol) obj->verts.normal[3 * v + k] = 0.0;
    }
  }

  double dvdd = 0.0;
  for (int t = 0; t < obj->tris.n; ++t)
  {
    double T[3][3];
    MFTLGetTriangleCorners(obj, t, T);
    const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
    const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
    const double S[3] = {0.5 * (e1[1] * e2[2] - e1[2] * e2[1]), 0.5 * (e1[2] * e2[0] - e1[0] * e2[2]), 0.5 * (e1[0] * e2[1] - e1[1] * e2[0])};
    for (int k = 0; k < 3; ++k)
    {
      const int v = obj->tris.verts[3 * t + k];
      if (v >= obj->verts.n_local) continue;
      const double *const u = &obj->verts.normal[3 * v];
      dvdd += (S[0] * u[0] + S[1] * u[1] + S[2] * u[2]) / 3.0;
    }
  }
  return dvdd;
}

/** @brief Geometry and derived arrays current for every local object: what redistribution needs. */
static void refresh_all(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

int MFTLEnsembleReconstructBodies(MFTL_Ensemble *ens, const int *bodies_in, int n_bodies_in, const MFTL_TopologyEvent *events, int n_events,
                                  const MFTL_TopologyOptions *opts, MFTL_TopologyStats *stats)
{
  MFTL_TopologyStats sink;
  MFTL_TopologyStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));

  if (ens == MFTL_NULL || opts == MFTL_NULL || n_bodies_in < 0 || (n_bodies_in > 0 && bodies_in == MFTL_NULL)) return MFTL_ERROR;
  if (!(opts->dx > 0.0)) return MFTL_ERROR;
  if (n_bodies_in == 0) return MFTL_SUCCESS;  // identical on every rank: the list is an agreed input

  MFTL_Domain *const dom = &ens->domain;
  const double t0 = MFTLTopoCommTime();
  st->n_events = n_events;
  for (int e = 0; e < n_events; ++e)
  {
    if (events[e].kind == MFTL_TOPOLOGY_BREAKUP)
      st->n_breakup++;
    else
      st->n_coalescence++;
  }

  /* ── The lattice and the clusters ─────────────────────────────────────── */
  MFTL_TopoLattice lat;
  MFTLTopoLatticeInit(&lat, dom, opts->dx);
  MFTLTopoCommCoords(dom, lat.coords);
  for (int k = 0; k < 3; ++k) st->lattice_cells[k] = lat.cells[k];

  MFTL_TopoCluster *cl = MFTL_NULL;
  const int n_cl = MFTLTopoBuildClusters(bodies_in, n_bodies_in, events, n_events, &cl);

  /* The sorted body list and the global size of each body. */
  int *const bodies = (int *) MFTLMallocOrDie((size_t) n_bodies_in * sizeof(int), "topology body list");
  memcpy(bodies, bodies_in, (size_t) n_bodies_in * sizeof(int));
  qsort(bodies, (size_t) n_bodies_in, sizeof(int), cmp_int_asc);
  int nb = 0;
  for (int i = 0; i < n_bodies_in; ++i)
    if (nb == 0 || bodies[nb - 1] != bodies[i]) bodies[nb++] = bodies[i];

  /* One message for two questions: per body its triangle count (SUM), per
   * cluster whether some body has a rim off the walls (SUM of flags). */
  long long *const sizes = (long long *) MFTLCallocOrDie((size_t) (nb + n_cl), sizeof(long long), "topology body sizes");
  for (int i = 0; i < nb; ++i)
  {
    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, bodies[i]);
    if (obj == MFTL_NULL) continue;
    sizes[i] = obj->tris.n_local;
    if (!MFTLTopoRimsOnWalls(obj)) sizes[nb + MFTLTopoClusterOf(cl, n_cl, bodies[i])] = 1;
  }
  MFTLTopoCommAllreduceLongLong(dom, sizes, nb + n_cl, 2);
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].refused = sizes[nb + c] > 0;
    if (cl[c].refused)
      st->n_clusters_refused++;
    else
      st->n_clusters++;
  }

  /* Which walls each cluster's rims lie on: six MAX slots per cluster. */
  long long *const walls = (long long *) MFTLCallocOrDie(6 * (size_t) MFTL_MAX(n_cl, 1), sizeof(long long), "topology rim walls");
  for (int i = 0; i < nb; ++i)
  {
    const MFTL_Object *const obj = MFTLEnsembleFindLocalObject(ens, bodies[i]);
    if (obj == MFTL_NULL) continue;
    const int c = MFTLTopoClusterOf(cl, n_cl, bodies[i]);
    const int m = MFTLTopoRimWalls(obj);
    for (int b = 0; b < 6; ++b)
      if (m & (1 << b)) walls[6 * c + b] = 1;
  }
  MFTLTopoCommAllreduceLongLong(dom, walls, 6 * n_cl, 1);
  for (int c = 0; c < n_cl; ++c)
  {
    cl[c].rim_walls = 0;
    for (int b = 0; b < 6; ++b)
      if (walls[6 * c + b]) cl[c].rim_walls |= 1 << b;
  }
  free(walls);

  /* ── Pseudo-normals: angle-weighted, exact on ghosts too ──────────────────
   *
   * Only the cluster bodies are recomputed; the exchange is ensemble-wide, which
   * for every other body re-copies its owners' current normals onto its ghosts
   * and changes nothing an owner holds. */
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    const int c = MFTLTopoClusterOf(cl, n_cl, obj->global_id);
    if (c < 0 || cl[c].refused) continue;
    if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_ANGLE);
  }
  MFTLUpdateGhostVertexNormals(ens);

  /* ── Extents, cutoff, pad, windows ───────────────────────────────────────── */
  double *const ext = (double *) MFTLMallocOrDie(12 * (size_t) MFTL_MAX(n_cl, 1) * sizeof(double), "topology extents");
  MFTLTopoLocalExtents(ens, cl, n_cl, ext);
  for (int c = 0; c < n_cl; ++c)
    if (cl[c].refused)
      for (int i = 0; i < 12; ++i) ext[12 * c + i] = MFTL_DBL_MAX;
  MFTLTopoCommAllreduceDouble(dom, ext, 12 * n_cl, 0);
  MFTLTopoResolveExtents(dom, cl, n_cl, ext);
  free(ext);

  /* The volume each cluster encloses now, for the correction at the end. */
  MFTLTopoChooseFlux(dom, cl, n_cl);
  {
    double *const vol = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(n_cl, 1), sizeof(double), "topology volumes");
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      const MFTL_Object *const obj = ens->local_objects[o];
      const int c = MFTLTopoClusterOf(cl, n_cl, obj->global_id);
      if (c < 0 || cl[c].refused || cl[c].flux_axis < 0) continue;
      double v = 0.0, a = 0.0;
      MFTLTopoObjectVolume(obj, &cl[c], &v, &a);
      vol[c] += v;
    }
    MFTLTopoCommAllreduceDouble(dom, vol, n_cl, 2);
    for (int c = 0; c < n_cl; ++c)
    {
      cl[c].volume_old = vol[c];
      if (!cl[c].refused) st->volume_before += vol[c];
    }
    free(vol);
  }

  const double dx_max = fmax(lat.spacing[0], fmax(lat.spacing[1], lat.spacing[2]));
  const double dx_min = fmin(lat.spacing[0], fmin(lat.spacing[1], lat.spacing[2]));
  const double diag = sqrt(lat.spacing[0] * lat.spacing[0] + lat.spacing[1] * lat.spacing[1] + lat.spacing[2] * lat.spacing[2]);
  lat.r_cut = diag + MFTLTopoMaxBias(opts) + 0.5 * dx_max;

  double overhang = MFTLTopoLocalOverhang(ens, cl, n_cl, &lat);
  MFTLTopoCommAllreduceDouble(dom, &overhang, 1, 1);
  lat.pad = (int) ceil((overhang + lat.r_cut) / dx_min) + 1;

  MFTLTopoMakeWindows(dom, &lat, cl, n_cl);

  /* ── The field ───────────────────────────────────────────────────────────── */
  MFTLTopoRasterise(ens, cl, n_cl, &lat);
  {
    int *const dest = (int *) MFTLMallocOrDie((size_t) dom->nproc * sizeof(int), "topology node destinations");
    double *records = MFTL_NULL;
    const bool far = MFTLTopoPackNodeExchange(dom, &lat, cl, n_cl, dest, &records);
    int n_recv = 0;
    double *const recv = MFTLTopoCommExchange(dom, records, dest, 3, far, &n_recv, MFTL_NULL);
    MFTLTopoMergeNodes(&lat, cl, n_cl, recv, n_recv);
    free(recv);
    free(records);
    free(dest);
  }
  MFTLTopoApplyBias(dom, &lat, cl, n_cl, events, opts);

  /* ── The surface ─────────────────────────────────────────────────────────── */
  MFTLTopoExtract(dom, &lat, cl, n_cl, opts->mc_snap);
  for (int c = 0; c < n_cl; ++c)
  {
    MFTLSdfGridFree(&cl[c].grid);  // the field is no longer needed
  }
  const double t_rec = MFTLTopoCommTime();

  /* ── Components ──────────────────────────────────────────────────────────── */
  MFTL_TopoPiece *const pieces = (MFTL_TopoPiece *) MFTLCallocOrDie((size_t) MFTL_MAX(n_cl, 1), sizeof(MFTL_TopoPiece), "topology pieces");
  MFTLTopoBuildPieces(dom, &lat, cl, n_cl, pieces);
  for (int c = 0; c < n_cl; ++c) MFTLLevelSetSurfaceFree(&cl[c].surf);

  int *const dest = (int *) MFTLMallocOrDie(2 * (size_t) dom->nproc * sizeof(int), "topology label destinations");
  int *const src = dest + dom->nproc;
  for (;;)
  {
    bool changed = MFTLTopoLabelLocal(pieces, n_cl);

    double *req = MFTL_NULL;
    MFTLTopoPackLabelRequests(pieces, n_cl, dom->nproc, dest, &req);
    const bool far = any_far(dom, dest);
    int n_req = 0;
    double *const got = MFTLTopoCommExchange(dom, req, dest, 3, far, &n_req, src);
    double *const answer = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(double), "topology label answers");
    if (MFTLTopoServeLabelRequests(pieces, n_cl, got, n_req, answer)) changed = MFTL_TRUE;
    double *const back = MFTLTopoCommReply(dom, answer, src, dest, 1, far);
    if (MFTLTopoApplyLabelReplies(pieces, n_cl, dom->nproc, back)) changed = MFTL_TRUE;
    free(back);
    free(answer);
    free(got);
    free(req);

    st->n_label_rounds++;
    if (!MFTLTopoCommAny(dom, changed)) break;
  }

  /* ── Every component's fate, decided identically everywhere ──────────────── */
  double *local_rec = MFTL_NULL;
  const int n_local_rec = MFTLTopoComponentRecords(&lat, pieces, n_cl, &local_rec);
  int n_all = 0;
  double *const all_rec = MFTLTopoCommAllgather(dom, local_rec, n_local_rec, 4, &n_all);
  free(local_rec);

  MFTL_TopoComponent *comps = MFTL_NULL;
  int n_bodies_global = ens->n_bodies_global;
  const int first_new = n_bodies_global;
  const int n_comps =
      MFTLTopoDecideComponents(all_rec, n_all, cl, n_cl, bodies, sizes, nb, MFTL_MAX(opts->min_body_triangles, 0), &n_bodies_global, &comps, st);
  free(all_rec);
  ens->n_bodies_global = n_bodies_global;

  /* ── The objects ─────────────────────────────────────────────────────────── */
  int *created = MFTL_NULL;
  long long **vkeys = MFTL_NULL, **tkeys = MFTL_NULL;
  const int n_created = MFTLTopoBuildObjects(ens, &lat, cl, n_cl, pieces, comps, n_comps, &created, &vkeys, &tkeys);
  MFTLTopoFreePieces(pieces, n_cl);
  free(pieces);

  /* Dense ids: a base per holder from the library's per-body scan. Every local
   * object takes part, contributing zero unless it was just built. */
  {
    const int nl = ens->n_local_objects;
    double *const val = (double *) MFTLCallocOrDie(2 * (size_t) MFTL_MAX(nl, 1), sizeof(double), "topology id counts");
    double *const tot = (double *) MFTLCallocOrDie(2 * (size_t) MFTL_MAX(nl, 1), sizeof(double), "topology id totals");
    for (int o = 0; o < nl; ++o)
    {
      const MFTL_Object *const obj = ens->local_objects[o];
      if (created_index(created, n_created, obj->global_id) < 0) continue;
      val[2 * o] = obj->verts.n_local;
      val[2 * o + 1] = obj->tris.n_local;
    }
    MFTLEnsembleExscanBodies(ens, val, tot, 2, MFTL_BODY_REDUCE_SCOPE_AUTO);

    for (int o = 0; o < nl; ++o)
    {
      MFTL_Object *const obj = ens->local_objects[o];
      if (created_index(created, n_created, obj->global_id) < 0) continue;
      const int vb = (int) val[2 * o], tb = (int) val[2 * o + 1];
      for (int v = 0; v < obj->verts.n_local; ++v) obj->verts.rec[v].global_id = vb + v;
      for (int t = 0; t < obj->tris.n_local; ++t) obj->tris.rec[t].global_id = tb + t;
      obj->next_vertex_gid = (int) tot[2 * o];
      obj->next_triangle_gid = (int) tot[2 * o + 1];
      obj->next_halfedge_gid = 3 * obj->next_triangle_gid;
      obj->global_n_vertices = obj->next_vertex_gid;
      obj->global_n_triangles = obj->next_triangle_gid;
    }
    free(val);
    free(tot);
  }
  {
    double *req = MFTL_NULL;
    MFTLTopoPackIdRequests(ens, created, n_created, vkeys, tkeys, dom->nproc, dest, &req);
    const bool far = any_far(dom, dest);
    int n_req = 0;
    double *const got = MFTLTopoCommExchange(dom, req, dest, 3, far, &n_req, src);
    double *const answer = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(n_req, 1) * sizeof(double), "topology id answers");
    MFTLTopoServeIdRequests(ens, created, n_created, vkeys, tkeys, got, n_req, answer);
    double *const back = MFTLTopoCommReply(dom, answer, src, dest, 1, far);
    MFTLTopoApplyIdReplies(ens, created, n_created, dom->nproc, back);
    free(back);
    free(answer);
    free(got);
    free(req);
  }
  for (int i = 0; i < n_created; ++i)
  {
    free(vkeys[i]);
    free(tkeys[i]);
  }
  free(vkeys);
  free(tkeys);

  free(created);
  free(dest);
  free(sizes);
  free(bodies);
  const double t_sep = MFTLTopoCommTime();

  /* ── Back to the canonical partition ───────────────────────────────────── */
  refresh_all(ens);
  int status = MFTLRedistribute(ens);
  refresh_all(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  const double t_red = MFTLTopoCommTime();

  if (status == MFTL_SUCCESS && opts->remesh != MFTL_NULL)
  {
    MFTL_RemeshStats rst;
    status = MFTLRemeshEnsemble(ens, opts->remesh, &rst);
    discard_collapsed(ens, comps, n_comps, MFTL_MAX(opts->min_body_triangles, 0), first_new, st);
  }

  /* ── Put the volume back ──────────────────────────────────────────────────
   *
   * LAST, after the remesh: a freshly extracted surface is all slivers, and
   * the pass that cleans them up smooths away about half a percent of a jet's
   * volume on its own -- measured, when the correction ran before it, as a
   * steady -0.5 per reconstruction the correction was blind to.
   *
   * Measured per cluster over the bodies that came out of it, restored by a
   * uniform offset d along a displacement field u: the area-weighted normals of
   * the OWNED vertices -- complete fans, the canonical halo guarantees it --
   * with a rim vertex's wall-normal component removed. The ghost copies are
   * then refreshed from their owners exactly as a solver's own vertex motion
   * would be. The step is Newton's: V(d) is a cubic in d whose derivative at 0
   * is the first variation of the enclosed volume, sum over triangles of
   * S_t . mean(u) (S_t the area vector), which each rank sums over the corners
   * it owns. Dividing by the AREA instead -- dV = A d, exact only when u is the
   * face normal -- converges linearly at about 0.03 per step. Measured on a
   * satellite remeshed at a target edge length near its size (a 15 % deficit):
   * 1.3e-4 / 3e-6 / 8e-8 after two / three / four area steps, 2.8e-5 / 1e-9
   * after two / three Newton steps; the pinched dumbbell reaches 1e-14 after
   * three. A fixed count, so every rank runs the same collectives: a stopping
   * test on a reduced double is two ranks disagreeing in the last ulp about
   * whether to enter another. */
  {
    double *const va = (double *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(n_cl, 1) * sizeof(double), "topology volume and derivative");
    const int n_iter = opts->conserve_volume ? 3 : 0;
    for (int iter = 0; iter <= n_iter; ++iter)
    {
      for (int i = 0; i < 2 * n_cl; ++i) va[i] = 0.0;
      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        MFTL_Object *const obj = ens->local_objects[o];
        const int c = came_from(comps, n_comps, obj->global_id);
        if (c < 0 || cl[c].flux_axis < 0) continue;
        double v = 0.0, a = 0.0;
        MFTLTopoObjectVolume(obj, &cl[c], &v, &a);
        va[2 * c] += v;
        if (iter < n_iter) va[2 * c + 1] += displacement_field(obj);
      }
      MFTLTopoCommAllreduceDouble(dom, va, 2 * n_cl, 2);

      double total = 0.0;
      for (int c = 0; c < n_cl; ++c)
        if (!cl[c].refused) total += va[2 * c];
      if (iter == 0) st->volume_raw = total;
      st->volume_after = total;
      if (iter == n_iter) break;

      for (int o = 0; o < ens->n_local_objects; ++o)
      {
        MFTL_Object *const obj = ens->local_objects[o];
        const int c = came_from(comps, n_comps, obj->global_id);
        if (c < 0 || cl[c].flux_axis < 0 || !(va[2 * c + 1] > 0.0)) continue;
        const double d = (cl[c].volume_old - va[2 * c]) / va[2 * c + 1];
        for (int v = 0; v < obj->verts.n_local; ++v)
        {
          double *const p = &obj->verts.pos[3 * v];
          for (int k = 0; k < 3; ++k) p[k] += d * obj->verts.normal[3 * v + k];
          MFTLApplyPeriodicity(dom, p);
        }
      }
      MFTLUpdateGhostPositions(ens);
    }
    free(va);
    refresh_all(ens);
  }
  free(comps);
  MFTLTopoFreeClusters(cl, n_cl);
  const double t_end = MFTLTopoCommTime();

  double t[5] = {t_rec - t0, t_sep - t_rec, t_red - t_sep, t_end - t_red, t_end - t0};
  MFTLTopoCommAllreduceDouble(dom, t, 5, 1);
  st->t_reconstruct = t[0];
  st->t_separate = t[1];
  st->t_redistribute = t[2];
  st->t_remesh = t[3];
  st->t_total = t[4];
  return status;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. The one-call form
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLEnsembleApplyTopologyChanges(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopologyStats *stats)
{
  MFTL_TopologyStats sink;
  MFTL_TopologyStats *const st = stats ? stats : &sink;
  memset(st, 0, sizeof(*st));
  if (ens == MFTL_NULL || opts == MFTL_NULL) return MFTL_ERROR;

  const double t0 = MFTLTopoCommTime();
  MFTL_TopologyEvent *events = MFTL_NULL;
  int n_events = 0;
  if (MFTLTopologyDetectEvents(ens, opts, &events, &n_events) != MFTL_SUCCESS) return MFTL_ERROR;
  double t_detect = MFTLTopoCommTime() - t0;
  MFTLTopoCommAllreduceDouble(&ens->domain, &t_detect, 1, 1);

  /* No event, no change: nothing below runs, on any rank. */
  if (n_events == 0)
  {
    st->t_detect = t_detect;
    st->t_total = t_detect;
    return MFTL_SUCCESS;
  }

  int *const bodies = (int *) MFTLMallocOrDie(2 * (size_t) n_events * sizeof(int), "topology event bodies");
  for (int e = 0; e < n_events; ++e)
  {
    bodies[2 * e] = events[e].body_a;
    bodies[2 * e + 1] = events[e].body_b;
  }

  const int status = MFTLEnsembleReconstructBodies(ens, bodies, 2 * n_events, events, n_events, opts, st);
  free(bodies);
  free(events);
  st->t_detect = t_detect;
  st->t_total += t_detect;
  return status;
}
