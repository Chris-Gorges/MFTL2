/*
 * Stage 16C — generalized (boundary-aware) remeshing support (plans/16 §4.3, §4.4).
 *
 *  - remesh-mode selection per body (§4.4.4): open bodies always use the generalized remesher,
 *    closed bodies only by explicit opt-in (MFTL_REMESH_MODE_GENERALIZED); closed bodies in the
 *    default mode keep the Stage-15 path bit-identically;
 *  - boundary-curve model per boundary tag (§4.3): REPRESENTED (interpolating Catmull-Rom spline
 *    through the loop's current vertices; corners are spline breakpoints) or CONSTRAINED
 *    (application projector); corners (turning angle > boundary_corner_angle, default 30 deg, or
 *    the MFTL_VFLAG_CORNER tag) are never moved, merged or collapsed;
 *  - the augmented link condition of Dey et al. (§4.4.1): one virtual vertex w adjacent to every
 *    boundary vertex and a virtual triangle (u, v, w) per boundary edge; the collapse of ab is
 *    topologically legal iff Lk(a) n Lk(b) = Lk(ab) (vertices and edges);
 *  - survivor / new-vertex placement: a boundary-edge collapse or split places the vertex on the
 *    boundary curve; an interior edge with one boundary endpoint keeps the boundary vertex at its
 *    position (16A S-A5);
 *  - boundary-vertex tangential smoothing along the boundary curve (arc-length redistribution);
 *  - valence targets at boundary vertices for flips (round(theta / 60 deg) + 1, minimum 2).
 *
 * All geometry is evaluated in local frames (minimum-image differences from an anchor vertex),
 * so seam-crossing boundaries behave exactly as translated ones (§4.1).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_remeshing.h"
#include "mftl_boundary_internal.h"

/* ------------------------------------------------------------------------------------------
 * mode selection and boundary-curve registry
 * ------------------------------------------------------------------------------------------ */
void MFTLObjectSetRemeshMode(MFTL_Object* obj, int mode) { obj->remesh_mode = (mode == MFTL_REMESH_MODE_GENERALIZED) ? mode : MFTL_REMESH_MODE_DEFAULT; }

int MFTLObjectRemeshGeneralized(const MFTL_Object* obj) { return obj->declared_open || obj->remesh_mode == MFTL_REMESH_MODE_GENERALIZED; }

int MFTLObjectSetBoundaryCurve(MFTL_Object* obj, int tag, int mode, MFTL_BoundaryProjector project, void* user, int freeze)
{
  if (tag < 0) return MFTL_ERROR;
  if (mode == MFTL_BCURVE_CONSTRAINED && project == MFTL_NULL) return MFTL_ERROR;
  for (int i = 0; i < obj->n_bcurve; ++i)
    if (obj->bcurve[i].tag == tag)
    {
      obj->bcurve[i].mode = mode;
      obj->bcurve[i].project = project;
      obj->bcurve[i].user = user;
      obj->bcurve[i].freeze = freeze ? 1 : 0;
      return MFTL_SUCCESS;
    }
  obj->bcurve = realloc(obj->bcurve, (size_t) (obj->n_bcurve + 1) * sizeof(MFTL_BoundaryCurve));
  MFTL_BoundaryCurve* c = &obj->bcurve[obj->n_bcurve++];
  c->tag = tag;
  c->mode = mode;
  c->project = project;
  c->user = user;
  c->freeze = freeze ? 1 : 0;
  return MFTL_SUCCESS;
}

static const MFTL_BoundaryCurve* mftlCurveOf(const MFTL_Object* obj, int tag)
{
  for (int i = 0; i < obj->n_bcurve; ++i)
    if (obj->bcurve[i].tag == tag) return &obj->bcurve[i];
  return MFTL_NULL;  // REPRESENTED, not frozen
}

double MFTLBoundaryCornerAngle(const MFTL_RemeshParameters* params)
{
  return (params && params->boundary_corner_angle > 0.0) ? params->boundary_corner_angle : MFTL_BOUNDARY_CORNER_ANGLE_DEFAULT;
}

/* ------------------------------------------------------------------------------------------
 * corners
 * ------------------------------------------------------------------------------------------ */
/* 1 corner, 0 regular boundary vertex, -1 not a (known) boundary vertex. A boundary vertex whose
   fan is not completely stored counts as a corner (conservative: never moved, and a spline
   breakpoint). */
int MFTLBoundaryVertexKind(const MFTL_Object* obj, int v, double corner_angle_deg)
{
  const MFTL_Vertex* vx = &obj->vertices[v];
  int prev = -1, next = -1;
  if (MFTLBoundaryNeighbours(obj, v, &prev, &next) != MFTL_SUCCESS)
  {
    // no outgoing BOUNDARY half-edge found: interior (closed fan), or unknown (incomplete fan,
    // only for ghost vertices: treated as a corner, i.e. a fixed spline breakpoint)
    return (MFTLVertexRing(obj, v, MFTL_NULL, 0, MFTL_NULL) == MFTL_RING_CLOSED) ? -1 : 1;
  }
  if (prev < 0) return 1;  // predecessor not stored: unknown, conservative
  if (vx->flags & MFTL_VFLAG_CORNER) return 1;
  double theta;
  if (MFTLVertexTurningAngle(obj, v, &theta) != MFTL_SUCCESS) return 1;
  return fabs(theta) > corner_angle_deg * MFTL_PI / 180.0 ? 1 : 0;
}

/* Spline breakpoint of the REPRESENTED curve: a corner (tag or geodesic turning angle, §4.3) when
   v's fan is complete; for a ghost vertex whose fan is not completely stored (a rank rim) the
   polyline turning angle between its two boundary edges stands in for the geodesic one (both
   detect corners; smooth boundaries stay smooth across rank cuts). No stored neighbours: 1. */
/* returns 1 corner (breakpoint), 0 regular, -1 unknown (neighbours not stored: the curve point is
   not evaluable here and the operation needing it is skipped, never degraded to a chord) */
static int mftlSplineBreakpoint(const MFTL_Object* obj, int v, double corner_angle_deg)
{
  if (obj->vertices[v].flags & MFTL_VFLAG_CORNER) return 1;
  int prev = -1, next = -1;
  if (MFTLBoundaryNeighbours(obj, v, &prev, &next) != MFTL_SUCCESS || prev < 0 || next < 0) return -1;
  if (MFTLVertexRing(obj, v, MFTL_NULL, 0, MFTL_NULL) == MFTL_RING_OPEN_COMPLETE) return MFTLBoundaryVertexKind(obj, v, corner_angle_deg) != 0;
  double e0[3], e1[3];
  MFTLGetMinimumImageVector(obj->domain, e0, obj->vertices[prev].pos, obj->vertices[v].pos);
  MFTLGetMinimumImageVector(obj->domain, e1, obj->vertices[v].pos, obj->vertices[next].pos);
  const double l0 = sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]), l1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
  if (!(l0 * l1 > 0.0)) return -1;
  double c = (e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2]) / (l0 * l1);
  c = c > 1.0 ? 1.0 : (c < -1.0 ? -1.0 : c);
  return acos(c) > corner_angle_deg * MFTL_PI / 180.0;
}

/* ------------------------------------------------------------------------------------------
 * boundary curve
 * ------------------------------------------------------------------------------------------ */
static inline void mftlCR(const double p0[3], const double p1[3], const double p2[3], const double p3[3], double t, double out[3])
{
  const double t2 = t * t, t3 = t2 * t;
  for (int k = 0; k < 3; ++k)
    out[k] = 0.5 * (2.0 * p1[k] + (-p0[k] + p2[k]) * t + (2.0 * p0[k] - 5.0 * p1[k] + 4.0 * p2[k] - p3[k]) * t2 + (-p0[k] + 3.0 * p1[k] - 3.0 * p2[k] + p3[k]) * t3);
}

/* Point at parameter t of the boundary curve on the boundary segment a -> b (consecutive along
   the loop, a -> b a BOUNDARY half-edge he_ab). Output in a's local frame (the periodic image of
   a's wrapped position). Corners (and vertices whose neighbours are not stored) are spline
   breakpoints: the missing control point is the linear extrapolation, so a straight boundary
   stays straight exactly. CONSTRAINED tags are projected afterwards. */
int MFTLBoundaryCurvePoint(const MFTL_Object* obj, int he_ab, double t, double corner_angle_deg, double out[3])
{
  const MFTL_Domain* dom = obj->domain;
  const MFTL_HalfEdge* he = &obj->half_edges[he_ab];
  if (!MFTLHeIsBoundary(he)) return MFTL_ERROR;
  const int b = he->vertex_idx;
  const int a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
  double p0[3], p1[3], p2[3], p3[3], d[3];
  memcpy(p1, obj->vertices[a].pos, sizeof(p1));
  MFTLGetMinimumImageVector(dom, d, obj->vertices[a].pos, obj->vertices[b].pos);
  for (int k = 0; k < 3; ++k) p2[k] = p1[k] + d[k];
  int pa = -1, na = -1, pb = -1, nb = -1;
  const int ka = mftlSplineBreakpoint(obj, a, corner_angle_deg);
  const int kb = mftlSplineBreakpoint(obj, b, corner_angle_deg);
  if (ka < 0 || kb < 0) return MFTL_ERROR;  // a control point is not stored on this rank
  MFTLBoundaryNeighbours(obj, a, &pa, &na);
  MFTLBoundaryNeighbours(obj, b, &pb, &nb);
  if (ka == 0 && pa >= 0 && pa != b)
  {
    MFTLGetMinimumImageVector(dom, d, obj->vertices[a].pos, obj->vertices[pa].pos);
    for (int k = 0; k < 3; ++k) p0[k] = p1[k] + d[k];
  }
  else
    for (int k = 0; k < 3; ++k) p0[k] = 2.0 * p1[k] - p2[k];
  if (kb == 0 && nb >= 0 && nb != a)
  {
    MFTLGetMinimumImageVector(dom, d, obj->vertices[b].pos, obj->vertices[nb].pos);
    for (int k = 0; k < 3; ++k) p3[k] = p2[k] + d[k];
  }
  else
    for (int k = 0; k < 3; ++k) p3[k] = 2.0 * p2[k] - p1[k];
  mftlCR(p0, p1, p2, p3, t, out);
  const MFTL_BoundaryCurve* c = mftlCurveOf(obj, MFTLHeBoundaryTag(he));
  if (c && c->mode == MFTL_BCURVE_CONSTRAINED)
  {
    double in[3];
    memcpy(in, out, sizeof(in));
    if (c->project(MFTLHeBoundaryTag(he), in, out, c->user) != MFTL_SUCCESS) return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* Sets the position of v to p given in the local frame of anchor position `frame` (a wrapped
   position): the move from v's current position is a minimum-image displacement (§3.2). */
void MFTLSetVertexPosFromFrame(MFTL_Object* obj, int v, const double p[3])
{
  const MFTL_Domain* dom = obj->domain;
  double old[3], np[3];
  memcpy(old, obj->vertices[v].pos, sizeof(old));
  memcpy(np, p, sizeof(np));
  MFTLApplyPeriodicity(dom, np);
  memcpy(obj->vertices[v].pos, np, sizeof(np));
  MFTLImgFollow(dom, obj->vertices[v].img, old, np);
}

/* ------------------------------------------------------------------------------------------
 * augmented link condition (§4.4.1)
 * ------------------------------------------------------------------------------------------ */
#define MFTL_LK_MAX 128
typedef struct
{
  int nv, ne;
  int v[MFTL_LK_MAX];  // vertex indices, -1 = the virtual vertex w
  int e[MFTL_LK_MAX][2];  // edges (sorted pair), -1 = w
} mftlLink;

static void mftlLkAddV(mftlLink* L, int x)
{
  for (int i = 0; i < L->nv; ++i)
    if (L->v[i] == x) return;
  if (L->nv < MFTL_LK_MAX) L->v[L->nv++] = x;
}

static void mftlLkAddE(mftlLink* L, int x, int y)
{
  int a = x < y ? x : y, b = x < y ? y : x;
  for (int i = 0; i < L->ne; ++i)
    if (L->e[i][0] == a && L->e[i][1] == b) return;
  if (L->ne < MFTL_LK_MAX)
  {
    L->e[L->ne][0] = a;
    L->e[L->ne][1] = b;
    L->ne++;
  }
}

/* link of vertex v in the augmented complex; requires a complete stored fan */
static int mftlVertexLink(const MFTL_Object* obj, int v, mftlLink* L)
{
  L->nv = L->ne = 0;
  int ring[MFTL_LK_MAX], nr = 0;
  const int cls = MFTLVertexRing(obj, v, ring, MFTL_LK_MAX, &nr);
  if (cls == MFTL_RING_INCOMPLETE || nr >= MFTL_LK_MAX) return MFTL_ERROR;
  for (int i = 0; i < nr; ++i)
  {
    const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];  // v -> x in triangle (v, x, y)
    const int x = h->vertex_idx;
    const int y = obj->half_edges[h->next_idx].vertex_idx;
    mftlLkAddV(L, x);
    mftlLkAddV(L, y);
    mftlLkAddE(L, x, y);
  }
  if (cls == MFTL_RING_OPEN_COMPLETE)
  {
    // virtual triangles (v, next, w) and (prev, v, w): link edges {next, w}, {prev, w}
    const int out_h = ring[nr - 1];  // outgoing BOUNDARY half-edge
    const int in_h = obj->half_edges[obj->half_edges[ring[0]].next_idx].next_idx;  // incoming BOUNDARY
    const int next = obj->half_edges[out_h].vertex_idx;
    const int prev = obj->half_edges[obj->half_edges[obj->half_edges[in_h].next_idx].next_idx].vertex_idx;
    mftlLkAddV(L, -1);
    mftlLkAddE(L, next, -1);
    mftlLkAddE(L, prev, -1);
  }
  return MFTL_SUCCESS;
}

/* Lk(a) n Lk(b) == Lk(ab) for the edge of half-edge he (a -> b) */
int MFTLAugmentedLinkConditionOk(const MFTL_Object* obj, int he_idx)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  const int b = he->vertex_idx;
  const int a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
  mftlLink La, Lb;
  if (mftlVertexLink(obj, a, &La) != MFTL_SUCCESS || mftlVertexLink(obj, b, &Lb) != MFTL_SUCCESS) return 0;
  // Lk(ab): the opposite vertices of the (real or virtual) triangles on the edge
  int lab[2], nlab = 0;
  lab[nlab++] = obj->half_edges[he->next_idx].vertex_idx;  // c
  if (MFTLHeIsLocal(he))
    lab[nlab++] = obj->half_edges[obj->half_edges[he->twin_idx].next_idx].vertex_idx;  // d
  else if (MFTLHeIsBoundary(he))
    lab[nlab++] = -1;  // w
  else
    return 0;  // REMOTE / unresolved: never collapsed
  // vertices
  int ncommon = 0;
  for (int i = 0; i < La.nv; ++i)
  {
    const int x = La.v[i];
    if (x == a || x == b) continue;
    int inb = 0;
    for (int j = 0; j < Lb.nv; ++j)
      if (Lb.v[j] == x) inb = 1;
    if (!inb) continue;
    int inab = 0;
    for (int j = 0; j < nlab; ++j)
      if (lab[j] == x) inab = 1;
    if (!inab) return 0;  // a common neighbour outside Lk(ab): pinch / loop merge / hole closing
    ncommon++;
  }
  if (ncommon != nlab) return 0;
  // edges (Lk(ab) has none)
  for (int i = 0; i < La.ne; ++i)
  {
    if (La.e[i][0] == a || La.e[i][0] == b || La.e[i][1] == a || La.e[i][1] == b) continue;
    for (int j = 0; j < Lb.ne; ++j)
      if (La.e[i][0] == Lb.e[j][0] && La.e[i][1] == Lb.e[j][1]) return 0;
  }
  return 1;
}

/* ------------------------------------------------------------------------------------------
 * collapse / split placement
 * ------------------------------------------------------------------------------------------ */
/* Survivor target of the generalized collapse of he (a -> b, b survives), in b's local frame.
   *kind: 0 free (midpoint / fit as usual), 1 kept at b's position (interior edge with boundary
   survivor), 2 on the boundary curve (boundary edge). */
int MFTLGeneralizedCollapseTarget(const MFTL_Object* obj, int he_idx, double corner_angle_deg, double target[3], int* kind)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  const int b = he->vertex_idx;
  *kind = 0;
  if (MFTLHeIsBoundary(he))
  {
    double p[3];  // in a's frame
    if (MFTLBoundaryCurvePoint(obj, he_idx, 0.5, corner_angle_deg, p) != MFTL_SUCCESS) return MFTL_ERROR;
    const int a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
    double d[3];
    MFTLGetMinimumImageVector(obj->domain, d, obj->vertices[a].pos, obj->vertices[b].pos);
    // expressed in b's frame: target = pos_b + (p - (pos_a + d))
    for (int k = 0; k < 3; ++k) target[k] = obj->vertices[b].pos[k] + (p[k] - (obj->vertices[a].pos[k] + d[k]));
    *kind = 2;
    return MFTL_SUCCESS;
  }
  if (MFTLBoundaryVertexKind(obj, b, corner_angle_deg) >= 0)
  {
    memcpy(target, obj->vertices[b].pos, 3 * sizeof(double));
    *kind = 1;
    return MFTL_SUCCESS;
  }
  // free: the minimum-image midpoint in b's frame
  const int a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
  double d[3];
  MFTLGetMinimumImageVector(obj->domain, d, obj->vertices[b].pos, obj->vertices[a].pos);
  for (int k = 0; k < 3; ++k) target[k] = obj->vertices[b].pos[k] + 0.5 * d[k];
  return MFTL_SUCCESS;
}

/* 1 if the triangle spanned by the edge vectors e1, e2 (from the moved corner) keeps the
   orientation n0 and is not degenerate: 2 A / l_max^2 > MFTL_DEGENERATE_SINE (scale-free). A sign
   test alone accepts a flat triangle whose corners became collinear (e.g. an ear on a straight
   boundary pulled onto the line), whose normal is rounding noise. */
#define MFTL_DEGENERATE_SINE 1e-8
static int mftlKeepsOrientation(const double n0[3], const double e1[3], const double e2[3])
{
  const double n1[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  if (n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2] <= 0.0) return 0;
  double e3[3];
  for (int k = 0; k < 3; ++k) e3[k] = e2[k] - e1[k];
  const double l2 = fmax(fmax(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2], e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]),
                         e3[0] * e3[0] + e3[1] * e3[1] + e3[2] * e3[2]);
  return sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]) > MFTL_DEGENERATE_SINE * l2;
}

/* 1 if placing the collapse survivor at `target` (b's frame) keeps every surviving fan triangle of
   a and b non-inverted and non-degenerate (mftlKeepsOrientation) */
int MFTLCollapseKeepsOrientation(const MFTL_Object* obj, int he_idx, const double target[3])
{
  const MFTL_Domain* dom = obj->domain;
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  const int b = he->vertex_idx;
  const int a = obj->half_edges[obj->half_edges[he->next_idx].next_idx].vertex_idx;
  const int ends[2] = {a, b};
  for (int e = 0; e < 2; ++e)
  {
    int ring[MFTL_LK_MAX], nr = 0;
    if (MFTLVertexRing(obj, ends[e], ring, MFTL_LK_MAX, &nr) == MFTL_RING_INCOMPLETE) return 0;
    const double* pv = obj->vertices[ends[e]].pos;
    for (int i = 0; i < nr; ++i)
    {
      const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];
      const int x = h->vertex_idx, y = obj->half_edges[h->next_idx].vertex_idx;
      if (x == a || x == b || y == a || y == b) continue;  // a removed triangle
      double dx[3], dy[3], tx[3], ty[3], t0[3];
      MFTLGetMinimumImageVector(dom, dx, pv, obj->vertices[x].pos);
      MFTLGetMinimumImageVector(dom, dy, pv, obj->vertices[y].pos);
      // target relative to this endpoint (target is in b's frame)
      double tb[3];
      MFTLGetMinimumImageVector(dom, tb, obj->vertices[b].pos, pv);  // pv - pos_b
      for (int k = 0; k < 3; ++k) t0[k] = (target[k] - obj->vertices[b].pos[k]) - tb[k];
      for (int k = 0; k < 3; ++k)
      {
        tx[k] = dx[k] - t0[k];
        ty[k] = dy[k] - t0[k];
      }
      double n0[3] = {dx[1] * dy[2] - dx[2] * dy[1], dx[2] * dy[0] - dx[0] * dy[2], dx[0] * dy[1] - dx[1] * dy[0]};
      if (!mftlKeepsOrientation(n0, tx, ty)) return 0;
    }
  }
  return 1;
}

/* New-vertex position of the generalized split of he (a -> b) in a's frame: on the boundary
   curve for a boundary edge (returns MFTL_SUCCESS), MFTL_ERROR for an interior edge (usual fit). */
int MFTLGeneralizedSplitTarget(const MFTL_Object* obj, int he_idx, double corner_angle_deg, double target[3])
{
  if (!MFTLHeIsBoundary(&obj->half_edges[he_idx])) return MFTL_ERROR;
  return MFTLBoundaryCurvePoint(obj, he_idx, 0.5, corner_angle_deg, target);
}

/* 1 if every vertex of v's stored fan is owned by v's owner. Parallel smoothing moves owned
   vertices concurrently on every rank and each rank sees the others' moves only after the ghost
   update, so a move is safe only when no other rank can move a corner of v's triangles in the
   same sweep. Always 1 on one rank. */
int MFTLVertexFanOwned(const MFTL_Object* obj, int v)
{
  const int me = obj->vertices[v].owner_proc;
  int ring[MFTL_LK_MAX], nr = 0;
  if (MFTLVertexRing(obj, v, ring, MFTL_LK_MAX, &nr) == MFTL_RING_INCOMPLETE) return 0;
  for (int i = 0; i < nr; ++i)
  {
    const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];
    if (obj->vertices[h->vertex_idx].owner_proc != me) return 0;
    if (obj->vertices[obj->half_edges[h->next_idx].vertex_idx].owner_proc != me) return 0;
  }
  return 1;
}

/* 1 if moving v from `from` to `to` (both in v's local frame) keeps every stored fan triangle of v
   non-inverted and non-degenerate (mftlKeepsOrientation) */
int MFTLMoveKeepsOrientation(const MFTL_Object* obj, int v, const double from[3], const double to[3])
{
  const MFTL_Domain* dom = obj->domain;
  const double* pv = obj->vertices[v].pos;
  int h0 = obj->vertices[v].out_he_idx;
  if (h0 < 0) return 1;
  // every stored triangle incident to v: walk both directions of the (possibly open) fan
  int ring[MFTL_LK_MAX], nr = 0;
  if (MFTLVertexRing(obj, v, ring, MFTL_LK_MAX, &nr) == MFTL_RING_INCOMPLETE) return 0;  // be safe
  for (int i = 0; i < nr; ++i)
  {
    const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];
    const int x = h->vertex_idx, y = obj->half_edges[h->next_idx].vertex_idx;
    double dx[3], dy[3], a0[3], a1[3], b0[3], b1[3];
    MFTLGetMinimumImageVector(dom, dx, pv, obj->vertices[x].pos);
    MFTLGetMinimumImageVector(dom, dy, pv, obj->vertices[y].pos);
    for (int k = 0; k < 3; ++k)
    {
      a0[k] = pv[k] + dx[k] - from[k], b0[k] = pv[k] + dy[k] - from[k];
      a1[k] = pv[k] + dx[k] - to[k], b1[k] = pv[k] + dy[k] - to[k];
    }
    double n0[3] = {a0[1] * b0[2] - a0[2] * b0[1], a0[2] * b0[0] - a0[0] * b0[2], a0[0] * b0[1] - a0[1] * b0[0]};
    if (!mftlKeepsOrientation(n0, a1, b1)) return 0;
  }
  return 1;
}

/* ------------------------------------------------------------------------------------------
 * valence targets
 * ------------------------------------------------------------------------------------------ */
int MFTLBoundaryValenceTarget(const MFTL_Object* obj, int v)
{
  double sum;
  if (MFTLVertexAngleSum(obj, v, &sum) != MFTL_RING_OPEN_COMPLETE) return 6;
  int t = (int) lround(sum / (MFTL_PI / 3.0)) + 1;
  return t < 2 ? 2 : t;
}

/* ------------------------------------------------------------------------------------------
 * boundary-vertex tangential smoothing along the boundary curve
 * ------------------------------------------------------------------------------------------ */
/* One Jacobi sweep over the owned, non-corner boundary vertices of a generalized body: each moves
   along its boundary curve to the arc-length midpoint (chord lengths) of its loop neighbours,
   times `relaxation`. Frozen tags, corners and vertices whose neighbours are not stored stay.
   A move that would invert a fan triangle is skipped. Returns the number of moved vertices. */
#define MFTL_BOUNDARY_SMOOTH_DEADBAND 0.1  // |L2 - L1| / (L1 + L2): spacing ratio within ~1.22

int MFTLSmoothBoundaryVertices(MFTL_Object* obj, double relaxation, double corner_angle_deg)
{
  const MFTL_Domain* dom = obj->domain;
  const int rank = dom->rank;
  const int nv = obj->n_vertices;
  double(*np)[3] = malloc((size_t) (nv > 0 ? nv : 1) * sizeof(*np));
  char* mv = calloc((size_t) (nv > 0 ? nv : 1), 1);
  for (int v = 0; v < nv; ++v)
  {
    MFTL_Vertex* vx = &obj->vertices[v];
    if (vx->is_ghost || vx->owner_proc != rank || (vx->flags & MFTL_FLAG_DEAD) || vx->out_he_idx < 0) continue;
    if (MFTLBoundaryVertexKind(obj, v, corner_angle_deg) != 0) continue;  // interior, corner or unknown
    if (!MFTLVertexFanOwned(obj, v)) continue;                              // concurrent moves on another rank
    int prev = -1, next = -1;
    if (MFTLBoundaryNeighbours(obj, v, &prev, &next) != MFTL_SUCCESS || prev < 0 || next < 0) continue;
    // outgoing boundary half-edge of v, incoming boundary half-edge prev -> v
    int ring[MFTL_LK_MAX], nr = 0;
    if (MFTLVertexRing(obj, v, ring, MFTL_LK_MAX, &nr) != MFTL_RING_OPEN_COMPLETE) continue;
    const int out_h = ring[nr - 1];
    const int in_h = obj->half_edges[obj->half_edges[ring[0]].next_idx].next_idx;
    const MFTL_BoundaryCurve* c = mftlCurveOf(obj, MFTLHeBoundaryTag(&obj->half_edges[out_h]));
    if (c && c->freeze) continue;
    if (MFTLHeBoundaryTag(&obj->half_edges[out_h]) != MFTLHeBoundaryTag(&obj->half_edges[in_h])) continue;  // tag change = breakpoint
    double dp[3], dn[3];
    MFTLGetMinimumImageVector(dom, dp, vx->pos, obj->vertices[prev].pos);
    MFTLGetMinimumImageVector(dom, dn, vx->pos, obj->vertices[next].pos);
    const double L1 = sqrt(dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2]);
    const double L2 = sqrt(dn[0] * dn[0] + dn[1] * dn[1] + dn[2] * dn[2]);
    if (!(L1 > 0.0) || !(L2 > 0.0)) continue;
    // dead band: every move re-samples the reconstructed curve, which accumulates the interpolation
    // error over repeated passes; move only where the spacing is actually uneven
    if (fabs(L2 - L1) <= MFTL_BOUNDARY_SMOOTH_DEADBAND * (L1 + L2)) continue;
    const double s = L1 + relaxation * (0.5 * (L1 + L2) - L1);  // arc-length target from prev
    double p[3];
    if (s < L1)
    {
      // on the segment prev -> v (curve point in prev's frame) at t = s / L1
      double q[3];
      if (MFTLBoundaryCurvePoint(obj, in_h, s / L1, corner_angle_deg, q) != MFTL_SUCCESS) continue;
      for (int k = 0; k < 3; ++k) p[k] = vx->pos[k] + dp[k] + (q[k] - obj->vertices[prev].pos[k]);
    }
    else
    {
      double q[3];
      if (MFTLBoundaryCurvePoint(obj, out_h, (s - L1) / L2, corner_angle_deg, q) != MFTL_SUCCESS) continue;
      for (int k = 0; k < 3; ++k) p[k] = q[k];  // already in v's frame
    }
    // no fan triangle may invert
    int ok = 1;
    for (int i = 0; i < nr && ok; ++i)
    {
      const MFTL_HalfEdge* h = &obj->half_edges[ring[i]];
      const int x = h->vertex_idx, y = obj->half_edges[h->next_idx].vertex_idx;
      double dx[3], dy[3], tx[3], ty[3];
      MFTLGetMinimumImageVector(dom, dx, vx->pos, obj->vertices[x].pos);
      MFTLGetMinimumImageVector(dom, dy, vx->pos, obj->vertices[y].pos);
      for (int k = 0; k < 3; ++k)
      {
        tx[k] = dx[k] - (p[k] - vx->pos[k]);
        ty[k] = dy[k] - (p[k] - vx->pos[k]);
      }
      double n0[3] = {dx[1] * dy[2] - dx[2] * dy[1], dx[2] * dy[0] - dx[0] * dy[2], dx[0] * dy[1] - dx[1] * dy[0]};
      if (!mftlKeepsOrientation(n0, tx, ty)) ok = 0;
    }
    if (!ok) continue;
    memcpy(np[v], p, sizeof(p));
    mv[v] = 1;
  }
  int moved = 0;
  for (int v = 0; v < nv; ++v)
    if (mv[v])
    {
      // re-checked against the current (possibly already moved) neighbours: Jacobi-safe
      if (!MFTLMoveKeepsOrientation(obj, v, obj->vertices[v].pos, np[v])) continue;
      MFTLSetVertexPosFromFrame(obj, v, np[v]);
      moved++;
    }
  free(np);
  free(mv);
  return moved;
}
