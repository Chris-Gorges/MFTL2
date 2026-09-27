/*
 * MFTL2 remeshing — MFTL1 edge criteria and sweep order for the remesh drivers
 * (MFTL_RemeshParameters.mftl1_criteria = 1).
 *
 * multiflow's front tracking must keep its MFTL1 remeshing behaviour. MFTL1 decides splits,
 * collapses and flips with the tests of edgesplit.c, edgecollapse.c and edgeflip.c; MFTL2's
 * own criteria (remeshing_core.c) are simpler (e.g. JIAO split = "longer than max", JIAO collapse =
 * "shorter than min") and coarsen a front very differently. The functions below reproduce
 * the MFTL1 tests on the half-edge structure:
 *
 *   edge E = a->b (half-edge he) in T1 = (a, b, c); twin b->a in T2 = (b, a, d);
 *   wing edges: T1: b-c, c-a; T2: a-d, d-b; c and d are the opposite vertices (V3, V4).
 *
 * Known, documented deviations: fresh triangle areas/normals are used where MFTL1 read cached
 * ones; MFTL1's Collapse_Neighbour_Valence_Check has no final return statement (undefined
 * behaviour) and is taken as "passes unless one of its explicit rejections fires"; MFTL2's
 * link-condition test is kept in addition as the manifold safety net of its collapse operator.
 */
#include <math.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "mftl_geometry.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"

#define MFTL1_VERYSMALL 1.0e-15
#define MFTL1_RAD_TO_DEG 57.2957795130823201e0
#define MFTL1_DEG_TO_RAD 1.7453292519943296e-2

typedef struct
{
  int he, twin;  // a->b and b->a
  int a, b, c, d;  // vertex indices (c opposite in T1, d opposite in T2)
  int t1, t2;  // triangles
  int he_bc, he_ca, he_ad, he_db;  // wing half-edges (in T1: b->c, c->a; in T2: a->d, d->b)
} mftl1Quad;

static int mftl1QuadOf(const MFTL_Object* obj, int he_idx, mftl1Quad* q)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->twin_idx == MFTL_ERROR) return MFTL_ERROR;
  const MFTL_HalfEdge* tw = &obj->half_edges[he->twin_idx];
  q->he = he_idx;
  q->twin = he->twin_idx;
  q->he_bc = he->next_idx;
  q->he_ca = obj->half_edges[q->he_bc].next_idx;
  q->he_ad = tw->next_idx;
  q->he_db = obj->half_edges[q->he_ad].next_idx;
  q->a = obj->half_edges[q->he_ca].vertex_idx;
  q->b = he->vertex_idx;
  q->c = obj->half_edges[q->he_bc].vertex_idx;
  q->d = obj->half_edges[q->he_ad].vertex_idx;
  q->t1 = he->triangle_idx;
  q->t2 = tw->triangle_idx;
  return MFTL_SUCCESS;
}

static double mftl1Dist(const MFTL_Object* obj, int u, int v)
{
  double e[3];
  MFTLGetMinimumImageVector(obj->domain, e, obj->vertices[u].pos, obj->vertices[v].pos);
  return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
}

// MFTL_EdgeOpposingAngle: angle at vertex w subtended by the edge (u, v), with MFTL1's
// normalisation floor and an unclamped acos (as MFTL1).
static double mftl1OpposingAngle(const MFTL_Object* obj, int u, int v, int w)
{
  double v0[3], v1[3];
  MFTLGetMinimumImageVector(obj->domain, v0, obj->vertices[w].pos, obj->vertices[u].pos);
  MFTLGetMinimumImageVector(obj->domain, v1, obj->vertices[w].pos, obj->vertices[v].pos);
  double l0 = sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
  double l1 = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
  if (l0 < MFTL1_VERYSMALL) l0 = MFTL1_VERYSMALL;
  if (l1 < MFTL1_VERYSMALL) l1 = MFTL1_VERYSMALL;
  for (int k = 0; k < 3; ++k)
  {
    v0[k] /= l0;
    v1[k] /= l1;
  }
  return acos(v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]);
}

// Unit normal of the triangle (u, v, w) (MFTL_TriangleNormal, floor on the magnitude).
static void mftl1Normal(const MFTL_Object* obj, int u, int v, int w, double n[3])
{
  double e0[3], e1[3];
  MFTLGetMinimumImageVector(obj->domain, e0, obj->vertices[u].pos, obj->vertices[v].pos);
  MFTLGetMinimumImageVector(obj->domain, e1, obj->vertices[u].pos, obj->vertices[w].pos);
  n[0] = e0[1] * e1[2] - e0[2] * e1[1];
  n[1] = e0[2] * e1[0] - e0[0] * e1[2];
  n[2] = e0[0] * e1[1] - e0[1] * e1[0];
  double l = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (l < MFTL1_VERYSMALL) l = MFTL1_VERYSMALL;
  n[0] /= l;
  n[1] /= l;
  n[2] /= l;
}

static double mftl1TriArea(const MFTL_Object* obj, int u, int v, int w)
{
  double e0[3], e1[3];
  MFTLGetMinimumImageVector(obj->domain, e0, obj->vertices[u].pos, obj->vertices[v].pos);
  MFTLGetMinimumImageVector(obj->domain, e1, obj->vertices[u].pos, obj->vertices[w].pos);
  double cx = e0[1] * e1[2] - e0[2] * e1[1], cy = e0[2] * e1[0] - e0[0] * e1[2], cz = e0[0] * e1[1] - e0[1] * e1[0];
  return 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
}

// MFTL_AngleBetweenTriangles of T1 = (a, b, c) and T2 = (b, a, d), in degrees.
static double mftl1DihedralDeg(const MFTL_Object* obj, const mftl1Quad* q)
{
  double n1[3], n2[3];
  mftl1Normal(obj, q->a, q->b, q->c, n1);
  mftl1Normal(obj, q->b, q->a, q->d, n2);
  return acos(n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) * MFTL1_RAD_TO_DEG;
}

static int mftl1Valence(const MFTL_Object* obj, int v)
{
  MFTL_Vertex* ring[64];
  return MFTLGetVertexOneRingVertices(obj, &obj->vertices[v], ring, 64, MFTL_FALSE);
}

// Is the one-ring of v completely stored (closed fan, no dead half-edge)? MFTL1's
// Boundary_Check_* reject operations touching ghost / incomplete neighbourhoods the same way.
static int mftl1RingClosed(const MFTL_Object* obj, int v)
{
  const MFTL_Vertex* vx = &obj->vertices[v];
  if (vx->flags & MFTL_FLAG_DEAD) return 0;
  if (vx->out_he_idx < 0 || vx->out_he_idx >= obj->n_half_edges) return 0;
  int h = vx->out_he_idx, count = 0;
  do
  {
    if (obj->half_edges[h].flags & MFTL_FLAG_DEAD) return 0;
    int tw = obj->half_edges[h].twin_idx;
    if (tw == MFTL_ERROR || (obj->half_edges[tw].flags & MFTL_FLAG_DEAD)) return 0;
    h = obj->half_edges[tw].next_idx;
    if (++count > 64) return 0;
  } while (h != vx->out_he_idx);
  return 1;
}

// Angle between the normals of the two triangles of half-edge h, in degrees.
static double mftl1HeDihedralDeg(const MFTL_Object* obj, int h)
{
  int tw = obj->half_edges[h].twin_idx;
  int t[2] = {obj->half_edges[h].triangle_idx, obj->half_edges[tw].triangle_idx};
  double n[2][3];
  for (int k = 0; k < 2; ++k)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[t[k]], vv);
    mftl1Normal(obj, (int) (vv[0] - obj->vertices), (int) (vv[1] - obj->vertices), (int) (vv[2] - obj->vertices), n[k]);
  }
  return acos(n[0][0] * n[1][0] + n[0][1] * n[1][1] + n[0][2] * n[1][2]) * MFTL1_RAD_TO_DEG;
}

// Boundary_Check_Collapse feature test: no spoke edge of v (other than the edge being
// collapsed, he/twin) may be sharper than maxAngleBetweenTris. The ring must be closed.
static int mftl1SpokesSmooth(const MFTL_Object* obj, int v, int he, int twin, double max_angle)
{
  const MFTL_Vertex* vx = &obj->vertices[v];
  int h = vx->out_he_idx, count = 0;
  do
  {
    if (h != he && h != twin && mftl1HeDihedralDeg(obj, h) > max_angle) return 0;
    h = obj->half_edges[obj->half_edges[h].twin_idx].next_idx;
    if (++count > 64) return 0;
  } while (h != vx->out_he_idx);
  return 1;
}

// Triangle across half-edge h (MFTL_ERROR at a boundary).
static int mftl1Across(const MFTL_Object* obj, int h)
{
  int tw = obj->half_edges[h].twin_idx;
  return (tw == MFTL_ERROR) ? MFTL_ERROR : obj->half_edges[tw].triangle_idx;
}

// Wing lengths of the quad.
static void mftl1WingLengths(const MFTL_Object* obj, const mftl1Quad* q, double L[4])
{
  L[0] = mftl1Dist(obj, q->b, q->c);
  L[1] = mftl1Dist(obj, q->c, q->a);
  L[2] = mftl1Dist(obj, q->a, q->d);
  L[3] = mftl1Dist(obj, q->d, q->b);
}

/* ---------------------------------------------------------------------------------------- */
/* Split (edgesplit.c)                                                                      */
/* ---------------------------------------------------------------------------------------- */
int MFTL1SplitCriterion(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  mftl1Quad q;
  if (mftl1QuadOf(obj, he_idx, &q) != MFTL_SUCCESS) return MFTL_FALSE;  // Boundary_Check
  const double E = mftl1Dist(obj, q.a, q.b);
  switch (params->criterion)
  {
    case MFTL_CRITERION_TRIANGLE_AREA:  // Triangle_Area_Threshold_Check_Split: BOTH triangles large
      return mftl1TriArea(obj, q.a, q.b, q.c) > params->max_area_threshold && mftl1TriArea(obj, q.b, q.a, q.d) > params->max_area_threshold;
    case MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE: return MFTLBotschKobbeltCurvatureThreshold(obj, he_idx, params, 1);
    case MFTL_CRITERION_JIAO:  // Triangle_Split_Thresholds_Jiao
    {
      if (E < params->reference_length) return MFTL_FALSE;
      double L[4];
      mftl1WingLengths(obj, &q, L);
      double LMAX = fmax(L[0], fmax(fmax(L[1], L[2]), L[3]));
      if (E > params->max_edge_length) return (E > LMAX - MFTL1_VERYSMALL) ? MFTL_TRUE : MFTL_FALSE;
      double LMIN = fmin(L[0], fmin(fmin(L[1], L[2]), L[3]));
      double angle = fmax(mftl1OpposingAngle(obj, q.a, q.b, q.c), mftl1OpposingAngle(obj, q.a, q.b, q.d));
      return (angle > 145.0 * MFTL1_DEG_TO_RAD && LMIN > params->min_edge_length) ? MFTL_TRUE : MFTL_FALSE;
    }
    case MFTL_CRITERION_BOTSCH_KOBBELT:
    default: return E > params->max_edge_length;
  }
}

/* ---------------------------------------------------------------------------------------- */
/* Collapse (edgecollapse.c)                                                                */
/* ---------------------------------------------------------------------------------------- */
static int mftl1CollapseCriterion(MFTL_Object* obj, const mftl1Quad* q, int he_idx, MFTL_RemeshParameters* params)
{
  const double E = mftl1Dist(obj, q->a, q->b);
  switch (params->criterion)
  {
    case MFTL_CRITERION_TRIANGLE_AREA:  // Triangle_Area_Threshold_Check_Collapse: BOTH triangles small
      return mftl1TriArea(obj, q->a, q->b, q->c) < params->min_area_threshold && mftl1TriArea(obj, q->b, q->a, q->d) < params->min_area_threshold;
    case MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE: return MFTLBotschKobbeltCurvatureThreshold(obj, he_idx, params, 2);
    case MFTL_CRITERION_JIAO:  // Triangle_Collapse_Thresholds_Jiao
    {
      double angle = fmin(mftl1OpposingAngle(obj, q->a, q->b, q->c), mftl1OpposingAngle(obj, q->a, q->b, q->d));
      double L[4];
      mftl1WingLengths(obj, q, L);
      double LMAX = fmax(L[0], fmax(fmax(L[1], L[2]), L[3]));
      double LMIN = fmin(L[0], fmin(fmin(L[1], L[2]), L[3]));
      const double R = 0.75, r = 0.25, S = R * params->reference_length;
      if (angle < 20.0 * MFTL1_DEG_TO_RAD && LMAX < params->reference_length) return MFTL_TRUE;
      if (E < r * LMAX) return MFTL_TRUE;
      if (LMAX < S && E < LMIN) return MFTL_TRUE;
      if (LMAX < params->reference_length && E < R * LMAX) return MFTL_TRUE;
      return MFTL_FALSE;
    }
    case MFTL_CRITERION_BOTSCH_KOBBELT:
    default: return E < params->min_edge_length;
  }
}

// The triangle across the edge of triangle t that touches neither a nor b.
static int mftl1OuterAcross(const MFTL_Object* obj, int t, int a, int b)
{
  if (t == MFTL_ERROR) return MFTL_ERROR;
  int h = obj->triangles[t].he_idx;
  for (int k = 0; k < 3; ++k, h = obj->half_edges[h].next_idx)
  {
    int to = obj->half_edges[h].vertex_idx;
    int from = obj->half_edges[obj->half_edges[obj->half_edges[h].next_idx].next_idx].vertex_idx;
    if (to != a && to != b && from != a && from != b) return mftl1Across(obj, h);
  }
  return MFTL_ERROR;
}

int MFTL1CollapseApproved(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params, int honour_locks)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return MFTL_FALSE;
  if (honour_locks && (he->flags & MFTL_FLAG_VISITED)) return MFTL_FALSE;
  mftl1Quad q;
  if (mftl1QuadOf(obj, he_idx, &q) != MFTL_SUCCESS) return MFTL_FALSE;  // Boundary_Check_Collapse
  if ((obj->vertices[q.a].flags | obj->vertices[q.b].flags) & MFTL_FLAG_DEAD) return MFTL_FALSE;
  if (honour_locks && ((obj->vertices[q.a].flags | obj->vertices[q.b].flags) & MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (!mftl1RingClosed(obj, q.a) || !mftl1RingClosed(obj, q.b)) return MFTL_FALSE;
  if (!mftl1SpokesSmooth(obj, q.a, q.he, q.twin, params->max_angle_between_tris)) return MFTL_FALSE;
  if (!mftl1SpokesSmooth(obj, q.b, q.he, q.twin, params->max_angle_between_tris)) return MFTL_FALSE;

  if (!mftl1CollapseCriterion(obj, &q, he_idx, params)) return MFTL_FALSE;

  // Collapse_Neighbour_Valence_Check: the outer neighbours of the two wing triangles on each
  // side must exist and differ.
  const int T2 = mftl1Across(obj, q.he_bc), T3 = mftl1Across(obj, q.he_ca);
  const int T4 = mftl1Across(obj, q.he_ad), T5 = mftl1Across(obj, q.he_db);
  const int o2 = mftl1OuterAcross(obj, T2, q.a, q.b), o3 = mftl1OuterAcross(obj, T3, q.a, q.b);
  const int o4 = mftl1OuterAcross(obj, T4, q.a, q.b), o5 = mftl1OuterAcross(obj, T5, q.a, q.b);
  if (o2 == MFTL_ERROR || o3 == MFTL_ERROR || o4 == MFTL_ERROR || o5 == MFTL_ERROR) return MFTL_FALSE;
  if (o2 == o3 || o4 == o5) return MFTL_FALSE;

  // at least two remaining triangles on each side: valence(a) >= 4 and valence(b) >= 4
  if (mftl1Valence(obj, q.a) - 2 < 2 || mftl1Valence(obj, q.b) - 2 < 2) return MFTL_FALSE;

  // MFTL2 collapse operator safety net (not an MFTL1 test): link condition
  MFTL_Vertex* ring_a[64];
  MFTL_Vertex* ring_b[64];
  int na = MFTLGetVertexOneRingVertices(obj, &obj->vertices[q.a], ring_a, 64, MFTL_FALSE);
  int nb = MFTLGetVertexOneRingVertices(obj, &obj->vertices[q.b], ring_b, 64, MFTL_FALSE);
  int common = 0;
  for (int i = 0; i < na; ++i)
    for (int j = 0; j < nb; ++j)
      if (ring_a[i] == ring_b[j]) common++;
  if (common != 2) return MFTL_FALSE;

  // Botsch & Kobbelt: no new edge longer than the high threshold (MFTL1 tests the fitted
  // midpoint; the raw midpoint is used here)
  if (params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT || params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT_CURVATURE)
  {
    double m[3], e[3];
    MFTLGetMinimumImageVector(obj->domain, e, obj->vertices[q.a].pos, obj->vertices[q.b].pos);
    for (int k = 0; k < 3; ++k) m[k] = obj->vertices[q.a].pos[k] + 0.5 * e[k];
    MFTL_Vertex** rings[2] = {ring_a, ring_b};
    int ns[2] = {na, nb};
    for (int s = 0; s < 2; ++s)
      for (int i = 0; i < ns[s]; ++i)
      {
        if (rings[s][i] == &obj->vertices[q.a] || rings[s][i] == &obj->vertices[q.b]) continue;
        double dv[3];
        MFTLGetMinimumImageVector(obj->domain, dv, m, rings[s][i]->pos);
        if (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2] > params->max_edge_length_sq) return MFTL_FALSE;
      }
  }
  return MFTL_TRUE;
}

/* ---------------------------------------------------------------------------------------- */
/* Flip (edgeflip.c)                                                                        */
/* ---------------------------------------------------------------------------------------- */
// Valence3_Check for one endpoint: the triangles across its two wing edges must be neither
// identical nor edge-adjacent.
static int mftl1Valence3Side(const MFTL_Object* obj, int wing1, int wing2)
{
  int X = mftl1Across(obj, wing1), Y = mftl1Across(obj, wing2);
  if (X == MFTL_ERROR || Y == MFTL_ERROR) return MFTL_FALSE;
  if (X == Y) return MFTL_FALSE;
  int h = obj->triangles[X].he_idx;
  int wing1_twin = obj->half_edges[wing1].twin_idx;
  for (int k = 0; k < 3; ++k, h = obj->half_edges[h].next_idx)
  {
    if (h == wing1_twin) continue;
    if (mftl1Across(obj, h) == Y) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

static double mftl1Quality(const MFTL_Object* obj, int u, int v, int w)
{
  double p = mftl1Dist(obj, u, v) + mftl1Dist(obj, v, w) + mftl1Dist(obj, u, w);
  return (p * p / mftl1TriArea(obj, u, v, w)) / 20.78460969;
}

int MFTL1FlipApproved(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return MFTL_FALSE;
  mftl1Quad q;
  if (mftl1QuadOf(obj, he_idx, &q) != MFTL_SUCCESS) return MFTL_FALSE;  // Boundary_Check_Flip
  if (!mftl1RingClosed(obj, q.a) || !mftl1RingClosed(obj, q.b) || !mftl1RingClosed(obj, q.c) || !mftl1RingClosed(obj, q.d)) return MFTL_FALSE;

  // MFTL2 flip operator safety (not MFTL1 tests): no valence < 3 after the flip and no
  // duplicate edge c-d.
  int va = mftl1Valence(obj, q.a), vb = mftl1Valence(obj, q.b);
  if (va <= 3 || vb <= 3) return MFTL_FALSE;
  {
    MFTL_Vertex* ring_c[64];
    int nc = MFTLGetVertexOneRingVertices(obj, &obj->vertices[q.c], ring_c, 64, MFTL_FALSE);
    for (int i = 0; i < nc; ++i)
      if (ring_c[i] == &obj->vertices[q.d]) return MFTL_FALSE;
  }

  // Edge_Obscure_Angle_Check: the quad's interior angles at a and b stay below 180 degrees
  double at_a = (mftl1OpposingAngle(obj, q.b, q.c, q.a) + mftl1OpposingAngle(obj, q.b, q.d, q.a)) * MFTL1_RAD_TO_DEG;
  double at_b = (mftl1OpposingAngle(obj, q.a, q.c, q.b) + mftl1OpposingAngle(obj, q.a, q.d, q.b)) * MFTL1_RAD_TO_DEG;
  if (at_a > 180.0 || at_b > 180.0) return MFTL_FALSE;

  // Triangle_Angle_Threshold
  if (mftl1DihedralDeg(obj, &q) > params->max_angle_between_tris) return MFTL_FALSE;

  // Valence3_Check (both endpoints)
  if (!mftl1Valence3Side(obj, q.he_bc, q.he_db)) return MFTL_FALSE;
  if (!mftl1Valence3Side(obj, q.he_ca, q.he_ad)) return MFTL_FALSE;

  // Perimeter_To_Area_Ratio_Check (fresh areas; MFTL1 read the cached ones for "before")
  double before = 0.5 * (mftl1Quality(obj, q.a, q.b, q.c) + mftl1Quality(obj, q.b, q.a, q.d));
  double after = 0.5 * (mftl1Quality(obj, q.c, q.d, q.a) + mftl1Quality(obj, q.c, q.d, q.b));
  if (fabs(1.0 - before) < fabs(1.0 - after)) return MFTL_FALSE;

  // Opposing_Angles_Check
  if ((mftl1OpposingAngle(obj, q.a, q.b, q.c) + mftl1OpposingAngle(obj, q.a, q.b, q.d)) * MFTL1_RAD_TO_DEG < 180.0) return MFTL_FALSE;

  // Equalize_To_Valence6_Check (absolute deviation)
  int vc = mftl1Valence(obj, q.c), vd = mftl1Valence(obj, q.d);
  int pre = abs(va - 6) + abs(vb - 6) + abs(vc - 6) + abs(vd - 6);
  int post = abs(va - 1 - 6) + abs(vb - 1 - 6) + abs(vc + 1 - 6) + abs(vd + 1 - 6);
  return (pre > post) ? MFTL_TRUE : MFTL_FALSE;
}

/* ---------------------------------------------------------------------------------------- */
/* Serial sweeps (MFTL_Refining / MFTL_Coarsening / MFTL_Flipping)                          */
/* ---------------------------------------------------------------------------------------- */
static double mftl1CurrentLength(const MFTL_Object* obj, int j)
{
  const MFTL_HalfEdge* he = &obj->half_edges[j];
  int prev = obj->half_edges[he->next_idx].next_idx;
  return mftl1Dist(obj, obj->half_edges[prev].vertex_idx, he->vertex_idx);
}

static int mftl1LengthCriterion(const MFTL_RemeshParameters* params)
{
  return params->criterion == MFTL_CRITERION_JIAO || params->criterion == MFTL_CRITERION_BOTSCH_KOBBELT;
}

/* One undirected edge per slot, identified by its two original half-edges (a collapse can
   kill either one; the slot then continues with the other). */
typedef struct
{
  int he, tw;
  double len;
  int pos;  // position in the previous order (stable-sort key, like uthash's HASH_SORT)
} mftl1Slot;

static int mftl1SlotAlive(const MFTL_Object* obj, const mftl1Slot* sl)
{
  if (!(obj->half_edges[sl->he].flags & MFTL_FLAG_DEAD)) return sl->he;
  if (sl->tw != MFTL_ERROR && !(obj->half_edges[sl->tw].flags & MFTL_FLAG_DEAD)) return sl->tw;
  return MFTL_ERROR;
}

static int mftl1CmpSlotLong(const void* x, const void* y)
{
  const mftl1Slot *a = x, *b = y;
  if (a->len > b->len) return -1;
  if (a->len < b->len) return 1;
  return (a->pos > b->pos) - (a->pos < b->pos);
}
static int mftl1CmpSlotShort(const void* x, const void* y)
{
  const mftl1Slot *a = x, *b = y;
  if (a->len < b->len) return -1;
  if (a->len > b->len) return 1;
  return (a->pos > b->pos) - (a->pos < b->pos);
}

// Append every interior edge whose canonical half-edge index lies in [from, n_half_edges)
// (the edges an operation created, in creation order) to the order.
static int mftl1AppendEdges(const MFTL_Object* obj, int from, mftl1Slot* order, int n)
{
  for (int j = from; j < obj->n_half_edges; ++j)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[j];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (he->twin_idx == MFTL_ERROR) continue;
    if (he->twin_idx < j && he->twin_idx >= from) continue;  // pair already appended
    if (he->twin_idx < from) continue;  // an existing edge's half-edge
    order[n].he = j;
    order[n].tw = he->twin_idx;
    order[n].len = 0.0;
    order[n].pos = n;
    n++;
  }
  return n;
}

// Refresh lengths and positions of the live slots (dropping dead ones).
static int mftl1RefreshSlots(const MFTL_Object* obj, mftl1Slot* order, int n)
{
  int m = 0;
  for (int i = 0; i < n; ++i)
  {
    int j = mftl1SlotAlive(obj, &order[i]);
    if (j == MFTL_ERROR || obj->half_edges[j].twin_idx == MFTL_ERROR) continue;
    order[m] = order[i];
    order[m].he = j;
    order[m].tw = obj->half_edges[j].twin_idx;
    order[m].len = mftl1CurrentLength(obj, j);
    order[m].pos = m;
    m++;
  }
  return m;
}

/* MFTL_Remeshing's three sweeps (MFTL_Refining, MFTL_Coarsening, MFTL_Flipping), including
   the edge ORDER that carries over between them: uthash's HASH_SORT reorders MFTL1's edge
   list in place, so the collapse sweep sees the edges shortest-first and the flip sweep
   visits them in the order the last sort left (plus edges created since, in creation
   order). The initial order is the half-edge index order. */
void MFTL1Sweeps(MFTL_Object* obj, MFTL_RemeshParameters* params, int* n_split, int* n_collapse, int* n_flip)
{
  *n_split = *n_collapse = *n_flip = 0;
  int cap = obj->n_half_edges + 64;
  mftl1Slot* order = malloc((size_t) cap * sizeof(*order));
  int n = mftl1RefreshSlots(obj, order, mftl1AppendEdges(obj, 0, order, 0));

  // --- MFTL_Refining: longest first, stop below the reference length -------------------
  if (!params->skip_split)
  {
    qsort(order, (size_t) n, sizeof(*order), mftl1CmpSlotLong);
    const int n_he0 = obj->n_half_edges;
    for (int i = 0; i < n; ++i)
    {
      int j = mftl1SlotAlive(obj, &order[i]);
      if (j == MFTL_ERROR) continue;
      const MFTL_HalfEdge* he = &obj->half_edges[j];
      if (he->twin_idx == MFTL_ERROR || (he->flags & MFTL_FLAG_VISITED)) continue;
      if (mftl1LengthCriterion(params) && mftl1CurrentLength(obj, j) < params->reference_length) break;
      if (!MFTL1SplitCriterion(obj, j, params)) continue;
      if (MFTLSplitEdgeWithFit(obj, j, params) >= 0) (*n_split)++;
    }
    int grow = obj->n_half_edges - n_he0;
    if (n + grow > cap)
    {
      cap = n + grow + 64;
      order = realloc(order, (size_t) cap * sizeof(*order));
    }
    n = mftl1AppendEdges(obj, n_he0, order, n);
    n = mftl1RefreshSlots(obj, order, n);
  }

  // --- MFTL_Coarsening: shortest first, stop above the reference length ------------------
  if (!params->skip_collapse)
  {
    qsort(order, (size_t) n, sizeof(*order), mftl1CmpSlotShort);
    int* slot_of_he = malloc(((size_t) obj->n_half_edges + 1) * sizeof(int));
    unsigned char* retired = calloc((size_t) n + 1, 1);
    for (int h = 0; h < obj->n_half_edges; ++h) slot_of_he[h] = -1;
    for (int i = 0; i < n; ++i)
    {
      slot_of_he[order[i].he] = i;
      slot_of_he[order[i].tw] = i;
    }
    for (int i = 0; i < n; ++i)
    {
      if (retired[i]) continue;
      int j = mftl1SlotAlive(obj, &order[i]);
      if (j == MFTL_ERROR || obj->half_edges[j].twin_idx == MFTL_ERROR) continue;
      if (mftl1LengthCriterion(params) && mftl1CurrentLength(obj, j) > params->reference_length) break;
      retired[i] = 2;  // examined (RemeshingFlag); stays in the order
      // Survivor convention of MFTL1 (MFTL_TriangleVertexesAndEdgesNew: VLeft, the origin of
      // the edge in its first triangle, survives): taking the lower-id triangle as the first
      // one, collapse the half-edge of the OTHER triangle, whose target is that origin. Exact
      // for edges as MFTL1 builds them; MFTL1's re-assignment of an edge's first triangle
      // during later operations is not tracked.
      int tw = obj->half_edges[j].twin_idx;
      if (obj->triangles[obj->half_edges[j].triangle_idx].global_id < obj->triangles[obj->half_edges[tw].triangle_idx].global_id) j = tw;
      if (!MFTL1CollapseApproved(obj, j, params, 0)) continue;
      // j = a->b collapses into the survivor b; the wing edges at a (c-a in T1, a-d in T2) die
      mftl1Quad q;
      if (mftl1QuadOf(obj, j, &q) != MFTL_SUCCESS) continue;
      MFTLCollapseEdgeWithFit(obj, j, params);
      if (!(obj->half_edges[j].flags & MFTL_FLAG_DEAD)) continue;
      (*n_collapse)++;
      retired[i] = 1;
      if (slot_of_he[q.he_ca] >= 0) retired[slot_of_he[q.he_ca]] = 1;
      if (slot_of_he[q.he_ad] >= 0) retired[slot_of_he[q.he_ad]] = 1;
    }
    int m = 0;
    for (int i = 0; i < n; ++i)
      if (retired[i] != 1) order[m++] = order[i];
    n = mftl1RefreshSlots(obj, order, m);
    free(retired);
    free(slot_of_he);
  }

  // --- MFTL_Flipping: in the current order ------------------------------------------------
  if (!params->skip_flip)
  {
    for (int i = 0; i < n; ++i)
    {
      int j = mftl1SlotAlive(obj, &order[i]);
      if (j == MFTL_ERROR) continue;
      if (!MFTL1FlipApproved(obj, j, params)) continue;
      MFTLFlipEdge(obj, j);
      (*n_flip)++;
    }
  }
  free(order);
}
