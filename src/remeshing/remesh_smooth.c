/**
 * @file remesh_smooth.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief Feature classification and tangential Laplacian smoothing.
 *
 * ## THE SMOOTHER IS A PROJECTION, AND THE FEATURE CLASS PICKS THE SUBSPACE
 *
 * Every vertex wants to move towards the centroid of its neighbourhood. Doing
 * that literally is Laplacian smoothing, and Laplacian smoothing shrinks
 * geometry: the displacement has a component along the surface normal, and that
 * component is exactly "cut the corner". A remesher must keep the surface where
 * it is while redistributing vertices ON it, so the displacement is PROJECTED
 * onto the subspace the vertex is allowed to move in:
 *
 *   FREE vertex    2D -- the tangent plane. Remove the component along the
 *                  vertex normal.
 *   RIDGE vertex   1D -- the feature curve. Keep only the component along the
 *                  ridge tangent.
 *   CORNER vertex  0D -- locked.
 *
 * Stated that way the three cases are one idea, and the middle one is the easy
 * one to get wrong. Moving a ridge vertex TO the midpoint of its two feature
 * neighbours -- the full displacement, no projection, at full strength -- is the
 * 2D-shrinkage failure applied to a 1D curve: it cuts the corner of the ridge,
 * pulling the vertex towards the chord between its neighbours. Repeated over the
 * tens of sweeps a remeshing pass performs, a preserved feature curve is
 * systematically eroded, which is the opposite of what marking it a feature was
 * for.
 *
 * ## WHERE THE TARGET POINT COMES FROM
 *
 * MFTL_SMOOTH_TRIANGLE_BASED is the AREA-WEIGHTED centroid of the incident
 * triangles,
 *
 *     q = SUM(A_i * c_i) / SUM(A_i)
 *
 * and the weight is the mechanism, not a refinement. It is what makes the
 * smoother equalise triangle AREA: a vertex with one large and one small incident
 * triangle is pulled towards the large one, which shrinks it and grows the small
 * one. The UNWEIGHTED mean of the same centroids is very nearly the umbrella
 * operator again -- it equalises edge lengths and does nothing about area -- so a
 * caller selecting "triangle based" for its isotropy would get a slightly more
 * expensive MFTL_SMOOTH_VERTEX_BASED instead. Measured in
 * docs/analysis.md §4, where dropping the weight costs volume
 * conservation (6.0 % against 3.5 %) rather than triangle quality.
 *
 * ## WHY THE NORMAL IS NOT COMPUTED HERE
 *
 * MFTLComputeVertexNormals() gives the exact area-weighted normal for every
 * vertex in one flat triangle pass -- no fan walk or dependent topology loads,
 * and correct across a periodic seam. Accumulating a
 * cross product per vertex over its own one-ring, inside this loop, is the same
 * arithmetic done 3.4x slower (src/geometry/vertex_geometry.c) and invites the
 * boundary bug on top: walking the ring with is_inner = true means the
 * (k+1) % degree wrap closes a fan that is not closed, and adds a cross product
 * between two edges that are not adjacent.
 */

#include "mftl_remeshing.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Fan traversal
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Outgoing half-edges of @p v in rotational order, plus fan openness.
 *
 * MFTLGetVertexOneRingHalfedges does the walk but does not report whether the
 * fan closed, and every caller here needs to know: on an OPEN fan there is one
 * more incident EDGE than there are outgoing half-edges, and it is reachable only
 * as the incoming half-edge prev(first). A consumer that iterates edges and does
 * not account for it silently drops one, and then misclassifies every boundary
 * vertex.
 *
 * @param      obj  The mesh.
 * @param      v    Vertex index.
 * @param[out] out  `[MFTL_REMESH_RING_MAX]` outgoing half-edges, rotational order.
 * @param[out] open Set when the fan is open, i.e. @p v is on a boundary.
 * @return Number of half-edges written, or 0 if the walk was truncated.
 */
static int fan_halfedges(const MFTL_Object *obj, int v, int *out, bool *open)
{
  const int first = obj->verts.rec[v].out_he_idx;
  *open = MFTL_FALSE;
  if (first == MFTL_ERROR) return 0;

  const int start = MFTLVertexFanStart(obj, first, open);

  int count = 0;
  int curr = start;
  do
  {
    if (count >= MFTL_REMESH_RING_MAX) return MFTL_ERROR;
    out[count++] = curr;

    const int twin = obj->tris.he[curr].twin_idx;
    if (twin == MFTL_ERROR) break;
    curr = MFTLHeNextIdx(twin);
  } while (curr != start);

  return count;
}

/**
 * @brief The neighbour reachable only through the closing edge of an open fan.
 *
 * @p first is the fan's first outgoing half-edge v->w. Its triangle is (v, w, x),
 * so next(first) targets x -- and the edge v-x exists in this partition only as
 * the incoming half-edge prev(first).
 *
 * @brief The neighbour an OUTGOING-half-edge walk cannot reach.
 *
 * The closing edge of an open fan exists only in the incoming direction, so a
 * walk over outgoing half-edges is short by exactly one neighbour -- and a
 * boundary vertex counted that way lands in the wrong feature class.
 *
 * @param obj   The mesh.
 * @param first First outgoing half-edge of the fan.
 * @return The missing neighbour's vertex index.
 */
static inline int open_fan_last_neighbour(const MFTL_Object *obj, int first) { return MFTLHeTarget(obj, MFTLHeNextIdx(first)); }

/* ═══════════════════════════════════════════════════════════════════════════
 * Feature detection
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees, double corner_angle_degrees, MFTL_RemeshStats *stats)
{
  MFTL_ASSERT(obj != MFTL_NULL);

  const int n_he = 3 * obj->tris.n;
  const int n_v = obj->verts.n;

  MFTL_Vertex *const vr = obj->verts.rec;
  MFTL_HalfEdge *const he = obj->tris.he;

  /* Every vertex is reclassified from scratch, so the counts start clean even
   * when there is no topology left to classify. */
  for (int v = 0; v < n_v; ++v) vr[v].flags &= ~(MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER);

  if (n_he == 0) return 0;

  /* The normals are recomputed rather than trusted. A detector reading whatever
   * happens to be in `tris.normal` classifies the mesh as it was one topological
   * edit ago, and after a split the new triangles' normals have never been
   * written at all -- so the very edges most likely to be misjudged are the ones
   * the remesher just created. */
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* A threshold of 180 degrees or more means "no dihedral angle is sharp". It is
   * a supported production setting, not a debug switch: on a closed smooth
   * interface every edge the detector flags is a false positive, and a false
   * positive here LOCKS a vertex the smoother needed to move. */
  /* An edge is sharp when the dot product of its two face normals falls BELOW the
   * threshold, so "never sharp" is a threshold below every possible dot product,
   * i.e. -2. Writing +2 here disables nothing -- it flags every edge in the mesh,
   * which locks every vertex and silently turns the smoother off instead. */
  const bool angle_enabled = (angle_degrees < 180.0);
  const double threshold_dot = angle_enabled ? MFTL_COS(angle_degrees * MFTL_PI / 180.0) : -2.0;

  const double *const tn = obj->tris.normal;

  int n_feature_edges = 0;

  for (int h = 0; h < n_he; ++h)
  {
    const int tw = he[h].twin_idx;

    /* A mesh boundary is a feature of the SURFACE rather than of its curvature,
     * so it is one at any threshold. It also has no dihedral angle to measure. */
    if (tw == MFTL_ERROR)
    {
      /* "No twin" and "mesh boundary" are different statements, and only one is
       * partition-independent (mftl_topology.h). On an OWNED triangle they
       * coincide: the redistribution halo gives every owned triangle all three of
       * its edge neighbours, so a missing twin really is the edge of the surface.
       * On a GHOST triangle a missing twin is the outer rim of the halo, and
       * flagging it would invent a feature curve that follows the partition --
       * every rank would preserve a different one, and the "curve" would move
       * with the next migration. MFTL_FLAG_BOUNDARY is the authoritative answer
       * there, because it was written while the mesh was still complete and
       * travels with the entity.
       *
       * MFTLHeIsGhost is identically false on a serial object, so this costs one
       * predictable compare and changes nothing about the serial classification. */
      if (MFTLHeIsGhost(obj, h) && !(he[h].flags & MFTL_FLAG_BOUNDARY))
      {
        he[h].flags &= ~MFTL_FLAG_FEATURE;
        continue;
      }

      he[h].flags |= MFTL_FLAG_FEATURE;
      n_feature_edges++;
      continue;
    }

    if (h > tw) continue;  // one visit per undirected edge

    const int t1 = MFTLHeTriIdx(h);
    const int t2 = MFTLHeTriIdx(tw);

    const double dot = tn[3 * t1 + 0] * tn[3 * t2 + 0] + tn[3 * t1 + 1] * tn[3 * t2 + 1] + tn[3 * t1 + 2] * tn[3 * t2 + 2];

    if (dot < threshold_dot)
    {
      he[h].flags |= MFTL_FLAG_FEATURE;
      he[tw].flags |= MFTL_FLAG_FEATURE;
      n_feature_edges++;
    }
    else
    {
      /* Cleared, not merely left alone: an edge that WAS sharp and has since been
       * smoothed out must stop being a feature, or the flag set only ever grows
       * and the mesh gradually freezes solid. */
      he[h].flags &= ~MFTL_FLAG_FEATURE;
      he[tw].flags &= ~MFTL_FLAG_FEATURE;
    }
  }

  /* ── Vertex classification ────────────────────────────────────────────────
   *
   * Counted over UNDIRECTED edges in one linear pass, deliberately not by walking
   * each vertex's fan. The fan walk visits outgoing half-edges, and the closing
   * edge of an OPEN fan has no outgoing direction -- so every boundary vertex
   * comes out one short and lands in the wrong class. mftl_topology.h documents
   * that trap on MFTLGetVertexOneRingHalfedges. This pass cannot walk into it: it
   * never asks a vertex anything, it asks the edges and tells the vertices. */
  const size_t nv_alloc = (size_t) (n_v > 0 ? n_v : 1);
  int *const feat_count = (int *) MFTLCallocOrDie(nv_alloc, sizeof(int), "remesh feature counts");

  /* The first two feature neighbours of each vertex, recorded during the same
   * pass. A vertex with exactly two of them needs their DIRECTIONS to decide
   * whether it is a ridge point or a corner, and picking them up here costs two
   * stores against a fan walk per candidate vertex afterwards. */
  int *const feat_nbr = (int *) MFTLMallocOrDie(nv_alloc * 2 * sizeof(int), "remesh feature neighbours");
  for (size_t i = 0; i < nv_alloc * 2; ++i) feat_nbr[i] = MFTL_ERROR;

  for (int h = 0; h < n_he; ++h)
  {
    if (!(he[h].flags & MFTL_FLAG_FEATURE)) continue;

    const int tw = he[h].twin_idx;
    if (tw != MFTL_ERROR && h > tw) continue;  // count the edge, not both of its halves

    const int o = MFTLHeOrigin(obj, h);
    const int t = he[h].vertex_idx;

    if (feat_count[o] < 2) feat_nbr[2 * o + feat_count[o]] = t;
    if (feat_count[t] < 2) feat_nbr[2 * t + feat_count[t]] = o;

    feat_count[o]++;
    feat_count[t]++;
  }

  /* A feature curve turning by more than corner_angle_degrees is not a curve the
   * vertex may slide along -- it is a corner OF that curve. Straight-through means
   * the two edge directions are opposite, i.e. their normalised dot product is
   * -1, so the test is "further from -1 than the threshold allows". */
  const double straight_dot = -MFTL_COS(corner_angle_degrees * MFTL_PI / 180.0);

  int n_corner = 0;
  int n_ridge = 0;

  for (int v = 0; v < n_v; ++v)
  {
    const int c = feat_count[v];
    if (c == 0) continue;

    bool is_ridge = MFTL_FALSE;

    if (c == 2 && feat_nbr[2 * v] != MFTL_ERROR && feat_nbr[2 * v + 1] != MFTL_ERROR)
    {
      double d1[3];
      double d2[3];
      const double *const p = MFTLVertexPos(obj, v);
      MFTLGetMinimumImageVector(obj->domain, d1, p, MFTLVertexPos(obj, feat_nbr[2 * v]));
      MFTLGetMinimumImageVector(obj->domain, d2, p, MFTLVertexPos(obj, feat_nbr[2 * v + 1]));

      const double l1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
      const double l2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];

      if (l1 > 0.0 && l2 > 0.0)
      {
        const double dot = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / MFTL_SQRT(l1 * l2);
        is_ridge = (dot <= straight_dot);
      }
    }

    if (is_ridge)
    {
      /* Two feature edges continuing roughly straight make a curve passing
       * THROUGH the vertex: it may slide along that curve. */
      vr[v].flags |= MFTL_FLAG_FEATURE;
      n_ridge++;
    }
    else
    {
      /* One feature edge is the free END of a curve, three or more is a junction,
       * and two that turn sharply is a corner of the curve itself. All three are
       * points the curve is pinned to; sliding any of them deforms the feature
       * rather than redistributing along it. */
      vr[v].flags |= MFTL_FLAG_CORNER;
      n_corner++;
    }
  }

  free(feat_nbr);
  free(feat_count);

  if (stats)
  {
    stats->n_feature_edges = n_feature_edges;
    stats->n_corner_vertices = n_corner;
    stats->n_ridge_vertices = n_ridge;
    stats->n_free_vertices = n_v - n_corner - n_ridge;
  }

  return n_feature_edges;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Smoothing
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Displacement of a RIDGE vertex, projected onto its feature curve.
 *
 * @param      obj         The mesh.
 * @param      v           A ridge vertex: exactly two feature edges that continue
 *                         roughly straight through it.
 * @param      fan         Outgoing half-edges of @p v, from fan_halfedges().
 * @param      n_fan       Their count.
 * @param      open        `true` when the fan is open.
 * @param[out] disp        The displacement, projected onto the ridge TANGENT.
 * @param[out] min_edge_sq Shortest incident squared edge length, for the step clamp.
 * @return `true` if a displacement was produced.
 */
static bool ridge_displacement(const MFTL_Object *obj, int v, const int *fan, int n_fan, bool open, double disp[3], double *min_edge_sq)
{
  const double *const p = MFTLVertexPos(obj, v);

  int nbr[2];
  int n_nbr = 0;

  for (int i = 0; i < n_fan && n_nbr < 2; ++i)
  {
    if (obj->tris.he[fan[i]].flags & MFTL_FLAG_FEATURE) nbr[n_nbr++] = MFTLHeTarget(obj, fan[i]);
  }

  /* The closing edge of an open fan, which has no outgoing half-edge to carry
   * its flag -- it is prev(first), reachable only as an incoming edge. */
  if (open && n_nbr < 2 && n_fan > 0)
  {
    const int closing = MFTLHePrevIdx(fan[0]);
    if (obj->tris.he[closing].flags & MFTL_FLAG_FEATURE) nbr[n_nbr++] = open_fan_last_neighbour(obj, fan[0]);
  }

  if (n_nbr != 2) return MFTL_FALSE;

  double d1[3];
  double d2[3];
  MFTLGetMinimumImageVector(obj->domain, d1, p, MFTLVertexPos(obj, nbr[0]));
  MFTLGetMinimumImageVector(obj->domain, d2, p, MFTLVertexPos(obj, nbr[1]));

  const double l1 = d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2];
  const double l2 = d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2];
  *min_edge_sq = MFTL_MIN(l1, l2);

  /* tau spans the curve; delta is the Laplacian pull towards the chord midpoint.
   * Keeping only delta's component along tau redistributes the vertex BETWEEN its
   * neighbours -- equalising the two edge lengths -- while the component that
   * would have pulled it off the curve and onto the chord is discarded. That
   * discarded component is the whole difference between this and a chord cut, and
   * it is the one that erodes ridges. */
  const double tau[3] = {d2[0] - d1[0], d2[1] - d1[1], d2[2] - d1[2]};
  const double tau_len_sq = tau[0] * tau[0] + tau[1] * tau[1] + tau[2] * tau[2];
  /* Relative to the two edges, not to 1: the test has to mean "the neighbours
   * coincide" at a millimetre as much as at a metre. */
  if (!(tau_len_sq > MFTL_VERYVERYSMALL * (l1 + l2))) return MFTL_FALSE;

  const double delta[3] = {0.5 * (d1[0] + d2[0]), 0.5 * (d1[1] + d2[1]), 0.5 * (d1[2] + d2[2])};
  const double s = (delta[0] * tau[0] + delta[1] * tau[1] + delta[2] * tau[2]) / tau_len_sq;

  disp[0] = s * tau[0];
  disp[1] = s * tau[1];
  disp[2] = s * tau[2];
  return MFTL_TRUE;
}

/**
 * @brief Displacement of a FREE vertex, projected onto its tangent plane.
 *
 * Both neighbourhood sums are accumulated as OFFSETS from the vertex rather than
 * as absolute positions. On a periodic domain that is the only form that works --
 * averaging absolute coordinates of a fan straddling a seam produces a point in
 * the middle of the domain -- and it costs nothing, because the offsets are the
 * minimum-image vectors the loop needs regardless.
 *
 * @brief The 2D case: displacement projected onto the TANGENT PLANE.
 * @param      obj         The mesh.
 * @param      v           A vertex with no feature edges.
 * @param      fan         Outgoing half-edges of @p v, from fan_halfedges().
 * @param      n_fan       Their count.
 * @param      open        `true` when the fan is open.
 * @param      mode        Which neighbourhood the target point is built from.
 * @param[out] disp        The displacement, projected onto the TANGENT PLANE.
 * @param[out] min_edge_sq Shortest incident squared edge length, for the step clamp.
 * @return `true` if a displacement was produced.
 */
static bool free_displacement(const MFTL_Object *obj, int v, const int *fan, int n_fan, bool open, MFTL_SmoothMode mode, double disp[3], double *min_edge_sq)
{
  const double *const p = MFTLVertexPos(obj, v);

  double target[3] = {0.0, 0.0, 0.0};  // offset of the target point from p
  double weight_sum = 0.0;
  double min_sq = MFTL_DBL_MAX;

  if (mode == MFTL_SMOOTH_VERTEX_BASED)
  {
    for (int i = 0; i < n_fan; ++i)
    {
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, p, MFTLVertexPos(obj, MFTLHeTarget(obj, fan[i])));

      target[0] += d[0];
      target[1] += d[1];
      target[2] += d[2];
      weight_sum += 1.0;
      min_sq = MFTL_MIN(min_sq, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }

    if (open && n_fan > 0)
    {
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, p, MFTLVertexPos(obj, open_fan_last_neighbour(obj, fan[0])));

      target[0] += d[0];
      target[1] += d[1];
      target[2] += d[2];
      weight_sum += 1.0;
      min_sq = MFTL_MIN(min_sq, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
  }
  else
  {
    /* One outgoing half-edge per incident triangle, and the triangle's other two
     * corners are target(o) and target(next(o)). Its centroid offset from p is
     * (d1 + d2) / 3 and twice its area is |d1 x d2| -- so the area weight costs
     * one cross product that is not otherwise needed, and no access to
     * `tris.area`, which the operators have left stale anyway. */
    for (int i = 0; i < n_fan; ++i)
    {
      const int o = fan[i];

      double d1[3];
      double d2[3];
      MFTLGetMinimumImageVector(obj->domain, d1, p, MFTLVertexPos(obj, MFTLHeTarget(obj, o)));
      MFTLGetMinimumImageVector(obj->domain, d2, p, MFTLVertexPos(obj, MFTLHeTarget(obj, MFTLHeNextIdx(o))));

      const double cr[3] = {d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2], d1[0] * d2[1] - d1[1] * d2[0]};
      const double area2 = MFTL_SQRT(cr[0] * cr[0] + cr[1] * cr[1] + cr[2] * cr[2]);

      target[0] += area2 * (d1[0] + d2[0]) * MFTL_ONE_THIRD;
      target[1] += area2 * (d1[1] + d2[1]) * MFTL_ONE_THIRD;
      target[2] += area2 * (d1[2] + d2[2]) * MFTL_ONE_THIRD;
      weight_sum += area2;

      min_sq = MFTL_MIN(min_sq, d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]);
    }

    if (open && n_fan > 0)
    {
      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, p, MFTLVertexPos(obj, open_fan_last_neighbour(obj, fan[0])));
      min_sq = MFTL_MIN(min_sq, d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
    }
  }

  /* Zero only when every fan triangle is exactly degenerate. A small but
   * non-zero sum is still a well-defined weighted mean -- numerator and
   * denominator carry the same areas -- so no length scale belongs here. */
  if (!(weight_sum > 0.0)) return MFTL_FALSE;

  const double inv = 1.0 / weight_sum;
  double delta[3] = {target[0] * inv, target[1] * inv, target[2] * inv};

  /* MFTLComputeVertexNormals writes exactly zero when a vertex's accumulated
   * normal had no length -- a deliberate sentinel meaning "the one-ring is
   * degenerate". Projecting against it would leave the full, un-projected
   * Laplacian displacement, which is the shrinking operator this function exists
   * to avoid. Leaving the vertex alone is the right answer. */
  const double *const n = &obj->verts.normal[3 * v];
  const double n_len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
  if (n_len_sq < 0.5) return MFTL_FALSE;

  const double dn = delta[0] * n[0] + delta[1] * n[1] + delta[2] * n[2];
  disp[0] = delta[0] - dn * n[0];
  disp[1] = delta[1] - dn * n[1];
  disp[2] = delta[2] - dn * n[2];

  *min_edge_sq = min_sq;
  return MFTL_TRUE;
}

void MFTLSmoothTangential(MFTL_Object *obj, const MFTL_RemeshParameters *params, int iterations)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);
  MFTL_ASSERT(MFTLObjectHasTopology(obj) && "MFTLSmoothTangential: derived topology is stale -- MFTLObjectBuildDerived first");

  const int n_v = obj->verts.n;
  if (n_v == 0 || obj->tris.n == 0 || iterations <= 0) return;

  /* Jacobi, not Gauss-Seidel: every new position is computed from the OLD ones
   * and applied afterwards. Sweeping in place would make the result depend on the
   * pool ordering, which MFTLObjectReorderMorton is free to change -- so the same
   * mesh would smooth differently after a reordering that is documented as
   * changing nothing but locality. */
  double *const new_pos = (double *) MFTLMallocOrDie((size_t) n_v * 3 * sizeof(double), "remesh smoothing positions");

  const double relax = params->smooth_relaxation;
  const double step_frac = params->smooth_max_step_fraction;

  for (int sweep = 0; sweep < iterations; ++sweep)
  {
    /* Normals are refreshed from current positions every sweep. */
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);

    memcpy(new_pos, obj->verts.pos, (size_t) n_v * 3 * sizeof(double));

    for (int v = 0; v < n_v; ++v)  // why also over ghosts? Dont I want to smooth only locals?
    {
      const MFTL_Vertex *const rec = &obj->verts.rec[v];

      /* MFTL_FLAG_FROZEN joins the skip mask rather than getting a branch of its
       * own: a frozen vertex is one another rank holds a copy of, so moving it
       * would silently give two ranks different positions for one vertex. Ghosts
       * carry the flag too, which is why the loop bound stays `verts.n` -- READING
       * a frozen or ghost neighbour's position is not merely allowed, it is
       * required. It is what makes this pass exact across a partition boundary
       * with no communication: the one-ring of every unfrozen vertex is complete
       * and current, because its owner was equally forbidden to move it. */
      if (rec->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_CORNER | MFTL_FLAG_FROZEN)) continue;
      if (rec->out_he_idx == MFTL_ERROR) continue;

      int fan[MFTL_REMESH_RING_MAX];
      bool open = MFTL_FALSE;
      const int n_fan = fan_halfedges(obj, v, fan, &open);
      if (n_fan <= 0) continue;  // isolated, or a fan too large to traverse

      double disp[3];
      double min_edge_sq = MFTL_DBL_MAX;

      const bool moved = (rec->flags & MFTL_FLAG_FEATURE) ? ridge_displacement(obj, v, fan, n_fan, open, disp, &min_edge_sq)
                                                          : free_displacement(obj, v, fan, n_fan, open, params->smooth_mode, disp, &min_edge_sq);
      if (!moved) continue;

      double scale = relax;

      /* One step may not exceed a fraction of the shortest incident edge. The
       * tangential projection makes a fold-over unlikely, not impossible: it is
       * taken against a vertex normal that is itself meaningless where the
       * one-ring has already degenerated, which is exactly where a large step
       * would do the most damage. The clamp costs one comparison against a length
       * the displacement functions have already computed. */
      const double disp_sq = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];
      const double limit_sq = step_frac * step_frac * min_edge_sq;

      if (disp_sq * relax * relax > limit_sq && disp_sq > 0.0) scale = MFTL_SQRT(limit_sq / disp_sq);

      double p[3] = {obj->verts.pos[3 * v + 0] + scale * disp[0], obj->verts.pos[3 * v + 1] + scale * disp[1], obj->verts.pos[3 * v + 2] + scale * disp[2]};
      MFTLApplyPeriodicity(obj->domain, p);

      new_pos[3 * v + 0] = p[0];
      new_pos[3 * v + 1] = p[1];
      new_pos[3 * v + 2] = p[2];
    }

    memcpy(obj->verts.pos, new_pos, (size_t) n_v * 3 * sizeof(double));
  }

  free(new_pos);

}
