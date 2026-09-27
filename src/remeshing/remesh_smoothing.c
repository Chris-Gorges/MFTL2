/*
 * MFTL2 remeshing — tangential smoothing (MFTLSmoothTangential, the params-driven
 * MFTLSmoothTangentialWithParams), feature detection, the MFTL1 triangle quality, the
 * curvature-adaptive Botsch-Kobbelt threshold and the gated flip (FlipApprovedWithParams).
 * Smoothed positions follow the displacement rule (image counters, plans/16 §3.2).
 */

#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_remeshing.h"
#include "mftl_images.h"
#include "mftl_boundary_internal.h"
#include "mftl_phase_set.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "remeshing_internal.h"

/**
 * @brief Tangential Laplacian smoothing (3 Jacobi sweeps): an interior vertex moves towards
 * the centroid of its neighbours (VERTEX_BASED) or of its fan triangles' centres
 * (TRIANGLE_BASED), projected onto its tangent plane and scaled by `relaxation`; a vertex on
 * a feature line (2 feature edges) moves along the line; corners, vertices with more feature
 * edges and vertices on a boundary or rim stay fixed. Positions are updated with the image
 * counters (wrap).
 */
void MFTLSmoothTangential(MFTL_Object *obj, MFTL_SmoothMode mode, double relaxation)
{
  if (obj->n_vertices == 0) return;

  // new positions of the sweep (simultaneous / Jacobi update)
  double (*new_pos)[3] = malloc(obj->n_vertices * sizeof(double[3]));
  assert(new_pos != MFTL_NULL);

  MFTL_Vertex *ring_verts[32];
  MFTL_Triangle *ring_tris[32];

  int smooth_iters = 3;
  for (int s = 0; s < smooth_iters; s++)
  {
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex *v = &obj->vertices[i];

      if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) continue;

      new_pos[i][0] = v->pos[0];
      new_pos[i][1] = v->pos[1];
      new_pos[i][2] = v->pos[2];

      // census of the fan: edges without a stored twin and feature edges
      int bnd_count = 0;
      int feat_count = 0;
      double line_center[3] = {0, 0, 0};

      // Neighbour contributions are re-imaged into v's periodic image (exact
      // copies when non-periodic), so a ring crossing a seam stays contiguous.
      MFTL_HalfEdge *curr = MFTLVOutHe(obj, v);
      MFTL_HalfEdge *start = curr;

      if (curr != MFTL_NULL)
      {
        do
        {
          if (curr->twin_idx == MFTL_ERROR)
          {
            bnd_count++;
            int nbr_v = obj->half_edges[curr->next_idx].vertex_idx;
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, obj->vertices[nbr_v].pos, v->pos);
            line_center[0] += u[0];
            line_center[1] += u[1];
            line_center[2] += u[2];
          }
          else if (curr->flags & MFTL_FLAG_FEATURE)
          {
            feat_count++;
            int nbr_v = curr->vertex_idx;
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, obj->vertices[nbr_v].pos, v->pos);
            line_center[0] += u[0];
            line_center[1] += u[1];
            line_center[2] += u[2];
          }

          MFTL_HalfEdge *twin = MFTLHeTwin(obj, curr);
          if (!twin) break;
          curr = MFTLHeNext(obj, twin);
        } while (curr != start);
      }

      if (bnd_count > 0 || feat_count > 2)
      {
        // boundary / rim or corner: fixed
      }
      else if (feat_count == 2)
      {
        // feature line: midpoint of its two line neighbours
        new_pos[i][0] = 0.5 * line_center[0];
        new_pos[i][1] = 0.5 * line_center[1];
        new_pos[i][2] = 0.5 * line_center[2];
      }
      else
      {
        // interior (0 or 1 feature edge): tangential move towards the centroid
        double center[3] = {0, 0, 0};
        int degree = 0;

        if (mode == MFTL_SMOOTH_VERTEX_BASED)
        {
          degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            double u[3];
            MFTLUnwrapToAnchor(obj->domain, u, ring_verts[k]->pos, v->pos);
            center[0] += u[0];
            center[1] += u[1];
            center[2] += u[2];
          }
        }
        else  // MFTL_SMOOTH_TRIANGLE_BASED
        {
          degree = MFTLGetVertexOneRing(obj, v, ring_tris, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            MFTL_Vertex *t_verts[3];
            MFTLGetTriangleVertices(obj, ring_tris[k], t_verts);
            double u0[3], u1[3], u2[3];
            MFTLUnwrapToAnchor(obj->domain, u0, t_verts[0]->pos, v->pos);
            MFTLUnwrapToAnchor(obj->domain, u1, t_verts[1]->pos, v->pos);
            MFTLUnwrapToAnchor(obj->domain, u2, t_verts[2]->pos, v->pos);
            center[0] += (u0[0] + u1[0] + u2[0]) / 3.0;
            center[1] += (u0[1] + u1[1] + u2[1]) / 3.0;
            center[2] += (u0[2] + u1[2] + u2[2]) / 3.0;
          }
        }

        if (degree > 0)
        {
          center[0] /= degree;
          center[1] /= degree;
          center[2] /= degree;

          double disp[3] = {center[0] - v->pos[0], center[1] - v->pos[1], center[2] - v->pos[2]};

          // fan normal (sum of the cross products of consecutive neighbour legs)
          double nx = 0, ny = 0, nz = 0;
          degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 32, MFTL_TRUE);
          for (int k = 0; k < degree; ++k)
          {
            MFTL_Vertex *v1 = ring_verts[k];
            MFTL_Vertex *v2 = ring_verts[(k + 1) % degree];

            double d1[3], d2[3];
            MFTLGetMinimumImageVector(obj->domain, d1, v->pos, v1->pos);
            MFTLGetMinimumImageVector(obj->domain, d2, v->pos, v2->pos);

            nx += (d1[1] * d2[2] - d1[2] * d2[1]);
            ny += (d1[2] * d2[0] - d1[0] * d2[2]);
            nz += (d1[0] * d2[1] - d1[1] * d2[0]);
          }

          double n_len = sqrt(nx * nx + ny * ny + nz * nz);
          if (n_len > 1e-12)
          {
            nx /= n_len;
            ny /= n_len;
            nz /= n_len;
          }

          // tangential part of the displacement
          double dot = disp[0] * nx + disp[1] * ny + disp[2] * nz;

          double tangent_disp[3] = {disp[0] - dot * nx, disp[1] - dot * ny, disp[2] - dot * nz};

          new_pos[i][0] = v->pos[0] + relaxation * tangent_disp[0];
          new_pos[i][1] = v->pos[1] + relaxation * tangent_disp[1];
          new_pos[i][2] = v->pos[2] + relaxation * tangent_disp[2];
        }
      }
    }

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if ((obj->vertices[i].flags & MFTL_FLAG_DEAD) || obj->vertices[i].out_he_idx == MFTL_ERROR) continue;

      obj->vertices[i].pos[0] = new_pos[i][0];
      obj->vertices[i].pos[1] = new_pos[i][1];
      obj->vertices[i].pos[2] = new_pos[i][2];
      // keep the wrapped-storage invariant (no-op when non-periodic); new_pos = pos + short move
      MFTLWrapPosImg(obj->domain, obj->vertices[i].pos, obj->vertices[i].img);
    }
  }
  free(new_pos);
}

// Sets MFTL_FLAG_FEATURE on both half-edges of every edge whose adjacent (cached) triangle
// normals differ by more than angle_degrees and clears it otherwise; a half-edge without a
// stored twin (boundary or rim) is always flagged.
void MFTLDetectFeatures(MFTL_Object *obj, double angle_degrees)
{
  // angle > threshold <=> dot < cos(threshold)
  double threshold_dot = MFTL_COS(angle_degrees * MFTL_PI / 180.0);

  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge *he = &obj->half_edges[i];

    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (he->twin_idx == MFTL_ERROR)
    {
      he->flags |= MFTL_FLAG_FEATURE;
      continue;
    }
    if (i > he->twin_idx) continue;  // each edge once

    int t1_idx = he->triangle_idx;
    int t2_idx = obj->half_edges[he->twin_idx].triangle_idx;

    double *n1 = obj->triangles[t1_idx].normal;
    double *n2 = obj->triangles[t2_idx].normal;

    double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];

    if (dot < threshold_dot)
    {
      he->flags |= MFTL_FLAG_FEATURE;
      obj->half_edges[he->twin_idx].flags |= MFTL_FLAG_FEATURE;
    }
    else
    {
      he->flags &= ~MFTL_FLAG_FEATURE;
      obj->half_edges[he->twin_idx].flags &= ~MFTL_FLAG_FEATURE;
    }
  }
}

/* ========================================================================== */
/* Criteria support, gated flips, params-driven smoothing                     */
/* ========================================================================== */

// MFTL1 triangle quality: q = (perimeter^2 / area) / 20.78460969.
// 1.0 = equilateral; LARGER is worse.
double MFTL_TriangleQuality(const MFTL_Object* obj, int tri_idx)
{
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);
  // minimum-image edge vectors (raw differences when non-periodic)
  double v01[3], v12[3], v20[3], v02[3];
  MFTLGetMinimumImageVector(obj->domain, v01, vv[0]->pos, vv[1]->pos);
  MFTLGetMinimumImageVector(obj->domain, v12, vv[1]->pos, vv[2]->pos);
  MFTLGetMinimumImageVector(obj->domain, v20, vv[2]->pos, vv[0]->pos);
  MFTLGetMinimumImageVector(obj->domain, v02, vv[0]->pos, vv[2]->pos);
  double e0 = v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2];
  double e1 = v12[0] * v12[0] + v12[1] * v12[1] + v12[2] * v12[2];
  double e2 = v20[0] * v20[0] + v20[1] * v20[1] + v20[2] * v20[2];
  double a = sqrt(e0), b = sqrt(e1), c = sqrt(e2);
  double p = a + b + c;
  double cx = v01[1] * v02[2] - v01[2] * v02[1];
  double cy = v01[2] * v02[0] - v01[0] * v02[2];
  double cz = v01[0] * v02[1] - v01[1] * v02[0];
  double A = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  if (A < MFTL_VERYSMALL) return 1e9;  // degenerate: worst quality
  return (p * p / A) / 20.78460969;
}

// Botsch-Kobbelt curvature-adaptive thresholds (MFTL1 edgesplit.c parity).
// kappa_max = mean + sqrt(mean^2 - gauss) at both endpoints;
// L = sqrt(6*eps/kappa - 3*eps^2) clamped to [min, max] edge length;
// split (flag 1) above 4L/3, collapse (flag 2) below 0.8L.
int MFTLBotschKobbeltCurvatureThreshold(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params, int split_or_collapse)
{
  MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v1 = obj->half_edges[he_prev].vertex_idx;
  int v2 = he->vertex_idx;

  double k1 = obj->vertices[v1].meancurvature + sqrt(MFTL_MAX(0.0, MFTL_SQ(obj->vertices[v1].meancurvature) - obj->vertices[v1].gaussiancurvature));
  double k2 = obj->vertices[v2].meancurvature + sqrt(MFTL_MAX(0.0, MFTL_SQ(obj->vertices[v2].meancurvature) - obj->vertices[v2].gaussiancurvature));
  k1 = MFTL_MAX(k1, MFTL_VERYSMALL);
  k2 = MFTL_MAX(k2, MFTL_VERYSMALL);

  double L1 = sqrt(MFTL_MAX(0.0, (6.0 * params->epsilon / k1) - 3.0 * MFTL_SQ(params->epsilon)));
  double L2 = sqrt(MFTL_MAX(0.0, (6.0 * params->epsilon / k2) - 3.0 * MFTL_SQ(params->epsilon)));
  double L = MFTL_MIN(L1, L2);
  L = MFTL_MIN(L, params->max_edge_length);
  L = MFTL_MAX(L, params->min_edge_length);

  double high = 4.0 * L / 3.0;
  double low = 0.8 * L;
  double edge_length = sqrt(MFTLGetEdgeLengthSquared(obj, he));

  if (split_or_collapse == 1) return edge_length > high;
  return edge_length < low;
}

static inline double mftlTriQualityFromPos(const MFTL_Domain* dom, const double a[3], const double b[3], const double c[3])
{
  // minimum-image edge vectors (raw differences when non-periodic)
  double v01[3], v12[3], v20[3], v02[3];
  MFTLGetMinimumImageVector(dom, v01, a, b);
  MFTLGetMinimumImageVector(dom, v12, b, c);
  MFTLGetMinimumImageVector(dom, v20, c, a);
  MFTLGetMinimumImageVector(dom, v02, a, c);
  double e0 = 0.0, e1 = 0.0, e2 = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    e0 += v01[k] * v01[k];
    e1 += v12[k] * v12[k];
    e2 += v20[k] * v20[k];
  }
  double p = sqrt(e0) + sqrt(e1) + sqrt(e2);
  double cx = v01[1] * v02[2] - v01[2] * v02[1];
  double cy = v01[2] * v02[0] - v01[0] * v02[2];
  double cz = v01[0] * v02[1] - v01[1] * v02[0];
  double A = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
  if (A < MFTL_VERYSMALL) return 1e9;
  return (p * p / A) / 20.78460969;
}

// mftlFlipApproved extended with the gate stack: MFTL1 quality threshold
// and the max-angle-between-triangles (dihedral) gate. params == NULL keeps
// the legacy valence-only behaviour.
int FlipApprovedWithParams(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  if (!MFTLHeIsLocal(&obj->half_edges[he_idx])) return MFTL_FALSE;  // interior LOCAL edges only
  if (params && params->mftl1_criteria && !MFTLObjectRemeshGeneralized(obj)) return MFTL1FlipApproved(obj, he_idx, params);
  if (!mftlFlipApproved(obj, he_idx)) return MFTL_FALSE;
  if (!params) return MFTL_TRUE;

  MFTL_HalfEdge* he_ab = &obj->half_edges[he_idx];
  int he_ba_idx = he_ab->twin_idx;
  int v_b = he_ab->vertex_idx;
  int v_a = obj->half_edges[he_ba_idx].vertex_idx;
  int v_c = obj->half_edges[he_ab->next_idx].vertex_idx;
  int v_d = obj->half_edges[obj->half_edges[he_ba_idx].next_idx].vertex_idx;

  const double* pa = obj->vertices[v_a].pos;
  const double* pb = obj->vertices[v_b].pos;
  const double* pc = obj->vertices[v_c].pos;
  const double* pd = obj->vertices[v_d].pos;

  // Quality gate: reject when a created triangle would be worse than the
  // configured worst quality.
  double q1 = mftlTriQualityFromPos(obj->domain, pc, pd, pa);
  double q2 = mftlTriQualityFromPos(obj->domain, pc, pd, pb);
  if (q1 > params->max_quality_threshold || q2 > params->max_quality_threshold) return MFTL_FALSE;

  // Generalized bodies: the created triangles are (c, a, d) and (d, b, c) with the
  // orientation of the original pair; the dihedral gate compares their consistently oriented
  // normals, so a flip of a non-convex quad (one created triangle inverted) is always rejected.
  // NOTE: the legacy gate below builds both normals from (c, d, x), which inverts the sign of the
  // dot product (it rejects convex flips and accepts folding ones). It is kept unchanged for closed
  // bodies in the default mode (bit-identity contract, plans/16 §0); see plans/16 §22.
  if (MFTLObjectRemeshGeneralized(obj))
  {
    double ca[3], cd[3], db[3], dc[3];
    MFTLGetMinimumImageVector(obj->domain, ca, pc, pa);
    MFTLGetMinimumImageVector(obj->domain, cd, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, db, pd, pb);
    MFTLGetMinimumImageVector(obj->domain, dc, pd, pc);
    const double n1[3] = {ca[1] * cd[2] - ca[2] * cd[1], ca[2] * cd[0] - ca[0] * cd[2], ca[0] * cd[1] - ca[1] * cd[0]};
    const double n2[3] = {db[1] * dc[2] - db[2] * dc[1], db[2] * dc[0] - db[0] * dc[2], db[0] * dc[1] - db[1] * dc[0]};
    const double l1 = sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]), l2 = sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    if (!(l1 > 0.0) || !(l2 > 0.0)) return MFTL_FALSE;  // degenerate created triangle
    double dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2);
    dot = dot > 1.0 ? 1.0 : (dot < -1.0 ? -1.0 : dot);
    const double max_angle = params->max_angle_between_tris < 179.0 ? params->max_angle_between_tris : 179.0;
    if (acos(dot) * 180.0 / MFTL_PI > max_angle) return MFTL_FALSE;
    // neither created triangle may invert with respect to the original pair (normals from the
    // current positions: cached triangle normals are stale within a pass)
    double ab[3], ac[3], ba[3], bd[3];
    MFTLGetMinimumImageVector(obj->domain, ab, pa, pb);
    MFTLGetMinimumImageVector(obj->domain, ac, pa, pc);
    MFTLGetMinimumImageVector(obj->domain, ba, pb, pa);
    MFTLGetMinimumImageVector(obj->domain, bd, pb, pd);
    const double no1[3] = {ab[1] * ac[2] - ab[2] * ac[1], ab[2] * ac[0] - ab[0] * ac[2], ab[0] * ac[1] - ab[1] * ac[0]};
    const double no2[3] = {ba[1] * bd[2] - ba[2] * bd[1], ba[2] * bd[0] - ba[0] * bd[2], ba[0] * bd[1] - ba[1] * bd[0]};
    const double nold[3] = {no1[0] + no2[0], no1[1] + no2[1], no1[2] + no2[2]};
    if (n1[0] * nold[0] + n1[1] * nold[1] + n1[2] * nold[2] <= 0.0) return MFTL_FALSE;
    if (n2[0] * nold[0] + n2[1] * nold[1] + n2[2] * nold[2] <= 0.0) return MFTL_FALSE;
    return MFTL_TRUE;
  }

  // Dihedral gate: reject when the angle between the new triangle normals
  // exceeds max_angle_between_tris (degrees).
  if (params->max_angle_between_tris < 179.0)
  {
    double n1[3], n2[3];
    // minimum-image legs (raw differences when non-periodic)
    double u1[3], u2[3], w1[3], w2[3];
    MFTLGetMinimumImageVector(obj->domain, u1, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, u2, pc, pa);
    MFTLGetMinimumImageVector(obj->domain, w1, pc, pd);
    MFTLGetMinimumImageVector(obj->domain, w2, pc, pb);
    n1[0] = u1[1] * u2[2] - u1[2] * u2[1];
    n1[1] = u1[2] * u2[0] - u1[0] * u2[2];
    n1[2] = u1[0] * u2[1] - u1[1] * u2[0];
    n2[0] = w1[1] * w2[2] - w1[2] * w2[1];
    n2[1] = w1[2] * w2[0] - w1[0] * w2[2];
    n2[2] = w1[0] * w2[1] - w1[1] * w2[0];
    double l1 = sqrt(n1[0] * n1[0] + n1[1] * n1[1] + n1[2] * n1[2]);
    double l2 = sqrt(n2[0] * n2[0] + n2[1] * n2[1] + n2[2] * n2[2]);
    if (l1 > MFTL_VERYSMALL && l2 > MFTL_VERYSMALL)
    {
      double dot = (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) / (l1 * l2);
      if (dot > 1.0) dot = 1.0;
      if (dot < -1.0) dot = -1.0;
      double angle = acos(dot) * 180.0 / MFTL_PI;
      if (angle > params->max_angle_between_tris) return MFTL_FALSE;
    }
  }

  return MFTL_TRUE;
}

// Tangential smoothing driven by params. QUALITY_DRIVEN accepts a move only
// when the local minimum triangle quality improves; CURVATURE_BASED weights
// the centroid by area * L_target(curvature) (MFTL1 tangentialrelaxation.c).
// Feature vertices and corners are left fixed in the QUALITY_DRIVEN / CURVATURE_BASED modes.
void MFTLSmoothTangentialWithParams(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  if (obj->n_vertices == 0) return;
  double relaxation = 1.0;
  int iters = params && params->max_smoothing_iter > 0 ? params->max_smoothing_iter : 3;
  MFTL_SmoothMode mode = params ? params->smooth_mode : MFTL_SMOOTH_VERTEX_BASED;

  double (*new_pos)[3] = malloc(obj->n_vertices * sizeof(double[3]));
  assert(new_pos != MFTL_NULL);
  MFTL_Vertex* ring_verts[64];
  MFTL_Triangle* ring_tris[32];
  const int generalized = MFTLObjectRemeshGeneralized(obj);

  for (int s = 0; s < iters; ++s)
  {
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost) continue;  // ghosts are moved by their owner, never locally
      if ((v->flags & MFTL_FLAG_DEAD) || v->out_he_idx == MFTL_ERROR) continue;
      new_pos[i][0] = v->pos[0];
      new_pos[i][1] = v->pos[1];
      new_pos[i][2] = v->pos[2];

      // feature / boundary census
      int bnd_count = 0;
      int feat_count = 0;
      int he = v->out_he_idx;
      int start = he;
      do
      {
        MFTL_HalfEdge* curr = &obj->half_edges[he];
        if (curr->twin_idx == MFTL_ERROR)
          bnd_count++;
        else if (curr->flags & MFTL_FLAG_FEATURE)
          feat_count++;
        if (curr->twin_idx == MFTL_ERROR) break;
        he = obj->half_edges[curr->twin_idx].next_idx;
      } while (he != start && he != MFTL_ERROR);

      int locked = (bnd_count > 0 || feat_count > 0);
      // A generalized fan larger than the one-ring buffers below (refinement around a
      // high-valence vertex) is left in place instead of overflowing them
      if (generalized)
      {
        int nfan = 0;
        MFTLVertexRing(obj, i, MFTL_NULL, 0, &nfan);
        if (nfan >= 32) continue;
      }
      if (mode == MFTL_SMOOTH_QUALITY_DRIVEN || mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        // QUALITY_DRIVEN / CURVATURE_BASED: feature vertices are fixed
        if (locked) continue;
      }
      else if (locked)
      {
        continue;  // legacy modes keep the existing ridge/corner handling elsewhere
      }

      // candidate centroid
      double center[3] = {0, 0, 0};
      double denom = 0.0;
      if (mode == MFTL_SMOOTH_CURVATURE_BASED)
      {
        // area * L_target weighted triangle centres (MFTL1 CurvatureBased)
        int nt = MFTLGetVertexOneRing(obj, v, ring_tris, 32, MFTL_TRUE);
        for (int k = 0; k < nt; ++k)
        {
          MFTL_Triangle* t = ring_tris[k];
          double Lavg = 0.0;
          MFTL_Vertex* tv[3];
          MFTLGetTriangleVertices(obj, t, tv);
          for (int j = 0; j < 3; ++j)
          {
            double mc = tv[j]->meancurvature;
            double gc = tv[j]->gaussiancurvature;
            double kmax = mc + sqrt(MFTL_MAX(0.0, MFTL_SQ(mc) - gc));
            kmax = MFTL_MAX(kmax, MFTL_VERYSMALL);
            double Lv = sqrt(MFTL_MAX(0.0, 6.0 * params->epsilon / kmax - 3.0 * MFTL_SQ(params->epsilon)));
            Lv = MFTL_MIN(Lv, params->max_edge_length);
            Lv = MFTL_MAX(Lv, params->min_edge_length);
            Lavg += Lv / 3.0;
          }
          double w = t->area * Lavg;
          // re-image the stored (wrapped) centroid into v's periodic image
          double tc[3];
          MFTLUnwrapToAnchor(obj->domain, tc, t->center, v->pos);
          center[0] += w * tc[0];
          center[1] += w * tc[1];
          center[2] += w * tc[2];
          denom += w;
        }
      }
      else
      {
        int degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 64, MFTL_TRUE);
        for (int k = 0; k < degree; ++k)
        {
          // neighbour re-imaged into v's periodic image (plain copy otherwise)
          double u[3];
          MFTLUnwrapToAnchor(obj->domain, u, ring_verts[k]->pos, v->pos);
          center[0] += u[0];
          center[1] += u[1];
          center[2] += u[2];
        }
        denom = degree;
      }
      if (denom <= 0) continue;
      center[0] /= denom;
      center[1] /= denom;
      center[2] /= denom;

      // vertex normal for the tangential projection
      double nx = 0, ny = 0, nz = 0;
      int degree = MFTLGetVertexOneRingVertices(obj, v, ring_verts, 64, MFTL_TRUE);
      for (int k = 0; k < degree; ++k)
      {
        MFTL_Vertex* v1 = ring_verts[k];
        MFTL_Vertex* v2 = ring_verts[(k + 1) % degree];
        double d1[3], d2[3];
        MFTLGetMinimumImageVector(obj->domain, d1, v->pos, v1->pos);
        MFTLGetMinimumImageVector(obj->domain, d2, v->pos, v2->pos);
        nx += (d1[1] * d2[2] - d1[2] * d2[1]);
        ny += (d1[2] * d2[0] - d1[0] * d2[2]);
        nz += (d1[0] * d2[1] - d1[1] * d2[0]);
      }
      double n_len = sqrt(nx * nx + ny * ny + nz * nz);
      if (n_len < 1e-12) continue;
      nx /= n_len;
      ny /= n_len;
      nz /= n_len;

      double disp[3] = {center[0] - v->pos[0], center[1] - v->pos[1], center[2] - v->pos[2]};
      double dot = disp[0] * nx + disp[1] * ny + disp[2] * nz;
      double cand[3] = {v->pos[0] + relaxation * (disp[0] - dot * nx), v->pos[1] + relaxation * (disp[1] - dot * ny),
                        v->pos[2] + relaxation * (disp[2] - dot * nz)};

      if (mode == MFTL_SMOOTH_QUALITY_DRIVEN)
      {
        // accept only when the local minimum quality improves
        MFTL_Triangle* tris[32];
        int nt = MFTLGetVertexOneRing(obj, v, tris, 32, MFTL_TRUE);
        double q_before = 0.0, q_after = 0.0;
        for (int k = 0; k < nt; ++k)
        {
          MFTL_Vertex* tv[3];
          MFTLGetTriangleVertices(obj, tris[k], tv);
          double qb = mftlTriQualityFromPos(obj->domain, tv[0]->pos, tv[1]->pos, tv[2]->pos);
          double p0[3], p1[3], p2[3];
          memcpy(p0, tv[0]->pos, sizeof(p0));
          memcpy(p1, tv[1]->pos, sizeof(p1));
          memcpy(p2, tv[2]->pos, sizeof(p2));
          if (tv[0] == v) memcpy(p0, cand, sizeof(p0));
          if (tv[1] == v) memcpy(p1, cand, sizeof(p1));
          if (tv[2] == v) memcpy(p2, cand, sizeof(p2));
          double qa = mftlTriQualityFromPos(obj->domain, p0, p1, p2);
          if (qb > q_before) q_before = qb;
          if (qa > q_after) q_after = qa;
        }
        if (q_after >= q_before) continue;  // no improvement: keep position
      }
      // generalized bodies never accept a move that inverts a fan triangle
      if (MFTLObjectRemeshGeneralized(obj) && !MFTLMoveKeepsOrientation(obj, i, v->pos, cand)) continue;
      // ... nor a move racing with another rank's move of a fan corner (checked only after the ghost update)
      if (MFTLObjectRemeshGeneralized(obj) && !MFTLVertexFanOwned(obj, i)) continue;

      memcpy(new_pos[i], cand, sizeof(cand));
    }

    for (int i = 0; i < obj->n_vertices; ++i)
    {
      if (obj->vertices[i].is_ghost) continue;  // ghosts are synced from their owner, never written here
      if ((obj->vertices[i].flags & MFTL_FLAG_DEAD) || obj->vertices[i].out_he_idx == MFTL_ERROR) continue;
      // The Jacobi candidates were checked against the OLD neighbour positions; for
      // generalized bodies each move is re-checked against the current ones when it is applied, so
      // simultaneous neighbour moves can never fold the surface
      if (generalized && !MFTLMoveKeepsOrientation(obj, i, obj->vertices[i].pos, new_pos[i])) continue;
      double old_pos[3];
      memcpy(old_pos, obj->vertices[i].pos, 3 * sizeof(double));
      memcpy(obj->vertices[i].pos, new_pos[i], 3 * sizeof(double));
      // keep the wrapped-storage invariant (no-op when non-periodic)
      MFTLApplyPeriodicity(obj->domain, obj->vertices[i].pos);
      MFTLImgFollow(obj->domain, obj->vertices[i].img, old_pos, obj->vertices[i].pos);  // displacement rule
    }
    // Boundary vertices of a generalized body (plans/16 §4.4.2) move only along their
    // boundary curve (arc-length redistribution); corners and frozen tags stay fixed
    if (MFTLObjectRemeshGeneralized(obj)) MFTLSmoothBoundaryVertices(obj, relaxation, MFTLBoundaryCornerAngle(params));
  }
  free(new_pos);
}
