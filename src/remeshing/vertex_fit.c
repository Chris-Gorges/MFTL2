/*
 * Vertex fitting for remeshing (Stage 08): paraboloid least-squares fit
 * (MFTL1 vertexfitting.c:ParaboloidFit, LAPACK dgelsy) and the barycentric
 * butterfly stencil (MFTL1 BarycentricInterp), used to place new split
 * midpoints and collapse survivors back onto the surface.
 *
 * Port notes:
 *  - Neighbour set = union of the one-ring vertices of the edge endpoints
 *    (MFTL1 lists the corners of the two adjacent triangles plus the four
 *    complementary triangles — the same set). If fewer than 6 neighbours
 *    are available the vertex keeps its midpoint position.
 *  - Local frame {u, v, w}: w = normalized sum of neighbour vertex normals
 *    (vertex normals must be current — the driver computes them), u =
 *    edge direction made orthogonal to w, v = w x u.
 *  - Least squares z = A x^2 + B xy + C y^2 + D x + E y + F with LAPACK
 *    dgelsy (rcond 1e-10). Without MFTL_HAVE_LAPACK, or when the fit is
 *    rank-deficient (info != 0 / rank < 6), falls back to the barycentric
 *    stencil, then to the midpoint.
 *  - The barycentric stencil applies ALL THREE coordinates; MFTL1 stored
 *    only x[2] (its x/y stores are commented out — treated here as a bug).
 *  - Feature edges are never fitted (the caller/driver keeps midpoints
 *    there); fitting is skipped when the required neighbourhood is missing.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_remeshing.h"

#ifdef MFTL_HAVE_LAPACK
int dgelsy_(const int* m, const int* n, const int* nrhs, double* a, const int* lda, double* b, const int* ldb, int* jpvt, const double* rcond,
            int* rank, double* work, const int* lwork, int* info);
#endif

#define MFTL_FIT_MAX_NBR 96

// Unit normal of a triangle from vertex positions (always current, unlike
// stored triangle normals which go stale within a remesh pass).
static void mftlTriNormalFromPos(const MFTL_Object* obj, int tri_idx, double n[3])
{
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);
  double e0[3] = {vv[1]->pos[0] - vv[0]->pos[0], vv[1]->pos[1] - vv[0]->pos[1], vv[1]->pos[2] - vv[0]->pos[2]};
  double e1[3] = {vv[2]->pos[0] - vv[0]->pos[0], vv[2]->pos[1] - vv[0]->pos[1], vv[2]->pos[2] - vv[0]->pos[2]};
  n[0] = e0[1] * e1[2] - e0[2] * e1[1];
  n[1] = e0[2] * e1[0] - e0[0] * e1[2];
  n[2] = e0[0] * e1[1] - e0[1] * e1[0];
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len < MFTL_VERYSMALL)
  {
    n[0] = n[1] = n[2] = 0.0;
    return;
  }
  n[0] /= len;
  n[1] /= len;
  n[2] /= len;
}

// Newell normal of a vertex's one-ring, computed from current positions
// (MFTL_OneRingVertexNormal equivalent; immune to stale stored normals).
static int mftlRingNormal(MFTL_Object* obj, int vi, double n[3])
{
  n[0] = n[1] = n[2] = 0.0;
  MFTL_Triangle* ring[64];
  int cnt = MFTLGetVertexOneRing(obj, &obj->vertices[vi], ring, 64, MFTL_FALSE);
  if (cnt == 0) return MFTL_ERROR;
  for (int i = 0; i < cnt; ++i)
  {
    double tn[3];
    mftlTriNormalFromPos(obj, (int) (ring[i] - obj->triangles), tn);
    n[0] += tn[0];
    n[1] += tn[1];
    n[2] += tn[2];
  }
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len < MFTL_VERYSMALL) return MFTL_ERROR;
  n[0] /= len;
  n[1] /= len;
  n[2] /= len;
  return MFTL_SUCCESS;
}

// Frame normal for the fit: normalized sum of the FRESH one-ring normals of
// the neighbour set (MFTL1 averages the neighbourhood vertex normals; stored
// normals go stale within a remesh pass, so they are recomputed here).
static int mftlFitFrameNormal(MFTL_Object* obj, const int* nbr, int nvert, double w[3])
{
  w[0] = w[1] = w[2] = 0.0;
  for (int i = 0; i < nvert; ++i)
  {
    double n[3];
    if (mftlRingNormal(obj, nbr[i], n) != MFTL_SUCCESS) continue;
    w[0] += n[0];
    w[1] += n[1];
    w[2] += n[2];
  }
  double len = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (len < MFTL_VERYSMALL) return MFTL_ERROR;
  w[0] /= len;
  w[1] /= len;
  w[2] /= len;
  return MFTL_SUCCESS;
}

static int mftlEdgeIsFeature(const MFTL_Object* obj, int he_idx)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_FEATURE) return 1;
  if (he->twin_idx != MFTL_ERROR && (obj->half_edges[he->twin_idx].flags & MFTL_FLAG_FEATURE)) return 1;
  return 0;
}

// Feature test for fitting/protection purposes: flagged OR geometrically
// sharp (angle between adjacent triangle normals exceeds the threshold), so
// protection holds even when the caller never ran MFTLDetectFeatures.
static int mftlEdgeIsFeatureEx(MFTL_Object* obj, int he_idx, double angle_degrees)
{
  if (mftlEdgeIsFeature(obj, he_idx)) return 1;
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->twin_idx == MFTL_ERROR) return 1;  // boundary: never fit
  double n1[3], n2[3];
  mftlTriNormalFromPos(obj, he->triangle_idx, n1);
  mftlTriNormalFromPos(obj, obj->half_edges[he->twin_idx].triangle_idx, n2);
  double dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
  return dot < MFTL_COS(angle_degrees * MFTL_PI / 180.0);
}

// Union of the endpoint one-rings (by vertex index). Returns the count.
static int mftlGatherFitNeighbours(MFTL_Object* obj, int he_idx, int* nbr)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int va = obj->half_edges[he_prev].vertex_idx;
  int vb = he->vertex_idx;

  int nn = 0;
  int mark[MFTL_FIT_MAX_NBR];

  MFTL_Vertex* ring[64];
  int seeds[2] = {va, vb};
  for (int s = 0; s < 2; ++s)
  {
    int cnt = MFTLGetVertexOneRingVertices(obj, &obj->vertices[seeds[s]], ring, 64, MFTL_FALSE);
    for (int i = 0; i < cnt; ++i)
    {
      int vi = (int) (ring[i] - obj->vertices);
      if (vi == va || vi == vb) continue;
      int found = 0;
      for (int m = 0; m < nn; ++m)
        if (mark[m] == vi)
        {
          found = 1;
          break;
        }
      if (!found && nn < MFTL_FIT_MAX_NBR) mark[nn++] = vi;
    }
  }
  memcpy(nbr, mark, (size_t) nn * sizeof(int));
  return nn;
}

// Butterfly stencil around the edge's two adjacent triangles:
//   x_new = 0.5(Va+Vb) + 0.25*C - 0.125*(V5+V6)
// with C the apex of he's triangle, V5/V6 the apices across edges (b,C) and
// (a,C). Falls back to the midpoint when a complementary triangle is missing.
static int mftlBarycentricFit(MFTL_Object* obj, int v_idx, int he_idx, double out[3])
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int va = obj->half_edges[he_prev].vertex_idx;
  int vb = he->vertex_idx;
  int vc = obj->half_edges[he_next].vertex_idx;  // apex of he's triangle

  // complementary triangle across edge (b -> c): twin of he_next
  int v5 = -1, v6 = -1;
  if (obj->half_edges[he_next].twin_idx != MFTL_ERROR)
  {
    int t_bc = obj->half_edges[obj->half_edges[he_next].twin_idx].triangle_idx;
    MFTL_Vertex* tv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[t_bc], tv);
    for (int i = 0; i < 3; ++i)
    {
      int vi = (int) (tv[i] - obj->vertices);
      if (vi != vb && vi != vc) v5 = vi;
    }
  }
  // complementary triangle across edge (a -> c): the half-edge (a->c) is he_prev
  if (obj->half_edges[he_prev].twin_idx != MFTL_ERROR)
  {
    int t_ac = obj->half_edges[obj->half_edges[he_prev].twin_idx].triangle_idx;
    MFTL_Vertex* tv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[t_ac], tv);
    for (int i = 0; i < 3; ++i)
    {
      int vi = (int) (tv[i] - obj->vertices);
      if (vi != va && vi != vc) v6 = vi;
    }
  }
  if (v5 < 0 || v6 < 0) return MFTL_ERROR;

  const double* pa = obj->vertices[va].pos;
  const double* pb = obj->vertices[vb].pos;
  const double* pc = obj->vertices[vc].pos;
  const double* p5 = obj->vertices[v5].pos;
  const double* p6 = obj->vertices[v6].pos;
  for (int k = 0; k < 3; ++k) out[k] = 0.5 * (pa[k] + pb[k]) + 0.25 * pc[k] - 0.125 * (p5[k] + p6[k]);
  (void) v_idx;
  return MFTL_SUCCESS;
}

// Core paraboloid fit: frame from u_dir and the summed normals of nbr[],
// least squares over the neighbour set. Returns MFTL_SUCCESS on a full-rank
// fit (position updated), MFTL_ERROR otherwise (caller keeps position).
static int mftlParaboloidFitAt(MFTL_Object* obj, int v_idx, const int* nbr, int nvert, const double u_dir[3], const double w_dir[3])
{
#ifdef MFTL_HAVE_LAPACK
  // Frame normal supplied by the caller, computed from CURRENT triangle
  // geometry (stored vertex normals go stale within a remesh pass).
  double w[3] = {w_dir[0], w_dir[1], w_dir[2]};
  double norm = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
  if (norm < MFTL_VERYSMALL) return MFTL_ERROR;
  w[0] /= norm;
  w[1] /= norm;
  w[2] /= norm;

  double c[3];
  memcpy(c, obj->vertices[v_idx].pos, 3 * sizeof(double));
  double u[3] = {u_dir[0], u_dir[1], u_dir[2]};
  double dotp = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
  for (int k = 0; k < 3; ++k) u[k] -= w[k] * dotp;
  norm = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  if (norm < MFTL_VERYSMALL) return MFTL_ERROR;
  u[0] /= norm;
  u[1] /= norm;
  u[2] /= norm;
  double v[3] = {w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]};

  // basis matrix P1 (rows u, v, w) and its inverse
  double P1[3][3] = {{u[0], u[1], u[2]}, {v[0], v[1], v[2]}, {w[0], w[1], w[2]}};
  double det = 0.0;
  for (int j = 0; j < 3; ++j) det += P1[0][j] * (P1[1][(j + 1) % 3] * P1[2][(j + 2) % 3] - P1[1][(j + 2) % 3] * P1[2][(j + 1) % 3]);
  if (fabs(det) < MFTL_VERYSMALL) return MFTL_ERROR;
  double InvP1[3][3];
  for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k)
      InvP1[j][k] = ((P1[(j + 1) % 3][(k + 1) % 3] * P1[(j + 2) % 3][(k + 2) % 3]) - (P1[(j + 1) % 3][(k + 2) % 3] * P1[(j + 2) % 3][(k + 1) % 3])) / det;

  // least squares: design matrix in LAPACK column-major layout
  int N2 = nvert, K2 = 6, nrhs = 1, lda = N2, ldb = N2, lwork = 100, info = 0, lsRank = 0;
  double P2[MFTL_FIT_MAX_NBR * 6];
  double B2[MFTL_FIT_MAX_NBR];
  for (int i = 0; i < nvert; ++i)
  {
    const double* xp = obj->vertices[nbr[i]].pos;
    double lp[3] = {0, 0, 0};
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k) lp[j] += InvP1[j][k] * (xp[k] - c[k]);
    P2[i + 0 * N2] = lp[0] * lp[0];
    P2[i + 1 * N2] = lp[0] * lp[1];
    P2[i + 2 * N2] = lp[1] * lp[1];
    P2[i + 3 * N2] = lp[0];
    P2[i + 4 * N2] = lp[1];
    P2[i + 5 * N2] = 1.0;
    B2[i] = lp[2];
  }
  int jpvt[6] = {0, 0, 0, 0, 0, 0};
  double rcond = 1.0e-10, work[100];
  dgelsy_(&N2, &K2, &nrhs, P2, &lda, B2, &ldb, jpvt, &rcond, &lsRank, work, &lwork, &info);
  if (info != 0 || lsRank < 6) return MFTL_ERROR;

  double A = B2[0], B = B2[1], C = B2[2], D = B2[3], E = B2[4], F = B2[5];
  double localV[3] = {0, 0, 0};
  const double* vp = obj->vertices[v_idx].pos;
  for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) localV[j] += InvP1[j][k] * (vp[k] - c[k]);

  double newW = A * localV[0] * localV[0] + B * localV[0] * localV[1] + C * localV[1] * localV[1] + D * localV[0] + E * localV[1] + F;
  // back to global coordinates (column convention, as MFTL1)
  double newpos[3];
  newpos[0] = (P1[0][0] * localV[0] + P1[1][0] * localV[1] + P1[2][0] * newW) + c[0];
  newpos[1] = (P1[0][1] * localV[0] + P1[1][1] * localV[1] + P1[2][1] * newW) + c[1];
  newpos[2] = (P1[0][2] * localV[0] + P1[1][2] * localV[1] + P1[2][2] * newW) + c[2];

  // Clamp: a fit that lands far outside the local neighbourhood (coarse or
  // partially-stale ring mid-pass) is rejected — the caller falls back to the
  // barycentric stencil / midpoint (plan: "fit placing a vertex far
  // off-surface on a coarse ring -> clamp / fallback").
  double scale = 0.0;
  for (int i = 0; i < nvert; ++i)
  {
    const double* xp = obj->vertices[nbr[i]].pos;
    double d0 = xp[0] - c[0], d1 = xp[1] - c[1], d2 = xp[2] - c[2];
    double d = sqrt(d0 * d0 + d1 * d1 + d2 * d2);
    if (d > scale) scale = d;
  }
  if (scale < MFTL_VERYSMALL) return MFTL_ERROR;
  double f0 = newpos[0] - c[0], f1 = newpos[1] - c[1], f2 = newpos[2] - c[2];
  if (sqrt(f0 * f0 + f1 * f1 + f2 * f2) > 0.25 * scale) return MFTL_ERROR;  // ~half a local edge

  memcpy(obj->vertices[v_idx].pos, newpos, 3 * sizeof(double));
  return MFTL_SUCCESS;
#else
  (void) obj;
  (void) v_idx;
  (void) nbr;
  (void) nvert;
  (void) u_dir;
  (void) w_dir;
  return MFTL_ERROR;  // no LAPACK: caller falls back
#endif
}

void MFTLFitVertex(MFTL_Object* obj, int v_idx, int he_idx, MFTL_RemeshParameters* params)
{
  if (!params) return;
  if (params->vertex_fit == MFTL_FIT_NONE) return;
  if (mftlEdgeIsFeatureEx(obj, he_idx, params->feature_angle_threshold)) return;  // never fit across features

  if (params->vertex_fit == MFTL_FIT_BARYCENTRIC)
  {
    double fit[3];
    if (mftlBarycentricFit(obj, v_idx, he_idx, fit) == MFTL_SUCCESS) memcpy(obj->vertices[v_idx].pos, fit, 3 * sizeof(double));
    return;
  }

  // MFTL_FIT_PARABOLOID (with barycentric / midpoint fallbacks)
  int nbr[MFTL_FIT_MAX_NBR];
  int nvert = mftlGatherFitNeighbours(obj, he_idx, nbr);
  if (nvert < 6)
  {
    double fit[3];
    if (mftlBarycentricFit(obj, v_idx, he_idx, fit) == MFTL_SUCCESS) memcpy(obj->vertices[v_idx].pos, fit, 3 * sizeof(double));
    return;  // otherwise keep the midpoint
  }

  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int va = obj->half_edges[he_prev].vertex_idx;
  int vb = he->vertex_idx;
  double u_dir[3];
  for (int k = 0; k < 3; ++k) u_dir[k] = obj->vertices[vb].pos[k] - obj->vertices[va].pos[k];
  double w_dir[3];
  if (mftlFitFrameNormal(obj, nbr, nvert, w_dir) != MFTL_SUCCESS)
  {
    double fit[3];
    if (mftlBarycentricFit(obj, v_idx, he_idx, fit) == MFTL_SUCCESS) memcpy(obj->vertices[v_idx].pos, fit, 3 * sizeof(double));
    return;
  }

  if (mftlParaboloidFitAt(obj, v_idx, nbr, nvert, u_dir, w_dir) != MFTL_SUCCESS)
  {
    double fit[3];
    if (mftlBarycentricFit(obj, v_idx, he_idx, fit) == MFTL_SUCCESS) memcpy(obj->vertices[v_idx].pos, fit, 3 * sizeof(double));
  }
}

int MFTLSplitEdgeWithFit(MFTL_Object* obj, int he_ab_idx, MFTL_RemeshParameters* params)
{
  int v_new = MFTLSplitEdge(obj, he_ab_idx);
  if (v_new != MFTL_ERROR && v_new >= 0) MFTLFitVertex(obj, v_new, he_ab_idx, params);
  return v_new;
}

void MFTLCollapseEdgeWithFit(MFTL_Object* obj, int he_ab_idx, MFTL_RemeshParameters* params)
{
  int feature = params ? mftlEdgeIsFeatureEx(obj, he_ab_idx, params->feature_angle_threshold) : mftlEdgeIsFeature(obj, he_ab_idx);
  int v_b = obj->half_edges[he_ab_idx].vertex_idx;  // survivor (he target)

  // Fit BEFORE the collapse using the intact edge stencil: the collapse
  // moves the survivor to the raw midpoint, which we then overwrite with the
  // fitted position (volume conservation, MFTL1 parity).
  double fitted[3] = {0, 0, 0};
  int have_fit = 0;
  if (params && params->vertex_fit != MFTL_FIT_NONE && !feature)
  {
    if (params->vertex_fit == MFTL_FIT_BARYCENTRIC)
    {
      have_fit = (mftlBarycentricFit(obj, v_b, he_ab_idx, fitted) == MFTL_SUCCESS);
    }
    else
    {
      // paraboloid: frame centred at the edge midpoint (the collapse target)
      const MFTL_HalfEdge* he = &obj->half_edges[he_ab_idx];
      int he_next = he->next_idx;
      int he_prev = obj->half_edges[he_next].next_idx;
      int va = obj->half_edges[he_prev].vertex_idx;
      double saved[3];
      memcpy(saved, obj->vertices[v_b].pos, 3 * sizeof(double));
      for (int k = 0; k < 3; ++k) obj->vertices[v_b].pos[k] = 0.5 * (saved[k] + obj->vertices[va].pos[k]);

      int nbr[MFTL_FIT_MAX_NBR];
      int nvert = mftlGatherFitNeighbours(obj, he_ab_idx, nbr);
      double w_dir[3];
      if (nvert >= 6 && mftlFitFrameNormal(obj, nbr, nvert, w_dir) == MFTL_SUCCESS)
      {
        double u_dir[3];
        for (int k = 0; k < 3; ++k) u_dir[k] = saved[k] - obj->vertices[va].pos[k];
        if (mftlParaboloidFitAt(obj, v_b, nbr, nvert, u_dir, w_dir) == MFTL_SUCCESS)
        {
          memcpy(fitted, obj->vertices[v_b].pos, 3 * sizeof(double));
          have_fit = 1;
        }
      }
      if (!have_fit && mftlBarycentricFit(obj, v_b, he_ab_idx, fitted) == MFTL_SUCCESS) have_fit = 1;
      memcpy(obj->vertices[v_b].pos, saved, 3 * sizeof(double));  // collapse recomputes the midpoint
    }
  }

  MFTLCollapseEdge(obj, he_ab_idx);
  if (have_fit) memcpy(obj->vertices[v_b].pos, fitted, 3 * sizeof(double));
}
