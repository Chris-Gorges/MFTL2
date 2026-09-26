/*
 * Per-vertex normals, curvatures (discrete cotangent Laplace–Beltrami) and
 * surface roughness, reproducing MFTL1's conventions exactly
 * (plans/06-normals-curvature-roughness.md):
 *
 *   - Normals        : angle/edge-length-weighted average of incident
 *                      triangle normals, w = sin(alpha)/(L0*L1)
 *                      (vertexfunctions.c:MFTL_VertexNormal).
 *   - Curvatures     : cotangent Laplace–Beltrami with mixed-Voronoi area
 *                      (oneringcomputations.c:MFTL_VertexLaplaceBeltrami):
 *                      per corner edge cotan = 0.25*(La^2+Lb^2-Lc^2)/A,
 *                      Voronoi area *= 0.125, obtuse triangles contribute
 *                      A/2 (at the obtuse vertex) or A/4 (otherwise),
 *                      mean = |0.5*sum(cotan*(V-Vj))/A_mix|  (= 2/R sphere),
 *                      gauss = 2*(2*pi - sum angles)/A_mix    (= 2/R^2).
 *                      THE CONVENTION IS LOAD-BEARING: Stage 08's adaptive
 *                      edge length L = sqrt(6*eps/kappa - 3*eps^2) assumes
 *                      mean = 2/R, gaussian = 2/R^2.
 *   - Roughness      : geometric Laplacian |V - sum(Vj/d)/sum(1/d)| (global
 *                      average returned, MFTL1 smoothing.c parity) and
 *                      dihedral-angle 1 - N_T0.N_T1 with per-vertex mean and
 *                      sample variance stored in roughness_mean/_variance.
 *
 * Parallel semantics: all quantities are one-ring sums. The caller must
 * guarantee a COMPLETE one-ring of triangles around every owned vertex —
 * after redistribution: MFTLUpdateGhostPositions(ens) + MFTLExtendGhostLayer
 * (ens, >= 1) — and must have current triangle properties
 * (MFTLComputeTriangleProperties). Computed values on ghost copies are then
 * refreshed with MFTLUpdateGhostPositions (its payload already transports
 * normal, curvatures and area), which mirrors MFTL1's two-phase exchange.
 * Vertices whose stored ring is cut by the patch boundary are skipped.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_memory.h"

#define MFTL_CURVATURE_MAX_RING 64  // safety cap for per-vertex ring buffers

// Guard: is this vertex local (owned by this rank, not a ghost)?
static inline int mftlCurvIsLocal(const MFTL_Vertex* v, int rank)
{
  return !v->is_ghost && v->owner_proc == rank && !(v->flags & MFTL_FLAG_DEAD);
}

// The stored 1-ring around v closes (v is interior to the STORED mesh,
// i.e. every ring edge has a twin) — required for correct one-ring sums.
static int mftlRingIsClosed(const MFTL_Object* obj, const MFTL_Vertex* v)
{
  if (v->out_he_idx < 0 || v->out_he_idx >= obj->n_half_edges) return 0;
  const MFTL_HalfEdge* start = &obj->half_edges[v->out_he_idx];
  const MFTL_HalfEdge* curr = start;
  int guard = 0;
  do
  {
    if (curr->twin_idx == MFTL_ERROR) return 0;
    const MFTL_HalfEdge* twin = &obj->half_edges[curr->twin_idx];
    curr = &obj->half_edges[twin->next_idx];
    if (++guard > obj->n_half_edges) return 0;  // cycle safety
  } while (curr != start);
  return 1;
}

// Gather the closed ring of triangle indices around local vertex vi.
// Returns the count, or 0 if the ring is not closed in the stored mesh.
static int mftlGatherRing(const MFTL_Object* obj, int vi, int* ring_tris)
{
  const MFTL_Vertex* v = &obj->vertices[vi];
  if (!mftlRingIsClosed(obj, v)) return 0;

  int count = 0;
  int he = v->out_he_idx;
  do
  {
    // remeshing leaves DEAD entities in the arrays until finalization:
    // a ring crossing one is incomplete and the vertex is skipped
    if (obj->half_edges[he].flags & MFTL_FLAG_DEAD) return 0;
    MFTL_ASSERT(count < MFTL_CURVATURE_MAX_RING && "valence exceeds ring buffer cap");
    if (count >= MFTL_CURVATURE_MAX_RING) return 0;
    ring_tris[count++] = obj->half_edges[he].triangle_idx;
    int twin = obj->half_edges[he].twin_idx;
    if (obj->half_edges[twin].flags & MFTL_FLAG_DEAD) return 0;
    he = obj->half_edges[twin].next_idx;
  } while (he != v->out_he_idx);
  return count;
}

// Corner indices (into a triangle's flattened vertex list) around vertex vi.
// Returns vi's corner k (0..2) and fills cyc[3] with the corner order.
static int mftlTriangleCorners(const MFTL_Object* obj, const MFTL_Triangle* tri, int vi, int cyc[3], double corner_pos[3][3])
{
  int he0 = tri->he_idx;
  int he1 = obj->half_edges[he0].next_idx;
  int he2 = obj->half_edges[he1].next_idx;
  int vidx[3] = {obj->half_edges[he2].vertex_idx, obj->half_edges[he0].vertex_idx, obj->half_edges[he1].vertex_idx};

  int kv = -1;
  for (int k = 0; k < 3; ++k)
  {
    cyc[k] = vidx[k];
    memcpy(corner_pos[k], obj->vertices[vidx[k]].pos, 3 * sizeof(double));
    if (vidx[k] == vi) kv = k;
  }
  // Seam-straddling triangles are "stretched" in raw (wrapped) storage:
  // re-image every corner into vi's periodic image so all angle/length
  // differences below are seam-safe (exact copy when non-periodic).
  if (MFTLDomainAnyPeriodic(obj->domain))
  {
    const double* anchor = obj->vertices[vi].pos;
    for (int k = 0; k < 3; ++k) MFTLUnwrapToAnchor(obj->domain, corner_pos[k], corner_pos[k], anchor);
  }
  return kv;
}

// Angle/edge-length weighted normal of vertex i from its (closed) stored ring. Returns the
// unnormalised accumulator in acc; when recompute_tri is set, every ring triangle's cached
// normal is recomputed from the current positions first (MFTL1 MFTL_OneRingVertexNormal).
static void mftlVertexNormalAccumulate(MFTL_Object* obj, int i, const int* ring, int n, int recompute_tri, double acc[3])
{
  acc[0] = acc[1] = acc[2] = 0.0;
  for (int r = 0; r < n; ++r)
  {
    MFTL_Triangle* tri = &obj->triangles[ring[r]];
    if (recompute_tri) MFTLGetTriangleProperties(obj, tri, MFTL_NULL, tri->normal, MFTL_NULL);
    int cyc[3];
    double cp[3][3];
    int kv = mftlTriangleCorners(obj, tri, i, cyc, cp);
    MFTL_ASSERT(kv >= 0);

    // corner angle at v toward the other two corners
    double e0[3] = {cp[(kv + 1) % 3][0] - cp[kv][0], cp[(kv + 1) % 3][1] - cp[kv][1], cp[(kv + 1) % 3][2] - cp[kv][2]};
    double e1[3] = {cp[(kv + 2) % 3][0] - cp[kv][0], cp[(kv + 2) % 3][1] - cp[kv][1], cp[(kv + 2) % 3][2] - cp[kv][2]};
    double l0 = sqrt(e0[0] * e0[0] + e0[1] * e0[1] + e0[2] * e0[2]);
    double l1 = sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]);
    double lprod = l0 * l1;
    if (!(lprod > 0.0)) continue;  // degenerate corner (scale-free guard, Stage 15)

    double c = (e0[0] * e1[0] + e0[1] * e1[1] + e0[2] * e1[2]) / lprod;
    if (c > 1.0) c = 1.0;
    if (c < -1.0) c = -1.0;
    double alpha = acos(c);
    double weight = sin(alpha) / lprod;

    acc[0] += weight * tri->normal[0];
    acc[1] += weight * tri->normal[1];
    acc[2] += weight * tri->normal[2];
  }
}

/* Stage 16C: triangles of an OPEN_COMPLETE fan (boundary vertex); 0 otherwise */
static int mftlGatherOpenFan(const MFTL_Object* obj, int vi, int* ring_tris)
{
  int hes[MFTL_CURVATURE_MAX_RING], n = 0;
  if (MFTLVertexRing(obj, vi, hes, MFTL_CURVATURE_MAX_RING, &n) != MFTL_RING_OPEN_COMPLETE || n >= MFTL_CURVATURE_MAX_RING) return 0;
  for (int k = 0; k < n; ++k)
  {
    if (obj->half_edges[hes[k]].flags & MFTL_FLAG_DEAD) return 0;
    ring_tris[k] = obj->half_edges[hes[k]].triangle_idx;
  }
  return n;
}

void MFTLComputeVertexNormals(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  int rank = obj->domain->rank;
  int ring[MFTL_CURVATURE_MAX_RING];

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlCurvIsLocal(v, rank)) continue;

    int n = mftlGatherRing(obj, i, ring);
    if (n == 0)
    {
      // Stage 16C (plans/16 §4.2): a boundary vertex (OPEN_COMPLETE fan) gets the angle-weighted
      // normal of its available fan; an incomplete fan is skipped (ghost-covered owned vertices
      // never have one after G2)
      n = mftlGatherOpenFan(obj, i, ring);
      if (n == 0) continue;
    }

    double acc[3];
    mftlVertexNormalAccumulate(obj, i, ring, n, 0, acc);

    double len = sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
    if (len > MFTL_VERYSMALL)
    {
      v->normal[0] = acc[0] / len;
      v->normal[1] = acc[1] / len;
      v->normal[2] = acc[2] / len;
    }
  }
}

int MFTLComputeVertexNormalAt(MFTL_Object* obj, int vi)
{
  int ring[MFTL_CURVATURE_MAX_RING];
  int n = mftlGatherRing(obj, vi, ring);
  if (n == 0) return MFTL_ERROR;
  double acc[3];
  mftlVertexNormalAccumulate(obj, vi, ring, n, 1, acc);
  // MFTL1 MFTL_NormalizeVector: floor on the magnitude, always normalised
  double len = sqrt(acc[0] * acc[0] + acc[1] * acc[1] + acc[2] * acc[2]);
  if (len < MFTL_VERYVERYSMALL) len = MFTL_VERYVERYSMALL;
  MFTL_Vertex* v = &obj->vertices[vi];
  v->normal[0] = acc[0] / len;
  v->normal[1] = acc[1] / len;
  v->normal[2] = acc[2] / len;
  return MFTL_SUCCESS;
}

// Laplace-Beltrami curvatures of vertex i from its (closed) stored ring; recompute_area
// refreshes every ring triangle's cached area first (MFTL1 MFTL_VertexLaplaceBeltrami).
static int mftlVertexCurvatureKernel(MFTL_Object* obj, int i, const int* ring, int n, int recompute_area)
{
  MFTL_Vertex* v = &obj->vertices[i];
  if (recompute_area)
    for (int r = 0; r < n; ++r) MFTLGetTriangleProperties(obj, &obj->triangles[ring[r]], &obj->triangles[ring[r]].area, MFTL_NULL, MFTL_NULL);
  double area_voronoi = 0.0;
  double gauss_angle_sum = 0.0;
  double mean_sum[3] = {0.0, 0.0, 0.0};

  for (int r = 0; r < n; ++r)
  {
    MFTL_Triangle* tri = &obj->triangles[ring[r]];
    int cyc[3];
    double cp[3][3];
    int kv = mftlTriangleCorners(obj, tri, i, cyc, cp);
    MFTL_ASSERT(kv >= 0);

    // squared edge lengths: L2[k] = edge opposite corner k
    double L2[3];
    for (int k = 0; k < 3; ++k)
    {
      double d0 = cp[(k + 1) % 3][0] - cp[(k + 2) % 3][0];
      double d1 = cp[(k + 1) % 3][1] - cp[(k + 2) % 3][1];
      double d2 = cp[(k + 1) % 3][2] - cp[(k + 2) % 3][2];
      L2[k] = d0 * d0 + d1 * d1 + d2 * d2;
    }

    double A = tri->area;
    if (!(A > 0.0)) continue;  // degenerate triangle guard (scale-free, Stage 15)

    // corner angle at v (opposite corner kv's edge)
    double f0[3] = {cp[(kv + 1) % 3][0] - cp[kv][0], cp[(kv + 1) % 3][1] - cp[kv][1], cp[(kv + 1) % 3][2] - cp[kv][2]};
    double f1[3] = {cp[(kv + 2) % 3][0] - cp[kv][0], cp[(kv + 2) % 3][1] - cp[kv][1], cp[(kv + 2) % 3][2] - cp[kv][2]};
    double lf0 = sqrt(f0[0] * f0[0] + f0[1] * f0[1] + f0[2] * f0[2]);
    double lf1 = sqrt(f1[0] * f1[0] + f1[1] * f1[1] + f1[2] * f1[2]);
    double cosv = (lf0 * lf1 > 0.0) ? (f0[0] * f1[0] + f0[1] * f1[1] + f0[2] * f1[2]) / (lf0 * lf1) : 1.0;
    if (cosv > 1.0) cosv = 1.0;
    if (cosv < -1.0) cosv = -1.0;
    double angle_at_v = acos(cosv);
    gauss_angle_sum += angle_at_v;

    // obtuse test at every corner: angle k obtuse <=> L2[k] > L2[k+1] + L2[k+2]
    int obtuse = (L2[0] > L2[1] + L2[2]) || (L2[1] > L2[2] + L2[0]) || (L2[2] > L2[0] + L2[1]);

    // the two edges CONTAINING v are opposite corners (kv+1) and (kv+2)
    for (int cc = 1; cc <= 2; ++cc)
    {
      int eo = (kv + cc) % 3;  // edge opposite corner eo contains v
      double cotan = 0.25 * (L2[(eo + 1) % 3] + L2[(eo + 2) % 3] - L2[eo]) / A;
      if (!obtuse) area_voronoi += cotan * L2[eo];

      // the other endpoint of that edge: use the unwrapped corner
      // positions (cp[kv] == v's image; seam-safe across the periodic wrap)
      const double* op = cp[(kv + 3 - cc) % 3];
      mean_sum[0] += cotan * (cp[kv][0] - op[0]);
      mean_sum[1] += cotan * (cp[kv][1] - op[1]);
      mean_sum[2] += cotan * (cp[kv][2] - op[2]);
    }

    if (obtuse) area_voronoi += (angle_at_v > 1.5708) ? 0.5 * A : 0.25 * A;
  }

  area_voronoi *= 0.125;
  if (!(area_voronoi > 0.0)) return MFTL_ERROR;  // scale-free guard (Stage 15)

  double lb0 = 0.5 * mean_sum[0] / area_voronoi;
  double lb1 = 0.5 * mean_sum[1] / area_voronoi;
  double lb2 = 0.5 * mean_sum[2] / area_voronoi;

  v->area = area_voronoi;
  v->meancurvature = sqrt(lb0 * lb0 + lb1 * lb1 + lb2 * lb2);
  v->gaussiancurvature = 2.0 * (MFTL_TWO_PI - gauss_angle_sum) / area_voronoi;
  return MFTL_SUCCESS;
}

void MFTLComputeVertexCurvatures(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  int rank = obj->domain->rank;
  int ring[MFTL_CURVATURE_MAX_RING];

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlCurvIsLocal(v, rank)) continue;

    int n = mftlGatherRing(obj, i, ring);
    if (n == 0)
    {
      // Stage 16C (plans/16 §4.2): no silent skip — curvature is defined only on closed fans;
      // boundary (and, before G2, incomplete) vertices are flagged
      v->flags |= MFTL_VFLAG_CURV_UNDEFINED;
      continue;
    }
    v->flags &= ~MFTL_VFLAG_CURV_UNDEFINED;

    mftlVertexCurvatureKernel(obj, i, ring, n, 0);
  }
}

int MFTLComputeVertexCurvatureAt(MFTL_Object* obj, int vi)
{
  int ring[MFTL_CURVATURE_MAX_RING];
  int n = mftlGatherRing(obj, vi, ring);
  if (n == 0)
  {
    obj->vertices[vi].flags |= MFTL_VFLAG_CURV_UNDEFINED;  // Stage 16C
    return MFTL_ERROR;
  }
  obj->vertices[vi].flags &= ~MFTL_VFLAG_CURV_UNDEFINED;
  return mftlVertexCurvatureKernel(obj, vi, ring, n, 1);
}

double MFTLComputeRoughness(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  int rank = obj->domain->rank;
  int ring[MFTL_CURVATURE_MAX_RING];

  double gl_sum = 0.0;
  int gl_count = 0;

  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlCurvIsLocal(v, rank)) continue;

    int n = mftlGatherRing(obj, i, ring);
    if (n == 0) continue;

    // --- distinct one-ring neighbor vertices ---
    int nbr[MFTL_CURVATURE_MAX_RING];
    int nn = 0;
    for (int r = 0; r < n; ++r)
    {
      const MFTL_Triangle* tri = &obj->triangles[ring[r]];
      int he0 = tri->he_idx;
      int he1 = obj->half_edges[he0].next_idx;
      int he2 = obj->half_edges[he1].next_idx;
      int vidx[3] = {obj->half_edges[he2].vertex_idx, obj->half_edges[he0].vertex_idx, obj->half_edges[he1].vertex_idx};
      for (int k = 0; k < 3; ++k)
      {
        if (vidx[k] == i) continue;
        int found = 0;
        for (int m = 0; m < nn; ++m)
          if (nbr[m] == vidx[k])
          {
            found = 1;
            break;
          }
        if (!found && nn < MFTL_CURVATURE_MAX_RING) nbr[nn++] = vidx[k];
      }
    }

    // --- geometric Laplacian: |V - sum(Vj/d)/sum(1/d)| ---
    double inv_sum = 0.0;
    double c[3] = {0.0, 0.0, 0.0};
    const int any_periodic = MFTLDomainAnyPeriodic(obj->domain);
    for (int m = 0; m < nn; ++m)
    {
      // re-image the neighbour into v's periodic image (exact copy when
      // non-periodic), otherwise the average and distance cross the seam
      double ou[3];
      const double* op = obj->vertices[nbr[m]].pos;
      if (any_periodic)
      {
        MFTLUnwrapToAnchor(obj->domain, ou, op, v->pos);
        op = ou;
      }
      double dx = v->pos[0] - op[0];
      double dy = v->pos[1] - op[1];
      double dz = v->pos[2] - op[2];
      double d = sqrt(dx * dx + dy * dy + dz * dz);
      if (d < MFTL_VERYSMALL) continue;
      double w = 1.0 / d;
      inv_sum += w;
      c[0] += w * op[0];
      c[1] += w * op[1];
      c[2] += w * op[2];
    }
    if (inv_sum > MFTL_VERYSMALL)
    {
      c[0] /= inv_sum;
      c[1] /= inv_sum;
      c[2] /= inv_sum;
      double gx = v->pos[0] - c[0];
      double gy = v->pos[1] - c[1];
      double gz = v->pos[2] - c[2];
      gl_sum += sqrt(gx * gx + gy * gy + gz * gz);
      gl_count++;
    }

    // --- dihedral roughness: 1 - N_T0.N_T1 over the ring edges from v ---
    // (dynamic valence cap; MFTL1 hard-coded 10, MFTL2 allows up to
    // MFTL_CURVATURE_MAX_RING via the ring buffer)
    double dihedral[MFTL_CURVATURE_MAX_RING * 2];
    int nd = 0;
    int he = v->out_he_idx;
    do
    {
      MFTL_ASSERT(nd < MFTL_CURVATURE_MAX_RING * 2);
      const MFTL_HalfEdge* h = &obj->half_edges[he];
      const MFTL_HalfEdge* twin = &obj->half_edges[h->twin_idx];  // ring closed: twin exists
      const MFTL_Triangle* t0 = &obj->triangles[h->triangle_idx];
      const MFTL_Triangle* t1 = &obj->triangles[twin->triangle_idx];
      dihedral[nd++] = 1.0 - (t0->normal[0] * t1->normal[0] + t0->normal[1] * t1->normal[1] + t0->normal[2] * t1->normal[2]);
      he = twin->next_idx;
    } while (he != v->out_he_idx && nd < MFTL_CURVATURE_MAX_RING * 2);

    if (nd >= 2)
    {
      double mean = 0.0;
      for (int k = 0; k < nd; ++k) mean += dihedral[k];
      mean /= (double) nd;
      double var = 0.0;
      for (int k = 0; k < nd; ++k) var += (dihedral[k] - mean) * (dihedral[k] - mean);
      var /= (double) (nd - 1);
      v->roughness_mean = mean;
      v->roughness_variance = var;
    }
  }

  double gl_global = 0.0;
  int gl_count_global = 0;
  MPI_Allreduce(&gl_sum, &gl_global, 1, MPI_DOUBLE, MPI_SUM, obj->domain->comm_cart);
  MPI_Allreduce(&gl_count, &gl_count_global, 1, MPI_INT, MPI_SUM, obj->domain->comm_cart);
  return gl_count_global > 0 ? gl_global / (double) gl_count_global : 0.0;
}

/* ------------------------------------------------------------------------------------------
 * Stage 16C (plans/16 §4.2): extrapolated curvature of boundary vertices
 * ------------------------------------------------------------------------------------------ */
static int mftlSolve5(double A[5][5], double b[5], double x[5])
{
  for (int c = 0; c < 5; ++c)
  {
    int p = c;
    for (int r = c + 1; r < 5; ++r)
      if (fabs(A[r][c]) > fabs(A[p][c])) p = r;
    if (!(fabs(A[p][c]) > 0.0)) return MFTL_ERROR;
    if (p != c)
    {
      for (int k = 0; k < 5; ++k)
      {
        double t = A[c][k];
        A[c][k] = A[p][k];
        A[p][k] = t;
      }
      double t = b[c];
      b[c] = b[p];
      b[p] = t;
    }
    for (int r = c + 1; r < 5; ++r)
    {
      const double f = A[r][c] / A[c][c];
      for (int k = c; k < 5; ++k) A[r][k] -= f * A[c][k];
      b[r] -= f * b[c];
    }
  }
  for (int r = 4; r >= 0; --r)
  {
    double s = b[r];
    for (int k = r + 1; k < 5; ++k) s -= A[r][k] * x[k];
    x[r] = s / A[r][r];
  }
  return MFTL_SUCCESS;
}

void MFTLComputeExtrapolatedCurvatures(MFTL_Object* obj)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  const int rank = obj->domain->rank;
  int ring[MFTL_CURVATURE_MAX_RING];
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (!mftlCurvIsLocal(v, rank)) continue;
    v->flags &= ~MFTL_VFLAG_CURV_EXTRAPOLATED;
    const int n = mftlGatherOpenFan(obj, i, ring);
    if (n == 0) continue;
    // frame: the open-fan normal
    double w[3];
    mftlVertexNormalAccumulate(obj, i, ring, n, 0, w);
    double wl = sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
    if (!(wl > 0.0)) continue;
    for (int k = 0; k < 3; ++k) w[k] /= wl;
    double u[3] = {1, 0, 0};
    if (fabs(w[0]) > 0.9) u[0] = 0, u[1] = 1;
    double dd = u[0] * w[0] + u[1] * w[1] + u[2] * w[2];
    for (int k = 0; k < 3; ++k) u[k] -= dd * w[k];
    double ul = sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
    for (int k = 0; k < 3; ++k) u[k] /= ul;
    double s[3] = {w[1] * u[2] - w[2] * u[1], w[2] * u[0] - w[0] * u[2], w[0] * u[1] - w[1] * u[0]};
    // stencil: the vertices of the fan triangles and of the stored fans of the fan vertices
    int st[4 * MFTL_CURVATURE_MAX_RING], ns = 0;
    for (int r = 0; r < n; ++r)
    {
      MFTL_Vertex* tv[3];
      MFTLGetTriangleVertices(obj, &obj->triangles[ring[r]], tv);
      for (int c = 0; c < 3; ++c)
      {
        const int x = (int) (tv[c] - obj->vertices);
        if (x == i) continue;
        MFTL_Vertex* nb[64];
        const int nn = MFTLGetVertexOneRingVertices(obj, tv[c], nb, 64, MFTL_FALSE);
        for (int q = -1; q < nn; ++q)
        {
          const int y = (q < 0) ? x : (int) (nb[q] - obj->vertices);
          if (y == i) continue;
          int dup = 0;
          for (int z = 0; z < ns; ++z)
            if (st[z] == y) dup = 1;
          if (!dup && ns < 4 * MFTL_CURVATURE_MAX_RING) st[ns++] = y;
        }
      }
    }
    if (ns < 6) continue;
    // least squares h = a x^2 + b x y + c y^2 + d x + e y
    double A[5][5] = {{0}}, B[5] = {0}, X[5];
    for (int q = 0; q < ns; ++q)
    {
      double dv[3];
      MFTLGetMinimumImageVector(obj->domain, dv, v->pos, obj->vertices[st[q]].pos);
      const double x = dv[0] * u[0] + dv[1] * u[1] + dv[2] * u[2];
      const double y = dv[0] * s[0] + dv[1] * s[1] + dv[2] * s[2];
      const double h = dv[0] * w[0] + dv[1] * w[1] + dv[2] * w[2];
      const double f[5] = {x * x, x * y, y * y, x, y};
      for (int r = 0; r < 5; ++r)
      {
        for (int c = 0; c < 5; ++c) A[r][c] += f[r] * f[c];
        B[r] += f[r] * h;
      }
    }
    if (mftlSolve5(A, B, X) != MFTL_SUCCESS) continue;
    const double a = X[0], b = X[1], c = X[2], d = X[3], e = X[4];
    const double g = 1.0 + d * d + e * e;
    const double twoH = ((1.0 + e * e) * 2.0 * a - 2.0 * b * d * e + (1.0 + d * d) * 2.0 * c) / pow(g, 1.5);
    v->curv_extrap = fabs(twoH);  // same convention as meancurvature (|2H|)
    v->flags |= MFTL_VFLAG_CURV_EXTRAPOLATED;
  }
}
