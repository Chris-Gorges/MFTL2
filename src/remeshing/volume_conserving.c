/*
 * Volume-conserving smoothing (Stage 09), MFTL1 parity:
 *
 *   - MFTLVolumeRecovery (volumeconservation.c:VolumeRecovery): global
 *     uniform normal shift h = (V - target)/Area, iterated to a tight
 *     tolerance. Restores GLOBAL volume exactly (to first order per shift).
 *   - MFTLEdgeVolumeConservingSmoothing / MFTLVolumeConservingSmoothing
 *     (volumeconservation.c, Kuprat 2001 Algorithm 4): per-edge coupled
 *     umbrella targets x_s, under-relaxed moves dx, and a common normal
 *     correction h = -(dx0.A0 + dx1.A1 + dx1.(v x dx0)) / |A0 + A1 + v x (dx0-dx1)|
 *     cancelling the first-order volume change of the pair move.
 *   - MFTLEdgeTSUR / MFTLTSUR3D (tsur3d.c, Jiao JCP 2004): project the
 *     endpoints to the common tangent plane n = n(n0+n1), then solve the
 *     volume-conserving height h = (V0+V1)/(V0/h0 + V1/h1).
 *   - MFTLRoughnessSmoothing (smoothing.c:Object_Roughness_Smoothing):
 *     per-edge trigger on dihedral roughness (Stage-06 roughness_mean) and
 *     geometric-Laplacian magnitude, dispatch to Kuprat or TSUR.
 *
 * Conventions and guards:
 *   - Feature protection (Stage 08): edges flagged MFTL_FLAG_FEATURE or
 *     geometrically sharp (dihedral > feature_angle_threshold) are never
 *     smoothed; vertices are only moved by edge steps on non-feature edges.
 *   - Dead/ghost entities are skipped; ring walks that cross them abort the
 *     step (the vertex pair is left untouched).
 *   - TSUR NaN guards: degenerate combined normal, h0/h1 ~ 0 (division
 *     inf), non-finite h -> step aborted with MFTL_ERROR.
 *   - Kuprat guard: |AA| near zero (flat/degenerate ring) -> step skipped.
 *
 * Parallel: PRECONDITION — a complete one-ring around every processed vertex
 * must be stored. The serial workflow (rank-0-local mesh, or ghosts via
 * MFTLExtendGhostLayer + MFTLUpdateGhostPositions) satisfies this; a fully
 * parallel edge-local version is Stage 10 work. Volume reductions are
 * collective (Stage 02 integrals over comm_cart). After smoothing owned
 * vertices, refresh ghost copies with MFTLUpdateGhostPositions.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_images.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_remeshing.h"

#define MFTL_VC_MAX_RING 64

/* ------------------------------------------------------------------------- */
/* Stage 10: serial-only precondition + parallel-safe drivers                */
/* ------------------------------------------------------------------------- */

// The serial edge-local drivers below move BOTH endpoints of every edge they
// visit and read their full one-rings. On a distributed mesh that would
// silently move ghost vertices and read truncated rings — the Stage-09 review
// carry-forward. Fail loudly (Debug AND Release) instead; distributed callers
// must use the parallel remesh driver (MFTLRemeshEnsembleParallel), which
// gates every edge to the MFTL1 parallel-interior rule via the *Parallel
// variants below.
static void mftlVcRequireCompleteMesh(const MFTL_Object* obj, const char* what)
{
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    if (obj->vertices[i].is_ghost)
    {
      fprintf(stderr,
              "FATAL ERROR: %s called on a distributed mesh (object %d stores %d vertices, at least one ghost).\n"
              "The serial edge-local smoothers must not run on a distributed mesh: use MFTLRemeshEnsembleParallel,\n"
              "or the MFTLVolumeConservingSmoothingParallel / MFTLTSUR3DParallel / MFTLRoughnessSmoothingParallel\n"
              "variants, which gate every edge to owned, ghost-free interiors.\n",
              what, obj->global_id, obj->n_vertices);
      exit(EXIT_FAILURE);
    }
  }
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    if (obj->triangles[t].is_ghost)
    {
      fprintf(stderr,
              "FATAL ERROR: %s called on a distributed mesh (object %d stores %d triangles, at least one ghost).\n"
              "Use MFTLRemeshEnsembleParallel (or the *Parallel smoother variants) on distributed meshes.\n",
              what, obj->global_id, obj->n_triangles);
      exit(EXIT_FAILURE);
    }
  }
}

// MFTL1 parallel-interior gate for edge-local smoothers (tsur3d.c:148,182):
// both endpoints owned, both adjacent triangles local and ghost-free. Rings
// around owned vertices are complete by the ghost-layer invariant, so the
// per-edge steps remain exactly local; ghost copies are refreshed by the
// driver's MFTLUpdateGhostPositions.
static int mftlVcEdgeInteriorParallel(const MFTL_Object* obj, int he_idx)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return 0;
  if (he->twin_idx == MFTL_ERROR) return 0;
  const MFTL_HalfEdge* tw = &obj->half_edges[he->twin_idx];
  if (tw->flags & MFTL_FLAG_DEAD) return 0;

  if (obj->triangles[he->triangle_idx].is_ghost) return 0;
  if (obj->triangles[tw->triangle_idx].is_ghost) return 0;

  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  if (obj->vertices[he->vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[he_next].vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[he_prev].vertex_idx].is_ghost) return 0;

  int tw_next = tw->next_idx;
  int tw_prev = obj->half_edges[tw_next].next_idx;
  if (obj->vertices[tw->vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[tw_next].vertex_idx].is_ghost) return 0;
  if (obj->vertices[obj->half_edges[tw_prev].vertex_idx].is_ghost) return 0;

  // Stage 15 (L7): the step moves both endpoints and rescales the surface scalar of every
  // triangle in their one-rings; a ring triangle stored here only as a ghost would get its
  // rescale on the ghost copy while its owner keeps the old value (mass loss). Require the
  // full rings to be locally owned.
  for (int e = 0; e < 2; ++e)
  {
    const MFTL_Vertex* v = &obj->vertices[e == 0 ? he->vertex_idx : tw->vertex_idx];
    int h = v->out_he_idx, guard = 0;
    if (h < 0) return 0;
    do
    {
      if (obj->half_edges[h].flags & MFTL_FLAG_DEAD) return 0;
      if (obj->triangles[obj->half_edges[h].triangle_idx].is_ghost) return 0;
      int t = obj->half_edges[h].twin_idx;
      if (t == MFTL_ERROR) return 0;
      h = obj->half_edges[t].next_idx;
      if (++guard > MFTL_VC_MAX_RING) return 0;
    } while (h != v->out_he_idx);
  }
  return 1;
}

/* ------------------------------------------------------------------------- */
/* Local helpers                                                             */
/* ------------------------------------------------------------------------- */

static void mftlVcTriNormal(const MFTL_Object* obj, int tri_idx, double n[3])
{
  MFTL_Vertex* vv[3];
  MFTLGetTriangleVertices(obj, &obj->triangles[tri_idx], vv);
  // minimum-image edge vectors: exact raw differences when non-periodic,
  // seam-safe otherwise (wrapped storage stretches straddling triangles)
  double e0[3], e1[3];
  MFTLGetMinimumImageVector(obj->domain, e0, vv[0]->pos, vv[1]->pos);
  MFTLGetMinimumImageVector(obj->domain, e1, vv[0]->pos, vv[2]->pos);
  n[0] = e0[1] * e1[2] - e0[2] * e1[1];
  n[1] = e0[2] * e1[0] - e0[0] * e1[2];
  n[2] = e0[0] * e1[1] - e0[1] * e1[0];
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len < MFTL_VERYVERYSMALL)
  {
    n[0] = n[1] = n[2] = 0.0;
    return;
  }
  n[0] /= len;
  n[1] /= len;
  n[2] /= len;
}

// Closed one-ring triangle walk; returns the count, 0 when the stored ring
// is incomplete (boundary/dead/ghost-laden).
static int mftlVcRingTris(const MFTL_Object* obj, int vi, int* ring_tris)
{
  const MFTL_Vertex* v = &obj->vertices[vi];
  if (v->flags & MFTL_FLAG_DEAD) return 0;
  if (v->out_he_idx < 0 || v->out_he_idx >= obj->n_half_edges) return 0;

  int count = 0;
  int he = v->out_he_idx;
  do
  {
    if (obj->half_edges[he].flags & MFTL_FLAG_DEAD) return 0;
    if (count >= MFTL_VC_MAX_RING) return 0;
    ring_tris[count++] = obj->half_edges[he].triangle_idx;
    int twin = obj->half_edges[he].twin_idx;
    if (twin == MFTL_ERROR || (obj->half_edges[twin].flags & MFTL_FLAG_DEAD)) return 0;
    he = obj->half_edges[twin].next_idx;
  } while (he != v->out_he_idx);
  return count;
}

// Feature test: flagged or geometrically sharp (Stage 08 protection).
static int mftlVcFeatureEdge(const MFTL_Object* obj, int he_idx, double angle_degrees)
{
  const MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_FEATURE) return 1;
  if (he->twin_idx == MFTL_ERROR) return 1;
  if (obj->half_edges[he->twin_idx].flags & MFTL_FLAG_FEATURE) return 1;
  double n1[3], n2[3];
  mftlVcTriNormal(obj, he->triangle_idx, n1);
  mftlVcTriNormal(obj, obj->half_edges[he->twin_idx].triangle_idx, n2);
  return (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2]) < MFTL_COS(angle_degrees * MFTL_PI / 180.0);
}

// A vertex incident to at least one sharp/feature edge is a feature vertex:
// volume smoothers never move it (Stage 08 feature protection).
static int mftlVcFeatureVertex(MFTL_Object* obj, int vi, double angle_degrees)
{
  const MFTL_Vertex* v = &obj->vertices[vi];
  if (v->flags & MFTL_FLAG_DEAD) return 1;
  if (v->out_he_idx < 0 || v->out_he_idx >= obj->n_half_edges) return 1;
  int he = v->out_he_idx;
  do
  {
    const MFTL_HalfEdge* curr = &obj->half_edges[he];
    if (curr->flags & MFTL_FLAG_DEAD) return 1;
    if (curr->twin_idx == MFTL_ERROR) return 1;  // boundary
    if (curr->flags & MFTL_FLAG_FEATURE) return 1;
    if (obj->half_edges[curr->twin_idx].flags & MFTL_FLAG_FEATURE) return 1;
    double n1[3], n2[3];
    mftlVcTriNormal(obj, curr->triangle_idx, n1);
    mftlVcTriNormal(obj, obj->half_edges[curr->twin_idx].triangle_idx, n2);
    if (n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] < MFTL_COS(angle_degrees * MFTL_PI / 180.0)) return 1;
    he = obj->half_edges[curr->twin_idx].next_idx;
  } while (he != v->out_he_idx);
  return 0;
}

// Fresh one-ring (Newell) unit normal of a vertex, from positions.
static int mftlVcRingNormal(MFTL_Object* obj, int vi, double n[3])
{
  n[0] = n[1] = n[2] = 0.0;
  int ring[MFTL_VC_MAX_RING];
  int cnt = mftlVcRingTris(obj, vi, ring);
  if (cnt == 0) return MFTL_ERROR;
  for (int i = 0; i < cnt; ++i)
  {
    double tn[3];
    mftlVcTriNormal(obj, ring[i], tn);
    n[0] += tn[0];
    n[1] += tn[1];
    n[2] += tn[2];
  }
  double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (len < MFTL_VERYVERYSMALL) return MFTL_ERROR;
  n[0] /= len;
  n[1] /= len;
  n[2] /= len;
  return MFTL_SUCCESS;
}

// Geometric-Laplacian magnitude (Stage-06 formula) of one vertex.
static double mftlVcLocalGL(MFTL_Object* obj, int vi)
{
  const MFTL_Vertex* v = &obj->vertices[vi];
  int ring[MFTL_VC_MAX_RING];
  int n = mftlVcRingTris(obj, vi, ring);
  if (n == 0) return 0.0;

  double inv_sum = 0.0, c[3] = {0, 0, 0};
  int seen[MFTL_VC_MAX_RING * 2];
  int nn = 0;
  for (int r = 0; r < n; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring[r]], vv);
    for (int k = 0; k < 3; ++k)
    {
      int wi = (int) (vv[k] - obj->vertices);
      if (wi == vi) continue;
      int found = 0;
      for (int m = 0; m < nn; ++m)
        if (seen[m] == wi)
        {
          found = 1;
          break;
        }
      if (!found && nn < MFTL_VC_MAX_RING * 2) seen[nn++] = wi;
    }
  }
  for (int m = 0; m < nn; ++m)
  {
    const double* op = obj->vertices[seen[m]].pos;
    double ou[3];
    if (MFTLDomainAnyPeriodic(obj->domain))
    {
      MFTLUnwrapToAnchor(obj->domain, ou, op, v->pos);  // seam-safe neighbour image
      op = ou;
    }
    double dx = v->pos[0] - op[0], dy = v->pos[1] - op[1], dz = v->pos[2] - op[2];
    double d = sqrt(dx * dx + dy * dy + dz * dz);
    if (d < MFTL_VERYVERYSMALL) continue;
    double w = 1.0 / d;
    inv_sum += w;
    c[0] += w * op[0];
    c[1] += w * op[1];
    c[2] += w * op[2];
  }
  if (inv_sum < MFTL_VERYVERYSMALL) return 0.0;
  c[0] /= inv_sum;
  c[1] /= inv_sum;
  c[2] /= inv_sum;
  double gx = v->pos[0] - c[0], gy = v->pos[1] - c[1], gz = v->pos[2] - c[2];
  return sqrt(gx * gx + gy * gy + gz * gz);
}

static double mftlVcDet3(const double m[3][3])
{
  return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
         m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
}

/* ------------------------------------------------------------------------- */
/* VolumeRecovery (MFTL1 VolumeRecovery)                                     */
/* ------------------------------------------------------------------------- */

double MFTLVolumeRecovery(MFTL_Object* obj, double target_volume)
{
  // Stage 16B class guard: open or wrapping bodies enclose no volume by themselves
  MFTLObjectEnsureWinding(obj);  // collective
  if (!MFTLObjectHasVolume(obj)) return 0.0;
  MFTL_ASSERT(obj != MFTL_NULL);

  // Effective first-order surface area for the normal shift. Each OWNED
  // triangle contributes (k/3)*A, where k = number of its corners owned by
  // this rank: a boundary triangle's area contribution scales with the
  // owned-corner fraction (the ghost corners follow via
  // MFTLUpdateGhostPositions). On serial meshes k == 3 everywhere and this
  // equals MFTLSurfaceCenter exactly. The shift h is a GLOBAL quantity
  // (every rank moves its owned vertices by the same h), so the area is
  // reduced with MPI_SUM and all ranks derive the identical h.
  const int vrank = obj->domain->rank;
  double effective_area = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != vrank || (tri->flags & MFTL_FLAG_DEAD)) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);
    int k = 0;
    for (int c = 0; c < 3; ++c)
      if (!vv[c]->is_ghost && vv[c]->owner_proc == vrank) k++;
    effective_area += ((double) k / 3.0) * tri->area;
  }
  if (obj->domain->nproc > 1)
  {
    double area_g = 0.0;
    MPI_Allreduce(&effective_area, &area_g, 1, MPI_DOUBLE, MPI_SUM, obj->domain->comm_cart);
    effective_area = area_g;
  }

  double rel_err = 0.0;

  // Secant calibration of the volume response. The model slope -A_eff (A_eff
  // = owned-corner-weighted area) is exact on serial meshes; on distributed
  // meshes frozen ghost corners and boundary-ring normals change the true
  // first-order response, so measure it after the first probe step and
  // Newton-correct with the measured slope afterwards.
  double slope = -effective_area;
  double volume_prev = 0.0;
  double shift_prev = 0.0;
  const int collective = obj->domain->nproc > 1;

  for (int iter = 0; iter < 10; ++iter)
  {
    double current_volume = MFTLObjectVolume(obj);  // collective

    double denom = fabs(target_volume) > MFTL_VERYVERYSMALL ? fabs(target_volume) : 1.0;
    rel_err = (current_volume - target_volume) / denom;

    // Consensus on EVERY loop exit (rank starvation can zero the local
    // area, and Allreduce results can differ in the last ulp between
    // ranks): a plain local break would let iteration counts diverge and
    // desynchronize the collectives inside this loop.
    int stop = (fabs(rel_err) < 1e-12);
    if (collective)
    {
      int stop_g = 0;
      MPI_Allreduce(&stop, &stop_g, 1, MPI_INT, MPI_MAX, obj->domain->comm_cart);
      stop = stop_g;
    }
    if (stop) break;

    // Measured secant slope from the previous step.
    if (iter > 0 && fabs(shift_prev) > MFTL_VERYVERYSMALL)
    {
      double measured = (current_volume - volume_prev) / shift_prev;
      if (fabs(measured) > MFTL_VERYVERYSMALL && measured * slope > 0.0) slope = measured;
    }

    stop = (fabs(slope) < MFTL_VERYVERYSMALL);
    if (collective)
    {
      int stop_g = 0;
      MPI_Allreduce(&stop, &stop_g, 1, MPI_INT, MPI_MAX, obj->domain->comm_cart);
      stop = stop_g;
    }
    if (stop) break;

    double h = -(current_volume - target_volume) / slope;
    volume_prev = current_volume;
    shift_prev = h;

    int rank = obj->domain->rank;
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost || v->owner_proc != rank || (v->flags & MFTL_FLAG_DEAD)) continue;
      double n[3];
      if (mftlVcRingNormal(obj, i, n) != MFTL_SUCCESS) continue;
      v->pos[0] -= n[0] * h;
      v->pos[1] -= n[1] * h;
      v->pos[2] -= n[2] * h;
      // storage invariant: positions stay wrapped inside the domain (no-op
      // when no axis is periodic); Stage 16B: the wrap is counted into img
      MFTLWrapPosImg(obj->domain, v->pos, v->img);
    }
  }

  // Final error (collective).
  double current_volume = MFTLObjectVolume(obj);
  double denom = fabs(target_volume) > MFTL_VERYVERYSMALL ? fabs(target_volume) : 1.0;
  return (current_volume - target_volume) / denom;
}

/* ------------------------------------------------------------------------- */
/* Kuprat Algorithm 4 (volumeconservation.c)                                 */
/* ------------------------------------------------------------------------- */

// Per-endpoint ring data: sum of ring-neighbour positions EXCLUDING the
// other edge endpoint, the ring triangle count, and the sign-aligned vector
// area of the ring. Positions are re-imaged into `anchor`'s periodic image
// (exact copies when non-periodic) so the sums are seam-safe; the caller
// anchors BOTH endpoints' rings at the same vertex for a common frame.
static int mftlVcEndpointData(MFTL_Object* obj, int vi, int v_other, const double anchor[3], double sum_x[3], double A[3], int* n_out)
{
  int ring[MFTL_VC_MAX_RING];
  int n = mftlVcRingTris(obj, vi, ring);
  if (n == 0) return MFTL_ERROR;

  const int any_periodic = MFTLDomainAnyPeriodic(obj->domain);

  sum_x[0] = sum_x[1] = sum_x[2] = 0.0;
  A[0] = A[1] = A[2] = 0.0;
  for (int r = 0; r < n; ++r)
  {
    MFTL_Triangle* tri = &obj->triangles[ring[r]];
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, tri, vv);

    // vector area of the triangle, oriented against the (position-based) normal
    double e0[3], e1[3];
    MFTLGetMinimumImageVector(obj->domain, e0, vv[0]->pos, vv[1]->pos);
    MFTLGetMinimumImageVector(obj->domain, e1, vv[0]->pos, vv[2]->pos);
    double S[3] = {e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]};
    double tn[3];
    mftlVcTriNormal(obj, ring[r], tn);
    if (S[0] * tn[0] + S[1] * tn[1] + S[2] * tn[2] < 0.0)
    {
      S[0] *= -1.0;
      S[1] *= -1.0;
      S[2] *= -1.0;
    }
    A[0] += S[0];
    A[1] += S[1];
    A[2] += S[2];

  }
  // ring neighbours, each ONCE, excluding the other edge endpoint (MFTL1
  // accumulates one corner per ring triangle via the exit edge — equivalent
  // to the deduplicated neighbour set).
  int seen[MFTL_VC_MAX_RING * 2];
  int nn = 0;
  for (int r = 0; r < n; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring[r]], vv);
    for (int k = 0; k < 3; ++k)
    {
      int wi = (int) (vv[k] - obj->vertices);
      if (wi == vi || wi == v_other) continue;
      int found = 0;
      for (int m = 0; m < nn; ++m)
        if (seen[m] == wi)
        {
          found = 1;
          break;
        }
      if (!found && nn < MFTL_VC_MAX_RING * 2)
      {
        seen[nn++] = wi;
        if (any_periodic)
        {
          double u[3];
          MFTLUnwrapToAnchor(obj->domain, u, vv[k]->pos, anchor);
          sum_x[0] += u[0];
          sum_x[1] += u[1];
          sum_x[2] += u[2];
        }
        else
        {
          sum_x[0] += vv[k]->pos[0];
          sum_x[1] += vv[k]->pos[1];
          sum_x[2] += vv[k]->pos[2];
        }
      }
    }
  }
  *n_out = n;
  return MFTL_SUCCESS;
}

// Stage 15 (L7, MFTL1 tsur3d.c / volumeconservation.c parity): after the library moved the
// two edge endpoints, every triangle of their one-rings keeps its surface-scalar MASS:
// c *= A_cached / A_new, and the cached area becomes A_new. A triangle in both rings is
// visited twice; the second visit has ratio exactly 1.
static void mftlSSRescaleRing(MFTL_Object* obj, const int* ring, int n)
{
  for (int r = 0; r < n; ++r)
  {
    MFTL_Triangle* t = &obj->triangles[ring[r]];
    double a = 0.0;
    MFTLGetTriangleProperties(obj, t, &a, MFTL_NULL, MFTL_NULL);
    if (a > 0.0) t->surface_scalar = t->surface_scalar * (t->area / a);
    t->area = a;
  }
}

void MFTLEdgeVolumeConservingSmoothing(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return;
  if (he->twin_idx == MFTL_ERROR) return;
  if (mftlVcFeatureEdge(obj, he_idx, params->feature_angle_threshold)) return;

  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v0 = obj->half_edges[he_prev].vertex_idx;
  int v1 = he->vertex_idx;
  if (mftlVcFeatureVertex(obj, v0, params->feature_angle_threshold)) return;
  if (mftlVcFeatureVertex(obj, v1, params->feature_angle_threshold)) return;

  // opposite corners of the two adjacent triangles (minimum-image edge
  // vector: raw difference when non-periodic)
  int vt1 = obj->half_edges[he_next].vertex_idx;
  int vt2 = obj->half_edges[obj->half_edges[he->twin_idx].next_idx].vertex_idx;
  double v[3];
  MFTLGetMinimumImageVector(obj->domain, v, obj->vertices[vt1].pos, obj->vertices[vt2].pos);

  // Both endpoint rings are summed in ONE frame anchored at v0's image, so
  // the coupled umbrella targets below mix compatible coordinates.
  const double* anchor = obj->vertices[v0].pos;
  double v1_img[3];
  MFTLUnwrapToAnchor(obj->domain, v1_img, obj->vertices[v1].pos, anchor);

  double sum_x[2][3], A[2][3];
  int n[2];
  if (mftlVcEndpointData(obj, v0, v1, anchor, sum_x[0], A[0], &n[0]) != MFTL_SUCCESS) return;
  if (mftlVcEndpointData(obj, v1, v0, anchor, sum_x[1], A[1], &n[1]) != MFTL_SUCCESS) return;

  // coupled umbrella targets (Kuprat paper notation)
  double x_s[2][3];
  double coeff = 1.0 / (((double) n[0]) * ((double) n[1]) - 1.0);
  for (int k = 0; k < 3; ++k) x_s[0][k] = coeff * (sum_x[1][k] + ((double) n[1]) * sum_x[0][k]);
  coeff = 1.0 / ((double) n[1]);
  for (int k = 0; k < 3; ++k) x_s[1][k] = coeff * (x_s[0][k] + sum_x[1][k]);

  double omega = params->laplacian_smoothing_under_relaxation;
  double dx[2][3];
  for (int k = 0; k < 3; ++k)
  {
    dx[0][k] = omega * (x_s[0][k] - obj->vertices[v0].pos[k]);
    dx[1][k] = omega * (x_s[1][k] - v1_img[k]);
  }

  double dx_subtract[3] = {dx[0][0] - dx[1][0], dx[0][1] - dx[1][1], dx[0][2] - dx[1][2]};
  double v_cross_dx_subtract[3] = {v[1] * dx_subtract[2] - v[2] * dx_subtract[1], v[2] * dx_subtract[0] - v[0] * dx_subtract[2],
                                   v[0] * dx_subtract[1] - v[1] * dx_subtract[0]};
  double AA[3] = {A[0][0] + A[1][0] + v_cross_dx_subtract[0], A[0][1] + A[1][1] + v_cross_dx_subtract[1],
                  A[0][2] + A[1][2] + v_cross_dx_subtract[2]};
  double AA_MAG = sqrt(AA[0] * AA[0] + AA[1] * AA[1] + AA[2] * AA[2]);
  if (AA_MAG < MFTL_VERYVERYSMALL) return;  // flat/degenerate ring guard

  double dot1 = dx[0][0] * A[0][0] + dx[0][1] * A[0][1] + dx[0][2] * A[0][2];
  double dot2 = dx[1][0] * A[1][0] + dx[1][1] * A[1][1] + dx[1][2] * A[1][2];
  double cross[3] = {v[1] * dx[0][2] - v[2] * dx[0][1], v[2] * dx[0][0] - v[0] * dx[0][2], v[0] * dx[0][1] - v[1] * dx[0][0]};
  double dot3 = dx[1][0] * cross[0] + dx[1][1] * cross[1] + dx[1][2] * cross[2];
  double h = -(dot1 + dot2 + dot3) / AA_MAG;

  for (int i = 0; i < 2; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      int vi = i == 0 ? v0 : v1;
      obj->vertices[vi].pos[k] += dx[i][k] + h * AA[k] / AA_MAG;
    }
    // keep the storage invariant: wrapped positions (no-op when non-periodic)
    MFTLWrapPosImg(obj->domain, obj->vertices[i == 0 ? v0 : v1].pos, obj->vertices[i == 0 ? v0 : v1].img);
  }

  // Stage 15 (L7): conserve the surface-scalar mass of the moved rings.
  int ss_ring[MFTL_VC_MAX_RING];
  int ss_n = mftlVcRingTris(obj, v0, ss_ring);
  mftlSSRescaleRing(obj, ss_ring, ss_n);
  ss_n = mftlVcRingTris(obj, v1, ss_ring);
  mftlSSRescaleRing(obj, ss_ring, ss_n);
}

void MFTLVolumeConservingSmoothing(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  mftlVcRequireCompleteMesh(obj, "MFTLVolumeConservingSmoothing");
  int maxiter = params->max_smoothing_laplacian_iter > 0 ? params->max_smoothing_laplacian_iter : 10;
  for (int it = 0; it < maxiter; ++it)
  {
    for (int j = 0; j < obj->n_half_edges; ++j)
    {
      if (obj->half_edges[j].twin_idx == MFTL_ERROR || j > obj->half_edges[j].twin_idx) continue;
      MFTLEdgeVolumeConservingSmoothing(obj, j, params);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* TSUR3D (tsur3d.c, Jiao JCP 2004)                                          */
/* ------------------------------------------------------------------------- */

int MFTLEdgeTSUR(MFTL_Object* obj, int he_idx, MFTL_RemeshParameters* params)
{
  MFTL_HalfEdge* he = &obj->half_edges[he_idx];
  if (he->flags & MFTL_FLAG_DEAD) return MFTL_ERROR;
  if (he->twin_idx == MFTL_ERROR) return MFTL_ERROR;
  if (mftlVcFeatureEdge(obj, he_idx, params->feature_angle_threshold)) return MFTL_ERROR;

  int he_next = he->next_idx;
  int he_prev = obj->half_edges[he_next].next_idx;
  int v0 = obj->half_edges[he_prev].vertex_idx;
  int v1 = he->vertex_idx;
  if (mftlVcFeatureVertex(obj, v0, params->feature_angle_threshold)) return MFTL_ERROR;
  if (mftlVcFeatureVertex(obj, v1, params->feature_angle_threshold)) return MFTL_ERROR;

  int ring0[MFTL_VC_MAX_RING], ring1[MFTL_VC_MAX_RING];
  int n0 = mftlVcRingTris(obj, v0, ring0);
  int n1 = mftlVcRingTris(obj, v1, ring1);
  if (n0 == 0 || n1 == 0) return MFTL_ERROR;

  // ring corner lists (two ordered corners per triangle, excluding the
  // centre). Positions are re-imaged into the centre vertex's periodic image
  // (exact copies when non-periodic) so a seam-straddling ring is contiguous.
  const MFTL_Domain* dom = obj->domain;
  const int any_periodic = MFTLDomainAnyPeriodic(dom);
  double corner0[MFTL_VC_MAX_RING * 2][3], corner1[MFTL_VC_MAX_RING * 2][3];
  int c0 = 0, c1 = 0;
  for (int r = 0; r < n0; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring0[r]], vv);
    for (int k = 0; k < 3; ++k)
    {
      int wi = (int) (vv[k] - obj->vertices);
      if (wi == v0) continue;
      memcpy(corner0[c0], vv[k]->pos, 3 * sizeof(double));
      if (any_periodic) MFTLUnwrapToAnchor(dom, corner0[c0], corner0[c0], obj->vertices[v0].pos);
      c0++;
    }
  }
  for (int r = 0; r < n1; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring1[r]], vv);
    for (int k = 0; k < 3; ++k)
    {
      int wi = (int) (vv[k] - obj->vertices);
      if (wi == v1) continue;
      memcpy(corner1[c1], vv[k]->pos, 3 * sizeof(double));
      if (any_periodic) MFTLUnwrapToAnchor(dom, corner1[c1], corner1[c1], obj->vertices[v1].pos);
      c1++;
    }
  }
  if (c0 == 0 || c1 == 0) return MFTL_ERROR;

  // 1. ring centroids p0/p1; endpoints TEMPORARILY moved there (MFTL1
  //    computes the area weights and normals in this shifted state).
  double p0[3] = {0, 0, 0}, p1[3] = {0, 0, 0};
  for (int i = 0; i < c0; ++i)
    for (int k = 0; k < 3; ++k) p0[k] += corner0[i][k] / (double) c0;
  for (int i = 0; i < c1; ++i)
    for (int k = 0; k < 3; ++k) p1[k] += corner1[i][k] / (double) c1;

  double tmp0[3], tmp1[3];
  memcpy(tmp0, obj->vertices[v0].pos, 3 * sizeof(double));
  memcpy(tmp1, obj->vertices[v1].pos, 3 * sizeof(double));
  memcpy(obj->vertices[v0].pos, p0, 3 * sizeof(double));
  memcpy(obj->vertices[v1].pos, p1, 3 * sizeof(double));

  double area_sum0 = 0.0, area_sum1 = 0.0, normal0[3] = {0, 0, 0}, normal1[3] = {0, 0, 0};
  for (int r = 0; r < n0; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring0[r]], vv);
    double e0[3], e1[3];
    MFTLGetMinimumImageVector(dom, e0, vv[0]->pos, vv[1]->pos);
    MFTLGetMinimumImageVector(dom, e1, vv[0]->pos, vv[2]->pos);
    double cx = e0[1] * e1[2] - e0[2] * e1[1], cy = e0[2] * e1[0] - e0[0] * e1[2], cz = e0[0] * e1[1] - e0[1] * e1[0];
    double area = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
    double tn[3];
    mftlVcTriNormal(obj, ring0[r], tn);
    area_sum0 += area;
    normal0[0] += area * tn[0];
    normal0[1] += area * tn[1];
    normal0[2] += area * tn[2];
  }
  for (int r = 0; r < n1; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring1[r]], vv);
    double e0[3], e1[3];
    MFTLGetMinimumImageVector(dom, e0, vv[0]->pos, vv[1]->pos);
    MFTLGetMinimumImageVector(dom, e1, vv[0]->pos, vv[2]->pos);
    double cx = e0[1] * e1[2] - e0[2] * e1[1], cy = e0[2] * e1[0] - e0[0] * e1[2], cz = e0[0] * e1[1] - e0[1] * e1[0];
    double area = 0.5 * sqrt(cx * cx + cy * cy + cz * cz);
    double tn[3];
    mftlVcTriNormal(obj, ring1[r], tn);
    area_sum1 += area;
    normal1[0] += area * tn[0];
    normal1[1] += area * tn[1];
    normal1[2] += area * tn[2];
  }

  // restore endpoints before the projection steps
  memcpy(obj->vertices[v0].pos, tmp0, 3 * sizeof(double));
  memcpy(obj->vertices[v1].pos, tmp1, 3 * sizeof(double));

  if (area_sum0 < MFTL_VERYVERYSMALL || area_sum1 < MFTL_VERYVERYSMALL) return MFTL_ERROR;
  for (int k = 0; k < 3; ++k)
  {
    normal0[k] /= area_sum0;
    normal1[k] /= area_sum1;
  }
  // MFTL1 normalises each endpoint normal (MFTL_NormalizeVector, 1e-15 floor) BEFORE the
  // sum; summing the unnormalised averages tilts n whenever |normal0| != |normal1|.
  const double len0 = MFTL_MAX(1e-15, sqrt(normal0[0] * normal0[0] + normal0[1] * normal0[1] + normal0[2] * normal0[2]));
  const double len1 = MFTL_MAX(1e-15, sqrt(normal1[0] * normal1[0] + normal1[1] * normal1[1] + normal1[2] * normal1[2]));
  for (int k = 0; k < 3; ++k)
  {
    normal0[k] /= len0;
    normal1[k] /= len1;
  }

  // 2. common tangent normal
  double normal[3] = {normal0[0] + normal1[0], normal0[1] + normal1[1], normal0[2] + normal1[2]};
  double nlen = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
  if (nlen < MFTL_VERYVERYSMALL) return MFTL_ERROR;  // degenerate normals guard
  normal[0] /= nlen;
  normal[1] /= nlen;
  normal[2] /= nlen;

  // 3-5. project the endpoints to the tangent plane
  double m[3], t0[3], t1[3];
  if (any_periodic)
  {
    // seam-safe midpoint: m = tmp0 + 0.5 * minimage(tmp0 -> tmp1)
    double e01[3];
    MFTLGetMinimumImageVector(dom, e01, tmp0, tmp1);
    for (int k = 0; k < 3; ++k) m[k] = tmp0[k] + 0.5 * e01[k];
  }
  else
  {
    // raw arithmetic (bit-identical to the pre-periodic path)
    m[0] = 0.5 * (tmp0[0] + tmp1[0]);
    m[1] = 0.5 * (tmp0[1] + tmp1[1]);
    m[2] = 0.5 * (tmp0[2] + tmp1[2]);
  }
  // v1's position re-imaged next to v0 (plain copy when non-periodic):
  // t1/p1 must live in the same frame as m.
  double tmp1_img[3];
  MFTLUnwrapToAnchor(dom, tmp1_img, tmp1, tmp0);
  for (int k = 0; k < 3; ++k) t0[k] = tmp0[k] - m[k];
  double h0 = t0[0] * normal[0] + t0[1] * normal[1] + t0[2] * normal[2];
  for (int k = 0; k < 3; ++k) t1[k] = tmp1_img[k] - m[k];
  double h1 = t1[0] * normal[0] + t1[1] * normal[1] + t1[2] * normal[2];
  p0[0] = tmp0[0] - h0 * normal[0];
  p0[1] = tmp0[1] - h0 * normal[1];
  p0[2] = tmp0[2] - h0 * normal[2];
  p1[0] = tmp1_img[0] - h1 * normal[0];
  p1[1] = tmp1_img[1] - h1 * normal[1];
  p1[2] = tmp1_img[2] - h1 * normal[2];

  // 6. local one-ring volumes about the projected points (vertices re-imaged
  // next to the projected point: a plain copy when non-periodic)
  double vol0 = 0.0, vol1 = 0.0, tetra[3][3];
  for (int r = 0; r < n0; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring0[r]], vv);
    for (int i = 0; i < 3; ++i)
    {
      double u[3];
      MFTLUnwrapToAnchor(dom, u, vv[i]->pos, p0);
      for (int k = 0; k < 3; ++k) tetra[i][k] = u[k] - p0[k];
    }
    vol0 += mftlVcDet3(tetra);
  }
  vol0 /= 6.0;
  for (int r = 0; r < n1; ++r)
  {
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(obj, &obj->triangles[ring1[r]], vv);
    for (int i = 0; i < 3; ++i)
    {
      double u[3];
      MFTLUnwrapToAnchor(dom, u, vv[i]->pos, p1);
      for (int k = 0; k < 3; ++k) tetra[i][k] = u[k] - p1[k];
    }
    vol1 += mftlVcDet3(tetra);
  }
  vol1 /= 6.0;

  // 7. volume-conserving height
  if (fabs(h0) < MFTL_VERYVERYSMALL || fabs(h1) < MFTL_VERYVERYSMALL) return MFTL_ERROR;
  double h = (vol0 + vol1) / (vol0 / h0 + vol1 / h1);
  if (!isfinite(h)) return MFTL_ERROR;  // NaN guard (MFTL1 isnan path)

  // 8. move + domain handling. Stage-13 resolution of the Stage-09
  // carry-forward: on periodic axes the moved endpoint WRAPS back into the
  // domain (clamping would pin the surface to the periodic face); on
  // non-periodic axes the original clamp is kept exactly.
  double nh[3] = {normal[0] * h, normal[1] * h, normal[2] * h};
  double* pos0 = obj->vertices[v0].pos;
  double* pos1 = obj->vertices[v1].pos;
  double old0[3], old1[3];  // Stage 16B: displacement rule for the image counters
  memcpy(old0, pos0, sizeof(old0));
  memcpy(old1, pos1, sizeof(old1));
  for (int k = 0; k < 3; ++k)
  {
    pos0[k] = p0[k] + nh[k];
    pos1[k] = p1[k] + nh[k];
  }
  MFTLApplyPeriodicity(dom, pos0);
  MFTLApplyPeriodicity(dom, pos1);
  MFTLImgFollow(dom, obj->vertices[v0].img, old0, pos0);
  MFTLImgFollow(dom, obj->vertices[v1].img, old1, pos1);
  for (int k = 0; k < 3; ++k)
  {
    if (dom->is_periodic[k]) continue;  // wrapped above
    double lo = obj->domain->global_min[k] + 1e-7;
    double hi = obj->domain->global_max[k] - 1e-7;
    pos0[k] = MFTL_MAX(lo, MFTL_MIN(pos0[k], hi));
    pos1[k] = MFTL_MAX(lo, MFTL_MIN(pos1[k], hi));
  }

  // Stage 15 (L7): conserve the surface-scalar mass of the moved rings.
  mftlSSRescaleRing(obj, ring0, n0);
  mftlSSRescaleRing(obj, ring1, n1);
  return MFTL_SUCCESS;
}

void MFTLTSUR3D(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  mftlVcRequireCompleteMesh(obj, "MFTLTSUR3D");
  for (int j = 0; j < obj->n_half_edges; ++j)
  {
    if (obj->half_edges[j].flags & MFTL_FLAG_DEAD) continue;
    if (obj->half_edges[j].twin_idx == MFTL_ERROR || j > obj->half_edges[j].twin_idx) continue;
    MFTLEdgeTSUR(obj, j, params);
  }
}

/* ------------------------------------------------------------------------- */
/* Roughness-triggered smoothing (smoothing.c:Object_Roughness_Smoothing)    */
/* ------------------------------------------------------------------------- */

void MFTLRoughnessSmoothing(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  mftlVcRequireCompleteMesh(obj, "MFTLRoughnessSmoothing");
  // Fill per-vertex dihedral roughness_mean (collective; also returns the
  // global geometric Laplacian, which we recompute per endpoint below).
  MFTLComputeRoughness(obj);

  for (int j = 0; j < obj->n_half_edges; ++j)
  {
    MFTL_HalfEdge* he = &obj->half_edges[j];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (he->twin_idx == MFTL_ERROR || j > he->twin_idx) continue;

    int he_next = he->next_idx;
    int he_prev = obj->half_edges[he_next].next_idx;
    int v0 = obj->half_edges[he_prev].vertex_idx;
    int v1 = he->vertex_idx;

    double gl1 = mftlVcLocalGL(obj, v0);
    double gl2 = mftlVcLocalGL(obj, v1);
    double max_gl = MFTL_MAX(gl1, gl2);
    double max_rough = MFTL_MAX(obj->vertices[v0].roughness_mean, obj->vertices[v1].roughness_mean);

    if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_KUPRAT)
    {
      if (max_rough > params->max_roughness || max_gl > params->max_geometric_laplacian)
        MFTLEdgeVolumeConservingSmoothing(obj, j, params);
    }
    else if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_TSUR)
    {
      if (max_rough > params->max_roughness) MFTLEdgeTSUR(obj, j, params);
    }
  }
}

/* ------------------------------------------------------------------------- */
/* Stage 10: parallel-safe drivers                                           */
/*                                                                         */
/* Same per-edge steps as the serial drivers, but every edge is gated to   */
/* the MFTL1 parallel-interior rule (mftlVcEdgeInteriorParallel): both     */
/* endpoints owned by this rank and both adjacent triangles local and      */
/* ghost-free. Rings around owned vertices are complete by the ghost-layer */
/* invariant, so each step is exactly local and cannot corrupt a ghost.    */
/* The caller (MFTLRemeshEnsembleShiftScheme) refreshes ghost copies with  */
/* MFTLUpdateGhostPositions afterwards.                                    */
/* ------------------------------------------------------------------------- */

void MFTLVolumeConservingSmoothingParallel(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  int maxiter = params->max_smoothing_laplacian_iter > 0 ? params->max_smoothing_laplacian_iter : 10;
  for (int it = 0; it < maxiter; ++it)
  {
    for (int j = 0; j < obj->n_half_edges; ++j)
    {
      if (obj->half_edges[j].twin_idx == MFTL_ERROR || j > obj->half_edges[j].twin_idx) continue;
      if (!mftlVcEdgeInteriorParallel(obj, j)) continue;
      MFTLEdgeVolumeConservingSmoothing(obj, j, params);
    }
  }
}

void MFTLTSUR3DParallel(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  for (int j = 0; j < obj->n_half_edges; ++j)
  {
    if (obj->half_edges[j].flags & MFTL_FLAG_DEAD) continue;
    if (obj->half_edges[j].twin_idx == MFTL_ERROR || j > obj->half_edges[j].twin_idx) continue;
    if (!mftlVcEdgeInteriorParallel(obj, j)) continue;
    MFTLEdgeTSUR(obj, j, params);
  }
}

void MFTLRoughnessSmoothingParallel(MFTL_Object* obj, MFTL_RemeshParameters* params)
{
  // Collective roughness fill (identical on all ranks); only interior edges
  // are smoothed, and only when BOTH endpoints are owned does the trigger
  // data mean the same as on a serial mesh.
  MFTLComputeRoughness(obj);

  for (int j = 0; j < obj->n_half_edges; ++j)
  {
    MFTL_HalfEdge* he = &obj->half_edges[j];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (he->twin_idx == MFTL_ERROR || j > he->twin_idx) continue;
    if (!mftlVcEdgeInteriorParallel(obj, j)) continue;

    int he_next = he->next_idx;
    int he_prev = obj->half_edges[he_next].next_idx;
    int v0 = obj->half_edges[he_prev].vertex_idx;
    int v1 = he->vertex_idx;

    double gl1 = mftlVcLocalGL(obj, v0);
    double gl2 = mftlVcLocalGL(obj, v1);
    double max_gl = MFTL_MAX(gl1, gl2);
    double max_rough = MFTL_MAX(obj->vertices[v0].roughness_mean, obj->vertices[v1].roughness_mean);

    if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_KUPRAT)
    {
      if (max_rough > params->max_roughness || max_gl > params->max_geometric_laplacian)
        MFTLEdgeVolumeConservingSmoothing(obj, j, params);
    }
    else if (params->volume_conserving_smoothing == MFTL_VCSMOOTH_TSUR)
    {
      if (max_rough > params->max_roughness) MFTLEdgeTSUR(obj, j, params);
    }
  }
}
