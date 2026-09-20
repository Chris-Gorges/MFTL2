/*
 * Object generators: capped cylinder, box, cube (refinable), and a single
 * quad face — MFTL1 parity per plans/04-generators.md.
 *
 * Lifecycle mirrors MFTLGenerateSphere (src/general/sphere.c): rank 0 builds
 * the mesh from position/index triples, half-edge topology is rebuilt with
 * MFTLRebuildTwins, gid counters are broadcast, and the caller distributes
 * with MFTLRedistributeListOfObjects. Generated meshes are deterministic and
 * therefore identical on any rank count.
 *
 * Winding: closed generators orient every triangle outward (each triangle is
 * checked against an outward reference point during construction), so
 * MFTLObjectVolume is POSITIVE and the surface is closed (Euler chi = 2).
 * MFTLGenerateFace is an open patch (chi = 1, volume undefined).
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "mftl_memory.h"

// ---------------------------------------------------------------------------
// Core builder: shared with mesh I/O (MFTLBuildTopologyFromTriples in
// topology.c). Positions are wrapped per axis here, as the generators do.
// ---------------------------------------------------------------------------
static void mftlBuildMeshFromTriples(MFTL_Object* obj, int nV, const double positions[][3], int nF, const int triples[][3])
{
  MFTL_Domain* domain = obj->domain;

  double (*wrapped)[3] = malloc(nV * sizeof(*wrapped));
  MFTL_ASSERT(wrapped != MFTL_NULL);
  for (int i = 0; i < nV; ++i)
  {
    memcpy(wrapped[i], positions[i], 3 * sizeof(double));
    MFTLApplyPeriodicity(domain, wrapped[i]);
  }

  MFTLBuildTopologyFromTriples(obj, nV, (const double (*)[3]) wrapped, nF, triples);
  free(wrapped);
}

// Collective tail identical to MFTLGenerateSphere: broadcast gid counters so
// non-owning ranks agree, update global stats, flatten connectivity.
static void mftlGeneratorFinish(MFTL_Object* obj)
{
  MFTL_Domain* domain = obj->domain;

  MPI_Bcast(&obj->next_vertex_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&obj->next_triangle_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&obj->next_halfedge_gid, 1, MPI_INT, 0, domain->comm_world);

  MFTLObjectUpdateGlobalStats(obj);
  MFTLFlattenConnectivity(obj);
}

// Orient one triangle so its normal faces `outward_ref` (a point strictly on
// the outward side of the triangle's plane). Swaps b/c in-place if needed.
static void mftlOrientOutward(const double pa[3], double pb[3], double pc[3], const double outward_ref[3])
{
  double e1[3] = {pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
  double e2[3] = {pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
  double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
  double to_ref[3] = {outward_ref[0] - pa[0], outward_ref[1] - pa[1], outward_ref[2] - pa[2]};

  if (n[0] * to_ref[0] + n[1] * to_ref[1] + n[2] * to_ref[2] < 0.0)
  {
    double tmp[3];
    memcpy(tmp, pb, sizeof(tmp));
    memcpy(pb, pc, sizeof(tmp));
    memcpy(pc, tmp, sizeof(tmp));
  }
}

// ---------------------------------------------------------------------------
// Box
// ---------------------------------------------------------------------------
void MFTLGenerateBox(MFTL_Object* obj, const double lo[3], const double hi[3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(lo != MFTL_NULL && hi != MFTL_NULL);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLGenerateBox: object must be empty");
  MFTL_Domain* domain = obj->domain;
  if (domain->rank != 0)
  {
    mftlGeneratorFinish(obj);
    return;
  }

  double positions[8][3];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k) positions[i][k] = (i & (1 << k)) ? hi[k] : lo[k];

  // 6 quad faces; each split into 2 triangles, oriented outward (reference
  // point one domain-size beyond the face centre).
  int triples[12][3];
  int t = 0;
  for (int axis = 0; axis < 3; ++axis)
  {
    int u = (axis + 1) % 3;
    int w = (axis + 2) % 3;
    for (int side = 0; side < 2; ++side)
    {
      double ref[3] = {0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), 0.5 * (lo[2] + hi[2])};
      double size = fmax(fmax(hi[0] - lo[0], hi[1] - lo[1]), hi[2] - lo[2]);
      ref[axis] += (side == 0 ? -1.0 : 1.0) * size;

      // 4 corners of the face, ordered CCW around the outward axis
      int corners[4];
      int c = 0;
      for (int i = 0; i < 8; ++i)
      {
        double p[3] = {positions[i][0], positions[i][1], positions[i][2]};
        if (fabs(p[axis] - (side == 0 ? lo[axis] : hi[axis])) < 1e-12) corners[c++] = i;
      }

      // order corners by angle around the face centre viewed from ref
      int quad[4];
      memcpy(quad, corners, sizeof(quad));

      // sort corners by angle around the face centre viewed from ref
      double centre[3] = {0, 0, 0};
      for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 3; ++k) centre[k] += positions[quad[i]][k] / 4.0;
      for (int i = 1; i < 4; ++i)
      {
        double ang_i = atan2(positions[quad[i]][w] - centre[w], positions[quad[i]][u] - centre[u]);
        for (int j = i + 1; j < 4; ++j)
        {
          double ang_j = atan2(positions[quad[j]][w] - centre[w], positions[quad[j]][u] - centre[u]);
          if (ang_j < ang_i)
          {
            int tmp = quad[i];
            quad[i] = quad[j];
            quad[j] = tmp;
            ang_i = ang_j;
          }
        }
      }

      // Ensure CCW when viewed from the reference (outward) side.
      double p0[3] = {positions[quad[0]][0], positions[quad[0]][1], positions[quad[0]][2]};
      double p1[3] = {positions[quad[1]][0], positions[quad[1]][1], positions[quad[1]][2]};
      double p2[3] = {positions[quad[2]][0], positions[quad[2]][1], positions[quad[2]][2]};
      double e1[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]};
      double e2[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      double n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
      double to_ref[3] = {ref[0] - p0[0], ref[1] - p0[1], ref[2] - p0[2]};
      if (n[0] * to_ref[0] + n[1] * to_ref[1] + n[2] * to_ref[2] < 0.0)
      {
        int tmp = quad[1];
        quad[1] = quad[3];
        quad[3] = tmp;
      }

      triples[t][0] = quad[0];
      triples[t][1] = quad[1];
      triples[t][2] = quad[2];
      t++;
      triples[t][0] = quad[0];
      triples[t][1] = quad[2];
      triples[t][2] = quad[3];
      t++;
    }
  }
  MFTL_ASSERT(t == 12);

  mftlBuildMeshFromTriples(obj, 8, (const double (*)[3]) positions, 12, (const int (*)[3]) triples);
  mftlGeneratorFinish(obj);
}

// ---------------------------------------------------------------------------
// Cube = box [-1,1]^3, then flat 1-to-4 refinement
// ---------------------------------------------------------------------------
static int mftlTessellateFlat(MFTL_Object* object)
{
  const int old_nv = object->n_vertices;
  const int old_nt = object->n_triangles;
  const int old_nhe = object->n_half_edges;

  MFTL_Vertex* old_vertices = object->vertices;
  MFTL_Triangle* old_triangles = object->triangles;
  MFTL_HalfEdge* old_half_edges = object->half_edges;

  const int old_nedges = old_nhe / 2;
  const int new_nv = old_nv + old_nedges;
  const int new_nt = old_nt * 4;
  const int new_nhe = new_nt * 3;

  MFTL_Vertex* new_vertices = calloc(new_nv, sizeof(MFTL_Vertex));
  MFTL_Triangle* new_triangles = calloc(new_nt, sizeof(MFTL_Triangle));
  MFTL_HalfEdge* new_half_edges = calloc(new_nhe, sizeof(MFTL_HalfEdge));
  if (!new_vertices || !new_triangles || !new_half_edges)
  {
    fprintf(stderr, "[Error] Allocation failed in mftlTessellateFlat\n");
    return MFTL_ERROR;
  }

  memcpy(new_vertices, old_vertices, old_nv * sizeof(MFTL_Vertex));
  for (int i = 0; i < old_nv; ++i) new_vertices[i].out_he_idx = MFTL_ERROR;

  int* midpoint_map = malloc(old_nhe * sizeof(int));
  for (int i = 0; i < old_nhe; ++i) midpoint_map[i] = MFTL_ERROR;

  int next_mid_idx = old_nv;
  for (int i = 0; i < old_nhe; ++i)
  {
    if (midpoint_map[i] != MFTL_ERROR) continue;
    int twin_idx = old_half_edges[i].twin_idx;
    if (twin_idx < 0 || twin_idx >= old_nhe) return MFTL_ERROR;

    const MFTL_Vertex* v1 = &old_vertices[old_half_edges[i].vertex_idx];
    const MFTL_Vertex* v0 = &old_vertices[old_half_edges[twin_idx].vertex_idx];

    MFTL_Vertex* mid = &new_vertices[next_mid_idx];
    mid->pos[0] = 0.5 * (v0->pos[0] + v1->pos[0]);
    mid->pos[1] = 0.5 * (v0->pos[1] + v1->pos[1]);
    mid->pos[2] = 0.5 * (v0->pos[2] + v1->pos[2]);
    mid->global_id = object->next_vertex_gid++;
    mid->owner_proc = object->domain->rank;
    mid->is_ghost = MFTL_FALSE;
    mid->out_he_idx = MFTL_ERROR;

    midpoint_map[i] = next_mid_idx;
    midpoint_map[twin_idx] = next_mid_idx;
    next_mid_idx++;
  }

  int new_tri_idx = 0;
  int new_he_idx = 0;
  for (int i = 0; i < old_nt; ++i)
  {
    const MFTL_Triangle* old_tri = &old_triangles[i];
    int he0 = old_tri->he_idx;
    int he1 = old_half_edges[he0].next_idx;
    int he2 = old_half_edges[he1].next_idx;

    int v0 = old_half_edges[he2].vertex_idx;
    int v1 = old_half_edges[he0].vertex_idx;
    int v2 = old_half_edges[he1].vertex_idx;
    int m0 = midpoint_map[he0];
    int m1 = midpoint_map[he1];
    int m2 = midpoint_map[he2];

    int sub[4][3] = {{v0, m0, m2}, {v1, m1, m0}, {v2, m2, m1}, {m0, m1, m2}};

    for (int j = 0; j < 4; ++j)
    {
      MFTL_Triangle* tri = &new_triangles[new_tri_idx];
      tri->global_id = object->next_triangle_gid++;
      tri->owner_proc = object->domain->rank;
      tri->is_ghost = MFTL_FALSE;
      tri->he_idx = new_he_idx;

      for (int k = 0; k < 3; ++k)
      {
        MFTL_HalfEdge* he = &new_half_edges[new_he_idx + k];
        he->global_id = object->next_halfedge_gid++;
        he->triangle_idx = new_tri_idx;
        he->vertex_idx = sub[j][(k + 1) % 3];
        he->next_idx = new_he_idx + ((k + 1) % 3);
        he->twin_idx = MFTL_ERROR;
        he->boundary = MFTL_TRUE;
        he->flags = 0;

        new_vertices[sub[j][k]].out_he_idx = new_he_idx + k;
      }
      new_tri_idx++;
      new_he_idx += 3;
    }
  }

  free(object->vertices);
  free(object->triangles);
  free(object->half_edges);
  free(midpoint_map);

  object->vertices = new_vertices;
  object->triangles = new_triangles;
  object->half_edges = new_half_edges;
  object->n_vertices = new_nv;
  object->n_triangles = new_nt;
  object->n_half_edges = new_nhe;
  object->capacity_vertices = new_nv;
  object->capacity_triangles = new_nt;
  object->capacity_half_edges = new_nhe;

  MFTLRebuildTwins(object);
  return MFTL_SUCCESS;
}

void MFTLGenerateCube(MFTL_Object* obj, int refine)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(refine >= 0);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLGenerateCube: object must be empty");
  MFTL_Domain* domain = obj->domain;

  // MFTLGenerateBox performs the first collective finish (unrefined gid
  // counters) on ALL ranks — do not early-return non-root ranks here.
  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  MFTLGenerateBox(obj, lo, hi);

  // Refine locally on rank 0.
  if (domain->rank == 0)
  {
    for (int r = 0; r < refine; ++r)
    {
      if (mftlTessellateFlat(obj) != MFTL_SUCCESS)
      {
        fprintf(stderr, "MFTLGenerateCube: flat tessellation failed\n");
        break;
      }
    }
  }

  // Refinement advances the gid counters on rank 0 only; re-broadcast them
  // (plus refreshed global stats) so every rank agrees, exactly the tail
  // MFTLGenerateSphere uses.
  mftlGeneratorFinish(obj);
}

// ---------------------------------------------------------------------------
// Capped cylinder, axis +Z, z in [0, L]
// ---------------------------------------------------------------------------
void MFTLGenerateCylinder(MFTL_Object* obj, double dr, double radius, double L, double ellipticity)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLGenerateCylinder: object must be empty");
  MFTL_ASSERT(dr > 0.0 && radius > 0.0 && L > 0.0);
  MFTL_Domain* domain = obj->domain;
  if (domain->rank != 0)
  {
    mftlGeneratorFinish(obj);
    return;
  }

  // MFTL1 parameterization
  const double n = 0.9;
  int Np = (int) llround(pow(radius, n) / (MFTL_PI * dr));
  if (Np < 3) Np = 3;
  int Nz = (int) llround(L / (10.0 * dr) + 1.0);
  if (Nz < 2) Nz = 2;

  // Ellipticity: [0,1) squeezes x, (-1,0) squeezes y (MFTL1 convention).
  double ex = 1.0, ey = 1.0;
  if (ellipticity >= 0.0 && ellipticity < 1.0)
    ex = sqrt(1.0 - ellipticity * ellipticity);
  else if (ellipticity < 0.0 && ellipticity > -1.0)
    ey = sqrt(1.0 - ellipticity * ellipticity);
  else
    fprintf(stderr, "Warning: Ellipticity bounds need to be -1.0 < e < 1.0.\n");

  const int nV = Np * Nz + 2;                       // + cap centres
  const int nF = 2 * Np * (Nz - 1) + 2 * Np;        // side + caps

  double (*positions)[3] = malloc(nV * sizeof(*positions));
  int (*triples)[3] = malloc(nF * sizeof(*triples));
  if (!positions || !triples)
  {
    fprintf(stderr, "[Error] Allocation failed in MFTLGenerateCylinder\n");
    free(positions);
    free(triples);
    return;
  }

  const double theta = 2.0 * MFTL_PI / (double) Np;
  for (int k = 0; k < Nz; ++k)
  {
    double z = L * (double) k / (double) (Nz - 1);
    for (int i = 0; i < Np; ++i)
    {
      double* p = positions[i + Np * k];
      p[0] = ex * radius * cos(theta * (double) i);
      p[1] = ey * radius * sin(theta * (double) i);
      p[2] = z;
    }
  }
  positions[Np * Nz][0] = 0.0;  // bottom cap centre
  positions[Np * Nz][1] = 0.0;
  positions[Np * Nz][2] = 0.0;
  positions[Np * Nz + 1][0] = 0.0;  // top cap centre
  positions[Np * Nz + 1][1] = 0.0;
  positions[Np * Nz + 1][2] = L;

  int t = 0;
  for (int k = 0; k < Nz - 1; ++k)
  {
    for (int i = 0; i < Np; ++i)
    {
      int i1 = (i + 1) % Np;
      int a = i + Np * k;
      int b = i1 + Np * k;
      int c = i1 + Np * (k + 1);
      int d = i + Np * (k + 1);

      // radial outward reference at the quad centre
      double mid_ang = theta * ((double) i + 0.5);
      double ref[3] = {2.0 * ex * radius * cos(mid_ang), 2.0 * ey * radius * sin(mid_ang),
                       0.5 * (positions[a][2] + positions[c][2])};

      memcpy(triples[t], (int[]) {a, b, c}, 3 * sizeof(int));
      mftlOrientOutward(positions[a], positions[b], positions[c], ref);
      t++;
      memcpy(triples[t], (int[]) {a, c, d}, 3 * sizeof(int));
      mftlOrientOutward(positions[a], positions[c], positions[d], ref);
      t++;
    }
  }

  int cb = Np * Nz;  // bottom cap centre
  int ct = Np * Nz + 1;
  for (int i = 0; i < Np; ++i)
  {
    int i1 = (i + 1) % Np;
    double ref_bottom[3] = {0.0, 0.0, -L};
    double ref_top[3] = {0.0, 0.0, 2.0 * L};

    memcpy(triples[t], (int[]) {cb, i1, i}, 3 * sizeof(int));
    mftlOrientOutward(positions[cb], positions[i1], positions[i], ref_bottom);
    t++;
    memcpy(triples[t], (int[]) {ct, i + Np * (Nz - 1), i1 + Np * (Nz - 1)}, 3 * sizeof(int));
    mftlOrientOutward(positions[ct], positions[i + Np * (Nz - 1)], positions[i1 + Np * (Nz - 1)], ref_top);
    t++;
  }
  MFTL_ASSERT(t == nF);

  mftlBuildMeshFromTriples(obj, nV, positions, nF, triples);
  free(positions);
  free(triples);
  mftlGeneratorFinish(obj);
}

// ---------------------------------------------------------------------------
// Open quad face (2 triangles along the v0-v3 diagonal; MFTL1 face.c split)
// ---------------------------------------------------------------------------
void MFTLGenerateFace(MFTL_Object* obj, const double v[4][3])
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(v != MFTL_NULL);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLGenerateFace: object must be empty");
  MFTL_Domain* domain = obj->domain;
  if (domain->rank != 0)
  {
    mftlGeneratorFinish(obj);
    return;
  }

  double positions[4][3];
  memcpy(positions, v, sizeof(positions));

  int triples[2][3] = {{0, 1, 3}, {0, 3, 2}};

  mftlBuildMeshFromTriples(obj, 4, (const double (*)[3]) positions, 2, (const int (*)[3]) triples);
  mftlGeneratorFinish(obj);
}
