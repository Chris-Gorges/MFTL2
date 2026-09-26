/**
 * @file sphere_creation.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief The icosphere: an icosahedron, N rounds of 1-to-4 midpoint
 *        subdivision, one final projection onto the sphere.
 *
 * Subdivision emits an indexed triangle list and lets
 * MFTLBuildTopologyFromIndexList() build the topology, so there is one
 * half-edge construction loop in the library. Old vertices keep their indices
 * (midpoints are appended), so the vertex pool grows in place.
 */

#include "mftl_meshgen.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "mftl_vectoroperations.h"

/**
 * @brief One level of 1-to-4 subdivision, in place.
 *
 *      v2
 *     /  \
 *   m2 -- m1
 *   / \  / \
 * v0 -- m0 -- v1
 *
 * Each edge's midpoint (the plain average, NOT projected: projection happens
 * once, after the last level) is created once: `midpoint_map[h]` is written for
 * a half-edge and its twin together, so the second triangle finds it by a
 * direct array read -- "the other side of this edge" is already an index, no
 * hash map needed. The sub-triangles keep their parent's rotational order, which
 * propagates the outward winding. Requires a closed mesh.
 *
 * @param object The mesh; every triangle becomes four.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an open mesh or a topology failure.
 */
static int tessellate_sphere(MFTL_Object *object)
{
  const int old_nv = object->verts.n;
  const int old_nt = object->tris.n;
  const int old_nhe = 3 * old_nt;
  const int old_nedges = old_nhe / 2;  // closed mesh: every edge has exactly two half-edges

  const int new_nv = old_nv + old_nedges;
  const int new_nt = 4 * old_nt;

  const int rank = object->domain->rank;

  /* ── 1. Midpoints, appended after the old vertices (growth preserves them). */
  MFTLObjectSetCounts(object, new_nv, new_nv, old_nt, old_nt);

  int *midpoint_map = (int *) MFTLMallocOrDie((size_t) old_nhe * sizeof(int), "tessellate_sphere midpoint map");
  for (int h = 0; h < old_nhe; ++h) midpoint_map[h] = MFTL_ERROR;

  double *const pos = object->verts.pos;
  MFTL_Vertex *const vr = object->verts.rec;
  const MFTL_HalfEdge *const he = object->tris.he;

  int next_mid = old_nv;

  for (int h = 0; h < old_nhe; ++h)
  {
    if (midpoint_map[h] != MFTL_ERROR) continue;  // already created via the twin

    const int twin = he[h].twin_idx;
    MFTL_ASSERT(twin != MFTL_ERROR && "tessellate_sphere: open mesh (boundary half-edge) is not supported");
    if (twin == MFTL_ERROR)
    {
      free(midpoint_map);
      return MFTL_ERROR;
    }

    const int v_end = he[h].vertex_idx;
    const int v_start = he[twin].vertex_idx;  // target of the twin == origin of h

    /* No minimum image needed: the mesh is built about the origin and only
     * translated (and wrapped) at the end. */
    pos[3 * next_mid + 0] = 0.5 * (pos[3 * v_start + 0] + pos[3 * v_end + 0]);
    pos[3 * next_mid + 1] = 0.5 * (pos[3 * v_start + 1] + pos[3 * v_end + 1]);
    pos[3 * next_mid + 2] = 0.5 * (pos[3 * v_start + 2] + pos[3 * v_end + 2]);

    vr[next_mid].global_id = object->next_vertex_gid++;
    vr[next_mid].owner_proc = rank;
    vr[next_mid].out_he_idx = MFTL_ERROR;  // assigned when the topology is rebuilt
    vr[next_mid].flags = 0;

    midpoint_map[h] = next_mid;
    midpoint_map[twin] = next_mid;
    next_mid++;
  }

  MFTL_ASSERT(next_mid == new_nv && "tessellate_sphere: midpoint count does not match the Euler formula");

  /* ── 2. Four sub-triangles per triangle, in the parent's rotational order. */
  int *tri_verts = (int *) MFTLMallocOrDie(3 * (size_t) new_nt * sizeof(int), "tessellate_sphere index list");

  int out = 0;
  for (int t = 0; t < old_nt; ++t)
  {
    const int h0 = 3 * t, h1 = 3 * t + 1, h2 = 3 * t + 2;

    const int v0 = he[h2].vertex_idx;  // origin of h0
    const int v1 = he[h0].vertex_idx;
    const int v2 = he[h1].vertex_idx;

    const int m0 = midpoint_map[h0];  // on edge v0-v1
    const int m1 = midpoint_map[h1];  // on edge v1-v2
    const int m2 = midpoint_map[h2];  // on edge v2-v0

    const int sub[4][3] = {
        {v0, m0, m2},  // corner 0
        {v1, m1, m0},  // corner 1
        {v2, m2, m1},  // corner 2
        {m0, m1, m2}  // centre
    };

    for (int j = 0; j < 4; ++j)
    {
      tri_verts[3 * out + 0] = sub[j][0];
      tri_verts[3 * out + 1] = sub[j][1];
      tri_verts[3 * out + 2] = sub[j][2];
      ++out;
    }
  }

  free(midpoint_map);

  /* ── 3. Rebuild the topology from the index list. */
  const int rc = MFTLBuildTopologyFromIndexList(object, tri_verts, new_nt);
  free(tri_verts);

  return rc;
}

/**
 * @brief Builds the regular icosahedron inscribed in the unit sphere: the 12
 *        vertices \f$(0, \pm1, \pm\varphi)\f$ and cyclic permutations, scaled by
 *        \f$1/\sqrt{1+\varphi^2}\f$ (so X = 0.5257, Z = 0.8507), and 20 faces
 *        wound counter-clockwise as seen from OUTSIDE.
 *
 * @param object The (empty) mesh to fill.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR.
 */
static int generate_icosahedron(MFTL_Object *object)
{
  const double X = 0.525731112119133606;
  const double Z = 0.850650808352039932;

  const double positions[12 * 3] = {-X, 0.0, Z, X,   0.0, Z,  -X,  0.0, -Z, X,   0.0, -Z, 0.0, Z,   X, 0.0, Z,  -X,
                                    0.0, -Z,  X, 0.0, -Z,  -X, Z,   X,   0.0, -Z, X,   0.0, Z,   -X, 0.0, -Z, -X, 0.0};

  const int tri_verts[20 * 3] = {0, 1, 4,  0, 4, 9,  9, 4,  5, 4,  8, 5, 4, 1,  8, 8, 1,  10, 8, 10, 3, 5, 8,  3, 5, 3, 2, 2,  3, 7,
                                 7, 3, 10, 7, 10, 6, 7, 6, 11, 11, 6, 0, 0, 6, 1, 6, 10, 1,  9, 11, 0, 9, 2, 11, 9, 5, 2, 7, 11, 2};

  return MFTLBuildIndexedMesh(object, positions, 12, tri_verts, 20);
}

int MFTLGenerateSphereLocal(MFTL_Object *object, double center[3], double radius, int refinements)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(object->domain != MFTL_NULL && "MFTLGenerateSphereLocal: object has no domain -- add it to an ensemble first");
  MFTL_ASSERT(radius >= 0.0 && "MFTLGenerateSphereLocal: radius must not be negative");
  MFTL_ASSERT(refinements >= 0 && "MFTLGenerateSphereLocal: refinements must not be negative");

  if (object == MFTL_NULL || object->domain == MFTL_NULL || center == MFTL_NULL) return MFTL_ERROR;
  if (radius < 0.0 || refinements < 0) return MFTL_ERROR;

  MFTL_Domain *domain = object->domain;
  int rc = MFTL_SUCCESS;

  /* Rank 0 builds; the others stay empty until the first redistribution. */
  if (domain->rank == 0)
  {
    rc = generate_icosahedron(object);

    for (int i = 0; rc == MFTL_SUCCESS && i < refinements; ++i) rc = tessellate_sphere(object);

    if (rc == MFTL_SUCCESS)
    {
      /* Project onto the sphere, scale, translate, wrap. */
      double *const pos = object->verts.pos;
      const int nv = object->verts.n;

      for (int v = 0; v < nv; ++v)
      {
        double *p = &pos[3 * v];
        MFTLNormalizeVector(p);
        p[0] = center[0] + radius * p[0];
        p[1] = center[1] + radius * p[1];
        p[2] = center[2] + radius * p[2];
        MFTLApplyPeriodicity(domain, p);
      }

      /* Closed, so this marks nothing; asserted. */
      const int n_boundary = MFTLMarkMeshBoundary(object);
      MFTL_ASSERT(n_boundary == 0 && "MFTLGenerateSphereLocal: the generated sphere is not closed");
      (void) n_boundary;
    }
  }
  else
  {
    /* Empty: mark the (vacuous) derived arrays valid. */
    MFTLObjectBuildDerived(object);
  }

  return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Publication: the collective half, separate so that N bodies cost two
 * collectives instead of 2N in strict sequence on rank 0's critical path (at
 * 5 us per collective, 10 s before the first step at a million bodies; not
 * measurable on a single node). The counters are rank 0's and are BROADCAST;
 * the counts are everybody's and are REDUCED -- one fused reduction would be
 * correct only while every non-root object is empty.
 * ════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Settles the ID counters and global counts of @p n bodies in two collectives.
 *
 * @param domain  The MPI world. All of @p objects must share it.
 * @param objects [n] bodies to publish, in the same order on every rank.
 * @param n       Bodies. Zero is a no-op, but every rank must still call.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on an allocation-free invalid argument.
 */
static int publish_bodies(MFTL_Domain *domain, MFTL_Object *const *objects, int n)
{
  if (domain == MFTL_NULL || (n > 0 && objects == MFTL_NULL)) return MFTL_ERROR;
  if (n <= 0) return MFTL_SUCCESS;

  /* One allocation, two slices: [3n counters | 2n local counts | 2n global]. */
  int *const buf = (int *) MFTLMallocOrDie((size_t) (7 * n) * sizeof(int), "generated body publication");
  int *const gids = buf;
  int *const local_counts = buf + 3 * n;
  int *const global_counts = buf + 5 * n;

  for (int o = 0; o < n; ++o)
  {
    const MFTL_Object *const obj = objects[o];
    MFTL_ASSERT(obj != MFTL_NULL);
    gids[3 * o + 0] = obj->next_vertex_gid;
    gids[3 * o + 1] = obj->next_triangle_gid;
    gids[3 * o + 2] = obj->next_halfedge_gid;

    /* Ownership is a prefix: n_local is the owned count. */
    local_counts[2 * o + 0] = obj->tris.n_local;
    local_counts[2 * o + 1] = obj->verts.n_local;
  }

  MPI_Bcast(gids, 3 * n, MPI_INT, 0, domain->comm_world);
  MPI_Allreduce(local_counts, global_counts, 2 * n, MPI_INT, MPI_SUM, domain->comm_world);

  for (int o = 0; o < n; ++o)
  {
    MFTL_Object *const obj = objects[o];
    obj->next_vertex_gid = gids[3 * o + 0];
    obj->next_triangle_gid = gids[3 * o + 1];
    obj->next_halfedge_gid = gids[3 * o + 2];

    obj->global_n_triangles = global_counts[2 * o + 0];
    obj->global_n_vertices = global_counts[2 * o + 1];
  }

  free(buf);
  return MFTL_SUCCESS;
}

int MFTLPublishGeneratedBodies(MFTL_Ensemble *ens)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  if (ens == MFTL_NULL) return MFTL_ERROR;

  return publish_bodies(&ens->domain, ens->local_objects, ens->n_local_objects);
}

int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements)
{
  const int rc = MFTLGenerateSphereLocal(object, center, radius, refinements);
  if (rc == MFTL_ERROR && (object == MFTL_NULL || object->domain == MFTL_NULL)) return rc;

  /* One body: one broadcast, one reduction. */
  MFTL_Object *one = object;
  (void) publish_bodies(object->domain, &one, 1);

  return rc;
}
