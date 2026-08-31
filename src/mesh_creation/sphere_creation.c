/**
 * @file sphere_creation.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief The icosphere: an icosahedron plus N rounds of 1-to-4 subdivision.
 *
 * Structured around one observation: subdivision naturally PRODUCES an indexed
 * triangle list, so it does not need half-edge construction of its own at all.
 * Hand-building the 4 triangles and 12 half-edges per old triangle would
 * duplicate -- slightly differently -- the loop in mesh_builder.c, which is
 * exactly how two code paths drift apart. Here subdivision computes midpoints and
 * emits the sub-triangles as vertex indices, and MFTLBuildTopologyFromIndexList
 * turns them into topology.
 *
 * Nor does it allocate and swap three parallel arrays per level. Old
 * vertices keep their indices under subdivision (midpoints are appended), so the
 * vertex pool is simply grown in place, and the triangle pool is rewritten.
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
 * @brief One level of 1-to-4 subdivision.
 *
 *      v2
 *     /  \
 *   m2 -- m1
 *   / \  / \
 * v0 -- m0 -- v1
 *
 * Midpoints are found without any hash map. The edge a midpoint belongs to is
 * identified by a HALF-EDGE index, and `midpoint_map[h]` is filled for both h
 * and its twin at once -- so the second half-edge of an edge finds the vertex
 * its partner created by a direct array read. That is O(1) per half-edge with no
 * hashing, and it is the reason this library needs no hash maps: on a half-edge
 * mesh, "the other side of this edge" is already an index.
 *
 * A closed mesh is required (every icosphere is one); a boundary half-edge here
 * would have no twin to record the shared midpoint through.
 *
 * @param object The mesh, subdivided IN PLACE: every triangle becomes four and
 *               every new vertex is projected back onto the sphere.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR if the topology could not be rebuilt.
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

  /* ── 1. Midpoints ────────────────────────────────────────────────────────
   *
   * The vertex pool is grown FIRST and the midpoints are appended, so old
   * vertices keep their indices and their positions are carried across by the
   * pool's own growth (which preserves contents). No copy loop, no swap. */
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

    /* No minimum-image correction: the icosphere is generated about the origin
     * and only translated onto its centre at the very end, so no edge of the
     * intermediate mesh can straddle a periodic face. */
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

  /* ── 2. The four sub-triangles of every old triangle ─────────────────────
   *
   * Emitted in the parent's rotational order, which is what propagates the
   * outward winding of the icosahedron face table to every refinement level. */
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

  /* ── 3. Rebuild the topology ─────────────────────────────────────────────
   *
   * From here the old half-edges are dead; MFTLBuildTopologyFromIndexList
   * rewrites the whole triangle pool, relinks the twins, re-marks the boundary
   * (nothing, on a closed mesh) and rebuilds the derived arrays. */
  const int rc = MFTLBuildTopologyFromIndexList(object, tri_verts, new_nt);
  free(tri_verts);

  return rc;
}

/**
 * @brief Builds a unit icosahedron: 12 vertices, 20 triangles, 60 half-edges.
 *
 * Faces are wound counter-clockwise as seen from OUTSIDE the body, so that
 * cross(v1 - v0, v2 - v0) yields an outward-pointing normal. Front tracking and
 * IBM rely on this: it is what makes the enclosed volume positive under the
 * divergence theorem and fixes the sign of surface tension and of the indicator
 * function. The 1-to-4 subdivision emits every sub-triangle in its parent's
 * rotational order, so the orientation propagates to all refinement levels.
 *
 * @param object The mesh, filled with the 12 vertices and 20 faces of a regular
 *               icosahedron on the unit sphere. The face table is wound
 *               counter-clockwise as seen from OUTSIDE.
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

int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(object->domain != MFTL_NULL && "MFTLGenerateSphere: object has no domain -- add it to an ensemble first");
  MFTL_ASSERT(radius >= 0.0 && "MFTLGenerateSphere: radius must not be negative");
  MFTL_ASSERT(refinements >= 0 && "MFTLGenerateSphere: refinements must not be negative");

  if (object == MFTL_NULL || object->domain == MFTL_NULL || center == MFTL_NULL) return MFTL_ERROR;
  if (radius < 0.0 || refinements < 0) return MFTL_ERROR;

  MFTL_Domain *domain = object->domain;
  int rc = MFTL_SUCCESS;

  /* ── Rank 0 builds; everyone else stays empty ────────────────────────────
   *
   * Generation is inherently serial and cheap next to a timestep, so it is not
   * worth partitioning: redistribution spreads the result afterwards, and it has
   * to run anyway. The only collective content here is the three ID counters and
   * the global-count reduction, which is O(1) messages. */
  if (domain->rank == 0)
  {
    rc = generate_icosahedron(object);

    for (int i = 0; rc == MFTL_SUCCESS && i < refinements; ++i) rc = tessellate_sphere(object);

    if (rc == MFTL_SUCCESS)
    {
      /* Project onto the sphere and move it into place. Streaming over the
       * position array: contiguous in, contiguous out, no records touched. */
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

      /* An icosphere is closed, so this finds nothing. Running it anyway is what
       * makes MFTL_FLAG_BOUNDARY authoritative for EVERY generated mesh: code
       * that branches on it behaves identically on closed and open surfaces
       * without having to ask which it was handed. */
      const int n_boundary = MFTLMarkMeshBoundary(object);
      MFTL_ASSERT(n_boundary == 0 && "MFTLGenerateSphere: the generated sphere is not closed");
      (void) n_boundary;
    }
  }
  else
  {
    /* Empty everywhere else. Marking the (vacuous) derived arrays valid keeps a
     * hot loop over zero triangles from tripping the staleness assertion, which
     * is otherwise the first thing a collective caller hits on a non-root rank. */
    MFTLObjectBuildDerived(object);
  }

  /* Every rank continues the same numbering scheme, so a rank that later
   * receives entities or creates new ones does not collide with rank 0's IDs.
   * One broadcast of three ints -- fused, because three separate Bcasts would be
   * three latencies for twelve bytes. */
  int gids[3] = {object->next_vertex_gid, object->next_triangle_gid, object->next_halfedge_gid};
  MPI_Bcast(gids, 3, MPI_INT, 0, domain->comm_world);
  object->next_vertex_gid = gids[0];
  object->next_triangle_gid = gids[1];
  object->next_halfedge_gid = gids[2];

  MFTLObjectUpdateGlobalStats(object);

  return rc;
}
