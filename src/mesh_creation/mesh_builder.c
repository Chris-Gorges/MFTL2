/**
 * @file mesh_builder.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief Indexed triangle list -> half-edge mesh.
 *
 * Every generator that is not built by subdivision goes through here, and so
 * would a mesh reader. Keeping one implementation is what stops the half-edge
 * conventions (slot k is the origin of half-edge 3t+k, dense global IDs, twins
 * linked, boundary recorded, derived arrays built) from being re-derived
 * slightly differently in each new generator.
 */

#include "mftl_meshgen.h"

#include <stdio.h>
#include <string.h>

#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"
#include "mftl_topology.h"

int MFTLBuildTopologyFromIndexList(MFTL_Object *object, const int *tri_verts, int n_triangles)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(tri_verts != MFTL_NULL);
  MFTL_ASSERT(object->verts.n >= 3 && "MFTLBuildTopologyFromIndexList: the vertex pool must be filled first");

  if (object == MFTL_NULL || tri_verts == MFTL_NULL) return MFTL_ERROR;
  if (n_triangles < 1 || object->verts.n < 3) return MFTL_ERROR;

  const int n_vertices = object->verts.n;
  const int rank = (object->domain != MFTL_NULL) ? object->domain->rank : 0;

  /* Grows the triangle pool -- and with it the half-edge array and every derived
   * array, because they are one allocation. Vertex counts are passed through
   * unchanged; this is the only sanctioned way to change a count. */
  MFTLObjectSetCounts(object, n_vertices, object->verts.n_local, n_triangles, n_triangles);

  /* Dense [0, n) per class, so the counters end exactly one past the last ID.
   * Triangles and half-edges restart at zero even when this is a re-build (as it
   * is for each subdivision level): the previous generation is retired wholesale
   * along with its IDs. Vertex IDs are deliberately NOT reset here -- the caller
   * owns them, and subdivision carries old vertices over verbatim. */
  object->next_triangle_gid = n_triangles;
  object->next_halfedge_gid = 3 * n_triangles;

  MFTL_Vertex *const vr = object->verts.rec;
  MFTL_Triangle *const tr = object->tris.rec;
  MFTL_HalfEdge *const he = object->tris.he;

  /* Every vertex's outgoing half-edge is re-derived below. Clearing first is what
   * makes this safe to call on an object whose previous half-edges have just been
   * retired -- a stale index there would point into someone else's fan. */
  for (int v = 0; v < n_vertices; ++v) vr[v].out_he_idx = MFTL_ERROR;

  for (int t = 0; t < n_triangles; ++t)
  {
    tr[t].global_id = t;
    tr[t].owner_proc = rank;
    tr[t].flags = 0;

    for (int k = 0; k < 3; ++k)
    {
      const int origin = tri_verts[3 * t + k];
      const int target = tri_verts[3 * t + (k + 1) % 3];

      MFTL_ASSERT(origin >= 0 && origin < n_vertices && target >= 0 && target < n_vertices && "MFTLBuildTopologyFromIndexList: vertex index out of range");
      MFTL_ASSERT(origin != target && "MFTLBuildTopologyFromIndexList: degenerate triangle (repeated vertex)");
      if (origin < 0 || origin >= n_vertices || target < 0 || target >= n_vertices) return MFTL_ERROR;

      const int h = 3 * t + k;  // implicit indexing: no he_idx, next_idx or triangle_idx to write
      he[h].global_id = h;
      he[h].vertex_idx = target;  // half-edges store their TARGET
      he[h].twin_idx = MFTL_ERROR;  // MFTLRebuildTwins links them below
      he[h].flags = 0;

      /* Any outgoing half-edge is a valid out_he_idx; later triangles simply
       * overwrite earlier ones. The one-ring rewind is what finds the true start
       * of a boundary vertex's fan, so no ordering is required here. */
      vr[origin].out_he_idx = h;
    }
  }

  MFTLRebuildTwins(object);

  /* The mesh is complete exactly here -- before any redistribution -- which is
   * the only moment at which "no twin" unambiguously means "no neighbouring
   * triangle in the whole surface". */
  MFTLMarkMeshBoundary(object);

  MFTLObjectBuildDerived(object);

  return MFTL_SUCCESS;
}

int MFTLBuildIndexedMesh(MFTL_Object *object, const double *positions, int n_vertices, const int *tri_verts, int n_triangles)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(object->tris.n == 0 && object->verts.n == 0 && "MFTLBuildIndexedMesh: object is not empty");

  if (object == MFTL_NULL || positions == MFTL_NULL || tri_verts == MFTL_NULL) return MFTL_ERROR;
  if (n_vertices < 3 || n_triangles < 1) return MFTL_ERROR;

  const int rank = (object->domain != MFTL_NULL) ? object->domain->rank : 0;

  MFTLObjectSetCounts(object, n_vertices, n_vertices, 0, 0);

  /* One memcpy: the caller's layout is already the pool's layout, xyz
   * interleaved. That is not a coincidence -- it is the same reason every
   * consumer gets a whole position in one cache line. */
  memcpy(object->verts.pos, positions, 3 * (size_t) n_vertices * sizeof(double));

  MFTL_Vertex *const vr = object->verts.rec;
  for (int v = 0; v < n_vertices; ++v)
  {
    vr[v].global_id = v;
    vr[v].owner_proc = rank;
    vr[v].out_he_idx = MFTL_ERROR;  // filled in by the topology pass
    vr[v].flags = 0;
  }
  object->next_vertex_gid = n_vertices;

  return MFTLBuildTopologyFromIndexList(object, tri_verts, n_triangles);
}
