/**
 * @file mesh_builder.c
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief Indexed triangle list -> half-edge mesh: the one conversion point, so
 *        the half-edge conventions are established in exactly one place.
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

/**
 * @brief Is @p tri_verts a usable index list: every index in range and no
 *        triangle with a repeated corner (a self-edge that could twin with
 *        itself)? Checked in Release, before anything is written.
 *
 * @param tri_verts   [3 * n_triangles] vertex indices.
 * @param n_triangles Number of triangles.
 * @param n_vertices  Size of the vertex pool the indices refer to.
 * @return `true` if every triangle has three distinct, in-range corners.
 */
static bool index_list_is_valid(const int *tri_verts, int n_triangles, int n_vertices)
{
  for (int t = 0; t < n_triangles; ++t)
  {
    const int a = tri_verts[3 * t + 0], b = tri_verts[3 * t + 1], c = tri_verts[3 * t + 2];
    if (a < 0 || a >= n_vertices || b < 0 || b >= n_vertices || c < 0 || c >= n_vertices) return MFTL_FALSE;
    if (a == b || b == c || c == a) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/**
 * @brief Is the (index-valid) list an ORIENTED edge-manifold: no directed edge
 *        used twice, and no two triangles with the same three corners?
 *
 * A directed edge used twice is a third triangle on an edge, or a neighbour
 * wound the other way, or a duplicated triangle: MFTLRebuildTwins() would pair
 * two of the half-edges and MFTLMarkMeshBoundary() would flag the rest as
 * surface boundary for good. A triangle and its reversal pair up on all three
 * edges into a closed two-triangle "pillow" that no other test sees. Every
 * other edge then has at most one half-edge per direction, which is what the
 * half-edge structure can represent. A vertex shared by otherwise separate fans
 * (a bowtie) is NOT refused.
 *
 * Directed edges are bucketed by origin, as in MFTLRebuildTwins(); the first
 * violation is printed to stderr.
 *
 * @param tri_verts   [3 * n_triangles] vertex indices, already index-valid.
 * @param n_triangles Number of triangles.
 * @param n_vertices  Size of the vertex pool.
 * @param caller      Name for the diagnostic.
 * @return `true` if the list can become a half-edge mesh.
 */
static bool index_list_is_manifold(const int *tri_verts, int n_triangles, int n_vertices, const char *caller)
{
  const int n_he = 3 * n_triangles;
  int *const scratch = (int *) MFTLMallocOrDie(sizeof(int) * ((size_t) n_vertices + (size_t) n_he), "mesh builder manifold check");
  int *const head = scratch;  // first directed edge leaving vertex v
  int *const next = scratch + n_vertices;  // next directed edge with the same origin
  for (int v = 0; v < n_vertices; ++v) head[v] = MFTL_ERROR;

  /* Directed edge 3t+k runs tri_verts[3t+k] -> tri_verts[3t+(k+1)%3]; its
   * triangle's third corner is tri_verts[3t+(k+2)%3]. */
  bool ok = MFTL_TRUE;
  for (int h = 0; h < n_he && ok; ++h)
  {
    const int t = h / 3, k = h % 3;
    const int u = tri_verts[h], v = tri_verts[3 * t + (k + 1) % 3], w = tri_verts[3 * t + (k + 2) % 3];

    for (int g = head[u]; g != MFTL_ERROR; g = next[g])
    {
      if (tri_verts[3 * (g / 3) + (g % 3 + 1) % 3] != v) continue;
      fprintf(stderr, "%s: triangles %d and %d both traverse edge %d -> %d (a non-manifold edge, a duplicate, or inconsistent orientation)\n", caller, g / 3, t,
              u, v);
      ok = MFTL_FALSE;
      break;
    }
    for (int g = (ok ? head[v] : MFTL_ERROR); g != MFTL_ERROR; g = next[g])
    {
      const int s = g / 3, j = g % 3;
      if (tri_verts[3 * s + (j + 1) % 3] != u || tri_verts[3 * s + (j + 2) % 3] != w) continue;
      fprintf(stderr, "%s: triangles %d and %d have the same three corners in opposite orders\n", caller, s, t);
      ok = MFTL_FALSE;
      break;
    }
    next[h] = head[u];
    head[u] = h;
  }

  free(scratch);
  return ok;
}

/**
 * @brief Builds triangles, half-edges, twins, the boundary flag and the derived
 *        arrays from a VALIDATED index list.
 *
 * @param object      Object with a filled vertex pool.
 * @param tri_verts   [3 * n_triangles] vertex indices, validated by the caller.
 * @param n_triangles Number of triangles, >= 1.
 */
static void build_topology(MFTL_Object *object, const int *tri_verts, int n_triangles)
{
  const int n_vertices = object->verts.n;
  const int rank = (object->domain != MFTL_NULL) ? object->domain->rank : 0;

  /* Grows the triangle pool (half-edges and derived arrays included). */
  MFTLObjectSetCounts(object, n_vertices, object->verts.n_local, n_triangles, n_triangles);

  /* Dense ids: triangles and half-edges restart at zero (the previous
   * generation is retired); vertex ids belong to the caller. */
  object->next_triangle_gid = n_triangles;
  object->next_halfedge_gid = 3 * n_triangles;

  MFTL_Vertex *const vr = object->verts.rec;
  MFTL_Triangle *const tr = object->tris.rec;
  MFTL_HalfEdge *const he = object->tris.he;

  /* Re-derived below; a stale value would index a retired half-edge. */
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

      const int h = 3 * t + k;  // implicit indexing: no he_idx, next_idx or triangle_idx to write
      he[h].global_id = h;
      he[h].vertex_idx = target;  // half-edges store their TARGET
      he[h].twin_idx = MFTL_ERROR;  // MFTLRebuildTwins links them below
      he[h].flags = 0;

      /* Any outgoing half-edge is valid (fan walks rewind to the start). */
      vr[origin].out_he_idx = h;
    }
  }

  MFTLRebuildTwins(object);

  /* The mesh is complete here, so "no twin" means "mesh boundary". */
  MFTLMarkMeshBoundary(object);

  MFTLObjectBuildDerived(object);
}

int MFTLBuildTopologyFromIndexList(MFTL_Object *object, const int *tri_verts, int n_triangles)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(tri_verts != MFTL_NULL);
  MFTL_ASSERT(object->verts.n >= 3 && "MFTLBuildTopologyFromIndexList: the vertex pool must be filled first");

  if (object == MFTL_NULL || tri_verts == MFTL_NULL) return MFTL_ERROR;
  if (n_triangles < 1 || object->verts.n < 3) return MFTL_ERROR;
  if (!index_list_is_valid(tri_verts, n_triangles, object->verts.n)) return MFTL_ERROR;
  if (!index_list_is_manifold(tri_verts, n_triangles, object->verts.n, "MFTLBuildTopologyFromIndexList")) return MFTL_ERROR;

  build_topology(object, tri_verts, n_triangles);
  return MFTL_SUCCESS;
}

int MFTLBuildIndexedMesh(MFTL_Object *object, const double *positions, int n_vertices, const int *tri_verts, int n_triangles)
{
  MFTL_ASSERT(object != MFTL_NULL);
  MFTL_ASSERT(object->tris.n == 0 && object->verts.n == 0 && "MFTLBuildIndexedMesh: object is not empty");

  if (object == MFTL_NULL || positions == MFTL_NULL || tri_verts == MFTL_NULL) return MFTL_ERROR;
  if (n_vertices < 3 || n_triangles < 1) return MFTL_ERROR;
  if (!index_list_is_valid(tri_verts, n_triangles, n_vertices)) return MFTL_ERROR;
  if (!index_list_is_manifold(tri_verts, n_triangles, n_vertices, "MFTLBuildIndexedMesh")) return MFTL_ERROR;

  const int rank = (object->domain != MFTL_NULL) ? object->domain->rank : 0;

  MFTLObjectSetCounts(object, n_vertices, n_vertices, 0, 0);

  /* The caller's layout is the pool's layout (xyz interleaved). */
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

  build_topology(object, tri_verts, n_triangles);

  /* A new mesh leaves with current geometry, so the first MFTLRedistribute()
   * routes by real centres. Without a domain nothing can route it yet. */
  if (object->domain != MFTL_NULL) MFTLComputeTriangleGeometry(object, object->tris.n);
  return MFTL_SUCCESS;
}
