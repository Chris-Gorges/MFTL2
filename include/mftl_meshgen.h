/**
 * @file mftl_meshgen.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief Procedural mesh generators.
 *
 * All generators are collective and share one contract: geometry is built on
 * rank 0 only, the global ID counters are broadcast so every rank continues the
 * same numbering scheme, and MFTLObjectUpdateGlobalStats() has run by the time
 * the call returns. Spreading the mesh across ranks is redistribution's job.
 *
 * Every generator leaves the object with:
 *   - twins linked (MFTLRebuildTwins) and the mesh boundary recorded
 *     (MFTLMarkMeshBoundary), so an open surface is fully described;
 *   - the derived per-triangle arrays built (MFTLObjectBuildDerived), so a
 *     hot-loop iterator works immediately;
 *   - triangle area/normal/centre NOT computed. Call MFTLEnsembleUpdate() or
 *     MFTLComputeTriangleGeometry() before anything reads them.
 */

#ifndef MFTL_MESHGEN_H
#define MFTL_MESHGEN_H

#include "mftl_types.h"

/**
 * @brief Builds a half-edge mesh from an indexed triangle list.
 *
 * The single place raw geometry becomes a valid MFTL mesh, and therefore the
 * single place the conventions every other module depends on are established:
 *
 *   - half-edge 3t+k belongs to triangle t, starts at tri_verts[3t+k] and ends
 *     at tri_verts[3t+(k+1)%3]. This is the implicit-indexing invariant
 *     (mftl_topology.h) and the "slot k is the origin of half-edge k" convention
 *     that MFTLGetTriangleVertices and `tris.verts` both assume;
 *   - global IDs are a dense permutation of [0, n) per entity class, and each
 *     next_*_gid counter ends exactly one past the last ID handed out;
 *   - every entity is owned by the calling rank, so the owned prefix covers the
 *     whole pools.
 *
 * Triangles must be wound consistently by the caller -- this function cannot
 * know which side of the surface is "outside" and does not try to guess.
 *
 * @param object      Target object. Must be EMPTY and linked to a domain.
 * @param positions   [3 * n_vertices] interleaved xyz.
 * @param n_vertices  Number of vertices, >= 3.
 * @param tri_verts   [3 * n_triangles] vertex indices, counter-clockwise as seen
 *                    from the side the normal should point to.
 * @param n_triangles Number of triangles, >= 1.
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments.
 *
 * @note Local (serial) operation -- no MPI. The collective bookkeeping belongs
 *       to the generator that calls it.
 */
int MFTLBuildIndexedMesh(MFTL_Object *object, const double *positions, int n_vertices, const int *tri_verts, int n_triangles);

/**
 * @brief The topology half of MFTLBuildIndexedMesh, for callers that have
 *        already filled the vertex pool.
 *
 * Assigns triangles and half-edges from @p tri_verts, links twins, records the
 * mesh boundary and builds the derived arrays. The vertex pool must already hold
 * `object->verts.n >= 3` vertices with their positions, global IDs and owners
 * set; every `out_he_idx` is re-derived here, so a stale one is not a problem.
 *
 * It exists for two callers that would otherwise duplicate this loop:
 *
 *   - subdivision, which grows the vertex pool in place (old vertices keep their
 *     indices, midpoints are appended) and only needs the connectivity rebuilt;
 *   - a mesh reader that streams positions into the pool rather than assembling
 *     a contiguous input array first.
 *
 * Triangle and half-edge global IDs restart at zero: the previous generation of
 * both is retired wholesale. Vertex IDs are untouched.
 *
 * @param object      Object with a filled vertex pool and a valid domain link.
 * @param tri_verts   [3 * n_triangles] vertex indices, counter-clockwise.
 * @param n_triangles Number of triangles, >= 1.
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments.
 *
 * @note Local (serial) operation -- no MPI.
 */
int MFTLBuildTopologyFromIndexList(MFTL_Object *object, const int *tri_verts, int n_triangles);

/**
 * @brief Generates a triangulated sphere by recursive subdivision of an icosahedron.
 *
 * Rank 0 builds the mesh (icosahedron, then @p refinements rounds of 1-to-4
 * subdivision, then projection onto the sphere); every other rank ends up with
 * an empty object. The counts on rank 0 are:
 *
 *   vertices   = 10 * 4^refinements + 2
 *   triangles  = 20 * 4^refinements
 *   half-edges =  3 * triangles
 *
 * Face normals point OUTWARD at every refinement level. The icosahedron face
 * table is wound counter-clockwise as seen from outside, so cross(v1-v0, v2-v0)
 * is outward, and the 1-to-4 subdivision emits every sub-triangle in its
 * parent's rotational order. This is what makes the enclosed volume positive
 * under the divergence theorem and fixes the sign of surface tension and of the
 * indicator function.
 *
 * The result is in SUBDIVISION order, not Morton order -- deterministic and easy
 * to reason about. Call MFTLObjectReorderMorton() afterwards for spatial
 * locality; nothing in the library requires it, and everything benefits.
 *
 * @note Collective: the ID counters are broadcast and the global stats reduced.
 *
 * @param object The object to generate into. Must already be in an ensemble.
 * @param center Sphere centre.
 * @param radius Sphere radius, >= 0. Zero is a valid degenerate sphere.
 * @param refinements Subdivision levels, >= 0. Zero yields the bare icosahedron.
 * @return MFTL_SUCCESS on success, MFTL_ERROR on failure.
 */
int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements);

#endif  // MFTL_MESHGEN_H
