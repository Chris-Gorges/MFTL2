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
 * same numbering scheme, and the global entity counts have been reduced by the
 * time the call returns. Spreading the mesh across ranks is redistribution's job.
 *
 * Each generator also comes in a LOCAL form (MFTLGenerateSphereLocal()) that
 * makes no MPI call, paired with MFTLPublishGeneratedBodies() which does the
 * collective half for every body at once. Building N bodies through the
 * collective form costs 2N collectives in strict sequence; through the split
 * form it costs 2. Use the split form for anything past a handful of bodies.
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

/**
 * @brief The LOCAL half of MFTLGenerateSphere(): builds the mesh, makes no MPI call.
 *
 * Identical to MFTLGenerateSphere() except that the ID counters are left as rank
 * 0 wrote them and `global_n_*` are not filled. Use it to build MANY bodies and
 * settle all of them with ONE MFTLPublishGeneratedBodies():
 *
 * @code
 * for (int b = 0; b < n_bodies; ++b) MFTLGenerateSphereLocal(objs[b], c[b], r, refine);
 * MFTLPublishGeneratedBodies(ens);
 * @endcode
 *
 * against the same loop over MFTLGenerateSphere(), which puts one MPI_Bcast and
 * one MPI_Allreduce per body in strict sequence on rank 0's critical path.
 *
 * @rank_local No MPI. Every rank must still call it, because rank 0 does the
 *             work and the others must reach the publication with the object in
 *             the state it expects (empty, derived arrays marked valid).
 *
 * @param object      The object to generate into. Must already be in an ensemble.
 * @param center      Sphere centre.
 * @param radius      Sphere radius, >= 0.
 * @param refinements Subdivision levels, >= 0.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on failure.
 */
int MFTLGenerateSphereLocal(MFTL_Object *object, double center[3], double radius, int refinements);

/**
 * @brief The COLLECTIVE half: settles ID counters and global counts for a whole
 *        ensemble in two messages rather than two per body.
 *
 * Broadcasts every object's three ID counters from rank 0 -- so a rank that
 * later receives or creates entities cannot collide with the generated IDs --
 * and reduces every object's owned entity counts into `global_n_vertices` and
 * `global_n_triangles`. Exactly what MFTLGenerateSphere() does for one body,
 * for the whole object list at once.
 *
 * @note Collective. Every rank must hold the same object list in the same order.
 *       An empty ensemble is a no-op, not an error, but every rank must still
 *       call: the two collectives happen regardless of what this rank holds.
 *
 * @param ens The ensemble. Every object's counters and `global_n_*` are settled.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble.
 */
int MFTLPublishGeneratedBodies(MFTL_Ensemble *ens);

#endif  // MFTL_MESHGEN_H
