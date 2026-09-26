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

#include "mftl_geometry.h"
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
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments -- including an
 *         index out of range or a triangle with a repeated corner, which are
 *         checked in Release too, before the object is modified.
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
 * @return MFTL_SUCCESS, or MFTL_ERROR on invalid arguments -- including an
 *         index out of range or a triangle with a repeated corner, which are
 *         checked in Release too, before the object is modified.
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
 * @note Collective. Call it after generation and BEFORE the first
 *       redistribution, while every rank still holds an object for every body in
 *       the same order (MFTLEnsembleAddObject() order). An empty ensemble is a
 *       no-op, not an error, but every rank must still call.
 *
 * @param ens The ensemble. Every object's counters and `global_n_*` are settled.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble.
 */
int MFTLPublishGeneratedBodies(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * Marching cubes  (see src/mesh_creation/marching_cubes.c)
 *
 * ## CUBE CONVENTIONS
 *
 * Corner c of a cell is at offset (c & 1, (c >> 1) & 1, (c >> 2) & 1) from the
 * cell's lower node. Edge e runs along axis a = e / 4 from its lower corner,
 * whose offsets on the other two axes b = (a+1)%3 and c = (a+2)%3 are
 * (e & 1) and ((e >> 1) & 1). A case MASK has bit c set when corner c is
 * INSIDE (negative).
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Most triangles one cell can produce, centroid fallback included. */
#define MFTL_MC_MAX_CASE_TRIANGLES 12
/** @brief Most separate surface pieces one cell can hold. */
#define MFTL_MC_MAX_CASE_POLYGONS 4
/** @brief Vertex index 12 + p in a case triangle is the centroid of polygon p. */
#define MFTL_MC_CENTROID 12

/**
 * @struct MFTL_MarchingCubesCase
 * @brief The triangulation of one of the 256 sign patterns of a cell.
 *
 * Generated once, from ONE face rule, rather than typed in: see
 * src/mesh_creation/marching_cubes.c for the rule and for why it makes the
 * surface crack-free and manifold across every pair of neighbouring cells.
 */
typedef struct MFTL_MarchingCubesCase
{
  int n_tris;  ///< Triangles in this case.
  /** Corner vertices: a cube edge in [0, 12), or #MFTL_MC_CENTROID + p. Wound so
   *  the normal points from inside (negative) to outside (positive). */
  signed char tri[MFTL_MC_MAX_CASE_TRIANGLES][3];
  int n_polys;  ///< Separate surface pieces (closed edge loops) in the cell.
  int poly_len[MFTL_MC_MAX_CASE_POLYGONS];  ///< Vertices of each loop.
  signed char poly[MFTL_MC_MAX_CASE_POLYGONS][12];  ///< Cube edges of each loop, in winding order.
  int n_centroids;  ///< Loops that needed a centroid vertex. Zero for every case in practice; see the tests.
} MFTL_MarchingCubesCase;

/**
 * @brief The triangulation of case @p mask.
 * @param mask Bit c set when corner c is inside. 0..255.
 * @return The case; generated on first use and then read-only.
 */
const MFTL_MarchingCubesCase *MFTLMarchingCubesCase(int mask);

/**
 * @struct MFTL_LevelSetSurface
 * @brief An extracted iso-surface as a triangle soup with LATTICE identities.
 *
 * Every vertex carries the key of the lattice edge it lies on and every
 * triangle the key of the cell that produced it, both computed from WRAPPED
 * global lattice indices. Two extractions of overlapping cell ranges -- on two
 * ranks, say -- therefore name a shared vertex identically without comparing a
 * single coordinate, and produce it at bit-identical positions from
 * bit-identical node values.
 *
 * Keys: node `(i, j, k)` has linear index `(k * N_y + j) * N_x + i` over
 * `n_global`; a lattice-edge vertex is `3 * node + axis`; a cell is its lower
 * node's index; a centroid vertex is `-(16 * cell + polygon) - 2`, negative so
 * it can never collide with an edge key.
 */
typedef struct MFTL_LevelSetSurface
{
  double *pos;  ///< [3*n_verts] positions from WRAPPED lattice nodes: bit-identical on every rank; a seam-crossing triangle has corners a period apart.
  long long *vkey;  ///< [n_verts] lattice identity of each vertex.
  int n_verts;  ///< Vertices.
  int cap_verts;  ///< Allocated vertex slots.
  int *tri;  ///< [3*n_tris] vertex indices, normals pointing to phi > 0.
  long long *tcell;  ///< [n_tris] key of the cell each triangle came from.
  signed char *tslot;  ///< [n_tris] triangle's index within its cell's case, < 16.
  int n_tris;  ///< Triangles.
  int cap_tris;  ///< Allocated triangle slots.
} MFTL_LevelSetSurface;

/**
 * @brief Marching cubes over the cells `[cell_lo, cell_hi]` of a grid window.
 *
 * A cell is extracted iff all eight corners are inside the window and none is
 * #MFTL_SDF_UNKNOWN. Values within `snap * min(spacing)` of zero are pushed out
 * to that distance, sign preserved and zero counted as positive, so no vertex
 * lands on a lattice node and no triangle comes out with a zero-length edge.
 *
 * Vertices on a lattice edge are shared between the cells of one call (one
 * vertex per edge, whichever cell reaches it first). A range on a periodic axis
 * must not exceed one period, or a cell would be extracted twice.
 *
 * @param      grid    The window.
 * @param      cell_lo First cell per axis, as a GLOBAL (unwrapped) lower-node index.
 * @param      cell_hi Last cell per axis, inclusive.
 * @param      snap    Relative snap distance, e.g. 1e-2. 0 disables it (then a
 *                     node exactly at zero is still counted as positive).
 * @param[out] out     Appended to; zero-initialise before the first call.
 * @return The number of triangles appended.
 * @rank_local
 */
int MFTLExtractLevelSet(const MFTL_SdfGrid *grid, const int cell_lo[3], const int cell_hi[3], double snap, MFTL_LevelSetSurface *out);

/**
 * @brief Releases a surface's arrays and zeroes it.
 * @param s The surface.
 */
void MFTLLevelSetSurfaceFree(MFTL_LevelSetSurface *s);

/**
 * @brief Triangulates the zero level set of a whole grid into an EMPTY object.
 *
 * MFTLExtractLevelSet() over every cell of the window, positions wrapped into
 * the object's domain, then MFTLBuildIndexedMesh() -- so the result obeys every
 * convention that function establishes, including the mesh boundary: where the
 * level set leaves the window the surface is open, and those half-edges carry
 * MFTL_FLAG_BOUNDARY. Outward normals point to phi > 0.
 *
 * A level set that does not cross the window leaves the object empty (with its
 * derived arrays marked current) and returns success.
 *
 * @param obj  An EMPTY object linked to a domain.
 * @param grid The values. On a periodic axis a window of exactly one period
 *             closes the surface across the seam.
 * @param snap Relative snap distance; see MFTLExtractLevelSet().
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid input.
 * @rank_local No MPI: the caller owns the global bookkeeping, exactly as for
 *             MFTLBuildIndexedMesh().
 */
int MFTLBuildMeshFromLevelSet(MFTL_Object *obj, const MFTL_SdfGrid *grid, double snap);

#endif  // MFTL_MESHGEN_H
