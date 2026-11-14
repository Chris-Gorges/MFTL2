/**
 * @file mftl_meshgen.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_meshgen
 *
 * @brief Mesh construction: indexed triangle lists, icospheres, and marching
 *        cubes on a (distributed, periodic) lattice.
 *
 * MFTLBuildIndexedMesh() / MFTLBuildTopologyFromIndexList() are the single place
 * raw connectivity becomes an MFTL mesh. Every builder leaves the object with
 * twins linked, #MFTL_FLAG_BOUNDARY recorded (while the mesh is complete) and
 * derived connectivity built. MFTLBuildIndexedMesh(), and every generator built
 * on it, also computes the triangle geometry (area, normal, centre), so a new
 * mesh can go straight into MFTLRedistribute(); MFTLBuildTopologyFromIndexList()
 * does not, because its caller still owns the positions.
 *
 * Sphere generation builds the whole body on rank 0 (other ranks get an empty
 * object); MFTLRedistribute() spreads it. The collective form
 * MFTLGenerateSphere() costs two collectives per body; the split form
 * MFTLGenerateSphereLocal() + MFTLPublishGeneratedBodies() costs two for the
 * whole ensemble. Theory: src/mesh_creation/README.md.
 */

#ifndef MFTL_MESHGEN_H
#define MFTL_MESHGEN_H

#include "mftl_geometry.h"
#include "mftl_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Builds a half-edge mesh from positions and an indexed triangle list.
 *
 * Establishes the conventions every module relies on: half-edge 3t+k starts at
 * `tri_verts[3t+k]` and ends at `tri_verts[3t+(k+1)%3]` (so slot k of
 * `tris.verts` is the origin of half-edge 3t+k); global ids are
 * `[0, n)` per class with half-edge id 3t+k, and the `next_*_gid` counters end
 * one past the last id; every entity is owned by the calling rank. The caller
 * owns the winding: the normal \f$(\vect{p}_1-\vect{p}_0)\times(\vect{p}_2-\vect{p}_0)\f$
 * points to the side the triangles are counter-clockwise from.
 *
 * Triangle geometry is current on return (when the object has a domain).
 *
 * The list must describe an ORIENTED edge-manifold: three distinct, in-range
 * corners per triangle, no directed edge used twice (that would be a third
 * triangle on an edge, a neighbour wound the other way, or a duplicate), and no
 * triangle together with its reversal. Anything else is refused with a
 * diagnostic on stderr before the object is touched -- a half-edge mesh cannot
 * represent it, and the surplus half-edges would be flagged as surface boundary
 * for good. Open rims and bowtie vertices are accepted.
 *
 * @param object      Target object; must be EMPTY and linked to a domain.
 * @param positions   [3 * n_vertices] interleaved xyz.
 * @param n_vertices  Number of vertices, >= 3.
 * @param tri_verts   [3 * n_triangles] vertex indices.
 * @param n_triangles Number of triangles, >= 1.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments (including an index
 *         out of range, a triangle with a repeated corner, or a list that is not
 *         an oriented edge-manifold; checked in Release before anything is
 *         modified).
 *
 * @rank_local The collective bookkeeping belongs to the caller.
 */
int MFTLBuildIndexedMesh(MFTL_Object *object, const double *positions, int n_vertices, const int *tri_verts, int n_triangles);

/**
 * @brief The topology half of MFTLBuildIndexedMesh(), for a vertex pool that is
 *        already filled (positions, global ids, owners).
 *
 * Writes triangles and half-edges, re-derives every `out_he_idx`, links twins,
 * marks the boundary and builds the derived arrays. Triangle and half-edge ids
 * restart at zero (the previous generation is retired); vertex ids are
 * untouched. Used by subdivision (which appends midpoints to the pool) and by the
 * spatial halo's patch builder.
 *
 * @param object      Object with a filled vertex pool (>= 3 vertices) and a domain.
 * @param tri_verts   [3 * n_triangles] vertex indices, counter-clockwise.
 * @param n_triangles Number of triangles, >= 1.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments, including a list
 *         that is not an oriented edge-manifold (the rules of
 *         MFTLBuildIndexedMesh(); checked in Release before anything is
 *         modified).
 *
 * @rank_local
 */
int MFTLBuildTopologyFromIndexList(MFTL_Object *object, const int *tri_verts, int n_triangles);

/**
 * @brief Generates an icosphere: a regular icosahedron, @p refinements rounds of
 *        1-to-4 midpoint subdivision, then one central projection onto the
 *        sphere of radius @p radius about @p center.
 *
 * On rank 0: \f$V = 10\cdot4^r+2\f$ vertices, \f$F = 20\cdot4^r\f$ triangles;
 * other ranks get an empty object. Face normals point OUTWARD at every level (the
 * face table is counter-clockwise seen from outside and subdivision preserves the
 * parent's rotation), which fixes the sign of the enclosed volume, of surface
 * tension and of the indicator function. Mean-ratio quality is at least 0.974
 * (mean 0.997) at every level; triangle areas vary by up to about a factor 2.
 * Positions are wrapped into a periodic domain, and triangle geometry is current
 * on return. The result is in subdivision order; MFTLObjectReorderMorton() gives
 * spatial locality.
 *
 * @param object      The object to generate into; must already be in an ensemble.
 * @param center      Sphere centre.
 * @param radius      Sphere radius, >= 0.
 * @param refinements Subdivision levels, >= 0 (0: the bare icosahedron).
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid arguments.
 *
 * @collective One broadcast (id counters) and one reduction (global counts).
 */
int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements);

/**
 * @brief The local half of MFTLGenerateSphere(): builds the mesh on rank 0 and
 *        makes no MPI call. Id counters and `global_n_*` are settled later, for
 *        all bodies at once, by MFTLPublishGeneratedBodies():
 *
 * @code
 * for (int b = 0; b < n_bodies; ++b) MFTLGenerateSphereLocal(objs[b], c[b], r, refine);
 * MFTLPublishGeneratedBodies(ens);
 * @endcode
 *
 * @param object      The object to generate into; must already be in an ensemble.
 * @param center      Sphere centre.
 * @param radius      Sphere radius, >= 0.
 * @param refinements Subdivision levels, >= 0.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on failure.
 *
 * @rank_local Every rank must still call it (non-root ranks mark their empty
 *             object's derived arrays valid).
 */
int MFTLGenerateSphereLocal(MFTL_Object *object, double center[3], double radius, int refinements);

/**
 * @brief The collective half: broadcasts every object's id counters from rank 0
 *        and reduces its owned counts into `global_n_*` -- two collectives for
 *        the whole ensemble.
 *
 * Call after generation and BEFORE the first redistribution, while every rank
 * holds an object for every body in the same order.
 *
 * @param ens The ensemble.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL ensemble.
 *
 * @collective An empty ensemble is a no-op, but every rank must still call.
 */
int MFTLPublishGeneratedBodies(MFTL_Ensemble *ens);

/* ═══════════════════════════════════════════════════════════════════════════
 * Marching cubes  (src/mesh_creation/marching_cubes.c)
 *
 * Cube conventions: corner c of a cell is at offset
 * (c & 1, (c >> 1) & 1, (c >> 2) & 1) from the cell's lower node. Edge e runs
 * along axis a = e / 4 from its lower corner, whose offsets on the other axes
 * b = (a+1)%3 and c = (a+2)%3 are (e & 1) and ((e >> 1) & 1). A case MASK has
 * bit c set when corner c is INSIDE (value < 0).
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
 * Generated from one face rule rather than tabulated, which makes the surface
 * crack-free and consistently oriented across neighbouring cells by
 * construction (see src/mesh_creation/README.md).
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
 * Every vertex carries the key of the lattice edge it lies on, every triangle
 * the key of the cell that produced it, both from WRAPPED global lattice
 * indices. Two extractions of overlapping cell ranges (on two ranks) therefore
 * name a shared vertex identically and place it at bit-identical coordinates
 * without comparing positions.
 *
 * Keys: node (i, j, k) has index `(k * N_y + j) * N_x + i` over `n_global`; an
 * edge vertex is `3 * node + axis`; a cell is its lower node's index; a centroid
 * vertex is `-(16 * cell + polygon) - 2` (negative, disjoint from edge keys).
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
 * A cell is extracted iff its eight corners are in the window and none is
 * #MFTL_SDF_UNKNOWN. A vertex lies on each crossed lattice edge at the linear
 * interpolant \f$t = \phi_0/(\phi_0-\phi_1)\f$ -- or, when the grid carries edge
 * crossings and this edge's slot holds exactly one (MFTLRasterizeEdgeCrossings()),
 * at that crossing, clamped to `[snap, 1 - snap]` of the edge. Values closer to
 * zero than `snap * min(spacing)` are pushed out to that distance (sign
 * preserved, zero counted positive), so no vertex lands on a lattice node. Vertices on one
 * lattice edge are shared between the cells of one call. On a periodic axis the
 * range must not exceed one period.
 *
 * @param      grid    The window.
 * @param      cell_lo First cell per axis, as a GLOBAL (unwrapped) lower-node index.
 * @param      cell_hi Last cell per axis, inclusive.
 * @param      snap    Relative snap distance (e.g. 1e-2); 0 disables snapping (a
 *                     node exactly at zero then still counts as positive).
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
 * MFTLExtractLevelSet() over every cell of the window, positions wrapped into the
 * object's domain, then MFTLBuildIndexedMesh(). Where the level set leaves the
 * window the surface is open and its rim carries #MFTL_FLAG_BOUNDARY; on a
 * periodic axis a window of exactly one period closes it across the seam.
 * Normals point to \f$\phi > 0\f$. A level set that does not cross the window
 * leaves the object empty (derived arrays marked current).
 *
 * @param obj  An EMPTY object linked to a domain.
 * @param grid The values.
 * @param snap Relative snap distance; see MFTLExtractLevelSet().
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on invalid input.
 * @rank_local
 */
int MFTLBuildMeshFromLevelSet(MFTL_Object *obj, const MFTL_SdfGrid *grid, double snap);

#ifdef __cplusplus
}
#endif

#endif  // MFTL_MESHGEN_H
