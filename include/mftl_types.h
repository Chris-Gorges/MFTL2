/**
 * @file mftl_types.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_types
 *
 * @brief Core data structures of MFTL.
 *
 * The lowest layer that knows about MFTL entities. It defines the structs and
 * nothing else -- no behaviour, no module APIs -- which keeps the type
 * definitions in exactly one place and prevents module headers from having to
 * include each other.
 *
 * ## THE ONE IDEA THIS FILE EXISTS TO EXPRESS
 *
 * A mesh is described by two things that want opposite memory layouts, and MFTL
 * stores each in the layout it wants:
 *
 *   TOPOLOGY AND IDENTITY -> Array of Structures.
 *   Who is my twin, which vertex do I end at, what is my global id, am I on the
 *   boundary. A remesher edits ONE entity at a time and touches most of its
 *   record when it does; an edge collapse is a handful of random accesses, not a
 *   sweep. AoS is right for that, and these records are deliberately small
 *   (12-16 bytes) so one cache line carries four of them.
 *
 *   GEOMETRY -> Structure of Arrays.
 *   Positions, normals, areas, centroids, curvatures, and the derived per-
 *   triangle frames. These are read and written by kernels that touch ONE
 *   component of EVERY entity, every timestep. That is the access pattern AoS is
 *   worst at, and it is where all the time goes.
 *
 * The split is easy to arrive at incrementally and hard to arrive at safely. Bolt
 * the geometry arrays onto the object with their own capacity (`geom_cap_*`),
 * then bolt a family of "flattened" arrays next to those with a third capacity
 * (`flat_cap_*`) and a fourth staleness counter (`flat_n_*`), and the result is
 * three capacities for one set of entities, kept in step by a comment that says
 * every growth site must remember to call three functions.
 *
 * Here there is ONE capacity per entity class, because a pool is ONE allocation:
 *
 *     MFTL_VertexPool  = one block  -> rec[] pos[] normal[] area[] curvatures[]
 *     MFTL_TrianglePool= one block  -> rec[] he[] area[] normal[] center[]
 *                                      verts[] neighbors[] anchor[] edge[]
 *
 * "The AoS grew but the SoA did not" is not a bug that can be made here; it is
 * not expressible. See mftl_memory.h (MFTLLayoutAdd) and src/object/object.c.
 *
 * ## THREE INVARIANTS THE REST OF THE LIBRARY IS ALLOWED TO ASSUME
 *
 * 1. HALF-EDGES ARE TRIANGLE-INDEXED. Half-edge 3t+k belongs to triangle t, and
 *    next(3t+k) == 3t+(k+1)%3. Therefore `he_idx`, `next_idx` and `triangle_idx`
 *    are not stored -- they are arithmetic (mftl_topology.h). The half-edge
 *    array lives INSIDE the triangle pool at 3x its capacity, so there is no
 *    separate half-edge capacity to keep in sync either. Anything that creates
 *    triangles must create their three half-edges at 3t, 3t+1, 3t+2.
 *
 * 2. OWNERSHIP IS A PREFIX. Entities [0, n_local) are owned by this rank;
 *    [n_local, n) are ghosts. There is no `is_ghost` field: it would be a second
 *    copy of a fact the index already carries -- and an index test, a flag and an
 *    `owner_proc == rank` comparison are three competing notions of ownership
 *    that can disagree. `owner_proc` survives on the records, because for a
 *    GHOST the interesting question is not "is it mine" (the index answers that)
 *    but "whose is it" (only the field answers that).
 *
 * 3. POSITIONS LIVE IN THE SoA, AND ONLY THERE. `verts.pos[3*v+k]` is the
 *    authoritative position of vertex v. Storing it in the entity struct AND in a
 *    second, Morton-ordered `flat_vert_pos` array forces a second index space, a
 *    sync function, a staleness flag and an "unwrapped or not" mode flag onto
 *    every consumer, and every one of those four is silent when it is wrong.
 *    Here the Morton order is applied to the POOL (MFTLObjectReorderMorton), so
 *    there is exactly one index space and exactly one copy of every coordinate.
 */

#ifndef MFTL_TYPES_H
#define MFTL_TYPES_H

#include <mpi.h>  // For MPI types like MPI_Comm
#include <stdbool.h>  // For bool type (C99)
#include <stddef.h>  // For size_t
#include <stdint.h>  // For int64_t

#include "mftl_constants.h"

//------------------------------------------------------------------------------
// Forward Declarations
//------------------------------------------------------------------------------

struct MFTL_HalfEdge;
struct MFTL_Vertex;
struct MFTL_Triangle;
struct MFTL_Object;
struct MFTL_Domain;
struct MFTL_Ensemble;

//------------------------------------------------------------------------------
// Entity records (AoS): topology and identity only
//------------------------------------------------------------------------------

/**
 * @struct MFTL_Vertex
 * @brief Identity and topology of one vertex. Its POSITION is not here.
 *
 * Everything geometric about a vertex -- position, normal, Voronoi area, the two
 * curvatures -- lives in the SoA arrays of MFTL_VertexPool at the same index.
 * What stayed is what a remesher edits one entity at a time.
 *
 * 16 bytes exactly: four records per cache line, and no padding to explain.
 */
typedef struct MFTL_Vertex
{
  int out_he_idx;  ///< Index of ONE outgoing half-edge, MFTL_ERROR if isolated. The 1-ring walk starts here.
  int global_id;  ///< Unique across all ranks. Stable under redistribution and under MFTLObjectReorderMorton.
  int owner_proc;  ///< The rank that owns this vertex. Equals domain->rank exactly for indices < n_local.
  int flags;  ///< Bit flags, see MFTL_FLAG_* in mftl_topology.h.
} MFTL_Vertex;

/**
 * @struct MFTL_Triangle
 * @brief Identity of one triangle. Its topology AND its geometry are implicit.
 *
 * There is no `he_idx`: the three half-edges of triangle t are 3t, 3t+1, 3t+2 by
 * construction (invariant 1 at the top of this file). There is no area, normal
 * or centroid either -- those are SoA arrays on MFTL_TrianglePool.
 *
 * What is left is 12 bytes of pure identity. That is not an accident of
 * trimming: a triangle in a half-edge mesh genuinely IS just a name for a cycle
 * of three half-edges, and every attempt to store more of it here was storing
 * something derivable.
 */
typedef struct MFTL_Triangle
{
  int global_id;  ///< Unique across all ranks.
  int owner_proc;  ///< The rank that owns this triangle.
  int flags;  ///< Bit flags, see MFTL_FLAG_* in mftl_topology.h.
} MFTL_Triangle;

/**
 * @struct MFTL_HalfEdge
 * @brief One directed edge of one triangle -- the entire mesh topology.
 *
 * Each undirected edge of a closed mesh is two opposite half-edges; on an open
 * surface the boundary edges have no partner and carry `twin_idx == MFTL_ERROR`.
 *
 * Only TWO topological fields are stored:
 *
 *   vertex_idx  the vertex this half-edge ENDS at
 *   twin_idx    the oppositely oriented half-edge, or MFTL_ERROR
 *
 * `next_idx` and `triangle_idx` are arithmetic on the half-edge's own index
 * (mftl_topology.h), and so is the origin vertex: origin(h) == target(prev(h)).
 * There is no `boundary` bool either -- it was an exact mirror of
 * `twin_idx == MFTL_ERROR`, and two spellings of one fact is one too many.
 *
 * 16 bytes: four half-edges per cache line, so a triangle's three half-edges
 * span one line or two, never three.
 */
typedef struct MFTL_HalfEdge
{
  int vertex_idx;  ///< Vertex at the END (target) of this directed edge.
  int twin_idx;  ///< Oppositely oriented half-edge, MFTL_ERROR if none in THIS partition.
  int global_id;  ///< Unique across all ranks.
  int flags;  ///< Bit flags, see MFTL_FLAG_* in mftl_topology.h.
} MFTL_HalfEdge;

//------------------------------------------------------------------------------
// Pools: one allocation each, AoS and SoA cut from the same block
//------------------------------------------------------------------------------

/**
 * @struct MFTL_VertexPool
 * @brief Every vertex of one object: AoS records plus the per-vertex SoA arrays.
 *
 * All pointers address one block owned by `block`, cut into cache-line-aligned
 * sub-arrays. They are all sized for `capacity` entities, always, because they
 * are all reserved by the same call. Never assign to `capacity` or to any of the
 * pointers outside src/object/object.c.
 *
 * Which arrays are AUTHORITATIVE (nothing else holds the value) and which are
 * DERIVED (recomputable from the mesh) matters for MFTLObjectReorderMorton,
 * which must permute the former and may simply rebuild the latter:
 *
 *   authoritative : rec, pos
 *   cached        : normal, area, mean_curv, gauss_curv   (recomputed per step,
 *                   but permuted anyway -- a reorder must not silently discard
 *                   geometry the caller just computed)
 */
typedef struct MFTL_VertexPool
{
  MFTL_Vertex *rec;  ///< [capacity]      identity + topology, AoS.

  double *pos;  ///< [3*capacity]    AUTHORITATIVE position, xyz interleaved.
  double *normal;  ///< [3*capacity]    vertex normal, xyz interleaved.
  double *area;  ///< [capacity]      Voronoi area.
  double *mean_curv;  ///< [capacity]      mean curvature.
  double *gauss_curv;  ///< [capacity]      Gaussian curvature.

  int n;  ///< Active vertices (owned + ghost).
  int n_local;  ///< Owned vertices; they occupy [0, n_local).
  int capacity;  ///< Entities every array above is sized for. One number, one writer.

  void *block;  ///< The single allocation. Freed with free().
} MFTL_VertexPool;

/**
 * @struct MFTL_TrianglePool
 * @brief Every triangle AND every half-edge of one object.
 *
 * Half-edges live here rather than in a pool of their own because there is
 * nothing independent about them: a triangle has exactly three, and they are at
 * 3t..3t+2. Giving them their own capacity would recreate exactly the
 * "reallocated independently" problem this layout exists to remove.
 *
 * The four derived connectivity/frame arrays are the hot-loop representation.
 * They are rebuilt by MFTLObjectBuildDerived / MFTLObjectRefreshFrames and are
 * only meaningful while the corresponding bit of MFTL_Object.derived is set:
 *
 *   verts, neighbors   invalidated by any TOPOLOGY change
 *   anchor, edge       invalidated by topology AND by vertex motion
 *
 * On what they duplicate, stated plainly so nobody has to rediscover it:
 * `verts[3t+k]` is a rotation of `he[3t+k'].vertex_idx`, and `neighbors[3t+k]`
 * is `he[3t+k].twin_idx / 3`. Both are stored anyway because a kernel that wants
 * only connectivity streams 12 B per triangle out of them, against 48 B if it
 * reads the half-edge records -- a 4x traffic reduction on exactly the loops
 * that are memory bound. See src/object/derived.c for the full accounting.
 */
typedef struct MFTL_TrianglePool
{
  MFTL_Triangle *rec;  ///< [capacity]      identity, AoS.
  MFTL_HalfEdge *he;  ///< [3*capacity]    topology, AoS. Half-edge 3t+k belongs to triangle t.

  /* --- Geometry: authoritative storage, written by the per-timestep pass --- */
  double *area;  ///< [capacity]      triangle area.
  double *normal;  ///< [3*capacity]    unit normal, xyz interleaved.
  double *center;  ///< [3*capacity]    centroid, xyz interleaved, WRAPPED into the domain.

  /* --- Derived connectivity and frames: rebuilt, never authoritative --------- */
  int *verts;  ///< [3*capacity]    vertex pool index; slot k is the ORIGIN of half-edge 3t+k.
  int *neighbors;  ///< [3*capacity]    triangle across half-edge 3t+k, MFTL_ERROR at a boundary.
  double *anchor;  ///< [3*capacity]    position of corner 0, replicated per triangle.
  double *edge;  ///< [6*capacity]    {e1, e2} = {v1-v0, v2-v0}, ALWAYS seam-consistent.

  int n;  ///< Active triangles (owned + ghost). Half-edges number 3*n.
  int n_local;  ///< Owned triangles; they occupy [0, n_local).
  int capacity;  ///< Triangles every array above is sized for. `he` holds 3*capacity.

  void *block;  ///< The single allocation. Freed with free().
} MFTL_TrianglePool;

//------------------------------------------------------------------------------
// Quality statistics
//------------------------------------------------------------------------------

/**
 * @struct MFTL_ObjectQualityStatistics
 * @brief Global mesh-quality summary of one object, filled by MFTLEnsembleUpdate.
 *
 * Every field is a GLOBAL value: the min/max are reduced across all ranks and
 * the averages are taken against the global entity counts, never as an average
 * of per-rank averages.
 */
typedef struct MFTL_ObjectQualityStatistics
{
  double min_triangle_area;  ///< Smallest triangle area anywhere in the mesh.
  double max_triangle_area;  ///< Largest triangle area anywhere in the mesh.
  double avg_triangle_area;  ///< Total area divided by the GLOBAL triangle count.

  /**
   * Worst mean-ratio quality in the mesh, \f$q = 4\sqrt{3}A/(a^2+b^2+c^2)\f$.
   *
   * 1 for an equilateral triangle, tending to 0 as one degenerates. It is
   * bounded above by 1 for every non-degenerate triangle, which is what makes an
   * average above 1 a usable alarm: it was measured at 1.023 on a mesh whose
   * areas were being floored by a clamped normalisation.
   */
  double min_aspect_ratio;
  double max_aspect_ratio;  ///< Best mean-ratio quality in the mesh. Never exceeds 1.
  double avg_aspect_ratio;  ///< Summed quality divided by the GLOBAL triangle count.

  double min_edge_length_sq;  ///< Shortest edge, SQUARED. Squares throughout, so no sqrt in the reduction.
  double max_edge_length_sq;  ///< Longest edge, squared.
  double avg_edge_length_sq;  ///< Summed squared length divided by the global half-edge count.
} MFTL_ObjectQualityStatistics;

//------------------------------------------------------------------------------
// The Object (one mesh)
//------------------------------------------------------------------------------

/** @brief `derived` bit: tris.verts and tris.neighbors are current. */
#define MFTL_DERIVED_TOPOLOGY 0x1u
/** @brief `derived` bit: tris.anchor and tris.edge are current. */
#define MFTL_DERIVED_FRAMES 0x2u

/**
 * @struct MFTL_Object
 * @brief One mesh -- typically one droplet or one body.
 *
 * Owns its two pools and borrows the domain of the ensemble it was added to. It
 * holds no MPI state itself.
 */
typedef struct MFTL_Object
{
  MFTL_VertexPool verts;  ///< Every vertex: AoS records plus the per-vertex SoA arrays.
  MFTL_TrianglePool tris;  ///< Every triangle AND every half-edge.

  /** Which derived arrays are current; see MFTL_DERIVED_* above. */
  unsigned derived;

  /* Global ID counters. Handing out an ID always means post-incrementing one of
   * these, so after generation each equals the entity count of its class.
   *
   * Density is not cosmetic: src/communication/ sizes its arrival lookup tables
   * from these counters, so holes accumulated over a few hundred timesteps
   * become an unboundedly growing table for a mesh of constant size. That is why
   * MFTLRemeshRenumberGlobalIds() runs at the end of every remeshing pass. */
  int next_vertex_gid;  ///< Next unused vertex global id. Equals the global vertex count when dense.
  int next_triangle_gid;  ///< Next unused triangle global id.
  int next_halfedge_gid;  ///< Next unused half-edge global id. Always 3x next_triangle_gid.

  int global_n_vertices;  ///< Owned vertices summed over all ranks.
  int global_n_triangles;  ///< Owned triangles summed over all ranks.

  int global_id;  ///< Identifies this object across all ranks.

  struct MFTL_Domain *domain;  ///< Borrowed from the ensemble; MFTL_NULL until added.

  MFTL_ObjectQualityStatistics quality;  ///< By value: nine doubles do not need their own allocation.
} MFTL_Object;

//------------------------------------------------------------------------------
// The Domain (the "world" and the MPI topology)
//------------------------------------------------------------------------------

/**
 * @struct MFTL_Domain
 * @brief The computational environment shared by every object of an ensemble.
 *
 * Decouples the container from the contents: MPI communicators, the process
 * grid, the physical bounds and the periodicity live here once, and objects
 * point at it.
 */
typedef struct MFTL_Domain
{
  // --- MPI Topology ---
  MPI_Comm comm_world;  ///< The library's private duplicate of the user's communicator.
  MPI_Comm comm_cart;   ///< Cartesian topology for geometric domain decomposition.
  MPI_Comm comm_graph;  ///< Distributed graph over the (up to) 26 surrounding ranks.

  int rank;  ///< My rank in comm_world.
  int nproc;  ///< Size of comm_world.

  int num_neighbors;  ///< Degree of comm_graph.
  int *neighbors;  ///< [num_neighbors] neighbour ranks, in the order MPI reports them.

  // --- Domain Decomposition ---
  int cuts[3];  ///< Process grid {nx, ny, nz}; the product is nproc.

  // --- Physical Bounds ---
  double global_min[3];  ///< Lower corner of the domain box, per axis.
  double global_max[3];  ///< Upper corner of the domain box, per axis.
  double domain_size[3];  ///< global_max - global_min.
  double domain_half_size[3];  ///< Precomputed for periodicity checks.
  double inv_domain_size[3];  ///< 1 / domain_size, 0 on a zero-length axis. Minimum-image wrapping is a
  ///< multiply-and-round, so the reciprocal is precomputed once here.

  bool is_periodic[3];  ///< Per-axis periodicity.

  int *rank_map;  ///< [nproc] rank_map[i + j*nx + k*nx*ny] = rank. MPI_Cart_rank is too slow for a hot path.
  double grid_factors[3];  ///< cuts[i] / domain_size[i], so owner lookup is a multiply.
} MFTL_Domain;

//------------------------------------------------------------------------------
// The Ensemble (the library handle)
//------------------------------------------------------------------------------

/**
 * @struct MFTL_Ensemble
 * @brief Top-level handle: one domain plus the objects living in it.
 */
typedef struct MFTL_Ensemble
{
  MFTL_Domain domain;  ///< By value: there is exactly one, and every object borrows a pointer to it.

  /**
   * Owned; MFTLEnsembleDestroy() destroys each of them.
   *
   * @warning Bodies are identified ON THE WIRE by their INDEX in this list, not
   *          by `global_id`. Every rank must therefore hold the same object list
   *          in the same order. A Debug-only checksum reduction inside the
   *          redistribution reports a violation rather than silently merging
   *          entities into the wrong body.
   */
  MFTL_Object **objects;
  int num_objects;  ///< Objects currently in the list.
  int capacity_objects;  ///< Slots allocated in `objects`.

  int total_global_triangles;  ///< Sum over all objects of global_n_triangles.

  /**
   * Opaque scratch owned by the communication module, created on the first
   * MFTLRedistribute() and released by MFTLEnsembleDestroy().
   *
   * It holds only per-call working storage -- rank-sized arrays, per-destination
   * payload buffers, the rebuild scratch and its id tables -- never mesh state.
   * Keeping it here rather than on #MFTL_Domain is deliberate: the domain is
   * read-only for the duration of a redistribution and a good deal of reasoning
   * about who may touch what depends on that staying true.
   */
  void *comm_workspace;
} MFTL_Ensemble;

#endif  // MFTL_TYPES_H
