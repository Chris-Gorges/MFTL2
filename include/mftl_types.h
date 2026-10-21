/**
 * @file mftl_types.h
 * @author Christian Gorges
 * @date 2026-08-14
 * @ingroup mftl_types
 *
 * @brief Core data structures of MFTL -- the single place every struct is defined.
 *
 * Layer 1: types only, no behaviour, so module headers never include each other.
 *
 * ## Layout: topology in AoS, geometry in SoA, one allocation per pool
 *
 * - **Topology and identity -> Array of Structures.** #MFTL_Vertex (16 B),
 *   #MFTL_Triangle (12 B), #MFTL_HalfEdge (16 B). A remesher edits one entity at
 *   a time and touches most of its record when it does.
 * - **Geometry -> Structure of Arrays.** Positions, normals, areas, centroids:
 *   kernels touch one component of every entity, every timestep.
 *
 * Both live in ONE allocation per entity class, cut into cache-line-aligned
 * sub-arrays (mftl_memory.h, src/object/object.c):
 *
 *     MFTL_VertexPool   = one block -> rec[] pos[] normal[]
 *     MFTL_TrianglePool = one block -> rec[] he[] area[] normal[] center[] verts[] neighbors[]
 *
 * so a pool has exactly one `capacity`, and "the AoS grew but an SoA array did
 * not" cannot be expressed.
 *
 * ## Three invariants every module may assume
 *
 * 1. **Half-edges are triangle-indexed.** Half-edge \f$3t+k\f$ belongs to
 *    triangle \f$t\f$ and \f$\mathrm{next}(3t+k) = 3t+(k+1) \bmod 3\f$, so
 *    `next`, `prev` and `face` are arithmetic (mftl_topology.h), and the
 *    half-edge array lives inside the triangle pool at 3x its capacity. Anything
 *    that creates a triangle creates its half-edges at 3t, 3t+1, 3t+2.
 * 2. **Ownership is a prefix.** Entities `[0, n_local)` are owned by this rank,
 *    `[n_local, n)` are ghosts. There is no `is_ghost` flag; `owner_proc` only
 *    answers "whose is it" for a ghost.
 * 3. **Positions live in `verts.pos` and only there.** Spatial (Morton) ordering
 *    is applied to the pool itself (MFTLObjectReorderMorton()), never stored as
 *    a permutation, so there is one index space and one copy of each coordinate.
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
 * Its position and normal live in the SoA arrays of #MFTL_VertexPool at the same
 * index. 16 bytes: four records per cache line.
 */
typedef struct MFTL_Vertex
{
  int out_he_idx;  ///< Index of ONE outgoing half-edge, MFTL_ERROR if isolated. The 1-ring walk starts here.
  int global_id;  ///< Unique across ranks; stable under redistribution and reordering, renumbered by remeshing.
  int owner_proc;  ///< Owning rank; equals this rank exactly for indices < n_local.
  int flags;  ///< Bit flags, see MFTL_FLAG_* in mftl_topology.h.
} MFTL_Vertex;

/**
 * @struct MFTL_Triangle
 * @brief Identity of one triangle. Its topology AND its geometry are implicit.
 *
 * Its half-edges are 3t, 3t+1, 3t+2 (invariant 1) and its area, normal and
 * centroid are SoA arrays of #MFTL_TrianglePool, so what remains is 12 bytes of
 * identity: a triangle is a name for a cycle of three half-edges.
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
 * Only two topological fields are stored: `vertex_idx` (the vertex the
 * half-edge ENDS at) and `twin_idx`. `next`, `prev` and the triangle are
 * arithmetic on the index, and \f$\mathrm{origin}(h) =
 * \mathrm{target}(\mathrm{prev}(h))\f$ (mftl_topology.h). 16 bytes, so a
 * triangle's three half-edges span one or two cache lines.
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

/** @brief Entity class of an attached field. */
typedef enum
{
  MFTL_FIELD_VERTEX = 0,  ///< One value per vertex.
  MFTL_FIELD_TRIANGLE = 1  ///< One value per triangle.
} MFTL_FieldEntity;

/** @brief Lifetime and remeshing policy of an attached double field. */
typedef enum
{
  MFTL_FIELD_RECOMPUTED = 0,  ///< Storage only; recompute after mesh or physical-input changes.
  MFTL_FIELD_INTERPOLATED = 1,  ///< Persistent samples, transferred without an integral constraint.
  MFTL_FIELD_CONSERVATIVE_DENSITY = 2  ///< Non-negative triangle density; remeshing preserves area-weighted amount.
} MFTL_FieldMode;

/** @brief One ensemble-owned field descriptor. Treat as read-only after registration. */
typedef struct
{
  char *name;  ///< Owned, unique, NUL-terminated name.
  MFTL_FieldEntity entity;  ///< Vertex or triangle pool.
  MFTL_FieldMode mode;  ///< Lifetime and remeshing transfer policy.
  int ncomp;  ///< Positive number of interleaved components.
} MFTL_FieldDescriptor;

/** @brief Run-wide registry, frozen when the first object is attached. */
typedef struct
{
  MFTL_FieldDescriptor *entries;  ///< [n] descriptors in registration order.
  int n;  ///< Number of registered fields.
  int stride[2];  ///< Persistent doubles per vertex / triangle on the wire.
  bool locked;  ///< Registration is no longer permitted.
} MFTL_FieldRegistry;

/**
 * @struct MFTL_VertexPool
 * @brief Every vertex of one object: AoS records plus the per-vertex SoA arrays.
 *
 * All pointers address sub-arrays of the one allocation `block`, all sized for
 * `capacity` vertices. Only src/object/object.c may assign `capacity`, `block`
 * or any pointer. `rec` and `pos` are authoritative; `normal` is computed
 * (MFTLComputeVertexNormals()) but is preserved by resizes and reorders.
 */
typedef struct MFTL_VertexPool
{
  MFTL_Vertex *rec;  ///< [capacity]      identity + topology, AoS.

  double *pos;  ///< [3*capacity]    AUTHORITATIVE position, xyz interleaved.
  double *normal;  ///< [3*capacity]    unit vertex normal, xyz interleaved; valid only after MFTLComputeVertexNormals().

  int n;  ///< Active vertices (owned + ghost).
  int n_local;  ///< Owned vertices; they occupy [0, n_local).
  int capacity;  ///< Entities every array above is sized for. One number, one writer.

  void *block;  ///< The single allocation. Freed with free().
} MFTL_VertexPool;

/**
 * @struct MFTL_TrianglePool
 * @brief Every triangle AND every half-edge of one object.
 *
 * A triangle has exactly three half-edges at 3t..3t+2, so they share the
 * triangle block (at 3x its capacity) instead of having a capacity of their own.
 *
 * `verts` and `neighbors` are derived, compact connectivity for hot loops, valid
 * only while #MFTL_DERIVED_TOPOLOGY is set (MFTLObjectBuildDerived()). They are
 * redundant with the half-edge records -- `verts[3t+k] = he[3t+(k+2)%3].vertex_idx`
 * and `neighbors[3t+k] = he[3t+k].twin_idx / 3` -- and are stored because a
 * connectivity-only loop then streams 12 B per triangle instead of 48 B.
 *
 * `area`, `normal` and `center` are written by MFTLComputeTriangleGeometry() /
 * MFTLEnsembleUpdate(); a redistribution leaves them unspecified
 * (#MFTL_DERIVED_GEOMETRY).
 */
typedef struct MFTL_TrianglePool
{
  MFTL_Triangle *rec;  ///< [capacity]      identity, AoS.
  MFTL_HalfEdge *he;  ///< [3*capacity]    topology, AoS. Half-edge 3t+k belongs to triangle t.

  /* --- Geometry: written by the per-timestep geometry pass ------------------ */
  double *area;  ///< [capacity]      triangle area.
  double *normal;  ///< [3*capacity]    unit normal, xyz interleaved.
  double *center;  ///< [3*capacity]    centroid, xyz interleaved, WRAPPED into the domain.

  /* --- Derived connectivity: rebuilt, never authoritative ------------------- */
  int *verts;  ///< [3*capacity]    vertex pool index; slot k is the ORIGIN of half-edge 3t+k.
  int *neighbors;  ///< [3*capacity]    triangle across half-edge 3t+k, MFTL_ERROR at a boundary.

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
 * Every field is a value for the WHOLE body: min/max are reduced over the ranks
 * holding it and averages are divided by global counts over owned entities
 * (never an average of per-rank averages). The reduction folds contributions in
 * a fixed rank order, so every holder gets bit-identical values.
 */
typedef struct MFTL_ObjectQualityStatistics
{
  double min_triangle_area;  ///< Smallest triangle area anywhere in the mesh.
  double max_triangle_area;  ///< Largest triangle area anywhere in the mesh.
  double avg_triangle_area;  ///< Total area divided by the GLOBAL triangle count.

  /**
   * Worst mean-ratio quality in the mesh, \f$q = 4\sqrt{3}A/(a^2+b^2+c^2)\f$
   * (#MFTL_FOUR_SQRT3): 1 for an equilateral triangle, \f$\to 0\f$ as one
   * degenerates, never above 1. Despite the name it is a quality, not an aspect
   * ratio: larger is better.
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

/**
 * @brief `derived` bit: `tris.area/normal/center` were computed over EVERY
 *        triangle, ghosts included, after the last change the library made to
 *        their inputs.
 *
 * Set by MFTLComputeTriangleGeometry() over `tris.n` triangles. Cleared by any
 * change of the entity set (MFTLObjectSetCounts() clears both bits), by the
 * remeshing operators, by a redistribution's rebuild and by
 * MFTLUpdateGhostPositions(); kept by MFTLObjectBuildDerived() and
 * MFTLObjectReorderMorton(), which move no vertex. Moving owned vertices through `verts.pos` does NOT clear it: the
 * library cannot see that. So a clear bit means "not current", and a set bit
 * means "current unless the caller moved vertices since". MFTLRedistribute()
 * recomputes the geometry of a body whose bit is clear before routing it.
 */
#define MFTL_DERIVED_GEOMETRY 0x2u

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

  /* Global id counters. Ids are handed out by post-incrementing them, and are
   * kept DENSE: after generation and after every remeshing pass (which
   * renumbers) each equals the global entity count of its class. Density
   * matters because src/communication/ sizes its arrival lookup tables from
   * these counters. */
  int next_vertex_gid;  ///< Next unused vertex global id. Equals the global vertex count when dense.
  int next_triangle_gid;  ///< Next unused triangle global id.
  int next_halfedge_gid;  ///< Next unused half-edge global id; always 3 x next_triangle_gid, since the id of half-edge k of triangle g is 3g+k.

  int global_n_vertices;  ///< Owned vertices summed over all ranks.
  int global_n_triangles;  ///< Owned triangles summed over all ranks.

  int global_id;  ///< Body id, dense in [0, n_bodies_global), assigned by MFTLEnsembleAddObject(). The one handle that survives a redistribution.

  /**
   * Something other than the owner map changed this mesh (entity counts, global
   * ids or flags), so the current partition no longer describes it and the next
   * redistribution must rebuild the body even if no entity changed owner.
   *
   * Set by MFTLObjectMarkPartitionDirty() -- and by MFTLObjectSetCounts(), the
   * choke point for every creation and destruction -- and cleared by
   * redistribution after a rebuild. Conservative by design: a spurious `true`
   * costs one rebuild, a missing one silently corrupts ghost ids.
   */
  bool partition_dirty;

  /**
   * The body has been rebuilt at least once since it last changed, so its pool
   * layout is at a fixed point and the next redistribution may skip it.
   *
   * Separate from `partition_dirty` because a rebuild is not idempotent: an
   * owned vertex whose incident triangles are all remote is dropped and
   * re-received, landing at the end of the pool. A body must therefore be clean
   * TWICE before a skipped pool equals a rebuilt one slot for slot
   * (`test_skipping_reproduces_the_full_rebuild`). Kept per body on the object,
   * so adding or removing a body never un-settles the others.
   */
  bool partition_settled;

  /**
   * Surface generation: a stamp from one process-wide counter, renewed by every
   * library call that can change this mesh's positions, topology, entity set or
   * pool order (MFTLObjectMarkChanged()). Something that cached a description of
   * the mesh -- a coupling binding's marker list and rows -- compares stamps to
   * learn that it is stale; a new object never repeats an old stamp, so a body
   * destroyed and recreated by a redistribution cannot pass for its predecessor.
   *
   * A caller writing `verts.pos` itself owes MFTLObjectMarkChanged(); a
   * redistribution that SKIPS an unchanged body leaves its stamp alone, because
   * its pool is then bit-identical.
   */
  uint64_t generation;

  struct MFTL_Domain *domain;  ///< Borrowed from the ensemble; MFTL_NULL until added.

  const MFTL_FieldRegistry *fields;  ///< Borrowed nonempty registry; NULL when no fields are attached.
  double **field_data;  ///< [fields->n] pointers into the pools; written only by object.c.

  MFTL_ObjectQualityStatistics quality;  ///< By value: nine doubles do not need their own allocation.
} MFTL_Object;

//------------------------------------------------------------------------------
// The Domain (the "world" and the MPI topology)
//------------------------------------------------------------------------------

/**
 * @struct MFTL_Domain
 * @brief The computational environment shared by every object of an ensemble.
 *
 * The MPI communicators, the Cartesian process grid, the physical bounds and the
 * periodicity, stored once; every object holds a borrowed pointer. Read-only
 * after MFTLEnsembleCreate().
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

  /**
   * `[26][3]`, parallel to `neighbors`: the cell offset of each neighbour from
   * this rank, components in {-1, 0, +1}, so "are two cells 26-neighbours" is a
   * componentwise test. On thin periodic grids several stencil offsets map to
   * the same rank (e.g. cuts = {2,1,1}); the FIRST offset found is recorded.
   */
  signed char neighbor_offset[26][3];

  // --- Domain Decomposition ---
  int cuts[3];  ///< Process grid {nx, ny, nz}; the product is nproc.

  // --- Physical Bounds ---
  double global_min[3];  ///< Lower corner of the domain box, per axis.
  double global_max[3];  ///< Upper corner of the domain box, per axis.
  double domain_size[3];  ///< global_max - global_min.
  double inv_domain_size[3];  ///< 1 / domain_size (0 on a zero-length axis), so the minimum image is a multiply-and-round.

  bool is_periodic[3];  ///< Per-axis periodicity.

  int *rank_map;  ///< [nproc] rank of Cartesian cell (i,j,k) at i + j*nx + k*nx*ny; a table lookup instead of MPI_Cart_rank.
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
   * The bodies THIS RANK holds entities of, sorted by ascending
   * #MFTL_Object::global_id; owned (MFTLEnsembleDestroy() destroys them). A
   * redistribution destroys bodies that left the rank and creates bodies that
   * arrived, so object pointers, list slots and `n_local_objects` are all
   * invalid afterwards; `global_id` + MFTLEnsembleFindLocalObject() is the
   * handle that survives.
   *
   * @warning Every rank must add the same bodies in the same order, so the same
   *          id names the same body everywhere (checked by a Debug checksum).
   */
  MFTL_Object **local_objects;
  int n_local_objects;  ///< Bodies THIS RANK holds. Never a global quantity.
  int capacity_objects;  ///< Slots allocated in `local_objects`.

  /**
   * Size of the body-id space: every `global_id` is in [0, n_bodies_global).
   * Identical on every rank and never decreasing -- a topology change raises it
   * for each new body and retires, never reuses, the ids of vanished ones, so it
   * may exceed the number of live bodies.
   *
   * This, not `n_local_objects`, is the count a collective may be guarded by:
   * an early return on a local count is a deadlock.
   */
  int n_bodies_global;

  int total_global_triangles;  ///< Owned triangles summed over all bodies and ranks (MFTLEnsembleUpdateGlobalCounts()).

  /**
   * Opaque per-call scratch of the communication module (rank-sized arrays,
   * payload buffers, rebuild scratch), created on the first MFTLRedistribute()
   * and freed by MFTLEnsembleDestroy(). Never mesh state; kept off
   * #MFTL_Domain so the domain stays read-only during a redistribution.
   */
  MFTL_FieldRegistry fields;  ///< Optional attached-field schema, owned by this ensemble.

  void *comm_workspace;
} MFTL_Ensemble;

#endif  // MFTL_TYPES_H
