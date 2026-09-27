/*
 * MFTL2 core — the public data model: vertices, triangles and half-edges in flat arrays per
 * object (MFTL_Object), the decomposition (MFTL_Domain) and the ensemble handle (MFTL_Ensemble),
 * plus object/ensemble lifecycle.
 *
 * Storage: an entity is OWNED when is_ghost == false (owner_proc == rank); the library
 * identifies owned entities by this flag. A multi-rank redistribution stores the owned entities
 * first and the ghost copies after them (see n_local_*). Entities are identified across ranks by
 * global_id; array indices are rank-local and change with every redistribution/compaction.
 * The khash maps translate global id -> local index.
 */

#ifndef MFTL_CORE_H
#define MFTL_CORE_H

#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "khash.h"

KHASH_MAP_INIT_INT(vertex_map, int)
KHASH_MAP_INIT_INT(halfedge_map, int)
KHASH_MAP_INIT_INT(triangle_map, int)
KHASH_MAP_INIT_INT(object_map, int)

struct MFTL_HalfEdge;
struct MFTL_Vertex;
struct MFTL_Triangle;
struct MFTL_Object;
struct MFTL_Domain;
struct MFTL_Ensemble;

/* Boundary-curve projector of a CONSTRAINED boundary tag (plans/16 §4.3). X_in is a point in
   the local frame of the operation (coordinates in the periodic image of the wrapped position
   of the operation's first vertex); X_out receives its projection onto the curve. Returns
   MFTL_SUCCESS, or MFTL_ERROR to refuse (the operation is then skipped). */
typedef int (*MFTL_BoundaryProjector)(int tag, const double X_in[3], double X_out[3], void *user);

typedef struct MFTL_BoundaryCurve
{
  int tag;
  int mode;  ///< MFTL_BCURVE_REPRESENTED / MFTL_BCURVE_CONSTRAINED
  MFTL_BoundaryProjector project;  ///< CONSTRAINED only
  void *user;
  int freeze;  ///< 1: boundary vertices of this tag are never moved by smoothing (PINNED supports)
} MFTL_BoundaryCurve;

//------------------------------------------------------------------------------
// Core Data Structures
//------------------------------------------------------------------------------

/**
 * @struct MFTL_Vertex
 * @brief A mesh vertex: wrapped position, periodic image counters, derived geometry,
 * ownership and one outgoing half-edge (entry point of the one-ring traversal).
 */
typedef struct MFTL_Vertex
{
  double pos[3];  ///< x, y, z coordinates of the vertex (WRAPPED into [min, max) on periodic axes).
  /* Periodic image counters (plans/16 §3.1). The continuous coordinate of the vertex is
     X = pos + img ⊙ L (img = 0 on non-periodic axes). Maintained by the displacement rule
     (§3.2) and re-gauged by MFTLCanonicaliseImages (§3.3). */
  int img[3];

  // --- Geometric Properties ---
  double normal[3];  ///< Vertex normal (MFTLComputeVertexNormals).
  double meancurvature;  ///< Mean curvature 2H (2/R on a sphere of radius R; MFTLComputeVertexCurvatures).
  double gaussiancurvature;  ///< Gaussian curvature in the MFTL1 convention: 2 x angle deficit / mixed area (2/R^2 on a sphere).
  double area;  ///< Mixed-Voronoi area associated with the vertex.
  double roughness_mean;  ///< Dihedral-angle roughness mean (owned vertices only).
  double roughness_variance;  ///< Dihedral-angle roughness sample variance (owned only).

  // --- Parallelism & Domain Information ---
  int owner_proc;  ///< Owner rank: the only rank whose copy is authoritative.
  bool is_ghost;  ///< True for a read-only copy of a vertex owned by another rank.
  int global_id;  ///< Global id, unique within the object across all ranks.

  // --- Topology ---
  int out_he_idx;  ///< Local index of an outgoing half-edge (the outgoing BOUNDARY one on a boundary vertex).

  // --- Application-Specific Data ---
  int flags;  ///< MFTL_FLAG_* / MFTL_VFLAG_* bits (remeshing marks, feature and curvature validity).

  /* Extrapolated curvature (2H, as meancurvature) of a boundary vertex from an open-stencil
     paraboloid fit; valid iff MFTL_VFLAG_CURV_EXTRAPOLATED (plans/16 §4.2). Rank-local, not
     packed (recomputed on demand). */
  double curv_extrap;

} MFTL_Vertex;

/**
 * @struct MFTL_Triangle
 * @brief A mesh triangle: one of its three half-edges, derived geometry, ownership, the opaque
 * user payload and the conserved surface scalar.
 *
 * A triangle is owned by the rank whose subdomain contains its centre (MFTLGetOwnerProc).
 */
typedef struct MFTL_Triangle
{
  // --- Topology ---
  int he_idx;  ///< Local index of one of the triangle's three half-edges.

  // --- Geometric Properties (MFTLComputeTriangleProperties) ---
  double normal[3];  ///< Unit normal; outward on a consistently (outward) wound closed body.
  double area;  ///< Triangle area.
  double center[3];  ///< Centroid (wrapped into the domain on periodic axes).

  // --- Parallelism & Domain Information ---
  int owner_proc;  ///< Owner rank (the rank whose subdomain contains the centre).
  bool is_ghost;  ///< True for a read-only copy of a triangle owned by another rank.
  int global_id;  ///< Global id, unique within the object across all ranks.

  // --- Application-Specific Data ---
  void *data;  ///< Opaque payload of MFTL_Object.data_size bytes (NULL when data_size == 0).
  int flags;  ///< MFTL_FLAG_* bits (remeshing marks).

  /* Conserved surface scalar (MFTL1 `scalarconcentration` parity, plans/15 item L7): a concentration
     per unit area whose MASS (surface_scalar * area) the library conserves. It travels with
     the triangle through every redistribution / ghost / pack operation, is remapped
     mass-conservingly by split / flip / collapse and rescaled (c *= A_old / A_new) by the
     TSUR3D / volume-conserving vertex moves, starts at 0 for generated / built triangles and
     is written/read by the native mesh I/O. Vertex moves made by the APPLICATION do not
     rescale it. */
  double surface_scalar;
} MFTL_Triangle;

/**
 * @struct MFTL_HalfEdge
 * @brief A directed edge of exactly one triangle.
 *
 * The three half-edges of a triangle form a cycle through next_idx in the triangle's winding
 * order. An interior edge consists of two opposite half-edges (twins); a physical boundary
 * edge has a single half-edge in the BOUNDARY state.
 */
typedef struct MFTL_HalfEdge
{
  // --- Topology ---
  int vertex_idx;  ///< Local index of the vertex at the END of this directed edge.
  int twin_idx;  ///< Local index of the opposite half-edge; MFTL_ERROR (-1) when it is not stored on this rank.
  int next_idx;  ///< Local index of the next half-edge of the same triangle (winding order).
  int triangle_idx;  ///< Local index of the triangle this half-edge belongs to.

  // --- Parallelism & Domain Information ---
  int global_id;  ///< Global id, unique within the object across all ranks.
  /* Twin state code (plans/16 §2.1; see MFTL_TWIN_* in mftl_constants.h):
     LOCAL     twin_idx >= 0 and twin_gid == half_edges[twin_idx].global_id;
     REMOTE    twin_idx == -1 and twin_gid >= 0: the gid of the live twin on another rank
               (an MPI rim of the ghost layer);
     BOUNDARY  twin_idx == -1 and twin_gid == MFTL_TWIN_NONE - tag: a genuine physical boundary.
     Read it only through the predicates of mftl_topology.h (MFTLHeIsLocal / IsRemote /
     IsBoundary / BoundaryTag / TwinGid / EdgeId). The undirected edge id is
     min(global_id, twin gid) for LOCAL/REMOTE and global_id for BOUNDARY. */
  int twin_gid;

  // --- Application-Specific Data ---
  int flags;  ///< MFTL_FLAG_* bits (remeshing marks).
  bool boundary;  ///< Cache of MFTLHeIsBoundary (BOUNDARY state); never an MPI-rim marker.
} MFTL_HalfEdge;

//------------------------------------------------------------------------------
// The Object (The Mesh)
//------------------------------------------------------------------------------
/**
 * @struct MFTL_Object
 * @brief One body (a triangulated surface, possibly several connected components): this
 * rank's part of it — owned entities and ghost copies (owned first after a multi-rank
 * redistribution) — plus id maps, the declared class and the periodic-winding cache.
 */
typedef struct MFTL_Object
{
  // --- Core Data Arrays (Pools) ---
  MFTL_Vertex *vertices;  ///< Vertices stored on this rank (owned, then ghosts).
  MFTL_Triangle *triangles;  ///< Triangles stored on this rank (owned, then ghosts).
  MFTL_HalfEdge *half_edges;  ///< Half-edges stored on this rank (those of owned triangles, then ghosts).

  int n_vertices;  ///< Number of stored vertices (owned + ghost).
  int n_triangles;  ///< Number of stored triangles (owned + ghost).
  int n_half_edges;  ///< Number of stored half-edges (owned + ghost).

  /* Owned-prefix counts: the first n_local_* entries of each array are the owned entities.
     Set only by a multi-rank redistribution, the builders (soups, MFTLBuildTopologyFromTriples),
     MFTLObjectAppendTriangleCopy and MFTLObjectClear; the generators, remeshing and a single-rank
     redistribution (a no-op) do not update them. Loop over the stored arrays and skip is_ghost
     entries unless the object has just been redistributed on more than one rank. */
  int n_local_triangles;  ///< Number of owned (non-ghost) triangles.
  int n_local_vertices;  ///< Same invariant for vertices[].
  int n_local_half_edges;  ///< Same invariant for half_edges[].

  int capacity_vertices;  ///< Allocated capacity for the vertices array.
  int capacity_triangles;  ///< Allocated capacity for the triangles array.
  int capacity_half_edges;  ///< Allocated capacity for the half_edges array.

  // next unused global ids (kept identical on all ranks; parallel remeshing uses disjoint
  // per-rank blocks and re-synchronises them)
  int next_vertex_gid;
  int next_triangle_gid;
  int next_halfedge_gid;

  int global_n_triangles;  // owned triangles summed over all ranks (MFTLObjectUpdateGlobalStats)
  int global_n_vertices;  // owned vertices summed over all ranks

  // --- global id -> local index maps ---
  khash_t(vertex_map) * vertex_lookup;
  khash_t(triangle_map) * triangle_lookup;
  khash_t(halfedge_map) * halfedge_lookup;

  // --- Flattened connectivity (MFTLFlattenConnectivity; NULL until built) ---
  int *flat_tri_verts;  ///< Size: 3 * n_triangles. Vert indices for each tri.
  int *flat_tri_neighbors;  ///< Size: 3 * n_triangles. Neighbor tri indices.
  int *vert_offsets;  ///< Size: n_vertices + 1. CSR offsets for V2T map.
  int *vert_tris;  ///< Size: 3 * n_triangles. CSR values for V2T map.

  int global_id;  ///< Object id (identical on all ranks).

  /* Size in bytes of the opaque per-triangle user payload (MFTL_Triangle.data).
   *
   * CONTRACT (mirrors MFTL1's Object->SizeOfData):
   * - The blob MUST be a flat POD: no embedded pointers — a pointer stored in
   *   the blob would only be a valid address on the rank that wrote it.
   * - data_size MUST be set collectively and identically on all ranks via
   *   MFTLInitializeObjectData() BEFORE any redistribution/ghosting. The
   *   transport layer ships data_size bytes per triangle as MPI_BYTE without
   *   any size negotiation; a mismatch across ranks SILENTLY CORRUPTS data.
   * - data_size == 0 (default) disables the payload entirely (fast path:
   *   every triangle's data pointer stays NULL). */
  size_t data_size;

  // domain of the ensemble the object belongs to (not owned; NULL when detached)
  struct MFTL_Domain *domain;

  /* The body is declared OPEN (MFTLObjectDeclareOpen, identical on every rank).
     Only a declared-open body may carry BOUNDARY half-edges; a closed body with an
     unmatched edge is an error, never silently opened (plans/16 §0, §2.2). */
  int declared_open;

  /* Cached per-axis wrap flags (plans/16 §3.3/§3.4), set by MFTLCanonicaliseImages:
     wrap[k] = 1 iff some component of the body winds along k (a cycle with non-zero lattice
     residual). wrap_valid = 0 until the first canonicalisation. */
  int wrap[3];
  int wrap_valid;
  /* winding lattice generators of the body (all components), from the last canonicalisation */
  int n_wind;
  int (*wind_gen)[3];
  /* Set when a canonicalisation finds different wrap flags than the
     previous valid state (a class transition); read and cleared by MFTLObjectTakeClassTransition. */
  int class_transition;

  /* MFTL_REMESH_MODE_DEFAULT / _GENERALIZED (plans/16 §4.4.4; identical on all ranks) */
  int remesh_mode;
  /* boundary-curve mode per boundary tag (plans/16 §4.3; identical on all ranks; tags without an
     entry are REPRESENTED) */
  int n_bcurve;
  struct MFTL_BoundaryCurve *bcurve;

} MFTL_Object;

//------------------------------------------------------------------------------
// The Domain (The "World" & MPI Topology)
//------------------------------------------------------------------------------
/**
 * @struct MFTL_Domain
 * @brief The decomposition shared by all objects of an ensemble: communicators, the
 * slab -> rank map, neighbour ranks, the global box and its periodicity.
 */
typedef struct MFTL_Domain
{
  // --- MPI Topology ---
  MPI_Comm comm_world;  ///< Duplicate of the communicator the ensemble was created on.
  MPI_Comm comm_cart;  ///< Cartesian communicator of the slab grid (periodic axes wrap).
  MPI_Comm comm_graph;  ///< Distributed-graph communicator over the neighbour ranks.

  int rank;  ///< This rank in comm_world.
  int nproc;  ///< Number of ranks.

  // de-duplicated ranks of the up to 26 neighbouring slabs (may include this rank on 1- or
  // 2-slab periodic axes)
  int num_neighbors;
  int *neighbors;

  int cuts[3];  ///< Number of slabs along x, y, z (product == nproc).

  double global_min[3];  ///< Lower corner of the global box.
  double global_max[3];  ///< Upper corner of the global box.
  double domain_size[3];  ///< Box size L = max - min (the period on periodic axes).
  double domain_half_size[3];  ///< L / 2 (minimum-image threshold).

  bool is_periodic[3];  ///< Periodicity per axis.

  int *rank_map;  // slab -> rank: rank_map[i + j*cuts[0] + k*cuts[0]*cuts[1]]
  double grid_factors[3];  // cuts[d] / domain_size[d] (uniform slab lookup)

  // --- Rectilinear decomposition ---
  // When axis_cuts[d] != NULL, slab i along axis d is [axis_cuts[d][i], axis_cuts[d][i+1])
  // (cuts[d] + 1 ascending coordinates; axis_cuts[d][0] / [cuts[d]] are the global box
  // faces) instead of the uniform split. rank_map may then be any bijection slab -> rank
  // (e.g. the PETSc DMDA x-fastest layout). NULL axes keep the uniform split.
  double *axis_cuts[3];

  // --- Parallel remeshing: x-only load-balanced ownership ---
  // When lb_nprocs > 0, MFTLGetOwnerProc ignores the Cartesian rank_map and
  // maps a position to the x-slab (slab index == owning rank) selected by the
  // sorted cut array lb_xcuts[0..lb_nprocs]. This implements MFTL1's x-only
  // 1-D ownership ranges (Ycut=Zcut=1) on top of MFTL2's cell ownership.
  // Slab boundaries are arbitrary x positions, so the load-balance optimizer
  // can place cuts at sub-cell resolution. Ghost safety: with an x-only
  // Cartesian grid (cuts[1]==cuts[2]==1) adjacent slabs are comm_graph
  // neighbours, and mesh edges are assumed not to span a whole slab width
  // (MFTL1's Sizethreshold guard). lb_nprocs == 0 disables the override.
  int lb_nprocs;  // number of active x-slabs; 0 = native cell ownership
  double lb_xmin;  // active window start (lb_xcuts[0])
  double lb_xmax;  // active window end   (lb_xcuts[lb_nprocs])
  double *lb_xcuts;  // lb_nprocs+1 sorted cut positions (NULL when disabled)

} MFTL_Domain;

//------------------------------------------------------------------------------
// The Ensemble (The Library Handle)
//------------------------------------------------------------------------------
/**
 * @struct MFTL_Ensemble
 * @brief The library handle: one domain and the objects (bodies) living in it. Every rank
 * holds the same object list in the same order.
 */
typedef struct MFTL_Ensemble
{
  MFTL_Domain domain;

  MFTL_Object **objects;
  int num_objects;
  int capacity_objects;

  int total_global_triangles;  // sum of the objects' global_n_triangles

  MFTL_EnsembleType type;
  MFTL_RemeshAlgorithm last_algorithm;  // algorithm used in the last remesh

} MFTL_Ensemble;

/**
 * @brief Allocates an empty object with object id global_id.
 *
 * The object has no entities and no domain until it is filled (generator, reader, builder)
 * and added to an ensemble (MFTLEnsembleAddObject).
 * @return The new object, or NULL on allocation failure.
 */
MFTL_Object *MFTLObjectCreate(int global_id);

/**
 * @brief Returns the opaque user payload of a triangle (MFTL_Triangle.data).
 * @return The blob pointer, or NULL when the object has no payload (data_size == 0).
 */
static inline void *MFTLGetTriangleData(const MFTL_Object *obj, const MFTL_Triangle *tri)
{
  return (obj->data_size == 0) ? MFTL_NULL : tri->data;
}

/**
 * @brief Copies obj->data_size bytes from src into the triangle's payload blob.
 * No-op when obj->data_size == 0. The blob must be a flat POD (see MFTL_Object.data_size).
 */
static inline void MFTLSetTriangleData(MFTL_Object *obj, MFTL_Triangle *tri, const void *src)
{
  if (obj->data_size == 0) return;
  memcpy(tri->data, src, obj->data_size);
}

/**
 * @brief Initializes the opaque per-triangle user payload for an object.
 *
 * Sets object->data_size and allocates a zeroed blob of data_size bytes for
 * every existing triangle (local and ghost). Must be called collectively with
 * the SAME data_size on all ranks before any redistribution/ghosting.
 * Idempotent for the same size; re-initializing with a DIFFERENT non-zero
 * size is rejected (no re-init).
 *
 * @param object The object.
 * @param data_size Payload size in bytes per triangle (0 = disable).
 * @return MFTL_SUCCESS on success, MFTL_ERROR on a re-init mismatch.
 */
int MFTLInitializeObjectData(MFTL_Object *object, size_t data_size);

/**
 * @brief Frees an object (entity arrays, payload blobs, maps, flattened connectivity) and sets
 * *p_object to NULL. The domain is not owned by the object; remove the object from its
 * ensemble first (MFTLEnsembleRemoveObject) unless the ensemble is being destroyed.
 */
void MFTLObjectDestroy(MFTL_Object **p_object);

/**
 * @brief Sets global_n_triangles / global_n_vertices of every object of the ensemble (owned
 * entities summed over all ranks). Collective.
 */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble);

/**
 * @brief Single-object variant of MFTLEnsembleUpdateGlobalCounts. Collective over the
 * object's domain (the object must be linked to one).
 */
void MFTLObjectUpdateGlobalStats(MFTL_Object *object);

/**
 * @brief Creates an ensemble on a uniform Cartesian decomposition of the global box.
 *
 * Duplicates comm, checks cuts[0]*cuts[1]*cuts[2] == number of ranks, stores the box and its
 * periodicity, builds the Cartesian and 26-neighbour distributed-graph communicators, and
 * registers the MFTL2 MPI datatypes with the first live ensemble. Collective over comm.
 *
 * @param comm The base MPI communicator (usually MPI_COMM_WORLD).
 * @param cuts An array of 3 integers specifying the domain decomposition grid (ranks in x, y, z).
 * Product must equal the total number of MPI ranks.
 * @param global_min Array of 3 doubles defining the minimum physical coordinates (x, y, z).
 * @param global_max Array of 3 doubles defining the maximum physical coordinates (x, y, z).
 * @param is_periodic Array of 3 booleans indicating if the domain is periodic in x, y, z.
 * @param initial_object_capacity The initial size of the object pointer array (e.g., 10).
 * The array can grow, but setting this avoids early reallocations.
 *
 * @return A pointer to the newly created MFTL_Ensemble, or NULL if initialization fails.
 */
MFTL_Ensemble *MFTLEnsembleCreate(MPI_Comm comm, int cuts[3], double global_min[3], double global_max[3], bool is_periodic[3], int initial_object_capacity);

/**
 * @brief Creates an ensemble on a RECTILINEAR decomposition.
 *
 * Same as MFTLEnsembleCreate, but the slab boundaries along every axis are given explicitly
 * (axis_cuts[d] holds cuts[d] + 1 ascending coordinates; NULL = uniform split of
 * [global_min[d], global_max[d]]) and the slab -> rank map is given explicitly
 * (rank_map[i + j*cuts[0] + k*cuts[0]*cuts[1]] = rank; NULL = MPI Cartesian order).
 * The arrays are copied. Collective over comm.
 */
MFTL_Ensemble *MFTLEnsembleCreateRectilinear(MPI_Comm comm, const int cuts[3], const double *const axis_cuts[3], const int *rank_map,
                                             const double global_min[3], const double global_max[3], const bool is_periodic[3], int initial_object_capacity);

/**
 * @brief Replaces the decomposition of an existing ensemble (collective).
 *
 * Rebuilds the Cartesian/graph communicators, the neighbour list, the rank map and the
 * axis cuts; the global box and periodicity are replaced too. Objects are NOT moved: call
 * MFTLRedistribute afterwards. Any active load-balancing override is cleared.
 */
int MFTLEnsembleSetDecomposition(MFTL_Ensemble *ens, const int cuts[3], const double *const axis_cuts[3], const int *rank_map, const double global_min[3],
                                 const double global_max[3], const bool is_periodic[3]);

/**
 * @brief Detaches an object from its ensemble WITHOUT destroying it.
 *
 * The object is removed from ens->objects (the order of the remaining objects is kept) and
 * its domain link is cleared. Returns MFTL_ERROR if the object is not in the ensemble.
 */
int MFTLEnsembleRemoveObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Builds a temporary domain on base's communicator with a different
 * (uniform) process grid — e.g. the x-only slab layout used by load-balanced remeshing on
 * a 3-D decomposition. Slab (i,j,k) maps to rank i + j*cuts[0] + k*cuts[0]*cuts[1].
 * Collective. Release with MFTLDomainFreeTemporary. The caller swaps it into
 * ens->domain (objects reference &ens->domain) and restores the original afterwards.
 */
int MFTLDomainCreateTemporary(const MFTL_Domain *base, const int cuts[3], MFTL_Domain *tmp);
void MFTLDomainFreeTemporary(MFTL_Domain *tmp);

/**
 * @brief Moves an object from one ensemble to another (appended to dst).
 *
 * Both ensembles must describe the same decomposition (the object keeps its entities and
 * ownership; only its domain link changes). Rank-local; call it identically on all ranks.
 */
int MFTLEnsembleTransferObject(MFTL_Ensemble *dst, MFTL_Ensemble *src, MFTL_Object *object);

/**
 * @brief Appends an object to the ensemble and links it to the ensemble's domain. Rank-local;
 * call it identically on all ranks (object lists must agree).
 */
int MFTLEnsembleAddObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Destroys an ensemble: its objects, its communicators and decomposition arrays, and —
 * with the last live ensemble — the MFTL2 MPI datatypes. Sets *p_ens to NULL. Collective.
 */
void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens);

/**
 * @brief Generates an outward-wound icosphere (20 * 4^refinements triangles) of the given
 * centre and radius into an empty object; see src/generators/sphere.c for the collective contract.
 */
int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements);

#endif  // MFTL_CORE_H