/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:35:27
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
 */

#ifndef MFTL_CORE_H
#define MFTL_CORE_H

// #define DEBUG

#include <mpi.h>  // For MPI types like MPI_Comm
#include <stdbool.h>  // For bool type (C99)
#include <stddef.h>  // For size_t
#include <stdint.h>  // For int64_t
#include <limits.h>
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "khash.h"

KHASH_MAP_INIT_INT(vertex_map, int)
KHASH_MAP_INIT_INT(halfedge_map, int)
KHASH_MAP_INIT_INT(triangle_map, int)
KHASH_MAP_INIT_INT(object_map, int)

//------------------------------------------------------------------------------
// Forward Declarations
//------------------------------------------------------------------------------
// Forward declarations are necessary because the structs refer to each other
// in a circular manner (e.g., a Vertex points to a HalfEdge, which points
// back to a Vertex).

struct MFTL_HalfEdge;
struct MFTL_Vertex;
struct MFTL_Triangle;
struct MFTL_Object;
struct MFTL_Domain;
struct MFTL_Ensemble;

/* Stage 16C (plans/16 §4.3): boundary-curve projector of a CONSTRAINED boundary tag. X_in is a
   point in the local frame of the operation (coordinates in the periodic image of the wrapped
   position of the operation's first vertex); X_out receives its projection onto the curve.
   Returns MFTL_SUCCESS, or MFTL_ERROR to refuse (the operation is then skipped). */
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
 * @brief Represents a point in 3D space.
 *
 * A vertex stores its geometric position and properties. Crucially, it holds a
 * pointer to one of its outgoing half-edges, which allows for efficient
 * traversal of its local neighborhood (the "1-ring").
 */
typedef struct MFTL_Vertex
{
  double pos[3];  ///< x, y, z coordinates of the vertex (WRAPPED into [min, max) on periodic axes).
  /* Stage 16B (plans/16 §3.1): periodic image counters. The continuous coordinate of the vertex
     is X = pos + img ⊙ L (img = 0 on non-periodic axes). Maintained by the displacement rule
     (§3.2) and re-gauged by MFTLCanonicaliseImages (§3.3). */
  int img[3];

  // --- Geometric Properties ---
  double normal[3];  ///< Computed normal vector at the vertex.
  double meancurvature;  ///< Computed mean curvature.
  double gaussiancurvature;  ///< Computed Gaussian curvature.
  double area;  ///< Voronoi area associated with the vertex.
  double roughness_mean;  ///< Dihedral-angle roughness mean (owned vertices only).
  double roughness_variance;  ///< Dihedral-angle roughness sample variance (owned only).

  // --- Parallelism & Domain Information ---
  int owner_proc;  ///< The MPI rank that has primary authority over this vertex.
  bool is_ghost;  ///< True if this vertex is a ghost copy from another processor.
  int global_id;  ///< A unique ID across all processors for tracking.

  // --- Topology ---
  int out_he_idx;  ///< Index of the outgoing half-edge.

  // --- Application-Specific Data ---
  int flags;  ///< Bit flags for marking (e.g., for remeshing).

  /* Stage 16C (plans/16 §4.2, §13): extrapolated curvature (2H, as meancurvature) of a boundary
     vertex from an open-stencil paraboloid fit; valid iff MFTL_VFLAG_CURV_EXTRAPOLATED. Local,
     not packed (recomputed on demand). */
  double curv_extrap;

} MFTL_Vertex;

/**
 * @struct MFTL_Triangle
 * @brief Represents a triangular element in the mesh.
 *
 * A Triangle is defined by three vertices. It points to one of the half-edges
 * that form its boundary, allowing for traversal of its edges and vertices.
 */
typedef struct MFTL_Triangle
{
  // --- Topology ---
  int he_idx;  ///< Index of one half-edge on the triangle's boundary.

  // --- Geometric Properties ---
  double normal[3];  ///< The geometric normal of the triangle.
  double area;  ///< The area of the triangle.
  double center[3];  ///< The geometric center (centroid) of the triangle.

  // --- Parallelism & Domain Information ---
  int owner_proc;  ///< The MPI rank that has primary authority over this triangle.
  bool is_ghost;  ///< True if this triangle is a ghost copy from another processor.
  int global_id;  ///< A unique ID across all processors.

  // --- Application-Specific Data ---
  void *data;  ///< Pointer to user-defined data associated with this triangle. TO-DO: put this into a hash define for better cache performance!?
  int flags;  ///< Bit flags for marking (e.g., for remeshing).

  /* Conserved surface scalar (Stage 15, MFTL1 `scalarconcentration` parity): a concentration
     per unit area whose MASS (surface_scalar * area) the library conserves. It travels with
     the triangle through every redistribution / ghost / pack operation, is remapped
     mass-conservingly by split / flip / collapse and rescaled (c *= A_old / A_new) by the
     TSUR3D / volume-conserving vertex moves, starts at 0 for generated / built triangles and
     is written/read by the native mesh I/O. Vertex moves made by the APPLICATION do not
     rescale it. See plans/15 (item L7). */
  double surface_scalar;
} MFTL_Triangle;

/**
 * @struct MFTL_HalfEdge
 * @brief The core of the half-edge data structure.
 *
 * Represents a directed edge segment on the boundary of a single triangle. Each
 * topological edge in the mesh is composed of two opposite-facing half-edges.
 * This structure enables extremely fast and robust traversal of the mesh
 * topology.
 */
typedef struct MFTL_HalfEdge
{
  // --- Topology ---
  int vertex_idx;  ///< Index of the vertex at the END of this directed edge.
  int twin_idx;  ///< Local index of the opposite half-edge; MFTL_ERROR when it is not stored here.
  int next_idx;  ///< Index of the next half-edge in the counter-clockwise direction.
  int triangle_idx;  ///< Index of the triangle this half-edge belongs to.

  // --- Parallelism & Domain Information ---
  int global_id;  ///< A unique ID across all processors.
  /* Twin state code (Stage 16A, plans/16 §2.1; see MFTL_TWIN_* in mftl_constants.h):
     LOCAL     twin_idx >= 0 and twin_gid == half_edges[twin_idx].global_id;
     REMOTE    twin_idx == -1 and twin_gid >= 0: the gid of the live twin on another rank
               (retained on the ghost-layer rim, Stage 15 item L3);
     BOUNDARY  twin_idx == -1 and twin_gid == MFTL_TWIN_NONE - tag: a genuine physical boundary.
     Read it only through the predicates of mftl_topology.h (MFTLHeIsLocal / IsRemote /
     IsBoundary / BoundaryTag / TwinGid / EdgeId). The undirected edge id is
     min(global_id, twin gid) for LOCAL/REMOTE and global_id for BOUNDARY. */
  int twin_gid;

  // --- Application-Specific Data ---
  int flags;  ///< Bit flags for marking (e.g., for remeshing).
  bool boundary;  ///< Cache of MFTLHeIsBoundary (BOUNDARY state); never a rim marker (Stage 16A).
} MFTL_HalfEdge;

//------------------------------------------------------------------------------
// The Object (The Mesh)
//------------------------------------------------------------------------------
/**
 * @struct MFTL_Object
 * @brief The main container for the entire mesh object.
 *
 * This struct holds the arrays of vertices, faces, and half-edges.
 */
typedef struct MFTL_Object
{
  // --- Core Data Arrays (Pools) ---
  MFTL_Vertex *vertices;  ///< Array of all vertices in this partition.
  MFTL_Triangle *triangles;  ///< Array of all triangles in this partition.
  MFTL_HalfEdge *half_edges;  ///< Array of all half-edges in this partition.

  int n_vertices;  ///< Current number of active vertices (local + ghost).
  int n_triangles;  ///< Current number of active triangles (local + ghost).
  int n_half_edges;  ///< Current number of active half-edges (local + ghost).

  /* These counts refer to the first n_local_* elements in the arrays.
     Ghost entities are stored after the local entities. */
  int n_local_triangles;  ///< Number of local (non-ghost) triangles.
  int n_local_vertices;  ///< Same invariant for vertices[].
  int n_local_half_edges;  ///< Same invariant for half_edges[].

  int capacity_vertices;  ///< Allocated capacity for the vertices array.
  int capacity_triangles;  ///< Allocated capacity for the triangles array.
  int capacity_half_edges;  ///< Allocated capacity for the half_edges array.

  // Global ID counters for new body creation or vertex insertion.
  int next_vertex_gid;
  int next_triangle_gid;
  int next_halfedge_gid;

  int global_n_triangles;  // Total across all ranks (excluding ghosts)
  int global_n_vertices;  // Total across all ranks (excluding ghosts)

  // --- Hash maps for fast lookup ---
  khash_t(vertex_map) * vertex_lookup;
  khash_t(triangle_map) * triangle_lookup;
  khash_t(halfedge_map) * halfedge_lookup;

  // --- Acceleration Structures (Flattened Topology) ---
  // These are allocated by MFTLFlattenConnectivity()
  int *flat_tri_verts;  ///< Size: 3 * n_triangles. Vert indices for each tri.
  int *flat_tri_neighbors;  ///< Size: 3 * n_triangles. Neighbor tri indices.
  int *vert_offsets;  ///< Size: n_vertices + 1. CSR offsets for V2T map.
  int *vert_tris;  ///< Size: 3 * n_triangles. CSR values for V2T map.

  int global_id;  ///< A unique ID across all processors for tracking.

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

  // Pointer to the domain this object lives in.
  struct MFTL_Domain *domain;

  /* Stage 16A: the body is declared OPEN (MFTLObjectDeclareOpen, identical on every rank).
     Only a declared-open body may carry BOUNDARY half-edges; a closed body with an
     unmatched edge is an error, never silently opened (plans/16 §0, §2.2). */
  int declared_open;

  /* Stage 16B (plans/16 §3.3/§3.4): cached per-axis wrap flags, set by MFTLCanonicaliseImages:
     wrap[k] = 1 iff some component of the body winds along k (a cycle with non-zero lattice
     residual). wrap_valid = 0 until the first canonicalisation. */
  int wrap[3];
  int wrap_valid;
  /* winding lattice generators of the body (all components), from the last canonicalisation */
  int n_wind;
  int (*wind_gen)[3];
  /* Stage 16C (plans/16 §4.4.4): set when a canonicalisation finds different wrap flags than the
     previous valid state (a class transition); read and cleared by MFTLObjectTakeClassTransition. */
  int class_transition;

  /* Stage 16C (plans/16 §4.4.4): MFTL_REMESH_MODE_DEFAULT / _GENERALIZED (identical on all ranks) */
  int remesh_mode;
  /* Stage 16C (plans/16 §4.3): boundary-curve mode per boundary tag (identical on all ranks;
     tags without an entry are REPRESENTED) */
  int n_bcurve;
  struct MFTL_BoundaryCurve *bcurve;

} MFTL_Object;

//------------------------------------------------------------------------------
// The Domain (The "World" & MPI Topology)
//------------------------------------------------------------------------------
/**
 * @struct MFTL_Domain
 * @brief Defines the computational environment, MPI topology, and physical bounds.
 * This struct is shared by all MFTL_Object instances in an MFTL_Ensemble. It decouples the "container"
 * from the "contents".
 */
typedef struct MFTL_Domain
{
  // --- MPI Topology ---
  MPI_Comm comm_world;  ///< The main communicator.
  MPI_Comm comm_cart;  ///< Cartesian topology for geometric domain decomposition.
  MPI_Comm comm_graph;  ///< Graph topology for actual mesh adjacencies.

  int rank;  ///< My rank.
  int nproc;  ///< Total processes.

  // Neighbors (Dynamic array)
  int num_neighbors;
  int *neighbors;

  // --- Domain Decomposition ---
  int cuts[3];  ///< Processor cuts {x, y, z}.

  // --- Physical Bounds ---
  double global_min[3];  ///< Global domain start.
  double global_max[3];  ///< Global domain end.
  double domain_size[3];  ///< Length of domain (max - min).
  double domain_half_size[3];  ///< Pre-calculated for periodicity checks.

  // --- Boundary Conditions ---
  bool is_periodic[3];  ///< x, y, z periodicity.

  // --- Lookup table for ijk to rank mapping ---
  int *rank_map;  // Linear array: rank_map[i + j*nx + k*nx*ny] = rank
  double grid_factors[3];  // Pre-calculated: cuts[i] / domain_size[i]

  // --- Stage 15 (L1): rectilinear decomposition ---
  // When axis_cuts[d] != NULL, slab i along axis d is [axis_cuts[d][i], axis_cuts[d][i+1])
  // (cuts[d] + 1 ascending coordinates; axis_cuts[d][0] / [cuts[d]] are the global box
  // faces) instead of the uniform split. rank_map may then be any bijection slab -> rank
  // (e.g. the PETSc DMDA x-fastest layout). NULL axes keep the uniform split.
  double *axis_cuts[3];

  // --- Parallel Remeshing (Stage 10): x-only load-balanced ownership ---
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
 * @brief The top-level handle for the meshing library state.
 */
typedef struct MFTL_Ensemble
{
  // The Environment (MPI + Physics)
  MFTL_Domain domain;

  // The Data (List of Objects)
  MFTL_Object **objects;
  int num_objects;
  int capacity_objects;  // If you add/remove objects dynamically

  // Global Statistics
  int total_global_triangles;

  // --- Parallel Remeshing ---
  MFTL_EnsembleType type;
  MFTL_RemeshAlgorithm last_algorithm;  // algorithm used in the last remesh

} MFTL_Ensemble;

/**
 * @brief Creates and performs basic initialization of a new MFTL_Object struct.
 *
 * This function allocates the main mesh struct and sets up the MPI
 * communicator. It does NOT allocate the main data arrays (pools) or define the
 * domain decomposition. That is handled by other functions.
 *
 * @param comm The MPI communicator the mesh will be part of. If MPI_COMM_NULL
 * is passed, the function will use MPI_COMM_WORLD.
 * @return A pointer to the newly allocated mesh, or NULL on failure.
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
 * @brief Frees all memory associated with an MFTL_Object mesh.
 *
 * This function cleans up all allocated data pools, the domain decomposition
 * information, and the MPI communicator before freeing the mesh struct itself.
 *
 * @param p_mesh A pointer to the object pointer (MFTL_Object**). This allows
 * the function to set the original pointer to NULL after freeing.
 */
void MFTLObjectDestroy(MFTL_Object **p_object);

/**
 * @brief Updates the global vertex and triangle counts for all objects in the ensemble.
 * * It sums up the locally owned (non-ghost) entities across all MPI ranks 
 * and stores the result in each object's global_n_* fields.
 * * @param ensemble The MFTL ensemble containing the domain and objects.
 */
void MFTLEnsembleUpdateGlobalCounts(MFTL_Ensemble *ensemble);

/**
 * @brief Updates the global vertex and triangle counts for a SINGLE object.
 * * This is useful when you have modified only one object (e.g. during generation)
 * and don't want to pay the cost of reducing data for the entire ensemble.
 * * @param object The MFTL object to update. Must have a valid domain link.
 */
void MFTLObjectUpdateGlobalStats(MFTL_Object *object);

/**
 * @brief Creates a new MFTL Ensemble, initializing the global simulation domain and MPI topology.
 *
 * This function serves as the primary entry point for setting up the library's environment.
 * It performs the following key tasks:
 * 1. Allocates the MFTL_Ensemble structure.
 * 2. Initializes the MFTL_Domain, including:
 * - Duplicating the provided MPI communicator for internal safety.
 * - Validating that the process grid (cuts) matches the MPI size.
 * - Setting up global physical bounds and periodicity flags.
 * - Creating the complex 26-neighbor distributed graph topology for efficient communication.
 * 3. Registers custom MPI types if this is the first active ensemble.
 * 4. Allocates the initial capacity for the object list.
 *
 * @note This function must be called collectively by all ranks in the communicator.
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
 * @brief Stage 15 (L1): creates an ensemble on a RECTILINEAR decomposition.
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
 * @brief Stage 15 (L1): replaces the decomposition of an existing ensemble (collective).
 *
 * Rebuilds the Cartesian/graph communicators, the neighbour list, the rank map and the
 * axis cuts; the global box and periodicity are replaced too. Objects are NOT moved: call
 * MFTLRedistribute afterwards. Any active load-balancing override is cleared.
 */
int MFTLEnsembleSetDecomposition(MFTL_Ensemble *ens, const int cuts[3], const double *const axis_cuts[3], const int *rank_map, const double global_min[3],
                                 const double global_max[3], const bool is_periodic[3]);

/**
 * @brief Stage 15 (L4): detaches an object from its ensemble WITHOUT destroying it.
 *
 * The object is removed from ens->objects (the order of the remaining objects is kept) and
 * its domain link is cleared. Returns MFTL_ERROR if the object is not in the ensemble.
 */
int MFTLEnsembleRemoveObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Stage 15: builds a temporary domain on base's communicator with a different
 * (uniform) process grid — e.g. the x-only slab layout used by load-balanced remeshing on
 * a 3-D decomposition. Slab (i,j,k) maps to rank i + j*cuts[0] + k*cuts[0]*cuts[1].
 * Collective. Release with MFTLDomainFreeTemporary. The caller swaps it into
 * ens->domain (objects reference &ens->domain) and restores the original afterwards.
 */
int MFTLDomainCreateTemporary(const MFTL_Domain *base, const int cuts[3], MFTL_Domain *tmp);
void MFTLDomainFreeTemporary(MFTL_Domain *tmp);

/**
 * @brief Stage 15 (L4): moves an object from one ensemble to another (appended to dst).
 *
 * Both ensembles must describe the same decomposition (the object keeps its entities and
 * ownership; only its domain link changes). Rank-local; call it identically on all ranks.
 */
int MFTLEnsembleTransferObject(MFTL_Ensemble *dst, MFTL_Ensemble *src, MFTL_Object *object);

/**
 * @brief Adds an object to the ensemble and links the domain.
 * @param ensemble The ensemble to add to.
 * @param object The object to add.
 * @return MFTL_SUCCESS on success.
 */
int MFTLEnsembleAddObject(MFTL_Ensemble *ensemble, MFTL_Object *object);

/**
 * @brief Destroys an MFTL Ensemble and frees all associated resources.
 *
 * This function performs a comprehensive cleanup:
 * 1. Iterates through all MFTL_Object pointers in the ensemble and calls MFTLObjectDestroy() on them.
 * 2. Frees the MFTL_Domain resources, including:
 * - The distributed graph, Cartesian, and world communicators.
 * - The neighbor rank arrays.
 * 3. Frees the ensemble structure itself.
 * 4. If this is the last active ensemble, it deregisters custom MPI types (Reference Counting).
 *
 * @param p_ens A double pointer to the ensemble.
 * The underlying pointer is set to NULL to prevent dangling reference usage.
 */
void MFTLEnsembleDestroy(MFTL_Ensemble **p_ens);

/**
 * @brief Public function to generate a sphere.
 * @param object The object to generate the sphere on.
 * @param center The center position of the sphere.
 * @param radius The radius of the sphere.
 * @param refinements The number of refinements for the sphere.
 */
int MFTLGenerateSphere(MFTL_Object *object, double center[3], double radius, int refinements);

#endif  // MFTL_CORE_H