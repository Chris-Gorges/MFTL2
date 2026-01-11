/*
 * @Author: Christian Gorges
 * @Date: 2025-08-09 14:35:27
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-22 19:52:21
 */

#ifndef MFTL_TYPES_H
#define MFTL_TYPES_H

// #define DEBUG

#include <mpi.h>  // For MPI types like MPI_Comm
#include <stdbool.h>  // For bool type (C99)
#include <stddef.h>  // For size_t
#include <stdint.h>  // For int64_t
#include <limits.h>

#include "mftl_constants.h"
#include "khash.h"

#ifdef MFTL_USE_64BIT
// Huge mode: supports > 4 billion items per MPI rank
typedef int64_t mftl_int_t;
#define MFTL_ID_MAX INT64_MAX
#else
// Standard mode: faster, uses 50% less RAM for topology
typedef int32_t mftl_int_t;
#define MFTL_ID_MAX INT32_MAX
#endif

#ifdef MFTL_USE_64BIT
// 64-bit Keys (GID), 64-bit Values (Local Index)
KHASH_MAP_INIT_INT64(vertex_map, mftl_int_t)
KHASH_MAP_INIT_INT64(halfedge_map, mftl_int_t)
KHASH_MAP_INIT_INT64(triangle_map, mftl_int_t)
KHASH_MAP_INIT_INT64(object_map, mftl_int_t)
#else
// 32-bit Keys (GID), 32-bit Values (Local Index)
KHASH_MAP_INIT_INT(vertex_map, mftl_int_t)
KHASH_MAP_INIT_INT(halfedge_map, mftl_int_t)
KHASH_MAP_INIT_INT(triangle_map, mftl_int_t)
KHASH_MAP_INIT_INT(object_map, mftl_int_t)
#endif

extern MPI_Datatype MPI_MFTL_INT;

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
  double pos[3];  ///< x, y, z coordinates of the vertex.

  // --- Geometric Properties ---
  double normal[3];  ///< Computed normal vector at the vertex.
  double meancurvature;  ///< Computed mean curvature.
  double gaussiancurvature;  ///< Computed Gaussian curvature.
  double area;  ///< Voronoi area associated with the vertex.

  // --- Parallelism & Domain Information ---
  mftl_int_t owner_proc;  ///< The MPI rank that has primary authority over this vertex.
  bool is_ghost;  ///< True if this vertex is a ghost copy from another processor.
  mftl_int_t global_id;  ///< A unique ID across all processors for tracking.

  // --- Topology ---
  mftl_int_t out_he_idx;  ///< Index of the outgoing half-edge.

  // --- Application-Specific Data ---
  mftl_int_t flags;  ///< Bit flags for marking (e.g., for remeshing).

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
  mftl_int_t he_idx;  ///< Index of one half-edge on the triangle's boundary.

  // --- Geometric Properties ---
  double normal[3];  ///< The geometric normal of the triangle.
  double area;  ///< The area of the triangle.
  double center[3];  ///< The geometric center (centroid) of the triangle.

  // --- Parallelism & Domain Information ---
  mftl_int_t owner_proc;  ///< The MPI rank that has primary authority over this triangle.
  bool is_ghost;  ///< True if this triangle is a ghost copy from another processor.
  mftl_int_t global_id;  ///< A unique ID across all processors.

  // --- Application-Specific Data ---
  void *data;  ///< Pointer to user-defined data associated with this triangle. TO-DO: put this into a hash define for better cache performance!?
  mftl_int_t flags;  ///< Bit flags for marking (e.g., for remeshing).
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
  mftl_int_t vertex_idx;  ///< Index of the vertex at the END of this directed edge.
  mftl_int_t twin_idx;  ///< Index of the oppositely oriented half-edge. NULL if on a boundary.
  mftl_int_t next_idx;  ///< Index of the next half-edge in the counter-clockwise direction.
  mftl_int_t triangle_idx;  ///< Index of the triangle this half-edge belongs to.

  // --- Parallelism & Domain Information ---
  mftl_int_t global_id;  ///< A unique ID across all processors.

  // --- Application-Specific Data ---
  mftl_int_t flags;  ///< Bit flags for marking (e.g., for remeshing).
  bool boundary;  ///<  True if this halfedge is a boundary edge.
} MFTL_HalfEdge;

/**
 * @struct MFTL_Object
 * @brief The main container for the entire mesh object.
 *
 * This struct holds the arrays of vertices, faces, and half-edges, as well as
 * information related to memory management and parallel decomposition.
 */
typedef struct MFTL_Object
{
  // --- Core Data Arrays (Pools) ---
  MFTL_Vertex *vertices;  ///< Array of all vertices in this partition.
  MFTL_Triangle *triangles;  ///< Array of all triangles in this partition.
  MFTL_HalfEdge *half_edges;  ///< Array of all half-edges in this partition.

  mftl_int_t n_vertices;  ///< Current number of active vertices.
  mftl_int_t n_triangles;  ///< Current number of active triangles.
  mftl_int_t n_half_edges;  ///< Current number of active half-edges.

  mftl_int_t capacity_vertices;  ///< Allocated capacity for the vertices array.
  mftl_int_t capacity_triangles;  ///< Allocated capacity for the triangles array.
  mftl_int_t capacity_half_edges;  ///< Allocated capacity for the half_edges array.

  // Global ID counters for new body creation or vertex insertion.
  mftl_int_t next_vertex_gid;
  mftl_int_t next_triangle_gid;
  mftl_int_t next_halfedge_gid;

  mftl_int_t global_n_triangles;  // Total across all ranks (excluding ghosts)
  mftl_int_t global_n_vertices;  // Total across all ranks (excluding ghosts)

  // --- Hash maps for fast lookup ---
  khash_t(vertex_map) * vertex_lookup;
  khash_t(triangle_map) * triangle_lookup;
  khash_t(halfedge_map) * halfedge_lookup;

  // --- TO-DO: Create a new struct for MPI communication and move this away from each single object ---
  // --- Parallel Processing Information ---
  // --- MPI int variables have to always be 32 bit ints. ---
  MPI_Comm comm;  ///< MPI communicator for this mesh object.
  int rank;  ///< The rank of this process in the communicator.
  int nproc;  ///< The total number of processes in the communicator.
  // --- MPI Topology ---
  MPI_Comm cart_comm;  // Cartesian communicator
  MPI_Comm graph_comm;  // Distributed graph communicator for neighbors
  int num_neighbors;  // The number of neighbour ranks
  int *neighbors;  // The array of neighbour ranks

  // --- Domain Decomposition Data ---
  int cuts[3];  ///< Processor cuts in each direction (e.g., {4, 2, 2}).

  // --- Bounding Box & Domain Info ---
  // double local_bbox_min[3];  ///< Bounding box min of the local mesh partition.
  // double local_bbox_max[3];  ///< Bounding box max of the local mesh partition.
  double global_bbox_min[3];  ///< Bounding box min of the entire global mesh.
  double global_bbox_max[3];  ///< Bounding box max of the entire global mesh.

  // --- Periodicity support ---
  bool is_periodic[3];  ///< Flags for periodicity in x, y, z.
  double domain_size[3];  ///< Full size of the domain in x, y, z.
  double domain_half_size[3];  ///< Pre-calculated half size for efficiency.

  mftl_int_t global_id;  ///< A unique ID across all processors for tracking.

} MFTL_Object;

#endif  // MFTL_TYPES_H