/*
 * MFTL2 parallel — ownership, redistribution and ghost communication: the packed wire records
 * and their MPI datatypes, the owner-of-a-position rule (MFTLGetOwnerProc), redistribution,
 * ghost refresh and the ghosting extensions.
 *
 * Wire format: the Packed* records below are transported as committed MPI struct types; their
 * field sets and order are part of the MPI wire format (and of the restart-independent
 * exchange between ranks) and must not change.
 */
#ifndef MFTL_COMMUNICATIONS_H
#define MFTL_COMMUNICATIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <stdbool.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "khash.h"

/**
 * @struct PackedVertex
 * @brief Pointer-free wire record of an MFTL_Vertex; references are global ids.
 */
typedef struct
{
  // --- Identity & Ownership ---
  int global_id;
  int owner_proc;

  // --- Geometric Properties ---
  double pos[3];
  double normal[3];
  double meancurvature;
  double gaussiancurvature;
  double area;
  double roughness_mean;  // roughness data crosses ranks with the vertex
  double roughness_variance;

  // --- State Information ---
  bool is_ghost;  // the SENDER's status: false = ownership-authoritative copy
  int flags;

  int id_he_out;  // gid of the outgoing half-edge

  int body_id;  // object id

  int img[3];  // periodic image counters (travel with pos)

} PackedVertex;

/**
 * @struct PackedTriangle
 * @brief Pointer-free wire record of an MFTL_Triangle (the payload travels separately).
 */
typedef struct
{
  // --- Identity & Ownership ---
  int global_id;
  int owner_proc;

  int he_id;  // gid of one half-edge of this triangle

  // --- Geometric Properties ---
  double normal[3];
  double area;
  double center[3];

  // --- State Information ---
  bool is_ghost;
  int flags;

  int body_id;  // object id

  double surface_scalar;  // conserved surface scalar (MFTL_Triangle.surface_scalar)
} PackedTriangle;

/**
 * @struct PackedHalfEdge
 * @brief Pointer-free wire record of an MFTL_HalfEdge. twin_gid carries the twin STATE code
 * (twin gid, or MFTL_TWIN_NONE - tag for a BOUNDARY half-edge); twin_idx and the boundary
 * cache are re-derived at the destination. owner_proc is not part of the MPI datatype.
 */
typedef struct
{
  // --- Identity & Ownership ---
  int global_id;
  int owner_proc;

  int vertex_gid;  // gid of the target vertex
  int twin_gid;  // twin state code (see above)
  int next_gid;  // gid of the next half-edge of the triangle
  int triangle_gid;  // gid of the triangle

  // --- State Information ---
  bool boundary;
  int flags;
} PackedHalfEdge;

/**
 * @struct PackedTriangleComplex
 * @brief A triangle with its three half-edges, the unit of triangle transport (its vertices
 * travel as PackedVertex records in a separate stream).
 */
typedef struct
{
  PackedTriangle triangle;
  PackedHalfEdge half_edges[3];
} PackedTriangleComplex;

// per-destination send buffer of a packing pass
typedef struct
{
  int count;  // elements packed so far (bytes for a payload stream)
  int capacity;
  void* buffer;
} MFTL_SendBuffer;

// Reply record of MFTLUpdateGhostPositions: the owner's current vertex data.
typedef struct
{
  int global_id;
  int found;  // 0 when the owner no longer holds this entity (legitimately
  // killed by a remesh op since the last sync); the requester keeps its
  // stale copy, which the next redistribution drops. Never fabricate data.
  double pos[3];
  double normal[3];
  double meancurvature;
  double gaussiancurvature;
  double area;
  double roughness_mean;
  double roughness_variance;
  int img[3];  // ghost copies carry the owner's image counters
} GhostUpdatePayload;

// growable list of gids
typedef struct
{
  int count;
  int capacity;
  int* gids;
} MFTL_GidList;

// Committed MPI datatypes of the records above (defined in src/parallel/comm_types.c; valid while at
// least one ensemble exists).
extern MPI_Datatype MPI_PackedVertex;
extern MPI_Datatype MPI_PackedTriangle;
extern MPI_Datatype MPI_PackedHalfEdge;
extern MPI_Datatype MPI_PackedTriangleComplex;
extern MPI_Datatype MPI_GhostUpdatePayload;

/**
 * @brief Creates and commits the MFTL2 MPI datatypes. Called by MFTLEnsembleCreate* for the
 * first live ensemble (reference counted); applications do not call it directly.
 */
void MFTLRegisterMpiTypes();

/**
 * @brief Frees the MFTL2 MPI datatypes; called by MFTLEnsembleDestroy with the last live
 * ensemble.
 */
void MFTLFreeMpiTypes();

// Slab of coordinate x in a sorted cut array c[0..n] (n slabs): the largest s in [0, n-1]
// with c[s] <= x; positions outside [c[0], c[n]] clamp to the end slabs.
static inline int MFTLSlabIndex(const double* c, int n, double x)
{
  int lo = 1, hi = n - 1, s = 0;
  while (lo <= hi)
  {
    int mid = lo + (hi - lo) / 2;
    if (c[mid] <= x)
    {
      s = mid;
      lo = mid + 1;
    }
    else
    {
      hi = mid - 1;
    }
  }
  return s;
}

/**
 * @brief Owner rank of a position: the rank of the slab containing it (half-open slabs
 * [lo, hi) per axis; uniform, rectilinear, or the x-only load-balancing override).
 *
 * Deterministic and identical on every rank, so every rank derives the same owner from the
 * same position. Positions outside the global box (possible transiently, e.g. a front vertex
 * advected through a wall) clamp to the boundary slabs.
 */
static inline int MFTLGetOwnerProc(const MFTL_Domain* domain, const double pos[3])
{
  // x-only load-balanced ownership override (MFTL1 ownership ranges). Slab index == owning
  // rank; cuts are arbitrary x positions.
  // Out-of-window positions clamp to the end slabs (positions may drift by
  // sub-edge amounts during remeshing; the window covers bbox +/- refLength).
  if (domain->lb_nprocs > 0)
  {
    if (pos[0] <= domain->lb_xmin) return 0;
    if (pos[0] >= domain->lb_xmax) return domain->lb_nprocs - 1;

    // Binary search: largest s in [1, lb_nprocs-1] with lb_xcuts[s] <= x.
    int lo = 1, hi = domain->lb_nprocs - 1, s = 0;
    while (lo <= hi)
    {
      int mid = lo + (hi - lo) / 2;
      if (domain->lb_xcuts[mid] <= pos[0])
      {
        s = mid;
        lo = mid + 1;
      }
      else
      {
        hi = mid - 1;
      }
    }
    return s;
  }

  // uniform slabs: truncation of (pos - min) * cuts / L, clamped to [0, cuts - 1]
  int ijk[3];
  for (int d = 0; d < 3; ++d)
  {
    if (domain->axis_cuts[d])  // rectilinear slabs
    {
      ijk[d] = MFTLSlabIndex(domain->axis_cuts[d], domain->cuts[d], pos[d]);
    }
    else
    {
      double f = (pos[d] - domain->global_min[d]) * domain->grid_factors[d];
      int c = (f > 0.0) ? (int) f : 0;
      ijk[d] = (c >= domain->cuts[d]) ? domain->cuts[d] - 1 : c;
    }
  }
  int i = ijk[0], j = ijk[1], k = ijk[2];

  int linear_idx = i + (j * domain->cuts[0]) + (k * domain->cuts[0] * domain->cuts[1]);
  return domain->rank_map[linear_idx];
}

/**
 * @brief Redistribution (collective): recomputes ownership (MFTLGetOwnerProc; vertices by pos,
 * triangles by their CACHED centre tri->center) and moves every entity to its owner, with one
 * ghost layer (each triangle also goes to the owners of its vertices and of its edge-neighbour
 * triangles). Call MFTLComputeTriangleProperties before it after any change of positions: the
 * cached centre decides ownership, and the cached area/normal/centre travel with the triangle
 * (ghost triangles keep them until the next redistribution).
 *
 * On return (a quiescent point, plans/16 §2.3) every rank stores its owned entities first and
 * its ghosts after them; ownership is exactly-once; every edge of an owned triangle is LOCAL or
 * BOUNDARY and the fan of every owned vertex is complete (ghost completion G1/G2, §2.5); the
 * twin states have been validated. Array indices change; global ids, image counters, flags,
 * the surface scalar and the opaque payload travel unchanged.
 *
 * MFTLRedistributeObject handles one object; MFTLRedistributeListOfObjects all objects of an
 * ensemble in one round. Returns MFTL_ERROR on a payload-size mismatch or a failed
 * completion/validation; no-op on a single rank.
 */
int MFTLRedistributeObject(MFTL_Object* object);
int MFTLRedistributeListOfObjects(MFTL_Ensemble* ensemble);

/**
 * @brief Redistributes an ensemble (collective): MFTLRedistributeObject for a single object,
 * MFTLRedistributeListOfObjects otherwise. MFTL_ERROR for an ensemble without objects.
 */
static inline int MFTLRedistribute(MFTL_Ensemble* ens)
{
  MFTL_ASSERT(ens != NULL && "MFTLRedistribute: NULL ensemble");

  if (ens->num_objects == 0 || ens->objects == MFTL_NULL) return MFTL_ERROR;

  if (ens->num_objects == 1) return MFTLRedistributeObject(ens->objects[0]);

  return MFTLRedistributeListOfObjects(ens);
}

// Packing / ghost-append building blocks of the transport paths (references are global ids
// in the records and remain global ids in appended ghosts until they are remapped).
void MFTLPackSingleVertex(MFTL_Object* obj, int vert_idx, PackedVertex* dst, bool is_ghost);
void MFTLPackSingleTriangle(MFTL_Object* obj, int tri_idx, PackedTriangleComplex* dst, bool is_ghost);

void MFTLAppendGhostVertex(MFTL_Object* obj, PackedVertex* pv);
void MFTLAppendGhostTriangle(MFTL_Object* obj, PackedTriangleComplex* ptc, const void* data);

/**
 * @brief Collective contract check — obj->data_size must be identical
 * on every rank before any triangle-payload transport. Prints a loud diagnostic
 * and returns MFTL_ERROR on mismatch; MFTL_SUCCESS when all ranks agree.
 * Called at the entry of every triangle-data transport path.
 */
int MFTLCheckTriangleDataSizeConsistency(MFTL_Object* obj);

/**
 * @brief Refreshes the ghost vertices of all objects from their owners (collective): position
 * and image counters, normal, curvatures, area and roughness. The ghost set is unchanged;
 * ghosts whose owner is not a graph neighbour or no longer holds them stay stale until the
 * next redistribution. Call it after moving owned vertices.
 * @param object_lookup Optional hash map (body id -> object index); MFTL_NULL = linear search.
 */
int MFTLUpdateGhostPositions(MFTL_Ensemble* ens, khash_t(object_map) * object_lookup);

/**
 * @brief Refreshes ghost TRIANGLE user payloads (MFTL_Triangle.data) from their owners.
 *
 * Mirrors MFTLUpdateGhostPositions structurally, but ships the opaque
 * per-triangle blobs (obj->data_size bytes, see MFTL_Object.data_size) instead
 * of vertex geometry: for every local ghost triangle the owner is asked for
 * the current blob (keyed by global id), and the reply overwrites the ghost
 * copy. Ghosts whose owner no longer holds the triangle (e.g. killed by
 * remeshing) keep their stale copy; the next redistribution drops them.
 *
 * No-op fast path when no object carries a payload (all data_size == 0).
 * Must be called collectively by all ranks.
 *
 * @param ens The ensemble containing the objects.
 * @param object_lookup Optional hash map (BodyID -> Index). Pass MFTL_NULL to use linear search fallback.
 * @return MFTL_SUCCESS on success.
 */
int MFTLScatterTriangleData(MFTL_Ensemble* ens, khash_t(object_map) * object_lookup);

/**
 * @brief Adds `steps` further ghost layers (collective): each step appends the one-ring
 * triangles of the current ghost vertices and their vertices (V -> T -> V), obtained from the
 * owners over the graph neighbours.
 */
int MFTLExtendGhostLayer(MFTL_Ensemble* ens, int steps);

/**
 * @brief Spatial ghosting (collective): owned triangles whose bounding box, inflated by
 * proximity_tolerance, overlaps a graph-neighbour rank's box become ghosts there. The box test
 * uses the stored positions (not periodic-aware).
 */
int MFTLEnsembleCreateSpatialGhosts(MFTL_Ensemble* ens, double proximity_tolerance);

/**
 * @brief Makes every locally OWNED triangle of obj (with its vertices and
 * payload) available as a ghost on EVERY rank whose box lies within `distance` of the
 * triangle's bounding box (MFTL1 MFTLSendLocalTrisToAllProcs parity; global exchange, not
 * limited to the 26 graph neighbours, rectilinear boxes honoured). Already stored entities
 * are not duplicated; topology is resolved afterwards. Collective; invalidates indices.
 */
int MFTLObjectSendLocalTrianglesToAll(MFTL_Object* obj, double distance);

#endif