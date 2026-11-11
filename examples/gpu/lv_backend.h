/*
 * @Author: Christian Gorges
 *
 * @brief The device side of the LeVeque GPU example, as one small C API with
 *        two implementations: CUDA (lv_backend_cuda.cu) and the directives
 *        (lv_backend_directives.c), which the compiler flags turn into OpenACC,
 *        OpenMP offload, OpenMP threads or serial C.
 *
 * The header carries NO MPI and NO MFTL type. The driver hands over raw views of
 * the library's pools (lv_mesh), so the backends compile with nvcc, nvc or gcc
 * alone, and all of them are judged on the same interface.
 *
 * Every call returns when its work is complete (no asynchrony leaks out), so a
 * host timer around a call measures that call.
 */
#ifndef LV_BACKEND_H
#define LV_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief A view of ONE object's pools: pointers into MFTL memory, nothing owned.
 *
 * Owned entities form the prefix of each pool: vertices [0, nv_owned) and
 * triangles [0, nt_owned); the rest are ghosts.
 */
typedef struct lv_mesh
{
  double *pos;  ///< verts.pos       [3 nv]
  double *vnormal;  ///< verts.normal    [3 nv]   device scratch, never transferred
  const int *tri_verts;  ///< tris.verts      [3 nt]   corner k of t = origin of half-edge 3t+k
  const int *tri_nbr;  ///< tris.neighbors  [3 nt]   triangle across half-edge 3t+k, -1 at a boundary
  const int *tri_rec;  ///< tris.rec        [3 nt]   MFTL_Triangle as int[3]; [3t] is the global id
  double *area;  ///< tris.area       [nt]
  double *normal;  ///< tris.normal     [3 nt]
  double *center;  ///< tris.center     [3 nt]
  int nv;
  int nv_owned;
  int nt;
  int nt_owned;
  int nv_capacity;  ///< verts.capacity: every vertex array above holds this many entries
  int nt_capacity;  ///< tris.capacity
} lv_mesh;

/** @brief The rank-local sums of the diagnostics, over OWNED triangles. */
typedef struct lv_measure_out
{
  double n_edge;  ///< undirected edges this rank counts (each edge once mesh-wide)
  double n_out_of_band;  ///< of those, outside [min_len, max_len]
  double area;  ///< surface area
  double q_sum;  ///< sum of the triangle quality 4 sqrt(3) A / sum |e|^2
  double n_inverted;  ///< triangles whose normal points away from their vertex normals
  double volume;  ///< flux volume, x-axis, about vol_ref
  double min_q;  ///< minimum quality, 1 if no triangle
} lv_measure_out;

/** @brief "cuda", "openacc-gpu", "openmp-host", ...: what this binary runs. */
const char *lv_backend_name(void);

/**
 * @brief Selects device `local_rank % n_devices` and initialises it.
 * @return 0, or non-zero when the backend has no device to run on.
 */
int lv_init(int local_rank);
void lv_finalize(void);

/**
 * @brief Uploads the whole mesh: positions, connectivity, triangle ids.
 *
 * Device memory is sized by the pool CAPACITIES and kept across attaches: it is
 * reallocated only when a pool moved or grew, otherwise the upload is a copy
 * into the existing allocation.
 */
void lv_attach(const lv_mesh *m);
/** @brief The mesh is about to be remeshed: nothing on the device may be used until the next attach. */
void lv_detach(void);
/** @brief Device (re)allocations of mesh storage so far, for the summary. */
long lv_device_allocations(void);

/** @brief RK4 step of the OWNED vertices, t -> t + dt, in the analytical LeVeque field of period T. */
void lv_advect(double t, double dt, double T);
/** @brief Device -> host: positions of the owned vertices. */
void lv_get_owned_positions(void);
/** @brief Host -> device: positions of the ghost vertices (after the host ghost sync). */
void lv_put_ghost_positions(void);

/** @brief Area, unit normal and centroid of every triangle, ghosts included. */
void lv_triangle_geometry(void);
/** @brief Device -> host: area, normal and centroid of every triangle. */
void lv_get_triangle_geometry(void);

/**
 * @brief Area-weighted vertex normals, then the per-triangle diagnostics reduced
 *        over the owned triangles. Only the 7 numbers come back.
 */
void lv_measure(double min_len, double max_len, double vol_ref, lv_measure_out *out);

#ifdef __cplusplus
}
#endif

#endif
