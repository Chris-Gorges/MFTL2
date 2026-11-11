/*
 * @Author: Christian Gorges
 *
 * @brief The device side of the many-spheres GPU example: one C API, two
 *        implementations -- CUDA (sp_backend_cuda.cu) and the directives
 *        (sp_backend_directives.c: OpenACC, OpenMP offload, OpenMP threads,
 *        serial C, chosen by the compiler flags).
 *
 * The device holds EVERY body of the rank at once, in fixed slots: body slot s
 * keeps its vertices at [s * nv_slot, s * nv_slot + nv) and its triangles at
 * [s * nt_slot, ...), in the order of the library's pools (owned first, then
 * ghosts). One launch then covers all bodies -- a launch costs ~16 us, so one
 * per body and phase would cost more than the work itself (README.md).
 *
 * Data moves only through the packed transfer engine: the driver lists
 * segments (slot, first entity, count, offset in the staging buffer), and one
 * call moves all of them with ONE copy, gathered or scattered on the device.
 * Like lv_backend.h, no MPI and no MFTL type appear here, and every call
 * returns when its work is complete.
 */
#ifndef SP_BACKEND_H
#define SP_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

/** @brief The periodic box, as the kernels need it: MFTL_Domain flattened. */
typedef struct sp_box
{
  double min[3];  ///< global_min
  double max[3];  ///< global_max
  double L[3];  ///< domain_size
  double invL[3];  ///< 1 / L on a periodic axis, 0 otherwise: the minimum image is then the identity
  double gf[3];  ///< grid_factors = cuts / L: the owner cell is a multiply
  int cuts[3];  ///< process grid
  int periodic[3];
} sp_box;

/** @brief One run of consecutive entities of one slot, and where it sits in the staging buffer. */
typedef struct sp_segment
{
  int slot;
  int first;  ///< first entity (vertex or triangle) in the slot
  int count;  ///< entities
  int offset;  ///< first entity in the staging buffer
} sp_segment;

/** @brief What a transfer moves. Entity class and width are fixed per field. */
typedef enum sp_field
{
  SP_POS,  ///< vertex positions, 3 doubles
  SP_VNORMAL,  ///< vertex normals, 3 doubles
  SP_AREA,  ///< triangle area, 1 double
  SP_NORMAL,  ///< triangle unit normal, 3 doubles
  SP_CENTER,  ///< triangle centroid, 3 doubles
  SP_TRI_VERTS,  ///< triangle corners, 3 ints, slot-local vertex indices
  SP_VOWNER,  ///< vertex owner rank, 1 int
  SP_TOWNER,  ///< triangle owner rank, 1 int
  SP_N_FIELDS
} sp_field;

/** @brief counts[SP_N_COUNTS * slot + k]: the slot's entities. A free slot has all four zero. */
enum
{
  SP_NV,
  SP_NV_OWNED,
  SP_NT,
  SP_NT_OWNED,
  SP_N_COUNTS
};

/**
 * @brief stats[SP_N_STATS * slot + k], over the slot's OWNED triangles: the
 *        first nine values of an MFTLEnsembleUpdate() row, in its order.
 */
enum
{
  SP_N_STATS = 9  ///< min area, q, |e|^2; max area, q, |e|^2; sum area, q, |e|^2
};

/** @brief "cuda", "openacc", "openmp", "GNU-omp-host", ...: what this binary runs. */
const char *sp_backend_name(void);

/**
 * @brief Selects device `local_rank % n_devices` and initialises it.
 * @return 0, or non-zero when the backend has no device to run on.
 */
int sp_init(int local_rank);
void sp_finalize(void);

/**
 * @brief Allocates the slot arrays, uploads the box and the rank map. Once.
 * @param n_slots  Bodies the rank can hold at once.
 * @param nv_slot  Vertices per slot: the largest body.
 * @param nt_slot  Triangles per slot.
 * @param rank_map `[nproc]` MFTL_Domain::rank_map.
 */
void sp_setup(int n_slots, int nv_slot, int nt_slot, const sp_box *box, const int *rank_map, int nproc);
/** @brief Device bytes held by the slot arrays. */
double sp_device_bytes(void);

/**
 * @brief The host staging buffers: `n` entries, valid until the next call of
 *        the same function. The driver packs into them before sp_upload and
 *        unpacks from them after sp_download. Pinned memory in CUDA.
 */
double *sp_stage_double(long n);
int *sp_stage_int(long n);

/** @brief Staging -> slots: every segment, one copy. Double fields read sp_stage_double(), int fields sp_stage_int(). */
void sp_upload(sp_field field, int n_seg, const sp_segment *seg);
/**
 * @brief Slots -> staging: every segment, one copy.
 * @return The staging buffer, the segments at their offsets: `const double *`
 *         or `const int *` by field. Valid until the next staging call.
 */
const void *sp_download(sp_field field, int n_seg, const sp_segment *seg);

/** @brief Host -> device: `[SP_N_COUNTS * n_slots]` entity counts per slot. */
void sp_set_counts(const int *counts);

/** @brief Every owned vertex of every slot: x += v dt, then wrapped into the box. */
void sp_advect(double vx, double vy, double vz, double dt);

/**
 * @brief The geometric owner of every vertex (its position) and triangle (its
 *        wrapped centroid), ghosts included -- MFTLGetOwnerProc() on the device.
 *
 * The result replaces the slot's SP_VOWNER / SP_TOWNER, which therefore must
 * start out as the library's `owner_proc`.
 *
 * @param[out] changed `[n_slots]` non-zero for a slot where any owner changed.
 */
void sp_classify(int *changed);

/**
 * @brief Area, unit normal and wrapped centroid of every triangle, ghosts
 *        included, fused with the per-slot statistics over the owned ones.
 * @param[out] stats `[SP_N_STATS * n_slots]`; a slot without owned triangles
 *                   holds the identities (DBL_MAX, -DBL_MAX, 0).
 */
void sp_geometry(double *stats);

/**
 * @brief Area-weighted vertex normals of every vertex: exact on the owned
 *        vertices, partial on the ghosts until the ghost exchange.
 */
void sp_normals(void);

/** @brief Bytes copied host -> device and device -> host so far, and the number of copies. */
void sp_traffic(double *h2d, double *d2h, long *n_copies);

#ifdef __cplusplus
}
#endif

#endif
