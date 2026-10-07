/**
 * @file coupling_internal.h
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief Private types and helpers shared by the files of src/coupling/.
 *
 * Who owns what:
 *
 *   solver_grid.c          the registry: locations, blocks, point sets, coverage
 *   coupling_kernels.c     the kernel table
 *   coupling_rows.c        one row from candidates: weights, moments, quality
 *   coupling_markers.c     markers from the surface, sampling check
 *   coupling_directory.c   box and lattice geometry, the directory, requests
 *   coupler.c              the state machine: request, window, bind, transfer
 *   coupling_comm.c        EVERY MPI call of the module
 */

#ifndef MFTL_COUPLING_INTERNAL_H
#define MFTL_COUPLING_INTERNAL_H

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "mftl_coupling.h"
#include "mftl_types.h"

/** @brief Longest location name, terminating NUL included. */
#define CPL_NAME_MAX 32

/** @brief Routes of a band sample. */
enum
{
  CPL_ROUTE_SELF = 0,  ///< Owned here; read and accumulated in place.
  CPL_ROUTE_HALO = 1,  ///< Owned elsewhere, but held in a valid ghost layer here; read in place, loads go to the owner.
  CPL_ROUTE_REMOTE = 2  ///< Fetched from the owner; loads go back to it.
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Kernels (coupling_kernels.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A resolved kernel: built-in or the application's. */
typedef struct cpl_kernel
{
  MFTL_KernelFn eval;
  void *ctx;
  double half_width;
  int moment_order;
  bool radial;
  bool nonnegative;
} cpl_kernel;

/**
 * @brief Resolves the kernel of a parameter block.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR for an unknown or invalid kernel.
 */
int MFTLCplKernelResolve(const MFTL_CouplerParams *p, cpl_kernel *k);

/* ═══════════════════════════════════════════════════════════════════════════
 * Registry (solver_grid.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One owned lattice block. */
typedef struct cpl_block
{
  int location;
  int piece;  ///< Index among the location's pieces on this rank.
  int64_t lo[3], hi[3];  ///< Owned index box, half-open.
  int64_t alo[3], adims[3], stride[3];  ///< Array layout.
  int ghost[3];  ///< Valid ghost layers of a halo-valid field.
  unsigned char *active;  ///< Owned box x-fastest, #MFTL_NULL if all active.
} cpl_block;

/** @brief One registered point set, sorted into Morton tiles. */
typedef struct cpl_points
{
  int location;
  int piece;
  int64_t n;
  uint64_t *key;
  double *pos;  ///< [3n]
  double *measure;  ///< [n]
  double *h;  ///< [3n]
  double *cell;  ///< [9n] anisotropic location: cell matrices (row-major), else #MFTL_NULL
  double *ext;  ///< [3n] half-extents
  int64_t *aidx;  ///< [n] element index in the field arrays
  int64_t *orig;  ///< [n] registration index
  int n_tiles;
  int64_t *tile_off;  ///< [n_tiles+1]
  double *tile_pbox;  ///< [n_tiles][6] box of the positions
  double *tile_ebox;  ///< [n_tiles][6] box of the cell extents (ownership)
  double tile_hmin[3], tile_hmax[3];  ///< over the set
  double *tile_h;  ///< [n_tiles][6] hmin xyz, hmax xyz per tile
} cpl_points;

/** @brief One location: global definition plus this rank's coverage. */
typedef struct cpl_location
{
  char name[CPL_NAME_MAX];
  MFTL_LatticeKind kind;
  int64_t dims[3];
  double origin[3], spacing[3];
  double *coords[3];  ///< rectilinear copies
  double *faces[3];  ///< rectilinear copies
  MFTL_BoundaryPolicy boundary;
  MFTL_WallParity wall[6];  ///< MIRROR: parity per face, x-lo .. z-hi
  bool anisotropic;  ///< point samples carry cell matrices

  MFTL_GridExposure exposure;
  int n_cov;
  int64_t *cov_index;  ///< [n_cov][6]
  double *cov_boxes;  ///< [n_cov][6]
  int n_prov;
  double *prov_boxes;  ///< [n_prov][6]
  double prov_hmin[3], prov_hmax[3];  ///< declared resolution bounds, 0 if not declared

  int n_pieces;
  int cap_pieces;
  int *piece_ref;  ///< block or point-set index of each piece

  /* constrained samples this rank owns (MFTL_ConstraintDesc), registration order */
  int64_t n_con;
  uint64_t *con_key;  ///< [n_con]
  int64_t *con_ptr;  ///< [n_con+1]
  uint64_t *con_free;  ///< [con_ptr[n_con]]
  double *con_coef;
  int *con_owner;
  uint64_t *con_sorted;  ///< [n_con] keys ascending
  int64_t *con_order;  ///< [n_con] registration index of each sorted key
} cpl_location;

/** @brief The registry. */
struct MFTL_SolverGrid
{
  MPI_Comm comm;
  int n_loc, cap_loc;
  cpl_location *loc;
  /* per location, built on first use by MFTLCplSampleFind(): point keys ascending with their piece and element */
  int64_t *kidx_n;  ///< [n_loc] -1 when not built
  uint64_t **kidx_key;
  int **kidx_piece;
  int64_t **kidx_elem;
  double **kidx_meas;
  int n_blocks, cap_blocks;
  cpl_block *blocks;
  int n_points, cap_points;
  cpl_points *points;
  uint64_t version_owned;  ///< bumped by anything that changes what this rank OWNS
  uint64_t version_export;  ///< bumped by anything that changes what it EXPORTS
};

/** @brief Is location @p l a lattice? */
static inline bool cpl_is_lattice(const cpl_location *l) { return l->kind != MFTL_LATTICE_NONE; }

/** @brief Hash of every location definition, for the cross-rank check. */
uint64_t MFTLCplGridDefinitionHash(const MFTL_SolverGrid *g);

/** @brief Validates the registration against the domain (reports on stderr). */
int MFTLCplGridValidate(const MFTL_SolverGrid *g, const MFTL_Domain *d);

/** @brief The registration index of constrained key @p key of location @p l on this rank, or -1. */
int64_t MFTLCplConstraintFind(const cpl_location *l, uint64_t key);

/**
 * @brief Where this rank stores sample @p key of location @p l: its piece,
 *        array element and measure. Lattices by index arithmetic, point sets
 *        through a sorted key index built on first use.
 * @return false if this rank does not own the key.
 */
bool MFTLCplSampleFind(MFTL_SolverGrid *g, int location, uint64_t key, int *piece, int64_t *elem, double *measure);

/** @brief Is owned lattice index @p idx exported (FULL, or inside a coverage box)? */
bool MFTLCplLatExported(const cpl_location *l, const int64_t idx[3]);

/* ═══════════════════════════════════════════════════════════════════════════
 * Lattice and box geometry (coupling_directory.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Canonical coordinate of lattice index @p j, `0 <= j < dims[k]`. */
double MFTLCplLatCoord(const cpl_location *l, int k, int64_t j);

/** @brief Width (measure and resolution) of lattice index @p j along @p k. */
double MFTLCplLatWidth(const cpl_location *l, int k, int64_t j);

/**
 * @brief Position of UNWRAPPED lattice index @p i: the canonical coordinate of
 *        `i mod N` plus the period shift on a periodic axis. The one formula for
 *        image positions, used by requester and provider alike.
 */
double MFTLCplLatPos(const cpl_location *l, const MFTL_Domain *d, int k, int64_t i);

/** @brief Floor division for a positive divisor. */
static inline int64_t cpl_floordiv(int64_t a, int64_t b) { return (a >= 0) ? a / b : -((-a + b - 1) / b); }

/** @brief Non-negative modulus. */
static inline int64_t cpl_mod(int64_t a, int64_t b)
{
  const int64_t r = a % b;
  return (r < 0) ? r + b : r;
}

/**
 * @brief The unwrapped index range `[a, b)` of samples along axis @p k whose
 *        positions lie in the closed interval `[lo, hi]`; clipped to the
 *        lattice on a non-periodic axis. Returns false when empty.
 */
bool MFTLCplLatRange(const cpl_location *l, const MFTL_Domain *d, int k, double lo, double hi, int64_t *a, int64_t *b);

/** @brief Distinct wrapped indices in `[a, b)` along axis @p k. */
int64_t MFTLCplLatDistinct(const cpl_location *l, const MFTL_Domain *d, int k, int64_t a, int64_t b);

/**
 * @brief Up to @p max wrapped intervals `[lo_out, hi_out)` covering the
 *        indices of `[a, b)` that fall in the owned range `[olo, ohi)` on axis
 *        @p k (images enumerated on a periodic axis). Returns the count.
 */
int MFTLCplLatIntersect(const cpl_location *l, const MFTL_Domain *d, int k, int64_t a, int64_t b, int64_t olo, int64_t ohi, int64_t *lo_out, int64_t *hi_out,
                      int max);

/**
 * @brief Wraps a position into `[min, max)` on every periodic axis, however
 *        many periods away it is.
 *
 * MFTLApplyPeriodicity() shifts by at most one period, which is all a mesh
 * wrapped every step needs -- but an application's marker can be anywhere, and
 * one left two periods out is binned, requested and bound in a frame no
 * provider answers, which comes back as a false NEED_COVERAGE. The seam case
 * is the other half: one ulp below `min` plus a period rounds onto `max`
 * itself, outside the half-open box; such a position is put on `min`, the
 * same point of the circle.
 */
static inline void cpl_wrap(const MFTL_Domain *d, double x[3])
{
  for (int k = 0; k < 3; ++k)
  {
    if (!d->is_periodic[k] || (x[k] >= d->global_min[k] && x[k] < d->global_max[k])) continue;
    double t = x[k] - d->global_min[k];
    t -= d->domain_size[k] * floor(t * d->inv_domain_size[k]);
    double y = d->global_min[k] + t;
    if (!(y < d->global_max[k]) || y < d->global_min[k]) y = d->global_min[k];
    x[k] = y;
  }
}

/** @brief Lattice key of wrapped index (i,j,k): `i + N0 (j + N1 k)`. */
static inline uint64_t cpl_lat_key(const cpl_location *l, int64_t i, int64_t j, int64_t k) { return (uint64_t) (i + l->dims[0] * (j + l->dims[1] * k)); }

/** @brief Inverse of cpl_lat_key(). */
static inline void cpl_lat_unkey(const cpl_location *l, uint64_t key, int64_t idx[3])
{
  const int64_t kk = (int64_t) key;
  idx[0] = kk % l->dims[0];
  idx[1] = (kk / l->dims[0]) % l->dims[1];
  idx[2] = kk / (l->dims[0] * l->dims[1]);
}

/** @brief Element offset of lattice index @p idx (unwrapped, inside the array) in a block's arrays. */
static inline int64_t cpl_block_elem(const cpl_block *b, const int64_t idx[3])
{
  return (idx[0] - b->alo[0]) * b->stride[0] + (idx[1] - b->alo[1]) * b->stride[1] + (idx[2] - b->alo[2]) * b->stride[2];
}

/** @brief Do closed boxes `[alo, ahi]` and `[blo, bhi]` intersect? */
static inline bool cpl_box_overlap(const double *alo, const double *ahi, const double *blo, const double *bhi)
{
  return alo[0] <= bhi[0] && blo[0] <= ahi[0] && alo[1] <= bhi[1] && blo[1] <= ahi[1] && alo[2] <= bhi[2] && blo[2] <= ahi[2];
}

/**
 * @brief Subtracts closed boxes from a list of closed boxes, in place.
 *
 * @p rem holds `*n_rem` boxes `[6]` with room for @p cap; each subtrahend
 * splits every box it cuts into at most six pieces. Pieces thinner than
 * @p sliver on an originally nondegenerate axis are dropped: they are planes
 * the subtrahend's closed face already covers, or floating-point seams between
 * abutting regions. Initially degenerate boxes are subtracted in their own
 * dimensions, so a certified plane, line or point is removed as well.
 * @return false if @p cap was exceeded (the result is then incomplete).
 */
bool MFTLCplBoxSubtract(double *rem, int *n_rem, int cap, const double *sub, const double sliver[3]);

/** @brief Integer index-box subtraction of half-open boxes; same contract. */
bool MFTLCplIboxSubtract(int64_t *rem, int *n_rem, int cap, const int64_t *sub);

/**
 * @brief The ranks whose boxes intersect the closed box `[lo, hi]`, periodic
 *        images included, deduplicated and sorted. Arithmetic on the process
 *        grid; never a loop over the ranks.
 * @return The number of such ranks. When it exceeds @p cap nothing is written:
 *         grow @p out to the returned count and call again (a caller sizes its
 *         buffer by what boxes actually touch, never by the process count).
 */
int MFTLCplRanksTouching(const MFTL_Domain *d, const double lo[3], const double hi[3], int *out, int cap);


/* ═══════════════════════════════════════════════════════════════════════════
 * Directory (coupling_directory.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief What one provider owns of one location, as the directory knows it. */
typedef struct cpl_dir_entry
{
  int provider;
  int location;
  int64_t ilo[3], ihi[3];  ///< lattice: owned index box
  double lo[3], hi[3];  ///< physical extent (lattice: positions +- half a width; points: ownership box)
  double hmin[3], hmax[3];  ///< resolution bounds inside
} cpl_dir_entry;

/**
 * @brief A uniform grid over the directory entries of one location, so a
 *        request box meets only the entries near it rather than all of them.
 */
typedef struct cpl_dir_index
{
  double lo[3], cell[3];
  int64_t dims[3];
  int64_t n_cells;
  int64_t *off;  ///< [n_cells+1] CSR
  int *ent;  ///< entry indices
  int *stamp;  ///< [n_dir] dedup stamps
  int stamp_now;
} cpl_dir_index;

/** @brief Builds the index of the entries of @p location (entries need not be sorted). */
void MFTLCplDirIndexBuild(cpl_dir_index *ix, const cpl_dir_entry *e, int n, int location);

/** @brief Frees an index. */
void MFTLCplDirIndexFree(cpl_dir_index *ix);

/**
 * @brief The entries whose (unshifted) box meets the closed box `[lo, hi]`,
 *        each once, appended to @p out (grown). Returns the count.
 */
int MFTLCplDirIndexQuery(cpl_dir_index *ix, const cpl_dir_entry *e, const double lo[3], const double hi[3], int **out, int *cap);

/**
 * @brief Which part of a region the directory's ownership boxes of one point
 *        location cover: a grid whose cells are either fully covered or carry
 *        the fragments nobody claims. Built once per directory, so a request
 *        box's "is every part of me owned by someone" costs cell lookups, not
 *        a subtraction of every overlapping ownership box.
 */
typedef struct cpl_cover
{
  double lo[3], cell[3];
  int64_t dims[3];
  int64_t *frag_off;  ///< [n_cells+1] CSR into frag; an empty range is a fully covered cell
  double *frag;  ///< [n][6] uncovered fragments
  bool built;
} cpl_cover;

/**
 * @brief Builds the cover of the closed box `[lo, hi]` (periodic images of the
 *        entries included) at a cell size of about @p cell.
 */
void MFTLCplCoverBuild(cpl_cover *cv, cpl_dir_index *ix, const cpl_dir_entry *e, const MFTL_Domain *d, const double lo[3], const double hi[3],
                       const double cell[3], const double sliver[3]);

/** @brief Frees a cover. */
void MFTLCplCoverFree(cpl_cover *cv);

/**
 * @brief The parts of the closed box `[lo, hi]` no ownership box covers,
 *        appended to @p out as boxes (grown). The box must lie inside the
 *        cover's region (the directory reach guarantees it); a part outside is
 *        reported as uncovered. Returns the number of boxes appended.
 */
int MFTLCplCoverQuery(const cpl_cover *cv, const double lo[3], const double hi[3], const double sliver[3], double **out, int *n_out, int *cap);

/**
 * @brief This rank's own directory entries, from the registry.
 * @return Number of entries; @p out is allocated (free it).
 */
int MFTLCplDirectoryOwnEntries(const MFTL_SolverGrid *g, const MFTL_Domain *d, cpl_dir_entry **out);

/* ═══════════════════════════════════════════════════════════════════════════
 * Rows (coupling_rows.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Scratch for row assembly, grown on demand and reused. */
typedef struct cpl_row_scratch
{
  int cap;
  double *d;  ///< [3 cap] dimensionless offsets
  double *phi;  ///< [3 cap] per-axis kernel values / radial
  double *psi;  ///< [cap] base weights
  double *r;  ///< [cap] argument scratch
} cpl_row_scratch;

/** @brief Frees row scratch. */
void MFTLCplRowScratchFree(cpl_row_scratch *s);

/**
 * @brief The numerical core of a row: base weights over given candidates at a
 *        given metric (no growth), then the reproduction policy.
 *
 * @param p     Policy.
 * @param kern  Kernel.
 * @param x     Marker position.
 * @param Ht    Kernel metric, scale and growth applied.
 * @param n     Number of candidates.
 * @param X     `[3n]` candidate positions.
 * @param q     `[n]` candidate measures.
 * @param[out] w     `[n]` weights.
 * @param[out] q_out Quality (n_grow and metric left to the caller).
 * @param s     Scratch, grown on demand.
 * @return #MFTL_COUPLING_OK or #MFTL_COUPLING_BAD_STENCIL.
 */
int MFTLCplRowWeights(const MFTL_CouplerParams *p, const cpl_kernel *kern, const double x[3], const double Ht[3], int n, const double *X, const double *q,
                    double *w, MFTL_RowQuality *q_out, cpl_row_scratch *s);

/**
 * @brief The same with a full CELL matrix @p Hm (row-major; columns the cell's
 *        edge vectors, scale and growth applied): offsets \f$H^{-1}(X - x)\f$,
 *        base weights normalised by \f$|\det H|\f$. A tensor kernel then
 *        follows the cell's own axes.
 */
int MFTLCplRowWeightsCell(const MFTL_CouplerParams *p, const cpl_kernel *kern, const double x[3], const double Hm[9], int n, const double *X, const double *q,
                        double *w, MFTL_RowQuality *q_out, cpl_row_scratch *s);

/** @brief Inverse and determinant of a cell matrix; false when singular relative to its column norms. */
bool MFTLCplCellInverse(const double Hm[9], double inv[9], double *det);

/* ═══════════════════════════════════════════════════════════════════════════
 * Coupler state (coupler.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief What the coupler knows globally about one location. */
typedef struct cpl_locinfo
{
  bool lattice;
  bool constrained;  ///< some rank registered constrained samples of this location
  int *con_peer;  ///< local constraint entries' owners, translated from solver ranks to coupler ranks
  bool anisotropic;  ///< point samples carry cell matrices
  bool uniform_metric;  ///< every sample of the location has the same resolution
  double hmin[3];  ///< global minimum resolution
  double hmax[3];  ///< global maximum resolution (the metric bound)
  double reach[3];  ///< the largest distance a row can look from its marker
} cpl_locinfo;

/** @brief One rank's band of one location in a window. */
typedef struct cpl_band
{
  int location;
  bool has_masked;  ///< some request box of this band held a masked (solid) sample
  int64_t n_dof;
  uint64_t *key;
  int *owner;
  unsigned char *route;
  int *piece;  ///< SELF/HALO: local piece
  int64_t *elem;  ///< SELF/HALO: element offset
  double *measure;
  double *h;  ///< [3 n_dof]
  double *cell;  ///< [9 n_dof] anisotropic point location: cell matrices, else #MFTL_NULL

  int64_t n_img;
  double *ipos;  ///< [3 n_img]
  int64_t *idof;  ///< [n_img]
  int64_t n_ibins;
  uint64_t *ibin_key;
  int64_t *ibin_off;  ///< [n_ibins+1]
  double bsize[3];

  /* requester side: one segment of DOFs per provider (self included) */
  int n_src;
  int *src_rank;
  int64_t *src_off, *src_nonhalo, *src_halo;
  int self_seg;  ///< index of this rank's segment, -1 if none

  /* provider side: one pack list per requester (self included) */
  int n_dst;
  int *dst_rank;
  int64_t *dst_off, *dst_nonhalo, *dst_halo;
  int64_t *pack_served;  ///< [total pack] index into the served table
  int64_t n_served;
  int *served_piece;
  int64_t *served_elem;
  int64_t *served_sample;  ///< owned-box linear index (blocks) or registration index (points)
  double *served_measure;
  int64_t *served_con;  ///< constrained location: registration index of each served sample's constraint, -1 if free; else #MFTL_NULL

  /* REQUESTED mask support: the index boxes / boxes received */
  int n_reqbox;
  double *reqbox;  ///< [n][6]: lattice: index box as doubles; points: physical box
} cpl_band;

struct MFTL_CouplingWindow
{
  MFTL_Coupler *c;
  uint64_t id;
  uint64_t grid_epoch;
  uint64_t export_epoch;  ///< the export epoch at which every served sample was last confirmed exported
  int n_loc;
  int *locs;
  cpl_band *band;
  double skin[3];
  double bin[3];
  int64_t n_bins;
  uint64_t *bin_key;
  int n_fetch_peers;
  int *fetch_peer;  ///< ranks this rank receives fetch values from (sorted, self excluded)
  int n_serve_peers;
  int *serve_peer;  ///< ranks this rank sends fetch values to (sorted, self excluded)
  int refcount;
  size_t bytes;
};

/** @brief Per-body snapshot taken by a bind. */
typedef struct cpl_body_snap
{
  int gid;
  uint64_t generation;
  int n_local_v, n_local_t;
} cpl_body_snap;

struct MFTL_CouplingBinding
{
  MFTL_Coupler *c;
  const MFTL_CouplingWindow *win;
  const MFTL_CouplingBinding *base;
  uint64_t window_id;
  int refcount;
  MFTL_MarkerKind kind;
  int64_t n_markers;
  MFTL_MarkerRef *ref;
  double *pos;  ///< [3n] row positions (frame of the window bin)
  double *base_pos;  ///< [3n] positions of the base markers (equal to pos for a base binding)
  int n_loc;
  int *loc_band;  ///< window band index per location
  int64_t **row_ptr;
  int64_t **col;
  double **w;
  double **wallc;  ///< [n_loc] MIRROR locations: [6 n_markers] wall coefficients; #MFTL_NULL otherwise
  int n_bodies;
  cpl_body_snap *bodies;
  uint64_t pos_checksum;
  MFTL_BindingQuality q;
  uint64_t *scratch_keys;
  int scratch_cap;
  struct cpl_quad *quad;  ///< QUADRATURE: the vertex table and ghost schedule (the base's, shared by its stage bindings)
  bool quad_owned;  ///< this binding frees @c quad
  int q_level, q_max;  ///< QUADRATURE: the spec this binding was gathered with
  double q_ratio;
};

struct MFTL_CouplingFetch
{
  const MFTL_CouplingWindow *win;
  uint64_t window_id;
  int n_fields;
  int *field_band;
  double **values;
  uint64_t *version;
  double *wall;  ///< [6 n_fields] each field's wall values (zero where none was given)
};

struct MFTL_TransferReady
{
  MFTL_Coupler *c;
  uint64_t id;  ///< Agreed preparation sequence, checked again at Begin.
  bool retained;  ///< Window/binding references acquired after successful preparation.
  MFTL_TransferDirection direction;
  const MFTL_CouplingWindow *win;
  const MFTL_CouplingBinding *binding;
  uint64_t window_id;
  int n_fields;
  MFTL_SolverField *fields;
  int *field_band;
  MFTL_SpreadForm form;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * Markers (coupling_markers.c)
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Markers gathered from the surface or the caller. */
typedef struct cpl_markers
{
  int64_t n;
  double *pos;  ///< [3n] wrapped
  MFTL_MarkerRef *ref;
  double *bary;  ///< QUADRATURE: [2n] barycentric coordinates (lambda1, lambda2) in the triangle's vertex order
  int n_bodies;
  cpl_body_snap *bodies;
} cpl_markers;

/**
 * @brief Gathers markers.
 * @param ens    The ensemble.
 * @param spec   What to gather.
 * @param levels QUADRATURE: points per edge of every owned triangle, bodies in
 *               list order then pool order; ignored otherwise.
 * @param m      Receives the markers.
 * @return #MFTL_COUPLING_OK or #MFTL_COUPLING_INVALID.
 */
int MFTLCplMarkersGather(const MFTL_Ensemble *ens, const MFTL_MarkerSpec *spec, const int *levels, cpl_markers *m);

/** @brief The barycentric coordinates (lambda1, lambda2) of sub-centroid @p q of a triangle cut @p m per edge. */
void MFTLCplQuadraturePoint(int m, int q, double *l1, double *l2);

/** @brief Frees what MFTLCplMarkersGather() allocated. */
void MFTLCplMarkersFree(cpl_markers *m);

/**
 * @brief The vertex side of a QUADRATURE binding: which vertex every point's
 *        three barycentric coordinates belong to, and the ghost schedule that
 *        completes per-vertex sums across ranks.
 *
 * The VERTEX TABLE is this rank's owned vertices in VERTICES marker order
 * (bodies in list order, then pool order), followed by the distinct GHOST
 * vertices the owned triangles touch, sorted by (owner rank, body id, vertex
 * id). Ghosts of one owner are therefore contiguous, and both sides of the
 * exchange know the order without sending it again.
 */
typedef struct cpl_quad
{
  uint64_t id;  ///< collective map identity, shared by local stage bindings
  int64_t n_own, n_ghost;
  int64_t *corner;  ///< [3 n_markers] vertex-table index of each point's three corners
  double *bary;  ///< [2 n_markers] (lambda1, lambda2)
  int n_owner;  ///< ranks owning this rank's ghosts
  int *owner_rank;  ///< [n_owner] ascending
  int64_t *owner_off;  ///< [n_owner + 1] into the ghost part of the table
  int n_req;  ///< ranks holding this rank's owned vertices as ghosts of their triangles
  int *req_rank;  ///< [n_req] ascending
  int64_t *req_off;  ///< [n_req + 1]
  int64_t *req_vt;  ///< owned vertex-table index of every requested vertex, requester by requester
} cpl_quad;

/**
 * @brief Builds the vertex table and the ghost schedule of QUADRATURE markers
 *        (one NBX). Collective over @p comm -- every rank calls it, with or
 *        without markers.
 * @return #MFTL_COUPLING_OK, or #MFTL_COUPLING_INVALID when a requested ghost
 *         is not owned where its record says (a local verdict, to be agreed).
 */
int MFTLCplQuadBuild(MPI_Comm comm, const MFTL_Ensemble *ens, const cpl_markers *m, cpl_quad **out, long *bytes, long *messages);

/** @brief Frees a vertex table and schedule. */
void MFTLCplQuadFree(cpl_quad **q);

/**
 * @brief Per-vertex sums of point contributions, completed across ranks:
 *        `acc[v][k] = sum_q w_q lambda_qv val[q][k]` for `k < ncomp`, and
 *        `acc[v][ncomp] = sum_q w_q lambda_qv` (the lumped mass), for the
 *        OWNED vertices. Collective.
 * @param comm      The coupler's private communicator.
 * @param q         The binding's vertex table and ghost schedule.
 * @param n_markers Points of the binding on this rank.
 * @param ref       `[n_markers]` the points (their weights are used).
 * @param ncomp     Components per value; 0 for the mass alone.
 * @param val       `[n_markers * ncomp]` point values; #MFTL_NULL when @p ncomp is 0.
 * @param[out] acc  `[n_own * (ncomp + 1)]` the completed sums.
 * @param[in,out] bytes    Payload bytes moved, added to.
 * @param[in,out] messages Messages sent, added to.
 */
void MFTLCplQuadAccumulate(MPI_Comm comm, const cpl_quad *q, int64_t n_markers, const MFTL_MarkerRef *ref, int ncomp, const double *val, double *acc,
                           long *bytes, long *messages);

/**
 * @brief Point values interpolated from vertex values: `out[q][k] = w_q *
 *        sum_c lambda_qc g[v_c][k]` (with @p weighted) or without the weight.
 *        The owned vertices' @p g travel to the ranks holding them as ghosts.
 *        Collective.
 * @param comm      The coupler's private communicator.
 * @param q         The binding's vertex table and ghost schedule.
 * @param n_markers Points of the binding on this rank.
 * @param ref       `[n_markers]` the points (their weights are used when @p weighted).
 * @param weighted  Multiply by the point's weight (a load) or not (a value).
 * @param ncomp     Components per value.
 * @param g         `[n_own * ncomp]` owned-vertex values.
 * @param[out] out  `[n_markers * ncomp]` point values.
 * @param[in,out] bytes    Payload bytes moved, added to.
 * @param[in,out] messages Messages sent, added to.
 */
void MFTLCplQuadInterpolate(MPI_Comm comm, const cpl_quad *q, int64_t n_markers, const MFTL_MarkerRef *ref, bool weighted, int ncomp, const double *g,
                            double *out, long *bytes, long *messages);

/** @brief FNV-1a over the bits of @p n positions. */
uint64_t MFTLCplPositionsChecksum(const double *pos, int64_t n);

/**
 * @brief Whether a binding's bodies still are what they were: same bodies,
 *        same generations, same owned counts.
 */
bool MFTLCplBodiesCurrent(const MFTL_Ensemble *ens, const cpl_body_snap *b, int n);

/* ═══════════════════════════════════════════════════════════════════════════
 * Communication (coupling_comm.c) -- the only file with MPI calls
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief One received message. */
typedef struct cpl_msg
{
  int peer;
  size_t bytes;
  unsigned char *data;
} cpl_msg;

/** @brief Message tags; the communicator is private to the coupler. */
enum
{
  CPL_TAG_DIRECTORY = 101,
  CPL_TAG_REQUEST = 102,
  CPL_TAG_REPLY = 103,
  CPL_TAG_FETCH = 104,
  CPL_TAG_SPREAD = 105,
  CPL_TAG_QUAD_SETUP = 106,
  CPL_TAG_QUAD = 107,
  CPL_TAG_CONSTRAINT_SETUP = 108,
  CPL_TAG_CONSTRAINT = 109,
  CPL_TAG_MIGRATION = 110
};

/** @brief Duplicates @p base. */
int MFTLCplCommDup(MPI_Comm base, MPI_Comm *out);
/** @brief Frees a duplicate. */
void MFTLCplCommFree(MPI_Comm *c);
/** @brief Rank and size. */
void MFTLCplCommRankSize(MPI_Comm c, int *rank, int *size);
/** @brief Whether @p a and @p b hold the same group of processes (any order). */
bool MFTLCplCommSameGroup(MPI_Comm a, MPI_Comm b);
/** @brief Translates only the supplied peer ranks between communicators; rejects out-of-range ranks. Local. */
bool MFTLCplTranslateRanks(MPI_Comm from, MPI_Comm to, int64_t n, const int *ranks, int *translated);
/** @brief The maximum of @p v over the ranks. */
int MFTLCplAgreeMax(MPI_Comm c, int v);

/** @brief Agrees status and token identity before a transfer posts messages. */
int MFTLCplAgreeTransfer(MPI_Comm c, int status, uint64_t token);
/** @brief In-place MAX of @p n ints. */
void MFTLCplAgreeMaxN(MPI_Comm c, int *v, int n);
/** @brief Whether @p h is the same on every rank (one reduction of two values). */
bool MFTLCplAgreeHash(MPI_Comm c, uint64_t h);
/** @brief In-place MAX of @p n doubles. */
void MFTLCplAllreduceMax(MPI_Comm c, double *v, int n);
/** @brief In-place SUM of @p n doubles. */
void MFTLCplAllreduceSum(MPI_Comm c, double *v, int n);
/** @brief In-place SUM of @p n longs. */
void MFTLCplAllreduceSumLong(MPI_Comm c, long *v, int n);
/** @brief In-place MAX of @p n longs. */
void MFTLCplAllreduceMaxLong(MPI_Comm c, long *v, int n);
/** @brief The rank holding the global maximum of @p v (lowest on ties), and the maximum. */
void MFTLCplMaxloc(MPI_Comm c, double v, double *vmax, int *rank);
/** @brief Broadcasts @p n ints from @p root. */
void MFTLCplBcastInt(MPI_Comm c, int *v, int n, int root);

/**
 * @brief Sparse exchange to destinations the receivers do not know (NBX):
 *        synchronous sends, probe-and-receive, a non-blocking barrier once this
 *        rank's sends have been matched. Messages to self are not allowed.
 * @param c        The coupler's private communicator.
 * @param tag      The phase's tag.
 * @param n_out    Number of destinations.
 * @param dest     `[n_out]` destination ranks.
 * @param sbuf     `[n_out]` payloads.
 * @param sbytes   `[n_out]` payload sizes.
 * @param[out] in    The received messages, sorted by source rank (allocated).
 * @param[out] n_in  Their number.
 * @param[in,out] bytes    Payload bytes moved, added to.
 * @param[in,out] messages Messages sent, added to.
 * @return #MFTL_SUCCESS.
 */
int MFTLCplNbx(MPI_Comm c, int tag, int n_out, const int *dest, unsigned char *const *sbuf, const size_t *sbytes, cpl_msg **in, int *n_in, long *bytes,
            long *messages);

/**
 * @brief Exchange with known peers, sizes unknown to the receiver: one message
 *        to each of @p dest, one from each of @p src (both sorted), receiving
 *        in source order.
 * @param c        The coupler's private communicator.
 * @param tag      The phase's tag.
 * @param n_out    Number of destinations.
 * @param dest     `[n_out]` destination ranks.
 * @param sbuf     `[n_out]` payloads.
 * @param sbytes   `[n_out]` payload sizes.
 * @param n_src    Number of sources.
 * @param src      `[n_src]` source ranks.
 * @param[out] in  `[n_src]` messages, in the order of @p src.
 * @param[in,out] bytes    Payload bytes moved, added to.
 * @param[in,out] messages Messages sent, added to.
 * @return #MFTL_SUCCESS.
 */
int MFTLCplExchangeKnown(MPI_Comm c, int tag, int n_out, const int *dest, unsigned char *const *sbuf, const size_t *sbytes, int n_src, const int *src,
                       cpl_msg *in, long *bytes, long *messages);

/** @brief Frees received messages. */
void MFTLCplMsgsFree(cpl_msg *m, int n);

/** @brief An exchange with sizes known on both sides, split in Begin/End. */
typedef struct cpl_sched
{
  int n_req;
  int cap_req;
  void *req;  ///< MPI_Request array
} cpl_sched;

/** @brief Posts the receive of @p bytes from @p peer into @p buf (chunked). */
void MFTLCplSchedRecv(MPI_Comm c, cpl_sched *s, int tag, int peer, void *buf, size_t bytes);
/** @brief Posts the send of @p bytes to @p peer from @p buf (chunked). */
void MFTLCplSchedSend(MPI_Comm c, cpl_sched *s, int tag, int peer, const void *buf, size_t bytes, long *messages);
/** @brief Completes every posted operation. */
void MFTLCplSchedWait(cpl_sched *s);
/** @brief Frees the request array. */
void MFTLCplSchedFree(cpl_sched *s);

/** @brief Wall clock. */
double MFTLCplWtime(void);

/** @brief Cartesian cell of @p rank in the ensemble's process grid (no communication). */
void MFTLCplCartCoords(const MFTL_Domain *d, int rank, int coords[3]);

/** @brief The current chunk limit in bytes. */
size_t MFTLCplChunkLimit(void);

/* ═══════════════════════════════════════════════════════════════════════════
 * Small helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief FNV-1a, folding @p n bytes into @p h. */
static inline uint64_t cpl_fnv(uint64_t h, const void *data, size_t n)
{
  const unsigned char *p = (const unsigned char *) data;
  for (size_t i = 0; i < n; ++i)
  {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

/** @brief The FNV-1a offset basis. */
#define CPL_FNV_INIT 0xcbf29ce484222325ull

/** @brief A growable byte buffer for message packing. */
typedef struct cpl_buf
{
  unsigned char *data;
  size_t n, cap;
} cpl_buf;

/** @brief Appends @p n bytes. */
void MFTLCplBufPut(cpl_buf *b, const void *src, size_t n);

/** @brief Reads @p n bytes at `*off`, advancing it; false past the end. */
static inline bool cpl_get(const unsigned char *data, size_t bytes, size_t *off, void *dst, size_t n)
{
  if (*off + n > bytes) return false;
  for (size_t i = 0; i < n; ++i) ((unsigned char *) dst)[i] = data[*off + i];
  *off += n;
  return true;
}

#endif  // MFTL_COUPLING_INTERNAL_H
