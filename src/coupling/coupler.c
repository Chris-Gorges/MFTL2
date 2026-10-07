/**
 * @file coupler.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief The coupler's state machine: directory, support request, window,
 *        binding, transfers. No MPI here -- every exchange goes through
 *        coupling_comm.c.
 *
 *     create/update ──> directory (NBX publication of ownership extents)
 *     request       ──> bins of marker positions, request boxes per location,
 *                       sub-requests to every owner the directory names (NBX),
 *                       replies (known peers), coverage verified, status agreed
 *     window        ──> band (unique samples, routes), images and their bins,
 *                       forward pack lists and the served table (no messages)
 *     bind / stage  ──> rows over the band, quality, status agreed
 *     fetch         ──> owners pack, peers unpack; SELF and HALO read in place
 *     interpolate   ──> U = W u                        (local)
 *     spread        ──> b = W^T F, reverse to owners, accumulated in rank order
 *
 * Two invariants carry the correctness argument and each is checked where it
 * is established: every sample a row could touch was certified present
 * (request), and every transpose contribution reaches exactly the one owner of
 * its sample, once (window: pack lists are unique per requester and key).
 */

#include "coupling_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Small utilities
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLCplBufPut(cpl_buf *b, const void *src, size_t n)
{
  if (b->n + n > b->cap)
  {
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->n + n) cap *= 2;
    b->data = (unsigned char *) MFTLReallocOrDie(b->data, cap, "coupling buffer");
    b->cap = cap;
  }
  if (n > 0) memcpy(b->data + b->n, src, n);
  b->n += n;
}

#define PUT(buf, v) MFTLCplBufPut((buf), &(v), sizeof(v))
#define GET(data, bytes, off, v) cpl_get((data), (bytes), (off), &(v), sizeof(v))

/** @brief Grows an array of @p type to hold @p need elements. */
#define GROW(ptr, cap, need, type)                                                         \
  do                                                                                       \
  {                                                                                        \
    if ((int64_t) (need) > (int64_t) (cap))                                                \
    {                                                                                      \
      int64_t _c = (cap) ? (cap) : 16;                                                     \
      while (_c < (int64_t) (need)) _c *= 2;                                               \
      (ptr) = (type *) MFTLReallocOrDie((ptr), (size_t) _c * sizeof(type), "coupling"); \
      (cap) = _c;                                                                          \
    }                                                                                      \
  } while (0)

/** @brief A 63-bit bin key from three signed bin indices. */
static inline uint64_t bin_key3(int64_t x, int64_t y, int64_t z)
{
  const int64_t off = INT64_C(1) << 20;
  return ((uint64_t) (x + off) << 42) | ((uint64_t) (y + off) << 21) | (uint64_t) (z + off);
}

static int64_t find_u64(const uint64_t *a, int64_t n, uint64_t key)
{
  int64_t lo = 0, hi = n;
  while (lo < hi)
  {
    const int64_t m = lo + (hi - lo) / 2;
    if (a[m] < key)
      lo = m + 1;
    else
      hi = m;
  }
  return (lo < n && a[lo] == key) ? lo : -1;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The coupler
 * ═══════════════════════════════════════════════════════════════════════════ */

struct MFTL_Coupler
{
  MFTL_Ensemble *ens;
  MFTL_SolverGrid *grid;
  MFTL_CouplerParams params;
  cpl_kernel kernel;
  MPI_Comm comm;
  int rank, nproc;
  int n_loc;
  cpl_locinfo *li;

  int n_dir;
  cpl_dir_entry *dir;
  cpl_dir_index *dix;  ///< [n_loc] grids over the directory, one per location
  cpl_cover *cover;  ///< [n_loc] what the directory's ownership boxes cover near this rank (point locations)
  double dir_reach[3];

  uint64_t seen_owned, seen_export;
  uint64_t grid_epoch;
  uint64_t window_counter;
  uint64_t transfer_counter;

  int active;  ///< 0 none, 1 fetch, 2 spread
  cpl_sched sched;
  const MFTL_TransferReady *active_ready;
  MFTL_CouplingFetch *active_fetch;
  double *sbuf, *rbuf;
  int64_t scap, rcap;
  double **spread_scratch;
  int64_t *spread_scratch_cap;
  int spread_scratch_n;

  cpl_row_scratch rs;
  MFTL_CouplerStats stats;
  long live;  ///< windows + bindings + fetches + transfer tokens alive
};

/** @brief Status agreement: the maximum over the ranks. */
static int agree(MFTL_Coupler *c, int local) { return MFTLCplAgreeMax(c->comm, local); }

/** @brief Registration edits invalidate consumers even before UpdateGrid. */
static bool grid_current(const MFTL_Coupler *c)
{
  return c->grid->version_owned == c->seen_owned && c->grid->version_export == c->seen_export;
}

static bool window_current(const MFTL_CouplingWindow *w)
{
  return grid_current(w->c) && w->grid_epoch == w->c->grid_epoch;
}

/** @brief Physical box of this rank's cell. */
static void own_box(const MFTL_Coupler *c, double lo[3], double hi[3])
{
  const MFTL_Domain *d = &c->ens->domain;
  int co[3];
  MFTLCplCartCoords(d, c->rank, co);
  for (int k = 0; k < 3; ++k)
  {
    const double w = d->domain_size[k] / (double) d->cuts[k];
    lo[k] = d->global_min[k] + (double) co[k] * w;
    hi[k] = (co[k] == d->cuts[k] - 1) ? d->global_max[k] : d->global_min[k] + (double) (co[k] + 1) * w;
  }
}

/** @brief Hash of the parameter fields every rank must share. */
static uint64_t params_hash(const MFTL_CouplerParams *p)
{
  uint64_t h = CPL_FNV_INIT;
  const int ints[4] = {(int) p->kernel, (int) p->reproduction, p->failure, (int) p->sampling};
  h = cpl_fnv(h, ints, sizeof ints);
  const double ds[10] = {p->max_growth, p->growth_step, p->support_scale, p->pilot_factor, p->max_condition, p->max_amplification, p->moment_tolerance,
                         p->sampling_ratio, p->directory_margin, (p->kernel == MFTL_KERNEL_USER) ? p->user_kernel.half_width : 0.0};
  h = cpl_fnv(h, ds, sizeof ds);
  if (p->kernel == MFTL_KERNEL_USER)
  {
    const int u[3] = {p->user_kernel.moment_order, p->user_kernel.radial ? 1 : 0, p->user_kernel.nonnegative ? 1 : 0};
    h = cpl_fnv(h, u, sizeof u);
  }
  return h;
}

static bool params_valid(const MFTL_CouplerParams *p)
{
  return p->max_growth >= 1.0 && isfinite(p->max_growth) && p->growth_step > 1.0 && isfinite(p->growth_step) && p->support_scale > 0.0 &&
         isfinite(p->support_scale) && p->pilot_factor > 0.0 && isfinite(p->pilot_factor) && p->max_condition >= 1.0 && p->max_amplification >= 1.0 &&
         p->moment_tolerance > 0.0 && p->moment_tolerance < 1.0 && p->sampling_ratio > 0.0 && p->directory_margin >= 0.0 &&
         isfinite(p->directory_margin) && (p->reproduction == MFTL_REPRODUCE_LINEAR || p->reproduction == MFTL_REPRODUCE_CONSTANT ||
                                           p->reproduction == MFTL_REPRODUCE_NONE) &&
         (p->failure & ~3) == 0 && (p->sampling == MFTL_SAMPLING_OFF || p->sampling == MFTL_SAMPLING_REPORT || p->sampling == MFTL_SAMPLING_ERROR);
}

/**
 * @brief Global per-location facts: resolution bounds, uniform metric, reach.
 *        One MAX reduction of 6 doubles per location.
 */
static void compute_locinfo(MFTL_Coupler *c)
{
  const MFTL_SolverGrid *g = c->grid;
  const int nl = g->n_loc;
  double *v = (double *) MFTLMallocOrDie(6 * (size_t) MFTL_MAX(nl, 1) * sizeof(double), "locinfo");
  for (int l = 0; l < nl; ++l)
  {
    const cpl_location *L = &g->loc[l];
    for (int k = 0; k < 3; ++k)
    {
      double hmin = MFTL_DBL_MAX, hmax = 0.0;
      if (cpl_is_lattice(L))
      {
        for (int64_t j = 0; j < L->dims[k]; ++j)
        {
          const double w = MFTLCplLatWidth(L, k, j);
          hmin = fmin(hmin, w);
          hmax = fmax(hmax, w);
          if (L->kind == MFTL_LATTICE_UNIFORM) break;
        }
      }
      else
      {
        for (int pc = 0; pc < L->n_pieces; ++pc)
        {
          const cpl_points *p = &g->points[L->piece_ref[pc]];
          if (p->n == 0) continue;
          hmin = fmin(hmin, p->tile_hmin[k]);
          hmax = fmax(hmax, p->tile_hmax[k]);
        }
        if (L->prov_hmin[k] > 0.0) hmin = fmin(hmin, L->prov_hmin[k]);
        if (L->prov_hmax[k] > 0.0) hmax = fmax(hmax, L->prov_hmax[k]);
      }
      v[6 * l + k] = hmax;
      v[6 * l + 3 + k] = -hmin;
    }
  }
  MFTLCplAllreduceMax(c->comm, v, 6 * nl);

  const double gmax = (c->params.failure & MFTL_STENCIL_GROW) ? c->params.max_growth : 1.0;
  for (int l = 0; l < nl; ++l)
  {
    cpl_locinfo *li = &c->li[l];
    li->lattice = cpl_is_lattice(&g->loc[l]);
    li->uniform_metric = true;
    for (int k = 0; k < 3; ++k)
    {
      li->hmax[k] = v[6 * l + k];
      li->hmin[k] = -v[6 * l + 3 + k];
      if (!(li->hmin[k] <= li->hmax[k]))
      {
        li->hmin[k] = 0.0; /* no samples anywhere */
        li->hmax[k] = 0.0;
      }
      if (li->hmin[k] != li->hmax[k]) li->uniform_metric = false;
    }
    for (int k = 0; k < 3; ++k)
    {
      double r = c->kernel.half_width * c->params.support_scale * li->hmax[k] * gmax;
      if (!li->uniform_metric) r = fmax(r, c->params.pilot_factor * li->hmax[k]);
      li->reach[k] = r;
    }
  }
  free(v);
}

/** @brief The largest reach of any location. */
static void max_reach(const MFTL_Coupler *c, double r[3])
{
  r[0] = r[1] = r[2] = 0.0;
  for (int l = 0; l < c->n_loc; ++l)
    for (int k = 0; k < 3; ++k) r[k] = fmax(r[k], c->li[l].reach[k]);
}

/** @brief Gaps thinner than this are floating-point seams between abutting ownership boxes, not gaps. */
static void point_sliver(const MFTL_Coupler *c, int l, double sliver[3])
{
  const MFTL_Domain *d = &c->ens->domain;
  const cpl_locinfo *li = &c->li[l];
  const double s0 = 1e-9 * fmax(fmax(li->hmin[0], li->hmin[1]), li->hmin[2]);
  const double s = (s0 > 0.0) ? s0 : 1e-12 * fmax(fmax(d->domain_size[0], d->domain_size[1]), d->domain_size[2]);
  sliver[0] = sliver[1] = sliver[2] = s;
}

/**
 * @brief Builds the provider directory: every rank publishes its ownership
 *        extents, widened by the directory reach, to every rank box they touch
 *        (NBX). Collective.
 */
static void build_directory(MFTL_Coupler *c)
{
  const MFTL_Domain *d = &c->ens->domain;
  cpl_dir_entry *own = MFTL_NULL;
  const int n_own = MFTLCplDirectoryOwnEntries(c->grid, d, &own);
  for (int i = 0; i < n_own; ++i) own[i].provider = c->rank;

  /* One buffer per destination rank, in a sparse list. */
  int n_dest = 0, cap_dest = 0;
  int *dest = MFTL_NULL;
  cpl_buf *bufs = MFTL_NULL;
  int *ranks = (int *) MFTLMallocOrDie((size_t) c->nproc * sizeof(int), "directory ranks");

  free(c->dir);
  c->dir = MFTL_NULL;
  int64_t n_dir = 0, cap_dir = 0;

  for (int i = 0; i < n_own; ++i)
  {
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = own[i].lo[k] - c->dir_reach[k];
      hi[k] = own[i].hi[k] + c->dir_reach[k];
    }
    const int nr = MFTLCplRanksTouching(d, lo, hi, ranks, c->nproc);
    for (int j = 0; j < nr; ++j)
    {
      if (ranks[j] == c->rank)
      {
        GROW(c->dir, cap_dir, n_dir + 1, cpl_dir_entry);
        c->dir[n_dir++] = own[i];
        continue;
      }
      int at = -1;
      for (int q = 0; q < n_dest; ++q)
        if (dest[q] == ranks[j]) at = q;
      if (at < 0)
      {
        if (n_dest == cap_dest)
        {
          cap_dest = MFTLGrowCapacity(cap_dest, n_dest + 1);
          dest = (int *) MFTLReallocOrDie(dest, (size_t) cap_dest * sizeof(int), "directory dest");
          bufs = (cpl_buf *) MFTLReallocOrDie(bufs, (size_t) cap_dest * sizeof(cpl_buf), "directory bufs");
        }
        at = n_dest++;
        dest[at] = ranks[j];
        memset(&bufs[at], 0, sizeof(cpl_buf));
      }
      MFTLCplBufPut(&bufs[at], &own[i], sizeof(cpl_dir_entry));
    }
  }
  free(ranks);

  unsigned char **sb = (unsigned char **) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "directory send");
  size_t *sz = (size_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "directory send");
  for (int q = 0; q < n_dest; ++q)
  {
    sb[q] = bufs[q].data;
    sz[q] = bufs[q].n;
  }
  cpl_msg *in = MFTL_NULL;
  int n_in = 0;
  MFTLCplNbx(c->comm, CPL_TAG_DIRECTORY, n_dest, dest, sb, sz, &in, &n_in, &c->stats.bytes_discovery, &c->stats.messages);
  for (int i = 0; i < n_in; ++i)
  {
    const int64_t ne = (int64_t) (in[i].bytes / sizeof(cpl_dir_entry));
    GROW(c->dir, cap_dir, n_dir + ne, cpl_dir_entry);
    memcpy(&c->dir[n_dir], in[i].data, (size_t) ne * sizeof(cpl_dir_entry));
    n_dir += ne;
  }
  MFTLCplMsgsFree(in, n_in);
  free(in);
  for (int q = 0; q < n_dest; ++q) free(bufs[q].data);
  free(bufs);
  free(dest);
  free(sb);
  free(sz);
  free(own);
  c->n_dir = (int) n_dir;
  if (c->dix)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->dix[l]);
  free(c->dix);
  c->dix = (cpl_dir_index *) MFTLCallocOrDie((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_dir_index), "directory index");
  for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexBuild(&c->dix[l], c->dir, c->n_dir, l);

  /* The cover of every point location over this rank's reach: the one place
   * ownership boxes are subtracted, once per directory instead of per request. */
  if (c->cover)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplCoverFree(&c->cover[l]);
  free(c->cover);
  c->cover = (cpl_cover *) MFTLCallocOrDie((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_cover), "directory cover");
  {
    double blo[3], bhi[3];
    own_box(c, blo, bhi);
    for (int l = 0; l < c->n_loc; ++l)
    {
      if (cpl_is_lattice(&c->grid->loc[l])) continue;
      double lo[3], hi[3], cell[3], sliver[3];
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = blo[k] - c->dir_reach[k];
        hi[k] = bhi[k] + c->dir_reach[k];
        if (!d->is_periodic[k])
        {
          lo[k] = fmax(lo[k], d->global_min[k]);
          hi[k] = fmin(hi[k], d->global_max[k]);
        }
        cell[k] = (c->li[l].reach[k] > 0.0) ? c->li[l].reach[k] : d->domain_size[k];
      }
      point_sliver(c, l, sliver);
      MFTLCplCoverBuild(&c->cover[l], &c->dix[l], c->dir, d, lo, hi, cell, sliver);
    }
  }
  c->stats.n_directory_builds++;
}

int MFTLCouplerCreate(MFTL_Ensemble *ens, MFTL_SolverGrid *grid, const MFTL_CouplerParams *params, MFTL_Coupler **out)
{
  if (out) *out = MFTL_NULL;
  if (ens == MFTL_NULL) return MFTL_COUPLING_INVALID;
  if (MFTLCplAgreeMax(ens->domain.comm_world, grid == MFTL_NULL || out == MFTL_NULL ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK) != MFTL_COUPLING_OK)
    return MFTL_COUPLING_INVALID;

  MFTL_Coupler *c = (MFTL_Coupler *) MFTLCallocOrDie(1, sizeof(MFTL_Coupler), "MFTLCouplerCreate");
  c->ens = ens;
  c->grid = grid;
  if (params)
    c->params = *params;
  else
    MFTLCouplerDefaultParams(&c->params);
  c->comm = MPI_COMM_NULL;

  int local = MFTL_COUPLING_OK;
  if (MFTLCplCommDup(ens->domain.comm_world, &c->comm) != MFTL_SUCCESS)
  {
    free(c);
    return MFTL_COUPLING_INVALID;
  }
  MFTLCplCommRankSize(c->comm, &c->rank, &c->nproc);

  if (!MFTLCplCommSameGroup(grid->comm, ens->domain.comm_world))
  {
    fprintf(stderr, "MFTL coupling: the solver communicator is not the ensemble's group of processes\n");
    local = MFTL_COUPLING_INVALID;
  }
  if (!params_valid(&c->params) || MFTLCplKernelResolve(&c->params, &c->kernel) != MFTL_SUCCESS)
  {
    fprintf(stderr, "MFTL coupling: invalid coupler parameters\n");
    local = MFTL_COUPLING_INVALID;
  }
  if (local == MFTL_COUPLING_OK && MFTLCplGridValidate(grid, &ens->domain) != MFTL_SUCCESS) local = MFTL_COUPLING_INVALID;

  int st = agree(c, local);
  if (st == MFTL_COUPLING_OK)
  {
    const bool same = MFTLCplAgreeHash(c->comm, MFTLCplGridDefinitionHash(grid) ^ (params_hash(&c->params) * 0x9E3779B97F4A7C15ull));
    if (!same)
    {
      if (c->rank == 0) fprintf(stderr, "MFTL coupling: locations, lattices or parameters differ between ranks\n");
      st = MFTL_COUPLING_INVALID;
    }
  }
  if (st != MFTL_COUPLING_OK)
  {
    MFTLCplCommFree(&c->comm);
    free(c);
    return st;
  }

  c->n_loc = grid->n_loc;
  c->li = (cpl_locinfo *) MFTLCallocOrDie((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_locinfo), "locinfo");
  compute_locinfo(c);
  double r[3];
  max_reach(c, r);
  for (int k = 0; k < 3; ++k) c->dir_reach[k] = (c->params.directory_margin > 0.0) ? c->params.directory_margin : 3.0 * r[k];
  build_directory(c);
  c->seen_owned = grid->version_owned;
  c->seen_export = grid->version_export;
  c->grid_epoch = 1;
  *out = c;
  return MFTL_COUPLING_OK;
}

static void free_transfer_staging(MFTL_Coupler *c)
{
  free(c->sbuf);
  free(c->rbuf);
  c->sbuf = c->rbuf = MFTL_NULL;
  c->scap = c->rcap = 0;
  for (int i = 0; i < c->spread_scratch_n; ++i) free(c->spread_scratch[i]);
  free(c->spread_scratch);
  free(c->spread_scratch_cap);
  c->spread_scratch = MFTL_NULL;
  c->spread_scratch_cap = MFTL_NULL;
  c->spread_scratch_n = 0;
}

void MFTLCouplerDestroy(MFTL_Coupler **pc)
{
  if (pc == MFTL_NULL || *pc == MFTL_NULL) return;
  MFTL_Coupler *c = *pc;
  if (agree(c, (c->active || c->live > 0) ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK) != MFTL_COUPLING_OK)
  {
    if (c->rank == 0) fprintf(stderr, "MFTL coupling: release all dependent handles and finish transfers before destroying the coupler\n");
    return;
  }
  MFTLCplSchedFree(&c->sched);
  free_transfer_staging(c);
  MFTLCplRowScratchFree(&c->rs);
  if (c->dix)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->dix[l]);
  free(c->dix);
  if (c->cover)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplCoverFree(&c->cover[l]);
  free(c->cover);
  free(c->dir);
  free(c->li);
  MFTLCplCommFree(&c->comm);
  free(c);
  *pc = MFTL_NULL;
}

int MFTLCouplerUpdateGrid(MFTL_Coupler *c)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = (c->active != 0) ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK;
  if (local == MFTL_COUPLING_OK && MFTLCplGridValidate(c->grid, &c->ens->domain) != MFTL_SUCCESS) local = MFTL_COUPLING_INVALID;
  if (c->grid->n_loc != c->n_loc) local = MFTL_COUPLING_INVALID;
  /* Two flags ride on the status agreement: did ANY rank change what it owns,
   * or what it exports? One MAX over three ints carries all of it. */
  int v[3] = {local, (c->grid->version_owned != c->seen_owned) ? 1 : 0, (c->grid->version_export != c->seen_export) ? 1 : 0};
  MFTLCplAgreeMaxN(c->comm, v, 3);
  const int st = v[0];
  if (st != MFTL_COUPLING_OK) return st;
  if (v[1])
  {
    compute_locinfo(c);
    build_directory(c);
  }
  else
  {
    compute_locinfo(c);
    c->stats.n_directory_kept++;
  }
  c->seen_owned = c->grid->version_owned;
  c->seen_export = c->grid->version_export;
  if (v[1] || v[2]) c->grid_epoch++;
  return MFTL_COUPLING_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Support requests
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A requester's query: one bin and one location (lattice), or one bin, location, provider and image (points). */
typedef struct
{
  int wl;
  int location;
  int kind;  ///< 0 lattice, 1 points
  int provider;  ///< points only
  int64_t bin;
  int64_t a[3], b[3];
  int64_t expected, got;
  double box[6];  ///< lattice: Q; points: Q in the provider frame
  int shift[3];
} query_t;

/** @brief One sample record a requester received. */
typedef struct
{
  int wl;
  int provider;
  uint64_t key;
  double pos[3];
  double measure;
  double h[3];
} rec_t;

/** @brief What a provider must send one requester for one location. */
typedef struct
{
  int requester;
  int wl;
  int64_t n, n_nonhalo, cap;
  uint64_t *key;
  int *piece;
  int64_t *elem;
  int64_t *sample;
  double *meas;
} pack_t;

static void pack_push(pack_t *pk, uint64_t key, int piece, int64_t elem, int64_t sample, double meas)
{
  if (pk->n == pk->cap)
  {
    pk->cap = (pk->cap == 0) ? 64 : 2 * pk->cap;
    pk->key = (uint64_t *) MFTLReallocOrDie(pk->key, (size_t) pk->cap * sizeof(uint64_t), "pack");
    pk->piece = (int *) MFTLReallocOrDie(pk->piece, (size_t) pk->cap * sizeof(int), "pack");
    pk->elem = (int64_t *) MFTLReallocOrDie(pk->elem, (size_t) pk->cap * sizeof(int64_t), "pack");
    pk->sample = (int64_t *) MFTLReallocOrDie(pk->sample, (size_t) pk->cap * sizeof(int64_t), "pack");
    pk->meas = (double *) MFTLReallocOrDie(pk->meas, (size_t) pk->cap * sizeof(double), "pack");
  }
  pk->key[pk->n] = key;
  pk->piece[pk->n] = piece;
  pk->elem[pk->n] = elem;
  pk->sample[pk->n] = sample;
  pk->meas[pk->n] = meas;
  pk->n++;
}

static void pack_free(pack_t *pk)
{
  free(pk->key);
  free(pk->piece);
  free(pk->elem);
  free(pk->sample);
  free(pk->meas);
}

/** @brief Box subtraction that never runs out of room (a dropped piece would hide a gap). */
static void box_sub_dyn(double **rem, int *n, int *cap, const double *sub_box, const double sliver[3])
{
  if (*cap < 6 * (*n) + 6)
  {
    *cap = 6 * (*n) + 6;
    *rem = (double *) MFTLReallocOrDie(*rem, 6 * (size_t) *cap * sizeof(double), "box subtraction");
  }
  MFTLCplBoxSubtract(*rem, n, *cap, sub_box, sliver);
}

static void ibox_sub_dyn(int64_t **rem, int *n, int *cap, const int64_t *sub_box)
{
  if (*cap < 6 * (*n) + 6)
  {
    *cap = 6 * (*n) + 6;
    *rem = (int64_t *) MFTLReallocOrDie(*rem, 6 * (size_t) *cap * sizeof(int64_t), "box subtraction");
  }
  MFTLCplIboxSubtract(*rem, n, *cap, sub_box);
}

struct MFTL_SupportRequest
{
  MFTL_Coupler *c;
  uint64_t grid_epoch;
  int status;
  int n_loc;
  int *locs;
  double skin[3], bin[3];
  int64_t n_bins;
  uint64_t *bin_key;
  int64_t *bin_idx;
  double *qbox;  ///< [n_loc][6] bounding box of each location's request boxes, unwrapped
  int64_t n_q, cap_q;
  query_t *q;
  int64_t n_rec, cap_rec;
  rec_t *rec;
  int n_pack, cap_pack;
  pack_t *pack;
  int64_t n_rbox, cap_rbox;
  double *rbox;
  int *rbox_wl;
  int n_missing, cap_missing;
  MFTL_RequestRegion *missing;
  int n_unprovided, cap_unprovided;
  MFTL_RequestRegion *unprovided;
  long n_served, n_sent;
};

void MFTLCouplerReleaseRequest(MFTL_SupportRequest **pr)
{
  if (pr == MFTL_NULL || *pr == MFTL_NULL) return;
  MFTL_SupportRequest *r = *pr;
  free(r->locs);
  free(r->bin_key);
  free(r->bin_idx);
  free(r->qbox);
  free(r->q);
  free(r->rec);
  for (int i = 0; i < r->n_pack; ++i) pack_free(&r->pack[i]);
  free(r->pack);
  free(r->rbox);
  free(r->rbox_wl);
  free(r->missing);
  free(r->unprovided);
  free(r);
  *pr = MFTL_NULL;
}

int MFTLCouplerGetRequests(const MFTL_SupportRequest *r, MFTL_RequestView *v)
{
  if (r == MFTL_NULL || v == MFTL_NULL) return MFTL_ERROR;
  v->n_missing = r->n_missing;
  v->missing = r->missing;
  v->n_unprovided = r->n_unprovided;
  v->unprovided = r->unprovided;
  v->n_requests_served = r->n_served;
  v->n_requests_sent = r->n_sent;
  return MFTL_SUCCESS;
}

static void add_region(MFTL_RequestRegion **arr, int *n, int *cap, const MFTL_RequestRegion *x)
{
  if (*n == *cap)
  {
    *cap = MFTLGrowCapacity(*cap, *n + 1);
    *arr = (MFTL_RequestRegion *) MFTLReallocOrDie(*arr, (size_t) *cap * sizeof(MFTL_RequestRegion), "request regions");
  }
  (*arr)[(*n)++] = *x;
}

/** @brief Resolves a window location list; returns false on an invalid one. */
static bool resolve_locations(const MFTL_Coupler *c, const MFTL_MarkerSpec *m, int *n_out, int **out)
{
  const int n = (m->n_locations > 0) ? m->n_locations : c->n_loc;
  int *l = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(int), "window locations");
  bool ok = true;
  for (int i = 0; i < n; ++i)
  {
    l[i] = (m->n_locations > 0) ? (m->locations ? m->locations[i] : -1) : i;
    if (l[i] < 0 || l[i] >= c->n_loc) ok = false;
    for (int j = 0; j < i; ++j)
      if (l[j] == l[i]) ok = false;
  }
  *n_out = n;
  *out = l;
  return ok && n > 0;
}

/**
 * @brief Bin size of a window: half the largest reach over its locations
 *        (never zero). A request box is a bin widened by the reach and the
 *        skin, so the union over-requests by at most one bin on each side of
 *        the true union of supports: half a reach. Wider bins mean fewer,
 *        fatter boxes -- with bins of a whole reach a 642-marker sphere asked
 *        for a band about twice its supports' volume.
 */
static void window_bin(const MFTL_Coupler *c, int n, const int *locs, double b[3])
{
  const MFTL_Domain *d = &c->ens->domain;
  for (int k = 0; k < 3; ++k)
  {
    b[k] = 0.0;
    for (int i = 0; i < n; ++i) b[k] = fmax(b[k], 0.5 * c->li[locs[i]].reach[k]);
    if (!(b[k] > 0.0)) b[k] = d->domain_size[k];
    /* 21-bit bin keys: never more than 2^19 bins per period. */
    const double floor_b = d->domain_size[k] / (double) (1 << 19);
    if (b[k] < floor_b) b[k] = floor_b;
  }
}

/** @brief Valid ghost boxes of this rank's blocks of lattice location @p l (unwrapped indices). */
static int halo_boxes(const MFTL_SolverGrid *g, int l, int64_t *boxes, int cap)
{
  const cpl_location *L = &g->loc[l];
  int n = 0;
  for (int pc = 0; pc < L->n_pieces; ++pc)
  {
    const cpl_block *b = &g->blocks[L->piece_ref[pc]];
    if (b->ghost[0] == 0 && b->ghost[1] == 0 && b->ghost[2] == 0) continue;
    if (n >= cap) break;
    for (int k = 0; k < 3; ++k)
    {
      boxes[6 * n + k] = b->lo[k] - b->ghost[k];
      boxes[6 * n + 3 + k] = b->hi[k] + b->ghost[k];
    }
    ++n;
  }
  return n;
}

/**
 * @brief Is wrapped lattice index @p j inside one of @p boxes for some
 *        periodic representative? Writes that representative to @p rep and
 *        returns the box index, or -1.
 */
static int halo_find(const cpl_location *L, const MFTL_Domain *d, const int64_t j[3], const int64_t *boxes, int n, int64_t rep[3])
{
  for (int bx = 0; bx < n; ++bx)
  {
    const int64_t *b = &boxes[6 * bx];
    bool ok = true;
    for (int k = 0; k < 3 && ok; ++k)
    {
      if (!d->is_periodic[k])
      {
        if (j[k] < b[k] || j[k] >= b[3 + k]) ok = false;
        rep[k] = j[k];
        continue;
      }
      const int64_t N = L->dims[k];
      /* smallest representative >= b[k] */
      const int64_t s = cpl_floordiv(b[k] - j[k] + N - 1, N);
      const int64_t i = j[k] + s * N;
      if (i >= b[3 + k]) ok = false;
      rep[k] = i;
    }
    if (ok) return bx;
  }
  return -1;
}

/** @brief Records a request box this rank received (for band masks). */
static void push_rbox(MFTL_SupportRequest *R, const double box[6], int wl)
{
  if (R->n_rbox == R->cap_rbox)
  {
    R->cap_rbox = (R->cap_rbox == 0) ? 64 : 2 * R->cap_rbox;
    R->rbox = (double *) MFTLReallocOrDie(R->rbox, 6 * (size_t) R->cap_rbox * sizeof(double), "request boxes");
    R->rbox_wl = (int *) MFTLReallocOrDie(R->rbox_wl, (size_t) R->cap_rbox * sizeof(int), "request boxes");
  }
  memcpy(&R->rbox[6 * R->n_rbox], box, 6 * sizeof(double));
  R->rbox_wl[R->n_rbox++] = wl;
}

/**
 * @brief Marks of one piece: which of its samples this requester gets, and the
 *        index range the marks span (so collecting them never scans the piece).
 */
typedef struct
{
  unsigned char *m;
  unsigned char *tile_full;  ///< points: tiles already marked whole
  int64_t n, lo, hi;
} marks_t;

static void mark(marks_t *mk, int64_t i)
{
  mk->m[i] = 1;
  if (i < mk->lo) mk->lo = i;
  if (i + 1 > mk->hi) mk->hi = i + 1;
}

/**
 * @brief Provider side: answers one requester's message.
 *
 * The requester's boxes overlap heavily -- each is a marker bin widened by the
 * reach and the skin, about ten bins across -- so a sample lies in hundreds of
 * them. Every box is COUNTED (that is the coverage proof, and on a FULL
 * unmasked block it is arithmetic), but a sample is MARKED, and so returned,
 * packed and later fetched, once per requester.
 */
static void serve(MFTL_Coupler *c, MFTL_SupportRequest *R, int requester, const unsigned char *msg, size_t bytes, cpl_buf *reply)
{
  const MFTL_SolverGrid *g = c->grid;
  const MFTL_Domain *d = &c->ens->domain;
  size_t off = 0;

  /* The requester's ghost boxes, per location. */
  int32_t n_halo = 0;
  GET(msg, bytes, &off, n_halo);
  int64_t *hbox = (int64_t *) MFTLMallocOrDie(6 * (size_t) MFTL_MAX(n_halo, 1) * sizeof(int64_t), "halo boxes");
  int *hloc = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_halo, 1) * sizeof(int), "halo boxes");
  for (int i = 0; i < n_halo; ++i)
  {
    int32_t l;
    GET(msg, bytes, &off, l);
    hloc[i] = l;
    for (int k = 0; k < 6; ++k) GET(msg, bytes, &off, hbox[6 * i + k]);
  }

  /* marks, per window location and piece, allocated when first touched */
  marks_t **marks = (marks_t **) MFTLCallocOrDie((size_t) R->n_loc, sizeof(marks_t *), "serve marks");
  for (int wl = 0; wl < R->n_loc; ++wl)
    marks[wl] = (marks_t *) MFTLCallocOrDie((size_t) MFTL_MAX(g->loc[R->locs[wl]].n_pieces, 1), sizeof(marks_t), "serve marks");
#define CPL_MARKS(wl, pc, size)                                                                    \
  do                                                                                               \
  {                                                                                                \
    if (marks[wl][pc].m == MFTL_NULL)                                                              \
    {                                                                                              \
      marks[wl][pc].n = (size);                                                                    \
      marks[wl][pc].m = (unsigned char *) MFTLCallocOrDie((size_t) MFTL_MAX(size, 1), 1, "marks"); \
      marks[wl][pc].lo = (size);                                                                   \
      marks[wl][pc].hi = 0;                                                                        \
    }                                                                                              \
  } while (0)

  int32_t n_sub = 0;
  GET(msg, bytes, &off, n_sub);
  PUT(reply, n_sub);
  R->n_served += n_sub;

  for (int s = 0; s < n_sub; ++s)
  {
    int32_t hdr[4];
    for (int i = 0; i < 4; ++i) GET(msg, bytes, &off, hdr[i]);
    const int32_t sub_id = hdr[0], loc = hdr[1], kind = hdr[2], wl = hdr[3];
    int64_t n_active = 0, n_inactive = 0, n_missing = 0;
    const cpl_location *L = &g->loc[loc];

    if (kind == 0)
    {
      int64_t a[3], b[3];
      for (int k = 0; k < 3; ++k) GET(msg, bytes, &off, a[k]);
      for (int k = 0; k < 3; ++k) GET(msg, bytes, &off, b[k]);
      {
        double rb[6];
        for (int k = 0; k < 3; ++k)
        {
          rb[k] = (double) a[k];
          rb[3 + k] = (double) b[k];
        }
        push_rbox(R, rb, wl);
      }

      for (int pc = 0; pc < L->n_pieces; ++pc)
      {
        const cpl_block *blk = &g->blocks[L->piece_ref[pc]];
        int64_t ilo[3][2], ihi[3][2];
        int ni[3];
        bool any = true;
        for (int k = 0; k < 3; ++k)
        {
          ni[k] = MFTLCplLatIntersect(L, d, k, a[k], b[k], blk->lo[k], blk->hi[k], ilo[k], ihi[k], 2);
          if (ni[k] == 0) any = false;
        }
        if (!any) continue;
        const int64_t nx = blk->hi[0] - blk->lo[0], ny = blk->hi[1] - blk->lo[1], nz = blk->hi[2] - blk->lo[2];
        CPL_MARKS(wl, pc, nx * ny * nz);
        marks_t *mk = &marks[wl][pc];
        const bool plain = (L->exposure == MFTL_GRID_FULL) && blk->active == MFTL_NULL;
        int64_t miss_here = 0;
        for (int iz = 0; iz < ni[2]; ++iz)
          for (int iy = 0; iy < ni[1]; ++iy)
            for (int ix = 0; ix < ni[0]; ++ix)
            {
              if (plain)
              {
                /* every sample of the box is there and exported: count by arithmetic, mark by rows */
                n_active += (ihi[0][ix] - ilo[0][ix]) * (ihi[1][iy] - ilo[1][iy]) * (ihi[2][iz] - ilo[2][iz]);
                for (int64_t k3 = ilo[2][iz]; k3 < ihi[2][iz]; ++k3)
                  for (int64_t j3 = ilo[1][iy]; j3 < ihi[1][iy]; ++j3)
                  {
                    const int64_t row = (ilo[0][ix] - blk->lo[0]) + nx * ((j3 - blk->lo[1]) + ny * (k3 - blk->lo[2]));
                    const int64_t len = ihi[0][ix] - ilo[0][ix];
                    memset(mk->m + row, 1, (size_t) len);
                    if (row < mk->lo) mk->lo = row;
                    if (row + len > mk->hi) mk->hi = row + len;
                  }
                continue;
              }
              for (int64_t k3 = ilo[2][iz]; k3 < ihi[2][iz]; ++k3)
                for (int64_t j3 = ilo[1][iy]; j3 < ihi[1][iy]; ++j3)
                  for (int64_t i3 = ilo[0][ix]; i3 < ihi[0][ix]; ++i3)
                  {
                    const int64_t idx[3] = {i3, j3, k3};
                    const int64_t sample = (i3 - blk->lo[0]) + nx * ((j3 - blk->lo[1]) + ny * (k3 - blk->lo[2]));
                    if (blk->active && !blk->active[sample])
                    {
                      ++n_inactive;
                      continue;
                    }
                    if (!MFTLCplLatExported(L, idx))
                    {
                      ++n_missing;
                      ++miss_here;
                      continue;
                    }
                    ++n_active;
                    mark(mk, sample);
                  }
              /* Report what is missing as index boxes: the matched box minus the exports. */
              if (miss_here > 0)
              {
                int cap_rem = 16, nrem = 1;
                int64_t *rem = (int64_t *) MFTLMallocOrDie(6 * (size_t) cap_rem * sizeof(int64_t), "missing regions");
                rem[0] = ilo[0][ix];
                rem[1] = ilo[1][iy];
                rem[2] = ilo[2][iz];
                rem[3] = ihi[0][ix];
                rem[4] = ihi[1][iy];
                rem[5] = ihi[2][iz];
                for (int cv = 0; cv < L->n_cov; ++cv) ibox_sub_dyn(&rem, &nrem, &cap_rem, &L->cov_index[6 * cv]);
                for (int r = 0; r < nrem; ++r)
                {
                  MFTL_RequestRegion reg;
                  memset(&reg, 0, sizeof reg);
                  reg.location = loc;
                  reg.piece = blk->piece;
                  reg.requester = requester;
                  for (int k = 0; k < 3; ++k)
                  {
                    reg.index_lo[k] = rem[6 * r + k];
                    reg.index_hi[k] = rem[6 * r + 3 + k];
                    reg.box[k] = MFTLCplLatCoord(L, k, reg.index_lo[k]);
                    reg.box[3 + k] = MFTLCplLatCoord(L, k, reg.index_hi[k] - 1);
                  }
                  add_region(&R->missing, &R->n_missing, &R->cap_missing, &reg);
                }
                free(rem);
                miss_here = 0;
              }
            }
        (void) nz;
      }
    }
    else
    {
      double box[6];
      for (int k = 0; k < 6; ++k) GET(msg, bytes, &off, box[k]);
      push_rbox(R, box, wl);

      for (int pc = 0; pc < L->n_pieces; ++pc)
      {
        const cpl_points *p = &g->points[L->piece_ref[pc]];
        for (int t = 0; t < p->n_tiles; ++t)
        {
          const double *pb = &p->tile_pbox[6 * t];
          if (!cpl_box_overlap(pb, pb + 3, box, box + 3)) continue;
          CPL_MARKS(wl, pc, p->n);
          marks_t *mk = &marks[wl][pc];
          const int64_t t0 = p->tile_off[t], t1 = p->tile_off[t + 1];
          if (pb[0] >= box[0] && pb[3] <= box[3] && pb[1] >= box[1] && pb[4] <= box[4] && pb[2] >= box[2] && pb[5] <= box[5])
          {
            /* the whole tile is inside: count it, and mark it unless an earlier box did */
            n_active += t1 - t0;
            if (mk->tile_full == MFTL_NULL) mk->tile_full = (unsigned char *) MFTLCallocOrDie((size_t) p->n_tiles, 1, "tile marks");
            if (!mk->tile_full[t])
            {
              for (int64_t i = t0; i < t1; ++i) mark(mk, i);
              mk->tile_full[t] = 1;
            }
            continue;
          }
          for (int64_t i = t0; i < t1; ++i)
          {
            const double *x = &p->pos[3 * i];
            if (x[0] < box[0] || x[0] > box[3] || x[1] < box[1] || x[1] > box[4] || x[2] < box[2] || x[2] > box[5]) continue;
            ++n_active;
            mark(mk, i);
          }
        }
      }
      /* SELECTED: what of my ownership in the box is not certified complete? */
      if (L->exposure == MFTL_GRID_SELECTED)
      {
        const double sl = 1e-9 * fmax(fmax(c->li[loc].hmin[0], c->li[loc].hmin[1]), c->li[loc].hmin[2]);
        const double sliver[3] = {sl, sl, sl};
        for (int r = 0; r < L->n_prov; ++r)
        {
          const double *D = &L->prov_boxes[6 * r];
          if (!cpl_box_overlap(D, D + 3, box, box + 3)) continue;
          int cap_rem = 16, nrem = 1;
          double *rem = (double *) MFTLMallocOrDie(6 * (size_t) cap_rem * sizeof(double), "missing regions");
          for (int k = 0; k < 3; ++k)
          {
            rem[k] = fmax(D[k], box[k]);
            rem[3 + k] = fmin(D[3 + k], box[3 + k]);
          }
          for (int cv = 0; cv < L->n_cov; ++cv) box_sub_dyn(&rem, &nrem, &cap_rem, &L->cov_boxes[6 * cv], sliver);
          for (int q = 0; q < nrem; ++q)
          {
            MFTL_RequestRegion reg;
            memset(&reg, 0, sizeof reg);
            reg.location = loc;
            reg.piece = -1;
            reg.requester = requester;
            memcpy(reg.box, &rem[6 * q], sizeof reg.box);
            add_region(&R->missing, &R->n_missing, &R->cap_missing, &reg);
            ++n_missing;
          }
          free(rem);
        }
      }
    }
    PUT(reply, sub_id);
    PUT(reply, n_active);
    PUT(reply, n_inactive);
    PUT(reply, n_missing);
  }
#undef CPL_MARKS

  /* The marked samples, once each: pack lists and the reply's records.
   * Pack order: unique keys ascending, the ghost-readable ones last. */
  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    const int loc = R->locs[wl];
    const cpl_location *L = &g->loc[loc];
    pack_t pk;
    memset(&pk, 0, sizeof pk);
    for (int pc = 0; pc < L->n_pieces; ++pc)
    {
      marks_t *mk = &marks[wl][pc];
      if (mk->m == MFTL_NULL) continue;
      if (cpl_is_lattice(L))
      {
        const cpl_block *blk = &g->blocks[L->piece_ref[pc]];
        const int64_t nx = blk->hi[0] - blk->lo[0], ny = blk->hi[1] - blk->lo[1];
        for (int64_t sm = mk->lo; sm < mk->hi; ++sm)
        {
          if (!mk->m[sm]) continue;
          const int64_t idx[3] = {blk->lo[0] + sm % nx, blk->lo[1] + (sm / nx) % ny, blk->lo[2] + sm / (nx * ny)};
          pack_push(&pk, cpl_lat_key(L, idx[0], idx[1], idx[2]), blk->piece, cpl_block_elem(blk, idx), sm,
                    MFTLCplLatWidth(L, 0, idx[0]) * MFTLCplLatWidth(L, 1, idx[1]) * MFTLCplLatWidth(L, 2, idx[2]));
        }
      }
      else
      {
        const cpl_points *p = &g->points[L->piece_ref[pc]];
        for (int64_t i = mk->lo; i < mk->hi; ++i)
          if (mk->m[i]) pack_push(&pk, p->key[i], p->piece, p->aidx[i], i, p->measure[i]);
      }
      free(mk->m);
      free(mk->tile_full);
    }
    free(marks[wl]);

    int64_t hb[6 * 32];
    int nh = 0;
    for (int i = 0; i < n_halo && nh < 32; ++i)
      if (hloc[i] == loc)
      {
        memcpy(&hb[6 * nh], &hbox[6 * i], 6 * sizeof(int64_t));
        ++nh;
      }
    pack_t out;
    memset(&out, 0, sizeof out);
    out.requester = requester;
    out.wl = wl;
    const size_t nn = (size_t) MFTL_MAX(pk.n, 1);
    out.cap = pk.n;
    out.key = (uint64_t *) MFTLMallocOrDie(nn * sizeof(uint64_t), "pack");
    out.piece = (int *) MFTLMallocOrDie(nn * sizeof(int), "pack");
    out.elem = (int64_t *) MFTLMallocOrDie(nn * sizeof(int64_t), "pack");
    out.sample = (int64_t *) MFTLMallocOrDie(nn * sizeof(int64_t), "pack");
    out.meas = (double *) MFTLMallocOrDie(nn * sizeof(double), "pack");
    if (pk.n > 0)
    {
      int *perm = (int *) MFTLMallocOrDie((size_t) pk.n * sizeof(int), "pack sort");
      uint64_t *k2 = (uint64_t *) MFTLMallocOrDie((size_t) pk.n * sizeof(uint64_t), "pack sort");
      for (int64_t i = 0; i < pk.n; ++i)
      {
        perm[i] = (int) i;
        k2[i] = pk.key[i];
      }
      MFTLSortPairs64(k2, perm, (int) pk.n);
      for (int pass = 0; pass < 2; ++pass)
      {
        for (int64_t i = 0; i < pk.n; ++i)
        {
          if (i > 0 && k2[i] == k2[i - 1]) continue;
          bool halo = false;
          if (requester != c->rank && cpl_is_lattice(L) && nh > 0)
          {
            int64_t j[3], rep[3];
            cpl_lat_unkey(L, k2[i], j);
            halo = halo_find(L, d, j, hb, nh, rep) >= 0;
          }
          if ((pass == 0) == halo) continue;
          const int sx = perm[i];
          out.key[out.n] = pk.key[sx];
          out.piece[out.n] = pk.piece[sx];
          out.elem[out.n] = pk.elem[sx];
          out.sample[out.n] = pk.sample[sx];
          out.meas[out.n] = pk.meas[sx];
          out.n++;
        }
        if (pass == 0) out.n_nonhalo = out.n;
      }
      free(perm);
      free(k2);
    }
    /* the records, in pack order */
    const int64_t nrec = out.n;
    PUT(reply, nrec);
    for (int64_t i = 0; i < out.n; ++i)
    {
      PUT(reply, out.key[i]);
      if (!cpl_is_lattice(L))
      {
        const cpl_points *p = &g->points[L->piece_ref[out.piece[i]]];
        const int64_t pi = out.sample[i];
        MFTLCplBufPut(reply, &p->pos[3 * pi], 3 * sizeof(double));
        PUT(reply, p->measure[pi]);
        MFTLCplBufPut(reply, &p->h[3 * pi], 3 * sizeof(double));
      }
    }
    /* a point's sample index is its position in the sorted set until here; make it the registration index */
    if (!cpl_is_lattice(L))
      for (int64_t i = 0; i < out.n; ++i) out.sample[i] = g->points[L->piece_ref[out.piece[i]]].orig[out.sample[i]];
    pack_free(&pk);
    if (out.n == 0)
    {
      pack_free(&out);
      continue;
    }
    if (R->n_pack == R->cap_pack)
    {
      R->cap_pack = MFTLGrowCapacity(R->cap_pack, R->n_pack + 1);
      R->pack = (pack_t *) MFTLReallocOrDie(R->pack, (size_t) R->cap_pack * sizeof(pack_t), "pack lists");
    }
    R->pack[R->n_pack++] = out;
  }
  free(marks);
  free(hbox);
  free(hloc);
}

/** @brief Requester side: reads one provider's reply into query counts and records. */
static int absorb_reply(MFTL_Coupler *c, MFTL_SupportRequest *R, int provider, const unsigned char *msg, size_t bytes)
{
  size_t off = 0;
  int32_t n_sub = 0;
  if (!GET(msg, bytes, &off, n_sub)) return MFTL_COUPLING_INVALID;
  int status = MFTL_COUPLING_OK;
  for (int s = 0; s < n_sub; ++s)
  {
    int32_t sub_id;
    int64_t n_active, n_inactive, n_missing;
    if (!GET(msg, bytes, &off, sub_id) || !GET(msg, bytes, &off, n_active) || !GET(msg, bytes, &off, n_inactive) || !GET(msg, bytes, &off, n_missing))
      return MFTL_COUPLING_INVALID;
    if (sub_id < 0 || sub_id >= R->n_q) return MFTL_COUPLING_INVALID;
    query_t *q = &R->q[sub_id];
    q->got += n_active + n_inactive + n_missing;
    if (n_missing > 0 && status < MFTL_COUPLING_NEED_COVERAGE) status = MFTL_COUPLING_NEED_COVERAGE;
  }
  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    int64_t nrec = 0;
    if (!GET(msg, bytes, &off, nrec) || nrec < 0) return MFTL_COUPLING_INVALID;
    const bool points = !cpl_is_lattice(&c->grid->loc[R->locs[wl]]);
    GROW(R->rec, R->cap_rec, R->n_rec + nrec, rec_t);
    for (int64_t i = 0; i < nrec; ++i)
    {
      rec_t *r = &R->rec[R->n_rec++];
      memset(r, 0, sizeof *r);
      r->wl = wl;
      r->provider = provider;
      if (!GET(msg, bytes, &off, r->key)) return MFTL_COUPLING_INVALID;
      if (points)
      {
        for (int k = 0; k < 3; ++k)
          if (!GET(msg, bytes, &off, r->pos[k])) return MFTL_COUPLING_INVALID;
        if (!GET(msg, bytes, &off, r->measure)) return MFTL_COUPLING_INVALID;
        for (int k = 0; k < 3; ++k)
          if (!GET(msg, bytes, &off, r->h[k])) return MFTL_COUPLING_INVALID;
      }
    }
  }
  return status;
}

static int cmp_rec_key_provider(const void *a, const void *b)
{
  const rec_t *x = (const rec_t *) a, *y = (const rec_t *) b;
  if (x->wl != y->wl) return (x->wl > y->wl) - (x->wl < y->wl);
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  return (x->provider > y->provider) - (x->provider < y->provider);
}

int MFTLCouplerRequestSupport(MFTL_Coupler *c, const MFTL_MarkerSpec *spec, const MFTL_WindowSpec *wspec, MFTL_SupportRequest **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const MFTL_Domain *d = &c->ens->domain;
  const MFTL_SolverGrid *g = c->grid;

  MFTL_SupportRequest *R = (MFTL_SupportRequest *) MFTLCallocOrDie(1, sizeof(MFTL_SupportRequest), "MFTLCouplerRequestSupport");
  R->c = c;
  R->grid_epoch = c->grid_epoch;

  int local = MFTL_COUPLING_OK;
  if (out == MFTL_NULL || spec == MFTL_NULL || wspec == MFTL_NULL || c->active) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && !resolve_locations(c, spec, &R->n_loc, &R->locs)) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && (g->version_owned != c->seen_owned || g->version_export != c->seen_export))
  {
    fprintf(stderr, "MFTL coupling: the grid changed since the last MFTLCouplerUpdateGrid()\n");
    local = MFTL_COUPLING_STALE;
  }
  for (int k = 0; k < 3 && local == MFTL_COUPLING_OK; ++k)
  {
    R->skin[k] = wspec->skin[k];
    if (!(R->skin[k] >= 0.0) || !isfinite(R->skin[k])) local = MFTL_COUPLING_INVALID;
  }
  cpl_markers m;
  memset(&m, 0, sizeof m);
  if (local == MFTL_COUPLING_OK) local = MFTLCplMarkersGather(c->ens, spec, &m);

  /* Every rank must ask for the same locations with the same skin. */
  uint64_t sig = CPL_FNV_INIT;
  if (local == MFTL_COUPLING_OK)
  {
    sig = cpl_fnv(sig, &R->n_loc, sizeof R->n_loc);
    sig = cpl_fnv(sig, R->locs, (size_t) R->n_loc * sizeof(int));
    sig = cpl_fnv(sig, R->skin, sizeof R->skin);
  }
  int st = agree(c, local);
  if (st == MFTL_COUPLING_OK && !MFTLCplAgreeHash(c->comm, sig)) st = MFTL_COUPLING_INVALID;
  if (st == MFTL_COUPLING_OK)
  {
    /* one image per support: the reach plus the skin stays below half a period */
    for (int i = 0; i < R->n_loc; ++i)
      for (int k = 0; k < 3; ++k)
        if (d->is_periodic[k] && !(c->li[R->locs[i]].reach[k] + R->skin[k] < 0.5 * d->domain_size[k])) st = MFTL_COUPLING_INVALID;
    if (st != MFTL_COUPLING_OK && c->rank == 0) fprintf(stderr, "MFTL coupling: kernel reach plus skin must stay below half a period\n");
    /* A point location nobody has described (no samples, no declared bounds)
     * has no reach: a request would silently be the whole domain. */
    for (int i = 0; i < R->n_loc && st == MFTL_COUPLING_OK; ++i)
    {
      const cpl_locinfo *li = &c->li[R->locs[i]];
      if (!li->lattice && !(li->hmax[0] > 0.0 && li->hmax[1] > 0.0 && li->hmax[2] > 0.0))
      {
        if (c->rank == 0)
          fprintf(stderr, "MFTL coupling: location '%s' has no samples and no declared resolution (MFTL_ProviderRegions h_max)\n",
                  g->loc[R->locs[i]].name);
        st = MFTL_COUPLING_INVALID;
      }
    }
  }
  if (st != MFTL_COUPLING_OK)
  {
    MFTLCplMarkersFree(&m);
    MFTLCouplerReleaseRequest(&R);
    c->stats.n_requests++;
    return st;
  }

  /* ── marker bins ── */
  window_bin(c, R->n_loc, R->locs, R->bin);
  {
    uint64_t *keys = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(m.n, 1) * sizeof(uint64_t), "marker bins");
    uint64_t *tmp = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(m.n, 1) * sizeof(uint64_t), "marker bins");
    for (int64_t i = 0; i < m.n; ++i)
    {
      int64_t ib[3];
      for (int k = 0; k < 3; ++k) ib[k] = (int64_t) floor((m.pos[3 * i + k] - d->global_min[k]) / R->bin[k]);
      keys[i] = bin_key3(ib[0], ib[1], ib[2]);
    }
    MFTLRadixSort64(keys, tmp, (int) m.n);
    int64_t nb = 0;
    for (int64_t i = 0; i < m.n; ++i)
      if (i == 0 || keys[i] != keys[i - 1]) keys[nb++] = keys[i];
    R->n_bins = nb;
    R->bin_key = keys;
    R->bin_idx = (int64_t *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(nb, 1) * sizeof(int64_t), "marker bins");
    const int64_t off = INT64_C(1) << 20, mask = (INT64_C(1) << 21) - 1;
    for (int64_t i = 0; i < nb; ++i)
    {
      R->bin_idx[3 * i + 0] = (int64_t) ((keys[i] >> 42) & (uint64_t) mask) - off;
      R->bin_idx[3 * i + 1] = (int64_t) ((keys[i] >> 21) & (uint64_t) mask) - off;
      R->bin_idx[3 * i + 2] = (int64_t) (keys[i] & (uint64_t) mask) - off;
    }
    free(tmp);
  }
  MFTLCplMarkersFree(&m);
  R->qbox = (double *) MFTLMallocOrDie(6 * (size_t) R->n_loc * sizeof(double), "request boxes");
  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    const cpl_locinfo *li = &c->li[R->locs[wl]];
    for (int k = 0; k < 3; ++k)
    {
      int64_t lo = INT64_MAX, hi = INT64_MIN;
      for (int64_t i = 0; i < R->n_bins; ++i)
      {
        lo = MFTL_MIN(lo, R->bin_idx[3 * i + k]);
        hi = MFTL_MAX(hi, R->bin_idx[3 * i + k]);
      }
      R->qbox[6 * wl + k] = (R->n_bins > 0) ? d->global_min[k] + (double) lo * R->bin[k] - li->reach[k] - R->skin[k] : 1.0;
      R->qbox[6 * wl + 3 + k] = (R->n_bins > 0) ? d->global_min[k] + (double) (hi + 1) * R->bin[k] + li->reach[k] + R->skin[k] : 0.0;
    }
  }

  /* ── does the directory reach far enough? (one MAX of three doubles) ── */
  {
    double blo[3], bhi[3], need[3] = {0.0, 0.0, 0.0};
    own_box(c, blo, bhi);
    double rmax[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < R->n_loc; ++i)
      for (int k = 0; k < 3; ++k) rmax[k] = fmax(rmax[k], c->li[R->locs[i]].reach[k]);
    for (int64_t i = 0; i < R->n_bins; ++i)
      for (int k = 0; k < 3; ++k)
      {
        const double qlo = d->global_min[k] + (double) R->bin_idx[3 * i + k] * R->bin[k] - rmax[k] - R->skin[k];
        const double qhi = qlo + R->bin[k] + 2.0 * (rmax[k] + R->skin[k]);
        double best = MFTL_DBL_MAX;
        for (int t = -1; t <= 1; ++t)
        {
          if (t != 0 && !d->is_periodic[k]) continue;
          const double sh = (double) t * d->domain_size[k];
          const double e = fmax(0.0, fmax(blo[k] + sh - qlo, qhi - bhi[k] - sh));
          best = fmin(best, e);
        }
        need[k] = fmax(need[k], best);
      }
    MFTLCplAllreduceMax(c->comm, need, 3);
    bool rebuild = false;
    for (int k = 0; k < 3; ++k)
      if (need[k] > c->dir_reach[k])
      {
        c->dir_reach[k] = 1.25 * need[k];
        rebuild = true;
      }
    if (rebuild) build_directory(c);
  }

  /* ── queries: one per (bin, location) for a lattice, per (bin, location, provider, image) for points ── */
  int *asked = (int *) MFTLMallocOrDie((size_t) c->nproc * sizeof(int), "asked providers");
  int n_dest = 0, cap_dest = 0;
  int *dest = MFTL_NULL;
  cpl_buf *dbuf = MFTL_NULL;
  int64_t **dsub = MFTL_NULL; /* per destination: list of query ids */
  int64_t *dsub_n = MFTL_NULL, *dsub_cap = MFTL_NULL;
  int local_status = MFTL_COUPLING_OK;
  int *cand = MFTL_NULL, cand_cap = 0;

#define ADD_DEST_QUERY(prov, qid)                                                                                          \
  do                                                                                                                       \
  {                                                                                                                        \
    int _at = -1;                                                                                                          \
    for (int _i = 0; _i < n_dest; ++_i)                                                                                    \
      if (dest[_i] == (prov)) _at = _i;                                                                                    \
    if (_at < 0)                                                                                                           \
    {                                                                                                                      \
      if (n_dest == cap_dest)                                                                                              \
      {                                                                                                                    \
        cap_dest = MFTLGrowCapacity(cap_dest, n_dest + 1);                                                                 \
        dest = (int *) MFTLReallocOrDie(dest, (size_t) cap_dest * sizeof(int), "request dest");                            \
        dsub = (int64_t **) MFTLReallocOrDie(dsub, (size_t) cap_dest * sizeof(int64_t *), "request dest");                 \
        dsub_n = (int64_t *) MFTLReallocOrDie(dsub_n, (size_t) cap_dest * sizeof(int64_t), "request dest");                \
        dsub_cap = (int64_t *) MFTLReallocOrDie(dsub_cap, (size_t) cap_dest * sizeof(int64_t), "request dest");            \
      }                                                                                                                    \
      _at = n_dest++;                                                                                                      \
      dest[_at] = (prov);                                                                                                  \
      dsub[_at] = MFTL_NULL;                                                                                               \
      dsub_n[_at] = 0;                                                                                                     \
      dsub_cap[_at] = 0;                                                                                                   \
    }                                                                                                                      \
    GROW(dsub[_at], dsub_cap[_at], dsub_n[_at] + 1, int64_t);                                                              \
    dsub[_at][dsub_n[_at]++] = (qid);                                                                                      \
  } while (0)

  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    const int loc = R->locs[wl];
    const cpl_location *L = &g->loc[loc];
    const cpl_locinfo *li = &c->li[loc];
    for (int64_t bi = 0; bi < R->n_bins; ++bi)
    {
      double Q[6];
      for (int k = 0; k < 3; ++k)
      {
        Q[k] = d->global_min[k] + (double) R->bin_idx[3 * bi + k] * R->bin[k] - li->reach[k] - R->skin[k];
        Q[3 + k] = d->global_min[k] + (double) (R->bin_idx[3 * bi + k] + 1) * R->bin[k] + li->reach[k] + R->skin[k];
      }
      if (cpl_is_lattice(L))
      {
        query_t qq;
        memset(&qq, 0, sizeof qq);
        qq.wl = wl;
        qq.location = loc;
        qq.kind = 0;
        qq.bin = bi;
        memcpy(qq.box, Q, sizeof Q);
        bool empty = false;
        qq.expected = 1;
        for (int k = 0; k < 3; ++k)
        {
          if (!MFTLCplLatRange(L, d, k, Q[k], Q[3 + k], &qq.a[k], &qq.b[k]))
          {
            empty = true;
            break;
          }
          qq.expected *= MFTLCplLatDistinct(L, d, k, qq.a[k], qq.b[k]);
        }
        if (empty) continue;
        GROW(R->q, R->cap_q, R->n_q + 1, query_t);
        const int64_t qid = R->n_q++;
        R->q[qid] = qq;
        int n_asked = 0;
        /* candidates near Q and its periodic images, then the exact index test */
        int n_cand = 0;
        for (int tz = -1; tz <= 1; ++tz)
          for (int ty = -1; ty <= 1; ++ty)
            for (int tx = -1; tx <= 1; ++tx)
            {
              const int t[3] = {tx, ty, tz};
              double qlo[3], qhi[3];
              bool okimg = true;
              for (int k = 0; k < 3; ++k)
              {
                if (t[k] != 0 && !d->is_periodic[k]) okimg = false;
                qlo[k] = Q[k] - (double) t[k] * d->domain_size[k];
                qhi[k] = Q[3 + k] - (double) t[k] * d->domain_size[k];
              }
              if (!okimg) continue;
              int *got = MFTL_NULL, gcap = 0;
              const int ng = MFTLCplDirIndexQuery(&c->dix[loc], c->dir, qlo, qhi, &got, &gcap);
              if (n_cand + ng > cand_cap)
              {
                cand_cap = MFTL_MAX(2 * cand_cap, n_cand + ng);
                cand = (int *) MFTLReallocOrDie(cand, (size_t) cand_cap * sizeof(int), "request candidates");
              }
              for (int i = 0; i < ng; ++i) cand[n_cand++] = got[i];
              free(got);
            }
        for (int ci = 0; ci < n_cand; ++ci)
        {
          const cpl_dir_entry *E = &c->dir[cand[ci]];
          bool hit = true;
          for (int k = 0; k < 3 && hit; ++k)
          {
            int64_t lo2[2], hi2[2];
            if (MFTLCplLatIntersect(L, d, k, qq.a[k], qq.b[k], E->ilo[k], E->ihi[k], lo2, hi2, 2) == 0) hit = false;
          }
          if (!hit) continue;
          /* Counting is a coverage proof only for disjoint ownership boxes.
           * A duplicated MASKED cell contributes to got but has no returned
           * key: without this check it can cancel a hole elsewhere exactly. */
          for (int cj = 0; cj < ci; ++cj)
          {
            const cpl_dir_entry *F = &c->dir[cand[cj]];
            if (F->provider == E->provider) continue; /* locally validated */
            bool overlap = true;
            for (int k = 0; k < 3 && overlap; ++k)
            {
              const int64_t lo = MFTL_MAX(E->ilo[k], F->ilo[k]), hi = MFTL_MIN(E->ihi[k], F->ihi[k]);
              int64_t il[2], ih[2];
              if (MFTLCplLatIntersect(L, d, k, qq.a[k], qq.b[k], lo, hi, il, ih, 2) == 0) overlap = false;
            }
            if (overlap) local_status = MFTL_COUPLING_INVALID;
          }
          bool seen = false;
          for (int i = 0; i < n_asked; ++i)
            if (asked[i] == E->provider) seen = true;
          if (seen) continue;
          asked[n_asked++] = E->provider;
          ADD_DEST_QUERY(E->provider, qid);
        }
      }
      else
      {
        /* Clip to the domain on the walls; ask every owner of every image that meets it. */
        bool empty = false;
        for (int k = 0; k < 3; ++k)
          if (!d->is_periodic[k])
          {
            Q[k] = fmax(Q[k], d->global_min[k]);
            Q[3 + k] = fmin(Q[3 + k], d->global_max[k]);
            if (Q[k] > Q[3 + k]) empty = true;
          }
        if (empty) continue;
        double sliver[3];
        point_sliver(c, loc, sliver);
        /* the queries: one per (provider, image) of every ownership box near Q */
        for (int tz = -1; tz <= 1; ++tz)
          for (int ty = -1; ty <= 1; ++ty)
            for (int tx = -1; tx <= 1; ++tx)
            {
              const int t[3] = {tx, ty, tz};
              double qlo[3], qhi[3];
              bool okimg = true;
              for (int k = 0; k < 3; ++k)
              {
                if (t[k] != 0 && !d->is_periodic[k]) okimg = false;
                qlo[k] = Q[k] - (double) t[k] * d->domain_size[k];
                qhi[k] = Q[3 + k] - (double) t[k] * d->domain_size[k];
              }
              if (!okimg) continue;
              int *got = MFTL_NULL, gcap = 0;
              const int ng = MFTLCplDirIndexQuery(&c->dix[loc], c->dir, qlo, qhi, &got, &gcap);
              for (int gi = 0; gi < ng; ++gi)
              {
                const cpl_dir_entry *E = &c->dir[got[gi]];
                /* one query per (provider, image) */
                int64_t found = -1;
                for (int64_t qi = R->n_q - 1; qi >= 0; --qi)
                {
                  const query_t *x = &R->q[qi];
                  if (x->bin != bi || x->wl != wl) break;
                  if (x->provider == E->provider && x->shift[0] == t[0] && x->shift[1] == t[1] && x->shift[2] == t[2]) found = qi;
                }
                if (found >= 0) continue;
                query_t qq;
                memset(&qq, 0, sizeof qq);
                qq.wl = wl;
                qq.location = loc;
                qq.kind = 1;
                qq.provider = E->provider;
                qq.bin = bi;
                for (int k = 0; k < 3; ++k)
                {
                  qq.shift[k] = t[k];
                  qq.box[k] = qlo[k];
                  qq.box[3 + k] = qhi[k];
                }
                GROW(R->q, R->cap_q, R->n_q + 1, query_t);
                R->q[R->n_q] = qq;
                ADD_DEST_QUERY(E->provider, R->n_q);
                R->n_q++;
              }
              free(got);
            }
        /* and is every part of Q owned by somebody? The cover answers, in the
         * image of Q that lies in this rank's frame. */
        int n_rem = 0, cap_rem = 0;
        double *rem = MFTL_NULL;
        {
          const cpl_cover *cv = &c->cover[loc];
          double best[6];
          memcpy(best, Q, sizeof best);
          for (int tz = -1; tz <= 1; ++tz)
            for (int ty = -1; ty <= 1; ++ty)
              for (int tx = -1; tx <= 1; ++tx)
              {
                const int t[3] = {tx, ty, tz};
                bool inside = true;
                double sh[6];
                for (int k = 0; k < 3; ++k)
                {
                  if (t[k] != 0 && !d->is_periodic[k]) inside = false;
                  sh[k] = Q[k] + (double) t[k] * d->domain_size[k];
                  sh[3 + k] = Q[3 + k] + (double) t[k] * d->domain_size[k];
                  const double rlo = cv->lo[k], rhi = cv->lo[k] + (double) cv->dims[k] * cv->cell[k];
                  if (sh[k] < rlo - sliver[k] || sh[3 + k] > rhi + sliver[k]) inside = false;
                }
                if (inside) memcpy(best, sh, sizeof best);
              }
          MFTLCplCoverQuery(cv, best, best + 3, sliver, &rem, &n_rem, &cap_rem);
        }
        for (int r = 0; r < n_rem; ++r)
        {
          MFTL_RequestRegion reg;
          memset(&reg, 0, sizeof reg);
          reg.location = loc;
          reg.piece = -1;
          reg.requester = c->rank;
          memcpy(reg.box, &rem[6 * r], sizeof reg.box);
          add_region(&R->unprovided, &R->n_unprovided, &R->cap_unprovided, &reg);
          if (local_status < MFTL_COUPLING_NEED_COVERAGE) local_status = MFTL_COUPLING_NEED_COVERAGE;
        }
        free(rem);
      }
    }
  }
#undef ADD_DEST_QUERY
  free(asked);
  free(cand);

  /* ── messages: my ghost boxes, then the queries ── */
  dbuf = (cpl_buf *) MFTLCallocOrDie((size_t) MFTL_MAX(n_dest, 1), sizeof(cpl_buf), "request buffers");
  int64_t hb[6 * 32];
  for (int i = 0; i < n_dest; ++i)
  {
    cpl_buf *b = &dbuf[i];
    int32_t n_halo = 0;
    const size_t at_nhalo = b->n;
    PUT(b, n_halo);
    for (int wl = 0; wl < R->n_loc; ++wl)
    {
      const int loc = R->locs[wl];
      if (!cpl_is_lattice(&g->loc[loc])) continue;
      const int nh = halo_boxes(g, loc, hb, 32);
      for (int h = 0; h < nh; ++h)
      {
        const int32_t l32 = loc;
        PUT(b, l32);
        MFTLCplBufPut(b, &hb[6 * h], 6 * sizeof(int64_t));
        ++n_halo;
      }
    }
    memcpy(b->data + at_nhalo, &n_halo, sizeof n_halo);
    const int32_t n_sub = (int32_t) dsub_n[i];
    PUT(b, n_sub);
    for (int64_t s = 0; s < dsub_n[i]; ++s)
    {
      const query_t *q = &R->q[dsub[i][s]];
      const int32_t hdr[4] = {(int32_t) dsub[i][s], q->location, q->kind, q->wl};
      MFTLCplBufPut(b, hdr, sizeof hdr);
      if (q->kind == 0)
      {
        MFTLCplBufPut(b, q->a, sizeof q->a);
        MFTLCplBufPut(b, q->b, sizeof q->b);
      }
      else
        MFTLCplBufPut(b, q->box, sizeof q->box);
    }
    R->n_sent += n_sub;
  }

  /* NBX for everyone else; my own queries are answered in place. */
  int self_at = -1;
  int n_out = 0;
  int *odest = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(int), "request send");
  unsigned char **osb = (unsigned char **) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "request send");
  size_t *osz = (size_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "request send");
  for (int i = 0; i < n_dest; ++i)
  {
    if (dest[i] == c->rank)
    {
      self_at = i;
      continue;
    }
    odest[n_out] = dest[i];
    osb[n_out] = dbuf[i].data;
    osz[n_out] = dbuf[i].n;
    ++n_out;
  }
  cpl_msg *in = MFTL_NULL;
  int n_in = 0;
  MFTLCplNbx(c->comm, CPL_TAG_REQUEST, n_out, odest, osb, osz, &in, &n_in, &c->stats.bytes_discovery, &c->stats.messages);

  /* Serve: requesters in rank order, myself at my position. */
  cpl_buf *replies = (cpl_buf *) MFTLCallocOrDie((size_t) MFTL_MAX(n_in, 1), sizeof(cpl_buf), "replies");
  cpl_buf self_reply = {MFTL_NULL, 0, 0};
  {
    int i = 0;
    bool self_done = (self_at < 0);
    while (i < n_in || !self_done)
    {
      if (!self_done && (i == n_in || in[i].peer > c->rank))
      {
        serve(c, R, c->rank, dbuf[self_at].data, dbuf[self_at].n, &self_reply);
        self_done = true;
        continue;
      }
      serve(c, R, in[i].peer, in[i].data, in[i].bytes, &replies[i]);
      ++i;
    }
  }
  int *rdest = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * sizeof(int), "reply dest");
  unsigned char **rsb = (unsigned char **) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * sizeof(unsigned char *), "reply send");
  size_t *rsz = (size_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_in, 1) * sizeof(size_t), "reply send");
  for (int i = 0; i < n_in; ++i)
  {
    rdest[i] = in[i].peer;
    rsb[i] = replies[i].data;
    rsz[i] = replies[i].n;
  }
  cpl_msg *rin = (cpl_msg *) MFTLCallocOrDie((size_t) MFTL_MAX(n_out, 1), sizeof(cpl_msg), "reply receive");
  MFTLCplExchangeKnown(c->comm, CPL_TAG_REPLY, n_in, rdest, rsb, rsz, n_out, odest, rin, &c->stats.bytes_discovery, &c->stats.messages);

  /* ── verify, provider by provider in rank order ── */
  {
    int i = 0;
    bool self_done = (self_at < 0);
    while (i < n_out || !self_done)
    {
      int s;
      if (!self_done && (i == n_out || odest[i] > c->rank))
      {
        s = absorb_reply(c, R, c->rank, self_reply.data, self_reply.n);
        self_done = true;
      }
      else
      {
        s = absorb_reply(c, R, odest[i], rin[i].data, rin[i].bytes);
        ++i;
      }
      if (s > local_status) local_status = s;
    }
  }
  for (int64_t qi = 0; qi < R->n_q; ++qi)
  {
    const query_t *q = &R->q[qi];
    if (q->kind != 0) continue;
    if (q->got > q->expected)
    {
      fprintf(stderr, "MFTL coupling: location '%s': %lld samples claimed where the lattice has %lld (duplicate ownership)\n", g->loc[q->location].name,
              (long long) q->got, (long long) q->expected);
      local_status = MFTL_COUPLING_INVALID;
    }
    else if (q->got < q->expected)
    {
      MFTL_RequestRegion reg;
      memset(&reg, 0, sizeof reg);
      reg.location = q->location;
      reg.piece = -1;
      reg.requester = c->rank;
      for (int k = 0; k < 3; ++k)
      {
        reg.index_lo[k] = q->a[k];
        reg.index_hi[k] = q->b[k];
      }
      memcpy(reg.box, q->box, sizeof reg.box);
      add_region(&R->unprovided, &R->n_unprovided, &R->cap_unprovided, &reg);
      if (local_status < MFTL_COUPLING_NEED_COVERAGE) local_status = MFTL_COUPLING_NEED_COVERAGE;
    }
  }
  /* One owner per key: the same key from two providers is duplicate ownership. */
  if (R->n_rec > 1) qsort(R->rec, (size_t) R->n_rec, sizeof(rec_t), cmp_rec_key_provider);
  for (int64_t i = 1; i < R->n_rec; ++i)
    if (R->rec[i].wl == R->rec[i - 1].wl && R->rec[i].key == R->rec[i - 1].key && R->rec[i].provider != R->rec[i - 1].provider)
    {
      fprintf(stderr, "MFTL coupling: location '%s': key %llu owned by ranks %d and %d\n", g->loc[R->locs[R->rec[i].wl]].name,
              (unsigned long long) R->rec[i].key, R->rec[i - 1].provider, R->rec[i].provider);
      local_status = MFTL_COUPLING_INVALID;
      break;
    }

  for (int i = 0; i < n_dest; ++i)
  {
    free(dbuf[i].data);
    free(dsub[i]);
  }
  free(dbuf);
  free(dsub);
  free(dsub_n);
  free(dsub_cap);
  free(dest);
  for (int i = 0; i < n_in; ++i) free(replies[i].data);
  free(replies);
  free(self_reply.data);
  MFTLCplMsgsFree(in, n_in);
  free(in);
  MFTLCplMsgsFree(rin, n_out);
  free(rin);
  free(odest);
  free(osb);
  free(osz);
  free(rdest);
  free(rsb);
  free(rsz);

  st = agree(c, local_status);
  R->status = st;
  c->stats.n_requests++;
  if (st == MFTL_COUPLING_NEED_COVERAGE) c->stats.n_need_coverage++;
  if (st == MFTL_COUPLING_OK || st == MFTL_COUPLING_NEED_COVERAGE)
    *out = R;
  else
    MFTLCouplerReleaseRequest(&R);
  return st;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Windows
 * ═══════════════════════════════════════════════════════════════════════════ */

static void free_band(cpl_band *b)
{
  free(b->key);
  free(b->owner);
  free(b->route);
  free(b->piece);
  free(b->elem);
  free(b->measure);
  free(b->h);
  free(b->ipos);
  free(b->idof);
  free(b->ibin_key);
  free(b->ibin_off);
  free(b->src_rank);
  free(b->src_off);
  free(b->src_nonhalo);
  free(b->src_halo);
  free(b->dst_rank);
  free(b->dst_off);
  free(b->dst_nonhalo);
  free(b->dst_halo);
  free(b->pack_served);
  free(b->served_piece);
  free(b->served_elem);
  free(b->served_sample);
  free(b->served_measure);
  free(b->reqbox);
  memset(b, 0, sizeof *b);
}

int MFTLCouplerReleaseWindow(MFTL_CouplingWindow **pw)
{
  if (pw == MFTL_NULL || *pw == MFTL_NULL) return MFTL_SUCCESS;
  MFTL_CouplingWindow *w = *pw;
  if (w->refcount > 0) return MFTL_ERROR;
  for (int i = 0; i < w->n_loc; ++i) free_band(&w->band[i]);
  free(w->band);
  free(w->locs);
  free(w->bin_key);
  free(w->fetch_peer);
  free(w->serve_peer);
  if (w->c)
  {
    w->c->live--;
    w->c->stats.bytes_memory -= (long) w->bytes;
  }
  free(w);
  *pw = MFTL_NULL;
  return MFTL_SUCCESS;
}

/** @brief One image entry while a band is assembled. */
typedef struct
{
  int64_t dof;
  int s[3];
  double pos[3];
  uint64_t bin;
  uint64_t key;
} img_t;

static int cmp_img_dof(const void *a, const void *b)
{
  const img_t *x = (const img_t *) a, *y = (const img_t *) b;
  if (x->dof != y->dof) return (x->dof > y->dof) - (x->dof < y->dof);
  for (int k = 0; k < 3; ++k)
    if (x->s[k] != y->s[k]) return (x->s[k] > y->s[k]) - (x->s[k] < y->s[k]);
  return 0;
}

static int cmp_img_bin(const void *a, const void *b)
{
  const img_t *x = (const img_t *) a, *y = (const img_t *) b;
  if (x->bin != y->bin) return (x->bin > y->bin) - (x->bin < y->bin);
  if (x->key != y->key) return (x->key > y->key) - (x->key < y->key);
  for (int k = 0; k < 3; ++k)
    if (x->s[k] != y->s[k]) return (x->s[k] > y->s[k]) - (x->s[k] < y->s[k]);
  return 0;
}

static int cmp_rec_provider_key(const void *a, const void *b)
{
  const rec_t *x = (const rec_t *) a, *y = (const rec_t *) b;
  if (x->provider != y->provider) return (x->provider > y->provider) - (x->provider < y->provider);
  return (x->key > y->key) - (x->key < y->key);
}

typedef struct
{
  int64_t piece_elem_hi;  ///< piece
  int64_t elem;
  int64_t idx;  ///< position in the concatenated pack lists
} served_t;

static int cmp_served(const void *a, const void *b)
{
  const served_t *x = (const served_t *) a, *y = (const served_t *) b;
  if (x->piece_elem_hi != y->piece_elem_hi) return (x->piece_elem_hi > y->piece_elem_hi) - (x->piece_elem_hi < y->piece_elem_hi);
  if (x->elem != y->elem) return (x->elem > y->elem) - (x->elem < y->elem);
  return (x->idx > y->idx) - (x->idx < y->idx);
}

/** @brief Builds one location's band from a request. Local. */
static int build_band(MFTL_Coupler *c, const MFTL_SupportRequest *R, int wl, rec_t *recs, int64_t n_recs, cpl_band *B)
{
  const MFTL_SolverGrid *g = c->grid;
  const MFTL_Domain *d = &c->ens->domain;
  const int loc = R->locs[wl];
  const cpl_location *L = &g->loc[loc];
  const cpl_locinfo *li = &c->li[loc];
  memset(B, 0, sizeof *B);
  B->location = loc;

  /* ── DOFs: provider order, then non-ghost keys, then ghost keys ── */
  if (n_recs > 1) qsort(recs, (size_t) n_recs, sizeof(rec_t), cmp_rec_provider_key);
  int64_t n_unique = 0;
  for (int64_t i = 0; i < n_recs; ++i)
    if (i == 0 || recs[i].key != recs[i - 1].key || recs[i].provider != recs[i - 1].provider) ++n_unique;

  const size_t nu = (size_t) MFTL_MAX(n_unique, 1);
  B->key = (uint64_t *) MFTLMallocOrDie(nu * sizeof(uint64_t), "band");
  B->owner = (int *) MFTLMallocOrDie(nu * sizeof(int), "band");
  B->route = (unsigned char *) MFTLMallocOrDie(nu, "band");
  B->piece = (int *) MFTLMallocOrDie(nu * sizeof(int), "band");
  B->elem = (int64_t *) MFTLMallocOrDie(nu * sizeof(int64_t), "band");
  B->measure = (double *) MFTLMallocOrDie(nu * sizeof(double), "band");
  B->h = (double *) MFTLMallocOrDie(3 * nu * sizeof(double), "band");
  int64_t *dof_of_rec = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_recs, 1) * sizeof(int64_t), "band");

  int64_t hb[6 * 32];
  const int nh = cpl_is_lattice(L) ? halo_boxes(g, loc, hb, 32) : 0;

  /* the self pack list (for SELF addresses) */
  const pack_t *self_pack = MFTL_NULL;
  for (int p = 0; p < R->n_pack; ++p)
    if (R->pack[p].requester == c->rank && R->pack[p].wl == wl) self_pack = &R->pack[p];

  int64_t n = 0;
  int cap_src = 0;
  B->self_seg = -1;
  for (int64_t i = 0; i < n_recs;)
  {
    const int prov = recs[i].provider;
    int64_t j = i;
    while (j < n_recs && recs[j].provider == prov) ++j;
    /* unique keys of this provider, split by the ghost predicate */
    const int64_t seg_off = n;
    for (int pass = 0; pass < 2; ++pass)
    {
      for (int64_t r = i; r < j; ++r)
      {
        if (r > i && recs[r].key == recs[r - 1].key) continue;
        bool halo = false;
        int64_t rep[3] = {0, 0, 0};
        int bx = -1;
        if (prov != c->rank && nh > 0)
        {
          int64_t jj[3];
          cpl_lat_unkey(L, recs[r].key, jj);
          bx = halo_find(L, d, jj, hb, nh, rep);
          halo = bx >= 0;
        }
        if ((pass == 0) == halo) continue;
        B->key[n] = recs[r].key;
        B->owner[n] = prov;
        B->route[n] = (prov == c->rank) ? CPL_ROUTE_SELF : (halo ? CPL_ROUTE_HALO : CPL_ROUTE_REMOTE);
        B->piece[n] = -1;
        B->elem[n] = -1;
        if (halo)
        {
          /* the bx-th ghost box belongs to the bx-th block with ghosts */
          int seen = 0;
          for (int pc = 0; pc < L->n_pieces; ++pc)
          {
            const cpl_block *blk = &g->blocks[L->piece_ref[pc]];
            if (blk->ghost[0] == 0 && blk->ghost[1] == 0 && blk->ghost[2] == 0) continue;
            if (seen++ == bx)
            {
              B->piece[n] = blk->piece;
              B->elem[n] = cpl_block_elem(blk, rep);
              break;
            }
          }
        }
        if (cpl_is_lattice(L))
        {
          int64_t jj[3];
          cpl_lat_unkey(L, recs[r].key, jj);
          double m = 1.0;
          for (int k = 0; k < 3; ++k)
          {
            const double w = MFTLCplLatWidth(L, k, jj[k]);
            B->h[3 * n + k] = w;
            m *= w;
          }
          B->measure[n] = m;
        }
        else
        {
          B->measure[n] = recs[r].measure;
          for (int k = 0; k < 3; ++k) B->h[3 * n + k] = recs[r].h[k];
        }
        ++n;
      }
    }
    /* map every record of this provider to its DOF: the segment is two
     * key-sorted runs (non-ghost, ghost), so a binary search in each */
    {
      int64_t split = seg_off;
      while (split < n && B->route[split] != CPL_ROUTE_HALO) ++split;
      for (int64_t r = i; r < j; ++r)
      {
        int64_t x = find_u64(B->key + seg_off, split - seg_off, recs[r].key);
        if (x >= 0)
          x += seg_off;
        else
        {
          x = find_u64(B->key + split, n - split, recs[r].key);
          if (x >= 0) x += split;
        }
        dof_of_rec[r] = x;
      }
    }
    if (B->n_src == cap_src)
    {
      cap_src = MFTLGrowCapacity(cap_src, B->n_src + 1);
      B->src_rank = (int *) MFTLReallocOrDie(B->src_rank, (size_t) cap_src * sizeof(int), "band sources");
      B->src_off = (int64_t *) MFTLReallocOrDie(B->src_off, (size_t) cap_src * sizeof(int64_t), "band sources");
      B->src_nonhalo = (int64_t *) MFTLReallocOrDie(B->src_nonhalo, (size_t) cap_src * sizeof(int64_t), "band sources");
      B->src_halo = (int64_t *) MFTLReallocOrDie(B->src_halo, (size_t) cap_src * sizeof(int64_t), "band sources");
    }
    int64_t nonhalo = 0;
    for (int64_t x = seg_off; x < n; ++x)
      if (B->route[x] != CPL_ROUTE_HALO) ++nonhalo;
    B->src_rank[B->n_src] = prov;
    B->src_off[B->n_src] = seg_off;
    B->src_nonhalo[B->n_src] = nonhalo;
    B->src_halo[B->n_src] = (n - seg_off) - nonhalo;
    if (prov == c->rank)
    {
      B->self_seg = B->n_src;
      /* SELF addresses: the self pack list is the same keys in the same order */
      if (self_pack == MFTL_NULL || self_pack->n != n - seg_off)
      {
        free(dof_of_rec);
        return MFTL_COUPLING_INVALID;
      }
      for (int64_t x = 0; x < self_pack->n; ++x)
      {
        if (self_pack->key[x] != B->key[seg_off + x])
        {
          free(dof_of_rec);
          return MFTL_COUPLING_INVALID;
        }
        B->piece[seg_off + x] = self_pack->piece[x];
        B->elem[seg_off + x] = self_pack->elem[x];
      }
    }
    B->n_src++;
    i = j;
  }
  B->n_dof = n;

  /* ── images ──
   * Every periodic image of every sample that lies inside the request's
   * bounding box. That is a superset of the images any requested box holds --
   * an image outside every support is simply never a candidate -- and it is
   * found per sample, in O(1), instead of per (sample, box). */
  int64_t n_img = 0, cap_img = 0;
  img_t *im = MFTL_NULL;
  const double *qb = &R->qbox[6 * wl];
  for (int64_t r = 0; r < n_recs; ++r)
  {
    const int64_t dof = dof_of_rec[r];
    if (dof < 0) continue;
    double base[3];
    if (cpl_is_lattice(L))
    {
      int64_t jj[3];
      cpl_lat_unkey(L, recs[r].key, jj);
      for (int k = 0; k < 3; ++k) base[k] = MFTLCplLatCoord(L, k, jj[k]);
    }
    else
      memcpy(base, recs[r].pos, sizeof base);
    int sh[3][3], ns[3];
    for (int k = 0; k < 3; ++k)
    {
      ns[k] = 0;
      for (int t = -1; t <= 1; ++t)
      {
        if (t != 0 && !d->is_periodic[k]) continue;
        const double x = base[k] + (double) t * d->domain_size[k];
        if (t == 0 && !d->is_periodic[k])
          sh[k][ns[k]++] = 0;
        else if (x >= qb[k] && x <= qb[3 + k])
          sh[k][ns[k]++] = t;
      }
    }
    for (int z = 0; z < ns[2]; ++z)
      for (int y = 0; y < ns[1]; ++y)
        for (int x = 0; x < ns[0]; ++x)
        {
          GROW(im, cap_img, n_img + 1, img_t);
          img_t *e = &im[n_img++];
          const int t3[3] = {sh[0][x], sh[1][y], sh[2][z]};
          e->dof = dof;
          e->key = recs[r].key;
          for (int k = 0; k < 3; ++k)
          {
            e->s[k] = t3[k];
            e->pos[k] = base[k] + (double) t3[k] * d->domain_size[k];
          }
        }
  }
  free(dof_of_rec);
  if (n_img > 1) qsort(im, (size_t) n_img, sizeof(img_t), cmp_img_dof);
  int64_t m = 0;
  for (int64_t i = 0; i < n_img; ++i)
    if (i == 0 || cmp_img_dof(&im[i], &im[i - 1]) != 0) im[m++] = im[i];
  n_img = m;
  for (int k = 0; k < 3; ++k)
  {
    /* bins as wide as the reach, so a support spans at most three per axis;
     * never so narrow that 21-bit keys could overflow, never zero */
    const double floor_b = d->domain_size[k] / (double) (1 << 18);
    B->bsize[k] = (li->reach[k] > 0.0) ? fmax(li->reach[k], floor_b) : d->domain_size[k];
  }
  for (int64_t i = 0; i < n_img; ++i)
  {
    int64_t ib[3];
    for (int k = 0; k < 3; ++k) ib[k] = (int64_t) floor((im[i].pos[k] - d->global_min[k]) / B->bsize[k]);
    im[i].bin = bin_key3(ib[0], ib[1], ib[2]);
  }
  if (n_img > 1) qsort(im, (size_t) n_img, sizeof(img_t), cmp_img_bin);
  B->n_img = n_img;
  B->ipos = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(n_img, 1) * sizeof(double), "band images");
  B->idof = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(n_img, 1) * sizeof(int64_t), "band images");
  int64_t nbins = 0;
  for (int64_t i = 0; i < n_img; ++i)
    if (i == 0 || im[i].bin != im[i - 1].bin) ++nbins;
  B->n_ibins = nbins;
  B->ibin_key = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(nbins, 1) * sizeof(uint64_t), "band bins");
  B->ibin_off = (int64_t *) MFTLMallocOrDie((size_t) (nbins + 1) * sizeof(int64_t), "band bins");
  nbins = 0;
  for (int64_t i = 0; i < n_img; ++i)
  {
    if (i == 0 || im[i].bin != im[i - 1].bin)
    {
      B->ibin_key[nbins] = im[i].bin;
      B->ibin_off[nbins] = i;
      ++nbins;
    }
    memcpy(&B->ipos[3 * i], im[i].pos, 3 * sizeof(double));
    B->idof[i] = im[i].dof;
  }
  B->ibin_off[nbins] = n_img;
  free(im);

  /* ── provider side: pack lists for every requester, and the served table ── */
  int64_t total = 0;
  for (int p = 0; p < R->n_pack; ++p)
    if (R->pack[p].wl == wl) ++B->n_dst, total += R->pack[p].n;
  B->dst_rank = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int), "band dst");
  B->dst_off = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->dst_nonhalo = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->dst_halo = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->pack_served = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(total, 1) * sizeof(int64_t), "band pack");
  served_t *sv = (served_t *) MFTLMallocOrDie((size_t) MFTL_MAX(total, 1) * sizeof(served_t), "band served");
  /* pack lists are already in requester rank order (they were served that way) */
  int nd = 0;
  int64_t at = 0;
  for (int p = 0; p < R->n_pack; ++p)
  {
    const pack_t *pk = &R->pack[p];
    if (pk->wl != wl) continue;
    B->dst_rank[nd] = pk->requester;
    B->dst_off[nd] = at;
    B->dst_nonhalo[nd] = pk->n_nonhalo;
    B->dst_halo[nd] = pk->n - pk->n_nonhalo;
    for (int64_t x = 0; x < pk->n; ++x)
    {
      sv[at + x].piece_elem_hi = pk->piece[x];
      sv[at + x].elem = pk->elem[x];
      sv[at + x].idx = at + x;
    }
    at += pk->n;
    ++nd;
  }
  if (total > 1) qsort(sv, (size_t) total, sizeof(served_t), cmp_served);
  int64_t ns = 0;
  for (int64_t i = 0; i < total; ++i)
    if (i == 0 || sv[i].piece_elem_hi != sv[i - 1].piece_elem_hi || sv[i].elem != sv[i - 1].elem) ++ns;
  B->n_served = ns;
  B->served_piece = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(ns, 1) * sizeof(int), "served");
  B->served_elem = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(ns, 1) * sizeof(int64_t), "served");
  B->served_sample = (int64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(ns, 1) * sizeof(int64_t), "served");
  B->served_measure = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(ns, 1) * sizeof(double), "served");
  ns = 0;
  for (int64_t i = 0; i < total; ++i)
  {
    if (i == 0 || sv[i].piece_elem_hi != sv[i - 1].piece_elem_hi || sv[i].elem != sv[i - 1].elem)
    {
      B->served_piece[ns] = (int) sv[i].piece_elem_hi;
      B->served_elem[ns] = sv[i].elem;
      ++ns;
    }
    B->pack_served[sv[i].idx] = ns - 1;
  }
  free(sv);
  /* sample index and measure of each served DOF, from any pack entry naming it */
  at = 0;
  for (int p = 0; p < R->n_pack; ++p)
  {
    const pack_t *pk = &R->pack[p];
    if (pk->wl != wl) continue;
    for (int64_t x = 0; x < pk->n; ++x)
    {
      const int64_t s = B->pack_served[at + x];
      B->served_sample[s] = pk->sample[x];
      B->served_measure[s] = pk->meas[x];
    }
    at += pk->n;
  }

  /* the request boxes this rank received for this location (band masks) */
  int64_t nrb = 0;
  for (int64_t i = 0; i < R->n_rbox; ++i)
    if (R->rbox_wl[i] == wl) ++nrb;
  B->n_reqbox = (int) nrb;
  B->reqbox = (double *) MFTLMallocOrDie(6 * (size_t) MFTL_MAX(nrb, 1) * sizeof(double), "request boxes");
  nrb = 0;
  for (int64_t i = 0; i < R->n_rbox; ++i)
    if (R->rbox_wl[i] == wl) memcpy(&B->reqbox[6 * nrb++], &R->rbox[6 * i], 6 * sizeof(double));
  return MFTL_COUPLING_OK;
}

static size_t band_bytes(const cpl_band *b)
{
  return (size_t) b->n_dof * (8 + 4 + 1 + 4 + 8 + 8 + 24) + (size_t) b->n_img * 32 + (size_t) b->n_ibins * 16 + (size_t) b->n_served * 28;
}

static int cmp_int(const void *a, const void *b)
{
  const int x = *(const int *) a, y = *(const int *) b;
  return (x > y) - (x < y);
}

int MFTLCouplerPrepareWindow(MFTL_Coupler *c, const MFTL_SupportRequest *Rc, MFTL_CouplingWindow **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = MFTL_COUPLING_OK;
  if (Rc == MFTL_NULL || out == MFTL_NULL || Rc->c != c || c->active)
    local = MFTL_COUPLING_INVALID;
  else if (Rc->status != MFTL_COUPLING_OK)
    local = MFTL_COUPLING_NEED_COVERAGE;
  else if (!grid_current(c) || Rc->grid_epoch != c->grid_epoch)
    local = MFTL_COUPLING_STALE;

  MFTL_CouplingWindow *w = MFTL_NULL;
  if (local == MFTL_COUPLING_OK)
  {
    MFTL_SupportRequest *R = (MFTL_SupportRequest *) Rc; /* records are re-sorted in place */
    w = (MFTL_CouplingWindow *) MFTLCallocOrDie(1, sizeof(MFTL_CouplingWindow), "window");
    w->c = c;
    w->grid_epoch = c->grid_epoch;
    w->n_loc = R->n_loc;
    w->locs = (int *) MFTLMallocOrDie((size_t) R->n_loc * sizeof(int), "window");
    memcpy(w->locs, R->locs, (size_t) R->n_loc * sizeof(int));
    memcpy(w->skin, R->skin, sizeof w->skin);
    memcpy(w->bin, R->bin, sizeof w->bin);
    w->n_bins = R->n_bins;
    w->bin_key = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(R->n_bins, 1) * sizeof(uint64_t), "window bins");
    memcpy(w->bin_key, R->bin_key, (size_t) R->n_bins * sizeof(uint64_t));
    w->band = (cpl_band *) MFTLCallocOrDie((size_t) R->n_loc, sizeof(cpl_band), "window bands");

    /* records are sorted by (wl, key, provider, q) from the request */
    if (R->n_rec > 1) qsort(R->rec, (size_t) R->n_rec, sizeof(rec_t), cmp_rec_key_provider);
    int64_t i = 0;
    for (int wl = 0; wl < R->n_loc && local == MFTL_COUPLING_OK; ++wl)
    {
      int64_t j = i;
      while (j < R->n_rec && R->rec[j].wl == wl) ++j;
      local = build_band(c, R, wl, R->rec ? &R->rec[i] : MFTL_NULL, j - i, &w->band[wl]);
      i = j;
    }
    if (local == MFTL_COUPLING_OK)
    {
      /* peers: union over the bands */
      int nf = 0, ns = 0, capf = 0, caps = 0;
      for (int wl = 0; wl < w->n_loc; ++wl)
      {
        const cpl_band *B = &w->band[wl];
        for (int s = 0; s < B->n_src; ++s)
          if (B->src_rank[s] != c->rank)
          {
            GROW(w->fetch_peer, capf, nf + 1, int);
            w->fetch_peer[nf++] = B->src_rank[s];
          }
        for (int s = 0; s < B->n_dst; ++s)
          if (B->dst_rank[s] != c->rank)
          {
            GROW(w->serve_peer, caps, ns + 1, int);
            w->serve_peer[ns++] = B->dst_rank[s];
          }
      }
      if (nf > 1) qsort(w->fetch_peer, (size_t) nf, sizeof(int), cmp_int);
      if (ns > 1) qsort(w->serve_peer, (size_t) ns, sizeof(int), cmp_int);
      int u = 0;
      for (int x = 0; x < nf; ++x)
        if (x == 0 || w->fetch_peer[x] != w->fetch_peer[x - 1]) w->fetch_peer[u++] = w->fetch_peer[x];
      w->n_fetch_peers = u;
      u = 0;
      for (int x = 0; x < ns; ++x)
        if (x == 0 || w->serve_peer[x] != w->serve_peer[x - 1]) w->serve_peer[u++] = w->serve_peer[x];
      w->n_serve_peers = u;
    }
  }

  const int st = agree(c, local);
  if (st != MFTL_COUPLING_OK)
  {
    if (w)
    {
      w->c = MFTL_NULL;
      MFTLCouplerReleaseWindow(&w);
    }
    return st;
  }
  w->id = ++c->window_counter;
  long self = 0, halo = 0, remote = 0, images = 0, served = 0;
  for (int wl = 0; wl < w->n_loc; ++wl)
  {
    const cpl_band *B = &w->band[wl];
    for (int64_t x = 0; x < B->n_dof; ++x)
    {
      if (B->route[x] == CPL_ROUTE_SELF)
        ++self;
      else if (B->route[x] == CPL_ROUTE_HALO)
        ++halo;
      else
        ++remote;
    }
    images += (long) B->n_img;
    served += (long) B->n_served;
    w->bytes += band_bytes(B);
  }
  c->stats.band_self = self;
  c->stats.band_halo = halo;
  c->stats.band_remote = remote;
  c->stats.band_images = images;
  c->stats.served = served;
  c->stats.fetch_peers = w->n_fetch_peers;
  c->stats.serve_peers = w->n_serve_peers;
  c->stats.n_windows++;
  c->stats.bytes_memory += (long) w->bytes;
  c->live++;
  *out = w;
  return MFTL_COUPLING_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Bindings
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLCouplerReleaseBinding(MFTL_CouplingBinding **pb)
{
  if (pb == MFTL_NULL || *pb == MFTL_NULL) return MFTL_SUCCESS;
  MFTL_CouplingBinding *b = *pb;
  if (b->refcount > 0) return MFTL_ERROR;
  for (int i = 0; i < b->n_loc; ++i)
  {
    free(b->row_ptr[i]);
    free(b->col[i]);
    free(b->w[i]);
  }
  free(b->row_ptr);
  free(b->col);
  free(b->w);
  free(b->loc_band);
  free(b->ref);
  free(b->pos);
  free(b->base_pos);
  free(b->bodies);
  free(b->scratch_keys);
  if (b->win) ((MFTL_CouplingWindow *) b->win)->refcount--;
  if (b->base) ((MFTL_CouplingBinding *) b->base)->refcount--;
  if (b->c) b->c->live--;
  free(b);
  *pb = MFTL_NULL;
  return MFTL_SUCCESS;
}

/** @brief Candidate images of a band in the closed box `[lo, hi]`, in fixed global bin order. */
typedef struct
{
  int64_t n, cap;
  int64_t *img;
} cand_t;

static void gather_candidates(const cpl_band *B, const MFTL_Domain *d, const double lo[3], const double hi[3], cand_t *cd)
{
  cd->n = 0;
  int64_t b0[3], b1[3];
  for (int k = 0; k < 3; ++k)
  {
    b0[k] = (int64_t) floor((lo[k] - d->global_min[k]) / B->bsize[k]);
    b1[k] = (int64_t) floor((hi[k] - d->global_min[k]) / B->bsize[k]);
  }
  for (int64_t z = b0[2]; z <= b1[2]; ++z)
    for (int64_t y = b0[1]; y <= b1[1]; ++y)
      for (int64_t x = b0[0]; x <= b1[0]; ++x)
      {
        const int64_t bi = find_u64(B->ibin_key, B->n_ibins, bin_key3(x, y, z));
        if (bi < 0) continue;
        for (int64_t e = B->ibin_off[bi]; e < B->ibin_off[bi + 1]; ++e)
        {
          const double *p = &B->ipos[3 * e];
          if (p[0] < lo[0] || p[0] > hi[0] || p[1] < lo[1] || p[1] > hi[1] || p[2] < lo[2] || p[2] > hi[2]) continue;
          GROW(cd->img, cd->cap, cd->n + 1, int64_t);
          cd->img[cd->n++] = e;
        }
      }
}

/** @brief Finds the window bin containing @p x within the skin; writes the frame position. */
static bool contain(const MFTL_CouplingWindow *w, const MFTL_Domain *d, const double x[3], double frame[3], double flo[3])
{
  for (int tz = 0; tz < 3; ++tz)
    for (int ty = 0; ty < 3; ++ty)
      for (int tx = 0; tx < 3; ++tx)
      {
        const int t[3] = {(tx == 2) ? -1 : tx, (ty == 2) ? -1 : ty, (tz == 2) ? -1 : tz};
        bool ok = true;
        double xi[3];
        for (int k = 0; k < 3; ++k)
        {
          if (t[k] != 0 && !d->is_periodic[k]) ok = false;
          xi[k] = x[k] + (double) t[k] * d->domain_size[k];
        }
        if (!ok) continue;
        int64_t ib[3];
        for (int k = 0; k < 3; ++k) ib[k] = (int64_t) floor((xi[k] - d->global_min[k]) / w->bin[k]);
        for (int dz = 0; dz < 3; ++dz)
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx)
            {
              const int64_t cb[3] = {ib[0] + ((dx == 2) ? -1 : dx), ib[1] + ((dy == 2) ? -1 : dy), ib[2] + ((dz == 2) ? -1 : dz)};
              if (find_u64(w->bin_key, w->n_bins, bin_key3(cb[0], cb[1], cb[2])) < 0) continue;
              bool in = true;
              double lo[3];
              for (int k = 0; k < 3; ++k)
              {
                lo[k] = d->global_min[k] + (double) cb[k] * w->bin[k];
                if (xi[k] < lo[k] - w->skin[k] || xi[k] > lo[k] + w->bin[k] + w->skin[k]) in = false;
              }
              if (!in) continue;
              memcpy(frame, xi, sizeof xi);
              memcpy(flo, lo, sizeof lo);
              return true;
            }
        /* A skin may span arbitrarily many bins. Keep the common nearby-bin
         * path above, then search the occupied bins rather than a potentially
         * enormous cube of empty bins. */
        if (w->skin[0] > w->bin[0] || w->skin[1] > w->bin[1] || w->skin[2] > w->bin[2])
          for (int64_t bi = 0; bi < w->n_bins; ++bi)
          {
            const uint64_t key = w->bin_key[bi], mask = (UINT64_C(1) << 21) - 1;
            bool in = true;
            double lo[3];
            for (int k = 0; k < 3; ++k)
            {
              const int64_t idx = (int64_t) ((key >> (42 - 21 * k)) & mask) - (INT64_C(1) << 20);
              lo[k] = d->global_min[k] + (double) idx * w->bin[k];
              if (xi[k] < lo[k] - w->skin[k] || xi[k] > lo[k] + w->bin[k] + w->skin[k]) in = false;
            }
            if (!in) continue;
            memcpy(frame, xi, sizeof xi);
            memcpy(flo, lo, sizeof lo);
            return true;
          }
      }
  return false;
}

/** @brief Builds every row of a binding. Local; returns the local status. */
static int assemble_rows(MFTL_Coupler *c, MFTL_CouplingBinding *b, const MFTL_CouplingWindow *w)
{
  const MFTL_Domain *d = &c->ens->domain;
  const MFTL_CouplerParams *P = &c->params;
  MFTL_BindingQuality *Q = &b->q;
  memset(Q, 0, sizeof *Q);
  Q->min_weight = 0.0;
  int status = MFTL_COUPLING_OK;

  cand_t cd = {0, 0, MFTL_NULL};
  int cap_c = 0;
  int64_t n_sel = 0, *sel = MFTL_NULL;
  double *X = MFTL_NULL, *q = MFTL_NULL, *wt = MFTL_NULL;

  const bool grow = (P->failure & MFTL_STENCIL_GROW) != 0 && P->max_growth > 1.0;
  const bool lower = (P->failure & MFTL_STENCIL_LOWER_ORDER) != 0 && P->reproduction == MFTL_REPRODUCE_LINEAR;

  for (int lp = 0; lp < b->n_loc; ++lp)
  {
    const cpl_band *B = &w->band[b->loc_band[lp]];
    const cpl_locinfo *li = &c->li[B->location];
    const MFTL_BoundaryPolicy bpol = c->grid->loc[B->location].boundary;
    int64_t cap_e = 0, ne = 0;
    b->row_ptr[lp] = (int64_t *) MFTLMallocOrDie((size_t) (b->n_markers + 1) * sizeof(int64_t), "rows");
    b->col[lp] = MFTL_NULL;
    b->w[lp] = MFTL_NULL;
    b->row_ptr[lp][0] = 0;

    for (int64_t i = 0; i < b->n_markers; ++i)
    {
      const double *x = &b->pos[3 * i];
      double H[3];
      bool ok = true;
      /* At most two gathers per row: the pilot neighbourhood (a variable
       * metric only) and the largest support the row may grow to -- the second
       * skipped when the first already holds it. Growth steps filter. Filtering
       * keeps the global bin order, so the rows are the same bits either way. */
      double Rp[3] = {0.0, 0.0, 0.0};
      if (li->uniform_metric)
      {
        for (int k = 0; k < 3; ++k) H[k] = li->hmax[k];
      }
      else
      {
        /* Shepard average of the sample resolutions over the pilot neighbourhood. */
        double plo[3], phi[3];
        for (int k = 0; k < 3; ++k)
        {
          Rp[k] = P->pilot_factor * li->hmax[k];
          plo[k] = x[k] - Rp[k];
          phi[k] = x[k] + Rp[k];
        }
        gather_candidates(B, d, plo, phi, &cd);
        double sw = 0.0, sh[3] = {0.0, 0.0, 0.0};
        for (int64_t e = 0; e < cd.n; ++e)
        {
          const double *p = &B->ipos[3 * cd.img[e]];
          double rho2 = 0.0;
          for (int k = 0; k < 3; ++k)
          {
            const double t = (p[k] - x[k]) / Rp[k];
            rho2 += t * t;
          }
          if (rho2 >= 1.0) continue;
          const double rho = sqrt(rho2), s = 1.0 - rho;
          const double wgt = s * s * s * s * (4.0 * rho + 1.0);
          const double *hd = &B->h[3 * B->idof[cd.img[e]]];
          sw += wgt;
          for (int k = 0; k < 3; ++k) sh[k] += wgt * hd[k];
        }
        if (!(sw > 0.0))
          ok = false;
        else
          for (int k = 0; k < 3; ++k) H[k] = sh[k] / sw;
      }
      if (ok)
      {
        const double gmax = grow ? P->max_growth : 1.0;
        double slo[3], shi[3];
        bool inside_pilot = !li->uniform_metric;
        for (int k = 0; k < 3; ++k)
        {
          const double Rs = c->kernel.half_width * P->support_scale * gmax * H[k];
          slo[k] = x[k] - Rs;
          shi[k] = x[k] + Rs;
          if (Rs > Rp[k]) inside_pilot = false;
        }
        if (!inside_pilot) gather_candidates(B, d, slo, shi, &cd);
      }

      MFTL_RowQuality rq;
      memset(&rq, 0, sizeof rq);
      int rc = MFTL_COUPLING_BAD_STENCIL;
      bool truncated = false;
      if (ok)
      {
        MFTL_CouplerParams pp = *P;
        for (int pass = 0; pass < (lower ? 2 : 1) && rc != MFTL_COUPLING_OK; ++pass)
        {
          if (pass == 1) pp.reproduction = MFTL_REPRODUCE_CONSTANT;
          double gfac = 1.0;
          for (int step = 0;; ++step)
          {
            double Ht[3], lo[3], hi[3];
            truncated = false;
            for (int k = 0; k < 3; ++k)
            {
              Ht[k] = P->support_scale * gfac * H[k];
              const double R = c->kernel.half_width * Ht[k];
              lo[k] = x[k] - R;
              hi[k] = x[k] + R;
              if (!d->is_periodic[k] && (lo[k] < d->global_min[k] || hi[k] > d->global_max[k])) truncated = true;
            }
            if (cd.n > cap_c)
            {
              cap_c = (int) MFTLGrowCapacity(cap_c, (int) cd.n);
              X = (double *) MFTLReallocOrDie(X, 3 * (size_t) cap_c * sizeof(double), "row candidates");
              q = (double *) MFTLReallocOrDie(q, (size_t) cap_c * sizeof(double), "row candidates");
              wt = (double *) MFTLReallocOrDie(wt, (size_t) cap_c * sizeof(double), "row candidates");
              sel = (int64_t *) MFTLReallocOrDie(sel, (size_t) cap_c * sizeof(int64_t), "row candidates");
            }
            n_sel = 0;
            for (int64_t e = 0; e < cd.n; ++e)
            {
              const double *pe = &B->ipos[3 * cd.img[e]];
              if (pe[0] < lo[0] || pe[0] > hi[0] || pe[1] < lo[1] || pe[1] > hi[1] || pe[2] < lo[2] || pe[2] > hi[2]) continue;
              memcpy(&X[3 * n_sel], pe, 3 * sizeof(double));
              q[n_sel] = B->measure[B->idof[cd.img[e]]];
              sel[n_sel++] = cd.img[e];
            }
            rc = MFTLCplRowWeights(&pp, &c->kernel, x, Ht, (int) n_sel, X, q, wt, &rq, &c->rs);
            rq.n_grow = step;
            if (rc == MFTL_COUPLING_OK && truncated && bpol == MFTL_BOUNDARY_ERROR) rc = MFTL_COUPLING_BAD_STENCIL;
            if (rc == MFTL_COUPLING_OK || !grow || gfac >= P->max_growth) break;
            gfac = fmin(gfac * P->growth_step, P->max_growth);
          }
          if (rc == MFTL_COUPLING_OK && pass == 1) rq.lower_order = true;
        }
      }

      if (rc != MFTL_COUPLING_OK)
      {
        if (Q->n_failed == 0) Q->first_failed_key = b->ref[i].key;
        Q->n_failed++;
        status = MFTL_COUPLING_BAD_STENCIL;
        b->row_ptr[lp][i + 1] = ne;
        continue;
      }
      for (int64_t e = 0; e < n_sel; ++e)
      {
        if (wt[e] == 0.0) continue;
        if (ne == cap_e)
        {
          cap_e = (cap_e == 0) ? 1024 : 2 * cap_e;
          b->col[lp] = (int64_t *) MFTLReallocOrDie(b->col[lp], (size_t) cap_e * sizeof(int64_t), "rows");
          b->w[lp] = (double *) MFTLReallocOrDie(b->w[lp], (size_t) cap_e * sizeof(double), "rows");
        }
        b->col[lp][ne] = B->idof[sel[e]];
        b->w[lp][ne] = wt[e];
        ++ne;
      }
      b->row_ptr[lp][i + 1] = ne;
      Q->n_rows++;
      if (rq.corrected) Q->n_corrected++;
      if (rq.lower_order) Q->n_lower_order++;
      if (rq.n_grow > 0) Q->n_grown++;
      if (truncated) Q->n_truncated++;
      Q->max_amplification = fmax(Q->max_amplification, rq.sum_abs);
      Q->min_weight = fmin(Q->min_weight, rq.min_weight);
      if (rq.corrected) Q->max_condition = fmax(Q->max_condition, rq.condition);
      Q->max_moment_residual = fmax(Q->max_moment_residual, rq.moment_residual);
    }
    Q->n_entries += ne;
    if (b->col[lp] == MFTL_NULL)
    {
      b->col[lp] = (int64_t *) MFTLMallocOrDie(sizeof(int64_t), "rows");
      b->w[lp] = (double *) MFTLMallocOrDie(sizeof(double), "rows");
    }
  }
  free(cd.img);
  free(sel);
  free(X);
  free(q);
  free(wt);
  return status;
}

/** @brief Local spacing check: worst edge over the finest resolution the directory knows nearby. */
static void spacing_local(const MFTL_Coupler *c, double *max_ratio, int64_t *n_off, int *worst_body, int *worst_tri)
{
  int *got = MFTL_NULL, gcap = 0;
  const MFTL_Ensemble *ens = c->ens;
  const MFTL_Domain *d = &ens->domain;
  *max_ratio = 0.0;
  *n_off = 0;
  *worst_body = -1;
  *worst_tri = -1;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double cr[3][3];
      MFTLGetTriangleCorners(obj, t, cr);
      double lmax = 0.0, lo[3], hi[3];
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(cr[0][k], fmin(cr[1][k], cr[2][k]));
        hi[k] = fmax(cr[0][k], fmax(cr[1][k], cr[2][k]));
      }
      for (int e = 0; e < 3; ++e)
      {
        double l2 = 0.0;
        for (int k = 0; k < 3; ++k)
        {
          const double dd = cr[(e + 1) % 3][k] - cr[e][k];
          l2 += dd * dd;
        }
        lmax = fmax(lmax, sqrt(l2));
      }
      double h = MFTL_DBL_MAX;
      for (int tz = -1; tz <= 1; ++tz)
        for (int ty = -1; ty <= 1; ++ty)
          for (int tx = -1; tx <= 1; ++tx)
          {
            const int ts[3] = {tx, ty, tz};
            bool okimg = true;
            double qlo[3], qhi[3];
            for (int k = 0; k < 3; ++k)
            {
              if (ts[k] != 0 && !d->is_periodic[k]) okimg = false;
              qlo[k] = lo[k] - ts[k] * d->domain_size[k];
              qhi[k] = hi[k] - ts[k] * d->domain_size[k];
            }
            if (!okimg) continue;
            for (int l = 0; l < c->n_loc; ++l)
            {
              const int ng = MFTLCplDirIndexQuery(&c->dix[l], c->dir, qlo, qhi, &got, &gcap);
              for (int gi = 0; gi < ng; ++gi)
              {
                const cpl_dir_entry *E = &c->dir[got[gi]];
                h = fmin(h, fmin(E->hmin[0], fmin(E->hmin[1], E->hmin[2])));
              }
            }
          }
      if (h == MFTL_DBL_MAX || !(h > 0.0)) continue;
      const double ratio = lmax / h;
      if (ratio > c->params.sampling_ratio) ++*n_off;
      if (ratio > *max_ratio)
      {
        *max_ratio = ratio;
        *worst_body = obj->global_id;
        *worst_tri = obj->tris.rec[t].global_id;
      }
    }
  }
  free(got);
}

/** @brief A fresh binding shell. */
static MFTL_CouplingBinding *new_binding(MFTL_Coupler *c, const MFTL_CouplingWindow *w, int n_loc, int64_t n)
{
  MFTL_CouplingBinding *b = (MFTL_CouplingBinding *) MFTLCallocOrDie(1, sizeof(MFTL_CouplingBinding), "binding");
  b->c = c;
  b->win = w;
  b->window_id = w->id;
  b->n_loc = n_loc;
  b->loc_band = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_loc, 1) * sizeof(int), "binding");
  b->row_ptr = (int64_t **) MFTLCallocOrDie((size_t) MFTL_MAX(n_loc, 1), sizeof(int64_t *), "binding");
  b->col = (int64_t **) MFTLCallocOrDie((size_t) MFTL_MAX(n_loc, 1), sizeof(int64_t *), "binding");
  b->w = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(n_loc, 1), sizeof(double *), "binding");
  b->n_markers = n;
  b->pos = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(n, 1) * sizeof(double), "binding");
  b->base_pos = (double *) MFTLMallocOrDie(3 * (size_t) MFTL_MAX(n, 1) * sizeof(double), "binding");
  b->ref = (MFTL_MarkerRef *) MFTLMallocOrDie((size_t) MFTL_MAX(n, 1) * sizeof(MFTL_MarkerRef), "binding");
  ((MFTL_CouplingWindow *) w)->refcount++;
  c->live++;
  return b;
}

/** @brief Agrees a bind's status, publishing or discarding the binding. */
static int finish_bind(MFTL_Coupler *c, MFTL_CouplingBinding *b, int local, MFTL_CouplingBinding **out)
{
  const int st = agree(c, local);
  if (st != MFTL_COUPLING_OK)
  {
    if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
    if (b) MFTLCouplerReleaseBinding(&b);
    return st;
  }
  *out = b;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerBind(MFTL_Coupler *c, const MFTL_MarkerSpec *spec, const MFTL_CouplingWindow *w, MFTL_CouplingBinding **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  c->stats.n_binds++;
  int local = MFTL_COUPLING_OK;
  if (spec == MFTL_NULL || w == MFTL_NULL || out == MFTL_NULL || w->c != c || c->active) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && !window_current(w)) local = MFTL_COUPLING_STALE;

  int n_loc = 0;
  int *locs = MFTL_NULL;
  cpl_markers m;
  memset(&m, 0, sizeof m);
  MFTL_CouplingBinding *b = MFTL_NULL;
  if (local == MFTL_COUPLING_OK && !resolve_locations(c, spec, &n_loc, &locs)) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK) local = MFTLCplMarkersGather(c->ens, spec, &m);
  if (local == MFTL_COUPLING_OK)
  {
    b = new_binding(c, w, n_loc, m.n);
    b->kind = spec->kind;
    for (int i = 0; i < n_loc; ++i)
    {
      b->loc_band[i] = -1;
      for (int j = 0; j < w->n_loc; ++j)
        if (w->locs[j] == locs[i]) b->loc_band[i] = j;
      if (b->loc_band[i] < 0) local = MFTL_COUPLING_INVALID;
    }
    memcpy(b->ref, m.ref, (size_t) m.n * sizeof(MFTL_MarkerRef));
    b->n_bodies = m.n_bodies;
    b->bodies = m.bodies;
    m.bodies = MFTL_NULL;
    b->pos_checksum = MFTLCplPositionsChecksum(m.pos, m.n);
    const MFTL_Domain *d = &c->ens->domain;
    for (int64_t i = 0; i < m.n && local == MFTL_COUPLING_OK; ++i)
    {
      double flo[3];
      if (!contain(w, d, &m.pos[3 * i], &b->pos[3 * i], flo)) local = MFTL_COUPLING_STALE;
    }
    if (local == MFTL_COUPLING_OK)
    {
      memcpy(b->base_pos, b->pos, 3 * (size_t) m.n * sizeof(double));
      local = assemble_rows(c, b, w);
    }
    if (local == MFTL_COUPLING_OK && c->params.sampling == MFTL_SAMPLING_ERROR && !spec->interpolation_only && spec->kind != MFTL_MARKERS_USER)
    {
      double r;
      int64_t n_off;
      int wb, wt;
      spacing_local(c, &r, &n_off, &wb, &wt);
      if (n_off > 0) local = MFTL_COUPLING_BAD_SAMPLING;
    }
  }
  free(locs);
  MFTLCplMarkersFree(&m);
  return finish_bind(c, b, local, out);
}

int MFTLCouplerBindStage(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, MFTL_CouplingBinding **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  c->stats.n_stage_binds++;
  int local = MFTL_COUPLING_OK;
  if (base == MFTL_NULL || stage == MFTL_NULL || out == MFTL_NULL || base->c != c || c->active || (base->n_markers > 0 && stage->pos == MFTL_NULL))
    local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && !window_current(base->win)) local = MFTL_COUPLING_STALE;
  if (local == MFTL_COUPLING_OK && base->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(c->ens, base->bodies, base->n_bodies)) local = MFTL_COUPLING_STALE;

  MFTL_CouplingBinding *b = MFTL_NULL;
  if (local == MFTL_COUPLING_OK)
  {
    const MFTL_CouplingWindow *w = base->win;
    const MFTL_Domain *d = &c->ens->domain;
    b = new_binding(c, w, base->n_loc, base->n_markers);
    b->base = base;
    ((MFTL_CouplingBinding *) base)->refcount++;
    b->kind = base->kind;
    memcpy(b->loc_band, base->loc_band, (size_t) base->n_loc * sizeof(int));
    memcpy(b->ref, base->ref, (size_t) base->n_markers * sizeof(MFTL_MarkerRef));
    memcpy(b->base_pos, base->base_pos, 3 * (size_t) base->n_markers * sizeof(double));
    b->n_bodies = base->n_bodies;
    b->bodies = (cpl_body_snap *) MFTLMallocOrDie((size_t) MFTL_MAX(base->n_bodies, 1) * sizeof(cpl_body_snap), "binding bodies");
    memcpy(b->bodies, base->bodies, (size_t) base->n_bodies * sizeof(cpl_body_snap));
    b->pos_checksum = base->pos_checksum;
    for (int64_t i = 0; i < base->n_markers && local == MFTL_COUPLING_OK; ++i)
    {
      const double *x0 = &base->base_pos[3 * i];
      double dv[3], xs[3];
      MFTLGetMinimumImageVector(d, dv, x0, &stage->pos[3 * i]);
      for (int k = 0; k < 3; ++k)
      {
        xs[k] = x0[k] + dv[k];
        if (!isfinite(xs[k])) local = MFTL_COUPLING_INVALID;
      }
      if (local != MFTL_COUPLING_OK) break;
      if (stage->weight) b->ref[i].weight = stage->weight[i];
      /* the base marker's bin first, then any bin; the frame follows the bin */
      double flo[3];
      bool in = true;
      int64_t ib[3];
      for (int k = 0; k < 3; ++k) ib[k] = (int64_t) floor((x0[k] - d->global_min[k]) / w->bin[k]);
      for (int k = 0; k < 3; ++k)
      {
        const double lo = d->global_min[k] + (double) ib[k] * w->bin[k];
        if (xs[k] < lo - w->skin[k] || xs[k] > lo + w->bin[k] + w->skin[k]) in = false;
      }
      if (in && find_u64(w->bin_key, w->n_bins, bin_key3(ib[0], ib[1], ib[2])) >= 0)
        memcpy(&b->pos[3 * i], xs, sizeof xs);
      else if (!contain(w, d, xs, &b->pos[3 * i], flo))
        local = MFTL_COUPLING_STALE;
    }
    if (local == MFTL_COUPLING_OK) local = assemble_rows(c, b, w);
  }
  return finish_bind(c, b, local, out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Transfers
 * ═══════════════════════════════════════════════════════════════════════════ */

void MFTLCouplerReleaseReady(MFTL_TransferReady **pr)
{
  if (pr == MFTL_NULL || *pr == MFTL_NULL) return;
  MFTL_TransferReady *r = *pr;
  if (r->c && r->c->active_ready == r) return;
  if (r->retained)
  {
    ((MFTL_CouplingWindow *) r->win)->refcount--;
    if (r->binding) ((MFTL_CouplingBinding *) r->binding)->refcount--;
    r->c->live--;
  }
  free((*pr)->fields);
  free((*pr)->field_band);
  free(*pr);
  *pr = MFTL_NULL;
}

void MFTLCouplerReleaseFetch(MFTL_CouplingFetch **pf)
{
  if (pf == MFTL_NULL || *pf == MFTL_NULL) return;
  MFTL_CouplingFetch *f = *pf;
  for (int i = 0; i < f->n_fields; ++i) free(f->values[i]);
  free(f->values);
  free(f->field_band);
  free(f->version);
  if (f->win)
  {
    MFTL_CouplingWindow *w = (MFTL_CouplingWindow *) f->win;
    w->refcount--;
    if (w->c) w->c->live--;
  }
  free(f);
  *pf = MFTL_NULL;
}

/** @brief Debug-only: have the bound positions been written without MFTLObjectMarkChanged()? */
static bool positions_moved_silently(const MFTL_Coupler *c, const MFTL_CouplingBinding *b)
{
#ifdef NDEBUG
  (void) c;
  (void) b;
  return false;
#else
  if (b->kind == MFTL_MARKERS_USER) return false;
  MFTL_MarkerSpec s;
  memset(&s, 0, sizeof s);
  s.kind = b->kind;
  cpl_markers m;
  if (MFTLCplMarkersGather(c->ens, &s, &m) != MFTL_COUPLING_OK) return true;
  const bool moved = (m.n != b->n_markers) || MFTLCplPositionsChecksum(m.pos, m.n) != b->pos_checksum;
  MFTLCplMarkersFree(&m);
  if (moved) fprintf(stderr, "MFTL coupling: bound vertex positions changed without MFTLObjectMarkChanged()\n");
  return moved;
#endif
}

int MFTLCouplerPrepareTransfer(MFTL_Coupler *c, const MFTL_TransferSpec *spec, MFTL_TransferReady **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = MFTL_COUPLING_OK;
  if (spec == MFTL_NULL || out == MFTL_NULL || c->active || spec->n_fields < 0 || (spec->n_fields > 0 && spec->fields == MFTL_NULL))
    local = MFTL_COUPLING_INVALID;
  const MFTL_CouplingWindow *w = MFTL_NULL;
  if (local == MFTL_COUPLING_OK)
  {
    if (spec->direction == MFTL_TRANSFER_FETCH)
      w = spec->window;
    else if (spec->direction == MFTL_TRANSFER_SPREAD && spec->binding)
      w = spec->binding->win;
    if (w == MFTL_NULL || w->c != c) local = MFTL_COUPLING_INVALID;
  }
  if (local == MFTL_COUPLING_OK && !window_current(w)) local = MFTL_COUPLING_STALE;
  if (local == MFTL_COUPLING_OK && spec->direction == MFTL_TRANSFER_SPREAD)
  {
    const MFTL_CouplingBinding *b = spec->binding;
    if (spec->form != MFTL_SPREAD_INTEGRATED && spec->form != MFTL_SPREAD_DENSITY) local = MFTL_COUPLING_INVALID;
    if (local == MFTL_COUPLING_OK && b->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(c->ens, b->bodies, b->n_bodies)) local = MFTL_COUPLING_STALE;
    if (local == MFTL_COUPLING_OK && positions_moved_silently(c, b)) local = MFTL_COUPLING_STALE;
  }

  MFTL_TransferReady *r = MFTL_NULL;
  uint64_t sig = CPL_FNV_INIT;
  if (local == MFTL_COUPLING_OK)
  {
    r = (MFTL_TransferReady *) MFTLCallocOrDie(1, sizeof(MFTL_TransferReady), "transfer");
    r->c = c;
    r->direction = spec->direction;
    r->win = w;
    r->binding = (spec->direction == MFTL_TRANSFER_SPREAD) ? spec->binding : MFTL_NULL;
    r->window_id = w->id;
    r->n_fields = spec->n_fields;
    r->form = spec->form;
    r->fields = (MFTL_SolverField *) MFTLMallocOrDie((size_t) MFTL_MAX(spec->n_fields, 1) * sizeof(MFTL_SolverField), "transfer");
    r->field_band = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(spec->n_fields, 1) * sizeof(int), "transfer");
    const int dir = (int) spec->direction, form = (int) spec->form;
    sig = cpl_fnv(sig, &dir, sizeof dir);
    sig = cpl_fnv(sig, &form, sizeof form);
    sig = cpl_fnv(sig, &spec->n_fields, sizeof spec->n_fields);
    sig = cpl_fnv(sig, &w->id, sizeof w->id);
    for (int f = 0; f < spec->n_fields && local == MFTL_COUPLING_OK; ++f)
    {
      const MFTL_SolverField *F = &spec->fields[f];
      r->fields[f] = *F;
      if (r->fields[f].stride == 0) r->fields[f].stride = 1;
      r->field_band[f] = -1;
      for (int j = 0; j < w->n_loc; ++j)
        if (w->locs[j] == F->location) r->field_band[f] = j;
      if (r->field_band[f] < 0 || r->fields[f].stride < 0)
      {
        local = MFTL_COUPLING_INVALID;
        break;
      }
      const int np = c->grid->loc[F->location].n_pieces;
      if (np > 0 && F->data == MFTL_NULL) local = MFTL_COUPLING_INVALID;
      for (int p = 0; p < np && local == MFTL_COUPLING_OK; ++p)
        if (F->data[p] == MFTL_NULL) local = MFTL_COUPLING_INVALID;
      if (spec->direction == MFTL_TRANSFER_SPREAD)
      {
        bool bound = false;
        for (int lp = 0; lp < spec->binding->n_loc; ++lp)
          if (spec->binding->loc_band[lp] == r->field_band[f]) bound = true;
        if (!bound) local = MFTL_COUPLING_INVALID;
      }
      const int fl[2] = {F->location, F->halo_valid ? 1 : 0};
      sig = cpl_fnv(sig, fl, sizeof fl);
    }
  }
  int st = agree(c, local);
  if (st == MFTL_COUPLING_OK && !MFTLCplAgreeHash(c->comm, sig))
  {
    if (c->rank == 0) fprintf(stderr, "MFTL coupling: transfer signatures differ between ranks\n");
    st = MFTL_COUPLING_INVALID;
  }
  if (st != MFTL_COUPLING_OK)
  {
    if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
    MFTLCouplerReleaseReady(&r);
    return st;
  }
  r->id = ++c->transfer_counter;
  r->retained = true;
  ((MFTL_CouplingWindow *) w)->refcount++;
  if (r->binding) ((MFTL_CouplingBinding *) r->binding)->refcount++;
  c->live++;
  *out = r;
  return MFTL_COUPLING_OK;
}

/** @brief Validate every rank before any transfer posts a message. Tokens may
 * have become stale since preparation, or ranks may pass different tokens. */
static int transfer_begin_status(MFTL_Coupler *c, const MFTL_TransferReady *r, MFTL_TransferDirection direction, const double *loads)
{
  int local = MFTL_COUPLING_OK;
  if (!r || r->c != c || r->direction != direction || c->active)
    local = MFTL_COUPLING_INVALID;
  else if (!window_current(r->win))
    local = MFTL_COUPLING_STALE;
  else if (direction == MFTL_TRANSFER_SPREAD)
  {
    const MFTL_CouplingBinding *b = r->binding;
    if (!loads && b->n_markers > 0 && r->n_fields > 0)
      local = MFTL_COUPLING_INVALID;
    else if ((b->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(c->ens, b->bodies, b->n_bodies)) || positions_moved_silently(c, b))
      local = MFTL_COUPLING_STALE;
  }
  const int st = MFTLCplAgreeTransfer(c->comm, local, r ? r->id : 0);
  if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
  return st;
}

/** @brief Doubles a peer exchanges for one field of one band: forward reads the halo flag, reverse sends everything. */
static int64_t seg_count(int64_t nonhalo, int64_t halo, bool halo_valid, bool reverse) { return nonhalo + ((reverse || !halo_valid) ? halo : 0); }

static int find_rank(const int *ranks, int n, int r)
{
  for (int i = 0; i < n; ++i)
    if (ranks[i] == r) return i;
  return -1;
}

static void ensure_staging(MFTL_Coupler *c, int64_t ns, int64_t nr)
{
  if (ns > c->scap)
  {
    c->scap = MFTL_MAX(ns, 2 * c->scap);
    free(c->sbuf);
    c->sbuf = (double *) MFTLMallocOrDie((size_t) c->scap * sizeof(double), "coupling staging");
  }
  if (nr > c->rcap)
  {
    c->rcap = MFTL_MAX(nr, 2 * c->rcap);
    free(c->rbuf);
    c->rbuf = (double *) MFTLMallocOrDie((size_t) c->rcap * sizeof(double), "coupling staging");
  }
}

/** @brief A field's value at (piece, element). */
static inline double *field_at(const MFTL_SolverField *F, int piece, int64_t elem) { return &F->data[piece][elem * F->stride]; }

int MFTLCouplerFetchBegin(MFTL_Coupler *c, const MFTL_TransferReady *r)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = transfer_begin_status(c, r, MFTL_TRANSFER_FETCH, MFTL_NULL);
  if (st != MFTL_COUPLING_OK) return st;
  const MFTL_CouplingWindow *w = r->win;
  MFTL_CouplingFetch *f = (MFTL_CouplingFetch *) MFTLCallocOrDie(1, sizeof(MFTL_CouplingFetch), "fetch");
  f->win = w;
  f->window_id = w->id;
  f->n_fields = r->n_fields;
  f->field_band = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(r->n_fields, 1) * sizeof(int), "fetch");
  f->values = (double **) MFTLCallocOrDie((size_t) MFTL_MAX(r->n_fields, 1), sizeof(double *), "fetch");
  f->version = (uint64_t *) MFTLMallocOrDie((size_t) MFTL_MAX(r->n_fields, 1) * sizeof(uint64_t), "fetch");
  for (int i = 0; i < r->n_fields; ++i)
  {
    f->field_band[i] = r->field_band[i];
    f->version[i] = r->fields[i].version;
    f->values[i] = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(w->band[r->field_band[i]].n_dof, 1) * sizeof(double), "fetch values");
  }
  ((MFTL_CouplingWindow *) w)->refcount++;
  c->live++;

  /* sizes */
  int64_t ns = 0, nr = 0;
  for (int p = 0; p < w->n_serve_peers; ++p)
    for (int i = 0; i < r->n_fields; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->dst_rank, B->n_dst, w->serve_peer[p]);
      if (s >= 0) ns += seg_count(B->dst_nonhalo[s], B->dst_halo[s], r->fields[i].halo_valid, false);
    }
  for (int p = 0; p < w->n_fetch_peers; ++p)
    for (int i = 0; i < r->n_fields; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->src_rank, B->n_src, w->fetch_peer[p]);
      if (s >= 0) nr += seg_count(B->src_nonhalo[s], B->src_halo[s], r->fields[i].halo_valid, false);
    }
  ensure_staging(c, ns, nr);

  /* receives */
  int64_t at = 0;
  for (int p = 0; p < w->n_fetch_peers; ++p)
  {
    int64_t len = 0;
    for (int i = 0; i < r->n_fields; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->src_rank, B->n_src, w->fetch_peer[p]);
      if (s >= 0) len += seg_count(B->src_nonhalo[s], B->src_halo[s], r->fields[i].halo_valid, false);
    }
    if (len > 0) MFTLCplSchedRecv(c->comm, &c->sched, CPL_TAG_FETCH, w->fetch_peer[p], c->rbuf + at, (size_t) len * sizeof(double));
    at += len;
  }
  /* pack and send */
  at = 0;
  for (int p = 0; p < w->n_serve_peers; ++p)
  {
    const int64_t start = at;
    for (int i = 0; i < r->n_fields; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->dst_rank, B->n_dst, w->serve_peer[p]);
      if (s < 0) continue;
      const int64_t len = seg_count(B->dst_nonhalo[s], B->dst_halo[s], r->fields[i].halo_valid, false);
      const MFTL_SolverField *F = &r->fields[i];
      for (int64_t x = 0; x < len; ++x)
      {
        const int64_t sv = B->pack_served[B->dst_off[s] + x];
        c->sbuf[at++] = *field_at(F, B->served_piece[sv], B->served_elem[sv]);
      }
    }
    if (at > start)
    {
      MFTLCplSchedSend(c->comm, &c->sched, CPL_TAG_FETCH, w->serve_peer[p], c->sbuf + start, (size_t) (at - start) * sizeof(double), &c->stats.messages);
      c->stats.bytes_fetch += (long) ((at - start) * (int64_t) sizeof(double));
    }
  }
  c->stats.bytes_fetch += (long) (nr * (int64_t) sizeof(double));

  /* in place: SELF always, HALO when the field's ghosts are valid */
  for (int i = 0; i < r->n_fields; ++i)
  {
    const cpl_band *B = &w->band[r->field_band[i]];
    const MFTL_SolverField *F = &r->fields[i];
    double *v = f->values[i];
    for (int64_t x = 0; x < B->n_dof; ++x)
    {
      if (B->route[x] == CPL_ROUTE_SELF || (B->route[x] == CPL_ROUTE_HALO && F->halo_valid)) v[x] = *field_at(F, B->piece[x], B->elem[x]);
    }
  }
  c->active = 1;
  c->active_ready = r;
  c->active_fetch = f;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerFetchEnd(MFTL_Coupler *c, MFTL_CouplingFetch **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = agree(c, c->active != 1 || out == MFTL_NULL ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK);
  if (st != MFTL_COUPLING_OK) return st;
  MFTLCplSchedWait(&c->sched);
  const MFTL_TransferReady *r = c->active_ready;
  const MFTL_CouplingWindow *w = r->win;
  MFTL_CouplingFetch *f = c->active_fetch;
  int64_t at = 0;
  for (int p = 0; p < w->n_fetch_peers; ++p)
    for (int i = 0; i < r->n_fields; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->src_rank, B->n_src, w->fetch_peer[p]);
      if (s < 0) continue;
      const int64_t len = seg_count(B->src_nonhalo[s], B->src_halo[s], r->fields[i].halo_valid, false);
      if (len > 0) memcpy(&f->values[i][B->src_off[s]], c->rbuf + at, (size_t) len * sizeof(double));
      at += len;
    }
  c->active = 0;
  c->active_ready = MFTL_NULL;
  c->active_fetch = MFTL_NULL;
  c->stats.n_fetches++;
  *out = f;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerInterpolateFetched(const MFTL_CouplingBinding *b, const MFTL_CouplingFetch *f, double *values)
{
  if (b == MFTL_NULL || f == MFTL_NULL || (values == MFTL_NULL && b->n_markers > 0 && f->n_fields > 0)) return MFTL_COUPLING_INVALID;
  if (f->window_id != b->window_id || f->win != b->win) return MFTL_COUPLING_STALE;
  if (!window_current(b->win)) return MFTL_COUPLING_STALE;
  if (b->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(b->c->ens, b->bodies, b->n_bodies)) return MFTL_COUPLING_STALE;
  if (positions_moved_silently(b->c, b)) return MFTL_COUPLING_STALE;
  const int nf = f->n_fields;
  /* Validate the whole batch before writing the first field. */
  for (int i = 0; i < nf; ++i)
  {
    bool bound = false;
    for (int j = 0; j < b->n_loc; ++j)
      if (b->loc_band[j] == f->field_band[i]) bound = true;
    if (!bound) return MFTL_COUPLING_INVALID;
  }
  for (int i = 0; i < nf; ++i)
  {
    int lp = -1;
    for (int j = 0; j < b->n_loc; ++j)
      if (b->loc_band[j] == f->field_band[i]) lp = j;
    if (lp < 0) return MFTL_COUPLING_INVALID;
    const int64_t *rp = b->row_ptr[lp];
    const int64_t *col = b->col[lp];
    const double *wt = b->w[lp];
    const double *v = f->values[i];
    for (int64_t m = 0; m < b->n_markers; ++m)
    {
      double acc = 0.0;
      for (int64_t e = rp[m]; e < rp[m + 1]; ++e) acc += wt[e] * v[col[e]];
      values[m * nf + i] = acc;
    }
  }
  return MFTL_COUPLING_OK;
}

int MFTLCouplerSpreadBegin(MFTL_Coupler *c, const MFTL_TransferReady *r, const double *loads)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = transfer_begin_status(c, r, MFTL_TRANSFER_SPREAD, loads);
  if (st != MFTL_COUPLING_OK) return st;
  const MFTL_CouplingBinding *b = r->binding;
  const MFTL_CouplingWindow *w = r->win;
  const int nf = r->n_fields;

  /* b = W^T F per field, over the band */
  if (c->spread_scratch_n < nf)
  {
    c->spread_scratch = (double **) MFTLReallocOrDie(c->spread_scratch, (size_t) nf * sizeof(double *), "spread scratch");
    c->spread_scratch_cap = (int64_t *) MFTLReallocOrDie(c->spread_scratch_cap, (size_t) nf * sizeof(int64_t), "spread scratch");
    for (int i = c->spread_scratch_n; i < nf; ++i)
    {
      c->spread_scratch[i] = MFTL_NULL;
      c->spread_scratch_cap[i] = 0;
    }
    c->spread_scratch_n = nf;
  }
  for (int i = 0; i < nf; ++i)
  {
    const cpl_band *B = &w->band[r->field_band[i]];
    if (B->n_dof > c->spread_scratch_cap[i])
    {
      free(c->spread_scratch[i]);
      c->spread_scratch_cap[i] = B->n_dof;
      c->spread_scratch[i] = (double *) MFTLMallocOrDie((size_t) B->n_dof * sizeof(double), "spread scratch");
    }
    double *s = c->spread_scratch[i];
    for (int64_t x = 0; x < B->n_dof; ++x) s[x] = 0.0;
    int lp = -1;
    for (int j = 0; j < b->n_loc; ++j)
      if (b->loc_band[j] == r->field_band[i]) lp = j;
    const int64_t *rp = b->row_ptr[lp];
    const int64_t *col = b->col[lp];
    const double *wt = b->w[lp];
    for (int64_t m = 0; m < b->n_markers; ++m)
    {
      const double F = loads[m * nf + i];
      for (int64_t e = rp[m]; e < rp[m + 1]; ++e) s[col[e]] += wt[e] * F;
    }
  }

  /* reverse exchange: I send to my providers, receive from my requesters */
  int64_t ns = 0, nr = 0;
  for (int p = 0; p < w->n_fetch_peers; ++p)
    for (int i = 0; i < nf; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->src_rank, B->n_src, w->fetch_peer[p]);
      if (s >= 0) ns += seg_count(B->src_nonhalo[s], B->src_halo[s], false, true);
    }
  for (int p = 0; p < w->n_serve_peers; ++p)
    for (int i = 0; i < nf; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->dst_rank, B->n_dst, w->serve_peer[p]);
      if (s >= 0) nr += seg_count(B->dst_nonhalo[s], B->dst_halo[s], false, true);
    }
  ensure_staging(c, ns, nr);
  int64_t at = 0;
  for (int p = 0; p < w->n_serve_peers; ++p)
  {
    int64_t len = 0;
    for (int i = 0; i < nf; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->dst_rank, B->n_dst, w->serve_peer[p]);
      if (s >= 0) len += seg_count(B->dst_nonhalo[s], B->dst_halo[s], false, true);
    }
    if (len > 0) MFTLCplSchedRecv(c->comm, &c->sched, CPL_TAG_SPREAD, w->serve_peer[p], c->rbuf + at, (size_t) len * sizeof(double));
    at += len;
  }
  at = 0;
  for (int p = 0; p < w->n_fetch_peers; ++p)
  {
    const int64_t start = at;
    for (int i = 0; i < nf; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->src_rank, B->n_src, w->fetch_peer[p]);
      if (s < 0) continue;
      const int64_t len = seg_count(B->src_nonhalo[s], B->src_halo[s], false, true);
      memcpy(c->sbuf + at, &c->spread_scratch[i][B->src_off[s]], (size_t) len * sizeof(double));
      at += len;
    }
    if (at > start)
    {
      MFTLCplSchedSend(c->comm, &c->sched, CPL_TAG_SPREAD, w->fetch_peer[p], c->sbuf + start, (size_t) (at - start) * sizeof(double), &c->stats.messages);
      c->stats.bytes_spread += (long) ((at - start) * (int64_t) sizeof(double));
    }
  }
  c->stats.bytes_spread += (long) (nr * (int64_t) sizeof(double));
  c->active = 2;
  c->active_ready = r;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerSpreadEnd(MFTL_Coupler *c)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = agree(c, c->active != 2 ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK);
  if (st != MFTL_COUPLING_OK) return st;
  MFTLCplSchedWait(&c->sched);
  const MFTL_TransferReady *r = c->active_ready;
  const MFTL_CouplingWindow *w = r->win;
  const int nf = r->n_fields;

  /* receive offsets per serve peer */
  int64_t *poff = (int64_t *) MFTLMallocOrDie((size_t) (w->n_serve_peers + 1) * sizeof(int64_t), "spread offsets");
  poff[0] = 0;
  for (int p = 0; p < w->n_serve_peers; ++p)
  {
    int64_t len = 0;
    for (int i = 0; i < nf; ++i)
    {
      const cpl_band *B = &w->band[r->field_band[i]];
      const int s = find_rank(B->dst_rank, B->n_dst, w->serve_peer[p]);
      if (s >= 0) len += seg_count(B->dst_nonhalo[s], B->dst_halo[s], false, true);
    }
    poff[p + 1] = poff[p] + len;
  }

  for (int i = 0; i < nf; ++i)
  {
    const cpl_band *B = &w->band[r->field_band[i]];
    const MFTL_SolverField *F = &r->fields[i];
    double *acc = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(B->n_served, 1), sizeof(double), "spread accumulation");
    /* within each peer's message, field i's segment starts after the earlier fields' */
    for (int dsi = 0; dsi < B->n_dst; ++dsi)
    {
      const int src = B->dst_rank[dsi]; /* ascending: the order every owner folds in */
      const int64_t len = seg_count(B->dst_nonhalo[dsi], B->dst_halo[dsi], false, true);
      if (src == c->rank)
      {
        const int ss = B->self_seg;
        const double *s = c->spread_scratch[i] + B->src_off[ss];
        for (int64_t x = 0; x < len; ++x) acc[B->pack_served[B->dst_off[dsi] + x]] += s[x];
        continue;
      }
      const int p = find_rank(w->serve_peer, w->n_serve_peers, src);
      int64_t off = poff[p];
      for (int j = 0; j < i; ++j)
      {
        const cpl_band *Bj = &w->band[r->field_band[j]];
        const int s = find_rank(Bj->dst_rank, Bj->n_dst, src);
        if (s >= 0) off += seg_count(Bj->dst_nonhalo[s], Bj->dst_halo[s], false, true);
      }
      for (int64_t x = 0; x < len; ++x) acc[B->pack_served[B->dst_off[dsi] + x]] += c->rbuf[off + x];
    }
    for (int64_t s = 0; s < B->n_served; ++s)
    {
      const double v = (r->form == MFTL_SPREAD_DENSITY) ? acc[s] / B->served_measure[s] : acc[s];
      *field_at(F, B->served_piece[s], B->served_elem[s]) += v;
    }
    free(acc);
  }
  free(poff);
  c->active = 0;
  c->active_ready = MFTL_NULL;
  c->stats.n_spreads++;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerFetch(MFTL_Coupler *c, const MFTL_CouplingWindow *w, int n_fields, const MFTL_SolverField *fields, MFTL_CouplingFetch **out)
{
  if (out) *out = MFTL_NULL;
  MFTL_TransferSpec s;
  memset(&s, 0, sizeof s);
  s.direction = MFTL_TRANSFER_FETCH;
  s.window = w;
  s.n_fields = out ? n_fields : -1;
  s.fields = fields;
  MFTL_TransferReady *r = MFTL_NULL;
  int st = MFTLCouplerPrepareTransfer(c, &s, &r);
  if (st != MFTL_COUPLING_OK) return st;
  st = MFTLCouplerFetchBegin(c, r);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerFetchEnd(c, out);
  MFTLCouplerReleaseReady(&r);
  return st;
}

int MFTLCouplerInterpolate(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int n_fields, const MFTL_SolverField *fields, double *values)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = MFTL_COUPLING_OK;
  if (!b || b->c != c || n_fields < 0 || (n_fields > 0 && (!fields || (b->n_markers > 0 && !values))))
    local = MFTL_COUPLING_INVALID;
  else if (!window_current(b->win) ||
           (b->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(c->ens, b->bodies, b->n_bodies)) || positions_moved_silently(c, b))
    local = MFTL_COUPLING_STALE;
  for (int i = 0; i < n_fields && local == MFTL_COUPLING_OK; ++i)
  {
    bool bound = false;
    for (int j = 0; j < b->n_loc; ++j)
      if (b->win->locs[b->loc_band[j]] == fields[i].location) bound = true;
    if (!bound) local = MFTL_COUPLING_INVALID;
  }
  int st = agree(c, local);
  if (st != MFTL_COUPLING_OK) return st;
  MFTL_CouplingFetch *f = MFTL_NULL;
  const MFTL_CouplingWindow *w = b ? b->win : MFTL_NULL;
  st = MFTLCouplerFetch(c, w, n_fields, fields, &f);
  if (st != MFTL_COUPLING_OK) return st;
  local = MFTLCouplerInterpolateFetched(b, f, values);
  MFTLCouplerReleaseFetch(&f);
  return agree(c, local);
}

int MFTLCouplerSpread(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int n_fields, const MFTL_SolverField *targets, const double *loads, MFTL_SpreadForm form)
{
  MFTL_TransferSpec s;
  memset(&s, 0, sizeof s);
  s.direction = MFTL_TRANSFER_SPREAD;
  s.binding = b;
  s.n_fields = n_fields;
  s.fields = targets;
  s.form = form;
  MFTL_TransferReady *r = MFTL_NULL;
  int st = MFTLCouplerPrepareTransfer(c, &s, &r);
  if (st != MFTL_COUPLING_OK) return st;
  st = MFTLCouplerSpreadBegin(c, r, loads);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerSpreadEnd(c);
  MFTLCouplerReleaseReady(&r);
  return st;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Inspection
 * ═══════════════════════════════════════════════════════════════════════════ */

int64_t MFTLCouplerBindingMarkerCount(const MFTL_CouplingBinding *b) { return b ? b->n_markers : 0; }

const MFTL_MarkerRef *MFTLCouplerBindingMarkers(const MFTL_CouplingBinding *b) { return b ? b->ref : MFTL_NULL; }

const double *MFTLCouplerBindingPositions(const MFTL_CouplingBinding *b) { return b ? b->pos : MFTL_NULL; }

int MFTLCouplerBindingQuality(const MFTL_CouplingBinding *b, MFTL_BindingQuality *q)
{
  if (b == MFTL_NULL || q == MFTL_NULL) return MFTL_ERROR;
  *q = b->q;
  return MFTL_SUCCESS;
}

int MFTLCouplerBindingRow(const MFTL_CouplingBinding *bc, int loc_pos, int64_t marker, int *n, const uint64_t **keys, const double **weights)
{
  if (bc == MFTL_NULL || n == MFTL_NULL || keys == MFTL_NULL || weights == MFTL_NULL || loc_pos < 0 || loc_pos >= bc->n_loc || marker < 0 ||
      marker >= bc->n_markers)
    return MFTL_ERROR;
  MFTL_CouplingBinding *b = (MFTL_CouplingBinding *) bc;
  const cpl_band *B = &b->win->band[b->loc_band[loc_pos]];
  const int64_t a = b->row_ptr[loc_pos][marker], e = b->row_ptr[loc_pos][marker + 1];
  const int m = (int) (e - a);
  if (m > b->scratch_cap)
  {
    b->scratch_cap = MFTLGrowCapacity(b->scratch_cap, m);
    b->scratch_keys = (uint64_t *) MFTLReallocOrDie(b->scratch_keys, (size_t) b->scratch_cap * sizeof(uint64_t), "row keys");
  }
  for (int i = 0; i < m; ++i) b->scratch_keys[i] = B->key[b->col[loc_pos][a + i]];
  *n = m;
  *keys = b->scratch_keys;
  *weights = &b->w[loc_pos][a];
  return MFTL_SUCCESS;
}

int MFTLCouplerBandMask(const MFTL_CouplingWindow *w, int location, int piece, MFTL_BandMaskKind kind, unsigned char *mask)
{
  if (w == MFTL_NULL || mask == MFTL_NULL || w->c == MFTL_NULL) return MFTL_ERROR;
  const MFTL_Coupler *c = w->c;
  if (!window_current(w)) return MFTL_ERROR;
  const MFTL_SolverGrid *g = c->grid;
  const MFTL_Domain *d = &c->ens->domain;
  if (location < 0 || location >= g->n_loc || piece < 0 || piece >= g->loc[location].n_pieces) return MFTL_ERROR;
  const cpl_location *L = &g->loc[location];
  int wl = -1;
  for (int j = 0; j < w->n_loc; ++j)
    if (w->locs[j] == location) wl = j;
  if (wl < 0) return MFTL_ERROR;
  const cpl_band *B = &w->band[wl];
  const int ref = L->piece_ref[piece];

  if (cpl_is_lattice(L))
  {
    const cpl_block *blk = &g->blocks[ref];
    const int64_t nx = blk->hi[0] - blk->lo[0], ny = blk->hi[1] - blk->lo[1], nz = blk->hi[2] - blk->lo[2];
    const int64_t n = nx * ny * nz;
    memset(mask, 0, (size_t) n);
    if (kind == MFTL_BAND_USED)
    {
      for (int64_t s = 0; s < B->n_served; ++s)
        if (B->served_piece[s] == piece) mask[B->served_sample[s]] = 1;
      return MFTL_SUCCESS;
    }
    for (int64_t k3 = blk->lo[2]; k3 < blk->hi[2]; ++k3)
      for (int64_t j3 = blk->lo[1]; j3 < blk->hi[1]; ++j3)
        for (int64_t i3 = blk->lo[0]; i3 < blk->hi[0]; ++i3)
        {
          const int64_t idx[3] = {i3, j3, k3};
          const int64_t s = (i3 - blk->lo[0]) + nx * ((j3 - blk->lo[1]) + ny * (k3 - blk->lo[2]));
          if (kind == MFTL_BAND_EXPORTED)
          {
            mask[s] = (MFTLCplLatExported(L, idx) && (!blk->active || blk->active[s])) ? 1 : 0;
            continue;
          }
          for (int r = 0; r < B->n_reqbox && !mask[s]; ++r)
          {
            bool in = true;
            for (int k = 0; k < 3 && in; ++k)
            {
              int64_t lo2[2], hi2[2];
              const int ni = MFTLCplLatIntersect(L, d, k, (int64_t) B->reqbox[6 * r + k], (int64_t) B->reqbox[6 * r + 3 + k], idx[k], idx[k] + 1, lo2, hi2, 2);
              if (ni == 0) in = false;
            }
            if (in) mask[s] = 1;
          }
        }
    return MFTL_SUCCESS;
  }

  const cpl_points *p = &g->points[ref];
  memset(mask, 0, (size_t) p->n);
  if (kind == MFTL_BAND_EXPORTED)
  {
    memset(mask, 1, (size_t) p->n);
    return MFTL_SUCCESS;
  }
  if (kind == MFTL_BAND_USED)
  {
    for (int64_t s = 0; s < B->n_served; ++s)
      if (B->served_piece[s] == piece) mask[B->served_sample[s]] = 1;
    return MFTL_SUCCESS;
  }
  for (int64_t i = 0; i < p->n; ++i)
  {
    const double *x = &p->pos[3 * i];
    for (int r = 0; r < B->n_reqbox; ++r)
    {
      const double *bx = &B->reqbox[6 * r];
      if (x[0] >= bx[0] && x[0] <= bx[3] && x[1] >= bx[1] && x[1] <= bx[4] && x[2] >= bx[2] && x[2] <= bx[5])
      {
        mask[p->orig[i]] = 1;
        break;
      }
    }
  }
  return MFTL_SUCCESS;
}

int MFTLCouplerSurfaceSpacing(MFTL_Coupler *c, MFTL_SpacingResult *out)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = agree(c, !out ? MFTL_COUPLING_INVALID : (!grid_current(c) ? MFTL_COUPLING_STALE : MFTL_COUPLING_OK));
  if (st != MFTL_COUPLING_OK) return st;
  double r;
  int64_t n_off;
  int wb, wt;
  spacing_local(c, &r, &n_off, &wb, &wt);
  double rmax;
  int who;
  MFTLCplMaxloc(c->comm, r, &rmax, &who);
  long cnt = (long) n_off;
  MFTLCplAllreduceSumLong(c->comm, &cnt, 1);
  int ids[2] = {wb, wt};
  MFTLCplBcastInt(c->comm, ids, 2, who);
  out->max_ratio = rmax;
  out->n_offending = cnt;
  out->worst_body = (rmax > 0.0) ? ids[0] : -1;
  out->worst_triangle_gid = (rmax > 0.0) ? ids[1] : -1;
  return (c->params.sampling == MFTL_SAMPLING_ERROR && cnt > 0) ? MFTL_COUPLING_BAD_SAMPLING : MFTL_COUPLING_OK;
}

int MFTLCouplerGetStats(const MFTL_Coupler *c, MFTL_CouplerStats *out)
{
  if (c == MFTL_NULL || out == MFTL_NULL) return MFTL_ERROR;
  *out = c->stats;
  out->bytes_memory += (long) ((size_t) c->n_dir * sizeof(cpl_dir_entry));
  return MFTL_SUCCESS;
}

int MFTLCouplerReduceStats(MFTL_Coupler *c, MFTL_CouplerStats *out)
{
  if (c == MFTL_NULL) return MFTL_ERROR;
  if (agree(c, out == MFTL_NULL ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK) != MFTL_COUPLING_OK) return MFTL_ERROR;
  MFTL_CouplerStats s;
  MFTLCouplerGetStats(c, &s);
  long sum[15] = {s.n_directory_builds, s.n_directory_kept, s.n_requests, s.n_need_coverage, s.n_windows, s.n_binds, s.n_stage_binds, s.n_stale,
                  s.n_fetches, s.n_spreads, s.bytes_discovery, s.bytes_fetch, s.bytes_spread, s.messages, s.bytes_memory};
  long mx[7] = {s.band_self, s.band_halo, s.band_remote, s.band_images, s.served, s.fetch_peers, s.serve_peers};
  MFTLCplAllreduceSumLong(c->comm, sum, 15);
  MFTLCplAllreduceMaxLong(c->comm, mx, 7);
  out->n_directory_builds = sum[0];
  out->n_directory_kept = sum[1];
  out->n_requests = sum[2];
  out->n_need_coverage = sum[3];
  out->n_windows = sum[4];
  out->n_binds = sum[5];
  out->n_stage_binds = sum[6];
  out->n_stale = sum[7];
  out->n_fetches = sum[8];
  out->n_spreads = sum[9];
  out->bytes_discovery = sum[10];
  out->bytes_fetch = sum[11];
  out->bytes_spread = sum[12];
  out->messages = sum[13];
  out->bytes_memory = sum[14];
  out->band_self = mx[0];
  out->band_halo = mx[1];
  out->band_remote = mx[2];
  out->band_images = mx[3];
  out->served = mx[4];
  out->fetch_peers = mx[5];
  out->serve_peers = mx[6];
  return MFTL_SUCCESS;
}

void MFTLCouplerResetStats(MFTL_Coupler *c)
{
  if (c == MFTL_NULL) return;
  const long mem = c->stats.bytes_memory;
  memset(&c->stats, 0, sizeof c->stats);
  c->stats.bytes_memory = mem;
}
