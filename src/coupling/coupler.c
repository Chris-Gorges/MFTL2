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
 *     create/update ──> directory shards (ownership and local resolution)
 *     request       ──> bins of marker positions, spatial certificate lookups,
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
#include "mftl_object.h"
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
    b->data = (unsigned char *) MFTLCplRealloc(b->data, cap, "coupling buffer");
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
      (ptr) = (type *) MFTLCplRealloc((ptr), (size_t) _c * sizeof(type), "coupling"); \
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

/** @brief Candidate images of a band in the closed box `[lo, hi]`, in fixed global bin order. */
typedef struct
{
  int64_t n, cap;
  int64_t *img;
} cand_t;

/** @brief Reused candidate workspace; bindings retain only their final rows. */
typedef struct
{
  cand_t candidates;
  int cap_c, cap_img;
  int64_t *sel, *isrc;
  double *X, *q, *wt, *ipar, *icoef;
} bind_scratch;

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
  cpl_dir_index *dix;  ///< [n_loc] compact directory bounds trees
  int n_shard;
  cpl_dir_entry *shard;
  cpl_dir_index *shard_index;

  uint64_t seen_owned, seen_export;
  uint64_t grid_epoch;  ///< bumped when some rank's sample layout or constraints changed: every window is stale
  uint64_t export_epoch;  ///< bumped when only exports changed: a window survives while its served samples stay exported
  uint64_t own_hash;  ///< hash of this rank's directory entries when the directory was built
  uint64_t window_counter;
  uint64_t transfer_counter;
  uint64_t quad_counter;

  int active;  ///< 0 none, 1 fetch, 2 spread
  cpl_sched sched;
  const MFTL_TransferReady *active_ready;
  MFTL_CouplingFetch *active_fetch;
  double *sbuf, *rbuf;
  int64_t scap, rcap;
  double **spread_scratch;
  int64_t *spread_scratch_cap;
  int spread_scratch_n;
  double *reaction;  ///< [6 * fields] this rank's wall reactions of the active spread
  int reaction_cap;
  double *acc;  ///< owner-side accumulation of one field's loads, reused across spreads
  int64_t acc_cap;

  cpl_row_scratch rs;
  bind_scratch bs;
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

/**
 * @brief Is a window current? Sample layout and constraints must be unchanged;
 *        a change of EXPORTS alone is survived when every sample it serves
 *        from this rank is still exported -- checked once per export epoch,
 *        here, and cached. The verdict is this rank's: it is the providers that
 *        can see a sample leave the export, so a collective caller agrees it.
 */
static bool window_current(const MFTL_CouplingWindow *wc)
{
  MFTL_CouplingWindow *w = (MFTL_CouplingWindow *) wc;
  const MFTL_Coupler *c = w->c;
  if (!grid_current(c) || w->grid_epoch != c->grid_epoch) return false;
  if (w->export_epoch == c->export_epoch) return true;
  const MFTL_SolverGrid *g = c->grid;
  for (int wl = 0; wl < w->n_loc; ++wl)
  {
    const cpl_band *B = &w->band[wl];
    const cpl_location *L = &g->loc[B->location];
    if (!cpl_is_lattice(L)) continue; /* a point sample stays exported until its set is re-registered: ownership */
    for (int64_t s = 0; s < B->n_served; ++s)
    {
      const cpl_block *blk = &g->blocks[L->piece_ref[B->served_piece[s]]];
      const int64_t nx = blk->hi[0] - blk->lo[0], ny = blk->hi[1] - blk->lo[1], sm = B->served_sample[s];
      const int64_t idx[3] = {blk->lo[0] + sm % nx, blk->lo[1] + (sm / nx) % ny, blk->lo[2] + sm / (nx * ny)};
      if (!MFTLCplLatExported(L, idx)) return false;
    }
  }
  w->export_epoch = c->export_epoch;
  return true;
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
  double *v = (double *) MFTLCplAlloc(7 * (size_t) MFTL_MAX(nl, 1) * sizeof(double), "locinfo");
  for (int l = 0; l < nl; ++l)
  {
    const cpl_location *L = &g->loc[l];
    v[6 * nl + l] = (L->n_con > 0) ? 1.0 : 0.0;
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
  MFTLCplAllreduceMax(c->comm, v, 7 * nl);

  for (int l = 0; l < nl; ++l)
  {
    cpl_locinfo *li = &c->li[l];
    li->lattice = cpl_is_lattice(&g->loc[l]);
    li->constrained = v[6 * nl + l] > 0.0;
    li->anisotropic = g->loc[l].anisotropic;
    li->uniform_metric = !li->anisotropic; /* cell matrices are averaged even when equal widths would say uniform */
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
    if (li->anisotropic) li->uniform_metric = false;

  }
  MFTLCplFree(v);
}

/**
 * @brief Validates the constrained samples of every location (collective when
 *        any rank has some): locally, that this rank owns each constrained
 *        sample once, that free samples it owns are owned and not themselves
 *        constrained, and that every named owner is a rank; then each claim
 *        "rank r owns free sample j" is sent to r (one NBX), which checks it.
 * @return This rank's verdict; the caller agrees it.
 */
static int validate_constraints(MFTL_Coupler *c)
{
  bool any = false;
  for (int l = 0; l < c->n_loc; ++l)
  {
    MFTLCplFree(c->li[l].con_peer);
    c->li[l].con_peer = MFTL_NULL;
    if (c->li[l].constrained) any = true;
  }
  if (!any) return MFTL_COUPLING_OK;
  MFTL_SolverGrid *g = c->grid;
  int status = MFTL_COUPLING_OK;
  int n_dest = 0, cap_dest = 0;
  int *dest = MFTL_NULL;
  cpl_buf *bufs = MFTL_NULL;
  for (int l = 0; l < c->n_loc; ++l)
  {
    const cpl_location *L = &g->loc[l];
    for (int64_t q = 0; q < L->n_con; ++q)
    {
      int pc;
      int64_t el;
      double m;
      if (!MFTLCplSampleFind(g, l, L->con_key[q], &pc, &el, &m) || (q > 0 && L->con_sorted[q] == L->con_sorted[q - 1]))
      {
        fprintf(stderr, "MFTL coupling: location '%s': constrained sample %llu is not owned here, or listed twice\n", L->name, (unsigned long long) L->con_key[q]);
        status = MFTL_COUPLING_INVALID;
      }
    }
    const int64_t ne = (L->n_con > 0) ? L->con_ptr[L->n_con] : 0;
    if (ne > 0)
    {
      c->li[l].con_peer = (int *) MFTLCplAlloc((size_t) ne * sizeof(int), "constraint peers");
      if (!MFTLCplTranslateRanks(g->comm, c->comm, ne, L->con_owner, c->li[l].con_peer))
      {
        status = MFTL_COUPLING_INVALID;
        continue;
      }
    }
    for (int64_t e = 0; e < ne; ++e)
    {
      const int owner = c->li[l].con_peer[e];
      const uint64_t key = L->con_free[e];
      if (owner >= c->nproc)
      {
        status = MFTL_COUPLING_INVALID;
        continue;
      }
      if (owner == c->rank)
      {
        int pc;
        int64_t el;
        double m;
        if (!MFTLCplSampleFind(g, l, key, &pc, &el, &m) || MFTLCplConstraintFind(L, key) >= 0)
        {
          fprintf(stderr, "MFTL coupling: location '%s': free sample %llu is not owned here, or is itself constrained\n", L->name, (unsigned long long) key);
          status = MFTL_COUPLING_INVALID;
        }
        continue;
      }
      int at = -1;
      for (int i = 0; i < n_dest; ++i)
        if (dest[i] == owner) at = i;
      if (at < 0)
      {
        if (n_dest == cap_dest)
        {
          cap_dest = MFTLGrowCapacity(cap_dest, n_dest + 1);
          dest = (int *) MFTLCplRealloc(dest, (size_t) cap_dest * sizeof(int), "constraint claims");
          bufs = (cpl_buf *) MFTLCplRealloc(bufs, (size_t) cap_dest * sizeof(cpl_buf), "constraint claims");
        }
        at = n_dest++;
        dest[at] = owner;
        memset(&bufs[at], 0, sizeof(cpl_buf));
      }
      const int32_t l32 = l;
      PUT(&bufs[at], l32);
      PUT(&bufs[at], key);
    }
  }
  unsigned char **sb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "constraint claims");
  size_t *sz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "constraint claims");
  for (int i = 0; i < n_dest; ++i)
  {
    sb[i] = bufs[i].data;
    sz[i] = bufs[i].n;
  }
  cpl_msg *in = MFTL_NULL;
  int n_in = 0;
  MFTLCplNbx(c->comm, CPL_TAG_CONSTRAINT_SETUP, n_dest, dest, sb, sz, &in, &n_in, &c->stats.bytes_discovery, &c->stats.messages);
  for (int r = 0; r < n_in; ++r)
  {
    size_t off = 0;
    int32_t l32;
    uint64_t key;
    while (GET(in[r].data, in[r].bytes, &off, l32) && GET(in[r].data, in[r].bytes, &off, key))
    {
      int pc;
      int64_t el;
      double m;
      if (l32 < 0 || l32 >= c->n_loc || !MFTLCplSampleFind(g, l32, key, &pc, &el, &m) || MFTLCplConstraintFind(&g->loc[l32], key) >= 0)
      {
        fprintf(stderr, "MFTL coupling: rank %d names this rank (%d) the owner of free sample %llu, which it does not own freely\n", in[r].peer, c->rank,
                (unsigned long long) key);
        status = MFTL_COUPLING_INVALID;
      }
    }
  }
  MFTLCplMsgsFree(in, n_in);
  MFTLCplFree(in);
  for (int i = 0; i < n_dest; ++i) MFTLCplFree(bufs[i].data);
  MFTLCplFree(bufs);
  MFTLCplFree(dest);
  MFTLCplFree(sb);
  MFTLCplFree(sz);
  return status;
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

/** @brief Rebuild the compact index after replacing the active directory cache. */
static void index_directory(MFTL_Coupler *c)
{
  if (c->dix)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->dix[l]);
  MFTLCplFree(c->dix);
  c->dix = MFTLCplCalloc((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_dir_index), "directory index");
  for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexBuild(&c->dix[l], c->dir, c->n_dir, l);
}

static int cmp_directory_entry(const void *a, const void *b)
{
  /* Entries are zero initialized, including padding, at their provider. */
  return memcmp(a, b, sizeof(cpl_dir_entry));
}

typedef struct
{
  int location;
  int metric;
  double box[6];
} directory_lookup;

/** @brief Fetch directory certificates from spatial shards. Queries are routed
 * by their physical boxes, never by an assumed existing provider list. Geometry
 * is cached only for the current support request; field traffic remains direct. */
static void lookup_directory(MFTL_Coupler *c, const directory_lookup *queries, int64_t n)
{
  const MFTL_Domain *d = &c->ens->domain;
  int nd = 0, dc = 0, rc = 32;
  int *dest = MFTL_NULL, *ranks = MFTLCplAlloc((size_t) rc * sizeof(int), "directory routing");
  cpl_buf *buf = MFTL_NULL;
  for (int64_t q = 0; q < n; ++q)
  {
    int nr = MFTLCplRanksTouching(d, queries[q].box, queries[q].box + 3, ranks, rc);
    if (nr > rc)
    {
      rc = nr;
      ranks = MFTLCplRealloc(ranks, (size_t) rc * sizeof(int), "directory routing");
      nr = MFTLCplRanksTouching(d, queries[q].box, queries[q].box + 3, ranks, rc);
    }
    for (int j = 0; j < nr; ++j)
    {
      int p = 0;
      while (p < nd && dest[p] != ranks[j]) ++p;
      if (p == nd)
      {
        if (nd == dc)
        {
          dc = dc ? 2 * dc : 16;
          dest = MFTLCplRealloc(dest, (size_t) dc * sizeof(int), "directory routing");
          buf = MFTLCplRealloc(buf, (size_t) dc * sizeof(cpl_buf), "directory routing");
        }
        dest[nd] = ranks[j];
        memset(&buf[nd++], 0, sizeof(cpl_buf));
      }
      MFTLCplBufPut(&buf[p], &queries[q], sizeof(directory_lookup));
    }
  }
  MFTLCplFree(ranks);
  unsigned char **sb = MFTLCplAlloc((size_t) MFTL_MAX(nd, 1) * sizeof *sb, "directory lookup");
  size_t *sz = MFTLCplAlloc((size_t) MFTL_MAX(nd, 1) * sizeof *sz, "directory lookup");
  for (int p = 0; p < nd; ++p) sb[p] = buf[p].data, sz[p] = buf[p].n;
  cpl_msg *in = MFTL_NULL;
  int ni = 0;
  MFTLCplNbx(c->comm, CPL_TAG_LOOKUP, nd, dest, sb, sz, &in, &ni, &c->stats.bytes_discovery, &c->stats.messages);
  for (int p = 0; p < nd; ++p) MFTLCplFree(buf[p].data);
  MFTLCplFree(buf); MFTLCplFree(dest); MFTLCplFree(sb); MFTLCplFree(sz);
  dest = MFTLCplAlloc((size_t) MFTL_MAX(ni, 1) * sizeof(int), "directory replies");
  buf = MFTLCplCalloc((size_t) MFTL_MAX(ni, 1), sizeof(cpl_buf), "directory replies");
  sb = MFTLCplAlloc((size_t) MFTL_MAX(ni, 1) * sizeof *sb, "directory replies");
  sz = MFTLCplAlloc((size_t) MFTL_MAX(ni, 1) * sizeof *sz, "directory replies");
  int *got = MFTL_NULL, gc = 0;
  unsigned char *seen = MFTLCplCalloc((size_t) MFTL_MAX(c->n_shard, 1), 1, "directory replies");
  for (int p = 0; p < ni; ++p)
  {
    memset(seen, 0, (size_t) c->n_shard);
    dest[p] = in[p].peer;
    const directory_lookup *qs = (const directory_lookup *) in[p].data;
    for (size_t i = 0; i < in[p].bytes / sizeof(directory_lookup); ++i)
      for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
          for (int x = -1; x <= 1; ++x)
          {
            const int shift[3] = {x, y, z};
            double lo[3], hi[3];
            bool valid = true;
            for (int k = 0; k < 3; ++k)
            {
              if (shift[k] && !d->is_periodic[k]) valid = false;
              lo[k] = qs[i].box[k] - shift[k] * d->domain_size[k];
              hi[k] = qs[i].box[3 + k] - shift[k] * d->domain_size[k];
            }
            if (!valid) continue;
            cpl_dir_index *ix = &c->shard_index[qs[i].location];
            const int ng = qs[i].metric ? MFTLCplDirMetricQuery(ix, c->shard, lo, hi, &got, &gc)
                                       : MFTLCplDirIndexQuery(ix, c->shard, lo, hi, &got, &gc);
            for (int j = 0; j < ng; ++j)
              if (!seen[got[j]])
              {
                seen[got[j]] = 1;
                MFTLCplBufPut(&buf[p], &c->shard[got[j]], sizeof(cpl_dir_entry));
              }
          }
    sb[p] = buf[p].data;
    sz[p] = buf[p].n;
  }
  MFTLCplFree(seen); MFTLCplFree(got);
  MFTLCplMsgsFree(in, ni); MFTLCplFree(in);
  cpl_msg *reply = MFTL_NULL;
  int nreply = 0;
  MFTLCplNbx(c->comm, CPL_TAG_LOOKUP_REPLY, ni, dest, sb, sz, &reply, &nreply, &c->stats.bytes_discovery, &c->stats.messages);
  size_t total = (size_t) c->n_dir;
  for (int p = 0; p < nreply; ++p) total += reply[p].bytes / sizeof(cpl_dir_entry);
  c->dir = MFTLCplRealloc(c->dir, MFTL_MAX(total, 1) * sizeof(cpl_dir_entry), "active directory");
  for (int p = 0; p < nreply; ++p)
  {
    const int add = (int) (reply[p].bytes / sizeof(cpl_dir_entry));
    memcpy(c->dir + c->n_dir, reply[p].data, (size_t) add * sizeof(cpl_dir_entry));
    c->n_dir += add;
  }
  if (c->n_dir > 1) qsort(c->dir, (size_t) c->n_dir, sizeof(cpl_dir_entry), cmp_directory_entry);
  int kept = 0;
  for (int i = 0; i < c->n_dir; ++i)
    if (i == 0 || cmp_directory_entry(&c->dir[i], &c->dir[i - 1])) c->dir[kept++] = c->dir[i];
  c->n_dir = kept;
  MFTLCplMsgsFree(reply, nreply); MFTLCplFree(reply);
  for (int p = 0; p < ni; ++p) MFTLCplFree(buf[p].data);
  MFTLCplFree(buf); MFTLCplFree(dest); MFTLCplFree(sb); MFTLCplFree(sz);
  index_directory(c);
}

/**
 * @brief Builds directory shards: each rank publishes ownership and local metric
 *        influence certificates to the rank boxes they touch
 *        (NBX). Collective.
 */
static void build_directory(MFTL_Coupler *c)
{
  const MFTL_Domain *d = &c->ens->domain;
  cpl_dir_entry *own = MFTL_NULL;
  const int n_own = MFTLCplDirectoryOwnEntries(c->grid, d, c->params.pilot_factor, &own);
  for (int i = 0; i < n_own; ++i) own[i].provider = c->rank;
  c->own_hash = cpl_fnv(cpl_fnv(CPL_FNV_INIT, &n_own, sizeof n_own), own, (size_t) n_own * sizeof(cpl_dir_entry));

  /* One buffer per destination rank, in a sparse list. */
  int n_dest = 0, cap_dest = 0;
  int *dest = MFTL_NULL;
  cpl_buf *bufs = MFTL_NULL;
  int cap_ranks = 32;
  int *ranks = (int *) MFTLCplAlloc((size_t) cap_ranks * sizeof(int), "directory ranks");

  MFTLCplFree(c->dir);
  c->dir = MFTL_NULL;
  int64_t n_dir = 0, cap_dir = 0;

  for (int i = 0; i < n_own; ++i)
  {
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = own[i].mlo[k] - c->params.directory_margin;
      hi[k] = own[i].mhi[k] + c->params.directory_margin;
    }
    int nr = MFTLCplRanksTouching(d, lo, hi, ranks, cap_ranks);
    if (nr > cap_ranks)
    {
      cap_ranks = nr;
      ranks = (int *) MFTLCplRealloc(ranks, (size_t) cap_ranks * sizeof(int), "directory ranks");
      nr = MFTLCplRanksTouching(d, lo, hi, ranks, cap_ranks);
    }
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
          dest = (int *) MFTLCplRealloc(dest, (size_t) cap_dest * sizeof(int), "directory dest");
          bufs = (cpl_buf *) MFTLCplRealloc(bufs, (size_t) cap_dest * sizeof(cpl_buf), "directory bufs");
        }
        at = n_dest++;
        dest[at] = ranks[j];
        memset(&bufs[at], 0, sizeof(cpl_buf));
      }
      MFTLCplBufPut(&bufs[at], &own[i], sizeof(cpl_dir_entry));
    }
  }
  MFTLCplFree(ranks);

  unsigned char **sb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "directory send");
  size_t *sz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "directory send");
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
  MFTLCplFree(in);
  for (int q = 0; q < n_dest; ++q) MFTLCplFree(bufs[q].data);
  MFTLCplFree(bufs);
  MFTLCplFree(dest);
  MFTLCplFree(sb);
  MFTLCplFree(sz);
  MFTLCplFree(own);
  c->n_dir = (int) n_dir;
  if (c->shard_index)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->shard_index[l]);
  MFTLCplFree(c->shard_index);
  MFTLCplFree(c->shard);
  c->n_shard = c->n_dir;
  c->shard = MFTLCplAlloc((size_t) MFTL_MAX(c->n_shard, 1) * sizeof(cpl_dir_entry), "directory shard");
  if (c->n_shard > 0) memcpy(c->shard, c->dir, (size_t) c->n_shard * sizeof(cpl_dir_entry)); /* dir is NULL on a rank providing nothing */
  c->shard_index = MFTLCplCalloc((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_dir_index), "directory shard");
  for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexBuild(&c->shard_index[l], c->shard, c->n_shard, l);
  index_directory(c);
  c->stats.n_directory_builds++;
}

int MFTLCouplerCreate(MFTL_Ensemble *ens, MFTL_SolverGrid *grid, const MFTL_CouplerParams *params, MFTL_Coupler **out)
{
  if (out) *out = MFTL_NULL;
  if (ens == MFTL_NULL) return MFTL_COUPLING_INVALID;
  if (MFTLCplAgreeMax(ens->domain.comm_world, grid == MFTL_NULL || out == MFTL_NULL ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK) != MFTL_COUPLING_OK)
    return MFTL_COUPLING_INVALID;

  MFTL_Coupler *c = (MFTL_Coupler *) MFTLCplCalloc(1, sizeof(MFTL_Coupler), "MFTLCouplerCreate");
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
    MFTLCplFree(c);
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
    MFTLCplFree(c);
    return st;
  }

  c->n_loc = grid->n_loc;
  c->li = (cpl_locinfo *) MFTLCplCalloc((size_t) MFTL_MAX(c->n_loc, 1), sizeof(cpl_locinfo), "locinfo");
  compute_locinfo(c);
  if (agree(c, validate_constraints(c)) != MFTL_COUPLING_OK)
  {
    for (int l = 0; l < c->n_loc; ++l) MFTLCplFree(c->li[l].con_peer);
    MFTLCplFree(c->li);
    MFTLCplCommFree(&c->comm);
    MFTLCplFree(c);
    return MFTL_COUPLING_INVALID;
  }
  build_directory(c);
  c->seen_owned = grid->version_owned;
  c->seen_export = grid->version_export;
  c->grid_epoch = 1;
  c->export_epoch = 1;
  *out = c;
  return MFTL_COUPLING_OK;
}

static void free_transfer_staging(MFTL_Coupler *c)
{
  MFTLCplFree(c->sbuf);
  MFTLCplFree(c->rbuf);
  c->sbuf = c->rbuf = MFTL_NULL;
  c->scap = c->rcap = 0;
  for (int i = 0; i < c->spread_scratch_n; ++i) MFTLCplFree(c->spread_scratch[i]);
  MFTLCplFree(c->spread_scratch);
  MFTLCplFree(c->spread_scratch_cap);
  c->spread_scratch = MFTL_NULL;
  c->spread_scratch_cap = MFTL_NULL;
  c->spread_scratch_n = 0;
  MFTLCplFree(c->reaction);
  c->reaction = MFTL_NULL;
  c->reaction_cap = 0;
  MFTLCplFree(c->acc);
  c->acc = MFTL_NULL;
  c->acc_cap = 0;
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
  MFTLCplFree(c->bs.candidates.img);
  MFTLCplFree(c->bs.sel);
  MFTLCplFree(c->bs.X);
  MFTLCplFree(c->bs.q);
  MFTLCplFree(c->bs.wt);
  MFTLCplFree(c->bs.isrc);
  MFTLCplFree(c->bs.ipar);
  MFTLCplFree(c->bs.icoef);
  if (c->dix)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->dix[l]);
  MFTLCplFree(c->dix);
  if (c->shard_index)
    for (int l = 0; l < c->n_loc; ++l) MFTLCplDirIndexFree(&c->shard_index[l]);
  MFTLCplFree(c->shard_index);
  MFTLCplFree(c->shard);
  MFTLCplFree(c->dir);
  for (int l = 0; l < c->n_loc; ++l) MFTLCplFree(c->li[l].con_peer);
  MFTLCplFree(c->li);
  MFTLCplCommFree(&c->comm);
  MFTLCplFree(c);
  *pc = MFTL_NULL;
}

int MFTLCouplerUpdateGrid(MFTL_Coupler *c)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = (c->active != 0) ? MFTL_COUPLING_INVALID : MFTL_COUPLING_OK;
  if (local == MFTL_COUPLING_OK && MFTLCplGridValidate(c->grid, &c->ens->domain) != MFTL_SUCCESS) local = MFTL_COUPLING_INVALID;
  if (c->grid->n_loc != c->n_loc) local = MFTL_COUPLING_INVALID;
  /* Two flags ride on the status agreement: did ANY rank change its layout/constraints,
   * or what it exports? One MAX over three ints carries all of it. */
  int v[3] = {local, (c->grid->version_owned != c->seen_owned) ? 1 : 0, (c->grid->version_export != c->seen_export) ? 1 : 0};
  MFTLCplAgreeMaxN(c->comm, v, 3);
  const int st = v[0];
  if (st != MFTL_COUPLING_OK) return st;
  compute_locinfo(c);
  if (agree(c, validate_constraints(c)) != MFTL_COUPLING_OK) return MFTL_COUPLING_INVALID;
  /* Layout or constraints changed somewhere: every window is stale, but the directory is
   * rebuilt only if some rank's ENTRIES changed -- re-registering the same
   * blocks or the same extents keeps it (one MAX of one int). */
  bool rebuild = false;
  if (v[1])
  {
    cpl_dir_entry *own = MFTL_NULL;
    const int n_own = MFTLCplDirectoryOwnEntries(c->grid, &c->ens->domain, c->params.pilot_factor, &own);
    for (int i = 0; i < n_own; ++i) own[i].provider = c->rank;
    const uint64_t h = cpl_fnv(cpl_fnv(CPL_FNV_INIT, &n_own, sizeof n_own), own, (size_t) n_own * sizeof(cpl_dir_entry));
    MFTLCplFree(own);
    rebuild = MFTLCplAgreeMax(c->comm, (h != c->own_hash) ? 1 : 0) != 0;
  }
  if (rebuild)
    build_directory(c);
  else
    c->stats.n_directory_kept++;
  c->seen_owned = c->grid->version_owned;
  c->seen_export = c->grid->version_export;
  if (v[1])
    c->grid_epoch++;
  else if (v[2])
    c->export_epoch++;
  return MFTL_COUPLING_OK;
}

/**
 * @brief The finest resolution the directory knows within the closed box
 *        `[lo, hi]` (periodic images included), over the locations @p locs (all
 *        when @p n_locs is 0); #MFTL_DBL_MAX when it knows none.
 */
static double local_hmin(const MFTL_Coupler *c, const double lo[3], const double hi[3], int n_locs, const int *locs, int **got, int *gcap)
{
  const MFTL_Domain *d = &c->ens->domain;
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
        const int nl = (n_locs > 0) ? n_locs : c->n_loc;
        for (int li = 0; li < nl; ++li)
        {
          const int l = (n_locs > 0) ? locs[li] : li;
          const int ng = MFTLCplDirIndexQuery(&c->dix[l], c->dir, qlo, qhi, got, gcap);
          for (int gi = 0; gi < ng; ++gi)
          {
            const cpl_dir_entry *E = &c->dir[(*got)[gi]];
            h = fmin(h, fmin(E->hmin[0], fmin(E->hmin[1], E->hmin[2])));
          }
        }
      }
  return h;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Markers, with the quadrature levels the directory decides
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Points per edge are capped here: m^2 points per triangle. */
#define CPL_QUAD_LEVEL_LIMIT 256

static bool quad_spec_valid(const MFTL_MarkerSpec *s)
{
  return s->quadrature_level >= 0 && s->quadrature_level <= CPL_QUAD_LEVEL_LIMIT && s->quadrature_max_level >= 0 &&
         s->quadrature_max_level <= CPL_QUAD_LEVEL_LIMIT && s->quadrature_ratio >= 0.0 && isfinite(s->quadrature_ratio);
}

/**
 * @brief Points per edge of every owned triangle (bodies in list order, then
 *        pool order): the fixed level, or the smallest m whose sub-triangles'
 *        longest edge is within `ratio` of the finest fluid resolution the
 *        directory knows around the triangle. @p n_capped counts the triangles
 *        that needed more than the cap.
 */
static int *quad_levels(const MFTL_Coupler *c, const MFTL_MarkerSpec *spec, const int *locs, int n_locs, int64_t *n_capped)
{
  const MFTL_Ensemble *ens = c->ens;
  int64_t nt = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) nt += ens->local_objects[o]->tris.n_local;
  int *lv = (int *) MFTLCplAlloc((size_t) MFTL_MAX(nt, 1) * sizeof(int), "quadrature levels");
  const double ratio = (spec->quadrature_ratio > 0.0) ? spec->quadrature_ratio : c->params.sampling_ratio;
  const int cap = (spec->quadrature_max_level > 0) ? spec->quadrature_max_level : 16;
  int *got = MFTL_NULL, gcap = 0;
  int64_t at = 0;
  *n_capped = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    for (int t = 0; t < obj->tris.n_local; ++t, ++at)
    {
      if (spec->quadrature_level > 0)
      {
        lv[at] = spec->quadrature_level;
        continue;
      }
      double cr[3][3], lo[3], hi[3], lmax = 0.0;
      MFTLGetTriangleCorners(obj, t, cr);
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
      const double h = local_hmin(c, lo, hi, n_locs, locs, &got, &gcap);
      double need = 1.0;
      if (h < MFTL_DBL_MAX && h > 0.0) need = ceil(lmax / (ratio * h));
      if (!(need >= 1.0)) need = 1.0;
      if (need > (double) cap)
      {
        ++*n_capped;
        need = (double) cap;
      }
      lv[at] = (int) need;
    }
  }
  MFTLCplFree(got);
  return lv;
}

/** @brief Gathers a spec's markers; quadrature levels from the directory. @p n_capped may be NULL. */
static int gather_markers(const MFTL_Coupler *c, const MFTL_MarkerSpec *spec, const int *locs, int n_locs, cpl_markers *m, int64_t *n_capped)
{
  if (n_capped) *n_capped = 0;
  if (spec->kind != MFTL_MARKERS_QUADRATURE) return MFTLCplMarkersGather(c->ens, spec, MFTL_NULL, m);
  memset(m, 0, sizeof *m);
  if (!quad_spec_valid(spec)) return MFTL_COUPLING_INVALID;
  int64_t capped = 0;
  int *lv = quad_levels(c, spec, locs, n_locs, &capped);
  const int st = MFTLCplMarkersGather(c->ens, spec, lv, m);
  MFTLCplFree(lv);
  if (n_capped) *n_capped = capped;
  return st;
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
  double cell[9];  ///< anisotropic point location only
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
    pk->key = (uint64_t *) MFTLCplRealloc(pk->key, (size_t) pk->cap * sizeof(uint64_t), "pack");
    pk->piece = (int *) MFTLCplRealloc(pk->piece, (size_t) pk->cap * sizeof(int), "pack");
    pk->elem = (int64_t *) MFTLCplRealloc(pk->elem, (size_t) pk->cap * sizeof(int64_t), "pack");
    pk->sample = (int64_t *) MFTLCplRealloc(pk->sample, (size_t) pk->cap * sizeof(int64_t), "pack");
    pk->meas = (double *) MFTLCplRealloc(pk->meas, (size_t) pk->cap * sizeof(double), "pack");
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
  MFTLCplFree(pk->key);
  MFTLCplFree(pk->piece);
  MFTLCplFree(pk->elem);
  MFTLCplFree(pk->sample);
  MFTLCplFree(pk->meas);
}

/** @brief Box subtraction that never runs out of room (a dropped piece would hide a gap). */
static void box_sub_dyn(double **rem, int *n, int *cap, const double *sub_box, const double sliver[3])
{
  if (*cap < 6 * (*n) + 6)
  {
    *cap = 6 * (*n) + 6;
    *rem = (double *) MFTLCplRealloc(*rem, 6 * (size_t) *cap * sizeof(double), "box subtraction");
  }
  MFTLCplBoxSubtract(*rem, n, *cap, sub_box, sliver);
}

static void ibox_sub_dyn(int64_t **rem, int *n, int *cap, const int64_t *sub_box)
{
  if (*cap < 6 * (*n) + 6)
  {
    *cap = 6 * (*n) + 6;
    *rem = (int64_t *) MFTLCplRealloc(*rem, 6 * (size_t) *cap * sizeof(int64_t), "box subtraction");
  }
  MFTLCplIboxSubtract(*rem, n, *cap, sub_box);
}

struct MFTL_SupportRequest
{
  MFTL_Coupler *c;
  uint64_t grid_epoch;
  uint64_t export_epoch;
  int status;
  int n_loc;
  int *locs;
  double skin[3], bin[3];
  int64_t n_bins;
  uint64_t *bin_key;
  int64_t *bin_idx;
  double *reach;  ///< [n_loc * n_bins][3] certified local support radius
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
  int64_t *n_inactive;  ///< [n_loc] masked samples the providers counted in this rank's request boxes
  long n_served, n_sent;
};

void MFTLCouplerReleaseRequest(MFTL_SupportRequest **pr)
{
  if (pr == MFTL_NULL || *pr == MFTL_NULL) return;
  MFTL_SupportRequest *r = *pr;
  MFTLCplFree(r->locs);
  MFTLCplFree(r->bin_key);
  MFTLCplFree(r->bin_idx);
  MFTLCplFree(r->qbox);
  MFTLCplFree(r->reach);
  MFTLCplFree(r->q);
  MFTLCplFree(r->rec);
  for (int i = 0; i < r->n_pack; ++i) pack_free(&r->pack[i]);
  MFTLCplFree(r->pack);
  MFTLCplFree(r->rbox);
  MFTLCplFree(r->rbox_wl);
  MFTLCplFree(r->missing);
  MFTLCplFree(r->unprovided);
  MFTLCplFree(r->n_inactive);
  MFTLCplFree(r);
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
    *arr = (MFTL_RequestRegion *) MFTLCplRealloc(*arr, (size_t) *cap * sizeof(MFTL_RequestRegion), "request regions");
  }
  (*arr)[(*n)++] = *x;
}

/** @brief Resolves a window location list; returns false on an invalid one. */
static bool resolve_locations(const MFTL_Coupler *c, const MFTL_MarkerSpec *m, int *n_out, int **out)
{
  const int n = (m->n_locations > 0) ? m->n_locations : c->n_loc;
  int *l = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n, 1) * sizeof(int), "window locations");
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
 * @brief Bin size: half the largest minimum reach over the requested locations
 *        (never zero). Only occupied bins are stored. Global minimum widths set
 *        search granularity, not the numerical metric; each bin gets its own
 *        certified reach over the bin plus skin before samples are requested.
 */
static void window_bin(const MFTL_Coupler *c, int n, const int *locs, double b[3])
{
  const MFTL_Domain *d = &c->ens->domain;
  for (int k = 0; k < 3; ++k)
  {
    b[k] = 0.0;
    for (int i = 0; i < n; ++i) b[k] = fmax(b[k], c->kernel.half_width * c->params.support_scale * 0.5 * c->li[locs[i]].hmin[k]);
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
    R->rbox = (double *) MFTLCplRealloc(R->rbox, 6 * (size_t) R->cap_rbox * sizeof(double), "request boxes");
    R->rbox_wl = (int *) MFTLCplRealloc(R->rbox_wl, (size_t) R->cap_rbox * sizeof(int), "request boxes");
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
  int64_t *hbox = (int64_t *) MFTLCplAlloc(6 * (size_t) MFTL_MAX(n_halo, 1) * sizeof(int64_t), "halo boxes");
  int *hloc = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n_halo, 1) * sizeof(int), "halo boxes");
  for (int i = 0; i < n_halo; ++i)
  {
    int32_t l = 0;
    GET(msg, bytes, &off, l);
    hloc[i] = l;
    for (int k = 0; k < 6; ++k) GET(msg, bytes, &off, hbox[6 * i + k]);
  }

  /* marks, per window location and piece, allocated when first touched */
  marks_t **marks = (marks_t **) MFTLCplCalloc((size_t) R->n_loc, sizeof(marks_t *), "serve marks");
  for (int wl = 0; wl < R->n_loc; ++wl)
    marks[wl] = (marks_t *) MFTLCplCalloc((size_t) MFTL_MAX(g->loc[R->locs[wl]].n_pieces, 1), sizeof(marks_t), "serve marks");
#define CPL_MARKS(wl, pc, size)                                                                    \
  do                                                                                               \
  {                                                                                                \
    if (marks[wl][pc].m == MFTL_NULL)                                                              \
    {                                                                                              \
      marks[wl][pc].n = (size);                                                                    \
      marks[wl][pc].m = (unsigned char *) MFTLCplCalloc((size_t) MFTL_MAX(size, 1), 1, "marks"); \
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
                int64_t *rem = (int64_t *) MFTLCplAlloc(6 * (size_t) cap_rem * sizeof(int64_t), "missing regions");
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
                MFTLCplFree(rem);
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
        int64_t cursor = 1, leaf;
        while ((leaf = MFTLCplBoxTreeNext(&p->tile_index, box, box + 3, &cursor)) >= 0)
          for (int t = (int) leaf * p->tile_index.group; t < MFTL_MIN((leaf + 1) * p->tile_index.group, p->n_tiles); ++t)
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
              if (mk->tile_full == MFTL_NULL) mk->tile_full = (unsigned char *) MFTLCplCalloc((size_t) p->n_tiles, 1, "tile marks");
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
          double *rem = (double *) MFTLCplAlloc(6 * (size_t) cap_rem * sizeof(double), "missing regions");
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
          MFTLCplFree(rem);
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
      MFTLCplFree(mk->m);
      MFTLCplFree(mk->tile_full);
    }
    MFTLCplFree(marks[wl]);

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
    out.key = (uint64_t *) MFTLCplAlloc(nn * sizeof(uint64_t), "pack");
    out.piece = (int *) MFTLCplAlloc(nn * sizeof(int), "pack");
    out.elem = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "pack");
    out.sample = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "pack");
    out.meas = (double *) MFTLCplAlloc(nn * sizeof(double), "pack");
    if (pk.n > 0)
    {
      int *perm = (int *) MFTLCplAlloc((size_t) pk.n * sizeof(int), "pack sort");
      uint64_t *k2 = (uint64_t *) MFTLCplAlloc((size_t) pk.n * sizeof(uint64_t), "pack sort");
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
      MFTLCplFree(perm);
      MFTLCplFree(k2);
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
        if (L->anisotropic) MFTLCplBufPut(reply, &p->cell[9 * pi], 9 * sizeof(double));
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
      R->pack = (pack_t *) MFTLCplRealloc(R->pack, (size_t) R->cap_pack * sizeof(pack_t), "pack lists");
    }
    R->pack[R->n_pack++] = out;
  }
  MFTLCplFree(marks);
  MFTLCplFree(hbox);
  MFTLCplFree(hloc);
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
    R->n_inactive[q->wl] += n_inactive;
    if (n_missing > 0 && status < MFTL_COUPLING_NEED_COVERAGE) status = MFTL_COUPLING_NEED_COVERAGE;
  }
  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    int64_t nrec = 0;
    if (!GET(msg, bytes, &off, nrec) || nrec < 0) return MFTL_COUPLING_INVALID;
    const bool points = !cpl_is_lattice(&c->grid->loc[R->locs[wl]]);
    const bool aniso = c->grid->loc[R->locs[wl]].anisotropic;
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
        if (aniso)
          for (int a = 0; a < 9; ++a)
            if (!GET(msg, bytes, &off, r->cell[a])) return MFTL_COUPLING_INVALID;
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

/** @brief Decodes a bin key into its three signed bin indices. */
static void bin_unkey3(uint64_t key, int64_t ib[3])
{
  const int64_t off = INT64_C(1) << 20, mask = (INT64_C(1) << 21) - 1;
  ib[0] = (int64_t) ((key >> 42) & (uint64_t) mask) - off;
  ib[1] = (int64_t) ((key >> 21) & (uint64_t) mask) - off;
  ib[2] = (int64_t) (key & (uint64_t) mask) - off;
}

/**
 * @brief The bins other ranks' markers occupy within the migration distance of
 *        this rank's box: each rank sends each of its bins to every rank whose
 *        box meets the bin widened by the distance (arithmetic on the process
 *        grid, periodic images included), by NBX. Collective.
 * @return The received bin keys (allocated; possibly repeated).
 */
static uint64_t *migration_bins(MFTL_Coupler *c, const uint64_t *keys, int64_t nb, const double bin[3], const double mig[3], int64_t *n_out)
{
  const MFTL_Domain *d = &c->ens->domain;
  int n_dest = 0, cap_dest = 0, cap_r = 32;
  int *dest = MFTL_NULL, *ranks = (int *) MFTLCplAlloc((size_t) cap_r * sizeof(int), "migration ranks");
  cpl_buf *bufs = MFTL_NULL;
  for (int64_t i = 0; i < nb; ++i)
  {
    int64_t ib[3];
    bin_unkey3(keys[i], ib);
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k)
    {
      lo[k] = d->global_min[k] + (double) ib[k] * bin[k] - mig[k];
      hi[k] = d->global_min[k] + (double) (ib[k] + 1) * bin[k] + mig[k];
    }
    int nr = MFTLCplRanksTouching(d, lo, hi, ranks, cap_r);
    if (nr > cap_r)
    {
      cap_r = nr;
      ranks = (int *) MFTLCplRealloc(ranks, (size_t) cap_r * sizeof(int), "migration ranks");
      nr = MFTLCplRanksTouching(d, lo, hi, ranks, cap_r);
    }
    for (int j = 0; j < nr; ++j)
    {
      if (ranks[j] == c->rank) continue;
      int at = -1;
      for (int q = 0; q < n_dest; ++q)
        if (dest[q] == ranks[j]) at = q;
      if (at < 0)
      {
        if (n_dest == cap_dest)
        {
          cap_dest = MFTLGrowCapacity(cap_dest, n_dest + 1);
          dest = (int *) MFTLCplRealloc(dest, (size_t) cap_dest * sizeof(int), "migration dest");
          bufs = (cpl_buf *) MFTLCplRealloc(bufs, (size_t) cap_dest * sizeof(cpl_buf), "migration dest");
        }
        at = n_dest++;
        dest[at] = ranks[j];
        memset(&bufs[at], 0, sizeof(cpl_buf));
      }
      MFTLCplBufPut(&bufs[at], &keys[i], sizeof(uint64_t));
    }
  }
  MFTLCplFree(ranks);
  unsigned char **sb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "migration send");
  size_t *sz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "migration send");
  for (int q = 0; q < n_dest; ++q)
  {
    sb[q] = bufs[q].data;
    sz[q] = bufs[q].n;
  }
  cpl_msg *in = MFTL_NULL;
  int n_in = 0;
  MFTLCplNbx(c->comm, CPL_TAG_MIGRATION, n_dest, dest, sb, sz, &in, &n_in, &c->stats.bytes_discovery, &c->stats.messages);
  int64_t total = 0;
  for (int r = 0; r < n_in; ++r) total += (int64_t) (in[r].bytes / sizeof(uint64_t));
  uint64_t *out = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(uint64_t), "migration bins");
  int64_t at = 0;
  for (int r = 0; r < n_in; ++r)
  {
    const int64_t n = (int64_t) (in[r].bytes / sizeof(uint64_t));
    memcpy(out + at, in[r].data, (size_t) n * sizeof(uint64_t));
    at += n;
  }
  MFTLCplMsgsFree(in, n_in);
  MFTLCplFree(in);
  for (int q = 0; q < n_dest; ++q) MFTLCplFree(bufs[q].data);
  MFTLCplFree(bufs);
  MFTLCplFree(dest);
  MFTLCplFree(sb);
  MFTLCplFree(sz);
  *n_out = total;
  return out;
}

int MFTLCouplerRequestSupport(MFTL_Coupler *c, const MFTL_MarkerSpec *spec, const MFTL_WindowSpec *wspec, MFTL_SupportRequest **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const MFTL_Domain *d = &c->ens->domain;
  const MFTL_SolverGrid *g = c->grid;

  MFTL_SupportRequest *R = (MFTL_SupportRequest *) MFTLCplCalloc(1, sizeof(MFTL_SupportRequest), "MFTLCouplerRequestSupport");
  R->c = c;
  R->grid_epoch = c->grid_epoch;
  R->export_epoch = c->export_epoch;

  int local = MFTL_COUPLING_OK;
  if (out == MFTL_NULL || spec == MFTL_NULL || wspec == MFTL_NULL || c->active) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && !resolve_locations(c, spec, &R->n_loc, &R->locs)) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK && (g->version_owned != c->seen_owned || g->version_export != c->seen_export))
  {
    fprintf(stderr, "MFTL coupling: the grid changed since the last MFTLCouplerUpdateGrid()\n");
    local = MFTL_COUPLING_STALE;
  }
  double migration[3] = {0.0, 0.0, 0.0};
  for (int k = 0; k < 3 && local == MFTL_COUPLING_OK; ++k)
  {
    R->skin[k] = wspec->skin[k];
    migration[k] = wspec->migration[k];
    if (!(R->skin[k] >= 0.0) || !isfinite(R->skin[k]) || !(migration[k] >= 0.0) || !isfinite(migration[k])) local = MFTL_COUPLING_INVALID;
  }
  cpl_markers m;
  memset(&m, 0, sizeof m);
  if (local == MFTL_COUPLING_OK) local = gather_markers(c, spec, R->locs, R->n_loc, &m, MFTL_NULL);

  /* Every rank must ask for the same locations with the same skin. */
  uint64_t sig = CPL_FNV_INIT;
  if (local == MFTL_COUPLING_OK)
  {
    sig = cpl_fnv(sig, &R->n_loc, sizeof R->n_loc);
    sig = cpl_fnv(sig, R->locs, (size_t) R->n_loc * sizeof(int));
    sig = cpl_fnv(sig, R->skin, sizeof R->skin);
    sig = cpl_fnv(sig, migration, sizeof migration);
  }
  int st = agree(c, local);
  if (st == MFTL_COUPLING_OK && !MFTLCplAgreeHash(c->comm, sig)) st = MFTL_COUPLING_INVALID;
  if (st == MFTL_COUPLING_OK)
  {
    /* Reject impossible periodic skins even for an empty marker batch, using
     * a lower bound rather than letting a remote coarse sample reject fine rows. */
    for (int i = 0; i < R->n_loc; ++i)
      for (int k = 0; k < 3; ++k)
      {
        const cpl_locinfo *li = &c->li[R->locs[i]];
        const double growth = c->params.failure & MFTL_STENCIL_GROW ? c->params.max_growth : 1.0;
        const double rmin = c->kernel.half_width * c->params.support_scale * growth * li->hmin[k];
        if (d->is_periodic[k] && !(rmin + R->skin[k] < 0.5 * d->domain_size[k])) st = MFTL_COUPLING_INVALID;
      }
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

  R->n_inactive = (int64_t *) MFTLCplCalloc((size_t) R->n_loc, sizeof(int64_t), "request masks");

  /* ── marker bins ── */
  window_bin(c, R->n_loc, R->locs, R->bin);
  {
    uint64_t *keys = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(m.n, 1) * sizeof(uint64_t), "marker bins");
    uint64_t *tmp = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(m.n, 1) * sizeof(uint64_t), "marker bins");
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
    if (migration[0] > 0.0 || migration[1] > 0.0 || migration[2] > 0.0)
    {
      /* send every bin to the ranks whose box lies within the migration
       * distance of it; take theirs in (one NBX, collective: the allowance is
       * part of the agreed signature) */
      int64_t n_recv = 0;
      uint64_t *recv = migration_bins(c, keys, nb, R->bin, migration, &n_recv);
      if (n_recv > 0)
      {
        keys = (uint64_t *) MFTLCplRealloc(keys, (size_t) (nb + n_recv) * sizeof(uint64_t), "marker bins");
        tmp = (uint64_t *) MFTLCplRealloc(tmp, (size_t) (nb + n_recv) * sizeof(uint64_t), "marker bins");
        memcpy(keys + nb, recv, (size_t) n_recv * sizeof(uint64_t));
        MFTLRadixSort64(keys, tmp, (int) (nb + n_recv));
        int64_t u = 0;
        for (int64_t i = 0; i < nb + n_recv; ++i)
          if (i == 0 || keys[i] != keys[i - 1]) keys[u++] = keys[i];
        nb = u;
      }
      MFTLCplFree(recv);
    }
    R->n_bins = nb;
    R->bin_key = keys;
    R->bin_idx = (int64_t *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(nb, 1) * sizeof(int64_t), "marker bins");
    const int64_t off = INT64_C(1) << 20, mask = (INT64_C(1) << 21) - 1;
    for (int64_t i = 0; i < nb; ++i)
    {
      R->bin_idx[3 * i + 0] = (int64_t) ((keys[i] >> 42) & (uint64_t) mask) - off;
      R->bin_idx[3 * i + 1] = (int64_t) ((keys[i] >> 21) & (uint64_t) mask) - off;
      R->bin_idx[3 * i + 2] = (int64_t) (keys[i] & (uint64_t) mask) - off;
    }
    MFTLCplFree(tmp);
  }
  MFTLCplMarkersFree(&m);
  /* Resolution certificates over every movement envelope, then ownership
   * certificates over its locally bounded numerical/metric support. */
  MFTLCplFree(c->dir);
  c->n_dir = c->n_shard;
  c->dir = MFTLCplAlloc((size_t) MFTL_MAX(c->n_dir, 1) * sizeof(cpl_dir_entry), "active directory");
  memcpy(c->dir, c->shard, (size_t) c->n_dir * sizeof(cpl_dir_entry));
  index_directory(c);
  const int64_t nq = R->n_bins * R->n_loc;
  directory_lookup *look = MFTLCplCalloc((size_t) MFTL_MAX(nq, 1), sizeof(directory_lookup), "resolution queries");
  R->reach = MFTLCplCalloc(3 * (size_t) MFTL_MAX(nq, 1), sizeof(double), "local support bounds");
  bool variable_metric = false;
  for (int wl = 0; wl < R->n_loc; ++wl) variable_metric |= !c->li[R->locs[wl]].uniform_metric;
  for (int wl = 0; wl < R->n_loc; ++wl)
    for (int64_t bi = 0; bi < R->n_bins; ++bi)
    {
      directory_lookup *q = &look[wl * R->n_bins + bi];
      q->location = R->locs[wl];
      q->metric = 1;
      for (int k = 0; k < 3; ++k)
      {
        q->box[k] = d->global_min[k] + (double) R->bin_idx[3 * bi + k] * R->bin[k] - R->skin[k];
        q->box[3 + k] = d->global_min[k] + (double) (R->bin_idx[3 * bi + k] + 1) * R->bin[k] + R->skin[k];
      }
    }
  /* Locations and their uniform-metric flags are agreed across ranks. Uniform
   * grids already have exact bounds, so omit both metric-lookup exchanges. */
  if (variable_metric) lookup_directory(c, look, nq);
  const double growth = c->params.failure & MFTL_STENCIL_GROW ? c->params.max_growth : 1.0;
  const double factor = fmax(c->params.pilot_factor, c->kernel.half_width * c->params.support_scale * growth);
  int *metric_entries = MFTL_NULL, metric_capacity = 0;
  R->qbox = MFTLCplAlloc(6 * (size_t) R->n_loc * sizeof(double), "request boxes");
  for (int wl = 0; wl < R->n_loc; ++wl)
  {
    const int loc = R->locs[wl];
    const cpl_locinfo *li = &c->li[loc];
    for (int k = 0; k < 3; ++k) R->qbox[6 * wl + k] = MFTL_DBL_MAX, R->qbox[6 * wl + 3 + k] = -MFTL_DBL_MAX;
    for (int64_t bi = 0; bi < R->n_bins; ++bi)
    {
      directory_lookup *q = &look[wl * R->n_bins + bi];
      double bound[3] = {0, 0, 0};
      if (li->uniform_metric)
        memcpy(bound, li->hmax, sizeof bound);
      else
        for (int z = -1; z <= 1; ++z)
          for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x)
            {
              const int shift[3] = {x, y, z};
              double lo[3], hi[3];
              bool valid = true;
              for (int k = 0; k < 3; ++k)
              {
                if (shift[k] && !d->is_periodic[k]) valid = false;
                lo[k] = q->box[k] - shift[k] * d->domain_size[k];
                hi[k] = q->box[3 + k] - shift[k] * d->domain_size[k];
              }
              if (!valid) continue;
              const int ng = MFTLCplDirMetricQuery(&c->dix[loc], c->dir, lo, hi, &metric_entries, &metric_capacity);
              for (int j = 0; j < ng; ++j)
                for (int k = 0; k < 3; ++k) bound[k] = fmax(bound[k], c->dir[metric_entries[j]].hmax[k]);
            }
      q->metric = 0;
      for (int k = 0; k < 3; ++k)
      {
        /* No metric certificate: still request a finite region so an ownership
         * hole is reported as coverage, rather than mistaken for a wall. */
        if (!(bound[k] > 0.0)) bound[k] = li->hmin[k] < MFTL_DBL_MAX ? li->hmin[k] : R->bin[k];
        const double reach = li->uniform_metric ? c->kernel.half_width * c->params.support_scale * growth * bound[k] : factor * bound[k];
        R->reach[3 * (wl * R->n_bins + bi) + k] = reach;
        if (d->is_periodic[k] && !(reach + R->skin[k] < 0.5 * d->domain_size[k])) st = MFTL_COUPLING_INVALID;
        q->box[k] -= reach;
        q->box[3 + k] += reach;
        R->qbox[6 * wl + k] = fmin(R->qbox[6 * wl + k], q->box[k]);
        R->qbox[6 * wl + 3 + k] = fmax(R->qbox[6 * wl + 3 + k], q->box[3 + k]);
      }
    }
  }
  MFTLCplFree(metric_entries);
  st = agree(c, st);
  if (st != MFTL_COUPLING_OK)
  {
    MFTLCplFree(look);
    MFTLCouplerReleaseRequest(&R);
    c->stats.n_requests++;
    return st;
  }
  lookup_directory(c, look, nq);
  MFTLCplFree(look);

  /* ── queries: one per (bin, location) for a lattice, per (bin, location, provider, image) for points ── */
  int cap_asked = 32;
  int *asked = (int *) MFTLCplAlloc((size_t) cap_asked * sizeof(int), "asked providers");
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
        dest = (int *) MFTLCplRealloc(dest, (size_t) cap_dest * sizeof(int), "request dest");                            \
        dsub = (int64_t **) MFTLCplRealloc(dsub, (size_t) cap_dest * sizeof(int64_t *), "request dest");                 \
        dsub_n = (int64_t *) MFTLCplRealloc(dsub_n, (size_t) cap_dest * sizeof(int64_t), "request dest");                \
        dsub_cap = (int64_t *) MFTLCplRealloc(dsub_cap, (size_t) cap_dest * sizeof(int64_t), "request dest");            \
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
    for (int64_t bi = 0; bi < R->n_bins; ++bi)
    {
      double Q[6];
      for (int k = 0; k < 3; ++k)
      {
        Q[k] = d->global_min[k] + (double) R->bin_idx[3 * bi + k] * R->bin[k] - R->reach[3 * (wl * R->n_bins + bi) + k] - R->skin[k];
        Q[3 + k] = d->global_min[k] + (double) (R->bin_idx[3 * bi + k] + 1) * R->bin[k] + R->reach[3 * (wl * R->n_bins + bi) + k] + R->skin[k];
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
                cand = (int *) MFTLCplRealloc(cand, (size_t) cand_cap * sizeof(int), "request candidates");
              }
              for (int i = 0; i < ng; ++i) cand[n_cand++] = got[i];
              MFTLCplFree(got);
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
          GROW(asked, cap_asked, n_asked + 1, int);
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
              MFTLCplFree(got);
            }
        /* and is every part of Q owned by somebody? The cover answers, in the
         * image of Q that lies in this rank's frame. */
        int n_rem = 1, cap_rem = 16;
        double *rem = MFTLCplAlloc(6 * (size_t) cap_rem * sizeof(double), "request cover");
        memcpy(rem, Q, sizeof Q);
        for (int z = -1; z <= 1 && n_rem; ++z)
          for (int y = -1; y <= 1 && n_rem; ++y)
            for (int x = -1; x <= 1 && n_rem; ++x)
            {
              const int shift[3] = {x, y, z};
              double lo[3], hi[3];
              bool valid = true;
              for (int k = 0; k < 3; ++k)
              {
                if (shift[k] && !d->is_periodic[k]) valid = false;
                lo[k] = Q[k] - shift[k] * d->domain_size[k];
                hi[k] = Q[3 + k] - shift[k] * d->domain_size[k];
              }
              if (!valid) continue;
              int *got = MFTL_NULL, gc = 0;
              const int ng = MFTLCplDirIndexQuery(&c->dix[loc], c->dir, lo, hi, &got, &gc);
              for (int j = 0; j < ng && n_rem; ++j)
              {
                double box[6];
                for (int k = 0; k < 3; ++k)
                {
                  box[k] = c->dir[got[j]].lo[k] + shift[k] * d->domain_size[k];
                  box[3 + k] = c->dir[got[j]].hi[k] + shift[k] * d->domain_size[k];
                }
                box_sub_dyn(&rem, &n_rem, &cap_rem, box, sliver);
              }
              MFTLCplFree(got);
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
        MFTLCplFree(rem);
      }
    }
  }
#undef ADD_DEST_QUERY
  MFTLCplFree(asked);
  MFTLCplFree(cand);

  /* ── messages: my ghost boxes, then the queries ── */
  dbuf = (cpl_buf *) MFTLCplCalloc((size_t) MFTL_MAX(n_dest, 1), sizeof(cpl_buf), "request buffers");
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
  int *odest = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(int), "request send");
  unsigned char **osb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(unsigned char *), "request send");
  size_t *osz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_dest, 1) * sizeof(size_t), "request send");
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
  cpl_buf *replies = (cpl_buf *) MFTLCplCalloc((size_t) MFTL_MAX(n_in, 1), sizeof(cpl_buf), "replies");
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
  int *rdest = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n_in, 1) * sizeof(int), "reply dest");
  unsigned char **rsb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(n_in, 1) * sizeof(unsigned char *), "reply send");
  size_t *rsz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_in, 1) * sizeof(size_t), "reply send");
  for (int i = 0; i < n_in; ++i)
  {
    rdest[i] = in[i].peer;
    rsb[i] = replies[i].data;
    rsz[i] = replies[i].n;
  }
  cpl_msg *rin = (cpl_msg *) MFTLCplCalloc((size_t) MFTL_MAX(n_out, 1), sizeof(cpl_msg), "reply receive");
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
    MFTLCplFree(dbuf[i].data);
    MFTLCplFree(dsub[i]);
  }
  MFTLCplFree(dbuf);
  MFTLCplFree(dsub);
  MFTLCplFree(dsub_n);
  MFTLCplFree(dsub_cap);
  MFTLCplFree(dest);
  for (int i = 0; i < n_in; ++i) MFTLCplFree(replies[i].data);
  MFTLCplFree(replies);
  MFTLCplFree(self_reply.data);
  MFTLCplMsgsFree(in, n_in);
  MFTLCplFree(in);
  MFTLCplMsgsFree(rin, n_out);
  MFTLCplFree(rin);
  MFTLCplFree(odest);
  MFTLCplFree(osb);
  MFTLCplFree(osz);
  MFTLCplFree(rdest);
  MFTLCplFree(rsb);
  MFTLCplFree(rsz);

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

/** @brief Free a snapshot's storage; its window reference has already been released. */
static void fetch_storage_free(MFTL_CouplingFetch *f)
{
  for (int i = 0; i < f->n_fields; ++i) MFTLCplFree(f->values[i]);
  MFTLCplFree(f->values);
  MFTLCplFree(f->field_band);
  MFTLCplFree(f->version);
  MFTLCplFree(f->wall);
  MFTLCplFree(f);
}

static void free_band(cpl_band *b)
{
  MFTLCplBricksFree(b);
  MFTLCplFree(b->key);
  MFTLCplFree(b->owner);
  MFTLCplFree(b->route);
  MFTLCplFree(b->piece);
  MFTLCplFree(b->elem);
  MFTLCplFree(b->measure);
  MFTLCplFree(b->h);
  MFTLCplFree(b->cell);
  MFTLCplFree(b->ipos);
  MFTLCplFree(b->image_index.box);
  MFTLCplFree(b->idof);
  MFTLCplFree(b->ibin_key);
  MFTLCplFree(b->ibin_off);
  MFTLCplFree(b->src_rank);
  MFTLCplFree(b->src_off);
  MFTLCplFree(b->src_nonhalo);
  MFTLCplFree(b->src_halo);
  MFTLCplFree(b->dst_rank);
  MFTLCplFree(b->dst_peer);
  MFTLCplFree(b->dst_off);
  MFTLCplFree(b->dst_nonhalo);
  MFTLCplFree(b->dst_halo);
  MFTLCplFree(b->pack_served);
  MFTLCplFree(b->served_piece);
  MFTLCplFree(b->served_elem);
  MFTLCplFree(b->served_sample);
  MFTLCplFree(b->served_measure);
  MFTLCplFree(b->served_con);
  MFTLCplFree(b->reqbox);
  memset(b, 0, sizeof *b);
}

int MFTLCouplerReleaseWindow(MFTL_CouplingWindow **pw)
{
  if (pw == MFTL_NULL || *pw == MFTL_NULL) return MFTL_SUCCESS;
  MFTL_CouplingWindow *w = *pw;
  if (w->refcount > 0) return MFTL_ERROR;
  while (w->fetch_pool)
  {
    MFTL_CouplingFetch *f = w->fetch_pool;
    w->fetch_pool = f->next;
    fetch_storage_free(f);
  }
  for (int i = 0; i < w->n_loc; ++i) free_band(&w->band[i]);
  MFTLCplFree(w->band);
  MFTLCplFree(w->locs);
  MFTLCplFree(w->bin_key);
  MFTLCplFree(w->reach);
  if (w->candidates)
    for (int64_t i = 0; i < w->n_loc * w->n_bins; ++i) MFTLCplFree(w->candidates[i].img);
  MFTLCplFree(w->candidates);
  MFTLCplFree(w->fetch_peer);
  MFTLCplFree(w->serve_peer);
  MFTLCplFree(w->fetch_segment);
  MFTLCplFree(w->serve_segment);
  if (w->c)
  {
    w->c->live--;
    w->c->stats.bytes_memory -= (long) w->bytes;
  }
  MFTLCplFree(w);
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
  uint64_t spatial;
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
  if (x->spatial != y->spatial) return (x->spatial > y->spatial) - (x->spatial < y->spatial);
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
  B->has_masked = R->n_inactive[wl] > 0;

  /* ── DOFs: provider order, then non-ghost keys, then ghost keys ── */
  if (n_recs > 1) qsort(recs, (size_t) n_recs, sizeof(rec_t), cmp_rec_provider_key);
  int64_t n_unique = 0;
  for (int64_t i = 0; i < n_recs; ++i)
    if (i == 0 || recs[i].key != recs[i - 1].key || recs[i].provider != recs[i - 1].provider) ++n_unique;

  const size_t nu = (size_t) MFTL_MAX(n_unique, 1);
  B->key = (uint64_t *) MFTLCplAlloc(nu * sizeof(uint64_t), "band");
  B->owner = (int *) MFTLCplAlloc(nu * sizeof(int), "band");
  B->route = (unsigned char *) MFTLCplAlloc(nu, "band");
  B->piece = (int *) MFTLCplAlloc(nu * sizeof(int), "band");
  B->elem = (int64_t *) MFTLCplAlloc(nu * sizeof(int64_t), "band");
  B->measure = (double *) MFTLCplAlloc(nu * sizeof(double), "band");
  B->h = (double *) MFTLCplAlloc(3 * nu * sizeof(double), "band");
  if (L->anisotropic) B->cell = (double *) MFTLCplAlloc(9 * nu * sizeof(double), "band");
  int64_t *dof_of_rec = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_recs, 1) * sizeof(int64_t), "band");

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
          if (B->cell) memcpy(&B->cell[9 * n], recs[r].cell, 9 * sizeof(double));
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
      B->src_rank = (int *) MFTLCplRealloc(B->src_rank, (size_t) cap_src * sizeof(int), "band sources");
      B->src_off = (int64_t *) MFTLCplRealloc(B->src_off, (size_t) cap_src * sizeof(int64_t), "band sources");
      B->src_nonhalo = (int64_t *) MFTLCplRealloc(B->src_nonhalo, (size_t) cap_src * sizeof(int64_t), "band sources");
      B->src_halo = (int64_t *) MFTLCplRealloc(B->src_halo, (size_t) cap_src * sizeof(int64_t), "band sources");
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
        MFTLCplFree(dof_of_rec);
        return MFTL_COUPLING_INVALID;
      }
      for (int64_t x = 0; x < self_pack->n; ++x)
      {
        if (self_pack->key[x] != B->key[seg_off + x])
        {
          MFTLCplFree(dof_of_rec);
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
  if (L->kind == MFTL_LATTICE_UNIFORM && L->boundary != MFTL_BOUNDARY_MIRROR && c->params.kernel == MFTL_KERNEL_PESKIN4 &&
      c->params.support_scale == 1.0 && c->params.reproduction == MFTL_REPRODUCE_LINEAR && B->n_dof <= INT32_MAX)
    MFTLCplBricksBuild(B, L);

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
  MFTLCplFree(dof_of_rec);
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
    B->bsize[k] = fmax(c->kernel.half_width * c->params.support_scale * li->hmin[k], floor_b);
  }
  for (int64_t i = 0; i < n_img; ++i)
  {
    int64_t ib[3];
    for (int k = 0; k < 3; ++k) ib[k] = (int64_t) floor((im[i].pos[k] - d->global_min[k]) / B->bsize[k]);
    im[i].bin = bin_key3(ib[0], ib[1], ib[2]);
    im[i].spatial = 0;
    if (!li->uniform_metric || !li->lattice)
    {
      uint64_t quant[3];
      for (int k = 0; k < 3; ++k)
      {
        const double u = (im[i].pos[k] - d->global_min[k]) / B->bsize[k] - (double) ib[k];
        quant[k] = (uint64_t) (fmax(0.0, fmin(u, 1.0)) * MFTL_MORTON_MAX);
      }
      im[i].spatial = MFTLMorton3D(quant[0], quant[1], quant[2]);
    }
  }
  if (n_img > 1) qsort(im, (size_t) n_img, sizeof(img_t), cmp_img_bin);
  B->n_img = n_img;
  B->ipos = (double *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(n_img, 1) * sizeof(double), "band images");
  B->idof = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(n_img, 1) * sizeof(int64_t), "band images");
  int64_t nbins = 0;
  for (int64_t i = 0; i < n_img; ++i)
    if (i == 0 || im[i].bin != im[i - 1].bin) ++nbins;
  B->n_ibins = nbins;
  B->ibin_key = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(nbins, 1) * sizeof(uint64_t), "band bins");
  B->ibin_off = (int64_t *) MFTLCplAlloc((size_t) (nbins + 1) * sizeof(int64_t), "band bins");
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
  MFTLCplFree(im);
  if (!li->uniform_metric || !li->lattice)
    MFTLCplBoxTreeBuild(&B->image_index, B->ipos, B->n_img, 32, true);

  /* ── provider side: pack lists for every requester, and the served table ── */
  int64_t total = 0;
  for (int p = 0; p < R->n_pack; ++p)
    if (R->pack[p].wl == wl) ++B->n_dst, total += R->pack[p].n;
  B->dst_rank = (int *) MFTLCplAlloc((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int), "band dst");
  B->dst_off = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->dst_nonhalo = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->dst_halo = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int64_t), "band dst");
  B->pack_served = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(int64_t), "band pack");
  served_t *sv = (served_t *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(served_t), "band served");
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
  B->served_piece = (int *) MFTLCplAlloc((size_t) MFTL_MAX(ns, 1) * sizeof(int), "served");
  B->served_elem = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(ns, 1) * sizeof(int64_t), "served");
  B->served_sample = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(ns, 1) * sizeof(int64_t), "served");
  B->served_measure = (double *) MFTLCplAlloc((size_t) MFTL_MAX(ns, 1) * sizeof(double), "served");
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
  MFTLCplFree(sv);
  /* sample index and measure of each served DOF, from any pack entry naming it;
   * on a constrained location also whose constraint it is */
  if (li->constrained)
  {
    B->served_con = (int64_t *) MFTLCplAlloc((size_t) MFTL_MAX(ns, 1) * sizeof(int64_t), "served constraints");
    for (int64_t x = 0; x < ns; ++x) B->served_con[x] = -1;
  }
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
      if (B->served_con) B->served_con[s] = MFTLCplConstraintFind(L, pk->key[x]);
    }
    at += pk->n;
  }

  /* the request boxes this rank received for this location (band masks) */
  int64_t nrb = 0;
  for (int64_t i = 0; i < R->n_rbox; ++i)
    if (R->rbox_wl[i] == wl) ++nrb;
  B->n_reqbox = (int) nrb;
  B->reqbox = (double *) MFTLCplAlloc(6 * (size_t) MFTL_MAX(nrb, 1) * sizeof(double), "request boxes");
  nrb = 0;
  for (int64_t i = 0; i < R->n_rbox; ++i)
    if (R->rbox_wl[i] == wl) memcpy(&B->reqbox[6 * nrb++], &R->rbox[6 * i], 6 * sizeof(double));
  return MFTL_COUPLING_OK;
}

static size_t band_bytes(const cpl_band *b)
{
  size_t bytes = (size_t) b->n_dof * (sizeof(uint64_t) + 2 * sizeof(int) + 1 + sizeof(int64_t) + 4 * sizeof(double));
  if (b->cell) bytes += 9 * (size_t) b->n_dof * sizeof(double);
  bytes += (size_t) b->n_img * (3 * sizeof(double) + sizeof(int64_t));
  bytes += (size_t) b->n_ibins * (sizeof(uint64_t) + sizeof(int64_t)) + sizeof(int64_t);
  bytes += (size_t) (b->n_src + b->n_dst) * (sizeof(int) + 3 * sizeof(int64_t));
  bytes += (size_t) b->n_dst * sizeof(int);
  bytes += (size_t) b->n_served * (sizeof(int) + 2 * sizeof(int64_t) + sizeof(double));
  if (b->served_con) bytes += (size_t) b->n_served * sizeof(int64_t);
  for (int i = 0; i < b->n_dst; ++i) bytes += (size_t) (b->dst_nonhalo[i] + b->dst_halo[i]) * sizeof(int64_t);
  bytes += 6 * (size_t) b->n_reqbox * sizeof(double);
  if (b->brick_hash) bytes += (size_t) b->n_bricks * (sizeof(uint64_t) + 64 * sizeof(int64_t)) + (size_t) b->brick_hash_size * sizeof(int64_t);
  if (b->image_index.box) bytes += 12 * (size_t) b->image_index.base * sizeof(double);
  return bytes;
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
  else if (!grid_current(c) || Rc->grid_epoch != c->grid_epoch || Rc->export_epoch != c->export_epoch)
    local = MFTL_COUPLING_STALE;

  MFTL_CouplingWindow *w = MFTL_NULL;
  if (local == MFTL_COUPLING_OK)
  {
    MFTL_SupportRequest *R = (MFTL_SupportRequest *) Rc; /* records are re-sorted in place */
    w = (MFTL_CouplingWindow *) MFTLCplCalloc(1, sizeof(MFTL_CouplingWindow), "window");
    w->c = c;
    w->grid_epoch = c->grid_epoch;
    w->export_epoch = c->export_epoch;
    w->n_loc = R->n_loc;
    w->locs = (int *) MFTLCplAlloc((size_t) R->n_loc * sizeof(int), "window");
    memcpy(w->locs, R->locs, (size_t) R->n_loc * sizeof(int));
    memcpy(w->skin, R->skin, sizeof w->skin);
    memcpy(w->bin, R->bin, sizeof w->bin);
    w->n_bins = R->n_bins;
    w->bin_key = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(R->n_bins, 1) * sizeof(uint64_t), "window bins");
    memcpy(w->bin_key, R->bin_key, (size_t) R->n_bins * sizeof(uint64_t));
    w->reach = MFTLCplAlloc(3 * (size_t) MFTL_MAX((int64_t) R->n_loc * R->n_bins, 1) * sizeof(double), "window support bounds");
    memcpy(w->reach, R->reach, 3 * (size_t) R->n_loc * (size_t) R->n_bins * sizeof(double));
    w->band = (cpl_band *) MFTLCplCalloc((size_t) R->n_loc, sizeof(cpl_band), "window bands");

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
  /* Merge sorted sparse peer lists once; no rank search in a transfer. */
  w->fetch_segment = (int *) MFTLCplAlloc((size_t) MFTL_MAX((size_t) w->n_fetch_peers * w->n_loc, 1) * sizeof(int), "window routes");
  w->serve_segment = (int *) MFTLCplAlloc((size_t) MFTL_MAX((size_t) w->n_serve_peers * w->n_loc, 1) * sizeof(int), "window routes");
  for (int wl = 0; wl < w->n_loc; ++wl)
  {
    cpl_band *B = &w->band[wl];
    B->dst_peer = MFTLCplAlloc((size_t) MFTL_MAX(B->n_dst, 1) * sizeof(int), "window routes");
    for (int d = 0; d < B->n_dst; ++d) B->dst_peer[d] = -1;
    int f = 0, s = 0;
    for (int p = 0; p < w->n_fetch_peers; ++p)
    {
      while (f < B->n_src && B->src_rank[f] < w->fetch_peer[p]) ++f;
      w->fetch_segment[(size_t) p * w->n_loc + wl] = f < B->n_src && B->src_rank[f] == w->fetch_peer[p] ? f : -1;
    }
    for (int p = 0; p < w->n_serve_peers; ++p)
    {
      while (s < B->n_dst && B->dst_rank[s] < w->serve_peer[p]) ++s;
      const int hit = s < B->n_dst && B->dst_rank[s] == w->serve_peer[p] ? s : -1;
      w->serve_segment[(size_t) p * w->n_loc + wl] = hit;
      if (hit >= 0) B->dst_peer[hit] = p;
    }
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
  w->bytes += sizeof *w + (size_t) w->n_loc * (sizeof(int) + sizeof(cpl_band)) + (size_t) w->n_bins * sizeof(uint64_t);
  w->bytes += (size_t) (w->n_fetch_peers + w->n_serve_peers) * ((size_t) w->n_loc + 1) * sizeof(int);
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
    MFTLCplFree(b->tensor[i]);
    MFTLCplFree(b->row_ptr[i]);
    MFTLCplFree(b->col[i]);
    MFTLCplFree(b->w[i]);
    if (b->wallc) MFTLCplFree(b->wallc[i]);
  }
  MFTLCplFree(b->tensor);
  MFTLCplFree(b->row_ptr);
  MFTLCplFree(b->col);
  MFTLCplFree(b->w);
  MFTLCplFree(b->wallc);
  MFTLCplFree(b->loc_band);
  MFTLCplFree(b->ref);
  MFTLCplFree(b->pos);
  MFTLCplFree(b->support_bin);
  MFTLCplFree(b->base_pos);
  MFTLCplFree(b->bodies);
  MFTLCplFree(b->scratch_keys);
  if (b->quad_owned) MFTLCplQuadFree(&b->quad);
  if (b->win) ((MFTL_CouplingWindow *) b->win)->refcount--;
  if (b->base) ((MFTL_CouplingBinding *) b->base)->refcount--;
  if (b->c) b->c->live--;
  MFTLCplFree(b);
  *pb = MFTL_NULL;
  return MFTL_SUCCESS;
}

static void gather_candidates(const cpl_band *B, const MFTL_Domain *d, const double lo[3], const double hi[3], cand_t *cd)
{
  if (B->image_index.box)
  {
    cd->n = 0;
    int64_t cursor = 1, leaf;
    while ((leaf = MFTLCplBoxTreeNext(&B->image_index, lo, hi, &cursor)) >= 0)
      for (int64_t i = leaf * B->image_index.group; i < MFTL_MIN((leaf + 1) * B->image_index.group, B->n_img); ++i)
      {
        const double *x = &B->ipos[3 * i];
        if (x[0] < lo[0] || x[0] > hi[0] || x[1] < lo[1] || x[1] > hi[1] || x[2] < lo[2] || x[2] > hi[2]) continue;
        GROW(cd->img, cd->cap, cd->n + 1, int64_t);
        cd->img[cd->n++] = i;
      }
    return;
  }

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

/** @brief Reuse candidates over the certified motion envelope. Filtering keeps
 * the band's canonical order, so cache hits change neither rows nor sums. */
static void window_candidates(MFTL_Coupler *c, const MFTL_CouplingWindow *wc, int wl, int64_t bin,
                              const double lo[3], const double hi[3], cand_t *cd)
{
  MFTL_CouplingWindow *w = (MFTL_CouplingWindow *) wc;
  const cpl_band *B = &w->band[wl];
  const MFTL_Domain *d = &c->ens->domain;
  if (!w->candidates)
  {
    const size_t bytes = (size_t) w->n_loc * (size_t) w->n_bins * sizeof(cpl_candidates);
    w->candidates = MFTLCplCalloc(1, bytes, "window candidates");
    w->bytes += bytes;
    c->stats.bytes_memory += (long) bytes;
  }
  cpl_candidates *list = &w->candidates[wl * w->n_bins + bin];
  if (!list->built)
  {
    double lower[3], upper[3];
    const uint64_t key = w->bin_key[bin], mask = (UINT64_C(1) << 21) - 1;
    for (int k = 0; k < 3; ++k)
    {
      const int64_t idx = (int64_t) ((key >> (42 - 21 * k)) & mask) - (INT64_C(1) << 20);
      const double r = w->reach[3 * (wl * w->n_bins + bin) + k];
      lower[k] = d->global_min[k] + (double) idx * w->bin[k] - r - w->skin[k];
      upper[k] = d->global_min[k] + (double) (idx + 1) * w->bin[k] + r + w->skin[k];
    }
    gather_candidates(B, d, lower, upper, cd);
    list->n = cd->n;
    const size_t bytes = (size_t) MFTL_MAX(list->n, 1) * sizeof(int64_t);
    list->img = MFTLCplAlloc(bytes, "window candidates");
    memcpy(list->img, cd->img, (size_t) list->n * sizeof(int64_t));
    list->built = true;
    w->bytes += bytes;
    c->stats.bytes_memory += (long) bytes;
    c->stats.n_candidate_builds++;
  }
  else
    c->stats.n_candidate_reuses++;
  cd->n = 0;
  for (int64_t j = 0; j < list->n; ++j)
  {
    const int64_t i = list->img[j];
    const double *x = &B->ipos[3 * i];
    if (x[0] < lo[0] || x[0] > hi[0] || x[1] < lo[1] || x[1] > hi[1] || x[2] < lo[2] || x[2] > hi[2]) continue;
    GROW(cd->img, cd->cap, cd->n + 1, int64_t);
    cd->img[cd->n++] = i;
  }
}

/** @brief Finds the window bin containing @p x within the skin; writes the frame position. */
static bool contain(const MFTL_CouplingWindow *w, const MFTL_Domain *d, const double x[3], double frame[3], double flo[3], int64_t *which)
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
        /* A position in a requested bin is inside, judged by the very
         * expression the request binned with: the reconstructed interval
         * [min + i b, min + (i+1) b] can round to exclude a position an ulp
         * below its upper face that floor() put in bin i, and with no skin
         * that marker would be STALE in the window requested for it, forever. */
        const int64_t direct_bin = find_u64(w->bin_key, w->n_bins, bin_key3(ib[0], ib[1], ib[2]));
        if (direct_bin >= 0)
        {
          *which = direct_bin;
          memcpy(frame, xi, sizeof xi);
          for (int k = 0; k < 3; ++k) flo[k] = d->global_min[k] + (double) ib[k] * w->bin[k];
          return true;
        }
        for (int dz = 0; dz < 3; ++dz)
          for (int dy = 0; dy < 3; ++dy)
            for (int dx = 0; dx < 3; ++dx)
            {
              const int64_t cb[3] = {ib[0] + ((dx == 2) ? -1 : dx), ib[1] + ((dy == 2) ? -1 : dy), ib[2] + ((dz == 2) ? -1 : dz)};
              const int64_t near_bin = find_u64(w->bin_key, w->n_bins, bin_key3(cb[0], cb[1], cb[2]));
              if (near_bin < 0) continue;
              bool in = true;
              double lo[3];
              for (int k = 0; k < 3; ++k)
              {
                lo[k] = d->global_min[k] + (double) cb[k] * w->bin[k];
                if (xi[k] < lo[k] - w->skin[k] || xi[k] > lo[k] + w->bin[k] + w->skin[k]) in = false;
              }
              if (!in) continue;
              *which = near_bin;
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
            *which = bi;
            memcpy(frame, xi, sizeof xi);
            memcpy(flo, lo, sizeof lo);
            return true;
          }
      }
  return false;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * What a lattice row is missing
 *
 * A lattice is declared globally, so a row can count, by arithmetic, the
 * lattice positions its support holds -- and compare with the samples it got.
 * Three counts per row, each over the OPEN support shrunk by a hair (so that a
 * sample a rounding error away from the support's edge, whose weight is zero
 * to round-off, is counted alike on both sides):
 *
 *   virtual   the lattice extended past a non-periodic end by its end spacing:
 *             the positions a row would have used with no wall, no lattice edge
 *             and no solid;
 *   in        the part of those the lattice actually has;
 *   present   the band samples the row was built over.
 *
 * present < virtual is a TRUNCATED row -- cut by a domain wall, by the edge of a
 * lattice smaller than the domain, or by masked (solid) samples -- and is what
 * the boundary policy judges. present < in where no provider reported a masked
 * sample, or present > in anywhere, cannot happen: coverage was proved, so it is
 * a coupler bug, and is reported as one rather than corrected away.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Margin, in kernel units, by which the counted support is shrunk. */
#define CPL_SUPPORT_HAIR 1e-10

/** @brief Position of a VIRTUAL lattice index: the lattice's own position inside it (every image on a periodic axis), extrapolated past a non-periodic end by the end spacing. */
static double lat_vpos(const cpl_location *L, const MFTL_Domain *d, int k, int64_t i)
{
  if (d->is_periodic[k]) return MFTLCplLatPos(L, d, k, i);
  const int64_t n = L->dims[k];
  if (i >= 0 && i < n) return MFTLCplLatCoord(L, k, i);
  if (L->kind == MFTL_LATTICE_UNIFORM) return L->origin[k] + (double) i * L->spacing[k];
  if (i < 0)
  {
    const double w = (n > 1) ? L->coords[k][1] - L->coords[k][0] : L->faces[k][1] - L->faces[k][0];
    return L->coords[k][0] + (double) i * w;
  }
  const double w = (n > 1) ? L->coords[k][n - 1] - L->coords[k][n - 2] : L->faces[k][1] - L->faces[k][0];
  return L->coords[k][n - 1] + (double) (i - (n - 1)) * w;
}

/** @brief The smallest index whose kernel argument `(vpos - x) inv` exceeds @p bound (reaches it when @p inclusive): galloping from an estimate. */
static int64_t lat_first_beyond(const cpl_location *L, const MFTL_Domain *d, int k, double x, double inv, double bound, bool inclusive)
{
#define CPL_PRED(i) (inclusive ? ((lat_vpos(L, d, k, (i)) - x) * inv >= bound) : ((lat_vpos(L, d, k, (i)) - x) * inv > bound))
  const int64_t n = L->dims[k];
  if (!isfinite(x) || !(inv > 0.0) || !isfinite(inv) || !isfinite(bound)) return 0; /* never gallop on a NaN */
  double est;
  if (L->kind == MFTL_LATTICE_UNIFORM)
    est = (x + bound / inv - L->origin[k]) / L->spacing[k];
  else
  {
    const double span = L->coords[k][n - 1] - L->coords[k][0];
    est = (n > 1 && span > 0.0) ? (x + bound / inv - L->coords[k][0]) * (double) (n - 1) / span : 0.0;
  }
  const double cap = 4.0 * (double) n + 1e6;
  if (!(est > -cap)) est = -cap;
  if (!(est < cap)) est = cap;
  int64_t lo = (int64_t) floor(est), hi;
  if (CPL_PRED(lo))
  {
    hi = lo;
    int64_t step = 1;
    lo = hi - step;
    while (CPL_PRED(lo) && step < (INT64_C(1) << 40))
    {
      hi = lo;
      step *= 2;
      lo = hi - step;
    }
  }
  else
  {
    int64_t step = 1;
    hi = lo + step;
    while (!CPL_PRED(hi) && step < (INT64_C(1) << 40))
    {
      lo = hi;
      step *= 2;
      hi = lo + step;
    }
  }
  /* invariant: !pred(lo), pred(hi) */
  while (hi - lo > 1)
  {
    const int64_t m = lo + (hi - lo) / 2;
    if (CPL_PRED(m))
      hi = m;
    else
      lo = m;
  }
#undef CPL_PRED
  return hi;
}

/**
 * @brief The three counts of a lattice row (see above), at marker @p x, kernel
 *        metric @p Ht and half-width @p hw, over its @p n candidates @p X.
 */
static void lat_row_counts(const cpl_location *L, const MFTL_Domain *d, bool radial, const double x[3], const double Ht[3], double hw, int n,
                           const double *X, int64_t *n_virtual, int64_t *n_in, int64_t *n_present)
{
  const double r = hw * (1.0 - CPL_SUPPORT_HAIR);
  double inv[3];
  int64_t a[3], b[3], ia[3], ib[3];
  for (int k = 0; k < 3; ++k)
  {
    inv[k] = 1.0 / Ht[k];
    a[k] = lat_first_beyond(L, d, k, x[k], inv[k], -r, false);
    b[k] = lat_first_beyond(L, d, k, x[k], inv[k], r, true);
    if (b[k] < a[k]) b[k] = a[k];
    ia[k] = d->is_periodic[k] ? a[k] : MFTL_MAX(a[k], 0);
    ib[k] = d->is_periodic[k] ? b[k] : MFTL_MIN(b[k], L->dims[k]);
    if (ib[k] < ia[k]) ib[k] = ia[k];
  }
  if (!radial)
  {
    *n_virtual = (b[0] - a[0]) * (b[1] - a[1]) * (b[2] - a[2]);
    *n_in = (ib[0] - ia[0]) * (ib[1] - ia[1]) * (ib[2] - ia[2]);
  }
  else
  {
    int64_t nv = 0, ni = 0;
    for (int64_t i2 = a[2]; i2 < b[2]; ++i2)
    {
      const double t2 = (lat_vpos(L, d, 2, i2) - x[2]) * inv[2];
      const bool in2 = i2 >= ia[2] && i2 < ib[2];
      for (int64_t i1 = a[1]; i1 < b[1]; ++i1)
      {
        const double t1 = (lat_vpos(L, d, 1, i1) - x[1]) * inv[1];
        const bool in1 = in2 && i1 >= ia[1] && i1 < ib[1];
        for (int64_t i0 = a[0]; i0 < b[0]; ++i0)
        {
          const double t0 = (lat_vpos(L, d, 0, i0) - x[0]) * inv[0];
          if (t0 * t0 + t1 * t1 + t2 * t2 >= r * r) continue;
          ++nv;
          if (in1 && i0 >= ia[0] && i0 < ib[0]) ++ni;
        }
      }
    }
    *n_virtual = nv;
    *n_in = ni;
  }
  int64_t np = 0;
  for (int j = 0; j < n; ++j)
  {
    const double t0 = (X[3 * j] - x[0]) * inv[0], t1 = (X[3 * j + 1] - x[1]) * inv[1], t2 = (X[3 * j + 2] - x[2]) * inv[2];
    if (radial ? (t0 * t0 + t1 * t1 + t2 * t2 < r * r) : (fabs(t0) < r && fabs(t1) < r && fabs(t2) < r)) ++np;
  }
  *n_present = np;
}

/**
 * @brief The mirror images of a row's real candidates (#MFTL_BOUNDARY_MIRROR),
 *        appended after them to @p X and @p q.
 *
 * Every non-empty set of the walls the support box `[lo, hi]` crosses (one per
 * non-periodic axis) reflects every real candidate across those walls, in x,
 * then y, then z; an image inside the box is kept. A candidate ON a wall is
 * its own image there and is not repeated. The image carries its sample's
 * value times the product of the parities, plus a share of each ODD face's
 * wall value: reflecting `v -> s v + (1 - s) u_w` axis by axis gives face k
 * the coefficient `(1 - s_k)` times the parities of the axes reflected after it.
 * The images' sources lie within the box (a reflection of a point within reach
 * of a marker inside the domain is within reach of it too), so the band holds
 * them. A marker outside the domain, or a support crossing both walls of one
 * axis, cannot be mirrored: @p bad is set.
 * @return The number of images appended.
 */
static int64_t mirror_images(const MFTL_Domain *d, const cpl_location *L, const double x[3], const double lo[3], const double hi[3], int64_t n_real, double *X,
                             double *q, int64_t *isrc, double *ipar, double *icoef, bool *bad)
{
  int face[3];
  int axes = 0;
  *bad = false;
  for (int k = 0; k < 3; ++k)
  {
    face[k] = -1;
    if (d->is_periodic[k]) continue;
    if (x[k] < d->global_min[k] || x[k] > d->global_max[k]) *bad = true;
    const bool cl = lo[k] < d->global_min[k], ch = hi[k] > d->global_max[k];
    if (cl && ch) *bad = true;
    if (cl || ch)
    {
      face[k] = cl ? 0 : 1;
      axes |= 1 << k;
    }
  }
  if (*bad || axes == 0) return 0;
  int64_t n = 0;
  for (int S = 1; S < 8; ++S)
  {
    if ((S & axes) != S) continue;
    for (int64_t j = 0; j < n_real; ++j)
    {
      const double *src = &X[3 * j];
      double p[3] = {src[0], src[1], src[2]};
      bool keep = true;
      double s = 1.0, coef[6] = {0, 0, 0, 0, 0, 0};
      for (int k = 0; k < 3 && keep; ++k)
      {
        if (!(S & (1 << k))) continue;
        const double W = face[k] ? d->global_max[k] : d->global_min[k];
        if (src[k] == W) keep = false; /* on the wall: its own image */
        p[k] = 2.0 * W - src[k];
        const int f = k + 3 * face[k];
        const double sk = (L->wall[f] == MFTL_WALL_ODD) ? -1.0 : 1.0;
        /* v -> sk v + (1 - sk) u_w: earlier faces' shares are multiplied by sk */
        for (int g = 0; g < 6; ++g) coef[g] *= sk;
        coef[f] += 1.0 - sk;
        s *= sk;
      }
      if (!keep) continue;
      if (p[0] < lo[0] || p[0] > hi[0] || p[1] < lo[1] || p[1] > hi[1] || p[2] < lo[2] || p[2] > hi[2]) continue;
      const int64_t at = n_real + n;
      memcpy(&X[3 * at], p, sizeof p);
      q[at] = q[j];
      isrc[n] = j;
      ipar[n] = s;
      memcpy(&icoef[6 * n], coef, sizeof coef);
      ++n;
    }
  }
  return n;
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

  bind_scratch *scratch = &c->bs;
  cand_t cd = scratch->candidates;
  int cap_c = scratch->cap_c, cap_img = scratch->cap_img;
  int64_t n_sel = 0, *sel = scratch->sel;
  double *X = scratch->X, *q = scratch->q, *wt = scratch->wt;
  int64_t *isrc = scratch->isrc;
  double *ipar = scratch->ipar, *icoef = scratch->icoef;

  const bool grow = (P->failure & MFTL_STENCIL_GROW) != 0 && P->max_growth > 1.0;
  const bool lower = (P->failure & MFTL_STENCIL_LOWER_ORDER) != 0 && P->reproduction == MFTL_REPRODUCE_LINEAR;

  for (int lp = 0; lp < b->n_loc; ++lp)
  {
    const cpl_band *B = &w->band[b->loc_band[lp]];
    const cpl_locinfo *li = &c->li[B->location];
    const cpl_location *Lc = &c->grid->loc[B->location];
    const MFTL_BoundaryPolicy bpol = Lc->boundary;
    const bool mirror = (bpol == MFTL_BOUNDARY_MIRROR);
    int64_t cap_e = 0, ne = 0;
    b->row_ptr[lp] = (int64_t *) MFTLCplAlloc((size_t) (b->n_markers + 1) * sizeof(int64_t), "rows");
    b->col[lp] = MFTL_NULL;
    b->w[lp] = MFTL_NULL;
    b->row_ptr[lp][0] = 0;
    if (mirror) b->wallc[lp] = (double *) MFTLCplCalloc(6 * (size_t) MFTL_MAX(b->n_markers, 1), sizeof(double), "wall coefficients");

    for (int64_t i = 0; i < b->n_markers; ++i)
    {
      const double *x = &b->pos[3 * i];
      cpl_tensor_row direct;
      double residual, amplification;
      const int direct_n = MFTLCplCartesianRow(B, Lc, d, &c->kernel, x, P->moment_tolerance, P->max_amplification, &direct, &residual, &amplification);
      if (direct_n >= 0)
      {
        if (!b->tensor[lp]) b->tensor[lp] = MFTLCplCalloc((size_t) b->n_markers, sizeof(cpl_tensor_row), "tensor rows");
        b->tensor[lp][i] = direct;
        b->row_ptr[lp][i + 1] = ne;
        Q->n_entries += direct_n;
        Q->n_rows++;
        Q->n_cartesian++;
        Q->max_amplification = fmax(Q->max_amplification, amplification);
        Q->max_moment_residual = fmax(Q->max_moment_residual, residual);
        continue;
      }
      double H[3], Hm[9] = {0.0}; /* Hm is read only when aniso set it; GCC cannot see that */
      bool ok = true;
      const bool aniso = (B->cell != MFTL_NULL);
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
        /* Sample-centred compact metric influences. The certified envelope
         * includes every contributor, even when its own resolution is coarser. */
        double plo[3], phi[3];
        for (int k = 0; k < 3; ++k)
        {
          Rp[k] = w->reach[3 * (b->loc_band[lp] * w->n_bins + b->support_bin[i]) + k];
          plo[k] = x[k] - Rp[k];
          phi[k] = x[k] + Rp[k];
        }
        window_candidates(c, w, b->loc_band[lp], b->support_bin[i], plo, phi, &cd);
        double sw = 0.0, sh[3] = {0.0, 0.0, 0.0}, sm[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t e = 0; e < cd.n; ++e)
        {
          const double *p = &B->ipos[3 * cd.img[e]];
          const int64_t dof = B->idof[cd.img[e]];
          const double *hd = &B->h[3 * dof];
          double rho2 = 0.0;
          for (int k = 0; k < 3; ++k)
          {
            const double t = (p[k] - x[k]) / (P->pilot_factor * hd[k]);
            rho2 += t * t;
          }
          if (rho2 >= 1.0) continue;
          const double rho = sqrt(rho2), s = 1.0 - rho;
          const double wgt = (B->measure[dof] / hd[0] / hd[1] / hd[2]) * s * s * s * s * (4.0 * rho + 1.0);
          sw += wgt;
          for (int k = 0; k < 3; ++k) sh[k] += wgt * hd[k];
          if (aniso)
          {
            const double *cl = &B->cell[9 * B->idof[cd.img[e]]];
            for (int a = 0; a < 9; ++a) sm[a] += wgt * cl[a];
          }
        }
        if (!(sw > 0.0))
          ok = false;
        else if (!aniso)
          for (int k = 0; k < 3; ++k) H[k] = sh[k] / sw;
        else
        {
          /* the averaged cell; its extent along each axis bounds the support box */
          for (int a = 0; a < 9; ++a) Hm[a] = sm[a] / sw;
          for (int k = 0; k < 3; ++k)
            H[k] = c->kernel.radial ? sqrt(Hm[3 * k] * Hm[3 * k] + Hm[3 * k + 1] * Hm[3 * k + 1] + Hm[3 * k + 2] * Hm[3 * k + 2])
                                    : fabs(Hm[3 * k]) + fabs(Hm[3 * k + 1]) + fabs(Hm[3 * k + 2]);
        }
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
        if (!inside_pilot) window_candidates(c, w, b->loc_band[lp], b->support_bin[i], slo, shi, &cd);
      }

      MFTL_RowQuality rq;
      memset(&rq, 0, sizeof rq);
      int rc = MFTL_COUPLING_BAD_STENCIL;
      bool truncated = false, internal = false;
      int64_t n_real = 0, n_img = 0;
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
              /* a point cloud has no positions to count: the support crossing a wall is the test (mirror images complete it) */
              if (!li->lattice && !mirror && !d->is_periodic[k] && (lo[k] < d->global_min[k] || hi[k] > d->global_max[k])) truncated = true;
            }
            /* room for the real candidates and up to seven images of each */
            const int64_t need_c = mirror ? 8 * cd.n : cd.n;
            if (need_c > cap_c)
            {
              cap_c = (int) MFTLGrowCapacity(cap_c, (int) need_c);
              X = (double *) MFTLCplRealloc(X, 3 * (size_t) cap_c * sizeof(double), "row candidates");
              q = (double *) MFTLCplRealloc(q, (size_t) cap_c * sizeof(double), "row candidates");
              wt = (double *) MFTLCplRealloc(wt, (size_t) cap_c * sizeof(double), "row candidates");
              sel = (int64_t *) MFTLCplRealloc(sel, (size_t) cap_c * sizeof(int64_t), "row candidates");
            }
            /* Ordinary locations can grow the candidate buffers without using
             * images. Their capacity says nothing about these separate arrays. */
            if (mirror && need_c > cap_img)
            {
              cap_img = (int) MFTLGrowCapacity(cap_img, (int) need_c);
              isrc = (int64_t *) MFTLCplRealloc(isrc, (size_t) cap_img * sizeof(int64_t), "row images");
              ipar = (double *) MFTLCplRealloc(ipar, (size_t) cap_img * sizeof(double), "row images");
              icoef = (double *) MFTLCplRealloc(icoef, 6 * (size_t) cap_img * sizeof(double), "row images");
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
            n_real = n_sel;
            n_img = 0;
            bool mirror_bad = false;
            if (mirror)
              n_img = mirror_images(d, Lc, x, lo, hi, n_real, X, q, isrc, ipar, icoef, &mirror_bad);
            n_sel = n_real + n_img;
            if (li->lattice)
            {
              /* exact, by counting lattice positions (see lat_row_counts): the
               * real samples against the lattice, real and images against the
               * virtual positions they complete */
              int64_t nv = 0, nin = 0, npr = 0;
              lat_row_counts(Lc, d, c->kernel.radial, x, Ht, c->kernel.half_width, (int) n_real, X, &nv, &nin, &npr);
              if (npr > nin || (npr < nin && !B->has_masked))
              {
                fprintf(stderr, "MFTL coupling: internal error: the row of marker %llu (location '%s') holds %lld of the %lld lattice samples in its support\n",
                        (unsigned long long) b->ref[i].key, Lc->name, (long long) npr, (long long) nin);
                internal = true;
              }
              int64_t npr_img = 0;
              if (n_img > 0)
              {
                int64_t nv2, nin2;
                lat_row_counts(Lc, d, c->kernel.radial, x, Ht, c->kernel.half_width, (int) n_img, X + 3 * n_real, &nv2, &nin2, &npr_img);
              }
              truncated = npr + npr_img < nv;
            }
            if (aniso)
            {
              double Ht9[9];
              for (int a = 0; a < 9; ++a) Ht9[a] = P->support_scale * gfac * Hm[a];
              rc = MFTLCplRowWeightsCell(&pp, &c->kernel, x, Ht9, (int) n_sel, X, q, wt, &rq, &c->rs);
            }
            else
              rc = MFTLCplRowWeights(&pp, &c->kernel, x, Ht, (int) n_sel, X, q, wt, &rq, &c->rs);
            if (mirror_bad) rc = MFTL_COUPLING_BAD_STENCIL;
            rq.n_grow = step;
            if (internal) rc = MFTL_COUPLING_INVALID;
            if (rc == MFTL_COUPLING_OK && truncated && bpol == MFTL_BOUNDARY_ERROR) rc = MFTL_COUPLING_BAD_STENCIL;
            if (internal) break;
            if (rc == MFTL_COUPLING_OK || !grow || gfac >= P->max_growth) break;
            gfac = fmin(gfac * P->growth_step, P->max_growth);
          }
          if (rc == MFTL_COUPLING_OK && pass == 1) rq.lower_order = true;
          if (internal) break;
        }
      }

      if (rc != MFTL_COUPLING_OK)
      {
        if (Q->n_failed == 0) Q->first_failed_key = b->ref[i].key;
        Q->n_failed++;
        if (internal)
          status = MFTL_COUPLING_INVALID;
        else if (status < MFTL_COUPLING_BAD_STENCIL)
          status = MFTL_COUPLING_BAD_STENCIL;
        b->row_ptr[lp][i + 1] = ne;
        continue;
      }
      const int64_t row0 = ne;
      for (int64_t e = 0; e < n_real; ++e)
      {
        if (wt[e] == 0.0) continue;
        if (ne == cap_e)
        {
          cap_e = (cap_e == 0) ? 1024 : 2 * cap_e;
          b->col[lp] = (int64_t *) MFTLCplRealloc(b->col[lp], (size_t) cap_e * sizeof(int64_t), "rows");
          b->w[lp] = (double *) MFTLCplRealloc(b->w[lp], (size_t) cap_e * sizeof(double), "rows");
        }
        b->col[lp][ne] = B->idof[sel[e]];
        b->w[lp][ne] = wt[e];
        ++ne;
      }
      if (n_img > 0)
      {
        /* compose: an image's weight goes to its sample times the parity, its
         * wall share to the faces; merged into the real entries in a fixed
         * order (real first, images in generation order) */
        double *wc = &b->wallc[lp][6 * i];
        for (int64_t e = 0; e < n_img; ++e)
        {
          const double we = wt[n_real + e];
          if (we == 0.0) continue;
          for (int f = 0; f < 6; ++f) wc[f] += we * icoef[6 * e + f];
          const int64_t col = B->idof[sel[isrc[e]]];
          int64_t at = -1;
          for (int64_t r = row0; r < ne; ++r)
            if (b->col[lp][r] == col) at = r;
          if (at < 0)
          {
            if (ne == cap_e)
            {
              cap_e = (cap_e == 0) ? 1024 : 2 * cap_e;
              b->col[lp] = (int64_t *) MFTLCplRealloc(b->col[lp], (size_t) cap_e * sizeof(int64_t), "rows");
              b->w[lp] = (double *) MFTLCplRealloc(b->w[lp], (size_t) cap_e * sizeof(double), "rows");
            }
            at = ne++;
            b->col[lp][at] = col;
            b->w[lp][at] = 0.0;
          }
          b->w[lp][at] += we * ipar[e];
        }
        /* an exact cancellation (an ODD image on its own sample) leaves no entry */
        int64_t keep = row0;
        for (int64_t r = row0; r < ne; ++r)
          if (b->w[lp][r] != 0.0)
          {
            b->col[lp][keep] = b->col[lp][r];
            b->w[lp][keep] = b->w[lp][r];
            ++keep;
          }
        ne = keep;
        Q->n_mirrored++;
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
      b->col[lp] = (int64_t *) MFTLCplAlloc(sizeof(int64_t), "rows");
      b->w[lp] = (double *) MFTLCplAlloc(sizeof(double), "rows");
    }
  }
  scratch->candidates = cd;
  scratch->cap_c = cap_c;
  scratch->cap_img = cap_img;
  scratch->sel = sel;
  scratch->X = X;
  scratch->q = q;
  scratch->wt = wt;
  scratch->isrc = isrc;
  scratch->ipar = ipar;
  scratch->icoef = icoef;
  return status;
}

/** @brief Local spacing check: worst edge over the finest resolution the directory knows nearby. */
static void spacing_local(const MFTL_Coupler *c, double *max_ratio, int64_t *n_off, int *worst_body, int *worst_tri)
{
  int *got = MFTL_NULL, gcap = 0;
  const MFTL_Ensemble *ens = c->ens;
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
      const double h = local_hmin(c, lo, hi, 0, MFTL_NULL, &got, &gcap);
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
  MFTLCplFree(got);
}

/** @brief A fresh binding shell. */
static MFTL_CouplingBinding *new_binding(MFTL_Coupler *c, const MFTL_CouplingWindow *w, int n_loc, int64_t n)
{
  MFTL_CouplingBinding *b = (MFTL_CouplingBinding *) MFTLCplCalloc(1, sizeof(MFTL_CouplingBinding), "binding");
  b->c = c;
  b->win = w;
  b->window_id = w->id;
  b->n_loc = n_loc;
  b->loc_band = (int *) MFTLCplAlloc((size_t) MFTL_MAX(n_loc, 1) * sizeof(int), "binding");
  b->row_ptr = (int64_t **) MFTLCplCalloc((size_t) MFTL_MAX(n_loc, 1), sizeof(int64_t *), "binding");
  b->col = (int64_t **) MFTLCplCalloc((size_t) MFTL_MAX(n_loc, 1), sizeof(int64_t *), "binding");
  b->w = (double **) MFTLCplCalloc((size_t) MFTL_MAX(n_loc, 1), sizeof(double *), "binding");
  b->wallc = (double **) MFTLCplCalloc((size_t) MFTL_MAX(n_loc, 1), sizeof(double *), "binding");
  b->tensor = MFTLCplCalloc((size_t) MFTL_MAX(n_loc, 1), sizeof(cpl_tensor_row *), "binding");
  b->n_markers = n;
  b->support_bin = MFTLCplAlloc((size_t) MFTL_MAX(n, 1) * sizeof(int64_t), "binding certificates");
  b->pos = (double *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(n, 1) * sizeof(double), "binding");
  b->base_pos = (double *) MFTLCplAlloc(3 * (size_t) MFTL_MAX(n, 1) * sizeof(double), "binding");
  b->ref = (MFTL_MarkerRef *) MFTLCplAlloc((size_t) MFTL_MAX(n, 1) * sizeof(MFTL_MarkerRef), "binding");
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

/**
 * @brief What every rank must bind alike, read from the SPEC as given -- the
 *        kind, the location list, the quadrature policy -- never from anything
 *        a rank derives only when its local checks pass: a rank already stale
 *        must sign the same as the others, or the agreed STALE becomes INVALID.
 */
static uint64_t bind_signature(const MFTL_MarkerSpec *spec)
{
  uint64_t h = CPL_FNV_INIT;
  if (spec == MFTL_NULL) return h;
  const int kind = (int) spec->kind, nl = spec->n_locations;
  h = cpl_fnv(h, &kind, sizeof kind);
  h = cpl_fnv(h, &nl, sizeof nl);
  if (nl > 0 && spec->locations) h = cpl_fnv(h, spec->locations, (size_t) nl * sizeof(int));
  if (spec->kind == MFTL_MARKERS_QUADRATURE)
  {
    const int ql[2] = {spec->quadrature_level, spec->quadrature_max_level};
    h = cpl_fnv(h, ql, sizeof ql);
    h = cpl_fnv(h, &spec->quadrature_ratio, sizeof spec->quadrature_ratio);
  }
  const int io = spec->interpolation_only ? 1 : 0;
  return cpl_fnv(h, &io, sizeof io);
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
  int64_t n_capped = 0;
  if (local == MFTL_COUPLING_OK && !resolve_locations(c, spec, &n_loc, &locs)) local = MFTL_COUPLING_INVALID;
  if (local == MFTL_COUPLING_OK) local = gather_markers(c, spec, locs, n_loc, &m, &n_capped);
  if (local == MFTL_COUPLING_OK)
  {
    b = new_binding(c, w, n_loc, m.n);
    b->kind = spec->kind;
    b->q_level = spec->quadrature_level;
    b->q_max = spec->quadrature_max_level;
    b->q_ratio = spec->quadrature_ratio;
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
      if (!contain(w, d, &m.pos[3 * i], &b->pos[3 * i], flo, &b->support_bin[i])) local = MFTL_COUPLING_STALE;
    }
    if (local == MFTL_COUPLING_OK)
    {
      memcpy(b->base_pos, b->pos, 3 * (size_t) m.n * sizeof(double));
      local = assemble_rows(c, b, w);
    }
    if (local == MFTL_COUPLING_OK && c->params.sampling == MFTL_SAMPLING_ERROR && !spec->interpolation_only && spec->kind != MFTL_MARKERS_USER)
    {
      if (spec->kind == MFTL_MARKERS_QUADRATURE)
      {
        /* the points resolve the fluid by construction, unless a triangle needed more than the cap */
        if (n_capped > 0) local = MFTL_COUPLING_BAD_SAMPLING;
      }
      else
      {
        double r;
        int64_t n_off;
        int wb, wt;
        spacing_local(c, &r, &n_off, &wb, &wt);
        if (n_off > 0) local = MFTL_COUPLING_BAD_SAMPLING;
      }
    }
  }
  /* One agreement covers the status AND what was bound: ranks binding
   * different kinds or locations would take different paths below. */
  int st = MFTLCplAgreeTransfer(c->comm, local, bind_signature(spec));
  if (st == MFTL_COUPLING_OK && spec->kind == MFTL_MARKERS_QUADRATURE)
  {
    /* the vertex table and its ghost schedule: collective, every rank enters */
    cpl_quad *qd = MFTL_NULL;
    local = MFTLCplQuadBuild(c->comm, c->ens, &m, &qd, &c->stats.bytes_discovery, &c->stats.messages);
    b->quad = qd;
    b->quad_owned = true;
    st = agree(c, local);
    if (st == MFTL_COUPLING_OK) qd->id = ++c->quad_counter;
  }
  MFTLCplFree(locs);
  MFTLCplMarkersFree(&m);
  if (st != MFTL_COUPLING_OK)
  {
    if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
    if (b) MFTLCouplerReleaseBinding(&b);
    return st;
  }
  *out = b;
  return MFTL_COUPLING_OK;
}

/**
 * @brief The rows of a stage: everything BindStage does except agreeing.
 *        Local. @p allow_active admits a transfer in flight (the local form
 *        reads only the window, never the transfer state).
 */
static int bind_stage_core(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, bool allow_active, MFTL_CouplingBinding **bp)
{
  *bp = MFTL_NULL;
  int local = MFTL_COUPLING_OK;
  if (base == MFTL_NULL || stage == MFTL_NULL || base->c != c || (c->active && !allow_active) || (base->n_markers > 0 && stage->pos == MFTL_NULL))
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
    b->quad = base->quad;
    b->quad_owned = false;
    b->q_level = base->q_level;
    b->q_max = base->q_max;
    b->q_ratio = base->q_ratio;
    /* memcpy from a NULL source is undefined even for zero bytes, and an empty
     * binding's arrays may be NULL: copy only what exists. */
    if (base->n_loc > 0) memcpy(b->loc_band, base->loc_band, (size_t) base->n_loc * sizeof(int));
    if (base->n_markers > 0)
    {
      memcpy(b->ref, base->ref, (size_t) base->n_markers * sizeof(MFTL_MarkerRef));
      memcpy(b->base_pos, base->base_pos, 3 * (size_t) base->n_markers * sizeof(double));
    }
    b->n_bodies = base->n_bodies;
    b->bodies = (cpl_body_snap *) MFTLCplAlloc((size_t) MFTL_MAX(base->n_bodies, 1) * sizeof(cpl_body_snap), "binding bodies");
    if (base->n_bodies > 0) memcpy(b->bodies, base->bodies, (size_t) base->n_bodies * sizeof(cpl_body_snap));
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
      if (stage->weight && (!isfinite(stage->weight[i]) || (b->kind == MFTL_MARKERS_QUADRATURE && !(stage->weight[i] > 0.0))))
        local = MFTL_COUPLING_INVALID;
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
      const int64_t base_bin = find_u64(w->bin_key, w->n_bins, bin_key3(ib[0], ib[1], ib[2]));
      if (in && base_bin >= 0)
      {
        memcpy(&b->pos[3 * i], xs, sizeof xs);
        b->support_bin[i] = base_bin;
      }
      else if (!contain(w, d, xs, &b->pos[3 * i], flo, &b->support_bin[i]))
        local = MFTL_COUPLING_STALE;
    }
    if (local == MFTL_COUPLING_OK) local = assemble_rows(c, b, w);
  }
  *bp = b;
  return local;
}

int MFTLCouplerBindStage(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, MFTL_CouplingBinding **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  c->stats.n_stage_binds++;
  MFTL_CouplingBinding *b = MFTL_NULL;
  const int local = (out == MFTL_NULL) ? MFTL_COUPLING_INVALID : bind_stage_core(c, base, stage, false, &b);
  return finish_bind(c, b, local, out);
}

int MFTLCouplerBindStageLocal(MFTL_Coupler *c, const MFTL_CouplingBinding *base, const MFTL_StageGeometry *stage, MFTL_CouplingBinding **out)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL || out == MFTL_NULL) return MFTL_COUPLING_INVALID;
  c->stats.n_stage_binds++;
  c->stats.n_local_stage_binds++;
  MFTL_CouplingBinding *b = MFTL_NULL;
  const int local = bind_stage_core(c, base, stage, true, &b);
  if (local != MFTL_COUPLING_OK)
  {
    if (local == MFTL_COUPLING_STALE) c->stats.n_stale++;
    if (b) MFTLCouplerReleaseBinding(&b);
    return local;
  }
  *out = b;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerAgreeStatus(MFTL_Coupler *c, int local)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  if (local < MFTL_COUPLING_OK || local > MFTL_COUPLING_INVALID) local = MFTL_COUPLING_INVALID;
  return agree(c, local);
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
  MFTLCplFree((*pr)->fields);
  MFTLCplFree((*pr)->field_band);
  MFTLCplFree((*pr)->fetch_off);
  MFTLCplFree((*pr)->serve_off);
  MFTLCplFree(*pr);
  *pr = MFTL_NULL;
}

void MFTLCouplerReleaseFetch(MFTL_CouplingFetch **pf)
{
  if (pf == MFTL_NULL || *pf == MFTL_NULL) return;
  MFTL_CouplingFetch *f = *pf;
  MFTL_CouplingWindow *w = (MFTL_CouplingWindow *) f->win;
  w->refcount--;
  if (w->c) w->c->live--;
  /* Live snapshots never enter this pool, so simultaneous RK stages remain
   * immutable. The window owns the cache and frees it with its geometry. */
  if (w->n_fetch_pool < 3)
  {
    f->next = w->fetch_pool;
    w->fetch_pool = f;
    w->n_fetch_pool++;
  }
  else
    fetch_storage_free(f);
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
  s.quadrature_level = b->q_level;
  s.quadrature_max_level = b->q_max;
  s.quadrature_ratio = b->q_ratio;
  cpl_markers m;
  int *locs = (int *) MFTLCplAlloc((size_t) MFTL_MAX(b->n_loc, 1) * sizeof(int), "binding locations");
  for (int i = 0; i < b->n_loc; ++i) locs[i] = b->win->locs[b->loc_band[i]];
  const int gst = gather_markers(c, &s, locs, b->n_loc, &m, MFTL_NULL);
  MFTLCplFree(locs);
  if (gst != MFTL_COUPLING_OK) return true;
  const bool moved = (m.n != b->n_markers) || MFTLCplPositionsChecksum(m.pos, m.n) != b->pos_checksum;
  MFTLCplMarkersFree(&m);
  if (moved) fprintf(stderr, "MFTL coupling: bound vertex positions changed without MFTLObjectMarkChanged()\n");
  return moved;
#endif
}

static int prepare_transfer(MFTL_Coupler *c, const MFTL_TransferSpec *spec, MFTL_TransferReady **out, int local)
{
  if (out) *out = MFTL_NULL;
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
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
    r = (MFTL_TransferReady *) MFTLCplCalloc(1, sizeof(MFTL_TransferReady), "transfer");
    r->c = c;
    r->direction = spec->direction;
    r->win = w;
    r->binding = (spec->direction == MFTL_TRANSFER_SPREAD) ? spec->binding : MFTL_NULL;
    r->window_id = w->id;
    r->n_fields = spec->n_fields;
    r->form = spec->form;
    r->fields = (MFTL_SolverField *) MFTLCplAlloc((size_t) MFTL_MAX(spec->n_fields, 1) * sizeof(MFTL_SolverField), "transfer");
    r->field_band = (int *) MFTLCplAlloc((size_t) MFTL_MAX(spec->n_fields, 1) * sizeof(int), "transfer");
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
      if (F->wall_value)
        for (int g = 0; g < 6; ++g)
          if (!isfinite(F->wall_value[g])) local = MFTL_COUPLING_INVALID;
      /* the solver's own halo accumulation would bypass the constraint transpose */
      if (F->halo_transpose && c->li[F->location].constrained) local = MFTL_COUPLING_INVALID;
      for (int p = 0; p < np && local == MFTL_COUPLING_OK; ++p)
        if (F->data[p] == MFTL_NULL) local = MFTL_COUPLING_INVALID;
      if (spec->direction == MFTL_TRANSFER_SPREAD)
      {
        bool bound = false;
        for (int lp = 0; lp < spec->binding->n_loc; ++lp)
          if (spec->binding->loc_band[lp] == r->field_band[f]) bound = true;
        if (!bound) local = MFTL_COUPLING_INVALID;
      }
      const int fl[3] = {F->location, F->halo_valid ? 1 : 0, F->halo_transpose ? 1 : 0};
      sig = cpl_fnv(sig, fl, sizeof fl);
    }
  }
  int st = MFTLCplAgreeValidated(c->comm, local, sig);
  if (st != MFTL_COUPLING_OK)
  {
    if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
    MFTLCouplerReleaseReady(&r);
    return st;
  }
  r->fetch_off = (int64_t *) MFTLCplCalloc((size_t) w->n_fetch_peers * r->n_fields + 1, sizeof(int64_t), "transfer offsets");
  r->serve_off = (int64_t *) MFTLCplCalloc((size_t) w->n_serve_peers * r->n_fields + 1, sizeof(int64_t), "transfer offsets");
  for (int side = 0; side < 2; ++side)
  {
    const int np = side ? w->n_serve_peers : w->n_fetch_peers;
    const int *segments = side ? w->serve_segment : w->fetch_segment;
    int64_t *off = side ? r->serve_off : r->fetch_off;
    for (int p = 0; p < np; ++p)
      for (int i = 0; i < r->n_fields; ++i)
      {
        const cpl_band *B = &w->band[r->field_band[i]];
        const int j = segments[(size_t) p * w->n_loc + r->field_band[i]];
        const bool omit_halo = r->direction == MFTL_TRANSFER_FETCH ? r->fields[i].halo_valid : r->fields[i].halo_transpose;
        int64_t len = 0;
        if (j >= 0) len = side ? B->dst_nonhalo[j] + (omit_halo ? 0 : B->dst_halo[j])
                              : B->src_nonhalo[j] + (omit_halo ? 0 : B->src_halo[j]);
        const int64_t a = (int64_t) p * r->n_fields + i;
        off[a + 1] = off[a] + len;
      }
  }
  r->id = ++c->transfer_counter;
  r->retained = true;
  ((MFTL_CouplingWindow *) w)->refcount++;
  if (r->binding) ((MFTL_CouplingBinding *) r->binding)->refcount++;
  c->live++;
  *out = r;
  return MFTL_COUPLING_OK;
}

int MFTLCouplerPrepareTransfer(MFTL_Coupler *c, const MFTL_TransferSpec *spec, MFTL_TransferReady **out)
{
  return prepare_transfer(c, spec, out, MFTL_COUPLING_OK);
}

/** @brief Whether every one of @p n loads is finite: a NaN spread would poison
 *         the owner's sample and, through the solver's next halo exchange, its
 *         neighbours'. One pass over what the spread reads anyway. */
static bool loads_finite(const double *loads, int64_t n)
{
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += loads[i] * 0.0; /* NaN for any NaN or infinity, 0 otherwise */
  return acc == 0.0;
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
    else if (!loads_finite(loads, b->n_markers * (int64_t) r->n_fields))
    {
      fprintf(stderr, "MFTL coupling: a spread load is not finite\n");
      local = MFTL_COUPLING_INVALID;
    }
    else if ((b->kind != MFTL_MARKERS_USER && !MFTLCplBodiesCurrent(c->ens, b->bodies, b->n_bodies)) || positions_moved_silently(c, b))
      local = MFTL_COUPLING_STALE;
  }
  const int st = MFTLCplAgreeTransfer(c->comm, local, r ? r->id : 0);
  if (st == MFTL_COUPLING_STALE) c->stats.n_stale++;
  return st;
}

/**
 * @brief Doubles a peer exchanges for one field of one band. Forward: the halo
 *        part travels unless the field's ghosts are valid (read in place).
 *        Reverse: it travels unless the solver completes its transpose
 *        (halo_transpose: the loads stay in this rank's ghost elements).
 */
static int64_t seg_fwd(int64_t nonhalo, int64_t halo, const MFTL_SolverField *F) { return nonhalo + (F->halo_valid ? 0 : halo); }
static int64_t seg_rev(int64_t nonhalo, int64_t halo, const MFTL_SolverField *F) { return nonhalo + (F->halo_transpose ? 0 : halo); }

static void ensure_staging(MFTL_Coupler *c, int64_t ns, int64_t nr)
{
  if (ns > c->scap)
  {
    c->scap = MFTL_MAX(ns, 2 * c->scap);
    MFTLCplFree(c->sbuf);
    c->sbuf = (double *) MFTLCplAlloc((size_t) c->scap * sizeof(double), "coupling staging");
  }
  if (nr > c->rcap)
  {
    c->rcap = MFTL_MAX(nr, 2 * c->rcap);
    MFTLCplFree(c->rbuf);
    c->rbuf = (double *) MFTLCplAlloc((size_t) c->rcap * sizeof(double), "coupling staging");
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
  MFTL_CouplingWindow *mw = (MFTL_CouplingWindow *) w;
  MFTL_CouplingFetch *f = mw->fetch_pool;
  if (f)
  {
    mw->fetch_pool = f->next;
    mw->n_fetch_pool--;
    bool same = f->n_fields == r->n_fields;
    for (int i = 0; i < r->n_fields && same; ++i) same = f->field_band[i] == r->field_band[i];
    if (!same)
    {
      fetch_storage_free(f);
      f = MFTL_NULL;
    }
    else
      c->stats.n_fetch_reused++;
  }
  if (!f)
  {
    f = (MFTL_CouplingFetch *) MFTLCplCalloc(1, sizeof(MFTL_CouplingFetch), "fetch");
    f->win = w;
    f->window_id = w->id;
    f->n_fields = r->n_fields;
    f->field_band = (int *) MFTLCplAlloc((size_t) MFTL_MAX(r->n_fields, 1) * sizeof(int), "fetch");
    f->values = (double **) MFTLCplCalloc((size_t) MFTL_MAX(r->n_fields, 1), sizeof(double *), "fetch");
    f->version = (uint64_t *) MFTLCplAlloc((size_t) MFTL_MAX(r->n_fields, 1) * sizeof(uint64_t), "fetch");
    f->wall = (double *) MFTLCplCalloc(6 * (size_t) MFTL_MAX(r->n_fields, 1), sizeof(double), "fetch walls");
    for (int i = 0; i < r->n_fields; ++i)
    {
      f->field_band[i] = r->field_band[i];
      f->values[i] = (double *) MFTLCplAlloc((size_t) MFTL_MAX(w->band[r->field_band[i]].n_dof, 1) * sizeof(double), "fetch values");
    }
  }
  f->next = MFTL_NULL;
  for (int i = 0; i < r->n_fields; ++i)
  {
    f->version[i] = r->fields[i].version;
    for (int g = 0; g < 6; ++g) f->wall[6 * i + g] = r->fields[i].wall_value ? r->fields[i].wall_value[g] : 0.0;
  }
  ((MFTL_CouplingWindow *) w)->refcount++;
  c->live++;

  const int64_t ns = r->serve_off[(int64_t) w->n_serve_peers * r->n_fields];
  const int64_t nr = r->fetch_off[(int64_t) w->n_fetch_peers * r->n_fields];
  ensure_staging(c, ns, nr);

  /* receives */
  int64_t at = 0;
  for (int p = 0; p < w->n_fetch_peers; ++p)
  {
    const int64_t len = r->fetch_off[(int64_t) (p + 1) * r->n_fields] - r->fetch_off[(int64_t) p * r->n_fields];
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
      const int s = w->serve_segment[(size_t) p * w->n_loc + r->field_band[i]];
      if (s < 0) continue;
      const int64_t len = seg_fwd(B->dst_nonhalo[s], B->dst_halo[s], &r->fields[i]);
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
      const int s = w->fetch_segment[(size_t) p * w->n_loc + r->field_band[i]];
      if (s < 0) continue;
      const int64_t len = seg_fwd(B->src_nonhalo[s], B->src_halo[s], &r->fields[i]);
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
    const double *wc = b->wallc[lp], *wv = &f->wall[6 * i];
    for (int64_t m = 0; m < b->n_markers; ++m)
    {
      double acc = 0.0;
      if (b->tensor[lp] && b->tensor[lp][m].count)
        acc = MFTLCplTensorValue(&b->tensor[lp][m], v);
      else
        for (int64_t e = rp[m]; e < rp[m + 1]; ++e) acc += wt[e] * v[col[e]];
      if (wc)
        for (int g = 0; g < 6; ++g) acc += wc[6 * m + g] * wv[g];
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
  if (c->reaction_cap < 6 * nf)
  {
    c->reaction_cap = 6 * nf;
    c->reaction = (double *) MFTLCplRealloc(c->reaction, (size_t) c->reaction_cap * sizeof(double), "wall reactions");
  }
  if (c->spread_scratch_n < nf)
  {
    c->spread_scratch = (double **) MFTLCplRealloc(c->spread_scratch, (size_t) nf * sizeof(double *), "spread scratch");
    c->spread_scratch_cap = (int64_t *) MFTLCplRealloc(c->spread_scratch_cap, (size_t) nf * sizeof(int64_t), "spread scratch");
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
      MFTLCplFree(c->spread_scratch[i]);
      c->spread_scratch_cap[i] = B->n_dof;
      c->spread_scratch[i] = (double *) MFTLCplAlloc((size_t) B->n_dof * sizeof(double), "spread scratch");
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
      if (b->tensor[lp] && b->tensor[lp][m].count)
        MFTLCplTensorSpread(&b->tensor[lp][m], F, s);
      else
        for (int64_t e = rp[m]; e < rp[m + 1]; ++e) s[col[e]] += wt[e] * F;
    }
    /* the walls' share: what an ODD face's prescribed value absorbs */
    double *R = &c->reaction[6 * i];
    for (int g = 0; g < 6; ++g) R[g] = 0.0;
    if (b->wallc[lp])
      for (int64_t m = 0; m < b->n_markers; ++m)
        for (int g = 0; g < 6; ++g) R[g] += b->wallc[lp][6 * m + g] * loads[m * nf + i];
  }

  /* reverse exchange: I send to my providers, receive from my requesters */
  const int64_t ns = r->fetch_off[(int64_t) w->n_fetch_peers * nf];
  const int64_t nr = r->serve_off[(int64_t) w->n_serve_peers * nf];
  ensure_staging(c, ns, nr);
  int64_t at = 0;
  for (int p = 0; p < w->n_serve_peers; ++p)
  {
    const int64_t len = r->serve_off[(int64_t) (p + 1) * nf] - r->serve_off[(int64_t) p * nf];
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
      const int s = w->fetch_segment[(size_t) p * w->n_loc + r->field_band[i]];
      if (s < 0) continue;
      const int64_t len = seg_rev(B->src_nonhalo[s], B->src_halo[s], &r->fields[i]);
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

/**
 * @brief Clears the declared ghost layers of every block of a field's location
 *        on this rank, ahead of a halo_transpose spread.
 *
 * A ghost element whose address IS an owned element of another block -- a
 * solver keeping two blocks in one array -- is left alone: it is the owned
 * sample itself. Every other ghost element is zeroed, so the solver's single
 * reverse accumulation adds exactly this spread's loads and nothing the array
 * held before.
 */
static void zero_ghost_layers(const MFTL_SolverGrid *g, const MFTL_Domain *d, int location, const MFTL_SolverField *F)
{
  const cpl_location *L = &g->loc[location];
  if (!cpl_is_lattice(L)) return;
  for (int pc = 0; pc < L->n_pieces; ++pc)
  {
    const cpl_block *blk = &g->blocks[L->piece_ref[pc]];
    if (blk->ghost[0] == 0 && blk->ghost[1] == 0 && blk->ghost[2] == 0) continue;
    for (int64_t k3 = blk->lo[2] - blk->ghost[2]; k3 < blk->hi[2] + blk->ghost[2]; ++k3)
      for (int64_t j3 = blk->lo[1] - blk->ghost[1]; j3 < blk->hi[1] + blk->ghost[1]; ++j3)
        for (int64_t i3 = blk->lo[0] - blk->ghost[0]; i3 < blk->hi[0] + blk->ghost[0]; ++i3)
        {
          const int64_t idx[3] = {i3, j3, k3};
          if (i3 >= blk->lo[0] && i3 < blk->hi[0] && j3 >= blk->lo[1] && j3 < blk->hi[1] && k3 >= blk->lo[2] && k3 < blk->hi[2]) continue;
          double *p = field_at(F, pc, cpl_block_elem(blk, idx));
          /* the wrapped index, if it exists, and whether this rank owns it at this very address */
          bool real = true;
          int64_t j[3];
          for (int k = 0; k < 3; ++k)
          {
            j[k] = d->is_periodic[k] ? cpl_mod(idx[k], L->dims[k]) : idx[k];
            if (j[k] < 0 || j[k] >= L->dims[k]) real = false;
          }
          bool alias = false;
          for (int q = 0; q < L->n_pieces && real && !alias; ++q)
          {
            const cpl_block *o = &g->blocks[L->piece_ref[q]];
            if (j[0] < o->lo[0] || j[0] >= o->hi[0] || j[1] < o->lo[1] || j[1] >= o->hi[1] || j[2] < o->lo[2] || j[2] >= o->hi[2]) continue;
            if (field_at(F, q, cpl_block_elem(o, j)) == p) alias = true;
          }
          if (!alias) *p = 0.0;
        }
  }
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

  /* constraint transposes: local ones and those owed to other ranks */
  uint64_t *con_local_key = MFTL_NULL;
  double *con_local_val = MFTL_NULL;
  int64_t con_local_cap = 0, con_local_vcap = 0;
  int con_n_dest = 0, con_dest_cap = 0, con_bufs_cap = 0;
  int *con_dest = MFTL_NULL;
  cpl_buf *con_bufs = MFTL_NULL;

  /* the solver completes the ghost route: clear its ghost layers first (then
   * the owned additions, then the halo loads -- that order survives aliasing) */
  for (int i = 0; i < nf; ++i)
    if (r->fields[i].halo_transpose) zero_ghost_layers(c->grid, &c->ens->domain, w->band[r->field_band[i]].location, &r->fields[i]);

  for (int i = 0; i < nf; ++i)
  {
    const cpl_band *B = &w->band[r->field_band[i]];
    const MFTL_SolverField *F = &r->fields[i];
    GROW(c->acc, c->acc_cap, MFTL_MAX(B->n_served, 1), double);
    double *acc = c->acc;
    for (int64_t x = 0; x < B->n_served; ++x) acc[x] = 0.0;
    /* within each peer's message, field i's segment starts after the earlier fields' */
    for (int dsi = 0; dsi < B->n_dst; ++dsi)
    {
      const int src = B->dst_rank[dsi]; /* ascending: the order every owner folds in */
      const int64_t len = seg_rev(B->dst_nonhalo[dsi], B->dst_halo[dsi], &r->fields[i]);
      if (src == c->rank)
      {
        const int ss = B->self_seg;
        const double *s = c->spread_scratch[i] + B->src_off[ss];
        for (int64_t x = 0; x < len; ++x) acc[B->pack_served[B->dst_off[dsi] + x]] += s[x];
        continue;
      }
      const int p = B->dst_peer[dsi];
      const int64_t off = r->serve_off[(int64_t) p * nf + i];
      for (int64_t x = 0; x < len; ++x) acc[B->pack_served[B->dst_off[dsi] + x]] += c->rbuf[off + x];
    }
    /* a constrained sample's load leaves it: P^T to its free samples, here
     * (after the routed loads) or on their owners (one NBX below) */
    int64_t n_local_con = 0;
    if (B->served_con)
    {
      const cpl_location *L = &c->grid->loc[B->location];
      for (int64_t s = 0; s < B->n_served; ++s)
      {
        const int64_t q = B->served_con[s];
        if (q < 0) continue;
        const double bc = acc[s];
        acc[s] = 0.0;
        if (F->constraint_load) F->constraint_load[q] += bc;
        for (int64_t e = L->con_ptr[q]; e < L->con_ptr[q + 1]; ++e)
        {
          const double v = L->con_coef[e] * bc;
          const int owner = c->li[B->location].con_peer[e];
          if (owner == c->rank)
          {
            GROW(con_local_key, con_local_cap, n_local_con + 1, uint64_t);
            GROW(con_local_val, con_local_vcap, n_local_con + 1, double);
            con_local_key[n_local_con] = L->con_free[e];
            con_local_val[n_local_con] = v;
            ++n_local_con;
            continue;
          }
          int at = -1;
          for (int d2 = 0; d2 < con_n_dest; ++d2)
            if (con_dest[d2] == owner) at = d2;
          if (at < 0)
          {
            GROW(con_dest, con_dest_cap, con_n_dest + 1, int);
            GROW(con_bufs, con_bufs_cap, con_n_dest + 1, cpl_buf);
            at = con_n_dest++;
            con_dest[at] = owner;
            memset(&con_bufs[at], 0, sizeof(cpl_buf));
          }
          const int32_t fi = i;
          PUT(&con_bufs[at], fi);
          PUT(&con_bufs[at], L->con_free[e]);
          PUT(&con_bufs[at], v);
        }
      }
    }
    for (int64_t s = 0; s < B->n_served; ++s)
    {
      const double v = (r->form == MFTL_SPREAD_DENSITY) ? acc[s] / B->served_measure[s] : acc[s];
      *field_at(F, B->served_piece[s], B->served_elem[s]) += v;
    }
    for (int64_t e = 0; e < n_local_con; ++e)
    {
      int pc;
      int64_t el;
      double m;
      if (MFTLCplSampleFind(c->grid, B->location, con_local_key[e], &pc, &el, &m))
        *field_at(F, pc, el) += (r->form == MFTL_SPREAD_DENSITY) ? con_local_val[e] / m : con_local_val[e];
    }
    /* halo loads into the ghost elements they were read from; a density is
     * linear in the load, so each part divides by the sample's own measure */
    if (F->halo_transpose)
      for (int64_t x = 0; x < B->n_dof; ++x)
        if (B->route[x] == CPL_ROUTE_HALO)
        {
          const double v = c->spread_scratch[i][x];
          *field_at(F, B->piece[x], B->elem[x]) += (r->form == MFTL_SPREAD_DENSITY) ? v / B->measure[x] : v;
        }
  }
  /* the constraint transposes owed to other ranks: one NBX when some field's
   * location is constrained (agreed: locations are in the signature) */
  bool any_con = false;
  for (int i = 0; i < nf; ++i)
    if (c->li[w->band[r->field_band[i]].location].constrained) any_con = true;
  if (any_con)
  {
    unsigned char **sb = (unsigned char **) MFTLCplAlloc((size_t) MFTL_MAX(con_n_dest, 1) * sizeof(unsigned char *), "constraint loads");
    size_t *sz = (size_t *) MFTLCplAlloc((size_t) MFTL_MAX(con_n_dest, 1) * sizeof(size_t), "constraint loads");
    for (int i = 0; i < con_n_dest; ++i)
    {
      sb[i] = con_bufs[i].data;
      sz[i] = con_bufs[i].n;
    }
    cpl_msg *in = MFTL_NULL;
    int n_in = 0;
    MFTLCplNbx(c->comm, CPL_TAG_CONSTRAINT, con_n_dest, con_dest, sb, sz, &in, &n_in, &c->stats.bytes_spread, &c->stats.messages);
    for (int m = 0; m < n_in; ++m) /* ascending source rank */
    {
      size_t off = 0;
      int32_t fi;
      uint64_t key;
      double v;
      while (GET(in[m].data, in[m].bytes, &off, fi) && GET(in[m].data, in[m].bytes, &off, key) && GET(in[m].data, in[m].bytes, &off, v))
      {
        if (fi < 0 || fi >= nf) continue;
        int pc;
        int64_t el;
        double meas;
        if (MFTLCplSampleFind(c->grid, w->band[r->field_band[fi]].location, key, &pc, &el, &meas))
          *field_at(&r->fields[fi], pc, el) += (r->form == MFTL_SPREAD_DENSITY) ? v / meas : v;
      }
    }
    MFTLCplMsgsFree(in, n_in);
    MFTLCplFree(in);
    MFTLCplFree(sb);
    MFTLCplFree(sz);
  }
  for (int i = 0; i < con_n_dest; ++i) MFTLCplFree(con_bufs[i].data);
  MFTLCplFree(con_bufs);
  MFTLCplFree(con_dest);
  MFTLCplFree(con_local_key);
  MFTLCplFree(con_local_val);
  /* wall reactions: one SUM when some field's location mirrors (agreed: the
   * locations are part of the signature, the policy part of their definition) */
  bool any_mirror = false;
  for (int i = 0; i < nf; ++i)
    if (c->grid->loc[w->band[r->field_band[i]].location].boundary == MFTL_BOUNDARY_MIRROR) any_mirror = true;
  if (any_mirror)
  {
    MFTLCplAllreduceSum(c->comm, c->reaction, 6 * nf);
    for (int i = 0; i < nf; ++i)
      if (r->fields[i].wall_reaction)
        for (int g = 0; g < 6; ++g) r->fields[i].wall_reaction[g] += c->reaction[6 * i + g];
  }
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
  MFTL_TransferSpec spec;
  memset(&spec, 0, sizeof spec);
  spec.direction = MFTL_TRANSFER_FETCH;
  spec.window = b ? b->win : MFTL_NULL;
  spec.n_fields = n_fields;
  spec.fields = fields;
  MFTL_TransferReady *ready = MFTL_NULL;
  int st = prepare_transfer(c, &spec, &ready, local);
  if (st != MFTL_COUPLING_OK) return st;
  MFTL_CouplingFetch *f = MFTL_NULL;
  st = MFTLCouplerFetchBegin(c, ready);
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerFetchEnd(c, &f);
  /* All interpolation preconditions were agreed in preparation. No state can
   * change during this synchronous call, so a trailing reduction is redundant. */
  if (st == MFTL_COUPLING_OK) st = MFTLCouplerInterpolateFetched(b, f, values);
  MFTLCouplerReleaseFetch(&f);
  MFTLCouplerReleaseReady(&ready);
  return st;
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
  const cpl_tensor_row *tensor = b->tensor[loc_pos] ? &b->tensor[loc_pos][marker] : MFTL_NULL;
  const bool factored = tensor && tensor->count;
  const int m = factored ? tensor->count : (int) (e - a);
  if (m > b->scratch_cap)
  {
    b->scratch_cap = MFTLGrowCapacity(b->scratch_cap, m);
    b->scratch_keys = (uint64_t *) MFTLCplRealloc(b->scratch_keys, (size_t) b->scratch_cap * sizeof(uint64_t), "row keys");
  }
  if (factored)
  {
    int64_t columns[64];
    MFTLCplTensorEntries(tensor, columns, b->scratch_weights);
    for (int i = 0; i < m; ++i) b->scratch_keys[i] = B->key[columns[i]];
    *weights = b->scratch_weights;
  }
  else
  {
    for (int i = 0; i < m; ++i) b->scratch_keys[i] = B->key[b->col[loc_pos][a + i]];
    *weights = &b->w[loc_pos][a];
  }
  *n = m;
  *keys = b->scratch_keys;
  return MFTL_SUCCESS;
}

int MFTLCouplerBindingWallCoefficients(const MFTL_CouplingBinding *b, int loc_pos, int64_t marker, double coef[6])
{
  if (b == MFTL_NULL || coef == MFTL_NULL || loc_pos < 0 || loc_pos >= b->n_loc || marker < 0 || marker >= b->n_markers) return MFTL_ERROR;
  for (int g = 0; g < 6; ++g) coef[g] = b->wallc[loc_pos] ? b->wallc[loc_pos][6 * marker + g] : 0.0;
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

/* ═══════════════════════════════════════════════════════════════════════════
 * Surface quadrature
 * ═══════════════════════════════════════════════════════════════════════════ */

int64_t MFTLCouplerQuadratureVertexCount(const MFTL_CouplingBinding *b) { return (b && b->kind == MFTL_MARKERS_QUADRATURE && b->quad) ? b->quad->n_own : 0; }

/** @brief Local checks of a quadrature conversion; the caller agrees the result. */
static int quad_check(const MFTL_Coupler *c, const MFTL_CouplingBinding *b, int ncomp, const void *in, const void *out_ptr, bool in_is_vertex)
{
  if (b == MFTL_NULL || b->c != c || b->kind != MFTL_MARKERS_QUADRATURE || b->quad == MFTL_NULL || ncomp <= 0) return MFTL_COUPLING_INVALID;
  const int64_t n_in = in_is_vertex ? b->quad->n_own : b->n_markers, n_out = in_is_vertex ? b->n_markers : b->quad->n_own;
  if ((n_in > 0 && in == MFTL_NULL) || (n_out > 0 && out_ptr == MFTL_NULL)) return MFTL_COUPLING_INVALID;
  if (!window_current(b->win) || !MFTLCplBodiesCurrent(c->ens, b->bodies, b->n_bodies)) return MFTL_COUPLING_STALE;
  return MFTL_COUPLING_OK;
}

/** @brief Identifies the map and the complete conversion protocol before any scheduled exchange. */
static uint64_t quad_signature(const MFTL_CouplingBinding *b, int ncomp, bool to_vertices, MFTL_SpreadForm form)
{
  const uint64_t id = (b && b->quad) ? b->quad->id : 0;
  const int args[3] = {ncomp, to_vertices ? 1 : 0, (int) form};
  return cpl_fnv(cpl_fnv(CPL_FNV_INIT, &id, sizeof id), args, sizeof args);
}

int MFTLCouplerQuadratureToVertices(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int ncomp, const double *Uq, double *Uv)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  const int st = MFTLCplAgreeTransfer(c->comm, quad_check(c, b, ncomp, Uq, Uv, false), quad_signature(b, ncomp, true, MFTL_SPREAD_INTEGRATED));
  if (st != MFTL_COUPLING_OK) return st;
  const cpl_quad *q = b->quad;
  double *acc = (double *) MFTLCplAlloc((size_t) MFTL_MAX(q->n_own, 1) * (size_t) (ncomp + 1) * sizeof(double), "quadrature projection");
  MFTLCplQuadAccumulate(c->comm, q, b->n_markers, b->ref, ncomp, Uq, acc, &c->stats.bytes_spread, &c->stats.messages);
  for (int64_t v = 0; v < q->n_own; ++v)
  {
    const double mass = acc[v * (ncomp + 1) + ncomp];
    for (int k = 0; k < ncomp; ++k) Uv[v * ncomp + k] = (mass > 0.0) ? acc[v * (ncomp + 1) + k] / mass : 0.0;
  }
  MFTLCplFree(acc);
  return MFTL_COUPLING_OK;
}

int MFTLCouplerQuadratureFromVertices(MFTL_Coupler *c, const MFTL_CouplingBinding *b, int ncomp, const double *Fv, MFTL_SpreadForm form, double *Fq)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = quad_check(c, b, ncomp, Fv, Fq, true);
  if (local == MFTL_COUPLING_OK && form != MFTL_SPREAD_INTEGRATED && form != MFTL_SPREAD_DENSITY) local = MFTL_COUPLING_INVALID;
  const int st = MFTLCplAgreeTransfer(c->comm, local, quad_signature(b, ncomp, false, form));
  if (st != MFTL_COUPLING_OK) return st;
  const cpl_quad *q = b->quad;
  const double *g = Fv;
  double *dens = MFTL_NULL;
  if (form == MFTL_SPREAD_INTEGRATED)
  {
    /* g = M^-1 F with the lumped mass the points themselves define */
    double *mass = (double *) MFTLCplAlloc((size_t) MFTL_MAX(q->n_own, 1) * sizeof(double), "quadrature mass");
    MFTLCplQuadAccumulate(c->comm, q, b->n_markers, b->ref, 0, MFTL_NULL, mass, &c->stats.bytes_spread, &c->stats.messages);
    dens = (double *) MFTLCplAlloc((size_t) MFTL_MAX(q->n_own, 1) * (size_t) ncomp * sizeof(double), "quadrature density");
    for (int64_t v = 0; v < q->n_own; ++v)
      for (int k = 0; k < ncomp; ++k) dens[v * ncomp + k] = (mass[v] > 0.0) ? Fv[v * ncomp + k] / mass[v] : 0.0;
    MFTLCplFree(mass);
    g = dens;
  }
  MFTLCplQuadInterpolate(c->comm, q, b->n_markers, b->ref, true, ncomp, g, Fq, &c->stats.bytes_spread, &c->stats.messages);
  MFTLCplFree(dens);
  return MFTL_COUPLING_OK;
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

int MFTLCouplerSurfaceSizing(MFTL_Coupler *c, double theta, double *sizes, double *global_min)
{
  if (c == MFTL_NULL) return MFTL_COUPLING_INVALID;
  int local = MFTL_COUPLING_OK;
  if (global_min == MFTL_NULL || !(theta > 0.0 && theta <= 1.0))
    local = MFTL_COUPLING_INVALID;
  else if (!grid_current(c))
    local = MFTL_COUPLING_STALE;
  const int st = agree(c, local);
  if (st != MFTL_COUPLING_OK) return st;
  const MFTL_Ensemble *ens = c->ens;
  const double f = 0.75 * theta * c->params.sampling_ratio;
  int *got = MFTL_NULL, gcap = 0;
  double lmin = MFTL_DBL_MAX;
  int64_t at = 0;
  double *hv = MFTL_NULL;
  int cap_hv = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    const int nl = obj->verts.n_local;
    if (nl > cap_hv)
    {
      cap_hv = nl;
      hv = (double *) MFTLCplRealloc(hv, (size_t) cap_hv * sizeof(double), "surface sizing");
    }
    for (int v = 0; v < nl; ++v) hv[v] = MFTL_DBL_MAX;
    /* every triangle of the pool: the halo is vertex-complete, so an owned vertex sees its whole fan */
    for (int t = 0; t < obj->tris.n; ++t)
    {
      int vv[3];
      MFTLGetTriangleVertices(obj, t, vv);
      if (vv[0] >= nl && vv[1] >= nl && vv[2] >= nl) continue;
      double cr[3][3], lo[3], hi[3];
      MFTLGetTriangleCorners(obj, t, cr);
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(cr[0][k], fmin(cr[1][k], cr[2][k]));
        hi[k] = fmax(cr[0][k], fmax(cr[1][k], cr[2][k]));
      }
      const double h = local_hmin(c, lo, hi, 0, MFTL_NULL, &got, &gcap);
      for (int q = 0; q < 3; ++q)
        if (vv[q] < nl && h < hv[vv[q]]) hv[vv[q]] = h;
    }
    for (int v = 0; v < nl; ++v, ++at)
    {
      const double L = (hv[v] < MFTL_DBL_MAX && hv[v] > 0.0) ? f * hv[v] : MFTL_DBL_MAX;
      if (sizes) sizes[at] = L;
      lmin = fmin(lmin, L);
    }
  }
  MFTLCplFree(hv);
  MFTLCplFree(got);
  double neg = -lmin;
  MFTLCplAllreduceMax(c->comm, &neg, 1);
  *global_min = -neg;
  return MFTL_COUPLING_OK;
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
  long sum[19] = {s.n_directory_builds, s.n_directory_kept, s.n_requests, s.n_need_coverage, s.n_windows, s.n_binds, s.n_stage_binds, s.n_stale,
                  s.n_fetches, s.n_spreads, s.bytes_discovery, s.bytes_fetch, s.bytes_spread, s.messages, s.bytes_memory, s.n_local_stage_binds, s.n_fetch_reused, s.n_candidate_builds, s.n_candidate_reuses};
  long mx[7] = {s.band_self, s.band_halo, s.band_remote, s.band_images, s.served, s.fetch_peers, s.serve_peers};
  MFTLCplAllreduceSumLong(c->comm, sum, 19);
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
  out->n_local_stage_binds = sum[15];
  out->n_fetch_reused = sum[16];
  out->n_candidate_builds = sum[17];
  out->n_candidate_reuses = sum[18];
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
