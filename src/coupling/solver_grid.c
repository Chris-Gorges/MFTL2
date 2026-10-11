/**
 * @file solver_grid.c
 * @author Christian Gorges
 * @date 2026-10-06
 * @ingroup mftl_coupling
 *
 * @brief The solver's registration: locations, owned lattice blocks, owned
 *        point sets, and where an export is complete. Every call is local.
 *
 * Two representations, because they need two kinds of coverage proof. A
 * LATTICE location is declared globally (identical tables on every rank), so a
 * requester can count the samples a box must contain and compare -- an exact
 * test needing nothing but integers. A POINT location cannot be enumerated by
 * anyone but its owner, so its owners certify regions instead. Point sets are
 * copied and sorted into Morton tiles of at most #TILE points, split further at
 * resolution changes: the tiles' cell-extent
 * boxes are the default ownership regions, and their position boxes the search
 * index a provider answers requests with.
 */

#include "coupling_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_arrayfunctions.h"
#include "mftl_constants.h"
#include "mftl_memory.h"
#include "mftl_object.h"

/** @brief Maximum points per Morton tile. */
#define TILE 64

MFTL_SolverGrid *MFTLSolverGridCreate(MPI_Comm solver_comm)
{
  MFTL_SolverGrid *g = (MFTL_SolverGrid *) MFTLCplCalloc(1, sizeof(MFTL_SolverGrid), "MFTLSolverGridCreate");
  g->comm = solver_comm;
  g->version_owned = 1;
  g->version_export = 1;
  return g;
}

static void free_points(cpl_points *p)
{
  MFTLCplFree(p->key);
  MFTLCplFree(p->pos);
  MFTLCplFree(p->measure);
  MFTLCplFree(p->h);
  MFTLCplFree(p->cell);
  MFTLCplFree(p->ext);
  MFTLCplFree(p->aidx);
  MFTLCplFree(p->orig);
  MFTLCplFree(p->tile_off);
  MFTLCplFree(p->tile_pbox);
  MFTLCplFree(p->tile_index.box);
  MFTLCplFree(p->tile_ebox);
  MFTLCplFree(p->tile_h);
  memset(p, 0, sizeof *p);
}

static void free_constraints(cpl_location *l)
{
  MFTLCplFree(l->con_key);
  MFTLCplFree(l->con_ptr);
  MFTLCplFree(l->con_free);
  MFTLCplFree(l->con_coef);
  MFTLCplFree(l->con_owner);
  MFTLCplFree(l->con_sorted);
  MFTLCplFree(l->con_order);
  l->con_key = MFTL_NULL;
  l->con_ptr = MFTL_NULL;
  l->con_free = MFTL_NULL;
  l->con_coef = MFTL_NULL;
  l->con_owner = MFTL_NULL;
  l->con_sorted = MFTL_NULL;
  l->con_order = MFTL_NULL;
  l->n_con = 0;
}

/** @brief Drops the key indices: any registration change makes them stale. */
static void free_key_index(MFTL_SolverGrid *g)
{
  if (g->kidx_n == MFTL_NULL) return;
  for (int l = 0; l < g->n_loc; ++l)
  {
    MFTLCplFree(g->kidx_key[l]);
    MFTLCplFree(g->kidx_piece[l]);
    MFTLCplFree(g->kidx_elem[l]);
    MFTLCplFree(g->kidx_meas[l]);
  }
  MFTLCplFree(g->kidx_n);
  MFTLCplFree(g->kidx_key);
  MFTLCplFree(g->kidx_piece);
  MFTLCplFree(g->kidx_elem);
  MFTLCplFree(g->kidx_meas);
  g->kidx_n = MFTL_NULL;
  g->kidx_key = MFTL_NULL;
  g->kidx_piece = MFTL_NULL;
  g->kidx_elem = MFTL_NULL;
  g->kidx_meas = MFTL_NULL;
}

static void free_coverage(cpl_location *l)
{
  MFTLCplFree(l->cov_index);
  MFTLCplFree(l->cov_boxes);
  MFTLCplFree(l->prov_boxes);
  l->cov_index = MFTL_NULL;
  l->cov_boxes = MFTL_NULL;
  l->prov_boxes = MFTL_NULL;
  l->n_cov = 0;
  l->n_prov = 0;
  for (int k = 0; k < 3; ++k) l->prov_hmin[k] = l->prov_hmax[k] = 0.0;
  l->exposure = MFTL_GRID_FULL;
}

void MFTLSolverGridReset(MFTL_SolverGrid *g)
{
  if (g == MFTL_NULL) return;
  for (int i = 0; i < g->n_blocks; ++i) MFTLCplFree(g->blocks[i].active);
  for (int i = 0; i < g->n_points; ++i) free_points(&g->points[i]);
  g->n_blocks = 0;
  g->n_points = 0;
  for (int l = 0; l < g->n_loc; ++l)
  {
    free_coverage(&g->loc[l]);
    free_constraints(&g->loc[l]);
    g->loc[l].n_pieces = 0;
  }
  free_key_index(g);
  g->version_owned++;
  g->version_export++;
}

void MFTLSolverGridDestroy(MFTL_SolverGrid **pg)
{
  if (pg == MFTL_NULL || *pg == MFTL_NULL) return;
  MFTL_SolverGrid *g = *pg;
  MFTLSolverGridReset(g);
  for (int l = 0; l < g->n_loc; ++l)
  {
    free_constraints(&g->loc[l]);
    for (int k = 0; k < 3; ++k)
    {
      MFTLCplFree(g->loc[l].coords[k]);
      MFTLCplFree(g->loc[l].faces[k]);
    }
    MFTLCplFree(g->loc[l].piece_ref);
  }
  MFTLCplFree(g->loc);
  MFTLCplFree(g->blocks);
  MFTLCplFree(g->points);
  MFTLCplFree(g);
  *pg = MFTL_NULL;
}

int MFTLSolverGridNumLocations(const MFTL_SolverGrid *g) { return g ? g->n_loc : 0; }

int MFTLSolverGridNumPieces(const MFTL_SolverGrid *g, int location)
{
  if (g == MFTL_NULL || location < 0 || location >= g->n_loc) return MFTL_ERROR;
  return g->loc[location].n_pieces;
}

static void complain(const char *what) { fprintf(stderr, "MFTL coupling: %s\n", what); }

int MFTLSolverGridAddLocation(MFTL_SolverGrid *g, const MFTL_LocationDesc *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->name == MFTL_NULL || desc->name[0] == '\0' || strlen(desc->name) >= CPL_NAME_MAX)
  {
    complain("location needs a name of 1 to 31 characters");
    return MFTL_ERROR;
  }
  const MFTL_LatticeDesc *lat = &desc->lattice;
  if (lat->kind != MFTL_LATTICE_NONE && lat->kind != MFTL_LATTICE_UNIFORM && lat->kind != MFTL_LATTICE_RECTILINEAR)
  {
    complain("unknown lattice kind");
    return MFTL_ERROR;
  }
  if (desc->boundary != MFTL_BOUNDARY_TRUNCATE && desc->boundary != MFTL_BOUNDARY_ERROR && desc->boundary != MFTL_BOUNDARY_MIRROR)
  {
    complain("unknown boundary policy");
    return MFTL_ERROR;
  }
  for (int f = 0; f < 6; ++f)
    if (desc->wall[f] != MFTL_WALL_EVEN && desc->wall[f] != MFTL_WALL_ODD)
    {
      complain("unknown wall parity");
      return MFTL_ERROR;
    }
  if (desc->anisotropic && lat->kind != MFTL_LATTICE_NONE)
  {
    complain("an anisotropic location is a point location (a lattice is axis-aligned by construction)");
    return MFTL_ERROR;
  }
  if (lat->kind != MFTL_LATTICE_NONE)
  {
    int64_t total = 1;
    for (int k = 0; k < 3; ++k)
    {
      if (lat->dims[k] <= 0)
      {
        complain("lattice dims must be positive");
        return MFTL_ERROR;
      }
      if (total > (INT64_C(1) << 52) / lat->dims[k])
      {
        complain("lattice too large for 52-bit keys");
        return MFTL_ERROR;
      }
      total *= lat->dims[k];
      if (lat->kind == MFTL_LATTICE_UNIFORM)
      {
        if (!(lat->spacing[k] > 0.0) || !isfinite(lat->spacing[k]) || !isfinite(lat->origin[k]))
        {
          complain("uniform lattice needs finite origin and positive spacing");
          return MFTL_ERROR;
        }
      }
      else
      {
        if (lat->coords[k] == MFTL_NULL || lat->faces[k] == MFTL_NULL)
        {
          complain("rectilinear lattice needs coords and faces on every axis");
          return MFTL_ERROR;
        }
        for (int64_t i = 0; i < lat->dims[k]; ++i)
        {
          const double f0 = lat->faces[k][i], f1 = lat->faces[k][i + 1], c = lat->coords[k][i];
          if (!(f1 > f0) || !(c >= f0) || !(c <= f1) || !isfinite(f0) || !isfinite(f1))
          {
            complain("rectilinear lattice: faces must ascend and enclose each coordinate");
            return MFTL_ERROR;
          }
        }
      }
    }
  }

  free_key_index(g);
  if (g->n_loc == g->cap_loc)
  {
    g->cap_loc = MFTLGrowCapacity(g->cap_loc, g->n_loc + 1);
    g->loc = (cpl_location *) MFTLCplRealloc(g->loc, (size_t) g->cap_loc * sizeof(cpl_location), "coupling locations");
  }
  cpl_location *l = &g->loc[g->n_loc];
  memset(l, 0, sizeof *l);
  snprintf(l->name, sizeof l->name, "%s", desc->name);
  l->kind = lat->kind;
  l->boundary = desc->boundary;
  for (int f = 0; f < 6; ++f) l->wall[f] = (desc->boundary == MFTL_BOUNDARY_MIRROR) ? desc->wall[f] : MFTL_WALL_EVEN;
  l->anisotropic = desc->anisotropic;
  l->exposure = MFTL_GRID_FULL;
  for (int k = 0; k < 3; ++k)
  {
    l->dims[k] = (lat->kind == MFTL_LATTICE_NONE) ? 0 : lat->dims[k];
    l->origin[k] = (lat->kind == MFTL_LATTICE_UNIFORM) ? lat->origin[k] : 0.0;
    l->spacing[k] = (lat->kind == MFTL_LATTICE_UNIFORM) ? lat->spacing[k] : 0.0;
    if (lat->kind == MFTL_LATTICE_RECTILINEAR)
    {
      l->coords[k] = (double *) MFTLCplAlloc((size_t) lat->dims[k] * sizeof(double), "rectilinear coords");
      l->faces[k] = (double *) MFTLCplAlloc((size_t) (lat->dims[k] + 1) * sizeof(double), "rectilinear faces");
      memcpy(l->coords[k], lat->coords[k], (size_t) lat->dims[k] * sizeof(double));
      memcpy(l->faces[k], lat->faces[k], (size_t) (lat->dims[k] + 1) * sizeof(double));
    }
  }
  g->version_owned++;
  return g->n_loc++;
}

static int add_piece(cpl_location *l, int ref)
{
  if (l->n_pieces == l->cap_pieces)
  {
    l->cap_pieces = MFTLGrowCapacity(l->cap_pieces, l->n_pieces + 1);
    l->piece_ref = (int *) MFTLCplRealloc(l->piece_ref, (size_t) l->cap_pieces * sizeof(int), "coupling pieces");
  }
  l->piece_ref[l->n_pieces] = ref;
  return l->n_pieces++;
}

int MFTLSolverGridAddBlock(MFTL_SolverGrid *g, const MFTL_BlockDesc *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->location < 0 || desc->location >= g->n_loc)
  {
    complain("block: invalid location");
    return MFTL_ERROR;
  }
  cpl_location *l = &g->loc[desc->location];
  if (!cpl_is_lattice(l))
  {
    complain("block: location is not a lattice");
    return MFTL_ERROR;
  }
  cpl_block b;
  memset(&b, 0, sizeof b);
  b.location = desc->location;
  int64_t vol = 1;
  for (int k = 0; k < 3; ++k)
  {
    b.lo[k] = desc->owned_lo[k];
    b.hi[k] = desc->owned_hi[k];
    if (b.lo[k] < 0 || b.hi[k] > l->dims[k] || b.lo[k] > b.hi[k])
    {
      complain("block: owned box outside the lattice");
      return MFTL_ERROR;
    }
    vol *= (b.hi[k] - b.lo[k]);
    b.alo[k] = desc->array_lo[k];
    b.adims[k] = desc->array_dims[k];
    b.ghost[k] = desc->ghost_layers[k];
    if (b.ghost[k] < 0 || b.adims[k] < 0 || b.alo[k] > b.lo[k] - b.ghost[k] || b.alo[k] + b.adims[k] < b.hi[k] + b.ghost[k])
    {
      complain("block: the array must cover the owned box and its declared ghost layers");
      return MFTL_ERROR;
    }
  }
  if (desc->stride[0] == 0 && desc->stride[1] == 0 && desc->stride[2] == 0)
  {
    b.stride[0] = 1;
    b.stride[1] = b.adims[0];
    b.stride[2] = b.adims[0] * b.adims[1];
  }
  else
    for (int k = 0; k < 3; ++k) b.stride[k] = desc->stride[k];

  /* Owned boxes of one location on one rank must not overlap: each sample is owned once. */
  for (int i = 0; i < g->n_blocks; ++i)
  {
    const cpl_block *o = &g->blocks[i];
    if (o->location != b.location) continue;
    bool overlap = true;
    for (int k = 0; k < 3; ++k)
      if (o->hi[k] <= b.lo[k] || b.hi[k] <= o->lo[k]) overlap = false;
    if (overlap && vol > 0)
    {
      complain("block: owned boxes of one location overlap on this rank");
      return MFTL_ERROR;
    }
  }
  if (desc->active != MFTL_NULL && vol > 0)
  {
    b.active = (unsigned char *) MFTLCplAlloc((size_t) vol, "block mask");
    memcpy(b.active, desc->active, (size_t) vol);
  }

  if (g->n_blocks == g->cap_blocks)
  {
    g->cap_blocks = MFTLGrowCapacity(g->cap_blocks, g->n_blocks + 1);
    g->blocks = (cpl_block *) MFTLCplRealloc(g->blocks, (size_t) g->cap_blocks * sizeof(cpl_block), "coupling blocks");
  }
  b.piece = add_piece(l, g->n_blocks);
  g->blocks[g->n_blocks++] = b;
  free_key_index(g);
  g->version_owned++;
  g->version_export++;
  return b.piece;
}

/** @brief Morton-sorts a copied point set and cuts it into tiles. */
static void tile_points(cpl_points *p)
{
  const int64_t n = p->n;
  if (n > 0)
  {
    double lo[3] = {p->pos[0], p->pos[1], p->pos[2]}, hi[3] = {lo[0], lo[1], lo[2]};
    for (int64_t i = 1; i < n; ++i)
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(lo[k], p->pos[3 * i + k]);
        hi[k] = fmax(hi[k], p->pos[3 * i + k]);
      }
    double sc[3];
    for (int k = 0; k < 3; ++k) sc[k] = (hi[k] > lo[k]) ? MFTL_MORTON_MAX / (hi[k] - lo[k]) : 0.0;
    uint64_t *code = (uint64_t *) MFTLCplAlloc((size_t) n * sizeof(uint64_t), "point morton");
    int *perm = (int *) MFTLCplAlloc((size_t) n * sizeof(int), "point morton");
    for (int64_t i = 0; i < n; ++i)
    {
      uint64_t c[3];
      for (int k = 0; k < 3; ++k) c[k] = (uint64_t) ((p->pos[3 * i + k] - lo[k]) * sc[k]);
      code[i] = MFTLMorton3D(c[0], c[1], c[2]);
      perm[i] = (int) i;
    }
    MFTLSortPairs64(code, perm, (int) n);
    MFTLCplFree(code);

    /* Apply the permutation to every array. */
    uint64_t *key = (uint64_t *) MFTLCplAlloc((size_t) n * sizeof(uint64_t), "points");
    double *pos = (double *) MFTLCplAlloc(3 * (size_t) n * sizeof(double), "points");
    double *mea = (double *) MFTLCplAlloc((size_t) n * sizeof(double), "points");
    double *h = (double *) MFTLCplAlloc(3 * (size_t) n * sizeof(double), "points");
    double *ext = (double *) MFTLCplAlloc(3 * (size_t) n * sizeof(double), "points");
    double *cell = p->cell ? (double *) MFTLCplAlloc(9 * (size_t) n * sizeof(double), "points") : MFTL_NULL;
    int64_t *aidx = (int64_t *) MFTLCplAlloc((size_t) n * sizeof(int64_t), "points");
    int64_t *orig = (int64_t *) MFTLCplAlloc((size_t) n * sizeof(int64_t), "points");
    for (int64_t i = 0; i < n; ++i)
    {
      const int64_t s = perm[i];
      key[i] = p->key[s];
      mea[i] = p->measure[s];
      aidx[i] = p->aidx[s];
      orig[i] = p->orig[s];
      for (int k = 0; k < 3; ++k)
      {
        pos[3 * i + k] = p->pos[3 * s + k];
        h[3 * i + k] = p->h[3 * s + k];
        ext[3 * i + k] = p->ext[3 * s + k];
      }
      if (cell) memcpy(&cell[9 * i], &p->cell[9 * s], 9 * sizeof(double));
    }
    MFTLCplFree(perm);
    MFTLCplFree(p->key);
    MFTLCplFree(p->pos);
    MFTLCplFree(p->measure);
    MFTLCplFree(p->h);
    MFTLCplFree(p->ext);
    MFTLCplFree(p->aidx);
    MFTLCplFree(p->orig);
    p->key = key;
    p->pos = pos;
    p->measure = mea;
    p->h = h;
    MFTLCplFree(p->cell);
    p->cell = cell;
    p->ext = ext;
    p->aidx = aidx;
    p->orig = orig;
  }

  /* Split Morton runs also at resolution jumps: an isolated coarse cell
   * must not enlarge the metadata footprint of an entire fine tile. */
  int tile_cap = 16;
  p->tile_off = MFTLCplAlloc((size_t) (tile_cap + 1) * sizeof(int64_t), "point tiles");
  p->n_tiles = 0;
  for (int64_t a = 0; a < n;)
  {
    double lo[3], hi[3];
    for (int k = 0; k < 3; ++k) lo[k] = hi[k] = p->h[3 * a + k];
    int64_t b = a + 1;
    for (; b < n && b < a + TILE; ++b)
    {
      bool split = false;
      for (int k = 0; k < 3; ++k)
        if (fmax(hi[k], p->h[3 * b + k]) > 2.0 * fmin(lo[k], p->h[3 * b + k])) split = true;
      if (split) break;
      for (int k = 0; k < 3; ++k)
      {
        lo[k] = fmin(lo[k], p->h[3 * b + k]);
        hi[k] = fmax(hi[k], p->h[3 * b + k]);
      }
    }
    if (p->n_tiles == tile_cap)
    {
      tile_cap *= 2;
      p->tile_off = MFTLCplRealloc(p->tile_off, (size_t) (tile_cap + 1) * sizeof(int64_t), "point tiles");
    }
    p->tile_off[p->n_tiles++] = a;
    a = b;
  }
  p->tile_off[p->n_tiles] = n;
  p->tile_pbox = (double *) MFTLCplAlloc(6 * (size_t) MFTL_MAX(p->n_tiles, 1) * sizeof(double), "point tiles");
  p->tile_ebox = (double *) MFTLCplAlloc(6 * (size_t) MFTL_MAX(p->n_tiles, 1) * sizeof(double), "point tiles");
  p->tile_h = (double *) MFTLCplAlloc(6 * (size_t) MFTL_MAX(p->n_tiles, 1) * sizeof(double), "point tiles");
  for (int k = 0; k < 3; ++k)
  {
    p->tile_hmin[k] = MFTL_DBL_MAX;
    p->tile_hmax[k] = 0.0;
  }
  for (int t = 0; t < p->n_tiles; ++t)
  {
    const int64_t a = p->tile_off[t], b = p->tile_off[t + 1];
    double *pb = &p->tile_pbox[6 * t], *eb = &p->tile_ebox[6 * t], *th = &p->tile_h[6 * t];
    for (int k = 0; k < 3; ++k)
    {
      pb[k] = eb[k] = th[k] = MFTL_DBL_MAX;
      pb[3 + k] = eb[3 + k] = th[3 + k] = -MFTL_DBL_MAX;
    }
    for (int64_t i = a; i < b; ++i)
      for (int k = 0; k < 3; ++k)
      {
        const double x = p->pos[3 * i + k], e = p->ext[3 * i + k], hh = p->h[3 * i + k];
        pb[k] = fmin(pb[k], x);
        pb[3 + k] = fmax(pb[3 + k], x);
        eb[k] = fmin(eb[k], x - e);
        eb[3 + k] = fmax(eb[3 + k], x + e);
        th[k] = fmin(th[k], hh);
        th[3 + k] = fmax(th[3 + k], hh);
      }
    for (int k = 0; k < 3; ++k)
    {
      p->tile_hmin[k] = fmin(p->tile_hmin[k], th[k]);
      p->tile_hmax[k] = fmax(p->tile_hmax[k], th[3 + k]);
    }
  }
  p->tile_off[p->n_tiles] = n;
  MFTLCplBoxTreeBuild(&p->tile_index, p->tile_pbox, p->n_tiles, 4, false);
}

int MFTLSolverGridAddPoints(MFTL_SolverGrid *g, const MFTL_PointSetDesc *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->location < 0 || desc->location >= g->n_loc)
  {
    complain("points: invalid location");
    return MFTL_ERROR;
  }
  cpl_location *l = &g->loc[desc->location];
  if (cpl_is_lattice(l))
  {
    complain("points: location is a lattice");
    return MFTL_ERROR;
  }
  const int64_t n = desc->n;
  if (n < 0 || n > (int64_t) 2000000000 || (n > 0 && (desc->keys == MFTL_NULL || desc->pos == MFTL_NULL || desc->measure == MFTL_NULL)))
  {
    complain("points: need keys, positions and measures");
    return MFTL_ERROR;
  }
  if (l->anisotropic)
  {
    if (n > 0 && desc->cell == MFTL_NULL)
    {
      complain("points: an anisotropic location needs a cell matrix per sample");
      return MFTL_ERROR;
    }
    for (int64_t i = 0; i < n; ++i)
    {
      double inv[9], det;
      bool fin = true;
      for (int a = 0; a < 9; ++a)
        if (!isfinite(desc->cell[9 * i + a])) fin = false;
      if (!fin || !MFTLCplCellInverse(&desc->cell[9 * i], inv, &det))
      {
        complain("points: a cell matrix is not finite or not invertible");
        return MFTL_ERROR;
      }
    }
  }
  else if (desc->h == MFTL_NULL)
    for (int k = 0; k < 3; ++k)
      if (!(desc->h_uniform[k] > 0.0) || !isfinite(desc->h_uniform[k]))
      {
        complain("points: h_uniform must be positive when h is NULL");
        return MFTL_ERROR;
      }
  for (int64_t i = 0; i < n; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      const double hh = l->anisotropic ? 1.0 : (desc->h ? desc->h[3 * i + k] : desc->h_uniform[k]);
      if (!isfinite(desc->pos[3 * i + k]) || !(hh > 0.0) || !isfinite(hh) ||
          (desc->extent && (!(desc->extent[3 * i + k] >= 0.0) || !isfinite(desc->extent[3 * i + k]))))
      {
        complain("points: non-finite position or non-positive resolution");
        return MFTL_ERROR;
      }
    }
    if (!(desc->measure[i] > 0.0) || !isfinite(desc->measure[i]))
    {
      complain("points: non-positive sampling measure");
      return MFTL_ERROR;
    }
    if (desc->array_index && desc->array_index[i] < 0)
    {
      complain("points: negative array index");
      return MFTL_ERROR;
    }
  }

  cpl_points p;
  memset(&p, 0, sizeof p);
  p.location = desc->location;
  p.n = n;
  const size_t nn = (size_t) MFTL_MAX(n, 1);
  p.key = (uint64_t *) MFTLCplAlloc(nn * sizeof(uint64_t), "points");
  p.pos = (double *) MFTLCplAlloc(3 * nn * sizeof(double), "points");
  p.measure = (double *) MFTLCplAlloc(nn * sizeof(double), "points");
  p.h = (double *) MFTLCplAlloc(3 * nn * sizeof(double), "points");
  p.ext = (double *) MFTLCplAlloc(3 * nn * sizeof(double), "points");
  p.aidx = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "points");
  p.orig = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "points");
  if (l->anisotropic) p.cell = (double *) MFTLCplAlloc(9 * nn * sizeof(double), "points");
  for (int64_t i = 0; i < n; ++i)
  {
    p.key[i] = desc->keys[i];
    p.measure[i] = desc->measure[i];
    p.aidx[i] = desc->array_index ? desc->array_index[i] : i;
    p.orig[i] = i;
    if (p.cell) memcpy(&p.cell[9 * i], &desc->cell[9 * i], 9 * sizeof(double));
    for (int k = 0; k < 3; ++k)
    {
      p.pos[3 * i + k] = desc->pos[3 * i + k];
      /* an anisotropic sample's resolution bound: its cell's extent along the axis */
      p.h[3 * i + k] = p.cell ? fabs(p.cell[9 * i + 3 * k]) + fabs(p.cell[9 * i + 3 * k + 1]) + fabs(p.cell[9 * i + 3 * k + 2])
                              : (desc->h ? desc->h[3 * i + k] : desc->h_uniform[k]);
      p.ext[3 * i + k] = desc->extent ? desc->extent[3 * i + k] : p.h[3 * i + k];
    }
  }
  tile_points(&p);

  if (g->n_points == g->cap_points)
  {
    g->cap_points = MFTLGrowCapacity(g->cap_points, g->n_points + 1);
    g->points = (cpl_points *) MFTLCplRealloc(g->points, (size_t) g->cap_points * sizeof(cpl_points), "coupling point sets");
  }
  p.piece = add_piece(l, g->n_points);
  g->points[g->n_points++] = p;
  free_key_index(g);
  g->version_owned++;
  g->version_export++;
  return p.piece;
}

int MFTLSolverGridSetCoverage(MFTL_SolverGrid *g, const MFTL_CoverageDesc *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->location < 0 || desc->location >= g->n_loc || desc->n_boxes < 0 ||
      (desc->exposure != MFTL_GRID_FULL && desc->exposure != MFTL_GRID_SELECTED))
  {
    complain("coverage: invalid description");
    return MFTL_ERROR;
  }
  cpl_location *l = &g->loc[desc->location];
  const int n = (desc->exposure == MFTL_GRID_SELECTED) ? desc->n_boxes : 0;
  if (n > 0 && ((cpl_is_lattice(l) && desc->index_boxes == MFTL_NULL) || (!cpl_is_lattice(l) && desc->boxes == MFTL_NULL)))
  {
    complain("coverage: boxes missing");
    return MFTL_ERROR;
  }
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k)
    {
      const bool bad = cpl_is_lattice(l) ? (desc->index_boxes[6 * i + k] > desc->index_boxes[6 * i + 3 + k])
                                         : !(desc->boxes[6 * i + k] <= desc->boxes[6 * i + 3 + k]) || !isfinite(desc->boxes[6 * i + k]) ||
                                               !isfinite(desc->boxes[6 * i + 3 + k]);
      if (bad)
      {
        complain("coverage: a box with lo > hi");
        return MFTL_ERROR;
      }
    }
  MFTLCplFree(l->cov_index);
  MFTLCplFree(l->cov_boxes);
  l->cov_index = MFTL_NULL;
  l->cov_boxes = MFTL_NULL;
  l->exposure = desc->exposure;
  l->n_cov = n;
  if (n > 0)
  {
    if (cpl_is_lattice(l))
    {
      l->cov_index = (int64_t *) MFTLCplAlloc(6 * (size_t) n * sizeof(int64_t), "coverage");
      memcpy(l->cov_index, desc->index_boxes, 6 * (size_t) n * sizeof(int64_t));
    }
    else
    {
      l->cov_boxes = (double *) MFTLCplAlloc(6 * (size_t) n * sizeof(double), "coverage");
      memcpy(l->cov_boxes, desc->boxes, 6 * (size_t) n * sizeof(double));
    }
  }
  g->version_export++;
  return MFTL_SUCCESS;
}

static int cmp_u64_idx(const void *a, const void *b)
{
  const uint64_t x = ((const uint64_t *) a)[0], y = ((const uint64_t *) b)[0];
  return (x > y) - (x < y);
}

int MFTLSolverGridSetConstraints(MFTL_SolverGrid *g, const MFTL_ConstraintDesc *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->location < 0 || desc->location >= g->n_loc || desc->n < 0 ||
      (desc->n > 0 && (desc->keys == MFTL_NULL || desc->row_ptr == MFTL_NULL)))
  {
    complain("constraints: invalid description");
    return MFTL_ERROR;
  }
  const int64_t n = desc->n, ne = (n > 0) ? desc->row_ptr[n] : 0;
  if (n > 0 && (desc->row_ptr[0] != 0 || ne < 0 || (ne > 0 && (desc->free_keys == MFTL_NULL || desc->coef == MFTL_NULL || desc->free_owner == MFTL_NULL))))
  {
    complain("constraints: rows need free keys, coefficients and owners");
    return MFTL_ERROR;
  }
  for (int64_t c = 0; c < n; ++c)
    if (desc->row_ptr[c + 1] < desc->row_ptr[c])
    {
      complain("constraints: row pointers must ascend");
      return MFTL_ERROR;
    }
  for (int64_t e = 0; e < ne; ++e)
    if (!isfinite(desc->coef[e]) || desc->free_owner[e] < 0)
    {
      complain("constraints: a coefficient is not finite or an owner is negative");
      return MFTL_ERROR;
    }
  cpl_location *l = &g->loc[desc->location];
  free_constraints(l);
  l->n_con = n;
  const size_t nn = (size_t) MFTL_MAX(n, 1), nne = (size_t) MFTL_MAX(ne, 1);
  l->con_key = (uint64_t *) MFTLCplAlloc(nn * sizeof(uint64_t), "constraints");
  l->con_ptr = (int64_t *) MFTLCplAlloc((nn + 1) * sizeof(int64_t), "constraints");
  l->con_free = (uint64_t *) MFTLCplAlloc(nne * sizeof(uint64_t), "constraints");
  l->con_coef = (double *) MFTLCplAlloc(nne * sizeof(double), "constraints");
  l->con_owner = (int *) MFTLCplAlloc(nne * sizeof(int), "constraints");
  l->con_sorted = (uint64_t *) MFTLCplAlloc(nn * sizeof(uint64_t), "constraints");
  l->con_order = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "constraints");
  l->con_ptr[0] = 0;
  if (n > 0)
  {
    memcpy(l->con_key, desc->keys, (size_t) n * sizeof(uint64_t));
    memcpy(l->con_ptr, desc->row_ptr, (size_t) (n + 1) * sizeof(int64_t));
  }
  if (ne > 0)
  {
    memcpy(l->con_free, desc->free_keys, (size_t) ne * sizeof(uint64_t));
    memcpy(l->con_coef, desc->coef, (size_t) ne * sizeof(double));
    memcpy(l->con_owner, desc->free_owner, (size_t) ne * sizeof(int));
  }
  /* the sorted index: (key, registration index) pairs */
  uint64_t *pairs = (uint64_t *) MFTLCplAlloc(2 * nn * sizeof(uint64_t), "constraints");
  for (int64_t c = 0; c < n; ++c)
  {
    pairs[2 * c] = l->con_key[c];
    pairs[2 * c + 1] = (uint64_t) c;
  }
  if (n > 1) qsort(pairs, (size_t) n, 2 * sizeof(uint64_t), cmp_u64_idx);
  for (int64_t c = 0; c < n; ++c)
  {
    l->con_sorted[c] = pairs[2 * c];
    l->con_order[c] = (int64_t) pairs[2 * c + 1];
  }
  MFTLCplFree(pairs);
  /* Windows cache constraint row indices and transfer routing. A map change
   * invalidates those caches even when sample ownership/export is unchanged;
   * UpdateGrid still retains the directory if its entries match. */
  g->version_owned++;
  g->version_export++;
  return MFTL_SUCCESS;
}

int64_t MFTLCplConstraintFind(const cpl_location *l, uint64_t key)
{
  int64_t lo = 0, hi = l->n_con;
  while (lo < hi)
  {
    const int64_t m = lo + (hi - lo) / 2;
    if (l->con_sorted[m] < key)
      lo = m + 1;
    else
      hi = m;
  }
  return (lo < l->n_con && l->con_sorted[lo] == key) ? l->con_order[lo] : -1;
}

/** @brief One point sample in a key index under construction. */
typedef struct
{
  uint64_t key;
  int piece;
  int64_t elem;
  double meas;
} kidx_t;

static int cmp_kidx(const void *a, const void *b)
{
  const uint64_t x = ((const kidx_t *) a)->key, y = ((const kidx_t *) b)->key;
  return (x > y) - (x < y);
}

bool MFTLCplSampleFind(MFTL_SolverGrid *g, int location, uint64_t key, int *piece, int64_t *elem, double *measure)
{
  const cpl_location *l = &g->loc[location];
  if (cpl_is_lattice(l))
  {
    if (key >= (uint64_t) (l->dims[0] * l->dims[1] * l->dims[2])) return false;
    int64_t idx[3];
    cpl_lat_unkey(l, key, idx);
    for (int pc = 0; pc < l->n_pieces; ++pc)
    {
      const cpl_block *b = &g->blocks[l->piece_ref[pc]];
      if (idx[0] < b->lo[0] || idx[0] >= b->hi[0] || idx[1] < b->lo[1] || idx[1] >= b->hi[1] || idx[2] < b->lo[2] || idx[2] >= b->hi[2]) continue;
      if (b->active)
      {
        const int64_t nx = b->hi[0] - b->lo[0], ny = b->hi[1] - b->lo[1];
        if (!b->active[(idx[0] - b->lo[0]) + nx * ((idx[1] - b->lo[1]) + ny * (idx[2] - b->lo[2]))]) return false;
      }
      *piece = pc;
      *elem = cpl_block_elem(b, idx);
      *measure = MFTLCplLatWidth(l, 0, idx[0]) * MFTLCplLatWidth(l, 1, idx[1]) * MFTLCplLatWidth(l, 2, idx[2]);
      return true;
    }
    return false;
  }
  if (g->kidx_n == MFTL_NULL)
  {
    const size_t nl = (size_t) MFTL_MAX(g->n_loc, 1);
    g->kidx_n = (int64_t *) MFTLCplAlloc(nl * sizeof(int64_t), "key index");
    g->kidx_key = (uint64_t **) MFTLCplCalloc(nl, sizeof(uint64_t *), "key index");
    g->kidx_piece = (int **) MFTLCplCalloc(nl, sizeof(int *), "key index");
    g->kidx_elem = (int64_t **) MFTLCplCalloc(nl, sizeof(int64_t *), "key index");
    g->kidx_meas = (double **) MFTLCplCalloc(nl, sizeof(double *), "key index");
    for (int i = 0; i < g->n_loc; ++i) g->kidx_n[i] = -1;
  }
  if (g->kidx_n[location] < 0)
  {
    int64_t total = 0;
    for (int pc = 0; pc < l->n_pieces; ++pc) total += g->points[l->piece_ref[pc]].n;
    kidx_t *t = (kidx_t *) MFTLCplAlloc((size_t) MFTL_MAX(total, 1) * sizeof(kidx_t), "key index");
    int64_t at = 0;
    for (int pc = 0; pc < l->n_pieces; ++pc)
    {
      const cpl_points *p = &g->points[l->piece_ref[pc]];
      for (int64_t i = 0; i < p->n; ++i, ++at)
      {
        t[at].key = p->key[i];
        t[at].piece = pc;
        t[at].elem = p->aidx[i];
        t[at].meas = p->measure[i];
      }
    }
    if (total > 1) qsort(t, (size_t) total, sizeof(kidx_t), cmp_kidx);
    const size_t nn = (size_t) MFTL_MAX(total, 1);
    g->kidx_key[location] = (uint64_t *) MFTLCplAlloc(nn * sizeof(uint64_t), "key index");
    g->kidx_piece[location] = (int *) MFTLCplAlloc(nn * sizeof(int), "key index");
    g->kidx_elem[location] = (int64_t *) MFTLCplAlloc(nn * sizeof(int64_t), "key index");
    g->kidx_meas[location] = (double *) MFTLCplAlloc(nn * sizeof(double), "key index");
    for (int64_t i = 0; i < total; ++i)
    {
      g->kidx_key[location][i] = t[i].key;
      g->kidx_piece[location][i] = t[i].piece;
      g->kidx_elem[location][i] = t[i].elem;
      g->kidx_meas[location][i] = t[i].meas;
    }
    MFTLCplFree(t);
    g->kidx_n[location] = total;
  }
  int64_t lo = 0, hi = g->kidx_n[location];
  const uint64_t *k = g->kidx_key[location];
  while (lo < hi)
  {
    const int64_t m = lo + (hi - lo) / 2;
    if (k[m] < key)
      lo = m + 1;
    else
      hi = m;
  }
  if (lo >= g->kidx_n[location] || k[lo] != key) return false;
  *piece = g->kidx_piece[location][lo];
  *elem = g->kidx_elem[location][lo];
  *measure = g->kidx_meas[location][lo];
  return true;
}

int MFTLSolverGridSetProviderRegions(MFTL_SolverGrid *g, const MFTL_ProviderRegions *desc)
{
  if (g == MFTL_NULL || desc == MFTL_NULL || desc->location < 0 || desc->location >= g->n_loc || desc->n_boxes < 0 ||
      (desc->n_boxes > 0 && desc->boxes == MFTL_NULL))
  {
    complain("provider regions: invalid description");
    return MFTL_ERROR;
  }
  cpl_location *l = &g->loc[desc->location];
  if (cpl_is_lattice(l))
  {
    complain("provider regions: a lattice location's ownership is its blocks");
    return MFTL_ERROR;
  }
  for (int i = 0; i < desc->n_boxes; ++i)
    for (int k = 0; k < 3; ++k)
      if (!(desc->boxes[6 * i + k] <= desc->boxes[6 * i + 3 + k]) || !isfinite(desc->boxes[6 * i + k]) || !isfinite(desc->boxes[6 * i + 3 + k]))
      {
        complain("provider regions: a box with lo > hi");
        return MFTL_ERROR;
      }
  for (int k = 0; k < 3; ++k)
    if (desc->h_min[k] < 0.0 || desc->h_max[k] < 0.0 || !isfinite(desc->h_min[k]) || !isfinite(desc->h_max[k]) ||
        (desc->h_min[k] > 0.0 && desc->h_max[k] > 0.0 && desc->h_min[k] > desc->h_max[k]))
    {
      complain("provider regions: invalid resolution bounds");
      return MFTL_ERROR;
    }
  for (int k = 0; k < 3; ++k)
  {
    l->prov_hmin[k] = desc->h_min[k];
    l->prov_hmax[k] = desc->h_max[k];
  }
  MFTLCplFree(l->prov_boxes);
  l->prov_boxes = MFTL_NULL;
  l->n_prov = desc->n_boxes;
  if (l->n_prov > 0)
  {
    l->prov_boxes = (double *) MFTLCplAlloc(6 * (size_t) l->n_prov * sizeof(double), "provider regions");
    memcpy(l->prov_boxes, desc->boxes, 6 * (size_t) l->n_prov * sizeof(double));
  }
  g->version_owned++;
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Validation against the domain, and the cross-rank definition hash
 * ═══════════════════════════════════════════════════════════════════════════ */

uint64_t MFTLCplGridDefinitionHash(const MFTL_SolverGrid *g)
{
  uint64_t h = CPL_FNV_INIT;
  h = cpl_fnv(h, &g->n_loc, sizeof g->n_loc);
  for (int i = 0; i < g->n_loc; ++i)
  {
    const cpl_location *l = &g->loc[i];
    h = cpl_fnv(h, l->name, strlen(l->name));
    const int kind = (int) l->kind, bnd = (int) l->boundary;
    h = cpl_fnv(h, &kind, sizeof kind);
    h = cpl_fnv(h, &bnd, sizeof bnd);
    h = cpl_fnv(h, l->wall, sizeof l->wall);
    const int an = l->anisotropic ? 1 : 0;
    h = cpl_fnv(h, &an, sizeof an);
    h = cpl_fnv(h, l->dims, sizeof l->dims);
    h = cpl_fnv(h, l->origin, sizeof l->origin);
    h = cpl_fnv(h, l->spacing, sizeof l->spacing);
    if (l->kind == MFTL_LATTICE_RECTILINEAR)
      for (int k = 0; k < 3; ++k)
      {
        h = cpl_fnv(h, l->coords[k], (size_t) l->dims[k] * sizeof(double));
        h = cpl_fnv(h, l->faces[k], (size_t) (l->dims[k] + 1) * sizeof(double));
      }
  }
  return h;
}

/** @brief Is @p x inside the domain along axis @p k (half-open on a periodic axis)? */
static bool inside_axis(const MFTL_Domain *d, int k, double x)
{
  if (d->is_periodic[k]) return x >= d->global_min[k] && x < d->global_max[k];
  return x >= d->global_min[k] && x <= d->global_max[k];
}

int MFTLCplGridValidate(const MFTL_SolverGrid *g, const MFTL_Domain *d)
{
  for (int i = 0; i < g->n_loc; ++i)
  {
    const cpl_location *l = &g->loc[i];
    if (!cpl_is_lattice(l)) continue;
    for (int k = 0; k < 3; ++k)
    {
      const double first = MFTLCplLatCoord(l, k, 0), last = MFTLCplLatCoord(l, k, l->dims[k] - 1);
      if (!inside_axis(d, k, first) || !inside_axis(d, k, last))
      {
        fprintf(stderr, "MFTL coupling: location '%s' axis %d: lattice samples outside the domain\n", l->name, k);
        return MFTL_ERROR;
      }
      if (d->is_periodic[k])
      {
        const double span = (l->kind == MFTL_LATTICE_UNIFORM) ? (double) l->dims[k] * l->spacing[k] : l->faces[k][l->dims[k]] - l->faces[k][0];
        if (fabs(span - d->domain_size[k]) > 1e-9 * d->domain_size[k])
        {
          fprintf(stderr, "MFTL coupling: location '%s' axis %d is periodic but the lattice spans %.17g, not the period %.17g\n", l->name, k, span,
                  d->domain_size[k]);
          return MFTL_ERROR;
        }
      }
    }
  }
  for (int i = 0; i < g->n_points; ++i)
  {
    const cpl_points *p = &g->points[i];
    for (int64_t j = 0; j < p->n; ++j)
      for (int k = 0; k < 3; ++k)
        if (!inside_axis(d, k, p->pos[3 * j + k]))
        {
          fprintf(stderr, "MFTL coupling: location '%s': point key %llu outside the domain on axis %d (%.17g)\n", g->loc[p->location].name,
                  (unsigned long long) p->key[j], k, p->pos[3 * j + k]);
          return MFTL_ERROR;
        }
  }
  /* Keys unique per location on this rank (across ranks: detected when two
   * owners answer one requester). */
  for (int i = 0; i < g->n_loc; ++i)
  {
    const cpl_location *l = &g->loc[i];
    if (cpl_is_lattice(l)) continue;
    int64_t total = 0;
    for (int pc = 0; pc < l->n_pieces; ++pc) total += g->points[l->piece_ref[pc]].n;
    if (total < 2) continue;
    uint64_t *k = (uint64_t *) MFTLCplAlloc((size_t) total * sizeof(uint64_t), "key check");
    uint64_t *tmp = (uint64_t *) MFTLCplAlloc((size_t) total * sizeof(uint64_t), "key check");
    int64_t m = 0;
    for (int pc = 0; pc < l->n_pieces; ++pc)
    {
      const cpl_points *p = &g->points[l->piece_ref[pc]];
      for (int64_t j = 0; j < p->n; ++j) k[m++] = p->key[j];
    }
    MFTLRadixSort64(k, tmp, (int) total);
    int dup = 0;
    for (int64_t j = 1; j < total; ++j)
      if (k[j] == k[j - 1]) dup = 1;
    MFTLCplFree(k);
    MFTLCplFree(tmp);
    if (dup)
    {
      fprintf(stderr, "MFTL coupling: location '%s': duplicate point keys on one rank\n", l->name);
      return MFTL_ERROR;
    }
  }
  /* A SELECTED point location must say where it owns samples and, before its
   * first export, how far its samples' kernels can reach. Another provider's
   * resolution cannot supply a bound for this provider's unseen samples. */
  for (int i = 0; i < g->n_loc; ++i)
  {
    const cpl_location *l = &g->loc[i];
    if (!cpl_is_lattice(l) && l->exposure == MFTL_GRID_SELECTED && l->n_prov == 0 && l->n_pieces > 0)
    {
      fprintf(stderr, "MFTL coupling: location '%s' is SELECTED but has no provider regions\n", l->name);
      return MFTL_ERROR;
    }
    if (!cpl_is_lattice(l) && l->exposure == MFTL_GRID_SELECTED && l->n_prov > 0)
    {
      bool has_samples = false;
      for (int pc = 0; pc < l->n_pieces; ++pc)
        if (g->points[l->piece_ref[pc]].n > 0) has_samples = true;
      if (!has_samples)
        for (int k = 0; k < 3; ++k)
          if (!(l->prov_hmin[k] > 0.0 && l->prov_hmax[k] > 0.0))
          {
            fprintf(stderr, "MFTL coupling: empty SELECTED provider of '%s' must declare resolution bounds on every axis\n", l->name);
            return MFTL_ERROR;
          }
    }
  }
  return MFTL_SUCCESS;
}

bool MFTLCplLatExported(const cpl_location *l, const int64_t idx[3])
{
  if (l->exposure == MFTL_GRID_FULL) return true;
  for (int c = 0; c < l->n_cov; ++c)
  {
    const int64_t *b = &l->cov_index[6 * c];
    if (idx[0] >= b[0] && idx[0] < b[3] && idx[1] >= b[1] && idx[1] < b[4] && idx[2] >= b[2] && idx[2] < b[5]) return true;
  }
  return false;
}
