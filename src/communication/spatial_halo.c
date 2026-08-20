/**
 * @file spatial_halo.c
 * @author Christian Gorges
 * @date 2026-08-16
 * @ingroup mftl_communication
 *
 * @brief The spatial halo: what of body B is near this rank's share of body A.
 *
 * The design argument -- why the arrivals get their own shadow object instead of
 * being appended to body B's ghost suffix -- is in mftl_communication.h §6 and
 * is not repeated here. What follows is how it is done.
 *
 * ## SHAPE OF THE PASS
 *
 *   0. index the request list BY BODY          O(bodies + requests)
 *   1. publish my extent of every body         MPI_Ineighbor_allgather  <-- async
 *   2. per-triangle boxes of every B body      overlapped with 1
 *   3. classify + pack                         one send buffer per destination
 *   4. exchange vertices and triangles         one MFTLExchangeStart, 2 payloads
 *   5. rebuild one shadow object per body      MFTLBuildTopologyFromIndexList
 *
 * Working per BODY rather than per PAIR (step 0) is what keeps a triangle that
 * is relevant to two different A bodies on the same destination from being
 * packed twice -- and therefore from being reported twice as a contact.
 */

#include "mftl_communication.h"

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "comm_internal.h"
#include "mftl_constants.h"
#include "mftl_ensemble.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"
#include "mftl_meshgen.h"
#include "mftl_object.h"
#include "mftl_topology.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. Small helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Sets a box to the empty state, i.e. the identity of a min/max merge.
 * @param[out] lo Receives +DBL_MAX on every axis.
 * @param[out] hi Receives -DBL_MAX on every axis.
 */
static void box_clear(double *lo, double *hi)
{
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = MFTL_DBL_MAX;
    hi[k] = -MFTL_DBL_MAX;
  }
}

/**
 * @brief Has anything been merged into this box?
 * @param lo Lower corner.
 * @param hi Upper corner.
 * @return `true` if the box is still in the state box_clear() left it in.
 */
static inline bool box_is_empty(const double *lo, const double *hi) { return lo[0] > hi[0]; }

/* ═══════════════════════════════════════════════════════════════════════════
 * EXTENTS ARE CENTRE + HALF-WIDTH, NOT MIN + MAX
 *
 * Everything geometric in this file is a question of the form "is this near
 * that", asked across a domain that wraps. A (lo, hi) box cannot express the
 * answer, for two reasons that look unrelated and are the same one:
 *
 *   IT MISSES CONTACTS. Two shares meeting across the seam are a domain length
 *   apart in stored coordinates -- rank p says "my part of A is at x ~ 9.9",
 *   rank q says "my part of B is at x ~ 0.1" -- and min/max comparison reports
 *   no overlap for two surfaces that are touching. The traversal tries every
 *   periodic image of B, but only for triangles it HAS, and this test is what
 *   decides whether they are sent at all. The miss therefore happens one stage
 *   before anything the BVH can see, and nobody reports the contact.
 *
 *   IT OVER-SENDS ENORMOUSLY. A share containing even one seam-straddling
 *   triangle has a min/max box spanning the WHOLE DOMAIN, because
 *   MFTLGetTriangleBounds is seam-consistent and correctly puts that triangle's
 *   corners just outside the box on one side. Such a share then overlaps
 *   everything, and every neighbour ships its entire share of every partner
 *   body. A droplet sitting on the seam -- routine in a periodic run -- turns
 *   the halo into an all-to-all broadcast of the mesh, every timestep, silently.
 *
 * Both disappear when an extent is stored as a CENTRE ON THE CIRCLE plus a
 * half-width. "Near" becomes the minimum-image distance between centres against
 * the sum of half-widths, which is the same predicate the rest of the library
 * already uses for every other periodic question (MFTLGetMinimumImageVector).
 * The wire payload is unchanged at six doubles per body per rank, and a
 * non-periodic axis reduces to exactly the old comparison.
 *
 * The representation degrades gracefully rather than failing: a share genuinely
 * wider than half the domain gets a half-width above L/2, the minimum-image test
 * then accepts everything, and the result is the old conservative behaviour --
 * extra traffic, never a missed contact.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Empty marker for an interval: no owned triangles contributed to it.
 *
 * @param h Per-axis half-widths.
 * @return `true` if the interval carries nothing (a negative half-width).
 */
static inline bool interval_is_empty(const double *h) { return h[0] < 0.0; }

/**
 * @brief Are two circular intervals within reach of each other?
 *
 * @brief Minimum-image proximity of two centre/half-width extents.
 * @param domain Supplies the periodicity and the domain sizes.
 * @param ca     Centre of the first extent.
 * @param ha     Half-widths of the first extent.
 * @param cb     Centre of the second extent.
 * @param hb     Half-widths of the second extent.
 * @return `true` if the two extents are within `ha + hb` on every axis, measured
 *         as a MINIMUM-IMAGE distance between the CENTRES.
 */
static inline bool interval_overlap(const MFTL_Domain *domain, const double *ca, const double *ha, const double *cb, const double *hb)
{
  for (int k = 0; k < 3; ++k)
  {
    double delta = ca[k] - cb[k];
    if (domain->is_periodic[k]) delta -= domain->domain_size[k] * nearbyint(delta * domain->inv_domain_size[k]);
    if (delta < 0.0) delta = -delta;
    if (delta > ha[k] + hb[k]) return MFTL_FALSE;
  }
  return MFTL_TRUE;
}

/**
 * @brief Wraps a scalar back into [global_min, global_max) on a periodic axis.
 *
 * @param domain Supplies the bounds and sizes.
 * @param k      Axis, 0..2.
 * @param v      A coordinate.
 * @return @p v brought into `[global_min[k], global_max[k])`.
 */
static inline double wrap_axis(const MFTL_Domain *domain, int k, double v)
{
  if (!domain->is_periodic[k] || domain->domain_size[k] <= 0.0) return v;
  if (v >= domain->global_max[k]) return v - domain->domain_size[k];
  if (v < domain->global_min[k]) return v + domain->domain_size[k];
  return v;
}

/**
 * @brief Bounds of the OWNED triangles of @p obj.
 *
 * Owned only, on both sides of the test: a rank's share of a body is what it
 * owns, and counting its topological ghosts would inflate the extent by one
 * halo layer in every direction for no gain -- the neighbour that owns those
 * ghosts is publishing them itself.
 *
 * @param      obj   The mesh.
 * @param[out] lo    Lower corner over the OWNED triangles.
 * @param[out] hi    Upper corner.
 * @param      shift Optional per-axis offset applied before measuring, used to
 *                   take the second (half-domain shifted) frame. May be NULL.
 */
static void owned_bounds(const MFTL_Object *obj, double *lo, double *hi, const double *shift)
{
  box_clear(lo, hi);
  if (obj == MFTL_NULL) return;

  const MFTL_Domain *const domain = obj->domain;
  const int nt = obj->tris.n_local;

  for (int t = 0; t < nt; ++t)
  {
    double tlo[3], thi[3];
    MFTLGetTriangleBounds(obj, t, tlo, thi);

    if (shift != MFTL_NULL && domain != MFTL_NULL)
    {
      for (int k = 0; k < 3; ++k)
      {
        if (!domain->is_periodic[k] || domain->domain_size[k] <= 0.0) continue;

        double moved = tlo[k] + shift[k];
        if (moved >= domain->global_max[k])
          moved -= domain->domain_size[k];
        else if (moved < domain->global_min[k])
          moved += domain->domain_size[k];

        const double d = moved - tlo[k];  // what the wrap actually applied
        tlo[k] = moved;
        thi[k] += d;
      }
    }

    for (int k = 0; k < 3; ++k)
    {
      if (tlo[k] < lo[k]) lo[k] = tlo[k];
      if (thi[k] > hi[k]) hi[k] = thi[k];
    }
  }
}

/**
 * @brief The owned extent of @p obj as a circular interval per axis.
 *
 * Measured in TWO FRAMES and the tighter one kept per axis: the raw stored
 * coordinates, and a frame displaced by half a domain. A body straddling the
 * seam is contiguous in the shifted frame; one straddling the middle is
 * contiguous in the raw frame; no body can straddle both, provided its true
 * extent is under L/2 -- the same condition the minimum-image convention assumes
 * everywhere else in the library.
 *
 * That is what turns a seam-straddling share from "spans the whole domain" into
 * its actual size, and it is why the answer is a centre rather than a box: the
 * centre of the tighter frame has to be mapped back through the shift, and a
 * min/max pair cannot survive that mapping.
 *
 * @param      obj    The mesh.
 * @param      domain Supplies the periodicity and the domain sizes.
 * @param[out] c      Per-axis centre.
 * @param[out] h      Per-axis half-width. Negative on every axis when this rank
 *                    owns nothing of the body.
 */
static void owned_interval(const MFTL_Object *obj, const MFTL_Domain *domain, double *c, double *h)
{
  double lo[3], hi[3];
  owned_bounds(obj, lo, hi, MFTL_NULL);

  if (box_is_empty(lo, hi))
  {
    for (int k = 0; k < 3; ++k)
    {
      c[k] = 0.0;
      h[k] = -1.0;
    }
    return;
  }

  double half[3];
  bool any_periodic = MFTL_FALSE;
  for (int k = 0; k < 3; ++k)
  {
    half[k] = 0.5 * domain->domain_size[k];
    if (domain->is_periodic[k] && domain->domain_size[k] > 0.0) any_periodic = MFTL_TRUE;
  }

  double slo[3], shi[3];
  if (any_periodic) owned_bounds(obj, slo, shi, half);

  for (int k = 0; k < 3; ++k)
  {
    double cc = 0.5 * (lo[k] + hi[k]);
    double hh = 0.5 * (hi[k] - lo[k]);

    if (any_periodic && domain->is_periodic[k] && domain->domain_size[k] > 0.0)
    {
      const double sh = 0.5 * (shi[k] - slo[k]);
      if (sh < hh)
      {
        hh = sh;
        cc = wrap_axis(domain, k, 0.5 * (slo[k] + shi[k]) - half[k]);  // back out of the shifted frame
      }
    }

    c[k] = wrap_axis(domain, k, cc);
    h[k] = hh;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

MFTL_SpatialHalo *MFTLSpatialHaloCreate(MFTL_Ensemble *ens)
{
  if (ens == MFTL_NULL) return MFTL_NULL;

  MFTL_SpatialHalo *const h = (MFTL_SpatialHalo *) MFTLCallocOrDie(1, sizeof(MFTL_SpatialHalo), "MFTL_SpatialHalo");
  h->num_objects = ens->num_objects;
  h->domain = &ens->domain;
  h->patch = (MFTL_Object **) MFTLCallocOrDie((size_t) MFTL_MAX(h->num_objects, 1), sizeof(MFTL_Object *), "spatial halo patches");
  h->global_radius = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(h->num_objects, 1), sizeof(double), "spatial halo radii");
  return h;
}

void MFTLSpatialHaloDestroy(MFTL_SpatialHalo **p_halo)
{
  if (p_halo == MFTL_NULL || *p_halo == MFTL_NULL) return;
  MFTL_SpatialHalo *const h = *p_halo;

  for (int b = 0; b < h->num_objects; ++b)
  {
    if (h->patch[b] != MFTL_NULL) MFTLObjectDestroy(&h->patch[b]);
  }
  free(h->patch);
  free(h->global_radius);
  free(h);
  *p_halo = MFTL_NULL;
}

int MFTLSpatialHaloResolveRadii(MFTL_Ensemble *ens, MFTL_SpatialHalo *halo)
{
  if (ens == MFTL_NULL || halo == MFTL_NULL || ens->objects == MFTL_NULL) return MFTL_ERROR;

  const int n_obj = ens->num_objects;
  if (n_obj <= 0) return MFTL_SUCCESS;
  MFTL_ASSERT(n_obj == halo->num_objects && "MFTLSpatialHaloResolveRadii: halo was created for a different object list");

  /* ── Two frames, one collective ─────────────────────────────────────────
   *
   * The raw stored extent of a body STRADDLING A PERIODIC SEAM spans the whole
   * domain: half its triangles sit at x ~ 0 and half at x ~ L, and min/max over
   * them says "this body is L wide". A 1 mm droplet at the seam would then
   * report a bounding radius of half the box, and every envelope scaled by it --
   * the halo margin, the leaf inflation, the narrow-phase gap -- would grow by
   * that same factor. Unlike the Morton bounding box, where a domain-wide root
   * is accepted locality loss, this one changes the ANSWER: contact would be
   * declared between surfaces that are nowhere near each other.
   *
   * So each body is measured twice, in the raw frame and in one displaced by
   * half a domain, and the SMALLER extent per axis wins. A body straddling the
   * seam is contiguous in the shifted frame; a body straddling the middle is
   * contiguous in the raw one; no body can straddle both, provided its true
   * extent is under L/2 -- the same condition the minimum-image convention
   * already assumes everywhere else in the library. Above L/2 the result is
   * still an upper bound, so the envelope errs wide rather than narrow.
   *
   * Cost is the reduce payload, not a second collective: 12 doubles per body
   * instead of 6, in the same MPI_MIN over the NEGATED maxima that already
   * carried both halves of every box.
   */
  const int stride = 12;  // [0..2] lo, [3..5] -hi, raw | [6..8] lo, [9..11] -hi, shifted
  double *const local = (double *) MFTLMallocOrDie((size_t) stride * (size_t) n_obj * sizeof(double), "spatial halo extents");
  double *const global = (double *) MFTLMallocOrDie((size_t) stride * (size_t) n_obj * sizeof(double), "spatial halo extents (out)");

  for (int i = 0; i < stride * n_obj; ++i) local[i] = MFTL_DBL_MAX;

  double half_domain[3];
  for (int k = 0; k < 3; ++k) half_domain[k] = 0.5 * ens->domain.domain_size[k];

  for (int b = 0; b < n_obj; ++b)
  {
    double lo[3], hi[3];

    owned_bounds(ens->objects[b], lo, hi, MFTL_NULL);
    if (box_is_empty(lo, hi)) continue;
    for (int k = 0; k < 3; ++k)
    {
      local[stride * b + k] = lo[k];
      local[stride * b + 3 + k] = -hi[k];
    }

    owned_bounds(ens->objects[b], lo, hi, half_domain);
    for (int k = 0; k < 3; ++k)
    {
      local[stride * b + 6 + k] = lo[k];
      local[stride * b + 9 + k] = -hi[k];
    }
  }

  MPI_Allreduce(local, global, stride * n_obj, MPI_DOUBLE, MPI_MIN, ens->domain.comm_world);

  for (int b = 0; b < n_obj; ++b)
  {
    if (global[stride * b] == MFTL_DBL_MAX)
    {
      halo->global_radius[b] = 0.0;  // nobody holds this body
      continue;
    }

    /* Half the LONGEST SIDE, not half the diagonal: for a sphere of radius R
     * this is R, so a caller scaling by "a thousandth of the body" gets what it
     * asked for rather than a silent sqrt(3) inflation. */
    double longest = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      double d = (-global[stride * b + 3 + k]) - global[stride * b + k];

      if (ens->domain.is_periodic[k] && ens->domain.domain_size[k] > 0.0)
      {
        const double d_shifted = (-global[stride * b + 9 + k]) - global[stride * b + 6 + k];
        if (d_shifted < d) d = d_shifted;
      }

      if (d > longest) longest = d;
    }
    halo->global_radius[b] = 0.5 * longest;
  }

  free(local);
  free(global);
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Rebuilding one shadow object from its arrivals
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Turns the arrivals for one body into a valid, self-contained MFTL_Object.
 *
 * The patch is built as if it were an ordinary little mesh -- vertices first,
 * then MFTLBuildTopologyFromIndexList to link its half-edges and twins -- and is
 * only afterwards told that none of it is owned. Two consequences are worth
 * spelling out because both are easy to get wrong:
 *
 *   THE VERTEX ROTATION. `vertex_gid[k]` on the wire is the TARGET of half-edge
 *   3t+k, while MFTLBuildTopologyFromIndexList wants the ORIGIN of 3t+k. Origin
 *   and target differ by a rotation -- origin(3t+k) == target(3t+(k+2)%3) -- so
 *   tri_verts[3t+k] = slot(vertex_gid[(k+2)%3]). Getting this wrong rotates every
 *   triangle, which flips no normal and breaks no invariant; it just silently
 *   moves every corner one step round.
 *
 *   THE BOUNDARY FLAGS ARE RESTORED, NOT DERIVED. Building the topology runs
 *   MFTLMarkMeshBoundary, which sets MFTL_FLAG_BOUNDARY on every twin-less
 *   half-edge -- and a patch is a CUT out of a larger surface, so its whole rim
 *   is twin-less without being a mesh boundary. Left alone, the patch would
 *   claim its cut edges are the edge of the surface. The transmitted flags are
 *   written over the derived ones afterwards, which is exactly why he_flags[]
 *   and the vertex flags are on the wire at all (they are the one thing a
 *   receiver cannot re-derive; see mftl_communication.h §1).
 *
 * @param halo    The halo. `halo->patch[body]` is REPLACED, not appended to.
 * @param body    Index of the body in the ensemble's object list.
 * @param verts   Received vertex records.
 * @param n_verts Number of received vertices.
 * @param tris    Received triangle records.
 * @param n_tris  Number of received triangles.
 * @param tbl     Epoch-stamped id-to-slot table, reused across bodies.
 * @param scratch Reusable int buffer for the de-duplication pass.
 */
static void rebuild_patch(MFTL_SpatialHalo *halo, int body, const MFTL_PackedVertex *verts, int n_verts, const MFTL_PackedTriangleComplex *tris, int n_tris,
                          mftl_gid_table *tbl, mftl_int_buffer *scratch)
{
  MFTL_Object *patch = halo->patch[body];

  if (n_tris <= 0 || n_verts < 3)
  {
    if (patch != MFTL_NULL) MFTLObjectDestroy(&halo->patch[body]);
    return;
  }

  if (patch == MFTL_NULL)
  {
    patch = MFTLObjectCreate(body);
    /* Borrowed, exactly as MFTLEnsembleAddObject would set it. The patch is
     * deliberately NOT added to the ensemble's object list: redistribution and
     * the ghost syncs iterate that list, and a patch must be invisible to both. */
    patch->domain = halo->domain;
    halo->patch[body] = patch;
  }

  /* ── De-duplicate the arriving vertices by global id ───────────────────────
   *
   * A vertex shared by two arriving triangles is sent once per triangle by the
   * packer's own dedup only within a single destination; across senders it can
   * still arrive twice. Collapsing them here is what makes the patch a connected
   * surface rather than a bag of disjoint triangles -- which in turn is what
   * lets MFTLBuildTopologyFromIndexList link twins across it. */
  int max_gid = 0;
  for (int i = 0; i < n_verts; ++i)
  {
    if (verts[i].global_id > max_gid) max_gid = verts[i].global_id;
  }
  gt_begin(tbl, max_gid + 1);

  ib_reset(scratch);

  int nv = 0;
  for (int i = 0; i < n_verts; ++i)
  {
    const int g = verts[i].global_id;
    if (g < 0 || gt_has(tbl, g)) continue;
    gt_put(tbl, g, nv++);
    ib_add(scratch, i);  // arrival index of the surviving copy, in slot order
  }

  if (nv < 3)
  {
    MFTLObjectDestroy(&halo->patch[body]);
    return;
  }

  /* ── Vertices ─────────────────────────────────────────────────────────── */
  MFTLObjectSetCounts(patch, nv, nv, 0, 0);

  int max_v_gid = 0;
  for (int s = 0; s < nv; ++s)
  {
    const MFTL_PackedVertex *const p = &verts[scratch->data[s]];
    double *const dst = &patch->verts.pos[3 * s];
    dst[0] = p->pos[0];
    dst[1] = p->pos[1];
    dst[2] = p->pos[2];

    MFTL_Vertex *const rec = &patch->verts.rec[s];
    rec->global_id = p->global_id;
    rec->owner_proc = p->owner_proc;
    rec->flags = p->flags;
    rec->out_he_idx = MFTL_ERROR;  // filled in by the topology build

    if (p->global_id > max_v_gid) max_v_gid = p->global_id;
  }

  /* ── Connectivity ─────────────────────────────────────────────────────── */
  int *const tri_verts = (int *) MFTLMallocOrDie(3 * (size_t) n_tris * sizeof(int), "spatial halo index list");
  int n_kept = 0;

  for (int i = 0; i < n_tris; ++i)
  {
    const MFTL_PackedTriangleComplex *const p = &tris[i];

    int slot[3];
    bool ok = MFTL_TRUE;
    for (int k = 0; k < 3; ++k)
    {
      const int g = p->vertex_gid[k];
      if (g < 0 || g >= tbl->capacity || !gt_has(tbl, g))
      {
        ok = MFTL_FALSE;
        break;
      }
      slot[k] = gt_get(tbl, g);
    }
    if (!ok)
    {
      MFTL_ASSERT(0 && "spatial halo: a triangle references a vertex that never arrived");
      continue;
    }

    /* origin(3t+k) == target(3t+(k+2)%3); the wire carries targets. */
    for (int k = 0; k < 3; ++k) tri_verts[3 * n_kept + k] = slot[(k + 2) % 3];
    n_kept++;
  }

  if (n_kept <= 0)
  {
    free(tri_verts);
    MFTLObjectDestroy(&halo->patch[body]);
    return;
  }

  MFTLBuildTopologyFromIndexList(patch, tri_verts, n_kept);
  free(tri_verts);

  /* ── Restore transmitted identity over the freshly derived one ─────────── */
  int max_t_gid = 0;
  {
    int w = 0;
    for (int i = 0; i < n_tris && w < n_kept; ++i)
    {
      const MFTL_PackedTriangleComplex *const p = &tris[i];

      bool ok = MFTL_TRUE;
      for (int k = 0; k < 3; ++k)
      {
        const int g = p->vertex_gid[k];
        if (g < 0 || g >= tbl->capacity || !gt_has(tbl, g)) ok = MFTL_FALSE;
      }
      if (!ok) continue;

      MFTL_Triangle *const rec = &patch->tris.rec[w];
      rec->global_id = p->global_id;
      rec->owner_proc = p->owner_proc;
      rec->flags = p->flags;

      for (int k = 0; k < 3; ++k)
      {
        MFTL_HalfEdge *const he = &patch->tris.he[3 * w + k];
        he->global_id = 3 * p->global_id + k;  // the implicit-indexing rule, mesh-wide
        he->flags = p->he_flags[k];
      }

      if (p->global_id > max_t_gid) max_t_gid = p->global_id;
      w++;
    }
  }

  /* The counters keep any consumer that sizes a table from them (ghost_sync
   * does) from under-allocating. They are NOT dense on a patch. */
  patch->next_vertex_gid = max_v_gid + 1;
  patch->next_triangle_gid = max_t_gid + 1;
  patch->next_halfedge_gid = 3 * (max_t_gid + 1);

  /* ── Nothing here is owned ─────────────────────────────────────────────── */
  MFTLObjectSetCounts(patch, patch->verts.n, 0, patch->tris.n, 0);
  MFTLObjectBuildDerived(patch);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. The exchange
 * ═══════════════════════════════════════════════════════════════════════════ */

int MFTLExtendSpatialHalo(MFTL_Ensemble *ens, const MFTL_SpatialHaloRequest *requests, int n_requests, MFTL_SpatialHalo *halo)
{
  if (ens == MFTL_NULL || ens->objects == MFTL_NULL || halo == MFTL_NULL) return MFTL_ERROR;
  if (n_requests < 0) return MFTL_ERROR;
  if (n_requests > 0 && requests == MFTL_NULL) return MFTL_ERROR;

  MFTL_Domain *const domain = &ens->domain;
  const int nproc = domain->nproc;
  const int nn = domain->num_neighbors;
  const int n_obj = ens->num_objects;
  if (n_obj <= 0) return MFTL_SUCCESS;
  MFTL_ASSERT(n_obj == halo->num_objects && "MFTLExtendSpatialHalo: halo was created for a different object list");
  MFTL_ASSERT(MFTL_MPI_PackedVertex != MPI_DATATYPE_NULL && "MFTLExtendSpatialHalo: call MFTLRegisterMpiTypes() first");

  /* ── 0. Index the request list BY BODY ──────────────────────────────────── */
  int *const partner_off = (int *) MFTLCallocOrDie((size_t) n_obj + 1, sizeof(int), "spatial halo partner offsets");
  int *const partner_list = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_requests, 1) * sizeof(int), "spatial halo partner list");
  double *const margin_of = (double *) MFTLCallocOrDie((size_t) n_obj, sizeof(double), "spatial halo margins");
  bool *const is_b_body = (bool *) MFTLCallocOrDie((size_t) n_obj, sizeof(bool), "spatial halo B flags");

  for (int r = 0; r < n_requests; ++r)
  {
    const int a = requests[r].body_a;
    const int b = requests[r].body_b;
    if (a < 0 || a >= n_obj || b < 0 || b >= n_obj) continue;
    is_b_body[b] = MFTL_TRUE;
    partner_off[b + 1]++;
    /* One body can be B in several pairs with different margins. The LARGEST
     * wins: a triangle wanted by any partner must be sent, and using the
     * smallest would silently drop contacts for the pair that asked for more. */
    if (requests[r].margin > margin_of[b]) margin_of[b] = requests[r].margin;
  }

  for (int b = 0; b < n_obj; ++b) partner_off[b + 1] += partner_off[b];

  {
    int *const cursor = (int *) MFTLMallocOrDie((size_t) n_obj * sizeof(int), "spatial halo cursors");
    for (int b = 0; b < n_obj; ++b) cursor[b] = partner_off[b];
    for (int r = 0; r < n_requests; ++r)
    {
      const int a = requests[r].body_a;
      const int b = requests[r].body_b;
      if (a < 0 || a >= n_obj || b < 0 || b >= n_obj) continue;
      partner_list[cursor[b]++] = a;
    }
    free(cursor);
  }

  /* Each call REPLACES the halo. Last call's patches describe where the bodies
   * were a timestep ago, and keeping them would report contacts against a
   * surface nobody occupies any more. */
  for (int b = 0; b < n_obj; ++b)
  {
    if (is_b_body[b] && halo->patch[b] != MFTL_NULL) MFTLObjectDestroy(&halo->patch[b]);
  }

  if (nproc < 2 || nn == 0 || n_requests == 0)
  {
    free(partner_off);
    free(partner_list);
    free(margin_of);
    free(is_b_body);
    return MFTL_SUCCESS;
  }

  /* ── 1. Publish my extent of every body, non-blocking ─────────────────────
   *
   * Six doubles per body: centre then half-width, per axis. Same payload the
   * (lo, hi) form cost, and the same one message -- see the note above
   * interval_overlap for why the representation is not a box.
   *
   * NON-BLOCKING because there is exactly one piece of real work that does not
   * depend on the answer, and it is proportional to the mesh rather than to the
   * neighbour count: building each B triangle's own interval (step 2). Nothing
   * after that can start early -- a triangle cannot be classified before the
   * neighbours' extents arrive, packed before it is classified, or sent before
   * it is packed -- so this is the only overlap available, and it is worth
   * taking because the allgather's latency is a fixed cost that the triangle
   * loop hides completely on any partition of a realistic size. */
  double *const my_extent = (double *) MFTLMallocOrDie(6 * (size_t) n_obj * sizeof(double), "spatial halo published extents");
  double *const nb_extent = (double *) MFTLMallocOrDie(6 * (size_t) n_obj * (size_t) nn * sizeof(double), "spatial halo neighbour extents");

  for (int b = 0; b < n_obj; ++b) owned_interval(ens->objects[b], domain, &my_extent[6 * b], &my_extent[6 * b + 3]);

  MPI_Request extent_req;
  MPI_Ineighbor_allgather(my_extent, 6 * n_obj, MPI_DOUBLE, nb_extent, 6 * n_obj, MPI_DOUBLE, domain->comm_graph, &extent_req);

  /* ── 2. Per-triangle intervals, WHILE the extents fly ─────────────────────
   *
   * Built ONCE per body and reused for every neighbour and every partner. The
   * version this replaces rebuilt each triangle's bounds inside the neighbour
   * loop, which is O(neighbours * triangles) box constructions per object PAIR.
   *
   * The margin is folded into the half-width here, so the classification below
   * is a bare distance compare with nothing hoisted out of it -- and, because a
   * triangle's own interval wraps like any other, a triangle straddling the seam
   * is described by its true size rather than by a box that leaves the domain. */
  double **const tri_box = (double **) MFTLCallocOrDie((size_t) n_obj, sizeof(double *), "spatial halo box table");

  for (int b = 0; b < n_obj; ++b)
  {
    if (!is_b_body[b]) continue;
    const MFTL_Object *const obj = ens->objects[b];
    if (obj == MFTL_NULL || obj->tris.n_local <= 0) continue;

    const int nt = obj->tris.n_local;
    double *const boxes = (double *) MFTLMallocOrDie(6 * (size_t) nt * sizeof(double), "spatial halo triangle intervals");
    const double m = margin_of[b];

    for (int t = 0; t < nt; ++t)
    {
      double lo[3], hi[3];
      MFTLGetTriangleBounds(obj, t, lo, hi);
      for (int k = 0; k < 3; ++k)
      {
        boxes[6 * t + k] = wrap_axis(domain, k, 0.5 * (lo[k] + hi[k]));
        boxes[6 * t + 3 + k] = 0.5 * (hi[k] - lo[k]) + m;
      }
    }
    tri_box[b] = boxes;
  }

  MPI_Wait(&extent_req, MPI_STATUS_IGNORE);

  /* ── 3. Classify and pack ───────────────────────────────────────────────── */
  MFTL_SendBuffer *const send_t = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "spatial halo triangle buffers");
  MFTL_SendBuffer *const send_v = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "spatial halo vertex buffers");

  /* Per-destination vertex de-duplication. A contact patch is a connected sheet,
   * so its triangles share most of their corners: sending three vertices per
   * triangle unconditionally would put roughly twice the bytes on the wire that
   * the surface actually contains. The stamp array is reset by bumping an epoch
   * rather than by clearing, so the reset is O(1) per (destination, body). */
  int max_nv = 0;
  for (int b = 0; b < n_obj; ++b)
  {
    if (ens->objects[b] != MFTL_NULL && ens->objects[b]->verts.n > max_nv) max_nv = ens->objects[b]->verts.n;
  }
  unsigned *const vstamp = (unsigned *) MFTLCallocOrDie((size_t) MFTL_MAX(max_nv, 1), sizeof(unsigned), "spatial halo vertex stamps");
  unsigned vepoch = 0u;

  for (int b = 0; b < n_obj; ++b)
  {
    if (tri_box[b] == MFTL_NULL) continue;

    const MFTL_Object *const obj = ens->objects[b];
    const int nt = obj->tris.n_local;
    const double *const boxes = tri_box[b];

    double my_c[3], my_h[3];
    owned_interval(obj, domain, my_c, my_h);
    if (interval_is_empty(my_h)) continue;

    /* ── The macro extent carries the margin, exactly like the triangles do ──
     *
     * A pre-filter may only ever be MORE permissive than the test it stands in
     * for. The per-triangle intervals below are widened by `m`, so an un-widened
     * macro extent is STRICTER than the thing it short-cuts, and it rejects the
     * whole neighbour in precisely the regime the margin exists for: two shares
     * that do not overlap but are within the contact envelope of each other. Two
     * flat surfaces a hair under the tolerance apart on opposite sides of a cut
     * have extents that touch nowhere, and every contact between them is lost
     * before a single triangle is examined.
     *
     * Overlapping bodies hide it completely -- their raw extents intersect
     * anyway -- so it takes a PROXIMITY case at a rank boundary to see it, which
     * is what test_the_tolerance_threshold_is_exact_from_both_sides is.
     */
    for (int k = 0; k < 3; ++k) my_h[k] += margin_of[b];

    for (int nidx = 0; nidx < nn; ++nidx)
    {
      const int dest = domain->neighbors[nidx];
      /* A rank can appear in its own 26-neighbourhood on a thin periodic grid.
       * Sending to itself would deliver a second copy of triangles it already
       * owns, and every one of them would then be reported as a contact twice. */
      if (dest == domain->rank) continue;

      /* MACRO TEST: if this rank's whole share of B misses that rank's share of
       * every partner A, skip the triangle loop entirely. */
      bool any = MFTL_FALSE;
      for (int pi = partner_off[b]; pi < partner_off[b + 1] && !any; ++pi)
      {
        const double *const c = &nb_extent[6 * ((size_t) nidx * n_obj + partner_list[pi])];
        if (interval_is_empty(c + 3)) continue;
        if (interval_overlap(domain, my_c, my_h, c, c + 3)) any = MFTL_TRUE;
      }
      if (!any) continue;

      vepoch++;
      if (vepoch == 0u)
      {
        memset(vstamp, 0, (size_t) MFTL_MAX(max_nv, 1) * sizeof(unsigned));
        vepoch = 1u;
      }

      for (int t = 0; t < nt; ++t)
      {
        bool wanted = MFTL_FALSE;
        for (int pi = partner_off[b]; pi < partner_off[b + 1]; ++pi)
        {
          const double *const c = &nb_extent[6 * ((size_t) nidx * n_obj + partner_list[pi])];
          if (interval_is_empty(c + 3)) continue;
          if (interval_overlap(domain, &boxes[6 * t], &boxes[6 * t + 3], c, c + 3))
          {
            wanted = MFTL_TRUE;  // packed once, however many partners want it
            break;
          }
        }
        if (!wanted) continue;

        /* --- the triangle --- */
        MFTL_SendBuffer *const sbt = &send_t[dest];
        if (sbt->count == sbt->capacity)
        {
          sbt->capacity = (sbt->capacity > 0) ? sbt->capacity * 2 : 256;
          MFTL_PackedTriangleComplex *const grown =
              (MFTL_PackedTriangleComplex *) MFTLMallocOrDie((size_t) sbt->capacity * sizeof(MFTL_PackedTriangleComplex), "spatial halo triangle send");
          if (sbt->count > 0) memcpy(grown, sbt->buffer, (size_t) sbt->count * sizeof(MFTL_PackedTriangleComplex));
          free(sbt->buffer);
          sbt->buffer = grown;
        }

        MFTL_PackedTriangleComplex *const pt = &((MFTL_PackedTriangleComplex *) sbt->buffer)[sbt->count++];
        pt->global_id = obj->tris.rec[t].global_id;
        pt->owner_proc = obj->tris.rec[t].owner_proc;
        pt->flags = obj->tris.rec[t].flags;
        pt->body_idx = b;
        for (int k = 0; k < 3; ++k)
        {
          const MFTL_HalfEdge *const he = &obj->tris.he[3 * t + k];
          /* The TARGET vertex of half-edge 3t+k -- not tris.verts[3t+k], which is
           * its ORIGIN. The two differ by a rotation. */
          pt->vertex_gid[k] = obj->verts.rec[he->vertex_idx].global_id;
          pt->he_flags[k] = he->flags;
        }

        /* --- its three vertices, each at most once per destination --- */
        for (int k = 0; k < 3; ++k)
        {
          const int v = obj->tris.he[3 * t + k].vertex_idx;
          if (vstamp[v] == vepoch) continue;
          vstamp[v] = vepoch;

          MFTL_SendBuffer *const sbv = &send_v[dest];
          if (sbv->count == sbv->capacity)
          {
            sbv->capacity = (sbv->capacity > 0) ? sbv->capacity * 2 : 512;
            MFTL_PackedVertex *const grown =
                (MFTL_PackedVertex *) MFTLMallocOrDie((size_t) sbv->capacity * sizeof(MFTL_PackedVertex), "spatial halo vertex send");
            if (sbv->count > 0) memcpy(grown, sbv->buffer, (size_t) sbv->count * sizeof(MFTL_PackedVertex));
            free(sbv->buffer);
            sbv->buffer = grown;
          }

          MFTL_PackedVertex *const pv = &((MFTL_PackedVertex *) sbv->buffer)[sbv->count++];
          const double *const src = &obj->verts.pos[3 * v];
          pv->pos[0] = src[0];
          pv->pos[1] = src[1];
          pv->pos[2] = src[2];
          pv->global_id = obj->verts.rec[v].global_id;
          pv->owner_proc = obj->verts.rec[v].owner_proc;
          pv->flags = obj->verts.rec[v].flags;
          pv->body_idx = b;
        }
      }
    }
  }

  free(vstamp);
  for (int b = 0; b < n_obj; ++b) free(tri_box[b]);
  free(tri_box);
  free(my_extent);
  free(nb_extent);

  /* ── 4. Exchange ────────────────────────────────────────────────────────── */
  MFTL_Exchange x[2];
  x[0].send_bufs = send_v;
  x[0].type = MFTL_MPI_PackedVertex;
  x[0].type_size = sizeof(MFTL_PackedVertex);
  x[0].recv_buf = MFTL_NULL;
  x[0].recv_count = 0;

  x[1].send_bufs = send_t;
  x[1].type = MFTL_MPI_PackedTriangleComplex;
  x[1].type_size = sizeof(MFTL_PackedTriangleComplex);
  x[1].recv_buf = MFTL_NULL;
  x[1].recv_count = 0;

  MFTL_ExchangeCtx ctx;
  MFTLExchangeStart(domain, x, 2, MFTL_FALSE, &ctx);
  MFTLExchangeWait(&ctx, 0);
  MFTLExchangeWait(&ctx, 1);

  for (int i = 0; i < nproc; ++i)
  {
    free(send_v[i].buffer);
    free(send_t[i].buffer);
  }
  free(send_v);
  free(send_t);

  /* ── 5. Group the arrivals by body and rebuild each patch ───────────────── */
  {
    const MFTL_PackedVertex *const av = (const MFTL_PackedVertex *) x[0].recv_buf;
    const MFTL_PackedTriangleComplex *const at = (const MFTL_PackedTriangleComplex *) x[1].recv_buf;

    /* Counting sort by body: the stream arrives grouped by SENDER, and a patch
     * has to see every contribution to its body at once. O(arrivals + bodies),
     * no comparisons. */
    int *const v_off = (int *) MFTLCallocOrDie((size_t) n_obj + 1, sizeof(int), "spatial halo vertex grouping");
    int *const t_off = (int *) MFTLCallocOrDie((size_t) n_obj + 1, sizeof(int), "spatial halo triangle grouping");

    for (int i = 0; i < x[0].recv_count; ++i)
    {
      const int b = av[i].body_idx;
      if (b >= 0 && b < n_obj) v_off[b + 1]++;
    }
    for (int i = 0; i < x[1].recv_count; ++i)
    {
      const int b = at[i].body_idx;
      if (b >= 0 && b < n_obj) t_off[b + 1]++;
    }
    for (int b = 0; b < n_obj; ++b)
    {
      v_off[b + 1] += v_off[b];
      t_off[b + 1] += t_off[b];
    }

    MFTL_PackedVertex *const sv = (MFTL_PackedVertex *) MFTLMallocOrDie((size_t) MFTL_MAX(x[0].recv_count, 1) * sizeof(MFTL_PackedVertex), "spatial halo vsort");
    MFTL_PackedTriangleComplex *const st =
        (MFTL_PackedTriangleComplex *) MFTLMallocOrDie((size_t) MFTL_MAX(x[1].recv_count, 1) * sizeof(MFTL_PackedTriangleComplex), "spatial halo tsort");

    {
      int *const cv = (int *) MFTLMallocOrDie((size_t) n_obj * sizeof(int), "spatial halo vcursor");
      int *const ct = (int *) MFTLMallocOrDie((size_t) n_obj * sizeof(int), "spatial halo tcursor");
      for (int b = 0; b < n_obj; ++b)
      {
        cv[b] = v_off[b];
        ct[b] = t_off[b];
      }
      for (int i = 0; i < x[0].recv_count; ++i)
      {
        const int b = av[i].body_idx;
        if (b >= 0 && b < n_obj) sv[cv[b]++] = av[i];
      }
      for (int i = 0; i < x[1].recv_count; ++i)
      {
        const int b = at[i].body_idx;
        if (b >= 0 && b < n_obj) st[ct[b]++] = at[i];
      }
      free(cv);
      free(ct);
    }

    mftl_gid_table tbl;
    gt_init(&tbl);
    mftl_int_buffer scratch;
    ib_init(&scratch, 1024);

    for (int b = 0; b < n_obj; ++b)
    {
      if (!is_b_body[b]) continue;
      rebuild_patch(halo, b, &sv[v_off[b]], v_off[b + 1] - v_off[b], &st[t_off[b]], t_off[b + 1] - t_off[b], &tbl, &scratch);
    }

    ib_free(&scratch);
    gt_free(&tbl);
    free(sv);
    free(st);
    free(v_off);
    free(t_off);
  }

  free(x[0].recv_buf);
  free(x[1].recv_buf);
  MFTLExchangeFree(&ctx);

  free(partner_off);
  free(partner_list);
  free(margin_of);
  free(is_b_body);

  return MFTL_SUCCESS;
}
