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
 * 2. The directory
 *
 * A SPARSE, SORTED KEY ARRAY, and every consumer in this file reads it that way.
 * The version this replaces was a `[n_bodies_global]` table of patch pointers
 * and radii -- the shape that has to be abandoned for exactly the reason
 * MFTL_Ensemble::local_objects was: it makes every rank pay for every droplet in
 * the simulation in order to describe the handful it is anywhere near.
 *
 * Two sets go in, and the second is why the directory cannot simply BE the
 * ensemble's local list:
 *
 *   - the bodies this rank HOLDS, so that a resolved radius has somewhere to
 *     live;
 *   - the bodies this rank RECEIVED A PATCH FOR, which is precisely the set it
 *     holds nothing of. A rank with droplet A and none of droplet B is the rank
 *     that must be told about B, and it has no local object to hang that on.
 *
 * Insertion is an ordered memmove rather than an append-and-sort, for the reason
 * the ensemble's own directory gives: the sort order is what makes `gid -> slot`
 * a binary search instead of a hash map the library does not have.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Grows the three parallel directory arrays to hold @p need entries.
 * @param h    The halo.
 * @param need Entries required.
 */
static void halo_reserve(MFTL_SpatialHalo *h, int need)
{
  if (need <= h->capacity) return;

  int cap = (h->capacity > 0) ? h->capacity : 8;
  while (cap < need) cap *= 2;

  int *const gid = (int *) MFTLMallocOrDie((size_t) cap * sizeof(int), "spatial halo body ids");
  MFTL_Object **const patch = (MFTL_Object **) MFTLCallocOrDie((size_t) cap, sizeof(MFTL_Object *), "spatial halo patches");
  double *const rad = (double *) MFTLCallocOrDie((size_t) cap, sizeof(double), "spatial halo radii");

  if (h->num_objects > 0)
  {
    memcpy(gid, h->body_gid, (size_t) h->num_objects * sizeof(int));
    memcpy(patch, h->patch, (size_t) h->num_objects * sizeof(MFTL_Object *));
    memcpy(rad, h->global_radius, (size_t) h->num_objects * sizeof(double));
  }

  free(h->body_gid);
  free(h->patch);
  free(h->global_radius);
  h->body_gid = gid;
  h->patch = patch;
  h->global_radius = rad;
  h->capacity = cap;
}

/**
 * @brief Directory slot of @p gid, inserting an empty entry if it is absent.
 * @param h   The halo.
 * @param gid The body id.
 * @return The slot. Never #MFTL_ERROR.
 */
static int halo_entry(MFTL_SpatialHalo *h, int gid)
{
  const int found = MFTLSpatialHaloFindBody(h, gid);
  if (found != MFTL_ERROR) return found;

  halo_reserve(h, h->num_objects + 1);

  int i = h->num_objects;
  while (i > 0 && h->body_gid[i - 1] > gid) --i;

  if (i < h->num_objects)
  {
    memmove(&h->body_gid[i + 1], &h->body_gid[i], (size_t) (h->num_objects - i) * sizeof(int));
    memmove(&h->patch[i + 1], &h->patch[i], (size_t) (h->num_objects - i) * sizeof(MFTL_Object *));
    memmove(&h->global_radius[i + 1], &h->global_radius[i], (size_t) (h->num_objects - i) * sizeof(double));
  }

  h->body_gid[i] = gid;
  h->patch[i] = MFTL_NULL;
  h->global_radius[i] = 0.0;
  h->num_objects++;
  return i;
}

/**
 * @brief Gives every body @p ens currently holds here a directory entry.
 *
 * Called at the top of every collective entry point, so a halo created before a
 * redistribution still works after one -- which matters, because a halo is
 * created once and a redistribution destroys and creates local objects on every
 * timestep. Entries for bodies that have LEFT are kept: they may still carry a
 * patch, and a patch is what a rank holding nothing of a body has instead of an
 * object.
 *
 * @param h   The halo.
 * @param ens The ensemble.
 */
static void halo_sync_bodies(MFTL_SpatialHalo *h, const MFTL_Ensemble *ens)
{
  halo_reserve(h, h->num_objects + ens->n_local_objects);
  for (int o = 0; o < ens->n_local_objects; ++o) (void) halo_entry(h, ens->local_objects[o]->global_id);
}

/**
 * @brief Appends one candidate pair. Ordering and uniqueness are imposed later.
 * @param h The halo.
 * @param a Smaller body id.
 * @param b Larger body id.
 */
static void halo_pair_add(MFTL_SpatialHalo *h, int a, int b)
{
  if (h->n_pairs == h->cap_pairs)
  {
    h->cap_pairs = (h->cap_pairs > 0) ? h->cap_pairs * 2 : 64;
    MFTL_SpatialHaloPair *const grown =
        (MFTL_SpatialHaloPair *) MFTLMallocOrDie((size_t) h->cap_pairs * sizeof(MFTL_SpatialHaloPair), "spatial halo pairs");
    if (h->n_pairs > 0) memcpy(grown, h->pair, (size_t) h->n_pairs * sizeof(MFTL_SpatialHaloPair));
    free(h->pair);
    h->pair = grown;
  }
  h->pair[h->n_pairs].body_a = a;
  h->pair[h->n_pairs].body_b = b;
  h->n_pairs++;
}

/**
 * @brief Orders pairs by (A, B), so removing duplicates is one linear sweep.
 * @param x First element.
 * @param y Second element.
 * @return Negative, zero or positive in the usual qsort sense.
 */
static int pair_cmp(const void *x, const void *y)
{
  const MFTL_SpatialHaloPair *const a = (const MFTL_SpatialHaloPair *) x;
  const MFTL_SpatialHaloPair *const b = (const MFTL_SpatialHaloPair *) y;
  if (a->body_a != b->body_a) return (a->body_a < b->body_a) ? -1 : 1;
  if (a->body_b != b->body_b) return (a->body_b < b->body_b) ? -1 : 1;
  return 0;
}

/**
 * @brief Sorts the pair list and removes duplicates in place.
 *
 * A pair can be produced twice -- once because both bodies are local and their
 * shares are near, once because a patch of the larger one also arrived -- and a
 * duplicate would traverse the same trees twice and report every contact in them
 * twice. Deduplicating here rather than at the two producers keeps the "reported
 * once" rule in one place.
 *
 * @param h The halo.
 */
static void halo_pairs_finish(MFTL_SpatialHalo *h)
{
  if (h->n_pairs < 2) return;

  qsort(h->pair, (size_t) h->n_pairs, sizeof(MFTL_SpatialHaloPair), pair_cmp);

  int w = 1;
  for (int i = 1; i < h->n_pairs; ++i)
  {
    if (h->pair[i].body_a == h->pair[w - 1].body_a && h->pair[i].body_b == h->pair[w - 1].body_b) continue;
    h->pair[w++] = h->pair[i];
  }
  h->n_pairs = w;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2b. Lifecycle
 * ═══════════════════════════════════════════════════════════════════════════ */

MFTL_SpatialHalo *MFTLSpatialHaloCreate(MFTL_Ensemble *ens)
{
  if (ens == MFTL_NULL) return MFTL_NULL;

  MFTL_SpatialHalo *const h = (MFTL_SpatialHalo *) MFTLCallocOrDie(1, sizeof(MFTL_SpatialHalo), "MFTL_SpatialHalo");
  h->domain = &ens->domain;
  halo_sync_bodies(h, ens);
  return h;
}

void MFTLSpatialHaloDestroy(MFTL_SpatialHalo **p_halo)
{
  if (p_halo == MFTL_NULL || *p_halo == MFTL_NULL) return;
  MFTL_SpatialHalo *const h = *p_halo;

  for (int e = 0; e < h->num_objects; ++e)
  {
    if (h->patch[e] != MFTL_NULL) MFTLObjectDestroy(&h->patch[e]);
  }
  free(h->body_gid);
  free(h->patch);
  free(h->global_radius);
  free(h->pair);
  free(h);
  *p_halo = MFTL_NULL;
}

int MFTLSpatialHaloResolveRadii(MFTL_Ensemble *ens, MFTL_SpatialHalo *halo)
{
  if (ens == MFTL_NULL || halo == MFTL_NULL) return MFTL_ERROR;

  halo_sync_bodies(halo, ens);

  const int b = ens->n_local_objects;

  /* ── Two frames, twelve slots, ONE per-body reduction ────────────────────
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
   * Cost is payload, not a second collective: 12 doubles per body instead of 6,
   * every one of them a MIN because the upper corners are carried NEGATED.
   *
   * ── The route is the body's HOLDERS, not comm_world ──────────────────────
   *
   * This used to be an `MPI_Allreduce` on two `[12 * n_bodies_global]` staging
   * buffers -- 192 bytes per body on every rank, to learn one number about each
   * of the few bodies this rank is near, and the reduction itself proportional
   * to the whole ensemble. It is a per-body reduction like any other, so it goes
   * through the one place that chooses a communicator, indexed by LOCAL SLOT.
   */
  enum
  {
    STRIDE = 12  /* [0..2] lo, [3..5] -hi, raw | [6..8] lo, [9..11] -hi, shifted */
  };

  MFTL_BodyReduceOp ops[STRIDE];
  for (int j = 0; j < STRIDE; ++j) ops[j] = MFTL_BODY_REDUCE_MIN;

  double *const val = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(b * STRIDE, 1) * sizeof(double), "spatial halo extents");

  double half_domain[3];
  for (int k = 0; k < 3; ++k) half_domain[k] = 0.5 * ens->domain.domain_size[k];

  for (int o = 0; o < b; ++o)
  {
    double *const v = &val[(size_t) o * STRIDE];
    for (int j = 0; j < STRIDE; ++j) v[j] = MFTL_DBL_MAX;

    double lo[3], hi[3];

    owned_bounds(ens->local_objects[o], lo, hi, MFTL_NULL);
    if (box_is_empty(lo, hi)) continue;  /* an identity row: this rank owns nothing of it */
    for (int k = 0; k < 3; ++k)
    {
      v[k] = lo[k];
      v[3 + k] = -hi[k];
    }

    owned_bounds(ens->local_objects[o], lo, hi, half_domain);
    for (int k = 0; k < 3; ++k)
    {
      v[6 + k] = lo[k];
      v[9 + k] = -hi[k];
    }
  }

  MFTLEnsembleReduceBodies(ens, val, STRIDE, ops, MFTL_BODY_REDUCE_SCOPE_AUTO);

  for (int o = 0; o < b; ++o)
  {
    const double *const v = &val[(size_t) o * STRIDE];
    const int e = MFTLSpatialHaloFindBody(halo, ens->local_objects[o]->global_id);
    MFTL_ASSERT(e != MFTL_ERROR && "MFTLSpatialHaloResolveRadii: halo_sync_bodies did not enter a held body");

    if (v[0] == MFTL_DBL_MAX)
    {
      halo->global_radius[e] = 0.0;  /* every holder that reached the reduction owns nothing of it */
      continue;
    }

    /* Half the LONGEST SIDE, not half the diagonal: for a sphere of radius R
     * this is R, so a caller scaling by "a thousandth of the body" gets what it
     * asked for rather than a silent sqrt(3) inflation. */
    double longest = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      double d = (-v[3 + k]) - v[k];

      if (ens->domain.is_periodic[k] && ens->domain.domain_size[k] > 0.0)
      {
        const double d_shifted = (-v[9 + k]) - v[6 + k];
        if (d_shifted < d) d = d_shifted;
      }

      if (d > longest) longest = d;
    }
    halo->global_radius[e] = 0.5 * longest;
  }

  free(val);

  /* ── And one hop further, because a MARGIN needs the OTHER body's size ────
   *
   * The reduction answers "how big is this body" only on the ranks that hold it.
   * That is not enough for the rank that has to decide what to SEND: it holds
   * body B and is asked for the parts of B within `MAX(tol_A, tol_B)` of a
   * neighbour's share of body A, and it holds nothing of A. Under-estimating
   * tol_A there is a MISSED CROSS-RANK CONTACT and nothing else -- silent, and
   * only at a rank boundary, which is the worst place for it.
   *
   * So each rank publishes `(body id, radius)` for the bodies it holds to its
   * 26 neighbours. One hop is exactly the right distance and not a guess: the
   * halo only ever exchanges payload along `comm_graph`, so a rank that is not a
   * graph neighbour is a rank this body will never be sent to.
   *
   * Two doubles per body per rank, `O(local bodies * degree)`. The dense form it
   * replaces gave every rank every radius, which is a strict superset of this
   * and `O(bodies in the ensemble)`.
   */
  const MFTL_Domain *const dom = &ens->domain;
  const int nn = dom->num_neighbors;
  if (dom->nproc < 2 || nn == 0) return MFTL_SUCCESS;

  double *const send = (double *) MFTLMallocOrDie(2 * (size_t) MFTL_MAX(b, 1) * sizeof(double), "spatial halo radius publish");
  for (int o = 0; o < b; ++o)
  {
    const int g = ens->local_objects[o]->global_id;
    send[2 * o] = (double) g;
    send[2 * o + 1] = MFTLSpatialHaloRadius(halo, g);
  }

  int *const rcount = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "spatial halo radius counts");
  int *const rdispl = (int *) MFTLMallocOrDie((size_t) nn * sizeof(int), "spatial halo radius displacements");

  const int scount = 2 * b;
  MPI_Neighbor_allgather(&scount, 1, MPI_INT, rcount, 1, MPI_INT, dom->comm_graph);

  size_t total = 0;
  for (int i = 0; i < nn; ++i)
  {
    rdispl[i] = (int) total;
    total += (size_t) rcount[i];
  }
  MFTL_ASSERT(total <= (size_t) 0x7fffffff && "MFTLSpatialHaloResolveRadii: neighbour radius payload exceeds an MPI int count");

  double *const recv = (double *) MFTLMallocOrDie(MFTL_MAX(total, (size_t) 1) * sizeof(double), "spatial halo radius receive");
  MPI_Neighbor_allgatherv(send, scount, MPI_DOUBLE, recv, rcount, rdispl, MPI_DOUBLE, dom->comm_graph);

  /* MAX, not assignment: a body held here already has the exact answer from the
   * reduction, and a holder's published copy is the same number to the bit --
   * the reduction is combined in ascending source-rank order precisely so that
   * it is. Taking the maximum makes the two paths commute instead of racing. */
  for (size_t i = 0; i + 1 < total; i += 2)
  {
    const int g = (int) recv[i];
    if (g < 0) continue;
    const int e = halo_entry(halo, g);
    if (recv[i + 1] > halo->global_radius[e]) halo->global_radius[e] = recv[i + 1];
  }

  free(recv);
  free(rdispl);
  free(rcount);
  free(send);
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
 * @param halo    The halo. `halo->patch[entry]` is REPLACED, not appended to.
 * @param entry   Directory slot of the body, from halo_entry().
 * @param verts   Received vertex records.
 * @param n_verts Number of received vertices.
 * @param tris    Received triangle records.
 * @param n_tris  Number of received triangles.
 * @param tbl     Epoch-stamped id-to-slot table, reused across bodies.
 * @param scratch Reusable int buffer for the de-duplication pass.
 */
static void rebuild_patch(MFTL_SpatialHalo *halo, int entry, const MFTL_PackedVertex *verts, int n_verts, const MFTL_PackedTriangleComplex *tris, int n_tris,
                          mftl_gid_table *tbl, mftl_int_buffer *scratch)
{
  MFTL_Object *patch = halo->patch[entry];

  if (n_tris <= 0 || n_verts < 3)
  {
    if (patch != MFTL_NULL) MFTLObjectDestroy(&halo->patch[entry]);
    return;
  }

  if (patch == MFTL_NULL)
  {
    patch = MFTLObjectCreate(halo->body_gid[entry]);
    /* Borrowed, exactly as MFTLEnsembleAddObject would set it. The patch is
     * deliberately NOT added to the ensemble's object list: redistribution and
     * the ghost syncs iterate that list, and a patch must be invisible to both. */
    patch->domain = halo->domain;
    halo->patch[entry] = patch;
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
  /* Entry count and id range. A halo patch is a thin shell of another rank's
   * share, so its ids are a sparse scattering over the whole body -- this is the
   * case gt_begin()'s sparse mode exists for. */
  gt_begin(tbl, n_verts, max_gid + 1);

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
    MFTLObjectDestroy(&halo->patch[entry]);
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
      if (g < 0 || !gt_has(tbl, g))
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
    MFTLObjectDestroy(&halo->patch[entry]);
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
        if (g < 0 || !gt_has(tbl, g)) ok = MFTL_FALSE;
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
 * 4. The body-extent exchange, the broad phase and the payload
 *
 * ── WHAT A RANK PUBLISHES, AND WHY IT IS NOT A DENSE TABLE ────────────────
 *
 * One record per body THIS RANK HOLDS: `(body id, centre, half-width, global
 * radius, contact envelope)`, nine doubles, through a neighbourhood
 * allgatherv. The version this replaces published `6 * n_bodies_global` doubles
 * to every neighbour whether or not it held any of them -- 48 bytes per body per
 * neighbour, i.e. 1.2 kB per body per rank on a full 26-neighbourhood, which is
 * a gigabyte at a million droplets to say "not mine" about almost all of them.
 * The sparse form is `O(local bodies * degree)`: constant in the ensemble.
 *
 * The extra three slots are what let the pair list be DERIVED rather than
 * enumerated. A rank cannot decide whether body B of its own is worth sending
 * towards body A of a neighbour without knowing A's envelope, and the envelope
 * scales with the body -- so the radius and the tolerance ride along rather than
 * being reconstructed from a second collective.
 *
 * ── THE BROAD PHASE ───────────────────────────────────────────────────────
 *
 * A uniform grid over the domain whose cell is at least as wide as the widest
 * body plus the largest envelope, so two extents that overlap are always within
 * one cell of each other and a query is the 27 cells around a centre. It is
 * built over the ARRIVING records and queried with this rank's own bodies, and
 * again over this rank's own bodies to find the local-local pairs. Cost is
 * `O(bodies + records)`; the all-pairs enumeration it replaces was `O(B^2)` on
 * every rank.
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief Doubles per published body record: gid, centre[3], half[3], radius, tol. */
#define MFTL_HALO_REC 9

/**
 * @brief A uniform grid over the domain, one bucket list per cell.
 *
 * Sized so that one cell is at least `2*max_half + max_margin` on every axis,
 * which is what makes "within one cell" a superset of "extents overlap" and
 * therefore what makes a 27-cell query exhaustive. The cell COUNT is capped
 * against the item count so that a domain far larger than the bodies does not
 * turn into an allocation proportional to the domain rather than to the data.
 */
typedef struct
{
  int n[3];  ///< Cells per axis, at least 1.
  double inv_w[3];  ///< Reciprocal cell width per axis.
  double origin[3];  ///< Lower corner of cell (0,0,0).
  int *head;  ///< [n0*n1*n2] first item in the cell, or MFTL_ERROR.
  int *next;  ///< [n_items] next item in the same cell, or MFTL_ERROR.
  int n_items;  ///< Items inserted.
} body_grid;

/**
 * @brief Builds a grid over @p n_items centres, resolving the cell size first.
 *
 * @param g        Receives the grid. Release with grid_free().
 * @param dom      Supplies the bounds and periodicity.
 * @param centre   First centre; `centre[i*stride + k]` is item i's axis k.
 * @param stride   Doubles between consecutive centres.
 * @param n_items  Items.
 * @param reach    Per-axis width two items' centres can be apart and still be
 *                 within reach of each other. The cell is at least this wide.
 */
static void grid_build(body_grid *g, const MFTL_Domain *dom, const double *centre, int stride, int n_items, const double reach[3])
{
  g->n_items = n_items;

  long cells = 1;
  for (int k = 0; k < 3; ++k)
  {
    const double size = dom->domain_size[k];
    double want = reach[k];
    if (!(want > 0.0)) want = MFTL_DBL_MAX;

    int nk = 1;
    if (size > 0.0 && want < size) nk = (int) (size / want);
    if (nk < 1) nk = 1;

    g->n[k] = nk;
    g->origin[k] = dom->global_min[k];
    cells *= nk;
  }

  /* A domain a thousand droplet radii wide would otherwise ask for a cell table
   * proportional to the DOMAIN rather than to the data. Halve the longest axis
   * until the table is proportional to what goes in it. */
  const long cap = 4L * (long) MFTL_MAX(n_items, 1) + 64L;
  while (cells > cap)
  {
    int worst = 0;
    for (int k = 1; k < 3; ++k)
    {
      if (g->n[k] > g->n[worst]) worst = k;
    }
    if (g->n[worst] <= 1) break;
    cells /= g->n[worst];
    g->n[worst] /= 2;
    cells *= g->n[worst];
  }

  for (int k = 0; k < 3; ++k)
  {
    const double w = (dom->domain_size[k] > 0.0) ? dom->domain_size[k] / (double) g->n[k] : 1.0;
    g->inv_w[k] = 1.0 / w;
  }

  g->head = (int *) MFTLMallocOrDie((size_t) cells * sizeof(int), "spatial halo grid heads");
  for (long i = 0; i < cells; ++i) g->head[i] = MFTL_ERROR;
  g->next = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_items, 1) * sizeof(int), "spatial halo grid links");

  for (int i = 0; i < n_items; ++i)
  {
    const double *const c = &centre[(size_t) i * stride];
    int idx = 0, mul = 1;
    for (int k = 0; k < 3; ++k)
    {
      int ck = (int) ((c[k] - g->origin[k]) * g->inv_w[k]);
      if (ck < 0) ck = 0;
      if (ck >= g->n[k]) ck = g->n[k] - 1;
      idx += ck * mul;
      mul *= g->n[k];
    }
    g->next[i] = g->head[idx];
    g->head[idx] = i;
  }
}

/** @brief Releases a grid. @param g The grid. */
static void grid_free(body_grid *g)
{
  free(g->head);
  free(g->next);
  g->head = MFTL_NULL;
  g->next = MFTL_NULL;
}

/**
 * @brief The distinct cell indices to visit along one axis around @p ck.
 *
 * Three cells normally. FEWER when the axis has three cells or less, because
 * `{ck-1, ck, ck+1}` then wraps onto cells already visited and every duplicate
 * would produce a duplicate candidate -- which on the send path is a triangle
 * packed twice and therefore a contact reported twice.
 *
 * @param      g   The grid.
 * @param      dom Supplies the periodicity.
 * @param      k   Axis.
 * @param      ck  Cell of the query point on that axis.
 * @param[out] out Up to 3 distinct cell indices.
 * @return How many were written.
 */
static int grid_axis_cells(const body_grid *g, const MFTL_Domain *dom, int k, int ck, int out[3])
{
  const int n = g->n[k];

  if (n <= 3)
  {
    if (!dom->is_periodic[k])
    {
      int m = 0;
      for (int i = ck - 1; i <= ck + 1; ++i)
      {
        if (i >= 0 && i < n) out[m++] = i;
      }
      return m;
    }
    for (int i = 0; i < n; ++i) out[i] = i;
    return n;
  }

  int m = 0;
  for (int d = -1; d <= 1; ++d)
  {
    int i = ck + d;
    if (i < 0 || i >= n)
    {
      if (!dom->is_periodic[k]) continue;
      i = (i + n) % n;
    }
    out[m++] = i;
  }
  return m;
}

/**
 * @brief Appends every item within one cell of @p c to @p out.
 *
 * @param      g   The grid.
 * @param      dom Supplies the periodicity.
 * @param      c   Query centre.
 * @param[out] out Receives item indices, appended.
 */
static void grid_collect(const body_grid *g, const MFTL_Domain *dom, const double *c, mftl_int_buffer *out)
{
  int base[3];
  for (int k = 0; k < 3; ++k)
  {
    int ck = (int) ((c[k] - g->origin[k]) * g->inv_w[k]);
    if (ck < 0) ck = 0;
    if (ck >= g->n[k]) ck = g->n[k] - 1;
    base[k] = ck;
  }

  int cx[3], cy[3], cz[3];
  const int nx = grid_axis_cells(g, dom, 0, base[0], cx);
  const int ny = grid_axis_cells(g, dom, 1, base[1], cy);
  const int nz = grid_axis_cells(g, dom, 2, base[2], cz);

  for (int iz = 0; iz < nz; ++iz)
  {
    for (int iy = 0; iy < ny; ++iy)
    {
      for (int ix = 0; ix < nx; ++ix)
      {
        const int idx = cx[ix] + cy[iy] * g->n[0] + cz[iz] * g->n[0] * g->n[1];
        for (int i = g->head[idx]; i != MFTL_ERROR; i = g->next[i]) ib_add(out, i);
      }
    }
  }
}

/**
 * @brief Inserts @p gid into a sorted, unique int list, growing it on demand.
 *
 * @param[in,out] list Sorted list.
 * @param[in,out] n    Entries in use.
 * @param[in,out] cap  Entries allocated.
 * @param         gid  The value.
 * @return The slot @p gid now occupies.
 */
static int sorted_insert(int **list, int *n, int *cap, int gid)
{
  int lo = 0, hi = *n - 1;
  while (lo <= hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    if ((*list)[mid] == gid) return mid;
    if ((*list)[mid] < gid)
      lo = mid + 1;
    else
      hi = mid - 1;
  }

  if (*n == *cap)
  {
    *cap = (*cap > 0) ? *cap * 2 : 16;
    int *const grown = (int *) MFTLMallocOrDie((size_t) *cap * sizeof(int), "spatial halo arrival bodies");
    if (*n > 0) memcpy(grown, *list, (size_t) *n * sizeof(int));
    free(*list);
    *list = grown;
  }

  if (lo < *n) memmove(&(*list)[lo + 1], &(*list)[lo], (size_t) (*n - lo) * sizeof(int));
  (*list)[lo] = gid;
  (*n)++;
  return lo;
}

/**
 * @brief Slot of @p gid in a sorted list, or #MFTL_ERROR.
 * @param list Sorted list.
 * @param n    Entries.
 * @param gid  The value.
 * @return The slot, or #MFTL_ERROR.
 */
static int sorted_find(const int *list, int n, int gid)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    const int mid = lo + ((hi - lo) >> 1);
    if (list[mid] == gid) return mid;
    if (list[mid] < gid)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return MFTL_ERROR;
}

/**
 * @brief Drops directory entries this rank neither holds nor has a patch for.
 *
 * Without it the directory only ever grows: a droplet that drifted past this
 * rank three thousand timesteps ago would still own a slot. With it the
 * directory is bounded by what is actually near, which is the property the
 * sparse form exists to have.
 *
 * @param h   The halo.
 * @param ens The ensemble, for the "still held here" half of the test.
 */
static void halo_prune(MFTL_SpatialHalo *h, const MFTL_Ensemble *ens)
{
  int w = 0;
  for (int e = 0; e < h->num_objects; ++e)
  {
    const bool keep = (h->patch[e] != MFTL_NULL) || (MFTLEnsembleFindLocalSlot(ens, h->body_gid[e]) != MFTL_ERROR);
    if (!keep) continue;

    h->body_gid[w] = h->body_gid[e];
    h->patch[w] = h->patch[e];
    h->global_radius[w] = h->global_radius[e];
    w++;
  }
  h->num_objects = w;
}

/**
 * @brief Fills `halo->pair` from this rank's own extents and its arrivals.
 *
 * ── TWO SOURCES, AND BOTH ARE NEEDED ──────────────────────────────────────
 *
 * LOCAL-LOCAL. Two bodies this rank holds shares of, whose shares are within the
 * pair's envelope. No halo is involved and none would ever arrive: the contact
 * is entirely inside this partition.
 *
 * LOCAL-ARRIVED. A body some neighbour holds, whose published extent is within
 * the envelope of one of this rank's bodies with a SMALLER id. That is precisely
 * the condition the sender applied when it decided to pack, mirrored -- so the
 * pair list and the patch that serves it are two readings of one decision rather
 * than two decisions that have to be kept in step.
 *
 * The A side is always the smaller body id and always held here, because only an
 * OWNED A-side primitive may report a contact. A pair produced by both sources
 * is deduplicated by halo_pairs_finish().
 *
 * @param ens       The ensemble.
 * @param halo      Receives the pairs.
 * @param my_rec    `[n_my * MFTL_HALO_REC]` this rank's published records.
 * @param n_my      Records in @p my_rec.
 * @param nb_rec    `[n_nb_rec * MFTL_HALO_REC]` the neighbours' records, or NULL.
 * @param n_nb_rec  Records in @p nb_rec.
 * @rank_local No MPI.
 */
static void build_near_pairs(MFTL_Ensemble *ens, MFTL_SpatialHalo *halo, const double *my_rec, int n_my, const double *nb_rec, int n_nb_rec)
{
  if (n_my == 0) return;

  const MFTL_Domain *const domain = &ens->domain;

  double reach[3] = {0.0, 0.0, 0.0};
  double max_tol = 0.0;
  for (int i = 0; i < n_my; ++i)
  {
    const double *const r = &my_rec[(size_t) i * MFTL_HALO_REC];
    for (int k = 0; k < 3; ++k) reach[k] = MFTL_MAX(reach[k], 2.0 * r[4 + k]);
    max_tol = MFTL_MAX(max_tol, r[8]);
  }
  for (int i = 0; i < n_nb_rec; ++i)
  {
    const double *const r = &nb_rec[(size_t) i * MFTL_HALO_REC];
    for (int k = 0; k < 3; ++k) reach[k] = MFTL_MAX(reach[k], 2.0 * r[4 + k]);
    max_tol = MFTL_MAX(max_tol, r[8]);
  }
  for (int k = 0; k < 3; ++k) reach[k] += max_tol;

  body_grid grid;
  grid_build(&grid, domain, my_rec + 1, MFTL_HALO_REC, n_my, reach);

  mftl_int_buffer cand;
  ib_init(&cand, 64);

  /* ── Local against local ─────────────────────────────────────────────── */
  for (int i = 0; i < n_my; ++i)
  {
    const double *const ri = &my_rec[(size_t) i * MFTL_HALO_REC];
    const int gi = (int) ri[0];

    ib_reset(&cand);
    grid_collect(&grid, domain, ri + 1, &cand);

    for (int ci = 0; ci < cand.size; ++ci)
    {
      const int j = cand.data[ci];
      if (j == i) continue;

      const double *const rj = &my_rec[(size_t) j * MFTL_HALO_REC];
      const int gj = (int) rj[0];
      if (gj <= gi) continue;  // each unordered pair once, A = the smaller id

      const double m = MFTL_MAX(ri[8], rj[8]);
      double wh[3];
      for (int k = 0; k < 3; ++k) wh[k] = ri[4 + k] + m;
      if (!interval_overlap(domain, ri + 1, wh, rj + 1, rj + 4)) continue;

      halo_pair_add(halo, gi, gj);
    }
  }

  /* ── Local against what a neighbour published ────────────────────────── */
  for (int r = 0; r < n_nb_rec; ++r)
  {
    const double *const rr = &nb_rec[(size_t) r * MFTL_HALO_REC];
    const int gr = (int) rr[0];

    ib_reset(&cand);
    grid_collect(&grid, domain, rr + 1, &cand);

    for (int ci = 0; ci < cand.size; ++ci)
    {
      const int i = cand.data[ci];
      const double *const ri = &my_rec[(size_t) i * MFTL_HALO_REC];
      const int gi = (int) ri[0];
      if (gi >= gr) continue;  // only the larger id travels, so only it can be the B side

      const double m = MFTL_MAX(ri[8], rr[8]);
      double wh[3];
      for (int k = 0; k < 3; ++k) wh[k] = ri[4 + k] + m;
      if (!interval_overlap(domain, ri + 1, wh, rr + 1, rr + 4)) continue;

      /* Only if something actually arrived. A neighbour whose share of the body
       * is near but whose triangles all missed the per-triangle test leaves no
       * patch, and traversing an empty B side would find nothing at a cost. */
      if (MFTLSpatialHaloPatch(halo, gr) == MFTL_NULL && MFTLEnsembleFindLocalSlot(ens, gr) == MFTL_ERROR) continue;

      halo_pair_add(halo, gi, gr);
    }
  }

  ib_free(&cand);
  grid_free(&grid);
  halo_pairs_finish(halo);
}

/**
 * @brief The implementation behind both public forms of the halo exchange.
 *
 * @param ens        The ensemble.
 * @param requests   [n_requests] explicit pairs, or #MFTL_NULL in @p near mode.
 * @param n_requests Length of @p requests.
 * @param tolerance  `[n_local_objects]` contact envelope per local slot, or
 *                   #MFTL_NULL for zero. Read only in @p near mode.
 * @param halo       Filled in.
 * @param near       #MFTL_TRUE to derive the pairs from the extents, #MFTL_FALSE
 *                   to take them from @p requests.
 * @return #MFTL_SUCCESS, or #MFTL_ERROR on a NULL argument.
 */
static int extend_impl(MFTL_Ensemble *ens, const MFTL_SpatialHaloRequest *requests, int n_requests, const double *tolerance, MFTL_SpatialHalo *halo, bool near)
{
  if (ens == MFTL_NULL || halo == MFTL_NULL) return MFTL_ERROR;
  if (n_requests < 0) return MFTL_ERROR;
  if (!near && n_requests > 0 && requests == MFTL_NULL) return MFTL_ERROR;

  MFTL_Domain *const domain = &ens->domain;
  const int nproc = domain->nproc;
  const int nn = domain->num_neighbors;

  halo_sync_bodies(halo, ens);
  const int nb = ens->n_local_objects;
  if (near) halo->n_pairs = 0;

  MFTL_ASSERT(MFTL_MPI_PackedVertex != MPI_DATATYPE_NULL && "MFTLExtendSpatialHalo: call MFTLRegisterMpiTypes() first");

  /* ── 0. This rank's own record for every body it holds ───────────────────
   *
   * Ascending by body id, because `local_objects` is: the neighbour blocks
   * arrive in that order too, which is what makes a partner lookup a binary
   * search over a sorted run rather than a table indexed by the id space. */
  double *const my_rec = (double *) MFTLMallocOrDie((size_t) MFTL_MAX(nb, 1) * MFTL_HALO_REC * sizeof(double), "spatial halo body records");
  int n_my = 0;

  for (int o = 0; o < nb; ++o)
  {
    double c[3], h[3];
    owned_interval(ens->local_objects[o], domain, c, h);
    if (interval_is_empty(h)) continue;  // owns nothing of it: there is nothing to send and nothing to be near

    const int e = MFTLSpatialHaloFindBody(halo, ens->local_objects[o]->global_id);
    double *const r = &my_rec[(size_t) n_my * MFTL_HALO_REC];
    r[0] = (double) ens->local_objects[o]->global_id;
    for (int k = 0; k < 3; ++k)
    {
      r[1 + k] = c[k];
      r[4 + k] = h[k];
    }
    r[7] = (e != MFTL_ERROR) ? halo->global_radius[e] : 0.0;
    r[8] = (tolerance != MFTL_NULL) ? tolerance[o] : 0.0;
    n_my++;
  }

  /* ── 1. Discard what this call is about to replace ───────────────────────
   *
   * The near-bodies form owns the whole directory, so every patch goes: last
   * call's patches describe where the bodies were a timestep ago, and keeping
   * one would report contacts against a surface nobody occupies any more. The
   * request form only speaks for the bodies it was asked about. */
  if (near)
  {
    for (int e = 0; e < halo->num_objects; ++e)
    {
      if (halo->patch[e] != MFTL_NULL) MFTLObjectDestroy(&halo->patch[e]);
    }
  }
  else
  {
    for (int r = 0; r < n_requests; ++r)
    {
      const int e = MFTLSpatialHaloFindBody(halo, requests[r].body_b);
      if (e != MFTL_ERROR && halo->patch[e] != MFTL_NULL) MFTLObjectDestroy(&halo->patch[e]);
    }
  }

  /* ── 2. Publish, non-blocking ────────────────────────────────────────────
   *
   * NON-BLOCKING because there is exactly one piece of real work that does not
   * depend on the answer, and it is proportional to the mesh rather than to the
   * neighbour count: building each body's per-triangle intervals. Nothing after
   * that can start early -- a triangle cannot be classified before the extents
   * arrive, packed before it is classified, or sent before it is packed.
   *
   * Two collectives rather than one, and for the same reason the per-body
   * reduction takes two: only the holder of a body knows it holds it, so the
   * lengths cannot be predicted from the id space. */
  int *const rcount = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(nn, 1), sizeof(int), "spatial halo record counts");
  int *const rdispl = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(nn, 1), sizeof(int), "spatial halo record displacements");
  double *nb_rec = MFTL_NULL;
  int n_nb_rec = 0;
  MPI_Request rec_req = MPI_REQUEST_NULL;
  const int scount = n_my * MFTL_HALO_REC;
  const bool exchanging = (nproc > 1 && nn > 0);

  if (exchanging)
  {
    MPI_Neighbor_allgather(&scount, 1, MPI_INT, rcount, 1, MPI_INT, domain->comm_graph);

    size_t total = 0;
    for (int i = 0; i < nn; ++i)
    {
      rdispl[i] = (int) total;
      total += (size_t) rcount[i];
    }
    MFTL_ASSERT(total <= (size_t) 0x7fffffff && "MFTLExtendSpatialHalo: neighbour record payload exceeds an MPI int count");

    nb_rec = (double *) MFTLMallocOrDie(MFTL_MAX(total, (size_t) 1) * sizeof(double), "spatial halo neighbour records");
    n_nb_rec = (int) (total / MFTL_HALO_REC);
    MPI_Ineighbor_allgatherv(my_rec, scount, MPI_DOUBLE, nb_rec, rcount, rdispl, MPI_DOUBLE, domain->comm_graph, &rec_req);
  }

  /* ── 3. Which of my bodies can travel, and how far a margin each needs ────
   *
   * Indexed by LOCAL SLOT. A request naming a body this rank does not hold is
   * silently dropped: there is nothing here to send, and the rank that does hold
   * it is making the same decision from its own list. */
  int *const req_off = (int *) MFTLCallocOrDie((size_t) nb + 1, sizeof(int), "spatial halo partner offsets");
  int *req_list = MFTL_NULL;
  double *const margin_of = (double *) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(double), "spatial halo margins");
  bool *const is_b = (bool *) MFTLCallocOrDie((size_t) MFTL_MAX(nb, 1), sizeof(bool), "spatial halo B flags");

  if (near)
  {
    for (int o = 0; o < nb; ++o) is_b[o] = MFTL_TRUE;  // any body is B for a smaller id
  }
  else
  {
    for (int r = 0; r < n_requests; ++r)
    {
      const int ob = MFTLEnsembleFindLocalSlot(ens, requests[r].body_b);
      if (ob == MFTL_ERROR) continue;
      is_b[ob] = MFTL_TRUE;
      req_off[ob + 1]++;
      /* One body can be B in several pairs with different margins. The LARGEST
       * wins: a triangle wanted by any partner must be sent, and using the
       * smallest would silently drop contacts for the pair that asked for more. */
      if (requests[r].margin > margin_of[ob]) margin_of[ob] = requests[r].margin;
    }
    for (int o = 0; o < nb; ++o) req_off[o + 1] += req_off[o];

    req_list = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(req_off[nb], 1) * sizeof(int), "spatial halo partner list");
    int *const cursor = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nb, 1) * sizeof(int), "spatial halo cursors");
    for (int o = 0; o < nb; ++o) cursor[o] = req_off[o];
    for (int r = 0; r < n_requests; ++r)
    {
      const int ob = MFTLEnsembleFindLocalSlot(ens, requests[r].body_b);
      if (ob == MFTL_ERROR) continue;
      req_list[cursor[ob]++] = requests[r].body_a;
    }
    free(cursor);
  }

  /* ── 4. Per-triangle intervals, WHILE the records fly ────────────────────
   *
   * Built ONCE per body and reused for every neighbour and every partner, and
   * only for the bodies that can actually travel. The margin is NOT folded in
   * here, because in the near-bodies form it is not known until the partners
   * are: it is folded into each PARTNER's half-width instead, once per
   * (body, neighbour, partner), so the inner test is still a bare distance
   * compare. */
  int *const box_off = (int *) MFTLCallocOrDie((size_t) nb + 1, sizeof(int), "spatial halo box offsets");
  for (int o = 0; o < nb; ++o) box_off[o + 1] = box_off[o] + (is_b[o] ? ens->local_objects[o]->tris.n_local : 0);

  double *const boxes = (double *) MFTLMallocOrDie(6 * (size_t) MFTL_MAX(box_off[nb], 1) * sizeof(double), "spatial halo triangle intervals");

  if (exchanging)
  {
    for (int o = 0; o < nb; ++o)
    {
      if (!is_b[o]) continue;

      const MFTL_Object *const obj = ens->local_objects[o];
      double *const bx = &boxes[6 * (size_t) box_off[o]];
      const int nt = obj->tris.n_local;

      for (int t = 0; t < nt; ++t)
      {
        double lo[3], hi[3];
        MFTLGetTriangleBounds(obj, t, lo, hi);
        for (int k = 0; k < 3; ++k)
        {
          bx[6 * t + k] = wrap_axis(domain, k, 0.5 * (lo[k] + hi[k]));
          bx[6 * t + 3 + k] = 0.5 * (hi[k] - lo[k]);
        }
      }
    }
    MPI_Wait(&rec_req, MPI_STATUS_IGNORE);
  }

  /* ── 5. Classify and pack ────────────────────────────────────────────────
   *
   * `margin_of[o]` is the widest envelope any matched partner asked for, and it
   * is what the triangle boxes are compared with. Taking the maximum rather than
   * the exact per-pair value can only send MORE, which is the safe direction:
   * the pre-filter a triangle test stands in for must never be stricter than the
   * test itself. */
  MFTL_SendBuffer *const send_t = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "spatial halo triangle buffers");
  MFTL_SendBuffer *const send_v = (MFTL_SendBuffer *) MFTLCallocOrDie((size_t) nproc, sizeof(MFTL_SendBuffer), "spatial halo vertex buffers");

  if (exchanging)
  {
    /* Which neighbour block each arriving record came from, so a match knows
     * where to send. One int per record, filled once. */
    int *const rec_src = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_nb_rec, 1) * sizeof(int), "spatial halo record sources");
    for (int i = 0; i < nn; ++i)
    {
      const int first = rdispl[i] / MFTL_HALO_REC;
      const int last = first + rcount[i] / MFTL_HALO_REC;
      for (int r = first; r < last; ++r) rec_src[r] = i;
    }

    /* The grid over the arriving records. Its cell has to cover the widest body
     * on either side plus the widest envelope, or "within one cell" would stop
     * being a superset of "within reach". */
    body_grid grid;
    {
      double reach[3] = {0.0, 0.0, 0.0};
      double max_tol = 0.0;
      for (int r = 0; r < n_nb_rec; ++r)
      {
        const double *const rc = &nb_rec[(size_t) r * MFTL_HALO_REC];
        for (int k = 0; k < 3; ++k) reach[k] = MFTL_MAX(reach[k], 2.0 * rc[4 + k]);
        max_tol = MFTL_MAX(max_tol, rc[8]);
      }
      for (int o = 0; o < n_my; ++o)
      {
        const double *const rc = &my_rec[(size_t) o * MFTL_HALO_REC];
        for (int k = 0; k < 3; ++k) reach[k] = MFTL_MAX(reach[k], 2.0 * rc[4 + k]);
        max_tol = MFTL_MAX(max_tol, rc[8]);
      }
      for (int r = 0; r < n_requests; ++r) max_tol = MFTL_MAX(max_tol, requests[r].margin);
      for (int k = 0; k < 3; ++k) reach[k] += max_tol;

      grid_build(&grid, domain, (n_nb_rec > 0) ? nb_rec + 1 : MFTL_NULL, MFTL_HALO_REC, n_nb_rec, reach);
    }

    /* Matches for one body, grouped by neighbour: `(record index)` runs with
     * `mat_off[nidx]`. Reused across bodies rather than reallocated. */
    mftl_int_buffer cand;
    ib_init(&cand, 64);
    int *const mat_cnt = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(nn, 1), sizeof(int), "spatial halo match counts");
    int *const mat_off = (int *) MFTLCallocOrDie((size_t) MFTL_MAX(nn, 1) + 1, sizeof(int), "spatial halo match offsets");
    mftl_int_buffer matched;
    ib_init(&matched, 64);
    mftl_int_buffer grouped;
    ib_init(&grouped, 64);

    int max_nv = 0;
    for (int o = 0; o < nb; ++o) max_nv = MFTL_MAX(max_nv, ens->local_objects[o]->verts.n);
    unsigned *const vstamp = (unsigned *) MFTLCallocOrDie((size_t) MFTL_MAX(max_nv, 1), sizeof(unsigned), "spatial halo vertex stamps");
    unsigned vepoch = 0u;

    /* One partner extent per match on ONE neighbour. Near mode bounds that by
     * that neighbour's record count; the request form bounds it by how many
     * times the caller named this body as B, and a caller is allowed to name the
     * same pair twice. */
    double *const ph =
        (double *) MFTLMallocOrDie(6 * (size_t) MFTL_MAX(MFTL_MAX(n_nb_rec, n_requests), 1) * sizeof(double), "spatial halo partner extents");

    for (int o = 0; o < nb; ++o)
    {
      if (!is_b[o]) continue;

      const MFTL_Object *const obj = ens->local_objects[o];
      const int nt = obj->tris.n_local;
      if (nt <= 0) continue;

      const int gb = obj->global_id;

      double my_c[3], my_h[3];
      owned_interval(obj, domain, my_c, my_h);
      if (interval_is_empty(my_h)) continue;

      /* ── Find the partners ──────────────────────────────────────────────
       *
       * NEAR: every arriving record for a SMALLER body id whose extent is within
       * the pair's envelope. The id order is the "reported once" rule -- the
       * holder of the smaller id is the rank that reports the pair, so it is the
       * rank that needs the copy, and sending both ways would report every
       * cross-rank contact twice.
       *
       * REQUEST: the A bodies the caller named, looked up in each neighbour's
       * sorted run. */
      ib_reset(&matched);
      for (int i = 0; i < nn; ++i) mat_cnt[i] = 0;

      const double my_tol = (tolerance != MFTL_NULL) ? tolerance[o] : 0.0;

      if (near)
      {
        ib_reset(&cand);
        grid_collect(&grid, domain, my_c, &cand);

        for (int ci = 0; ci < cand.size; ++ci)
        {
          const int r = cand.data[ci];
          const double *const rc = &nb_rec[(size_t) r * MFTL_HALO_REC];
          const int ga = (int) rc[0];
          if (ga >= gb) continue;  // only the larger id travels
          if (domain->neighbors[rec_src[r]] == domain->rank) continue;  // a thin periodic grid can name this rank

          const double m = MFTL_MAX(my_tol, rc[8]);
          double wh[3];
          for (int k = 0; k < 3; ++k) wh[k] = my_h[k] + m;
          if (!interval_overlap(domain, my_c, wh, rc + 1, rc + 4)) continue;

          ib_add(&matched, r);
          mat_cnt[rec_src[r]]++;
          if (m > margin_of[o]) margin_of[o] = m;
        }
      }
      else
      {
        double wh[3];
        for (int k = 0; k < 3; ++k) wh[k] = my_h[k] + margin_of[o];

        for (int i = 0; i < nn; ++i)
        {
          if (domain->neighbors[i] == domain->rank) continue;

          const int first = rdispl[i] / MFTL_HALO_REC;
          const int nrec = rcount[i] / MFTL_HALO_REC;

          for (int pi = req_off[o]; pi < req_off[o + 1]; ++pi)
          {
            const int ga = req_list[pi];

            /* Binary search the neighbour's run, which is ascending by body id
             * because every rank publishes its local directory in order. */
            int lo = 0, hi = nrec - 1, found = MFTL_ERROR;
            while (lo <= hi)
            {
              const int mid = lo + ((hi - lo) >> 1);
              const int g = (int) nb_rec[(size_t) (first + mid) * MFTL_HALO_REC];
              if (g == ga)
              {
                found = first + mid;
                break;
              }
              if (g < ga)
                lo = mid + 1;
              else
                hi = mid - 1;
            }
            if (found == MFTL_ERROR) continue;

            const double *const rc = &nb_rec[(size_t) found * MFTL_HALO_REC];
            if (!interval_overlap(domain, my_c, wh, rc + 1, rc + 4)) continue;

            ib_add(&matched, found);
            mat_cnt[i]++;
          }
        }
      }

      if (matched.size == 0) continue;

      /* Group the matches by neighbour so that each destination packs each
       * triangle at most once -- a triangle wanted by two partners on the same
       * rank must not be sent twice, or the contact is reported twice. */
      mat_off[0] = 0;
      for (int i = 0; i < nn; ++i) mat_off[i + 1] = mat_off[i] + mat_cnt[i];

      ib_reset(&grouped);
      for (int i = 0; i < matched.size; ++i) ib_add(&grouped, 0);
      {
        int *const cur = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(nn, 1) * sizeof(int), "spatial halo group cursors");
        for (int i = 0; i < nn; ++i) cur[i] = mat_off[i];
        for (int i = 0; i < matched.size; ++i) grouped.data[cur[rec_src[matched.data[i]]]++] = matched.data[i];
        free(cur);
      }

      const double m = margin_of[o];
      const double *const bx = &boxes[6 * (size_t) box_off[o]];

      for (int i = 0; i < nn; ++i)
      {
        const int np = mat_off[i + 1] - mat_off[i];
        if (np == 0) continue;

        const int dest = domain->neighbors[i];

        /* One partner extent per match, with the margin folded into the
         * HALF-WIDTH so the triangle loop below is a bare distance compare. */
        for (int j = 0; j < np; ++j)
        {
          const double *const rc = &nb_rec[(size_t) grouped.data[mat_off[i] + j] * MFTL_HALO_REC];
          for (int k = 0; k < 3; ++k)
          {
            ph[6 * j + k] = rc[1 + k];
            ph[6 * j + 3 + k] = rc[4 + k] + m;
          }
        }

        vepoch++;
        if (vepoch == 0u)
        {
          memset(vstamp, 0, (size_t) MFTL_MAX(max_nv, 1) * sizeof(unsigned));
          vepoch = 1u;
        }

        for (int t = 0; t < nt; ++t)
        {
          bool wanted = MFTL_FALSE;
          for (int j = 0; j < np; ++j)
          {
            if (interval_overlap(domain, &bx[6 * t], &bx[6 * t + 3], &ph[6 * j], &ph[6 * j + 3]))
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
          pt->body_gid = gb;
          for (int k = 0; k < 3; ++k)
          {
            const MFTL_HalfEdge *const he = &obj->tris.he[3 * t + k];
            /* The TARGET vertex of half-edge 3t+k -- not tris.verts[3t+k], which
             * is its ORIGIN. The two differ by a rotation. */
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
            pv->body_gid = gb;
          }
        }
      }
    }

    free(ph);
    free(vstamp);
    ib_free(&grouped);
    ib_free(&matched);
    ib_free(&cand);
    free(mat_off);
    free(mat_cnt);
    grid_free(&grid);
    free(rec_src);
  }

  free(boxes);
  free(box_off);
  free(is_b);
  free(margin_of);
  free(req_list);
  free(req_off);

  /* ── 6. Exchange and rebuild ─────────────────────────────────────────────── */
  if (exchanging)
  {
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

    const MFTL_PackedVertex *const av = (const MFTL_PackedVertex *) x[0].recv_buf;
    const MFTL_PackedTriangleComplex *const at = (const MFTL_PackedTriangleComplex *) x[1].recv_buf;

    /* Which bodies arrived, as a sorted unique list. It is short -- a rank is
     * near a handful of partners -- so the arrivals are bucketed by a position
     * in THAT list rather than by body id, which is the whole reason the
     * grouping pass no longer costs an array per body in the ensemble. */
    int *blist = MFTL_NULL;
    int n_blist = 0, cap_blist = 0;
    for (int i = 0; i < x[1].recv_count; ++i) (void) sorted_insert(&blist, &n_blist, &cap_blist, at[i].body_gid);
    for (int i = 0; i < x[0].recv_count; ++i) (void) sorted_insert(&blist, &n_blist, &cap_blist, av[i].body_gid);

    /* Every entry is created BEFORE any slot is looked up: halo_entry() is an
     * ordered insert, so it moves the slots of everything after it. */
    for (int i = 0; i < n_blist; ++i) (void) halo_entry(halo, blist[i]);

    int *const entry_of = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_blist, 1) * sizeof(int), "spatial halo arrival entries");
    for (int i = 0; i < n_blist; ++i) entry_of[i] = MFTLSpatialHaloFindBody(halo, blist[i]);

    /* Counting sort by body: the stream arrives grouped by SENDER, and a patch
     * has to see every contribution to its body at once. */
    int *const v_off = (int *) MFTLCallocOrDie((size_t) n_blist + 1, sizeof(int), "spatial halo vertex grouping");
    int *const t_off = (int *) MFTLCallocOrDie((size_t) n_blist + 1, sizeof(int), "spatial halo triangle grouping");

    for (int i = 0; i < x[0].recv_count; ++i) v_off[sorted_find(blist, n_blist, av[i].body_gid) + 1]++;
    for (int i = 0; i < x[1].recv_count; ++i) t_off[sorted_find(blist, n_blist, at[i].body_gid) + 1]++;
    for (int i = 0; i < n_blist; ++i)
    {
      v_off[i + 1] += v_off[i];
      t_off[i + 1] += t_off[i];
    }

    MFTL_PackedVertex *const sv =
        (MFTL_PackedVertex *) MFTLMallocOrDie((size_t) MFTL_MAX(x[0].recv_count, 1) * sizeof(MFTL_PackedVertex), "spatial halo vsort");
    MFTL_PackedTriangleComplex *const st =
        (MFTL_PackedTriangleComplex *) MFTLMallocOrDie((size_t) MFTL_MAX(x[1].recv_count, 1) * sizeof(MFTL_PackedTriangleComplex), "spatial halo tsort");

    {
      int *const cv = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_blist, 1) * sizeof(int), "spatial halo vcursor");
      int *const ct = (int *) MFTLMallocOrDie((size_t) MFTL_MAX(n_blist, 1) * sizeof(int), "spatial halo tcursor");
      for (int i = 0; i < n_blist; ++i)
      {
        cv[i] = v_off[i];
        ct[i] = t_off[i];
      }
      for (int i = 0; i < x[0].recv_count; ++i) sv[cv[sorted_find(blist, n_blist, av[i].body_gid)]++] = av[i];
      for (int i = 0; i < x[1].recv_count; ++i) st[ct[sorted_find(blist, n_blist, at[i].body_gid)]++] = at[i];
      free(cv);
      free(ct);
    }

    mftl_gid_table tbl;
    gt_init(&tbl);
    mftl_int_buffer scratch;
    ib_init(&scratch, 1024);

    for (int i = 0; i < n_blist; ++i)
    {
      rebuild_patch(halo, entry_of[i], &sv[v_off[i]], v_off[i + 1] - v_off[i], &st[t_off[i]], t_off[i + 1] - t_off[i], &tbl, &scratch);
    }

    /* A patch is this rank's only handle on a body it holds nothing of, so the
     * radius it needs for its envelope has to come off the wire. The sender
     * published it in the same record that made the pair a candidate. */
    for (int r = 0; r < n_nb_rec; ++r)
    {
      const double *const rc = &nb_rec[(size_t) r * MFTL_HALO_REC];
      const int e = MFTLSpatialHaloFindBody(halo, (int) rc[0]);
      if (e == MFTL_ERROR) continue;
      if (MFTLEnsembleFindLocalSlot(ens, (int) rc[0]) != MFTL_ERROR) continue;  // held here: the reduction already answered
      halo->global_radius[e] = MFTL_MAX(halo->global_radius[e], rc[7]);
    }

    ib_free(&scratch);
    gt_free(&tbl);
    free(sv);
    free(st);
    free(v_off);
    free(t_off);
    free(entry_of);
    free(blist);
    free(x[0].recv_buf);
    free(x[1].recv_buf);
    MFTLExchangeFree(&ctx);
  }

  free(send_v);
  free(send_t);

  /* ── 7. The candidate pairs, if the caller asked us to decide them ───────── */
  if (near) build_near_pairs(ens, halo, my_rec, n_my, nb_rec, n_nb_rec);

  free(nb_rec);
  free(rdispl);
  free(rcount);
  free(my_rec);

  /* Both forms, not just the near one: the directory picks up an entry for every
   * body a neighbour published a radius for, and over a run of a droplet
   * ensemble those would accumulate every body that ever drifted past. Pruning
   * to "held here, or a patch arrived" is what keeps the directory proportional
   * to what is actually nearby. */
  halo_prune(halo, ens);
  return MFTL_SUCCESS;
}

int MFTLExtendSpatialHalo(MFTL_Ensemble *ens, const MFTL_SpatialHaloRequest *requests, int n_requests, MFTL_SpatialHalo *halo)
{
  return extend_impl(ens, requests, n_requests, MFTL_NULL, halo, MFTL_FALSE);
}

int MFTLExtendSpatialHaloNearBodies(MFTL_Ensemble *ens, const double *tolerance, MFTL_SpatialHalo *halo)
{
  return extend_impl(ens, MFTL_NULL, 0, tolerance, halo, MFTL_TRUE);
}

