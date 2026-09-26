/**
 * @file topo_detect.c
 * @author Christian Gorges
 * @date 2026-09-25
 * @ingroup mftl_topology_change
 *
 * @brief Detection: which pairs of triangles are a thin feature, and of which kind.
 *
 * ## THE PREDICATE
 *
 * Two triangles A and B -- of one body or of two -- are an EVENT when
 *
 *   1. they share no vertex (MFTLEnsembleIntersectObjects' self filter);
 *   2. their unit normals OPPOSE: n_A . n_B <= -cos(theta);
 *   3. each lies on the same side of the other, as seen along its normal:
 *        s_A = n_A . (c_B - c_A)   and   s_B = n_B . (c_A - c_B)
 *      both negative means each is BEHIND the other -- the gap between them is
 *      inside the body, a thread or sheet: BREAKUP. Both positive means each is
 *      in FRONT of the other -- the gap is outside, a film: COALESCENCE. Mixed
 *      signs are a fold seen edge-on, not a thin feature, and are not events;
 *   4. their distance is within the threshold of THAT kind.
 *
 * Condition 3 is what the original design's detection criteria left implicit, and it is
 * the one that makes the same machinery serve both directions of a topology
 * change: the RT cap rolling onto its own stem and a jet's neck thinning are
 * both "two opposing triangles within delta", and only the side the gap is on
 * tells them apart.
 *
 * ## THE CANONICAL LIST
 *
 * Detection is local and many pairs describe one feature -- a neck is found
 * once per triangle pair around it, and a pair across a rank boundary by both
 * ranks' owners in general. Every event point is therefore snapped to a cell of
 * a quarter of max(smaller active threshold, dx) -- the lattice spacing too,
 * because a bias bump is at least 1.5 dx wide and events closer than that are
 * one feature (a quantum on the threshold alone shredded one jet neck into 779
 * events) -- and events are merged on (bodies, kind, cell),
 * keeping the smallest gap. After the gather, the same canonicalisation over the
 * whole list makes it a pure function of the geometry: identical on every rank
 * and at every process count.
 */

#include "topo_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_bvh.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_memory.h"

/** @brief State the pair callback needs. */
typedef struct
{
  const MFTL_Domain *dom;
  double cos_tol;
  double delta_b;
  double delta_c;
  MFTL_TopoEventList *out;
} detect_ctx;

static void push_event(MFTL_TopoEventList *l, const MFTL_TopologyEvent *e)
{
  if (l->n == l->cap)
  {
    l->cap = l->cap ? 2 * l->cap : 64;
    l->e = (MFTL_TopologyEvent *) MFTLReallocOrDie(l->e, (size_t) l->cap * sizeof(MFTL_TopologyEvent), "topology events");
  }
  l->e[l->n++] = *e;
}

/** @brief Unit normal and centroid of corners @p T; returns false for a degenerate triangle. */
static bool normal_and_centre(const double T[3][3], double n[3], double c[3])
{
  const double e1[3] = {T[1][0] - T[0][0], T[1][1] - T[0][1], T[1][2] - T[0][2]};
  const double e2[3] = {T[2][0] - T[0][0], T[2][1] - T[0][1], T[2][2] - T[0][2]};
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];
  const double mag = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  if (!(mag > 0.0)) return MFTL_FALSE;
  for (int k = 0; k < 3; ++k)
  {
    n[k] /= mag;
    c[k] = (T[0][k] + T[1][k] + T[2][k]) / 3.0;
  }
  return MFTL_TRUE;
}

static void detect_pair(const MFTL_BVH *bvhA, int pa, const MFTL_BVH *bvhB, int pb, void *ud)
{
  detect_ctx *const ctx = (detect_ctx *) ud;

  double TA[3][3], TB[3][3], nA[3], nB[3], cA[3], cB[3];
  MFTL_BVHPrimVerts(bvhA, pa, TA);
  MFTL_BVHPrimVerts(bvhB, pb, TB);
  if (!normal_and_centre(TA, nA, cA) || !normal_and_centre(TB, nB, cB)) return;

  /* B in the periodic image nearest A: the traversal tried every image, but the
   * callback is handed the stored corners. */
  double d[3];
  MFTLGetMinimumImageVector(ctx->dom, d, cA, cB);
  const double shift[3] = {cA[0] + d[0] - cB[0], cA[1] + d[1] - cB[1], cA[2] + d[2] - cB[2]};
  for (int j = 0; j < 3; ++j)
    for (int k = 0; k < 3; ++k) TB[j][k] += shift[k];

  if (nA[0] * nB[0] + nA[1] * nB[1] + nA[2] * nB[2] > -ctx->cos_tol) return;

  const double sA = nA[0] * d[0] + nA[1] * d[1] + nA[2] * d[2];
  const double sB = -(nB[0] * d[0] + nB[1] * d[1] + nB[2] * d[2]);

  int kind;
  double delta;
  if (sA < 0.0 && sB < 0.0)
  {
    kind = MFTL_TOPOLOGY_BREAKUP;
    delta = ctx->delta_b;
  }
  else if (sA > 0.0 && sB > 0.0)
  {
    kind = MFTL_TOPOLOGY_COALESCENCE;
    delta = ctx->delta_c;
  }
  else
    return;
  if (!(delta > 0.0)) return;

  const double gap = sqrt(MFTL_TriTriDistanceSq(TA, TB));
  if (gap > delta) return;

  int ta, tb;
  const MFTL_Object *const oa = MFTL_BVHPrimTriangle(bvhA, pa, &ta);
  const MFTL_Object *const ob = MFTL_BVHPrimTriangle(bvhB, pb, &tb);

  MFTL_TopologyEvent ev;
  ev.body_a = MFTL_MIN(oa->global_id, ob->global_id);
  ev.body_b = MFTL_MAX(oa->global_id, ob->global_id);
  ev.kind = kind;
  ev.gap = gap;
  for (int k = 0; k < 3; ++k) ev.point[k] = cA[k] + 0.5 * d[k];
  MFTLApplyPeriodicity(ctx->dom, ev.point);
  push_event(ctx->out, &ev);
}

/**
 * @brief The cheap half of the predicate, run before the exact narrow phase:
 *        only OPPOSING normals can be an event.
 *
 * On a smooth surface nearly every triangle within the envelope of another has
 * a nearly parallel normal, and the exact proximity test costs two orders of
 * magnitude more than this dot product. See MFTL_BVHBuildParams::pair_filter.
 */
static bool opposing_normals(const MFTL_BVH *bvhA, int pa, const MFTL_BVH *bvhB, int pb, void *ud)
{
  const detect_ctx *const ctx = (const detect_ctx *) ud;
  double TA[3][3], TB[3][3], nA[3], nB[3], cA[3], cB[3];
  MFTL_BVHPrimVerts(bvhA, pa, TA);
  MFTL_BVHPrimVerts(bvhB, pb, TB);
  if (!normal_and_centre(TA, nA, cA) || !normal_and_centre(TB, nB, cB)) return MFTL_FALSE;
  return nA[0] * nB[0] + nA[1] * nB[1] + nA[2] * nB[2] <= -ctx->cos_tol;
}

int MFTLTopoDetectLocal(MFTL_Ensemble *ens, const MFTL_TopologyOptions *opts, MFTL_TopoEventList *out)
{
  detect_ctx ctx;
  ctx.dom = &ens->domain;
  ctx.cos_tol = cos(opts->normal_tol_deg * MFTL_PI / 180.0);
  ctx.delta_b = opts->delta_breakup;
  ctx.delta_c = opts->delta_coalescence;
  ctx.out = out;

  /* The envelope is ABSOLUTE and the same for every body: a thread breaks at
   * delta whatever the size of the drop it belongs to. */
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 0.0;
  p.proximity_floor = MFTL_MAX(opts->delta_breakup, opts->delta_coalescence);
  p.self_pairs = MFTL_TRUE;
  p.pair_filter = opposing_normals;

  return MFTLEnsembleIntersectObjects(ens, &p, detect_pair, &ctx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Canonical form
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Canonical order: bodies, kind, snapped point, then gap.
 *
 * After snapping, two events describing one cell carry BIT-identical points
 * (the same arithmetic on the same integers), so the point itself is the merge
 * key and the comparator needs no state.
 */
static int cmp_event(const void *x, const void *y)
{
  const MFTL_TopologyEvent *a = (const MFTL_TopologyEvent *) x, *b = (const MFTL_TopologyEvent *) y;
  if (a->body_a != b->body_a) return (a->body_a < b->body_a) ? -1 : 1;
  if (a->body_b != b->body_b) return (a->body_b < b->body_b) ? -1 : 1;
  if (a->kind != b->kind) return (a->kind < b->kind) ? -1 : 1;
  for (int k = 0; k < 3; ++k)
    if (a->point[k] != b->point[k]) return (a->point[k] < b->point[k]) ? -1 : 1;
  if (a->gap != b->gap) return (a->gap < b->gap) ? -1 : 1;
  return 0;
}

void MFTLTopoCanonicaliseEvents(const MFTL_Domain *dom, double quantum, MFTL_TopoEventList *list)
{
  if (list->n == 0) return;

  /* Snap each point to the centre of its cell: the point then no longer depends
   * on which of the pairs describing one feature was found first. */
  for (int i = 0; i < list->n; ++i)
  {
    for (int k = 0; k < 3; ++k)
    {
      const double c = floor((list->e[i].point[k] - dom->global_min[k]) / quantum);
      list->e[i].point[k] = dom->global_min[k] + (c + 0.5) * quantum;
    }
    MFTLApplyPeriodicity(dom, list->e[i].point);
  }

  qsort(list->e, (size_t) list->n, sizeof(MFTL_TopologyEvent), cmp_event);

  /* Same bodies, kind and cell: keep the first, which has the smallest gap. */
  int w = 0;
  for (int i = 0; i < list->n; ++i)
  {
    if (w > 0)
    {
      const MFTL_TopologyEvent *p = &list->e[w - 1], *q = &list->e[i];
      if (p->body_a == q->body_a && p->body_b == q->body_b && p->kind == q->kind && p->point[0] == q->point[0] && p->point[1] == q->point[1] &&
          p->point[2] == q->point[2])
        continue;
    }
    list->e[w++] = list->e[i];
  }
  list->n = w;
}
