/**
 * @file remesh_ops.c
 * @author Christian Gorges
 * @date 2026-08-15
 * @ingroup mftl_remeshing
 *
 * @brief The three topological operators and their approval predicates.
 *
 * ## THE SHAPE EVERY OPERATOR HAS, AND WHY
 *
 *   1. READ the patch into locals: vertex indices, rim twin indices, rim flags.
 *   2. ALLOCATE whatever new entities are needed.
 *   3. WRITE the triangles, place the flags, re-link the rim.
 *   4. REPAIR the affected vertices' out_he_idx.
 *
 * Step 1 must complete before step 2 because allocation can move both pool
 * blocks, and step 3 must not begin before step 2 for the same reason. That is
 * the one rule the AoS-in-a-pool layout imposes on a remesher, and it is cheap:
 * a patch is a dozen integers.
 *
 * Step 3 is where implicit indexing shows up. When triangle t is rewritten, the
 * logical edge that occupied slot 1 may end up in slot 2, so the rim FLAGS have
 * to be carried explicitly -- a half-edge record does not move with its edge,
 * only its contents do. MFTL_FLAG_FEATURE and MFTL_FLAG_BOUNDARY are both
 * properties of the EDGE; losing one during a split is invisible until a ridge
 * develops a gap several timesteps later.
 *
 * ## WHY THE PREDICATES ARE THIS LONG
 *
 * Every clause in MFTLCollapseApproved corresponds to a mesh that is still a
 * valid data structure and no longer a valid surface. None of them announce
 * themselves: a violated link condition produces a non-manifold vertex whose
 * one-ring walk silently visits half the fan, and a fold-over produces an
 * inverted triangle that the feature detector then reads as a sharp edge and the
 * smoother locks in place forever. The predicates are the whole correctness
 * argument of the module; the execution functions assume it has been made.
 */

#include "mftl_remeshing.h"

#include <stdlib.h>

#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_object.h"
#include "mftl_topology.h"
#include "remesh_internal.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * Shared predicate helpers
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Is @p h the canonical representative of its undirected edge?
 *
 * Every operator acts on an EDGE, not on a direction, so the sweeps must offer
 * each edge exactly once or an edge gets split twice and a flip gets undone by
 * its own twin. The smaller index wins; a boundary half-edge has no twin and is
 * always canonical.
 *
 * @brief Is @p h the half of its undirected edge the operators are offered?
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` for the half-edge of smaller index, so an undirected edge is
 *         considered exactly once per sweep.
 */
static inline bool he_is_canonical(const MFTL_Object *obj, int h)
{
  const int tw = obj->tris.he[h].twin_idx;
  return (tw == MFTL_ERROR) || (h < tw);
}

/**
 * @brief Common preconditions: the edge and both endpoints are live and untouched.
 *
 * @param obj The mesh.
 * @param h   Half-edge index.
 * @return `true` if @p h is alive, untouched this sweep, and its triangle is too.
 */
static bool edge_is_operable(const MFTL_Object *obj, int h)
{
  if (h < 0 || h >= 3 * obj->tris.n) return MFTL_FALSE;
  if (tri_is_dead(obj, MFTLHeTriIdx(h))) return MFTL_FALSE;
  if (obj->tris.he[h].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  const int a = MFTLHeOrigin(obj, h);
  const int b = MFTLHeTarget(obj, h);
  if (obj->verts.rec[a].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;
  if (obj->verts.rec[b].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) return MFTL_FALSE;

  return MFTL_TRUE;
}

/**
 * @brief One-ring vertices of @p v, refusing rather than truncating.
 * @return The neighbour count, or MFTL_ERROR if the fan did not fit.
 *
 * @param      obj The mesh.
 * @param      v   Vertex index.
 * @param[out] out `[MFTL_REMESH_RING_MAX]` the one-ring neighbours.
 * @return Number of neighbours, or 0 if the walk was truncated -- every operator
 *         REFUSES rather than proceeds on a truncated ring.
 */
static int ring_vertices(const MFTL_Object *obj, int v, int *out)
{
  const int n = MFTLGetVertexOneRingVertices(obj, v, out, MFTL_REMESH_RING_MAX, MFTL_FALSE);
  return (n >= MFTL_REMESH_RING_MAX) ? MFTL_ERROR : n;
}

/**
 * @brief One-ring triangles of @p v, refusing rather than truncating.
 *
 * @param      obj The mesh.
 * @param      v   Vertex index.
 * @param[out] out `[MFTL_REMESH_RING_MAX]` the incident triangles.
 * @return Number of triangles, or 0 if the walk was truncated.
 */
static int ring_triangles(const MFTL_Object *obj, int v, int *out)
{
  const int n = MFTLGetVertexOneRing(obj, v, out, MFTL_REMESH_RING_MAX, MFTL_FALSE);
  return (n >= MFTL_REMESH_RING_MAX) ? MFTL_ERROR : n;
}

/**
 * @brief Number of entries of @p a that also appear in @p b.
 *
 * @param a First sorted-or-not index list.
 * @param na Length of @p a.
 * @param b Second index list.
 * @param nb Length of @p b.
 * @return Number of indices appearing in both. This is the link condition's
 *         counting step: two endpoints must share exactly two neighbours (one at
 *         a mesh boundary edge).
 */
static int count_common(const int *a, int na, const int *b, int nb)
{
  int common = 0;
  for (int i = 0; i < na; ++i)
  {
    for (int j = 0; j < nb; ++j)
    {
      if (a[i] == b[j])
      {
        common++;
        break;
      }
    }
  }
  return common;
}

/**
 * @brief True if @p v lies on a feature curve (ridge vertex or feature corner).
 *
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return `true` if @p v carries #MFTL_FLAG_FEATURE, #MFTL_FLAG_CORNER or
 *         #MFTL_FLAG_BOUNDARY.
 */
static inline bool vertex_on_feature(const MFTL_Object *obj, int v)
{
  return (obj->verts.rec[v].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_CORNER)) != 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SPLIT
 * ═══════════════════════════════════════════════════════════════════════════ */

bool MFTLSplitApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);

  if (!edge_is_operable(obj, h)) return MFTL_FALSE;
  if (!he_is_canonical(obj, h)) return MFTL_FALSE;

  return MFTLGetEdgeLengthSquared(obj, h) > params->max_edge_length_sq;
}

int MFTLSplitEdge(MFTL_Object *obj, int h)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  if (h < 0 || h >= 3 * obj->tris.n) return MFTL_ERROR;
  if (tri_is_dead(obj, MFTLHeTriIdx(h))) return MFTL_ERROR;

  /* ── 1. Read the patch ─────────────────────────────────────────────────────
   *
   * Upper triangle T1 = (A, B, C) with h = A->B; lower T2 = (B, A, D) across the
   * twin, if there is one. Everything captured here is a value: after step 2 the
   * pool blocks may live somewhere else entirely. */
  const MFTL_HalfEdge *he = obj->tris.he;

  const int t1 = MFTLHeTriIdx(h);
  const int hn = MFTLHeNextIdx(h);  // B->C
  const int hp = MFTLHePrevIdx(h);  // C->A

  const int va = he[hp].vertex_idx;
  const int vb = he[h].vertex_idx;
  const int vc = he[hn].vertex_idx;

  const int w_bc = he[hn].twin_idx;
  const int w_ca = he[hp].twin_idx;

  const int f_ab = he[h].flags;
  const int f_bc = he[hn].flags;
  const int f_ca = he[hp].flags;

  const int ht = he[h].twin_idx;

  int t2 = MFTL_ERROR;
  int vd = MFTL_ERROR;
  int w_ad = MFTL_ERROR;
  int w_db = MFTL_ERROR;
  int f_ba = 0;
  int f_ad = 0;
  int f_db = 0;

  if (ht != MFTL_ERROR)
  {
    const int hnt = MFTLHeNextIdx(ht);  // A->D
    const int hpt = MFTLHePrevIdx(ht);  // D->B

    t2 = MFTLHeTriIdx(ht);
    vd = he[hnt].vertex_idx;
    w_ad = he[hnt].twin_idx;
    w_db = he[hpt].twin_idx;
    f_ba = he[ht].flags;
    f_ad = he[hnt].flags;
    f_db = he[hpt].flags;
  }

  double pm[3];
  midpoint_periodic(obj->domain, MFTLVertexPos(obj, va), MFTLVertexPos(obj, vb), pm);

  /* ── 2. Allocate. `he` is dangling from here on. ──────────────────────────── */
  const int vm = remesh_new_vertex(obj, pm);
  const int t3 = remesh_new_triangle(obj);
  const int t4 = (ht != MFTL_ERROR) ? remesh_new_triangle(obj) : MFTL_ERROR;

  MFTL_HalfEdge *const hw = obj->tris.he;

  /* ── 3a. Upper half: T1 becomes (A, M, C), T3 is (M, B, C) ─────────────────
   *
   * The sub-edges A-M and M-B are both halves of the edge that was split, so
   * they inherit its flags -- including MFTL_FLAG_BOUNDARY, which is what keeps
   * an open surface's rim recognisable, and MFTL_FLAG_FEATURE, which is what
   * keeps a ridge continuous across the new vertex. */
  tri_write(obj, t1, va, vm, vc);
  hw[3 * t1 + 0].flags = f_ab;  // A->M
  hw[3 * t1 + 1].flags = 0;  // M->C, brand new and interior by construction
  hw[3 * t1 + 2].flags = f_ca;  // C->A

  tri_write(obj, t3, vm, vb, vc);
  hw[3 * t3 + 0].flags = f_ab;  // M->B
  hw[3 * t3 + 1].flags = f_bc;  // B->C
  hw[3 * t3 + 2].flags = 0;  // C->M

  he_pair(obj, 3 * t1 + 1, 3 * t3 + 2);  // M->C  <->  C->M
  he_pair_outer(obj, 3 * t1 + 2, w_ca);
  he_pair_outer(obj, 3 * t3 + 1, w_bc);

  /* ── 3b. Lower half: T2 becomes (B, M, D), T4 is (M, A, D) ────────────────── */
  if (ht != MFTL_ERROR)
  {
    tri_write(obj, t2, vb, vm, vd);
    hw[3 * t2 + 0].flags = f_ba;  // B->M
    hw[3 * t2 + 1].flags = 0;  // M->D
    hw[3 * t2 + 2].flags = f_db;  // D->B

    tri_write(obj, t4, vm, va, vd);
    hw[3 * t4 + 0].flags = f_ba;  // M->A
    hw[3 * t4 + 1].flags = f_ad;  // A->D
    hw[3 * t4 + 2].flags = 0;  // D->M

    he_pair(obj, 3 * t2 + 1, 3 * t4 + 2);  // M->D  <->  D->M
    he_pair_outer(obj, 3 * t2 + 2, w_db);
    he_pair_outer(obj, 3 * t4 + 1, w_ad);

    /* The split edge itself, now two edges each with two sides. */
    he_pair(obj, 3 * t1 + 0, 3 * t4 + 0);  // A->M  <->  M->A
    he_pair(obj, 3 * t3 + 0, 3 * t2 + 0);  // M->B  <->  B->M
  }
  else
  {
    /* A mesh boundary edge splits into two mesh boundary edges. The flags said so
     * already (f_ab carries MFTL_FLAG_BOUNDARY); only the twins need stating. */
    hw[3 * t1 + 0].twin_idx = MFTL_ERROR;
    hw[3 * t3 + 0].twin_idx = MFTL_ERROR;
    obj->verts.rec[vm].flags |= MFTL_FLAG_BOUNDARY;
  }

  /* ── 4. Outgoing half-edges ───────────────────────────────────────────────── */
  obj->verts.rec[va].out_he_idx = 3 * t1 + 0;  // A->M
  obj->verts.rec[vm].out_he_idx = 3 * t1 + 1;  // M->C
  obj->verts.rec[vc].out_he_idx = 3 * t1 + 2;  // C->A
  obj->verts.rec[vb].out_he_idx = 3 * t3 + 1;  // B->C  (slot 0 is M->B, which starts at M)
  if (ht != MFTL_ERROR) obj->verts.rec[vd].out_he_idx = 3 * t2 + 2;  // D->B

  /* The two halves are exempt from further operations this sweep: without this a
   * long edge is split, and one of its halves -- still longer than the target on
   * a badly graded mesh -- is split again by the same sweep, which is how a
   * remesher produces a local explosion of tiny triangles in one pass. Marking
   * the VERTEX prevents the collapse sweep from removing what this one just made,
   * which is the classic split/collapse ping-pong. */
  hw[3 * t1 + 0].flags |= MFTL_FLAG_VISITED;
  hw[3 * t3 + 0].flags |= MFTL_FLAG_VISITED;
  if (ht != MFTL_ERROR)
  {
    hw[3 * t4 + 0].flags |= MFTL_FLAG_VISITED;
    hw[3 * t2 + 0].flags |= MFTL_FLAG_VISITED;
  }
  obj->verts.rec[vm].flags |= MFTL_FLAG_VISITED;

  /* M inherits the split edge's classification, so the operator's output is
   * self-consistent without waiting for the next MFTLDetectFeatures: splitting a
   * feature edge puts M on that curve with exactly the two sub-edges as its
   * feature edges, which is the definition of a ridge point. Leaving it clear
   * would mark M free, and a free vertex on a feature curve is one the smoother
   * moves off it. */
  if (f_ab & MFTL_FLAG_FEATURE) obj->verts.rec[vm].flags |= MFTL_FLAG_FEATURE;

  MFTLObjectInvalidateDerived(obj);
  return vm;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * COLLAPSE
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Where the surviving vertex ends up.
 *
 * The midpoint, EXCEPT when the survivor lies on preserved geometry and the
 * retired vertex does not. Then it keeps its own position.
 *
 * The exception covers more than corners, and getting its extent wrong is subtle:
 * collapsing an ordinary face vertex INTO a ridge vertex is perfectly legal --
 * the ridge is not being retired -- but placing the survivor at the midpoint
 * drags it OFF the ridge and into the face. Repeated along a sharp edge, that
 * erodes the edge one collapse at a time while every approval clause reports
 * success, because none of them look at where the survivor lands.
 *
 * When BOTH endpoints lie on the same curve the midpoint is used, which is exact
 * on a straight feature and cuts the corner by O(h^2) on a curved one. That is
 * acceptable precisely because the edge being collapsed is short by construction.
 *
 * @param      obj The mesh.
 * @param      a   The vertex that would be retired.
 * @param      b   The vertex that would survive.
 * @param[out] out Where the surviving vertex goes: the midpoint, unless exactly
 *                 one endpoint is a feature or boundary vertex, in which case that
 *                 endpoint's position is kept so the curve is not pulled off itself.
 */
static void collapse_target_position(const MFTL_Object *obj, int a, int b, double out[3])
{
  const bool a_is_pinned = vertex_on_feature(obj, a) || MFTLVertexIsMeshBoundary(obj, a);
  const bool b_is_pinned = vertex_on_feature(obj, b) || MFTLVertexIsMeshBoundary(obj, b);

  /* A corner outranks a ridge point. Collapsing along a feature curve INTO its
   * corner is legal and useful, but the corner is the one thing on that curve
   * that has a definite position -- averaging it with a neighbour rounds it off. */
  if ((obj->verts.rec[b].flags & MFTL_FLAG_CORNER) || (b_is_pinned && !a_is_pinned))
  {
    const double *pb = MFTLVertexPos(obj, b);
    out[0] = pb[0];
    out[1] = pb[1];
    out[2] = pb[2];
    return;
  }

  midpoint_periodic(obj->domain, MFTLVertexPos(obj, a), MFTLVertexPos(obj, b), out);
}

bool MFTLCollapseApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);

  int reason = MFTL_REMESH_REJECT_NONE;
  bool ok = MFTL_FALSE;

  if (!edge_is_operable(obj, h)) goto done;

  const int a = MFTLHeOrigin(obj, h);  // retired
  const int b = MFTLHeTarget(obj, h);  // survives

  /* ── Wanted? ─────────────────────────────────────────────────────────────── */
  if (MFTLGetEdgeLengthSquared(obj, h) >= params->min_edge_length_sq) goto done;

  /* ── Features and boundaries ──────────────────────────────────────────────
   *
   * A corner is defined by being pinned, so it cannot be the vertex that goes.
   * A vertex on a feature curve may only collapse into another one, or the curve
   * is dragged off itself -- and the same statement for a mesh boundary, where
   * the consequence is that the surface's rim moves inwards. */
  reason = MFTL_REMESH_REJECT_FEATURE;
  if (obj->verts.rec[a].flags & MFTL_FLAG_CORNER) goto done;
  if (vertex_on_feature(obj, a) && !vertex_on_feature(obj, b)) goto done;
  if (MFTLVertexIsMeshBoundary(obj, a) && !MFTLVertexIsMeshBoundary(obj, b)) goto done;

  /* ── The link condition FOR CURVES, which is a separate statement ─────────
   *
   * Both endpoints on a feature curve, but the edge between them NOT on it: the
   * edge is a chord across the surface joining two points of the curve, and
   * collapsing it merges them into one. On an open patch that means two rim
   * vertices joined by an interior edge, and the collapse pinches the rim inwards
   * -- the surface grows a serrated notch that nothing reports, because the
   * result is still a perfectly valid manifold with a perfectly valid boundary.
   * On a cube it means two points on DIFFERENT sharp edges, merged into the
   * middle of the face between them.
   *
   * The vertex link condition above cannot see this. It counts shared
   * NEIGHBOURS, and these two have exactly the two they are supposed to have; the
   * degeneracy is in the curve, not in the surface. This is the same clause the
   * classical link condition states by giving a boundary vertex a virtual
   * neighbour outside the mesh. */
  if (!(obj->tris.he[h].flags & MFTL_FLAG_FEATURE) && vertex_on_feature(obj, a) && vertex_on_feature(obj, b)) goto done;
  if (obj->tris.he[h].twin_idx != MFTL_ERROR && MFTLVertexIsMeshBoundary(obj, a) && MFTLVertexIsMeshBoundary(obj, b)) goto done;

  /* ── Link condition ───────────────────────────────────────────────────────
   *
   * |N(a) INTERSECT N(b)| must be exactly 2 (the two wing vertices), or exactly 1
   * at a mesh boundary edge, which has one wing. More than that means a and b are
   * joined by a path OTHER than through their shared triangles, and collapsing
   * welds two sheets of the surface into one non-manifold vertex. The data
   * structure survives it; every subsequent one-ring walk from that vertex
   * silently visits only part of the fan. */
  reason = MFTL_REMESH_REJECT_LINK;

  int ring_a[MFTL_REMESH_RING_MAX];
  int ring_b[MFTL_REMESH_RING_MAX];
  const int na = ring_vertices(obj, a, ring_a);
  const int nb = ring_vertices(obj, b, ring_b);
  if (na == MFTL_ERROR || nb == MFTL_ERROR)
  {
    reason = MFTL_REMESH_REJECT_OTHER;
    goto done;
  }

  const bool boundary_edge = (obj->tris.he[h].twin_idx == MFTL_ERROR);
  const int common = count_common(ring_a, na, ring_b, nb);
  if (common != (boundary_edge ? 1 : 2)) goto done;

  /* ── Valence ──────────────────────────────────────────────────────────────
   *
   * The merged vertex inherits the union of both neighbourhoods minus a and b
   * themselves. Capping it at MFTL_MAX_VALENCE is not a quality preference: the
   * library's fan walkers stop there, so a vertex above it becomes untraversable
   * and every later query about it is quietly truncated. */
  const int merged_valence = na + nb - common - 2;
  if (merged_valence < 3 || merged_valence > MFTL_MAX_VALENCE) goto done;

  /* The wing vertices each lose one incident triangle. Below valence 4 that
   * leaves a vertex with two triangles sharing two edges -- a pocket, not a
   * surface. */
  for (int i = 0; i < na; ++i)
  {
    for (int j = 0; j < nb; ++j)
    {
      if (ring_a[i] != ring_b[j]) continue;

      int ring_w[MFTL_REMESH_RING_MAX];
      const int nw = ring_vertices(obj, ring_a[i], ring_w);
      if (nw == MFTL_ERROR || nw < 4) goto done;
      break;
    }
  }

  /* ── Stretch ──────────────────────────────────────────────────────────────
   *
   * Every edge of the merged vertex must stay inside the band, or the next split
   * sweep undoes this collapse and the two sweeps oscillate for as long as the
   * caller is willing to iterate. */
  reason = MFTL_REMESH_REJECT_OTHER;

  double pm[3];
  collapse_target_position(obj, a, b, pm);

  for (int pass = 0; pass < 2; ++pass)
  {
    const int *ring = pass ? ring_b : ring_a;
    const int n = pass ? nb : na;

    for (int i = 0; i < n; ++i)
    {
      if (ring[i] == a || ring[i] == b) continue;

      double d[3];
      MFTLGetMinimumImageVector(obj->domain, d, pm, MFTLVertexPos(obj, ring[i]));
      if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > params->max_edge_length_sq) goto done;
    }
  }

  /* ── Fold-over ────────────────────────────────────────────────────────────
   *
   * Every triangle that SURVIVES the collapse is re-evaluated with both endpoints
   * moved to the merge point, and its normal must not turn by more than the
   * threshold. The two triangles being removed are skipped -- they degenerate to
   * a line by construction, which is the point of removing them.
   *
   * This clause has no counterpart in the previous version, and its absence there
   * is not a missing nicety but the head of a feedback loop: the inverted
   * triangle a collapse produces presents a ~180 degree dihedral angle, the
   * feature detector reads that as a sharp edge, and the smoother then LOCKS
   * every vertex of it. The inversion becomes permanent, and it seeds more of
   * them on the next pass. */
  reason = MFTL_REMESH_REJECT_FOLD;

  const int t_keep_1 = MFTLHeTriIdx(h);
  const int t_keep_2 = boundary_edge ? MFTL_ERROR : MFTLHeTriIdx(obj->tris.he[h].twin_idx);

  for (int pass = 0; pass < 2; ++pass)
  {
    int fan[MFTL_REMESH_RING_MAX];
    const int nf = ring_triangles(obj, pass ? b : a, fan);
    if (nf == MFTL_ERROR)
    {
      reason = MFTL_REMESH_REJECT_OTHER;
      goto done;
    }

    for (int i = 0; i < nf; ++i)
    {
      const int t = fan[i];
      if (t == t_keep_1 || t == t_keep_2) continue;

      double n_old[3];
      double n_new[3];
      tri_area_vector_substituted(obj, t, MFTL_ERROR, MFTL_ERROR, MFTL_NULL, n_old);
      tri_area_vector_substituted(obj, t, a, b, pm, n_new);

      if (area_vector_cos(n_old, n_new) < params->max_normal_deviation_cos) goto done;
    }
  }

  reason = MFTL_REMESH_REJECT_NONE;
  ok = MFTL_TRUE;

done:
  if (reject_reason) *reject_reason = ok ? MFTL_REMESH_REJECT_NONE : reason;
  return ok;
}

/**
 * @brief Re-links the two rim half-edges of a removed triangle to each other.
 *
 * Removing a triangle merges two of its edges into one, so the neighbours across
 * those edges become directly adjacent. Either side may be absent, in which case
 * the survivor becomes a mesh boundary half-edge and is flagged as one -- the
 * flag is what tells a later pass that this is the edge of the SURFACE rather
 * than of the partition, and it cannot be re-derived after the fact.
 *
 * The feature bit is OR-ed rather than picked, so a ridge that ran through one of
 * the two edges still runs through the merged one.
 *
 * @param obj The mesh.
 * @param x   Rim half-edge just written into a new slot.
 * @param y   Its partner outside the patch, or #MFTL_ERROR.
 */
static void glue_rim(MFTL_Object *obj, int x, int y)
{
  MFTL_HalfEdge *const he = obj->tris.he;

  const int fx = (x != MFTL_ERROR) ? he[x].flags : 0;
  const int fy = (y != MFTL_ERROR) ? he[y].flags : 0;
  const int feature = (fx | fy) & MFTL_FLAG_FEATURE;

  if (x != MFTL_ERROR)
  {
    he[x].twin_idx = y;
    he[x].flags |= feature;
    if (y == MFTL_ERROR) he[x].flags |= MFTL_FLAG_BOUNDARY;
  }
  if (y != MFTL_ERROR)
  {
    he[y].twin_idx = x;
    he[y].flags |= feature;
    if (x == MFTL_ERROR) he[y].flags |= MFTL_FLAG_BOUNDARY;
  }
}

/**
 * @brief Flags triangle @p t and its three half-edges dead.
 *
 * @param obj The mesh.
 * @param t   Triangle index. Flagged #MFTL_FLAG_DEAD along with its three
 *            half-edges; the compaction removes it later.
 */
static void kill_triangle(MFTL_Object *obj, int t)
{
  obj->tris.rec[t].flags |= MFTL_FLAG_DEAD;
  for (int k = 0; k < 3; ++k) obj->tris.he[3 * t + k].flags |= MFTL_FLAG_DEAD;
}

int MFTLCollapseEdge(MFTL_Object *obj, int h)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  if (h < 0 || h >= 3 * obj->tris.n) return MFTL_ERROR;
  if (tri_is_dead(obj, MFTLHeTriIdx(h))) return MFTL_ERROR;

  /* A collapse allocates nothing, so unlike the split this pointer stays valid
   * for the whole function. */
  MFTL_HalfEdge *const he = obj->tris.he;

  const int t1 = MFTLHeTriIdx(h);
  const int hn = MFTLHeNextIdx(h);  // B->C
  const int hp = MFTLHePrevIdx(h);  // C->A

  const int a = he[hp].vertex_idx;  // retired
  const int b = he[h].vertex_idx;  // survives

  /* Both fans, captured while the topology still describes them. The union of
   * their triangles is exactly the patch this operation can disturb, which is
   * what makes the out_he repair at the end bounded and complete. */
  int fan_a[MFTL_REMESH_RING_MAX];
  int fan_b[MFTL_REMESH_RING_MAX];
  const int n_fan_a = MFTLGetVertexOneRingHalfedges(obj, a, fan_a, MFTL_REMESH_RING_MAX, MFTL_FALSE);
  const int n_fan_b = MFTLGetVertexOneRingHalfedges(obj, b, fan_b, MFTL_REMESH_RING_MAX, MFTL_FALSE);
  if (n_fan_a <= 0 || n_fan_a >= MFTL_REMESH_RING_MAX) return MFTL_ERROR;
  if (n_fan_b <= 0 || n_fan_b >= MFTL_REMESH_RING_MAX) return MFTL_ERROR;

  double pm[3];
  collapse_target_position(obj, a, b, pm);

  /* ── 1. Every half-edge that ENDS at A now ends at B ──────────────────────
   *
   * For an outgoing half-edge o of A, prev(o) is the incoming one, and the set
   * {prev(o)} over the fan is every half-edge targeting A -- including the
   * closing edge of an open fan, which exists only in the incoming direction and
   * which a walk over outgoing edges alone would miss. */
  for (int i = 0; i < n_fan_a; ++i) he[MFTLHePrevIdx(fan_a[i])].vertex_idx = b;

  /* ── 2. Remove the incident triangles ─────────────────────────────────────── */
  const int ht = he[h].twin_idx;

  glue_rim(obj, he[hn].twin_idx, he[hp].twin_idx);
  kill_triangle(obj, t1);

  if (ht != MFTL_ERROR)
  {
    const int hnt = MFTLHeNextIdx(ht);  // A->D
    const int hpt = MFTLHePrevIdx(ht);  // D->B

    glue_rim(obj, he[hnt].twin_idx, he[hpt].twin_idx);
    kill_triangle(obj, MFTLHeTriIdx(ht));
  }

  /* ── 3. Retire A, move B ──────────────────────────────────────────────────── */
  obj->verts.rec[a].flags |= MFTL_FLAG_DEAD;
  obj->verts.rec[b].flags |= (obj->verts.rec[a].flags & MFTL_FLAG_BOUNDARY);
  obj->verts.rec[b].flags |= MFTL_FLAG_VISITED;

  obj->verts.pos[3 * b + 0] = pm[0];
  obj->verts.pos[3 * b + 1] = pm[1];
  obj->verts.pos[3 * b + 2] = pm[2];

  /* ── 4. Repair out_he over the patch ──────────────────────────────────────
   *
   * Applying the global rule (`out_he[origin(x)] = x` for every live half-edge x)
   * to the patch alone. Deriving it rather than repairing each affected vertex
   * individually is one correctness argument instead of one per case -- and the
   * cases are exactly where a hand repair goes wrong: B's replacement must come
   * from a triangle that survived, C's old outgoing edge died with T1, and on an
   * open fan neither has an obvious substitute. */
  for (int pass = 0; pass < 2; ++pass)
  {
    const int *fan = pass ? fan_b : fan_a;
    const int n = pass ? n_fan_b : n_fan_a;

    for (int i = 0; i < n; ++i)
    {
      const int t = MFTLHeTriIdx(fan[i]);
      if (tri_is_dead(obj, t)) continue;

      for (int k = 0; k < 3; ++k) obj->verts.rec[MFTLHeOrigin(obj, 3 * t + k)].out_he_idx = 3 * t + k;
    }
  }

  MFTLObjectInvalidateDerived(obj);
  return MFTL_SUCCESS;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * FLIP
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The valence a vertex would ideally have: 6 inside, 4 on a mesh boundary.
 *
 * @param obj The mesh.
 * @param v   Vertex index.
 * @return 6 for an interior vertex, 4 on a mesh boundary -- the valence of a
 *         regular triangulation of the plane, and of its half.
 */
static inline int optimal_valence(const MFTL_Object *obj, int v) { return MFTLVertexIsMeshBoundary(obj, v) ? 4 : 6; }

/**
 * @brief Squared deviation of one vertex's valence from its optimum.
 * @param val Actual valence.
 * @param opt Optimal valence from optimal_valence().
 * @return \\f$(\\mathrm{val} - \\mathrm{opt})^2\\f$.
 */
static inline int valence_defect(int val, int opt) { return (val - opt) * (val - opt); }

bool MFTLFlipApproved(const MFTL_Object *obj, int h, const MFTL_RemeshParameters *params, int *reject_reason)
{
  MFTL_ASSERT(obj != MFTL_NULL && params != MFTL_NULL);

  if (reject_reason) *reject_reason = MFTL_REMESH_REJECT_NONE;

  if (!edge_is_operable(obj, h)) return MFTL_FALSE;
  if (!he_is_canonical(obj, h)) return MFTL_FALSE;

  const int ht = obj->tris.he[h].twin_idx;
  if (ht == MFTL_ERROR) return MFTL_FALSE;  // a boundary edge has no quad to re-cut
  if (tri_is_dead(obj, MFTLHeTriIdx(ht))) return MFTL_FALSE;

  /* A feature edge is a feature because of where it runs; flipping it moves it. */
  if (obj->tris.he[h].flags & (MFTL_FLAG_FEATURE | MFTL_FLAG_BOUNDARY)) return MFTL_FALSE;

  const int va = MFTLHeOrigin(obj, h);
  const int vb = MFTLHeTarget(obj, h);
  const int vc = MFTLHeTarget(obj, MFTLHeNextIdx(h));
  const int vd = MFTLHeTarget(obj, MFTLHeNextIdx(ht));

  int ring[MFTL_REMESH_RING_MAX];

  const int val_a = ring_vertices(obj, va, ring);
  if (val_a == MFTL_ERROR) return MFTL_FALSE;
  const int val_b = ring_vertices(obj, vb, ring);
  if (val_b == MFTL_ERROR) return MFTL_FALSE;

  /* A and B each lose a neighbour. Below 4 they would drop under the minimum a
   * manifold interior vertex can have. */
  if (val_a <= 3 || val_b <= 3) return MFTL_FALSE;

  const int val_c = ring_vertices(obj, vc, ring);
  if (val_c == MFTL_ERROR) return MFTL_FALSE;

  /* If C and D are already joined, the flip creates a SECOND edge between them:
   * two distinct undirected edges with the same endpoints, which no half-edge
   * traversal can tell apart. Cheap to test here, unrecoverable afterwards. */
  for (int i = 0; i < val_c; ++i)
  {
    if (ring[i] == vd) return MFTL_FALSE;
  }

  const int val_d = ring_vertices(obj, vd, ring);
  if (val_d == MFTL_ERROR) return MFTL_FALSE;

  if (val_c + 1 > MFTL_MAX_VALENCE || val_d + 1 > MFTL_MAX_VALENCE) return MFTL_FALSE;

  /* ── The valence criterion ────────────────────────────────────────────────
   *
   * Evaluated before the geometry, and not only because comparing four integers
   * is cheaper than two cross products: it is what makes "wanted but refused"
   * a distinguishable outcome. A flip the valence criterion declines is the
   * criterion working; a flip it wants and the geometry vetoes is a fold, and
   * only the second is worth a counter.
   *
   * STRICTLY smaller, never equal: two flips that leave the defect unchanged
   * would alternate for as many iterations as the caller allows, and each one
   * reports itself as work done, so the convergence test never fires either. */
  const int opt_a = optimal_valence(obj, va);
  const int opt_b = optimal_valence(obj, vb);
  const int opt_c = optimal_valence(obj, vc);
  const int opt_d = optimal_valence(obj, vd);

  const int before = valence_defect(val_a, opt_a) + valence_defect(val_b, opt_b) + valence_defect(val_c, opt_c) + valence_defect(val_d, opt_d);
  const int after = valence_defect(val_a - 1, opt_a) + valence_defect(val_b - 1, opt_b) + valence_defect(val_c + 1, opt_c) + valence_defect(val_d + 1, opt_d);

  if (after >= before) return MFTL_FALSE;

  /* ── Geometry: would the new diagonal fold the quad? ──────────────────────
   *
   * The valence criterion is purely combinatorial and will happily re-cut a
   * strongly non-convex quad, which produces a sliver overlapping its neighbour.
   * The test is that both new faces still face the way the pair used to. */
  const int t1 = MFTLHeTriIdx(h);
  const int t2 = MFTLHeTriIdx(ht);

  double n1[3];
  double n2[3];
  tri_area_vector_substituted(obj, t1, MFTL_ERROR, MFTL_ERROR, MFTL_NULL, n1);
  tri_area_vector_substituted(obj, t2, MFTL_ERROR, MFTL_ERROR, MFTL_NULL, n2);

  const double n_ref[3] = {n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]};

  const double *pa = MFTLVertexPos(obj, va);
  const double *pb = MFTLVertexPos(obj, vb);
  const double *pc = MFTLVertexPos(obj, vc);
  const double *pd = MFTLVertexPos(obj, vd);

  /* (A,D,C) and (D,B,C): the same quad cut along the other diagonal. */
  double ad[3];
  double ac[3];
  double db[3];
  double dc[3];
  MFTLGetMinimumImageVector(obj->domain, ad, pa, pd);
  MFTLGetMinimumImageVector(obj->domain, ac, pa, pc);
  MFTLGetMinimumImageVector(obj->domain, db, pd, pb);
  MFTLGetMinimumImageVector(obj->domain, dc, pd, pc);

  const double m1[3] = {ad[1] * ac[2] - ad[2] * ac[1], ad[2] * ac[0] - ad[0] * ac[2], ad[0] * ac[1] - ad[1] * ac[0]};
  const double m2[3] = {db[1] * dc[2] - db[2] * dc[1], db[2] * dc[0] - db[0] * dc[2], db[0] * dc[1] - db[1] * dc[0]};

  if (area_vector_cos(n_ref, m1) < params->max_normal_deviation_cos || area_vector_cos(n_ref, m2) < params->max_normal_deviation_cos)
  {
    if (reject_reason) *reject_reason = MFTL_REMESH_REJECT_FOLD;
    return MFTL_FALSE;
  }

  return MFTL_TRUE;
}

int MFTLFlipEdge(MFTL_Object *obj, int h)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  if (h < 0 || h >= 3 * obj->tris.n) return MFTL_ERROR;

  MFTL_HalfEdge *const he = obj->tris.he;

  const int ht = he[h].twin_idx;
  if (ht == MFTL_ERROR) return MFTL_ERROR;

  const int t1 = MFTLHeTriIdx(h);
  const int t2 = MFTLHeTriIdx(ht);

  const int hn = MFTLHeNextIdx(h);  // B->C
  const int hp = MFTLHePrevIdx(h);  // C->A
  const int hnt = MFTLHeNextIdx(ht);  // A->D
  const int hpt = MFTLHePrevIdx(ht);  // D->B

  const int va = he[hp].vertex_idx;
  const int vb = he[h].vertex_idx;
  const int vc = he[hn].vertex_idx;
  const int vd = he[hnt].vertex_idx;

  /* Rim twins and flags, captured before the rewrite moves every logical edge to
   * a different slot. */
  const int w_bc = he[hn].twin_idx;
  const int w_ca = he[hp].twin_idx;
  const int w_ad = he[hnt].twin_idx;
  const int w_db = he[hpt].twin_idx;

  const int f_bc = he[hn].flags;
  const int f_ca = he[hp].flags;
  const int f_ad = he[hnt].flags;
  const int f_db = he[hpt].flags;

  /* The quad is A -> D -> B -> C. Cutting it along the other diagonal gives
   * (A, D, C) and (D, B, C), both still counter-clockwise. */
  tri_write(obj, t1, va, vd, vc);
  he[3 * t1 + 0].flags = f_ad;  // A->D
  he[3 * t1 + 1].flags = 0;  // D->C, the new diagonal
  he[3 * t1 + 2].flags = f_ca;  // C->A

  tri_write(obj, t2, vd, vb, vc);
  he[3 * t2 + 0].flags = f_db;  // D->B
  he[3 * t2 + 1].flags = f_bc;  // B->C
  he[3 * t2 + 2].flags = 0;  // C->D

  he_pair(obj, 3 * t1 + 1, 3 * t2 + 2);  // D->C  <->  C->D
  he_pair_outer(obj, 3 * t1 + 0, w_ad);
  he_pair_outer(obj, 3 * t1 + 2, w_ca);
  he_pair_outer(obj, 3 * t2 + 0, w_db);
  he_pair_outer(obj, 3 * t2 + 1, w_bc);

  /* A and B lost the edge they were joined by, so whichever of their outgoing
   * half-edges pointed along it no longer exists. Setting all four is cheaper
   * than working out which two needed it. */
  obj->verts.rec[va].out_he_idx = 3 * t1 + 0;  // A->D
  obj->verts.rec[vd].out_he_idx = 3 * t1 + 1;  // D->C
  obj->verts.rec[vc].out_he_idx = 3 * t1 + 2;  // C->A
  obj->verts.rec[vb].out_he_idx = 3 * t2 + 1;  // B->C

  /* The new diagonal is off limits for the rest of the sweep. The valence
   * criterion's strict inequality already makes an immediate flip-back
   * impossible, but only while the four valences are unchanged -- a later flip in
   * the same quad's neighbourhood can restore the condition. */
  he[3 * t1 + 1].flags |= MFTL_FLAG_VISITED;
  he[3 * t2 + 2].flags |= MFTL_FLAG_VISITED;

  MFTLObjectInvalidateDerived(obj);
  return MFTL_SUCCESS;
}
