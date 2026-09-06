/*
 * @Author: Christian Gorges
 * @Date: 2026-08-16
 *
 * @brief Where the LBVH meets the spatial halo: rank boundaries, periodicity,
 *        rebuild timing, and the four ways a distributed contact query goes
 *        quietly wrong.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT THIS SUITE IS NOT TESTING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Not Guigue-Devillers. Not MFTL_Orient3D. Not the exact triangle-triangle
 * distance. Those are pure geometry, they take three doubles per corner and give
 * the same answer on every rank of every machine, and tests/test_bvh.c covers
 * them -- including the one check that makes THIS file's reference trustworthy,
 * `test_traversal_agrees_with_the_brute_force_baseline`.
 *
 * What is tested here is everything that only exists once a mesh is spread over
 * a process grid, where the four failure modes are:
 *
 *   MISSING    a contact whose two triangles are owned by different ranks and
 *              which no rank can see, because neither holds both sides. The
 *              spatial halo exists to prevent this, and every gap in it is
 *              silent -- the contact is simply never reported by anybody.
 *
 *   DUPLICATE  the same contact reported by two ranks, because both ended up
 *              holding both sides. Every contact force built from it is applied
 *              twice, and the simulation stays plausible while being wrong.
 *
 *   SPURIOUS   a contact reported between surfaces that are not in contact,
 *              because some envelope was derived from a partition-dependent or
 *              periodically-corrupted length.
 *
 *   DIVERGENT  a different answer at a different process count. Not wrong on any
 *              single run, and impossible to notice without running two.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE MASTER TEST, AND WHY IT IS TABLE-DRIVEN
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Every one of those four is a statement about a SET, so one assertion covers
 * all four at once:
 *
 *     the set of (global_tri_A, global_tri_B) pairs gathered from every rank
 *     of the DISTRIBUTED run must equal, as a multiset,
 *     the set produced by the same pipeline before the mesh was partitioned.
 *
 * Nothing weaker does. Counting hits misses duplicates that cancel a miss;
 * checking "every reference pair was found" misses duplicates entirely;
 * checking "no unexpected pair" misses everything absent.
 *
 * The reference is produced by the SAME pipeline running on the SAME ensemble
 * while rank 0 still holds the whole mesh (MFTLGenerateSphere's contract), so
 * the tolerance, the periodic image enumeration and the narrow phase are
 * identical on both sides and the only difference is the partition. That
 * reference is then itself checked, per case and on rank 0, against an
 * independent O(nA * nB) sweep that uses no BVH, no Morton order and no
 * traversal at all -- so a bug in the shared pipeline cannot hide by appearing
 * on both sides.
 *
 * The geometric CASES -- on the boundary, before it, behind it, on an edge
 * between four ranks, on a corner between eight, at the periodic seam, far from
 * the origin -- are rows of a table rather than functions, because they differ
 * only in where the scene is placed. One assertion, thirty placements.
 *
 * Every case additionally reports how many reference pairs are genuinely
 * CROSS-RANK under the resulting partition, and the rows that are supposed to
 * exercise the halo assert that this count is non-zero. Without it the suite
 * degrades silently the day a change to the decomposition puts every contact
 * inside one rank: the sets would still match and the halo -- the only thing
 * here not already covered serially -- would no longer be exercised at all.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_bvh.h"
#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;
static int g_rank = 0;
static int g_nproc = 1;

/** @brief Pair key packing. Asserted against the real triangle counts below. */
#define KEY_SHIFT 20
#define KEY_LIMIT (1 << KEY_SHIFT)

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The process grid
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A process grid for @p nproc, preferring cubic over thin.
 *
 * The listed counts are the ones this file is registered at, and each is a
 * different geometry rather than just more ranks:
 *
 *   1    no decomposition at all -- the reference path, and the guarantee that
 *        every collective is still entered
 *   2    2x1x1, one cut. Every rank boundary is a FACE and the neighbour list is
 *        shorter than the 26-offset stencil
 *   4    2x2x1, the first grid with a real EDGE: four ranks meet on a line
 *   8    2x2x2, the first grid with a CORNER: eight ranks meet at a point, and a
 *        triangle near it is wanted by seven neighbours at once
 *   27   3x3x3, the first grid with an INTERIOR rank -- one whose 26 neighbours
 *        are 26 distinct ranks, with no periodic aliasing collapsing them onto
 *        each other. That is the only configuration in which the neighbour
 *        de-duplication is not doing half the work by accident.
 */
static void choose_cuts(int nproc, int cuts[3])
{
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;

  for (int nx = (int) cbrt((double) nproc) + 2; nx >= 1; --nx)
  {
    if (nproc % nx != 0) continue;
    const int rest = nproc / nx;
    for (int ny = (int) sqrt((double) rest) + 2; ny >= 1; --ny)
    {
      if (rest % ny != 0) continue;
      const int nz = rest / ny;
      /* Prefer the most cubic factorisation: it maximises the number of
       * neighbours that are genuinely distinct ranks. */
      const int spread = MFTL_MAX(MFTL_MAX(nx, ny), nz) - MFTL_MIN(MFTL_MIN(nx, ny), nz);
      const int best = MFTL_MAX(MFTL_MAX(cuts[0], cuts[1]), cuts[2]) - MFTL_MIN(MFTL_MIN(cuts[0], cuts[1]), cuts[2]);
      if (spread < best)
      {
        cuts[0] = nx;
        cuts[1] = ny;
        cuts[2] = nz;
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Pair sets
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  long long *k;
  int n;
  int cap;
} keyset;

static void ks_push(keyset *s, long long v)
{
  if (s->n == s->cap)
  {
    s->cap = s->cap ? s->cap * 2 : 512;
    s->k = (long long *) realloc(s->k, (size_t) s->cap * sizeof(long long));
  }
  s->k[s->n++] = v;
}

static void ks_free(keyset *s)
{
  free(s->k);
  s->k = MFTL_NULL;
  s->n = 0;
  s->cap = 0;
}

static int cmp_ll(const void *a, const void *b)
{
  const long long x = *(const long long *) a;
  const long long y = *(const long long *) b;
  return (x > y) - (x < y);
}

static void ks_sort(keyset *s) { qsort(s->k, (size_t) s->n, sizeof(long long), cmp_ll); }

/**
 * @brief Multiset equality, with the first difference printed.
 *
 * MULTISET and not set: a pair legitimately appears more than once when two
 * periodic images of B both reach A, which is two distinct physical contacts
 * with two distinct normals. Collapsing to a set here would hide exactly the
 * duplication this file exists to detect, so the counts must match too.
 */
static int ks_equal(const keyset *ref, const keyset *got, const char *label)
{
  if (ref->n != got->n)
  {
    printf("\n  [%s] reference has %d pairs, distributed has %d\n  ", label, ref->n, got->n);
    return 0;
  }
  for (int i = 0; i < ref->n; ++i)
  {
    if (ref->k[i] != got->k[i])
    {
      printf("\n  [%s] first difference at %d: reference (%lld,%lld), distributed (%lld,%lld)\n  ", label, i, ref->k[i] >> KEY_SHIFT,
             ref->k[i] & (KEY_LIMIT - 1), got->k[i] >> KEY_SHIFT, got->k[i] & (KEY_LIMIT - 1));
      return 0;
    }
  }
  return 1;
}

/** @brief Gathers every rank's keys onto rank 0 and sorts them. */
static void ks_gather(const keyset *local, keyset *out)
{
  int *counts = MFTL_NULL;
  int *displs = MFTL_NULL;
  if (g_rank == 0)
  {
    counts = (int *) malloc((size_t) g_nproc * sizeof(int));
    displs = (int *) malloc((size_t) g_nproc * sizeof(int));
  }

  MPI_Gather(&local->n, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int total = 0;
  if (g_rank == 0)
  {
    for (int i = 0; i < g_nproc; ++i)
    {
      displs[i] = total;
      total += counts[i];
    }
    out->k = (long long *) malloc((size_t) (total > 0 ? total : 1) * sizeof(long long));
    out->cap = total > 0 ? total : 1;
  }
  else
  {
    out->k = MFTL_NULL;
    out->cap = 0;
  }
  out->n = total;

  MPI_Gatherv(local->k, local->n, MPI_LONG_LONG, out->k, counts, displs, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

  if (g_rank == 0)
  {
    ks_sort(out);
    free(counts);
    free(displs);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Collecting from the callback
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  keyset ks;
  const MFTL_BVH *ba;
  const MFTL_BVH *bb;
  int self_pairs;  ///< Events where objA == objB. Must always be zero.
} collector;

static void collect(const MFTL_Object *objA, int primA, const MFTL_Object *objB, int primB, void *user_data)
{
  collector *const c = (collector *) user_data;
  if (objA == objB) c->self_pairs++;

  const long long ga = MFTL_BVHPrimGlobalId(c->ba, primA);
  const long long gb = MFTL_BVHPrimGlobalId(c->bb, primB);
  ks_push(&c->ks, ga * KEY_LIMIT + gb);
}

/**
 * @brief Counts events without naming them.
 *
 * For the ensemble entry point, which owns its BVHs internally and hands the
 * callback only the objects -- so there is no hierarchy to resolve a primitive
 * index against, and the pair identity is not available. That is fine for the
 * statements that are about HOW MANY and about WHICH OBJECTS.
 */
typedef struct
{
  int hits;
  int self;
} tally;

static void count_pairs(const MFTL_Object *objA, int primA, const MFTL_Object *objB, int primB, void *user_data)
{
  (void) primA;
  (void) primB;
  tally *const t = (tally *) user_data;
  t->hits++;
  if (objA == objB) t->self++;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. Scene builders
 *
 * Both follow MFTLGenerateSphere's contract: rank 0 builds, the ID counters are
 * broadcast, the global stats are reduced. That contract is what makes an
 * unpartitioned reference possible at all -- one rank really does hold the whole
 * mesh before MFTLRedistribute runs.
 * ═══════════════════════════════════════════════════════════════════════════ */

static MFTL_Object *add_sphere(int gid, const double c[3], double r, int refine)
{
  MFTL_Object *const o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(g_ens, o);
  double centre[3] = {c[0], c[1], c[2]};
  MFTLGenerateSphere(o, centre, r, refine);
  return o;
}

/**
 * @brief An OPEN triangulated patch: a rectangle of @p nu x @p nv quads.
 *
 * There is no open-surface generator in the library yet (CLAUDE.md lists it as
 * not-yet-ported), and this suite needs one for two reasons that a closed sphere
 * cannot supply:
 *
 *   - a FREE RIM. A rim half-edge has no twin, so the only contact witness on it
 *     is an edge-edge distance; if the halo delivers a rim triangle without the
 *     vertices that close it, the contact is missed in a way no closed surface
 *     would reveal.
 *   - a FLAT face at an exactly representable coordinate, which is what makes
 *     the tolerance boundary test (gap == tol exactly) meaningful. Two spheres
 *     cannot be placed at an exact surface separation; two planes can.
 */
static MFTL_Object *add_patch(int gid, const double origin[3], const double u[3], const double v[3], int nu, int nv)
{
  MFTL_Object *const o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(g_ens, o);

  if (g_rank == 0)
  {
    const int nvu = nu + 1;
    const int nvv = nv + 1;
    const int nvert = nvu * nvv;
    const int ntri = 2 * nu * nv;

    double *const pos = (double *) malloc((size_t) nvert * 3 * sizeof(double));
    int *const tri = (int *) malloc((size_t) ntri * 3 * sizeof(int));

    for (int j = 0; j < nvv; ++j)
    {
      const double tv = (double) j / (double) nv;
      for (int i = 0; i < nvu; ++i)
      {
        const double tu = (double) i / (double) nu;
        const int w = j * nvu + i;
        double p[3];
        for (int k = 0; k < 3; ++k) p[k] = origin[k] + tu * u[k] + tv * v[k];
        MFTLApplyPeriodicity(&g_ens->domain, p);
        pos[3 * w + 0] = p[0];
        pos[3 * w + 1] = p[1];
        pos[3 * w + 2] = p[2];
      }
    }

    int t = 0;
    for (int j = 0; j < nv; ++j)
    {
      for (int i = 0; i < nu; ++i)
      {
        const int v00 = j * nvu + i;
        const int v10 = v00 + 1;
        const int v01 = v00 + nvu;
        const int v11 = v01 + 1;

        tri[3 * t + 0] = v00;
        tri[3 * t + 1] = v10;
        tri[3 * t + 2] = v11;
        t++;
        tri[3 * t + 0] = v00;
        tri[3 * t + 1] = v11;
        tri[3 * t + 2] = v01;
        t++;
      }
    }

    MFTLBuildIndexedMesh(o, pos, nvert, tri, ntri);
    free(pos);
    free(tri);
  }
  else
  {
    /* Same reason MFTLGenerateSphere does it: a hot loop over zero triangles
     * must not trip the derived-array staleness assertion. */
    MFTLObjectBuildDerived(o);
  }

  int gids[3] = {o->next_vertex_gid, o->next_triangle_gid, o->next_halfedge_gid};
  MPI_Bcast(gids, 3, MPI_INT, 0, MPI_COMM_WORLD);
  o->next_vertex_gid = gids[0];
  o->next_triangle_gid = gids[1];
  o->next_halfedge_gid = gids[2];

  MFTLObjectUpdateGlobalStats(o);
  return o;
}

/** @brief The per-timestep preamble every consumer of triangle geometry owes. */
static void refresh(void)
{
  for (int i = 0; i < g_ens->n_local_objects; ++i)
  {
    MFTLComputeTriangleGeometry(g_ens->local_objects[i], g_ens->local_objects[i]->tris.n);
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. The independent oracle
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Every contacting pair, by exhaustive search. No BVH, no Morton, no halo.
 *
 * Runs on ONE rank over a mesh that rank holds entirely, so it is only meaningful
 * before redistribution -- which is exactly where the reference is taken.
 *
 * The AABB pre-filter is not an optimisation that changes the answer: the boxes
 * are the true double bounds inflated by the same @p tol the narrow phase tests
 * against, so a rejected pair is provably further apart than tol. It exists so a
 * 320x320 sweep across 27 periodic images stays under a millisecond.
 *
 * @param tol Must be the tolerance the pipeline resolved, read back off the BVH.
 *            Re-deriving it here would make the oracle share the derivation it
 *            is supposed to be independent of.
 */
static void brute_contacts(const MFTL_Object *A, const MFTL_Object *B, double tol, keyset *out)
{
  const MFTL_Domain *const d = A->domain;
  const int nA = A->tris.n_local;
  const int nB = B->tris.n_local;

  double off[3][3];
  int n_off[3];
  for (int k = 0; k < 3; ++k)
  {
    off[k][0] = 0.0;
    n_off[k] = 1;
    if (d != MFTL_NULL && d->is_periodic[k] && d->domain_size[k] > 0.0)
    {
      off[k][1] = -d->domain_size[k];
      off[k][2] = +d->domain_size[k];
      n_off[k] = 3;
    }
  }

  double *const boxA = (double *) malloc(6 * (size_t) MFTL_MAX(nA, 1) * sizeof(double));
  for (int t = 0; t < nA; ++t) MFTLGetTriangleBounds(A, t, &boxA[6 * t], &boxA[6 * t + 3]);

  for (int ix = 0; ix < n_off[0]; ++ix)
  {
    for (int iy = 0; iy < n_off[1]; ++iy)
    {
      for (int iz = 0; iz < n_off[2]; ++iz)
      {
        const double sh[3] = {off[0][ix], off[1][iy], off[2][iz]};

        for (int tb = 0; tb < nB; ++tb)
        {
          double lo[3], hi[3], TB[3][3];
          MFTLGetTriangleBounds(B, tb, lo, hi);
          MFTLGetTriangleCorners(B, tb, TB);
          for (int k = 0; k < 3; ++k)
          {
            lo[k] += sh[k] - tol;
            hi[k] += sh[k] + tol;
          }
          for (int c = 0; c < 3; ++c)
          {
            TB[c][0] += sh[0];
            TB[c][1] += sh[1];
            TB[c][2] += sh[2];
          }

          for (int ta = 0; ta < nA; ++ta)
          {
            const double *const al = &boxA[6 * ta];
            const double *const ah = &boxA[6 * ta + 3];
            if (ah[0] < lo[0] || al[0] > hi[0]) continue;
            if (ah[1] < lo[1] || al[1] > hi[1]) continue;
            if (ah[2] < lo[2] || al[2] > hi[2]) continue;

            double TA[3][3];
            MFTLGetTriangleCorners(A, ta, TA);
            if (MFTL_TriTriProximity(TA, TB, tol))
              ks_push(out, (long long) A->tris.rec[ta].global_id * KEY_LIMIT + B->tris.rec[tb].global_id);
          }
        }
      }
    }
  }

  free(boxA);
  ks_sort(out);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. Running the pipeline
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief What one pipeline run produced, beyond the pairs themselves.
 *
 * `patch_defect` is a bitmask rather than a boolean so a failure names itself:
 * bit 0 a triangle arrived twice, bit 1 a triangle this rank already owns
 * arrived as a ghost, bit 2 the patch claims local ownership.
 */
typedef struct
{
  double tolerance;
  /**
   * @brief Could this rank have reported anything at all?
   *
   * True iff both trees have a node, i.e. this rank owns an A-side primitive AND
   * holds or was sent a B-side one. False makes MFTL_IntersectObjects return
   * before it looks at a box, so nothing about this rank's envelope is
   * observable -- which is why the envelope comparison below is conditioned on
   * it rather than taken over every rank.
   */
  int active;
  int halo_tris;  ///< Triangles this rank received for body B.
  int patch_defect;
  int self_pairs;
} run_info;

/**
 * @brief Body @p gid as an object, or a shared EMPTY one if this rank holds none.
 *
 * A rank holds an MFTL_Object only for the bodies it has entities of, so
 * `local_objects[b]` is body `b` only before the first redistribution. Contact
 * detection needs every body anyway -- its request list drives collectives and
 * must be identical on every rank -- so it reads them through the ensemble's
 * dense view, and so does this test.
 */
static MFTL_Object *bodyv(int gid) { return MFTLEnsembleAcquireDenseView(g_ens)[gid]; }

/**
 * @brief MFTL_FOREACH_OWNED_VERTEX over body @p gid, re-resolved at the point of use.
 *
 * A redistribution destroys the object of a body a rank keeps nothing of, so a
 * pointer cached before one is dangling after it. On a rank that holds none of
 * the body this iterates the shared empty placeholder, i.e. nothing.
 */
#define FOREACH_BODY_VERTEX(gid, ...)                   \
  do                                                    \
  {                                                     \
    MFTL_Object *const _b = bodyv(gid);                 \
    MFTL_FOREACH_OWNED_VERTEX(_b, __VA_ARGS__);         \
  } while (0)

/**
 * @brief The explicit two-body contact pipeline, in the order the header states.
 *
 * The radii come FIRST because the margin is a fraction of the body and the
 * body's size is only knowable collectively; the halo comes before the traversal
 * because the traversal is purely local and can only report what it holds.
 */
static void run_pipeline(int ia, int ib, const MFTL_BVHBuildParams *p, keyset *local_out, run_info *info)
{
  MFTL_Object *const a = bodyv(ia);
  MFTL_Object *const b = bodyv(ib);

  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(g_ens);
  MFTLSpatialHaloResolveRadii(g_ens, halo);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(a, p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(b, p);
  MFTL_BVHSetGlobalRadius(ba, MFTLSpatialHaloRadius(halo, ia));
  MFTL_BVHSetGlobalRadius(bb, MFTLSpatialHaloRadius(halo, ib));

  MFTL_SpatialHaloRequest req;
  req.body_a = ia;
  req.body_b = ib;
  req.margin = MFTL_MAX(ba->tolerance, bb->tolerance);

  MFTLExtendSpatialHalo(g_ens, &req, 1, halo);
  MFTL_BVHSetHalo(bb, MFTLSpatialHaloPatch(halo, ib));

  collector c;
  c.ks.k = MFTL_NULL;
  c.ks.n = 0;
  c.ks.cap = 0;
  c.ba = ba;
  c.bb = bb;
  c.self_pairs = 0;

  MFTL_IntersectObjects(a, ba, b, bb, collect, &c);

  info->tolerance = MFTL_MAX(ba->tolerance, bb->tolerance);
  info->active = (ba->nodes_used > 0 && bb->nodes_used > 0);
  info->self_pairs = c.self_pairs;
  info->halo_tris = MFTLSpatialHaloCount(halo, ib);
  info->patch_defect = 0;

  /* ── What arrived must be exactly what is missing, once ─────────────────
   *
   * A triangle delivered twice would be tested twice against every local A
   * triangle near it, so the pair would be reported twice by ONE rank -- a
   * duplicate the reporting rule cannot catch, because both copies are on the
   * correct side of it. A triangle this rank already owns would be worse: it is
   * already an A-side-eligible primitive of B's own tree, so the same contact
   * would be found through two different routes. */
  {
    const MFTL_Object *const patch = MFTLSpatialHaloPatch(halo, ib);
    if (patch != MFTL_NULL)
    {
      if (patch->tris.n_local != 0 || patch->verts.n_local != 0) info->patch_defect |= 4;

      for (int t = 0; t < patch->tris.n; ++t)
      {
        const int gid = patch->tris.rec[t].global_id;
        for (int u = t + 1; u < patch->tris.n; ++u)
        {
          if (patch->tris.rec[u].global_id == gid) info->patch_defect |= 1;
        }
        for (int u = 0; u < b->tris.n_local; ++u)
        {
          if (b->tris.rec[u].global_id == gid) info->patch_defect |= 2;
        }
      }
    }
  }

  *local_out = c.ks;

  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. Was the halo actually needed?
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief owner[gid] for every triangle of @p obj, agreed by every rank.
 *
 * Global IDs are dense per entity class (inv_global_ids_are_dense), which is
 * what lets this be a plain array indexed by ID and one MPI_MAX rather than a
 * gather of pairs. -1 means nobody claimed it, which cannot happen for a body
 * that has been redistributed.
 */
static int *owner_table(const MFTL_Object *obj)
{
  /* The LENGTH has to be agreed before the reduction is posted. `obj` may be the
   * dense view's shared empty object on a rank that holds none of this body, and
   * that one reports a global triangle count of zero -- so sizing the buffer from
   * it locally gives two ranks two different message lengths, which MPI reports
   * as a truncation rather than as the disagreement it is. */
  int n = obj->global_n_triangles;
  MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  int *const mine = (int *) malloc((size_t) MFTL_MAX(n, 1) * sizeof(int));
  int *const all = (int *) malloc((size_t) MFTL_MAX(n, 1) * sizeof(int));

  for (int i = 0; i < n; ++i) mine[i] = -1;
  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g >= 0 && g < n) mine[g] = g_rank;
  }

  MPI_Allreduce(mine, all, MFTL_MAX(n, 1), MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  free(mine);
  return all;
}

/**
 * @brief How many reference pairs have their two triangles on different ranks.
 *
 * THE NON-VACUITY MEASURE. A case whose contacts all land inside one rank tests
 * the traversal and nothing else -- the halo could be deleted entirely and the
 * sets would still match. Rows of the table that exist to exercise the halo
 * assert this is non-zero, so the suite cannot quietly stop covering it.
 */
static int count_cross_rank(const keyset *ref, const int *ownA, const int *ownB)
{
  int n = 0;
  for (int i = 0; i < ref->n; ++i)
  {
    const int ga = (int) (ref->k[i] >> KEY_SHIFT);
    const int gb = (int) (ref->k[i] & (KEY_LIMIT - 1));
    if (ownA[ga] != ownB[gb]) n++;
  }
  return n;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 8. The case table
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef enum
{
  SCENE_SPHERES,  ///< two overlapping spheres, contact lens on the anchor
  SCENE_SPHERE_PATCH,  ///< sphere pressed through an open rectangular patch
  SCENE_RIM_ON_CUT,  ///< sphere against a patch whose FREE RIM sits on the anchor
  SCENE_GRAZE  ///< sphere tangent to the patch silhouette: contact area -> a point
} scene_kind;

typedef enum
{
  ANCHOR_INTERIOR,  ///< centre of rank 0's cell: no cut anywhere near the contact
  ANCHOR_FACE,  ///< a cut plane: two ranks meet
  ANCHOR_EDGE,  ///< two cut planes: four ranks meet on a line
  ANCHOR_CORNER,  ///< three cut planes: eight ranks meet at a point
  ANCHOR_SEAM  ///< the periodic seam, which is also the outer process boundary
} anchor_kind;

typedef struct
{
  const char *name;
  scene_kind scene;
  anchor_kind anchor;
  double offset;  ///< Along x, in units of the body radius. <0 before the cut, >0 behind it.
  bool periodic;
  bool anisotropic;  ///< Subdomains with very different edge lengths.
  bool far_origin;  ///< Domain centred at 1e6: float AABB against double narrow phase.
  bool expect_cross;  ///< Assert the reference really contains cross-rank pairs.
} ghost_case;

/* The offsets are in units of the body radius R, and the three values are the
 * user-visible cases "contact before / on / behind the boundary":
 *
 *   -1.0   the contact lens is a full radius inside the local rank. One body
 *          still crosses the cut, so the neighbour holds triangles and ghosts of
 *          this body -- which is precisely the configuration in which a BVH built
 *          over the redistribution halo would report every contact twice.
 *    0.0   the cut passes through the contact lens itself. Every contact near
 *          the plane is cross-rank and only the spatial halo can see it.
 *   +1.0   the lens is a full radius into the NEIGHBOUR, with this rank still
 *          holding part of one body. The mirror of -1.0, and it fails
 *          differently: here the local rank must NOT report what the neighbour
 *          owns both sides of.
 */
static const ghost_case g_cases[] = {
    /* name                       scene               anchor           off   per    aniso  far    cross */
    {"spheres/interior", SCENE_SPHERES, ANCHOR_INTERIOR, 0.0, false, false, false, false},
    {"spheres/face-before", SCENE_SPHERES, ANCHOR_FACE, -1.0, false, false, false, false},
    {"spheres/face-on", SCENE_SPHERES, ANCHOR_FACE, 0.0, false, false, false, true},
    {"spheres/face-behind", SCENE_SPHERES, ANCHOR_FACE, +1.0, false, false, false, false},
    {"spheres/edge-before", SCENE_SPHERES, ANCHOR_EDGE, -1.0, false, false, false, false},
    {"spheres/edge-on", SCENE_SPHERES, ANCHOR_EDGE, 0.0, false, false, false, true},
    {"spheres/edge-behind", SCENE_SPHERES, ANCHOR_EDGE, +1.0, false, false, false, false},
    {"spheres/corner-before", SCENE_SPHERES, ANCHOR_CORNER, -1.0, false, false, false, false},
    {"spheres/corner-on", SCENE_SPHERES, ANCHOR_CORNER, 0.0, false, false, false, true},
    {"spheres/corner-behind", SCENE_SPHERES, ANCHOR_CORNER, +1.0, false, false, false, false},

    {"patch/face-before", SCENE_SPHERE_PATCH, ANCHOR_FACE, -1.0, false, false, false, false},
    {"patch/face-on", SCENE_SPHERE_PATCH, ANCHOR_FACE, 0.0, false, false, false, true},
    {"patch/face-behind", SCENE_SPHERE_PATCH, ANCHOR_FACE, +1.0, false, false, false, false},
    {"patch/edge-on", SCENE_SPHERE_PATCH, ANCHOR_EDGE, 0.0, false, false, false, true},
    {"patch/corner-on", SCENE_SPHERE_PATCH, ANCHOR_CORNER, 0.0, false, false, false, true},

    {"rim-on-cut/face", SCENE_RIM_ON_CUT, ANCHOR_FACE, 0.0, false, false, false, true},
    {"rim-on-cut/corner", SCENE_RIM_ON_CUT, ANCHOR_CORNER, 0.0, false, false, false, true},

    {"graze/face", SCENE_GRAZE, ANCHOR_FACE, 0.0, false, false, false, true},
    {"graze/corner", SCENE_GRAZE, ANCHOR_CORNER, 0.0, false, false, false, true},

    {"periodic/seam", SCENE_SPHERES, ANCHOR_SEAM, 0.0, true, false, false, true},
    {"periodic/seam-before", SCENE_SPHERES, ANCHOR_SEAM, -1.0, true, false, false, false},
    {"periodic/seam-behind", SCENE_SPHERES, ANCHOR_SEAM, +1.0, true, false, false, false},
    {"periodic/seam-patch", SCENE_SPHERE_PATCH, ANCHOR_SEAM, 0.0, true, false, false, true},
    {"periodic/face", SCENE_SPHERES, ANCHOR_FACE, 0.0, true, false, false, true},
    {"periodic/corner", SCENE_SPHERES, ANCHOR_CORNER, 0.0, true, false, false, true},

    {"aniso/face", SCENE_SPHERES, ANCHOR_FACE, 0.0, false, true, false, true},
    {"aniso/corner", SCENE_SPHERES, ANCHOR_CORNER, 0.0, false, true, false, true},
    {"aniso/periodic-seam", SCENE_SPHERES, ANCHOR_SEAM, 0.0, true, true, false, true},

    {"far-origin/face", SCENE_SPHERES, ANCHOR_FACE, 0.0, false, false, true, true},
    {"far-origin/corner", SCENE_SPHERES, ANCHOR_CORNER, 0.0, false, false, true, true},
};

#define N_CASES ((int) (sizeof(g_cases) / sizeof(g_cases[0])))

/** @brief Body radius, in domain units. The domain is 10 wide on its short axis. */
#define CASE_R 1.0
/** @brief Sphere refinement: 320 triangles, enough for several per rank at np=27. */
#define CASE_REFINE 2

/**
 * @brief Builds the ensemble and the two bodies for one case.
 *
 * The anchor is derived from the ACTUAL process grid at run time, so the same
 * table means "on a face" at every process count without the rows knowing what
 * the count is. Where an axis has no interior cut -- a thin grid, or np == 1 --
 * the anchor falls back to the centre of rank 0's cell, and the row degenerates
 * into a weaker but still valid placement rather than becoming meaningless.
 */
static void build_case(const ghost_case *c)
{
  int cuts[3];
  choose_cuts(g_nproc, cuts);

  double lo[3], hi[3];
  const double base = c->far_origin ? 1.0e6 : 0.0;
  const double len[3] = {10.0, c->anisotropic ? 40.0 : 10.0, c->anisotropic ? 2.5 : 10.0};
  for (int k = 0; k < 3; ++k)
  {
    lo[k] = base;
    hi[k] = base + len[k];
  }

  bool per[3] = {c->periodic, c->periodic, c->periodic};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

  /* ── Where the contact goes ───────────────────────────────────────────── */
  double anchor[3];
  for (int k = 0; k < 3; ++k) anchor[k] = lo[k] + 0.5 * len[k] / (double) cuts[k];  // centre of rank 0's cell

  const bool on_cut[3] = {c->anchor == ANCHOR_FACE || c->anchor == ANCHOR_EDGE || c->anchor == ANCHOR_CORNER,
                          c->anchor == ANCHOR_EDGE || c->anchor == ANCHOR_CORNER, c->anchor == ANCHOR_CORNER};

  for (int k = 0; k < 3; ++k)
  {
    if (on_cut[k] && cuts[k] > 1) anchor[k] = lo[k] + len[k] / (double) cuts[k];  // the first interior cut plane
  }

  if (c->anchor == ANCHOR_SEAM) anchor[0] = lo[0];  // the periodic seam: rank 0's low face

  anchor[0] += c->offset * CASE_R;

  /* ── The bodies ───────────────────────────────────────────────────────── */
  const double d = 1.6 * CASE_R;  // centre separation: a substantial overlap lens

  switch (c->scene)
  {
    case SCENE_SPHERES:
    {
      const double ca[3] = {anchor[0] - 0.5 * d, anchor[1], anchor[2]};
      const double cb[3] = {anchor[0] + 0.5 * d, anchor[1], anchor[2]};
      add_sphere(0, ca, CASE_R, CASE_REFINE);
      add_sphere(1, cb, CASE_R, CASE_REFINE);
      break;
    }

    case SCENE_SPHERE_PATCH:
    {
      /* Sphere centre 0.8R before the anchor plane, so its +x cap crosses it. */
      const double ca[3] = {anchor[0] - 0.8 * CASE_R, anchor[1], anchor[2]};
      add_sphere(0, ca, CASE_R, CASE_REFINE);

      const double org[3] = {anchor[0], anchor[1] - 1.5 * CASE_R, anchor[2] - 1.5 * CASE_R};
      const double u[3] = {0.0, 3.0 * CASE_R, 0.0};
      const double v[3] = {0.0, 0.0, 3.0 * CASE_R};
      add_patch(1, org, u, v, 6, 6);
      break;
    }

    case SCENE_RIM_ON_CUT:
    {
      /* The patch's +y FREE RIM lies exactly on the anchor's y coordinate, so on
       * a grid with a y cut the rim half-edges -- the ones with no twin, whose
       * only contact witness is an edge-edge distance -- are the triangles that
       * have to travel. */
      const double ca[3] = {anchor[0] - 0.8 * CASE_R, anchor[1] - 0.4 * CASE_R, anchor[2]};
      add_sphere(0, ca, CASE_R, CASE_REFINE);

      const double org[3] = {anchor[0], anchor[1] - 3.0 * CASE_R, anchor[2] - 1.5 * CASE_R};
      const double u[3] = {0.0, 3.0 * CASE_R, 0.0};
      const double v[3] = {0.0, 0.0, 3.0 * CASE_R};
      add_patch(1, org, u, v, 6, 6);
      break;
    }

    case SCENE_GRAZE:
    {
      /* GRAZING / SILHOUETTE. The patch lies in the plane y = anchor.y and its
       * +x free rim is exactly on the anchor; the sphere sits directly above
       * that rim and dips 0.03 R through the plane. So the contact region is not
       * a lens but a thin crescent cut off by the rim, every witness is an
       * edge-edge pair between two near-parallel edges, and the contact area
       * tends to a point as the dip goes to zero -- the worst-conditioned input
       * the narrow phase sees.
       *
       * The dip is 0.03 R and not zero on purpose. An icosphere's facets are
       * INSIDE the sphere they approximate, so a plane tangent to the ideal
       * sphere misses the mesh by about 0.02 R at this refinement; a test that
       * placed the plane at exactly R would be asserting a property of the
       * discretisation error rather than of the traversal. */
      const double ca[3] = {anchor[0], anchor[1] + 0.97 * CASE_R, anchor[2]};
      add_sphere(0, ca, CASE_R, CASE_REFINE);

      const double org[3] = {anchor[0] - 3.0 * CASE_R, anchor[1], anchor[2] - 1.5 * CASE_R};
      const double u[3] = {3.0 * CASE_R, 0.0, 0.0};
      const double v[3] = {0.0, 0.0, 3.0 * CASE_R};
      add_patch(1, org, u, v, 6, 6);
      break;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 9. THE MASTER TEST
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Every placement in the table: distributed answer == unpartitioned answer.
 *
 * Runs to the end rather than stopping at the first failure, because when a
 * geometric family breaks it is far more informative to see WHICH rows go with
 * it -- all the corner rows, or every periodic row, or only the far-origin ones.
 * Every verdict is reduced before it is used (see `agreed`), so a rank-local
 * failure cannot leave the others blocked in the next collective.
 */
static int test_the_distributed_answer_matches_the_unpartitioned_one(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  int failures = 0;
  int total_cross = 0;
  int total_halo = 0;

  for (int ci = 0; ci < N_CASES; ++ci)
  {
    const ghost_case *const c = &g_cases[ci];
    int ok = 1;

    if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
    build_case(c);
    refresh();

    int gt[2] = {bodyv(0)->global_n_triangles, bodyv(1)->global_n_triangles};
    MPI_Allreduce(MPI_IN_PLACE, gt, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (gt[0] >= KEY_LIMIT || gt[1] >= KEY_LIMIT) ok = 0;

    /* ── Phase 1: the reference, while rank 0 holds everything ───────────── */
    keyset local_ref = {MFTL_NULL, 0, 0};
    keyset ref = {MFTL_NULL, 0, 0};
    run_info ref_info;
    run_pipeline(0, 1, &p, &local_ref, &ref_info);
    ks_gather(&local_ref, &ref);
    ks_free(&local_ref);

    /* ...checked against a sweep that shares none of the pipeline's machinery. */
    if (g_rank == 0)
    {
      const MFTL_Object *const a = bodyv(0);
      const MFTL_Object *const b = bodyv(1);
      if ((double) a->tris.n_local * (double) b->tris.n_local <= 4.0e5)
      {
        keyset brute = {MFTL_NULL, 0, 0};
        brute_contacts(a, b, ref_info.tolerance, &brute);
        if (!ks_equal(&brute, &ref, c->name))
        {
          printf("  ^ the REFERENCE disagrees with the exhaustive sweep, so the case below is meaningless\n  ");
          ok = 0;
        }
        ks_free(&brute);
      }
    }

    /* The reference must be non-empty, or every comparison below is vacuous. */
    int n_ref = ref.n;
    MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n_ref <= 0)
    {
      if (g_rank == 0) printf("\n  [%s] the reference found no contacts at all\n  ", c->name);
      ok = 0;
    }

    /* ── Phase 2: spread it out and ask again ────────────────────────────── */
    if (MFTLRedistribute(g_ens) != MFTL_SUCCESS) ok = 0;
    refresh();

    keyset local_got = {MFTL_NULL, 0, 0};
    keyset got = {MFTL_NULL, 0, 0};
    run_info got_info;
    run_pipeline(0, 1, &p, &local_got, &got_info);
    ks_gather(&local_got, &got);
    ks_free(&local_got);

    /* ── The verdicts ────────────────────────────────────────────────────── */
    if (got_info.patch_defect != 0)
    {
      printf("\n  [%s] rank %d: halo patch defect mask %d\n  ", c->name, g_rank, got_info.patch_defect);
      ok = 0;
    }
    if (got_info.self_pairs != 0) ok = 0;

    /* ── The envelope must not have moved when the mesh was partitioned ─────
     *
     * That is the DIVERGENT failure mode, and it would change the set for a
     * reason that has nothing to do with ghosting.
     *
     * ON THE RANKS THAT CAN REPORT, which is the whole of the property and not a
     * weakening of it. A body's global radius reaches the ranks that hold it
     * (the per-body reduction) and their graph neighbours (the radius publish),
     * because those are exactly the ranks the halo can ever deliver to -- a rank
     * two hops from body B receives no patch of it, holds none of it, and so
     * returns from the traversal before it looks at a single box. Its envelope
     * is unobservable. Asserting it everywhere is asserting that every rank in
     * the job knows the size of every body in the ensemble, which is the
     * `O(bodies)` property this pipeline deliberately no longer has. */
    if (got_info.active && fabs(got_info.tolerance - ref_info.tolerance) > 1e-12 * MFTL_MAX(ref_info.tolerance, 1.0))
    {
      printf("\n  [%s] rank %d: tolerance moved: %.17g -> %.17g\n  ", c->name, g_rank, ref_info.tolerance, got_info.tolerance);
      ok = 0;
    }

    int cross = 0;
    {
      int *const ownA = owner_table(bodyv(0));
      int *const ownB = owner_table(bodyv(1));
      if (g_rank == 0) cross = count_cross_rank(&ref, ownA, ownB);
      MPI_Bcast(&cross, 1, MPI_INT, 0, MPI_COMM_WORLD);
      free(ownA);
      free(ownB);
    }

    if (c->expect_cross && g_nproc > 1 && cross == 0)
    {
      if (g_rank == 0) printf("\n  [%s] expected cross-rank contacts, found none -- the halo is not being exercised\n  ", c->name);
      ok = 0;
    }

    if (g_rank == 0 && !ks_equal(&ref, &got, c->name)) ok = 0;

    ks_free(&ref);
    ks_free(&got);

    int halo_sum = 0;
    MPI_Allreduce(&got_info.halo_tris, &halo_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    total_cross += cross;
    total_halo += halo_sum;

    if (g_rank == 0)
      printf("\n    %-24s %5d pairs %5d cross-rank %5d ghosted%s", c->name, n_ref, cross, halo_sum, (ci == N_CASES - 1) ? "\n  " : "");

    ok = agreed(ok, MPI_COMM_WORLD);
    if (!ok)
    {
      failures++;
      if (g_rank == 0) printf("   <-- FAILED");
    }
  }

  /* ── The suite must not be able to go quiet ──────────────────────────────
   *
   * Every assertion above is satisfied by a run in which no contact was ever
   * cross-rank and nothing was ever ghosted: the sets would match because the
   * halo was never needed. That is not a hypothetical -- it is what happens the
   * day the bodies shrink, the domain grows, or the grid changes shape. So the
   * TOTALS are asserted too, and they are the reason this test can be trusted
   * without reading the table above. */
  if (g_nproc > 1)
  {
    TEST_ASSERT(agreed(total_cross > 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(total_halo > 0, MPI_COMM_WORLD));
  }

  TEST_ASSERT(agreed(failures == 0, MPI_COMM_WORLD));
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 10. The cases that are not placements
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief A cubic domain of side @p side at the origin, with the standard grid. */
static void plain_ensemble(double side, bool periodic)
{
  int cuts[3];
  choose_cuts(g_nproc, cuts);

  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {side, side, side};
  bool per[3] = {periodic, periodic, periodic};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);
}

/**
 * @brief A body straddling the periodic seam must have a BODY-sized radius.
 *
 * The contact envelope is `factor * bounding_radius`, and the bounding radius
 * comes from a min/max over stored coordinates. A 1-unit sphere sitting on the
 * seam has half its triangles at x ~ 0 and half at x ~ L, so the raw extent says
 * "this body is L wide" and every envelope scaled by it grows by that factor.
 *
 * Unlike the Morton bounding box -- where a domain-wide root for a straddling
 * object is accepted locality loss (CLAUDE.md) -- this one changes the ANSWER:
 * contact would be declared between surfaces nowhere near each other, and the
 * error grows with the domain rather than staying bounded.
 *
 * MFTLSpatialHaloResolveRadii measures each body in two frames, raw and shifted
 * by half a domain, and keeps the smaller extent. This asserts the straddling
 * body comes out the same size as the identical body in the interior.
 */
static int test_a_seam_straddling_body_is_not_measured_as_domain_sized(void)
{
  plain_ensemble(10.0, true);

  const double at_seam[3] = {0.0, 5.0, 5.0};  // centred ON the seam: half on each side
  const double inside[3] = {5.0, 5.0, 5.0};
  add_sphere(0, at_seam, 1.0, 2);
  add_sphere(1, inside, 1.0, 2);
  refresh();

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(g_ens);
  MFTLSpatialHaloResolveRadii(g_ens, halo);

  const double r_seam = MFTLSpatialHaloRadius(halo, 0);
  const double r_inside = MFTLSpatialHaloRadius(halo, 1);

  /* The interior sphere is the control: whatever an icosphere's half-extent is,
   * the straddling one must agree with it. Comparing against 1.0 instead would
   * bake in the discretisation error of the refinement level. */
  TEST_ASSERT(agreed(r_inside > 0.9 && r_inside < 1.01, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(fabs(r_seam - r_inside) < 1.0e-9, MPI_COMM_WORLD));

  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief Two bodies at opposite ends of a periodic box are neighbours.
 *
 * And on a decomposed domain they are neighbours whose two halves live on ranks
 * that are adjacent only THROUGH the seam. The extents those ranks publish are a
 * domain length apart in stored coordinates, so the classification that decides
 * what to send has to be periodic-aware in its own right -- the traversal's
 * image loop cannot rescue a triangle that was never transmitted.
 */
static int test_contact_through_the_seam_survives_the_decomposition(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  plain_ensemble(10.0, true);

  const double ca[3] = {9.6, 5.0, 5.0};
  const double cb[3] = {0.6, 5.0, 5.0};  // periodic gap 1.0 between centres of radius-1 spheres
  add_sphere(0, ca, 1.0, 2);
  add_sphere(1, cb, 1.0, 2);
  refresh();

  keyset local_ref = {MFTL_NULL, 0, 0};
  keyset ref = {MFTL_NULL, 0, 0};
  run_info ri;
  run_pipeline(0, 1, &p, &local_ref, &ri);
  ks_gather(&local_ref, &ref);
  ks_free(&local_ref);

  int n_ref = ref.n;
  MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(n_ref > 0, MPI_COMM_WORLD));

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  keyset local_got = {MFTL_NULL, 0, 0};
  keyset got = {MFTL_NULL, 0, 0};
  run_info gi;
  run_pipeline(0, 1, &p, &local_got, &gi);
  ks_gather(&local_got, &got);
  ks_free(&local_got);

  int same = 1;
  if (g_rank == 0) same = ks_equal(&ref, &got, "through-the-seam");
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));

  ks_free(&ref);
  ks_free(&got);
  return 0;
}

/**
 * @brief A body that wraps onto itself must not collide with itself.
 *
 * A sphere of radius 3 in a box of side 5 overlaps its own periodic images: the
 * traversal's image loop offers B at -L and +L, and if the pair (X, X) were ever
 * enumerated it would report the whole surface as being in contact. Two separate
 * statements are checked, because they fail independently:
 *
 *   - a single body produces NOTHING. MFTLEnsembleIntersectObjects iterates
 *     j > i, so self-intersection is out of scope by construction rather than by
 *     a filter that could be removed -- and self-intersection of one surface is
 *     a different algorithm anyway, since adjacent triangles share vertices and
 *     always "touch".
 *   - with a second body present, no reported event has objA == objB. That rules
 *     out the image loop leaking a body into its own query.
 */
static int test_a_self_overlapping_body_reports_no_self_contact(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-3;
  p.proximity_floor = 0.0;

  plain_ensemble(5.0, true);

  const double c0[3] = {2.5, 2.5, 2.5};
  add_sphere(0, c0, 3.0, 2);  // diameter 6 in a box of side 5: it wraps onto itself
  refresh();
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  /* One body. The pipeline must run every collective and find nothing. */
  {
    tally t = {0, 0};
    TEST_ASSERT(agreed(MFTLEnsembleIntersectObjects(g_ens, &p, count_pairs, &t) == MFTL_SUCCESS, MPI_COMM_WORLD));

    int total = 0;
    MPI_Allreduce(&t.hits, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(total == 0, MPI_COMM_WORLD));
  }

  /* Two bodies, the big one still wrapping onto itself: every reported event
   * must name two DIFFERENT objects, whichever periodic image produced it. */
  {
    /* ON the big sphere's surface, and reached through the seam: 3 away in x
     * from a body of radius 3, wrapped back into a box of side 5. */
    const double c1[3] = {0.5, 2.5, 2.5};
    add_sphere(1, c1, 0.4, 1);
    refresh();
    TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
    refresh();

    tally t = {0, 0};
    TEST_ASSERT(agreed(MFTLEnsembleIntersectObjects(g_ens, &p, count_pairs, &t) == MFTL_SUCCESS, MPI_COMM_WORLD));

    int self_total = 0;
    int hit_total = 0;
    MPI_Allreduce(&t.self, &self_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&t.hits, &hit_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    TEST_ASSERT(agreed(self_total == 0, MPI_COMM_WORLD));
    /* And the small body really is inside the big one, so the query is not
     * vacuously silent -- it found genuine A-B contacts and no A-A ones. */
    TEST_ASSERT(agreed(hit_total > 0, MPI_COMM_WORLD));
  }

  return 0;
}

/**
 * @brief Two bodies passing through each other between rebuilds: EXPECTED MISS.
 *
 * This is a documented limit, not a bug. The library is a DISCRETE-time contact
 * detector: MFTL_GenerateLBVH and MFTL_RefitLBVH see the configuration at one
 * instant, and nothing sweeps the volume between two instants. A pair whose
 * relative displacement in one step exceeds the sum of their sizes therefore
 * passes through without ever being coincident at a sampled instant, and both
 * samples correctly report nothing.
 *
 * Continuous collision detection would fix it and is deliberately not
 * implemented: it needs the swept volume of every triangle, which is a different
 * primitive, a different BVH and a per-step velocity field the module does not
 * receive. The test pins the behaviour so that "we found no contact" is a
 * recorded property rather than an assumption -- a caller whose timestep is set
 * by a CFL condition on the flow, not by the approach velocity of two bodies,
 * has to know this.
 */
static int test_tunnelling_between_two_timesteps_is_a_known_miss(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-3;
  p.proximity_floor = 0.0;

  plain_ensemble(10.0, false);

  /* Step n: A approaching from the left, B from the right, 4 radii apart. */
  const double ca[3] = {3.0, 5.0, 5.0};
  const double cb[3] = {7.0, 5.0, 5.0};
  add_sphere(0, ca, 1.0, 2);
  add_sphere(1, cb, 1.0, 2);
  refresh();
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  int hits_before = 0;
  {
    keyset ks = {MFTL_NULL, 0, 0};
    run_info info;
    run_pipeline(0, 1, &p, &ks, &info);
    const int n = ks.n;
    MPI_Allreduce(&n, &hits_before, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ks_free(&ks);
  }

  /* Step n+1: they have swapped sides. Nothing sampled the crossing. */
  FOREACH_BODY_VERTEX(0, { pos[0] += 4.0; });
  FOREACH_BODY_VERTEX(1, { pos[0] -= 4.0; });

  MFTLUpdateGhostPositions(g_ens);
  for (int i = 0; i < g_ens->n_local_objects; ++i) MFTLComputeTriangleGeometry(g_ens->local_objects[i], g_ens->local_objects[i]->tris.n);
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  int hits_after = 0;
  {
    keyset ks = {MFTL_NULL, 0, 0};
    run_info info;
    run_pipeline(0, 1, &p, &ks, &info);
    const int n = ks.n;
    MPI_Allreduce(&n, &hits_after, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ks_free(&ks);
  }

  /* EXPECTED: nothing, at both sampled instants. If this ever starts failing,
   * something has begun reporting contacts that are not present AT the sampled
   * configuration -- which is a bug, not the arrival of CCD. */
  TEST_ASSERT(agreed(hits_before == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(hits_after == 0, MPI_COMM_WORLD));
  return 0;
}

/**
 * @brief A body that changes rank ownership wholesale in one step.
 *
 * Migration and ghosting interact through a single ordering rule: the halo is
 * derived from what each rank OWNS, so it is valid only for the partition that
 * produced it. Growing it before the redistribution -- or reusing last step's --
 * describes an ownership map that no longer exists, and the failure is silent
 * because the stale patch still contains perfectly valid triangles.
 *
 * The scene translates both bodies far enough that every triangle changes owner,
 * then compares the answer for the NEW configuration against the unpartitioned
 * pipeline for that same configuration.
 */
static int test_a_body_that_migrates_wholesale_keeps_the_answer(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  const double start_a[3] = {1.6, 1.6, 1.6};
  const double start_b[3] = {3.2, 1.6, 1.6};
  const double step = 5.6;  // to the opposite corner: every triangle changes owner

  /* ── The reference: the DESTINATION configuration, never migrated ───────── */
  keyset ref = {MFTL_NULL, 0, 0};
  {
    plain_ensemble(10.0, false);
    const double ca[3] = {start_a[0] + step, start_a[1] + step, start_a[2] + step};
    const double cb[3] = {start_b[0] + step, start_b[1] + step, start_b[2] + step};
    add_sphere(0, ca, 1.0, 2);
    add_sphere(1, cb, 1.0, 2);
    refresh();

    keyset local = {MFTL_NULL, 0, 0};
    run_info info;
    run_pipeline(0, 1, &p, &local, &info);
    ks_gather(&local, &ref);
    ks_free(&local);

    MFTLEnsembleDestroy(&g_ens);
  }

  int n_ref = ref.n;
  MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(n_ref > 0, MPI_COMM_WORLD));

  /* ── The same configuration, reached by migrating into it ───────────────── */
  plain_ensemble(10.0, false);
  add_sphere(0, start_a, 1.0, 2);
  add_sphere(1, start_b, 1.0, 2);
  refresh();
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  /* Sanity: the scene really did start somewhere else. */
  int moved_ranks = 0;
  {
    const int held = (bodyv(0)->tris.n_local > 0) ? 1 : 0;
    MPI_Allreduce(&held, &moved_ranks, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  FOREACH_BODY_VERTEX(0, {
    pos[0] += step;
    pos[1] += step;
    pos[2] += step;
  });
  FOREACH_BODY_VERTEX(1, {
    pos[0] += step;
    pos[1] += step;
    pos[2] += step;
  });

  /* The documented per-timestep order: pull the vertices the neighbours moved,
   * recompute the centres for ALL triangles INCLUDING GHOSTS -- redistribution
   * routes ghosts by their cached centre, so refreshing only the owned ones
   * makes two ranks disagree about where a ghost is going -- then repartition. */
  MFTLUpdateGhostPositions(g_ens);
  for (int i = 0; i < g_ens->n_local_objects; ++i) MFTLComputeTriangleGeometry(g_ens->local_objects[i], g_ens->local_objects[i]->tris.n);
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  keyset local_got = {MFTL_NULL, 0, 0};
  keyset got = {MFTL_NULL, 0, 0};
  run_info gi;
  run_pipeline(0, 1, &p, &local_got, &gi);
  ks_gather(&local_got, &got);
  ks_free(&local_got);

  int same = 1;
  if (g_rank == 0) same = ks_equal(&ref, &got, "migration");
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(moved_ranks > 0, MPI_COMM_WORLD));

  ks_free(&ref);
  ks_free(&got);
  return 0;
}

/**
 * @brief The contact predicate at exactly the tolerance, from both sides.
 *
 * `MFTL_TriTriProximity` declares contact at `distance <= tolerance`, and the
 * cases that decide whether a lubrication model switches on live within an ulp
 * of that threshold. Two spheres cannot be placed at an exact surface
 * separation -- the discretisation moves the surface inside the sphere by an
 * amount that depends on the refinement -- so this uses two FLAT patches, whose
 * triangle-triangle distance is exactly the coordinate difference.
 *
 * Every number here is a negative power of two, so the gap, the tolerance and
 * their squares are all exact in double and the comparison is not a coin flip on
 * the last bit. The patches straddle a rank boundary, so the threshold is
 * evaluated on a pair the halo had to deliver.
 */
static int test_the_tolerance_threshold_is_exact_from_both_sides(void)
{
  const double tol = 0.0625;  // 2^-4
  const double eps = 0.0009765625;  // 2^-10, comfortably above any rounding here
  const double gaps[3] = {tol - eps, tol, tol + eps};
  const int expect[3] = {1, 1, 0};  // "<=" means the exact hit is a contact

  for (int g = 0; g < 3; ++g)
  {
    MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
    p.proximity_factor = 0.0;  // pure absolute envelope: the gap IS the physics
    p.proximity_floor = tol;

    if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);

    int cuts[3];
    choose_cuts(g_nproc, cuts);
    double lo[3] = {0.0, 0.0, 0.0};
    double hi[3] = {8.0, 8.0, 8.0};
    bool per[3] = {false, false, false};
    g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

    /* The cut plane the patches straddle, so the pair is cross-rank wherever the
     * grid has more than one cell on y. */
    const double cy = (cuts[1] > 1) ? 8.0 / (double) cuts[1] : 4.0;
    const double cz = (cuts[2] > 1) ? 8.0 / (double) cuts[2] : 4.0;
    const double x0 = 4.0;

    const double orgA[3] = {x0 - 0.5 * gaps[g], cy - 1.0, cz - 1.0};
    const double orgB[3] = {x0 + 0.5 * gaps[g], cy - 1.0, cz - 1.0};
    const double u[3] = {0.0, 2.0, 0.0};
    const double v[3] = {0.0, 0.0, 2.0};

    add_patch(0, orgA, u, v, 4, 4);
    add_patch(1, orgB, u, v, 4, 4);
    refresh();

    TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
    refresh();

    keyset ks = {MFTL_NULL, 0, 0};
    run_info info;
    run_pipeline(0, 1, &p, &ks, &info);

    const int n = ks.n;
    int total = 0;
    MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    ks_free(&ks);

    /* The envelope must be the floor exactly: a relative factor of zero means
     * the body size must not enter at all. */
    TEST_ASSERT(agreed(info.tolerance == tol, MPI_COMM_WORLD));

    if (expect[g])
    {
      if (total == 0 && g_rank == 0) printf("\n  [threshold] gap %.17g <= tol %.17g reported nothing\n  ", gaps[g], tol);
      TEST_ASSERT(agreed(total > 0, MPI_COMM_WORLD));
    }
    else
    {
      if (total != 0 && g_rank == 0) printf("\n  [threshold] gap %.17g > tol %.17g reported %d pairs\n  ", gaps[g], tol, total);
      TEST_ASSERT(agreed(total == 0, MPI_COMM_WORLD));
    }
  }

  return 0;
}

/**
 * @brief A micrometre droplet and a metre body in one run, and what that costs.
 *
 * This is a CHARACTERISATION, not a correctness assertion: it records a
 * consequence of a deliberate design choice so that changing the choice is
 * visible rather than accidental.
 *
 * `MFTL_IntersectObjects` tests a pair against `MAX(tol_A, tol_B)`. The maximum
 * is the only safe direction for the BROAD phase -- each tree's leaves are
 * inflated by its own tolerance, so admitting the larger one is what keeps the
 * narrow-phase threshold inside what the traversal can deliver. But it means the
 * envelope of a pair is set by the LARGER BODY, so a droplet a thousand times
 * smaller than its partner is judged against an envelope that may exceed its own
 * diameter.
 *
 * Two statements are asserted:
 *   - both bodies are detected at their own scale, which is what the scaled
 *     tolerance bought over the inherited absolute 1e-4 / 1e-6 pair;
 *   - the pair envelope really is the big body's, and it really does exceed the
 *     small body's radius. A caller that needs the small body's own gap must
 *     therefore state it as `proximity_floor` and set `proximity_factor` to zero,
 *     which is what that field is for.
 */
static int test_two_length_scales_share_the_larger_envelope(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-2;
  p.proximity_floor = 0.0;

  plain_ensemble(4.0, false);

  const double r_big = 1.0;  // a metre-scale body, in metres
  const double r_small = 1.0e-3;  // a millimetre droplet

  const double c_big[3] = {1.0, 2.0, 2.0};
  const double c_small[3] = {1.0 + r_big + 0.5 * r_small, 2.0, 2.0};  // just touching
  add_sphere(0, c_big, r_big, 2);
  add_sphere(1, c_small, r_small, 2);
  refresh();

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(g_ens);
  MFTLSpatialHaloResolveRadii(g_ens, halo);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(bodyv(0), &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(bodyv(1), &p);
  MFTL_BVHSetGlobalRadius(ba, MFTLSpatialHaloRadius(halo, 0));
  MFTL_BVHSetGlobalRadius(bb, MFTLSpatialHaloRadius(halo, 1));

  /* Each body's OWN envelope is a fixed fraction of itself -- three orders of
   * magnitude apart, which is exactly what a single hardcoded length cannot do. */
  TEST_ASSERT(agreed(ba->tolerance > 5.0e-3 && ba->tolerance < 2.0e-2, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(bb->tolerance > 5.0e-6 && bb->tolerance < 2.0e-5, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ba->tolerance > 100.0 * bb->tolerance, MPI_COMM_WORLD));

  /* And the PAIR uses the larger, which here is bigger than the droplet itself.
   * Recorded deliberately: this is the price of the max rule, and the escape is
   * proximity_floor. */
  const double pair_tol = MFTL_MAX(ba->tolerance, bb->tolerance);
  TEST_ASSERT(agreed(pair_tol == ba->tolerance, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(pair_tol > r_small, MPI_COMM_WORLD));

  MFTL_SpatialHaloRequest req = {0, 1, pair_tol};
  MFTLExtendSpatialHalo(g_ens, &req, 1, halo);
  MFTL_BVHSetHalo(bb, MFTLSpatialHaloPatch(halo, 1));

  collector c;
  c.ks.k = MFTL_NULL;
  c.ks.n = 0;
  c.ks.cap = 0;
  c.ba = ba;
  c.bb = bb;
  c.self_pairs = 0;
  MFTL_IntersectObjects(bodyv(0), ba, bodyv(1), bb, collect, &c);

  int total = 0;
  MPI_Allreduce(&c.ks.n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(total > 0, MPI_COMM_WORLD));

  ks_free(&c.ks);
  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/**
 * @brief Partitions with 0, 1 and 2 triangles, which are the LBVH's edge cases.
 *
 * A tree over one primitive has no internal node at all; over two it has exactly
 * one, and the radix split has no differing bit to split on if both Morton codes
 * are equal. Over zero there is no root. All three occur constantly in
 * production -- a rank at the edge of the process grid holding the last sliver
 * of a body -- and all three are reached here from the pool side rather than by
 * hoping the redistribution produces them.
 *
 * The empty case is the one with a collective consequence: a rank holding
 * nothing still has to enter the radius reduction, the extent allgather and the
 * payload exchange, or the ranks that do have work hang.
 */
static int test_degenerate_partitions_build_and_traverse(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-2;
  p.proximity_floor = 1.0e-6;

  plain_ensemble(10.0, false);

  /* Two single triangles, overlapping, plus an empty body. */
  MFTL_Object *const a = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(g_ens, a);
  MFTL_Object *const b = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(g_ens, b);

  if (g_rank == 0)
  {
    const double pa[9] = {4.0, 4.0, 5.0, 6.0, 4.0, 5.0, 5.0, 6.0, 5.0};
    const int ta[3] = {0, 1, 2};
    MFTLBuildIndexedMesh(a, pa, 3, ta, 1);

    /* Two triangles forming a quad that crosses A's plane. */
    const double pb[12] = {5.0, 4.5, 4.0, 5.0, 4.5, 6.0, 5.0, 5.5, 6.0, 5.0, 5.5, 4.0};
    const int tb[6] = {0, 1, 2, 0, 2, 3};
    MFTLBuildIndexedMesh(b, pb, 4, tb, 2);
  }
  else
  {
    MFTLObjectBuildDerived(a);
    MFTLObjectBuildDerived(b);
  }

  for (int i = 0; i < 2; ++i)
  {
    MFTL_Object *const o = bodyv(i);
    int gids[3] = {o->next_vertex_gid, o->next_triangle_gid, o->next_halfedge_gid};
    MPI_Bcast(gids, 3, MPI_INT, 0, MPI_COMM_WORLD);
    o->next_vertex_gid = gids[0];
    o->next_triangle_gid = gids[1];
    o->next_halfedge_gid = gids[2];
    MFTLObjectUpdateGlobalStats(o);
  }
  refresh();

  /* Before redistribution: rank 0 has 1 and 2 primitives, every other rank has
   * none. All three degenerate counts, in one configuration. */
  {
    MFTL_BVH *const ba = MFTL_GenerateLBVH(a, &p);
    MFTL_BVH *const bb = MFTL_GenerateLBVH(b, &p);

    TEST_ASSERT(agreed(ba != MFTL_NULL && bb != MFTL_NULL, MPI_COMM_WORLD));
    if (g_rank == 0)
    {
      TEST_ASSERT(ba->n_prims == 1 && ba->nodes_used == 1);
      TEST_ASSERT(bb->n_prims == 2);
    }
    else
    {
      TEST_ASSERT(ba->n_prims == 0 && bb->n_prims == 0);
    }

    /* A refit of a one-node and a zero-node tree must be a no-op, not a crash. */
    TEST_ASSERT(agreed(MFTL_RefitLBVH(ba, a) == MFTL_SUCCESS, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTL_RefitLBVH(bb, b) == MFTL_SUCCESS, MPI_COMM_WORLD));

    collector c;
    c.ks.k = MFTL_NULL;
    c.ks.n = 0;
    c.ks.cap = 0;
    c.ba = ba;
    c.bb = bb;
    c.self_pairs = 0;
    MFTL_IntersectObjects(a, ba, b, bb, collect, &c);

    int total = 0;
    MPI_Allreduce(&c.ks.n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(total > 0, MPI_COMM_WORLD));  // the quad really does cross the triangle
    ks_free(&c.ks);

    MFTL_DestroyBVH(ba);
    MFTL_DestroyBVH(bb);
  }

  /* And the whole collective pipeline over a mesh most ranks hold none of. */
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  {
    keyset ks = {MFTL_NULL, 0, 0};
    run_info info;
    run_pipeline(0, 1, &p, &ks, &info);
    const int n = ks.n;
    int total = 0;
    MPI_Allreduce(&n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    TEST_ASSERT(agreed(total > 0, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(info.patch_defect == 0, MPI_COMM_WORLD));
    ks_free(&ks);
  }

  return 0;
}

/**
 * @brief A body wider than a subdomain, and why ONE-HOP ghosting is still enough.
 *
 * ── The expectation, stated ────────────────────────────────────────────────
 *
 * MFTLExtendSpatialHalo only ever sends to the 26 ranks of `domain->comm_graph`.
 * The obvious worry is that a body spanning several subdomains needs triangles
 * from ranks two or more hops away, and that the halo would silently lose them.
 *
 * IT DOES NOT, and the reason is that the reach of a contact has nothing to do
 * with the size of the body. Ownership is decided by a triangle's CENTRE, so if
 * triangle a (centre in rank p) and triangle b (centre in rank q) are within the
 * envelope of each other, then
 *
 *     dist(cell p, cell q) <= diam(a)/2 + diam(b)/2 + tol
 *
 * and p and q are geometric neighbours whenever that is under one subdomain
 * width. The condition is on the TRIANGLE, not the object -- so a droplet
 * spanning ten ranks is fine, and a coarse mesh whose individual triangles are
 * wider than a subdomain is not. The latter is the real limit, it is a statement
 * about resolution against decomposition rather than about this module, and it
 * is already outside what redistribution's own edge-adjacent ghost layer
 * supports.
 *
 * The test puts a body four subdomains wide on the thin axis and asserts the
 * answer is still exact. If multi-hop ghosting is ever genuinely needed, this is
 * the test that will start failing, and the fix belongs in the neighbourhood
 * definition rather than in the halo.
 */
static int test_a_body_wider_than_a_subdomain_is_still_exact(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  int cuts[3];
  choose_cuts(g_nproc, cuts);

  /* A deliberately thin z axis: at np = 8 the subdomain is 1.0 deep and at
   * np = 27 it is 0.667, against a body of diameter 2. */
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {10.0, 10.0, 2.0};
  bool per[3] = {false, false, false};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

  const double cz = 1.0;
  const double cx = (cuts[0] > 1) ? 10.0 / (double) cuts[0] : 5.0;
  const double ca[3] = {cx - 0.8, 5.0, cz};
  const double cb[3] = {cx + 0.8, 5.0, cz};
  add_sphere(0, ca, 1.0, 2);
  add_sphere(1, cb, 1.0, 2);
  refresh();

  keyset local_ref = {MFTL_NULL, 0, 0};
  keyset ref = {MFTL_NULL, 0, 0};
  run_info ri;
  run_pipeline(0, 1, &p, &local_ref, &ri);
  ks_gather(&local_ref, &ref);
  ks_free(&local_ref);

  int n_ref = ref.n;
  MPI_Bcast(&n_ref, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(n_ref > 0, MPI_COMM_WORLD));

  /* The premise: the body really is wider than a cell on at least one axis. */
  const double cell_z = 2.0 / (double) cuts[2];
  TEST_ASSERT(agreed(g_nproc == 1 || cuts[2] == 1 || 2.0 > cell_z, MPI_COMM_WORLD));

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  keyset local_got = {MFTL_NULL, 0, 0};
  keyset got = {MFTL_NULL, 0, 0};
  run_info gi;
  run_pipeline(0, 1, &p, &local_got, &gi);
  ks_gather(&local_got, &got);
  ks_free(&local_got);

  int same = 1;
  if (g_rank == 0) same = ks_equal(&ref, &got, "wider-than-a-subdomain");
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));

  ks_free(&ref);
  ks_free(&got);
  return 0;
}

/**
 * @brief Many bodies, few of them near each other: the ANSWER is the all-pairs
 *        answer and the WORK is not.
 *
 * ══ WHAT THIS REPLACES, AND WHY IT NEEDED REPLACING ═══════════════════════
 *
 * MFTLEnsembleIntersectObjects used to enumerate `B*(B-1)/2` ordered pairs into
 * a request array, on every rank, every call -- 134 million records and two
 * gigabytes at 16 384 bodies, before a triangle was looked at. The pairs are now
 * DERIVED from the body extents the halo exchange already carries, so the
 * question this test asks is the one that matters about that change: does the
 * cheaper enumeration still find every contact, and is it actually cheaper?
 *
 * ══ THE FIXTURE ═══════════════════════════════════════════════════════════
 *
 * Six well-separated COUPLES. Bodies 2k and 2k+1 overlap; nothing else is within
 * reach of anything. So of the 66 unordered pairs exactly 6 are contacts, which
 * makes both halves of the statement checkable:
 *
 *   COMPLETE   the total contact count must equal an exhaustive O(nA*nB) sweep
 *              over ALL 66 pairs, run on one rank while it still holds the whole
 *              ensemble. That oracle shares no code with the BVH, the halo or
 *              the broad phase, so a bug in what they have in common cannot hide
 *              by appearing on both sides.
 *   TIGHT      every candidate pair the broad phase kept must be one of the six.
 *              A grid that admitted a far-apart pair would still give the right
 *              answer -- the traversal would reject it -- so counting hits alone
 *              cannot see it, and the all-pairs enumeration passes that half
 *              trivially.
 *
 * And the directory is asserted to be SPARSE: summed over ranks it must be below
 * `nproc * bodies`, which is what the `[n_bodies_global]` table it replaced
 * always was.
 */
static int test_the_broad_phase_finds_the_pairs_without_enumerating_them(void)
{
  enum
  {
    N_COUPLES = 6,
    N_BODIES = 2 * N_COUPLES
  };

  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 1.0e-3;
  p.proximity_floor = 0.0;

  plain_ensemble(12.0, true);

  /* Couple k at (2 + 4*(k%3), 2 + 4*(k/3), 6). The two spheres of a couple are
   * 1.5 apart with radius 0.8, so they overlap; the nearest body of another
   * couple is 4 - 2*1.55 = 0.9 away, which is three orders above the envelope. */
  for (int k = 0; k < N_COUPLES; ++k)
  {
    const double cx = 2.0 + 4.0 * (double) (k % 3);
    const double cy = 2.0 + 4.0 * (double) (k / 3);

    const double a[3] = {cx - 0.75, cy, 6.0};
    const double b[3] = {cx + 0.75, cy, 6.0};
    add_sphere(2 * k, a, 0.8, 1);
    add_sphere(2 * k + 1, b, 0.8, 1);
  }
  refresh();

  /* ── The oracle, while rank 0 holds everything ─────────────────────────── */
  int ref_hits = 0;
  if (g_rank == 0)
  {
    /* The envelope each body would be judged at. Every body is the same size
     * here, so this is one number, but it is READ off a BVH rather than assumed
     * -- the point of the comparison is that the pipeline and the oracle test
     * against the same threshold. */
    MFTL_BVH *const probe = MFTL_GenerateLBVH(bodyv(0), &p);
    const double tol = probe->tolerance;
    MFTL_DestroyBVH(probe);

    for (int i = 0; i < N_BODIES; ++i)
    {
      for (int j = i + 1; j < N_BODIES; ++j)
      {
        keyset one = {MFTL_NULL, 0, 0};
        brute_contacts(bodyv(i), bodyv(j), tol, &one);
        ref_hits += one.n;
        ks_free(&one);
      }
    }
  }
  MPI_Bcast(&ref_hits, 1, MPI_INT, 0, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(ref_hits > 0, MPI_COMM_WORLD));

  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  /* ── COMPLETE: the distributed pipeline finds exactly those ────────────── */
  {
    tally t = {0, 0};
    TEST_ASSERT(agreed(MFTLEnsembleIntersectObjects(g_ens, &p, count_pairs, &t) == MFTL_SUCCESS, MPI_COMM_WORLD));

    int got = 0, self = 0;
    MPI_Allreduce(&t.hits, &got, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&t.self, &self, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (g_rank == 0 && got != ref_hits) printf("\n  [broad-phase] exhaustive sweep %d contacts, pipeline %d\n  ", ref_hits, got);
    TEST_ASSERT(agreed(got == ref_hits, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(self == 0, MPI_COMM_WORLD));
  }

  /* ── TIGHT: what the broad phase actually proposed ─────────────────────── */
  {
    MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(g_ens);
    MFTLSpatialHaloResolveRadii(g_ens, halo);

    const int nb = g_ens->n_local_objects;
    double *const tol = (double *) calloc((size_t) (nb > 0 ? nb : 1), sizeof(double));
    for (int o = 0; o < nb; ++o)
    {
      MFTL_BVH *const one = MFTL_GenerateLBVH(g_ens->local_objects[o], &p);
      MFTL_BVHSetGlobalRadius(one, MFTLSpatialHaloRadius(halo, g_ens->local_objects[o]->global_id));
      tol[o] = one->tolerance;
      MFTL_DestroyBVH(one);
    }

    TEST_ASSERT(agreed(MFTLExtendSpatialHaloNearBodies(g_ens, tol, halo) == MFTL_SUCCESS, MPI_COMM_WORLD));
    free(tol);

    /* Every kept pair must be a couple: `2k` with `2k+1`. */
    int stray = 0;
    int found = 0;  // bitmask of the couples this rank proposed
    for (int i = 0; i < halo->n_pairs; ++i)
    {
      const int a = halo->pair[i].body_a;
      const int b = halo->pair[i].body_b;
      if (b == a + 1 && (a % 2) == 0)
        found |= 1 << (a / 2);
      else
        stray++;
    }

    int stray_total = 0, found_all = 0, dir_total = 0, pairs_total = 0;
    MPI_Allreduce(&stray, &stray_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&found, &found_all, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
    MPI_Allreduce(&halo->num_objects, &dir_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&halo->n_pairs, &pairs_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    if (g_rank == 0)
    {
      printf("\n    np=%d: %d candidate pairs over all ranks (all-pairs would be %d), directory %d slots (dense would be %d)\n  ", g_nproc, pairs_total,
             g_nproc * N_BODIES * (N_BODIES - 1) / 2, dir_total, g_nproc * N_BODIES);
    }

    /* No pair between two couples was ever proposed... */
    TEST_ASSERT(agreed(stray_total == 0, MPI_COMM_WORLD));
    /* ...and every couple WAS, so the tightness above is not the tightness of an
     * empty list. */
    TEST_ASSERT(agreed(found_all == (1 << N_COUPLES) - 1, MPI_COMM_WORLD));
    /* And the directory is sparse: the table this replaced was `n_bodies_global`
     * slots on every rank, unconditionally. */
    TEST_ASSERT(agreed(g_nproc == 1 || dir_total < g_nproc * N_BODIES, MPI_COMM_WORLD));

    MFTLSpatialHaloDestroy(&halo);
  }

  return 0;
}

/**
 * @brief The refit fast path must agree with a full rebuild, on a PARTITIONED mesh.
 *
 * tests/test_bvh.c already checks that refit == rebuild after small motion on
 * one rank. What it cannot check is that the equivalence survives a partition,
 * where the tree also carries a spatial halo: `MFTL_RefitLBVH` deliberately does
 * NOT re-run the exchange, so the remote primitives keep the positions they
 * arrived with. That is correct only while the neighbours' surfaces move less
 * than the tolerance, and the two answers must agree in exactly that regime.
 */
static int test_refit_matches_rebuild_on_a_partitioned_mesh(void)
{
  MFTL_BVHBuildParams p = MFTL_BVHDefaultParams();
  p.proximity_factor = 2.0e-2;
  p.proximity_floor = 0.0;

  plain_ensemble(10.0, false);

  int cuts[3];
  choose_cuts(g_nproc, cuts);
  const double cx = (cuts[0] > 1) ? 10.0 / (double) cuts[0] : 5.0;

  const double ca[3] = {cx - 0.8, 5.0, 5.0};
  const double cb[3] = {cx + 0.8, 5.0, 5.0};
  add_sphere(0, ca, 1.0, 2);
  add_sphere(1, cb, 1.0, 2);
  refresh();
  TEST_ASSERT(agreed(MFTLRedistribute(g_ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh();

  MFTL_SpatialHalo *halo = MFTLSpatialHaloCreate(g_ens);
  MFTLSpatialHaloResolveRadii(g_ens, halo);

  MFTL_BVH *const ba = MFTL_GenerateLBVH(bodyv(0), &p);
  MFTL_BVH *const bb = MFTL_GenerateLBVH(bodyv(1), &p);
  MFTL_BVHSetGlobalRadius(ba, MFTLSpatialHaloRadius(halo, 0));
  MFTL_BVHSetGlobalRadius(bb, MFTLSpatialHaloRadius(halo, 1));

  MFTL_SpatialHaloRequest req = {0, 1, MFTL_MAX(ba->tolerance, bb->tolerance)};
  MFTLExtendSpatialHalo(g_ens, &req, 1, halo);
  MFTL_BVHSetHalo(bb, MFTLSpatialHaloPatch(halo, 1));

  /* A motion small against the tolerance, so the stale remote geometry is still
   * within the regime the refit path documents. */
  FOREACH_BODY_VERTEX(0, { pos[0] += 1.0e-4; });
  MFTLUpdateGhostPositions(g_ens);
  refresh();

  TEST_ASSERT(agreed(MFTL_RefitLBVH(ba, bodyv(0)) == MFTL_SUCCESS, MPI_COMM_WORLD));

  collector cr;
  cr.ks.k = MFTL_NULL;
  cr.ks.n = 0;
  cr.ks.cap = 0;
  cr.ba = ba;
  cr.bb = bb;
  cr.self_pairs = 0;
  MFTL_IntersectObjects(bodyv(0), ba, bodyv(1), bb, collect, &cr);
  ks_sort(&cr.ks);

  /* The same query from a tree built from scratch on the moved geometry. */
  MFTL_BVH *const ba2 = MFTL_GenerateLBVH(bodyv(0), &p);
  MFTL_BVHSetGlobalRadius(ba2, MFTLSpatialHaloRadius(halo, 0));

  collector cf;
  cf.ks.k = MFTL_NULL;
  cf.ks.n = 0;
  cf.ks.cap = 0;
  cf.ba = ba2;
  cf.bb = bb;
  cf.self_pairs = 0;
  MFTL_IntersectObjects(bodyv(0), ba2, bodyv(1), bb, collect, &cf);
  ks_sort(&cf.ks);

  int same = (cr.ks.n == cf.ks.n);
  for (int i = 0; same && i < cr.ks.n; ++i) same = (cr.ks.k[i] == cf.ks.k[i]);
  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));

  int total = 0;
  MPI_Allreduce(&cr.ks.n, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(agreed(total > 0, MPI_COMM_WORLD));

  ks_free(&cr.ks);
  ks_free(&cf.ks);
  MFTL_DestroyBVH(ba);
  MFTL_DestroyBVH(ba2);
  MFTL_DestroyBVH(bb);
  MFTLSpatialHaloDestroy(&halo);
  return 0;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();

  if (g_rank == 0)
  {
    int cuts[3];
    choose_cuts(g_nproc, cuts);
    printf("--- Running MFTL BVH Ghosting Tests (np = %d, grid %dx%dx%d, %d placements) ---\n", g_nproc, cuts[0], cuts[1], cuts[2], N_CASES);
  }

  RUN_TEST(test_the_distributed_answer_matches_the_unpartitioned_one);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_a_seam_straddling_body_is_not_measured_as_domain_sized);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_contact_through_the_seam_survives_the_decomposition);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_a_self_overlapping_body_reports_no_self_contact);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_tunnelling_between_two_timesteps_is_a_known_miss);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_a_body_that_migrates_wholesale_keeps_the_answer);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_the_tolerance_threshold_is_exact_from_both_sides);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_two_length_scales_share_the_larger_envelope);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_degenerate_partitions_build_and_traverse);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_a_body_wider_than_a_subdomain_is_still_exact);

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  RUN_TEST(test_the_broad_phase_finds_the_pairs_without_enumerating_them);

  RUN_TEST(test_refit_matches_rebuild_on_a_partitioned_mesh);

  if (g_rank == 0) TEST_REPORT();

  if (g_ens != MFTL_NULL) MFTLEnsembleDestroy(&g_ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();

  return (g_tests_failed > 0) ? 1 : 0;
}
