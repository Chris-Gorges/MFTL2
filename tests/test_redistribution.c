/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for redistribution and ghost synchronisation.
 *
 * Registered at several process counts (see tests/CMakeLists.txt), because the
 * rank count changes which code path is reachable:
 *
 *   np = 1   the no-op path, and the guarantee that a single-rank mesh survives
 *            a redistribution untouched
 *   np = 2   a thin 2x1x1 grid, where periodic wrap-around maps several of the
 *            26 neighbour offsets onto the same rank
 *   np = 4   2x2x1: the first grid with a genuine diagonal neighbour
 *   np = 8   2x2x2: every one of the 26 offsets is populated
 *
 * EVERY assertion goes through agreed(): TEST_ASSERT returns immediately, so an
 * assertion failing on only some ranks would leave the rest blocked in the next
 * collective -- a hang instead of a clean failure.
 *
 * ── One path no rank count here can reach ──────────────────────────────────
 *
 * The vertex de-duplication in src/communication/redistribute.c hands out one bit
 * per remote destination and falls back to a sort-and-unique for destinations
 * beyond MFTL_DEDUP_BITS. Reaching that needs an object with more than 32 remote
 * destinations, i.e. more ranks than a test machine has. To exercise it, set
 * MFTL_DEDUP_BITS to 1 and run this suite -- every test must still pass, because
 * correctness never depends on the bound. Done, and it does; the define is back
 * at 32 because the fallback is the slow path, not the wrong one.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static int g_rank = 0;
static int g_nproc = 1;

/* ─────────────────────────────────────────────────────────────────────────────
 * Fixtures
 * ─────────────────────────────────────────────────────────────────────────── */

static MFTL_Ensemble *make_ensemble(bool periodic, int capacity)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);

  double lo[3] = {-1.0, -1.0, -1.0};
  double hi[3] = {1.0, 1.0, 1.0};
  bool per[3] = {periodic, periodic, periodic};

  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, capacity);
}

/**
 * @brief Adds a generated sphere with its triangle geometry already computed.
 *
 * The centres must exist before the first redistribution: ownership is decided
 * from them, and a freshly generated mesh has none.
 */
static MFTL_Object *add_sphere(MFTL_Ensemble *ens, int id, double cx, double cy, double cz, double r, int refine)
{
  MFTL_Object *obj = MFTLObjectCreate(id);
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {cx, cy, cz};
  MFTLGenerateSphere(obj, centre, r, refine);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  return obj;
}

/**
 * @brief The body with @p gid, or MFTL_NULL if this rank now holds nothing of it.
 *
 * A REDISTRIBUTION DOES NOT PRESERVE MFTL_Object POINTERS. A rank that ends up
 * holding no entity of a body destroys its object, and one that receives a body
 * it had nothing of creates a new one -- so a pointer cached before the call may
 * be dangling after it, and the body may be at a different slot. The body id is
 * the handle that survives, which is the same rule global ids and pool indices
 * already follow across a remesh.
 *
 * Every test below re-resolves through this after every redistribution, and that
 * is not defensive style: it is the contract being exercised.
 */
static MFTL_Object *body(const MFTL_Ensemble *ens, int gid) { return MFTLEnsembleFindLocalObject(ens, gid); }

/**
 * @brief MFTL_FOREACH_OWNED_VERTEX over body @p gid, re-resolved, no-op if absent.
 *
 * The shape a solver actually has to use: the object is looked up by id at the
 * point of use rather than cached across the redistribution that may have
 * destroyed it, and a rank that holds none of the body simply moves nothing.
 */
#define FOREACH_BODY_VERTEX(ens, gid, ...)                     \
  do                                                           \
  {                                                            \
    MFTL_Object *const _b = body((ens), (gid));                \
    if (_b != MFTL_NULL) MFTL_FOREACH_OWNED_VERTEX(_b, __VA_ARGS__); \
  } while (0)

/** @brief Refreshes geometry for every local triangle. */
static void refresh_geometry(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }
}

/** @brief Owned triangle count, owned vertex count, owned surface area. */
static void owned_totals(const MFTL_Ensemble *ens, double out[3])
{
  out[0] = out[1] = out[2] = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    out[0] += (double) obj->tris.n_local;
    out[1] += (double) obj->verts.n_local;
    for (int t = 0; t < obj->tris.n_local; ++t) out[2] += obj->tris.area[t];
  }
}

static void global_totals(const MFTL_Ensemble *ens, double out[3])
{
  double local[3];
  owned_totals(ens, local);
  MPI_Allreduce(local, out, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/** @brief 1 if every object of the ensemble satisfies inv_partition_is_sound. */
static int ensemble_is_sound(const MFTL_Ensemble *ens)
{
  /* The directory itself, before the meshes in it: sorted, no empty object, every
   * id inside the space. A redistribution that failed to destroy a departed body
   * leaves every per-object check below passing. */
  if (!inv_local_directory_is_sound(ens)) return 0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    if (!inv_partition_is_sound(ens->local_objects[o], g_rank)) return 0;
  }
  return 1;
}

/**
 * @brief Every global id is owned by EXACTLY one rank, for both entity classes.
 *
 * The property that makes "sum the owned entities" a correct global reduction,
 * and the one a de-duplication bug destroys without breaking any local check:
 * two ranks both owning a triangle, or nobody owning it, are both invisible from
 * inside a single partition.
 *
 * @note Collective.
 */
static int ownership_is_exclusive(const MFTL_Ensemble *ens, int gid)
{
  const MFTL_Object *const obj = body(ens, gid);

  /* The id RANGES have to be agreed before anything is sized by them. A rank
   * holding nothing of the body knows neither, and every path below is
   * collective, so deriving them locally would have that rank size its buffers
   * differently -- or return early and leave the others in the Allreduce. */
  int range[2] = {obj ? obj->next_triangle_gid : 0, obj ? obj->next_vertex_gid : 0};
  MPI_Allreduce(MPI_IN_PLACE, range, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  const int nt = range[0];
  const int nv = range[1];
  const int n = (nt > nv) ? nt : nv;
  if (n <= 0) return 1;

  int *mine = (int *) calloc((size_t) n, sizeof(int));
  int *total = (int *) malloc((size_t) n * sizeof(int));
  if (!mine || !total)
  {
    free(mine);
    free(total);
    return 0;
  }

  int ok = 1;

  for (int t = 0; obj != MFTL_NULL && t < obj->tris.n_local; ++t)
  {
    const int g = obj->tris.rec[t].global_id;
    if (g < 0 || g >= nt) ok = 0;
    else mine[g]++;
  }
  MPI_Allreduce(mine, total, nt, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int g = 0; g < nt; ++g)
    if (total[g] != 1) ok = 0;

  memset(mine, 0, (size_t) n * sizeof(int));
  for (int v = 0; obj != MFTL_NULL && v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    if (g < 0 || g >= nv) ok = 0;
    else mine[g]++;
  }
  MPI_Allreduce(mine, total, nv, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  for (int g = 0; g < nv; ++g)
    if (total[g] != 1) ok = 0;

  free(mine);
  free(total);
  return ok;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * A partition digest, for the skip
 *
 * The same construction examples/parallel_advection_redistribute_list.c uses,
 * because the same question is being asked: is one way of arriving at a
 * partition indistinguishable from another?
 *
 * TWO digests, and the pair is the point. The SET digest is a sum, so it depends
 * on what the partition contains and not on where in the pool anything sits; the
 * ORDER digest is a chain, so it also depends on the slot order. A skip that got
 * the contents right and the order wrong would pass the first and fail the
 * second -- which is exactly what happened before the settling rule went in.
 *
 * Ghosts contribute identity and owner only, which is what makes halo MEMBERSHIP
 * part of the comparison without making the halo's storage order part of it for
 * the SET form.
 * ─────────────────────────────────────────────────────────────────────────── */

/** @brief splitmix64, so that a one-bit change of any input changes the digest. */
static uint64_t dg_mix(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/** @brief Folds one value into a running hash. */
static uint64_t dg_in(uint64_t h, uint64_t v) { return dg_mix(h ^ dg_mix(v)); }

/** @brief Reinterprets a double as its bit pattern, so equality is exact. */
static uint64_t dg_bits(double d)
{
  uint64_t u;
  memcpy(&u, &d, sizeof u);
  return u;
}

/**
 * @brief `{set digest, slot-order digest}` of a whole ensemble, summed over ranks.
 * @param      ens The ensemble.
 * @param[out] out Receives the two reduced digests.
 * @note Collective.
 */
static void partition_digest(const MFTL_Ensemble *ens, uint64_t out[2])
{
  uint64_t mine[2] = {0, 0x243f6a8885a308d3ULL};

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    const uint64_t body = (uint64_t) obj->global_id;

    for (int v = 0; v < obj->verts.n; ++v)
    {
      const bool owned = (v < obj->verts.n_local);
      uint64_t h = dg_in(dg_in(body, (uint64_t) obj->verts.rec[v].global_id), owned ? 1u : 2u);
      h = dg_in(h, (uint64_t) obj->verts.rec[v].owner_proc);
      /* Positions for EVERY vertex, ghosts included: a skip retains a ghost's
       * local record rather than replacing it from the wire, so this is the one
       * place that difference could show up. */
      for (int k = 0; k < 3; ++k) h = dg_in(h, dg_bits(obj->verts.pos[3 * v + k]));
      h = dg_in(h, (uint64_t) obj->verts.rec[v].flags);
      mine[0] += h;
      mine[1] = dg_in(mine[1], h);
    }

    for (int t = 0; t < obj->tris.n; ++t)
    {
      const bool owned = (t < obj->tris.n_local);
      uint64_t h = dg_in(dg_in(body, (uint64_t) obj->tris.rec[t].global_id), owned ? 3u : 4u);
      h = dg_in(h, (uint64_t) obj->tris.rec[t].owner_proc);
      h = dg_in(h, (uint64_t) obj->tris.rec[t].flags);
      for (int k = 0; k < 3; ++k)
      {
        const MFTL_HalfEdge *const he = &obj->tris.he[3 * t + k];
        h = dg_in(h, (uint64_t) obj->verts.rec[he->vertex_idx].global_id);
        h = dg_in(h, (uint64_t) he->flags);
        h = dg_in(h, (uint64_t) he->global_id);

        /* The partner named by IDENTITY -- its triangle's global id and its
         * corner -- rather than by slot, for the same reason the whole digest is
         * gid-keyed: the index is storage, the pair is the topology. */
        const int tw = he->twin_idx;
        if (tw == MFTL_ERROR) h = dg_in(h, 0xffffffffULL);
        else h = dg_in(h, dg_in((uint64_t) obj->tris.rec[tw / 3].global_id, (uint64_t) (tw % 3)));
      }
      mine[0] += h;
      mine[1] = dg_in(mine[1], h);
    }
  }

  /* Summed over ranks: unsigned wrap-around is defined, and it makes the result
   * describe WHAT the partition is rather than which rank happens to hold what. */
  MPI_Allreduce(mine, out, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
}

/** @brief Bodies redistribution has left alone since the counters were reset. */
static long skipped_so_far(void)
{
  MFTL_RedistributeStats st;
  MFTLRedistributeGetStats(&st);
  return st.n_objects_skipped;
}

/** @brief One timestep of the documented loop, up to and including step 3. */
static void advect_one_step(MFTL_Ensemble *ens, const double vel[3], double dt)
{
  /* `vel` rather than `v`: MFTL_FOREACH_OWNED_VERTEX names its loop index `v`. */
  const double dx = vel[0] * dt, dy = vel[1] * dt, dz = vel[2] * dt;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *obj = ens->local_objects[o];
    FOREACH_BODY_VERTEX(ens, 0, pos[0] += dx; pos[1] += dy; pos[2] += dz; MFTLApplyPeriodicity(&ens->domain, pos););
  }
  MFTLUpdateGhostPositions(ens);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
}

/** @brief Sorted list of the global ids this rank OWNS. Caller frees. */
static int *owned_triangle_gids(const MFTL_Object *obj, int *count)
{
  const int n = obj->tris.n_local;
  *count = n;
  int *ids = (int *) malloc((size_t) (n > 0 ? n : 1) * sizeof(int));
  for (int t = 0; t < n; ++t) ids[t] = obj->tris.rec[t].global_id;
  MFTLSortIds(ids, n);
  return ids;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 1. Wire format and MPI types
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The wire format is what the header claims, and it is compact.
 *
 * The sizes are pinned by _Static_assert in the header as well; this states the
 * COMPARISON, because the whole point of dropping the geometry was the ratio and
 * a silent regression to a fatter format would still compile.
 */
static int test_wire_format_is_compact(void)
{
  TEST_ASSERT_EQUAL_INT(40, (int) sizeof(MFTL_PackedVertex));
  TEST_ASSERT_EQUAL_INT(40, (int) sizeof(MFTL_PackedTriangleComplex));
  TEST_ASSERT_EQUAL_INT(32, (int) sizeof(MFTL_GhostUpdatePayload));

  /* A vertex carries 3 doubles and 4 ints with no padding at all; a triangle
   * complex is exactly ten ints. */
  TEST_ASSERT_EQUAL_INT(3 * (int) sizeof(double) + 4 * (int) sizeof(int), (int) sizeof(MFTL_PackedVertex));
  TEST_ASSERT_EQUAL_INT(10 * (int) sizeof(int), (int) sizeof(MFTL_PackedTriangleComplex));

  /* The MPI datatypes must agree with the C stride, or an array of them reads
   * shifted garbage. */
  MPI_Aint lb = 0, extent = 0;
  MPI_Type_get_extent(MFTL_MPI_PackedVertex, &lb, &extent);
  TEST_ASSERT_EQUAL_INT((int) sizeof(MFTL_PackedVertex), (int) extent);
  MPI_Type_get_extent(MFTL_MPI_PackedTriangleComplex, &lb, &extent);
  TEST_ASSERT_EQUAL_INT((int) sizeof(MFTL_PackedTriangleComplex), (int) extent);
  MPI_Type_get_extent(MFTL_MPI_GhostUpdatePayload, &lb, &extent);
  TEST_ASSERT_EQUAL_INT((int) sizeof(MFTL_GhostUpdatePayload), (int) extent);

  return 0;
}

/** @brief Registering twice is a no-op, so a library and its user may both call it. */
static int test_mpi_types_register_is_idempotent(void)
{
  const MPI_Datatype before = MFTL_MPI_PackedVertex;
  MFTLRegisterMpiTypes();
  TEST_ASSERT(MFTL_MPI_PackedVertex == before);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 2. The initial scatter
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Generation puts everything on rank 0; redistribution spreads it.
 *
 * This is the case that forces the GLOBAL communication path: rank 0's
 * destination set is every rank that holds part of the sphere, which at nproc > 27
 * exceeds the neighbourhood and at any nproc includes ranks that are not
 * neighbours of rank 0.
 */
static int test_scatter_from_rank0(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  TEST_ASSERT(ens != MFTL_NULL);

  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);

  const int global_t = obj->global_n_triangles;
  const int global_v = obj->global_n_vertices;

  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ownership_is_exclusive(ens, 0), MPI_COMM_WORLD));

  double totals[3];
  refresh_geometry(ens);
  global_totals(ens, totals);
  TEST_ASSERT_EQUAL_INT(global_t, (int) totals[0]);
  TEST_ASSERT_EQUAL_INT(global_v, (int) totals[1]);

  /* Every rank that holds part of a sphere spanning the whole box must hold
   * something -- otherwise the partitioning is not doing anything and the rest of
   * the suite would be testing a serial code path. */
  int nonempty = (obj->tris.n_local > 0) ? 1 : 0;
  int total_nonempty = 0;
  MPI_Allreduce(&nonempty, &total_nonempty, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT_EQUAL_INT(g_nproc, total_nonempty);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Redistributing an already-distributed mesh changes nothing.
 *
 * The kept-entity path is what makes this the CHEAP case, and it is also where a
 * de-duplication bug shows up first: a second pass re-receives every ghost the
 * partition already holds, so anything that fails to recognise an arrival as a
 * duplicate grows the pool on every call.
 */
static int test_redistribute_is_idempotent(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.05, -0.05, 0.1, 0.6, 3);

  MFTLRedistribute(ens);
  refresh_geometry(ens);

  const int n_v = obj->verts.n, n_vl = obj->verts.n_local;
  const int n_t = obj->tris.n, n_tl = obj->tris.n_local;
  int n_before = 0;
  int *before = owned_triangle_gids(obj, &n_before);

  MFTLRedistribute(ens);

  int n_after = 0;
  int *after = owned_triangle_gids(obj, &n_after);

  int same = (n_v == obj->verts.n) && (n_vl == obj->verts.n_local) && (n_t == obj->tris.n) && (n_tl == obj->tris.n_local) && (n_before == n_after);
  for (int i = 0; same && i < n_before; ++i) same = (before[i] == after[i]);

  free(before);
  free(after);

  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 3. The incremental twin repair
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The repaired twins are exactly the twins a full rebuild would produce.
 *
 * Redistribution never runs MFTLRebuildTwins: a twin between two kept triangles
 * survives as an index remap, and only half-edges that actually lost their
 * partner are re-paired. That is the single most delicate optimisation in the
 * module, so it is checked against the reference implementation directly.
 */
static int test_twin_repair_matches_full_rebuild(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.65, 3);

  MFTLRedistribute(ens);

  /* A second round after motion, so the repair set contains both "my twin left"
   * and "a new neighbour arrived" rather than only freshly received triangles. */
  FOREACH_BODY_VERTEX(ens, 0, pos[0] += 0.11; pos[1] += 0.07; pos[2] -= 0.05; MFTLApplyPeriodicity(&ens->domain, pos););
  MFTLUpdateGhostPositions(ens);
  refresh_geometry(ens);
  MFTLRedistribute(ens);

  const int n_he = 3 * obj->tris.n;
  int *repaired = (int *) malloc((size_t) (n_he > 0 ? n_he : 1) * sizeof(int));
  for (int h = 0; h < n_he; ++h) repaired[h] = obj->tris.he[h].twin_idx;

  MFTLRebuildTwins(obj);

  int same = 1;
  for (int h = 0; h < n_he; ++h)
    if (repaired[h] != obj->tris.he[h].twin_idx) same = 0;
  free(repaired);

  MFTLObjectBuildDerived(obj);  // MFTLRebuildTwins invalidated them

  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief The derived arrays a redistribution emits are the ones a rebuild builds.
 *
 * rebuild_object() no longer closes with MFTLObjectBuildDerived(). It writes
 * `tris.verts` and `tris.neighbors` inside its relink loop, its arrival loop and
 * repair_twins(), because every input is already in a register there -- so the
 * rotation `verts[3t+k] == he[3t+(k+2)%3].vertex_idx` and the arithmetic
 * `neighbors[3t+k] == he[3t+k].twin_idx / 3` are now spelled in TWO places.
 *
 * inv_derived_matches_topology() already catches a disagreement on every
 * redistributed mesh, but only as one clause of inv_mesh_is_sound(). This names
 * it: it takes the arrays redistribution produced, re-derives them with the
 * function that owns the convention, and compares them element for element. A
 * drift between the two sites fails HERE, where the comment says what broke.
 */
static int test_derived_arrays_match_a_full_rebuild(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.65, 3);

  MFTLRedistribute(ens);

  /* A second round after motion, so the mesh contains all four kinds of slot the
   * emission has to cover: kept owned, kept ghost, arrived owned, arrived ghost
   * -- plus a repair set with genuinely re-paired twins. */
  FOREACH_BODY_VERTEX(ens, 0, pos[0] += 0.11; pos[1] += 0.07; pos[2] -= 0.05; MFTLApplyPeriodicity(&ens->domain, pos););
  MFTLUpdateGhostPositions(ens);
  refresh_geometry(ens);
  MFTLRedistribute(ens);

  const int n = 3 * obj->tris.n;
  int *emitted = (int *) malloc((size_t) (n > 0 ? 2 * n : 2) * sizeof(int));
  for (int i = 0; i < n; ++i) emitted[i] = obj->tris.verts[i];
  for (int i = 0; i < n; ++i) emitted[n + i] = obj->tris.neighbors[i];

  MFTLObjectBuildDerived(obj);

  int same = 1;
  for (int i = 0; i < n; ++i)
    if (emitted[i] != obj->tris.verts[i] || emitted[n + i] != obj->tris.neighbors[i]) same = 0;
  free(emitted);

  /* Non-vacuity: an all-ghost or empty pool would compare equal trivially. */
  const int worth_checking = (obj->tris.n_local > 0 && obj->tris.n > obj->tris.n_local);

  TEST_ASSERT(agreed(same, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ens->domain.nproc < 2 || worth_checking, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 4. The per-timestep loop
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Rigid advection across rank and periodic boundaries conserves everything.
 *
 * A rigid translation conserves surface area exactly and redistribution conserves
 * entities exactly, so any drift in either means an entity was lost, duplicated,
 * or measured across the periodic seam instead of through it. The velocity is
 * chosen irrational-ish per axis so that vertices land ON cell faces at
 * unpredictable times -- the tie-breaking rule in MFTLGetOwnerProc is what makes
 * two ranks agree there, and a run that never hits a face would not test it.
 */
static int test_advection_conserves_entities(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, -0.3, 0.2, 0.1, 0.45, 3);

  MFTLRedistribute(ens);
  refresh_geometry(ens);

  double reference[3];
  global_totals(ens, reference);

  const double vel[3] = {0.25, 0.13, -0.07};
  const double dt = 1.0;

  int ok = 1;
  for (int step = 0; step < 12; ++step)
  {
    FOREACH_BODY_VERTEX(ens, 0, pos[0] += vel[0] * dt; pos[1] += vel[1] * dt; pos[2] += vel[2] * dt; MFTLApplyPeriodicity(&ens->domain, pos););

    if (MFTLUpdateGhostPositions(ens) != MFTL_SUCCESS) ok = 0;
    refresh_geometry(ens);

    double totals[3];
    global_totals(ens, totals);
    if ((int) totals[0] != (int) reference[0]) ok = 0;
    if ((int) totals[1] != (int) reference[1]) ok = 0;
    if (fabs(totals[2] - reference[2]) > 1e-9 * reference[2]) ok = 0;

    if (!ensemble_is_sound(ens)) ok = 0;
    if (!ownership_is_exclusive(ens, 0)) ok = 0;

    if (MFTLRedistribute(ens) != MFTL_SUCCESS) ok = 0;
  }

  refresh_geometry(ens);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Ghost positions really are the owner's positions after a sync.
 *
 * Every owned vertex is moved to a position that is a pure function of its global
 * id, so the check needs no cross-rank comparison at all: a ghost is correct iff
 * its position is that function of ITS id. A sync that dropped a request, or that
 * mismatched replies against requests, fails immediately.
 */
static void positions_from_gid(int gid, double out[3])
{
  out[0] = 0.9 * sin(0.7 * (double) gid);
  out[1] = 0.9 * sin(1.3 * (double) gid + 0.4);
  out[2] = 0.9 * sin(2.1 * (double) gid + 1.1);
}

static int test_ghost_positions_are_synced(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.6, 3);

  MFTLRedistribute(ens);

  FOREACH_BODY_VERTEX(ens, 0, positions_from_gid(rec->global_id, pos););

  const int rc = MFTLUpdateGhostPositions(ens);

  int ok = (rc == MFTL_SUCCESS);
  int n_ghosts = 0;
  for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
  {
    double expect[3];
    positions_from_gid(obj->verts.rec[v].global_id, expect);
    for (int k = 0; k < 3; ++k)
      if (fabs(obj->verts.pos[3 * v + k] - expect[k]) > 1e-15) ok = 0;
    n_ghosts++;
  }

  /* At more than one rank the halo is not allowed to be empty everywhere, or the
   * test would pass vacuously. */
  int total_ghosts = 0;
  MPI_Allreduce(&n_ghosts, &total_ghosts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));
  if (g_nproc > 1) TEST_ASSERT(total_ghosts > 0);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 5. What the halo is FOR
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Distributed vertex normals equal the ones computed on the whole mesh.
 *
 * The end-to-end statement of vertex completeness. MFTLComputeVertexNormals
 * accumulates over ghost triangles as well and communicates nothing, so an owned
 * vertex gets the exact answer iff EVERY triangle incident to it is present
 * locally. A halo that is one triangle short produces a normal that is merely
 * slightly wrong -- no crash, no assertion, and no other check notices.
 *
 * Compared with the chord |a - b|, never with acos(a.b): the arc cosine has
 * infinite derivative at 1, so it reports ~1e-8 rad of "error" for two vectors
 * that differ by one ulp (see tests/test_vertex_normals.c).
 */
static int test_vertex_normals_match_serial_reference(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.1, -0.2, 0.05, 0.55, 3);

  const int n_global_v = obj->global_n_vertices;
  double *reference = (double *) calloc((size_t) (n_global_v > 0 ? 3 * n_global_v : 3), sizeof(double));

  /* Rank 0 still holds the complete mesh, so its answer is the ground truth. */
  if (g_rank == 0)
  {
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const int g = obj->verts.rec[v].global_id;
      for (int k = 0; k < 3; ++k) reference[3 * g + k] = obj->verts.normal[3 * v + k];
    }
  }
  MPI_Bcast(reference, 3 * n_global_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MFTLRedistribute(ens);
  refresh_geometry(ens);
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);

  int ok = 1;
  for (int v = 0; v < obj->verts.n_local; ++v)
  {
    const int g = obj->verts.rec[v].global_id;
    double d = 0.0;
    for (int k = 0; k < 3; ++k)
    {
      const double e = obj->verts.normal[3 * v + k] - reference[3 * g + k];
      d += e * e;
    }
    if (sqrt(d) > 1e-12) ok = 0;
  }

  free(reference);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 6. Several bodies in one communication round
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief The multi-body path keeps the bodies apart and conserves each of them.
 *
 * Every body is interleaved into the same pair of messages and separated again
 * from the per-body count payload. The failure mode being tested for is not a
 * crash: entities merged into the wrong body still form a valid-looking mesh, so
 * the check is per-body conservation and per-body soundness.
 */
/**
 * @brief Distance between vertex @p v's normal and the serial reference for it.
 *
 * The CHORD |a - b|, never acos(a.b): the arc cosine has infinite derivative at
 * 1, so two unit vectors agreeing to one ulp measure ~1.5e-8 rad apart and look
 * like a bug. tests/test_vertex_normals.c records the false failure that
 * produced, and CLAUDE.md makes it a rule.
 */
static double normal_chord(const MFTL_Object *obj, const double *reference, int v)
{
  const int g = obj->verts.rec[v].global_id;
  double d = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    const double e = obj->verts.normal[3 * v + k] - reference[3 * g + k];
    d += e * e;
  }
  return sqrt(d);
}

/**
 * @brief After MFTLEnsembleUpdateVertexNormals every vertex is exact -- GHOSTS TOO.
 *
 * The sibling test above checks the owned prefix, which MFTLComputeVertexNormals
 * gets right on its own. This one checks the half it cannot: a ghost vertex's
 * far-side faces live on another rank, so the local accumulation leaves it
 * holding a partial sum -- correctly normalised, and quietly wrong. Only the
 * exchange fixes that.
 *
 * The test is written so it cannot pass vacuously. It asserts first that this
 * partition actually HAS ghost vertices (at nproc > 1), and second that the
 * un-exchanged normals really are wrong -- otherwise a no-op implementation of
 * the exchange would pass and the trap would still be open.
 */
static int test_ghost_vertex_normals_are_exact(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, -0.15, 0.25, -0.05, 0.6, 3);

  const int n_global_v = obj->global_n_vertices;
  double *reference = (double *) calloc((size_t) (n_global_v > 0 ? 3 * n_global_v : 3), sizeof(double));

  /* Rank 0 still holds the complete mesh, so its answer is the ground truth. */
  if (g_rank == 0)
  {
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const int g = obj->verts.rec[v].global_id;
      for (int k = 0; k < 3; ++k) reference[3 * g + k] = obj->verts.normal[3 * v + k];
    }
  }
  MPI_Bcast(reference, 3 * n_global_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MFTLRedistribute(ens);
  refresh_geometry(ens);

  const int n_ghosts = obj->verts.n - obj->verts.n_local;

  /* 1. Without the exchange, the ghosts hold partial sums. If this rank has any
   *    ghost at all, at least one of them must DISAGREE -- if they all agreed,
   *    the test below would be proving nothing. */
  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);
  int local_ghost_wrong = 0;
  for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
  {
    if (normal_chord(obj, reference, v) > 1e-12) local_ghost_wrong = 1;
  }

  int any_ghosts = (n_ghosts > 0) ? 1 : 0;
  int total_ghosts = 0, total_wrong = 0;
  MPI_Allreduce(&any_ghosts, &total_ghosts, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_ghost_wrong, &total_wrong, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (g_nproc > 1)
  {
    TEST_ASSERT(total_ghosts > 0);  // the partition must actually have a halo
    TEST_ASSERT(total_wrong > 0);  // ... and the halo must actually be wrong
  }

  /* 2. With the exchange, every vertex in the pool is exact. */
  TEST_ASSERT(agreed(MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_AREA) == MFTL_SUCCESS, MPI_COMM_WORLD));

  int ok = 1;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    if (normal_chord(obj, reference, v) > 1e-12) ok = 0;
  }

  free(reference);
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Position and normal exchange preserve derived connectivity.
 *
 * Neither operation changes triangle topology, so both must leave the compact
 * vertex/neighbor arrays current.
 */
static int test_ghost_syncs_leave_topology_current(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.6, 2);

  MFTLRedistribute(ens);
  refresh_geometry(ens);

  TEST_ASSERT(agreed(MFTLObjectHasTopology(obj), MPI_COMM_WORLD));

  TEST_ASSERT(agreed(MFTLEnsembleUpdateVertexNormals(ens, MFTL_NORMAL_WEIGHT_AREA) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLObjectHasTopology(obj), MPI_COMM_WORLD));

  TEST_ASSERT(agreed(MFTLUpdateGhostPositions(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLObjectHasTopology(obj), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

static int test_multibody_round(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 4);

  const double c[5][3] = {{-0.6, -0.6, -0.6}, {0.6, 0.6, 0.6}, {-0.6, 0.6, 0.0}, {0.6, -0.6, 0.2}, {0.0, 0.0, 0.0}};
  const int refine[5] = {0, 1, 2, 1, 2};

  int expect_t[5], expect_v[5];
  for (int o = 0; o < 5; ++o)
  {
    MFTL_Object *obj = add_sphere(ens, 100 + o, c[o][0], c[o][1], c[o][2], 0.25, refine[o]);
    expect_t[o] = obj->global_n_triangles;
    expect_v[o] = obj->global_n_vertices;
  }

  TEST_ASSERT_EQUAL_INT(5, ens->n_local_objects);
  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  /* By BODY ID from here on. The list is this rank's, so slot `o` is body `o`
   * only while the rank still holds all five -- which is exactly what a
   * redistribution stops being true. */
  int ok = 1;
  for (int b = 0; b < 5; ++b)
  {
    const MFTL_Object *const obj = body(ens, b);
    int local[2] = {obj ? obj->tris.n_local : 0, obj ? obj->verts.n_local : 0};
    int total[2] = {0, 0};
    MPI_Allreduce(local, total, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total[0] != expect_t[b] || total[1] != expect_v[b]) ok = 0;
    if (!ownership_is_exclusive(ens, b)) ok = 0;

    /* The body id must survive the round trip -- it is the identity the body
     * travels under, so a redistribution that lost or permuted it would merge
     * entities into the wrong body. MFTLEnsembleAddObject() hands the ids out
     * densely in add order, so it is `b` whatever add_sphere() was passed. */
    if (obj != MFTL_NULL && obj->global_id != b) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY: the whole point of five small bodies on four ranks is that not
   * every rank holds every body. If it did, nothing above would be testing the
   * sparse list at all. */
  {
    int somebody_is_missing = (ens->n_local_objects < 5);
    MPI_Allreduce(MPI_IN_PLACE, &somebody_is_missing, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
    int nproc = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    TEST_ASSERT(nproc == 1 || somebody_is_missing);
  }

  /* A second round after motion: now every body has kept entities AND arrivals,
   * which is the state the run reconstruction actually has to get right. */
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *lb = ens->local_objects[o];
    MFTL_FOREACH_OWNED_VERTEX(lb, pos[0] += 0.31; pos[1] -= 0.23; MFTLApplyPeriodicity(&ens->domain, pos););
  }
  TEST_ASSERT(agreed(MFTLUpdateGhostPositions(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  refresh_geometry(ens);
  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  ok = 1;
  for (int b = 0; b < 5; ++b)
  {
    const MFTL_Object *const obj = body(ens, b);
    int local[2] = {obj ? obj->tris.n_local : 0, obj ? obj->verts.n_local : 0};
    int total[2] = {0, 0};
    MPI_Allreduce(local, total, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (total[0] != expect_t[b] || total[1] != expect_v[b]) ok = 0;
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * 7. Edge cases
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief A body that moves out of a rank entirely leaves nothing behind.
 *
 * The degenerate end of the kept path: the kept set is EMPTY, so the compaction
 * has to produce a pool of size zero rather than the previous contents, and the
 * receiving rank has to build a partition from arrivals alone with nothing to
 * de-duplicate against.
 */
static int test_body_can_leave_a_rank_entirely(void)
{
  if (g_nproc < 2) return 0;  // nothing to move to

  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);

  /* Centred in cell (0,0,0) and small enough to stay inside it on every axis, so
   * that before the move exactly one rank holds the body and nobody holds a
   * ghost of it. */
  double cell[3];
  for (int d = 0; d < 3; ++d) cell[d] = 2.0 / (double) ens->domain.cuts[d];
  const double smallest = MFTL_MIN(cell[0], MFTL_MIN(cell[1], cell[2]));

  const double x0 = -1.0 + 0.5 * cell[0];
  const double y0 = -1.0 + 0.5 * cell[1];
  const double z0 = -1.0 + 0.5 * cell[2];
  add_sphere(ens, 7, x0, y0, z0, 0.3 * smallest, 2);

  MFTLRedistribute(ens);
  refresh_geometry(ens);

  /* "Holds nothing of it" is now a STRONGER statement than it was: the rank does
   * not merely have an empty pool, it has no object for the body at all. That is
   * the property this whole test exists to pin, so it is asserted as the absence
   * of the object rather than as a zero count. */
  const int home = ens->domain.rank_map[0];
  int ok = 1;
  {
    const MFTL_Object *const b = body(ens, 0);
    if (g_rank == home)
    {
      if (b == MFTL_NULL) ok = 0;
      else if (b->tris.n_local != b->global_n_triangles) ok = 0;
      else if (b->tris.n != b->tris.n_local) ok = 0;  // no halo: nothing else holds any of it
    }
    else if (b != MFTL_NULL)
    {
      ok = 0;
    }
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* One cell to the right, in one step. */
  FOREACH_BODY_VERTEX(ens, 0, pos[0] += cell[0]; MFTLApplyPeriodicity(&ens->domain, pos););
  refresh_geometry(ens);

  double probe[3] = {x0 + cell[0], y0, z0};
  MFTLApplyPeriodicity(&ens->domain, probe);
  const int dest = MFTLGetOwnerProc(&ens->domain, probe);

  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  ok = 1;
  {
    const MFTL_Object *const b = body(ens, 0);
    if (g_rank == dest)
    {
      /* Created here on arrival: this rank held no object for the body a moment
       * ago and holds all of it now. */
      if (b == MFTL_NULL) ok = 0;
      else if (b->tris.n_local != b->global_n_triangles) ok = 0;
    }
    else if (b != MFTL_NULL)
    {
      /* Destroyed here on departure -- including on `home`, which held the whole
       * body before the move and must now hold no object at all. */
      ok = 0;
    }
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY: the move has to actually cross a rank boundary, or the test
   * asserts nothing about create-on-arrival or destroy-on-departure. */
  TEST_ASSERT(agreed(dest != home, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A body straddling a periodic seam is partitioned correctly.
 *
 * The seam is where an "obvious" implementation goes wrong twice: a centroid
 * computed without the minimum image lands in the middle of the box rather than
 * at the seam, and a position that is not wrapped clamps to the boundary cell
 * instead of wrapping into it.
 */
static int test_body_straddling_a_periodic_seam(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, -1.0, 0.0, 0.0, 0.4, 3);  // centred exactly on the seam

  const int global_t = obj->global_n_triangles;

  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ownership_is_exclusive(ens, 0), MPI_COMM_WORLD));

  refresh_geometry(ens);
  double totals[3];
  global_totals(ens, totals);
  TEST_ASSERT_EQUAL_INT(global_t, (int) totals[0]);

  /* The sphere is centred on the seam, so both sides of the box must hold part
   * of it -- otherwise the wrap silently pushed everything to one side. */
  int has_low = 0, has_high = 0;
  {
    const MFTL_Object *const b = body(ens, 0);
    for (int v = 0; b != MFTL_NULL && v < b->verts.n_local; ++v)
    {
      if (b->verts.pos[3 * v] < -0.5) has_low = 1;
      if (b->verts.pos[3 * v] > 0.5) has_high = 1;
    }
  }
  int flags[2] = {has_low, has_high};
  int any[2] = {0, 0};
  MPI_Allreduce(flags, any, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  TEST_ASSERT(any[0] == 1 && any[1] == 1);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief A non-periodic domain: no wrap, and the outer cells are walls. */
static int test_non_periodic_domain(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE, 1);
  MFTL_Object *obj = add_sphere(ens, 0, 0.0, 0.0, 0.0, 0.7, 3);

  const int global_t = obj->global_n_triangles;

  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ownership_is_exclusive(ens, 0), MPI_COMM_WORLD));

  refresh_geometry(ens);
  double totals[3];
  global_totals(ens, totals);
  TEST_ASSERT_EQUAL_INT(global_t, (int) totals[0]);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief An object with no entities anywhere survives a redistribution.
 *
 * It is not a contrived case: MFTLGenerateSphere leaves every rank but 0 exactly
 * like this, so the empty path runs on nproc-1 ranks of every single call.
 */
/* ─────────────────────────────────────────────────────────────────────────────
 * N. Leaving unchanged bodies alone
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief A skipped body is INDISTINGUISHABLE from a rebuilt one, slot for slot.
 *
 * The acceptance test for the whole optimisation. Two ensembles are built from
 * identical inputs and advected in lockstep through the documented per-timestep
 * loop; one is redistributed with MFTLRedistributeSetSkipUnchangedBodies on and
 * the other with it off, and after every step both partition digests must be
 * equal on both ensembles.
 *
 * Why BOTH digests. The set digest catches a wrong partition: a missing ghost, a
 * duplicated owner, a stale global id, a position that did not travel. The order
 * digest catches a right partition reached by a different route -- entities in
 * different slots. That is not hypothetical and it is not harmless-by-inspection:
 * rebuild_object is not idempotent, so an early version of the skip passed the
 * set digest and failed the order digest from the FIRST step on any body large
 * enough to span three ranks. The settling rule in redistribute.c section 6b is
 * what closes it, and this test is what would notice if it were removed.
 *
 * NON-VACUITY. A run in which nothing was ever skipped compares a code path
 * against itself and would pass forever, so the number of skipped bodies is
 * asserted to be positive. It cannot be at np = 1, where redistribution
 * short-circuits before any of this, so the assertion is made only above that.
 */
static int test_skipping_reproduces_the_full_rebuild(void)
{
  /* Small bodies on a coarse grid: most of them are nowhere near a cut plane in
   * any given step, which is the regime the skip exists for. Two of them start
   * deliberately ON a plane so the run also contains bodies that never qualify. */
  const int n_bodies = 12;
  MFTL_Ensemble *ens_on = make_ensemble(MFTL_TRUE, n_bodies);
  MFTL_Ensemble *ens_off = make_ensemble(MFTL_TRUE, n_bodies);

  for (int i = 0; i < n_bodies; ++i)
  {
    const double cx = -0.8 + 0.13 * (double) i;
    const double cy = -0.6 + 0.09 * (double) ((i * 5) % 11);
    const double cz = 0.7 - 0.11 * (double) ((i * 3) % 7);
    add_sphere(ens_on, i, cx, cy, cz, 0.09, 2);
    add_sphere(ens_off, i, cx, cy, cz, 0.09, 2);
  }

  MFTLRedistributeSetSkipUnchangedBodies(MFTL_TRUE);
  MFTLRedistribute(ens_on);
  MFTLRedistributeSetSkipUnchangedBodies(MFTL_FALSE);
  MFTLRedistribute(ens_off);

  /* Irrational-ish per axis, so vertices land on cell faces at unpredictable
   * times rather than never or all at once. */
  const double velocity[3] = {0.031, 0.017, 0.023};

  MFTLRedistributeStatsReset();
  long skipped = 0;
  int ok = 1;

  for (int step = 0; step < 24 && ok; ++step)
  {
    advect_one_step(ens_on, velocity, 1.0);
    advect_one_step(ens_off, velocity, 1.0);

    const long before = skipped_so_far();
    MFTLRedistributeSetSkipUnchangedBodies(MFTL_TRUE);
    MFTLRedistribute(ens_on);
    skipped += skipped_so_far() - before;

    MFTLRedistributeSetSkipUnchangedBodies(MFTL_FALSE);
    MFTLRedistribute(ens_off);

    uint64_t d_on[2], d_off[2];
    partition_digest(ens_on, d_on);
    partition_digest(ens_off, d_off);

    if (d_on[0] != d_off[0] || d_on[1] != d_off[1]) ok = 0;
    if (!ensemble_is_sound(ens_on)) ok = 0;
  }

  MFTLRedistributeSetSkipUnchangedBodies(MFTL_TRUE);  // back to the library default

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY, as a SUM over ranks rather than as "every rank skipped
   * something". It used to be the latter and passed for the wrong reason: every
   * rank held an object for every body, so the ten bodies a rank had nothing of
   * were counted as skipped every step. A rank now holds only the bodies it has
   * entities of -- one or two of twelve at eight ranks -- and can legitimately
   * have all of them cross a plane in a given step. The sum is what the claim
   * "skipping happens, and the meshes still agree" actually rests on. */
  long total_skipped = 0;
  MPI_Allreduce(&skipped, &total_skipped, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_nproc > 1) TEST_ASSERT(total_skipped > 0);

  MFTLEnsembleDestroy(&ens_on);
  MFTLEnsembleDestroy(&ens_off);
  return 0;
}

/**
 * @brief A stamped body is rebuilt even when nothing about its ownership moved.
 *
 * The half of the precondition an owner-change test cannot see. The spanning
 * remesher renumbers a body's OWNED entities and leaves its ghost triangle ids
 * on the previous numbering, because redistribution drops and re-receives every
 * ghost triangle and that is what repairs them; a skip removes the repair, on a
 * converged body, every step. MFTLObjectMarkPartitionDirty is what says so, and
 * this asserts that saying so is enough.
 *
 * It is measured through the counter rather than through the symptom, because
 * the symptom is silent: the stale ids sit in the pool until something reads
 * them, which in a production loop is nothing at all.
 */
static int test_a_stamped_body_is_never_skipped(void)
{
  const int n_bodies = 8;
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, n_bodies);
  for (int i = 0; i < n_bodies; ++i) add_sphere(ens, i, -0.7 + 0.19 * (double) i, 0.1 * (double) (i % 3), -0.2 + 0.07 * (double) (i % 5), 0.09, 2);

  MFTLRedistributeSetSkipUnchangedBodies(MFTL_TRUE);
  MFTLRedistribute(ens);

  const double velocity[3] = {0.019, 0.011, 0.013};

  /* Two steps to leave the initial scatter behind and let the bodies settle, so
   * that the baseline step below genuinely has something to skip. */
  for (int step = 0; step < 3; ++step)
  {
    advect_one_step(ens, velocity, 1.0);
    MFTLRedistribute(ens);
  }

  advect_one_step(ens, velocity, 1.0);
  long before = skipped_so_far();
  MFTLRedistribute(ens);
  const long baseline = skipped_so_far() - before;

  /* Same loop, same motion, but every body now says the partition no longer
   * describes it. Nothing may be skipped. */
  advect_one_step(ens, velocity, 1.0);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLObjectMarkPartitionDirty(ens->local_objects[o]);
  before = skipped_so_far();
  MFTLRedistribute(ens);
  const long stamped = skipped_so_far() - before;

  TEST_ASSERT(agreed(stamped == 0, MPI_COMM_WORLD));

  /* Non-vacuity as a SUM, for the same reason as in
   * test_skipping_reproduces_the_full_rebuild: a rank holds only the bodies it
   * has entities of, so at eight ranks it can hold one or two and legitimately
   * have all of them cross a plane in the baseline step. "Somebody skipped
   * something" is what makes `stamped == 0` a comparison rather than a tautology;
   * "every rank did" used to hold only because every rank carried an empty object
   * for every body it had nothing of. */
  long total_baseline = 0;
  MPI_Allreduce(&baseline, &total_baseline, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  if (g_nproc > 1) TEST_ASSERT(total_baseline > 0);
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Pool capacity after the scatter
 * ─────────────────────────────────────────────────────────────────────────── */

/**
 * @brief Bodies wasting more than @p factor x their live count, on this rank.
 *
 * The bound is `factor`, not 2, and the difference is the whole anti-thrash
 * argument: a shrink lands capacity in [n, 2n), but a body already between 2n
 * and `factor * n` is deliberately LEFT ALONE, because firing there is what
 * makes a jittering count reallocate every timestep. So `factor * n` is what an
 * armed pass guarantees, and asserting 2n instead would fail on bodies the
 * policy is correctly declining to touch.
 *
 * @param ens    The ensemble.
 * @param factor Waste factor to test against, i.e. the armed slack.
 * @return Count of bodies over the bound, this rank only.
 */
static int bodies_over_the_bound(const MFTL_Ensemble *ens, double factor)
{
  int bad = 0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    if (obj->verts.n == 0 && obj->tris.n == 0) continue;

    const double v_bound = MFTL_MAX((double) MFTL_INIT_CAPACITY, factor * (double) obj->verts.n);
    const double t_bound = MFTL_MAX((double) MFTL_INIT_CAPACITY, factor * (double) obj->tris.n);
    if ((double) obj->verts.capacity > v_bound || (double) obj->tris.capacity > t_bound) bad++;
  }
  return bad;
}

/** @brief Bytes both pool blocks of every body occupy on this rank. */
static long capacity_bytes(const MFTL_Ensemble *ens)
{
  long bytes = 0;
  for (int o = 0; o < ens->n_local_objects; ++o) bytes += (long) MFTLObjectPoolBytes(ens->local_objects[o]);
  return bytes;
}

/**
 * @brief The scatter must not leave rank 0 holding every body at full capacity.
 *
 * MFTLGenerateSphere() builds on rank 0, pools are grow-only, and redistribution
 * lowers `n` and never `capacity` -- so without MFTLRedistributeSetShrinkSlack()
 * rank 0 keeps a full-mesh block for every body it retains a sliver of, for the
 * whole run. That is deferred issue 20 in CLAUDE.md, and this is its regression
 * test: it FAILS with the slack at 0.
 */
static int test_generation_capacity_does_not_stay_resident_on_rank_0(void)
{
  const int n_bodies = 16;
  MFTL_Ensemble *ens = make_ensemble(MFTL_FALSE, n_bodies + 1);

  /* Spread over the box so that at np > 1 rank 0 ends up holding slivers of
   * several bodies rather than all of one or none of any. Refine 3 is 642
   * vertices and 1280 triangles per body -- large enough that a sliver is far
   * under the 2x bound, small enough to stay cheap at np = 8. */
  for (int i = 0; i < n_bodies; ++i)
  {
    const double cx = -0.75 + 0.10 * (double) i;
    const double cy = -0.55 + 0.07 * (double) ((i * 5) % 11);
    const double cz = 0.60 - 0.09 * (double) ((i * 3) % 7);
    add_sphere(ens, i, cx, cy, cz, 0.14, 3);
  }

  /* One body placed so that the x = 0 cut shaves a THIN CAP off it. Rank 0 is
   * Cartesian cell (0,0,0) -- the low corner on every axis -- so the cap lands
   * there, on the rank that generated the whole sphere and therefore holds a
   * full-mesh block. That is issue 20 in one body: a few dozen triangles in
   * storage sized for 1280.
   *
   * The spread bodies above cannot produce it. A small body either lands whole
   * on one rank (capacity == n, nothing wasted) or splits near its middle
   * (~2x, which the slack-4 gate correctly declines to touch). Only a cut near
   * a body's EDGE stranded enough to be worth recovering, and without this body
   * the test would pass while covering nothing. */
  add_sphere(ens, n_bodies, 0.15, -0.5, -0.5, 0.18, 3);

  /* NON-VACUITY, measured before anything is redistributed. On rank 0 every body
   * is at capacity == n, which satisfies the bound -- so the interesting count
   * is what a rank holds AFTER the scatter with the shrink off. Without this the
   * test would pass at np = 1 while covering nothing: there, rank 0 legitimately
   * keeps every body whole and the bound is met trivially. */
  MFTLRedistributeSetShrinkSlack(0.0);
  MFTLRedistribute(ens);
  refresh_geometry(ens);  // redistribution leaves tris.center undefined by design
  MFTLRedistribute(ens);  // the settling rebuild, so nothing below is mid-transition
  refresh_geometry(ens);

  int local_bad = bodies_over_the_bound(ens, MFTL_SHRINK_SLACK_DEFAULT);
  int bad_without_shrink = 0;
  MPI_Allreduce(&local_bad, &bad_without_shrink, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  long local_bytes = capacity_bytes(ens);
  long max_before = 0, sum_before = 0;
  MPI_Allreduce(&local_bytes, &max_before, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_bytes, &sum_before, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  /* Above np = 1 the scatter MUST have stranded capacity somewhere, or this
   * test is measuring nothing and the bound below is vacuous. */
  if (g_nproc > 1) TEST_ASSERT(agreed(bad_without_shrink > 0, MPI_COMM_WORLD));

  /* Now arm it and redistribute again. Nothing about the mesh changed, so every
   * shrink that fires is pure recovery. */
  MFTLRedistributeSetShrinkSlack(MFTL_SHRINK_SLACK_DEFAULT);
  MFTLRedistribute(ens);
  MFTLRedistributeSetShrinkSlack(0.0);  // back to the library default

  /* PRIMARY: no body is left wasting more than the armed slack, with the
   * MFTL_INIT_CAPACITY floor. Per body rather than in aggregate, which is
   * sharper -- one stranded full-mesh block cannot hide behind fifteen small
   * ones the way a total would let it. */
  TEST_ASSERT(agreed(bodies_over_the_bound(ens, MFTL_SHRINK_SLACK_DEFAULT) == 0, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));

  /* SECONDARY: capacity was actually RECOVERED, not merely redistributed.
   *
   * Deliberately not "rank 0 is no longer an outlier against the mean". Sixteen
   * small bodies plus one sliver do not make rank 0 an outlier in the first
   * place -- measured max/mean is about 1.2 here -- so that assertion would pass
   * without testing anything, which is the failure mode this file's non-vacuity
   * rule exists to prevent. The outlier claim needs issue 20's real shape,
   * hundreds of bodies generated on one rank, and it is priced in RSS rather
   * than in capacity bytes by examples/parallel_advection_scaling.c. What is
   * checkable here is that the total fell, and it does: 7.87 -> 7.61 MB at
   * np = 2, 9.22 -> 8.95 MB at np = 4 and 8. */
  local_bytes = capacity_bytes(ens);
  long max_after = 0, sum_after = 0;
  MPI_Allreduce(&local_bytes, &max_after, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(&local_bytes, &sum_after, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  if (g_nproc > 1) TEST_ASSERT(agreed(sum_after < sum_before, MPI_COMM_WORLD));

  if (g_rank == 0)
  {
    printf("      capacity bytes: max %ld -> %ld, total %ld -> %ld over %d ranks\n", max_before, max_after, sum_before, sum_after, g_nproc);
  }

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Shrinking is a memory policy: the mesh must come out bit-identical.
 *
 * The same protocol test_skipping_reproduces_the_full_rebuild() uses, and for
 * the same reason -- both digest halves, every step. That test cannot stand in
 * for this one: its digest folds entity content and slot order and deliberately
 * says nothing about capacity, so it would pass while a shrink corrupted a pool.
 */
static int test_shrinking_does_not_change_the_mesh(void)
{
  const int n_bodies = 8;
  MFTL_Ensemble *ens_on = make_ensemble(MFTL_TRUE, n_bodies);
  MFTL_Ensemble *ens_off = make_ensemble(MFTL_TRUE, n_bodies);

  for (int i = 0; i < n_bodies; ++i)
  {
    const double cx = -0.7 + 0.19 * (double) i;
    const double cy = -0.5 + 0.11 * (double) ((i * 5) % 11);
    const double cz = 0.6 - 0.13 * (double) ((i * 3) % 7);
    add_sphere(ens_on, i, cx, cy, cz, 0.11, 2);
    add_sphere(ens_off, i, cx, cy, cz, 0.11, 2);
  }

  MFTLRedistributeSetShrinkSlack(MFTL_SHRINK_SLACK_DEFAULT);
  MFTLRedistribute(ens_on);
  MFTLRedistributeSetShrinkSlack(0.0);
  MFTLRedistribute(ens_off);

  const double velocity[3] = {0.031, 0.017, 0.023};
  int ok = 1;

  for (int step = 0; step < 20 && ok; ++step)
  {
    advect_one_step(ens_on, velocity, 1.0);
    advect_one_step(ens_off, velocity, 1.0);

    MFTLRedistributeSetShrinkSlack(MFTL_SHRINK_SLACK_DEFAULT);
    MFTLRedistribute(ens_on);
    MFTLRedistributeSetShrinkSlack(0.0);
    MFTLRedistribute(ens_off);

    uint64_t d_on[2], d_off[2];
    partition_digest(ens_on, d_on);
    partition_digest(ens_off, d_off);

    if (d_on[0] != d_off[0] || d_on[1] != d_off[1]) ok = 0;
    if (!ensemble_is_sound(ens_on)) ok = 0;
  }

  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens_on);
  MFTLEnsembleDestroy(&ens_off);
  return 0;
}

/**
 * @brief A body with no entities anywhere ends up held by nobody, and that is fine.
 *
 * The invariant is "a rank has an object for a body exactly when it has entities
 * of it", so a body that is empty on every rank has no local representation
 * anywhere -- its id stays allocated in the space, and nothing else of it exists.
 * The old behaviour was an empty MFTL_Object on every rank, which is the
 * per-body footprint this design removes.
 *
 * It also pins the pointer contract at its sharpest: `empty` is dangling the
 * moment the redistribution returns, so the test may not look at it. Bodies are
 * re-resolved by id, and this one resolves to nothing.
 */
static int test_empty_object_is_a_no_op(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 2);

  MFTLEnsembleAddObject(ens, MFTLObjectCreate(0));  // body 0: never given any entities
  add_sphere(ens, 1, 0.0, 0.0, 0.0, 0.4, 2);        // body 1: a real mesh

  TEST_ASSERT_EQUAL_INT(2, ens->n_local_objects);
  TEST_ASSERT_EQUAL_INT(2, ens->n_bodies_global);

  TEST_ASSERT(agreed(MFTLRedistribute(ens) == MFTL_SUCCESS, MPI_COMM_WORLD));

  /* Body 0 is gone from every rank; body 1 is wherever its entities went. The id
   * space is untouched -- ids are retired with the ensemble, not with a
   * partition. */
  TEST_ASSERT(agreed(body(ens, 0) == MFTL_NULL, MPI_COMM_WORLD));
  TEST_ASSERT_EQUAL_INT(2, ens->n_bodies_global);

  TEST_ASSERT(agreed(ensemble_is_sound(ens), MPI_COMM_WORLD));
  TEST_ASSERT(agreed(ownership_is_exclusive(ens, 1), MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A body arrives on a rank, leaves it, and comes back -- repeatedly.
 *
 * The lifecycle a sparse object list introduces, exercised as a cycle rather than
 * as a single transition, because the interesting failures are the ones that need
 * a rank to have held the body BEFORE:
 *
 *   - a stale `body_state` reused for the arriving body, so the rebuild reads a
 *     kept-list pointer belonging to whatever occupied that workspace slot last;
 *   - a `partition_settled` flag surviving the destruction, so a body that has
 *     just arrived is skipped on the next step and never gets its settling
 *     rebuild;
 *   - the id counters not travelling with the body, so a rank that created it
 *     sizes its arrival tables from zero.
 *
 * A small sphere marched along x through every cell of the grid, one cell per
 * step, so it visits every rank in the row and returns to the first.
 */
static int test_a_body_arrives_leaves_and_returns(void)
{
  if (g_nproc < 2) return 0;  // nowhere to go

  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);

  double cell[3];
  for (int d = 0; d < 3; ++d) cell[d] = 2.0 / (double) ens->domain.cuts[d];
  const double smallest = MFTL_MIN(cell[0], MFTL_MIN(cell[1], cell[2]));

  add_sphere(ens, 0, -1.0 + 0.5 * cell[0], -1.0 + 0.5 * cell[1], -1.0 + 0.5 * cell[2], 0.3 * smallest, 2);
  MFTLRedistribute(ens);
  refresh_geometry(ens);

  const int expect_t = ownership_is_exclusive(ens, 0);  // also the first structural check
  TEST_ASSERT(agreed(expect_t, MPI_COMM_WORLD));

  MFTL_RedistributeStats st0;
  MFTLRedistributeGetStats(&st0);

  /* Two full laps of the x row, so the body returns to ranks it has already been
   * destroyed on -- which is the state a fresh arrival cannot reach. */
  const int laps = 2 * ens->domain.cuts[0];

  int ok = 1;
  int steps_held = 0;
  for (int step = 0; step < laps; ++step)
  {
    FOREACH_BODY_VERTEX(ens, 0, pos[0] += cell[0]; MFTLApplyPeriodicity(&ens->domain, pos););

    if (MFTLUpdateGhostPositions(ens) != MFTL_SUCCESS) ok = 0;
    refresh_geometry(ens);
    if (MFTLRedistribute(ens) != MFTL_SUCCESS) ok = 0;
    refresh_geometry(ens);

    if (!ensemble_is_sound(ens)) ok = 0;
    if (!ownership_is_exclusive(ens, 0)) ok = 0;

    const MFTL_Object *const b = body(ens, 0);
    if (b != MFTL_NULL)
    {
      steps_held++;

      /* The counters travelled with the body: a rank that created it this step
       * must still size an id table for the whole mesh. */
      if (b->next_vertex_gid != b->global_n_vertices) ok = 0;
      if (b->next_halfedge_gid != 3 * b->next_triangle_gid) ok = 0;
    }
  }
  TEST_ASSERT(agreed(ok, MPI_COMM_WORLD));

  /* NON-VACUITY, and it is two claims. Bodies really were created and destroyed
   * -- without that the whole lifecycle is untested -- and this rank really did
   * both hold the body and not hold it, without which "returns" is a word rather
   * than a state that was reached. */
  MFTL_RedistributeStats st1;
  MFTLRedistributeGetStats(&st1);

  long churn[2] = {st1.n_objects_created - st0.n_objects_created, st1.n_objects_destroyed - st0.n_objects_destroyed};
  MPI_Allreduce(MPI_IN_PLACE, churn, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(churn[0] > 0);
  TEST_ASSERT(churn[1] > 0);

  /* `anywhere`, not `agreed`: the body marches along x, so ranks off that row
   * never hold it at all -- and requiring every rank to have both held and not
   * held it would fail on precisely the grids where the test covers most. What
   * has to be true of the RUN is that some rank went through the full cycle. */
  TEST_ASSERT(anywhere(steps_held > 0 && steps_held < laps, MPI_COMM_WORLD));

  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief An ensemble with no objects at all is a no-op, not an error. */
static int test_empty_ensemble_is_a_no_op(void)
{
  MFTL_Ensemble *ens = make_ensemble(MFTL_TRUE, 1);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLRedistribute(ens));
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLUpdateGhostPositions(ens));
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Invalid arguments are rejected without communicating.
 *
 * Rejection has to be LOCAL: a collective that some ranks enter and others reject
 * would hang, so an argument check that communicated would be worse than no check.
 */
static int test_invalid_arguments(void)
{
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRedistribute(MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLRedistributeWithOwners(MFTL_NULL, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLUpdateGhostPositions(MFTL_NULL));
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * main
 * ─────────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  MFTLRegisterMpiTypes();

  if (g_rank == 0) printf("\n=== test_redistribution (%d ranks) ===\n", g_nproc);

  RUN_TEST(test_wire_format_is_compact);
  RUN_TEST(test_mpi_types_register_is_idempotent);
  RUN_TEST(test_scatter_from_rank0);
  RUN_TEST(test_redistribute_is_idempotent);
  RUN_TEST(test_twin_repair_matches_full_rebuild);
  RUN_TEST(test_derived_arrays_match_a_full_rebuild);
  RUN_TEST(test_advection_conserves_entities);
  RUN_TEST(test_ghost_positions_are_synced);
  RUN_TEST(test_vertex_normals_match_serial_reference);
  RUN_TEST(test_ghost_vertex_normals_are_exact);
  RUN_TEST(test_ghost_syncs_leave_topology_current);
  RUN_TEST(test_multibody_round);
  RUN_TEST(test_body_can_leave_a_rank_entirely);
  RUN_TEST(test_body_straddling_a_periodic_seam);
  RUN_TEST(test_non_periodic_domain);
  RUN_TEST(test_skipping_reproduces_the_full_rebuild);
  RUN_TEST(test_a_stamped_body_is_never_skipped);
  RUN_TEST(test_generation_capacity_does_not_stay_resident_on_rank_0);
  RUN_TEST(test_shrinking_does_not_change_the_mesh);
  RUN_TEST(test_a_body_arrives_leaves_and_returns);
  RUN_TEST(test_empty_object_is_a_no_op);
  RUN_TEST(test_empty_ensemble_is_a_no_op);
  RUN_TEST(test_invalid_arguments);

  if (g_rank == 0) TEST_REPORT();

  /* A failure on ANY rank fails the run: the per-rank counters are local, and a
   * bug that only shows up on rank 3 is exactly the bug worth catching. */
  int local_failed = g_tests_failed;
  int total_failed = 0;
  MPI_Allreduce(&local_failed, &total_failed, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failed > 0) ? 1 : 0;
}
