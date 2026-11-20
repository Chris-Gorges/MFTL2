/**
 * @file test_ghost_pull.c
 * @brief The library's one ghost request/reply engine, MFTLGhostPull(), and its
 *        column payload, MFTLUpdateGhostColumns().
 *
 * Every reference value is a function of (body id, pool, global id, component)
 * computed here, so the check shares no code with the engine: owned entities
 * get the value, ghosts get poison, and after the pull every ghost must hold
 * its owner's value to the bit. Each distributed case counts what it covered
 * (ghosts refreshed, ghosts whose owner is not a graph neighbour) and asserts
 * the count, so a partition that left everything on one rank fails instead of
 * passing vacuously.
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "test_utils.h"

static int rank, nproc;

/** @brief All ranks' verdict, so no rank returns from a test while another waits in a collective. */
static int agreed(int ok)
{
  int all = 0;
  MPI_Allreduce(&ok, &all, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return all;
}

static long sum_long(long v)
{
  long all = 0;
  MPI_Allreduce(&v, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  return all;
}

/** @brief The reference: exact in binary64 for every argument used here. */
static double value_of(int body, int pool, int gid, int comp) { return body * 1048576.0 + pool * 524288.0 + gid * 8.0 + comp; }

static const double POISON = -12345.0;

/** @brief Periodic [-1,1]^3; slabs {nproc,1,1} when asked, so ranks two cells apart are not graph neighbours. */
static MFTL_Ensemble *make_ensemble(bool slabs)
{
  int cuts[3];
  test_grid_for(nproc, cuts);
  if (slabs)
  {
    cuts[0] = nproc;
    cuts[1] = cuts[2] = 1;
  }
  double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1};
  bool periodic[3] = {true, true, true};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 2);
}

static void add_sphere(MFTL_Ensemble *e, double x, double radius, int refine)
{
  MFTL_Object *o = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(e, o);
  const double centre[3] = {x, 0.1, -0.05};
  MFTLGenerateSphere(o, centre, radius, refine);
  MFTLComputeTriangleGeometry(o, o->tris.n);
}

/**
 * @brief Owned rows take the reference, ghost rows the poison, in every pulled
 *        array: positions (comp 0-2), normals (3-5), sizing (6), the vertex
 *        field (7-8) and the triangle field (0-2 of pool 1).
 */
static void fill(MFTL_Ensemble *e, int vfield, int tfield)
{
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    double *vf = MFTLObjectField(obj, vfield), *tf = MFTLObjectField(obj, tfield);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const bool ghost = v >= obj->verts.n_local;
      const int g = obj->verts.rec[v].global_id;
      for (int k = 0; k < 3; ++k)
      {
        obj->verts.pos[3 * v + k] = ghost ? POISON : value_of(obj->global_id, 0, g, k);
        obj->verts.normal[3 * v + k] = ghost ? POISON : value_of(obj->global_id, 0, g, 3 + k);
      }
      obj->verts.sizing[v] = ghost ? POISON : value_of(obj->global_id, 0, g, 6);
      for (int c = 0; c < 2; ++c) vf[2 * v + c] = ghost ? POISON : value_of(obj->global_id, 0, g, 7 + c);
    }
    for (int t = 0; t < obj->tris.n; ++t)
    {
      const bool ghost = t >= obj->tris.n_local;
      for (int c = 0; c < 3; ++c) tf[3 * t + c] = ghost ? POISON : value_of(obj->global_id, 1, obj->tris.rec[t].global_id, c);
    }
  }
}

/** @brief Every row of every pulled array holds the reference, ghosts included. */
static int every_row_is_the_owners(const MFTL_Ensemble *e, int vfield, int tfield, bool with_normals)
{
  int ok = 1;
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    const double *vf = MFTLObjectField(obj, vfield), *tf = MFTLObjectField(obj, tfield);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      const int g = obj->verts.rec[v].global_id;
      for (int k = 0; k < 3; ++k)
      {
        ok &= obj->verts.pos[3 * v + k] == value_of(obj->global_id, 0, g, k);
        if (with_normals) ok &= obj->verts.normal[3 * v + k] == value_of(obj->global_id, 0, g, 3 + k);
      }
      if (with_normals)
      {
        ok &= obj->verts.sizing[v] == value_of(obj->global_id, 0, g, 6);
        for (int c = 0; c < 2; ++c) ok &= vf[2 * v + c] == value_of(obj->global_id, 0, g, 7 + c);
      }
    }
    if (with_normals)
      for (int t = 0; t < obj->tris.n; ++t)
        for (int c = 0; c < 3; ++c) ok &= tf[3 * t + c] == value_of(obj->global_id, 1, obj->tris.rec[t].global_id, c);
  }
  return ok;
}

/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * Five arrays of both pools in ONE pull on the canonical partition: every ghost
 * takes its owner's row to the bit, owned rows are untouched, the geometry bit
 * is cleared because positions were pulled -- and the AUTO route stays on the
 * graph, because nothing here is out of reach (a pinned global route would be a
 * scaling regression no value can show).
 */
static int test_every_ghost_takes_its_owners_rows(void)
{
  MFTL_Ensemble *e = make_ensemble(false);
  const int vfield = MFTLEnsembleRegisterField(e, "v2", MFTL_FIELD_VERTEX, 2, MFTL_FIELD_RECOMPUTED);
  const int tfield = MFTLEnsembleRegisterField(e, "t3", MFTL_FIELD_TRIANGLE, 3, MFTL_FIELD_RECOMPUTED);
  add_sphere(e, 0.0, 0.65, 3);
  add_sphere(e, 0.8, 0.2, 2);
  int ok = MFTLRedistribute(e) == MFTL_SUCCESS;

  long ghosts = 0;
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    MFTLObjectEnableSizing(obj);
    ghosts += (obj->verts.n - obj->verts.n_local) + (obj->tris.n - obj->tris.n_local);
  }
  fill(e, vfield, tfield);

  const MFTL_GhostColumn cols[5] = {{MFTL_GHOST_COLUMN_POSITION, 0},
                                    {MFTL_GHOST_COLUMN_NORMAL, 0},
                                    {MFTL_GHOST_COLUMN_SIZING, 0},
                                    {MFTL_GHOST_COLUMN_FIELD, vfield},
                                    {MFTL_GHOST_COLUMN_FIELD, tfield}};
  MFTLRedistributeStatsReset();
  ok &= MFTLUpdateGhostColumns(e, cols, 5, MFTL_GHOST_ROUTE_AUTO) == MFTL_SUCCESS;
  MFTL_RedistributeStats st;
  MFTLRedistributeGetStats(&st);
  ok &= st.g_n_route_global == 0;
  ok &= st.g_n_ghosts_requested == ghosts;
  ok &= every_row_is_the_owners(e, vfield, tfield, true);
  for (int o = 0; o < e->n_local_objects; ++o) ok &= (e->local_objects[o]->derived & MFTL_DERIVED_GEOMETRY) == 0;

  /* The public wrapper on the graph route: positions alone, again from poison. */
  fill(e, vfield, tfield);
  ok &= MFTLUpdateGhostPositions(e) == MFTL_SUCCESS;
  ok &= every_row_is_the_owners(e, vfield, tfield, false);

  const long all_ghosts = sum_long(ghosts);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(agreed(ok));
  TEST_ASSERT(nproc == 1 || all_ghosts > 0);  // non-vacuity
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */

/** @brief Keys well above 2^32, unique per (body, pool, id), identical on every holder. */
static long long key_of(int body, int pool, int gid) { return ((long long) gid * 2 + pool) * 1000003LL + (long long) body * 7 + (1LL << 41); }

typedef struct
{
  int applied;
  int wrong;
} keyed_tally;

static void serve_owner_id(const MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, const int *offset, int n, unsigned char *reply, void *ctx)
{
  (void) slot;
  (void) ctx;
  for (int j = 0; j < n; ++j)
  {
    const int id = (idx[j] == MFTL_ERROR) ? MFTL_ERROR : (pool == MFTL_FIELD_VERTEX) ? obj->verts.rec[idx[j]].global_id : obj->tris.rec[idx[j]].global_id;
    memcpy(reply + offset[j], &id, sizeof id);
  }
}

static int apply_check_id(MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, int n, const unsigned char *rows, void *ctx)
{
  (void) slot;
  keyed_tally *const tally = (keyed_tally *) ctx;
  for (int j = 0; j < n; ++j)
  {
    int id;
    memcpy(&id, rows + (size_t) j * sizeof id, sizeof id);
    const int mine = (pool == MFTL_FIELD_VERTEX) ? obj->verts.rec[idx[j]].global_id : obj->tris.rec[idx[j]].global_id;
    tally->wrong += (id != mine) || (obj->global_id != 0);  // body 1 takes no part
    tally->applied++;
  }
  return 0;
}

/**
 * A KEYED pull (the topology change's form) resolves 64-bit keys above 2^32 to
 * the right owned entity in both pools, and a body whose key array is NULL
 * takes no part at all: none of its ghosts is asked, none is answered.
 */
static int test_a_keyed_pull_resolves_64_bit_keys(void)
{
  MFTL_Ensemble *e = make_ensemble(false);
  add_sphere(e, 0.0, 0.65, 3);
  add_sphere(e, 0.8, 0.2, 2);
  int ok = MFTLRedistribute(e) == MFTL_SUCCESS;

  const int nl = e->n_local_objects;
  long long **owned = calloc((size_t) (2 * nl + 2), sizeof(long long *));
  const long long **by_slot = calloc((size_t) (2 * nl + 2), sizeof(long long *));
  long expected = 0;
  for (int o = 0; o < nl; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    if (obj->global_id != 0) continue;  // body 1 does not take part
    for (int p = 0; p < 2; ++p)
    {
      const int n = p ? obj->tris.n : obj->verts.n;
      owned[2 * o + p] = malloc(((size_t) n + 1) * sizeof(long long));
      for (int i = 0; i < n; ++i) owned[2 * o + p][i] = key_of(obj->global_id, p, p ? obj->tris.rec[i].global_id : obj->verts.rec[i].global_id);
      expected += n - (p ? obj->tris.n_local : obj->verts.n_local);
    }
  }
  for (int o = 0; o < nl; ++o)
  {
    by_slot[o] = owned[2 * o];
    by_slot[nl + 1 + o] = owned[2 * o + 1];
  }

  keyed_tally tally = {0, 0};
  MFTL_GhostPull pull;
  pull.row_bytes[MFTL_FIELD_VERTEX] = pull.row_bytes[MFTL_FIELD_TRIANGLE] = (int) sizeof(int);
  pull.route = MFTL_GHOST_ROUTE_AUTO;
  pull.keys[MFTL_FIELD_VERTEX] = by_slot;
  pull.keys[MFTL_FIELD_TRIANGLE] = by_slot + nl + 1;
  pull.serve = serve_owner_id;
  pull.apply = apply_check_id;
  pull.ctx = &tally;
  ok &= MFTLGhostPull(e, &pull) == MFTL_SUCCESS;
  ok &= tally.wrong == 0;
  ok &= tally.applied == expected;

  for (int i = 0; i < 2 * nl + 2; ++i) free(owned[i]);
  free(owned);
  free(by_slot);
  const long all = sum_long(tally.applied);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(agreed(ok));
  TEST_ASSERT(nproc == 1 || all > 0);  // non-vacuity
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */

/**
 * A partition whose ghosts are owned by ranks that are NOT graph neighbours
 * (slabs, ownership alternating between rank 0 and rank nproc/2 by id): the
 * AUTO route agrees on comm_world and refreshes every ghost; the graph route
 * cannot, keeps the old values and says so (Release only -- Debug asserts on
 * the orphan by design). Vacuous below four ranks, where every rank is a
 * neighbour of every other; registered at np = 4 and 8.
 */
static int test_the_auto_route_reaches_owners_off_the_graph(void)
{
  MFTL_Ensemble *e = make_ensemble(true);
  const int vfield = MFTLEnsembleRegisterField(e, "v2", MFTL_FIELD_VERTEX, 2, MFTL_FIELD_RECOMPUTED);
  const int tfield = MFTLEnsembleRegisterField(e, "t3", MFTL_FIELD_TRIANGLE, 3, MFTL_FIELD_RECOMPUTED);
  add_sphere(e, -0.75, 0.15, 2);
  int ok = MFTLRedistribute(e) == MFTL_SUCCESS;

  for (int o = 0; o < e->n_local_objects; ++o) MFTLComputeTriangleGeometry(e->local_objects[o], e->local_objects[o]->tris.n);

  /* Every holder computes the same owner for an entity: a function of its id. */
  const int far_rank = nproc / 2;
  const int held = e->n_local_objects;  // the local count is stale after the redistribution below
  MFTL_OwnerMap *maps = calloc((size_t) held + 1, sizeof(MFTL_OwnerMap));
  int **arrays = calloc(2 * (size_t) held + 2, sizeof(int *));
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    int *vo = malloc(((size_t) obj->verts.n + 1) * sizeof(int)), *to = malloc(((size_t) obj->tris.n + 1) * sizeof(int));
    for (int v = 0; v < obj->verts.n; ++v) vo[v] = (obj->verts.rec[v].global_id % 2) ? far_rank : 0;
    for (int t = 0; t < obj->tris.n; ++t) to[t] = (obj->tris.rec[t].global_id % 2) ? 0 : far_rank;
    maps[o].vert_owner = vo;
    maps[o].tri_owner = to;
    arrays[2 * o] = vo;
    arrays[2 * o + 1] = to;
  }
  ok &= MFTLRedistributeWithOwners(e, maps) == MFTL_SUCCESS;
  for (int i = 0; i < 2 * held + 2; ++i) free(arrays[i]);
  free(arrays);
  free(maps);

  long off_graph = 0;
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    MFTL_Object *obj = e->local_objects[o];
    MFTLObjectEnableSizing(obj);
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v) off_graph += MFTLDomainNeighborIndex(&e->domain, obj->verts.rec[v].owner_proc) == MFTL_ERROR;
  }
  const long all_off_graph = sum_long(off_graph);

#ifdef NDEBUG
  /* The graph route: an unreachable owner leaves the poison and fails the call
   * on exactly the ranks that hold such a ghost. */
  fill(e, vfield, tfield);
  const int graph_rc = MFTLUpdateGhostPositions(e);
  ok &= (graph_rc == MFTL_ERROR) == (off_graph > 0);
  for (int o = 0; o < e->n_local_objects; ++o)
  {
    const MFTL_Object *obj = e->local_objects[o];
    for (int v = obj->verts.n_local; v < obj->verts.n; ++v)
      if (MFTLDomainNeighborIndex(&e->domain, obj->verts.rec[v].owner_proc) == MFTL_ERROR) ok &= obj->verts.pos[3 * v] == POISON;
  }
#endif

  fill(e, vfield, tfield);
  const MFTL_GhostColumn cols[5] = {{MFTL_GHOST_COLUMN_POSITION, 0},
                                    {MFTL_GHOST_COLUMN_NORMAL, 0},
                                    {MFTL_GHOST_COLUMN_SIZING, 0},
                                    {MFTL_GHOST_COLUMN_FIELD, vfield},
                                    {MFTL_GHOST_COLUMN_FIELD, tfield}};
  MFTLRedistributeStatsReset();
  ok &= MFTLUpdateGhostColumns(e, cols, 5, MFTL_GHOST_ROUTE_AUTO) == MFTL_SUCCESS;
  MFTL_RedistributeStats st;
  MFTLRedistributeGetStats(&st);
  ok &= every_row_is_the_owners(e, vfield, tfield, true);
  ok &= st.g_n_route_global == ((all_off_graph > 0) ? 1 : 0);

  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(agreed(ok));
  TEST_ASSERT(nproc < 4 || all_off_graph > 0);  // non-vacuity from four ranks up
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────── */

static void serve_nothing(const MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, const int *offset, int n, unsigned char *reply, void *ctx)
{
  (void) obj, (void) slot, (void) pool, (void) idx, (void) offset, (void) n, (void) reply, (void) ctx;
}

static int apply_nothing(MFTL_Object *obj, int slot, MFTL_FieldEntity pool, const int *idx, int n, const unsigned char *rows, void *ctx)
{
  (void) obj, (void) slot, (void) pool, (void) idx, (void) n, (void) rows, (void) ctx;
  return 0;
}

/**
 * Invalid arguments are refused on every rank before any communication (so a
 * refusal cannot strand a rank in a collective), and a column the bodies do
 * not have is NOT silently skipped: the ghosts keep their values and every
 * rank that holds one reports the failure.
 */
static int test_refusals_and_a_missing_column(void)
{
  MFTL_Ensemble *e = make_ensemble(false);
  add_sphere(e, 0.0, 0.65, 2);
  int ok = MFTLRedistribute(e) == MFTL_SUCCESS;

  MFTL_GhostPull pull;
  pull.row_bytes[0] = 8;
  pull.row_bytes[1] = 0;
  pull.route = MFTL_GHOST_ROUTE_GRAPH;
  pull.keys[0] = pull.keys[1] = MFTL_NULL;
  pull.serve = MFTL_NULL;
  pull.apply = apply_nothing;
  pull.ctx = MFTL_NULL;
  ok &= MFTLGhostPull(e, &pull) == MFTL_ERROR;  // no owner side
  pull.serve = serve_nothing;
  pull.row_bytes[0] = 0;
  ok &= MFTLGhostPull(e, &pull) == MFTL_ERROR;  // no pool
  pull.row_bytes[0] = -4;
  ok &= MFTLGhostPull(e, &pull) == MFTL_ERROR;  // negative width

  const MFTL_GhostColumn twice[2] = {{MFTL_GHOST_COLUMN_NORMAL, 0}, {MFTL_GHOST_COLUMN_NORMAL, 0}};
  const MFTL_GhostColumn no_field = {MFTL_GHOST_COLUMN_FIELD, 3};
  ok &= MFTLUpdateGhostColumns(e, twice, 2, MFTL_GHOST_ROUTE_GRAPH) == MFTL_ERROR;
  ok &= MFTLUpdateGhostColumns(e, &no_field, 1, MFTL_GHOST_ROUTE_GRAPH) == MFTL_ERROR;
  ok &= MFTLUpdateGhostColumns(e, twice, 0, MFTL_GHOST_ROUTE_GRAPH) == MFTL_ERROR;

  /* No body has `verts.sizing`: nothing can be served or applied. */
  long ghosts = 0;
  for (int o = 0; o < e->n_local_objects; ++o) ghosts += e->local_objects[o]->verts.n - e->local_objects[o]->verts.n_local;
  const MFTL_GhostColumn sizing = {MFTL_GHOST_COLUMN_SIZING, 0};
  const int rc = MFTLUpdateGhostColumns(e, &sizing, 1, MFTL_GHOST_ROUTE_AUTO);
  ok &= (rc == MFTL_ERROR) == (ghosts > 0);

  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(agreed(ok));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  RUN_TEST(test_every_ghost_takes_its_owners_rows);
  RUN_TEST(test_a_keyed_pull_resolves_64_bit_keys);
  RUN_TEST(test_the_auto_route_reaches_owners_off_the_graph);
  RUN_TEST(test_refusals_and_a_missing_column);
  if (rank == 0) TEST_REPORT();
  const int fail = g_tests_failed;
  MPI_Finalize();
  return fail ? 1 : 0;
}
