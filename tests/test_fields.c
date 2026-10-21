/* Optional fields: independently check identity, mass, ghosts and rejected paths. */
#include "mftl_core.h"
#include "mftl_topology_change.h"
#include "test_utils.h"
#include "test_mesh_invariants.h"
#include <stdint.h>

static int rank, nproc;
static int field_agreed(int value)
{
  int result;
  MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  return result;
}

static MFTL_Ensemble *make_ens(MPI_Comm comm, bool slabs)
{
  int n;
  MPI_Comm_size(comm, &n);
  int cuts[3];
  test_grid_for(n, cuts);
  if (slabs)
  {
    cuts[0] = n;
    cuts[1] = cuts[2] = 1;
  }
  double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1};
  bool periodic[3] = {true, true, true};
  return MFTLEnsembleCreate(comm, cuts, lo, hi, periodic, 2);
}

static int register_fields(MFTL_Ensemble *e)
{
  int ok = MFTLEnsembleRegisterField(e, "vertex", MFTL_FIELD_VERTEX, 7, MFTL_FIELD_INTERPOLATED) == 0;
  ok &= MFTLEnsembleRegisterField(e, "triangle", MFTL_FIELD_TRIANGLE, 9, MFTL_FIELD_INTERPOLATED) == 1;
  ok &= MFTLEnsembleRegisterField(e, "scratch", MFTL_FIELD_VERTEX, 4, MFTL_FIELD_RECOMPUTED) == 2;
  ok &= MFTLEnsembleRegisterField(e, "concentration", MFTL_FIELD_TRIANGLE, 2, MFTL_FIELD_CONSERVATIVE_DENSITY) == 3;
  return ok;
}

static MFTL_Object *sphere(MFTL_Ensemble *e, double x, double radius, int refine)
{
  MFTL_Object *o = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(e, o);
  double center[3] = {x, 0, 0};
  MFTLGenerateSphere(o, center, radius, refine);
  return o;
}

static double tag(int body, int field, int gid, int comp, int step) { return body * 1000000.0 + field * 100000.0 + gid * 16.0 + comp + step * 0.125; }

static void set_tags(MFTL_Object *o, int step, bool poison_ghosts)
{
  for (int f = 0; f < o->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &o->fields->entries[f];
    const int n = d->entity ? o->tris.n : o->verts.n;
    const int nl = d->entity ? o->tris.n_local : o->verts.n_local;
    double *v = MFTLObjectField(o, f);
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < d->ncomp; ++c)
      {
        int gid = d->entity ? o->tris.rec[i].global_id : o->verts.rec[i].global_id;
        v[(size_t) i * d->ncomp + c] = poison_ghosts && i >= nl ? -999.0 : tag(o->global_id, f, gid, c, step);
      }
  }
}

static int check_tags(const MFTL_Object *o, int step, bool scratch)
{
  int ok = MFTLObjectPoolsAreConsistent(o);
  for (int f = 0; f < o->fields->n; ++f)
  {
    const MFTL_FieldDescriptor *d = &o->fields->entries[f];
    if (!scratch && d->mode == MFTL_FIELD_RECOMPUTED) continue;
    const int n = d->entity ? o->tris.n : o->verts.n;
    for (int i = 0; i < n; ++i)
      for (int c = 0; c < d->ncomp; ++c)
      {
        int gid = d->entity ? o->tris.rec[i].global_id : o->verts.rec[i].global_id;
        ok &= o->field_data[f][(size_t) i * d->ncomp + c] == tag(o->global_id, f, gid, c, step);
      }
  }
  return ok;
}

static int test_registry_and_storage(void)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_WORLD, false);
  int ok = register_fields(e);
  ok &= MFTLEnsembleFindField(e, "triangle") == 1;
  ok &= MFTLEnsembleRegisterField(e, "vertex", MFTL_FIELD_VERTEX, 1, MFTL_FIELD_INTERPOLATED) == MFTL_ERROR;
  ok &= MFTLEnsembleRegisterField(e, "bad", MFTL_FIELD_VERTEX, 1, MFTL_FIELD_CONSERVATIVE_DENSITY) == MFTL_ERROR;
  if (nproc > 1) ok &= MFTLEnsembleRegisterField(e, rank == 0 ? "one" : "two", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_INTERPOLATED) == MFTL_ERROR;
  MFTL_Object *o = MFTLObjectCreate(0);
  /* Attach an already populated standalone mesh without changing its capacity. */
  const double pos[9] = {0, 0, 0, .2, 0, 0, 0, .2, 0};
  const int tri[3] = {0, 1, 2};
  MFTLBuildIndexedMesh(o, pos, 3, tri, 1);
  MFTLEnsembleAddObject(e, o);
  ok &= MFTLEnsembleRegisterField(e, "late", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_INTERPOLATED) == MFTL_ERROR;
  set_tags(o, 0, false);
  MFTLObjectReserve(o, 1000, 1000);
  ok &= check_tags(o, 0, true);
  for (int f = 0; f < e->fields.n; ++f)
  {
    const MFTL_FieldDescriptor *d = &e->fields.entries[f];
    int cap = d->entity ? o->tris.capacity : o->verts.capacity;
    o->field_data[f][(size_t) cap * d->ncomp - 1] = 123.0;
  }
  MFTLObjectShrinkToFit(o, 1.0);
  ok &= check_tags(o, 0, true);
  MFTLObjectClear(o);
  MFTLObjectReleasePools(o);
  for (int f = 0; f < e->fields.n; ++f) ok &= MFTLObjectField(o, f) == NULL;
  MFTLObjectSetCounts(o, 3, 3, 1, 1);
  for (int f = 0; f < e->fields.n; ++f) ok &= MFTLObjectField(o, f)[0] == 0.0;
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

static int test_movement_and_ghosts(void)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_WORLD, false);
  int ok = register_fields(e);
  MFTL_Object *o = sphere(e, 0, .65, 2);
  set_tags(o, 0, false);
  o = sphere(e, .85, .2, 1);
  set_tags(o, 0, false);
  ok &= MFTLRedistribute(e) == MFTL_SUCCESS;
  int invalid = rank == 0 ? -1 : 0;
  const int duplicate[2] = {0, 0};
  ok &= MFTLUpdateGhostFields(e, &invalid, 1, false) == MFTL_ERROR;
  ok &= MFTLUpdateGhostFields(e, duplicate, 2, false) == MFTL_ERROR;
  ok &= MFTLUpdateGhostFields(e, NULL, 0, false) == MFTL_SUCCESS;
  long ghosts = 0;
  const int ids[4] = {0, 1, 2, 3};
  for (int step = 1; step <= 10; ++step)
  {
    for (int j = 0; j < e->n_local_objects; ++j)
    {
      o = e->local_objects[j];
      ok &= check_tags(o, step - 1, false);
      MFTLObjectReorderMorton(o);
      ok &= check_tags(o, step - 1, false);
      set_tags(o, step, true);
      ghosts += o->verts.n - o->verts.n_local + o->tris.n - o->tris.n_local;
    }
    ok &= MFTLUpdateGhostFields(e, ids, 4, false) == MFTL_SUCCESS;
    for (int j = 0; j < e->n_local_objects; ++j)
    {
      o = e->local_objects[j];
      ok &= check_tags(o, step, true);
      for (int v = 0; v < o->verts.n_local; ++v)
      {
        o->verts.pos[3 * v] += .19;
        MFTLApplyPeriodicity(&e->domain, o->verts.pos + 3 * v);
      }
      MFTLObjectMarkChanged(o);
    }
    ok &= MFTLUpdateGhostFields(e, ids, 4, true) == MFTL_SUCCESS;
    for (int j = 0; j < e->n_local_objects; ++j) MFTLComputeTriangleGeometry(e->local_objects[j], e->local_objects[j]->tris.n);
    ok &= MFTLRedistribute(e) == MFTL_SUCCESS;
    for (int j = 0; j < e->n_local_objects; ++j) ok &= check_tags(e->local_objects[j], step, false);
  }
  long allghosts;
  MPI_Allreduce(&ghosts, &allghosts, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  ok &= nproc == 1 || allghosts > 0;
  /* Only the selected scratch field changes; field-only sync preserves generation. */
  uint64_t *gen = malloc((size_t) (e->n_local_objects + 1) * sizeof(uint64_t));
  for (int j = 0; j < e->n_local_objects; ++j)
  {
    gen[j] = e->local_objects[j]->generation;
    set_tags(e->local_objects[j], 11, true);
  }
  int only = 2;
  ok &= MFTLUpdateGhostFields(e, &only, 1, false) == MFTL_SUCCESS;
  for (int j = 0; j < e->n_local_objects; ++j)
  {
    o = e->local_objects[j];
    ok &= o->generation == gen[j];
    for (int v = o->verts.n_local; v < o->verts.n; ++v) ok &= o->field_data[0][7 * v] == -999.0;
    for (int t = o->tris.n_local; t < o->tris.n; ++t) ok &= o->field_data[1][9 * t] == -999.0;
    for (int v = 0; v < o->verts.n; ++v) ok &= o->field_data[2][4 * v] == tag(o->global_id, 2, o->verts.rec[v].global_id, 0, 11);
  }
  free(gen);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

static void initialize_physics(MFTL_Object *o)
{
  for (int v = 0; v < o->verts.n; ++v)
    for (int c = 0; c < 7; ++c) o->field_data[0][7 * v + c] = .5 + .4 * sin(o->verts.pos[3 * v] + c);
  for (int t = 0; t < o->tris.n; ++t)
  {
    double p[3][3];
    MFTLGetTriangleCorners(o, t, p);
    o->field_data[3][2 * t] = 1.0 + .7 * sin(3 * (p[0][0] + p[1][0] + p[2][0]));
    o->field_data[3][2 * t + 1] = 0;
    for (int c = 0; c < 9; ++c) o->field_data[1][9 * t + c] = 4.125;
  }
}

static double mass(MFTL_Ensemble *e)
{
  double mine = 0, all;
  for (int o = 0; o < e->n_local_objects; ++o)
    for (int t = 0; t < e->local_objects[o]->tris.n_local; ++t)
      if (!(e->local_objects[o]->tris.rec[t].flags & MFTL_FLAG_DEAD))
        mine += MFTLGetTriangleArea(e->local_objects[o], t) * e->local_objects[o]->field_data[3][2 * t];
  MPI_Allreduce(&mine, &all, 1, MPI_DOUBLE, MPI_SUM, e->domain.comm_world);
  return all;
}

static int physics_valid(MFTL_Ensemble *e)
{
  int ok = 1;
  for (int j = 0; j < e->n_local_objects; ++j)
  {
    MFTL_Object *o = e->local_objects[j];
    for (int t = 0; t < o->tris.n; ++t)
      if (!(o->tris.rec[t].flags & MFTL_FLAG_DEAD))
      {
        ok &= isfinite(o->field_data[3][2 * t]) && o->field_data[3][2 * t] >= 0;
        ok &= o->field_data[3][2 * t + 1] == 0;
        for (int c = 0; c < 9; ++c) ok &= fabs(o->field_data[1][9 * t + c] - 4.125) < 1e-12;
      }
    for (int v = 0; v < o->verts.n; ++v)
      for (int c = 0; c < 7; ++c) ok &= o->field_data[0][7 * v + c] >= .1 - 1e-14 && o->field_data[0][7 * v + c] <= .9 + 1e-14;
  }
  return ok;
}

static int test_individual_operators(void)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_SELF, false);
  int ok = register_fields(e);
  MFTL_Object *o = sphere(e, 0, .6, 2);
  initialize_physics(o);
  double initial = mass(e);
  int split = MFTLSplitEdge(o, 0, NULL);
  ok &= split >= 0;
  ok &= fabs(mass(e) - initial) < 1e-12 * initial;
  MFTLRemeshCompactObject(o, NULL);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, .4, 2, MFTL_SMOOTH_TRIANGLE_BASED, 180);
  for (int v = 0; v < o->verts.n; ++v) o->verts.rec[v].flags &= ~MFTL_FLAG_VISITED;
  for (int h = 0; h < 3 * o->tris.n; ++h) o->tris.he[h].flags &= ~MFTL_FLAG_VISITED;
  int collapsed = 0;
  for (int h = 0; h < 3 * o->tris.n; ++h)
    if (MFTLCollapseApproved(o, h, &p, NULL))
    {
      collapsed = MFTLCollapseEdge(o, h, &p) == MFTL_SUCCESS;
      break;
    }
  if (!collapsed) fprintf(stderr, "no approved collapse in fixture\n");
  ok &= collapsed;
  ok &= fabs(mass(e) - initial) < 1e-12 * initial;
  MFTLRemeshCompactObject(o, NULL);
  /* A regular sphere need not have a quality-improving flip. Exercise the
     public primitive on a remote interior edge instead. */
  int flipped = MFTLFlipEdge(o, 3 * (o->tris.n / 2)) == MFTL_SUCCESS;
  if (!flipped) fprintf(stderr, "no approved flip in fixture\n");
  ok &= flipped;
  ok &= fabs(mass(e) - initial) < 1e-12 * initial;
  MFTLObjectBuildDerived(o);
  MFTLDetectFeatures(o, 180, p.feature_corner_angle, NULL);
  MFTLSmoothTangential(o, &p, 3);
  ok &= fabs(mass(e) - initial) < 1e-12 * initial && physics_valid(e);
  MFTLRemeshCompactObject(o, NULL);
  MFTLObjectReorderMorton(o);
  ok &= fabs(mass(e) - initial) < 1e-12 * initial;
  ok &= inv_mesh_is_sound(o, o->domain->rank);
  if (!ok)
    fprintf(stderr, "operators: split=%d collapsed=%d flipped=%d masserr=%g physics=%d\n", split, collapsed, flipped, mass(e) - initial, physics_valid(e));
  /* A rejected split leaves counts, positions and field values intact. */
  int nv = o->verts.n, nt = o->tris.n;
  double bad = NAN;
  o->field_data[3][0] = bad;
  int refused = MFTLSplitEdge(o, 0, NULL);
  ok &= refused == MFTL_ERROR && nv == o->verts.n && nt == o->tris.n && isnan(o->field_data[3][0]);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

static int remesh_path(MFTL_RemeshParallelStrategy strategy)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_WORLD, false);
  int ok = register_fields(e);
  const bool dispersed = strategy == MFTL_REMESH_PARALLEL_DISPERSED;
  MFTL_Object *o = sphere(e, dispersed ? .95 : 0, dispersed ? .18 : .65, 2);
  initialize_physics(o);
  if (dispersed)
  {
    o = sphere(e, -.3, .15, 2);
    initialize_physics(o);
  }
  double initial = mass(e);
  ok &= MFTLRedistribute(e) == MFTL_SUCCESS;
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, dispersed ? .055 : .13, 2, MFTL_SMOOTH_TRIANGLE_BASED, 180);
  p.parallel_strategy = strategy;
  p.parallel_phase_budget = 8;
  long ops = 0;
  for (int i = 0; i < 2; ++i)
  {
    MFTL_RemeshStats st;
    ok &= MFTLRemeshEnsemble(e, &p, &st) == MFTL_SUCCESS;
    ops += st.n_split + st.n_collapse + st.n_flip;
    double after = mass(e);
    if (fabs(after - initial) > 1e-11 * initial) fprintf(stderr, "rank %d mass %.17g -> %.17g\n", rank, initial, after);
    ok &= fabs(after - initial) < 1e-11 * initial && physics_valid(e);
    const int ids[3] = {0, 1, 3};
    ok &= MFTLUpdateGhostFields(e, ids, 3, false) == MFTL_SUCCESS;
    for (int j = 0; j < e->n_local_objects; ++j) ok &= inv_partition_is_sound(e->local_objects[j], rank);
  }
  long allops;
  MPI_Allreduce(&ops, &allops, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  ok &= allops > 0;
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}
static int test_spanning(void) { return remesh_path(MFTL_REMESH_PARALLEL_SPANNING); }
static int test_dispersed(void) { return remesh_path(MFTL_REMESH_PARALLEL_DISPERSED); }

static int test_global_route_and_guards(void)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_WORLD, true);
  int ok = register_fields(e);
  MFTL_Object *o = sphere(e, -.75, .08, 1);
  set_tags(o, 0, false);
  MFTL_OwnerMap *maps = calloc((size_t) e->n_local_objects, sizeof(*maps));
  int *vo = malloc((size_t) (o->verts.n + 1) * sizeof(int)), *to = malloc((size_t) (o->tris.n + 1) * sizeof(int));
  for (int i = 0; i < o->verts.n; ++i) vo[i] = nproc / 2;
  for (int i = 0; i < o->tris.n; ++i) to[i] = nproc / 2;
  maps[0].vert_owner = vo;
  maps[0].tri_owner = to;
  MFTLRedistributeStatsReset();
  ok &= MFTLRedistributeWithOwners(e, maps) == MFTL_SUCCESS;
  MFTL_RedistributeStats stats;
  MFTLRedistributeGetStats(&stats);
  if (nproc >= 4) ok &= stats.n_route_global > 0;
  for (int j = 0; j < e->n_local_objects; ++j) ok &= check_tags(e->local_objects[j], 0, false);
  free(vo);
  free(to);
  free(maps);
  for (int j = 0; j < e->n_local_objects; ++j) MFTLComputeTriangleGeometry(e->local_objects[j], e->local_objects[j]->tris.n);
  ok &= MFTLRedistribute(e) == MFTL_SUCCESS;
  const int ids[3] = {0, 1, 3};
  ok &= MFTLUpdateGhostFields(e, ids, 3, false) == MFTL_SUCCESS;
  MFTL_TopologyOptions opt = MFTLTopologyDefaultOptions();
  opt.dx = .04;
  int body = 0;
  double before = mass(e);
  ok &= MFTLEnsembleReconstructBodies(e, &body, 1, NULL, 0, &opt, NULL) == MFTL_ERROR;
  ok &= MFTLEnsembleApplyTopologyChanges(e, &opt, NULL) == MFTL_ERROR;
  ok &= before == mass(e);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

static int test_open_rim_and_compaction(void)
{
  MFTL_Ensemble *e = make_ens(MPI_COMM_SELF, false);
  int ok = register_fields(e);
  MFTL_Object *o = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(e, o);
  enum
  {
    N = 6,
    NV = (N + 1) * (N + 1),
    NT = 2 * N * N
  };
  double pos[3 * NV];
  int tri[3 * NT], t = 0;
  for (int y = 0; y <= N; ++y)
    for (int x = 0; x <= N; ++x)
    {
      int v = y * (N + 1) + x;
      pos[3 * v] = -.45 + .9 * x / N;
      pos[3 * v + 1] = -.45 + .9 * y / N;
      pos[3 * v + 2] = 0;
    }
  for (int y = 0; y < N; ++y)
    for (int x = 0; x < N; ++x)
    {
      int v = y * (N + 1) + x;
      tri[3 * t] = v;
      tri[3 * t + 1] = v + 1;
      tri[3 * t + 2] = v + N + 2;
      ++t;
      tri[3 * t] = v;
      tri[3 * t + 1] = v + N + 2;
      tri[3 * t + 2] = v + N + 1;
      ++t;
    }
  MFTLBuildIndexedMesh(o, pos, NV, tri, NT);
  initialize_physics(o);
  double initial = mass(e);
  int rim = 0;
  while (rim < 3 * o->tris.n && o->tris.he[rim].twin_idx != MFTL_ERROR) ++rim;
  ok &= rim < 3 * o->tris.n && MFTLSplitEdge(o, rim, NULL) >= 0;
  ok &= fabs(mass(e) - initial) < 1e-12 * initial;
  MFTLRemeshCompactObject(o, NULL);
  MFTL_RemeshParameters p;
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, .24, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180);
  MFTL_RemeshStats st;
  ok &= MFTLRemeshEnsemble(e, &p, &st) == MFTL_SUCCESS;
  ok &= st.n_collapse > 0 && fabs(mass(e) - initial) < 1e-12 * initial && physics_valid(e);
  ok &= inv_mesh_is_sound(o, o->domain->rank);
  /* Explicit [owned][ghost][created] fixture: new entities must move BEFORE
   * surviving ghosts. No MPI is involved in this local permutation test. */
  set_tags(o, 0, false);
  const int nv = o->verts.n, nt = o->tris.n;
  o->verts.n_local = nv / 2;
  o->tris.n_local = nt / 2;
  for (int v = 0; v < nv; ++v) o->verts.rec[v].owner_proc = v < nv / 2 ? 0 : 1;
  for (int j = 0; j < nt; ++j) o->tris.rec[j].owner_proc = j < nt / 2 ? 0 : 1;
  /* An owned entity beyond the old prefix represents a new append. */
  o->verts.rec[nv - 1].owner_proc = 0;
  o->tris.rec[nt - 1].owner_proc = 0;
  MFTLRemeshCompactObject(o, NULL);
  ok &= o->verts.n_local == nv / 2 + 1 && o->tris.n_local == nt / 2 + 1;
  ok &= check_tags(o, 0, false);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

static int test_scratch_reconstruction_and_wire_cost(void)
{
  int ok = 1;
  MFTL_RedistributeStats stats[3];
  for (int mode = 0; mode < 3; ++mode)
  {
    MFTL_Ensemble *e = make_ens(MPI_COMM_WORLD, false);
    if (mode)
    {
      MFTL_FieldMode m = mode == 1 ? MFTL_FIELD_RECOMPUTED : MFTL_FIELD_INTERPOLATED;
      ok &= MFTLEnsembleRegisterField(e, "v", MFTL_FIELD_VERTEX, 7, m) == 0;
      ok &= MFTLEnsembleRegisterField(e, "t", MFTL_FIELD_TRIANGLE, 9, m) == 1;
    }
    sphere(e, 0, .6, 2);
    MFTLRedistributeStatsReset();
    ok &= MFTLRedistribute(e) == MFTL_SUCCESS;
    MFTLRedistributeGetStats(&stats[mode]);
    if (mode == 1)
    {
      for (int j = 0; j < e->n_local_objects; ++j)
      {
        MFTL_Object *o = e->local_objects[j];
        for (int i = 0; i < 7 * o->verts.n; ++i) o->field_data[0][i] = 123.0;
        for (int i = 0; i < 9 * o->tris.n; ++i) o->field_data[1][i] = 456.0;
      }
      MFTL_TopologyOptions opt = MFTLTopologyDefaultOptions();
      opt.dx = .12;
      opt.remesh = false;
      int body = 0;
      MFTL_TopologyStats st;
      ok &= MFTLEnsembleReconstructBodies(e, &body, 1, NULL, 0, &opt, &st) == MFTL_SUCCESS;
      ok &= st.n_clusters > 0;
      long rows = 0, total;
      for (int j = 0; j < e->n_local_objects; ++j)
      {
        MFTL_Object *o = e->local_objects[j];
        ok &= o->fields == &e->fields && MFTLObjectPoolsAreConsistent(o);
        rows += o->tris.n_local;
        for (int f = 0; f < 2; ++f)
        {
          int n = f ? o->tris.n : o->verts.n, nc = f ? 9 : 7;
          for (int i = 0; i < n * nc; ++i) ok &= o->field_data[f][i] == 0;
        }
      }
      MPI_Allreduce(&rows, &total, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      ok &= total > 0;
      const int ids[2] = {0, 1};
      ok &= MFTLUpdateGhostFields(e, ids, 2, false) == MFTL_SUCCESS;
    }
    MFTLEnsembleDestroy(&e);
  }
  ok &= stats[0].n_bytes_recv == stats[1].n_bytes_recv && stats[0].n_bytes_sent == stats[1].n_bytes_sent;
  ok &= stats[2].n_bytes_recv - stats[0].n_bytes_recv == (long) sizeof(double) * (7 * stats[2].n_verts_received + 9 * stats[2].n_tris_received);
  TEST_ASSERT(field_agreed(ok));
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MFTLRegisterMpiTypes();
  RUN_TEST(test_registry_and_storage);
  RUN_TEST(test_movement_and_ghosts);
  RUN_TEST(test_individual_operators);
  RUN_TEST(test_spanning);
  RUN_TEST(test_dispersed);
  RUN_TEST(test_global_route_and_guards);
  RUN_TEST(test_open_rim_and_compaction);
  RUN_TEST(test_scratch_reconstruction_and_wire_cost);
  if (rank == 0) TEST_REPORT();
  int fail = g_tests_failed;
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return fail ? 1 : 0;
}
