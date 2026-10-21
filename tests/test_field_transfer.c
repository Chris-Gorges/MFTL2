/* Locality, geometric conservation, positivity and independent accuracy checks
 * for the local conservative transfer. Prescribed smoothing positions isolate
 * field transport from the mesh-quality algorithm. */
#include "mftl_core.h"
#include "../src/remeshing/remesh_fields.h"
#include "test_utils.h"

static MFTL_Ensemble *grid(int n, double shift)
{
  int cuts[3] = {1, 1, 1};
  double lo[3] = {-2, -2, -2}, hi[3] = {2, 2, 2};
  bool periodic[3] = {true, true, true};
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_SELF, cuts, lo, hi, periodic, 1);
  MFTLEnsembleRegisterField(e, "density", MFTL_FIELD_TRIANGLE, 2, MFTL_FIELD_CONSERVATIVE_DENSITY);
  MFTLEnsembleRegisterField(e, "sample", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_INTERPOLATED);
  MFTLEnsembleRegisterField(e, "last", MFTL_FIELD_TRIANGLE, 1, MFTL_FIELD_CONSERVATIVE_DENSITY);
  MFTLEnsembleRegisterField(e, "vertex", MFTL_FIELD_VERTEX, 1, MFTL_FIELD_INTERPOLATED);
  MFTL_Object *o = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(e, o);
  const int nv = (n + 1) * (n + 1), nt = 2 * n * n;
  double *p = malloc((size_t) nv * 3 * sizeof(double));
  int *tri = malloc((size_t) nt * 3 * sizeof(int));
  for (int y = 0; y <= n; ++y)
    for (int x = 0; x <= n; ++x)
    {
      const int v = y * (n + 1) + x;
      p[3 * v] = shift - .8 + 1.6 * x / n;
      p[3 * v + 1] = -.8 + 1.6 * y / n;
      p[3 * v + 2] = 0;
      MFTLApplyPeriodicity(&e->domain, p + 3 * v);
    }
  int t = 0;
  for (int y = 0; y < n; ++y)
    for (int x = 0; x < n; ++x)
    {
      const int v = y * (n + 1) + x;
      int ids[6] = {v, v + 1, v + n + 2, v, v + n + 2, v + n + 1};
      memcpy(tri + 3 * t, ids, sizeof(ids));
      t += 2;
    }
  MFTLBuildIndexedMesh(o, p, nv, tri, nt);
  for (int i = 0; i < nt; ++i)
  {
    o->field_data[0][2 * i] = .1 + pow(sin(i + 1.0), 2);
    o->field_data[0][2 * i + 1] = 0.0;
    o->field_data[1][i] = 7.0;
    o->field_data[2][i] = 2.0;
  }
  for (int v = 0; v < nv; ++v) o->field_data[3][v] = v;
  free(p);
  free(tri);
  return e;
}

static double amount(const MFTL_Object *o, int field, int component)
{
  double sum = 0.0;
  const int nc = o->fields->entries[field].ncomp;
  for (int t = 0; t < o->tris.n; ++t)
    if (!(o->tris.rec[t].flags & MFTL_FLAG_DEAD)) sum += MFTLGetTriangleArea(o, t) * o->field_data[field][nc * t + component];
  return sum;
}

static int check_values(const MFTL_Object *o)
{
  int ok = 1;
  for (int t = 0; t < o->tris.n; ++t)
    if (!(o->tris.rec[t].flags & MFTL_FLAG_DEAD))
    {
      ok &= isfinite(o->field_data[0][2 * t]) && o->field_data[0][2 * t] >= 0;
      ok &= o->field_data[0][2 * t + 1] == 0;
      ok &= isfinite(o->field_data[2][t]) && o->field_data[2][t] >= 0;
      ok &= fabs(o->field_data[1][t] - 7) < 1e-12;
    }
  return ok;
}

static int test_edit_patches(void)
{
  int ok = 1;
  for (int operation = 0; operation < 3; ++operation)
  {
    MFTL_Ensemble *e = grid(8, 0);
    MFTL_Object *o = e->local_objects[0];
    const int h = 3 * (2 * (3 * 8 + 3)) + 2;
    const int t0 = MFTLHeTriIdx(h), t1 = MFTLHeTriIdx(o->tris.he[h].twin_idx);
    const int a = MFTLHeOrigin(o, h), b = MFTLHeTarget(o, h), nt = o->tris.n;
    unsigned char *patch = calloc((size_t) nt, 1);
    double *old = malloc((size_t) nt * 2 * sizeof(double));
    memcpy(old, o->field_data[0], (size_t) nt * 2 * sizeof(double));
    const double before = amount(o, 0, 0), parent0 = old[2 * t0] * MFTLGetTriangleArea(o, t0),
                 parent1 = old[2 * t1] * MFTLGetTriangleArea(o, t1);
    for (int t = 0; t < nt; ++t)
    {
      int v[3];
      MFTLGetTriangleVertices(o, t, v);
      patch[t] = t == t0 || t == t1;
      if (operation == 1)
        for (int k = 0; k < 3; ++k) patch[t] |= v[k] == a || v[k] == b;
    }
    if (operation == 0)
    {
      ok &= MFTLSplitEdge(o, h, NULL) >= 0;
      const double after0 = o->field_data[0][2 * t0] * MFTLGetTriangleArea(o, t0) + o->field_data[0][2 * nt] * MFTLGetTriangleArea(o, nt);
      const double after1 = o->field_data[0][2 * t1] * MFTLGetTriangleArea(o, t1) + o->field_data[0][2 * (nt + 1)] * MFTLGetTriangleArea(o, nt + 1);
      ok &= fabs(after0 - parent0) < 1e-13 && fabs(after1 - parent1) < 1e-13;
    }
    else if (operation == 1)
    {
      MFTL_RemeshParameters params;
      MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, .4, 1, MFTL_SMOOTH_TRIANGLE_BASED, 180);
      ok &= MFTLCollapseApproved(o, h, &params, NULL);
      ok &= MFTLCollapseEdge(o, h, &params) == MFTL_SUCCESS;
    }
    else
      ok &= MFTLFlipEdge(o, h) == MFTL_SUCCESS;
    ok &= fabs(amount(o, 0, 0) - before) < 1e-12 * before && check_values(o);
    for (int t = 0; t < nt; ++t)
      if (!patch[t]) ok &= memcmp(old + 2 * t, o->field_data[0] + 2 * t, 2 * sizeof(double)) == 0;
    free(patch);
    free(old);
    MFTLEnsembleDestroy(&e);
  }
  TEST_ASSERT(ok);
  return 0;
}

static int test_smoothing_geometry_and_locality(void)
{
  MFTL_Ensemble *e = grid(8, 0);
  MFTL_Object *o = e->local_objects[0];
  int ok = 1;
  const int v = 4 * 9 + 4;
  double *next = malloc((size_t) o->verts.n * 3 * sizeof(double));
  memcpy(next, o->verts.pos, (size_t) o->verts.n * 3 * sizeof(double));
  next[3 * v] += .04;
  next[3 * v + 1] -= .025;
  const double before = amount(o, 0, 0);
  double *old = malloc((size_t) o->tris.n * 2 * sizeof(double));
  memcpy(old, o->field_data[0], (size_t) o->tris.n * 2 * sizeof(double));
  ok &= MFTLRemeshFieldSmooth(o, next);
  memcpy(o->verts.pos, next, (size_t) o->verts.n * 3 * sizeof(double));
  ok &= fabs(amount(o, 0, 0) - before) < 1e-12 * before && check_values(o);
  int changes = 0;
  for (int t = 0; t < o->tris.n; ++t)
  {
    int corners[3];
    MFTLGetTriangleVertices(o, t, corners);
    bool incident = corners[0] == v || corners[1] == v || corners[2] == v;
    const int same = memcmp(old + 2 * t, o->field_data[0] + 2 * t, 2 * sizeof(double)) == 0;
    if (incident) changes += !same;
    else ok &= same;
    /* A spatially constant field stays constant on this fixed planar domain.
     * Keeping each triangle's old mass fails this check. */
    ok &= fabs(o->field_data[2][t] - 2.0) < 2e-13;
  }
  ok &= changes > 0;
  free(old);
  free(next);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(ok);
  return 0;
}

static int test_transaction_and_limiter(void)
{
  int ok = 1;
  MFTL_Ensemble *e = grid(4, 0);
  MFTL_Object *o = e->local_objects[0];
  const int h = 3 * (2 * (1 * 4 + 1)) + 2, bad = MFTLHeTriIdx(h);
  const int nv = o->verts.n, nt = o->tris.n;
  double *old = malloc((size_t) nt * 2 * sizeof(double));
  memcpy(old, o->field_data[0], (size_t) nt * 2 * sizeof(double));
  o->field_data[2][bad] = NAN; /* Invalid later field: earlier ones must not commit. */
  ok &= MFTLSplitEdge(o, h, NULL) == MFTL_ERROR && o->verts.n == nv && o->tris.n == nt;
  ok &= memcmp(old, o->field_data[0], (size_t) nt * 2 * sizeof(double)) == 0;
  ok &= MFTLFlipEdge(o, h) == MFTL_ERROR;
  ok &= memcmp(old, o->field_data[0], (size_t) nt * 2 * sizeof(double)) == 0;
  double *next = malloc((size_t) nv * 3 * sizeof(double));
  memcpy(next, o->verts.pos, (size_t) nv * 3 * sizeof(double));
  next[3 * (2 * 5 + 2)] += .02;
  ok &= !MFTLRemeshFieldSmooth(o, next);
  ok &= memcmp(old, o->field_data[0], (size_t) nt * 2 * sizeof(double)) == 0;
  for (int v = 0; v < nv; ++v) ok &= o->field_data[3][v] == v;
  o->field_data[2][bad] = 2;
  o->tris.rec[bad].flags |= MFTL_FLAG_FROZEN;
  ok &= !MFTLRemeshFieldSmooth(o, next);
  o->tris.rec[bad].flags &= ~MFTL_FLAG_FROZEN;
  next[3 * (2 * 5 + 2)] += 1.0; /* Fold an incident triangle: reject before writes. */
  ok &= !MFTLRemeshFieldSmooth(o, next);
  ok &= memcmp(old, o->field_data[0], (size_t) nt * 2 * sizeof(double)) == 0;
  free(old);
  free(next);
  MFTLEnsembleDestroy(&e);

  /* Translate a two-triangle patch far enough to request more internal flux
   * than one triangle owns. The rim supplies no material. */
  e = grid(1, 0);
  o = e->local_objects[0];
  for (int t = 0; t < 2; ++t) o->field_data[0][2 * t] = 1;
  const double before = amount(o, 0, 0);
  double translated[12];
  memcpy(translated, o->verts.pos, sizeof(translated));
  for (int v = 0; v < 4; ++v) translated[3 * v] += 1.4;
  ok &= MFTLRemeshFieldSmooth(o, translated);
  memcpy(o->verts.pos, translated, sizeof(translated));
  ok &= fabs(amount(o, 0, 0) - before) < 1e-12 * before && check_values(o);
  ok &= fmin(o->field_data[0][0], o->field_data[0][2]) == 0;
  ok &= fabs(fmax(o->field_data[0][0], o->field_data[0][2]) - 2) < 1e-12;
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(ok);
  return 0;
}

static int test_local_fallbacks(void)
{
  int ok = 1;
  MFTL_Ensemble *e = grid(1, 0);
  MFTL_Object *o = e->local_objects[0];
  /* Both replacement centroids lie in the zero-valued old triangle, while a
   * thin old triangle holds material. A zero predictor must not lose it. */
  const double sliver[12] = {0, 0, 0, 1, 0, 0, 0, .01, 0, 1, 1, 0};
  memcpy(o->verts.pos, sliver, sizeof(sliver));
  o->field_data[0][0] = 0;
  o->field_data[0][2] = 1;
  const double before = amount(o, 0, 0);
  ok &= MFTLFlipEdge(o, 2) == MFTL_SUCCESS;
  ok &= fabs(amount(o, 0, 0) - before) < 1e-13 && check_values(o);
  ok &= o->field_data[0][0] > 0 && o->field_data[0][2] > 0;
  MFTLEnsembleDestroy(&e);

  e = grid(1, 0);
  o = e->local_objects[0];
  /* Opposing face normals do not define an edge tangent frame. Retain each
   * triangle's material through the whole sweep, without a partial flux pass. */
  const double folded[12] = {0, 0, 0, 1, 0, 0, .2, .2, 0, 0, 1, 0};
  memcpy(o->verts.pos, folded, sizeof(folded));
  double old_mass[2], next[12];
  for (int t = 0; t < 2; ++t) old_mass[t] = o->field_data[0][2 * t] * MFTLGetTriangleArea(o, t);
  memcpy(next, folded, sizeof(next));
  next[0] += .01;
  ok &= MFTLRemeshFieldSmooth(o, next);
  memcpy(o->verts.pos, next, sizeof(next));
  for (int t = 0; t < 2; ++t) ok &= fabs(o->field_data[0][2 * t] * MFTLGetTriangleArea(o, t) - old_mass[t]) < 1e-13;
  ok &= check_values(o);
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(ok);
  return 0;
}

static int test_periodic_equivalence(void)
{
  int ok = 1;
  MFTL_Ensemble *e[2] = {grid(8, 0), grid(8, 1.8)};
  for (int i = 0; i < 2; ++i)
  {
    MFTL_Object *o = e[i]->local_objects[0];
    double *next = malloc((size_t) o->verts.n * 3 * sizeof(double));
    memcpy(next, o->verts.pos, (size_t) o->verts.n * 3 * sizeof(double));
    next[3 * (4 * 9 + 5)] += .03;
    next[3 * (4 * 9 + 5) + 1] -= .01;
    for (int v = 0; v < o->verts.n; ++v) MFTLApplyPeriodicity(&e[i]->domain, next + 3 * v);
    ok &= MFTLRemeshFieldSmooth(o, next);
    memcpy(o->verts.pos, next, (size_t) o->verts.n * 3 * sizeof(double));
    MFTLObjectInvalidateDerived(o);
    free(next);
    ok &= MFTLFlipEdge(o, 3 * (2 * (3 * 8 + 4)) + 2) == MFTL_SUCCESS;
    ok &= MFTLSplitEdge(o, 3 * (2 * (5 * 8 + 4)) + 2, NULL) >= 0;
  }
  MFTL_Object *a = e[0]->local_objects[0], *b = e[1]->local_objects[0];
  ok &= a->tris.n == b->tris.n;
  for (int t = 0; t < a->tris.n; ++t) ok &= fabs(a->field_data[0][2 * t] - b->field_data[0][2 * t]) < 1e-12;
  MFTLEnsembleDestroy(&e[0]);
  MFTLEnsembleDestroy(&e[1]);
  TEST_ASSERT(ok);
  return 0;
}

static int test_planar_accuracy(void)
{
  int ok = 1;
  const char *names[4] = {"smooth", "flip", "split", "collapse"};
  for (int operation = 0; operation < 4; ++operation)
  {
    double previous = 0;
    for (int n = 8; n <= 32; n *= 2)
    {
      MFTL_Ensemble *e = grid(n, 0);
      MFTL_Object *o = e->local_objects[0];
      for (int t = 0; t < o->tris.n; ++t)
      {
        double p[3][3];
        MFTLGetTriangleCorners(o, t, p);
        /* Centroid evaluation is the exact cell average of this linear field. */
        o->field_data[0][2 * t] = 2 + (p[0][0] + p[1][0] + p[2][0]) / 3 + .3 * (p[0][1] + p[1][1] + p[2][1]) / 3;
      }
      const double before = amount(o, 0, 0);
      if (operation == 0)
      {
        double *next = malloc((size_t) o->verts.n * 3 * sizeof(double));
        memcpy(next, o->verts.pos, (size_t) o->verts.n * 3 * sizeof(double));
        for (int y = 1; y < n; ++y)
          for (int x = 1; x < n; ++x)
          {
            const int v = y * (n + 1) + x;
            next[3 * v] += .13 * (1.6 / n) * (x % 2 ? -1 : 1);
            next[3 * v + 1] += .07 * (1.6 / n) * (y % 2 ? -1 : 1);
          }
        ok &= MFTLRemeshFieldSmooth(o, next);
        memcpy(o->verts.pos, next, (size_t) o->verts.n * 3 * sizeof(double));
        free(next);
      }
      else
      {
        MFTL_RemeshParameters params;
        MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 3.2 / n, 1, MFTL_SMOOTH_TRIANGLE_BASED, 180);
        int count = 0, step = operation == 3 ? 3 : 1, start = operation == 3 ? 1 : 0;
        for (int y = start; y < n - start; y += step)
          for (int x = start; x < n - start; x += step)
          {
            int h = 6 * (y * n + x) + 2;
            if (operation == 1) count += MFTLFlipEdge(o, h) == MFTL_SUCCESS;
            else if (operation == 2) count += MFTLSplitEdge(o, h, NULL) >= 0;
            else if (MFTLCollapseApproved(o, h, &params, NULL)) count += MFTLCollapseEdge(o, h, &params) == MFTL_SUCCESS;
          }
        ok &= count > 0;
      }
      double error = 0;
      for (int t = 0; t < o->tris.n; ++t)
      {
        if (o->tris.rec[t].flags & MFTL_FLAG_DEAD) continue;
        double p[3][3];
        MFTLGetTriangleCorners(o, t, p);
        const double exact = 2 + (p[0][0] + p[1][0] + p[2][0]) / 3 + .3 * (p[0][1] + p[1][1] + p[2][1]) / 3;
        error += MFTLGetTriangleArea(o, t) * fabs(o->field_data[0][2 * t] - exact);
        ok &= fabs(o->field_data[2][t] - 2) < 1e-12;
      }
      error /= before;
      printf("\n  planar %s N=%d relative L1=%.9g", names[operation], n, error);
      /* Collapse uses separated interior patches: their fraction of the domain
       * grows between N=8 and N=16, so whole-domain error decreases more slowly. */
      if (previous) ok &= previous / error > (operation == 3 ? 1.2 : 1.7);
      previous = error;
      ok &= fabs(amount(o, 0, 0) - before) < 1e-12 * before && check_values(o);
      MFTLEnsembleDestroy(&e);
    }
  }
  TEST_ASSERT(ok);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  RUN_TEST(test_edit_patches);
  RUN_TEST(test_smoothing_geometry_and_locality);
  RUN_TEST(test_transaction_and_limiter);
  RUN_TEST(test_local_fallbacks);
  RUN_TEST(test_periodic_equivalence);
  RUN_TEST(test_planar_accuracy);
  TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return g_tests_failed ? 1 : 0;
}
