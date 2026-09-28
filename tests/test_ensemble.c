/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Unit tests for the ensemble, the domain, and the fused per-timestep pass.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"
#include "test_mesh_invariants.h"
#include "test_utils.h"

static MFTL_Ensemble *g_ens = MFTL_NULL;

/* ─────────────────────────────────────────────────────────────────────────────
 * Creation and the domain
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_domain_bounds_are_precomputed_consistently(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  for (int k = 0; k < 3; ++k)
  {
    TEST_ASSERT_EQUAL_DOUBLE(d->global_max[k] - d->global_min[k], d->domain_size[k], 1e-15);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, d->domain_size[k] * d->inv_domain_size[k], 1e-15);
    TEST_ASSERT_EQUAL_DOUBLE((double) d->cuts[k] / d->domain_size[k], d->grid_factors[k], 1e-15);
  }

  TEST_ASSERT(d->comm_world != MPI_COMM_NULL);
  TEST_ASSERT(d->comm_cart != MPI_COMM_NULL);
  TEST_ASSERT(d->comm_graph != MPI_COMM_NULL);
  TEST_ASSERT(d->rank >= 0 && d->rank < d->nproc);
  return 0;
}

static int test_rank_map_is_a_permutation_of_the_ranks(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  int *seen = (int *) calloc((size_t) d->nproc, sizeof(int));
  TEST_ASSERT(seen != MFTL_NULL);

  for (int i = 0; i < d->nproc; ++i)
  {
    const int r = d->rank_map[i];
    TEST_ASSERT(r >= 0 && r < d->nproc);
    TEST_ASSERT_EQUAL_INT(0, seen[r]);  // every rank appears exactly once
    seen[r] = 1;
  }
  free(seen);
  return 0;
}

static int test_neighbour_list_has_no_duplicates(void)
{
  const MFTL_Domain *d = &g_ens->domain;

  /* A distributed graph must not list a neighbour twice. On a thin
   * decomposition with periodic wrap-around several of the 26 offsets map onto
   * the same rank, which is exactly the case the de-duplication exists for. */
  for (int i = 0; i < d->num_neighbors; ++i)
  {
    TEST_ASSERT(d->neighbors[i] >= 0 && d->neighbors[i] < d->nproc);
    for (int j = i + 1; j < d->num_neighbors; ++j) TEST_ASSERT(d->neighbors[i] != d->neighbors[j]);
  }
  return 0;
}

static int test_create_rejects_a_mismatched_process_grid(void)
{
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  double lo[3] = {0, 0, 0};
  double hi[3] = {1, 1, 1};

  // Deliberately wrong: the product cannot equal the communicator size.
  int bad[3] = {nproc + 1, 1, 1};
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, bad, lo, hi, MFTL_NULL, 4);
  TEST_ASSERT(e == MFTL_NULL);

  int zero[3] = {0, 1, 1};
  e = MFTLEnsembleCreate(MPI_COMM_WORLD, zero, lo, hi, MFTL_NULL, 4);
  TEST_ASSERT(e == MFTL_NULL);

  // NULL arguments are rejected without touching MPI.
  TEST_ASSERT(MFTLEnsembleCreate(MPI_COMM_WORLD, MFTL_NULL, lo, hi, MFTL_NULL, 4) == MFTL_NULL);
  return 0;
}

static int test_destroy_is_null_safe(void)
{
  MFTLEnsembleDestroy(MFTL_NULL);
  MFTL_Ensemble *e = MFTL_NULL;
  MFTLEnsembleDestroy(&e);
  TEST_ASSERT(e == MFTL_NULL);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * Objects
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_add_object_links_the_domain(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 1);
  TEST_ASSERT(e != MFTL_NULL);

  MFTL_Object *a = MFTLObjectCreate(100);
  TEST_ASSERT(a->domain == MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLEnsembleAddObject(e, a));
  TEST_ASSERT(a->domain == &e->domain);
  TEST_ASSERT_EQUAL_INT(1, e->n_local_objects);

  // The list grows past its initial capacity of one.
  for (int i = 0; i < 10; ++i) MFTLEnsembleAddObject(e, MFTLObjectCreate(200 + i));
  TEST_ASSERT_EQUAL_INT(11, e->n_local_objects);
  TEST_ASSERT(e->capacity_objects >= 11);

  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLEnsembleAddObject(e, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLEnsembleAddObject(MFTL_NULL, a));

  // Destroying the ensemble destroys every object it owns (checked by ASan/LSan).
  MFTLEnsembleDestroy(&e);
  return 0;
}

/**
 * The ensemble hands out body ids, densely, in add order.
 *
 * This is the identity bodies travel under: redistribution puts `global_id` on
 * the wire and `slot_of_body` turns it back into an index, so an id that was not
 * dense, or not agreed between ranks, merges entities into the wrong body without
 * anything crashing. Whatever the caller passed to MFTLObjectCreate() is
 * provisional and must be overwritten -- checked here with ids that are
 * deliberately neither dense nor ordered nor distinct from each other's
 * positions.
 */
static int test_add_object_hands_out_dense_body_ids(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 1);
  TEST_ASSERT(e != MFTL_NULL);
  TEST_ASSERT_EQUAL_INT(0, e->n_bodies_global);

  const int provisional[5] = {999, -3, 0, 7, 999};
  for (int i = 0; i < 5; ++i)
  {
    MFTL_Object *obj = MFTLObjectCreate(provisional[i]);
    TEST_ASSERT_EQUAL_INT(provisional[i], obj->global_id);  // provisional until it joins
    TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, MFTLEnsembleAddObject(e, obj));
    TEST_ASSERT_EQUAL_INT(i, obj->global_id);  // and overwritten on the way in
  }

  TEST_ASSERT_EQUAL_INT(5, e->n_local_objects);
  TEST_ASSERT_EQUAL_INT(5, e->n_bodies_global);

  /* Dense in list order, which is what makes a gid an array index rather than a
   * lookup -- the property src/communication/ sizes its per-body arrays from. */
  for (int i = 0; i < e->n_local_objects; ++i) TEST_ASSERT_EQUAL_INT(i, e->local_objects[i]->global_id);

  /* A rejected add hands out no id: n_bodies_global must not drift, or two ranks
   * that disagree only about a failed call would disagree about the id space. */
  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, MFTLEnsembleAddObject(e, MFTL_NULL));
  TEST_ASSERT_EQUAL_INT(5, e->n_bodies_global);

  /* An object that never joins an ensemble keeps what it was given -- that is the
   * spatial halo's shadow patches and every serial test fixture. */
  MFTL_Object *loose = MFTLObjectCreate(4242);
  TEST_ASSERT_EQUAL_INT(4242, loose->global_id);
  MFTLObjectDestroy(&loose);

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_global_counts_sum_the_owned_prefix(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 4);

  MFTL_Object *a = MFTLObjectCreate(1);
  MFTL_Object *b = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(e, a);
  MFTLEnsembleAddObject(e, b);

  MFTLObjectSetCounts(a, 20, 12, 30, 18);
  MFTLObjectSetCounts(b, 7, 7, 9, 9);

  MFTLEnsembleUpdateGlobalCounts(e);

  const int np = e->domain.nproc;
  TEST_ASSERT_EQUAL_INT(12 * np, a->global_n_vertices);
  TEST_ASSERT_EQUAL_INT(18 * np, a->global_n_triangles);
  TEST_ASSERT_EQUAL_INT(7 * np, b->global_n_vertices);
  TEST_ASSERT_EQUAL_INT(9 * np, b->global_n_triangles);
  TEST_ASSERT_EQUAL_INT((18 + 9) * np, e->total_global_triangles);

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_global_counts_on_an_empty_ensemble(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 4);
  MFTLEnsembleUpdateGlobalCounts(e);
  TEST_ASSERT_EQUAL_INT(0, e->total_global_triangles);
  MFTLEnsembleDestroy(&e);
  return 0;
}

/* ─────────────────────────────────────────────────────────────────────────────
 * MFTLEnsembleUpdate
 * ─────────────────────────────────────────────────────────────────────────── */

static int test_update_computes_geometry_and_quality(void)
{
  MFTL_Object *obj = MFTLObjectCreate(50);
  MFTLEnsembleAddObject(g_ens, obj);

  /* A unit square split along its diagonal. Every quantity below is exact by
   * hand, which is what makes this a better fixture than a sphere: two right
   * triangles of legs 1, so area 1/2, squared edge lengths {1, 1, 2}, and
   * mean-ratio quality 4*sqrt(3)*(1/2)/4 = sqrt(3)/2. */
  const double pos[12] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
  const int tv[6] = {0, 1, 2, 0, 2, 3};
  MFTLBuildIndexedMesh(obj, pos, 4, tv, 2);

  MFTLEnsembleUpdate(g_ens);

  const MFTL_ObjectQualityStatistics *q = &obj->quality;
  TEST_ASSERT_EQUAL_DOUBLE(0.5, q->min_triangle_area, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.5, q->max_triangle_area, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(0.5, q->avg_triangle_area, 1e-12);

  TEST_ASSERT_EQUAL_DOUBLE(sqrt(3.0) / 2.0, q->min_aspect_ratio, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(sqrt(3.0) / 2.0, q->max_aspect_ratio, 1e-12);

  TEST_ASSERT_EQUAL_DOUBLE(1.0, q->min_edge_length_sq, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(2.0, q->max_edge_length_sq, 1e-12);
  TEST_ASSERT_EQUAL_DOUBLE(4.0 / 3.0, q->avg_edge_length_sq, 1e-12);

  // The same pass writes the geometry arrays.
  for (int t = 0; t < 2; ++t)
  {
    TEST_ASSERT_EQUAL_DOUBLE(0.5, obj->tris.area[t], 1e-12);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, obj->tris.normal[3 * t + 2], 1e-12);
  }

  /* This kernel carries its OWN copy of the normalisation, so the unit-length
   * guarantee has to be asserted against it separately from the geometry module's
   * two kernels. A normal that is short rather than misdirected passes every
   * direction test above. */
  TEST_ASSERT(inv_triangle_normals_are_unit(obj, 1e-12));
  return 0;
}

static int test_update_agrees_with_the_scalar_geometry_path(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){-4, -4, -4}, (double[3]){4, 4, 4}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);
  double c[3] = {0.3, -0.2, 0.1};
  MFTLGenerateSphere(obj, c, 1.25, 3);

  // Reference from the always-valid half-edge path.
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  const int nt = obj->tris.n;
  double *ref = (double *) malloc(7 * (size_t) nt * sizeof(double));
  TEST_ASSERT(ref != MFTL_NULL);
  for (int t = 0; t < nt; ++t)
  {
    ref[7 * t + 0] = obj->tris.area[t];
    for (int k = 0; k < 3; ++k)
    {
      ref[7 * t + 1 + k] = obj->tris.normal[3 * t + k];
      ref[7 * t + 4 + k] = obj->tris.center[3 * t + k];
    }
  }

  // Scribble over it, then let the streaming pass rewrite it.
  for (int t = 0; t < nt; ++t) obj->tris.area[t] = -1.0;
  MFTLEnsembleUpdate(e);

  /* The fast path and the reference path must agree to round-off. If they ever
   * diverge, the fast one is what every consumer is silently using. */
  for (int t = 0; t < nt; ++t)
  {
    TEST_ASSERT_EQUAL_DOUBLE(ref[7 * t + 0], obj->tris.area[t], 1e-14);
    for (int k = 0; k < 3; ++k)
    {
      TEST_ASSERT_EQUAL_DOUBLE(ref[7 * t + 1 + k], obj->tris.normal[3 * t + k], 1e-12);
      TEST_ASSERT_EQUAL_DOUBLE(ref[7 * t + 4 + k], obj->tris.center[3 * t + k], 1e-12);
    }
  }

  free(ref);
  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_update_on_an_object_with_no_triangles(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);
  MFTLObjectBuildDerived(obj);

  MFTLEnsembleUpdate(e);

  /* An object nobody holds any triangles of must not report DBL_MAX as a
   * measurement -- the MIN identity has to be zeroed on the way out. */
  TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->quality.min_triangle_area, 0.0);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->quality.max_triangle_area, 0.0);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, obj->quality.avg_aspect_ratio, 0.0);

  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_update_on_an_empty_ensemble_is_a_noop(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){0, 0, 0}, (double[3]){1, 1, 1}, MFTL_NULL, 2);
  MFTLEnsembleUpdate(e);  // must not deadlock or dereference anything
  MFTLEnsembleDestroy(&e);
  return 0;
}

static int test_update_only_touches_the_owned_prefix(void)
{
  MFTL_Ensemble *e = MFTLEnsembleCreate(MPI_COMM_WORLD, (int[3]){1, 1, 1}, (double[3]){-4, -4, -4}, (double[3]){4, 4, 4}, MFTL_NULL, 2);
  MFTL_Object *obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(e, obj);
  double c[3] = {0, 0, 0};
  MFTLGenerateSphere(obj, c, 1.0, 2);

  // Declare the tail to be ghosts and mark their areas.
  const int n_local = obj->tris.n / 2;
  obj->tris.n_local = n_local;
  for (int t = n_local; t < obj->tris.n; ++t) obj->tris.area[t] = -7.0;

  MFTLEnsembleUpdate(e);

  /* Ghost geometry is deliberately NOT refreshed here: the ghost exchange
   * synchronises vertices, not derived triangle quantities. Anything preceding a
   * redistribution must call MFTLComputeTriangleGeometry(obj, obj->tris.n). */
  for (int t = 0; t < n_local; ++t) TEST_ASSERT(obj->tris.area[t] > 0.0);
  for (int t = n_local; t < obj->tris.n; ++t) TEST_ASSERT_EQUAL_DOUBLE(-7.0, obj->tris.area[t], 0.0);

  MFTLEnsembleDestroy(&e);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-2.0, -3.0, -4.0};
  double hi[3] = {2.0, 5.0, 4.0};
  bool per[3] = {true, false, true};
  g_ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 8);

  printf("--- Running MFTL Ensemble Tests ---\n");

  RUN_TEST(test_domain_bounds_are_precomputed_consistently);
  RUN_TEST(test_rank_map_is_a_permutation_of_the_ranks);
  RUN_TEST(test_neighbour_list_has_no_duplicates);
  RUN_TEST(test_create_rejects_a_mismatched_process_grid);
  RUN_TEST(test_destroy_is_null_safe);
  RUN_TEST(test_add_object_links_the_domain);
  RUN_TEST(test_add_object_hands_out_dense_body_ids);
  RUN_TEST(test_global_counts_sum_the_owned_prefix);
  RUN_TEST(test_global_counts_on_an_empty_ensemble);
  RUN_TEST(test_update_computes_geometry_and_quality);
  RUN_TEST(test_update_agrees_with_the_scalar_geometry_path);
  RUN_TEST(test_update_on_an_object_with_no_triangles);
  RUN_TEST(test_update_on_an_empty_ensemble_is_a_noop);
  RUN_TEST(test_update_only_touches_the_owned_prefix);

  TEST_REPORT();

  MFTLEnsembleDestroy(&g_ens);
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
