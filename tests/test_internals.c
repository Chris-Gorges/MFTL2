/*
 * Unit tests of small internal helpers: radix sort, sorted-array de-duplication, the
 * minimum-image vector, periodic wrapping and edge length.
 */
#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_arrayfunctions.h"
#include "mftl_geometry.h"

// --- Sorting Tests ---
int test_sort_ids()
{
  int data[] = {5, 1, 3, 2, 4};
  MFTLSortIds(data, 5);
  for (int i = 0; i < 5; i++) TEST_ASSERT_EQUAL_INT(i + 1, data[i]);

  // Empty
  MFTLSortIds(NULL, 0);  // Should not crash

  // Reverse
  int rev[] = {3, 2, 1};
  MFTLSortIds(rev, 3);
  TEST_ASSERT_EQUAL_INT(1, rev[0]);
  TEST_ASSERT_EQUAL_INT(2, rev[1]);
  TEST_ASSERT_EQUAL_INT(3, rev[2]);

  return 0;
}

int test_unique_int_array()
{
  int data[] = {1, 1, 2, 2, 3, 3, 3, 4, 10, 10};
  int n = 10;
  // Array must be sorted for UniqueIntArray to work! (Based on doc)
  // Our data is sorted.

  int new_n = MFTLUniqueIntArray(data, n);

  TEST_ASSERT_EQUAL_INT(5, new_n);  // 1, 2, 3, 4, 10
  TEST_ASSERT_EQUAL_INT(1, data[0]);
  TEST_ASSERT_EQUAL_INT(2, data[1]);
  TEST_ASSERT_EQUAL_INT(3, data[2]);
  TEST_ASSERT_EQUAL_INT(4, data[3]);
  TEST_ASSERT_EQUAL_INT(10, data[4]);

  return 0;
}

// --- Geometry Tests ---
int test_min_image_vector()
{
  MFTL_Domain d;
  memset(&d, 0, sizeof(MFTL_Domain));

  // Setup 10x10x10 domain, periodic
  for (int i = 0; i < 3; i++)
  {
    d.domain_size[i] = 10.0;
    d.domain_half_size[i] = 5.0;
    d.is_periodic[i] = true;
  }

  double p1[3] = {1.0, 1.0, 1.0};
  double p2[3] = {9.0, 1.0, 1.0};  // Distance is 8, but periodic dist should be -2
  double vec[3];

  MFTLGetMinimumImageVector(&d, vec, p1, p2);

  // Expecting p2 - p1 = (8, 0, 0)
  // Corrected: 8 > 5 -> 8 - 10 = -2.
  TEST_ASSERT_EQUAL_DOUBLE(-2.0, vec[0], 1e-9);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, vec[1], 1e-9);
  TEST_ASSERT_EQUAL_DOUBLE(0.0, vec[2], 1e-9);

  return 0;
}

int test_apply_periodicity()
{
  MFTL_Domain d;
  memset(&d, 0, sizeof(MFTL_Domain));
  for (int i = 0; i < 3; i++)
  {
    d.global_min[i] = 0.0;
    d.global_max[i] = 10.0;
    d.domain_size[i] = 10.0;
    d.is_periodic[i] = true;
  }

  double p[3] = {11.0, -1.0, 5.0};
  MFTLApplyPeriodicity(&d, p);

  TEST_ASSERT_EQUAL_DOUBLE(1.0, p[0], 1e-9);  // 11 -> 1
  TEST_ASSERT_EQUAL_DOUBLE(9.0, p[1], 1e-9);  // -1 -> 9
  TEST_ASSERT_EQUAL_DOUBLE(5.0, p[2], 1e-9);  // Unchanged

  return 0;
}

// --- Topology Helper Tests ---
int test_edge_length()
{
  MFTL_Object obj;
  memset(&obj, 0, sizeof(MFTL_Object));
  MFTL_Domain d;
  memset(&d, 0, sizeof(MFTL_Domain));

  // Setup Domain
  for (int i = 0; i < 3; i++)
  {
    d.domain_size[i] = 10.0;
    d.domain_half_size[i] = 5.0;
    d.is_periodic[i] = true;
  }
  obj.domain = &d;

  // 3 Vertices used by triangle
  obj.n_vertices = 3;
  obj.vertices = (MFTL_Vertex*) calloc(3, sizeof(MFTL_Vertex));

  // V0=(1,0,0), V1=(9,0,0), V2=(5,5,0)
  obj.vertices[0].pos[0] = 1.0;
  obj.vertices[1].pos[0] = 9.0;
  obj.vertices[2].pos[0] = 5.0;
  obj.vertices[2].pos[1] = 5.0;

  for (int i = 0; i < 3; i++) obj.vertices[i].out_he_idx = i;

  // 3 HalfEdges
  obj.n_half_edges = 3;
  obj.half_edges = (MFTL_HalfEdge*) calloc(3, sizeof(MFTL_HalfEdge));

  // HE0: 0->1
  obj.half_edges[0].vertex_idx = 1;
  obj.half_edges[0].next_idx = 1;  // -> HE1

  // HE1: 1->2
  obj.half_edges[1].vertex_idx = 2;
  obj.half_edges[1].next_idx = 2;  // -> HE2

  // HE2: 2->0
  obj.half_edges[2].vertex_idx = 0;
  obj.half_edges[2].next_idx = 0;  // -> HE0

  // MFTLGetEdgeLength(he0) Origin should be Target of HE2 (0)
  // Target of HE0 is 1.
  // Distance between 1.0 and 9.0 in X (periodic 10.0) is 2.0.

  double len = MFTLGetEdgeLength(&obj, &obj.half_edges[0]);

  free(obj.vertices);
  free(obj.half_edges);

  TEST_ASSERT_EQUAL_DOUBLE(2.0, len, 1e-9);
  return 0;
}

int main()
{
  RUN_TEST(test_sort_ids);
  RUN_TEST(test_unique_int_array);
  RUN_TEST(test_min_image_vector);
  RUN_TEST(test_apply_periodicity);
  RUN_TEST(test_edge_length);

  TEST_REPORT();
  return (g_tests_failed > 0);
}
