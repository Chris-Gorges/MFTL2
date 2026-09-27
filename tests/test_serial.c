/*
 * Serial topology/geometry basics on a hand-built single triangle: area, and the integrity
 * check on a valid and a corrupted half-edge cycle.
 */

#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_topology.h"
#include "mftl_geometry.h"

// --- Setup Helper ---
void create_single_triangle_mesh(MFTL_Object* obj)
{
  memset(obj, 0, sizeof(MFTL_Object));

  // Allocate
  obj->n_vertices = 3;
  obj->n_triangles = 1;
  obj->n_half_edges = 3;

  obj->vertices = (MFTL_Vertex*) calloc(3, sizeof(MFTL_Vertex));
  obj->triangles = (MFTL_Triangle*) calloc(1, sizeof(MFTL_Triangle));
  obj->half_edges = (MFTL_HalfEdge*) calloc(3, sizeof(MFTL_HalfEdge));

  // Setup Domain (Mocking non-periodic for simplicity)
  obj->domain = (MFTL_Domain*) calloc(1, sizeof(MFTL_Domain));
  for (int i = 0; i < 3; i++)
  {
    obj->domain->is_periodic[i] = 0;
    obj->domain->domain_size[i] = 100.0;
    obj->domain->domain_half_size[i] = 50.0;
  }

  // Vertices: Right Triangle (0,0), (1,0), (0,1)
  obj->vertices[0].pos[0] = 0.0;
  obj->vertices[0].pos[1] = 0.0;
  obj->vertices[0].pos[2] = 0.0;
  obj->vertices[1].pos[0] = 1.0;
  obj->vertices[1].pos[1] = 0.0;
  obj->vertices[1].pos[2] = 0.0;
  obj->vertices[2].pos[0] = 0.0;
  obj->vertices[2].pos[1] = 1.0;
  obj->vertices[2].pos[2] = 0.0;

  for (int i = 0; i < 3; i++)
  {
    obj->vertices[i].global_id = i;
    obj->vertices[i].out_he_idx = i;
  }

  // HalfEdges (0->1->2->0)
  // HE 0 (0->1)
  obj->half_edges[0].vertex_idx = 1;  // Target
  obj->half_edges[0].next_idx = 1;
  obj->half_edges[0].triangle_idx = 0;
  obj->half_edges[0].twin_idx = -1;  // No twin
  obj->half_edges[0].boundary = MFTL_TRUE;

  // HE 1 (1->2)
  obj->half_edges[1].vertex_idx = 2;  // Target
  obj->half_edges[1].next_idx = 2;
  obj->half_edges[1].triangle_idx = 0;
  obj->half_edges[1].twin_idx = -1;
  obj->half_edges[1].boundary = MFTL_TRUE;

  // HE 2 (2->0)
  obj->half_edges[2].vertex_idx = 0;  // Target
  obj->half_edges[2].next_idx = 0;
  obj->half_edges[2].triangle_idx = 0;
  obj->half_edges[2].twin_idx = -1;
  obj->half_edges[2].boundary = MFTL_TRUE;

  // Triangle
  obj->triangles[0].he_idx = 0;
}

void cleanup_mesh(MFTL_Object* obj)
{
  if (obj->vertices) free(obj->vertices);
  if (obj->triangles) free(obj->triangles);
  if (obj->half_edges) free(obj->half_edges);
  if (obj->domain) free(obj->domain);
}

// --- Math Test ---
int test_triangle_area()
{
  MFTL_Object obj;
  create_single_triangle_mesh(&obj);

  // Calculate Area using Library Function
  double area = MFTLGetTriangleArea(&obj, &obj.triangles[0]);

  cleanup_mesh(&obj);

  // Expected: 0.5 * 1 * 1 = 0.5
  TEST_ASSERT_EQUAL_DOUBLE(0.5, area, 1e-9);
  return 0;
}

// --- Integrity Tests ---
int test_integrity_valid()
{
  MFTL_Object obj;
  create_single_triangle_mesh(&obj);

  int status = MFTLCheckMeshIntegrity(&obj);

  cleanup_mesh(&obj);

  TEST_ASSERT_EQUAL_INT(MFTL_SUCCESS, status);
  return 0;
}

int test_integrity_invalid()
{
  MFTL_Object obj;
  create_single_triangle_mesh(&obj);

  // Break topology
  obj.half_edges[0].next_idx = -1;

  // Silence stderr for expected error?
  // For unit tests, sometimes we want to suppress it, but here it's fine.
  int status = MFTLCheckMeshIntegrity(&obj);

  cleanup_mesh(&obj);

  TEST_ASSERT_EQUAL_INT(MFTL_ERROR, status);
  return 0;
}

int main()
{
  RUN_TEST(test_triangle_area);
  RUN_TEST(test_integrity_valid);
  RUN_TEST(test_integrity_invalid);
  TEST_REPORT();
  return (g_tests_failed > 0);
}
