/*
 * Triangle-triangle intersection through the LBVH pair traversal on two hand-built
 * single-triangle objects (exactly one hit, triangle 0 of each).
 */
#include "mftl_core.h"
#include "mftl_bvh.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int intersection_count = 0;

void test_intersection_cb(MFTL_Object* objA, int triA_id, MFTL_Object* objB, int triB_id, void* user_data)
{
  intersection_count++;
  printf("Hit: A[%lld] intersects B[%lld]\n", (long long) triA_id, (long long) triB_id);

  // We expect exactly triangle 0 of A and triangle 0 of B to intersect.
  assert(triA_id == 0);
  assert(triB_id == 0);
}

int main(int argc, char** argv)
{
  (void) argc;
  (void) argv;

  printf("Running Intersection unit test...\n");

  // We only need to populate vertices, flat_tri_verts, and n_triangles
  // because the BVH generator does not depend on the complex half-edge topology.
  MFTL_Object* objA = calloc(1, sizeof(MFTL_Object));
  objA->n_triangles = 1;
  objA->vertices = calloc(3, sizeof(MFTL_Vertex));
  objA->flat_tri_verts = calloc(3, sizeof(int));

  // Triangle A0: flat on Z=0
  objA->vertices[0].pos[0] = 0.0;
  objA->vertices[0].pos[1] = 0.0;
  objA->vertices[0].pos[2] = 0.0;
  objA->vertices[1].pos[0] = 2.0;
  objA->vertices[1].pos[1] = 0.0;
  objA->vertices[1].pos[2] = 0.0;
  objA->vertices[2].pos[0] = 0.0;
  objA->vertices[2].pos[1] = 2.0;
  objA->vertices[2].pos[2] = 0.0;
  objA->flat_tri_verts[0] = 0;
  objA->flat_tri_verts[1] = 1;
  objA->flat_tri_verts[2] = 2;

  MFTL_Object* objB = calloc(1, sizeof(MFTL_Object));
  objB->n_triangles = 2;
  objB->vertices = calloc(6, sizeof(MFTL_Vertex));
  objB->flat_tri_verts = calloc(6, sizeof(int));

  // Triangle B0: intersects A0 (pierces it vertically at x=0.5, y=0.5)
  objB->vertices[0].pos[0] = 0.5;
  objB->vertices[0].pos[1] = 0.5;
  objB->vertices[0].pos[2] = -1.0;
  objB->vertices[1].pos[0] = 0.5;
  objB->vertices[1].pos[1] = 0.5;
  objB->vertices[1].pos[2] = 1.0;
  objB->vertices[2].pos[0] = 2.5;
  objB->vertices[2].pos[1] = 0.5;
  objB->vertices[2].pos[2] = 0.0;
  objB->flat_tri_verts[0] = 0;
  objB->flat_tri_verts[1] = 1;
  objB->flat_tri_verts[2] = 2;

  // Triangle B1: Completely disjoint from A0 (way off in space)
  objB->vertices[3].pos[0] = 10.0;
  objB->vertices[3].pos[1] = 10.0;
  objB->vertices[3].pos[2] = 10.0;
  objB->vertices[4].pos[0] = 12.0;
  objB->vertices[4].pos[1] = 10.0;
  objB->vertices[4].pos[2] = 10.0;
  objB->vertices[5].pos[0] = 10.0;
  objB->vertices[5].pos[1] = 12.0;
  objB->vertices[5].pos[2] = 10.0;
  objB->flat_tri_verts[3] = 3;
  objB->flat_tri_verts[4] = 4;
  objB->flat_tri_verts[5] = 5;

  MFTL_BVH* bvhA = MFTL_GenerateLBVH(objA);
  assert(bvhA != NULL);

  MFTL_BVH* bvhB = MFTL_GenerateLBVH(objB);
  assert(bvhB != NULL);

  MFTL_IntersectObjects(objA, bvhA, objB, bvhB, test_intersection_cb, NULL);

  // We expect exactly 1 intersection: A[0] and B[0].
  assert(intersection_count == 1);

  printf("Unit test passed! Exactly 1 intersection found as expected.\n");

  MFTL_DestroyBVH(bvhA);
  MFTL_DestroyBVH(bvhB);
  free(objA->vertices);
  free(objA->flat_tri_verts);
  free(objA);
  free(objB->vertices);
  free(objB->flat_tri_verts);
  free(objB);

  return 0;
}
