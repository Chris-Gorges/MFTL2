#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_memory.h"  // For memory macros
#include <mpi.h>

// Mocking/Reusing allocation helpers if they are static in remeshing.c
// Since we are linking against MFTL, we can't access static functions.
// We will manually allocate or assume MFTLObjectCreate does the job.

// -----------------------------------------------------------------------------
// HELPER: Create a clean object
// -----------------------------------------------------------------------------
MFTL_Object* CreateTestObject(MFTL_Ensemble** out_ens)
{
  // Create dummy ensemble to provide domain
  int cuts[3] = {1, 1, 1};
  double xmin[3] = {0, 0, 0};
  double xmax[3] = {1, 1, 1};
  bool per[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, per, 1);

  MFTL_Object* obj = MFTLObjectCreate(0);
  if (!obj)
  {
    printf("Failed to create object\n");
    exit(1);
  }
  MFTLEnsembleAddObject(ens, obj);
  *out_ens = ens;
  return obj;
}

// -----------------------------------------------------------------------------
// HELPER: Add Vertex
// -----------------------------------------------------------------------------
mftl_int_t AddVertex(MFTL_Object* obj, double x, double y, double z)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);
  mftl_int_t idx = obj->n_vertices++;
  MFTL_Vertex* v = &obj->vertices[idx];
  v->pos[0] = x;
  v->pos[1] = y;
  v->pos[2] = z;
  v->out_he_idx = MFTL_ERROR;
  v->flags = MFTL_FLAG_NONE;
  v->global_id = idx;  // Simple GID
  return idx;
}

// -----------------------------------------------------------------------------
// HELPER: Add HalfEdge
// -----------------------------------------------------------------------------
mftl_int_t AddHalfEdge(MFTL_Object* obj)
{
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 1, MFTL_HalfEdge);
  mftl_int_t idx = obj->n_half_edges++;
  MFTL_HalfEdge* he = &obj->half_edges[idx];
  he->next_idx = MFTL_ERROR;
  he->twin_idx = MFTL_ERROR;
  he->vertex_idx = MFTL_ERROR;
  he->triangle_idx = MFTL_ERROR;
  he->flags = MFTL_FLAG_NONE;
  he->global_id = idx;
  return idx;
}

// -----------------------------------------------------------------------------
// HELPER: Add Triangle
// -----------------------------------------------------------------------------
mftl_int_t AddTriangle(MFTL_Object* obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  mftl_int_t idx = obj->n_triangles++;
  MFTL_Triangle* t = &obj->triangles[idx];
  t->he_idx = MFTL_ERROR;
  t->flags = MFTL_FLAG_NONE;
  t->global_id = idx;
  return idx;
}

// -----------------------------------------------------------------------------
// HELPER: Create Single Triangle Mesh
// -----------------------------------------------------------------------------
// Vertices: 0, 1, 2. Edges: 0->1, 1->2, 2->0.
// -----------------------------------------------------------------------------
// HELPER: Validate Complete Mesh Connectivity
// -----------------------------------------------------------------------------
int ValidateMeshConnectivity(MFTL_Object* obj)
{
  // 1. Check Vertices
  for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
  {
    if (obj->vertices[i].flags & MFTL_FLAG_DEAD) continue;

    mftl_int_t out_he = obj->vertices[i].out_he_idx;
    if (out_he == MFTL_ERROR)
    {
      printf("Vertex %lld (Dead? %d) has out_he == -1\n", (long long) i, (obj->vertices[i].flags & MFTL_FLAG_DEAD) ? 1 : 0);
    }
    TEST_ASSERT_NOT_EQUAL_INT(MFTL_ERROR, out_he);

    // START CHECK from out_he and circle around
    // Ensure all outgoing edges from this vertex actually point to this vertex
    MFTL_HalfEdge* start = &obj->half_edges[out_he];
    MFTL_HalfEdge* curr = start;
    int count = 0;
    bool boundary_hit = false;
    do
    {
      // Check consistency: Origin of curr must be i
      // Origin(he) is Target(Prev(he)). Prev is tough without Prev pointer.
      // But checking if MFTLHeOrigin(obj, curr) == &obj->vertices[i] is valid.
      // Let's use the explicit logic:
      // If I am at vertex V, 'curr' is an outgoing edge V->Dest.
      // The previous edge in that triangle points to V.
      // We can check MFTLHeOrigin logic.

      // Simple check: The edge implicitly starts at 'i'.
      // Its twin (if exists) points to 'i'.
      if (curr->twin_idx != MFTL_ERROR)
      {
        TEST_ASSERT_EQUAL_INT(i, obj->half_edges[curr->twin_idx].vertex_idx);
      }

      // Advance CCW: Twin -> Next
      if (curr->twin_idx != MFTL_ERROR)
      {
        curr = &obj->half_edges[obj->half_edges[curr->twin_idx].next_idx];
      }
      else
      {
        boundary_hit = true;
        break;
      }
      count++;
      if (count > 100)
      {
        printf("Infinite loop at vertex %lld\n", (long long) i);
        TEST_FAIL();
      }

    } while (curr != start);
  }

  // 2. Check Half-Edges
  for (mftl_int_t i = 0; i < obj->n_half_edges; ++i)
  {
    if (obj->half_edges[i].flags & MFTL_FLAG_DEAD) continue;

    MFTL_HalfEdge* he = &obj->half_edges[i];

    // Next must be valid
    TEST_ASSERT_NOT_EQUAL_INT(MFTL_ERROR, he->next_idx);
    TEST_ASSERT(!(obj->half_edges[he->next_idx].flags & MFTL_FLAG_DEAD));

    // Target Vertex must be valid
    TEST_ASSERT_NOT_EQUAL_INT(MFTL_ERROR, he->vertex_idx);
    TEST_ASSERT(!(obj->vertices[he->vertex_idx].flags & MFTL_FLAG_DEAD));

    // Twin consistency
    if (he->twin_idx != MFTL_ERROR)
    {
      TEST_ASSERT_EQUAL_INT(i, obj->half_edges[he->twin_idx].twin_idx);
    }

    // Triangle consistency: if not boundary
    if (he->triangle_idx != MFTL_ERROR)
    {
      TEST_ASSERT(!(obj->triangles[he->triangle_idx].flags & MFTL_FLAG_DEAD));
      // Triangle must point to one of its edges
      mftl_int_t t_he = obj->triangles[he->triangle_idx].he_idx;
      // It doesn't have to be THIS edge, but it must be in the loop
      MFTL_HalfEdge* t_start = &obj->half_edges[t_he];
      MFTL_HalfEdge* t_curr = t_start;
      bool found = false;
      do
      {
        if (t_curr == he) found = true;
        t_curr = &obj->half_edges[t_curr->next_idx];
      } while (t_curr != t_start);
      TEST_ASSERT_TRUE(found);
    }
  }
  return 0;
}

#define TEST_VALIDATE(obj)                            \
  do                                                  \
  {                                                   \
    if (ValidateMeshConnectivity(obj) != 0) return 1; \
  } while (0)

// -----------------------------------------------------------------------------
// HELPER: Create Full One Ring Mesh (Polygon)
// -----------------------------------------------------------------------------
// Creates a central vertex (0) and N surrounding vertices.
// 'is_closed': if true, connects the last spoke to the first (Internal vertex 0).
//              if false, leaves a gap (Boundary vertex 0).
void CreateFullOneRing(MFTL_Object* obj, int n_spokes, bool is_closed)
{
  // Center
  mftl_int_t center_v = AddVertex(obj, 0, 0, 0);

  // Ring Vertices
  mftl_int_t* ring_v = malloc(n_spokes * sizeof(mftl_int_t));
  for (int i = 0; i < n_spokes; ++i)
  {
    double angle = 2.0 * 3.14159 * i / (is_closed ? n_spokes : (n_spokes - 1));  // Semi-circle if open
    if (!is_closed) angle = 3.14159 * i / (n_spokes - 1);  // 0 to 180 degrees
    ring_v[i] = AddVertex(obj, cos(angle), sin(angle), 0);
  }

  // Create Triangles (Fan)
  // Triangle i connects Center -> Ring[i] -> Ring[i+1] -> Center
  int n_tris = is_closed ? n_spokes : (n_spokes - 1);
  mftl_int_t first_spoke_he = MFTL_ERROR;
  mftl_int_t prev_spoke_twin = MFTL_ERROR;

  for (int i = 0; i < n_tris; ++i)
  {
    int next_i = (i + 1) % n_spokes;

    mftl_int_t t = AddTriangle(obj);

    // HEs:
    // he_c_r1: Center -> Ring[i]
    // he_r1_r2: Ring[i] -> Ring[next]
    // he_r2_c: Ring[next] -> Center
    mftl_int_t he_c_r1 = AddHalfEdge(obj);
    mftl_int_t he_r1_r2 = AddHalfEdge(obj);
    mftl_int_t he_r2_c = AddHalfEdge(obj);

    // Setup Cycle
    obj->half_edges[he_c_r1].vertex_idx = ring_v[i];
    obj->half_edges[he_c_r1].next_idx = he_r1_r2;
    obj->half_edges[he_c_r1].triangle_idx = t;

    obj->half_edges[he_r1_r2].vertex_idx = ring_v[next_i];
    obj->half_edges[he_r1_r2].next_idx = he_r2_c;
    obj->half_edges[he_r1_r2].triangle_idx = t;

    obj->half_edges[he_r2_c].vertex_idx = center_v;
    obj->half_edges[he_r2_c].next_idx = he_c_r1;
    obj->half_edges[he_r2_c].triangle_idx = t;

    obj->triangles[t].he_idx = he_c_r1;

    // Link Twins (Spokes)
    // The spoke he_r2_c (Ring[next]->Center) is the twin of the NEXT triangle's he_c_r1 (Center->Ring[next])
    // The spoke he_c_r1 (Center->Ring[i]) is the twin of the PREV triangle's he_r2_c

    if (prev_spoke_twin != MFTL_ERROR)
    {
      obj->half_edges[he_c_r1].twin_idx = prev_spoke_twin;
      obj->half_edges[prev_spoke_twin].twin_idx = he_c_r1;
    }

    prev_spoke_twin = he_r2_c;  // This will twin with the next triangle's first edge

    // Close the loop if needed
    if (is_closed && i == n_tris - 1)
    {
      // The last return spoke (he_r2_c) twins with the VERY FIRST spoke (he_c_r1 from iter 0)
      obj->half_edges[he_r2_c].twin_idx = first_spoke_he;
      obj->half_edges[first_spoke_he].twin_idx = he_r2_c;
    }

    if (i == 0) first_spoke_he = he_c_r1;

    // Out Pointers (Simple Set)
    obj->vertices[center_v].out_he_idx = he_c_r1;
    obj->vertices[ring_v[i]].out_he_idx = he_r1_r2;
    // For open ring, ensure the last vertex gets an outgoing edge
    if (!is_closed && i == n_tris - 1)
    {
      // The last vertex is ring_v[next_i].
      // It has outgoing edge he_r2_c (Ring[next]->Center).
      obj->vertices[ring_v[next_i]].out_he_idx = he_r2_c;
    }
  }
  // Allow boundary vertices to point to boundary edges if free

  free(ring_v);
}

// -----------------------------------------------------------------------------
// HELPER: Create Single Triangle Mesh (Legacy kept for split test)
// -----------------------------------------------------------------------------
void CreateSingleTriangle(MFTL_Object* obj, mftl_int_t v[3], mftl_int_t he[3], mftl_int_t* tri)
{
  v[0] = AddVertex(obj, 0.0, 0.0, 0.0);
  v[1] = AddVertex(obj, 1.0, 0.0, 0.0);
  v[2] = AddVertex(obj, 0.0, 1.0, 0.0);

  he[0] = AddHalfEdge(obj);  // 0->1
  he[1] = AddHalfEdge(obj);  // 1->2
  he[2] = AddHalfEdge(obj);  // 2->0

  *tri = AddTriangle(obj);

  // Link HE -> Vertex (Target)
  obj->half_edges[he[0]].vertex_idx = v[1];
  obj->half_edges[he[1]].vertex_idx = v[2];
  obj->half_edges[he[2]].vertex_idx = v[0];

  // Link HE -> Next
  obj->half_edges[he[0]].next_idx = he[1];
  obj->half_edges[he[1]].next_idx = he[2];
  obj->half_edges[he[2]].next_idx = he[0];

  // Link HE -> Triangle
  obj->half_edges[he[0]].triangle_idx = *tri;
  obj->half_edges[he[1]].triangle_idx = *tri;
  obj->half_edges[he[2]].triangle_idx = *tri;

  // Link Triangle -> HE
  obj->triangles[*tri].he_idx = he[0];

  // Link Vertex -> Out HE (Origin)
  obj->vertices[v[0]].out_he_idx = he[0];
  obj->vertices[v[1]].out_he_idx = he[1];
  obj->vertices[v[2]].out_he_idx = he[2];
}

// -----------------------------------------------------------------------------
// TEST CASE: Split Boundary Edge
// -----------------------------------------------------------------------------
int TestSplitBoundary()
{
  MFTL_Ensemble* ens = NULL;
  MFTL_Object* obj = CreateTestObject(&ens);
  mftl_int_t v[3], he[3], tri;
  CreateSingleTriangle(obj, v, he, &tri);

  TEST_VALIDATE(obj);  // Pre-check

  // Split Edge 0 (0->1)
  mftl_int_t initial_v = obj->n_vertices;
  mftl_int_t initial_t = obj->n_triangles;

  mftl_int_t new_v_idx = MFTLSplitEdge(obj, he[0]);

  // Checks
  TEST_ASSERT_EQUAL_INT(initial_v + 1, obj->n_vertices);
  TEST_ASSERT_EQUAL_INT(initial_t + 1, obj->n_triangles);  // 1 -> 2

  TEST_VALIDATE(obj);  // Post-check

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// -----------------------------------------------------------------------------
// TEST CASE: Split Internal Edge
// -----------------------------------------------------------------------------
int TestSplitInternal()
{
  MFTL_Ensemble* ens = NULL;
  MFTL_Object* obj = CreateTestObject(&ens);

  // Use Full 1-Ring (4 spokes, Closed) to have nice internal edges
  CreateFullOneRing(obj, 4, true);
  // Center is 0. Ring is 1,2,3,4.
  // Edges from Center are 0->1, 0->2, 0->3, 0->4.
  // All are internal because it's a closed loop around 0.

  TEST_VALIDATE(obj);

  // Split one of the spoke edges, e.g., obj->half_edges[0] which is likely 0->1 (or close to it)
  // Find an edge starting at 0
  mftl_int_t edge_to_split = obj->vertices[0].out_he_idx;

  MFTLSplitEdge(obj, edge_to_split);

  TEST_VALIDATE(obj);

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// -----------------------------------------------------------------------------
// TEST CASE: Collapse Internal Edge (Full Ring)
// -----------------------------------------------------------------------------
int TestCollapseInternal()
{
  MFTL_Ensemble* ens = NULL;
  MFTL_Object* obj = CreateTestObject(&ens);

  // Create Hexagon (6 triangles) around Center(0)
  // This simulates a perfect internal vertex.
  CreateFullOneRing(obj, 6, true);

  TEST_VALIDATE(obj);

  mftl_int_t initial_v = obj->n_vertices;
  mftl_int_t initial_t = obj->n_triangles;

  // Pick an edge to collapse: Center -> Ring[0]
  // In CreateFullOneRing, edges from Center are created as he_c_r1.
  // The first created half-edge is at index 0? No, AddHalfEdge increments.
  // Safe way: Get out_he from Center.
  mftl_int_t he_collapse = obj->vertices[0].out_he_idx;

  // Verify it is a valid spoke
  TEST_ASSERT(obj->half_edges[he_collapse].vertex_idx != 0);  // Target is ring

  // Collapse!
  MFTLCollapseEdge(obj, he_collapse);

  TEST_VALIDATE(obj);

  // Check Counts
  // Collapsing an internal spoke removes 2 triangles, 1 vertex (the center moves to Ring[0]), and 3 edges (the spoke + 2 boundary wings).
  // Wait, Center(A) moves to Ring[0](B).
  // Triangles (Center, Ring[0], Ring[1]) and (Center, Ring[5], Ring[0]) die.
  // 6 Triangles -> 4 Triangles?
  // 2 triangles die.
  int dead_tris = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
    if (obj->triangles[i].flags & MFTL_FLAG_DEAD) dead_tris++;
  TEST_ASSERT_EQUAL_INT(2, dead_tris);

  int dead_verts = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
    if (obj->vertices[i].flags & MFTL_FLAG_DEAD) dead_verts++;
  TEST_ASSERT_EQUAL_INT(1, dead_verts);  // Center is dead

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// -----------------------------------------------------------------------------
// TEST CASE: Collapse Boundary Edge
// -----------------------------------------------------------------------------
int TestCollapseBoundary()
{
  MFTL_Ensemble* ens = NULL;
  MFTL_Object* obj = CreateTestObject(&ens);

  // Create Semi-Circle (4 spokes, Open)
  // Center(0) is on the boundary line (if 180 deg) or just a boundary vertex.
  // Ring: 1, 2, 3, 4.
  // T0: 0-1-2, T1: 0-2-3, T2: 0-3-4.
  // Center 0 is physically shared by all.
  // Boundary edges are: 0->1 and 0->4 (and the ring chain 1-2-3-4).
  CreateFullOneRing(obj, 4, false);  // 4 spokes -> 3 triangles

  TEST_VALIDATE(obj);

  // Collapse a spoke that is NOT on the boundary first (Internal to the fan checks)
  // Or collapse a boundary spoke?
  // Let's collapse 0->2 (Internal spoke of the fan).
  // 0->2 is shared by T0(0-1-2) and T1(0-2-3).
  // Should remove 2 triangles. 0 merges to 2.

  // Find edge 0->2.
  // It's not guaranteed index. Use geometry or iteration.
  mftl_int_t he_tocal = MFTL_ERROR;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    if (obj->half_edges[i].flags & MFTL_FLAG_DEAD) continue;
    // Find edge starting at 0
    // We can't check Origin easily without he->prev.
    // Check Target?
    // If we find edge 2->0, its twin is 0->2.
    // Edge 2->0 is a spoke returning to center.
    // In CreateFullOneRing, Centers are index 0. Vertices 1,2,3...
    // Vertex 2 is the "middle" spoke.
    // Vertices are added 0 (center), then 1,2,3,4.
    // So target is index 2.
    if (obj->half_edges[i].vertex_idx == 0)
    {  // Points to center
      // Origin is ?
      // Actually, let's just use the twin.
      mftl_int_t twin = obj->half_edges[i].twin_idx;
      if (twin != MFTL_ERROR && obj->half_edges[twin].vertex_idx == 2)
      {
        he_tocal = twin;
        break;
      }
    }
  }
  TEST_ASSERT_NOT_EQUAL_INT(MFTL_ERROR, he_tocal);

  MFTLCollapseEdge(obj, he_tocal);

  TEST_VALIDATE(obj);

  // Collapse a true boundary edge? e.g. 0->1.
  // This requires robust boundary handling.
  // T0(0-1-2) dies. 0 merges to 1.
  // T1(0-2-3) updates to (1-2-3).
  // Only 1 triangle dies.
  // Let's verify this separately if needed, but the internal-of-boundary collapse is good for now.

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);

  RUN_TEST(TestSplitBoundary);
  RUN_TEST(TestSplitInternal);
  RUN_TEST(TestCollapseInternal);
  RUN_TEST(TestCollapseBoundary);

  TEST_REPORT();
  MPI_Finalize();
  return 0;
}
