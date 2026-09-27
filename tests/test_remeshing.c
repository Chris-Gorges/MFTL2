/*
 * Serial remeshing operators (1 rank): boundary and interior edge splits and collapses on small
 * hand-built meshes, a full remesh pass, and a split/flip/collapse sequence with an opaque
 * per-triangle payload attached (payload lifecycle and bytes).
 */
#include "test_utils.h"
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"
#include "mftl_memory.h"  // For memory macros
#include <mpi.h>
#include <math.h>

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
int AddVertex(MFTL_Object* obj, double x, double y, double z)
{
  MFTL_ENSURE_CAPACITY(obj->vertices, &obj->n_vertices, &obj->capacity_vertices, 1, MFTL_Vertex);
  int idx = obj->n_vertices++;
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
int AddHalfEdge(MFTL_Object* obj)
{
  MFTL_ENSURE_CAPACITY(obj->half_edges, &obj->n_half_edges, &obj->capacity_half_edges, 1, MFTL_HalfEdge);
  int idx = obj->n_half_edges++;
  MFTL_HalfEdge* he = &obj->half_edges[idx];
  he->next_idx = MFTL_ERROR;
  he->twin_idx = MFTL_ERROR;
  he->twin_gid = MFTL_TWIN_NONE;  // Stage 16A: unpaired = BOUNDARY (tag 0); see FinishOpenMesh
  he->boundary = true;
  he->vertex_idx = MFTL_ERROR;
  he->triangle_idx = MFTL_ERROR;
  he->flags = MFTL_FLAG_NONE;
  he->global_id = idx;
  return idx;
}

// Stage 16A: hand-built fans are open bodies; LOCAL pairs get their twin gids and every
// unpaired half-edge is BOUNDARY (tag 0), with the boundary-vertex out-half-edge convention.
static void FinishOpenMesh(MFTL_Object* obj)
{
  MFTLObjectDeclareOpen(obj, 1);
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (he->twin_idx >= 0)
      he->twin_gid = obj->half_edges[he->twin_idx].global_id;
    else
      MFTLHeSetBoundary(he, 0);
  }
  MFTLRefreshBoundaryCache(obj);
}

// -----------------------------------------------------------------------------
// HELPER: Add Triangle
// -----------------------------------------------------------------------------
int AddTriangle(MFTL_Object* obj)
{
  MFTL_ENSURE_CAPACITY(obj->triangles, &obj->n_triangles, &obj->capacity_triangles, 1, MFTL_Triangle);
  int idx = obj->n_triangles++;
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
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    if (obj->vertices[i].flags & MFTL_FLAG_DEAD) continue;

    int out_he = obj->vertices[i].out_he_idx;
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
  for (int i = 0; i < obj->n_half_edges; ++i)
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
      int t_he = obj->triangles[he->triangle_idx].he_idx;
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
  int center_v = AddVertex(obj, 0, 0, 0);

  // Ring Vertices
  int* ring_v = malloc(n_spokes * sizeof(int));
  for (int i = 0; i < n_spokes; ++i)
  {
    double angle = 2.0 * 3.14159 * i / (is_closed ? n_spokes : (n_spokes - 1));  // Semi-circle if open
    if (!is_closed) angle = 3.14159 * i / (n_spokes - 1);  // 0 to 180 degrees
    ring_v[i] = AddVertex(obj, cos(angle), sin(angle), 0);
  }

  // Create Triangles (Fan)
  // Triangle i connects Center -> Ring[i] -> Ring[i+1] -> Center
  int n_tris = is_closed ? n_spokes : (n_spokes - 1);
  int first_spoke_he = MFTL_ERROR;
  int prev_spoke_twin = MFTL_ERROR;

  for (int i = 0; i < n_tris; ++i)
  {
    int next_i = (i + 1) % n_spokes;

    int t = AddTriangle(obj);

    // HEs:
    // he_c_r1: Center -> Ring[i]
    // he_r1_r2: Ring[i] -> Ring[next]
    // he_r2_c: Ring[next] -> Center
    int he_c_r1 = AddHalfEdge(obj);
    int he_r1_r2 = AddHalfEdge(obj);
    int he_r2_c = AddHalfEdge(obj);

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
  FinishOpenMesh(obj);
}

// -----------------------------------------------------------------------------
// HELPER: Create Single Triangle Mesh (Legacy kept for split test)
// -----------------------------------------------------------------------------
void CreateSingleTriangle(MFTL_Object* obj, int v[3], int he[3], int* tri)
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
  FinishOpenMesh(obj);
}

// -----------------------------------------------------------------------------
// TEST CASE: Split Boundary Edge
// -----------------------------------------------------------------------------
int TestSplitBoundary()
{
  MFTL_Ensemble* ens = NULL;
  MFTL_Object* obj = CreateTestObject(&ens);
  int v[3], he[3], tri;
  CreateSingleTriangle(obj, v, he, &tri);

  TEST_VALIDATE(obj);  // Pre-check

  // Split Edge 0 (0->1)
  int initial_v = obj->n_vertices;
  int initial_t = obj->n_triangles;

  int new_v_idx = MFTLSplitEdge(obj, he[0]);

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
  int edge_to_split = obj->vertices[0].out_he_idx;

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

  int initial_v = obj->n_vertices;
  int initial_t = obj->n_triangles;

  // Pick an edge to collapse: Center -> Ring[0]
  // In CreateFullOneRing, edges from Center are created as he_c_r1.
  // The first created half-edge is at index 0? No, AddHalfEdge increments.
  // Safe way: Get out_he from Center.
  int he_collapse = obj->vertices[0].out_he_idx;

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
  // Stage 16A: 5 spokes -> 4 triangles. Collapsing spoke 0-2 of the 4-spoke fan would remove
  // T0, whose two wings are both BOUNDARY (a dangling edge; forbidden by the link condition,
  // plans/16 §2.2 — the collapse gates reject it). Spoke 0-3 of the 5-spoke fan is admissible:
  // T1(0-2-3) and T2(0-3-4) each have one BOUNDARY and one LOCAL wing.
  CreateFullOneRing(obj, 5, false);

  TEST_VALIDATE(obj);

  // Collapse a spoke that is NOT on the boundary first (Internal to the fan checks)
  // Or collapse a boundary spoke?
  // Let's collapse 0->2 (Internal spoke of the fan).
  // 0->2 is shared by T0(0-1-2) and T1(0-2-3).
  // Should remove 2 triangles. 0 merges to 2.

  // Find edge 0->2.
  // It's not guaranteed index. Use geometry or iteration.
  int he_tocal = MFTL_ERROR;
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
      int twin = obj->half_edges[i].twin_idx;
      if (twin != MFTL_ERROR && obj->half_edges[twin].vertex_idx == 3)
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
// Stage 08: full driver pass over a sphere keeps a valid closed mesh.
// -----------------------------------------------------------------------------
int TestFullRemeshPass()
{
  int cuts[3] = {1, 1, 1};
  double xmin[3] = {-2, -2, -2};
  double xmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, per, 1);
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);

  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, 3);
  MFTLComputeTriangleProperties(obj);

  int v_before = obj->n_vertices;
  int t_before = obj->n_triangles;
  double vol_before = fabs(MFTLObjectVolume(obj));

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 0.28, 6, MFTL_SMOOTH_VERTEX_BASED, 30.0);
  MFTLRemeshEnsemble(ens, &params);

  // After the full pass: valid, closed, integrity-passing, live entities only.
  TEST_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  TEST_ASSERT(obj->n_vertices > 0 && obj->n_triangles > 0);
  TEST_ASSERT(obj->n_triangles != t_before || obj->n_vertices != v_before);  // the pass did something

  // No dead entities may remain after finalization.
  for (int i = 0; i < obj->n_vertices; ++i)
    TEST_ASSERT(!(obj->vertices[i].flags & MFTL_FLAG_DEAD));
  for (int i = 0; i < obj->n_triangles; ++i)
    TEST_ASSERT(!(obj->triangles[i].flags & MFTL_FLAG_DEAD));

  double v = fabs(MFTLObjectVolume(obj));
  if (obj->domain->rank == 0) printf("  full pass: V %.6f -> %.6f (drift %.3e)\n", vol_before, v, fabs(v - vol_before) / vol_before);
  TEST_ASSERT(fabs(v - vol_before) / vol_before < 0.05);  // bounded drift pre-Stage-09

  MFTLEnsembleDestroy(&ens);
  return 0;
}

// -----------------------------------------------------------------------------
// Stage 12: opaque per-triangle payloads through the remeshing lifecycle.
// Split reuses one triangle (blob rides along) and creates zeroed-blob
// triangles; flip is in-place (byte-identical); collapse kills triangles
// whose blobs the PATH B compaction in MFTLFinalizeRemeshing must free.
// -----------------------------------------------------------------------------
typedef struct
{
  int gid;
  double scalar;
  double vec[3];
  unsigned char tag[8];
} RemeshTriData;

// Deterministic fill from the triangle global id. The memset pins the struct
// padding bytes so memcmp comparisons are byte-exact.
static void FillRemeshTriData(int gid, RemeshTriData* d)
{
  memset(d, 0, sizeof(RemeshTriData));
  d->gid = gid;
  d->scalar = 0.5 + 1e-3 * gid;
  for (int k = 0; k < 3; ++k) d->vec[k] = sin(0.02 * gid * (k + 1));
  for (int j = 0; j < 8; ++j) d->tag[j] = (unsigned char) (gid * 17 + j);
}

// The four vertices of an interior edge's quad: endpoints (a, b) and the two
// wing tips (c, d). he->vertex_idx is the TARGET vertex of a half-edge.
static void EdgeQuadVertices(MFTL_Object* obj, int he_idx, int* v_a, int* v_b, int* v_c, int* v_d)
{
  int he_bc = obj->half_edges[he_idx].next_idx;
  int he_ca = obj->half_edges[he_bc].next_idx;
  int he_ba = obj->half_edges[he_idx].twin_idx;
  int he_ad = obj->half_edges[he_ba].next_idx;
  *v_a = obj->half_edges[he_ca].vertex_idx;  // he_ca: C->A
  *v_b = obj->half_edges[he_idx].vertex_idx;  // he_ab: A->B
  *v_c = obj->half_edges[he_bc].vertex_idx;  // he_bc: B->C
  *v_d = obj->half_edges[he_ad].vertex_idx;  // he_ad: A->D
}

static void SnapshotRemeshData(MFTL_Object* obj, int** out_gids, RemeshTriData** out_data, int* out_n)
{
  int n = obj->n_triangles;
  int* gids = malloc(n * sizeof(int));
  RemeshTriData* data = malloc(n * sizeof(RemeshTriData));
  for (int i = 0; i < n; ++i)
  {
    gids[i] = obj->triangles[i].global_id;
    memcpy(&data[i], obj->triangles[i].data, sizeof(RemeshTriData));
  }
  *out_gids = gids;
  *out_data = data;
  *out_n = n;
}

// Verify the Stage-12 data policy against a pre-op snapshot: every triangle
// carries a blob; a gid present in the snapshot must be BYTE-IDENTICAL
// (surviving / split-reused / flipped triangles); a gid absent from the
// snapshot (created by splits) must be fully ZERO (documented zero-init
// policy — remeshing does not repartition user data).
static int VerifyRemeshData(MFTL_Object* obj, const int* snap_gids, const RemeshTriData* snap_data, int n_snap, int* out_surviving, int* out_new)
{
  RemeshTriData zero;
  memset(&zero, 0, sizeof(RemeshTriData));

  int fails = 0, n_surv = 0, n_new = 0;
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    MFTL_Triangle* t = &obj->triangles[i];
    if (t->data == MFTL_NULL)
    {
      fails++;
      continue;
    }
    int s = -1;
    for (int j = 0; j < n_snap; ++j)
      if (snap_gids[j] == t->global_id)
      {
        s = j;
        break;
      }
    if (s >= 0)
    {
      if (memcmp(t->data, &snap_data[s], sizeof(RemeshTriData)) != 0) fails++;
      else n_surv++;
    }
    else
    {
      if (memcmp(t->data, &zero, sizeof(RemeshTriData)) != 0) fails++;
      else n_new++;
    }
  }
  *out_surviving = n_surv;
  *out_new = n_new;
  return fails;
}

int TestRemeshingTriangleData()
{
  // np1: ensemble + generator path so obj->domain is valid (remeshing's
  // mftlNewTriangle reads obj->domain->rank).
  int cuts[3] = {1, 1, 1};
  double xmin[3] = {-2, -2, -2};
  double xmax[3] = {2, 2, 2};
  bool per[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, per, 1);
  MFTL_Object* obj = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ens, obj);

  double center[3] = {0.0, 0.0, 0.0};
  MFTLGenerateSphere(obj, center, 1.0, 2);
  MFTLComputeTriangleProperties(obj);

  TEST_ASSERT(MFTLInitializeObjectData(obj, sizeof(RemeshTriData)) == MFTL_SUCCESS);
  for (int i = 0; i < obj->n_triangles; ++i)
  {
    RemeshTriData d;
    FillRemeshTriData(obj->triangles[i].global_id, &d);
    MFTLSetTriangleData(obj, &obj->triangles[i], &d);
  }

  int* snap_gids = NULL;
  RemeshTriData* snap_data = NULL;
  int n_snap = 0;
  SnapshotRemeshData(obj, &snap_gids, &snap_data, &n_snap);

  int start_v = obj->n_vertices;
  int start_t = obj->n_triangles;
  int start_h = obj->n_half_edges;

  // Track the vertices each op touches so the next op works on an untouched,
  // purely original neighborhood (the raw operators have no gates).
  int v_mark_cap = obj->n_vertices + 16;  // the splits below add exactly 3
  char* vtouched = calloc(v_mark_cap, sizeof(char));
  int n_h0 = obj->n_half_edges;  // splits only APPEND; original indices stay valid

  // --- 1. Several splits on disjoint interior edges ---
  int n_splits = 0;
  for (int he = 0; he < n_h0 && n_splits < 3; ++he)
  {
    MFTL_HalfEdge* e = &obj->half_edges[he];
    if (e->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    if (e->twin_idx == MFTL_ERROR) continue;
    if (obj->half_edges[e->twin_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    int v_a, v_b, v_c, v_d;
    EdgeQuadVertices(obj, he, &v_a, &v_b, &v_c, &v_d);
    if (vtouched[v_a] || vtouched[v_b] || vtouched[v_c] || vtouched[v_d]) continue;
    vtouched[v_a] = vtouched[v_b] = vtouched[v_c] = vtouched[v_d] = 1;
    int v_new = MFTLSplitEdge(obj, he);
    TEST_ASSERT(v_new >= 0 && v_new < v_mark_cap);
    vtouched[v_new] = 1;
    n_splits++;
  }
  TEST_ASSERT_EQUAL_INT(3, n_splits);

  // --- 2. One in-place flip (keeps both triangle gids -> byte-identical) ---
  int flipped = 0;
  for (int he = 0; he < n_h0 && !flipped; ++he)
  {
    MFTL_HalfEdge* e = &obj->half_edges[he];
    if (e->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    if (e->twin_idx == MFTL_ERROR) continue;
    if (obj->half_edges[e->twin_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    int v_a, v_b, v_c, v_d;
    EdgeQuadVertices(obj, he, &v_a, &v_b, &v_c, &v_d);
    if (vtouched[v_a] || vtouched[v_b] || vtouched[v_c] || vtouched[v_d]) continue;

    // Same safety gates as mftlFlipApproved: valence must stay >= 3 post-flip and
    // C-D must not already be connected (non-manifold otherwise).
    MFTL_Vertex* ring[32];
    int val_a = MFTLGetVertexOneRingVertices(obj, &obj->vertices[v_a], ring, 32, MFTL_FALSE);
    int val_b = MFTLGetVertexOneRingVertices(obj, &obj->vertices[v_b], ring, 32, MFTL_FALSE);
    if (val_a <= 3 || val_b <= 3) continue;
    int val_c = MFTLGetVertexOneRingVertices(obj, &obj->vertices[v_c], ring, 32, MFTL_FALSE);
    int cd_connected = 0;
    for (int k = 0; k < val_c; ++k)
      if (ring[k] == &obj->vertices[v_d])
      {
        cd_connected = 1;
        break;
      }
    if (cd_connected) continue;

    MFTLFlipEdge(obj, he);
    vtouched[v_a] = vtouched[v_b] = vtouched[v_c] = vtouched[v_d] = 1;
    flipped = 1;
  }
  TEST_ASSERT_TRUE(flipped);

  // --- 3. One collapse (kills 2 triangles -> PATH B must free their blobs) ---
  int collapsed = 0;
  for (int he = 0; he < n_h0 && !collapsed; ++he)
  {
    MFTL_HalfEdge* e = &obj->half_edges[he];
    if (e->flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    if (e->twin_idx == MFTL_ERROR) continue;
    if (obj->half_edges[e->twin_idx].flags & (MFTL_FLAG_DEAD | MFTL_FLAG_VISITED)) continue;
    int v_a, v_b, v_c, v_d;
    EdgeQuadVertices(obj, he, &v_a, &v_b, &v_c, &v_d);
    if (vtouched[v_a] || vtouched[v_b] || vtouched[v_c] || vtouched[v_d]) continue;

    int dead_before = 0;
    for (int i = 0; i < obj->n_triangles; ++i)
      if (obj->triangles[i].flags & MFTL_FLAG_DEAD) dead_before++;

    MFTLCollapseEdge(obj, he);  // raw op: aborts cleanly only on dead entities

    int dead_after = 0;
    for (int i = 0; i < obj->n_triangles; ++i)
      if (obj->triangles[i].flags & MFTL_FLAG_DEAD) dead_after++;

    if (dead_after != dead_before + 2) continue;  // refused — try another edge
    collapsed = 1;
  }
  TEST_ASSERT_TRUE(collapsed);

  // --- 4. Finalize with PATH B compaction (frees the dead blobs) ---
  MFTLFinalizeRemeshing(obj, start_v, start_t, start_h, MFTL_TRUE);
  TEST_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  TEST_ASSERT(obj->n_triangles > 0);

  int n_surv = 0, n_new = 0;
  int fails = VerifyRemeshData(obj, snap_gids, snap_data, n_snap, &n_surv, &n_new);
  printf("  explicit ops: surviving=%d new(zeroed)=%d fails=%d\n", n_surv, n_new, fails);
  TEST_ASSERT_EQUAL_INT(0, fails);
  TEST_ASSERT(n_surv > 0);                        // non-vacuous: split-reused + flipped + untouched
  TEST_ASSERT_EQUAL_INT(2 * n_splits, n_new);     // each interior split creates exactly 2

  free(snap_gids);
  free(snap_data);
  free(vtouched);

  // Clear the VISITED flags the explicit ops set; the driver gates on them.
  for (int j = 0; j < obj->n_half_edges; ++j) obj->half_edges[j].flags &= ~MFTL_FLAG_VISITED;
  for (int j = 0; j < obj->n_vertices; ++j) obj->vertices[j].flags &= ~MFTL_FLAG_VISITED;

  // --- 5. Full serial driver pass with data attached; same invariants ---
  SnapshotRemeshData(obj, &snap_gids, &snap_data, &n_snap);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, 0.28, 2, MFTL_SMOOTH_VERTEX_BASED, 30.0);
  MFTLRemeshEnsemble(ens, &params);

  TEST_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  TEST_ASSERT(obj->n_triangles > 0);

  fails = VerifyRemeshData(obj, snap_gids, snap_data, n_snap, &n_surv, &n_new);
  printf("  full pass:    surviving=%d new(zeroed)=%d fails=%d\n", n_surv, n_new, fails);
  TEST_ASSERT_EQUAL_INT(0, fails);

  free(snap_gids);
  free(snap_data);
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
  RUN_TEST(TestFullRemeshPass);
  RUN_TEST(TestRemeshingTriangleData);

  TEST_REPORT();
  MPI_Finalize();
  return (g_tests_failed > 0);
}
