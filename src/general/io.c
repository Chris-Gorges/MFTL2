/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:39:02
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 15:57:58
 */

#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"
#include "khash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Edge map for the GTS writer: undirected edge (min<<32 | max) -> 1-based edge id.
KHASH_INIT(mftl_gts_edge_map, khint64_t, int, 1, kh_int64_hash_func, kh_int64_hash_equal)

/**
 * @brief Writes the object from a single process to a legacy VTK file.
 *
 * This function is for simple, serial visualization. It writes the entire
 * object data held by the calling process to a single .vtk file.
 *
 * @param object The object object to write.
 * @param base_filename The base name for the output file (e.g.,
 * "output/sphere"). .vtk will be appended.
 * @return 0 on success, -1 on error.
 */
int MFTLWriteVtk(const MFTL_Object *object, const char *base_filename)
{
  if (!object)
  {
    fprintf(stderr, "Error: Cannot write a NULL object.\n");
    return MFTL_ERROR;
  }

  char filename[512];
  snprintf(filename, 512, "%s.vtk", base_filename);

  FILE *fp = fopen(filename, "w");
  if (!fp)
  {
    fprintf(stderr, "Error: Could not open file %s for writing.\n", filename);
    return MFTL_ERROR;
  }

  // --- Setup Base Pointers ---
  const MFTL_Vertex *const v_base = object->vertices;
  const MFTL_HalfEdge *const he_base = object->half_edges;
  const MFTL_Triangle *const t_base = object->triangles;

  // --- VTK Header ---
  fprintf(fp, "# vtk DataFile Version 2.0\n");
  fprintf(fp, "MFTL Generated object\n");
  fprintf(fp, "ASCII\n");
  fprintf(fp, "DATASET POLYDATA\n");

  // --- Points (Vertices) ---
  fprintf(fp, "POINTS %d float\n", object->n_vertices);
  for (size_t i = 0; i < object->n_vertices; ++i)
  {
    fprintf(fp, "%lf %lf %lf\n", v_base[i].pos[0], v_base[i].pos[1], v_base[i].pos[2]);
  }

  // --- Polygons (Triangles) ---
  fprintf(fp, "\nPOLYGONS %d %d\n", object->n_triangles, 4 * object->n_triangles);

  for (size_t i = 0; i < object->n_triangles; ++i)
  {
    const MFTL_Triangle *tri = &t_base[i];

    // Navigate Topology via Indices
    int he0_idx = tri->he_idx;
    int he1_idx = he_base[he0_idx].next_idx;
    int he2_idx = he_base[he1_idx].next_idx;

    // Vertices of the triangle:
    int v0_idx = he_base[he2_idx].vertex_idx;
    int v1_idx = he_base[he0_idx].vertex_idx;
    int v2_idx = he_base[he1_idx].vertex_idx;

    // Note: VTK expects 0-based indices into the POINTS list written above.
    // This corresponds exactly to our local indices.
    fprintf(fp, "3 %d %d %d\n", v0_idx, v1_idx, v2_idx);
  }

  fclose(fp);
  printf("Successfully wrote object to %s\n", filename);

  return MFTL_SUCCESS;
}

/**
 * @brief Helper to check if a triangle wraps around the periodic boundary.
 * Calculates the unwrapped coordinates if it does.
 * * @param p0, p1, p2 Pointers to the raw coordinates of the triangle vertices.
 * @param domain The domain object containing periodicity information.
 * @param out_p1, out_p2 Output arrays for the shifted coordinates (p0 stays fixed).
 * @return 1 if the triangle wraps (needs new vertices), 0 if normal.
 */
static int MFTLCheckWrap(MFTL_Domain *domain, double p0[3], double p1[3], double p2[3], double out_p1[3], double out_p2[3])
{
  double v1[3], v2[3];

  // 1. Calculate shortest vector from anchor (p0) to neighbors (p1, p2)
  // This automatically handles the "diff > half_L" logic
  MFTLGetMinimumImageVector(domain, v1, p0, p1);
  MFTLGetMinimumImageVector(domain, v2, p0, p2);

  int wrapped = 0;

  for (int d = 0; d < 3; ++d)
  {
    // 2. Reconstruct absolute coordinates: Position = Anchor + Vector
    out_p1[d] = p0[d] + v1[d];
    out_p2[d] = p0[d] + v2[d];

    // 3. Determine if wrapping occurred
    // If the new coordinate differs from the raw input, a boundary shift happened.
    if (out_p1[d] != p1[d] || out_p2[d] != p2[d])
    {
      wrapped = 1;
    }
  }

  return wrapped;
}

/**
 * @brief Writes ALL MFTL objects to a SINGLE parallel VTK file set (.pvtu + .vtu).
 * * Features:
 * - Aggregates multiple independent meshes (spheres) into one UnstructuredGrid.
 * - Maintains "Smart Periodic Unwrapping" for each object.
 * - Adds "BodyID" data array to distinguish between objects.
 * * @param ensemble The ensemble object containing the objects to write.
 * @param filename_prefix Base path/name for output files.
 */
void MFTLWriteVtu(MFTL_Ensemble *ensemble, const char *filename_prefix)
{
  if (ensemble->num_objects == 0) return;

  int num_objects = ensemble->num_objects;
  MFTL_Domain *domain = &ensemble->domain;
  int my_rank = domain->rank;
  int nproc = domain->nproc;

  // --- PRE-CALCULATION: Count Wrapped Triangles per Object ---
  int *wrapped_counts = (int *) calloc(ensemble->num_objects, sizeof(int));
  double dummy_p1[3], dummy_p2[3];

  int total_file_points = 0;
  int total_file_cells = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    int current_wrapped = 0;

    for (int i = 0; i < obj->n_triangles; ++i)
    {
      int he0 = obj->triangles[i].he_idx;
      int he1 = obj->half_edges[he0].next_idx;
      int he2 = obj->half_edges[he1].next_idx;

      double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(domain, p0, p1, p2, dummy_p1, dummy_p2))
      {
        current_wrapped++;
      }
    }
    wrapped_counts[o] = current_wrapped;

    // Total Points = Original Vertices + (3 * Wrapped Triangles)
    total_file_points += obj->n_vertices + (current_wrapped * 3);
    total_file_cells += obj->n_triangles;
  }

  // --- OPEN FILE ---
  char filename_vtu[1024];
  snprintf(filename_vtu, sizeof(filename_vtu), "%s_%05d.vtu", filename_prefix, my_rank);
  FILE *fp = fopen(filename_vtu, "w");
  if (!fp)
  {
    fprintf(stderr, "[Rank %d] ERROR: Could not open file %s.\n", my_rank, filename_vtu);
    free(wrapped_counts);
    return;
  }

  // --- XML HEADER ---
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <UnstructuredGrid>\n");
  // UPDATED: Use "%d" for counts
  fprintf(fp,
          "    <Piece NumberOfPoints=\""
          "%d"
          "\" NumberOfCells=\""
          "%d"
          "\">\n",
          total_file_points, total_file_cells);

  // ======================================================================
  // 1. POINTS (Coordinates)
  // ======================================================================
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n");

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];

    // A. Original Vertices
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %.10e %.10e %.10e\n", obj->vertices[i].pos[0], obj->vertices[i].pos[1], obj->vertices[i].pos[2]);
    }

    // B. Visual Ghost Vertices (Unwrapped)
    for (int i = 0; i < obj->n_triangles; ++i)
    {
      int he0 = obj->triangles[i].he_idx;
      int he1 = obj->half_edges[he0].next_idx;
      int he2 = obj->half_edges[he1].next_idx;

      double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      double fix_p1[3], fix_p2[3];
      if (MFTLCheckWrap(domain, p0, p1, p2, fix_p1, fix_p2))
      {
        fprintf(fp, "          %.10e %.10e %.10e\n", p0[0], p0[1], p0[2]);
        fprintf(fp, "          %.10e %.10e %.10e\n", fix_p1[0], fix_p1[1], fix_p1[2]);
        fprintf(fp, "          %.10e %.10e %.10e\n", fix_p2[0], fix_p2[1], fix_p2[2]);
      }
    }
  }
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </Points>\n");

  // ======================================================================
  // 2. CELLS (Connectivity)
  // ======================================================================
  fprintf(fp, "      <Cells>\n");
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"connectivity\" format=\"ascii\">\n", "Int32");

  // Global Offset Tracker
  int global_vertex_offset = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    int local_ghost_start_idx = global_vertex_offset + obj->n_vertices;
    int current_ghost_idx = local_ghost_start_idx;

    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      int he0 = obj->triangles[i].he_idx;
      int he1 = obj->half_edges[he0].next_idx;
      int he2 = obj->half_edges[he1].next_idx;

      double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(domain, p0, p1, p2, dummy_p1, dummy_p2))
      {
        fprintf(fp,
                "          "
                "%d"
                " "
                "%d"
                " "
                "%d"
                "\n",
                current_ghost_idx, current_ghost_idx + 1, current_ghost_idx + 2);
        current_ghost_idx += 3;
      }
      else
      {
        fprintf(fp,
                "          "
                "%d"
                " "
                "%d"
                " "
                "%d"
                "\n",
                global_vertex_offset + obj->half_edges[he2].vertex_idx, global_vertex_offset + obj->half_edges[he0].vertex_idx,
                global_vertex_offset + obj->half_edges[he1].vertex_idx);
      }
    }

    global_vertex_offset += obj->n_vertices + (wrapped_counts[o] * 3);
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Offsets ---
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"offsets\" format=\"ascii\">\n", "Int32");
  int running_offset = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    for (int i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      running_offset += 3;
      fprintf(fp,
              "          "
              "%d"
              "\n",
              running_offset);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Types ---
  fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int o = 0; o < num_objects; ++o)
  {
    for (int i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          5\n");
    }
  }
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </Cells>\n");

  // ======================================================================
  // 3. POINT DATA
  // ======================================================================
  fprintf(fp, "      <PointData Scalars=\"VertexOwner\">\n");
  // VertexOwner is usually an MPI rank (int32), keeping as Int32.
  fprintf(fp, "        <DataArray type=\"Int32\" Name=\"VertexOwner\" format=\"ascii\">\n");

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];

    // A. Original
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %d\n", obj->vertices[i].owner_proc);
    }
    // B. Ghosts
    for (int i = 0; i < obj->n_triangles; ++i)
    {
      int he0 = obj->triangles[i].he_idx;
      int he1 = obj->half_edges[he0].next_idx;
      int he2 = obj->half_edges[he1].next_idx;
      double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(domain, p0, p1, p2, dummy_p1, dummy_p2))
      {
        fprintf(fp, "          %d\n", obj->vertices[obj->half_edges[he2].vertex_idx].owner_proc);
        fprintf(fp, "          %d\n", obj->vertices[obj->half_edges[he0].vertex_idx].owner_proc);
        fprintf(fp, "          %d\n", obj->vertices[obj->half_edges[he1].vertex_idx].owner_proc);
      }
    }
  }
  fprintf(fp, "        </DataArray>\n");
  fprintf(fp, "      </PointData>\n");

  // ======================================================================
  // 4. CELL DATA (TriangleOwner & BodyID)
  // ======================================================================
  fprintf(fp, "      <CellData Scalars=\"BodyID\">\n");

  // A. Triangle Owner (Int32 for ranks)
  fprintf(fp, "        <DataArray type=\"Int32\" Name=\"TriangleOwner\" format=\"ascii\">\n");
  for (int o = 0; o < num_objects; ++o)
  {
    for (int i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          %d\n", ensemble->objects[o]->triangles[i].owner_proc);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // B. Body ID (Object Global ID - uses int)
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"BodyID\" format=\"ascii\">\n", "Int32");
  for (int o = 0; o < num_objects; ++o)
  {
    int body_id = (int) ensemble->objects[o]->global_id;
    for (int i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      fprintf(fp,
              "          "
              "%d"
              "\n",
              body_id);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  fprintf(fp, "      </CellData>\n");

  // --- Footer ---
  fprintf(fp, "    </Piece>\n");
  fprintf(fp, "  </UnstructuredGrid>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
  free(wrapped_counts);

  // ======================================================================
  // Step 2: Rank 0 Writes .pvtu
  // ======================================================================
  if (my_rank == 0)
  {
    char filename_pvtu[1024];
    snprintf(filename_pvtu, sizeof(filename_pvtu), "%s.pvtu", filename_prefix);
    fp = fopen(filename_pvtu, "w");
    if (fp)
    {
      fprintf(fp, "<?xml version=\"1.0\"?>\n");
      fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
      fprintf(fp, "  <PUnstructuredGrid GhostLevel=\"0\">\n");

      fprintf(fp, "    <PPoints>\n");
      fprintf(fp, "      <PDataArray type=\"Float64\" NumberOfComponents=\"3\"/>\n");
      fprintf(fp, "    </PPoints>\n");

      fprintf(fp, "    <PPointData Scalars=\"VertexOwner\">\n");
      fprintf(fp, "      <PDataArray type=\"Int32\" Name=\"VertexOwner\"/>\n");
      fprintf(fp, "    </PPointData>\n");

      fprintf(fp, "    <PCellData Scalars=\"BodyID\">\n");
      fprintf(fp, "      <PDataArray type=\"Int32\" Name=\"TriangleOwner\"/>\n");
      fprintf(fp, "      <PDataArray type=\"%s\" Name=\"BodyID\"/>\n", "Int32");
      fprintf(fp, "    </PCellData>\n");

      for (int i = 0; i < nproc; ++i)
      {
        const char *last_slash = strrchr(filename_prefix, '/');
        const char *basename = (last_slash) ? last_slash + 1 : filename_prefix;
        fprintf(fp, "    <Piece Source=\"%s_%05d.vtu\"/>\n", basename, i);
      }
      fprintf(fp, "  </PUnstructuredGrid>\n");
      fprintf(fp, "</VTKFile>\n");
      fclose(fp);
    }
  }
}

/* ========================================================================== */
/* GTS mesh I/O (Stage 05)                                                    */
/*                                                                            */
/* GTS file layout (whitespace-token based; line breaks are irrelevant):      */
/*   nV nE nT GtsSurface GtsFace GtsEdge GtsVertex                            */
/*   nV x  : "x y z"                  (vertex coordinates)                    */
/*   nE x  : "v1 v2"                  (1-based vertex indices, undirected)    */
/*   nT x  : "e1 e2 e3"               (1-based edge indices)                  */
/* ========================================================================== */

// Serializes the three edges of a triangle (given as 0-based edge ids) into a
// vertex triple (v0, v1, v2): v0->v1 is corner edge 0, v1->v2 corner edge 1,
// v2->v0 corner edge 2. Also reports the corner edge ids and whether each
// corner is traversed in the edge's stored direction (dir 0) or reversed (1)
// — the stored orientations are arbitrary, so a global consistency pass
// (2-coloring of the dual graph) orients the whole surface afterwards.
static int mftlGtsResolveTriangle(const int edges[][2], int e1, int e2, int e3, int tri[3], int tri_edge[3], int tri_dir[3])
{
  int e[3] = {e1, e2, e3};
  tri_edge[0] = e1;
  tri_edge[1] = e2;
  tri_edge[2] = e3;

  int v0 = edges[e[0]][0];
  int v1 = edges[e[0]][1];
  tri_dir[0] = 0;  // v0->v1 follows the stored direction by construction

  int v2 = -1;
  int last = -1;
  for (int k = 1; k < 3; ++k)
  {
    int a = edges[e[k]][0];
    int b = edges[e[k]][1];
    if (a == v1)
    {
      v2 = b;
      tri_dir[k] = 0;
    }
    else if (b == v1)
    {
      v2 = a;
      tri_dir[k] = 1;
    }
    else
      last = k;
  }
  if (v2 < 0 || last < 0) return MFTL_ERROR;

  // The remaining edge must connect v2 back to v0; record its direction.
  int a = edges[e[last]][0];
  int b = edges[e[last]][1];
  if (a == v2 && b == v0)
    tri_dir[last] = 0;
  else if (a == v0 && b == v2)
    tri_dir[last] = 1;
  else
    return MFTL_ERROR;

  tri[0] = v0;
  tri[1] = v1;
  tri[2] = v2;
  return MFTL_SUCCESS;
}

int MFTLReadGTS(MFTL_Object* obj, const char* path)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(path != MFTL_NULL);
  MFTL_ASSERT(obj->n_vertices == 0 && "MFTLReadGTS: object must be empty");
  MFTL_Domain* domain = obj->domain;
  int rank = domain->rank;

  int ok = 1;

  double (*positions)[3] = NULL;
  int (*triples)[3] = NULL;
  int nV = 0, nF = 0;

  if (rank == 0)
  {
    FILE* fp = fopen(path, "r");
    if (!fp)
    {
      fprintf(stderr, "MFTLReadGTS: cannot open '%s'\n", path);
      ok = 0;
    }
    else
    {
      int nE = 0;
      char cls0[128], cls1[128], cls2[128], cls3[128];
      if (fscanf(fp, "%d %d %d %127s %127s %127s %127s", &nV, &nE, &nF, cls0, cls1, cls2, cls3) != 7)
      {
        fprintf(stderr, "MFTLReadGTS: malformed header in '%s'\n", path);
        ok = 0;
      }
      else
      {
        positions = malloc((size_t) nV * sizeof(*positions));
        int (*edges)[2] = malloc((size_t) nE * sizeof(*edges));
        triples = malloc((size_t) nF * sizeof(*triples));
        int (*tri_edge)[3] = malloc((size_t) nF * sizeof(*tri_edge));
        int (*tri_dir)[3] = malloc((size_t) nF * sizeof(*tri_dir));
        if (!positions || !edges || !triples || !tri_edge || !tri_dir)
        {
          fprintf(stderr, "MFTLReadGTS: allocation failed\n");
          ok = 0;
        }
        else
        {
          for (int i = 0; i < nV && ok; ++i)
            if (fscanf(fp, "%lf %lf %lf", &positions[i][0], &positions[i][1], &positions[i][2]) != 3)
            {
              fprintf(stderr, "MFTLReadGTS: vertex %d malformed in '%s'\n", i, path);
              ok = 0;
            }

          for (int i = 0; i < nE && ok; ++i)
          {
            int a, b;
            if (fscanf(fp, "%d %d", &a, &b) != 2 || a < 1 || a > nV || b < 1 || b > nV)
            {
              fprintf(stderr, "MFTLReadGTS: edge %d malformed in '%s'\n", i, path);
              ok = 0;
              break;
            }
            edges[i][0] = a - 1;
            edges[i][1] = b - 1;
          }

          for (int t = 0; t < nF && ok; ++t)
          {
            int e1, e2, e3;
            if (fscanf(fp, "%d %d %d", &e1, &e2, &e3) != 3 || e1 < 1 || e1 > nE || e2 < 1 || e2 > nE || e3 < 1 || e3 > nE)
            {
              fprintf(stderr, "MFTLReadGTS: triangle %d malformed in '%s'\n", t, path);
              ok = 0;
              break;
            }
            if (mftlGtsResolveTriangle((const int (*)[2]) edges, e1 - 1, e2 - 1, e3 - 1, triples[t], tri_edge[t], tri_dir[t]) != MFTL_SUCCESS)
            {
              fprintf(stderr, "MFTLReadGTS: triangle %d edges do not form a cycle in '%s'\n", t, path);
              ok = 0;
              break;
            }
          }

          // Every edge of a closed manifold is shared by exactly two triangles.
          int* edge_count = calloc((size_t) nE, sizeof(int));
          int* edge_tri = malloc((size_t) 2 * nE * sizeof(int));
          int* edge_corner = malloc((size_t) 2 * nE * sizeof(int));
          if (!edge_count || !edge_tri || !edge_corner)
          {
            ok = 0;
          }
          else
          {
            for (int t = 0; t < nF && ok; ++t)
              for (int j = 0; j < 3; ++j)
              {
                int e = tri_edge[t][j];
                if (edge_count[e] >= 2)
                {
                  fprintf(stderr, "MFTLReadGTS: edge %d shared by more than 2 triangles in '%s'\n", e, path);
                  ok = 0;
                  break;
                }
                edge_tri[2 * e + edge_count[e]] = t;
                edge_corner[2 * e + edge_count[e]] = j;
                edge_count[e]++;
              }
            for (int e = 0; e < nE && ok; ++e)
              if (edge_count[e] != 2)
              {
                fprintf(stderr, "MFTLReadGTS: edge %d shared by %d triangles (not a closed manifold) in '%s'\n", e, edge_count[e], path);
                ok = 0;
                break;
              }

            // Consistent orientation: 2-color the dual graph so every edge is
            // traversed in opposite directions by its two triangles.
            if (ok)
            {
              char* state = calloc((size_t) nF, sizeof(char));  // 0 unseen, 1 as-is, 2 flipped
              int* queue = malloc((size_t) nF * sizeof(int));
              int qh = 0, qt = 0;
              state[0] = 1;
              queue[qt++] = 0;
              while (qh < qt && ok)
              {
                int t = queue[qh++];
                int tflip = state[t] == 2;
                for (int j = 0; j < 3; ++j)
                {
                  int e = tri_edge[t][j];
                  int eff_dir = tri_dir[t][j] ^ tflip;
                  int slot = edge_tri[2 * e] == t ? 1 : 0;  // the OTHER triangle
                  int u = edge_tri[2 * e + slot];
                  int ucorner = edge_corner[2 * e + slot];
                  int required = 1 - eff_dir;
                  if (state[u] == 0)
                  {
                    state[u] = (tri_dir[u][ucorner] == required) ? 1 : 2;
                    queue[qt++] = u;
                  }
                  else if ((tri_dir[u][ucorner] ^ (state[u] == 2)) != required)
                  {
                    fprintf(stderr, "MFTLReadGTS: non-orientable surface (conflicting winding at edge %d) in '%s'\n", e, path);
                    ok = 0;
                    break;
                  }
                }
              }
              if (ok && qt != nF)
              {
                fprintf(stderr, "MFTLReadGTS: surface is not connected (%d of %d triangles reached) in '%s'\n", qt, nF, path);
                ok = 0;
              }
              if (ok)
                for (int t = 0; t < nF; ++t)
                  if (state[t] == 2)
                  {
                    int tmp = triples[t][1];
                    triples[t][1] = triples[t][2];
                    triples[t][2] = tmp;
                  }
              free(state);
              free(queue);
            }
          }
          free(edge_count);
          free(edge_tri);
          free(edge_corner);
        }
        free(tri_edge);
        free(tri_dir);
        free(edges);
      }
      fclose(fp);
    }

    // Canonicalize the global winding: flip every triangle so the enclosed
    // signed volume (same formula as MFTLObjectVolume) is POSITIVE.
    if (ok)
    {
      double S = 0.0;
      for (int t = 0; t < nF; ++t)
      {
        const double* p0 = positions[triples[t][0]];
        const double* p1 = positions[triples[t][1]];
        const double* p2 = positions[triples[t][2]];
        S += p0[0] * (p1[1] * p2[2] - p1[2] * p2[1]) + p1[0] * (p2[1] * p0[2] - p2[2] * p0[1]) + p2[0] * (p0[1] * p1[2] - p0[2] * p1[1]);
      }
      if (S < 0.0)
      {
        for (int t = 0; t < nF; ++t)
        {
          int tmp = triples[t][1];
          triples[t][1] = triples[t][2];
          triples[t][2] = tmp;
        }
      }
    }
  }

  // Collective agreement: all ranks bail out together on a bad file.
  MPI_Bcast(&ok, 1, MPI_INT, 0, domain->comm_world);
  if (!ok)
  {
    free(positions);
    free(triples);
    return MFTL_ERROR;
  }

  if (rank == 0)
  {
    MFTLBuildTopologyFromTriples(obj, nV, (const double (*)[3]) positions, nF, (const int (*)[3]) triples);

    // Reject open or non-manifold input: a closed GTS surface leaves no
    // boundary half-edge after the twin rebuild.
    for (int i = 0; i < obj->n_half_edges; ++i)
    {
      if (obj->half_edges[i].twin_idx == MFTL_ERROR)
      {
        fprintf(stderr, "MFTLReadGTS: '%s' is not a closed manifold surface (boundary edge found)\n", path);
        ok = 0;
        break;
      }
    }
  }
  free(positions);
  free(triples);

  MPI_Bcast(&ok, 1, MPI_INT, 0, domain->comm_world);
  if (!ok) return MFTL_ERROR;

  // Collective finish identical to the generators: broadcast gid counters so
  // non-owning ranks agree, update global stats, flatten connectivity. The
  // caller distributes with MFTLRedistributeListOfObjects.
  MPI_Bcast(&obj->next_vertex_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&obj->next_triangle_gid, 1, MPI_INT, 0, domain->comm_world);
  MPI_Bcast(&obj->next_halfedge_gid, 1, MPI_INT, 0, domain->comm_world);
  MFTLObjectUpdateGlobalStats(obj);
  MFTLFlattenConnectivity(obj);

  if (rank == 0)
  {
    MFTLComputeTriangleProperties(obj);
    MFTL_ASSERT(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
  }
  return MFTL_SUCCESS;
}

int MFTLWriteGTS(MFTL_Object* obj, const char* path)
{
  MFTL_ASSERT(obj != MFTL_NULL);
  MFTL_ASSERT(path != MFTL_NULL);
  MFTL_Domain* domain = obj->domain;
  if (domain->rank != 0) return MFTL_SUCCESS;  // single-file GTS: rank 0 writes

  // Count owned triangles first. GTS is a single-file format: refuse to write
  // a PARTIAL surface when the mesh has been redistributed (rank 0 then owns
  // only its slice). Requires current global counts (MFTLEnsembleUpdateGlobal
  // Counts / MFTLObjectUpdateGlobalStats after redistribution).
  int nF = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (!obj->triangles[t].is_ghost && obj->triangles[t].owner_proc == domain->rank) nF++;
  if (nF == 0)
  {
    fprintf(stderr, "MFTLWriteGTS: no owned triangles on rank 0 (redistributed mesh?)\n");
    return MFTL_ERROR;
  }
  if (nF != obj->global_n_triangles)
  {
    fprintf(stderr,
            "MFTLWriteGTS: rank 0 owns %d of %d triangles — the surface is distributed after "
            "redistribution; refusing to write a partial GTS file (write before redistributing)\n",
            nF, obj->global_n_triangles);
    return MFTL_ERROR;
  }

  FILE* fp = fopen(path, "w");
  if (!fp)
  {
    fprintf(stderr, "MFTLWriteGTS: cannot open '%s' for writing\n", path);
    return MFTL_ERROR;
  }

  // First pass: renumber referenced vertices and collect unique edges in
  // first-appearance order (undirected key min<<32|max).
  int* vmap = malloc((size_t) obj->n_vertices * sizeof(int));
  for (int i = 0; i < obj->n_vertices; ++i) vmap[i] = -1;
  int nV = 0;
  int nE = 0;

  khash_t(mftl_gts_edge_map)* emap = kh_init(mftl_gts_edge_map);
  int (*edges)[2] = malloc((size_t)(3 * nF) * sizeof(*edges));
  int (*tri_e)[3] = malloc((size_t) nF * sizeof(*tri_e));

  int tw = 0;  // owned-triangle write index
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != domain->rank) continue;

    int he_idx[3];
    he_idx[0] = tri->he_idx;
    he_idx[1] = obj->half_edges[he_idx[0]].next_idx;
    he_idx[2] = obj->half_edges[he_idx[1]].next_idx;

    for (int j = 0; j < 3; ++j)
    {
      int he = he_idx[j];

      // edge endpoints: start = target of next(next), end = target of he
      int n1 = obj->half_edges[he].next_idx;
      int n2 = obj->half_edges[n1].next_idx;
      int a = obj->half_edges[n2].vertex_idx;  // start
      int b = obj->half_edges[he].vertex_idx;  // end

      if (vmap[a] < 0) vmap[a] = nV++;
      if (vmap[b] < 0) vmap[b] = nV++;

      khint64_t key = ((khint64_t)(a < b ? a : b) << 32) | (khint64_t)(a < b ? b : a);
      khiter_t k = kh_get(mftl_gts_edge_map, emap, key);
      int eid;
      if (k == kh_end(emap))
      {
        int ret;
        k = kh_put(mftl_gts_edge_map, emap, key, &ret);
        kh_value(emap, k) = nE;
        edges[nE][0] = a;
        edges[nE][1] = b;
        eid = nE;
        nE++;
      }
      else
        eid = kh_value(emap, k);

      tri_e[tw][j] = eid;
    }
    tw++;
  }

  // Write header + vertices in the new numbering (% .17g preserves doubles
  // bit-exactly, so read->write->read is stable).
  fprintf(fp, "%d %d %d GtsSurface GtsFace GtsEdge GtsVertex\n", nV, nE, nF);
  double (*out_pos)[3] = malloc((size_t) nV * sizeof(*out_pos));
  for (int i = 0; i < obj->n_vertices; ++i)
    if (vmap[i] >= 0) memcpy(out_pos[vmap[i]], obj->vertices[i].pos, 3 * sizeof(double));
  for (int i = 0; i < nV; ++i) fprintf(fp, "%.17g %.17g %.17g\n", out_pos[i][0], out_pos[i][1], out_pos[i][2]);
  free(out_pos);

  for (int e = 0; e < nE; ++e) fprintf(fp, "%d %d\n", vmap[edges[e][0]] + 1, vmap[edges[e][1]] + 1);

  // Triangles in owned order, as edge triples (1-based).
  tw = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != domain->rank) continue;
    fprintf(fp, "%d %d %d\n", tri_e[tw][0] + 1, tri_e[tw][1] + 1, tri_e[tw][2] + 1);
    tw++;
  }

  fclose(fp);
  free(vmap);
  free(edges);
  free(tri_e);
  kh_destroy(mftl_gts_edge_map, emap);
  return MFTL_SUCCESS;
}

void MFTLWritePvtu(MFTL_Ensemble* ens, const char* stem)
{
  MFTL_ASSERT(ens != MFTL_NULL);
  MFTL_ASSERT(stem != MFTL_NULL);
  // MFTLWriteVtu writes the per-rank <stem>_%05d.vtu pieces on every rank and
  // the rank-0 collection index <stem>.pvtu referencing all pieces.
  MFTLWriteVtu(ens, stem);
}
/* ========================================================================== */
/* Stage 15: native restart mesh (MFTL1 `.mesh` format, surface scalar incl.) */
/* ========================================================================== */
/*
 * Format (MFTL1 MFTL_ObjectWriteMeshForeach / MFTLObjectReadNativeMesh, FRONTTRACKING build):
 *   nv ne nt
 *   nv x  "owner gid x y z"                (%.15e)
 *   ne x  "owner edge_id v0_gid v1_gid"
 *   nt x  "owner gid e0 e1 e2 scalar"      (%.15e)
 * A triangle's vertex order follows from its first two edges exactly as in MFTL1
 * (MFTL_FindTriangleEdgesandVertices), so files are exchangeable between the libraries.
 * Written edges are the undirected edges, id = min(half-edge gid, twin gid).
 */
static int mftlNativeEdgeId(const MFTL_Object* obj, const MFTL_HalfEdge* he)
{
  int tg = MFTLHeTwinGid(obj, he);
  return (tg != MFTL_ERROR && tg < he->global_id) ? tg : he->global_id;
}

int MFTLWriteNativeMesh(const MFTL_Object* obj, const char* path)
{
  FILE* fp = fopen(path, "w");
  if (!fp)
  {
    fprintf(stderr, "MFTLWriteNativeMesh: cannot open %s\n", path);
    return MFTL_ERROR;
  }
  int ne = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[i];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (mftlNativeEdgeId(obj, he) == he->global_id) ne++;  // canonical half-edge of its edge
  }
  fprintf(fp, "%d %d %d\n", obj->n_vertices, ne, obj->n_triangles);
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    const MFTL_Vertex* v = &obj->vertices[i];
    fprintf(fp, "%d %d %.15e %.15e %.15e\n", v->owner_proc, v->global_id, v->pos[0], v->pos[1], v->pos[2]);
  }
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    const MFTL_HalfEdge* he = &obj->half_edges[i];
    if (he->flags & MFTL_FLAG_DEAD) continue;
    if (mftlNativeEdgeId(obj, he) != he->global_id) continue;
    fprintf(fp, "%d %d %d %d\n", obj->triangles[he->triangle_idx].owner_proc, he->global_id, MFTLHeOrigin(obj, he)->global_id,
            MFTLHeTarget(obj, he)->global_id);
  }
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    const MFTL_Triangle* tri = &obj->triangles[t];
    MFTL_HalfEdge* he[3];
    MFTLGetTriangleHalfedges(obj, tri, he);
    fprintf(fp, "%d %d %d %d %d %.15e\n", tri->owner_proc, tri->global_id, mftlNativeEdgeId(obj, he[0]), mftlNativeEdgeId(obj, he[1]),
            mftlNativeEdgeId(obj, he[2]), tri->surface_scalar);
  }
  fclose(fp);
  return MFTL_SUCCESS;
}

typedef struct
{
  int id, v0, v1;
} mftlNativeEdge;

static int mftlCmpNativeEdge(const void* a, const void* b)
{
  int x = ((const mftlNativeEdge*) a)->id, y = ((const mftlNativeEdge*) b)->id;
  return (x > y) - (x < y);
}

static const mftlNativeEdge* mftlFindNativeEdge(const mftlNativeEdge* e, int n, int id)
{
  int lo = 0, hi = n - 1;
  while (lo <= hi)
  {
    int mid = lo + (hi - lo) / 2;
    if (e[mid].id == id) return &e[mid];
    if (e[mid].id < id)
      lo = mid + 1;
    else
      hi = mid - 1;
  }
  return MFTL_NULL;
}

int MFTLReadNativeMesh(MFTL_Object* obj, const char* path)
{
  int nv = 0, ne = 0, nt = 0, bad = 0;
  int *vgid = MFTL_NULL, *tgid = MFTL_NULL, (*tvg)[3] = MFTL_NULL;
  double (*pos)[3] = MFTL_NULL;
  double* scalar = MFTL_NULL;
  mftlNativeEdge* edges = MFTL_NULL;

  if (path)
  {
    FILE* fp = fopen(path, "r");
    if (!fp || fscanf(fp, "%d %d %d", &nv, &ne, &nt) != 3 || nv < 0 || ne < 0 || nt < 0)
    {
      fprintf(stderr, "MFTLReadNativeMesh: cannot read %s\n", path);
      bad = 1;
      nv = ne = nt = 0;
    }
    vgid = (int*) malloc((size_t) (nv > 0 ? nv : 1) * sizeof(int));
    pos = malloc((size_t) (nv > 0 ? nv : 1) * sizeof(*pos));
    edges = (mftlNativeEdge*) malloc((size_t) (ne > 0 ? ne : 1) * sizeof(mftlNativeEdge));
    tgid = (int*) malloc((size_t) (nt > 0 ? nt : 1) * sizeof(int));
    tvg = malloc((size_t) (nt > 0 ? nt : 1) * sizeof(*tvg));
    scalar = (double*) malloc((size_t) (nt > 0 ? nt : 1) * sizeof(double));
    int owner;
    for (int i = 0; i < nv && !bad; ++i)
      if (fscanf(fp, "%d %d %lf %lf %lf", &owner, &vgid[i], &pos[i][0], &pos[i][1], &pos[i][2]) != 5) bad = 1;
    for (int i = 0; i < ne && !bad; ++i)
      if (fscanf(fp, "%d %d %d %d", &owner, &edges[i].id, &edges[i].v0, &edges[i].v1) != 4) bad = 1;
    if (!bad) qsort(edges, (size_t) ne, sizeof(mftlNativeEdge), mftlCmpNativeEdge);
    for (int t = 0; t < nt && !bad; ++t)
    {
      int e[3];
      if (fscanf(fp, "%d %d %d %d %d %lf", &owner, &tgid[t], &e[0], &e[1], &e[2], &scalar[t]) != 6)
      {
        bad = 1;
        break;
      }
      const mftlNativeEdge* E0 = mftlFindNativeEdge(edges, ne, e[0]);
      const mftlNativeEdge* E1 = mftlFindNativeEdge(edges, ne, e[1]);
      if (!E0 || !E1)
      {
        bad = 1;
        break;
      }
      // MFTL1 MFTL_FindTriangleEdgesandVertices: vertex order from the first two edges
      int V0 = E0->v0, V1, V2;
      if (V0 == E1->v0)
      {
        V1 = E1->v1;
        V2 = E0->v1;
      }
      else if (E0->v1 == E1->v1)
      {
        V1 = E0->v1;
        V2 = E1->v0;
      }
      else if (V0 == E1->v1)
      {
        V1 = E1->v0;
        V2 = E0->v1;
      }
      else
      {
        V1 = E0->v1;
        V2 = E1->v1;
      }
      tvg[t][0] = V0;
      tvg[t][1] = V1;
      tvg[t][2] = V2;
    }
    if (fp) fclose(fp);
    if (bad)
    {
      fprintf(stderr, "MFTLReadNativeMesh: malformed file %s\n", path);
      nv = nt = 0;
    }
  }

  int rc = MFTLBuildFromSoup(obj, nv, vgid, (const double (*)[3]) pos, MFTL_NULL, nt, tgid, (const int (*)[3]) tvg, MFTL_NULL);
  if (rc == MFTL_SUCCESS)
  {
    for (int t = 0; t < nt; ++t)
    {
      khiter_t k = kh_get(triangle_map, obj->triangle_lookup, tgid[t]);
      if (k != kh_end(obj->triangle_lookup)) obj->triangles[kh_value(obj->triangle_lookup, k)].surface_scalar = scalar[t];
    }
    for (int t = 0; t < obj->n_triangles; ++t)
      MFTLGetTriangleProperties(obj, &obj->triangles[t], &obj->triangles[t].area, obj->triangles[t].normal, obj->triangles[t].center);
  }
  free(vgid);
  free(pos);
  free(edges);
  free(tgid);
  free(tvg);
  free(scalar);
  int gbad = 0;
  MPI_Allreduce(&bad, &gbad, 1, MPI_INT, MPI_MAX, obj->domain->comm_cart);
  return (gbad || rc != MFTL_SUCCESS) ? MFTL_ERROR : MFTL_SUCCESS;
}
