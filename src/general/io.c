/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:39:02
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 15:57:58
 */

#include "mftl_types.h"
#include "mftl_constants.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
mftl_int_t MFTLWriteVtk(const MFTL_Object *object, const char *base_filename)
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
    // We need the 3 vertex INDICES relative to the list above.
    // Fortunately, v_base index IS the file index.

    mftl_int_t he0_idx = tri->he_idx;
    mftl_int_t he1_idx = he_base[he0_idx].next_idx;
    mftl_int_t he2_idx = he_base[he1_idx].next_idx;

    // Vertices of the triangle:
    // v0 = Origin of he0 (Target of he2)
    // v1 = Target of he0
    // v2 = Target of he1
    mftl_int_t v0_idx = he_base[he2_idx].vertex_idx;
    mftl_int_t v1_idx = he_base[he0_idx].vertex_idx;
    mftl_int_t v2_idx = he_base[he1_idx].vertex_idx;

    // Note: VTK expects 0-based indices into the POINTS list written above.
    // This corresponds exactly to our local indices.
    // (Your old code wrote global_id here, which is usually wrong for legacy VTK
    // unless it's a serial run where global_id == local_id).
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
 * @param domain_size Array of domain dimensions.
 * @param is_periodic Array of periodicity flags.
 * @param out_p1, out_p2 Output arrays for the shifted coordinates (p0 stays fixed).
 * @return 1 if the triangle wraps (needs new vertices), 0 if normal.
 */
static int MFTLCheckWrap(const double p0[3], const double p1[3], const double p2[3], const double domain_size[3], const bool is_periodic[3], double out_p1[3],
                         double out_p2[3])
{
  int wrapped = 0;
  // Initialize outputs with original values
  memcpy(out_p1, p1, 3 * sizeof(double));
  memcpy(out_p2, p2, 3 * sizeof(double));

  for (int d = 0; d < 3; ++d)
  {
    if (!is_periodic[d]) continue;

    double L = domain_size[d];
    double half_L = L * 0.5;
    double diff1 = p1[d] - p0[d];
    double diff2 = p2[d] - p0[d];

    // Check Neighbor 1 (relative to p0)
    if (diff1 > half_L)
    {
      out_p1[d] -= L;
      wrapped = 1;
    }
    else if (diff1 < -half_L)
    {
      out_p1[d] += L;
      wrapped = 1;
    }

    // Check Neighbor 2 (relative to p0)
    if (diff2 > half_L)
    {
      out_p2[d] -= L;
      wrapped = 1;
    }
    else if (diff2 < -half_L)
    {
      out_p2[d] += L;
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
 * * @param objects Array of pointers to MFTL_Object.
 * @param num_objects Number of objects in the array.
 * @param filename_prefix Base path/name for output files.
 */
void MFTLWriteVtu(MFTL_Object **objects, int num_objects, const char *filename_prefix)
{
  if (num_objects == 0) return;

  // Assume all objects share the same communicator/rank/periodicity
  MFTL_Object *ref_obj = objects[0];
  int my_rank, nproc;
  MPI_Comm_rank(ref_obj->cart_comm, &my_rank);
  MPI_Comm_size(ref_obj->cart_comm, &nproc);

  // --- PRE-CALCULATION: Count Wrapped Triangles per Object ---
  // We need to know the total file size before writing the header.
  // We store the wrapped count for each object to avoid re-calculating it later.
  long long *wrapped_counts = (long long *) calloc(num_objects, sizeof(long long));
  double dummy_p1[3], dummy_p2[3];

  long long total_file_points = 0;
  long long total_file_cells = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = objects[o];
    long long current_wrapped = 0;

    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

      const double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      const double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      const double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(p0, p1, p2, obj->domain_size, obj->is_periodic, dummy_p1, dummy_p2))
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
    MPI_Abort(ref_obj->cart_comm, 1);
  }

  // --- XML HEADER ---
  fprintf(fp, "<?xml version=\"1.0\"?>\n");
  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
  fprintf(fp, "  <UnstructuredGrid>\n");
  fprintf(fp, "    <Piece NumberOfPoints=\"%lld\" NumberOfCells=\"%lld\">\n", total_file_points, total_file_cells);

  // ======================================================================
  // 1. POINTS (Coordinates)
  // ======================================================================
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n");

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = objects[o];

    // A. Original Vertices
    for (size_t i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %.10e %.10e %.10e\n", obj->vertices[i].pos[0], obj->vertices[i].pos[1], obj->vertices[i].pos[2]);
    }

    // B. Visual Ghost Vertices (Unwrapped)
    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

      const double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      const double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      const double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      double fix_p1[3], fix_p2[3];
      if (MFTLCheckWrap(p0, p1, p2, obj->domain_size, obj->is_periodic, fix_p1, fix_p2))
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
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");

  // Global Offset Tracker: Tracks how many vertices were written by PREVIOUS objects
  long long global_vertex_offset = 0;

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = objects[o];

    // The visual ghosts for THIS object start immediately after THIS object's original vertices
    long long local_ghost_start_idx = global_vertex_offset + obj->n_vertices;
    long long current_ghost_idx = local_ghost_start_idx;

    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

      const double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      const double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      const double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(p0, p1, p2, obj->domain_size, obj->is_periodic, dummy_p1, dummy_p2))
      {
        // Wrapped: Use the next 3 available ghost indices
        fprintf(fp, "          %lld %lld %lld\n", current_ghost_idx, current_ghost_idx + 1, current_ghost_idx + 2);
        current_ghost_idx += 3;
      }
      else
      {
        // Normal: Use original indices, shifted by the global offset
        fprintf(fp, "          %lld %lld %lld\n", global_vertex_offset + (long long) obj->half_edges[he2].vertex_idx,
                global_vertex_offset + (long long) obj->half_edges[he0].vertex_idx, global_vertex_offset + (long long) obj->half_edges[he1].vertex_idx);
      }
    }

    // Advance Global Offset by total points in THIS object (orig + ghosts)
    global_vertex_offset += obj->n_vertices + (wrapped_counts[o] * 3);
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Offsets ---
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  long long running_offset = 0;
  for (int o = 0; o < num_objects; ++o)
  {
    for (size_t i = 0; i < objects[o]->n_triangles; ++i)
    {
      running_offset += 3;
      fprintf(fp, "          %lld\n", running_offset);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Types ---
  fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int o = 0; o < num_objects; ++o)
  {
    for (size_t i = 0; i < objects[o]->n_triangles; ++i)
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
  fprintf(fp, "        <DataArray type=\"Int32\" Name=\"VertexOwner\" format=\"ascii\">\n");

  for (int o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = objects[o];

    // A. Original
    for (size_t i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %d\n", obj->vertices[i].owner_proc);
    }
    // B. Ghosts (Re-run check logic)
    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      // (Repeat wrap check to fill ghost owners)
      // ... omitting full expansion for brevity, assume copy-paste of logic above ...
      // Just ensure you fetch v->owner_proc
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;
      const double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      const double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      const double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;
      if (MFTLCheckWrap(p0, p1, p2, obj->domain_size, obj->is_periodic, dummy_p1, dummy_p2))
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

  // A. Triangle Owner
  fprintf(fp, "        <DataArray type=\"Int32\" Name=\"TriangleOwner\" format=\"ascii\">\n");
  for (int o = 0; o < num_objects; ++o)
  {
    for (size_t i = 0; i < objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          %d\n", objects[o]->triangles[i].owner_proc);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // B. Body ID (Object Global ID)
  fprintf(fp, "        <DataArray type=\"Int64\" Name=\"BodyID\" format=\"ascii\">\n");
  for (int o = 0; o < num_objects; ++o)
  {
    long long body_id = (long long) objects[o]->global_id;
    // If you haven't set global_id yet, use 'o' loop index
    for (size_t i = 0; i < objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          %lld\n", body_id);
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
      fprintf(fp, "      <PDataArray type=\"Int64\" Name=\"BodyID\"/>\n");
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