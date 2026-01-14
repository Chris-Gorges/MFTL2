/*
 * @Author: Christian Gorges
 * @Date: 2025-08-12 12:39:02
 * @Last Modified by: Christian Gorges
 * @Last Modified time: 2025-09-24 15:57:58
 */

#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_geometry.h"
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
    mftl_int_t he0_idx = tri->he_idx;
    mftl_int_t he1_idx = he_base[he0_idx].next_idx;
    mftl_int_t he2_idx = he_base[he1_idx].next_idx;

    // Vertices of the triangle:
    mftl_int_t v0_idx = he_base[he2_idx].vertex_idx;
    mftl_int_t v1_idx = he_base[he0_idx].vertex_idx;
    mftl_int_t v2_idx = he_base[he1_idx].vertex_idx;

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

  mftl_int_t num_objects = ensemble->num_objects;
  MFTL_Domain *domain = &ensemble->domain;
  int my_rank = domain->rank;
  int nproc = domain->nproc;

  // --- PRE-CALCULATION: Count Wrapped Triangles per Object ---
  mftl_int_t *wrapped_counts = (mftl_int_t *) calloc(ensemble->num_objects, sizeof(mftl_int_t));
  double dummy_p1[3], dummy_p2[3];

  mftl_int_t total_file_points = 0;
  mftl_int_t total_file_cells = 0;

  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    mftl_int_t current_wrapped = 0;

    for (mftl_int_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

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
  // UPDATED: Use MFTL_FMT for counts
  fprintf(fp, "    <Piece NumberOfPoints=\"" MFTL_FMT "\" NumberOfCells=\"" MFTL_FMT "\">\n", total_file_points, total_file_cells);

  // ======================================================================
  // 1. POINTS (Coordinates)
  // ======================================================================
  fprintf(fp, "      <Points>\n");
  fprintf(fp, "        <DataArray type=\"Float64\" Name=\"Coordinates\" NumberOfComponents=\"3\" format=\"ascii\">\n");

  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];

    // A. Original Vertices
    for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %.10e %.10e %.10e\n", obj->vertices[i].pos[0], obj->vertices[i].pos[1], obj->vertices[i].pos[2]);
    }

    // B. Visual Ghost Vertices (Unwrapped)
    for (mftl_int_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

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
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"connectivity\" format=\"ascii\">\n", MFTL_VTK_TYPE);

  // Global Offset Tracker
  mftl_int_t global_vertex_offset = 0;

  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];
    mftl_int_t local_ghost_start_idx = global_vertex_offset + obj->n_vertices;
    mftl_int_t current_ghost_idx = local_ghost_start_idx;

    for (size_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;

      double *p0 = obj->vertices[obj->half_edges[he2].vertex_idx].pos;
      double *p1 = obj->vertices[obj->half_edges[he0].vertex_idx].pos;
      double *p2 = obj->vertices[obj->half_edges[he1].vertex_idx].pos;

      if (MFTLCheckWrap(domain, p0, p1, p2, dummy_p1, dummy_p2))
      {
        fprintf(fp, "          " MFTL_FMT " " MFTL_FMT " " MFTL_FMT "\n", current_ghost_idx, current_ghost_idx + 1, current_ghost_idx + 2);
        current_ghost_idx += 3;
      }
      else
      {
        fprintf(fp, "          " MFTL_FMT " " MFTL_FMT " " MFTL_FMT "\n", global_vertex_offset + obj->half_edges[he2].vertex_idx,
                global_vertex_offset + obj->half_edges[he0].vertex_idx, global_vertex_offset + obj->half_edges[he1].vertex_idx);
      }
    }

    global_vertex_offset += obj->n_vertices + (wrapped_counts[o] * 3);
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Offsets ---
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"offsets\" format=\"ascii\">\n", MFTL_VTK_TYPE);
  mftl_int_t running_offset = 0;
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    for (mftl_int_t i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      running_offset += 3;
      fprintf(fp, "          " MFTL_FMT "\n", running_offset);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // --- Types ---
  fprintf(fp, "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    for (mftl_int_t i = 0; i < ensemble->objects[o]->n_triangles; ++i)
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

  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    MFTL_Object *obj = ensemble->objects[o];

    // A. Original
    for (mftl_int_t i = 0; i < obj->n_vertices; ++i)
    {
      fprintf(fp, "          %d\n", obj->vertices[i].owner_proc);
    }
    // B. Ghosts
    for (mftl_int_t i = 0; i < obj->n_triangles; ++i)
    {
      mftl_int_t he0 = obj->triangles[i].he_idx;
      mftl_int_t he1 = obj->half_edges[he0].next_idx;
      mftl_int_t he2 = obj->half_edges[he1].next_idx;
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
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    for (mftl_int_t i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          %d\n", ensemble->objects[o]->triangles[i].owner_proc);
    }
  }
  fprintf(fp, "        </DataArray>\n");

  // B. Body ID (Object Global ID - uses mftl_int_t)
  fprintf(fp, "        <DataArray type=\"%s\" Name=\"BodyID\" format=\"ascii\">\n", MFTL_VTK_TYPE);
  for (mftl_int_t o = 0; o < num_objects; ++o)
  {
    mftl_int_t body_id = (mftl_int_t) ensemble->objects[o]->global_id;
    for (mftl_int_t i = 0; i < ensemble->objects[o]->n_triangles; ++i)
    {
      fprintf(fp, "          " MFTL_FMT "\n", body_id);
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
      fprintf(fp, "      <PDataArray type=\"%s\" Name=\"BodyID\"/>\n", MFTL_VTK_TYPE);
      fprintf(fp, "    </PCellData>\n");

      for (mftl_int_t i = 0; i < nproc; ++i)
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