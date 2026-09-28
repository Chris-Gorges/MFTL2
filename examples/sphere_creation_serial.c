/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief The standard MFTL lifecycle, end to end.
 *
 *   MPI_Init
 *     -> MFTLEnsembleCreate          (collective: domain + MPI topology)
 *     -> MFTLObjectCreate            (an empty mesh container)
 *     -> MFTLEnsembleAddObject       (links the object to the domain)
 *     -> MFTLGenerateSphere          (collective: rank 0 builds, IDs broadcast)
 *     -> MFTLObjectReorderMorton     (local: spatial renumbering of the pools)
 *     -> MFTLEnsembleUpdate          (collective: geometry + quality, one pass)
 *     -> MFTLEnsembleDestroy         (cascades to every object it owns)
 *   MPI_Finalize
 *
 * It also writes a legacy .vtk file so the result can be looked at, through
 * MFTLWriteVtk -- the raw single-rank dump. That writer emits `verts.normal` and
 * `tris.area` without validating them, which is exactly what this example wants:
 * on a sphere every vertex normal must point straight out from the centre, so a
 * sign error in the winding or in the accumulation is visible at a glance in the
 * viewer rather than only in the printed statistics.
 */

#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* --- 1. The world -------------------------------------------------------
   *
   * One process grid cell per rank, a box big enough for the sphere, and no
   * periodicity. Collective: every rank must call it. */
  int cuts[3] = {1, 1, 1};
  double global_min[3] = {-2.0, -2.0, -2.0};
  double global_max[3] = {2.0, 2.0, 2.0};
  bool periodicity[3] = {false, false, false};

  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  cuts[0] = nproc;  // a thin decomposition, so the example runs at any rank count

  MFTL_Ensemble *ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, global_min, global_max, periodicity, 10);
  if (!ensemble)
  {
    fprintf(stderr, "Rank %d: failed to create the MFTL ensemble.\n", rank);
    MPI_Finalize();
    return 1;
  }
  if (rank == 0) printf("Ensemble created: %d rank(s), grid {%d, %d, %d}.\n", nproc, cuts[0], cuts[1], cuts[2]);

  /* --- 2. The object ------------------------------------------------------
   *
   * An empty container. It has no domain until it is added to the ensemble, and
   * the ensemble takes ownership -- there is no MFTLObjectDestroy below. */
  MFTL_Object *object = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(ensemble, object);

  /* --- 3. Generate --------------------------------------------------------
   *
   * Collective. Rank 0 builds the icosphere; every rank leaves with the same ID
   * counters and the same global entity counts. */
  double center[3] = {0.0, 0.0, 0.0};
  const double radius = 1.0;
  const int refinements = 5;

  if (MFTLGenerateSphere(object, center, radius, refinements) != MFTL_SUCCESS)
  {
    fprintf(stderr, "Rank %d: failed to generate the sphere.\n", rank);
    MFTLEnsembleDestroy(&ensemble);
    MPI_Finalize();
    return 1;
  }

  if (rank == 0)
  {
    printf("Sphere generated (refinement %d): %d vertices, %d triangles, %d half-edges.\n", refinements, object->global_n_vertices,
           object->global_n_triangles, 3 * object->global_n_triangles);
  }

  /* --- 4. Renumber along the space-filling curve --------------------------
   *
   * Local, and optional -- nothing requires it, everything benefits. It permutes
   * the AoS records and every SoA array together and rebuilds the derived
   * arrays, so the object is immediately usable afterwards. Pool INDICES change;
   * global IDs do not. */
  MFTLObjectReorderMorton(object);
  if (rank == 0) printf("Pools renumbered along the Morton curve.\n");

  /* --- 5. The per-timestep pass -------------------------------------------
   *
   * One gathered pass writes area, normal and centre AND accumulates the quality statistics, then three fused reductions make
   * them global. Collective. */
  MFTLEnsembleUpdate(ensemble);

  /* --- 5b. Per-vertex geometry --------------------------------------------
   *
   * Area-weighted vertex normals: one triangle scatter, one
   * normalising pass. LOCAL -- it accumulates over ghost triangles too, so an
   * owned vertex needs no communication to get the exact answer. */
  MFTLComputeVertexNormals(object, MFTL_NORMAL_WEIGHT_AREA);

  if (rank == 0)
  {
    const MFTL_ObjectQualityStatistics *q = &object->quality;
    const double exact_area = 4.0 * MFTL_PI * radius * radius;

    /* On a sphere the vertex normal must be the radial direction, so the worst
     * deviation is a discretisation error and must fall with refinement. */
    double worst_deg = 0.0;
    for (int v = 0; v < object->verts.n; ++v)
    {
      const double *p = MFTLVertexPos(object, v);
      const double *n = &object->verts.normal[3 * v];
      const double rl = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
      double dot = (p[0] * n[0] + p[1] * n[1] + p[2] * n[2]) / rl;
      if (dot > 1.0) dot = 1.0;
      const double d = acos(dot) * 180.0 / MFTL_PI;
      if (d > worst_deg) worst_deg = d;
    }
    printf("\nVertex normals: worst deviation from radial %.4f deg\n", worst_deg);

    printf("\nMesh quality (global):\n");
    printf("  triangle area     min %.6e  avg %.6e  max %.6e\n", q->min_triangle_area, q->avg_triangle_area, q->max_triangle_area);
    printf("  mean-ratio        min %.6f  avg %.6f  max %.6f   (1 = equilateral)\n", q->min_aspect_ratio, q->avg_aspect_ratio, q->max_aspect_ratio);
    printf("  edge length^2     min %.6e  avg %.6e  max %.6e\n", q->min_edge_length_sq, q->avg_edge_length_sq, q->max_edge_length_sq);
    printf("  total area        %.9f   (exact %.9f, error %.4f %%)\n", q->avg_triangle_area * object->global_n_triangles, exact_area,
           100.0 * (exact_area - q->avg_triangle_area * object->global_n_triangles) / exact_area);
  }

  /* --- 6. Validate and write ---------------------------------------------- */
  if (rank == 0)
  {
    MFTLCheckMeshIntegrity(object);
    MFTLWriteVtk(object, "sphere");
    printf("\nWrote sphere.vtk (%d points, %d polygons).\n", object->verts.n, object->tris.n);
  }

  /* --- 7. Tear down -------------------------------------------------------
   *
   * Destroys the domain, every object, and both pool blocks of each. Two free()
   * calls per object -- one per pool -- because a pool is one allocation. */
  MFTLEnsembleDestroy(&ensemble);
  if (rank == 0) printf("Ensemble and objects destroyed.\n");

  MPI_Finalize();
  return 0;
}
