#include <stdlib.h>
#include <time.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_io.h"
#include "mftl_geometry.h"

int main(int argc, char* argv[])
{
  // --- MPI Initialization ---
  MPI_Init(&argc, &argv);

  srand(42);  // Use a fixed seed for reproducible benchmarks

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // --- Argument Parsing ---
  if (argc != 7)
  {
    if (rank == 0)
    {
      fprintf(stderr, "Usage: %s <cuts_x> <cuts_y> <cuts_z> <periodic_x> <periodic_y> <periodic_z>\n", argv[0]);
    }
    MPI_Finalize();
    return -1;
  }

  int cuts[3];
  bool periodicity[3];

  cuts[0] = atoi(argv[1]);
  cuts[1] = atoi(argv[2]);
  cuts[2] = atoi(argv[3]);

  periodicity[0] = (atoi(argv[4]) != 0);
  periodicity[1] = (atoi(argv[5]) != 0);
  periodicity[2] = (atoi(argv[6]) != 0);

  if (rank == 0)
  {
    printf("Running with %d processes in a %d x %d x %d grid.\n", nproc, cuts[0], cuts[1], cuts[2]);
  }

  // --- Simulation Constants ---
  int NUM_SPHERES = 1000;
  double SPHERE_RADIUS = 0.1;
  int REFINEMENTS = 4;
  double global_min[3] = {-4.0, -2.0, -2.0};
  double global_max[3] = {4.0, 2.0, 2.0};

  // --- Create Ensemble ---
  MFTL_Ensemble* ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, global_min, global_max, periodicity, NUM_SPHERES);
  if (!ensemble)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (size_t o = 0; o < NUM_SPHERES; o++)
  {
    // --- Create the Mesh Object ---
    MFTL_Object* object = MFTLObjectCreate((int) o);
    MFTLEnsembleAddObject(ensemble, object);

    // --- Generate random center coordinates ---
    double rand_x = (double) rand() / RAND_MAX;
    double rand_y = (double) rand() / RAND_MAX;
    double rand_z = (double) rand() / RAND_MAX;

    double center[3] = {-3.0 + rand_x * (3.0 - (-3.0)), -1.85 + rand_y * (1.85 - (-1.85)), -1.85 + rand_z * (1.85 - (-1.85))};

    MFTLGenerateSphere(object, center, SPHERE_RADIUS, REFINEMENTS);
    // Note: ComputeTriangleProperties might be needed after gen
    MFTLComputeTriangleProperties(object);
  }

  // Redistribute all at once
  MFTLRedistributeListOfObjects(ensemble);

  double dt = 0.1, u = 1.0;
  int step = 0;

  double total_redist_time = 0.0;
  double total_simulation_time = 0.0;

  char filename_prefix[256];

  MPI_Barrier(MPI_COMM_WORLD);
  double t_loop_start = MPI_Wtime();

  for (double time = 0.0; time < 50.0; time += dt)
  {
    if (rank == 0) printf("start timestep: %i \n", step);

    for (size_t o = 0; o < ensemble->num_objects; o++)
    {
      MFTL_Object* object = ensemble->objects[o];
      MFTL_Vertex* vertices = object->vertices;
      for (int i = 0; i < object->n_vertices; i++)
      {
        MFTL_Vertex* v = &vertices[i];
        if (v->owner_proc == rank)
        {
          v->pos[0] += u * dt;
          MFTLApplyPeriodicity(&ensemble->domain, v->pos);
        }
      }
    }

    double t_update_start = MPI_Wtime();
    MFTLUpdateGhostPositions(ensemble, MFTL_NULL);
    double t_update_end = MPI_Wtime();
    total_redist_time += (t_update_end - t_update_start);

    for (size_t o = 0; o < ensemble->num_objects; o++)
    {
      MFTLComputeTriangleProperties(ensemble->objects[o]);
    }

    double t_redist_start = MPI_Wtime();
    MFTLRedistributeListOfObjects(ensemble);
    double t_redist_end = MPI_Wtime();
    total_redist_time += (t_redist_end - t_redist_start);

    // snprintf(filename_prefix, sizeof(filename_prefix), "list_output_%04d", step);
    // MFTLWriteVtu(ensemble, filename_prefix);

    step++;
  }

  double t_loop_end = MPI_Wtime();
  total_simulation_time = t_loop_end - t_loop_start;

  double max_redist_time, max_total_time;
  MPI_Reduce(&total_redist_time, &max_redist_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&total_simulation_time, &max_total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    printf("\n=== LIST BENCHMARK RESULTS ===\n");
    printf("Total Loop Time:       %f s\n", max_total_time);
    printf("Redistribution Time:   %f s\n", max_redist_time);
    printf("Redistribution %%:      %.2f %%\n", (max_redist_time / max_total_time) * 100.0);
    printf("====================================\n\n");
  }

  MFTLEnsembleDestroy(&ensemble);
  MPI_Finalize();
  return 0;
}