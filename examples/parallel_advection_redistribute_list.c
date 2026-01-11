#include <stdlib.h>
#include <time.h>
#include "mftl_types.h"
#include "mftl_constants.h"
#include "mftl_object.h"
#include "mftl_communication.h"
#include "mftl_io.h"
#include "mftl_geometry.h"

int main(int argc, char* argv[])
{
  // --- MPI Initialization ---
  MPI_Init(&argc, &argv);

  srand(42);  // Use a fixed seed for reproducible benchmarks

  // Register our custom types right after initialization
  MFTLRegisterMpiTypes();

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  // --- Argument Parsing ---
  // We expect 6 arguments: 3 for cuts (x,y,z) and 3 for periodicity (x,y,z)
  // argc will be 1 (program name) + 6 = 7
  if (argc != 7)
  {
    if (rank == 0)
    {
      // Only Rank 0 prints the error message to avoid clutter.
      fprintf(stderr, "Usage: %s <cuts_x> <cuts_y> <cuts_z> <periodic_x> <periodic_y> <periodic_z>\n", argv[0]);
      fprintf(stderr, "Example: mpirun -n 2 %s 2 1 1 1 1 0\n", argv[0]);
      fprintf(stderr, "  (for a 2x1x1 grid, periodic in x and y, not z)\n");
    }
    MPI_Finalize();
    return -1;
  }

  // Declare the arrays that will hold the parsed values.
  int cuts[3];
  bool periodicity[3];

  // Parse the arguments from the argv string array.
  // atoi() converts a string to an integer.
  cuts[0] = atoi(argv[1]);
  cuts[1] = atoi(argv[2]);
  cuts[2] = atoi(argv[3]);

  // For booleans, we'll treat 1 as true and 0 as false.
  // The '!= 0' comparison safely converts the integer to a boolean.
  periodicity[0] = (atoi(argv[4]) != 0);
  periodicity[1] = (atoi(argv[5]) != 0);
  periodicity[2] = (atoi(argv[6]) != 0);

  // Optional: Have Rank 0 print the configuration being used.
  if (rank == 0)
  {
    printf("Running with %d processes in a %d x %d x %d grid.\n", nproc, cuts[0], cuts[1], cuts[2]);
    printf("Periodicity: [x=%s, y=%s, z=%s]\n", periodicity[0] ? "true" : "false", periodicity[1] ? "true" : "false", periodicity[2] ? "true" : "false");
  }

  // --- Simulation Constants ---
  const int NUM_SPHERES = 1000;
  const double SPHERE_RADIUS = 0.1;
  const int REFINEMENTS = 4;
  const double global_min[3] = {-4.0, -2.0, -2.0};
  const double global_max[3] = {4.0, 2.0, 2.0};

  // --- 1. Allocate the List of Object Pointers ---
  // We need an array that holds 'NUM_SPHERES' pointers to MFTL_Object structs.
  MFTL_Object** objects_list = (MFTL_Object**) malloc(NUM_SPHERES * sizeof(MFTL_Object*));
  if (!objects_list)
  {
    fprintf(stderr, "Rank %d: Failed to allocate objects list.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  for (size_t o = 0; o < NUM_SPHERES; o++)
  {
    // --- Create the Mesh Object ---
    objects_list[o] = MFTLObjectCreate(MPI_COMM_WORLD);
    if (!objects_list[o])
    {
      fprintf(stderr, "Rank %d: Failed to create mesh object.\n", rank);
      MPI_Finalize();
      return -1;
    }
    // SET THE ID
    objects_list[o]->global_id = (mftl_int_t) o;

    // --- Initialize the Domain ---
    MFTLObjectInitDomain(objects_list[o], cuts, global_min, global_max, periodicity);

    // --- Generate random center coordinates ---
    // 1. Get a random double between 0.0 and 1.0
    double rand_x = (double) rand() / RAND_MAX;
    double rand_y = (double) rand() / RAND_MAX;
    double rand_z = (double) rand() / RAND_MAX;

    // 2. Scale and shift to the desired range [min, max]
    // Formula: min + random_value * (max - min)
    const double center[3] = {
        -3.0 + rand_x * (3.0 - (-3.0)),  // Range: [-2.0, 2.0]
        -1.85 + rand_y * (1.85 - (-1.85)),  // Range: [-1.85, 1.85]
        -1.85 + rand_z * (1.85 - (-1.85))  // Range: [-1.85, 1.85]
    };

    // printf("Rank %d: Generating sphere %zu with center %f %f %f\n", rank, o, center[0], center[1], center[2]);

    int result = MFTLGenerateSphere(objects_list[o], center, SPHERE_RADIUS, REFINEMENTS);
    // if (result != 0)
    // {
    //   fprintf(stderr, "Rank %d: Failed to generate sphere.\n", rank);
    // }
    // else
    // {
    //   printf("Rank %d: Sphere generated with %d vertices and %d triangles.\n", rank, objects_list[o]->global_n_vertices, objects_list[o]->global_n_triangles);
    // }

    MFTLComputeTriangleProperties(objects_list[o]);
  }

  int result = MFTLRedistributeListOfObjects(objects_list, NUM_SPHERES, MFTL_TRUE);
  // if (result != MFTL_SUCCESS)
  // {
  //   fprintf(stderr, "Rank %d: Failed to redistribute object.\n", rank);
  //   return -1;
  // }

  char filename_prefix[256];
  // snprintf(filename_prefix, sizeof(filename_prefix), "mesh_t_%04d", 0);
  // MFTLWriteVtu(objects_list, NUM_SPHERES, filename_prefix);

  double dt = 0.1, u = 1.0;
  int step = 0;

  // --- TIMERS INITIALIZATION ---
  double total_redist_time = 0.0;
  double total_simulation_time = 0.0;

  MPI_Barrier(MPI_COMM_WORLD);
  double t_loop_start = MPI_Wtime();  // START TOTAL TIMER

  for (double time = 0.0; time < 50.0; time += dt)  // NOTE: Your list file had 100.0, sequential had 2.0. MATCH THESE!
  {
    if (rank == 0) printf("start timestep: %i \n", step);

    for (size_t o = 0; o < NUM_SPHERES; o++)
    {
      MFTL_Object* object = objects_list[o];
      MFTL_Vertex* vertices = object->vertices;
      for (int i = 0; i < object->n_vertices; i++)
      {
        MFTL_Vertex* v = &vertices[i];
        if (v->owner_proc == rank)
        {
          v->pos[0] += u * dt;
          MFTLApplyPeriodicity(object, v->pos);
        }
      }
    }
    double t_update_start = MPI_Wtime();
    MFTLUpdateGhostPositions(objects_list, NUM_SPHERES, MFTL_NULL);
    double t_update_end = MPI_Wtime();
    total_redist_time += (t_update_end - t_update_start);
    for (size_t o = 0; o < NUM_SPHERES; o++)
    {
      MFTL_Object* object = objects_list[o];
      MFTLComputeTriangleProperties(object);
    }

    // --- MEASURE COMPONENT TIME ---
    double t_redist_start = MPI_Wtime();
    MFTLRedistributeListOfObjects(objects_list, NUM_SPHERES, MFTL_TRUE);
    double t_redist_end = MPI_Wtime();
    total_redist_time += (t_redist_end - t_redist_start);
    // -----------------------------

    // snprintf(filename_prefix, sizeof(filename_prefix), "mesh_t_%04d", step);
    // MFTLWriteVtu(objects_list, NUM_SPHERES, filename_prefix);

    step++;
  }

  double t_loop_end = MPI_Wtime();  // END TOTAL TIMER
  total_simulation_time = t_loop_end - t_loop_start;

  // --- AGGREGATE AND PRINT RESULTS ---
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

  // --- 5. Destroy the Mesh Object ---
  // All processes must call destroy to free the mesh and communicator.
  for (size_t i = 0; i < NUM_SPHERES; i++)
  {
    MFTL_Object* object = objects_list[i];
    MFTLObjectDestroy(&object);
    // printf("Rank %d: Mesh object destroyed.\n", rank);
  }
  free(objects_list);

  // --- MPI Finalization ---
  MPI_Finalize();
  return 0;
}