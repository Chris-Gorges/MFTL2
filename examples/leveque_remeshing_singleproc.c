/*
 * Example (one rank): LeVeque single-vortex advection of a sphere with RK4 and periodic serial
 * remeshing; VTU output of the deformed front.
 */
#include <stdlib.h>
#include <time.h>
#include "mftl_core.h"
#include "mftl_constants.h"
#include "mftl_communication.h"
#include "mftl_io.h"
#include "mftl_geometry.h"
#include <math.h>
#include "mftl_remeshing.h"

#define pi (3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825342117067)

int AdvectFront(MFTL_Object* Object, double time, double dt, double T)
{
  double t = time, k1[3] = {0.0, 0.0, 0.0}, k2[3] = {0.0, 0.0, 0.0}, k3[3] = {0.0, 0.0, 0.0}, k4[3] = {0.0, 0.0, 0.0}, X[3] = {0.0, 0.0, 0.0};

  MFTL_Vertex* Vertex_base = Object->vertices;

  for (int i = 0; i < Object->n_vertices; ++i)
  {
    MFTL_Vertex* Vertex = &Vertex_base[i];

    t = time;

    X[0] = Vertex->pos[0];
    X[1] = Vertex->pos[1];
    X[2] = Vertex->pos[2];

    k1[0] = (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k1[1] = (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k1[2] = (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = Vertex->pos[0] + 0.5 * dt * k1[0];
    X[1] = Vertex->pos[1] + 0.5 * dt * k1[1];
    X[2] = Vertex->pos[2] + 0.5 * dt * k1[2];
    t = time + 0.5 * dt;

    k2[0] = (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k2[1] = (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k2[2] = (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = Vertex->pos[0] + 0.5 * dt * k2[0];
    X[1] = Vertex->pos[1] + 0.5 * dt * k2[1];
    X[2] = Vertex->pos[2] + 0.5 * dt * k2[2];

    k3[0] = (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k3[1] = (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k3[2] = (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    X[0] = Vertex->pos[0] + dt * k3[0];
    X[1] = Vertex->pos[1] + dt * k3[1];
    X[2] = Vertex->pos[2] + dt * k3[2];
    t = time + dt;

    k4[0] = (2.0) * pow(sin(pi * X[0]), 2.0) * sin(2.0 * pi * X[1]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k4[1] = (-1.0) * pow(sin(pi * X[1]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[2]) * cos(pi * t / T);
    k4[2] = (-1.0) * pow(sin(pi * X[2]), 2.0) * sin(2.0 * pi * X[0]) * sin(2.0 * pi * X[1]) * cos(pi * t / T);

    Vertex->pos[0] += (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
    Vertex->pos[1] += (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);
    Vertex->pos[2] += (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]);
  }

  /* Euler Method */
  // for (MFTL_Vertex *Vertex = Object->Vertexes; Vertex != MFTL_NULL; Vertex = Vertex->hh.next)
  // {
  //   Vertex->pos[0] += (2.0) * pow(sin(pi * Vertex->pos[0]), 2.0) * sin(2.0 * pi * Vertex->pos[1]) * sin(2.0 * pi * Vertex->pos[2]) * cos(pi * time / T) * dt;
  //   Vertex->pos[1] += (-1.0) * pow(sin(pi * Vertex->pos[1]), 2.0) * sin(2.0 * pi * Vertex->pos[0]) * sin(2.0 * pi * Vertex->pos[2]) * cos(pi * time / T) * dt;
  //   Vertex->pos[2] += (-1.0) * pow(sin(pi * Vertex->pos[2]), 2.0) * sin(2.0 * pi * Vertex->pos[0]) * sin(2.0 * pi * Vertex->pos[1]) * cos(pi * time / T) * dt;
  // }

  return (MFTL_SUCCESS);
}

int main(int argc, char* argv[])
{
  // --- MPI Initialization ---
  MPI_Init(&argc, &argv);

  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3];
  bool periodicity[3];

  cuts[0] = 1;
  cuts[1] = 1;
  cuts[2] = 1;

  periodicity[0] = MFTL_FALSE;
  periodicity[1] = MFTL_FALSE;
  periodicity[2] = MFTL_FALSE;

  double xmin[] = {0.0, 0.0, 0.0};
  double xmax[] = {1.0, 1.0, 1.0};

  // --- Create Ensemble ---
  MFTL_Ensemble* ensemble = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, periodicity, 1);
  if (!ensemble)
  {
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // --- Create the Mesh Object ---
  MFTL_Object* object = MFTLObjectCreate((int) 0);
  MFTLEnsembleAddObject(ensemble, object);

  double center[3] = {0.35, 0.35, 0.35};

  MFTLGenerateSphere(object, center, 0.15, 5);
  MFTLComputeTriangleProperties(object);

  double length_sum = 0.0;
  for (int i = 0; i < object->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &object->half_edges[i];
    double len = MFTLGetEdgeLength(object, he);
    length_sum += len;
  }

  // average edge length
  double avg_len = length_sum / object->n_half_edges;

  // Setup parameters
  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, avg_len, 10, MFTL_SMOOTH_TRIANGLE_BASED, 45.0);

  // --- Simulation Constants ---
  double T = 3.0;
  double dt = 0.01;
  int step = 0;

  char filename_prefix[256];
  snprintf(filename_prefix, sizeof(filename_prefix), "output_%04d", step);
  MFTLWriteVtu(ensemble, filename_prefix);

  for (double time = 0.0; time < T; time += dt)
  {
    printf("Step %d, Time %f\n", step, time);
    AdvectFront(object, time, dt, T);

    MFTLComputeTriangleProperties(object);

    MFTLRemeshEnsemble(ensemble, &params);

    step++;
    snprintf(filename_prefix, sizeof(filename_prefix), "output_%04d", step);
    MFTLWriteVtu(ensemble, filename_prefix);
  }

  MFTLEnsembleDestroy(&ensemble);
  MPI_Finalize();
  return 0;
}