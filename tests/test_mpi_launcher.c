/*
 * Does `mpiexec -n N` start ONE job of N ranks?
 *
 * A launcher that does not belong to the MPI library the tests are linked
 * against (a second MPI on the PATH, or Ubuntu 24.04's MPICH 4.2.0 package)
 * starts N independent singletons of size 1 instead. Each of them then runs
 * a distributed test as a serial one, and most such tests PASS -- the halo is
 * empty, the reductions are trivially exact -- so the suite reports green
 * while testing nothing distributed. Every MPI test requires this one as a
 * CTest fixture, so a broken launcher fails here, by name, and the rest are
 * not run.
 *
 * Usage: mpiexec -n N test_mpi_launcher N
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int expected = argc > 1 ? atoi(argv[1]) : -1;

  const int ok = size == expected;
  if (!ok)
    printf("MPI launcher: this process sees %d rank(s), %d were requested -- mpiexec does not match the MPI library\n", size, expected);
  else if (rank == 0)
    printf("MPI launcher: %d ranks in one job\n", size);

  MPI_Finalize();
  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
