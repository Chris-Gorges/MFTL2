/*
 * The smallest program a solver would write against MFTL: a distributed
 * ensemble, one generated sphere, a redistribution and the body's integrals,
 * checked against the sphere's area and volume. It is compiled as C (main.c)
 * and as C++ (main.cpp includes this file), so it checks that the installed
 * headers and package work from both languages.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_core.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  MFTLRegisterMpiTypes(); /* the wire datatypes every exchange uses */

  int cuts[3] = {0, 0, 0};
  MPI_Dims_create(nproc, 3, cuts);
  double lo[3] = {-2.0, -2.0, -2.0}, hi[3] = {2.0, 2.0, 2.0};
  bool periodic[3] = {false, false, false};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 4);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);

  const double radius = 1.0;
  double centre[3] = {0.0, 0.0, 0.0};
  int ok = MFTLGenerateSphere(obj, centre, radius, 4) == MFTL_SUCCESS;
  obj = MFTL_NULL; /* a redistribution may destroy it; the handle that survives is the global id */
  ok = ok && MFTLRedistribute(ens) == MFTL_SUCCESS;

  /* Every holder of the body reduces to the same integrals; count it once. */
  double area = 0.0, volume = 0.0;
  MFTL_BodyIntegrals *integrals = (MFTL_BodyIntegrals *) malloc((size_t) (ens->n_local_objects + 1) * sizeof(MFTL_BodyIntegrals));
  ok = ok && MFTLEnsembleBodyIntegrals(ens, integrals) == MFTL_SUCCESS;
  for (int i = 0; ok && i < ens->n_local_objects; ++i)
    if (integrals[i].lowest_holder == rank)
    {
      area += integrals[i].area;
      volume += integrals[i].volume;
    }
  free(integrals);

  double sums[2] = {area, volume};
  MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  int all_ok = ok;
  MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

  /* A 5120-triangle icosphere is inscribed: both integrals lie a little below
   * the sphere's, by well under one per cent. */
  const double area_exact = 4.0 * MFTL_PI * radius * radius;
  const double volume_exact = 4.0 / 3.0 * MFTL_PI * radius * radius * radius;
  const double area_err = fabs(sums[0] - area_exact) / area_exact;
  const double volume_err = fabs(sums[1] - volume_exact) / volume_exact;
  const int pass = all_ok && area_err < 1e-2 && volume_err < 1e-2;

  if (rank == 0)
    printf("%s: %d rank(s), grid %dx%dx%d, area %.6f (rel. err %.2e), volume %.6f (rel. err %.2e)\n", pass ? "PASS" : "FAIL", nproc, cuts[0], cuts[1], cuts[2],
           sums[0], area_err, sums[1], volume_err);

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
