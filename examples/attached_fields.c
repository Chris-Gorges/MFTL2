/**
 * @file attached_fields.c
 * @brief Optional fields on a stationary sphere: measure remeshing mass and profile error separately.
 *
 * Run `mpirun -n 4 attached_fields [refinement=2] [passes=6] [mode=2]`.
 * Mode 0 registers nothing, 1 registers recomputed storage, 2 carries persistent
 * triangle concentration and vertex samples. Mode 3 carries only the triangle
 * scalar, for measuring its cost. There is no physical advection.
 * The alternating target sizes deliberately exercise repeated mesh changes.
 * This is a transfer diagnostic, not a surface transport solver.
 */
#include "mftl_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/** @brief Smooth, positive blob centred at the sphere's north pole.
 * @param p Point on or inside the sphere.
 * @return Reference concentration.
 */
static double profile(const double p[3])
{
  const double r = sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
  return .05 + exp(8.0 * (p[2] / r - 1.0));
}

/** @brief Fill initial or recomputed values from current geometry.
 * @param ens Distributed mesh.
 * @param concentration Triangle field id.
 * @param samples Vertex field id, or -1 for none.
 */
static void fill(MFTL_Ensemble *ens, int concentration, int samples)
{
  for (int j = 0; j < ens->n_local_objects; ++j)
  {
    MFTL_Object *obj = ens->local_objects[j];
    double *c = MFTLObjectField(obj, concentration), *v = MFTLObjectField(obj, samples);
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
    for (int t = 0; t < obj->tris.n_local; ++t) c[t] = profile(obj->tris.center + 3 * t);
    if (v)
      for (int i = 0; i < obj->verts.n_local; ++i)
        for (int k = 0; k < 3; ++k) v[3 * i + k] = obj->verts.pos[3 * i + k];
  }
}

/** @brief Integrate owned triangles only.
 * @param ens Distributed mesh.
 * @param id Concentration id.
 * @param out Mass, area-weighted absolute profile error, and first moment.
 */
static void measure(MFTL_Ensemble *ens, int id, double out[5])
{
  double local[5] = {0};
  for (int j = 0; j < ens->n_local_objects; ++j)
  {
    MFTL_Object *obj = ens->local_objects[j];
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
    const double *c = MFTLObjectField(obj, id);
    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      const double a = MFTLGetTriangleArea(obj, t);
      const double *p = obj->tris.center + 3 * t;
      local[0] += a * c[t];
      local[1] += a * fabs(c[t] - profile(p));
      for (int k = 0; k < 3; ++k) local[2 + k] += a * c[t] * p[k];
    }
  }
  MPI_Allreduce(local, out, 5, MPI_DOUBLE, MPI_SUM, ens->domain.comm_world);
}

/** @brief Run the field example and print one CSV record.
 * @param argc Argument count.
 * @param argv Optional refinement, pass count and mode.
 * @return Zero on success.
 */
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  int rank, np;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  const int refine = argc > 1 ? atoi(argv[1]) : 2, passes = argc > 2 ? atoi(argv[2]) : 6, mode = argc > 3 ? atoi(argv[3]) : 2;
  if (refine < 1 || refine > 7 || passes < 1 || mode < 0 || mode > 3)
  {
    if (rank == 0) fprintf(stderr, "Usage: attached_fields [refinement 1..7] [passes >=1] [mode 0..3]\n");
    MFTLFreeMpiTypes();
    MPI_Finalize();
    return 1;
  }
  int cuts[3] = {0};
  MPI_Dims_create(np, 3, cuts);
  double lo[3] = {-1, -1, -1}, hi[3] = {1, 1, 1}, center[3] = {0, 0, 0};
  bool periodic[3] = {true, true, true};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);
  int concentration = -1, samples = -1;
  if (mode)
  {
    concentration =
        MFTLEnsembleRegisterField(ens, "concentration", MFTL_FIELD_TRIANGLE, 1, mode >= 2 ? MFTL_FIELD_CONSERVATIVE_DENSITY : MFTL_FIELD_RECOMPUTED);
    if (mode != 3) samples = MFTLEnsembleRegisterField(ens, "samples", MFTL_FIELD_VERTEX, 3, mode == 2 ? MFTL_FIELD_INTERPOLATED : MFTL_FIELD_RECOMPUTED);
  }
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  MFTLGenerateSphere(obj, center, .6, refine);
  if (mode) fill(ens, concentration, samples);
  MFTLRedistribute(ens);
  for (int j = 0; j < ens->n_local_objects; ++j) MFTLComputeTriangleGeometry(ens->local_objects[j], ens->local_objects[j]->tris.n);
  if (mode == 1) fill(ens, concentration, samples);
  const int ids[2] = {concentration, samples};
  const int nfields = mode == 3 ? 1 : 2;
  if (mode) MFTLUpdateGhostFields(ens, ids, nfields, false);
  double initial[5] = {0}, final[5] = {0};
  if (mode) measure(ens, concentration, initial);
  MFTL_RemeshParameters p;
  const double target = .66 / ldexp(1.0, refine);
  MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, target, 2, MFTL_SMOOTH_TRIANGLE_BASED, 180);
  int ok = 1;
  long operations = 0;
  double migrations = 0.0;
  MPI_Barrier(MPI_COMM_WORLD);
  const double start = MPI_Wtime();
  for (int step = 0; step < passes; ++step)
  {
    MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, target * (step % 2 ? 1.2 : .8), 2, MFTL_SMOOTH_TRIANGLE_BASED, 180);
    MFTL_RemeshStats st;
    ok &= MFTLRemeshEnsemble(ens, &p, &st) == MFTL_SUCCESS;
    operations += st.n_split + st.n_collapse + st.n_flip;
    migrations += st.t_migrate;
    if (mode == 1) fill(ens, concentration, samples);
    if (mode) ok &= MFTLUpdateGhostFields(ens, ids, nfields, false) == MFTL_SUCCESS;
  }
  double elapsed = MPI_Wtime() - start, seconds;
  MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (mode) measure(ens, concentration, final);
  double drift = 0;
  if (mode)
    for (int k = 0; k < 3; ++k)
    {
      const double d = final[2 + k] / final[0] - initial[2 + k] / initial[0];
      drift += d * d;
    }
  long allops;
  MPI_Reduce(&operations, &allops, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (mode >= 2) ok &= isfinite(final[0]) && fabs(final[0] - initial[0]) < 1e-10 * initial[0];
  int allok;
  MPI_Allreduce(&ok, &allok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (rank == 0)
  {
    puts("ranks,refinement,passes,mode,operations,seconds,migration_seconds,relative_mass_error,relative_L1_error,centroid_drift");
    printf("%d,%d,%d,%d,%ld,%.9g,%.9g,%.9g,%.9g,%.9g\n", np, refine, passes, mode, allops, seconds, migrations, mode ? fabs(final[0] - initial[0]) / initial[0] : 0.0,
           mode ? final[1] / initial[0] : 0.0, sqrt(drift));
  }
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return allok ? 0 : 1;
}
