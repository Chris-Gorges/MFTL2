/*
 * Manual example ("Phase sets"): two closed films z = 0.3 and z = 0.7 that wrap the periodic x
 * and y directions of a triply periodic unit box. Neither film encloses a volume by itself (a
 * single wrapping film: UNDEFINED), but together they bound the slab 0.3 < z < 0.7: the phase
 * set of both films is DEFINED with V = 0.4. Queries return a status and give a value only
 * when the status is MFTL_DEFINED.
 *
 * Run: mpirun -np <N> ./manual_phase_set             (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_phase_set.h"
#include "mftl_topology.h"

/* closed film at height z0 spanning x and y; up = 1: normals +z, else -z (triangles on rank 0) */
static MFTL_Object* film(MFTL_Ensemble* ens, int gid, double z0, int up, int rank)
{
  const int N = 8;
  int vg[N * N], tg[2 * N * N], tv[2 * N * N][3];
  double P[N * N][3];
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
    {
      const int v = j * N + i, a = v, b = j * N + (i + 1) % N, c = ((j + 1) % N) * N + (i + 1) % N, q = ((j + 1) % N) * N + i;
      vg[v] = v, P[v][0] = (double) i / N, P[v][1] = (double) j / N, P[v][2] = z0;
      tg[2 * v] = 2 * v, tg[2 * v + 1] = 2 * v + 1;
      tv[2 * v][0] = a, tv[2 * v][1] = up ? b : c, tv[2 * v][2] = up ? c : b;
      tv[2 * v + 1][0] = a, tv[2 * v + 1][1] = up ? c : q, tv[2 * v + 1][2] = up ? q : c;
    }
  MFTL_Object* o = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, o);
  MFTLBuildFromSoup(o, rank == 0 ? N * N : 0, vg, (const double(*)[3]) P, MFTL_NULL, rank == 0 ? 2 * N * N : 0, tg, (const int(*)[3]) tv, MFTL_NULL);
  MFTLComputeTriangleProperties(o);
  return o;
}

static const char* name(MFTL_PhaseStatus s) { return s == MFTL_DEFINED ? "DEFINED" : (s == MFTL_UNDEFINED ? "UNDEFINED" : "UNSUPPORTED"); }

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool periodic[3] = {true, true, true};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 2);

  /* sigma = +1: the phase lies on the side into which the stored normals do NOT point. The
     lower film points down (-z) and the upper film up (+z), so the phase is between them. */
  MFTL_Object* lower = film(ens, 0, 0.3, 0, rank);
  MFTL_Object* upper = film(ens, 1, 0.7, 1, rank);
  MFTLRedistribute(ens);

  /* A single wrapping film: the flux Phi_z = -1 (one face area) is not zero -> UNDEFINED. */
  MFTL_PhaseSet* one = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(one, upper, +1);
  double V1 = -1.0;
  MFTL_PhaseSetInfo info;
  const MFTL_PhaseStatus s1 = MFTLPhaseSetVolume(one, &V1);
  MFTLPhaseSetClassify(one, &info);
  MFTLPhaseSetDestroy(&one);

  /* Both films: closed, Phi = 0, consistently oriented -> DEFINED. */
  MFTL_PhaseSet* slab = MFTLPhaseSetCreate(&ens->domain);
  MFTLPhaseSetAddMember(slab, lower, +1);
  MFTLPhaseSetAddMember(slab, upper, +1);
  double V2 = -1.0, C[3];
  MFTL_PhaseStatus cst[3];
  const MFTL_PhaseStatus s2 = MFTLPhaseSetVolume(slab, &V2);
  MFTLPhaseSetCentre(slab, C, cst);
  const double qin[3] = {0.5, 0.5, 0.5}, qout[3] = {0.5, 0.5, 0.9};
  int w_in = -1, w_out = -1;
  MFTLPhaseSetInside(slab, qin, &w_in);
  MFTLPhaseSetInside(slab, qout, &w_out);
  MFTLPhaseSetDestroy(&slab);

  if (rank == 0)
  {
    printf("upper film alone: %s (%s)\n", name(s1), info.reason);
    printf("slab: %s, V = %.12f (expected 0.4); centre z %s = %.6f, x %s\n", name(s2), V2, name(cst[2]), C[2], name(cst[0]));
    printf("inside(0.5,0.5,0.5) = %d, inside(0.5,0.5,0.9) = %d (1 inside, 0 outside, 2 on boundary)\n", w_in, w_out);
  }
  const int ok = s1 == MFTL_UNDEFINED && s2 == MFTL_DEFINED && fabs(V2 - 0.4) < 1e-12 && w_in == MFTL_PHASE_INSIDE && w_out == MFTL_PHASE_OUTSIDE;
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}
