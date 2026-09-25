#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "mftl_core.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size != 6) { MPI_Finalize(); return 2; }
  int cuts[3] = {3, 2, 1};
  double lo[3] = {0, 0, 0}, hi[3] = {3, 2, 1};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, NULL, 1);
  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);
  enum { S = 20, N = 3 * S + 1 };
  const int closed = argc > 1, vp = closed ? 4 : 2;
  const int NV = vp * N, NT = closed ? 8 * (N-1) + 4 : 2 * (N-1);
  if (rank == 0)
  {
    double c[N][2], pos[3 * NV];
    int tri[3 * NT];
    for (int j = 0; j < N; ++j)
    {
      if (j <= S) { c[j][0] = 0.2 + 1.3 * j / S; c[j][1] = 0.2; }
      else if (j <= 2*S) { c[j][0] = 1.5; c[j][1] = 0.2 + 1.3 * (j-S) / S; }
      else { c[j][0] = 1.5 + 1.3 * (j-2*S) / S; c[j][1] = 1.5; }
    }
    for (int j = 0; j < N; ++j)
    {
      int a = j ? j-1 : j, b = j+1 < N ? j+1 : j;
      double dx = c[b][0]-c[a][0], dy = c[b][1]-c[a][1];
      double len = hypot(dx,dy);
      for (int k = 0; k < vp; ++k)
      {
        double w = closed ? ((k == 0 || k == 3) ? 0.01 : -0.01) : (k ? 0.01 : -0.01);
        pos[3*(vp*j+k)] = c[j][0] - w * dy / len;
        pos[3*(vp*j+k)+1] = c[j][1] + w * dx / len;
        pos[3*(vp*j+k)+2] = 0.5 + (closed ? (k < 2 ? 0.01 : -0.01) : 0.0);
      }
    }
    int tcur = 0;
    for (int j = 0; j < N-1; ++j)
    {
      for (int k = 0; k < (closed ? 4 : 1); ++k)
      {
        int a = vp*j+k, b = vp*(j+1)+k;
        int cidx = vp*j+(k+1)%vp, d = vp*(j+1)+(k+1)%vp;
        tri[tcur++] = a; tri[tcur++] = b; tri[tcur++] = cidx;
        tri[tcur++] = cidx; tri[tcur++] = b; tri[tcur++] = d;
      }
    }
    if (closed)
    {
      int end = 4*(N-1);
      int caps[12] = {0,1,2, 0,2,3, end,end+2,end+1, end,end+3,end+2};
      for (int i = 0; i < 12; ++i) tri[tcur++] = caps[i];
    }
    int rc = MFTLBuildIndexedMesh(obj, pos, NV, tri, NT);
    printf("connected %s: vertices=%d triangles=%d build=%d integrity=%d\n", closed ? "closed tube" : "strip", NV, NT, rc, MFTLCheckMeshIntegrity(obj));
  }
  obj->next_vertex_gid = NV; obj->next_triangle_gid = NT; obj->next_halfedge_gid = 3*NT;
  obj->global_n_vertices = NV; obj->global_n_triangles = NT;
  MFTLObjectBuildDerived(obj);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  int rc = MFTLRedistribute(ens);
  int fit = MFTLEnsembleBodyHoldersFitTheGraph(ens), allfit;
  MPI_Allreduce(&fit, &allfit, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  obj = MFTLEnsembleFindLocalObject(ens, 0);
  double automatic = obj ? obj->tris.n_local : 0.0;
  double global = automatic;
  MFTL_BodyReduceOp sum = MFTL_BODY_REDUCE_SUM;
  MFTLEnsembleReduceBodies(ens, &automatic, 1, &sum, MFTL_BODY_REDUCE_SCOPE_AUTO);
  MFTLEnsembleReduceBodies(ens, &global, 1, &sum, MFTL_BODY_REDUCE_SCOPE_GLOBAL);
  double local = obj ? obj->tris.n_local : 0.0, expected_prefix = 0.0;
  double prefix = local, total = 0.0;
  MPI_Exscan(&local, &expected_prefix, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (rank == 0) expected_prefix = 0.0;
  MFTLEnsembleExscanBodies(ens, &prefix, &total, 1, MFTL_BODY_REDUCE_SCOPE_AUTO);
  if (obj) printf("rank=%d redistribute=%d graph_fit=%d all_fit=%d local_tri=%d auto_sum=%.0f global_sum=%.0f auto_prefix=%.0f expected_prefix=%.0f\n", rank, rc, fit, allfit, obj->tris.n_local, automatic, global, prefix, expected_prefix);
  int failed = obj && (automatic != global || prefix != expected_prefix || total != global), anyfailed;
  MPI_Allreduce(&failed, &anyfailed, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return anyfailed ? 1 : 0;
}
