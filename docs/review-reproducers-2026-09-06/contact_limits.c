#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mftl_core.h"

static void hit(const MFTL_Object *a, int ia, const MFTL_Object *b, int ib, void *data)
{
  (void)a; (void)ia; (void)b; (void)ib;
  ++*(int *)data;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int tiny = argc > 1 && strcmp(argv[1], "tiny") == 0;
  if (size != (tiny ? 1 : 4))
  {
    if (rank == 0) fprintf(stderr, "Use one rank for tiny mode, four ranks otherwise.\n");
    MFTLFreeMpiTypes();
    MPI_Finalize();
    return 2;
  }
  int cuts[3] = {size, 1, 1};
  double lo[3] = {0,0,0}, hi[3] = {tiny ? 1.0 : (double)size,1,1};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, NULL, 2);
  for (int i = 0; i < 2; ++i)
  {
    MFTL_Object *obj = MFTLObjectCreate(i);
    MFTLEnsembleAddObject(ens, obj);
    if (rank == 0)
    {
      double x = tiny ? 0.5 : 0.9 + 1.2*i;
      double s = tiny ? 1e-7 : 0.1;
      double pos[9] = {x,0.4,0.4, x+(tiny?s:0),0.4+s,0.4, x,0.4,0.4+s};
      int tri[3] = {0,1,2};
      MFTLBuildIndexedMesh(obj,pos,3,tri,1);
    }
    obj->next_vertex_gid = 3; obj->next_triangle_gid = 1; obj->next_halfedge_gid = 3;
    obj->global_n_vertices = 3; obj->global_n_triangles = 1;
    MFTLObjectBuildDerived(obj);
    MFTLComputeTriangleGeometry(obj,obj->tris.n);
  }
  MFTL_BVHBuildParams params = MFTL_BVHDefaultParams();
  if (!tiny) { params.proximity_factor = 0; params.proximity_floor = 1.3; }
  int mine = 0, before = 0, after = 0;
  int rc1 = MFTLEnsembleIntersectObjects(ens,&params,hit,&mine);
  MPI_Allreduce(&mine,&before,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  int rr = MFTLRedistribute(ens);
  mine = 0;
  int rc2 = MFTLEnsembleIntersectObjects(ens,&params,hit,&mine);
  MPI_Allreduce(&mine,&after,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (rank == 0) printf("tiny=%d ranks=%d before=%d after=%d rc_before=%d redistribute=%d rc_after=%d\n",tiny,size,before,after,rc1,rr,rc2);
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return before == 1 && after == 1 ? 0 : 1;
}
