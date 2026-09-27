/*
 * Manual example ("Open surfaces and boundary loops"): an open, gently curved square patch
 * built from a triangle soup and declared open, with its boundary derived from the mesh
 * (BoundaryFromMesh). Shows the twin states, the boundary loop, the geometry of a boundary edge
 * and the curvature flags of boundary vertices.
 *
 * Run: mpirun -np <N> ./manual_open_surface          (any N >= 1)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_topology.h"

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int rank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  double lo[3] = {-1.0, -1.0, -1.0}, hi[3] = {1.0, 1.0, 1.0};
  bool periodic[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, periodic, 1);

  /* (n+1)^2 vertices on [-0.5,0.5]^2, height z = 0.1 (x^2 + y^2); the soup is given on rank 0 */
  const int n = 12, nv = (n + 1) * (n + 1), nt = 2 * n * n;
  int* vg = malloc((size_t) nv * sizeof(int));
  double(*P)[3] = malloc((size_t) nv * sizeof(*P));
  int* tg = malloc((size_t) nt * sizeof(int));
  int(*tv)[3] = malloc((size_t) nt * sizeof(*tv));
  for (int j = 0; j <= n; ++j)
    for (int i = 0; i <= n; ++i)
    {
      const int v = j * (n + 1) + i;
      vg[v] = v;
      P[v][0] = -0.5 + (double) i / n, P[v][1] = -0.5 + (double) j / n;
      P[v][2] = 0.1 * (P[v][0] * P[v][0] + P[v][1] * P[v][1]);
    }
  int t = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
    {
      const int a = j * (n + 1) + i, b = a + 1, c = a + n + 2, d = a + n + 1;
      tg[t] = t, tv[t][0] = a, tv[t][1] = b, tv[t][2] = c, t++;
      tg[t] = t, tv[t][0] = a, tv[t][1] = c, tv[t][2] = d, t++;
    }

  /* The class is DECLARED: only a body declared open may have boundary edges, and only edges
     the input declares become BOUNDARY (here: every unmatched edge, BoundaryFromMesh). */
  MFTL_Object* patch = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, patch);
  MFTLObjectDeclareOpen(patch, 1);
  const MFTL_BoundaryDecl from_mesh = {1, 0, MFTL_NULL};
  if (MFTLBuildFromSoupEx(patch, rank == 0 ? nv : 0, vg, (const double(*)[3]) P, MFTL_NULL, rank == 0 ? nt : 0, tg, (const int(*)[3]) tv, MFTL_NULL,
                          &from_mesh) != MFTL_SUCCESS)
    MPI_Abort(MPI_COMM_WORLD, 1);
  MFTLComputeTriangleProperties(patch);
  MFTLRedistribute(ens);

  /* Twin states of the stored half-edges: LOCAL, REMOTE (MPI rim) or BOUNDARY (tagged). */
  long cnt[3] = {0, 0, 0}, g[3];
  for (int h = 0; h < patch->n_half_edges; ++h)
  {
    const MFTL_HalfEdge* he = &patch->half_edges[h];
    if (patch->triangles[he->triangle_idx].is_ghost) continue;
    cnt[MFTLHeIsLocal(he) ? 0 : (MFTLHeIsRemote(he) ? 1 : 2)]++;
  }
  MPI_Allreduce(cnt, g, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  const int valid = MFTLValidateTwinStates(patch) == MFTL_SUCCESS; /* collective */

  /* Boundary loops (collective): identical on every rank. */
  MFTL_BoundaryLoopSet loops;
  MFTLBoundaryLoops(patch, &loops);

  /* Geometry of one owned boundary half-edge: tangent t along the half-edge, outward conormal
     nu = t x n (in the surface, pointing away from the patch). */
  double len = 0, tt[3] = {0}, nu[3] = {0};
  int have = 0;
  for (int h = 0; h < patch->n_half_edges && !have; ++h)
    if (MFTLHeIsBoundary(&patch->half_edges[h]) && !patch->triangles[patch->half_edges[h].triangle_idx].is_ghost)
      have = MFTLBoundaryEdgeGeometry(patch, h, &len, tt, nu, MFTL_NULL) == MFTL_SUCCESS;

  /* Curvature is defined on closed fans only: boundary vertices are flagged. */
  MFTLUpdateGhostPositions(ens, MFTL_NULL);
  MFTLComputeVertexNormals(patch);
  MFTLComputeVertexCurvatures(patch);
  long nb = 0, gnb;
  for (int i = 0; i < patch->n_vertices; ++i)
    if (!patch->vertices[i].is_ghost && patch->vertices[i].flags & MFTL_VFLAG_CURV_UNDEFINED) nb++;
  MPI_Allreduce(&nb, &gnb, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);

  double cm[3];
  const double area = MFTLSurfaceCenter(patch, cm); /* defined for open bodies */
  if (rank == 0)
  {
    printf("owned half-edges: %ld LOCAL, %ld REMOTE, %ld BOUNDARY; twin states valid: %d\n", g[0], g[1], g[2], valid);
    printf("%d loop(s): %d edges, tag %d, length %.4f, winding (%d,%d,%d)\n", loops.n_loops, loops.loops[0].n, loops.loops[0].tag, loops.loops[0].length,
           loops.loops[0].winding[0], loops.loops[0].winding[1], loops.loops[0].winding[2]);
    if (have) printf("a boundary edge: length %.4f, t = (%.2f,%.2f,%.2f), nu = (%.2f,%.2f,%.2f)\n", len, tt[0], tt[1], tt[2], nu[0], nu[1], nu[2]);
    printf("boundary vertices flagged CURV_UNDEFINED: %ld (expected %d); area %.6f\n", gnb, 4 * n, area);
  }
  const int ok = valid && loops.n_loops == 1 && loops.loops[0].n == 4 * n && g[2] == 4 * n && gnb == 4 * n;
  MFTLFreeBoundaryLoops(&loops);
  free(vg), free(P), free(tg), free(tv);
  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return ok ? 0 : 1;
}
