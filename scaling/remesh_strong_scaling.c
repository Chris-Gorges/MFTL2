/*
 * @Author: Christian Gorges
 * @Date: 2026-11-06
 *
 * @brief STRONG scaling of a remeshed front-tracking timestep: one fixed
 *        problem, varied rank count. Ideal: T(P) ~ 1/P.
 *
 * N x N x N bodies on a lattice of spacing 1 in a periodic box of edge N, in the
 * travelling strain flow of remesh_common.h, remeshed every step. The regime is
 * chosen, not derived, and stays the same along the curve while the rank boxes
 * shrink:
 *
 *   --regime dispersed (default N = 18, 5832 droplets x 1280 triangles, R = 0.25):
 *     R/h grows from 0.06 at 1 node to 0.17 at 30 nodes, so every body stays
 *     much smaller than a rank box and N_bodies stays above P. The DISPERSED
 *     driver's answer is the serial answer to the bit and the rest of the step
 *     is partition-independent, so the final digest of every point of the sweep
 *     must be the same; collect_remesh.sh checks it.
 *
 *   --regime spanning (default N = 2, 8 bubbles x 327 680 triangles, R = 0.3):
 *     R/h grows from 0.6 at 1 node to 1.8 at 30 nodes. Each body sits on a
 *     lattice point at the corner of the box, so on any grid that divides N it
 *     is cut through its centre by three planes. Both spanning modes run, one
 *     after the other, in the same job; their accuracy columns are comparable
 *     across P and with each other, their digests are not (a placement order
 *     that depends on the grid is the definition of spanning).
 *
 * Usage:
 *   mpirun -n P remesh_strong_scaling [--per-axis N] [shared options]
 *
 *     --per-axis N   bodies per axis    (default 18 dispersed, 2 spanning)
 *
 *   Example (one point; the sweep is scaling/launch_remesh_strong.sh):
 *     mpirun -n 8 build/remesh_strong_scaling --regime dispersed --per-axis 6 --steps 40
 */

#include "remesh_common.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  rm_options opt;
  rm_options_defaults(&opt, "remesh_strong");
  int per_axis = 0;
  bool bad = false;
  for (int i = 1; i < argc && !bad; ++i)
  {
    const char *a = argv[i];
    if (!strcmp(a, "--per-axis") && i + 1 < argc)
      bad = !sc_parse_int(argv[++i], &per_axis);
    else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
    {
      if (rank == 0)
        printf(
            "mpirun -n P remesh_strong_scaling [options]\n"
            "  --per-axis N        bodies per axis                 (default 18 dispersed, 2 spanning)\n"
            "%s",
            RM_COMMON_USAGE);
      MPI_Finalize();
      return 0;
    }
    else
    {
      const int r = rm_parse_common(argc, argv, &i, &opt);
      if (r != 1)
      {
        if (rank == 0) fprintf(stderr, "%s option: %s (see --help)\n", (r < 0) ? "Invalid or incomplete" : "Unknown", a);
        bad = true;
      }
    }
  }
  if (per_axis <= 0) per_axis = (opt.regime == RM_SPANNING) ? 2 : 18;
  if (!bad && (per_axis > 1000)) bad = true;
  if (bad || rm_options_finish(&opt, nproc, rank) != MFTL_SUCCESS)
  {
    if (bad && rank == 0) fprintf(stderr, "Invalid command line; see --help.\n");
    MPI_Finalize();
    return 2;
  }

  MFTLRegisterMpiTypes();

  rm_problem prob;
  memset(&prob, 0, sizeof prob);
  prob.mode = "remesh_strong";
  prob.regime = opt.regime;
  prob.n_bodies = per_axis * per_axis * per_axis;
  prob.lat.a = 1.0;
  for (int d = 0; d < 3; ++d)
  {
    prob.lat.n[d] = per_axis;
    /* Dispersed: in the middle of a lattice cell. Spanning: on the lattice
     * point, i.e. on a rank-box corner for every grid that divides N. */
    prob.lat.origin[d] = (opt.regime == RM_SPANNING) ? 0.0 : 0.5;
    prob.gmin[d] = 0.0;
    prob.gmax[d] = (double) per_axis;
  }
  prob.radius = opt.radius;
  snprintf(prob.placement, sizeof prob.placement, "%d^3 bodies of radius %.3g on a lattice of spacing 1, %s", per_axis, opt.radius,
           (opt.regime == RM_SPANNING) ? "centred on the lattice points (rank-box corners where the grid divides N)" : "centred in the lattice cells");

  const int code = rm_run(&prob, &opt);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return code;
}
