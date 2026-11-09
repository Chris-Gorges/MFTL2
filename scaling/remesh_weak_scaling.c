/*
 * @Author: Christian Gorges
 * @Date: 2026-11-06
 *
 * @brief WEAK scaling of a remeshed front-tracking timestep: the work PER RANK
 *        is fixed and the problem grows with P. Ideal: T(P) flat.
 *
 * Every rank box is the same unit cube and the periodic domain is the process
 * grid, so bodies per rank, R/h and the motion against the rank box are
 * independent of P. The flow (remesh_common.h) gives every body the same
 * history, and the lattice is aligned to the rank boxes, so every rank box is a
 * COPY of every other at every step: what changes along the curve is P itself.
 *
 *   --regime dispersed (default q = 3: 27 droplets per rank x 1280 triangles):
 *     q x q x q bodies per rank box, spacing 1/q, R = 0.25/q, R/h = 0.083.
 *
 *   --regime spanning (default m = 2: one bubble per 2 x 2 x 2 rank boxes, 327 680
 *     triangles, i.e. 40 960 per rank): spacing m, R = 0.3 m, R/h = 0.6. Each
 *     body is centred on a rank-box corner and cut through its centre by three
 *     planes, so it needs every cut count divisible by m. Without --cuts the
 *     most cubic such grid is chosen; with 36 cores per node there is none (36
 *     has only 2^2), so the sweep runs 32 ranks per node by default
 *     (launch_remesh_weak.sh). Both spanning modes run in the same job.
 *
 * Usage:
 *   mpirun -n P remesh_weak_scaling [--per-axis q | --boxes-per-body m] [shared options]
 *
 *     --per-axis q         dispersed: bodies per rank box per axis   (default 3)
 *     --boxes-per-body m   spanning: rank boxes per body per axis    (default 2)
 *
 *   Example (one point; the sweep is scaling/launch_remesh_weak.sh):
 *     mpirun -n 8 build/remesh_weak_scaling --regime spanning --refine 5 --steps 40
 */

#include "remesh_common.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  rm_options opt;
  rm_options_defaults(&opt, "remesh_weak");
  int per_axis = 3, boxes = 2;
  bool bad = false;
  for (int i = 1; i < argc && !bad; ++i)
  {
    const char *a = argv[i];
    if (!strcmp(a, "--per-axis") && i + 1 < argc)
      bad = !sc_parse_int(argv[++i], &per_axis);
    else if (!strcmp(a, "--boxes-per-body") && i + 1 < argc)
      bad = !sc_parse_int(argv[++i], &boxes);
    else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
    {
      if (rank == 0)
        printf(
            "mpirun -n P remesh_weak_scaling [options]\n"
            "  --per-axis q        dispersed: bodies per rank box per axis      (default 3)\n"
            "  --boxes-per-body m  spanning: rank boxes per body per axis       (default 2)\n"
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
  if (!bad && (per_axis < 1 || per_axis > 64 || boxes < 1 || boxes > 64)) bad = true;
  const bool span = (opt.regime == RM_SPANNING);
  if (!bad && span && !opt.sc.cuts_given)
  {
    if (!rm_divisible_cuts(nproc, boxes, opt.sc.cuts))
    {
      if (rank == 0)
        fprintf(stderr,
                "no process grid of %d ranks has every axis divisible by --boxes-per-body %d; use a multiple of %d ranks (e.g. 32 per node) or --cuts.\n",
                nproc, boxes, boxes * boxes * boxes);
      MPI_Finalize();
      return 2;
    }
    opt.sc.cuts_given = true;
  }
  if (bad || rm_options_finish(&opt, nproc, rank) != MFTL_SUCCESS)
  {
    if (bad && rank == 0) fprintf(stderr, "Invalid command line; see --help.\n");
    MPI_Finalize();
    return 2;
  }
  if (span && (opt.sc.cuts[0] % boxes || opt.sc.cuts[1] % boxes || opt.sc.cuts[2] % boxes))
  {
    if (rank == 0)
      fprintf(stderr, "--cuts %d %d %d: every axis must be divisible by --boxes-per-body %d.\n", opt.sc.cuts[0], opt.sc.cuts[1], opt.sc.cuts[2], boxes);
    MPI_Finalize();
    return 2;
  }

  MFTLRegisterMpiTypes();

  rm_problem prob;
  memset(&prob, 0, sizeof prob);
  prob.mode = "remesh_weak";
  prob.regime = opt.regime;
  prob.lat.a = span ? (double) boxes : 1.0 / (double) per_axis;
  prob.n_bodies = 1;
  for (int d = 0; d < 3; ++d)
  {
    prob.lat.n[d] = span ? opt.sc.cuts[d] / boxes : opt.sc.cuts[d] * per_axis;
    prob.lat.origin[d] = span ? 0.0 : 0.5 * prob.lat.a;
    prob.gmin[d] = 0.0;
    prob.gmax[d] = (double) opt.sc.cuts[d];  // unit rank boxes
    prob.n_bodies *= prob.lat.n[d];
  }
  prob.radius = opt.radius * prob.lat.a;
  if (span)
    snprintf(prob.placement, sizeof prob.placement, "one body of radius %.3g per %d^3 unit rank boxes, centred on a rank-box corner", prob.radius, boxes);
  else
    snprintf(prob.placement, sizeof prob.placement, "%d^3 bodies of radius %.3g per unit rank box, centred in the cells of a %d^3 sub-lattice", per_axis,
             prob.radius, per_axis);

  const int code = rm_run(&prob, &opt);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return code;
}
