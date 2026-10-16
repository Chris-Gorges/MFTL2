/*
 * @Author: Christian Gorges
 * @Date: 2026-10-13
 *
 * @brief WEAK scaling of the periodic many-sphere advection timestep: the work
 *        PER RANK is fixed and the problem grows with P. Ideal: T(P) flat.
 *
 * Every rank box is the same B x B x B cube and starts with exactly M bodies on
 * the same sub-lattice; the periodic domain is B times the process grid. So
 * bodies per rank, the halo fraction, the body size against the rank box and the
 * per-step motion against the rank box are all independent of P. With the
 * default motion (flow, unjittered) every rank box is an exact COPY of every
 * other at every step, so every rank does identical work at every P (the report
 * shows a structural imbalance of 1.000), and the only thing that changes along
 * the curve is P itself -- the machinery, the neighbour count and the network.
 * The loop, the profile, the motion laws and the checks are in scaling_common.h.
 *
 * Growing the body count inside a FIXED domain instead would also keep the work
 * per rank constant, but it makes the problem denser as P grows, and the curve
 * would then mix "the library got slower" with "the problem got harder".
 *
 * Read the curve in three regimes:
 *
 *   * P < 27: periodic aliasing collapses neighbour offsets onto the same rank,
 *     so each rank talks to fewer than 26 distinct neighbours. Communication per
 *     rank still RISES with P here, by construction. The graph degree is in the
 *     report.
 *   * P <= 36: one node; ranks share its memory bandwidth, so even the local
 *     phases can slow down. The summary reports efficiency against one full node.
 *   * P > 36: the interconnect. A flat curve here is the claim under test. Two
 *     terms are expected to grow with P regardless and are reported separately:
 *     the redistribution's O(nproc) `setup` bucket and its per-RANK workspace.
 *
 * Rank 0 generates a bounded batch, distributes only those new bodies, and
 * releases its staging workspace. Setup remains serial in sphere generation
 * but never holds the whole global mesh on rank 0. It is excluded from timing.
 *
 * Usage:
 *   mpirun -n P weak_scaling [--spheres-per-rank M] [--box B] [shared options]
 *
 *     --spheres-per-rank M   bodies per rank box                   (default 1000)
 *     --box B                rank box edge                         (default 10)
 *
 *   Example (one point; the sweep is scaling/launch_weak.sh):
 *     mpirun -n 8 build/weak_scaling --spheres-per-rank 1000 --box 10 --report weak_np8.txt
 */

#include "scaling_common.h"

/** @brief Everything weak_centre() needs. */
typedef struct
{
  sc_lattice lat;  ///< The sub-lattice of ONE rank box.
  double box;
  int cuts[3];
  double size[3];
  int per_rank;
  double jitter;
} weak_layout;

/**
 * @brief Start position of body @p body: rank box `body / M`, slot `body % M`.
 *
 * The box index is a GEOMETRIC cell of the process grid, not an MPI rank, so the
 * placement is independent of how MPI numbers the ranks. With jitter <= 0.5 a
 * centre stays inside its box, so every rank box starts with exactly M centres.
 *
 * @param      ctx  A weak_layout.
 * @param      body Body index.
 * @param[out] out  The centre, wrapped into the domain.
 */
static void weak_centre(const void *ctx, int body, double out[3])
{
  const weak_layout *L = (const weak_layout *) ctx;
  const int c = body / L->per_rank, j = body % L->per_rank;
  const int cell[3] = {c % L->cuts[0], (c / L->cuts[0]) % L->cuts[1], c / (L->cuts[0] * L->cuts[1])};
  double off[3];
  sc_lattice_point(&L->lat, j, L->per_rank, body, L->jitter, off);
  for (int d = 0; d < 3; ++d)
  {
    double p = (double) cell[d] * L->box + off[d];
    while (p < 0.0) p += L->size[d];
    while (p >= L->size[d]) p -= L->size[d];
    out[d] = p;
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  sc_options opt;
  sc_options_defaults(&opt, "weak", 200);
  int per_rank = 1000;
  double box = 10.0;
  bool bad = false;

  for (int i = 1; i < argc && !bad; ++i)
  {
    const char *a = argv[i];
    if (!strcmp(a, "--spheres-per-rank") && i + 1 < argc)
      bad = !sc_parse_int(argv[++i], &per_rank);
    else if (!strcmp(a, "--box") && i + 1 < argc)
      bad = !sc_parse_double(argv[++i], &box);
    else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
    {
      if (rank == 0)
        printf("mpirun -n P weak_scaling [options]\n"
               "  --spheres-per-rank M  bodies per rank box               (default 1000)\n"
               "  --box B               rank box edge                     (default 10)\n"
               "%s",
               SC_COMMON_USAGE);
      MPI_Finalize();
      return 0;
    }
    else
    {
      const int r = sc_parse_common(argc, argv, &i, &opt);
      if (r != 1)
      {
        if (rank == 0) fprintf(stderr, "%s option: %s (see --help)\n", (r < 0) ? "Invalid or incomplete" : "Unknown", a);
        bad = true;
      }
    }
  }
  if (!bad && (per_rank < 1 || box <= 0.0 || (long) per_rank * nproc > 2000000000L))
  {
    if (rank == 0) fprintf(stderr, "--spheres-per-rank must be >= 1 (and M*P must fit an int), --box > 0.\n");
    bad = true;
  }
  if (!isfinite(box * box * box * nproc) || box * box * box <= 0.0) bad = true;
  if (bad || sc_options_finish(&opt, nproc, rank) != MFTL_SUCCESS)
  {
    if (bad && rank == 0) fprintf(stderr, "Invalid command line; see --help.\n");
    MPI_Finalize();
    return 2;
  }

  MFTLRegisterMpiTypes();

  weak_layout layout;
  layout.box = box;
  layout.per_rank = per_rank;
  layout.jitter = opt.jitter;
  for (int d = 0; d < 3; ++d)
  {
    layout.cuts[d] = opt.cuts[d];
    layout.size[d] = box * (double) opt.cuts[d];
  }
  const double cube[3] = {box, box, box};
  sc_lattice_build(cube, per_rank, &layout.lat);

  const double vol_per_body = box * box * box / (double) per_rank;

  sc_problem prob;
  memset(&prob, 0, sizeof prob);
  prob.mode = "weak";
  prob.n_bodies = per_rank * nproc;
  for (int d = 0; d < 3; ++d)
  {
    prob.gmin[d] = 0.0;
    prob.gmax[d] = layout.size[d];
  }
  prob.radius = sc_radius(vol_per_body, opt.phi, &layout.lat);
  prob.spacing = cbrt(vol_per_body);
  snprintf(prob.placement, sizeof prob.placement, "exactly %d per rank box on a %dx%dx%d sub-lattice (jitter %.2f cells), phi %.3g", per_rank,
           layout.lat.n[0], layout.lat.n[1], layout.lat.n[2], opt.jitter, opt.phi);
  prob.centre = weak_centre;
  prob.centre_ctx = &layout;

  const int code = sc_run(&prob, &opt);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return code;
}
