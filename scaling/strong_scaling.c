/*
 * @Author: Christian Gorges
 * @Date: 2026-10-13
 *
 * @brief STRONG scaling of the periodic many-sphere advection timestep: the
 *        total problem is fixed, the rank count grows. Ideal: T(P) = T(P0) P0/P.
 *
 * N icospheres sit on a lattice filling a fixed, fully periodic cube and are
 * advected for a fixed number of steps of the documented loop (advect, ghost
 * sync, triangle geometry, redistribute, triangle geometry again, vertex
 * normals). Everything measured, every check and the three motion laws are in
 * scaling_common.h; this file only says how the problem is built.
 *
 * The launch sweep uses full 36-rank nodes: P = 36,72,144,288,576,1080.
 * The default lattice is 48^3 spheres. It divides the default Cartesian grids
 * through P=576 exactly; P=1080 has nonintegral lattice spacings in some rank
 * boxes. Keep the physical problem fixed and report the measured imbalance.
 *
 * What changes with P besides the work per rank -- read it next to the curve:
 *
 *   * the rank box shrinks while the bodies do not, so the halo fraction, the
 *     share of bodies cut by a plane, the share of bodies a step changes, and
 *     the per-step motion as a fraction of a rank box all GROW with P. The report prints all three. That is what
 *     strong scaling is, not an artefact; it is why the communication buckets
 *     fall more slowly than the local-work buckets.
 *   * below 27 ranks (3x3x3) periodic aliasing collapses neighbour offsets onto
 *     the same rank, so the graph degree is below 26 and rises with P. Between
 *     1 and 36 ranks everything also shares one node's memory bandwidth. The
 *     launch script's summary therefore reports efficiency against the smallest
 *     P AND against the largest measured single-node point (36 by default).
 *   * at P = 1 MFTLRedistribute takes a serial short-circuit; that point is the
 *     sequential reference, not a one-rank instance of the same exchange.
 *
 * Usage:
 *   mpirun -n P strong_scaling [--spheres N] [--domain L] [shared options]
 *
 *     --spheres N   total bodies                                (default 110592)
 *     --domain L    edge of the periodic cube    (default cbrt(N): spacing 1)
 *
 *   Example (one point; the sweep is scaling/launch_strong.sh):
 *     mpirun -n 4 build/strong_scaling --spheres 4096 --steps 50 --report strong_np4.txt
 */

#include "scaling_common.h"

/** @brief Everything strong_centre() needs. */
typedef struct
{
  sc_lattice lat;
  double gmin[3];
  double size[3];
  int n_bodies;
  double jitter;
} strong_layout;

/**
 * @brief Start position of body @p body: its slot on the whole-box lattice.
 * @param      ctx  A strong_layout.
 * @param      body Body index.
 * @param[out] out  The centre, wrapped into the box.
 */
static void strong_centre(const void *ctx, int body, double out[3])
{
  const strong_layout *L = (const strong_layout *) ctx;
  double off[3];
  sc_lattice_point(&L->lat, body, L->n_bodies, body, L->jitter, off);
  for (int d = 0; d < 3; ++d)
  {
    double p = L->gmin[d] + off[d];
    while (p < L->gmin[d]) p += L->size[d];
    while (p >= L->gmin[d] + L->size[d]) p -= L->size[d];
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
  sc_options_defaults(&opt, "strong", 200);
  int n_spheres = 110592;
  double edge = 0.0;
  bool bad = false;

  for (int i = 1; i < argc && !bad; ++i)
  {
    const char *a = argv[i];
    if (!strcmp(a, "--spheres") && i + 1 < argc)
      bad = !sc_parse_int(argv[++i], &n_spheres);
    else if (!strcmp(a, "--domain") && i + 1 < argc)
      bad = !sc_parse_double(argv[++i], &edge) || edge <= 0.0;
    else if (!strcmp(a, "--help") || !strcmp(a, "-h"))
    {
      if (rank == 0)
        printf("mpirun -n P strong_scaling [options]\n"
               "  --spheres N         total bodies                       (default 110592)\n"
               "  --domain L          periodic cube edge                 (default cbrt(N))\n"
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
  if (!bad && (n_spheres < 1 || edge < 0.0))
  {
    if (rank == 0) fprintf(stderr, "--spheres must be >= 1 and --domain > 0.\n");
    bad = true;
  }
  if (bad || sc_options_finish(&opt, nproc, rank) != MFTL_SUCCESS)
  {
    if (bad && rank == 0) fprintf(stderr, "Invalid command line; see --help.\n");
    MPI_Finalize();
    return 2;
  }

  MFTLRegisterMpiTypes();

  /* Spacing 1 unless the edge is given: the same density as the weak driver's
   * default, so the two curves describe the same physical problem. */
  if (edge <= 0.0) edge = cbrt((double) n_spheres);
  if (!isfinite(edge * edge * edge) || edge * edge * edge <= 0.0)
  {
    if (rank == 0) fprintf(stderr, "domain volume is outside the supported floating-point range\n");
    MFTLFreeMpiTypes();
    MPI_Finalize();
    return 2;
  }

  strong_layout layout;
  for (int d = 0; d < 3; ++d)
  {
    layout.gmin[d] = 0.0;
    layout.size[d] = edge;
  }
  layout.n_bodies = n_spheres;
  layout.jitter = opt.jitter;
  sc_lattice_build(layout.size, n_spheres, &layout.lat);

  const double vol_per_body = edge * edge * edge / (double) n_spheres;

  sc_problem prob;
  memset(&prob, 0, sizeof prob);
  prob.mode = "strong";
  prob.n_bodies = n_spheres;
  for (int d = 0; d < 3; ++d)
  {
    prob.gmin[d] = 0.0;
    prob.gmax[d] = edge;
  }
  prob.radius = sc_radius(vol_per_body, opt.phi, &layout.lat);
  prob.spacing = cbrt(vol_per_body);
  snprintf(prob.placement, sizeof prob.placement, "%dx%dx%d lattice over the whole box (jitter %.2f cells), phi %.3g", layout.lat.n[0],
           layout.lat.n[1], layout.lat.n[2], opt.jitter, opt.phi);
  prob.centre = strong_centre;
  prob.centre_ctx = &layout;

  const int code = sc_run(&prob, &opt);

  MFTLFreeMpiTypes();
  MPI_Finalize();
  return code;
}
