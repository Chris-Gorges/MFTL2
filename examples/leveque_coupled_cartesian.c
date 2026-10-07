/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief LeVeque's deformation test with the velocity known ONLY on a
 *        structured Cartesian grid, reaching the interface through the coupler.
 *
 * The solver is a staggered (MAC) grid of N^3 cells -- u on the x-faces, v on
 * the y-faces, w on the z-faces, each component its own location -- or, with
 * `-layout collocated`, three components at the cell centres. Its
 * decomposition is the solver's own: z-slabs in rank order by default, which
 * agree with MFTL's process grid only at np = 1, so the coupler has to find the
 * owners of every sample through its directory; `-solver aligned` uses MFTL's
 * boxes with two valid ghost layers, so most of the band is read in place.
 *
 * The interface is advected with RK4 through the coupler (examples/
 * leveque_coupling_common.h): stage binds at the stage positions, one fetch per
 * distinct stage time, Peskin's 4-point kernel with linear moment correction
 * at the walls. It is compared with the same RK4 on the exact field:
 *
 *   PASS A  without remeshing, every vertex against its own exact trajectory,
 *           for each grid of the sweep -- position error, interpolation error,
 *           volume, and the observed order of the position error in h.
 *   PASS B  remeshed every step on the finest grid, the way a front tracker
 *           runs: volume, area and distance to the exact sphere at t = T for
 *           the coupled and the direct run side by side.
 *
 * USAGE
 *
 *   mpirun -n P leveque_coupled_cartesian [options]
 *
 *     -n LIST        grids of the sweep, comma-separated  (default 16,32,64)
 *     -refine R      sphere subdivision level            (default 3)
 *     -T time        period                              (default 3)
 *     -dt step       timestep                            (default 0.02)
 *     -steps S       stop after S steps                  (default: all of T)
 *     -layout L      mac | collocated                    (default mac)
 *     -solver D      slab | aligned                      (default slab)
 *     -kernel K      peskin4 | bspline3 | cosine4 | roma3 (default peskin4)
 *     -pass P        a | b | ab                          (default ab)
 *     -verify        exit non-zero unless the errors converge and stay small
 */

#include "leveque_coupling_common.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The Cartesian solver
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int N;
  bool mac;
  int ghost;
  int loc[3];
  int64_t lo[3][3], hi[3][3], alo[3][3], adims[3][3], n_el[3];
  double *spatial[3], *value[3];
  double *ptr[3];
  bool has[3];
} cart;

/** @brief Owned index range of component @p comp along axis @p k for this rank. */
static void owned_range(const cart *C, bool aligned, int comp, int k, int64_t *lo, int64_t *hi)
{
  const int np = lv_nproc(), rank = lv_rank();
  const int64_t n_cells = C->N;
  const bool faces = C->mac && comp == k; /* this axis carries N+1 face samples */
  int cuts[3], co[3];
  lv_grid_for(np, cuts);
  co[2] = rank % cuts[2];
  co[1] = (rank / cuts[2]) % cuts[1];
  co[0] = rank / (cuts[1] * cuts[2]);
  int parts = 1, me = 0;
  if (aligned)
  {
    parts = cuts[k];
    me = co[k];
  }
  else if (k == 2)
  {
    parts = np;
    me = rank;
  }
  *lo = n_cells * me / parts;
  *hi = n_cells * (me + 1) / parts;
  if (faces && me == parts - 1) *hi = n_cells + 1; /* the last part also owns the closing face */
}

static void cart_set_time(lv_solver *s, double t)
{
  cart *C = (cart *) s->impl;
  const double f = lv_time_factor(t, s->T);
  for (int c = 0; c < 3; ++c)
  {
    for (int64_t e = 0; e < C->n_el[c]; ++e) C->value[c][e] = C->spatial[c][e] * f;
    s->field[c].version = ++s->version;
  }
}

static void cart_build(lv_solver *s, int N, bool mac, bool aligned, int ghost, double T)
{
  cart *C = (cart *) calloc(1, sizeof(cart));
  C->N = N;
  C->mac = mac;
  C->ghost = ghost;
  const double h = 1.0 / (double) N;
  s->grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  s->impl = C;
  s->T = T;
  s->set_time = cart_set_time;
  const char *names[3] = {"u", "v", "w"};
  for (int c = 0; c < 3; ++c)
  {
    if (mac || c == 0)
    {
      MFTL_LocationDesc ld;
      memset(&ld, 0, sizeof ld);
      ld.name = mac ? names[c] : "cells";
      ld.lattice.kind = MFTL_LATTICE_UNIFORM;
      for (int k = 0; k < 3; ++k)
      {
        const bool faces = mac && c == k;
        ld.lattice.dims[k] = faces ? N + 1 : N;
        ld.lattice.spacing[k] = h;
        ld.lattice.origin[k] = faces ? 0.0 : 0.5 * h;
      }
      C->loc[c] = MFTLSolverGridAddLocation(s->grid, &ld);
    }
    else
      C->loc[c] = C->loc[0];
    bool empty = false;
    C->n_el[c] = 1;
    for (int k = 0; k < 3; ++k)
    {
      owned_range(C, aligned, c, k, &C->lo[c][k], &C->hi[c][k]);
      if (C->hi[c][k] <= C->lo[c][k]) empty = true;
      C->alo[c][k] = C->lo[c][k] - ghost;
      C->adims[c][k] = (C->hi[c][k] - C->lo[c][k]) + 2 * ghost;
      C->n_el[c] *= C->adims[c][k];
    }
    if (empty) C->n_el[c] = 0;
    C->has[c] = !empty;
    C->spatial[c] = (double *) malloc((size_t) (C->n_el[c] + 1) * sizeof(double));
    C->value[c] = (double *) malloc((size_t) (C->n_el[c] + 1) * sizeof(double));
    /* the field at every array element: owned and ghost; NaN where no sample exists */
    for (int64_t e = 0; e < C->n_el[c]; ++e)
    {
      int64_t i[3] = {C->alo[c][0] + e % C->adims[c][0], C->alo[c][1] + (e / C->adims[c][0]) % C->adims[c][1],
                      C->alo[c][2] + e / (C->adims[c][0] * C->adims[c][1])};
      double x[3];
      bool real = true;
      for (int k = 0; k < 3; ++k)
      {
        const bool faces = mac && c == k;
        const int64_t dim = faces ? N + 1 : N;
        if (i[k] < 0 || i[k] >= dim) real = false;
        x[k] = (faces ? 0.0 : 0.5 * h) + (double) i[k] * h;
      }
      double u[3];
      lv_spatial(x, u);
      C->spatial[c][e] = real ? u[c] : NAN;
    }
    if (!empty && (mac || c == 0))
    {
      MFTL_BlockDesc bd;
      memset(&bd, 0, sizeof bd);
      bd.location = C->loc[c];
      for (int k = 0; k < 3; ++k)
      {
        bd.owned_lo[k] = C->lo[c][k];
        bd.owned_hi[k] = C->hi[c][k];
        bd.array_lo[k] = C->alo[c][k];
        bd.array_dims[k] = C->adims[c][k];
        bd.ghost_layers[k] = ghost;
      }
      MFTLSolverGridAddBlock(s->grid, &bd);
    }
    C->ptr[c] = C->value[c];
    memset(&s->field[c], 0, sizeof s->field[c]);
    s->field[c].location = C->loc[c];
    s->field[c].data = C->has[c] ? &C->ptr[c] : MFTL_NULL;
    s->field[c].stride = 1;
    s->field[c].halo_valid = ghost > 0;
  }
}

static void cart_free(lv_solver *s)
{
  cart *C = (cart *) s->impl;
  for (int c = 0; c < 3; ++c)
  {
    free(C->spatial[c]);
    free(C->value[c]);
  }
  free(C);
  MFTLSolverGridDestroy(&s->grid);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  const int rank = lv_rank(), np = lv_nproc();

  int Ns[16] = {16, 32, 64}, nN = 3, refine = 3, max_steps = -1;
  double T = 3.0, dt = 0.02;
  bool mac = true, aligned = false, do_a = true, do_b = true, verify = false;
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-n") && i + 1 < argc)
    {
      nN = 0;
      for (char *tok = strtok(argv[++i], ","); tok && nN < 16; tok = strtok(MFTL_NULL, ",")) Ns[nN++] = atoi(tok);
    }
    else if (!strcmp(argv[i], "-refine") && i + 1 < argc)
      refine = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-T") && i + 1 < argc)
      T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-dt") && i + 1 < argc)
      dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "-steps") && i + 1 < argc)
      max_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-layout") && i + 1 < argc)
      mac = strcmp(argv[++i], "collocated") != 0;
    else if (!strcmp(argv[i], "-solver") && i + 1 < argc)
      aligned = !strcmp(argv[++i], "aligned");
    else if (!strcmp(argv[i], "-kernel") && i + 1 < argc)
    {
      const char *k = argv[++i];
      P.kernel = !strcmp(k, "bspline3") ? MFTL_KERNEL_BSPLINE3 : !strcmp(k, "cosine4") ? MFTL_KERNEL_COSINE4 : !strcmp(k, "roma3") ? MFTL_KERNEL_ROMA3
                                                                                                                                       : MFTL_KERNEL_PESKIN4;
    }
    else if (!strcmp(argv[i], "-pass") && i + 1 < argc)
    {
      const char *p = argv[++i];
      do_a = strchr(p, 'a') != MFTL_NULL;
      do_b = strchr(p, 'b') != MFTL_NULL;
    }
    else if (!strcmp(argv[i], "-verify"))
      verify = true;
    else
    {
      if (rank == 0) fprintf(stderr, "unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MFTL_KernelInfo ki;
  MFTLCouplingKernelInfo(P.kernel, &ki);
  if (rank == 0)
  {
    printf("# LeVeque deformation, velocity on a %s Cartesian grid, through the coupler\n", mac ? "staggered (MAC)" : "collocated");
    printf("#   %d ranks; solver decomposition %s (%s); kernel %s with linear moment correction\n", np, aligned ? "aligned with MFTL's" : "z-slabs",
           aligned ? "2 valid ghost layers" : "no ghosts", ki.name);
    printf("#   sphere r = 0.15 at (0.35,0.35,0.35), refine %d; T = %g, dt = %g; RK4 with stage binds, 3 fetches per step\n", refine, T, dt);
    fflush(stdout);
  }

  int failed = 0;
  lv_pass_a prev;
  double hprev = 0.0;
  lv_pass_a *res = (lv_pass_a *) calloc((size_t) nN, sizeof(lv_pass_a));
  if (do_a && rank == 0) lv_print_pass_a_header();
  for (int g = 0; g < nN && do_a; ++g)
  {
    lv_solver s;
    memset(&s, 0, sizeof s);
    cart_build(&s, Ns[g], mac, aligned, aligned ? 2 : 0, T);
    const int st = lv_run_pass_a(&s, &P, refine, T, dt, max_steps, &res[g]);
    if (st != MFTL_COUPLING_OK)
    {
      if (rank == 0) printf("  N = %d: coupling failed: %s\n", Ns[g], MFTLCouplingStatusName(st));
      failed = 1;
    }
    else if (rank == 0)
    {
      char label[32];
      snprintf(label, sizeof label, "N=%d", Ns[g]);
      lv_print_pass_a_row(label, 1.0 / Ns[g], &res[g], g > 0 ? &prev : MFTL_NULL, hprev);
      fflush(stdout);
    }
    prev = res[g];
    hprev = 1.0 / Ns[g];
    cart_free(&s);
  }
  if (do_a && rank == 0)
  {
    const lv_pass_a *a = &res[nN - 1];
    printf("#   direct run alone (exact field, same RK4): V/V0-1 at T = %+.2e, max | |x-c|-R | at T = %.3e\n", a->vol_direct_err, a->sphere_direct);
    printf("#   coupled, finest grid:                     V/V0-1 at T = %+.2e, max | |x-c|-R | at T = %.3e\n", a->vol_coupled_err, a->sphere_coupled);
    printf("#   coupling on the finest grid: %ld windows for %ld steps (%ld restarts), %ld fetches, %.3f s (window %.3f, bind %.3f, fetch %.3f);\n",
           a->windows, a->steps, a->restarts, a->fetches, a->t_coupling, a->t_window, a->t_bind, a->t_fetch);
    printf("#   band max per rank: self %ld, halo %ld, remote %ld; %ld fetch peers; %.1f MB fetched in total\n", a->stats.band_self, a->stats.band_halo,
           a->stats.band_remote, a->stats.fetch_peers, a->stats.bytes_fetch / 2.0 / 1e6);
  }

  if (do_b)
  {
    lv_solver s;
    memset(&s, 0, sizeof s);
    cart_build(&s, Ns[nN - 1], mac, aligned, aligned ? 2 : 0, T);
    lv_pass_b b;
    const int st = lv_run_pass_b(&s, &P, refine, T, dt, max_steps, &b);
    if (st != MFTL_COUPLING_OK)
    {
      if (rank == 0) printf("  pass B: coupling failed: %s\n", MFTLCouplingStatusName(st));
      failed = 1;
    }
    else if (rank == 0)
    {
      char label[64];
      snprintf(label, sizeof label, "%s grid N=%d", mac ? "MAC" : "collocated", Ns[nN - 1]);
      lv_print_pass_b(label, &b);
    }
    if (verify && st == MFTL_COUPLING_OK)
    {
      /* the coupled front tracker must stay a closed sphere and track the direct volume */
      if ((long) (b.coupled.n_vert - b.coupled.n_edge + b.coupled.n_tri) != 2) failed = 1;
      /* A robustness guard, not the accuracy test (pass A is): two remeshed
       * runs of an under-resolved filament drift apart by several percent at
       * maximum deformation for any small velocity difference -- 5.8 % at
       * N = 32, t = T/2, where each run's own remeshing loss is about 20 %. A
       * broken coupler is off by far more, or fails. */
      if (fabs(b.coupled.volume - b.direct.volume) > 0.10 * b.direct0.volume) failed = 1;
      if (b.failed_remesh) failed = 1;
    }
    cart_free(&s);
  }

  if (verify && do_a && !failed)
  {
    /* second order in h, and errors that are small on the finest grid */
    const lv_pass_a *a = &res[nN - 1];
    if (nN >= 2)
    {
      const double order = log(res[nN - 2].pos_rms[1] / a->pos_rms[1]) / log((double) Ns[nN - 1] / Ns[nN - 2]);
      if (!(order > 1.5)) failed = 1;
      if (rank == 0) printf("VERIFY: observed order %.2f (need > 1.5)\n", order);
    }
    if (!(a->interp_max < 0.5 / (Ns[nN - 1] / 16.0) / (Ns[nN - 1] / 16.0))) failed = 1;
  }
  if (verify && rank == 0) printf("VERIFY: %s\n", failed ? "FAILED" : "ok");
  free(res);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
