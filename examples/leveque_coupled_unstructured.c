/*
 * @Author: Christian Gorges
 * @Date: 2026-10-06
 *
 * @brief LeVeque's deformation test with the velocity known ONLY at the cells
 *        of a fully unstructured tetrahedral mesh, reaching the interface
 *        through the coupler.
 *
 * THE MESH. The vertices of an n^3 lattice of cubes are moved: optionally
 * graded (`-grade G`: x -> x + G sin(2 pi x) / (2 pi), finer at the walls and
 * coarser inside), then jittered at random in every direction the boundary
 * allows (`-jitter J` of the local spacing). Each cube is split into six
 * tetrahedra along its main diagonal (a conforming Kuhn split), every cell's
 * volume is checked to be positive, and the result is a 6 n^3-cell mesh whose
 * cells differ in shape, size and orientation everywhere. The solver is a
 * cell-centred finite-volume code: one velocity sample per cell, at the
 * centroid, with the cell volume as its measure and the cube root of the
 * volume as its resolution.
 *
 * THE PARTITION is the solver's: cells in Morton order of their centroids,
 * cut into P contiguous chunks -- irregular shapes that have nothing to do with
 * MFTL's process grid, so the coupler finds every owner through its directory.
 *
 * THE ROWS are Peskin's kernel, moment-corrected to reproduce linear fields
 * exactly on the irregular cloud, at a metric that follows the local cell size
 * (a Shepard average of the samples' resolutions).
 *
 * The passes are those of leveque_coupled_cartesian.c (see
 * leveque_coupling_common.h): A, without remeshing, against per-vertex exact
 * trajectories for every mesh of the sweep; B, remeshed every step on the
 * finest mesh, against the direct run.
 *
 * USAGE
 *
 *   mpirun -n P leveque_coupled_unstructured [options]
 *
 *     -n LIST        cubes per axis of the sweep, comma-separated (default 8,16,32)
 *     -refine R      sphere subdivision level                     (default 3)
 *     -T, -dt, -steps, -pass, -verify   as in leveque_coupled_cartesian
 *     -jitter J      vertex jitter as a fraction of the spacing   (default 0.2)
 *     -grade G       grading amplitude, 0 <= G < 1                (default 0.3)
 *     -scale S       kernel metric over the cell resolution       (default 1.3)
 */

#include "leveque_coupling_common.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The mesh, generated identically on every rank
 * ═══════════════════════════════════════════════════════════════════════════ */

static double hash_unit(uint64_t k)
{
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdull;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ull;
  k ^= k >> 33;
  return (double) (k >> 11) / (double) (1ull << 52) - 1.0;
}

static double grade(double s, double G) { return s + G * sin(2.0 * MFTL_PI * s) / (2.0 * MFTL_PI); }

typedef struct
{
  int64_t nt;
  double *centroid;  ///< [3 nt]
  double *volume;  ///< [nt]
  double *extent;  ///< [3 nt]
  uint64_t *morton;
} tetmesh;

/** @brief The six tetrahedra of a cube along its main diagonal, as paths through the corners. */
static const int g_kuhn[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};

static int tetmesh_build(int n, double jitter, double G, tetmesh *m)
{
  const int64_t nv1 = n + 1;
  double *vx = (double *) malloc(3 * (size_t) (nv1 * nv1 * nv1) * sizeof(double));
  for (int64_t k = 0; k <= n; ++k)
    for (int64_t j = 0; j <= n; ++j)
      for (int64_t i = 0; i <= n; ++i)
      {
        const int64_t id = i + nv1 * (j + nv1 * k);
        const int64_t idx[3] = {i, j, k};
        for (int a = 0; a < 3; ++a)
        {
          const double s = (double) idx[a] / n;
          double x = grade(s, G);
          if (idx[a] > 0 && idx[a] < n)
          {
            /* jitter by a fraction of the local spacing, never past a neighbour */
            const double hl = fmin(grade(s, G) - grade(s - 1.0 / n, G), grade(s + 1.0 / n, G) - grade(s, G));
            x += jitter * hl * hash_unit((uint64_t) id * 3u + (uint64_t) a);
          }
          vx[3 * id + a] = x;
        }
      }
  m->nt = 6 * (int64_t) n * n * n;
  m->centroid = (double *) malloc(3 * (size_t) m->nt * sizeof(double));
  m->volume = (double *) malloc((size_t) m->nt * sizeof(double));
  m->extent = (double *) malloc(3 * (size_t) m->nt * sizeof(double));
  m->morton = (uint64_t *) malloc((size_t) m->nt * sizeof(uint64_t));
  int64_t t = 0, inverted = 0;
  for (int64_t k = 0; k < n; ++k)
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < n; ++i)
        for (int p = 0; p < 6; ++p, ++t)
        {
          int64_t c[3] = {i, j, k};
          double v[4][3];
          for (int q = 0; q < 4; ++q)
          {
            if (q > 0) c[g_kuhn[p][q - 1]] += 1;
            memcpy(v[q], &vx[3 * (c[0] + nv1 * (c[1] + nv1 * c[2]))], 3 * sizeof(double));
          }
          double e1[3], e2[3], e3[3];
          for (int a = 0; a < 3; ++a)
          {
            e1[a] = v[1][a] - v[0][a];
            e2[a] = v[2][a] - v[0][a];
            e3[a] = v[3][a] - v[0][a];
          }
          double det = e1[0] * (e2[1] * e3[2] - e2[2] * e3[1]) - e1[1] * (e2[0] * e3[2] - e2[2] * e3[0]) + e1[2] * (e2[0] * e3[1] - e2[1] * e3[0]);
          /* the undistorted orientation of path p: even permutations positive */
          const int sign = (p == 0 || p == 3 || p == 4) ? 1 : -1;
          if (det * sign <= 0.0) ++inverted;
          m->volume[t] = fabs(det) / 6.0;
          for (int a = 0; a < 3; ++a)
          {
            m->centroid[3 * t + a] = 0.25 * (v[0][a] + v[1][a] + v[2][a] + v[3][a]);
            double ext = 0.0;
            for (int q = 0; q < 4; ++q) ext = fmax(ext, fabs(v[q][a] - m->centroid[3 * t + a]));
            m->extent[3 * t + a] = ext;
          }
          uint64_t mc[3];
          for (int a = 0; a < 3; ++a) mc[a] = (uint64_t) (m->centroid[3 * t + a] * 2097151.0);
          m->morton[t] = MFTLMorton3D(mc[0], mc[1], mc[2]);
        }
  free(vx);
  return (int) inverted;
}

static void tetmesh_free(tetmesh *m)
{
  free(m->centroid);
  free(m->volume);
  free(m->extent);
  free(m->morton);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The unstructured solver
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int64_t n;
  double *spatial[3], *value[3], *ptr[3];
  double h_eff;
} fv;

static void fv_set_time(lv_solver *s, double t)
{
  fv *F = (fv *) s->impl;
  const double f = lv_time_factor(t, s->T);
  for (int c = 0; c < 3; ++c)
  {
    for (int64_t i = 0; i < F->n; ++i) F->value[c][i] = F->spatial[c][i] * f;
    s->field[c].version = ++s->version;
  }
}

static int fv_build(lv_solver *s, int n, double jitter, double G, double T)
{
  tetmesh m;
  const int inverted = tetmesh_build(n, jitter, G, &m);
  if (inverted) return inverted;
  const int np = lv_nproc(), rank = lv_rank();
  /* Morton order, cut in P contiguous chunks */
  int *perm = (int *) malloc((size_t) m.nt * sizeof(int));
  uint64_t *keys = (uint64_t *) malloc((size_t) m.nt * sizeof(uint64_t));
  for (int64_t t = 0; t < m.nt; ++t)
  {
    perm[t] = (int) t;
    keys[t] = m.morton[t];
  }
  MFTLSortPairs64(keys, perm, (int) m.nt);
  const int64_t a = m.nt * rank / np, b = m.nt * (rank + 1) / np;
  fv *F = (fv *) calloc(1, sizeof(fv));
  F->n = b - a;
  F->h_eff = cbrt(1.0 / (double) m.nt);
  uint64_t *k = (uint64_t *) malloc((size_t) (F->n + 1) * sizeof(uint64_t));
  double *pos = (double *) malloc(3 * (size_t) (F->n + 1) * sizeof(double)), *vol = (double *) malloc((size_t) (F->n + 1) * sizeof(double));
  double *h = (double *) malloc(3 * (size_t) (F->n + 1) * sizeof(double)), *ext = (double *) malloc(3 * (size_t) (F->n + 1) * sizeof(double));
  for (int c = 0; c < 3; ++c)
  {
    F->spatial[c] = (double *) malloc((size_t) (F->n + 1) * sizeof(double));
    F->value[c] = (double *) malloc((size_t) (F->n + 1) * sizeof(double));
    F->ptr[c] = F->value[c];
  }
  for (int64_t i = 0; i < F->n; ++i)
  {
    const int64_t t = perm[a + i];
    k[i] = (uint64_t) t;
    memcpy(&pos[3 * i], &m.centroid[3 * t], 3 * sizeof(double));
    memcpy(&ext[3 * i], &m.extent[3 * t], 3 * sizeof(double));
    vol[i] = m.volume[t];
    h[3 * i] = h[3 * i + 1] = h[3 * i + 2] = cbrt(m.volume[t]);
    double u[3];
    lv_spatial(&pos[3 * i], u);
    for (int c = 0; c < 3; ++c) F->spatial[c][i] = u[c];
  }
  s->grid = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "cells";
  ld.lattice.kind = MFTL_LATTICE_NONE;
  const int loc = MFTLSolverGridAddLocation(s->grid, &ld);
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.n = F->n;
  pd.keys = k;
  pd.pos = pos;
  pd.measure = vol;
  pd.h = h;
  pd.extent = ext;
  MFTLSolverGridAddPoints(s->grid, &pd);
  s->impl = F;
  s->T = T;
  s->set_time = fv_set_time;
  for (int c = 0; c < 3; ++c)
  {
    memset(&s->field[c], 0, sizeof s->field[c]);
    s->field[c].location = loc;
    s->field[c].data = &F->ptr[c];
    s->field[c].stride = 1;
  }
  free(perm);
  free(keys);
  free(k);
  free(pos);
  free(vol);
  free(h);
  free(ext);
  tetmesh_free(&m);
  return 0;
}

static void fv_free(lv_solver *s)
{
  fv *F = (fv *) s->impl;
  for (int c = 0; c < 3; ++c)
  {
    free(F->spatial[c]);
    free(F->value[c]);
  }
  free(F);
  MFTLSolverGridDestroy(&s->grid);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  const int rank = lv_rank(), np = lv_nproc();

  int Ns[16] = {8, 16, 32}, nN = 3, refine = 3, max_steps = -1;
  double T = 3.0, dt = 0.02, jitter = 0.2, G = 0.3;
  bool do_a = true, do_b = true, verify = false;
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.support_scale = 1.3;
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
    else if (!strcmp(argv[i], "-jitter") && i + 1 < argc)
      jitter = atof(argv[++i]);
    else if (!strcmp(argv[i], "-grade") && i + 1 < argc)
      G = atof(argv[++i]);
    else if (!strcmp(argv[i], "-scale") && i + 1 < argc)
      P.support_scale = atof(argv[++i]);
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
  if (rank == 0)
  {
    printf("# LeVeque deformation, velocity at the cells of an unstructured tetrahedral mesh, through the coupler\n");
    printf("#   %d ranks; cells partitioned in Morton-ordered chunks; jitter %.2f, grading %.2f\n", np, jitter, G);
    printf("#   Peskin kernel at %.2f x the Shepard-averaged cell resolution, moment-corrected to linear exactness\n", P.support_scale);
    printf("#   sphere r = 0.15 at (0.35,0.35,0.35), refine %d; T = %g, dt = %g; RK4 with stage binds, 3 fetches per step\n", refine, T, dt);
    fflush(stdout);
  }

  int failed = 0;
  lv_pass_a *res = (lv_pass_a *) calloc((size_t) nN, sizeof(lv_pass_a));
  double *hs = (double *) calloc((size_t) nN, sizeof(double));
  if (do_a && rank == 0) lv_print_pass_a_header();
  for (int g = 0; g < nN && do_a; ++g)
  {
    lv_solver s;
    memset(&s, 0, sizeof s);
    if (fv_build(&s, Ns[g], jitter, G, T) != 0)
    {
      if (rank == 0) fprintf(stderr, "n = %d: inverted cells; lower -jitter or -grade\n", Ns[g]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    hs[g] = ((fv *) s.impl)->h_eff;
    const int st = lv_run_pass_a(&s, &P, refine, T, dt, max_steps, &res[g]);
    if (st != MFTL_COUPLING_OK)
    {
      if (rank == 0) printf("  n = %d: coupling failed: %s\n", Ns[g], MFTLCouplingStatusName(st));
      failed = 1;
    }
    else if (rank == 0)
    {
      char label[32];
      snprintf(label, sizeof label, "n=%d", Ns[g]);
      lv_print_pass_a_row(label, hs[g], &res[g], g > 0 ? &res[g - 1] : MFTL_NULL, g > 0 ? hs[g - 1] : 0.0);
      fflush(stdout);
    }
    fv_free(&s);
  }
  if (do_a && rank == 0)
  {
    const lv_pass_a *a = &res[nN - 1];
    printf("#   (h is the mean cell spacing, the cube root of the volume per cell; %d cells on the finest mesh)\n", 6 * Ns[nN - 1] * Ns[nN - 1] * Ns[nN - 1]);
    printf("#   direct run alone (exact field, same RK4): V/V0-1 at T = %+.2e, max | |x-c|-R | at T = %.3e\n", a->vol_direct_err, a->sphere_direct);
    printf("#   coupled, finest mesh:                     V/V0-1 at T = %+.2e, max | |x-c|-R | at T = %.3e\n", a->vol_coupled_err, a->sphere_coupled);
    printf("#   coupling on the finest mesh: %ld windows for %ld steps (%ld restarts), %ld fetches, %.3f s (window %.3f, bind %.3f, fetch %.3f);\n",
           a->windows, a->steps, a->restarts, a->fetches, a->t_coupling, a->t_window, a->t_bind, a->t_fetch);
    printf("#   band max per rank: self %ld, remote %ld; %ld fetch peers; %.1f MB fetched in total\n", a->stats.band_self, a->stats.band_remote,
           a->stats.fetch_peers, a->stats.bytes_fetch / 2.0 / 1e6);
  }

  if (do_b)
  {
    lv_solver s;
    memset(&s, 0, sizeof s);
    fv_build(&s, Ns[nN - 1], jitter, G, T);
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
      snprintf(label, sizeof label, "tetrahedral mesh n=%d (%d cells)", Ns[nN - 1], 6 * Ns[nN - 1] * Ns[nN - 1] * Ns[nN - 1]);
      lv_print_pass_b(label, &b);
    }
    if (verify && st == MFTL_COUPLING_OK)
    {
      if ((long) (b.coupled.n_vert - b.coupled.n_edge + b.coupled.n_tri) != 2) failed = 1;
      /* A robustness guard, not the accuracy test (pass A is): two remeshed
       * runs of an under-resolved filament drift apart by several percent at
       * maximum deformation for any small velocity difference -- 5.8 % at
       * N = 32, t = T/2, where each run's own remeshing loss is about 20 %. A
       * broken coupler is off by far more, or fails. */
      if (fabs(b.coupled.volume - b.direct.volume) > 0.10 * b.direct0.volume) failed = 1;
      if (b.failed_remesh) failed = 1;
    }
    fv_free(&s);
  }

  if (verify && do_a && !failed && nN >= 2)
  {
    const double order = log(res[nN - 2].pos_rms[1] / res[nN - 1].pos_rms[1]) / log(hs[nN - 2] / hs[nN - 1]);
    if (!(order > 1.5)) failed = 1;
    if (rank == 0) printf("VERIFY: observed order %.2f (need > 1.5)\n", order);
  }
  if (verify && rank == 0) printf("VERIFY: %s\n", failed ? "FAILED" : "ok");
  free(res);
  free(hs);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
