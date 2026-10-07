/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Who completes what, and when: the solver-completed transpose of the
 *        ghost route, and stage bindings whose status the caller agrees.
 *
 * Both features move part of the coupler's correctness argument to the caller,
 * so both tests act out the caller's half themselves -- a reverse halo
 * accumulation written here with MPI_Alltoallv and the global partition, a
 * status agreement made by the test -- and compare the result with the path
 * where the coupler does everything. A test that only checked "the coupler
 * returned OK" would pass with the transpose done twice or not at all.
 */

#include "test_coupling_irregular.h"

/* ═════════════════════════════════════════════════════════════════════════════
 * The solver completes the ghost route's transpose
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief The global box index owning wrapped lattice index @p j, or -1. */
static int owner_box(const adv_blocks *B, const int64_t j[3])
{
  for (int b = 0; b < B->n_all; ++b)
  {
    const int64_t *x = B->all_box[b];
    if (j[0] >= x[0] && j[0] < x[3] && j[1] >= x[1] && j[1] < x[4] && j[2] >= x[2] && j[2] < x[5]) return b;
  }
  return -1;
}

/**
 * @brief What a solver's reverse halo accumulation does: every element of
 *        every block's declared ghost layers is added into the owner of its
 *        wrapped index (wherever that owner is) and the ghost element is
 *        consumed. Component @p comp of the interleaved arrays.
 */
static void solver_reverse_halo(adv_blocks *B, const adv_lat *A, int comp)
{
  /* (wrapped key, value) records per destination rank */
  int *cnt = (int *) calloc((size_t) g_nproc, sizeof(int));
  for (int pass = 0; pass < 2; ++pass)
  {
    double *buf = MFTL_NULL;
    int *dsp = MFTL_NULL, *fill = MFTL_NULL;
    if (pass == 1)
    {
      dsp = (int *) calloc((size_t) g_nproc + 1, sizeof(int));
      for (int r = 0; r < g_nproc; ++r) dsp[r + 1] = dsp[r] + 2 * cnt[r];
      buf = (double *) malloc((size_t) (dsp[g_nproc] > 0 ? dsp[g_nproc] : 1) * sizeof(double));
      fill = (int *) calloc((size_t) g_nproc, sizeof(int));
    }
    for (int i = 0; i < B->n; ++i)
      for (int64_t e = 0; e < B->n_elem[i]; ++e)
      {
        int64_t idx[3], j[3];
        adv_elem_index(B, i, e, idx);
        bool owned = true, ghost = true, real = true;
        for (int k = 0; k < 3; ++k)
        {
          if (idx[k] < B->lo[i][k] || idx[k] >= B->hi[i][k]) owned = false;
          if (idx[k] < B->lo[i][k] - B->ghost[i][k] || idx[k] >= B->hi[i][k] + B->ghost[i][k]) ghost = false;
          j[k] = A->per[k] ? ref_mod(idx[k], A->N[k]) : idx[k];
          if (j[k] < 0 || j[k] >= A->N[k]) real = false;
        }
        if (owned || !ghost || !real) continue;
        const int ob = owner_box(B, j);
        const int dest = B->all_owner[ob];
        if (pass == 0)
        {
          cnt[dest]++;
          continue;
        }
        const double key = (double) (j[0] + A->N[0] * (j[1] + A->N[1] * j[2]));
        buf[dsp[dest] + 2 * fill[dest]] = key;
        buf[dsp[dest] + 2 * fill[dest] + 1] = B->a[i][e * B->ncomp + comp];
        fill[dest]++;
        B->a[i][e * B->ncomp + comp] = 0.0; /* consumed */
      }
    if (pass == 1)
    {
      int *scnt = (int *) malloc((size_t) g_nproc * sizeof(int)), *rcnt = (int *) malloc((size_t) g_nproc * sizeof(int));
      int *rdsp = (int *) calloc((size_t) g_nproc + 1, sizeof(int));
      for (int r = 0; r < g_nproc; ++r) scnt[r] = 2 * cnt[r];
      MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, MPI_COMM_WORLD);
      for (int r = 0; r < g_nproc; ++r) rdsp[r + 1] = rdsp[r] + rcnt[r];
      double *rbuf = (double *) malloc((size_t) (rdsp[g_nproc] > 0 ? rdsp[g_nproc] : 1) * sizeof(double));
      MPI_Alltoallv(buf, scnt, dsp, MPI_DOUBLE, rbuf, rcnt, rdsp, MPI_DOUBLE, MPI_COMM_WORLD);
      for (int q = 0; q < rdsp[g_nproc]; q += 2)
      {
        const int64_t key = (int64_t) rbuf[q];
        const int64_t j[3] = {key % A->N[0], (key / A->N[0]) % A->N[1], key / (A->N[0] * A->N[1])};
        for (int i = 0; i < B->n; ++i)
        {
          if (j[0] < B->lo[i][0] || j[0] >= B->hi[i][0] || j[1] < B->lo[i][1] || j[1] >= B->hi[i][1] || j[2] < B->lo[i][2] || j[2] >= B->hi[i][2]) continue;
          const int64_t el = (j[0] - B->alo[i][0]) + B->adims[i][0] * ((j[1] - B->alo[i][1]) + B->adims[i][1] * (j[2] - B->alo[i][2]));
          B->a[i][el * B->ncomp + comp] += rbuf[q + 1];
        }
      }
      free(rbuf);
      free(scnt);
      free(rcnt);
      free(rdsp);
      free(buf);
      free(dsp);
      free(fill);
    }
  }
  free(cnt);
}

/**
 * @brief halo_transpose: the loads of samples read from ghost layers stay in
 *        those ghost elements, every other ghost element is cleared, and one
 *        reverse halo accumulation by the SOLVER (written here) gives exactly
 *        what the coupler's own reverse route gives -- integrated and density,
 *        side by side in one batch with fields that do not opt in. Ghosts
 *        holding garbage beforehand add nothing; padding beyond them is never
 *        touched; the reverse payload shrinks by the halo's share.
 */
static int test_the_solver_completes_the_halo_transpose(void)
{
  const bool per[3] = {true, false, true};
  adv_lat A = adv_default_lattice(true, per);
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  /* four interleaved components: (transpose, coupler) x (integrated, density) */
  adv_decompose(g, loc, &A, 3 * g_nproc + 1, 515u, 3, 4, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  adv_markers M;
  adv_markers_make(&ens->domain, &A, 2.0 * A.h[0], 60, 33u, 0.5 * A.h[1], &M);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, (int) M.n_all, M.all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplerStats st;
  MFTLCouplerGetStats(c, &st);
  /* non-vacuity: some rank reads from its ghost layers */
  if (g_nproc > 1) TEST_ASSERT(anywhere(st.band_halo > 0, MPI_COMM_WORLD));

  /* targets: owned 0, ghost layers garbage, padding a sentinel */
  for (int i = 0; i < B.n; ++i)
    for (int64_t e = 0; e < B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&B, i, e, idx);
      bool owned = true, ghost = true;
      for (int k = 0; k < 3; ++k)
      {
        if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) owned = false;
        if (idx[k] < B.lo[i][k] - B.ghost[i][k] || idx[k] >= B.hi[i][k] + B.ghost[i][k]) ghost = false;
      }
      for (int cpt = 0; cpt < 4; ++cpt) B.a[i][4 * e + cpt] = owned ? 0.0 : (ghost ? -777.0 : -555.0);
    }
  MFTL_SolverField F[2];
  double *p0[ADV_MAX_BOXES], *p1[ADV_MAX_BOXES];
  for (int i = 0; i < B.n; ++i)
  {
    p0[i] = B.a[i];
    p1[i] = B.a[i] + 1;
  }
  memset(F, 0, sizeof F);
  F[0].location = F[1].location = loc;
  F[0].data = B.n ? p0 : MFTL_NULL;
  F[1].data = B.n ? p1 : MFTL_NULL;
  F[0].stride = F[1].stride = 4;
  F[0].halo_transpose = true;
  double *L = (double *) malloc((size_t) (n > 0 ? 2 * n : 1) * sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < n; ++i) L[2 * i] = L[2 * i + 1] = hash01(ref[i].key + 1000u);
  MFTLCouplerResetStats(c);
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  long bytes_mixed = 0;
  MFTLCouplerGetStats(c, &st);
  bytes_mixed = st.bytes_spread;
  /* the density pair, in a second batch (components 2 and 3) */
  double *p2[ADV_MAX_BOXES], *p3[ADV_MAX_BOXES];
  for (int i = 0; i < B.n; ++i)
  {
    p2[i] = B.a[i] + 2;
    p3[i] = B.a[i] + 3;
  }
  MFTL_SolverField D[2] = {F[0], F[1]};
  D[0].data = B.n ? p2 : MFTL_NULL;
  D[1].data = B.n ? p3 : MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, D, L, MFTL_SPREAD_DENSITY) == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* no garbage survives in a transposed field's ghost layers; the others' ghosts and all padding are untouched */
  int ghosts_ok = 1;
  for (int i = 0; i < B.n; ++i)
    for (int64_t e = 0; e < B.n_elem[i]; ++e)
    {
      int64_t idx[3];
      adv_elem_index(&B, i, e, idx);
      bool owned = true, ghost = true;
      for (int k = 0; k < 3; ++k)
      {
        if (idx[k] < B.lo[i][k] || idx[k] >= B.hi[i][k]) owned = false;
        if (idx[k] < B.lo[i][k] - B.ghost[i][k] || idx[k] >= B.hi[i][k] + B.ghost[i][k]) ghost = false;
      }
      if (owned) continue;
      for (int cpt = 0; cpt < 4; ++cpt)
      {
        const double v = B.a[i][4 * e + cpt];
        if (!ghost && v != -555.0) ghosts_ok = 0;
        if (ghost && (cpt == 1 || cpt == 3) && v != -777.0) ghosts_ok = 0;
        if (ghost && (cpt == 0 || cpt == 2) && v == -777.0) ghosts_ok = 0;
      }
    }
  TEST_ASSERT(agreed(ghosts_ok, MPI_COMM_WORLD));

  /* the solver's half: one reverse halo accumulation of the transposed components */
  solver_reverse_halo(&B, &A, 0);
  solver_reverse_halo(&B, &A, 2);
  double worst = 0.0, scale = 0.0;
  for (int i = 0; i < B.n; ++i)
    for (int64_t k3 = B.lo[i][2]; k3 < B.hi[i][2]; ++k3)
      for (int64_t j3 = B.lo[i][1]; j3 < B.hi[i][1]; ++j3)
        for (int64_t i3 = B.lo[i][0]; i3 < B.hi[i][0]; ++i3)
        {
          const int64_t el = (i3 - B.alo[i][0]) + B.adims[i][0] * ((j3 - B.alo[i][1]) + B.adims[i][1] * (k3 - B.alo[i][2]));
          worst = fmax(worst, fabs(B.a[i][4 * el + 0] - B.a[i][4 * el + 1]));
          worst = fmax(worst, fabs(B.a[i][4 * el + 2] - B.a[i][4 * el + 3]) * A.h[0] * A.h[1] * A.h[2]);
          scale = fmax(scale, fabs(B.a[i][4 * el + 1]));
        }
  MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &scale, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  if (g_rank == 0 && !(worst <= 1e-14 * fmax(scale, 1.0))) printf("\n  transposed vs routed: %.3e (scale %.3e)", worst, scale);
  TEST_ASSERT(agreed(worst <= 1e-14 * fmax(scale, 1.0) && scale > 0.0, MPI_COMM_WORLD));

  /* the reverse payload: without any transposed field it is larger, when there is a halo */
  MFTLCouplerResetStats(c);
  F[0].halo_transpose = false;
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerGetStats(c, &st);
  TEST_ASSERT(agreed(st.bytes_spread >= bytes_mixed, MPI_COMM_WORLD));
  if (g_nproc > 1) TEST_ASSERT(anywhere(st.bytes_spread > bytes_mixed, MPI_COMM_WORLD));

  /* the flag is part of the signature: ranks that disagree are refused, nothing written */
  F[0].halo_transpose = (g_rank == 0);
  const double keep = (B.n > 0 && B.n_elem[0] > 0) ? B.a[0][0] : 0.0;
  if (g_nproc > 1)
  {
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 2, F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(B.n == 0 || B.n_elem[0] == 0 || B.a[0][0] == keep, MPI_COMM_WORLD));
  }
  free(L);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  adv_markers_free(&M);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief Two blocks registered over ONE array, so the ghost layer of the first
 *        is, in memory, the owned samples of the second. Clearing ghost layers
 *        for a transposed spread must not clear them: the result equals the
 *        coupler-routed spread.
 */
static int test_ghost_layers_aliasing_owned_samples_survive(void)
{
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 12;
    A.h[k] = 1.0 / 12.0;
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = true;
    A.gmin[k] = 0.0;
  }
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  /* this rank's z-slab, split at its middle into two blocks over one array with two ghost layers on every side */
  const int64_t z0 = A.N[2] * g_rank / g_nproc, z1 = A.N[2] * (g_rank + 1) / g_nproc, zm = (z0 + z1) / 2;
  const int G = 2;
  const int64_t ad[3] = {A.N[0] + 2 * G, A.N[1] + 2 * G, (z1 - z0) + 2 * G};
  const int64_t nel = ad[0] * ad[1] * ad[2];
  double *arr[2] = {(double *) malloc((size_t) (nel > 0 ? nel : 1) * sizeof(double)), (double *) malloc((size_t) (nel > 0 ? nel : 1) * sizeof(double))};
  int npieces = 0;
  const int64_t zs[3] = {z0, zm, z1};
  for (int h = 0; h < 2; ++h)
  {
    if (zs[h + 1] <= zs[h]) continue;
    MFTL_BlockDesc bd;
    memset(&bd, 0, sizeof bd);
    bd.location = loc;
    for (int k = 0; k < 3; ++k)
    {
      bd.owned_lo[k] = (k == 2) ? zs[h] : 0;
      bd.owned_hi[k] = (k == 2) ? zs[h + 1] : A.N[k];
      bd.array_lo[k] = (k == 2) ? z0 - G : -G;
      bd.array_dims[k] = ad[k];
      bd.ghost_layers[k] = G;
    }
    MFTLSolverGridAddBlock(g, &bd);
    ++npieces;
  }
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  adv_markers M;
  adv_markers_make(&ens->domain, &A, 2.0 * A.h[0], 40, 404u, 0.0, &M);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, (int) M.n_all, M.all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double *L = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < n; ++i) L[i] = 1.0 + hash01(ref[i].key);
  double *ptr[2][2] = {{arr[0], arr[0]}, {arr[1], arr[1]}};
  for (int v = 0; v < 2; ++v)
  {
    /* owned samples hold 5 (spreading ADDS: a cleared alias loses it), ghosts 0 */
    for (int64_t e = 0; e < nel; ++e)
    {
      const int64_t k3 = z0 - G + e / (ad[0] * ad[1]), j3 = -G + (e / ad[0]) % ad[1], i3 = -G + e % ad[0];
      arr[v][e] = (k3 >= z0 && k3 < z1 && j3 >= 0 && j3 < A.N[1] && i3 >= 0 && i3 < A.N[0]) ? 5.0 : 0.0;
    }
    MFTL_SolverField F;
    memset(&F, 0, sizeof F);
    F.location = loc;
    F.data = npieces ? ptr[v] : MFTL_NULL;
    F.halo_transpose = (v == 0);
    TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &F, L, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  }
  /* the slab's owned samples: identical to the routed spread up to the halo the solver has not yet accumulated --
   * which on this decomposition is only the OTHER ranks' samples (z outside [z0, z1)) */
  double worst = 0.0, total[2] = {0.0, 0.0};
  for (int64_t k3 = z0; k3 < z1; ++k3)
    for (int64_t j3 = 0; j3 < A.N[1]; ++j3)
      for (int64_t i3 = 0; i3 < A.N[0]; ++i3)
      {
        const int64_t e = (i3 + G) + ad[0] * ((j3 + G) + ad[1] * (k3 - z0 + G));
        worst = fmax(worst, fabs(arr[0][e] - arr[1][e]));
        total[0] += arr[0][e] - 5.0;
        total[1] += arr[1][e] - 5.0;
      }
  /* a sample owned on this rank is SELF, never in a ghost: no part of its load may have been cleared */
  MPI_Allreduce(MPI_IN_PLACE, &worst, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  /* summed over ranks, the transposed field misses exactly what sits in other ranks' ghosts; add those */
  double ghosts = 0.0;
  for (int64_t e = 0; e < nel; ++e)
  {
    const int64_t k3 = z0 - G + e / (ad[0] * ad[1]), j3 = -G + (e / ad[0]) % ad[1], i3 = -G + e % ad[0];
    const bool owned = (k3 >= z0 && k3 < z1 && j3 >= 0 && j3 < A.N[1] && i3 >= 0 && i3 < A.N[0]);
    if (!owned) ghosts += arr[0][e];
  }
  double s3[3] = {total[0], total[1], ghosts};
  MPI_Allreduce(MPI_IN_PLACE, s3, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  if (g_nproc == 1)
    TEST_ASSERT(agreed(worst <= 1e-14, MPI_COMM_WORLD)); /* one rank: nothing is HALO, all is SELF */
  TEST_ASSERT(agreed(fabs(s3[0] + s3[2] - s3[1]) <= 1e-12 * fabs(s3[1]) && fabs(s3[1]) > 0.0, MPI_COMM_WORLD));
  free(L);
  free(arr[0]);
  free(arr[1]);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  adv_markers_free(&M);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Stage bindings whose status the caller agrees
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief Bitwise equality of the rows of two bindings over one location. */
static int same_rows(const MFTL_CouplingBinding *a, const MFTL_CouplingBinding *b)
{
  const int64_t n = MFTLCouplerBindingMarkerCount(a);
  if (n != MFTLCouplerBindingMarkerCount(b)) return 0;
  for (int64_t i = 0; i < n; ++i)
  {
    int n1, n2;
    const uint64_t *k1, *k2;
    const double *w1, *w2;
    MFTLCouplerBindingRow(a, 0, i, &n1, &k1, &w1);
    uint64_t kk[1024];
    double ww[1024];
    if (n1 > 1024) return 0;
    memcpy(kk, k1, (size_t) n1 * sizeof(uint64_t));
    memcpy(ww, w1, (size_t) n1 * sizeof(double));
    MFTLCouplerBindingRow(b, 0, i, &n2, &k2, &w2);
    if (n1 != n2 || memcmp(kk, k2, (size_t) n1 * sizeof(uint64_t)) != 0 || memcmp(ww, w2, (size_t) n1 * sizeof(double)) != 0) return 0;
  }
  return 1;
}

/**
 * @brief MFTLCouplerBindStageLocal(): the rows of the checked stage bind, to the
 *        bit, with no communication -- it can be called by SOME ranks only
 *        (here the even ones; a collective inside would deadlock the test), and
 *        while a fetch is in flight. A failure on one rank is that rank's alone
 *        until the caller agrees it, and then the ordinary repair (a wider
 *        window) works.
 */
static int test_local_stage_binds_are_local(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  MFTL_Ensemble *ens = make_ens(lo, hi, pp);
  const double cc[3] = {0.5, 0.45, 0.55};
  add_sphere(ens, cc, 0.27, 3);
  scatter(ens);
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 16;
    A.h[k] = 1.0 / 16.0;
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = true;
  }
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc, 1717u, 1, 1, false, &B);
  rnd_ctx rc = {{16, 16, 16}, 3u};
  adv_fill(&B, &A, 0, f_random, &rc, true);
  const double skin = 0.02;
  coupled x;
  TEST_ASSERT(agreed(couple(ens, g, MFTL_NULL, skin, &x) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t nm = MFTLCouplerBindingMarkerCount(x.b);
  const double *p0 = MFTLCouplerBindingPositions(x.b);
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(x.b);
  double *ps = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  for (int64_t i = 0; i < nm; ++i)
    for (int k = 0; k < 3; ++k) ps[3 * i + k] = p0[3 * i + k] + 0.4 * skin * hash01(ref[i].key * 5u + (uint64_t) k);
  MFTL_StageGeometry sg = {ps, MFTL_NULL};

  /* checked and local: the same bits */
  MFTL_CouplingBinding *bc = MFTL_NULL, *bl = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(x.c, x.b, &sg, &bc) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindStageLocal(x.c, x.b, &sg, &bl) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(same_rows(bc, bl), MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bl);

  /* SOME ranks only: no communication inside, or this deadlocks */
  int st_even = MFTL_COUPLING_OK;
  if (g_rank % 2 == 0)
  {
    for (int rep = 0; rep < 1 + g_rank; ++rep)
    {
      st_even = MFTLCouplerBindStageLocal(x.c, x.b, &sg, &bl);
      MFTLCouplerReleaseBinding(&bl);
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  TEST_ASSERT(agreed(st_even == MFTL_COUPLING_OK, MPI_COMM_WORLD));

  /* while a fetch is in flight: rows built during the exchange */
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, true);
  MFTL_TransferSpec ts = {MFTL_TRANSFER_FETCH, x.win, MFTL_NULL, 1, &F, MFTL_SPREAD_INTEGRATED};
  MFTL_TransferReady *ready = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(x.c, &ts, &ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerFetchBegin(x.c, ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int st_flight = MFTLCouplerBindStageLocal(x.c, x.b, &sg, &bl);
  MFTL_CouplingFetch *f = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerFetchEnd(x.c, &f) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(st_flight == MFTL_COUPLING_OK && same_rows(bc, bl), MPI_COMM_WORLD));
  double *U1 = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double)), *U2 = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(bc, f, U1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolateFetched(bl, f, U2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(nm == 0 || memcmp(U1, U2, (size_t) nm * sizeof(double)) == 0, MPI_COMM_WORLD));
  MFTLCouplerReleaseFetch(&f);
  MFTLCouplerReleaseReady(&ready);
  MFTLCouplerReleaseBinding(&bl);

  /* one rank's stage leaves the window: STALE there, OK elsewhere -- until agreed */
  int holder = (nm > 0) ? g_rank : -1;
  MPI_Allreduce(MPI_IN_PLACE, &holder, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  double *far = (double *) malloc(3 * (size_t) (nm > 0 ? nm : 1) * sizeof(double));
  memcpy(far, ps, 3 * (size_t) nm * sizeof(double));
  if (g_rank == holder)
    for (int k = 0; k < 3; ++k) far[k] = ref[0].key % 2 ? 0.02 : 0.98; /* a corner no marker's bin is near */
  MFTL_StageGeometry sf = {far, MFTL_NULL};
  const int mine = MFTLCouplerBindStageLocal(x.c, x.b, &sf, &bl);
  TEST_ASSERT(agreed(g_rank == holder ? (mine == MFTL_COUPLING_STALE && bl == MFTL_NULL) : (mine == MFTL_COUPLING_OK), MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bl);
  TEST_ASSERT(agreed(MFTLCouplerAgreeStatus(x.c, mine) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  /* an out-of-range status is not trusted */
  TEST_ASSERT(agreed(MFTLCouplerAgreeStatus(x.c, 99) == MFTL_COUPLING_INVALID, MPI_COMM_WORLD));

  /* the surface moved: a local stage bind of the old base is STALE, on its own */
  MFTLEnsembleMarkChanged(ens);
  const int moved = MFTLCouplerBindStageLocal(x.c, x.b, &sg, &bl);
  TEST_ASSERT(agreed(moved == MFTL_COUPLING_STALE || (nm == 0 && ens->n_local_objects == 0), MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&bl);

  MFTL_CouplerStats s;
  MFTLCouplerReduceStats(x.c, &s);
  TEST_ASSERT(agreed(s.n_local_stage_binds >= 4 * g_nproc && s.n_stage_binds > s.n_local_stage_binds, MPI_COMM_WORLD));
  free(far);
  free(U1);
  free(U2);
  free(ps);
  MFTLCouplerReleaseBinding(&bc);
  adv_free_blocks(&B);
  teardown(&x, &g, &ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Updates that keep what they can
 * ═════════════════════════════════════════════════════════════════════════════ */

/** @brief This rank's blocks of a lattice exported SELECTED over each block's whole owned box, minus an optional index box. */
static void export_blocks(MFTL_SolverGrid *g, int loc, const adv_blocks *B, const int64_t *minus)
{
  int64_t boxes[6 * (ADV_MAX_BOXES + 8)];
  int nb = 0;
  for (int i = 0; i < B->n; ++i)
  {
    if (B->hi[i][0] <= B->lo[i][0] || B->hi[i][1] <= B->lo[i][1] || B->hi[i][2] <= B->lo[i][2]) continue;
    int64_t rem[6 * 8];
    int nrem = 1;
    for (int k = 0; k < 3; ++k)
    {
      rem[k] = B->lo[i][k];
      rem[3 + k] = B->hi[i][k];
    }
    if (minus)
    {
      /* carve the minus box out along x only: up to two pieces */
      bool hit = true;
      for (int k = 0; k < 3; ++k)
        if (minus[3 + k] <= rem[k] || minus[k] >= rem[3 + k]) hit = false;
      if (hit)
      {
        /* keep everything outside the minus box: six slabs */
        int64_t keep[6 * 8];
        int nk = 0;
        int64_t cur[6];
        memcpy(cur, rem, sizeof cur);
        for (int k = 0; k < 3; ++k)
        {
          if (cur[k] < minus[k])
          {
            memcpy(&keep[6 * nk], cur, sizeof cur);
            keep[6 * nk + 3 + k] = minus[k];
            ++nk;
            cur[k] = minus[k];
          }
          if (cur[3 + k] > minus[3 + k])
          {
            memcpy(&keep[6 * nk], cur, sizeof cur);
            keep[6 * nk + k] = minus[3 + k];
            ++nk;
            cur[3 + k] = minus[3 + k];
          }
        }
        memcpy(rem, keep, 6 * (size_t) nk * sizeof(int64_t));
        nrem = nk;
      }
    }
    memcpy(&boxes[6 * nb], rem, 6 * (size_t) nrem * sizeof(int64_t));
    nb += nrem;
  }
  MFTL_CoverageDesc cv = {loc, MFTL_GRID_SELECTED, nb, boxes, MFTL_NULL};
  MFTLSolverGridSetCoverage(g, &cv);
}

/**
 * @brief A change of EXPORTS alone keeps every window whose served samples
 *        stay exported -- no new request, the same bits -- and stales, on
 *        every rank, a window one of whose served samples left the export;
 *        restoring the export revives it. A change of what is OWNED stales
 *        every window, and re-registering IDENTICAL ownership keeps the
 *        directory all the same.
 */
static int test_a_selection_change_keeps_windows_that_stay_exported(void)
{
  const bool per[3] = {true, false, true};
  adv_lat A = adv_default_lattice(true, per);
  MFTL_Ensemble *ens = adv_ensemble(&A);
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc + 1, 31u, 1, 1, false, &B);
  export_blocks(g, loc, &B, MFTL_NULL);
  rnd_ctx rc = {{A.N[0], A.N[1], A.N[2]}, 12u};
  adv_fill(&B, &A, 0, f_random, &rc, false);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* a few markers in the middle of the box: their band is far from the box corner carved below */
  double all[3 * 8];
  for (int i = 0; i < 8; ++i)
    for (int k = 0; k < 3; ++k) all[3 * i + k] = A.gmin[k] + (0.4 + 0.03 * (double) ((i + k) % 5)) * A.L[k];
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t n = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, 8, all, 0, MFTL_NULL, &w, &b, &n) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_SolverField F = adv_field(&B, loc, 0, 1, false);
  double U0[8], U1[8];
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U0) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTL_CouplerStats s0, s1;
  MFTLCouplerReduceStats(c, &s0);

  /* 1. the export shrinks somewhere the band does not reach: kept -- the
   *    window, its bindings, and a transfer token prepared before the change */
  MFTL_TransferSpec ts = {MFTL_TRANSFER_FETCH, w, MFTL_NULL, 1, &F, MFTL_SPREAD_INTEGRATED};
  MFTL_TransferReady *ready = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerPrepareTransfer(c, &ts, &ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const int64_t corner[6] = {0, 0, 0, 2, 2, 2};
  export_blocks(g, loc, &B, corner);
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  {
    MFTL_CouplingFetch *ff = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerFetchBegin(c, ready) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerFetchEnd(c, &ff) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseFetch(&ff);
    MFTLCouplerReleaseReady(&ready);
  }
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n == 0 || memcmp(U0, U1, (size_t) n * sizeof(double)) == 0, MPI_COMM_WORLD));
  MFTL_CouplingBinding *b2 = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerBindStage(c, b, &(MFTL_StageGeometry){MFTLCouplerBindingPositions(b), MFTL_NULL}, &b2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReleaseBinding(&b2);
  MFTLCouplerReduceStats(c, &s1);
  TEST_ASSERT(s1.n_directory_builds == s0.n_directory_builds && s1.n_directory_kept > s0.n_directory_kept && s1.n_requests == s0.n_requests);

  /* 2. one sample the band SERVES leaves the export, on one rank: stale everywhere */
  {
    /* the sample nearest the first marker, by the reference lattice */
    int64_t j[3];
    for (int k = 0; k < 3; ++k) j[k] = (int64_t) floor((all[k] - A.origin[k]) / A.h[k] + 0.5);
    const int64_t hole[6] = {j[0], j[1], j[2], j[0] + 1, j[1] + 1, j[2] + 1};
    bool mine = false;
    for (int i = 0; i < B.n; ++i)
      if (j[0] >= B.lo[i][0] && j[0] < B.hi[i][0] && j[1] >= B.lo[i][1] && j[1] < B.hi[i][1] && j[2] >= B.lo[i][2] && j[2] < B.hi[i][2]) mine = true;
    export_blocks(g, loc, &B, mine ? hole : corner);
  }
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U1) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerBindStage(c, b, &(MFTL_StageGeometry){MFTLCouplerBindingPositions(b), MFTL_NULL}, &b2) == MFTL_COUPLING_STALE,
                     MPI_COMM_WORLD));

  /* 3. the export restored: the same window serves again, the same bits */
  export_blocks(g, loc, &B, MFTL_NULL);
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U1) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(n == 0 || memcmp(U0, U1, (size_t) n * sizeof(double)) == 0, MPI_COMM_WORLD));

  /* 4. the same ownership re-registered: directory kept, every window stale */
  MFTLCouplerReduceStats(c, &s0);
  MFTLSolverGridReset(g);
  adv_blocks B2;
  adv_decompose(g, loc, &A, 2 * g_nproc + 1, 31u, 1, 1, false, &B2); /* identical partition */
  export_blocks(g, loc, &B2, MFTL_NULL);
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReduceStats(c, &s1);
  TEST_ASSERT(s1.n_directory_builds == s0.n_directory_builds && s1.n_directory_kept > s0.n_directory_kept);
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U1) == MFTL_COUPLING_STALE, MPI_COMM_WORLD));

  /* 5. a different ownership: rebuilt */
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerReduceStats(c, &s0);
  MFTLSolverGridReset(g);
  adv_blocks B3;
  adv_decompose(g, loc, &A, 2 * g_nproc + 3, 32u, 1, 1, false, &B3);
  TEST_ASSERT(agreed(MFTLCouplerUpdateGrid(c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  MFTLCouplerReduceStats(c, &s1);
  TEST_ASSERT(s1.n_directory_builds > s0.n_directory_builds);
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  adv_free_blocks(&B2);
  adv_free_blocks(&B3);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/* ═════════════════════════════════════════════════════════════════════════════
 * Windows that absorb a redistribution
 * ═════════════════════════════════════════════════════════════════════════════ */

/**
 * @brief A marker handed to another rank -- as a redistribution does -- binds
 *        in the window that rank already has when the window was requested
 *        with a migration allowance, and is STALE there without one. Rows are
 *        the same bits as a fresh window's.
 */
static int test_a_migrated_user_marker_binds_in_the_old_window(void)
{
  if (g_nproc < 2) return 0;
  const bool per[3] = {true, true, true};
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 24;
    A.h[k] = 1.0 / 24.0;
    A.origin[k] = 0.5 * A.h[k];
    A.L[k] = 1.0;
    A.per[k] = per[k];
  }
  MFTL_Ensemble *ens = adv_ensemble(&A);
  const MFTL_Domain *d = &ens->domain;
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
  adv_blocks B;
  adv_decompose(g, loc, &A, 2 * g_nproc, 909u, 1, 1, false, &B);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* rank 0's box along x is [0, 1/cuts); the marker sits just inside its upper face, rank 0 owns it;
   * every other rank holds one marker deep inside its own box, far from that face */
  const double skin = 0.03, wx = 1.0 / (double) d->cuts[0];
  const double edge[3] = {wx - 0.01, 0.5 * (1.0 / (double) d->cuts[1]), 0.5 * (1.0 / (double) d->cuts[2])};
  int co[3];
  MPI_Cart_coords(d->comm_cart, g_rank, 3, co);
  double mine[3];
  for (int k = 0; k < 3; ++k) mine[k] = ((double) co[k] + 0.5) / (double) d->cuts[k];
  int left_of_edge = -1; /* the rank owning [wx, 2 wx) at the marker's y, z: the receiver */
  {
    double p[3] = {wx + 0.01, edge[1], edge[2]};
    left_of_edge = MFTLGetOwnerProc(d, p);
  }
  for (int mig = 0; mig < 2; ++mig)
  {
    MFTL_MarkerSpec ms;
    memset(&ms, 0, sizeof ms);
    ms.kind = MFTL_MARKERS_USER;
    double pos[3];
    uint64_t key = (uint64_t) g_rank + 1u;
    if (g_rank == 0)
      memcpy(pos, edge, sizeof pos);
    else
      memcpy(pos, mine, sizeof pos);
    ms.n_user = 1;
    ms.user_pos = pos;
    ms.user_keys = &key;
    MFTL_WindowSpec ws = {{skin, skin, skin}, {mig ? skin : 0.0, mig ? skin : 0.0, mig ? skin : 0.0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    MFTL_CouplingWindow *w = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(c, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    /* the marker moves by half the skin across the face, and now belongs to the receiver */
    double moved[3] = {edge[0] + 0.5 * skin, edge[1], edge[2]};
    double *bpos[2] = {MFTL_NULL, MFTL_NULL};
    uint64_t bkeys[2];
    int nb = 0;
    if (g_rank != 0)
    {
      bpos[nb] = mine;
      bkeys[nb++] = key;
    }
    if (g_rank == left_of_edge)
    {
      bpos[nb] = moved;
      bkeys[nb++] = 1u;
    }
    double flat[6];
    for (int i = 0; i < nb; ++i) memcpy(&flat[3 * i], bpos[i], 3 * sizeof(double));
    ms.n_user = nb;
    ms.user_pos = flat;
    ms.user_keys = bkeys;
    MFTL_CouplingBinding *b = MFTL_NULL;
    const int st = MFTLCouplerBind(c, &ms, w, &b);
    if (mig == 0)
      TEST_ASSERT(agreed(st == MFTL_COUPLING_STALE, MPI_COMM_WORLD));
    else
    {
      TEST_ASSERT(agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      /* the same rows as a fresh request for the markers where they are now */
      MFTL_WindowSpec ws0 = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
      MFTL_SupportRequest *r2 = MFTL_NULL;
      MFTL_CouplingWindow *w2 = MFTL_NULL;
      MFTL_CouplingBinding *b2 = MFTL_NULL;
      TEST_ASSERT(agreed(MFTLCouplerRequestSupport(c, &ms, &ws0, &r2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c, r2, &w2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      MFTLCouplerReleaseRequest(&r2);
      TEST_ASSERT(agreed(MFTLCouplerBind(c, &ms, w2, &b2) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
      TEST_ASSERT(agreed(same_rows(b, b2), MPI_COMM_WORLD));
      MFTLCouplerReleaseBinding(&b2);
      MFTLCouplerReleaseWindow(&w2);
    }
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
  }
  MFTLCouplerDestroy(&c);
  adv_free_blocks(&B);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A real redistribution: a sphere translated by less than the skin and
 *        redistributed -- vertices change owner -- binds in the window
 *        requested before, on every rank, with no request.
 *
 * Without the allowance this connected surface binds too, most of the time:
 * the new owner's own vertices next to the migrated ones keep the same bins in
 * its window. The allowance is what makes it a guarantee rather than a
 * property of the mesh -- test_a_migrated_user_marker_binds_in_the_old_window
 * is the case where nothing else covers the migrated marker.
 */
static int test_a_redistributed_surface_binds_in_the_old_window(void)
{
  const double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  const bool pp[3] = {true, true, true};
  adv_lat A;
  memset(&A, 0, sizeof A);
  for (int k = 0; k < 3; ++k)
  {
    A.N[k] = 20;
    A.h[k] = 0.05;
    A.origin[k] = 0.025;
    A.L[k] = 1.0;
    A.per[k] = true;
  }
  int ok_all = 1;
  long migrated_any = 0;
  for (int mig = 1; mig >= 1; --mig)
  {
    MFTL_Ensemble *ens = make_ens(lo, hi, pp);
    /* off-centre: a sphere centred ON a rank boundary has vertices on the plane
     * and none within a small translation of it, so nothing would migrate */
    const double cc[3] = {0.4931, 0.4877, 0.5113};
    add_sphere(ens, cc, 0.3, 3);
    scatter(ens);
    MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
    const int loc = adv_add_location(g, "cells", &A, MFTL_BOUNDARY_TRUNCATE);
    adv_blocks B;
    adv_decompose(g, loc, &A, 2 * g_nproc, 616u, 1, 1, false, &B);
    MFTL_Coupler *c = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_MarkerSpec ms;
    memset(&ms, 0, sizeof ms);
    ms.kind = MFTL_MARKERS_VERTICES;
    const double skin = 0.02;
    MFTL_WindowSpec ws = {{skin, skin, skin}, {mig ? skin : 0.0, mig ? skin : 0.0, mig ? skin : 0.0}};
    MFTL_SupportRequest *req = MFTL_NULL;
    MFTL_CouplingWindow *w = MFTL_NULL;
    TEST_ASSERT(agreed(MFTLCouplerRequestSupport(c, &ms, &ws, &req) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    TEST_ASSERT(agreed(MFTLCouplerPrepareWindow(c, req, &w) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTLCouplerReleaseRequest(&req);
    /* owners of every vertex before */
    int64_t nv = 0;
    for (int o = 0; o < ens->n_local_objects; ++o) nv += ens->local_objects[o]->verts.n_local;
    double *before = (double *) malloc((size_t) (nv > 0 ? 2 * nv : 1) * sizeof(double));
    int64_t at = 0;
    for (int o = 0; o < ens->n_local_objects; ++o)
      for (int v = 0; v < ens->local_objects[o]->verts.n_local; ++v, ++at)
      {
        before[2 * at] = (double) ens->local_objects[o]->verts.rec[v].global_id;
        before[2 * at + 1] = (double) g_rank;
      }
    int tot = 0;
    double *all_before = allgather_doubles(before, (int) nv, 2, &tot);
    /* translate by 0.6 skin along a diagonal, wrap, announce, redistribute */
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTL_Object *obj = ens->local_objects[o];
      for (int v = 0; v < obj->verts.n; ++v)
      {
        for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] += 0.6 * skin;
        MFTLApplyPeriodicity(&ens->domain, &obj->verts.pos[3 * v]);
      }
    }
    MFTLEnsembleMarkChanged(ens);
    scatter(ens);
    long migrated = 0;
    for (int o = 0; o < ens->n_local_objects; ++o)
      for (int v = 0; v < ens->local_objects[o]->verts.n_local; ++v)
      {
        const int gid = ens->local_objects[o]->verts.rec[v].global_id;
        for (int i = 0; i < tot; ++i)
          if ((int) all_before[2 * i] == gid && (int) all_before[2 * i + 1] != g_rank) ++migrated;
      }
    migrated_any += migrated;
    MFTL_CouplingBinding *b = MFTL_NULL;
    const int st = MFTLCouplerBind(c, &ms, w, &b);
    ok_all = ok_all && agreed(st == MFTL_COUPLING_OK, MPI_COMM_WORLD);
    MFTLCouplerReleaseBinding(&b);
    MFTLCouplerReleaseWindow(&w);
    free(all_before);
    free(before);
    MFTLCouplerDestroy(&c);
    adv_free_blocks(&B);
    MFTLSolverGridDestroy(&g);
    MFTLEnsembleDestroy(&ens);
  }
  TEST_ASSERT(ok_all);
  if (g_nproc > 1) TEST_ASSERT(anywhere(migrated_any > 0, MPI_COMM_WORLD)); /* non-vacuity: vertices changed owner */
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling lifecycle (np = %d) ===\n", g_nproc);

  RUN_TEST(test_the_solver_completes_the_halo_transpose);
  RUN_TEST(test_ghost_layers_aliasing_owned_samples_survive);
  RUN_TEST(test_local_stage_binds_are_local);
  RUN_TEST(test_a_selection_change_keeps_windows_that_stay_exported);
  RUN_TEST(test_a_migrated_user_marker_binds_in_the_old_window);
  RUN_TEST(test_a_redistributed_surface_binds_in_the_old_window);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
