/*
 * @Author: Christian Gorges
 * @Date: 2026-10-07
 *
 * @brief Anisotropic point locations: samples with a full cell matrix, rows in
 *        the cell's own frame.
 *
 * The fixture is a ROTATED, ANISOTROPIC lattice of points -- X = X0 + C (i,j,k)
 * with C = R diag(a, b, c) -- registered as an unstructured cloud with C as
 * every sample's cell. In the frame of C it is a complete unit lattice, so a
 * tensor kernel there has its lattice moments: Peskin's rows need no moment
 * correction and stay nonnegative, exactly as on an axis-aligned lattice. The
 * axis-aligned treatment of the same cloud does not have that property, and
 * the test shows the difference rather than assuming it.
 */

#include "test_coupling_irregular.h"

/** @brief The rotated anisotropic lattice: its cell, origin, and index range covering the box interior. */
typedef struct
{
  double C[9];  ///< row-major, columns the edge vectors
  double Ci[9];
  double X0[3];
  int lo, hi;  ///< index range per axis enumerated (samples outside the box are dropped)
  bool varying;  ///< the cell rotates with position (then C is the cell at X0)
} rot_lattice;

/** @brief The test's own 3x3 inverse (cofactors) and determinant. */
static void inv3(const double M[9], double I[9], double *det)
{
  const double a = M[0], b = M[1], c = M[2], d = M[3], e = M[4], f = M[5], g = M[6], h = M[7], i = M[8];
  const double D = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  *det = D;
  I[0] = (e * i - f * h) / D;
  I[1] = (c * h - b * i) / D;
  I[2] = (b * f - c * e) / D;
  I[3] = (f * g - d * i) / D;
  I[4] = (a * i - c * g) / D;
  I[5] = (c * d - a * f) / D;
  I[6] = (d * h - e * g) / D;
  I[7] = (b * g - a * h) / D;
  I[8] = (a * e - b * d) / D;
}

static void mat_mul(const double A[9], const double B[9], double out[9])
{
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c) out[3 * r + c] = A[3 * r] * B[c] + A[3 * r + 1] * B[3 + c] + A[3 * r + 2] * B[6 + c];
}

static void rot_xz(double ax, double az, double R[9])
{
  const double cx = cos(ax), sx = sin(ax), cz = cos(az), sz = sin(az);
  const double Rz[9] = {cz, -sz, 0, sz, cz, 0, 0, 0, 1}, Rx[9] = {1, 0, 0, 0, cx, -sx, 0, sx, cx};
  mat_mul(Rx, Rz, R);
}

static rot_lattice rot_make(void)
{
  rot_lattice L;
  memset(&L, 0, sizeof L);
  double R[9];
  rot_xz(0.35, 0.52, R);
  const double D[9] = {0.045, 0, 0, 0, 0.06, 0, 0, 0, 0.052};
  mat_mul(R, D, L.C);
  double det;
  inv3(L.C, L.Ci, &det);
  L.X0[0] = 0.5013;
  L.X0[1] = 0.4987;
  L.X0[2] = 0.5021;
  L.lo = -16;
  L.hi = 16;
  return L;
}

/** @brief The cell at a point of the varying fixture: rotation growing with x. */
static void cell_at(const rot_lattice *L, const double x[3], double C[9])
{
  if (!L->varying)
  {
    memcpy(C, L->C, 9 * sizeof(double));
    return;
  }
  double R[9];
  rot_xz(0.35 + 0.8 * (x[0] - 0.5), 0.52 - 0.6 * (x[1] - 0.5), R);
  const double D[9] = {0.045, 0, 0, 0, 0.06, 0, 0, 0, 0.052};
  mat_mul(R, D, C);
}

/** @brief Sample (i,j,k): position; false if outside the unit box (with a margin). */
static bool rot_sample(const rot_lattice *L, int i, int j, int k, double x[3])
{
  const double u[3] = {(double) i, (double) j, (double) k};
  for (int r = 0; r < 3; ++r) x[r] = L->X0[r] + L->C[3 * r] * u[0] + L->C[3 * r + 1] * u[1] + L->C[3 * r + 2] * u[2];
  if (L->varying)
  {
    /* a smooth curvilinear jitter: the lattice bends, so do the cells */
    x[0] += 0.01 * sin(2.0 * x[1]);
    x[1] += 0.01 * sin(2.0 * x[2]);
  }
  for (int r = 0; r < 3; ++r)
    if (x[r] < 0.02 || x[r] > 0.98) return false;
  return true;
}

static uint64_t rot_key(int i, int j, int k) { return (uint64_t) ((i + 64) + 128 * ((j + 64) + 128 * (k + 64))); }

/** @brief Registers this rank's share (z-slabs) of the rotated lattice at one location. */
static int rot_register(MFTL_SolverGrid *g, const rot_lattice *L, bool aniso, int64_t *n_out, double **pos_out, uint64_t **key_out)
{
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = aniso ? "rotated" : "aligned";
  ld.lattice.kind = MFTL_LATTICE_NONE;
  ld.anisotropic = aniso;
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  const int span = L->hi - L->lo + 1;
  const int64_t cap = (int64_t) span * span * span;
  double *pos = (double *) malloc(3 * (size_t) cap * sizeof(double)), *meas = (double *) malloc((size_t) cap * sizeof(double));
  double *cell = (double *) malloc(9 * (size_t) cap * sizeof(double)), *h = (double *) malloc(3 * (size_t) cap * sizeof(double));
  uint64_t *key = (uint64_t *) malloc((size_t) cap * sizeof(uint64_t));
  int64_t n = 0;
  for (int i = L->lo; i <= L->hi; ++i)
    for (int j = L->lo; j <= L->hi; ++j)
      for (int k = L->lo; k <= L->hi; ++k)
      {
        double x[3];
        if (!rot_sample(L, i, j, k, x)) continue;
        const int owner = (int) (x[2] * g_nproc);
        if ((owner < g_nproc ? owner : g_nproc - 1) != g_rank) continue;
        memcpy(&pos[3 * n], x, sizeof x);
        double C[9], Ci[9], det;
        cell_at(L, x, C);
        inv3(C, Ci, &det);
        memcpy(&cell[9 * n], C, sizeof C);
        meas[n] = fabs(det);
        for (int r = 0; r < 3; ++r) h[3 * n + r] = fabs(C[3 * r]) + fabs(C[3 * r + 1]) + fabs(C[3 * r + 2]);
        key[n] = rot_key(i, j, k);
        ++n;
      }
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.n = n;
  pd.keys = key;
  pd.pos = pos;
  pd.measure = meas;
  pd.h = h;
  pd.cell = aniso ? cell : MFTL_NULL;
  MFTLSolverGridAddPoints(g, &pd);
  free(meas);
  free(cell);
  free(h);
  *n_out = n;
  *pos_out = pos;
  *key_out = key;
  return loc;
}

/** @brief Brute force over the global rotated lattice: candidates in the parallelepiped support, the row core with the cell. */
static int rot_ref_row(const rot_lattice *L, const MFTL_CouplerParams *P, const double x[3], uint64_t *keys, double *w, int cap)
{
  double *X = (double *) malloc(3 * (size_t) cap * sizeof(double)), *q = (double *) malloc((size_t) cap * sizeof(double));
  int n = 0;
  double det;
  double Ci[9];
  inv3(L->C, Ci, &det);
  for (int i = L->lo; i <= L->hi; ++i)
    for (int j = L->lo; j <= L->hi; ++j)
      for (int k = L->lo; k <= L->hi; ++k)
      {
        double p[3];
        if (!rot_sample(L, i, j, k, p)) continue;
        const double r[3] = {p[0] - x[0], p[1] - x[1], p[2] - x[2]};
        bool in = true;
        for (int a = 0; a < 3; ++a)
          if (fabs(Ci[3 * a] * r[0] + Ci[3 * a + 1] * r[1] + Ci[3 * a + 2] * r[2]) >= 2.0 * P->support_scale) in = false;
        if (!in || n >= cap) continue;
        memcpy(&X[3 * n], p, sizeof p);
        q[n] = fabs(det);
        keys[n++] = rot_key(i, j, k);
      }
  const int st = MFTLCouplingAssembleRowCell(P, x, L->C, n, X, q, w, MFTL_NULL);
  free(X);
  free(q);
  return (st == MFTL_COUPLING_OK) ? n : -1;
}

/** @brief Markers well inside the cloud. */
static int rot_markers(double *all, int cap)
{
  uint64_t s = 8086u;
  int n = 0;
  for (; n < cap; ++n)
    for (int k = 0; k < 3; ++k) all[3 * n + k] = 0.32 + 0.36 * adv_unif(&s);
  return n;
}

static MFTL_Ensemble *rot_box(void)
{
  int cuts[3];
  test_grid_for(g_nproc, cuts);
  double lo[3] = {0, 0, 0}, hi[3] = {1, 1, 1};
  bool per[3] = {false, false, false};
  return MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, per, 4);
}

/**
 * @brief A rotated anisotropic lattice registered as an anisotropic cloud:
 *        rows equal brute force in the cell's frame, need NO moment correction,
 *        are nonnegative, interpolate an affine field exactly and spread as
 *        their transpose. Registered axis-aligned instead, the same cloud needs
 *        corrections everywhere.
 */
static int test_a_rotated_lattice_in_its_own_frame(void)
{
  rot_lattice L = rot_make();
  MFTL_Ensemble *ens = rot_box();
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  int64_t n_rot = 0, n_al = 0;
  double *pos_rot, *pos_al;
  uint64_t *key_rot, *key_al;
  const int loc_rot = rot_register(g, &L, true, &n_rot, &pos_rot, &key_rot);
  const int loc_al = rot_register(g, &L, false, &n_al, &pos_al, &key_al);
  MFTL_CouplerParams P;
  MFTLCouplerDefaultParams(&P);
  P.max_amplification = 10.0; /* the axis-aligned rows of this cloud are poor; they must still be built to be compared */
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, &P, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double all[3 * 40];
  const int na = rot_markers(all, 40);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t nm = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, na, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  /* rows of the anisotropic location against brute force */
  const double *bp = MFTLCouplerBindingPositions(b);
  uint64_t rk[2048];
  double rw[2048];
  double worst = 0.0, minw = 0.0;
  for (int64_t i = 0; i < nm; ++i)
  {
    int ne = 0;
    const uint64_t *keys;
    const double *wts;
    MFTLCouplerBindingRow(b, loc_rot, i, &ne, &keys, &wts);
    uint64_t kc[2048];
    double wc[2048];
    memcpy(kc, keys, (size_t) ne * sizeof(uint64_t));
    memcpy(wc, wts, (size_t) ne * sizeof(double));
    for (int e = 0; e < ne; ++e) minw = fmin(minw, wc[e]);
    const int nr = rot_ref_row(&L, &P, &bp[3 * i], rk, rw, 2048);
    worst = fmax(worst, nr < 0 ? 1.0 : adv_row_diff(ne, kc, wc, nr, rk, rw));
  }
  if (worst > 1e-12) printf("\n  rank %d: anisotropic rows off by %.3e", g_rank, worst);
  TEST_ASSERT(agreed(worst <= 1e-12, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(minw >= 0.0, MPI_COMM_WORLD));
  /* the quality per location: binding quality is over both locations, so bind each on its own */
  long corr[2] = {0, 0};
  for (int which = 0; which < 2; ++which)
  {
    MFTL_CouplingBinding *bl = MFTL_NULL;
    int64_t nn = 0;
    MFTL_CouplingWindow *wl = MFTL_NULL;
    const int lw = which ? loc_al : loc_rot;
    TEST_ASSERT(agreed(adv_bind_list(c, na, all, 1, &lw, &wl, &bl, &nn) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
    MFTL_BindingQuality q;
    MFTLCouplerBindingQuality(bl, &q);
    corr[which] = (long) q.n_corrected;
    MFTLCouplerReleaseBinding(&bl);
    MFTLCouplerReleaseWindow(&wl);
  }
  MPI_Allreduce(MPI_IN_PLACE, corr, 2, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(corr[0] == 0);  /* the cell's frame: a complete lattice */
  TEST_ASSERT(corr[1] == na); /* axis-aligned: every row corrected */

  /* an affine field, both locations: exact (the axis-aligned one through its correction) */
  affine Af = {0.3, {1.2, -0.7, 2.1}};
  double *u_rot = (double *) malloc((size_t) (n_rot > 0 ? n_rot : 1) * sizeof(double)), *u_al = (double *) malloc((size_t) (n_al > 0 ? n_al : 1) * sizeof(double));
  for (int64_t i = 0; i < n_rot; ++i) u_rot[i] = f_affine(&pos_rot[3 * i], &Af);
  for (int64_t i = 0; i < n_al; ++i) u_al[i] = f_affine(&pos_al[3 * i], &Af);
  MFTL_SolverField F[2];
  memset(F, 0, sizeof F);
  F[0].location = loc_rot;
  F[0].data = &u_rot;
  F[1].location = loc_al;
  F[1].data = &u_al;
  double *U = (double *) malloc((size_t) (nm > 0 ? 2 * nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 2, F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double aerr = 0.0;
  for (int64_t i = 0; i < nm; ++i)
    for (int f = 0; f < 2; ++f) aerr = fmax(aerr, fabs(U[2 * i + f] - f_affine(&bp[3 * i], &Af)));
  TEST_ASSERT(agreed(aerr <= 1e-12, MPI_COMM_WORLD));

  /* power identity on the anisotropic location */
  double *Ld = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double)), *T = (double *) calloc((size_t) (n_rot > 0 ? n_rot : 1), sizeof(double));
  const MFTL_MarkerRef *ref = MFTLCouplerBindingMarkers(b);
  for (int64_t i = 0; i < nm; ++i) Ld[i] = hash01(ref[i].key + 3u);
  for (int64_t i = 0; i < n_rot; ++i) u_rot[i] = hash01(key_rot[i]);
  MFTL_SolverField Fr = F[0], Ft = F[0];
  Ft.data = &T;
  double *Ur = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &Fr, Ur) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  TEST_ASSERT(agreed(MFTLCouplerSpread(c, b, 1, &Ft, Ld, MFTL_SPREAD_INTEGRATED) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double pw[3] = {0, 0, 0};
  for (int64_t i = 0; i < nm; ++i)
  {
    pw[0] += Ur[i] * Ld[i];
    pw[2] += fabs(Ur[i] * Ld[i]);
  }
  for (int64_t i = 0; i < n_rot; ++i) pw[1] += u_rot[i] * T[i];
  MPI_Allreduce(MPI_IN_PLACE, pw, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  TEST_ASSERT(fabs(pw[0] - pw[1]) <= 1e-13 * pw[2]);
  free(Ur);
  free(T);
  free(Ld);
  free(U);
  free(u_rot);
  free(u_al);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  free(pos_rot);
  free(pos_al);
  free(key_rot);
  free(key_al);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/**
 * @brief A bent lattice whose cells rotate with position: the marker's cell is
 *        the Shepard average of its neighbours', rows are corrected, and an
 *        affine field is still interpolated exactly.
 */
static int test_cells_that_rotate_with_position(void)
{
  rot_lattice L = rot_make();
  L.varying = true;
  MFTL_Ensemble *ens = rot_box();
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  int64_t n = 0;
  double *pos;
  uint64_t *key;
  const int loc = rot_register(g, &L, true, &n, &pos, &key);
  MFTL_Coupler *c = MFTL_NULL;
  TEST_ASSERT(agreed(MFTLCouplerCreate(ens, g, MFTL_NULL, &c) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  double all[3 * 30];
  const int na = rot_markers(all, 30);
  MFTL_CouplingWindow *w = MFTL_NULL;
  MFTL_CouplingBinding *b = MFTL_NULL;
  int64_t nm = 0;
  TEST_ASSERT(agreed(adv_bind_list(c, na, all, 0, MFTL_NULL, &w, &b, &nm) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  affine Af = {-0.2, {0.5, 1.5, -1.0}};
  double *u = (double *) malloc((size_t) (n > 0 ? n : 1) * sizeof(double));
  for (int64_t i = 0; i < n; ++i) u[i] = f_affine(&pos[3 * i], &Af);
  MFTL_SolverField F;
  memset(&F, 0, sizeof F);
  F.location = loc;
  F.data = &u;
  double *U = (double *) malloc((size_t) (nm > 0 ? nm : 1) * sizeof(double));
  TEST_ASSERT(agreed(MFTLCouplerInterpolate(c, b, 1, &F, U) == MFTL_COUPLING_OK, MPI_COMM_WORLD));
  const double *bp = MFTLCouplerBindingPositions(b);
  double aerr = 0.0;
  for (int64_t i = 0; i < nm; ++i) aerr = fmax(aerr, fabs(U[i] - f_affine(&bp[3 * i], &Af)));
  TEST_ASSERT(agreed(aerr <= 1e-12, MPI_COMM_WORLD));
  MFTL_BindingQuality q;
  MFTLCouplerBindingQuality(b, &q);
  TEST_ASSERT(anywhere(q.n_corrected > 0, MPI_COMM_WORLD));
  free(U);
  free(u);
  MFTLCouplerReleaseBinding(&b);
  MFTLCouplerReleaseWindow(&w);
  MFTLCouplerDestroy(&c);
  free(pos);
  free(key);
  MFTLSolverGridDestroy(&g);
  MFTLEnsembleDestroy(&ens);
  return 0;
}

/** @brief Cells that cannot be inverted, missing cells and an anisotropic lattice are refused at registration. */
static int test_invalid_cells_are_refused(void)
{
  MFTL_SolverGrid *g = MFTLSolverGridCreate(MPI_COMM_WORLD);
  MFTL_LocationDesc ld;
  memset(&ld, 0, sizeof ld);
  ld.name = "lat";
  ld.lattice.kind = MFTL_LATTICE_UNIFORM;
  ld.anisotropic = true;
  for (int k = 0; k < 3; ++k)
  {
    ld.lattice.dims[k] = 4;
    ld.lattice.spacing[k] = 0.25;
    ld.lattice.origin[k] = 0.125;
  }
  TEST_ASSERT(MFTLSolverGridAddLocation(g, &ld) == MFTL_ERROR);
  memset(&ld, 0, sizeof ld);
  ld.name = "cloud";
  ld.anisotropic = true;
  const int loc = MFTLSolverGridAddLocation(g, &ld);
  TEST_ASSERT(loc >= 0);
  const uint64_t key = 1;
  const double pos[3] = {0.5, 0.5, 0.5}, meas = 1e-3;
  const double singular[9] = {0.1, 0.1, 0.0, 0.2, 0.2, 0.0, 0.0, 0.0, 0.1}; /* two equal columns */
  const double nan_cell[9] = {0.1, 0, 0, 0, NAN, 0, 0, 0, 0.1};
  MFTL_PointSetDesc pd;
  memset(&pd, 0, sizeof pd);
  pd.location = loc;
  pd.n = 1;
  pd.keys = &key;
  pd.pos = pos;
  pd.measure = &meas;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR); /* no cell */
  pd.cell = singular;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  pd.cell = nan_cell;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == MFTL_ERROR);
  const double ok_cell[9] = {0.1, 0.02, 0.0, -0.01, 0.1, 0.0, 0.0, 0.03, 0.1};
  pd.cell = ok_cell;
  TEST_ASSERT(MFTLSolverGridAddPoints(g, &pd) == 0);
  MFTLSolverGridDestroy(&g);
  return 0;
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);
  MFTLRegisterMpiTypes();
  if (g_rank == 0) printf("\n=== Coupling anisotropic cells (np = %d) ===\n", g_nproc);

  RUN_TEST(test_a_rotated_lattice_in_its_own_frame);
  RUN_TEST(test_cells_that_rotate_with_position);
  RUN_TEST(test_invalid_cells_are_refused);

  if (g_rank == 0) TEST_REPORT();
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (g_tests_failed > 0) ? 1 : 0;
}
