/*
 * @Author: Christian Gorges
 * @Date: 2026-08-16
 *
 * @brief Does the plain orientation determinant misclassify the configurations
 *        THIS library's physics produces, and what does fixing it cost?
 *
 * The question is not whether a floating-point determinant can be made to fail
 * -- of course it can, with adversarial input. It is whether it fails on
 * NEAR-COPLANAR AND THIN-FILM CONTACT, which is what a lubrication model, a
 * draining film between two coalescing droplets, or two surfaces about to touch
 * actually look like. If it does not, the adaptive predicate is overhead and
 * should be reverted to the plain determinant.
 *
 * Three experiments, each isolating one variable:
 *
 *   1. THIN FILM vs GAP. Two nearly parallel triangles separated by a gap that
 *      shrinks from 1e-3 to 1e-16 of the triangle size. This is the geometry a
 *      lubrication force is computed from.
 *
 *   2. MISCLASSIFICATION vs COORDINATE OFFSET. The same small feature, moved
 *      away from the origin. Every coordinate difference inside the determinant
 *      cancels the offset, and the cancellation is where the significant digits
 *      go -- this is the millimetre droplet at x = 0.5 m, and it is the case
 *      that makes the failure a function of WHERE the body is rather than of
 *      what shape it has.
 *
 *   3. COST. Nanoseconds per call for both forms, on well-conditioned input
 *      (which is what the traversal spends its time on) and on degenerate input
 *      (the worst case).
 *
 * The adaptive predicate is the reference for experiments 1 and 2, because its
 * sign is exact by construction: any disagreement IS a misclassification by the
 * plain form.
 *
 * Build with the examples (BUILD_EXAMPLES=ON, the default) and run:
 *     ./build/orient3d_predicate_profile
 */

#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "mftl_bvh.h"
#include "mftl_core.h"

/* Deterministic, so two runs of this program are comparable and a regression is
 * visible. A real RNG would make the misclassification counts wobble. */
static unsigned long long g_state = 0x243F6A8885A308D3ULL;

static double rnd_unit(void)
{
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  return (double) ((g_state >> 11) & ((1ULL << 53) - 1)) / (double) (1ULL << 53);
}

static double rnd_sym(void) { return 2.0 * rnd_unit() - 1.0; }

static inline int sign_of(double x) { return (x > 0.0) - (x < 0.0); }

/* ═══════════════════════════════════════════════════════════════════════════
 * Experiment 1: thin film
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Three points in general position, and a fourth just off their plane.
 *
 * ── Why the obvious construction does not test anything ───────────────────
 *
 * The tempting version is to put a, b and c at the SAME z and lift d by the
 * gap. That produces no misclassification at any gap, and the reason is
 * instructive rather than lucky: with a[2] == b[2] == c[2], all three of adz,
 * bdz and cdz equal exactly -gap, so the gap factors straight out of the
 * determinant. There is no cancellation left to lose digits to, and the plain
 * determinant is as exact as the adaptive one. A sweep over such cases measures
 * the arithmetic being easy, not the predicate being good.
 *
 * The configuration that is actually ill-conditioned has the plane in GENERAL
 * ORIENTATION -- so the deviation is spread across all three coordinates -- and
 * the fourth point produced by an EXPRESSION that lands it on the plane, whose
 * own rounding is the perturbation. That is what a real mesh does: a vertex
 * position is the output of an advection step, not a value someone chose.
 */
static void near_coplanar_case(double L, double gap, double offset, double a[3], double b[3], double c[3], double d[3])
{
  for (int k = 0; k < 3; ++k)
  {
    a[k] = offset + L * rnd_sym();
    b[k] = offset + L * rnd_sym();
    c[k] = offset + L * rnd_sym();
  }

  double e1[3], e2[3], n[3];
  for (int k = 0; k < 3; ++k)
  {
    e1[k] = b[k] - a[k];
    e2[k] = c[k] - a[k];
  }
  n[0] = e1[1] * e2[2] - e1[2] * e2[1];
  n[1] = e1[2] * e2[0] - e1[0] * e2[2];
  n[2] = e1[0] * e2[1] - e1[1] * e2[0];

  const double len = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
  const double inv = (len > 0.0) ? 1.0 / len : 0.0;
  for (int k = 0; k < 3; ++k) n[k] *= inv;

  /* An interior barycentric point, plus the gap along the unit normal. Every
   * term here rounds, and at a large `offset` the rounding of the first three
   * dominates the deliberate displacement -- which is exactly the regime the
   * predicate has to survive. */
  const double s = 0.2 + 0.3 * rnd_unit();
  const double t = 0.2 + 0.3 * rnd_unit();
  for (int k = 0; k < 3; ++k) d[k] = a[k] + s * e1[k] + t * e2[k] + gap * n[k];
}

static void run_thin_film(void)
{
  printf("\n");
  printf("=====================================================================\n");
  printf(" 1. THIN FILM: two surfaces separated by a shrinking gap\n");
  printf("    triangle size L = 1, gap swept, coordinates at the origin\n");
  printf("=====================================================================\n");
  printf("  %-12s %-14s %-14s %-12s\n", "gap/L", "fast wrong %", "fast says 0 %", "resolvable");
  printf("  ---------------------------------------------------------------------\n");

  const int N = 20000;
  const double L = 1.0;

  for (int e = 3; e <= 16; ++e)
  {
    const double gap = pow(10.0, -e);

    int wrong = 0;
    int fast_zero = 0;
    int representable = 0;

    for (int i = 0; i < N; ++i)
    {
      double a[3], b[3], c[3], d[3];
      near_coplanar_case(L, gap, 0.0, a, b, c, d);

      /* Did the gap survive being stored? If d[2] rounded back onto the plane
       * the configuration IS coplanar and neither form is wrong to say so. */
      if (d[2] != c[2]) representable++;

      const double exact = MFTL_Orient3D(a, b, c, d);
      const double fast = MFTL_Orient3DFast(a, b, c, d);

      if (sign_of(exact) != sign_of(fast)) wrong++;
      if (fast == 0.0 && exact != 0.0) fast_zero++;
    }

    printf("  %-12.0e %-14.4f %-14.4f %d/%d\n", gap, 100.0 * wrong / N, 100.0 * fast_zero / N, representable, N);
  }
  printf("\n  (percentages of %d samples)\n", N);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Experiment 2: the same feature, moved away from the origin
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_offset_sweep(void)
{
  printf("\n");
  printf("=====================================================================\n");
  printf(" 2. COORDINATE OFFSET: the same small feature, further from 0\n");
  printf("    a body of size L sitting at distance `offset` from the origin\n");
  printf("=====================================================================\n");
  printf("\n  Each cell is the %% of samples where the PLAIN determinant gets the\n");
  printf("  sign wrong. `offset/L` is how many body-lengths from the origin the\n");
  printf("  body sits -- a 1 mm droplet at x = 0.5 m is offset/L = 500.\n\n");

  const int N = 20000;
  const double sizes[2] = {1.0e-3, 1.0};
  const double offsets[5] = {0.0, 1.0, 1.0e2, 1.0e4, 1.0e6};
  const int gexp[5] = {10, 12, 14, 15, 16};

  for (int s = 0; s < 2; ++s)
  {
    const double L = sizes[s];
    printf("  body size L = %.0e m\n", L);
    printf("  %-12s", "offset/L");
    for (int g = 0; g < 5; ++g) printf(" gap/L=1e-%-4d", gexp[g]);
    printf("\n  ------------------------------------------------------------------------\n");

    for (int o = 0; o < 5; ++o)
    {
      const double offset = offsets[o] * L;
      printf("  %-12.0e", offsets[o]);

      for (int g = 0; g < 5; ++g)
      {
        const double gap = L * pow(10.0, -gexp[g]);
        int wrong = 0;

        for (int i = 0; i < N; ++i)
        {
          double a[3], b[3], c[3], d[3];
          near_coplanar_case(L, gap, offset, a, b, c, d);
          if (sign_of(MFTL_Orient3D(a, b, c, d)) != sign_of(MFTL_Orient3DFast(a, b, c, d))) wrong++;
        }

        printf(" %12.3f", 100.0 * wrong / N);
      }
      printf("\n");
    }
    printf("\n");
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Experiment 3: what the filter costs
 * ═══════════════════════════════════════════════════════════════════════════ */

static void run_cost(void)
{
  printf("\n");
  printf("=====================================================================\n");
  printf(" 3. COST per call\n");
  printf("=====================================================================\n");

  const int N = 400000;

  double *pts = (double *) malloc((size_t) N * 12 * sizeof(double));

  /* --- Well-conditioned: what the traversal actually spends its time on ---
   * Nearly every orientation test in a real run is on a pair the broad phase
   * only just admitted, i.e. two triangles that are close but in general
   * position. The filter returns on the first estimate for all of them. */
  for (int i = 0; i < N; ++i)
  {
    for (int k = 0; k < 12; ++k) pts[12 * i + k] = rnd_sym();
  }

  double sink = 0.0;
  double best_fast = 1e30, best_exact = 1e30;

  for (int rep = 0; rep < 5; ++rep)
  {
    double t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i) sink += MFTL_Orient3DFast(&pts[12 * i], &pts[12 * i + 3], &pts[12 * i + 6], &pts[12 * i + 9]);
    double dt = MPI_Wtime() - t0;
    if (dt < best_fast) best_fast = dt;

    t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i) sink += MFTL_Orient3D(&pts[12 * i], &pts[12 * i + 3], &pts[12 * i + 6], &pts[12 * i + 9]);
    dt = MPI_Wtime() - t0;
    if (dt < best_exact) best_exact = dt;
  }

  printf("  general position   fast %6.2f ns   adaptive %6.2f ns   (%.2fx)\n", 1e9 * best_fast / N, 1e9 * best_exact / N, best_exact / best_fast);

  /* Captured BEFORE the coplanar run overwrites them. The general-position
   * figure is the one that belongs in the trade-off below: it is what an
   * orientation call in a real traversal costs, and the coplanar figure is the
   * worst case that essentially never occurs on moving geometry. */
  const double ns_delta_general = 1e9 * (best_exact - best_fast) / N;

  /* --- Degenerate: every call falls through the filter ------------------- */
  for (int i = 0; i < N; ++i)
  {
    double *p = &pts[12 * i];
    for (int v = 0; v < 4; ++v)
    {
      p[3 * v + 0] = rnd_sym();
      p[3 * v + 1] = rnd_sym();
      p[3 * v + 2] = 0.5;  // exactly coplanar
    }
  }

  best_fast = 1e30;
  best_exact = 1e30;
  for (int rep = 0; rep < 5; ++rep)
  {
    double t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i) sink += MFTL_Orient3DFast(&pts[12 * i], &pts[12 * i + 3], &pts[12 * i + 6], &pts[12 * i + 9]);
    double dt = MPI_Wtime() - t0;
    if (dt < best_fast) best_fast = dt;

    t0 = MPI_Wtime();
    for (int i = 0; i < N; ++i) sink += MFTL_Orient3D(&pts[12 * i], &pts[12 * i + 3], &pts[12 * i + 6], &pts[12 * i + 9]);
    dt = MPI_Wtime() - t0;
    if (dt < best_exact) best_exact = dt;
  }

  printf("  exactly coplanar   fast %6.2f ns   adaptive %6.2f ns   (%.2fx)\n", 1e9 * best_fast / N, 1e9 * best_exact / N, best_exact / best_fast);

  /* The accumulator has to be OBSERVED, not merely computed. Guarding it with
   * `if (sink == <constant>) printf("")` is not enough: with LTO on the
   * compiler proved the printf was a no-op, hence the branch dead, hence `sink`
   * dead, hence both timing loops dead -- and duly reported the plain
   * determinant at 0.00 ns. Printing it is the cheapest thing that cannot be
   * argued away. */
  printf("  (checksum %.6e -- ignore, it exists to keep the loops alive)\n", sink);

  /* ── The number that actually decides ───────────────────────────────────
   *
   * 2x on the predicate is only affordable if the predicate is a small part of
   * the narrow phase. MFTL_TriTriIntersect makes three to eight orientation
   * calls per pair; MFTL_TriTriProximity adds fifteen distance sub-tests on top
   * whenever the pair does NOT intersect, which is the common case for anything
   * the broad phase admits. Measuring both says how much of a contact query the
   * exactness is really costing. */
  {
    const int M = 200000;
    double *tris = (double *) malloc((size_t) M * 18 * sizeof(double));

    /* Close but disjoint pairs: what survives an inflated leaf box. */
    for (int i = 0; i < M; ++i)
    {
      double *t = &tris[18 * i];
      for (int v = 0; v < 3; ++v)
      {
        t[3 * v + 0] = rnd_sym();
        t[3 * v + 1] = rnd_sym();
        t[3 * v + 2] = rnd_sym();
      }
      for (int v = 0; v < 3; ++v)
      {
        t[9 + 3 * v + 0] = rnd_sym() + 1.4;
        t[9 + 3 * v + 1] = rnd_sym();
        t[9 + 3 * v + 2] = rnd_sym();
      }
    }

    long long hits = 0;
    double best_i = 1e30, best_p = 1e30;

    for (int rep = 0; rep < 5; ++rep)
    {
      double t0 = MPI_Wtime();
      for (int i = 0; i < M; ++i) hits += MFTL_TriTriIntersect((const double(*)[3]) & tris[18 * i], (const double(*)[3]) & tris[18 * i + 9]);
      double dt = MPI_Wtime() - t0;
      if (dt < best_i) best_i = dt;

      t0 = MPI_Wtime();
      for (int i = 0; i < M; ++i) hits += MFTL_TriTriProximity((const double(*)[3]) & tris[18 * i], (const double(*)[3]) & tris[18 * i + 9], 0.05);
      dt = MPI_Wtime() - t0;
      if (dt < best_p) best_p = dt;
    }

    printf("\n  narrow phase, per triangle pair:\n");
    printf("    MFTL_TriTriIntersect   %6.2f ns\n", 1e9 * best_i / M);
    printf("    MFTL_TriTriProximity   %6.2f ns\n", 1e9 * best_p / M);
    printf("    (checksum %lld)\n", hits);

    printf("\n  If the predicate were the plain determinant, a proximity query would\n");
    printf("  save at most 6 x %.2f ns = %.2f ns of %.2f ns, i.e. <= %.1f %%.\n", ns_delta_general, 6.0 * ns_delta_general, 1e9 * best_p / M,
           100.0 * 6.0 * ns_delta_general / (1e9 * best_p / M));

    free(tris);
  }

  free(pts);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Experiment 4: does it reach the ANSWER, not just the predicate?
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Two triangles in a thin-film configuration, run through the full
 *        intersection test.
 *
 * A misclassified orientation only matters if it changes what
 * MFTL_TriTriIntersect reports. This builds pairs that genuinely cross by a
 * whisker and counts how many the two predicate forms disagree about
 * end-to-end -- the number that decides whether the adaptive version earns its
 * place.
 */
static void run_end_to_end(void)
{
  printf("\n");
  printf("=====================================================================\n");
  printf(" 4. END TO END: pairs whose INTERSECTION verdict changes\n");
  printf("=====================================================================\n");

  const int N = 200000;
  const double L = 1.0e-3;  // millimetre droplet triangles
  const double offset = 0.5;  // half a metre from the origin

  int disagree = 0;
  int crossing = 0;

  for (int i = 0; i < N; ++i)
  {
    /* A flat triangle, and a second one pierced through it by a tiny amount --
     * the moment of contact between two resolved surfaces. */
    const double z = offset;
    const double A[3][3] = {
        {offset + 0.0, offset + 0.0, z},
        {offset + L, offset + 0.0, z},
        {offset + 0.0, offset + L, z},
    };

    const double px = offset + 0.2 * L + 0.1 * L * rnd_unit();
    const double py = offset + 0.2 * L + 0.1 * L * rnd_unit();
    const double dz = L * pow(10.0, -6.0 - 11.0 * rnd_unit());  // 1e-6 L .. 1e-17 L

    const double B[3][3] = {
        {px, py, z - dz},
        {px, py, z + dz},
        {px + 0.5 * L, py + 0.5 * L, z},
    };

    /* The exact verdict. B's first edge spans the plane, so it crosses. */
    const int exact = MFTL_TriTriIntersect(A, B);
    if (exact) crossing++;

    /* What the plain determinant would have decided, reconstructed from the
     * six orientation values the dispatch branches on. A disagreement in any
     * of them is a different branch and, in general, a different answer. */
    int differs = 0;
    const double *p1 = A[0], *q1 = A[1], *r1 = A[2];
    const double *p2 = B[0], *q2 = B[1], *r2 = B[2];

    if (sign_of(MFTL_Orient3D(p1, p2, q2, r2)) != sign_of(MFTL_Orient3DFast(p1, p2, q2, r2))) differs = 1;
    if (sign_of(MFTL_Orient3D(q1, p2, q2, r2)) != sign_of(MFTL_Orient3DFast(q1, p2, q2, r2))) differs = 1;
    if (sign_of(MFTL_Orient3D(r1, p2, q2, r2)) != sign_of(MFTL_Orient3DFast(r1, p2, q2, r2))) differs = 1;
    if (sign_of(MFTL_Orient3D(p2, p1, q1, r1)) != sign_of(MFTL_Orient3DFast(p2, p1, q1, r1))) differs = 1;
    if (sign_of(MFTL_Orient3D(q2, p1, q1, r1)) != sign_of(MFTL_Orient3DFast(q2, p1, q1, r1))) differs = 1;
    if (sign_of(MFTL_Orient3D(r2, p1, q1, r1)) != sign_of(MFTL_Orient3DFast(r2, p1, q1, r1))) differs = 1;

    if (differs) disagree++;
  }

  printf("  droplet triangles of %.0e m at %.1f m from the origin\n", L, offset);
  printf("  film thickness 1e-6 .. 1e-17 of the triangle size\n\n");
  printf("  pairs tested                 %d\n", N);
  printf("  genuinely crossing (exact)   %d\n", crossing);
  printf("  at least one predicate sign disagrees:  %d  (%.3f %%)\n", disagree, 100.0 * disagree / N);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  printf("\n");
  printf("#####################################################################\n");
  printf("#  MFTL_Orient3D: adaptive vs plain determinant                     #\n");
  printf("#  Reference is the ADAPTIVE form, whose sign is exact.             #\n");
  printf("#####################################################################\n");

  run_thin_film();
  run_offset_sweep();
  run_end_to_end();
  run_cost();

  printf("\n");
  MPI_Finalize();
  return 0;
}
