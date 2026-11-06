/*
 * The EXACT surface of the LeVeque (Enright) deformation test, evaluated at
 * arbitrary points -- the reference every error in this study is measured
 * against.
 *
 * The velocity is u(x, t) = U(x) cos(pi t / T) with a fixed spatial field U, so
 * the flow map from 0 to t is the AUTONOMOUS flow of U over the pseudo-time
 *
 *     tau(t) = (T / pi) sin(pi t / T),
 *
 * and the surface at t is the initial sphere carried by it. A point x is traced
 * back by integrating dX/dtau = -U(X) over [0, tau(t)] together with its first
 * and second variational equations (J = dX0/dx, K = d2X0/dx2), which gives the
 * level set phi(x, t) = |X0 - c| - R, its gradient J^T g0 and its Hessian
 * J^T H0 J + sum_i g0_i K_i exactly up to the RK4 truncation error. Then
 *
 *     d = phi / |grad phi|           signed distance (first order; one Newton step)
 *     n = grad phi / |grad phi|      exact unit normal
 *     H = (tr S) / 2, K = ((tr S)^2 - |S|^2) / 2,  S = P Hess P / |grad phi|
 *
 * with H and K taken at the projected point x - d n, i.e. on the surface.
 *
 * usage: leveque_reference T t steps_per_unit in.bin out.bin
 *   in.bin : int64 N, then N*3 doubles
 *   out.bin: N*7 doubles: d, n[3], H, K, |grad phi| (at x)
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static const double PI = 3.14159265358979323846;
static const double C0[3] = {0.35, 0.35, 0.35};
static const double R0 = 0.15;

/* Component i is AMP[i] * prod_k F_{TYPE[i][k]}(x_k), F_1 = sin^2(pi x), F_0 = sin(2 pi x). */
static const int TYPE[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
static const double AMP[3] = {2.0, -1.0, -1.0};

static void field(const double x[3], double u[3], double A[3][3], double B[3][3][3], int need_b)
{
  double f[2][3], fp[2][3], fpp[2][3];
  for (int k = 0; k < 3; ++k)
  {
    const double s = sin(PI * x[k]), c = cos(PI * x[k]);
    const double s2 = 2.0 * s * c, c2 = c * c - s * s;  // sin(2 pi x), cos(2 pi x)
    f[1][k] = s * s;
    fp[1][k] = PI * s2;
    fpp[1][k] = 2.0 * PI * PI * c2;
    f[0][k] = s2;
    fp[0][k] = 2.0 * PI * c2;
    fpp[0][k] = -4.0 * PI * PI * s2;
  }
  for (int i = 0; i < 3; ++i)
  {
    double F[3], Fp[3], Fpp[3];
    for (int k = 0; k < 3; ++k)
    {
      F[k] = f[TYPE[i][k]][k];
      Fp[k] = fp[TYPE[i][k]][k];
      Fpp[k] = fpp[TYPE[i][k]][k];
    }
    u[i] = AMP[i] * F[0] * F[1] * F[2];
    A[i][0] = AMP[i] * Fp[0] * F[1] * F[2];
    A[i][1] = AMP[i] * F[0] * Fp[1] * F[2];
    A[i][2] = AMP[i] * F[0] * F[1] * Fp[2];
    if (!need_b) continue;
    B[i][0][0] = AMP[i] * Fpp[0] * F[1] * F[2];
    B[i][1][1] = AMP[i] * F[0] * Fpp[1] * F[2];
    B[i][2][2] = AMP[i] * F[0] * F[1] * Fpp[2];
    B[i][0][1] = B[i][1][0] = AMP[i] * Fp[0] * Fp[1] * F[2];
    B[i][0][2] = B[i][2][0] = AMP[i] * Fp[0] * F[1] * Fp[2];
    B[i][1][2] = B[i][2][1] = AMP[i] * F[0] * Fp[1] * Fp[2];
  }
}

/* y = [X(3) | J(9) | K(27)], dy/dtau = -(U, A J, A K + B J J). */
static void rhs(const double *y, double *dy, int need_k)
{
  double u[3], A[3][3], B[3][3][3];
  field(y, u, A, B, need_k);
  const double *J = y + 3, *K = y + 12;
  for (int i = 0; i < 3; ++i) dy[i] = -u[i];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
    {
      double s = 0.0;
      for (int l = 0; l < 3; ++l) s += A[i][l] * J[3 * l + j];
      dy[3 + 3 * i + j] = -s;
    }
  if (!need_k) return;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        double s = 0.0;
        for (int l = 0; l < 3; ++l) s += A[i][l] * K[9 * l + 3 * j + k];
        for (int l = 0; l < 3; ++l)
          for (int m = 0; m < 3; ++m) s += B[i][l][m] * J[3 * l + j] * J[3 * m + k];
        dy[12 + 9 * i + 3 * j + k] = -s;
      }
}

static void trace(const double x[3], double tau, int steps, int need_k, double *y)
{
  const int n = need_k ? 39 : 12;
  for (int i = 0; i < 39; ++i) y[i] = 0.0;
  for (int k = 0; k < 3; ++k) y[k] = x[k];
  y[3] = y[7] = y[11] = 1.0;
  if (tau <= 0.0) return;

  const double h = tau / steps;
  double k1[39], k2[39], k3[39], k4[39], yt[39];
  for (int s = 0; s < steps; ++s)
  {
    rhs(y, k1, need_k);
    for (int i = 0; i < n; ++i) yt[i] = y[i] + 0.5 * h * k1[i];
    rhs(yt, k2, need_k);
    for (int i = 0; i < n; ++i) yt[i] = y[i] + 0.5 * h * k2[i];
    rhs(yt, k3, need_k);
    for (int i = 0; i < n; ++i) yt[i] = y[i] + h * k3[i];
    rhs(yt, k4, need_k);
    for (int i = 0; i < n; ++i) y[i] += (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
  }
}

/* phi, grad phi and (if need_k) the Hessian at x. */
static double level_set(const double *y, double grad[3], double hess[3][3], int need_k)
{
  double g0[3], r = 0.0;
  for (int k = 0; k < 3; ++k)
  {
    g0[k] = y[k] - C0[k];
    r += g0[k] * g0[k];
  }
  r = sqrt(r);
  for (int k = 0; k < 3; ++k) g0[k] /= r;
  const double *J = y + 3, *K = y + 12;
  for (int j = 0; j < 3; ++j)
  {
    grad[j] = 0.0;
    for (int i = 0; i < 3; ++i) grad[j] += g0[i] * J[3 * i + j];
  }
  if (need_k)
  {
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        double s = 0.0;
        for (int i = 0; i < 3; ++i)
          for (int l = 0; l < 3; ++l) s += (((i == l) ? 1.0 : 0.0) - g0[i] * g0[l]) / r * J[3 * i + j] * J[3 * l + k];
        for (int i = 0; i < 3; ++i) s += g0[i] * K[9 * i + 3 * j + k];
        hess[j][k] = s;
      }
  }
  return r - R0;
}

int main(int argc, char **argv)
{
  if (argc != 6)
  {
    fprintf(stderr, "usage: %s T t steps_per_unit in.bin out.bin\n", argv[0]);
    return 1;
  }
  const double T = atof(argv[1]), t = atof(argv[2]);
  const double per_unit = atof(argv[3]);
  const double tau = (T / PI) * sin(PI * t / T);
  const int steps = (tau > 1e-12) ? (int) ceil(tau * per_unit) : 0;

  FILE *f = fopen(argv[4], "rb");
  int64_t n = 0;
  if (!f || fread(&n, sizeof n, 1, f) != 1) return 1;
  double *x = malloc((size_t) n * 3 * sizeof(double));
  double *out = malloc((size_t) n * 7 * sizeof(double));
  if (fread(x, sizeof(double), (size_t) n * 3, f) != (size_t) n * 3) return 1;
  fclose(f);

  for (int64_t p = 0; p < n; ++p)
  {
    double y[39], grad[3], hess[3][3];
    trace(x + 3 * p, tau, steps, 0, y);
    const double phi = level_set(y, grad, hess, 0);
    const double g = sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
    double nrm[3], xs[3];
    for (int k = 0; k < 3; ++k) nrm[k] = grad[k] / g;
    const double d = phi / g;
    for (int k = 0; k < 3; ++k) xs[k] = x[3 * p + k] - d * nrm[k];

    trace(xs, tau, steps, 1, y);
    double grad_s[3];
    level_set(y, grad_s, hess, 1);
    const double gs = sqrt(grad_s[0] * grad_s[0] + grad_s[1] * grad_s[1] + grad_s[2] * grad_s[2]);
    double ns[3], S[3][3];
    for (int k = 0; k < 3; ++k) ns[k] = grad_s[k] / gs;
    double P[3][3];
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b) P[a][b] = ((a == b) ? 1.0 : 0.0) - ns[a] * ns[b];
    double trS = 0.0, normS2 = 0.0;
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
      {
        double s = 0.0;
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) s += P[a][i] * hess[i][j] * P[j][b];
        S[a][b] = s / gs;
      }
    for (int a = 0; a < 3; ++a)
    {
      trS += S[a][a];
      for (int b = 0; b < 3; ++b) normS2 += S[a][b] * S[a][b];
    }
    double *o = out + 7 * p;
    o[0] = d;
    o[1] = nrm[0];
    o[2] = nrm[1];
    o[3] = nrm[2];
    o[4] = 0.5 * trS;
    o[5] = 0.5 * (trS * trS - normS2);
    o[6] = g;
  }

  f = fopen(argv[5], "wb");
  if (!f) return 1;
  fwrite(out, sizeof(double), (size_t) n * 7, f);
  fclose(f);
  free(x);
  free(out);
  return 0;
}
