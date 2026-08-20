/*
 * @Author: Christian Gorges
 * @Date: 2026-08-15
 *
 * @brief LeVeque deformation field with per-timestep remeshing, one rank.
 *
 * The standard front-tracking stress test: a sphere in a time-reversed vortex
 * that stretches it into a thin doubly-wound filament by t = T/2 and returns it
 * to its initial shape at t = T. The interface's area grows by an order of
 * magnitude and then shrinks back, so a front tracker MUST remesh every step, and
 * the error at t = T measures everything the remesher did along the way.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THIS EXAMPLE IS ALSO THE MEASUREMENT FOR THE QUALITY ANALYSIS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Run it with -feature <degrees> to sweep the feature-detection threshold, and
 * with -nofold / -umbrella / -nosmooth to switch individual mechanisms off. It
 * prints one line per step:
 *
 *   step  t  ntri  minQ  avgQ  folds  feat  corner  ridge  free  V/V0  A/A0
 *
 * which is what docs/remeshing-quality-analysis.md is built from. The columns
 * that matter are `free` and `minQ`: a run whose free-vertex count falls while
 * the mesh grows is a run whose smoother is progressively switching itself off,
 * and minQ follows it down within one step of the first feature edge appearing.
 *
 * Usage:
 *   leveque_remeshing_singleproc [-refine N] [-feature DEG] [-corner DEG]
 *                                [-T time] [-dt step] [-steps N]
 *                                [-nofold] [-umbrella] [-nosmooth] [-vtu]
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_remeshing.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The velocity field
 * ═══════════════════════════════════════════════════════════════════════════ */

static void leveque_velocity(const double x[3], double t, double T, double u[3])
{
  const double s = cos(MFTL_PI * t / T);

  u[0] = 2.0 * sin(MFTL_PI * x[0]) * sin(MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[1] = -sin(MFTL_PI * x[1]) * sin(MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[2] = -sin(MFTL_PI * x[2]) * sin(MFTL_PI * x[2]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * s;
}

/**
 * @brief Advects every owned vertex one step with classical RK4.
 *
 * MFTL_FOREACH_OWNED_VERTEX rather than a hand-written loop: it exploits the
 * ownership prefix so there is no per-vertex rank test, and it clears the frame
 * bit on the way out, so the geometry pass that follows cannot silently use
 * frames describing where the mesh was before the step.
 */
static void advect(MFTL_Object *obj, double t, double dt, double T)
{
  MFTL_FOREACH_OWNED_VERTEX(obj, {
    double k1[3];
    double k2[3];
    double k3[3];
    double k4[3];
    double x[3];

    leveque_velocity(pos, t, T, k1);

    for (int k = 0; k < 3; ++k) x[k] = pos[k] + 0.5 * dt * k1[k];
    leveque_velocity(x, t + 0.5 * dt, T, k2);

    for (int k = 0; k < 3; ++k) x[k] = pos[k] + 0.5 * dt * k2[k];
    leveque_velocity(x, t + 0.5 * dt, T, k3);

    for (int k = 0; k < 3; ++k) x[k] = pos[k] + dt * k3[k];
    leveque_velocity(x, t + dt, T, k4);

    for (int k = 0; k < 3; ++k) pos[k] += (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k]);
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double min_q;
  double avg_q;
  int n_inverted;
  double volume;
  double area;
} mesh_report;

/**
 * @brief Quality, enclosed volume, and the count of LOCALLY folded triangles.
 *
 * A triangle counts as folded when its own normal points away from the mean of
 * its three vertices' normals. That is a LOCAL test, and it has to be: the
 * obvious global one -- does the face normal point away from the surface's
 * centroid -- is only valid for a star-shaped surface, and by t = T/2 the sphere
 * is a thin doubly-wound filament for which "away from the centroid" means
 * nothing. Measured with the global test, this run reports 7800 "inversions" at
 * peak deformation and 0 at t = T, which is a statement about the shape rather
 * than about the mesh.
 *
 * The vertex normals are the exact area-weighted ones, which
 * MFTLRemeshObject leaves current.
 */
static mesh_report measure(MFTL_Object *obj)
{
  mesh_report r = {1.0, 0.0, 0, 0.0, 0.0};
  if (obj->tris.n == 0) return r;

  MFTLComputeVertexNormals(obj);

  for (int t = 0; t < obj->tris.n; ++t)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, t, v);

    const double *a = MFTLVertexPos(obj, v[0]);
    const double *b = MFTLVertexPos(obj, v[1]);
    const double *c = MFTLVertexPos(obj, v[2]);

    r.volume += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0]);

    double e1[3];
    double e2[3];
    double e3[3];
    for (int k = 0; k < 3; ++k)
    {
      e1[k] = b[k] - a[k];
      e2[k] = c[k] - b[k];
      e3[k] = a[k] - c[k];
    }

    const double sum_sq = e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2] + e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2] + e3[0] * e3[0] + e3[1] * e3[1] +
                          e3[2] * e3[2];

    /* The area reaches zero on a degenerate triangle, which is what makes
     * q = 4*sqrt(3)*A / (a^2+b^2+c^2) usable as a diagnostic here: it stays inside
     * [0, 1]. That was not true before the degeneracy guard was moved off the
     * magnitude and onto the reciprocal's divisor -- with the area floored at
     * 5e-7 this example measured an AVERAGE q of 1.023 on a degenerate mesh. See
     * docs/remeshing-quality-analysis.md. */
    double n[3];
    double area = 0.0;
    MFTLGetTriangleProperties(obj, t, &area, n, MFTL_NULL);
    r.area += area;

    const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
    if (q < r.min_q) r.min_q = q;
    r.avg_q += q;

    double vn[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
    {
      for (int k = 0; k < 3; ++k) vn[k] += obj->verts.normal[3 * v[i] + k];
    }
    if (n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] <= 0.0) r.n_inverted++;
  }

  r.volume /= 6.0;
  r.avg_q /= (double) obj->tris.n;
  return r;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int nproc = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  if (nproc != 1)
  {
    fprintf(stderr, "This example is the SERIAL remesher -- run it on one rank.\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* --- Options --------------------------------------------------------- */
  int refinements = 4;
  double feature_angle = 180.0;  // detection off by default; see the analysis doc
  double corner_angle = 45.0;
  double T = 3.0;
  double dt = 0.01;
  int max_steps = -1;
  bool write_vtu = MFTL_FALSE;
  bool fold_guard = MFTL_TRUE;
  bool do_smooth = MFTL_TRUE;
  MFTL_SmoothMode smooth_mode = MFTL_SMOOTH_TRIANGLE_BASED;

  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-refine") && i + 1 < argc)
      refinements = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-feature") && i + 1 < argc)
      feature_angle = atof(argv[++i]);
    else if (!strcmp(argv[i], "-corner") && i + 1 < argc)
      corner_angle = atof(argv[++i]);
    else if (!strcmp(argv[i], "-T") && i + 1 < argc)
      T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-dt") && i + 1 < argc)
      dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "-steps") && i + 1 < argc)
      max_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-vtu"))
      write_vtu = MFTL_TRUE;
    else if (!strcmp(argv[i], "-nofold"))
      fold_guard = MFTL_FALSE;
    else if (!strcmp(argv[i], "-nosmooth"))
      do_smooth = MFTL_FALSE;
    else if (!strcmp(argv[i], "-umbrella"))
      smooth_mode = MFTL_SMOOTH_VERTEX_BASED;
    else
    {
      fprintf(stderr, "unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  /* --- Setup ----------------------------------------------------------- */
  int cuts[3] = {1, 1, 1};
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  double xmin[3] = {0.0, 0.0, 0.0};
  double xmax[3] = {1.0, 1.0, 1.0};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, periodic, 1);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);

  double centre[3] = {0.35, 0.35, 0.35};
  MFTLGenerateSphere(obj, centre, 0.15, refinements);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* The target length is the mesh's own mean edge, so the remesher's job is to
   * HOLD the initial resolution while the surface stretches by an order of
   * magnitude -- not to change it. */
  const double target = MFTLMeanEdgeLength(obj);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, 3, smooth_mode, feature_angle);
  params.feature_corner_angle = corner_angle;
  params.smooth_iterations = do_smooth ? 3 : 0;
  if (!fold_guard) params.max_normal_deviation_cos = -2.0;

  const mesh_report r0 = measure(obj);

  printf("# LeVeque deformation, serial remeshing\n");
  printf("#   sphere r=0.15 at (0.35,0.35,0.35), %d refinements -> %d triangles\n", refinements, obj->tris.n);
  printf("#   target edge %.5f   T=%.2f dt=%.3f\n", target, T, dt);
  printf("#   feature angle %.1f deg%s, corner angle %.1f deg, fold guard %s, smoothing %d sweeps (%s)\n", feature_angle,
         (feature_angle >= 180.0) ? " (detection DISABLED)" : "", corner_angle, fold_guard ? "on" : "OFF", params.smooth_iterations,
         (smooth_mode == MFTL_SMOOTH_TRIANGLE_BASED) ? "area-weighted" : "umbrella");
  printf("#   V0 = %.6e   A0 = %.6e\n#\n", r0.volume, r0.area);
  printf("# %5s %7s %7s %8s %8s %6s %6s %7s %7s %7s %10s %9s\n", "step", "t", "ntri", "minQ", "avgQ", "inv", "feat", "corner", "ridge", "free", "V/V0",
         "A/A0");

  int step = 0;
  char name[256];

  if (write_vtu)
  {
    snprintf(name, sizeof(name), "leveque_%04d", step);
    MFTL_VtkOptions vtk = MFTLVtkDefaults();
    MFTLWriteVtu(ens, name, &vtk);
  }

  for (double t = 0.0; t < T - 0.5 * dt; t += dt)
  {
    advect(obj, t, dt, T);

    /* No redistribution on one rank, so the per-timestep order is the simple
     * one: move the owned vertices, rebuild the frames, then remesh. */
    MFTLObjectRefreshFrames(obj);

    MFTL_RemeshStats st;
    MFTLRemeshObject(obj, &params, &st);

    step++;
    const mesh_report r = measure(obj);

    printf("  %5d %7.3f %7d %8.5f %8.5f %6d %6d %7d %7d %7d %10.6f %9.4f\n", step, t + dt, obj->tris.n, r.min_q, r.avg_q, r.n_inverted, st.n_feature_edges,
           st.n_corner_vertices, st.n_ridge_vertices, st.n_free_vertices, r.volume / r0.volume, r.area / r0.area);
    fflush(stdout);

    if (write_vtu)
    {
      snprintf(name, sizeof(name), "leveque_%04d", step);
      MFTL_VtkOptions vtk = MFTLVtkDefaults();
    MFTLWriteVtu(ens, name, &vtk);
    }

    if (max_steps > 0 && step >= max_steps) break;
  }

  const mesh_report rf = measure(obj);
  printf("#\n# FINAL  triangles %d   minQ %.5f   avgQ %.5f   inverted %d   V/V0 %.6f\n", obj->tris.n, rf.min_q, rf.avg_q, rf.n_inverted, rf.volume / r0.volume);

  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return 0;
}
