/**
 * @file settings_driven_run.c
 * @author Christian Gorges
 * @date 2026-09-29
 *
 * @brief The shape of a solver's MFTL integration, configured by a settings
 *        file: every library option comes from the file, the flow's own
 *        parameters stay the program's.
 *
 *     mpirun -np 4 settings_driven_run examples/settings/settings_driven_run.toml
 *     mpirun -np 4 settings_driven_run run.toml remesh.every=2 output.every=10 --steps 50
 *     mpirun -np 2 settings_driven_run run.toml --steps 12 --verify
 *
 * Arguments: the settings file (optional; the defaults otherwise), any number
 * of `section.key=value` overrides, and the flow's own `--T`, `--dt`, `--steps`,
 * `--report`, `--quiet` and `--verify`. The flow is the LeVeque deformation of
 * examples/leveque_remeshing_parallel.c, which stretches a sphere into a thin
 * spiral sheet and brings it back at t = T.
 *
 * ## THE PATTERN
 *
 *     MFTLSettingsDefaults -> MFTLSettingsLoad -> MFTLSettingsLoadArgs
 *     MFTLSettingsCreateEnsemble -> generate the bodies -> MFTLSettingsResolve
 *     per step: move owned vertices -> MFTLUpdateGhostPositions -> geometry
 *               -> remesh (or redistribute) -> topology -> contact -> output,
 *               each when MFTLSettingsDue() says so
 *
 * Each library call takes its block straight out of the settings --
 * `&cfg.remesh.params`, `&cfg.topology.options`, `&cfg.contact.params`,
 * `&cfg.output.vtk` -- so nothing here restates a default, and a new option
 * reaches this program without a line of it changing.
 *
 * `--verify` is the check `test_settings_driven_run_conserves` runs: over a few
 * steps the enclosed volume must stay within 5 %, every body must stay a closed
 * sphere (Euler characteristic 2) and no triangle may be inverted. It is the
 * same kind of physical check as `test_leveque_parallel_conserves`, because
 * the failure it guards against -- a caller holding an MFTL_Object* across a
 * redistribution -- leaves every structural invariant intact.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The flow -- the program's own business, not the library's
 * ═══════════════════════════════════════════════════════════════════════════ */

static void leveque_velocity(const double x[3], double t, double T, double u[3])
{
  const double s = cos(MFTL_PI * t / T);
  u[0] = 2.0 * sin(MFTL_PI * x[0]) * sin(MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[1] = -sin(MFTL_PI * x[1]) * sin(MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[2] = -sin(MFTL_PI * x[2]) * sin(MFTL_PI * x[2]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * s;
}

/** RK4 on the OWNED vertices; the ghost sync that follows brings the copies along. */
static void advect(MFTL_Object *obj, double t, double dt, double T)
{
  MFTL_FOREACH_OWNED_VERTEX(obj, {
    double k1[3], k2[3], k3[3], k4[3], x[3];
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
 * Diagnostics, over OWNED entities only
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double n_tri, n_vert, n_edge, volume, area, n_inverted, n_bodies;
} report;

/** Does this rank count the undirected edge at @p h -- mesh-wide exactly once? */
static int counts_this_edge(const MFTL_Object *obj, int h)
{
  const int t = MFTLHeTriIdx(h);
  if (t >= obj->tris.n_local) return 0;
  const int tw = obj->tris.he[h].twin_idx;
  if (tw == MFTL_ERROR) return 1;
  const int to = MFTLHeTriIdx(tw);
  if (to < obj->tris.n_local) return h < tw;
  return obj->tris.rec[t].global_id < obj->tris.rec[to].global_id;
}

static void measure(MFTL_Ensemble *ens, report *out)
{
  double s[6] = {0, 0, 0, 0, 0, 0};
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);
    s[0] += obj->tris.n_local;
    s[1] += obj->verts.n_local;
    for (int h = 0; h < 3 * obj->tris.n; ++h) s[2] += counts_this_edge(obj, h);

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      double c[3][3], n[3], area = 0.0;
      MFTLGetTriangleCorners(obj, t, c);
      s[3] += (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
               c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])) /
              6.0;
      MFTLGetTriangleProperties(obj, t, &area, n, MFTL_NULL);
      s[4] += area;

      /* Folded: the face normal points away from the mean of its vertex normals. */
      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      double vn[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k) vn[k] += obj->verts.normal[3 * v[i] + k];
      if (n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] <= 0.0) s[5] += 1.0;
    }
  }
  double g[6];
  MPI_Allreduce(s, g, 6, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  out->n_tri = g[0];
  out->n_vert = g[1];
  out->n_edge = g[2];
  out->volume = g[3];
  out->area = g[4];
  out->n_inverted = g[5];
  out->n_bodies = ens->n_bodies_global;
}

static void count_contact(const MFTL_BVH *a, int pa, const MFTL_BVH *b, int pb, void *user_data)
{
  (void) a, (void) pa, (void) b, (void) pb;
  ++*(long *) user_data;
}

/** The step a solver would do for geometry: every local body, ghosts included (routing reads the centres). */
static void geometry(MFTL_Ensemble *ens)
{
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();
  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  /* ── The program's own arguments; `section.key=value` is left to MFTL ── */
  const char *settings_path = NULL;
  double T = 3.0, dt = 0.02;
  int max_steps = -1, report_every = 10;
  bool quiet = false, verify = false;
  for (int i = 1; i < argc; ++i)
  {
    if (MFTLSettingsIsOverride(argv[i])) continue;
    if (!strcmp(argv[i], "--T") && i + 1 < argc)
      T = atof(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      max_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--report") && i + 1 < argc)
      report_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--quiet"))
      quiet = true;
    else if (!strcmp(argv[i], "--verify"))
      verify = true;
    else if (argv[i][0] != '-' && !settings_path)
      settings_path = argv[i];
    else
    {
      if (rank == 0) fprintf(stderr, "unknown argument: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
  }

  /* ── 1. Settings: defaults, then the file, then the command line ──────── */
  MFTL_Settings cfg;
  MFTLSettingsDefaults(&cfg);
  if (settings_path && MFTLSettingsLoad(&cfg, settings_path, MPI_COMM_WORLD) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  if (MFTLSettingsLoadArgs(&cfg, argc, argv, MPI_COMM_WORLD) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);

  /* ── 2. The ensemble [domain], under the [redistribute] policy ─────────── */
  MFTL_Ensemble *ens = MFTLSettingsCreateEnsemble(&cfg, MPI_COMM_WORLD);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  /* ── 3. The bodies: one [sphere]. The pointer is dead after the scatter. ─ */
  MFTL_Object *sphere = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, sphere);
  if (MFTLGenerateSphere(sphere, cfg.sphere.center, cfg.sphere.radius, cfg.sphere.refinements) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  MFTLComputeTriangleGeometry(sphere, sphere->tris.n);
  sphere = NULL;

  /* ── 4. What depends on the mesh: the target length and all it implies ── */
  if (MFTLSettingsResolve(&cfg, ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  if (rank == 0 && !quiet)
  {
    printf("# settings_driven_run: %d rank(s), grid %d x %d x %d, LeVeque T = %g, dt = %g\n", nproc, ens->domain.cuts[0], ens->domain.cuts[1],
           ens->domain.cuts[2], T, dt);
    printf("# the configuration of this run -- save it and pass it back to reproduce it:\n");
    MFTLSettingsWrite(&cfg, stdout, 0);
    printf("\n");
  }

  if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  geometry(ens);
  /* What the volume correction restores: the volume each body has now. It
   * travels with the body and a topology change hands it on. */
  if (cfg.volume.every > 0 && MFTLEnsembleSetReferenceVolumes(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);

  report r0;
  measure(ens, &r0);
  if (rank == 0 && !quiet)
    printf("# %5s %8s %9s %10s %8s %5s %6s %8s %8s\n", "step", "t", "ntri", "V/V0", "A/A0", "inv", "bodies", "topo_ev", "contacts");

  /* ── 5. The loop ─────────────────────────────────────────────────────── */
  const int n_steps = (int) llround(T / dt);
  int step = 0;
  long topo_events = 0, contacts = 0;
  for (; step < n_steps && (max_steps < 0 || step < max_steps); ++step)
  {
    const double t = step * dt;

    /* Re-read the list every step: a redistribution may destroy or create objects. */
    for (int o = 0; o < ens->n_local_objects; ++o) advect(ens->local_objects[o], t, dt, T);
    MFTLUpdateGhostPositions(ens);
    /* Undulations come from the advection: removed before the remesh, whose
     * Phong projection (onto its own input) then keeps the smoothing. */
    if (MFTLSettingsDue(cfg.undulation.every, step) && MFTLEnsembleRemoveUndulations(ens, &cfg.undulation.params, MFTL_NULL) != MFTL_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 1);
    geometry(ens);

    if (MFTLSettingsDue(cfg.remesh.every, step))
    {
      MFTL_RemeshStats st;
      if (MFTLRemeshEnsemble(ens, &cfg.remesh.params, &st) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    else if (MFTLRedistribute(ens) != MFTL_SUCCESS)
      MPI_Abort(MPI_COMM_WORLD, 1);
    geometry(ens);

    if (MFTLSettingsDue(cfg.topology.every, step))
    {
      MFTL_TopologyStats ts;
      if (MFTLEnsembleApplyTopologyChanges(ens, &cfg.topology.options, &ts) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      topo_events += ts.n_events;
      geometry(ens);
    }

    /* Last of the mesh operations, so it restores what the advection, the
     * remesh and any topology change moved. */
    if (MFTLSettingsDue(cfg.volume.every, step))
    {
      if (MFTLEnsembleCorrectVolumes(ens, &cfg.volume.params, MFTL_NULL) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      geometry(ens);
    }

    if (MFTLSettingsDue(cfg.contact.every, step))
    {
      long mine = 0, all = 0;
      if (MFTLEnsembleIntersectObjects(ens, &cfg.contact.params, count_contact, &mine) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      MPI_Allreduce(&mine, &all, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
      contacts += all;
    }

    if (MFTLSettingsDue(cfg.output.every, step))
    {
      char name[MFTL_SETTINGS_STRING_MAX + 16];
      snprintf(name, sizeof name, "%s_%06d", cfg.output.prefix, step);
      if (MFTLWriteVtu(ens, name, &cfg.output.vtk) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (!quiet && report_every > 0 && (step + 1) % report_every == 0)
    {
      report r;
      measure(ens, &r);
      if (rank == 0)
        printf("  %5d %8.3f %9.0f %10.6f %8.4f %5.0f %6.0f %8ld %8ld\n", step + 1, t + dt, r.n_tri, r.volume / r0.volume, r.area / r0.area, r.n_inverted,
               r.n_bodies, topo_events, contacts);
    }
  }

  report rf;
  measure(ens, &rf);
  const double euler = rf.n_vert - rf.n_edge + rf.n_tri;
  if (rank == 0)
    printf("# %d steps: V/V0 %.6f, A/A0 %.4f, %.0f triangles, %.0f inverted, V-E+F %.0f\n", step, rf.volume / r0.volume, rf.area / r0.area, rf.n_tri,
           rf.n_inverted, euler);

  int failed = 0;
  if (verify)
  {
    const double v = rf.volume / r0.volume;
    failed = !(v > 0.95 && v < 1.05) || rf.n_inverted != 0.0 || euler != 2.0 * rf.n_bodies;
    if (rank == 0) printf("VERIFY: %s\n", failed ? "FAILED" : "ok");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
