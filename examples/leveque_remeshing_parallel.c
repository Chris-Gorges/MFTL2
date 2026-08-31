/*
 * @Author: Christian Gorges
 * @Date: 2026-08-17
 *
 * @brief LeVeque deformation with DISTRIBUTED per-timestep remeshing.
 *
 * The parallel counterpart of leveque_remeshing_singleproc.c, and the same
 * physics: a sphere in a time-reversed vortex that stretches into a thin
 * doubly-wound filament by t = T/2 and returns to its initial shape at t = T.
 * The interface's area grows by an order of magnitude and shrinks back, so the
 * mesh must be remeshed every step, and the state at t = T measures everything
 * the remesher did on the way.
 *
 * It exists to answer four questions that the unit tests deliberately cannot,
 * because each of them is about a RUN rather than about a call:
 *
 *   CORRECTNESS   does the distributed answer track the serial one over 300
 *                 remeshing passes, or does a small per-pass bias accumulate?
 *                 A single pass can be within tolerance of serial and still
 *                 drift, and drift is what a front tracker cannot afford.
 *
 *   DETERMINISM   does the same run at the same rank count produce the same
 *                 mesh twice? The design claims it must -- no rank ever touches
 *                 an entity another rank holds, so nothing depends on message
 *                 arrival order. `-digest` checks it to the BIT.
 *
 *   SCALABILITY   what does a remeshed timestep cost as P grows, and where does
 *                 the time go? Split by phase, migration, kernel and advection.
 *
 *   EFFICIENCY    how many phases and migrations does a REAL deforming mesh
 *                 need, as opposed to the one-shot refinement the tests use?
 *                 That is CLAUDE.md deferred issue 11's open question and this
 *                 is the run that answers it.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW TO COMPARE ACROSS RANK COUNTS -- AND HOW NOT TO
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Remeshing is a sequence of local decisions taken against the mesh the previous
 * one produced, so two partitions legitimately produce two different, equally
 * valid triangulations. Nothing here compares meshes element by element across
 * rank counts. What is compared is what the remesher is FOR:
 *
 *   the enclosed volume        conserved to the scheme's accuracy
 *   the surface area           the same function of time
 *   the triangle count         within a few percent
 *   min / mean triangle quality
 *   the fraction of edges outside the length band
 *
 * The BIT-level digest is only ever compared between two runs at the SAME rank
 * count, where it is a statement about determinism and nothing else.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   mpirun -n P leveque_remeshing_parallel [options]
 *
 *     -refine N     sphere subdivision level          (default 3)
 *     -T time       period                            (default 3.0)
 *     -dt step      timestep                          (default 0.02)
 *     -steps N      stop after N steps                (default: all of T)
 *     -strategy S   auto | spanning | dispersed       (default spanning)
 *     -feature DEG  feature angle, >=180 disables     (default 180)
 *     -every N      remesh every N steps              (default 1)
 *     -quiet        final summary only
 *     -digest       print the bitwise mesh digest each step
 *     -smooth N     smoothing sweeps per iteration       (default 3)
 *     -forceparallel  take the phase driver even at nproc == 1
 *     -budget N     cap the phase count per remesh       (default 32)
 *     -vtu          write leveque_par_%04d.vtu
 *
 * The default strategy is SPANNING rather than AUTO on purpose: the sphere
 * starts at r = 0.15 in a unit box, so at small P it is smaller than a rank box
 * and AUTO would begin on the dispersed path, switch to spanning as the filament
 * stretches, and switch back -- correct, and useless as a scaling measurement
 * because the two paths cost different things. Run `-strategy auto` to watch
 * that switch happen; it is printed in the `str` column.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"
#include "mftl_remeshing.h"

/* ═══════════════════════════════════════════════════════════════════════════
 * The velocity field -- identical to the serial example
 * ═══════════════════════════════════════════════════════════════════════════ */

static void leveque_velocity(const double x[3], double t, double T, double u[3])
{
  const double s = cos(MFTL_PI * t / T);

  u[0] = 2.0 * sin(MFTL_PI * x[0]) * sin(MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[1] = -sin(MFTL_PI * x[1]) * sin(MFTL_PI * x[1]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[2]) * s;
  u[2] = -sin(MFTL_PI * x[2]) * sin(MFTL_PI * x[2]) * sin(2.0 * MFTL_PI * x[0]) * sin(2.0 * MFTL_PI * x[1]) * s;
}

/**
 * @brief RK4 advection of the OWNED vertices.
 *
 * Owned only, and that is the whole parallel content of this function: a ghost
 * is another rank's vertex and its owner is advancing it with the same field to
 * the same answer. Advecting it here as well would be harmless arithmetic and a
 * lie about ownership -- and MFTLUpdateGhostPositions immediately afterwards
 * overwrites it anyway.
 */
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
 * Diagnostics
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double n_tri;
  double n_vert;
  double n_edge;
  double n_out_of_band;
  double volume;
  double area;
  double min_q;
  double avg_q;
  double n_inverted;
} report;

/** @brief Does this rank count the undirected edge at @p h, mesh-wide exactly once? */
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

/**
 * @brief The whole-mesh state, in two reductions.
 *
 * Everything is over OWNED entities only. A ghost is a copy of a triangle
 * another rank owns, so counting it here would inflate every integral by the
 * halo fraction -- by an amount that GROWS WITH THE RANK COUNT, which is exactly
 * the shape of error that would make a scaling study report the wrong thing.
 *
 * The fold test is LOCAL -- a triangle counts as folded when its own normal
 * points away from the mean of its three vertices' normals. The obvious global
 * test (does the face normal point away from the surface centroid) is only valid
 * for a star-shaped surface, and by t = T/2 this one is a thin doubly-wound
 * filament for which "away from the centroid" means nothing.
 */
static void measure(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, report *out)
{
  double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double min_q = 1.0;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    MFTL_Object *const obj = ens->objects[o];
    if (obj->tris.n == 0) continue;

    MFTLComputeVertexNormals(obj);

    s[0] += (double) obj->tris.n_local;
    s[1] += (double) obj->verts.n_local;

    for (int h = 0; h < 3 * obj->tris.n; ++h)
    {
      if (!counts_this_edge(obj, h)) continue;

      const double len = MFTLGetEdgeLength(obj, h);
      s[2] += 1.0;
      if (len > params->max_edge_length || len < params->min_edge_length) s[3] += 1.0;
    }

    for (int t = 0; t < obj->tris.n_local; ++t)
    {
      /* Seam-consistent corners from the shared geometry accessor. */
      double c[3][3];
      MFTLGetTriangleCorners(obj, t, c);

      s[4] += (c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1]) - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0]) +
               c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])) /
              6.0;

      double n[3];
      double area = 0.0;
      MFTLGetTriangleProperties(obj, t, &area, n, MFTL_NULL);
      s[5] += area;

      double sum_sq = 0.0;
      for (int j = 0; j < 3; ++j)
      {
        const int k2 = (j + 1) % 3;
        for (int k = 0; k < 3; ++k)
        {
          const double d = c[k2][k] - c[j][k];
          sum_sq += d * d;
        }
      }

      const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
      if (q < min_q) min_q = q;
      s[6] += q;

      int v[3];
      MFTLGetTriangleVertices(obj, t, v);
      double vn[3] = {0.0, 0.0, 0.0};
      for (int i = 0; i < 3; ++i)
      {
        for (int k = 0; k < 3; ++k) vn[k] += obj->verts.normal[3 * v[i] + k];
      }
      if (n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] <= 0.0) s[7] += 1.0;
    }
  }

  double g[8];
  double g_min;
  MPI_Allreduce(s, g, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&min_q, &g_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

  out->n_tri = g[0];
  out->n_vert = g[1];
  out->n_edge = g[2];
  out->n_out_of_band = g[3];
  out->volume = g[4];
  out->area = g[5];
  out->min_q = g_min;
  out->avg_q = (g[0] > 0.0) ? g[6] / g[0] : 0.0;
  out->n_inverted = g[7];
}

/**
 * @brief A bitwise digest of the whole distributed mesh, order-independent.
 *
 * XOR, not a sum, and that is the entire trick. A floating-point sum reduced
 * over ranks is not associative, so two runs that produce the IDENTICAL mesh can
 * report different sums purely because MPI chose a different reduction tree --
 * which would make a determinism test report non-determinism in the harness. XOR
 * is associative and exact, so MPI_BXOR gives the same answer for the same
 * multiset of contributions whatever order they arrive in and whichever rank
 * holds them.
 *
 * Each owned vertex contributes a mix of its GLOBAL ID and the raw bits of its
 * three coordinates. Including the id is what makes the digest sensitive to a
 * renumbering, which is a real difference in the mesh's identity even when the
 * geometry is unchanged; multiplying by an odd constant before the XOR stops two
 * vertices with mirrored coordinates from cancelling.
 */
static uint64_t mesh_digest(const MFTL_Ensemble *ens)
{
  uint64_t local = 0;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      uint64_t h = (uint64_t) (uint32_t) obj->verts.rec[v].global_id * 0x9E3779B97F4A7C15ull;
      h ^= (uint64_t) (uint32_t) o * 0xC2B2AE3D27D4EB4Full;

      for (int k = 0; k < 3; ++k)
      {
        uint64_t bits;
        const double x = obj->verts.pos[3 * v + k];
        memcpy(&bits, &x, sizeof bits);

        /* Rotate per component so that (x,y,z) and (y,z,x) are distinguishable. */
        h ^= (bits << (k * 7)) | (bits >> (64 - k * 7 - 1) >> 1);
        h *= 0xFF51AFD7ED558CCDull;
        h ^= h >> 29;
      }
      local ^= h;
    }
  }

  uint64_t global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_UINT64_T, MPI_BXOR, MPI_COMM_WORLD);
  return global;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

static void grid_for(int nproc, int cuts[3])
{
  /* The factorisation matters more than it looks: a 1D slab decomposition gives
   * the frozen band ONE axis to live on and would flatter the shift schedule,
   * because tube and corner conflicts -- the ones the odd-parity phases exist
   * for -- cannot occur. Cubic wherever the rank count allows it. */
  static const int table[][4] = {{1, 1, 1, 1}, {2, 2, 1, 1}, {3, 3, 1, 1}, {4, 2, 2, 1}, {6, 3, 2, 1},
                                 {8, 2, 2, 2}, {12, 3, 2, 2}, {16, 4, 2, 2}, {24, 4, 3, 2}, {27, 3, 3, 3}, {32, 4, 4, 2}};

  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i)
  {
    if (table[i][0] == nproc)
    {
      cuts[0] = table[i][1];
      cuts[1] = table[i][2];
      cuts[2] = table[i][3];
      return;
    }
  }
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
}

static double now(void) { return MPI_Wtime(); }

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int refinements = 3;
  double T = 3.0;
  double dt = 0.02;
  int max_steps = -1;
  int remesh_every = 1;
  double feature_angle = 180.0;
  bool quiet = MFTL_FALSE;
  bool want_digest = MFTL_FALSE;
  bool write_vtu = MFTL_FALSE;
  bool force_parallel = MFTL_FALSE;
  int phase_budget = 0;
  int smooth_iterations = 3;
  MFTL_RemeshParallelStrategy strategy = MFTL_REMESH_PARALLEL_SPANNING;

  for (int i = 1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "-refine") && i + 1 < argc)
      refinements = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-T") && i + 1 < argc)
      T = atof(argv[++i]);
    else if (!strcmp(argv[i], "-dt") && i + 1 < argc)
      dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "-steps") && i + 1 < argc)
      max_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-every") && i + 1 < argc)
      remesh_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-feature") && i + 1 < argc)
      feature_angle = atof(argv[++i]);
    else if (!strcmp(argv[i], "-quiet"))
      quiet = MFTL_TRUE;
    else if (!strcmp(argv[i], "-digest"))
      want_digest = MFTL_TRUE;
    else if (!strcmp(argv[i], "-vtu"))
      write_vtu = MFTL_TRUE;
    else if (!strcmp(argv[i], "-smooth") && i + 1 < argc)
      smooth_iterations = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-forceparallel"))
      force_parallel = MFTL_TRUE;
    else if (!strcmp(argv[i], "-budget") && i + 1 < argc)
      phase_budget = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-strategy") && i + 1 < argc)
    {
      const char *v = argv[++i];
      if (!strcmp(v, "auto"))
        strategy = MFTL_REMESH_PARALLEL_AUTO;
      else if (!strcmp(v, "spanning"))
        strategy = MFTL_REMESH_PARALLEL_SPANNING;
      else if (!strcmp(v, "dispersed"))
        strategy = MFTL_REMESH_PARALLEL_DISPERSED;
      else
      {
        if (rank == 0) fprintf(stderr, "unknown strategy: %s\n", v);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    else
    {
      if (rank == 0) fprintf(stderr, "unknown option: %s\n", argv[i]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  int cuts[3];
  grid_for(nproc, cuts);

  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  double xmin[3] = {0.0, 0.0, 0.0};
  double xmax[3] = {1.0, 1.0, 1.0};

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, periodic, 1);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  MFTL_Object *obj = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj);

  /* Every rank generates the whole sphere and the first redistribution scatters
   * it. Wasteful at scale and irrelevant here -- it is one call before the
   * timing starts, and it removes any question of the initial condition
   * differing between rank counts. */
  double centre[3] = {0.35, 0.35, 0.35};
  MFTLGenerateSphere(obj, centre, 0.15, refinements);
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* The target is the mesh's own initial mean edge, so the remesher's job is to
   * HOLD the starting resolution while the surface stretches by an order of
   * magnitude -- not to change it. Computed before the scatter, while this rank
   * still holds the whole mesh, so every rank gets the same number. */
  const double target = MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, 3, MFTL_SMOOTH_TRIANGLE_BASED, feature_angle);
  params.smooth_iterations = smooth_iterations;
  params.parallel_strategy = strategy;
  if (phase_budget > 0) params.parallel_phase_budget = phase_budget;

  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
  MFTLEnsembleUpdateGlobalCounts(ens);

  report r0;
  measure(ens, &params, &r0);

  const double h_min = MFTLDomainGetHMin(&ens->domain);
  const double delta = 3.0 * params.max_edge_length;
  const int coarsen_g = MFTLRemeshCoarseningFactor(&ens->domain, &params);

  if (rank == 0 && !quiet)
  {
    printf("# LeVeque deformation, DISTRIBUTED remeshing\n");
    printf("#   %d ranks on a %dx%dx%d grid over the unit cube   h_min = %.4f\n", nproc, cuts[0], cuts[1], cuts[2], h_min);
    printf("#   sphere r=0.15 at (0.35,0.35,0.35), %d refinements -> %.0f triangles\n", refinements, r0.n_tri);
    printf("#   target edge %.5f  (band %.5f .. %.5f)   T=%.2f dt=%.3f  remesh every %d step(s)\n", target, params.min_edge_length,
           params.max_edge_length, T, dt, remesh_every);
    printf("#   strategy %s   feature angle %.1f%s\n", (strategy == MFTL_REMESH_PARALLEL_AUTO)        ? "AUTO"
                                                       : (strategy == MFTL_REMESH_PARALLEL_SPANNING) ? "SPANNING"
                                                                                                     : "DISPERSED",
           feature_angle, (feature_angle >= 180.0) ? " (detection DISABLED)" : "");
    printf("#   band precondition: h_min=%.4f %s 4*delta=%.4f%s\n", h_min, (h_min > 4.0 * delta) ? ">" : "<=", 4.0 * delta,
           (coarsen_g > 1) ? "   *** VIOLATED ***" : "");
    if (coarsen_g > 1) printf("#   MFTLRemeshCoarseningFactor says %d ranks would have to be grouped\n", coarsen_g);
    printf("#   V0 = %.8e   A0 = %.8e\n#\n", r0.volume, r0.area);
    printf("# %5s %7s %9s %8s %8s %5s %6s %5s %5s %10s %8s %8s %9s\n", "step", "t", "ntri", "minQ", "avgQ", "inv", "oob", "ph", "mig", "V/V0", "A/A0",
           "t_rem", "str");
    fflush(stdout);
  }

  /* --- The loop ---------------------------------------------------------- */
  double wall_total = 0.0, t_advect = 0.0, t_ghost = 0.0, t_geom = 0.0, t_remesh = 0.0, t_measure = 0.0;
  long total_phases = 0, total_migrations = 0, n_remeshes = 0;
  int not_converged = 0;
  double peak_tri = r0.n_tri;

  char name[256];
  if (write_vtu)
  {
    MFTL_VtkOptions vtk = MFTLVtkDefaults();
    snprintf(name, sizeof(name), "leveque_par_%04d", 0);
    MFTLWriteVtu(ens, name, &vtk);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = now();

  int step = 0;
  for (double t = 0.0; t < T - 0.5 * dt; t += dt)
  {
    double a = now();
    advect(obj, t, dt, T);
    t_advect += now() - a;

    /* The ghost sync is NOT optional and it is not an optimisation. A triangle's
     * three vertices can belong to three ranks, so the geometry pass below --
     * whose centres decide where the first migration sends everything -- would
     * otherwise be computed partly from last step's positions, and two ranks
     * would disagree about a ghost triangle's destination. */
    a = now();
    MFTLUpdateGhostPositions(ens);
    t_ghost += now() - a;

    a = now();
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
    t_geom += now() - a;

    MFTL_RemeshStats st;
    memset(&st, 0, sizeof st);

    const bool do_remesh = ((step % remesh_every) == 0);
    if (do_remesh)
    {
      a = now();
      /* -forceparallel takes the phase driver even at nproc == 1, which is the
       * only way to separate "the phase DRIVER behaves differently from the
       * serial one" from "the PARTITION changes the answer": at one rank the
       * driver runs with an empty frozen set, so any difference it produces is
       * the driver's and not the partitioning's. */
      const int rc = force_parallel ? MFTLRemeshEnsembleParallel(ens, &params, &st) : MFTLRemeshEnsemble(ens, &params, &st);
      if (rc != MFTL_SUCCESS)
      {
        if (rank == 0) fprintf(stderr, "remesh failed at step %d\n", step);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      t_remesh += now() - a;

      total_phases += st.n_phases;
      total_migrations += st.n_migrations;
      n_remeshes++;
      if (!st.converged) not_converged++;
    }
    else
    {
      /* Without a remesh nothing has re-partitioned, so ownership still has to
       * be re-decided -- the mesh moved. */
      a = now();
      if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
      t_remesh += now() - a;
    }

    step++;

    a = now();
    report r;
    measure(ens, &params, &r);
    t_measure += now() - a;

    if (r.n_tri > peak_tri) peak_tri = r.n_tri;

    uint64_t dg = 0;
    if (want_digest) dg = mesh_digest(ens);

    if (rank == 0 && !quiet)
    {
      printf("  %5d %7.3f %9.0f %8.5f %8.5f %5.0f %6.0f %5d %5d %10.6f %8.4f %8.4f %9s", step, t + dt, r.n_tri, r.min_q, r.avg_q, r.n_inverted,
             r.n_out_of_band, st.n_phases, st.n_migrations, r.volume / r0.volume, r.area / r0.area, t_remesh,
             (st.strategy == MFTL_REMESH_PARALLEL_DISPERSED) ? "disp" : "span");
      if (want_digest) printf("  %016llx", (unsigned long long) dg);
      printf("\n");
      fflush(stdout);
    }

    if (write_vtu)
    {
      MFTL_VtkOptions vtk = MFTLVtkDefaults();
      snprintf(name, sizeof(name), "leveque_par_%04d", step);
      MFTLWriteVtu(ens, name, &vtk);
    }

    if (max_steps > 0 && step >= max_steps) break;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  wall_total = now() - t_start;

  report rf;
  measure(ens, &params, &rf);
  const uint64_t final_digest = mesh_digest(ens);

  /* --- Summary ----------------------------------------------------------- */
  double times[5] = {t_advect, t_ghost, t_geom, t_remesh, t_measure};
  double tmax[5];
  MPI_Allreduce(times, tmax, 5, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
  {
    const double work = wall_total - tmax[4];  // the diagnostics are not the simulation

    printf("#\n");
    printf("=== LeVeque, %d ranks, %d steps, %.0f -> %.0f triangles (peak %.0f) ===\n", nproc, step, r0.n_tri, rf.n_tri, peak_tri);
    printf("  CORRECTNESS\n");
    printf("    V/V0 at t=T            %.6f      (deviation %+.3f %%)\n", rf.volume / r0.volume, 100.0 * (rf.volume / r0.volume - 1.0));
    printf("    A/A0 at t=T            %.6f\n", rf.area / r0.area);
    printf("    min / avg quality      %.5f / %.5f\n", rf.min_q, rf.avg_q);
    printf("    inverted triangles     %.0f\n", rf.n_inverted);
    printf("    edges out of band      %.0f of %.0f  (%.2f %%)\n", rf.n_out_of_band, rf.n_edge, 100.0 * rf.n_out_of_band / MFTL_MAX(rf.n_edge, 1.0));
    printf("    Euler V-E+F            %.0f   (2 for a sphere)\n", rf.n_vert - rf.n_edge + rf.n_tri);
    printf("    non-converged passes   %d of %ld\n", not_converged, n_remeshes);
    printf("  DETERMINISM\n");
    printf("    final mesh digest      %016llx\n", (unsigned long long) final_digest);
    printf("    (identical between two runs at the SAME rank count iff the pass is deterministic)\n");
    printf("  EFFICIENCY\n");
    printf("    phases per remesh      %.2f\n", n_remeshes ? (double) total_phases / (double) n_remeshes : 0.0);
    printf("    migrations per remesh  %.2f\n", n_remeshes ? (double) total_migrations / (double) n_remeshes : 0.0);
    printf("  SCALABILITY  (max over ranks)\n");
    printf("    wall clock             %8.3f s   (%.2f ms/step)\n", wall_total, 1000.0 * wall_total / MFTL_MAX(step, 1));
    printf("    remesh                 %8.3f s   %5.1f %%\n", tmax[3], 100.0 * tmax[3] / MFTL_MAX(work, 1e-12));
    printf("    advection              %8.3f s   %5.1f %%\n", tmax[0], 100.0 * tmax[0] / MFTL_MAX(work, 1e-12));
    printf("    ghost sync             %8.3f s   %5.1f %%\n", tmax[1], 100.0 * tmax[1] / MFTL_MAX(work, 1e-12));
    printf("    pre-migration geometry %8.3f s   %5.1f %%\n", tmax[2], 100.0 * tmax[2] / MFTL_MAX(work, 1e-12));
    printf("    diagnostics (excluded) %8.3f s\n", tmax[4]);
    printf("    remesh per triangle    %8.1f ns   (against the mean of start and peak)\n",
           1e9 * tmax[3] / MFTL_MAX(1.0, (double) step * 0.5 * (r0.n_tri + peak_tri)));
    printf("==============================================\n");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return 0;
}
