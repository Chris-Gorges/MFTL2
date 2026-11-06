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
 *     -sizing M     uniform | curvature (default uniform):
 *                   curvature refines below the start resolution where the
 *                   radius of curvature falls under about 8 edges
 *                   (tolerance 0.02 L, floor L / 4)
 *     -placement P  midpoint | paraboloid | paraboloid_normals: where splits
 *                   and collapses put the new vertex (default midpoint);
 *                   sets both of the next two
 *     -split_placement P     the same, for splits only
 *     -collapse_placement P  the same, for collapses only
 *     -vpcollapse 0|1        move a collapse's chosen point so the collapse
 *                   keeps the enclosed volume (default 0)
 *     -fitradius R  Wendland support of the paraboloid fit, in local edge
 *                   lengths (default 3)
 *     -projection J none | phong: project smoothed vertices onto the Phong
 *                   surface of each remesh call's input (default none)
 *     -verify       exit non-zero if the enclosed volume, the winding or the
 *                   Euler characteristic moved. Registered as a CTest at np = 2
 *     -feature DEG  feature angle, >=180 disables     (default 180)
 *     -every N      remesh every N steps              (default 1)
 *     -quiet        final summary only
 *     -digest       print the bitwise mesh digest each step
 *     -smooth N     smoothing sweeps per iteration       (default 3)
 *     -iterations N outer iterations per remesh (per placement in
 *                   the phase driver)                   (default 3)
 *     -forceparallel  take the phase driver even at nproc == 1
 *     -budget N     cap the phase count per remesh       (default 32)
 *     -vtu          write leveque_par_%04d.vtu
 *     -noremesh     advect only: the baseline that separates the time
 *                   integration's error from the remesher's
 *     -snapshots P  np = 1 only: write the mesh at t = 0, T/4, T/2, 3T/4 and T
 *                   as legacy binary VTK P_q0.vtk .. P_q4.vtk (dt must divide
 *                   T/4)
 *     -csv FILE     per-step time series (step, t, counts, V/V0, quality, ms)
 *     -rotate DEG   rotate the initial sphere about its centre by DEG degrees
 *                   (axis (3,5,8)/|.|): the same exact solution on a
 *                   differently aligned mesh, for run-to-run spread
 *     -volcorrect W off | end | both: restore every body's initial volume with
 *                   MFTLEnsembleCorrectVolumes() once per step after the remesh
 *                   (end), or also right after the advection (both); default off
 *     -volsteps N   Newton steps of the volume correction (default 3)
 *     -tsur3d N     TSUR3D sweeps per step, MFTLEnsembleRemoveUndulations()
 *                   (default 0: off)
 *     -tsur3d_at W  advect | end: run them after the advection, before the
 *                   remesh (default), or after the remesh
 *     -tsur3d_angle DEG  leave edges with a kink below DEG alone (default: the
 *                   library's, 15; 0 treats every edge)
 *     -field F      leveque | rotation: the deformation field (default), or one
 *                   rigid turn per period about the sphere's centre, whose exact
 *                   surface is the initial sphere at every time
 *     -noise EPS    add a white-noise velocity of amplitude EPS per component to
 *                   every vertex every step, hashed from (vertex id, step): a
 *                   model of the sub-grid velocity fluctuations an interpolated
 *                   flow field puts on markers (default 0)
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

/** @brief 1: a rigid rotation about the sphere's centre instead of the LeVeque field (-field rotation). */
static int g_rotation_field = 0;

static void leveque_velocity(const double x[3], double t, double T, double u[3])
{
  if (g_rotation_field)
  {
    /* One full turn per period about (1,1,1)/sqrt(3) through the sphere's centre:
     * the exact surface is the initial sphere at EVERY time, and every vertex
     * moves tangentially -- whatever moves normal to the surface is error. */
    const double w = 2.0 * MFTL_PI / T / sqrt(3.0), r[3] = {x[0] - 0.35, x[1] - 0.35, x[2] - 0.35};
    u[0] = w * (r[2] - r[1]);
    u[1] = w * (r[0] - r[2]);
    u[2] = w * (r[1] - r[0]);
    return;
  }
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
/** @brief A uniform number in [-1, 1] from (id, step, component): the same on every rank. */
static double hash_noise(uint64_t id, uint64_t step, uint64_t k)
{
  uint64_t z = id * 0x9E3779B97F4A7C15ull + step * 0xD1B54A32D192ED03ull + k * 0x8CB92BA72F3D8DD7ull + 0x632BE59BD9B4E019ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return 2.0 * ((double) (z >> 11) / 9007199254740992.0) - 1.0;
}

/** @brief The sub-grid fluctuation: a displacement eps * dt * xi per component, xi white in space and time. */
static void add_noise(MFTL_Object *obj, double eps, double dt, int step)
{
  for (int v = 0; v < obj->verts.n_local; ++v)
    for (int k = 0; k < 3; ++k) obj->verts.pos[3 * v + k] += eps * dt * hash_noise((uint64_t) obj->verts.rec[v].global_id, (uint64_t) step, (uint64_t) k);
}

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
/**
 * @brief Distance of the owned vertices to the initial sphere, which is the exact
 *        surface at t = T: RMS, mean (signed, negative inside) and maximum, in
 *        units of R0 = 0.15. Vertex averages, reduced over all ranks.
 */
static void sphere_distance(MFTL_Ensemble *ens, double out[3])
{
  double acc[3] = {0.0, 0.0, 0.0}, mx = 0.0;
  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];
    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      const double *p = MFTLVertexPos(obj, v);
      const double dx = p[0] - 0.35, dy = p[1] - 0.35, dz = p[2] - 0.35;
      const double d = (sqrt(dx * dx + dy * dy + dz * dz) - 0.15) / 0.15;
      acc[0] += d * d;
      acc[1] += d;
      acc[2] += 1.0;
      if (fabs(d) > mx) mx = fabs(d);
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, acc, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &mx, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  out[0] = sqrt(acc[0] / MFTL_MAX(acc[2], 1.0));
  out[1] = acc[1] / MFTL_MAX(acc[2], 1.0);
  out[2] = mx;
}

static void measure(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, report *out)
{
  double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  double min_q = 1.0;

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    MFTL_Object *const obj = ens->local_objects[o];
    if (obj->tris.n == 0) continue;

    MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);

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

  /* The enclosed volume from the library: each body counted by its lowest
   * holder, so the SUM below adds it once. */
  {
    const int b = ens->n_local_objects;
    MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) malloc((size_t) (b > 0 ? b : 1) * sizeof(MFTL_BodyIntegrals));
    if (MFTLEnsembleBodyIntegrals(ens, bi) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
    for (int o = 0; o < b; ++o)
      if (bi[o].lowest_holder == ens->domain.rank) s[4] += bi[o].volume;
    free(bi);
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

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *const obj = ens->local_objects[o];

    for (int v = 0; v < obj->verts.n_local; ++v)
    {
      uint64_t h = (uint64_t) (uint32_t) obj->verts.rec[v].global_id * 0x9E3779B97F4A7C15ull;
      /* The BODY ID, not the list slot `o`. A rank holds a sparse subset of the
       * ensemble, so a slot names a different body on every rank and a digest
       * keyed on it would differ between two runs that produced the identical
       * mesh. With one body the two agree, which is why the recorded digests are
       * unchanged. */
      h ^= (uint64_t) (uint32_t) obj->global_id * 0xC2B2AE3D27D4EB4Full;

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

/** @brief Appends @p n big-endian copies of @p bytes-wide values (legacy VTK BINARY is big-endian). */
static void write_big_endian(FILE *f, const void *data, size_t width, size_t n)
{
  const unsigned char *p = (const unsigned char *) data;
  unsigned char buf[8];
  for (size_t i = 0; i < n; ++i)
  {
    for (size_t b = 0; b < width; ++b) buf[b] = p[i * width + (width - 1 - b)];
    fwrite(buf, 1, width, f);
  }
}

/**
 * @brief The whole mesh of a ONE-rank run as legacy binary VTK polydata.
 *
 * One rank only: the study that reads these evaluates exact errors per vertex
 * and wants the mesh whole, and at np = 1 the object IS the whole mesh.
 */
static void write_snapshot(const MFTL_Object *obj, const char *prefix, int quarter, double t)
{
  char path[512];
  snprintf(path, sizeof(path), "%s_q%d.vtk", prefix, quarter);
  FILE *f = fopen(path, "wb");
  if (!f)
  {
    fprintf(stderr, "cannot write %s\n", path);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  const int nv = obj->verts.n;
  const int nt = obj->tris.n;
  fprintf(f, "# vtk DataFile Version 3.0\nleveque t=%.17g\nBINARY\nDATASET POLYDATA\nPOINTS %d double\n", t, nv);
  write_big_endian(f, obj->verts.pos, sizeof(double), (size_t) 3 * (size_t) nv);
  fprintf(f, "\nPOLYGONS %d %d\n", nt, 4 * nt);
  for (int tri = 0; tri < nt; ++tri)
  {
    int v[3];
    MFTLGetTriangleVertices(obj, tri, v);
    const int rec[4] = {3, v[0], v[1], v[2]};
    write_big_endian(f, rec, sizeof(int), 4);
  }
  fprintf(f, "\n");
  fclose(f);
}

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

/**
 * @brief Parses a placement name; aborts on an unknown one.
 * @param v    The name.
 * @param rank This rank, to print the error once.
 * @return The placement.
 */
static MFTL_RemeshPlacement parse_placement(const char *v, int rank)
{
  if (!strcmp(v, "midpoint")) return MFTL_REMESH_PLACE_MIDPOINT;
  if (!strcmp(v, "paraboloid")) return MFTL_REMESH_PLACE_PARABOLOID;
  if (!strcmp(v, "paraboloid_normals")) return MFTL_REMESH_PLACE_PARABOLOID_NORMALS;
  if (rank == 0) fprintf(stderr, "unknown placement: %s\n", v);
  MPI_Abort(MPI_COMM_WORLD, 1);
  return MFTL_REMESH_PLACE_MIDPOINT;
}

/**
 * @brief The name of a placement, for the header.
 * @param p The placement.
 * @return Its name.
 */
static const char *placement_name(MFTL_RemeshPlacement p)
{
  return (p == MFTL_REMESH_PLACE_PARABOLOID) ? "paraboloid" : (p == MFTL_REMESH_PLACE_PARABOLOID_NORMALS) ? "paraboloid_normals" : "midpoint";
}

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
  int iterations = 3;
  MFTL_RemeshParallelStrategy strategy = MFTL_REMESH_PARALLEL_SPANNING;
  MFTL_RemeshSizing sizing = MFTL_REMESH_SIZING_UNIFORM;
  MFTL_RemeshPlacement split_placement = MFTL_REMESH_PLACE_MIDPOINT;
  MFTL_RemeshPlacement collapse_placement = MFTL_REMESH_PLACE_MIDPOINT;
  bool vp_collapse = MFTL_FALSE;
  MFTL_RemeshProjection projection = MFTL_REMESH_PROJECT_NONE;
  double fit_radius = 0.0;  // 0: the library default
  bool no_remesh = MFTL_FALSE;
  const char *snapshot_prefix = MFTL_NULL;
  const char *csv_path = MFTL_NULL;
  double rotate_deg = 0.0;
  bool verify = MFTL_FALSE;
  int volcorrect = 0;  // 0 off, 1 end of step, 2 after advection and at the end
  int vol_steps = 3;
  int tsur_sweeps = 0;
  bool tsur_at_end = MFTL_FALSE;
  double tsur_angle = -1.0;  // < 0: the library default
  double noise_eps = 0.0;

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
    else if (!strcmp(argv[i], "-verify"))
      verify = MFTL_TRUE;
    else if (!strcmp(argv[i], "-digest"))
      want_digest = MFTL_TRUE;
    else if (!strcmp(argv[i], "-vtu"))
      write_vtu = MFTL_TRUE;
    else if (!strcmp(argv[i], "-smooth") && i + 1 < argc)
      smooth_iterations = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-iterations") && i + 1 < argc)
      iterations = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-forceparallel"))
      force_parallel = MFTL_TRUE;
    else if (!strcmp(argv[i], "-budget") && i + 1 < argc)
      phase_budget = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-sizing") && i + 1 < argc)
    {
      const char *v = argv[++i];
      if (!strcmp(v, "uniform"))
        sizing = MFTL_REMESH_SIZING_UNIFORM;
      else if (!strcmp(v, "curvature"))
        sizing = MFTL_REMESH_SIZING_CURVATURE;
      else
      {
        if (rank == 0) fprintf(stderr, "unknown sizing: %s\n", v);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    else if (!strcmp(argv[i], "-placement") && i + 1 < argc)
      split_placement = collapse_placement = parse_placement(argv[++i], rank);
    else if (!strcmp(argv[i], "-split_placement") && i + 1 < argc)
      split_placement = parse_placement(argv[++i], rank);
    else if (!strcmp(argv[i], "-collapse_placement") && i + 1 < argc)
      collapse_placement = parse_placement(argv[++i], rank);
    else if (!strcmp(argv[i], "-vpcollapse") && i + 1 < argc)
      vp_collapse = atoi(argv[++i]) != 0;
    else if (!strcmp(argv[i], "-projection") && i + 1 < argc)
    {
      const char *v = argv[++i];
      if (!strcmp(v, "none"))
        projection = MFTL_REMESH_PROJECT_NONE;
      else if (!strcmp(v, "phong"))
        projection = MFTL_REMESH_PROJECT_PHONG;
      else
      {
        if (rank == 0) fprintf(stderr, "unknown projection: %s\n", v);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    else if (!strcmp(argv[i], "-fitradius") && i + 1 < argc)
      fit_radius = atof(argv[++i]);
    else if (!strcmp(argv[i], "-noremesh"))
      no_remesh = MFTL_TRUE;
    else if (!strcmp(argv[i], "-snapshots") && i + 1 < argc)
      snapshot_prefix = argv[++i];
    else if (!strcmp(argv[i], "-csv") && i + 1 < argc)
      csv_path = argv[++i];
    else if (!strcmp(argv[i], "-rotate") && i + 1 < argc)
      rotate_deg = atof(argv[++i]);
    else if (!strcmp(argv[i], "-volcorrect") && i + 1 < argc)
    {
      const char *v = argv[++i];
      volcorrect = !strcmp(v, "end") ? 1 : (!strcmp(v, "both") ? 2 : 0);
      if (strcmp(v, "off") && !volcorrect)
      {
        if (rank == 0) fprintf(stderr, "unknown -volcorrect: %s\n", v);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    else if (!strcmp(argv[i], "-volsteps") && i + 1 < argc)
      vol_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-tsur3d") && i + 1 < argc)
      tsur_sweeps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-tsur3d_at") && i + 1 < argc)
      tsur_at_end = !strcmp(argv[++i], "end");
    else if (!strcmp(argv[i], "-tsur3d_angle") && i + 1 < argc)
      tsur_angle = atof(argv[++i]);
    else if (!strcmp(argv[i], "-noise") && i + 1 < argc)
      noise_eps = atof(argv[++i]);
    else if (!strcmp(argv[i], "-field") && i + 1 < argc)
      g_rotation_field = !strcmp(argv[++i], "rotation");
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

  if (snapshot_prefix && nproc != 1)
  {
    if (rank == 0) fprintf(stderr, "-snapshots needs np = 1\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* The quarter periods as step numbers; a snapshot is only meaningful at the
   * exact time, so dt has to divide T/4. */
  int quarter_step[5] = {0, -1, -1, -1, -1};
  if (snapshot_prefix)
  {
    for (int q = 1; q <= 4; ++q)
    {
      const double s_real = 0.25 * (double) q * T / dt;
      quarter_step[q] = (int) lround(s_real);
      if (fabs(s_real - (double) quarter_step[q]) > 1e-6)
      {
        fprintf(stderr, "-snapshots: dt = %g does not divide T/4 = %g\n", dt, 0.25 * T);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
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
  if (rotate_deg != 0.0)
  {
    /* Rodrigues about a fixed, deliberately unaligned axis through the centre:
     * the vertices stay on the sphere, so the exact solution is unchanged and
     * only the mesh's alignment with the flow differs. */
    const double norm = sqrt(3.0 * 3.0 + 5.0 * 5.0 + 8.0 * 8.0);
    const double k[3] = {3.0 / norm, 5.0 / norm, 8.0 / norm};
    const double th = rotate_deg * MFTL_PI / 180.0, c = cos(th), sn = sin(th);
    for (int v = 0; v < obj->verts.n; ++v)
    {
      double *p = MFTLVertexPosMut(obj, v);
      const double x[3] = {p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]};
      const double kx = k[0] * x[0] + k[1] * x[1] + k[2] * x[2];
      const double cr[3] = {k[1] * x[2] - k[2] * x[1], k[2] * x[0] - k[0] * x[2], k[0] * x[1] - k[1] * x[0]};
      for (int d = 0; d < 3; ++d) p[d] = centre[d] + x[d] * c + cr[d] * sn + k[d] * kx * (1.0 - c);
    }
  }
  MFTLComputeTriangleGeometry(obj, obj->tris.n);

  /* The target is the mesh's own initial mean edge, so the remesher's job is to
   * HOLD the starting resolution while the surface stretches by an order of
   * magnitude -- not to change it. Computed before the scatter, while this rank
   * still holds the whole mesh, so every rank gets the same number. */
  const double target = MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, iterations, MFTL_SMOOTH_TRIANGLE_BASED, feature_angle);
  params.smooth_iterations = smooth_iterations;
  params.parallel_strategy = strategy;
  if (phase_budget > 0) params.parallel_phase_budget = phase_budget;

  /* The start resolution is the coarsest target; see the usage block. */
  if (sizing == MFTL_REMESH_SIZING_CURVATURE) MFTLRemeshUseCurvatureSizing(&params, 0.02 * target, 0.25 * target, target);
  params.split_placement = split_placement;
  params.collapse_placement = collapse_placement;
  params.collapse_preserves_volume = vp_collapse;
  params.projection = projection;
  if (fit_radius > 0.0) params.fit_radius = fit_radius;

  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLEnsembleUpdateGlobalCounts(ens);

  report r0;
  measure(ens, &params, &r0);
  /* The volume the corrections restore: the initial one, on the body itself, so
   * it reaches every rank a redistribution hands a share to. */
  if (volcorrect) MFTLEnsembleSetReferenceVolumes(ens);
  MFTL_UndulationParameters und = MFTLUndulationDefaultParameters();
  und.sweeps = tsur_sweeps;
  if (tsur_angle >= 0.0) und.min_angle = tsur_angle;
  MFTL_VolumeCorrectionParameters volp = MFTLVolumeCorrectionDefaultParameters();
  volp.iterations = vol_steps;

  const double h_min = MFTLDomainGetHMin(&ens->domain);
  const double delta = 3.0 * params.max_edge_length;
  int coarsen_group[3];
  MFTLRemeshCoarseningFactors(&ens->domain, &params, 0.0, coarsen_group);
  const int coarsen_g = coarsen_group[0] * coarsen_group[1] * coarsen_group[2];

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
    if (coarsen_g > 1)
      printf("#   MFTLRemeshCoarseningFactors says %dx%dx%d = %d ranks would have to be grouped\n", coarsen_group[0], coarsen_group[1], coarsen_group[2],
             coarsen_g);
    printf("#   sizing %s   placement split %s, collapse %s%s (fit radius %.3g)   projection %s%s\n",
           (sizing == MFTL_REMESH_SIZING_CURVATURE) ? "curvature" : "uniform", placement_name(split_placement), placement_name(collapse_placement),
           vp_collapse ? " + volume-preserving" : "", params.fit_radius, (projection == MFTL_REMESH_PROJECT_PHONG) ? "phong" : "none",
           no_remesh ? "   NO REMESHING" : "");
    printf("#   volume correction %s (%d Newton steps)   TSUR3D %d sweep(s) %s (min angle %.3g)   noise %.3g\n",
           volcorrect == 2 ? "after advection and at the end" : (volcorrect == 1 ? "at the end of each step" : "off"), vol_steps, tsur_sweeps,
           tsur_at_end ? "at the end" : "after advection", und.min_angle, noise_eps);
    printf("#   V0 = %.8e   A0 = %.8e\n#\n", r0.volume, r0.area);
    printf("# %5s %7s %9s %8s %8s %5s %6s %5s %5s %10s %8s %8s %9s\n", "step", "t", "ntri", "minQ", "avgQ", "inv", "oob", "ph", "mig", "V/V0", "A/A0",
           "t_rem", "str");
    fflush(stdout);
  }

  /* --- The loop ---------------------------------------------------------- */
  double wall_total = 0.0, t_advect = 0.0, t_ghost = 0.0, t_geom = 0.0, t_remesh = 0.0, t_measure = 0.0, t_correct = 0.0;
  double worst_vol_before = 0.0;
  long long n_tsur_moved = 0, n_tsur_band = 0;
  long total_phases = 0, total_migrations = 0, n_remeshes = 0;
  int not_converged = 0;
  double t_reference = 0.0, t_projection = 0.0, reference_bytes = 0.0;
  long n_projected = 0, n_projection_missed = 0;
  MFTLRemeshResetFitCounters();
  double peak_tri = r0.n_tri;

  if (snapshot_prefix) write_snapshot(ens->local_objects[0], snapshot_prefix, 0, 0.0);

  FILE *csv = MFTL_NULL;
  if (csv_path && rank == 0)
  {
    csv = fopen(csv_path, "w");
    if (!csv) MPI_Abort(MPI_COMM_WORLD, 1);
    fprintf(csv, "step,t,ntri,nvert,min_q,avg_q,inverted,out_of_band,V_over_V0,A_over_A0,remesh_ms,splits,collapses,flips\n");
  }

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
    /* THE LOCAL LIST, RE-READ EVERY STEP, not the pointer MFTLObjectCreate
     * returned. A redistribution destroys the object of a body a rank keeps
     * nothing of and creates one for a body arriving, so `obj` above is valid
     * only until the opening MFTLRedistribute -- and advecting through it leaves
     * this rank's real share of the sphere standing still while the other rank's
     * moves. The surface then loses volume every step with every structural
     * invariant intact, which is exactly how invisible this failure is: at
     * np = 1 nothing is ever destroyed and the same code is correct. */
    for (int o = 0; o < ens->n_local_objects; ++o) advect(ens->local_objects[o], t, dt, T);
    if (noise_eps > 0.0)
      for (int o = 0; o < ens->n_local_objects; ++o) add_noise(ens->local_objects[o], noise_eps, dt, step);
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLObjectMarkChanged(ens->local_objects[o]);
    t_advect += now() - a;

    /* The corrections that belong to the ADVECTION: before the remesh, whose
     * Phong projection then keeps them (it projects onto its own input). */
    a = now();
    if (tsur_sweeps > 0 && !tsur_at_end)
    {
      MFTL_UndulationStats ust;
      MFTLEnsembleRemoveUndulations(ens, &und, &ust);
      n_tsur_moved += ust.n_moved;
      n_tsur_band += ust.n_moved_band;
    }
    if (volcorrect == 2)
    {
      MFTL_VolumeCorrectionStats vst;
      MFTLEnsembleCorrectVolumes(ens, &volp, &vst);
      if (vst.max_error_before > worst_vol_before) worst_vol_before = vst.max_error_before;
    }
    t_correct += now() - a;

    /* The ghost sync is NOT optional and it is not an optimisation. A triangle's
     * three vertices can belong to three ranks, so the geometry pass below --
     * whose centres decide where the first migration sends everything -- would
     * otherwise be computed partly from last step's positions, and two ranks
     * would disagree about a ghost triangle's destination. */
    a = now();
    MFTLUpdateGhostPositions(ens);
    t_ghost += now() - a;

    a = now();
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    t_geom += now() - a;

    MFTL_RemeshStats st;
    memset(&st, 0, sizeof st);

    const bool do_remesh = !no_remesh && ((step % remesh_every) == 0);
    const double t_remesh_before = t_remesh;
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
      t_reference += st.t_reference;
      t_projection += st.t_projection;
      if (st.reference_bytes > reference_bytes) reference_bytes = st.reference_bytes;
      n_projected += st.n_projected;
      n_projection_missed += st.n_projection_missed;
      n_remeshes++;
      if (!st.converged) not_converged++;
    }
    else
    {
      /* Without a remesh nothing has re-partitioned, so ownership still has to
       * be re-decided -- the mesh moved. */
      a = now();
      if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
      for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
      t_remesh += now() - a;
    }

    /* The corrections at the end of the step, on the canonical partition. */
    a = now();
    if (tsur_sweeps > 0 && tsur_at_end)
    {
      MFTL_UndulationStats ust;
      MFTLEnsembleRemoveUndulations(ens, &und, &ust);
      n_tsur_moved += ust.n_moved;
      n_tsur_band += ust.n_moved_band;
    }
    if (volcorrect >= 1)
    {
      MFTL_VolumeCorrectionStats vst;
      MFTLEnsembleCorrectVolumes(ens, &volp, &vst);
      if (vst.max_error_before > worst_vol_before) worst_vol_before = vst.max_error_before;
    }
    if ((tsur_sweeps > 0 && tsur_at_end) || volcorrect >= 1)
      for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
    t_correct += now() - a;

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

    if (csv)
    {
      fprintf(csv, "%d,%.10g,%.0f,%.0f,%.8g,%.8g,%.0f,%.0f,%.12g,%.12g,%.6g,%d,%d,%d\n", step, t + dt, r.n_tri, r.n_vert, r.min_q, r.avg_q, r.n_inverted,
              r.n_out_of_band, r.volume / r0.volume, r.area / r0.area, 1000.0 * (t_remesh - t_remesh_before), st.n_split, st.n_collapse, st.n_flip);
      fflush(csv);
    }

    if (snapshot_prefix)
    {
      for (int q = 1; q <= 4; ++q)
        if (step == quarter_step[q]) write_snapshot(ens->local_objects[0], snapshot_prefix, q, t + dt);
    }

    if (max_steps > 0 && step >= max_steps) break;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  wall_total = now() - t_start;
  if (csv) fclose(csv);

  report rf;
  measure(ens, &params, &rf);
  double sph[3];
  sphere_distance(ens, sph);
  const uint64_t final_digest = mesh_digest(ens);

  /* --- Summary ----------------------------------------------------------- */
  double times[6] = {t_advect, t_ghost, t_geom, t_remesh, t_measure, t_correct};
  double tmax[6];
  MPI_Allreduce(times, tmax, 6, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  if (rank == 0)
  {
    const double work = wall_total - tmax[4];  // the diagnostics are not the simulation

    printf("#\n");
    printf("=== LeVeque, %d ranks, %d steps, %.0f -> %.0f triangles (peak %.0f) ===\n", nproc, step, r0.n_tri, rf.n_tri, peak_tri);
    printf("  CORRECTNESS\n");
    printf("    V/V0 at t=T            %.6f      (deviation %+.3f %%)\n", rf.volume / r0.volume, 100.0 * (rf.volume / r0.volume - 1.0));
    printf("    A/A0 at t=T            %.6f\n", rf.area / r0.area);
    if (max_steps <= 0)  // the whole period: the exact surface is the initial sphere again
      printf("    off the exact sphere   rms %.4f %%, mean %+.4f %%, max %.4f %% of R0   (rotation-invariant)\n", 100.0 * sph[0], 100.0 * sph[1],
             100.0 * sph[2]);
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
    if (volcorrect || tsur_sweeps > 0)
      printf("    corrections            %8.3f s   %5.1f %%   (TSUR3D %lld moves, %lld in colour rounds; largest volume error corrected %.2e)\n", tmax[5],
             100.0 * tmax[5] / MFTL_MAX(work, 1e-12), n_tsur_moved, n_tsur_band, worst_vol_before);
    printf("    diagnostics (excluded) %8.3f s\n", tmax[4]);
    printf("    remesh per triangle    %8.1f ns   (against the mean of start and peak)\n",
           1e9 * tmax[3] / MFTL_MAX(1.0, (double) step * 0.5 * (r0.n_tri + peak_tri)));
    if (projection != MFTL_REMESH_PROJECT_NONE)
    {
      printf("  PROJECTION  (rank 0)\n");
      printf("    reference copies       %8.3f s   %5.1f %% of remesh   (largest %.0f kB)\n", t_reference, 100.0 * t_reference / MFTL_MAX(tmax[3], 1e-12),
             reference_bytes / 1024.0);
      printf("    projections            %8.3f s   %5.1f %% of remesh   (%ld moves, %.1f ns each, %ld missed)\n", t_projection,
             100.0 * t_projection / MFTL_MAX(tmax[3], 1e-12), n_projected, 1e9 * t_projection / MFTL_MAX(1.0, (double) (n_projected + n_projection_missed)),
             n_projection_missed);
    }
    if (split_placement != MFTL_REMESH_PLACE_MIDPOINT || collapse_placement != MFTL_REMESH_PLACE_MIDPOINT)
    {
      MFTL_RemeshFitCounters fc;
      MFTLRemeshGetFitCounters(&fc);
      printf("  PLACEMENT  (rank 0)\n");
      printf("    fits                   %8.3f s   %5.1f %% of remesh   (%ld fits, %.1f ns each, %ld rank-deficient, %ld fell back to the midpoint)\n", fc.t_fit,
             100.0 * fc.t_fit / MFTL_MAX(tmax[3], 1e-12), fc.n_fits, 1e9 * fc.t_fit / MFTL_MAX(1.0, (double) fc.n_fits), fc.n_rank_deficient, fc.n_fallbacks);
      printf("    fits with ghost samples %ld\n", fc.n_with_ghosts);
    }
    printf("==============================================\n");
  }

  /* ── --verify: the example guards itself ─────────────────────────────────
   *
   * A benchmark whose whole purpose is to report timings has to be right before
   * its numbers mean anything, and this one has a failure mode that no library
   * test can see: the CALLER holding an MFTL_Object* across a redistribution.
   * That pointer is invalidated -- a rank keeping nothing of a body destroys its
   * object and one receiving a body creates a new one -- so advecting through it
   * leaves part of the surface standing still. Every structural invariant holds,
   * the mesh stays a valid closed sphere with good triangles, and the ENCLOSED
   * VOLUME is the only thing that moves.
   *
   * It is also invisible at np = 1, where nothing is ever destroyed. So the
   * check is registered at np = 2, and the threshold is loose on purpose: over
   * twelve steps the correct answer loses well under a percent, and the bug this
   * exists for lost 93 %.
   */
  int failed = 0;
  if (verify)
  {
    const double v_ratio = rf.volume / r0.volume;
    const double a_ratio = rf.area / r0.area;

    if (!(v_ratio > 0.95 && v_ratio < 1.05)) failed = 1;
    if (rf.n_inverted != 0.0) failed = 1;
    if (rf.n_vert - rf.n_edge + rf.n_tri != 2.0) failed = 1;

    if (rank == 0)
    {
      printf("VERIFY: V/V0 %.6f  A/A0 %.6f  inverted %.0f  Euler %.0f  -> %s\n", v_ratio, a_ratio, rf.n_inverted, rf.n_vert - rf.n_edge + rf.n_tri,
             failed ? "FAILED" : "ok");
    }
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
