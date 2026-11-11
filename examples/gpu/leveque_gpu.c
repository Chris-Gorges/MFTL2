/*
 * @Author: Christian Gorges
 *
 * @brief The parallel LeVeque remeshing test with the per-vertex and
 *        per-triangle work on the GPU -- one driver, three GPU programming
 *        models behind lv_backend.h: CUDA, OpenACC and OpenMP offload.
 *
 * The physics is that of leveque_remeshing_parallel.c: a sphere (r = 0.15 at
 * (0.35, 0.35, 0.35)) in the time-reversed LeVeque vortex, advected with the
 * ANALYTICAL velocity by RK4 and remeshed every step. No flow solver and no
 * coupler: the point is the front tracker's own per-step work.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT RUNS WHERE, ONE STEP
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   device   lv_advect                  RK4 of the owned vertices
 *   D -> H   lv_get_owned_positions     the host needs them for the ghost sync
 *   host     MFTLUpdateGhostPositions   MPI
 *   H -> D   lv_put_ghost_positions
 *   device   lv_triangle_geometry       area, normal, centroid of every triangle
 *   D -> H   lv_get_triangle_geometry   redistribution routes by the centroids
 *   host     MFTLRemeshEnsemble         redistribution + remeshing, pointer-heavy
 *   H -> D   lv_attach                  the remeshed mesh
 *   device   lv_measure                 vertex normals + the diagnostics, 7 numbers back
 *
 * MFTL itself is untouched: the backends read and write the library's pools
 * through lv_mesh, and the driver sets #MFTL_DERIVED_GEOMETRY after the download,
 * which is what MFTLComputeTriangleGeometry() would have done.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW IT CHECKS ITSELF
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   -verify        every step, every device result against the library on the
 *                  SAME input: positions, triangle geometry, the 7 measure sums.
 *                  The device values are kept, so the run is not perturbed.
 *                  At the end: no inverted triangle, Euler characteristic 2.
 *   -csv / -compare  the whole trajectory against a `-cpu` run at the same rank
 *                  count: the same mesh after every remesh, and every
 *                  diagnostic equal to round-off.
 *
 * Round-off agreement over a whole run needs a mesh that resolves the flow. At
 * refine 3 the sphere's filament is under-resolved -- the CPU run itself loses
 * a quarter of its volume -- and the remesher amplifies a last-bit difference
 * until a decision goes the other way, as it does on the host alone for a
 * last-bit change of the time. Hence the default of 4 (README.md).
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * USAGE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   mpirun -n P leveque_gpu_<backend> [options]
 *
 *     -refine N      sphere subdivision level            (default 4)
 *     -T time        period                              (default 3.0)
 *     -dt step       timestep                            (default 0.02)
 *     -steps N       stop after N steps                  (default: all of T)
 *     -cpu           the library on the host, no device: the reference
 *     -verify        check every device result, and the mesh at the end
 *     -csv FILE      per-step diagnostics
 *     -compare FILE  require the trajectory of FILE (a -csv of a -cpu run at
 *                    the same P and options) to round-off
 *     -quiet         summary only
 *
 * Exit status 77 when the backend finds no device (CTest's skip code).
 */

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "lv_backend.h"
#include "lv_kernels.h"
#include "mftl_core.h"

/* lv_mesh reads the triangle records as int[3] with the global id first. */
_Static_assert(sizeof(MFTL_Triangle) == 3 * sizeof(int) && offsetof(MFTL_Triangle, global_id) == 0, "MFTL_Triangle layout changed");

/** @brief The flux-volume reference plane x = 0.5; any value works for a closed body. */
#define LV_VOL_REF 0.5

typedef struct
{
  double n_tri, n_vert, n_edge, n_out_of_band, volume, area, min_q, avg_q, n_inverted;
} report;

enum
{
  T_ADVECT,
  T_GEOM,
  T_MEASURE,
  T_D2H,
  T_H2D,
  T_GHOST,
  T_REMESH,
  T_N
};
static const char *const timer_name[T_N] = {"advect", "geometry", "measure", "D->H", "H->D", "ghost sync", "remesh"};

/* ═══════════════════════════════════════════════════════════════════════════
 * The mesh this rank holds
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The one object of this rank, or NULL. Re-read after every remesh: a
 *        redistribution destroys and creates objects.
 */
static MFTL_Object *the_object(const MFTL_Ensemble *ens)
{
  if (ens->n_local_objects > 1)
  {
    fprintf(stderr, "leveque_gpu: one body per rank expected, this rank holds %d\n", ens->n_local_objects);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  if (ens->n_local_objects == 0 || ens->local_objects[0]->tris.n == 0) return NULL;
  return ens->local_objects[0];
}

static lv_mesh view_of(MFTL_Object *obj)
{
  if (!MFTLObjectHasTopology(obj)) MFTLObjectBuildDerived(obj);

  lv_mesh m;
  m.pos = obj->verts.pos;
  m.vnormal = obj->verts.normal;
  m.tri_verts = obj->tris.verts;
  m.tri_nbr = obj->tris.neighbors;
  m.tri_rec = &obj->tris.rec[0].global_id;
  m.area = obj->tris.area;
  m.normal = obj->tris.normal;
  m.center = obj->tris.center;
  m.nv = obj->verts.n;
  m.nv_owned = obj->verts.n_local;
  m.nt = obj->tris.n;
  m.nt_owned = obj->tris.n_local;
  m.nv_capacity = obj->verts.capacity;
  m.nt_capacity = obj->tris.capacity;
  return m;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * The host reference: the library, as in leveque_remeshing_parallel.c
 * ═══════════════════════════════════════════════════════════════════════════ */

static void cpu_advect(MFTL_Object *obj, double t, double dt, double T)
{
  for (int v = 0; v < obj->verts.n_local; ++v) lv_rk4(&obj->verts.pos[3 * v], t, dt, T);
}

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

/** @brief The body of measure() in leveque_remeshing_parallel.c, rank-local, through the library. */
static void cpu_measure_local(MFTL_Object *obj, const MFTL_RemeshParameters *params, lv_measure_out *o)
{
  memset(o, 0, sizeof *o);
  o->min_q = 1.0;

  MFTLComputeVertexNormals(obj, MFTL_NORMAL_WEIGHT_AREA);

  for (int h = 0; h < 3 * obj->tris.n; ++h)
  {
    if (!counts_this_edge(obj, h)) continue;
    const double len = MFTLGetEdgeLength(obj, h);
    o->n_edge += 1.0;
    if (len > params->max_edge_length || len < params->min_edge_length) o->n_out_of_band += 1.0;
  }

  for (int t = 0; t < obj->tris.n_local; ++t)
  {
    double c[3][3], n[3], area = 0.0;
    MFTLGetTriangleCorners(obj, t, c);
    MFTLGetTriangleProperties(obj, t, &area, n, MFTL_NULL);
    o->area += area;

    double sum_sq = 0.0;
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
      {
        const double d = c[(j + 1) % 3][k] - c[j][k];
        sum_sq += d * d;
      }
    const double q = (sum_sq > 0.0) ? MFTL_FOUR_SQRT3 * area / sum_sq : 0.0;
    if (q < o->min_q) o->min_q = q;
    o->q_sum += q;

    int v[3];
    MFTLGetTriangleVertices(obj, t, v);
    double vn[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 3; ++k) vn[k] += obj->verts.normal[3 * v[i] + k];
    if (n[0] * vn[0] + n[1] * vn[1] + n[2] * vn[2] <= 0.0) o->n_inverted += 1.0;
  }

  const double centre[3] = {0.5, 0.5, 0.5};
  double area_unused = 0.0;
  MFTLBodyFluxVolume(obj, 0, LV_VOL_REF, centre, &o->volume, &area_unused);
}

/** @brief Rank-local sums -> the global report. */
static void reduce_report(MFTL_Object *obj, const lv_measure_out *o, double volume, report *out)
{
  double s[8] = {0, 0, 0, 0, 0, 0, 0, 0}, min_q = 1.0, g[8], g_min;
  if (obj)
  {
    s[0] = (double) obj->tris.n_local;
    s[1] = (double) obj->verts.n_local;
    s[2] = o->n_edge;
    s[3] = o->n_out_of_band;
    s[5] = o->area;
    s[6] = o->q_sum;
    s[7] = o->n_inverted;
    min_q = o->min_q;
  }
  s[4] = volume;
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

/** @brief measure() of leveque_remeshing_parallel.c: the volume from the library's body integrals. */
static void cpu_measure(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, report *out)
{
  MFTL_Object *obj = the_object(ens);
  lv_measure_out o;
  if (obj) cpu_measure_local(obj, params, &o);

  double volume = 0.0;
  const int b = ens->n_local_objects;
  MFTL_BodyIntegrals *bi = (MFTL_BodyIntegrals *) malloc((size_t) (b > 0 ? b : 1) * sizeof(MFTL_BodyIntegrals));
  if (MFTLEnsembleBodyIntegrals(ens, bi) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int i = 0; i < b; ++i)
    if (bi[i].lowest_holder == ens->domain.rank) volume += bi[i].volume;
  free(bi);

  reduce_report(obj, &o, volume, out);
}

static void gpu_measure(MFTL_Ensemble *ens, const MFTL_RemeshParameters *params, report *out, lv_measure_out *local)
{
  MFTL_Object *obj = the_object(ens);
  if (obj) lv_measure(params->min_edge_length, params->max_edge_length, LV_VOL_REF, local);
  reduce_report(obj, local, obj ? local->volume : 0.0, out);
}

/**
 * @brief A hash of the connectivity, independent of partition and storage
 *        order: two runs with the same key hold the same mesh, triangle for
 *        triangle. Edge flips and split/collapse pairs leave every count alone;
 *        they change this.
 */
static uint64_t mesh_key(const MFTL_Ensemble *ens)
{
  const MFTL_Object *obj = the_object(ens);
  uint64_t local = 0, key = 0;
  for (int t = 0; obj && t < obj->tris.n_local; ++t)
  {
    uint64_t g[3];
    for (int k = 0; k < 3; ++k) g[k] = (uint32_t) obj->verts.rec[obj->tris.verts[3 * t + k]].global_id;
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2 - i; ++j)
        if (g[j] > g[j + 1])
        {
          const uint64_t s = g[j];
          g[j] = g[j + 1];
          g[j + 1] = s;
        }

    uint64_t z = ((g[0] << 32 | g[1]) * 0x9E3779B97F4A7C15ull) ^ (g[2] * 0xD1B54A32D192ED03ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    local += z ^ (z >> 31);
  }
  MPI_Allreduce(&local, &key, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
  return key;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * -verify: the device against the library, on the same input
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  double pos;  ///< max |x_dev - x_host| over owned vertices
  double area;  ///< max |A_dev - A_host| / A_host
  double normal;  ///< max |n_dev - n_host|
  double center;  ///< max |c_dev - c_host|
  double sums;  ///< max relative difference of area, q_sum, volume; abs of min_q
  double counts;  ///< max |difference| of n_edge, n_out_of_band, n_inverted
} errors;

/* Largest values seen over refine 3-5 (150 steps; 40 at refine 5), np 1-4, every
 * backend: 1.1e-16, 3.2e-14, 2.1e-14, 1.1e-16, 2.2e-14, 0 -- each tolerance is 30
 * to 900 times that. FMA contraction and the device's own sin/cos make them
 * non-zero, nothing else may. Refine 2 comes within 2x for area and normals: its
 * filament degenerates into slivers (the library inverts triangles there too,
 * README.md), whose normal is ill-conditioned. */
static const errors verify_tol = {1e-13, 1e-12, 1e-12, 1e-13, 1e-11, 0.0};

static double *g_scratch = NULL;
static size_t g_scratch_n = 0;

static double *scratch(size_t n)
{
  if (n > g_scratch_n)
  {
    free(g_scratch);
    g_scratch_n = n + n / 2;
    g_scratch = (double *) malloc(g_scratch_n * sizeof(double));
    if (!g_scratch) MPI_Abort(MPI_COMM_WORLD, 1);
  }
  return g_scratch;
}

static double max_abs_diff(const double *a, const double *b, size_t n)
{
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = fmax(m, fabs(a[i] - b[i]));
  return m;
}

static double rel_diff(double a, double b) { return fabs(a - b) / fmax(fabs(b), 1e-300); }

/** @brief Device geometry (on the host now) against MFTLComputeTriangleGeometry(); then put the device values back. */
static void verify_geometry(MFTL_Object *obj, errors *e)
{
  const size_t nt = (size_t) obj->tris.n;
  double *dev = scratch(7 * nt);
  memcpy(dev, obj->tris.area, nt * sizeof(double));
  memcpy(dev + nt, obj->tris.normal, 3 * nt * sizeof(double));
  memcpy(dev + 4 * nt, obj->tris.center, 3 * nt * sizeof(double));

  MFTLComputeTriangleGeometry(obj, obj->tris.n);
  for (size_t t = 0; t < nt; ++t) e->area = fmax(e->area, rel_diff(dev[t], obj->tris.area[t]));
  e->normal = fmax(e->normal, max_abs_diff(dev + nt, obj->tris.normal, 3 * nt));
  e->center = fmax(e->center, max_abs_diff(dev + 4 * nt, obj->tris.center, 3 * nt));

  memcpy(obj->tris.area, dev, nt * sizeof(double));
  memcpy(obj->tris.normal, dev + nt, 3 * nt * sizeof(double));
  memcpy(obj->tris.center, dev + 4 * nt, 3 * nt * sizeof(double));
}

static void verify_measure(MFTL_Object *obj, const MFTL_RemeshParameters *params, const lv_measure_out *dev, errors *e)
{
  lv_measure_out h;
  cpu_measure_local(obj, params, &h);

  e->sums = fmax(e->sums, rel_diff(dev->area, h.area));
  e->sums = fmax(e->sums, rel_diff(dev->q_sum, h.q_sum));
  e->sums = fmax(e->sums, fabs(dev->volume - h.volume) / fmax(fabs(h.volume), h.area * 1e-3));
  e->sums = fmax(e->sums, fabs(dev->min_q - h.min_q));
  e->counts = fmax(e->counts, fabs(dev->n_edge - h.n_edge));
  e->counts = fmax(e->counts, fabs(dev->n_out_of_band - h.n_out_of_band));
  e->counts = fmax(e->counts, fabs(dev->n_inverted - h.n_inverted));
}

/* ═══════════════════════════════════════════════════════════════════════════
 * -csv / -compare
 * ═══════════════════════════════════════════════════════════════════════════ */

static void csv_row(FILE *f, int step, double t, const report *r, const report *r0, uint64_t mesh)
{
  fprintf(f, "%d,%.10g,%.0f,%.0f,%.0f,%.0f,%.0f,%.17g,%.17g,%.17g,%.17g,%016" PRIx64 "\n", step, t, r->n_tri, r->n_vert, r->n_edge, r->n_out_of_band,
          r->n_inverted, r->min_q, r->avg_q, r->volume / r0->volume, r->area / r0->area, mesh);
}

typedef struct
{
  int step;
  double ntri, nvert, nedge, oob, inv, min_q, avg_q, v, a;
  uint64_t mesh;
} csv_rec;

static int csv_read(const char *path, csv_rec **out)
{
  FILE *f = fopen(path, "r");
  if (!f) return -1;
  char line[512];
  int n = 0, cap = 0;
  csv_rec *r = NULL;
  while (fgets(line, sizeof line, f))
  {
    csv_rec x;
    double t;
    if (sscanf(line, "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%" SCNx64, &x.step, &t, &x.ntri, &x.nvert, &x.nedge, &x.oob, &x.inv, &x.min_q, &x.avg_q, &x.v,
               &x.a, &x.mesh) != 12)
      continue;  // the header
    if (n == cap)
    {
      cap = cap ? 2 * cap : 256;
      r = (csv_rec *) realloc(r, (size_t) cap * sizeof *r);
    }
    r[n++] = x;
  }
  fclose(f);
  *out = r;
  return n;
}

/**
 * @brief The trajectory against a -cpu run's. Returns 0 when it agrees.
 *
 * Agreement means one discrete trajectory: the same mesh after every remesh
 * (mesh_key) and every diagnostic equal to round-off. Remeshing is a sequence
 * of threshold decisions, so this is only possible where the mesh resolves the
 * flow -- where it does not, the remesher amplifies round-off until a decision
 * goes the other way, on the host alone as well. That is reported, with how far
 * the two runs then drift apart, and fails.
 */
static int compare_trajectory(const char *path, const csv_rec *mine, int n_mine)
{
  csv_rec *ref = NULL;
  const int n_ref = csv_read(path, &ref);
  if (n_ref <= 0)
  {
    fprintf(stderr, "COMPARE: cannot read %s\n", path);
    return 1;
  }

  /* Seen: up to 6e-13 at refine 4 (150 steps) and 5 (60), np 1-4, every backend (README.md). */
  const double tol = 1e-11;
  const int n = (n_ref < n_mine) ? n_ref : n_mine;
  int split = -1, apart = -1;
  double d_max = 0.0, d_v = 0.0, d_a = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const csv_rec *a = &mine[i], *b = &ref[i];
    const double d = fmax(fmax(rel_diff(a->v, b->v), rel_diff(a->a, b->a)), fmax(fabs(a->avg_q - b->avg_q), fabs(a->min_q - b->min_q)));
    if (split < 0 && a->mesh != b->mesh) split = a->step;
    if (apart < 0 && d > tol) apart = a->step;
    d_max = fmax(d_max, d);
    d_v = fmax(d_v, fabs(a->v - b->v));
    d_a = fmax(d_a, fabs(a->a - b->a));
  }
  const int ok = n == n_mine && split < 0 && apart < 0;

  printf("COMPARE vs %s: %d steps", path, n);
  if (ok)
    printf(", the same mesh at every step, max rel. difference %.3e (tol %.0e) -> ok\n", d_max, tol);
  else
  {
    printf(" -> FAILED\n");
    if (n != n_mine) printf("COMPARE: %s has %d steps, this run %d\n", path, n_ref, n_mine);
    if (apart > 0) printf("COMPARE: round-off agreement through step %d\n", apart - 1);
    if (split > 0) printf("COMPARE: the same mesh through step %d, then a remeshing decision went the other way\n", split - 1);
    printf("COMPARE: max |d V/V0| %.3e, max |d A/A0| %.3e (an under-resolved mesh amplifies round-off: see README.md)\n", d_v, d_a);
  }
  free(ref);
  return !ok;
}

/* ═══════════════════════════════════════════════════════════════════════════ */

static void grid_for(int nproc, int cuts[3])
{
  static const int table[][4] = {{1, 1, 1, 1},  {2, 2, 1, 1},  {3, 3, 1, 1},  {4, 2, 2, 1},  {6, 3, 2, 1}, {8, 2, 2, 2},
                                 {12, 3, 2, 2}, {16, 4, 2, 2}, {24, 4, 3, 2}, {27, 3, 3, 3}, {32, 4, 4, 2}};
  for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); ++i)
    if (table[i][0] == nproc)
    {
      cuts[0] = table[i][1];
      cuts[1] = table[i][2];
      cuts[2] = table[i][3];
      return;
    }
  cuts[0] = nproc;
  cuts[1] = 1;
  cuts[2] = 1;
}

static void usage_error(int rank, const char *msg, const char *arg)
{
  if (rank == 0) fprintf(stderr, "leveque_gpu: %s%s%s\n", msg, arg ? ": " : "", arg ? arg : "");
  MPI_Finalize();
  exit(2);
}

static double parse_double(int rank, const char *s, double lo)
{
  char *end = NULL;
  const double x = strtod(s, &end);
  if (!s[0] || *end || !isfinite(x) || x <= lo) usage_error(rank, "invalid number", s);
  return x;
}

static int parse_int(int rank, const char *s, long lo, long hi)
{
  char *end = NULL;
  const long x = strtol(s, &end, 10);
  if (!s[0] || *end || x < lo || x > hi) usage_error(rank, "invalid integer", s);
  return (int) x;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MFTLRegisterMpiTypes();

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int refinements = 4, max_steps = -1, use_cpu = 0, verify = 0, quiet = 0;
  double T = 3.0, dt = 0.02;
  const char *csv_path = NULL, *compare_path = NULL;

  for (int i = 1; i < argc; ++i)
  {
    const char *a = argv[i];
    const int has_value = i + 1 < argc;
    if (!strcmp(a, "-refine") && has_value)
      refinements = parse_int(rank, argv[++i], 0, 8);
    else if (!strcmp(a, "-T") && has_value)
      T = parse_double(rank, argv[++i], 0.0);
    else if (!strcmp(a, "-dt") && has_value)
      dt = parse_double(rank, argv[++i], 0.0);
    else if (!strcmp(a, "-steps") && has_value)
      max_steps = parse_int(rank, argv[++i], 0, 1000000);
    else if (!strcmp(a, "-csv") && has_value)
      csv_path = argv[++i];
    else if (!strcmp(a, "-compare") && has_value)
      compare_path = argv[++i];
    else if (!strcmp(a, "-cpu"))
      use_cpu = 1;
    else if (!strcmp(a, "-verify"))
      verify = 1;
    else if (!strcmp(a, "-quiet"))
      quiet = 1;
    else
      usage_error(rank, "unknown or incomplete option (see the head of leveque_gpu.c)", a);
  }

  /* Every rank on one node shares its devices round-robin. */
  int local_rank = 0;
  {
    MPI_Comm node;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local_rank);
    MPI_Comm_free(&node);
  }
  if (!use_cpu)
  {
    int status = lv_init(local_rank), worst = 0;
    MPI_Allreduce(&status, &worst, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (worst != 0)
    {
      if (rank == 0) printf("leveque_gpu (%s): no device on some rank -- skipped\n", lv_backend_name());
      MFTLFreeMpiTypes();
      MPI_Finalize();
      return 77;
    }
  }
  const char *backend = use_cpu ? "cpu (library)" : lv_backend_name();
  /* The *-host builds run the same pragmas on CPU threads: no device, no copies. */
  const int on_device = !use_cpu && !strstr(backend, "host");

  /* --- Setup: leveque_remeshing_parallel.c with its defaults ------------- */
  int cuts[3];
  grid_for(nproc, cuts);
  bool periodic[3] = {MFTL_FALSE, MFTL_FALSE, MFTL_FALSE};
  double xmin[3] = {0.0, 0.0, 0.0}, xmax[3] = {1.0, 1.0, 1.0};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, xmin, xmax, periodic, 1);
  if (!ens) MPI_Abort(MPI_COMM_WORLD, 1);

  MFTL_Object *obj0 = MFTLObjectCreate(0);
  MFTLEnsembleAddObject(ens, obj0);
  double centre[3] = {0.35, 0.35, 0.35};
  MFTLGenerateSphere(obj0, centre, 0.15, refinements);
  MFTLComputeTriangleGeometry(obj0, obj0->tris.n);
  const double target = MFTLEnsembleMeanEdgeLength(ens);

  MFTL_RemeshParameters params;
  MFTLInitRemeshParams(&params, MFTL_REMESH_ISOTROPIC, target, 3, MFTL_SMOOTH_TRIANGLE_BASED, 180.0);
  params.smooth_iterations = 3;
  params.parallel_strategy = MFTL_REMESH_PARALLEL_SPANNING;

  if (MFTLRedistribute(ens) != MFTL_SUCCESS) MPI_Abort(MPI_COMM_WORLD, 1);
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLEnsembleUpdateGlobalCounts(ens);

  int attached = 0;
  if (!use_cpu && the_object(ens))
  {
    const lv_mesh m = view_of(the_object(ens));
    lv_attach(&m);
    attached = 1;
  }

  report r0;
  lv_measure_out dev_sums;
  if (use_cpu)
    cpu_measure(ens, &params, &r0);
  else
    gpu_measure(ens, &params, &r0, &dev_sums);

  int n_steps = (int) lround(T / dt);
  if (max_steps >= 0 && max_steps < n_steps) n_steps = max_steps;

  if (rank == 0 && !quiet)
  {
    printf("# LeVeque deformation, distributed remeshing, device work on: %s\n", backend);
    printf("#   %d ranks on a %dx%dx%d grid   sphere r=0.15, %d refinements -> %.0f triangles   T=%.2f dt=%.3f, %d steps\n", nproc, cuts[0], cuts[1], cuts[2],
           refinements, r0.n_tri, T, dt, n_steps);
    printf("#   V0 = %.12e   A0 = %.12e\n#\n", r0.volume, r0.area);
    printf("# %5s %7s %9s %8s %8s %5s %6s %18s %18s\n", "step", "t", "ntri", "minQ", "avgQ", "inv", "oob", "V/V0", "A/A0");
  }

  FILE *csv = NULL;
  if (csv_path && rank == 0)
  {
    csv = fopen(csv_path, "w");
    if (!csv) MPI_Abort(MPI_COMM_WORLD, 1);
    fprintf(csv, "step,t,ntri,nvert,nedge,out_of_band,inverted,min_q,avg_q,V_over_V0,A_over_A0,mesh\n");
  }
  csv_rec *traj = (csv_rec *) calloc((size_t) (n_steps > 0 ? n_steps : 1), sizeof(csv_rec));

  /* --- The loop ------------------------------------------------------------ */
  double tm[T_N] = {0};
  double bytes_h2d = 0.0, bytes_d2h = 0.0, peak_tri = r0.n_tri;
  errors err = {0, 0, 0, 0, 0, 0};
  report r = r0;

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_start = MPI_Wtime();

  /* Time is ACCUMULATED, as in leveque_remeshing_parallel.c: step * dt differs
   * from it in the last bits, and that alone flips a remeshing decision about
   * a hundred steps in -- -cpu would then no longer be the original example. */
  double t = 0.0;
  for (int step = 0; step < n_steps; ++step, t += dt)
  {
    MFTL_Object *obj = the_object(ens);
    double a;

    if (use_cpu)
    {
      a = MPI_Wtime();
      if (obj) cpu_advect(obj, t, dt, T);
      tm[T_ADVECT] += MPI_Wtime() - a;
    }
    else if (obj)
    {
      double *before = NULL;
      if (verify) before = (double *) memcpy(scratch(3 * (size_t) obj->verts.n_local), obj->verts.pos, 3 * (size_t) obj->verts.n_local * sizeof(double));

      a = MPI_Wtime();
      lv_advect(t, dt, T);
      tm[T_ADVECT] += MPI_Wtime() - a;

      a = MPI_Wtime();
      lv_get_owned_positions();
      tm[T_D2H] += MPI_Wtime() - a;
      bytes_d2h += 24.0 * obj->verts.n_local;

      if (verify)
      {
        for (int v = 0; v < obj->verts.n_local; ++v) lv_rk4(before + 3 * v, t, dt, T);
        err.pos = fmax(err.pos, max_abs_diff(before, obj->verts.pos, 3 * (size_t) obj->verts.n_local));
      }
    }
    if (obj) MFTLObjectMarkChanged(obj);

    /* The ghost sync stays on the host: it is MPI, and the triangle centroids
     * computed next decide where redistribution sends every triangle -- two
     * ranks must see the same ghost positions. */
    a = MPI_Wtime();
    MFTLUpdateGhostPositions(ens);
    tm[T_GHOST] += MPI_Wtime() - a;

    if (use_cpu)
    {
      a = MPI_Wtime();
      if (obj) MFTLComputeTriangleGeometry(obj, obj->tris.n);
      tm[T_GEOM] += MPI_Wtime() - a;
    }
    else if (obj)
    {
      a = MPI_Wtime();
      lv_put_ghost_positions();
      tm[T_H2D] += MPI_Wtime() - a;
      bytes_h2d += 24.0 * (obj->verts.n - obj->verts.n_local);

      a = MPI_Wtime();
      lv_triangle_geometry();
      tm[T_GEOM] += MPI_Wtime() - a;

      a = MPI_Wtime();
      lv_get_triangle_geometry();
      tm[T_D2H] += MPI_Wtime() - a;
      bytes_d2h += 56.0 * obj->tris.n;

      /* What MFTLComputeTriangleGeometry() sets after a pass over tris.n: the
       * redistribution inside the remesh then routes by these centroids
       * instead of recomputing them. */
      obj->derived |= MFTL_DERIVED_GEOMETRY;
      if (verify) verify_geometry(obj, &err);

      a = MPI_Wtime();
      lv_detach();
      attached = 0;
      tm[T_H2D] += MPI_Wtime() - a;
    }

    MFTL_RemeshStats st;
    memset(&st, 0, sizeof st);
    a = MPI_Wtime();
    if (MFTLRemeshEnsemble(ens, &params, &st) != MFTL_SUCCESS)
    {
      if (rank == 0) fprintf(stderr, "remesh failed at step %d\n", step);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    tm[T_REMESH] += MPI_Wtime() - a;

    obj = the_object(ens);
    if (use_cpu)
    {
      a = MPI_Wtime();
      cpu_measure(ens, &params, &r);
      tm[T_MEASURE] += MPI_Wtime() - a;
    }
    else
    {
      /* A rank that held nothing before the remesh may hold a share now, and vice versa. */
      if (obj)
      {
        a = MPI_Wtime();
        const lv_mesh m = view_of(obj);
        lv_attach(&m);
        attached = 1;
        tm[T_H2D] += MPI_Wtime() - a;
        bytes_h2d += 24.0 * m.nv + 36.0 * m.nt;
      }

      a = MPI_Wtime();
      gpu_measure(ens, &params, &r, &dev_sums);
      tm[T_MEASURE] += MPI_Wtime() - a;
      if (verify && obj) verify_measure(obj, &params, &dev_sums, &err);
    }
    if (r.n_tri > peak_tri) peak_tri = r.n_tri;

    const uint64_t mesh = (csv_path || compare_path) ? mesh_key(ens) : 0;
    traj[step] =
        (csv_rec){step + 1, r.n_tri, r.n_vert, r.n_edge, r.n_out_of_band, r.n_inverted, r.min_q, r.avg_q, r.volume / r0.volume, r.area / r0.area, mesh};
    if (csv) csv_row(csv, step + 1, t + dt, &r, &r0, mesh);
    if (rank == 0 && !quiet)
      printf("  %5d %7.3f %9.0f %8.5f %8.5f %5.0f %6.0f %18.15f %18.15f\n", step + 1, t + dt, r.n_tri, r.min_q, r.avg_q, r.n_inverted, r.n_out_of_band,
             r.volume / r0.volume, r.area / r0.area);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double wall = MPI_Wtime() - t_start;
  if (csv) fclose(csv);
  if (attached) lv_detach();

  /* --- Summary --------------------------------------------------------------- */
  double tmax[T_N], bytes[2] = {bytes_h2d, bytes_d2h}, bmax[2];
  MPI_Reduce(tm, tmax, T_N, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(bytes, bmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  errors gerr;
  MPI_Reduce(&err, &gerr, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  long allocs = use_cpu ? 0 : lv_device_allocations(), allocs_max = 0;
  MPI_Reduce(&allocs, &allocs_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

  int failed = 0;
  if (rank == 0)
  {
    const double per = 1000.0 / (n_steps > 0 ? n_steps : 1);
    const double offload = tmax[T_ADVECT] + tmax[T_GEOM] + tmax[T_MEASURE] + tmax[T_D2H] + tmax[T_H2D];
    printf("==============================================\n");
    printf("  LeVeque on %s, %d ranks, refine %d: %d steps, peak %.0f triangles, final %.0f\n", backend, nproc, refinements, n_steps, peak_tri, r.n_tri);
    printf("  final  V/V0 %.12f   A/A0 %.6f   min q %.4f   avg q %.4f   inverted %.0f\n", r.volume / r0.volume, r.area / r0.area, r.min_q, r.avg_q,
           r.n_inverted);
    printf("  TIME per step (max over ranks)            ms/step   share\n");
    for (int i = 0; i < T_N; ++i)
    {
      const char *where = (i <= T_MEASURE) ? (on_device ? "device" : "host  ") : (i <= T_H2D && on_device ? "PCIe  " : "host  ");
      printf("    %-12s %s %10.4f %6.1f %%\n", timer_name[i], where, per * tmax[i], 100.0 * tmax[i] / wall);
    }
    printf("    %-19s %10.4f\n", "wall", per * wall);
    printf("    %-19s %10.4f %6.1f %%   (advect + geometry + measure + transfers)\n", "offloadable part", per * offload, 100.0 * offload / wall);
    if (on_device && tmax[T_H2D] + tmax[T_D2H] > 0.0)
      printf("    transfers: H->D %.2f MB/step at %.2f GB/s, D->H %.2f MB/step at %.2f GB/s (pageable host memory)\n", 1e-6 * bmax[0] / n_steps,
             1e-9 * bmax[0] / fmax(tmax[T_H2D], 1e-12), 1e-6 * bmax[1] / n_steps, 1e-9 * bmax[1] / fmax(tmax[T_D2H], 1e-12));
    if (!use_cpu)
      printf("    device memory: %ld (re)allocations of the mesh arrays in %d steps (max over ranks); every other upload reuses them\n", allocs_max, n_steps);
  }

  if (verify)
  {
    if (rank == 0)
    {
      /* Structure only: the volume is -compare's, against the host run. */
      const int mesh_ok = r.n_inverted == 0.0 && r.n_vert - r.n_edge + r.n_tri == 2.0;
      if (!mesh_ok) failed = 1;
      printf("VERIFY mesh: inverted %.0f  Euler %.0f -> %s\n", r.n_inverted, r.n_vert - r.n_edge + r.n_tri, mesh_ok ? "ok" : "FAILED");
      if (!use_cpu)
      {
        const double *e = &gerr.pos, *tol = &verify_tol.pos;
        static const char *const what[6] = {"positions", "area (rel)", "normals", "centroids", "measure sums", "measure counts"};
        int dev_ok = 1;
        printf("VERIFY device vs library, every step, max over ranks:\n");
        for (int i = 0; i < 6; ++i)
        {
          const int ok = e[i] <= tol[i];
          dev_ok &= ok;
          printf("    %-15s %10.3e   (tol %.0e) %s\n", what[i], e[i], tol[i], ok ? "ok" : "FAILED");
        }
        if (!dev_ok) failed = 1;
      }
    }
  }

  if (compare_path && rank == 0 && compare_trajectory(compare_path, traj, n_steps)) failed = 1;
  MPI_Bcast(&failed, 1, MPI_INT, 0, MPI_COMM_WORLD);

  free(traj);
  free(g_scratch);
  if (!use_cpu) lv_finalize();
  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return failed;
}
