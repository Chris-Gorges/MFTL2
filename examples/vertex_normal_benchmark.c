/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Three ways to compute a vertex normal, measured against each other.
 *
 * A vertex normal is a sum over the faces incident to that vertex. The sum can
 * be driven from either end, and the two ends have completely different memory
 * behaviour:
 *
 *   A  SCATTER over triangles   -- what MFTLComputeVertexNormals does. Two
 *      contiguous input streams, a flat loop, no pointer chasing; pays a zeroing
 *      pass, a read-modify-write into three indexed vertex slots, and a
 *      normalisation pass.
 *
 *   B  GATHER over vertices, fan walk -- the half-edge structure used the way it
 *      was designed. One sequential write stream and no zeroing pass; pays a
 *      data-dependent trip count and a chain of DEPENDENT loads, since each hop
 *      of the rotation is he -> twin -> next and cannot issue until the previous
 *      hop has returned.
 *
 *   C  GATHER over vertices, CSR adjacency -- the same gather with the fan
 *      flattened into vert_offsets/vert_tris first. This removes B's dependent
 *      loads: the incident triangle list is contiguous and every element of it
 *      is known without touching the previous one. It costs a build pass and
 *      16 bytes of permanent state per triangle.
 *
 * C is here specifically to settle the note in src/object/derived.c: the CSR
 * adjacency was deleted from this version for want of a consumer, and vertex
 * normals are the first plausible consumer. Its build cost is timed separately
 * and reported, because on a mesh whose topology is static the build amortises
 * over many timesteps and on a remeshed one it does not.
 *
 * All three write the same array and are checked against each other before any
 * timing is believed.
 *
 * Build Release for anything you intend to quote:
 *     python3 build_all.py && ./build/vertex_normal_benchmark
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mftl_core.h"

#define REP 20
#define TRIALS 9

/* ═══════════════════════════════════════════════════════════════════════════
 * B: gather over vertices, walking the half-edge fan
 * ═══════════════════════════════════════════════════════════════════════════ */

static void vn_fan_gather(MFTL_Object *obj, double *restrict out)
{
  const MFTL_HalfEdge *restrict he = obj->tris.he;
  const MFTL_Vertex *restrict vr = obj->verts.rec;
  const double *restrict ed = obj->tris.edge;
  const int nv = obj->verts.n;

  for (int v = 0; v < nv; ++v)
  {
    double sx = 0.0, sy = 0.0, sz = 0.0;

    int start = vr[v].out_he_idx;
    if (start != MFTL_ERROR)
    {
      /* Rewind only where it is needed. On a CLOSED fan any half-edge is a valid
       * start, and MFTLVertexFanStart would walk a full extra lap to discover
       * that -- doubling the work of the entire kernel. MFTL_FLAG_BOUNDARY says
       * which vertices actually need it, so this is the fast path a well-written
       * gather would take, and B is measured with it rather than without. */
      if (MFTLVertexIsMeshBoundary(obj, v)) start = MFTLVertexFanStart(obj, start, MFTL_NULL);

      int curr = start;
      do
      {
        const int t = MFTLHeTriIdx(curr);

        const double e1x = ed[6 * t + 0], e1y = ed[6 * t + 1], e1z = ed[6 * t + 2];
        const double e2x = ed[6 * t + 3], e2y = ed[6 * t + 4], e2z = ed[6 * t + 5];

        sx += e1y * e2z - e1z * e2y;
        sy += e1z * e2x - e1x * e2z;
        sz += e1x * e2y - e1y * e2x;

        const int twin = he[curr].twin_idx;
        if (twin == MFTL_ERROR) break;
        curr = MFTLHeNextIdx(twin);
      } while (curr != start);
    }

    const double len_sq = sx * sx + sy * sy + sz * sz;
    const double inv = (len_sq > MFTL_VERYSMALL) ? 1.0 / MFTL_SQRT(len_sq) : 0.0;
    out[3 * v + 0] = sx * inv;
    out[3 * v + 1] = sy * inv;
    out[3 * v + 2] = sz * inv;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * C: gather over vertices, through a vertex -> triangle CSR
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  int *offsets; /* [nv+1] */
  int *tris;    /* [3*nt] */
  int nv;
} csr_adjacency;

/** @brief Counting sort of the 3*nt (vertex, triangle) incidences. */
static void csr_build(const MFTL_Object *obj, csr_adjacency *csr)
{
  const int nv = obj->verts.n;
  const int nt = obj->tris.n;
  const int *restrict tv = obj->tris.verts;

  memset(csr->offsets, 0, (size_t) (nv + 1) * sizeof(int));

  for (int i = 0; i < 3 * nt; ++i) csr->offsets[tv[i] + 1]++;
  for (int v = 0; v < nv; ++v) csr->offsets[v + 1] += csr->offsets[v];

  /* cursor[] would be a second nv-sized array; walking a local copy of the
   * offsets forward instead keeps the build to the two arrays it returns. */
  int *cursor = (int *) MFTLMallocOrDie((size_t) nv * sizeof(int), "csr cursor");
  memcpy(cursor, csr->offsets, (size_t) nv * sizeof(int));

  for (int t = 0; t < nt; ++t)
  {
    for (int k = 0; k < 3; ++k) csr->tris[cursor[tv[3 * t + k]]++] = t;
  }

  free(cursor);
  csr->nv = nv;
}

static void vn_csr_gather(const MFTL_Object *obj, const csr_adjacency *csr, double *restrict out)
{
  const double *restrict ed = obj->tris.edge;
  const int *restrict off = csr->offsets;
  const int *restrict vt = csr->tris;
  const int nv = csr->nv;

  for (int v = 0; v < nv; ++v)
  {
    double sx = 0.0, sy = 0.0, sz = 0.0;

    const int lo = off[v];
    const int hi = off[v + 1];
    for (int i = lo; i < hi; ++i)
    {
      const int t = vt[i];

      const double e1x = ed[6 * t + 0], e1y = ed[6 * t + 1], e1z = ed[6 * t + 2];
      const double e2x = ed[6 * t + 3], e2y = ed[6 * t + 4], e2z = ed[6 * t + 5];

      sx += e1y * e2z - e1z * e2y;
      sy += e1z * e2x - e1x * e2z;
      sz += e1x * e2y - e1y * e2x;
    }

    const double len_sq = sx * sx + sy * sy + sz * sz;
    const double inv = (len_sq > MFTL_VERYSMALL) ? 1.0 / MFTL_SQRT(len_sq) : 0.0;
    out[3 * v + 0] = sx * inv;
    out[3 * v + 1] = sy * inv;
    out[3 * v + 2] = sz * inv;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Harness
 * ═══════════════════════════════════════════════════════════════════════════ */

typedef struct
{
  MFTL_Object *obj;
  csr_adjacency csr;
  double *out;
} bench_ctx;

static void run_scatter(bench_ctx *c) { MFTLComputeVertexNormals(c->obj); }
static void run_fan(bench_ctx *c) { vn_fan_gather(c->obj, c->out); }
static void run_csr(bench_ctx *c) { vn_csr_gather(c->obj, &c->csr, c->out); }
static void run_csr_build(bench_ctx *c) { csr_build(c->obj, &c->csr); }

/** @brief Best of TRIALS timed runs, in nanoseconds per vertex. */
static double best_ns(void (*fn)(bench_ctx *), bench_ctx *c, double elements)
{
  fn(c); /* warm-up */

  double best = 1e30;
  for (int r = 0; r < TRIALS; ++r)
  {
    const double t0 = MPI_Wtime();
    for (int i = 0; i < REP; ++i) fn(c);
    const double dt = (MPI_Wtime() - t0) / (REP * elements) * 1e9;
    if (dt < best) best = dt;
  }
  return best;
}

/** @brief Largest disagreement between a reference copy and the object's array. */
static double max_diff(const double *a, const double *b, int nv)
{
  double worst = 0.0;
  for (int i = 0; i < 3 * nv; ++i)
  {
    const double d = fabs(a[i] - b[i]);
    if (d > worst) worst = d;
  }
  return worst;
}

/**
 * @brief The physical check: on a sphere the normal must be the radial direction.
 * @return Largest angular error in degrees over all vertices.
 */
static double max_radial_error_deg(const MFTL_Object *obj, const double centre[3])
{
  double worst = 0.0;
  for (int v = 0; v < obj->verts.n; ++v)
  {
    const double *p = MFTLVertexPos(obj, v);
    double r[3] = {p[0] - centre[0], p[1] - centre[1], p[2] - centre[2]};
    const double rl = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (rl < MFTL_VERYSMALL) continue;

    const double *n = &obj->verts.normal[3 * v];
    double dot = (r[0] * n[0] + r[1] * n[1] + r[2] * n[2]) / rl;
    if (dot > 1.0) dot = 1.0;
    if (dot < -1.0) dot = -1.0;

    const double err = acos(dot) * 180.0 / MFTL_PI;
    if (err > worst) worst = err;
  }
  return worst;
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0)
  {
    MPI_Finalize();
    return 0;
  }

  int cuts[3] = {1, 1, 1};
  double lo[3] = {-2.0, -2.0, -2.0};
  double hi[3] = {2.0, 2.0, 2.0};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, lo, hi, MFTL_NULL, 8);

  printf("Vertex normals: scatter over triangles vs gather over vertices\n");
  printf("ns per VERTEX, best of %d runs of %d repetitions. Lower is better.\n\n", TRIALS, REP);

  printf("%-4s %9s %9s | %10s %10s %10s | %9s %9s | %10s %9s\n", "ref", "verts", "tris", "A scatter", "B fan", "C csr", "A vs B", "A vs C",
         "csr build", "radial err");
  printf("-------------------------------------------------------------------------------------------------------------\n");

  for (int ref = 3; ref <= 8; ++ref)
  {
    MFTL_Object *obj = MFTLObjectCreate(1);
    MFTLEnsembleAddObject(ens, obj);
    double centre[3] = {0.0, 0.0, 0.0};
    MFTLGenerateSphere(obj, centre, 1.0, ref);
    MFTLObjectReorderMorton(obj);

    const int nv = obj->verts.n;
    const int nt = obj->tris.n;

    bench_ctx c;
    c.obj = obj;
    c.out = (double *) MFTLMallocOrDie(3 * (size_t) nv * sizeof(double), "variant output");
    c.csr.offsets = (int *) MFTLMallocOrDie((size_t) (nv + 1) * sizeof(int), "csr offsets");
    c.csr.tris = (int *) MFTLMallocOrDie(3 * (size_t) nt * sizeof(int), "csr tris");
    csr_build(obj, &c.csr);

    /* Correctness first. All three must produce the same array; the sums are
     * taken in different orders, so they agree to rounding rather than exactly. */
    MFTLComputeVertexNormals(obj);
    double *ref_normals = (double *) MFTLMallocOrDie(3 * (size_t) nv * sizeof(double), "reference normals");
    memcpy(ref_normals, obj->verts.normal, 3 * (size_t) nv * sizeof(double));

    vn_fan_gather(obj, c.out);
    const double d_ab = max_diff(ref_normals, c.out, nv);

    vn_csr_gather(obj, &c.csr, c.out);
    const double d_ac = max_diff(ref_normals, c.out, nv);

    const double radial = max_radial_error_deg(obj, centre);

    const double t_a = best_ns(run_scatter, &c, (double) nv);
    const double t_b = best_ns(run_fan, &c, (double) nv);
    const double t_c = best_ns(run_csr, &c, (double) nv);
    const double t_build = best_ns(run_csr_build, &c, (double) nv);

    printf("%-4d %9d %9d | %10.3f %10.3f %10.3f | %9.1e %9.1e | %10.3f %8.3f deg\n", ref, nv, nt, t_a, t_b, t_c, d_ab, d_ac, t_build, radial);

    free(ref_normals);
    free(c.out);
    free(c.csr.offsets);
    free(c.csr.tris);
  }

  printf("\nNotes\n");
  printf("  A  MFTLComputeVertexNormals -- scatter over triangles. Streams tris.verts and\n");
  printf("     tris.edge, scatter-adds into verts.normal, then normalises in a second pass.\n");
  printf("  B  fan walk over the half-edge structure. One write stream and no zeroing pass,\n");
  printf("     but every hop of the rotation is a load that depends on the previous one.\n");
  printf("  C  the same gather through a vertex->triangle CSR, which removes B's dependent\n");
  printf("     loads. 'csr build' is what one rebuild costs, per vertex, and it must be paid\n");
  printf("     on every topology change -- so on a remeshed mesh it is part of C's price.\n");
  printf("\n");
  printf("  'A vs B' and 'A vs C' are the largest disagreement between the variants. They\n");
  printf("  sum the same faces in different orders, so agreement is to rounding, not exact.\n");
  printf("  'radial err' is the physical check: on a sphere the vertex normal must point\n");
  printf("  radially outward, and the error must fall as the mesh is refined.\n");

  MFTLEnsembleDestroy(&ens);
  MPI_Finalize();
  return 0;
}
