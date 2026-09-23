/*
 * Stage 15 library features (plans/15-mftl1-to-mftl2-swap.md, items L1-L7):
 *   L1 rectilinear per-axis cuts + explicit slab -> rank map (ownership, neighbours,
 *      re-decomposition of an existing ensemble)
 *   L2 caller-held point query (batch == single-point verdicts)
 *   L3 retained twin gid (local invariant + global symmetry, rim included)
 *   L4 detach / transfer an object between ensembles
 *   L5 single-vertex normal / Laplace-Beltrami
 *   L7 conserved surface scalar (transport, split/collapse/flip/TSUR mass conservation)
 * plus the explicit-id soup builder and the MFTL1-format native mesh round trip.
 *
 * Every check is collective: a failure on any rank fails the test on all ranks, so no rank
 * leaves a collective early. Registered at 1, 2, 3 and 4 ranks.
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mftl_core.h"
#include "mftl_communication.h"
#include "mftl_geometry.h"
#include "mftl_io.h"
#include "mftl_remeshing.h"
#include "mftl_topology.h"

static int g_rank = 0, g_nproc = 1, g_failed = 0, g_checks = 0;

#define CHECK(cond)                                                                                       \
  do                                                                                                      \
  {                                                                                                       \
    int _l = (cond) ? 0 : 1, _g = 0;                                                                      \
    MPI_Allreduce(&_l, &_g, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);                                         \
    g_checks++;                                                                                           \
    if (_g)                                                                                               \
    {                                                                                                     \
      if (_l) fprintf(stderr, "[rank %d] CHECK failed: %s (%s:%d)\n", g_rank, #cond, __FILE__, __LINE__); \
      g_failed++;                                                                                         \
    }                                                                                                     \
  } while (0)

// factor nproc into a 3-D grid (x-fastest)
static void grid3(int np, int g[3])
{
  g[0] = np;
  g[1] = 1;
  g[2] = 1;
  if (np == 4)
  {
    g[0] = 2;
    g[1] = 2;
  }
}

static const double GMIN[3] = {-1.5, -1.5, -1.5}, GMAX[3] = {1.5, 1.5, 1.5};

// Rectilinear ensemble with non-uniform cuts and a REVERSED slab -> rank map (not the MPI
// Cartesian order), so ownership only works through the explicit map.
static MFTL_Ensemble* make_rect_ensemble(const int g[3], double shift)
{
  double* cuts[3];
  for (int d = 0; d < 3; ++d)
  {
    cuts[d] = (double*) malloc((size_t) (g[d] + 1) * sizeof(double));
    for (int i = 0; i <= g[d]; ++i)
    {
      double u = (double) i / (double) g[d];
      cuts[d][i] = GMIN[d] + (GMAX[d] - GMIN[d]) * u;
      if (i > 0 && i < g[d]) cuts[d][i] += shift * (d + 1) * 0.1;  // non-uniform interior cuts
    }
  }
  int* rmap = (int*) malloc((size_t) g_nproc * sizeof(int));
  for (int i = 0; i < g_nproc; ++i) rmap[i] = g_nproc - 1 - i;
  bool per[3] = {false, false, false};
  MFTL_Ensemble* ens = MFTLEnsembleCreateRectilinear(MPI_COMM_WORLD, g, (const double* const*) cuts, rmap, GMIN, GMAX, per, 4);
  for (int d = 0; d < 3; ++d) free(cuts[d]);
  free(rmap);
  return ens;
}

static int expected_owner(const MFTL_Domain* dom, const double p[3])
{
  int ijk[3];
  for (int d = 0; d < 3; ++d)
  {
    ijk[d] = 0;
    for (int i = 1; i < dom->cuts[d]; ++i)
      if (p[d] >= dom->axis_cuts[d][i]) ijk[d] = i;
  }
  return dom->rank_map[ijk[0] + ijk[1] * dom->cuts[0] + ijk[2] * dom->cuts[0] * dom->cuts[1]];
}

static double global_mass(MFTL_Object* obj)
{
  double m = 0.0, gm = 0.0;
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    if (tri->is_ghost || tri->owner_proc != g_rank || (tri->flags & MFTL_FLAG_DEAD)) continue;
    double a = 0.0;
    MFTLGetTriangleProperties(obj, tri, &a, NULL, NULL);
    m += tri->surface_scalar * a;
  }
  MPI_Allreduce(&m, &gm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return gm;
}

static int global_owned_tris(MFTL_Object* obj)
{
  int n = 0, gn = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (!obj->triangles[t].is_ghost && obj->triangles[t].owner_proc == g_rank) n++;
  MPI_Allreduce(&n, &gn, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  return gn;
}

static void set_scalar_field(MFTL_Object* obj)
{
  for (int t = 0; t < obj->n_triangles; ++t)
  {
    MFTL_Triangle* tri = &obj->triangles[t];
    MFTLGetTriangleProperties(obj, tri, &tri->area, tri->normal, tri->center);
    tri->surface_scalar = 1.0 + 0.5 * tri->center[0] + 0.25 * tri->center[1] * tri->center[2];
  }
}

// ---------------------------------------------------------------------------------------
// L1
// ---------------------------------------------------------------------------------------
static void test_rectilinear_ownership(void)
{
  int g[3];
  grid3(g_nproc, g);
  MFTL_Ensemble* ens = make_rect_ensemble(g, 0.3);
  CHECK(ens != NULL);
  MFTL_Domain* dom = &ens->domain;
  int bad = 0;
  for (int s = 0; s < 500; ++s)
  {
    double p[3];
    for (int d = 0; d < 3; ++d) p[d] = GMIN[d] + (GMAX[d] - GMIN[d]) * (double) ((s * 7919 + d * 104729) % 1000) / 999.0;
    if (MFTLGetOwnerProc(dom, p) != expected_owner(dom, p)) bad++;
  }
  CHECK(bad == 0);
  // points outside the box clamp to the boundary slabs (no assertion)
  double out[3] = {GMIN[0] - 1.0, GMAX[1] + 1.0, 0.0};
  int o = MFTLGetOwnerProc(dom, out);
  CHECK(o >= 0 && o < g_nproc);
  // neighbour relation is symmetric
  int* nb = (int*) calloc((size_t) g_nproc * g_nproc, sizeof(int));
  int* gnb = (int*) calloc((size_t) g_nproc * g_nproc, sizeof(int));
  for (int i = 0; i < dom->num_neighbors; ++i) nb[g_rank * g_nproc + dom->neighbors[i]] = 1;
  MPI_Allreduce(nb, gnb, g_nproc * g_nproc, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  int asym = 0;
  for (int a = 0; a < g_nproc; ++a)
    for (int b = 0; b < g_nproc; ++b)
      if (gnb[a * g_nproc + b] != gnb[b * g_nproc + a]) asym++;
  CHECK(asym == 0);
  free(nb);
  free(gnb);
  MFTLEnsembleDestroy(&ens);
}

// ---------------------------------------------------------------------------------------
// L3 + L7 transport + L1 redistribution under a re-decomposition
// ---------------------------------------------------------------------------------------
static int twin_gid_violations(MFTL_Object* obj, int* rim_unknown)
{
  int bad = 0;
  *rim_unknown = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
  {
    MFTL_HalfEdge* he = &obj->half_edges[i];
    if (he->twin_idx != MFTL_ERROR)
    {
      if (he->twin_gid != obj->half_edges[he->twin_idx].global_id) bad++;
    }
    else if (he->twin_gid == MFTL_ERROR)
    {
      (*rim_unknown)++;
    }
  }
  return bad;
}

// every (he, twin) pair seen on any rank must be symmetric and unique
static int twin_gid_global_asymmetry(MFTL_Object* obj)
{
  int n = obj->n_half_edges;
  int* loc = (int*) malloc((size_t) (2 * (n > 0 ? n : 1)) * sizeof(int));
  for (int i = 0; i < n; ++i)
  {
    loc[2 * i] = obj->half_edges[i].global_id;
    loc[2 * i + 1] = MFTLHeTwinGid(obj, &obj->half_edges[i]);
  }
  int* counts = (int*) malloc((size_t) g_nproc * sizeof(int));
  int* displs = (int*) malloc((size_t) g_nproc * sizeof(int));
  int ns = 2 * n;
  MPI_Allgather(&ns, 1, MPI_INT, counts, 1, MPI_INT, MPI_COMM_WORLD);
  int tot = 0;
  for (int p = 0; p < g_nproc; ++p)
  {
    displs[p] = tot;
    tot += counts[p];
  }
  int* all = (int*) malloc((size_t) (tot > 0 ? tot : 1) * sizeof(int));
  MPI_Allgatherv(loc, ns, MPI_INT, all, counts, displs, MPI_INT, MPI_COMM_WORLD);
  int maxg = 0;
  for (int i = 0; i < tot; i += 2) maxg = all[i] > maxg ? all[i] : maxg;
  int n_err = 0, n_dis = 0, n_asym = 0;
  int* twin = (int*) malloc((size_t) (maxg + 1) * sizeof(int));
  for (int i = 0; i <= maxg; ++i) twin[i] = -2;
  int bad = 0;
  for (int i = 0; i < tot; i += 2)
  {
    int h = all[i], t = all[i + 1];
    if (t < 0 || t > maxg)
    {
      bad++;
      n_err++;
      continue;
    }
    if (twin[h] == -2)
      twin[h] = t;
    else if (twin[h] != t)
    {
      bad++;  // two ranks disagree
      n_dis++;
    }
  }
  for (int h = 0; h <= maxg; ++h)
    if (twin[h] >= 0 && twin[twin[h]] != h)
    {
      bad++;
      n_asym++;
    }
  if (bad && g_rank == 0 && getenv("S15_VERBOSE")) printf("  twin asymmetry: %d unknown, %d disagree, %d non-symmetric\n", n_err, n_dis, n_asym);
  free(loc);
  free(counts);
  free(displs);
  free(all);
  free(twin);
  return bad;
}

static MFTL_Object* make_sphere(MFTL_Ensemble* ens, int gid, int refine)
{
  MFTL_Object* obj = MFTLObjectCreate(gid);
  MFTLEnsembleAddObject(ens, obj);
  double c[3] = {0.05, -0.03, 0.02};
  MFTLGenerateSphere(obj, c, 1.0, refine);
  MFTLComputeTriangleProperties(obj);
  MFTLRedistribute(ens);
  MFTLEnsembleUpdateGlobalCounts(ens);
  return obj;
}

static void test_twin_gid_and_transport(void)
{
  int g[3];
  grid3(g_nproc, g);
  MFTL_Ensemble* ens = make_rect_ensemble(g, 0.3);
  MFTL_Object* obj = make_sphere(ens, 7, 3);
  const int ntri = global_owned_tris(obj);
  CHECK(ntri == 20 * 64);

  // ownership of every stored triangle follows its centre through the rectilinear map
  int wrong = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (obj->triangles[t].owner_proc != expected_owner(&ens->domain, obj->triangles[t].center)) wrong++;
  CHECK(wrong == 0);

  int rim = 0;
  CHECK(twin_gid_violations(obj, &rim) == 0);
  CHECK(rim == 0);  // closed sphere: every rim half-edge knows its remote twin
  CHECK(twin_gid_global_asymmetry(obj) == 0);

  // surface scalar travels byte-exact with the triangle (keyed by gid)
  for (int t = 0; t < obj->n_triangles; ++t) obj->triangles[t].surface_scalar = 1.0e-3 * obj->triangles[t].global_id + 0.125;
  // re-decompose (different cuts, same map) and redistribute
  int g2[3] = {g[2], g[0], g[1]};
  if (g_nproc == 4)
  {
    g2[0] = 1;
    g2[1] = 2;
    g2[2] = 2;
  }
  double* cuts[3];
  for (int d = 0; d < 3; ++d)
  {
    cuts[d] = (double*) malloc((size_t) (g2[d] + 1) * sizeof(double));
    for (int i = 0; i <= g2[d]; ++i) cuts[d][i] = GMIN[d] + (GMAX[d] - GMIN[d]) * (double) i / (double) g2[d] - ((i > 0 && i < g2[d]) ? 0.2 : 0.0);
  }
  bool per[3] = {false, false, false};
  CHECK(MFTLEnsembleSetDecomposition(ens, g2, (const double* const*) cuts, NULL, GMIN, GMAX, per) == MFTL_SUCCESS);
  for (int d = 0; d < 3; ++d) free(cuts[d]);
  MFTLRedistribute(ens);
  CHECK(global_owned_tris(obj) == ntri);
  int bad_scalar = 0;
  for (int t = 0; t < obj->n_triangles; ++t)
    if (obj->triangles[t].surface_scalar != 1.0e-3 * obj->triangles[t].global_id + 0.125) bad_scalar++;
  CHECK(bad_scalar == 0);
  CHECK(twin_gid_violations(obj, &rim) == 0);
  CHECK(rim == 0);
  CHECK(twin_gid_global_asymmetry(obj) == 0);
  MFTLEnsembleDestroy(&ens);
}

// ---------------------------------------------------------------------------------------
// L7: mass conservation through the edge operators and TSUR
// ---------------------------------------------------------------------------------------
static void test_scalar_conservation(void)
{
  int g[3];
  grid3(g_nproc, g);
  // crit 0: MFTL2 criteria; crit 1: MFTL1 edge criteria (JIAO, as multiflow's front tracking)
  for (int run = 0; run < 6; ++run)
  {
    const int algo = run % 3;  // 0: serial bracket kernel, 1: load-balanced shift scheme, 2: native interior-only
    const int crit = run / 3;
    MFTL_Ensemble* ens = make_rect_ensemble(g, 0.2);
    MFTL_Object* obj = make_sphere(ens, 3, 2);
    // jitter owned vertices so split/collapse/flip all fire, then refresh ghosts
    for (int i = 0; i < obj->n_vertices; ++i)
    {
      MFTL_Vertex* v = &obj->vertices[i];
      if (v->is_ghost) continue;
      double f = 1.0 + 0.15 * sin(7.0 * v->pos[0] + 3.0 * v->pos[1]) * cos(5.0 * v->pos[2]);
      for (int k = 0; k < 3; ++k) v->pos[k] = 0.05 * (k == 0) + (v->pos[k] - 0.05 * (k == 0)) * f;
    }
    MFTLUpdateGhostPositions(ens, NULL);
    set_scalar_field(obj);
    const double m0 = global_mass(obj);

    MFTL_RemeshParameters p;
    MFTLInitRemeshParams(&p, MFTL_REMESH_ISOTROPIC, 0.12, 3, MFTL_SMOOTH_VERTEX_BASED, 60.0);
    p.skip_tangential_smoothing = 1;  // application-style vertex moves do not rescale the scalar
    p.skip_volume_recovery = 1;
    p.fixed_iterations = 1;
    p.quiet = 1;
    p.volume_conserving_smoothing = getenv("S15_NOTSUR") ? MFTL_VCSMOOTH_NONE : MFTL_VCSMOOTH_TSUR;
    if (getenv("S15_NOFLIP")) p.skip_flip = 1;
    if (getenv("S15_NOCOLLAPSE")) p.skip_collapse = 1;
    if (getenv("S15_NOSPLIT")) p.skip_split = 1;
    p.load_balance = (algo == 1);
    p.boundary_cleanup = 0;
    if (crit == 1)
    {
      p.mftl1_criteria = 1;
      p.criterion = MFTL_CRITERION_JIAO;
      p.reference_length = 0.4;  // coarsening regime: exercises the MFTL1 collapse tests
      p.max_edge_length = 1.5 * p.reference_length;
      p.min_edge_length = 0.25 * p.reference_length;
      p.max_edge_length_sq = p.max_edge_length * p.max_edge_length;
      p.min_edge_length_sq = p.min_edge_length * p.min_edge_length;
    }
    if (algo == 0)
    {
      MFTL_SerialBracket b;
      MFTLEnsembleBeginSerial(ens, &b);
      MFTLRemeshEnsemble(ens, &p);
      MFTLEnsembleEndSerial(ens, &b, 1);
    }
    else
    {
      MFTLRemeshEnsembleShiftScheme(ens, &p);
    }
    long ops[3] = {p.stat_splits, p.stat_collapses, p.stat_flips}, gops[3];
    MPI_Allreduce(ops, gops, 3, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (g_rank == 0) printf("  algo %d crit %d ops: %ld splits %ld collapses %ld flips\n", algo, crit, gops[0], gops[1], gops[2]);
    if (crit == 0) CHECK(gops[0] > 0);
    if (crit == 1) CHECK(gops[1] > 0);
    const double m1 = global_mass(obj);
    if (fabs(m1 - m0) > 1e-11 * fabs(m0) && g_rank == 0) printf("  algo %d: mass %.17g -> %.17g (rel %.3e)\n", algo, m0, m1, (m1 - m0) / m0);
    CHECK(fabs(m1 - m0) <= 1e-11 * fabs(m0));
    int rim = 0;
    CHECK(twin_gid_violations(obj, &rim) == 0);
    CHECK(twin_gid_global_asymmetry(obj) == 0);
    CHECK(MFTLCheckMeshIntegrity(obj) == MFTL_SUCCESS);
    MFTLEnsembleDestroy(&ens);
  }
}

// ---------------------------------------------------------------------------------------
// L2 + L5
// ---------------------------------------------------------------------------------------
static void test_point_query_and_vertex_ops(void)
{
  int g[3];
  grid3(g_nproc, g);
  MFTL_Ensemble* ens = make_rect_ensemble(g, 0.25);
  MFTL_Object* obj = make_sphere(ens, 5, 3);

  const int n = 64;
  double* pts = (double*) malloc((size_t) 3 * n * sizeof(double));
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 3; ++k) pts[3 * i + k] = -1.3 + 2.6 * (double) ((i * 37 + k * 11) % 64) / 63.0;
  int* many = (int*) malloc((size_t) n * sizeof(int));
  MFTL_PointQuery* q = MFTLPointQueryCreate(obj);
  const int builds = MFTLDebugPointQueryBVHBuilds();
  MFTLPointQueryInsideMany(q, n, pts, many);
  int mismatch = 0, inside = 0;
  for (int i = 0; i < n; ++i)
  {
    int single = MFTLPointQueryInside(q, &pts[3 * i]);
    int reference = MFTLPointInsideObject(obj, &pts[3 * i]);
    if (single != many[i] || reference != many[i]) mismatch++;
    inside += (many[i] == 1);
  }
  CHECK(mismatch == 0);
  CHECK(inside > 0 && inside < n);
  CHECK(MFTLDebugPointQueryBVHBuilds() == builds + n);  // only the reference calls rebuilt
  MFTLPointQueryDestroy(&q);
  CHECK(q == NULL);
  free(pts);
  free(many);

  // per-vertex Laplace-Beltrami == object-level pass; mean curvature ~ 2/R
  MFTLComputeTriangleProperties(obj);
  MFTLComputeVertexCurvatures(obj);
  MFTLComputeVertexNormals(obj);
  int bad = 0, done = 0;
  for (int i = 0; i < obj->n_vertices; ++i)
  {
    MFTL_Vertex* v = &obj->vertices[i];
    if (v->is_ghost) continue;
    double H = v->meancurvature, K = v->gaussiancurvature, nrm[3] = {v->normal[0], v->normal[1], v->normal[2]};
    if (MFTLComputeVertexCurvatureAt(obj, i) != MFTL_SUCCESS || MFTLComputeVertexNormalAt(obj, i) != MFTL_SUCCESS)
    {
      bad++;
      continue;
    }
    if (fabs(v->meancurvature - H) > 1e-12 * H || fabs(v->gaussiancurvature - K) > 1e-9 * fabs(K)) bad++;
    double dn = fabs(nrm[0] - v->normal[0]) + fabs(nrm[1] - v->normal[1]) + fabs(nrm[2] - v->normal[2]);
    if (dn > 1e-12) bad++;
    if (fabs(v->meancurvature - 2.0) > 0.05) bad++;
    done++;
  }
  CHECK(bad == 0);
  CHECK(done > 0 || obj->n_vertices == 0);
  MFTLEnsembleDestroy(&ens);
}

// ---------------------------------------------------------------------------------------
// soup builder + native mesh round trip + L4
// ---------------------------------------------------------------------------------------
static double volume_of(MFTL_Object* obj)
{
  MFTLComputeTriangleProperties(obj);
  return MFTLObjectVolume(obj);
}

static void test_soup_io_transfer(void)
{
  int g[3];
  grid3(g_nproc, g);
  MFTL_Ensemble* ens = make_rect_ensemble(g, 0.1);

  // reference: a serial sphere on every rank (PETSC_COMM_SELF style), used as the soup
  MFTL_Ensemble* self = MFTLEnsembleCreate(MPI_COMM_SELF, (int[3]) {1, 1, 1}, (double*) GMIN, (double*) GMAX, (bool[3]) {false, false, false}, 1);
  MFTL_Object* ref = MFTLObjectCreate(1);
  MFTLEnsembleAddObject(self, ref);
  double c[3] = {0.1, 0.0, -0.1};
  MFTLGenerateSphere(ref, c, 0.8, 2);
  MFTLComputeTriangleProperties(ref);
  const double vref = MFTLObjectVolume(ref);

  // each rank contributes an overlapping share of the triangles (duplicates across ranks)
  int nT = 0, nV = ref->n_vertices;
  int* tg = (int*) malloc((size_t) ref->n_triangles * sizeof(int));
  int (*tv)[3] = malloc((size_t) ref->n_triangles * sizeof(*tv));
  int* vg = (int*) malloc((size_t) nV * sizeof(int));
  double (*vp)[3] = malloc((size_t) nV * sizeof(*vp));
  for (int i = 0; i < nV; ++i)
  {
    vg[i] = 1000 + ref->vertices[i].global_id;  // arbitrary (non-contiguous) global ids
    memcpy(vp[i], ref->vertices[i].pos, sizeof(vp[i]));
  }
  for (int t = 0; t < ref->n_triangles; ++t)
  {
    if ((t % g_nproc) != g_rank && ((t + 1) % g_nproc) != g_rank) continue;
    MFTL_Vertex* vv[3];
    MFTLGetTriangleVertices(ref, &ref->triangles[t], vv);
    tg[nT] = 50000 + 3 * ref->triangles[t].global_id;
    for (int k = 0; k < 3; ++k) tv[nT][k] = 1000 + vv[k]->global_id;
    nT++;
  }
  MFTL_Object* soup = MFTLObjectCreate(2);
  MFTLEnsembleAddObject(ens, soup);
  CHECK(MFTLBuildFromSoup(soup, nV, vg, (const double (*)[3]) vp, NULL, nT, tg, (const int (*)[3]) tv, NULL) == MFTL_SUCCESS);
  int rim = 0;
  CHECK(twin_gid_violations(soup, &rim) == 0);
  CHECK(rim == 0);
  CHECK(twin_gid_global_asymmetry(soup) == 0);  // identical half-edge ids for duplicated triangles
  // a second build on the same (non-empty) object fails collectively
  CHECK(MFTLBuildFromSoup(soup, nV, vg, (const double (*)[3]) vp, NULL, nT, tg, (const int (*)[3]) tv, NULL) == MFTL_ERROR);
  MFTLComputeTriangleProperties(soup);
  MFTLRedistribute(ens);
  CHECK(global_owned_tris(soup) == ref->n_triangles);
  CHECK(fabs(volume_of(soup) - vref) < 1e-12 * vref);
  CHECK(MFTLCheckMeshIntegrity(soup) == MFTL_SUCCESS);

  // native mesh round trip (MFTL1 format) with the surface scalar
  set_scalar_field(soup);
  const double m0 = global_mass(soup);
  char path[256];
  snprintf(path, sizeof(path), "stage15_native_%d.mesh", (int) getpid());
  MPI_Bcast(path, (int) sizeof(path), MPI_CHAR, 0, MPI_COMM_WORLD);
  MFTL_SerialBracket b;
  MFTLEnsembleBeginSerial(ens, &b);
  int wrc = (g_rank == 0) ? MFTLWriteNativeMesh(soup, path) : MFTL_SUCCESS;
  MFTLEnsembleEndSerial(ens, &b, 1);
  CHECK(wrc == MFTL_SUCCESS);
  MFTL_Object* back = MFTLObjectCreate(9);
  MFTLEnsembleAddObject(ens, back);
  CHECK(MFTLReadNativeMesh(back, g_rank == 0 ? path : NULL) == MFTL_SUCCESS);
  MFTLRedistribute(ens);
  CHECK(global_owned_tris(back) == ref->n_triangles);
  CHECK(fabs(volume_of(back) - vref) < 1e-12 * vref);  // orientation preserved
  CHECK(fabs(global_mass(back) - m0) < 1e-13 * fabs(m0));
  if (g_rank == 0) unlink(path);

  // L4: transfer `back` into a second ensemble with the same decomposition
  MFTL_Ensemble* ens2 = make_rect_ensemble(g, 0.1);
  CHECK(ens->num_objects == 2 && ens->objects[1] == back);
  CHECK(MFTLEnsembleTransferObject(ens2, ens, back) == MFTL_SUCCESS);
  CHECK(ens->num_objects == 1 && ens->objects[0] == soup);
  CHECK(ens2->num_objects == 1 && back->domain == &ens2->domain);
  CHECK(MFTLEnsembleRemoveObject(ens, back) == MFTL_ERROR);
  MFTLRedistribute(ens2);
  CHECK(global_owned_tris(back) == ref->n_triangles);

  free(tg);
  free(tv);
  free(vg);
  free(vp);
  MFTLEnsembleDestroy(&ens2);
  MFTLEnsembleDestroy(&ens);
  MFTLEnsembleDestroy(&self);
}

// L6: distance ghosting reaches every rank within the distance, with complete topology
static void test_send_to_all(void)
{
  int g[3];
  grid3(g_nproc, g);
  MFTL_Ensemble* ens = make_rect_ensemble(g, 0.3);
  MFTL_Object* obj = make_sphere(ens, 11, 2);
  const int nt_before = obj->n_triangles;
  CHECK(MFTLObjectSendLocalTrianglesToAll(obj, 10.0) == MFTL_SUCCESS);  // distance > box: everything everywhere
  CHECK(obj->n_triangles == 320);
  CHECK(obj->n_triangles >= nt_before);
  int rim = 0;
  CHECK(twin_gid_violations(obj, &rim) == 0);
  CHECK(rim == 0);
  int open_he = 0;
  for (int i = 0; i < obj->n_half_edges; ++i)
    if (obj->half_edges[i].twin_idx == MFTL_ERROR) open_he++;
  CHECK(open_he == 0);  // whole closed surface stored on every rank
  CHECK(global_owned_tris(obj) == 320);
  MFTLRedistribute(ens);  // ghosts are dropped again by the next redistribution
  CHECK(global_owned_tris(obj) == 320);
  MFTLEnsembleDestroy(&ens);
}

int main(int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nproc);

  if (g_rank == 0) printf("== test_rectilinear_ownership\n");
  fflush(stdout);
  test_rectilinear_ownership();
  if (g_rank == 0) printf("== test_twin_gid_and_transport\n");
  fflush(stdout);
  test_twin_gid_and_transport();
  if (g_rank == 0) printf("== test_scalar_conservation\n");
  fflush(stdout);
  test_scalar_conservation();
  if (g_rank == 0) printf("== test_point_query_and_vertex_ops\n");
  fflush(stdout);
  test_point_query_and_vertex_ops();
  if (g_rank == 0) printf("== test_send_to_all\n");
  test_send_to_all();
  if (g_rank == 0) printf("== test_soup_io_transfer\n");
  fflush(stdout);
  test_soup_io_transfer();

  if (g_rank == 0) printf("Stage 15 library tests: %d checks, %d failed (np=%d)\n", g_checks, g_failed, g_nproc);
  MPI_Finalize();
  return g_failed ? 1 : 0;
}
