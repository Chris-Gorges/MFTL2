/*
 * @Author: Christian Gorges
 * @Date: 2026-09-24
 *
 * @brief Open surfaces end to end: generate, inspect the boundary, distribute,
 *        advect, write.
 *
 * The counterpart to examples/sphere_creation_serial.c and
 * examples/parallel_advection_redistribute_list.c for surfaces WITH a boundary
 * -- a film, a membrane, an inflow patch. Everything a closed droplet does works
 * here too; what is new is that some half-edges have no twin at all, and this
 * example shows how to work with them rather than around them:
 *
 *   1. build a flat patch and a curved cap (both open, both disks)
 *   2. report the combinatorics, including the Euler characteristic, which is 1
 *      for a disk rather than the 2 of a closed sphere
 *   3. walk the boundary as a LOOP with MFTLHeNextBoundary, and measure its
 *      perimeter -- the operation a boundary condition or a rim remesh needs
 *   4. run the standard per-timestep loop (move -> sync ghosts -> recompute ->
 *      redistribute) and verify, PER BODY, that surface area, entity counts,
 *      the rim and the Euler characteristic are conserved while the surfaces
 *      cross rank and periodic boundaries
 *   5. write a .pvtu
 *
 * The one rule worth carrying away: after redistribution `twin_idx ==
 * MFTL_ERROR` no longer means "the surface ends here" -- it also happens at the
 * outer rim of the ghost halo. The topological question is answered by
 * MFTLHeIsMeshBoundary (MFTL_FLAG_BOUNDARY), which is written once at
 * generation time and travels with the entity from then on.
 *
 * ── Why the generators live in this file ───────────────────────────────────
 *
 * The library has no open-surface generator (CLAUDE.md lists it as absent), and
 * it does not need one to produce these meshes: MFTLBuildIndexedMesh owns every
 * half-edge convention -- implicit indexing, dense ids, twins, the boundary
 * marking -- so a caller supplies positions and a winding-consistent triangle
 * list and gets a fully described open surface back. Each shape below is a
 * thirty-line function. The collective half every generator shares (broadcast
 * the id counters, reduce the counts) is MFTLPublishGeneratedBodies, taken ONCE
 * for both bodies.
 *
 * No remeshing here, on purpose: this is the example that establishes that an
 * open surface survives the plain per-timestep machinery. The remeshing ones
 * build on it -- rayleigh_taylor_open_surface.c, travelling_waves_open_surface.c
 * and liquid_jet_open_surface.c.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P open_surface_creation [cx cy cz] [options]
 *
 *     cx cy cz     process grid, cx*cy*cz == P. Defaults to (P,1,1).
 *     --steps N           timesteps of rigid translation   (default 20)
 *     --no-output         do not write open_surfaces.pvtu
 *     --require-rim-cut   also fail if at np > 1 no rim triangle ever met a
 *                         partition boundary (what the CTest passes)
 *
 * Exits non-zero if any conservation or soundness check failed on any step.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

#define DT 0.05

static const double GLOBAL_MIN[3] = {-2.0, -2.0, -2.0};
static const double GLOBAL_MAX[3] = {2.0, 2.0, 2.0};

/* Tilted on purpose: a patch parallel to a cut plane would only ever be split
 * two ways, which is not a representative partition. */
static const double PATCH_ORIGIN[3] = {-1.4, -1.4, -1.1};
static const double PATCH_U[3] = {2.8, 0.3, 1.5};
static const double PATCH_V[3] = {0.2, 2.8, 0.9};
#define PATCH_NU 16
#define PATCH_NV 12

static const double CAP_CENTER[3] = {0.3, -0.4, 0.2};
static const double CAP_AXIS[3] = {0.3, -1.0, 0.8};
#define CAP_RADIUS 1.0
#define CAP_ANGLE (0.45 * MFTL_PI)
#define CAP_RINGS 6
#define CAP_SECTORS 18

/* The two bodies, by the dense id MFTLEnsembleAddObject hands out. */
#define BODY_PATCH 0
#define BODY_CAP 1
#define N_BODIES 2

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. The two open shapes, as indexed triangle lists
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief The (nu+1) x (nv+1) grid spanned by edge_u and edge_v, 2*nu*nv triangles.
 *
 * Both children of a cell are wound so that (v1-v0) x (v2-v0) points along
 * edge_u x edge_v:
 *
 *     v01 ---- v11        A = (v00, v10, v11)   (u) x (u+v) = u x v
 *      |  \  B  |         B = (v00, v11, v01)   (u+v) x (v) = u x v
 *      |  A  \  |
 *     v00 ---- v10
 *
 * A single fixed diagonal: the orientation argument is the same for every cell,
 * and the rim stays a plain rectangle of exactly 2*(nu+nv) edges.
 */
static int build_patch(MFTL_Object *obj, const double origin[3], const double edge_u[3], const double edge_v[3], int nu, int nv)
{
  const int nx = nu + 1;
  const int n_verts = nx * (nv + 1);
  const int n_tris = 2 * nu * nv;

  double *pos = (double *) malloc(3 * (size_t) n_verts * sizeof(double));
  int *tris = (int *) malloc(3 * (size_t) n_tris * sizeof(int));
  if (!pos || !tris)
  {
    free(pos);
    free(tris);
    return MFTL_ERROR;
  }

  for (int j = 0; j <= nv; ++j)
  {
    for (int i = 0; i <= nu; ++i)
    {
      const double tu = (double) i / (double) nu;
      const double tv = (double) j / (double) nv;
      for (int d = 0; d < 3; ++d) pos[3 * (j * nx + i) + d] = origin[d] + tu * edge_u[d] + tv * edge_v[d];
    }
  }

  int w = 0;
  for (int j = 0; j < nv; ++j)
  {
    for (int i = 0; i < nu; ++i)
    {
      const int v00 = j * nx + i;
      const int v10 = v00 + 1;
      const int v01 = v00 + nx;
      const int v11 = v01 + 1;

      tris[w++] = v00;
      tris[w++] = v10;
      tris[w++] = v11;

      tris[w++] = v00;
      tris[w++] = v11;
      tris[w++] = v01;
    }
  }

  const int rc = MFTLBuildIndexedMesh(obj, pos, n_verts, tris, n_tris);
  free(pos);
  free(tris);
  return rc;
}

/**
 * @brief A spherical cap about @p axis: a pole plus n_rings rings of n_sectors.
 *
 * Built in a right-handed frame (u, w, a) with u x w == a. A triangle listed as
 * (p, p + dtheta, p + dtheta + dphi) then has (v1-v0) x (v2-v0) along
 * theta_hat x phi_hat == r_hat, i.e. OUTWARD, and the hub fan around the pole is
 * the same statement in the limit theta -> 0.
 */
static int build_cap(MFTL_Object *obj, const double center[3], const double axis[3], double radius, double polar_angle, int n_rings, int n_sectors)
{
  const int n_verts = 1 + n_rings * n_sectors;
  const int n_tris = n_sectors * (2 * n_rings - 1);

  /* The frame: `a` along the axis, `u` from the world axis least aligned with it
   * (so the cross product is never near-degenerate), `w = a x u`. */
  double a[3] = {axis[0], axis[1], axis[2]};
  MFTLNormalizeVector(a);

  double helper[3] = {0.0, 0.0, 0.0};
  const double ax = fabs(a[0]), ay = fabs(a[1]), az = fabs(a[2]);
  helper[(ax <= ay && ax <= az) ? 0 : (ay <= az) ? 1 : 2] = 1.0;

  double u[3], w[3];
  MFTLVectorCrossVector(u, helper, a);
  MFTLNormalizeVector(u);
  MFTLVectorCrossVector(w, a, u);

  double *pos = (double *) malloc(3 * (size_t) n_verts * sizeof(double));
  int *tris = (int *) malloc(3 * (size_t) n_tris * sizeof(int));
  if (!pos || !tris)
  {
    free(pos);
    free(tris);
    return MFTL_ERROR;
  }

  for (int d = 0; d < 3; ++d) pos[d] = center[d] + radius * a[d];

  for (int r = 1; r <= n_rings; ++r)
  {
    const double theta = polar_angle * (double) r / (double) n_rings;
    for (int s = 0; s < n_sectors; ++s)
    {
      const double phi = MFTL_TWO_PI * (double) s / (double) n_sectors;
      const int v = 1 + (r - 1) * n_sectors + s;
      for (int d = 0; d < 3; ++d)
      {
        pos[3 * v + d] = center[d] + radius * (sin(theta) * cos(phi) * u[d] + sin(theta) * sin(phi) * w[d] + cos(theta) * a[d]);
      }
    }
  }

  int t = 0;
  for (int s = 0; s < n_sectors; ++s)
  {
    tris[3 * t + 0] = 0;
    tris[3 * t + 1] = 1 + s;
    tris[3 * t + 2] = 1 + (s + 1) % n_sectors;
    ++t;
  }
  for (int r = 1; r < n_rings; ++r)
  {
    const int inner = 1 + (r - 1) * n_sectors;
    const int outer = 1 + r * n_sectors;
    for (int s = 0; s < n_sectors; ++s)
    {
      const int s1 = (s + 1) % n_sectors;

      tris[3 * t + 0] = inner + s;
      tris[3 * t + 1] = outer + s;
      tris[3 * t + 2] = outer + s1;
      ++t;

      tris[3 * t + 0] = inner + s;
      tris[3 * t + 1] = outer + s1;
      tris[3 * t + 2] = inner + s1;
      ++t;
    }
  }

  const int rc = MFTLBuildIndexedMesh(obj, pos, n_verts, tris, n_tris);
  free(pos);
  free(tris);
  return rc;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. Boundary inspection -- on the COMPLETE mesh, i.e. rank 0 before the scatter
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief Walks every boundary loop, reporting loops and perimeter.
 *
 * A walk rather than a filter because a boundary edge has no twin, so there is
 * no "outer face" half-edge cycle to follow. MFTLHeNextBoundary reconstructs the
 * successor by rotating around the shared vertex until the fan runs out, which
 * turns an unordered set of rim half-edges into the ORDERED loops a boundary
 * condition is applied along.
 *
 * Only meaningful on a complete mesh: on a partition the fan also runs out at the
 * halo rim, and the walk would follow it.
 *
 * @return The number of boundary half-edges.
 */
static int inspect_boundary(const MFTL_Object *obj, int *n_loops, double *perimeter)
{
  *n_loops = 0;
  *perimeter = 0.0;

  const int n_he = MFTLNumHalfEdges(obj);
  if (n_he < 1) return 0;

  char *seen = (char *) calloc((size_t) n_he, 1);
  if (!seen) return 0;

  int n_boundary = 0;
  for (int h0 = 0; h0 < n_he; ++h0)
  {
    /* The flag, not twin_idx: on a distributed mesh the two differ, and only
     * the flag means "the surface ends here". */
    if (!MFTLHeIsMeshBoundary(obj, h0) || seen[h0]) continue;

    (*n_loops)++;
    int h = h0;
    do
    {
      seen[h] = 1;
      *perimeter += MFTLGetEdgeLength(obj, h);
      n_boundary++;
      h = MFTLHeNextBoundary(obj, h);
    } while (h != MFTL_ERROR && h != h0);
  }

  free(seen);
  return n_boundary;
}

static void report(const MFTL_Object *obj, const char *name)
{
  int n_loops = 0;
  double perimeter = 0.0;
  const int B = inspect_boundary(obj, &n_loops, &perimeter);

  const int V = obj->verts.n;
  const int F = obj->tris.n;
  /* A boundary edge carries ONE half-edge, an interior edge two, so the edge
   * count is (3F + B)/2 -- not 3F/2, which only holds when closed. */
  const int E = (3 * F + B) / 2;

  double area = 0.0;
  for (int t = 0; t < F; ++t) area += MFTLGetTriangleArea(obj, t);

  int n_boundary_verts = 0;
  for (int v = 0; v < V; ++v) n_boundary_verts += MFTLVertexIsMeshBoundary(obj, v) ? 1 : 0;

  printf("  %-18s V=%5d  E=%5d  F=%5d  |  V-E+F=%2d  |  boundary: %d half-edges in %d loop(s), %d vertices\n", name, V, E, F, V - E + F, B, n_loops,
         n_boundary_verts);
  printf("  %-18s area = %.10f, perimeter = %.10f\n", "", area, perimeter);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. Per-body conservation over the distributed mesh
 * ═══════════════════════════════════════════════════════════════════════════ */

/** @brief What is summed per body over OWNED entities, and reduced. */
enum
{
  TOT_AREA,
  TOT_TRIS,
  TOT_VERTS,
  TOT_RIM,  ///< owned boundary half-edges
  TOT_RIM_CUT,  ///< owned boundary half-edges in triangles another rank holds a copy of
  TOT_HOLES,  ///< owned twin-less half-edges that are NOT mesh boundary
  TOT_N
};

/**
 * @brief Per-body totals, indexed by BODY ID so that two ranks holding different
 *        bodies still add the same quantities together.
 *
 * Indexed by `global_id` rather than by list slot because it crosses ranks: a
 * rank holds a sparse, gid-sorted subset of the ensemble after the first
 * redistribution, so slot 0 is the patch on one rank and the cap on the next.
 * A `[n_bodies_global]` buffer is the right shape for two bodies in an example
 * and the wrong one for a million droplets -- MFTLEnsembleReduceBodies is the
 * scalable form.
 *
 * TOT_HOLES is the open-surface form of "the ghost halo is complete": an owned
 * triangle may lack a twin only where MFTL_FLAG_BOUNDARY says no neighbouring
 * triangle exists anywhere. Testing `twin_idx` alone would flag every rim
 * triangle; skipping the test at every twin-less edge would hide a genuinely
 * short halo.
 */
static void body_totals(const MFTL_Ensemble *ens, double out[N_BODIES][TOT_N])
{
  double local[N_BODIES][TOT_N];
  memset(local, 0, sizeof local);

  for (int o = 0; o < ens->n_local_objects; ++o)
  {
    const MFTL_Object *obj = ens->local_objects[o];
    double *acc = local[obj->global_id];

    /* The area from positions, not from tris.area: a redistribution leaves the
     * cached geometry undefined by design. */
    for (int t = 0; t < obj->tris.n_local; ++t) acc[TOT_AREA] += MFTLGetTriangleArea(obj, t);

    for (int h = 0; h < MFTLNumLocalHalfEdges(obj); ++h)
    {
      if (MFTLHeIsMeshBoundary(obj, h))
      {
        acc[TOT_RIM] += 1.0;

        /* Replicated: another rank holds a copy, because one of its corners or
         * edge neighbours is a ghost here -- the rule redistribution sends by.
         * This rim piece meets a PARTITION boundary, which is what makes a
         * parallel run say anything about parallel rims. */
        int v[3], nb[3];
        MFTLGetTriangleVertices(obj, MFTLHeTriIdx(h), v);
        MFTLGetTriangleNeighbors(obj, MFTLHeTriIdx(h), nb);
        for (int k = 0; k < 3; ++k)
        {
          if (v[k] >= obj->verts.n_local || nb[k] >= obj->tris.n_local)
          {
            acc[TOT_RIM_CUT] += 1.0;
            break;
          }
        }
      }
      else if (MFTLHeIsLocalBoundary(obj, h))
        acc[TOT_HOLES] += 1.0;
    }

    acc[TOT_TRIS] += (double) obj->tris.n_local;
    acc[TOT_VERTS] += (double) obj->verts.n_local;
  }

  MPI_Allreduce(local, out, N_BODIES * TOT_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/** @brief V - E + F from owned counts, with E = (3F + B)/2. */
static int euler_of(const double tot[TOT_N]) { return (int) (tot[TOT_VERTS] - 0.5 * (3.0 * tot[TOT_TRIS] + tot[TOT_RIM]) + tot[TOT_TRIS]); }

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  int cuts[3] = {nproc, 1, 1};
  int n_steps = 20;
  bool write_output = MFTL_TRUE;
  bool require_rim_cut = MFTL_FALSE;

  int i = 1;
  if (argc >= 4 && argv[1][0] != '-')
  {
    for (int d = 0; d < 3; ++d) cuts[d] = atoi(argv[1 + d]);
    i = 4;
  }
  for (; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      n_steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--no-output"))
      write_output = MFTL_FALSE;
    else if (!strcmp(argv[i], "--require-rim-cut"))
      require_rim_cut = MFTL_TRUE;
    else
    {
      if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
      MPI_Finalize();
      return 1;
    }
  }
  if (cuts[0] * cuts[1] * cuts[2] != nproc)
  {
    if (rank == 0) fprintf(stderr, "Process grid %dx%dx%d does not match %d ranks.\n", cuts[0], cuts[1], cuts[2], nproc);
    MPI_Finalize();
    return 1;
  }

  /* Required once per process before anything communicates mesh entities. */
  MFTLRegisterMpiTypes();

  bool periodic[3] = {MFTL_TRUE, MFTL_TRUE, MFTL_TRUE};
  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cuts, (double *) GLOBAL_MIN, (double *) GLOBAL_MAX, periodic, N_BODIES);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* ── 1. Generate two open surfaces ──────────────────────────────────────
   *
   * Rank 0 builds, everyone else holds an empty object -- the same contract as
   * every library generator -- and ONE MFTLPublishGeneratedBodies settles the id
   * counters and global counts of both bodies. */
  MFTL_Object *patch = MFTLObjectCreate(BODY_PATCH);
  MFTL_Object *cap = MFTLObjectCreate(BODY_CAP);
  MFTLEnsembleAddObject(ens, patch);
  MFTLEnsembleAddObject(ens, cap);

  if (rank == 0)
  {
    if (build_patch(patch, PATCH_ORIGIN, PATCH_U, PATCH_V, PATCH_NU, PATCH_NV) != MFTL_SUCCESS ||
        build_cap(cap, CAP_CENTER, CAP_AXIS, CAP_RADIUS, CAP_ANGLE, CAP_RINGS, CAP_SECTORS) != MFTL_SUCCESS)
    {
      fprintf(stderr, "Building the open surfaces failed.\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* MFTLGetOwnerProc wants wrapped coordinates, and a patch may well be placed
     * across a periodic face -- that is the interesting case, not one to avoid. */
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTL_Object *obj = ens->local_objects[o];
      for (int v = 0; v < obj->verts.n; ++v) MFTLApplyPeriodicity(&ens->domain, MFTLVertexPosMut(obj, v));
    }
  }
  else
  {
    /* Empty, with its (vacuous) derived arrays marked valid, exactly as a
     * library generator leaves a non-root rank. */
    MFTLObjectBuildDerived(patch);
    MFTLObjectBuildDerived(cap);
  }
  MFTLPublishGeneratedBodies(ens);

  /* ── 2. Inspect, while rank 0 still holds the complete meshes ─────────── */
  if (rank == 0)
  {
    printf("=== open surfaces on %d rank(s), grid %dx%dx%d ===\n", nproc, cuts[0], cuts[1], cuts[2]);
    printf("A closed surface has V-E+F = 2; an open one (a disk) has 1.\n\n");
    report(patch, "rectangular patch");
    report(cap, "spherical cap");

    const double exact_patch =
        sqrt(MFTL_SQ(PATCH_U[1] * PATCH_V[2] - PATCH_U[2] * PATCH_V[1]) + MFTL_SQ(PATCH_U[2] * PATCH_V[0] - PATCH_U[0] * PATCH_V[2]) +
             MFTL_SQ(PATCH_U[0] * PATCH_V[1] - PATCH_U[1] * PATCH_V[0]));
    const double exact_cap = MFTL_TWO_PI * CAP_RADIUS * CAP_RADIUS * (1.0 - cos(CAP_ANGLE));
    printf("\n  analytic patch area %.10f (a planar patch is discretised exactly)\n", exact_patch);
    printf("  analytic cap   area %.10f (approached from below under refinement)\n\n", exact_cap);

    MFTLCheckMeshIntegrity(patch);
    MFTLCheckMeshIntegrity(cap);
  }

  /* ── 3. Distribute ──────────────────────────────────────────────────────
   *
   * Ownership is decided from the CACHED triangle centre, which a freshly built
   * mesh does not have yet.
   *
   * `patch` and `cap` are dead from here on. A redistribution destroys the
   * object of a body a rank keeps nothing of and creates one for a body that
   * arrives, so the handle that survives is the body id and every loop below
   * re-reads ens->local_objects. */
  for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);
  MFTLRedistribute(ens);
  patch = cap = MFTL_NULL;

  double reference[N_BODIES][TOT_N];
  body_totals(ens, reference);
  if (rank == 0)
  {
    for (int b = 0; b < N_BODIES; ++b)
    {
      printf("distributed %-6s %5.0f triangles, %5.0f vertices, %4.0f rim half-edges, V-E+F = %d, area %.12f\n", (b == BODY_PATCH) ? "patch:" : "cap:",
             reference[b][TOT_TRIS], reference[b][TOT_VERTS], reference[b][TOT_RIM], euler_of(reference[b]), reference[b][TOT_AREA]);
    }
  }

  /* ── 4. The standard per-timestep loop ───────────────────────────────────
   *
   *   move owned vertices -> sync ghosts -> recompute ALL triangle geometry ->
   *   redistribute
   *
   * Identical to the closed-surface loop; the boundary changes nothing about the
   * ORDER, only about which invariants hold at the rim. */
  const double velocity[3] = {0.9, 0.41, 0.63};
  int failures = 0;
  double rim_cut_max = 0.0;

  for (int step = 0; step < n_steps; ++step)
  {
    for (int o = 0; o < ens->n_local_objects; ++o)
    {
      MFTL_Object *obj = ens->local_objects[o];
      MFTL_FOREACH_OWNED_VERTEX(obj, {
        for (int d = 0; d < 3; ++d) pos[d] += velocity[d] * DT;
        MFTLApplyPeriodicity(&ens->domain, pos);
      });
    }

    MFTLUpdateGhostPositions(ens);

    /* ALL triangles, ghosts included: redistribution routes ghost triangles by
     * their cached centre too. */
    for (int o = 0; o < ens->n_local_objects; ++o) MFTLComputeTriangleGeometry(ens->local_objects[o], ens->local_objects[o]->tris.n);

    MFTLRedistribute(ens);

    double total[N_BODIES][TOT_N];
    body_totals(ens, total);
    for (int b = 0; b < N_BODIES; ++b) rim_cut_max = MFTL_MAX(rim_cut_max, total[b][TOT_RIM_CUT]);

    /* Rigid translation conserves area exactly; redistribution conserves
     * entities exactly and never touches a flag. Any drift means a triangle is
     * being measured across a periodic seam, a ghost is a step behind, or the rim
     * lost a triangle in the destination analysis. */
    for (int b = 0; b < N_BODIES; ++b)
    {
      const double rel = fabs(total[b][TOT_AREA] - reference[b][TOT_AREA]) / reference[b][TOT_AREA];
      const bool ok = rel < 1e-9 && total[b][TOT_TRIS] == reference[b][TOT_TRIS] && total[b][TOT_VERTS] == reference[b][TOT_VERTS] &&
                      total[b][TOT_RIM] == reference[b][TOT_RIM] && total[b][TOT_HOLES] == 0.0 && euler_of(total[b]) == 1;
      if (ok) continue;

      if (rank == 0)
      {
        fprintf(stderr, "step %d, body %d: area rel dev %.3e, %.0f tris, %.0f verts, %.0f rim, %.0f halo holes, V-E+F %d\n", step, b, rel, total[b][TOT_TRIS],
                total[b][TOT_VERTS], total[b][TOT_RIM], total[b][TOT_HOLES], euler_of(total[b]));
      }
      failures++;
    }
  }

  if (rank == 0) printf("%d steps: %s\n", n_steps, failures == 0 ? "area, entities, rim, Euler characteristic and halo conserved" : "FAILED");
  if (rank == 0) printf("rim across ranks: up to %.0f rim half-edges in triangles another rank also holds\n", rim_cut_max);

  /* A parallel run whose rims never met a partition boundary checked nothing
   * about parallel rims; the CTest makes that a failure rather than a pass. */
  if (require_rim_cut && nproc > 1 && rim_cut_max == 0.0)
  {
    if (rank == 0) fprintf(stderr, "VACUOUS: no rim ever met a partition boundary -- choose a grid whose cut planes cross the surfaces\n");
    failures++;
  }

  /* ── 5. Write ───────────────────────────────────────────────────────────── */
  if (write_output)
  {
    MFTLWriteVtu(ens, "open_surfaces", MFTL_NULL);
    if (rank == 0) printf("wrote open_surfaces.pvtu\n");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (failures > 0) ? 1 : 0;
}
