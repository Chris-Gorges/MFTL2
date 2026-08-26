/*
 * @Author: Christian Gorges
 * @Date: 2026-08-14
 *
 * @brief Many-body periodic advection: the standard per-timestep MFTL loop.
 *
 * This is the reference for what a front-tracking solver actually does with this
 * library every timestep, and the benchmark for how much of that time goes into
 * communication. N spheres are placed in a periodic box, translated rigidly, and
 * redistributed across the process grid as they cross rank boundaries.
 *
 * ── The loop order is the part worth copying ────────────────────────────────
 *
 *   1. move only the vertices THIS rank owns, and wrap them back into the box
 *   2. MFTLUpdateGhostPositions      -- pull the vertices the neighbours moved
 *   3. MFTLComputeTriangleGeometry   -- areas/normals/CENTRES for ALL triangles
 *   4. MFTLRedistribute              -- ownership is decided from those centres
 *   5. MFTLComputeTriangleGeometry   -- restore the geometry redistribution drops
 *
 * Step 2 before step 3 is not optional: a triangle's three vertices can be owned
 * by three different ranks, so its geometry is only complete after the ghosts
 * have been synchronised. Step 3 before step 4 is not optional either --
 * redistribution routes by the CACHED centre, so a stale centre routes to the
 * wrong rank.
 *
 * And step 3 must cover ALL triangles, not only the owned ones. Redistribution
 * reads the cached centre of GHOST triangles as well -- that is how a rank knows
 * where its neighbour's triangle is going, and therefore where to send the
 * triangles adjacent to it. Refreshing only the owned ones makes two ranks
 * disagree about a ghost's destination, and someone's locally owned triangle
 * silently loses an edge neighbour. Nothing crashes; the ghost halo is just
 * short. A Debug build asserts on it inside the redistribution.
 *
 * Note what is NOT in the loop: no MFTLObjectRefreshFrames. Step 3 reads the
 * half-edge structure, and step 4 rebuilds the derived arrays on its way out, so
 * a frame refresh here would recompute something that is about to be recomputed.
 * --quality adds MFTLEnsembleUpdate AFTER step 4, where the frames are current.
 *
 * ── Step 5, and why the benchmark would lie without it ─────────────────────
 *
 * A redistribution that PACKS the area, the normal and the centroid onto the wire
 * ends a timestep with valid triangle geometry. This one deliberately does not:
 * geometry is recomputable from positions, so dropping it is what buys the 3x
 * smaller messages (see include/mftl_communication.h). The object that comes out
 * of MFTLRedistribute has valid topology and valid positions, and
 * `tris.area/normal/center` hold unspecified values.
 *
 * Timing the redistribution alone therefore credits it with a saving it has not
 * finished paying for. Step 5 pays it: one geometry pass over all triangles,
 * restoring the end-of-step state a geometry-carrying wire format would have
 * delivered. It is timed as its own phase so both readings are available --
 * redistribute alone, and redistribute plus the recompute it defers.
 *
 * `--no-post-geom` removes it, which is legitimate for a solver that consumes
 * triangle geometry between steps 3 and 4 (the natural place: that is the
 * configuration the forces belong to) and never looks at it afterwards. Such a
 * solver genuinely owes nothing, because step 3 of the next timestep recomputes
 * from the moved positions regardless.
 *
 * The pass used is MFTLComputeTriangleGeometry rather than MFTLEnsembleUpdate,
 * because restoring what the wire dropped is geometry only: ALL triangles, no
 * quality statistics and no reduction. examples/ensemble_update_benchmark.c
 * measures what the fused alternative would cost instead.
 *
 * ── Usage ──────────────────────────────────────────────────────────────────
 *
 *   mpirun -n P parallel_advection_redistribute_list cx cy cz px py pz [options]
 *
 *     cx cy cz     process grid; cx*cy*cz must equal P
 *     px py pz     periodicity per axis (0/1). A non-periodic axis gets zero
 *                  advection velocity, since nothing here reflects at a wall.
 *
 *     --spheres N  number of bodies              (default 64)
 *     --refine N   subdivision level per body    (default 3, = 1280 triangles)
 *     --steps N    timesteps                     (default 100)
 *     --dt X       timestep                      (default 0.1)
 *     --verify     check conservation every step, exit non-zero on violation
 *     --quality    also run MFTLEnsembleUpdate (quality statistics) per step
 *     --normals    also run MFTLEnsembleUpdateVertexNormals per step
 *     --no-post-geom  drop step 5 (see above)
 *     --vtk N      write the surface every N steps
 *     --vtk-mode M  binary (default) | ascii | shared
 *                   binary/ascii write one .vtu per rank plus a .pvtu index;
 *                   shared writes ONE .vtu collectively with MPI-IO
 *
 *   A run big enough for the numbers to mean anything:
 *     mpirun -n 8 ... 4 2 1 1 1 1 --spheres 1000 --refine 4 --steps 500
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "mftl_core.h"

/* ── Configuration ──────────────────────────────────────────────────────── */

typedef struct
{
  int cuts[3];
  bool periodic[3];
  int n_spheres;
  int refinements;
  int steps;
  double dt;
  double radius;
  bool verify;
  bool quality;
  bool normals;
  bool post_geom;
  bool digest;  ///< Print the partition digest at the end.  ///< Step 5: restore the geometry redistribution drops.
  int vtk_every;  ///< 0 = never.
  MFTL_VtkOptions vtk;
} config;

static int parse_args(int argc, char **argv, int rank, int nproc, config *cfg)
{
  cfg->n_spheres = 64;
  cfg->refinements = 3;
  cfg->steps = 100;
  cfg->dt = 0.1;
  cfg->radius = 0.1;
  cfg->verify = false;
  cfg->quality = false;
  cfg->normals = false;
  cfg->post_geom = true;
  cfg->digest = false;
  cfg->vtk_every = 0;
  cfg->vtk = MFTLVtkDefaults();

  if (argc < 7)
  {
    if (rank == 0)
    {
      fprintf(stderr,
              "Usage: %s <cuts_x> <cuts_y> <cuts_z> <per_x> <per_y> <per_z> [options]\n"
              "  --spheres N   number of bodies            (default 64)\n"
              "  --refine N    subdivision level           (default 3)\n"
              "  --steps N     timesteps                   (default 100)\n"
              "  --dt X        timestep                    (default 0.1)\n"
              "  --radius X    sphere radius               (default 0.1)\n"
              "  --verify      check conservation each step\n"
              "  --quality     also run MFTLEnsembleUpdate each step\n"
              "  --normals     also run MFTLEnsembleUpdateVertexNormals each step\n"
              "  --no-post-geom  skip the geometry pass after redistribution\n"
              "  --digest      print a GID-keyed partition checksum at the end\n"
              "  --vtk N       write the surface every N steps\n"
              "  --vtk-mode M  binary (default) | ascii | shared\n",
              argv[0]);
    }
    return MFTL_ERROR;
  }

  for (int d = 0; d < 3; ++d)
  {
    cfg->cuts[d] = atoi(argv[1 + d]);
    cfg->periodic[d] = (atoi(argv[4 + d]) != 0);
  }

  for (int i = 7; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--verify"))
      cfg->verify = true;
    else if (!strcmp(argv[i], "--quality"))
      cfg->quality = true;
    else if (!strcmp(argv[i], "--normals"))
      cfg->normals = true;
    else if (!strcmp(argv[i], "--digest"))
      cfg->digest = true;
    else if (!strcmp(argv[i], "--no-post-geom"))
      cfg->post_geom = false;
    else if (!strcmp(argv[i], "--vtk") && i + 1 < argc)
      cfg->vtk_every = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--vtk-mode") && i + 1 < argc)
    {
      const char *m = argv[++i];
      if (!strcmp(m, "binary"))
      {
        cfg->vtk.layout = MFTL_IO_FILE_PER_RANK;
        cfg->vtk.encoding = MFTL_VTK_BINARY;
      }
      else if (!strcmp(m, "ascii"))
      {
        cfg->vtk.layout = MFTL_IO_FILE_PER_RANK;
        cfg->vtk.encoding = MFTL_VTK_ASCII;
      }
      else if (!strcmp(m, "shared"))
      {
        cfg->vtk.layout = MFTL_IO_SINGLE_FILE;
        cfg->vtk.encoding = MFTL_VTK_BINARY;
      }
      else
      {
        if (rank == 0) fprintf(stderr, "Unknown --vtk-mode: %s (binary | ascii | shared)\n", m);
        return MFTL_ERROR;
      }
    }
    else if (!strcmp(argv[i], "--spheres") && i + 1 < argc)
      cfg->n_spheres = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--refine") && i + 1 < argc)
      cfg->refinements = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
      cfg->steps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--dt") && i + 1 < argc)
      cfg->dt = atof(argv[++i]);
    else if (!strcmp(argv[i], "--radius") && i + 1 < argc)
      cfg->radius = atof(argv[++i]);
    else
    {
      if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
      return MFTL_ERROR;
    }
  }

  if (cfg->cuts[0] * cfg->cuts[1] * cfg->cuts[2] != nproc)
  {
    if (rank == 0) fprintf(stderr, "Process grid %dx%dx%d does not match %d ranks.\n", cfg->cuts[0], cfg->cuts[1], cfg->cuts[2], nproc);
    return MFTL_ERROR;
  }
  if (cfg->n_spheres < 1 || cfg->refinements < 0 || cfg->steps < 0 || cfg->radius <= 0.0)
  {
    if (rank == 0) fprintf(stderr, "spheres must be >= 1, refine and steps >= 0, radius > 0.\n");
    return MFTL_ERROR;
  }
  return MFTL_SUCCESS;
}

/* ── Conservation ───────────────────────────────────────────────────────── */

/**
 * @brief Rank-local surface area and entity counts, over OWNED entities only.
 *
 * Ghosts are copies; counting them would count their owner's contribution twice.
 * Ownership is an array PREFIX, so the counts are two reads and the area sum is
 * a contiguous walk -- there is no is_ghost test anywhere in here.
 */
static void local_totals(const MFTL_Ensemble *ens, double out[3])
{
  out[0] = out[1] = out[2] = 0.0;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    for (int i = 0; i < obj->tris.n_local; ++i) out[0] += obj->tris.area[i];
    out[1] += (double) obj->tris.n_local;
    out[2] += (double) obj->verts.n_local;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Partition digest
 *
 * Two numbers that identify the partition this run produced, so that a binary
 * built before a change and one built after can be shown to agree rather than
 * assumed to. That is the acceptance criterion for anything that claims to be a
 * pure performance change.
 *
 * Both are keyed by GLOBAL ID, never by pool slot. A slot is an implementation
 * detail that a capacity change or a different growth policy is allowed to move;
 * "which entity does this rank own, where is it, and what is it connected to" is
 * not. The set digest combines entities with ADDITION, so it is invariant to the
 * order they are visited in and compares the partition as a set; the order digest
 * mixes the running value, so it also catches a within-rank permutation. A change
 * that alters the second and not the first has reordered the pool without
 * changing the answer -- worth knowing, and not automatically a defect.
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief One round of a 64-bit avalanche mix (splitmix64's finaliser).
 * @param x Input.
 * @return Well-distributed output; every input bit affects every output bit.
 */
static uint64_t mix64(uint64_t x)
{
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

/** @brief Folds one value into a running hash. */
static uint64_t mix_in(uint64_t h, uint64_t v) { return mix64(h ^ mix64(v)); }

/** @brief Reinterprets a double as its bit pattern, so equality is exact. */
static uint64_t bits_of(double d)
{
  uint64_t u;
  memcpy(&u, &d, sizeof u);
  return u;
}

/**
 * @brief Computes this rank's contribution to the two partition digests.
 *
 * Owned entities carry their identity, their position and their connectivity;
 * ghosts carry identity and owner only, which is what makes the halo MEMBERSHIP
 * part of the comparison without making the halo's storage order part of it.
 *
 * A twin is recorded as the partner's `(triangle global id, corner)` rather than
 * as a half-edge index, for the same reason the whole digest is GID-keyed: the
 * index is a slot and the pair is the topology.
 *
 * @param      ens The ensemble.
 * @param[out] out `{set digest, order digest}`.
 */
static void partition_digest(const MFTL_Ensemble *ens, uint64_t out[2])
{
  uint64_t set_h = 0, ord_h = 0x243f6a8885a308d3ULL;

  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *const obj = ens->objects[o];
    const uint64_t body = (uint64_t) obj->global_id;

    for (int v = 0; v < obj->verts.n; ++v)
    {
      const bool owned = (v < obj->verts.n_local);
      uint64_t h = mix_in(mix_in(body, (uint64_t) obj->verts.rec[v].global_id), owned ? 1u : 2u);
      h = mix_in(h, (uint64_t) obj->verts.rec[v].owner_proc);
      if (owned)
      {
        h = mix_in(h, (uint64_t) obj->verts.rec[v].flags);
        for (int k = 0; k < 3; ++k) h = mix_in(h, bits_of(obj->verts.pos[3 * v + k]));
      }
      set_h += h;
      ord_h = mix_in(ord_h, h);
    }

    for (int t = 0; t < obj->tris.n; ++t)
    {
      const bool owned = (t < obj->tris.n_local);
      uint64_t h = mix_in(mix_in(body, (uint64_t) obj->tris.rec[t].global_id), owned ? 3u : 4u);
      h = mix_in(h, (uint64_t) obj->tris.rec[t].owner_proc);
      if (owned)
      {
        h = mix_in(h, (uint64_t) obj->tris.rec[t].flags);
        for (int k = 0; k < 3; ++k)
        {
          const MFTL_HalfEdge *const he = &obj->tris.he[3 * t + k];
          h = mix_in(h, (uint64_t) obj->verts.rec[he->vertex_idx].global_id);
          h = mix_in(h, (uint64_t) he->flags);

          /* The partner named by identity: its triangle's global id and its
           * corner, or a sentinel when this edge has no partner here. */
          const int tw = he->twin_idx;
          if (tw == MFTL_ERROR) h = mix_in(h, 0xffffffffULL);
          else h = mix_in(h, mix_in((uint64_t) obj->tris.rec[tw / 3].global_id, (uint64_t) (tw % 3)));
        }
      }
      set_h += h;
      ord_h = mix_in(ord_h, h);
    }
  }

  out[0] = set_h;
  out[1] = ord_h;
}

/**
 * @brief Nanoseconds one MPI_Wtime() call costs on this machine.
 *
 * The instrumentation reads the clock once per phase boundary per BODY, so the
 * perturbation scales with the body count and cannot be waved away with "a timer
 * is free". Measuring it here rather than quoting a constant keeps the correction
 * valid across machines and MPI implementations, which differ by an order of
 * magnitude in how MPI_Wtime is implemented.
 *
 * Best of five batches rather than a mean: the quantity wanted is the cost of the
 * call itself, and every disturbance can only add to it.
 *
 * @return Cost of one call, in nanoseconds.
 */
static double measure_clock_cost(void)
{
  const int batch = 20000;
  double best = 1e30;

  for (int r = 0; r < 5; ++r)
  {
    const double t0 = MPI_Wtime();
    double sink = 0.0;
    for (int i = 0; i < batch; ++i) sink += MPI_Wtime();
    const double dt = MPI_Wtime() - t0;

    /* Keep `sink` observable so the loop cannot be optimised away. */
    if (sink == 0.0) best = 0.0;
    if (dt < best) best = dt;
  }
  return 1e9 * best / (double) batch;
}

/**
 * @brief Prints the inside of the redistribution and the ghost sync.
 *
 * The phase table above says WHICH phase is expensive; this one says which part
 * of the expensive phase is. Every field is a per-rank mean, for the reason
 * given at the caller: per-phase maxima land on different ranks and do not sum
 * to the whole.
 *
 * The residue line is the check on the taxonomy. #MFTL_RedistributeStats
 * documents its buckets as disjoint, so `t_total` minus the rest is measurement
 * noise plus whatever has no timer; a residue of more than a few per cent means
 * a phase is missing one.
 *
 * @param rank  This rank.
 * @param nproc Ranks in the communicator.
 * @param steps Timed steps, for the per-step columns.
 */
static void report_internals(int rank, int nproc, int steps)
{
  MFTL_RedistributeStats st;
  MFTLRedistributeGetStats(&st);

  /* Field order is fixed by these two tables and nothing else reads them, so a
   * new counter is one line in each. */
  const char *const tname[] = {"setup",     "owner classification",
                               "mask clear", "destination analysis",
                               "route decision", "pack",
                               "exchange: counts", "exchange: post",
                               "exchange: wait", "arrival grouping",
                               "pool growth", "kept: gather",
                               "kept: place back", "kept: relink half-edges",
                               "arrivals", "twin repair",
                               "derived: topology", "derived: FRAMES",
                               "teardown"};
  double tval[] = {st.t_setup,     st.t_owners,       st.t_masks,        st.t_destinations,     st.t_route,
                   st.t_pack,      st.t_exchange_counts, st.t_exchange_data, st.t_wait,          st.t_group,
                   st.t_reserve,   st.t_kept_gather,  st.t_kept_place,   st.t_relink,           st.t_arrivals,
                   st.t_twins,     st.t_derived_topology, st.t_derived_frames, st.t_teardown};
  const int nt = (int) (sizeof(tval) / sizeof(tval[0]));

  const char *const gname[] = {"collect requests", "exchange: counts", "exchange: requests", "owner id table",
                               "answer",           "exchange: replies", "apply",             "teardown"};
  double gval[] = {st.g_t_collect, st.g_t_counts, st.g_t_requests, st.g_t_table, st.g_t_answer, st.g_t_replies, st.g_t_apply, st.g_t_teardown};
  const int ng = (int) (sizeof(gval) / sizeof(gval[0]));

  double totals[2] = {st.t_total, st.g_t_total};
  long counters[] = {st.n_calls,          st.n_verts_scanned,  st.n_tris_scanned,     st.n_verts_owner_changed,
                     st.n_tris_owner_changed, st.n_verts_kept, st.n_tris_kept,        st.n_verts_received,
                     st.n_tris_received,  st.n_destinations,   st.n_messages,         st.n_bytes_sent,
                     st.n_bytes_recv,     st.n_route_neighbor, st.n_route_global,     st.n_objects_visited,
                     st.n_objects_empty_local, st.n_allocs,    st.n_frees,            st.n_bytes_allocated,
                     st.n_gid_table_bytes, st.n_dedup_overflow, st.n_clock_reads,     st.g_n_ghosts_requested,
                     st.g_n_orphaned,     st.g_n_bytes_allocated};
  const int nc = (int) (sizeof(counters) / sizeof(counters[0]));

  double rt[32], rg[16], rtot[2];
  long rc[32];
  MPI_Reduce(tval, rt, nt, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(gval, rg, ng, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(totals, rtot, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(counters, rc, nc, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank != 0) return;

  const double inv_p = 1.0 / (double) nproc;
  const double per_step = (steps > 0) ? (double) steps : 1.0;
  const double red_total = rtot[0] * inv_p;
  const double gs_total = rtot[1] * inv_p;

  printf("\n  -- inside the redistribution (mean per rank) ------------------------\n");
  printf("  %-26s %9.3f s              (%.3f ms/step)\n", "TOTAL", red_total, 1000.0 * red_total / per_step);
  double acc = 0.0;
  for (int i = 0; i < nt; ++i)
  {
    const double v = rt[i] * inv_p;
    acc += v;
    if (v <= 0.0) continue;
    printf("  %-26s %9.3f s   %5.1f %%   (%.4f ms/step)\n", tname[i], v, (red_total > 0.0) ? 100.0 * v / red_total : 0.0, 1000.0 * v / per_step);
  }
  printf("  %-26s %9.3f s   %5.1f %%\n", "residue (untimed)", red_total - acc, (red_total > 0.0) ? 100.0 * (red_total - acc) / red_total : 0.0);

  printf("  -- inside the ghost sync (mean per rank) ----------------------------\n");
  printf("  %-26s %9.3f s              (%.3f ms/step)\n", "TOTAL", gs_total, 1000.0 * gs_total / per_step);
  acc = 0.0;
  for (int i = 0; i < ng; ++i)
  {
    const double v = rg[i] * inv_p;
    acc += v;
    if (v <= 0.0) continue;
    printf("  %-26s %9.3f s   %5.1f %%   (%.4f ms/step)\n", gname[i], v, (gs_total > 0.0) ? 100.0 * v / gs_total : 0.0, 1000.0 * v / per_step);
  }
  printf("  %-26s %9.3f s   %5.1f %%\n", "residue (untimed)", gs_total - acc, (gs_total > 0.0) ? 100.0 * (gs_total - acc) / gs_total : 0.0);

  /* Counters are SUMS over ranks, not means: "how many vertices did the fleet
   * classify" is the question, and dividing it by the rank count would only
   * re-hide the load imbalance the halo block above reports. */
  const double mb = 1.0 / (1024.0 * 1024.0);
  printf("  -- what it moved (all ranks, per step) ------------------------------\n");
  printf("  vertices classified        %12.0f   of which %.2f %% changed owner\n", (double) rc[1] / per_step,
         (rc[1] > 0) ? 100.0 * (double) rc[3] / (double) rc[1] : 0.0);
  printf("  triangles classified       %12.0f   of which %.2f %% changed owner\n", (double) rc[2] / per_step,
         (rc[2] > 0) ? 100.0 * (double) rc[4] / (double) rc[2] : 0.0);
  printf("  vertices kept / received   %12.0f / %.0f\n", (double) rc[5] / per_step, (double) rc[7] / per_step);
  printf("  triangles kept / received  %12.0f / %.0f\n", (double) rc[6] / per_step, (double) rc[8] / per_step);
  printf("  messages / destinations    %12.0f / %.0f\n", (double) rc[10] / per_step, (double) rc[9] / per_step);
  printf("  wire volume                %12.2f MB/step out, %.2f MB/step in\n", (double) rc[11] * mb / per_step, (double) rc[12] * mb / per_step);
  printf("  route neighbour / global   %12ld / %ld  calls\n", rc[13], rc[14]);
  printf("  bodies visited / empty     %12.0f / %.0f  per step\n", (double) rc[15] / per_step, (double) rc[16] / per_step);
  printf("  allocations                %12.0f /step   %.2f MB/step requested\n", (double) rc[17] / per_step, (double) rc[19] * mb / per_step);
  printf("  of which the id tables     %12.2f MB/step   <- sized from the GLOBAL id range\n", (double) rc[20] * mb / per_step);
  printf("  ghost sync requests        %12.0f /step   %.0f orphaned\n", (double) rc[23] / per_step, (double) rc[24]);
  if (rc[21] > 0) printf("  dedup budget exhausted     %12ld times\n", rc[21]);

  /* The instrument charging itself. MEASURED, not assumed: one MPI_Wtime is tens
   * of nanoseconds and the per-body buckets are entered once per BODY per call,
   * so at a thousand droplets this is several per cent of the phase and the
   * breakdown above is only honest if it says so. */
  const double clock_ns = measure_clock_cost();
  const double reads_per_rank_step = (double) rc[22] / per_step / (double) nproc;
  printf("  clock reads (instrument)   %12.0f /step   %.1f ns each = %.4f ms/step (%.1f %% of the phase)\n", reads_per_rank_step, clock_ns,
         reads_per_rank_step * clock_ns * 1e-6, (red_total > 0.0) ? 100.0 * reads_per_rank_step * clock_ns * 1e-9 * per_step / red_total : 0.0);
  printf("======================================================================\n");
}

/** @brief Owned and ghost triangle counts, for the halo-fraction report. */
static void local_halo(const MFTL_Ensemble *ens, long out[2])
{
  out[0] = out[1] = 0;
  for (int o = 0; o < ens->num_objects; ++o)
  {
    const MFTL_Object *obj = ens->objects[o];
    out[0] += obj->tris.n_local;
    out[1] += obj->tris.n - obj->tris.n_local;
  }
}

/** @brief Bytes MFTLUpdateGhostPositions has to move for this partition. */
static long local_ghost_bytes(const MFTL_Ensemble *ens)
{
  long n = 0;
  for (int o = 0; o < ens->num_objects; ++o) n += ens->objects[o]->verts.n - ens->objects[o]->verts.n_local;
  return n * (long) sizeof(MFTL_GhostUpdatePayload);
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank = 0, nproc = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);

  config cfg;
  if (parse_args(argc, argv, rank, nproc, &cfg) != MFTL_SUCCESS)
  {
    MPI_Finalize();
    return 1;
  }

  /* Required once per process before anything communicates mesh entities. */
  MFTLRegisterMpiTypes();

  const double global_min[3] = {-4.0, -2.0, -2.0};
  const double global_max[3] = {4.0, 2.0, 2.0};
  const double radius = cfg.radius;

  MFTL_Ensemble *ens = MFTLEnsembleCreate(MPI_COMM_WORLD, cfg.cuts, (double *) global_min, (double *) global_max, cfg.periodic, cfg.n_spheres);
  if (!ens)
  {
    fprintf(stderr, "Rank %d: MFTLEnsembleCreate failed.\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rank == 0)
  {
    printf("%d ranks in a %dx%dx%d grid | periodic %d%d%d | %d spheres of radius %g, refine %d (%d triangles each) | %d steps\n", nproc, cfg.cuts[0],
           cfg.cuts[1], cfg.cuts[2], (int) cfg.periodic[0], (int) cfg.periodic[1], (int) cfg.periodic[2], cfg.n_spheres, cfg.radius, cfg.refinements,
           20 << (2 * cfg.refinements), cfg.steps);
    printf("wire format: %d B per vertex, %d B per triangle complex, %d B per ghost update\n", (int) sizeof(MFTL_PackedVertex),
           (int) sizeof(MFTL_PackedTriangleComplex), (int) sizeof(MFTL_GhostUpdatePayload));
  }

  /* Fixed seed on every rank: the centres must agree, since MFTLGenerateSphere is
   * collective and only rank 0 keeps the mesh. */
  srand(42);

  for (int o = 0; o < cfg.n_spheres; ++o)
  {
    MFTL_Object *obj = MFTLObjectCreate(o);
    MFTLEnsembleAddObject(ens, obj);

    /* Kept clear of the non-periodic faces; a periodic axis may straddle. */
    double centre[3];
    for (int d = 0; d < 3; ++d)
    {
      const double margin = cfg.periodic[d] ? 0.0 : 2.0 * radius;
      const double lo = global_min[d] + margin;
      const double hi = global_max[d] - margin;
      centre[d] = lo + ((double) rand() / (double) RAND_MAX) * (hi - lo);
    }

    MFTLGenerateSphere(obj, centre, radius, cfg.refinements);

    /* Ownership is decided from the cached centre, and a freshly generated mesh
     * has none. MFTLGenerateSphere already wrapped the positions. */
    MFTLComputeTriangleGeometry(obj, obj->tris.n);
  }

  /* Scatter: everything is on rank 0 after generation. */
  MFTLRedistribute(ens);

  /* A non-periodic axis has no wall model here, so nothing may drift out of it. */
  const double velocity[3] = {cfg.periodic[0] ? 1.0 : 0.0, cfg.periodic[1] ? 0.37 : 0.0, cfg.periodic[2] ? 0.61 : 0.0};

  double reference[3] = {0.0, 0.0, 0.0};
  if (cfg.verify)
  {
    /* The pools were just rebuilt, so the areas are not valid yet -- one geometry
     * pass before the reference is taken, exactly as step 3 does in the loop. */
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);

    double local[3];
    local_totals(ens, local);
    MPI_Allreduce(local, reference, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) printf("reference: area %.12f, %d triangles, %d vertices\n", reference[0], (int) reference[1], (int) reference[2]);
  }

  double t_advect = 0.0, t_ghost = 0.0, t_geom = 0.0, t_redist = 0.0, t_post = 0.0, t_extra = 0.0, t_vtk = 0.0;
  long ghost_bytes = 0;
  long vtk_writes = 0;
  int failures = 0;

  /* After the initial scatter, so its first-touch faults and its one-off
   * all-to-all do not land in the steady-state figures. */
  MFTLRedistributeStatsReset();

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_loop_start = MPI_Wtime();

  for (int step = 0; step < cfg.steps; ++step)
  {
    /* 1. Move what we own, and only what we own. */
    double t0 = MPI_Wtime();
    for (int o = 0; o < ens->num_objects; ++o)
    {
      MFTL_Object *obj = ens->objects[o];
      MFTL_FOREACH_OWNED_VERTEX(obj, pos[0] += velocity[0] * cfg.dt; pos[1] += velocity[1] * cfg.dt; pos[2] += velocity[2] * cfg.dt;
                                MFTLApplyPeriodicity(&ens->domain, pos););
    }
    t_advect += MPI_Wtime() - t0;

    /* 2. Pull in the vertices our neighbours moved. */
    ghost_bytes += local_ghost_bytes(ens);
    t0 = MPI_Wtime();
    MFTLUpdateGhostPositions(ens);
    t_ghost += MPI_Wtime() - t0;

    /* 3. Every triangle now has current vertices: refresh its geometry. ALL of
     *    them, ghosts included -- see the note in the header comment. */
    t0 = MPI_Wtime();
    for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
    t_geom += MPI_Wtime() - t0;

    if (cfg.verify)
    {
      /* Rigid translation conserves surface area exactly, and redistribution
       * conserves entities exactly. Either one drifting means a triangle
       * straddling a periodic seam is being measured across the box, or an
       * entity was lost or duplicated in the exchange. */
      double local[3], total[3];
      local_totals(ens, local);
      MPI_Allreduce(local, total, 3, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      const double rel = fabs(total[0] - reference[0]) / reference[0];
      if (rel > 1e-9 || total[1] != reference[1] || total[2] != reference[2])
      {
        if (rank == 0)
        {
          fprintf(stderr, "step %d: area %.12f (rel dev %.3e), %d triangles, %d vertices -- expected %.12f / %d / %d\n", step, total[0], rel, (int) total[1],
                  (int) total[2], reference[0], (int) reference[1], (int) reference[2]);
        }
        failures++;
      }
    }

    /* 4. Re-decide ownership from the fresh centres. */
    t0 = MPI_Wtime();
    MFTLRedistribute(ens);
    t_redist += MPI_Wtime() - t0;

    /* 5. Restore what the redistribution deliberately did not send. The object
     *    now has valid topology and valid positions; its areas, normals and
     *    centres hold unspecified values until this runs. This is the phase
     *    that makes the per-step total describe a complete end state. */
    if (cfg.post_geom)
    {
      t0 = MPI_Wtime();
      for (int o = 0; o < ens->num_objects; ++o) MFTLComputeTriangleGeometry(ens->objects[o], ens->objects[o]->tris.n);
      t_post += MPI_Wtime() - t0;
    }

    /* Optional consumers of the rebuilt derived arrays. Both go AFTER the
     * redistribution: it leaves the frames current, so neither needs a refresh
     * of its own, and both would otherwise operate on the old partition. */
    if (cfg.quality || cfg.normals)
    {
      t0 = MPI_Wtime();
      if (cfg.quality) MFTLEnsembleUpdate(ens);
      /* The ensemble form, not a loop over MFTLComputeVertexNormals: that one
       * leaves every GHOST vertex holding a partial sum -- correctly normalised
       * and quietly wrong. This adds one ghost exchange and makes the field
       * exact for the whole pool. */
      if (cfg.normals) MFTLEnsembleUpdateVertexNormals(ens);
      t_extra += MPI_Wtime() - t0;
    }

    /* The dump goes last, so what lands on disk is the state the step ended in.
     * It is timed separately and excluded from the phase percentages: writing
     * the surface is not part of the timestep, it is what you do to look at it. */
    if (cfg.vtk_every > 0 && step % cfg.vtk_every == 0)
    {
      char prefix[256];
      snprintf(prefix, sizeof prefix, "advection_%04d", step);

      t0 = MPI_Wtime();
      if (MFTLWriteVtu(ens, prefix, &cfg.vtk) != MFTL_SUCCESS)
      {
        if (rank == 0) fprintf(stderr, "step %d: MFTLWriteVtu failed for %s\n", step, prefix);
        failures++;
      }
      t_vtk += MPI_Wtime() - t0;
      vtk_writes++;
    }

    if (rank == 0 && cfg.steps >= 20 && step % (cfg.steps / 10) == 0) printf("  step %d/%d\n", step, cfg.steps);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  const double t_total = MPI_Wtime() - t_loop_start;

  /* The phase breakdown is a MEAN over ranks, not a max. Taking the max per phase
   * would pick a different rank for each one, and the parts would then not add up
   * to the whole -- the slowest rank in the exchange is usually not the slowest
   * one in the rebuild. */
  const double mine[8] = {t_total, t_advect, t_ghost, t_geom, t_redist, t_extra, t_post, t_vtk};
  double sum[8];
  MPI_Reduce(mine, sum, 8, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* Partition quality, which is what "imbalance" has to mean here: the wall clock
   * is bounded by barriers and is therefore identical on every rank by
   * construction, so max/mean of the TIME would always read 1.00. Max/mean of the
   * owned triangle count is the thing that actually varies. */
  long halo[2];
  local_halo(ens, halo);
  long halo_sum[2] = {0, 0}, owned_max = 0, gbytes = 0;
  MPI_Reduce(halo, halo_sum, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&halo[0], &owned_max, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&ghost_bytes, &gbytes, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0 && cfg.steps > 0)
  {
    for (int i = 0; i < 8; ++i) sum[i] /= (double) nproc;
    const double mean_total = sum[0];
    const double rest = mean_total - sum[1] - sum[2] - sum[3] - sum[4] - sum[5] - sum[6] - sum[7];
    const double owned_mean = (double) halo_sum[0] / (double) nproc;
    const double per_step = (double) cfg.steps;

    printf("\n=== advection benchmark, %d ranks, %d steps ===\n", nproc, cfg.steps);
    printf("  wall clock                 %8.3f s\n", mean_total);
    printf("  owned triangles            %8ld   (%.0f per rank on average)\n", halo_sum[0], owned_mean);
    printf("  ghost triangles            %8ld   (%.1f %% halo overhead)\n", halo_sum[1], 100.0 * (double) halo_sum[1] / (double) halo_sum[0]);
    printf("  partition imbalance        %8.2f   (busiest rank / average)\n", (owned_mean > 0.0) ? (double) owned_max / owned_mean : 1.0);
    printf("  ghost sync volume          %8.2f MB/step (all ranks)\n", (double) gbytes / (1024.0 * 1024.0) / per_step);
    printf("  -- mean per rank ---------------------------\n");
    printf("  redistribute        %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[4], 100.0 * sum[4] / mean_total, 1000.0 * sum[4] / per_step);
    printf("  ghost sync          %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[2], 100.0 * sum[2] / mean_total, 1000.0 * sum[2] / per_step);
    printf("  triangle geometry   %8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[3], 100.0 * sum[3] / mean_total, 1000.0 * sum[3] / per_step);
    if (cfg.post_geom)
      printf("  post-redist geometry%8.3f s   %5.1f %%   (%.3f ms/step)\n", sum[6], 100.0 * sum[6] / mean_total, 1000.0 * sum[6] / per_step);
    printf("  advection           %8.3f s   %5.1f %%\n", sum[1], 100.0 * sum[1] / mean_total);
    if (cfg.quality || cfg.normals) printf("  quality + normals   %8.3f s   %5.1f %%\n", sum[5], 100.0 * sum[5] / mean_total);
    if (vtk_writes > 0) printf("  vtk output          %8.3f s   %5.1f %%   (%.1f ms per dump, %ld dumps)\n", sum[7], 100.0 * sum[7] / mean_total,
                               1000.0 * sum[7] / (double) vtk_writes, vtk_writes);
    printf("  verify + rest       %8.3f s   %5.1f %%\n", rest, 100.0 * rest / mean_total);
    printf("  redistribute cost   %8.1f ns per owned triangle per step\n", 1e9 * sum[4] / (per_step * (owned_mean > 0.0 ? owned_mean : 1.0)));

    /* The honest per-step total: the redistribution plus the geometry pass it
     * defers, i.e. the cost of reaching a state with valid triangle geometry. */
    if (cfg.post_geom)
    {
      printf("  redistribute + post %8.3f s              (%.3f ms/step)  <- the honest total\n", sum[4] + sum[6],
             1000.0 * (sum[4] + sum[6]) / per_step);
    }
    printf("==============================================\n");
  }

  report_internals(rank, nproc, cfg.steps);

  if (cfg.digest)
  {
    uint64_t mine[2], all[2];
    partition_digest(ens, mine);

    /* Summed, so the result does not depend on which rank holds what -- only on
     * WHAT the partition is. Wrapping is defined for unsigned and is exactly the
     * order-independent combination wanted. */
    MPI_Reduce(mine, all, 2, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("partition digest: set %016llx  order %016llx\n", (unsigned long long) all[0], (unsigned long long) all[1]);
  }


  int total_failures = 0;
  MPI_Allreduce(&failures, &total_failures, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (cfg.verify && rank == 0)
  {
    printf("%s\n", total_failures == 0 ? "verification: OK" : "verification: FAILED");
  }

  MFTLEnsembleDestroy(&ens);
  MFTLFreeMpiTypes();
  MPI_Finalize();
  return (total_failures > 0) ? 1 : 0;
}
