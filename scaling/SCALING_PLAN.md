# One larger workload per scaling mode

The current launchers sweep **rank count only**. They do not loop over problem
sizes. Both default to full 36-core nodes, with P = 36, 72, 144, 288.
`MAX_NODES=8` caps submissions; `NODES` selects candidate node counts. The
candidate list also includes 16 and 30 nodes, enabled by raising the cap.
Explicit `RANKS` takes precedence over `NODES`.

| Mode | Fixed parameter | Global triangles | Average owned triangles/rank |
|---|---|---:|---:|
| Strong | 110,592 spheres, refinement 3 | 141,557,760 | 3,932,160 at P=36; 491,520 at P=288 |
| Weak | 1,000 spheres/rank, refinement 3 | 46,080,000 at P=36; 368,640,000 at P=288 | 1,280,000 |

The optional 30-node point has 131,072 triangles/rank in strong mode and
1,382,400,000 global triangles in weak mode. It is not submitted by default.

There are 1,280 triangles and 642 vertices per sphere. Body spacing is 1 in
both modes: the strong domain has edge 48; each weak rank box has edge 10.
Keep body density, refinement and motion fixed across a curve.

The strong size is 3.375 times the earlier 32,768-sphere default. Weak work
per rank is 15.625 times the earlier 64-sphere default. These are substantial
workloads without using most of a 120 GB node's RAM for mesh storage alone.
Actual memory and runtime on the cluster still need measurement.

## Why 36 ranks per node throughout

The first point is one fully occupied node; later points add equally occupied
nodes. There is no 32-to-36-rank density change, no partially occupied default
node, and no automatic repacking to make an oversized rank count fit.

The 48³ strong lattice divides the default Cartesian grids through P=576.
At P=1080 the process-grid axes do not all divide 48, so some imbalance is
expected. Keep the problem fixed and report that imbalance. Weak mode repeats
the same 10³ lattice inside every rank box.

A complete 26-neighbour stencil requires at least three cuts on **every** axis.
The graph degree in the reports explains changes from neighbour aliasing;
rank count alone is not sufficient.

## Bounded-memory initialization

Rank 0 still generates spheres, but only **512 new spheres per batch**:

1. Generate and publish a temporary ensemble containing just the new batch.
2. Redistribute that batch using the final domain's existing communicators.
3. Restore global body IDs and append each rank's received pieces to its final
   local directory.
4. Release the batch directory and communication workspace.

Earlier batches are never redistributed during initialization. Redistributing
the entire growing ensemble every 100 spheres would repeatedly traverse mesh
already placed and make startup unnecessarily expensive. Releasing each batch's
send buffers also avoids retaining a high-water allocation for every destination
visited over the whole initialization.

Batch size is configurable with `INIT_BATCH` / `--init-batch`. The 512 default
keeps new mesh near 108 MiB while amortizing collective/allocation overhead;
100 is valid too. Strong uses 216 batches; the largest weak point uses 2,110.
Setup is outside timestep timing, but its total duration, generation/distribution
cost, batch count, batch pool size and memory high-water marks are reported.

`--verify-init` optionally checks global body/entity IDs, exact coordinates and
triangle connectivity before/after scattering with a distributed checksum.
Conservation of counts and area is always checked. The regression suite checks
equivalent initialization with batch sizes 1, 7, 100 and 10,000.

Batched startup exposed a library bug: growing the redistribution workspace
for new arrivals discarded existing body pointers. The accompanying library
fix preserves that list; the regression forces growth while old and incoming
bodies coexist. The cluster build needs this fix as well as the new drivers.

## Memory budget

The current compact pools use about 140 bytes/triangle and 64 bytes/vertex.
The following are **mesh-only estimates**, before spare capacity, ghost copies,
objects, MPI and redistribution workspace:

| Allocation | Mesh storage |
|---|---:|
| One 512-sphere staging batch on rank 0 | 107.6 MiB |
| Strong, total across all ranks | 22.69 GiB |
| Strong, per node at the P=36 baseline | 22.69 GiB |
| Weak, per rank | 0.205 GiB |
| Weak, per full 36-rank node at every point | 7.39 GiB |
| Weak, global at P=1080 | 221.57 GiB, distributed across 30 nodes |

The weak global mesh no longer has to fit on rank 0. Peak root memory includes
its final local partition plus one batch and the batch's send/receive workspace.
The strong one-node baseline is the tightest expected memory case. These sizes
leave considerable room below 120 GB for capacity growth and communication;
they are not a substitute for checking the first site's RSS measurements.

A local four-rank pilot (4,096 spheres, refinement 3, 200 timed steps, one
repeat) measured the following. RSS covers the full process lifetime; the
node sum adds individual rank high-water marks.

| Initialization batch | Setup | Maximum rank RSS | Node RSS peak-sum |
|---|---:|---:|---:|
| 4,096 (all at once) | 0.601 s | 1,334 MiB | 2,916 MiB |
| 512 | 0.284 s | 567 MiB | 1,999 MiB |

Both runs conserved area and entity counts. This supports the batching change;
it is not a prediction of cluster runtime or MPI memory usage.

`--max-gen-mb` now guards **one staging batch**, not the global mesh.
It is not a total-RSS limit. The drivers also reject total triangle counts beyond
the library's current int-sized global counter: 1.3824 billion at the largest
weak point remains below INT_MAX. This does not justify arbitrary larger
refinement/count/decomposition settings.

The node peak-sum adds ranks' individual RSS high-water marks. Those peaks need
not coincide and shared pages can be counted repeatedly; it is an upper bound
rather than a simultaneous node measurement. Keep Slurm/cgroup memory data
where available. `MEMORY_PER_NODE=default` omits the explicit memory request,
matching the site's accepted submission. It does not guarantee enough memory
for the large runs: inspect the actual allocation in the job log.
`MEMORY_PER_NODE=0` explicitly requests all configured memory, or use a
site-approved size. Physical RAM and scheduler-allocated RAM can differ.

## Runtime and allocation cost

Each point requests one hour and normally measures 3 × 200 steps after 5 warm-up
steps. The 3,000-second soft budget includes initialization; between repeats,
the program uses 1.5 times the previous repeat's wall time plus a 30-second
reporting allowance to decide whether another complete repeat fits. It never
shortens a timed trajectory or adds a per-step budget collective.

If the soft budget omits repeats, `budget_stop=1` and actual/requested repeat
counts appear in the report and `samples.tsv`; compare sample counts when
interpreting minima. If setup has exhausted the budget before any repeat, the
run fails. The first repeat is not predicted, so a sufficiently slow job can
still time out without a report. Slurm provides the hard cap.

Four points per mode and one job per point keep the default campaign to 8 jobs.
The sum of node counts is 15 per mode, so the one-hour limits correspond to at
most 30 benchmark node-hours (excluding smoke/collection jobs). With
`MAX_NODES=30`, all six candidates correspond to 12 jobs and 126 node-hours.
Actual use should be lower; scheduling/queue time is separate. Lower caps keep
the workload fixed, but naturally provide less evidence about large-P scaling.

Keep `STEPS=200`: it sets the motion per step. Each trajectory translates the
lattice by (3,2,1) body spacings. `CYCLES` repeats trajectories at unchanged
speed if a later campaign needs longer timing windows. Repeats advance the
mesh rather than resetting it. A full unjittered lattice reproduces geometry
up to body labels and floating-point effects; drift/jitter are different
workloads and should have separate curves.

## What the curves say about the parallelization threshold

Let T(N,P) be complete timestep time for N global owned triangles on P ranks.

- Strong fixes N. With the current baseline P0=36, speedup is
  T(N,36)/T(N,P), and efficiency is that speedup times 36/P.
  A flat or rising time identifies where **this size** stops benefiting from
  additional nodes. The minimum-time point and the minimum P×T point can differ.
- Weak fixes N/P. Efficiency is T(36n,36)/T(Pn,P). A flat curve shows how well
  **this amount of work per rank** scales as the machine and problem grow.

A strong curve already exposes loss of efficiency as local work shrinks.
Weak scaling holds that local-work level constant, so one weak curve does not
find the smallest worthwhile problem. Several sizes would be needed to map a
general crossover, but **that is not part of this campaign**.

Without P=1, these runs measure scaling beyond one full node; they do not say
whether MPI beats a serial execution of a tiny problem. Report fastest P,
efficiency, triangles/rank, P×T and nodes×T for the measured workload. Do not
treat equal WORK and COMM timing as a universal threshold: those buckets mix
local communication work and waiting, and the useful limit comes from total
time improvement and resource cost.

## Profiling interpretation

The wall clock is the scaling metric. Phase timings, 18 redistribution buckets,
ghost-exchange timings, message/byte traffic, owner changes, skipped bodies,
structural imbalance and memory identify where to investigate next.

- MPI waits and collective times include load imbalance and network delays.
  Equal owned-triangle counts need not imply equal ghost/body work.
- Summing maxima of accumulated phase times does not reconstruct a
  step-by-step critical path when the slowest rank changes.
- Per-step quantiles mix deterministic cut crossings and system noise.
- O(P) workspace, collective costs and network contention can grow in weak
  scaling even when local mesh work stays fixed.
- Clock-read overhead is only an estimate. A later profile-ON/OFF comparison
  measures the full instrumentation effect.
- `--sync-phases`, `--no-skip`, and drift are diagnostic workload variants;
  do not mix them into the default curve.
- If MPI dominates, profile/trace selected points. If geometry or local rebuild
  dominates, use CPU sampling and memory-bandwidth/cache counters there.

Reports retain every complete repeat, and `samples.tsv` retains job minima,
medians and maxima. One job per point limits evidence about job-to-job noise;
that is a deliberate cost choice for this initial campaign.

This benchmark covers advection, ghost synchronization, redistribution and
geometry/normals, not remeshing, collision search or a coupled flow solver.
