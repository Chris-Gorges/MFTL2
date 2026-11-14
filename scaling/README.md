# MFTL scaling on Slurm

Periodic many-sphere advection with ghost synchronization, redistribution,
triangle geometry and vertex normals — and, separately, the same with the
remesher in the loop ([Remeshing scaling](#remeshing-scaling)). See
[QUICKSTART.md](QUICKSTART.md) for step-by-step build and submission
instructions, [SCALING_PLAN.md](SCALING_PLAN.md) for memory estimates and
interpretation, and [Measured results](#measured-results) for the numbers the
top-level README quotes.

**Each launcher uses one problem size. Only the rank count is swept.**

| Setting | Strong | Weak |
|---|---:|---:|
| Spheres | 110,592 total | 1,000 per rank |
| Refinement | 3 | 3 |
| Triangles | 141,557,760 total | 1,280,000 per rank |
| Default rank counts | 36, 72, 144, 288 | same |
| Ranks per node | 36 throughout | 36 throughout |
| Job limit | 1 hour | 1 hour |
| Default repeats | 3 × 200 steps inside one job/point | same |
| Initialization | batches of 512 new spheres | same |

At the default maximum of 8 nodes, the weak problem has 288,000 spheres and
368,640,000 triangles. The two launchers submit 8 benchmark jobs total, plus
collection jobs. `MAX_NODES=30` also enables the 16- and 30-node candidates.

## Run

```bash
module purge
module load os mvapich2
python3 build_all.py --comm-profile
scaling/launch_strong.sh --dry-run
scaling/launch_weak.sh --dry-run
scaling/launch_strong.sh
scaling/launch_weak.sh
```

Configure `scaling/site.sh` for the site's modules, MPI launcher,
partition/account and binding. Use the one-/two-node smoke test in QUICKSTART
before the production campaign. Launchers require a current Release build with
`MFTL_COMM_PROFILE=ON`.

All launcher parameters are environment overrides. `MAX_NODES=4` limits a run to
1, 2 and 4 nodes; `NODES="1 3 6" MAX_NODES=6` selects custom points. Candidate
nodes above the cap are recorded as skipped, and excluded from dependencies
and missing-result checks. The default candidates are `1 2 4 8 16 30`.
Ranks are derived as nodes × `RANKS_PER_NODE`; neither the cap nor the node list
changes the workload. Explicit `RANKS="36 72"` overrides `NODES` when needed.

`MEMORY_PER_NODE=default` (or empty) omits `--mem`; `0` requests all configured
memory, and a size such as `100G` requests a fixed amount. Default omission
matches the site's working template. Check the actual allocation's memory for
large workloads; exclusive CPUs do not imply all node RAM is allocated.

`BUILD_DIR` selects another build. `EXTRA_ARGS` and `SBATCH_EXTRA` are
whitespace-delimited argument lists, not shell commands or quoted expressions.

## Measured results

The two default campaigns of [Run](#run), as last run on 2026-10-19. These are
the numbers quoted in the top-level README; they have not been re-measured since,
and the library has changed after the commit below.

| | |
|---|---|
| Machine | Slurm cluster; nodes of 2 × Intel Xeon Gold 6240 (2 × 18 cores, 2.6 GHz), 36 ranks per node pinned one per core, 1–8 nodes |
| Compiler | GCC 11; Release (`-O3 -DNDEBUG`), no `-march`, `MFTL_COMM_PROFILE=ON` |
| MPI | MVAPICH2 2.3.7 |
| Library | commit `ac631b7` (2026-10-16); `src/`, `include/` and `CMakeLists.txt` unmodified (the sweep's `snapshot/source.diff` is empty) |
| Timing | one job per point: 5 warm-up steps, then 3 repeats of 200 steps; ms per step of the fastest repeat, and of the median one |

"comm" is the exchange and synchronisation share of the step (redistribution
buckets plus ghost exchanges); "redist" is the `MFTLRedistribute` phase.

**Strong scaling**: 110,592 spheres × 1,280 triangles = 141.6 M triangles in a
48³ periodic box.

| P | nodes | grid | triangles/rank | ms/step | median | speedup | efficiency | redist ms | comm |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 36 | 1 | 4×3×3 | 3,932,160 | 393.1 | 393.3 | 1.00 | 1.00 | 88.0 | 2.7 % |
| 72 | 2 | 6×4×3 | 1,966,080 | 211.5 | 211.8 | 1.86 | 0.93 | 53.0 | 6.6 % |
| 144 | 4 | 6×6×4 | 983,040 | 110.3 | 112.7 | 3.56 | 0.89 | 29.5 | 9.2 % |
| 288 | 8 | 8×6×6 | 491,520 | 59.9 | 66.6 | 6.56 | 0.82 | 18.1 | 15.2 % |

**Weak scaling**: 1,000 spheres (1.28 M triangles) in a 10³ box per rank.

| P | nodes | grid | triangles | ms/step | median | efficiency | redist ms | comm |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 36 | 1 | 4×3×3 | 46.1 M | 145.8 | 146.2 | 1.00 | 37.1 | 3.9 % |
| 72 | 2 | 6×4×3 | 92.2 M | 142.2 | 142.3 | 1.03 | 37.1 | 9.5 % |
| 144 | 4 | 6×6×4 | 184.3 M | 157.2 | 158.0 | 0.93 | 44.1 | 15.3 % |
| 288 | 8 | 8×6×6 | 368.6 M | 161.6 | 162.6 | 0.90 | 45.2 | 19.3 % |

Efficiency is against one full node (P = 36): strong T(36)·36 / (T(P)·P), weak
T(36) / T(P). Every point passed the run's own checks: triangle and vertex
counts exact and area conserved to 1e-15, no orphaned ghosts, no global
fallback route. The repeats agree to within 3 % except strong P = 288, where
they spread by 12 %; the median column shows what that costs. The remeshing
campaigns ([Remeshing scaling](#remeshing-scaling)) have no cluster results yet.

## Initialization, timestep and runtime

Initialization generates a temporary batch on rank 0, scatters only that new
batch, appends the received local objects, and frees its communication workspace.
Previously distributed mesh stays in place. About 108 MiB of new mesh is staged
at a time, plus the rank's accumulated local mesh and communication buffers.
This removes the need for all global mesh to coexist on rank 0.

The timed step is:

1. Advect owned vertices and apply periodic wrapping.
2. Synchronize ghost positions.
3. Compute triangle geometry, including ghosts.
4. Redistribute.
5. Recompute triangle geometry after redistribution.
6. Compute vertex normals and synchronize ghost normals.

Setup is reported separately. Five warm-up steps precede three timed repeats
of 200 steps. Uniform flow advances the lattice by (3,2,1) body spacings per
trajectory. Keep `STEPS` fixed for comparisons because it determines motion per
step; `CYCLES` extends timing at unchanged speed.

`TIME_LIMIT=01:00:00` is the Slurm allocation cap. `MAX_SECONDS=3000` is a soft
program budget that includes setup and checks between complete timed repeats.
It can omit later repeats, preserving completed measurements and recording the
actual count and `budget_stop`. It cannot predict or interrupt the first repeat;
a job that reaches the Slurm limit is excluded from valid results.
Use `MAX_SECONDS=0` to disable only the soft budget.

## Results and profiling

Each invocation writes `results/scaling/<mode>_<date>_<unique>/`, also linked
as `<mode>_latest`:

| File | Contents |
|---|---|
| `summary.txt` | tables over P followed by every full report |
| `samples.tsv` | successful jobs' timing distribution, repeat counts, budget flags and setup metrics |
| `points.tsv` | expected points, submission status and Slurm job IDs |
| `sweep.txt` | workload, commit and site settings |
| `snapshot/` | executable, configuration, driver sources, build cache and source diff |
| `reports/` | individual reports and failure markers |
| `slurm/` | job logs, binding and node information |

A small `afterany` collection job writes the summary. Collect manually at any
time with `scaling/collect.sh <directory>`. Failed/incomplete reports are
excluded. `OUT_DIR`, if specified, must be new; queued jobs use their snapshots
even if the working checkout is rebuilt later.

The profile includes phase min/mean/max, 18 redistribution buckets,
ghost-exchange timers, traffic, owner changes, skipped bodies, allocations,
mesh/ghost imbalance, per-step timing quantiles, initialization and memory.
The main curve uses the fastest complete repeat; all repeat times are retained.
The baseline is one full 36-rank node. Look at repeat spread and budget flags
before interpreting small differences.

Strong scaling tests where adding nodes stops helping this fixed problem.
Weak scaling tests growth at this fixed work per rank. Neither one curve maps
a universal minimum worthwhile problem size; no additional size sweep is
scheduled. See SCALING_PLAN for the precise formulas and profiling limits.

## Remeshing scaling

The same machinery runs a REMESHED front-tracking timestep: advect (RK4),
ghost sync, geometry, `MFTLRemeshEnsemble` with its closing redistribution,
every step. One launcher per curve, `REGIME` picks the remeshing driver:

| Launcher | `REGIME=dispersed` (default problem) | `REGIME=spanning` |
|---|---|---|
| `launch_remesh_strong.sh` | 18³ = 5832 droplets, refine 3, 7.5 M triangles; R/h 0.06 → 0.17 | 2³ = 8 bubbles, refine 7, 2.6 M triangles; R/h 0.6 → 1.8 |
| `launch_remesh_weak.sh` | 27 droplets per rank box (34.6 k triangles/rank), R/h 0.083 | one bubble per 2×2×2 rank boxes on a box corner (41 k triangles/rank), R/h 0.6; **32 ranks/node** (every axis needs an even cut count) |

The flow is a Taylor–Green-type straining cell that travels with a uniform
drift, both multiplied by cos(πt/T): every body is stretched into a prolate
ellipsoid and carried (1.5, 1, 0.5) lattice spacings across the cut planes
and back, and the flow map at T is the identity. So at every P the report can
check what a remesher can lose — volume, distance of every vertex to the exact
sphere, triangle quality, edges outside the band, Euler characteristic — with
no serial reference run. Every body has the same history, and in the weak
drivers every rank box is a copy of every other.

A **spanning** job runs BOTH spanning modes (`MFTL_RemeshSpanningMode`: once,
nested) one after the other on the same problem (`SPANNING_MODES=once` or
`nested` for one); the summary puts them side by side per P. A **dispersed**
strong sweep must end on the SAME digest at every P — the dispersed driver is
the serial remesher to the bit and nothing else in the step depends on the
partition — and the summary says whether it did.

```bash
REGIME=dispersed scaling/launch_remesh_strong.sh --dry-run
REGIME=spanning  scaling/launch_remesh_strong.sh
REGIME=dispersed scaling/launch_remesh_weak.sh
REGIME=spanning  scaling/launch_remesh_weak.sh
```

Overrides: `PER_AXIS`, `BOXES`, `REFINE`, `STEPS` (per trajectory, default
100), `TRAJECTORIES` (timed, default 2), `WARMUP` (default 1), `SIZING`
(uniform | curvature), `PROJECTION` (phong | none), `SPANNING_MODES`,
`EXTRA_ARGS` (e.g. `--strain 0.8 --remesh-every 2 --budget 4`). Results go to
`results/scaling/remesh_<regime>_<strong|weak>_<date>_<unique>/`; `collect.sh`
hands such a directory to `collect_remesh.sh`. The per-step times are maxima
over ranks behind a barrier per step; the remesh call's own split (migrate,
sweep, reference, projection, ids) is in the tables.

## Files and validation

| File | Purpose |
|---|---|
| `site.sh` | site configuration |
| `launch_strong.sh`, `launch_weak.sh` | the two fixed-size campaigns |
| `launch_remesh_strong.sh`, `launch_remesh_weak.sh` | the remeshing campaigns, `REGIME=dispersed\|spanning` |
| `sweep_common.sh` | submission, snapshots and manifest |
| `job_scaling.sh` | one Slurm job per rank count |
| `collect.sh` | summary and sample extraction (a remeshing sweep goes to `collect_remesh.sh`) |
| `collect_remesh.sh` | remeshing summary: per-mode tables, once vs nested, the dispersed digest check |
| `strong_scaling.c`, `weak_scaling.c` | workload placement and size rules |
| `scaling_common.h` | instrumented timestep, validation and reports |
| `scaling_init.h` | bounded-memory batch generation/distribution |
| `remesh_strong_scaling.c`, `remesh_weak_scaling.c`, `remesh_common.h` | the remeshing drivers: problem, flow, remeshed step, accuracy checks, report |
| `test_scaling.py` | launcher/collector and optional MPI regression checks |
| `job.sh` | original site template, kept for reference |

```bash
python3 scaling/test_scaling.py
python3 scaling/test_scaling.py --mpi
```

The MPI checks cover 1, 2, 4 and 8 ranks and compare multiple initialization batch
sizes. Local validation does not validate the cluster's Slurm/MVAPICH setup.
