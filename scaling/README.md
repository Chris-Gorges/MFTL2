# MFTL scaling on Slurm

Periodic many-sphere advection with ghost synchronization, redistribution,
triangle geometry and vertex normals. See [QUICKSTART.md](QUICKSTART.md) for
the colleague's build/submission instructions and [SCALING_PLAN.md](SCALING_PLAN.md)
for memory estimates and interpretation.

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
`MFTL_COMM_PROFILE=ON`. The library workspace-growth fix in this change must be
included in that build.

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
See [BUILD_WARNINGS.md](BUILD_WARNINGS.md) for the GCC 11/MVAPICH build log.

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

## Files and validation

| File | Purpose |
|---|---|
| `site.sh` | site configuration |
| `launch_strong.sh`, `launch_weak.sh` | the two fixed-size campaigns |
| `sweep_common.sh` | submission, snapshots and manifest |
| `job_scaling.sh` | one Slurm job per rank count |
| `collect.sh` | summary and sample extraction |
| `strong_scaling.c`, `weak_scaling.c` | workload placement and size rules |
| `scaling_common.h` | instrumented timestep, validation and reports |
| `scaling_init.h` | bounded-memory batch generation/distribution |
| `test_scaling.py` | launcher/collector and optional MPI regression checks |
| `job.sh` | original site template, kept for reference |

```bash
python3 scaling/test_scaling.py
python3 scaling/test_scaling.py --mpi
```

The MPI checks cover 1, 2, 4 and 8 ranks and compare multiple initialization batch
sizes. Local validation does not validate the cluster's Slurm/MVAPICH setup.
