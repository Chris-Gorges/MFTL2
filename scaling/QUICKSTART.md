# Run the scaling tests on the cluster

The default campaign tests strong and weak scaling for periodic advection of spheres with geometric computations, ghost sync and redistribution, across **four rank counts**. Each mode uses one problem size.

| Mode | Spheres | Triangles |
|---|---:|---:|
| Strong | 110,592 total (48³) | 141,557,760 total |
| Weak | 1,000 per rank (10³) | 1,280,000 per rank; 368,640,000 at 288 ranks |

Both use 36 ranks/node throughout: **36, 72, 144, 288 ranks**
on **1, 2, 4, 8 nodes**. Each job requests at most **one hour**.

## 1. Copy the complete change and configure the site

Edit **`scaling/site.sh`** once:

- Set `SBATCH_EXTRA` for the partition/account, if required.
- Check that `site_modules` matches the modules used to build.
- Check `CORES_PER_NODE=36`, `RANKS_PER_NODE=36`, `MAX_NODES=8`, and a homogeneous node type.
- `MEMORY_PER_NODE=default` (the default) omits `--mem`, matching this site's
  working reference job. `0` explicitly requests all configured memory;
  an integer size such as `100G` requests a fixed amount. Use site-approved
  values. An empty value also omits the flag; no script edits are needed.
- Keep source/results on a filesystem visible from every compute node.

Node counts can be changed for any launch without editing files:

```bash
MAX_NODES=4 scaling/launch_strong.sh --dry-run       # 1, 2, 4 nodes
MAX_NODES=10 scaling/launch_strong.sh --dry-run      # 1, 2, 4, 8 nodes
NODES="1 2 4 8" MAX_NODES=8 scaling/launch_weak.sh --dry-run
```

`NODES` is the candidate list (default `1 2 4 8 16 30`); `MAX_NODES` is a cap,
so setting it to 10 does not add a 10-node measurement. To include that point,
set `NODES="1 2 4 8 10" MAX_NODES=10`. `MAX_NODES=30` restores the earlier
six-point campaign. Skipped points are recorded without being treated as failed
runs or collector dependencies. Changing these controls leaves problem size,
rank density, timing and profiling settings intact. Explicit `RANKS` overrides
`NODES`; clear an exported `RANKS` when switching back to node selection.

## 2. Build and inspect the submission commands

```bash
module purge
module load os mvapich2
python3 build_all.py --comm-profile
scaling/launch_strong.sh --dry-run
scaling/launch_weak.sh --dry-run
```

The build must say `comm profiling : ON`; both executables must exist in
`build/`. Check each printed `sbatch` command: 36 tasks/node and
`--time=01:00:00`. Dry runs submit nothing.

If an earlier build says profiling OFF, run the command above with
`--comm-profile`. This rebuilds the library and executables. Existing smoke jobs
submitted by these launchers use saved executable copies and remain unchanged.
After a successful smoke test and rebuild, new full submissions use the new
binary; no repeat of the smoke test is required just to change profiling.
Check `MFTL_COMM_PROFILE: on` in the new run reports as well.

For a faster incremental rebuild of only the scaling programs:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DENABLE_SANITIZERS=OFF -DBUILD_EXAMPLES=ON -DMFTL_COMM_PROFILE=ON
cmake --build build --target strong_scaling weak_scaling --parallel 4
```

## 3. Check the site's launcher with a short smoke test

```bash
NODES="1 2" MAX_NODES=2 MEMORY_PER_NODE=default SPHERES=512 STEPS=20 REPEAT=1 TIME_LIMIT=00:10:00 MAX_SECONDS=480 scaling/launch_strong.sh
```

Open `results/scaling/strong_latest/summary.txt` after completion. Both rows
must say `ok`; headers must show `optimised+NDEBUG: yes` and profiling on.
Check the Slurm logs for one rank per physical core, one host at P=36 and
two hosts at P=72. Placement mismatches abort.

This small smoke test validates submission, binding and reporting; it does not
establish that the production workload fits or meets its runtime target.

## 4. Submit the real campaign

```bash
scaling/launch_strong.sh
scaling/launch_weak.sh
```

These submit **8 benchmark jobs total**, plus two small collection jobs.
Each benchmark normally runs five warm-up steps and three repeats of 200 steps.
Generation/distribution is reported separately from timestep performance.

Initialization uses batches of 512 new spheres (about 108 MiB of new mesh at
refinement 3), distributing only the current batch. The final local mesh and
MPI workspace need additional memory. Set `INIT_BATCH=100` if smaller batches
are useful at the site; this does not change the physical problem or trajectory.

There are two runtime limits:

- Slurm's `TIME_LIMIT=01:00:00` caps the allocation.
- `MAX_SECONDS=3000` gives the program a 50-minute soft budget, including
  setup. It checks between complete timed repeats and can omit later repeats
  if the previous repeat predicts insufficient time. Such reports retain
  completed measurements and explicitly flag the reduced repeat count.

The program cannot predict the first repeat or interrupt a stalled collective.
An unexpectedly slow initialization/first repeat can still hit Slurm's limit;
a timed-out job is excluded from the timing tables. No cluster runtime is
guaranteed by a laptop pilot. Check the first one-node reports before relying
on the complete campaign.

`JOB_REPEATS=1` keeps this initial campaign short. Three repeats are within
each job, not three separate allocations. Jobs release their nodes when done;
the one-hour request is a maximum. At the limit, the benchmark jobs consume
30 node-hours across both modes, not counting smoke/collection jobs.
Slurm chooses their order and concurrency.

## 5. Read and copy results

`results/scaling/<mode>_<date>_<unique>/` contains:

- `summary.txt`: timing tables and complete individual reports.
- `samples.tsv`: successful measurements, actual/requested repeats, budget
  flags and initialization metrics.
- `points.tsv`: submitted jobs and missing/failed points.
- `snapshot/`: the executable/configuration used by those jobs.
- `reports/` and `slurm/`: reports and logs.

The directory is also linked as `<mode>_latest`. Do not remove it while jobs
are queued/running. For an intermediate summary:

```bash
scaling/collect.sh results/scaling/strong_latest
```

Use `squeue` to watch jobs and `sacct` with the IDs in `points.tsv` to diagnose
TIMEOUT or OUT_OF_MEMORY. A `.failed` marker can also mean the job is still
running. Copy the full result directories back, including logs and snapshots.

Background, memory estimates and interpretation: [SCALING_PLAN.md](SCALING_PLAN.md).

## Submission errors and pending jobs

`CPU count per node can not be satisfied` / `Requested node configuration is
not available` means Slurm rejected the requested resource combination. It does
not by itself prove that 36 ranks/node is wrong. Here, removing `--mem` made the
same CPU request acceptable, which points to the site's memory configuration or
policy; the exact cause requires the partition/node settings.

```bash
sinfo -N -p gauss -o '%N %c %m %t'
scontrol show partition gauss
scontrol show job 4120
squeue -u "$USER" -o '%.18i %.24j %.9P %.2t %.6D %R'
```

Use the actual partition and job ID. `Resources` means the job was accepted and
is waiting for resources. `Dependency` on `mftl-strong-collect` or
`mftl-weak-collect` is expected: the summary job waits for the benchmark jobs.
The wider job-name column avoids confusing the collector with a benchmark.

Omitting `--mem` does not guarantee access to the node's physical 120 GB.
Check `ReqTRES`/`AllocTRES` and memory fields in `scontrol show job`, now also
recorded in the job log. Confirm enough memory for the large workload before
the production campaign; if necessary set a permitted `MEMORY_PER_NODE`.
No automatic retry silently lowers a rejected memory request.

The compiler warnings and their severity are explained in [BUILD_WARNINGS.md](BUILD_WARNINGS.md).
