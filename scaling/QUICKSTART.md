# Run the scaling tests on the cluster

The default campaign tests strong and weak scaling for periodic advection of spheres with geometric computations, ghost sync and redistribution, across **six rank counts**.

| Mode | Spheres | Triangles |
|---|---:|---:|
| Strong | 110,592 total (48³) | 141,557,760 total |
| Weak | 1,000 per rank (10³) | 1,280,000 per rank; 1,382,400,000 at 1,080 ranks |

Both use 36 ranks/node throughout: **36, 72, 144, 288, 576, 1,080 ranks**
on **1, 2, 4, 8, 16, 30 nodes**. Each job requests at most **one hour**.

## 1. Copy the complete change and configure the site

Edit **`scaling/site.sh`** once:

- Set `SBATCH_EXTRA` for the partition/account, if required.
- Check that `site_modules` matches the modules used to build.
- Check `CORES_PER_NODE=36`, `MAX_NODES=30`, and a homogeneous node type.
- `MEMORY_PER_NODE=0` requests all Slurm-configured node memory. Use a
  site-approved explicit value if required.
- Keep source/results on a filesystem visible from every compute node.

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

## 3. Check the site's launcher with a short smoke test

```bash
RANKS="36 72" SPHERES=512 STEPS=20 REPEAT=1 TIME_LIMIT=00:10:00 MAX_SECONDS=480 scaling/launch_strong.sh
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

These submit **12 benchmark jobs total**, plus two small collection jobs.
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
126 node-hours across both modes, not counting smoke/collection jobs.
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
