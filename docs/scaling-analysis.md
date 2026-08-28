# Scaling analysis {#mftl_page_scaling}

`docs/performance-analysis.md` §10 and `docs/parallel-remeshing-analysis.md` both
name the same hole in their own words: **nothing in MFTL had been measured above
four ranks on real hardware**, and there was no harness that could do it. Every
other benchmark in `examples/` is a single-shot program that prints one table and
is never run twice at two rank counts by anything automated.

This page is the method for closing that, and the record of what the instrument
has been shown to do. The instrument is
`examples/parallel_advection_scaling.c` plus `run_scaling.py`; the cluster side is
`scaling/job.sh` and `scaling/launch.sh`.

**The result tables below are a SHAKEDOWN on a laptop, not a scaling result.**
They exist to show the harness measures what it claims to. The cluster tables are
deliberately empty until someone runs it on a cluster.

---

# Three axes, and only two of them are scaling laws {#scaling_axes}

| Axis | Held fixed | Answers |
|---|---|---|
| **Strong** | total work; the rank count grows | "How much faster can I make *this* problem?" Amdahl-limited. Ideal: time proportional to 1/P. |
| **Weak** | work **per rank**; ranks and total work grow together | "How much *bigger* a problem can I run?" Limited by how overhead grows with P. Ideal: a flat line. |
| **Granularity** | total triangle count; the **split** between body count and triangles-per-body varies | "Does the same work scale differently as 4096 tiny bodies or 16 large ones?" |

The third one is the question that prompted this page, and it is worth stating
plainly: **granularity is not a scaling law.** It changes the *constant* in both of
the others, so it is swept as a family — one strong and one weak curve per
granularity, plus an iso-triangle ladder at a fixed rank count that isolates it.

It is not cosmetic in MFTL specifically. Many small bodies against few large ones
is the same distinction the remesher already dispatches on
(`MFTLRemeshChooseStrategy`), and it drives the halo fraction, the per-object loop
overhead, and the number of distinct redistribution destinations. Two runs with an
identical triangle count can differ by a large factor in
communication-to-computation ratio — and in the shakedown below they differ by 2×.

## The weak-scaling domain grows with the process grid {#scaling_weak_domain}

In weak mode the domain is the per-rank box times `cuts[]`, so every rank owns a
constant physical volume holding a constant number of bodies. Bodies per rank, the
halo fraction and the neighbour count are then invariant by construction and the
ideal curve is genuinely flat.

The alternative — hold the domain fixed and add bodies — also keeps work per rank
constant, but makes the problem *denser* as P grows. The resulting curve mixes "the
machinery got slower" with "the problem got harder", and nothing in it can separate
the two. It is not offered.

---

# How to reproduce {#scaling_reproduce}

```bash
python3 build_all.py                     # Release; the benchmark is meaningless otherwise

# one point, by hand
mpirun -n 4 ./build/parallel_advection_scaling --mode strong --spheres 256 --refine 3
mpirun -n 4 ./build/parallel_advection_scaling --mode weak --spheres-per-rank 32 --refine 3

# the whole sweep, plots and report
python3 run_scaling.py --preset laptop --all
python3 run_scaling.py --plots-only      # re-render from the store; runs nothing
```

On a cluster, edit the marked site block at the top of `scaling/job.sh` and
`scaling/launch.sh`, then:

```bash
./scaling/launch.sh --dry-run            # check the sbatch lines
./scaling/launch.sh
python3 run_scaling.py --plots-only      # once the jobs have landed
```

Results accumulate in `results/scaling/raw/` as one JSON per run, and **every
invocation re-renders every plot and the report from the whole store**. Adding a
rank count later means running it (or copying its JSON back from the cluster) and
rendering again; nothing needs to be re-run, and a partly failed sweep still
renders.

---

# Six traps, and the design that avoids each {#scaling_traps}

These are the reason to read this page before trusting a curve. Every one of them
produces a plausible-looking number.

## 1. Generation peak vs stranded capacity: two problems, and only one is fixed {#scaling_trap_pools}

`MFTLGenerateSphere` builds the whole sphere on **rank 0**; every other rank gets an empty object.
That is one cost. Growth being a ratchet — `vertex_pool_reserve` and `triangle_pool_reserve` open
with `if (capacity <= p->capacity) return;` — used to be a second one on top of it. They need
separating, because they have different fixes and only one of them has landed.

### The generation peak is real and unfixed

Every body exists at full size on rank 0 *before* the opening `MFTLRedistribute`, so nothing that
runs inside that redistribution can lower the peak. `5120 · 4^refine + 176` bytes per body: 80 MB
for 256 bodies at refine 3, 640 MiB for 512 at refine 4.

Batching — generate K bodies, redistribute, repeat — does **not** fix it, and it is worth recording
why because it looks like it should. It shaves only the transient on top of a base it cannot touch,
while costing O(N²/K) setup work, because `collect_destinations` re-derives the entire halo on every
call and `rebuild_object` re-runs over every entity already placed. Measured at 512 bodies, K = 32:
~8× the setup work for a ~30 % cut in a peak dominated by the base.

What the benchmark does instead: prints the footprint, refuses above `--max-gen-mb`, and times setup
separately. **In weak mode that setup cost grows linearly with P on one rank while the timed loop
does not** — the harness's Amdahl ceiling, not the library's, and excluded from every scaling total.
The real fix is a distributed generator, and it does not exist.

### The stranded capacity was the smaller half, and it is fixed

After the scatter, what rank 0 does with those blocks splits in two:

- **Bodies it keeps nothing of** are released outright — `MFTLObjectSetCounts(0,0,0,0)` plus
  `MFTLObjectReleasePools` in `src/communication/redistribute.c`. This is most of them. Measured at
  256 bodies, refine 3, np = 4: **80 MB generated, 33.5 MB of pool capacity left on rank 0.**
- **Bodies it keeps a sliver of** hold a full-mesh block against a few dozen triangles.
  `MFTLObjectShrinkToFit()` recovers those, armed by `MFTLRedistributeSetShrinkSlack()`.

### What arming it is worth

Capacity is a ratchet, so over a run every rank accumulates, per body, the largest share it ever
held — measure at the END of the loop, not after the scatter. np = 4, 128 bodies, refine 3,
200 steps, `examples/parallel_advection_scaling.c`. Capacity figures are bit-reproducible across
runs; timings are three runs each.

| | `--shrink-slack 0` | `--shrink-slack 4` |
|---|---|---|
| end capacity, max/rank | 22.8 MB | 20.8 MB (−8.8 %) |
| end capacity, total | 73.3 MB | 68.0 MB (−7.2 %) |
| stranded against live | 1.79× | 1.66× |
| redistribute ms/step | 0.906 / 0.931 / 0.960 | 0.941 / 0.960 / 0.962 |
| end RSS, sum over ranks | 213.3–217.7 MB | 210.4–212.6 MB |

**8 % of capacity for about 2 % of the redistribution phase.** That is a trade, so it ships
disabled, like `coarsen_frozen_fraction` and `serial_gather_max_triangles` and unlike
`parallel_coarsen`. Most of the residual 1.66× sits in the 1×–4× band the gate declines on purpose:
it cannot go to 2 without thrashing against the ×2 growth ladder, and `MFTL_SHRINK_SLACK_DEFAULT`
derives that with a worked sequence. Arm it when bodies genuinely shrink — remeshing that coarsens,
or a long run where shares drift far.

### Measure it in RSS, never in a byte counter

Whether `free()` returns pages to the operating system is the allocator's decision. glibc raises its
mmap threshold dynamically as mmapped chunks are freed, up to 32 MiB, after which blocks come from
the heap break and freeing them may not lower RSS at all; macOS madvise-frees large allocations. So
`MFTL_RedistributeStats.n_bytes_reclaimed` is an **upper bound** on the win, and
`current_rss_bytes()` is the measurement. It is deliberately not `peak_rss_bytes()`: `ru_maxrss` is
a high-water mark, cannot fall, and would report "no change" for a change of hundreds of megabytes.

## 2. Counting migration warms the caches for the phase being timed {#scaling_trap_migration}

`MFTLGetOwnerProc` is documented as *"the inner statement of `compute_owners`"*, so
a local pass over `tris.center` reproduces the routing rule exactly and gives an
honest migration count. But that pass streams `tris.center`, `tris.verts`,
`tris.neighbors` and `verts.pos` — precisely the arrays `compute_owners` and
`collect_destinations` are about to read. Run immediately before
`MFTLRedistribute`, it makes the redistribution look faster than it is.

It is therefore behind `--migration-stats`, default **off**, and the banner says so
when it is on. `run_scaling.py` never enables it in a run whose timings are quoted.

**Two different numbers come out of it, and conflating them is the easy mistake.**
*Ownership churn* is the owned triangles whose destination is not this rank.
*Wire volume* is the distinct remote destinations summed over owned triangles — a
redistribution sends each owned triangle to as many as seven places (its centre's
owner, its three corners' owners, its three edge neighbours' owners) because the
halo is re-materialised from scratch every call. On an advecting mesh the churn is
a thin rim and the wire traffic is most of the surface. Measured on a small np = 4
case: **130 triangles/step of churn against 490 of wire traffic.** Only the second
predicts bytes.

## 3. The first redistribute is not a redistribute {#scaling_trap_scatter}

Rank 0 holds every body after generation and no destination is inside its
26-neighbourhood, so the first `MFTLRedistribute` takes the global `MPI_Alltoallv`
path and moves the entire mesh. It is a different operation from a steady-state
redistribution, and it is *excluded* from the timed loop rather than warmed.

`--warmup` defaults to **3** rather than 1 for the weaker version of the same
effect: the step after setup still migrates atypically much, because the advection
has not reached a steady state.

## 4. A timer that ran out of bits looks exactly like a scaling limit {#scaling_trap_timer}

At high rank counts the per-rank advection and geometry phases shrink toward
`MPI_Wtick`, and a strong-scaling curve that flattens for that reason is
indistinguishable from one that flattened physically. Every phase mean below
`100 · MPI_Wtick` per step is flagged in the report with `!` and drawn with a
**hollow marker** in the plots. On the shakedown machine `MPI_Wtick` is 1e-6 s, so
the floor is 0.1 ms per step — which the local phases of a small case sit below,
and the report says so rather than reporting them.

`MPI_WTIME_IS_GLOBAL` is false on stock Open MPI and MPICH, and that is fine: every
quantity here is a difference of two `MPI_Wtime` calls **on the same rank**, so a
constant per-rank offset cancels exactly and only rate drift survives. The one
forbidden operation is comparing timestamps across ranks; `--sync-phases` is the
legal way to ask that question.

## 5. A curve measured in blocks turns machine state into a scaling law {#scaling_trap_blocks}

`docs/performance-analysis.md` §7 records the same binary measuring 9.4 ms and
6.1 ms in two sessions purely from thermal and background state, and states the rule
the whole document follows: interleave the variants, report minima, never quote a
run taken in a block.

Applied to a rank sweep, that means **repeats go round robin over the rank counts**
— `run_scaling.py` cycles P = 1, 2, 4, 8 … R times, and `scaling/launch.sh` submits
the same way rather than queueing all repeats of np = 1 together. Running R
consecutive repeats at each P instead lets a thermal ramp or a node-placement
gradient masquerade as a scaling law.

Within one run the headline is the **mean over timed steps**, because only means add
up to the wall clock — which is what makes the "sum of phase means vs. wall clock"
cross-check possible. The **minimum** is taken over *repeats*, not over steps:
min-over-steps systematically flatters a bursty phase, and redistribute is bursty
(a triangle crosses a cut plane or it does not). The report also carries p50 and
p95 of the per-step redistribute time, which is what separates a rising floor from
a growing tail.

## 6. Random placement puts Poisson noise inside the weak-scaling curve {#scaling_trap_placement}

Uniform random body placement has Poisson fluctuations in bodies-per-rank. Those
fluctuations change with P and with the body count, so on a weak curve they are
indistinguishable from a real overhead trend — the curve measures placement luck.

Bodies therefore sit on a **jittered lattice**, which is stratified sampling:
bodies-per-rank is near-constant by construction. The radius is derived from a
volume fraction rather than fixed, so density is invariant along the iso-triangle
ladder and across P. All randomness is splitmix64 keyed on the body index, so
placement is a pure function of that index and is identical on every rank, at every
rank count, and on every platform — `rand()` is none of those across libcs.

**Acceptance criterion:** a weak run must report a structural imbalance near 1.0 at
every P. If it does not, the lattice is not doing its job and no number from that
sweep means anything.

---

# What is measured {#scaling_measured}

Per phase — advection, ghost sync, geometry, redistribute, post-redistribute
geometry, quality, normals, VTK, and an instrumentation bucket — the harness
reports mean, max, min and coefficient of variation over ranks, plus an imbalance
of max/mean. **For a collective phase that imbalance is a lower bound**, because
the collective has already equalised the finish times; the structural imbalance
(busiest rank's owned triangles over the mean) is printed alongside and is the
honest partition metric. Max over mean of a barrier-bounded wall clock always reads
1.00 and measures nothing.

Four totals are printed, and their disagreement is itself the measurement:

```
wall clock (barriered loop bounds)   <- the scaling number
sum of phase means                   <- the breakdown that adds up
sum of phase maxima                  <- an UPPER BOUND on the critical path
unattributed                         <- wall minus sum of means
```

The gap between the barriered wall clock and the sum of maxima is exactly the
imbalance that is not additive across phases.

Also recorded: halo fraction, bodies touched per rank (which separates "a small
share of many bodies" from "an idle rank"), ghost-sync payload, peak RSS
(normalised — `ru_maxrss` is bytes on macOS and kilobytes on Linux), throughput,
and VTU bytes and rate from `stat()`ing what was actually written.

---

# Shakedown: Apple M-series laptop, 4 performance cores {#scaling_shakedown}

**These are not scaling results.** `docs/pages/analysis.md` already states why the
development machine cannot produce them: above four ranks half the ranks land on
efficiency cores. They are here as evidence that the instrument measures what it
claims to. One repeat, 10 timed steps, `--phi 0.05`.

## Strong: 256 bodies × refine 3 = 327,680 triangles

| P | grid | ms/step | speedup | efficiency | comm % | imbalance | halo % |
|---|---|---|---|---|---|---|---|
| 1 | 1x1x1 | 8.216 | 1.00 | 1.00 | 38.7 | 1.000 | 0.00 |
| 2 | 2x1x1 | 11.239 | 0.73 | 0.37 | 69.3 | 1.011 | 1.04 |
| 4 | 2x2x1 | 11.751 | 0.70 | 0.17 | 77.9 | 1.077 | 2.06 |
| 8 | 2x2x2 | 21.736 | 0.38 | 0.05 | 88.1 | 1.182 | 2.99 |

The per-timestep loop redistributes **every step**, so at this size it is
communication-bound from the first added rank: the communication share goes
38.7 % → 88.1 % over the sweep while the structural imbalance stays under 1.2. The
curve is negative on this machine, which is what a four-performance-core laptop
running eight ranks is expected to produce, and is exactly the measurement the
library's own documentation says needs a cluster.

## Granularity: the iso-triangle ladder at np = 4

Every rung is 327,680 triangles. Only the split changes.

| bodies | refine | tri/body | ns/triangle/step | comm % | halo % | imbalance |
|---|---|---|---|---|---|---|
| 4096 | 1 | 80 | 72.5 | 75.8 | 3.02 | 1.04 |
| 1024 | 2 | 320 | 39.4 | 73.1 | 2.13 | 1.08 |
| 256 | 3 | 1280 | 55.8 | 76.3 | 2.06 | 1.08 |
| 64 | 4 | 5120 | 36.0 | 79.3 | 1.26 | 1.38 |
| 16 | 5 | 20480 | 47.7 | 84.9 | 0.99 | 1.18 |

**The same work costs about 2× more as 4096 tiny bodies than as 64 large ones**,
and the two supporting quantities move monotonically in opposite directions: the
halo fraction falls 3.02 % → 0.99 % as bodies grow, while the load imbalance rises
as there are fewer bodies to distribute. That is the granularity axis doing exactly
what it was added to expose — and it is invisible to both scaling laws, since every
row is the same triangle count at the same rank count. The `ms/step` column is noisy
at one repeat; the halo and imbalance trends are not.

## Cluster {#scaling_cluster}

Not yet run. The open questions this harness exists to answer:

- Does the communication share keep rising past 88 %, or does it plateau once a
  rank holds tens of thousands of triangles rather than a few thousand?
- Does the weak curve stay flat once the domain really grows with the grid, or does
  the 26-neighbour exchange start to cost more at P in the hundreds?
- Where does the granularity 2× go at a rank count where a large body genuinely
  spans several rank boxes?
- Does the shared MPI-IO file beat P per-rank files on a parallel filesystem?
  `docs/performance-analysis.md` §8 says the laptop is the configuration least able
  to tell them apart, and the harness now measures both.
