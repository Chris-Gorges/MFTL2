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

# the spanning-vs-dispersed redistribution matrix (section below), np = 2,3,4
python3 run_scaling.py --preset redist --results results/scaling-redist
```

**Give a new commit its own `--results` directory.** The store is loaded whole and
repeats collapse to a minimum without consulting the recorded git hash, so results
from two commits in one directory silently merge into one curve. `results/scaling/`
is `e38d23d`; `results/scaling-redist/` is `dbc0556`.

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
runs inside that redistribution can lower the peak. `3680 · 4^refine + 176` bytes per body: 57.5 MiB
for 256 bodies at refine 3, 460 MiB for 512 at refine 4.

That constant is `140 B/triangle + 88 B/vertex`, and it is the POOL BLOCK SIZE from
`triangle_layout_compute` and `vertex_layout_compute` in `src/object/object.c` — not an estimate.
**It read 5120 until 2026-08-31**, because the triangle block lost the 72 B/triangle persistent
frames in commit `71199c6` and neither the benchmark's own guard nor any of the four places that
quote the formula were updated with it. Every generation figure in this document and in
`examples/parallel_advection_scaling.c` was therefore 1.391× too high, which made `--max-gen-mb`
refuse runs the machine could do. There is no compile-time link between the two — if a pool array
is added or removed, the constant in the benchmark has to move too.

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

# Shakedown at `e38d23d`: Apple M-series laptop, 4 performance cores {#scaling_shakedown}

Kept as the BEFORE for the redistribution measurement below, which was taken at
`dbc0556` after five commits on that path.

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

# The redistribution matrix at `dbc0556`: spanning against dispersed {#scaling_redist}

Five commits landed on the redistribution path after the shakedown above
(`2272622` instrumentation and five fixes, `8b37d2c` the body skip, `707c0d7`
`MFTLObjectShrinkToFit`, `71199c6` the frame deletion, `dbc0556` the review
close-out). This re-measures with three changes of method, each of which the
shakedown's own numbers argue for:

- **Baseline np = 2, not np = 1.** At one rank no ghosts exist and the
  redistribution communicates with nobody, so np = 1 is a different operation
  rather than the same one on fewer ranks. The shakedown's headline speedup of
  0.73 at np = 2 is mostly the cost of becoming parallel at all.
- **Stop at np = 4.** The machine has four performance cores. Every rank above
  that lands on an efficiency core, which is a hardware cliff indistinguishable
  from a scaling limit in a plot.
- **Sweep SIZE over 64x**, 81 920 to 5 242 880 triangles. The shakedown measured
  one size, 327 680, and concluded the loop is communication-bound. That is a
  statement about that size.

Two regimes over one ladder. **SPANNING** is one body of radius 1.5 in a 4^3
domain, split across every rank (`bodies touched` is 1..1 of 1 at every point).
**DISPERSED** is 64 to 4096 refine-3 bodies each well inside a rank box. The grid
is pinned to a slab, `--cuts P 1 1`, so the rank count is the only thing moving
along a curve; `run_scaling.py --preset redist` is the whole configuration.

Minimum over three repeats, 30 timed steps, 3 warm-up, round robin over the points.

## 1. A spanning body costs about 2x per triangle, at every size {#scaling_redist_granularity}

| triangles | spanning ns/tri/step | dispersed ns/tri/step | ratio |
|---|---|---|---|
| 81,920 | 11.2 | 7.6 | 1.47x |
| 327,680 | 13.0 | 7.1 | 1.83x |
| 1,310,720 | 13.7 | 7.2 | 1.90x |
| 5,242,880 | 15.2 | 6.8 | 2.24x |

(np = 4, strong.) This is the granularity 2x the iso-triangle ladder found — but
that ladder stopped at 16 bodies, and the open question this document has carried
was where the 2x goes *"at a rank count where a large body genuinely spans several
rank boxes"*. It does not go anywhere: it is still 2x at one body, and it **widens
with size** while the dispersed cost per triangle stays flat.

## 2. The overhead amortises for dispersed bodies and never for a spanning one {#scaling_redist_amortise}

Redistribute as a share of the phase budget, np = 4:

| triangles | spanning | dispersed |
|---|---|---|
| 81,920 | 61.9 % | 54.3 % |
| 327,680 | 64.5 % | 49.2 % |
| 1,310,720 | 65.9 % | 42.6 % |
| 5,242,880 | 67.4 % | **39.1 %** |

**The two rows answer the size question in opposite directions.** Growing a
dispersed ensemble 64x takes the redistribution from 54 % of the step to 39 %:
the per-body and per-message costs are fixed, the mesh work is not, so the
overhead dilutes. Growing a single body 64x moves its share from 62 % to 67 % —
the wrong way. Nothing dilutes, because for one body every cost in the phase is
proportional to the mesh.

Strong scaling 2 -> 4 ranks says the same thing from the other side, and the
trends are monotone and opposite:

| triangles | spanning speedup | dispersed speedup |
|---|---|---|
| 81,920 | 1.18 | 1.01 |
| 327,680 | 1.11 | 1.08 |
| 1,310,720 | 1.05 | 1.10 |
| 5,242,880 | **0.98** | **1.17** |

At 5.2 M triangles a spanning body is fractionally *slower* on four ranks than on
two. Ideal is 2.00 in both columns; nothing here is close, and that is the
laptop's answer, not the library's — but the SIGN of the trend is a code
statement and the two rows disagree about it.

## 3. For a spanning body the halo grows with the cut count, so the ghost sync anti-scales {#scaling_redist_ghost}

Where the spanning row's time goes, 5 242 880 triangles, ms/step:

| P | advect | ghost sync | geometry | redistribute | post-geom | halo % |
|---|---|---|---|---|---|---|
| 2 | 1.548 | 5.456 | 9.854 | 50.706 | 9.783 | 0.13 |
| 3 | 1.081 | 11.410 | 7.778 | 52.714 | 7.191 | 0.22 |
| 4 | 0.869 | 11.297 | 7.272 | 53.152 | 6.308 | 0.31 |

The three local phases scale (advection 1.78x over 2x ranks). **Redistribute is
flat and ghost sync doubles**, and both have the same cause: more cut planes
through ONE body. Per-rank ownership halves from P=2 to P=4, but the body's
surface near a plane does not — so the halo grows (0.13 -> 0.31 %) and so does the
fraction of entities that change cell each step. The two communication phases are
sized by the cut planes, not by the per-rank share, so neither shrinks when the
share does. Slicing an ensemble of SMALL bodies has no equivalent effect, because
a body is mostly interior to one cell and migrates whole: the dispersed row at the
same triangle count has ghost sync at 0.750 -> 0.892 ms, a phase small enough not
to matter.

**The np = 2 anchor is also structurally different, not merely smaller.** On a
periodic axis with two cells the -1 and +1 neighbour offsets land on the same rank
and are deduplicated (`src/ensemble/ensemble.c`), so a pinned `2 1 1` grid has ONE
distinct neighbour per rank where `3 1 1` and `4 1 1` have two. That is a second
reason the P2 -> P3 step is not comparable to P3 -> P4, and it is why the weak
tables print a second baseline. Read the trend from P3 -> P4.

## 4. Above ~300k triangles per rank the weak curve measures the memory system {#scaling_redist_bandwidth}

Weak efficiency against P = 2 degrades with the per-rank size: 0.82 at 20 k
triangles per rank, 0.65 at 1.31 M. The advection phase says why, and it is not
communication — that phase is `MFTL_FOREACH_OWNED_VERTEX` plus
`MFTLApplyPeriodicity`, it makes no MPI call, and in weak mode its per-rank work
is constant by construction:

| per-rank triangles | advect at P=2 | at P=4 | slowdown |
|---|---|---|---|
| 20,480 | 0.014 | 0.015 | 1.07x |
| 81,920 | 0.056 | 0.069 | 1.23x |
| 327,680 | 0.274 | 0.407 | 1.49x |
| 1,310,720 | 1.315 | 1.907 | **1.45x** |

A communication-free kernel doing identical per-rank work cannot slow down 1.45x
for any reason internal to the library. At 20 k triangles a rank's working set is
in cache; at 1.31 M it streams from DRAM, and four cores share one memory
controller. **So the weak rows above 327 680 triangles per rank are a property of
this laptop and must not be quoted as scaling.** The same contention is inside the
strong rows, where it is harder to see because the per-rank size falls as P rises.

## 5. Grid shape: pinning slabs is free for dispersed and cheaper for spanning {#scaling_redist_shape}

Each case also ran one np = 4 point at `2 2 1` against its `4 1 1` curve point.

| case | 4x1x1 | 2x2x1 | ratio | halo slab -> square |
|---|---|---|---|---|
| disp_5243k strong | 35.63 | 35.70 | 1.00x | 0.72 -> 0.74 % |
| disp_1311k strong | 9.44 | 9.24 | 0.98x | 1.22 -> 0.97 % |
| span_5243k strong | 79.43 | 86.54 | **1.09x** | 0.31 -> 0.25 % |
| span_1311k strong | 18.00 | 20.44 | **1.14x** | 0.63 -> 0.50 % |
| disp_5243k weak | 30.77 | 33.99 | 1.10x | 0.25 -> 0.60 % |
| disp_328k weak | 1.56 | 2.08 | **1.34x** | 0.75 -> 1.44 % |

Three separate results. For **dispersed strong** the shape is a wash, so pinning
slabs to isolate P costs nothing. For **weak** the square grid is 1.10-1.34x
slower and its halo roughly doubles — a rank gains ghost layers on four faces
instead of two, at an identical rank box. For a **large spanning body** the square
grid is 1.09-1.14x slower *while carrying LESS halo*, so that cost is not halo
volume; it is the destination and message structure of the redistribution, and it
is the one row where the pinned slab is not merely a control but the faster
configuration.

## 6. The generation estimate was 1.391x too high, and it is fixed {#scaling_redist_genfix}

`examples/parallel_advection_scaling.c` estimated the rank-0 generation peak at
212 B/triangle. The triangle pool block has been 140 B/triangle since commit
`71199c6` deleted the 72 B/triangle persistent frames; nothing updated the
constant, or the four places in the tree that quote the resulting per-body
formula. Every generation figure ever printed or documented was 1.391x too high,
and `--max-gen-mb` refused runs the machine could do. Corrected to
`3680 * 4^refine + 176` bytes per body throughout — one refine-9 body and 4096
refine-3 bodies are both 920 MB, which is how the two rows of this matrix are
memory-matched. Measured peak RSS at the top rung: 1.84 GB on the busiest rank.

## What this does NOT settle {#scaling_redist_open}

Everything above is four ranks on a laptop with four performance cores and one
memory controller. Result 4 says outright that the largest rung is bandwidth-bound
here. The questions that need a cluster are unchanged, and one is now sharper:
**the dispersed share falls 54 % -> 39 % over 64x of mesh; does it keep falling,
or is 39 % the floor?** A rank holding tens of thousands of triangles rather than
1.3 million is the regime that answers it.

# What every rank pays for bodies it does not hold {#scaling_replicated_bodies}

Every rank keeps an `MFTL_Object` for every body in the ensemble, enters every
per-body loop for it, and appears in every per-body collective for it — whether or
not it holds a single triangle of it. In a **weak** sweep the body count grows
with the rank count, so all of that grows linearly with P while the work per rank
stays flat. This section is the instrumentation that makes it visible, and what
has changed so far.

## The instrument: three workspace classes and an empty-body count

`MFTL_RedistributeStats` now carries the persistent redistribution workspace split
by what it is proportional to — `ws_bytes_per_body`, `ws_bytes_per_rank`,
`ws_bytes_per_entity`. They are HIGH-WATER MARKS read off the capacities, not the
running `n_bytes_allocated` churn sum, so they are a footprint. The benchmark
reports **min..max over ranks**, and the gap is load-bearing: capacity is a
ratchet and rank 0 held the whole ensemble during the opening scatter, so its
per-entity mark is the whole mesh's forever. Reading only the maximum blames the
library for the harness's serial generation.

`--mode weak --spheres-per-rank 32 --refine 1`, Apple M-series laptop
(np = 8 is oversubscribed; these are byte counts, not timings, so that is fine):

| np | bodies | empty bodies/step | per-body kB | per-rank kB | per-entity kB (min..max) |
|---|---|---|---|---|---|
| 2 | 64  | 3.4 %  | 11.0 | 0.2 | 18.2 .. 50.2 |
| 4 | 128 | 18.1 % | 11.0 | 0.4 | 26.2 .. 98.2 |
| 8 | 256 | 37.2 % | 20.5 | 0.8 | 26.2 .. 194.2 |

Read the columns against each other, because only one of them is allowed to move:

- **`per-entity` min is flat** (18 → 26 → 26 kB). That is the control. Weak scaling
  holds the work per rank constant, so this figure is what "constant" looks like,
  and without it the other columns say nothing.
- **`per-body` has min == max** and steps up on the doubling ladder. Every rank
  pays the identical amount, held or not — which is the whole shape of the defect
  in one number.
- **`per-rank` tracks P exactly** (0.2 → 0.4 → 0.8). Separate defect, same sweep.
- **The empty-body fraction is the headline: 3.4 % → 18.1 % → 37.2 % over 4× of
  ranks**, heading for 100 %. At np = 8, 762 of every 2048 per-body visits per step
  are on a body this rank holds nothing of.
- **`per-entity` max doubles with P** and it is NOT a library defect: it is rank 0's
  ratchet from the opening scatter, the workspace analogue of the stranded pool
  capacity in §scaling_trap_pools. It is why the minimum is reported at all.

## What the per-body quality reduction costs, and why it stays

`MFTLEnsembleUpdate` puts 11 doubles PER BODY into three `MPI_Allreduce`s every
timestep — 1.4 MB per reduction at 16 384 bodies, 88 MB at a million. Measured
here, `--spheres 16384 --refine 0`, np = 4, best of three: **0.110 ms/step, 28 %
of the timestep.** An ensemble-wide aggregate instead — 11 doubles full stop —
was measured at **0.091 ms/step and 24 %**, i.e. **1.21×** for a 16384× smaller
payload, because four ranks on one node reduce through shared memory at
~200 GB/s. On this machine the payload is nearly free; on a real network it is
the whole cost.

**The aggregate is not the fix, and it was reverted.** Per-body quality is what a
solver reads to decide whether a GIVEN droplet needs remeshing; a single number
over every triangle of every body says that something degenerated and not which
body it was. So the payload stays per body, and what has to change is the set of
ranks each body's slot is reduced over — only its holders have anything to
contribute. That converts O(bodies) into O(bodies held locally) without giving up
a number the caller wants, and it is the same mechanism every other per-body
reduction in the library needs. **The 1.21× above is therefore the upper bound on
what a holder-set reduction can win here, and it is small precisely because this
laptop cannot see message volume.** Re-measure on the cluster.

**Generation's collectives are batched.** `MFTLGenerateSphere` did one `MPI_Bcast`
of three ints and one `MPI_Allreduce` of two, PER BODY, in strict sequence on the
one rank building everything. `MFTLGenerateSphereLocal` + `MFTLPublishGeneratedBodies`
is 2 collectives for the whole ensemble.

**And it measures nothing here, which is worth recording rather than omitting.**
Setup at 512 bodies, np = 4: 0.0119–0.0131 s batched against 0.0136–0.0142 s per
body; at 2048 bodies the two are indistinguishable (0.029–0.032 s either way).
Shared-memory collectives cost about what the surrounding loop iteration does, and
setup is dominated by rank 0 actually subdividing the icospheres. It is a
message-COUNT change justified structurally: 2N collectives in strict sequence at
even 5 µs each is 20 ms at 2048 bodies and 10 s at a million, none of it
overlappable, all of it before the first timestep. Re-measure on the cluster
before quoting a number.

## Body identity has moved; the storage has not

Bodies are now named on the wire by `MFTL_Object.global_id`, handed out densely by
`MFTLEnsembleAddObject`, rather than by their position in `MFTL_Ensemble.objects`.
`MFTL_OwnerMap` is keyed the same way, `slot_of_body` in `redistribute.c` is the
one place a wire id becomes a local index, and the redistribution's Debug checksum
now reduces `(num_objects, n_bodies_global, skip switch)` instead of hashing every
id — because hashing the id list asserts exactly the property being removed.

**It changes no measurement, and that is the point.** Every rank still holds every
body, so `global_id ==` the list position and every number in the table above is
unchanged to the byte. Splitting the identity change from the storage change is
what makes each of them testable on its own: the whole suite had to pass
unmodified, and the only two edits it needed were assertions that had hard-coded
the *old* identity scheme (`test_multibody_round`'s `100 + o`, `test_io`'s
`bid[t] - 100`).

One piece of state moved as part of it. The "has this body's layout settled" bit
was the third slice of redistribution's vote bitset, indexed by list position, and
is now `MFTL_Object.partition_settled`. A workspace bitset indexed by position has
to be discarded whenever the list changes, so **adding one body un-settled every
body** and cost the whole ensemble a rebuild; on the object, a body carries its own
answer and one created on arrival simply starts unsettled. It fits in padding that
was already there, so `sizeof(MFTL_Object)` is still 264 B.

## The object list is sparse, and the per-rank body work is flat

A rank now holds an `MFTL_Object` exactly for the bodies it holds entities of:
`MFTL_Ensemble.local_objects` is a gid-sorted subset, redistribution destroys a
body whose entities have all left and creates one arriving where the rank held
nothing, and `n_local_objects` is a local quantity while `n_bodies_global` is the
ensemble.

### Measured against the old code, in one session {#scaling_before_after}

Everything below is a **differential**: `cd7fcd2` (the replicated object list and
the `comm_world` per-body reductions) built and run against the current tree, on
the same laptop, in the same session, with byte-identical benchmark flags —
`--steps 60 --repeat 3 --quality`, best of three repeats after three warm-up
steps. The older tables in this document were taken at different commits and are
kept only for their trends.

**Weak: `--mode weak --spheres-per-rank 32 --refine 1`.** The body count grows
with the rank count, which is the sweep the whole change exists for.

| np | bodies | body visits/rank/step | empty bodies | ms/step | quality ms/step |
|---|---|---|---|---|---|
| 2 | 64  | 64 → **36.4** (1.76x) | 2.4 % → **2.1 %**  | 0.096 → 0.074 | 0.0266 → 0.0203 |
| 4 | 128 | 128 → **41.0** (3.12x) | 18.5 % → **4.3 %** | 0.127 → 0.129 | 0.0332 → 0.0356 |
| 8 | 256 | 256 → **46.8** (5.48x) | 34.6 % → **6.3 %** | 0.410 → 0.415 | 0.1409 → 0.1392 |

**The reduction factor is the point, not the absolute.** It rises 1.76x → 3.12x →
5.48x over 4x of ranks, i.e. per-rank body work went from *tracking P* to nearly
flat — 36.4, 41.0, 46.8 against 64, 128, 256. The residual rise is halo sharing
(a rank holds its own ~32 bodies plus a sliver of its neighbours'), not the
ensemble growing, and the empty-body fraction says the same thing from the other
side: it was heading for 100 % and now sits under 7 %.

**Body count at fixed np = 4, `--refine 0`.** The other axis: hold the ranks
still and grow the ensemble.

| bodies | body visits/rank/step | empty bodies | ms/step | quality ms/step |
|---|---|---|---|---|
| 256   | 256 → **79.4** (3.22x)     | 12.4 % → **2.2 %** | 0.081 → 0.081 | 0.0223 → 0.0207 |
| 1024  | 1024 → **298.3** (3.43x)   | 9.6 % → **2.3 %**  | 0.260 → 0.260 | 0.0735 → 0.0696 |
| 4096  | 4096 → **1123.1** (3.65x)  | 7.8 % → **2.3 %**  | 1.554 → 1.342 | 0.4155 → 0.3776 |
| 16384 | 16384 → **4353.1** (3.76x) | 6.5 % → **2.4 %**  | 11.60 → 11.15 | 3.174 → 2.939 |

**Strong: 256 bodies x refine 3 = 327,680 triangles.** The regression check — a
fixed problem on more ranks, where the change must cost nothing.

| P | ms/step | quality ms/step | body visits/rank/step | halo % | imbalance |
|---|---|---|---|---|---|
| 1 | 4.590 → 4.583 (1.00x) | 1.9235 → 1.9162 | 256 → 256 (1.00x) | 0.00 | 1.000 |
| 2 | 3.488 → 3.535 (0.99x) | 1.0286 → 1.0238 | 256 → **145.2** (1.76x) | 0.86 | 1.013 |
| 4 | 2.847 → 2.809 (1.01x) | 0.6900 → 0.6622 | 256 → **85.2** (3.00x) | 1.77 | 1.026 |
| 8 | 3.922 → 3.943 (0.99x) | 1.4882 → 1.3822 | 256 → **46.9** (5.46x) | 2.72 | 1.091 |

### The honest reading: the counted quantities moved, the clock did not

**Wall clock is a wash on this machine — 0.98x to 1.16x, and the noise floor is
wider than most of that.** Five independent np = 2 weak runs of the current tree
give step medians of 0.092, 0.075, 0.073, 0.073 and 0.072 ms, a **±28 % spread**,
so every "win" under about 1.3x in the tables above is unmeasurable here and the
1.30x at np = 2 in the weak table is noise. What IS reproducible to the digit is
every counted column: body visits, empty fraction, workspace bytes and the
reduction routes are bit-stable across runs.

That is the expected outcome and not a disappointment. Four performance cores
reduce through shared memory at ~200 GB/s and a per-body loop iteration on a body
this rank holds nothing of is a handful of predicted branches; the cost this
change removes is **memory per rank and message volume per rank**, and neither is
on the critical path of a four-core laptop. The measurement that would show it
needs P in the hundreds — the same conclusion §scaling_shakedown reached about
the scaling curves themselves.

**RSS barely moves either, and the reason is worth stating.** 280.9 → 281.4 MB at
16 384 bodies. The per-body arrays this change made local are tens of bytes per
body against the ~5 kB of mesh each body carries at refine 0, so at any body
count that fits on a laptop the pools dominate. The arrays that were removed
scale with `bodies x ranks`, which is the axis a laptop cannot grow.

## Per-body reductions go to the body's HOLDERS {#scaling_holder_reductions}

Three collectives in a timestep reduce one record per body: the redistribution's
skip vote (1 value), `MFTLEnsembleUpdateGlobalCounts` (2), and
`MFTLEnsembleUpdate`'s quality statistics (11). All three staged into a buffer
indexed by BODY ID and called `MPI_Allreduce` on `comm_world`, because that is
the only index that means the same thing on two ranks holding different bodies.

`MFTLEnsembleReduceBodies` (`src/ensemble/ensemble_reduce.c`) replaces the
communicator rather than the payload. The caller's buffer is indexed by LOCAL
SLOT, and the reduction is one `MPI_Neighbor_allgatherv` of `(body id, payload)`
records over the 26-neighbour graph whenever every body's holders provably fit
inside one 26-neighbourhood, with the old `comm_world` reduction as the fallback.

**Three separate things changed, and only one of them is bytes.**

**1. The per-rank staging memory is gone.** `MFTLEnsembleUpdate` allocated
`22 x bodies` doubles on every rank on every call — the six logical arrays are
local and global min, max and sum, i.e. 176 B/body, so **148 kB at 864 bodies and
176 MB at a million** — to learn eleven numbers about the bodies it holds. It
is now `11 x bodies held`, a few kilobytes at any ensemble size. Same for the
counts and for the vote bitset. This one is exact, not measured, and it is the
largest of the three.

**2. A global synchronisation became a neighbourhood one.** What remains on
`comm_world` per reduction is one `MPI_Allreduce` of a SINGLE int, agreeing which
route to take; the data moves over `comm_graph`. At `P` in the thousands that is
the difference between a payload-carrying global tree and 26 point-to-point
messages, and it is the part this laptop cannot measure at all.

**3. Bytes: WORSE below about 45 ranks, better above, and the crossover is
arithmetic rather than a guess.** Weak sweep, `--spheres-per-rank 32 --refine 1
--quality`, all three reductions live, payload bytes per rank per step:

| np | bodies | bodies held | holder route | previous `comm_world` model | ratio |
|---|---|---|---|---|---|
| 2  | 64  | ≤ 38 | 12.1 kB | 6.2 kB  | **1.97** |
| 4  | 128 | ≤ 47 | 23.0 kB | 12.3 kB | **1.87** |
| 8  | 256 | ≤ 52 | 42.3 kB | 24.6 kB | **1.72** |
| 27 | 864 | ≤ 60 | 141 kB  | 83.1 kB | **1.70** |

The neighbourhood route sends every held body to every neighbour, so its payload
is `b_local x (1+d)` records against the old route's `B_global`. With `d` capped
at 26 the ratio is `1.4 x (1+s) x (1+d) / P`, where `(1+s)` is the sharing factor
(`b_local x P / B_global`, measured at 1.19 here). Setting it to 1 gives a
crossover at **P ~ 45**, i.e. anything past a 3x3x3 grid, and the advantage then
grows LINEARLY with P: 22x fewer bytes at P = 1000, 220x at P = 10 000. The
measured ratios above fall from 1.97 to 1.70 exactly as `(1+d)/P` saturates.

**Per consumer the crossover differs by two orders of magnitude, and the vote is
the awkward one.** Quality (11 slots against 88 B/body) crosses at P ~ 35;
counts (2 slots against 8 B/body) at P ~ 96; the vote (1 slot against **one BIT**
per body) at P ~ 4100. A bitset is simply a very good encoding, and nothing with
a `(gid, value)` record beats it at small `B`. It is kept on the same mechanism
anyway, for the two reasons above and because at the design point it wins on
bytes too: at `P = 10 000, B = 10^6` the bitset is 125 kB per rank per step
against 43 kB — and `b_local` stays at ~100 while `B` does not. A narrower record
type for one-slot reductions is the obvious optimisation if a measurement ever
asks for it; it was not taken, because two payload encodings is two mechanisms.

### Stage 3 in isolation: the route changed, the clock did not {#scaling_holder_ab}

The table above bundles the sparse list with the holder route. Forcing the three
call sites to `MFTL_BODY_REDUCE_SCOPE_GLOBAL` isolates the route alone — same
sparse directory, same everything, only the communicator differs:

| case | bodies | P | quality ms/step | redistribute ms/step | reduce kB/rank/step |
|---|---|---|---|---|---|
| weak  | 64   | 2 | 0.0277 → 0.0203 | 0.0351 → 0.0261 | 6.0 → **11.9** |
| weak  | 128  | 4 | 0.0378 → 0.0356 | 0.0677 → 0.0600 | 12.0 → **22.4** |
| weak  | 256  | 8 | 0.1336 → 0.1392 | 0.1929 → 0.1980 | 24.0 → **40.9** |
| fixed | 256  | 4 | 0.0220 → 0.0207 | 0.0344 → 0.0353 | 24.0 → **43.4** |
| fixed | 1024 | 4 | 0.0735 → 0.0696 | 0.1002 → 0.1052 | 96.0 → **163.1** |
| fixed | 4096 | 4 | 0.3677 → 0.3776 | 0.4885 → 0.5245 | 384.0 → **614.2** |

**Nothing here is outside the noise floor** (0.93x–1.37x against a ±28 % spread),
and the byte column is *worse* by 1.5x–2.0x exactly as the model predicts at these
rank counts. So this is not a measured speed-up; it is a measured **shape change**
whose payoff is off the end of this machine. What the experiment does confirm is
that both routes are live, that the droplet fixtures never fall back
(`global = 0` in every run above), and that the byte ratio tracks
`1.4 x (1+s) x (1+d)/P` rather than drifting.

**The route is decided, not assumed, and the test is local.** A rank checks that
every body it holds owns at least one entity here, that every OTHER holder named
by its ghosts' `owner_proc` is a graph neighbour, and that those neighbours' cell
offsets span at most one cell per axis; `MPI_LAND` of that is exactly "the
neighbourhood route is exact". It reads only the ghost suffix, so a body held
whole costs nothing, and it is re-derived on every call rather than cached — a
stale verdict is a silently wrong reduction, and the invalidation rule would have
to be maintained by every future holder-set change.

**Two failures the shakedown found, both silent, both worth recording.**

*A generated ensemble vetoes, and has to.* Right after `MFTLGenerateSphere` rank 0
holds every entity and every other rank holds an EMPTY object. Such a rank owns
nothing, so it appears in no other holder's halo and cannot be discovered from
outside — it would sit out the exchange and keep whatever was in its buffer. At
np = 4 on a 4x1x1 grid rank 2 is not even a neighbour of rank 0, and
`test_ensemble_update_tolerates_empty_ranks` failed on exactly that. The empty
holder therefore vetoes for the whole ensemble. After a redistribution no empty
object exists, so this costs nothing in the steady state.

*`comm_graph` carries a SELF-LOOP on a thin grid.* `domain_init_comm` skips the
`(0,0,0)` offset, not the own RANK: on an axis with a single cut and periodicity
every offset along it wraps back onto this cell, so the 26-stencil legitimately
resolves to this rank. An exchange does not care; a REDUCTION does, because the
self block is then folded in twice — invisibly for MIN and MAX, and doubling
every SUM. Grids of shape `{2,1,1}` and `{2,2,1}` have the self-loop and
`{2,2,2}` and `{3,3,3}` do not, which is precisely the set of rank counts at
which the remeshing suite failed.

## What none of this fixes yet

~~Contact detection is untouched and still enumerates every body pair.~~ ~~The
remesher's own per-body reductions still stage by body id on `comm_world`.~~ Both
were done in a second pass; the account is in `docs/scalability-update-status.md`
§Stage 4 (a status document, read in place — it is deliberately not a manual
page), and none of it is measured here in TIME — it is counted quantities and bit-for-bit digest
comparisons, because on a four-core laptop none of these paths is on the critical
one. In short: the all-pairs request list and the `48 x B x nn`-byte extent table
are gone (a neighbourhood exchange of 9 doubles per HELD body, plus a uniform grid
over the arrivals); `MFTLRemeshMeasureBodies`, the per-phase `2B` Exscan and
Allreduce, the dispersed id counters and the Debug watermark check all go through
`MFTLEnsembleReduceBodies` or its new sibling `MFTLEnsembleExscanBodies`.

**What remained after that pass was one thing, and it is now gone too:** the
dispersed balancer's replicated greedy. Its assignment has to be identical on
every rank and its greedy is over rank *loads*, so as long as every rank ran the
greedy it genuinely needed replicated per-body data — a property of the algorithm
and not of the route, which is why it outlived every other `O(B)` term. The fix
was to stop having every rank run it: the owner of a body's reduced centroid
decides that body alone and publishes `(body, destination, weight)` one hop along
`comm_graph`, which every holder of it is on. Payload and staging are
`O(local bodies x graph degree)`.

The balance is unchanged — 4.00 / 8.00 / 27.00 -> 1.25 at np = 4 / 8 / 27 on the
clustered fixture, no rank idle — and the counted quantity that moved is the only
observable it has: on a rank holding NONE of the ensemble, the balancer stages
25 / 49 / 85 doubles at np = 2 / 4 / 8 **and that number does not change when the
body count doubles**, against `4B` on every rank before. Gated by
`test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold`, because no digest
and no imbalance figure can see the difference. Full account in
`docs/parallel-remeshing-analysis.md` §7.4.

Nothing in the library is now proportional to `n_bodies_global` on a per-timestep
or per-remesh path. What is left is proportional to `nproc`: the `[nproc]` inverse
neighbour table in `MFTLRemeshSyncGhostIds` and the several `[nproc]` arrays inside
`redistribute_core` (CLAUDE.md deferred issue 6), which dominate it.

Generation is still rank-0 serial, so the peak of §scaling_trap_pools stands and
setup time still grows linearly with `nproc` in a weak sweep.

# Cluster {#scaling_cluster}

Not yet run. The open questions this harness exists to answer:

- Does the communication share keep rising past 88 %, or does it plateau once a
  rank holds tens of thousands of triangles rather than a few thousand? *Partly
  answered by §scaling_redist_amortise: over 64× of mesh the DISPERSED share falls
  54 % → 39 % while the SPANNING share rises 62 % → 67 %. Whether 39 % is a floor
  needs a rank holding tens of thousands rather than 1.3 million triangles.*
- Does the weak curve stay flat once the domain really grows with the grid, or does
  the 26-neighbour exchange start to cost more at P in the hundreds? *Unanswerable
  here: §scaling_redist_bandwidth shows the laptop's weak curve is memory-bound
  above ~300 k triangles per rank, so it measures the machine.*
- ~~Where does the granularity 2× go at a rank count where a large body genuinely
  spans several rank boxes?~~ **Answered** by §scaling_redist_granularity: nowhere.
  It is still 1.5–2.2× at ONE body, and it widens with mesh size rather than
  closing. Worth re-checking at P in the hundreds, where a spanning body is sliced
  far more ways than by four cut planes.
- Does the shared MPI-IO file beat P per-rank files on a parallel filesystem?
  `docs/performance-analysis.md` §8 says the laptop is the configuration least able
  to tell them apart, and the harness now measures both.
