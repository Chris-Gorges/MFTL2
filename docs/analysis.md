# MFTL: measurements, decisions and open issues {#mftl_page_measurements}

The one document behind the code comments that cite `docs/analysis.md §N`. It
records what was **measured**, what was **decided** because of it, what was
**considered and rejected**, and what is **still open**. The invariants
themselves live in `CLAUDE.md` and the `src/<module>/README.md` chapters; this
file is the evidence behind them.

Everything here was measured on an Apple M-series laptop (4 performance cores,
Release + LTO) unless stated. **Nothing has been measured above four ranks on
real hardware**: runs at np = 8 and 27 are oversubscribed, so their ratios are
usable and their absolute times are not. Re-run the named example before quoting
a number; §10 lists the traps that produced plausible wrong answers first.

| § | topic | main source |
|---|---|---|
| 1 | data layout and single-rank kernels | `layout_benchmark`; the kernel benchmarks were retired with the frames (§1.2), so their tables are the record |
| 2 | redistribution | `parallel_advection_redistribute_list` |
| 3 | parallel I/O | `MFTLWriteVtu` harness |
| 4 | serial remeshing quality | `leveque_remeshing_singleproc` |
| 5 | distributed remeshing | `leveque_remeshing_parallel`, `remesh_regime_sweep` |
| 6 | scaling and the per-body collectives | `parallel_advection_scaling`, `run_scaling.py` |
| 7 | silent failures found, and how each is now guarded | tests named inline |
| 8 | architectures evaluated and not taken | |
| 9 | open issues | |
| 10 | measurement traps | |

---

## 1. Data layout and single-rank kernels

### 1.1 Record sizes

`MFTL_Vertex` 16 B, `MFTL_Triangle` 12 B, `MFTL_HalfEdge` 16 B. What is *not* in
them is a second copy of something the record already determines: `he_idx`,
`next_idx` and `triangle_idx` are arithmetic (implicit indexing), positions live
only in the SoA, ownership is the pool prefix, and "boundary" is a flag bit. A
triangle's topology is 60 B (`12 + 3·16`) against 104 B for self-describing
records. Pool blocks are **64 B/vertex and 140 B/triangle** (`vertex_layout_compute`,
`triangle_layout_compute` in `src/object/object.c`), which gives a generated
sphere `3440 · 4^refine + 128` bytes. The vertex block lost 24 B when
`verts.area`, `verts.mean_curv` and `verts.gauss_curv` — allocated, permuted and
migrated, but never produced by anything — were deleted; the triangle block lost
72 B with the frames (§1.2).

### 1.2 Geometry kernels, and why persistent frames were removed

The half-edge walk and a streaming kernel over cached per-triangle frames
(`anchor` + two seam-corrected edges, 72 B/triangle) are bit-identical. In
isolation the frames were 1.25–1.57× faster non-periodic and 1.94–2.00× periodic.
Inside the real timestep they bought ~1 % (0.535 → 0.529 ms/step), because the
geometry pass after a redistribution runs on freshly rewritten pools and is bound
by first touch rather than arithmetic, and the redistribution spent 3.6 ms/step
(11 % of the phase) rebuilding frames that step 1 of the next timestep discarded
unread.

**Decision: frames deleted.** Measured at np = 4, 256 bodies, refine 3, 500 steps:
whole loop **1.23× faster**, redistribution + post-geometry 1.28×, and still
1.22× with `--quality --normals` forcing the gathered consumers; −72 B/triangle of
capacity; partition digests identical. Geometry now gathers positions through
`tris.verts` and builds minimum-image edges in registers.

### 1.3 Morton renumbering: a negative result on this machine

`MFTLObjectReorderMorton` cuts the mean edge index distance ~160× and buys **no
measurable time** at any size up to a 15 MiB position array (0.98–1.02×), almost
certainly because the M-series L2 absorbs the scattered pattern. It is kept
because it is what lets the SFC order be the *pool* order (no second index space)
and it is the LBVH's input. **Re-measure on an x86 node with 1–2 MiB L2 per core**
before deciding its place in a per-timestep loop.

### 1.4 Position layout

Interleaved `pos[3v+k]` against split component arrays and a 32-byte padded
interleave, bit-identical outputs, ns/element:

| pattern | packed | pad4 | split |
|---|---|---|---|
| stream-reduce (bbox) | 0.49–0.55 | 0.49–0.55 | 0.50–0.56 |
| stream-map (translate) | 0.34–0.37 | 0.51–0.53 | **0.34–0.35** |
| gather (3 positions/triangle) | 1.93–2.28 | 1.91–2.26 | **1.89–1.90** |
| scatter-accumulate (vertex normals) | **2.22–2.27** | 2.22–2.28 | 2.73–2.77 |

Split wins the gather through address generation (a 24 B stride needs a multiply,
8 B does not) and loses the scatter by the same mechanism in reverse (one paired
128-bit RMW against three scalar ones). The two per-timestep patterns pull in
opposite directions by similar amounts, so the layout was decided on the API:
**interleaved stays** (`MFTLVertexPos()` returns a plain `double[3]`), and `pad4`
is rejected.

### 1.5 Vertex normals: scatter, and the CSR stays deleted

Scatter 5.24–5.35 ns/vertex, fan walk 17.8–18.4 (dependent loads), CSR gather
5.28–5.41 plus 5.65–5.70 to rebuild the CSR after every topology change and
16 B/triangle of state. **The vertex→triangle CSR stays deleted** — a dead heat
at a permanent cost.

### 1.6 The fused update kernel

`MFTLEnsembleUpdate`'s kernel, eleven variants on one mesh, flat over a 256×
size range (so not memory bound):

- **One `sqrt` and one divide per triangle are 26 % of the pass** and neither is
  removable (the kernel needs `|n|` and `n/|n|`). The loop does not fail to
  vectorise because of its nine-way reduction: splitting the reduction measures
  worse (3.53 vs 3.45 ns), and clang refuses even when a pragma forces it.
- **Separating the quality statistics costs 25 %** (two passes 4.29–4.61 vs 3.45
  fused). They stay fused.
- A step-function abstraction costs 15 %, which is why the kernel stays a
  file-static function next to the reduction it feeds.
- The statistics are **not** nearly free: +90 % on the streaming path (1.64 →
  3.11 ns), because `q = 4√3·A/(a²+b²+c²)` adds a second divide to a loop already
  limited by one. Fused against the naive walk it is still 1.61× (non-periodic)
  and 2.27× (periodic) faster for the same work — call it for the statistics, not
  as a free upgrade.

### 1.7 Scale: every geometric test is dimensionless

A front-tracking run in SI units meshes a millimetre droplet with edges near
1e-4, so no test may compare a dimensional quantity against an absolute constant.
Three generations of that bug were found and fixed:

- **Normals.** Clamping `|n|²` at 1e-12 before the square root floored areas at
  5e-7, made `q` unbounded (an *average* of 1.023 measured) and returned
  proportionally short normals. Clamping the divisor instead only moved the floor
  (a 1e-8 triangle got a normal of length 0.112). The fix is a test against zero,
  `inv = (mag > 0) ? 1/mag : 0`: +1.1 % (7.33 → 7.41 ns/tri), unit at every scale.
- **Quality and vertex normals.** Both now test against zero rather than 1e-12
  (`test_quality_statistics_are_scale_invariant`).
- **The remesher's fold test** compared a squared area vector (dimension L⁴)
  against 1e-15, i.e. an effective cutoff of sin θ ≈ 3.2e-8/ℓ²: 1e-5 at ℓ = 0.05,
  3 % at ℓ = 1e-3, and every triangle below ℓ ≈ 1.8e-4 — below which **every
  collapse and flip was refused** and the mesh could only be refined. It is now a
  per-triangle relative test (`|e₁×e₂|² ≤ 1e-15·|e₁|²|e₂|²`), and the four
  smoothing guards test against zero or against their own edges.
  `test_remeshing_is_independent_of_the_length_unit` remeshes a sphere and the
  same sphere scaled by 2⁻²⁰ and requires the identical mesh times 2⁻²⁰, bit for
  bit (428 splits / 647 collapses / 687 flips both ways; with the old test the
  scaled run made 0 collapses and 0 flips and ended at 2128 triangles against
  842). At unit scale the LeVeque digests are unchanged; the Rayleigh–Taylor
  topology run diverges chaotically from step 155 and ends better on every
  reported measure (worst quality 0.650 → 0.728, V/V₀ 1.00211 → 1.00209).

Morton key scaling likewise needs only a *finite* scale, not a range above 1e-12.

---

## 2. Redistribution

### 2.1 What a distributed timestep costs, and what it defers

np = 4 on 2×2×1, periodic 8×4×4 box, rigidly translating r = 0.1 spheres (first
measurement, before §2.4–2.5):

| per step | 256 bodies, 327 680 tris | 1000 bodies, 1 280 000 tris |
|---|---|---|
| redistribute | 2.71 ms | 10.67 ms |
| + the geometry it defers | **3.25 ms** | **13.19 ms** |
| ghost sync | 0.32 ms | 0.83 ms |

Quote the second row. Redistribution ships only topology and positions
(`MFTL_PackedVertex` 96 → 40 B, triangle 120 → 40 B, ghost payload 80 → 32 B),
so triangle geometry is unspecified afterwards and the recompute (6.4–7.9
ns/triangle, first-touch bound) belongs to its cost. Kept entities never enter
the wire format (the kept set is ~99 % of a partition), there are no hash maps
(dense ids, epoch-stamped tables), and twins are repaired, not rebuilt. The
ghost sync improved by its 2.5× payload reduction and no more: at these sizes it
is three neighbourhood collectives, not bandwidth. Per-destination send buffers
allocated and freed on every call are why a first benchmark run is slower than
every repeat (66–99 allocations, 20–39 MB requested per step before §2.3).

### 2.2 Inside the phase, and the ceiling on a delta engine

`MFTL_RedistributeStats` has nineteen disjoint buckets (timers behind
`-DMFTL_COMM_PROFILE=ON`, residue 1.5 %). At 1000 bodies np = 4 the pool
permutation work (gather, place-back, relink, topology rebuild) was 39 % of the
phase and the work any design keeps (owner classification, destination analysis)
34 %. **So a delta architecture that touched only what moved was capped at about
1.6× on the phase, 1.2× on the timestep** — which is why §8.1 was not built.
Counters: 95 % of entities unchanged per step; bodies visited with nothing to do
48 / 71 / 84 % at np = 2 / 4 / 8 before the sparse object list (§6.4).

### 2.3 Allocator traffic is a footprint problem, not a time one

A persistent workspace, a density-adaptive id table, geometric growth, releasing
departed bodies and routing single bodies through the workspace cut allocations
2.5×, requested MB 2.6× (16× for one large body at np = 8) and the id table 77–183×
— with digests bit-identical — and changed the phase time by **nothing** (0.91–
1.08× against a control phase that moved by the same spread). macOS maps large
blocks lazily, so untouched allocation never reached the critical path.
**Anything that is not a reduction in touched bytes should be assumed to buy
nothing until an interleaved A/B says otherwise.** Re-measure on a cluster
allocator.

### 2.4 Skipping unchanged bodies

`MFTLRedistributeSetSkipUnchangedBodies` (default on). Estimated beforehand at up
to 68 % of the phase for many small bodies and 0 % for one large one; measured
**2.11× / 1.63× on the phase at np = 2 / 4**, 85.6 % / 77.2 % of bodies skipped, whole timestep 1.38×
(redistribution 62.7 % → 50.3 % of it); arming costs nothing measurable (26.72 →
26.70 ms/step). Digests, set and slot-order form, identical with the skip on and
off.

**Why it is correct.** Redistribution sends a triangle to the owner of its centre,
its three corners and its three edge neighbours; read backwards, a rank's local
state is a pure function of (global topology, owner map), and positions only
decide the owner map. A topology change under the freeze discipline is invisible
to that rule: an unfrozen entity has no ghost corner and no ghost edge neighbour,
so every clause of every new triangle resolves to the acting rank, and an edge
neighbour held elsewhere is frozen on its owner and cannot be rewritten. What a
skip *does* break is **ghost triangle global ids**: renumbering rewrites owned ids
and relies on the next redistribution dropping and re-receiving every ghost
triangle. Hence the guard is `MFTLObjectMarkPartitionDirty`, stamped by
`MFTLObjectSetCounts` (the choke point every creation and destruction passes) and
both renumbering sites — not "did topology change".

**A body must be clean twice.** `rebuild_object` is not idempotent (an owned vertex
whose incident triangles all left is dropped and re-received at the end of the
run), so one settling rebuild per dirty→clean transition makes a skipped pool
identical slot for slot; the set digest passed and the order digest failed from
step 1 until this was found. Cost: ~1.4 % of otherwise skippable bodies.

What remains after the skip is halo-proportional (twin repair 1.05×, pool growth
1.07×, against 3.6–5.2× for per-entity buckets) plus `compute_owners`, which is
the decision itself (open issue in §9).

### 2.5 The remaining review items: 1.02–1.04×

Emitting `tris.verts`/`tris.neighbors` inside the relink instead of a closing
pass, a one-pass two-cursor `build_kept_lists`, a debug-only clear, and a
branch-free "whole stencil on one rank" test in `collect_destinations`:
**1.038–1.040×** on the phase where the skip cannot help, 1.02× where it does.
Two of those four carry the win (−20 % on destination analysis, a pure cut in
touched bytes); moving the derived-array stores into the dependency-bound relink
returned only a third of its traffic estimate (0.588 ms saved, 0.369 of it back
in `t_relink`) — **which loop the stores land in matters as much as how many
there are.** Digests identical in 16 configurations.

### 2.6 Stale centres: a bit, not a spot check

Redistribution routes triangles by `tris.center`, and before this was guarded
only a Debug assertion (a full geometry pass) checked it. The realistic failure
was a freshly generated ensemble, whose centres were never computed: every
triangle went to one rank, structurally valid. Generators now leave geometry
current, and `MFTL_DERIVED_GEOMETRY` records "computed over every triangle since
the library last changed its inputs" (cleared by a ghost sync, a topology change,
a rebuild); `MFTLRedistribute` recomputes a body whose bit is clear.

Rejected first: recomputing the centres of the last owned and last ghost
triangle per body and comparing. It caught more in principle (vertices moved
after the pass), but cost **+0.8 to +1.25 %** of the redistribution phase at
1000 droplets, np = 4 (two gathers per body, mostly cache misses) — and the case
only it catches is the benign one: all holders then route by the same old
centres. The bit costs one test per body: paired-ratio median ×1.003 (IQR
×0.995–1.009, 40 interleaved rounds), digests identical in 18 configurations,
and the recompute never fires in any example or test that follows the loop.

---

## 3. Parallel I/O

`MFTLWriteVtu`, 327 680 owned triangles, best of five:

| ranks | ascii, N files | binary, N files | binary, 1 file (MPI-IO) |
|---|---|---|---|
| 1 | 125.6 ms | **3.96 ms** | 4.05 ms |
| 2 | 69.7 ms | **2.68 ms** | 3.27 ms |
| 4 | 41.6 ms | **1.84 ms** | 4.11 ms |

**The 23–32× is the encoding** (`printf` was the bottleneck); the block writer
itself is worth 1.20–1.27× at equal encoding; the shared MPI-IO file is 1.0 /
0.82 / 0.45× on one SSD, where two-phase aggregation has nothing to aggregate
(not the truncate: removing it changed 4.01 → 4.06 ms). `MFTL_IO_SINGLE_FILE` is
kept for what was not measured — P+1 files per dump is what fails first on a
parallel filesystem — and **must be measured on Lustre/GPFS before choosing**;
`MPI_INFO_NULL` leaves every collective-buffering hint untouched.

---

## 4. Serial remeshing quality

`leveque_remeshing_singleproc`, r = 0.15 sphere, 5120 triangles, time-reversed
vortex, T = 3, 300 steps, target = initial mean edge.

| finding | effect |
|---|---|
| spurious feature detection on a smooth surface | **dominant**: min q 0.76 → 0.0002, 4784 → 16 702 triangles at 45° |
| smoothing is what maintains the mesh | without it: min q 0, **65 % volume loss** |
| collapse fold guard | a brake, not a cause: worthless on a correct setup, 101 048 vs 16 702 triangles once features are spurious |
| moving ridge vertices to the chord midpoint | erodes a 24-gon ridge to 0.84 of its radius in 20 sweeps; invisible here |
| unweighted "triangle-based" smoothing | not a quality bug; doubles volume loss (3.5 % → 6.0 %) |

**Feature detection is the cause, and it is dose-dependent**: at 45° four edges
cross the threshold at step 10, the worst triangle drops from 0.797 to 0.045 and
freezes, because a constrained vertex is exempt from smoothing, collapse and flip
at once and the sliver keeps itself classified — a ratchet to 80 % constrained
vertices by step 300. Refusing to lock corners makes it six times worse: *any*
degree of freedom removed on false evidence is harmful. **Set
`feature_angle_threshold` to 180 for any smooth closed interface**; a mesh
boundary stays a feature at every threshold.

---

## 5. Distributed remeshing

`leveque_remeshing_parallel`: 150 steps, remeshed every step, unit cube with walls.
What the two drivers do is in `src/remeshing/README.md`; this section measures.

### 5.1 The phase budget is the parameter that matters

| budget | np = 1 phases | V/V₀ | min q | ms/step | np = 4 V/V₀ | np = 4 min q | np = 4 ms/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.00 | 0.769 | 0.856 | 20.4 | 0.745 | 0.897 | 12.9 |
| 4 | 3.69 | 0.733 | 0.836 | 37.8 | **0.771** | **0.879** | **16.7** |
| 8 | 5.73 | 0.683 | 0.459 | 52.7 | 0.670 | 0.796 | 35.4 |
| 32 | 13.29 | 0.682 | 0.278 | 99.0 | 0.639 | 0.621 | 84.7 |

Serial reference: V/V₀ 0.751, min q 0.876, 7.1 ms/step. Re-converging an
already-good mesh every step is split/collapse churn that erodes an
under-resolved surface, and every phase is a migration. **A solver loop should
set 2–8** (4 was the best compromise; 8 is a full covering set) and treat
`converged == false` as normal. The default stays 32 because the one-shot case
fails loudly when the budget is too small and the loop case fails quietly when it
is too large.

### 5.2 Correctness against the serial answer

Budget 4, refine 3: every run has Euler characteristic 2, no inverted triangle,
< 0.2 % of edges out of band; V/V₀ 0.718–0.771 across np = 1–8, not monotone in P
(different edge choices on an under-resolved filament, not a bias). Edge-length
histogram total variation against a converged serial reference: **0.0000** at
np = 1, 0.027–0.044 above. At refine 4 volume is conserved to 3.4–4.1 % flat in
P, and **every distributed run beats the serial one** (min q 0.85 vs 0.72, four
times fewer out-of-band edges): the ~3.9× extra remeshing work is free quality
when the mesh can resolve the surface and erosion when it cannot — match the
budget to the resolution.

### 5.3 Determinism

Bit-for-bit identical digests on repeated runs at np = 1, 2, 4 (`f45a22917d6f42d0`,
`d61ab5fdc0d81af4`, `0275cc47ff9d5fa4`, 40 steps, budget 4), because no rank ever
edits an entity another rank holds. This is why the shift/freeze schedule was
chosen over a lock/ask/agree protocol. Digests are compared only within a rank
count.

### 5.4 Phases and scalability

At budget 4 the phase count **saturates** (3.83 → 3.96 from np = 2 to 8); each
phase is one migration. Fixed-problem overhead np = 2 → 8: **+39 % (refine 3),
+33 % (refine 4) for four times the ranks** (np = 1 takes the serial path and is
not a scaling datum). 96–99.5 % of the timestep is inside `MFTLRemeshEnsemble`;
ghost sync is 0.3–0.6 %. A shorter sweep (work lists) was not built: run
`examples/remesh_phase_profile.c` on the target machine first — if `t_migrate`
dominates `t_sweep`, it buys nothing.

### 5.5 The length band must be a collective measurement

`MFTLMeanEdgeLength` is local; a rank holding none of a body measures 0 and gets
the band `[0, 0]`. Measured: a 105-triangle cap grew to **148 287 triangles in
eight phases** with every invariant intact. Closed by
`MFTLEnsembleMeanEdgeLength` and by `MFTLRemeshAgreeParams`, one Allreduce per
call that refuses on every rank a band invalid on any rank or a parameter block
that differs between ranks (`test_the_edge_length_band_is_a_collective_measurement`,
`test_a_band_bad_on_one_rank_is_refused_on_every_rank`). A per-rank refusal
would return on the bad rank and leave the others in the next collective. The
check sits at each path's first collective, after the local work before it: at
the top of the call it was a new synchronisation point and cost +0.8 % of a
LeVeque step at np = 4 (load imbalance of that work, not the 0.7 µs
reduction); placed there, a 40-round interleaved A/B gives ×0.997.

### 5.6 Coarsening, the gather, and the frozen share

The frozen share is the share of every migration that buys nothing. Measured on
an r = 0.7 sphere: 45 / 24 % (np = 8, refine 2 / 3), 66 / 37 / 20 % (np = 27,
refine 2 / 3 / 4). The geometric model `1−(1−2δ/h)³` **over-predicts by about 2×**
(71 % vs 36.6 %), so set `coarsen_frozen_fraction` from the measured field. The
band condition `h > 4δ` itself permits 87.5 % frozen.

- **Band trigger** (always armed): not negotiable; the alternative is an
  un-remeshed lattice. Arming the decision costs about +0.5 % of a timestep and is
  a digest-identical no-op on a healthy partition.
- **Frozen trigger** (off): at 36.6 % frozen it turns a pass that did not converge
  and hid 32 edges into one that converged and hid none, **for twice the wall
  clock**. Useful window roughly 30–60 %.
- **Single-rank gather** (off): exactly two migrations whatever the phase count,
  nothing frozen, always converges, 2.8× faster at 3 triangles/rank; the
  crossover is **~50–100 triangles per rank**, which an absolute count cannot
  express (use ≈ 50 × nproc, capped by one rank's memory).

### 5.7 The dispersed load balancer

`dispersed_balance_slack` 1.25, on. Clustered droplets (4P bodies in one rank's
box): imbalance **2.00 / 4.00 / 8.00 / 27.00 → 1.25** at np = 2 / 4 / 8 / 27, no
rank idle, 1.45 / 2.5 / 5.0 / 3.3× less wall clock; neutral on a lattice where
nothing needs fixing. One rank — the owner of the reduced centroid — decides each
body and publishes `(gid, dest, weight)` one hop along `comm_graph`: `O(local
bodies × degree)` payload, 25 / 49 / 85 doubles on a non-holder at np = 2 / 4 / 8
and unchanged when the ensemble doubles (`test_the_balancer_payload_ignores_bodies_a_rank_does_not_hold`),
against `4B` doubles on every rank for the replicated greedy it replaced. Two
failures of the local rule were found by measurement: a proportional ask returns
grants smaller than one body (108 bodies over 27 ranks stalled at 2.00), and an
identical tie-break starves one rank (now broken on the neighbour's relative cell
offset). Two ask/grant/ship rounds suffice everywhere measured; three ship. The
balancer decides who remeshes, not who holds the mesh afterwards.

---

## 6. Scaling and the per-body collectives

### 6.1 Method

`parallel_advection_scaling` + `run_scaling.py` (+ `scaling/` for clusters).
Strong, weak (the domain grows with the grid, so the ideal is flat) and
**granularity** (same triangles, different body count — a family, not a law). Six
design rules, each against a trap: report the rank-0 generation peak and never
fold setup into a total; keep migration counting off in timed runs (it warms the
caches); exclude the first redistribute (a global scatter); flag phases below
100·`MPI_Wtick`; run repeats round-robin over rank counts; place bodies on a
jittered lattice (Poisson noise otherwise masquerades as overhead). Give each
commit its own `--results` directory.

**Generation is rank-0 serial** (`3440 · 4^refine + 128` B/body; the benchmark
used a 1.39× too-high constant until it was re-derived from the pool layouts).
Batching is not the fix (`O(N²/K)` setup for ~30 % of the peak). The stranded
capacity after the scatter is: bodies rank 0 keeps nothing of are released;
slivers are recovered by `MFTLObjectShrinkToFit` (slack 4: −8.8 % capacity for
~2 % of the phase, so it ships off). Measure memory in current RSS, never with a
byte counter or `ru_maxrss`.

### 6.2 Shakedown and granularity

Strong, 327 680 triangles, redistributing every step: 8.2 / 11.2 / 11.8 / 21.7
ms/step at np = 1 / 2 / 4 / 8, communication share 39 % → 88 % — negative on a
four-core laptop, as expected. The iso-triangle ladder at np = 4: the **same work
costs about 2× more as 4096 bodies of 80 triangles than as 64 of 5120**; the halo
fraction falls 3.0 → 1.0 % as bodies grow while imbalance rises.

### 6.3 Spanning against dispersed

np = 2–4, `--cuts P 1 1`, 81 920 → 5 242 880 triangles:

- a spanning body costs **1.5–2.2× more per triangle**, widening with size;
- redistribution's share of a dispersed step falls **54 % → 39 %** over 64× of
  mesh and a spanning one rises 62 % → 67 %; strong speedup 2 → 4 ranks at 5.2 M
  triangles is 1.17 dispersed and **0.98** spanning;
- for a spanning body the ghost sync doubles and the redistribution is flat with P:
  both are sized by the cut planes through one surface, not by the share;
- above ~300 k triangles per rank the weak curve measures this laptop's memory
  controller (a communication-free phase slows 1.45×) and must not be quoted;
- a square grid is a wash for dispersed strong, 1.10–1.34× slower for weak (more
  halo) and 1.09–1.14× slower for a large spanning body *with less halo*.

Open: does the dispersed share keep falling below 39 % at tens of thousands of
triangles per rank?

### 6.4 The sparse object list

A rank holds an `MFTL_Object` exactly for the bodies it holds entities of
(`local_objects`, gid-sorted); `n_bodies_global` is the id space. Differential
against the replicated list, same session:

| weak, 32 bodies/rank | body visits/rank/step | empty bodies |
|---|---|---|
| np = 2 | 64 → **36.4** (1.76×) | 2.4 → 2.1 % |
| np = 4 | 128 → **41.0** (3.12×) | 18.5 → 4.3 % |
| np = 8 | 256 → **46.8** (5.48×) | 34.6 → 6.3 % |

Per-rank body work went from tracking P to nearly flat; strong scaling is
unchanged (0.99–1.01×). **Wall clock did not move** (±28 % noise floor; four
cores reduce through shared memory): what was removed is memory and message
volume per rank, which a laptop cannot see. No term proportional to
`n_bodies_global` remains on a per-timestep or per-remesh path. The API
consequence — an `MFTL_Object*`, list slot or local count is invalid across a
redistribution — is in §7.

### 6.5 Per-body reductions over the holders

`MFTLEnsembleReduceBodies` / `MFTLEnsembleExscanBodies` reduce by LOCAL slot
over each body's holders: one neighbourhood allgather of `(gid, payload)`
records, folded in ascending source-rank order so every holder gets the same
bits. `MFTLEnsembleUpdate` used to stage 176 B/body (176 MB at a million
bodies) to learn about the bodies it holds.

**Bytes are worse below ~45 ranks and better above**: the payload is
`b_local·(1+d)` records against `B_global`, ratio ≈ `1.4·(1+s)·(1+d)/P`; measured
1.97× → 1.70× more at np = 2 → 27, modelled 22× less at P = 1000. Per consumer the
crossover is P ≈ 35 (quality), 96 (counts), 4100 (the skip vote, whose former
bitset is a very good encoding). The route is invisible to any test of the answer,
so `MFTLEnsembleGetBodyReduceStats` counts it: a droplet timestep takes 3 holder
reductions and 0 global, a droplet remesh 11 and 0 (a pinned scope once staged
`O(B)` per phase and passed every digest).

**The route is certified, and the first certificate was wrong.** The holder route
is exact iff every two holders of a body are graph neighbours. The old test —
visible ghost owners are neighbours and span one cell — is necessary but not
sufficient: a tube bent around a corner of a 3×2 grid passes it on every rank
while its two ends never hear from each other (386 of 484 triangles, overlapping
id ranges). Now: (1) every holder owns an entity and (2) every ghost owner is a
neighbour, which on a connected surface makes the holders connected on the graph;
(3) **after** the exchange, each holder checks that it and every neighbour that
published a record for the body span at most one cell per axis — a connected set
with that property is a clique. The verdict of all three rides the one int the
route decision always cost, with the exact `comm_world` route as the fallback
(`test_a_bent_body_is_certified_after_the_exchange`, np = 27). Assumption: a body
is one connected surface.

### 6.6 The contact broad phase

Pairs are derived, not enumerated: each rank publishes nine doubles per body it
holds (centre, half-width, radius, envelope) one hop along `comm_graph`, and the
arrivals are binned into a uniform grid with cells ≥ `2·max_half + max_envelope`.
Six touching couples among twelve bodies: 6 / 6 / 9 / 18 / 6 candidate pairs at
np = 1 / 2 / 4 / 8 / 27 against 66–1782 for all pairs; the old enumeration was
134 million records at 16 384 bodies. Extents are centre + half-width on the
circle, never `(lo, hi)`, because a box both misses seam contacts and over-sends
(314 vs 56 triangles at np = 27).

---

## 7. Silent failures found, and what guards each

Every one produced a valid mesh or a plausible number.

| failure | symptom | guard |
|---|---|---|
| caller holds an `MFTL_Object*` across a redistribution | V/V₀ = 0.074 after 12 steps at np = 2, every invariant intact; correct at np = 1 | `test_leveque_parallel_conserves` (np = 2) |
| guard on a *local* body count before a collective | deadlock once lists became sparse | every such guard reads `n_bodies_global` |
| switch derived from a local count changes a message count | ranks post different numbers of `Alltoallv` | "does this call have an ensemble" |
| empty holder (generated ensemble) | never discovered, keeps its own buffer | veto in the pre-exchange scan |
| `comm_graph` self-loop on `{2,1,1}` / `{2,2,1}` | every SUM doubled, MIN/MAX fine | self blocks dropped from the fold |
| per-body verdict summed over ranks | "4 of 1 bodies" | `lowest_holder[o] == rank` licence |
| pinned `SCOPE_GLOBAL` on the per-phase id scan | `O(B)` staging per phase, digests identical | route counters, `test_a_droplet_remesh_takes_no_global_reduction` |
| holder route certified before the exchange | wrong sums, overlapping id ranges on a bent body | post-exchange certificate (§6.5) |
| rank-0 `MPI_Exscan` output assumed zero | undefined by MPI; overlapping ids / file offsets on another MPI | zeroed after the wait (ensemble scan, VTU writer) |
| local length band | 148 287 triangles from 105 | §5.5 |
| shifted cut plane truncated instead of floored | a permanently un-remeshed seam band | `test_a_sphere_on_the_periodic_seam_remeshes_everywhere` |
| arithmetic mean of a seam-straddling body | measured as half the domain, classified spanning forever | circular mean, `test_a_seam_straddling_droplet_is_not_measured_as_domain_sized` |
| marching-cubes vertex from unwrapped node indices | holes and χ = 15 at np = 2 on a periodic seam | `test_seam_vertices_get_one_owner_on_every_rank` |
| absolute degeneracy thresholds | short normals, `q > 1`, no coarsening below ℓ ≈ 1.8e-4 | §1.7 tests |
| broad-phase grid sized before its cap | `long` overflow, heap overflow with two 1e-7 triangles | `test_a_micron_contact_does_not_overflow_the_broad_phase` |

---

## 8. Architectures evaluated and not taken

**8.1 A stateful delta redistribution engine** (sparse/dense action batches,
validate-then-apply, cached owner→copy schedules). Ceiling ~1.6× on the phase
(§2.2); the body skip (§2.4) took most of it for a small fraction of the
complexity. Revisit only if a cluster profile shows the per-entity buckets
dominating a spanning run.

**8.2 Ghost synchronisation as a push.** The pull costs three collectives and 40
B/ghost. *Which* local vertices a neighbour needs is exactly derivable; *where*
they go is not locally derivable. Broadcast to all neighbours: 2 collectives but
~640 B/ghost modelled at 26 neighbours. Geometric skin: needs `dt·v_max` from the
caller. **An exact cached schedule push** (1 collective, 24 B/ghost) is the likely
final design, owned by its own module with a partition generation; it pays only
when a solver redistributes less than once per step. Not built.

**8.3 An ensemble-wide quality aggregate** instead of per-body statistics: 1.21×
at 16 384 bodies on a laptop, but a solver needs to know *which* body degenerated.
Reverted in favour of §6.5.

**8.4 Batched generation** (§6.1), **work-list remeshing sweeps** (§5.4),
**a vertex→triangle CSR** (§1.5), **split or padded position layouts** (§1.4),
**persistent triangle frames** (§1.2): each measured and rejected.

---

## 9. Open issues

Correctness, most severe first:

1. **Contacts beyond one graph hop are missed silently.** The spatial halo and
   the radius publish reach one hop; a proximity envelope wider than a rank box
   (or a body larger than one) loses cross-rank contacts after redistribution
   with every call succeeding. Reproduced on 4 ranks, `cuts {4,1,1}`, domain
   `[0,4]×[0,1]²`, one small triangle per body in the planes x = 0.9 and
   x = 2.1, `proximity_factor = 0`, `proximity_floor = 1.3`: one contact before
   `MFTLRedistribute`, zero after. Minimum fix: reject an unsupported reach
   collectively; full fix: multi-hop forwarding or an expanded graph.
2. **Non-manifold input is accepted.** The builder rejects out-of-range and
   repeated vertices, but an edge with three faces or a same-direction duplicate
   is paired partially and reported as an open boundary by
   `MFTLCheckMeshIntegrity`.
3. **A failed ghost-id sync during a spanning remesh** still renumbers owned ids
   and proceeds into the closing redistribution; it should be an agreed
   fail-stop.
4. **BVH float boxes are absolute**, so a unit triangle translated to 1e9 gets a
   256× looser fallback radius and a much wider broad phase (the ensemble radius
   hint repairs the tolerance, not the boxes).
5. **Segment–segment distance mishandles zero-length segments** (squared distance
   3 instead of 2 for a point-degenerate triangle).
6. `MFTLUpdateGhostPositions` has no global fallback; it now *reports* orphaned
   ghosts as an error, and is safe only because every coarsened partition is
   undone before the next timestep.
7. VTU output is Int32: more than 2³¹ points or ~716 M triangles is refused
   collectively, not written.

Scaling and engineering:

- **Nothing above four ranks on real hardware.** Specifically: the holder route's
  P ≈ 45 crossover, neighbourhood vs `comm_world` collectives at P = 10⁴, the
  shared-file I/O layout on a parallel filesystem, Morton ordering on small-L2
  x86.
- **`compute_owners` is the largest non-collective bucket** (30 % of the phase with
  the skip on) and the skip cannot touch it. A per-body slack against a
  caller-supplied motion bound (`dt·v_max`) would prove "no entity crossed a
  plane" without scanning; it must ship off, with a Debug verifier.
- `O(nproc)` metadata remains per call: rank-indexed tables in
  `redistribute_core`, the ghost sync's inverse neighbour table,
  `MFTLRemeshSyncGhostIds`, the spatial halo's send tables.
- Per-call allocation in the body reductions and the ghost-exchange schedule
  rebuilt every step (both belong in persistent workspaces).
- Generation is rank-0 serial and is the setup Amdahl ceiling.
- A route choice is visible only to the route counters; two gate tests cover the
  sites a droplet ensemble reaches.
- The uniform broad-phase grid collapses to one cell when one body spans the
  domain (a cliff, not a slope); a two-level grid would remove it.
- The balancer moves weight one cell per pass.
- Contact detection is discrete-time (no CCD), rebuilds every BVH per call, and
  the narrow-phase batch is not vectorised; nothing in `src/bvh/` has been
  profiled in time.
- A topology change re-triangulates its whole cluster; detection queries the
  whole ensemble every call; `n_bodies_global` counts retired ids (one heuristic
  reads it as a body count).
- Integer limits: counts, ids and MPI counts are `int`; unchecked `3·n`
  arithmetic; MPI errors are fatal by default while the API advertises
  `MFTL_ERROR`. Thread safety is undefined (lazy predicate initialisation,
  process-global statistics).
- `MFTLDetectFeatures` classifies ghost vertices over partial fans; harmless
  today (ghosts are frozen) and untested.

---

## 10. Measurement traps

Each of these produced a plausible wrong answer first.

1. **Blocks instead of interleaving.** One binary measured 9.4 ms/step right
   after the test suite and 6.1 on an idle machine; two collections twenty
   minutes apart differed by 2.7× on identical binaries; and a single collection drifts upward
   by 16 % over six reps. Alternate the variants within one run, form ratios
   within a rep, report how many reps agree in sign, and always print a control
   phase the change cannot touch.
2. **Short runs measure the scheduler**: 50 steps spread 1.5× between identical
   binaries; 200 steps spread 1.05×.
3. **A cold first run of the baseline** pays page faults its repeats do not (a
   "9.7×" that was 5.1× on every repeat).
4. **An instrumented binary against an uninstrumented one** — the per-body
   timeline reads the clock ten times per body.
5. **Probing locality with a store-bound or dependency-bound loop** (§1.3): the
   store stream or the FP adder hides the gather.
6. **Benchmarking through an address-taken struct** — its 15 % was nearly read
   as the gain from unrolling.
7. **`acos(a·b)` to compare nearly identical vectors**: infinite derivative at 1,
   so one ulp reads as 1e-6°. Use the chord `|a − b|`.
8. **A cache-flush setup heavier than the measurement** measured DRAM contention
   (0.73–1.26× swings at four ranks).
9. **Trusting one harness because its number had a story** (§1.2 read 1.03–1.24×
   until two other harnesses disagreed; it was 1.40–1.57×).
10. **Blaming the obvious culprit without measuring** (the MPI-IO truncate).
11. **Geometric models of partition quantities** — the frozen-share model
    over-predicted by 2×, and the skippable-body fraction was estimated at
    60–75 % and measured at 77–86 %.
12. **Laptop weak scaling above ~300 k triangles per rank** measures the memory
    controller, and anything above four ranks measures efficiency cores.
