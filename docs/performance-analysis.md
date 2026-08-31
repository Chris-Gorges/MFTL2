# Performance analysis {#mftl_page_perf}

> Historical note (2026-08-30): the persistent triangle-frame implementation
> measured in several sections below has been removed. Those results are kept as
> the evidence behind that decision, not as a description of the current API.

Everything here was measured with the benchmarks in `examples/` in a Release
build (`python3 build_all.py`); each section names the one it came from.
Reproduce before quoting; the machine matters more than usual for three of these
results.

Machine for the numbers below: Apple M-series, macOS, Apple clang, `-O3` + LTO,
single rank. **This is a laptop-class part with an unusually large L2, and two of
the findings below almost certainly do not transfer to an x86 cluster node.**
They are marked.

---

## 1. Struct sizes

| type | size | what is *not* in it |
|---|---|---|
| `MFTL_Vertex` | **16 B** | `pos` (SoA), `is_ghost` (the prefix), the per-vertex geometry (SoA) |
| `MFTL_Triangle` | **12 B** | `area`, `normal`, `center` (SoA), the three corner indices (derived) |
| `MFTL_HalfEdge` | **16 B** | `he_idx`, `next_idx`, `triangle_idx` (arithmetic), `boundary` (a flag bit) |

Every omission in the third column is a field that would otherwise be a *second
copy of a fact the record already determines*. `he_idx`, `next_idx` and
`triangle_idx` are arithmetic on the half-edge index (implicit indexing, see
`mftl_topology.h`); `pos` is authoritative in the vertex SoA and nowhere else;
`is_ghost` is the array prefix; and `boundary` would be an exact mirror of
`twin_idx == MFTL_ERROR` inside one partition, which is the wrong statement
anyway (@ref inv_boundary).

Per triangle the AoS footprint is therefore `12 + 3*16 = 60 B`. A half-edge
record carrying its own index, next, triangle and boundary flag is 24 B, which
would make the same triangle `32 + 3*24 = 104 B` — 1.7× the traffic for the same
mesh, in the loops that stream it every timestep.

## 2. Geometry: half-edge walk vs. streaming frames

Both write the same three arrays, in the same binary, on the same mesh. `max
|diff|` is the largest disagreement between their outputs, and it is **exactly
zero everywhere** — the two are the same arithmetic with the gather hoisted out,
not an approximation of each other. `tests/test_geometry.c` asserts that with
`memcmp` rather than a tolerance.

| domain | refinement | triangles | walk ns/tri | frame ns/tri | speedup |
|---|---|---|---|---|---|
| non-periodic | 4 | 5 120 | 2.148 | 1.367 | 1.57× |
| non-periodic | 5 | 20 480 | 2.393 | 1.562 | 1.53× |
| non-periodic | 6 | 81 920 | 2.405 | 1.575 | 1.53× |
| non-periodic | 7 | 327 680 | 2.563 | 2.045 | 1.25× |
| **periodic** | 4 | 5 120 | 3.516 | 1.758 | **2.00×** |
| **periodic** | 5 | 20 480 | 3.711 | 1.904 | **1.95×** |
| **periodic** | 6 | 81 920 | 3.748 | 1.929 | **1.94×** |
| **periodic** | 7 | 327 680 | 3.769 | 1.938 | **1.94×** |

**The periodic row is the one that matters**, because a front-tracking run has a
periodic box. It is also where the frames do the thing they exist for: the walk
applies a minimum-image correction to every edge of every triangle on every pass,
while `tris.edge` had it applied once at frame-build time. The whole margin
between the two columns on a non-periodic domain is the gather; on a periodic one
it is the gather plus that correction, and the correction is the bigger half.

Since `MFTLComputeTriangleGeometry` now DISPATCHES on `MFTL_DERIVED_FRAMES`
(§9), this table is no longer a choice a caller makes — it is what the caller
gets automatically whenever the frames happen to be current.

### This table said 1.03–1.24× until the harness was fixed

The earlier revision of this section reported 1.03×, 1.07×, 1.24×, 1.16× and drew
a conclusion about the frames being worth much less than expected. Two thirds of
that was the measurement.

`examples/geometry_update_benchmark.c` timed the two variants in BLOCKS — twenty
consecutive passes of the walk, then twenty of the frame kernel — and averaged
within each block. Both halves are wrong in the same direction as trap #6 in §11:
whichever variant runs second meets a cache and a clock the first one left
behind, and averaging lets one descheduled pass contaminate a whole trial instead
of being discarded by the minimum. Interleaving the two and taking the minimum of
individual runs moved the non-periodic figure from 1.05× to 1.40–1.57× on the
identical mesh and binary.

It was caught because two other harnesses disagreed with it: this file's own
§9 benchmark and a third written specifically to arbitrate. **Three harnesses
that agree are worth more than one that is trusted**, and the disagreement was
the only signal that anything was wrong — the wrong number was entirely
plausible, and it had a ready-made explanation attached to it.

What survives from that earlier conclusion is the half that was never about the
harness: the size of this margin is set by how fast the *general* path is, not by
how fast the frames are. A walk that chases pointers through 96-byte records
makes the frames look worth 3×; the walk here dereferences nothing and strides
12-byte records, so it closes most of that gap on its own — just not all of it.

## 3. Morton renumbering — a negative result

Measured on a deliberately gather-bound kernel: three position reads per triangle
through `tris.verts`, one 8-byte store out, no loop-carried dependency.

| refinement | triangles | `verts.pos` | mean edge index distance | | ns/tri | | speedup |
|---|---|---|---|---|---|---|---|
| | | | before | after | before | after | |
| 4 | 5 120 | 60 KiB | 666.6 | 61.2 | 0.850 | 0.840 | 1.01× |
| 5 | 20 480 | 240 KiB | 2 615.2 | 125.3 | 0.869 | 0.886 | 0.98× |
| 6 | 81 920 | 960 KiB | 10 353.1 | 253.5 | 0.884 | 0.868 | 1.02× |
| 7 | 327 680 | 3.8 MiB | 41 189.1 | 510.7 | 0.843 | 0.849 | 0.99× |
| 8 | 1 310 720 | 15 MiB | 164 301.7 | 1 025.5 | 0.857 | 0.860 | 1.00× |

`MFTLObjectReorderMorton` reduces the mean edge index distance by a factor of
~160, exactly as designed — and **buys no measurable time on this machine at any
size up to a 15 MiB position array.**

The most likely explanation is cache capacity: Apple M-series L2 is large enough
that even the scattered pattern mostly hits, so there is no miss for locality to
remove. **An x86 cluster node with 1–2 MiB of L2 per core is a different
experiment and is the one that decides whether the reorder earns its cost.** Run
this benchmark there before either keeping it in a per-timestep loop or removing
it from the library.

What the reorder is worth independently of this measurement:

- it is what lets the SFC ordering be the *pool* ordering, which is what removes
  the second index space (`sfc_perm` / `sfc_inv`) and the duplicated position
  array a stored permutation would need. That simplification stands regardless of
  the timing.
- it is a prerequisite for a linear BVH build, where the Morton codes are the
  input rather than a locality hint.

### Two measurement mistakes made while producing that table

Both are recorded because each produced a *plausible* wrong answer first.

1. **Probing with `MFTLObjectRefreshFrames`.** That loop writes 72 B per triangle
   sequentially; the store stream hides the gather it was supposed to expose.
   Measured that way the reorder appeared to be worth 0.97×, 1.93× and 0.97× at
   three consecutive refinements — no pattern, i.e. noise presented as a result.
2. **Accumulating into a scalar.** `acc += ...` carries a dependency through the
   FP adder, ~4 cycles per triangle, and nine gathered loads fit underneath it.
   The kernel then measures the adder and reports 1.00× however the pool is
   ordered — the right answer for the wrong reason, which is worse than a wrong
   one.

Timings are the **minimum** of five runs, not the mean: a single descheduled run
should not decide the answer.

## 4. Position layout: interleaved, padded, or split?

`examples/layout_benchmark.c`. `verts.pos` is interleaved (`pos[3v+k]`); the
alternatives are three component arrays (`split`) and a 32-byte-per-vertex
interleave (`pad4`). All three are cut from one cache-line-aligned block, and
every pair is checked to produce bit-identical results before its timings count.

ns per element, Morton-ordered pool, ref 5–8 (ref 4 is warm-up):

| pattern | what it is | pack | pad4 | split | winner |
|---|---|---|---|---|---|
| P1 stream-reduce | bounding box over `verts.pos` | 0.49–0.55 | 0.49–0.55 | 0.50–0.56 | tie |
| P2 stream-map | translate every vertex | 0.34–0.37 | 0.51–0.53 | **0.34–0.35** | split |
| P3 gather | `build_frames`, 3 positions per triangle | 1.93–2.28 | 1.91–2.26 | **1.89–1.90** | split |
| P4 scatter-acc | vertex normals, scatter into 3 slots | **2.22–2.27** | 2.22–2.28 | 2.73–2.77 | pack |

**The four patterns do not agree, and the mechanism is different in each — read
from the disassembly rather than inferred.**

- **P1 is a tie because the layout never enters the answer.** The loop is bound
  by six `fmin`/`fmax` per vertex, not by memory.
- **P2 favours split** while the data is in cache and levels off once it is not.
  This is the only pattern where per-component vectorisation applies. `pad4` is
  last everywhere here: a third more bytes for nothing.
- **P3 favours split because of ADDRESS GENERATION, not cache lines.** A stride
  of 3 doubles is 24 B, which no addressing mode can scale, so every gathered
  vertex compiles to an `smaddl` — a multiply — in front of its load:

  ```asm
  smaddl  x14, w14, w13, x9      ; packed: index * 24 + base
  ldr     d0, [x14]
  ...
  ldr     d0, [x9, x15, lsl #3]  ; split: one instruction, no multiply
  ```

  `pad4` is the control that confirms it: 32 B *is* scalable (`add x13, x9, x13,
  lsl #5`), and `pad4` duly matches `split` at small sizes and then falls back at
  large ones where its extra traffic dominates.
- **P4 favours packed, and it is the mirror image.** The read-modify-write of
  three contiguous doubles compiles to `ldr q` / `fadd.2d` / `str q` for x,y plus
  a scalar pair for z — four memory ops per vertex. Split needs three separate
  load-add-store triples: six. Scattering into one array pairs up; scattering
  into three cannot.

**Decision: `verts.pos` stays interleaved, and so do `tris.normal` /
`tris.center`.** The two patterns that run every timestep — P3 (`build_frames`)
and P4 (vertex normals) — pull in opposite directions by comparable amounts
(split +1.02–1.20× on the gather, −1.22× on the scatter), so the layout question
is close to a wash on time. That makes it an API question instead, and
interleaved wins that outright: `MFTLVertexPos()` returns a plain
`const double[3]`, which is what every geometry routine in the library takes.
Splitting would rewrite `MFTLGetMinimumImageVector`, `MFTLApplyPeriodicity`,
`MFTLGetOwnerProc`, every generator and every I/O path for a measured net of
roughly zero.

**`pad4` is rejected outright.** It wins only where address generation binds and
loses where traffic binds, and this library is heading towards large meshes.

## 5. Vertex normals: scatter beats gather by 3.4×

`examples/vertex_normal_benchmark.c`, ns per vertex:

| ref | vertices | A scatter | B fan gather | C CSR gather | CSR rebuild |
|---|---|---|---|---|---|
| 5 | 10 242 | **5.35** | 17.84 | 5.28 | 5.65 |
| 6 | 40 962 | **5.24** | 18.00 | 5.29 | 5.68 |
| 7 | 163 842 | **5.24** | 18.05 | 5.33 | 5.66 |
| 8 | 655 362 | **5.27** | 18.44 | 5.41 | 5.70 |

Flat in mesh size, and all three agree to 4.4e-16.

The fan walk is measured **with** its fast path — skipping `MFTLVertexFanStart`
on every vertex not flagged `MFTL_FLAG_BOUNDARY`. Without it, it is 8× slower
rather than 3.4×, because the rewind walks a whole extra lap to discover that a
closed fan is closed. What remains after that is dependent loads: each hop is
`he -> twin -> next` and cannot issue its address until the previous hop has
returned, so a valence-6 fan serialises six round trips the scatter never makes.

**This also settles the CSR question left open in `src/object/derived.c`.** The
vertex→triangle adjacency was deleted from this version for want of a consumer;
vertex normals are the first plausible one, so it was measured rather than
assumed. A CSR gather removes the fan walk's dependent loads entirely and lands
in a dead heat with the scatter — not a win — while costing 16 B per triangle of
permanent state and a rebuild as expensive as the whole normal pass on every
topology change. **The CSR stays deleted, now for a measured reason.**

## 6. The update kernel, taken apart

`examples/update_kernel_benchmark.c` rebuilds `update_kernel` eleven ways on one
mesh in one binary. ns per triangle, ref 5–8 — **flat across a 256× range in mesh
size, so this loop is not memory bound at any size that fits in RAM**:

| variant | ns/tri | verdict |
|---|---|---|
| U0 shipping (fused, interleaved out) | **3.45–3.46** | baseline |
| U1 geometry only | 2.25–2.26 | statistics cost 1.19 |
| U2 statistics only | 2.73–2.74 | |
| U3 U1 then U2, two passes | 4.29–4.61 | **splitting costs 25 %** |
| U4 fused, split output arrays | 3.40–3.42 | 1.3 %, not worth an API change |
| U5 via a step function, ×1 | 3.95–3.98 | the abstraction costs 15 % |
| U6 / U7 step function, ×2 / ×4 | 3.64–3.68 | recovers half of U5's penalty |
| U8 **2-way unroll, plain scalars** | 3.52–3.54 | **a wash vs U0, slightly worse** |
| U10 periodicity wrap unswitched away | 3.63–3.65 | not faster; below the noise |
| U11 vectorisation pragma forced | 3.44–3.46 | identical — see below |
| U9 **probe**, sqrt+divide deleted | 1.33–1.91 | **they cost 0.90, i.e. 26 %** |

### Vectorisation: asked, forced, and priced

The obvious diagnosis is that the loop fails to vectorise because of the nine-way
statistics reduction. **That attribution is wrong.** Three separate results say
so:

1. Asked directly — and note `-Rpass-missed`, without which clang says *nothing*
   about this loop and only remarks on the unpack loop at the bottom of the
   file — clang reports *"the cost-model indicates that vectorization is not
   beneficial"*. That is a profitability verdict, not a legality one.
2. Splitting the reduction into two independent accumulator sets, written with
   eighteen plain scalars so nothing hides in a struct, measures **3.53 against
   3.45**. The standard fix for a reduction bottleneck makes it slightly worse,
   which means the reduction was not the bottleneck.
3. Overruling the cost model with `#pragma clang loop vectorize(enable)` yields
   *"loop not vectorized: the optimizer was unable to perform the requested
   transformation"* and times identically. So the loop is not merely judged
   unprofitable — in this shape it is not vectorisable at all.

**Where the time actually goes: one `sqrt` and one reciprocal per triangle are
26 % of the whole pass**, and neither is algebraically removable — the kernel
needs both `|n|` for the area and `n/|n|` for the unit normal. That is a
throughput limit on one functional unit, which is exactly why unrolling the
reductions changes nothing.

**Conclusion: there is nothing left in vectorisation here, and the reason is not
that it is too complicated.** Each of the three changes that would enable it
measures neutral-to-negative on its own (U4 +1.3 %, U8 −2 %, U10 −5 %), and the
compiler declines the transformation even when told to do it anyway.

### And the answer to "should quality statistics be separated from area/normal/center?"

**No — measured at 25 %.** Two passes cost 4.29–4.61 against 3.45 fused, *even
though* the second pass re-reads data the first left in cache. The statistics
ride along on vectors already in registers; separating them means recomputing the
cross product and the three edge lengths from scratch.

## 7. Redistribution: what a distributed timestep costs, and what it defers

Measured with `examples/parallel_advection_redistribute_list.c`. Four ranks on a
2×2×1 grid of a fully periodic 8×4×4 box, spheres of radius 0.1 translating
rigidly, Release + LTO. **Minimum of three runs** — see the trap at the end of
this section, because the first number produced here was wrong by a factor of
two.

| per step, 4 ranks | 256 bodies, 327 680 tris | 1000 bodies, 1 280 000 tris |
|---|---|---|
| redistribute | **2.71 ms** | **10.67 ms** |
| redistribute + the geometry it defers | **3.25 ms** | **13.19 ms** |
| ghost sync | **0.32 ms** | **0.83 ms** |
| wall clock (whole run) | **0.42 s** (100 steps) | **1.03 s** (60 steps) |

### The second row is the one to quote, and it was missing at first

Redistribution deliberately does not put `area`, `normal` and `center` on the
wire — that is what buys point 2 below — and hands back an object whose triangle
geometry is undefined (`include/mftl_communication.h`). A timing of the
`redistribute` row alone therefore credits it with a saving it has not finished
paying for.

`parallel_advection_redistribute_list` runs that pass as step 5 and times it
separately: **0.54 ms and 2.52 ms per step in the two cases, 13–15 % of the
timestep**. The honest headline is the second row.

Both readings are kept because both are real. A solver that consumes triangle
geometry between the ghost sync and the redistribution — the natural place, since
that is the configuration the forces belong to — never looks at it afterwards and
genuinely owes nothing; the next timestep recomputes from the moved positions
regardless. `--no-post-geom` measures that case, and it comes out at 2.62 ms
against 2.71, i.e. the redistribution phase itself does not care either way.

### Where the time is not spent, and why

Three design decisions dominate this number, and only one of them is about bytes
on the wire.

**1. Entities that stay never enter the wire format — the largest of the three.**
Treating "stays here" as just another destination means packing into an
`MFTL_Packed*`, concatenating kept with received, radix-sorting the whole merged
stream by global id, de-duplicating, unpacking, and inserting one hash entry per
vertex and four per triangle. On a steadily advecting mesh the kept set is ~99 %
of the partition and the arriving set is a one-dimensional rim, so almost all of
that work is a round trip that moves nothing. Here the kept entities are marked
during the same destination analysis that decides what to send, and rebuilt with
a permutation: a gather into scratch and two `memcpy`s.

**2. Only topology travels.** Every recomputable quantity is off the wire:

| | with geometry on the wire | here |
|---|---|---|
| `MFTL_PackedVertex` | 96 B | **40 B** (2.4×) |
| `MFTL_PackedTriangleComplex` | 120 B | **40 B** (3.0×) |
| `MFTL_GhostUpdatePayload` | 80 B | **32 B** (2.5×) |

A vertex position is the only geometric quantity that is authoritative, so it
travels; areas, normals, centres and curvatures are recomputed at the receiver.
What that recompute actually costs is not an estimate — it is the second row of
the table above: **6.4 ns per triangle at 256 bodies and 7.9 ns at 1000**,
against the 2.9 ns §9 measures for the same function on a single hot mesh. The
factor of ~2.5 is the difference between one large object in cache and a thousand
small ones walked in sequence, and it is the honest number for this trade,
because it is the one measured in the loop the trade was made for.

Half-edge global ids are not sent at all — they are `3 · triangle_gid + k`
mesh-wide (`inv_halfedge_gids_are_implicit`), which is the wire-format half of
implicit indexing.

**3. No hash maps, and no full twin rebuild.** Global ids are dense per entity
class, so "have I already got id *g*" is a probe into an epoch-stamped array
rather than a `khash` lookup, and the table is never cleared — bumping the epoch
retires every entry at once. Twins are *repaired* rather than rebuilt: a twin
between two kept triangles survives as an index remap, and the half-edges that
did lose their partner are collected for free by the remap pass that was going to
touch them anyway. `MFTLRebuildTwins` does not run during a redistribution at
all. `test_twin_repair_matches_full_rebuild` re-runs it afterwards and asserts
that it changes nothing.

### What did NOT get fast, and why

Ghost synchronisation improves by roughly its 2.5× payload reduction and no more,
because at these sizes it is not bandwidth-bound: it is three neighbourhood
collectives and the volume is well under a megabyte per step. Nothing maintains a
global-id → slot map for it to inherit, so the owner scatters its owned vertices
into a dense id table on demand, once per object that actually received a
request. That trades a per-timestep `O(owned)` scatter for never
maintaining a map — a good trade here, and it would stop being one if the ghost
sync ran many times per redistribution. See the deferred issue in `CLAUDE.md`:
the schedule could be cached at the end of a redistribution, which would remove
the lookup entirely.

### These absolute numbers are not the ones this section first carried

An earlier revision reported 6.06 ms for the 256-body case, 2.3× faster than the
table above, measured on the same machine with the same binary. The whole machine
was in a different state (thermal, or background load; it was not established
which).

That is the argument for a back-to-back A/B protocol rather than a caveat about
it. **Any measurement taken on a different day than the thing it is compared
against would have been wrong by a factor of two here**, in whichever direction
the day happened to fall — which is why every comparison in this document
interleaves its variants and reports minima, and why none of them are quoted from
runs taken in blocks.

### The measurement trap in this section

A benchmark run immediately after `run_local_tests.py` measured **9.4 ms/step
against 6.1 ms on an idle machine** — 50 % of thermal state and background load,
on the identical binary and the identical mesh. Nothing was re-measured until the
machine was quiescent and every variant could be run consecutively.

The related trap is first touch. A redistribution allocates and frees its
per-destination send buffers on every call (deferred issue 6 in `CLAUDE.md`), so
the *first* run of a benchmark pays page faults the later ones do not — which is
exactly the shape of an "improvement" that evaporates under repetition. Every
figure here is the minimum of three consecutive runs.

### 7b. What is INSIDE the redistribution, measured

Everything above times the phase from outside. Until 2026-08-25 nothing timed it
from inside: `git grep MPI_Wtime -- src/` returned exactly one hit and it was in
`remesh_comm.c`. Every published statement about *which part* of the
redistribution is expensive — including three architecture recommendations in
`docs/` — was inference from the source.

`MFTL_RedistributeStats` (`include/mftl_communication.h`) replaces that with
nineteen disjoint phase buckets and twenty-six counters. The timers are behind
`-DMFTL_COMM_PROFILE=ON` (`python3 build_all.py --comm-profile`) because two of
them are not free; the counters are unconditional. Measured residue between
`t_total` and the sum of the buckets: **1.5 %**, so the taxonomy is complete.

**Where the time goes. 1000 bodies, 1 280 000 triangles, np = 4:**

| bucket | ms/step | share |
|---|---:|---:|
| kept: gather into scratch | 5.72 | 17 % |
| destination analysis | 5.29 | 16 % |
| kept: relink half-edges | 3.85 | 11 % |
| derived: **frames** | 3.60 | 11 % |
| owner classification | 2.72 | 8 % |
| route decision (Allreduce) | 2.25 | 7 % |
| derived: topology arrays | 1.79 | 5 % |
| arrivals | 1.69 | 5 % |
| pool growth | 1.67 | 5 % |
| kept: place back | 1.65 | 5 % |
| exchange: wait | 0.98 | 3 % |
| twin repair | 0.93 | 3 % |
| pack / exchange / grouping / setup / teardown | 1.20 | 4 % |

Two readings of that table are worth stating, because they are what an
architecture decision turns on.

**The block a delta engine would remove is 39 %; the block it would not is 34 %.**
Gather, place-back, relink and the topology rebuild — 13.0 ms — are the work that
exists only because the pool is permuted wholesale on every call, and a planner
that touched only what moved would not do them. Owner classification, destination
analysis and the frame rebuild — 11.6 ms — survive any such rewrite: the first two
are `O(N + H)` by construction, and the third is required by the postcondition.
So the ceiling on the whole proposed delta architecture is about **1.6× on the
phase**, or about 1.2× on the timestep. Worth knowing before spending ten phases
on it.

**The frame rebuild is dead work in the documented loop.** Redistribution builds
`tris.anchor`/`tris.edge` as step 4; step 1 of the next iteration moves owned
vertices, and #MFTL_FOREACH_OWNED_VERTEX cleared `MFTL_DERIVED_FRAMES` on the way
out. Nothing read them in between, and step 3's `MFTLComputeTriangleGeometry`
then took its walk path *because* they were gone. 3.60 ms/step, 11 % of the
phase, 72 B/triangle written for a value that is discarded unread — unless the
caller runs `MFTLEnsembleUpdate` or `MFTLComputeVertexNormals` immediately after
redistribution, which is the only case that benefits.

**This is the one entry in the table above that no longer exists.** It was
resolved not by making the rebuild cheaper but by deleting persistent frames
outright (§9), which is why `MFTL_DERIVED_FRAMES` is named here as plain text: the
enumerator is gone. Everything else in the table is still a live bucket, subject
to §7e and §7f.

**What the counters say about scaling.** These are exact and independent of
machine state, which is what makes them the useful half of the measurement:

| | np = 2 | np = 4 | np = 8 |
|---|---:|---:|---:|
| arrival id table, MB/rank/step, one 327 680-tri body | 15.2 | 31.0 | **62.0** |
| bodies visited with nothing to do, 1000 bodies | 48 % | **71 %** | **84 %** |
| triangles that changed owner | 2.7 % | 4.8 % | 7.9 % |

The first row is a weak-scaling defect, not a constant: the table was
direct-addressed and sized from `next_vertex_gid`, the body's GLOBAL vertex
count, so it grew linearly with the process count at a FIXED problem size while
the entries it held shrank. At np = 8 it was 97 % empty. The second row is the
cost of a replicated object list — `rebuild_object` ran in full for every body in
the ensemble, including an id-table epoch bump, a `MFTLObjectSetCounts(0,0,0,0)`
and a complete `MFTLObjectBuildDerived`, on the 84 % of bodies a rank holds no
part of. The third row is the number that motivates a delta engine at all: **95 %
of the partition is unchanged from one timestep to the next.**

### 7c. Five no-regret fixes: 2–19× less allocator traffic, and no faster

Five changes were then made to the existing engine, chosen because they are
correct regardless of which architecture eventually wins:

1. a persistent communication workspace on `MFTL_Ensemble`, replacing per-call
   allocation of the rank-sized arrays, the per-destination payload buffers and
   the whole rebuild scratch;
2. an id table that picks direct addressing or open-addressed hashing from the
   density of the ids it is given, instead of always direct-addressing the global
   range;
3. geometric rather than exact pool growth on the redistribution path;
4. an early return for bodies a rank holds no part of, plus
   `MFTLObjectReleasePools()` for a body that has left;
5. routing a single-body ensemble through the ensemble entry point, so it reaches
   the workspace at all.

**What they bought, exactly:**

| per timestep, all ranks | before | after | |
|---|---:|---:|---|
| allocations, 1000 bodies np = 4 | 99 | 40 | **2.5×** |
| MB requested, 1000 bodies np = 4 | 20.1 | 7.8 | **2.6×** |
| MB requested, one big body np = 8 | 72.0 | 4.5 | **16.1×** |
| id-table MB, one big body np = 8 | 62.0 | 0.8 | **77×** |
| id-table MB, one big body np = 4 | 31.0 | 0.17 | **183×** |

The id table is now a one-time allocation rather than a per-timestep one, and it
no longer grows with the process count. Partition digests — a GID-keyed checksum
of ownership, positions, flags, connectivity and twin partners, printed by
`parallel_advection_redistribute_list --digest` — are **bit-identical before and
after at np = 1, 2, 4 and 8 on all three cases**, in both the set form and the
slot-order form. The changes are provably behaviour-neutral.

**And they did not make it faster.** Interleaved A/B of the two binaries, both
with the timers compiled out, minimum of five alternating repeats:

| case | np = 1 | np = 2 | np = 4 | np = 8 |
|---|---:|---:|---:|---:|
| 256 bodies | 1.08× | 0.94× | 0.98× | 0.94× |
| 1000 bodies | 1.07× | 0.95× | 0.95× | 0.95× |
| one big body | 1.06× | 0.98× | 0.98× | 0.91× |

The honest reading is **no change**. The control proves it: `triangle geometry`,
a phase none of these five changes touches, moves by 0.89–1.08× across the same
runs — the same spread. Everything in that table is inside the measurement error
of this machine.

The one real signal is the ghost sync on a single large body, **1.10–1.40×**,
which is fix 2 working as designed: the owner-side table is now sized by the
vertices this rank owns rather than by the body's global vertex count.

**Why removing megabytes of allocator traffic bought no time**, which is the
result worth carrying forward: macOS hands out large blocks as lazily-mapped zero
pages, so `calloc` of a multi-megabyte table costs almost nothing until it is
touched, and only the touched fraction ever faults in. The allocation was never
on the critical path; it was a *footprint* problem, and it has been fixed as one.
Re-measure on a cluster before assuming the same holds under a different
allocator and a different page-fault cost.

### 7d. Two mistakes made producing 7b and 7c, both worth repeating

**A 50-step run measures the OS scheduler.** The first comparison runs used 50
timesteps, about 0.4 s of measurement, and produced spreads of 1.5× between
identical binaries — larger than any effect being looked for, and stable enough
across repeats to look like a result. Ten runs of one binary at 200 steps spread
by **1.05×**. Every number in 7c is at 200 steps (100 for the largest case) for
that reason. The related trap: two data sets collected twenty minutes apart on
this machine differed by **2.7× on identical binaries**, so nothing is compared
across collections — the two binaries alternate within each repeat.

**An instrumented binary cannot be compared against an uninstrumented one.** The
first attempt at 7c put the new build (with timers) against the old build
(without) and reported a uniform slowdown. Two of the timers are genuinely
expensive — the per-body timeline reads the clock ten times per BODY, and the
ownership-change counter adds a compare to `compute_owners`' per-entity loops,
which is the loop CLAUDE.md records a 1.70× regression on. That is why
`MFTL_COMM_PROFILE` exists and defaults to off, and why the table in 7c is
measured with it off on both sides.

### 7e. Skipping unchanged bodies: 1.6–2.1×, and the frames rebuild is now the biggest bucket left

The optimisation 7b's table implies but 7c did not attempt: a body for which
neither input to the rebuild changed — the global topology and the owner map —
already holds the answer, so nothing needs doing to it.
`MFTLRedistributeSetSkipUnchangedBodies` (default **on**) leaves such a body
alone, and the reasoning about *when* that is legal is in
`src/communication/README.md` and `docs/redistribution_skip_correctness.md`.

**The number the whole idea turns on had never been measured**, and it is a BODY
fraction, not an entity fraction — the ~95 % of entities that stay put says
nothing about it, because a body straddling a cut plane very likely has *some*
entity cross every step. Measured, 1000 bodies × refine 3 = 1 280 000 triangles,
the same case as 7b:

| | np = 1 | np = 2 | np = 4 |
|---|---:|---:|---:|
| bodies skipped | — (serial short-circuit) | **85.6 %** | **77.2 %** |
| redistribution phase | 1.00× | **2.11×** | **1.63×** |

Interleaved A/B **inside one process** — `--ab 20` alternates the switch every 20
steps and reports both arms, 400 steps, first block of each arm discarded. That
is not a nicety: 7d records two collections twenty minutes apart differing by
2.7× on identical binaries, and this is the first measurement here that could
alternate the two configurations within a single run rather than across binaries.
np = 1 is the control — the serial path short-circuits before any of this, and it
reads 1.004×.

#### What it does to a whole timestep

The phase share is the question a solver actually asks, and 7b put the
redistribution at 60-64 % of a timestep. Same case, np = 4, 200 steps, one run
per arm:

| mean per rank, ms/step | skip off | skip on | |
|---|---:|---:|---:|
| **redistribute** | **12.20 (62.7 %)** | **7.12 (50.3 %)** | **1.71x** |
| post-redistribution geometry | 2.77 (14.2 %) | 3.02 (21.4 %) | — |
| triangle geometry | 3.07 (15.7 %) | 2.99 (21.1 %) | — |
| ghost sync | 0.90 (4.6 %) | 0.52 (3.7 %) | — |
| advection | 0.52 (2.7 %) | 0.49 (3.4 %) | — |
| *redistribute + post ("the honest total")* | *14.97* | *10.14* | *1.48x* |
| **whole timestep** | **19.47** | **14.14** | **1.38x** |

So the redistribution drops from **62.7 % of a timestep to 50.3 %**, and the step
as a whole gets **1.38x faster**. Nothing else moved: the two geometry passes and
the advection are within noise of each other and only their *shares* grew.

**Do not compare the absolute milliseconds in this table against the interleaved
A/B above.** They were collected minutes apart and the un-skipped arm reads
12.20 ms/step here against 26.80 there on the same binary and the same input --
a 2.2x collection-to-collection spread, which is 7d's warning arriving on
schedule. Only ratios taken WITHIN one collection mean anything on this machine,
and the two independent ones agree: 1.63x interleaved, 1.71x here.

**It is provably behaviour-neutral.** Both partition digests — the set form and
the **slot-order** form — are identical with the skip on and off, at np = 4 on
1000 small bodies, at np = 4 on 8 domain-spanning bodies, and at np = 8 on 500
bodies. `tests/test_redistribution.c` asserts the same thing every step of a
24-step run.

**And arming it costs nothing measurable.** The change verdict needs a compare
per entity in `compute_owners`' two loops, which is the loop CLAUDE.md records a
1.70× regression on, so the counters that used to live behind
`MFTL_COMM_PROFILE` are now unconditional. HEAD against the new binary with the
skip switched *off*, alternating within one collection, five repeats, minima:
**26.72 → 26.70 ms/step at np = 4**. The spread within either binary is 1.6 %.

#### Where the time went, and what is left

`-DMFTL_COMM_PROFILE=ON`, np = 4, 200 steps. Absolute values are inflated by the
instrument — 53.5 ms/step against 26.8 un-instrumented, because the per-body
timeline reads the clock ten times per body and there are a thousand of them — so
read the columns against each other, not against 7b.

| bucket, ms/step | skip off | skip on | |
|---|---:|---:|---:|
| kept: relink half-edges | 5.53 | 1.05 | **5.2×** |
| kept: place back | 2.11 | 0.45 | **4.7×** |
| derived: topology | 2.34 | 0.51 | **4.6×** |
| kept: gather | 6.85 | 1.65 | **4.1×** |
| destination analysis | 7.60 | 2.09 | **3.6×** |
| arrivals | 3.01 | 1.45 | 2.1× |
| twin repair | 1.58 | 1.50 | 1.05× |
| pool growth | 1.30 | 1.21 | 1.07× |
| owner classification | 4.19 | 4.15 | 1.01× — the decision itself |
| **derived: FRAMES** | **4.81** | **5.32** | **0.90×** |
| TOTAL | 53.53 | 35.31 | 1.52× |

**Three results are worth carrying forward.**

**The frames rebuild does not shrink, and it is now the largest non-collective
bucket.** A skipped body still pays it, because step 1 of the documented loop
moves owned vertices through `MFTL_FOREACH_OWNED_VERTEX`, which clears
`MFTL_DERIVED_FRAMES` — the vertices really did move even though nobody changed
hands, and the postcondition promises the bit set. It goes from 9.0 % of the
phase to 15.1 % of it. §9 already measures those 72 B/triangle as written and
discarded unread in this exact loop; this makes weakening the postcondition the
obvious next change rather than one of several.

**The buckets that shrink least are the ones proportional to the HALO, not to the
mesh.** Twin repair moves by 1.05× and pool growth by 1.07×, against 3.6–5.2× for
everything proportional to the entity count. The bodies a skip removes are the
ones nothing crossed — interior bodies, with the smallest halo churn — so what is
left is disproportionately halo work. A skip is a reduction in *touched bytes*,
which §7c identifies as the only kind of change that has ever bought time here,
and it reduces them for the per-entity passes rather than for the per-halo ones.

**`skip vote` reads as 13.5–19.2 % of the phase and is not a cost.** It is
7.20 ms/step with the skip switched OFF, where it decides nothing at all, and
6.77 ms/step with it on. That is the same thing `t_route`'s header comment says
about itself: it is the first collective after the object loop, so it absorbs the
spread between ranks, and inserting a reduction there makes *it* the imbalance
sink rather than adding one. The authoritative reading is the un-instrumented
one above — 26.72 against 26.70 ms/step — where the vote is present in both arms
and the totals are indistinguishable.

#### The second pass is nearly free, and that is a statement about the halo

Splitting the object loop in two — classify everything, vote, then pick up the
bodies another rank saw change — was expected to cost an ownership rescan on the
bodies this rank got wrong. Measured at np = 4: **43 bodies per step out of 4000
visits, and all 43 of them are settling rebuilds** rather than reclassifications.
The "clean here, dirty elsewhere" case essentially never occurs, and the reason is
the vertex-complete halo: an entity that changes owner is almost always already in
the pool of every rank that holds the body, so every rank sees the change on its
own. What the second pass actually pays for is the settling rule (below), at
about 1.4 % of the bodies that would otherwise be skipped.

#### One thing that had to be found by measurement: the rebuild is not idempotent

The first working version compared **identical** on the set digest and
**different** on the slot-order digest, from the first timestep, on any body
large enough to span three ranks. `rebuild_object` lays the pool out as
`[kept owned | new owned | kept ghost | new ghost]`, and "kept" means "a corner of
one of MY triangles" — so an owned vertex whose incident triangles all belong to
other ranks is *not* kept, it is dropped, re-received, and lands at the end of the
owned run rather than where it was. Running the rebuild twice on an unchanged
body therefore moves entities the first run left in place: the second run is the
one that reaches a fixed point, and the initial scatter's output is not one
either.

A skip taken before that fixed point is reached is still a correct partition, and
the set digest says so. It is a different *storage order*, which is exactly the
class of difference the order digest exists to catch and the reason 7c reports
both. The fix is one bit per body — a body must be seen clean **twice** before it
is skipped, so it gets one settling rebuild per dirty-to-clean transition. Cost:
the 43 bodies per step above. Benefit: the optimisation is invisible rather than
merely harmless, and `--digest` stays a complete acceptance test instead of a
partial one.

### 7f. The last four items of the review: 1.02–1.04× on the phase, and the top-ranked one was worth a third of its estimate

`docs/redistribution_optimizations_review.md` ranked eight ideas. Optimization 1
(skip unchanged bodies) is 7e; Optimization 3 (the frames rebuild) was resolved by
deleting frames outright. This is the remaining four, implemented together:

| item | change | where |
|---|---|---|
| **4a** | emit `tris.verts`/`tris.neighbors` inside the relink, the arrival loop and `repair_twins()` instead of a closing `MFTLObjectBuildDerived()` | `rebuild_object` |
| **2a** | `build_kept_lists`: four record scans → one pass with two cursors, reading the dense `vert_owner[]`/`tri_owner[]` instead of the AoS records | `build_kept_lists` |
| **4b** | the `v_final_of_old` clear feeds only an assert, so it is `#ifndef NDEBUG` now; the TRIANGLE clear is load-bearing and stays | `rebuild_object` |
| **2b** | `collect_destinations`: a branch-free XOR-OR reduction answers "does the whole stencil sit on one rank", and `cand[7]` is built only when it does not | `collect_destinations` |

**The review's shares no longer rank them, and that is the first thing to say.**
Every percentage in its verdict table was measured with the skip OFF and the
frames still present. 4a was ranked "best effort/reward in the list" at 5 % of the
phase; against the current profile it is 1.7 % when the skip fires. What these
four are actually for is the case the skip *cannot* help — one domain-spanning
body, a remeshing step where every body is `partition_dirty`, the settling
rebuilds — because every bucket they touch is one that already shrinks 3.6–5.2×
under a skip.

**End-to-end, un-instrumented, np = 4, 200 steps, 6 reps:**

| arm | 1000 bodies, `--no-skip` | 8 spanning bodies |
|---|---:|---:|
| 4a alone | 1.017× (4/6 reps) | 1.008× (4/6) |
| 2a + 2b + 4b | 1.028× (**6/6**) | 1.039× (**6/6**) |
| **all four** | **1.038× (6/6)** | **1.040× (4/6)** |
| *control: `triangle geometry`* | *0.979–1.002×* | *0.983–1.002×* |

With the skip on and 1000 small bodies it is 1.02× and inside the noise, exactly
as the re-ranking predicts.

#### Where it came from, and the one number that contradicts the estimate

`-DMFTL_COMM_PROFILE=ON`, same case, 5 reps, deltas taken pairwise per rep. Only
the buckets that moved are listed; the largest move among the buckets these
changes cannot touch was +0.067 ms/step, which is the noise floor here.

| bucket, ms/step | skip off | | | skip on | | |
|---|---:|---:|---:|---:|---:|---:|
| | before | after | sign | before | after | sign |
| destination analysis | 1.899 | 1.514 | **5/5** | 0.625 | 0.498 | **5/5** |
| derived: topology | 0.595 | 0.009 | **5/5** | 0.140 | 0.003 | **5/5** |
| kept: relink half-edges | 1.240 | 1.668 | 0/5 | 0.273 | 0.352 | 0/5 |
| **TOTAL** | **9.603** | **9.177** | **5/5** | **4.843** | **4.733** | 4/5 |

**2a + 2b take 20 % off the destination analysis**, which is the second-largest
bucket of the phase, and they do it in both regimes. That is a straight reduction
in touched bytes: `build_kept_lists` went from 34 B/vertex and 26 B/triangle over
four scans to 5 B/entity over one, and `cand[7]` stopped being written to the
stack — confirmed in the `-O3` arm64 disassembly, where the array lived at
`[x29, #-124]` with four to seven stores per owned triangle and three reloads for
the all-equal test, and where the frame-slot memory operations of the whole
function fell from 30 to 17.

**4a IS a win, and it is worth a third of what the traffic arithmetic predicts.**
Isolated in its own A/B: the closing pass costs 0.588 ms/step and **0.369 of it
comes straight back into `t_relink`**, for a net 0.22 ms/step — 2.3 % of the
phase, not the 5 % (or the 6 % the 0.588 alone would suggest). The review's model
was "trade one full pass for 24 B/tri of stores into lines you are already
sequential over", and what it leaves out is *which loop*: `build_topology_arrays`
is a pure stream over three contiguous arrays that the prefetcher handles
perfectly, while the relink is a dependency-bound loop already doing two random
gathers per half-edge. Moving stores from the first kind of loop into the second
is not free even when the byte count falls. The bucket had to be retired for the
same reason it was worth doing: what remains of it is one assignment, so it now
reads 0.003–0.009 ms/step and the sum `t_relink + t_arrivals` is the honest
place to look.

Both are still worth having: 4a removes a whole pass over the pool (which is a
footprint statement as much as a time one, and §7c is clear that footprint is
what has historically mattered on a cluster), and it costs one duplicated
convention that `inv_derived_matches_topology()` already pins.

#### All four are behaviour-neutral, verified as such

Partition digests — set form **and** slot-order form — are identical to the
un-patched tree at np = 1, 2, 4 and 8, on 1000 small bodies, on 8 domain-spanning
bodies and on 500 bodies at np = 8, each with the skip on and off: **16
configurations, and each of the four items was checked separately as it landed.**
`tests/test_derived_arrays_match_a_full_rebuild` re-derives both arrays with
`MFTLObjectBuildDerived` after a redistribution and compares them element for
element; rotating the emitted `tris.verts` by one corner makes it fail at np = 2
and np = 4, which is how it was confirmed to be able to fail at all.

#### A third measurement trap, on top of 7d's two

7d says compare interleaved, within one collection. That is not sufficient here.
**Inside a single collection this machine drifts upward monotonically** — the
un-patched arm went from 9.08 to 10.54 ms/step over six reps, 16 %, while the
effect being looked for is 4 % — so a minimum taken across arms compares
different points on the drift curve and can return either sign. Every ratio above
is formed **inside one rep** and then averaged, and the count of reps in which
the sign agrees is reported next to it, because that count is what distinguishes
a 4 % effect from a 16 % ramp.

The same mistake was made once more while producing the bucket table: the first
version ran all of A and then all of B. Every bucket then moved in the same
direction, including `owner classification`, which none of these changes touches,
and the table was unreadable. Interleaving the two arms fixed it. A bucket the
change cannot reach is the control, and it must be printed.

## 8. I/O: 23–32× from dropping ASCII — and MPI-IO is not the reason

Source: a standalone harness that generates the mesh and times `MFTLWriteVtu`
alone. 256 spheres at refinement 3 in a periodic box, 327 680 owned triangles.
Best of five, barriers around each write.

| ranks | ascii, N files | **binary, N files** | binary, 1 file (MPI-IO) |
|---|---|---|---|
| 1 | 125.6 ms | **3.96 ms** | 4.05 ms |
| 2 | 69.7 ms | **2.68 ms** | 3.27 ms |
| 4 | 41.6 ms | **1.84 ms** | 4.11 ms |

Piece file on rank 0, 4 ranks: 5.15 MB (ascii) → 3.37 MB (binary), and 3.30 MB
with the owned-only default.

### The win is the encoding, and almost nothing else

Three effects are separable in that table, and they are wildly different sizes.

1. **ASCII → raw binary: 23–32×.** This is the whole result. `"%.10e"` costs on
   the order of a hundred nanoseconds and 17 bytes for a value that occupies 8;
   the binary assemble phase is a `memcpy`. The writer was never disk-bound —
   it was bound by `printf`.
2. **Implementation, at the same encoding: 1.20–1.27×.** Measured against a
   writer emitting the identical ASCII format one `fprintf` per line: this is
   what block assembly plus buffered `fwrite` buys, together with computing each
   triangle's periodic-wrap decision once instead of four times. Modest, because
   both are still dominated by decimal formatting. It is the honest ceiling on
   "optimise the writer without changing the format".
3. **MPI-IO shared file: 1.0× at one rank, 0.82× at two, 0.45× at four.** It is
   *slower*, and it gets worse with rank count.

### Why the shared file loses here, and why that is still not an argument against it

The per-rank layout scales — 3.96 ms → 1.84 ms from 1 to 4 ranks, 2.15× — and
the shared file does not move at all (4.05 → 4.11 ms). That is exactly what a
single local SSD predicts: `MPI_File_write_at_all` earns its keep by letting the
implementation aggregate many small non-contiguous slices into a few large
stripe-aligned writes, and on one device with no stripes there is nothing to
aggregate, only a shared file to contend for.

Checked before writing this down: it is not `MPI_File_set_size`. Removing the
truncate moved the 4-rank figure from 4.01 ms to 4.06 ms, i.e. not at all. The
cost is in the collective writes themselves.

So the answer to "is MPI-IO much more efficient than the .vtu/.pvtu writer" is
**no on this machine, and the question is aimed at the wrong thing** — the 30×
was sitting in the number formatting. What the shared file actually buys is
unmeasured here and is a different property: **P+1 files per dump is what stops
working first at scale.** At 4096 ranks a per-dump file-per-rank pattern is a
metadata storm on Lustre/GPFS long before bandwidth is the issue, and one file
also means one thing to move, checksum and open. `MFTL_IO_SINGLE_FILE` exists for
that, is correct, and **must be re-measured on a parallel filesystem before
anyone chooses between the two.** It should not be chosen on the strength of the
table above, which says only that a laptop is not a cluster.

### What the layout bought the writer

The analysis pass is a consequence of §2 rather than of anything I/O-specific.
`tris.anchor` and `tris.edge` already hold corner 0 and the two seam-consistent
edge vectors, so the unwrapped corners of a triangle straddling a periodic face
are `p0`, `p0+e1`, `p0+e2` — already computed. Recovering them from scratch takes
two `MFTLGetMinimumImageVector` calls inside a helper the writer needs four times
per triangle per file. Here the whole pass streams `tris.verts`, `tris.anchor`
and `tris.edge` — 84 contiguous bytes per triangle, no gather, no periodic
arithmetic.

## 9. The fused update against the naive loop — 1.6× flat, 2.3× periodic

Source: `examples/ensemble_update_benchmark.c`, single rank, best of 50
interleaved repetitions, 327 680 triangles. Four variants; A and D produce
geometry *and* the nine quality statistics, B and C produce geometry only.

| ns per triangle | A naive walk | B the walk, geometry only | C streaming frames | D `MFTLEnsembleUpdate` |
|---|---|---|---|---|
| non-periodic | 5.07 | 2.87 | 1.64 | **3.11** |
| periodic | 7.88 | 4.04 | 1.90 | **3.45** |

- **D vs A — the headline, same work: 1.61× non-periodic, 2.27× periodic.**
- **D vs B: 0.91× non-periodic, 1.19× periodic.** D does strictly more than B.

Stable to ±5 % across three consecutive runs; A and D agree to 0.0 absolute on
every area, normal and centroid and to 3e-16 relative on the nine statistics.

### Two results here contradict the obvious expectation

**The statistics are not nearly free.** The estimate that follows from "they ride
on vectors already in registers" is a few per cent. They cost **+90 % on the
fused path** (C → D: 1.64 → 3.11 ns) and +77 % the naive way (B → A). What makes
the estimate wrong is the *baseline*, not the statistics: C is a pure streaming
loop at 1.64 ns/triangle, and against something that fast, three subtractions,
three dot products, a second divide and nine min/max reductions are simply a lot
of arithmetic. A geometry baseline slow enough to be gather-bound hides all of
it. §6's finding that this kernel sits on the divide/sqrt unit is the same
observation from the other side — the quality ratio `q = 4√3·A/(a²+b²+c²)` adds a
*second* divide to a loop already limited by one.

Fusing is still the right call, and that is unchanged: doing the same work in two
passes (A − B = 2.20 ns) costs more than doing it fused (D − C = 1.47 ns), which
is the 25 % figure `CLAUDE.md` already records.

**`MFTLEnsembleUpdate` is slower than the geometry-only path on a non-periodic
domain.** 3.11 against 2.87 ns/triangle — it computes strictly more, so this is
not a defect, but it does mean *the fused pass is not a free upgrade*. Call it
when the statistics are wanted. On a periodic domain, which is what a real run
has, it wins even against geometry-only (3.45 against 4.04), because the naive
path pays a minimum-image correction per edge per pass and the frames paid it
once at build time.

### The opportunity, taken — and what it was actually worth

An earlier revision of this section recorded variant C as an unexploited
opportunity: `MFTLComputeTriangleGeometry` walked the half-edges even when the
frames were current, and the streaming kernel was measurably faster.

It is now a DISPATCH inside that function (`src/geometry/triangle_geometry.c`):
one test on `obj->derived`, outside the loop, streaming when the frames are
current and walking when they are not. No call site changed, and no caller has to
know. The rejected alternative was a second public entry point, which would have
made every call site restate a fact the object already knows — and a call site
that got it wrong would read stale frames.

The two kernels are **bit-identical**, which is what makes a silent dispatch
legitimate rather than a hazard: `build_frames` computes the same minimum-image
differences from the same positions, so the streaming path is the same arithmetic
with the gather hoisted out. `test_triangle_geometry_paths_are_bit_identical`
asserts it with `memcmp`.

Where the fast path is taken, with no call site changing:

| after | frames | path |
|---|---|---|
| `MFTL_FOREACH_OWNED_VERTEX` | cleared by the iterator | walk |
| `MFTLUpdateGhostPositions` | cleared by the exchange | walk |
| `MFTLRedistribute` | rebuilt on the way out | **stream** |
| `MFTLGenerateSphere` | built by the generator | **stream** |

**And now the honest part.** The kernel is 1.25–1.57× (non-periodic) and
1.94–2.00× (periodic) faster in isolation, and a controlled A/B on the ensemble
the advection example actually builds — 256 bodies, 4 ranks, flipping only the
frame bit — measures 1.43–1.44×. But in the example's step 5 the phase moved from
0.535 to 0.529 ms per step: **about one percent.**

The reason is that step 5 runs on pools the redistribution has just rebuilt, so
it is paying first-touch on freshly written memory rather than executing a
kernel. §7 already recorded that phase at 6.4 ns/triangle against 2.9 ns for the
same function on a resident mesh; the dispatch improves the 2.9 and leaves the
gap alone. A faster kernel cannot help a pass that is waiting for DRAM.

So the change is kept — it is free, it is bit-identical, and it is worth 1.4–2.0×
wherever the data is resident — but **it does not speed up the one production
loop in this repository**, and anyone quoting the 2× should say against what.

## 10. What was not measured

- Scaling beyond 4 ranks on real hardware. The numbers above are 4 MPI ranks on
  an Apple M4, which has 4 performance cores and 6 efficiency cores — at 8 ranks
  the ratios still hold but the absolute per-triangle costs roughly triple,
  because half the ranks land on efficiency cores. **The scaling curve has to be
  re-measured on a cluster node**; nothing here says anything about how the
  exchange behaves at P in the hundreds.
- The split between the graph-communicator path and the global path.
  `MFTLExchangeStart` supports both and the redistribution picks between them
  with one `MPI_Allreduce`, but the benchmark only ever hits the global path on
  the initial scatter.
- The 4× traffic claim for `tris.verts` / `tris.neighbors` in
  `src/object/derived.c`. That is an arithmetic statement about bytes touched
  (12 B versus 48 B per triangle), not a timing. §5 is the first consumer that
  isolates the arrays, and it did not separate them from the alternative.
- **I/O on a parallel filesystem.** §8 is one laptop and one SSD, which is the
  configuration least able to distinguish the two file layouts. The number that
  decides between them — how a shared collective write behaves against P per-rank
  files at P in the thousands on Lustre or GPFS — is not in this document, and
  the table in §8 must not be used as if it were.
- **MPI-IO hints.** The shared-file backend passes `MPI_INFO_NULL`. Collective
  buffering parameters (`cb_nodes`, `cb_buffer_size`, striping) are where the
  remaining tuning lives on a real filesystem and none of it was touched.
- **Reading meshes back.** There is still no reader in this library, so nothing
  here says what an I/O round trip costs.

## 11. Measurement traps hit while producing this document

Recorded because each produced a *plausible* wrong answer first.

1. **Probing Morton locality with a store-bound loop** (§3): the store stream
   hides the gather it was meant to expose.
2. **Accumulating into a scalar** (§3): the FP-adder dependency hides nine
   gathered loads, so the kernel measures the adder.
3. **Comparing two nearly-identical vectors with `acos(a·b)`** (`tests/`): the
   arc cosine has infinite derivative at 1, so a dot product one ulp below 1
   reports 1.5e-8 rad. Two normal fields agreeing to 1.5 ulp measured 1.5e-6
   degrees apart and looked like a bug. Use the chord `|a − b|`, whose derivative
   is 1, and keep `acos` for angles that are genuinely large.
4. **Benchmarking an unrolled variant through a struct** (§6): `acc_t[WAY]` is an
   address-taken array, and the 15 % it costs was very nearly read as the gain
   from unrolling. U5 exists solely so that U6/U7 have the right baseline, and
   U8 exists because even that was not conclusive enough.
5. **Quoting a cold first run of the BASELINE** (§7): the old implementation
   allocates and frees tens of megabytes per timestep, so its first run pays
   first-touch faults that its later runs do not. The improvement read 9.7× once
   and 5.1× on every repeat. A speed-up that shrinks by half under repetition was
   measuring the baseline's page faults, not the new algorithm.
6. **Timing I/O variants in blocks rather than interleaved** (§8): writing all
   the ascii dumps, then all the binary ones, then all the shared ones handed
   each successive mode a differently warm page cache. The owned-only default
   measured *slower* than the same writer including ghosts — i.e. less data
   taking more time — which is what exposed it. Interleaving the modes removed
   the inversion. Same trap as §6's interleaving rule, in a place where the
   operating system rather than the cache was doing the remembering.
7. **Blaming a slow result on my own code before checking** (§8): the shared-file
   backend calls `MPI_File_set_size` to truncate, and "the truncate is the
   problem" is the obvious reading of a 2.2× loss. Measured: removing it changed
   4.01 ms to 4.06 ms. Had it not been measured, the write-up would have carried
   a plausible and wrong explanation for a real effect.
8. **Trusting one harness because its answer was explainable** (§2): the
   walk-vs-frame comparison read 1.03–1.24× for as long as only one benchmark
   measured it, and that number came with a satisfying story — "the baseline got
   fast, so the margin shrank". The harness was timing the two variants in blocks
   and averaging within each. Interleaving and taking minima gave 1.40–1.57× on
   the same mesh. Nothing was suspicious about the wrong number; it was caught
   only because a second and then a third harness disagreed with it.
9. **A cache-flush experiment that measured the flush** (§9): to find out whether
   the streaming kernel still wins on cold data, each variant was preceded by a
   walk over a 64 MB buffer. At four ranks that is 256 MB of eviction traffic
   into the same DRAM, and the result swung between 0.73× and 1.26× across
   repeats — a 27 % "regression" that was pure contention. The same experiment at
   one rank is stable and says 1.39–1.72×. A microbenchmark whose *setup* is
   heavier than its measurement is measuring the setup.

Timings throughout are the **minimum of nine runs**, and the first refinement
level of each table is warm-up-affected and should be ignored.
