# Performance analysis {#mftl_page_perf}

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
