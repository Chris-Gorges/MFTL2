`src/ensemble/` — The library handle {#mftl_mod_ensemble}
====================================

**Public header:** `include/mftl_ensemble.h` · **Layer:** 3 ·
**Module group:** @ref mftl_ensemble

| File | Job |
|---|---|
| `ensemble.c` | lifecycle: build the MPI topology, own the object list, tear it all down |
| `ensemble_update.c` | the fused per-timestep pass: all triangle geometry **and** the quality statistics, in one streaming loop plus three reductions |

---

## Ensemble → Domain → Objects {#ens_model}

- **#MFTL_Ensemble** — one #MFTL_Domain plus a list of `MFTL_Object*`. It **owns**
  the objects; `MFTLEnsembleDestroy` cascades.
- **#MFTL_Domain** — the MPI and physical world shared by every object: the
  Cartesian grid, the rank map, the 26-neighbour graph, the global bounds and the
  periodicity. Held **by value**, because there is exactly one.
- **#MFTL_Object** — one mesh. It holds a **borrowed** domain pointer, set by
  `MFTLEnsembleAddObject`, and no MPI state of its own.

Decoupling the container from the contents is what lets ten thousand droplets
share one process grid without ten thousand copies of it.

---

## The MPI topology is built once {#ens_topology}

`domain_init_comm` sets up three things at creation so that **nothing on a hot
path has to call an MPI topology function again**:

| Field | What it is | Why it is materialised |
|---|---|---|
| `comm_cart` | the Cartesian communicator | it *defines* the mapping from grid coordinates to ranks |
| `rank_map` | that mapping as a flat `[nproc]` array | `MPI_Cart_rank` is a function call with internal bookkeeping, and owner lookup happens **once per triangle per redistribution** |
| `comm_graph` | a distributed graph over the (up to) 26 surrounding ranks | it makes *neighbourhood collectives* available — the message count per rank is bounded by the neighbourhood degree rather than by $P$, so communication cost does not grow when the job does |

Also precomputed on the domain, for the same reason:

$$
  \texttt{grid\_factors}_k = \frac{\texttt{cuts}_k}{L_k}, \qquad
  \texttt{inv\_domain\_size}_k = \frac{1}{L_k}
$$

so owner lookup is a multiply and a table read, and the minimum-image correction
is a multiply and a round. Neither ever divides.

Cartesian **reordering is disabled**, so ranks keep the same order as the input
communicator — a caller that has already placed its ranks deliberately is not
overruled.

### De-duplicating the neighbour stencil

The 26 offsets $\{-1,0,1\}^3 \setminus \{0\}$ do not always land on 26 distinct
ranks. On a thin grid — $2\times1\times1$, or a periodic axis with `cuts = 2` —
wrap-around maps several offsets onto the same rank, and one onto **this** rank.
`domain_init_comm` collapses those, so the graph degree is the number of *distinct*
neighbours.

That is exactly why `test_meshgen_parallel` is registered at $P = 4$ on a
$4\times1\times1$ grid, and why `test_bvh_ghosting` runs at $P = 27$: a
$3\times3\times3$ grid is the smallest one with an **interior** rank whose 26
neighbours are 26 *distinct* ranks. Below it, periodic aliasing collapses several
offsets and the de-duplication is doing half the work by accident.

---

## The fused update pass {#ens_update}

`MFTLEnsembleUpdate` runs one streaming loop per object that produces `tris.area`,
`tris.normal`, `tris.center` **and** nine quality accumulators from the same
loads, then reduces them across ranks.

Four properties make the loop cheap, and only the last is a micro-optimisation:

1. **One pass, not two.** Geometry and the edge-length statistics read the same
   edges. The three edges of a triangle are $\mathbf{e}_1$, $\mathbf{e}_2$ and
   $\mathbf{e}_2 - \mathbf{e}_1$, so the statistics cost three subtractions and
   three dot products on vectors already in registers — against a second pass
   that re-gathers eighteen doubles per triangle.
2. **No minimum-image work in the loop.** `tris.edge` carries the seam correction
   already applied, so it is paid once per frame rebuild instead of nine
   comparisons and eighteen fused operations per triangle per timestep.
3. **No gather at all.** Two contiguous arrays in, three contiguous arrays out.
   Nothing in the loop dereferences anything.
4. **The owned prefix, not an ownership test.** Owned triangles are an array
   prefix, so the loop bound *is* the ownership filter.

### The nine accumulators

$$
  \begin{array}{lll}
    \min A, & \max A, & \textstyle\sum A \\
    \min q, & \max q, & \textstyle\sum q \\
    \min \ell^2, & \max \ell^2, & \textstyle\sum \ell^2
  \end{array}
  \qquad
  q = \frac{4\sqrt{3}\,A}{a^2 + b^2 + c^2}
$$

$q$ is the mean-ratio quality: 1 for an equilateral triangle, tending to 0 as one
degenerates, and **bounded above by 1** for every non-degenerate triangle — which
is what makes an average above 1 a usable alarm.

Lengths are kept **squared** throughout, so the reduction never takes a square
root.

### fmin/fmax, not a comparison

The accumulators use `fmin`/`fmax` rather than `if (a < lo) lo = a;`, and that is
a measured choice. Checked with `-Rpass-analysis=loop-vectorize`: with the
if-form (and with a ternary) clang reports *"value that could not be identified as
reduction"*, i.e. the loop is disqualified before anything else is considered.
With `fmin`/`fmax` that objection disappears.

@note On AArch64 `fmin`/`fmax` lower to single `fminnm`/`fmaxnm` instructions; on
      x86-64 they need extra NaN fixup around `minsd`/`maxsd`, so the 4–5 %
      measured is **not portable** and is worth re-measuring on the cluster.

### Three collectives, not $3B$

The nine slots per object are laid out contiguously — 3 MIN, 3 MAX, 5 SUM (the
two extra SUM slots are the triangle and edge counts) — so the whole ensemble's
statistics are **three `MPI_Allreduce` calls, one per operator**, regardless of
how many bodies there are. At $B$ bodies that is 3 collectives instead of $3B$,
and the message is $11B$ doubles either way.

The averages are then taken against the **global** entity counts, never as an
average of per-rank averages: a rank holding a tenth of the mesh must not
contribute a tenth of the weight to a mean it computed over its own share.

### Why the kernel is a file-static here rather than in `src/geometry/`

This is the library's one deliberate exception to the "arithmetic belongs to
`src/geometry/`" rule, and it is **performance over tidiness**.

The kernel carries nine live reduction accumulators. Moving it across a
translation-unit boundary would force them into an address-taken struct, which
`docs/performance-analysis.md` §6 measures at **15 %**. The kernel stays next to
the reduction it feeds.

### The statistics are not nearly free

They cost **+90 %** on top of the geometry (1.64 → 3.11 ns/triangle), which is an
order of magnitude more than the "a few extra flops on data already in registers"
estimate the fusion argument suggests.

The reason is that the streaming geometry baseline is *fast*, and $q$ adds a
**second divide** to a loop already limited by one. Fusing is still
right (two passes cost 2.20 ns against 1.47 fused), but:

@warning `MFTLEnsembleUpdate` is **slower than `MFTLComputeTriangleGeometry` on a
         non-periodic domain** while computing strictly more. Call it for the
         statistics, not as a free upgrade. On a periodic domain — what a real run
         has — it wins at **1.19×**.

---

## Vertex normals need an EXCHANGE, not a reduction {#ens_normals}

`MFTLEnsembleUpdateVertexNormals` is the per-vertex counterpart, and the
difference in **kind** is the interesting part:

| | Collective | Why |
|---|---|---|
| `MFTLEnsembleUpdate` | **reduction** | a minimum over one rank's partition is not the mesh's minimum |
| `MFTLEnsembleUpdateVertexNormals` | **exchange** | owned normals are already exact; the ghosts are the half that is missing |

The pass is `MFTLComputeVertexNormals` (local, exact on owned vertices because
the halo is vertex-complete) followed by `MFTLUpdateGhostVertexNormals` (one
request/reply round). Nothing is summed across ranks, because nothing needs to be.

---

## What is deliberately local {#ens_local}

`MFTLDomainGetHMin` returns $\min_k L_k/\mathrm{cuts}_k$ over the axes that are
actually cut — the narrowest rank box, and the length scale every parallel scheme
with a boundary band is measured against.

Axes with fewer than two cuts are **skipped** rather than counted as zero: an
effectively 2D decomposition has no interior plane on the flat axis, so that axis
constrains nothing. It returns 0 only when no axis is cut, i.e. $P = 1$.

It is local, and every rank computes the same value from the same replicated
fields — which is what lets a driver branch on it without splitting the ranks.
