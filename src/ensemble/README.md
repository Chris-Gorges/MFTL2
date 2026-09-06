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

@warning **The de-duplication removes duplicate offsets, not this rank.** The
         collapse above can put this rank's OWN id in `domain->neighbors`, so
         `comm_graph` carries a self-loop on any grid with a single cut on a
         periodic axis. An exchange does not care — the block arrives and the
         data is right. A **reduction** does: fold the self block in a second
         time and MIN and MAX hide it while every SUM doubles.
         `MFTLEnsembleReduceBodies` drops it explicitly, and shapes
         $\{2,1,1\}$ and $\{2,2,1\}$ are where a test notices.

Each neighbour's **cell offset** is kept beside its rank in
`MFTL_Domain::neighbor_offset`. Relative rather than absolute: an offset needs no
`MPI_Cart_coords` call and no seam correction, and "are these two cells
26-neighbours" becomes "is every component of the difference within 1", which is
the whole of the holder test below.

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
2. **Periodicity is unswitched.** Non-periodic objects pay only position gathers
   and subtractions; periodic objects use branchless minimum-image arithmetic.
3. **No retained position-derived state.** Compact `tris.verts` is streamed and
   current positions are gathered directly, so no rebuild pass precedes this one.
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

### One reduction, over each body's HOLDERS

The eleven slots per object are laid out contiguously — 3 MIN, 3 MAX, 5 SUM (the
two extra SUM slots are the triangle and edge counts) — as **one record per body
this rank holds**, and the whole ensemble's statistics are a single call to
`MFTLEnsembleReduceBodies`.

The three operators in one record cost nothing there, and that is the point of
the shape: the neighbourhood route is an *exchange* and combines in local memory,
so mixing min, max and sum is one message rather than three. Only the
`comm_world` fallback still issues one `MPI_Allreduce` per operator, which is
what this used to be unconditionally.

The averages are then taken against the **global** entity counts, never as an
average of per-rank averages: a rank holding a tenth of the mesh must not
contribute a tenth of the weight to a mean it computed over its own share.

## Per-body reductions {#ens_reduce}

`MFTLEnsembleReduceBodies` (`ensemble_reduce.c`) is the one place the library
decides which communicator a per-body reduction uses. Its users are the quality
statistics above, `MFTLEnsembleUpdateGlobalCounts`, redistribution's skip vote,
the spatial halo's body radii, the distributed remesher's body measurement and its
id counters.

### The caller's buffer is indexed by LOCAL SLOT

An `MPI_Allreduce` aligns element $k$ of every rank's buffer, so a per-body
payload staged for one has to be indexed by body **id** — the only index two
ranks holding different bodies agree about. That is the obvious shape, and the
one that stops scaling: at $10^6$ bodies every rank allocates, fills with
identities and reduces $10^6$ slots to learn about the hundred it holds.

So `values` is `[n_local_objects][n_per_body]`, reduced in place. Moving the id
indexing inside the function is most of the saving — otherwise the collective
shrinks and the per-rank memory does not.

### When the neighbourhood is enough

A rank holds an entity of a body only if it **owns** one: redistribution sends a
triangle to the owner of each of its three corners and each of its three edge
neighbours, so a rank holding nothing but ghosts cannot exist. The holders of a
body are therefore the cells its entities occupy, and the question is whether
that set fits inside one 26-neighbourhood.

Three local statements decide it, with no geometry and no body-global
measurement. Let $V_R(X)$ be the holders rank $R$ can see for body $X$ — itself,
plus the `owner_proc` of every ghost of $X$ in its pool:

1. $R$ owns at least one entity of $X$;
2. every member of $V_R(X)$ is a graph neighbour of $R$;
3. their cell offsets span at most one cell per axis.

`MPI_LAND` of that is exactly "the neighbourhood route is exact", and it reads
only the **ghost suffix** — a body held whole costs nothing at all.

Statement 1 is what makes $V$ complete. A rank holding an **empty** object owns
nothing, appears in no other holder's halo, and can therefore neither be
discovered nor served; it cannot be seen from outside, so it vetoes. That is not
a corner case — it is the state a generated ensemble is in before its first
redistribution, and it is what `test_ensemble_update_tolerates_empty_ranks`
caught at $P = 4$.

Given it, the converse holds. Take holders $c, c'$ two cells or more apart; the
surface is connected, so there is a chain of holders from $c$ to $c'$ whose
consecutive members share an entity and therefore see each other. If any step of
that chain is two cells or more, statement 2 fails at its start. If every step is
one cell, the chain covers at least two cells in unit steps, so some member has
chain neighbours on **opposite sides** of it along one axis — both in its own $V$
— and statement 3 fails there.

@warning The converse rests on one assumption: a body is a **connected** surface.
         Two components on ranks that neither see each other nor have anything
         between them would pass every local test. Nothing in MFTL produces such
         a body, but a caller that assembles one from an index list owes
         `MFTL_BODY_REDUCE_SCOPE_GLOBAL`.

The verdict is re-derived on every call and never cached: it is invalidated by
any holder-set change anywhere in the job, and a stale one is a silently wrong
reduction rather than a crash.

### Redistribution takes the test itself, and has to

`MFTLEnsembleBodyHoldersFitTheGraph` describes the pools **as they are**.
`compute_owners` overwrites every entity's `owner_proc` with its *destination*
rank while classifying, so after the object loop the pools describe the partition
that is about to exist while the skip vote has to reach the ranks holding the
body now. Redistribution therefore takes the answer before classifying, agrees
it, and passes `MFTL_BODY_REDUCE_SCOPE_HOLDERS` or `..._GLOBAL` explicitly.

### The answer is bitwise identical on every holder

Contributions are folded in **ascending source-rank order**, with this rank
spliced in at its own position. Floating-point addition is not associative and
`domain->neighbors` is in an MPI-chosen order that differs from rank to rank, so
receive order would leave two holders of one body with `quality` values differing
in the last ulp — and a solver branching on such a value to decide whether to
remesh would then have two ranks entering different collectives. `MPI_Allreduce`
gives that guarantee for free; here it has to be arranged.

## Per-body prefix sums {#ens_exscan}

`MFTLEnsembleExscanBodies` is the reduction's sibling, and it exists because
**handing out ids is a scan and not a combination**. Two holders of one body that
both split an edge each create a vertex, and the two need *different* global ids,
so each holder needs the count of what the holders **before** it created. No
`MFTL_BodyReduceOp` produces that.

It shares everything that can be shared: the same `(body id, payload)`
neighbourhood allgatherv, the same ascending-source-rank total order — which is
what "before" means — and the same route test and global fallback (`MPI_Exscan`
plus `MPI_Allreduce`, both posted before either is waited on). The total comes out
of the same records, so asking for it costs no extra collective.

The property it has and the reduction does not is a **tiling**: over a body's
holders in ascending rank order, each range must begin exactly where the previous
one ended, and the last must close exactly on the total. That is what
`test_the_scan_tiles_every_body_exactly_once` gathers and walks per body, and it
is the statement that makes the ids provably disjoint.

Its consumer is the distributed remesher, where it replaced an `MPI_Exscan` plus
an `MPI_Allreduce` of `2 × n_bodies_global` ints **per phase** and `11 × bodies`
ints of scratch per pass. See `src/remeshing/README.md` §rem_ids.

### Counting a per-body verdict exactly once

Every holder of a body comes out of a reduction with the same value, so summing a
per-body verdict over ranks counts a shared body **once per holder**. The library
learned this from a message that read "4 of 1 bodies are wider than a rank box".

The fix costs one slot: a `MFTL_BODY_REDUCE_MIN` of the contributing rank is the
body's **lowest-numbered holder**, and `lowest_holder[o] == rank` is the licence to
count body `o`. It rides a reduction already being taken and needs nobody to know
the holder set. `MFTLRemeshMeasureBodies` carries it as its ninth slot.

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
