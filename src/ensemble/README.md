`src/ensemble/` — The library handle and the per-body collectives {#mftl_mod_ensemble}
=================================================================

**Public header:** `include/mftl_ensemble.h` · **Layer:** 3 ·
**Module group:** @ref mftl_ensemble

| File | Job |
|---|---|
| `ensemble.c` | lifecycle: MPI topology, the sorted local object directory, global counts |
| `ensemble_update.c` | the fused geometry + quality pass, and the collective vertex-normal pass |
| `ensemble_reduce.c` | per-body reductions and exclusive scans over each body's holders |
| `ensemble_extents.c` | the periodic extent rule, checked over every body |

---

## Ensemble → Domain → Objects {#ens_model}

- **#MFTL_Ensemble** — one #MFTL_Domain (by value) plus the bodies this rank
  holds. It owns them; `MFTLEnsembleDestroy` cascades.
- **#MFTL_Domain** — the MPI and physical world shared by all bodies: private
  duplicate of the user's communicator, Cartesian grid, rank map, 26-neighbour
  graph, bounds, periodicity. Read-only after creation.
- **#MFTL_Object** — one mesh, holding a borrowed domain pointer and no MPI
  state.

**The local object list is sparse.** `local_objects` holds exactly the bodies
this rank has entities of, sorted by ascending `global_id`; a redistribution
destroys bodies that left and creates bodies that arrived. Body ids are handed
out densely by `MFTLEnsembleAddObject` (every rank must add the same bodies in
the same order) and are the only handle that survives a redistribution —
`MFTLEnsembleFindLocalObject(ens, gid)` maps back, by binary search.
`n_bodies_global` is the size of the id space (topology changes retire ids, so
it can exceed the live body count) and is the count every collective guard must
use: an early return on the local count deadlocks the other ranks.

@warning An `MFTL_Object*`, a list slot or `n_local_objects` held across a
         redistribution is invalid. At one rank the stale pointer happens to
         work, which is why `test_leveque_parallel_conserves` checks a physical
         quantity (enclosed volume) at $P = 2$.

---

## The MPI topology is built once {#ens_topology}

| Field | What it is | Why it is materialised |
|---|---|---|
| `comm_cart` | Cartesian communicator, no rank reordering | defines grid coordinates → ranks |
| `rank_map` | that mapping as an `[nproc]` array | owner lookup is per entity |
| `comm_graph` | symmetric distributed graph over the distinct ranks of the 26 surrounding cells | neighbourhood collectives: message count bounded by the degree, not by $P$ |
| `neighbor_offset` | each neighbour's cell offset in $\{-1,0,1\}^3$ | "are two cells neighbours" becomes a componentwise test |
| `grid_factors`, `inv_domain_size` | $\mathrm{cuts}_k/L_k$, $1/L_k$ | owner lookup and minimum image without division |

On thin periodic grids ($2\times1\times1$, any periodic axis with 1 or 2 cuts)
several stencil offsets alias the same rank; the graph lists it once, with its
first offset. With a single cut on a periodic axis the stencil wraps onto the
rank itself, so `comm_graph` contains a **self-loop** — harmless for an
exchange, but a reduction must not fold that block twice. $3\times3\times3$ is
the smallest grid with an interior rank whose 26 neighbours are distinct, which
is why the distributed tests run at $P = 27$.

---

## The fused update pass {#ens_update}

`MFTLEnsembleUpdate` runs one streaming loop over the owned prefix of each
object: it writes area, unit normal and centroid and accumulates, from the same
registers,

$$
  \min/\max/\textstyle\sum \text{ of } A,\qquad
  \min/\max/\textstyle\sum \text{ of } q = \frac{4\sqrt3\,A}{a^2+b^2+c^2},\qquad
  \min/\max/\textstyle\sum \text{ of } \ell^2 \in \{a^2, b^2, c^2\},
$$

using that the three edges are $\mathbf{e}_1$, $\mathbf{e}_2$ and
$\mathbf{e}_2-\mathbf{e}_1$. $q$ is the mean-ratio quality: 1 for an equilateral
triangle, $\to 0$ as it degenerates, never above 1. Lengths stay squared. Eleven
values per body — 3 MIN, 3 MAX, 5 SUM including the triangle and half-edge
counts — go through **one** `MFTLEnsembleReduceBodies` call; averages divide by
global counts. The statistics are **per body**, because that is what decides
whether one particular droplet needs remeshing.

Measured points worth knowing (docs/analysis.md §1.6):

- The kernel stays file-static next to its reduction: moving it into
  `src/geometry/` would put the nine accumulators into an address-taken struct,
  15 % slower.
- The statistics cost **+90 %** over geometry alone (1.64 → 3.11 ns/triangle): $q$
  adds a second divide to a loop bound by one. Fusing is still right (two passes:
  2.20 vs 1.47 ns), but on a non-periodic domain `MFTLEnsembleUpdate` is slower
  than `MFTLComputeTriangleGeometry` — call it for the statistics.
- The loop does not vectorise, and not because of the reduction: one `sqrt` and
  one divide per triangle are a quarter of the pass and irreducible.
- `fmin`/`fmax` accumulators let clang recognise the reductions; they are single
  instructions on AArch64 but need NaN fix-up on x86-64.

`MFTLEnsembleUpdateVertexNormals` is the per-vertex counterpart and needs an
**exchange**, not a reduction: owned normals are already exact (vertex-complete
halo); one ghost exchange (`MFTLUpdateGhostVertexNormals`, three neighbourhood
collectives) supplies the ghost half.

---

## Per-body reductions over the holders {#ens_reduce}

`MFTLEnsembleReduceBodies` is the one place the library chooses the communicator
for a per-body collective. Its users: quality statistics, global counts, the
redistribution skip vote, the spatial halo's body radii, and the distributed
remesher's body measurement and id counters.

**Indexing by local slot.** `MPI_Allreduce` aligns element $k$ of every rank's
buffer, so a per-body payload for it must be indexed by body id — at $10^6$
bodies every rank would stage $10^6$ rows to learn about the few it holds. The
caller's buffer is `[n_local_objects][n_per_body]`, reduced in place.

**Two routes.** The *holder route* publishes one `(body id, row)` record per held
body over `comm_graph` (one count exchange and one allgatherv) and folds locally:
$O(\text{local bodies} \times \text{degree})$, independent of $P$ and of the
body count, and exact iff every two holders of every body are graph neighbours.
The *global route* is an `MPI_Allreduce` per distinct operator over a
body-id-indexed buffer: always exact, $O(\text{bodies})$ per rank.

**Certificate.** For body $X$ with holder set $H(X)$, every holder $R$ checks

1. $R$ owns at least one entity of $X$;
2. every rank owning a ghost of $X$ in $R$'s pool is a graph neighbour of $R$;
3. after the exchange, $R$ and every neighbour that published a record for $X$
   span at most one cell per axis.

*1 and 2 make $H(X)$ connected on the graph*: redistribution sends each triangle
to the owners of its corners and edge neighbours, so the owners of two incident
entities hold each other's entity as a ghost and are, by 2, neighbours; the
surface is connected and, by 1, every holder is an owner. *3 makes a connected
$H(X)$ a clique*: holders $a, c$ at graph distance 2 would share a holder
neighbour $m$ whose record set contains both, so by 3 their offsets from $m$
differ by at most one per axis and $c$ lies in $a$'s stencil — a contradiction.
The three verdicts are agreed with one `MPI_LAND`; on refusal the records are
discarded and the global route runs.

Checking 3 on the halo-visible holders alone, before the exchange, is **not**
sufficient: a body bent around a grid corner, cells
$(0,0)\to(1,0)\to(1,1)\to(2,1)$, passes on every rank while its ends, two cells
apart, never hear from each other
(`test_a_bent_body_is_certified_after_the_exchange`, $P = 27$). An empty holder
fails statement 1 and vetoes — the state of a generated ensemble before its first
redistribution. The certificate is re-derived on every call: any holder change
anywhere invalidates it.

<!-- figure: holder-certificate
The holder certificate on a 3×3 slice of the process grid. (a) When a body's
holders span at most one cell per axis they are pairwise graph neighbours:
every holder hears every other, and the reduction folds locally. (b) A body
bent around a grid corner. Each rank sees only neighbouring holders, so a check
before the exchange passes, yet $a$ and $c$, two cells apart, never hear from
each other. After the exchange $m$ holds records spanning three cells and
vetoes; the global route runs.
-->

@warning The argument assumes each body is **one connected surface**. Nothing in
         MFTL produces a disconnected body (topology changes give every
         component its own id); a caller assembling one must pass
         `MFTL_BODY_REDUCE_SCOPE_GLOBAL`.

Redistribution rewrites `owner_proc` with destination ranks while classifying,
so it takes statements 1–2 *before* that and passes the verdict to
`MFTLEnsembleReduceBodiesPrescanned`; statement 3 reads only the records.

**Bitwise identity.** Contributions are folded in ascending **source rank**
order with this rank spliced in at its own position (and a self-loop block
dropped), so every holder obtains bit-identical values — a solver that branches
on them never sends two ranks into different collectives.

**Bytes.** The holder route sends every held body to every neighbour, so it moves
more bytes than the global route below roughly 45 ranks and far fewer above
(measured 1.97× more at $P = 2$, 1.70× at $P = 27$; modelled 22× less at $P = 1000$; docs/analysis.md §6.5). The
decisive gain is that nothing scales with the body count.
`MFTLEnsembleGetBodyReduceStats` counts which route ran: both give identical
answers, so a call pinned to the global route is visible only there
(`test_a_droplet_timestep_takes_no_global_reduction`).

---

## Per-body exclusive scans {#ens_exscan}

Handing out ids is a **scan**, not a reduction: two holders that each created
vertices of one body need disjoint id ranges, so each needs the count created by
the holders before it. `MFTLEnsembleExscanBodies` reuses the reduction's records,
order, certificate and fallback (`MPI_Exscan` + `MPI_Allreduce`); "before" means
ascending rank, so the holders' ranges tile $[0, \text{total})$ exactly
(`test_the_scan_tiles_every_body_exactly_once`). Its consumer is the distributed
remesher (@ref mftl_mod_remeshing).

**Counting a per-body verdict once.** Every holder receives the same reduced
value, so summing a per-body verdict over ranks counts a shared body once per
holder. A MIN slot of the contributing rank yields the body's lowest-numbered
holder, and `lowest_holder == rank` is the licence to count it
(`MFTLRemeshMeasureBodies` carries this slot).

---

## The periodic extent rule {#ens_periodic_extent}

`MFTLEnsembleCheckPeriodicExtents` checks, for every body, that on every periodic
axis it either wraps the axis or is narrower than half the period
(@ref geom_periodic_extent). One `MFTLEnsembleReduceBodies` call carries the
four-frame extent (24 MIN slots) and the lowest holder (one more), each body's
verdict is counted once by its lowest holder, and one `MPI_Allreduce` of two
integers makes the report global; two more single-double reductions name the
first offender only when there is one. No periodic axis: no MPI call at all.
It is a CHECK the application calls -- at output steps, after a topology change --
rather than a refusal inside a module, because no module is wrong for a body in
the band $[L/2, 3L/4)$; the rule is the supported envelope, and this is its
safety check.

---

## Decomposition metric {#ens_local}

`MFTLDomainGetHMin` $= \min_{k:\,\mathrm{cuts}_k\ge2} L_k/\mathrm{cuts}_k$, the
narrowest rank box over the cut axes (uncut axes have no interior plane), is the
length the distributed remesher's band condition and strategy choice are
measured against. It is 0 for a single process and identical on every rank.
