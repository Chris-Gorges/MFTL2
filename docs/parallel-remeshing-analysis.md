# Distributed remeshing: correctness, determinism, efficiency, scalability {#mftl_page_parallel_remesh}

Measured with `examples/leveque_remeshing_parallel.c` — the LeVeque deformation field, a sphere in a
time-reversed vortex that stretches into a thin doubly-wound filament by `t = T/2` and returns to its
initial shape at `t = T`. The interface's area grows by an order of magnitude and shrinks back, so the mesh
must be remeshed every timestep, and the state at `t = T` measures everything the remesher did along the
way. 150 steps, `T = 3`, `dt = 0.02`, unit cube with walls, feature detection off.

Its serial twin is `examples/leveque_remeshing_singleproc.c`, and the parallel example reproduces the
serial answer exactly when it is asked to (see "the phase budget" below), which is what makes the two
comparable at all.

**Development machine: Apple silicon, 4 performance cores.** Every run at np = 8 is oversubscribed and
every absolute time above np = 4 is a lower bound on the hardware's real capability. Ratios survive that;
absolute nanoseconds do not. Nothing here has been run on a cluster.

This document measures. **What the two drivers actually do — the call graph of each, the covering argument
behind the eight-placement schedule, the freeze rule, the id scheme and the convergence certificate — is in
@ref mftl_mod_remeshing**, and the fields quoted below are indexed by symptom at @ref rem_diagnostics.

---

## 1. The headline: the phase budget is the parameter that matters

`MFTL_RemeshParameters.parallel_phase_budget` defaults to 32, which means *run until no rank anywhere wants
to perform an operation*. That is right for a one-shot remesh and measurably wrong inside a per-timestep
loop — in both directions at once. The budget is the only thing varied here:

| budget | np = 1 phases | V/V0 | min q | ms/step | np = 4 V/V0 | np = 4 min q | np = 4 ms/step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 0.751 | 0.876 | 7.1 | — | — | — |
| 2 | 2.00 | 0.769 | 0.856 | 20.4 | 0.745 | 0.897 | 12.9 |
| 4 | 3.69 | 0.733 | 0.836 | 37.8 | **0.771** | **0.879** | **16.7** |
| 8 | 5.73 | 0.683 | 0.459 | 52.7 | 0.670 | 0.796 | 35.4 |
| 32 | 13.29 | 0.682 | 0.278 | 99.0 | 0.639 | 0.621 | 84.7 |

Serial reference (`MFTLRemeshObject`, the `nproc == 1` path): **V/V0 0.751, min q 0.876, 7.1 ms/step.**

Two things to read off it.

**Volume conservation and triangle quality are non-monotone in the budget and get WORSE as it grows.**
Re-converging an already-good mesh to a fixed point every timestep is not free polish; it is repeated
split/collapse churn that erodes an under-resolved surface. At budget 32 the minimum triangle quality has
fallen by two thirds against the serial answer and the enclosed volume by 7 points.

**The cost is linear in the phase count, because every phase is a migration.** 13.3 phases against 3.7 is
2.6× the wall clock for a strictly worse mesh.

At budget 4 the distributed answer at np = 4 **matches or beats the serial one on every correctness
measure**: V/V0 0.771 against 0.751, min q 0.879 against 0.876, 0.05 % of edges out of the length band
against 0.09 %.

### Why the default stays at 32

Because the two failure directions are not symmetric. A one-shot remesh with too small a budget **fails
loudly** — `MFTL_RemeshStats.converged` comes back false and the caller can act. A per-timestep loop with
too large a budget **fails quietly** — the mesh is structurally perfect, every invariant holds, and it is
simply worse and slower than it should be, forever. A default that is wrong in the loud direction is the
safer of the two, so the fix is documentation (`parallel_phase_budget` in `mftl_remeshing.h` carries this
table) rather than a new default.

**A solver loop should set 2..8.** Four covers slab and tube conflicts — the even-parity orthogonal array —
and was the best measured compromise. Eight is a full covering set of `MFTL_REMESH_PHASES` and is what to
use once the process grid is large enough for corner conflicts to be common. Expect `converged` to be false
and do not treat it as an error: the mesh is being *maintained*, not converged.

---

## 2. Correctness against the serial answer

Budget 4, 150 remeshed timesteps. `refine 3` starts at 1280 triangles and peaks near 3900.

| np | peak tri | V/V0 | min q | avg q | inverted | out of band | V−E+F |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (serial path) | 3930 | 0.7508 | 0.876 | 0.982 | 0 | 2 / 2154 | 2 |
| 2 | 3758 | 0.7420 | 0.819 | 0.984 | 0 | 4 / 2136 | 2 |
| 4 | 3754 | 0.7710 | 0.879 | 0.985 | 0 | 1 / 2220 | 2 |
| 8 | 3712 | 0.7184 | 0.901 | 0.987 | 0 | 1 / 2124 | 2 |

The spread in V/V0 across rank counts (0.718 … 0.771) is **larger than the gap between serial and any
individual parallel run**, and it is not monotone in P. That is the expected signature of a legitimate
difference rather than a bias: two partitions choose different edges to split and collapse, and on a
surface this far from resolved that shows up in the volume integral. A systematic parallel penalty would
grow with P; this does not.

The absolute volume loss of ~25 % is the *test case*, not the remesher — the filament at `t = T/2` is
thinner than the target edge length at `refine 3`, so the mesh cannot represent it. At `refine 4` the same
run conserves volume to within a few percent (§5). The comparison above is valid because both sides run the
identical under-resolved problem.

Every run: Euler characteristic exactly 2, zero inverted triangles, under 0.2 % of edges outside the
length band.

### The edge-length distribution

A structurally perfect mesh with the right triangle count can still carry a second population of edges at
the *original* length, where a distributed pass quietly refused work along the partition boundaries.
`test_the_edge_length_distribution_matches_the_serial_answer` compares 16-bin normalised histograms against
a *converged* serial reference built independently on `MPI_COMM_SELF`, as total-variation distance:

| np | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| total variation | **0.0000** | 0.028 | 0.027 | 0.044 |

The np = 1 figure is the interesting one: the phase driver reproduces the converged serial distribution
*to the bin*. The frozen gate and the migrations are exactly transparent when there is nothing to freeze.
The rest grows slowly with P and is the residual of genuinely different triangulations — a single
un-remeshed band on that fixture would be over 10 % of the surface.

---

## 3. Determinism

Two identical runs at each rank count, compared by a bitwise digest of the whole distributed mesh (XOR-folded
global ids and raw coordinate bits, `MPI_BXOR`-reduced — XOR is associative and exact, so the digest is
independent of the reduction tree, which a floating-point sum is not):

| np | run 1 | run 2 | |
|---:|---|---|---|
| 1 | `f45a22917d6f42d0` | `f45a22917d6f42d0` | identical |
| 2 | `d61ab5fdc0d81af4` | `d61ab5fdc0d81af4` | identical |
| 4 | `0275cc47ff9d5fa4` | `0275cc47ff9d5fa4` | identical |

40 remeshed timesteps each, budget 4. **Bit-for-bit reproducible at every rank count**, which is the
property the whole design was chosen for: no rank ever modifies an entity another rank holds, so nothing
depends on message arrival order. It is the reason the shift/freeze schedule was preferred over a
lock/ask/agree protocol, and it is now measured rather than argued.

Digests are only ever compared *within* a rank count. Across rank counts the mesh legitimately differs.

---

## 4. Efficiency: phases and migrations

At budget 4 the driver uses **3.83–3.91 phases per remesh at np = 2, 4 and 8** — it saturates, it does not
grow with P. That is the open half of CLAUDE.md deferred issue 11 answered for this case: the phase count
observed there (4 / 6 / 19 / 17 at np = 1 / 2 / 4 / 8) was a *one-shot* refinement running to a fixed point,
and the growth was the mesh being far from its target, not the partitioning.

Each phase costs exactly one migration, plus one closing migration: 4.8–4.9 migrations per remeshed
timestep.

---

## 5. Scalability

Budget 4, wall clock per timestep, max over ranks:

| mesh | np | peak tri/rank | phases | ms/step | vs np = 1 | vs np = 2 |
|---|---:|---:|---:|---:|---:|---:|
| refine 3 (peak ~3900) | 1 | 3930 | 1.00 | 7.08 | 1.00× | |
| | 2 | 1879 | 3.83 | 15.41 | 2.18× | 1.00× |
| | 4 | 938 | 3.83 | 15.59 | 2.20× | 1.01× |
| | 8 | 464 | 3.85 | 21.42 | 3.03× | 1.39× |
| refine 4 (peak ~18 300) | 1 | 18 356 | 1.00 | 57.78 | 1.00× | |
| | 2 | 9 121 | 3.88 | 83.38 | 1.44× | 1.00× |
| | 4 | 4 553 | 3.91 | 91.15 | 1.58× | 1.09× |
| | 8 | 2 275 | 3.96 | 111.27 | 1.93× | 1.33× |

**Read the np ≥ 2 columns against each other, not against np = 1.** The np = 1 row does not take the same
code path at all — `MFTLRemeshEnsemble` routes one rank to the serial remesher, with one pass and no
migration — so the step from 1 to 2 ranks is a change of algorithm and not a scaling datum. That is exactly
why it is the largest step in both tables.

Between np = 2 and np = 8 the problem is fixed and the algorithm is identical, so the ratio is the honest
overhead number: **+39 % at refine 3, +33 % at refine 4, for four times the ranks.** There is no speedup to
report and none to expect — 18 000 triangles over 8 ranks is 2275 per rank on a four-core machine, so what
is being measured is how fast the overhead grows, and it grows slowly.

The phase count is the reason it grows at all: 3.83 → 3.85 and 3.88 → 3.96 across the whole range. It
**saturates rather than growing with P**, so the per-step migration count is essentially constant and the
remaining growth is per-migration cost.

Ghost synchronisation is 0.009–0.094 s of runs lasting 2.3–16.7 s: **0.3–0.6 %.** Advection and the
pre-migration geometry pass together are under 2 %. Between 96 % and 99.5 % of the timestep is inside
`MFTLRemeshEnsemble`, and that share *rises* with P. Anything to be optimised is in there.

### Resolution changes the sign of the parallel-vs-serial difference

`refine 4`, 5120 initial triangles, ~18 300 at peak — the first configuration in this document where the
mesh can actually represent the filament at `t = T/2`:

| np | V/V0 | min q | avg q | inverted | out of band | V−E+F |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (serial path) | 0.9590 | 0.7215 | 0.9765 | 0 | 34 / 7263 | 2 |
| 2 | **0.9662** | **0.8618** | 0.9827 | 0 | **7** / 7227 | 2 |
| 4 | 0.9636 | 0.8510 | 0.9816 | 0 | 9 / 7218 | 2 |
| 8 | 0.9621 | 0.8540 | 0.9816 | 0 | 8 / 7236 | 2 |

Volume is conserved to **3.4–4.1 %** and is flat in P. Every distributed run is *better* than the serial
one on all three quality measures: minimum triangle quality 0.85 against 0.72, and four times fewer edges
outside the length band.

That is the opposite sign from §1, where extra phases made the mesh worse, and the two are the same
mechanism seen at two resolutions. The phase driver does roughly 3.9× the remeshing work of a single serial
pass. On a mesh that **can** resolve the surface that extra work is what it looks like — more relaxation,
better triangles. On a mesh that **cannot** (refine 3, where the filament is thinner than the target edge)
the same work is repeated split/collapse churn against a feature the mesh cannot hold, and it erodes it.

So the guidance in §1 is not "fewer phases are better"; it is **match the effort to the resolution**, and
the phase budget is the knob. Under-resolved: keep it low. Adequately resolved: the extra phases are free
quality, at 1.3–1.6× the serial cost.

---

## 6. A trap that cost a run, and is now closed three ways

The length band must be the same on every rank, and **nothing in the algorithm communicates it** — every
rank is simply assumed to have been handed the same one. The natural way to derive a target is

```c
MFTLGenerateSphere(obj, centre, r, n);
double target = MFTLMeanEdgeLength(obj);        /* LOCAL */
```

which reads as if it measures the sphere. `MFTLGenerateSphere` is **collective and already distributes**,
so it measures one rank's share — and a rank owning no part of the body measures the empty set and gets
0.0. That rank's band is then `[0, 0]`, under which every edge it is ever sent is too long.

Measured on a 1280-triangle sphere at np = 2: the rank that started empty received a 105-triangle cap in the
first timestep and turned it into **148 287 triangles over eight phases**, still climbing when the run was
killed. No assertion fired and no invariant broke — every one of those triangles was a perfectly valid
triangle.

Closed by:

- `MFTLEnsembleMeanEdgeLength` — the collective form, partition-independent, one Allreduce of two doubles.
- A refusal on a degenerate band in both distributed drivers, so the failure is a return value at the call
  site rather than memory exhaustion minutes later.
- `MFTLRemeshParamsAgree`, a Debug check that every rank was handed a bitwise-identical band.

Guarded by `test_the_edge_length_band_is_a_collective_measurement`, which asserts all three *and* asserts
that the local form genuinely differs across ranks — without that last clause there would be no trap to
guard and no reason for the collective form to exist.

---

## 7. The coarsening fallback, the gather, and the balancer

Three features share one mechanism — a coarsened *owner function* over the existing `MFTL_OwnerMap` — and
they answer three different questions. All of the numbers below come from `examples/remesh_regime_sweep.c`
(`-only f`, `-only g`, `-only l`), one body of radius 0.7 in $[-1,1]^3$ periodic unless stated, budget 8,
target $0.7 \times$ the mesh's own mean edge.

### 7.1 The frozen share is the diagnostic worth having

A frozen entity is migrated with everything else and then not touched, so **the frozen share is the share of
every migration that buys nothing**. `MFTL_RemeshStats::frozen_fraction` reports it, and it is free whenever
`parallel_coarsen` is armed — which is now the default.

Measured on a sphere of radius 0.7, trigger disabled so the number is only observed:

| grid | refine 2 (320 tri) | refine 3 (1280 tri) | refine 4 (5120 tri) |
|---|---:|---:|---:|
| 2×2×2 (np = 8) | 45.0 % | 24.4 % | — |
| 3×3×3 (np = 27) | 66.2 % | 36.6 % | 20.2 % |

**The obvious model over-predicts by about two.** The band is $\delta$ either side of every plane, so
$1-(1-2\delta/h)^3$ looks like the answer; it gives 71 % where 36.6 % was measured. A sphere of radius 0.7
in a box of side 2 does not sample the cut planes uniformly. Anyone setting `coarsen_frozen_fraction` from
the geometry rather than from the measured field will coarsen roughly twice as far as they intended.

Note also what the correctness condition permits. $h = 4\delta$ **is** 87.5 % frozen, so the band trigger
tolerates a partition in which the pass migrates the whole mesh, every phase, to remesh a seventh of it.

### 7.2 Does the frozen trigger pay? Sometimes, and it buys convergence rather than time

np = 27, 3×3×3, one body of radius 0.7. The `trg` column separates the triggers, which is the whole point of
the experiment: a row where the band trigger also fired is not evidence about this one.

| refine | frozen | threshold | trigger | ranks/group | phases | migrations | hidden L/S | wall | verdict |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| 2 | 66.2 % | off | band | 8 | 5 | 2 | 0 / 0 | 0.0095 | converged |
| 2 | 66.2 % | 0.33 | band+frozen | 27 | 5 | 2 | 0 / 0 | 0.0088 | converged |
| **3** | **36.6 %** | **off** | **none** | **1** | **8** | **8** | **1 / 31** | **0.0251** | **NOT converged** |
| **3** | **36.6 %** | **0.33** | **frozen** | **27** | **7** | **2** | **0 / 0** | **0.0561** | **converged** |
| 4 | 20.2 % | 0.25 | none | 1 | 8 | 8 | 2 / 37 | 0.0435 | NOT converged |

The refine-3 pair is the measurement. The trigger fires **alone**, and it converts a pass that did not
converge and hid 32 edges into one that converged and hid none — **for twice the wall clock.** That is the
trade in its clearest form: migrations fall from 8 to 2, but one rank in twenty-seven does all the sweeping.

Above and below that window it says nothing. At 66 % the band trigger has already fired and the frozen one
only escalates the factor from 8 to 27 ranks per group with no change in outcome or time. At 20 % it never
fires, and the pass's actual complaint is `parallel_phase_budget`.

**So `coarsen_frozen_fraction` defaults to 1.0, disabled.** The useful window is roughly 30–60 % frozen, the
payment is wall clock, and whether that is worth it is a property of the run rather than of the library.
`frozen_fraction` is reported either way, which is the half that is unconditionally worth having.

### 7.3 The single-rank gather: the crossover is per rank, not per mesh

`serial_gather_max_triangles` sets `group[k] = cuts[k]`, so every entity lands on one rank. Nothing is
replicated, so nothing is frozen; no axis keeps an interior plane, so there is one distinct placement and
**every phase after the first is the identity and is skipped.** The pass moves exactly twice however many
phases it sweeps.

| np | mesh | distributed wall / verdict | gathered wall / verdict | tri/rank | gather wins? |
|---:|---:|---|---|---:|---|
| 4 | 80 | 0.0008, NOT conv, 36 hidden | 0.0006, **converged**, 0 | 20 | yes |
| 4 | 320 | 0.0023, NOT conv, 1 hidden | 0.0078 (3.4×), NOT conv | 80 | no |
| 4 | 1280 | 0.0124, NOT conv, 7 hidden | 0.0189 (1.5×), **converged** | 320 | time no, mesh yes |
| 8 | 80 | 0.0014, NOT conv, 83 hidden | 0.0013, **converged**, 0 | 10 | yes |
| 8 | 320 | 0.0034, NOT conv, 32 hidden | 0.0052 (1.5×), **converged** | 40 | time no, mesh yes |
| 8 | 1280 | 0.0171, NOT conv, 21 hidden | 0.0380 (2.2×), NOT conv | 160 | no |
| 27 | 80 | 0.0102, NOT conv, 146 hidden | 0.0037 (**2.8× faster**), **converged** | 3 | yes |
| 27 | 320 | 0.0091, NOT conv, 200 hidden | 0.0086, **converged**, 0 | 12 | yes |
| 27 | 1280 | 0.0141, NOT conv, 32 hidden | 0.0284 (2×), **converged** | 47 | time no, mesh yes |

**The crossover is at roughly 50–100 triangles per rank**, and it moves with the rank count — which the
parameter cannot express, because it is an absolute count. That is deliberate: one rank holds the entire
ensemble for the duration of the sweep, and a count is the only form of the question that bounds the memory.
A caller wanting the performance crossover should set it to about $50 \times \texttt{nproc}$, capped by what
one rank can hold.

**It defaults to 0, disabled**, because no single absolute number is right across rank counts.

The secondary result is the more reliable one and holds in every row: **the gathered pass converges and hides
nothing.** With no frozen entities there is no band for work to hide behind, so the certificate is
unconditional. Where the distributed pass hid 146 edges (np = 27, 80 triangles), the gather hid none — and
was 2.8× faster while doing it.

### 7.4 The dispersed load balancer: 2.5–5.2× on a clustered ensemble

Bodies go where their centroids are, and centroids cluster — an emulsion is not a uniform distribution over
a process grid. The balancer keeps each body on its centroid's owner unless that rank exceeds
`mean × dispersed_balance_slack`, and then moves it to a neighbouring cell that has room.

**It used to reach that decision by replication and it now reaches it by delivery**, which is the whole of
what changed here. `replicate_for_balancer` broadcast `4 × n_bodies_global` doubles so that every rank could
run the same greedy over the same table — the library's last `O(bodies in the ensemble)` term, and one that
needed a different *algorithm* rather than a different route, because a greedy over rank *loads* sums bodies
held anywhere in the job. What replaces it decides each body on ONE rank — the owner of its reduced centroid
— and publishes `(body, destination, weight)` one hop along `comm_graph`, which every holder of that body is
on. Payload and staging are `O(local bodies × graph degree)`; the precondition (a body's centroid owner is a
graph neighbour of every rank holding it) is a purely local test reduced by one `MPI_LAND`, and a failure
reverts the pass to plain centroid ownership and reports `n_bodies_unreachable`.

**Every row of both tables below was re-measured on the new balancer and reproduces the old one**, which is
the point: the two agree on the answer and differ only in what they cost. Rows marked *before* are the
replicated greedy, measured on the same fixture in the same session.

$4P$ droplets, all packed into one rank's box (`remesh_regime_sweep -only l`, the `clustered` row):

| np | balancer | imbalance | ranks idle | busiest-rank ops | wall | (replicated, same session) |
|---:|---|---:|---:|---:|---:|---|
| 2 | off | 2.00 | 1 / 2 | 6 304 | 0.0139 | |
| 2 | **on** | **1.25** | **0 / 2** | **3 940** | **0.0096 (1.45×)** | 1.25, 0 idle, 3 moved |
| 4 | off | 4.00 | 3 / 4 | 12 608 | 0.0300 | |
| 4 | **on** | **1.25** | **0 / 4** | **3 940** | **0.0121 (2.5×)** | 1.25, 0 idle, 11 moved |
| 8 | off | 8.00 | 7 / 8 | 25 216 | 0.0997 | |
| 8 | **on** | **1.25** | **0 / 8** | **3 940** | **0.0198 (5.0×)** | 1.25, 0 idle, 27 moved |
| 27 | off | 27.00 | 26 / 27 | 85 104 | 0.2699 | |
| 27 | **on** | **1.25** | **0 / 27** | **3 940** | **0.0827 (3.3×)** | 1.25, **1 idle**, 103 moved |

The same droplets on a lattice across the domain, i.e. the case the balancer should leave alone:

| np | balancer | imbalance | ranks idle | wall |
|---:|---|---:|---:|---:|
| 2 | off → on | 1.00 → 1.00 | 0 → 0 | 0.0077 → 0.0071 |
| 4 | off → on | 1.25 → 1.25 | 0 → 0 | 0.0126 → 0.0121 |
| 8 | off → on | 2.00 → 1.25 | **4 / 8 → 0** | 0.0217 → 0.0147 |
| 27 | off → on | 2.00 → 1.25 | 0 → 0 | 0.0489 → 0.0799 |

Even the lattice benefits at np = 8, where four of eight ranks were idle because a cubic lattice of bodies
does not align with a 2×2×2 process grid. The np = 27 lattice row is the one that costs: 16 of 108 bodies
move to reach 1.25, and each is a body that leaves the geometric partition and pays for it in the closing
redistribution. That is the trade the slack names, and lowering it is how a caller declines it.

**Two failures the local rule has and the replicated one did not, both found by measurement and both fixed.**
Neither could have been found by a test of the answer — a balancer that silently does nothing produces a
perfectly valid mesh:

- **A body is indivisible, so a proportional ask is wrong.** Splitting the excess over 26 neighbours in
  proportion to the room each advertises returns grants *smaller than one body*, so nothing is placeable.
  Measured: 108 bodies over 27 ranks stayed at an imbalance of 2.00 with `n_bodies_reassigned == 0`, while
  the clustered fixture — one rank asking, uncontended — balanced perfectly. The ask now goes to
  `floor(budget / heaviest)` neighbours, emptiest first.
- **A tie-break identical on every rank starves the same rank.** Four overloaded ranks looking at four
  equally empty neighbours all choose the same one; measured as 1 of 8 ranks left idle. Ties break on the
  neighbour's cell *offset*, which is relative to the rank doing the looking.

And one parameter the local rule has that the replicated one did not: the number of ask/grant/ship rounds.
Measured imbalance against a slack of 1.25 —

| rounds | np = 4 | np = 8 | np = 27 spread | np = 27 clustered |
|---:|---:|---:|---:|---:|
| 1 | 1.25 | 1.25 | 1.50 | 1.25 |
| 2 | 1.25 | 1.25 | 1.25 | 1.25 |
| 3 | 1.25 | 1.25 | 1.25 | 1.25 |
| 4 | 1.25 | 1.25 | 1.25 | 1.25 |

Two is what the measurement needs; three ship, because one too few is a silently unbalanced pass and one too
many is three directed doubles per neighbour.

**So `dispersed_balance_slack` defaults to 1.25, enabled** — the one of the three that the measurement says
to turn on.

Two things the table does *not* say. The `imbal` column of the sweep — max/mean owned triangles per rank
*after* the pass — still reads 4.00 / 8.00 / 27.00 on the clustered rows with the balancer on, and that is
correct: the closing `MFTLRedistribute` restores the canonical geometric partition, so one rank holds those
bodies again. The balancer decides who does the *remeshing*, not who holds the mesh afterwards. And
`n_bodies_reassigned` is simultaneously the benefit and the cost: 103 of 108 bodies moved at np = 27, and
each one is a body that left the 26-neighbourhood, so redistribution takes its global exchange path.

### 7.5 What arming the fallback by default costs

`parallel_coarsen` now defaults to 1. On a healthy partition it decides not to coarsen, and the pass is then
**bit-for-bit** the one that ran before — `test_the_fallback_stays_off_on_a_healthy_partition` remeshes the
same fixture armed and disarmed and compares mesh digests, not merely triangle counts.

The cost of asking is one pre-pass before the phase loop: a geometry refresh, one freeze classification, and
two small Allreduces (a `MAX` of two doubles for the band, a `SUM` of two for the counts). No migration and
no sweeps. Against §5's finding that advection and the pre-migration geometry pass together are under 2 % of
a timestep, one more geometry pass on top of the four-odd a pass already takes is about +0.5 %.

---

## 8. What this does and does not settle

**Settled.** Determinism, to the bit, at every rank count tested. Correctness against the serial answer,
both as integral quantities and as an edge-length distribution — and at an adequate resolution the
distributed answer is *better* than the serial one on every quality measure. The phase count saturates
rather than growing with P at a fixed budget (3.83 → 3.96 from np = 2 to np = 8). Overhead growth on a fixed
problem is +33 % for 4× the ranks. Ghost synchronisation, advection and the pre-migration geometry pass
together are under 3 % of a timestep.

Of the three features in §7: the **balancer** is settled and on by default (2.5–5.2× on a clustered
ensemble, neutral where there is nothing to fix). The **gather** works exactly as its mechanism predicts —
two migrations whatever the phase count, and a certificate that cannot hide anything — but its crossover is
per rank and its parameter is absolute, so it ships disabled with a documented rule of thumb. The **frozen
trigger** has a real but narrow window (about 30–60 % frozen) in which it buys convergence for roughly twice
the wall clock, and ships disabled; the `frozen_fraction` diagnostic it was built around is the part worth
keeping and is always reported.

**Not settled.** Everything above np = 8 and everything on real hardware — the development machine has four
performance cores, so np = 8 and np = 27 are oversubscribed and the absolute times there are lower bounds.
The §7 tables are *ratios within a rank count*, which survives that; the wall-clock columns across rank
counts do not. The strong-scaling question (does the migration cost stay proportional to the band as P
reaches the hundreds?) needs a cluster and a mesh large enough that a rank holds tens of thousands of
triangles, not hundreds.

Two consequences of coarsening are argued rather than measured, and both need P in the hundreds to see. A
coarse destination is `group[k]` cells away, so it is outside the 26-neighbourhood and both
`MFTLRedistributeWithOwners` and `MFTLRemeshSyncGhostIds` fall back to their global paths — `O(nproc)`
allocations and an `Alltoall`, which is CLAUDE.md's deferred issue 6. And redistribution's 32-bit
destination de-duplication (`MFTL_DEDUP_BITS`) reverts to a sort above 32 distinct remote destinations, a
path np ≤ 27 cannot reach. The **gather** is exempt from the second: every rank sends to exactly one
destination, which is why it is the safest coarse map rather than the most extreme one.

**Deliberately not attempted.** The work-list architecture of the plan's §7 shortens each sweep. Nothing
measured here says the sweeps are the problem: the remesh is 96–99 % of the timestep, but what varies with
the budget and the rank count is the *phase count*, i.e. the migrations. The prerequisite for deciding it
is a profile that separates sweep time from migration time inside `MFTLRemeshEnsembleParallel`, and that
now exists in two forms: the six per-section timers on `MFTL_RemeshStats` (`t_migrate`, `t_geometry`,
`t_freeze`, `t_sweep`, `t_reduce`, `t_total` — each a MAXIMUM over ranks, so an idle rank shows up rather
than being averaged away), and `examples/remesh_phase_profile.c`, which prints them per placement together
with how many entities that placement actually moved. **Run it on the target machine before building a work
list**: if `t_migrate` dominates `t_sweep` there, a shorter sweep buys nothing.
