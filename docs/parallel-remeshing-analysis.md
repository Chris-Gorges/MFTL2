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

## 7. What this does and does not settle

**Settled.** Determinism, to the bit, at every rank count tested. Correctness against the serial answer,
both as integral quantities and as an edge-length distribution — and at an adequate resolution the
distributed answer is *better* than the serial one on every quality measure. The phase count saturates
rather than growing with P at a fixed budget (3.83 → 3.96 from np = 2 to np = 8). Overhead growth on a fixed
problem is +33 % for 4× the ranks. Ghost synchronisation, advection and the pre-migration geometry pass
together are under 3 % of a timestep.

**Not settled.** Everything above np = 8 and everything on real hardware — the development machine has four
performance cores, so np = 8 is oversubscribed and the absolute times above np = 4 are lower bounds. The
strong-scaling question (does the migration cost stay proportional to the band as P reaches the hundreds?)
needs a cluster and a mesh large enough that a rank holds tens of thousands of triangles, not hundreds.

**Deliberately not attempted.** The work-list architecture of the plan's §7 shortens each sweep. Nothing
measured here says the sweeps are the problem: the remesh is 96–99 % of the timestep, but what varies with
the budget and the rank count is the *phase count*, i.e. the migrations. A profile that separates sweep
time from migration time inside `MFTLRemeshEnsembleParallel` is the prerequisite, and it does not exist yet.
