# 06 — Coupler improvement plan

Status: PLANNED, 2026-10-11. Based on inspection of the current implementation.
No tests or benchmarks were run for this assessment; benefits below are expected,
not measured. Implement priorities 1–3 first, then priority 4.

## Existing scope and guarantees

The essential coupler for front tracking and IBM is present: velocity
interpolation, exact-transpose spreading of integrated surface-tension forces,
indicator-gradient sources and IBM forces, independent fluid/surface rank
decompositions, checked coverage, and reusable search/communication windows.
See the [module documentation](../src/coupling/README.md) for the current API,
numerical contracts and limitations.

The remaining work is mainly efficiency and robustness. Keep the numerical scope
and solver-independent sample interface. The omitted divergence-free Cartesian
scheme and solver-specific interfaces remain intentional exclusions.

## First priorities

### 1. Use known-peer exchanges for directory replies

`lookup_directory` currently performs another NBX discovery phase after receiving
requests, although it already knows whom to answer and where replies must come
from. Preserve those peer lists and use the existing `MFTLCplExchangeKnown` helper
for replies, including empty replies.

This removes one discovery barrier per lookup without changing the certificates
returned or the numerical operator. It is a small internal change.

Relevant code: [directory lookup](../src/coupling/coupler.c),
[communication helpers](../src/coupling/coupling_comm.c).

### 2. Search all cached fetch buffers for a compatible layout

`MFTLCouplerFetchBegin` checks only the first released buffer in the window's pool.
If it is incompatible, it is discarded even if another cached buffer would fit.
Search the bounded pool for a matching field layout before allocating.

This avoids allocations when alternating field batches. It is a very small
change. Live snapshots must remain immutable, and the pool must remain bounded.

Relevant code: [fetch-buffer reuse](../src/coupling/coupler.c).

### 3. Replace repeated linear searches through peer lists

Directory and request construction repeatedly scan the destinations already
collected. Use a small sparse lookup table from peer rank to buffer/list index.
Allocate in proportion to actual peers, not the total MPI process count.

This helps irregular partitions with many communication partners. It is a small
internal change; preserve deterministic packing and accumulation order.

Relevant code: [directory and request construction](../src/coupling/coupler.c).

### 4. Reuse released binding storage

Candidate lists are cached, but each RK stage still allocates fresh binding and
row arrays. Retain compatible released storage and reuse its capacity for later
bindings, following the lifetime discipline already used for fetch snapshots.

This reduces allocation overhead. It is a moderate change: reuse only released
storage, preserve every live binding, and recompute all position-dependent weights.
Retained storage must remain bounded and included in memory accounting.

Relevant code: `new_binding`, stage binding and row assembly in
[coupler.c](../src/coupling/coupler.c).

## Main remaining scalability concern: synchronization

Field traffic is sparse, but prepared fetches still perform two global validation
reductions, and discovery uses global barriers. A nonblocking barrier still
requires every participating rank to enter.
[MPI barrier documentation](https://docs.open-mpi.org/en/main/man-openmpi/man3/MPI_Ibarrier.3.html)

Combine agreements where correctness permits and overlap communication with useful
work. Begin/End transfers and local stage binding already provide some of the
necessary infrastructure. Do not simply delete checks: stale handles, incompatible
calls and bad loads must still fail consistently across ranks before unsafe
communication or output writes.

Persistent MPI requests are a later option. They can reduce repeated setup
overhead, but do not reduce transferred data or eliminate global agreements.
Evaluate them after the simpler changes and profiling, rather than adding another
transport backend now.
[MPI persistent communication](https://www.mpi-forum.org/docs/mpi-4.1/mpi41-report/node83.htm)

## Other remaining limits and later work

- **Highly graded, partially exported meshes.** Selected point providers declare
  one resolution bound per rank/location. Optional bounds per region would reduce
  unnecessary sample imports while still certifying unseen samples.
- **Search-bin granularity uses the global minimum spacing.** Numerical support
  is local, but a distant fine region can cause excessive small bins and duplicated
  candidate lists elsewhere. Resolution-dependent bin levels would address this;
  it is a larger change, and must preserve complete coverage and deterministic rows.
- **Grid changes require coarse updates.** Point sets need re-registration, and
  ownership changes invalidate windows. Incremental updates would help frequent
  AMR or repartitioning.
- **The memory limit aborts on exhaustion.** Optional caches could instead be
  skipped or reclaimed when memory is tight, preserving the same numerical result.
  This does not imply that mandatory storage can always fit the budget.
- **Large-machine scaling is unmeasured.** Correctness through 27 ranks does not
  establish efficient scaling to thousands. Measure setup, binding, transfers,
  synchronization and memory separately before making scaling or speedup claims.

## Validation for implementation

Use the existing independent row/spread references and protocol tests for affected
paths. Check empty peers, irregular and reordered partitions, periodic boundaries,
stale handles and immutable live objects. Cache and routing changes must preserve
coverage, interpolation weights, the exact transpose and established reproducibility
guarantees. Add focused regressions for alternating cache layouts and released
binding reuse where existing tests do not exercise the new behavior.

Performance measurements are a separate step. Keep expected benefits distinct from
measured results, and use profiling to choose between synchronization, local work,
data volume and memory improvements after priorities 1–4.
