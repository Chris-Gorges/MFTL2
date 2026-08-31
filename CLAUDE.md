# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MFTL (MultiFlow Triangulation Library) is a HPC C11 library for distributed-memory (MPI) manipulation of closed and
open triangulated surface meshes, for front tracking and IBM CFD simulations.

Read "The one idea" below before changing anything structural — most of the file layout follows from it, and a
change that ignores it will look local and will not be.

What exists today:

- ensemble / domain (MPI topology, process grid, periodic bounds)
- object pools (the AoS+SoA container, one allocation per entity class)
- half-edge topology with **implicit indexing** (navigation, twin rebuilding, boundary marking, integrity checking)
- derived per-triangle arrays and the `MFTL_FOREACH_*` hot-loop iterators
- Morton renumbering of the pools (`MFTLObjectReorderMorton`)
- periodic-aware geometry
- sphere generation by icosahedron subdivision, over the shared `MFTLBuildIndexedMesh` builder
- the fused per-timestep geometry + quality pass (`MFTLEnsembleUpdate`)
- area-weighted vertex normals (`MFTLComputeVertexNormals`, `src/geometry/`) and their ensemble-wide form
  (`MFTLEnsembleUpdateVertexNormals`), which adds the ghost exchange that makes them exact on ghosts too
- array functions (stable radix sort for keys and key/value pairs)
- redistribution and ghost synchronisation (`src/communication/`, `MFTLRedistribute`,
  `MFTLUpdateGhostPositions`)
- VTK output (`src/io/`, `MFTLWriteVtk` serial debug dump, `MFTLWriteVtu` parallel: ascii or raw binary,
  one file per rank or one shared file via MPI-IO)
- isotropic remeshing (`src/remeshing/`, `MFTLRemeshObject`): split/collapse/flip/smooth, feature
  detection, and the three-subspace tangential smoother — serial operators, no MPI anywhere in them
- **DISTRIBUTED remeshing**, two strategies, chosen by one ratio (body radius vs rank box width):
  `src/remeshing/remesh_parallel.c` (`MFTLRemeshEnsembleParallel`) moves the CUT PLANES over the {0, h/2}³
  shift/freeze schedule, for bodies larger than a rank box; `src/remeshing/remesh_dispersed.c`
  (`MFTLRemeshEnsembleDispersed`) moves the BODIES onto the rank owning their centroid, for bodies smaller
  than one. `MFTLRemeshEnsemble` dispatches on `nproc` and then on `MFTLRemeshChooseStrategy`
- **an armed coarsening fallback for the spanning path** (`parallel_coarsen`, default on), which merges rank
  boxes through the owner map when the frozen band no longer fits — plus a single-rank gather
  (`serial_gather_max_triangles`) that is the same mechanism at `group = cuts`, and a frozen-share
  diagnostic (`MFTL_RemeshStats.frozen_fraction`). Both extra triggers ship off; see the invariants below
- **a deterministic, collective-free load balancer for the dispersed path** (`dispersed_balance_slack`,
  default on), which spreads clustered bodies over ranks from replicated data alone
- **all the remeshing module's MPI in one file** (`src/remeshing/remesh_comm.c`): migration, the id
  collectives and the ghost id exchange. Enforced by `test_remeshing_mpi_is_confined`
- LBVH and object–object contact detection (`src/bvh/`, `MFTL_GenerateLBVH` / `MFTL_RefitLBVH` /
  `MFTL_IntersectObjects`): Morton radix-split tree, explicit-stack dual traversal,
  Shewchuk-adaptive orientation and an exact Guigue–Devillers narrow phase
- the spatial halo (`src/communication/spatial_halo.c`, `MFTLExtendSpatialHalo`): the triangles of one
  body that neighbouring ranks hold near this rank's share of another, delivered as a standalone
  shadow `MFTL_Object` per body

**Absent**: open-surface generators, GTS I/O, topological ghost-layer growth, and a mesh READER of any
format. Do not add references to these from the current modules — build the module first.

## The one idea

A mesh is described by two things that want opposite memory layouts, and each is stored in the layout it wants:

- **Topology and identity → AoS.** `MFTL_Vertex` (16 B), `MFTL_Triangle` (12 B), `MFTL_HalfEdge` (16 B). A remesher
  edits one entity at a time and touches most of its record when it does.
- **Geometry → SoA.** Positions, normals, areas, centroids and curvatures. These
  are read and written by kernels that touch one component of every entity, every timestep.

**Both live in ONE allocation per entity class.** `MFTL_VertexPool` and `MFTL_TrianglePool` each hold a single
`block` cut into cache-line-aligned sub-arrays by `MFTLLayoutAdd` (`mftl_memory.h`), and each has exactly one
`capacity`. Three capacities per object (`capacity_*`, `geom_cap_*`, `flat_cap_*`) kept in step by a rule written
in a comment is the arrangement this replaces; a growth site that forgets one produces a heap overflow. Here the
AoS and every SoA array are the same `malloc`, so that class of bug is not expressible.

`src/object/object.c` is the **only** file allowed to write a pool pointer, a `capacity`, or a `block`. Everything
else changes size through `MFTLObjectReserve` / `MFTLObjectReserveAdditional` / `MFTLObjectSetCounts`, and
gives storage back through `MFTLObjectShrinkToFit` / `MFTLObjectReleasePools` — never implicitly.

### Three invariants every module may assume

1. **Half-edges are triangle-indexed.** Half-edge `3t+k` belongs to triangle `t`, and `next(3t+k) == 3t+(k+1)%3`.
   `he_idx`, `next_idx` and `triangle_idx` are therefore *not stored* — they are arithmetic (`MFTLTriFirstHe`,
   `MFTLHeTriIdx`, `MFTLHeNextIdx`, `MFTLHePrevIdx`). The half-edge array lives inside the triangle pool at 3× its
   capacity. **Anything that creates a triangle must create its three half-edges at `3t`, `3t+1`, `3t+2`.**
2. **Ownership is a prefix.** Entities `[0, n_local)` are owned by this rank; `[n_local, n)` are ghosts. There is no
   `is_ghost` field. `owner_proc` survives only to answer "whose is it" for a ghost. An index test, a flag and an
   `owner_proc == rank` comparison are three competing notions of ownership that can disagree; the prefix is the
   only one that cannot go stale, because it IS the storage order.
3. **Positions live in the SoA and only there.** `verts.pos[3*v+k]` is authoritative. There is no second copy and no
   second index space: the Morton ordering is applied to the *pool* (`MFTLObjectReorderMorton`), not stored as a
   permutation. Storing it as a permutation (`sfc_perm`/`sfc_inv`) is the trap this avoids: a second index space
   forces a sync function, a staleness flag and an "unwrapped or not" mode flag onto every consumer.

### Derived arrays and staleness

`tris.verts` and `tris.neighbors` are rebuildable caches, tracked by `obj->derived`:

| bit | arrays | invalidated by |
|---|---|---|
| `MFTL_DERIVED_TOPOLOGY` | `verts`, `neighbors` | any topology change |

`MFTLObjectBuildDerived` rebuilds both arrays. Geometric consumers gather the
authoritative positions and construct seam-corrected edges in registers, so
vertex motion invalidates no derived state.

`tris.verts` and `tris.neighbors` are *redundant* with the half-edge records (a rotation of `he.vertex_idx`, and
`he.twin_idx / 3`). They are stored for traffic — 12 B/triangle instead of 48 — not for instructions. See
`src/object/derived.c`, which also records which of the ten arrays a flattened representation grows to were
deleted, and why.

**There are no hash maps.** Global IDs are dense per entity class, and "the other side of this edge" is already an
index — that is what `khash` was doing work for. Do not reintroduce one without saying which lookup genuinely
cannot be an array.

## Build

Requires CMake ≥ 3.13, an MPI installation, and a C11 compiler.

```bash
python3 build_all.py              # Release (optimized, no sanitizers) — for performance runs
python3 build_all.py --debug      # Debug WITH ASan+UBSan — default for day-to-day dev
python3 build_all.py --valgrind   # Debug WITHOUT sanitizers — required under Valgrind
python3 build_all.py --lib-only   # skip examples/
```

`build_all.py` always deletes and recreates `build/`, so flags never mix between runs. `src/**/*.c` is globbed with
`CONFIGURE_DEPENDS`, so a new `.c` is picked up without editing `CMakeLists.txt`.

`-march=native` is **off by default**: cluster login nodes routinely differ in ISA from compute nodes, which turns
into SIGILL at run time. Use `-DMFTL_ARCH_NATIVE=ON` for a same-machine run, or `-DMFTL_ARCH_FLAGS=-march=...`.

## Tests

```bash
python3 run_local_tests.py            # rebuilds (Release) then runs the full CTest suite
ctest --output-on-failure              # from build/
ctest --output-on-failure -R object    # one module
```

Forty CTest entries, built on `tests/test_utils.h` (`RUN_TEST`, `TEST_ASSERT*`, `TEST_REPORT`). **Run the suite in
the `--debug` build too** — `tests/test_object.c` deliberately writes the last addressable element of every pool
array, so an array that was not grown with the rest is an ASan failure rather than a silent corruption.

**The two distributed remeshing strategies are each other's only cross-check.** They share no code past
`MFTLRemeshIterate` — different migrations, different id schemes, different convergence criteria — so
`test_the_two_strategies_reach_the_same_mesh` (`tests/test_remeshing_dispersed.c`) is evidence about the
ANSWER rather than about one implementation. Its tolerance is 4 % on the triangle count and 1 % on the
volume and must not be tightened to equality: split and collapse are order-dependent, so the two
legitimately choose different edges and land on different triangulations of the same surface at the same
resolution. Measured 0.4–2.5 % and 0.3 % at np = 1, 2, 4, 8.

**A distributed contact query is tested as a SET, against the same pipeline run before the mesh was partitioned.**
`tests/test_bvh_ghosting.c` is the one test that finds ghosting bugs without knowing what they are. It has four
failure modes to separate and they pull in different directions — a MISSING cross-rank contact, a DUPLICATE reported
by two ranks, a SPURIOUS one from a partition-dependent envelope, and a DIVERGENT answer at a different process
count — and multiset equality of the gathered `(gid_A, gid_B)` pairs rules out all four at once. Nothing weaker
does: counting hits misses a duplicate that cancels a miss, and "every reference pair was found" misses duplicates
entirely. The reference is itself checked per case against an exhaustive O(nA·nB) sweep that shares none of the
pipeline's machinery, so a bug in the shared code cannot hide by appearing on both sides. Thirty geometric
placements — face, edge, corner, before/on/behind the cut, periodic seam, anisotropic cells, origin at 1e6 — are
ROWS OF A TABLE resolved against the actual process grid at run time, not thirty functions. It runs at np = 1, 2, 4,
8 and **27**; 3×3×3 is the smallest grid with an interior rank whose 26 neighbours are 26 distinct ranks, and below
it periodic aliasing collapses several stencil offsets onto the same rank.

**Every row reports how many of its reference pairs are genuinely cross-rank, and the totals are asserted.** A case
whose contacts all land inside one rank tests the traversal and nothing else — the halo could be deleted and the
sets would still match. That is not hypothetical: it is what happens when the bodies shrink, the domain grows or the
grid changes shape. Any new distributed test needs an equivalent non-vacuity measure, or it will pass forever
without covering anything.

**Never compare two vectors that should be identical with `acos(a·b)`.** The arc cosine has infinite derivative at
1, so a dot product one ulp below 1 reports 1.5e-8 rad — a noise floor of ~1e-6 degrees regardless of how exact the
inputs are. Use the chord `|a − b|`. `tests/test_vertex_normals.c` records the false failure this produced.

`tests/test_mesh_invariants.h` holds the structural checks a mesh must satisfy (implicit indexing, twin symmetry,
`out_he` really outgoing, derived arrays agreeing with the records, ownership prefix, dense global IDs). Use
`inv_mesh_is_sound()` after any operation rather than re-deriving them, and `inv_partition_is_sound()` after
anything distributed — it adds the three statements that only mean something once a mesh is spread over ranks:
local global-ID uniqueness, a VERTEX-COMPLETE halo (`inv_halo_is_vertex_complete`, which is exactly what
`MFTLComputeVertexNormals` needs to be communication-free), and every owned triangle having all three edge
neighbours.

In a multi-rank test **every assertion must be made on an `MPI_Allreduce`d verdict** (see `agreed()`). `TEST_ASSERT`
returns immediately, so an assertion failing on only some ranks leaves the rest blocked in the next collective — a
hang instead of a clean failure.

## Documentation

**The documentation build is NOT in this repository.** It lives in
`MFTL_Documentation`, normally checked out beside this one, and reads this tree from wherever it is. What
stays here is only what is read in place: the Doxygen comments in `include/` and `src/**`, the ten
`src/<module>/README.md` theory chapters (they render on GitHub next to the code they explain) and the four
`docs/*-analysis.md` measurement documents (some thirty comments in `src/`, `include/`, `tests/` and
`examples/` cite them by path).

```bash
cd ../MFTL_Documentation
python3 build_mkdocs.py                 # -> build/site/index.html   (the web manual)
python3 build_mkdocs.py --serve --open  # live-reload dev server on :8000
python3 build_docs.py --strict          # the Doxygen manual; any warning is a failure
python3 build_docs.py --no-graphs       # much faster while editing prose
```

It locates this checkout through `mftl_paths.py`: `--source-dir`, `$MFTL_SOURCE_DIR`, its own
`mftl-source.txt`, or a sibling `MFTL_6`/`MFTL` directory — so side-by-side checkouts need no configuration.

**Both manuals build with ZERO warnings, and that is the acceptance criterion.** `WARN_IF_UNDOCUMENTED`,
`WARN_IF_INCOMPLETE_DOC`, `WARN_NO_PARAMDOC` and `WARN_IF_UNDOC_ENUM_VAL` are all on, so **a new function
here without `@param`/`@return` fails `build_docs.py --strict` there**. Run it after touching any header —
it is the one consequence of the split that a change in this repository can still trigger.

`CMakeLists.txt` here has no `docs` target and never calls `find_package(Doxygen)`; the CMake route to the
manual is `MFTL_Documentation`'s own `CMakeLists.txt` (`LANGUAGES NONE`, so it compiles nothing).

Everything about how the manuals are wired — the two-root Doxyfile template, the MkDocs plugin order, the
mkdoxy `src-dirs` placeholder the hooks replace, the computed member anchors, the deliberately-disabled
Markdown extensions — is documented in `MFTL_Documentation/README.md` and in its `pages/building.md`. Read
those before changing anything there; nothing in this repository depends on them.

## Architecture

### Header layering

**One module = one `src/<module>/` directory = one `include/mftl_<module>.h`.** Every `.c` includes its own header
*first*, which makes headers self-contained by construction. A header may only include headers from a strictly
lower layer:

| Layer | Headers | Contains |
|---|---|---|
| 0 | `mftl_constants.h`, `mftl_memory.h`, `mftl_vectoroperations.h`, `mftl_arrayfunctions.h` | macros and plain-C helpers, no MFTL types |
| 1 | `mftl_types.h` | all structs — the single place they are defined |
| 2 | `mftl_topology.h`, `mftl_geometry.h` | inline accessors over layer 1 |
| 3 | `mftl_object.h`, `mftl_ensemble.h`, `mftl_meshgen.h`, `mftl_communication.h`, `mftl_io.h`, `mftl_remeshing.h`, `mftl_bvh.h` | module APIs |
| 4 | `mftl_core.h` | umbrella for application code; library sources must not use it |

Two checks worth running after touching headers:

```bash
# every header must compile alone
for h in include/*.h; do printf '#include "%s"\n' "$(basename $h)" > /tmp/sc.c; \
  mpicc -std=c11 -fsyntax-only -Iinclude /tmp/sc.c || echo "NOT SELF-CONTAINED: $h"; done

# the library must export only MFTL* symbols, with no duplicates
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | grep -v _MFTL   # must be empty
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | sort | uniq -d  # must be empty
```

### Where a geometry function lives

Geometry appears in two modules and the split is by SCOPE, not by entity type. This is the rule; it was not
obvious from the file layout before, which is why it is written down:

| | one object, LOCAL, no MPI | every object, COLLECTIVE |
|---|---|---|
| triangles | `MFTLComputeTriangleGeometry` (`src/geometry/`) | `MFTLEnsembleUpdate` (`src/ensemble/`) |
| vertices | `MFTLComputeVertexNormals` (`src/geometry/`) | `MFTLEnsembleUpdateVertexNormals` (`src/ensemble/`) |

An `MFTLEnsemble*` function is a loop over the object list plus exactly the collective its result needs; the
arithmetic belongs to `src/geometry/`. The naming carries the same information, so a call site says which it is
without a lookup. Note that the two collectives are of different KINDS and that this is the interesting part:
the triangle statistics need a REDUCTION (a minimum over one rank's partition is not the mesh's minimum), while
the vertex normals need an EXCHANGE (owned normals are already exact; the ghosts are the half that is missing).

**One deliberate exception, and it is performance over tidiness.** `MFTLEnsembleUpdate`'s kernel is a file-static
function in `src/ensemble/ensemble_update.c` rather than in `src/geometry/`, because it carries nine live
reduction accumulators. Moving it across a translation-unit boundary would force them into an address-taken
struct, which the performance doc §6 measures at 15 %. The kernel stays next to the reduction it feeds.

### Object model: Ensemble → Domain → Objects → pools

- **`MFTL_Ensemble`** — one `MFTL_Domain` plus a list of `MFTL_Object*`. `MFTLEnsembleCreate()` (collective),
  `MFTLEnsembleDestroy()` cascades to every object it owns.
- **`MFTL_Domain`** — the MPI/physical world shared by all objects: Cartesian grid (`cuts[3]`), `rank_map`, the
  26-neighbour `comm_graph`, global bounds, periodicity. Objects hold a borrowed pointer set by
  `MFTLEnsembleAddObject()`.
- **`MFTL_Object`** — one mesh, i.e. two pools plus ID counters and quality statistics.

## Conventions

- Functions return `int` using `MFTL_SUCCESS`/`MFTL_ERROR`. `MFTL_ASSERT(...)` compiles to `assert()` in Debug and
  to nothing in Release — never put side effects inside it.
- `MFTL_ERROR` (-1) doubles as the "no such entity" index sentinel.
- Pool sizes change only through `mftl_object.h`. Growth is a RATCHET: `MFTLObjectReserve` never shrinks, because
  a mesh of stable size must not re-fault its pages every timestep and because `redistribute.c` pre-grows and
  relies on the next reserve early-returning. `MFTLObjectShrinkToFit(obj, slack)` is the only way down, it is
  explicit and gated, and its slack must be ≥ 4 — see `MFTL_SHRINK_SLACK_DEFAULT`.
  Plain, genuinely independent arrays use `MFTL_ENSURE_CAPACITY` (geometric) or `MFTL_RESERVE_ADDITIONAL` (exact)
  from `mftl_memory.h`. All allocation aborts on OOM by design.
- File-static helpers are `snake_case`; everything with external linkage is `MFTLPascalCase` (functions) or
  `MFTL_PascalCase` (types). Nothing but `MFTL*` may be exported.
- Formatting follows `.clang-format` (Google-based, 2-space indent, 160 columns, Allman braces).
- Collective functions (anything touching `MFTL_Ensemble`/`MFTL_Domain`, e.g. `MFTLEnsembleCreate`,
  `MFTLGenerateSphere`, `MFTLEnsembleUpdate`) must be called by every rank. `MFTLObjectReorderMorton`,
  `MFTLEnsembleUpdateVertexNormals`, `MFTLUpdateGhostVertexNormals`, `MFTLUpdateGhostGlobalIds` and
  `MFTLRemeshEnsembleParallel` are collective too.
  `MFTLObjectReorderMorton`, `MFTLObjectBuildDerived`, `MFTLComputeVertexNormals`, `MFTLRemeshMarkFrozen` and
  `MFTLDilateFrozenBand` are deliberately **local** — each rank works on its own
  partition. `test_vertex_normals_np3` exists to make an accidentally-introduced collective deadlock in CI.
  `MFTLRemeshEnsemble` DISPATCHES: `nproc == 1` takes the serial path unchanged; above that
  `MFTLRemeshChooseStrategy` (itself collective — it measures the bodies) picks the dispersed or the
  spanning driver. **Both drivers must be reached by every rank**, so the choice is made from reduced data
  and never from anything local; two ranks entering different drivers is an immediate deadlock with no
  diagnostic.
- The per-timestep order WITHOUT redistribution is: move owned vertices (`MFTL_FOREACH_OWNED_VERTEX`) →
  `MFTLEnsembleUpdate` → `MFTLComputeVertexNormals` (if per-vertex geometry is wanted).
- The per-timestep order WITH redistribution is: move owned vertices → `MFTLUpdateGhostPositions` →
  `MFTLComputeTriangleGeometry(obj, obj->tris.n)` → `MFTLRedistribute` → (optionally) `MFTLEnsembleUpdate` and
  `MFTLComputeVertexNormals`. Three things about it are load-bearing, and all three are silent when wrong:
  the ghost sync must precede the geometry pass (a triangle's three vertices can belong to three ranks); the
  geometry pass must cover the GHOSTS too, because redistribution routes them by their cached centre and
  refreshing only the owned ones makes two ranks disagree about a ghost's destination. Geometry and quality
  consumers go after redistribution when they need the new partition.
- **A BODY THAT DID NOT MOVE IS NOT TOUCHED, AND THE PRECONDITION IS NOT "DID TOPOLOGY CHANGE".** A rank's
  local state after a redistribution is a pure function of (global topology, owner map) — read the halo rule
  backwards and it defines the OUTPUT, not a procedure. Positions are not an input; they DECIDE the owner map,
  and once it is fixed, moving a vertex inside its own cell changes nothing about who holds what. So
  `MFTLRedistributeSetSkipUnchangedBodies` (default ON) leaves such a body entirely alone.

  A topology change under the freeze discipline is invisible to the halo rule — an unfrozen entity has no ghost
  corner and no ghost edge neighbour, so every triangle an operator builds out of it has all seven halo clauses
  resolving to the acting rank. What a skip genuinely breaks is **ghost triangle global ids**:
  `redensify_global_ids` renumbers owned entities and deliberately leaves ghosts alone, because redistribution
  drops and re-receives every ghost triangle and THAT is the repair. Skip it and the stale ids sit in the pool
  colliding with fresh ones — silently, on a converged body, every step. So the guard is
  `MFTLObjectMarkPartitionDirty` (`mftl_object.h`), stamped by `MFTLObjectSetCounts` (the choke point every
  entity creation and destruction already passes through), by both renumbering sites and by
  `MFTLRemeshMarkFrozen`. **Anything that edits a mesh without going through `MFTLObjectSetCounts` owes that
  call.** Derivation in `docs/redistribution_skip_correctness.md`.

  Three consequences. The vote is per body through ONE `MPI_BOR` bitset, never per body per collective — that
  splits the object loop in two, and pass 2 re-runs the idempotent `compute_owners` rather than storing
  `O(sum_verts + sum_tris)` of owner arrays. A skipped body now needs no derived
  work because motion leaves compact connectivity current. **A body must be clean
  TWICE before it is skipped**: `rebuild_object` is not idempotent (an owned vertex whose incident triangles are
  all remote is dropped and re-received, so it lands at the end of the run), so one settling rebuild per
  dirty→clean transition is what makes a skipped pool bit-identical to a rebuilt one *slot for slot* rather
  than merely equal as a set. Guarded by `test_skipping_reproduces_the_full_rebuild`, which compares BOTH
  digests — the set form passed while the order form failed from step 1, which is how the non-idempotence was
  found.

- **Redistribution invalidates all cached geometry except the vertex positions.** It returns an object with valid
  topology, valid positions and both derived bits set; `tris.area/normal/center` and
  `verts.normal/area/mean_curv/gauss_curv` hold unspecified values until something recomputes them. That is what
  buys the 3× smaller wire format — see `include/mftl_communication.h`. A solver that consumes triangle geometry
  AFTER the redistribution therefore owes one more pass — the price of not shipping the geometry on the wire. `parallel_advection_redistribute_list` runs it as step 5 and times it separately, because comparing the
  two versions' redistribute timings alone credits this one with a saving it has not finished paying for.
- Bodies are identified on the wire by their INDEX in the ensemble's object list, not by `global_id`. Every rank
  must therefore hold the same object list in the same order; a Debug-only checksum reduction inside the
  redistribution reports a violation rather than silently merging entities into the wrong body.
- **`MFTLComputeVertexNormals` needs a VERTEX-COMPLETE halo and no communication.** It accumulates over all
  triangles including ghosts, so an owned vertex gets the exact answer iff every triangle incident to it is present
  locally. That is a stronger requirement on redistribution than "one layer of edge-adjacent ghost triangles", and
  it is what buys the pass zero communication. Ghost vertices hold a PARTIAL SUM — correctly normalised and quietly
  wrong, which is the trap: it looks exactly like a real normal. `MFTLEnsembleUpdateVertexNormals` closes it with
  one ghost exchange; call that unless you are certain you read owned vertices only.

## Invariants worth preserving

- **Only OWNED entities go into a `.vtu`.** A ghost is a copy of a triangle another rank owns, so writing the whole
  pool puts every halo triangle in the merged dataset once per rank holding it. The picture
  is unchanged (the copies are coincident) and every integral the visualisation tool computes over the surface is
  wrong by the halo fraction, silently. `include_ghosts` restores the old behaviour for looking AT the halo.
  Guarded by `test_ghost_inclusion` and `test_cell_count_is_the_owned_count`.
- **The periodic-wrap test in `src/io/vtu_writer.c` has no tolerance, on purpose.** It asks whether a reconstructed
  corner `p0 + e` left the domain box. A false positive costs three extra points carrying correct coordinates and
  renders identically; a false negative draws a triangle across the whole domain. Any slack added to suppress the
  first buys the second, so there is none. `test_wrap_detection_paths_agree` pins it against the gathering
  criterion used by the half-edge connectivity fallback.
- **`MFTLWriteVtu` must never write `tris.area` or `tris.normal`.** Nothing on the object records whether they are
  current — `derived` tracks connectivity, not geometry — and a redistribution leaves them
  undefined by design, so a production file containing them would look right and be stale. `MFTLWriteVtk` writes
  them deliberately, because the raw dump's contract is to show what is actually in memory.

- **`MFTL_FLAG_FROZEN` means REPLICATED, not "near a cut plane", and the difference is what makes the
  parallel remesher need no negotiation.** An owned entity is frozen exactly when some other rank holds a
  copy of it, and `MFTLRemeshMarkFrozen` derives that from the rule redistribution SENT by: a triangle
  travels to the owner of each of its three corners and each of its three edge neighbours, so it is
  replicated iff `corner >= verts.n_local || neighbour >= tris.n_local`. Two streaming reads, six integer
  compares, no distance test and no message. Three things follow, and all three are load-bearing:

  A vertex is frozen if ANY incident triangle is, so an unfrozen vertex has every incident triangle owned
  and unreplicated — which is why gating a COLLAPSE, whose modified set is an entire fan, is two flag tests.
  Combined with the vertex-complete halo it also means every fan an approval predicate walks is complete, so
  the link condition, the wing valences and the fold-over test give the same answers they would on the whole
  mesh. **And every ghost in a rank's pool is frozen ON ITS OWNER** — it is here because this rank owns a
  corner or an edge neighbour of it, which from the owner's side IS the definition of replicated. So ghost
  copies are still exact when the sweeps finish, and a phase needs NO ghost position sync before the next
  migration. `test_every_ghost_agrees_with_its_owner` inside
  `test_a_partitioned_sphere_remeshes_on_a_periodic_domain` checks that to the bit, and
  `frozen_matches_replication` checks the equivalence itself in both directions — the "replicated but not
  frozen" direction is the fatal one, and its symptoms appear one or two migrations after the cause.

- **COARSENING IS AN ARMED FALLBACK, NOT A DEFAULT ACTION, AND THE DIFFERENCE IS CHECKED BY DIGEST.**
  `parallel_coarsen` defaults to 1, so every parallel pass runs the decision — but
  `MFTLRemeshCoarseningFactors` returns `{1,1,1}` on any axis already wide enough and
  `MFTLRemeshCoarseOwnerProc` at `{1,1,1}` is bit-for-bit `MFTLRemeshOwnerProc`, so a healthy partition
  produces the IDENTICAL mesh. `test_the_fallback_stays_off_on_a_healthy_partition` remeshes one fixture
  armed and disarmed and compares `mesh_digest`, not triangle counts — a weaker test would pass while the
  armed path perturbed the mesh, because a remesher's output is a valid mesh either way. Cost of arming: one
  pre-pass (geometry refresh, one freeze classification, two small Allreduces), about +0.5 % of a timestep.

  **Three triggers, and `coarsen_trigger` exists because they call for opposite responses.** BAND
  (`h <= 4*delta`) is not negotiable — the alternative is an un-remeshed lattice. FROZEN
  (`coarsen_frozen_fraction`) and GATHER (`serial_gather_max_triangles`) are trades with cheaper
  alternatives the library cannot take on its own. Both ship DISABLED, and both defaults are measured
  rather than assumed — see `docs/parallel-remeshing-analysis.md` §7.

- **THE FROZEN SHARE IS THE SHARE OF EVERY MIGRATION THAT BUYS NOTHING, AND `h > 4*delta` PERMITS 87.5 % OF
  IT.** A frozen entity travels with the mesh and is then not touched, so `MFTL_RemeshStats.frozen_fraction`
  is the most useful single number about a spanning pass — it is free whenever the fallback is armed, and it
  is worth logging even with its trigger off. Two traps. The obvious model `1-(1-2*delta/h)^3`
  **over-predicts by about two** (71 % predicted, 36.6 % measured at np = 27) because a sphere in a box does
  not sample the cut planes uniformly, so set `coarsen_frozen_fraction` from the MEASURED field and never
  from the geometry. And the remedy is priced in ranks: at 36.6 % frozen the trigger converts a pass that did
  not converge and hid 32 edges into one that converged and hid none, **for twice the wall clock**. Useful
  window roughly 30–60 %; above it the band trigger has already fired, below it the real complaint is
  `parallel_phase_budget`.

- **THE SINGLE-RANK GATHER IS MAXIMAL COARSENING, AND IT IS THE SAFEST COARSE MAP, NOT THE MOST EXTREME.**
  `serial_gather_max_triangles` sets `group[k] = cuts[k]`; nothing is replicated so nothing is frozen, no
  axis keeps an interior plane so `n_distinct_placements` is 1, and every later phase is an identity
  migration that is skipped — **exactly two migrations however many phases it sweeps**. No new communication
  code. Every rank sends to ONE destination, so redistribution's 32-bit `MFTL_DEDUP_BITS` is never stressed;
  intermediate coarsening is what can reach that (untested — np <= 27 cannot produce 32 distinct
  destinations). The measured crossover is **50–100 triangles per rank**, which an absolute count cannot
  express, so it defaults to 0; it is a count and not a ratio because one rank holds the whole ensemble and a
  count is the only form of the question that bounds the memory. A gathered pass converges and hides nothing
  in every row measured.

- **DISPERSED IDLENESS SPLITS IN TWO, AND ONLY ONE HALF IS FIXABLE.** With `N_bodies >= P` a pile-up is
  clustered centroids and `dispersed_balance_slack` fixes it (default 1.25, ON): measured imbalance
  4.00/8.00/27.00 -> 1.25 at np = 4/8/27, **2.5–5.2× less wall clock**, idle ranks to zero. Below
  `N_bodies == P` nothing can, because a body is the smallest unit of work this strategy has;
  `n_ranks_idle` versus `load_imbalance` is how to tell which you have. The assignment **costs no
  collective** — the per-body weights ride the reduction `MFTLRemeshMeasureBodies` already takes (its stride
  went 7 -> 8 for exactly this) — and that is a constraint, not a bonus: `MFTL_OwnerMap` is undefined if two
  ranks name different destinations, so every tie-break is total (ranks by id, bodies by index) and
  **`domain->neighbors` must never be used** to find the 26 candidates, because it is this rank's list and
  differs on every rank. The cells are re-derived from the body's own centroid. The balancer decides who
  REMESHES, not who holds the mesh afterwards — the closing `MFTLRedistribute` restores the geometric
  partition.

- **COARSENING REACHES THE DISPERSED PATH ONLY AS A YARDSTICK, AND THE RELAXATION IS BOUNDED BY BODY COUNT,
  NOT BY GEOMETRY.** `r_max > 0.4*h_min` sends a spray to the spanning driver purely because the boxes
  shrank, so with the fallback armed the test is taken against the coarsened box. The destinations do NOT
  move — coarse routing would restrict them to `P/G` ranks and undo the balancer. Unbounded, this is worse
  than useless: the factors clamp at `cuts[k]`, so on a 2x1x1 grid the coarse box is the WHOLE DOMAIN, every
  body fits, SPANNING becomes unreachable, and a jet is serialised onto one rank while reporting itself
  perfectly centred. So it requires `N_bodies >= nproc` (below it, dispersed is already leaving ranks idle
  and relaxing buys no parallelism) and refuses to merge the entire grid (that is the gather, and it belongs
  to the spanning path where a triangle count bounds the memory). Guarded by
  `test_small_bodies_at_a_fine_grid_stay_dispersed`, which asserts BOTH directions on a `{nproc,1,1}` grid —
  the suite's cubic grids have two cells per axis up to np = 8, where every merge is the whole grid and the
  refusal is correct.

- **A SHIFTED CUT PLANE NEEDS A FLOOR, NOT A TRUNCATION, AND ONLY A BODY ON THE SEAM CAN SEE IT.**
  `MFTLRemeshOwnerProc` biases by `cuts` before the `(int)` cast on a periodic axis. Without the shift the
  argument is non-negative for any wrapped position, so `(int)` IS `floor`; with a shift of up to h/2 it
  reaches −0.5, where `(int)` truncates *towards zero* and attributes the sliver
  `[global_min, global_min + shift)` to cell 0 instead of to cell `cuts−1` across the seam.

  What makes that hard to catch is that **the wrong rule is still a rule**. Every rank computes the same
  answer, the partition stays consistent, and every structural invariant still holds — cell 0 just grows to
  1.5 h and cell `cuts−1` shrinks to 0.5 h, which is exactly what a non-periodic axis does on purpose. The
  one thing that changes is that THE SEAM PLANE NEVER MOVES, so an entity on it is frozen in every phase of
  the schedule and ends up in a permanently un-remeshed band. A body away from the seam never enters the
  sliver, so both original spanning tests passed under the broken rule. Found by mutation testing, guarded
  by `test_a_sphere_on_the_periodic_seam_remeshes_everywhere`, which fails on convergence *and* reports the
  sphere as an open surface with 78 boundary half-edges.

  The same reasoning applies to `MFTLRemeshMeasureBodies`, one level up: it averages a periodic coordinate
  as an ANGLE (circular mean, `atan2` of the area-weighted mean direction) because the arithmetic mean of a
  seam-straddling body's coordinates lands in the middle of the domain and its radius comes out as half the
  box. Nothing fails — the body is simply classified SPANNING forever and takes the phase schedule it does
  not need. Guarded by `test_a_seam_straddling_droplet_is_not_measured_as_domain_sized`.

- **REMESHING REACHES THE NETWORK THROUGH ITS OWN FILE, AND THAT IS A HARD RULE.** Every MPI call the
  remesher makes is in `src/remeshing/remesh_comm.c`; the operators, the smoother, the compaction and both
  distributed drivers contain none. `test_remeshing_mpi_is_confined` (a CMake source check, not a runtime
  one) fails the build if that stops being true, because no runtime test can see it.

  The rule exists because the first version broke it in two places that were individually small, correct
  and fully tested, and collectively the wrong shape — a feature used a few times per hundred timesteps
  paid for once per timestep, in the two loops the library is most careful about:

  `MFTLGetOwnerProc` gained a per-axis cut-plane offset and a periodicity branch so the shifted
  decomposition could be spelled as a subtraction. That function IS the body of `compute_owners`, i.e. it
  runs once per vertex and once per triangle on every redistributed timestep. **Measured at 1.70× —
  1.79 → 3.05 ns/entity, 42 → 89 instructions for the same loop** (`-O3`, Apple silicon, best of seven,
  interleaved). The end-to-end redistribute timing on the development laptop varies by 2× between sessions
  and cannot resolve it; the isolated number is the honest one.

  The per-timestep ghost exchange gained a third field, "tell me my new global id", and with it two
  `field ==` tests inside the loop that applies replies to ghost vertices.

  Both are reverted. What replaces them is `MFTL_OwnerMap` (`mftl_communication.h`) — **one** seam, a NULL
  pointer on the production path and one predicted branch per body, hoisted out of both loops in
  `compute_owners`. It is also what the dispersed strategy needs, and that one is not a plane test at all,
  so the seam had to be an ownership ARRAY rather than a cleverer ownership function.

- **A rank must not judge work it only holds as a GHOST.** The first version of the parallel convergence
  certificate counted "approved, then refused by the freeze gate". It reported 3 permanently blocked
  operations on an np = 4 sphere that was in fact fully converged: a rank examines every half-edge in its
  pool, ghosts included, and around a ghost vertex its one-ring is incomplete, so it approves a collapse the
  owner correctly rejects. The gate therefore runs BEFORE the approval (it is cheaper anyway), and the
  certificate is `count_hidden_edges` — out-of-band edges with a frozen endpoint, which needs no fan walk and
  so cannot be given a wrong answer by a partial view. Convergence is `ops == 0 && hidden == 0`, with a
  full-table streak of silent placements as the fallback for the one false alarm `hidden` still has (an
  out-of-band edge no operator would fix anyway, sitting on a band).

- **A remesher's approval predicates are the whole correctness argument, and every clause is silent when
  removed.** `MFTLCollapseApproved` refuses on five grounds and none of them announce themselves: a violated
  link condition welds two sheets into a non-manifold vertex whose one-ring walk then silently visits half
  the fan; a fold-over leaves an inverted triangle; a violated stretch bound makes split and collapse
  oscillate forever. Two of the clauses are easy to omit because the obvious form of the link condition does
  not imply them — the wing-valence floor (a tetrahedron passes the neighbour count and still dissolves), and
  the **feature/boundary link condition**: two vertices on a feature curve joined by an edge NOT on that curve
  are a chord across the surface, and collapsing it pinches the curve inwards. On an open patch that serrates
  the rim; on a cube it merges two sharp edges into the middle of the face between them. Guarded by
  `test_collapse_refuses_to_dissolve_a_tetrahedron` and `test_remesh_of_an_open_patch_keeps_its_rim`.
- **The feature classification must describe the mesh being SMOOTHED, not the one the sweeps started from.**
  `MFTLRemeshObject` calls `MFTLDetectFeatures` twice per iteration for this reason. A split of a rim edge
  creates a midpoint that IS on the rim, and a vertex the smoother believes to be free gets the 2D
  tangent-plane treatment and slides straight off it — after which the surface has a serrated notch that no
  structural check can see, because the result is a perfectly valid manifold with a perfectly valid boundary.
- **Smoothing is a PROJECTION, and the feature class chooses the subspace: 2D tangent plane, 1D ridge tangent,
  0D locked.** Stated that way the three cases are one idea. Moving a ridge vertex TO the midpoint of its two
  feature neighbours — the full displacement, no projection — is the 2D shrinkage problem reintroduced on a 1D
  curve: a regular 24-gon ridge collapses to 0.84 of its radius in 20 sweeps. Guarded by
  `test_a_regular_ridge_is_a_fixed_point`, which asserts drift below 1e-12.
- **A vertex with exactly two feature edges is a ridge point only if the curve runs roughly STRAIGHT through
  it.** Counting alone makes the four corners of an open patch ridge points, free to slide along the diagonal
  between their neighbours — off the patch. `feature_corner_angle` is a separate threshold from
  `feature_angle_threshold` because it applies where that one does not: a mesh boundary is a feature curve at
  any dihedral threshold, including a disabled one.
- **A NORMAL IS ALWAYS A UNIT VECTOR, at every scale.** `MFTLGetTriangleProperties` (`mftl_geometry.h`),
  `src/geometry/triangle_geometry.c`, `src/ensemble/ensemble_update.c` and `MFTLNormalizeVector`
  (`mftl_vectoroperations.h`) all compute `mag = sqrt(|v|²)` UNCLAMPED and then
  `inv = (mag > 0.0) ? 1.0/mag : 0.0`. Two spellings are wrong and both were in the tree until 2026-08-15:
  clamping `|v|²` *before* the square root floors the magnitude at `sqrt(1e-12) = 1e-6` (and, where an area
  is derived from the same `mag`, floors that at 5e-7 — so `quality.min_triangle_area` could never report a
  degenerate triangle and `q = 4√3·A/(a²+b²+c²)` became unbounded above, an *average* q of 1.023 having been
  measured on a degenerate mesh); clamping the divisor with `MFTL_MAX(mag, tiny)` keeps the instruction
  count but only MOVES the floor — at `tiny = MFTL_VERYVERYSMALL` a triangle with 1e-8 edges still came out
  with a normal of length 0.112. A short normal is the most dangerous shape of wrong available here: it
  points the right way, so every sign test, winding check and picture still passes, while it silently
  scales surface tension and every other quantity built from it. The test against zero has no cliff and
  costs one instruction (`fcmp`+`fcsel` against `fmax`): **7.33 → 7.41 ns/tri, +1.1 %**, best of six over
  two rounds — pay it. Guarded by `test_triangle_normals_are_unit_at_every_scale` (1e-8 … 1e+8, all three
  kernels), `test_normalize_is_unit_across_magnitudes` (1e-150 … 1e+150), `test_normals_of_a_whole_mesh_are_unit`,
  `test_a_very_small_triangle_is_measured_exactly`, and `inv_triangle_normals_are_unit` /
  `inv_vertex_normals_are_unit` in `tests/test_mesh_invariants.h` — which are deliberately NOT part of
  `inv_mesh_is_sound`, because redistribution leaves `tris.normal` unspecified by design. The expression must
  stay identical in the walk and stream kernels or `test_triangle_geometry_paths_are_bit_identical` fails.
- **Face normals point outward, everywhere.** The icosahedron face table in `src/mesh_creation/sphere_creation.c`
  is wound counter-clockwise as seen from outside, and the 1-to-4 subdivision emits every sub-triangle in its
  parent's rotational order. This is what makes the enclosed volume positive under the divergence theorem and fixes
  the sign of surface tension and of the indicator function. Guarded by `test_triangle_winding_is_outward` and
  `test_enclosed_volume_is_positive_and_converges`.
- **"No twin" and "mesh boundary" are different statements, and only one is partition-independent.**
  `twin_idx == MFTL_ERROR` means "no partner in THIS partition" — true at the rim of an open surface *and* at the
  outer edge of a ghost halo. `flags & MFTL_FLAG_BOUNDARY` means "no partner anywhere in the mesh"; it is written
  once by `MFTLMarkMeshBoundary` while the mesh is still complete, and from then on travels with the entity.
  Re-deriving it from `twin_idx` after redistribution would promote every halo rim into a fake boundary.
- **A boundary vertex has one more neighbour than it has incident triangles.** The closing edge of an open fan is
  reachable only as an *incoming* half-edge. `MFTLGetVertexOneRingVertices` handles it;
  `MFTLGetVertexOneRingHalfedges` deliberately does not (its contract is outgoing half-edges). Guarded by
  `test_one_ring_at_a_boundary_vertex_has_one_extra_neighbour`.
- **A spatial ghost is a triangle in a SHADOW OBJECT, not an entity in the body's pools.**
  `MFTLExtendSpatialHalo` (`src/communication/spatial_halo.c`) puts what the neighbours sent into
  `MFTL_SpatialHalo.patch[b]` — an ordinary, self-consistent `MFTL_Object` with `n_local == 0`, twins
  linked within the patch, derived arrays built. It is deliberately NOT in `ens->objects`, so
  redistribution and both ghost syncs never see it. The BVH borrows it and stores no geometry of its own;
  `MFTL_BVHPrimVerts` resolves a primitive to (object, triangle) and both kinds go through
  `MFTLGetTriangleCorners`.

  Appending them to the body's own ghost suffix instead — which looks free, because the ownership prefix is
  untouched — fails for two reasons that are invisible until they bite.
  First, four passes bounded by `tris.n` rather than `n_local` change SILENTLY: the Morton bounding box in
  `reorder_vertices` (a partner body across the domain coarsens the cells for the *owned* mesh),
  `MFTLMeanEdgeLength` (each appended triangle contributes three twin-less half-edges, so the remeshing
  target length is set partly by another body — and it carries no ghost assertion), the vertex-normal
  scatter, and `MFTLRebuildTwins` (which re-pairs by vertex SLOT and can weld two bodies together).
  Second, there is no good answer for the arriving vertex GIDs: de-duplicate them onto existing slots and
  `inv_halo_is_vertex_complete` breaks; give them fresh slots and `inv_local_global_ids_are_unique`
  breaks. Both are in `inv_partition_is_sound`. Doing it properly needs the two-way ownership split to
  become three-way, i.e. a change to invariant 2 of `mftl_types.h`. Guarded by
  `test_halo_does_not_touch_the_bodies` and `test_halo_patch_is_a_valid_object`.

  Guarded by `test_halo_does_not_touch_the_bodies` and `test_halo_patch_is_a_valid_object`.
- **The spatial halo is TRANSIENT and cannot be otherwise.** Redistribution keeps only owned triangles
  (`collect_destinations` scans `[0, tris.n_local)`), so a halo cannot survive a repartition even in
  principle, and the documented loop redistributes every step. Each `MFTLExtendSpatialHalo` REPLACES the
  previous patches rather than accumulating; grow it after redistribution and before contact detection.
- **An extent in `src/communication/spatial_halo.c` is a CENTRE ON THE CIRCLE plus a HALF-WIDTH, never a
  `(lo, hi)` box.** Applies to what a rank publishes about its share and to each triangle's own contribution;
  "near" is the minimum-image distance between centres against the sum of half-widths. A box gets it wrong in
  both directions at once. It MISSES: two shares meeting across the seam are a domain length apart in stored
  coordinates, so min/max says they do not touch — and because the classification decides what is *sent*, the
  miss lands one stage before anything `MFTL_IntersectObjects`'s periodic image loop can see, so nobody
  reports the contact. It OVER-SENDS: a share holding one seam-straddling triangle has a box spanning the
  whole domain, because `MFTLGetTriangleBounds` is correctly seam-consistent and puts that triangle's corners
  just outside the box on one side — so the share overlaps everything and every neighbour ships its entire
  share of every partner. Measured at np = 27 on a seam contact: 314 triangles ghosted with boxes against 56
  with intervals, and 320 against 0 for a placement with no cross-rank contact at all. A rank's own extent is
  measured in two frames, raw and shifted by half a domain, keeping the tighter per axis; that is what makes a
  straddling share report its true size, and it is the same trick `MFTLSpatialHaloResolveRadii` needs so a
  droplet on the seam does not report a bounding radius of half the box (which would scale every envelope
  derived from it). Payload is six doubles per body per rank either way. Guarded by
  `test_contact_through_the_seam_survives_the_decomposition` and
  `test_a_seam_straddling_body_is_not_measured_as_domain_sized`.
- **A pre-filter must be strictly MORE permissive than the test it short-cuts, and the halo's macro test
  carries the margin for that reason.** The per-triangle intervals are widened by the margin, so an un-widened
  macro extent is *stricter* than what it stands in for and rejects the whole neighbour in exactly the regime
  the margin exists for: two shares that do not overlap but are within the contact envelope of each other.
  Overlapping bodies hide it completely — their raw extents intersect anyway — so only a PROXIMITY contact at
  a rank boundary reveals it, which is what `test_the_tolerance_threshold_is_exact_from_both_sides` is.
- **Periodicity is TWO problems in the BVH and they need two different fixes.** Within an object, a
  triangle straddling a seam has wrapped vertices, so a centroid built from the raw coordinates lands in
  the middle of the domain and its AABB spans the box — the corruption a centroid normalisation on a periodic
  domain always invites, fixed by gathering current corners and applying
  minimum-image edges. Between objects, two bodies on
  opposite faces are neighbours and no per-triangle arithmetic can see it; that is fixed by traversing
  the pair once per periodic IMAGE (up to 27, each rejected by one root-AABB test). A straddling object's
  ROOT box legitimately spans the domain and is not a bug — same accepted cost, for the same reason, as
  `MFTLObjectReorderMorton`. Guarded by `test_periodic_morton_of_a_straddling_object` and
  `test_contact_across_a_periodic_face`.
- **The contact envelope scales with the body and must not depend on the process count.**
  `tol = MAX(proximity_floor, proximity_factor * bounding_radius)` replaces a hardcoded 1e-4 (AABB inflation)
  and 1e-6 (proximity), both absolute and therefore correct at exactly one length scale:
  1e-4 is a tenth of a 1 mm droplet's radius, and 1e-6 is 1e-6 of a 1 m body's. `bounding_radius` is half
  the LONGEST SIDE of the root box, not half its diagonal, so it equals `R` for a sphere and `factor`
  means what it says. It is measured over one rank's partition, so `MFTL_CreateSpatialGhosts` overwrites
  it with the body's GLOBAL radius (`radius_hint`) from one `MPI_Allreduce` — otherwise splitting a
  droplet over eight ranks would change the separation at which contact is detected. Guarded by
  `test_detection_at_two_disparate_length_scales` and `test_tolerance_is_independent_of_the_partition`.
- **A distributed contact is reported by the rank that owns the A-SIDE primitive, and only that rank.**
  So a BVH is built over OWNED triangles only (including the redistribution halo makes two ranks report
  the same contact, exactly as writing ghosts into a `.vtu` double-counts the surface), only object B's
  triangles travel, and the traversal skips any leaf pair whose A side is remote. Get it wrong in one
  direction and cross-rank contacts are missed; wrong in the other and every cross-rank contact force is
  applied twice. `test_distributed_contacts_match_the_serial_answer` compares the gathered SET against
  the answer computed on one rank while it still held the whole mesh, which is the only check that rules
  out both at once.
- **`MFTL_Orient3D` is adaptive; nothing else in `src/bvh/` is, and that is deliberate.** It is the only
  predicate whose SIGN decides the answer — Guigue–Devillers branches on it up to EIGHT times per pair
  (six to classify the corners against the two planes, two in `check_min_max`; the permutation cascade
  evaluates none, it branches on signs already computed), and a wrong sign gives a clean "no
  intersection" for two triangles that cross. Everything else there
  degrades continuously. `examples/orient3d_predicate_profile.c` measures the trade both ways and the
  answer is not one-sided: the plain determinant did not misclassify once in 20 000 samples at gaps down
  to 1e-12 of the body size (three orders below where a lubrication model works), and only fails at
  1e-14…1e-16. It is kept because the cost is ≤ 6.2 % of a full `MFTL_TriTriProximity` call (2.00× on the
  predicate, but the narrow phase is dominated by its fifteen distance sub-tests) and the failure it
  removes is silent. **`src/bvh/bvh_predicates.c` must be compiled with `-ffp-contract=off`** (set in
  `CMakeLists.txt`): an FMA contracts `x - a*b` into one rounding, `Two_Product`'s residue comes out zero,
  and every exactness guarantee evaporates while the numbers still look plausible.
- **A half-edge's global ID is `3 * triangle_gid + k`, mesh-wide.** Established by
  `MFTLBuildTopologyFromIndexList`, preserved by `MFTLObjectReorderMorton` (which moves half-edge triples as a
  unit) and by redistribution. It is the wire-format half of implicit indexing: half-edge global IDs are never
  transmitted, the receiver reconstructs them. Guarded by `inv_halfedge_gids_are_implicit`.
- **Global IDs are a dense permutation of `[0, n)` per entity class**, and `next_*_gid == n_*` after generation.
  Subdivision retires the whole previous generation of triangles and half-edges, so those counters restart at zero;
  `next_vertex_gid` does not, because old vertices are carried over and keep their IDs. Density is what lets a
  receiver size a lookup table by ID range instead of hashing. Guarded by `inv_global_ids_are_dense`.

## Performance

`docs/performance-analysis.md` holds the measured picture, `docs/remeshing-quality-analysis.md` the
equivalent for the remesher, and `docs/parallel-remeshing-analysis.md` the distributed one — correctness
against serial, bitwise determinism, phase counts and scaling, all from
`examples/leveque_remeshing_parallel.c`. Read it before optimising anything, and re-run the
benchmarks rather than trusting the tables — each answers one question:

| benchmark | question |
|---|---|
| `layout_benchmark.c` | interleaved vs padded vs split xyz, on four access patterns |
| `parallel_advection_redistribute_list.c` | what a whole timestep costs, how much of it is communication, and (`--ab N`) what skipping unchanged bodies is worth |
| `orient3d_predicate_profile.c` | does the plain orientation determinant misclassify thin-film contact, and what does the adaptive one cost |
| `leveque_remeshing_parallel.c` | over 150 remeshed timesteps, does the distributed answer track the serial one, is it reproducible to the bit, and what does a phase cost |
| `parallel_advection_scaling.c` | strong, weak and granularity scaling of the timestep; `run_scaling.py` sweeps the rank counts and renders the curves |

**Nine results contradict what the design predicted, and are the reason to read the doc rather than the code
comments:**

1. Persistent triangle frames made isolated consumer kernels faster, especially
   periodically, but their build cost and 72 B/triangle footprint did not
   amortise in the moving/remeshing workflows. They were removed; geometry now
   dispatches only between compact-index and construction-time half-edge paths.
2. Morton renumbering buys no measurable time on Apple silicon despite cutting mean edge index distance 160×.
   **Needs re-measuring on an x86 node with 1–2 MiB L2 per core.**
3. Splitting `verts.pos` into three component arrays is faster for gathers and *slower* for scatters, by similar
   amounts. It is a wash, so the layout is decided on the API instead. The mechanism is address generation
   (a 24-byte stride needs a multiply; an 8-byte stride does not), not cache lines.
4. **The update kernel does not fail to vectorise because of the nine-way reduction** — the obvious diagnosis,
   and the wrong one. Splitting the reduction makes it slightly *worse*, and clang refuses the transformation
   even when a pragma forces it. 26 % of the pass is one `sqrt` plus one divide, which is irreducible.
5. **The quality statistics are not nearly free.** "A few extra flops on data already in registers" predicts a
   few per cent; they cost **+90 %** (1.64 → 3.11 ns/triangle). The streaming geometry baseline is fast, and
   `q = 4√3·A/(a²+b²+c²)` adds a *second* divide to a loop already limited by one (§4).
   Fusing is still right (two passes cost 2.20 ns against 1.47 fused), but `MFTLEnsembleUpdate` is **slower than
   `MFTLComputeTriangleGeometry` on a non-periodic domain** while computing strictly more. Call it for the
   statistics, not as a free upgrade. On a periodic domain — what a real run has — it wins at 1.19×.
6. **Skipping unchanged bodies is worth 1.6-2.1x of the redistribution phase, and the number it turns on is a
   BODY fraction that nothing had ever measured** - 77-86 % at 1000 bodies, np = 2-4, against an estimate of
   60-75 % made from a radius/cell-width ratio. The former position-derived
   rebuild became the largest non-collective bucket at 15 % and was subsequently
   removed together with persistent frames.
   obvious next change. And the buckets that shrink least are the ones proportional to the HALO rather than to
   the mesh - twin repair 1.05x against 3.6-5.2x for everything per-entity - because the bodies a skip removes
   are the interior ones with the least halo churn. Full table in `docs/performance-analysis.md` §7e.

7. **MPI-IO did not make output faster; dropping ASCII did.** A shared collective file is 1.0× at one rank and
   0.45× at four against one file per rank on a laptop SSD, because there is nothing for two-phase buffering to
   aggregate. The 23–32× came entirely from raw binary replacing `printf`. `MFTL_IO_SINGLE_FILE` is kept for the
   property that was NOT measured — P+1 files per dump is what stops working first on a parallel filesystem.

8. **The LeVeque remeshing test's poor final mesh was caused by FEATURE DETECTION ON A SMOOTH SURFACE**, not
   by any single operator. The sphere has no features, so every flagged edge is a false positive, and each one
   removes smoothing degrees of freedom from vertices that needed them. Four spurious edges are enough: at the
   step they appear, the worst triangle in the mesh drops from 0.797 to 0.045 and then FREEZES there, because a
   constrained vertex is simultaneously exempt from smoothing, from collapse and from flip. Final minimum
   quality is 0.76 with detection off against 0.0002 at 45°, monotone across five thresholds. Two secondary
   results contradict the obvious guess: the collapse fold-over guard is worth nothing measurable on a correct
   configuration but bounds a runaway once features are spurious (16 702 triangles against 101 048), and the
   MISSING AREA WEIGHTING in "triangle-based" smoothing costs volume conservation (3.5 % against 6.0 %) rather
   than triangle quality, which is a wash. Set `feature_angle_threshold` to 180 for any smooth closed
   interface. Full measurement in `docs/remeshing-quality-analysis.md`.

9. **MOVING WORK OUT OF A STREAMING PASS INTO A DEPENDENCY-BOUND LOOP RECOVERS A THIRD OF IT, NOT ALL OF IT.**
   Redistribution used to close with `MFTLObjectBuildDerived`; `rebuild_object` now emits `tris.verts` /
   `tris.neighbors` inside the relink, where every input is already in a register. The traffic argument
   (−48 B/tri, one pass fewer) predicted 5 % of the phase. Measured: **the pass costs 0.588 ms/step and
   0.369 of it comes straight back into `t_relink`**, so the trade is 0.22 ms/step, 2.3 %. What the model left
   out is WHICH loop the stores land in — `build_topology_arrays` is a pure stream the prefetcher handles
   perfectly, the relink already does two random gathers per half-edge. It is kept (one pass fewer over the
   pool is a footprint statement too, and §7c says footprint is what has mattered), but the
   destination-analysis items it was ranked ABOVE are the ones carrying the win: `build_kept_lists` reading
   the dense owner arrays in one two-cursor pass instead of the AoS records in four scans, plus a branch-free
   stencil test in `collect_destinations`, take **20 % off the destination analysis** in both regimes. All
   four together: 1.038–1.040x on the phase where the body skip cannot fire, 1.02x where it can.
   `docs/performance-analysis.md` §7f closes out `docs/redistribution_optimizations_review.md`.

## Known deferred issues

1. **`MFTLObjectReorderMorton` has no proven benefit on the development machine.** It is kept because it is what
   makes one index space possible (its real justification) and because it is a prerequisite for an LBVH — not
   because it was measured to be faster. Decide its place in a per-timestep loop only after running the benchmark
   on the cluster.
2. **The half-edge triple invariant is unenforced at the type level.** Nothing stops a future remeshing routine
   from creating half-edges out of triple alignment; it would corrupt every traversal silently.
   `inv_implicit_indexing()` catches it in tests, and `MFTLCheckMeshIntegrity` does not. Consider asserting it in
   `MFTLObjectSetCounts` when remeshing lands.
3. **`tris.verts` / `tris.neighbors` are redundant state whose benefit is argued, not measured.** The 4× traffic
   figure is arithmetic. Given finding §3 in the performance doc, re-examine when a consumer that isolates them
   exists (redistribution's halo analysis is the first).
4. **The update kernel sits on the FP divide/sqrt unit and cannot be vectorised out of it.** 26 % of the pass is
   one `sqrt` plus one reciprocal per triangle, and the kernel needs both `|n|` and `n/|n|`, so neither is
   algebraically removable. A hardware `rsqrt` estimate plus Newton iteration is the only remaining lever; it was
   NOT measured, it costs exact reproducibility and portability, and it should not be attempted before the same
   benchmark has been re-run on the cluster — the divide/sqrt throughput ratio is the whole question and it is
   microarchitecture-specific.

5. **The ghost-exchange SCHEDULE is rebuilt from scratch every timestep, and could be cached.** Only the holder of
   a ghost knows that it holds it, so the request phase of `MFTLUpdateGhostPositions` cannot be removed — but it
   could be amortised: after a redistribution the schedule is fixed until the next one, so the owner side could
   keep its resolved slot list and every subsequent sync would be a single `MPI_Neighbor_alltoallv` of positions
   with no lookup at all. Not done, because the per-timestep loop this library is built around redistributes every
   step and there would be nothing to amortise over. Revisit if a solver appears that redistributes every N steps.
   It is also the reason the ghost sync improved by less than its 2.5× payload reduction (performance doc §7).

6. **The per-destination send buffers are allocated and freed on every call.** At the density-derived hint they
   are ~128 KB each, so a redistribution mallocs and frees a few megabytes per timestep and pays the first-touch
   faults again each time. That churn is what makes a first benchmark run slower than every repeat of it
   (performance doc §7). Keeping them as scratch on the domain would remove it; not
   done because it puts mutable per-call state on a struct that is otherwise read-only during a redistribution.

7. **Step 5 of the per-timestep loop is bound by first touch, not only by arithmetic.** The geometry pass that restores
   what redistribution drops runs on pools the redistribution has just rewritten. The remaining lever is fusing the recompute INTO the redistribution's
   rebuild, where the arrays are already in registers and already being written. Not attempted; it would put
   geometry into a module that deliberately transmits none.

9. **The two I/O file layouts have only been compared on a laptop with one SSD**, which is the configuration least
   able to tell them apart, and the one where the shared file cannot win. Nothing here says how a collective write
   to one file behaves against P per-rank files at P in the thousands on Lustre or GPFS — which is the only
   question that decides between them. The shared-file backend also passes `MPI_INFO_NULL`; collective-buffering
   hints are untouched and are where the remaining tuning lives.

10. **Nothing has been measured above 4 ranks on real hardware.** The development machine has 4 performance cores,
   so an 8-rank run puts half the ranks on efficiency cores and the absolute per-triangle costs roughly triple.
   Ratios measured within one rank count survive that, but the SCALING of the exchange — global path versus graph
   path, and the behaviour at P in the hundreds — is entirely unmeasured.

11. **The phase count is now measured on a real deforming case, and the open question moved.**
   `docs/parallel-remeshing-analysis.md` measures the LeVeque deformation with per-timestep distributed
   remeshing at np = 1, 2, 4, 8. At a fixed `parallel_phase_budget` of 4 the phase count **saturates**
   (3.83 → 3.96 from np = 2 to np = 8), overhead on a fixed problem grows 33 % for 4× the ranks, and the
   distributed mesh is *better* than the serial one on every quality measure at an adequate resolution.
   The 4/6/19/17 figures that worried this entry were a ONE-SHOT refinement running to a fixed point; the
   growth was the mesh being far from its target, not the partitioning.

   **What replaced it: `parallel_phase_budget`'s default of 32 is wrong for a per-timestep loop, and
   quietly so.** It means "run until no rank wants to do anything", which re-converges an already-good
   mesh every step. Measured at np = 4: budget 4 gives V/V0 0.771 and min q 0.879 at 16.7 ms/step;
   budget 32 gives 0.639 and 0.621 at 84.7 ms/step. Worse on every axis and five times slower. The default
   is kept because the one-shot case fails LOUDLY when the budget is too small (`converged` comes back
   false) and the loop case fails quietly when it is too large — but a solver must set it. Full table on
   the field in `mftl_remeshing.h`.

   Still open above np = 8 and on real hardware: the development machine has four performance cores, so
   np = 8 is oversubscribed and every absolute time above np = 4 is a lower bound. Whether migration cost
   stays proportional to the band at P in the hundreds needs a cluster and a mesh where a rank holds tens
   of thousands of triangles rather than a few thousand.

18. **Two consequences of coarsening are argued rather than measured, and both need P in the hundreds.** A
   coarse destination is `group[k]` cells away, so it is outside the 26-neighbourhood and BOTH
   `MFTLRedistributeWithOwners` and `MFTLRemeshSyncGhostIds` fall back to their global paths — `O(nproc)`
   allocations and an `Alltoall`, which is deferred issue 6 above, now reachable once per remesh rather
   than never. And redistribution's 32-bit destination de-duplication (`MFTL_DEDUP_BITS`) reverts to a sort
   above 32 distinct remote destinations, a path np <= 27 cannot reach and nothing therefore tests. The
   GATHER is exempt from the second — every rank sends to exactly one destination.

19. **`MFTLUpdateGhostPositions` has no global fallback and drops silently, so a coarsened partition must
   never outlive a remeshing call.** `src/communication/ghost_sync.c` counts a ghost whose owner is not a
   graph neighbour as `orphaned` and skips it; the file records that the fix is not done because that
   function runs once per TIMESTEP rather than once per remesh and may not grow an O(nproc) path without a
   measurement. It is safe today only because both drivers close with a plain `MFTLRedistribute` back to
   the canonical fine partition. Anything that makes a coarse partition persist has to fix this first.

12. **The BVH's leaf-leaf narrow phase is batched but NOT vectorised.** `flush_batch` in
   `src/bvh/bvh_traverse.c` receives a contiguous run of up to 64 independent (primA, primB) tests, which
   is the shape a SIMD kernel wants, and runs them scalar. The batch exists so that vectorising is a
   change to that one function; it was not attempted because `MFTL_TriTriIntersect` is a decision tree
   with early-out at every level, so the useful form is a gather of the FIRST orientation test across a
   batch rather than a naive widening — and that is a measurement exercise, not a transcription.
   `MFTL_TriTriProximity` measures 176 ns/pair against `MFTL_TriTriIntersect`'s 31, so the fifteen
   distance sub-tests are where the time is and where the vectorisation should start.

13. **The spatial-halo overlap list is rebuilt from scratch every call, and could be cached** when the
   decomposition is static. Not done, for the same reason as deferred issue 5 above: the per-timestep loop
   this library is built around redistributes every step, so every rank's extent for every body changes
   every step and the cache would be invalidated exactly as often as it was built. If a solver appears
   that redistributes every N steps, it belongs on `MFTL_SpatialHalo`, which the caller already owns —
   NOT on `MFTL_Domain`, which is read-only during a redistribution.

   The same handle is where a SURVIVES-REDISTRIBUTION halo would go if one is ever wanted. It is not
   wanted yet: `collect_destinations` keeps only owned triangles, so preserving a halo means teaching
   redistribution to forward it, stopping `compute_owners` clobbering its `owner_proc`, and satisfying
   `centres_are_current` for it — and the halo is stale exactly when ownership changed, which is the
   moment it would have to be right.

14. **Nothing in `src/bvh/` has been profiled at all beyond `orient3d_predicate_profile.c`.** The
   explicit-stack traversal, the refit-only fast path and the batched leaf phase are all argued from
   structure rather than measured; there is no BVH entry in `docs/performance-analysis.md`.
   Build/refit/traverse timings are the first thing to measure before optimising any of it.

15. **Remeshing renumbers global ids at the end of every pass, and it has to.** Splits hand out fresh ids and
   collapses retire them, so ids stay unique but stop being dense. Density is not cosmetic here:
   `src/communication/redistribute.c` and `ghost_sync.c` size their arrival lookup tables from `next_*_gid`,
   so holes accumulated over a few hundred timesteps become an unboundedly growing table for a mesh of
   constant size. The renumbering is what keeps `inv_global_ids_are_dense` true; the cost is that pool indices
   AND global ids are both unstable across a remesh, so nothing external may hold either.

16. **Contact detection is DISCRETE-TIME: there is no CCD, and two bodies can tunnel through each other
   between rebuilds.** `MFTL_GenerateLBVH` and `MFTL_RefitLBVH` see one instant; nothing sweeps the volume
   between two. A pair whose relative displacement in one step exceeds the sum of their sizes passes through
   without ever being coincident at a sampled instant, and both samples correctly report nothing. This bites
   when the timestep is set by a CFL condition on the FLOW rather than by the approach velocity of two
   surfaces, which is the normal case. Fixing it needs the swept volume of every triangle — a different
   primitive, a different tree, and a per-step velocity the module does not receive. Pinned as a recorded
   property rather than an assumption by `test_tunnelling_between_two_timesteps_is_a_known_miss`, which
   asserts the miss; if that test ever starts failing, something has begun reporting contacts that are not
   present at the sampled configuration, which IS a bug.

17. **A pair's envelope is `MAX(tol_A, tol_B)`, so a µm droplet against a m body is judged at the metre
   body's scale.** The maximum is the only safe direction for the broad phase — each tree's leaves are
   inflated by its own tolerance, so admitting the larger keeps the narrow-phase threshold inside what the
   traversal can deliver — but it means the envelope of a pair is set by the larger body and can exceed the
   smaller body's own diameter. The escape exists and is documented on the field: set `proximity_factor` to 0
   and state the physical gap as `proximity_floor`. Characterised, not asserted away, by
   `test_two_length_scales_share_the_larger_envelope`. Whether a per-pair rule that is neither the max nor
   the min is wanted is a physics question, not a code one, and has not been asked.

20. **THE GENERATION PEAK IS RANK-0 SERIAL AND STILL UNFIXED; THE STRANDED CAPACITY IT USED TO LEAVE
   BEHIND IS FIXED, AND WAS NEVER THE LARGER HALF.** This entry used to say that a generated
   ensemble's whole footprint stayed resident on rank 0 for the run. That was two claims welded
   together, and measuring them apart is what this entry now records.

   `MFTLGenerateSphere` builds every body whole on rank 0 — `5120 * 4^refine + 176` bytes each,
   80 MB for 256 bodies at refine 3, 640 MiB for 512 at refine 4. **That peak is real, it is serial,
   and nothing here touches it**: every body exists at full size before the opening
   `MFTLRedistribute` runs. `examples/parallel_advection_scaling.c` prints it and refuses above
   `--max-gen-mb`. Fixing it needs a generator that builds distributed, which does not exist.
   Batched generation is NOT the fix — it shaves only the packed-buffer transient and costs
   `O(N^2/K)` because `collect_destinations` re-derives the whole halo per call (measured at 512
   bodies, K = 32: ~8x the setup work for a ~30 % cut).

   What happens to that 80 MB after the scatter splits in two, and the split is the useful part:

   - **Bodies rank 0 keeps NOTHING of** are released outright — `MFTLObjectSetCounts(0,0,0,0)` plus
     `MFTLObjectReleasePools` in `redistribute.c`, added in commit `2272622`. This is most of them,
     and it is why the old wording was wrong. Measured at 256 bodies, refine 3, np = 4: **80 MB
     generated, 33.5 MB of pool capacity left on rank 0 afterwards.**
   - **Bodies rank 0 keeps a SLIVER of** hold a full-mesh block against a few dozen triangles.
     `MFTLObjectShrinkToFit` (`src/object/object.c`) is the fix, armed per call by
     `MFTLRedistributeSetShrinkSlack`.

   **The residual is single-digit percent, which is why the shrink ships OFF.** Growth is a ratchet,
   so over a run every rank accumulates, per body, the largest share it ever held. Measured at
   np = 4, 128 bodies, refine 3, 200 steps — capacity figures are bit-reproducible across runs:

   | | slack 0 (off) | slack 4 |
   |---|---|---|
   | end capacity, max/rank | 22.8 MB | 20.8 MB (−8.8 %) |
   | end capacity, total | 73.3 MB | 68.0 MB (−7.2 %) |
   | stranded against live | 1.79x | 1.66x |
   | redistribute ms/step, 3 runs | 0.906 / 0.931 / 0.960 | 0.941 / 0.960 / 0.962 |

   **8 % of capacity for about 2 % of the redistribution phase, i.e. a TRADE**, which is the shape
   this library ships disabled (`coarsen_frozen_fraction`, `serial_gather_max_triangles`) rather
   than the shape it ships armed. Most of the remaining 1.66x sits in the 1x–4x band the gate
   declines on purpose: the gate cannot go to 2 without thrashing against the doubling growth
   ladder, and `MFTL_SHRINK_SLACK_DEFAULT` derives why. Arm it when bodies genuinely shrink —
   remeshing that coarsens, or a long run where shares drift far — not as a default.

   **Measure it in RSS, never in a byte counter.** Whether `free()` returns pages to the OS is the
   allocator's decision (glibc raises its mmap threshold dynamically up to 32 MiB, after which
   blocks come from `brk`), so `n_bytes_reclaimed` is an upper bound on the win.
   `current_rss_bytes()` in the benchmark is the measurement, and it is deliberately not
   `peak_rss_bytes()` — a high-water mark cannot fall.

   Generation remains the benchmark's Amdahl ceiling: it is rank-0 serial, so in a WEAK scaling
   sweep setup time grows linearly with `nproc` while the timed loop does not, and it must never be
   folded into a scaling total. Full reasoning in `docs/scaling-analysis.md`.

21. **`compute_owners` IS NOW THE LARGEST NON-COLLECTIVE BUCKET, AND THE BODY SKIP CANNOT TOUCH IT BECAUSE
   IT IS THE DECISION.** With the skip on and the frames gone it is 1.4 of 4.7 ms/step at np = 4 on 1000
   bodies — 30 % of the phase — and it is the one bucket that does not move when a body is skipped, because
   the classification is what produces the verdict.

   **The sufficient condition that would remove it needs no scan and no collective, only a number the
   caller already has.** The question is "did any entity change cell". Cache per body the minimum distance
   from any entity to its own cell's boundary planes — `slack[o]`, three `fabs`/`min` per entity, in
   registers, inside the loop that is already reading the position. A flow solver knows how far anything
   can have moved since: it is `dt * v_max`, the CFL number that set the timestep. Then
   `motion_bound < slack[o]` PROVES no entity crossed a plane, so the body votes clean and the
   classification is skipped; `slack[o] -= motion_bound` on each skipped step, rescan when it reaches zero.
   Exact, conservative, and it amortises one scan over `slack / bound` steps — a droplet of radius 0.1 in a
   0.5-wide cell moving 0.01 per step would scan about every fifteenth step instead of every step.

   **It must ship OFF, and the reasons are the interesting part.** It adds work to `compute_owners` — the
   loop this file records a 1.70x regression on — in order to save that same loop later, so the extra work
   has to be selected by one branch hoisted per body exactly as the `MFTL_OwnerMap` seam is. It moves a
   correctness precondition to the CALLER: a solver that understates its bound gets a silently wrong
   partition, so it needs a Debug-only verifier that runs the scan anyway and asserts the verdict, and the
   switch belongs in `check_object_lists_agree`'s checksum the way `g_skip_unchanged` already is. It must
   be disabled whenever `maps != MFTL_NULL`, because the remesher moves the cut planes and a cached slack
   then describes planes that no longer exist. And `slack[]` belongs in the `comm_workspace` next to
   `dirty_votes`, under the same `prev_num_objects` invalidation rule — NOT on `MFTL_Object`.

   Fusing the classification into `MFTLComputeTriangleGeometry` instead, which writes `tris.center` one
   step earlier and would save the 24 B/tri stream, was rejected: it covers only the triangle half (nothing
   in the library writes `verts.pos` — the solver does) and it puts routing into the geometry API, which is
   the boundary deferred issue 7 declines to cross.

### Settled, so nobody re-opens them

- **The vertex→triangle CSR stays deleted.** `src/object/derived.c` asked for it to be reconsidered when a consumer
  appeared. Vertex normals are that consumer, and §5 of the performance doc measures a CSR gather as a dead heat
  with the triangle scatter while costing 16 B/triangle of permanent state and a rebuild per topology change.
- **Quality statistics stay fused with area/normal/center.** Splitting them into two passes measures 25 % slower.
- **`verts.pos`, `tris.normal` and `tris.center` stay xyz-interleaved**, and `pad4` (32 B/vertex) is rejected.
