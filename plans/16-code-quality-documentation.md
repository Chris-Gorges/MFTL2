# 16 — Repository-wide code-quality cleanup + MFTL2 manual

Status: NOT STARTED
Depends on: 15 (must be COMPLETE and fully ACCEPTED before Stage 16 begins)
Implemented by:
Reviewed by:
Acceptance: NOT PASSED

## Objective
A repository-wide cleanup of MFTL2 — comments, naming, file organization, and a new user/
developer manual — **without intentionally changing numerical behaviour or public
functionality**. This stage runs only after the MFTL1→MFTL2 migration (Stage 15) is complete
and accepted, so the code, public API, and multiflow coupling are stable and the manual can
describe MFTL2 *as it actually exists*. Nothing here should alter results: it is a
readability, consistency, and documentation pass, plus carefully-scoped structural refactors
that preserve the build, the API, and Git history.

**Hard rule for the whole stage:** behaviour-preserving. Any change that could alter a
numerical result, a public signature, a struct layout, or an MPI wire format is out of scope
and must be split out into an explicitly-approved follow-up (see "Change taxonomy"). When in
doubt, don't.

## Relevant existing code (state at the time of writing, before Stage 14/15 land)
- **Layout.** `src/general/` (15 files, a catch-all mixing geometry, communication, BVH/
  intersection, I/O, generators, sorting), `src/remeshing/` (4 files), `include/` (17 headers,
  **all currently treated as public**, no public/private split), `tests/` (30 files, explicitly
  registered in `tests/CMakeLists.txt`), `examples/` (5). Root `README.md`.
- **Build.** `CMakeLists.txt` collects library sources with `file(GLOB_RECURSE MFTL_SOURCES
  "src/*.c")` — new/split/moved `.c` files under `src/` are picked up automatically on the next
  `cmake` configure (the GLOB caveat: a bare `make` won't notice new files; re-run `cmake`).
  Examples are globbed; tests are registered one-by-one. `MFTL_USE_LAPACK` is **OFF by
  default** (LAPACK is optional, contrary to the current `README.md`, which lists it as
  required). Build via `python3 build_all.py [--debug|--valgrind]`; test via `ctest`
  (note: `/home/berend/bin/ctest` is a broken shim — the real binary is `/usr/bin/ctest`).
- **Largest files (LOC), i.e. split candidates:** `communication.c` 3842, `remeshing_core.c`
  1880, `mftl_bvh.c` 1359, `overlap.c` 1215, `remeshing_parallel.c` 1029, `volume_conserving.c`
  910, `io.c` 888, `topology.c` 734.
- **File-header styles (three, inconsistent):** (a) editor boilerplate
  (`@Author: Christian Gorges / @Date / @Last Modified by / @Last Modified time`) on the
  original MFTL2 files; (b) hand-written descriptive purpose headers on the migration-added
  files (e.g. `geometry_object.c` — the good style); (c) none (e.g. `mftl_bvh.c` opens on
  `#include`).
- **Comment styles.** Migration stages added many genuinely valuable comments (periodicity
  minimum-image reasoning, MPI ownership/authority in the redistribution heap fix, sign/tolerance
  conventions, curvature `2/R` convention, overlap prism sign). Mixed in are tutorial/AI-style
  artefacts: numbered "step" section dividers (`// --- 1. Create MPI Datatype …`, `// 2. Pack
  the 3 Half-Edges`), conversational narration ("we/our", "Zero Copy", "Blind Fill"), and
  restate-the-code lines. Only **3** `TODO`s remain (`communication.c:1151`/`:1400` linear
  `target_obj` search → hash map; `:2156` buffer-size heuristic).
- **Naming (measured).** Public functions: ~177 `MFTLCamelCase` + ~33 `MFTL_Snake` (math
  helpers/macros). Public types: `MFTL_PascalCase` (`MFTL_Object`, `MFTL_Vertex`,
  `MFTL_Triangle`, `MFTL_HalfEdge`, `MFTL_Domain`, `MFTL_Ensemble`, …). Internal `static`
  helpers are **inconsistent**: `mftlCamelCase` (migration-added) vs `snake_case`
  (`new_triangle`, `new_vertex`, `permute_vertices`, `buffer_init`, …). Locals mix `snake_case`
  and `camelCase`.
- **Known duplication (surfaced in the Stage-11/12/13 reviews):** two collective-anchor helpers
  (`MFTLObjectUnwrapAnchor` in `geometry_object.c` vs `mftlPQUnwrapAnchor` in `point_query.c`);
  two `unwrapToAnchor` spellings (header inline `MFTLUnwrapToAnchor` + file-local
  `mftlUnwrapToAnchor`); the ~500-line receive-side reconstruction duplicated between
  `MFTLRedistributeObject` and `MFTLRedistributeListOfObjects`; small vector/geometry helpers
  re-implemented across `overlap.c`/`curvature.c`/`vertex_fit.c` vs `mftl_vectoroperations.h`.
- **Reference documentation.** MFTL1 manual `/home/berend/git/Code/mftl-doc/MFTL.tex` (2558
  lines, LaTeX; chapters: Introduction, Data structure, Distribution, Generation, Redistribution,
  Properties, Basic functions, Hierarchical/bounding-box, Point functions, Remesh, Intersection,
  Output, Init, Cleanup, Example listing, Future work; with `figs/`, equations, `lstlisting`
  code blocks). Conceptual and well-structured — the right *level* to emulate — but its data
  model (uthash `Edge`/`Vertex`, `id`, `processor`, `MFTL_Point`, OBB tree) is **obsolete** for
  MFTL2 and must not be reproduced.

## Proposed changes — by substage
Each substage leaves the tree compiling and the full suite green, and is independently
reviewable. Recommended order **16A → 16B → 16C → 16D → 16E**; 16D (manual) may proceed in
parallel with 16A–16C since it touches no source.

### 16A — Comments & source documentation (purely cosmetic/doc)
- **Standard file header.** Adopt one concise header for every `.c`/`.h`: one-line purpose,
  the subsystem, and (where relevant) the numerical/parallel contract the file implements.
  Retire the editor `@Author/@Last Modified` boilerplate (Git already tracks authorship/dates);
  keep an `Author:`/`Part of MFTL2` line only if the project wants attribution. Apply the
  `geometry_object.c`-style descriptive header everywhere; add one to the header-less files.
- **Prune narration.** Remove restate-the-code comments, numbered "step" dividers used purely
  as scaffolding, and conversational "we/our/Let's" phrasing. Replace section dividers with
  short intent lines only where they aid navigation of a long function.
- **Preserve & sharpen the valuable comments.** Explicitly keep and, where terse, expand:
  minimum-image/periodicity reasoning, MPI ownership/ghost/authority invariants (the
  redistribution heap-fix rationale), sign/winding conventions, tolerance semantics
  (`MFTL_TOLERANCE`, `MFTL_REL_TOLERANCE`, `MFTL_VERYSMALL`), curvature `2/R` convention, unit
  conventions, and non-obvious algorithmic choices (LBVH, prism-volume sign, cotangent
  Laplace–Beltrami, TSUR clamp-on-periodic-axes).
- **Correct stale/misleading comments** against the post-Stage-15 code (e.g. any comment still
  describing pre-Stage-13 non-periodic-only behaviour, or referencing removed WIP).
- **Resolve/close TODOs.** Remove TODOs whose issue is already fixed; for the remaining real
  ones (linear `target_obj` search; buffer-size heuristic) either fix trivially if
  behaviour-preserving and low-risk, or convert to a single tracked "Known limitations" note
  (do **not** leave silent). Reconcile any "RESOLVED" markers in `plans/00` with the code.
- **Terminology pass.** One glossary of canonical terms (object, ensemble, vertex, triangle,
  half-edge, twin, ghost, owner/`owner_proc`, `global_id`, local vs global index, slab, seam,
  minimum image, domain) and apply it consistently in comments (e.g. stop mixing
  "processor"/"rank"/"proc", "gid"/"global id").
- **Do not add comments for completeness** — self-evident code stays uncommented.

### 16B — Naming & code style (mostly cosmetic; internal-only)
- **Safe to rename (internal linkage, no ABI/API/wire impact):** file-`static` helper functions
  and local variables. Standardize internal `static` helpers on the migration convention
  `mftlLowerCamel(...)`; rename genuinely ambiguous/short/AI-looking locals to descriptive
  `lowerCamelCase` (or keep the file's dominant `snake_case` if that file is uniformly snake —
  consistency-within-file beats a global flag day). Target only names that are actually unclear;
  do **not** sweep every `i`/`n`/`t`.
- **Must NOT rename (compatibility-sensitive — freeze):**
  - All public functions (`MFTL…`, `MFTL_…`), public types (`MFTL_Object`, `MFTL_Vertex`,
    `MFTL_Triangle`, `MFTL_HalfEdge`, `MFTL_Domain`, `MFTL_Ensemble`, `MFTL_Overlap`,
    `MFTL_RemeshParameters`, enums, …), macros and constants (`MFTL_YES`/`MFTL_NO`/`MFTL_NULL`,
    `MFTL_ERROR`/`MFTL_SUCCESS`, `MFTL_TOLERANCE`, `MFTL_VERYSMALL`, `MFTL_REL_TOLERANCE`,
    `MFTL_TRI_PACK_SIZE`, `MFTL_FLAG_*`, …).
  - **Struct field names** — reached directly by the Stage-14/15 multiflow adapter/shim and/or
    baked into MPI packed layouts: `pos`, `global_id`, `owner_proc`, `is_ghost`, `flags`,
    `data`, `data_size`, `center`, `normal`, `area`, curvature fields, and the half-edge index
    fields `he_idx`/`twin_idx`/`next_idx`/`vertex_idx`. Renaming these is an API/ABI change → out
    of scope.
  - `Packed*` struct fields and member order (byte-transport sensitive — Stage 12).
- **Style unification (cosmetic):** brace/indent consistent with the existing `.clang-format` if
  one exists, else adopt one matching the dominant style and run it **once**, reviewing the diff
  for accidental semantic changes. Keep the formatting commit separate from content commits.

### 16C — Directory & file organization (safe structural refactor)
Recommend only moves/splits that materially improve maintainability; preserve build, API, and
Git history (`git mv` for moves; split by lifting contiguous function groups so `git blame`
follows). Because the library uses `GLOB_RECURSE src/*.c`, new files under `src/` need only a
`cmake` re-configure — no per-file CMake edits (call this out so reviewers re-run cmake).
- **Split the oversized catch-all `src/general/`** into subsystem directories, e.g.
  `src/geometry/` (`geometry_object.c`, `curvature.c`, `point_query.c`, `transform.c`,
  `convex_hull.c`), `src/parallel/` (`communication.c`, `ensemble.c`), `src/intersection/`
  (`mftl_bvh.c`, `overlap.c`), `src/io/` (`io.c`), `src/generators/` (`generators.c`,
  `sphere.c`), `src/core/` (`object.c`, `topology.c`, `radixsort.c`). (Directory names to be
  finalized in review — the point is subsystem separation, not the exact taxonomy.)
- **Split the largest files along existing internal seams (no logic change):**
  - `communication.c` (3842) → MPI datatype construction; buffer/exchange helpers; the two
    redistribution paths; ghost-layer extension + `MFTLUpdateGhostPositions`; spatial ghosting;
    `MFTLScatterTriangleData` + data-size consistency. **First de-duplicate** the shared
    receive-side reconstruction (see below), then split — splitting duplicated code twice is
    worse.
  - `remeshing_core.c` (1880) → operators (split/collapse/flip) vs smoothing vs quality/feature
    detection.
  - `mftl_bvh.c` (1359) → LBVH build vs traversal/queries vs periodic interval overlap +
    spatial ghosts; `overlap.c` (1215) → frame/projection vs tessellation vs driver.
- **De-duplicate into shared modules (behaviour-preserving; verify bit-identical):** unify the
  two collective-anchor helpers and the two `unwrapToAnchor` spellings into one home
  (`mftl_geometry.h`/a periodicity unit); factor the duplicated redistribution reconstruction
  into one helper used by both `MFTLRedistribute*`; route ad-hoc vector/geometry math through
  `mftl_vectoroperations.h`. Each merge must be proven identical (same inputs → same bytes) by
  the existing suite.
- **Public vs private headers.** Introduce `include/` = public API only, `src/**/internal/` (or
  `*_internal.h`) for cross-file-but-non-public declarations, so the eventual multiflow
  include surface (Stage 14/15) is unambiguous. Move internal-only declarations out of the
  public headers. This is API-*shape* clarifying, not API-*changing* (no public symbol removed).
- **Naming of files/dirs.** Resolve the `mftl_bvh.c` vs `communication.c` prefix inconsistency
  (either all `mftl_*` or none — recommend dropping the `mftl_` file prefix since the directory
  already namespaces, or standardize on it; pick one). Cosmetic; do last.
- **Avoid** restructuring for aesthetics alone; every move needs a maintainability rationale.

### 16D — MFTL2 user/developer manual (new documentation)
Write a new manual describing MFTL2 **as it exists after Stage 15**, conceptually modelled on
`MFTL.tex` (prose + equations + tables/diagrams + short code listings; not auto-generated API
reference). **Recommended form and location: a LaTeX manual** at `doc/MFTL2.tex` (mirroring the
mftl-doc layout: `doc/figs/`, `doc/*.bib` if needed, a `doc/README` on how to build the PDF),
so it is directly comparable to the MFTL1 manual and version-controlled alongside the code. A
short top-level `README.md` stays as the quick-start and points to the manual. (If the project
prefers, the same content can target Doxygen+prose or Markdown/Sphinx — decide in review; LaTeX
is the default recommendation for parity with `MFTL.tex`.) Do **not** copy MFTL1 struct
listings or the OBB/uthash model.

Required chapters (at least):
- **Purpose & scope** of MFTL2; what problems it solves (fast, periodic, parallel triangulated
  surfaces for the multiflow front-tracking / IBM solver).
- **Differences from MFTL1** (table): half-edge over flattened arrays + `khash` vs three uthash
  tables; `global_id` + array index vs sequential `id`; `owner_proc`/`is_ghost` vs `processor`;
  MPI-Cartesian decomposition + BVH/LBVH; **periodicity** (new); optional LAPACK.
- **Data structures & the half-edge representation** — `MFTL_Object` (`vertices[]`,
  `triangles[]`, `half_edges[]`, `khash` id→index maps, ghosts stored after locals),
  `MFTL_HalfEdge` (`he_idx`/`twin_idx`/`next_idx`/`vertex_idx`, twin pairing, boundary flag),
  with a half-edge diagram and a one-ring-traversal figure.
- **Objects, ensembles, vertices, triangles, global IDs** — lifecycle and identity; local vs
  global indices; `MFTL_Ensemble` as the multi-object/communicator container.
- **Ownership & ghost entities** — `owner_proc`, the owner guard for integrals, ghost layers,
  the "stale non-neighbour ghost" contract, authority/exactly-once ownership in redistribution.
- **MPI Cartesian decomposition** — `MPI_Cart` grid, x-only 1-D load-balancing assumption
  (`cuts={nproc,1,1}`), neighbour lists, global vs neighbour communication paths.
- **Periodicity** — `is_periodic[]`/`domain_size`/`domain_half_size`, `MFTLGetMinimumImageVector`
  / `MFTLApplyPeriodicity` / unwrap-to-anchor, the **minimum-image constraint** (feature < L/2)
  and `MFTLCheckMinimumImageConstraint`, and how each subsystem stays seam-safe.
- **Object generation & mesh I/O** — sphere/cube/cylinder/box generators; GTS reader/writer,
  VTK/PVTU output; winding canonicalization on read/generate.
- **Geometry queries & BVH/intersection** — AABB, point-in-object ray casting, LBVH build/
  traversal, tri–tri intersection, periodic-aware broad phase, overlap-volume quantification.
- **Curvature & geometric quantities** — cotangent Laplace–Beltrami, mean `2/R`/Gaussian `2/R²`
  conventions, roughness; volume/centroid/inertia definitions (with equations).
- **Serial & parallel remeshing** — split/collapse/flip/smooth, criteria variants, vertex
  fitting (barycentric/paraboloid), feature protection; the parallel shift scheme + boundary
  cleanup.
- **Smoothing & volume conservation** — TSUR3D, Kuprat, `VolumeRecovery`; conservation
  guarantees and tolerances.
- **Load balancing** — the x-only slab scheme and its limits.
- **Opaque per-triangle data** — `data`/`data_size`, the flat-POD contract, byte-exact transport.
- **Interaction with multiflow** — the Stage-14 adapter + Stage-15 shim, the dual decomposition
  and `sendnonlocal*` bridge, the edge-facade over half-edges (link to plans 14/15).
- **Invariants & conventions** — a single consolidated table mirroring `plans/00` §"Cross-cutting
  invariants" (winding/`V>0`, curvature convention, tolerance, rank-independence, opaque data).
- **Public API examples** — short listings for the main workflows (generate/redistribute/
  integrate; remesh cycle; intersection/overlap; attach & transport `CellData`).
- **Building & linking** (CMake options incl. optional LAPACK; `MPI_HOME`), **running tests**
  (`build_all.py`, `ctest`/`run_local_tests.py`, MPICH no `--oversubscribe`, `lsan.supp`,
  np 1/2/4/8), **typical usage workflow**, and **limitations & known performance compromises**
  (x-only LB; boundary-cleanup gather-to-rank-0; per-call point-query unwrap + BVH build;
  per-triangle heap `data` blobs; Allgatherv in overlap — cross-reference `plans/00`).
- Correct the current `README.md` inaccuracies while here (LAPACK optional not required; AABB/
  BVH not "oriented bounding box tree").

### 16E — Final consistency & regression review
- Whole-tree terminology/style consistency check; confirm no comment describes MFTL1 internals
  as MFTL2 behaviour; confirm the manual matches the shipped code (spot-check every documented
  signature/convention against the source).
- Full behaviour-preservation gate (see Acceptance).

## Change taxonomy (required — classify every proposed change into exactly one)
1. **Purely cosmetic / documentation** — comments, file headers, the manual, `README.md`
   corrections, formatting. No token of compiled code semantics changes. (16A, 16D, 16B-format.)
2. **Safe structural refactoring** — internal-`static`/local renames, file splits/moves,
   de-duplication into shared helpers, public/private header separation. Compiles, links, and
   passes the full suite bit-identically; no public symbol or wire format changes. (16B-names,
   16C.)
3. **API-affecting** — anything touching a public signature, public type/macro/constant,
   struct field name/order, or MPI packed layout. **Out of scope for Stage 16**; requires
   explicit approval and its own stage. List any that get *tempting* here and defer them.
4. **Defer rather than perform** — behaviour/perf changes surfaced by the cleanup but not
   behaviour-preserving: the linear-search→hash-map optimizations, the boundary-cleanup
   distributed variant, the point-query BVH-cache, the contiguous `data` pool, the paraboloid
   fit-tolerance question (Stage-13 review carry-forward). Document them as "Known limitations /
   future work" (feed the manual's limitations chapter and `plans/00`); do not implement.

## Files/functions likely to change
Nearly all `.c`/`.h` (comments/headers — 16A); a bounded set of `.c` files for internal renames
(16B); moves/splits under `src/**` + `include/**` and a `cmake` re-configure (16C, no per-file
CMake edits for the globbed library, but `tests/CMakeLists.txt` if any test file moves); new
`doc/` tree (16D). **No production logic edits** beyond mechanical de-duplication proven
identical.

## Numerical/physical considerations
- Zero intended numerical change. De-duplication merges (16C) are the only places where floating-
  point *could* drift (reassociation, a helper computing in a different order); each merge must be
  proven **bit-identical** on the regression suite before it is accepted, else it is reverted or
  reclassified as "defer".
- Winding/`V>0`, curvature `2/R`, tolerance semantics, and the opaque-data contract are frozen;
  the manual documents them but the code is untouched.

## MPI/PETSc considerations
- Packed struct layouts, MPI datatypes, ghost/redistribution wire formats, and communicator
  handling are frozen (any change is category 3 → out of scope). File splits must keep the MPI
  datatype registration and the two redistribution paths semantically identical.
- Re-run the MPI suite on 1/2/4/8 after every substage; a split that reorders collective calls
  or changes buffer sizing is a behaviour change, not a refactor.

## Possible failure modes
- A "harmless" comment edit that deletes a load-bearing rationale (e.g. why a ghost stays stale,
  why the prism sign is chosen) → treat the preserved-comments list as a checklist; review each
  deletion.
- A de-duplication that is subtly *not* identical (different summation order) → bit-identical
  gate catches volume/curvature drift; if it can't be made identical, defer.
- A file split that drops a `static` (now needed cross-file) or changes include order and breaks
  the build/ODR → compile on Release + ASan/UBSan after each split.
- GLOB not re-detecting a new file → document "re-run cmake"; CI does a clean configure.
- Manual drifting from code → 16E cross-checks documented signatures/conventions against source;
  a doc-vs-code mismatch is an acceptance failure.
- Renaming a name that looked internal but is actually reached by multiflow → the frozen list in
  16B; grep the multiflow adapter/shim (Stage 14/15) before any struct-field or public rename.

## Unit/regression tests to add / run
- No new behavioural tests are required (behaviour is frozen); the **existing** suite is the
  oracle. Optionally add a tiny `doc/`-build smoke check (LaTeX compiles) and a CI lint that the
  standard file header is present and the editor boilerplate is gone.
- Every substage must pass, unchanged, on 1/2/4/8 ranks in both Release and ASan/UBSan/LSan.

## Acceptance criteria
- **No intentional numerical behaviour change** — the full regression suite produces identical
  pass/fail and identical reported quantities (bit-identical where the suite pins values) before
  and after each substage.
- **Public API compatibility preserved** — no public function/type/macro/constant/struct-field/
  wire-format change (category 3) without explicit prior approval; a symbol diff of the public
  headers is empty (modulo approved items).
- **Full Release suite passes** (default LAPACK-off config; and, if LAPACK is enabled, the
  pre-existing paraboloid-tolerance item from the Stage-13 review is acknowledged, not
  introduced here).
- **Full ASan/UBSan/LSan suite passes.**
- **MPI tests pass on 1/2/4/8 ranks.**
- **Documentation agrees with the final implementation** — every signature, convention, and
  invariant in the manual is verified against the Stage-15 source (16E).
- **No stale MFTL1 references presented as MFTL2 behaviour** — in comments or the manual.
- Each substage independently leaves the project compiling and testable; commits are grouped by
  category (formatting separate from content) for reviewability.

## Substage summary
| Substage | Content | Change category |
|---|---|---|
| 16A | Comments, file headers, TODO resolution, terminology | 1 (cosmetic/doc) |
| 16B | Internal renames + one-shot formatting | 2 (+ 1 for format) |
| 16C | Dir/file reorg, large-file splits, de-dup, public/private headers | 2 (safe structural) |
| 16D | MFTL2 LaTeX manual under `doc/` + README fixes | 1 (doc) |
| 16E | Consistency + full behaviour-preservation regression gate | verification |
