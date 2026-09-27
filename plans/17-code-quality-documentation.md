# 17 — Repository-wide code-quality cleanup + MFTL2 manual

Status: ACCEPTED / COMPLETE (2026-09-27) — 17A–17E implemented and individually gated; all
acceptance criteria met (see "Implementation record" and "Acceptance" at the end). The substages
were not committed (explicit instruction); their separately preserved, individually gated patch
snapshots are the per-substage evidence.
Depends on: 15 (accepted) and 16 (generalized surfaces — must be COMPLETE and ACCEPTED before Stage 17 begins)
Implemented by: Claude Code (2026-09-26)
Reviewed by: independent cold review (Claude Code subagent, 2026-09-26; findings resolved, see record)
Acceptance: PASSED (2026-09-27)

## Objective
A repository-wide cleanup of MFTL2 — comments, naming, file organization, and a new user/
developer manual — **without intentionally changing numerical behaviour or public
functionality**. This stage runs only after the MFTL1→MFTL2 migration (Stage 15) and the
generalized-surface model (Stage 16) are complete and accepted, so the code, public API, and
multiflow coupling are stable and the manual can describe MFTL2 *as it actually exists after
Stage 16*. Nothing here should alter results: it is a readability, consistency, and
documentation pass, plus carefully-scoped structural refactors that preserve the build, the
API, and Git history.

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
- **Header consumers (for 17C4 visibility checks).** The installed public headers are consumed
  not only by `tests/` and `examples/` in this repo but also by **multiflow** (the Stage-14/15
  integration layer at `multiflow/src/include/mftl_adapter*.h`, linked via
  `find_package(MFTL2)` / `MFTL2::MFTL2`) and by external consumers of the installed package
  (see `external_link_test/`, which builds against an installed MFTL2 from the build tree and
  an install prefix). Any of these can pin a declaration that looks internal from inside this
  repo.
- **Reference documentation.** MFTL1 manual `/home/berend/git/Code/mftl-doc/MFTL.tex` (2558
  lines, LaTeX `report`; chapters: Introduction, Data structure, Distribution, Generation,
  Redistribution, Properties, Basic functions, Hierarchical/bounding-box, Point functions,
  Remesh, Intersection, Output, Init, Cleanup, Example listing, Future work; with `figs/`,
  equations, `lstlisting` code blocks, and a complete worked example program in
  `lst.fulltest`). Its **pedagogical form is the model for the MFTL2 manual**: human-readable
  prose that explains each concept before the API, every geometric/numerical formula as a
  worked equation, figures/diagrams for the data structures and algorithms, one chapter per
  subsystem, and practical, realistic code listings. Its *content* — the uthash
  `Edge`/`Vertex`/`Triangle` model, `id`/`processor` fields, `MFTL_Point`, the OBB tree, the
  `remeshSequence` hash-table driver, the `MFTLRedistributeObjectGeneral` API — is **obsolete**
  for MFTL2 and must not be reproduced.

## Proposed changes — by substage
Each substage leaves the tree compiling and the full suite green, and is independently
reviewable. Recommended order **17A → 17B → 17C1 → 17C2 → 17C3 → 17C4 → 17D → 17E**; 17D
(manual) may proceed in parallel with 17A–17C since it touches no source. Any repository-wide
formatting run is a **final isolated step** after 17C4 has passed its gate (see 17B); it is
never mixed into a substantive-cleanup commit.

### 17A — Comments & source documentation (purely cosmetic/doc)
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
- **Correct stale/misleading comments** against the post-Stage-16 code (e.g. any comment still
  describing pre-Stage-13 non-periodic-only behaviour, pre-Stage-16 twin/rim semantics, or
  referencing removed WIP).
- **Resolve/close TODOs.** Remove TODOs whose issue is already fixed; for the remaining real
  ones (linear `target_obj` search; buffer-size heuristic) either fix trivially if
  behaviour-preserving and low-risk, or convert to a single tracked "Known limitations" note
  (do **not** leave silent). Reconcile any "RESOLVED" markers in `plans/00` with the code.
- **Terminology pass.** One glossary of canonical terms (object, ensemble, vertex, triangle,
  half-edge, twin, ghost, owner/`owner_proc`, `global_id`, local vs global index, slab, seam,
  rim, physical boundary, boundary loop, tag, image counter, cut edge, wrapping axis, phase
  set, minimum image, domain) and apply it consistently in comments (e.g. stop mixing
  "processor"/"rank"/"proc", "gid"/"global id"). The glossary must match the Stage-16 model
  (LOCAL/REMOTE/BOUNDARY twin states; rim ≠ boundary ≠ seam).
- **Do not add comments for completeness** — self-evident code stays uncommented.

### 17B — Naming & code style (mostly cosmetic; internal-only)
- **Naming convention (decided).** Internal/file-`static` helper functions and descriptive
  local variables are standardized on **lowerCamelCase** (statics keep the migration prefix
  form `mftlLowerCamel(...)`). **Conventional short mathematical/index names** (`i`, `j`, `k`,
  `n`, `t`, `e`, `v`, `p`, `q`, `d`, `dx`, `n1`/`n2`, …) are **kept** where they are clear —
  do not sweep them. Target only names that are genuinely unclear or misleading (e.g.
  `buffer_init` → `mftlBufferInit`, `new_triangle` → `mftlNewTriangle`); renaming a correct,
  unambiguous name is churn, not cleanup. Consistency within a file is required after the pass
  (no half-renamed files).
- **Must NOT rename (compatibility-sensitive — freeze):**
  - All public functions (`MFTL…`, `MFTL_…`), public types (`MFTL_Object`, `MFTL_Vertex`,
    `MFTL_Triangle`, `MFTL_HalfEdge`, `MFTL_Domain`, `MFTL_Ensemble`, `MFTL_Overlap`,
    `MFTL_RemeshParameters`, `MFTL_PhaseSet`, enums, …), macros and constants (`MFTL_YES`/
    `MFTL_NO`/`MFTL_NULL`, `MFTL_ERROR`/`MFTL_SUCCESS`, `MFTL_UNDEFINED`, `MFTL_TOLERANCE`,
    `MFTL_VERYSMALL`, `MFTL_REL_TOLERANCE`, `MFTL_TRI_PACK_SIZE`, `MFTL_FLAG_*`,
    `MFTL_TWIN_NONE`, …).
  - **Struct field names** — reached directly by the Stage-14/15 multiflow adapter/shim and/or
    baked into MPI packed layouts: `pos`, `img`, `global_id`, `owner_proc`, `is_ghost`,
    `flags`, `data`, `data_size`, `center`, `normal`, `area`, curvature fields, and the
    half-edge index fields `he_idx`/`twin_idx`/`next_idx`/`vertex_idx`/`twin_gid`. Renaming
    these is an API/ABI change → out of scope.
  - `Packed*` struct fields and member order (byte-transport sensitive — Stage 12).
- **Formatting policy (decided).** No repository-wide formatting diff mixed with substantive
  cleanup. Two permitted modes only:
  1. **Touched files:** format a file consistently with `.clang-format` in the substage that
     already edits it for other reasons (formatting hunks may share that file's commit);
  2. **Broader formatting:** if a repo-wide formatting run is wanted, it is a **final, isolated
     step** performed only after all semantic-neutral refactoring (17A–17C4) has passed its
     gates — its own commit containing nothing else, reviewed for accidental semantic changes,
     followed by a full regression re-run.

### 17C — Structural cleanup (safe structural refactor, strictly ordered)
Recommend only moves/splits that materially improve maintainability; preserve build, API, and
Git history (`git mv` for moves; split by lifting contiguous function groups so `git blame`
follows). Because the library uses `GLOB_RECURSE src/*.c`, new files under `src/` need only a
`cmake` re-configure — no per-file CMake edits (call this out so reviewers re-run cmake).

17C is split internally into four sub-steps, **each of which must compile and pass the full
regression gate (Release + ASan/UBSan/LSan, np 1/2/4/8, bit-identical) before the next
begins.** They are not batched, and no two are ever combined into one commit:

- **17C1 — Directory/file moves.** Split the oversized catch-all `src/general/` into subsystem
  directories, e.g. `src/geometry/` (`geometry_object.c`, `curvature.c`, `point_query.c`,
  `transform.c`, `convex_hull.c`), `src/parallel/` (`communication.c`, `ensemble.c`),
  `src/intersection/` (`mftl_bvh.c`, `overlap.c`), `src/io/` (`io.c`), `src/generators/`
  (`generators.c`, `sphere.c`), `src/core/` (`object.c`, `topology.c`, `radixsort.c`).
  (Directory names to be finalized in review — the point is subsystem separation, not the
  exact taxonomy.) Also resolve the `mftl_bvh.c` vs `communication.c` file-prefix
  inconsistency here (either all `mftl_*` or none — recommend dropping the `mftl_` file prefix
  since the directory already namespaces; pick one). Pure `git mv`, **no content edits** in
  this step beyond include-path fixes forced by the moves.
- **17C2 — Large-file splits.** Split along existing internal seams (no logic change, no
  de-duplication — code moves verbatim):
  - `communication.c` (3842) → MPI datatype construction; buffer/exchange helpers; the two
    redistribution paths; ghost-layer extension + `MFTLUpdateGhostPositions`; spatial ghosting;
    `MFTLScatterTriangleData` + data-size consistency.
  - `remeshing_core.c` (1880) → operators (split/collapse/flip) vs smoothing vs quality/feature
    detection.
  - `mftl_bvh.c` (1359) → LBVH build vs traversal/queries vs periodic interval overlap +
    spatial ghosts; `overlap.c` (1215) → frame/projection vs tessellation vs driver.
  - **Explicitly:** the redistribution-reconstruction de-duplication is **not** part of this
    step — it belongs to 17C3 with a full gate in between (see below).
- **17C3 — De-duplication / shared helpers.** Behaviour-preserving merges, each proven
  **bit-identical** by the existing suite (same inputs → same bytes) before acceptance, else
  reverted or reclassified "defer": unify the two collective-anchor helpers and the two
  `unwrapToAnchor` spellings into one home (`mftl_geometry.h`/a periodicity unit); factor the
  duplicated ~500-line receive-side reconstruction into one helper used by both
  `MFTLRedistribute*`; route ad-hoc vector/geometry math through `mftl_vectoroperations.h`.
  **Do not combine the redistribution-code de-duplication with splitting or moving
  `communication.c`** — the move (17C1), the split (17C2) and this merge (17C3) are three
  separate gated steps.
- **17C4 — Public/private header cleanup.** Introduce `include/` = public API only,
  `src/**/internal/` (or `*_internal.h`) for cross-file-but-non-public declarations, so the
  multiflow include surface (Stage 14/15) is unambiguous. **Header-visibility rule (hard
  gate):** before moving any declaration or header out of the public include tree, verify that
  it is **not consumed** by (a) multiflow (grep the adapter/shim and FT sources), (b)
  `examples/`, (c) `tests/`, or (d) an external `find_package(MFTL2)` consumer
  (`external_link_test/` against the build tree and the install prefix). A declaration that is
  actually consumed by any of these is **public in practice**: removing or un-installing it is
  an API change (category 3) and must be **deferred**, not performed in Stage 17. What remains
  after that filter is genuinely internal and may be moved — this step is then API-*shape*
  clarifying, not API-*changing* (no consumed public symbol is removed).

Avoid restructuring for aesthetics alone; every move needs a maintainability rationale.

### 17D — MFTL2 user/developer manual (new documentation)
- **Location (decided):** the manual is `/home/berend/git/Code/MFTL2/doc/MFTL2.tex`, a LaTeX
  document version-controlled alongside the code (mirroring the mftl-doc layout: `doc/figs/`,
  `doc/multiphase.bib` or a new `.bib` if citations are needed, `doc/README.md` with build
  instructions). A short top-level `README.md` stays as the quick-start and points to the
  manual.
- **Pedagogical model.** Read the complete MFTL1 manual
  `/home/berend/git/Code/mftl-doc/MFTL.tex` (2558 lines) and emulate its *form*: human-readable
  prose that explains each concept before introducing the API, every geometric/numerical
  formula as a worked equation, figures/diagrams for the data structures and algorithms (e.g.
  the half-edge diagram, one-ring traversal, distribution/ghost figure, LBVH traversal,
  remesh-operator figures — the counterparts of MFTL.tex's `objectstructure`, `distribution`,
  `tessalate`, `raytracing`, `ref_algorithm`, `coarsen_algo` figures), one chapter per
  subsystem, and practical code listings including at least one complete worked program (the
  counterpart of `lst.fulltest`). It is a manual, not an auto-generated API reference.
- **Fidelity rule.** The manual describes MFTL2 **as it exists after accepted Stage 16** — not
  after Stage 15, and never MFTL1. Do not copy MFTL1 APIs or data structures (uthash lists,
  `id`/`processor` fields, `MFTL_Point`, OBB tree, the `remeshSequence` hash-table driver,
  `MFTLRedistributeObjectGeneral`, …). Every struct, signature, constant, convention and
  behaviour written into the manual must be verified against the actual post-Stage-16 MFTL2
  source; 17E re-checks every documented signature/convention against the shipped code, and a
  doc-vs-code mismatch is an acceptance failure.
- **Build (decided):** add a simple, documented way to produce `MFTL2.pdf` — a `doc/Makefile`
  (`pdflatex` ×2 + `bibtex` as needed) and/or an optional CMake target (e.g.
  `-DMFTL_BUILD_MANUAL=ON` / `make manual`, off by default so LaTeX is not a build
  dependency), documented in `doc/README.md` and the top-level `README.md`. A LaTeX-compile
  smoke check (the PDF builds from a clean checkout) joins the Stage-17 gates.
- **Examples (decided):** include practical, **compilable** examples for the major MFTL2
  workflows (at minimum): create an ensemble + generate/read a body + redistribute + compute
  integrals; transform/move a body and refresh ghosts; a serial and a parallel remesh cycle;
  curvature/normal queries; point queries and BVH intersection/overlap; attach & transport
  per-triangle `data`; set up a periodic domain (minimum-image rules, counter-aware position
  updates); declare an open body / enable periodic fronts and query a phase-boundary set.
  Listings must compile against the shipped library — prefer `\lstinputlisting` of files that
  live in (or are CI-compiled like) `examples/`, or snippets mechanically verified against
  them. A stale, non-compiling listing is an acceptance failure.

Required chapters (at least):
- **Purpose & scope** of MFTL2; what problems it solves (fast, periodic, parallel triangulated
  surfaces — closed, open and wrapping — for the multiflow front-tracking / IBM solver).
- **Differences from MFTL1** (table): half-edge over flattened arrays + `khash` vs three uthash
  tables; `global_id` + array index vs sequential `id`; `owner_proc`/`is_ghost` vs `processor`;
  MPI-Cartesian decomposition + BVH/LBVH vs OBB tree; **periodicity** (new); **open and
  wrapping surfaces** (new); optional LAPACK.
- **Data structures & the half-edge representation** — `MFTL_Object` (`vertices[]`,
  `triangles[]`, `half_edges[]`, `khash` id→index maps, ghosts stored after locals),
  `MFTL_HalfEdge` (`he_idx`/`twin_idx`/`next_idx`/`vertex_idx`/`twin_gid`, twin pairing), with
  a half-edge diagram and a one-ring-traversal figure.
- **The generalized surface model (Stage 16)** — open surfaces and declared body classes
  (`declared_open`); the three half-edge twin states **LOCAL / REMOTE / BOUNDARY** with the
  `twin_gid` encoding (`−2 − tag`); the **physical boundary vs MPI rim vs periodic seam**
  distinction (the invariant table of `plans/16` §8 — coordinates are never consulted);
  manifold/orientable builder validation; quiescent points and `MFTLValidateTwinStates`;
  remote-twin discovery and the ghost-completion guarantees (G1/G2); **boundary tags and
  distributed boundary loops** (loop construction, winding vectors, per-loop handles).
- **Periodicity and image counters** — `is_periodic[]`/`domain_size`, the wrapped-storage
  convention, per-vertex per-axis **image counters `img[3]`** and continuous coordinates
  `X = pos + img ⊙ L`; the counter lifecycle (construction/lift via `MFTLLiftImages`,
  counter-aware `MFTLVertexSetPos`/`MFTLVertexDisplace` and wrap-after-motion, split/collapse
  inheritance, MPI/I-O transport); **cut edges** and per-axis wrap classification
  (`WRAP_k`/`COMPACT_k`); **wrapping surfaces** (ribbons, films, multi-axis wrapping); the
  per-edge half-period minimum-image rule and `MFTLCheckMinimumImageConstraint` (no
  object-extent restriction after Stage 16).
- **Phase-boundary sets and defined/undefined global quantities** — `MFTL_PhaseSet` (bodies,
  orientation signs, wall closures); the finite-volume phase-boundary conditions (closed or
  wall-closed members, an admissible compact axis, net flux `Φ(S) = 0`, consistent
  orientation); the admissible-axis volume/centre formulas (with equations); **`MFTL_UNDEFINED`
  semantics** for volume, volume centre, indicator and inside/outside when the conditions fail
  (with the worked examples: single wrapping sheet, slab bounded by two films, wall-closed
  film, triply-wrapping surface); which surface-only quantities are always defined.
- **Objects, ensembles, vertices, triangles, global IDs** — lifecycle and identity; local vs
  global indices; `MFTL_Ensemble` as the multi-object/communicator container.
- **Ownership & ghost entities** — `owner_proc`, the owner guard for integrals, ghost layers,
  the "stale non-neighbour ghost" contract, authority/exactly-once ownership in redistribution.
- **MPI Cartesian decomposition** — `MPI_Cart` grid, x-only 1-D load-balancing assumption
  (`cuts={nproc,1,1}`), neighbour lists, global vs neighbour communication paths.
- **Object generation & mesh I/O** — sphere/cube/cylinder/box generators; GTS reader/writer,
  VTK/PVTU output; native I/O v2 (classes, tags, image counters; restart on a different rank
  count); winding canonicalization on read/generate.
- **Geometry queries & BVH/intersection** — AABB, point-in-object ray casting (and its
  phase-set preconditions), LBVH build/traversal, tri–tri intersection, periodic-aware broad
  phase, overlap-volume quantification.
- **Curvature & geometric quantities** — cotangent Laplace–Beltrami, mean `2/R`/Gaussian `2/R²`
  conventions, roughness; curvature validity flags at boundaries (`CURV_UNDEFINED`/
  `CURV_EXTRAPOLATED`); boundary edge geometry (tangent/conormal); volume/centroid/inertia
  definitions (with equations) and when they are defined.
- **Serial, parallel and generalized remeshing** — split/collapse/flip/smooth, criteria
  variants, vertex fitting (barycentric/paraboloid), feature protection; the parallel shift
  scheme + boundary cleanup; the **generalized rules for surfaces with boundary**: the
  augmented (virtual-vertex) link condition, boundary curves (REPRESENTED spline / CONSTRAINED
  projector), corners, boundary split/flip/smoothing rules, class-guarded drivers (volume
  recovery only when the phase set's volume is defined), class-transition reporting.
- **Smoothing & volume conservation** — TSUR3D, Kuprat, `VolumeRecovery`; conservation
  guarantees and tolerances; interior-only restrictions on open surfaces.
- **Load balancing** — the x-only slab scheme and its limits.
- **Opaque per-triangle data** — `data`/`data_size`, the flat-POD contract, byte-exact
  transport; the conserved surface scalar (L7) semantics.
- **Interaction with multiflow** — the Stage-14 adapter + Stage-15 shim, the dual decomposition
  and `sendnonlocal*` bridge, the edge-facade over half-edges; the Stage-16 capability matrix,
  `PeriodicFronts`/`OpenSurface` declarations, adapter class checks, and the
  **boundary-condition requirements for open bodies** (surface-scalar NEUMANN/DIRICHLET loop
  BCs; surface tension REJECT/PINNED with support reaction; CSF only pinned on a wall closure;
  IBM mass-property rules) — what multiflow must declare per body/tag before a module is
  allowed to run.
- **Invariants & conventions** — a single consolidated table mirroring `plans/00` §"Cross-cutting
  invariants" extended with the Stage-16 invariants (twin states, image counters, loop tags,
  defined/undefined rules; winding/`V>0`, curvature convention, tolerance, rank-independence,
  opaque data).
- **Public API examples** — the compilable listings listed above.
- **Building & linking** (CMake options incl. optional LAPACK; `MPI_HOME`; `find_package(MFTL2)`
  consumption), **building the manual** (`doc/README.md`), **running tests** (`build_all.py`,
  `ctest`/`run_local_tests.py`, MPICH no `--oversubscribe`, `lsan.supp`, np 1/2/4/8), **typical
  usage workflow**, and **limitations & known performance compromises** (x-only LB;
  boundary-cleanup gather-to-rank-0; per-call point-query unwrap + BVH build; per-triangle heap
  `data` blobs; Allgatherv in overlap; Stage-16 intentionally-unsupported list — cross-reference
  `plans/00` and `plans/16` §12).
- Correct the current `README.md` inaccuracies while here (LAPACK optional not required; AABB/
  BVH not "oriented bounding box tree").

### 17E — Final consistency & regression review
- Whole-tree terminology/style consistency check; confirm no comment describes MFTL1 internals
  as MFTL2 behaviour; confirm the manual matches the shipped code (spot-check every documented
  signature/convention against the post-Stage-16 source) and that every manual listing compiles.
- The optional repository-wide formatting run, if wanted, happens here as a single isolated
  commit (see 17B formatting policy), followed by the full gate.
- Full behaviour-preservation gate (see Acceptance).

## Change taxonomy (required — classify every proposed change into exactly one)
1. **Purely cosmetic / documentation** — comments, file headers, the manual, `README.md`
   corrections, formatting. No token of compiled code semantics changes. (17A, 17D, 17B
   formatting policy.)
2. **Safe structural refactoring** — internal-`static`/local renames, file splits/moves,
   de-duplication into shared helpers, public/private header separation of **verified
   unconsumed** declarations. Compiles, links, and passes the full suite bit-identically; no
   public symbol or wire format changes. (17B names, 17C1–17C4.)
3. **API-affecting** — anything touching a public signature, public type/macro/constant,
   struct field name/order, an MPI packed layout, **or the removal/un-installing of an
   installed declaration that multiflow, examples, tests, or an external `find_package(MFTL2)`
   consumer actually uses** (17C4 visibility rule). **Out of scope for Stage 17**; requires
   explicit approval and its own stage. List any that get *tempting* here and defer them.
4. **Defer rather than perform** — behaviour/perf changes surfaced by the cleanup but not
   behaviour-preserving: the linear-search→hash-map optimizations, the boundary-cleanup
   distributed variant, the point-query BVH-cache, the contiguous `data` pool, the paraboloid
   fit-tolerance question (Stage-13 review carry-forward). Document them as "Known limitations /
   future work" (feed the manual's limitations chapter and `plans/00`); do not implement.

## Files/functions likely to change
Nearly all `.c`/`.h` (comments/headers — 17A); a bounded set of `.c` files for internal renames
(17B); moves/splits under `src/**` + `include/**` and a `cmake` re-configure (17C1–17C4, no
per-file CMake edits for the globbed library, but `tests/CMakeLists.txt` if any test file
moves); new `doc/` tree (`doc/MFTL2.tex`, `doc/figs/`, `doc/Makefile` or CMake manual target,
`doc/README.md`, possibly a `.bib`) and example-listing sources under `examples/` (17D). **No
production logic edits** beyond mechanical de-duplication proven identical.

## Numerical/physical considerations
- Zero intended numerical change. De-duplication merges (17C3) are the only places where
  floating-point *could* drift (reassociation, a helper computing in a different order); each
  merge must be proven **bit-identical** on the regression suite before it is accepted, else it
  is reverted or reclassified as "defer".
- Winding/`V>0`, curvature `2/R`, tolerance semantics, the opaque-data contract, and the
  Stage-16 twin-state/image-counter semantics are frozen; the manual documents them but the
  code is untouched.

## MPI/PETSc considerations
- Packed struct layouts (including `PackedVertex.img`), MPI datatypes, ghost/redistribution
  wire formats, and communicator handling are frozen (any change is category 3 → out of
  scope). File splits must keep the MPI datatype registration and the two redistribution paths
  semantically identical.
- Re-run the MPI suite on 1/2/4/8 after every substage (and after every 17C sub-step); a split
  that reorders collective calls or changes buffer sizing is a behaviour change, not a
  refactor.

## Possible failure modes
- A "harmless" comment edit that deletes a load-bearing rationale (e.g. why a ghost stays stale,
  why the prism sign is chosen) → treat the preserved-comments list as a checklist; review each
  deletion.
- A de-duplication that is subtly *not* identical (different summation order) → bit-identical
  gate catches volume/curvature drift; if it can't be made identical, defer.
- A file split that drops a `static` (now needed cross-file) or changes include order and breaks
  the build/ODR → compile on Release + ASan/UBSan after each split.
- GLOB not re-detecting a new file → document "re-run cmake"; CI does a clean configure.
- Combining two 17C sub-steps (e.g. deduplicating the redistribution reconstruction in the same
  commit as the `communication.c` split) → a regression becomes unattributable; the gates are
  per-sub-step precisely so each diff is individually provable bit-identical. Reject mixed
  commits in review.
- Moving an "internal-looking" declaration out of the public include tree that multiflow, an
  example, a test, or an external `find_package(MFTL2)` consumer actually includes → downstream
  build break discovered late; the 17C4 visibility check (grep multiflow + build
  `external_link_test/` against the modified install) runs **before** the move.
- A repository-wide formatting diff slipped into a substantive cleanup commit → review noise and
  destroyed `git blame`; formatting follows the 17B policy (touched files, or one isolated final
  commit).
- Manual drifting from code → 17E cross-checks documented signatures/conventions against source
  and compiles every listing; a doc-vs-code mismatch or a non-compiling listing is an acceptance
  failure. The manual must describe the post-Stage-16 model — a chapter written against the
  Stage-15 API is an acceptance failure.
- Renaming a name that looked internal but is actually reached by multiflow → the frozen list in
  17B; grep the multiflow adapter/shim (Stage 14/15) before any struct-field or public rename.

## Unit/regression tests to add / run
- No new behavioural tests are required (behaviour is frozen); the **existing** suite is the
  oracle. Add a `doc/` build smoke check (`MFTL2.pdf` compiles from a clean checkout) and a
  compilability check for the manual's example listings; optionally a CI lint that the standard
  file header is present and the editor boilerplate is gone.
- Every substage — and every 17C sub-step individually — must pass, unchanged, on 1/2/4/8 ranks
  in both Release and ASan/UBSan/LSan.

## Acceptance criteria
- **No intentional numerical behaviour change** — the full regression suite produces identical
  pass/fail and identical reported quantities (bit-identical where the suite pins values) before
  and after each substage.
- **Public API compatibility preserved** — no public function/type/macro/constant/struct-field/
  wire-format change (category 3) without explicit prior approval; a symbol diff of the public
  headers is empty (modulo approved items), and no installed declaration consumed by multiflow,
  examples, tests or an external `find_package(MFTL2)` consumer was removed.
- **Full Release suite passes** (default LAPACK-off config; and, if LAPACK is enabled, the
  pre-existing paraboloid-tolerance item from the Stage-13 review is acknowledged, not
  introduced here).
- **Full ASan/UBSan/LSan suite passes.**
- **MPI tests pass on 1/2/4/8 ranks.**
- **Each 17C sub-step gated individually** — 17C1, 17C2, 17C3, 17C4 must each be independently
  reviewable and must compile and pass the complete gate before the next begins. Normally this
  is visible as separate commits; when the implementer is explicitly prohibited from committing,
  separately preserved and individually gated patch snapshots are equivalent evidence. In
  particular the redistribution de-duplication is never in the `communication.c` move/split
  step.
- **Documentation agrees with the final implementation** — every signature, convention, and
  invariant in the manual is verified against the post-Stage-16 source (17E); every code
  listing in the manual compiles against the shipped library; `MFTL2.pdf` builds via the
  documented command/target from a clean checkout.
- **No stale MFTL1 (or pre-Stage-16) references presented as current MFTL2 behaviour** — in
  comments or the manual.
- Each substage independently leaves the project compiling and testable; commits are grouped by
  category (formatting separate from content; any repo-wide formatting is one isolated final
  commit) for reviewability.

## Substage summary
| Substage | Content | Change category |
|---|---|---|
| 17A | Comments, file headers, TODO resolution, terminology | 1 (cosmetic/doc) |
| 17B | Internal renames (lowerCamelCase statics/locals; frozen public surface) | 2 |
| 17C1 | Directory/file moves (`git mv`, include-path fixes only) | 2 (safe structural) |
| 17C2 | Large-file splits (verbatim function-group lifts) | 2 (safe structural) |
| 17C3 | De-duplication / shared helpers (bit-identical gate) | 2 (safe structural) |
| 17C4 | Public/private header cleanup (visibility-verified) | 2 (safe structural) |
| 17D | MFTL2 LaTeX manual `doc/MFTL2.tex` (post-Stage-16 model), PDF build, compilable examples, README fixes | 1 (doc) |
| 17E | Consistency + full behaviour-preservation regression gate; optional isolated repo-wide formatting commit | verification (+ 1) |

## Implementation record (2026-09-26)

Baseline: MFTL2 `511fe23` (Stage 16 accepted), multiflow `4a61244` (branch `berend-mftl`, not
modified by Stage 17). Every substage was gated on a snapshot of the tree before the next one
began: the comment-stripped token streams (`gcc -fpreprocessed -E -P`) and the per-function
disassembly of a Release/no-LTO library build were compared with the previous state, and the full
ctest suite was run in Release (LAPACK off) and Debug+ASan/UBSan/LSan (plus Release+LAPACK and
Debug for 17C3 and the final tree). Every test's complete output was compared with the baseline
run (all printed quantities, order-insensitive because ranks interleave): identical in every
gate. Nothing was committed; the per-substage patch series and the snapshots are kept with the
implementation notes.

### 17A — comments and file headers (category 1)
- A descriptive header (purpose, subsystem, contract) on every `.c`/`.h` (library, tests,
  examples); `@Author/@Date/@Last Modified` boilerplate removed; narration and restating comments
  removed; rationale comments kept and sharpened.
- Stale comments corrected against the Stage-16 code, among them: `MFTLObjectCreate/Destroy`,
  integral anchors (no extent restriction), `MFTLCheckMinimumImageConstraint` (edge criterion
  only), transforms (translate does not wrap), curvature (Gaussian `2/R^2`, boundary flags, G2),
  redistribution contract (the CACHED triangle centre decides the owner; cached geometry travels
  with the triangle), `n_local_*` (only maintained by multi-rank redistribution, the builders,
  `MFTLObjectAppendTriangleCopy` and `MFTLObjectClear`), spatial ghosts (not periodic-aware),
  radix sort (unsigned order), `MFTLVolumeRecovery` (cap 10; no-op for open/wrapping bodies),
  shift scheme (temporary x-only layout), `MFTLSwapEdge` (declared, not implemented), BVH
  (coplanar pairs reported, absolute tolerances), one-ring traversal directions.
- TODOs resolved or converted (linear body search → documented deferred optimisation).
- Gate: token streams identical (0 code differences in 76 files), disassembly identical.

### 17B — internal renames (category 2)
- 36 internal helpers renamed to `mftlLowerCamel` (32 static; 4 non-static but undeclared in any
  header: `buffer_init`, `glue_wings`, `CountFeatureEdges`, `FlipApproved`). No consumer
  (multiflow, tests, examples, external) references any of them. Two diagnostic string literals
  keep the old names so the object code stays byte-identical.
- Kept by decision: the `ps*`/`mftl1*` file prefixes, the upstream-derived radix-sort helpers,
  public-in-practice `MFTL_*` exported functions, all public API; no local-variable sweep; no
  formatting.
- Gate: token streams identical modulo the rename map; disassembly identical after mapping names.

### 17C1 — directory moves (category 2)
- `src/general/` split into `src/core/`, `src/parallel/`, `src/periodic/`, `src/geometry/`,
  `src/intersection/`, `src/io/`, `src/generators/` (`git mv`); `mftl_bvh.c` → `bvh.c`. Only path
  comments, the `CMakeLists.txt` comment and the allowlist of `tools/check_twin_gid_readers.sh`
  changed. Gate: token streams and disassembly identical.

### 17C2 — file splits (category 2)
- `communication.c` → `comm_types.c` (MPI record types), `redistribute.c` (packing, exchange,
  both redistribution paths), `ghost_refresh.c`, `ghost_layers.c`, with the private header
  `src/parallel/comm_internal.h`; `remeshing_core.c` → `remeshing_core.c` (operators and gates),
  `remesh_smoothing.c`, `remesh_driver.c`, with `src/remeshing/remeshing_internal.h`.
- Every top-level definition is token-identical; the only linkage change is
  `mftlMpiExchangeBuffers` (static → internal extern). Disassembly differs only by cross-file
  inlining/register allocation. Gate outputs identical.
- Not split (deferred, category 4): `bvh.c`, `overlap.c`, `phase_set.c`, `twin_states.c`, `io.c` —
  cohesive single algorithms whose split would require de-staticising several helpers.

### 17C3 — de-duplication (category 2)
- The two collective anchor helpers are one (`mftlObjectUnwrapAnchor`, declared in
  `src/geometry/geometry_internal.h`): identical collectives in identical order, identical
  outputs, no arithmetic. Gate (all four configurations) identical.
- A pre-existing test race was fixed (test-only): the concurrently running `MeshIO-np*` tests
  shared output file names (reproduced on the baseline build).
- Deferred (category 4): the redistribution receive-side reconstruction (the two paths differ in
  payload offsets and completion placement; not provably bit-identical); merging
  `MFTLUnwrapToAnchor` with the integer-image unwrap of the integrals (different formulas);
  routing ad-hoc vector arithmetic through `mftl_vectoroperations.h` (operation order).

### 17C4 — header visibility (category 2)
- `mftl_twin_internal.h` (self-declared non-public, consumed by no consumer) moved to the private
  include directory `src/internal/`; no longer installed. `mftl_boundary_internal.h` stays public
  (the test suite uses it). Unconsumed declarations in the public headers were kept (removing
  them would change the installed API).
- Install to a prefix, `external_link_test` through `find_package(MFTL2)`, every installed header
  compiles on its own, multiflow builds against the install. Gate identical.

### 17D — manual
- `doc/MFTL2.tex` + `doc/chapters/` (TikZ figures inline), `doc/Makefile`, `doc/README.md`,
  CMake option `MFTL_BUILD_MANUAL` (target `manual`, off by default). 8 example programs
  `examples/manual_*.c` are included with `\lstinputlisting` and run by 32 new tests
  (`Manual-<program>-np<N>`, 1/2/4/8 ranks). README corrected (LAPACK optional; AABB/LBVH, not
  OBB). The PDF builds from a clean tree without warnings about undefined references.

### 17E — final review and regression
- Independent cold review (token/disassembly classification of every change, public API and
  exported-symbol audit, source and manual audit, examples on 1/2/4/8 ranks): behaviour
  preservation confirmed; one doc mismatch (one-ring figure/direction) and several comment/doc
  items fixed (comment/doc only, re-verified token- and disassembly-identical).
- MFTL2 on the final tree: Release (LAPACK off/on), Debug, ASan/UBSan/LSan — 129/129 each, the
  original 97 tests' outputs identical to the baseline; 1 000-step Stage-16 invariant fuzz on
  1/2/4/8 ranks: 10 002 checks each, 0 failed, logs identical to the baseline.
- multiflow (unchanged, rebuilt against an installed final MFTL2 with LAPACK; every comparison
  against the same runs with the installed baseline MFTL2):
  - adapter unit tests (`tools/mftladapter_tests`, 1–4 ranks): 1 541 checks, 0 failed, output
    identical to the baseline;
  - closed-surface legacy A/B (`FT_Stage16D/closed_ab`: FT, FT_Sharp_CSF, FT_Sharp_Integral,
    IBM-MFTL A/B, Laplace, marching cubes; 12 runs): all 58 HDF5 files bit-identical (h5diff);
    open IBM mesh rejected;
  - FT_Stage16D quick (D5 seam/corner advection, D6 pinch-off, P_* seam family, restarts on
    1/2/8 ranks, open patch, wrapping tube, rejections): 57 checks, 0 failed; all 39 per-step
    `front_log.txt` byte-identical to the baseline;
  - FT_Stage16E quick (open-surface scalar Neumann/Dirichlet, pinned surface tension with support
    reaction, wall closures, CSF, rejections): 89 checks, 0 failed; all 51 per-step
    `f16e_log.txt` byte-identical to the baseline.
- Formatting: no repository-wide formatting run (skipped by decision; it would be an isolated
  final commit).

### Defects found and deferred (behaviour changes; not fixed in Stage 17)
- D1: `MFTLPointQueryInsideMany`/`MFTLPointQueryInside` deadlock on a non-periodic domain when
  some rank stores no triangle of the body (that rank takes the per-point fallback). multiflow's
  adapter uses the caller-held query. Documented in the header and the manual.
- D2: the error path of `MFTLEnsembleCreate` destroys communicators that were never created and
  decrements the ensemble count that was never incremented.
- Carried over, still open: the `plans/15` §R6–R7 should-fix items, the remaining scale-dependent
  absolute thresholds (BVH inflation 1e-4, proximity 1e-6) and multiflow's zero-mass surface-scalar
  result in the FT configuration (all behaviour changes).

### Acceptance (2026-09-27)
- **ACCEPTED / COMPLETE.** Baseline: MFTL2 `511fe2354cac4a5e8eac2616dca1da15570815ff` (Stage 16
  accepted) and multiflow `4a612445dba8afe1222a4ddae21636c90b14ebf4` (branch `berend-mftl`). The
  final Stage-17 working tree passed the complete cold-review gate (17E) after its findings were
  resolved.
- Per-substage evidence: the gated snapshots and patch series base → 17A → 17B → 17C1 → 17C2 →
  17C3 → 17C4 → 17D → 17E, each passing its complete gate before the next began (the substages
  were not committed by explicit instruction; see the amended criterion above).
- Acceptance evidence (unchanged, recorded above): the manual `doc/MFTL2.tex` (PDF builds from a
  clean tree) and its 8 example programs (32 tests on 1/2/4/8 ranks); the installed-package
  consumer (`external_link_test` via `find_package(MFTL2)`, standalone installed headers,
  multiflow built against the install); the MFTL2 suite on 1/2/4/8 ranks in Release (LAPACK
  off/on), Debug and ASan/UBSan/LSan; the 1 000-step invariant fuzz; the multiflow adapter tests,
  closed-surface A/B (bit-identical HDF5) and FT_Stage16D/16E suites (byte-identical logs).
- Deferred follow-up correctness work (behaviour changes, not part of Stage 17):
  - D1: `MFTLPointQueryInsideMany` / `MFTLPointQueryInside` deadlock when a non-periodic
    distributed body has a rank with zero local triangles.
  - D2: `MFTLEnsembleCreate` error cleanup frees communicator and MPI-datatype state it does not
    own (communicators never created; ensemble count never incremented).
