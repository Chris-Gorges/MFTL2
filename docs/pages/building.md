Building, testing and generating this manual {#mftl_page_building}
============================================

[TOC]

# Requirements {#build_requirements}

| | Minimum | Notes |
|---|---|---|
| CMake | 3.13 | `CONFIGURE_DEPENDS` on the source glob needs it |
| C compiler | C11 | GCC, Clang/AppleClang and MSVC are handled in `CMakeLists.txt` |
| MPI | any | found with `find_package(MPI REQUIRED)` |
| Doxygen | 1.9.7+ | documentation only; 1.18 is what the config is written against |
| Graphviz | any | documentation only; without it the graphs are simply omitted |
| Python | 3.9+ | `build_all.py`, `build_docs.py` and `build_mkdocs.py` |
| MkDocs Material | see `requirements-docs.txt` | web manual only; installed into `.venv-docs/` on first use |

---

# Building the library {#build_library}

```bash
python3 build_all.py              # Release (optimized, no sanitizers) — for performance runs
python3 build_all.py --debug      # Debug WITH ASan+UBSan — default for day-to-day dev
python3 build_all.py --valgrind   # Debug WITHOUT sanitizers — required under Valgrind
python3 build_all.py --lib-only   # skip examples/
```

`build_all.py` always deletes and recreates `build/`, so flags never mix between
runs. `src/**/*.c` is globbed with `CONFIGURE_DEPENDS`, so a new `.c` is picked up
without editing `CMakeLists.txt`.

Or configure by hand:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Architecture flags {#build_arch}

`-march=native` is **off by default**. Cluster login nodes routinely differ in
ISA from compute nodes, which turns into `SIGILL` at run time.

```bash
cmake -S . -B build -DMFTL_ARCH_NATIVE=ON              # same-machine run
cmake -S . -B build -DMFTL_ARCH_FLAGS=-march=sapphirerapids
```

## CMake options {#build_options}

| Option | Default | Effect |
|---|---|---|
| `ENABLE_SANITIZERS` | `ON` | ASan + UBSan in Debug. **Public**, so every target linking MFTL is instrumented too — mixing is what produces bogus ASan reports |
| `BUILD_EXAMPLES` | `ON` | the benchmarks under `examples/` |
| `BUILD_TESTING` | `ON` | the CTest suite |
| `MFTL_ARCH_NATIVE` | `OFF` | `-march=native` in Release |
| `MFTL_ARCH_FLAGS` | *(empty)* | extra Release architecture flags |
| `MFTL_BUILD_DOCS` | `OFF` | adds the `docs` target — see @ref build_docs |
| `MFTL_DOCS_WARN_AS_ERROR` | `OFF` | a Doxygen warning fails the `docs` target |

@warning One compile flag in `CMakeLists.txt` is **correctness, not tuning**:
         `src/bvh/bvh_predicates.c` is built with `-ffp-contract=off`. The
         adaptive orientation predicate recovers the exact residue of \f$ab\f$ by
         evaluating \f$x - ab\f$ in a second rounding step; a fused multiply-add
         contracts that into one rounding, the residue comes out zero, and every
         exactness guarantee in the file silently evaporates while the numbers
         still look plausible. Do not remove it.

---

# Running the tests {#build_tests}

```bash
python3 run_local_tests.py            # rebuilds (Release) then runs the full CTest suite
cd build && ctest --output-on-failure
cd build && ctest --output-on-failure -R object    # one module
```

Forty CTest entries, built on `tests/test_utils.h` (`RUN_TEST`, `TEST_ASSERT*`,
`TEST_REPORT`).

@note **Run the suite in the `--debug` build too.** `tests/test_object.c`
      deliberately writes the last addressable element of every pool array, so an
      array that was not grown with the rest is an ASan failure rather than a
      silent corruption.

## Why a test is registered at several rank counts {#build_ranks}

The rank count is not a scaling knob in this suite — it changes which code path
is reachable:

| P | Grid | What only this count sees |
|---|---|---|
| 1 | — | the no-op path, and that every collective is still entered with nothing to exchange |
| 2 | 2×1×1 | periodic wrap maps several of the 26 neighbour offsets onto the same rank, so the neighbour list is shorter than the stencil |
| 4 | 2×2×1 | the first grid with a real diagonal neighbour — the **tube** conflict |
| 8 | 2×2×2 | every one of the 26 offsets is populated; **corner** conflicts; the scatter from rank 0 reaches non-neighbours and forces the global path |
| 27 | 3×3×3 | the first grid with an **interior** rank whose 26 neighbours are 26 *distinct* ranks. Below it, periodic aliasing collapses several stencil offsets onto the same rank and the de-duplication is doing half the work by accident |

@warning In a multi-rank test **every assertion must be made on an
         `MPI_Allreduce`d verdict** (see `agreed()` in `tests/test_utils.h`).
         `TEST_ASSERT` returns immediately, so an assertion failing on only some
         ranks leaves the rest blocked in the next collective — a hang instead of
         a clean failure.

## The source-level test {#build_source_test}

`test_remeshing_mpi_is_confined` is a CMake script, not a binary. It fails the
build if any file under `src/remeshing/` other than `remesh_comm.c` contains an
MPI call. No runtime test can see that property, and it is exactly the one that
was lost the first time round.

---

# Generating this manual {#build_docs}

Two routes, producing identical output. Both substitute the same placeholders
into `docs/doxygen/Doxyfile.in`.

## Standalone — nothing to configure {#build_docs_standalone}

```bash
python3 build_docs.py                 # -> build/docs/html/index.html
python3 build_docs.py --open          # and open it in a browser
python3 build_docs.py --strict        # any Doxygen warning is a failure
python3 build_docs.py --no-graphs     # skip Graphviz (much faster)
python3 build_docs.py --output-dir /tmp/mftl-docs
```

This route does **not** invoke CMake and does not touch `build/`'s CMake cache,
so it stays out of the way of a fast edit–compile–test cycle.

## Through CMake — off by default {#build_docs_cmake}

```bash
cmake -S . -B build -DMFTL_BUILD_DOCS=ON
cmake --build build --target docs
```

@note The option defaults to **OFF** deliberately. Doxygen with call graphs
      enabled takes far longer than the library compiles, and nothing in the
      normal build depends on it. With the option OFF, `CMakeLists.txt` does not
      even call `find_package(Doxygen)`, so configuration time is unchanged.

      With it ON, `docs` is still an opt-in target: it is **not** attached to
      `all`, so `cmake --build build` alone never runs Doxygen.

## What the generated manual contains {#build_docs_contents}

| Section | Source |
|---|---|
| Main page | `docs/pages/mainpage.md` |
| Theory & Guides | `docs/pages/*.md` plus the three analysis documents in `docs/` |
| Modules | `docs/doxygen/groups.dox` + `@ingroup` tags, one group per `src/` directory |
| Module theory pages | `src/*/README.md`, bound in as full pages |
| API reference | Doxygen comments in the public headers and `src/**/*.c` |
| Call / caller graphs | Graphviz, as interactive SVG |
| `#include` graphs | Graphviz, per file and per directory |

Formulas are written as `\\f$ ... \\f$` / `\\f[ ... \\f]` in the sources and as
`$ ... $` / `$$ ... $$` in the Markdown pages; `docs/doxygen/mathjax.js` enables
both delimiter sets, so a README renders correctly on GitHub **and** in the
manual from one source. Shared macros live in `docs/doxygen/formulas.tex`.

## Warnings are the acceptance criterion {#build_docs_warnings}

The configuration turns on `WARN_IF_UNDOCUMENTED`, `WARN_IF_INCOMPLETE_DOC`,
`WARN_NO_PARAMDOC` and `WARN_IF_UNDOC_ENUM_VAL`, and writes them to
`build/docs/doxygen-warnings.log`. A clean run produces an empty log.

```bash
python3 build_docs.py --strict        # exits non-zero if the log is non-empty
```

---

# The web manual — MkDocs Material {#build_site}

The same corpus is also published as a modern static site: MkDocs with the
Material theme for the prose, [mkdoxy](https://mkdoxy.kubaandrysek.cz) for the
API reference, and the **whole Doxygen manual mounted inside it** at
`/doxygen/`, so a reader gets the call graphs and the annotated sources without
building or deploying a second thing.

```bash
python3 build_mkdocs.py                 # -> build/site/index.html
python3 build_mkdocs.py --serve --open  # live-reload dev server on :8000
python3 build_mkdocs.py --strict        # any warning, Doxygen or MkDocs, is a failure
python3 build_mkdocs.py --no-doxygen --no-graphs --serve   # fastest prose loop
```

The first run creates `.venv-docs/` and installs `requirements-docs.txt` into
it. That environment is private to the documentation for the same reason
`build_all.py` recreates `build/` every time: a documentation dependency must
never be able to change what the library build does. Nothing in
`requirements-docs.txt` is needed to compile, test or use MFTL.

## There is still one source per subject {#build_site_sources}

No prose is copied. `docs/mkdocs/gen_pages.py` reads the files Doxygen reads and
rewrites Doxygen's Markdown dialect into Material's at build time:

| Site section | Generated from |
|---|---|
| Overview | `docs/pages/mainpage.md` |
| Guide | `docs/pages/architecture.md`, `building.md`, `timestep.md`, `invariants.md` |
| Modules | `docs/pages/modules.md` and the ten `src/*/README.md` |
| Analysis | the three `docs/*-analysis.md` |
| API reference | Doxygen XML, via mkdoxy |
| `/doxygen/` | `build/docs/html`, copied in by `docs/mkdocs/hooks.py` |

`@ref`, `@subpage`, `#MFTL_Symbol` and `MFTLFunction()` are resolved against
Doxygen's own `index.xml`, so a cross-reference in the prose lands on the exact
member anchor of the generated API page rather than on a guessed path. The
Doxyfile mkdoxy runs is itself derived from `docs/doxygen/Doxyfile.in` by
`docs/mkdocs/doxyfile.py` — the aliases, the `PREDEFINED` set and the extraction
policy are read from one template, so the two manuals cannot document
differently configured libraries.

@note The plugin order in `mkdocs.yml` is load-bearing: **mkdoxy must run before
      `gen-files`**. mkdoxy produces the Doxygen XML that the page generator
      resolves cross-references against, and MkDocs fires plugin events in the
      order the plugins are declared. Swap the two and every API link in the
      prose silently degrades to a plain code span on the first build of a clean
      tree — the pages still build, and nothing reports it.

## Which manual answers which question {#build_site_split}

The two are generated from the same sources but they are not interchangeable,
and the split is by what the tool can produce rather than by taste:

| Question | Where |
|---|---|
| What does this function do | either — identical content |
| What is the theory behind this module | the site: the module pages, with a navigable structure and a search index over the prose |
| Who calls this, and what does it call | the Doxygen manual — Graphviz call and caller graphs |
| What does the annotated source look like | the Doxygen manual — cross-linked source browser |
| Which headers include which | the Doxygen manual — `#include` dependency graphs |

@warning `build/site/` is a **static** directory and nothing in it is generated
         at view time, so it can be served from anywhere. It does, however,
         contain the whole Doxygen manual, which is ~65 MB of SVG. Deploying it
         to a path other than the site root requires `site_url` in `mkdocs.yml`
         to be set — the links between the two halves are relative, but
         Material's search index and social assets are not.

---

# Header hygiene {#build_hygiene}

Two checks worth running after touching headers:

```bash
# every header must compile alone
for h in include/*.h; do printf '#include "%s"\n' "$(basename $h)" > /tmp/sc.c; \
  mpicc -std=c11 -fsyntax-only -Iinclude /tmp/sc.c || echo "NOT SELF-CONTAINED: $h"; done

# the library must export only MFTL* symbols, with no duplicates
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | grep -v _MFTL   # must be empty
nm -g build/lib/libMFTL.a | grep ' T ' | awk '{print $NF}' | sort | uniq -d  # must be empty
```
