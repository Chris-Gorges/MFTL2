# Contributing to MFTL

MFTL is under active development. Issues and pull requests are welcome; for anything structural, open an issue
first, because most of the file layout follows from a few invariants (see "The idea" in the README and the
`src/<module>/README.md` theory chapters) and a change that ignores them tends to look local and not be.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug     # Debug enables ASan + UBSan by default
cmake --build build -j
ctest --test-dir build --output-on-failure -LE np27
```

`python3 build_all.py [--debug | --valgrind | --lib-only]` wraps the same steps with a clean build directory.

**Run the suite in Debug too**, not only in Release. Several tests deliberately write the last addressable element
of every pool array, so an array that was not grown with the rest is an AddressSanitizer failure rather than silent
corruption.

Tests carry CTest labels: `serial`, `mpi`, `np<N>` (rank count), `example` (a physical run of an example
program), `np27` (oversubscribed on a laptop) and `long`. `ctest -L <label>` / `-LE <label>` select them; every
MPI test declares its rank count as `PROCESSORS`, so `ctest -j N` schedules ranks rather than processes.

## Code conventions

- C11, no compiler extensions. Lines you add or change follow `.clang-format`; CI checks exactly those lines with
  a pinned clang-format (`pip install -r tools/requirements-format.txt`, then `tools/check_format.sh` before a
  commit, or `tools/check_format.sh --fix` to reformat those lines). Existing code is not reformatted wholesale.
- One module = one `src/<module>/` directory = one `include/mftl_<module>.h`. Every `.c` includes its own header
  first, and a header includes only headers from a strictly lower layer. `tools/check_headers.sh` verifies that
  every header compiles alone, as C and as C++.
- Exported symbols are `MFTL*` only (`tools/check_symbols.sh`). File-static helpers are `snake_case`.
- Functions return `MFTL_SUCCESS` / `MFTL_ERROR`. `MFTL_ASSERT` vanishes in Release, so never put side effects in it.
- Pool sizes change only through `mftl_object.h`; `src/object/object.c` is the only file that writes a pool pointer
  or a capacity.
- A module's MPI calls live in that module's `*_comm.c` file. Tests enforce this for remeshing, topology change,
  coupling and correction.

## Writing tests

- Use `tests/test_utils.h` and check meshes with the invariants in `tests/test_mesh_invariants.h`
  (`inv_mesh_is_sound()`, and `inv_partition_is_sound()` after anything distributed).
- In a multi-rank test, assert on a verdict reduced across ranks with `agreed()`. An assertion that fails on only
  some ranks returns early on those ranks and leaves the others blocked in the next collective.
- A distributed test needs a non-vacuity measure: show that the case actually crosses a rank boundary, otherwise it
  can pass forever without covering anything.
- Compare a distributed result against the same call on one rank (`MPI_COMM_SELF`) wherever the answer is meant
  to be partition-independent.

## Documentation

Public functions carry Doxygen comments with `@param` and `@return`; the manual is built with zero warnings, and an
undocumented parameter is a build failure there.
