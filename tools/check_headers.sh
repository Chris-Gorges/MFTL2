#!/usr/bin/env bash
# Every public header must compile on its own, as C11 and as C++.
#
# On its own, because "every .c includes its own header first" only makes a
# header self-contained if nothing else is included before it anywhere; this
# checks it directly. As C++, because the library's clients are mostly C++ flow
# solvers: a C-only construct outside a macro (a bare _Static_assert, `restrict`
# in a prototype, an implicit void* conversion in an inline function) breaks
# their build, and nothing in this C11 code base would notice.
#
# Usage: tools/check_headers.sh
#        MFTL_CC / MFTL_CXX override the compilers (default: the MPI wrappers,
#        which supply mpi.h). Not CC / CXX: a build environment usually sets
#        those to the bare compiler.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
CC="${MFTL_CC:-mpicc}"
CXX="${MFTL_CXX:-mpicxx}"
# The wrappers append their link flags even to a syntax-only run, which clang
# reports as unused arguments; GCC ignores the unknown -Wno- switch.
quiet=-Wno-unused-command-line-argument
# What MFTL::MFTL hands every C++ consumer (CMakeLists.txt): without it Open MPI
# 4 and MPICH declare the removed MPI C++ bindings, whose own headers do not
# compile cleanly under -Wextra -Werror with GCC.
cxx_defs="-DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

failed=0
for h in "$root"/include/*.h; do
  name="$(basename "$h")"
  printf '#include "%s"\n' "$name" > "$tmp/c.c"
  printf '#include "%s"\n' "$name" > "$tmp/cxx.cpp"
  if ! "$CC" -std=c11 -Wall -Wextra -Werror -fsyntax-only $quiet -I"$root/include" "$tmp/c.c"; then
    echo "NOT SELF-CONTAINED AS C11: $name"
    failed=1
  fi
  if ! "$CXX" -std=c++17 -Wall -Wextra -Werror -fsyntax-only $quiet $cxx_defs -I"$root/include" "$tmp/cxx.cpp"; then
    echo "DOES NOT COMPILE AS C++17: $name"
    failed=1
  fi
done

if [ "$failed" -eq 0 ]; then
  echo "all $(ls "$root"/include/*.h | wc -l | tr -d ' ') headers compile alone as C11 and as C++17"
fi
exit "$failed"
