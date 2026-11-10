#!/usr/bin/env bash
# Finds, on macOS, the UBSan errors that only glibc's headers would reveal.
#
# glibc declares memcpy, qsort and friends nonnull, so a zero-length copy from a
# NULL array is a UBSan error on Linux and silent on a Mac (CLAUDE.md, "two
# portability traps"). This builds Debug with ASan+UBSan and tools/glibc_nonnull.h
# force-included into every C file, runs CTest, and lists every UBSan report.
# UBSan does not halt, so one run finds all of them, not the first per rank.
#
# Usage: tools/check_glibc_nonnull.sh [BUILD_DIR] [CTEST_ARGS...]
#        BUILD_DIR   default build-nonnull (kept, so a second run only rebuilds)
#        CTEST_ARGS  e.g. -LE "np27|long" for the push-sized suite, -R coupling
#
# The whole suite takes about 15 minutes on an M4 (np27 included). Keep the Mac
# awake and online: Open MPI starts no multi-rank job without a network
# interface, and every MPI test then fails with "no available interfaces".
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$root/build-nonnull}"
[ $# -gt 0 ] && shift

cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON \
  "-DCMAKE_C_FLAGS=-include $root/tools/glibc_nonnull.h" > "$build.configure.log" 2>&1 ||
  { cat "$build.configure.log" >&2; exit 2; }
cmake --build "$build" -j "$(getconf _NPROCESSORS_ONLN)" > "$build.build.log" 2>&1 ||
  { tail -n 40 "$build.build.log" >&2; exit 2; }

logs="$build/ubsan"
rm -rf "$logs" && mkdir -p "$logs"
status=0
(cd "$build" && UBSAN_OPTIONS="halt_on_error=0:print_stacktrace=1:log_path=$logs/ubsan" ASAN_OPTIONS=detect_leaks=0 \
  ctest -j "$(getconf _NPROCESSORS_ONLN)" --timeout 1800 --output-on-failure "$@") || status=$?

reports="$(cat "$logs"/ubsan.* 2> /dev/null | grep "runtime error" | sed "s|^$root/||" | sort | uniq -c | sort -rn || true)"
if [ -n "$reports" ]; then
  echo
  echo "UBSan reports (count, site) -- stacks in $logs:"
  echo "$reports"
  exit 1
fi
echo
echo "no UBSan reports"
exit "$status"
